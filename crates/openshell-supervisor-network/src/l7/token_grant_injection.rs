// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint-bound runtime credential injection for HTTP relay paths: dynamic
//! token grants and proxy-delivered static credentials.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use miette::{Result, miette};
use openshell_core::proto::{ProviderCredentialTokenGrant, ProviderProfileCredential};
use openshell_core::provider_credentials::ProxyDeliveredCredential;
use openshell_ocsf::{
    ActionId, ActivityId, DispositionId, Endpoint, HttpActivityBuilder, HttpRequest, SeverityId,
    StatusId, Url as OcsfUrl, ctx::ctx as ocsf_ctx, ocsf_emit,
};
use tracing::warn;

use crate::l7::provider::L7Request;
use crate::l7::relay::L7EvalContext;

pub struct TokenGrantRequest<'a> {
    pub provider_key: &'a str,
    pub token_endpoint: &'a str,
    pub jwt_svid_audience: &'a str,
    pub client_assertion_type: &'a str,
    pub audience: &'a str,
    pub scopes: &'a [String],
    pub cache_ttl_seconds: i64,
    pub grant_type: i32,
    pub requested_token_type: &'a str,
}

pub trait TokenGrantResolver: Send + Sync {
    fn obtain<'a>(
        &'a self,
        request: TokenGrantRequest<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
}

#[derive(Default)]
pub struct SpiffeTokenGrantResolver;

impl TokenGrantResolver for SpiffeTokenGrantResolver {
    fn obtain<'a>(
        &'a self,
        request: TokenGrantRequest<'a>,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            crate::token_grant::obtain_provider_token(
                crate::token_grant::ObtainProviderTokenRequest {
                    provider_name: request.provider_key,
                    token_endpoint: request.token_endpoint,
                    jwt_svid_audience: request.jwt_svid_audience,
                    client_assertion_type: request.client_assertion_type,
                    audience: request.audience,
                    scopes: request.scopes,
                    cache_ttl_override: request.cache_ttl_seconds,
                    grant_type: request.grant_type,
                    requested_token_type: request.requested_token_type,
                },
            )
            .await
        })
    }
}

pub fn default_resolver() -> Arc<dyn TokenGrantResolver> {
    Arc::new(SpiffeTokenGrantResolver)
}

/// Checks for endpoint-bound token grant credentials and injects an
/// Authorization header before forwarding the request upstream.
pub async fn inject_if_needed(req: L7Request, ctx: &L7EvalContext) -> Result<L7Request> {
    let request_path = req.target.split('?').next().unwrap_or(req.target.as_str());
    let token_grant_credential = match ctx.dynamic_credentials.as_ref() {
        None => None,
        Some(dyn_creds) => {
            // A poisoned registry must fail closed. Treating it as "no
            // credential" would forward the request unauthenticated.
            let creds_guard = dyn_creds
                .read()
                .map_err(|_| miette!("dynamic credential registry is poisoned"))?;
            creds_guard
                .iter()
                .filter_map(|(key, cred)| {
                    let score =
                        dynamic_credential_key_match_score(key, &ctx.host, ctx.port, request_path)?;
                    cred.token_grant
                        .is_some()
                        .then(|| (score, key.clone(), cred.clone()))
                })
                .max_by_key(|(score, key, _)| (*score, key.clone()))
                .map(|(_, key, cred)| (key, cred))
        }
    };

    if let Some((provider_key, cred)) = token_grant_credential
        && let Some(ref token_grant) = cred.token_grant
    {
        let resolver = ctx
            .token_grant_resolver
            .as_ref()
            .ok_or_else(|| miette!("token grant resolver unavailable"))?;
        let request = token_grant_request(&provider_key, token_grant);

        match resolver.obtain(request).await {
            Ok(access_token) => {
                let modified_raw_header =
                    inject_token_grant_header(&req.raw_header, &cred, &access_token)?;
                let provider_key = ocsf_message_field(&provider_key);
                ocsf_emit!(
                    HttpActivityBuilder::new(ocsf_ctx())
                        .activity(ActivityId::Other)
                        .action(ActionId::Allowed)
                        .disposition(DispositionId::Allowed)
                        .severity(SeverityId::Informational)
                        .http_request(HttpRequest::new(
                            &req.action,
                            OcsfUrl::new("http", &ctx.host, request_path, ctx.port),
                        ))
                        .dst_endpoint(Endpoint::from_domain(&ctx.host, ctx.port))
                        .message(format!(
                            "Token grant successful for {} to {}:{}",
                            provider_key, ctx.host, ctx.port
                        ))
                        .build()
                );
                return Ok(L7Request {
                    action: req.action,
                    target: req.target,
                    query_params: req.query_params,
                    raw_header: modified_raw_header,
                    body_length: req.body_length,
                });
            }
            Err(e) => {
                warn!(
                    host = %ctx.host,
                    port = ctx.port,
                    provider = %provider_key,
                    error = %e,
                    "Token grant failed: {e}"
                );
                let provider_key = ocsf_message_field(&provider_key);
                ocsf_emit!(
                    HttpActivityBuilder::new(ocsf_ctx())
                        .activity(ActivityId::Fail)
                        .action(ActionId::Denied)
                        .disposition(DispositionId::Blocked)
                        .severity(SeverityId::Medium)
                        .status(StatusId::Failure)
                        .http_request(HttpRequest::new(
                            &req.action,
                            OcsfUrl::new("http", &ctx.host, request_path, ctx.port),
                        ))
                        .dst_endpoint(Endpoint::from_domain(&ctx.host, ctx.port))
                        .message(format!(
                            "Token grant failed for {} to {}:{}: {}",
                            provider_key, ctx.host, ctx.port, e
                        ))
                        .build()
                );
                return Err(miette!("Token grant failed: {}", e));
            }
        }
    }

    Ok(req)
}

/// Injects an endpoint-bound static credential that opted into proxy delivery.
///
/// The static credential bindings in `ctx.provider_credentials` are the single
/// source of truth: a binding whose endpoint authorizes this request and whose
/// delivery mode is `proxy` is resolved through the request-scoped
/// `ctx.secret_resolver` and written as the complete outbound header. Requests
/// with no matching proxy-delivered binding pass through unchanged.
///
/// Every outcome that touches a credential is recorded as an OCSF HTTP
/// activity event, without the credential value.
pub fn inject_static_if_needed(req: L7Request, ctx: &L7EvalContext) -> Result<L7Request> {
    let request_path = req.target.split('?').next().unwrap_or(req.target.as_str());
    let Some(state) = ctx.provider_credentials.as_ref() else {
        return Ok(req);
    };
    let bindings =
        state.proxy_delivered_credentials_for_endpoint(&ctx.host, ctx.port, request_path);
    if bindings.is_empty() {
        return Ok(req);
    }
    let env_keys = bindings
        .iter()
        .map(|binding| ocsf_message_field(&binding.env_key))
        .collect::<Vec<_>>()
        .join(",");

    match inject_proxy_delivered_header(&req.raw_header, ctx, &bindings) {
        Ok((header_name, raw_header)) => {
            ocsf_emit!(
                HttpActivityBuilder::new(ocsf_ctx())
                    .activity(ActivityId::Other)
                    .action(ActionId::Allowed)
                    .disposition(DispositionId::Allowed)
                    .severity(SeverityId::Informational)
                    .http_request(HttpRequest::new(
                        &req.action,
                        OcsfUrl::new("http", &ctx.host, request_path, ctx.port),
                    ))
                    .dst_endpoint(Endpoint::from_domain(&ctx.host, ctx.port))
                    .message(format!(
                        "Proxy-delivered credential {} injected as {} to {}:{}",
                        env_keys,
                        ocsf_message_field(&header_name),
                        ctx.host,
                        ctx.port
                    ))
                    .build()
            );
            Ok(L7Request {
                action: req.action,
                target: req.target,
                query_params: req.query_params,
                raw_header,
                body_length: req.body_length,
            })
        }
        Err(error) => {
            warn!(
                host = %ctx.host,
                port = ctx.port,
                env_keys = %env_keys,
                error = %error,
                "Proxy-delivered credential injection failed"
            );
            ocsf_emit!(
                HttpActivityBuilder::new(ocsf_ctx())
                    .activity(ActivityId::Fail)
                    .action(ActionId::Denied)
                    .disposition(DispositionId::Blocked)
                    .severity(SeverityId::Medium)
                    .status(StatusId::Failure)
                    .http_request(HttpRequest::new(
                        &req.action,
                        OcsfUrl::new("http", &ctx.host, request_path, ctx.port),
                    ))
                    .dst_endpoint(Endpoint::from_domain(&ctx.host, ctx.port))
                    .message(format!(
                        "Proxy-delivered credential {} injection failed for {}:{}: {}",
                        env_keys, ctx.host, ctx.port, error
                    ))
                    .build()
            );
            Err(error)
        }
    }
}

/// Resolve every matching proxy-delivered binding and build the header.
///
/// Aliases of one credential resolve to the same header and collapse into a
/// single injection. Distinct headers mean two providers bound the same
/// endpoint, which the gateway rejects at attach time; the proxy fails closed
/// rather than choosing one. Returns the header name and the rewritten request.
fn inject_proxy_delivered_header(
    raw_header: &[u8],
    ctx: &L7EvalContext,
    bindings: &[ProxyDeliveredCredential],
) -> Result<(String, Vec<u8>)> {
    let resolver = ctx
        .secret_resolver
        .as_deref()
        .ok_or_else(|| miette!("proxy-delivered credential resolver unavailable"))?;
    let mut header: Option<(String, String)> = None;
    for binding in bindings {
        let value = resolver
            .resolve_current_env_key_checked(&binding.env_key, "proxy-delivered credential")
            .map_err(|error| miette!("proxy-delivered credential {}: {error}", binding.env_key))?
            .ok_or_else(|| {
                miette!(
                    "proxy-delivered credential {} is unavailable in the current provider revision",
                    binding.env_key
                )
            })?;
        let candidate = injected_credential_header(
            &binding.auth_style,
            &binding.header_name,
            value,
            "proxy delivery",
        )?;
        match &header {
            None => header = Some(candidate),
            Some(existing) if *existing == candidate => {}
            Some(_) => {
                return Err(miette!(
                    "multiple proxy-delivered credentials match this endpoint; attach only one matching provider"
                ));
            }
        }
    }
    let (header_name, header_value) =
        header.ok_or_else(|| miette!("no proxy-delivered credential binding matched"))?;
    let raw_header = inject_header(raw_header, &header_name, &header_value)?;
    Ok((header_name, raw_header))
}

fn ocsf_message_field(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch.is_control() { '_' } else { ch })
        .collect()
}

fn token_grant_request<'a>(
    provider_key: &'a str,
    token_grant: &'a ProviderCredentialTokenGrant,
) -> TokenGrantRequest<'a> {
    TokenGrantRequest {
        provider_key,
        token_endpoint: &token_grant.token_endpoint,
        jwt_svid_audience: &token_grant.jwt_svid_audience,
        client_assertion_type: &token_grant.client_assertion_type,
        audience: &token_grant.audience,
        scopes: &token_grant.scopes,
        cache_ttl_seconds: token_grant.cache_ttl_seconds,
        grant_type: token_grant.grant_type,
        requested_token_type: &token_grant.requested_token_type,
    }
}

#[cfg(test)]
fn dynamic_credential_key_matches(key: &str, host: &str, port: u16, request_path: &str) -> bool {
    dynamic_credential_key_match_score(key, host, port, request_path).is_some()
}

fn dynamic_credential_key_match_score(
    key: &str,
    host: &str,
    port: u16,
    request_path: &str,
) -> Option<u32> {
    let mut parts = key.splitn(4, '\t');
    let endpoint_host = parts.next()?;
    let endpoint_port = parts.next()?;
    let endpoint_path = parts.next()?;
    let _provider_key = parts.next()?;

    if endpoint_port.parse::<u16>().ok() != Some(port) {
        return None;
    }

    if !openshell_core::host_pattern::host_matches(endpoint_host, host).unwrap_or(false)
        || !crate::l7::endpoint_path_matches(endpoint_path, request_path)
    {
        return None;
    }

    Some(
        host_pattern_specificity(&endpoint_host.to_ascii_lowercase())
            + endpoint_path_specificity(endpoint_path),
    )
}

fn host_pattern_specificity(pattern: &str) -> u32 {
    let wildcard_penalty = count_as_u32(pattern.matches('*').count());
    let label_count = count_as_u32(pattern.split('.').filter(|label| !label.is_empty()).count());
    let literal_chars = count_as_u32(pattern.chars().filter(|ch| *ch != '*').count());
    100_000u32
        .saturating_sub(wildcard_penalty.saturating_mul(10_000))
        .saturating_add(label_count.saturating_mul(100))
        .saturating_add(literal_chars)
}

fn endpoint_path_specificity(path: &str) -> u32 {
    if path.is_empty() || path == "**" {
        return 0;
    }
    1_000_000u32.saturating_add(count_as_u32(path.chars().filter(|ch| *ch != '*').count()))
}

fn count_as_u32(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
}

fn inject_token_grant_header(
    raw_header: &[u8],
    credential: &ProviderProfileCredential,
    access_token: &str,
) -> Result<Vec<u8>> {
    let (header_name, header_value) = injected_credential_header(
        &credential.auth_style,
        &credential.header_name,
        access_token,
        "token grant",
    )?;
    inject_header(raw_header, &header_name, &header_value)
}

/// Build the outbound header for a runtime-injected credential.
///
/// `context` names the caller ("token grant" or "proxy delivery") in error
/// messages. Values are validated against the placement so a malformed
/// credential can never produce a malformed or header-injecting request.
fn injected_credential_header(
    auth_style: &str,
    configured_header_name: &str,
    value: &str,
    context: &str,
) -> Result<(String, String)> {
    match auth_style.trim().to_ascii_lowercase().as_str() {
        "" | "bearer" => {
            validate_bearer_value(value, context)?;
            let header_name = if configured_header_name.trim().is_empty() {
                "Authorization"
            } else {
                configured_header_name.trim()
            };
            validate_header_name(header_name, context)?;
            Ok((header_name.to_string(), format!("Bearer {value}")))
        }
        "header" => {
            let header_name = configured_header_name.trim();
            if header_name.is_empty() {
                return Err(miette!("{context} auth_style header requires header_name"));
            }
            validate_header_name(header_name, context)?;
            validate_header_value(value, context)?;
            Ok((header_name.to_string(), value.to_string()))
        }
        other => Err(miette!(
            "{context} auth_style '{other}' is not supported; use bearer or header"
        )),
    }
}

fn validate_bearer_value(value: &str, context: &str) -> Result<()> {
    if context == "token grant" {
        // Preserve the historical token grant wording.
        return crate::token_grant::validate_access_token(value);
    }
    if !openshell_core::provider_credentials::is_token68(value) {
        return Err(miette!(
            "{context} bearer credential is not a valid token68 value; check the stored provider credential"
        ));
    }
    Ok(())
}

fn validate_header_value(value: &str, context: &str) -> Result<()> {
    if value
        .bytes()
        .any(|byte| (byte < b' ' && byte != b'\t') || byte == 0x7f)
    {
        return Err(miette!(
            "{context} credential contains invalid HTTP header value characters"
        ));
    }
    Ok(())
}

fn validate_header_name(header_name: &str, context: &str) -> Result<()> {
    let valid = !header_name.is_empty()
        && header_name.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        });
    if !valid {
        return Err(miette!(
            "{context} header_name is not a valid HTTP header name"
        ));
    }
    match header_name.to_ascii_lowercase().as_str() {
        "host" | "content-length" | "transfer-encoding" | "connection" => Err(miette!(
            "{context} header_name may not override HTTP framing or connection headers"
        )),
        _ => Ok(()),
    }
}

fn inject_header(raw_header: &[u8], header_name: &str, header_value: &str) -> Result<Vec<u8>> {
    let header_end = raw_header
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| miette!("HTTP headers missing final CRLF CRLF"))?;

    let header_block = std::str::from_utf8(&raw_header[..header_end])
        .map_err(|_| miette!("HTTP headers contain invalid UTF-8"))?;
    let mut lines = header_block.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| miette!("HTTP headers missing request line"))?;

    let inserted_header = format!("{header_name}: {header_value}");
    let mut new_raw_header = Vec::with_capacity(raw_header.len() + inserted_header.len() + 2);
    new_raw_header.extend_from_slice(request_line.as_bytes());
    new_raw_header.extend_from_slice(b"\r\n");

    for line in lines {
        if line.is_empty() {
            break;
        }
        if line
            .split_once(':')
            .is_some_and(|(name, _)| name.trim().eq_ignore_ascii_case(header_name))
        {
            continue;
        }
        new_raw_header.extend_from_slice(line.as_bytes());
        new_raw_header.extend_from_slice(b"\r\n");
    }

    new_raw_header.extend_from_slice(inserted_header.as_bytes());
    new_raw_header.extend_from_slice(&raw_header[header_end..]);

    Ok(new_raw_header)
}

#[cfg(test)]
pub mod test_support {
    use super::*;
    use openshell_core::proto::{
        ProviderCredentialTokenGrant, ProviderCredentialTokenGrantSubjectToken,
        ProviderCredentialTokenGrantType, ProviderProfileCredential,
    };
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    struct FakeTokenGrantResolver {
        requests: Arc<Mutex<Vec<OwnedTokenGrantRequest>>>,
        response: std::result::Result<String, String>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct OwnedTokenGrantRequest {
        provider_key: String,
        token_endpoint: String,
        jwt_svid_audience: String,
        client_assertion_type: String,
        audience: String,
        scopes: Vec<String>,
        cache_ttl_seconds: i64,
        grant_type: i32,
        requested_token_type: String,
    }

    pub struct TokenGrantTestFixture {
        dynamic_credentials: Arc<std::sync::RwLock<HashMap<String, ProviderProfileCredential>>>,
        resolver: Arc<dyn TokenGrantResolver>,
        requests: Arc<Mutex<Vec<OwnedTokenGrantRequest>>>,
    }

    impl TokenGrantTestFixture {
        pub fn success(key: &str, token: &str) -> Self {
            Self::new(key, Ok(token))
        }

        pub fn success_token_exchange(key: &str, token: &str) -> Self {
            Self::new_with_grant(key, Ok(token), token_exchange_grant())
        }

        pub fn failure(key: &str, error: &str) -> Self {
            Self::new(key, Err(error))
        }

        pub fn failure_token_exchange(key: &str, error: &str) -> Self {
            Self::new_with_grant(key, Err(error), token_exchange_grant())
        }

        fn new(key: &str, response: std::result::Result<&str, &str>) -> Self {
            Self::new_with_grant(key, response, token_grant())
        }

        fn new_with_grant(
            key: &str,
            response: std::result::Result<&str, &str>,
            token_grant: ProviderCredentialTokenGrant,
        ) -> Self {
            let requests = Arc::new(Mutex::new(Vec::new()));
            let resolver = Arc::new(FakeTokenGrantResolver {
                requests: requests.clone(),
                response: response.map(str::to_string).map_err(str::to_string),
            });

            let mut dynamic_credentials = HashMap::new();
            dynamic_credentials.insert(
                key.to_string(),
                ProviderProfileCredential {
                    name: "access_token".to_string(),
                    auth_style: "bearer".to_string(),
                    header_name: "Authorization".to_string(),
                    token_grant: Some(token_grant),
                    ..Default::default()
                },
            );

            Self {
                dynamic_credentials: Arc::new(std::sync::RwLock::new(dynamic_credentials)),
                resolver,
                requests,
            }
        }

        pub fn dynamic_credentials(
            &self,
        ) -> Arc<std::sync::RwLock<HashMap<String, ProviderProfileCredential>>> {
            self.dynamic_credentials.clone()
        }

        pub fn resolver(&self) -> Arc<dyn TokenGrantResolver> {
            self.resolver.clone()
        }

        pub fn assert_no_requests(&self) {
            let requests = self
                .requests
                .lock()
                .expect("fake token grant requests lock poisoned");
            assert!(requests.is_empty(), "unexpected token grant requests");
        }

        pub fn assert_one_request(&self, expected_provider_key: &str) {
            let requests = self
                .requests
                .lock()
                .expect("fake token grant requests lock poisoned");
            assert_eq!(requests.len(), 1);

            let request = &requests[0];
            assert_eq!(request.provider_key, expected_provider_key);
            assert_eq!(request.token_endpoint, "https://auth.example.com/token");
            assert_eq!(request.jwt_svid_audience, "https://auth.example.com");
            assert_eq!(
                request.client_assertion_type,
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
            );
            assert_eq!(request.audience, "api://example");
            assert_eq!(request.scopes, ["read"]);
            assert_eq!(request.cache_ttl_seconds, 300);
            assert_eq!(
                request.grant_type,
                ProviderCredentialTokenGrantType::ClientCredentials as i32
            );
            assert!(request.requested_token_type.is_empty());
        }

        pub fn assert_one_token_exchange_request(&self, expected_provider_key: &str) {
            let requests = self
                .requests
                .lock()
                .expect("fake token grant requests lock poisoned");
            assert_eq!(requests.len(), 1);

            let request = &requests[0];
            assert_eq!(request.provider_key, expected_provider_key);
            assert_eq!(request.token_endpoint, "https://auth.example.com/token");
            assert_eq!(request.jwt_svid_audience, "https://auth.example.com");
            assert_eq!(
                request.client_assertion_type,
                "urn:ietf:params:oauth:client-assertion-type:jwt-spiffe"
            );
            assert_eq!(request.audience, "api://example");
            assert_eq!(request.scopes, ["read"]);
            assert_eq!(request.cache_ttl_seconds, 300);
            assert_eq!(
                request.grant_type,
                ProviderCredentialTokenGrantType::TokenExchange as i32
            );
            assert_eq!(
                request.requested_token_type,
                "urn:ietf:params:oauth:token-type:access_token"
            );
        }
    }

    fn token_grant() -> ProviderCredentialTokenGrant {
        ProviderCredentialTokenGrant {
            token_endpoint: "https://auth.example.com/token".to_string(),
            audience: "api://example".to_string(),
            jwt_svid_audience: "https://auth.example.com".to_string(),
            client_assertion_type: "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
                .to_string(),
            scopes: vec!["read".to_string()],
            cache_ttl_seconds: 300,
            audience_overrides: Vec::new(),
            grant_type: ProviderCredentialTokenGrantType::ClientCredentials as i32,
            subject_token: None,
            requested_token_type: String::new(),
        }
    }

    fn token_exchange_grant() -> ProviderCredentialTokenGrant {
        ProviderCredentialTokenGrant {
            client_assertion_type: "urn:ietf:params:oauth:client-assertion-type:jwt-spiffe"
                .to_string(),
            grant_type: ProviderCredentialTokenGrantType::TokenExchange as i32,
            subject_token: Some(ProviderCredentialTokenGrantSubjectToken {
                source: "provider_credential".to_string(),
                credential: "user_oidc_token".to_string(),
                subject_token_type: "urn:ietf:params:oauth:token-type:id_token".to_string(),
            }),
            requested_token_type: "urn:ietf:params:oauth:token-type:access_token".to_string(),
            ..token_grant()
        }
    }

    impl TokenGrantResolver for FakeTokenGrantResolver {
        fn obtain<'a>(
            &'a self,
            request: TokenGrantRequest<'a>,
        ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
            let owned = OwnedTokenGrantRequest {
                provider_key: request.provider_key.to_string(),
                token_endpoint: request.token_endpoint.to_string(),
                jwt_svid_audience: request.jwt_svid_audience.to_string(),
                client_assertion_type: request.client_assertion_type.to_string(),
                audience: request.audience.to_string(),
                scopes: request.scopes.to_vec(),
                cache_ttl_seconds: request.cache_ttl_seconds,
                grant_type: request.grant_type,
                requested_token_type: request.requested_token_type.to_string(),
            };
            Box::pin(async move {
                self.requests
                    .lock()
                    .expect("fake token grant requests lock poisoned")
                    .push(owned);
                self.response.clone().map_err(|err| miette!("{err}"))
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l7::provider::{BodyLength, L7Request};
    use crate::l7::token_grant_injection::test_support::TokenGrantTestFixture;
    use openshell_core::proto::{
        ProviderCredentialDelivery, StaticCredentialBinding, StaticCredentialEndpointBinding,
    };
    use openshell_core::provider_credentials::ProviderCredentialState;

    fn credential(auth_style: &str, header_name: &str) -> ProviderProfileCredential {
        ProviderProfileCredential {
            auth_style: auth_style.to_string(),
            header_name: header_name.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn dynamic_credential_key_matches_endpoint_host_port_and_path() {
        let key = "api.example.com\t443\t/repos/**\tgithub:access_token";

        assert!(dynamic_credential_key_matches(
            key,
            "api.example.com",
            443,
            "/repos/owner/repo"
        ));
        assert!(dynamic_credential_key_matches(
            "api.example.com\t443\t/repos/**\trev:42\tgithub:access_token",
            "api.example.com",
            443,
            "/repos/owner/repo"
        ));
        assert!(!dynamic_credential_key_matches(
            key,
            "uploads.example.com",
            443,
            "/repos/owner/repo"
        ));
        assert!(!dynamic_credential_key_matches(
            key,
            "api.example.com",
            80,
            "/repos/owner/repo"
        ));
        assert!(!dynamic_credential_key_matches(
            key,
            "api.example.com",
            443,
            "/orgs/owner"
        ));
    }

    #[test]
    fn dynamic_credential_key_matches_wildcard_hosts_and_empty_path() {
        let key = "*.example.com\t443\t\tprovider:access_token";

        assert!(dynamic_credential_key_matches(
            key,
            "api.example.com",
            443,
            "/anything"
        ));
        assert!(!dynamic_credential_key_matches(
            key,
            "api.other.com",
            443,
            "/anything"
        ));
        assert!(!dynamic_credential_key_matches(
            key,
            "nested.api.example.com",
            443,
            "/anything"
        ));
    }

    #[test]
    fn dynamic_credential_key_matches_case_insensitive_intra_label_wildcard() {
        let key = "*-API.Example.COM\t443\t\tprovider:access_token";

        assert!(dynamic_credential_key_matches(
            key,
            "tenant-api.example.com",
            443,
            "/anything"
        ));
        assert!(!dynamic_credential_key_matches(
            key,
            "api.deep.example.com",
            443,
            "/anything"
        ));
    }

    #[test]
    fn dynamic_credential_key_matches_double_wildcard_hosts() {
        let key = "**.example.com\t443\t\tprovider:access_token";

        assert!(dynamic_credential_key_matches(
            key,
            "api.example.com",
            443,
            "/anything"
        ));
        assert!(dynamic_credential_key_matches(
            key,
            "nested.api.example.com",
            443,
            "/anything"
        ));
    }

    #[test]
    fn dynamic_credential_match_score_prefers_path_specific_key() {
        let default_key = "alpha.default.svc.cluster.local\t80\t\tprovider:access_token";
        let path_key = "alpha.default.svc.cluster.local\t80\t/admin/**\tprovider:access_token";
        let request_path = "/admin/users";

        let default_score = dynamic_credential_key_match_score(
            default_key,
            "alpha.default.svc.cluster.local",
            80,
            request_path,
        )
        .expect("default key should match");
        let path_score = dynamic_credential_key_match_score(
            path_key,
            "alpha.default.svc.cluster.local",
            80,
            request_path,
        )
        .expect("path key should match");

        assert!(path_score > default_score);
    }

    #[test]
    fn inject_token_grant_header_replaces_existing_authorization() {
        let raw = b"GET /v1 HTTP/1.1\r\nHost: api.example.com\r\nauthorization: Bearer stale-token\r\nAccept: application/json\r\n\r\n";

        let rewritten =
            inject_token_grant_header(raw, &credential("bearer", "Authorization"), "grant-token")
                .expect("header should rewrite");
        let rewritten = String::from_utf8(rewritten).expect("rewritten header should be UTF-8");

        assert!(rewritten.contains("Authorization: Bearer grant-token\r\n"));
        assert!(!rewritten.contains("stale-token"));
        assert_eq!(
            rewritten
                .lines()
                .filter(|line| line
                    .split_once(':')
                    .is_some_and(|(name, _)| name.eq_ignore_ascii_case("authorization")))
                .count(),
            1
        );
    }

    #[test]
    fn inject_token_grant_header_replaces_existing_authorization_with_ows_before_colon() {
        let raw = b"GET /v1 HTTP/1.1\r\nHost: api.example.com\r\nAuthorization : Bearer stale-token\r\nAccept: application/json\r\n\r\n";

        let rewritten =
            inject_token_grant_header(raw, &credential("bearer", "Authorization"), "grant-token")
                .expect("header should rewrite");
        let rewritten = String::from_utf8(rewritten).expect("rewritten header should be UTF-8");

        assert!(rewritten.contains("Authorization: Bearer grant-token\r\n"));
        assert!(!rewritten.contains("stale-token"));
        assert_eq!(
            rewritten
                .lines()
                .filter(|line| line
                    .split_once(':')
                    .is_some_and(|(name, _)| name.trim().eq_ignore_ascii_case("authorization")))
                .count(),
            1
        );
    }

    #[test]
    fn token_grant_header_rejects_framing_and_connection_headers() {
        for header_name in ["Host", "Content-Length", "Transfer-Encoding", "Connection"] {
            let err =
                injected_credential_header("header", header_name, "grant-token", "token grant")
                    .expect_err("framing header override should be rejected");
            assert_eq!(
                err.to_string(),
                "token grant header_name may not override HTTP framing or connection headers"
            );
        }
    }

    #[test]
    fn proxy_delivery_named_header_accepts_safe_non_token_value() {
        let header = injected_credential_header(
            "header",
            "x-api-key",
            "key with spaces = allowed",
            "proxy delivery",
        )
        .expect("safe HTTP field value");
        assert_eq!(
            header,
            (
                "x-api-key".to_string(),
                "key with spaces = allowed".to_string()
            )
        );

        let error = injected_credential_header(
            "header",
            "x-api-key",
            "safe\r\ninjected: value",
            "proxy delivery",
        )
        .expect_err("CRLF injection must be rejected");
        assert_eq!(
            error.to_string(),
            "proxy delivery credential contains invalid HTTP header value characters"
        );
    }

    #[test]
    fn proxy_delivery_bearer_value_error_names_the_stored_credential() {
        let error = injected_credential_header("bearer", "", "has space", "proxy delivery")
            .expect_err("non-token68 bearer value must be rejected");
        assert_eq!(
            error.to_string(),
            "proxy delivery bearer credential is not a valid token68 value; check the stored provider credential"
        );

        let error = injected_credential_header("bearer", "", "has space", "token grant")
            .expect_err("token grant keeps its own wording");
        assert_eq!(
            error.to_string(),
            "token grant returned a malformed access token"
        );
    }

    #[test]
    fn inject_token_grant_header_preserves_header_terminator_before_body() {
        let raw = b"POST /v1 HTTP/1.1\r\nHost: api.example.com\r\nContent-Length: 2\r\n\r\nOK";

        let rewritten = inject_token_grant_header(raw, &credential("bearer", ""), "grant-token")
            .expect("header should rewrite");

        assert_eq!(
            rewritten,
            b"POST /v1 HTTP/1.1\r\nHost: api.example.com\r\nContent-Length: 2\r\nAuthorization: Bearer grant-token\r\n\r\nOK"
        );
    }

    #[test]
    fn inject_token_grant_header_uses_custom_header_style() {
        let raw = b"GET /v1 HTTP/1.1\r\nHost: api.example.com\r\nX-Api-Token: stale-token\r\n\r\n";

        let rewritten =
            inject_token_grant_header(raw, &credential("header", "X-Api-Token"), "grant-token")
                .expect("header should rewrite");
        let rewritten = String::from_utf8(rewritten).expect("rewritten header should be UTF-8");

        assert!(rewritten.contains("X-Api-Token: grant-token\r\n"));
        assert!(!rewritten.contains("stale-token"));
        assert!(!rewritten.contains("Bearer grant-token"));
    }

    #[test]
    fn inject_token_grant_header_rejects_malformed_access_token() {
        let raw = b"GET /v1 HTTP/1.1\r\nHost: api.example.com\r\n\r\n";

        let err = inject_token_grant_header(
            raw,
            &credential("bearer", "Authorization"),
            "grant-token\r\nX-Injected: yes",
        )
        .expect_err("malformed token must not be injected");

        assert_eq!(
            err.to_string(),
            "token grant returned a malformed access token"
        );
    }

    /// A proxy-delivered binding for `env_key` at `api.example.com:443/v1/**`.
    fn proxy_binding(
        identity: &str,
        auth_style: &str,
        header_name: &str,
    ) -> StaticCredentialBinding {
        StaticCredentialBinding {
            endpoints: vec![StaticCredentialEndpointBinding {
                host: "api.example.com".to_string(),
                port: 443,
                path: "/v1/**".to_string(),
            }],
            credential_identity: identity.to_string(),
            workload_credential_handle: String::new(),
            delivery: ProviderCredentialDelivery::Proxy as i32,
            auth_style: auth_style.to_string(),
            header_name: header_name.to_string(),
        }
    }

    fn proxy_state(
        env: &[(&str, &str)],
        bindings: Vec<(&str, StaticCredentialBinding)>,
    ) -> ProviderCredentialState {
        ProviderCredentialState::from_bound_environment(
            7,
            env.iter()
                .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                .collect(),
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            bindings
                .into_iter()
                .map(|(key, binding)| (key.to_string(), binding))
                .collect(),
            Vec::new(),
        )
        .expect("valid provider state")
    }

    fn proxy_ctx(state: &ProviderCredentialState, request_path: &str) -> L7EvalContext {
        L7EvalContext {
            host: "api.example.com".to_string(),
            port: 443,
            secret_resolver: state.resolver_for_endpoint("api.example.com", 443, request_path),
            provider_credentials: Some(state.clone()),
            provider_credential_revision: Some(7),
            ..Default::default()
        }
    }

    fn proxy_request(target: &str, raw_header: &[u8], body: &[u8]) -> L7Request {
        let mut raw = raw_header.to_vec();
        raw.extend_from_slice(body);
        L7Request {
            action: "POST".to_string(),
            target: target.to_string(),
            query_params: std::collections::HashMap::new(),
            raw_header: raw,
            body_length: BodyLength::ContentLength(body.len() as u64),
        }
    }

    #[test]
    fn proxy_delivery_replaces_header_without_changing_body() {
        let state = proxy_state(
            &[("API_KEY", "real-secret")],
            vec![("API_KEY", proxy_binding("provider:API_KEY", "bearer", ""))],
        );
        let ctx = proxy_ctx(&state, "/v1/chat");
        let body = br#"{"messages":[{"content":"ordinary environment output"}]}"#;
        let request = proxy_request(
            "/v1/chat",
            b"POST /v1/chat HTTP/1.1\r\nHost: api.example.com\r\nAuthorization: Bearer openshell-managed\r\n\r\n",
            body,
        );

        let injected = inject_static_if_needed(request, &ctx).expect("credential injection");
        let text = String::from_utf8(injected.raw_header).expect("HTTP request is UTF-8");
        assert!(text.contains("Authorization: Bearer real-secret\r\n"));
        assert!(!text.contains("openshell-managed"));
        assert_eq!(text.matches("Authorization:").count(), 1);
        assert!(text.ends_with(std::str::from_utf8(body).expect("body is UTF-8")));
    }

    #[test]
    fn proxy_delivery_named_header_is_added_when_absent() {
        let state = proxy_state(
            &[("API_KEY", "key with spaces")],
            vec![(
                "API_KEY",
                proxy_binding("provider:API_KEY", "header", "x-api-key"),
            )],
        );
        let ctx = proxy_ctx(&state, "/v1/chat");
        let request = proxy_request(
            "/v1/chat",
            b"POST /v1/chat HTTP/1.1\r\nHost: api.example.com\r\n\r\n",
            b"{}",
        );

        let injected = inject_static_if_needed(request, &ctx).expect("credential injection");
        let text = String::from_utf8(injected.raw_header).expect("HTTP request is UTF-8");
        assert!(text.contains("x-api-key: key with spaces\r\n"), "{text}");
        assert!(text.ends_with("\r\n\r\n{}"), "{text}");
    }

    #[test]
    fn proxy_delivery_passes_through_when_no_binding_matches() {
        let state = proxy_state(
            &[("API_KEY", "real-secret")],
            vec![("API_KEY", proxy_binding("provider:API_KEY", "bearer", ""))],
        );
        let ctx = proxy_ctx(&state, "/v2/other");
        let raw = b"POST /v2/other HTTP/1.1\r\nHost: api.example.com\r\nAuthorization: Bearer public\r\n\r\n";
        let request = proxy_request("/v2/other", raw, b"");

        let passed = inject_static_if_needed(request, &ctx).expect("no injection needed");
        assert_eq!(passed.raw_header, raw.to_vec());

        // Environment-delivered bindings never trigger injection.
        let mut environment_binding = proxy_binding("provider:API_KEY", "bearer", "");
        environment_binding.delivery = ProviderCredentialDelivery::Environment as i32;
        let state = proxy_state(
            &[("API_KEY", "real-secret")],
            vec![("API_KEY", environment_binding)],
        );
        let ctx = proxy_ctx(&state, "/v1/chat");
        let raw = b"POST /v1/chat HTTP/1.1\r\nHost: api.example.com\r\nAuthorization: Bearer public\r\n\r\n";
        let passed = inject_static_if_needed(proxy_request("/v1/chat", raw, b""), &ctx)
            .expect("environment delivery is untouched");
        assert_eq!(passed.raw_header, raw.to_vec());
    }

    #[test]
    fn proxy_delivery_collapses_aliases_and_rejects_competing_providers() {
        let state = proxy_state(
            &[("API_KEY", "real-secret"), ("API_KEY_ALIAS", "real-secret")],
            vec![
                ("API_KEY", proxy_binding("provider:API_KEY", "bearer", "")),
                (
                    "API_KEY_ALIAS",
                    proxy_binding("provider:API_KEY_ALIAS", "bearer", ""),
                ),
            ],
        );
        let ctx = proxy_ctx(&state, "/v1/chat");
        let injected = inject_static_if_needed(
            proxy_request(
                "/v1/chat",
                b"POST /v1/chat HTTP/1.1\r\nHost: api.example.com\r\n\r\n",
                b"",
            ),
            &ctx,
        )
        .expect("aliases resolve to one header");
        let text = String::from_utf8(injected.raw_header).expect("UTF-8");
        assert_eq!(
            text.matches("Authorization: Bearer real-secret\r\n")
                .count(),
            1
        );

        let state = proxy_state(
            &[("A_KEY", "secret-a"), ("B_KEY", "secret-b")],
            vec![
                ("A_KEY", proxy_binding("provider-a:A_KEY", "bearer", "")),
                ("B_KEY", proxy_binding("provider-b:B_KEY", "bearer", "")),
            ],
        );
        let ctx = proxy_ctx(&state, "/v1/chat");
        let error = inject_static_if_needed(
            proxy_request(
                "/v1/chat",
                b"POST /v1/chat HTTP/1.1\r\nHost: api.example.com\r\n\r\n",
                b"",
            ),
            &ctx,
        )
        .expect_err("two providers must fail closed");
        let message = error.to_string();
        assert!(
            message.contains("attach only one matching provider"),
            "{message}"
        );
        assert!(!message.contains("secret-a") && !message.contains("secret-b"));
    }

    #[test]
    fn proxy_delivery_fails_closed_without_resolver_or_after_revocation() {
        let state = proxy_state(
            &[("API_KEY", "real-secret")],
            vec![("API_KEY", proxy_binding("provider:API_KEY", "bearer", ""))],
        );
        let mut ctx = proxy_ctx(&state, "/v1/chat");
        ctx.secret_resolver = None;
        let error = inject_static_if_needed(
            proxy_request(
                "/v1/chat",
                b"POST /v1/chat HTTP/1.1\r\nHost: api.example.com\r\n\r\n",
                b"",
            ),
            &ctx,
        )
        .expect_err("missing resolver must not forward unauthenticated");
        assert!(error.to_string().contains("resolver unavailable"));

        // Revocation clears the live bindings, so a request re-scoped after
        // revocation has nothing to inject and passes through unchanged. The
        // relay paths re-scope immediately before injection and guard the
        // revision, so a stale pre-revocation resolver never reaches this call.
        state.revoke_static_provider_environment(8);
        let ctx = proxy_ctx(&state, "/v1/chat");
        assert!(ctx.secret_resolver.is_none());
        let raw = b"POST /v1/chat HTTP/1.1\r\nHost: api.example.com\r\nAuthorization: Bearer public\r\n\r\n";
        let passed = inject_static_if_needed(proxy_request("/v1/chat", raw, b""), &ctx)
            .expect("revoked bindings leave the request untouched");
        assert_eq!(passed.raw_header, raw.to_vec());
    }

    #[test]
    fn proxy_delivery_rejects_bearer_values_that_are_not_token68() {
        let state = proxy_state(
            &[("API_KEY", "has space")],
            vec![("API_KEY", proxy_binding("provider:API_KEY", "bearer", ""))],
        );
        let ctx = proxy_ctx(&state, "/v1/chat");
        let error = inject_static_if_needed(
            proxy_request(
                "/v1/chat",
                b"POST /v1/chat HTTP/1.1\r\nHost: api.example.com\r\n\r\n",
                b"",
            ),
            &ctx,
        )
        .expect_err("malformed bearer value must not be injected");
        let message = error.to_string();
        assert!(
            message.contains("proxy delivery bearer credential"),
            "{message}"
        );
        assert!(!message.contains("has space"), "{message}");
    }

    #[tokio::test]
    async fn inject_if_needed_uses_configured_resolver() {
        let fixture = TokenGrantTestFixture::success(
            "api.example.com\t443\t/v1/**\tprovider:access_token",
            "grant-token",
        );

        let ctx = L7EvalContext {
            host: "api.example.com".into(),
            port: 443,
            policy_name: "api".into(),
            binary_path: "/usr/bin/curl".into(),
            ancestors: vec![],
            cmdline_paths: vec![],
            secret_resolver: None,
            dynamic_credentials: Some(fixture.dynamic_credentials()),
            token_grant_resolver: Some(fixture.resolver()),
            ..Default::default()
        };
        let req = L7Request {
            action: "GET".to_string(),
            target: "/v1/projects".to_string(),
            query_params: std::collections::HashMap::new(),
            raw_header: b"GET /v1/projects HTTP/1.1\r\nHost: api.example.com\r\n\r\n".to_vec(),
            body_length: BodyLength::None,
        };

        let rewritten = inject_if_needed(req, &ctx)
            .await
            .expect("fake token grant should inject");
        let rewritten =
            String::from_utf8(rewritten.raw_header).expect("rewritten request should be UTF-8");

        assert!(rewritten.contains("Authorization: Bearer grant-token\r\n"));
        fixture.assert_one_request("api.example.com\t443\t/v1/**\tprovider:access_token");
    }

    #[tokio::test]
    async fn inject_if_needed_passes_token_exchange_grant_to_resolver() {
        let fixture = TokenGrantTestFixture::success_token_exchange(
            "api.example.com\t443\t/v1/**\tprovider:access_token",
            "grant-token",
        );

        let ctx = L7EvalContext {
            host: "api.example.com".into(),
            port: 443,
            policy_name: "api".into(),
            binary_path: "/usr/bin/curl".into(),
            ancestors: vec![],
            cmdline_paths: vec![],
            secret_resolver: None,
            activity_tx: None,
            dynamic_credentials: Some(fixture.dynamic_credentials()),
            token_grant_resolver: Some(fixture.resolver()),
            ..Default::default()
        };
        let req = L7Request {
            action: "GET".to_string(),
            target: "/v1/projects".to_string(),
            query_params: std::collections::HashMap::new(),
            raw_header: b"GET /v1/projects HTTP/1.1\r\nHost: api.example.com\r\n\r\n".to_vec(),
            body_length: BodyLength::None,
        };

        let rewritten = inject_if_needed(req, &ctx)
            .await
            .expect("fake token exchange grant should inject");
        let rewritten =
            String::from_utf8(rewritten.raw_header).expect("rewritten request should be UTF-8");

        assert!(rewritten.contains("Authorization: Bearer grant-token\r\n"));
        fixture.assert_one_token_exchange_request(
            "api.example.com\t443\t/v1/**\tprovider:access_token",
        );
    }

    #[tokio::test]
    async fn inject_if_needed_rejects_malformed_resolver_token() {
        let fixture = TokenGrantTestFixture::success(
            "api.example.com\t443\t/v1/**\tprovider:access_token",
            "grant-token\r\nX-Injected: yes",
        );

        let ctx = L7EvalContext {
            host: "api.example.com".into(),
            port: 443,
            policy_name: "api".into(),
            binary_path: "/usr/bin/curl".into(),
            ancestors: vec![],
            cmdline_paths: vec![],
            secret_resolver: None,
            dynamic_credentials: Some(fixture.dynamic_credentials()),
            token_grant_resolver: Some(fixture.resolver()),
            ..Default::default()
        };
        let req = L7Request {
            action: "GET".to_string(),
            target: "/v1/projects".to_string(),
            query_params: std::collections::HashMap::new(),
            raw_header: b"GET /v1/projects HTTP/1.1\r\nHost: api.example.com\r\n\r\n".to_vec(),
            body_length: BodyLength::None,
        };

        let err = inject_if_needed(req, &ctx)
            .await
            .expect_err("malformed resolver token should fail closed");

        assert_eq!(
            err.to_string(),
            "token grant returned a malformed access token"
        );
        fixture.assert_one_request("api.example.com\t443\t/v1/**\tprovider:access_token");
    }
}
