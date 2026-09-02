// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Supervisor-owned loopback bridge for managed agent admission.

use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use base64::Engine as _;
use openshell_core::proto::{
    AgentConversationEvaluation, AgentConversationResult, AgentConversationTarget, Decision,
    RequestContext, SupervisorMiddlewarePhase,
};
use openshell_supervisor_network::opa::OpaEngine;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tracing::{debug, warn};

pub const BRIDGE_ADDR: &str = "127.0.0.1:8193";
pub const BRIDGE_PATH: &str = "/v1/agent/conversation";
pub const BRIDGE_URL: &str = "http://127.0.0.1:8193/v1/agent/conversation";
pub const BRIDGE_URL_ENV: &str = "OPENSHELL_AGENT_CONVERSATION_URL";

const MAX_ADMISSION_BODY_BYTES: usize =
    openshell_supervisor_middleware::MAX_MIDDLEWARE_PAYLOAD_BYTES;
const MAX_BRIDGE_BODY_BYTES: usize = MAX_ADMISSION_BODY_BYTES.div_ceil(3) * 4 + 64 * 1024;

#[derive(Debug, Clone)]
pub struct BridgeSelection {
    pub middleware_name: String,
    pub bindings: Vec<BridgeBinding>,
    pub provider_scheme: String,
    pub provider_host: String,
    pub provider_port: u32,
    pub middleware_config: prost_types::Struct,
}

#[derive(Debug, Clone)]
pub struct BridgeBinding {
    pub harness: String,
    pub hook: String,
    pub schema_version: String,
    pub max_payload_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct BridgeRuntimeSnapshot {
    pub generation: u64,
    pub selection: Option<BridgeSelection>,
}

#[derive(Clone)]
struct BridgeState {
    engine: Arc<OpaEngine>,
    runtime: watch::Receiver<BridgeRuntimeSnapshot>,
    sandbox_id: Arc<str>,
    sandbox_name: Arc<str>,
    workspace: watch::Receiver<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BridgeRequest {
    harness_version: String,
    hook: String,
    schema_version: String,
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    submission_id: String,
    #[serde(rename = "request_body_b64", deserialize_with = "deserialize_base64")]
    request_body: Vec<u8>,
}

#[derive(Debug, PartialEq, Serialize)]
struct BridgeResponse {
    decision: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(
        rename = "replacement_body_b64",
        serialize_with = "serialize_optional_base64"
    )]
    replacement_body: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    handle: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason_code: Option<String>,
}

#[derive(Debug, Serialize)]
struct BridgeError {
    error: &'static str,
}

pub fn spawn(
    listener: TcpListener,
    engine: Arc<OpaEngine>,
    runtime: watch::Receiver<BridgeRuntimeSnapshot>,
    sandbox_id: String,
    sandbox_name: String,
    workspace: watch::Receiver<String>,
) -> tokio::task::JoinHandle<()> {
    let state = BridgeState {
        engine,
        runtime,
        sandbox_id: Arc::from(sandbox_id),
        sandbox_name: Arc::from(sandbox_name),
        workspace,
    };
    tokio::spawn(async move {
        let app = Router::new()
            .route(BRIDGE_PATH, post(evaluate))
            .layer(DefaultBodyLimit::max(MAX_BRIDGE_BODY_BYTES))
            .with_state(state);
        if let Err(error) = axum::serve(listener, app).await {
            warn!(%error, "agent admission bridge stopped");
        }
    })
}

async fn evaluate(State(state): State<BridgeState>, Json(input): Json<BridgeRequest>) -> Response {
    let runtime = state.runtime.borrow().clone();
    if runtime.generation != state.engine.current_generation() {
        return admission_unavailable();
    }
    let Some(selection) = runtime.selection else {
        return admission_unavailable();
    };
    let Some(binding) = selected_binding(&selection, &input) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(BridgeError {
                error: "invalid_admission_request",
            }),
        )
            .into_response();
    };

    let Ok(generation) = state.engine.generation_guard(runtime.generation) else {
        return admission_unavailable();
    };
    let Ok(runner) = state.engine.middleware_runner() else {
        return admission_unavailable();
    };
    if generation.ensure_current().is_err() {
        return admission_unavailable();
    }
    let workspace = state.workspace.borrow().clone();
    let evaluation = build_evaluation(
        &state.sandbox_id,
        &state.sandbox_name,
        &workspace,
        &selection,
        binding,
        input,
    );

    let result = match runner.evaluate_agent_conversation(evaluation).await {
        Ok(result) => result,
        Err(error) => {
            debug!(error = %error, "agent admission evaluation failed");
            return admission_unavailable();
        }
    };
    if generation.ensure_current().is_err() {
        return admission_unavailable();
    }

    match bridge_result(
        &runner,
        &state.sandbox_id,
        runtime.generation,
        &selection,
        result,
    ) {
        Ok(result) => Json(result).into_response(),
        Err(error) => (StatusCode::BAD_GATEWAY, Json(BridgeError { error })).into_response(),
    }
}

fn deserialize_base64<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: Deserializer<'de>,
{
    let encoded = String::deserialize(deserializer)?;
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(D::Error::custom)
}

fn serialize_optional_base64<S>(body: &Option<Vec<u8>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    body.as_ref()
        .map(|body| base64::engine::general_purpose::STANDARD.encode(body))
        .serialize(serializer)
}

fn build_evaluation(
    sandbox_id: &str,
    sandbox_name: &str,
    workspace: &str,
    selection: &BridgeSelection,
    binding: &BridgeBinding,
    input: BridgeRequest,
) -> AgentConversationEvaluation {
    AgentConversationEvaluation {
        phase: SupervisorMiddlewarePhase::AgentContext as i32,
        context: Some(RequestContext {
            request_id: uuid::Uuid::new_v4().to_string(),
            sandbox_id: sandbox_id.to_string(),
            sandbox_name: sandbox_name.to_string(),
            workspace: workspace.to_string(),
            originating_process: None,
        }),
        config: Some(selection.middleware_config.clone()),
        target: Some(AgentConversationTarget {
            harness: binding.harness.clone(),
            harness_version: input.harness_version,
            hook: binding.hook.clone(),
            schema_version: binding.schema_version.clone(),
            scheme: selection.provider_scheme.clone(),
            host: selection.provider_host.clone(),
            port: selection.provider_port,
            path: String::new(),
        }),
        middleware_name: selection.middleware_name.clone(),
        session_id: input.session_id,
        turn_id: input.submission_id,
        request_body: input.request_body,
    }
}

fn bridge_result(
    runner: &openshell_supervisor_middleware::ChainRunner,
    sandbox_id: &str,
    policy_generation: u64,
    selection: &BridgeSelection,
    result: AgentConversationResult,
) -> Result<BridgeResponse, &'static str> {
    match Decision::try_from(result.decision).unwrap_or(Decision::Unspecified) {
        Decision::Allow => {
            let handle = if result.attestation.is_empty() {
                None
            } else {
                let port =
                    u16::try_from(selection.provider_port).map_err(|_| "invalid_provider_port")?;
                Some(
                    runner
                        .issue_agent_admission_handle(
                            openshell_supervisor_middleware::AgentAdmissionGrantInput {
                                sandbox_id,
                                middleware_name: &selection.middleware_name,
                                scheme: &selection.provider_scheme,
                                host: &selection.provider_host,
                                port,
                                policy_generation,
                                attestation: result.attestation,
                            },
                        )
                        .map_err(|_| "admission_store_unavailable")?,
                )
            };
            Ok(BridgeResponse {
                decision: "allow",
                replacement_body: result
                    .has_replacement_body
                    .then_some(result.replacement_body),
                handle,
                reason_code: None,
            })
        }
        Decision::Deny => Ok(BridgeResponse {
            decision: "deny",
            replacement_body: None,
            handle: None,
            reason_code: (!result.reason_code.is_empty()).then_some(result.reason_code),
        }),
        Decision::Unspecified => Err("invalid_admission_response"),
    }
}

fn admission_unavailable() -> Response {
    (
        StatusCode::BAD_GATEWAY,
        Json(BridgeError {
            error: "admission_unavailable",
        }),
    )
        .into_response()
}

fn selected_binding<'a>(
    selection: &'a BridgeSelection,
    input: &BridgeRequest,
) -> Option<&'a BridgeBinding> {
    if !is_stable_identifier(&input.harness_version)
        || input.request_body.is_empty()
        || input.request_body.len() > MAX_ADMISSION_BODY_BYTES
    {
        return None;
    }
    selection.bindings.iter().find(|binding| {
        binding.hook == input.hook
            && binding.schema_version == input.schema_version
            && input.request_body.len() <= binding.max_payload_bytes
    })
}

fn is_stable_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(harness_version: &str, body_len: usize) -> BridgeRequest {
        BridgeRequest {
            harness_version: harness_version.into(),
            hook: "user_input".into(),
            schema_version: "example.input.v1".into(),
            session_id: String::new(),
            submission_id: String::new(),
            request_body: vec![b'x'; body_len],
        }
    }

    #[test]
    fn admission_request_requires_a_stable_harness_version() {
        let selection = selection(1);
        assert!(selected_binding(&selection, &request("", 1)).is_none());
        assert!(selected_binding(&selection, &request("bad version", 1)).is_none());
        assert!(selected_binding(&selection, &request("sdk-v1", 1)).is_some());
        let mut unadvertised = request("sdk-v1", 1);
        unadvertised.schema_version = "other.v1".into();
        assert!(selected_binding(&selection, &unadvertised).is_none());
    }

    #[test]
    fn admission_request_json_requires_the_base64_field() {
        let input: BridgeRequest = serde_json::from_value(serde_json::json!({
            "harness_version": "sdk-v1",
            "hook": "user_input",
            "schema_version": "example.input.v1",
            "request_body_b64": "eHg="
        }))
        .unwrap();
        assert_eq!(input.request_body, b"xx");

        assert!(
            serde_json::from_value::<BridgeRequest>(serde_json::json!({
                "harness_version": "sdk-v1",
                "hook": "user_input",
                "schema_version": "example.input.v1",
                "request_body_b64": "not base64"
            }))
            .is_err()
        );

        assert!(
            serde_json::from_value::<BridgeRequest>(serde_json::json!({
                "harness_version": "sdk-v1",
                "hook": "user_input",
                "schema_version": "example.input.v1",
                "request_body": [120]
            }))
            .is_err()
        );
    }

    #[test]
    fn admission_request_enforces_the_logical_body_limit() {
        let max_selection = selection(MAX_ADMISSION_BODY_BYTES);
        assert!(selected_binding(&max_selection, &request("sdk-v1", 0)).is_none());
        assert!(
            selected_binding(&max_selection, &request("sdk-v1", MAX_ADMISSION_BODY_BYTES))
                .is_some()
        );
        assert!(
            selected_binding(
                &max_selection,
                &request("sdk-v1", MAX_ADMISSION_BODY_BYTES + 1)
            )
            .is_none()
        );
        assert!(selected_binding(&selection(1), &request("sdk-v1", 2)).is_none());
    }

    fn selection(limit: usize) -> BridgeSelection {
        BridgeSelection {
            middleware_name: "operator/guard".into(),
            bindings: vec![BridgeBinding {
                harness: "another-agent".into(),
                hook: "user_input".into(),
                schema_version: "example.input.v1".into(),
                max_payload_bytes: limit,
            }],
            provider_scheme: "https".into(),
            provider_host: "api.example.com".into(),
            provider_port: 8443,
            middleware_config: prost_types::Struct::default(),
        }
    }

    #[test]
    fn evaluation_uses_advertised_binding_and_trusted_destination() {
        let selection = selection(1024);
        let binding = &selection.bindings[0];
        let evaluation = build_evaluation(
            "sandbox-1",
            "friendly-sandbox",
            "workspace-1",
            &selection,
            binding,
            request("plugin-v2", 2),
        );
        let target = evaluation.target.expect("target");
        assert_eq!(target.harness, "another-agent");
        assert_eq!(target.harness_version, "plugin-v2");
        assert_eq!(target.hook, "user_input");
        assert_eq!(target.schema_version, "example.input.v1");
        assert_eq!(target.host, "api.example.com");
        assert_eq!(target.port, 8443);
        assert!(target.path.is_empty());
    }

    #[test]
    fn result_mapping_returns_an_opaque_retryable_handle() {
        let runner = openshell_supervisor_middleware::ChainRunner::default();
        let selection = selection(1024);
        let allow = bridge_result(
            &runner,
            "sandbox-1",
            7,
            &selection,
            AgentConversationResult {
                decision: Decision::Allow as i32,
                attestation: b"receipt".to_vec(),
                replacement_body: b"redacted".to_vec(),
                has_replacement_body: true,
                metadata: std::collections::HashMap::from([("internal".into(), "value".into())]),
                ..Default::default()
            },
        )
        .expect("allow");
        assert_eq!(allow.decision, "allow");
        let response_json = serde_json::to_value(&allow).unwrap();
        assert_eq!(response_json["replacement_body_b64"], "cmVkYWN0ZWQ=");
        assert!(response_json.get("metadata").is_none());
        let handle = allow.handle.expect("handle");
        assert_ne!(handle.as_bytes(), b"receipt");
        let request = openshell_supervisor_middleware::AgentAdmissionRequest {
            sandbox_id: "sandbox-1",
            middleware_name: "operator/guard",
            scheme: "https",
            host: "api.example.com",
            port: 8443,
            policy_generation: 7,
        };
        assert_eq!(
            runner
                .resolve_agent_admission_handle(&handle, &request)
                .unwrap(),
            Some(b"receipt".to_vec())
        );
        assert_eq!(
            runner
                .resolve_agent_admission_handle(&handle, &request)
                .unwrap(),
            Some(b"receipt".to_vec())
        );
        assert_eq!(allow.replacement_body, Some(b"redacted".to_vec()));

        let empty_replacement = bridge_result(
            &runner,
            "sandbox-1",
            7,
            &selection,
            AgentConversationResult {
                decision: Decision::Allow as i32,
                has_replacement_body: true,
                ..Default::default()
            },
        )
        .expect("empty replacement");
        assert_eq!(
            serde_json::to_value(&empty_replacement).unwrap()["replacement_body_b64"],
            ""
        );

        let deny = bridge_result(
            &runner,
            "sandbox-1",
            7,
            &selection,
            AgentConversationResult {
                decision: Decision::Deny as i32,
                reason_code: "policy_denied".into(),
                ..Default::default()
            },
        )
        .expect("deny");
        assert_eq!(deny.decision, "deny");
        assert_eq!(deny.reason_code.as_deref(), Some("policy_denied"));
        assert_eq!(deny.handle, None);
        assert_eq!(deny.replacement_body, None);
    }

    #[test]
    fn result_mapping_allows_without_an_attestation_handle() {
        let response = bridge_result(
            &openshell_supervisor_middleware::ChainRunner::default(),
            "sandbox-1",
            7,
            &selection(1024),
            AgentConversationResult {
                decision: Decision::Allow as i32,
                ..Default::default()
            },
        )
        .expect("allow without attestation");

        assert_eq!(response.decision, "allow");
        assert_eq!(response.handle, None);
    }
}
