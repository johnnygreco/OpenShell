// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Supervisor-owned loopback bridge for managed agent admission.

use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use openshell_core::proto::{
    AgentConversationEvaluation, AgentConversationResult, AgentConversationTarget, Decision,
    RequestContext, SupervisorMiddlewarePhase,
};
use openshell_supervisor_network::opa::OpaEngine;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tracing::{debug, warn};

pub const BRIDGE_ADDR: &str = "127.0.0.1:8193";
pub const BRIDGE_PATH: &str = "/v1/agent/conversation";
pub const BRIDGE_URL: &str = "http://127.0.0.1:8193/v1/agent/conversation";
pub const BRIDGE_URL_ENV: &str = "OPENSHELL_AGENT_CONVERSATION_URL";
pub const LEGACY_PI_BRIDGE_URL_ENV: &str = "OPENSHELL_PI_CONVERSATION_URL";

const MAX_BRIDGE_BODY_BYTES: usize = 256 * 1024;
const MAX_ADMISSION_BODY_BYTES: usize = 32 * 1024;

#[derive(Debug, Clone)]
pub struct BridgeSelection {
    pub middleware_name: String,
    pub harness: String,
    pub hook: String,
    pub schema_version: String,
    pub provider_scheme: String,
    pub provider_host: String,
    pub provider_port: u32,
    pub max_payload_bytes: usize,
    pub middleware_config: prost_types::Struct,
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
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    submission_id: String,
    request_body: Vec<u8>,
}

#[derive(Debug, PartialEq, Serialize)]
struct BridgeResponse {
    decision: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    replacement_body: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    receipt: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<std::collections::HashMap<String, String>>,
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
    if !is_valid_admission_request(&input, selection.max_payload_bytes) {
        return (
            StatusCode::BAD_REQUEST,
            Json(BridgeError {
                error: "invalid_admission_request",
            }),
        )
            .into_response();
    }

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

    match bridge_result(result) {
        Ok(result) => Json(result).into_response(),
        Err(error) => (StatusCode::BAD_GATEWAY, Json(BridgeError { error })).into_response(),
    }
}

fn build_evaluation(
    sandbox_id: &str,
    sandbox_name: &str,
    workspace: &str,
    selection: &BridgeSelection,
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
            harness: selection.harness.clone(),
            harness_version: input.harness_version,
            hook: selection.hook.clone(),
            schema_version: selection.schema_version.clone(),
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

fn bridge_result(result: AgentConversationResult) -> Result<BridgeResponse, &'static str> {
    match Decision::try_from(result.decision).unwrap_or(Decision::Unspecified) {
        Decision::Allow => Ok(BridgeResponse {
            decision: "allow",
            replacement_body: result
                .has_replacement_body
                .then_some(result.replacement_body),
            receipt: (!result.attestation.is_empty()).then_some(result.attestation),
            reason_code: None,
            metadata: (!result.metadata.is_empty()).then_some(result.metadata),
        }),
        Decision::Deny => Ok(BridgeResponse {
            decision: "deny",
            replacement_body: None,
            receipt: None,
            reason_code: (!result.reason_code.is_empty()).then_some(result.reason_code),
            metadata: None,
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

fn is_valid_admission_request(input: &BridgeRequest, binding_limit: usize) -> bool {
    is_stable_identifier(&input.harness_version)
        && !input.request_body.is_empty()
        && input.request_body.len() <= MAX_ADMISSION_BODY_BYTES
        && input.request_body.len() <= binding_limit
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
            session_id: String::new(),
            submission_id: String::new(),
            request_body: vec![b'x'; body_len],
        }
    }

    #[test]
    fn admission_request_requires_a_stable_harness_version() {
        assert!(!is_valid_admission_request(&request("", 1), 1));
        assert!(!is_valid_admission_request(&request("bad version", 1), 1));
        assert!(is_valid_admission_request(&request("extension-v1", 1), 1));
    }

    #[test]
    fn admission_request_enforces_the_logical_body_limit() {
        assert!(!is_valid_admission_request(
            &request("extension-v1", 0),
            usize::MAX
        ));
        assert!(is_valid_admission_request(
            &request("extension-v1", MAX_ADMISSION_BODY_BYTES,),
            usize::MAX
        ));
        assert!(!is_valid_admission_request(
            &request("extension-v1", MAX_ADMISSION_BODY_BYTES + 1,),
            usize::MAX
        ));
        assert!(!is_valid_admission_request(&request("extension-v1", 2), 1));
    }

    #[test]
    fn evaluation_uses_advertised_binding_and_trusted_destination() {
        let selection = BridgeSelection {
            middleware_name: "operator/guard".into(),
            harness: "another-agent".into(),
            hook: "user_input".into(),
            schema_version: "example.input.v1".into(),
            provider_scheme: "https".into(),
            provider_host: "api.example.com".into(),
            provider_port: 8443,
            max_payload_bytes: 1024,
            middleware_config: prost_types::Struct::default(),
        };
        let evaluation = build_evaluation(
            "sandbox-1",
            "friendly-sandbox",
            "workspace-1",
            &selection,
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
    fn result_mapping_preserves_only_operation_outputs() {
        let allow = bridge_result(AgentConversationResult {
            decision: Decision::Allow as i32,
            attestation: b"receipt".to_vec(),
            replacement_body: b"redacted".to_vec(),
            has_replacement_body: true,
            ..Default::default()
        })
        .expect("allow");
        assert_eq!(allow.decision, "allow");
        assert_eq!(allow.receipt, Some(b"receipt".to_vec()));
        assert_eq!(allow.replacement_body, Some(b"redacted".to_vec()));

        let deny = bridge_result(AgentConversationResult {
            decision: Decision::Deny as i32,
            reason_code: "policy_denied".into(),
            ..Default::default()
        })
        .expect("deny");
        assert_eq!(deny.decision, "deny");
        assert_eq!(deny.reason_code.as_deref(), Some("policy_denied"));
        assert_eq!(deny.receipt, None);
        assert_eq!(deny.replacement_body, None);
    }
}
