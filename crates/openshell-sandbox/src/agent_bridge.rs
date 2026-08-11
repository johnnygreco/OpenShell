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
    AgentConversationEvaluation, AgentConversationTarget, Decision, RequestContext,
    SupervisorMiddlewarePhase,
};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tracing::{debug, warn};

pub const BRIDGE_ADDR: &str = "127.0.0.1:8193";
pub const BRIDGE_PATH: &str = "/v1/agent/conversation";
pub const BRIDGE_URL: &str = "http://127.0.0.1:8193/v1/agent/conversation";
pub const BRIDGE_URL_ENV: &str = "OPENSHELL_PI_CONVERSATION_URL";

const MAX_BRIDGE_BODY_BYTES: usize = 256 * 1024;
const MAX_ADMISSION_BODY_BYTES: usize = 32 * 1024;
const PI_HARNESS_VERSION: &str = "extension-v1";

#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub middleware_name: String,
    pub sandbox_id: String,
    pub provider_host: String,
    pub middleware_config: prost_types::Struct,
}

#[derive(Clone)]
struct BridgeState {
    runner: openshell_supervisor_middleware::ChainRunner,
    config: Arc<BridgeConfig>,
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

#[derive(Debug, Serialize)]
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
    runner: openshell_supervisor_middleware::ChainRunner,
    config: BridgeConfig,
) -> tokio::task::JoinHandle<()> {
    let state = BridgeState {
        runner,
        config: Arc::new(config),
    };
    tokio::spawn(async move {
        let app = Router::new()
            .route(BRIDGE_PATH, post(evaluate))
            .layer(DefaultBodyLimit::max(MAX_BRIDGE_BODY_BYTES))
            .with_state(state);
        if let Err(error) = axum::serve(listener, app).await {
            warn!(%error, "Pi admission bridge stopped");
        }
    })
}

async fn evaluate(State(state): State<BridgeState>, Json(input): Json<BridgeRequest>) -> Response {
    if !is_valid_admission_request(&input) {
        return (
            StatusCode::BAD_REQUEST,
            Json(BridgeError {
                error: "invalid_admission_request",
            }),
        )
            .into_response();
    }

    let evaluation = AgentConversationEvaluation {
        phase: SupervisorMiddlewarePhase::AgentContext as i32,
        context: Some(RequestContext {
            request_id: uuid::Uuid::new_v4().to_string(),
            sandbox_id: state.config.sandbox_id.clone(),
            originating_process: None,
        }),
        config: Some(state.config.middleware_config.clone()),
        target: Some(AgentConversationTarget {
            harness: "pi".into(),
            harness_version: input.harness_version,
            hook: "rendered_prompt_admission".into(),
            schema_version: "openshell.pi-input.v1".into(),
            scheme: "https".into(),
            host: state.config.provider_host.clone(),
            port: 443,
            path: "/v1/chat/completions".into(),
        }),
        middleware_name: state.config.middleware_name.clone(),
        session_id: input.session_id,
        turn_id: input.submission_id,
        request_body: input.request_body,
        ..Default::default()
    };

    let result = match state.runner.evaluate_agent_conversation(evaluation).await {
        Ok(result) => result,
        Err(error) => {
            debug!(error = %error, "Pi admission evaluation failed");
            return (
                StatusCode::BAD_GATEWAY,
                Json(BridgeError {
                    error: "admission_unavailable",
                }),
            )
                .into_response();
        }
    };

    match Decision::try_from(result.decision).unwrap_or(Decision::Unspecified) {
        Decision::Allow => Json(BridgeResponse {
            decision: "allow",
            replacement_body: result
                .has_replacement_body
                .then_some(result.replacement_body),
            receipt: (!result.attestation.is_empty()).then_some(result.attestation),
            reason_code: None,
            metadata: Some(result.metadata),
        })
        .into_response(),
        Decision::Deny => Json(BridgeResponse {
            decision: "deny",
            replacement_body: None,
            receipt: None,
            reason_code: (!result.reason_code.is_empty()).then_some(result.reason_code),
            metadata: None,
        })
        .into_response(),
        Decision::Unspecified => (
            StatusCode::BAD_GATEWAY,
            Json(BridgeError {
                error: "invalid_admission_response",
            }),
        )
            .into_response(),
    }
}

fn is_valid_admission_request(input: &BridgeRequest) -> bool {
    input.harness_version == PI_HARNESS_VERSION
        && !input.request_body.is_empty()
        && input.request_body.len() <= MAX_ADMISSION_BODY_BYTES
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
    fn admission_request_requires_the_pinned_harness_version() {
        assert!(!is_valid_admission_request(&request("0.84.1", 1)));
        assert!(is_valid_admission_request(&request(PI_HARNESS_VERSION, 1)));
    }

    #[test]
    fn admission_request_enforces_the_logical_body_limit() {
        assert!(!is_valid_admission_request(&request(PI_HARNESS_VERSION, 0)));
        assert!(is_valid_admission_request(&request(
            PI_HARNESS_VERSION,
            MAX_ADMISSION_BODY_BYTES,
        )));
        assert!(!is_valid_admission_request(&request(
            PI_HARNESS_VERSION,
            MAX_ADMISSION_BODY_BYTES + 1,
        )));
    }
}
