//! OpenAI-compatible mock server for deterministic agent tests.
//!
//! This module keeps the API surface small on purpose: a single
//! chat-completions-style endpoint, deterministic preset routing, and request
//! history that tests can assert against.

use crate::backend::MockLLMBackend;
use anyhow::Result;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use mofa_foundation::orchestrator::ModelOrchestrator;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use tower::util::ServiceExt;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MockChatCompletionRoute {
    pub prompt_substring: String,
    pub response: String,
}

impl MockChatCompletionRoute {
    /// Define one substring-based routing rule for the mock server.
    pub fn new(prompt_substring: impl Into<String>, response: impl Into<String>) -> Self {
        Self {
            prompt_substring: prompt_substring.into(),
            response: response.into(),
        }
    }
}

#[derive(Clone)]
struct MockServerState {
    backend: MockLLMBackend,
    request_history: Arc<RwLock<Vec<ChatCompletionRequest>>>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: i64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChoice {
    index: usize,
    message: ChatCompletionMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorPayload,
}

#[derive(Debug, Serialize)]
struct ErrorPayload {
    message: String,
}

pub struct MockLlmServer {
    backend: MockLLMBackend,
    request_history: Arc<RwLock<Vec<ChatCompletionRequest>>>,
}

impl MockLlmServer {
    /// Build a fresh server with isolated backend and request history state.
    pub async fn start() -> Result<Self> {
        Ok(Self {
            backend: MockLLMBackend::new(),
            request_history: Arc::new(RwLock::new(Vec::new())),
        })
    }

    pub fn backend(&self) -> MockLLMBackend {
        self.backend.clone()
    }

    pub fn add_route(&self, route: MockChatCompletionRoute) {
        self.backend
            .add_response(&route.prompt_substring, &route.response);
    }

    pub fn add_response(&self, prompt_substring: &str, response: &str) {
        self.backend.add_response(prompt_substring, response);
    }

    /// Add a repeating response sequence for repeated matching requests.
    pub fn add_response_sequence(&self, prompt_substring: &str, responses: Vec<&str>) {
        self.backend
            .add_response_sequence(prompt_substring, responses);
    }

    pub fn set_fallback(&mut self, response: &str) {
        self.backend.set_fallback(response);
    }

    pub fn request_history(&self) -> Vec<ChatCompletionRequest> {
        self.request_history.read().expect("lock poisoned").clone()
    }

    pub fn clear_request_history(&self) {
        self.request_history.write().expect("lock poisoned").clear();
    }

    pub async fn chat_completions(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<(StatusCode, serde_json::Value)> {
        // Drive the Axum router in-process so tests keep HTTP semantics without
        // depending on socket permissions in CI or sandboxes.
        let app = self.router();
        let http_request = Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(request)?))?;

        let response = app.oneshot(http_request).await?;
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
        let json = serde_json::from_slice(&body)?;

        Ok((status, json))
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }

    /// Recreate the router from shared state for each in-process request.
    fn router(&self) -> Router {
        let state = MockServerState {
            backend: self.backend.clone(),
            request_history: Arc::clone(&self.request_history),
        };

        Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state)
    }
}

async fn chat_completions(
    State(state): State<MockServerState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (StatusCode, Json<ErrorEnvelope>)> {
    // Keep the original OpenAI-shaped payload so tests can assert exact inputs.
    state
        .request_history
        .write()
        .expect("lock poisoned")
        .push(request.clone());

    // Collapse chat messages into one deterministic prompt for the underlying
    // substring-based mock backend.
    let prompt = request
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let response = state
        .backend
        .infer(&request.model, &prompt)
        .await
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorEnvelope {
                    error: ErrorPayload {
                        message: error.to_string(),
                    },
                }),
            )
        })?;

    Ok(Json(ChatCompletionResponse {
        id: format!("mockcmpl-{}", chrono::Utc::now().timestamp_millis()),
        object: "chat.completion",
        created: chrono::Utc::now().timestamp(),
        model: request.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".into(),
                content: response,
            },
            finish_reason: "stop",
        }],
    }))
}
