//! Focused coverage for the mock chat-completions surface and request history.

use mofa_foundation::orchestrator::OrchestratorError;
use mofa_testing::mock_server::{
    ChatCompletionMessage, ChatCompletionRequest, MockChatCompletionRoute, MockLlmServer,
};
use axum::http::StatusCode;

// Build the smallest OpenAI-shaped request body used by the mock server.
fn request(model: &str, contents: &[&str]) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: model.into(),
        messages: contents
            .iter()
            .map(|content| ChatCompletionMessage {
                role: "user".into(),
                content: (*content).into(),
            })
            .collect(),
    }
}

#[tokio::test]
async fn mock_server_returns_openai_style_response() {
    let mut server = MockLlmServer::start().await.unwrap();
    server.add_response("hello", "Hello from mock server");

    let (status, body) = server
        .chat_completions(&request("mock-model", &["say hello"]))
        .await
        .unwrap();

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "mock-model");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from mock server"
    );

    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn mock_server_collapses_multi_message_requests_for_backend_routing() {
    let mut server = MockLlmServer::start().await.unwrap();
    server.add_response("second line", "Matched collapsed prompt");

    let payload = ChatCompletionRequest {
        model: "mock-model".into(),
        messages: vec![
            ChatCompletionMessage {
                role: "system".into(),
                content: "first line".into(),
            },
            ChatCompletionMessage {
                role: "user".into(),
                content: "second line".into(),
            },
        ],
    };

    let (status, body) = server.chat_completions(&payload).await.unwrap();

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        server.backend().infer_history(),
        vec!["first line\nsecond line".to_string()]
    );
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Matched collapsed prompt"
    );

    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn mock_server_records_request_history_and_backend_infer_history() {
    let mut server = MockLlmServer::start().await.unwrap();
    server.add_route(MockChatCompletionRoute::new("weather", "Sunny"));

    // Assert both layers: the OpenAI-shaped request and the collapsed backend prompt.
    let payload = request("mock-model", &["what is the weather?"]);
    server.chat_completions(&payload).await.unwrap();

    assert_eq!(server.request_history(), vec![payload]);
    assert_eq!(
        server.backend().infer_history(),
        vec!["what is the weather?".to_string()]
    );
    assert_eq!(server.backend().infer_count_for("weather"), 1);

    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn mock_server_supports_response_sequences_across_requests() {
    let mut server = MockLlmServer::start().await.unwrap();
    server.add_response_sequence("turn", vec!["first reply", "second reply"]);

    let (_, first) = server
        .chat_completions(&request("mock-model", &["turn one"]))
        .await
        .unwrap();
    let (_, second) = server
        .chat_completions(&request("mock-model", &["turn two"]))
        .await
        .unwrap();
    let (_, third) = server
        .chat_completions(&request("mock-model", &["turn three"]))
        .await
        .unwrap();

    assert_eq!(first["choices"][0]["message"]["content"], "first reply");
    assert_eq!(second["choices"][0]["message"]["content"], "second reply");
    assert_eq!(third["choices"][0]["message"]["content"], "second reply");

    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn mock_server_records_failed_requests_for_assertions() {
    let mut server = MockLlmServer::start().await.unwrap();
    server
        .backend()
        .fail_next(1, OrchestratorError::InferenceFailed("boom".into()));

    let (status, body) = server
        .chat_completions(&request("mock-model", &["please fail"]))
        .await
        .unwrap();

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("boom")
    );
    assert_eq!(
        server.backend().infer_history(),
        vec!["please fail".to_string()]
    );
    assert_eq!(server.request_history().len(), 1);

    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn mock_server_clear_request_history_resets_history() {
    let mut server = MockLlmServer::start().await.unwrap();
    server.add_response("ping", "pong");

    server
        .chat_completions(&request("mock-model", &["ping"]))
        .await
        .unwrap();

    assert_eq!(server.request_history().len(), 1);
    server.clear_request_history();
    assert!(server.request_history().is_empty());

    server.shutdown().await.unwrap();
}

#[tokio::test]
async fn mock_server_uses_fallback_when_no_route_matches() {
    let mut server = MockLlmServer::start().await.unwrap();
    server.set_fallback("Fallback reply");

    let (status, body) = server
        .chat_completions(&request("mock-model", &["no route here"]))
        .await
        .unwrap();

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["content"], "Fallback reply");

    server.shutdown().await.unwrap();
}
