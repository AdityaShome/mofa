//! Focused coverage for prompt history tracking on `MockLLMBackend`.

use mofa_foundation::orchestrator::{
    ModelOrchestrator, ModelProviderConfig, ModelType, OrchestratorError,
};
use mofa_testing::backend::MockLLMBackend;
use std::collections::HashMap;

// Keep model setup tiny so the tests stay focused on history behavior.
fn make_config(name: &str) -> ModelProviderConfig {
    ModelProviderConfig {
        model_name: name.into(),
        model_path: "/mock".into(),
        device: "cpu".into(),
        model_type: ModelType::Llm,
        max_context_length: None,
        quantization: None,
        extra_config: HashMap::new(),
    }
}

// Reuse one ready-to-infer backend shape across all infer-history tests.
async fn ready_backend() -> MockLLMBackend {
    let backend = MockLLMBackend::new();
    backend.register_model(make_config("m")).await.unwrap();
    backend.load_model("m").await.unwrap();
    backend
}

#[tokio::test]
async fn infer_history_preserves_call_order() {
    let backend = ready_backend().await;

    let _ = backend.infer("m", "first prompt").await;
    let _ = backend.infer("m", "second prompt").await;

    assert_eq!(
        backend.infer_history(),
        vec!["first prompt".to_string(), "second prompt".to_string()]
    );
}

#[tokio::test]
async fn infer_history_records_failed_calls() {
    let backend = ready_backend().await;
    backend.fail_next(1, OrchestratorError::InferenceFailed("boom".into()));

    // History is recorded before failure injection is evaluated.
    let _ = backend.infer("m", "will fail").await;

    assert_eq!(backend.infer_history(), vec!["will fail".to_string()]);
}

#[tokio::test]
async fn infer_count_for_counts_matching_prompts() {
    let backend = ready_backend().await;

    let _ = backend.infer("m", "alpha request").await;
    let _ = backend.infer("m", "beta request").await;
    let _ = backend.infer("m", "alpha second").await;

    assert_eq!(backend.infer_count_for("alpha"), 2);
    assert_eq!(backend.infer_count_for("beta"), 1);
    assert_eq!(backend.infer_count_for("missing"), 0);
}

#[tokio::test]
async fn infer_count_for_allows_overlapping_matches() {
    let backend = ready_backend().await;

    let _ = backend.infer("m", "hello world").await;
    let _ = backend.infer("m", "hello there").await;
    let _ = backend.infer("m", "world only").await;

    assert_eq!(backend.infer_count_for("hello"), 2);
    assert_eq!(backend.infer_count_for("world"), 2);
    assert_eq!(backend.infer_count_for("hello world"), 1);
}

#[tokio::test]
async fn clear_infer_history_resets_recorded_prompts() {
    let backend = ready_backend().await;

    let _ = backend.infer("m", "first").await;
    let _ = backend.infer("m", "second").await;
    backend.clear_infer_history();

    assert!(backend.infer_history().is_empty());
    assert_eq!(backend.infer_count_for("first"), 0);
}
