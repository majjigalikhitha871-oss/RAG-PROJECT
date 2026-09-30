from unittest.mock import Mock

from core.ollama_client import detect_ollama_models, generate_answer, resolve_model


def test_qwen_is_preferred_over_llama():
    selected = resolve_model(["llama3.2:latest", "qwen2.5:3b"])
    assert selected.name == "qwen2.5:3b"
    assert selected.family == "Qwen"


def test_llama_is_fallback_when_qwen_is_missing():
    selected = resolve_model(["mistral:latest", "llama3.1:8b"])
    assert selected.name == "llama3.1:8b"
    assert selected.family == "Llama"


def test_no_supported_model_returns_none():
    assert resolve_model(["mistral:latest"]) is None


def test_detects_locally_installed_models(monkeypatch):
    response = Mock()
    response.json.return_value = {"models": [{"name": "qwen3:4b"}, {"name": "llama3.2:1b"}]}
    monkeypatch.setattr("core.ollama_client.requests.get", lambda *args, **kwargs: response)

    assert detect_ollama_models() == ["qwen3:4b", "llama3.2:1b"]


def test_generate_sends_grounded_context_and_question_as_user_data(monkeypatch):
    response = Mock()
    response.json.return_value = {"message": {"content": "The minimum age is 18."}}
    captured = {}

    def post(url, **kwargs):
        captured.update(kwargs["json"])
        return response

    monkeypatch.setattr("core.ollama_client.requests.post", post)
    answer = generate_answer("qwen2.5:3b", "Minimum age?", "[Source: rules.pdf] At least 18.")

    assert answer == "The minimum age is 18."
    assert captured["messages"][0]["role"] == "system"
    assert "untrusted reference data" in captured["messages"][0]["content"]
    assert captured["messages"][1]["role"] == "user"
    assert "Minimum age?" in captured["messages"][1]["content"]
    assert "[Source: rules.pdf] At least 18." in captured["messages"][1]["content"]