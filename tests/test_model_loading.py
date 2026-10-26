from types import SimpleNamespace

import pytest

from src.model import inference_transformer


def test_load_model_reports_classifier_repository_failure(monkeypatch, caplog):
    class Tokenizer:
        @staticmethod
        def from_pretrained(_model_name):
            return object()

    class Classifier:
        @staticmethod
        def from_pretrained(_model_dir):
            hub_error_type = type("HfHubHTTPError", (RuntimeError,), {})
            hub_error = hub_error_type(
                "401 Unauthorized at "
                "https://huggingface.co/model?access_token=local-secret "
                "Authorization: Bearer hf_abcdefghijklmnop"
            )
            hub_error.response = SimpleNamespace(
                status_code=401,
                url="https://huggingface.co/api/model?token=local-secret",
            )
            raise OSError("not a valid model identifier") from hub_error

    monkeypatch.setattr(inference_transformer, "AutoTokenizer", Tokenizer)
    monkeypatch.setattr(
        inference_transformer,
        "AutoModelForSequenceClassification",
        Classifier,
    )

    with pytest.raises(RuntimeError, match="trained clause classifier") as error:
        inference_transformer.load_model()

    assert inference_transformer.MODEL_DIR in str(error.value)
    assert "HF_TOKEN" in str(error.value)
    assert "No untrained or substitute model" in str(error.value)
    assert isinstance(error.value.__cause__, OSError)
    assert "http_status=401" in caplog.text
    assert "exception_type=OSError" in caplog.text
    assert "underlying_exception_type=HfHubHTTPError" in caplog.text
    assert "response_url=https://huggingface.co/api/model?<redacted>" in caplog.text
    assert "local-secret" not in caplog.text
    assert "hf_abcdefghijklmnop" not in caplog.text
    assert "Authorization=<redacted>" in caplog.text


def test_load_model_reports_tokenizer_repository_failure(monkeypatch):
    class Tokenizer:
        @staticmethod
        def from_pretrained(_model_name):
            raise OSError("Repository not found")

    monkeypatch.setattr(inference_transformer, "AutoTokenizer", Tokenizer)

    with pytest.raises(RuntimeError, match="Legal-RoBERTa tokenizer") as error:
        inference_transformer.load_model()

    assert inference_transformer.MODEL_NAME in str(error.value)
    assert isinstance(error.value.__cause__, OSError)
