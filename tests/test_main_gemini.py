import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.api.main_gemini as api


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_content(self, payload):
        self.calls += 1
        response = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)


class FakeModelFactory:
    def __init__(self, classification_response=None, explanation_response="Explanation text"):
        self.created = []
        self.classification_model = FakeModel([classification_response or valid_classification_json()])
        self.explanation_model = FakeModel([explanation_response])

    def __call__(self, model_name, generation_config=None):
        self.created.append((model_name, generation_config))
        if generation_config:
            return self.classification_model
        return self.explanation_model


def valid_classification_json():
    return json.dumps(
        {
            "classification": "Parabasal",
            "confidence": 96,
            "probabilities": {
                "Dyskeratotic": 1,
                "Koilocytotic": 1,
                "Metaplastic": 1,
                "Parabasal": 96,
                "Superficial-Intermediate": 1,
            },
            "reasoning": "The cell morphology best matches parabasal cells.",
        }
    )


@pytest.fixture(autouse=True)
def configured_backend(monkeypatch):
    monkeypatch.setattr(
        api,
        "settings",
        api.BackendSettings(
            api_key="test-key",
            classification_model="gemini-2.5-flash",
            explanation_model="gemini-2.5-flash",
        ),
    )
    api.reset_model_provider()
    yield
    api.reset_model_provider()


@pytest.fixture
def client():
    return TestClient(api.app)


def test_no_file_uses_predictable_error_body(client):
    response = client.post("/predict")

    assert response.status_code == 422
    assert response.json()["success"] is False
    assert "error" in response.json()


def test_predict_rejects_unsupported_content_type(client):
    response = client.post(
        "/predict",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "success": False,
        "error": "Unsupported file type. Upload an image file.",
    }


def test_predict_rejects_invalid_image_bytes(client):
    response = client.post(
        "/predict",
        files={"file": ("bad.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "success": False,
        "error": "Invalid image file.",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"classification": "NotAClass", "confidence": 90, "probabilities": {}},
        {"classification": "Parabasal", "probabilities": {}},
        {
            "classification": "Parabasal",
            "confidence": "high",
            "probabilities": {
                "Dyskeratotic": 1,
                "Koilocytotic": 1,
                "Metaplastic": 1,
                "Parabasal": 96,
                "Superficial-Intermediate": 1,
            },
        },
        {
            "classification": "Parabasal",
            "confidence": 250,
            "probabilities": {
                "Dyskeratotic": 1,
                "Koilocytotic": 1,
                "Metaplastic": 1,
                "Parabasal": 96,
                "Superficial-Intermediate": 1,
            },
        },
    ],
)
def test_validate_classification_payload_rejects_invalid_payloads(payload):
    with pytest.raises(api.UpstreamResponseError):
        api.validate_classification_payload(payload)


def test_predict_rejects_invalid_gemini_payload(client, monkeypatch):
    factory = FakeModelFactory(
        classification_response=json.dumps(
            {
                "classification": "NotAClass",
                "confidence": 250,
                "probabilities": {"NotAClass": 250},
                "reasoning": "invalid",
            }
        )
    )
    monkeypatch.setattr(api.genai, "GenerativeModel", factory)

    response = client.post(
        "/predict",
        files={"file": ("cell.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 502
    assert response.json()["success"] is False
    assert "Invalid Gemini classification response" in response.json()["error"]


def test_model_info_reflects_missing_gemini_config(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "settings",
        api.BackendSettings(
            api_key=None,
            classification_model="gemini-2.5-flash",
            explanation_model="gemini-2.5-flash",
        ),
    )
    api.reset_model_provider()

    model_info = client.get("/model-info")
    prediction = client.post(
        "/predict",
        files={"file": ("cell.png", png_bytes(), "image/png")},
    )

    assert model_info.status_code == 200
    assert model_info.json()["model_loaded"] is False
    assert model_info.json()["prediction_ready"] is False
    assert model_info.json()["classification_model"] == "gemini-2.5-flash"
    assert prediction.status_code == 503
    assert prediction.json() == {
        "success": False,
        "error": "Gemini API key is not configured.",
    }


def test_predict_maps_rate_limit_to_429(client, monkeypatch):
    factory = FakeModelFactory(classification_response=Exception("429 Resource exhausted"))
    monkeypatch.setattr(api.genai, "GenerativeModel", factory)

    response = client.post(
        "/predict",
        files={"file": ("cell.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 429
    assert response.json() == {
        "success": False,
        "error": "Gemini API rate limit reached. Please wait a few moments and try again.",
    }


def test_predict_reuses_gemini_model_clients(client, monkeypatch):
    factory = FakeModelFactory()
    monkeypatch.setattr(api.genai, "GenerativeModel", factory)

    first = client.post("/predict", files={"file": ("cell.png", png_bytes(), "image/png")})
    second = client.post("/predict", files={"file": ("cell.png", png_bytes(), "image/png")})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["prediction"] == "Parabasal"
    assert second.json()["prediction"] == "Parabasal"
    assert len(factory.created) == 2
    assert factory.classification_model.calls == 2
    assert factory.explanation_model.calls == 2
