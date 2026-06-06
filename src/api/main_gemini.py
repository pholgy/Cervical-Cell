import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError


load_dotenv()

CLASSES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]

CELL_DESCRIPTIONS = {
    "Dyskeratotic": "Abnormal keratin production, often associated with HPV infection",
    "Koilocytotic": "Cells showing HPV-related changes with perinuclear halos",
    "Metaplastic": "Cells undergoing transformation, often benign",
    "Parabasal": "Immature squamous cells from basal layers",
    "Superficial-Intermediate": "Mature squamous cells from upper layers",
}

SUPPORTED_UPLOAD_TYPES = {
    "application/octet-stream",
    "image/bmp",
    "image/gif",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/webp",
}


@dataclass(frozen=True)
class BackendSettings:
    api_key: str | None
    classification_model: str = "gemini-2.5-flash"
    explanation_model: str = "gemini-2.5-flash"

    @property
    def prediction_ready(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def from_env(cls) -> "BackendSettings":
        api_key = os.getenv("GEMINI_API_KEY")
        return cls(
            api_key=api_key.strip() if api_key and api_key.strip() else None,
            classification_model=os.getenv("GEMINI_CLASSIFICATION_MODEL", "gemini-2.5-flash"),
            explanation_model=os.getenv("GEMINI_EXPLANATION_MODEL", "gemini-2.5-flash"),
        )


class UpstreamResponseError(ValueError):
    """Gemini returned a syntactically valid response that failed this API's contract."""


class GeminiModelProvider:
    def __init__(self, current_settings: BackendSettings):
        self.current_settings = current_settings
        self._classification_model = None
        self._explanation_model = None

    def get_classification_model(self):
        if self._classification_model is None:
            self._classification_model = genai.GenerativeModel(
                self.current_settings.classification_model,
                generation_config={"response_mime_type": "application/json"},
            )
        return self._classification_model

    def get_explanation_model(self):
        if self._explanation_model is None:
            self._explanation_model = genai.GenerativeModel(self.current_settings.explanation_model)
        return self._explanation_model


settings = BackendSettings.from_env()
if settings.api_key:
    genai.configure(api_key=settings.api_key)

model_provider = GeminiModelProvider(settings)

app = FastAPI(title="Cervical Cell Classification API (Gemini Vision)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def reset_model_provider() -> None:
    global model_provider
    model_provider = GeminiModelProvider(settings)


def api_error(status_code: int, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"success": False, "error": message},
    )


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": str(exc.detail)},
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_exception(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Request validation failed.",
            "details": exc.errors(),
        },
    )


def validate_upload_type(content_type: str | None) -> None:
    if not content_type:
        return

    normalized = content_type.lower().split(";")[0].strip()
    if normalized in SUPPORTED_UPLOAD_TYPES or normalized.startswith("image/"):
        return

    raise api_error(400, "Unsupported file type. Upload an image file.")


def load_upload_image(file: UploadFile, image_bytes: bytes) -> Image.Image:
    validate_upload_type(file.content_type)

    if not image_bytes:
        raise api_error(400, "Invalid image file.")

    try:
        Image.open(io.BytesIO(image_bytes)).verify()
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image
    except (UnidentifiedImageError, OSError):
        raise api_error(400, "Invalid image file.")


def parse_percent(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise UpstreamResponseError(f"{field_name} must be numeric") from exc

    if parsed < 0 or parsed > 100:
        raise UpstreamResponseError(f"{field_name} must be between 0 and 100")

    return parsed


def validate_classification_payload(payload: Any) -> tuple[str, float, dict[str, float], str]:
    if not isinstance(payload, dict):
        raise UpstreamResponseError("classification response must be an object")

    missing_fields = {"classification", "confidence", "probabilities"} - payload.keys()
    if missing_fields:
        raise UpstreamResponseError(f"classification response missing fields: {', '.join(sorted(missing_fields))}")

    predicted_class = payload["classification"]
    if predicted_class not in CLASSES:
        raise UpstreamResponseError(f"unsupported classification: {predicted_class}")

    confidence_percent = parse_percent(payload["confidence"], "confidence")

    raw_probabilities = payload["probabilities"]
    if not isinstance(raw_probabilities, dict):
        raise UpstreamResponseError("probabilities must be an object")

    expected_classes = set(CLASSES)
    probability_classes = set(raw_probabilities.keys())
    if probability_classes != expected_classes:
        missing = expected_classes - probability_classes
        extra = probability_classes - expected_classes
        details = []
        if missing:
            details.append(f"missing probabilities for: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unexpected probabilities for: {', '.join(sorted(extra))}")
        raise UpstreamResponseError("; ".join(details))

    probabilities_percent = {
        class_name: parse_percent(raw_probabilities[class_name], f"probabilities.{class_name}")
        for class_name in CLASSES
    }
    probability_total = sum(probabilities_percent.values())
    if probability_total < 95 or probability_total > 105:
        raise UpstreamResponseError(f"probabilities must total approximately 100, got {probability_total:.1f}")

    reasoning = payload.get("reasoning", "")
    return (
        predicted_class,
        confidence_percent / 100.0,
        {class_name: probabilities_percent[class_name] / 100.0 for class_name in CLASSES},
        str(reasoning),
    )


def parse_classification_response(response_text: str) -> tuple[str, float, dict[str, float], str]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise UpstreamResponseError("Gemini returned invalid JSON") from exc

    return validate_classification_payload(payload)


def build_classification_prompt() -> str:
    return f"""You are an expert pathologist specializing in cervical cytology. Analyze this microscopy image and classify the cells you observe into ONE of these five categories:

1. **Dyskeratotic**: {CELL_DESCRIPTIONS["Dyskeratotic"]}
2. **Koilocytotic**: {CELL_DESCRIPTIONS["Koilocytotic"]}
3. **Metaplastic**: {CELL_DESCRIPTIONS["Metaplastic"]}
4. **Parabasal**: {CELL_DESCRIPTIONS["Parabasal"]}
5. **Superficial-Intermediate**: {CELL_DESCRIPTIONS["Superficial-Intermediate"]}

IMPORTANT: You MUST respond with ONLY valid JSON in this exact format (no extra text before or after):
{{
  "classification": "<one of the 5 cell types exactly as written above>",
  "confidence": <number between 0-100>,
  "probabilities": {{
    "Dyskeratotic": <0-100>,
    "Koilocytotic": <0-100>,
    "Metaplastic": <0-100>,
    "Parabasal": <0-100>,
    "Superficial-Intermediate": <0-100>
  }},
  "reasoning": "<brief 2-3 sentence explanation>"
}}

Base your classification on visible features:
- Cell morphology and shape
- Nuclear characteristics (size, color, structure)
- Cytoplasmic features (color, texture)
- Presence of halos or abnormal keratinization
- Cell maturity indicators

Even if the image is unclear or you're uncertain, you MUST provide your best classification with probability estimates. The probabilities should sum to approximately 100."""


def build_explanation_prompt(
    predicted_class: str,
    confidence: float,
    probabilities: dict[str, float],
    reasoning: str,
) -> str:
    formatted_probabilities = "\n".join(
        f"- {class_name}: {probability * 100:.1f}%" for class_name, probability in probabilities.items()
    )

    return f"""You are a medical AI assistant. Write a comprehensive medical explanation in MARKDOWN format (NOT JSON) for healthcare professionals.

Classification Result:
- Predicted Cell Type: {predicted_class}
- Confidence: {confidence * 100:.1f}%
- Initial Reasoning: {reasoning}

All Probabilities:
{formatted_probabilities}

Write a markdown-formatted explanation with these sections:

## Cell Type Explanation
Explain what {predicted_class} cells are (2-3 sentences).

## Clinical Significance
What this finding indicates and its clinical importance.

## Model Confidence
Why the model is confident ({confidence * 100:.1f}%) based on the probabilities and image features.

## Considerations & Recommendations
Important clinical points and recommendations for follow-up.

Keep it concise, professional, and actionable. Use proper markdown formatting with headers (##), bold (**text**), and lists. Max 200 words. DO NOT use JSON format - use plain markdown text."""


def normalize_explanation(text: str) -> str:
    explanation = text.strip()
    if not explanation.startswith("{"):
        return explanation

    try:
        json_explanation = json.loads(explanation)
    except json.JSONDecodeError:
        print("[WARN] Explanation looked like JSON but could not be parsed; returning raw explanation.")
        return explanation

    markdown_parts = []
    for key, value in json_explanation.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                markdown_parts.append(f"## {subkey.replace('_', ' ').title()}\n{subvalue}\n")
        else:
            markdown_parts.append(f"## {key.replace('_', ' ').title()}\n{value}\n")
    return "\n".join(markdown_parts)


def map_gemini_error(exc: Exception) -> HTTPException:
    message = str(exc)
    normalized = message.lower()

    if "429" in message or "resource exhausted" in normalized:
        return api_error(429, "Gemini API rate limit reached. Please wait a few moments and try again.")

    if "api" in normalized or "quota" in normalized or "gemini" in normalized:
        return api_error(502, f"Gemini API error: {message}")

    return api_error(500, f"Prediction failed: {message}")


@app.get("/")
async def root():
    return {
        "message": "Cervical Cell Classification API (Gemini Vision)",
        "status": "running",
        "prediction_ready": settings.prediction_ready,
    }


@app.get("/model-info")
async def model_info():
    return {
        "model_loaded": settings.prediction_ready,
        "prediction_ready": settings.prediction_ready,
        "model_name": settings.classification_model,
        "classification_model": settings.classification_model,
        "explanation_model": settings.explanation_model,
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "api_based": True,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not settings.prediction_ready:
        raise api_error(503, "Gemini API key is not configured.")

    start_time = time.time()
    image_bytes = await file.read()
    image = load_upload_image(file, image_bytes)

    try:
        print("[INFO] Classifying image with Gemini Vision...")
        classification_start = time.time()
        classification_response = model_provider.get_classification_model().generate_content(
            [build_classification_prompt(), image]
        )
        classification_time = time.time() - classification_start

        response_text = classification_response.text.strip()
        print(f"[DEBUG] Raw response: {response_text[:200]}...")

        predicted_class, confidence, probabilities, reasoning = parse_classification_response(response_text)
        print(f"[OK] Classification complete: {predicted_class} ({confidence * 100:.1f}%)")

        print("[INFO] Generating detailed medical explanation...")
        explanation_start = time.time()
        explanation_response = model_provider.get_explanation_model().generate_content(
            build_explanation_prompt(predicted_class, confidence, probabilities, reasoning)
        )
        explanation_time = time.time() - explanation_start
        ai_explanation = normalize_explanation(explanation_response.text)
        print(f"[OK] Explanation generated: {len(ai_explanation)} characters")
        print(
            "[INFO] Gemini timings: "
            f"classification={classification_time:.3f}s explanation={explanation_time:.3f}s"
        )
    except UpstreamResponseError as exc:
        raise api_error(502, f"Invalid Gemini classification response: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise map_gemini_error(exc) from exc

    return {
        "success": True,
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "processing_time": f"{time.time() - start_time:.3f}s",
        "model_name": settings.classification_model,
        "ai_explanation": ai_explanation,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
