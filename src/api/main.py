import os
import sys
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')

app = FastAPI(title="Cervical Cell Classification API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model - try improved model first, fallback to others
MODEL_PATHS = [
    os.path.join(config.MODEL_DIR, 'improved_best.h5'),
    os.path.join(config.MODEL_DIR, 'resnet50_100epochs.h5'),
    os.path.join(config.MODEL_DIR, 'best_model.h5'),
]
model = None
MODEL_NAME = None

@app.on_event("startup")
async def load_model_on_startup():
    global model, MODEL_NAME
    for model_path in MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                MODEL_NAME = os.path.basename(model_path)
                print(f"[OK] Model loaded successfully: {MODEL_NAME}")
                return
            except Exception as e:
                print(f"[FAIL] Failed to load {model_path}: {e}")
    print(f"Warning: No model found. Tried: {MODEL_PATHS}")

def preprocess_image(image_bytes):
    """Preprocess uploaded image"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

    # Normalize
    img = img.astype('float32') / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

@app.get("/")
async def root():
    return {"message": "Cervical Cell Classification API", "status": "running"}

@app.get("/model-info")
async def model_info():
    if model is None:
        return {"error": "Model not loaded"}

    return {
        "model_loaded": True,
        "model_name": MODEL_NAME,
        "classes": config.CLASSES,
        "num_classes": len(config.CLASSES),
        "input_size": config.IMG_SIZE
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        # Read image
        image_bytes = await file.read()

        # Preprocess
        start_time = time.time()
        img = preprocess_image(image_bytes)

        # Predict
        predictions = model.predict(img)[0]
        processing_time = time.time() - start_time

        # Get top prediction
        predicted_class_idx = int(np.argmax(predictions))
        predicted_class = config.CLASSES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])

        # All probabilities
        all_probabilities = {
            config.CLASSES[i]: float(predictions[i])
            for i in range(len(config.CLASSES))
        }

        # Generate AI explanation
        try:
            print("[INFO] Generating AI explanation with Gemini...")
            prompt = f"""You are a medical AI assistant explaining cervical cell classification results to healthcare professionals.

Classification Result:
- Predicted Cell Type: {predicted_class}
- Confidence: {confidence*100:.1f}%

All Probabilities:
{chr(10).join([f'- {cls}: {prob*100:.1f}%' for cls, prob in all_probabilities.items()])}

Cell Type Descriptions:
- Dyskeratotic: Abnormal keratin production, often associated with HPV infection
- Koilocytotic: Cells showing HPV-related changes with perinuclear halos
- Metaplastic: Cells undergoing transformation, often benign
- Parabasal: Immature squamous cells from basal layers
- Superficial-Intermediate: Mature squamous cells from upper layers

Please provide:
1. A brief explanation of what this cell type means (2-3 sentences)
2. Clinical significance and what it might indicate
3. Why the model is confident in this prediction based on the probabilities
4. Any important considerations or recommendations

Keep it concise, professional, and actionable. Max 150 words."""

            response = gemini_model.generate_content(prompt)
            ai_explanation = response.text
            print(f"[OK] AI explanation generated: {len(ai_explanation)} characters")
        except Exception as e:
            ai_explanation = f"AI explanation unavailable: {str(e)}"
            print(f"[ERROR] Failed to generate AI explanation: {str(e)}")

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": all_probabilities,
            "processing_time": f"{processing_time:.3f}s",
            "model_name": MODEL_NAME,
            "ai_explanation": ai_explanation
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
