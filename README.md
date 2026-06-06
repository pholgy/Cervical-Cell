# Cervical Cell Classification Project

AI-assisted cervical cell classification with a FastAPI/Gemini backend and a Next.js frontend.

## Backend Setup

Install only the deployed Gemini/FastAPI runtime:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install backend test tooling:

```powershell
pip install -r requirements-dev.txt
```

Run the deployed backend:

```powershell
uvicorn src.api.main_gemini:app --reload --host 0.0.0.0 --port 8000
```

Required environment variable for real predictions:

```powershell
$env:GEMINI_API_KEY = "<your key>"
```

Optional model overrides:

```powershell
$env:GEMINI_CLASSIFICATION_MODEL = "gemini-2.5-flash"
$env:GEMINI_EXPLANATION_MODEL = "gemini-2.5-flash"
```

## Frontend Setup

```powershell
npm --prefix frontend ci
npm --prefix frontend run dev
```

## Verify

Full local check:

```powershell
.\scripts\check.ps1
```

First run from a clean checkout:

```powershell
.\scripts\check.ps1 -Install
```

Backend-only tests:

```powershell
python -m pytest tests -q
```

## API Endpoints

- `GET /` - app health and prediction readiness
- `GET /model-info` - configured Gemini models and class metadata
- `POST /predict` - upload an image for classification

Example:

```powershell
curl.exe -X POST "http://localhost:8000/predict" -F "file=@path\to\image.bmp"
```

## Training and Legacy TensorFlow Path

The default runtime dependencies intentionally do not include TensorFlow, OpenCV, NumPy, scikit-learn, or matplotlib. Install the training dependency set only when working on dataset loading, local model training, or the legacy TensorFlow API:

```powershell
pip install -r requirements-training.txt
```

Training and legacy paths also require local dataset folders and model files, which are intentionally gitignored:

- `im_Dyskeratotic/`
- `im_Koilocytotic/`
- `im_Metaplastic/`
- `im_Parabasal/`
- `im_Superficial-Intermediate/`
- `models/*.h5`

Legacy TensorFlow API:

```powershell
uvicorn src.api.main:app --reload
```

Training:

```powershell
python src/train_model.py
```

## Project Structure

```text
Cervical-Cell/
|-- api/index.py                  # Vercel adapter for FastAPI app
|-- src/api/main_gemini.py        # Deployed Gemini Vision backend
|-- src/api/main.py               # Legacy TensorFlow backend
|-- src/data_loader.py            # Dataset loading for training
|-- src/train_model.py            # Base model training script
|-- train_*.py                    # Additional training scripts
|-- frontend/app/                 # Next.js app routes and UI
|-- tests/test_main_gemini.py     # Backend API contract tests
|-- requirements.txt              # Deployed backend runtime dependencies
|-- requirements-dev.txt          # Backend test dependencies
|-- requirements-training.txt     # Training and legacy TensorFlow dependencies
`-- scripts/check.ps1             # Full local verification
```
