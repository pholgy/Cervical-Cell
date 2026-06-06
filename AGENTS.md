# Cervical Cell

AI-assisted cervical cell classification app with a FastAPI/Gemini backend path and a Next.js frontend.

## Stack
- Python 3.12, FastAPI, Uvicorn, Gemini Python SDK, Pillow.
- Next.js 16, React 19, TypeScript, Tailwind CSS, Gemini JS SDK.
- Large datasets, `.bmp` images, `.h5` model files, and `.env` files are intentionally gitignored.

## Setup
- Backend runtime: `python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt`
- Backend tests: `pip install -r requirements-dev.txt`
- Training/runtime legacy path: `pip install -r requirements-training.txt`
- Frontend: `npm --prefix frontend ci`
- Required secret for real predictions: `GEMINI_API_KEY`

## Run
- Backend API: `uvicorn src.api.main_gemini:app --reload --host 0.0.0.0 --port 8000`
- Frontend app: `npm --prefix frontend run dev`
- Vercel Python entrypoint: `api/index.py` imports `src.api.main_gemini:app`

## Verify
- Full local check: `.\scripts\check.ps1`
- First run on a clean checkout: `.\scripts\check.ps1 -Install`
- The check compiles Python files, runs backend tests, runs the lightweight structure verifier, and builds the frontend.

## Code Map
| Need | Look Here |
| --- | --- |
| Python Gemini API | `src/api/main_gemini.py` |
| Vercel API adapter | `api/index.py` |
| Legacy TensorFlow API | `src/api/main.py` |
| Data/model constants | `config.py` |
| Dataset loading | `src/data_loader.py` |
| Training scripts | `src/train_model.py`, `train_*.py`, `quick_train.py` |
| Frontend pages | `frontend/app/page.tsx`, `frontend/app/upload/page.tsx`, `frontend/app/results/page.tsx` |
| Frontend prediction route | `frontend/app/api/predict/route.ts` |
| Frontend patient route | `frontend/app/api/patients/route.ts` |

## Hard Rules
- Do not commit `GEMINI_API_KEY`, `.env*`, datasets, `.bmp` images, `.h5` model files, or generated build output.
- Treat `src/api/main_gemini.py` as the deployed Python API unless the task explicitly targets the TensorFlow path.
- Do not add TensorFlow/OpenCV-only verification to the default check unless the dependencies and model/data files are part of the task.
- Keep TensorFlow/OpenCV/scikit-learn dependencies in `requirements-training.txt`, not the default deployed runtime file.
- Keep backend and frontend prediction response shapes aligned when editing either prediction path.
- For medical-facing text, preserve uncertainty and avoid unsupported diagnostic claims or invented performance metrics.
- Use PowerShell-native commands in this Windows workspace.

## Session Notes
- Before editing, check `git status --short --branch`.
- Keep changes scoped to the requested feature or fix.
- After edits, run `.\scripts\check.ps1` or explain exactly which part could not run and why.
