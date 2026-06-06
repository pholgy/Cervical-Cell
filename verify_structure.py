"""Verify the repository shape without importing heavyweight training dependencies."""
import os


print("=" * 60)
print("Project Structure Verification")
print("=" * 60)

required_files = [
    "config.py",
    "requirements.txt",
    "requirements-dev.txt",
    "src/api/main_gemini.py",
    "api/index.py",
    "tests/test_main_gemini.py",
    "README.md",
    "AGENTS.md",
]

optional_training_paths = [
    "src/data_loader.py",
    "src/train_model.py",
    "src/api/main.py",
    "requirements-training.txt",
]

gitignored_runtime_assets = [
    "data",
    "models",
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate",
]

print("\nChecking required runtime/test files:")
all_required_exist = True
for path in required_files:
    exists = os.path.exists(path)
    status = "[OK]" if exists else "[MISS]"
    print(f"  {status} {path}")
    if not exists:
        all_required_exist = False

print("\nChecking optional training files:")
for path in optional_training_paths:
    status = "[OK]" if os.path.exists(path) else "[MISS]"
    print(f"  {status} {path}")

print("\nChecking local-only data/model assets:")
for path in gitignored_runtime_assets:
    status = "[LOCAL]" if os.path.exists(path) else "[NOT PRESENT]"
    print(f"  {status} {path}/")

print("\n" + "=" * 60)
if all_required_exist:
    print("SUCCESS: Required runtime/test structure is present.")
else:
    print("ERROR: Required runtime/test files are missing.")

print("\nDefault backend workflow:")
print("1. pip install -r requirements-dev.txt")
print("2. python -m pytest tests -q")
print("3. uvicorn src.api.main_gemini:app --reload")
print("\nTraining workflow, only when local datasets/models are available:")
print("1. pip install -r requirements-training.txt")
print("2. python src/train_model.py")
print("=" * 60)
