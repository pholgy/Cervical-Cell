"""Verify project structure without dependencies"""
import os

print("=" * 60)
print("Project Structure Verification")
print("=" * 60)

files_to_check = [
    'config.py',
    'requirements.txt',
    'src/data_loader.py',
    'src/train_model.py',
    'src/api/main.py',
    'test_data_loading.py',
    'README.md',
    'PROJECT_DOCUMENTATION.md'
]

print("\nChecking files:")
all_exist = True
for file in files_to_check:
    exists = os.path.exists(file)
    status = "[OK]" if exists else "[MISS]"
    print(f"  {status} {file}")
    if not exists:
        all_exist = False

print("\nChecking directories:")
dirs_to_check = ['data', 'src', 'src/api', 'models']
for dir in dirs_to_check:
    exists = os.path.exists(dir)
    status = "[OK]" if exists else "[MISS]"
    print(f"  {status} {dir}/")

print("\n" + "=" * 60)
if all_exist:
    print("SUCCESS: All files created!")
else:
    print("WARNING: Some files missing")

print("\nTo use the project:")
print("1. pip install -r requirements.txt")
print("2. python test_data_loading.py")
print("3. python src/train_model.py")
print("4. python src/api/main.py")
print("=" * 60)
