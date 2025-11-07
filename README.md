# Cervical Cell Classification Project

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Test Data Loading
```bash
python test_data_loading.py
```

### 2. Train Model
```bash
python src/train_model.py
```

### 3. Run API Server
```bash
python src/api/main.py
```
or
```bash
uvicorn src.api.main:app --reload
```

### 4. Test API
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.bmp"
```

## Project Structure
```
peakily-ml-proj/
├── data/                   # Dataset images
├── src/
│   ├── data_loader.py      # Data loading & preprocessing
│   ├── train_model.py      # Model training
│   └── api/
│       └── main.py         # FastAPI server
├── models/
│   └── best_model.h5       # Trained model
├── config.py               # Configuration
├── requirements.txt        # Dependencies
└── test_data_loading.py    # Test script
```

## API Endpoints

- `GET /` - Health check
- `GET /model-info` - Model information
- `POST /predict` - Upload image for prediction
