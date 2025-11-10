"""
FastAPI application for InspectAI defect detection
Real-time inference API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from PIL import Image
import io
import base64
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from inference import DefectPredictor


# Initialize FastAPI app
app = FastAPI(
    title="InspectAI - Defect Detection API",
    description="Industrial defect detection using deep learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global predictor instance
predictor = None


def load_model(model_path: str = "models/checkpoints/best_acc.pth"):
    """Load the trained model"""
    global predictor
    try:
        predictor = DefectPredictor(model_path, device='cuda')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using dummy predictor for demo...")
        predictor = None


# Response models
class PredictionResponse(BaseModel):
    """Response model for predictions"""
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_info: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    model_path = Path("models/checkpoints/best_acc.pth")
    if model_path.exists():
        load_model(str(model_path))
    else:
        print("No trained model found. Please train a model first.")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the home page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>InspectAI - Defect Detection</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
                max-width: 800px;
                width: 100%;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
                text-align: center;
            }
            .subtitle {
                color: #666;
                text-align: center;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                background: #f8f9ff;
            }
            .upload-area:hover {
                background: #f0f2ff;
                border-color: #764ba2;
            }
            .upload-icon {
                font-size: 4em;
                margin-bottom: 20px;
            }
            input[type="file"] {
                display: none;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 25px;
                font-size: 1.1em;
                cursor: pointer;
                transition: transform 0.2s;
                margin-top: 20px;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            }
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            #preview {
                margin: 20px 0;
                text-align: center;
            }
            #preview img {
                max-width: 100%;
                max-height: 400px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            #result {
                margin-top: 30px;
                padding: 25px;
                border-radius: 15px;
                display: none;
            }
            .result-good {
                background: #d4edda;
                border: 2px solid #28a745;
            }
            .result-defect {
                background: #f8d7da;
                border: 2px solid #dc3545;
            }
            .result-title {
                font-size: 1.8em;
                font-weight: bold;
                margin-bottom: 15px;
            }
            .confidence {
                font-size: 1.3em;
                margin: 10px 0;
            }
            .probabilities {
                margin-top: 15px;
                text-align: left;
            }
            .prob-bar {
                margin: 10px 0;
            }
            .prob-label {
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
            }
            .bar {
                height: 25px;
                border-radius: 12px;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                transition: width 0.5s;
            }
            .loader {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
                display: none;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 InspectAI</h1>
            <p class="subtitle">Industrial Defect Detection System</p>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📸</div>
                <h3>Click to upload or drag & drop</h3>
                <p>Supported formats: JPG, PNG, BMP</p>
            </div>
            
            <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
            
            <div id="preview"></div>
            
            <div style="text-align: center;">
                <button class="btn" id="predictBtn" onclick="predict()" disabled>
                    Analyze Image
                </button>
            </div>
            
            <div class="loader" id="loader"></div>
            
            <div id="result"></div>
        </div>

        <script>
            let selectedFile = null;

            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (file) {
                    selectedFile = file;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('preview').innerHTML = 
                            '<img src="' + e.target.result + '" alt="Preview">';
                        document.getElementById('predictBtn').disabled = false;
                        document.getElementById('result').style.display = 'none';
                    };
                    reader.readAsDataURL(file);
                }
            }

            async function predict() {
                if (!selectedFile) return;

                const formData = new FormData();
                formData.append('file', selectedFile);

                document.getElementById('predictBtn').disabled = true;
                document.getElementById('loader').style.display = 'block';
                document.getElementById('result').style.display = 'none';

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (response.ok) {
                        displayResult(data);
                    } else {
                        alert('Error: ' + data.detail);
                    }
                } catch (error) {
                    alert('Error making prediction: ' + error);
                } finally {
                    document.getElementById('predictBtn').disabled = false;
                    document.getElementById('loader').style.display = 'none';
                }
            }

            function displayResult(data) {
                const resultDiv = document.getElementById('result');
                const isGood = data.class_name === 'Good';
                
                resultDiv.className = isGood ? 'result-good' : 'result-defect';
                
                const goodProb = data.probabilities['Good'];
                const defectProb = data.probabilities['Defect'];
                
                resultDiv.innerHTML = `
                    <div class="result-title">${isGood ? '✅' : '❌'} ${data.class_name}</div>
                    <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</div>
                    <div class="probabilities">
                        <div class="prob-bar">
                            <div class="prob-label">
                                <span>Good</span>
                                <span>${(goodProb * 100).toFixed(1)}%</span>
                            </div>
                            <div style="background: #e0e0e0; border-radius: 12px;">
                                <div class="bar" style="width: ${goodProb * 100}%; background: #28a745;"></div>
                            </div>
                        </div>
                        <div class="prob-bar">
                            <div class="prob-label">
                                <span>Defect</span>
                                <span>${(defectProb * 100).toFixed(1)}%</span>
                            </div>
                            <div style="background: #e0e0e0; border-radius: 12px;">
                                <div class="bar" style="width: ${defectProb * 100}%; background: #dc3545;"></div>
                            </div>
                        </div>
                    </div>
                `;
                
                resultDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = predictor is not None
    model_info = predictor.get_model_info() if model_loaded else None

    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_info=model_info
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_defect(file: UploadFile = File(...)):
    """
    Predict defect from uploaded image

    Args:
        file: Image file

    Returns:
        Prediction results
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Make prediction
        result = predictor.predict(image)

        return PredictionResponse(
            class_id=result['class'],
            class_name=result['label'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            message="Prediction successful"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict defects from multiple images

    Args:
        files: List of image files

    Returns:
        List of prediction results
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    results = []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            result = predictor.predict(image)
            results.append({
                'filename': file.filename,
                **result
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })

    return JSONResponse(content={'predictions': results})


@app.get("/docs")
async def get_docs():
    """API documentation"""
    return {
        "endpoints": {
            "/": "Web interface",
            "/health": "Health check",
            "/predict": "Single image prediction",
            "/predict/batch": "Batch prediction",
            "/docs": "API documentation"
        },
        "usage": {
            "predict": "POST image file to /predict",
            "batch": "POST multiple image files to /predict/batch"
        }
    }


if __name__ == "__main__":
    # Run the API
    print("Starting InspectAI API...")
    print("Access the web interface at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
