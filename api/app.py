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
        <title>InspectAI - Industrial Defect Detection</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { 
                margin: 0; 
                padding: 0; 
                box-sizing: border-box; 
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
                background: #f5f7fa;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
                color: #2d3748;
            }
            
            .container {
                background: white;
                border-radius: 16px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 10px 15px rgba(0, 0, 0, 0.1);
                padding: 48px;
                max-width: 720px;
                width: 100%;
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 1px solid #e2e8f0;
                padding-bottom: 24px;
            }
            
            h1 {
                color: #1a202c;
                margin-bottom: 8px;
                font-size: 28px;
                font-weight: 600;
                letter-spacing: -0.5px;
            }
            
            .subtitle {
                color: #718096;
                font-size: 14px;
                font-weight: 400;
            }
            
            .upload-area {
                border: 2px dashed #cbd5e0;
                border-radius: 12px;
                padding: 48px 24px;
                text-align: center;
                cursor: pointer;
                transition: all 0.2s ease;
                background: #f7fafc;
            }
            
            .upload-area:hover {
                background: #edf2f7;
                border-color: #4299e1;
            }
            
            .upload-icon {
                font-size: 48px;
                margin-bottom: 16px;
                color: #a0aec0;
            }
            
            .upload-text {
                color: #4a5568;
                font-size: 16px;
                margin-bottom: 8px;
                font-weight: 500;
            }
            
            .upload-hint {
                color: #a0aec0;
                font-size: 13px;
            }
            
            input[type="file"] {
                display: none;
            }
            
            .btn {
                background: #2b6cb0;
                color: white;
                border: none;
                padding: 12px 32px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
                margin-top: 24px;
                width: 100%;
            }
            
            .btn:hover:not(:disabled) {
                background: #2c5282;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(43, 108, 176, 0.4);
            }
            
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            #preview {
                margin: 24px 0;
                text-align: center;
            }
            
            #preview img {
                max-width: 100%;
                max-height: 360px;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }
            
            #result {
                margin-top: 32px;
                padding: 24px;
                border-radius: 12px;
                display: none;
                border: 2px solid;
            }
            
            .result-good {
                background: #f0fdf4;
                border-color: #22c55e;
            }
            
            .result-defect {
                background: #fef2f2;
                border-color: #ef4444;
            }
            
            .result-header {
                display: flex;
                align-items: center;
                margin-bottom: 16px;
            }
            
            .result-icon {
                font-size: 24px;
                margin-right: 12px;
            }
            
            .result-title {
                font-size: 20px;
                font-weight: 600;
                color: #1a202c;
            }
            
            .confidence {
                font-size: 15px;
                margin: 12px 0;
                color: #4a5568;
            }
            
            .confidence-value {
                font-weight: 600;
                font-size: 18px;
            }
            
            .probabilities {
                margin-top: 20px;
            }
            
            .prob-bar {
                margin: 16px 0;
            }
            
            .prob-label {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
                font-size: 14px;
                color: #4a5568;
            }
            
            .prob-label-name {
                font-weight: 500;
            }
            
            .prob-label-value {
                font-weight: 600;
                color: #2d3748;
            }
            
            .bar-container {
                background: #e2e8f0;
                height: 8px;
                border-radius: 4px;
                overflow: hidden;
            }
            
            .bar {
                height: 100%;
                border-radius: 4px;
                transition: width 0.6s ease;
            }
            
            .bar-good {
                background: #22c55e;
            }
            
            .bar-defect {
                background: #ef4444;
            }
            
            .loader {
                border: 3px solid #e2e8f0;
                border-top: 3px solid #2b6cb0;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 0.8s linear infinite;
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
            <div class="header">
                <h1>InspectAI</h1>
                <p class="subtitle">AI-Powered Industrial Defect Detection</p>
            </div>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📁</div>
                <p class="upload-text">Click to upload or drag & drop</p>
                <p class="upload-hint">Supported: JPG, PNG, BMP</p>
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
                    <div class="result-header">
                        <span class="result-icon">${isGood ? '✓' : '✕'}</span>
                        <span class="result-title">${data.class_name}</span>
                    </div>
                    <div class="confidence">
                        Confidence: <span class="confidence-value">${(data.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="probabilities">
                        <div class="prob-bar">
                            <div class="prob-label">
                                <span class="prob-label-name">Good</span>
                                <span class="prob-label-value">${(goodProb * 100).toFixed(1)}%</span>
                            </div>
                            <div class="bar-container">
                                <div class="bar bar-good" style="width: ${goodProb * 100}%;"></div>
                            </div>
                        </div>
                        <div class="prob-bar">
                            <div class="prob-label">
                                <span class="prob-label-name">Defect</span>
                                <span class="prob-label-value">${(defectProb * 100).toFixed(1)}%</span>
                            </div>
                            <div class="bar-container">
                                <div class="bar bar-defect" style="width: ${defectProb * 100}%;"></div>
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
