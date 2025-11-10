# InspectAI - Complete Usage Guide

## 🚀 Getting Started (Windows)

### Step 1: Environment Setup

```bash
# Open Anaconda Prompt or Command Prompt

# Navigate to project directory
cd path\to\InspectAI

# Run quick start script (recommended)
quick_start.bat

# OR manually:
conda create -n InspectAI python=3.11
conda activate InspectAI
pip install -r requirements.txt
python setup.py
```

### Step 2: Prepare Your Data

#### Option A: Use Sample Dataset (For Testing)

```bash
conda activate InspectAI
python src/prepare_data.py --action sample --samples 100
```

This creates a small synthetic dataset in `data/` directory for quick testing.

#### Option B: Use Your Own Dataset

1. Organize your images:
```
my_dataset/
├── good/       # Non-defective images
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── defect/     # Defective images
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

2. Split into train/val/test:
```bash
python src/prepare_data.py --action split --source my_dataset/ --output data/
```

#### Option C: Download Public Datasets

**MVTec AD Dataset** (Recommended):
1. Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Download a category (e.g., "metal_nut")
3. Organize into good/defect folders
4. Run split command

**Kaggle Casting Defects**:
1. Visit: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
2. Download dataset
3. Organize and split

### Step 3: Train the Model

```bash
conda activate InspectAI
python src/train.py
```

**What happens during training:**
- Loads data from `data/train` and `data/val`
- Trains for 50 epochs (configurable)
- Saves checkpoints to `models/experiment_YYYYMMDD_HHMMSS/`
- Creates training curves and confusion matrix
- Evaluates on test set

**Training typically takes:**
- With GPU (GTX 1660 Ti): ~5-10 minutes for 50 epochs (sample dataset)
- With CPU: ~30-60 minutes

**Monitor training:**
```bash
# View latest experiment
cd models
dir /od
cd experiment_<latest>
type training.log  # View training log
```

### Step 4: Run the API

```bash
conda activate InspectAI
cd api
python app.py
```

**Access the application:**
- Web UI: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Step 5: Make Predictions

#### Using Web Interface
1. Open http://localhost:8000
2. Click upload area or drag & drop image
3. Click "Analyze Image"
4. View results

#### Using Python Script
```bash
python src/inference.py path\to\image.jpg models\checkpoints\best_acc.pth
```

#### Using API (Python)
```python
import requests

url = "http://localhost:8000/predict"
files = {'file': open('test_image.jpg', 'rb')}
response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### Using API (curl)
```bash
curl -X POST "http://localhost:8000/predict" ^
     -H "accept: application/json" ^
     -H "Content-Type: multipart/form-data" ^
     -F "file=@test_image.jpg"
```

## ⚙️ Configuration

### Training Configuration

Edit `config.py` or directly in `src/train.py`:

```python
config = {
    'model_type': 'full',      # or 'light' for faster inference
    'epochs': 50,              # Number of training epochs
    'batch_size': 32,          # Batch size (reduce if out of memory)
    'learning_rate': 0.001,    # Learning rate
    'optimizer': 'adamw',      # Optimizer choice
    'scheduler': 'cosine',     # Learning rate scheduler
    'early_stop_patience': 10, # Early stopping patience
}
```

### Model Selection

**Full Model** (Recommended for accuracy):
- 11M parameters
- Better accuracy (~94%)
- Slower inference (~8ms)

**Light Model** (For speed):
- 2M parameters  
- Good accuracy (~92%)
- Faster inference (~3ms)

Change in config:
```python
config['model_type'] = 'light'
```

## 🔧 Troubleshooting

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size in config
```python
config['batch_size'] = 16  # or even 8
```

### No GPU Detected

**Warning**: `CUDA not available. Training will use CPU`

**Check**:
1. Is NVIDIA GPU installed?
2. Is CUDA installed? `nvidia-smi` should work
3. Is PyTorch with CUDA installed?
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Fix** (if False):
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
```bash
conda activate InspectAI
pip install -r requirements.txt
```

### API Won't Start

**Error**: `Address already in use`

**Solution**: Change port in `api/app.py`:
```python
uvicorn.run("app:app", host="0.0.0.0", port=8001)  # Changed to 8001
```

### Model File Not Found

**Error**: `FileNotFoundError: models/checkpoints/best_acc.pth`

**Solution**: Train a model first
```bash
python src/train.py
```

## 📊 Understanding Results

### Training Output

After training, check `models/experiment_YYYYMMDD_HHMMSS/`:

1. **training_curves.png**: Visualize training progress
   - Look for: Loss decreasing, accuracy increasing
   - Red flag: Huge gap between train and val (overfitting)

2. **confusion_matrix.png**: See classification errors
   - Diagonal = correct predictions
   - Off-diagonal = errors

3. **test_results.json**: Detailed metrics
```json
{
  "accuracy": 0.94,      // Overall accuracy
  "precision": 0.93,     // True defects / Predicted defects
  "recall": 0.95,        // True defects / Actual defects
  "f1_score": 0.94       // Harmonic mean of precision & recall
}
```

### Interpreting Predictions

```json
{
  "class_name": "Defect",
  "confidence": 0.95,
  "probabilities": {
    "Good": 0.05,
    "Defect": 0.95
  }
}
```

- **Confidence > 0.9**: Very confident
- **Confidence 0.7-0.9**: Confident
- **Confidence 0.5-0.7**: Uncertain (review manually)
- **Confidence < 0.5**: Low confidence (classified as opposite)

## 🎯 Best Practices

### Data Preparation

1. **Balanced dataset**: Equal or similar numbers of good/defect samples
2. **High quality**: Clear, well-lit images
3. **Consistent**: Same angle, lighting, resolution
4. **Diverse defects**: Include various defect types
5. **Clean labels**: No mislabeled images

### Training

1. **Start with sample dataset** to verify setup
2. **Monitor overfitting**: Val loss should track train loss
3. **Use early stopping**: Prevents wasting time
4. **Save checkpoints**: Don't lose progress
5. **Try different configs**: Learning rate, batch size, etc.

### Deployment

1. **Test thoroughly** before production
2. **Set confidence threshold** appropriate for your use case
3. **Monitor performance** in production
4. **Retrain periodically** with new data
5. **Version control** your models

## 📈 Performance Optimization

### Training Faster

```python
# Use light model
config['model_type'] = 'light'

# Increase batch size (if GPU allows)
config['batch_size'] = 64

# Fewer epochs (if convergence is fast)
config['epochs'] = 30

# Reduce image size
config['img_size'] = 128  # Instead of 224
```

### Inference Faster

```python
# Use light model
model_type = 'light'

# Batch processing
results = predictor.predict_batch(images)

# Lower resolution
img_size = 128
```

## 🔄 Retraining

When to retrain:
- New defect types appear
- Performance degrades
- Data distribution changes
- Want to improve accuracy

How to retrain:
1. Add new images to `data/train/`
2. Run: `python src/train.py`
3. Compare new model with old
4. Deploy if better

## 📦 Deployment Options

### Local Server
```bash
python api/app.py
```

### Docker
```bash
docker build -t inspectai .
docker run -p 8000:8000 -v %CD%/models:/app/models inspectai
```

### Cloud Deployment
- AWS: EC2 + Docker
- Google Cloud: GCE + Docker  
- Azure: VM + Docker
- Heroku: Using Dockerfile

## 📞 Getting Help

If you encounter issues:

1. **Check this guide** first
2. **Check README.md** for general info
3. **Look at error message** carefully
4. **Search GitHub issues** for similar problems
5. **Check requirements** are all installed

## 🎓 Next Steps

Once comfortable with basics:

1. **Experiment with hyperparameters**
2. **Try different architectures**
3. **Implement multi-class classification**
4. **Add object detection** for localization
5. **Optimize for edge deployment**
6. **Set up MLOps pipeline**

---

**Good luck with your defect detection system!**

For detailed technical information, see README.md
