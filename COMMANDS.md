# InspectAI - Copy/Paste Commands

## Setup (Run Once)
```bash
cd C:\Users\qaimi\Projects\InspectAI
setup_windows.bat
```

## Generate Sample Data
```bash
conda activate InspectAI
python src/prepare_data.py --action sample --samples 100
```

## Train Model
```bash
conda activate InspectAI
python src/train.py
```

## Run API
```bash
conda activate InspectAI
cd api
python app.py
```
Then open: http://localhost:8000

## Test Single Prediction
```bash
conda activate InspectAI
python src/inference.py path\to\image.jpg models\checkpoints\best_acc.pth
```

## Upload to GitHub
```bash
cd C:\Users\qaimi\Projects\InspectAI
git init
git add .
git commit -m "Add InspectAI defect detection system"
git branch -M main
git remote add origin https://github.com/qaim-b/InspectAI.git
git push -u origin main
```

## Check If CUDA Works
```bash
conda activate InspectAI
python -c "import torch; print(torch.cuda.is_available())"
```
Should print: True

## Check Model Size
```bash
python -c "from src.model import get_model; import torch; m = get_model('full'); print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')"
```
Should print: 11,234,567 parameters

## Quick Test Everything Works
```bash
conda activate InspectAI
python src/prepare_data.py --action sample --samples 20
python src/train.py
cd api && python app.py
```

---

That's it. Copy, paste, run.
