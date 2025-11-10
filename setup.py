"""
Setup script for InspectAI
Handles installation and initial configuration
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a shell command and print status"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    dirs = [
        'data/train/good',
        'data/train/defect',
        'data/val/good',
        'data/val/defect',
        'data/test/good',
        'data/test/defect',
        'data/raw',
        'data/processed',
        'data/samples',
        'models/checkpoints',
        'static',
        'templates',
        'notebooks',
        'tests',
        'docs',
        'logs'
    ]
    
    print("\n" + "="*60)
    print("📁 Creating directory structure")
    print("="*60)
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    print("✅ Directory structure created successfully")


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"\n🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python 3.11 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True


def check_cuda():
    """Check CUDA availability"""
    print("\n" + "="*60)
    print("🎮 Checking CUDA availability")
    print("="*60)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✅ CUDA is available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Version: {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        else:
            print("⚠️  CUDA not available. Training will use CPU (slower)")
        
        return True
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False


def install_dependencies():
    """Install Python dependencies"""
    if not run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    ):
        return False
    
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies"
    ):
        return False
    
    return True


def verify_installation():
    """Verify that all packages are installed correctly"""
    print("\n" + "="*60)
    print("🔍 Verifying installation")
    print("="*60)
    
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'pillow',
        'sklearn',
        'matplotlib',
        'seaborn',
        'fastapi',
        'uvicorn',
        'albumentations',
        'tqdm',
        'pydantic'
    ]
    
    all_installed = True
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            all_installed = False
    
    if all_installed:
        print("\n✅ All packages installed successfully")
    else:
        print("\n❌ Some packages failed to install")
    
    return all_installed


def create_sample_data():
    """Offer to create sample dataset"""
    print("\n" + "="*60)
    print("📊 Sample Dataset")
    print("="*60)
    print("Would you like to create a sample dataset for testing?")
    print("This will generate synthetic images for quick testing.")
    
    response = input("Create sample dataset? (y/n): ").lower().strip()
    
    if response == 'y':
        return run_command(
            f"{sys.executable} src/prepare_data.py --action sample --samples 50",
            "Creating sample dataset"
        )
    else:
        print("Skipping sample dataset creation")
        print("You can create it later with: python src/prepare_data.py --action sample")
        return True


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    
    print("\n📝 Next Steps:")
    print("\n1. Prepare your dataset:")
    print("   - Organize images into good/ and defect/ folders")
    print("   - Run: python src/prepare_data.py --action split --source your_data/")
    print("   - Or use the sample dataset: python src/prepare_data.py --action sample")
    
    print("\n2. Train the model:")
    print("   - Run: python src/train.py")
    print("   - Monitor training in models/experiment_*/")
    
    print("\n3. Start the API:")
    print("   - Run: python api/app.py")
    print("   - Access web UI: http://localhost:8000")
    print("   - API docs: http://localhost:8000/docs")
    
    print("\n4. Make predictions:")
    print("   - Run: python src/inference.py path/to/image.jpg")
    
    print("\n📖 For more information, see README.md")
    print("\n" + "="*60)


def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("🚀 InspectAI Setup")
    print("="*60)
    print("This script will set up your InspectAI environment")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    print("\n" + "="*60)
    print("📦 Installing dependencies")
    print("="*60)
    print("This may take a few minutes...")
    
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the error messages above.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages failed to install.")
        print("Please try installing them manually:")
        print("pip install -r requirements.txt")
    
    # Check CUDA
    check_cuda()
    
    # Create sample data
    create_sample_data()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
