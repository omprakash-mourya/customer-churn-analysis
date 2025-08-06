"""
Main entry point for Customer Churn Analysis.

This script provides a command-line interface to run various components
of the customer churn prediction pipeline.
"""

import argparse
import os
import sys
import subprocess

def run_training():
    """Run model training pipeline."""
    print("🚀 Starting model training...")
    result = subprocess.run([sys.executable, "models/train_model.py"], 
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0

def run_api():
    """Start FastAPI server."""
    print("🔌 Starting FastAPI server...")
    subprocess.run(["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

def run_dashboard():
    """Start Streamlit dashboard."""
    print("🎨 Starting Streamlit dashboard...")
    subprocess.run(["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501"])

def run_tests():
    """Run tests."""
    print("🧪 Running tests...")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    return result.returncode == 0

def setup_environment():
    """Setup environment and install dependencies."""
    print("🔧 Setting up environment...")
    try:
        # Install requirements
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        if result.returncode == 0:
            print("✅ Dependencies installed successfully!")
            return True
        else:
            print("❌ Failed to install dependencies")
            return False
    except Exception as e:
        print(f"❌ Error setting up environment: {e}")
        return False

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Customer Churn Fire Project")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup environment and install dependencies')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train churn prediction models')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start FastAPI server')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Start Streamlit dashboard')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run complete pipeline (train + api + dashboard)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    print("🔥 Customer Churn Fire Project")
    print("=" * 50)
    
    if args.command == 'setup':
        success = setup_environment()
        if success:
            print("\n🎉 Setup complete! You can now run:")
            print("  python main.py train    # Train models")
            print("  python main.py api      # Start API server")
            print("  python main.py dashboard # Start dashboard")
    
    elif args.command == 'train':
        success = run_training()
        if success:
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed!")
    
    elif args.command == 'api':
        run_api()
    
    elif args.command == 'dashboard':
        run_dashboard()
    
    elif args.command == 'test':
        success = run_tests()
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
    
    elif args.command == 'all':
        print("🚀 Running complete pipeline...")
        
        # Step 1: Training
        print("\n1️⃣ Training models...")
        if not run_training():
            print("❌ Training failed, stopping pipeline")
            return
        
        # Step 2: Start API in background
        print("\n2️⃣ Starting API server...")
        import threading
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        # Step 3: Start dashboard
        print("\n3️⃣ Starting dashboard...")
        run_dashboard()

if __name__ == "__main__":
    main()
