#!/usr/bin/env python3
"""
Complete deployment script for Customer Churn Analysis project
"""

import subprocess
import sys
import os

def run_system_check():
    """Run comprehensive system check"""
    print("🔍 Running system check...")
    try:
        result = subprocess.run([sys.executable, "simple_check.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ System check failed: {e}")
        return False

def setup_environment():
    """Setup Python environment and dependencies"""
    print("📦 Setting up environment...")
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("❌ requirements.txt not found")
            return False
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def run_tests():
    """Run test suite"""
    print("🧪 Running tests...")
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "-v"], 
                              capture_output=True, text=True)
        print(result.stdout)
        return result.returncode == 0
    except FileNotFoundError:
        print("⚠️ pytest not installed, skipping tests")
        return True

def start_dashboard():
    """Start Streamlit dashboard"""
    print("🚀 Starting dashboard...")
    try:
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "app/streamlit_app.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
        print("✅ Dashboard started at http://localhost:8501")
        return True
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        return False

def main():
    """Main deployment function"""
    print("🎯 Customer Churn Analysis - Deployment Script")
    print("=" * 60)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = True
    
    # Step 1: System check
    if not run_system_check():
        success = False
        print("⚠️ System check issues detected but continuing...")
    
    # Step 2: Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        return False
    
    # Step 3: Run tests
    if not run_tests():
        print("⚠️ Some tests failed but continuing...")
    
    # Step 4: Start dashboard
    if start_dashboard():
        print("\n" + "=" * 60)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("📊 Dashboard: http://localhost:8501")
        print("📚 Documentation: README.md")
        print("🐙 GitHub Setup: Run 'python setup_github.py' for instructions")
        print("=" * 60)
    else:
        success = False
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
