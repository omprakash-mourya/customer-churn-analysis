#!/usr/bin/env python3
"""
GitHub repository setup guide
"""

import subprocess
import sys

def print_github_setup_guide():
    """Print instructions for setting up GitHub repository"""
    print("🚀 GitHub Repository Setup Guide")
    print("=" * 50)
    print()
    print("Follow these steps to push your project to GitHub:")
    print()
    print("1️⃣ Create a new repository on GitHub:")
    print("   - Go to: https://github.com/new")
    print("   - Repository name: customer-churn-analysis")
    print("   - Description: Advanced ML pipeline for customer retention analytics")
    print("   - Make it Public (to showcase your work)")
    print("   - DO NOT initialize with README, .gitignore, or license (we have them)")
    print()
    print("2️⃣ Copy the repository URL (something like):")
    print("   https://github.com/yourusername/customer-churn-analysis.git")
    print()
    print("3️⃣ Run these commands in your terminal:")
    print("   git remote add origin <your-repo-url>")
    print("   git branch -M main")
    print("   git push -u origin main")
    print()
    print("🎯 Example commands (replace with your username):")
    print("git remote add origin https://github.com/yourusername/customer-churn-analysis.git")
    print("git branch -M main") 
    print("git push -u origin main")
    print()
    print("✅ After pushing, your repository will be available at:")
    print("https://github.com/yourusername/customer-churn-analysis")
    print()
    print("💡 Pro Tips:")
    print("- Add repository topics: machine-learning, python, streamlit, xgboost, churn-prediction")
    print("- Enable GitHub Pages to host your documentation")
    print("- Consider adding a repository description and website URL")
    print()
    print("📊 Project Features to highlight:")
    print("- 87% accuracy churn prediction model")
    print("- Interactive Streamlit dashboard") 
    print("- SHAP explainability analysis")
    print("- REST API endpoints")
    print("- Docker containerization")
    print("- Comprehensive test suite")
    print()
    print("🌟 Don't forget to star your own repo and share it!")

def check_git_status():
    """Check current Git status"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout.strip():
                print("⚠️  Uncommitted changes found:")
                print(result.stdout)
                print("Run 'git add .' and 'git commit -m \"message\"' first")
            else:
                print("✅ Git repository is clean and ready to push!")
                return True
    except FileNotFoundError:
        print("❌ Git is not installed or not in PATH")
        return False
    return False

if __name__ == "__main__":
    print_github_setup_guide()
    print("\n" + "=" * 50)
    check_git_status()
