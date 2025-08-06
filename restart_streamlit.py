#!/usr/bin/env python3
"""
Force restart Streamlit and clear all caches
"""

import subprocess
import sys
import time
import os

def restart_streamlit():
    """Restart Streamlit application"""
    print("🔄 Restarting Streamlit...")
    
    # Kill any existing streamlit processes
    try:
        subprocess.run(["taskkill", "/f", "/im", "streamlit.exe"], 
                      capture_output=True, text=True, shell=True)
        print("✅ Killed existing Streamlit processes")
    except:
        print("ℹ️ No existing Streamlit processes found")
    
    # Wait a moment
    time.sleep(2)
    
    # Start Streamlit
    print("🚀 Starting Streamlit...")
    subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "app/streamlit_app.py", 
        "--server.port=8501",
        "--server.address=localhost",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ])
    
    print("✅ Streamlit restarted!")
    print("🌐 Open: http://localhost:8501")
    print("💡 The cache will be automatically cleared with the new TTL settings")

if __name__ == "__main__":
    restart_streamlit()
