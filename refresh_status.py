#!/usr/bin/env python3
"""
Force refresh Streamlit status by clearing all cache files
"""

import os
import shutil
import tempfile
import streamlit as st

def clear_streamlit_cache():
    """Clear Streamlit cache directories"""
    cache_dirs = [
        os.path.expanduser("~/.streamlit"),
        ".streamlit",
        "__pycache__",
        ".streamlit/cache"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Cleared cache: {cache_dir}")
            except Exception as e:
                print(f"⚠️ Could not clear {cache_dir}: {e}")
    
    print("\n🔄 Cache cleared! Please refresh your browser")
    print("💡 In browser: Press Ctrl+F5 or Cmd+Shift+R")
    print("💡 In Streamlit: Press 'C' key or use menu > Clear Cache")

if __name__ == "__main__":
    clear_streamlit_cache()
