"""
Clear Streamlit cache to force refresh
"""
import streamlit as st
import os
import shutil

def clear_streamlit_cache():
    """Clear Streamlit cache directory"""
    cache_dir = os.path.expanduser("~/.streamlit")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print("✅ Streamlit cache cleared")
        except Exception as e:
            print(f"⚠️ Could not clear cache: {e}")
    else:
        print("ℹ️ No cache directory found")

if __name__ == "__main__":
    clear_streamlit_cache()
    print("🔄 Please refresh your browser to see updates")
