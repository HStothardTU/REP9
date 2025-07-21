#!/usr/bin/env python3
"""
Demo Runner for Net Zero Transport Model
This script helps launch the Streamlit app with proper configuration for demos.
"""

import subprocess
import sys
import os

def main():
    print("🚦 Net Zero Transport Model - Demo Launcher")
    print("=" * 50)
    
    # Check if required packages are installed
    try:
        import streamlit
        import pandas
        import numpy
        import pulp
        import requests
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return
    
    # Set Streamlit configuration for demo
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    print("🚀 Starting Streamlit app...")
    print("📊 The app will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the demo")
    print("-" * 50)
    
    try:
        # Run the Streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py", "--server.port=8501"])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped. Thanks for trying the Net Zero Transport Model!")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

if __name__ == "__main__":
    main() 