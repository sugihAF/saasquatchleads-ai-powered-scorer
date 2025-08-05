#!/usr/bin/env python3
"""
Startup script for SaaS Lead Scorer
Runs both FastAPI backend and Streamlit frontend
"""

import subprocess
import time
import os
import sys
import threading
import signal
from pathlib import Path

def run_fastapi():
    """Run FastAPI backend"""
    print("🚀 Starting FastAPI backend on port 8501...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0",
            "--port", "8501",
            "--reload"
        ], check=True)
    except subprocess.CalledProcessError:
        print("❌ FastAPI backend failed to start")
    except KeyboardInterrupt:
        print("🛑 FastAPI backend stopped")

def run_streamlit():
    """Run Streamlit frontend"""
    print("🎨 Starting Streamlit frontend on port 8502...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port=8502",
            "--server.address=0.0.0.0",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except subprocess.CalledProcessError:
        print("❌ Streamlit frontend failed to start")
    except KeyboardInterrupt:
        print("🛑 Streamlit frontend stopped")

def main():
    print("🎯 SaaS Lead Scorer - Starting Full Stack Application")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ Error: Please run this script from the saasquatchleads directory")
        print("   where app.py is located.")
        sys.exit(1)
    
    print("✅ Found app.py - starting services...")
    print()
    
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Wait a moment for FastAPI to start
    print("⏳ Waiting for FastAPI to initialize...")
    time.sleep(5)
    
    # Start Streamlit in main thread
    try:
        run_streamlit()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all services...")

if __name__ == "__main__":
    main()
