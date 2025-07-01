"""
Script to run the Air Quality Prediction API server
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def main():
    """Run the API server"""
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    reload = os.environ.get("API_RELOAD", "true").lower() == "true"
    
    print(f"Starting Air Quality Prediction API server on {host}:{port}")
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    main() 