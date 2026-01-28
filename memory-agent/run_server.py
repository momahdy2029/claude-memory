"""Run the memory agent server (for background/production use)."""
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8100)),
        reload=False,  # No reload for background/production
        log_level="warning"  # Less verbose logging
    )
