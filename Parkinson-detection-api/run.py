"""
Run the FastAPI application
"""
import uvicorn
from app.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,  # Set to False in production
        log_level="info",
        access_log=True
    )
