import uvicorn
from app.app import app

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="localhost" , port=8000, reload=True, log_level="info")