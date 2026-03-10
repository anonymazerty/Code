from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.routers import generation, composition, mcqs
import time
from scalar_fastapi import get_scalar_api_reference
    
app = FastAPI()
origins = ['http://localhost:5173']

app.add_middleware(     
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   

app.include_router(generation.router)
app.include_router(composition.router)
app.include_router(mcqs.router)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(time.time() - start_time)
    return response

@app.get("/scalar", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
    )