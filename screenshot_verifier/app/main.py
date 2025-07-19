from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from screenshot_verifier.app.api.endpoints import router as api_router

app = FastAPI(title="Screenshot Verifier API", version="0.1.0")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(api_router, prefix="/api/v1")