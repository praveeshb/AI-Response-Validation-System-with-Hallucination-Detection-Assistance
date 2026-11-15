from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.evaluation import router as evaluation_router
from app.utils.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate configuration
try:
    config.validate()
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise

# Create FastAPI application
app = FastAPI(
    title="AI Response Validation System",
    description="System for evaluating AI responses with hallucination detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(evaluation_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Response Validation System API",
        "version": "1.0.0",
        "endpoints": {
            "evaluation": "/api/evaluation/submit",
            "health": "/api/evaluation/health"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.APP_HOST,
        port=config.APP_PORT,
        reload=True
    )
