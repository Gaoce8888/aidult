"""
FastAPI application for Screenshot Authenticity AI API
"""
import asyncio
import io
import time
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image
import cv2
import numpy as np

# Import project modules
from ..core.authenticity_engine import AuthenticityEngine
from ..utils.logging_config import setup_logging
from ..utils.rate_limiter import RateLimiter
from ..utils.security import SecurityManager
from config.config import settings

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.api.title,
    version=settings.api.version,
    description=settings.api.description,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Security
security = HTTPBearer()
security_manager = SecurityManager()

# Rate limiter
rate_limiter = RateLimiter(
    requests=settings.api.rate_limit_requests,
    window=settings.api.rate_limit_window
)

# Global instances
authenticity_engine: Optional[AuthenticityEngine] = None


# Pydantic models
class AnalysisRequest(BaseModel):
    """Request model for screenshot analysis"""
    context: Optional[Dict] = Field(default=None, description="Additional context information")
    source: Optional[str] = Field(default=None, description="Source platform (android/ios)")
    app_type: Optional[str] = Field(default=None, description="App type (payment/social/shopping)")
    priority: Optional[str] = Field(default="normal", description="Analysis priority (low/normal/high)")


class AnalysisResponse(BaseModel):
    """Response model for screenshot analysis"""
    authentic: bool = Field(description="Whether the screenshot is authentic")
    confidence: float = Field(description="Confidence score (0-1)")
    risk_assessment: Dict = Field(description="Detailed risk assessment")
    risk_factors: List[Dict] = Field(description="List of risk factors found")
    detection_summary: Dict = Field(description="Summary of detection methods used")
    request_id: str = Field(description="Unique request identifier")
    analysis_time_ms: float = Field(description="Total analysis time in milliseconds")


class SystemStatus(BaseModel):
    """System status response model"""
    status: str = Field(description="Overall system status")
    detectors: Dict = Field(description="Status of individual detectors")
    performance: Dict = Field(description="Performance metrics")
    version: str = Field(description="API version")
    uptime_seconds: float = Field(description="System uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(description="Error message")
    error_code: str = Field(description="Error code")
    request_id: Optional[str] = Field(description="Request identifier")
    timestamp: float = Field(description="Error timestamp")


# Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    if not rate_limiter.is_allowed(request.client.host):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    response = await call_next(request)
    return response


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://*.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Trusted host middleware
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.yourdomain.com", "localhost"]
    )


# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key or JWT token"""
    try:
        # Validate token
        user_info = security_manager.validate_token(credentials.credentials)
        return user_info
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )


async def validate_image_file(file: UploadFile) -> bytes:
    """Validate uploaded image file"""
    # Check file size
    if file.size > settings.api.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size of {settings.api.max_file_size} bytes"
        )
    
    # Read file content
    content = await file.read()
    
    # Validate image format
    try:
        image = Image.open(io.BytesIO(content))
        if image.format.lower() not in [fmt.upper() for fmt in settings.api.allowed_image_formats]:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image format. Allowed formats: {settings.api.allowed_image_formats}"
            )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )
    
    return content


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global authenticity_engine
    
    try:
        # Initialize authenticity engine
        authenticity_engine = AuthenticityEngine(settings.detection.dict())
        logger.info("Authenticity engine initialized successfully")
        
        # Log startup information
        logger.info(f"API server starting up - Version: {settings.api.version}")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info(f"Environment: {settings.environment}")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global authenticity_engine
    
    try:
        if authenticity_engine:
            authenticity_engine.close()
            logger.info("Authenticity engine closed")
        
        logger.info("API server shutting down")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# API Routes
@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Screenshot Authenticity AI API",
        "version": settings.api.version,
        "status": "operational",
        "docs": "/docs" if settings.debug else None
    }


@app.get("/health", response_model=Dict)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.api.version
    }


@app.get("/status", response_model=SystemStatus)
async def system_status(user: Dict = Depends(get_current_user)):
    """Get detailed system status"""
    try:
        status_info = authenticity_engine.get_system_status() if authenticity_engine else {}
        
        return SystemStatus(
            status="operational" if authenticity_engine else "initializing",
            detectors=status_info.get('detectors', {}),
            performance=status_info.get('performance', {}),
            version=settings.api.version,
            uptime_seconds=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/verify/screenshot", response_model=AnalysisResponse)
async def verify_screenshot(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    analysis_request: AnalysisRequest = AnalysisRequest(),
    user: Dict = Depends(get_current_user)
):
    """
    Verify screenshot authenticity
    
    This endpoint accepts an image file and optional context information,
    then performs comprehensive authenticity analysis using multiple detection methods.
    """
    start_time = time.time()
    request_id = request.state.request_id
    
    try:
        # Validate input
        if not authenticity_engine:
            raise HTTPException(
                status_code=503,
                detail="Authenticity engine not available"
            )
        
        # Validate and process image
        image_content = await validate_image_file(file)
        
        # Save temporary file for analysis
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / f"{request_id}_{file.filename}"
        
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(image_content)
        
        # Prepare context
        context = analysis_request.context or {}
        context.update({
            'source': analysis_request.source,
            'app_type': analysis_request.app_type,
            'priority': analysis_request.priority,
            'user_id': user.get('user_id'),
            'request_id': request_id,
            'filename': file.filename,
            'file_size': len(image_content)
        })
        
        # Perform analysis
        logger.info(f"Starting analysis for request {request_id}")
        analysis_result = await authenticity_engine.analyze_screenshot(
            str(temp_file_path), 
            context
        )
        
        # Cleanup temporary file
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        # Calculate total analysis time
        analysis_time_ms = (time.time() - start_time) * 1000
        
        # Prepare response
        response = AnalysisResponse(
            authentic=analysis_result.get('authentic', True),
            confidence=analysis_result.get('confidence', 0.0),
            risk_assessment=analysis_result.get('risk_assessment', {}),
            risk_factors=analysis_result.get('risk_factors', []),
            detection_summary=analysis_result.get('detection_summary', {}),
            request_id=request_id,
            analysis_time_ms=analysis_time_ms
        )
        
        # Log result
        logger.info(
            f"Analysis completed for request {request_id}: "
            f"authentic={response.authentic}, confidence={response.confidence:.3f}, "
            f"time={analysis_time_ms:.1f}ms"
        )
        
        # Store result for analytics (background task)
        background_tasks.add_task(
            store_analysis_result, 
            request_id, 
            analysis_result, 
            context, 
            user
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in screenshot verification for request {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during analysis"
        )


@app.post("/api/v1/verify/batch", response_model=List[AnalysisResponse])
async def verify_batch_screenshots(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    analysis_request: AnalysisRequest = AnalysisRequest(),
    user: Dict = Depends(get_current_user)
):
    """
    Verify multiple screenshots in batch
    
    This endpoint accepts multiple image files and processes them in parallel
    for improved throughput.
    """
    request_id = request.state.request_id
    
    try:
        # Validate batch size
        max_batch_size = 10  # Configurable limit
        if len(files) > max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=f"Batch size exceeds maximum allowed size of {max_batch_size}"
            )
        
        # Process files in parallel
        tasks = []
        for i, file in enumerate(files):
            # Create individual request for each file
            individual_request = Request(request.scope.copy())
            individual_request.state.request_id = f"{request_id}_{i}"
            
            task = verify_screenshot(
                individual_request,
                background_tasks,
                file,
                analysis_request,
                user
            )
            tasks.append(task)
        
        # Wait for all analyses to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle individual file errors
                responses.append(AnalysisResponse(
                    authentic=True,  # Default to authentic on error
                    confidence=0.0,
                    risk_assessment={},
                    risk_factors=[],
                    detection_summary={},
                    request_id=f"{request_id}_{i}",
                    analysis_time_ms=0.0
                ))
                logger.error(f"Error processing file {i} in batch {request_id}: {result}")
            else:
                responses.append(result)
        
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch screenshot verification for request {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during batch analysis"
        )


@app.get("/api/v1/analysis/{request_id}", response_model=Dict)
async def get_analysis_result(
    request_id: str,
    user: Dict = Depends(get_current_user)
):
    """
    Retrieve analysis result by request ID
    
    This endpoint allows retrieval of previously performed analysis results
    for audit and review purposes.
    """
    try:
        # This would typically query a database
        # For now, return a placeholder response
        return {
            "message": "Analysis result retrieval not implemented",
            "request_id": request_id
        }
    except Exception as e:
        logger.error(f"Error retrieving analysis result {request_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/models/status", response_model=Dict)
async def get_models_status(user: Dict = Depends(get_current_user)):
    """Get status of all available models"""
    try:
        if not authenticity_engine or not authenticity_engine.deep_learning_detector:
            raise HTTPException(status_code=503, detail="Models not available")
        
        models_info = {}
        for model_name in ['efficientnet', 'vit', 'dual_stream', 'multi_scale', 'ensemble']:
            model_summary = authenticity_engine.deep_learning_detector.get_model_summary(model_name)
            models_info[model_name] = model_summary
        
        return {
            "available_models": models_info,
            "default_model": "ensemble",
            "status": "operational"
        }
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Background tasks
async def cleanup_temp_file(file_path: Path):
    """Clean up temporary files"""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary file {file_path}: {e}")


async def store_analysis_result(
    request_id: str, 
    analysis_result: Dict, 
    context: Dict, 
    user: Dict
):
    """Store analysis result for analytics and audit"""
    try:
        # This would typically store in a database
        logger.info(f"Storing analysis result for request {request_id}")
        
        # Example: Store in database
        # await database.store_analysis_result({
        #     'request_id': request_id,
        #     'result': analysis_result,
        #     'context': context,
        #     'user': user,
        #     'timestamp': time.time()
        # })
        
    except Exception as e:
        logger.error(f"Error storing analysis result for request {request_id}: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            request_id=getattr(request.state, 'request_id', None),
            timestamp=time.time()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            request_id=getattr(request.state, 'request_id', None),
            timestamp=time.time()
        ).dict()
    )


# Main function for running the server
def main():
    """Main function to run the API server"""
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        access_log=True,
        log_level="info" if not settings.debug else "debug"
    )


if __name__ == "__main__":
    main()