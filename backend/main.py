"""
FastAPI backend for Neural Machine Translation
Provides REST API for translation services
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from backend.model_service import ModelService

# ============================================================================
# Initialize FastAPI app
# ============================================================================

app = FastAPI(
    title="Neural Machine Translation API",
    description="English to French translation with LSTM and Attention models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# CORS Configuration
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Initialize Model Service
# ============================================================================

model_service = ModelService()

# ============================================================================
# Request/Response Models
# ============================================================================

class TranslationRequest(BaseModel):
    """Request model for translation"""
    text: str = Field(..., min_length=1, max_length=500, description="Text to translate (English)")
    model_type: str = Field(default="attention", description="Model type: 'baseline' or 'attention'")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace')
        return v.strip()
    
    @validator('model_type')
    def valid_model_type(cls, v):
        if v not in ['baseline', 'attention']:
            raise ValueError('model_type must be "baseline" or "attention"')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, how are you?",
                "model_type": "attention"
            }
        }


class TranslationResponse(BaseModel):
    """Response model for translation"""
    translation: str = Field(..., description="Translated text (French)")
    source: str = Field(..., description="Original source text")
    model_type: str = Field(..., description="Model used for translation")
    attention_weights: Optional[List[List[float]]] = Field(None, description="Attention weights matrix")
    source_tokens: Optional[List[str]] = Field(None, description="Source tokens")
    target_tokens: Optional[List[str]] = Field(None, description="Target tokens")
    
    class Config:
        schema_extra = {
            "example": {
                "translation": "Bonjour, comment allez-vous?",
                "source": "Hello, how are you?",
                "model_type": "attention",
                "attention_weights": [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
                "source_tokens": ["hello", "how", "are"],
                "target_tokens": ["bonjour", "comment"]
            }
        }


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation"""
    texts: List[str] = Field(..., min_items=1, max_items=50)
    model_type: str = Field(default="attention")
    
    @validator('model_type')
    def valid_model_type(cls, v):
        if v not in ['baseline', 'attention']:
            raise ValueError('model_type must be "baseline" or "attention"')
        return v


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    type: str
    total_parameters: int
    trainable_parameters: int
    device: str
    vocab_size_en: int
    vocab_size_fr: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    models_loaded: dict


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("=" * 80)
    print("STARTING NMT API SERVER")
    print("=" * 80)
    
    try:
        model_service.load_models()
        print("\n✓ API server ready")
        print("=" * 80)
    except Exception as e:
        print(f"\n✗ Error loading models: {e}")
        print("\nPlease ensure:")
        print("  1. Models are trained (run: python train.py)")
        print("  2. Data is processed (run: python utils/data_loader.py)")
        print("=" * 80)
        # Don't exit - allow server to start but endpoints will return errors


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\nShutting down NMT API server...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """
    Health check endpoint
    Returns API status and loaded models
    """
    return HealthResponse(
        status="running",
        message="Neural Machine Translation API is running",
        models_loaded={
            "baseline": model_service.baseline_model is not None,
            "attention": model_service.attention_model is not None
        }
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if (model_service.baseline_model or model_service.attention_model) else "unhealthy",
        message="Service is operational" if (model_service.baseline_model or model_service.attention_model) else "No models loaded",
        models_loaded={
            "baseline": model_service.baseline_model is not None,
            "attention": model_service.attention_model is not None
        }
    )


@app.post("/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate(request: TranslationRequest):
    """
    Translate English text to French
    
    - **text**: English text to translate (required)
    - **model_type**: 'baseline' or 'attention' (default: 'attention')
    
    Returns translated text and optionally attention weights for visualization
    """
    try:
        result = model_service.translate(request.text, request.model_type)
        
        return TranslationResponse(
            translation=result['translation'],
            source=result['source'],
            model_type=result['model_type'],
            attention_weights=result.get('attention_weights'),
            source_tokens=result.get('source_tokens'),
            target_tokens=result.get('target_tokens')
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation error: {str(e)}"
        )


@app.post("/translate/batch", tags=["Translation"])
async def batch_translate(request: BatchTranslationRequest):
    """
    Translate multiple texts at once
    
    - **texts**: List of English texts to translate
    - **model_type**: 'baseline' or 'attention'
    
    Returns list of translations
    """
    try:
        results = model_service.batch_translate(request.texts, request.model_type)
        return {"translations": results}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch translation error: {str(e)}"
        )


@app.post("/compare", tags=["Translation"])
async def compare_models(text: str):
    """
    Compare translations from both baseline and attention models
    
    - **text**: English text to translate
    
    Returns translations from both models for comparison
    """
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty"
        )
    
    try:
        results = model_service.compare_translations(text.strip())
        return results
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison error: {str(e)}"
        )


@app.get("/models", tags=["Models"])
async def get_models():
    """
    Get information about loaded models
    
    Returns details about model architecture and parameters
    """
    models_info = {}
    
    for model_type in ['baseline', 'attention']:
        info = model_service.get_model_info(model_type)
        if info:
            models_info[model_type] = info
    
    if not models_info:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models are loaded"
        )
    
    return models_info


@app.get("/models/{model_type}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_type: str):
    """
    Get information about a specific model
    
    - **model_type**: 'baseline' or 'attention'
    """
    if model_type not in ['baseline', 'attention']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="model_type must be 'baseline' or 'attention'"
        )
    
    info = model_service.get_model_info(model_type)
    
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{model_type} model not loaded"
        )
    
    return ModelInfo(**info)


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 80)
    print("STARTING FASTAPI SERVER")
    print("=" * 80)
    print(f"Host: {config.API_HOST}")
    print(f"Port: {config.API_PORT}")
    print(f"Docs: http://localhost:{config.API_PORT}/docs")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level="info"
    )