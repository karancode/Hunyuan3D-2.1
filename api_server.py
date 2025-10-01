# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
Clean and simplified Hunyuan3D API server with mesh processing.
"""
import argparse
import asyncio
import os
import sys
import time
import traceback
import uuid

import torch
import trimesh
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# Add paths for Hunyuan3D modules
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Import from root-level modules
from api_models import GenerationRequest, HealthResponse
from logger_utils import build_logger
from constants import (
    SERVER_ERROR_MSG, DEFAULT_SAVE_DIR, API_TITLE, API_DESCRIPTION,
    API_VERSION, API_CONTACT, API_LICENSE_INFO, API_TAGS_METADATA
)
from model_worker import ModelWorker

# Import mesh processing functions
from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover, FaceReducer

# Global variables
SAVE_DIR = DEFAULT_SAVE_DIR
worker_id = str(uuid.uuid4())[:6]
logger = build_logger("api_server", f"{SAVE_DIR}/api_server.log")

# Global worker and mesh processors
worker = None
model_semaphore = None
floater_remover = None
degenerate_remover = None
face_reducer = None

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact=API_CONTACT,
    license_info=API_LICENSE_INFO,
    tags_metadata=API_TAGS_METADATA
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_mesh(mesh: trimesh.Trimesh, max_faces: int = 40000) -> trimesh.Trimesh:
    """
    Process mesh by removing floaters, degenerate faces, and reducing face count.
    
    Args:
        mesh: Input trimesh object
        max_faces: Maximum number of faces for the output mesh
        
    Returns:
        Processed trimesh object
    """
    start_time = time.time()
    
    # Remove floating components
    logger.info("Removing floaters...")
    mesh = floater_remover(mesh)
    
    # Remove degenerate faces
    logger.info("Removing degenerate faces...")
    mesh = degenerate_remover(mesh)
    
    # Reduce face count if needed
    if mesh.faces.shape[0] > max_faces:
        logger.info(f"Reducing faces from {mesh.faces.shape[0]} to {max_faces}...")
        mesh = face_reducer(mesh, max_facenum=max_faces)
    
    processing_time = time.time() - start_time
    logger.info(f"Mesh processing completed in {processing_time:.2f} seconds")
    
    return mesh


@app.post("/generate", tags=["generation"])
async def generate_3d_model(request: GenerationRequest):
    """
    Generate a 3D model from an input image with mesh processing.

    This endpoint:
    1. Takes an image and generates a 3D mesh
    2. Processes the mesh (removes floaters, degenerate faces, reduces face count)
    3. Exports as STL file for 3D printing

    Returns:
        FileResponse: The processed 3D model as STL file
    """
    logger.info(f"Starting 3D generation for worker {worker_id}")
    
    # Convert Pydantic model to dict for compatibility
    params = request.dict()
    uid = uuid.uuid4()
    
    try:
        # Generate initial mesh
        logger.info("Generating 3D mesh...")
        file_path, uid = worker.generate(uid, params)
        
        # Load and process mesh
        logger.info("Loading and processing mesh...")
        mesh = trimesh.load(file_path)
        
        # Apply mesh processing
        processed_mesh = process_mesh(mesh, max_faces=request.face_count)
        
        # Export as STL
        stl_path = file_path.replace('.glb', '_processed.stl')
        processed_mesh.export(stl_path)
        
        logger.info(f"Successfully generated and processed 3D model: {stl_path}")
        return FileResponse(
            stl_path, 
            filename=f"hunyuan3d_model_{str(uid)[:8]}.stl",
            media_type="application/octet-stream"
        )
        
    except ValueError as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=400, detail=f"Generation failed: {str(e)}")
    except torch.cuda.CudaError as e:
        logger.error(f"CUDA error: {e}")
        raise HTTPException(status_code=500, detail="GPU processing error")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health_check():
    """
    Health check endpoint to verify the service is running.

    Returns:
        HealthResponse: Service health status and worker identifier
    """
    return JSONResponse({
        "status": "healthy", 
        "worker_id": worker_id
    }, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hunyuan3D API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2.1', help="Model path")
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-1', help="Model subfolder")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument('--mc_algo', type=str, default='mc', help="Marching cubes algorithm")
    parser.add_argument("--limit-model-concurrency", type=int, default=5, help="Model concurrency limit")
    parser.add_argument('--enable_flashvdm', action='store_true', help="Enable FlashVDM")
    parser.add_argument('--compile', action='store_true', help="Compile model")
    parser.add_argument('--low_vram_mode', action='store_true', help="Low VRAM mode")
    parser.add_argument('--cache-path', type=str, default='./gradio_cache', help="Cache directory")
    args = parser.parse_args()
    
    logger.info(f"Starting Hunyuan3D API Server with args: {args}")

    # Update SAVE_DIR based on cache-path argument
    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger.info(f"Using cache directory: {SAVE_DIR}")

    # Initialize mesh processors
    logger.info("Initializing mesh processors...")
    floater_remover = FloaterRemover()
    degenerate_remover = DegenerateFaceRemover()
    face_reducer = FaceReducer()
    logger.info("Mesh processors initialized")

    # Initialize model semaphore
    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    # Initialize model worker
    logger.info("Initializing model worker...")
    worker = ModelWorker(
        model_path=args.model_path,
        subfolder=args.subfolder,
        device=args.device,
        low_vram_mode=args.low_vram_mode,
        worker_id=worker_id,
        model_semaphore=model_semaphore,
        save_dir=SAVE_DIR,
        mc_algo=args.mc_algo,
        enable_flashvdm=args.enable_flashvdm,
        compile=args.compile
    )
    logger.info("Model worker initialized")

    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")