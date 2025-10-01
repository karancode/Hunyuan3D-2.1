"""
Constants and error messages for Hunyuan3D API server.
"""

# Error messages
SERVER_ERROR_MSG = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
MODERATION_MSG = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

# Default values
DEFAULT_SAVE_DIR = 'gradio_cache'
DEFAULT_WORKER_ID = None  # Will be generated if None

# API metadata
API_TITLE = "Hunyuan3D API Server"
API_DESCRIPTION = """
# Hunyuan3D 2.1 API Server

This API server provides a clean, simple endpoint for generating 3D models from 2D images using the Hunyuan3D model.

## Features

- **3D Shape Generation**: Convert 2D images to 3D meshes
- **Mesh Processing**: Automatic floater removal, degenerate face removal, and face reduction
- **STL Export**: Direct STL output optimized for 3D printing
- **Background Removal**: Automatic background removal from input images
- **Clean Architecture**: Simple, focused API for production use

## Usage

1. Use `/generate` for direct 3D model generation from images → STL file
2. Use `/health` to verify service status

## Model Information

- **Model**: Hunyuan3D-2.1 by Tencent
- **License**: TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
- **Output**: STL files ready for 3D printing
"""
API_VERSION = "2.1.0"
API_CONTACT = {
    "name": "Hunyuan3D Team",
    "url": "https://github.com/Tencent/Hunyuan3D",
}
API_LICENSE_INFO = {
    "name": "TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT",
    "url": "https://github.com/Tencent/Hunyuan3D/blob/main/LICENSE",
}

# API tags metadata
API_TAGS_METADATA = [
    {
        "name": "generation",
        "description": "3D model generation endpoint. Generate processed 3D models from 2D images as STL files.",
    },
    {
        "name": "status",
        "description": "Health check endpoint. Verify service status and availability.",
    },
] 