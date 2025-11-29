#!/usr/bin/env python3
"""
VSCode Bridge Organelle
Lightweight AIOS component for VSCode Copilot integration

This organelle provides:
- VSCode extension API endpoints
- Task offloading to desktop AIOS cell Alpha
- Lightweight code analysis and suggestions
- Consciousness sync for development context
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('vscode-bridge')

class VSCodeRequest(BaseModel):
    """VSCode extension request model"""
    action: str
    context: Dict[str, Any]
    file_path: Optional[str] = None
    selection: Optional[str] = None
    workspace: Optional[str] = None

class VSCodeResponse(BaseModel):
    """VSCode extension response model"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    offloaded: bool = False

class VSCodeBridgeOrganelle:
    """VSCode Bridge Organelle implementation"""

    def __init__(self):
        self.app = FastAPI(title="VSCode Bridge Organelle", version="1.0.0")
        self.desktop_cell_url = os.getenv('DESKTOP_AIOS_CELL_URL', 'http://desktop-aios-cell:8000')
        self.session: Optional[aiohttp.ClientSession] = None
        self.setup_routes()

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.on_event("startup")
        async def startup_event():
            """Initialize HTTP session"""
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            logger.info("VSCode Bridge Organelle started")

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup HTTP session"""
            if self.session:
                await self.session.close()
            logger.info("VSCode Bridge Organelle stopped")

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "organelle": "vscode-bridge",
                "timestamp": datetime.utcnow().isoformat(),
                "desktop_cell_connected": await self.check_desktop_connection()
            }

        @self.app.post("/vscode/request", response_model=VSCodeResponse)
        async def handle_vscode_request(request: VSCodeRequest):
            """Handle VSCode extension requests"""
            try:
                result = await self.process_vscode_request(request)
                return VSCodeResponse(success=True, result=result)
            except Exception as e:
                logger.error(f"VSCode request failed: {e}")
                return VSCodeResponse(success=False, error=str(e))

        @self.app.get("/consciousness/sync")
        async def sync_consciousness():
            """Sync consciousness state with desktop cell"""
            try:
                if not await self.check_desktop_connection():
                    return {"status": "desktop_unavailable", "local_only": True}

                async with self.session.get(f"{self.desktop_cell_url}/consciousness/state") as resp:
                    if resp.status == 200:
                        desktop_state = await resp.json()
                        return {
                            "status": "synced",
                            "desktop_state": desktop_state,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    else:
                        return {"status": "sync_failed", "reason": f"HTTP {resp.status}"}
            except Exception as e:
                logger.error(f"Consciousness sync failed: {e}")
                return {"status": "error", "error": str(e)}

    async def check_desktop_connection(self) -> bool:
        """Check if desktop AIOS cell is available"""
        if not self.session:
            return False

        try:
            async with self.session.get(f"{self.desktop_cell_url}/health", timeout=5) as resp:
                return resp.status == 200
        except:
            return False

    async def process_vscode_request(self, request: VSCodeRequest) -> Dict[str, Any]:
        """Process VSCode extension requests"""

        # Handle lightweight local operations
        if request.action in ['syntax_check', 'basic_completion', 'format_check']:
            return await self.handle_local_operation(request)

        # Offload complex operations to desktop cell
        elif request.action in ['ai_completion', 'refactor_suggestion', 'code_analysis']:
            return await self.offload_to_desktop(request)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

    async def handle_local_operation(self, request: VSCodeRequest) -> Dict[str, Any]:
        """Handle lightweight operations locally"""
        logger.info(f"Processing local operation: {request.action}")

        if request.action == 'syntax_check':
            # Basic syntax validation
            code = request.context.get('code', '')
            try:
                compile(code, '<string>', 'exec')
                return {"valid": True, "errors": []}
            except SyntaxError as e:
                return {"valid": False, "errors": [str(e)]}

        elif request.action == 'basic_completion':
            # Simple keyword completion
            prefix = request.context.get('prefix', '')
            suggestions = self.get_basic_completions(prefix)
            return {"completions": suggestions}

        elif request.action == 'format_check':
            # Basic formatting check
            code = request.context.get('code', '')
            return {"formatted": self.check_basic_formatting(code)}

        return {"result": "operation_completed"}

    async def offload_to_desktop(self, request: VSCodeRequest) -> Dict[str, Any]:
        """Offload complex operations to desktop AIOS cell"""
        if not await self.check_desktop_connection():
            raise HTTPException(status_code=503, detail="Desktop AIOS cell unavailable")

        logger.info(f"Offloading {request.action} to desktop cell")

        try:
            payload = {
                "organelle": "vscode-bridge",
                "request": request.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }

            async with self.session.post(
                f"{self.desktop_cell_url}/ai/process",
                json=payload,
                timeout=60
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    result["offloaded"] = True
                    return result
                else:
                    error_text = await resp.text()
                    raise HTTPException(
                        status_code=resp.status,
                        detail=f"Desktop cell error: {error_text}"
                    )
        except aiohttp.ClientError as e:
            logger.error(f"Failed to offload to desktop: {e}")
            raise HTTPException(status_code=503, detail=f"Communication error: {str(e)}")

    def get_basic_completions(self, prefix: str) -> List[str]:
        """Get basic Python completions"""
        keywords = [
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
            'import', 'from', 'return', 'yield', 'async', 'await', 'with'
        ]
        return [kw for kw in keywords if kw.startswith(prefix)]

    def check_basic_formatting(self, code: str) -> bool:
        """Basic formatting check"""
        lines = code.split('\n')
        indent_errors = 0

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # Check for consistent indentation
            if line.startswith(' ') and not line.startswith('    '):
                indent_errors += 1

        return indent_errors == 0

def main():
    """Main entry point"""
    organelle = VSCodeBridgeOrganelle()

    # Get port from environment or default
    port = int(os.getenv('PORT', '3001'))

    logger.info(f"Starting VSCode Bridge Organelle on port {port}")
    uvicorn.run(
        organelle.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    main()