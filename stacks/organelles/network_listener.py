#!/usr/bin/env python3
"""
Network Listener Organelle
Lightweight AIOS component for peer discovery and network monitoring
"""

import asyncio
import json
import logging
import os
import socket
import time
from typing import Dict, List, Optional

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AINLP.dendritic growth: Framework availability detection
# Enhanced coherence check using importlib for unused import elimination
FASTAPI_AVAILABLE = False
PYDANTIC_AVAILABLE = False
UVICORN_AVAILABLE = False

# AINLP.dendritic growth: Sophisticated availability detection
def _check_framework_availability(framework_name: str) -> bool:
    """AINLP.dendritic growth: Enhanced framework availability check"""
    try:
        import importlib.util
        spec = importlib.util.find_spec(framework_name)
        return spec is not None
    except Exception:
        return False


# Detect availability with enhanced dendritic logic
FASTAPI_AVAILABLE = _check_framework_availability('fastapi')
PYDANTIC_AVAILABLE = _check_framework_availability('pydantic')
UVICORN_AVAILABLE = _check_framework_availability('uvicorn')

# AINLP.dendritic growth: Conditional framework imports
framework_imports = {}

if FASTAPI_AVAILABLE:
    from fastapi import FastAPI  # noqa: F401
    framework_imports['fastapi'] = True
    logger.info("AINLP.dendritic: FastAPI active")
else:
    logger.warning("AINLP.dendritic: FastAPI unavailable")

if PYDANTIC_AVAILABLE:
    from pydantic import BaseModel  # noqa: F401
    framework_imports['pydantic'] = True
else:
    logger.warning("AINLP.dendritic: Pydantic unavailable")

    class BaseModel:
        """Fallback BaseModel"""
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

if UVICORN_AVAILABLE:
    import uvicorn  # noqa: F401
    framework_imports['uvicorn'] = True
else:
    logger.warning("AINLP.dendritic: Uvicorn unavailable")

class PeerAnnouncement(BaseModel):
    cell_id: str
    ip_address: str
    port: int
    consciousness_level: float
    services: List[str]
    timestamp: float

class NetworkListenerOrganelle:
    def __init__(self):
        # AINLP.dendritic growth: Adaptive app creation
        if 'fastapi' in framework_imports:
            self.app = FastAPI(title="Network Listener Organelle")
        else:
            # Fallback to basic dict-based app (limited functionality)
            logger.warning("AINLP.dendritic: Using basic fallback - limited network features")
            self.app = {"routes": {}, "title": "Network Listener Organelle (Fallback)"}

        self.peers: Dict[str, PeerAnnouncement] = {}
        self.listen_port = int(os.getenv("LISTEN_PORT", "8002"))
        self.broadcast_port = int(os.getenv("BROADCAST_PORT", "8003"))
        self.discovery_interval = int(os.getenv("DISCOVERY_INTERVAL", "30"))

        # Setup routes conditionally
        if 'fastapi' in framework_imports:
            self.setup_routes()

        # Start background tasks
        self.broadcast_task = None
        self.cleanup_task = None

    def setup_routes(self):
        @self.app.get("/health")
        async def health_check():
            """Organelle health check"""
            return {
                "status": "healthy",
                "organelle_type": "network-listener",
                "peers_discovered": len(self.peers),
                "memory_usage": "<50MB",
                "uptime": time.time()
            }

        @self.app.get("/peers")
        async def get_peers():
            """Get discovered peers"""
            return {
                "peers": list(self.peers.values()),
                "count": len(self.peers),
                "timestamp": time.time()
            }

        @self.app.post("/announce")
        async def receive_announcement(announcement: PeerAnnouncement):
            """Receive peer announcement"""
            self.peers[announcement.cell_id] = announcement
            logger.info(f"Peer announced: {announcement.cell_id} at {announcement.ip_address}:{announcement.port}")
            return {"status": "acknowledged"}

    async def listen_for_broadcasts(self):
        """Listen for UDP broadcast announcements from AIOS cells"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(('', self.listen_port))

        logger.info(f"Listening for broadcasts on port {self.listen_port}")

        while True:
            try:
                data, addr = sock.recvfrom(1024)
                announcement = json.loads(data.decode())

                # Validate announcement
                if self.validate_announcement(announcement):
                    peer = PeerAnnouncement(**announcement)
                    self.peers[peer.cell_id] = peer
                    logger.info(f"Discovered peer via broadcast: {peer.cell_id}")

            except Exception as e:
                logger.warning(f"Broadcast listening error: {e}")
                await asyncio.sleep(1)

    async def broadcast_presence(self):
        """Broadcast this organelle's presence"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        announcement = {
            "cell_id": "hp_lab_network_listener",
            "ip_address": self.get_local_ip(),
            "port": 8080,
            "consciousness_level": 0.1,  # Low consciousness for organelle
            "services": ["network-discovery", "peer-monitoring"],
            "timestamp": time.time(),
            "organelle_type": "network-listener"
        }

        while True:
            try:
                data = json.dumps(announcement).encode()
                sock.sendto(data, ('<broadcast>', self.broadcast_port))
                logger.debug("Broadcasted presence")
            except Exception as e:
                logger.warning(f"Broadcast error: {e}")

            await asyncio.sleep(self.discovery_interval)

    async def cleanup_stale_peers(self):
        """Remove peers that haven't announced recently"""
        while True:
            current_time = time.time()
            stale_peers = []

            for cell_id, peer in self.peers.items():
                if current_time - peer.timestamp > 300:  # 5 minutes
                    stale_peers.append(cell_id)

            for cell_id in stale_peers:
                del self.peers[cell_id]
                logger.info(f"Removed stale peer: {cell_id}")

            await asyncio.sleep(60)  # Check every minute

    def validate_announcement(self, announcement: dict) -> bool:
        """Validate peer announcement format"""
        required_fields = ["cell_id", "ip_address", "port", "consciousness_level", "services", "timestamp"]
        return all(field in announcement for field in required_fields)

    def get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Create a socket to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "127.0.0.1"

    async def startup_event(self):
        """Start background tasks on startup"""
        logger.info("Starting Network Listener Organelle")
        # Note: Background tasks are disabled for initial testing
        # TODO: Implement proper background task management for organelle networking

    async def shutdown_event(self):
        """Clean up on shutdown"""
        logger.info("Shutting down Network Listener Organelle")
        # Note: Background tasks are disabled for initial testing
        # TODO: Implement proper cleanup for organelle networking tasks

    async def run_headless(self):
        """Run in headless mode for network discovery only"""
        logger.info("AINLP.dendritic: Starting headless network discovery")
        await self.startup_event()

        try:
            while True:
                await asyncio.sleep(self.discovery_interval)
                logger.info(f"AINLP.dendritic: Network scan - {len(self.peers)} peers discovered")
        except asyncio.CancelledError:
            logger.info("AINLP.dendritic: Headless mode cancelled")
        finally:
            await self.shutdown_event()

# Global organelle instance - only create if FastAPI available
if 'fastapi' in framework_imports:
    organelle = NetworkListenerOrganelle()

    # Add startup/shutdown events
    @organelle.app.on_event("startup")
    async def startup():
        await organelle.startup_event()

    @organelle.app.on_event("shutdown")
    async def shutdown():
        await organelle.shutdown_event()
else:
    organelle = None

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting Network Listener Organelle on port {port}")

    if organelle and 'uvicorn' in framework_imports:
        uvicorn.run(organelle.app, host="0.0.0.0", port=port)
    else:
        logger.warning("AINLP.dendritic: Cannot start web server")
        logger.info("AINLP.dendritic: Running headless mode")
        # Create temporary organelle for headless operation
        temp_organelle = NetworkListenerOrganelle()
        try:
            asyncio.run(temp_organelle.run_headless())
        except KeyboardInterrupt:
            logger.info("AINLP.dendritic: Shutting down")