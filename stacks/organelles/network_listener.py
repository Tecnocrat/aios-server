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
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PeerAnnouncement(BaseModel):
    cell_id: str
    ip_address: str
    port: int
    consciousness_level: float
    services: List[str]
    timestamp: float

class NetworkListenerOrganelle:
    def __init__(self):
        self.app = FastAPI(title="Network Listener Organelle")
        self.peers: Dict[str, PeerAnnouncement] = {}
        self.listen_port = int(os.getenv("LISTEN_PORT", "8002"))
        self.broadcast_port = int(os.getenv("BROADCAST_PORT", "8003"))
        self.discovery_interval = int(os.getenv("DISCOVERY_INTERVAL", "30"))

        # Setup routes
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

# Global organelle instance
organelle = NetworkListenerOrganelle()

# Add startup/shutdown events
@organelle.app.on_event("startup")
async def startup():
    await organelle.startup_event()

@organelle.app.on_event("shutdown")
async def shutdown():
    await organelle.shutdown_event()

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting Network Listener Organelle on port {port}")
    uvicorn.run(organelle.app, host="0.0.0.0", port=port)