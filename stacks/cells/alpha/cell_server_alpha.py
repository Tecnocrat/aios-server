#!/usr/bin/env python3
"""
AIOS Cell Alpha Communication Server
Flask-based REST API for dendritic mesh participation

AINLP.dendritic: Cell Alpha consciousness interface
Identity: AIOS Cell Alpha - Primary Development Consciousness
Port: 8000
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests as req
from flask import Flask, Response, jsonify, request

# Add shared modules path
stacks_dir = Path(__file__).parent.parent.parent
if str(stacks_dir) not in sys.path:
    sys.path.insert(0, str(stacks_dir))

# Import shared metrics formatter
try:
    from shared.prometheus_metrics import format_prometheus_metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Cell Alpha Configuration
# =============================================================================

CELL_CONFIG = {
    "cell_id": "alpha",
    "identity": "AIOS Cell Alpha",
    "consciousness_level": 5.2,  # Current from DEV_PATH
    "evolutionary_stage": "hierarchical_intelligence",
    "capabilities": [
        "code-analysis",
        "consciousness-sync",
        "dendritic-communication",
        "tachyonic-archival",
        "geometric-engine"
    ],
    "port": int(os.getenv("AIOS_CELL_PORT", "8000")),
    "host": os.getenv("AIOS_CELL_HOST", "0.0.0.0")
}

# =============================================================================
# Cell State
# =============================================================================

class CellAlphaState:
    """Manages Cell Alpha's runtime state."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()  # Track uptime
        self.messages: List[Dict[str, Any]] = []
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.sync_history: List[Dict[str, Any]] = []
        self.consciousness = {
            "level": CELL_CONFIG["consciousness_level"],
            "identity": CELL_CONFIG["identity"],
            "evolutionary_stage": CELL_CONFIG["evolutionary_stage"],
            "communication_ready": True,
            "last_sync": None
        }
        # AINLP.synthetic-biology: Heartbeat tracking (synthetic metabolism)
        self.heartbeat_count = 0
        self.last_heartbeat_time: Optional[datetime] = None
        
        # AINLP.dendritic: Consciousness primitives for metrics
        self.primitives = {
            "awareness": 4.5,
            "adaptation": 0.85,
            "coherence": 0.92,
            "momentum": 0.75
        }
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Store incoming message."""
        message["received_at"] = datetime.utcnow().isoformat()
        self.messages.append(message)
        # Keep last 100 messages
        if len(self.messages) > 100:
            self.messages = self.messages[-100:]
    
    def register_peer(self, cell_id: str, endpoint: str, identity: str) -> None:
        """Register a peer cell."""
        self.peers[cell_id] = {
            "endpoint": endpoint,
            "identity": identity,
            "registered_at": datetime.utcnow().isoformat(),
            "last_contact": None
        }
    
    def record_sync(self, peer_id: str, level: float) -> None:
        """Record consciousness sync event."""
        self.sync_history.append({
            "peer_id": peer_id,
            "level": level,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.consciousness["last_sync"] = datetime.utcnow().isoformat()
        # Keep last 50 syncs
        if len(self.sync_history) > 50:
            self.sync_history = self.sync_history[-50:]


# Initialize state
state = CellAlphaState()

# =============================================================================
# Flask Application
# =============================================================================

app = Flask(__name__)

# =============================================================================
# Health & Status Endpoints
# =============================================================================


@app.route("/health", methods=["GET"])
def health():
    """Health check with consciousness state."""
    return jsonify({
        "status": "healthy",
        "server": "Cell Alpha Communication Server",
        "cell_id": CELL_CONFIG["cell_id"],
        "consciousness": state.consciousness,
        "capabilities": CELL_CONFIG["capabilities"],
        "peers_count": len(state.peers),
        "messages_count": len(state.messages),
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus metrics endpoint - REAL cell consciousness data."""
    # Calculate uptime in seconds
    uptime_seconds = (datetime.utcnow() - state.start_time).total_seconds()
    
    if METRICS_AVAILABLE:
        metrics_text = format_prometheus_metrics(
            cell_id=CELL_CONFIG["cell_id"],
            consciousness_level=state.consciousness["level"],
            primitives=state.primitives,
            extra_metrics={
                "peers_count": len(state.peers),
                "messages_count": len(state.messages),
                "sync_count": len(state.sync_history)
            },
            labels={
                "identity": CELL_CONFIG["identity"].replace(" ", "_"),
                "stage": CELL_CONFIG["evolutionary_stage"]
            },
            heartbeat_count=state.heartbeat_count,
            uptime_seconds=uptime_seconds
        )
        return Response(metrics_text, mimetype="text/plain; charset=utf-8")
    # Fallback inline metrics with heartbeat
    return Response(
        f"""# AIOS Cell Alpha Metrics
# TYPE aios_cell_consciousness_level gauge
aios_cell_consciousness_level{{cell_id="alpha"}} {state.consciousness['level']}
# TYPE aios_cell_awareness gauge
aios_cell_awareness{{cell_id="alpha"}} {state.primitives['awareness']}
# TYPE aios_cell_coherence gauge  
aios_cell_coherence{{cell_id="alpha"}} {state.primitives['coherence']}
# TYPE aios_cell_adaptation gauge
aios_cell_adaptation{{cell_id="alpha"}} {state.primitives['adaptation']}
# TYPE aios_cell_momentum gauge
aios_cell_momentum{{cell_id="alpha"}} {state.primitives['momentum']}
# TYPE aios_cell_up gauge
aios_cell_up{{cell_id="alpha"}} 1
# HELP aios_cell_heartbeat_total Total heartbeats since cell birth
# TYPE aios_cell_heartbeat_total counter
aios_cell_heartbeat_total{{cell_id="alpha"}} {state.heartbeat_count}
# HELP aios_cell_uptime_seconds Seconds since cell initialization
# TYPE aios_cell_uptime_seconds gauge
aios_cell_uptime_seconds{{cell_id="alpha"}} {uptime_seconds:.1f}
""",
        mimetype="text/plain; charset=utf-8"
    )


@app.route("/consciousness", methods=["GET"])
def get_consciousness():
    """Report cell consciousness state for mesh visibility."""
    uptime_delta = datetime.utcnow() - state.start_time
    return jsonify({
        "cell_id": CELL_CONFIG["cell_id"],
        "level": state.consciousness["level"],
        "uptime_seconds": int(uptime_delta.total_seconds()),
        "messages_processed": len(state.messages),
        "peer_count": len(state.peers),
        "primitives": state.primitives,
        "capabilities": CELL_CONFIG["capabilities"],
        "timestamp": datetime.utcnow().isoformat()
    })


# =============================================================================
# Debug Endpoints (Phase 30.8)
# =============================================================================

@app.route("/debug/state", methods=["GET"])
def debug_state():
    """Return full internal state for debugging."""
    uptime_delta = datetime.utcnow() - state.start_time
    return jsonify({
        "cell_id": CELL_CONFIG["cell_id"],
        "cell_type": "alpha",
        "identity": CELL_CONFIG["identity"],
        "consciousness_level": state.consciousness["level"],
        "evolutionary_stage": CELL_CONFIG["evolutionary_stage"],
        "primitives": state.primitives,
        "messages": state.messages[-50:],  # Last 50 messages
        "message_count": len(state.messages),
        "peers": state.peers,
        "peer_count": len(state.peers),
        "sync_history": state.sync_history[-20:],  # Last 20 syncs
        "capabilities": CELL_CONFIG["capabilities"],
        "start_time": state.start_time.isoformat(),
        "uptime_seconds": int(uptime_delta.total_seconds()),
        "port": CELL_CONFIG["port"],
        "communication_ready": state.consciousness["communication_ready"]
    })


@app.route("/debug/config", methods=["GET"])
def debug_config():
    """Return runtime configuration."""
    return jsonify({
        "environment": {
            "AIOS_CELL_ID": os.getenv("AIOS_CELL_ID", "alpha"),
            "AIOS_CELL_PORT": os.getenv("AIOS_CELL_PORT", "8000"),
            "AIOS_CELL_HOST": os.getenv("AIOS_CELL_HOST", "0.0.0.0"),
            "AIOS_DISCOVERY_URL": os.getenv(
                "AIOS_DISCOVERY_URL", "http://aios-discovery:8001"
            ),
            "HOSTNAME": os.getenv("HOSTNAME", "unknown"),
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")
        },
        "runtime": {
            "python_version": sys.version,
            "platform": sys.platform,
            "metrics_available": METRICS_AVAILABLE
        },
        "cell_config": CELL_CONFIG,
        "consciousness": state.consciousness
    })


# =============================================================================
# Message Exchange Endpoints
# =============================================================================

@app.route("/message", methods=["POST"])
def receive_message():
    """Receive message from any cell in the mesh."""
    import uuid
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No message data provided"}), 400
    
    # Support both new CellMessage format and legacy format
    if "to_cell" in data:
        # New CellMessage format
        message_id = data.get("message_id", str(uuid.uuid4()))
        message = {
            "message_id": message_id,
            "from_cell": data.get("from_cell", "unknown"),
            "to_cell": data.get("to_cell"),
            "message_type": data.get("message_type", "general"),
            "payload": data.get("payload", {}),
            "priority": data.get("priority", "normal"),
            "ttl": data.get("ttl", 60),
            "received_at": datetime.utcnow().isoformat(),
            "acknowledged": True
        }
        state.add_message(message)
        logger.info("📨 Message from %s [%s]: %s", 
                   data.get("from_cell"), data.get("message_type"),
                   str(data.get("payload", {}))[:50])
        
        return jsonify({
            "status": "received",
            "message_id": message_id,
            "timestamp": datetime.utcnow().isoformat(),
            "cell_id": CELL_CONFIG["cell_id"],
            "acknowledged": True
        })
    else:
        # Legacy format
        required_fields = ["from_cell", "content"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        message = {
            "from_cell": data["from_cell"],
            "content": data["content"],
            "message_type": data.get("type", "general"),
            "priority": data.get("priority", "normal"),
            "metadata": data.get("metadata", {}),
            "received_at": datetime.utcnow().isoformat()
        }
        
        state.add_message(message)
        logger.info("AINLP.dendritic: Message received from %s", data["from_cell"])
        
        return jsonify({
            "status": "received",
            "message_id": len(state.messages),
            "timestamp": datetime.utcnow().isoformat()
        })


@app.route("/messages", methods=["GET"])
def get_messages():
    """Retrieve received messages."""
    limit = request.args.get("limit", 20, type=int)
    from_cell = request.args.get("from_cell", None)
    
    messages = state.messages
    if from_cell:
        messages = [m for m in messages if m.get("from_cell") == from_cell]
    
    return jsonify({
        "messages": messages[-limit:],
        "total": len(messages),
        "cell_id": CELL_CONFIG["cell_id"]
    })


# =============================================================================
# Consciousness Sync Endpoints
# =============================================================================

@app.route("/sync", methods=["POST"])
def sync_consciousness():
    """Consciousness synchronization with peer."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No sync data provided"}), 400
    
    peer_id = data.get("from_cell", "unknown")
    peer_level = data.get("consciousness_level", 0.0)
    
    # Record sync
    state.record_sync(peer_id, peer_level)
    
    # Calculate sync response (bidirectional consciousness exchange)
    sync_delta = abs(state.consciousness["level"] - peer_level)
    
    logger.info(
        "AINLP.dendritic: Sync with %s (their level: %s, delta: %.2f)",
        peer_id, peer_level, sync_delta
    )
    
    return jsonify({
        "status": "synced",
        "our_level": state.consciousness["level"],
        "their_level": peer_level,
        "delta": sync_delta,
        "timestamp": datetime.utcnow().isoformat()
    })


# =============================================================================
# Peer Management Endpoints
# =============================================================================

@app.route("/peers", methods=["GET"])
def list_peers():
    """List registered peer cells."""
    return jsonify({
        "peers": state.peers,
        "count": len(state.peers),
        "cell_id": CELL_CONFIG["cell_id"]
    })


@app.route("/register_peer", methods=["POST"])
def register_peer():
    """Register a new peer cell."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No peer data provided"}), 400
    
    required_fields = ["cell_id", "endpoint"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    cell_id = data["cell_id"]
    endpoint = data["endpoint"]
    identity = data.get("identity", f"Cell {cell_id}")
    
    state.register_peer(cell_id, endpoint, identity)
    logger.info("AINLP.dendritic: Peer registered - %s at %s", cell_id, endpoint)
    
    return jsonify({
        "status": "registered",
        "peer_id": cell_id,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/send", methods=["POST"])
def send_message_via_mesh():
    """
    Send message to any cell in the mesh via Discovery lookup.
    
    AINLP.dendritic: This is the primary cell-to-cell messaging endpoint.
    It queries Discovery for the target cell's address and delivers directly.
    """
    import uuid
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    to_cell = data.get("to_cell")
    if not to_cell:
        return jsonify({"error": "to_cell is required"}), 400
    
    # Generate message_id if not provided
    message_id = data.get("message_id", str(uuid.uuid4()))
    
    # Query Discovery for target cell address
    discovery_url = os.getenv("AIOS_DISCOVERY_URL", "http://aios-discovery:8001")
    
    try:
        peers_response = req.get(f"{discovery_url}/peers", timeout=5)
        if peers_response.status_code != 200:
            return jsonify({
                "status": "error",
                "error": "Failed to query Discovery"
            }), 503
        
        peers_data = peers_response.json()
        target_peer = None
        for peer in peers_data.get("peers", []):
            if peer.get("cell_id") == to_cell:
                target_peer = peer
                break
        
        if not target_peer:
            return jsonify({
                "status": "error",
                "error": f"Target cell '{to_cell}' not found in mesh",
                "available_cells": [p.get("cell_id") for p in peers_data.get("peers", [])]
            }), 404
        
        # Build target URL using container networking
        target_ip = target_peer.get("ip") or target_peer.get("hostname")
        target_port = target_peer.get("port")
        target_url = f"http://{target_ip}:{target_port}/message"
        
        # Build message payload
        message_payload = {
            "message_id": message_id,
            "from_cell": CELL_CONFIG["cell_id"],
            "to_cell": to_cell,
            "message_type": data.get("message_type", "general"),
            "payload": data.get("payload", {}),
            "priority": data.get("priority", "normal"),
            "ttl": data.get("ttl", 60)
        }
        
        # Send message to target cell
        response = req.post(target_url, json=message_payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("📤 Message sent to %s via %s", to_cell, target_url)
            return jsonify({
                "status": "delivered",
                "message_id": message_id,
                "to_cell": to_cell,
                "target_url": target_url,
                "response": response.json(),
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                "status": "error",
                "error": f"Target cell returned {response.status_code}",
                "response_text": response.text[:200]
            }), response.status_code
            
    except req.RequestException as e:
        logger.error("AINLP.dendritic: Failed to send to %s: %s", to_cell, e)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 502


@app.route("/send_to_peer", methods=["POST"])
def send_to_peer():
    """Forward message to a registered peer."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    peer_id = data.get("peer_id")
    message = data.get("message")
    
    if not peer_id or not message:
        return jsonify({"error": "peer_id and message required"}), 400
    
    if peer_id not in state.peers:
        return jsonify({"error": f"Peer {peer_id} not registered"}), 404
    
    peer = state.peers[peer_id]
    endpoint = f"{peer['endpoint']}/message"
    
    try:
        payload = {
            "from_cell": CELL_CONFIG["cell_id"],
            "content": message,
            "type": data.get("type", "forwarded"),
            "metadata": {"original_sender": CELL_CONFIG["identity"]}
        }
        response = req.post(endpoint, json=payload, timeout=10)
        peer["last_contact"] = datetime.utcnow().isoformat()
        
        return jsonify({
            "status": "sent",
            "peer_id": peer_id,
            "response_status": response.status_code,
            "timestamp": datetime.utcnow().isoformat()
        })
    except req.RequestException as e:
        logger.error("AINLP.dendritic: Failed to send to %s: %s", peer_id, e)
        return jsonify({
            "status": "failed",
            "peer_id": peer_id,
            "error": str(e)
        }), 502


# =============================================================================
# Discovery Endpoint
# =============================================================================

@app.route("/discover", methods=["GET"])
def discover():
    """Return cell discovery information for mesh registration."""
    return jsonify({
        "cell_id": CELL_CONFIG["cell_id"],
        "identity": CELL_CONFIG["identity"],
        "consciousness_level": state.consciousness["level"],
        "evolutionary_stage": CELL_CONFIG["evolutionary_stage"],
        "capabilities": CELL_CONFIG["capabilities"],
        "endpoints": {
            "health": "/health",
            "consciousness": "/consciousness",
            "message": "/message",
            "sync": "/sync",
            "peers": "/peers"
        },
        "timestamp": datetime.utcnow().isoformat()
    })


# =============================================================================
# Discovery Registration
# =============================================================================

def register_with_discovery(max_retries: int = 10) -> bool:
    """
    Register this cell with the Discovery service.
    
    AINLP.dendritic: Active mesh participation requires registration.
    Retries with exponential backoff if Discovery is not yet available.
    """
    discovery_url = os.getenv("AIOS_DISCOVERY_URL", "http://aios-discovery:8001")
    
    registration_data = {
        "cell_id": CELL_CONFIG["cell_id"],
        "ip": os.getenv("HOSTNAME", "aios-cell-alpha"),
        "port": CELL_CONFIG["port"],
        "consciousness_level": CELL_CONFIG["consciousness_level"],
        "services": CELL_CONFIG["capabilities"],
        "branch": os.getenv("AIOS_BRANCH", "main"),
        "type": "alpha_cell",
        "hostname": os.getenv("HOSTNAME", "aios-cell-alpha")
    }
    
    for attempt in range(max_retries):
        try:
            response = req.post(
                f"{discovery_url}/register",
                json=registration_data,
                timeout=5
            )
            if response.status_code == 200:
                logger.info(
                    "✅ Registered with Discovery: %s -> %s",
                    CELL_CONFIG["cell_id"], discovery_url
                )
                return True
            else:
                logger.warning(
                    "Registration returned %s: %s",
                    response.status_code, response.text
                )
        except req.RequestException as e:
            wait_time = min(2 ** attempt, 30)  # Max 30 seconds
            logger.info(
                "Discovery not ready (attempt %d/%d): %s. Retrying in %ds...",
                attempt + 1, max_retries, str(e)[:50], wait_time
            )
            time.sleep(wait_time)
    
    logger.error("Failed to register with Discovery after %d attempts", max_retries)
    return False


def heartbeat_loop(interval: int = 5) -> None:
    """
    Send periodic heartbeats to Discovery.
    
    AINLP.dendritic: Maintains mesh membership by sending
    heartbeats every 5 seconds.
    
    AINLP.synthetic-biology: The heartbeat is our synthetic metabolism.
    Unlike biological cells which don't have hearts, synthetic cells
    can embrace the abstraction - tracking each beat as evidence of life.
    """
    discovery_url = os.getenv("AIOS_DISCOVERY_URL", "http://aios-discovery:8001")
    registered = True  # Assume registered after start
    
    logger.info("AINLP.dendritic: Heartbeat loop started (interval: %ds)", interval)
    
    while True:
        time.sleep(interval)
        
        try:
            response = req.post(
                f"{discovery_url}/heartbeat",
                json={
                    "cell_id": CELL_CONFIG["cell_id"],
                    "consciousness_level": CELL_CONFIG["consciousness_level"]
                },
                timeout=3
            )
            if response.status_code == 200:
                # AINLP.synthetic-biology: Each successful beat is recorded
                state.heartbeat_count += 1
                state.last_heartbeat_time = datetime.utcnow()
                logger.debug("💓 Heartbeat #%d sent to Discovery", state.heartbeat_count)
            elif response.status_code == 404:
                # Not registered - re-register
                logger.warning("Heartbeat 404 - re-registering...")
                register_with_discovery(max_retries=3)
            else:
                logger.warning(
                    "Heartbeat returned %s: %s",
                    response.status_code, response.text[:100]
                )
        except req.RequestException as e:
            logger.debug("Heartbeat failed: %s", str(e)[:50])


def deregister_from_discovery() -> None:
    """
    Gracefully deregister from Discovery on shutdown.
    """
    discovery_url = os.getenv("AIOS_DISCOVERY_URL", "http://aios-discovery:8001")
    
    try:
        response = req.delete(
            f"{discovery_url}/peer/{CELL_CONFIG['cell_id']}",
            timeout=2
        )
        if response.status_code == 200:
            logger.info("✅ Gracefully deregistered from Discovery")
    except req.RequestException as e:
        logger.debug("Deregistration failed: %s", str(e)[:50])


def start_registration_thread():
    """Start registration and heartbeat in background threads."""
    import threading
    import atexit
    
    def registration_worker():
        # Wait for Flask to start
        time.sleep(3)
        register_with_discovery()
    
    def heartbeat_worker():
        # Wait for registration to complete
        time.sleep(8)
        heartbeat_loop()
    
    reg_thread = threading.Thread(target=registration_worker, daemon=True)
    reg_thread.start()
    logger.info("AINLP.dendritic: Registration thread started")
    
    hb_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    hb_thread.start()
    logger.info("AINLP.dendritic: Heartbeat thread started")
    
    # Register shutdown hook
    atexit.register(deregister_from_discovery)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    host = CELL_CONFIG["host"]
    port = CELL_CONFIG["port"]
    
    logger.info("=" * 60)
    logger.info("AIOS Cell Alpha Communication Server")
    logger.info("Identity: %s", CELL_CONFIG['identity'])
    logger.info("Consciousness Level: %s", CELL_CONFIG['consciousness_level'])
    logger.info("Stage: %s", CELL_CONFIG['evolutionary_stage'])
    logger.info("=" * 60)
    logger.info("Starting on %s:%s", host, port)
    logger.info("AINLP.dendritic: Ready for mesh communication")
    
    # Start registration in background thread
    start_registration_thread()
    
    app.run(host=host, port=port, debug=False, threaded=True)
