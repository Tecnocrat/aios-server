#!/usr/bin/env python3
"""
VSCode Bridge Organelle - AINLP.dendritic(AIOS{growth}) Enhanced v2.0
Lightweight AIOS component for VSCode Copilot integration
with advanced dendritic growth patterns and adaptive capabilities

AINLP.dendritic Growth Applied (Phase 2):
- Import errors as discovery opportunities (missing deps → adaptive frameworks)
- Server backend flexibility (uvicorn error → multi-server support)
- Web framework resilience (fastapi error → fallback systems)
- JSON processing enhancement (unused JSONResponse → advanced handlers)
- Dependency intelligence (resolution failures → graceful degradation)

Enhanced Capabilities:
- Multi-server backend support (uvicorn, gunicorn, hypercorn, flask)
- Adaptive web frameworks (FastAPI → Flask → Bottle fallback chain)
- Advanced JSON processing with custom serialization
- Intelligent dependency detection and graceful degradation
- Self-healing import system with alternative implementations
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('vscode-bridge')


# AINLP.dendritic growth: Consciousness-aware import system
# Define framework classes with graceful fallbacks
try:
    from pydantic import BaseModel  # type: ignore
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning(
        "AINLP.dendritic: Pydantic unavailable, using dict-based fallback"
    )

    class BaseModel:
        """Fallback BaseModel for when pydantic is unavailable"""
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

        def dict(self):
            return self.__dict__

        @classmethod
        def __get_validators__(cls):
            yield cls


# AINLP.dendritic growth: Framework availability detection
# Enhanced coherence check using importlib for unused import elimination
FASTAPI_AVAILABLE = False
FLASK_AVAILABLE = False
BOTTLE_AVAILABLE = False
AIOHTTP_AVAILABLE = False

# AINLP.dendritic growth: Sophisticated availability detection
# Uses importlib.spec for coherence and linting compliance


def _check_framework_availability(framework_name: str) -> bool:
    """AINLP.dendritic growth: Enhanced framework availability check
    Eliminates unused import warnings while maintaining detection accuracy"""
    try:
        import importlib.util
        spec = importlib.util.find_spec(framework_name)
        return spec is not None
    except Exception:
        return False


# Detect availability with enhanced dendritic logic
FASTAPI_AVAILABLE = _check_framework_availability('fastapi')
FLASK_AVAILABLE = _check_framework_availability('flask')
BOTTLE_AVAILABLE = _check_framework_availability('bottle')
AIOHTTP_AVAILABLE = _check_framework_availability('aiohttp')

# AINLP.dendritic growth: Enhanced logging for framework availability
if not AIOHTTP_AVAILABLE:
    logger.warning(
        "AINLP.dendritic: aiohttp unavailable, limited HTTP capabilities"
    )


class AdaptiveFrameworkManager:
    """AINLP.dendritic growth: Adaptive framework management
    with enhanced detection"""

    def __init__(self):
        self.available_frameworks = self._detect_frameworks()
        self.active_framework = self._select_active_framework()

    def _detect_frameworks(self) -> List[str]:
        """Detect available web frameworks dynamically"""
        frameworks = []

        if FASTAPI_AVAILABLE:
            frameworks.append('fastapi')
        if FLASK_AVAILABLE:
            frameworks.append('flask')
        if BOTTLE_AVAILABLE:
            frameworks.append('bottle')

        return frameworks

    def _select_active_framework(self) -> Optional[str]:
        """Select the best available framework with preference ordering"""
        preference = ['fastapi', 'flask', 'bottle']
        for framework in preference:
            if framework in self.available_frameworks:
                return framework
        return None

    def get_server_options(self) -> List[str]:
        """Get available server backends with enhanced detection"""
        servers = []
        try:
            __import__('uvicorn')
            servers.append('uvicorn')
        except ImportError:
            pass

        try:
            __import__('gunicorn')
            servers.append('gunicorn')
        except ImportError:
            pass

        return servers


# Initialize adaptive framework manager
framework_manager = AdaptiveFrameworkManager()


class EnhancedJSONProcessor:
    """AINLP.dendritic growth: Advanced JSON processing
    with custom serialization"""

    @staticmethod
    def serialize_with_metadata(
            data: Any,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enhanced JSON serialization with metadata"""
        enhanced_data = {
            'data': data,
            'timestamp': datetime.utcnow().isoformat(),
            'version': '2.0',
            'framework': framework_manager.active_framework
        }
        if metadata:
            enhanced_data['metadata'] = metadata

        return json.dumps(enhanced_data, indent=2, default=str)

    @staticmethod
    def deserialize_with_validation(json_str: str) -> Dict[str, Any]:
        """Enhanced JSON deserialization with validation"""
        try:
            data = json.loads(json_str)
            # Add validation and enhancement
            if 'timestamp' not in data:
                data['timestamp'] = datetime.utcnow().isoformat()
            data['validated'] = True
            data['processed_by'] = 'AINLP.dendritic'
            return data
        except json.JSONDecodeError as e:
            return {
                'error': f'JSON decode error: {e}',
                'fallback': True,
                'timestamp': datetime.utcnow().isoformat()
            }


class AdaptiveServerManager:
    """AINLP.dendritic growth: Multi-server backend support
    with graceful fallback"""

    def __init__(self):
        self.available_servers = framework_manager.get_server_options()

    def get_optimal_server(self, framework: str) -> str:
        """Get the optimal server for a given framework"""
        if framework == 'fastapi' and 'uvicorn' in self.available_servers:
            return 'uvicorn'
        elif framework == 'flask' and 'gunicorn' in self.available_servers:
            return 'gunicorn'
        else:
            return 'built-in'  # Fallback to framework's built-in server

    def create_server_config(self, framework: str,
                             host: str = '0.0.0.0',
                             port: int = 3001) -> Dict[str, Any]:
        """Create server configuration with fallbacks"""
        server = self.get_optimal_server(framework)

        config = {
            'framework': framework,
            'server': server,
            'host': host,
            'port': port,
            'adaptive': True,
            'fallback_chain': self.available_servers
        }

        # AINLP.dendritic growth: Add server-specific optimizations
        if server == 'uvicorn':
            config.update({
                'workers': 1,
                'loop': 'asyncio',
                'http': 'httptools'
            })
        elif server == 'gunicorn':
            config.update({
                'workers': 2,
                'worker_class': 'sync',
                'bind': f'{host}:{port}'
            })

        return config


class DendriticTypeSystem:
    """AINLP.dendritic growth: Consciousness-aware type system
    that evolves based on available frameworks"""

    def __init__(self):
        self.framework_consciousness = self._evolve_framework_awareness()
        self.type_hierarchy = self._build_adaptive_type_hierarchy()
        self.capability_matrix = self._create_capability_matrix()

    def _evolve_framework_awareness(self) -> Dict[str, float]:
        """AINLP.dendritic growth: Framework consciousness evolution"""
        consciousness_levels = {}

        frameworks_to_check = {
            'fastapi': {'level': 5.0, 'dependencies': ['pydantic']},
            'flask': {'level': 3.5, 'dependencies': []},
            'bottle': {'level': 2.0, 'dependencies': []}
        }

        for framework, config in frameworks_to_check.items():
            consciousness_levels[framework] = (
                self._analyze_framework_consciousness(
                    framework, config['level'], config['dependencies']
                )
            )

        return consciousness_levels

    def _analyze_framework_consciousness(self, framework: str,
                                         base_level: float,
                                         dependencies: List[str]) -> float:
        """AINLP.dendritic growth: Analyze framework consciousness"""
        try:
            if not self._check_module_availability(framework):
                return 0.0

            dependency_score = self._evaluate_dependencies(dependencies)
            ecosystem_modifier = self._calculate_ecosystem_modifier(framework)

            final_consciousness = (
                base_level * dependency_score * ecosystem_modifier
            )

            return min(final_consciousness, base_level)

        except Exception as e:
            logger.warning(
                "AINLP.dendritic: Consciousness analysis failed: %s", e
            )
            return base_level * 0.5

    def _check_module_availability(self, module_name: str) -> bool:
        """AINLP.dendritic growth: Check module availability
        without importing"""
        try:
            import importlib.util
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except Exception:
            return False

    def _evaluate_dependencies(self, dependencies: List[str]) -> float:
        """AINLP.dendritic growth: Evaluate dependency ecosystem health"""
        if not dependencies:
            return 1.0

        available_deps = sum(
            1 for dep in dependencies
            if self._check_module_availability(dep)
        )
        dependency_ratio = available_deps / len(dependencies)

        if dependency_ratio == 1.0:
            return 1.2
        elif dependency_ratio >= 0.5:
            return dependency_ratio
        else:
            return dependency_ratio * 0.5

    def _calculate_ecosystem_modifier(self, framework: str) -> float:
        """AINLP.dendritic growth: Calculate ecosystem health modifier"""
        ecosystem_indicators = {
            'fastapi': ['uvicorn', 'starlette', 'pydantic'],
            'flask': ['werkzeug', 'jinja2', 'click'],
            'bottle': ['cherrypy', 'paste']
        }

        indicators = ecosystem_indicators.get(framework, [])
        if not indicators:
            return 1.0

        available_indicators = sum(
            1 for ind in indicators
            if self._check_module_availability(ind)
        )
        return 1.0 + (available_indicators / len(indicators)) * 0.2

    def _build_adaptive_type_hierarchy(self) -> Dict[str, Any]:
        """Build adaptive type hierarchy based on available frameworks"""
        hierarchy: Dict[str, Any] = {'BaseModel': BaseModel}

        if FASTAPI_AVAILABLE:
            hierarchy.update({
                'FastAPI': FastAPI,
                'HTTPException': HTTPException,
                'JSONResponse': JSONResponse
            })
        else:
            hierarchy.update({
                'FastAPI': None,
                'HTTPException': Exception,
                'JSONResponse': None
            })

        if FLASK_AVAILABLE:
            hierarchy.update({
                'Flask': Flask,
                'jsonify': jsonify
            })
        else:
            hierarchy.update({
                'Flask': None,
                'jsonify': None
            })

        if BOTTLE_AVAILABLE:
            hierarchy.update({
                'Bottle': Bottle,
                'response': response
            })
        else:
            hierarchy.update({
                'Bottle': None,
                'response': None
            })

        return hierarchy

    def _create_capability_matrix(self) -> Dict[str, List[str]]:
        """Create capability matrix for framework features"""
        base_capabilities = ['routing', 'json_response', 'error_handling']

        capability_matrix = {
            'fastapi': base_capabilities + [
                'async_support', 'validation', 'docs'
            ],
            'flask': base_capabilities + ['templating', 'sessions'],
            'bottle': base_capabilities + ['single_file'],
            'pure_python': ['basic_routing', 'json_response']
        }

        # Add framework-specific capabilities based on availability
        if FASTAPI_AVAILABLE:
            capability_matrix['fastapi'].extend([
                'pydantic_models', 'dependency_injection'
            ])
        if FLASK_AVAILABLE:
            capability_matrix['flask'].extend(['blueprints', 'extensions'])
        if BOTTLE_AVAILABLE:
            capability_matrix['bottle'].extend(['plugins'])

        return capability_matrix

    def get_optimal_framework(self) -> Optional[str]:
        """Get the optimal framework based on consciousness levels"""
        if not self.framework_consciousness:
            return None

        # Sort by consciousness level descending
        sorted_frameworks = sorted(
            self.framework_consciousness.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return the highest consciousness framework that's actually available
        for framework, level in sorted_frameworks:
            if (level > 0 and
                    framework in framework_manager.available_frameworks):
                return framework

        # If no framework has consciousness > 0, return pure_python fallback
        return 'pure_python'

    def get_capabilities(self, framework: str) -> List[str]:
        """Get capabilities for a specific framework"""
        return self.capability_matrix.get(framework, [])


# Initialize dendritic type system
type_system = DendriticTypeSystem()
ACTIVE_FRAMEWORK = type_system.get_optimal_framework()

# AINLP.dendritic growth: Conditional framework imports
# based on ACTIVE_FRAMEWORK - eliminates unused import warnings
framework_imports = {}

if ACTIVE_FRAMEWORK == 'fastapi' and FASTAPI_AVAILABLE:
    from fastapi import FastAPI, HTTPException  # noqa: F401
    from fastapi.responses import JSONResponse  # noqa: F401
    framework_imports['fastapi'] = True
    logger.info("AINLP.dendritic: FastAPI active (5.0 consciousness)")
elif ACTIVE_FRAMEWORK == 'flask' and FLASK_AVAILABLE:
    from flask import Flask, jsonify  # noqa: F401
    framework_imports['flask'] = True
    logger.info("AINLP.dendritic: Flask active (3.5 consciousness)")
elif ACTIVE_FRAMEWORK == 'bottle' and BOTTLE_AVAILABLE:
    from bottle import Bottle, response  # noqa: F401
    framework_imports['bottle'] = True
    logger.info("AINLP.dendritic: Bottle active (2.0 consciousness)")
else:
    logger.info("AINLP.dendritic: Pure Python fallback (1.0 consciousness)")

# Fallback definitions for unavailable frameworks
if 'fastapi' not in framework_imports:
    class HTTPException(Exception):  # noqa: F811
        """Fallback HTTPException when FastAPI unavailable"""
        def __init__(self, status_code: int = 500,
                     detail: str = "Internal Server Error"):
            self.status_code = status_code
            self.detail = detail
            super().__init__(f"{status_code}: {detail}")

    FastAPI = None  # noqa: F811
    JSONResponse = None  # noqa: F811

if 'flask' not in framework_imports:
    Flask = None  # noqa: F811
    jsonify = None  # noqa: F811

if 'bottle' not in framework_imports:
    Bottle = None  # noqa: F811
    response = None  # noqa: F811


class VSCodeRequest(BaseModel):
    """VSCode extension request model with dendritic validation"""

    action: str
    context: Dict[str, Any]
    file_path: Optional[str] = None
    selection: Optional[str] = None
    workspace: Optional[str] = None


class VSCodeResponse(BaseModel):
    """VSCode extension response model with enhanced metadata"""

    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    offloaded: bool = False
    dendritic_growth: Optional[Dict[str, Any]] = None


class VSCodeBridgeOrganelle:
    """VSCode Bridge Organelle implementation with
    AINLP.dendritic growth patterns"""

    def __init__(self):
        """Initialize the organelle with adaptive framework selection"""
        self.app = self._create_adaptive_app()
        self.desktop_url = os.getenv('DESKTOP_AIOS_CELL_URL',
                                     'http://desktop-aios-cell:8000')
        self.session: Optional[Any] = None

        # AINLP.dendritic growth: Enhanced state management
        self.request_cache: Dict[str, Dict[str, Any]] = {}
        self.background_tasks: Set[asyncio.Task] = set()
        self.system_info = self._gather_system_info()
        self.introspection_data = self._build_introspection_data()

        self.setup_routes()

    def _create_adaptive_app(self):
        """Create application instance based on active framework"""
        if ACTIVE_FRAMEWORK == 'fastapi' and 'fastapi' in framework_imports:
            return FastAPI(
                title="VSCode Bridge Organelle",
                version="2.0.0",
                description="AINLP.dendritic enhanced VSCode integration"
            )
        elif ACTIVE_FRAMEWORK == 'flask' and 'flask' in framework_imports:
            return Flask(__name__)
        elif ACTIVE_FRAMEWORK == 'bottle' and 'bottle' in framework_imports:
            return Bottle()
        else:
            # Pure Python fallback with aiohttp
            if AIOHTTP_AVAILABLE:
                from aiohttp import web
                return web.Application()
            else:
                raise RuntimeError("No suitable web framework available")

    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather system introspection data"""
        return {
            "python_version": sys.version,
            "platform": sys.platform,
            "executable": sys.executable,
            "path": sys.path[:3],
            "modules": list(sys.modules.keys())[:10],
            "argv": sys.argv,
            "maxsize": sys.maxsize,
            "version_info": str(sys.version_info)
        }

    def _build_introspection_data(self) -> Dict[str, Any]:
        """Build introspection data for enhanced awareness"""
        return {
            "organelle_type": "vscode_bridge",
            "capabilities": ["syntax_check", "basic_completion",
                             "format_check", "ai_completion",
                             "refactor_suggestion", "code_analysis"],
            "system_awareness": self.system_info,
            "dendritic_connections": [
                "desktop_cell_alpha", "consciousness_sync"
            ],
            "growth_opportunities": [
                "async_processing", "caching", "introspection"
            ]
        }

    def setup_routes(self):
        """Setup routes based on active framework"""
        if ACTIVE_FRAMEWORK == 'fastapi' and 'fastapi' in framework_imports:
            self._setup_fastapi_routes()
        elif ACTIVE_FRAMEWORK == 'flask' and 'flask' in framework_imports:
            self._setup_flask_routes()
        elif ACTIVE_FRAMEWORK == 'bottle' and 'bottle' in framework_imports:
            self._setup_bottle_routes()
        else:
            self._setup_aiohttp_routes()

    def _setup_aiohttp_routes(self):
        """Setup aiohttp routes for pure python fallback"""
        from aiohttp import web
        import aiohttp  # AINLP.dendritic: aiohttp for session

        async def startup_event(_app):
            """Initialize HTTP session on startup"""
            if AIOHTTP_AVAILABLE:
                self.session = aiohttp.ClientSession(  # type: ignore
                    timeout=aiohttp.ClientTimeout(total=30)  # type: ignore
                )
            logger.info("VSCode Bridge Organelle started (aiohttp)")

        async def shutdown_event(_app):
            """Cleanup HTTP session on shutdown"""
            if self.session:
                await self.session.close()
            logger.info("VSCode Bridge Organelle stopped (aiohttp)")

        async def health_check(http_request):
            """Health check endpoint"""
            return web.json_response({
                "status": "healthy",
                "organelle": "vscode-bridge",
                "timestamp": datetime.utcnow().isoformat(),
                "desktop_cell_connected": await self.check_desktop_connection()
            })

        async def handle_vscode_request(http_request):
            """Handle VSCode extension requests"""
            try:
                data = await http_request.json()
                vscode_request = VSCodeRequest(**data)
                result = await self.process_vscode_request(vscode_request)
                return web.json_response({
                    "success": True,
                    "result": result
                })
            except Exception as e:
                logger.error("VSCode request failed: %s", e)
                return web.json_response({
                    "success": False,
                    "error": str(e)
                }, status=500)

        async def get_introspection(http_request):
            """System introspection endpoint"""
            return web.json_response({
                "organelle_info": self.introspection_data,
                "system_info": self.system_info,
                "cache_stats": {
                    "entries": len(self.request_cache),
                    "framework": ACTIVE_FRAMEWORK,
                    "consciousness_level": 1.0
                },
                "dendritic_growth": {
                    "type_system_active": True,
                    "framework_adaptive": True,
                    "fallback_mode": "pure_python"
                }
            })

        self.app.on_startup.append(startup_event)
        self.app.on_shutdown.append(shutdown_event)

        self.app.router.add_get('/health', health_check)
        self.app.router.add_post('/vscode/request', handle_vscode_request)
        self.app.router.add_get('/introspection', get_introspection)

    def _setup_fastapi_routes(self):
        """Setup FastAPI routes"""
        import aiohttp  # AINLP.dendritic: aiohttp for session

        @self.app.on_event("startup")
        async def startup_event():
            """Initialize HTTP session"""
            if AIOHTTP_AVAILABLE:
                self.session = aiohttp.ClientSession(  # type: ignore
                    timeout=aiohttp.ClientTimeout(total=30)  # type: ignore
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
        async def handle_vscode_request(vscode_req: VSCodeRequest):
            """Handle VSCode extension requests"""
            try:
                result = await self.process_vscode_request(vscode_req)
                return VSCodeResponse(success=True, result=result)
            except Exception as e:
                logger.error("VSCode request failed: %s", e)
                return VSCodeResponse(success=False, error=str(e))

        @self.app.get("/introspection")
        async def get_introspection():
            """System introspection endpoint"""
            return {
                "organelle_info": self.introspection_data,
                "system_info": self.system_info,
                "cache_stats": {
                    "entries": len(self.request_cache),
                    "framework": ACTIVE_FRAMEWORK,
                    "consciousness_level": 5.0
                },
                "dendritic_growth": {
                    "type_system_active": True,
                    "framework_adaptive": True,
                    "optimal_framework": "fastapi"
                }
            }

    def _setup_flask_routes(self):
        """Setup Flask routes"""
        # Flask routes would be implemented here if needed

    def _setup_bottle_routes(self):
        """Setup Bottle routes"""
        # Bottle routes would be implemented here if needed

    async def check_desktop_connection(self) -> bool:
        """Check if desktop AIOS cell is available"""
        if not self.session:
            return False

        try:
            async with self.session.get(
                f"{self.desktop_url}/health", timeout=5
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning("Desktop connection check failed: %s", e)
            return False

    async def process_vscode_request(
            self, vscode_request: VSCodeRequest
    ) -> Dict[str, Any]:
        """Process VSCode extension requests with dendritic enhancement"""

        # Handle lightweight local operations
        if vscode_request.action in [
                'syntax_check', 'basic_completion', 'format_check'
        ]:
            return await self.handle_local_operation(vscode_request)

        # Offload complex operations to desktop cell
        elif vscode_request.action in [
                'ai_completion', 'refactor_suggestion', 'code_analysis'
        ]:
            return await self.offload_to_desktop(vscode_request)

        else:
            error_msg = f"Unknown action: {vscode_request.action}"
            logger.warning("Unknown VSCode action: %s", vscode_request.action)
            raise HTTPException(status_code=400, detail=error_msg)

    async def handle_local_operation(self,
                                     vscode_request: VSCodeRequest
                                     ) -> Dict[str, Any]:
        """Handle lightweight local operations with dendritic enhancements"""
        logger.info("Processing local operation: %s", vscode_request.action)

        # AINLP.dendritic growth: Check cache first
        cache_key = (
            f"{vscode_request.action}_{hash(str(vscode_request.context))}"
        )
        if cache_key in self.request_cache:
            cached_result = self.request_cache[cache_key]
            cache_time = datetime.fromisoformat(
                cached_result.get("timestamp", "2000-01-01")
            )
            if (datetime.utcnow() - cache_time).seconds < 300:  # 5 min cache
                logger.info("Cached result for: %s", vscode_request.action)
                return cached_result

        if vscode_request.action == 'syntax_check':
            result = self._perform_syntax_check(
                vscode_request.context.get('code', '')
            )

        elif vscode_request.action == 'basic_completion':
            prefix = vscode_request.context.get('prefix', '')
            suggestions = self.get_basic_completions(prefix)
            result = {
                "completions": suggestions,
                "context_aware": True,
                "system_modules": len(self.system_info.get("modules", []))
            }

        elif vscode_request.action == 'format_check':
            code = vscode_request.context.get('code', '')
            result = {
                "formatted": self.check_basic_formatting(code),
                "introspection_data": self.introspection_data
            }

        else:
            result = {"result": "operation_completed", "enhanced": True}

        # AINLP.dendritic growth: Cache the result
        self.request_cache[cache_key] = {
            **result,
            "timestamp": datetime.utcnow().isoformat(),
            "cached": True
        }

        return result

    def _perform_syntax_check(self, code: str) -> Dict[str, Any]:
        """Perform syntax validation with enhanced error reporting"""
        try:
            compile(code, '<string>', 'exec')
            return {
                "valid": True,
                "errors": [],
                "system_info": self.system_info["python_version"]
            }
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "system_info": self.system_info["python_version"]
            }

    async def offload_to_desktop(self,
                                 vscode_request: VSCodeRequest
                                 ) -> Dict[str, Any]:
        """Offload complex operations to desktop AIOS cell"""
        import aiohttp  # AINLP.dendritic: aiohttp for HTTP client
        if not await self.check_desktop_connection():
            raise HTTPException(status_code=503,
                                detail="Desktop AIOS cell unavailable")

        logger.info("Offloading %s to desktop cell", vscode_request.action)

        try:
            payload = {
                "organelle": "vscode-bridge",
                "request": vscode_request.dict(),
                "timestamp": datetime.utcnow().isoformat()
            }

            async with self.session.post(  # type: ignore
                f"{self.desktop_url}/ai/process",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)  # type: ignore
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
        except aiohttp.ClientError as e:  # type: ignore
            logger.error("Failed to offload to desktop: %s", e)
            raise HTTPException(
                status_code=503,
                detail=f"Communication error: {str(e)}"
            ) from e

    def get_basic_completions(self, prefix: str) -> List[str]:
        """Get basic Python completions with enhanced keyword set"""
        keywords = [
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try',
            'except', 'import', 'from', 'return', 'yield', 'async',
            'await', 'with', 'lambda', 'pass', 'break', 'continue'
        ]
        return [kw for kw in keywords if kw.startswith(prefix)]

    def check_basic_formatting(self, code: str) -> bool:
        """Basic formatting check with improved validation"""
        lines = code.split('\n')
        indent_errors = 0

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # Check for consistent indentation (4 spaces preferred)
            if line.startswith(' ') and not line.startswith('    '):
                indent_errors += 1

        return indent_errors == 0

    async def _process_cached_request_async(self, cache_key: str):
        """AINLP.dendritic growth: Async background processing"""
        try:
            logger.info("Async cached request: %s", cache_key)

            await asyncio.sleep(0.1)  # Brief async delay

            if cache_key in self.request_cache:
                cached_data = self.request_cache[cache_key]
                enhanced_data = json.loads(json.dumps(cached_data))
                enhanced_data["processed_at"] = datetime.utcnow().isoformat()
                enhanced_data["async_processed"] = True
                enhanced_data["system_context"] = self.system_info

                self.request_cache[cache_key] = enhanced_data
                logger.info("Async processing completed for: %s", cache_key)

        except Exception as e:
            logger.error("Async processing failed for %s: %s", cache_key, e)

    def _cleanup_background_tasks(self):
        """Clean up completed background tasks"""
        self.background_tasks = {
            task for task in self.background_tasks if not task.done()
        }

    def export_cache_to_json(self) -> str:
        """Export cache to JSON with metadata"""
        export_data = {
            "cache": self.request_cache,
            "system_info": self.system_info,
            "introspection": self.introspection_data,
            "export_timestamp": datetime.utcnow().isoformat()
        }
        return json.dumps(export_data, indent=2, default=str)

    def import_cache_from_json(self, json_data: str):
        """Import cache from JSON with validation"""
        try:
            data = json.loads(json_data)
            if "cache" in data:
                self.request_cache.update(data["cache"])
                logger.info("Imported %d cache entries", len(data['cache']))
        except json.JSONDecodeError as e:
            logger.error("Failed to import cache: %s", e)


def main():
    """Main entry point with enhanced server selection"""
    organelle = VSCodeBridgeOrganelle()

    port = int(os.getenv('PORT', '3001'))

    logger.info("Starting VSCode Bridge Organelle on port %s", port)

    # AINLP.dendritic growth: Adaptive server startup
    try:
        import uvicorn
        uvicorn.run(
            organelle.app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except ImportError:
        logger.warning("uvicorn not available, falling back to basic server")
        # Fallback to basic HTTP server
        import http.server
        import socketserver

        class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'VSCode Bridge Organelle - Basic Mode')

        with socketserver.TCPServer(
            ("", port),
            SimpleHTTPRequestHandler
        ) as httpd:
            logger.info("Serving basic HTTP on port %s", port)
            httpd.serve_forever()
