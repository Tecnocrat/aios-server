#!/usr/bin/env python3
"""
AIOS Cell Prometheus Metrics Module
AINLP.dendritic: Shared metrics exposition for all AIOS cells

Provides standardized Prometheus metrics format for consciousness monitoring.
Each cell imports this module to expose real-time metrics.
"""

import time
from typing import Any, Dict, Optional


def format_prometheus_metrics(
    cell_id: str,
    consciousness_level: float,
    primitives: Optional[Dict[str, float]] = None,
    extra_metrics: Optional[Dict[str, float]] = None,
    labels: Optional[Dict[str, str]] = None
) -> str:
    """
    Format cell metrics in Prometheus exposition format.
    
    Args:
        cell_id: Unique cell identifier (alpha, nous, discovery)
        consciousness_level: Current consciousness level (0.0-5.0)
        primitives: Dict of consciousness primitives {awareness, adaptation, coherence, momentum}
        extra_metrics: Additional custom metrics
        labels: Extra labels to add to all metrics
    
    Returns:
        Prometheus-formatted metrics string
    """
    # Build label string
    base_labels = {"cell_id": cell_id}
    if labels:
        base_labels.update(labels)
    label_str = ",".join(f'{k}="{v}"' for k, v in base_labels.items())
    
    # Default primitives
    if primitives is None:
        primitives = {
            "awareness": consciousness_level * 0.9,
            "adaptation": 0.5,
            "coherence": 0.8,
            "momentum": 0.3
        }
    
    lines = [
        "# HELP aios_cell_consciousness_level Current cell consciousness level (0.0-5.0)",
        "# TYPE aios_cell_consciousness_level gauge",
        f"aios_cell_consciousness_level{{{label_str}}} {consciousness_level:.4f}",
        "",
        "# HELP aios_cell_awareness Consciousness awareness primitive",
        "# TYPE aios_cell_awareness gauge",
        f"aios_cell_awareness{{{label_str}}} {primitives.get('awareness', 0.0):.4f}",
        "",
        "# HELP aios_cell_adaptation Consciousness adaptation speed",
        "# TYPE aios_cell_adaptation gauge",
        f"aios_cell_adaptation{{{label_str}}} {primitives.get('adaptation', 0.0):.4f}",
        "",
        "# HELP aios_cell_coherence Consciousness coherence level",
        "# TYPE aios_cell_coherence gauge",
        f"aios_cell_coherence{{{label_str}}} {primitives.get('coherence', 0.0):.4f}",
        "",
        "# HELP aios_cell_momentum Consciousness growth momentum",
        "# TYPE aios_cell_momentum gauge",
        f"aios_cell_momentum{{{label_str}}} {primitives.get('momentum', 0.0):.4f}",
        "",
        "# HELP aios_cell_up Cell availability (1=up, 0=down)",
        "# TYPE aios_cell_up gauge",
        f"aios_cell_up{{{label_str}}} 1",
        "",
        "# HELP aios_cell_scrape_timestamp_seconds Time of last metrics scrape",
        "# TYPE aios_cell_scrape_timestamp_seconds gauge",
        f"aios_cell_scrape_timestamp_seconds{{{label_str}}} {time.time():.0f}",
    ]
    
    # Add extra metrics
    if extra_metrics:
        lines.append("")
        for metric_name, value in extra_metrics.items():
            safe_name = metric_name.replace("-", "_").replace(" ", "_").lower()
            lines.extend([
                f"# HELP aios_cell_{safe_name} Custom cell metric: {metric_name}",
                f"# TYPE aios_cell_{safe_name} gauge",
                f"aios_cell_{safe_name}{{{label_str}}} {value:.4f}",
                "",
            ])
    
    return "\n".join(lines) + "\n"


def format_mesh_metrics(
    cells_online: int,
    total_consciousness: float,
    mesh_coherence: str,
    cell_statuses: Optional[Dict[str, Dict[str, Any]]] = None
) -> str:
    """
    Format mesh-level aggregated metrics.
    
    Args:
        cells_online: Number of cells currently online
        total_consciousness: Sum of all cell consciousness levels
        mesh_coherence: Mesh coherence status (COHERENT, DEGRADED, etc.)
        cell_statuses: Dict of cell_id -> {level, status}
    
    Returns:
        Prometheus-formatted mesh metrics string
    """
    coherence_value = 1.0 if mesh_coherence == "COHERENT" else 0.5 if mesh_coherence == "DEGRADED" else 0.0
    avg_consciousness = total_consciousness / cells_online if cells_online > 0 else 0.0
    
    lines = [
        "# HELP aios_mesh_cells_online Number of online cells in mesh",
        "# TYPE aios_mesh_cells_online gauge",
        f"aios_mesh_cells_online {cells_online}",
        "",
        "# HELP aios_mesh_total_consciousness Sum of all cell consciousness levels",
        "# TYPE aios_mesh_total_consciousness gauge",
        f"aios_mesh_total_consciousness {total_consciousness:.4f}",
        "",
        "# HELP aios_mesh_average_consciousness Average consciousness across mesh",
        "# TYPE aios_mesh_average_consciousness gauge",
        f"aios_mesh_average_consciousness {avg_consciousness:.4f}",
        "",
        "# HELP aios_mesh_coherence Mesh coherence (1=coherent, 0.5=degraded, 0=offline)",
        "# TYPE aios_mesh_coherence gauge",
        f"aios_mesh_coherence {coherence_value:.1f}",
        "",
        f'# HELP aios_mesh_coherence_status Mesh coherence status string',
        f'# TYPE aios_mesh_coherence_status gauge',
        f'aios_mesh_coherence_status{{status="{mesh_coherence}"}} 1',
    ]
    
    # Per-cell status if provided
    if cell_statuses:
        lines.append("")
        lines.append("# HELP aios_mesh_cell_level Individual cell consciousness in mesh")
        lines.append("# TYPE aios_mesh_cell_level gauge")
        for cell_id, info in cell_statuses.items():
            level = info.get("level", 0.0)
            status = info.get("status", "unknown")
            lines.append(
                f'aios_mesh_cell_level{{cell_id="{cell_id}",status="{status}"}} {level:.4f}'
            )
    
    return "\n".join(lines) + "\n"
