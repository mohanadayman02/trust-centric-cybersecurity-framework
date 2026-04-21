"""Model package for IDS agents."""

from .detection_agents import (
    BaseDetectionAgent,
    BehavioralAnalysisAgent,
    RandomForestAgent,
    SVMAgent,
    TrafficAnalysisAgent,
)

__all__ = [
    "BaseDetectionAgent",
    "BehavioralAnalysisAgent",
    "TrafficAnalysisAgent",
    "RandomForestAgent",
    "SVMAgent",
]
