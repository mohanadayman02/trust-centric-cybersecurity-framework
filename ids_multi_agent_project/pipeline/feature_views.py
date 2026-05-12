"""Feature-view utilities for NSL-KDD experiments using multiple decision sources.

Terminology updated: 'agents' -> 'models' or 'decision sources' in docstrings and reports.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List


def get_nsl_kdd_feature_views(feature_names: List[str]) -> Dict[str, List[str]]:
    """Return NSL-KDD feature groups for four agents.

    Index mapping (1-based in paper, converted to 0-based python slices):
    - basic features: 1-9   -> [0:9]
    - content features: 10-22 -> [9:22]
    - time traffic: 23-31   -> [22:31]
    - host traffic: 32-41   -> [31:41]
    """
    if len(feature_names) < 41:
        raise ValueError(
            "NSL-KDD feature grouping requires at least 41 non-label features. "
            f"Got {len(feature_names)} feature(s)."
        )

    return {
        "basic_agent": list(feature_names[0:9]),
        "content_agent": list(feature_names[9:22]),
        "time_traffic_agent": list(feature_names[22:31]),
        "host_traffic_agent": list(feature_names[31:41]),
    }


def map_processed_feature_views(
    processed_feature_names: List[str],
    raw_feature_views: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Map raw NSL-KDD views to post-preprocessing feature names.

    Supports transformed names such as:
    - num__duration
    - cat__protocol_type_tcp
    """
    mapped_views: Dict[str, List[str]] = {}

    for view_name, raw_columns in raw_feature_views.items():
        mapped_columns: List[str] = []

        for raw_col in raw_columns:
            candidates = [
                name
                for name in processed_feature_names
                if name == f"num__{raw_col}"
                or name == f"cat__{raw_col}"
                or name.startswith(f"cat__{raw_col}_")
                or name == raw_col
                or name.startswith(f"{raw_col}_")
            ]
            mapped_columns.extend(candidates)

        # Deduplicate while preserving order.
        seen = set()
        deduped = []
        for col in mapped_columns:
            if col not in seen:
                seen.add(col)
                deduped.append(col)

        if not deduped:
            raise ValueError(
                "Feature-view mapping failed after preprocessing for view "
                f"'{view_name}'. Could not map any transformed columns from raw view "
                f"columns: {raw_columns}."
            )

        mapped_views[view_name] = deduped

    return mapped_views


def get_generic_feature_views(
    feature_names: List[str],
    traffic_features: List[str] | None = None,
    connection_features: List[str] | None = None,
    content_features: List[str] | None = None,
) -> Dict[str, List[str]]:
    """Generate generic four-agent feature views for datasets without fixed structure.
    
    This function creates feature subsets for four agents by organizing features
    into semantic groups: traffic patterns, connection behavior, and content/derived features.
    
    Args:
        feature_names: All available feature names after preprocessing
        traffic_features: Known traffic/rate features (e.g., ['rate', 'sload', 'dload'] for UNSW)
        connection_features: Known connection features (e.g., ['state', 'tcprtt'] for UNSW)
        content_features: Known content/derived features (e.g., ['ct_srv_src', 'ct_state_ttl'] for UNSW)
    
    Returns:
        Dict with keys: basic_agent, content_agent, time_traffic_agent, host_traffic_agent
    """
    all_features = list(feature_names)
    
    # Default semantic grouping for UNSW-like datasets
    if traffic_features is None:
        traffic_features = [f for f in all_features if any(x in f.lower() for x in ['rate', 'load', 'pkt', 'byte', 'jit', 'loss'])]
    
    if connection_features is None:
        connection_features = [f for f in all_features if any(x in f.lower() for x in ['state', 'rtt', 'ttl', 'tcp', 'win', 'ack'])]
    
    if content_features is None:
        content_features = [f for f in all_features if any(x in f.lower() for x in ['ct_', 'response', 'http', 'ftp'])]
    
    # Remaining features for general use
    used = set(traffic_features) | set(connection_features) | set(content_features)
    remaining = [f for f in all_features if f not in used]
    
    # Assign to agents
    basic_cols = remaining[:max(1, len(remaining) // 3)]
    content_cols = content_features + remaining[max(1, len(remaining) // 3):]
    time_traffic_cols = traffic_features + connection_features[:max(1, len(connection_features) // 2)]
    host_traffic_cols = connection_features[max(1, len(connection_features) // 2):] + remaining[max(1, len(remaining) // 3):2 * max(1, len(remaining) // 3)]
    
    # Ensure each view has at least one feature and remove duplicates while preserving all
    views = {
        "basic_agent": basic_cols if basic_cols else all_features[:max(1, len(all_features) // 4)],
        "content_agent": list(dict.fromkeys(content_cols)) if content_cols else all_features[max(1, len(all_features) // 4):max(2, len(all_features) // 2)],
        "time_traffic_agent": list(dict.fromkeys(time_traffic_cols)) if time_traffic_cols else all_features[max(2, len(all_features) // 2):],
        "host_traffic_agent": list(dict.fromkeys(host_traffic_cols)) if host_traffic_cols else all_features[:len(all_features)],
    }
    
    # Validate
    for view_name, cols in views.items():
        if not cols:
            raise ValueError(f"Feature view '{view_name}' is empty. Cannot create agent views with {len(all_features)} features.")
    
    return views


def get_unsw_feature_views(feature_names: List[str]) -> Dict[str, List[str]]:
    """Get UNSW-NB15 specific feature views for four agents.
    
    UNSW-NB15 feature semantic grouping:
    - Traffic rate features: rate, sload, dload, sinpkt, dinpkt, sjit, djit, sloss, dloss
    - Connection state features: state, tcprtt, synack, ackdat, sttl, dttl, swin, dwin, stcpb, dtcpb
    - Content/derived features: ct_srv_src, ct_state_ttl, ct_dst_ltm, etc., response_body_len
    """
    traffic_features = [f for f in feature_names if any(x in f.lower() for x in ['rate', 'load', 'pkt', 'jit', 'loss'])]
    connection_features = [f for f in feature_names if any(x in f.lower() for x in ['state', 'rtt', 'ttl', 'tcp', 'win', 'ack'])]
    content_features = [f for f in feature_names if any(x in f.lower() for x in ['ct_', 'response', 'http', 'ftp', 'sm_ips'])]
    
    return get_generic_feature_views(
        feature_names,
        traffic_features=traffic_features,
        connection_features=connection_features,
        content_features=content_features,
    )
