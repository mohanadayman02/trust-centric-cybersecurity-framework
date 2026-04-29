"""Feature-view utilities for NSL-KDD multi-agent experiments."""

from __future__ import annotations

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
