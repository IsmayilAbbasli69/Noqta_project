import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value

def cleaned_values(series: pd.Series) -> list[str]:
    values = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    return values.tolist()

def normalize_label(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")

def build_type_lookup(types: list[str]) -> dict[str, str]:
    return {normalize_label(item): item for item in types}

def parse_json_object(text: str) -> dict:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("LLM cavabından JSON parse edilə bilmədi.")

def resolve_type_candidates(requested: list[str], lookup: dict[str, str]) -> tuple[list[str], list[str]]:
    resolved = []
    unresolved = []
    lookup_items = list(lookup.items())
    for raw in requested:
        key = normalize_label(str(raw))
        if not key:
            continue
        if key in lookup:
            resolved.append(lookup[key])
            continue
        matches = [orig for norm, orig in lookup_items if key in norm or norm in key]
        if matches:
            resolved.extend(matches[:3])
        else:
            unresolved.append(str(raw))

    unique = list(dict.fromkeys(resolved))
    return unique, unresolved

def haversine_km(center_lat: float, center_lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    lat1 = np.radians(center_lat)
    lon1 = np.radians(center_lon)
    lat2 = np.radians(lats)
    lon2 = np.radians(lons)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c
