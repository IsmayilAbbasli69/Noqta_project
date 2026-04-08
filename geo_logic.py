import math
import pandas as pd
import numpy as np
from core_utils import haversine_km

def _radius_bbox(center_lat: float, center_lon: float, radius_km: float) -> tuple[float, float, float, float]:
    lat_delta = radius_km / 110.574
    lon_scale = max(math.cos(math.radians(center_lat)), 0.01)
    lon_delta = radius_km / (111.320 * lon_scale)
    return center_lat - lat_delta, center_lat + lat_delta, center_lon - lon_delta, center_lon + lon_delta


def _count_and_sample_within_radius(
    center_lat: float,
    center_lon: float,
    points_df: pd.DataFrame,
    radius_km: float,
    sample_size: int = 3,
) -> tuple[int, list[dict]]:
    if points_df is None or points_df.empty:
        return 0, []

    min_lat, max_lat, min_lon, max_lon = _radius_bbox(center_lat, center_lon, radius_km)
    bbox_df = points_df[
        (points_df["lat"] >= min_lat)
        & (points_df["lat"] <= max_lat)
        & (points_df["lon"] >= min_lon)
        & (points_df["lon"] <= max_lon)
    ]
    if bbox_df.empty:
        return 0, []

    points = bbox_df[["lat", "lon"]].to_numpy(dtype=float)
    distances = haversine_km(center_lat, center_lon, points[:, 0], points[:, 1])
    within_mask = distances <= radius_km
    if not np.any(within_mask):
        return 0, []

    within_df = bbox_df.loc[within_mask].copy()
    within_df["distance_km"] = distances[within_mask]
    within_df = within_df.sort_values("distance_km").head(sample_size)

    samples = [
        {
            "name": str(row.get("NAME") or "Bilinməyən"),
            "address": str(row.get("STRT_ADDR") or ""),
            "street": str(row.get("STRT_NAME_CLN") or ""),
            "distance_km": round(float(row["distance_km"]), 3),
            "maps_url": f"https://www.google.com/maps?q={float(row['lat']):.6f},{float(row['lon']):.6f}",
        }
        for _, row in within_df.iterrows()
    ]
    return int(len(within_df)), samples


def _display_street_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "Bilinməyən Küçə"

    lowered = text.lower()
    if lowered in {"hwy", "highway", "road", "rd", "mwy"}:
        return "Şosse / Magistral"

    cleaned = text.replace("hwy", "highway").replace("Hwy", "Highway")
    return cleaned


def _pick_diverse_locations(rows: list[dict], top_n: int) -> list[dict]:
    picked = []
    used_streets: set[str] = set()

    for row in rows:
        street_key = str(row.get("STRT_NAME") or "").strip().lower()
        if street_key in used_streets:
            continue

        too_close = False
        for chosen in picked:
            if haversine_km(
                float(row["center_lat"]),
                float(row["center_lon"]),
                float(chosen["center_lat"]),
                float(chosen["center_lon"]),
            ) < 0.75:
                too_close = True
                break

        if too_close:
            continue

        picked.append(row)
        if street_key:
            used_streets.add(street_key)
        if len(picked) >= top_n:
            break

    if len(picked) < top_n:
        for row in rows:
            if row in picked:
                continue
            picked.append(row)
            if len(picked) >= top_n:
                break

    return picked[:top_n]


def _get_socio_economic_weight(city: str, state: str) -> float:
    """
    Yalnız Bakı və ətraf qəsəbələr (Abşeron daxil) üçün sıxlıq/gəlir indekslərinə 
    əsasən hesablanmış regional weight multiplier.
    """
    c_norm = str(city).lower().strip()
    s_norm = str(state).lower().strip()
    
    baku_aliases = {"baku", "baki", "bakı"}
    high_income_baku = {"nasimi", "nəsimi", "sabail", "səbail", "sabayil", "yasamal", "narimanov", "nərimanov", "khatai", "xətai"}
    mid_income_baku = {"binagadi", "binəqədi", "nizami"}
    
    # İstifadəçinin qeyd etdiyi Bakı ətrafı və Abşeron qəsəbələri
    baku_absheron_settlements = {
        "hokmali", "jeyranbatan", "khirdalan", "mahammadi", 
        "masazir", "mehdiabad", "novkhani", "novkhani (villas)", 
        "pirekeshkul shehercik", "saray", "absheron", "abşeron"
    }
    
    weight = 1.0
    
    if c_norm in baku_aliases or s_norm in baku_aliases:
        weight = 1.25 # Bakının ümumi baza çəkisi
        
        # Bakı mərkəz (Sıxlıq yüksək, Alıcılıq yüksək)
        if s_norm in high_income_baku:
            weight += 0.25 
        # Bakı orta mərkəz
        elif s_norm in mid_income_baku:
            weight += 0.10
        # Şəhərkənarı bölgələr
        else:
            weight -= 0.05
            
    elif c_norm in baku_absheron_settlements or s_norm in baku_absheron_settlements:
        weight = 1.20 # Xırdalan, Masazır və digər qeyd edilən qəsəbələrin çəkisi
    elif c_norm in {"sumgayit", "sumqayit", "sumgait", "sumqayıt"}:
        weight = 1.15
        
    return weight


def run_logic_module(
    scoped_df: pd.DataFrame,
    target_types: list[str],
    key_types: list[str],
    radius_km: float,
    top_n: int,
) -> dict:
    if scoped_df.empty:
        return {"error": "Filtrdən sonra analiz üçün data qalmadı."}

    df = scoped_df.copy()
    
    # Küçə adlarını təmizlə və boşluqları müəyyənləşdir
    df["STRT_NAME_CLN"] = df["STRT_NAME"].fillna("").astype(str).str.strip()
    df.loc[df["STRT_NAME_CLN"] == "", "STRT_NAME_CLN"] = "Bilinməyən Küçə"
    
    # Koordinatları 3 rəqəmə qədər yuvarlaqlaşdırırıq (~110 metr kvadrat / grid)
    # Bu bizə reallıqdan uzaq və ya sırf küçə adı eynidir deyə səhv hesablamaların qarşısını alır
    df["lat_grid"] = df["lat"].round(3)
    df["lon_grid"] = df["lon"].round(3)

    # Rayon və ya şəhər üçün ümumi sıxlıq indeksi
    region_counts = df.groupby(["CITY", "STATE"], dropna=False).size()
    max_region_log = math.log1p(region_counts.max()) if not region_counts.empty else 1.0
    region_weights = {
        k: 0.8 + 0.7 * (math.log1p(v) / max_region_log)
        for k, v in region_counts.items()
    }

    target_types = [str(item) for item in target_types if str(item).strip()]
    target_data = df[df["TYPE"].isin(target_types)].copy() if target_types else df.iloc[0:0].copy()
    target_count_total = int(len(target_data))
    key_data = {k: df[df["TYPE"] == k].copy() for k in key_types}

    grouped = df.groupby(["CITY", "STATE", "lat_grid", "lon_grid"], dropna=False)
    
    rows = []
    target_type_label = ", ".join(target_types) if target_types else "UNKNOWN"

    # Hər koordinat gridini sadəcə qruplaşdırıb hesablayırıq. 
    # Bir gridin mütləq bir obyektdə dayanması şərt deyil.
    for (city, state, lat_g, lon_g), block in grouped:
        center_lat = lat_g
        center_lon = lon_g
        
        street_counts = block["STRT_NAME_CLN"].value_counts()
        best_street = street_counts.index[0] if not street_counts.empty else "Bilinməyən Küçə"
        best_street = _display_street_name(best_street)

        if not target_data.empty:
            min_lat, max_lat, min_lon, max_lon = _radius_bbox(center_lat, center_lon, radius_km)
            target_bbox = target_data[
                (target_data["lat"] >= min_lat)
                & (target_data["lat"] <= max_lat)
                & (target_data["lon"] >= min_lon)
                & (target_data["lon"] <= max_lon)
            ]
            if target_bbox.empty:
                competitors = 0
            else:
                target_coords = target_bbox[["lat", "lon"]].to_numpy(dtype=float)
                comp_distances = haversine_km(center_lat, center_lon, target_coords[:, 0], target_coords[:, 1])
                competitors = int((comp_distances <= radius_km).sum())
        else:
            competitors = 0

        key_counts = {}
        key_samples = {}
        key_total = 0
        
        for k in key_types:
            points_df = key_data.get(k)
            if points_df is None or points_df.empty:
                key_counts[k] = 0
                continue

            near_count, samples = _count_and_sample_within_radius(center_lat, center_lon, points_df, radius_km)
            key_counts[k] = near_count
            key_total += near_count
            if samples:
                key_samples[k] = samples

        demand_proxy = math.log1p(len(block))
        key_type_coverage = sum(1 for v in key_counts.values() if v > 0)
        
        support_density = key_total / max(len(block), 1)
        region_w = region_weights.get((city, state), 1.0)
        socio_w = _get_socio_economic_weight(city, state)

        # Qlobal xal = (Yerli dəstək xalı) * (Riyazi regional sıxlıq çəkisi (obyekt sayına görə)) * (Sosiokulturoloji və gəlir çəkisi)
        base_score = (4.0 * support_density) + (2.5 * key_type_coverage) + (0.5 * demand_proxy) - (1.75 * competitors)
        score = base_score * region_w * socio_w

        gmaps_url = f"https://www.google.com/maps?q={center_lat:.6f},{center_lon:.6f}"

        rows.append(
            {
                "CITY": city,
                "STATE": state,
                "STRT_NAME": best_street,
                "center_lat": round(center_lat, 6),
                "center_lon": round(center_lon, 6),
                "competitors": competitors,
                "supporting_total": key_total,
                "support_density": round(support_density, 4),
                "supporting_type_coverage": key_type_coverage,
                "demand_proxy": round(demand_proxy, 3),
                "score": round(score, 3),
                "key_breakdown": key_counts,
                "nearby_examples": key_samples,
                "maps_url": gmaps_url,
            }
        )

    ranked = sorted(rows, key=lambda x: x["score"], reverse=True)
    ranked = _pick_diverse_locations(ranked, top_n)
    return {
        "target_type": target_type_label,
        "target_types": target_types,
        "target_count_in_scope": target_count_total,
        "radius_km": radius_km,
        "candidates_evaluated": len(rows),
        "top_locations": ranked,
    }
