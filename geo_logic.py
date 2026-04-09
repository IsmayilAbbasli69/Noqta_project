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


def _sample_nearest_points(
    center_lat: float,
    center_lon: float,
    points_df: pd.DataFrame,
    radius_km: float,
    sample_size: int = 3,
) -> list[dict]:
    if points_df is None or points_df.empty:
        return []

    points = points_df[["lat", "lon"]].to_numpy(dtype=float)
    distances = haversine_km(center_lat, center_lon, points[:, 0], points[:, 1])
    within_mask = distances <= radius_km
    if not np.any(within_mask):
        return []

    within_df = points_df.loc[within_mask].copy()
    within_df["distance_km"] = distances[within_mask]
    within_df = within_df.sort_values("distance_km").head(sample_size)

    return [
        {
            "name": str(row.get("NAME") or "Bilinməyən"),
            "type": str(row.get("TYPE") or ""),
            "street": str(row.get("STRT_NAME_CLN") or ""),
            "distance_km": round(float(row["distance_km"]), 3),
            "maps_url": f"https://www.google.com/maps?q={float(row['lat']):.6f},{float(row['lon']):.6f}",
        }
        for _, row in within_df.iterrows()
    ]


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


# Additional Baku district statistics (provided by the user)
BAKU_DISTRICT_DENSITY = {
    "nasimi": 21950.0,
    "nizami": 9295.0,
    "khatai": 9233.0,
    "narimanov": 9060.0,
    "yasamal": 9710.0,
    "sabail": 3400.0,
    "binagadi": 2374.0,
    "surakhani": 1741.0,
    "sabunchu": 1380.0,
    "pirallahi": 627.0,
    "khazar": 557.0,
    "garadagh": 109.0,
}

BAKU_DISTRICT_SALARY = {
    "binagadi": 944.0,
    "khatai": 1265.3,
    "khazar": 1603.3,
    "garadagh": 1623.3,
    "narimanov": 1130.6,
    "nasimi": 1462.0,
    "nizami": 1232.5,
    "pirallahi": 2358.2,
    "sabunchu": 999.7,
    "sabail": 2028.8,
    "surakhani": 917.5,
    "yasamal": 1127.7,
}

BAKU_DISTRICT_EMPLOYED = {
    "binagadi": 69184.0,
    "khatai": 92818.0,
    "khazar": 33106.0,
    "garadagh": 24569.0,
    "narimanov": 115737.0,
    "nasimi": 226018.0,
    "nizami": 62996.0,
    "pirallahi": 7408.0,
    "sabunchu": 38473.0,
    "sabail": 117846.0,
    "surakhani": 22110.0,
    "yasamal": 170710.0,
}

DISTRICT_ALIASES = {
    "nəsimi": "nasimi",
    "насиминский": "nasimi",
    "nasimi": "nasimi",
    "nizami": "nizami",
    "низаминский": "nizami",
    "xətai": "khatai",
    "xetai": "khatai",
    "хатаинский": "khatai",
    "khatai": "khatai",
    "nərimanov": "narimanov",
    "nərimanovski": "narimanov",
    "narimanov": "narimanov",
    "наримановский": "narimanov",
    "yasamal": "yasamal",
    "ясамальский": "yasamal",
    "səbail": "sabail",
    "sebail": "sabail",
    "sabail": "sabail",
    "сабаильский": "sabail",
    "binəqədi": "binagadi",
    "bineqedi": "binagadi",
    "binagadi": "binagadi",
    "бинагадинский": "binagadi",
    "suraxanı": "surakhani",
    "surakhani": "surakhani",
    "сураханский": "surakhani",
    "sabunçu": "sabunchu",
    "sabuncu": "sabunchu",
    "sabunchu": "sabunchu",
    "сабунчинский": "sabunchu",
    "pirallahı": "pirallahi",
    "pirallahi": "pirallahi",
    "пираллахинский": "pirallahi",
    "xəzər": "khazar",
    "xezer": "khazar",
    "khazar": "khazar",
    "хазарский": "khazar",
    "qaradağ": "garadagh",
    "qaradag": "garadagh",
    "garadagh": "garadagh",
    "гарадагский": "garadagh",
}


def _canonical_district_name(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    cleaned = (
        raw.replace(" district", "")
        .replace(" rayonu", "")
        .replace(" rayonu", "")
        .replace(" район", "")
        .replace("район", "")
        .strip()
    )
    return DISTRICT_ALIASES.get(cleaned, cleaned)


def _market_snapshot(city: str, state: str, socio_weight: float) -> dict:
    district_key = _canonical_district_name(state)

    # Estimated ranges used for decision-support context.
    rent_azn_m2_ranges = {
        "nasimi": "18-35",
        "sabail": "20-38",
        "yasamal": "16-30",
        "narimanov": "14-28",
        "khatai": "13-26",
        "nizami": "10-20",
        "binagadi": "8-17",
        "sabunchu": "7-15",
        "surakhani": "7-14",
        "khazar": "7-16",
        "pirallahi": "5-11",
        "garadagh": "5-10",
    }

    land_azn_m2_ranges = {
        "nasimi": "1800-4000",
        "sabail": "2200-4800",
        "yasamal": "1500-3200",
        "narimanov": "1300-3000",
        "khatai": "1200-2800",
        "nizami": "900-1900",
        "binagadi": "750-1700",
        "sabunchu": "650-1400",
        "surakhani": "600-1300",
        "khazar": "700-1600",
        "pirallahi": "450-950",
        "garadagh": "420-900",
    }

    avg_salary = BAKU_DISTRICT_SALARY.get(district_key, 1374.9)
    avg_density = BAKU_DISTRICT_DENSITY.get(district_key, None)

    if socio_weight >= 1.65:
        purchasing_power = "yuksek"
    elif socio_weight >= 1.35:
        purchasing_power = "orta-yuksek"
    elif socio_weight >= 1.15:
        purchasing_power = "orta"
    else:
        purchasing_power = "asagi-orta"

    return {
        "district_key": district_key,
        "estimated_land_price_azn_m2": land_azn_m2_ranges.get(district_key, "namelum"),
        "estimated_rent_azn_m2_month": rent_azn_m2_ranges.get(district_key, "namelum"),
        "avg_salary_azn": round(float(avg_salary), 1),
        "population_density_km2": int(avg_density) if avg_density is not None else None,
        "purchasing_power_level": purchasing_power,
        "note": "teqribi intervaldir, son qerar ucun lokal bazar yoxlamasi vacibdir",
        "city": str(city),
        "state": str(state),
    }


def _get_socio_economic_weight(city: str, state: str) -> float:
    """
    Regional weight multiplier based on density/income indicators.
    It is primarily calibrated for Baku and nearby settlements (including Absheron).
    """
    c_norm = str(city).lower().strip()
    s_norm = str(state).lower().strip()
    district_key = _canonical_district_name(s_norm)
    
    baku_aliases = {"baku", "baki", "bakı"}
    high_income_baku = {"nasimi", "sabail", "yasamal", "narimanov", "khatai"}
    mid_income_baku = {"binagadi", "nizami"}
    
    # Baku-adjacent and Absheron settlements provided in project scope
    baku_absheron_settlements = {
        "hokmali", "jeyranbatan", "khirdalan", "mahammadi", 
        "masazir", "mehdiabad", "novkhani", "novkhani (villas)", 
        "pirekeshkul shehercik", "saray", "absheron", "abşeron"
    }
    
    weight = 1.0
    
    if district_key in BAKU_DISTRICT_DENSITY:
        # Dynamic weight based on density + salary + employment data
        density = BAKU_DISTRICT_DENSITY[district_key]
        salary = BAKU_DISTRICT_SALARY.get(district_key, 1374.9)
        employed = BAKU_DISTRICT_EMPLOYED.get(district_key, 980975.0 / 12.0)

        density_norm = math.log1p(density) / math.log1p(max(BAKU_DISTRICT_DENSITY.values()))
        salary_norm = salary / 1374.9  # Baku city average monthly salary
        employed_norm = math.log1p(employed) / math.log1p(max(BAKU_DISTRICT_EMPLOYED.values()))

        # Clamp salary effect to avoid overweighting this signal
        salary_norm = min(max(salary_norm, 0.7), 1.6)

        composite = (0.45 * density_norm) + (0.35 * salary_norm) + (0.20 * employed_norm)
        return 0.85 + (0.95 * composite)

    if c_norm in baku_aliases or s_norm in baku_aliases:
        weight = 1.25 # Base uplift for Baku
        
        # Central Baku (high density, stronger purchasing power)
        if district_key in high_income_baku or s_norm in high_income_baku:
            weight += 0.25 
        # Mid-tier Baku districts
        elif district_key in mid_income_baku or s_norm in mid_income_baku:
            weight += 0.10
        # Outer urban areas
        else:
            weight -= 0.05
            
    elif c_norm in baku_absheron_settlements or s_norm in baku_absheron_settlements:
        weight = 1.20 # Weight for Khirdalan, Masazir, and nearby settlements
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
    
    # Clean street names and normalize blanks
    df["STRT_NAME_CLN"] = df["STRT_NAME"].fillna("").astype(str).str.strip()
    df.loc[df["STRT_NAME_CLN"] == "", "STRT_NAME_CLN"] = "Bilinməyən Küçə"
    
    # Round coordinates to 3 decimals (~110m grid cell)
    # This prevents overly noisy or duplicate-like location candidates.
    df["lat_grid"] = df["lat"].round(3)
    df["lon_grid"] = df["lon"].round(3)

    # Region-level density index by city/state group
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

    # Evaluate each coordinate grid as a candidate location.
    # A grid does not need to coincide with a single exact object point.
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
                competitor_samples = []
            else:
                target_coords = target_bbox[["lat", "lon"]].to_numpy(dtype=float)
                comp_distances = haversine_km(center_lat, center_lon, target_coords[:, 0], target_coords[:, 1])
                competitors = int((comp_distances <= radius_km).sum())
                competitor_samples = _sample_nearest_points(
                    center_lat,
                    center_lon,
                    target_bbox,
                    radius_km,
                    sample_size=3,
                )
        else:
            competitors = 0
            competitor_samples = []

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

        # Global score = local support score * regional density weight * socio-economic weight
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
                "nearby_competitors": competitor_samples,
                "market_snapshot": _market_snapshot(city, state, socio_w),
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
