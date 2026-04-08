# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from core_utils import load_env_file, cleaned_values, build_type_lookup, resolve_type_candidates
from llm_agent import get_groq_client, call_groq_json, call_groq_text
from geo_logic import run_logic_module

st.set_page_config(page_title="Azerbaijan Objects Map", page_icon="map", layout="wide")

DATA_PATH = Path("dataset/azrbaycanda_obyektlr_v_mkanlar.geojson")
load_env_file()

@st.cache_data(show_spinner=True)
def load_geojson(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    rows = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates", [None, None])

        if not isinstance(coords, list) or len(coords) < 2:
            continue
        try:
            lon_f = float(coords[0])
            lat_f = float(coords[1])
        except (TypeError, ValueError, IndexError):
            continue

        if not (np.isfinite(lon_f) and np.isfinite(lat_f)):
            continue

        rows.append({
            "OBJECTID": props.get("OBJECTID"),
            "CITY": props.get("CITY"),
            "STATE": props.get("STATE"),
            "STRT_NAME": props.get("STRT_NAME"),
            "STRT_ADDR": props.get("STRT_ADDR"),
            "NAME": props.get("NAME"),
            "TYPE": props.get("TYPE"),
            "lon": lon_f,
            "lat": lat_f,
        })
    return pd.DataFrame(rows)


def filtered_frame(df, cities, states, types):
    out = df
    if cities: out = out[out["CITY"].isin(cities)]
    if states: out = out[out["STATE"].isin(states)]
    if types: out = out[out["TYPE"].isin(types)]
    return out


st.title("Azərbaycan: Obyektər bazası & AI Biznes Strategiya")
if not DATA_PATH.exists():
    st.error(f"Dataset tapılmadı: {DATA_PATH}")
    st.stop()

with st.spinner("Dataset yüklənir..."):
    df = load_geojson(DATA_PATH)

if df.empty:
    st.warning("Dataset boşdur.")
    st.stop()

cities = cleaned_values(df["CITY"])
obj_types = cleaned_values(df["TYPE"])

st.sidebar.header("Filtrlər")
selected_cities = st.sidebar.multiselect("Şəhər", cities)
state_source = df[df["CITY"].isin(selected_cities)] if selected_cities else df
states = cleaned_values(state_source["STATE"])
selected_states = st.sidebar.multiselect("Rayon", states)
selected_types = st.sidebar.multiselect("Tip", obj_types)

result = filtered_frame(df, selected_cities, selected_states, selected_types)
map_df = result[np.isfinite(result["lat"]) & np.isfinite(result["lon"])].copy()

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Ümumi sətir (Dataset)", f"{len(df):,}")
metric2.metric("Obyekt sayı (Filtrli)", f"{len(map_df):,}")
metric3.metric("Obyekt növləri", f"{len(obj_types):,}")

if not map_df.empty:
    sample_df = map_df.sample(n=min(len(map_df), 10000))
    view_state = pdk.ViewState(latitude=float(sample_df["lat"].mean()), longitude=float(sample_df["lon"].mean()), zoom=8)
    layer = pdk.Layer("ScatterplotLayer", data=sample_df, get_position="[lon, lat]", get_fill_color="[0, 130, 255, 140]", get_radius=40, pickable=True)
    st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v11", initial_view_state=view_state, layers=[layer]))

# AI BÖLMƏSİ
st.divider()
st.header("AI Strategiya Modulu")
st.markdown("Biznes ideyanızı yazın, LLM məntiqi hesablamalarla **dəqiq koordinatlar və səbəblər** verəcək.")

api_key = os.getenv("API_KEY", "").strip()
business_request = st.text_area(
    "Biznes ideyası:",
    placeholder="Mən ucuz restoran açmaq istəyirəm, tələbələrə yaxın olsun, dayanacaqlar da olsun.",
    height=100
)

col1, col2 = st.columns(2)
with col1:
    radius_km = st.slider("Axtarış radiusu (km)", 0.5, 5.0, 1.5, 0.5)
with col2:
    top_n = st.slider("Top təklif sayı", 3, 10, 3, 1)

if st.button("Biznes Analizini Başlat", type="primary"):
    if not business_request.strip():
        st.error("Zəhmət olmasa ideyanı yazın.")
        st.stop()
    if not api_key:
        st.error("`.env` içində API_KEY (Groq) daxil edilməyib.")
        st.stop()

    client = get_groq_client(api_key=api_key)
    type_lookup = build_type_lookup(obj_types)

        # LLM 1 - PARSER: Məntiqi və Açar sözləri çıxarır
    parser_system_prompt = f"""Sən yüksək səviyyəli data və biznes analitikisən.
İstifadəçinin biznes ideyasını analiz et və onun üçün MƏNTİQİ PLAN çıxar.
Hədəf biznes növünü və ona müştəri cəlb edəcək DƏSTƏK (keys) obyekt növlərini tap.
Xüsusilə nəzərə al ki, istifadəçi MÖVCUD obyekt axtarmır! YENİ OBYEKT AÇMAQ istəyir.
Buna görə də "keys" olaraq rəqiblərin obyekt növlərini YOX, müştəri gətirəcək növləri seç.
Məsələn: Ucuz restoran üçün dəstək obyektlər məktəb, dayanacaq, universitet və parklardır. (RESTAURANT yazma!). Aptek üçün xəstəxana və klinikalar yaz (PHARMACY yazma!).

QAYDALAR:
- Mümkünsə 2 target qaytar: əsas biznes növü və onun çox yaxın alternativi və ya eyni bazarın ikinci versiyası.
- Hədəf listi boş olmasın; ən azı 1, ideal halda 2 obyekt növü ver.
- "keys" mütləq ən azı 2 dəstək növü olsun.
- Koordinat yazma. Koordinatı yalnız logic module hesablayacaq.

Yalnız və yalnız aşağıdakı formatda JSON qaytar:
{{
  "reasoning": "Müştəri gətirəcək hədəflərin izahı...",
    "object": "ƏSAS_HƏDƏF_NÖV",
    "targets": ["ƏSAS_HƏDƏF_NÖV", "MÜMKÜN_2-Cİ_HƏDƏF_NÖV"],
    "keys": ["DƏSTƏK_NÖV1", "DƏSTƏK_NÖV2"]
}}

İcazə verilən Dataset TYPE siyahısı: {json.dumps(obj_types, ensure_ascii=False)}
    """
    
    with st.spinner("AI ideyanızı analiz edir (Mərhələ 1: Kateqoriyalar tapılır)..."):
        try:
            intent_json = call_groq_json(client, parser_system_prompt, business_request)
        except Exception as e:
            st.error(f"API Xətası (Mərhələ 1): {e}")
            st.stop()

    req_obj = intent_json.get("object", "")
    req_targets = intent_json.get("targets", [])
    req_keys = intent_json.get("keys", [])
    
    if not isinstance(req_targets, list): req_targets = [str(req_targets)]
    if not isinstance(req_keys, list): req_keys = [str(req_keys)]
    resolved_obj, _ = resolve_type_candidates([str(req_obj)] + [str(x) for x in req_targets], type_lookup)
    resolved_keys, _ = resolve_type_candidates([str(x) for x in req_keys], type_lookup)
    target_types = resolved_obj[:2]

    if not target_types and not resolved_keys:
        st.warning("LLM dataya uyğun düzgün obyekt tipi tapa bilmədi. Sözləri dəyişib yoxlayın.")
        st.stop()

    # LOGIC LAYER - Məsafə və sıxlıq analizi
    with st.spinner("Ərazi sıxlığı və rəqabət hesablanır... (Bu proses məlumatın həcmindən asılı olaraq bir neçə saniyə çəkə bilər)"):
        # LLM-in işini asanlaşdırmaq üçün target listi yalnız rəqib/target sıxlığını saymaq üçün istifadə olunur.
        # Amma biz İDEAL BOŞLUQ axtarırıqsa, target-ə çox yaxın yerlər MINUS balla çıxmalıdır.
        base_df = df # Tam verilənlər bazası üzərindən axtarış etsin ki, bütün məktəbləri bilsin.
        if selected_cities:
            base_df = base_df[base_df["CITY"].isin(selected_cities)]
        if selected_states:
            base_df = base_df[base_df["STATE"].isin(selected_states)]
            
        logic_payload = run_logic_module(base_df, target_types, resolved_keys, radius_km, top_n)
        logic_payload["ai_reasoning"] = intent_json.get("reasoning", "")

    if "error" in logic_payload or not logic_payload.get("top_locations"):
        st.warning("Göstərilən şərtlərə uyğun ideal lokasiya (grid) tapılmadı.")
        st.stop()

    # LLM 2 - INTERPRETER: Riyazi nəticəni insan dilinə və praktik addımlara çevirir
    strategist_prompt = """Sən biznes məsləhətçisisən. İstifadəçinin YENİ biznes yeri üçün aparılan axtarışlarının riyazi nәticələri sənə JSON formasında veriləcək.

QAYDALAR:
- İstifadəçiyə HEÇ VAXT "burada rəqib azdır, ona görə restoran aç" demə, əgər ətrafda ona DƏSTƏK olan (müşdərini cəlb edəcək, məs. məktəb, ofis, xəstəxana və s.) obyekt yoxdursa! Yalnız rəqibin olmaması ora yaxşı yer demək deyil.
- Təqdim olunan Top lokasiyaların NİYƏ LİSTƏ SALINDIĞINI əsaslandır (əsasən orada yerləşən DƏSTƏK obyektlərin sayına və çoxluğuna/növünə istinad edərək).
- "nearby_examples" içindəki data yalnız hədəf üçün tapılmış real DƏSTƏK obyektləridir. Mütləq adlarını, küçələrini və məsafəsini qeyd edərək istifadəçiyə sübut kimi göstər (məs: 'Oradakı 120 saylı məktəbdən 200m aralıdasız, bu çox böyük üstünlükdür').
- Əgər bir lokalda DƏSTƏK obyekt ümumiyyətlə yoxdursa, bunu "pis qeyd/risk" olaraq vurğula.
- Logic payload-da `target_types` listi ola bilər; bunu nəzərə al və çıxışda yalnız verilən koordinat/linklərdən istifadə et.
- Koordinat uydurma. Özün yeni koordinat hesablamırsan, yalnız `maps_url` və `center_lat/center_lon` əsasında danış.

STRUKTUR:
1. Analitik Baxış: (Ümumi tapıntılar və verilmiş "ai_reasoning" fikrini daha mükəmməl yaz). Seçilmiş dəstək sahələr niyə məhz o biznesə xeyir edəcək?
2. YENİ BİZNES ÜÇÜN İDEAL LOKASİYALAR (Top 1, Top 2, Top 3 şəklində reytinqə uyğun sırala):
   Hər biri barədə:
   - Xal (Score): Bu məkan nə qədər güclüdür? (JSON-dakı `score` əsasında).
   - Yer: Rayon, Şəhər, Küçə ("STRT_NAME").
   - Həqiqi "Google Maps" dəqiq koordinat linki mütləq yapışdırılsın.
   - Analiz: Hər məkan təkrarlanmamış formada unikal səbəblərlə izah edilsin. Əgər Top 1 və Top 2 yaxındırsa, fərqini izah et (məsələn: "Top 2 ünvanı bir az aralıdadır və Məktəbə daha yaxındır"). Mütləq "nearby_examples" içindəki unikal adlara istinad et. Hər 3 məkanı eyni sözlərlə qətiyyən təkrar etmə!
3. Biznesi qurmağa başlamaq üçün addımlar və tövsiyə...
"""
    
    with st.spinner("Strateji hesabat hazırlanır (Mərhələ 3: Yekun rəy)..."):
        try:
            final_strategy = call_groq_text(client, strategist_prompt, json.dumps(logic_payload, ensure_ascii=False))
        except Exception as e:
            st.error(f"API Xətası (Mərhələ 3): {e}")
            st.stop()

    st.success("Təhlil tamamlandı!")
    st.markdown(final_strategy)
    
    with st.expander("Mühəndis: Detallı JSON Görüntüsü (Debug)"):
        st.json(logic_payload)
