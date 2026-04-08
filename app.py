# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.express as px

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


st.title("Azərbaycan: Obyektlər bazası & AI Biznes Strategiya")
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

st.sidebar.header("Məlumat Filtrləri")
selected_cities = st.sidebar.multiselect("Şəhər", cities, placeholder="Şəhər seçin")
state_source = df[df["CITY"].isin(selected_cities)] if selected_cities else df
states = cleaned_values(state_source["STATE"])
selected_states = st.sidebar.multiselect("Rayon", states, placeholder="Rayon seçin")
selected_types = st.sidebar.multiselect("Obyekt Tipi", obj_types, placeholder="Tip seçin")

result = filtered_frame(df, selected_cities, selected_states, selected_types)
map_df = result[np.isfinite(result["lat"]) & np.isfinite(result["lon"])].copy()

# Əsas UI Tabları
tab1, tab2, tab3 = st.tabs(["İnteraktiv Xəritə", "AI Biznes Analitikası", "Məlumat Vizuallaşdırması"])

with tab1:
    st.subheader("Bölgələr üzrə Obyekt Xəritəsi (3D və İnteraktiv)")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Ümumi sətir (Dataset)", f"{len(df):,}")
    metric2.metric("Obyekt sayı (Filtrli)", f"{len(map_df):,}")
    metric3.metric("Obyekt növləri", f"{len(obj_types):,}")

    if not map_df.empty:
        # Vizuallaşdırma üçün sürətli və 3D dəstəkli PyDeck istifadə edirik
        sample_df = map_df.sample(n=min(len(map_df), 15000))
        mean_lat = float(sample_df["lat"].mean())
        mean_lon = float(sample_df["lon"].mean())
        
        # Hexagon qatı - ərazi sıxlığını 3D qüllələr kimi göstərmək üçündür
        layer_hex = pdk.Layer(
            "HexagonLayer",
            data=sample_df,
            get_position="[lon, lat]",
            radius=150,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=False, # Hexagonları hover etməmək (sıxlıq üçün qalsın)
            extruded=True,
            color_range=[
                [165, 214, 167],
                [129, 199, 132],
                [76, 175, 80],   
                [255, 183, 77],
                [255, 152, 0],   
                [230, 81, 0]
            ]
        )
        
        # Animasiyalı tək obyekt nöqtələri
        layer_scatter = pdk.Layer(
            "ScatterplotLayer",
            data=sample_df,
            get_position="[lon, lat]",
            get_fill_color="[255, 152, 0, 200]", # Nöqtələr üçün parametr dəyişildi
            get_radius=50,
            pickable=True, # Hover edilə bilən yalnız tək nöqtələr olsun
            auto_highlight=True,
            highlight_color=[255, 255, 255, 255]
        )

        view_state = pdk.ViewState(
            latitude=mean_lat,
            longitude=mean_lon,
            zoom=8,
            pitch=45,
            bearing=15
        )

        # Xəritəni qəşəng və sürətli 3D effekti ilə render edirik
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v11",
                initial_view_state=view_state,
                layers=[layer_hex, layer_scatter],
                tooltip={
                    "html": "<b>Ad:</b> {NAME} <br/> <b>Növ:</b> {TYPE} <br/> <b>Şəhər:</b> {CITY}, {STATE}",
                    "style": {
                        "backgroundColor": "steelblue",
                        "color": "white"
                    }
                }
            ),
            use_container_width=True
        )
    else:
        st.info("Seçilmiş filtrlərə uyğun məlumat tapılmadı.")

with tab3:
    st.subheader("Əsas Statistika və Qrafiklər")
    if not result.empty:
        c1, c2 = st.columns(2)
        with c1:
            # Şəhərlərə görə obyekt sayı
            top_cities = result['CITY'].value_counts().head(10).reset_index()
            top_cities.columns = ['Şəhər', 'Obyekt Sayı']
            fig_cities = px.bar(
                top_cities, x='Şəhər', y='Obyekt Sayı', 
                title='Top 10 Şəhər (Obyekt sayına görə)', 
                color_discrete_sequence=['#4CAF50'] # Yaşıl rəng
            )
            st.plotly_chart(fig_cities, use_container_width=True)

        with c2:
            # Tiplərə görə obyekt sayı
            top_types = result['TYPE'].value_counts().head(10).reset_index()
            top_types.columns = ['Obyekt Tipi', 'Say']
            fig_types = px.pie(
                top_types, values='Say', names='Obyekt Tipi', 
                title='Ən çox yayılmış Top 10 Obyekt Tipi', 
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Greens_r # Yaşıl-narıncı gradient
            )
            fig_types.update_traces(marker=dict(colors=['#4CAF50', '#66BB6A', '#81C784', '#A5D6A7', '#C8E6C9', '#FF9800', '#FFB74D', '#FFCC80', '#FFE0B2', '#FFF3E0']))
            st.plotly_chart(fig_types, use_container_width=True)
            
        st.dataframe(result.head(100), use_container_width=True)
    else:
        st.info("Vizualizasiya üçün məlumat yoxdur.")

# AI BÖLMƏSİ (Report Formatında)
with tab2:
    st.subheader("AI Biznes Strategiya Modulu")
    st.markdown("Aşağıdakı bölməyə biznes ideyanızı yazın. Süni intellekt məntiqi analiz edərək ən uyğun məkanları və koordinatları tapacaq.")
    
    api_key = os.getenv("API_KEY", "").strip()
    
    # Standart parametrlər
    radius_km = 1.5
    top_n = 3

    if "final_report_html" not in st.session_state:
        st.session_state.final_report_html = ""
    if "last_business_request" not in st.session_state:
        st.session_state.last_business_request = ""

    # Input daxil etmə forması
    with st.form("business_idea_form"):
        business_request = st.text_area(
            "Biznes ideyanızı ətraflı təsvir edin:",
            placeholder="Məsələn: Mən tələbələrə və ofis işçilərinə xitab edən ucuz fast-food restoranı açmaq istəyirəm...",
            height=120
        )
        submit_button = st.form_submit_button("Analiz Et və Hesabat Hazırla", type="primary")

    st.markdown("---")
    st.markdown("<h3 style='text-align: center; color: #4CAF50;'>Biznes Strategiya Hesabatı</h3>", unsafe_allow_html=True)
    report_placeholder = st.empty()

    if st.session_state.final_report_html:
        report_placeholder.markdown(st.session_state.final_report_html, unsafe_allow_html=True)
        if st.session_state.last_business_request:
            st.caption(f"Son analiz edilən sorğu: {st.session_state.last_business_request}")
    else:
        report_placeholder.info("Hesabat burada görünəcək. İdeyanızı yazıb 'Analiz Et və Hesabat Hazırla' düyməsinə klikləyin.")

    if submit_button:
        if not api_key:
            st.error("`.env` içində API_KEY (Groq) daxil edilməyib.")
            st.stop()
        
        if not business_request.strip():
            st.error("Zəhmət olmasa, biznes ideyanızı yazın.")
            st.stop()

        st.session_state.last_business_request = business_request.strip()

        with st.spinner("AI ideyanızı analiz edir və mövqeləri hesablayır (Bu proses bir neçə saniyə çəkə bilər)..."):
            client = get_groq_client(api_key=api_key)
            type_lookup = build_type_lookup(obj_types)

            # LLM 1 - PARSER
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
            
            try:
                intent_json = call_groq_json(client, parser_system_prompt, business_request)
            except Exception as e:
                report_placeholder.error(f"API Xətası (Mərhələ 1): {e}")
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
                report_placeholder.warning("LLM dataya uyğun düzgün obyekt tipi tapa bilmədi. İdeyanı daha fərqli sözlərlə yazın.")
                st.stop()

            base_df = df
            if selected_cities:
                base_df = base_df[base_df["CITY"].isin(selected_cities)]
            if selected_states:
                base_df = base_df[base_df["STATE"].isin(selected_states)]
                
            logic_payload = run_logic_module(base_df, target_types, resolved_keys, radius_km, top_n)
            logic_payload["ai_reasoning"] = intent_json.get("reasoning", "")

            if "error" in logic_payload or not logic_payload.get("top_locations"):
                report_placeholder.warning("Göstərilən şərtlərə uyğun ideal lokasiya (grid) tapılmadı.")
                st.stop()

            strategist_prompt = """Sən təcrübəli biznes məsləhətçisisən. İstifadəçinin YENİ biznes yeri üçün aparılan axtarışların riyazi nəticələri sənə JSON formasında veriləcək.

QAYDALAR:
- İstifadəçiyə HEÇ VAXT "burada rəqib azdır, ona görə restoran aç" demə, əgər ətrafda ona DƏSTƏK olan (müşdərini cəlb edəcək, məs. məktəb, ofis, xəstəxana və s.) obyekt yoxdursa!
- Təqdim olunan Top lokasiyaların NİYƏ LİSTƏ SALINDIĞINI sadə və inandırıcı dillə əsaslandır.
- Ton məsləhətçi tonu olsun: istifadəçiyə "bura yaxşıdır, çünki ...", "risk var, çünki ..." kimi səbəb-nəticə cümlələri ilə izah et.
- Xam data sıralamaq əvəzinə praktik qərar dəstəyi ver: nəyi niyə seçdiyini 1-2 cümlə ilə izah et.
- "nearby_examples" içindəki data hədəf üçün tapılmış real DƏSTƏK obyektləridir. Adlarını, küçələrini və məsafəsini mütləq istifadə et. (Məs: <b>Hədəf adı:</b> 200m)
- Koordinat/link uydurma. Yalnız Json-da gələn `maps_url` istifadə et.
- HƏR Top məkan üçün `maps_url` linkini məcburi şəkildə ayrıca sətirdə yaz: <b>Google Maps:</b> https://www.google.com/maps?q=...
- HƏR Top məkan üçün koordinatı da yaz: <b>Koordinat:</b> LAT, LON (yalnız center_lat/center_lon dəyərləri)
- FORMAT: Cavabını vizual cəhətdən gözəl görünən professional Markdown ilə ver, vacib nöqtələri (məsələn Reytinq, Ünvan, Xal) HTML <b> və ya <strong> tag-ləri ilə, ya da rəngli taglərlə (məs: <span style="color:#FF9800;">Xal: 95</span>) xüsusiləşdir. Mətni həddindən artıq bolt etmə, ancaq detallı başlıq və rəqəmləri vurğula.

STRUKTUR:
1. <b>Analitik Baxış</b>: (Ümumi tapıntılar və verilmiş "ai_reasoning" fikri)
2. <b>Məkan Analizi və Təkliflər</b>: (Top 1, Top 2, Top 3 xallar, səbəblər)
3. <b>Nəticə və Addımlar</b>: Biznesi qurmağa başlamaq üçün konkret və qısa tövsiyələr.
"""
            
            try:
                final_strategy = call_groq_text(client, strategist_prompt, json.dumps(logic_payload, ensure_ascii=False))

                # LLM linki buraxsa belə, top nəticələr üçün Google Maps linklərini həmişə göstəririk.
                top_locations = logic_payload.get("top_locations", [])
                maps_lines = ["<hr/><h4 style='margin-bottom:8px;'>Google Maps Koordinat Linkləri</h4>"]
                for idx, loc in enumerate(top_locations[:top_n], start=1):
                    city = str(loc.get("CITY") or "")
                    state = str(loc.get("STATE") or "")
                    lat = loc.get("center_lat")
                    lon = loc.get("center_lon")
                    maps_url = str(loc.get("maps_url") or "")
                    maps_lines.append(
                        f"<p style='margin:6px 0;'><b>Top {idx}</b> - {city}, {state}<br/>"
                        f"<b>Koordinat:</b> {lat}, {lon}<br/>"
                        f"<b>Google Maps:</b> <a href='{maps_url}' target='_blank'>{maps_url}</a></p>"
                    )

                maps_block = "".join(maps_lines)
                # Cavabı orta hissədə böyük şəkildə yazdırırıq
                final_report_html = (
                    f"<div style='background-color:#ffffff; padding:20px; border-radius:10px; "
                    f"box-shadow: 0px 4px 6px rgba(0,0,0,0.05);'>{final_strategy}{maps_block}</div>"
                )
                st.session_state.final_report_html = final_report_html
                report_placeholder.markdown(final_report_html, unsafe_allow_html=True)
            except Exception as e:
                report_placeholder.error(f"API Xətası (Mərhələ 3): {e}")
