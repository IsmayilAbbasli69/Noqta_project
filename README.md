# Azerbaijan Objects Map (Streamlit)

Bu layihə `dataset/azrbaycanda_obyektlr_v_mkanlar.geojson` faylını oxuyur və xəritədə filtrli baxış təqdim edir.

## Funksiyalar
- `CITY` üzrə filtr
- `STATE` üzrə filtr
- `TYPE` üzrə filtr
- Xəritədə nöqtə göstərimi
- Filtrlənmiş cədvəl
- CSV export
- LLM -> Logic Module -> LLM strategiya axını

## LLM axını necə işləyir
1. 1-ci LLM istifadəçi mətnini strukturlaşdırılmış JSON-a çevirir: `{"object": ..., "keys": [...]}`
2. Logic module həmin JSON əsasında məsafə və rəqib sıxlığını hesablayır
3. 2-ci LLM logic nəticəsini biznes strategiya mətni kimi izah edir

## Quraşdırma

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## API açarı

Layihə kök qovluğunda `.env` faylı yaradın və əlavə edin:

```env
API_KEY=your_groq_api_key
```

## İşə salma

```bash
streamlit run app.py
```

Sonra brauzerdə Streamlit URL açılacaq (adətən `http://localhost:8501`).
