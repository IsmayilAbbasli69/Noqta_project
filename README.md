# Nöqtə: Geo-AI Business Location Intelligence

## Overview
Nöqtə is a Streamlit-based decision-support system that recommends high-potential business locations using a hybrid approach:
- LLM-driven intent parsing and strategic report generation
- Rule-based geospatial scoring with distance, demand, competition, and socio-economic signals

The platform is designed for hackathon/demo scenarios where users describe a business idea and receive:
1. Ranked location candidates
2. Explainable reasons
3. Google Maps links for direct validation
4. Market context (estimated land/rent ranges, purchasing power)

## Dataset
The project uses a large public geospatial dataset from Open Data Azerbaijan (`opendata.az`) with approximately 300K+ records, stored in:

`dataset/azrbaycanda_obyektlr_v_mkanlar.geojson`

Each record contains fields such as:
- `CITY`
- `STATE`
- `TYPE`
- `NAME`
- `STRT_NAME`
- Coordinates (`lat`, `lon`)

## Problem It Solves
Business owners often ask: “Where should I open this business?”

Nöqtə answers this with a transparent pipeline instead of a pure black-box response:
- It first interprets user intent
- Then computes location scores mathematically
- Finally explains results in business language

## End-to-End Pipeline
### 1) User Input
The user enters a business idea in natural language (for example: budget restaurant near students and transit).

### 2) LLM Stage 1: Intent Parsing
The first LLM call converts user text into structured intent:
- `object`
- `targets`
- `keys`

This stage identifies:
- Target business category (or close alternatives)
- Supporting POI categories that can bring demand

### 3) Type Resolution
Parsed labels are normalized and mapped to actual dataset `TYPE` values.

### 4) Geo-Logic Scoring Engine
The core algorithm (`geo_logic.py`) evaluates many candidate grid cells and computes an explainable score for each.

### 5) LLM Stage 2: Strategy Narrative
The second LLM call transforms the scoring payload into an advisory report with recommendations, risk framing, and market context.

### 6) Visual Output
The app renders:
- 3D map layers
- Summary charts
- Competitor analysis charts/tables from returned AI payload
- Clickable Google Maps links

## Mathematical/Algorithmic Design
The engine is a heuristic multi-factor scoring model (not a trained ML model).

### Candidate Generation
Coordinates are rounded to a 3-decimal grid (about 110m resolution), reducing noise and creating candidate cells.

### Distance Computation
Distance uses Haversine geometry for spherical accuracy.

### Signals Per Candidate
For each candidate center:
- `support_density`: nearby support POIs per local block size
- `supporting_type_coverage`: number of unique support categories present
- `demand_proxy`: `log1p(local_block_size)`
- `competitors`: number of target-type competitors in radius

### Base Score
The base score formula:

`base_score = 4.0*support_density + 2.5*coverage + 0.5*demand_proxy - 1.75*competitors`

### Weighted Final Score
Final score applies two multipliers:
1. `region_w`: data density weight by city/state
2. `socio_w`: socio-economic weight (Baku district density/salary/employment aware)

`final_score = base_score * region_w * socio_w`

### Diversity Control
Top locations are filtered for diversity:
- avoids repeated same-street picks
- avoids near-duplicate candidates (distance threshold)

### Output Payload
Each candidate includes:
- center coordinates
- score and score components
- support examples (`nearby_examples`)
- competitor examples (`nearby_competitors`)
- market snapshot (`market_snapshot`)
- Google Maps URL

## Market Context Layer
For Baku districts, the logic layer injects estimated contextual fields:
- land price range (AZN/m2)
- rent range (AZN/m2/month)
- salary and density references
- purchasing power level

These are advisory ranges for decision support, not legal valuation.

## Tech Stack
- Python 3.x
- Streamlit
- Pandas / NumPy
- PyDeck (3D map rendering)
- Plotly (analytics charts)
- Groq API (LLM calls)

## Repository Structure
- `app.py`: Streamlit app, UI flow, LLM orchestration, visualization
- `geo_logic.py`: geospatial scoring and ranking engine
- `core_utils.py`: normalization, JSON parsing, type resolution, haversine math
- `llm_agent.py`: Groq API integration utilities
- `dataset/`: GeoJSON source data
- `requirements.txt`: Python dependencies

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables
Create `.env` in the project root:

```env
API_KEY=your_groq_api_key
```

## Run
```bash
streamlit run app.py
```

Then open the local Streamlit URL (typically `http://localhost:8501`).


