# ReadFit — Reading Fit Analyzer
AASD 4016 Full Stack Data Science Systems · Group Project

Predicts how hard a passage is to read (trained model) AND whether a specific
student can decode it (deterministic phonics verifier).

## Quick start
    pip install -r requirements.txt
    # place CLEAR_corpus_final.xlsx in data/  (see below)
    python src/train.py          # ~25s, writes models/
    python app.py                # http://localhost:5000

## Data
    curl -L -o data/CLEAR_corpus_final.xlsx \
      https://raw.githubusercontent.com/scrosseye/CLEAR-Corpus/main/CLEAR_corpus_final.xlsx

CLEAR Corpus 6.01 — 4,724 excerpts, target `BT_easiness` derived from human
pairwise judgements. Metadata MIT-licensed.

## Endpoints
    GET  /            HTML form
    POST /analyze     HTML result
    POST /predict     JSON  (the data service)
    GET  /predict     JSON via query string
    GET  /benchmark   model comparison
    GET  /metrics     aggregate stats
    GET  /history     recent predictions
    GET  /lessons     phonics lessons
    GET  /health      liveness

## Deploy (Module 5 sequence)
    docker build -t tomiajibola/readfit:1.0 .
    docker push tomiajibola/readfit:1.0
    # on the GCP VM:
    sudo docker run -d --restart on-failure -p 5000:5000 tomiajibola/readfit:1.0

## Dashboard
    API_URL=http://34.139.45.76:5000 streamlit run dashboard/dashboard.py
