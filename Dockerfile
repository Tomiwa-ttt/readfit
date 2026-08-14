# ReadFit API - Module 4 pattern, python:3.11-slim (sklearn 1.6 needs modern python)
FROM python:3.11-slim

ADD . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# gunicorn, not the Flask dev server: e2-micro has 1GB RAM and the dev server
# warns against production use in the Module 4 slides.
ENTRYPOINT ["gunicorn"]
CMD ["--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
