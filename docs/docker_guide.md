# Docker Guide — ML-Pipeline-Diabetes

Generated: 2026-05-26  
Image: `ml-pipeline-diabetes:latest`  
Base: `python:3.11-slim` (multi-stage build)  
Compressed image size: ~176 MB

---

## Build

```bash
docker build -t ml-pipeline-diabetes:latest .
```

---

## Run

```bash
docker run -d -p 8000:8000 --name diabetes-api ml-pipeline-diabetes:latest
```

The API will be available at `http://localhost:8000`.  
Interactive API docs (Swagger UI): `http://localhost:8000/docs`

---

## Smoke-Test curl Examples

### Health check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok","model":"LogisticRegression","version":"1.0.0"}
```

---

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

Expected response:

```json
{
  "prediction": 1,
  "label": "Diabetes",
  "probability": {
    "no_diabetes": 0.2728,
    "diabetes": 0.7272
  }
}
```

---

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
      },
      {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "SkinThickness": 29,
        "Insulin": 0,
        "BMI": 26.6,
        "DiabetesPedigreeFunction": 0.351,
        "Age": 31
      }
    ]
  }'
```

Expected response:

```json
{
  "predictions": [
    {"prediction": 1, "label": "Diabetes",    "probability": {"no_diabetes": 0.2728, "diabetes": 0.7272}},
    {"prediction": 0, "label": "No Diabetes", "probability": {"no_diabetes": 0.9613, "diabetes": 0.0387}}
  ]
}
```

---

### Swagger UI

Open a browser and navigate to:

```
http://localhost:8000/docs
```

---

## Post-Deploy Tests (replace `localhost:8000` with your live URL)

```bash
LIVE_URL="https://your-service.onrender.com"  # or fly.dev / railway.app / etc.

curl ${LIVE_URL}/health

curl -X POST ${LIVE_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}'
```

---

## Useful Docker Commands Reference

| Command | Description |
|---|---|
| `docker build -t ml-pipeline-diabetes:latest .` | Build the image |
| `docker run -d -p 8000:8000 --name diabetes-api ml-pipeline-diabetes:latest` | Start container in background |
| `docker run --rm -p 8000:8000 ml-pipeline-diabetes:latest` | Start container, auto-remove on stop |
| `docker ps` | List running containers |
| `docker logs diabetes-api` | View container logs |
| `docker logs -f diabetes-api` | Follow (tail) container logs |
| `docker exec -it diabetes-api /bin/bash` | Shell into running container |
| `docker stop diabetes-api` | Stop the container |
| `docker rm diabetes-api` | Remove the stopped container |
| `docker stop diabetes-api && docker rm diabetes-api` | Stop and remove in one step |
| `docker images ml-pipeline-diabetes` | List image(s) |
| `docker rmi ml-pipeline-diabetes:latest` | Delete the image |
| `docker system prune -f` | Remove all unused images and containers |

---

## Image Details

| Property | Value |
|---|---|
| Image name | `ml-pipeline-diabetes:latest` |
| Base image | `python:3.11-slim` |
| Build strategy | Multi-stage (builder + runtime) |
| Compressed size | ~176 MB |
| Exposed port | 8000 |
| Runtime user | `appuser` (non-root) |
| Entrypoint | `uvicorn app:app --host 0.0.0.0 --port 8000` |
| Model artifact | `models/final_pipeline.pkl` (LogisticRegression pipeline) |
