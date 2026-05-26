# Deployment Guide — ML Pipeline Diabetes (Render)

Generated: 2026-05-26 | Service: `ml-pipeline-diabetes` | Platform: Render

---

## Prerequisites

- GitHub repository is **public** at `https://github.com/ramleo/ML-Pipeline-Diabetes`
- A free account at [render.com](https://render.com) (sign up with GitHub for seamless repo access)
- `render.yaml` is committed to the root of the repository (auto-detected by Render)

---

## Option 1 — Deploy via Render Dashboard (Manual)

1. Log in to [dashboard.render.com](https://dashboard.render.com)
2. Click **"New +"** → select **"Web Service"**
3. Click **"Connect a repository"** → authorise GitHub if prompted
4. Search for and select **`ramleo/ML-Pipeline-Diabetes`**
5. Fill in the service settings:

   | Field | Value |
   |---|---|
   | **Name** | `ml-pipeline-diabetes` |
   | **Region** | Oregon (US West) — or closest to your users |
   | **Branch** | `main` |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
   | **Instance Type** | Free (or Starter for always-on) |

6. Under **"Environment Variables"**, add:

   | Key | Value |
   |---|---|
   | `PYTHON_VERSION` | `3.11.0` |

7. Click **"Create Web Service"** — Render will pull the repo, install dependencies, and start the server.
8. Wait for the build log to show `Uvicorn running on http://0.0.0.0:XXXX` — status changes to **Live**.

---

## Option 2 — Deploy via `render.yaml` (Auto-Detected)

Render automatically detects `render.yaml` in the repository root on every push.

1. Go to [dashboard.render.com/new/blueprint](https://dashboard.render.com/new/blueprint)
2. Connect the repository **`ramleo/ML-Pipeline-Diabetes`**
3. Render reads `render.yaml` and pre-fills all settings
4. Click **"Apply"** — the service is created and deployed automatically

Any subsequent `git push origin main` triggers a new deploy with zero additional configuration.

---

## Environment Variables

| Key | Value | Required | Notes |
|---|---|---|---|
| `PYTHON_VERSION` | `3.11.0` | Yes | Pins the Python runtime on Render |
| `PORT` | _(auto-set by Render)_ | — | Do not set manually; Render injects this |

---

## Post-Deploy Smoke Tests

Replace `<YOUR_RENDER_URL>` with your live Render URL (format: `https://ml-pipeline-diabetes.onrender.com`).

### Health check
```bash
curl https://<YOUR_RENDER_URL>/health
# Expected: {"status": "ok"}
```

### Single prediction
```bash
curl -X POST https://<YOUR_RENDER_URL>/predict \
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
# Expected: {"prediction": 1, "probability": [...]}
```

### Batch prediction
```bash
curl -X POST https://<YOUR_RENDER_URL>/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50},
      {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31}
    ]
  }'
# Expected: {"predictions": [1, 0], "probabilities": [...]}
```

### Interactive API docs
```
https://<YOUR_RENDER_URL>/docs       ← Swagger UI
https://<YOUR_RENDER_URL>/redoc      ← ReDoc
```

---

## Monitoring & Logs

- **Live logs:** Render Dashboard → select service → **"Logs"** tab — streams stdout/stderr in real time
- **Metrics:** Dashboard → **"Metrics"** tab — CPU, memory, and request latency graphs
- **Events:** Dashboard → **"Events"** tab — full deploy history with timestamps and commit SHAs
- **Health checks:** Render pings `/health` every 30 seconds; the service is marked unhealthy and restarted automatically if the endpoint returns non-2xx
- **Alerts:** Dashboard → **"Notifications"** — configure email/Slack alerts for deploy failures or downtime
- **Free tier note:** Free-tier services spin down after 15 minutes of inactivity; the first request after spin-down may take 30–60 seconds (cold start). Upgrade to **Starter** ($7/mo) for always-on behaviour.

---

## Re-Deploy Instructions

### Automatic (recommended)
Every `git push origin main` triggers a new deploy automatically — no manual action needed.

```bash
git add .
git commit -m "Your change description"
git push origin main
```

Render detects the push via webhook, rebuilds the image, and performs a zero-downtime swap.

### Manual re-deploy
1. Render Dashboard → select service `ml-pipeline-diabetes`
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

### Rollback
1. Dashboard → **"Events"** tab
2. Find the previous successful deploy
3. Click **"Rollback to this deploy"**

---

## Useful Links

| Resource | URL |
|---|---|
| Render Dashboard | https://dashboard.render.com |
| New Web Service (Blueprint) | https://dashboard.render.com/new/blueprint |
| Service direct link | https://dashboard.render.com/web/ml-pipeline-diabetes |
| GitHub Repository | https://github.com/ramleo/ML-Pipeline-Diabetes |
| Render Docs — Python | https://render.com/docs/deploy-fastapi |
