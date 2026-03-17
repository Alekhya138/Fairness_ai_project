# FairLens v2 — Universal AI Bias Detector

Works on **ANY** user-uploaded CSV. No preloaded demo dependency.

---

## 📁 Structure
```
fairlens2/
├── frontend/
│   └── index.html              ← Complete single-file frontend
├── backend/
│   ├── app.py                  ← Flask API — all fairness logic
│   └── requirements.txt
├── data/
│   ├── generate.py             ← Generates 3 test datasets
│   ├── hiring_race_bias.csv    ← Test: race bias in hiring
│   ├── credit_gender_bias.csv  ← Test: gender bias in lending
│   └── medical_age_bias.csv    ← Test: age bias in medical trials
└── README.md
```

---

## 🚀 Quick Start

### Option A — Frontend Only (no backend)
```bash
open frontend/index.html
```
Works standalone using client-side CSV parsing (PapaParse) + JS simulation.
Upload any CSV and get bias analysis immediately.

### Option B — Full Stack (real ML backend)
```bash
cd backend
pip install -r requirements.txt
python app.py
# → http://localhost:5000

# Open frontend:
open ../frontend/index.html
```

---

## 📊 Test Datasets

| File | Domain | Protected Attr | Bias Level |
|---|---|---|---|
| `hiring_race_bias.csv` | Job hiring | race | ~30% gap (Black vs White) |
| `credit_gender_bias.csv` | Loan approval | gender | ~14% gap (Female vs Male) |
| `medical_age_bias.csv` | Clinical trial | age_group | ~36% gap (Over50 vs Under30) |

---

## 🔷 API Endpoints

| Route | Method | Purpose |
|---|---|---|
| `GET  /api/health` | GET | Status check |
| `POST /api/inspect` | POST | Upload CSV → get column info + auto-detection |
| `POST /api/analyze` | POST | Full bias analysis + mitigation |
| `POST /api/whatif` | POST | What-if simulator |
| `GET  /api/history` | GET | Past analyses |
| `GET  /api/history/<id>` | GET | Load specific analysis |

### POST /api/analyze parameters
```
file          : multipart CSV file
sensitive_col : column name (e.g. "gender")
target_col    : binary outcome column (e.g. "approved")
mitigation    : "pre,in,post" (comma separated)
```

---

## 🔷 How Bias Detection Works

1. **Load & Clean**: Handle nulls, strip whitespace, encode categoricals
2. **Auto-detect**: Keyword-match sensitive cols + binary target cols
3. **Baseline model**: LogisticRegression → predictions → fairness metrics
4. **Metrics**: DI, SPD, EO, Accuracy → Fairness Score 0–100
5. **Conditional fix**: Only applied if bias_detected=True
6. **Comparison**: Before/after metrics + % improvement

---

## 🔷 Two-Mode Architecture

The frontend operates in two modes automatically:

**Client Mode** (no backend):
- PapaParse reads CSV in browser
- Keyword-based column detection
- Statistical simulation of bias metrics
- All charts, heatmaps, what-if fully functional

**Server Mode** (Flask running):
- Real LogisticRegression model
- Fairlearn ExponentiatedGradient in-processing
- Per-group threshold optimization
- SQLite history persistence
- Feature importance from model coefficients

---

## 🔷 Bias Metrics Reference

| Metric | Formula | Fair | Biased |
|---|---|---|---|
| Disparate Impact | min_rate/max_rate | ≥0.80 | <0.60 |
| Stat. Parity Diff | max_rate−min_rate | ≈0 | >0.20 |
| Equal Opportunity | ΔTrue Positive Rate | ≈0 | >0.15 |
| Fairness Score | Composite 0–100 | ≥75 | <50 |

---

## 🔷 Demo Flow

1. Open `frontend/index.html`
2. Upload `data/credit_gender_bias.csv`
3. Auto-detected: gender → approved
4. Click **Analyze Dataset**
5. See RED alert — DI ~0.77, FS ~52
6. Read XAI explanation: "Female approval rate X% lower than Male"
7. Click **Apply Bias Fix**
8. See GREEN result — DI improved, FS 70+
9. Try What-If: change gender → decision flips → bias confirmed

---
MIT License
