# ML Pipeline Studio

> Train, evaluate, and predict with any supervised learning dataset — no code required.

ML Pipeline Studio is a professional desktop application for machine learning. Upload a CSV, pick your target column, choose an algorithm, and get a fully trained model with charts, explanations, and downloadable results — all through a clean, dark UI.

---

## What it does

```
┌─────────────────────────────────────────────────────────────────┐
│                      ML Pipeline Studio                         │
│                                                                 │
│   📂 Upload CSV                                                 │
│        │                                                        │
│        ▼                                                        │
│   🎯 Pick target column  ──►  🤖 Choose algorithm              │
│                                      │                          │
│                                      ▼                          │
│                              ⚙️  Train model                    │
│                                      │                          │
│              ┌───────────────────────┼──────────────────┐      │
│              ▼                       ▼                   ▼      │
│        📊 Results               ⚡ Predict          💾 Export  │
│     Metrics, charts,         Single row or         Excel report │
│     SHAP explanations        batch CSV file        (.xlsx)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

**Training**
- Works with any CSV — classification or regression, any number of columns
- Algorithms: Random Forest, Logistic Regression, XGBoost, LightGBM, and more
- Automatic hyperparameter tuning with Optuna
- Detects problem type (binary, multiclass, regression) automatically

**Results & Explainability**
- Performance metrics with plain-English explanations
- Confusion matrix, ROC curve, Precision-Recall curve
- Feature importance chart
- SHAP values — shows exactly why the model made each prediction

**Predictions**
- Single prediction form — fill in values, get a result instantly
- Batch predictions — upload a CSV, download results
- Confidence score and risk level for every prediction

**Built for real use**
- Secure login with per-user data isolation — each account sees only their own models
- Results saved to disk — survive server restarts
- Export full report as Excel (.xlsx) — opens in Excel or Numbers on Mac
- Admin panel to manage users

---

## Pages

```
┌──────────────┬──────────────────────────────────────────────────┐
│   Sidebar    │                  Main content                    │
│              │                                                  │
│  🧠 Train    │  Upload data → configure → train → see metrics  │
│  ⚡ Predict  │  Enter values or upload CSV → get predictions    │
│  📊 Results  │  Charts, SHAP, confusion matrix, export         │
│  🛡️ Admin    │  Manage users (admin only)                      │
│              │                                                  │
│  [Sign out]  │                                                  │
│  [Reset all] │                                                  │
└──────────────┴──────────────────────────────────────────────────┘
```

---

## Getting started

**1. Clone and install**

```bash
git clone https://github.com/adebayopelumi/ml-pipeline-studio.git
cd ml-pipeline-studio
pip install -r requirements.txt
```

**2. Set up your login**

Copy `config/auth_config.example.yaml` to `config/auth_config.yaml` and fill in your details. Passwords are auto-hashed on first run.

**3. Run the app**

```bash
streamlit run app/main.py
```

---

## Tech stack

| Layer | Tools |
|-------|-------|
| UI | Streamlit |
| ML | scikit-learn, XGBoost, LightGBM |
| Tuning | Optuna |
| Explainability | SHAP |
| Charts | Plotly |
| Auth | streamlit-authenticator |
| API | FastAPI |
| Export | openpyxl |
| Desktop app | pywebview (macOS) |

---

## How to use it

**Training a model:**
1. Go to the **Train** page
2. Upload your CSV file
3. Select the column you want to predict
4. Choose an algorithm and tuning mode
5. Click **Train** — the model trains and results appear instantly

**Making predictions:**
1. Go to the **Predict** page
2. Fill in the input form for a single prediction, or upload a CSV for batch predictions
3. Results show the prediction, confidence score, and a SHAP chart explaining the reasoning

**Viewing results:**
1. Go to the **Results** page
2. Browse metrics, charts, and SHAP explanations
3. Click **Download full report (.xlsx)** to export everything

---

## License

MIT
