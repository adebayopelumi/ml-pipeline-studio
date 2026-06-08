# ML Pipeline Studio

Train, evaluate, and predict with any dataset — no code required.

ML Pipeline Studio is a desktop application for machine learning. Upload a CSV, pick your target column, choose an algorithm, and walk away with a trained model, performance charts, SHAP explanations, and a downloadable Excel report — all through a clean dark interface.

---

## The workflow

Upload a dataset → pick what you want to predict → train a model → understand the results → make predictions.

Every step happens in the app. No notebooks, no terminal, no code.

---

## What's inside

**Train**
Upload any CSV file. The app figures out whether you're doing classification or regression, lets you pick the target column and algorithm, and trains the model. Optuna handles hyperparameter tuning automatically if you want it.

**Results**
After training you get the full picture — accuracy metrics explained in plain English, a confusion matrix, ROC and Precision-Recall curves, feature importance, and SHAP values that break down exactly why the model made each decision.

**Predict**
Fill in a form to get a single prediction with a confidence score and a SHAP waterfall chart. Or upload a CSV and download the full batch results.

**Admin**
Add and remove user accounts. Each user's data is completely separate — nobody sees anyone else's models or results.

---

## Algorithms supported

Random Forest · Logistic Regression · XGBoost · LightGBM · Ridge · Lasso · Gradient Boosting · SVM

Works for binary classification, multiclass classification, and regression.

---

## Getting started

Clone the repo and install dependencies:

```bash
git clone https://github.com/adebayopelumi/ml-pipeline-studio.git
cd ml-pipeline-studio
pip install -r requirements.txt
```

Copy the example auth config and fill in your details:

```bash
cp config/auth_config.example.yaml config/auth_config.yaml
```

Run the app:

```bash
streamlit run app/main.py
```

---

## Built with

Streamlit · scikit-learn · XGBoost · LightGBM · Optuna · SHAP · Plotly · FastAPI · pywebview

---

## License

MIT
