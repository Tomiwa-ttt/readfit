"""
ReadFit - model training + benchmarking.
OWNER: Lalit Kumar (Track 2)

Trains a readability regressor on the CLEAR corpus (4,724 human-rated excerpts).
Target = BT_easiness, a Bradley-Terry score derived from ~111,000 pairwise human
judgements. It is NOT a formula - which is why beating Flesch-Kincaid is a real result.

Writes:
  models/vectorizer.joblib
  models/readability_model.joblib
  models/metrics.json      <- feeds BOTH the benchmarking slide and the dashboard

Run:  python src/train.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import linguistic_matrix, LINGUISTIC_NAMES, flesch_kincaid_grade

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "CLEAR_corpus_final.xlsx")
MODELS = os.path.join(ROOT, "models")
SEED = 42


def load():
    df = pd.read_excel(DATA)
    df = df[["Excerpt", "BT_easiness", "Flesch-Kincaid-Grade-Level"]].dropna()
    df = df[df["Excerpt"].str.split().str.len() > 20]
    print(f"loaded {len(df)} excerpts")
    return df


def main():
    os.makedirs(MODELS, exist_ok=True)
    t0 = time.time()
    df = load()

    X_text = df["Excerpt"].tolist()
    y = df["BT_easiness"].values
    fk = df["Flesch-Kincaid-Grade-Level"].values

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)

    print("extracting linguistic features...")
    L = linguistic_matrix(X_text)
    scaler = StandardScaler().fit(L[tr])
    Ls = scaler.transform(L)

    print("fitting tf-idf...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                            min_df=3, sublinear_tf=True, stop_words=None)
    tfidf.fit([X_text[i] for i in tr])
    T = tfidf.transform(X_text)

    Combined = hstack([csr_matrix(Ls), T]).tocsr()

    results = []

    def record(name, yhat_te, note=""):
        rmse = float(np.sqrt(mean_squared_error(y[te], yhat_te)))
        results.append({
            "model": name,
            "rmse": round(rmse, 4),
            "mae": round(float(mean_absolute_error(y[te], yhat_te)), 4),
            "r2": round(float(r2_score(y[te], yhat_te)), 4),
            "note": note,
        })
        print(f"  {name:38s} RMSE={rmse:.4f}")

    print("benchmarking...")
    # 1. Floor: predict the mean.
    d = DummyRegressor(strategy="mean").fit(Combined[tr], y[tr])
    record("Baseline - predict mean", d.predict(Combined[te]), "no information")

    # 2. The honest incumbent: Flesch-Kincaid alone, fitted linearly.
    lr = LinearRegression().fit(fk[tr].reshape(-1, 1), y[tr])
    record("Flesch-Kincaid grade level only", lr.predict(fk[te].reshape(-1, 1)),
           "the formula schools use today - this is the number to beat")

    # 3. Our 15 linguistic features, linear.
    r1 = Ridge(alpha=1.0).fit(Ls[tr], y[tr])
    record("Ridge - 15 linguistic features", r1.predict(Ls[te]), "no lexical content")

    # 4. Trees on linguistic features.
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                               random_state=SEED, n_jobs=-1).fit(Ls[tr], y[tr])
    record("RandomForest - linguistic", rf.predict(Ls[te]))

    gb = GradientBoostingRegressor(random_state=SEED).fit(Ls[tr], y[tr])
    record("GradientBoosting - linguistic", gb.predict(Ls[te]))

    # 5. Linguistic + TF-IDF, tuned. This is the candidate for deployment.
    print("grid searching Ridge(linguistic + tf-idf)...")
    gs = GridSearchCV(Ridge(), {"alpha": [0.3, 1.0, 3.0, 10.0, 30.0]},
                      cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(Combined[tr], y[tr])
    best = gs.best_estimator_
    record(f"Ridge - linguistic + TF-IDF (tuned a={gs.best_params_['alpha']})",
           best.predict(Combined[te]), "DEPLOYED MODEL")

    results.sort(key=lambda r: r["rmse"])
    winner = results[0]
    print(f"\nwinner: {winner['model']}  RMSE={winner['rmse']}")

    # Tuning curve for the slide.
    tuning = [{"alpha": float(a), "cv_rmse": round(float(-s), 4)}
              for a, s in zip(gs.cv_results_["param_alpha"].data,
                              gs.cv_results_["mean_test_score"])]

    # Feature importance for the dashboard (from the RF - interpretable).
    importance = sorted(
        [{"feature": n, "importance": round(float(v), 4)}
         for n, v in zip(LINGUISTIC_NAMES, rf.feature_importances_)],
        key=lambda x: -x["importance"])

    # Band cut-points from the TRAINING distribution only.
    cuts = [float(np.quantile(y[tr], q)) for q in (0.2, 0.4, 0.6, 0.8)]

    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "CLEAR Corpus 6.01 (CommonLit + Georgia State)",
        "n_total": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
        "target": "BT_easiness (Bradley-Terry, human pairwise judgements)",
        "seed": SEED,
        "benchmark": results,
        "winner": winner,
        "tuning_curve": tuning,
        "feature_importance": importance,
        "band_cuts": cuts,
        "train_seconds": round(time.time() - t0, 1),
    }

    joblib.dump({"tfidf": tfidf, "scaler": scaler}, os.path.join(MODELS, "vectorizer.joblib"))
    joblib.dump(best, os.path.join(MODELS, "readability_model.joblib"))
    with open(os.path.join(MODELS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nwrote models/ in {metrics['train_seconds']}s")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
