import numpy as np
import pandas as pd

def compute_priors(y):
    classes = np.unique(y)
    priors = {}
    n = len(y)

    for c in classes:
        priors[c] = np.sum(y == c) / n

    return priors


def compute_gaussian_params(X, y, classes):
    means = {}
    vars_ = {}
    eps = 1e-9

    for c in classes:
        X_c = X[y == c]
        means[c] = np.mean(X_c, axis=0)
        vars_[c] = np.var(X_c, axis=0) + eps

    return means, vars_


def gaussian_log_pdf(x, mean, var):
    return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)


def predict_gaussian_naive_bayes(X, priors, means, vars_, classes):
    predictions = []

    for x in X:
        class_scores = {}

        for c in classes:
            log_prior = np.log(priors[c])
            log_likelihood = np.sum(gaussian_log_pdf(x, means[c], vars_[c]))
            class_scores[c] = log_prior + log_likelihood

        best_class = max(class_scores, key=class_scores.get)
        predictions.append(best_class)

    return np.array(predictions)


def predict_gaussian_naive_bayes_proba(X, priors, means, vars_, classes):
    probs_out = []

    for x in X:
        log_scores = []

        for c in classes:
            log_prior = np.log(priors[c])
            log_likelihood = np.sum(gaussian_log_pdf(x, means[c], vars_[c]))
            log_scores.append(log_prior + log_likelihood)

        log_scores = np.array(log_scores)
        log_scores -= np.max(log_scores)
        scores = np.exp(log_scores)
        probs = scores / np.sum(scores)

        if 1 in classes:
            idx_1 = np.where(classes == 1)[0][0]
            probs_out.append(probs[idx_1])
        else:
            probs_out.append(0.0)

    return np.array(probs_out)


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix_scratch(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))

    return TP, FP, FN, TN


def print_confusion_matrix(y_true, y_pred, label=""):
    TP, FP, FN, TN = confusion_matrix_scratch(y_true, y_pred)

    acc = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    print(f"\n{'='*40}")
    print(f"Confusion Matrix - {label}")
    print(f"{'='*40}")
    print(f"                 Predicted 0   Predicted 1")
    print(f"  Actual 0   :     {TN:5d}         {FP:5d}")
    print(f"  Actual 1   :     {FN:5d}         {TP:5d}")
    print(f"{'='*40}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    feature_cols = [
        'completions_home',           'completions_away',
        'attempts_home',              'attempts_away',
        'passing_yards_home',         'passing_yards_away',
        'passing_tds_home',           'passing_tds_away',
        'passing_interceptions_home', 'passing_interceptions_away',
        'passing_first_downs_home',   'passing_first_downs_away',
        'sacks_suffered_home',        'sacks_suffered_away',
        'sack_yards_lost_home',       'sack_yards_lost_away',
        'carries_home',               'carries_away',
        'rushing_yards_home',         'rushing_yards_away',
        'rushing_tds_home',           'rushing_tds_away',
        'rushing_first_downs_home',   'rushing_first_downs_away',
        'rushing_fumbles_lost_home',  'rushing_fumbles_lost_away',
        'sack_fumbles_lost_home',     'sack_fumbles_lost_away',
        'def_sacks_home',             'def_sacks_away',
        'def_interceptions_home',     'def_interceptions_away',
        'def_tackles_for_loss_home',  'def_tackles_for_loss_away',
        'def_fumbles_forced_home',    'def_fumbles_forced_away',
        'def_qb_hits_home',           'def_qb_hits_away',
        'def_tds_home',               'def_tds_away',
        'fg_made_home',               'fg_made_away',
        'fg_missed_home',             'fg_missed_away',
        'fg_pct_home',                'fg_pct_away',
        'pat_made_home',              'pat_made_away',
        'penalties_home',             'penalties_away',
        'penalty_yards_home',         'penalty_yards_away',
        'overtime',
        'rest_home',
        'rest_away',
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]

    df_clean = df[feature_cols + ['home_win']].dropna()
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce').dropna()

    X = df_clean[feature_cols].values.astype(float)
    y = df_clean['home_win'].values.astype(int)

    print(f"  Features selected : {len(feature_cols)}")
    print(f"  Dataset shape     : {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Class balance     : {y.mean():.2%} home wins")

    return X, y, feature_cols


def train_val_split(X, y, val_ratio=0.2):
    N = len(y)
    split = int(N * (1 - val_ratio))

    X_train = X[:split]
    y_train = y[:split]
    X_val = X[split:]
    y_val = y[split:]

    print(f"  Train: {len(y_train)} samples | Val: {len(y_val)} samples")
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    import os

    print("=" * 60)
    print("  NFL Winner Prediction — Gaussian Naive Bayes")
    print("=" * 60)

    DATA_PATH = "../data/nfl_data.csv"

    
    
    X, y, feature_names = load_and_prepare_data(DATA_PATH)
 

 
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2)


    classes = np.unique(y_train)
    priors = compute_priors(y_train)
    means, vars_ = compute_gaussian_params(X_train, y_train, classes)

  
    y_train_pred = predict_gaussian_naive_bayes(X_train, priors, means, vars_, classes)
    y_val_pred = predict_gaussian_naive_bayes(X_val, priors, means, vars_, classes)

    train_acc = compute_accuracy(y_train, y_train_pred)
    val_acc = compute_accuracy(y_val, y_val_pred)

    print(f"\n  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy  : {val_acc:.4f}")

    print_confusion_matrix(y_train, y_train_pred, label="Training Set")
    print_confusion_matrix(y_val, y_val_pred, label="Validation Set")

    print("\n[5] Class priors:")
    for c in classes:
        print(f"  P(y={c}) = {priors[c]:.4f}")

  