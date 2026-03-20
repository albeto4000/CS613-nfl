import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    """Squash any real number into the range (0, 1)."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def z_score_normalize(X_train, X_val=None):
    mean = np.mean(X_train, axis=0)
    std  = np.std(X_train, axis=0)
    std[std == 0] = 1.0  

    X_train_norm = (X_train - mean) / std

    if X_val is not None:
        X_val_norm = (X_val - mean) / std
        return X_train_norm, X_val_norm, mean, std

    return X_train_norm, mean, std


def add_bias_column(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def compute_accuracy(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    return np.mean(y_pred == y_true)


def compute_confusion_matrix(y_true, y_pred_prob, threshold=0.5):

    y_pred = (y_pred_prob >= threshold).astype(int)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    return TP, FP, FN, TN


def compute_metrics(TP, FP, FN, TN):
    """Compute accuracy, precision, recall, and F1 from confusion matrix counts."""
    accuracy  = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return accuracy, precision, recall, f1


def print_evaluation(y_true, y_pred_prob, label=""):
    """Print confusion matrix and metrics for a set of predictions."""
    TP, FP, FN, TN = compute_confusion_matrix(y_true, y_pred_prob)
    accuracy, precision, recall, f1 = compute_metrics(TP, FP, FN, TN)

    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"                 Predicted 0   Predicted 1")
    print(f"  Actual 0   :     {TN:5d}         {FP:5d}")
    print(f"  Actual 1   :     {FN:5d}         {TP:5d}")
    print(f"{'='*40}")
    print(f"  Accuracy   : {accuracy:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1 Score   : {f1:.4f}")



class LogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000, seed=42):
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.seed          = seed

        self.w            = None
        self.mean         = None
        self.std          = None
        self.train_losses = []
        self.val_losses   = []
        self.train_accs   = []
        self.val_accs     = []


    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_tr, X_v = self._normalize_and_add_bias(X_train, X_val)
        self._initialize_weights(X_tr.shape[1])

        y_train = y_train.astype(float)
        y_val   = y_val.astype(float) if y_val is not None else None

        for epoch in range(self.epochs):
            self._gradient_descent_step(X_tr, y_train)
            self._record_metrics(X_tr, y_train, X_v, y_val)
            self._print_progress(epoch)

        print("\n  Training complete.")

    def predict_proba(self, X):
        """Return predicted win probabilities for each game in X."""
        X_norm = (X - self.mean) / self.std
        X_bias = add_bias_column(X_norm)
        return sigmoid(X_bias @ self.w)

    def predict(self, X, threshold=0.5):
        """Return binary predictions (1 = home win, 0 = home loss)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def plot_training_curves(self, title="Logistic Regression"):
        """Plot loss and accuracy curves over training epochs."""
        epochs = range(1, len(self.train_losses) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        self._plot_curve(axes[0], epochs, self.train_losses, self.val_losses,
                         ylabel="Binary Cross-Entropy Loss",
                         title=f"{title} - Loss Curve")
        self._plot_curve(axes[1], epochs, self.train_accs, self.val_accs,
                         ylabel="Accuracy",
                         title=f"{title} - Accuracy Curve")

        plt.tight_layout()
        plt.savefig("lr_training_curves.png", dpi=150)
        plt.show()
        print("  Saved: lr_training_curves.png")

    def print_top_features(self, feature_names, top_n=15):
        """Print and plot the features with the largest absolute weights."""
        top_features, top_weights = self._get_top_features(feature_names, top_n)

        print(f"\n  Top {top_n} Feature Weights (by absolute magnitude):")
        for name, weight in zip(top_features, top_weights):
            print(f"    {name:<40s}  {weight:+.4f}")

        self._plot_feature_importance(top_features, top_weights)

   
    def _normalize_and_add_bias(self, X_train, X_val):
        """Normalize features and prepend the bias column."""
        if X_val is not None:
            X_tr, X_v, self.mean, self.std = z_score_normalize(X_train, X_val)
            return add_bias_column(X_tr), add_bias_column(X_v)

        X_tr, self.mean, self.std = z_score_normalize(X_train)
        return add_bias_column(X_tr), None

    def _initialize_weights(self, num_weights):
        """Set weights to small uniform random values for symmetry breaking."""
        rng    = np.random.default_rng(self.seed)
        self.w = rng.uniform(-1e-4, 1e-4, size=num_weights)

    def _gradient_descent_step(self, X, y_true):
        """Compute gradient and update weights by one step."""
        y_hat    = sigmoid(X @ self.w)
        gradient = (1.0 / len(y_true)) * (X.T @ (y_hat - y_true))
        self.w  -= self.learning_rate * gradient

    def _record_metrics(self, X_tr, y_train, X_v, y_val):
        """Append loss and accuracy for the current epoch."""
        y_hat_train = sigmoid(X_tr @ self.w)
        self.train_losses.append(binary_cross_entropy(y_train, y_hat_train))
        self.train_accs.append(compute_accuracy(y_train, y_hat_train))

        if X_v is not None and y_val is not None:
            y_hat_val = sigmoid(X_v @ self.w)
            self.val_losses.append(binary_cross_entropy(y_val, y_hat_val))
            self.val_accs.append(compute_accuracy(y_val, y_hat_val))

    def _print_progress(self, epoch):
        """Print a progress update every 100 epochs."""
        if (epoch + 1) % 100 != 0:
            return
        train_loss = self.train_losses[-1]
        train_acc  = self.train_accs[-1]
        if self.val_losses:
            print(f"  Epoch {epoch+1:5d} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Loss: {self.val_losses[-1]:.4f} | "
                  f"Val Acc: {self.val_accs[-1]:.4f}")
        else:
            print(f"  Epoch {epoch+1:5d} | Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f}")

    def _get_top_features(self, feature_names, top_n):
        """Return the top_n features and weights sorted by absolute magnitude."""
        weights    = self.w[1:]  # skip bias weight at index 0
        sorted_idx = np.argsort(np.abs(weights))[::-1][:top_n]
        return [feature_names[i] for i in sorted_idx], weights[sorted_idx]

    def _plot_curve(self, ax, epochs, train_values, val_values, ylabel, title):
        """Plot a single train/validation curve on the given axis."""
        ax.plot(epochs, train_values, label="Train", color="blue")
        if val_values:
            ax.plot(epochs, val_values, label="Val", color="red", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_feature_importance(self, top_features, top_weights):
        colors = ["steelblue" if w > 0 else "tomato" for w in top_weights]
        plt.figure(figsize=(10, 6))
        plt.barh(top_features[::-1], top_weights[::-1], color=colors[::-1])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Weight Value")
        plt.title("Logistic Regression — Top Feature Weights\n"
                  "(Blue = predicts home win | Red = predicts home loss)")
        plt.tight_layout()
        plt.savefig("lr_feature_importance.png", dpi=150)
        plt.show()
        print("  Saved: lr_feature_importance.png")



# Removes IDs, team names, raw scores, and non-numeric list columns.
EXCLUDED_COLUMNS = [
    "game_id", "season", "season_type", "week_home",
    "team_home", "season_type_home", "opponent_team_home", "id",
    "week_away", "team_away", "opponent_team_away",
    "home_team", "away_team",
    "home_score", "away_score",
    "fg_made_list_home",  "fg_missed_list_home",  "fg_blocked_list_home",
    "fg_made_list_away",  "fg_missed_list_away",  "fg_blocked_list_away",
]


def load_and_prepare_data(filepath):

    df = pd.read_csv(filepath)

    feature_cols = _select_feature_columns(df)
    df_clean     = _clean_data(df, feature_cols)
    feature_cols = [c for c in feature_cols if c in df_clean.columns]

    X = df_clean[feature_cols].values.astype(float)
    y = df_clean["home_win"].values.astype(float)

    print(f"  Features selected : {len(feature_cols)}")
    print(f"  Dataset shape     : {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Class balance     : {y.mean():.2%} home wins")

    return X, y, feature_cols


def _select_feature_columns(df):
    """Return column names that are not in the exclusion list or the target."""
    excluded = set(EXCLUDED_COLUMNS + ["home_win"])
    return [c for c in df.columns if c not in excluded]


def _is_numeric_column(series):
    """Return True if converting to numeric does not introduce new NaN values."""
    original_nulls  = series.isna().sum()
    converted_nulls = pd.to_numeric(series, errors="coerce").isna().sum()
    return converted_nulls == original_nulls


def _drop_non_numeric_columns(df, feature_cols):
    """Remove columns that cannot be represented as numbers."""
    numeric_cols = [c for c in feature_cols if _is_numeric_column(df[c])]
    dropped      = set(feature_cols) - set(numeric_cols)
    if dropped:
        print(f"  Dropping non-numeric columns: {sorted(dropped)}")
    return numeric_cols


def _clean_data(df, feature_cols):
    """Remove non-numeric columns and fill NaNs representing zero activity."""
    df_subset    = df[feature_cols + ["home_win"]].dropna(subset=["home_win"])
    feature_cols = _drop_non_numeric_columns(df_subset, feature_cols)
    return df_subset[feature_cols + ["home_win"]].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)


def train_val_split(X, y, val_ratio=0.2):
    split  = int(len(y) * (1 - val_ratio))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"  Train: {len(y_train)} samples | Val: {len(y_val)} samples")
    return X_train, y_train, X_val, y_val


def run_learning_rate_experiment(X_train, y_train, X_val, y_val):
    """Train one model per learning rate and return results."""
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    results = {}

    for lr in learning_rates:
        print(f"\n  --- Learning rate: {lr} ---")
        model = LogisticRegression(learning_rate=lr, epochs=500, seed=42)
        model.fit(X_train, y_train, X_val, y_val)
        results[lr] = {
            "model"     : model,
            "train_acc" : compute_accuracy(y_train, model.predict_proba(X_train)),
            "val_acc"   : compute_accuracy(y_val,   model.predict_proba(X_val)),
        }

    return results


def print_learning_rate_summary(results):
    """Print a comparison table and return the learning rate with best val accuracy."""
    best_lr = max(results, key=lambda lr: results[lr]["val_acc"])

    print("\n  Learning rate comparison:")
    print(f"  {'LR':<8} {'Train Acc':>10} {'Val Acc':>10}")
    print(f"  {'-'*30}")
    for lr, res in results.items():
        marker = " <-- best" if lr == best_lr else ""
        print(f"  {lr:<8} {res['train_acc']:>10.4f} {res['val_acc']:>10.4f}{marker}")

    return best_lr


def train_final_model(best_lr, X_train, y_train, X_val, y_val):
    """Train the final model using the best learning rate for 1000 epochs."""
    print(f"\n  Training final model (lr={best_lr}, 1000 epochs)...")
    model = LogisticRegression(learning_rate=best_lr, epochs=1000, seed=42)
    model.fit(X_train, y_train, X_val, y_val)
    return model


def main():
    print("=" * 60)
    print("  NFL Winner Prediction — Logistic Regression From Scratch")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_data.csv")

    print("\n[1] Loading data...")
    X, y, feature_names = load_and_prepare_data(data_path)

    print("\n[2] Splitting data...")
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2)

    print("\n[3] Running learning rate experiments...")
    results = run_learning_rate_experiment(X_train, y_train, X_val, y_val)

    print("\n[4] Selecting best learning rate...")
    best_lr = print_learning_rate_summary(results)

    print("\n[5] Training final model...")
    final_model = train_final_model(best_lr, X_train, y_train, X_val, y_val)

    print("\n[6] Evaluating final model...")
    print_evaluation(y_train, final_model.predict_proba(X_train), label="Training Set")
    print_evaluation(y_val,   final_model.predict_proba(X_val),   label="Validation Set")

    print("\n[7] Generating plots...")
    final_model.plot_training_curves(title="Logistic Regression (NFL)")
    final_model.print_top_features(feature_names, top_n=15)

    print("\nDone!")


if __name__ == "__main__":
    main()
