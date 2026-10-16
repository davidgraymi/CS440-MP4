from dataclasses import dataclass

import numpy as np

from mp4 import logistic_prediction


L1_STRENGTH = 0.001
POSITIVE_WEIGHT_POWER = 1.0

# Compare every subset of normalization, L1 regularization, and positive-label weighting
# Keep the optimizer settings fixed across all eight configurations
ABLATION_CONFIGS = [
    {"normalize": False, "l1_lambda": 0.0, "positive_weight_power": 0.0},
    {"normalize": True, "l1_lambda": 0.0, "positive_weight_power": 0.0},
    {"normalize": False, "l1_lambda": L1_STRENGTH, "positive_weight_power": 0.0},
    {"normalize": False, "l1_lambda": 0.0, "positive_weight_power": POSITIVE_WEIGHT_POWER},
    {"normalize": True, "l1_lambda": L1_STRENGTH, "positive_weight_power": 0.0},
    {"normalize": True, "l1_lambda": 0.0, "positive_weight_power": POSITIVE_WEIGHT_POWER},
    {"normalize": False, "l1_lambda": L1_STRENGTH, "positive_weight_power": POSITIVE_WEIGHT_POWER},
    {"normalize": True, "l1_lambda": L1_STRENGTH, "positive_weight_power": POSITIVE_WEIGHT_POWER},
]


# Store feature statistics computed from the training data
# mean and std each have shape (1, num_features), or are None without normalization
# Reuse these statistics when transforming validation or test data
@dataclass
class PreprocessState:
    mean: np.ndarray | None
    std: np.ndarray | None


# Store the classifier weights and preprocessing statistics
# weights has shape (num_features + 1, num_predicates), including a final bias row
# preprocess_state is the PreprocessState used to transform the model's inputs
@dataclass
class PredicateModel:
    weights: np.ndarray
    preprocess_state: PreprocessState


# -------------- provided helpers --------------

# Provided helper to load the predicate dataset from an npz file
# Return a dictionary with train_X, train_y, test_X, test_y, and predicate_names
# train_X and test_X have shape (num_train, num_features) and (num_test, num_features)
# train_y and test_y have shape (num_train, num_predicates) and (num_test, num_predicates)
# predicate_names has shape (num_predicates,)
def load_predicate_data(path: str = "data/task_planning_perception_features.npz") -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


# Provided helper to predict all predicate probabilities using logistic_prediction
# X has shape (num_samples, num_features + 1), including a final column of ones
# weights has shape (num_features + 1, num_predicates)
# Return probabilities with shape (num_samples, num_predicates)
def predicate_prediction(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return logistic_prediction(X, weights, lambda x: x)


# Provided helper to compute exact-match accuracy as a scalar
# weights has shape (num_features + 1, num_predicates)
# X has shape (num_samples, num_features + 1), including a final column of ones
# y has shape (num_samples, num_predicates)
# Threshold probabilities at 0.5 and count a sample only when all its labels are correct
def _exact_match(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    y_pred = predicate_prediction(X, weights) >= 0.5
    return float(np.mean(np.all(y_pred == y.astype(bool), axis=1)))


# Provided helper to record losses and exact-match accuracies at one update
# weights has shape (num_features + 1, num_predicates)
# X_train and X_val have shape (num_train, num_features + 1) and (num_val, num_features + 1)
# train_y and val_y have shape (num_train, num_predicates) and (num_val, num_predicates)
# positive_weights has shape (num_predicates,)
# Return a dictionary with update, train_loss, val_loss, train_exact_match, and val_exact_match
def _history_row(update: int, 
                 weights: np.ndarray,
                 X_train: np.ndarray,
                 train_y: np.ndarray,
                 X_val: np.ndarray,
                 val_y: np.ndarray,
                 l1_lambda: float,
                 positive_weights: np.ndarray) -> dict:
    return {
        "update": update,
        "train_loss": compute_loss(weights, X_train, train_y, l1_lambda, positive_weights),
        "val_loss": compute_loss(weights, X_val, val_y, l1_lambda, positive_weights),
        "train_exact_match": _exact_match(weights, X_train, train_y),
        "val_exact_match": _exact_match(weights, X_val, val_y),
    }


# Provided mini-batch SGD loop that returns the best validation checkpoint
# train_X and val_X have shape (num_train, num_features) and (num_val, num_features)
# train_y and val_y have shape (num_train, num_predicates) and (num_val, num_predicates)
# learning_rate is the step size and batch_size is the number of samples per batch
# max_updates is the number of gradient updates and seed controls initialization and shuffling
# Evaluate at update 0, every eval_every updates, and the final update
# Return a PredicateModel and a list of history dictionaries, keeping the first checkpoint on ties
# The model's weights have shape (num_features + 1, num_predicates)
# The final history row also records selected_update and selected_val_exact_match
def train_model(train_X: np.ndarray,
                train_y: np.ndarray,
                val_X: np.ndarray,
                val_y: np.ndarray,
                normalize: bool = False,
                l1_lambda: float = 0.0,
                positive_weight_power: float = 0.0,
                learning_rate: float = 0.09,
                batch_size: int = 16,
                max_updates: int = 9000,
                eval_every: int = 50,
                seed: int = 0) -> tuple[PredicateModel, list[dict]]:
    if batch_size <= 0 or eval_every <= 0 or max_updates < 0:
        raise ValueError("batch_size and eval_every must be positive; max_updates cannot be negative")

    rng = np.random.default_rng(seed)
    X_train, preprocessing = preprocess_data(train_X, normalize=normalize)
    X_val, _ = preprocess_data(val_X, preprocessing=preprocessing)
    positive_weights = get_positive_weights(train_y, positive_weight_power)
    weights = rng.normal(0.0, 0.01, size=(X_train.shape[1], train_y.shape[1])).astype(np.float32)

    history = [_history_row(
        0, weights, X_train, train_y, X_val, val_y,
        l1_lambda, positive_weights,
    )]
    best_weights = weights.copy()
    best_val_exact = history[0]["val_exact_match"]
    best_update = 0

    permutation = np.empty(0, dtype=int)
    batch_start = 0
    for update in range(1, max_updates + 1):
        if batch_start >= len(permutation):
            permutation = rng.permutation(X_train.shape[0])
            batch_start = 0
        indices = permutation[batch_start:batch_start + batch_size]
        batch_start += batch_size
        gradient = compute_gradient(
            weights, X_train[indices], train_y[indices],
            l1_lambda=l1_lambda,
            positive_weights=positive_weights,
        )
        weights = weights - learning_rate * gradient

        if update % eval_every != 0 and update != max_updates:
            continue
        row = _history_row(
            update, weights, X_train, train_y, X_val, val_y,
            l1_lambda, positive_weights,
        )
        history.append(row)
        if row["val_exact_match"] > best_val_exact + 1e-12:
            best_weights = weights.copy()
            best_val_exact = row["val_exact_match"]
            best_update = update

    history[-1]["selected_update"] = best_update
    history[-1]["selected_val_exact_match"] = best_val_exact
    return PredicateModel(best_weights, preprocessing), history


# Provided helper to compare all ABLATION_CONFIGS using validation exact-match accuracy
# train_X and val_X have shape (num_train, num_features) and (num_val, num_features)
# train_y and val_y have shape (num_train, num_predicates) and (num_val, num_predicates)
# Use the same seed for each training run and keep the first configuration on ties
# Return best_config and a list of result dictionaries in ABLATION_CONFIGS order
# Each result contains the configuration values, val_exact_match, and selected_update
def validation_experiment(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    seed: int = 0,
) -> tuple[dict, list[dict]]:
    results = []
    best_config = None
    best_score = -np.inf
    for config in ABLATION_CONFIGS:
        model, history = train_model(train_X, train_y, val_X, val_y, **config, seed=seed)
        X_val, _ = preprocess_data(val_X, preprocessing=model.preprocess_state)
        score = _exact_match(model.weights, X_val, val_y)
        row = dict(config)
        row["val_exact_match"] = score
        row["selected_update"] = history[-1]["selected_update"]
        results.append(row)
        if score > best_score + 1e-12:
            best_config = dict(config)
            best_score = score
    return best_config, results


# Provided helper to save training and validation exact-match curves
# history is the list of dictionaries returned by train_model
# path is the output image filename
def plot_training_curves(history: list[dict], path: str) -> None:
    import matplotlib.pyplot as plt

    updates = [row["update"] for row in history]
    train_exact = [row["train_exact_match"] for row in history]
    val_exact = [row["val_exact_match"] for row in history]
    plt.figure(figsize=(6, 4))
    plt.plot(updates, train_exact, label="train")
    plt.plot(updates, val_exact, label="validation")
    plt.xlabel("SGD update")
    plt.ylabel("exact match")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -------------- student-modified functions (Parts 7.1-7.4) --------------

# Transform the input features and append a bias column of ones
# X has shape (num_samples, num_features), without a bias column
# normalize controls whether to compute training statistics when preprocessing is None
# preprocessing is a fitted PreprocessState, or None when fitting training data
# Return transformed features with shape (num_samples, num_features + 1) and a PreprocessState
# Reuse supplied statistics for validation and test data without modifying X in place
def preprocess_data(X: np.ndarray, 
                    normalize: bool = False,
                    preprocessing: PreprocessState | None = None) -> tuple[np.ndarray, PreprocessState]:
    if preprocessing is None:
        if normalize:
            # TODO(7.2): set mean and std to column statistics with shape (1, num_features)
            mean = np.mean(X, axis=0, keepdims=True)
            std = np.std(X, axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
        else:
            mean = None
            std = None
        preprocessing = PreprocessState(mean=mean, std=std)

    transformed = np.asarray(X, dtype=float)
    if preprocessing.mean is not None:
        # TODO(7.2): set transformed to be the transformed features with shape (num_samples, num_features)
        transformed = (transformed - preprocessing.mean) / preprocessing.std
    bias = np.ones((transformed.shape[0], 1), dtype=transformed.dtype)
    return np.concatenate([transformed, bias], axis=1), preprocessing


# Compute binary cross-entropy loss with optional weighting and regularization
# weights has shape (num_features + 1, num_predicates), including a final bias row
# X has shape (num_samples, num_features + 1), including a final column of ones
# y has shape (num_samples, num_predicates) and contains binary labels
# positive_weights has shape (num_predicates,), or is None for equal weights
# l1_lambda is the nonnegative L1 regularization strength
# Return a scalar loss, summing over predicates and averaging over samples
# Apply regularization only to weights[:-1], which has shape (num_features, num_predicates)
def compute_loss(weights: np.ndarray,
                 X: np.ndarray,
                 y: np.ndarray,
                 l1_lambda: float = 0.0,
                 positive_weights: np.ndarray | None = None) -> float:
    if positive_weights is None:
        positive_weights = np.ones(y.shape[1])
    pred = np.clip(np.asarray(predicate_prediction(X, weights), dtype=float), 1e-8, 1.0 - 1e-8)
    # TODO(7.1): compute elementwise BCE with shape (num_samples, num_predicates)
    accum = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            accum += y[i,j] * np.log(pred[i,j]) + (1 - y[i,j]) * np.log(1 - pred[i,j])
    bce = - accum / X.shape[0]
    # TODO(7.4): account for positive_weights in your bce computation
    # TODO(7.3): compute the L1 penalty on non bias features (all rows except the last)
    penalty = 0.0
    return float(bce + penalty)


# Compute the gradient of compute_loss with respect to weights
# weights has shape (num_features + 1, num_predicates), including a final bias row
# X has shape (num_samples, num_features + 1), including a final column of ones
# y has shape (num_samples, num_predicates) and contains binary labels
# positive_weights has shape (num_predicates,), or is None for equal weights
# l1_lambda is the nonnegative L1 regularization strength
# Return a gradient with shape (num_features + 1, num_predicates)
# Add regularization terms only to gradient[:-1], leaving the bias gradient unchanged
def compute_gradient(weights: np.ndarray,
                     X: np.ndarray,
                     y: np.ndarray,
                     l1_lambda: float = 0.0,
                     positive_weights: np.ndarray | None = None) -> np.ndarray:
    if positive_weights is None:
        positive_weights = np.ones(y.shape[1])
    pred = predicate_prediction(X, weights)
    # TODO(7.1): replace the raise error below with gradient of shape (num_features + 1, num_predicates)
    gradient = (X.T @ (pred - y)) / X.shape[0]
    # TODO(7.4): account for positive_weights in your bce computation
    # TODO(7.3): add the L1 gradient term to gradient[:-1] only
    return gradient


# Compute one positive-label weight for each predicate
# y has shape (num_samples, num_predicates) and contains binary labels
# positive_weight_power is the exponent applied to the negative-to-positive count ratio
# Return weights with shape (num_predicates,), clipped to [0.5, 20.0]
# When positive_weight_power is 0, return ones
def get_positive_weights(y: np.ndarray, positive_weight_power: float) -> np.ndarray:
    if positive_weight_power == 0.0:
        return np.ones(y.shape[1], dtype=float)
    # TODO(7.4): compute the negative-to-positive ratio as per the instructions
    raise NotImplementedError("7.4: positive-label weights")
