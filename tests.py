from pathlib import Path
import numpy as np

from mp4 import *
import predicate_classifier as pc

def load_data(data_id: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path("data") / f"data_{data_id}.npz")
    return data["x"], data["y"]

def test_create_linear_data(num_samples: int = 100, slope: float = 2, intercept: float = 1, x_range: list[float] = [-10,10], noise: float = 0) -> None:
    x, y = create_linear_data(num_samples=num_samples, slope=slope, 
                              intercept=intercept, x_range=x_range, noise=noise)
    assert type(x) == np.ndarray and type(y) == np.ndarray
    assert x.shape == (num_samples,1)
    assert y.shape == (num_samples,1)
    assert np.all(x >= x_range[0]) and np.all(x <= x_range[1])
    residual = y - (slope*x + intercept)
    assert np.all(residual <= noise + 1e-8)
    assert np.all(residual >= -noise - 1e-8)
    if noise == 0:
        assert np.all(np.isclose(residual, 0))
    else:
        assert np.any(np.abs(residual) > 1e-8)

def test_get_simple_linear_features(num_samples: int = 10, num_features: int = 1) -> None:
    x = np.zeros((num_samples, num_features))
    features = get_simple_linear_features(x)
    assert type(features) == np.ndarray
    assert features.shape == (num_samples, num_features+1)
    assert np.all(features[:,-1] == 1)

def test_linear_prediction() -> None:
    x = np.array([[1],[2],[3]])
    A = np.array([[4],[-1]])
    y = linear_prediction(x, A, get_modified_features=get_simple_linear_features)
    assert type(y) == np.ndarray
    assert y.shape == (3,1)
    assert np.all(y == np.array([[3],[7],[11]]))

def test_mse_loss() -> None:
    y_true = np.array([[1],[2],[3]])
    y_pred = np.array([[2],[3],[4]])
    loss = mse_loss(y_true, y_pred)
    assert np.isscalar(loss) or np.ndim(loss) == 0
    assert loss == 1

    y_true = np.array([[1],[2],[3]])
    y_pred = np.array([[1],[2],[3]])
    loss = mse_loss(y_true, y_pred)
    assert loss == 0

    y_true = np.array([[1],[2],[3]])
    y_pred = np.array([[3],[2],[1]])
    loss = mse_loss(y_true, y_pred)
    assert np.isclose(loss, 8/3)

def test_analytical_linear_regression() -> None:
    x = np.array([[1],[2],[3]])
    y = np.array([[3],[7],[11]])
    X = get_simple_linear_features(x)
    A = analytical_linear_regression(X, y)
    assert type(A) == np.ndarray
    assert A.shape == (2,1)
    assert np.all(np.isclose(A, np.array([[4],[-1]])))

    x, y = create_linear_data(num_samples=100, slope=2, intercept=1, x_range=[-10,10], noise=0)
    X = get_simple_linear_features(x)
    A = analytical_linear_regression(X, y)
    assert np.all(np.isclose(A, np.array([[2],[1]])))

    np.random.seed(0)
    x, y = create_linear_data(num_samples=100, slope=-3, intercept=2, x_range=[0,5], noise=0.1)
    X = get_simple_linear_features(x)
    A = analytical_linear_regression(X, y)
    # 0.2 is arbitrary... just to make sure it's close
    assert np.all(np.abs(A - np.array([[-3],[2]])) < 0.2)

def test_linear_regression_gradient() -> None:
    # basically just test that the gradient at the true A is close to 0
    x, y = create_linear_data(num_samples=100, slope=2, intercept=1, x_range=[-10,10], noise=0)
    X = get_simple_linear_features(x)
    grad = get_linear_regression_gradient(np.array([[2],[1]]), X, y)
    assert type(grad) == np.ndarray
    assert grad.shape == (2,1) # same shape as A
    assert np.linalg.norm(grad) < 0.001

    np.random.seed(0)
    x, y = create_linear_data(num_samples=100, slope=-10, intercept=5, x_range=[0,5], noise=0.1)
    X = get_simple_linear_features(x)
    grad = get_linear_regression_gradient(np.array([[-10],[5]]), X, y)
    assert np.linalg.norm(grad) < 0.2 # 0.2 is arbitrary... just to make sure it's close enough

def test_gradient_descent() -> None:
    x, y = create_linear_data(num_samples=100, slope=2, intercept=1, x_range=[-10,10], noise=0)
    X = get_simple_linear_features(x)
    A_init = np.array([[0],[0]])
    get_gradient = lambda A: get_linear_regression_gradient(A, X, y)
    A = gradient_descent(get_gradient, A_init, learning_rate=0.01, num_iterations=1000)
    assert np.all(np.isclose(A, np.array([[2],[1]]), atol=0.1))

    np.random.seed(0)
    x, y = create_linear_data(num_samples=100, slope=-10, intercept=5, x_range=[0,5], noise=0.5)
    X = get_simple_linear_features(x)
    A_init = np.array([[0],[0]])
    get_gradient = lambda A: get_linear_regression_gradient(A, X, y)
    A = gradient_descent(get_gradient, A_init, learning_rate=0.01, num_iterations=1000)
    assert np.all(np.isclose(A, np.array([[-10],[5]]), atol=0.5))

def test_stochastic_gradient_descent() -> None:
    np.random.seed(0)
    x, y = create_linear_data(num_samples=100, slope=2, intercept=1, x_range=[-10,10], noise=0.1)
    X = get_simple_linear_features(x)
    A_init = np.array([[0],[0]])
    get_batch_gradient = lambda A, indices: get_linear_regression_gradient(A, X[indices], y[indices])
    A = stochastic_gradient_descent(get_batch_gradient, A_init, learning_rate=0.01, num_epochs=100, data_size=100, batch_size=20)
    assert type(A) == np.ndarray
    assert A.shape == A_init.shape
    assert compute_model_error(x, y, A, get_simple_linear_features) < compute_model_error(x, y, A_init, get_simple_linear_features)

def test_create_sine_data(num_samples: int = 100, x_range: list[float] = [-np.pi,np.pi], noise: float = 0) -> None:
    # similar test as linear
    x, y = create_sine_data(num_samples=num_samples, x_range=x_range, noise=noise)
    assert type(x) == np.ndarray and type(y) == np.ndarray
    assert x.shape == (num_samples,1)
    assert y.shape == (num_samples,1)
    assert np.all(x >= x_range[0]) and np.all(x <= x_range[1])
    residual = y - np.sin(x)
    assert np.all(residual <= noise + 1e-8)
    assert np.all(residual >= -noise - 1e-8)
    if noise == 0:
        assert np.all(np.isclose(residual, 0))
    else:
        assert np.any(np.abs(residual) > 1e-8)

def test_get_polynomial_features(num_data: int, num_features: int, degree: int) -> None:
    x = np.random.randn(num_data, num_features)
    X = get_polynomial_features(x, degree=degree)
    assert type(X) == np.ndarray
    assert X.shape == (num_data, num_features * (degree+1))
    for i in range(degree+1):
        assert np.all(np.isclose(X[:, i*num_features : (i+1)*num_features], x**(degree-i)))

def test_ik_loss() -> None:
    arm = Arm(np.ones(2))
    q = np.array([0.0,0.0])
    goal_ee = np.array([1.0,1.0])
    loss = ik_loss(arm, q, goal_ee)
    assert loss == np.sqrt(2)
    goal_ee = np.array([2.0,0.0])
    loss = ik_loss(arm, q, goal_ee)
    assert loss == 0.0

def test_sample_directions() -> None:
    num_samples = 100
    num_features = 3
    directions = sample_directions(num_samples, num_features)
    assert type(directions) == np.ndarray
    assert directions.shape == (num_samples, num_features)
    assert np.all(np.isclose(np.linalg.norm(directions, axis=1), 1.0))

def test_estimate_ik_gradient() -> None:
    arm = Arm(np.ones(4))
    q = np.array([0.0,0.0,0.0,0.0])
    goal_ee = np.array([0.0,5.0])
    loss = lambda q: ik_loss(arm, q, goal_ee)
    num_samples = 100
    np.random.seed(0)
    grad = estimate_ik_gradient(loss, q, num_samples)
    assert type(grad) == np.ndarray
    assert grad.shape == q.shape
    # test that unit vector
    assert np.isclose(np.linalg.norm(grad), 1.0)
    # check that moving in opposite direction of grad is improving the loss
    assert loss(q - grad) < loss(q)

def test_logistic_loss() -> None:
    y_true = np.array([[1],[0]])
    y_pred = np.array([[0.9],[0.1]])
    loss = logistic_loss(y_pred, y_true)
    assert np.isscalar(loss) or np.ndim(loss) == 0
    assert np.isclose(loss, -2*np.log(0.9))

    y_pred = np.array([[0.1],[0.9]])
    loss = logistic_loss(y_pred, y_true)
    assert np.isclose(loss, -2*np.log(0.1))

def test_logistic_error() -> None:
    y_true = np.array([[1],[0]])
    y_pred = np.array([[0.9],[0.1]])
    error = logistic_error(y_pred, y_true)
    assert np.isscalar(error) or np.ndim(error) == 0
    assert error == 0

    y_pred = np.array([[0.1],[0.9]])
    error = logistic_error(y_pred, y_true)
    assert error == 1

    y_pred = np.array([[0.4],[0.4]])
    error = logistic_error(y_pred, y_true)
    assert np.isclose(error, 0.5)

def test_logistic_prediction() -> None:
    # should be 0.5 where y=0, 1 where y is large and 0 where y is small
    # y=0 at x=-3/2
    A = np.array([[2],[3]])
    x = np.array([[-3/2], [-100], [100]])
    y = logistic_prediction(x, A, get_modified_features=get_simple_linear_features)
    assert type(y) == np.ndarray
    assert y.shape == (3,1)
    assert np.isclose(y[0,0], 0.5)
    assert np.isclose(y[1,0], 0)
    assert np.isclose(y[2,0], 1)

def test_logistic_regression_gradient() -> None:
    A = np.array([[2],[-3]])
    x = np.array([[3/2], [-100], [100]])
    X = get_simple_linear_features(x)
    y_true = np.array([[0],[0],[1]])
    grad = get_logistic_regression_gradient(A, X, y_true)
    assert type(grad) == np.ndarray
    assert grad.shape == A.shape
    # grad should be 1/2 * X[0]
    assert np.all(np.isclose(grad, 1/2 * X[0].reshape(-1,1)))

def test_get_logistic_regression_features() -> None:
    # needs to get low error on all three datasets
    for data_idx in range(3):
        print(f"Testing data {data_idx} GD")
        x,y = load_data(data_idx)
        get_features = get_logistic_regression_features
        X = get_features(x)
        assert type(X) == np.ndarray
        assert X.shape[0] == x.shape[0]
        assert X.ndim == 2
        assert np.all(np.isfinite(X))
        np.random.seed(1)
        A_init = np.random.rand(X.shape[1], 1)
        # gd
        get_gradient = lambda A: get_logistic_regression_gradient(A, X, y)
        A_gd = gradient_descent(get_gradient, A_init, learning_rate=0.01, num_iterations=10000)
        y_pred = logistic_prediction(x, A_gd, get_features)
        final_error = logistic_error(y_pred, y)
        assert final_error <= 0.1 # 90% accuracy
        
        # sgd
        print(f"Testing data {data_idx} SGD")
        get_batch_gradient = lambda A, indices: get_logistic_regression_gradient(A, X[indices], y[indices])
        A_sgd = stochastic_gradient_descent(get_batch_gradient, A_init, learning_rate=0.001, 
                                            num_epochs=10000, data_size=x.shape[0], batch_size=20)
        y_pred = logistic_prediction(x, A_sgd, get_features)
        final_error = logistic_error(y_pred, y)
        assert final_error <= 0.1 # 90% accuracy


def _load_predicate_split():
    data = pc.load_predicate_data()
    rng = np.random.default_rng(315)
    indices = rng.permutation(data["train_X"].shape[0])
    val_size = int(round(0.25 * len(indices)))
    val_idx = indices[:val_size]
    fit_idx = indices[val_size:]
    return (
        data["train_X"][fit_idx],
        data["train_y"][fit_idx],
        data["train_X"][val_idx],
        data["train_y"][val_idx],
        data["test_X"],
        data["test_y"],
    )


def evaluate_predicate_model(model: pc.PredicateModel, X: np.ndarray, y: np.ndarray) -> dict:
    Xp, _ = pc.preprocess_data(X, preprocessing=model.preprocess_state)
    y_pred = pc.predicate_prediction(Xp, model.weights) >= 0.5
    y_bool = y.astype(bool)
    true_positive = np.sum(y_pred & y_bool, axis=0)
    true_negative = np.sum(~y_pred & ~y_bool, axis=0)
    false_positive = np.sum(y_pred & ~y_bool, axis=0)
    false_negative = np.sum(~y_pred & y_bool, axis=0)
    recall = true_positive / np.maximum(true_positive + false_negative, 1)
    specificity = true_negative / np.maximum(true_negative + false_positive, 1)
    f1 = 2 * true_positive / np.maximum(
        2 * true_positive + false_positive + false_negative, 1
    )
    return {
        "predicate_accuracy": float(np.mean(y_pred == y_bool)),
        "balanced_accuracy": float(np.mean((recall + specificity) / 2)),
        "mean_f1": float(np.mean(f1)),
        "exact_match": float(np.mean(np.all(y_pred == y_bool, axis=1))),
    }


def test_predicate_preprocess_data() -> None:
    X = np.array([[1.0, 10.0, 4.0], [3.0, 14.0, 4.0], [5.0, 18.0, 4.0]])
    X_copy = X.copy()
    Xp, preprocessing = pc.preprocess_data(X, normalize=True)
    assert Xp.shape == (3, 4)
    assert preprocessing.mean.shape == (1, 3)
    assert preprocessing.std.shape == (1, 3)
    assert np.allclose(Xp[:, :3].mean(axis=0), 0.0)
    assert np.allclose(Xp[:, :2].std(axis=0), 1.0)
    assert np.all(Xp[:, 2] == 0.0)
    assert preprocessing.std[0, 2] == 1.0
    assert np.all(Xp[:, -1] == 1.0)
    assert np.array_equal(X, X_copy)

    X_val = np.array([[101.0, 222.0, 4.0]])
    Xv, _ = pc.preprocess_data(X_val, preprocessing=preprocessing)
    assert np.allclose(Xv[:, :3], [[60.01249869818786, 63.68673331236263, 0.0]])


def _predicate_loss_inputs():
    X = np.array([[1.0, -2.0, 1.0], [0.5, 1.0, 1.0], [-1.0, 0.0, 1.0]])
    y = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    weights = np.array([[0.4, -0.7], [-0.3, 0.2], [0.1, -0.2]])
    prediction = pc.predicate_prediction(X, weights)
    assert prediction.shape == y.shape
    assert np.all(np.isfinite(prediction))
    return X, y, weights, prediction


def test_predicate_loss(stage="all") -> None:
    # Check one notebook stage, or all stages for grading
    assert stage in ("baseline", "regularized", "weighted", "all")
    X, y, weights, _ = _predicate_loss_inputs()
    if stage in ("baseline", "all"):
        actual = pc.compute_loss(weights, X, y)
        assert np.isscalar(actual) or np.ndim(actual) == 0
        assert np.isclose(actual, 1.144435114525260, rtol=1e-7, atol=1e-8)
        assert np.isfinite(pc.compute_loss(np.full_like(weights, 1000.0), X, y))
    if stage in ("regularized", "all"):
        actual = pc.compute_loss(weights, X, y, l1_lambda=0.2)
        assert np.isclose(actual, 1.464435114525260, rtol=1e-7, atol=1e-8)
        bias_only = np.zeros_like(weights)
        bias_only[-1] = [2.0, -3.0]
        assert np.isclose(
            pc.compute_loss(bias_only, X, y, l1_lambda=10.0),
            pc.compute_loss(bias_only, X, y, l1_lambda=0.0),
        )
    if stage in ("weighted", "all"):
        actual = pc.compute_loss(weights, X, y, positive_weights=np.array([3.0, 1.5]))
        assert np.isclose(actual, 2.131805350847712, rtol=1e-7, atol=1e-8)


def test_predicate_gradient(stage="all") -> None:
    # Expected gradients each have shape (3, 2), including the final bias row
    assert stage in ("baseline", "regularized", "weighted", "all")
    X, y, weights, _ = _predicate_loss_inputs()
    cases = [
        ("baseline", {}, np.array([
            [0.191567540802259, 0.099465632098974],
            [0.333159929603255, -0.338315870944071],
            [-0.108060803738847, -0.249997743586011],
        ])),
        ("regularized", {"l1_lambda": 0.2}, np.array([
            [0.391567540802259, -0.100534367901026],
            [0.133159929603255, -0.138315870944071],
            [-0.108060803738847, -0.249997743586011],
        ])),
        ("weighted", {"positive_weights": np.array([3.0, 1.5])}, np.array([
            [0.408035955740110, 0.113504278655554],
            [0.666146455476431, -0.436085467430293],
            [-0.657515744549875, -0.410690784871924],
        ])),
    ]
    for name, options, expected in cases:
        if stage in (name, "all"):
            actual = pc.compute_gradient(weights, X, y, **options)
            assert type(actual) == np.ndarray
            assert actual.shape == weights.shape
            assert np.allclose(actual, expected, rtol=1e-7, atol=1e-8)


def test_predicate_positive_weights() -> None:
    y = np.array([
        [1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0],
    ])
    weights = pc.get_positive_weights(y, positive_weight_power=0.5)
    expected = np.array([1.732050807568877, 1.732050807568877, 0.577350269189626, 0.5, 2.0])
    assert np.allclose(weights, expected)
    assert np.allclose(pc.get_positive_weights(y, positive_weight_power=0.0), np.ones(5))
    rare = np.zeros((1000, 1))
    rare[0] = 1.0
    assert pc.get_positive_weights(rare, 1.0)[0] == 20.0


def test_predicate_training_performance() -> None:
    train_X, train_y, val_X, val_y, test_X, test_y = _load_predicate_split()
    # Train the provided final ablation configuration to check the completed components
    config = pc.ABLATION_CONFIGS[-1]
    model, history = pc.train_model(train_X, train_y, val_X, val_y, **config, seed=7)
    assert isinstance(history, list) and len(history) > 0
    assert "selected_update" in history[-1]
    assert "selected_val_exact_match" in history[-1]
    val_metrics = evaluate_predicate_model(model, val_X, val_y)
    test_metrics = evaluate_predicate_model(model, test_X, test_y)
    print("predicate validation metrics:", val_metrics)
    print("predicate test metrics:", test_metrics)
    assert np.isclose(val_metrics["exact_match"], history[-1]["selected_val_exact_match"], atol=1e-8)
    assert val_metrics["balanced_accuracy"] >= 0.98
    assert val_metrics["mean_f1"] >= 0.98
    assert val_metrics["exact_match"] >= 0.98
    assert test_metrics["exact_match"] >= 0.97


if __name__ == "__main__":
    # you should comment out what you're not testing
    test_create_linear_data(num_samples=10, slope=2, intercept=-1, x_range=[-10,10], noise=0)
    test_create_linear_data(num_samples=10, slope=-3, intercept=2, x_range=[0,5], noise=0.2)
    test_get_simple_linear_features(num_samples=10, num_features=1)
    test_get_simple_linear_features(num_samples=1, num_features=4)
    test_linear_prediction()
    test_mse_loss()
    test_analytical_linear_regression()
    test_linear_regression_gradient()
    test_gradient_descent()
    test_stochastic_gradient_descent()
    test_create_sine_data()
    test_get_polynomial_features(num_data=10, num_features=1, degree=3)
    test_get_polynomial_features(num_data=10, num_features=2, degree=3)
    test_get_polynomial_features(num_data=100, num_features=1, degree=1)
    test_ik_loss()
    test_sample_directions()
    test_estimate_ik_gradient()
    test_logistic_error()
    test_logistic_prediction()
    test_logistic_loss()
    test_logistic_regression_gradient()
    test_get_logistic_regression_features()
    test_predicate_loss()
    test_predicate_gradient()
    test_predicate_preprocess_data()
    test_predicate_positive_weights()
    test_predicate_training_performance()
