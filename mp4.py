from collections.abc import Callable

import numpy as np
from planar_arm import Arm


# -------------- setup linear regression --------------

# Create noisy samples from a line
# num_samples is the number of data points
# slope and intercept define the true line
# x_range is the range to sample x from
# noise is the max absolute noise added to y
# Return x and y, each with shape (num_samples, 1)
def create_linear_data(num_samples: int, slope: float, intercept: float, x_range: list[float] = [-1.0, 1.0], noise: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    size = (num_samples, 1)
    x = np.random.uniform(low=x_range[0], high=x_range[1], size=size)
    e = np.random.uniform(low=-noise, high=noise, size=size)
    y = slope * x + intercept + e
    return x, y


# Add a bias feature to the input data
# x has shape (num_samples, num_features)
# Return features with shape (num_samples, num_features + 1)
def get_simple_linear_features(x: np.ndarray) -> np.ndarray:
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)


# Predict outputs with a linear model
# x has shape (num_samples, num_features)
# A has shape (num_modified_features, 1)
# get_modified_features maps x to shape (num_samples, num_modified_features)
# Return predictions with shape (num_samples, 1)
def linear_prediction(x: np.ndarray, A: np.ndarray, get_modified_features: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return np.dot(get_modified_features(x), A)


# Compute mean squared error
# y_pred and y_true each have shape (num_samples, 1)
# Return the mean squared error as a scalar
def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)


# Compute the regression model loss on a dataset
# x has shape (num_samples, num_features)
# y has shape (num_samples, 1)
# A has shape (num_modified_features, 1)
# get_modified_features maps x to shape (num_samples, num_modified_features)
# Return the loss for the model on x and y
def compute_model_error(x: np.ndarray, y: np.ndarray, A: np.ndarray, get_modified_features: Callable[[np.ndarray], np.ndarray]) -> float:
    return mse_loss(linear_prediction(x, A, get_modified_features), y)


# Solve linear regression analytically
# X has shape (num_samples, num_modified_features)
# y has shape (num_samples, 1)
# Return A with shape (num_modified_features, 1)
def analytical_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.inv(X.T @ X) @ X.T @ y


# -------------- gradient descent for linear regression --------------

# Compute the gradient of linear regression MSE
# A has shape (num_modified_features, 1)
# X has shape (num_samples, num_modified_features)
# y has shape (num_samples, 1)
# Return the gradient with the same shape as A
def get_linear_regression_gradient(A: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (2 / y.shape[0]) * X.T @ (X @ A - y)


# Optimize parameters with gradient descent
# get_gradient takes A and returns the gradient with the same shape as A
# A_init has shape (num_modified_features, 1)
# learning_rate is the gradient descent step size
# num_iterations is the number of gradient descent updates
# Return optimized parameters with the same shape as A_init
def gradient_descent(get_gradient: Callable[[np.ndarray], np.ndarray], A_init: np.ndarray, learning_rate: float, num_iterations: int) -> np.ndarray:
    A = A_init
    for _ in range(num_iterations):
        A = A - learning_rate * get_gradient(A)
    return A


# -------------- stochastic gradient descent for linear regression --------------

# Optimize parameters with stochastic gradient descent
# get_batch_gradient takes parameters A and a batch of data indices
# A_init has shape (num_modified_features, 1)
# learning_rate is the stochastic gradient descent step size
# num_epochs is the number of passes through the dataset
# data_size is the number of samples in the dataset
# batch_size is the number of samples in each batch
# Return optimized parameters with the same shape as A_init
def stochastic_gradient_descent(get_batch_gradient: Callable[[np.ndarray, np.ndarray], np.ndarray], A_init: np.ndarray, learning_rate: float, num_epochs: int, data_size: int, batch_size: int) -> np.ndarray:
    A = A_init
    max_batch = data_size - 1
    for _ in range(num_epochs):
        i_low = -batch_size
        i_high = 0
        while i_high < max_batch:
            i_low += batch_size
            i_high = min(i_high + batch_size, max_batch)
            A = A - learning_rate * get_batch_gradient(A, np.arange(i_low, i_high))
    return A


# -------------- polynomial regression for sine function --------------

# Create noisy samples from a sine curve
# num_samples is the number of data points
# x_range is the range to sample x from
# noise is the max absolute noise added to y
# Return x and y, each with shape (num_samples, 1)
def create_sine_data(num_samples: int, x_range: list[float] = [0.0, 2*np.pi], noise: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    size = (num_samples, 1)
    x = np.random.uniform(low=x_range[0], high=x_range[1], size=size)
    e = np.random.uniform(low=-noise, high=noise, size=size)
    return x, np.sin(x) + e


# Transform inputs into polynomial features
# x has shape (num_samples, num_features)
# degree is the highest polynomial degree to include
# Return features with shape (num_samples, num_features * (degree + 1))
def get_polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    return np.concatenate([x ** p for p in range(degree, -1, -1)], axis=1)


# -------------- inverse kinematics via gradient descent --------------

# Compute end-effector distance to a goal
# arm is an Arm object
# config has shape (num_joints,)
# goal has shape (2,)
# Return the distance from the end effector to the goal
def ik_loss(arm: Arm, config: np.ndarray, goal: np.ndarray) -> float:
    return np.linalg.norm(arm.forward_kinematics(config)[-1] - goal)


# we provide a more complex loss function that includes obstacles
# this loss is high when the arm is close to an obstacle
# obstacles has shape (num_obstacles, 3); each row stores (x, y, radius)
# where each obstacle is a circle with (x,y,radius)
def ik_loss_with_obstacles(arm: Arm, config: np.ndarray, goal: np.ndarray, obstacles: list[np.ndarray] | np.ndarray) -> float:
    # first compute the ik loss without obstacles
    ee_loss = ik_loss(arm, config, goal)
    # now compute the obstacle loss as a sum of harmonic losses (1/distance)
    workspace_config = arm.forward_kinematics(config)
    total_obstacle_loss = 0
    for obstacle in obstacles:
        # find the closest joint to the obstacle 
        # (technically we should do line-segment to circle distance)
        obstacle_dist = np.min(np.linalg.norm(workspace_config - obstacle[:2], axis=1))
        # if the joint is inside the obstacle, return infinity
        if obstacle_dist < obstacle[2]:
            return np.inf
        # otherwise, compute the harmonic loss
        total_obstacle_loss += 1 / (obstacle_dist - obstacle[2])
        # we could instead use a quadratic penalty
        # total_obstacle_loss += -(obstacle_dist - obstacle[2])**2
    return ee_loss + total_obstacle_loss


# Sample random directions for estimating a gradient
# num_samples is the number of directions to sample
# num_features is the number of dimensions in each direction
# Return directions with shape (num_samples, num_features)
# Each direction should have unit norm
def sample_directions(num_samples: int, num_features: int) -> np.ndarray:
    directions = np.random.normal(size=(num_samples, num_features))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / norms


# Estimate a local IK gradient direction by sampling
# loss maps a configuration to a scalar loss
# config has shape (num_features,)
# num_samples is the number of directions to sample
# Return a unit vector with shape (num_features,)
def estimate_ik_gradient(loss: Callable[[np.ndarray], float], config: np.ndarray, num_samples: int) -> np.ndarray:
    directions = sample_directions(num_samples, config.shape[0])
    eps = 0.01
    pos_config = config + directions * eps
    neg_config = config - directions * eps
    pl = np.array([loss(pc) for pc in pos_config])
    nl = np.array([loss(nc) for nc in neg_config])
    grad_dir = (pl - nl) @ directions
    return grad_dir / np.linalg.norm(grad_dir)

# -------------- logistic regression for provided data --------------

# Compute binary classification error
# y_pred has shape (num_samples, 1)
# y_true has shape (num_samples, 1)
# Return the average classification error as a scalar
def logistic_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_class = (y_pred > 0.5).astype(int)
    y_diff = (y_class != y_true).astype(int)
    return np.mean(y_diff)


# Predict class probabilities with logistic regression
# x has shape (num_samples, num_features)
# A has shape (num_modified_features, 1)
# get_modified_features maps x to shape (num_samples, num_modified_features)
# Return predicted probabilities with shape (num_samples, 1)
def logistic_prediction(x: np.ndarray, A: np.ndarray, get_modified_features: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    z = get_modified_features(x) @ A
    return 1 / (1 + np.e ** -z)


# Compute logistic loss for binary classification
# y_pred has shape (num_samples, 1)
# y_true has shape (num_samples, 1)
# Return the logistic loss as a scalar
def logistic_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.min(np.sum(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)))


# Compute the gradient of logistic regression loss
# A has shape (num_modified_features, 1)
# X has shape (num_samples, num_modified_features)
# y has shape (num_samples, 1)
# Return the gradient with the same shape as A
def get_logistic_regression_gradient(A: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X.T @ (logistic_prediction(X, A, lambda x: x) - y)


# Create features for the logistic regression datasets
# x has shape (num_samples, num_features)
# Return features with shape (num_samples, num_modified_features)
def get_logistic_regression_features(x: np.ndarray) -> np.ndarray:
    feats_poly = np.concatenate([x ** p for p in range(4, -1, -1)], axis=1)
    outer = np.einsum('mi,mj->mij', x, x)
    row, col = np.triu_indices(x.shape[1], k=1)
    interactions = outer[:, row, col]
    design = np.hstack([
        interactions,
        feats_poly,
    ])
    return design
