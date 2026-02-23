import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

def standardize_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """ 
    Standardizes the Dataset X for features to have mean = 0 and standarad Deviation = 1

    Args:
        X (np.ndarray): The input feature matrix of shape (N,D) where N is the number of samples and D is the number of features.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]
        - X_Standardized: The scaled feature matrix.
        - mean: The mean of each feature.
        - std: The Standard Deviation of each feature.
    """

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std = np.where(std == 0, 1e-8, std) # Need to Prevent division by zero error

    X_Standardized = (X - mean)/std

    return X_Standardized, mean, std


def create_design_matrix(X: np.ndarray) -> np.ndarray:

    """
    Creates the Design Matrix, phi, by prepending the columns of ones to X.

    Args:
        X(np.ndarray): This is input feature matrix of shape (N,D)
    
    Return:
        np.ndarray: The Design Matrix of shape (N,D+1)
    """

    num_samples = X.shape[0]
    ones_columns = np.ones((num_samples,1))

    phi = np.hstack((ones_columns, X))

    return phi


def calculate_erms(y_true: np.ndarray, y_predicted: np.ndarray) -> float:

    """
    Calculates the Root Mean Square Error (E_RMS) as defined in PRML.

    Args:
        y_true (np.ndarray): The ground truth target values of shape (N,).
        y_pred (np.ndarray): The model's predicted values of shape (N,).

    Returns:
        float: The calculated E_RMS value.
        
    Raises:
        ValueError: If the shapes of y_true and y_pred do not match.
    """

    if y_true.shape != y_predicted.shape:
        raise ValueError("Shape mismatch: y_true is {y_true.shape} and y_predicted is {y_predicted.shape}")

    N = len(y_true)

    sum_of_squares = np.sum((y_predicted - y_true)**2)

    erms = np.sqrt((sum_of_squares)/N)

    return float(erms)


def plot_erms(x_values: list[float] | np.ndarray, train_erms: list[float] | np.ndarray, test_erms: list[float] | np.ndarray, x_label: str = "Model Complexity (Degree 1)") -> None:

    """
    Plots the Training and Testing E_RMS to visualize model performance and overfitting.

    Args:
        x_values: list[float] | np.ndarray: The x-axis values (e.g., polynomial degrees, dataset sizes).
        train_erms: list[float] | np.ndarray: E_RMS values for the training set.
        test_erms: list[float] | np.ndarray: E_RMS values for the test set.
        x_label: (str, optional): Label for the X-axis. Defaults to "Model Complexity...".
    """

    plt.figure(figsize=(8,6))

    # Plotting Train and Test lines with different markers and colors

    plt.plot(x_values, train_erms, marker='o', color='blue', label='Training $E_{RMS}$', linewidth=2)
    plt.plot(x_values, test_erms, marker='o', color='red', label='Test $E_{RMS}$', linewidth=2)

    # Formatting the Figure and graphs for better look 

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("$E_{RMS}$", fontsize=12)
    plt.title("Root Mean Square Error ($E_{RMS}$) vs. " + x_label.split('(')[0].strip(), fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Ensuring layout is tight so labels aren't cut off
    plt.tight_layout()
    plt.show()

