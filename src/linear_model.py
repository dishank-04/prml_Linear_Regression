import numpy as np
from src.utils import create_design_matrix

class LinearRegressionPRML:

    def __init__(self, degree=3) -> None:
        self.weights: np.ndarray | None = None # Intialized as None, Will be numpy array of shape (D+1, ) after fitting.
        self.degree = degree # Stores the degree for Polynomial Regression

    def fit(self, X:np.ndarray, t:np.ndarray) -> None:

        """
        Fits the model to the training data using the Normal Equation.

        Args:
            X (np.ndarray): Training data feature matrix of shape (N, D).
            t (np.ndarray): Target values vector of shape (N,).

        Raises:
            ValueError: If X and t have mismatched row counts.
            np.linalg.LinAlgError: If the design matrix is singular (non-invertible).
        """

        if X.shape[0] != t.shape[0]:
            raise ValueError("Number of samples in X must match the number of targets in t")

        phi = create_design_matrix(X, degree=self.degree) # Creating Design Matrix

        #Componenets required for computing weights

        phi_T = phi.T

        try:

            ''' Trying to calcuate weigths, w = (phi.T @ phi)^-1 @ phi.T @ t'''

            covariance_matrix = phi_T @ phi
            inverse_cov = np.linalg.inv(covariance_matrix)

            self.weights = inverse_cov @ phi_T @ t

        except:

            # Industry standard fallback: use pseudo-inverse if matrix is singular
            print("Warning: Matrix is singular. Falling back to pseudo-inverse.")

            self.weights = (np.linalg.pinv(phi_T @ phi)) @ phi_T @ t
        

    def predict(self, X: np.ndarray) -> np.ndarray:

        """
        Predicts target values for the given input data.

        Args:
            X (np.ndarray): Input feature matrix of shape (N, D).

        Returns:
            np.ndarray: Predicted target values of shape (N,).
            
        Raises:
            RuntimeError: If predict is called before the model is fitted.
        """

        if self.weights is None:
            raise RuntimeError("Model is not fitted yet. Run 'fit' method first")
        
        phi = create_design_matrix(X, degree=self.degree)

        predictions = phi @ self.weights
        return predictions
    

    

