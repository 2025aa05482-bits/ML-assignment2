# """
# Logistic Regression Model Implementation
# """
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
# import numpy as np


# class LogisticRegressionModel:
#     """Logistic Regression classifier wrapper with hyperparameter tuning support."""
    
#     def __init__(self, C=1.0, max_iter=100, solver='lbfgs', penalty='l2', random_state=42):
#         """
#         Initialize Logistic Regression model.
        
#         Parameters:
#         -----------
#         C : float, default=1.0
#             Inverse of regularization strength
#         max_iter : int, default=100
#             Maximum number of iterations
#         solver : str, default='lbfgs'
#             Algorithm to use for optimization
#         penalty : str, default='l2'
#             Regularization penalty type
#         random_state : int, default=42
#             Random seed for reproducibility
#         """
#         self.C = C
#         self.max_iter = max_iter
#         self.solver = solver
#         self.penalty = penalty
#         self.random_state = random_state
#         self.model = None
        
#     def build_model(self):
#         """Build and return the Logistic Regression model."""
#         self.model = LogisticRegression(
#             C=self.C,
#             max_iter=self.max_iter,
#             solver=self.solver,
#             penalty=self.penalty,
#             random_state=self.random_state
#         )
#         return self.model
    
#     def train(self, X_train, y_train):
#         """
#         Train the model on the given data.
        
#         Parameters:
#         -----------
#         X_train : array-like
#             Training features
#         y_train : array-like
#             Training labels
#         """
#         if self.model is None:
#             self.build_model()
#         self.model.fit(X_train, y_train)
#         return self
    
#     def predict(self, X):
#         """Make predictions on the given data."""
#         return self.model.predict(X)
    
#     def predict_proba(self, X):
#         """Get probability predictions."""
#         return self.model.predict_proba(X)
    
#     def evaluate(self, X_test, y_test):
#         """
#         Evaluate model performance.
        
#         Returns:
#         --------
#         dict : Dictionary containing accuracy, precision, recall, f1, and confusion matrix
#         """
#         y_pred = self.predict(X_test)
#         y_proba = self.predict_proba(X_test)
        
#         metrics = {
#             'accuracy': accuracy_score(y_test, y_pred),
#             'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
#             'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
#             'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
#             'confusion_matrix': confusion_matrix(y_test, y_pred)
#         }
        
#         # Add ROC-AUC for binary classification
#         if len(np.unique(y_test)) == 2:
#             metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
        
#         return metrics
    
#     @staticmethod
#     def get_param_grid():
#         """Return hyperparameter grid for tuning."""
#         return {
#             'C': [0.01, 0.1, 1.0, 10.0],
#             'max_iter': [100, 200, 500],
#             'solver': ['lbfgs', 'liblinear', 'saga'],
#             'penalty': ['l2']
#         }
    
#     @staticmethod
#     def get_param_info():
#         """Return information about hyperparameters for UI."""
#         return {
#             'C': {
#                 'type': 'float',
#                 'min': 0.001,
#                 'max': 100.0,
#                 'default': 1.0,
#                 'description': 'Inverse of regularization strength (smaller = stronger)'
#             },
#             'max_iter': {
#                 'type': 'int',
#                 'min': 50,
#                 'max': 1000,
#                 'default': 100,
#                 'description': 'Maximum number of iterations for solver'
#             },
#             'solver': {
#                 'type': 'select',
#                 'options': ['lbfgs', 'liblinear', 'saga', 'newton-cg'],
#                 'default': 'lbfgs',
#                 'description': 'Optimization algorithm'
#             }
#         }


"""
Logistic Regression Model for Healthcare Dataset
(Assignment-2 Ready | Multi-class | All Metrics Included)
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef
)


class LogisticRegressionModel:
    """
    Logistic Regression Classifier
    Supports Multi-class classification with full evaluation metrics
    """

    def __init__(self, C=0.5, max_iter=2000, solver="lbfgs", random_state=42):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.model = None

    # -------------------------------
    # Build Model
    # -------------------------------
    def build_model(self):
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
            class_weight='balanced',
            tol=1e-4
        )
        return self.model

    # -------------------------------
    # Train Model
    # -------------------------------
    def train(self, X_train, y_train):
        self.build_model()
        self.model.fit(X_train, y_train)
        return self

    # -------------------------------
    # Predict
    # -------------------------------
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    # -------------------------------
    # Evaluation (ALL REQUIRED METRICS)
    # -------------------------------
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        
        # ROC-AUC calculation with error handling
        try:
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                # Binary classification
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                # Multi-class classification
                metrics["roc_auc"] = roc_auc_score(
                    y_test,
                    y_proba,
                    multi_class="ovr",
                    average="weighted",
                    labels=self.model.classes_
                )
        except ValueError:
            # ROC-AUC not computable (e.g., single class in test set)
            metrics["roc_auc"] = None

        return metrics

    # -------------------------------
    # Hyperparameter Info for UI
    # -------------------------------
    @staticmethod
    def get_param_info():
        """Return information about hyperparameters for UI."""
        return {
            'C': {
                'type': 'float',
                'min': 0.001,
                'max': 100.0,
                'default': 0.5,
                'description': 'Inverse of regularization strength (smaller = stronger)'
            },
            'max_iter': {
                'type': 'int',
                'min': 50,
                'max': 3000,
                'default': 2000,
                'description': 'Maximum number of iterations for solver'
            },
            'solver': {
                'type': 'select',
                'options': ['lbfgs', 'liblinear', 'saga', 'newton-cg'],
                'default': 'lbfgs',
                'description': 'Optimization algorithm'
            }
        }


