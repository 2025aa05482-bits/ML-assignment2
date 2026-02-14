"""
K-Nearest Neighbors Model Implementation
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np


class KNNModel:
    """K-Nearest Neighbors classifier wrapper with hyperparameter tuning support."""
    
    def __init__(self, n_neighbors=15, weights='distance', metric='minkowski', p=2):
        """
        Initialize KNN model.
        
        Parameters:
        -----------
        n_neighbors : int, default=5
            Number of neighbors to use
        weights : str, default='uniform'
            Weight function ('uniform' or 'distance')
        metric : str, default='minkowski'
            Distance metric to use
        p : int, default=2
            Power parameter for Minkowski metric (1=manhattan, 2=euclidean)
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.model = None
        
    def build_model(self):
        """Build and return the KNN model."""
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p
        )
        return self.model
    
    def train(self, X_train, y_train):
        """
        Train the model on the given data.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        """
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions on the given data."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Returns:
        --------
        dict : Dictionary containing accuracy, precision, recall, f1, and confusion matrix
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # ROC-AUC calculation with multi-class support
        try:
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(
                    y_test, y_proba, 
                    multi_class='ovr', 
                    average='weighted'
                )
        except ValueError:
            metrics['roc_auc'] = None
        
        return metrics
    
    @staticmethod
    def get_param_grid():
        """Return hyperparameter grid for tuning."""
        return {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski', 'euclidean', 'manhattan'],
            'p': [1, 2]
        }
    
    @staticmethod
    def get_param_info():
        """Return information about hyperparameters for UI."""
        return {
            'n_neighbors': {
                'type': 'int',
                'min': 1,
                'max': 50,
                'default': 15,
                'description': 'Number of neighbors to consider'
            },
            'weights': {
                'type': 'select',
                'options': ['uniform', 'distance'],
                'default': 'distance',
                'description': 'Weight function for predictions'
            },
            'metric': {
                'type': 'select',
                'options': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
                'default': 'minkowski',
                'description': 'Distance metric'
            },
            'p': {
                'type': 'int',
                'min': 1,
                'max': 5,
                'default': 2,
                'description': 'Power parameter (1=Manhattan, 2=Euclidean)'
            }
        }

