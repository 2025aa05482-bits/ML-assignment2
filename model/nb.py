"""
Naive Bayes Model Implementation
"""
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np


class NaiveBayesModel:
    """Naive Bayes classifier wrapper with hyperparameter tuning support."""
    
    def __init__(self, nb_type='gaussian', var_smoothing=1e-8, alpha=1.0):
        """
        Initialize Naive Bayes model.
        
        Parameters:
        -----------
        nb_type : str, default='gaussian'
            Type of Naive Bayes ('gaussian', 'multinomial', 'bernoulli')
        var_smoothing : float, default=1e-9
            Variance smoothing for Gaussian NB
        alpha : float, default=1.0
            Laplace smoothing parameter for Multinomial/Bernoulli NB
        """
        self.nb_type = nb_type
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.model = None
        
    def build_model(self):
        """Build and return the Naive Bayes model."""
        if self.nb_type == 'gaussian':
            self.model = GaussianNB(var_smoothing=self.var_smoothing)
        elif self.nb_type == 'multinomial':
            self.model = MultinomialNB(alpha=self.alpha)
        elif self.nb_type == 'bernoulli':
            self.model = BernoulliNB(alpha=self.alpha)
        else:
            raise ValueError(f"Unknown NB type: {self.nb_type}")
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
        
        # For multinomial NB, ensure non-negative values
        if self.nb_type == 'multinomial':
            X_train = np.abs(X_train)
        
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Make predictions on the given data."""
        if self.nb_type == 'multinomial':
            X = np.abs(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions."""
        if self.nb_type == 'multinomial':
            X = np.abs(X)
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
            'nb_type': ['gaussian', 'bernoulli'],
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6],
            'alpha': [0.1, 0.5, 1.0, 2.0]
        }
    
    @staticmethod
    def get_param_info():
        """Return information about hyperparameters for UI."""
        return {
            'nb_type': {
                'type': 'select',
                'options': ['gaussian', 'bernoulli'],
                'default': 'gaussian',
                'description': 'Type of Naive Bayes classifier'
            },
            'var_smoothing': {
                'type': 'float',
                'min': 1e-12,
                'max': 1e-3,
                'default': 1e-8,
                'description': 'Variance smoothing (Gaussian NB only)',
                'format': 'scientific'
            },
            'alpha': {
                'type': 'float',
                'min': 0.0,
                'max': 10.0,
                'default': 1.0,
                'description': 'Laplace smoothing parameter'
            }
        }

