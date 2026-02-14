"""
XGBoost Model Implementation
"""
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np


class XGBoostModel:
    """XGBoost classifier wrapper with hyperparameter tuning support."""
    
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.05, 
                 subsample=0.8, colsample_bytree=0.8, gamma=0.1, 
                 scale_pos_weight=1, random_state=42):
        """
        Initialize XGBoost model.
        
        Parameters:
        -----------
        n_estimators : int, default=200
            Number of boosting rounds
        max_depth : int, default=8
            Maximum depth of trees
        learning_rate : float, default=0.1
            Step size shrinkage (eta)
        subsample : float, default=0.8
            Subsample ratio of training instances
        colsample_bytree : float, default=0.8
            Subsample ratio of columns
        gamma : float, default=0
            Minimum loss reduction for split
        scale_pos_weight : float, default=1
            Balancing of positive and negative weights (for imbalanced data)
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.model = None
        
    def build_model(self):
        """Build and return the XGBoost model."""
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric='mlogloss',
            verbosity=0,
            reg_alpha=0.1,
            reg_lambda=1.0
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
    
    def get_feature_importance(self):
        """Return feature importances from the trained model."""
        if self.model is not None:
            return self.model.feature_importances_
        return None
    
    @staticmethod
    def get_param_grid():
        """Return hyperparameter grid for tuning."""
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }
    
    @staticmethod
    def get_param_info():
        """Return information about hyperparameters for UI."""
        return {
            'n_estimators': {
                'type': 'int',
                'min': 10,
                'max': 500,
                'default': 300,
                'description': 'Number of boosting rounds'
            },
            'max_depth': {
                'type': 'int',
                'min': 1,
                'max': 20,
                'default': 6,
                'description': 'Maximum depth of trees'
            },
            'learning_rate': {
                'type': 'float',
                'min': 0.001,
                'max': 1.0,
                'default': 0.05,
                'description': 'Learning rate (eta)'
            },
            'subsample': {
                'type': 'float',
                'min': 0.1,
                'max': 1.0,
                'default': 1.0,
                'description': 'Subsample ratio of training data'
            },
            'colsample_bytree': {
                'type': 'float',
                'min': 0.1,
                'max': 1.0,
                'default': 1.0,
                'description': 'Subsample ratio of columns'
            },
            'gamma': {
                'type': 'float',
                'min': 0.0,
                'max': 10.0,
                'default': 0.1,
                'description': 'Minimum loss reduction for split'
            },
            'scale_pos_weight': {
                'type': 'float',
                'min': 1.0,
                'max': 20.0,
                'default': 1.0,
                'description': 'Balancing weight for imbalanced data'
            }
        }

