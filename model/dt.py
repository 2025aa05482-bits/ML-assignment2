"""
Decision Tree Model Implementation
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np


class DecisionTreeModel:
    """Decision Tree classifier wrapper with hyperparameter tuning support."""
    
    def __init__(self, max_depth=12, min_samples_split=5, min_samples_leaf=2, 
                 criterion='gini', random_state=42):
        """
        Initialize Decision Tree model.
        
        Parameters:
        -----------
        max_depth : int or None, default=None
            Maximum depth of the tree
        min_samples_split : int, default=2
            Minimum samples required to split an internal node
        min_samples_leaf : int, default=1
            Minimum samples required at a leaf node
        criterion : str, default='gini'
            Function to measure split quality ('gini' or 'entropy')
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.model = None
        
    def build_model(self):
        """Build and return the Decision Tree model."""
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.random_state,
            class_weight='balanced'  # Helps with imbalanced datasets
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
            'max_depth': [3, 5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    
    @staticmethod
    def get_param_info():
        """Return information about hyperparameters for UI."""
        return {
            'max_depth': {
                'type': 'int',
                'min': 1,
                'max': 50,
                'default': 12,
                'description': 'Maximum depth of the tree (None for unlimited)'
            },
            'min_samples_split': {
                'type': 'int',
                'min': 2,
                'max': 20,
                'default': 5,
                'description': 'Minimum samples to split a node'
            },
            'min_samples_leaf': {
                'type': 'int',
                'min': 1,
                'max': 20,
                'default': 2,
                'description': 'Minimum samples at a leaf node'
            },
            'criterion': {
                'type': 'select',
                'options': ['gini', 'entropy'],
                'default': 'gini',
                'description': 'Split quality measure'
            }
        }

