# 🧠 ML Classification Playground

**Repository:** [https://github.com/2025aa05482-bits/ML-assignment2](https://github.com/2025aa05482-bits/ML-assignment2)

An interactive Streamlit application for exploring, training, and comparing machine learning classification models.

## 📋 Features

- **Multiple ML Models**: Compare 6 different classification algorithms
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Random Forest
  - XGBoost

- **Interactive Hyperparameter Tuning**: Adjust model parameters in real-time via the sidebar

- **Built-in Datasets**: 
  - Iris
  - Wine
  - Breast Cancer
  - Synthetic (custom generated)

- **Custom Data Upload**: Upload your own CSV files for classification

- **Comprehensive Metrics**:
  - Accuracy, Precision, Recall, F1 Score
  - ROC-AUC (for binary classification)
  - Confusion Matrix visualization
  - Feature Importance plots

- **Beautiful Modern UI**: Dark theme with gradient accents and smooth animations

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone/download this repository:
```bash
cd ML-Aaasignment2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
ML-Aaasignment2/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── model/
    ├── __init__.py    # Package initialization
    ├── logistic.py    # Logistic Regression implementation
    ├── dt.py          # Decision Tree implementation
    ├── knn.py         # K-Nearest Neighbors implementation
    ├── nb.py          # Naive Bayes implementation
    ├── rf.py          # Random Forest implementation
    └── xgb.py         # XGBoost implementation
```

## 🎮 Usage Guide

### 1. Select Data Source
- Choose from **Sample Dataset** or **Upload CSV**
- For sample datasets, pick from Iris, Wine, Breast Cancer, or Synthetic
- For CSV upload, select your target column

### 2. Choose a Model
- Select from 6 available classification models
- Read the model description to understand its approach

### 3. Configure Hyperparameters
- Adjust model-specific parameters using sliders and dropdowns
- Each parameter includes a helpful description

### 4. Training Settings
- Set test/train split ratio (default: 20% test)
- Enable/disable feature scaling
- Set random state for reproducibility

### 5. Train & Evaluate
- Click "Train Model" to start training
- View comprehensive metrics and visualizations
- Compare confusion matrix and metrics radar chart

## 🔧 Model Details

| Model | Best For | Key Parameters |
|-------|----------|----------------|
| Logistic Regression | Linear separable data | C, max_iter, solver |
| Decision Tree | Interpretable models | max_depth, min_samples_split |
| KNN | Small datasets | n_neighbors, weights, metric |
| Naive Bayes | Text classification, fast training | nb_type, var_smoothing |
| Random Forest | General purpose, feature importance | n_estimators, max_depth |
| XGBoost | High accuracy, competitions | learning_rate, max_depth, n_estimators |

## 📊 Metrics Explained

- **Accuracy**: Overall correct predictions / total predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)  
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (binary classification only)

## 🎨 Customization

Each model class in the `model/` directory follows a consistent interface:

```python
class ModelName:
    def __init__(self, **hyperparameters)
    def build_model(self)
    def train(self, X_train, y_train)
    def predict(self, X)
    def predict_proba(self, X)
    def evaluate(self, X_test, y_test)
    
    @staticmethod
    def get_param_grid()      # For grid search
    
    @staticmethod
    def get_param_info()      # For UI generation
```

## 📝 License

This project is created for educational purposes as part of ML Assignment 2.

## 🤝 Contributing

Feel free to extend this project by:
- Adding new models
- Implementing cross-validation
- Adding model persistence (save/load)
- Creating model comparison views

