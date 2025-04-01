# Drug Classification  - Machine Learning Project 💊🔍

This project focuses on classifying drugs based on given features using machine learning techniques. It is a multi-class classification problem where the dataset includes various attributes that help in predicting the category of a drug.

## Project Overview
**Objective**: Classify drugs based on given medical attributes using machine learning techniques.

**Dataset**: California housing data - https://www.kaggle.com/datasets/camnugent/california-housing-prices?select=housing.csv

**Approach**: Implement various ML models, including Logistic Regression, Random Forest, SVM , KNN , Decision Tree , Naive Bayers  and GridSearchCV for hyperparameter tuning & Ensemble techinique like Hard Voting & Stacking to collectively increase the performance of above models.

## Features & Methodology
- **Data Preprocessing**: Handle missing values, outliers, and categorical encoding.
- **Feature Engineering**: Extract meaningful medical attributes to improve classification accuracy.
- **Data Scaling**: Normalize features using StandardScaler.
- **Data Visualization**: Utilize pair plots (Seaborn pairplot) to explore relationships between features.
- **Model Training**: Train multiple models (Logistic Regression, Random Forest , KNN , SVM , Decision Tree , Naive Bayers) & Ensemble Techinique - Hard Voting & Stacking.
- **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV.
- **Performance Evaluation**: Evaluate models using accuracy, precision, recall, and F1-score.
- **Confusion Matrix**: Visualize classification performance using a confusion matrix.
- **ROC-AUC Curve**: Analyze model performance with Receiver Operating Characteristic (ROC) and Area Under the Curve (AUC) scores.


## Libraries Used
- Scikit-learn (Logistic Regression , OneVsRestClassifier , Random Forest , SVM , KNN  , DecisionTree, NaiveBayers )
- NumPy
- Pandas
- Matplotlib
- Seaborn

## How to Run the Code?
**Install required libraries:**
  ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
  ```


## Installation
To get started, clone the repository and install the required packages:
```bash
git clone https://github.com/yourusername/drug-classification.git
cd drug-classification
```
```bash
pip install -r requirements.txt
```

## Usage
Run the main script to preprocess the data, train the models, and evaluate their performance:
```bash
python main.py
```
