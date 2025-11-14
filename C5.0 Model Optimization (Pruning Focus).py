import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Finance_data.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Data Preprocessing
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Handle missing values
df = df.dropna()

# Encode categorical variables
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"\nEncoded {len(categorical_columns)} categorical columns")
print("Categorical columns:", list(categorical_columns))

# Feature Selection - removing highly correlated or redundant features
# Based on the paper's methodology of pruning unnecessary complexity
print("\n" + "="*50)
print("FEATURE SELECTION & PRUNING")
print("="*50)

# Define target variable - predicting Investment_Avenues (Yes/No)
target_column = 'Investment_Avenues'

# Features to drop (based on domain knowledge and redundancy)
features_to_drop = [
    'Investment_Avenues',  # Target variable
    'Stock_Marktet',  # Redundant with Equity_Market (note: typo in original)
]

# Remove features that are direct preferences (keeping only derived insights)
X = df.drop(columns=features_to_drop, errors='ignore')
y = df[target_column]

print(f"Target variable: {target_column}")
print(f"Number of features after pruning: {X.shape[1]}")
print(f"Features retained: {list(X.columns)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Model Development - C5.0 Style Decision Tree
print("\n" + "="*50)
print("MODEL DEVELOPMENT - OPTIMIZED DECISION TREE")
print("="*50)

# Base Decision Tree (simulating C5.0 with pruning)
# Using cost complexity pruning (ccp_alpha) as optimization technique
dt_model = DecisionTreeClassifier(
    criterion='entropy',  # C5.0 uses information gain (entropy)
    max_depth=5,  # Pruning - limit tree depth
    min_samples_split=10,  # Pruning - minimum samples to split
    min_samples_leaf=5,  # Pruning - minimum samples in leaf
    random_state=42,
    ccp_alpha=0.01  # Cost complexity pruning parameter
)

dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluate base model
print("\nBase Decision Tree Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_dt, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_dt, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_dt, average='weighted'):.4f}")

# Boosting Optimization - AdaBoost (simulating C5.0 boosting)
print("\n" + "="*50)
print("BOOSTING OPTIMIZATION")
print("="*50)

# AdaBoost with pruned decision trees as base estimators
boosted_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        criterion='entropy',
        max_depth=3,  # Even more pruned for boosting
        min_samples_split=10,
        min_samples_leaf=5,
        ccp_alpha=0.01
    ),
    n_estimators=50,  # Number of boosting iterations
    learning_rate=1.0,
    random_state=42
)

boosted_model.fit(X_train, y_train)
y_pred_boosted = boosted_model.predict(X_test)

# Evaluate boosted model
print("\nBoosted Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_boosted):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_boosted, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_boosted, average='weighted'):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_boosted, average='weighted'):.4f}")

# Detailed Classification Report
print("\n" + "="*50)
print("DETAILED CLASSIFICATION REPORT")
print("="*50)
print("\nBoosted Model Classification Report:")
print(classification_report(y_test, y_pred_boosted))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_boosted)
print(cm)

# Feature Importance Analysis
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': boosted_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model Comparison
models = ['Base Decision Tree', 'Boosted Model']
accuracies = [
    accuracy_score(y_test, y_pred_dt),
    accuracy_score(y_test, y_pred_boosted)
]
axes[0, 0].bar(models, accuracies, color=['#3498db', '#2ecc71'])
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Model Performance Comparison')
axes[0, 0].set_ylim([0.5, 1.0])
for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

# 2. Feature Importance
top_features = feature_importance.head(10)
axes[0, 1].barh(top_features['Feature'], top_features['Importance'], color='#e74c3c')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Top 10 Feature Importance')
axes[0, 1].invert_yaxis()

# 3. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title('Confusion Matrix - Boosted Model')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# 4. Performance Metrics Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
dt_scores = [
    accuracy_score(y_test, y_pred_dt),
    precision_score(y_test, y_pred_dt, average='weighted'),
    recall_score(y_test, y_pred_dt, average='weighted'),
    f1_score(y_test, y_pred_dt, average='weighted')
]
boosted_scores = [
    accuracy_score(y_test, y_pred_boosted),
    precision_score(y_test, y_pred_boosted, average='weighted'),
    recall_score(y_test, y_pred_boosted, average='weighted'),
    f1_score(y_test, y_pred_boosted, average='weighted')
]

x = np.arange(len(metrics))
width = 0.35
axes[1, 1].bar(x - width/2, dt_scores, width, label='Base DT', color='#3498db')
axes[1, 1].bar(x + width/2, boosted_scores, width, label='Boosted', color='#2ecc71')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Performance Metrics Comparison')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics, rotation=45)
axes[1, 1].legend()
axes[1, 1].set_ylim([0.5, 1.0])

plt.tight_layout()
plt.savefig('c50_analysis_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved as 'c50_analysis_results.png'")
plt.show()

# Rule Extraction (Explainability)
print("\n" + "="*50)
print("EXPLAINABLE AI - DECISION RULES")
print("="*50)

from sklearn.tree import export_text

# Extract rules from one of the base estimators
base_tree = boosted_model.estimators_[0]
tree_rules = export_text(base_tree, feature_names=list(X.columns), max_depth=3)
print("\nSample Decision Rules (from first base estimator):")
print(tree_rules[:1000] + "...")  # Print first 1000 characters

# Summary Statistics
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"\n✓ Dataset processed: {df.shape[0]} records")
print(f"✓ Features after pruning: {X.shape[1]}")
print(f"✓ Base Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"✓ Boosted Model Accuracy: {accuracy_score(y_test, y_pred_boosted):.4f}")
print(f"✓ Improvement from boosting: {(accuracy_score(y_test, y_pred_boosted) - accuracy_score(y_test, y_pred_dt)):.4f}")
print("\n✓ Analysis complete!")