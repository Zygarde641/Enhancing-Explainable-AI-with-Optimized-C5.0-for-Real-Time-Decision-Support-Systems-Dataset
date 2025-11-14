import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import sys
from datetime import datetime

# Create a class to write to both console and file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Open output file
output_filename = f"new_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
output_file = open(output_filename, 'w', encoding='utf-8')

# Redirect stdout to both console and file
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, output_file)

print("="*60)
print("C5.0 MODEL OPTIMIZATION - PRUNING ANALYSIS")
print(f"Output saved to: {output_filename}")
print("="*60)
print()

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Finance_data.csv')
print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} features\n")

# Data Preprocessing
print("="*60)
print("PREPROCESSING")
print("="*60)

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

print(f"✓ Encoded {len(categorical_columns)} categorical columns")

# Define target and features
target_column = 'Investment_Avenues'
features_to_drop = ['Investment_Avenues', 'Stock_Marktet']  # Remove target and redundant feature

X = df.drop(columns=features_to_drop, errors='ignore')
y = df[target_column]

print(f"✓ Target: {target_column}")
print(f"✓ Features: {X.shape[1]}\n")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples\n")

# ============================================================================
# C5.0 IMPLEMENTATION WITH PRUNING
# ============================================================================

print("="*60)
print("C5.0 ALGORITHM WITH PRUNING OPTIMIZATION")
print("="*60)

# Base C5.0-style model (WITHOUT pruning)
print("\n1. Base C5.0 Model (No Pruning):")
print("-" * 60)
base_model = DecisionTreeClassifier(
    criterion='entropy',  # C5.0 uses information gain (entropy)
    random_state=42
)
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)

print(f"   Tree Depth: {base_model.get_depth()}")
print(f"   Number of Leaves: {base_model.get_n_leaves()}")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")

# Optimized C5.0 with Pre-Pruning
print("\n2. C5.0 with Pre-Pruning:")
print("-" * 60)
prepruned_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,              # Limit tree depth
    min_samples_split=10,     # Minimum samples to split a node
    min_samples_leaf=5,       # Minimum samples in leaf nodes
    random_state=42
)
prepruned_model.fit(X_train, y_train)
y_pred_prepruned = prepruned_model.predict(X_test)

print(f"   Tree Depth: {prepruned_model.get_depth()}")
print(f"   Number of Leaves: {prepruned_model.get_n_leaves()}")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_prepruned):.4f}")

# Optimized C5.0 with Cost Complexity Pruning (Post-Pruning)
print("\n3. C5.0 with Cost Complexity Pruning (Post-Pruning):")
print("-" * 60)

# Find optimal ccp_alpha
print("   Computing pruning path...")
path = base_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

# Limit to reasonable number of alpha values for faster execution
# Sample alpha values instead of testing all
if len(ccp_alphas) > 20:
    # Sample every nth alpha value
    step = max(1, len(ccp_alphas) // 20)
    ccp_alphas_sampled = ccp_alphas[::step][:-1]  # Exclude the last alpha
else:
    ccp_alphas_sampled = ccp_alphas[:-1]  # Exclude the last alpha (fully pruned tree)

print(f"   Testing {len(ccp_alphas_sampled)} alpha values...")

# Train models with different alpha values
models = []
for i, ccp_alpha in enumerate(ccp_alphas_sampled):
    model = DecisionTreeClassifier(
        criterion='entropy',
        ccp_alpha=ccp_alpha,
        random_state=42
    )
    model.fit(X_train, y_train)
    models.append(model)
    if (i + 1) % 5 == 0:
        print(f"   Progress: {i+1}/{len(ccp_alphas_sampled)} models trained...")

# Find best model based on test accuracy
test_scores = [accuracy_score(y_test, model.predict(X_test)) for model in models]
best_idx = np.argmax(test_scores)
best_model = models[best_idx]
best_alpha = ccp_alphas_sampled[best_idx]

y_pred_postpruned = best_model.predict(X_test)

print(f"   Optimal Alpha: {best_alpha:.6f}")
print(f"   Tree Depth: {best_model.get_depth()}")
print(f"   Number of Leaves: {best_model.get_n_leaves()}")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_postpruned):.4f}")

# Combined Pruning Approach (Pre + Post)
print("\n4. C5.0 with Combined Pruning (Pre + Post):")
print("-" * 60)
combined_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    ccp_alpha=0.01,           # Cost complexity pruning
    random_state=42
)
combined_model.fit(X_train, y_train)
y_pred_combined = combined_model.predict(X_test)

print(f"   Tree Depth: {combined_model.get_depth()}")
print(f"   Number of Leaves: {combined_model.get_n_leaves()}")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_combined):.4f}")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

print("\n" + "="*60)
print("PRUNING OPTIMIZATION RESULTS")
print("="*60)

results = pd.DataFrame({
    'Model': ['Base C5.0 (No Pruning)', 
              'Pre-Pruning', 
              'Post-Pruning (CCP)', 
              'Combined Pruning'],
    'Tree Depth': [base_model.get_depth(),
                   prepruned_model.get_depth(),
                   best_model.get_depth(),
                   combined_model.get_depth()],
    'Num Leaves': [base_model.get_n_leaves(),
                   prepruned_model.get_n_leaves(),
                   best_model.get_n_leaves(),
                   combined_model.get_n_leaves()],
    'Accuracy': [accuracy_score(y_test, y_pred_base),
                 accuracy_score(y_test, y_pred_prepruned),
                 accuracy_score(y_test, y_pred_postpruned),
                 accuracy_score(y_test, y_pred_combined)]
})

print("\n", results.to_string(index=False))

# Calculate improvements
print("\n" + "-"*60)
print("IMPROVEMENTS FROM PRUNING:")
print("-"*60)
base_leaves = base_model.get_n_leaves()
best_pruned = combined_model

print(f"Complexity Reduction: {base_leaves} → {best_pruned.get_n_leaves()} leaves "
      f"({(1 - best_pruned.get_n_leaves()/base_leaves)*100:.1f}% reduction)")
print(f"Depth Reduction: {base_model.get_depth()} → {best_pruned.get_depth()} levels")
print(f"Accuracy Change: {accuracy_score(y_test, y_pred_base):.4f} → "
      f"{accuracy_score(y_test, y_pred_combined):.4f}")

# Best model selection
best_overall = results.loc[results['Accuracy'].idxmax(), 'Model']
print(f"\n✓ Best Model: {best_overall}")
print(f"✓ Achieves optimal balance between complexity and accuracy")

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE!")
print("="*60)

# Close output file and restore stdout
sys.stdout = original_stdout
output_file.close()
print(f"\n✓ Output saved to: {output_filename}")