# ============================================================
# PURPOSE OF THIS SCRIPT
# ============================================================
# This script builds and optimizes a TF-IDF + Linear Regression
# model to predict movie ratings based on keywords from the 
# processed datasets. It includes:
# - Loading train/test datasets
# - TF-IDF vectorization of keywords
# - Initial model training
# - Hyperparameter optimization (GridSearchCV)
# - Model evaluation (metrics, visualization)
# - Cross-validation analysis
#
# The model predicts vote_average (movie rating) using
# TF-IDF features extracted from movie keywords.
# ============================================================


# ============================================================
# Step 1: Imports
# ============================================================
print("Step 1: Importing libraries...")

import pandas as pd                          # for data manipulation
import numpy as np                           # for numerical operations
import os                                    # for file path management
import ast                                   # for safely evaluating strings
import matplotlib.pyplot as plt              # for plotting
import seaborn as sns                        # for enhanced plotting
from sklearn.feature_extraction.text import TfidfVectorizer  # for TF-IDF vectorization
from sklearn.linear_model import Ridge                       # for regression model
from sklearn.model_selection import GridSearchCV, cross_val_score  # for optimization
from sklearn.metrics import (                # for model evaluation
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from tqdm import tqdm                        # for progress bars
import warnings
warnings.filterwarnings('ignore')

print("   Libraries imported successfully.")


# ============================================================
# Step 2: Resolve paths and load datasets
# ============================================================
print("\nStep 2: Loading datasets...")

# Absolute path to this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Data directory is assumed to be ../../Data/Datasets relative to this script
DATA_DIR = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "Data", "Datasets"))

# Load train and test datasets
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_dataset.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_dataset.csv"))

print(f"   Train dataset shape: {train_df.shape}")
print(f"   Test dataset shape: {test_df.shape}")


# ============================================================
# Step 3: Feature engineering with TF-IDF vectorization
# ============================================================
print("\nStep 3: Preparing TF-IDF features from keywords...")

def extract_keywords_as_text(keywords_str):
    """
    Extract keywords from string representation and convert to text format.
    
    Parameters:
        keywords_str (str): String representation of keywords list
    
    Returns:
        str: Space-separated keywords or empty string if invalid
    """
    try:
        if isinstance(keywords_str, str):
            keywords_list = ast.literal_eval(keywords_str)
            if isinstance(keywords_list, list) and len(keywords_list) > 0:
                # Join keywords with spaces for TF-IDF vectorization
                return ' '.join(str(kw) for kw in keywords_list)
    except:
        pass
    return ''

# Extract keywords as text strings
print("   Converting keywords to text format...")
train_keywords_text = train_df['keywords'].apply(extract_keywords_as_text)
test_keywords_text = test_df['keywords'].apply(extract_keywords_as_text)

print(f"   Train documents with keywords: {(train_keywords_text.str.len() > 0).sum()}/{len(train_keywords_text)}")
print(f"   Test documents with keywords: {(test_keywords_text.str.len() > 0).sum()}/{len(test_keywords_text)}")

# Initialize TF-IDF vectorizer
print("\n   Initializing TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=100,           # Limit to top 100 features to avoid overfitting
    min_df=2,                   # Minimum document frequency
    max_df=0.8,                 # Maximum document frequency (ignore very common terms)
    lowercase=True,
    stop_words='english'
)

# Fit TF-IDF on training data
print("   Fitting TF-IDF vectorizer on training keywords...")
X_train_tfidf = tfidf_vectorizer.fit_transform(train_keywords_text)
X_test_tfidf = tfidf_vectorizer.transform(test_keywords_text)

# Convert sparse matrices to dense arrays for easier manipulation
X_train = X_train_tfidf.toarray()
X_test = X_test_tfidf.toarray()

# Get target variable
y_train = train_df['vote_average'].copy()
y_test = test_df['vote_average'].copy()

print(f"\n   X_train shape: {X_train.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   X_test shape: {X_test.shape}")
print(f"   y_test shape: {y_test.shape}")

# Get feature names (top keywords)
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"\n   Number of TF-IDF features: {len(feature_names)}")
print(f"   Top 10 features: {', '.join(feature_names[:10])}")


# ============================================================
# Step 4: Train initial TF-IDF + Ridge model
# ============================================================
print("\nStep 4: Training initial TF-IDF model...")

# Create initial Ridge regression model
ridge_model = Ridge(
    alpha=1.0,
    solver='auto',
    random_state=42
)

# Train the model
ridge_model.fit(X_train, y_train)

# Evaluate initial model
y_pred_initial = ridge_model.predict(X_test)
initial_r2 = r2_score(y_test, y_pred_initial)
initial_rmse = np.sqrt(mean_squared_error(y_test, y_pred_initial))
initial_mae = mean_absolute_error(y_test, y_pred_initial)

print(f"   Initial model trained.")
print(f"   Initial R² Score: {initial_r2:.4f}")
print(f"   Initial RMSE: {initial_rmse:.4f}")
print(f"   Initial MAE: {initial_mae:.4f}")


# ============================================================
# Step 5: Hyperparameter optimization with GridSearchCV
# ============================================================
print("\nStep 5: Optimizing hyperparameters...")

# Define parameter grid for Ridge regression
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'solver': ['auto', 'svd']
}

print(f"   Testing hyperparameters with 5-fold CV...")
print(f"   - alpha: {param_grid['alpha']}")
print(f"   - solver: {param_grid['solver']}")

# Create GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring='r2',
    verbose=2  # Show detailed progress
)

print("\n   Training models (this may take a minute)...\n")
# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model
best_ridge_model = grid_search.best_estimator_

print(f"\n   ✓ GridSearch completed!")
print(f"   Best parameters found: {grid_search.best_params_}")
print(f"   Best cross-validation R² Score: {grid_search.best_score_:.4f}")


# ============================================================
# Step 6: Evaluate optimized model
# ============================================================
print("\nStep 6: Evaluating optimized model...")

# Make predictions with the optimized model
y_pred_optimized = best_ridge_model.predict(X_test)

# Calculate evaluation metrics
r2_optimized = r2_score(y_test, y_pred_optimized)
rmse_optimized = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)

print(f"   Optimized R² Score: {r2_optimized:.4f}")
print(f"   Optimized RMSE: {rmse_optimized:.4f}")
print(f"   Optimized MAE: {mae_optimized:.4f}")

# Calculate improvement
improvement_r2 = ((r2_optimized - initial_r2) / abs(initial_r2)) * 100 if initial_r2 != 0 else 0
improvement_rmse = ((initial_rmse - rmse_optimized) / initial_rmse) * 100 if initial_rmse != 0 else 0

print(f"\n   Improvement in R² Score: {improvement_r2:+.2f}%")
print(f"   Improvement in RMSE: {improvement_rmse:+.2f}%")


# ============================================================
# Step 7: Cross-validation analysis
# ============================================================
print("\nStep 7: Performing cross-validation analysis...")

# 5-fold cross-validation on the optimized model
cv_scores = cross_val_score(
    best_ridge_model,
    X_train,
    y_train,
    cv=5,
    scoring='r2'
)

print(f"   Cross-validation R² Scores: {cv_scores}")
print(f"   Mean CV R² Score: {cv_scores.mean():.4f}")
print(f"   Std CV R² Score: {cv_scores.std():.4f}")


# ============================================================
# Step 8: Feature importance analysis (TF-IDF coefficients)
# ============================================================
print("\nStep 8: Analyzing feature importance (TF-IDF coefficients)...")

# Get model coefficients
coefficients = best_ridge_model.coef_

# Create feature importance dataframe
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n   Top 10 Most Important Keywords:")
print(feature_importance.head(10))

print(f"\n   Top 5 Positive Impact Keywords:")
print(feature_importance.nlargest(5, 'Coefficient')[['Feature', 'Coefficient']])

print(f"\n   Top 5 Negative Impact Keywords:")
print(feature_importance.nsmallest(5, 'Coefficient')[['Feature', 'Coefficient']])


# ============================================================
# Step 9: Visualization
# ============================================================
print("\nStep 9: Creating visualizations...")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Actual vs Predicted (scatter plot)
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_pred_optimized, alpha=0.5, s=20)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Rating')
ax1.set_ylabel('Predicted Rating')
ax1.set_title('Actual vs Predicted Ratings')
ax1.grid(True, alpha=0.3)

# 2. Residuals plot
ax2 = plt.subplot(2, 3, 2)
residuals = y_test - y_pred_optimized
ax2.scatter(y_pred_optimized, residuals, alpha=0.5, s=20)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Rating')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals Plot')
ax2.grid(True, alpha=0.3)

# 3. Distribution of residuals
ax3 = plt.subplot(2, 3, 3)
ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Residuals')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Residuals')
ax3.grid(True, alpha=0.3)

# 4. Top 10 important keywords (by absolute coefficient)
ax4 = plt.subplot(2, 3, 4)
top_features = feature_importance.head(10)
colors = ['green' if x > 0 else 'red' for x in top_features['Coefficient']]
ax4.barh(top_features['Feature'], top_features['Coefficient'], color=colors)
ax4.set_xlabel('Coefficient Value')
ax4.set_title('Top 10 Keywords by Impact')
ax4.grid(True, alpha=0.3, axis='x')
ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# 5. Cross-validation scores
ax5 = plt.subplot(2, 3, 5)
ax5.bar(range(1, len(cv_scores) + 1), cv_scores, color='skyblue', edgecolor='black')
ax5.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
ax5.set_xlabel('Fold')
ax5.set_ylabel('R² Score')
ax5.set_title('Cross-Validation Scores')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Model comparison (initial vs optimized)
ax6 = plt.subplot(2, 3, 6)
metrics = ['R² Score', 'RMSE', 'MAE']
initial_values = [initial_r2, initial_rmse, initial_mae]
optimized_values = [r2_optimized, rmse_optimized, mae_optimized]

x = np.arange(len(metrics))
width = 0.35

ax6.bar(x - width/2, initial_values, width, label='Initial Model', alpha=0.8)
ax6.bar(x + width/2, optimized_values, width, label='Optimized Model', alpha=0.8)
ax6.set_ylabel('Value')
ax6.set_title('Model Comparison: Initial vs Optimized')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save the figure
output_path = os.path.join(script_dir, "model_evaluation.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"   Visualization saved to: {output_path}")

plt.show()


# ============================================================
# Step 10: Summary and conclusions
# ============================================================
print("\n" + "="*60)
print("SUMMARY AND CONCLUSIONS")
print("="*60)

print(f"\nInitial Model Performance:")
print(f"  - R² Score: {initial_r2:.4f}")
print(f"  - RMSE: {initial_rmse:.4f}")
print(f"  - MAE: {initial_mae:.4f}")

print(f"\nOptimized Model Performance:")
print(f"  - R² Score: {r2_optimized:.4f}")
print(f"  - RMSE: {rmse_optimized:.4f}")
print(f"  - MAE: {mae_optimized:.4f}")

print(f"\nCross-Validation Results:")
print(f"  - Mean R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

print(f"\nBest Hyperparameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")

print(f"\nTop 5 Most Impactful Keywords:")
for idx, row in feature_importance.head(5).iterrows():
    direction = "positive" if row['Coefficient'] > 0 else "negative"
    print(f"  - {row['Feature']}: {row['Coefficient']:.4f} ({direction})")

print("\n" + "="*60)
