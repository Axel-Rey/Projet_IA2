# ============================================================
# PURPOSE OF THIS SCRIPT
# ============================================================
# This script builds and optimizes a k-Nearest Neighbors model to
# predict movie ratings based on features from the processed
# datasets. It includes:
# - Loading train/test datasets
# - Feature extraction and preparation
# - Initial model training
# - Hyperparameter optimization (GridSearchCV)
# - Model evaluation (metrics, visualization)
# - Cross-validation analysis
#
# The model predicts vote_average (movie rating) using
# features like budget, runtime, popularity, genres, etc.
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
from sklearn.neighbors import KNeighborsRegressor  # for the model
from sklearn.model_selection import GridSearchCV, cross_val_score  # for optimization
from sklearn.preprocessing import StandardScaler  # for feature scaling
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
# Step 3: Feature engineering and preparation
# ============================================================
print("\nStep 3: Preparing features...")

def extract_features(df):
    """
    Extract and prepare features from the dataset for the model.
    
    Parameters:
        df (pd.DataFrame): Input dataset with raw features
    
    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    df = df.copy()
    
    # Handle missing values in numeric features
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(df['budget'].median())
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(df['runtime'].median())
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(df['popularity'].median())
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(df['vote_count'].median())
    
    # Feature: number of genres
    df['num_genres'] = df['genres'].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0)
    
    # Feature: number of actors
    df['num_actors'] = df['actors'].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0)
    
    # Feature: number of crew members
    df['num_crew'] = df['crew'].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0)
    
    # Feature: number of keywords
    df['num_keywords'] = df['keywords'].apply(lambda x: len(ast.literal_eval(x)) if isinstance(x, str) else 0)
    
    # Feature: budget per minute of runtime (avoid division by zero)
    df['budget_per_minute'] = df['budget'] / (df['runtime'] + 1)
    
    # Feature: log popularity (to handle skewed distribution)
    df['log_popularity'] = np.log1p(df['popularity'])
    
    # Feature: log vote count
    df['log_vote_count'] = np.log1p(df['vote_count'])
    
    return df

# Apply feature engineering to both datasets
train_df = extract_features(train_df)
test_df = extract_features(test_df)

# Select features for the model (excluding target and non-numeric columns)
feature_columns = [
    'budget', 'runtime', 'popularity', 'vote_count',
    'num_genres', 'num_actors', 'num_crew', 'num_keywords',
    'budget_per_minute', 'log_popularity', 'log_vote_count'
]

# Prepare X (features) and y (target)
X_train = train_df[feature_columns].copy()
y_train = train_df['vote_average'].copy()

X_test = test_df[feature_columns].copy()
y_test = test_df['vote_average'].copy()

print(f"\n   X_train shape: {X_train.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   X_test shape: {X_test.shape}")
print(f"   y_test shape: {y_test.shape}")

# ============================================================
# Step 3b: Feature Scaling (Important for kNN)
# ============================================================
print("\nStep 3b: Scaling features (important for kNN)...")

# Initialize the scaler
scaler = StandardScaler()

# Fit on training data and transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames for consistency
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)

print(f"   Features scaled successfully.")


# ============================================================
# Step 4: Train initial kNN model
# ============================================================
print("\nStep 4: Training initial kNN model...")

# Create initial model with reasonable defaults
knn_model = KNeighborsRegressor(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    n_jobs=-1
)

# Train the model
knn_model.fit(X_train_scaled, y_train)

# Evaluate initial model
y_pred_initial = knn_model.predict(X_test_scaled)
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

# Define parameter grid for hyperparameter tuning (focused on key parameters for kNN)
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

print(f"   Testing hyperparameters with 5-fold CV...\"")
print(f"   - n_neighbors: {param_grid['n_neighbors']}")
print(f"   - weights: {param_grid['weights']}")

# Create GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(
    KNeighborsRegressor(n_jobs=-1),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2  # Show detailed progress
)

print("\n   Training models (this may take a minute)...\n")
# Fit GridSearchCV on scaled data
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_knn_model = grid_search.best_estimator_

print(f"\n   ✓ GridSearch completed!")
print(f"   Best parameters found: {grid_search.best_params_}")
print(f"   Best cross-validation R² Score: {grid_search.best_score_:.4f}")


# ============================================================
# Step 6: Evaluate optimized model
# ============================================================
print("\nStep 6: Evaluating optimized model...")

# Make predictions with the optimized model
y_pred_optimized = best_knn_model.predict(X_test_scaled)

# Calculate evaluation metrics
r2_optimized = r2_score(y_test, y_pred_optimized)
rmse_optimized = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)

print(f"   Optimized R² Score: {r2_optimized:.4f}")
print(f"   Optimized RMSE: {rmse_optimized:.4f}")
print(f"   Optimized MAE: {mae_optimized:.4f}")

# Calculate improvement
improvement_r2 = ((r2_optimized - initial_r2) / abs(initial_r2)) * 100 if initial_r2 != 0 else 0
improvement_rmse = ((initial_rmse - rmse_optimized) / initial_rmse) * 100

print(f"\n   Improvement in R² Score: {improvement_r2:+.2f}%")
print(f"   Improvement in RMSE: {improvement_rmse:+.2f}%")


# ============================================================
# Step 7: Cross-validation analysis
# ============================================================
print("\nStep 7: Performing cross-validation analysis...")

# 5-fold cross-validation on the optimized model
cv_scores = cross_val_score(
    best_knn_model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring='r2'
)

print(f"   Cross-validation R² Scores: {cv_scores}")
print(f"   Mean CV R² Score: {cv_scores.mean():.4f}")
print(f"   Std CV R² Score: {cv_scores.std():.4f}")


# ============================================================
# Step 8: Feature Importance Analysis (Based on proximity impact)
# ============================================================
print("\nStep 8: Analyzing feature statistics...")

# For kNN, we calculate feature importance based on variance and range
feature_stats = pd.DataFrame({
    'Feature': feature_columns,
    'Mean': X_train_scaled.mean(),
    'Std': X_train_scaled.std(),
    'Min': X_train_scaled.min(),
    'Max': X_train_scaled.max()
})

print(f"\n   Feature Statistics:")
print(feature_stats)


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

# 4. Feature statistics (Top 10 by Std)
ax4 = plt.subplot(2, 3, 4)
top_features = feature_stats.nlargest(10, 'Std')
ax4.barh(top_features['Feature'], top_features['Std'])
ax4.set_xlabel('Standard Deviation')
ax4.set_title('Top 10 Features by Standard Deviation')
ax4.grid(True, alpha=0.3, axis='x')

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

print("\n" + "="*60)
