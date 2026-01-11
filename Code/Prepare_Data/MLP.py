# ============================================================================
# MLP REGRESSION - ENTRAÎNEMENT ET ÉVALUATION
# ============================================================================
# Ce script charge les données préprocessées et entraîne un modèle MLP Regressor
# avec optimisation des hyperparamètres.
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
import pickle
import os
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore')

print("="*80)
print("MLP REGRESSION - MOVIE RATING PREDICTION")
print("="*80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES PRÉPROCESSÉES
# ============================================================================

print("\nStep 1: Loading preprocessed data...")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(BASE_DIR, "Data", "Processed")

# Charger les données
X_train_full = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test_final = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train_full = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))['vote_average'].values
y_test_final = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))['vote_average'].values
train_metadata = pd.read_csv(os.path.join(DATA_DIR, "train_metadata.csv"))
test_metadata = pd.read_csv(os.path.join(DATA_DIR, "test_metadata.csv"))

# Charger l'encodeur de genres
with open(os.path.join(DATA_DIR, "genre_encoder.pkl"), 'rb') as f:
    mlb = pickle.load(f)

print(f"   X_train: {X_train_full.shape}")
print(f"   X_test: {X_test_final.shape}")
print(f"   y_train: {y_train_full.shape}")
print(f"   y_test: {y_test_final.shape}")

# ============================================================================
# 2. SPLIT TRAIN/DEV POUR OPTIMISATION
# ============================================================================

print("\nStep 2: Creating train/dev split for hyperparameter optimization...")

X_train, X_dev, y_train, y_dev = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"   Training set: {X_train.shape[0]} movies")
print(f"   Development set: {X_dev.shape[0]} movies")

# ============================================================================
# 3. SCALING
# ============================================================================

print("\nStep 3: Scaling features...")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_final_scaled = scaler.transform(X_test_final)

print("   Features scaled with RobustScaler")

# ============================================================================
# 4. FONCTIONS POUR CRÉER LE MODÈLE
# ============================================================================

def create_model_MLPR(hidden_layer_sizes=(100,), activation='relu', solver='adam', 
                      max_iter=300, learning_rate='constant', alpha=0.01):
    """Créer un MLPRegressor avec les paramètres donnés"""
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        learning_rate=learning_rate,
        alpha=alpha,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    return model

def create_model_MLPR_test(hidden_layer_sizes=1, neurone_layer=100, activation='relu', 
                           solver='adam', max_iter=300, learning_rate='constant', alpha=0.01):
    """Créer un MLPRegressor avec architecture répétitive"""
    model = MLPRegressor(
        hidden_layer_sizes=(neurone_layer,) * hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
        learning_rate=learning_rate,
        alpha=alpha,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    return model

# ============================================================================
# 5. ANALYSE DES HYPERPARAMÈTRES
# ============================================================================

print("\n" + "="*80)
print("HYPERPARAMETER ANALYSIS")
print("="*80)

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.5]) 
plt.subplots_adjust(hspace=0.5, wspace=0.4)

structure = [50, 100, 150, 200]

# 1. Nombre de couches cachées
print("\n[1/4] Hidden Layer Sizes...")
errC = []
couches = [1, 2, 3, 4, 5]
for struc in structure:
    for couche in tqdm(couches, desc=f"Structure {struc}"):
        for replicate in range(3):
            mdl = create_model_MLPR_test(hidden_layer_sizes=couche, neurone_layer=int(struc/couche))
            mdl.fit(X_train_scaled, y_train)
            mae = mean_absolute_error(y_dev, mdl.predict(X_dev_scaled))
            errC.append({'Couches': couche, 'MAE': mae, 'Structure': struc})
dataCouche = pd.DataFrame(errC)

ax1 = fig.add_subplot(gs[0, 0])
sns.lineplot(x='Couches', y='MAE', hue='Structure', data=dataCouche, errorbar=('ci', 95), palette='viridis', ax=ax1)
ax1.set_title('Hidden Layer Sizes')
ax1.set_xlabel('Number of Layers')
ax1.set_ylabel('MAE')
ax1.grid(True, alpha=0.4)
ax1.legend().remove()

# 2. Fonction d'activation
print("\n[2/4] Activation Function...")
errA = []
activations = ['logistic', 'tanh', 'relu']
for struc in structure:
    for activation in tqdm(activations, desc=f"Structure {struc}"):
        for replicate in range(3):
            mdl = create_model_MLPR_test(hidden_layer_sizes=3, neurone_layer=int(struc/3), activation=activation)
            mdl.fit(X_train_scaled, y_train)
            mae = mean_absolute_error(y_dev, mdl.predict(X_dev_scaled))
            errA.append({'Activation': activation, 'MAE': mae, 'Structure': struc})
dataActivation = pd.DataFrame(errA)

ax2 = fig.add_subplot(gs[1, 1])
sns.barplot(x='Activation', y='MAE', hue='Structure', data=dataActivation, palette='viridis', ax=ax2)
ax2.set_title('Activation Function')
ax2.set_xlabel('Activation')
ax2.set_ylabel('MAE')
ax2.grid(True, alpha=0.4)
ax2.legend().remove()

# 3. Learning rate
print("\n[3/4] Learning Rate...")
errL = []
learning_rates = ['constant', 'invscaling', 'adaptive']
for struc in structure:
    for lr in tqdm(learning_rates, desc=f"Structure {struc}"):
        for replicate in range(3):
            mdl = create_model_MLPR_test(hidden_layer_sizes=3, neurone_layer=int(struc/3), learning_rate=lr)
            mdl.fit(X_train_scaled, y_train)
            mae = mean_absolute_error(y_dev, mdl.predict(X_dev_scaled))
            errL.append({'Learning Rate': lr, 'MAE': mae, 'Structure': struc})
dataLearningRate = pd.DataFrame(errL)

ax3 = fig.add_subplot(gs[1, 0])
sns.barplot(x='Learning Rate', y='MAE', hue='Structure', data=dataLearningRate, palette='viridis', ax=ax3)
ax3.set_title('Learning Rate')
ax3.set_xlabel('Learning Rate')
ax3.set_ylabel('MAE')
ax3.grid(True, alpha=0.4)
ax3.legend().remove()

# 4. Max iterations
print("\n[4/4] Max Iterations...")
errI = []
max_iters = np.linspace(100, 1000, 10).astype(int)
for struc in structure:
    for max_iter in tqdm(max_iters, desc=f"Structure {struc}"):
        for replicate in range(3):
            mdl = create_model_MLPR_test(hidden_layer_sizes=3, neurone_layer=int(struc/3), max_iter=max_iter)
            mdl.fit(X_train_scaled, y_train)
            mae = mean_absolute_error(y_dev, mdl.predict(X_dev_scaled))
            errI.append({'Max Iter': max_iter, 'MAE': mae, 'Structure': struc})
dataMaxIter = pd.DataFrame(errI)

ax4 = fig.add_subplot(gs[0, 1])
sns.lineplot(x='Max Iter', y='MAE', hue='Structure', data=dataMaxIter, errorbar=('ci', 95), palette='viridis', ax=ax4)
ax4.set_title('Max Iter')
ax4.set_xlabel('Max Iter')
ax4.set_ylabel('MAE')
ax4.grid(True, alpha=0.4)
ax4.legend().remove()

ax4.annotate("Multi Layer Perceptron Regressor Parameters Analysis", 
             fontsize=14, 
             xy=(0.3, 1.25), 
             xycoords='axes fraction',  
             ha='center',  
             va='center')

# Légende globale
ax5 = fig.add_subplot(gs[:, 2])
handles, labels = ax3.get_legend_handles_labels()
ax5.legend(handles, labels, loc='center', frameon=False)
ax5.axis('off')
ax5.annotate("LEGEND:  \n \nNumber of neurons", 
             fontsize=12, 
             xy=(0.45, 0.62), 
             xycoords='axes fraction',  
             ha='center',  
             va='center')

plt.savefig('mlp_hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. MEILLEUR MODÈLE
# ============================================================================

print("\n" + "="*80)
print("TRAINING BEST MODEL")
print("="*80)

# Déterminer la meilleure configuration
best_structure = dataCouche.groupby('Structure')['MAE'].mean().idxmin()
best_couches = dataCouche[dataCouche['Structure'] == best_structure].groupby('Couches')['MAE'].mean().idxmin()
best_activation = dataActivation.groupby('Activation')['MAE'].mean().idxmin()
best_lr = dataLearningRate.groupby('Learning Rate')['MAE'].mean().idxmin()

print(f"\n🏆 BEST CONFIGURATION:")
print(f"   Structure: {best_structure} neurons")
print(f"   Layers: {best_couches}")
print(f"   Activation: {best_activation}")
print(f"   Learning Rate: {best_lr}")

# Entraîner le meilleur modèle
model = create_model_MLPR_test(
    hidden_layer_sizes=best_couches, 
    neurone_layer=int(best_structure/best_couches),
    activation=best_activation,
    learning_rate=best_lr,
    max_iter=500
)

model.fit(X_train_full_scaled, y_train_full)

# Prédictions
y_pred_train = model.predict(X_train_full_scaled)
y_pred_test = model.predict(X_test_final_scaled)

# Métriques
mae_train = mean_absolute_error(y_train_full, y_pred_train)
mae_test = mean_absolute_error(y_test_final, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train_full, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test_final, y_pred_test))
r2_train = r2_score(y_train_full, y_pred_train)
r2_test = r2_score(y_test_final, y_pred_test)

print(f"\n📊 TRAINING SET:")
print(f"   MAE: {mae_train:.4f}")
print(f"   RMSE: {rmse_train:.4f}")
print(f"   R²: {r2_train:.4f}")

print(f"\n📊 TEST SET:")
print(f"   MAE: {mae_test:.4f}")
print(f"   RMSE: {rmse_test:.4f}")
print(f"   R²: {r2_test:.4f}")

# Diagnostic overfitting
overfitting_gap = mae_test - mae_train
print(f"\n🔍 OVERFITTING DIAGNOSTIC:")
print(f"   MAE Gap (Test - Train): {overfitting_gap:.4f}")
if overfitting_gap < 0.05:
    print(f"   Status: ✅ Excellent - No overfitting")
elif overfitting_gap < 0.15:
    print(f"   Status: ✅ Good - Low overfitting")
elif overfitting_gap < 0.3:
    print(f"   Status: ⚠️ Moderate overfitting")
else:
    print(f"   Status: ❌ High overfitting")

# ============================================================================
# 7. VISUALISATION DES PRÉDICTIONS
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot
ax1 = axes[0]
ax1.scatter(y_test_final, y_pred_test, alpha=0.5, s=30, color='#1f77b4', edgecolors='black', linewidth=0.5)
ax1.plot([y_test_final.min(), y_test_final.max()], [y_test_final.min(), y_test_final.max()], 
         'r--', linewidth=3, label='Perfect Prediction')
ax1.set_xlabel('True Ratings', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Ratings', fontsize=12, fontweight='bold')
ax1.set_title(f'Predictions vs Reality\nMAE: {mae_test:.3f}, R²: {r2_test:.3f}', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.4)

# Distribution des erreurs
ax2 = axes[1]
errors = y_test_final - y_pred_test
ax2.hist(errors, bins=50, color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=1)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=3, label='Zero Error')
ax2.set_xlabel('Prediction Error (True - Predicted)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title(f'Error Distribution\nMean Error: {errors.mean():.3f}', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.4, axis='y')

plt.tight_layout()
plt.savefig('mlp_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. EXEMPLES DE PRÉDICTIONS
# ============================================================================

print("\n" + "="*80)
print("SAMPLE PREDICTIONS")
print("="*80)

sample_indices = np.random.choice(len(y_test_final), 10, replace=False)
sample_titles = test_metadata.iloc[sample_indices]['title'].values

print(f"\n{'Title':<45} {'True':>8} {'Predicted':>10} {'Error':>8}")
print("-" * 75)
for idx, title in zip(sample_indices, sample_titles):
    true_val = y_test_final[idx]
    pred_val = y_pred_test[idx]
    error = true_val - pred_val
    title_short = title[:42] + '...' if len(title) > 45 else title
    print(f"{title_short:<45} {true_val:>8.2f} {pred_val:>10.2f} {error:>8.2f}")

# ============================================================================
# 9. EXPORT DU MODÈLE
# ============================================================================

print("\n" + "="*80)
print("EXPORTING MODEL")
print("="*80)

MODEL_DIR = os.path.join(BASE_DIR, "Models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Sauvegarder le modèle et le scaler
with open(os.path.join(MODEL_DIR, "mlp_model.pkl"), 'wb') as f:
    pickle.dump(model, f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), 'wb') as f:
    pickle.dump(scaler, f)

print(f"   Model saved: {os.path.join(MODEL_DIR, 'mlp_model.pkl')}")
print(f"   Scaler saved: {os.path.join(MODEL_DIR, 'scaler.pkl')}")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Test MAE: {mae_test:.4f} (average error of ±{mae_test:.2f} points)")
print(f"Test R²: {r2_test:.4f} ({100*r2_test:.1f}% of variance explained)")