# ============================================================================
# MOVIE DATASET PREPROCESSING
# ============================================================================
# Ce script prépare et nettoie les datasets de films pour être utilisés
# par les modèles de machine learning. Il inclut :
# - Chargement des données brutes depuis les CSVs
# - Nettoyage et conversion des types de données
# - Fusion de plusieurs sources en un dataset unique
# - Feature engineering (popularité, comptages, qualité acteurs/crew)
# - Gestion des valeurs manquantes
# - Encodage des features catégorielles (genres)
# - Split train/test
# - Export des datasets finaux pour utilisation ultérieure
#
# Note: Aucun modèle ML n'est entraîné dans ce script.
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from collections import Counter
import ast
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MOVIE DATASET PREPROCESSING")
print("="*80)

# ============================================================================
# 1. RÉSOLUTION DES CHEMINS
# ============================================================================

print("\nStep 1: Resolving project paths...")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(BASE_DIR, "Data")
EXPORT_DIR = os.path.join(DATA_DIR, "Processed")

os.makedirs(EXPORT_DIR, exist_ok=True)

print(f"   Data directory: {DATA_DIR}")
print(f"   Export directory: {EXPORT_DIR}")

# ============================================================================
# 2. CHARGEMENT DES DONNÉES BRUTES
# ============================================================================

print("\nStep 2: Loading raw CSV files...")

def parse_json_list(json_str, key):
    """Parse JSON strings and extract key values"""
    try:
        return [item[key] for item in ast.literal_eval(json_str)]
    except:
        return []

# Charger les datasets
movies = pd.read_csv(os.path.join(DATA_DIR, "movies_metadata.csv"), low_memory=False)
credits = pd.read_csv(os.path.join(DATA_DIR, "credits.csv"))
keywords = pd.read_csv(os.path.join(DATA_DIR, "keywords.csv"))

print(f"   Movies: {movies.shape}")
print(f"   Credits: {credits.shape}")
print(f"   Keywords: {keywords.shape}")

# ============================================================================
# 3. SÉLECTION ET NETTOYAGE DES COLONNES
# ============================================================================

print("\nStep 3: Selecting relevant columns and cleaning...")

# Sélectionner colonnes pertinentes
movies = movies[["id", "title", "genres", "runtime", "vote_average", "vote_count", 
                 "popularity", "original_language", "release_date",
                 "production_companies", "production_countries"]]

# Convertir types
for col in ["id", "runtime"]:
    movies[col] = pd.to_numeric(movies[col], errors="coerce")
movies = movies.dropna(subset=["id", "title"])
movies["id"] = movies["id"].astype(int)

print(f"   Movies after cleaning: {movies.shape}")

# ============================================================================
# 4. PARSING DES COLONNES JSON
# ============================================================================

print("\nStep 4: Parsing JSON-like columns...")

movies["genres"] = movies["genres"].apply(lambda x: parse_json_list(x, "name"))
credits["actors"] = credits["cast"].apply(lambda x: parse_json_list(x, "name"))
credits["crew"] = credits["crew"].apply(lambda x: parse_json_list(x, "name"))
keywords["keywords"] = keywords["keywords"].apply(lambda x: parse_json_list(x, "name"))
movies["production_companies"] = movies["production_companies"].apply(lambda x: parse_json_list(x, "name"))
movies["production_countries"] = movies["production_countries"].apply(lambda x: parse_json_list(x, "iso_3166_1"))

print("   JSON parsing completed")

# ============================================================================
# 5. FUSION DES DATASETS
# ============================================================================

print("\nStep 5: Merging datasets...")

df = movies.merge(credits[["id", "actors", "crew"]], on="id", how="left")
df = df.merge(keywords[["id", "keywords"]], on="id", how="left")

# Convertir et filtrer
for col in ["popularity", "vote_count", "vote_average"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df[(df['vote_average'] > 0.0) & (df['vote_count'] >= 10)].copy()

print(f"   Merged dataset: {df.shape}")
print(f"   After filtering (vote_count >= 10): {len(df)} movies")

# ============================================================================
# 6. FEATURE ENGINEERING - FEATURES TEMPORELLES
# ============================================================================

print("\nStep 6: Engineering temporal features...")

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['film_age'] = 2024 - df['release_year']
df['is_recent'] = (df['release_year'] >= 2010).astype(int)

print("   Temporal features created")

# ============================================================================
# 7. FEATURE ENGINEERING - FEATURES DE PRODUCTION
# ============================================================================

print("\nStep 7: Engineering production features...")

df['is_english'] = (df['original_language'] == 'en').astype(int)
df['num_production_companies'] = df['production_companies'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['num_production_countries'] = df['production_countries'].apply(lambda x: len(x) if isinstance(x, list) else 0)

print("   Production features created")

# ============================================================================
# 8. FEATURE ENGINEERING - FEATURES DE COMPTAGE
# ============================================================================

print("\nStep 8: Engineering counting features...")

df['num_actors'] = df['actors'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['num_crew'] = df['crew'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['num_genres'] = df['genres'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['num_keywords'] = df['keywords'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else ['Unknown'])

print("   Counting features created")

# ============================================================================
# 9. FEATURE ENGINEERING - TRANSFORMATIONS
# ============================================================================

print("\nStep 9: Engineering transformed features...")

df['log_popularity'] = np.log1p(df['popularity'])
df['log_vote_count'] = np.log1p(df['vote_count'])
df['log_runtime'] = np.log1p(df['runtime'].fillna(0))
df['sqrt_vote_count'] = np.sqrt(df['vote_count'])
df['popularity_per_vote'] = df['popularity'] / (df['vote_count'] + 1)
df['vote_confidence'] = df['sqrt_vote_count'] / (df['sqrt_vote_count'] + 10)

print("   Transformed features created")

# ============================================================================
# 10. TRAIN-TEST SPLIT
# ============================================================================

print("\nStep 10: Splitting into train and test sets...")

df['rating_quantile'] = pd.qcut(df['vote_average'], q=10, labels=False, duplicates='drop')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['rating_quantile'])
train_df = train_df.drop('rating_quantile', axis=1)
test_df = test_df.drop('rating_quantile', axis=1)

print(f"   Training set: {len(train_df)} movies")
print(f"   Test set: {len(test_df)} movies")

# ============================================================================
# 11. FEATURES DE POPULARITÉ ET QUALITÉ ACTEURS/CREW
# ============================================================================

print("\nStep 11: Computing actor/crew popularity and quality features...")

def calculate_popularity_quality(train_data):
    """Calculer popularité et qualité des acteurs/crew"""
    actor_pop = Counter()
    crew_pop = Counter()
    actor_rating = {}
    crew_rating = {}
    
    for _, row in train_data.iterrows():
        if isinstance(row['actors'], list):
            actor_pop.update(row['actors'])
            for actor in row['actors']:
                actor_rating.setdefault(actor, []).append(row['vote_average'])
        if isinstance(row['crew'], list):
            crew_pop.update(row['crew'])
            for person in row['crew']:
                crew_rating.setdefault(person, []).append(row['vote_average'])
    
    actor_avg = {k: np.mean(v) for k, v in actor_rating.items()}
    crew_avg = {k: np.mean(v) for k, v in crew_rating.items()}
    
    return actor_pop, crew_pop, actor_avg, crew_avg

def quality_features(names, pop_dict, qual_dict, default=6.0):
    """Extract quality metrics"""
    if not isinstance(names, list) or len(names) == 0:
        return 0.0, 0.0, 0, default, default
    pops = [pop_dict.get(name, 0) for name in names]
    quals = [qual_dict.get(name, default) for name in names]
    return (np.mean(pops), max(pops), sum(p > 5 for p in pops), np.mean(quals), min(quals))

actor_pop, crew_pop, actor_avg, crew_avg = calculate_popularity_quality(train_df)

for df_temp in [train_df, test_df]:
    result = df_temp["actors"].apply(lambda x: pd.Series(quality_features(x, actor_pop, actor_avg)))
    df_temp[["actor_pop_mean", "actor_pop_max", "actor_pop_known", "actor_quality_mean", "actor_quality_min"]] = result
    
    result = df_temp["crew"].apply(lambda x: pd.Series(quality_features(x, crew_pop, crew_avg)))
    df_temp[["crew_pop_mean", "crew_pop_max", "crew_pop_known", "crew_quality_mean", "crew_quality_min"]] = result
    
    df_temp['quality_signal'] = (df_temp['actor_quality_mean'] + df_temp['crew_quality_mean']) / 2

print("   Actor/crew quality features created")

# ============================================================================
# 12. ENCODAGE DES GENRES
# ============================================================================

print("\nStep 12: Encoding genres as multi-hot vectors...")

mlb = MultiLabelBinarizer()
train_genres_encoded = mlb.fit_transform(train_df['genres'])
test_genres_encoded = mlb.transform(test_df['genres'])

train_genres_df = pd.DataFrame(train_genres_encoded, columns=[f'genre_{g}' for g in mlb.classes_], index=train_df.index)
test_genres_df = pd.DataFrame(test_genres_encoded, columns=[f'genre_{g}' for g in mlb.classes_], index=test_df.index)

print(f"   Encoded {len(mlb.classes_)} unique genres")

# ============================================================================
# 13. GESTION DES VALEURS MANQUANTES
# ============================================================================

print("\nStep 13: Handling missing values...")

feature_cols = [
    'runtime', 'popularity', 'vote_count',
    'actor_pop_mean', 'actor_pop_max', 'actor_pop_known', 'actor_quality_mean', 'actor_quality_min',
    'crew_pop_mean', 'crew_pop_max', 'crew_pop_known', 'crew_quality_mean', 'crew_quality_min',
    'num_actors', 'num_crew', 'num_genres', 'num_keywords',
    'log_popularity', 'log_vote_count', 'log_runtime', 'sqrt_vote_count',
    'popularity_per_vote', 'vote_confidence', 'quality_signal',
    'release_year', 'release_month', 'film_age', 'is_recent',
    'is_english', 'num_production_companies', 'num_production_countries'
]

for col in feature_cols:
    train_df[col] = train_df[col].fillna(0)
    test_df[col] = test_df[col].fillna(0)

print("   Missing values handled")

# ============================================================================
# 14. PRÉPARATION DES MATRICES FINALES
# ============================================================================

print("\nStep 14: Preparing final feature matrices...")

X_train = pd.concat([train_df[feature_cols].reset_index(drop=True), train_genres_df.reset_index(drop=True)], axis=1)
X_test = pd.concat([test_df[feature_cols].reset_index(drop=True), test_genres_df.reset_index(drop=True)], axis=1)

y_train = train_df['vote_average'].values
y_test = test_df['vote_average'].values

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   y_test shape: {y_test.shape}")

# ============================================================================
# 15. STATISTIQUES DESCRIPTIVES
# ============================================================================

print("\n" + "="*80)
print("DATASET STATISTICS")
print("="*80)

print(f"\n📊 RATING DISTRIBUTION:")
print(f"   Mean: {y_train.mean():.2f}")
print(f"   Std: {y_train.std():.2f}")
print(f"   Min: {y_train.min():.2f}")
print(f"   Max: {y_train.max():.2f}")
print(f"   Median: {np.median(y_train):.2f}")

print(f"\n📊 FEATURE SUMMARY:")
print(f"   Total features: {X_train.shape[1]}")
print(f"   Numeric features: {len(feature_cols)}")
print(f"   Genre features: {len(mlb.classes_)}")

print(f"\n📊 GENRE DISTRIBUTION (Top 10):")
genre_counts = train_genres_df.sum().sort_values(ascending=False).head(10)
for genre, count in genre_counts.items():
    genre_name = genre.replace('genre_', '')
    percentage = (count / len(train_df)) * 100
    print(f"   {genre_name:<20} {count:>6} ({percentage:>5.1f}%)")

# ============================================================================
# 16. EXPORT DES DATASETS
# ============================================================================

print("\n" + "="*80)
print("EXPORTING PROCESSED DATASETS")
print("="*80)

# Export des DataFrames avec métadonnées
train_df.to_csv(os.path.join(EXPORT_DIR, "train_metadata.csv"), index=False)
test_df.to_csv(os.path.join(EXPORT_DIR, "test_metadata.csv"), index=False)

# Export des matrices de features
X_train.to_csv(os.path.join(EXPORT_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(EXPORT_DIR, "X_test.csv"), index=False)

# Export des labels
pd.DataFrame({'vote_average': y_train}).to_csv(os.path.join(EXPORT_DIR, "y_train.csv"), index=False)
pd.DataFrame({'vote_average': y_test}).to_csv(os.path.join(EXPORT_DIR, "y_test.csv"), index=False)

# Export du MultiLabelBinarizer pour réutilisation
import pickle
with open(os.path.join(EXPORT_DIR, "genre_encoder.pkl"), 'wb') as f:
    pickle.dump(mlb, f)

print("\n✓ Files exported:")
print(f"   - {os.path.join(EXPORT_DIR, 'train_metadata.csv')}")
print(f"   - {os.path.join(EXPORT_DIR, 'test_metadata.csv')}")
print(f"   - {os.path.join(EXPORT_DIR, 'X_train.csv')}")
print(f"   - {os.path.join(EXPORT_DIR, 'X_test.csv')}")
print(f"   - {os.path.join(EXPORT_DIR, 'y_train.csv')}")
print(f"   - {os.path.join(EXPORT_DIR, 'y_test.csv')}")
print(f"   - {os.path.join(EXPORT_DIR, 'genre_encoder.pkl')}")

print("\n" + "="*80)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("="*80)
print("\n💡 Next step: Run the ML training script with the processed data")