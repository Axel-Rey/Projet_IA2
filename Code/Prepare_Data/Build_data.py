# ============================================================
# PURPOSE OF THIS SCRIPT
# ============================================================
# This script prepares and cleans movie datasets to be used
# by machine learning models. It includes:
# - Loading raw data from CSVs
# - Cleaning and converting data types
# - Merging multiple sources into a single dataset
# - Feature engineering (popularity, counts, weighted rating)
# - Handling missing values
# - Encoding categorical features (genres)
# - Splitting into train/test sets
# - Exporting final datasets for later use
# 
# Note: No ML modeling is performed in this script.
# ============================================================


# ============================================================
# Imports
# ============================================================
print("Step 1: Importing libraries...")

import pandas as pd
import numpy as np
import ast   # to safely parse JSON-like strings in CSV
import os    # for file and path handling

print("   Libraries imported successfully.")


# ============================================================
# Resolve project paths
# ============================================================
print("\nStep 2: Resolving project paths...")

# Absolute path to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data folder is assumed to be ../../Data relative to this script
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Data"))

print(f"   Data directory: {DATA_DIR}")


# ============================================================
# Load raw CSV datasets
# ============================================================
print("\nStep 3: Loading CSV files...")

# Movie metadata (title, budget, popularity, etc.)
movies = pd.read_csv(os.path.join(DATA_DIR, "movies_metadata.csv"), low_memory=False)
print(f"   Movies dataset shape: {movies.shape}")

# Movie credits (cast and crew)
credits = pd.read_csv(os.path.join(DATA_DIR, "credits.csv"))
print(f"   Credits dataset shape: {credits.shape}")

# Movie keywords
keywords = pd.read_csv(os.path.join(DATA_DIR, "keywords.csv"))
print(f"   Keywords dataset shape: {keywords.shape}")


# ============================================================
# Select only relevant columns from movies dataset
# ============================================================
print("\nStep 4: Selecting relevant columns...")

# Only keep columns we will use in modeling or feature engineering
movies = movies[[
    "id", "title", "genres", "runtime", "budget",
    "vote_average", "vote_count", "popularity"
]]
print(f"   Movies dataset after column selection: {movies.shape}")


# ============================================================
# Convert types and clean basic data
# ============================================================
print("\nStep 5: Cleaning data and converting types...")

# Convert numeric fields from string to proper numeric types
movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
movies["budget"] = pd.to_numeric(movies["budget"], errors="coerce")
movies["runtime"] = pd.to_numeric(movies["runtime"], errors="coerce")

# Remove rows with missing id or title (cannot use these rows)
movies = movies.dropna(subset=["id", "title"])

# Ensure movie IDs are integers
movies["id"] = movies["id"].astype(int)

print(f"   Movies dataset after cleaning: {movies.shape}")


# ============================================================
# Helper function to parse JSON-like text columns
# ============================================================
def parse_json_list(json_str, key):
    """
    Safely parse a JSON-like string representing a list of dictionaries
    and extract values associated with a specific key.
    
    If parsing fails or the string is empty, returns an empty list.
    """
    try:
        data = ast.literal_eval(json_str)
        return [item[key] for item in data]
    except:
        return []


# ============================================================
# Parse list columns: genres, actors, crew, keywords
# ============================================================
print("\nStep 6: Parsing JSON-like columns...")

movies["genres"] = movies["genres"].apply(lambda x: parse_json_list(x, "name"))
credits["actors"] = credits["cast"].apply(lambda x: parse_json_list(x, "name"))
credits["crew"] = credits["crew"].apply(lambda x: parse_json_list(x, "name"))
keywords["keywords"] = keywords["keywords"].apply(lambda x: parse_json_list(x, "name"))

print("   JSON parsing completed.")


# ============================================================
# Merge datasets into a single dataframe
# ============================================================
print("\nStep 7: Merging datasets...")

# Merge movie metadata with credits
df = movies.merge(
    credits[["id", "actors", "crew"]],
    on="id",
    how="left"
)

# Merge with keywords
df = df.merge(
    keywords[["id", "keywords"]],
    on="id",
    how="left"
)

print(f"   Merged dataframe shape: {df.shape}")


# ============================================================
# Weighted rating feature
# ============================================================
print("\nStep 8: Computing weighted rating...")

# Weighted rating formula balances movies with few votes vs many votes
df["weighted_rating"] = (df["vote_average"] * df["vote_count"]) / (df["vote_count"] + 100)

print("   Weighted rating computed.")


# ============================================================
# Train/test split
# ============================================================
print("\nStep 9: Splitting into train and test sets...")

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print(f"   Train set shape: {train_df.shape}")
print(f"   Test set shape: {test_df.shape}")


# ============================================================
# Compute popularity features for actors and crew
# ============================================================
print("\nStep 10: Computing popularity features...")

from collections import Counter

# Count how many times each actor appears in TRAIN dataset
actor_popularity = Counter()
for actors in train_df["actors"]:
    actor_popularity.update(actors)

# Count crew appearances
crew_popularity = Counter()
for crew in train_df["crew"]:
    crew_popularity.update(crew)

# Function to extract popularity metrics
def popularity_features(names, popularity_dict):
    """
    For a list of names (actors or crew), compute:
    - mean popularity
    - max popularity
    - number of well-known people (appeared > 5 times)
    """
    if not isinstance(names, list) or len(names) == 0:
        return 0.0, 0.0, 0
    pops = [popularity_dict.get(name, 0) for name in names]
    return sum(pops)/len(pops), max(pops), sum(p > 5 for p in pops)

# Apply to actors
train_df[["actor_pop_mean", "actor_pop_max", "actor_pop_known"]] = train_df["actors"].apply(
    lambda x: pd.Series(popularity_features(x, actor_popularity))
)
test_df[["actor_pop_mean", "actor_pop_max", "actor_pop_known"]] = test_df["actors"].apply(
    lambda x: pd.Series(popularity_features(x, actor_popularity))
)

# Apply to crew
train_df[["crew_pop_mean", "crew_pop_max", "crew_pop_known"]] = train_df["crew"].apply(
    lambda x: pd.Series(popularity_features(x, crew_popularity))
)
test_df[["crew_pop_mean", "crew_pop_max", "crew_pop_known"]] = test_df["crew"].apply(
    lambda x: pd.Series(popularity_features(x, crew_popularity))
)

print("   Popularity features added.")


# ============================================================
# Handle missing values
# ============================================================
print("\nStep 11: Handling missing values...")

numeric_cols = [
    'budget', 'runtime', 'popularity', 'vote_count',
    'actor_pop_mean', 'actor_pop_max', 'actor_pop_known',
    'crew_pop_mean', 'crew_pop_max', 'crew_pop_known'
]

# Replace NaN in numeric columns with 0
for col in numeric_cols:
    train_df[col] = train_df[col].fillna(0)
    test_df[col] = test_df[col].fillna(0)

# Ensure at least one genre per movie
train_df['genres'] = train_df['genres'].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else ['Unknown'])
test_df['genres'] = test_df['genres'].apply(lambda x: x if isinstance(x, list) and len(x) > 0 else ['Unknown'])

print("   Missing values handled.")


# ============================================================
# Count-based features
# ============================================================
print("\nStep 12: Creating count-based features (number of actors, crew, genres, keywords)...")

for col, src in [
    ("num_actors", "actors"),
    ("num_crew", "crew"),
    ("num_genres", "genres"),
    ("num_keywords", "keywords")
]:
    train_df[col] = train_df[src].apply(lambda x: len(x) if isinstance(x, list) else 0)
    test_df[col] = test_df[src].apply(lambda x: len(x) if isinstance(x, list) else 0)

print("   Count-based features created.")


# ============================================================
# Encode genres with MultiLabelBinarizer
# ============================================================
print("\nStep 13: Encoding genres as multi-hot vectors...")

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
train_genres_encoded = mlb.fit_transform(train_df['genres'])
test_genres_encoded = mlb.transform(test_df['genres'])

print(f"   Number of genre features: {len(mlb.classes_)}")


# ============================================================
# Export cleaned datasets
# ============================================================
print("\nStep 14: Exporting processed datasets...")

DATASETS_EXPORT_DIR = os.path.join(DATA_DIR, "Datasets")
os.makedirs(DATASETS_EXPORT_DIR, exist_ok=True)

train_df.to_csv(os.path.join(DATASETS_EXPORT_DIR, "train_dataset.csv"), index=False)
test_df.to_csv(os.path.join(DATASETS_EXPORT_DIR, "test_dataset.csv"), index=False)

print("   Datasets successfully exported.")
print(f"   Train dataset: {os.path.join(DATASETS_EXPORT_DIR, 'train_dataset.csv')}")
print(f"   Test dataset:  {os.path.join(DATASETS_EXPORT_DIR, 'test_dataset.csv')}")
print("\nData preparation completed successfully.")
