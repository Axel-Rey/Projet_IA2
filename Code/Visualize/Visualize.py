# ============================================================
# PURPOSE OF THIS SCRIPT
# ============================================================
# This script performs visualization on processed movie datasets.
# It covers:
# - Bias / missing-information visualization (post-preprocessing)
# - Distribution analysis: genres, votes, budget, runtime
# - Popularity analysis: actors and crew
# - Correlation analysis between numeric features
# - Identification of top movies, actors, and crew members
#
# All plots are displayed interactively and saved into a single PDF.
# ============================================================


# ============================================================
# Step 1: Imports
# ============================================================
print("Step 1: Importing libraries...")

import pandas as pd # for data manipulation
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for enhanced plotting
import os # for file path management
from collections import Counter # for counting occurrences
import ast # for safely evaluating strings as Python literals
from matplotlib.backends.backend_pdf import PdfPages # for saving plots to PDF

print("   Libraries imported successfully.")


# ============================================================
# Step 2: Load processed datasets
# ============================================================
print("\nStep 2: Loading processed datasets...")

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(script_dir, "../../Data/Datasets"))

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_dataset.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_dataset.csv"))

print(f"   Train shape: {train_df.shape}")
print(f"   Test shape : {test_df.shape}")


# ============================================================
# Step 3: Prepare PDF
# ============================================================
print("\nStep 3: Preparing PDF...")

pdf_path = os.path.join(script_dir, "movie_visualizations.pdf")
pdf = PdfPages(pdf_path)

print("   PDF path:", pdf_path)


# ============================================================
# Utility: show and save plots
# ============================================================
def show_and_save(fig, title=None):
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    pdf.savefig(fig)
    plt.close(fig)


# ============================================================
# Step 4: Bias & missing-information visualization
# ============================================================
print("\nStep 4: Visualizing biases and missing information...")

bias_data = {
    "Unknown genre": (train_df["genres"].apply(lambda x: ast.literal_eval(x) == ["Unknown"]).mean()),
    "No actors listed": (train_df["actors"].apply(lambda x: len(ast.literal_eval(x)) == 0).mean()),
    "No crew listed": (train_df["crew"].apply(lambda x: len(ast.literal_eval(x)) == 0).mean()),
    "No keywords listed": (train_df["keywords"].apply(lambda x: len(ast.literal_eval(x)) == 0).mean()),
    "Budget = 0": (train_df["budget"] == 0).mean(),
    "Runtime = 0": (train_df["runtime"] == 0).mean(),
}

bias_df = pd.DataFrame.from_dict(
    bias_data, orient="index", columns=["proportion"]
).sort_values("proportion", ascending=False)

fig = plt.figure(figsize=(10, 6))
sns.barplot(x=bias_df["proportion"], y=bias_df.index)
plt.xlabel("Proportion of movies")
plt.ylabel("Bias / Missing information type")
show_and_save(fig, "Biases and Missing Information in Dataset")


# ============================================================
# Step 5: Genres distribution
# ============================================================
print("\nStep 5: Plotting genres distribution...")

all_genres = Counter(
    g for sub in train_df["genres"] for g in ast.literal_eval(sub)
)
n_movies = len(train_df)

genres_df = (
    pd.DataFrame.from_dict(all_genres, orient="index", columns=["count"])
    .sort_values("count", ascending=False)
)
genres_df["percentage"] = genres_df["count"] / n_movies * 100

# Keep top 19 + Unknown
if "Unknown" in genres_df.index:
    top_genres_df = pd.concat([
        genres_df.head(19),
        genres_df.loc[["Unknown"]]
    ]).drop_duplicates()
else:
    top_genres_df = genres_df.head(20)

colors = [
    "red" if genre == "Unknown" else "steelblue"
    for genre in top_genres_df.index
]

fig = plt.figure(figsize=(10, 6))
plt.barh(
    top_genres_df.index,
    top_genres_df["percentage"],
    color=colors
)
plt.xlabel("Percentage of movies (%)")
plt.ylabel("Genre")
plt.gca().invert_yaxis()
show_and_save(fig, "Genre Distribution")


# ============================================================
# Step 6: Vote distributions
# ============================================================
print("\nStep 6: Vote distributions...")

fig = plt.figure(figsize=(8, 5))
sns.histplot(train_df["vote_average"], bins=20, kde=True)
plt.xlabel("Vote Average")
plt.ylabel("Count")
show_and_save(fig, "Distribution of Vote Average")

fig = plt.figure(figsize=(8, 5))
sns.histplot(train_df["weighted_rating"], bins=20, kde=True)
plt.xlabel("Weighted Rating")
plt.ylabel("Count")
show_and_save(fig, "Distribution of Weighted Rating")


# ============================================================
# Step 7: Budget and runtime
# ============================================================
print("\nStep 7: Budget and runtime distributions...")

fig = plt.figure(figsize=(8, 5))
sns.histplot(train_df["budget"], bins=30, kde=True)
plt.xlabel("Budget")
plt.ylabel("Count")
show_and_save(fig, "Distribution of Movie Budgets")

fig = plt.figure(figsize=(8, 5))
sns.histplot(
    train_df["runtime"][train_df["runtime"] <= 300],
    bins=30,
    kde=True
)
plt.xlabel("Runtime (minutes)")
plt.ylabel("Count")
plt.xlim(0, 300)
show_and_save(fig, "Distribution of Movie Runtime")



# ============================================================
# Step 8: Actor and crew popularity
# ============================================================
print("\nStep 8: Actor and crew popularity...")

fig = plt.figure(figsize=(8, 5))
sns.histplot(train_df["actor_pop_mean"], bins=20, kde=True)
plt.xlabel("Mean Actor Popularity")
plt.ylabel("Count")
show_and_save(fig, "Distribution of Mean Actor Popularity")

fig = plt.figure(figsize=(8, 5))
sns.histplot(train_df["crew_pop_mean"], bins=20, kde=True)
plt.xlabel("Mean Crew Popularity")
plt.ylabel("Count")
show_and_save(fig, "Distribution of Mean Crew Popularity")


# ============================================================
# Step 9: Correlation heatmap
# ============================================================
print("\nStep 9: Correlation analysis...")

numeric_cols = [
    "budget", "runtime", "popularity", "vote_count",
    "vote_average", "weighted_rating",
    "actor_pop_mean", "crew_pop_mean"
]

fig = plt.figure(figsize=(10, 8))
sns.heatmap(
    train_df[numeric_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
show_and_save(fig, "Correlation Between Numeric Features")


# ============================================================
# Step 10: Top movies
# ============================================================
print("\nStep 10: Top popular movies...")

top_movies = train_df.sort_values("popularity", ascending=False).head(20)

fig = plt.figure(figsize=(10, 8))
sns.barplot(x="popularity", y="title", data=top_movies)
plt.xlabel("Popularity")
plt.ylabel("Movie title")
show_and_save(fig, "Top 20 Most Popular Movies")


# ============================================================
# Step 11: Top actors and crew
# ============================================================
print("\nStep 11: Top actors and crew...")

all_actors = Counter(
    actor for sub in train_df["actors"] for actor in ast.literal_eval(sub)
)
actors, counts = zip(*all_actors.most_common(20))

fig = plt.figure(figsize=(10, 8))
sns.barplot(x=list(counts), y=list(actors))
plt.xlabel("Number of appearances")
plt.ylabel("Actor")
show_and_save(fig, "Top 20 Most Frequent Actors")

all_crew = Counter(
    member for sub in train_df["crew"] for member in ast.literal_eval(sub)
)
crew, counts = zip(*all_crew.most_common(20))

fig = plt.figure(figsize=(10, 8))
sns.barplot(x=list(counts), y=list(crew))
plt.xlabel("Number of appearances")
plt.ylabel("Crew member")
show_and_save(fig, "Top 20 Most Frequent Crew Members")


# ============================================================
# Step 12: Close PDF
# ============================================================
pdf.close()

print("\nAll visualizations completed successfully.")
print("PDF saved at:", pdf_path)
