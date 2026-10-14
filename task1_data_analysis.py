"""
Task 1: Data analysis with Pandas + Matplotlib
Usage:  python task1_data_analysis.py [path_to_csv]
If no CSV is given (or the file isn't found), a sample sales dataset is generated
so the script always runs. Replace with any Kaggle CSV (e.g. Titanic, Iris, sales).
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # saves figures to files; remove this line to pop up windows
import matplotlib.pyplot as plt


def make_sample_data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    region = rng.choice(["North", "South", "East", "West"], n)
    category = rng.choice(["Electronics", "Clothing", "Grocery"], n)
    ad_spend = rng.uniform(1000, 10000, n).round(2)
    customers = (ad_spend / 50 + rng.normal(0, 15, n)).round().clip(5)
    revenue = (ad_spend * 2.5 + customers * 30 + rng.normal(0, 2000, n)).round(2)
    discount = rng.uniform(0, 30, n).round(1)
    return pd.DataFrame({"Region": region, "Category": category, "AdSpend": ad_spend,
                         "Customers": customers, "Discount": discount, "Revenue": revenue})


# ---------- 1. Load ----------
path = sys.argv[1] if len(sys.argv) > 1 else None
if path and os.path.exists(path):
    df = pd.read_csv(path)
    print(f"Loaded {path}")
else:
    df = make_sample_data()
    df.to_csv("sample_sales.csv", index=False)
    print("No CSV supplied -> using generated sample data (saved as sample_sales.csv)")

# ---------- 2. Basic analysis ----------
print("\nShape:", df.shape)
print("\nFirst rows:\n", df.head())
print("\nInfo:")
df.info()
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()

# Pick the column to average (last numeric column by default - change as needed)
target = num_cols[-1]
print(f"\nAverage of '{target}': {df[target].mean():.2f}")
print(f"Median: {df[target].median():.2f} | Std dev: {df[target].std():.2f}")

# ---------- 3. Visualizations ----------
# Bar chart: average of target per category
if cat_cols:
    cat = cat_cols[0]
    grp = df.groupby(cat)[target].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(7, 4))
    grp.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title(f"Average {target} by {cat}")
    plt.ylabel(f"Average {target}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("bar_chart.png", dpi=150)
    plt.close()
    print(f"\nHighest average {target}: {grp.index[0]} ({grp.iloc[0]:.2f})")

# Scatter plot: two numeric columns
if len(num_cols) >= 2:
    x, y = num_cols[0], target
    plt.figure(figsize=(6, 4))
    plt.scatter(df[x], df[y], alpha=0.6, color="darkorange", edgecolor="k")
    plt.title(f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig("scatter_plot.png", dpi=150)
    plt.close()

# Heatmap: correlation matrix
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticklabels(num_cols)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=150)
    plt.close()

    # ---------- 4. Insights ----------
    c = corr[target].drop(target).sort_values(key=abs, ascending=False)
    print("\n=== INSIGHTS ===")
    print(f"- Strongest correlation with {target}: {c.index[0]} (r = {c.iloc[0]:.2f})")
    print(f"- Weakest correlation with {target}: {c.index[-1]} (r = {c.iloc[-1]:.2f})")
    print(f"- {target} ranges from {df[target].min():.2f} to {df[target].max():.2f}, mean {df[target].mean():.2f}")
    print(f"- Missing values in dataset: {int(df.isnull().sum().sum())}")

print("\nSaved: bar_chart.png, scatter_plot.png, heatmap.png")
