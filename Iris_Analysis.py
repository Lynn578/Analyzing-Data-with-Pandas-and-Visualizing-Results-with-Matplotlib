# ==========================================
# Analyzing Iris Dataset with Pandas and Matplotlib
# ==========================================

# ---- Import Libraries ----
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ======================
# Task 1: Load and Explore the Dataset
# ======================

# Load Iris dataset from sklearn
iris = load_iris()

# Convert to pandas DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("✅ Iris dataset loaded successfully!")

# Display first few rows to inspect structure
print("\nFirst five rows of the dataset:")
print(df.head())

# ---- Explore Dataset ----
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# ---- Clean Dataset ----
# The Iris dataset has no missing values, but let's drop them if they exist.
df = df.dropna()
print("\n✅ Dataset cleaned. Any missing values?")
print(df.isnull().sum())

# ======================
# Task 2: Basic Data Analysis
# ======================

print("\nBasic Statistics of Numerical Columns:")
print(df.describe())

# Group by species and compute mean for each numerical feature
grouped = df.groupby("species").mean()
print("\nMean values per species:")
print(grouped)

# Observations (write your own when submitting):
# - Setosa has the smallest petal length and width.
# - Virginica has the largest measurements overall.
# - Versicolor lies between Setosa and Virginica.
# - Petal length is a very strong distinguishing feature.

# ======================
# Task 3: Data Visualization
# ======================

sns.set(style="whitegrid")  # Apply nice seaborn style

# ---- 1. Line Chart: Mean Petal Length per Species ----
plt.figure(figsize=(8,5))
grouped["petal length (cm)"].plot(marker="o")
plt.title("Mean Petal Length per Species (Trend View)")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# ---- 2. Bar Chart: Average Sepal Width per Species ----
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="sepal width (cm)", data=df, ci=None)
plt.title("Average Sepal Width per Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width (cm)")
plt.show()

# ---- 3. Histogram: Distribution of Petal Length ----
plt.figure(figsize=(8,5))
sns.histplot(df["petal length (cm)"], kde=True, bins=20)
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# ---- 4. Scatter Plot: Sepal Length vs Petal Length ----
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df)
plt.title("Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# ======================
# Final Observations
# ======================

print("""
FINAL OBSERVATIONS / FINDINGS:

1. Setosa is clearly distinct with very small petal length/width.
2. Virginica has the largest petal and sepal measurements.
3. Versicolor lies in between and overlaps slightly with Virginica.
4. The histogram shows a bimodal distribution of petal length (Setosa vs others).
5. Scatter plot shows distinct clustering, making Iris a great dataset for classification.
""")
