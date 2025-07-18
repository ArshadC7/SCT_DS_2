# TITANIC DATA CLEANING & EXPLORATORY DATA ANALYSIS (EDA)

# 1. Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Styling for plots
sns.set_theme(context="notebook", palette="pastel")

# 2. Load dataset
df = pd.read_csv('train.csv')  # Ensure 'train.csv' is in the same folder
print("✅ Dataset Loaded. Shape:", df.shape)
print(df.head())

# 3. Dataset Summary
print("\n--- Dataset Info ---")
df.info()

print("\n--- Missing Value Percentages ---")
print(df.isna().mean().sort_values(ascending=False) * 100)

print("\n--- Statistical Summary ---")
print(df.describe(include='all').T)

# 4. Data Cleaning & Feature Engineering

# Fill missing Embarked with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Extract Deck from Cabin
df["Deck"] = df["Cabin"].astype(str).str[0].replace("n", "U")  # 'nan' becomes 'U' (Unknown)

# Fill Age using median by Sex & Pclass groups
df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(lambda x: x.fillna(x.median()))

# Feature: Family Size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Feature: Is Alone
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Feature: Title from Name
df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\s*\.", expand=False)
rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
df["Title"] = df["Title"].replace(rare_titles, "Rare")

# Optional: Log Fare
df["Fare_log"] = np.log1p(df["Fare"])

print("\n✅ Data Cleaning & Feature Engineering Completed.")

# 5. Exploratory Data Analysis (EDA)

# Overall Survival Count
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.xticks([0,1], ['Died', 'Survived'])
plt.show()

# Survival Rate by Sex
sns.barplot(x="Sex", y="Survived", data=df, ci=None)
plt.title("Survival Rate by Sex")
plt.show()

# Survival Rate by Pclass
sns.barplot(x="Pclass", y="Survived", data=df, ci=None)
plt.title("Survival Rate by Pclass")
plt.show()

# Age distribution vs survival
sns.kdeplot(data=df, x="Age", hue="Survived", common_norm=False, fill=True)
plt.title("Age Distribution: Survived vs Died")
plt.show()

# Fare vs Survival (Log Fare)
sns.boxplot(x="Survived", y="Fare_log", data=df)
plt.title("Log Fare vs Survival")
plt.show()

# Survival by Family Size
sns.barplot(x="FamilySize", y="Survived", data=df, ci=None)
plt.title("Survival Rate by Family Size")
plt.show()

# Survival by Title
sns.barplot(x="Title", y="Survived", data=df, ci=None)
plt.title("Survival Rate by Title")
plt.xticks(rotation=45)
plt.show()

# Deck vs Survival
deck_order = ['A','B','C','D','E','F','G','U']
sns.barplot(x="Deck", y="Survived", data=df, order=deck_order, ci=None)
plt.title("Survival Rate by Cabin Deck")
plt.show()

# Embarked vs Pclass vs Survival
pd.crosstab([df["Embarked"], df["Pclass"]], df["Survived"], normalize='index').plot(kind='bar', stacked=True)
plt.title("Survival by Embarked & Pclass")
plt.ylabel("Proportion within group")
plt.show()

# Correlation heatmap
num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
plt.figure(figsize=(10,6))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



