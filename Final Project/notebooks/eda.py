import pandas as pd

# -------------------------------
# STEP 1: Load the dataset
# -------------------------------
df = pd.read_csv("dataset/ai4i2020.csv")

# -------------------------------
# STEP 2: Display first 15 rows
# -------------------------------
print("========== FIRST 15 ROWS ==========")
print(df.head(15))

# -------------------------------
# STEP 3: Display dataset size
# -------------------------------
print("\n========== DATASET SHAPE ==========")
print(df.shape)

# -------------------------------
# STEP 4: Display column names
# -------------------------------
print("\n========== COLUMN NAMES ==========")
print(df.columns)

# -------------------------------
# STEP 5: Check missing values
# -------------------------------
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# -------------------------------
# STEP 6: Check duplicate rows
# -------------------------------
print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

# -------------------------------
# STEP 7: Dataset Information
# -------------------------------
print("\n========== DATASET INFORMATION ==========")
df.info()