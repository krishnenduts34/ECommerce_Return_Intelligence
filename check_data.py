import pandas as pd
# Load the dataset
file_path = "data/returns_sustainability_dataset.csv"
df = pd.read_csv(file_path)

# Display basic information
print("Dataset loaded successfully!")
print()

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nReturn Status values:")
print(df["Return_Status"].value_counts())

# Create the target variable
df["Return_Flag"] = df["Return_Status"].map({
    "Not Returned": 0,
    "Returned": 1
})

print("\nReturn_Flag values:")
print(df["Return_Flag"].value_counts())