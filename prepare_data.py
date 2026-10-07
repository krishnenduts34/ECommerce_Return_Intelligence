import pandas as pd

# Load the original dataset
file_path = "data/returns_sustainability_dataset.csv"
df = pd.read_csv(file_path)

# Create the target variable
df["Return_Flag"] = df["Return_Status"].map({
    "Not Returned": 0,
    "Returned": 1
})

# Convert Order_Date to datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Create useful date features
df["Order_Year"] = df["Order_Date"].dt.year
df["Order_Month"] = df["Order_Date"].dt.month
df["Order_DayOfWeek"] = df["Order_Date"].dt.dayofweek

# Remove columns that should NOT be used for prediction
# These contain post-return information or unique identifiers.
columns_to_remove = [
    "Order_ID",
    "Product_ID",
    "User_ID",
    "Order_Date",
    "Return_Status",
    "Return_Reason",
    "Days_to_Return",
    "Return_Cost",
    "Profit_Loss",
    "CO2_Saved",
    "Waste_Avoided"
]

df_ml = df.drop(columns=columns_to_remove, errors="ignore")

# Save the prepared dataset
output_path = "data/prepared_data.csv"
df_ml.to_csv(output_path, index=False)

# Display the result
print("ML dataset prepared successfully!")
print()

print("Number of rows:", df_ml.shape[0])
print("Number of columns:", df_ml.shape[1])

print("\nFeatures used for machine learning:")
print(df_ml.columns.tolist())

print("\nTarget distribution:")
print(df_ml["Return_Flag"].value_counts())

print("\nPrepared dataset saved to:")
print(output_path)