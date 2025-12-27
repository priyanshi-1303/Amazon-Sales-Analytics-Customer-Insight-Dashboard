import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/amazon.csv")


# Basic info
print(df.shape)
print(df.head())

# Clean price columns
price_cols = ['discounted_price', 'actual_price']

for col in price_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        .astype(float)
    )

print(df[price_cols].head())

# Clean rating column
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Clean discount percentage column
df['discount_percentage'] = (
    df['discount_percentage']
    .astype(str)
    .str.replace('%', '', regex=False)
    .astype(float)
)

print(df[['rating', 'discount_percentage']].head())

# Check missing values
print(df.isnull().sum())

# Drop duplicate rows
df = df.drop_duplicates()

# Drop rows where key columns are missing
df = df.dropna(subset=['product_id', 'product_name', 'rating', 'actual_price', 'discounted_price'])

print("After cleaning shape:", df.shape)

# Feature engineering
df['discount_amount'] = df['actual_price'] - df['discounted_price']

df['price_drop_pct'] = (
    df['discount_amount'] / df['actual_price'] * 100
)

print(df[['actual_price', 'discounted_price', 'discount_amount', 'price_drop_pct']].head())


# Category-wise analysis
category_summary = (
    df.groupby('category')
      .agg(
          avg_rating=('rating', 'mean'),
          avg_discount_pct=('price_drop_pct', 'mean'),
          product_count=('product_id', 'count')
      )
      .reset_index()
      .sort_values(by='product_count', ascending=False)
)

print(category_summary.head())


# Save cleaned dataset for dashboard
df.to_csv("data/amazon_cleaned.csv", index=False)

print("amazon_cleaned.csv saved successfully")

