import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Read the File
file_path="/Users/dhairyasinghal/Desktop/Project/airbnb.csv"
df=pd.read_csv(file_path)

# Clean and convert numeric columns that may contain commas
numeric_columns = ['reviews', 'guests', 'beds', 'bathrooms', 'price', 'rating']
for col in numeric_columns:
    df[col] = df[col].astype(str).str.replace(",", "")  # remove commas
    df[col] = pd.to_numeric(df[col], errors='coerce')   # convert to numeric
    
#Display Basic Information
print(df.info())
print(df.head())

# # Convert 'rating' to numeric before filling missing values
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating'] = df['rating'].fillna(df['rating'].median())

# Fixing missing values in reviews and host_name
df['rating'].fillna(df['rating'].median(), inplace=True)
df['reviews'] = df['reviews'].fillna(0)
df['host_name'] = df['host_name'].fillna('Unknown')

# Drop unnecessary columns
df.drop(columns=['checkin', 'checkout'], inplace=True)

#Discriptive Statistics
print("\n----- Basic Info -----") # This prints a simple header to visually separate the output in the console.
print(df.info())

print("\n----- Summary Statistics -----") # Prints a header to mark the beginning of the summary statistics.
print(df.describe(include='all')) # This gives descriptive statistics for the entire DataFrame. It will try to describe all columns, numeric and non-numeric 

print("\n----- First 5 Rows -----") # Another header to separate this section.
print(df.head()) # Returns the first 5 rows of the DataFrame.

# Visualization:

# Rating distribution
plt.figure(figsize=(8,5)) # sets the figure size to 8 inches wide and 5 inches tall.
sns.histplot(df['rating'], bins=20, kde=True) # This is a Seaborn histogram plot. Uses the "rating" column from the DataFrame.  Divides the range of values into 20 bins (bars). Adds a Kernel Density Estimate (KDE) curve, which is a smooth line that shows the probability density of the distribution.
plt.title("Distribution of Ratings") # Adds a title to the plot.
plt.xlabel("Rating") # Labels the x-axis.
plt.ylabel("Count") # Labels the y-axis, indicating how many values fall into each bin.
plt.show()

# Price distribution (filtered for extreme values)
plt.figure(figsize=(8,5)) # This code visualizes the distribution of prices, but it filters out extremely high values (above 20,000) to focus on the more common range.
sns.histplot(df[df['price'] < 20000]['price'], bins=30, kde=True) # This filters the DataFrame df to only include rows where the price is less than 20,000.
plt.title("Price Distribution (Under 20,000)")
plt.xlabel("Price")
plt.ylabel("Listings Count")
plt.show()

# Top 10 countries by number of listings
plt.figure(figsize=(10,5)) # Creates a new plot with dimensions 10 inches wide by 5 inches tall.
top_countries = df['country'].value_counts().head(10)
sns.barplot(x=top_countries.index, y=top_countries.values)
plt.title("Top 10 Countries with Most Listings") # It helps you quickly identify which countries have the most listings, giving insight into geographic distribution.
plt.ylabel("Number of Listings")
plt.xticks(rotation=45)
plt.show()

# Top 10 hosts with most listings
print("\n----- Top 10 Hosts by Listings -----") # Takes the top 10 hosts with the highest listing counts.
top_hosts = df['host_name'].value_counts().head(10)
print(top_hosts)

# Correlation heatmap for numerical features
plt.figure(figsize=(8,6)) # Creates a new plot with a size of 8x6 inches — enough space to fit the heatmap clearly.
corr_cols = ['price', 'rating', 'reviews', 'beds', 'bathrooms', 'guests'] # Selects a subset of numerical columns from your DataFrame that you want to analyze for correlation.
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm') #  Output is a square matrix showing pairwise correlation values ranging from -1 to +1: +1 → Perfect positive correlation (both go up together), 0 → No linear correlation, -1 → Perfect negative correlation (one goes up, the other down)Draws the heatmap using Seaborn.
plt.title("Correlation Between Features")
plt.show()

# Average price by country (top 10)
print("\n----- Average Price by Country (Top 10) -----")
avg_price_by_country = df.groupby('country')['price'].mean().sort_values(ascending=False).head(10) # This gives insight into which countries tend to have more expensive listings on average. It can help with pricing strategy, targeting premium markets, or comparing affordability across regions.
print(avg_price_by_country)

# Top 10 Most Expensive Listings
print("\n----- Top 10 Most Expensive Listings -----")
expensive_listings = df[['name', 'price', 'country', 'host_name']].sort_values(by='price', ascending=False).head(10) # Displays the top 10 expensive listings with their name, price, country, and host. Returns the top 10 rows — i.e., the 10 most expensive listings.
print(expensive_listings)