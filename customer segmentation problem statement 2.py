import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Datasets
vehicle_data_url = "https://www.kaggle.com/datasets/dataclusterlabs/indian-vehicle-dataset"
cab_booking_data_url = "https://www.kaggle.com/datasets/datahackers/cab-booking-stats"

# Read the datasets
vehicle_df = pd.read_csv(vehicle_data_url)
cab_df = pd.read_csv(cab_booking_data_url)

# Preprocessing Vehicle Dataset
vehicle_df.dropna(inplace=True)  
vehicle_df['Price_per_km'] = vehicle_df['Price_per_km'].astype(float)

# Preprocessing Cab Booking Dataset
cab_df.dropna(inplace=True)  
cab_df['Booking_Count'] = cab_df['Booking_Count'].astype(int)

# Visualization 1: Price Distribution by Vehicle Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Vehicle_Type', y='Price_per_km', data=vehicle_df)
plt.title("Price Distribution by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Price per km (₹)")
plt.xticks(rotation=45)
plt.show()

# Visualization 2: Cab Booking Trends Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x='Month', y='Booking_Count', hue='City', data=cab_df, marker='o')
plt.title("Cab Booking Trends Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Bookings")
plt.legend(title="City")
plt.grid(True)
plt.show()

# Visualization 3: Correlation Heatmap (Vehicle Dataset)
plt.figure(figsize=(10, 8))
sns.heatmap(vehicle_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Vehicle Data")
plt.show()

# Visualization 4: Top 10 Cities by Cab Bookings
top_cities = cab_df.groupby('City')['Booking_Count'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
top_cities.plot(kind='bar', color='skyblue')
plt.title("Top 10 Cities by Cab Bookings")
plt.xlabel("City")
plt.ylabel("Total Bookings")
plt.xticks(rotation=45)
plt.show()

# Save processed datasets for future use
vehicle_df.to_csv("processed_vehicle_data.csv", index=False)
cab_df.to_csv("processed_cab_booking_data.csv", index=False)