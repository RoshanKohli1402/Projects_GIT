import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("sales_data.csv")

plt.figure(figsize=(10, 6))
sns.lineplot(x='date', y='sales', data=data)
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='inventory_item', y='usage', data=data)
plt.title("Inventory Usage")
plt.xlabel("Inventory Item")
plt.ylabel("Usage")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
waste_data = data.groupby('item')['waste'].sum()
waste_data.plot(kind='pie', autopct='%1.1f%%')
plt.title("Waste Analysis by Item")
plt.ylabel('')
plt.show()

# Descriptive Statistics
print("Descriptive Statistics:\n", data.describe())

# Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# Seasonal Analysis - Average Sales by Day of Week
data['day_of_week'] = pd.to_datetime(data['date']).dt.day_name()
avg_sales_by_day = data.groupby('day_of_week')['sales'].mean().sort_values()

plt.figure(figsize=(10, 6))
sns.barplot(x=avg_sales_by_day.index, y=avg_sales_by_day.values)
plt.title("Average Sales by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Sales")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Feature Engineering (Replace with actual features)
data['day'] = pd.to_datetime(data['date']).dt.day
data['month'] = pd.to_datetime(data['date']).dt.month
X = data[['day', 'month', 'previous_day_sales']]  # replace with relevant features
y = data['sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")