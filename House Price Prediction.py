import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create a dummy DataFrame with sample house prices data
dummy_data = {'size_squft': [1000, 1200, 1500, 1800, 2000, 2500, 3000],
              'price_lakh': [30, 35, 45, 55, 60, 75, 90]}
dummy_df = pd.DataFrame(dummy_data)

# Save the dummy DataFrame to a CSV file named 'house_prices.csv'
dummy_df.to_csv('house_prices.csv', index=False)

print("Dummy 'house_prices.csv' created successfully.")
df = pd.read_csv("house_prices.csv")

x = df[['size_squft']]
y = df['price_lakh']

model = LinearRegression()
model.fit(x, y)

predicted_price = model.predict([[2200]])

print("Predicted House Price for 2200 sqft house:", predicted_price[0])


plt.scatter(x, y)
plt.plot(x, model.predict(x))
plt.xlabel('Size (sq ft)')
plt.ylabel('Price (lakhs)')
plt.title("House Price Prediction using Linear Regression")

plt.show()