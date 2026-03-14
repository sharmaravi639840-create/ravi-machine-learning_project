from pandas.tseries.offsets import Day
from sklearn. linear_model import LinearRegression

Day = [[30], [28], [22], [25], [27]]

sales = [10, 152, 200, 20, 30]

model = LinearRegression()

model.fit (temperature, sales)

new_temp = [[28]] #28 °C

predicted_sales = model.predict(new_temp)

print("Temperature:", new_temp[0][0], "°C")
print("Predicted Ice-cream Sales:", predicted_sales[0])