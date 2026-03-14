import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset (Hours studied vs Marks)
data = {
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Marks': [10,20,35,40,50,55,65,70,85,95]
}

df = pd.DataFrame(data)

# Feature and target
X = df[['Hours']]
y = df['Marks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict marks
predicted = model.predict([[7]])

print("Predicted Marks for 7 hours study:", predicted[0])

# Graph
plt.scatter(X,y,color='blue')
plt.plot(X,model.predict(X),color='red')
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.show()