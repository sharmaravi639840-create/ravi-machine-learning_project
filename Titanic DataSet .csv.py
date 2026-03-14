import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("titanic.csv")

# Select important features
df = df[['Survived','Pclass','Sex','Age','Fare']]

# Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Convert categorical data
df['Sex'] = df['Sex'].map({'male':0, 'female':1})

# Features and target
X = df[['Pclass','Sex','Age','Fare']]
y = df['Survived']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)