# Q1: Spam Detection using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = {
    'text': ['Win money now', 'Hello friend', 'Free prize', 'Meeting tomorrow', 'Claim reward'],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam']
}

df = pd.DataFrame(data)

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

df['length'] = df['text'].apply(len)

X = df[['length']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
