# Q2: K-Fold Cross Validation

import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

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

model = LogisticRegression()

kf = KFold(n_splits=3, shuffle=True)

scores = cross_val_score(model, X, y, cv=kf)

print("K-Fold Scores:", scores)
print("Average Accuracy:", scores.mean())
