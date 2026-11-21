import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data.csv")


def extract_features(password):
    length = len(password)
    uppercase = sum(c.isupper() for c in password)
    lowercase = sum(c.islower() for c in password)
    digits = sum(c.isdigit() for c in password)
    special = sum(not c.isalnum() for c in password)

    return [length, uppercase, lowercase, digits, special]


X = df["password"].apply(extract_features)
X = pd.DataFrame(
    X.tolist(),
    columns=["length", "uppercase", "lowercase", "digits", "special"],
)

label = LabelEncoder()
Y = label.fit_transform(df["strength"])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=3, stratify=Y
)

model = RandomForestClassifier(n_estimators=100, random_state=3)
model.fit(X_train, Y_train)

print("Accuracy:", model.score(X_test, Y_test))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(label, open("label.pkl", "wb"))

print("Model trained and saved successfully!")
