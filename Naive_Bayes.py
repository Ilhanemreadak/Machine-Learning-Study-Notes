import pandas as pd

dataset = pd.read_csv("Veriler/breast-cancer.csv")
print(dataset.head())

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset["diagnosis"] = labelencoder.fit_transform(dataset["diagnosis"].values)

X = dataset.drop("diagnosis", axis=1)
y = dataset["diagnosis"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, predictions)
print(conf_matrix)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))




 