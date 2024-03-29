import pandas as pd
dataset = pd.read_csv("Veriler/breast-cancer.csv")

dataset.shape

dataset.head()

dataset.tail()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset["diagnosis"] = labelencoder.fit_transform(dataset["diagnosis"].values)

dataset.head()

dataset.tail()

from sklearn.model_selection import train_test_split

train, test = train_test_split(dataset, test_size=0.3)

X_train = train.drop("diagnosis",axis=1)
y_train = train.loc[:,"diagnosis"]

X_test = test.drop("diagnosis",axis=1)
y_test = test.loc[:,"diagnosis"]

from sklearn.linear_model import LogisticRegression

model_1 = LogisticRegression()

model_1.fit(X_train,y_train)

predictions = model_1.predict(X_test)
predictions

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

from sklearn.svm import LinearSVC

model_2 = LinearSVC()

model_2.fit(X_train,y_train)

predictions = model_2.predict(X_test)
predictions

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))