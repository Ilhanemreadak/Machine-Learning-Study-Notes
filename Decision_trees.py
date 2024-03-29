import pandas as pd
import six
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Veriler/DecisionTrees_titanic.csv") # Datasetimizi atadık

print(dataset.head()) # Datasetimizden ilk 5 veriyi görüntülüyoruz 

X = dataset.drop("Survived", axis=1) # Tahmin edilecek veriyi sütününü giriş verilerimizden atıyoruz
y= dataset.loc[:, "Survived"] # Sonuçların bulunduğu sütunun içindeki verileri y değişkenine atıyoruz

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3) # Test ve Train datalarımızı bölüyoruz



dtmodel = DecisionTreeClassifier() # DecisionTree modelimizi oluşturuyoruz
dtmodel.fit(X_train, y_train) # Dataseti modele oturtuyoruz
dt_pred = dtmodel.predict(X_test) # Modelimize X test datasını vererek tahminde bulunduktan sonra sonuçları dt_pred'e atıyoruz.

print("Decision Trees Classification Report :")
print(classification_report(y_test, dt_pred)) # Classification Report ile modelimizin başarısını ölçüyoruz ve yazdırıyoruz.

rfmodel = RandomForestClassifier()  # RandomForest modelimizi oluşturuyoruz
rfmodel.fit(X_train, y_train) # Dataseti modele oturtuyoruz
rf_pred = rfmodel.predict(X_test) # Modelimize X test datasını vererek tahminde bulunduktan sonra sonuçları rf_pred'e atıyoruz.


print("Random Forest Classification Report :")
print(classification_report(y_test, rf_pred)) # Classification Report ile RandomForest modelimizin başarısını ölçüyoruz ve yazdırıyoruz.

# DecisionTree ve RandomForest modellerimizi yazdırdıktan sonra başarı oranlarını karşılaştırıyoruz.

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(10,10)) # Figure boyutumuzu oluşturuyoruz
plot_tree(decision_tree=dtmodel,max_depth=2,feature_names=X.columns,filled=True,impurity=True,rounded=True,precision=1) # DecisionTree modelimizi plotunu oluşturuyoruz.

plt.show() # Ağaç yapısını yazdırıyoruz.