import pandas as pd

dataset = pd.read_csv("Veriler/breast-cancer.csv") # Verilerimizi çekiyoruz.
print(dataset.head()) # Verilerimizi incelemek için verisetimizin ilk 5 elamanını yazıdırıyor.

from sklearn.preprocessing import LabelEncoder

# Hatırlatma : LabelEncoder, kategorik (sınıflandırılabilir) verileri sayısal değerlere dönüştürmek için kullanılan bir araçtır.

labelencoder = LabelEncoder() # LabelEncoder nesnemizi oluşturuyoruz.

# LabelEncoder ile "diagnosis" yani teşhislerimizi sayısal değerlere çeviriyoruz. Örn: Kanser = 1 , Kanser Değil = 0
dataset["diagnosis"] = labelencoder.fit_transform(dataset["diagnosis"].values)

X = dataset.drop("diagnosis", axis=1) # Eğitim setimizden teşhisleri çıkarıyoruz çünkü hedefimiz teşhis etmek.
y = dataset["diagnosis"] # Teşhis başlığını ve altındaki değerleri yani tüm sütunu çıkış testimize atıyoruz.

from sklearn.model_selection import train_test_split

"""

Bu satırda, veri seti X ve hedef değişken y arasındaki ilişkiyi koruyarak, 
X ve y verilerini eğitim ve test setlerine ayırmak için train_test_split fonksiyonu kullanılarak X_train, X_test, y_train, y_test değişkenlerine atama yapılmıştır.
Eğitim verisi, toplam veri setinin %30'u olarak belirlenmiştir.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)

from sklearn.naive_bayes import GaussianNB

"""
Bu satırda, veri seti X ve hedef değişken y arasındaki ilişkiyi koruyarak, 
X ve y verilerini eğitim ve test setlerine ayırmak için train_test_split fonksiyonu kullanılarak X_train, X_test, y_train, y_test değişkenlerine atama yapılmıştır. 
Eğitim verisi, toplam veri setinin %30'u olarak belirlenmiştir.
"""

model = GaussianNB() # GaussianNB Modelimizi oluşturuyoruz.
model.fit(X_train, y_train) # Verimizi modelimize oturtuyoruz.

predictions = model.predict(X_test)# Modelimiz artık tahminlerini yapıyor ve sonuçları atıyoruz.
print(predictions) # Sonuçlarımızı yazdırıyoruz.

# GaussianNB modelimizin performans ölçümlerimizi yapıyoruz.

from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, predictions)
print(conf_matrix)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))




 