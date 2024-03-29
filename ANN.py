from sklearn.datasets import fetch_openml   # OpenML platformundan veri seti indirmek için kullanacağız. 
from sklearn.model_selection import train_test_split

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)    # Mnist 28x28 boyutlarında el yazısıyla yazılmış sayıları içeren pixel tablosu veri setidir.
# return_X_y parametresiyle veri kümesini X ve y olarak ayrı ayrı döndürüp atamamızı sağlar.
# as_frame parametresi if=TRUE return Pandas Dataframe, if=FALSE return Numpy dizisi
X = X / 255.0 # 255 Piksele sahip olduğumuz verileri normalize edebilmek için 255'e bölüyoruz.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)    #Test ve train datalarımızı ayırıyoruz ve atıyoruz.

import pandas as pd

data = pd.DataFrame(X) #Dataframe'e oturttuk
data.insert(784, "label", y) # 784. indise label başlığı altında y'deki verileri koyduk.

print(data.head()) # İlk 5 veriyi görüntüle

print(X_train[2]) # Train datasının 3. elemanını görüntüle

import matplotlib.pyplot as plt

"""for i in range(0,5): # Örnek olarak ilk 5 veriyi görselleştiriyoruz.
    plt.imshow(X[i].reshape((28,28)), cmap='gray')
    print(plt.show())"""
    
from sklearn.neural_network import MLPClassifier

"""
MLPClassifier, sklearn.neural_network modülünden gelir ve çok katmanlı algılayıcı (MLP) temelli sınıflandırma modelini uygular.
MLP, giriş katmanı, gizli katmanlar ve çıkış katmanı olmak üzere üç türden oluşur. Her katman, birbirine bağlı düğümlerden oluşur.
Her düğüm, giriş verilerini alır, ağırlıklı toplamını hesaplar ve bir aktivasyon fonksiyonuna (örneğin, sigmoid veya ReLU) uygular.
Gizli katmanlar, giriş katmanından ve çıkış katmanından olmak üzere en az bir tane olabilir.
"""

"""
hidden_layer_sizes: MLP'deki gizli katmanların ve her katmandaki düğüm sayısının belirlendiği bir parametredir.

activation: Gizli katmanlardaki düğümlerde kullanılacak aktivasyon fonksiyonunu belirler.
Örneğin, "logistic" sigmoid fonksiyonunu kullanırken, "relu" ReLU (Rectified Linear Unit) fonksiyonunu kullanır.


"""

mlp = MLPClassifier(hidden_layer_sizes=1, activation="logistic")
mlp1 = MLPClassifier(hidden_layer_sizes=100, activation="logistic")
mlp2 = MLPClassifier(hidden_layer_sizes=100, activation="logistic")

# Modellerimize datamızı oturtuyoruz.
mlp.fit(X_train, y_train)
mlp1.fit(X_train, y_train)
mlp2.fit(X_train, y_train)

#   Modellerimiz tahminlerini yapıyor.
NN_pred = mlp.predict(X_test)
print(NN_pred)

NN1_pred = mlp1.predict(X_test)
print(NN1_pred)

NN2_pred = mlp2.predict(X_test)
print(NN2_pred)

#Test amaçlı gerçek verimiz ile Mlp2 modelimizin sonuçlarını gözlemliyoruz.
print(f"Actual Value: {y_test[0]}")
print(f"Predicted Value: {NN2_pred[0]}")
# Ve sonucu görselleştiriyoruz.
plt.imshow(X_test[0].reshape((28,28)), cmap='gray')
print(plt.show())

#Test amaçlı gerçek verimiz ile tüm modellerimizin sonuçlarını gözlemleyip karşılaştırıyoruz.
print(f"Actual Value: {y_test[1]}")
print(f"Predicted Value For 1 Hidden Layer: {NN_pred[1]}")
print(f"Predicted Value For 100 Hidden Layer: {NN1_pred[1]}")
print(f"Predicted Value For 1000 Hidden Layer: {NN2_pred[1]}")

# Yine karşılaştırmamızın gerçek değerini görselleştiriyoruz.
plt.imshow(X_test[1].reshape((28, 28)), cmap='gray')
plt.show()


# Confusion Matrix ve Classification Reportumuz ile modellerin başarı oranlarını gözlemliyoruz.
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(confusion_matrix(y_test, NN_pred))
print(classification_report(y_test, NN_pred))

print(confusion_matrix(y_test, NN1_pred))
print(classification_report(y_test, NN1_pred))

print(confusion_matrix(y_test, NN1_pred))
print(classification_report(y_test, NN1_pred))




