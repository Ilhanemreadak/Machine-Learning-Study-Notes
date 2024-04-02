import pandas as pd

dataset = pd.read_csv("Veriler/breast-cancer.csv") # Verilerimizi csv'den okuyup datasetimize atıyoruz.

print(dataset.shape) # Datasetimizin şeklini yani kaça kaç olduğunu yazdırıyoruz.

print(dataset.head())   # Datsetimizdeki ilk 5 satırı yazdırıyoruz. 
print(dataset.tail())   # Datasetimizdeki son 5 satırı yazdırıyoruz.

from sklearn.preprocessing import LabelEncoder # LabelEncoder, kategorik (sınıflandırılabilir) verileri sayısal değerlere dönüştürmek için kullanılan bir araçtır.
labelencoder = LabelEncoder() # LabelEncoder nesnemizi oluşturuyoruz.

# LabelEncoder ile "diagnosis" yani teşhislerimizi sayısal değerlere çeviriyoruz. Örn: Kanser = 1 , Kanser Değil = 0
dataset["diagnosis"] = labelencoder.fit_transform(dataset["diagnosis"].values)

print(dataset.head()) # Datasetimizin yeni halini yazdırıyoruz.

from sklearn.model_selection import train_test_split 

# Train Test Split data setimizi "train" ve "test" olmak üzere iki bölüme ayırıyor.test_size parametresi modelimizi test etmek için datasetimizin ne kadarını teste ayırdığını belirliyor.
train, test = train_test_split(dataset, test_size=0.3) 

X_train = train.drop("diagnosis",axis=1) # Eğitim setimizden teşhisleri çıkarıyoruz çünkü hedefimiz teşhis etmek.
y_train = train.loc[:,"diagnosis"] # Teşhis başlığını ve altındaki değerleri yani tüm sütunu çıkış testimize atıyoruz.

# Test için aynı ayrımları yapıyoruz.
X_test = test.drop("diagnosis",axis=1) 
y_test = test.loc[:,"diagnosis"]

from sklearn.linear_model import LogisticRegression

"""
    LogisticRegression, sınıflandırma problemlerini çözmek için kullanılan bir makine öğrenimi modelidir.
    Adından da anlaşılacağı gibi, temelinde bir regresyon modeli yatar,
    ancak çıktı değerini bir logistik fonksiyon (sigmoid) kullanarak 0 ile 1 arasında bir olasılık değeri olarak dönüştürür.
    Bu nedenle, çoğunlukla ikili (binary) sınıflandırma problemlerinde kullanılır.
    Verilen bir girdiye dayanarak, bir verinin belirli bir sınıfa (etikete) ait olma olasılığını tahmin etmeye çalışır.
"""

model_1 = LogisticRegression() # Logistic Regression Modelimizi oluşturuyoruz.

model_1.fit(X_train,y_train) # Verimizi modelimize oturtuyoruz.

predictions = model_1.predict(X_test) # Modelimiz artık tahminlerini yapıyor ve sonuçları atıyoruz.

print(predictions) # Sonuçlarımızı yazdırıyoruz.

from sklearn.metrics import confusion_matrix

"""
    Confusion matrix (Karmaşıklık Matrisi), sınıflandırma modelinin performansını değerlendirmek için kullanılan bir metriktir. 
    Karmaşıklık matrisi, modelin gerçek ve tahmin edilen sınıflar arasındaki ilişkiyi görselleştirir.

    Karmaşıklık matrisi genellikle dört ana bileşeni içerir:

    True Positive (TP): Gerçek pozitif örneklerin sayısı. Yani, modelin doğru bir şekilde pozitif olarak tahmin ettiği örneklerin sayısı.
    False Positive (FP): Gerçek negatif olan ancak modelin yanlış bir şekilde pozitif olarak tahmin ettiği örneklerin sayısı.
    True Negative (TN): Gerçek negatif örneklerin sayısı. Yani, modelin doğru bir şekilde negatif olarak tahmin ettiği örneklerin sayısı.
    False Negative (FN): Gerçek pozitif olan ancak modelin yanlış bir şekilde negatif olarak tahmin ettiği örneklerin sayısı.

    Bu bileşenler, sınıflandırma modelinin performansını değerlendirmek için kullanılan farklı metriklerin hesaplanmasına olanak tanır,
    örneğin doğruluk (accuracy), hassasiyet (precision), duyarlılık (recall) ve F1 puanı gibi.
    Confusion matrix, modelin hangi sınıfları karıştırdığını ve hangi hataları yaptığını net bir şekilde gösterir,
    böylece modelin iyileştirilmesi için odaklanılması gereken alanları belirlemeye yardımcı olur.

"""
confusion_matrix(y_test, predictions)

from sklearn.metrics import classification_report

"""

    Classification report (Sınıflandırma Raporu), sınıflandırma modelinin performansını daha ayrıntılı olarak değerlendirmek için kullanılan bir metriktir. 
    Bu rapor, her bir sınıf için doğruluk, hassasiyet, duyarlılık ve F1 puanı gibi metrikleri sağlar.

    Bir sınıflandırma raporu genellikle şu bilgileri içerir:

    Precision (Hassasiyet): Modelin belirli bir sınıfı doğru şekilde tahmin etme kabiliyetidir. TP / (TP + FP) formülü ile hesaplanır.
    Recall (Duyarlılık): Gerçekten pozitif olan tüm pozitiflerin oranıdır. TP / (TP + FN) formülü ile hesaplanır.
    F1-Score: Hassasiyet ve duyarlılığın harmonik ortalamasıdır. 2 * (Precision * Recall) / (Precision + Recall) formülü ile hesaplanır.
    Support: Her sınıfın veri kümesinde ne kadar temsil edildiğini gösteren sayıdır.
    Accuracy (Doğruluk): Doğru sınıflandırılan örneklerin toplam örnekler üzerindeki oranıdır. (TP + TN) / (TP + TN + FP + FN) formülü ile hesaplanır.
    Macro avg: Her sınıfın metriklerinin ortalaması alınarak hesaplanan değerdir.
    Weighted avg: Her sınıfın metriklerinin ağırlıklı ortalaması alınarak hesaplanan değerdir, her sınıfın ağırlığı veri kümesindeki dağılıma göre belirlenir.
    Sample avg: Her sınıfın metriklerinin örnek başına ortalamasıdır.
    Classification report, her sınıfın performansını ayrıntılı olarak değerlendirir ve modelin farklı sınıflar arasındaki performans farklarını gösterir.
    Bu, modelin güçlü ve zayıf yönlerini belirlemek ve iyileştirmek için odaklanılacak alanları tanımlamak için kullanılır.

"""

print(classification_report(y_test, predictions))


from sklearn.svm import LinearSVC

"""
LinearSVC (Linear Support Vector Classifier), destek vektör makinesi (SVM) algoritmasının bir türüdür ve sınıflandırma problemlerini çözmek için kullanılır.
LinearSVC, doğrusal olarak ayrılabilir sınıflar arasında karar sınırlarını belirlemek için kullanılır.
Temel olarak, LinearSVC, veri noktalarını bir doğrusal hiper düzlemle sınıflandırır.
Bu hiper düzlem, iki sınıfı en iyi şekilde ayıran optimal bir şekilde yerleştirilmiş bir doğru veya düzlem olabilir.
Model, eğitim veri seti üzerinde bu hiper düzlemi bulmaya çalışır ve bu hiper düzlemi tanımlayan öznitelikleri ve sınırları belirler.
"""

model_2 = LinearSVC()   # LinearSVC Modelimizi oluşturuyoruz.

model_2.fit(X_train,y_train) # Verimizi modelimize oturtuyoruz.

predictions = model_2.predict(X_test) # Modelimiz artık tahminlerini yapıyor ve sonuçları atıyoruz.

print(predictions) # Sonuçlarımızı yazdırıyoruz.

# Tekrardan Logistic Regression modelimizdeki gibi LinearSVC modelimizin performans ölçümlerimizi yapıyoruz ve ikisini karşılaştırıyoruz.

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))