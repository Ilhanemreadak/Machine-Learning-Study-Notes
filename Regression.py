import pandas as pd

# Datalarımızı okuyup değişkenlere atıyoruz.
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.columns) # Eğitim setimizin içindeki featuresleri görmek için sütünları yazdırıyoruz.

from sklearn.linear_model import LinearRegression

model = LinearRegression() # Linear Regression modelimizi oluşturuyoruz.

X_train = train.drop('SalePrice', axis=1) # Fiyatı bulacağımız için fiyatı girişimizden atıyoruz
y_train = train.loc[:,'SalePrice'] # Fiyatların bulunduğu SalePrice sütününü y çıktısına atıyoruz.

model.fit(X_train,y_train) # Train datalarımızı modelimize veriyoruz.

# Tekrardan test için aynı veri işlemlerini yapıyoruz.
X_test = test.drop('SalePrice', axis=1) 
y_test = test.loc[:,'SalePrice']

predictions = model.predict(X_test) # Train datasıyla eğitilmiş modelimizi daha önce görmediği test datasıyla tahminler oluşturuyor ve sonucu döndürüyor.

# Gerçek veriler ve modelimizin tahminleri arasındaki farkları dataylı inceleyebilmek için bir frame oluşturuyoruz ve başlıklar oluşturup bu başlıklara verileri atıyoruz.
comparison = pd.DataFrame({"Actual Values": y_test,"Predictions": predictions})

print(comparison.head())

from sklearn.metrics import mean_squared_error
from numpy import sqrt

"""
    MSE bir regresyon modelinin tahminlerinin gerçek değerlerden ne kadar uzak olduğunu ölçen bir performans ölçüsüdür.
    MSE, tahminler ile gerçek değerler arasındaki farkların karesinin ortalamasını ifade eder.
    RMSE ise MSE sonucunun karekökü alınmış halidir karekök alma nedenimiz bazen mse sonuçlar karşılaştırılamayacak kadar büyük olabilir bundan dolayı karekökünü alıyoruz.
"""

rmse = sqrt(mean_squared_error(y_test, predictions))  # Ortalama hatamızın hesabı için rmse hesabını yapıyoruz.
print(rmse)

print(train.corr()["SalePrice"].sort_values(ascending=False).head(10)) # Tahmin hesabı için kullanılan özelliklerin hesaplamalardaki etkisel katsayılarını gösteriyoruz

correlations = train.corr() # Tüm özellikler arasındaki korelasyon matrisini oluşturur. Her bir hücre, iki özellik arasındaki korelasyon katsayısını içerir.

print(correlations)

saleprice_correlations = correlations["SalePrice"] # Satış fiyatını ile diğer özellikler arasındaki korelasyonu atıyoruz ve yazdırıyoruz.
print(saleprice_correlations)

saleprice_correlations.sort_values(ascending=False).head(10) # Satış fiyatını ile diğer özellikler arasındaki büyükten küçüğe sıralıyoruz ve en etkili 10 özelliği ve yazdırıyoruz.