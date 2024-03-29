import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("Veriler/reg_train.csv") # Verilerimizi okuyup data değişkenine atıyoruz.

X = data.drop('SalePrice',axis=1)   # X datasından SalePrice sütununu çıkarıyoruz çünkü bu sonucumuz olacak ve x'e girişlerimizi veriyoruz.
y = data.loc[:,'SalePrice'] # SalesPrice featuresinin sütünunu içindeki bilgiler ile bir değişkene atıyoruz vektör.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # test ve train datalarımızı ayırıyoruz.

linear_reg = LinearRegression() # Linear Regression modelimizi oluşturuyoruz. 
ridge_reg = Ridge(alpha=0.05) # Ridge Regression modelimizi oluşturuyoruz ve Alpha değişkeni cezalandırma puanı olarak modelin karmaşıklığını azaltmak amacıyla kullanılmıştır.

scaler = MinMaxScaler() #scaler nesnesi yaratıyoruz bu nesneyi verilerimizi normalize etmek için kullanacağız.
normalized_data = scaler.fit_transform(data) # Verimizi max min değerlere göre inceleyerek normalize ediyoruz.

normalized_data_pd = pd.DataFrame(normalized_data) # Yeni verimizi frame'e oturtuyoruz

Xn = normalized_data_pd.drop('SalePrice',axis=1)    # Normalize edilmiş datadan SalePrice sütununu çıkarıyoruz çünkü bu sonucumuz olacak ve x'e girişlerimizi veriyoruz.
Yn = normalized_data_pd.loc[:,'SalePrice']  # Normalize edilmiş datadan SalePrice sütununu Y ye atıyoruz.

Xn_train, Xn_test, Yn_train, Yn_test = train_test_split(Xn, Yn, test_size = 0.3)    # test ve train datalarımızı ayırıyoruz.


linear_pred = linear_reg.predict(X_test) # Artık linear regression modelimize test datamızı veriyoruz ve çıkardığı tahminleri linear_pred değişkenine atıyoruz. 
ridge_pred = ridge_reg.predict(Xn_test) # Üsttekinden farklı olaraktan ridge regression modelimizde giriş olarak normalize edilmiş veriyi sunuyoruz.

"""
    MSE bir regresyon modelinin tahminlerinin gerçek değerlerden ne kadar uzak olduğunu ölçen bir performans ölçüsüdür.
    MSE, tahminler ile gerçek değerler arasındaki farkların karesinin ortalamasını ifade eder.
"""
#   Burada modellerin tahminsel hata sonuçlarını MSE ile hesaplayıp performans karşlaştırması yapacağız.
linear_mse = mean_squared_error(y_test, linear_pred)
ridge_mse = mean_squared_error(Yn_test, ridge_pred)

print(f"MSE without Ridge: {linear_mse}")
print(f"MSE with Ridge : {ridge_mse}")