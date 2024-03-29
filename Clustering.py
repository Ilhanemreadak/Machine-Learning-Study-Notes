import pandas as pd
dataset = pd.read_csv("Veriler/Live.csv")
dataset.head()

from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)

model.fit(dataset)

labels = model.predict(dataset) # bir modeli kullanarak veri noktaları üzerinde tahminler yapmak için kullanılır.

import numpy as np

np.unique(labels, return_counts=True) # Bu fonksiyon, bir dizi içindeki tekrar eden öğeleri kaldırır ve yalnızca benzersiz öğeleri tutar.
#   Tahmin edilen etiketlerin benzersiz sayısını ve her bir etiketin veri kümesinde kaç kez bulunduğunu döndürür.

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouettes = []

ks = list(range(2,15))

for n_cluster in ks:
    kmeans = KMeans(n_clusters=n_cluster).fit(dataset) # n_cluster kadar yani range(2,12) iterasyonuyla cluster sayısını arttırıyoruz.
    label = kmeans.labels_ # Her örneğin atanmış küme etiketlerini alır.
    sil_coeff = silhouette_score(dataset, label, metric='euclidean') #  Silhouette skoru hesaplanır. Bu, kümeleme sonuçlarının ne kadar homojen olduğunu değerlendirir.
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    silhouettes.append(sil_coeff)
    
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.scatter(ks, silhouettes, marker='x', c='r') # Küme sayısına göre Silhouette skorlarını scatter plot olarak çizer.
plt.plot(ks, silhouettes) # Küme sayısına göre Silhouette skorlarını çizgi grafiği olarak çizer.
plt.xlabel('k')
plt.ylabel('Silhouette score')
plt.show()

"""
    Silhouette Skoru bize kaç cluster kullanmamız gerektiğini gösterir.
    For döngüsünde 2den 12 ye kadar tek tek bu cluster sayılarını denedik.
    En iyi sonucun 3 cluster ile aldık yani bu sınıflandırmada 3 sınıf oluşmuş olduğunu anladık.
    
"""

