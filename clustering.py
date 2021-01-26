import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X1 = pd.read_pickle('MeasurmentsALL.pkl')


'''scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA(n_components = 10).fit(X)
print(pca.components_)
print(pca.explained_variance_)'''

k = 4 # mid, ind, thumb, none
kmeans = KMeans(n_clusters = k)
kmeans.fit(X1)
predictions = kmeans.predict(X1)

# Plotting
fig1 = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig1, elev=30, azim=150)
ax.scatter(X1.iloc[:, 3], X1.iloc[:, 7], X1.iloc[:, 11], c = predictions, edgecolor="k", s=50)
ax.set_xlabel("flex_ind2_max")
ax.set_ylabel("flex_mid2_max")
ax.set_zlabel("flex_thumb2_max")
plt.title("K Means", fontsize=14)
plt.show()


