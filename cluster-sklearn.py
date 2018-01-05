import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

#X = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])
centers = [[0,10],[10,10],[5,0]]
X,_ = make_blobs(n_samples = 200, centers = centers, cluster_std = 2)

# plt.scatter(X[:,0],X[:,1],s = 150)
# plt.show()
clf = KMeans(n_clusters = 3)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.","r.","c.","b.","k.","o."]

for i in range(len(X)):
	plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize = 10)
plt.scatter(centroids[:,0],centroids[:,1],marker = 'x', s=150, lineWidth = 5)
plt.show()
