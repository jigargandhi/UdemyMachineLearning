# -*- coding: utf-8 -*-
#agglomerative bottom up approach
#divisive = top down approach
# Agglomerative
# Step 1 - Each Point as 1 cluster
# Step 2 - Each closest point as 1 cluster
# Step 3 - Each two cluster combine
# Step 4 Repeat step 3 till only one cluster is left

# Closesness of clusters can be measured by 
# a. Euclidean distance of centroids
# b. Closest Point
# c. Farthest Point
# d. Avergate distance between all the points

# HC holds memory in dendogram

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values


# using dendogram to find optimal number of cluster
#ward is a method of distance
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()


#fitting hierarchical clustering to the mall dataset

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 5, affinity='euclidean', linkage ='ward')
y_hc = cluster.fit_predict(X)

# Only 2d visualization can be done here

plt.scatter(X[y_hc==0, 0],X[y_hc==0, 1], s = 100, color='red', label ='Careful')
plt.scatter(X[y_hc==1, 0],X[y_hc==1, 1], s = 100, color='blue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0],X[y_hc==2, 1], s = 100, color='green', label ='Cluster 3')
plt.scatter(X[y_hc==3, 0],X[y_hc==3, 1], s = 100, color='cyan', label ='Cluster 4')
plt.scatter(X[y_hc==4, 0],X[y_hc==4, 1], s = 100, color='magenta', label ='Cluster 5')
#plt.scatter(cluster..cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s= 300, color='yellow', label='centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()


