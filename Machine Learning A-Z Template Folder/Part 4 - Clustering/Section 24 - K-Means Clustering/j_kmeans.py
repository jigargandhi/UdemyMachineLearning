#Kmeans gives us a new way of finding clusters in the data
#To find the ideal number of clusters calculate  Within Cluster Sum of Squared
# for different clusters and plot it. Then use your judgement/ elbow rule to find 
#the right amount of clusters. 
# How to reduce the effect of outliers in the Kmeans?

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init ='k-means++', random_state=0)
    kmeans.fit(X)
    # intertia_ is another name for WCSS
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('NClusters')
plt.xlabel('WCSS')
plt.show()

# applying kmeans
kmeans = KMeans(n_clusters = 5, init= 'k-means++', random_state=0)
# y_kmeans is a vector of clusters to which each point belong
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans==0, 0],X[y_kmeans==0, 1], s = 100, color='red', label ='Careful')
plt.scatter(X[y_kmeans==1, 0],X[y_kmeans==1, 1], s = 100, color='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0],X[y_kmeans==2, 1], s = 100, color='green', label ='Cluster 3')
plt.scatter(X[y_kmeans==3, 0],X[y_kmeans==3, 1], s = 100, color='cyan', label ='Cluster 4')
plt.scatter(X[y_kmeans==4, 0],X[y_kmeans==4, 1], s = 100, color='magenta', label ='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s= 300, color='yellow', label='centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()