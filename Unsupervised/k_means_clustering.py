import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class KMeans:

    def __init__(self,k=3,max_iters=100,tol=1e-4):
        self.k=k
        self.max_iters=max_iters
        self.tol=tol
        self.centroids=None
        self.labels=None

    def fit(self,X):

        np.random.seed(42)
        indices=np.random.choice(len(X),self.k,replace=False)#chose same number of centroids as k form len(X) datapoints
        self.centroids=X[indices]

        for iterations in range(self.max_iters):

            distances=np.sqrt(((X-self.centroids[:,np.newaxis])**2).sum(axis=2))
            self.labels=np.argmin(distances,axis=0)

            new_centroids=np.array([X[self.labels==i].mean(axis=0)
                                    for i in range(self.k)])
            
            if np.all(np.abs(new_centroids-self.centroids<self.tol)):
                break

            self.centroids=new_centroids

        return self
    

    def predict(self,X):
        distances=np.sqrt(((X-self.centroids[:,np.newaxis])**2).sum(axis=2))
        return np.argmin(distances,axis=0)
    
    def inertia(self,X):
        return sum([((X[self.labels==i]-self.centroids[i])**2).sum()
                   for i in range(self.k)])
    

X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)
kmeans = KMeans(k=4)
kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
            c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Clustering')
plt.show()
