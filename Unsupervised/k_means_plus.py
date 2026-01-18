import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeansPlusPlus:
        
    def fit(self,X):

        np.random.seed(42)
        self.centroids=[X[np.random.randint(len(X))]]

        for _ in range(1,self.k):
            distances=np.array([min([np.linalg.norm(x-c)**2
                                     for c in self.centroids])
                                     for x in X])
            probablities=distances/distances.sum()
            cumulative_probs=probablities.cumsum()

            r=np.random.rand()

            for idx,cum_prob in enumerate(cumulative_probs):
                if r < cum_prob:
                    self.centroids.append(X[idx])
                    break
        
        self.centroids=np.array(self.centroids)

        return super().fit(X) # continuse with standard K means
