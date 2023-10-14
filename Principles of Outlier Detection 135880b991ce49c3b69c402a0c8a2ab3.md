# Principles of Outlier Detection

# Outlier Detection:

An outlier is a sample(observation) that differs from other in the same dataset, they fall outside the expected distribution or pattern of our our data, they can be caused by human error, experimental error,…

Outlier detection the process of detecting these anommalies either to discard them from our data or to flag them to the appropriate person

# Impact of outliers:

if left undetected outliers can cause problems like:

- they can signficantly change values of stastical measuments like mean and standard deviation
- they have a huge impact on model performance affecting it’s accuracy, leading to wrong preditions, and can causse overfitting, the image below illustrates this in the case of linear regression the best fit line drastically change due to the presence of outliers

![Untitled](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/Untitled.png)

# Types of outliers:

- **Global outlier:** a data point that devieates significantly from the entirety of the dataset, to detect them we can use statistical methods like z-score or ML models

![Untitled](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/Untitled%201.png)

- **Collective outlier:** they are data points that form a cluster that deviates significantly from the entirety of the dataset we can detect them with clustering or density based methods

![Untitled](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/Untitled%202.png)

- **Contextual outlier :** data point that are considered outliers in a specific context or subset of the data and might be considered normal in other, detecting them requires context aware approaches.

![Untitled](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/Untitled%203.png)

# Outlier Detection Algorithmes:

# Isolation forest:

Isolation forest is a variation of the random forrest algorithm used in unsupervised learning tasks to detect anomalies. Given a dataset X  It follows these steps to grow out an isolation tree

1. randomly select a feature p and a split value q
2. split X into 2 subsets(i.e the left and right branches) using the rule p<q
3. repeat the previous steps until a node has one isolated sample or a specified maximum depth is reached

the algorithem then repeats these 3 steps creating a new independant isolation tree each time thus an isolation forest.

Now how does it detect an anomaly? Well the idea is quite simple, anomalies are by definition different in terms of values from the other samples so they’ll require less splits to isolate (they’re gonna be located closer to the route of tree).

 Isolation trees will assign samples anomaly socres based on their distance from the root (number of edges) . The final score is the aggregate of the scores from all the trees . Insatnces with higher score are more likely to be outliers.

![Untitled](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/Untitled%204.png)

# Implementation:

Scikit-learn provides an easy to use implementation of the isolation forest algorithm all we have to do is

```python
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the dataset
num_samples = 200
outlier_percentage = 0.1

# Generate random samples from a normal distribution (inliers)
mean = [0, 0]
cov = [[1, 0], [0, 1]]
inliers = np.random.multivariate_normal(mean, cov, num_samples)

# Generate random samples for outliers
num_outliers = int(num_samples * outlier_percentage)
outliers = np.random.uniform(low=-10, high=10, size=(num_outliers, 2))

# Combine inliers and outliers
dataset = np.concatenate((inliers, outliers), axis=0)

# Plot the dataset
plt.scatter(dataset[:, 0], dataset[:, 1], color='blue', label='Inliers')
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label='Outliers')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('2D Dataset with Outliers')
plt.show()
```

![download.png](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/download.png)

```python
from sklearn.ensemble import IsolationForest

iForest = IsolationForest(n_estimators=20, verbose=2)
iForest.fit(dataset)
pred = iForest.predict(dataset)
plt.scatter(dataset[:, 0], dataset[:, 1], c=pred, cmap='RdBu')
plt.colorbar(label='Simplified Anomaly Score')
plt.show()
```

![download.png](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/download%201.png)

# Advantages:

- **Automatic Anomaly Scoring:** Isolations forests provides anomaly socres for each instances allowing for ranking and prioratization of anomalies
- **Complexity:** the time complexity of isolation forests is O(nlog(n)) so they are computaionally efficient. Plus the algorithm can build trees in parallel improving performance
- **Ability to Handle High-Dimensional Data:** Isolation Forests perform well even in high-dimensional spaces. Unlike many other outlier detection algorithms, they are not affected by the curse of dimensionality.

# Drawbacks:

- **Difficulty in Dense Regions:** Isolation Forests struggle to detect anomalies in dense regions of the dataset( like in our example). In such cases, the algorithm may require a larger number of isolation trees to effectivel   y separate anomalies from the majority of the data.
- **Sensitivity to Hyperparameters:** Isolation Forests have a few hyperparameters, such as the number of isolation trees and the subsampling size. The performance of the algorithm can be sensitive to the choice of these hyperparameters, requiring careful tuning.

# Extension:

Another drawback of isolation forests is that it’s decision boundries can only be parallel to the axes(vertical or horizantel) this can lead to a large number of splits for only few samples.

Extended Isolation forests adresses this by selecting a random slope and a random intercept from the available values to create a decision boundry for the split.

![Untitled](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/Untitled%205.png)

# DBSCAN:

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density based clustering algorithm that can be used for anomaly detection. it groups together dense regions of data points into clusters and seperates outliers.

### Key parameters:

DBSCAN has 2 key parameters:

- **eps**: the minimum distance between 2points so they’re considered neighbors
- **minPts:** the minimum number of points to define a cluster (including the point itself)

the algorithm uses these 2 parameters to classify points as:

- **Core point:** a point that has at least minPts within eps distance
- **Border point:** a point that has less than minPts neighbors but it’s reachable from a core point
- **Outlier:** not a core point and not reachable from any core points

### The Algorithm:

DBSCAN follows these simple steps:

1. Randomly select a point and check if it’s a core point 
2. if yes mark it as a core point and create a cluster with all it’s neighbors 
3. for every point in the cluster if it’s a core point then all it’s neighbors are added to the cluster too, if it’s a border point then it’s added to the cluster but it doesn’t contribute to expanding it

we repeat these steps until we have created all possible clusters, points that left unmarked are classified as outliers.

### Parameter Selection:

It’s pretty obvious from the previous description of the algorithm that the choice of the **eps**&**minPts** will greatly impact the results of DBSCAN therefore we need to understand how to select them.

- the value of **minPoints** should scale up with the size of the dataset, a general rule is      **minPoints ≥ D+1**  with D the dimension of the dataset, ofc it wouldn’t make sense in the case of 1D dataset because each point will be in it’s own cluser so it should be at least **3**
- If the value of **eps** is too small then a lot of points will be classified as outliers and if it’s too big then a lots of clusters will be merged into on. To choose the appropriate value we can use the    **K-distance graph,** it plots the average distance between each point to it’s k-nearest neighbors in ascending order, in this graph we should be able to observe a sudden change in curvature (elbow) it corresponds to the shift from high density regions to low density regions, the corresponding value can be used as estimate for eps.

![Untitled](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/Untitled%206.png)

### Implementation:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Generate a synthetic dataset with clusters and noise points
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.7, random_state=42)
rng = np.random.RandomState(42)
noise = rng.uniform(low=-10, high=10, size=(30, 2))
X = np.concatenate((X, noise), axis=0)

# Apply DBSCAN
dbscan = DBSCAN(eps=1, min_samples=5)
labels = dbscan.fit_predict(X)

plt.figure(figsize=(10,6))
plt.scatter(X[:,0], X[:,1],c=labels, cmap='Paired')
plt.title("Clusters determined by DBSCAN")
```

![download.png](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/download%202.png)

### Advantages:

- Does not require to specify the number of clusters beforehand
- DBSCAN is robust to outliers unlike other clustering algos
- Partition clustering like k-means,PAM clustering or hierarchical clustering like DIANA are suitable for spherical shaped or convex clusters howeverthey fail to recognize arbitrary shaped clusters, DBSCAN does not have that limitation

![Untitled](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/Untitled%207.png)

### Disadvantages:

- DBSCAN is sensitive to parameter selection (eps&minPts) as their choice has a huge impact on the result
- DBSCAN suffers from the curse of dimensionality since as the dimensions increase the concept of density becomes less meaningful leading to worse perfromance
- DBSACN struggles with datasets where the density of clusters varies since the chosen epts&minPts cannot generalize to the different clusters.

# Local Outlier Factor (LOF):

LOF is a local density based anomaly detection algorithm, it produces an anomaly score for each point in the dataset to determine if it’s an outlier or not by comparing it’s density to the density of neighboring points. The calculation of this score depends on other concepts and calculations done sequentially that we are going to present:

### K-distance & K-neighbors:

First we choose K the number of neighbors to consider for our calculations, with this k we can determine:

- K-distance of point A: the distance between A and it’s Kᵗʰ nearest neighbor
- K-neighbors of point A (Nₖ(A) ): includes all the point that in or on the circle of center A and radius K-distance

 

![1_kOwduufhfK0yWWd3N4r5rQ.webp](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/1_kOwduufhfK0yWWd3N4r5rQ.webp)

### Reachability distance:

$$
RD(A,B)=max(K-distance(B),distance(A,B))
$$

So basicly  if A is one of the K-neighbors of B then the reachability distance will be k-distance of B else it’s the distance between A and B. this value is asymetric RD(A←B)≠RD(B←A) because B can be one the K-neighbors of A but A is not one of B

### Local Reachability Density:

![Screenshot343.png](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/Screenshot343.png)

the local reachability density for a point A is the inverse of the avereage reachability distance between A and it’s K-neighbors, it measures the density of the K-enighbors around a point.           the closer the points are the smaller the distnace the bigger the density

### Local Outlier Factor:

![Screenshot344.png](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/Screenshot344.png)

LOF is the the ratio of the averge LRD of the K-neighbors of A to the LRD of A, it should be approximatly equal to 1 if A is not an outlier because the density of a point and it’s neighbors are almost equal howver if the density of a point is less then the average of it’s neighbors then it’s an outlier.

- LOF~1 ⇒ inlier
- LOF>1 ⇒ outlier

### Implementation:

```python
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# Generate sample data
np.random.seed(42)
X, _ = make_blobs(n_samples=100, centers=2, cluster_std=0.7, random_state=42)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.vstack([X, X_outliers])

n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1

# Fit the LOF model
lof = LocalOutlierFactor(n_neighbors=20)  # Adjust the n_neighbors parameter as needed
y_pred = lof.fit_predict(X)

n_errors = (y_pred != ground_truth).sum()
# Get the negative outlier factor scores
lof_scores = lof.negative_outlier_factor_

print('nmber of missclassified outliers is:',n_errors)
```

`nmber of missclassified outliers is: 4`

```python
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection

def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])

plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
# plot circles with radius proportional to the outlier scores
radius = (lof_scores.max() - lof_scores) / (lof_scores.max() - lof_scores.min())
scatter = plt.scatter(
    X[:, 0],
    X[:, 1],
    s=1000 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)

plt.xlabel("prediction errors: %d" % (n_errors))
plt.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)
plt.title("Local Outlier Factor (LOF)")
plt.show()
```

![download.png](Principles%20of%20Outlier%20Detection%20135880b991ce49c3b69c402a0c8a2ab3/download%203.png)

### Advantages:

The major advantge of LOF compared to other detection algorithmes is it’s ability to detect local outliers, for example distance based approches work well in detecting global outliers but struggle to detect local ones (outliers that are close to a particular cluster),  LOF aliviates these problems by focusing on local densities

### Disadvantages:

- LOF assumes that outliers have lower densities compared to their neighbors so it can struggle when handling data with variying density clusters or overlapping one, it can missclassify points from low density clusters as outliers.
- Sensitivty to parameters as LOF requires specifying the **K** hyperparmas beforehand, this choice has significant impact on the results of the algorithm so it may require trying different ones or specific domain knowledge