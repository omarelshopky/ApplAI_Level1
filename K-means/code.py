# Import the needed libraries
import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt
# =====================================================================
  
# Initialize our train-test Data
data = pd.read_excel("Training Data.xlsx")
cluster_array = data.iloc[:,[0,1,2]].values
ExamplesCentroids = np.zeros(cluster_array.shape[0],dtype= int)
# =====================================================================
  
# Calculate Euclidean Distance
def calc_distance(X1, X2):
  distance = np.sqrt(sum((X1 - X2)**2))
  return distance
# =====================================================================

# Get Examples Centroids
def assign_clusters(centroids, cluster_array):
  ExamplesCentroids = np.zeros(cluster_array.shape[0],dtype= int)
  cen_num = -2
  for i in range(len(cluster_array)):
    c = 1000000000
    for j in range(len(centroids)):
      x = calc_distance(cluster_array[i],centroids[j])
      if x < c:
        c = x
        cen_num = j
    ExamplesCentroids[i] = cen_num
  return ExamplesCentroids
# =====================================================================

#Calculate the Mean for the current centroids
def calc_centroids(cluster_array, ExamplesCentroids, K):
    m, n = cluster_array.shape
    centroids = np.zeros((K, n))
    for i in range(k):
      x_sum=0
      y_sum=0
      o = 0
      for j in range(len(ExamplesCentroids)):
        if ExamplesCentroids[j] == i:
          x_sum += cluster_array[j][0]
          y_sum += cluster_array[j][1]
          o += 1
      centroids[i][0] = x_sum / o
      centroids[i][1] = y_sum / o  
    return centroids
# =============================================================

#Put all together
k = 3
centroids = [cluster_array[i+2] for i in range(k)]
for i in range(20):
    ExamplesCentroids = assign_clusters(centroids, cluster_array)
    centroids = calc_centroids(cluster_array,ExamplesCentroids,k)

plt.subplots(figsize=(9,6))
plt.scatter(cluster_array[:, 0], cluster_array[:, 1], 
            c=ExamplesCentroids, cmap=plt.cm.Spectral)

test = pd.read_csv('Test Data.txt',header = None).to_numpy()
test_data_categories = assign_clusters(centroids, test)
print(test_data_categories)

#Save output to a text file so we can process it
import sys
f = open("Categories.txt", "w")
for i in test_data_categories:
  print(i, file = f)
f.close()
  
  
