# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import urllib.request
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

election = pd.read_csv('/content/dataB.csv')
features = pd.read_csv('/content/dataC.csv')
feature = features[["yshuv","Totalpop16","AhuzOlim1990","Bagrut_1516","Eshkol_15","Shetah_shiput_13","DiabRate14_16"]] 

election.isna().sum()

feature.isna().sum()

feature.loc[feature.AhuzOlim1990.isna(), 'AhuzOlim1990'] = feature.AhuzOlim1990.mean()

feature.rename(columns={'yshuv': 'yshuv_symbol'}, inplace=True)

election.rename(columns={'To': 'to'}, inplace=True)
election.rename(columns={'Symbol of a settlement': 'yshuv_symbol'}, inplace=True)

election.rename(columns={'To.1': 'TO'}, inplace=True)

data = pd.merge(feature,election, on ='yshuv_symbol',how='inner')
data.shape

data.rename(columns={'Totalpop16': 'Totalpop'}, inplace=True)

data['label'] = feature.Totalpop16.astype(str) + '_' + feature['Bagrut_1516'].astype(str) + '_' + feature['Eshkol_15'].astype(str)
data.head()

import sys
sys.setrecursionlimit(2000)

print(sys.getrecursionlimit())

"""#Hierarchical clustering Algorithm """

data.columns

data.iloc[:,:-1]

data.drop(['BJ','Insulations','Truth','Name of settlement'] , axis = 1, inplace = True )

Z_list = []
data_scaling_methods= [MinMaxScaler(),StandardScaler()]
linkage_methods = ['single','complete','average','centroid']
pictures = {}

for scaler in data_scaling_methods:
  for linkage in linkage_methods:
    plt.figure(figsize = (25,10))
    scaler.fit(data.iloc[:,:-1])
    scaled_data = pd.DataFrame(scaler.transform(data.iloc[:,:-1]))
    #print(scaled_data )
    Z = sch.linkage(scaled_data , method=linkage)
    Z_list.append(Z)
    print('The data scaling method is: ',scaler ,"and the linkage method is: ", linkage)
    dend = sch.dendrogram(Z , labels = data['label'].values, leaf_rotation=90)
    plt.show()

"""# K-Means Algorithm"""

# scaler = MinMaxScaler()
# scaler.fit(election.iloc[:,:-1])

SSE = []
for k in range(1,10):
  model = KMeans(n_clusters = k,init='k-means++')
  model.fit(data.iloc[:,:-1])
  SSE.append(model.inertia_)

plt.plot(range(1,10),SSE) 
plt.scatter(range(1,10),SSE,c= 'r',marker = '*') 
plt.xlabel("number of clusters") 
plt.ylabel("SSE") 
plt.grid() 
plt.show() 

kmeans_model = KMeans(n_clusters = 2) 
kmeans_model.fit(data.iloc[:,:-1])  
print("The final size of centers of {} clusters list is : {}" .format(2,kmeans_model.cluster_centers_.shape[1]) )
print("The sum square error is(SSE):", kmeans_model.inertia_) 
print("The number of iterations is:",  kmeans_model.n_iter_) 
print("*"*50)
kmeans_model = KMeans(n_clusters = 3) 
kmeans_model.fit(data.iloc[:,:-1])  
print("The final size of centers of {} clusters list is : {}" .format(3,kmeans_model.cluster_centers_.shape[1]) )
print("The sum square error is(SSE):", kmeans_model.inertia_) 
print("The number of iterations is:",  kmeans_model.n_iter_)

def rand_index(c1,c2):
  n=len(c1)
  alpha , beta, gamma, delta = 0,0,0,0
  for i in range(n-1):
    for j in range(i+1,n):
      if ((c1[i]==c1[j]) & (c2[i]==c2[j])):
        alpha += 1
      elif((c1[i]!=c1[j]) & (c2[i]!=c2[j])):
        beta += 1
      elif((c1[i]==c1[j]) & (c2[i]!=c2[j])):
        gamma += 1
      elif((c1[i]!=c1[j]) & (c2[i]==c2[j])):
        delta += 1
  #print(alpha,beta,gamma,delta)
  return (alpha+beta)/(alpha+beta+gamma+delta)

def pre_randindex():
  kmeans_model = KMeans(n_clusters = 3) 
  kmeans_model.fit(data.iloc[:,:-1])
  kmeans_clusters = kmeans_model.cluster_centers_
  scaler = MinMaxScaler()
  scaler.fit( data.iloc[:,:-1] )
  scaled_data = pd.DataFrame(scaler.transform( data.iloc[:,:-1]))
  Z = sch.linkage(scaled_data,method='complete')
  gap = []
  for i in range(1,30):   #i = num__of_clusters
          kmeans_cluster = kmeans_clusters[i%3][i]
          Hierarchical_clusters = sch.fcluster(Z_list[i],i,criterion='maxclust')
          randIndex = rand_index(Hierarchical_clusters,kmeans_cluster)
          gap = np.append(gap,randIndex)
          #print(i,randIndex)
  return gap

gap = pre_randindex()
plt.plot(1-gap)

kmeans_model = KMeans(n_clusters = 3) 
  kmeans_model.fit(election.iloc[:,:-1])
  kmeans_clusters = kmeans_model.cluster_centers_
  print(kmeans_clusters[0][0])

pd.Series(kmeans_model.labels_).unique()

Z_list[0].shape

len(election)