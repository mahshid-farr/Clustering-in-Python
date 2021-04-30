#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# for plot styling
import seaborn as sns; sns.set()  
from sklearn.cluster import KMeans


# In[7]:


data = pd.read_csv(r'C:\Users\mahsh\OneDrive\Desktop\data.csv')
data


# In[8]:


plt.scatter(data['Weight'],data['Cholesterol'])
plt.xlabel('Weight')
plt.ylabel('Cholesterol')
plt.show()


# In[11]:


data.isnull().values.any()
print(f'Is there any null value?{data.isnull().values.any()}')
data[data.isnull().any(axis=1)]


# In[46]:


kmeans=KMeans(n_clusters=3)
kmeans.fit(data)
y_kmeans = kmeans.predict(data)


# In[47]:


clusters=data.copy()
clusters['cluster_pred']=kmeans.fit_predict(data)


# In[48]:


plt.scatter(clusters['Weight'],clusters['Cholesterol'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Weight')
plt.ylabel('Cholesterol')
plt.show()


# In[49]:


from sklearn.metrics import silhouette_score
range_n_clusters = list (range(2,10))
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict(data)
    centers = clusterer.cluster_centers_

    score = silhouette_score(data, preds)
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))


# In[50]:


sil = []

for k in range(2, 10):
  kmeans = KMeans(n_clusters = k).fit(data)  
  preds = kmeans.fit_predict(data)
  sil.append(silhouette_score(data, preds, metric = 'euclidean'))


plt.plot(range(2, 10), sil)
plt.title('Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sil')
plt.show()


# In[51]:


score = silhouette_score (data, y_kmeans, metric='euclidean')
score


# In[56]:


kmeans_new=KMeans(n_clusters=2)
kmeans.fit(data)
clusters_new=data.copy()
clusters_new['cluster_pred']=kmeans_new.fit_predict(data)
clusters_new


# In[57]:


plt.scatter(clusters_new['Weight'],clusters_new['Cholesterol'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Weight')
plt.ylabel('Cholesterol')
plt.show()


# In[ ]:




