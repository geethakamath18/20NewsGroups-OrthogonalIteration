#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.cm as cm
import sklearn.metrics


# In[2]:


def linear_transform(a, e):
    assert a.ndim == 1
    assert np.allclose(1, np.sum(e**2))
    u = a - np.sign(a[0]) * np.linalg.norm(a) * e  
    v = u / np.linalg.norm(u)
    H = np.eye(len(a)) - 2 * np.outer(v, v)
    return H


# In[3]:


def QR(matrix):    
    n,m=matrix.shape #TALL-WIDE
    assert n >= m  
    Q = np.eye(n)
    R = matrix.copy()
    for i in range(m - int(n==m)):
        r = R[i:, i]
        if np.allclose(r[1:], 0):
            continue   
        # e is the i-th basis vector of the minor matrix.
        e = np.zeros(n-i)
        e[0] = 1  
        H = np.eye(n)
        H[i:, i:] = linear_transform(r, e)
        Q = Q @ H.T
        R = H @ R   
    return Q, R


# In[4]:


class PCA:
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = bool(whiten)
    
    def fit(self, X):
        n, m = X.shape
        self.mu = X.mean(axis=0)
        X = X - self.mu
        C = X.T @ X / (n-1) #Eigen Decomposition
        C_k = C
        Q_k = np.eye( C.shape[1] )
        for k in range(100): #number of iterations=100
            Q, R = QR(C_k)
            Q_k = Q_k @ Q
            C_k = R @ Q
        self.eigenvalues =  np.diag(C_k)
        self.eigenvectors = Q_k
        if self.n_components is not None:  # truncate the number of components
            self.eigenvalues = self.eigenvalues[0:self.n_components]
            self.eigenvectors = self.eigenvectors[:, 0:self.n_components]      
        descending_order = np.flip(np.argsort(self.eigenvalues)) #eigenvalues in descending order
        self.eigenvalues = self.eigenvalues[descending_order]
        self.eigenvectors = self.eigenvectors[:, descending_order]
        return self

    def transform(self, X):
        X = X - self.mu
        if self.whiten:
            X = X / self.std
        return X @ self.eigenvectors


# In[5]:


newsgroups_train1 = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes')) #without metadata
newsgroups_train2 = fetch_20newsgroups(subset='train') #with metadata


# In[6]:


categories = newsgroups_train1.target_names


# In[7]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=200, stop_words='english')


# In[8]:


afterTFIDF1 = tfidf_vectorizer.fit_transform(newsgroups_train1.data)
afterTFIDF2 = tfidf_vectorizer.fit_transform(newsgroups_train2.data)


# In[9]:


data1 = afterTFIDF1.toarray()
data2 = afterTFIDF2.toarray()


# In[10]:


pca1 = PCA(whiten=False, n_components=2)
pca2 = PCA(whiten=False, n_components=2)


# In[11]:


pca1.fit(data1)
final1 = pca1.transform(afterTFIDF1.toarray())
pca2.fit(data2)
final2 = pca2.transform(afterTFIDF2.toarray())


# In[12]:


final1.shape


# In[13]:


colors = cm.rainbow(np.linspace(0, 1, len(categories)))


# In[14]:


plt.figure(figsize=(15,15))
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title('Principal Component Analysis-without metadata')
plt.scatter(final1[:,0], final1[:,1],c=colors[newsgroups_train1.target])
for i in range(20):
    plt.scatter([],[],colors[i],label=categories[i])
plt.legend(loc='best',markerscale=6)
plt.show()


# In[16]:


plt.figure(figsize=(25,15))
plt.title('Principal Component Analysis-without metadata')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.scatter(final2[:,0], final2[:,1],c=colors[newsgroups_train2.target])
for i in range(20):
    plt.scatter([],[],colors[i],label=categories[i])
plt.legend(loc='best',markerscale=10)
plt.show()


# In[ ]:




