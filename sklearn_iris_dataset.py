#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[34]:


X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names 
print(iris.target_names )
type(X)


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
print(X_train.shape)
print(X_test.shape)


# In[32]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# from sklearn.tree import DecisionTreeClassifier
# knn = DecisionTreeClassifier()
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)




# In[33]:


from sklearn import metrics 
print(metrics.accuracy_score(y_test, y_pred))


# In[36]:


sample = [[3,5,4,2], [2,3,6,4]]
predictions = knn.predict(sample)
pred_species = [iris.target_names[p] for p in predictions]
print(pred_species)



# In[41]:


from joblib import dump, load
dump(knn, 'mlbrain.joblib')


# In[43]:


model = load('mlbrain.joblib')
model.predict(X_test)


# In[45]:


sample = [[3,5,4,2], [2,3,6,4]]
predictions = model.predict(sample)
pred_species = [iris.target_names[p] for p in predictions]
print(pred_species)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




