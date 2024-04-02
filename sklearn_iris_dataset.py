#!/usr/bin/env python
# coding: utf-8


from sklearn.datasets import load_iris
iris = load_iris()


X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names 
print(iris.target_names )
type(X)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
print(X_train.shape)
print(X_test.shape)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# from sklearn.tree import DecisionTreeClassifier
# knn = DecisionTreeClassifier()
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)




from sklearn import metrics 
print(metrics.accuracy_score(y_test, y_pred))



sample = [[3,5,4,2], [2,3,6,4]]
predictions = knn.predict(sample)
pred_species = [iris.target_names[p] for p in predictions]
print(pred_species)


from joblib import dump, load
dump(knn, 'mlbrain.joblib')

model = load('mlbrain.joblib')
model.predict(X_test)


sample = [[3,5,4,2], [2,3,6,4]]
predictions = model.predict(sample)
pred_species = [iris.target_names[p] for p in predictions]
print(pred_species)


