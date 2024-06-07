import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#load data
cancer = datasets.load_breast_cancer()

#create features and labels
X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

#create classifier
classes = ['malignant' 'benign']

clf = svm.SVC(kernel='poly', C = 2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

#evaluate the classifier
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)
