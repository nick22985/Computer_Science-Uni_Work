#%%
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

#%%
training = pd.read_csv('Data/Q2/training.csv')
testing = pd.read_csv('Data/Q2/testing.csv')
testing_y = testing['class']
#spliting data
testingData, validatingData = train_test_split(testing, stratify=testing_y, test_size=0.5, random_state=5)
#All Data
total_data = testing.append(training)
Y_total = total_data['class']
# Divide into input X and Y for test
X_test = testingData.drop('class', axis=1)
Y_test = testingData['class']
# Divide into input X and Y for train
X_train = training.drop('class', axis=1)
Y_train = training['class']
# Divide into input X and Y for validation set
X_validate = validatingData.drop('class', axis=1)
Y_validate = validatingData['class']

#%% Functions
def eval_model(model, dataset, X_train, Y_train, X_test, Y_test):
	fig = plt.figure(figsize=[25, 8])
	ax = fig.add_subplot(1, 2, 1)
	conf = plot_confusion_matrix(model, X_train, Y_train, normalize='true', ax=ax)
	pred = model.predict(X_train)
	conf.ax_.set_title('Training Set Performance ' + dataset + " " + str(sum(pred == Y_train)/len(Y_train)));
	ax = fig.add_subplot(1, 2, 2)
	conf = plot_confusion_matrix(model, X_validate, Y_validate, normalize='true', ax=ax)
	pred = model.predict(X_test)
	conf.ax_.set_title('Validate Set Performance ' + dataset + " " + str(sum(pred == Y_test)/len(Y_test)));
	fig.show()

#%% Data imbalance
plt.figure()
sorted_Y_total = sorted(Y_total)
plt.hist(sorted_Y_total, 4)
plt.title('Histogram Total data set')
plt.figure()
sorted_Y_train = sorted(Y_train)
plt.hist(sorted_Y_train, 4)
plt.title('Histogram Training data set')
plt.figure()
sorted_Y_test = sorted(Y_test)
plt.hist(sorted_Y_test, 4)
plt.title('Histogram Test data set')
plt.figure()
sorted_Y_validate = sorted(Y_validate)
plt.hist(sorted_Y_validate, 4)
plt.title('Histogram Validation data set')
plt.show()
print("Training Data: ", len(Y_train), "/", len(total_data), "=", round((len(Y_train)/len(total_data)), 2) )
print("Testing Data: ", len(Y_test), "/", len(total_data), "=", round((len(Y_test)/len(total_data)), 2) )
print("Validation Data: ", len(Y_validate), "/", len(total_data), "=", round((len(Y_validate)/len(total_data)), 2) )

#%% SVM Linear
svm = SVC(C=1.0, kernel='linear')
svm.fit(X_train, Y_train)
eval_model(svm, "SVM Linear", X_train, Y_train, X_test, Y_test)

#%% SVM RBF
svm = SVC(C=1.0, kernel='rbf')
svm.fit(X_train, Y_train)
eval_model(svm, "RBF", X_train, Y_train, X_test, Y_test)

#%% SVC poly
svm = SVC(C=1.0, kernel='poly')
svm.fit(X_train, Y_train)
eval_model(svm, 'Poly', X_train, Y_train, X_test, Y_test)

#%%
param_grid = [
  {'C': [10, 1, 0.1, 0.001, 0.0001, 0.00001], 'kernel': ['linear']},
  {'C': [10, 1, 0.1, 0.001, 0.0001, 0.00001], 'gamma': [10, 1, 0.1, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']},
  {'C': [10, 1, 0.1, 0.001, 0.0001, 0.00001], 'degree': [0.1, 2, 3, 5, 9, 11, 20, 50], 'kernel': ['poly']},
 ]
svm = SVC()
grid_search = GridSearchCV(svm, param_grid)
grid_search.fit(X_train, Y_train)
grid_search.cv_results_
print(grid_search.cv_results_)

#%% Best System
best_system = numpy.argmin(grid_search.cv_results_['rank_test_score'])
params = grid_search.cv_results_['params'][best_system]
print(params)
svm = SVC().set_params(**params)
svm.fit(X_train, Y_train)
eval_model(svm, "Best System", X_train, Y_train, X_test, Y_test)



#%% best param comparison svm
print("bestOne vs One")
svm = SVC(C=100, gamma=0.0001)
svm.fit(X_train, Y_train)
eval_model(svm, "bestOne vs One", X_train, Y_train, X_validate, Y_validate)

#%%
print("One vs One with balanced weight")
svm = SVC(C=100, gamma=0.0001, class_weight='balanced')
svm.fit(X_train, Y_train)
eval_model(svm, "One vs One with balanced weight", X_train, Y_train, X_validate, Y_validate)

#%%
print("One vs One RBF")
svm = SVC(kernel='rbf', C=100, gamma=0.0001, max_iter=10e6)
svm.fit(X_train, Y_train)
eval_model(svm, "One vs One RBF", X_train, Y_train, X_validate, Y_validate)

#%%
print("One vs One RBF with balanced weight")
svm = SVC(kernel='rbf', C=100, gamma=0.0001, class_weight='balanced')
svm.fit(X_train, Y_train)
eval_model(svm, "One vs One RBF with balanced weight", X_train, Y_train, X_validate, Y_validate)

#%%
print("One vs One Poly")
svm = SVC(kernel='poly', C=100, degree=3, max_iter=10e6)
svm.fit(X_train, Y_train)
eval_model(svm, "One vs One Poly", X_train, Y_train, X_validate, Y_validate)

#%%
print("One vs One Poly with balanced weight")
svm = SVC(kernel='poly', C=100, degree=3, class_weight='balanced')
svm.fit(X_train, Y_train)
eval_model(svm, "One vs One Poly with balanced weight", X_train, Y_train, X_validate, Y_validate)

#%%
print("One vs All")
onevsall_svm = OneVsRestClassifier(SVC(C=100, gamma=0.0001))
onevsall_svm.fit(X_train, Y_train)
eval_model(onevsall_svm, "One vs All", X_train, Y_train, X_validate, Y_validate)

#%%
print("One vs All with balanced weights")
onevsall_svm = OneVsRestClassifier(SVC(C=100, gamma=0.0001, class_weight='balanced'))
onevsall_svm.fit(X_train, Y_train)
eval_model(onevsall_svm, "One vs All with balanced weights", X_train, Y_train, X_validate, Y_validate)

#%%
print("One vs Rest RBF")
onevsall_svm = OneVsRestClassifier(SVC(kernel='rbf', C=100, gamma=0.0001))
onevsall_svm.fit(X_train, Y_train)
eval_model(onevsall_svm, "One vs Rest RBF", X_train, Y_train, X_validate, Y_validate)

#%%
print("One vs Rest balanced weights RBF")
onevsall_svm = OneVsRestClassifier(SVC(kernel='rbf', C=100, gamma=0.0001, class_weight='balanced'))
onevsall_svm.fit(X_train, Y_train)
eval_model(onevsall_svm, "One vs Rest balanced weights RBF", X_train, Y_train, X_validate, Y_validate)

#%%
print("One vs Rest Polynomial")
onevsall_svm = OneVsRestClassifier(SVC(kernel='poly', C=100, degree=3, max_iter=10e6))
onevsall_svm.fit(X_train, Y_train)
eval_model(onevsall_svm, "One vs Rest Polynomial", X_train, Y_train, X_validate, Y_validate)

#%%
print("One vs Rest balanced weights Polynomial")
onevsall_svm = OneVsRestClassifier(SVC(kernel='poly', C=100, degree=3, max_iter=10e6, class_weight='balanced'))
onevsall_svm.fit(X_train, Y_train)
eval_model(onevsall_svm, "One vs Rest balanced weights Polynomial", X_train, Y_train, X_validate, Y_validate)

svm = SVC()
svm.fit(X_train, Y_train)
eval_model(svm, "Testing", X_train, Y_train, X_test, Y_test)

#%%
print("\n Linear SVM \n")
for C in [numpy.inf, 1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001]:
	print('Linear SVM, with C = {}'.format(C))
	svm = SVC(C=C, kernel='linear', max_iter=10e5)
	svm.fit(X_train, Y_train)
	eval_model(svm, "Validating",  X_train, Y_train, X_validate, Y_validate)

#%%
print("\n\n RBF SVM \n\n")
for kernel_scale in [1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001, 'scale']:
	print('RBF SVM, with scale = {}'.format(kernel_scale))
	svm = SVC(kernel='rbf', gamma=kernel_scale, max_iter=10e6)
	svm.fit(X_train, Y_train)
	eval_model(svm, 'RBF SVM, with scale = {}'.format(kernel_scale),  X_train, Y_train, X_validate, Y_validate)
#%%
print("\n\n RBF SVM \n\n")
for C in [numpy.inf, 1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001]:
	for kernel_scale in [1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001, 'scale']:
		print('RBF SVM, with C = {} and scale = {}'.format(C, kernel_scale))
		svm = SVC(C=C, kernel='rbf', gamma=kernel_scale, max_iter=10e6)
		svm.fit(X_train, Y_train)
		eval_model(svm, 'RBF SVM, with C = {} and scale = {}'.format(C, kernel_scale),  X_train, Y_train, X_validate, Y_validate)
#%%
print("\n\n Polynomial SVM \n\n")
for degree in [0.1, 2, 3, 5, 9, 11, 20, 50]:
	print('Polynomial Kernel SVM, with degree = {}'.format(degree))
	svm = SVC(kernel='poly', degree=degree, max_iter=10e6)
	svm.fit(X_train, Y_train)
	eval_model(svm, 'Polynomial Kernel SVM, with degree = {}'.format(degree),  X_train, Y_train, X_validate, Y_validate)
#%%
print("\n\n Polynomial SVM \n\n")
for C in [1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001]:
	for degree in [0.1, 2, 3, 5, 9, 11, 20, 50]:
		print('Polynomial Kernel SVM, with C = {} and degree = {}'.format(C, degree))
		svm = SVC(C=C, kernel='poly', degree=degree, max_iter=10e6)
		svm.fit(X_train, Y_train)
		eval_model(svm, 'Polynomial Kernel SVM, with C = {} and degree = {}'.format(C, degree),  X_train, Y_train, X_validate, Y_validate)


#%% K-Nearest Neighbours Classifier
cknn = KNeighborsClassifier(n_neighbors=10, weights='uniform')
cknn.fit(X_train, Y_train)
eval_model(cknn, 'K-Nearest Neighbours Classifier', X_train, Y_train, X_test, Y_test)

#%%
cknn = KNeighborsClassifier(n_neighbors=198, weights='uniform')
cknn.fit(X_train, Y_train)
eval_model(cknn, 'Uniform', X_train, Y_train, X_test, Y_test)

#%%
cknn = KNeighborsClassifier(n_neighbors=198, weights='distance')
cknn.fit(X_train, Y_train)
eval_model(cknn, 'distance', X_train, Y_train, X_test, Y_test)

#%%
cknn = KNeighborsClassifier()
params = {'n_neighbors' : list(range(1,21)), 'weights' : ['uniform', 'distance']}
rand_search = RandomizedSearchCV(cknn, params, n_iter=20)
rand_search.fit(X_train, Y_train)
print(rand_search.cv_results_)
#%%
best_system = numpy.argmin(rand_search.cv_results_['rank_test_score'])
params = rand_search.cv_results_['params'][best_system]
print(params)
cknn = KNeighborsClassifier().set_params(**params)
cknn.fit(X_train, Y_train)
eval_model(cknn, 'Best Kneighbors', X_train, Y_train, X_test, Y_test)

#%% optimal knn param
#%%
rf = RandomForestClassifier(n_estimators=250, max_depth=20, random_state=0)
rf.fit(X_train, Y_train)
eval_model(rf, "random forest", X_train, Y_train, X_test, Y_test)

#%%
fig = plt.figure(figsize=(20,15))
_ = tree.plot_tree(rf.estimators_[0], filled=True) 
fig = plt.figure(figsize=(20,15))
_ = tree.plot_tree(rf.estimators_[4], filled=True) 
fig = plt.figure(figsize=(20,15))
_ = tree.plot_tree(rf.estimators_[42], filled=True) 

# %%
