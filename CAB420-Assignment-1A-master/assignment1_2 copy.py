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
Y_total = totalData['class']
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
def eval_model(model, X_data, Y_data, dataset):
	pred = model.predict(X_data)	
	print(dataset + ' Accuracy: ' + str(sum(pred == Y_data)/len(Y_data)))

def summ_model(model, dataset, X_train, Y_train, X_validate, Y_validate):
	eval_model(model, X_validate, Y_validate, dataset)
	visualise_model(model, X_train, Y_train, X_validate, Y_validate)

def visualise_model(model, X_train, Y_train, X_validate, Y_validate):
	fig = plt.figure(figsize=[25, 8])
	ax = fig.add_subplot(1, 2, 1)
	conf = plot_confusion_matrix(model, X_train, Y_train, normalize='true', ax=ax)
	conf.ax_.set_title('Training Set Performance');
	ax = fig.add_subplot(1, 2, 2)
	conf = plot_confusion_matrix(model, X_validate, Y_validate, normalize='true', ax=ax)
	conf.ax_.set_title('Validate Set Performance');
	fig.show()
 
def final_model_summary(given_model):
	model = given_model
	# Accuracy
	eval_model(model, X_train, Y_train, "Training")
	eval_model(model, X_validate, Y_validate, "Validating")
	eval_model(model, X_test, Y_test, "Testing")

	# Confusion matrix
	fig = plt.figure(figsize=[25, 12])
	ax = fig.add_subplot(2, 2, 1)
	conf = plot_confusion_matrix(model, X_train, Y_train, normalize='true', ax=ax)
	conf.ax_.set_title('Training Set Performance');

	ax = fig.add_subplot(2, 2, 2)
	conf = plot_confusion_matrix(model, X_validate, Y_validate, normalize='true', ax=ax)
	conf.ax_.set_title('Validate Set Performance');

	ax = fig.add_subplot(2, 2, 3)
	conf = plot_confusion_matrix(model, X_test, Y_test, normalize='true', ax=ax)
	conf.ax_.set_title('Testing Set Performance');
	fig.show()

cknn = KNeighborsClassifier(n_neighbors=3)
cknn.fit(X_train, Y_train)
eval_model(cknn, X_train, Y_train, 'training')
#%%
cknn = KNeighborsClassifier(n_neighbors=1, weights='distance')
cknn = KNeighborsClassifier(n_neighbors=1)
cknn.fit(X_train, Y_train)


#%%
eval_model(cknn, X_train, Y_train, "Training")
eval_model(cknn, X_validate, Y_validate, "Validating")
eval_model(cknn, X_test, Y_test, "Testing")

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

#%% Confusion matrix
fig = plt.figure(figsize=[25, 12])
ax = fig.add_subplot(2, 2, 1)
conf = plot_confusion_matrix(cknn, X_train, Y_train, normalize='true', ax=ax)
conf.ax_.set_title('Training Set Performance');
ax = fig.add_subplot(2, 2, 2)
conf = plot_confusion_matrix(cknn, X_validate, Y_validate, normalize='true', ax=ax)
conf.ax_.set_title('Validate Set Performance');
ax = fig.add_subplot(2, 2, 3)
conf = plot_confusion_matrix(cknn, X_test, Y_test, normalize='true', ax=ax)
conf.ax_.set_title('Testing Set Performance');
fig.show()

# %% Optimal Parameters
param_grid = [
  {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
  {'C': [0.1, 1, 10, 100, 1000], 'degree': [3, 4, 5, 6], 'kernel': ['poly']},
 ]
svm = SVC()
grid_search = GridSearchCV(svm, param_grid)
grid_search.fit(X_train, Y_train)
grid_search.cv_results_
best_system = numpy.argmin(grid_search.cv_results_['rank_test_score'])
params = grid_search.cv_results_['params'][best_system]
print(params)
svm = SVC().set_params(**params)
svm.fit(X_train, Y_train)
summ_model(svm, "best", X_train, Y_train, X_test, Y_test)

#%%
# Normal SVM
svm = SVC()
svm.fit(X_train, Y_train)
summ_model(svm, "Testing", X_train, Y_train, X_test, Y_test)

print("\n Linear SVM \n")
for C in [numpy.inf, 1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001]:
	print('Linear SVM, with C = {}'.format(C))
	svm = SVC(C=C, kernel='linear', max_iter=10e5)
	svm.fit(X_train, Y_train)
	summ_model(svm, "Validating",  X_train, Y_train, X_validate, Y_validate)

print("\n\n RBF SVM \n\n")
for kernel_scale in [1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001, 'scale']:
	print('RBF SVM, with scale = {}'.format(kernel_scale))
	svm = SVC(kernel='rbf', gamma=kernel_scale, max_iter=10e6)
	svm.fit(X_train, Y_train)
	summ_model(svm, "Validating",  X_train, Y_train, X_validate, Y_validate)

print("\n\n RBF SVM \n\n")
for C in [numpy.inf, 1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001]:
	for kernel_scale in [1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001, 'scale']:
		print('RBF SVM, with C = {} and scale = {}'.format(C, kernel_scale))
		svm = SVC(C=C, kernel='rbf', gamma=kernel_scale, max_iter=10e6)
		svm.fit(X_train, Y_train)
		summ_model(svm, "Validating",  X_train, Y_train, X_validate, Y_validate)

print("\n\n Polynomial SVM \n\n")
for degree in [0.1, 2, 3, 5, 9, 11, 20, 50]:
	print('Polynomial Kernel SVM, with degree = {}'.format(degree))
	svm = SVC(kernel='poly', degree=degree, max_iter=10e6)
	svm.fit(X_train, Y_train)
	summ_model(svm, "Validating",  X_train, Y_train, X_validate, Y_validate)

print("\n\n Polynomial SVM \n\n")
for C in [1000, 100, 10, 1, 0.1, 0.001, 0.0001, 0.00001]:
	for degree in [0.1, 2, 3, 5, 9, 11, 20, 50]:
		print('Polynomial Kernel SVM, with C = {} and degree = {}'.format(C, degree))
		svm = SVC(C=C, kernel='poly', degree=degree, max_iter=10e6)
		svm.fit(X_train, Y_train)
		summ_model(svm, "Validating",  X_train, Y_train, X_validate, Y_validate)
  
#%%
svm = OneVsRestClassifier(SVC(C=100, gamma=0.0001))
svm = OneVsRestClassifier(SVC(C=100, gamma=0.0001, class_weight='balanced'))
svm.fit(X_train, Y_train)

#%% Best Comparison SVM
print("Optimal One vs One")
svm = SVC(C=100, gamma=0.0001)
svm.fit(X_train, Y_train)
summ_model(svm, "Optimal One vs One", X_train, Y_train, X_validate, Y_validate)
final_model_summary(svm)
print("\n")

print("Optimal One vs One with balanced weight")
svm = SVC(C=100, gamma=0.0001, class_weight='balanced')
svm.fit(X_train, Y_train)
summ_model(svm, "n/a", X_train, Y_train, X_validate, Y_validate)
final_model_summary(svm)
print("\n")

print("Optimal One vs One RBF")
svm = SVC(kernel='rbf', C=100, gamma=0.0001, max_iter=10e6)
svm.fit(X_train, Y_train)
summ_model(svm, "n/a", X_train, Y_train, X_validate, Y_validate)
final_model_summary(svm)
print("\n")

print("Optimal One vs One RBF with balanced weight")
svm = SVC(kernel='rbf', C=100, gamma=0.0001, class_weight='balanced')
svm.fit(X_train, Y_train)
summ_model(svm, "n/a", X_train, Y_train, X_validate, Y_validate)
final_model_summary(svm)
print("\n")

print("Optimal One vs One Poly")
svm = SVC(kernel='poly', C=100, degree=3, max_iter=10e6)
svm.fit(X_train, Y_train)
summ_model(svm, "n/a", X_train, Y_train, X_validate, Y_validate)
final_model_summary(svm)
print("\n")

print("Optimal One vs One Poly with balanced weight")
svm = SVC(kernel='poly', C=100, degree=3, class_weight='balanced')
svm.fit(X_train, Y_train)
summ_model(svm, "n/a", X_train, Y_train, X_validate, Y_validate)
final_model_summary(svm)
print("\n")
# %%
final_model_summary(svm)
# %%
