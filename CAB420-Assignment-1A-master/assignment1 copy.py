#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
from statsmodels import api as sm
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import mean_squared_error

# %%
data = pd.read_csv('Data/Q1/communities.csv')
print(data.head())
print(data.shape)
print(data.size)
# %% Raw Data (This is the raw data)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(data[' ViolentCrimesPerPop '])
ax.set_xlabel('Observation')
ax.set_ylabel('Violent crimes Pop')
ax.set_title('Total Violent Crimes')

# %% 
## For your analysis, you should disregard the first five columns (state, county, community,
## communityname string and fold
## Data Cleaning
print('Before Cleaing first 5 colums')
#print(data.columns.values)
print(data.shape)
data = data.drop([' state ', ' county ', ' community ', ' communityname string', ' fold '], axis=1)
print('After Cleaning first 5 colums')
#print(data)
#print(data.columns.values)
print(data.shape)

# %%
## Searches for all ? and replaces it with nan
threshold = 300
data = data.replace('?', np.nan)
columns_to_remove = []
for column in data.columns.values:
    if np.sum(data[column].isna()) > threshold:
        # add this column to the list that should be removed
        columns_to_remove.append(column)

print('To Remove')
print(columns_to_remove)
print(len(columns_to_remove))    
# now lets remove them
data = data.drop(columns_to_remove, axis=1)
print(data.shape)

#%% Check for nan Data
print(np.sum(data.isna(), axis=1))
print(np.sum(np.sum(data.isna(), axis=1) > 0))
nans = data.isna()
print(type(nans))
nans.to_csv('nans.csv')
data_filtered = data.dropna(axis=0)
data_filtered.head()
print('Final dataset shape = {}'.format(data_filtered.shape))
print(data.iloc[11, :])
data.to_csv('question1.csv')
#%%
#Splitting for test &A validation data
training_data, testing_data = train_test_split(data_filtered, test_size=0.3, random_state=1)
testing_data, validating_data = train_test_split(testing_data, test_size=0.4, random_state=1)

x_var = [x for x in data.columns.values]
x_var.remove(' ViolentCrimesPerPop ')
y_var = ' ViolentCrimesPerPop '

# %% making x and y for training
y_train = np.array(training_data[y_var], dtype=np.float64)
x_train = np.array(training_data[x_var], dtype=np.float64)
x_train = sm.add_constant(x_train)

#%% Making x and y for testing
y_test = np.array(testing_data[y_var], dtype=np.float64)
x_test = np.array(testing_data[x_var], dtype=np.float64)
x_test = sm.add_constant(x_test)

#%% Making x and y for validating data
y_validate = np.array(validating_data[y_var], dtype=np.float64)
x_validate = np.array(validating_data[x_var], dtype=np.float64)
x_validate = sm.add_constant(x_validate)

#%% Creating model
print(data_filtered.isnull().sum())
model = sm.OLS(y_train, x_train)
model_fit = model.fit()

#%% Linear Model
model_fit = model.fit()
test_eval = model_fit.predict(x_test)
print('Linear Model Testing Data: RMSE = {}'.format(
    np.sqrt(mean_squared_error(y_test, test_eval))))
print(model_fit.summary())
plt.figure()
plt.plot(y_test, 'b', label='True Data')
plt.plot(test_eval, 'r--', label='Predicted Data')
plt.legend()
resid = y_test - test_eval
fig, ax = plt.subplots(figsize=(8,6))
sm.qqplot(resid, ax=ax, line='s')
plt.title('Plot for Linear Regression')
fig = plt.figure(figsize=[8,8])
ax = fig.add_subplot(1,1,1)
ax.hist(resid)
plt.show()

# %% Lasso Model
model_best = model.fit_regularized(alpha=0.001, L1_wt=1)
test_eval = model_best.predict(x_test)
print('L1 Lasso Model: alpha = 0.001, RMSE = {}'.format(
    np.sqrt(mean_squared_error(y_test, model_best.predict(x_test))))
)
plt.figure()
plt.plot(y_test, 'b', label='True Data')
plt.plot(test_eval, 'r--', label='Predicted Data')
plt.legend()

resid = y_test - test_eval

fig, ax=plt.subplots(figsize=(8,6))
sm.qqplot(resid, ax=ax, line='s')
plt.title('Q-Q Plot for Lasso Regression Using Best Model')
fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot(1, 1, 1)
ax.hist(resid)
plt.show()

# %% Lasso Ridge Regression
alpha = 1.001
model_L1_fit = model.fit_regularized(alpha=alpha, L1_wt=1)
pred = model_L1_fit.predict(x_validate)
print('L1: alpha = {}, RMSE = {}'.format(
    alpha, np.sqrt(mean_squared_error(y_validate, model_L1_fit.predict(x_validate)))
))

model_L2_fit = model.fit_regularized(aplha=alpha, L1_wt=0)
pred = model_L2_fit.predict(x_validate)
print('L2: alpha = {},  RMSE = {}'.format(
    alpha, np.sqrt(mean_squared_error(y_validate, model_L2_fit.predict(x_validate)))))

best_rmse = 10e12
best_alpha = []
best_L1_L2 = []

best_rmse_ridge = 10e12
best_alpha_ridge = []

init_alpha_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]
alpha_list_from_one  = np.linspace(1, 1000, 500)
alpha_list  = []
for alpha in init_alpha_list:
    alpha_list.append(alpha)
for alpha in alpha_list_from_one:
    alpha_list.append(alpha)

for alpha in alpha_list:
    model_cross_fit = model.fit_regularized(alpha=alpha, L1_wt=1)
    pred = model_cross_fit.predict(x_validate)
    rmse = np.sqrt(mean_squared_error(y_validate, model_cross_fit.predict(x_validate)))
    print('Lasso,  alpha = {},  RMSE = {}'.format(alpha, rmse))
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print("\n")

for alpha_ridge in alpha_list:
    model_cross_fit_ridge = model.fit_regularized(alpha=alpha_ridge, L1_wt=0)
    pred_ridge = model_cross_fit_ridge.predict(x_validate)
    rmse_ridge = np.sqrt(mean_squared_error(y_validate, model_cross_fit_ridge.predict(x_validate)))
    print('Ridge,  alpha = {},  RMSE = {}'.format(alpha_ridge, rmse_ridge))
    if rmse_ridge < best_rmse_ridge:
        best_rmse_ridge = rmse_ridge
        best_alpha_ridge = alpha_ridge

print('\nBest Model Lasso: alpha = {}, RMSE = {}'.format(best_alpha, best_rmse))
print('\nBest Model Ridge: alpha = {}, RMSE = {}'.format(best_alpha_ridge, best_rmse_ridge))

# %% Ridge Model
model_best = model.fit_regularized(aplha=0.001, L1_wt=0)
test_eval = model_best.predict(x_test)
print('L2 Ridge model: alpha = 0.001. RMSE = {}'.format(
    np.sqrt(mean_squared_error(y_validate, model_best.predict(x_validate)))))
plt.figure()
plt.plot(y_test, 'b', label='True Data')
plt.plot(test_eval, 'r--', label='Predicted Data')
plt.legend()
resid = y_test - test_eval
fig, ax = plt.subplots(figsize=(8,6))
sm.qqplot(resid, ax=ax, line='s')
plt.title('Q-Q Plot for Ridge Regression Using Best model')
fig = plt.figure(figsize=[8,8])
ax = fig.add_subplot(1,1,1)
ax.hist(resid)
plt.show()

# %%
# # Splitting data into Training, Validation and testing sets
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
# print("shape of original dataset :", data.shape)
# print("shape of input - training set", x_train.shape)
# print("shape of output - training set", y_train.shape)
# print("shape of input - testing set", x_test.shape)
# print("shape of output - testing set", y_test.shape)
# # need to do validation