#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels
from statsmodels import api as sm
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge

## Q1
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
y_val = np.array(validating_data[y_var], dtype=np.float64)
x_val = np.array(validating_data[x_var], dtype=np.float64)
x_val = sm.add_constant(x_val)

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
ax.set_title('Coefficients')
plt.show()

#%% Linear model Stats
linear = LinearRegression(fit_intercept = False).fit(x_train, y_train)
fig = plt.figure(figsize=[25, 16])
ax = fig.add_subplot(4, 1, 1)
ax.bar(range(len(linear.coef_)), linear.coef_)
ax.set_title('Coefficients')
ax = fig.add_subplot(4, 1, 2)
ax.plot(linear.predict(x_train), label='Predicted')
ax.plot(y_train, label='Actual')
ax.set_title('Training Data')
ax.legend()
ax = fig.add_subplot(4, 1, 3)
ax.plot(linear.predict(x_val), label='Predicted')
ax.plot(y_val, label='Actual')
ax.set_title('Validation Data')
ax.legend()
ax = fig.add_subplot(4, 1, 4)
ax.plot(linear.predict(x_test), label='Predicted')
ax.plot(y_test, label='Actual')
ax.set_title('Testing Data')
ax.legend();

#%% Lasso
glenn_1 = Ridge(fit_intercept=False, alpha=0.001).fit(X = x_train, y = y_train)
glenn_2 = Ridge(fit_intercept=False, alpha=0.01).fit(X = x_train, y = y_train)
glenn_3 = Ridge(fit_intercept=False, alpha=10).fit(X = x_train, y = y_train)

fig = plt.figure(figsize=[25, 16])
ax = fig.add_subplot(4, 1, 1)
w = 0.2
pos = np.arange(0, len(linear.coef_), 1)
ax.bar(pos - w*2, linear.coef_, width=w, label='linear')
ax.bar(pos - w, glenn_1.coef_, width=w, label='alpha=0.01')
ax.bar(pos, glenn_2.coef_, width=w, label='alpha=2.5')
ax.bar(pos + w, glenn_3.coef_, width=w, label='alpha=10')
ax.legend()
ax.set_title('Coefficients')
ax = fig.add_subplot(4, 1, 2)
ax.plot(linear.predict(x_train), label='linear')
ax.plot(glenn_1.predict(x_train), label='lambda=0.01')
ax.plot(glenn_2.predict(x_train), label='lambda=2.5')
ax.plot(glenn_3.predict(x_train), label='lambda=10')
ax.plot(y_train, label='Actual')
ax.set_title('Training Data')
ax.legend()
ax = fig.add_subplot(4, 1, 3)
ax.plot(linear.predict(x_val), label='linear')
ax.plot(glenn_1.predict(x_val), label='lambda=0.01')
ax.plot(glenn_2.predict(x_val), label='lambda=2.5')
ax.plot(glenn_3.predict(x_val), label='lambda=10')
ax.plot(y_val, label='Actual')
ax.set_title('Validation Data')
ax.legend()
ax = fig.add_subplot(4, 1, 4)
ax.plot(linear.predict(x_test), label='linear')
ax.plot(glenn_1.predict(x_test), label='lambda=0.01')
ax.plot(glenn_2.predict(x_test), label='lambda=2.5')
ax.plot(glenn_3.predict(x_test), label='lambda=10')
ax.plot(y_test, label='Actual')
ax.set_title('Testing Data')
ax.legend();


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


# %% Ridge Model
model_best = model.fit_regularized(alpha=0.001, L1_wt=0)
test_eval = model_best.predict(x_test)
print('L2 Ridge model: alpha = 0.001. RMSE = {}'.format(
    np.sqrt(mean_squared_error(y_val, model_best.predict(x_val)))))
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

#%%
# Q2
