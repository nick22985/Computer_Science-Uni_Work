
# model_fit = model.fit()
# test_eval = model_fit.predict(x_test)
# print('Linear Model Testing Data: RMSE = {}'.format(
#     numpy.sqrt(mean_squared_error(y_test, test_eval))))
# print(model_fit.summary())
# plt.figure()
# plt.plot(y_test, 'b', label='True Data')
# plt.plot(test_eval, 'r--', label='Predicted Data')
# plt.legend()
# resid = y_test - test_eval
# fig, ax = plt.subplots(figsize=(8,6))
# sm.qqplot(resid, ax=ax, line='s')
# plt.title('Plot for Linear Regression')
# fig = plt.figure(figsize=[8,8])
# ax = fig.add_subplot(1,1,1)
# ax.hist(resid)
# plt.show()


# linear Laso
# alpha = 1.001
# model_L1_fit = model.fit_regularized(alpha=alpha, L1_wt=1)
# pred = model_L1_fit.predict(x_validate)
# print('L1: alpha = {}, RMSE = {}'.format(
#     alpha, numpy.sqrt(mean_squared_error(y_validate, model_L1_fit.predict(x_validate)))
# ))

# model_L2_fit = model.fit_regularized(aplha=alpha, L1_wt=0)
# pred = model_L2_fit.predict(x_validate)
# print('L2: alpha = {},  RMSE = {}'.format(
#     alpha, numpy.sqrt(mean_squared_error(y_validate, model_L2_fit.predict(x_validate)))))

# best_rmse = 10e12
# best_alpha = []
# best_L1_L2 = []

# best_rmse_ridge = 10e12
# best_alpha_ridge = []

# init_alpha_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]
# alpha_list_from_one  = numpy.linspace(1, 1000, 500)
# alpha_list  = []
# for alpha in init_alpha_list:
#     alpha_list.append(alpha)
# for alpha in alpha_list_from_one:
#     alpha_list.append(alpha)

# for alpha in alpha_list:
#     model_cross_fit = model.fit_regularized(alpha=alpha, L1_wt=1)
#     pred = model_cross_fit.predict(x_validate)
#     rmse = numpy.sqrt(mean_squared_error(y_validate, model_cross_fit.predict(x_validate)))
#     print('Lasso,  alpha = {},  RMSE = {}'.format(alpha, rmse))
#     if rmse < best_rmse:
#         best_rmse = rmse
#         best_alpha = alpha

# print("\n")

# for alpha_ridge in alpha_list:
#     model_cross_fit_ridge = model.fit_regularized(alpha=alpha_ridge, L1_wt=0)
#     pred_ridge = model_cross_fit_ridge.predict(x_validate)
#     rmse_ridge = numpy.sqrt(mean_squared_error(y_validate, model_cross_fit_ridge.predict(x_validate)))
#     print('Ridge,  alpha = {},  RMSE = {}'.format(alpha_ridge, rmse_ridge))
#     if rmse_ridge < best_rmse_ridge:
#         best_rmse_ridge = rmse_ridge
#         best_alpha_ridge = alpha_ridge

# print('\nBest Model Lasso: alpha = {}, RMSE = {}'.format(best_alpha, best_rmse))
# print('\nBest Model Ridge: alpha = {}, RMSE = {}'.format(best_alpha_ridge, best_rmse_ridge))