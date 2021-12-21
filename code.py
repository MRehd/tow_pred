# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from math import sqrt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm
from scipy import stats

# Importing data
training = pd.read_csv('training.csv')
validation = pd.read_csv('validation.csv')

# Replacing null values
training.replace('(null)', np.nan, inplace = True)
validation.replace('(null)', np.nan, inplace = True)

# Defining features and target variables for the regression model
features = ['ActualFlightTime', 'ActualTotalFuel', 'FLownPassengers', 'BagsCount', 'FlightBagsWeight']
target = ['ActualTOW']

# Dropping NaN values and converting the regression variables data types to float
training.dropna(inplace=True)
for column in features + target:
    training[column] = training[column].astype(float)

# Filtering outliers
training = training[(training['FLownPassengers'] > 24) & (training['FlightBagsWeight'] < 2500)]

# Replacing missing values from the validation set with the median of each variable
for column in features:
    validation[column] = validation[column].astype(float)
    median = validation[column].median()
    if validation[column].isnull().any() == True:
        validation[column] = validation[column].replace(np.nan, median)

# Heatmap of correlation between variables
plt.figure(figsize=(6,4))
corr = training[features+target].corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, 1)] = True
sns.heatmap(corr, mask=mask, annot=True, cmap="Reds")
plt.show()
print('Correlation Heatmap')

# Redefining features for the regression models (removing highly correlated variables)
features = ['ActualTotalFuel', 'FLownPassengers', 'FlightBagsWeight']
print(f'Selected features: {features}')

print('Variables relationship and distribution')
plt.figure(figsize=(4,2))
sns.pairplot(training[features + target]) #['ActualFlightTime', 'ActualTotalFuel', 'FLownPassengers', 'BagsCount', 'FlightBagsWeight', 'ActualTOW']
plt.show()

# Training the regression model with standardized variables and checking the Mean Squared Error, coefficients and interception
X_normalized = training[features].values
X_normalized = StandardScaler().fit_transform(X_normalized)
y_normalized = training[target].values

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_normalized, y_normalized, test_size = 0.33, random_state = 324)

regressor_n = LinearRegression()
regressor_n.fit(X_train_n, y_train_n)

y_prediction_n = regressor_n.predict(X_test_n)

RMSE_n = sqrt(mean_squared_error(y_true = y_test_n, y_pred = y_prediction_n))
print(RMSE_n)

f_stat = sm.add_constant(pd.DataFrame(X_normalized, columns = features))
results = sm.OLS(pd.DataFrame(y_normalized, columns = target), f_stat).fit()
A = np.identity(len(results.params))
A = A[1:,:]
print(f'F_test results: {results.f_test(A)}')

# Normality of residuals
norm_res = y_test_n - y_prediction_n
print(f'Normality of residuals: {np.mean(norm_res)} - This value should be close to 0. The variables used provided the smallest value.')


# Checking for Homoscedasticity
plt.figure(figsize=(6,4))
plt.xlabel('Residual')
plt.ylabel('Predictions')
plt.grid()
sns.scatterplot(x = [i[0] for i in norm_res], y = [i[0] for i in y_prediction_n])
plt.show()
print('No correlation between the predictions and the residuals')
correlation, p_value = stats.pearsonr([i[0] for i in norm_res], [i[0] for i in y_prediction_n])
print(correlation, p_value)

print(f'the Mean Squared Error is: {RMSE_n}')
print(f'the coefficients are: {regressor_n.coef_}')
print(f'the interception is: {regressor_n.intercept_}')


# Testing the regression model on the validation dataset and checking statistics
X_v_n = validation[features].values
X_v_n = StandardScaler().fit_transform(X_v_n)
y_v_n = regressor_n.predict(X_v_n)

validation['prediction'] = y_v_n

print('Description of the predictions')
print(validation['prediction'].describe())

print('Description of the training TOW')
print(training['ActualTOW'].describe())

# Visualizing the regression model
def regression_plot(dataset, var1, var2, axs):
    linear_regressor_1 = LinearRegression().fit(np.array(dataset[var1]).reshape(-1, 1), dataset[var2])
    pred_1 = linear_regressor_1.predict(np.array(dataset[var1]).reshape(-1, 1))

    sns.set_theme(style = "white")
    sns.scatterplot(x = dataset[var1], y = dataset[var2], ax=axs)
    sns.lineplot(x = dataset[var1], y = pred_1.flatten(), color = 'blue', ax=axs)
    return

fig, ax = plt.subplots(2, 3)
plt.tight_layout()
fig.set_figheight(8)
fig.set_figwidth(8)

print('Visualizing the regression model')
regression_plot(validation, 'FLownPassengers', 'prediction', ax[0,0])
regression_plot(validation, 'ActualTotalFuel', 'prediction', ax[0,1])
regression_plot(validation, 'FlightBagsWeight', 'prediction', ax[0,2])

regression_plot(training, 'FLownPassengers', 'ActualTOW', ax[1,0])
regression_plot(training, 'ActualTotalFuel', 'ActualTOW', ax[1,1])
regression_plot(training, 'FlightBagsWeight', 'ActualTOW', ax[1,2])

plt.show()

fig2, ax2 = plt.subplots(1, 2)

plt.tight_layout()
sns.histplot(y_v_n, stat='probability', legend=False, ax = ax2[1], bins=30, kde=True)
ax2[1].set_title("Predicted TOW")

sns.histplot(y_normalized, stat='probability', legend=False, ax = ax2[0], bins=30, kde=True)
ax2[0].set_title("Training TOW")

plt.show()