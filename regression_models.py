import matplotlib.pyplot as plt
import numpy as np
import csv
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from os import environ

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

suppress_qt_warnings()

# Get the video data from files
ids = []
def get_start_video_data(path):
    f = open(path, "r")
    data = []
    for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        row = []
        # Only get videos which are also in the end video data
        if line[0].replace("\"", "") not in ids:
            continue

        row.append(int(line[3].replace("\"", "")))
        data.append(row[0])
    f.close()
    return data

def get_end_video_data(path):
    f = open(path, "r")
    data = []
    for line in csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
        # Only get data with less than 1,000,000 views to reduce noise
        if (int(line[3]) < 1000000):
            ids.append(line[0].replace("\"", ""))
            data.append(int(line[3].replace("\"", "")))
    return data


# Get the video data from files
y = get_end_video_data("Dataset/data_2023_06_24_00_04_40.csv")
x = get_start_video_data("Dataset/data_2023_06_09_23_01_13.csv")

# Convert the data to numpy arrays
x, y = np.array(x), np.array(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=15)

# Sort the data by the x values
# This will ensure the graphs are plotted nicely
x_test, y_test = (np.array(list(t)) for t in zip(*sorted(zip(x_test, y_test))))
x_train, y_train = (np.array(list(t)) for t in zip(*sorted(zip(x_train, y_train))))


# Create and fit the models
linear_model = LinearRegression()
linear_model.fit(x_train.reshape(-1, 1), y_train)

poly_model = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
poly_model.fit(x_train.reshape(-1, 1), y_train)

ridge_model = make_pipeline(PolynomialFeatures(degree=4), Ridge())
ridge_model.fit(x_train.reshape(-1, 1), y_train)

lasso_model = make_pipeline(PolynomialFeatures(degree=4), Lasso())
lasso_model.fit(x_train.reshape(-1, 1), y_train)

tree_model = tree.DecisionTreeRegressor(max_depth=3)
tree_model.fit(x_train.reshape(-1, 1), y_train)

rfr_model = RandomForestRegressor(max_depth=3)
rfr_model.fit(x_train.reshape(-1, 1), y_train)

gbr_model = GradientBoostingRegressor(max_depth=2)
gbr_model.fit(x_train.reshape(-1, 1), y_train)



x_plot = []
for i in range(0, 900000, 10000):
    x_plot.append(i)
x_plot = np.array(x_plot)

# Plot the data
fig = plt.figure()
ax = plt.axes()

ax.set_xlabel("Start video views")
ax.set_ylabel("End video views")
ax.set_title("Regression Models")

ax.scatter(x_train, y_train, color='purple', label='Training data')
ax.scatter(x_test, y_test, color='blue', label='Test data')
#ax.plot(x_plot, poly_model.predict(x_plot.reshape(-1, 1)), color='green', label='Polynomial Regression')
#ax.plot(x_plot, linear_model.predict(x_plot.reshape(-1, 1)), color='orange', label='Linear Regression')
#ax.plot(x_plot, ridge_model.predict(x_plot.reshape(-1, 1)), color='black', label='Ridge Regression')
#ax.plot(x_plot, lasso_model.predict(x_plot.reshape(-1, 1)), color='red', label='Lasso Regression')
ax.plot(x_plot, tree_model.predict(x_plot.reshape(-1, 1)), color='green', label='Decision Tree Regression')
ax.plot(x_plot, rfr_model.predict(x_plot.reshape(-1, 1)), color='orange', label='Random Forest Regression')
ax.plot(x_plot, gbr_model.predict(x_plot.reshape(-1, 1)), color='red', label='Gradient Boosting Regression')

ax.legend()
plt.show()

# Export the decision tree to a file
dot_data = tree.export_graphviz(tree_model, out_file=None, feature_names=['x'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree_model")


# Calculate performance data for each model
linear_slope = linear_model.coef_[0]
linear_intercept = linear_model.intercept_

poly_pred = poly_model.predict(x_test.reshape(-1, 1))
linear_pred = linear_model.predict(x_test.reshape(-1, 1))
tree_pred = tree_model.predict(x_test.reshape(-1, 1))
ridge_pred = ridge_model.predict(x_test.reshape(-1, 1))
lasso_pred = lasso_model.predict(x_test.reshape(-1, 1))
rfr_pred = rfr_model.predict(x_test.reshape(-1, 1))
gbr_pred = gbr_model.predict(x_test.reshape(-1, 1))

poly_mse = mean_squared_error(y_test, poly_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
tree_mse = mean_squared_error(y_test, tree_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)
rfr_mse = mean_squared_error(y_test, rfr_pred)
gbr_mse = mean_squared_error(y_test, gbr_pred)

poly_r2 = r2_score(y_test, poly_pred)
linear_r2 = r2_score(y_test, linear_pred)
tree_r2 = r2_score(y_test, tree_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
lasso_r2 = r2_score(y_test, lasso_pred)
rfr_r2 = r2_score(y_test, rfr_pred)
gbr_r2 = r2_score(y_test, gbr_pred)

poly_mae = mean_absolute_error(y_test, poly_pred)
linear_mae = mean_absolute_error(y_test, linear_pred)
tree_mae = mean_absolute_error(y_test, tree_pred)
ridge_mae = mean_absolute_error(y_test, ridge_pred)
lasso_mae = mean_absolute_error(y_test, lasso_pred)
rfr_mae = mean_absolute_error(y_test, rfr_pred)
gbr_mae = mean_absolute_error(y_test, gbr_pred)


# Produce an equation for the polynomial regression curve
# Create an instance of PolynomialFeatures
poly_features = PolynomialFeatures(degree=4)

# Fit and transform the x variable to polynomial features
x_poly = poly_features.fit_transform(x_train.reshape(-1, 1))

# Fit the polynomial regression model
poly_reg = LinearRegression()
poly_reg.fit(x_poly, y_train)

# Retrieve the coefficients from the model
coefficients = poly_reg.coef_

# Retrieve the feature names from the PolynomialFeatures object
feature_names = poly_features.get_feature_names(['x'])

# Build the polynomial equation string
poly_equation = " + ".join(f"{coeff:.25f} * {feat}" for coeff, feat in zip(coefficients, feature_names))

# Add the intercept term
poly_equation += f" + {poly_reg.intercept_:.10f}"


# Output performance data
print("Polynomial Regression")
print("Polynomial Equation:", poly_equation)
print("Mean squared error:", poly_mse)
print('coefficient of determination:', poly_r2)
print("Mean absolute error:", poly_mae)

print("Linear Regression")
print("Slope:", linear_slope)
print("Intercept:", linear_intercept)
print("Mean squared error:", linear_mse)
print('coefficient of determination:', linear_r2)
print("Mean absolute error:", linear_mae)

print("Decision Tree Regression")
print("Mean squared error:", tree_mse)
print('coefficient of determination:', tree_r2)
print("Mean absolute error:", tree_mae)

print("Ridge Regression")
print("Mean squared error:", ridge_mse)
print('coefficient of determination:', ridge_r2)
print("Mean absolute error:", ridge_mae)

print("Lasso Regression")
print("Mean squared error:", lasso_mse)
print('coefficient of determination:', lasso_r2)
print("Mean absolute error:", lasso_mae)

print("Random Forest Regression")
print("Mean squared error:", rfr_mse)
print('coefficient of determination:', rfr_r2)
print("Mean absolute error:", rfr_mae)

print("Gradient Boosting Regression")
print("Mean squared error:", gbr_mse)
print('coefficient of determination:', gbr_r2)
print("Mean absolute error:", gbr_mae)
