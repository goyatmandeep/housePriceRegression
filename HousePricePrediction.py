# -*- coding: utf-8 -*-
"""
California House Prices Prediction

"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def seperate_label(train, test):
    test_label = test["median_house_value"]
    train_label = train["median_house_value"]
    train.drop(["median_house_value"], axis=1, inplace=True)
    test.drop(["median_house_value"], axis=1, inplace=True)
    return train, test, train_label, test_label

def mse(model, x, y):
    temp = mean_squared_error(model.predict(x), y)
    return np.sqrt(temp)

def cvs(model, x, y, cv=5):
    score = cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=cv)
    score = np.sqrt(-score)
    #print("\nCross Validation Score "+str(score))
    #print("Variance "+str(score.var()))
    return score.mean()



path = r"housing.csv"
housing_data = pd.read_csv(path) #Load the csv file.
housing_data.hist(bins=40, figsize=(15, 10))

#Preprocessing
print("\t\t\tPreprocessing Data")


imputer = SimpleImputer(strategy="median")
temp_data = housing_data.drop("ocean_proximity", axis=1)
imputer.fit(temp_data)
transformed_data = imputer.transform(temp_data)
housing_new = pd.DataFrame(transformed_data, columns=temp_data.columns)
housing_new["ocean_proximity"] = housing_data["ocean_proximity"]


housing_new["population_per_house"] = housing_new["population"]/housing_new["households"]
housing_new["rooms_per_house"] = housing_new["total_rooms"]/housing_new["households"]


encoder = LabelBinarizer()
ocean_encoded = encoder.fit_transform(housing_new["ocean_proximity"])
ocean_encoded = pd.DataFrame(ocean_encoded)
housing_new.drop(["ocean_proximity"], axis=1, inplace=True)
housing_new = housing_new.join(ocean_encoded)

'''def split_training_set(data_set=housing_data, ratio=0.2):
    np.random.seed(0)
    shuffle_ind = np.random.permutation(len(housing_data))
    index = int(ratio*len(data_set))
    train_ind = shuffle_ind[index:]
    test_ind = shuffle_ind[:index]
    return housing_data.iloc[train_ind], housing_data.iloc[test_ind]
    
train_set, test_set = split_training_set(housing_new, 0.2)
'''

sns.pairplot(housing_new[['median_house_value','median_income','total_rooms','housing_median_age']])
housing_new["income_cat"] = np.ceil(housing_new["median_income"]/2)
housing_new["income_cat"].where(housing_new["income_cat"]<5, 5, inplace=True)

housing_corr = housing_new.corr()
correlation = housing_corr["median_house_value"].sort_values(ascending=False)


temp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

for train_ind, test_ind in temp.split(housing_new, housing_new["income_cat"]):
    housing_train = housing_new.loc[train_ind]
    housing_test = housing_new.loc[test_ind]
    
for cat in (housing_train, housing_test):
    cat.drop(["income_cat"], axis=1, inplace=True)


housing_train, housing_test, train_label, test_label = seperate_label(housing_train, housing_test)

scaler = StandardScaler()
scaler.fit_transform(housing_train)
scaler.fit_transform(housing_test)

#Training Model
print("\t\t\tTraining Models")

#Linear Regression
lin = LinearRegression()
lin.fit(housing_train, train_label)
s1 = cvs(lin, housing_train, train_label, 3)
print("Linear Regression MSE " +str(mse(lin, housing_train, train_label)))

#SVM
cs = [1, 10, 30]
gammas = [0.01, 0.03, 0.1]
svm_rbf = svm.SVR(degree=2)
grid = GridSearchCV(svm_rbf, {'C':cs, 'gamma':gammas}, cv=3)
grid.fit(housing_train, train_label)
#print(grid.best_params_)
s2 = cvs(svm_rbf, housing_train, train_label, 3)
print("\nSVM (RBF kernel) MSE "+str(mse(grid.best_estimator_, housing_train, train_label)))


#Polynomial Regression
lin2 = LinearRegression()    
poly = PolynomialFeatures(degree=2)
x = poly.fit_transform(housing_train)
y = poly.fit_transform(housing_test)
lin2.fit(x, train_label)
s3 = cvs(lin2, housing_train, train_label, 3)
print("Polynomial Regression MSE "+str(mse(lin2, y, test_label)))

#RandomForestRegressor
rfr = RandomForestRegressor()
mf = [6, 8]
ne = [30, 35, 40]
grid = GridSearchCV(rfr, [{'max_features':mf, 'n_estimators':ne}], cv=2)
grid.fit(housing_train, train_label)

rfr_final = grid.best_estimator_

#On test set
rfr_final.fit(housing_train, train_label)
s3 = cvs(rfr_final, housing_train, train_label, 3)
print("\nMSE on test set (Random Forest) "+str(mse(rfr_final, housing_test, test_label)))
print(grid.best_params_)
