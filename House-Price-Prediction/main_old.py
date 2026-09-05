import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection  import cross_val_score
# from sklearn.preprocessing import OrdinalEncoder  # Uncomment if you prefer ordinal


#1. load the data 
housing = pd.read_csv("housing.csv")

#2. Create a stratified test set based on income category
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins= [0.,1.5,3.0,4.5,6.0,np.inf],
    labels =[1,2,3,4,5]
)

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat",axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat",axis=1)


# Work on a copy of training data
housing = strat_train_set.copy()   


#3.  separate predictions and labels
housing_label = housing["median_house_value"].copy()
housing = housing.drop("median_house_value",axis=1)


#4.  seprate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]


#5. pipe line
#numerical pipline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())

])

#categorical pipeline
cat_pipeline = Pipeline([
    # ("ordinal", OrdinalEncoder())  # Use this if you prefer ordinal encoding
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline,num_attribs),
    ("cat", cat_pipeline,cat_attribs)
])


#.6  Transform the data
housing_prepared =full_pipeline.fit_transform(housing)
# print(housing_prepared.shape)
# print("complete")


#.7 Train the models

#  Linear regression 
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_label)

#decision  tree 
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared,housing_label)


#random forest 
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared,housing_label)


# predict using training data
lin_preds = lin_reg.predict(housing_prepared)
tree_preds = tree_reg.predict(housing_prepared)
forest_preds = forest_reg.predict(housing_prepared)


# # calculate RMSE
# lin_rmse = mean_squared_error(housing_label,lin_preds ,squared=False)
# tree_rmse = mean_squared_error(housing_labels, tree_preds, squared=False)
# forest_rmse = mean_squared_error(housing_labels, forest_preds, squared=False)
 

 # calculate RMSE
# lin_rmse = np.sqrt(mean_squared_error(housing_label, lin_preds))
random_lin_rmse =-cross_val_score(lin_reg,housing_prepared,housing_label, scoring = "neg_root_mean_squared_error",cv=10)
# tree_rmse = np.sqrt(mean_squared_error(housing_label, tree_preds))
tree_rmses = -cross_val_score(tree_reg,housing_prepared,housing_label, scoring = "neg_root_mean_squared_error",cv=10)
# forest_rmse = np.sqrt(mean_squared_error(housing_label, forest_preds))
random_forest_rmses = -cross_val_score(forest_reg,housing_prepared,housing_label, scoring = "neg_root_mean_squared_error",cv=10)

# print("Linear Regression RMSE:", lin_rmse)
print(pd.Series(random_lin_rmse).describe())
# print("Decision Tree RMSE:", tree_rmses)
print(pd.Series(tree_rmses).describe())
# print(pd.Series(tree_rmses).describe()
print(pd.Series(random_forest_rmses ).describe())