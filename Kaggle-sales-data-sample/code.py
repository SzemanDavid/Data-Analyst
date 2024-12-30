# Loading my Google Drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

!pip install openpyxl

file_path = '/content/drive/MyDrive/Python/Sales-data/sales_data_sample.csv'
data = pd.read_csv(file_path, encoding='latin1')

#the first 5 line
print(data.head())

# Searching for missing data
print(data.isnull().sum())

#fill the missing data with 0 (because there are just the place names missing)
data.fillna(0, inplace=True)
print(data.isnull().sum())

#delete the columns that are not neccesary
columns_to_drop = ['ORDERNUMBER', 'ORDERLINENUMBER', 'QTR_ID', 'MSRP',
                   'PRODUCTCODE', 'CUSTOMERNAME', 'PHONE', 'ADDRESSLINE1',
                   'ADDRESSLINE2','STATE', 'POSTALCODE', 'TERRITORY',
                   'CONTACTLASTNAME', 'CONTACTFIRSTNAME']
if all(col in data.columns for col in columns_to_drop):
    data.drop(columns_to_drop, axis=1, inplace=True)
    print("Columns dropped successfully.")
else:
    print("Columns have already been deleted.")

# type of the datas and save to csv
print(data.dtypes)

path='/content/drive/MyDrive/Python/Sales-data/salesdata.csv'
data.to_csv(path, index=False)
print(f"File successfully saved to {path}")

#use the new saved data
path='/content/drive/MyDrive/Python/Sales-data/salesdata.csv'
data = pd.read_csv(path, encoding='latin1')

##########################################################
#VISUALIZATION

#sales distribution
plt.figure(figsize=(12, 6))
sns.histplot(data['SALES'], kde=True, color='green')
plt.xlabel('Sales amount')
plt.ylabel('Frequency')
plt.title('Sales distribution')
plt.show()

#boxplot for the outliers and the most expensive sale
sns.boxplot(x=data['SALES'], color='green')
plt.xlabel('Sales amount')
plt.title('Sales boxplot')
plt.show()
print("The most expensive sale:")
print(data.loc[data['SALES'].idxmax()])

#Product purchase distribution
sns.countplot(x='PRODUCTLINE', data=data, color='green', order=data['PRODUCTLINE'].value_counts().index)
plt.title('Product purchase distribution')
plt.xlabel('')
plt.ylabel('Number of purchases')
plt.xticks(rotation=45)
plt.show()

#Monthly purchase count
#orderdate column to datetime form
data['ORDERDATE'] = pd.to_datetime(data['ORDERDATE'])
data['Month'] = data['ORDERDATE'].dt.to_period('M')
monthly_sales_count = data.groupby('Month').size()
plt.figure(figsize=(10, 6))
monthly_sales_count.plot(kind='line', marker='o', color='green')
plt.title('Monthly purchase count')
plt.xlabel('Month')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45)
plt.show()

#Sales amount boxplot by products
plt.figure(figsize=(10, 6))
sns.boxplot(x='PRODUCTLINE', y='SALES', data=data, color='green', order=data['PRODUCTLINE'].value_counts().index)
plt.title('Sales boxplot by product')
plt.xlabel('Product')
plt.ylabel('Sales amount')
plt.xticks(rotation=45)
plt.show()

#Sales by country
plt.figure(figsize=(12, 6))
sns.countplot(x='COUNTRY', data=data, color='green', order=data['COUNTRY'].value_counts().index)
plt.title('Sales by country')
plt.xlabel('Country')
plt.ylabel('Number of purchases')
plt.xticks(rotation=45)
plt.show()

#########################################################################
#CLASSIFICATION
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform


label_encoders = {}
for col in ['STATUS', 'PRODUCTLINE', 'CITY', 'COUNTRY', 'DEALSIZE']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

#X and y_class definiton
X = data[['QUANTITYORDERED', 'PRICEEACH', 'MONTH_ID', 'YEAR_ID', 'STATUS', 'PRODUCTLINE', 'CITY', 'COUNTRY', 'DEALSIZE']]

data['Sales_Category'] = pd.qcut(data['SALES'], q=3, labels=['Low', 'Medium', 'High'])
y_class = data['Sales_Category']


X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

#model: RandomForest Classification
randomforest = RandomForestClassifier(n_estimators=20, verbose=0, random_state=42)
hyperparameter = {
    "max_depth": [3, 4, 5, 6, 10, 15],
    "n_estimators": [50, 100, 200],
    "criterion": ['gini', "entropy", "log_loss"],
    "min_samples_split": loguniform(0.01, 1.0),
    "max_samples": uniform(0.1, 0.9)}

search_rf = RandomizedSearchCV(
    estimator=randomforest,
    param_distributions=hyperparameter,
    n_iter=100,
    scoring="accuracy",
    cv=5,
    random_state=42,
    verbose=10)

search_rf.fit(X_train_class, y_train_class)

#result
print()
print("**Best Parameters:**", search_rf.best_params_)
best_model = search_rf.best_estimator_
y_pred = best_model.predict(X_test_class)
print("\nAccuracy:", accuracy_score(y_test_class, y_pred))
print("\nClassification Report:\n", classification_report(y_test_class, y_pred))

########################################################################
#REGRESSION
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
#use the new saved data
path='/content/drive/MyDrive/Python/Sales-data/salesdata.csv'
data = pd.read_csv(path, encoding='latin1')
data.info()
regdata=data
regdata['PRODUCTLINE'] = data['PRODUCTLINE'].astype('category').cat.codes
regdata['CITY'] = data['CITY'].astype('category').cat.codes
regdata['COUNTRY'] = data['COUNTRY'].astype('category').cat.codes
regdata['DEALSIZE'] = data['DEALSIZE'].astype('category').cat.codes
regdata['STATUS'] = data['STATUS'].astype('category').cat.codes
regdata.info()

X_reg = regdata[['QUANTITYORDERED', 'PRICEEACH', 'MONTH_ID', 'YEAR_ID', 'STATUS', 'PRODUCTLINE', 'CITY', 'COUNTRY', 'DEALSIZE']]
y_reg = regdata['SALES']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

decisiontree = DecisionTreeRegressor(random_state=42)
parameter = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 4, 6, 8, 12],
    'min_samples_leaf': [0.1, 2, 4, 6, 8, 12],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error']}

search_dt = GridSearchCV(
    estimator=decisiontree,
    param_grid=parameter,
    scoring='r2',
    cv=5,
    verbose=3,
    n_jobs=-1)

search_dt.fit(X_train_reg, y_train_reg)

#result
print()
print("Best Parameters:", search_dt.best_params_)
best_model = search_dt.best_estimator_
y_pred_reg = best_model.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

feature_importances = best_model.feature_importances_
sns.barplot(x=feature_importances, y=X_reg.columns)
plt.title("Feature Importance")
plt.show()
