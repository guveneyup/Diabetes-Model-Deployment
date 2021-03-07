# Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Data set settings

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# READ DATA SET

df = pd.read_csv("dataset/diabetes.csv")

# FUNCTIONS

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.05)
    quartile3 = dataframe[col_name].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# EDA
check_df(df)
df.describe().T

# Missing Values
zero_columns = [i for i in df.columns if (df[i].min() == 0 and i not in ["Pregnancies", "Outcome"])]
zero_columns


for i in zero_columns:
    df[[i]] = df[[i]].replace(0, np.NaN)


df.isnull().sum()

# fill in missing observations

for i in zero_columns:
    df[i] = df[i].fillna(df.groupby("Outcome")[i].transform("median"))

# OUTLIER VALUES

num_cols = [i for i in df.columns if df[i].dtypes != "O" and df[i].nunique() > 10]

for i in num_cols:
    print(i, check_outlier(df, i))


replace_with_thresholds(df, "Insulin")
replace_with_thresholds(df, "SkinThickness")


for i in num_cols:
    print(i, check_outlier(df, i))

# Feature Engineering
df["PREG_AGE"] = df["Pregnancies"] * df["Age"]

df["Glucose_BMI"] = df["Glucose"] * df["BMI"]

df["Insulin_Glucose"] = df["Insulin"] * df["Glucose"]

df["Insulin_BMI"] = df["Insulin"] * df["BMI"]

df["INSULİN_AGE"] = df["Insulin"] * df["Age"]


def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))

# LABEL ENCODING

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# ONE-HOT ENCODING

df = pd.get_dummies(df, drop_first=True)

df.shape

# SET VARIABLE

df.drop("Pregnancies", axis=1, inplace=True)
df.to_pickle("DiabetesFinalSet_pickle.pkl")

y = df["Outcome"]
x = df.drop("Outcome", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

x_train.shape
y_train.shape

# MODEL
cls = DecisionTreeClassifier()

model = cls.fit(x_train, y_train)
proba = model.predict_proba(x_test)[:, 1]
pred = model.predict(x_test)

roc_auc_score(y_test, proba)
accuracy_score(y_test, pred)
recall_score(y_test, pred)
precision_score(y_test, pred)
model.get_params()

# MODEL TUNING

cart_model = DecisionTreeClassifier(random_state=17)

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(1, 16)}


cart_cv = GridSearchCV(cart_model, cart_params, cv=5, n_jobs=-1, verbose=True)
cart_cv.fit(x_train, y_train)
cart_cv.best_params_

# FINAL MODEL
cart_tuned = DecisionTreeClassifier(**cart_cv.best_params_).fit(x_train, y_train)
y_pred = cart_tuned.predict(x_test)
y_prob = cart_tuned.predict_proba(x_test)[:, 1]

roc_auc_score(y_test, y_prob)
accuracy_score(y_test, y_pred)
recall_score(y_test, y_pred)
precision_score(y_test, y_pred)


# VARIABLE IMPORTANCE LEVEL
def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')



plot_importance(cart_tuned, x_train)

# MODEL TO SAVE

pickle.dump(cart_tuned, open('cart_model.pkl', 'wb'))

