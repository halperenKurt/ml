import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

df_train = pd.read_csv("datasets/final_combined_data_v3.csv")
df_test = pd.read_csv("datasets/test.csv")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
check_df(df_train)
check_df(df_test)
df_train = df_train.dropna(subset=["Incident_Rate", "Case_Fatality_Ratio"])
df_train = df_train.drop(columns=["FIPS","Admin2","Last_Update","Province_State","Lat", "Long_","Combined_Key","Country_Region"])
df_train.columns
cat_cols, num_cols, cat_but_car = grab_col_names(df_train)
test_cat_cols, test_num_cols, test_cat_but_car = grab_col_names(df_test,cat_th=5,car_th=10)
missing_values_table(df_train, True)

for col in num_cols:
    print(col, check_outlier(df_train, col))

correlation_matrix(df_train,num_cols)

sc = StandardScaler()
df_train[num_cols] = sc.fit_transform(df_train[num_cols])

kmeans = KMeans(n_clusters=5, random_state=42)
df_train['Covid_Threat_Level'] = kmeans.fit_predict(df_train[num_cols])
df_train['Covid_Threat_Level'] = df_train['Covid_Threat_Level'] + 1

y = df_train["Covid_Threat_Level"]
X = df_train.drop(["Covid_Threat_Level"], axis=1)

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [#('LR', LogisticRegression()),
                    #('KNN', KNeighborsClassifier()),
                    #"SVC", SVC()),
                    #("CART", DecisionTreeClassifier()),
                    ("RF", RandomForestClassifier()),
                    #('Adaboost', AdaBoostClassifier()),
                    #('GBM', GradientBoostingClassifier()),
                    #('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        classifier.fit(X, y)
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y)

# knn_params = {"n_neighbors": range(2, 50)}
#
# cart_params = {'max_depth': range(1, 20),
#                "min_samples_split": range(2, 30)}
#
# rf_params = {"max_depth": [8, 15, None],
#              "max_features": [5, 7, "auto"],
#              "min_samples_split": [15, 20],
#              "n_estimators": [200, 300]}
#
# lightgbm_params = {"learning_rate": [0.01, 0.1],
#                    "n_estimators": [300, 500]}
#
# classifiers = [('KNN', KNeighborsClassifier(), knn_params),
#                ("CART", DecisionTreeClassifier(), cart_params),
#                ("RF", RandomForestClassifier(), rf_params),
#                 ('LightGBM', LGBMClassifier(), lightgbm_params)]


# def hyperparameter_optimization(X, y, cv=3, scoring="accuracy"):
#     print("Hyperparameter Optimization....")
#     best_models = {}
#     for name, classifier, params in classifiers:
#         print(f"########## {name} ##########")
#         cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
#         print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")
#
#         gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
#         final_model = classifier.set_params(**gs_best.best_params_)
#
#         cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
#         print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
#         print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
#         best_models[name] = final_model
#     return best_models
#
#
# best_models = hyperparameter_optimization(X, y)
print("*****************RANDOM FOREST*****************")
rf_model = RandomForestClassifier(max_depth= None, max_features= 5, min_samples_split= 15, n_estimators = 300).fit(X, y)
cv_results = cross_validate(rf_model, X, y, cv=3, scoring=["roc_auc","f1","accuracy"])
print("ROC_AUC :",cv_results["test_roc_auc"].mean().mean())
print("F1 SCORE : ",cv_results["test_f1"].mean())
print("ACCURACY : ",cv_results["test_accuracy"].mean())


