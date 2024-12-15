import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)

def covid19_eda(dataframe):
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
    def grab_col_names(dataframe, cat_th=10, car_th=20):
        """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

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
    def correlation_matrix(df, cols):
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
        plt.show(block=True)

    def missing_values_table(dataframe, na_name=False):
        na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

        n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
        ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
        missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
        print(missing_df, end="\n")

        if na_name:
                return na_columns

    check_df(dataframe)
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    missing_values_table(dataframe, True)
    correlation_matrix(dataframe, num_cols)

def train_data_prep(dataframe):
    # Helper function that separates columns into categorical and numerical
    def grab_col_names(dataframe, cat_th=10, car_th=20):
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        return cat_cols, num_cols, cat_but_car

    dataframe = dataframe.dropna(subset=["Incident_Rate", "Case_Fatality_Ratio"])
    dataframe = dataframe.drop(columns=["FIPS", "Admin2", "Last_Update", "Province_State", "Lat", "Long_", "Combined_Key", "Country_Region"])

    # standardization
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    sc = StandardScaler()
    dataframe[num_cols] = sc.fit_transform(dataframe[num_cols])

    # Add 'Covid_Threat_Level' column by clustering by KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    dataframe['Covid_Threat_Level'] = kmeans.fit_predict(dataframe[num_cols])
    dataframe['Covid_Threat_Level'] = dataframe['Covid_Threat_Level'] + 1

    X = dataframe.drop(["Covid_Threat_Level"], axis=1)
    y = dataframe["Covid_Threat_Level"]

    return X, y


def prepare_test_data(dataframe):
    # Helper function that separates columns into categorical and numerical
    def grab_col_names(dataframe, cat_th=5, car_th=10):
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

        return cat_cols, num_cols, cat_but_car

    columns_to_fill = ['Incident_Rate', 'Case_Fatality_Ratio']
    dataframe[columns_to_fill] = dataframe[columns_to_fill].fillna(dataframe[columns_to_fill].mean())
    #dataframe = dataframe.dropna(subset=["Incident_Rate", "Case_Fatality_Ratio"])

    # Drop unnecessary columns
    dataframe = dataframe.drop(columns=["FIPS", "Admin2", "Last_Update", "Province_State", "Lat", "Long_", "Combined_Key", "Country_Region"])
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    # Standardize numerical columns
    sc = StandardScaler()
    dataframe[num_cols] = sc.fit_transform(dataframe[num_cols])

    # Add 'Covid_Threat_Level' column as NaN
    if "Covid_Threat_Level" not in dataframe.columns:
        dataframe["Covid_Threat_Level"] = np.nan

    y = dataframe["Covid_Threat_Level"]
    X = dataframe.drop(["Covid_Threat_Level"], axis=1)


    return X,y


def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                    ('KNN', KNeighborsClassifier()),
                    ("RF", RandomForestClassifier()),
                   ]

    for name, classifier in classifiers:
        classifier.fit(X, y)
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)} ({name}) ")

#best parameters according to hyperparameter optimization
knn_params = {"n_neighbors": [3]}

rf_params = {
    "max_depth": [None],
    "max_features": [5],
    "min_samples_split": [15],
    "n_estimators": [300]
}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
                ("RF", RandomForestClassifier(), rf_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="accuracy"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

#ensemble modeling 
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    return voting_clf

#predict the test dataset
def predict_test(test_file_path, model_file_path):
    df_test = pd.read_csv(test_file_path)
    X_test, y_test = prepare_test_data(df_test)

    voting_clf = joblib.load(model_file_path)
    predictions = voting_clf.predict(X_test)
    return predictions
#pipeline
def main():
    df_train = pd.read_csv("datasets/final_data.csv")
    df_test = pd.read_csv("datasets/test_data.csv")
    covid19_eda(df_train)
    X_train, y_train = train_data_prep(df_train)
    base_models(X_train, y_train , scoring="accuracy")
    best_models = hyperparameter_optimization(X_train, y_train)
    voting_clf = voting_classifier(best_models, X_train, y_train)
    joblib.dump(voting_clf, "voting_covid19_clf.pkl")
    predictions = predict_test("datasets/test_data.csv", "voting_covid19_clf.pkl")
    df_test["Covid_Threat_Level"] = predictions
    df_test.to_csv("test_data_predict.csv", index=False)
    print("Predictions saved")
    return voting_clf

if __name__ == "__main__":
    print("Covid Threat Prediction starts")
    main()


