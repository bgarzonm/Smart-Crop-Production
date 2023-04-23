import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVR

sys.path.append("..")
import utility.plot_settings

df = pd.read_csv("../data/suelos_data.csv")

df = df[(df["Departamento"] == "TOLIMA") & (df["Cultivo"] == "Arroz")]

df.columns

select_columns = [
    "Topografia",
    "Drenaje",
    "Riego",
    "pH agua:suelo 2,5:1,0",
    "Fósforo (P) Bray II mg/kg",
    "Azufre (S) Fosfato monocalcico mg/kg",
    "Calcio (Ca) intercambiable cmol(+)/kg",
    "Magnesio (Mg) intercambiable cmol(+)/kg",
    "Potasio (K) intercambiable cmol(+)/kg",
    "Sodio (Na) intercambiable cmol(+)/kg",
]

df = df[select_columns]


new_column_names = {
    "Topografia": "Topography",
    "Drenaje": "Drainage",
    "Riego": "Irrigation",
    "pH agua:suelo 2,5:1,0": "pH",
    "Fósforo (P) Bray II mg/kg": "P",
    "Azufre (S) Fosfato monocalcico mg/kg": "S",
    "Calcio (Ca) intercambiable cmol(+)/kg": "Ca",
    "Magnesio (Mg) intercambiable cmol(+)/kg": "Mg",
    "Potasio (K) intercambiable cmol(+)/kg": "K",
    "Sodio (Na) intercambiable cmol(+)/kg": "Na",
}

df = df.rename(columns=new_column_names)


def plot_and_encode(df, column_name):
    # Plot the distribution of the column
    df[column_name].value_counts().plot(kind="bar")
    plt.title(f"Distribution of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Count")
    plt.show()

    # Apply one-hot encoding to the column
    one_hot = pd.get_dummies(df[column_name], prefix=column_name)
    del df[column_name]
    df = pd.concat([df, one_hot], axis=1)
    return df


# Transform Topography
df = plot_and_encode(df, "Topography")

# Transform Drainage
df = plot_and_encode(df, "Drainage")

# Transform Irrigation
df = plot_and_encode(df, "Irrigation")


df.info()


def convert_to_numeric(df, column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    return df


soil_params = ["pH", "P", "S", "Ca", "Mg", "K", "Na"]

for column in soil_params:
    df = convert_to_numeric(df, column)


def plot_histograms(df, columns):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()

    for i, column in enumerate(columns):
        sns.histplot(data=df, x=column, kde=True, bins=20, edgecolor="black", ax=axs[i])
        axs[i].set_title(column, size=16)
        axs[i].set_xlabel("Value", size=12)
        axs[i].set_ylabel("Count", size=12)

    plt.suptitle("Histograms of Soil Analysis Parameters", size=24)
    plt.tight_layout()
    plt.show()


soil_params = ["pH", "P", "S", "Ca", "Mg", "K", "Na"]
plot_histograms(df, soil_params)


def plot_soil_params_boxplot(df, soil_params):

    fig, axs = plt.subplots(nrows=1, ncols=len(soil_params), figsize=(25, 5))

    # Loop through each soil parameter and create a box plot
    for i, param in enumerate(soil_params):
        sns.boxplot(data=df, y=param, ax=axs[i], color="#ffb03b")
        axs[i].set_title(param, fontsize=16)
        axs[i].set_ylabel("")

    # Set the main title for the plot
    fig.suptitle("Boxplots of Soil Parameters", fontsize=20)

    # Adjust the spacing between subplots
    fig.subplots_adjust(wspace=0.5)

    # Show the plot
    plt.show()


plot_soil_params_boxplot(df, soil_params)


def detect_outlier(x):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    IQR = q3 - q1
    lower_limit = q1 - (1.5 * IQR)
    upper_limit = q3 + (1.5 * IQR)
    print(f"Lower limit: {lower_limit} Upper limit: {upper_limit}")
    print(f"Minimum value: {x.min()}   Maximum Value: {x.max()}")
    outliers = x[(x < lower_limit) | (x > upper_limit)]
    if len(outliers) > 0:
        print(f"Removing {len(outliers)} outliers")
        x = x[(x >= lower_limit) & (x <= upper_limit)]
    else:
        print("No outliers detected")
    return x


detect_outlier(df["K"])
# Now iterate over all the soil_param
for k in soil_params:
    df[k] = detect_outlier(df[k])


# Now we can plot again to see the results

plot_soil_params_boxplot(df, soil_params)


df.isna().sum()


def fill_na_values(df):
    # Fill in missing values for each column with the median value
    for col in soil_params:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    return df


# Assuming your DataFrame is named 'df'
df = fill_na_values(df)

df.info()
# Select features for modelfrom sklearn.model_selection import train_test_split

def create_ph_model(df):

    X = df.drop("pH", axis=1)
    y = df["pH"]

    # Create training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "decision tree": {
            "model": DecisionTreeRegressor(),
            "params": {"decisiontreeregressor__splitter": ["best", "random"]},
        },
        "svm": {
            "model": SVR(gamma="auto"),
            "params": {"svr__C": [1, 10, 100, 1000], "svr__kernel": ["rbf", "linear"]},
        },
        "random_forest": {
            "model": RandomForestRegressor(),
            "params": {"randomforestregressor__n_estimators": [1, 5, 10]},
        },
        "k regressor": {
            "model": KNeighborsRegressor(),
            "params": {
                "kneighborsregressor__n_neighbors": [5, 10, 20, 25],
                "kneighborsregressor__weights": ["uniform", "distance"],
            },
        },
    }

    score = []
    details = []
    best_param = {}
    for mdl, par in models.items():
        pipe = make_pipeline(preprocessing.StandardScaler(), par["model"])
        res = model_selection.GridSearchCV(pipe, par["params"], cv=5)
        res.fit(X_train, y_train)
        score.append(
            {
                "Model name": mdl,
                "Best score": res.best_score_,
                "Best param": res.best_params_,
            }
        )
        details.append(pd.DataFrame(res.cv_results_))
        best_param[mdl] = res.best_estimator_

    score = pd.DataFrame(score)
    score = score.sort_values(by="Best score", ascending=False)
    best_model = best_param[score["Model name"].iloc[0]]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print("Accuracy score: ", metrics.accuracy_score(y_test, y_pred))
    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(metrics.classification_report(y_test, y_pred))
    print("ROC_AUC score: ", metrics.roc_auc_score(y_test, y_pred))

    return pd.DataFrame(score)

create_ph_model(df)

df['pH'].plot()

df['pH'].describe()