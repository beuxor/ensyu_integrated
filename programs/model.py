import numpy as np
import time
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import optuna
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import optuna.integration.lightgbm as olgb


## 準備
def get_data():
    return X_train, X_validation, y_train, y_validation, X_test

def objective(trial):
    X_train, X_val, y_train, y_val, X_test = get_data()
    
    params_dist = {
        'boosting'          : 'gbdt',
        'metric'            : 'auc',
        'objective'         : 'binary',
        'learning_rate'     : 0.05,
        'lambda_l1'         : trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2'         : trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        
        'num_leaves'        :   trial.suggest_int('num_leaves', 2, 64, 4),
        'max_depth'         :   trial.suggest_int('max_depth', 3, 7),
        'min_data_in_leaf':  trial.suggest_int('min_data_in_leaf', 10, 40, 10),
        
        'feature_fraction'  : trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction'  : trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq'      : trial.suggest_int('bagging_freq', 0, 10),
        'min_child_samples' : trial.suggest_int('min_child_samples', 5, 100),
        'seed'              : 42,
        'verbosity'         : -1,
    }
    
    lgb_train = olgb.Dataset(X_train, y_train)
    lgb_eval  = olgb.Dataset(X_val, y_val, reference=lgb_train)

    evaluation_results = {}
    model = olgb.train(params_dist,
                      lgb_train,
                      valid_sets=(lgb_train, lgb_eval),
                      valid_names=["Train", "Test"],
                      num_boost_round=100000,
                      callbacks=[olgb.early_stopping(100), 
                                        olgb.log_evaluation(),
                                        olgb.record_evaluation(evaluation_results)]
                      )
    
    y_prob_val = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_val = np.round(y_prob_val)

    score = roc_auc_score(np.round(y_val.values), np.round(y_pred_val))
    return score 

def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    return title

def preprocess2(df, features):
#     features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
#     features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
#     features = ["Pclass", "Sex", "AgeGroup", "FamilySize", "Ticket", "FareGroup", "CabinGroup", "Embarked", "Title"]
#     features = ["AgeGroup", "FamilySize", "FareGroup", "CabinGroup", "Embarked", "Title"]
    target = 'Survived'
    # Age Group
    set_bins = [-10, 0, 5, 10, 20, 30, 40, 50, 60, 70, 80]
    bin_labels = ["None", "0-5", "5-10", "10代", "20代", "30代", "40代", "50代", "60代", "70代"]
    df['AgeGroup'] = pd.cut(df["Age"], bins = set_bins, labels = bin_labels, right=False)
    
    # Family Size 
    df["FamilySize"] = df['SibSp'] + df['Parch']
    
    # Fare 
    set_bins = [-1, 7, 10, 20, 50, 100, 1000]
    bin_labels = ["lower", "low", "mid-low", "mid-high", "high", "higher"]
    df['FareGroup'] = pd.cut(df["Fare"], bins = set_bins, labels = bin_labels)
    
    # Cabin 
    df['CabinGroup'] = df['Cabin'].isna()
    # Title 
    df["Title"] = df['Name'].apply(extract_title)
    
    cats = ['Sex', 'Ticket', 'Cabin', 'Embarked', "AgeGroup", "FareGroup", "CabinGroup", "Title"]
    for c in cats:
        df[c] = df[c].astype('category')
        
    return df, features, target

def get_acc(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_binary = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, y_pred_binary)
    print("Accuracy:", accuracy)
    return accuracy

def pred_test(model, df_test_p, features):
    y_pred = model.predict(df_test_p[features])
    y_pred_binary = [round(value) for value in y_pred]

    data = {
        "PassengerId": pas_id,
        'Survived': y_pred_binary
    }
    df_sub = pd.DataFrame(data)
    
    # データフレームをCSVファイルに書き込む
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    df_sub.to_csv('./../results/output_{}.csv'.format(timestamp), index=False)
    print("save!", timestamp)



# CSVファイルの読み込み
data_file = "./../titanic/train.csv"
df = pd.read_csv(data_file)

test_file = "./../titanic/test.csv"
df_test = pd.read_csv(test_file)


## 前処理
features = ["AgeGroup", "FamilySize", "FareGroup", "CabinGroup", "Embarked", "Title"]
features = ['Pclass', 'Sex', 'Age', 'FamilySize', 'Embarked', 'FareGroup']
features = ['Pclass', 'Sex', 'Age', 'FamilySize', "SibSp", "Parch", 'Embarked', 'Fare']
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']

df_p, features, target = preprocess2(df, features)
df_p.head()
pas_id = df_test["PassengerId"].values
df_test_p, features, _ = preprocess2(df_test, features)

X = df_p[features]
y = df_p[target]
# 訓練データとテストデータに分割する
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2,random_state=0)

## 学習
train = False
if train:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    params = study.best_params
else:  
    params = {
        'boosting_type':'gbdt',
        'objective':'binary',
        'metric':'auc',
    #    'metric': 'binary_logloss',
        'num_leaves':16,
        'learning_rate':0.1,
        'n_estimators':100000,
        'random_state':0
    }
    params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
    }
    params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'max_depth': 8,
    'min_child_samples': 20,
    'min_child_weight': 0.01
}

# インスタンスの作成
model = lgb.LGBMClassifier(**params)

# モデルの学習
model.fit(
    X_train, 
    y_train,
    eval_set = [(X_train, y_train),(X_validation, y_validation)],
    early_stopping_rounds=100)

# 正解率

# params = model.get_params(iteration = clf.best_iteration_)
# model = lgb.LGBMRegressor()
# model.set_params(**params)

for feat, imp in zip(features, model.feature_importances_):
    print("split", feat, "\t", imp)


lgb.plot_importance(model) 
lgb.plot_importance(model, importance_type = "gain")

plt.show()


get_acc(model, X_test, y_test)
print(params)
pred_test(model, df_test_p, features)


