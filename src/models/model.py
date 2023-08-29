import joblib
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
sys.path.append('src/data')
from data_preprocessing import transformed_data
from train_model import train_model
from predict_model import predict_model
from split_data import split_data
import xgboost as xgb

def main():
    df = pd.read_csv('data/processed/data.csv')
    data = split_data(df, 0.2, 42)
    models = {
        'DTC': {'model': DecisionTreeClassifier, 'args' : {'criterion': 'entropy', 'max_depth': 17}},
        'KNN': {'model': KNeighborsClassifier, 'args' : {'metric': 'minkowski', 'n_neighbors': 10, 'p': 1}},
        'RF': {'model': RandomForestClassifier, 'args' : {'criterion': 'gini', 'max_depth': 20, 'n_estimators': 20}},
        'SVC': {'model': SVC, 'args' : {'kernel': 'rbf'}},
        'XGB': {'model': xgb.XGBClassifier, 'args' : {'objective': 'binary:logistic','eval_metric': 'logloss','max_depth': 3,'learning_rate': 0.1,'n_estimators': 100}}
    }    
    
    model_list = []
    for i in models:
        model = train_model(data, models[i]['model'], models[i]['args'])
        model_list.append(model)
        
    print(model_list)
    
    # model = train_model(data, models['DC']['model'], models['DC']['args'])
    for i in model_list:
        accuracy, confustion_matrix, classification_report = predict_model(i, data)
        
    for i in model_list:
        name = str(i).split('(')[0]
        joblib.dump(value = i, filename=f'src/models/{name}.pkl')        

if __name__ == '__main__':
    main()