import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from imblearn.over_sampling import SMOTE

def transformed_data(data):
    numerical_features = ['age', 'balance', 'duration', 'campaign', 'DaysSinceLastContact']
    categorical_features = ['job', 'marital', 'education', 'contact']
    PTY = PowerTransformer(method='yeo-johnson')
    PTArray = PTY.fit_transform(data[numerical_features])
    PTYData = pd.DataFrame(PTArray, columns=numerical_features)
    PTYData.agg(['skew', 'kurtosis']).T
    zTransform = np.abs(zscore(PTYData.balance))
    print('Number of enries where p(z) > 3: ', len(np.where(zTransform > 3)[0]))
    print('Data loss upon removing entries where p(z) > 3: ', (len(np.where(zTransform > 3)[0]))/len(PTYData.balance))
    df2 = data.drop(np.where(zTransform > 3)[0], axis=0).reset_index(drop=True)
    numerical_transformer = PowerTransformer(method='yeo-johnson')
    categorical_transformer = OneHotEncoder(drop='first')
    preprocessor = ColumnTransformer(
        transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
        ]
    )
# Create a pipeline with the preprocessor
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
# Fit and transform the data
    transformed_data = pipeline.fit_transform(df2)
    X_transformed = transformed_data[:, :-1]
    y_transformed = transformed_data[:, -1]
    smote = SMOTE()
    X_smote, y_smote = smote.fit_resample(X_transformed, y_transformed)
    return X_smote, y_smote