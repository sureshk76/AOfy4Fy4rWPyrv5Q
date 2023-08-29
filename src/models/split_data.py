import sys
from sklearn.model_selection import train_test_split
sys.path.append('src/data')
import data_preprocessing
sys.path.insert(0, '/src/data')

def split_data (data, testSize, randomState):
    X, y = data_preprocessing.transformed_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state = randomState)
    data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
    return data