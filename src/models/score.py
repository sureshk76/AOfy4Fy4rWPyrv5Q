from azureml.core.model import Model
import joblib
import json
import numpy
import pandas as pd

models = ['DecisionTreeClassifier.pkl', 'KNeighborsClassifier.pkl', 'RandomForestClassifier.pkl', 'SVC.pkl', 'XGBClassifier.pkl']

def init():
    global model
    model_path = Model.get_model_path(model_name = f'src/models/{models[0]}')
    model = joblib.load(model_path)

def run(raw_data, request_headers):
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = model.predict(data)
    
    return {"result": result.tolist()}

init()

test_row = '{"data": [[ 1.24240000e+04,  1.83982352e+00, -2.53780490e+00 , 6.94641678e-01, -1.14121928e+00 ,-3.18021484e-01 , 0.00000000e+00 , 0.00000000e+00, 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00 , 0.00000000e+00, 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00, 0.00000000e+00 , 1.00000000e+00 , 0.00000000e+00 , 0.00000000e+00, 0.00000000e+00 , 1.00000000e+00], [1.24250000e+04 ,-1.59153705e+00 ,-5.00065540e-01 , 3.86565870e-01, -1.14121928e+00, -3.18021484e-01,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  1.00000000e+00, 0.00000000e+00,  0.00000000e+00], [ 1.24280000e+04, -1.00779288e+00, -5.16351885e-01, -1.04333502e+00, 7.05140876e-02, -3.18021484e-01,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  1.00000000e+00, 0.00000000e+00, 0.00000000e+00]]}'
request_header = {}
prediction = run(test_row, request_header)
print("Test result: ", prediction)