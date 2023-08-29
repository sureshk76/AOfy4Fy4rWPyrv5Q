
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def train_model(data, model, args):
    reg_model = model(**args)
    reg_model.fit(data['train']['X'], data['train']['y'])
    return reg_model

