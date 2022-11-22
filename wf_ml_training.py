import xgboost as xgb
import pickle

def model_XGBoost(train_X, train_Y):
    my_model = xgb.XGBClassifier()
    my_model.fit(train_X.numpy(), train_Y)
    filename = 'models/XGBoost.sav'
    pickle.dump(my_model, open(filename, 'wb'))

def ml_training(train_X, train_Y):
    model_XGBoost(train_X,train_Y)