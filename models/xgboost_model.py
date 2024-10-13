import xgboost as xgb

def train_xgboost_model(X_train, y_train, X_val, y_val):
    param = {'eta': 0.03, 'max_depth': 180, 'subsample': 1.0, 'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    eval_list = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(param, dtrain, 100, eval_list, early_stopping_rounds=10)
    return model
