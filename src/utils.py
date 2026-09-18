import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import pickle
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_models(x_train,y_train,x_test,y_test,models,diff_param):


    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]

            para=diff_param[list(models.keys())[i]]

            

            gd=GridSearchCV(estimator=model,param_grid=para,cv=3,scoring="r2")
            gd.fit(x_train,y_train)

            best_params=gd.best_params_
            model.set_params(**best_params)


            model.fit(x_train,y_train)

            y_train_pred=model.predict(x_train)

            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)


            report[list(models.keys())[i]]=test_model_score






        return report


    except Exception as e:
        raise CustomException(e,sys)

    
def load_obj(file_path):
    try:
        with open(file_path,"rb") as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e,sys)


