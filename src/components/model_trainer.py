import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj,evaluate_models
@dataclass
class modeltrainigconfig:
    trained_model_file_path=os.path.join("aritifacts","modlel.pkl")


class modeltrainer:
    def __init__(self):
        self.model_training_config=modeltrainigconfig()


    def initaite_model_training(self,train_array,test_array):
        try:
            logging.info("spliting training and testing data")

            x_train,x_test,y_train,y_test=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]

            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }



            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                             models=models)


            best_model_score=max(list(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]
            if  best_model_score<0.6:
                raise CustomException("no best model found")
            logging.info(f"Best found model on both training and testing dataset")


            save_obj(

                file_path=self.model_training_config.trained_model_file_path,
                obj=best_model

            )

            predicted=best_model.predict(x_test)
            r2_scoree=r2_score(y_test,predicted)

            return r2_scoree
        except Exception as e:
            raise CustomException(e,sys)