import sys
import os
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from src.utils import save_obj


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('aritifacts', 'preprocessor.pkl')

class DataTransformation:

    def __init__ (self):
        self.data_transformation_config=DataTransformationConfig()

    def  get_data_transformer_obj(self):
        try:
            
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("sclar",StandardScaler())
                

                ]
            )

            cat_pipeline=Pipeline(
                steps=[

                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("onehot",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))

                ]
            )


            logging.info("numerical encoding & standardization done")
            logging.info("cat encoding done ")

            preproccesor=ColumnTransformer(
                [
                    ("num_pipe",num_pipeline,numerical_columns),
                    ("cat_pipe",cat_pipeline,categorical_columns)
                ]


            )

            return preproccesor
        except Exception as e:

            raise CustomException(e,sys)


    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data complete")

            preprocessing_obj=self.get_data_transformer_obj()
            target_coloumn_name="math_score"
            numerical_feature=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop([target_coloumn_name],axis=1)
            target_feature_train_df=train_df[target_coloumn_name]


            input_feature_test_df=test_df.drop([target_coloumn_name],axis=1)
            target_feature_test_df=test_df[target_coloumn_name]

            logging.info("applying preprocessing on training and test dataset")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr=np.c_[

                input_feature_train_arr,np.array(target_feature_train_df)
            ]


            test_arr=np.c_[
            
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)

    