from APS_FAULT_DETECTION.entity.config_entity import ModelTrainerConfig
from APS_FAULT_DETECTION.exception import CustomException
from APS_FAULT_DETECTION.logger import logging
from xgboost import XGBClassifier
import os, sys
from APS_FAULT_DETECTION.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from APS_FAULT_DETECTION.utils import creating_pickle_file, load_numpy_file

class ModelTrainer:
    def __init__(self, model_trainer_config : ModelTrainerConfig,
                 data_transformation_artifact :DataTransformationArtifact
                 ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def model_training(self, x, y):
        try:
            xgboost_model = XGBClassifier()
            xgboost_model.fit(x, y)
            return xgboost_model
        except Exception as e:
            raise CustomException(e, sys)

    def Initiate_Model_Training(self):
        try:
            
            train_arr = load_numpy_file(self.data_transformation_artifact.transformed_train_data)
            test_arr = load_numpy_file(self.data_transformation_artifact.transformed_test_data)
            logging.info(f'training array shape: {train_arr.shape} and testing array shape : {test_arr.shape}')

            os.makedirs(os.path.dirname(self.model_trainer_config.model_path))
            logging.info('Making the model_trainer_directory')

            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]
            logging.info('Data splitted into independent and dependent variables')


            model = self.model_training(x = x_train, y = y_train)
            logging.info('model training on trainng data done')

            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)
            logging.info('Prediction of y for both training and testing data done')

            training_f1_score = f1_score(y_true = y_train, y_pred = y_pred_train)
            testing_f1_score = f1_score(y_true = y_test, y_pred = y_pred_test)
            logging.info(f'f1_Score for training is : {training_f1_score}, f1_Score for testing is : {testing_f1_score}')

            if testing_f1_score < self.model_trainer_config.expected_score:
                raise Exception('Model accuracy too weak to proceed')
            
            dif = abs(training_f1_score-testing_f1_score)
            
            if dif > self.model_trainer_config.threshold_value:
                raise Exception('Model is overfitted')
            
            creating_pickle_file(variable = model, path = self.model_trainer_config.model_path)
            logging.info('Pickle file created for the model')

            model_trainer_artifact = ModelTrainerArtifact(model_file=self.model_trainer_config.model_path,
                                                          f1_training_score=training_f1_score,
                                                          f1_testing_score=testing_f1_score
                                                          )
            
            logging.info('Model Training phase completed')
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)

