from APS_FAULT_DETECTION.entity.config_entity import DataTransformationConfig
from APS_FAULT_DETECTION.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from APS_FAULT_DETECTION.exception import CustomException
from APS_FAULT_DETECTION.logger import logging
import os, sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from APS_FAULT_DETECTION.utils import save_numpy_file, creating_pickle_file
from imblearn.combine import SMOTETomek

class DataTransformation:
    def __init__(self, data_transformation_config : DataTransformationConfig,
                 data_ingestion_artifact : DataIngestionArtifact
                 ):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def transformer_file_generator(self)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps = [
                ('imputer', simple_imputer),
                ('scaler', robust_scaler)
            ]
            )
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)
    
    def Initiate_Data_Transformation(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info('Read both training and testing file and ready for trand=foramtion phase')

            TARGET_COLUMN = 'class'
            X_train_df = train_df.drop(TARGET_COLUMN, axis = 1)
            y_train_df = train_df[TARGET_COLUMN]
            X_test_df = test_df.drop(TARGET_COLUMN, axis = 1)
            y_test_df = test_df[TARGET_COLUMN]
            logging.info('Splitted the data into independant and dependant variables')

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_data))
            logging.info('Data transformation dictionary created')

            transormation_file = self.transformer_file_generator()
            transormation_file.fit(X_train_df)
            logging.info('Data fitted into the pipeline')

            transformed_X_train_df = transormation_file.transform(X_train_df)
            transformed_X_test_df = transormation_file.transform(X_test_df)
            logging.info('Both Training and testing data transformed')

            label_encoder = LabelEncoder()
            label_encoder.fit(y_train_df)
            transformed_y_train_df = label_encoder.transform(y_train_df)
            transformed_y_test_df = label_encoder.transform(y_test_df)
            logging.info('Label encoding done on both training and testing file')

            smt = SMOTETomek(random_state=42)
            logging.info(f"Before resampling in training set Input: {transformed_X_train_df.shape} Target:{transformed_y_train_df.shape}")
            transformed_X_train_df, transformed_y_train_df = smt.fit_resample(transformed_X_train_df, transformed_y_train_df)
            logging.info(f"After resampling in training set Input: {transformed_X_train_df.shape} Target:{transformed_y_train_df.shape}")
            
            logging.info(f"Before resampling in testing set Input: {transformed_X_test_df.shape} Target:{transformed_y_test_df.shape}")
            transformed_X_test_df, transformed_y_test_df = smt.fit_resample(transformed_X_test_df, transformed_y_test_df)
            logging.info(f"After resampling in testing set Input: {transformed_X_test_df.shape} Target:{transformed_y_test_df.shape}")

            train_df = np.c_[transformed_X_train_df, transformed_y_train_df]
            test_df = np.c_[transformed_X_test_df, transformed_y_test_df]
            
            save_numpy_file(data = train_df, path = self.data_transformation_config.transformed_train_data)
            logging.info('saved transformed training file')
            save_numpy_file(data = test_df, path = self.data_transformation_config.transformed_test_data)
            logging.info('saving transformed test file')

            creating_pickle_file(variable = transormation_file, 
                                 path = self.data_transformation_config.transformer_file_path
                                 )
            logging.info('Pickle file saved for pipeline')
            
            creating_pickle_file(variable = label_encoder, 
                                 path = self.data_transformation_config.encoder_file_path
                                 )
            logging.info('pickle file saved for encoder')

            data_transformation_artifact = DataTransformationArtifact(
                transformed_test_data=self.data_transformation_config.transformed_test_data,
                transformer_file_path=self.data_transformation_config.transformer_file_path,
                transformed_train_data=self.data_transformation_config.transformed_train_data,
                encoder_file_path=self.data_transformation_config.encoder_file_path,
            )
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)
