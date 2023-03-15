from APS_FAULT_DETECTION.entity.config_entity import DataValidationConfig, TrainingPipelineConfig
from APS_FAULT_DETECTION.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from APS_FAULT_DETECTION.exception import CustomException
from APS_FAULT_DETECTION.logger import logging
import os, sys
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
from APS_FAULT_DETECTION.utils import create_yaml_file, object_to_float

class DataValidation:
    def __init__(self, data_validation_config = DataValidationConfig, 
                 data_ingestion_artifact = DataIngestionArtifact
                 ):
        try:
            self.data_validation_config = data_validation_config
            self.data_validation_artifact = data_ingestion_artifact
            self.report_file = {}
            self.validate_dic = {}
        except Exception as e:
            raise CustomException(e,sys)
        
    def checking_null_values(self, data, report_key):
        try:
            null_column = []
            for column in data.columns:
                x = (data[column].isnull().sum())/data.shape[0] 
                if x > self.data_validation_config.null_threshold:
                    null_column.append(column)
            
            self.validate_dic[report_key] = null_column

            data = data.drop(null_column, axis = 1)

            if len(data.columns) == 0:
                return None
            else:
                return data
        except Exception as e:
            raise CustomException(e, sys)

    def checking_columns(self, base_df, current_df, report_key):
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            missing_columns = []

            for column in base_columns:
                if column not in current_columns:
                    missing_columns.append(column)

            self.validate_dic[report_key] = missing_columns

            if (len(missing_columns)/len(base_df.columns))*100 > 70:
                return False
            else:
                return True
        except Exception as e:
            raise CustomException(e, sys)

    def data_drift(self, base_df, current_df, report_key):
        try:
            logging.info(f'{base_df.shape}, {current_df.shape}')
            for column in current_df.columns:
                logging.info(f'{column}')
                score = ks_2samp(base_df[column], current_df[column])
                if score.pvalue < 0.05:
                    self.report_file[column] = {
                        'pvalue' : score.pvalue,
                        'column_name' : column,
                        'good_distribution' : False
                    }
                else:
                    self.report_file[column] = {
                        'pvalue' : score.pvalue,
                        'column_name' : column,
                        'good_distribution' : True
                    }
            resultList = list(self.report_file.items())
            self.validate_dic[report_key]=resultList
            
            return self.validate_dic
        
        except Exception as e:
            raise CustomException(e, sys)

    def Initiate_Data_Validation(self):
        try:
            #Reading training and testing current data
            current_train_df = pd.read_csv(self.data_validation_artifact.train_file_path)
            current_test_df = pd.read_csv(self.data_validation_artifact.test_file_path)
            logging.info(f'Current training data looks like : {current_train_df.shape} and Current testing data looks like : {current_test_df.shape}')

            #Reading base data
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({'na':np.nan}, inplace = True)
            logging.info(f'Base data looks like : {base_df.shape}')

            #Checking Null values and removing columns with more than 30% null values
            current_train_df = self.checking_null_values(data = current_train_df, report_key = 'checking null values Current Training Data')
            current_test_df = self.checking_null_values(data = current_test_df, report_key = 'checking null values Current Testing Data')
            logging.info(f'After removing columns with null values more than 30% training data shape looks like : {current_train_df.shape} ')
            logging.info(f'After removing columns with null values more than 30% testing data shape looks like : {current_test_df.shape} ')

            #Converting object to float
            TARGET_COLUMN = 'class'
            current_train_df = object_to_float(data = current_train_df, TARGET_COLUMN=TARGET_COLUMN, kind = 'current training')
            current_test_df = object_to_float(data = current_test_df, TARGET_COLUMN=TARGET_COLUMN, kind = 'current testing')
            base_df = object_to_float(data = base_df, TARGET_COLUMN=TARGET_COLUMN, kind = 'base')

            #Checking the columns for training data and performing data drift
            if self.checking_columns(base_df = base_df, current_df = current_train_df, report_key = 'checking columns for Current Training Data') == True:
                self.data_drift(base_df = base_df, current_df = current_train_df, report_key = 'data drift for Training_file')

            #Checking the columns for testing data and performing data drift
            if self.checking_columns(base_df = base_df, current_df = current_test_df, report_key = 'checking columns for Current testing Data') == True:
                end_report = self.data_drift(base_df = base_df, current_df = current_test_df, report_key = 'data drift for Testing_file')

            create_yaml_file(data = end_report, report_path = self.data_validation_config.report_path)

            data_validation_artifact = DataValidationArtifact(
                report_path=self.data_validation_config.report_path
                )
            logging.info('Data Validation phase completed')
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)