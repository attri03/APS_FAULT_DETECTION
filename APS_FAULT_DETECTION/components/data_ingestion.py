from APS_FAULT_DETECTION.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from APS_FAULT_DETECTION.entity.artifact_entity import DataIngestionArtifact
import pandas as pd
from APS_FAULT_DETECTION.utils import get_collection_as_dataframe 
import numpy as np
from sklearn.model_selection import train_test_split
import os, sys
from APS_FAULT_DETECTION.exception import CustomException
from APS_FAULT_DETECTION.logger import logging

class DataIngestion:
    def __init__(self, data_ingestion_config = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def Inititate_Data_Ingestion(self)->DataIngestionArtifact:

        try:
            df:pd.DataFrame = get_collection_as_dataframe(Database_name=self.data_ingestion_config.database_name, Collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Data fetched from mongoDB and looks like : {df.shape}")

            df.replace({'na': np.nan}, inplace = True)
            logging.info('Replaced NA values with np.nan values')

            df_train, df_test = train_test_split(df, test_size = self.data_ingestion_config.test_size, random_state = 42)
            logging.info(f'Data splitted into training and testing data. Training data looks like {df_train.shape} and testing data looks like {df_test.shape}')

            os.makedirs(self.data_ingestion_config.dic_folder_path)
            logging.info('Artifact directory formed')

            df.to_csv(self.data_ingestion_config.raw_file_path, index=False)
            logging.info('Raw data saved to concered directory')

            df_train.to_csv(self.data_ingestion_config.train_file_path, index=False)
            logging.info('Training data saved to concered directory')

            df_test.to_csv(self.data_ingestion_config.test_file_path, index=False)
            logging.info('Testing data saved to concered directory')

            data_ingestion_artifact = DataIngestionArtifact(
                raw_file_path=self.data_ingestion_config.raw_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path    
                )
            logging.info('Data Ingestion phase completed')
            
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)



