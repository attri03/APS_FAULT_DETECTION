import pandas as pd
from APS_FAULT_DETECTION.config import mongo_client
from APS_FAULT_DETECTION.exception import CustomException
import os, sys
from APS_FAULT_DETECTION.logger import logging
import yaml

def get_collection_as_dataframe(Database_name:str, Collection_name:str)->pd.DataFrame:
    try:
        logging.info('Collecting the data from mongodb')
        df = pd.DataFrame(list(mongo_client[Database_name][Collection_name].find()))
        logging.info('Removing _id from the dataset')
        if '_id' in df.columns:
            df = df.drop('_id', axis = 1)
        return df
    except Exception as e:
        raise CustomException(e,sys)
    
def create_yaml_file(data, report_path):
    try:
        file_dir = os.path.dirname(report_path)
        os.makedirs(file_dir,exist_ok=True)
        with open(report_path,"w") as file_writer:
            yaml.dump(data,file_writer)
        logging.info('Report for data validation generated and saved')
    except Exception as e:
        raise CustomException(e, sys)
    
def object_to_float(data, TARGET_COLUMN, kind):
    try:
        for column in data.columns:
            if column != TARGET_COLUMN:
                data[column] = data[column].astype(float)
        logging.info(f'Converted data types from object to float for {kind} data')
        return data
    except Exception as e:
        raise CustomException(e, sys)