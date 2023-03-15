import pandas as pd
from APS_FAULT_DETECTION.config import mongo_client
from APS_FAULT_DETECTION.exception import CustomException
import os, sys
from APS_FAULT_DETECTION.logger import logging

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