import os, sys
from datetime import datetime

TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
RAW_FILE_PATH = 'raw_data.csv'
REPORT_FILE_NAME = 'report.yml'

class TrainingPipelineConfig:
    def __init__(self):
        self.artifact_dir = os.path.join(os.getcwd(), 'artifact', f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")

class DataIngestionConfig:
    def __init__(self, training_pipeline_config = TrainingPipelineConfig()):
        self.database_name = 'aps'
        self.collection_name = 'sensor'
        self.dic_folder_path:str = os.path.join(training_pipeline_config.artifact_dir, "data_ingestion")
        self.train_file_path:str = os.path.join(self.dic_folder_path, TRAIN_FILE_NAME)
        self.test_file_path:str = os.path.join(self.dic_folder_path, TEST_FILE_NAME)
        self.raw_file_path:str = os.path.join(self.dic_folder_path, RAW_FILE_PATH)
        self.test_size = 0.2

class DataValidationConfig:
    def __init__(self, training_pipeline_config = TrainingPipelineConfig()):
        self.report_path:str = os.path.join(training_pipeline_config.artifact_dir, "data_validation", REPORT_FILE_NAME)
        self.null_threshold = 0.3
        self.base_file_path = 'aps_data.csv'