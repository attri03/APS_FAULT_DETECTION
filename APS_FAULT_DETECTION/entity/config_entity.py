import os, sys
from datetime import datetime

TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
RAW_FILE_PATH = 'raw_data.csv'
REPORT_FILE_NAME = 'report.yml'
TRANSFORMER_FILE_NAME = 'transformer.pkl'
ENCODER_FILE_NAME ='encoder.pkl'
TRANSFORMED_TRAIN_DATA = 'transformed_train.npy'
TRANSFORMED_TEST_DATA = 'transformed_test.npy'
MODEL_FILE = 'model.pkl'

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

class DataTransformationConfig:
    def __init__(self, training_pipeline_config = TrainingPipelineConfig()):
        self.transformer_file_path:str = os.path.join(training_pipeline_config.artifact_dir, 'data_transformation', TRANSFORMER_FILE_NAME)
        self.encoder_file_path:str = os.path.join(training_pipeline_config.artifact_dir, 'data_transformation', ENCODER_FILE_NAME)
        self.transformed_train_data:str = os.path.join(training_pipeline_config.artifact_dir, 'data_transformation', TRANSFORMED_TRAIN_DATA)
        self.transformed_test_data:str = os.path.join(training_pipeline_config.artifact_dir, 'data_transformation', TRANSFORMED_TEST_DATA)

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config = TrainingPipelineConfig()):
        self.model_path = os.path.join(training_pipeline_config.artifact_dir, 'model_trainer', MODEL_FILE)
        self.expected_score = 0.7
        self.threshold_value = 0.1
