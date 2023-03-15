from APS_FAULT_DETECTION.components.data_ingestion import DataIngestion
from APS_FAULT_DETECTION.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig

if __name__ == "__main__":
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    print(data_ingestion.Inititate_Data_Ingestion())