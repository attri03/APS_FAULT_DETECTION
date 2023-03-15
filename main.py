from APS_FAULT_DETECTION.components.data_ingestion import DataIngestion
from APS_FAULT_DETECTION.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from APS_FAULT_DETECTION.components.data_validation import DataValidation
from APS_FAULT_DETECTION.entity.artifact_entity import DataIngestionArtifact
from APS_FAULT_DETECTION.components.data_transformation import DataTransformation

if __name__ == "__main__":
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    print(data_ingestion.Inititate_Data_Ingestion())

    data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
    data_ingestion_artifact = DataIngestionArtifact(raw_file_path=data_ingestion_config.raw_file_path,
                                                    train_file_path=data_ingestion_config.train_file_path,
                                                    test_file_path=data_ingestion_config.test_file_path
                                                    )
    data_validation = DataValidation(data_validation_config = data_validation_config,data_ingestion_artifact = data_ingestion_artifact )
    print(data_validation.Initiate_Data_Validation())

    data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
    data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                             data_ingestion_artifact=data_ingestion_artifact)
    print(data_transformation.Initiate_Data_Transformation())