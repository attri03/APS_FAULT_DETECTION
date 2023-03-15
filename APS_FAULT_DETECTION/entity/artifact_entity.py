from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_file_path:str
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    report_path:str

@dataclass
class DataTransformationArtifact:
    transformed_train_data:str
    transformed_test_data:str
    encoder_file_path:str
    transformer_file_path:str