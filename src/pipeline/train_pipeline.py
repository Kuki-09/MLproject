from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

def training_pipeline_start():
    try:
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        modeltrainer = ModelTrainer()
        r2_score = modeltrainer.initiate_model_trainer(train_arr, test_arr)
        
        print(f"Training completed successfully. Best model R2 Score: {r2_score}")

    except Exception as e:
        raise CustomException(e,sys)

if __name__ == '__main__':
    training_pipeline_start()