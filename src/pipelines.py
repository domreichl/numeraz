from azure.ai.ml import dsl, MLClient
from azure.ai.ml.entities import Pipeline

from config import Config


class Pipelines:
    def __init__(self, config: Config, ml_client: MLClient):
        self.config = config
        self.ml_client = ml_client

    def get_pipeline(self, name: str) -> Pipeline:
        return getattr(self, f"_{name}")()

    def _model_training(self) -> Pipeline:
        preprocess_data = self.ml_client.components.get("preprocess_data")

        @dsl.pipeline(
            name="model_training",
            description="creates a prod model",
            compute=self.config.compute_instance,
            experiment_name=self.config.experiment_name,
        )
        def _pipeline() -> Pipeline:
            preprocessing = preprocess_data(data_uri=self.config.data_asset_uri)

            return {
                "train_data": preprocessing.outputs.train_data,
                "test_data": preprocessing.outputs.test_data,
            }

        return _pipeline()
