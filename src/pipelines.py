from azure.ai.ml import MLClient, dsl
from azure.ai.ml.entities import PipelineJob

from config import Config


class Pipelines:
    def __init__(self, config: Config, ml_client: MLClient):
        self.config = config
        self.ml_client = ml_client

    def get_pipeline(self, name: str, force_rerun: bool = False) -> PipelineJob:
        return getattr(self, f"_{name}")(force_rerun)

    def _model_training(self, force_rerun: bool) -> PipelineJob:
        preprocess_data = self.ml_client.components.get("preprocess_data")
        base_models = self.ml_client.components.get("train_base_models")

        @dsl.pipeline(
            name="model_training",
            description="trains base models, evaluates ensembles, and registers a prod model",
            compute=self.config.compute_instance,
            experiment_name=self.config.experiment_name,
            force_rerun=force_rerun,
        )
        def _pipeline():
            preprocessing = preprocess_data(data_uri=self.config.data_asset_uri)
            base_training = base_models(train_data=preprocessing.outputs.train_data)

            return {
                "train_data": preprocessing.outputs.train_data,
                "test_data": preprocessing.outputs.test_data,
                "base_models_dir": base_training.outputs.base_models_dir,
                "val_predictions": base_training.outputs.val_predictions,
            }

        return _pipeline()
