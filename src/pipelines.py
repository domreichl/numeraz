from azure.ai.ml import MLClient, dsl
from azure.ai.ml.entities import PipelineJob

from config import Config


class Pipelines:
    def __init__(self, config: Config, ml_client: MLClient):
        self.config = config
        self.ml_client = ml_client

    def get_pipeline(
        self, name: str, force_rerun: bool = False, reuse_inputs: bool = False
    ) -> PipelineJob:
        return getattr(self, f"_{name}")(force_rerun, reuse_inputs)

    def _model_training(self, force_rerun: bool, reuse_inputs: bool) -> PipelineJob:
        settings = dict(
            name="model_training",
            description="trains base models, evaluates ensembles, and registers a prod model",
            compute=self.config.compute_instance,
            experiment_name=self.config.experiment_name,
            force_rerun=force_rerun,
            tags={"force_rerun": str(force_rerun), "reuse_inputs": str(reuse_inputs)},
        )
        preprocess_data = self.ml_client.components.get("preprocess_data")
        base_models = self.ml_client.components.get("train_base_models")
        ensembles = self.ml_client.components.get("evaluate_ensembles")
        prod_model = self.ml_client.components.get("build_prod_model")

        if reuse_inputs:

            @dsl.pipeline(**settings)
            def _pipeline():
                preprocessing = preprocess_data(data_uri=self.config.data_asset_uri)
                base_training = base_models(
                    train_data=self.config.component_inputs["train_data"]
                )
                ensembling = ensembles(
                    base_models_dir=self.config.component_inputs["base_models_dir"]
                )
                prod_creation = prod_model(
                    train_data=self.config.component_inputs["train_data"],
                    test_data=self.config.component_inputs["test_data"],
                    best_ensemble=self.config.component_inputs["best_ensemble"],
                )
                return {
                    "train_data": preprocessing.outputs.train_data,
                    "test_data": preprocessing.outputs.test_data,
                    "base_models_dir": base_training.outputs.base_models_dir,
                    "best_ensemble": ensembling.outputs.best_ensemble,
                    "prod_model_dir": prod_creation.outputs.prod_model_dir,
                }

        else:

            @dsl.pipeline(**settings)
            def _pipeline():
                preprocessing = preprocess_data(data_uri=self.config.data_asset_uri)
                base_training = base_models(train_data=preprocessing.outputs.train_data)
                ensembling = ensembles(
                    base_models_dir=base_training.outputs.base_models_dir
                )
                prod_creation = prod_model(
                    train_data=preprocessing.outputs.train_data,
                    test_data=preprocessing.outputs.test_data,
                    best_ensemble=ensembling.outputs.best_ensemble,
                )
                return {
                    "train_data": preprocessing.outputs.train_data,
                    "test_data": preprocessing.outputs.test_data,
                    "base_models_dir": base_training.outputs.base_models_dir,
                    "best_ensemble": ensembling.outputs.best_ensemble,
                    "prod_model_dir": prod_creation.outputs.prod_model_dir,
                }

        return _pipeline()
