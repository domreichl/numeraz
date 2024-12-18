import json
from argparse import ArgumentParser, Namespace

from azure.ai.ml import Input, Output
from azure.ai.ml.entities import CommandComponent


def parse_args(args: list) -> dict:
    parser = ArgumentParser()
    for arg in args:
        parser.add_argument(f"--{arg}", type=str, required=True)
    args: Namespace = parser.parse_args()
    return vars(args)


class Components:
    def __init__(self, config):
        self.config = config
        self.arguments = dict(
            code=self.config.src_path,
            environment=config.latest_env_name,
            experiment_name=self.config.experiment_name,
            compute=self.config.compute_instance,
        )
        self.command = "PYTHONPATH=.. python -m components.{name} {args}"

    def get_component(self, name: str) -> CommandComponent:
        return getattr(self, f"_{name}")(name)

    def _preprocess_data(self, name: str) -> CommandComponent:
        args = f"--feature_set {self.config.feature_set}"
        args += " --data_uri ${{inputs.data_uri}} --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}}"
        return CommandComponent(
            name=name,
            display_name=name,
            command=self.command.format(name=name, args=args),
            inputs={"data_uri": Input(path=self.config.data_asset_uri, mode="direct")},
            outputs={
                "train_data": Output(type="uri_file"),
                "test_data": Output(type="uri_file"),
            },
            **self.arguments,
        )

    def _train_base_models(self, name: str) -> CommandComponent:
        args = f"--main_target {self.config.main_target} --hparams '{json.dumps(self.config.hparams)}'"
        args += " --train_data ${{inputs.train_data}} --base_models_dir ${{outputs.base_models_dir}}"
        return CommandComponent(
            name=name,
            display_name=name,
            command=self.command.format(name=name, args=args),
            inputs={"train_data": Input(type="uri_file")},
            outputs={"base_models_dir": Output(type="uri_folder")},
            **self.arguments,
        )

    def _evaluate_ensembles(self, name: str) -> CommandComponent:
        args = f"--main_target {self.config.main_target}"
        args += " --base_models_dir ${{inputs.base_models_dir}} --best_ensemble ${{outputs.best_ensemble}}"
        return CommandComponent(
            name=name,
            display_name=name,
            command=self.command.format(name=name, args=args),
            inputs={"base_models_dir": Input(type="uri_folder")},
            outputs={"best_ensemble": Output(type="uri_file")},
            **self.arguments,
        )

    def _build_prod_model(self, name: str) -> CommandComponent:
        args = f"--experiment_name {self.config.experiment_name} --model_name {self.config.model_name} --feature_set {self.config.feature_set}"
        args += f" --main_target {self.config.main_target} --hparams '{json.dumps(self.config.hparams)}'"
        args += " --data_uri ${{inputs.data_uri}} --train_data ${{inputs.train_data}} --test_data ${{inputs.test_data}}"
        args += " --best_ensemble ${{inputs.best_ensemble}} --prod_model_info ${{inputs.prod_model_info}} --prod_model_dir ${{outputs.prod_model_dir}}"
        return CommandComponent(
            name=name,
            display_name=name,
            command=self.command.format(name=name, args=args),
            inputs={
                "data_uri": Input(path=self.config.data_asset_uri, mode="direct"),
                "train_data": Input(type="uri_file"),
                "test_data": Input(type="uri_file"),
                "best_ensemble": Input(type="uri_file"),
                "prod_model_info": Input(
                    path=self.config.prod_model_info, mode="direct"
                ),
            },
            outputs={"prod_model_dir": Output(type="uri_folder")},
            **self.arguments,
        )
