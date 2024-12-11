from argparse import ArgumentParser, Namespace
from azure.ai.ml import command, Input, Output
from azure.ai.ml.entities import Command


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

    def get_component(self, name: str) -> Command:
        return getattr(self, f"_{name}")(name)

    def _create_data_asset(self, name: str) -> Command:
        args = f"\
            --data_asset_name {self.config.data_asset_name} \
            --data_asset_version {self.config.data_asset_version} \
            --subscription_id {self.config.subscription_id} \
            --resource_group {self.config.resource_group} \
            --workspace_name {self.config.workspace_name} \
            --numerai_data_version {self.config.numerai_data_version}"
        return command(
            display_name=name,
            command=self.command.format(name=name, args=args),
            **self.arguments,
        )

    def _preprocess_data(self, name: str) -> Command:
        args = f" \
            --feature_set {self.config.feature_set} \
            --data_uri ${{inputs.data_uri}} \
            --train_data ${{outputs.train_data}} \
            --test_data ${{outputs.test_data}}"
        return command(
            display_name=name,
            command=self.command.format(name=name, args=args),
            inputs={"data_uri": Input(path=self.config.data_asset_uri, mode="direct")},
            outputs={
                "train_data": Output(type="uri_file"),
                "test_data": Output(type="uri_file"),
            },
            **self.arguments,
        )
