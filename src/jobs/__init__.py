from azure.ai.ml import command
from azure.ai.ml.entities import Command


class Jobs:
    def __init__(self, config):
        self.config = config
        self.arguments = dict(
            code=self.config.src_path,
            environment=config.latest_env_name,
            experiment_name=self.config.experiment_name,
            compute=self.config.compute_instance,
        )
        self.command = "PYTHONPATH=.. python -m jobs.{name} {args}"

    def get_job(self, name: str) -> Command:
        return getattr(self, f"_{name}")(name)

    def _create_data_assets(self, name: str) -> Command:
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
