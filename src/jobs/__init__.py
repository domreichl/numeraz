import json

from azure.ai.ml import Input, command
from azure.ai.ml.entities import Command, Sweep
from azure.ai.ml.sweep import QUniform


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

    def _tune_hparams(self, name: str) -> Sweep:
        args = f"--main_target {self.config.main_target} --hparams '{json.dumps(self.config.hparams)}'"
        args += " --train_data ${{inputs.train_data}} --n_estimators ${{inputs.n_estimators}}"
        trial_command: Command = command(
            command=self.command.format(name=name, args=args),
            inputs={
                "train_data": Input(
                    path=self.config.component_inputs["train_data"], type="uri_file"
                ),
                "n_estimators": 1,
            },
            **self.arguments,
        )
        trial = trial_command(n_estimators=QUniform(1, 25000, 100))
        sweep: Sweep = trial.sweep(
            primary_metric="corr_sharpe",
            goal="maximize",
            sampling_algorithm="bayesian",
            max_concurrent_trials=2,
            max_total_trials=100,
            timeout=60 * 60 * 4,
        )
        sweep.display_name = f"{name}__n_estimators"
        sweep.experiment_name = f"{self.config.experiment_name}_sweep"

        return sweep
