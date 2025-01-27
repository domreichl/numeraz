import webbrowser
from typing import Union

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Command,
    CommandComponent,
    Component,
    Data,
    Environment,
    Job,
    PipelineJob,
    Sweep,
)
from azure.identity import DefaultAzureCredential

from components import Components
from config import Config
from jobs import Jobs
from pipelines import Pipelines


class NumerazAPI:
    def __init__(self):
        self.config = Config()
        self.ml_client: MLClient = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=self.config.subscription_id,
            resource_group_name=self.config.resource_group,
            workspace_name=self.config.workspace_name,
        )
        self.config.latest_env_name = self._get_latest_env()

    def register_component(self, name: str):
        try:
            prev_component: Component = self.ml_client.components.get(name)
            version = str(int(prev_component.version) + 1)
        except:
            version = "1"
        components = Components(self.config)
        component: CommandComponent = components.get_component(name)
        component.version = version
        component = self.ml_client.components.create_or_update(component, force=True)
        print(
            f"Registered component '{component.name}' with version '{component.version}'"
        )

        return component

    def run_job(self, name: str) -> Job:
        jobs = Jobs(self.config)
        command: Union[Command, Sweep] = jobs.get_job(name)
        job: Job = self.ml_client.jobs.create_or_update(command)
        print(f"Started job '{job.display_name}' with name '{job.name}'")

        return job

    def run_pipeline(
        self, name: str, force_rerun: bool, reuse_inputs: bool, stream_job: bool
    ):
        pipelines = Pipelines(self.config, self.ml_client)
        pipeline_job: PipelineJob = pipelines.get_pipeline(
            name, force_rerun, reuse_inputs
        )
        job: Job = self.ml_client.jobs.create_or_update(pipeline_job)
        webbrowser.open(job.services["Studio"].endpoint)
        if stream_job:
            self.ml_client.jobs.stream(job.name)

    def update_conda(self):
        env: Environment = self._get_latest_env()
        updated_env = Environment(
            name=env.name,
            version=str(int(env.version) + 1),
            description=env.description,
            image=env.image,
            conda_file="conda.yml",
            tags=env.tags,
        )
        env = self.ml_client.environments.create_or_update(updated_env)
        print(f"Updated environment '{env.name}' to version '{env.version}'")

    def _get_data_asset(self) -> Data:
        return self.ml_client.data.get(
            name=self.config.data_asset_name, version=self.config.data_asset_version
        )

    def _get_latest_env(self) -> Environment:
        return max(
            self.ml_client.environments.list(self.config.environment_name),
            key=lambda env: int(env.version),
        )

    def _get_latest_env_name(self) -> str:
        env: Environment = self._get_latest_env()
        return f"{env.name}:{env.version}"
