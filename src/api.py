import webbrowser
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Command, Component, Data, Environment, Job, Pipeline
from azure.identity import DefaultAzureCredential

from components import Components
from config import Config
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

    def register_component(self, job: Job) -> Component:
        try:
            prev_component: Component = self.ml_client.components.get(job.display_name)
            version = str(int(prev_component.version) + 1)
        except:
            version = "1"
        component: Component = job.component
        component.name = job.display_name
        component.version = version
        component = self.ml_client.components.create_or_update(component, force=True)
        print(
            f"Registered component '{component.name}' with version '{component.version}'"
        )

        return component

    def run_job(self, name: str) -> Job:
        components = Components(self.config)
        command: Command = components.get_component(name)
        job: Job = self.ml_client.jobs.create_or_update(command)
        print(f"Started job '{job.display_name}' with name '{job.name}'")

        return job

    def run_pipeline(self, name: str):
        pipelines = Pipelines(self.config, self.ml_client)
        pipeline: Pipeline = pipelines.get_pipeline(name)
        job: Job = self.ml_client.jobs.create_or_update(pipeline)
        webbrowser.open(job.services["Studio"].endpoint)
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
