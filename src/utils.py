from azure.ai.ml import MLClient
from azure.ai.ml.entities import Component, Job

from components import Components
from config import Config


def run_job(name: str, config: Config, ml_client: MLClient) -> Job:
    components = Components(config)
    command = components.get_component(name)
    job: Job = ml_client.jobs.create_or_update(command)
    print(f"Started job '{job.display_name}' with name '{job.name}'")

    return job


def register_component(job: Job, ml_client: MLClient):
    try:
        component: Component = ml_client.components.get(job.display_name)
        version = str(int(component.version) + 1)
    except:
        version = "1"
    component: Component = job.component
    component.name = job.display_name
    component.version = version
    # component.inputs = "TODO"
    # component.outputs = "TODO"
    component = ml_client.components.create_or_update(component, force=True)
    print(f"Registered component '{component.name}' with version '{component.version}'")
