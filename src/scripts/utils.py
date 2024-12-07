from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment


def update_conda(conda_file_path: str, env: Environment, ml_client: MLClient):
    updated_env = Environment(
        name=env.name,
        version=str(int(env.version) + 1),
        description=env.description,
        image=env.image,
        conda_file=conda_file_path,
        tags=env.tags,
    )
    ml_client.create_or_update(updated_env)
