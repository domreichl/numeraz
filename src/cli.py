import click
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment

from config import Config
from utils import (
    get_ml_client,
    get_latest_env,
    get_latest_env_name,
    run_job,
    register_component,
)


@click.group()
def cli():
    pass


config = Config()
ml_client: MLClient = get_ml_client(config)
config.latest_env_name = get_latest_env_name(config, ml_client)


@cli.command()
@click.argument("name")
def job(name: str):
    run_job(name, config, ml_client)


@cli.command()
@click.argument("name")
def component(name: str):
    job = run_job(name, config, ml_client)
    register_component(job, ml_client)


@cli.command()
def update_conda():
    env: Environment = get_latest_env(ml_client)
    updated_env = Environment(
        name=env.name,
        version=str(int(env.version) + 1),
        description=env.description,
        image=env.image,
        conda_file="conda.yml",
        tags=env.tags,
    )
    env = ml_client.environments.create_or_update(updated_env)
    print(f"Updated environment '{env.name}' to version '{env.version}'")


if __name__ == "__main__":
    cli()
