import click
from azure.ai.ml import command, MLClient
from azure.ai.ml.entities import Environment


from config import Config
from scripts import utils


@click.group()
def cli():
    pass


@cli.command()
@click.argument("script_name")
def job(script_name: str):
    config = Config()
    ml_client: MLClient = config.get_ml_client()
    default_args = f" \
        --data_asset_name {config.data_asset_name} \
        --data_asset_version {config.data_asset_version} \
        --experiment_name {config.experiment_name} \
        --subscription_id {config.subscription_id} \
        --resource_group {config.resource_group} \
        --workspace_name {config.workspace_name}"
    cmd = f"python {script_name}.py" + default_args

    match script_name:
        case "create_data_asset":
            cmd += f" --numerai_data_version {config.numerai_data_version}"
        case _:
            raise Exception(f"Script '{script_name}' is not implemented")

    job = command(
        command=cmd,
        code=f"{config.src_path}/scripts",
        environment=config.get_latest_env_name(ml_client),
        compute=config.compute_instance,
        display_name=script_name,
        experiment_name=config.experiment_name,
    )
    ml_client.create_or_update(job)


@cli.command()
@click.argument("entity")
def update(entity):
    config = Config()
    ml_client: MLClient = config.get_ml_client()

    match entity:
        case "conda":
            env: Environment = config.get_latest_env(ml_client)
            utils.update_conda(config.conda_file_path, env, ml_client)
        case _:
            raise Exception(f"Update of entity '{entity}' is not implemented")


if __name__ == "__main__":
    cli()
