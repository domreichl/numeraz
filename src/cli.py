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

    match script_name:
        case "create_data_asset":
            cmd = f"python {script_name}.py -n {config.data_asset_name} -v {config.data_asset_version} \
                -ndv {config.numerai_data_version} -exp {config.experiment_name} \
                -sid {config.subscription_id} -rg {config.resource_group} -ws {config.workspace_name}"
        case _:
            raise Exception(f"Script '{script_name}' is not implemented")

    job = command(
        command=cmd,
        code=f"{config.src_path}/scripts/{script_name}.py",
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
