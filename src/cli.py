import click
from azure.ai.ml import command, MLClient

from config import Config


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
        environment=config.get_latest_env(ml_client),
        compute=config.compute_instance,
        display_name=script_name,
        experiment_name=config.experiment_name,
    )
    ml_client.create_or_update(job)


if __name__ == "__main__":
    cli()
