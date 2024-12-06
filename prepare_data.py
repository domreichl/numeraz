from azure.ai.ml import command, MLClient

from config import Config


config = Config()
ml_client: MLClient = config.get_ml_client()

job = command(
    code="src/create_data_asset.py",
    command=f"python create_data_asset.py -v {config.numerai_data_version} \
        -dan {config.data_asset_name} -dav {config.data_asset_version} \
        -sid {config.subscription_id} -rg {config.resource_group} -ws {config.workspace_name}",
    environment=config.get_latest_env(ml_client),
    compute=config.compute_instance,
    display_name="data_asset_creation",
    experiment_name="initial",
)
ml_client.create_or_update(job)
