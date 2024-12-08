import webbrowser
from azure.ai.ml import dsl, Input, Output, MLClient
from azure.identity import DefaultAzureCredential

from utils import parse_args
from components import get_component


args: dict = parse_args(
    [
        "data_asset_name",
        "data_asset_version",
        "subscription_id",
        "resource_group",
        "workspace_name",
        "experiment_name",
        "compute_instance",
    ]
)
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=args["subscription_id"],
    resource_group_name=args["resource_group"],
    workspace_name=args["workspace_name"],
)

DESCR = "Model Training"


@dsl.pipeline(compute=args["compute_instance"], description=DESCR)
def model_training_pipeline(
    data_input, train_test_ratio, learning_rate, registered_model_name
):
    data_prep_job = data_prep_component(
        data=data_input, train_test_ratio=train_test_ratio
    )
    train_job = train_component(
        train_data=data_prep_job.outputs.train_data,
        test_data=data_prep_job.outputs.test_data,
        learning_rate=learning_rate,
        registered_model_name=registered_model_name,
    )
    ensemble_job = ensemble_component(...)
    evaluate = evaluate_model(
        model_name="taxi-model",
        model_input=train.outputs.model_output,
        test_data=prep.outputs.test_data,
    )
    register = register_model(
        model_name="taxi-model",
        model_path=train.outputs.model_output,
        evaluation_output=evaluate.outputs.evaluation_output,
    )

    return {
        "pipeline_job_train_data": data_prep_job.outputs.train_data,
        "pipeline_job_test_data": data_prep_job.outputs.test_data,
        "pipeline_job_trained_model": train.outputs.model_output,
        "pipeline_job_score_report": evaluate.outputs.evaluation_output,
    }


numerai_data = ml_client.data.get(
    name=args["data_asset_name"], version=args["data_asset_version"]
)
pipeline = model_training_pipeline(
    data_input=Input(path=numerai_data.id, type="uri_folder"),
    train_test_ratio=0.2,
    learning_rate=0.25,
    registered_model_name="numeraz-0",
)

pipeline_job = ml_client.jobs.create_or_update(
    pipeline, experiment_name=args["experiment_name"]
)
webbrowser.open(pipeline_job.services["Studio"].endpoint)
ml_client.jobs.stream(pipeline_job.name)
