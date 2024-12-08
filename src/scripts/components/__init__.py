from azure.ai.ml import command, Input, Output, MLClient
from azure.ai.ml.entities import Command, Component

from components import c


def get_component(name: str) -> Command:
    match name:
        case "data_prep":
            return command(
                name=name,
                display_name="prepare-data",
                description="reads a .xl input, split the input to train and test",
                inputs={
                    "data": Input(type="uri_folder"),
                    "test_train_ratio": Input(type="number"),
                },
                outputs=dict(
                    train_data=Output(type="uri_folder", mode="rw_mount"),
                    test_data=Output(type="uri_folder", mode="rw_mount"),
                ),
                code=data_prep_src_dir,
                command="""python data_prep.py \
                        --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} \
                        --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}} \
                        """,
                environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
            )
        case "model_training":
            return command(
                name=name,
                display_name="train-model",
                code=os.path.join(parent_dir, "train"),
                command="python train.py \
                        --train_data ${{inputs.train_data}} \
                        --model_output ${{outputs.model_output}}",
                environment=args.environment_name + "@latest",
                inputs={"train_data": Input(type="uri_folder")},
                outputs={"model_output": Output(type="uri_folder")},
            )
        case "model_ensembling":
            pass
        case "model_evaluation":
            return command(
                name=name,
                display_name="evaluate-model",
                code=os.path.join(parent_dir, "evaluate"),
                command="python evaluate.py \
                        --model_name ${{inputs.model_name}} \
                        --model_input ${{inputs.model_input}} \
                        --test_data ${{inputs.test_data}} \
                        --evaluation_output ${{outputs.evaluation_output}}",
                environment=args.environment_name + "@latest",
                inputs={
                    "model_name": Input(type="string"),
                    "model_input": Input(type="uri_folder"),
                    "test_data": Input(type="uri_folder"),
                },
                outputs={"evaluation_output": Output(type="uri_folder")},
            )
        case "model_registration":
            register_model = command(
                name=name,
                display_name="register-model",
                code=os.path.join(parent_dir, "register"),
                command="python register.py \
                        --model_name ${{inputs.model_name}} \
                        --model_path ${{inputs.model_path}} \
                        --evaluation_output ${{inputs.evaluation_output}} \
                        --model_info_output_path ${{outputs.model_info_output_path}}",
                environment=args.environment_name + "@latest",
                inputs={
                    "model_name": Input(type="string"),
                    "model_path": Input(type="uri_folder"),
                    "evaluation_output": Input(type="uri_folder"),
                },
                outputs={"model_info_output_path": Output(type="uri_folder")},
            )
        case _:
            raise NotImplementedError(f"Component '{name}' is not implemented")


def register_component(name: str, ml_client: MLClient) -> Component:
    # TODO: when is this actually needed?
    # only when loading components from yml file?
    # or also when reusing components (e.g., in CI/CD)?
    component: Component = ml_client.create_or_update(get_component(name).component)
    print(f"Registered component '{component.name}' with version '{component.version}'")
    return component
