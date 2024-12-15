import click

from api import NumerazAPI


@click.group()
def cli():
    pass


api = NumerazAPI()


@cli.command()
@click.argument("name")
def job(name: str):
    api.run_job(name)


@cli.command()
@click.argument("name")
def component(name: str):
    api.register_component(name)


@cli.command()
@click.argument("name")
@click.option("--reuse-inputs", is_flag=True, default=False)
@click.option("--force-rerun", is_flag=True, default=False)
@click.option("--stream-job", is_flag=True, default=False)
def pipeline(name: str, force_rerun: bool, reuse_inputs: bool, stream_job: bool):
    api.run_pipeline(name, force_rerun, reuse_inputs, stream_job)


@cli.command()
@click.argument("entity")
def update(entity: str):
    match entity:
        case "conda":
            api.update_conda()
        case _:
            raise NotImplementedError(f"Update of entity '{entity}' is not implemented")


if __name__ == "__main__":
    cli()
