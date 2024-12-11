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
    job = api.run_job(name)
    api.register_component(job)


@cli.command()
@click.argument("name")
def pipeline(name: str):
    api.run_pipeline(name)


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
