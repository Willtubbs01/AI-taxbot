import typer
from .dataset import app as dataset_app
from .inputs import app as inputs_app

app = typer.Typer(help="TaxMoE command line interface.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(inputs_app, name="inputs")

if __name__ == "__main__":
    app()
