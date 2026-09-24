import typer
from .dataset import app as dataset_app

app = typer.Typer(help="TaxMoE command line interface.")
app.add_typer(dataset_app, name="dataset")

if __name__ == "__main__":
    app()
