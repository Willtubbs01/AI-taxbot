import typer
from .dataset import app as dataset_app
from .inputs import app as inputs_app
from .model import app as model_app
from .benchmark import app as benchmark_app
from .train import app as train_app

app = typer.Typer(help="TaxMoE command line interface.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(inputs_app, name="inputs")
app.add_typer(model_app, name="model")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(train_app, name="train")

if __name__ == "__main__":
    app()
