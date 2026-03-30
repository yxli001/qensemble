# QEnsemble 
## Environment

Check that you have `poethepoet` installed with `poe --version`. If not, install it with `pipx install poethepoet`.

Run the following to set up your environment:

```sh
cp .env.example .env # and edit .env as needed

conda env create -f environment.yml
conda activate qensemble

# install this repo in editable mode so `python -m qensemble...` works from repo root
pip install -e .

poe install
```

It is recommended to have the VSCode Ruff extension installed so you can see linting errors in your editor. You can also run `poe lint` to check for linting errors and `poe format` to automatically fix them.

Linting will automatically be run on commit to ensure consistent code style.

## Running Single Experiments
To run an experiment, create a config file in `configs/` and then run:

```sh
python -m qensemble.run --config <path_to_config> 
```

Reference configs are provided in `configs/` for MNIST, CIFAR-10, and JSC datasets. You can modify these or create your own configs to run different experiments.

## Running Sweeps
TODO