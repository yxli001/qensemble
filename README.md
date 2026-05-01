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
python -m qensemble.main --config <path_to_config> 
```

Reference configs are provided in `configs/` for MNIST, CIFAR-10, and JSC datasets. You can modify these or create your own configs to run different experiments.

For ensemble runs, `ensemble.independent` defaults to `false`. Set it to `true` when you want each ensemble member to train independently and only average them at evaluation/export time.

## Running Sweeps
Use the repo scripts instead of raw `wandb` commands.

1. Create a sweep (this script prints the sweep ID in W&B output).

```sh
bash scripts/sweep_create.sh <path_to_sweep_config>
```

Example:

```sh
bash scripts/sweep_create.sh configs/mnist/sweep-architecture.yaml
```

2. Run one agent for that sweep ID.

```sh
bash scripts/sweep_run.sh qensemble/qensemble/<sweep_id>
```

By default, each agent keeps requesting new runs until the sweep is exhausted, so you do not need one agent per run. To cap how many runs one agent should execute, pass `--count` explicitly.

3. Run multiple agents in parallel for the same sweep.

```sh
bash scripts/sweep_run.sh 3 qensemble/qensemble/<sweep_id>
```

### Choose which GPU to use
Set `CUDA_VISIBLE_DEVICES` before launching `sweep_run.sh`.

Run on GPU 0:

```sh
CUDA_VISIBLE_DEVICES=0 bash scripts/sweep_run.sh qensemble/qensemble/<sweep_id>
```

Run on GPU 2:

```sh
CUDA_VISIBLE_DEVICES=2 bash scripts/sweep_run.sh qensemble/qensemble/<sweep_id>
```

Run on multiple GPUs in one process:

```sh
CUDA_VISIBLE_DEVICES=0,1 bash scripts/sweep_run.sh qensemble/qensemble/<sweep_id>
```

Run multiple agents from one command on a selected GPU set:

```sh
CUDA_VISIBLE_DEVICES=2 bash scripts/sweep_run.sh 3 qensemble/qensemble/<sweep_id>
```