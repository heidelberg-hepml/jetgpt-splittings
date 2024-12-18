# jetGPT

Code for LHC event generation with autoregressive transformers, supported by various classifiers. The implemented processes are $p p\to Z(\mu \mu)$, $p p \to t (b j j) \bar t (\bar b j j)$, both with variable numbers of extra jets.

## 1. Getting started

Clone the repository.

```bash
git clone https://github.com/heidelberg-hepml/jetgpt
```

Create a virtual environment and install requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Modify the base_dir in config/default.yaml. Ask Maeve or Jonas for the training data and put the path to the datasets in the data section of config/zmumu.yaml or config/ttbar.yaml.

## 2. Running experiments

The most basic way of running an experiment is
```
python run.py
```

This will load the default config file (config/zmumu.yaml) and start the corresponding experiment using the default generator (config/generator/jetgpt.yaml) and default classifier (config/classifier/mlp.py). Note that hydra collects multiple .yaml file into the full configuration used for the run, this is handled by the defaults section within the main loaded file.

We use the tool hydra for configuration management, allowing us to easily override all parameters without having to modify the .yaml file for each run. A typical run would look like
```
python run.py model=jetgpt exp_name=jetgpt_test run_name=helloworld exp_type=zmumu training_gen.nepochs=2 sampling.nsamples=10000
```

Tests are organized as experiments, which each contain multiple runs. We use mlflow to track metrics during training and save them to a database object within the mlflow folder for each experiment. The mlflow web interface is useful to keep an overview over the sometimes confusing amount of metrics arising during the training of the different models within this code. A local mlflow web interface using port 4242 can be started with the command
```
mlflow ui --port 4242 --backend-store-uri sqlite:///path/to/mlflow/mlflow.db
```

Finally, existing runs can be reloaded to perform additional tests or continue training. For a previous run with exp_name=exp and run_name=run, one can use the --config-name (-cn) and --config-path (-cp) options to overwrite the defaults in run.py. Further, one can use the warm_start_idx option to specify which model state should be loaded (defaults to 0, which is the model after the first run), and warm_start_stage to specify which model stage should be loaded (defaults to gen which is plain generator; relevant when discformer is trained). The full command is
```
python run.py -cp runs/exp/run -cn config warm_start_idx=0 warm_start_stage=gen
```
Currently there is no option for loading classifiers to keep the code simple, but this would be straight-forward because their weights are saved anyway for early stopping. 