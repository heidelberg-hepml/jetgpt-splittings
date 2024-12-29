<div align="center">

# Extrapolating Jet Radiation with Autoregressive Transformers

[![JetGPT](http://img.shields.io/badge/paper-arxiv.2412.12074-B31B1B.svg)](https://arxiv.org/abs/2412.12074)
[![pytorch](https://img.shields.io/badge/PyTorch_2.2+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

</div>

This repository contains the official implementation of **Extrapolating Jet Radiation with Autoregressive Transformers)** by Anja Butter, Francois Charton, [Javier Marino Villadamigo](mailto:marino@thphys.uni-heidelberg.de), [Ayodele Ore](ore@thphys.uni-heidelberg.de), Tilman Plehn, and [Jonas Spinner](mailto:j.spinner@thphys.uni-heidelberg.de).

## 1. Getting started

Clone the repository.

```bash
git clone https://github.com/heidelberg-hepml/jetgpt-splittings
```

Create a virtual environment and install requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

* We have two datasets implemented. First, the $p p\to Z(\mu \mu) + n\ j$ dataset is used in the paper but not published. We are happy to share it upon request. Alternatively, the $p p \to t (b j j) \bar t (\bar b j j) + n\ j$ dataset is also implemented and available on [this link](https://www.thphys.uni-heidelberg.de/~plehn/data/event_generation_ttbar.hdf5), a simple script to download and extract the dataset is available [here](https://github.com/heidelberg-hepml/lorentz-gatr/blob/main/data/collect_data.py ). 

## 2. Running experiments

The most basic way of running an experiment is
```
python run.py
```

This will load the default config file (`config_quick/zmumu.yaml`) and start the corresponding experiment using the default generator (`config_quick/model/jetgpt.yaml`). Note that hydra collects multiple `.yaml` file into the full configuration used for the run, this is handled by the defaults section within the main loaded file.

We use the tool hydra for configuration management, allowing us to easily override all parameters without having to modify the .yaml file for each run. A typical run would look like
```
python run.py model=jetgpt exp_name=jetgpt_test run_name=helloworld exp_type=zmumu training.iterations=10000 evaluation.nsamples=10000
```

If you want to reproduce the results presented in the paper, use the config cards in the `config/` folder.

## 3. Citation

If you find this code useful in your research, please cite the following paper

```bibtex
@article{Butter:2024zbd,
    author = "Butter, Anja and Charton, Fran\c{c}ois and Villadamigo, Javier Mari\~no and Ore, Ayodele and Plehn, Tilman and Spinner, Jonas",
    title = "{Extrapolating Jet Radiation with Autoregressive Transformers}",
    eprint = "2412.12074",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "12",
    year = "2024"
}
```




