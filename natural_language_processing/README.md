# Path Signatures for Natural Language Processing using RoughPy

[nlp_demo.ipynb](nlp_demo.ipynb) is a [Jupyter](https://jupyter.org/) notebook which demonstrates the use of path signatures obtained using [RoughPy](https://roughpy.org/) for natural language processing.

## How to Run

The notebook attempts to install the packages listed in [requirements.txt](requirements.txt) automatically when executed.
A typical process for installing the package dependencies involves creating a new Python virtual environment and then inside the environment executing:

```{bash}
pip install -r requirements.txt
```

Then install PyTorch manually following the [official instructions](https://pytorch.org/get-started/locally/).
This notebook has been updated to run with PyTorch=2.1.1.
Alternatively (not recommended), you can try running:

```{bash}
pip install -r requirements_torch.txt
```

This notebook has been updated to run with Python=3.11. With running the notebook in Jupyter, you may need to run the following to install an IPython kernel:

```{bash}
python -m ipykernel install --user --name=nlp-roughpy-env
```

## Note

Previously, the notebook was written to run with Python=3.7 and used [iisignature](https://pypi.org/project/iisignature/) for signature computation. The notebook has been updated to run with Python=3.11.7 and uses [RoughPy](https://roughpy.org/) (version 0.1.1).
