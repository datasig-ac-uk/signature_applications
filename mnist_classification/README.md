# Path Signatures for Handwritten Digit Classification using RoughPy

[handwritten_digit_classification.ipynb](handwritten_digit_classification.ipynb) is a [Jupyter](https://jupyter.org/) notebook which demonstrates the use of path signatures obtained using [RoughPy](https://roughpy.org/) for handwritten digit classification.

## How to Run

The notebook attempts to install the packages listed in [requirements.txt](requirements.txt) automatically when executed.
A typical process for installing the package dependencies involves creating a new Python virtual environment and then inside the environment executing:

```{bash}
pip install -r requirements.txt
```

This notebook has been updated to run with Python=3.11. With running the notebook in Jupyter, you may need to run the following to install an IPython kernel:

```{bash}
python -m ipykernel install --user --name=lang-analysis-env
```

## Note

Previously, the notebook was written to run with Python=3.7 and used [esig](https://esig.readthedocs.io/en/latest/index.html) for signature computation. The notebook has been updated to run with Python=3.11 and uses [RoughPy](https://roughpy.org/) (version 0.1.1).
