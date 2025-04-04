# Path Signatures for Drone Identification using Esig
[drone_identification.ipynb](drone_identification.ipynb) is a [Jupyter](https://jupyter.org/) notebook which demonstrates the use of path signatures obtained using esig.
The hypothetical use case involves distinguishing between simulated drone and non-drone objects.
The notebook is viewable directly from GitHub, however Jupyter is recommended for best results when rendering mathematical notation.

## Getting started

First install Poetry according to instructions at: https://python-poetry.org/docs/

```bash
poetry install

jupyter notebook drone_identification.ipynb
```

## Running the notebook

Please note that the notebook involves some heavy computation.
For ease of use, the notebook caches results inside the directory [cached_signatures](cached_signatures).
This directory has been pre-populated. If you would like to ensure that results are recomputed, simply delete the cached_signatures directory and re-run the notebook.
As is suggested in the notebook, users with access to limited computational resources might consider reducing the value of N_INCIDENT_SIGNALS from 3000 to 300, for exploratory purposes.
Users with SSH access to abundant computational resources might consider running Jupyter remotely, [via a tunnel](https://docs.anaconda.com/anaconda/user-guide/tasks/remote-jupyter-notebook/) if necessary.

## Dependencies

This notebook's dependencies are listed in [pyproject.toml](pyproject.toml).
