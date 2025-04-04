# Path Signatures for Human Action Recognition using Esig
[jhmdb_demo.ipynb](jhmdb_demo.ipynb) is a [Jupyter](https://jupyter.org/) notebook which demonstrates the use of path signatures obtained using esig for human action recognition.
The analysis is closely related to the work of [Yang et al. (2019)](https://arxiv.org/abs/1707.03993).
The notebook is partially viewable directly from GitHub, however Jupyter or Binder are recommended for viewing the example videos included in the notebook.

## Getting started

First install Poetry according to instructions at: https://python-poetry.org/docs/

You will also need to install [ffmpeg](https://www.ffmpeg.org/).
Please consult https://www.ffmpeg.org/download.html and decide on your preferred installation for your machine.
The author has used [brew](https://formulae.brew.sh/formula/ffmpeg), using a Mac.

```bash
poetry install

jupyter notebook jhmdb_demo.ipynb
```

## Dependencies

This notebook's dependencies are listed in the [pyproject.toml](pyproject.toml) file.
The videos in this notebook make use of ffmpeg.
