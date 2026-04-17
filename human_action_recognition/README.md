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

### Warning

This notebooks in this repository are for demonstration purposes only. The code here might not be suitable for
deployment. Some of the dependencies are out of date and contain known security vulnerabilities. We strongly recommend
that you review the code and dependencies for each notebook to ensure they are consistent with your organization's
security policies, current best-practices, and any compliance requirements before using them for any production
deployments.