# Using Path Signatures for Anomaly Detection in Natural Language Processing

[alphabet_analysis.ipynb](alphabet_analysis.ipynb) is a [Jupyter](https://jupyter.org/) notebook which demonstrates the use of path signatures for anomaly detection in natural language processing. We use [RoughPy](https://roughpy.org/) for computing path signatures and use the [`signature_mahalanobis_knn`](https://github.com/datasig-ac-uk/signature_mahalanobis_knn) library for anomaly detection. The full details of the anomaly detection methodology can be found in [[1]](https://arxiv.org/abs/2006.03487) _Dimensionless Anomaly Detection on Multivariate Streams with Variance Norm and Path Signature_ by Zhen Shao, Ryan Sze-Yin Chan, Thomas Cochrane, Peter Foster, Terry Lyons.


Note that the data is pre-processed in [language_dataset_anomalies_data.ipynb](language_dataset_anomalies_data.ipynb) and we train a character-level RoBERTa model in [train_english_char_bert.ipynb](train_english_char_bert.ipynb).

## How to Run

A typical process for installing the package dependencies involves creating a new Python virtual environment and then inside the environment executing:

```
pip install -r requirements.txt
```

This notebook was developed with Python=3.11.8. With running the notebook in Jupyter, you may need to run the following to install an IPython kernel:

```
python -m ipykernel install --user --name=anomaly-detection-roughpy-env
```
