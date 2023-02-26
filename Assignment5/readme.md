# Assignment 5: Categorical data

In [this workbook](prasmus3_aisys_assignment5.ipynb), we process a [used cars dataset from Kaggle](https://www.kaggle.com/datasets/lepchenkov/usedcarscatalog) to prepare it for use in a deep learning algorithm. The raw dataset included multiple nominal categorical features which we dummied. We imputed missing data for ten observations. We dropped one categorical feature that had over 1000 categories and whose inclusion would have made the dataset undesirably sparse.

The notebook does the following:
* Downloads [data](https://github.com/pgr-me/data/tree/main/aisys/module5/cars.csv)
* Drops columns
* Drops observations (in this case we impute and do not do this)
* Fills in missing data
* Dummies nominal columns

The output is ready for use in a deep learning neural network or any machine learning algorithm that requires numeric data.

Please note this notebook is also available on [Google Colab](https://colab.research.google.com/drive/1cEk6Dsjm6vFMfE9-23t5v7eUbznj8S_h#scrollTo=aNvRH-1cc0zt).
