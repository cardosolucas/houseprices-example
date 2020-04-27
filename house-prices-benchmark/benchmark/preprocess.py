import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import skew

def preprocess():
    dataset = pd.read_csv("data/train.csv")

    # Log transform the target
    dataset["SalePrice"] = np.log1p(dataset["SalePrice"])

    features = dataset.loc[:,'MSSubClass':'SaleCondition']
    dataset = dataset.drop(dataset.loc[:,'MSSubClass':'SaleCondition'], axis=1)

    # Log transform skewed numeric features
    numeric_feats = features.dtypes[features.dtypes != "object"].index

    skewed_feats = features[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    features[skewed_feats] = np.log1p(features[skewed_feats])

    # Convert categorical variables into dummy variables
    features = pd.get_dummies(features)

    # Filling NA's with the mean of the column
    features = features.fillna(features.mean())

    dataset = pd.concat([dataset, features], axis=1)
    train, test = train_test_split(dataset, test_size=0.3)
    train_y = train["SalePrice"]
    train_x = train.drop(["Id", "SalePrice"], axis=1)
    test_y = test["SalePrice"]
    test_x = test.drop(["Id", "SalePrice"], axis=1)

    save_dict = {
        'train_y': train_y,
        'train_x': train_x,
        'test_x': test_x,
        'test_y': test_y
    }
    return save_dict