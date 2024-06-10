import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from typing import Tuple

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def train(df: pd.DataFrame)->Tuple[DictVectorizer, LinearRegression]:
    feature = ['PULocationID', 'DOLocationID']
    target = ['duration']
    dv = DictVectorizer()
    train_dict = df[feature].to_dict(orient="records")
    X_train = dv.fit_transform(train_dict)
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("Model Intercept ",lr.intercept_)
    return dv, lr


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'