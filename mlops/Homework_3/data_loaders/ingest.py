import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def load_data(*args, **kwargs)-> pd.DataFrame:
    data_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    try:
        df = pd.read_parquet(data_url)
        print("Data loaded successfully")
        print("Number of records: ", len(df))
    except Exception as e:
        print(f"Unexpected error occurred. {e}")
    return df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'