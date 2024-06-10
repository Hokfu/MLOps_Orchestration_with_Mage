import mlflow
import mlflow.sklearn
import pickle
import os
if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

if not os.path.exists('models'):
    os.mkdir('models')
else:
    print("Directory 'models' already exists.")

@data_exporter
def export_data(dv_lr):
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("yellow-taxi-experiment")
    dv, lr = dv_lr
    with open('models/dv.bin','wb') as fout:
        pickle.dump(dv,fout)
    with open('models/lin_reg.bin','wb') as fout:
        pickle.dump(lr,fout)
    with mlflow.start_run():
        mlflow.log_artifact(local_path='models/dv.bin', artifact_path='dv_pickle')
        mlflow.sklearn.log_model(sk_model=lr,
        artifact_path='models/lin_reg.bin',registered_model_name='linear regression model')



