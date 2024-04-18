import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import log_loss, f1_score

mlflow.set_tracking_uri("sqlite:///mlruns.db")

exp_name = 'Aplicação - Projeto Kobe'
data_cols = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']

exp_kobe_project_app = mlflow.get_experiment_by_name(exp_name)

if exp_kobe_project_app is None:
    exp_id = mlflow.create_experiment(exp_name)
    exp_kobe_project_app = mlflow.get_experiment(exp_id)
exp_id = exp_kobe_project_app.experiment_id

with mlflow.start_run(experiment_id=exp_id, run_name='PipelineAplicacao'):
    
    model_uri = f'models:/model_kobe@staging'
    loaded_model = mlflow.sklearn.load_model(model_uri)
    data_prod = pd.read_parquet('../data/raw/dataset_kobe_prod.parquet')
    data_prod.dropna(inplace=True)

    data_prod_features = data_prod[data_cols]
    y_predicted = loaded_model.predict_proba(data_prod_features)[:,1]
    data_prod['predict_score'] = y_predicted
    data_prod.to_parquet('../data/processed/prediction_prod.parquet')
    mlflow.log_artifact('../data/processed/prediction_prod.parquet')

    mlflow.log_metric('log_loss', log_loss(data_prod['shot_made_flag'], y_predicted))
    mlflow.log_metric('f1_score', f1_score(data_prod['shot_made_flag'], y_predicted.round()))