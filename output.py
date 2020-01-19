"""
produce markdown

1. Read in 2020 data
2. 

"""
from config import config
import pandas as pd
import numpy as np
import sklearn.metrics as met

def benchmark_winpct():
    """Extract the win percentage"""
    mrd2020 = config.get('dir', 'mrd2020')
    df = pd.read_csv(mrd2020)
    # TODO: why does the 2020 data have results past today's date???

def breakdown_accuracy(df, result, proba, pred, name):
    """Take a model score and break it down.
    pred is a 0/1 column, where proba is the probabilty score for home team/team1"""
    acc = {}
    acc['all'] = [met.accuracy_score(y_true = df[result], y_pred = df[pred])]
    for i in np.arange(0.5, 1.0, 0.1):
        min = round(i, 1)
        max = round(min + 0.1, 1)
        df_filt = df[(min < df[proba]) & (df[score] <= max)]
        _acc = met.accuracy_score(y_true = df_filt[result], y_pred=df_filt[pred])
        acc[f'{str(min)}:{str(max)}'] = [_acc]
    
    df_acc = pd.DataFrame(acc)
    df_acc.index = [name]
    return df_acc

def benchmark_538():
    # Scrape Nate Silver's results
    df = pd.read_csv('https://projects.fivethirtyeight.com/nba-model/nba_elo_latest.csv')
    df['result'] = (df.score1 > df.score2).astype('int')
    # Look at past results only!
    df = df[df.date < config.get('date', 'today')]

    models = ['elo', 'carm-elo', 'raptor']
    df_acc = pd.DataFrame()
    for model in models:
        df[f'{model}_pred'] = (df[f'{model}_prob1'] > df[f'{model}_prob2']).astype('int')
        acc = breakdown_accuracy(df, result='result', proba=f'{model}_prob1', pred=f'{model}_pred', name=model)
        df_acc = df_acc.append(acc)

    return df_acc.transpose()

