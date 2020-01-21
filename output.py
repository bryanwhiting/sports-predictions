"""
produce markdown

1. Read in 2020 data
2. 

# Todo:
- Make graph of accuracy over time
    - Add a feature which is season_week
    - Group by season_week, calculate accuracy for all model types.

"""
from config import config, save
import pandas as pd
import numpy as np
from prefect import task, Flow
from datetime import date
from sklearn.metrics import accuracy_score

# @task
def df_with_538():
    """Create a combined dataset
    Pulls from mrd_yr2020 and 538, checks they have the same scores
    and then merges them together

    df5 = pd.read_csv('https://projects.fivethirtyeight.com/nba-model/nba_elo_latest.csv')
    df5 = df5[['date', 'team1', 'team2', 'elo_prob1', 'carm-elo_prob1', 'raptor_prob1', 'score1', 'score2']]
    """
    # 538 dataset:
    df5 = pd.read_csv('https://projects.fivethirtyeight.com/nba-model/nba_elo_latest.csv')
    df5 = df5[['date', 'team1', 'team2', 'elo_prob1', 'carm-elo_prob1', 'raptor_prob1', 'score1', 'score2']]
    # My MRD
    mrd2020 = config.get('dir', 'mrd2020')
    df = pd.read_csv(mrd2020)
    df = df[df.date < config.get('date', 'today')]
    df = df.merge(df5, on=['date', 'team1', 'team2'])

    if len(df[df['pg_score1'] != df['score1']]) > 0:
        print("WARNING score1 differs. consider re-pulling your data")
    
    if len(df[df['pg_score2'] != df['score2']]) > 0:
        print("WARNING score2 differs. consider re-pulling your data")

    # Remove duplicated 538 columns
    df = df.drop(['score1', 'score2'], axis=1)

    # Calculate season_week
    df['season_week'] = (pd.to_datetime(df.date).dt.date - date(2019, 10, 20)).apply(lambda x: x.days//7)
    return df


def breakdown_accuracy(df, result, proba, pred, name):
    """Take a model score and break it down.
    pred is a 0/1 column, where proba is the probabilty score for home team/team1"""
    acc = {}
    acc['all'] = [accuracy_score(y_true = df[result], y_pred = df[pred])]
    for i in np.arange(0.0, 1.0, 0.1):
        min = round(i, 1)
        max = round(min + 0.1, 1)
        df_filt = df[(min < df[proba]) & (df[proba] <= max)]
        # Returns calibration (should be between bounds), which is 1-acc when prob < 0.5
        _acc = df_filt.result.mean()
        # _acc = accuracy_score(y_true = df_filt[result], y_pred=df_filt[pred])
        acc[f'{str(min)}:{str(max)}'] = [_acc]
    
    df_acc = pd.DataFrame(acc)
    df_acc.index = [name]
    return df_acc

# @task
def benchmark_winpct(df):
    """Extract the win percentage"""

    # home team win pct
    acc_home = df.result.mean() 

    # Prediction = Home team win pct > 0.5
    df['pred_home_winpct'] = (df['team1_lag1_win_pct'] > 0.5).astype('int')
    acc_home_win_pct = breakdown_accuracy(df, result='result', proba='team1_lag1_win_pct', pred='pred_home_winpct', name='home_winpct')
    df['acc_pred_home_winpct'] = (df['result'] == df['pred_home_winpct']).astype('int')

    # home win pct > away win pct
    pred = (df['team1_lag1_win_pct'] > df['team2_lag1_win_pct']).astype('int')
    acc_home_vs_away = accuracy_score(y_true=df.result, y_pred=pred)

    df_acc = acc_home_win_pct.append(pd.DataFrame({
        'all': [acc_home, acc_home_vs_away]
    }, index=['home_overall', 'home_vs_away_winpct']),
    sort=False)

    # Does prediction improve after filtering out first month?
    # df = df[df.ymdhms > '2019-12-01']
    # pred = (df['home_lag1_win_pct'] > .5).astype('int')
    # accuracy_score(y_true=df.result, y_pred=pred)
    
    # Predicting using the away team win pct
    # pred = 1-(df['away_lag1_win_pct'] > .5).astype('int')
    # accuracy_score(y_true=df.result, y_pred=pred)

    return df_acc

# @task
def benchmark_538(df):
    """
    Compare accuracy with result.mean
    x = df[(df.elo_prob1 > 0.2) & (df.elo_prob1 < 0.3)]
    # Accuracy:
    print(accuracy_score(x.result, x.elo_prob1 > .5))
    print(x.result == (x.elo_prob1 > 0.5).astype('int'))
    # Calibration (should be between 0.2 and 0.3 if calibrated)
    print(x.result.mean())
    """
    # Compare Nate Silver's results

    models = ['elo', 'carm-elo', 'raptor']
    df_acc = pd.DataFrame()
    for model in models:
        df[f'pred_{model}'] = (df[f'{model}_prob1'] > .5).astype('int')
        acc = breakdown_accuracy(df, result='result', proba=f'{model}_prob1', pred=f'pred_{model}', name=model)
        df_acc = df_acc.append(acc)

        # Save out binary accuracy for groupby-accuracy by season_week later
        df[f'acc_pred_{model}'] = (df[f'pred_{model}'] == df['result']).astype('int')

    return df_acc

# @task
def week_accuracies(df):
    pred_cols = [c for c in df.columns if c.startswith('acc_pred_')]
    # Within-week accuracies
    df_within_week =  (df.groupby(['season_week'])[pred_cols].mean().reset_index())

    # up-to-week accuracy 
    max_weeks = df.season_week.max()
    df_upto_week = pd.DataFrame()
    for i in range(0, max_weeks + 1):
        df_filt = df[df.season_week <= i]
        df_week = pd.DataFrame(df_filt[pred_cols].mean()).transpose()
        df_week['season_week'] = i
        df_upto_week = df_upto_week.append(df_week)

    df_upto_week = df_upto_week[['season_week'] + pred_cols]
    return (df_within_week, df_upto_week)

# @task
def combine_benchmarks(df_acc1, df_acc2):
    df_acc = df_acc1.append(df_acc2).transpose().fillna('')
    return df_acc


@task
def main():
    df = df_with_538()
    # Each of these functions is updating df
    df_acc1 = benchmark_winpct(df)
    df_acc2 = benchmark_538(df)
    df_acc = combine_benchmarks(df_acc1, df_acc2)
    # pandas groupby means
    df_within_week, df_upto_week = week_accuracies(df)
    # Save out
    save(df_acc, dir='output', filename='acc_overall', main=True, date=True)
    save(df_within_week, dir='output', filename='acc_within_week', main=True, date=True)
    save(df_upto_week, dir='output', filename='acc_upto_week', main=True, date=True)


with Flow('Benchmarks') as flow_bench:
    # NOTE TO SELF:
    # I didn't use prefect for individual tasks becasue I saw
    # some reallys trange behavior where the accuracies weren't saving out
    # That, and I like putting breakpoints in the main to see
    # how data are passed from one task to another
    m = main()


if __name__ == "__main__":
    state = flow_bench.run()
    # Debug:
    # state.result[df_weeks].result