import argparse
import os
import pandas as pd


def construct_data(group, t, raw_path='../datasets/raw',
                   save_path='../datasets/processed'):
    raw_path = f'{raw_path}/{group}_t{t}.csv'
    df = pd.read_csv(raw_path)
    label1 = f"t{t}_close_return_rate"
    label2 = f"t{t}_close_return_rate_label"
    features = df.columns.to_list()
    features = [x for x in features if 'return' not in x]
    features = list(set(features) - set(['kdcode', 'dt']))
    df[label2] = df.apply(
        lambda x: 1.0 if x[label1] > 0 else (-1.0 if x[label1] <= 0 else 0.0), axis=1)

    p = df.groupby("kdcode")['dt'].count() == (df['dt'].nunique())
    p = p.reset_index()
    stock_list = p[p['dt'] == True]['kdcode'].to_list()
    print(f'the number of shares with everyday data：{len(stock_list)}')
    trade_dates = df['dt'].unique()
    print(f'total {len(trade_dates)} days')
    df = df[df.kdcode.isin(stock_list)]
    daily_data_df = df.copy()
    for f in features:
        daily_data_df[f] = (daily_data_df[f] - daily_data_df[f].mean()) / daily_data_df[f].std()
    base_file_path = f"{save_path}/{group}_t{t}/"
    if not os.path.exists(base_file_path):
        os.mkdir(base_file_path)
    for name, content in daily_data_df.groupby('kdcode'):
        file_name = f'{base_file_path}{name}.csv'
        content[features + [label2, label1]].to_csv(file_name, header=False, index=False)
    if not os.path.exists(f'{save_path}/trading_dates.csv'):
        kk = pd.DataFrame(trade_dates)
        kk.to_csv(path_or_buf=f'{save_path}/trading_dates.csv', header=False, index=False, index_label=None)
    print(f'finish constructing of {group}_t{t}')


if __name__ == "__main__":
    # construct_data("p", 3)
    # construct_data("p", 10)
    for g in ['p', 'pf', 'pr', 'pi']:
        for t in [3, 5, 10]:
            construct_data(g, t)
