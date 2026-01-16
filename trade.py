import os
import pandas as pd
import numpy as np
import glob

def calculate_alpha_statistics(file_path, index_df):
    df_return = pd.read_csv(file_path,index_col=False)
    # print(df_return)
    df_return = df_return[df_return['datetime'].isin(index_df['datetime'])]
    index_df_filtered = index_df[index_df['datetime'].isin(df_return['datetime'])]
    
    portfolio_df_performance = df_return.set_index(['datetime'])
    index_df_performance = index_df_filtered.set_index(['datetime'])
    
    alpha_df_performance = pd.DataFrame()
    alpha_df_performance['portfolio_daily_return'] = portfolio_df_performance['daily_return']
    alpha_df_performance['index_daily_return'] = index_df_performance['daily_return']
    alpha_df_performance['alpha_daily_return'] = alpha_df_performance['portfolio_daily_return'] - alpha_df_performance['index_daily_return']
    alpha_df_performance['portfolio_net_value'] = (alpha_df_performance['portfolio_daily_return'] + 1).cumprod()
    alpha_df_performance['index_net_value'] = (alpha_df_performance['index_daily_return'] + 1).cumprod()
    alpha_df_performance['alpha_net_value'] = (alpha_df_performance['alpha_daily_return'] + 1).cumprod()
    
    net_value_columns = ['portfolio_net_value', 'index_net_value', 'alpha_net_value']
    
    alpha_statistics_df = pd.DataFrame(index=alpha_df_performance[net_value_columns].columns, 
                                       columns=["年化收益", "年化波动率", "最大回撤率", "夏普率", "Calmar", "IR", "月度胜率"])
    
    alpha_df_performance.index = pd.to_datetime(alpha_df_performance.index)
    monthly_statistics_df = alpha_df_performance[net_value_columns].resample('m').last()
    monthly_statistics_df = pd.concat([alpha_df_performance[:1][net_value_columns], monthly_statistics_df])
    monthly_statistics_df = monthly_statistics_df.pct_change().dropna()
    monthly_statistics_df.index = monthly_statistics_df.index.date
    
    yearly_statistics_df = alpha_df_performance[net_value_columns].resample('y').last()
    yearly_statistics_df = pd.concat([alpha_df_performance[:1][net_value_columns], yearly_statistics_df])
    yearly_statistics_df = yearly_statistics_df.pct_change().dropna()
    yearly_statistics_df.index = yearly_statistics_df.index.date
    
    alpha_statistics_df.loc[:, "年化收益"] = ((alpha_df_performance[net_value_columns].tail(1)) ** (252 / len(alpha_df_performance)) - 1).mean()
    alpha_statistics_df.loc[:, "年化波动率"] = np.std(alpha_df_performance[net_value_columns] / alpha_df_performance[net_value_columns].shift(1) - 1, axis=0) * np.sqrt(252)
    alpha_statistics_df.loc[:, "累积收益"] = (alpha_df_performance[net_value_columns].tail(1) - 1).mean()
    alpha_statistics_df.loc[:, "累积波动率"] = np.std(alpha_df_performance[net_value_columns] / alpha_df_performance[net_value_columns].shift(1) - 1, axis=0)
    alpha_statistics_df.loc[:, "最大回撤率"] = ((alpha_df_performance[net_value_columns] - alpha_df_performance[net_value_columns].cummax()) / alpha_df_performance[net_value_columns].cummax()).min()
    alpha_statistics_df.loc[:, "夏普率"] = alpha_statistics_df["年化收益"] / alpha_statistics_df["年化波动率"]
    alpha_statistics_df.loc[:, "Calmar"] = alpha_statistics_df["年化收益"] / abs(alpha_statistics_df["最大回撤率"])
    alpha_statistics_df.loc[:, "IR"] = (alpha_df_performance[net_value_columns] / alpha_df_performance[net_value_columns].shift(1) - 1).mean() * np.sqrt(252) / np.std(alpha_df_performance[net_value_columns] / alpha_df_performance[net_value_columns].shift(1) - 1, axis=0)
    alpha_statistics_df.loc[:, "月度胜率"] = monthly_statistics_df[monthly_statistics_df > 0].count() / monthly_statistics_df.count()
    
    return alpha_statistics_df

dataset = 'sp500'
# Load index data
index_df = pd.read_csv(f'/mnt/petrelfs/chengdawei/lustre/wavlet/dataset/{dataset}_index_2024.csv',index_col=False)

# File path pattern
file_pattern = f'/mnt/petrelfs/chengdawei/lustre/wavlet/{dataset}_data/return_{dataset}_*_0_0_6_pt.csv'

# Iterate over all matching files
for file_path in glob.glob(file_pattern):
    portfolio_net_value_stats = calculate_alpha_statistics(file_path, index_df)
    # print(f"Results for {file_path}:")
    print(portfolio_net_value_stats)
