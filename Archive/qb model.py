# QB Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import ff_functions as ff

# Pulling rb, qb and rec redzone data from csv files
df_qb = pd.read_csv('qb.csv')
df_rz_rec = pd.read_csv('rz_rec.csv')
df_rz_rush = pd.read_csv('rz_rush.csv')
df_def = pd.read_csv('team_def.csv')
df_draft = pd.read_csv('draft.csv')
df_adv_qb = pd.read_csv('adv_qb.csv')
#df_adv_rush = pd.read_csv('adv_rush.csv')

# Create a df that has all teams for all years as a columns and has a flag for each position and if the team had a top 10 pick in that postion
df_draft = df_draft[df_draft['Pick'] <= 10]
df_draft = df_draft[~df_draft['Position'].isin(['CB', 'DB', 'DE', 'DT', 'G', 'ILB', 'LB', 'OL', 'S', 'T'])]
df_draft = df_draft.groupby(['Team', 'Year', 'Position']).count().reset_index()
df_draft = df_draft.pivot(index=['Team', 'Year'], columns='Position', values='Player')
df_draft = df_draft.reset_index()
df_draft = df_draft.fillna(0)
df_draft['QB_top10'] = np.where(df_draft['QB'] > 0, 1, 0)
df_draft['RB_top10'] = np.where(df_draft['RB'] > 0, 1, 0)
df_draft['WR_top10'] = np.where(df_draft['WR'] > 0, 1, 0)
df_draft['TE_top10'] = np.where(df_draft['TE'] > 0, 1, 0)
df_draft = df_draft.drop(columns=['QB', 'RB', 'WR', 'TE'])

df_qb = df_qb.merge(df_draft[['Team', 'Year', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10']], how='left', left_on=['Tm', 'Year'], right_on=['Team', 'Year'])
df_qb = df_qb.merge(df_rz_rush[['Player', 'Year', 'Yds_20', 'Yds_10']], how='left', left_on=['Player', 'Year'], right_on=['Player', 'Year'], suffixes=('', '_rush'))
df_qb = df_qb.merge(df_def[['Tm', 'Year', 'Pass_QB_def_norm', 'Rush_def_norm']], how='left', left_on=['Tm', 'Year'], right_on=['Tm', 'Year'])
df_qb['Rush_share'] = df_qb['Att_rush'] / df_qb['Team_Att_rush']

# Dropping all columns whose YEAR is less than 2019 due to advanced stats not being available
df_qb = df_qb[df_qb['Year'] >= 2019]

#Cleaning advanced qb data, dropping all Tm columns
df_adv_qb = df_adv_qb.drop(columns=['Tm', 'Tm_x', 'Tm_y'])
#df_adv_rush = df_adv_rush.drop(columns=['Tm'])
df_qb = df_qb.merge(df_adv_qb, how='left', left_on=['Player', 'Year'], right_on=['Player', 'Year'])
#df_qb = df_qb.merge(df_adv_rush, how='left', left_on=['Player', 'Year'], right_on=['Player', 'Year'], suffixes=('', '_rush'))

# Getting a count of number of years a player has played in the NFL (excluding 2022)
df_qb['Year_count'] = df_qb.groupby('Player')['Year'].transform('count') - 1

# Dropping all players with less than 3 years of data
df_qb = df_qb[df_qb['Year_count'] >= 2]

# Shifting the data for each player by 1 year, changing all columns to have a _prev suffix
df_qb = df_qb.sort_values(by=['Player', 'Year'])
df_qb = df_qb.reset_index(drop=True)
df_qb = df_qb.rename(columns={'Player': 'Player'})

#Cleaning data
df_qb = df_qb.drop(columns=['Team', 'FantPos'])

#Replacing all values that have % with a decimal
for column in df_qb.columns:
    if '%' in column:
        df_qb[column] = df_qb[column].str.replace('%', '').astype(float) / 100

#Dropping all columns that have a % in the name
df_qb = df_qb[df_qb.columns.drop(list(df_qb.filter(regex='%')))]

# Use a for loop to create a new column for each stat with a _prev suffix
for col in df_qb.columns:
    if col not in ['Player', 'Year', 'Tm', 'Pos', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Pass_QB_def_norm', 'Rush_def_norm']:
        df_qb[col + '_prev'] = df_qb.groupby('Player')[col].shift(1)
        df_qb[col + '_2yr_prev'] = df_qb.groupby('Player')[col].shift(2)

#df_qb['Att_pass_prev'] = df_qb.groupby('Player')['Att_pass'].shift(1)
#df_qb['Att_pass_2yr_prev'] = df_qb.groupby('Player')['Att_pass'].shift(2)

#df_qb['Cmp_prev'] = df_qb.groupby('Player')['Cmp'].shift(1)
#df_qb['Cmp_2yr_prev'] = df_qb.groupby('Player')['Cmp'].shift(2)

#df_qb['Yds_pass_prev'] = df_qb.groupby('Player')['Yds_pass'].shift(1)
#df_qb['Yds_pass_2yr_prev'] = df_qb.groupby('Player')['Yds_pass'].shift(2)

#df_qb['TD_pass_prev'] = df_qb.groupby('Player')['TD_pass'].shift(1)
#df_qb['TD_pass_2yr_prev'] = df_qb.groupby('Player')['TD_pass'].shift(2)

#df_qb['Int_prev'] = df_qb.groupby('Player')['Int_pass'].shift(1)
#df_qb['Int_2yr_prev'] = df_qb.groupby('Player')['Int_pass'].shift(2)

#df_qb['Att_rush_prev'] = df_qb.groupby('Player')['Att_rush'].shift(1)
#df_qb['Att_rush_2yr_prev'] = df_qb.groupby('Player')['Att_rush'].shift(2)

#df_qb['TD_rush_prev'] = df_qb.groupby('Player')['TD_rush'].shift(1)
#df_qb['TD_rush_2yr_prev'] = df_qb.groupby('Player')['TD_rush'].shift(2)

#df_qb['Yds_20_prev'] = df_qb.groupby('Player')['Yds_20'].shift(1)
#df_qb['Yds_20_2yr_prev'] = df_qb.groupby('Player')['Yds_20'].shift(2)

#df_qb['Yds_10_prev'] = df_qb.groupby('Player')['Yds_10'].shift(1)
#df_qb['Yds_10_2yr_prev'] = df_qb.groupby('Player')['Yds_10'].shift(2)

#df_qb['Yds_rec_prev'] = df_qb.groupby('Player')['Yds_rec'].shift(1)
#df_qb['Yds_rec_2yr_prev'] = df_qb.groupby('Player')['Yds_rec'].shift(2)

#df_qb['Yds_rush_prev'] = df_qb.groupby('Player')['Yds_rush'].shift(1)
#df_qb['Yds_rush_2yr_prev'] = df_qb.groupby('Player')['Yds_rush'].shift(2)

#df_qb['Y/R_prev'] = df_qb.groupby('Player')['Y/R'].shift(1)
#df_qb['Y/R_2yr_prev'] = df_qb.groupby('Player')['Y/R'].shift(2)

#df_qb['Fmb_prev'] = df_qb.groupby('Player')['Fmb'].shift(1)
#df_qb['Fmb_2yr_prev'] = df_qb.groupby('Player')['Fmb'].shift(2)

#df_qb['G_prev'] = df_qb.groupby('Player')['G'].shift(1)
#df_qb['G_2yr_prev'] = df_qb.groupby('Player')['G'].shift(2)

#df_qb['GS_prev'] = df_qb.groupby('Player')['GS'].shift(1)
#df_qb['GS_2yr_prev'] = df_qb.groupby('Player')['GS'].shift(2)

#df_qb['PPR_prev'] = df_qb.groupby('Player')['PPR'].shift(1)
#df_qb['PPR_2yr_prev'] = df_qb.groupby('Player')['PPR'].shift(2)

#df_qb['PosRank_prev'] = df_qb.groupby('Player')['PosRank'].shift(1)
#df_qb['PosRank_2yr_prev'] = df_qb.groupby('Player')['PosRank'].shift(2)

#df_qb['OvRank_prev'] = df_qb.groupby('Player')['OvRank'].shift(1)
#df_qb['VBD_prev'] = df_qb.groupby('Player')['VBD'].shift(1)
#df_qb['FantPt_prev'] = df_qb.groupby('Player')['FantPt'].shift(1)
#df_qb['DKPt_prev'] = df_qb.groupby('Player')['DKPt'].shift(1)
#df_qb['FDPt_prev'] = df_qb.groupby('Player')['FDPt'].shift(1)
#df_qb['Team_Off_prev'] = df_qb.groupby('Player')['Team_Yds'].shift(1)
#df_qb['Y/A_prev'] = df_qb.groupby('Player')['Y/A'].shift(1)

#df_qb['Rush_share_prev'] = df_qb.groupby('Player')['Rush_share'].shift(1)
#df_qb['Rush_share_2yr_prev'] = df_qb.groupby('Player')['Rush_share'].shift(2)

#df_qb['Yds_20_prev'] = df_qb.groupby('Player')['Yds_20'].shift(1)
#df_qb['Yds_20_2yr_prev'] = df_qb.groupby('Player')['Yds_20'].shift(2)

#df_qb['Yds_10_prev'] = df_qb.groupby('Player')['Yds_10'].shift(1)
#df_qb['Yds_10_2yr_prev'] = df_qb.groupby('Player')['Yds_10'].shift(2)
df_qb = df_qb.fillna(0)

# Recursively looking at each players history in Yds_rec and creating a historical average for each player excluding the current and prior year data
df_qb['PosRank_historical_avg'] = df_qb.groupby('Player')['PosRank'].transform(lambda x: x.expanding().mean().shift(2))
df_qb['PPR_historical_avg'] = df_qb.groupby('Player')['PPR'].transform(lambda x: x.expanding().mean().shift(2))
df_qb['Yds_pass_historical_avg'] = df_qb.groupby('Player')['Yds_pass'].transform(lambda x: x.expanding().mean().shift(2))
df_qb['TD_pass_historical_avg'] = df_qb.groupby('Player')['TD_pass'].transform(lambda x: x.expanding().mean().shift(2))
df_qb['GS_historical_avg'] = df_qb.groupby('Player')['GS'].transform(lambda x: x.expanding().mean().shift(2))
df_qb['Yds_rush_historical_avg'] = df_qb.groupby('Player')['Yds_rush'].transform(lambda x: x.expanding().mean().shift(2))
df_qb['Int_historical_avg'] = df_qb.groupby('Player')['Int_pass'].transform(lambda x: x.expanding().mean().shift(2))

# Dropping all columns that dont have a _prev suffix except for Player, Year and PPR
# Use a loop
for col in df_qb.columns:
    if '_prev' not in col and col not in ['Player', 'Year', 'PPR', 'qb_PPR', 'Year_count', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10', 'Yds_rec_historical_avg', 'GS_historical_avg', 'PosRank_historical_avg', 'PPR_historical_avg', 'Tgt_historical_avg', 'Tgt_share_historical_avg', 'Yds_rec_share_historical_avg', 'Yds_rec_historical_avg', 'TD_rec_historical_avg', 'TD_rec_share_historical_avg', 'Yds_pass_historical_avg', 'TD_pass_historical_avg', 'Yds_rush_historical_avg', 'Int_historical_avg', 'Rush_share_historical_avg', 'Yds_20_historical_avg', 'Yds_10_historical_avg', 'Yds_20_share_historical_avg', 'Yds_10_share_historical_avg', 'Y/R_historical_avg', 'Fmb_historical_avg', 'G_historical_avg', 'GS_historical_avg', 'PPR_historical_avg', 'PosRank_historical_avg', 'OvRank_historical_avg', 'VBD_historical_avg', 'FantPt_historical_avg', 'DKPt_historical_avg', 'FDPt_historical_avg', 'Team_Off_historical_avg', 'Y/A_historical_avg']:
        df_qb = df_qb.drop(columns=col)

# Normalizing all columns that have a _prev suffix except any column that has rank in the name
for col in df_qb.columns:
    if col not in ['Player', 'Year', 'PPR', 'Pass_QB_def_norm', 'Rush_def_norm', 'QB_top10', 'RB_top10', 'WR_top10', 'TE_top10']:
        df_qb[col] = (df_qb[col] - df_qb[col].mean()) / df_qb[col].std()

# Dropping the smallest 2 years for each player
df_qb = df_qb.sort_values(by=['Player', 'Year'])
df_qb = df_qb.reset_index(drop=True)
df_qb = df_qb.groupby('Player').apply(lambda x: x.iloc[2:]).reset_index(drop=True)

# Testing to see if the model is better with or without certain columns
df_qb = df_qb.drop(columns=['FDPt_prev'])
#df_qb = df_qb.drop(columns=['qb_PPR'])
df_qb = df_qb.drop(columns=['Year_count'])
df_qb = df_qb.drop(columns=['FantPt_prev'])
df_qb = df_qb.drop(columns=['DKPt_prev'])

#df_qb = df_qb.drop(columns=['Fmb_prev'])

#df_qb = df_qb.drop(columns=['Y/R_prev'])
#df_qb = df_qb.drop(columns=['OvRank_prev'])
#--------------------
#df_qb = df_qb.drop(columns=['Tgt_share_prev'])
#df_qb = df_qb.drop(columns=['Tier_1_Sign'])
#df_qb = df_qb.drop(columns=['PPR_prev'])
#df_qb = df_qb.drop(columns=['PPR_2yr_prev'])
#df_qb = df_qb.drop(columns=['PosRank_prev'])

#df_qb = df_qb[['Player', 'Year', 'PPR', 'PPR_prev', 'PosRank_prev', 'Yds_rush_prev', 'Rush_share_prev', 'Yds_20_rush_prev']]

# Run model evaluation function 10 times and average the and r2 values and add average PPR score to result dataframe
result, r2, mae, rb_coef, best = ff.evaluate_model(df_qb, 2022, 'PPR', 'RandomForestRegressor')
x_result, x_r2, x_mae, x_rb_coef, x_best = ff.evaluate_model(df_qb, 2022, 'PPR', 'XGBRegressor')
s_result, s_r2, s_mae, s_rb_coef, s_best = ff.evaluate_model(df_qb, 2022, 'PPR', 'SVR')

print('Average R2: ', r2)
print('Average MSE: ', mae)
print('Best R2: ', best)

# add a column that is the difference between the players PPR and the predicted PPR
result['PPR_diff'] = result['PPR'] - result['Predicted']
s_result['PPR_diff'] = s_result['PPR'] - s_result['Predicted']
x_result['PPR_diff'] = x_result['PPR'] - x_result['Predicted']

X = df_qb.drop('PPR', axis=1)
y = df_qb['PPR']