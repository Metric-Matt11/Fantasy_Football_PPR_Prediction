# Description: This program will set a value for each player 
#              based on their performance in the previous 3 season.
#              The idea for this model is to create a ranking for a fantasy football draft.

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dtale
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import ff_functions as ff
from pathlib import Path
import os

# Creating dataframes for each position, defense, and schedule
#df_qb, df_rb, df_wr, df_te = ff.player_scrape(2019, 2022)
#df_def = ff.team_def_scrape(2019, 2022)
#df_schedule = ff.nfl_schedule(2019, 2022)
df_qb_adv, df_qb_adv_team = ff.qb_adv_stats(2022, 2024) #----------------NEED TO LOAD
#df_rz_pass, df_rz_rush, df_rz_rec = ff.redzone_scrape(2022, 2024) #------NEED TO LOAD

# Define possible directories
possible_paths = [
    Path.home() / "OneDrive - WellSky" / "Documents" / "Python Projects" / "FF",
    Path.home() / "Documents" / "Python Projects" / "Fantasy football"
]

# Find the correct path
project_dir = None
for p in possible_paths:
    if p.exists():
        project_dir = p
        break

if not project_dir:
    raise FileNotFoundError("No valid project folder found.")

# Change working directory
os.chdir(project_dir)
print(f"Working in: {os.getcwd()}")

# List of files to load
files = {
    "df_wr": "wr",
    "df_te": "te",
    "df_qb": "qb",
    "df_rb": "rb",
    "df_def": "team_def",
    "adv_rec": "adv_rec",
    "df_rz_rec": "rz_rec",
    "df_rz_pass": "rz_pass",
    "df_rz_rush": "rz_rush",
    "schedule_2022": "nfl_schedule_2022",
    "schedule_2023": "nfl_schedule_2023",
    "schedule_2024": "nfl_schedule_2024"
}

# Read CSVs into a dictionary of DataFrames
data = {}
for key, base_name in files.items():
    csv_path = project_dir / f"{base_name}.csv"
    xlsx_path = project_dir / f"{base_name}.xlsx"

    if csv_path.exists():
        data[key] = pd.read_csv(csv_path)
        print(f"Loaded CSV: {csv_path.name}")
    elif xlsx_path.exists():
        data[key] = pd.read_excel(xlsx_path)
        print(f"Loaded XLSX: {xlsx_path.name}")
    else:
        print(f"⚠️ File not found: {base_name}.csv or {base_name}.xlsx")

df_wr = data.get("wr")
df_qb = data.get("qb")
df_rb = data.get("rb")
df_te = data.get("te")
df_def = data.get("team_def")
#df_qb_adv = data.get("adv_rec")
df_qb_adv_team = data.get("adv_rec")
df_rz_rec = data.get("rz_rec")
df_rz_pass = data.get("rz_pass")
df_rz_rush = data.get("rz_rush")

years = [2022, 2023, 2024]
df_list = []

for year in years:
    df = data.get(f"schedule_{year}")
    if df is None:
        print(f"⚠️ Missing schedule file for {year}")
        continue

    # Add Year column
    df['Year'] = year
    df_list.append(df)

# Combine all into one DataFrame
df_schedule = pd.concat(df_list, ignore_index=True)

#Dropping players whose team is not in the nfl_schedule dataframe
#This should drop players with no team or multiple teams

for df_name in ['df_qb', 'df_rb', 'df_wr', 'df_te']:
    df = locals().get(df_name)
    if df is not None:
        locals()[df_name] = df[df['Tm'].isin(df_schedule['TEAM'])]
    else:
        print(f"⚠️ Warning: {df_name} is None and cannot be filtered by team.")

#########################GOT TO HERE

# Joining df_qb_adv to df_qb on player and year
df_qb = df_qb.merge(df_qb_adv[['Player', 'Year', 'OnTgt']], how='left', left_on=['Player', 'Year'], right_on=['Player', 'Year'])

# Creating a df with qbs that have the most OnTgts for each team and year
starting_qb_team = df_qb.groupby(['Tm', 'Year'])['OnTgt'].max().reset_index()
starting_qb = df_qb.merge(starting_qb_team, how='inner', left_on=['Tm', 'Year', 'OnTgt'], right_on=['Tm', 'Year', 'OnTgt'])

# Creating a column that is the OnTgt for a players previous year
starting_qb['OnTgt_prev'] = starting_qb.groupby('Player')['OnTgt'].shift(1)
starting_qb['OnTgt_prev_norm'] = starting_qb.groupby('Player')['OnTgt_prev'].transform(lambda x: (x - x.mean()) / x.std())
starting_qb = starting_qb.fillna(0)

# Joining starting starting_qb to df_wr on team and year.
df_wr = df_wr.merge(starting_qb[['Tm', 'Year', 'Player', 'OnTgt_prev_norm']], how='left', left_on=['Tm', 'Year'], right_on=['Tm', 'Year'])
df_wr = df_wr.merge(df_rz_rec[['Player', 'Year', 'Yds_20', 'Yds_10']], how='left', left_on=['Player_x', 'Year'], right_on=['Player', 'Year'])

# Creating WR features
df_wr['Tgt_share'] = df_wr['Tgt'] / df_wr['Team_Tgt']

#Replace TD_share with Endzone and Redzone targets
#df_wr['TD_share'] = df_wr['TD_rec'] / df_wr['Team_TD']

# Joining df_def_sched by the opponent that week and year to df_schedule
# Using a loop to go through each week and year
# Curretnly only using 17 weeks since 2020 does not have a week 18
for x in range(1, 18):
    df_schedule = df_schedule.merge(df_def, how='left', left_on=['week_' + str(x), 'Year'], right_on=['Tm', 'Year'])
    df_schedule = df_schedule.rename(columns={'Pass_QB_def_norm': 'Pass_QB_def_norm_' + str(x), 'Rush_def_norm': 'Rush_def_norm_' + str(x)})
    df_schedule = df_schedule.drop(columns=['Tm'])
df_schedule = df_schedule.fillna(0)

# Creating a column that is the sum of the normalized defensive stats using a loop
# This will be used to determine the strength of the defense for each team
df_schedule['Pass_QB_def_norm_sum'] = 0
df_schedule['Rush_def_norm_sum'] = 0
for x in range(1, 18):
    df_schedule['Pass_QB_def_norm_sum'] = df_schedule['Pass_QB_def_norm_sum'] + df_schedule['Pass_QB_def_norm_' + str(x)]
    df_schedule['Rush_def_norm_sum'] = df_schedule['Rush_def_norm_sum'] + df_schedule['Rush_def_norm_' + str(x)]
df_schedule = df_schedule[['TEAM', 'Year', 'Pass_QB_def_norm_sum', 'Rush_def_norm_sum']]

# Joining df_schedule to each position dataframe
df_qb = df_qb.merge(df_schedule, how='left', left_on=['Tm', 'Year'], right_on=['TEAM', 'Year'])
df_rb = df_rb.merge(df_schedule, how='left', left_on=['Tm', 'Year'], right_on=['TEAM', 'Year'])
df_wr = df_wr.merge(df_schedule, how='left', left_on=['Tm', 'Year'], right_on=['TEAM', 'Year'])
df_te = df_te.merge(df_schedule, how='left', left_on=['Tm', 'Year'], right_on=['TEAM', 'Year'])

# Joining df_qb_adv_team to df_wr
df_wr = df_wr.merge(df_qb_adv_team, how='left', left_on=['Tm', 'Year'], right_on=['Tm', 'Year'])
df_wr['Yds_rec/Att'] = df_wr['Yds_rec'] / df_wr['Team_Att']

# Creating a column with the count of how many years of data we have for each player
df_wr['Years'] = df_wr.groupby('Player_x')['Year'].transform('count')
df_wr = df_wr[df_wr['Years'] >= 4]
df_wr = df_wr.fillna(0)

# Droping columns that are not needed
df_wr = df_wr.drop(columns=['Player_y', 'Player', 'Tm', 'TEAM', 'Years'])

# Shifting the data for each player by 1 year, changing all columns to have a _prev suffix
df_wr = df_wr.sort_values(by=['Player_x', 'Year'])
df_wr = df_wr.reset_index(drop=True)
df_wr = df_wr.rename(columns={'Player_x': 'Player'})
df_wr['Tgt_prev'] = df_wr.groupby('Player')['Tgt'].shift(1)
df_wr['Rec_prev'] = df_wr.groupby('Player')['Rec'].shift(1)
df_wr['Yds_rec_prev'] = df_wr.groupby('Player')['Yds_rec'].shift(1)
df_wr['TD_rec_prev'] = df_wr.groupby('Player')['TD_rec'].shift(1)
df_wr['Tgt_share_prev'] = df_wr.groupby('Player')['Tgt_share'].shift(1)
df_wr['Team_Yds_prev'] = df_wr.groupby('Player')['Team_Yds'].shift(1)
df_wr['Team_Att_prev'] = df_wr.groupby('Player')['Team_Att'].shift(1)
df_wr['Yds_rec/Att_prev'] = df_wr.groupby('Player')['Yds_rec/Att'].shift(1)
df_wr['Yds_rec/Att_prev2'] = df_wr.groupby('Player')['Yds_rec/Att'].shift(2)
df_wr['Yds_20_prev'] = df_wr.groupby('Player')['Yds_20'].shift(1)
df_wr['Yds_10_prev'] = df_wr.groupby('Player')['Yds_10'].shift(1)
df_wr['VBD_prev'] = df_wr.groupby('Player')['VBD'].shift(1)
df_wr['PPR_prev'] = df_wr.groupby('Player')['PPR'].shift(1)
df_wr = df_wr.fillna(0)

# Dropping columns that dont have _prev suffix with exception of PPR, Player, Year and age
wr_model = df_wr.drop(columns=['Tgt', 'Rec', 'Yds_rec', 'TD_rec', 'Tgt_share', 'Team_Yds', 'Team_Att', 'Yds_rec/Att', 'Yds_20', 'Yds_10', 'VBD'])

# Creating a column that has the average of the previous 2 years of data exluding the current year
#df_wr['TD_rec_2yr_avg'] = df_wr.groupby('Player_x')['TD_rec'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
#df_wr['TD_share_2yr_avg'] = df_wr.groupby('Player_x')['TD_share'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
#df_wr['Tgt_share_2yr_avg'] = df_wr.groupby('Player_x')['Tgt_share'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
#df_wr['Team_Yds_2yr_avg'] = df_wr.groupby('Player_x')['Team_Yds'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
#df_wr['Pass_QB_def_norm_sum_2yr_avg'] = df_wr.groupby('Player_x')['Pass_QB_def_norm_sum'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
#df_wr['Team_OnTgt_norm_2yr_avg'] = df_wr.groupby('Player_x')['Team_OnTgt_norm'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
#df_wr['Yds_rec/Att_2yr_avg'] = df_wr.groupby('Player_x')['Yds_rec/Att'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
#df_wr['PPR_2yr_avg'] = df_wr.groupby('Player_x')['PPR'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
#df_wr['Yds_20_2yr_avg'] = df_wr.groupby('Player_x')['Yds_20'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)
#df_wr['Yds_10_2yr_avg'] = df_wr.groupby('Player_x')['Yds_10'].transform(lambda x: (x.shift(1) + x.shift(2)) / 2)

# Figure out how to use 2yr ppr avg
#wr_model = df_wr[['Player_x', 'Year', 'Age', 'Pass_QB_def_norm_sum', 'TD_rec_2yr_avg', 'Tgt_share_2yr_avg', 'Team_Yds_2yr_avg', 'OnTgt_prev_norm', 'Yds_rec/Att_2yr_avg', 'PPR_2yr_avg', 'Yds_20_2yr_avg', 'Yds_10_2yr_avg', 'PPR']]

result, r2, mse, wr_coef = ff.evaluate_model(wr_model, 2022, [2021, 2020, 2019], 'PPR', RandomForestRegressor())

print(r2, mse)
print(result)
d = dtale.show(result)
d.open_browser()
tst


