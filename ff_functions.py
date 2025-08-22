# Description: This program will be a library of functions to be used in Fantasy Football Player Valuation.py
#
# Import libraries

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.svm import SVR
import xlrd
from playwright.sync_api import sync_playwright
from tqdm import tqdm
import time

def player_scrape(begin_year, end_year):
    """
    Scrapes fantasy football data from pro-football-reference.com using Playwright
    from begin_year to end_year. Returns 4 DataFrames: QB, RB, WR, TE.
    """

    column_headers = [
        'Rk', 'Player', 'Tm', 'FantPos', 'Age', 'G', 'GS',
        'Cmp', 'Att_pass', 'Yds_pass', 'TD_pass', 'Int_pass',
        'Att_rush', 'Yds_rush', 'Y/A', 'TD_rush', 'Tgt', 'Rec',
        'Yds_rec', 'Y/R', 'TD_rec', 'Fmb', 'FL', 'TD_total', '2PM',
        '2PP', 'FantPt', 'PPR', 'DKPt', 'FDPt', 'VBD', 'PosRank', 'OvRank'
    ]

    df_all = pd.DataFrame()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for year in range(begin_year, end_year + 1):
            url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
            print(f"Scraping {url} ...")

            try:
                page.goto(url, timeout=60000)
                time.sleep(2)  # allow page to fully load
                soup = BeautifulSoup(page.content(), 'html.parser')
                table = soup.find('table', {'id': 'fantasy'})

                if table is None:
                    print(f"⚠️ Table not found for year {year}")
                    continue

                table_body = table.find('tbody')
                rows = table_body.find_all('tr')
                data_rows = [[td.get_text(strip=True) for td in row.find_all('td')] for row in rows if row.find_all('td')]

                df = pd.DataFrame(data_rows, columns=column_headers[1:])
                df = df.replace('', 0)
                df = df.apply(pd.to_numeric, errors='ignore')
                df['Year'] = year
                df_all = pd.concat([df_all, df], ignore_index=True)

            except Exception as e:
                print(f"❌ Error scraping year {year}: {e}")
                continue

        browser.close()

    if df_all.empty:
        raise ValueError("No data was scraped. Check network connection or website structure.")

    # Clean team names
    team_replacements = {
        'GNB': 'GB', 'KAN': 'KC', 'NOR': 'NO', 'SFO': 'SF',
        'TAM': 'TB', 'LVR': 'LV', 'NWE': 'NE'
    }
    df_all['Tm'] = df_all['Tm'].replace(team_replacements)

    # Compute team aggregates
    df_all['Team_Yds'] = df_all.groupby(['Tm', 'Year'])['Yds_pass'].transform('sum') + \
                         df_all.groupby(['Tm', 'Year'])['Yds_rush'].transform('sum')
    df_all['Team_Tgt'] = df_all.groupby(['Tm', 'Year'])['Tgt'].transform('sum')
    df_all['Team_Att_rush'] = df_all.groupby(['Tm', 'Year'])['Att_rush'].transform('sum')
    df_all['Team_Yds_rush'] = df_all.groupby(['Tm', 'Year'])['Yds_rush'].transform('sum')

    # Split by position
    df_qb = df_all[df_all['FantPos'] == 'QB'].reset_index(drop=True)
    df_rb = df_all[df_all['FantPos'] == 'RB'].reset_index(drop=True)
    df_wr = df_all[df_all['FantPos'] == 'WR'].reset_index(drop=True)
    df_te = df_all[df_all['FantPos'] == 'TE'].reset_index(drop=True)

    # Clean player names
    for df in [df_qb, df_rb, df_wr, df_te]:
        df['Player'] = df['Player'].str.replace('*', '', regex=False).str.replace('+', '', regex=False)

    # Export to CSV
    df_qb.to_csv('qb.csv', index=False)
    df_rb.to_csv('rb.csv', index=False)
    df_wr.to_csv('wr.csv', index=False)
    df_te.to_csv('te.csv', index=False)

    return df_qb, df_rb, df_wr, df_te

def team_def_scrape(begin_year, end_year):
    """
    This function will scrape data from pro-football-reference.com for team defense data, loop through the begin_year to end_year seasons
    and returns a dataframe with the data

    Parameters:
    ----------
    begin_year : int
        The first year to scrape data from
    end_year : int
        The last year to scrape data from
    
    Returns:
    -------
    dataframe
        A dataframe with the data scraped from pro-football-reference.com
    """
    for x in range(begin_year, end_year + 1):
        url = 'https://www.pro-football-reference.com/years/' + str(x) + '/opp.htm'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        table = soup.find('table')
        table_head = table.find('thead')
        table_body = table.find('tbody')
        table_head_rows = table_head.find_all('tr')
        table_body_rows = table_body.find_all('tr')
        column_headers = ['Rk', 'Tm', 'G', 'PF', 'Yds', 'Ply', 'Y/P', 'TO', 'FL', '1stD', 'Cmp', 'Att', 'Yds_pass', 'TD_pass', 'Int_pass', 'NY/A', '1stD_pass', 'Att_rush', 'Yds_rush', 'TD_rush', 'Y/A', '1stD_rush', 'Pen', 'Yds_pen', '1stPy', 'Sc%', 'TO%', 'EXP']
        data_rows = [[td.getText() for td in table_body_rows[i].find_all('td')] for i in range(len(table_body_rows))]
        df = pd.DataFrame(data_rows, columns=column_headers[1:])
        df = df.dropna()
        df = df.replace('', 0)
        df = df.apply(pd.to_numeric, errors='ignore')
        df['Year'] = x
        if x == begin_year:
            df_def = df
        else:
            df_def = df_def.append(df)
    df_def['Tm'] = df_def['Tm'].replace('Los Angeles Rams', 'LAR')
    df_def['Tm'] = df_def['Tm'].replace('Los Angeles Chargers', 'LAC')
    df_def['Tm'] = df_def['Tm'].replace('New England Patriots', 'NE')
    df_def['Tm'] = df_def['Tm'].replace('New Orleans Saints', 'NO')
    df_def['Tm'] = df_def['Tm'].replace('New York Giants', 'NYG')
    df_def['Tm'] = df_def['Tm'].replace('New York Jets', 'NYJ')
    df_def['Tm'] = df_def['Tm'].replace('San Francisco 49ers', 'SF')
    df_def['Tm'] = df_def['Tm'].replace('Tampa Bay Buccaneers', 'TB')
    df_def['Tm'] = df_def['Tm'].replace('Tennessee Titans', 'TEN')
    df_def['Tm'] = df_def['Tm'].replace('Washington Football Team', 'WSH')
    df_def['Tm'] = df_def['Tm'].replace('Arizona Cardinals', 'ARI')
    df_def['Tm'] = df_def['Tm'].replace('Atlanta Falcons', 'ATL')
    df_def['Tm'] = df_def['Tm'].replace('Baltimore Ravens', 'BAL')
    df_def['Tm'] = df_def['Tm'].replace('Buffalo Bills', 'BUF')
    df_def['Tm'] = df_def['Tm'].replace('Carolina Panthers', 'CAR')
    df_def['Tm'] = df_def['Tm'].replace('Chicago Bears', 'CHI')
    df_def['Tm'] = df_def['Tm'].replace('Cincinnati Bengals', 'CIN')
    df_def['Tm'] = df_def['Tm'].replace('Cleveland Browns', 'CLE')
    df_def['Tm'] = df_def['Tm'].replace('Dallas Cowboys', 'DAL')
    df_def['Tm'] = df_def['Tm'].replace('Denver Broncos', 'DEN')
    df_def['Tm'] = df_def['Tm'].replace('Detroit Lions', 'DET')
    df_def['Tm'] = df_def['Tm'].replace('Green Bay Packers', 'GB')
    df_def['Tm'] = df_def['Tm'].replace('Houston Texans', 'HOU')
    df_def['Tm'] = df_def['Tm'].replace('Indianapolis Colts', 'IND')
    df_def['Tm'] = df_def['Tm'].replace('Jacksonville Jaguars', 'JAX')
    df_def['Tm'] = df_def['Tm'].replace('Kansas City Chiefs', 'KC')
    df_def['Tm'] = df_def['Tm'].replace('Miami Dolphins', 'MIA')
    df_def['Tm'] = df_def['Tm'].replace('Minnesota Vikings', 'MIN')
    df_def['Tm'] = df_def['Tm'].replace('Oakland Raiders', 'LV')
    df_def['Tm'] = df_def['Tm'].replace('Philadelphia Eagles', 'PHI')
    df_def['Tm'] = df_def['Tm'].replace('Pittsburgh Steelers', 'PIT')
    df_def['Tm'] = df_def['Tm'].replace('Seattle Seahawks', 'SEA')
    df_def['Tm'] = df_def['Tm'].replace('Washington Commanders', 'WSH')
    df_def['Tm'] = df_def['Tm'].replace('Las Vegas Raiders', 'LV')
    df_def['Tm'] = df_def['Tm'].replace('San Diego Chargers', 'LAC')
    df_def['Tm'] = df_def['Tm'].replace('St. Louis Rams', 'LAR')
    df_def['Tm'] = df_def['Tm'].replace('Washington Redskins', 'WSH')

    df_def = df_def[['Tm','EXP', 'TO', 'NY/A', 'Y/A', 'Year']]

    # Normalize the data
    df_def['Pass_QB_def_norm'] = df_def.groupby('Year')['NY/A'].transform(lambda x: (x - x.mean()) / x.std())
    df_def['Rush_def_norm'] = df_def.groupby('Year')['Y/A'].transform(lambda x: (x - x.mean()) / x.std())

    # Write the dataframe to a csv file
    df_def.to_csv('team_def.csv', index=False)

    return df_def

def redzone_scrape(begin_year, end_year):
    """
    This function scrapes https://www.pro-football-reference.com/years/ for redzone passing, rushing, receiving and defense data 
    
    Parameters:
    ----------
    begin_year : int
        The first year to scrape data from
    end_year : int
        The last year to scrape data from
    
    Returns:
    -------
    df : DataFrame
        A dataframe with the compiled redzone data
    """
    
    # Add headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Creating a for loop to go through passing, rushing and receiving data
    for y in ['passing', 'rushing', 'receiving']:
        if y == 'passing':
            column_headers = ['Tm', 'Cmp_20', 'Att_20', 'Cmp%_20', 'Yds_20', 'TD_20', 'Int_20', 'Cmp_10', 'Att_10', 'Cmp%_10', 'Yds_10', 'TD_10', 'Int_10', 'Highlights']
        elif y == 'rushing':
            column_headers = ['Tm', 'Att_20', 'Yds_20', 'TD_20', '%Rush_20', 'Att_10', 'Yds_10', 'TD_10', '%Rush_9', 'Att_5', 'Yds_5', 'TD_5', '%Rush_5', 'Highlights']
        else:
            column_headers = ['Tm', 'Tgt_20', 'Rec_20', 'Ctch%_20', 'Yds_20', 'TD_20', '%Tgt_20', 'Tgt_10', 'Rec_10', 'Ctch%_10', 'Yds_10', 'TD_10', '%Tgt_10', 'Highlights']
        
        df_name = None  # Initialize df_name
        
        for x in range(begin_year, end_year + 1):
            url = f'https://www.pro-football-reference.com/years/{x}/redzone-{y}.htm'
            print(f"Scraping: {url}")
            
            try:
                # Add delay between requests to be respectful
                time.sleep(1)
                
                page = requests.get(url, headers=headers, timeout=10)
                page.raise_for_status()  # Raises an HTTPError for bad responses
                
                soup = BeautifulSoup(page.text, 'html.parser')
                table = soup.find('table', {'id': 'fantasy_rz'})
                
                # Check if table exists
                if table is None:
                    print(f"Warning: No table found for {y} in {x}")
                    # Try alternative table IDs
                    possible_ids = ['redzone', 'rz_stats', 'stats_table']
                    for alt_id in possible_ids:
                        table = soup.find('table', {'id': alt_id})
                        if table is not None:
                            print(f"Found table with ID: {alt_id}")
                            break
                    
                    if table is None:
                        # Try finding table by class or other attributes
                        table = soup.find('table', {'class': 'stats_table'})
                        if table is None:
                            print(f"Skipping {y} for year {x} - no table found")
                            continue
                
                table_head = table.find('thead')
                table_body = table.find('tbody')
                
                if table_body is None:
                    print(f"Warning: No table body found for {y} in {x}")
                    continue
                
                table_body_rows = table_body.find_all('tr')
                
                if not table_body_rows:
                    print(f"Warning: No data rows found for {y} in {x}")
                    continue
                
                # Extract data
                data_rows_player = [[td.getText() for td in table_body_rows[i].find_all('th')] for i in range(len(table_body_rows))]
                data_rows = [[td.getText() for td in table_body_rows[i].find_all('td')] for i in range(len(table_body_rows))]
                
                # Create dataframes
                df = pd.DataFrame(data_rows, columns=column_headers[:len(data_rows[0]) if data_rows else 0])
                df_player = pd.DataFrame(data_rows_player, columns=['Player'])
                df['Player'] = df_player['Player']
                df = df.dropna()
                df = df.replace('', 0)
                df = df.apply(pd.to_numeric, errors='ignore')
                df['Year'] = x
                
                # Standardize team names
                team_replacements = {
                    'GNB': 'GB', 'KAN': 'KC', 'NOR': 'NO', 'SFO': 'SF',
                    'TAM': 'TB', 'LVR': 'LV', 'NWE': 'NE'
                }
                df['Tm'] = df['Tm'].replace(team_replacements)
                
                # Append to main dataframe
                if df_name is None:
                    df_name = df
                else:
                    df_name = pd.concat([df_name, df], ignore_index=True)
                
                print(f"Successfully scraped {len(df)} rows for {y} in {x}")
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {url}: {e}")
                continue
            except Exception as e:
                print(f"Error processing {y} for year {x}: {e}")
                continue
        
        # Store the dataframe for each category
        if y == 'passing':
            df_rz_pass = df_name if df_name is not None else pd.DataFrame()
        elif y == 'rushing':
            df_rz_rush = df_name if df_name is not None else pd.DataFrame()
        else:
            df_rz_rec = df_name if df_name is not None else pd.DataFrame()
    
    # Write the dataframes to csv files
    try:
        if not df_rz_pass.empty:
            df_rz_pass.to_csv('rz_pass.csv', index=False)
            print("Saved rz_pass.csv")
        if not df_rz_rush.empty:
            df_rz_rush.to_csv('rz_rush.csv', index=False)
            print("Saved rz_rush.csv")
        if not df_rz_rec.empty:
            df_rz_rec.to_csv('rz_rec.csv', index=False)
            print("Saved rz_rec.csv")
    except Exception as e:
        print(f"Error saving CSV files: {e}")
    
    return df_rz_pass, df_rz_rush, df_rz_rec

def nfl_schedule(begin_year, end_year):
    """
    This function looks in the file path for xlsx files that meet the criteria and compiles them into one dataframe

    Parameters:
    ----------
    begin_year : int
        The first year to scrape data from
    end_year : int
        The last year to scrape data from

    Returns:
    -------
    df : DataFrame
        A dataframe with the compiled schedule data
    """
    for x in range(begin_year, end_year + 1):
        df = pd.read_excel('nfl_schedule_' + str(x) + '.xlsx')
        df['Year'] = x
        if x == begin_year:
            df_schedule = df
        else:
            df_schedule = df_schedule.append(df)
    return df_schedule

def qb_adv_stats(begin_year, end_year, table_id):
    """
    This function scrapes https://www.pro-football-reference.com/years/2022/passing_advanced.htm for advanced stats for QBs
    The options are advanced_air_yards, advanced_accuracy and advanced_pressure
    
    Parameters:
    ----------
    begin_year : int
        The first year to scrape data from
    end_year : int
        The last year to scrape data from

    Returns:
    -------
    df : DataFrame
        2 dataframes, one with qb advanced stats and one with team qb advanced stats
    """
    for x in range(begin_year, end_year + 1):
        url = 'https://www.pro-football-reference.com/years/' + str(x) + '/passing_advanced.htm'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        table = soup.find('table', {'id': table_id})
        table_head = table.find('thead')
        table_body = table.find('tbody')
        table_head_rows = table_head.find_all('tr')
        table_body_rows = table_body.find_all('tr')
        if table_id == 'advanced_accuracy':
            column_headers = ['Player', 'Tm', 'Age', 'Pos', 'G', 'GS', 'Cmp', 'Att', 'Yds', 'Bats', 'ThAwy', 'Spikes', 'Drops', 'Drop%', 'BadTh', 'Bad%', 'OnTgt', 'OnTgt%']
        elif table_id == 'advanced_air_yards':
            column_headers = ['Player', 'Tm', 'Age', 'Pos', 'G', 'GS', 'Cmp', 'Att', 'Yds', 'IAY', 'IAY/PA', 'CAY', 'CAY/Cmp', 'CAY/PA', 'YAC', 'YAC/Cmp']
        elif table_id == 'advanced_pressure':
            column_headers = ['Player', 'Tm', 'Age', 'Pos', 'G', 'GS', 'Cmp', 'Att', 'Yds', 'Sk', 'PktTime', 'Bltz', 'Hrry', 'Hits', 'Prss', 'Prss%', 'Scrm', 'Yds/Scr']
        data_rows = [[td.getText() for td in table_body_rows[i].find_all('td')] for i in range(len(table_body_rows))]
        df = pd.DataFrame(data_rows, columns=column_headers[0:])
        df = df.dropna()
        df = df.replace('', 0)
        df = df.apply(pd.to_numeric, errors='ignore')
        df['Year'] = x
        if x == begin_year:
            df_qb_adv = df
        else:
            df_qb_adv = df_qb_adv.append(df)
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('GNB', 'GB')
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('KAN', 'KC')
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('NOR', 'NO')
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('SFO', 'SF')
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('TAM', 'TB')
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('LVR', 'LV')
    df_qb_adv['Tm'] = df_qb_adv['Tm'].replace('NWE', 'NE')
    df_qb_adv = df_qb_adv[df_qb_adv['Pos'] == 'QB']

    # Getting rid of all special characters in the player names
    df_qb_adv['Player'] = df_qb_adv['Player'].str.replace('*', '')
    df_qb_adv['Player'] = df_qb_adv['Player'].str.replace('+', '')
    df_qb_adv['Player'] = df_qb_adv['Player'].str.replace('\\', '')
    df_qb_adv['Player'] = df_qb_adv['Player'].str.replace('\'', '')

    return df_qb_adv

def evaluate_model(df, df_testing_year, target, model_x):
    """
    Evaluates a predictive model and returns predictions, performance metrics, and model object.

    Parameters:
    ----------
    df : DataFrame
        The dataframe to use for training and testing.
    df_testing_year : int
        The year to use for testing.
    target : str    
        The target variable.
    model_x : str
        One of: "SVR", "RandomForestRegressor", "XGBRegressor"

    Returns:
    -------
    df_test : DataFrame with predictions
    r2 : float
    mae : float
    df_coef : DataFrame of feature importances (or 0 for SVR)
    best : dict of best hyperparameters
    model : trained model
    """
    print("\n🔧 Starting model evaluation...\n")
    start_time = time.time()

    # Ensure the dataframe is not modified in place
    df = df.copy()

    # Set index
    df.set_index('Player', inplace=True)

    print("📦 Splitting training and testing sets...")
    df_train = df[df['Year'] != df_testing_year].drop(columns=['Year'])
    df_test = df[df['Year'] == df_testing_year].drop(columns=['Year'])

    print("🧹 Dropping missing values...")
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    print("🔍 Selecting numeric features...")
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]

    # Ensure X_train is numeric
    X_train = X_train.select_dtypes(include=[np.number])
    
    print(f"🧠 Model selected: {model_x}")
    if model_x == "XGBRegressor":
        param_grid = [{
            'model': [XGBRegressor()],
            'model__n_estimators': [25, 50],
            'model__max_depth': [5, 10],
            'model__learning_rate': [0.1, 0.01],
        }]
    elif model_x == "SVR":
        param_grid = [{
            'model': [SVR()],
            'model__kernel': ['linear', 'rbf'],
            'model__C': [1, 10],
            'model__gamma': ['scale', 'auto']
        }]
    elif model_x == "RandomForestRegressor":
        param_grid = [{
            'model': [RandomForestRegressor()],
            'model__n_estimators': [50, 100],
            'model__max_depth': [10, 20],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2],
        }]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR() if model_x == "SVR" else RandomForestRegressor())
    ])

    print("🔄 Running GridSearchCV...")
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
    model = grid.fit(X_train, y_train)

    print("✅ Grid search complete. Best parameters:")
    print(model.best_params_)

    print("📈 Generating predictions...")
    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    # Match X_test to training features
    trained_features = model.best_estimator_.named_steps['scaler'].feature_names_in_
    X_test = X_test.reindex(columns=trained_features, fill_value=0)

    y_pred = model.predict(X_test)

    print("🧮 Calculating evaluation metrics...")
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"🎯 R-squared: {r2:.4f}")
    print(f"📉 Mean Absolute Error: {mae:.4f}")

    print("📊 Attaching predictions to test set...")
    df_test['Predicted'] = y_pred

    print("🧾 Extracting feature importances...")
    if model_x in ["XGBRegressor", "RandomForestRegressor"]:
        importances = model.best_estimator_.named_steps['model'].feature_importances_
        df_coef = pd.DataFrame({'Feature': trained_features, 'Coefficient': importances})
    else:
        df_coef = 0

    elapsed_time = time.time() - start_time
    print(f"✅ Model evaluation completed in {elapsed_time:.2f} seconds.\n")

    return df_test, r2, mae, df_coef, model.best_params_, model
    
def nfl_schedule_scrape(begin_year, end_year):
    """
    This function scraped the nfl schedule from https://thehuddle.com/2019/04/18/2019-nfl-schedule-team-by-week-grid/

    Parameters:
    ----------
    begin_year : int
        The first year to scrape data from
    end_year : int
        The last year to scrape data from

    Returns:   
    -------
    df : DataFrame
        A dataframe with the compiled schedule data
    """
    for x in range(begin_year, end_year + 1):
        url = 'https://thehuddle.com/' + str(x) + '/04/18/' + str(x) + '-nfl-schedule-team-by-week-grid/'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        table = soup.find('table', {'class':'table-responsive-inner'})
        table_head = table.find('thead')
        table_body = table.find('tbody')
        table_head_rows = table_head.find_all('tr')
        table_body_rows = table_body.find_all('tr')
        column_headers = ['TEAM', 'week_1', 'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7', 'week_8', 'week_9', 'week_10', 'week_11', 'week_12', 'week_13', 'week_14', 'week_15', 'week_16', 'week_17']
        data_rows = [[td.getText() for td in table_body_rows[i].find_all('td')] for i in range(len(table_body_rows))]
        df = pd.DataFrame(data_rows, columns=column_headers[0:])
        df = df.dropna()
        df = df.replace('', 0)
        df = df.apply(pd.to_numeric, errors='ignore')
        df['Year'] = x
        if x == begin_year:
            df_schedule = df
        else:
            df_schedule = df_schedule.append(df)

def model_evaluation(df, df_testing_year, target, model):
    """
    This function evaluates a model by calculating r2 and mse. It will also create a df with the features and their coefficients

    Parameters:
    ----------
    df : DataFrame
        The dataframe to use for the model
    df_testing_year : int
        The year to use for testing
    target : str    
        The target variable
    model : object
        The model to use for the evaluation
    """
    # Setting player as the index
    df = df.set_index('Player')

    # Setting the training data to be all years except the testing year
    df_train = df[df['Year'] != df_testing_year]
    df_test = df[df['Year'] == df_testing_year]

    # Drop the year column
    df_train = df_train.drop('Year', axis=1)
    df_test = df_test.drop('Year', axis=1)

    # Drop rows that have NaN values
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    # Fit the model to the training data
    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]

    # Create a pipeline for RandomForestRegressor
    #pipe = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor())])
    pipe = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor())])

    # Creating grid search parameters selecting only significant parameters
    param_grid = [{'model': [RandomForestRegressor()],
                    'model__n_estimators': [25, 50, 75, 100],
                    'model__max_depth': [10, 20, 30],
                    'model__min_samples_split': [2, 5, 10], 
                    'model__min_samples_leaf': [1, 2, 5, 10], 
                    'model__max_features': ['log2']
                    }]
    
    # Create grid search object, setting scoring to r2
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)

    # Fit the model
    model = grid.fit(X_train, y_train)
    
    # Make predictions on the testing data
    X_test = df_test.drop(target, axis=1)
    y_test = df_test[target]
    y_pred = model.predict(X_test)

    # Calculate r2 and mse score
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Add the predicted values to the testing data frame
    df_test['Predicted'] = y_pred

    # Create a data frame with the features and their coefficients
    df_coef = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.best_estimator_.named_steps['model'].feature_importances_})

    # Print the best parameters
    best = grid.best_params_
    
    # Return the testing data frame and evaluation metrics
    return df_test, r2, mae, df_coef, best

def draft_scrape(begin_year, end_year):
    """
    This function will scrape https://www.pro-football-reference.com/years/2023/draft.htm for draft data

    Parameters:
    ----------
    begin_year : int
        The first year to scrape data from
    end_year : int
        The last year to scrape data from

    Returns:
    -------
    df : DataFrame
        A dataframe with the compiled draft data
    """
    for x in range(begin_year, end_year + 1):
        url = 'https://www.pro-football-reference.com/years/' + str(x) + '/draft.htm'
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        table = soup.find('table', {'id': 'drafts'})
        table_head = table.find('thead')
        table_body = table.find('tbody')
        table_head_rows = table_head.find_all('tr')
        table_body_rows = table_body.find_all('tr')
        column_headers = ['Pick', 'Team', 'Player', 'Position', 'Age', 'AP1', 'PB', 'St', 'CarAV', 'DrAV', 'G', 'Cmp', 'Att', 'Yds_pass', 'TD_pass', 'Int_pass', 'Att_rush', 'Yds_rush', 'TD_rush', 'Rec', 'Yds_rec', 'TD_rec', 'Solo', 'Int_def', 'Sk', 'x', 'College/Univ', 'y']
        data_rows = [[td.getText() for td in table_body_rows[i].find_all('td')] for i in range(len(table_body_rows))]
        df = pd.DataFrame(data_rows, columns=column_headers[0:])
        df = df.dropna()
        df = df.replace('', 0)
        df = df.apply(pd.to_numeric, errors='ignore')
        df['Year'] = x
        # Drop all columns that arent Pick, Team, Player, Position, College/Univ and Year
        df = df.drop(['Age', 'AP1', 'PB', 'St', 'CarAV', 'DrAV', 'G', 'Cmp', 'Att', 'Yds_pass', 'TD_pass', 'Int_pass', 'Att_rush', 'Yds_rush', 'TD_rush', 'Rec', 'Yds_rec', 'TD_rec', 'Solo', 'Int_def', 'Sk', 'x', 'y'], axis=1)
        if x == begin_year:
            df_draft = df
        else:
            df_draft = df_draft.append(df)
        df['Team'] = df['Team'].replace('GNB', 'GB')
        df['Team'] = df['Team'].replace('KAN', 'KC')
        df['Team'] = df['Team'].replace('NOR', 'NO')
        df['Team'] = df['Team'].replace('SFO', 'SF')
        df['Team'] = df['Team'].replace('TAM', 'TB')
        df['Team'] = df['Team'].replace('LVR', 'LV')
        df['Team'] = df['Team'].replace('NWE', 'NE')
        
    # Write the dataframe to a csv file
    df_draft.to_csv('draft.csv', index=False)

def rb_adv_cleaning(begin_year, end_year):
    # Cleaning adv_rush_2018 to get rid of all special characters in the player names
    for x in range(begin_year, end_year + 1):
        file_name = 'adv_rush_' + str(x) + '.xlsx'
        df = pd.read_excel(file_name)
        df['Year'] = x
        if x == begin_year:
            df_adv = df
        else:
            df_adv = df_adv.append(df)
    df_adv['Player'] = df_adv['Player'].str.replace('*', '')
    df_adv['Player'] = df_adv['Player'].str.replace('+', '')
    df_adv['Player'] = df_adv['Player'].str.replace('\\', '')
    df_adv['Player'] = df_adv['Player'].str.replace('\'', '')
    df_adv['Tm'] = df_adv['Tm'].replace('GNB', 'GB')
    df_adv['Tm'] = df_adv['Tm'].replace('KAN', 'KC')
    df_adv['Tm'] = df_adv['Tm'].replace('NOR', 'NO')
    df_adv['Tm'] = df_adv['Tm'].replace('SFO', 'SF')
    df_adv['Tm'] = df_adv['Tm'].replace('TAM', 'TB')
    df_adv['Tm'] = df_adv['Tm'].replace('LVR', 'LV')
    df_adv['Tm'] = df_adv['Tm'].replace('NWE', 'NE')
    df_adv = df_adv.drop(['Age', 'Pos', 'G', 'GS', 'Att', 'Yds'], axis=1)

    #Write the dataframe to a csv file
    df_adv.to_csv('adv_rush.csv', index=False)

def rec_adv_cleaning(begin_year, end_year):
    # Cleaning adv_rush_2018 to get rid of all special characters in the player names
    for x in range(begin_year, end_year + 1):
        file_name = 'rec_adv_' + str(x) + '.xls'
        #Reading file with html xml parser
        df = pd.read_html(file_name)[0]
        df['Year'] = x
        if x == begin_year:
            df_adv = df
        else:
            df_adv = df_adv.append(df)
    df_adv['Player'] = df_adv['Player'].str.replace('*', '')
    df_adv['Player'] = df_adv['Player'].str.replace('+', '')
    df_adv['Player'] = df_adv['Player'].str.replace('\\', '')
    df_adv['Player'] = df_adv['Player'].str.replace('\'', '')
    df_adv['Tm'] = df_adv['Tm'].replace('GNB', 'GB')
    df_adv['Tm'] = df_adv['Tm'].replace('KAN', 'KC')
    df_adv['Tm'] = df_adv['Tm'].replace('NOR', 'NO')
    df_adv['Tm'] = df_adv['Tm'].replace('SFO', 'SF')
    df_adv['Tm'] = df_adv['Tm'].replace('TAM', 'TB')
    df_adv['Tm'] = df_adv['Tm'].replace('LVR', 'LV')
    df_adv['Tm'] = df_adv['Tm'].replace('NWE', 'NE')
    df_adv = df_adv.drop(['Age', 'Pos', 'G', 'GS', 'Tgt', 'Rec', 'Yds', 'TD'], axis=1)

    #Write the dataframe to a csv file
    df_adv.to_csv('adv_rec.csv', index=False)

#adv_acc = qb_adv_stats(2019, 2022, 'advanced_accuracy')
#adv_air = qb_adv_stats(2019, 2022, 'advanced_air_yards')
#adv_press = qb_adv_stats(2019, 2022, 'advanced_pressure')

#Dropping Age, Pos, G, GS, Cmp, Att and Yds for each dataframe then merging them together
#adv_acc = adv_acc.drop(['Age', 'Pos', 'G', 'GS', 'Cmp', 'Att', 'Yds'], axis=1)
#adv_air = adv_air.drop(['Age', 'Pos', 'G', 'GS', 'Cmp', 'Att', 'Yds'], axis=1)
#adv_press = adv_press.drop(['Age', 'Pos', 'G', 'GS', 'Cmp', 'Att', 'Yds'], axis=1)

#adv_qb = pd.merge(adv_acc, adv_air, on=['Player', 'Tm', 'Year'])
#adv_qb = pd.merge(adv_qb, adv_press, on=['Player', 'Tm', 'Year'])

#Putting adv_qb to csv
#adv_qb.to_csv('adv_qb.csv', index=False)

#Putting adv_qb to csv
