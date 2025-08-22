# Description: Value fantasy football players based on previous 3 seasons' performance (heavily weighted recent years).

import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import ff_functions as ff
import warnings
warnings.filterwarnings('ignore')

# Set base directory for all data files
base_dir = r'C:\Users\matthew.jones\OneDrive - WellSky\Documents\Python Projects\FF\Data Files'
result_dir = r'C:\Users\matthew.jones\OneDrive - WellSky\Documents\Python Projects\FF\2025 Predictions'

print("Loading and validating data...")
# Load data with validation
required_files = ['rb.csv', 'qb.csv', 'rz_rec.csv', 'adv_rec.csv', 'rz_rush.csv', 'adv_rush.csv']
for file in required_files:
    if not os.path.exists(os.path.join(base_dir, file)):
        raise FileNotFoundError(f"Required file {file} not found in {base_dir}")

df_rb = pd.read_csv(os.path.join(base_dir, 'rb.csv'))
df_qb = pd.read_csv(os.path.join(base_dir, 'qb.csv'))
df_rz_rec = pd.read_csv(os.path.join(base_dir, 'rz_rec.csv'))
df_adv_rec = pd.read_csv(os.path.join(base_dir, 'adv_rec.csv'))
df_rz_rush = pd.read_csv(os.path.join(base_dir, 'rz_rush.csv'))
df_adv_rush = pd.read_csv(os.path.join(base_dir, 'adv_rush.csv'))

print(f"Loaded rb data: {df_rb.shape}")
print(f"Year range: {df_rb['Year'].min()} - {df_rb['Year'].max()}")

# Validate required columns exist
required_rb_cols = ['Player', 'Year', 'Tm', 'PPR']
missing_cols = [col for col in required_rb_cols if col not in df_rb.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in rb data: {missing_cols}")

# Function to safely merge datasets and handle duplicate columns
def safe_merge_with_suffix(left_df, right_df, merge_keys, suffix_map):
    """
    Safely merge datasets, handling duplicate columns systematically
    suffix_map: dict mapping dataset name to suffix (e.g., {'rec': '_receiving', 'rush': '_rushing'})
    """
    # Get columns that would conflict (excluding merge keys)
    left_cols = set(left_df.columns) - set(merge_keys)
    right_cols = set(right_df.columns) - set(merge_keys)
    conflicting_cols = left_cols.intersection(right_cols)
    
    if conflicting_cols:
        print(f"Found conflicting columns: {conflicting_cols}")
        # Add suffixes to right dataframe conflicting columns
        right_df_renamed = right_df.copy()
        for col in conflicting_cols:
            if col in right_df_renamed.columns:
                new_name = f"{col}{suffix_map['right']}"
                right_df_renamed = right_df_renamed.rename(columns={col: new_name})
                print(f"  Renamed {col} -> {new_name}")
        
        # Merge with renamed columns
        result = left_df.merge(right_df_renamed, on=merge_keys, how='left')
    else:
        # No conflicts, merge normally
        result = left_df.merge(right_df, on=merge_keys, how='left')
    
    return result

# Merge dataframes with systematic duplicate handling
print("Merging datasets with duplicate column resolution...")

# Red zone receiving data
rz_rec_merge_cols = ['Player', 'Year']
available_rz_rec_cols = [col for col in ['Yds_20', 'Yds_10'] if col in df_rz_rec.columns]
if available_rz_rec_cols:
    df_rb = safe_merge_with_suffix(
        df_rb, 
        df_rz_rec[rz_rec_merge_cols + available_rz_rec_cols], 
        rz_rec_merge_cols,
        {'right': '_rz_rec'}
    )
    print(f"Merged red zone receiving data: {available_rz_rec_cols}")

# Red zone rushing data
rz_rush_merge_cols = ['Player', 'Year']
available_rz_rush_cols = [col for col in ['Att_20', 'Att_10', 'TD_20', 'TD_10'] if col in df_rz_rush.columns]
if available_rz_rush_cols:
    df_rb = safe_merge_with_suffix(
        df_rb,
        df_rz_rush[rz_rush_merge_cols + available_rz_rush_cols],
        rz_rush_merge_cols,
        {'right': '_rz_rush'}
    )
    print(f"Merged red zone rushing data: {available_rz_rush_cols}")

# Calculate target share safely
if 'Tgt' in df_rb.columns and 'Team_Tgt' in df_rb.columns:
    df_rb['Tgt_share'] = df_rb['Tgt'] / df_rb['Team_Tgt'].replace(0, np.nan)
    df_rb['Tgt_share'] = df_rb['Tgt_share'].fillna(0)

# Calculate rush share safely
if 'Att' in df_rb.columns and 'Team_Att_rush' in df_rb.columns:
    df_rb['Rush_share'] = df_rb['Att'] / df_rb['Team_Att_rush'].replace(0, np.nan)
    df_rb['Rush_share'] = df_rb['Rush_share'].fillna(0)

# QB data: Only keep QB with highest PPR per team/year
if 'PPR' in df_qb.columns:
    qb_cols = ['Tm', 'Year', 'PPR']
    df_qb_clean = df_qb[qb_cols].copy()
    df_qb_clean = df_qb_clean.loc[df_qb_clean.groupby(['Tm', 'Year'])['PPR'].idxmax()]
    df_qb_clean = df_qb_clean.rename(columns={'PPR': 'qb_PPR'})
    df_rb = safe_merge_with_suffix(
        df_rb,
        df_qb_clean,
        ['Tm', 'Year'],
        {'right': '_qb'}
    )
    print("Merged QB data")

# Merge advanced receiving stats with systematic duplicate handling
if not df_adv_rec.empty:
    print("Merging advanced receiving data...")
    # Check what columns exist
    print(f"Advanced receiving columns: {list(df_adv_rec.columns)}")
    
    df_rb = safe_merge_with_suffix(
        df_rb,
        df_adv_rec,
        ['Player', 'Year'],
        {'right': '_adv_rec'}
    )
    print("Merged advanced receiving data")

# Merge advanced rushing stats with systematic duplicate handling
if not df_adv_rush.empty:
    print("Merging advanced rushing data...")
    # Check what columns exist
    print(f"Advanced rushing columns: {list(df_adv_rush.columns)}")
    
    df_rb = safe_merge_with_suffix(
        df_rb,
        df_adv_rush,
        ['Player', 'Year'],
        {'right': '_adv_rush'}
    )
    print("Merged advanced rushing data")

# Check for any remaining duplicate columns and resolve them
print("Checking for any remaining duplicate columns...")
duplicate_cols = []
unique_cols = []
col_counts = {}

for col in df_rb.columns:
    base_name = col.split('_x')[0].split('_y')[0]  # Remove _x, _y suffixes
    if base_name in col_counts:
        col_counts[base_name] += 1
        duplicate_cols.append(col)
    else:
        col_counts[base_name] = 1
        unique_cols.append(col)

if duplicate_cols:
    print(f"Found remaining duplicate-like columns: {duplicate_cols}")
    
    # Strategy: Keep the most relevant version of each duplicate
    columns_to_drop = []
    for col in duplicate_cols:
        if col.endswith('_x'):
            # Usually _x is the original, keep it and rename
            new_name = col.replace('_x', '_original')
            df_rb = df_rb.rename(columns={col: new_name})
            print(f"Renamed {col} -> {new_name}")
        elif col.endswith('_y'):
            # _y is usually the merged version, decide based on context
            base_name = col.replace('_y', '')
            if f"{base_name}_x" in df_rb.columns:
                # If we have both _x and _y, need to decide which to keep
                # For now, keep _y (merged data) and drop _x
                if f"{base_name}_x" in df_rb.columns:
                    columns_to_drop.append(f"{base_name}_x")
                # Rename _y to clean name
                df_rb = df_rb.rename(columns={col: base_name})
                print(f"Renamed {col} -> {base_name}, will drop {base_name}_x")
    
    # Drop the columns we decided to remove
    if columns_to_drop:
        df_rb = df_rb.drop(columns=columns_to_drop)
        print(f"Dropped duplicate columns: {columns_to_drop}")

# Final column check
print(f"Final dataset shape: {df_rb.shape}")
print("Sample of final columns:")
for i, col in enumerate(df_rb.columns):
    if i < 20:  # Show first 20 columns
        print(f"  {col}")
    elif i == 20:
        print(f"  ... and {len(df_rb.columns) - 20} more columns")
        break

# Calculate years played (excluding current year)
df_rb = df_rb.sort_values(['Player', 'Year']).reset_index(drop=True)
df_rb['Career_Games'] = df_rb.groupby('Player').cumcount() + 1
df_rb['Years_Experience'] = df_rb.groupby('Player')['Year'].transform(lambda x: x - x.min())

# Drop unnecessary columns - updated list to avoid dropping renamed columns
drop_cols = [
    'FantPos', 'Team', '2PP', 'TD_pass', 'Yds_pass', 'Cmp', '2PM', 'Rec', 'FantPt', 'Fmb_prev',
    'Att_pass', 'Team_Yds_rush', 'Team_Att_rush', 'Int_2yr_prev', 'Fmb', 'Rk_prev', 
    'Pos_Rank', 'PosRank_prev', 'Rk', 'DKPt', 'FDPt'  # Added DKPt to prevent data leakage
]
cols_to_drop = [col for col in drop_cols if col in df_rb.columns]
if cols_to_drop:
    df_rb.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped unnecessary columns: {len(cols_to_drop)}")

# Create shifted features for previous and 2 years ago
print("Creating lagged features...")
id_cols = ['Player', 'Year', 'Tm', 'Pos'] + [col for col in df_rb.columns if any(x in col.lower() for x in ['top10', 'rank'])]
shift_cols = [col for col in df_rb.columns if col not in id_cols and col != 'PPR']

# Memory efficient feature creation
shifted_features = {}
for col in shift_cols:
    if col in df_rb.columns:
        shifted_features[f'{col}_prev'] = df_rb.groupby('Player')[col].shift(1)
        shifted_features[f'{col}_2yr_prev'] = df_rb.groupby('Player')[col].shift(2)

# Add shifted features to dataframe
df_rb = pd.concat([df_rb, pd.DataFrame(shifted_features, index=df_rb.index)], axis=1)
print(f"Created {len(shifted_features)} lagged features")

# Create efficiency ratios instead of raw volume stats (Fix #2)
print("Creating efficiency ratios to avoid volume double-counting...")
efficiency_features = {}

# Drop efficiency (instead of raw drops)
if 'Drop' in df_rb.columns and 'Tgt' in df_rb.columns:
    efficiency_features['Drop_Rate'] = df_rb['Drop'] / df_rb['Tgt'].replace(0, np.nan)

# Reception efficiency  
if 'Rec' in df_rb.columns and 'Tgt' in df_rb.columns:
    efficiency_features['Catch_Rate'] = df_rb['Rec'] / df_rb['Tgt'].replace(0, np.nan)

# Receiving yards per target efficiency
if 'Yds_rec' in df_rb.columns and 'Tgt' in df_rb.columns:
    efficiency_features['RecYds_per_Tgt'] = df_rb['Yds_rec'] / df_rb['Tgt'].replace(0, np.nan)

# Receiving TD efficiency  
if 'TD_rec' in df_rb.columns and 'Tgt' in df_rb.columns:
    efficiency_features['RecTD_per_Tgt'] = df_rb['TD_rec'] / df_rb['Tgt'].replace(0, np.nan)

# Rushing efficiency
if 'Yds_rush' in df_rb.columns and 'Att' in df_rb.columns:
    efficiency_features['RushYds_per_Att'] = df_rb['Yds_rush'] / df_rb['Att'].replace(0, np.nan)

if 'TD_rush' in df_rb.columns and 'Att' in df_rb.columns:
    efficiency_features['RushTD_per_Att'] = df_rb['TD_rush'] / df_rb['Att'].replace(0, np.nan)

# Air yards efficiency - handle potential duplicate columns
air_col = None
if 'Air' in df_rb.columns:
    air_col = 'Air'
elif 'Air_adv_rec' in df_rb.columns:
    air_col = 'Air_adv_rec'

if air_col and 'Tgt' in df_rb.columns:
    efficiency_features['Air_per_Tgt'] = df_rb[air_col] / df_rb['Tgt'].replace(0, np.nan)

# YAC efficiency - handle potential duplicate columns
yac_col = None
if 'YAC' in df_rb.columns:
    yac_col = 'YAC'
elif 'YAC_adv_rec' in df_rb.columns:
    yac_col = 'YAC_adv_rec'

if yac_col and 'Rec' in df_rb.columns:
    efficiency_features['YAC_per_Rec'] = df_rb[yac_col] / df_rb['Rec'].replace(0, np.nan)

# Rushing YAC efficiency - handle potential duplicate columns
yac_rush_col = None
if 'YAC_rush' in df_rb.columns:
    yac_rush_col = 'YAC_rush'
elif 'YAC_adv_rush' in df_rb.columns:
    yac_rush_col = 'YAC_adv_rush'

if yac_rush_col and 'Att' in df_rb.columns:
    efficiency_features['YAC_rush_per_Att'] = df_rb[yac_rush_col] / df_rb['Att'].replace(0, np.nan)

# Yards before contact efficiency - handle potential duplicate columns
ybc_col = None
if 'YBC' in df_rb.columns:
    ybc_col = 'YBC'
elif 'YBC_adv_rush' in df_rb.columns:
    ybc_col = 'YBC_adv_rush'

if ybc_col and 'Att' in df_rb.columns:
    efficiency_features['YBC_per_Att'] = df_rb[ybc_col] / df_rb['Att'].replace(0, np.nan)

# Broken tackle efficiency - handle potential duplicate columns
brktrl_col = None
if 'BrkTkl' in df_rb.columns:
    brktrl_col = 'BrkTkl'
elif 'BrkTkl_adv_rush' in df_rb.columns:
    brktrl_col = 'BrkTkl_adv_rush'
elif 'BrkTkl_adv_rec' in df_rb.columns:
    brktrl_col = 'BrkTkl_adv_rec'

if brktrl_col:
    # Safely get reception and attempt columns, handling potential missing/renamed columns
    rec_values = pd.Series(0, index=df_rb.index)  # Default to 0
    att_values = pd.Series(0, index=df_rb.index)  # Default to 0
    
    # Try to find reception column (could be Rec or renamed)
    for col in df_rb.columns:
        if col == 'Rec' or (col.startswith('Rec') and not any(x in col.lower() for x in ['rate', 'per', 'ratio'])):
            rec_values = df_rb[col].fillna(0)
            break
    
    # Try to find attempt column (should be Att)
    if 'Att' in df_rb.columns:
        att_values = df_rb['Att'].fillna(0)
    
    total_touches = rec_values + att_values
    # Only calculate efficiency if we have meaningful touch data
    if total_touches.sum() > 0:
        efficiency_features['BrkTkl_per_Touch'] = df_rb[brktrl_col] / total_touches.replace(0, np.nan)
    else:
        print(f"Warning: Could not calculate BrkTkl_per_Touch - no valid touch data found")

# Red zone efficiency
if 'TD_10' in df_rb.columns and 'Att_10' in df_rb.columns:
    efficiency_features['RZ_TD_Rate'] = df_rb['TD_10'] / df_rb['Att_10'].replace(0, np.nan)

# Add efficiency features
if efficiency_features:
    efficiency_df = pd.DataFrame(efficiency_features, index=df_rb.index).fillna(0)
    df_rb = pd.concat([df_rb, efficiency_df], axis=1)
    print(f"Created {len(efficiency_features)} efficiency features")

    # Remove raw volume stats that are now captured by efficiency ratios
    # Be careful with column names that might have suffixes
    volume_stats_to_remove = []
    for base_stat in ['Drop', 'Air', 'YAC', 'YAC_rush', 'YBC', 'BrkTkl']:
        # Remove the base stat and any suffixed versions
        for col in df_rb.columns:
            if col == base_stat or col.startswith(f"{base_stat}_adv_") or col.startswith(f"{base_stat}_"):
                if col not in ['YAC_per_Rec', 'YAC_rush_per_Att', 'YBC_per_Att', 'BrkTkl_per_Touch']:  # Keep efficiency ratios
                    volume_stats_to_remove.append(col)
    
    existing_to_remove = [col for col in volume_stats_to_remove if col in df_rb.columns]
    if existing_to_remove:
        df_rb.drop(columns=existing_to_remove, inplace=True)
        print(f"Removed volume stats now captured by efficiency ratios: {existing_to_remove}")

# Historical averages (excluding last two years) - more memory efficient
print("Creating historical averages...")
hist_cols = ['Yds_rec', 'Att', 'GS', 'PosRank', 'PPR', 'Tgt', 'Tgt_share', 'Rush_share', 'Yds_10', 'Yds_20', 'Att_10', 'Att_20', 'TD_10', 'TD_20', '1D', 'Rec/Br', 'Att/Br']
hist_cols = [col for col in hist_cols if col in df_rb.columns]

def safe_2yr_avg(x):
    if len(x) < 2:
        return x.shift(1)
    else:
        return (x.shift(1) + x.shift(2)) / min(2, len(x) - 1)

historical_features = {}
for col in hist_cols:
    historical_features[f'{col}_2yr_avg'] = df_rb.groupby('Player')[col].transform(safe_2yr_avg)

df_rb = pd.concat([df_rb, pd.DataFrame(historical_features, index=df_rb.index)], axis=1)
print(f"Created {len(historical_features)} historical average features")

# Fill missing values strategically
print("Handling missing values...")
numeric_cols = df_rb.select_dtypes(include=[np.number]).columns
categorical_cols = df_rb.select_dtypes(include=['object']).columns

# Fill numeric columns
for col in numeric_cols:
    if col != 'PPR':  # Don't fill target variable yet
        df_rb[col] = df_rb[col].fillna(df_rb[col].median())

# Fill categorical columns
for col in categorical_cols:
    df_rb[col] = df_rb[col].fillna('Unknown')

df_rb = df_rb.sort_values(['Player', 'Year']).reset_index(drop=True)

# Separate current year data for predictions
print("Preparing prediction data...")
current_year = 2024
df_rb_current = df_rb[df_rb['Year'] == current_year].copy()

# Fix #2: Don't overwrite existing 2024 PPR data
if not df_rb_current.empty:
    df_rb_current['PPR'] = df_rb_current['PPR'].fillna(0)  # Only fill NaN values
    print(f"Found {len(df_rb_current)} players with 2024 data")
else:
    print("No 2024 data found for predictions")

# Prepare training data
df_train = df_rb[df_rb['Year'] < current_year].copy()
print(f"Training data: {len(df_train)} records from {df_train['Year'].min()}-{df_train['Year'].max()}")

# Identify features programmatically (Fix #4)
print("Identifying feature types...")
all_numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

# Remove ID and target columns
exclude_cols = ['PPR', 'Year']
feature_cols = [col for col in all_numeric_cols if col not in exclude_cols]

# Programmatically identify current vs historical features
current_year_indicators = ['tgt', 'yds', 'td', 'att', 'gs', 'adot', 'air', 'yac', 'ybc', 'bltz', 'hrry', 'qbhit', 'prss', 'drop', 'int', 'rat', '1d', 'rec/br', 'att/br', 'brktrl', 'rush_share', 'tgt_share']
historical_indicators = ['_prev', '_2yr_prev', '_2yr_avg', 'career_', 'years_']

current_year_features = []
historical_features = []

for col in feature_cols:
    col_lower = col.lower()
    if any(indicator in col_lower for indicator in historical_indicators):
        historical_features.append(col)
    elif any(indicator in col_lower for indicator in current_year_indicators):
        current_year_features.append(col)
    elif col_lower in ['age', 'experience']:
        current_year_features.append(col)  # Age is current year
    else:
        # Default to historical for safety
        historical_features.append(col)

print(f"Current year features: {len(current_year_features)}")
print(f"Historical features: {len(historical_features)}")
print(f"Current year sample: {current_year_features[:5]}")
print(f"Historical sample: {historical_features[:5]}")

# Correlation filtering with validation
print("Filtering highly correlated features...")
corr_threshold = 0.99
train_numeric = df_train[feature_cols].copy()

# Check for constant columns first
constant_cols = [col for col in train_numeric.columns if train_numeric[col].nunique() <= 1]
if constant_cols:
    print(f"Removing constant columns: {constant_cols}")
    feature_cols = [col for col in feature_cols if col not in constant_cols]
    train_numeric = train_numeric.drop(columns=constant_cols)

corr_matrix = train_numeric.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
feature_cols = [col for col in feature_cols if col not in to_drop]

if to_drop:
    print(f"Removed {len(to_drop)} highly correlated features")

print(f"Final feature count: {len(feature_cols)}")

# Update feature type lists after correlation filtering
current_year_features = [col for col in current_year_features if col in feature_cols]
historical_features = [col for col in historical_features if col in feature_cols]

# Prepare prediction data with weighted historical features and rookie/sophomore adjustments
print("Calculating weighted historical features with experience-level adjustments...")
if not df_rb_current.empty:
    df_rb_current = df_rb_current.set_index('Player')
    
    # Calculate weighted historical features with adjustments for limited experience players
    weighted_hist = []
    rookie_adjustments = {}
    
    # Calculate league averages and top performer benchmarks for rookie adjustments
    veteran_data = df_rb[df_rb.groupby('Player')['Year'].transform('count') >= 3]
    league_avg_features = {}
    top_performer_features = {}
    
    for col in historical_features:
        if col in veteran_data.columns and len(veteran_data[col].dropna()) > 0:
            league_avg_features[col] = veteran_data[col].median()
            top_performer_features[col] = veteran_data[col].quantile(0.75)  # 75th percentile
    
    print(f"Calculated benchmarks from {len(veteran_data)} veteran player-seasons")
    
    for pid, group in df_rb.groupby('Player'):
        # Get last 3 seasons before current year
        last_years = group[group['Year'] < current_year].sort_values('Year').iloc[-3:]
        n_years = len(last_years)
        
        # Enhanced experience classification
        if n_years == 0:
            experience_level = 'rookie'
        elif n_years == 1:
            experience_level = 'sophomore'  
        elif n_years == 2:
            experience_level = 'young_veteran'
        else:
            experience_level = 'veteran'
        
        # Weights from newest to oldest: 85%, 10%, 5%
        if n_years == 3:
            weights = np.array([0.05, 0.10, 0.85])  # oldest -> most recent
        elif n_years == 2:
            weights = np.array([0.10, 0.90])
        elif n_years == 1:
            weights = np.array([1.0])
        else:
            weights = np.array([1.0])
        
        row = {'Player': pid, 'years_available': n_years, 'experience_level': experience_level}
        
        # Calculate features based on experience level
        for col in historical_features:
            if experience_level == 'rookie':
                # Rookies: Use optimistic projection (midpoint between league average and top performer)
                if col in league_avg_features and col in top_performer_features:
                    row[col] = (league_avg_features[col] + top_performer_features[col]) / 2
                else:
                    row[col] = df_train[col].median() if col in df_train.columns else 0
                    
            elif experience_level == 'sophomore':
                # Sophomores: Blend their rookie year (50%) with optimistic projection (50%)
                if col in last_years.columns and len(last_years) > 0:
                    rookie_performance = last_years[col].iloc[-1] if not pd.isna(last_years[col].iloc[-1]) else 0
                    if col in league_avg_features and col in top_performer_features:
                        optimistic_proj = (league_avg_features[col] + top_performer_features[col]) / 2
                        row[col] = 0.5 * rookie_performance + 0.5 * optimistic_proj
                    else:
                        row[col] = rookie_performance
                else:
                    # Fallback to rookie treatment
                    if col in league_avg_features and col in top_performer_features:
                        row[col] = (league_avg_features[col] + top_performer_features[col]) / 2
                    else:
                        row[col] = df_train[col].median() if col in df_train.columns else 0
                        
            else:
                # Veterans: Use traditional weighted average
                if col in last_years.columns:
                    values = last_years[col].fillna(0).values
                    if len(values) > 0 and not np.all(values == 0):
                        row[col] = np.average(values, weights=weights[:len(values)])
                    else:
                        row[col] = df_train[col].median() if col in df_train.columns else 0
                else:
                    row[col] = 0
        
        weighted_hist.append(row)
    
    weighted_hist_df = pd.DataFrame(weighted_hist).set_index('Player')
    
    # Merge weighted historical features, keeping current year features as-is
    df_rb_current = df_rb_current.drop(columns=[c for c in historical_features if c in df_rb_current.columns], errors='ignore')
    df_rb_current = df_rb_current.merge(weighted_hist_df, left_index=True, right_index=True, how='left')
    
    # Ensure all required features are present
    missing_features = [col for col in feature_cols if col not in df_rb_current.columns]
    for col in missing_features:
        df_rb_current[col] = df_train[col].median()
    
    print(f"Experience level distribution:")
    print(f"  Rookies: {len(weighted_hist_df[weighted_hist_df['experience_level'] == 'rookie'])}")
    print(f"  Sophomores: {len(weighted_hist_df[weighted_hist_df['experience_level'] == 'sophomore'])}")
    print(f"  Young Veterans: {len(weighted_hist_df[weighted_hist_df['experience_level'] == 'young_veteran'])}")
    print(f"  Veterans: {len(weighted_hist_df[weighted_hist_df['experience_level'] == 'veteran'])}")
    
    # Show some example rookie adjustments
    rookie_examples = weighted_hist_df[weighted_hist_df['experience_level'] == 'rookie'].head(3)
    if len(rookie_examples) > 0:
        print(f"\nRookie projection examples (using 75th percentile veteran benchmarks):")
        for idx, row in rookie_examples.iterrows():
            sample_features = [col for col in ['Att_prev', 'Yds_rush_prev', 'PPR_prev'] if col in row.index][:3]
            feature_vals = [f"{col.replace('_prev', '')}: {row[col]:.1f}" for col in sample_features]
            print(f"  {idx}: {', '.join(feature_vals)}")

# Cross-validation setup (Fix #5)
print("Setting up model validation...")
def validate_model_with_cv(df_train, feature_cols, target_col='PPR'):
    """Perform time series cross-validation"""
    X = df_train[feature_cols].copy()
    y = df_train[target_col].copy()
    
    # Remove any remaining NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    # Time series split by year
    years = df_train[mask]['Year'].values
    unique_years = sorted(df_train['Year'].unique())
    
    print(f"Available years for CV: {unique_years}")
    
    cv_scores = []
    for i in range(len(unique_years) - 2):  # Need at least 2 years for training
        train_years = unique_years[:i+2]
        test_year = unique_years[i+2]
        
        train_mask = np.isin(years, train_years)
        test_mask = years == test_year
        
        train_count = np.sum(train_mask)
        test_count = np.sum(test_mask)
        
        print(f"CV fold: train on {train_years} ({train_count} samples), test on {test_year} ({test_count} samples)")
        
        if train_count < 50 or test_count < 10:  # Minimum sample sizes
            print(f"Skipping fold due to insufficient samples")
            continue
            
        X_train_cv = X[train_mask]
        X_test_cv = X[test_mask]
        y_train_cv = y[train_mask]
        y_test_cv = y[test_mask]
        
        # Create temporary dataframe for ff.evaluate_model
        temp_df_train = pd.DataFrame(X_train_cv)
        temp_df_train['Year'] = [train_years[-1]] * len(temp_df_train)  # Use last training year
        temp_df_train['PPR'] = y_train_cv
        temp_df_train['Player'] = range(len(temp_df_train))  # Dummy player IDs
        
        temp_df_test = pd.DataFrame(X_test_cv) 
        temp_df_test['Year'] = [test_year] * len(temp_df_test)
        temp_df_test['PPR'] = y_test_cv
        temp_df_test['Player'] = range(len(temp_df_train), len(temp_df_train) + len(temp_df_test))
        
        temp_df = pd.concat([temp_df_train, temp_df_test], ignore_index=True)
        
        try:
            _, r2, mae, _, _, model = ff.evaluate_model(temp_df, test_year, 'PPR', 'SVR')
            cv_scores.append({'test_year': test_year, 'r2': r2, 'mae': mae})
            print(f"CV result for {test_year}: R² = {r2:.3f}, MAE = {mae:.3f}")
        except Exception as e:
            print(f"CV failed for test year {test_year}: {e}")
    
    return cv_scores

# Run cross-validation
cv_results = validate_model_with_cv(df_train, feature_cols)
if cv_results:
    avg_r2 = np.mean([result['r2'] for result in cv_results])
    avg_mae = np.mean([result['mae'] for result in cv_results])
    print(f"Cross-validation results: R² = {avg_r2:.3f}, MAE = {avg_mae:.3f}")
else:
    print("Cross-validation could not be performed - insufficient data")
    avg_r2 = None
    avg_mae = None

# Train final model
print("Training final model...")
df_rb_filtered = df_train[['Player', 'Year', 'PPR'] + feature_cols].copy()

# Check if we have data for the current year in our dataset
years_available = sorted(df_rb_filtered['Year'].unique())
print(f"Years available in dataset: {years_available}")

# If no current year data, use the most recent year for evaluation
if current_year not in years_available:
    eval_year = max(years_available)
    print(f"No {current_year} data found. Using {eval_year} as evaluation year.")
    
    # Train on all data except the most recent year
    train_data = df_rb_filtered[df_rb_filtered['Year'] != eval_year].copy()
    test_data = df_rb_filtered[df_rb_filtered['Year'] == eval_year].copy()
    
    print(f"Training on {len(train_data)} records, testing on {len(test_data)} records from {eval_year}")
    
    if len(test_data) == 0:
        raise ValueError(f"No test data available for year {eval_year}")
    
    s_result, s_r2, s_mae, s_rb_coef, s_best, test = ff.evaluate_model(df_rb_filtered, eval_year, 'PPR', 'SVR')
else:
    # Use current year if it exists in data
    s_result, s_r2, s_mae, s_rb_coef, s_best, test = ff.evaluate_model(df_rb_filtered, current_year, 'PPR', 'SVR')

print(f"Final model performance: R² = {s_r2:.3f}, MAE = {s_mae:.3f}")

# Make predictions
if not df_rb_current.empty:
    print("Making predictions...")
    # Get trained features and ensure alignment
    trained_features = test.best_estimator_.named_steps['scaler'].feature_names_in_
    
    # Align prediction data to match training features exactly
    df_rb_current_aligned = df_rb_current.reindex(columns=trained_features, fill_value=0)
    df_rb_current_aligned = df_rb_current_aligned.fillna(0)
    
    # Make predictions
    predictions = test.predict(df_rb_current_aligned)
    df_rb_current['PPR_prediction_2025'] = predictions
    
    # Get feature importance
    X_train = df_train[feature_cols].copy()
    y_train = df_train['PPR'].copy()
    X_train_aligned = X_train.reindex(columns=trained_features, fill_value=0)
    
    perm = permutation_importance(test.best_estimator_, X_train_aligned, y_train, n_repeats=3, random_state=42, n_jobs=-1)
    coef_df = pd.DataFrame({'Feature': trained_features, 'Importance': perm.importances_mean, 'Std': perm.importances_std})
    coef_df = coef_df.sort_values('Importance', ascending=False)
    coef_df = coef_df[coef_df['Importance'] > 0]
    
    # Get top features for output
    top_features = coef_df.head(15)['Feature'].tolist()  # Increased to 15 for better insights
    available_top_features = [f for f in top_features if f in df_rb_current_aligned.columns]
    
    # Prepare output - handle duplicate players first
    print("Preparing output data...")
    
    # Reset index to avoid duplicate index issues
    df_rb_current_reset = df_rb_current.reset_index()
    df_rb_current_aligned_reset = df_rb_current_aligned.reset_index()
    
    # Remove duplicates early (keep first occurrence)
    df_rb_current_unique = df_rb_current_reset.drop_duplicates(subset=['Player'], keep='first')
    df_rb_current_aligned_unique = df_rb_current_aligned_reset.drop_duplicates(subset=['Player'], keep='first')
    
    # Set Player as index again for alignment
    df_rb_current_unique = df_rb_current_unique.set_index('Player')
    df_rb_current_aligned_unique = df_rb_current_aligned_unique.set_index('Player')
    
    # Sort by predictions
    df_rb_current_unique = df_rb_current_unique.sort_values('PPR_prediction_2025', ascending=False)
    
    # Create comprehensive output using unique data
    df_output = df_rb_current_unique[['PPR_prediction_2025']].copy()
    
    # Add metadata
    if 'experience_level' in df_rb_current_unique.columns:
        df_output['experience_level'] = df_rb_current_unique['experience_level']
    if 'years_available' in df_rb_current_unique.columns:
        df_output['years_available'] = df_rb_current_unique['years_available']
    
    # Add top features safely
    for feature in available_top_features[:10]:
        if feature in df_rb_current_aligned_unique.columns:
            # Align indices properly
            feature_values = df_rb_current_aligned_unique[feature]
            # Only add values for players that exist in both dataframes
            common_players = df_output.index.intersection(feature_values.index)
            df_output.loc[common_players, feature] = feature_values.loc[common_players]
    
    # Final reset and cleanup
    df_output = df_output.reset_index()
    
    print(f"Output prepared for {len(df_output)} unique players")
    
    # Enhanced metrics with CV results
    metrics = {
        'Final_R2': [s_r2], 
        'Final_MAE': [s_mae], 
        'CV_Avg_R2': [avg_r2 if avg_r2 is not None else 'N/A'],
        'CV_Avg_MAE': [avg_mae if avg_mae is not None else 'N/A'],
        'Best_Params': [str(s_best)],
        'Features_Used': [len(trained_features)],
        'Training_Records': [len(df_train)]
    }
    metrics_df = pd.DataFrame(metrics)
    
    insight = f"Top predictive features: {', '.join(available_top_features[:5])}. Model trained on {len(trained_features)} features from {len(df_train)} records."
    metrics_df['Model_Insight'] = insight
    
    # Data quality summary
    quality_summary = {
        'Rookies': [len(df_output[df_output['experience_level'] == 'rookie']) if 'experience_level' in df_output.columns else 0],
        'Sophomores': [len(df_output[df_output['experience_level'] == 'sophomore']) if 'experience_level' in df_output.columns else 0], 
        'Veterans': [len(df_output[df_output['experience_level'] == 'veteran']) if 'experience_level' in df_output.columns else 0],
        'Total_Predictions': [len(df_output)]
    }
    quality_df = pd.DataFrame(quality_summary)

    # Save results
    print("Saving results...")
    with pd.ExcelWriter(os.path.join(result_dir, 'rb_model_report.xlsx')) as writer:
        metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)
        quality_df.to_excel(writer, sheet_name='Data_Quality', index=False)
        coef_df.head(25).to_excel(writer, sheet_name='Feature_Importance', index=False)
        df_output.to_excel(writer, sheet_name='Predictions', index=False)
        
        # CV results if available
        if cv_results:
            pd.DataFrame(cv_results).to_excel(writer, sheet_name='Cross_Validation', index=False)
    
    print(f"Predictions completed for {len(df_output)} players")
    print(f"Results saved to: {os.path.join(result_dir, 'rb_model_report.xlsx')}")
    
    # Enhanced analysis
    if 'experience_level' in df_output.columns:
        print("\nPrediction Analysis by Experience Level:")
        for exp_level in ['rookie', 'sophomore', 'young_veteran', 'veteran']:
            subset = df_output[df_output['experience_level'] == exp_level]
            if len(subset) > 0:
                avg_pred = subset['PPR_prediction_2025'].mean()
                min_pred = subset['PPR_prediction_2025'].min()
                max_pred = subset['PPR_prediction_2025'].max()
                print(f"  {exp_level.replace('_', ' ').title()}: {len(subset)} players, Avg: {avg_pred:.1f}, Range: {min_pred:.1f}-{max_pred:.1f}")
    
    # Show top predictions by experience level
    print(f"\nTop Predictions by Experience Level:")
    if 'experience_level' in df_output.columns:
        for exp_level in ['rookie', 'sophomore', 'young_veteran', 'veteran']:
            subset = df_output[df_output['experience_level'] == exp_level].head(3)
            if len(subset) > 0:
                print(f"\n  {exp_level.replace('_', ' ').title()}:")
                for i, (idx, row) in enumerate(subset.iterrows(), 1):
                    print(f"    {i}. {row['Player']:20s} - {row['PPR_prediction_2025']:6.1f} PPR")
    
    # Overall top performers
    print(f"\nOverall Top 10 Predicted Performers for 2025:")
    for i, (idx, row) in enumerate(df_output.head(10).iterrows(), 1):
        exp_level = row.get('experience_level', 'Unknown').replace('_', ' ').title()
        print(f"  {i:2d}. {row['Player']:20s} - {row['PPR_prediction_2025']:6.1f} PPR ({exp_level})")

else:
    print("No current year data available for predictions")

print("RB model training and prediction complete!")