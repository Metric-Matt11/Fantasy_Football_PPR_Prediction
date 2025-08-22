# Fantasy_Football_PPR_Prediction
This repo is used to store scripts and files used to predict seasonal value for WR, TE and RB players. 

## Upadate Steps

### 1. Auto Updating Player Stats
You need to update the stats for past player perfomance. I typically use 3 years of data (Ex: For 2025 prediction, I pull 2022-2023 data). Theses are the qb.csv, te.csv, rb.csv and wr.csv. The ff.player_scrape function that I created works best for this (Use the top lines in Fantasy Football Player Valuation.py as ex). 

This is also a good time to update rz stats. Use the ff.redzone_scrape function and it produces these files: df_rz_pass, df_rz_rush, df_rz_rec.

### 2. Manual Updating 
This should be fixed in the future but for now schedule and advanced pass, rush and rec stats need to be manually updated. 

For advanced stats, the best way is to go to https://www.pro-football-reference.com/years/2024/ and go to the player stats tab. The tables can then be exported to excel and combined. 

For schedule, espn.com works best.

**IMPORTANT NOTE** 
All data files should match the previous format. It is easiest to have the prior years file open to ensure proper formatting and columns. We also want to fill all blanks with 0s. The best way to do this is to use the =if(b2="", 0, b2) formula to the right of the table and copy and paste the values when done. 

### 3. Updating the Models 
The only thing that needs updated is the current year (which is the latest year of data, so for 2025 latest year = 2024). 

### 4. (Optional) Download Fantasy Pro Rankings and Compare Results 
New this year, I will be comparing the Fantasy Pro Player Rankings to my results. The idea is to pick out players that might be over or under valued.

## Future Improvements
1. Right now the R2 values are super inflated. The main issue is data leakage. This needs addressed.
2. Seperating young players from veterans could use improving.
3. Adding in a age regression curve to RB data.
4. Creating a model for incoming rookies
5. Adding some sort of team adjustment. Like a player switching teams. Their useage could be compeltly different and their QB perfomance could also differ drastically. 

## Historical Results
### 2023
- Finished with most total points in both leagues 
- Finished 3rd in one league and was eliminated in round 1 of the other
- Top value pick was TJ. Hockenson
- Tyreek Hill carried the team

