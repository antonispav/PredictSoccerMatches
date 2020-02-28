import pandas as pd
import numpy as np
import sqlite3
# This script file provides the data for MLP.py

class data:
    def __init__(self):
        self.B365_odds = None
        self.BW_odds = None
        self.IW_odds = None
        self.LB_odds = None
        self.result = None
        self.MergedTable = None
        self.labels = None

    def get_mlp_data(self):
        # Define the database file.
        database_file = 'database.sqlite'

        # Create a connection to the database.
        print('Establishing connection to \n',database_file)
        conn = sqlite3.connect(database_file)

        #extract data from Match table
        Match = pd.read_sql_query("select * from Match", conn)
        Match = Match.iloc[:,np.r_[0:11,85:97]].dropna()
        #Make a new column(year) from the match date
        Match['year'] = pd.DatetimeIndex(Match['date']).year
        # print("Match \n",Match)
        #extract data from TeamAttributes table
        TeamAttributes = pd.read_sql_query("select * from Team_Attributes", conn)
        #TeamAttributes = TeamAttributes.iloc[:,np.r_[0:6,7:25]]
        TeamAttributes = TeamAttributes.loc[:,["date","team_api_id","buildUpPlaySpeed", "buildUpPlayPassing", "chanceCreationPassing", "chanceCreationCrossing", "chanceCreationShooting", "defencePressure", "defenceAggregation", "defenceTeamWidth"]]

        #Following columns could be used for more data
        # buildUpPlaySpeedClass : Balanced , Fast , Slow
        # buildUpPlayDribblingClass : Little,Normal,Lots
        # buildUpPlayPassing : Mixed , Short , Long
        # buildUpPlayPositioningClass : Organised,Free Form
        # chanceCreationPassingClass : Normal , Risky , Safe
        # chanceCreationCrossingClass : Normal,Lots,Little
        # chanceCreationShootingClass : Normal,Lots,Little
        # chanceCreationPositioningClass : Organised , Free Form
        # defencePressureClass : Medium,Deep,High

        TeamAttributes = TeamAttributes.fillna(0)
        #TeamAttributes table has many entries for the same team but on different dates(date column)
        #We want to keep only the Year from the date column(1 entry for every year) to a new column(year)
        TeamAttributes['year'] = pd.DatetimeIndex(TeamAttributes['date']).year

        # print("\n\n TeamAttributes\n" , TeamAttributes)

        #calculate the results for match_api_id
        result = Match.loc[:,["match_api_id","home_team_goal","away_team_goal"]]
        result["result"] = result["home_team_goal"] - result["away_team_goal"]
        #https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
        #Convert Categorical Data to Numerical Data
        #label encoding or an integer encoding
        result.loc[result['result'] > 0, 'result'] = 0 #represent Home_Win with 0
        result.loc[result['result'] == 0, 'result'] = 1 #represent Draw with 1
        result.loc[result['result'] < 0, 'result'] = 2 #represent Away_Win with 2

        #merge Match table and the result of each match_api_id game
        Match = Match.merge(result.loc[:,["match_api_id","result"]] , on="match_api_id")

        Match["home_team_api_id"] = Match["home_team_api_id"].astype(int)
        Match["away_team_api_id"] = Match["away_team_api_id"].astype(int)
        TeamAttributes["team_api_id"] = TeamAttributes["team_api_id"].astype(int)
        Match["year"] = Match["year"].astype(int)
        TeamAttributes["year"] = TeamAttributes["year"].astype(int)

        #Select the columns that we need
        Match = Match.loc[:,["match_api_id","year","home_team_api_id","away_team_api_id","B365H","B365D","B365A","BWH","BWD","BWA","IWH","IWD","IWA","LBH","LBD","LBA","result"]]
        TeamAttributes = TeamAttributes.loc[:,["year","team_api_id","buildUpPlaySpeed", "buildUpPlayPassing", "chanceCreationPassing", "chanceCreationCrossing", "chanceCreationShooting", "defencePressure", "defenceAggregation", "defenceTeamWidth"]]

        # print("\n IMPORTANT -- \n Is home_team_api_id IN team_api_id ? \n",Match["home_team_api_id"].isin(TeamAttributes["team_api_id"]).value_counts(),"\n Is away_team_api_id IN team_api_id ? \n",Match["away_team_api_id"].isin(TeamAttributes["team_api_id"]).value_counts())

        #Merge the odds from Match table and home_team attributes from TeamAttributes table based on Team id and Year(left join)
        MergedTable = Match.merge(TeamAttributes , how="left", left_on=["home_team_api_id","year"], right_on=["team_api_id","year"])
        # print("\n\n1o\n",list(MergedTable),"\n",MergedTable)

        #Merge the new table(MergedTable) and away_team attributes based on Team id and date(left join)
        MergedTable = MergedTable.merge(TeamAttributes , how="left" ,left_on=["away_team_api_id","year"], right_on=["team_api_id","year"] )

        #Drop the columns tha we dont need anymore(we used them for merges)
        MergedTable = MergedTable.drop(["year","match_api_id","team_api_id_x","team_api_id_y","home_team_api_id","away_team_api_id"] , axis=1).fillna(0)

        #for labes take the result column
        self.labels = MergedTable.loc[:,"result"]
        #remove the result column from data
        self.MergedTable = MergedTable.drop(['result'] , axis=1)
        # print("\n\n\n",self.MergedTable,"\n",list(self.MergedTable),"\n\n LABALES \n\n",self.labels,"\n\n NaN Count : \n\n",len(MergedTable)-MergedTable.count())

        # Close connection.
        conn.close()
        print('Closed connection to database: %s\n',database_file);
