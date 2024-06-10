import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import json

import sys

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)

def check_usage():
    if len(sys.argv) != 3:
        print('python extract_nba_rolling_stats.py <rolling_period> <list of stats>')
        sys.exit()    

class NBAData:
    '''Class object to read in a specified excel sheet of data
    from the NBA data excel file'''

    def __init__(self, sheet_name):
        self.sheet_name = sheet_name
    
    def data(self):
        excel_path = excel_path = '/'.join(['.','Data',
                                            'NBA Stats_2021-2023_01292024.xlsx'])
        
        excel_sheet_name = self.sheet_name.title()

        sheet_index_cols = {'Players':'player_id',
                            'Teams':'team_id',
                            'Stats':None,
                            'Games':'game_id'}

        print('Reading in %s' %excel_sheet_name)
        raw_data = pd.read_excel(excel_path,
                             sheet_name = excel_sheet_name,
                             header = 0,
                             index_col = sheet_index_cols[excel_sheet_name],
                             engine = 'openpyxl')

        if excel_sheet_name == 'Stats':
            # Clean stats data and add column for fantasy points
            # Convert 'min' to numeric and fill NaN with 0
            raw_data['min'] = pd.to_numeric(raw_data['min'],
                                        errors = 'coerce')\
                            .fillna(0)
            
            raw_data.drop(raw_data[raw_data['min'] == 0].index,
                          inplace = True)

            # Shorthand turnover column
            raw_data.rename(columns = {'turnover':'to'},
                        inplace = True)

            # Fantasy points (PrizePicks)
            ## Points = 1
            ## Rebound = 1.2
            ## Assists = 1.5
            ## Block = 3
            ## Steals = 3
            ## Turnover = -1
            raw_data['fpts'] = raw_data['pts']\
                            + (1.2 * raw_data['reb'])\
                            + (1.5 * raw_data['ast'])\
                            + (3 * raw_data['blk'])\
                            + (3 * raw_data['stl'])\
                            + (-1 * raw_data['to'])
        elif excel_sheet_name == 'Games':
            raw_data['game_date'] == pd.to_datetime(raw_data['game_date'],
                                                errors = 'coerce')

        return raw_data

class UserParameters:
    '''Class object to store specified parameters and
    return objects useful for script'''

    def __init__(self, rolling_period, stat_cats):
        self.rolling_period = rolling_period
        self.stat_cats = stat_cats
        self.valid_cats = ['fga','fgm',
                        'fta','ftm',
                        'fg3a','fg3m',
                        'oreb','dreb','reb',
                        'pts','ast',
                        'stl','blk',
                        'pf','to',
                        'fpts']

    def check_valid(self):# Check if specified rolling period is a valid integer
        try:
            int(self.rolling_period)
        except TypeError:
            print('Specified rolling period not of type integer')
            sys.exit()

        if len(self.stat_cats.split(',')) > 0:
            specified_stats = [cat.strip()
                            for cat in self.stat_cats.split(',')]

            invalid_cats = [cat
                            for cat in specified_stats
                            if cat not in self.valid_cats]

            if len(invalid_cats) > 0:
                raise KeyError('Invalid statistical category provided')
    
    def per36stat_col_names(self):
        return [cat+'_per36' for cat in self.stat_cats] + [cat+'_mean' for cat in self.stat_cats]

    def pergamestat_col_names(self):
        return {cat:cat+'_pg'
                for cat in self.valid_cats}

def merge_for_rolling():
    # Get stats data
    stats_data = NBAData('stats').data()

    # Get games data
    games_data = NBAData('games').data()

    # Get player data
    players_data = NBAData('players').data()

    # Merge date of games
    stats_dates = pd.merge(stats_data,
                            games_data[['game_date']],
                            how = 'left',
                            left_on = 'game_id',
                            right_index = True)\
                        .sort_values(['game_date','player_id'])
    
    # Merge player positions
    stats_dates_pos = pd.merge(stats_dates,
                                players_data[['player_position']],
                                how = 'left',
                                left_on = 'player_id',
                                right_index = True)
    
    return stats_dates_pos

def calc_shift_rolling_stats(data: pd.DataFrame,
                             method: str,
                             parameters: UserParameters):
    '''Function to calculate the rolling statistics
    provided a dataset, its contents, and parameters'''

    def get_team_rolling_stats(df,
                               params: UserParameters):
        # Define a groupby function to get rolling team statistics
        print('Calculating individual team rolling statistics...')
        def team_rolling_mean(group):
            group_indexed = group.set_index('game_date')
            
            rolling_group = group_indexed\
                            [params.stat_cats]\
                            .rolling(params.rolling_period,
                                    min_periods = 1)\
                            .mean()\
                            .rename(columns = params.pergamestat_col_names())
            return rolling_group
        
        # Overall Team Defensive Efficiency
        team_def_eff = df\
                            .groupby(['opponent_team_id',
                                      'game_date',
                                    #   'player_position'
                                      ])\
                            [params.stat_cats]\
                            .sum()\
                            .reset_index()
        
        team_stats_rolling_def_eff = team_def_eff.groupby(['opponent_team_id',
                                                            #    'player_position'
                                                               ])\
                                    .apply(team_rolling_mean)\
                                    .reset_index()\
                                    .rename(columns = {'opponent_team_id':'team_id'})

        team_stats_rolling_def_eff.rename(columns = {value:'opp_'+value
                                                    for key,value in params.pergamestat_col_names().items()},
                                        inplace = True)
        
        # Overall Team Offensive Efficiency
        team_off_eff = df\
                            .groupby(['team_id','game_date',
                                    #   'player_position'
                                      ])\
                            [params.stat_cats]\
                            .sum()\
                            .reset_index()
        
        team_stats_rolling_off_eff = team_off_eff.groupby(['team_id',
                                                            #    'player_position'
                                                               ])\
                                    .apply(team_rolling_mean)\
                                    .reset_index()
        
        # Defensive Efficiency by Position
        team_def_eff_pos = df\
                            .groupby(['opponent_team_id',
                                      'game_date',
                                      'player_position'
                                      ])\
                            [params.stat_cats]\
                            .sum()\
                            .reset_index()
        
        team_pos_stats_rolling_def_eff = team_def_eff_pos.groupby(['opponent_team_id',
                                                               'player_position'
                                                               ])\
                                    .apply(team_rolling_mean)\
                                    .reset_index()\
                                    .rename(columns = {'opponent_team_id':'team_id'})

        team_pos_stats_rolling_def_eff.rename(columns = {value:'opp_pos_'+value
                                                    for key,value in params.pergamestat_col_names().items()},
                                        inplace = True)
        
        # Offensive Efficiency by Position
        team_off_eff_pos = df\
                            .groupby(['team_id','game_date',
                                      'player_position'
                                      ])\
                            [params.stat_cats]\
                            .sum()\
                            .reset_index()
        
        team_pos_stats_rolling_off_eff = team_off_eff_pos.groupby(['team_id',
                                                               'player_position'
                                                               ])\
                                    .apply(team_rolling_mean)\
                                    .reset_index()
        
        team_pos_stats_rolling_off_eff.rename(columns = {col:'pos_'+col
                                                         for col in team_pos_stats_rolling_off_eff.columns
                                                         if ('_pg' in col)},
                                                inplace = True)
        

        # Merge team defensive and offensive efficiencies
        team_eff = pd.merge(team_stats_rolling_off_eff,
                                team_stats_rolling_def_eff,
                                how = 'inner',
                                left_on = ['team_id',
                                           'game_date',
                                        #    'player_position'
                                           ],
                                right_on = ['team_id',
                                            'game_date',
                                            # 'player_position'
                                            ])
        
        team_eff_pos = pd.merge(team_pos_stats_rolling_off_eff,
                                team_pos_stats_rolling_def_eff,
                                how = 'outer',
                                left_on = ['team_id',
                                           'game_date',
                                           'player_position'
                                           ],
                                right_on = ['team_id',
                                            'game_date',
                                            'player_position'
                                            ])
        
        overall_team_eff = pd.merge(team_eff,
                                    team_eff_pos,
                                    how = 'inner',
                                    left_on = ['team_id',
                                               'game_date'],
                                    right_on = ['team_id',
                                                'game_date'])
        
        return overall_team_eff

    def get_league_team_rolling_stats(df,params):
        print('Calculating league team rolling statistics...')
        def league_team_stats(group):
            '''
            Grouping function to get standard deviation of stat per36 for each player on a given date
            Need to account for offseason/days where there are no games
            '''
            team_stat_cats = ['_'.join([cat,'pg']) for cat in params.stat_cats]
            opp_stat_cats = ['_'.join(['opp',cat,'pg']) for cat in params.stat_cats]

            resample_stat_cats = team_stat_cats + opp_stat_cats
            
            # Sort by date and player prior to resampling
            group_sorted = group.sort_values(['game_date','team_id'])\
                                .set_index(['game_date'])
            
            # Need to add rolling logic
            group_rolling = group_sorted\
                            .groupby('team_id')\
                            [resample_stat_cats]\
                            .rolling(params.rolling_period,
                                    min_periods = 1)\
                            .mean()\
                            .reset_index()\
                            .set_index('game_date')

            # Resample data to daily by each player
            ## Forward fill blank values
            group_resampled = group_rolling.groupby('team_id')\
                                .resample('1D')\
                                [resample_stat_cats]\
                                .last()
            
            # Were games played on date
            ## Due to resample, dates in the off season were added
            ## Need to remove; will cause calculations over at the beginning of each season
            date_no_minutes = group_resampled.groupby(level = 1)\
                                .apply(lambda x: x.isna()\
                                                .all()
                                    )
            
            # Drop dates with no games
            ## Includes in-season breaks
            date_no_games = date_no_minutes[(date_no_minutes == True).all(axis = 1)].index
            
            group_resampled.drop(index = date_no_games,
                                level = 1,
                                inplace = True)
            
            # Forward fill values by player
            final_group_resampled = group_resampled.groupby(level = [0])[resample_stat_cats].ffill()
            
            # Calculate the standard deviation of pts_per36 for all players by date
            final_rolling_stats = final_group_resampled.groupby(level = 1).agg(['mean',
                                                                                'std'])
            
            return final_rolling_stats
        
        # Calculate teams offensive pts production by position per game
        team_pos_off_total = df\
                                .groupby(['game_date',
                                            'player_position',
                                            'team_id'])\
                                [params.stat_cats]\
                                .sum()\
                                .reset_index()\
                                .rename(columns = params.pergamestat_col_names())

        # Calculate teams defensive pts production by position per game
        team_pos_def_total = df\
                                .groupby(['game_date',
                                            'player_position',
                                            'opponent_team_id'])\
                                [params.stat_cats]\
                                .sum()\
                                .reset_index()\
                                .rename(columns = {'opponent_team_id':'team_id'})

        team_pos_def_total.rename(columns = {key:'opp_'+value
                                                for key,value in params.pergamestat_col_names().items()},
                                    inplace = True)

        # Merge teams offensive and defensive production by position
        team_pos_eff_total = pd.merge(team_pos_off_total,
                                    team_pos_def_total,
                                    how = 'outer',
                                    left_on = ['game_date',
                                               'team_id',
                                               'player_position'
                                               ],
                                    right_on = ['game_date',
                                                'team_id',
                                                'player_position'
                                                ])
        
        team_pos_rolling_eff = team_pos_eff_total\
                                .groupby('player_position')\
                                .apply(league_team_stats)
        
        team_pos_rolling_eff.columns = ['_'.join(i)
                                        for i in team_pos_rolling_eff.columns]
        
        team_pos_rolling_eff.reset_index(inplace = True)

        return team_pos_rolling_eff

    def get_individual_rolling_stats(df,params):
        print('Calculating individual player rolling statistics...')
        player_stat_cats = params.stat_cats + ['min']

        def player_rolling_sum(group):
            group_indexed = group.sort_values('game_date')\
                            .set_index('game_date')
            
            rolling_group = group_indexed\
                                [player_stat_cats]\
                            .rolling(params.rolling_period,
                                        min_periods = 1)\
                            .agg(['sum','mean'])
            
            rolling_group.columns = ['_'.join(i) for i in rolling_group.columns]
            
            rolling_min_mean = group_indexed\
                                [['min']]\
                                .rolling(params.rolling_period,
                                        min_periods = 1)\
                                .mean()
            
            rolling_merged = pd.merge(rolling_min_mean,
                                    rolling_group,
                                    how = 'left',
                                    left_index = True,
                                    right_index = True,
                                    suffixes = ['_mean','_cumsum'])
            
            return rolling_merged

        player_stats_rolling_eff = df.groupby('player_id')\
                                    .apply(player_rolling_sum)

        for cat in params.stat_cats:
            player_stats_rolling_eff[cat+'_per36'] = player_stats_rolling_eff[cat+'_sum']\
                                                    * (36/player_stats_rolling_eff['min_sum'])
        
        actual_eff_merged = pd.merge(df[['game_date','player_id','player_position',
                                                    'min'] + params.stat_cats],
                                    player_stats_rolling_eff.reset_index()\
                                        .drop(['min_sum'],
                                            axis = 1),
                                    how = 'left',
                                    left_on = ['player_id','game_date'],
                                    right_on = ['player_id','game_date'])#\
                            #.sort_values(['player_id','game_date'])

        return actual_eff_merged

    def get_league_player_rolling_stats(df,params):
        print('Calculating league player rolling statistics...')
        def league_player_pts_stats(group):
            '''
            Grouping function to get standard deviation of pts_per36 for each player on a given date
            Need to account for offseason/days where there are no games
            '''
            # Sort by date and player prior to resampling
            group_sorted = group.sort_values(['game_date','player_id'])\
                                .set_index(['game_date'])
            
            # group_rolling = group_sorted\
            #             .groupby('team_id')\
            #             [resample_stat_cats]\
            #             .rolling(user_params.rolling_period,
            #                     min_periods = 1)\
            #             .mean()\
            #             .reset_index()\
            #             .set_index('game_date')

            # Resample data to daily by each player
            ## Forward fill blank values
            group_resampled = group_sorted.groupby(['player_id',
                                                ])\
                                .resample('1D')\
                                [params.per36stat_col_names()]\
                                .last()
            
            # Were games played on date
            ## Due to resample, dates in the off season were added
            ## Need to remove; will cause calculations over at the beginning of each season
            date_no_minutes = group_resampled.groupby(level = 1)\
                                .apply(lambda x: x.isna()\
                                                .all()
                                    )
            
            # Drop dates with no games
            ## Includes in-season breaks
            date_no_games = date_no_minutes[(date_no_minutes[params.per36stat_col_names()] == True).all(axis = 1)].index
            
            group_resampled.drop(index = date_no_games,
                                level = 1,
                                inplace = True)
            
            # Forward fill values by player
            final_group_resampled = group_resampled.groupby(level = [0])[params.per36stat_col_names()].ffill()
            
            # Calculate the standard deviation of pts_per36 for all players by date
            final_rolling_stats = final_group_resampled.groupby(level = [1]).agg(['mean',
                                                                                'std'])
            final_rolling_stats.columns = ['_'.join(i) for i in final_rolling_stats.columns]
            
            return final_rolling_stats
        
        league_player_rolling_stats = df.groupby('player_position',
                                                                dropna = False)\
                                            .apply(league_player_pts_stats)
        
        league_player_rolling_stats.reset_index(inplace = True)

        return league_player_rolling_stats

    if method == 'indiv_teams':
        rolling_stats = get_team_rolling_stats(data, parameters)

        print('Shifting data...')
        data_shifted = rolling_stats.groupby(['team_id',
                                              'player_position'
                                              ])\
                        .apply(lambda x: x.sort_values('game_date')\
                                        .set_index('game_date')\
                                        [[col for col in rolling_stats.columns if col.endswith('_pg')]]\
                                        .shift(1))\
                        .reset_index()
    elif method == 'league_teams':
        rolling_stats = get_league_team_rolling_stats(data, parameters)

        print('Shifting data...')
        data_shifted = rolling_stats.groupby('player_position')\
                                        .apply(lambda x: x.sort_values('game_date')\
                                                        .set_index('game_date')\
                                                        [[col for col in rolling_stats.columns if '_pg_' in col]]\
                                                        .shift(1))\
                                        .reset_index()
    elif method == 'indiv_players':
        rolling_stats = get_individual_rolling_stats(data,parameters)

        print('Shifting data...')
        data_shifted = rolling_stats.groupby(['player_id','player_position'])\
                                    .apply(lambda x: x.sort_values('game_date')\
                                                    .set_index('game_date')\
                                                    [parameters.per36stat_col_names() + ['min_mean']]\
                                                    .shift(1))\
                                    .reset_index()
    elif method == 'league_players':
        data_shifted = get_league_player_rolling_stats(data, parameters)

        print('Shifting data...')
        # data_shifted = rolling_stats\
        #                 .groupby('player_position')\
        #                 .apply(lambda x: x.sort_values('game_date')\
        #                                 .set_index('game_date')\
        #                                 [[col for col in rolling_stats.columns if '_per36_' in col]]\
        #                                 .shift(1))\
        #                 .reset_index()
    else:
        raise KeyError("Method for calculating and shifting rolling statistics invalid."
                       "Specify either 'indiv_teams', 'league_teams', 'indiv_players', or 'league_players'")
    
    return data_shifted

def merge_to_final(original_data,
                   p: UserParameters,
                   **kwargs):
    indiv_teams_data = kwargs.get('indiv_teams_data')
    league_teams_data = kwargs.get('league_teams_data')
    indiv_players_data = kwargs.get('indiv_players_data')
    league_players_data = kwargs.get('league_players_data')

    print('Merging all data...')
    # Merge data
    ## Merge Team stats
    rolling_stats_teams = pd.merge(indiv_teams_data.rename(columns = {col:'team_'+col
                                                                        for col in indiv_teams_data.columns
                                                                        if col.endswith('_pg')}),
                                    league_teams_data.rename(columns = {col:'league_'+col
                                                                            for col in league_teams_data.columns
                                                                            if '_pg_' in col}),
                                    how = 'outer',
                                    left_on = ['player_position',
                                               'game_date'],
                                    right_on = ['player_position',
                                                'game_date'])\
                                .sort_values(['game_date',
                                              'player_position',
                                              'team_id'])
    
    ## Merge Player stats
    rolling_stats_players = pd.merge(indiv_players_data.rename(columns = {**{'min_mean':'player_avg_min'},\
                                                                        **{col:'player_'+col for col in p.per36stat_col_names()}
                                                                                }),
                                    league_players_data.rename(columns = {col:'league_player_'+col
                                                                            for col in league_players_data
                                                                            if '_per36_' in col}),
                                    how = 'outer',
                                    left_on = ['player_position','game_date'],
                                    right_on = ['player_position','game_date'])\
                            .sort_values(['game_date','player_position','player_id'])

    ## Merge Player and Team stats to relevant original data
    # Table of relevant fields from original data
    rel_stats = original_data[['game_date',
                                'player_id','player_position',
                                'team_id','opponent_team_id',
                                'min']
                                + p.stat_cats]
    
    rel_rolling_players = pd.merge(rel_stats,
                                    rolling_stats_players,
                                    how = 'outer',
                                    left_on = ['game_date','player_id','player_position'],
                                    right_on = ['game_date','player_id','player_position'])
    
    rel_rolling_team_off = pd.merge(rel_rolling_players,
                                    rolling_stats_teams[['team_id','player_position','game_date']
                                                        + [col for col in rolling_stats_teams if (('_pg' in col)
                                                                                                & ('_opp_' not in col))]],
                                    how = 'left',
                                    left_on = ['game_date','team_id','player_position'],
                                    right_on = ['game_date','team_id','player_position'])
    
    rel_rolling_team_opp = pd.merge(rel_rolling_team_off,
                                    rolling_stats_teams[['team_id','player_position','game_date']
                                                        + [col for col in rolling_stats_teams if (('_pg' in col)
                                                                                                & ('_opp_' in col))]]\
                                        .rename(columns = {col:'opponent_'+col
                                                        for col in rolling_stats_teams.columns
                                                        if ('team_opp_' in col)}),
                                    how = 'left',
                                    left_on = ['game_date','opponent_team_id','player_position'],
                                    right_on = ['game_date','team_id','player_position'],
                                suffixes = ['','_y'])\
                            .drop('team_id_y',
                                axis = 1)
    
    return rel_rolling_team_opp

def get_rolling_stats(original_data,
                      p: UserParameters):
    # Individual Teams
    team_rolling_stats = calc_shift_rolling_stats(data = original_data,
                                            parameters = p,
                                            method = 'indiv_teams')
    
    # League by Teams
    league_rolling_stats_teams = calc_shift_rolling_stats(data = original_data,
                                                    parameters = p,
                                                    method = 'league_teams')

    # Individual Players
    player_rolling_stats = calc_shift_rolling_stats(data = original_data,
                                            parameters = p,
                                            method = 'indiv_players')
    
    # League by Players
    league_rolling_stats_players = calc_shift_rolling_stats(data = player_rolling_stats,
                                                      parameters = p,
                                                      method = 'league_players')
    
    full_rolling_merged = merge_to_final(original_data,
                                         p = p,
                                         indiv_teams_data = team_rolling_stats,
                                         league_teams_data = league_rolling_stats_teams,
                                         indiv_players_data = player_rolling_stats,
                                         league_players_data = league_rolling_stats_players
                                         )\
                            .drop_duplicates(['game_date','player_id','opponent_team_id'],
                                             keep = 'last')
    
    return full_rolling_merged

def standardize_data(data,
                     stat_cats):
    data_copy = data.copy()

    print('Standardizing data...')
    for cat in stat_cats:
        # Normalize cat
        if cat == 'fpts':
            data_copy[cat+'_normed'] = data_copy[cat].apply(lambda x: (x**(1/2)).real)
            data_copy[cat+'_normed_scaled'] = (data_copy[cat+'_normed'] - data_copy[cat+'_normed'].min())/(data_copy[cat+'_normed'].max() - data_copy[cat+'_normed'].min())
        elif cat == 'pts':
            data_copy[cat+'_normed'] = data_copy[cat].apply(lambda x: (x**(1/3)).real)
            data_copy[cat+'_normed_scaled'] = (data_copy[cat+'_normed'] - data_copy[cat+'_normed'].min())/(data_copy[cat+'_normed'].max() - data_copy[cat+'_normed'].min())
        else:
            data_copy[cat+'_normed'] = (data_copy[cat] - data_copy[cat].min())/(data_copy[cat].max() - data_copy[cat].min())

        # Team Data Standardization
        data_copy['team_'+cat+'_pg_stand'] = (data_copy['team_'+cat+'_pg'] - data_copy['league_'+cat+'_pg_mean'])/data_copy['league_'+cat+'_pg_std']

        # Opponent Team Data Standardization
        data_copy['opponent_team_opp_pos_'+cat+'_pg_stand'] = (data_copy['opponent_team_opp_pos_'+cat+'_pg'] - data_copy['league_opp_'+cat+'_pg_mean'])/data_copy['league_opp_'+cat+'_pg_std']
        
        # Player Data Standardization
        data_copy['player_'+cat+'_per36_stand'] = (data_copy['player_'+cat+'_per36'] - data_copy['league_player_'+cat+'_per36_mean'])/data_copy['league_player_'+cat+'_per36_std']

        data_copy[cat+'_mean_stand'] = (data_copy['player_'+cat+'_mean'] - data_copy[cat+'_mean_mean'])/data_copy[cat+'_mean_std']
    
    data_copy['min_stand'] = (data_copy['min'] - data_copy['min'].min())/(data_copy['min'].max() - data_copy['min'].min())

    return data_copy
   
def export_data(data):
    print('Writing to excel...')
    file_name = input('  Specify output excel file name (without file extension): ')

    excel_path = '/'.join(['.','Data',
                           file_name + '.xlsx'])
    
    with pd.ExcelWriter(excel_path) as writer:
        data.to_excel(writer,
                      index = False)

class run_extract():
    def __init__(self,roll_period, categories):
        self.roll_period = roll_period
        self.categories = categories
    
        self.user_params = UserParameters(rolling_period = self.roll_period,
                                        stat_cats = self.categories)
        
        print('Calculating stats for %s' %self.user_params.stat_cats)

        self.extracted_data = merge_for_rolling()
        
        self.rolling_shifted_data = get_rolling_stats(self.extracted_data,
                                                self.user_params)
        
        self.standardized_data = standardize_data(self.rolling_shifted_data,
                                            self.user_params.stat_cats)
        
        print('Done!')
            
    def export(self):
        export_data(self.standardized_data)

def main():
    check_usage()

    input_rolling_period = int(sys.argv[1])
    input_stat_cats = [cat.strip()
                for cat in sys.argv[2].split(',')]

    run_extract(roll_period = input_rolling_period,
                categories= input_stat_cats)

if __name__ == '__main__':
    main()