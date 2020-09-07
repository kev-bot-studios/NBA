# -*- coding: utf-8 -*-

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import plotly.tools as tls
import base64
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import io
import dash_table


##########################################################################
############################ Helper Functions ############################
##########################################################################

def get_shooting_comps(player, df, year = 2019, n = 5):
    """
    Parameters
    ----------
    player : str
        Name of player to display shooting statistics for
    df : pd.DataFrame
        Player season statistics 
    year : int, optional
        Year to query shooting statistics for. The default is 2019.
    n : int, optional
        Number of shooting comps to retrieve. The default is 5.

    Returns
    -------
    fin : pd.DataFrame
        Shooting comps
    """
    
    # Filter data to year of choice
    summ = df[df['Year'] == year].copy()
    
    # Rank players based on TS% (exclude players who haven't played many games)
    sub = summ[(summ['PTS'] > 2) & (summ['G'] > 5)].copy()
    sub = sub.sort_values('TS%', ascending = False)
    sub.loc[:, 'TS% Rank'] = np.arange(1, len(sub) + 1, 1)
    summ.loc[:, 'TS% Rank'] = sub.loc[:, 'TS% Rank']
    
    
    # Get n most similar players on TS%
    if player not in summ.Player.unique():
        return (pd.DataFrame(columns = summ.columns))
    
    else:   
        player_tsp = summ[summ.Player == player]['TS%'].item()
        sim_players = (summ['TS%'] - player_tsp).abs().sort_values(ascending = True).head(n).index
        fin = summ.loc[sim_players, :].copy()
        fin = fin.sort_values('TS%', ascending = False)
        fin = fin.reset_index()
    

    return(fin)


def create_player_summary(player, df):
    """
    Parameters
    ----------
    player : str
        Name of player to display statistics for
    df : pd.DataFrame
        Player season statistics

    Returns
    -------
    df_summary : pd.DataFrame
        Player comparison dataframe, non-shooting
    """
    
    # Get player, position, starting position and league average stats for 2019
    df_player = df[(df['Player'] == player) & (df['Year'] == 2019)].copy()
    pos = df_player.tail(1).Pos.values[0]
    df_pos = df[(df['Pos'] == pos) & (df['Year'] == 2019)].copy()
    df_league = df[df['Year'] == 2019].copy()
    df_starter = df_pos[(df_pos['GS'] / df_pos['G'] > .8)].copy() #Starting position proxy
    
    # Create aggregate statistics at player, position, starting position and league level
    df_summary = pd.DataFrame(columns = [player, f'Starting {pos}', f'All {pos}', 'League'], 
                              index = ['PTS', 'AST','ORB', 'BLK', 'STL', 'DRB'])
    
    for stat in df_summary.index:
        df_summary.loc[stat, player] = df_player[stat].mean()
        df_summary.loc[stat, f'Starting {pos}'] = df_starter[stat].mean()
        df_summary.loc[stat, f'All {pos}'] = df_pos[stat].mean()
        df_summary.loc[stat, 'League'] = df_league[stat].mean()
        
    # Format display for Dash
    colnames = df_summary.columns
    for col in colnames:
        df_summary[col] = df_summary[col].map('{:,.2f}'.format)
    df_summary = df_summary.reset_index()
    df_summary.columns.values[0] = 'Stat'
    df_summary.loc[:, 'Stat'] = ['PPG','APG','ORB','BPG','SPG','DRB']

    return(df_summary) 
    


def create_hexbin_plot(player, df2):
    """
    Parameters
    ----------
    player : str
        Name of player to create hexbin shot chart for
    df2 : pd.DataFrame
        Player shooting statistics

    Returns
    -------
    shot_chart_dict : dict
        Dictionary with shot chart image layout to append to empty figure
    """
    
    df_sub = df2[df2['PLAYER_NAME'] == player]
    mpl_fig = plt.figure()
    plt.axis('off')
    plt.hexbin(x = 'LOC_X', y = 'LOC_Y', data = df_sub, gridsize= 25, cmap="Blues_r" , mincnt=1)
    plt.xlim([-250, 250])
    plt.ylim([-48, 471])
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read()).decode()
    encoded_image_shot = "data:image/png;base64," + pic_hash
    
    shot_chart_dict = dict(source = encoded_image_shot, xref="x",
            yref="y",
            x=-350,
            y=523,
            sizex=675,
            sizey=621,
            sizing="stretch",
            opacity=.5,
            visible = True,
            layer="above")
    
    return(shot_chart_dict)


def create_court_plot(src_court):
    """
    Parameters
    ----------
    src_court : str
        File path for image location of basketball court

    Returns
    -------
    court_dict: dict
        Dictionary with court image layout to append to figure with shot chart
    """
    
    with open(src_court, "rb") as image_file:
        
        encoded_string_court = base64.b64encode(image_file.read()).decode()
        #add the prefix that plotly will want when using the string as source
        encoded_image_court = "data:image/png;base64," + encoded_string_court
    
    court_dict = dict(source = encoded_image_court, xref="x",
                yref="y",
                x=-250,
                y=423,
                sizex=500,
                sizey=471,
                sizing="stretch",
                opacity=1,
                visible = True,
                layer="below")
    
    return(court_dict)



def load_data():
    """
    Returns
    -------
    Tuple : tuple
        tuple containing player season data (df), player shot data (df2), list
            of players (player_options), list of team options (team_options),
            court dictionary (court), player stat comp dataframe (df_summary),
            and player shot stat comp dataframe (df_shoot)
    """
        
    # Load datasets
    src1 = '/Users/kevincory/Documents/Side_Projects/NBA/data/data80.csv'
    src2 = '/Users/kevincory/Documents/Side_Projects/NBA/data/nba_shotchartdetail_2018-19.csv'
    src_court = '/Users/kevincory/Documents/Side_Projects/NBA/assets/nba_court.jpg'
    src_kobe = '/Users/kevincory/Documents/Side_Projects/NBA/assets/kobe-thumb.jpg'
    df = pd.read_csv(src1)
    df2 = pd.read_csv(src2)
    
    # Data cleanup / organization
    df = df.dropna(subset = ['Player'])
    rec_players = df2.PLAYER_NAME.unique()
    df = df[df['Player'].isin(rec_players)].copy()
    
    # Account for players who switched teams
    df_group = df.copy().drop(['Unnamed: 0', 'Pos', 'Tm'], axis = 1)
    cols = df_group.columns.drop(['Player','Year','G'])
    df_group.loc[:, cols] = df.copy().loc[:, cols].multiply(df_group.G.astype(int), axis = 0)
    agg = df_group.groupby(['Player', 'Year']).sum()[cols.tolist() + ['G']]
    agg.loc[:, cols] = agg.loc[:, cols].div(agg.G, axis = 0)
    pos_tm = df.drop_duplicates(['Player', 'Year'])[['Player', 'Year', 'Pos', 'Tm']].copy()
    df = pd.merge(agg, pos_tm, how = 'inner', on = ['Player', 'Year'])
    df['TS%'] = (df['PTS']) / (2 * df['FGA'] + .44 * df['FTA'])
        
    
    # Initailize starting player, team to analyze
    team = 'HOU'
    player = 'James Harden'
    
    # Filter data for display in Dash
    players = df[df['Tm'] == team].Player.unique()
    teams = df['Tm'].unique()
    player_options = [{'label': player, 'value': player} for player in players]
    team_options = [{'label': team, 'value': team} for team in teams]
    df_summary = create_player_summary(player, df)
    df_shoot = get_shooting_comps(player, df)
    
    
    # Create court image
    court = create_court_plot(src_court)
    
    
    return(df, df2, player, player_options, team_options, court, df_summary, df_shoot)
    

##########################################################################
############################### App Layout ###############################
##########################################################################

    
df, df2, player, player_options, team_options, court_dict, df_summary, df_shoot = load_data()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

colors = {
    'background': "#082255",
    'text': "#fff", 
    'chart': "#42C4F7"
}

fig = px.bar(df, x="Year", y=["PTS"], title = "PPG")
fig2 = px.bar(df, x="Year", y=["AST"], title = 'APG')
fig3 = px.bar(df, x = "Year", y = ["ORB"], title = 'ORB')
fig4 = go.Figure(data ={})
fig4.add_layout_image(court_dict)
fig4.add_layout_image(create_hexbin_plot(player, df2))
fig5 = px.bar(df_shoot, x = "Player", y = ['TS%', 'FG%', 'FT%', '3P%'], 
              title = 'True Shooting Percentage', barmode = 'group')



fig.update_layout(plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'])
fig2.update_layout(plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'])
fig3.update_layout(plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'])
fig4.update_layout(yaxis = dict(range = [-50, 420], title = "", showticklabels = False), 
                   xaxis = dict(range = [-250, 250], title = 'Lighter = More Shots', showticklabels = False), 
                   title = "Shot Chart (2018 - 2019 Season)",
                   plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color = colors['text'])
fig5.update_layout(plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'])


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    
    html.Div(children = [
        
        html.Div(
            html.Img(src = app.get_asset_url('kobe-thumb.jpg'),
                     style = {'height':'30%', 'width':'30%'}), 
                     style = {'display':'inline-block', 'width':'15%', 'textAlign':'left'}),
         
        
        html.Div(
            html.H1(
                children='NBA Dashboard',
                style={
                    'textAlign': 'center',
                    'color': colors['text'],
                }), style = {'display':'inline-block','width':'70%','textAlign':'center', 'vertical-align':'bottom'}),
        
        
        html.Div(
            html.Img(src = app.get_asset_url('kobe-thumb.jpg'),
                     style = {'height':'30%', 'width':'30%'}), 
                     style = {'display':'inline-block', 'width':'15%', 'textAlign':'right'}),
        
        ]),
            

    html.Div(children='A snapshot of a player\'s career, and a deeper dive into their 2018-2019 season', 
             style= {'textAlign': 'center',
                     'color': colors['text']
    }),
    
    
    html.Label('Choose Team', style = {'color':colors['text']}),
        dcc.Dropdown(
            id = 'team_select',
            options= team_options,
            value='HOU',
    ),
    
    html.Label('Choose Player', style = {'color':colors['text']}),
        dcc.Dropdown(
            id = 'player_select',
            options= player_options,
            value='Allen Iverson',
    ),  
    
    html.Label('Statistics', style = {'color':colors['text']}),
        dcc.RadioItems(
            id='stats_type',
            options=[{'label': i, 'value': i} for i in ['Offense', 'Defense']],
            value='Offense',
            labelStyle={'display': 'inline-block', 'color':colors['text']}
    ),

    
    html.Div(children = [
        
        
        html.Div(
            dcc.Graph(
            id='player_graph1',
            figure=fig
            ), style ={'width': '33.3%','display': 'inline-block'}),
        
        html.Div(
            dcc.Graph(
                id = 'player_graph2',
                figure=fig2
            ), style = {'width': '33.3%','display': 'inline-block'}),
        
        html.Div(
            dcc.Graph(
                id = 'player_graph3',
                figure=fig3
            ), style = {'width': '33.3%', 'display': 'inline-block'}),
        
    ]),
    
    
    html.Div(children = [
        
        html.Div(
            
            dcc.Graph(
            id='shot_chart',
            figure = fig4,
            style = {'textAlign':'center',
                     'margin':{'b':0, 't':0, 'l':0, 'r':0}}
            ),      style = {'width':'33.3%',
                        'display':'inline-block'}), 

                        
        html.Div(
            dcc.Graph(
            id='comps_chart',
            figure = fig5
            ), style = {'width': '33.3%', 
                        'display':'inline-block',
                        'vertical-align':'top'}),
        
                    
        html.Div(children = [
            
            html.Div(
                html.H1(
                    children='Season Stat Comparison (2018-2019 Season)',
                    style={
                        'textAlign': 'left',
                        'color': colors['text'],
                        'font-size':'1.9rem',
                        'margin-top':'3.2rem',
                        'font-family':'Arial'
                    }), style = {'vertical-align':'top'}),
                
            
            html.Div(
                dash_table.DataTable(
                id = 'stats_2019', 
                columns=[{"name": i, "id": i} for i in df_summary.columns],
                data=df_summary.to_dict('records'), 
                style_as_list_view = True,
                css=[{'selector': 'table','rule': 'table-layout:fixed; width: 85%'}],
                style_header = {'fontWeight':'bold'},
                style_table = {'minWidth':'100%', 'margin-top':'25px',
                               'margin-left':'25px'},
                style_cell={'width': '{}%'.format(len(df_summary.columns)), 'height':40,
                            'textAlign':'center', 'whiteSpace':'normal',
                            'backgroundColor':colors['text']}
                ), style = {'textAlign':'center'}),
            
                ], style = {'width': '33.3%', 
                            'display':'inline-block',
                            'vertical-align':'top'}),
                
        ]),
                        

])
                            
                            
############################################################################
############################ Callback Functions ############################
############################################################################
    

@app.callback(
    [Output('player_graph1', 'figure'),
    Output('player_graph2', 'figure'),
    Output('player_graph3', 'figure')],
    [Input('player_select', 'value'),
    Input('team_select', 'value'),
    Input('stats_type', 'value')])

def update_player_stats(player, team, stats_type):
    
    sub = df[df['Tm'] == team].copy()
    df_player = df[df.Player == player].copy()
    trans_dur = 250
    
    if stats_type == 'Offense':
        
        fig = px.bar(df_player, x="Year", y=["PTS"], color = 'Tm', hover_name = 'Player',
                     hover_data = {'variable':False,'Tm':True, 'Pos':True,'Age':True},
                     color_discrete_sequence = px.colors.sequential.Teal)
        fig.update_layout(transition_duration=trans_dur, title = "Points per Game (PPG)", 
                          showlegend = True, yaxis_title = 'PPG',
                          plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], 
                          font_color = colors['text'], legend_title_text = None)
        
        fig2 = px.bar(df_player, x="Year", y=["AST"], color = 'Tm', hover_name = 'Player',
                      hover_data = {'variable':False,'Tm':True, 'Pos':True,'Age':True},
                      color_discrete_sequence = px.colors.sequential.Teal)
        fig2.update_layout(transition_duration=trans_dur, title = 'Assists per Game (APG)', 
                           showlegend = False, yaxis_title = 'APG',
                           plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], 
                           font_color = colors['text'], legend_title_text = None)
        
        fig3 = px.bar(df_player, x="Year", y=["ORB"], color = 'Tm', hover_name = 'Player',
                      hover_data = {'variable':False,'Tm':True, 'Pos':True,'Age':True},
                      color_discrete_sequence = px.colors.sequential.Teal)
        fig3.update_layout(transition_duration=trans_dur, title = 'Offensive Rebounds per Game (ORB)', 
                           showlegend = False, yaxis_title = 'ORB',
                           plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], 
                           font_color = colors['text'], legend_title_text = None)
    
    else:
        
        fig = px.bar(df_player, x="Year", y=["BLK"], color = 'Tm', hover_name = 'Player',
                     hover_data = {'variable':False,'Tm':True, 'Pos':True,'Age':True},
                     color_discrete_sequence = px.colors.sequential.Teal)
        fig.update_layout(transition_duration=trans_dur, title = 'Blocks per Game (BPG)', 
                          showlegend = True, yaxis_title = 'BPG',
                          plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], 
                          font_color = colors['text'], legend_title_text = None)
        
        fig2 = px.bar(df_player, x="Year", y=["STL"], color = 'Tm', hover_name = 'Player',
                      hover_data = {'variable':False,'Tm':True, 'Pos':True,'Age':True},
                      color_discrete_sequence = px.colors.sequential.Teal)
        fig2.update_layout(transition_duration=trans_dur, title = 'Steals per Game (SPG)', 
                           showlegend = False, yaxis_title = 'SPG',
                           plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], 
                           font_color = colors['text'], legend_title_text = None)
        
        fig3 = px.bar(df_player, x="Year", y=["DRB"], color = 'Tm', hover_name = 'Player',
                      hover_data = {'variable':False,'Tm':True, 'Pos':True,'Age':True},
                      color_discrete_sequence = px.colors.sequential.Teal)
        fig3.update_layout(transition_duration=trans_dur, title = 'Defensive Rebounds per Game (DRB)', 
                           showlegend = False, yaxis_title = 'DRB',
                           plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], 
                           font_color = colors['text'], legend_title_text = None)
        
    return fig, fig2, fig3



@app.callback(
    [Output('player_select', 'options'),
    Output('player_select', 'value')],
    [Input('team_select', 'value')])

def update_team(Team):
    
    df_team = df[df['Tm'] == Team]
    players = df_team['Player'].unique()
    player_options =[{'label': player, 'value': player} for player in players]
    player = players[0]
    return player_options, player



@app.callback(
    [Output('shot_chart', 'figure')],
    [Input('player_select', 'value')])

def update_shot_plot(player):
    
    fig4 = go.Figure(data ={})
    df_shot = df2[df2['PLAYER_NAME'] == player].copy()
    fig4.add_layout_image(court_dict)
    fig4.add_layout_image(create_hexbin_plot(player, df2))
    fig4.update_layout(yaxis = dict(range = [-50, 420], title = "", showticklabels = False), 
               xaxis = dict(range = [-250, 250], title = 'Lighter = More Shots', showticklabels = False), 
               title = "Shot Chart (2018 - 2019 Season)",
               plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], 
               font_color = colors['text'])
    
    return fig4, 
    


@app.callback(
    [Output('comps_chart', 'figure')],
    [Input('player_select', 'value')])

def update_comps_chart(player):
    
    df_comps = get_shooting_comps(player, df, year = 2019, n = 5)
    if df_comps.empty:
        fig5 = px.bar( dict(data = [{},{},{},{}])) 
    else:
        fig5 = px.bar(df_comps, x = 'Player', y = ['TS%', 'FG%', 'FT%', '3P%'], 
                      title = 'True Shooting Percentage (2018 - 2019 Season)', barmode = 'group', 
                      hover_name = 'Player', 
                      hover_data = {'Player':False,'Tm':True,'Pos':True,'TS% Rank':True})
        
    fig5.update_layout(plot_bgcolor=colors['background'], paper_bgcolor=colors['background'],
                       font_color = colors['text'], yaxis = {'tickformat':'.0%', 
                       'title':'Shooting Percentage'}, legend_title_text = None)

    return fig5, 


@app.callback(
    [Output('stats_2019', 'data'),
     Output('stats_2019', 'columns')],
    [Input('player_select', 'value')])

def update_dataTable(player):
    
    res = create_player_summary(player, df)
    res_dict = res.to_dict('records')
    cols = [{"name": i, "id": i} for i in res.columns]
    
    return res_dict, cols



if __name__ == '__main__':
    app.run_server(debug = True)
    
    

    
