#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   ConfidencePickEm.py
@Time    :   2023/10/11 13:19:41
@Author  :   Taylor Firman
@Version :   0.1.0
@Contact :   tefirman@gmail.com
@Desc    :   Simulation tools for a Confidence Pick'em league where every player picks
a winner for each game of the week and assigns a point value based on how confident 
they are in that winner (between 1 and the number of games). For each correct pick, 
the player receives the amount of points assigned to that game and the player with 
the most points that week wins.
'''

from .teams import teams
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
import datetime
import os
import datetime
import numpy as np
import optparse

nfl_teams = pd.DataFrame(teams)

def load_games(probs_html: str) -> pd.DataFrame:
    tempData = open(probs_html,"r")
    response = tempData.read()
    tempData.close()
    soup = BeautifulSoup(response, "html.parser")
    tables = soup.find_all("div", attrs={"class": "ysf-matchup-dist yspmainmodule"})
    inds, dates, teams, pcts, pts, spreads = [], [], [], [], [], []
    for table in tables:
        inds.extend(2*[tables.index(table)])
        dates.extend(2*[table.find_all("div", attrs={"class": "hd"})[0].text])
        teams.extend(table.find_all("dd", attrs={"class": "team"}))
        pcts.extend(table.find_all("dd", attrs={"class": "percent"}))
        deets = table.find_all("td")
        pts.append(float(deets[0].text))
        pts.append(float(deets[2].text))
        spreads.append(-1*float(deets[6].text.split()[0]))
        spreads.append(-1*spreads[-1])
    probs = pd.DataFrame({"game_ind":inds,"game_date":dates,"team":[team.text[:-1] for team in teams],\
    "pick_prob":[pct.text[:-1] for pct in pcts],"pick_pts":pts,"spread":spreads})
    probs.game_date = pd.to_datetime(str(datetime.datetime.now().year) + \
    ", " + probs.game_date, format="%Y, %A, %b %d, %I:%M %p %Z")
    probs["still_to_play"] = probs.game_date.dt.tz_localize(None) > datetime.datetime.now()
    probs.team = probs.team.str.replace("@ ","").str.replace("(LAR)", "Rams")\
    .str.replace("(LAC)", "Chargers").str.replace("(NYJ)", "Jets").str.replace("(NYG)", "Giants")
    probs = pd.merge(left=probs,right=nfl_teams[["name","yahoo"]].rename(columns={"name":"team","yahoo":"pick"}),how="inner",on="team")
    probs.pick_prob = probs.pick_prob.astype(float)/100.0
    probs["win_prob"] = 0.5 - 0.035*probs.spread
    # 3.5% probability per point spread
    games = pd.merge(left=probs.iloc[::2],right=probs.iloc[1::2],how="inner",\
    on=["game_ind","game_date","still_to_play"],suffixes=("_fave","_dog"))
    return games

def load_picks(pick_html: str) -> pd.DataFrame:
    tempData = open(pick_html,"r")
    response = tempData.read()
    tempData.close()
    soup = BeautifulSoup(response, "html.parser")
    tables = soup.find_all("table", attrs={"class": "yspNflPickGroupPickTable yspNflPickGroupPickTablePadded"})
    actual = pd.concat(pd.read_html(StringIO(str(tables))), ignore_index=True)
    actual = actual.iloc[:,:-2].reset_index(drop=True)
    picks = pd.DataFrame(columns=["player","pick","fave","underdog","spread"])
    for ind in range(1,actual.shape[1]):
        picks = pd.concat([picks,actual.iloc[4:,[0,ind]].rename(columns={0:"player",ind:"pick"})],ignore_index=True)
        picks.fave = picks.fave.fillna(actual.iloc[0,ind])
        picks.underdog = picks.underdog.fillna(actual.iloc[2,ind])
        picks.spread = picks.spread.fillna(actual.iloc[1,ind])
    picks["points_bid"] = picks.pick.str.split(" \(").str[-1].str.split("\)").str[0]
    picks["pick"] = picks.pick.str.split(" \(").str[0]
    winners = soup.find_all("td", attrs={"class": "yspNflPickWin"})
    winners = [team.text for team in winners]
    picks.loc[picks.fave.isin(winners),"winner"] = picks.loc[picks.fave.isin(winners),"fave"]
    picks.loc[picks.underdog.isin(winners),"winner"] = picks.loc[picks.underdog.isin(winners),"underdog"]
    picks["finished"] = ~picks.winner.isnull()
    picks["points_won"] = picks.pick.isin(winners).astype(int)*picks.points_bid
    return picks

def simulate_picks(games: pd.DataFrame, picks: pd.DataFrame, num_sims: int = 1000, num_entries: int = 50) -> pd.DataFrame:
    sims = pd.concat(num_sims*num_entries*[games.loc[games.still_to_play]],ignore_index=True)
    sims['entry'] = sims.index%(games.still_to_play.sum()*num_entries)//games.still_to_play.sum()
    sims['num_sim'] = sims.index//(games.still_to_play.sum()*num_entries)
#     sims = pd.merge(left=sims,right=picks,how='left',left_on=["entry","team1_abbrev"],right_on=["entry","pick"])
#     sims = pd.merge(left=sims,right=picks,how='left',left_on=["entry","team2_abbrev"],right_on=["entry","pick"],suffixes=("","_2"))
#     sims.loc[~sims['pick_2'].isnull(),"pick"] = sims.loc[~sims['pick_2'].isnull(),'pick_2']
#     sims.loc[~sims['points_bid_2'].isnull(),"points_bid"] = sims.loc[~sims['points_bid_2'].isnull(),'points_bid_2']
#     del sims['pick_2'], sims['points_bid_2'], sims['player'], sims['player_2'], sims['points_won'], sims['points_won_2']
#     already_picked = sims.loc[~sims.pick.isnull()].reset_index(drop=True)
#     sims = sims.loc[sims.pick.isnull()].reset_index(drop=True)
#     sims['pick_sim'] = np.random.rand(sims.shape[0])
#     home_pick = sims.pick_sim < sims.pick_prob1
#     sims.loc[home_pick,'pick'] = sims.loc[home_pick,'team1_abbrev']
#     sims.loc[~home_pick,'pick'] = sims.loc[~home_pick,'team2_abbrev']
#     # sims.loc[home_pick,'pts_avg'] = 1.131*sims.loc[home_pick,'pick_pts1'] - 0.259
#     # sims.loc[~home_pick,'pts_avg'] = 1.131*sims.loc[~home_pick,'pick_pts2'] - 0.259
#     sims.loc[home_pick,'pts_avg'] = sims.loc[home_pick,'pick_pts1']
#     sims.loc[~home_pick,'pts_avg'] = sims.loc[~home_pick,'pick_pts2']
#     sims['rel_pts'] = sims['pts_avg']/games.shape[0]
#     # sims['pts_stdev_true'] = (-0.441*sims['rel_pts']**2.0 + 0.446*sims['rel_pts'] + 0.097)*games.shape[0]
#     sims['pts_stdev_true'] = (-0.5*sims['rel_pts']**2.0 + 0.519*sims['rel_pts'] + 0.08)*games.shape[0]
#     sims['pts_stdev'] = sims['pts_stdev_true']*1.09 # Simulation fudge factor
#     all_picks = pd.DataFrame({"entry":[val//games.shape[0] for val in range(games.shape[0]*num_entries)],\
#     "points_bid":[val%games.shape[0] + 1 for val in range(games.shape[0]*num_entries)]})
#     all_picks = pd.merge(left=all_picks,right=picks,how='left',on=['entry','points_bid'])
#     all_picks = all_picks.loc[all_picks.pick.isnull()]
#     # print(all_picks.groupby('entry').size().sort_values())
#     # print(picks.loc[picks.entry.isin([15,18,25,31,35,46,53,29]),['player','entry']].drop_duplicates())
#     if all_picks.entry.nunique() == num_entries:
#         sims['points_bid_sim'] = sims.apply(lambda x: np.random.normal(x['pts_avg'],x['pts_stdev']),axis=1)
#         sims = sims.sort_values(by=['num_sim','entry','points_bid_sim'],ascending=True)
#         sims['points_bid'] = all_picks.points_bid.tolist()*num_sims
#     # comparison = pd.merge(left=sims.groupby('pick').points_bid.mean().reset_index().rename(columns={"points_bid":"actual_avg"}),\
#     # right=sims.groupby('pick').points_bid.std().reset_index().rename(columns={"points_bid":"actual_stdev"}),how='inner',on='pick')
#     # comparison = pd.merge(left=comparison,right=sims.groupby('pick')[['pts_avg','pts_stdev','pts_stdev_true']].mean().reset_index(),how='inner',on='pick')
#     # print(comparison)
#     # print(comparison[['actual_avg','pts_avg','actual_stdev','pts_stdev','pts_stdev_true']].corr())
#     # print("St. Dev. Mean Squared Error: " + str(sum((comparison.actual_stdev - comparison.pts_stdev_true)**2.0)))
#     sims = pd.concat([sims,already_picked],ignore_index=True).sort_values(by=['num_sim','entry'],ascending=True)
    return sims

def some_function():
    """A sample function"""
    pass
