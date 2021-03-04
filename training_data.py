import numpy as np
import pandas as pd
import mutex_utils as mu
import download as dl


def process_data(year, league):
    filename = 'data/'+str(league)+'_'+str(year)+'_tdata.pkl'
    if mu.mutex_process(filename):
        scores_data = dl.download_scores(year, league)
        nb_scores = scores_data.shape[0]
        x_data = np.zeros(shape=(nb_scores, 5, 8))
        y_data = np.zeros(shape=(nb_scores, 1))
        x_data[:, :, :] = np.nan
        y_data[:, :] = np.nan
        curr_score = 0
        current_season = -1
        classement = pd.DataFrame(columns=["team", "points", "win", "draw", "lost", "matchs", "BP", "BC", "DB"])
        dict_teams = dict()
        for i in range(0, nb_scores):
            gameline = scores_data.iloc[i, :]
            if gameline.Season != current_season:
                if not classement.empty:
                    print(current_season)
                    print(i)
                    print(classement)
                classement.drop(classement.index, inplace=True)
                current_season = gameline.Season
                dict_teams.clear()
            hometeam = gameline.HomeTeam
            awayteam = gameline.AwayTeam
            # fill the NN inputs
            if ((hometeam in dict_teams) and (awayteam in dict_teams)
                    and classement.loc[classement.team == hometeam, "matchs"].values[0] > 4
                    and classement.loc[classement.team == awayteam, "matchs"].values[0] > 4):
                x_data[curr_score, :, 0] = dict_teams[hometeam]["BP"]
                x_data[curr_score, :, 1] = dict_teams[hometeam]["BC"]
                x_data[curr_score, :, 2] = dict_teams[hometeam]["cl"]
                x_data[curr_score, :, 3] = dict_teams[hometeam]["loc"]
                x_data[curr_score, :, 4] = dict_teams[awayteam]["BP"]
                x_data[curr_score, :, 5] = dict_teams[awayteam]["BC"]
                x_data[curr_score, :, 6] = dict_teams[awayteam]["cl"]
                x_data[curr_score, :, 7] = dict_teams[awayteam]["loc"]
                if gameline.FTHG > gameline.FTAG:
                    y_data[curr_score, 0] = 0
                elif gameline.FTHG == gameline.FTAG:
                    y_data[curr_score, 0] = 1
                else:
                    y_data[curr_score, 0] = 2
                curr_score = curr_score + 1
            # add team into the ranking if necesseray
            clhome = classement.loc[classement.team == hometeam, :]
            if clhome.empty:
                classement.loc[classement.shape[0]] = [hometeam, 0, 0, 0, 0, 0, 0, 0, 0]
            claway = classement.loc[classement.team == awayteam, :]
            if claway.empty:
                classement.loc[classement.shape[0]] = [awayteam, 0, 0, 0, 0, 0, 0, 0, 0]

            # update rank
            if gameline.FTHG > gameline.FTAG:
                classement.loc[classement.team == hometeam, "win"] = \
                    classement.loc[classement.team == hometeam, "win"] + 1
                classement.loc[classement.team == awayteam, "lost"] = \
                    classement.loc[classement.team == awayteam, "lost"] + 1
            elif gameline.FTHG == gameline.FTAG:
                classement.loc[classement.team == hometeam, "draw"] = \
                    classement.loc[classement.team == hometeam, "draw"] + 1
                classement.loc[classement.team == awayteam, "draw"] = \
                    classement.loc[classement.team == awayteam, "draw"] + 1
            else:
                classement.loc[classement.team == hometeam, "lost"] = \
                    classement.loc[classement.team == hometeam, "lost"] + 1
                classement.loc[classement.team == awayteam, "win"] = \
                    classement.loc[classement.team == awayteam, "win"] + 1
            classement.loc[classement.team == hometeam, "BP"] = \
                classement.loc[classement.team == hometeam, "BP"] + gameline.FTHG
            classement.loc[classement.team == hometeam, "BC"] = \
                classement.loc[classement.team == hometeam, "BC"] + gameline.FTAG
            classement.loc[classement.team == awayteam, "BP"] = \
                classement.loc[classement.team == awayteam, "BP"] + gameline.FTAG
            classement.loc[classement.team == awayteam, "BC"] = \
                classement.loc[classement.team == awayteam, "BC"] + gameline.FTHG

            if current_season < 1994:
                classement.points = classement.win * 2 + classement.draw
            else:
                classement.points = classement.win * 3 + classement.draw
            classement.matchs = classement.win + classement.draw + classement.lost
            classement.DB = classement.BP - classement.BC
            classement = classement.sort_values(by=["points", "DB"], ascending=False)
            classement = classement.reset_index(drop=True)

            # fill hometeam dict
            clhome_after = classement.loc[classement.team == hometeam, :]
            claway_after = classement.loc[classement.team == awayteam, :]
            if hometeam not in dict_teams:
                dict_teams[hometeam] = {'BP': np.empty(5), 'BC': np.empty(5), 'cl': np.empty(5), 'loc': np.empty(5)}
                dict_teams[hometeam]['BP'][:] = np.nan
                dict_teams[hometeam]['BC'][:] = np.nan
                dict_teams[hometeam]['cl'][:] = np.nan
                dict_teams[hometeam]['loc'][:] = np.nan
            nanv = np.where(np.isnan(dict_teams[hometeam]['BP']))[0]
            if len(nanv) == 0:
                dict_teams[hometeam]['BP'] = np.concatenate(([gameline.FTHG], dict_teams[hometeam]['BP'][:-1]))
                dict_teams[hometeam]['BC'] = np.concatenate(([gameline.FTAG], dict_teams[hometeam]['BC'][:-1]))
                dict_teams[hometeam]['cl'] = \
                    np.concatenate(([clhome_after.index[0] + 1], dict_teams[hometeam]['cl'][:-1]))
                dict_teams[hometeam]['loc'] = np.concatenate(([0], dict_teams[hometeam]['loc'][:-1]))
            else:
                dict_teams[hometeam]['BP'][len(nanv) - 1] = gameline.FTHG
                dict_teams[hometeam]['BC'][len(nanv) - 1] = gameline.FTAG
                dict_teams[hometeam]['cl'][len(nanv) - 1] = clhome_after.index[0] + 1
                dict_teams[hometeam]['loc'][len(nanv) - 1] = 0

            # fill awayteam dict
            if awayteam not in dict_teams:
                dict_teams[awayteam] = {'BP': np.empty(5), 'BC': np.empty(5), 'cl': np.empty(5), 'loc': np.empty(5)}
                dict_teams[awayteam]['BP'][:] = np.nan
                dict_teams[awayteam]['BC'][:] = np.nan
                dict_teams[awayteam]['cl'][:] = np.nan
                dict_teams[awayteam]['loc'][:] = np.nan
            nanv = np.where(np.isnan(dict_teams[awayteam]['BP']))[0]
            if len(nanv) == 0:
                dict_teams[awayteam]['BP'] = np.concatenate(([gameline.FTAG], dict_teams[awayteam]['BP'][:-1]))
                dict_teams[awayteam]['BC'] = np.concatenate(([gameline.FTHG], dict_teams[awayteam]['BC'][:-1]))
                dict_teams[awayteam]['cl'] = \
                    np.concatenate(([claway_after.index[0] + 1], dict_teams[awayteam]['cl'][:-1]))
                dict_teams[awayteam]['loc'] = np.concatenate(([1], dict_teams[awayteam]['loc'][:-1]))
            else:
                dict_teams[awayteam]['BP'][len(nanv) - 1] = gameline.FTAG
                dict_teams[awayteam]['BC'][len(nanv) - 1] = gameline.FTHG
                dict_teams[awayteam]['cl'][len(nanv) - 1] = claway_after.index[0] + 1
                dict_teams[awayteam]['loc'][len(nanv) - 1] = 1
        index_data = np.where(np.isnan(x_data[:, 0, 0]))[0][0]
        x_data = x_data[range(index_data), :, :]
        y_data = y_data[range(index_data), :]
        tdata = {'x_data': x_data, 'y_data': y_data,
                 'teams_state': dict_teams, 'ranking_state': classement}
        mu.mutex_save(tdata, filename)
    processed_data = mu.mutex_load(filename)
    return processed_data


def aggr_data(years, leagues):
    x_aggr_data = np.empty([len(years)*len(leagues)*500, 5, 8])
    y_aggr_data = np.empty([len(years)*len(leagues)*500, 1])
    x_aggr_data[:, :, :] = np.nan
    y_aggr_data[:, :] = np.nan
    curr_d = 0
    for ye in range(np.min(years), np.max(years)+1):
        for le in leagues:
            loc_data = process_data(ye, le)
            nb_data = loc_data['x_data'].shape[0]
            x_aggr_data[range(curr_d, curr_d + nb_data), :, :] = loc_data['x_data']
            y_aggr_data[range(curr_d, curr_d + nb_data), :] = loc_data['y_data']
            curr_d = curr_d + nb_data
    index_data = np.where(np.isnan(x_aggr_data[:, 0, 0]))[0][0]
    x_aggr_data = x_aggr_data[range(index_data), :, :]
    y_aggr_data = y_aggr_data[range(index_data), :]
    tdata_aggr = {'x_data': x_aggr_data, 'y_data': y_aggr_data}
    return tdata_aggr


def get_training_data():
    train_data = aggr_data(years=range(1993, 2020), leagues=['F1', 'E0'])
    test_data = aggr_data(years=range(2020, 2021), leagues=['F1'])
    tdata = {'x_train': train_data['x_data'], 'y_train': train_data['y_data'],
             'x_test': test_data['x_data'], 'y_test': test_data['y_data']}
    return tdata


# tt = process_data(year=2020, league='F1')
# bb = aggr_data(years=[1993, 1995], leagues=['F1', 'E0'])
# td = get_training_data()
