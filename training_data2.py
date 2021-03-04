import numpy as np
import pandas as pd
import mutex_utils as mu
import download as dl


def process_data(year, league):
    filename = 'data/'+str(league)+'_'+str(year)+'_tdata2.pkl'
    #print(filename)
    if mu.mutex_process(filename):
        scores_data = dl.download_scores(year, league)
        nb_scores = scores_data.shape[0]
        x_data = np.zeros(shape=(nb_scores, 20, 8+15))
        y_data = np.zeros(shape=(nb_scores, 1))
        x_data[:, :, :] = np.nan
        y_data[:, :] = np.nan
        curr_score = 0
        current_season = -1
        classement = pd.DataFrame(columns=["team", "points", "win", "draw", "lost", "matchs", "BP", "BC", "DB"]+
                                          [str("m")+str(k)+str(t) for k in range(1, 6) for t in ["team", "loc", "res"]])
        for i in range(0, nb_scores):
            gameline = scores_data.iloc[i, :]
            #print(gameline.Date)
            if gameline.Season != current_season:
                if not classement.empty:
                    print(current_season)
                    print(i)
                    print(classement)
                classement.drop(classement.index, inplace=True)
                current_season = gameline.Season
            hometeam = gameline.HomeTeam
            awayteam = gameline.AwayTeam
            # fill the NN inputs
            if ((hometeam in classement['team'].unique()) and (awayteam in classement['team'].unique())
                    and not classement.isnull().values.any()):
                cl = pd.concat([classement.loc[classement.team == hometeam, :],
                                classement.loc[classement.team == awayteam, :],
                                classement.loc[np.logical_and(classement.team != hometeam, classement.team != awayteam), :]])
                x_data[curr_score, :, :] = cl.drop('team', axis=1)
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
                classement.loc[classement.shape[0]] = [hometeam] + \
                                                      [0 for gg in range(0, 8)] + [np.nan for gg in range(0, 15)]
            claway = classement.loc[classement.team == awayteam, :]
            if claway.empty:
                classement.loc[classement.shape[0]] = [awayteam] + \
                                                      [0 for gg in range(0, 8)] + [np.nan for gg in range(0, 15)]

            # update rank
            home_res = -1
            away_res = -1
            if gameline.FTHG > gameline.FTAG:
                home_res = 0
                away_res = 2
                classement.loc[classement.team == hometeam, "win"] = \
                    classement.loc[classement.team == hometeam, "win"] + 1
                classement.loc[classement.team == awayteam, "lost"] = \
                    classement.loc[classement.team == awayteam, "lost"] + 1
            elif gameline.FTHG == gameline.FTAG:
                home_res = 1
                away_res = 1
                classement.loc[classement.team == hometeam, "draw"] = \
                    classement.loc[classement.team == hometeam, "draw"] + 1
                classement.loc[classement.team == awayteam, "draw"] = \
                    classement.loc[classement.team == awayteam, "draw"] + 1
            else:
                home_res = 2
                away_res = 0
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

            #add match historic
            home_igame = -1
            away_igame = -1
            for k in reversed(range(1, 6)):
                if np.isnan(classement.loc[classement.team == hometeam,:]["m" + str(k) + "team"].iloc[0]) and home_igame == -1:
                    home_igame = k
                if np.isnan(classement.loc[classement.team == awayteam,:]["m" + str(k) + "team"].iloc[0]) and away_igame == -1:
                    away_igame = k
            if home_igame == -1:
                for k in reversed(range(2, 6)):
                    classement.loc[classement.team == hometeam, ["m" + str(k) + "team"]] = \
                        classement.loc[classement.team == hometeam, :]["m" + str(k-1) + "team"]
                    classement.loc[classement.team == hometeam, ["m" + str(k) + "loc"]] = \
                        classement.loc[classement.team == hometeam, :]["m" + str(k - 1) + "loc"]
                    classement.loc[classement.team == hometeam, ["m" + str(k) + "res"]] = \
                        classement.loc[classement.team == hometeam, :]["m" + str(k - 1) + "res"]
                home_igame = 1
            if away_igame == -1:
                for k in reversed(range(2, 6)):
                    classement.loc[classement.team == awayteam, ["m" + str(k) + "team"]] = \
                        classement.loc[classement.team == awayteam, :]["m" + str(k - 1) + "team"]
                    classement.loc[classement.team == awayteam, ["m" + str(k) + "loc"]] = \
                        classement.loc[classement.team == awayteam, :]["m" + str(k - 1) + "loc"]
                    classement.loc[classement.team == awayteam, ["m" + str(k) + "res"]] = \
                        classement.loc[classement.team == awayteam, :]["m" + str(k - 1) + "res"]
                away_igame = 1
            classement.loc[classement.team == hometeam, ["m" + str(home_igame) + u for u in ["team", "loc", "res"]]] = \
                [classement.loc[classement.team == awayteam,:].index[0]+1, 0, home_res]
            classement.loc[classement.team == awayteam, ["m" + str(away_igame) + u for u in ["team", "loc", "res"]]] = \
                [classement.loc[classement.team == hometeam,:].index[0]+1, 1, away_res]

            #sort ranking
            classement = classement.sort_values(by=["points", "DB"], ascending=False)
            classement = classement.reset_index(drop=True)

        index_data = np.where(np.isnan(x_data[:, 0, 0]))[0][0]
        x_data = x_data[range(index_data), :, :]
        y_data = y_data[range(index_data), :]
        tdata = {'x_data': x_data, 'y_data': y_data,
                 'ranking_state': classement}
        mu.mutex_save(tdata, filename)
    processed_data = mu.mutex_load(filename)
    return processed_data

def aggr_data(dict_leagues):
    nb_seasons = sum(len(v) for v in dict_leagues.values())
    x_aggr_data = np.empty([nb_seasons*500, 20, 8+15])
    y_aggr_data = np.empty([nb_seasons*500, 1])
    x_aggr_data[:, :, :] = np.nan
    y_aggr_data[:, :] = np.nan
    curr_d = 0
    for le in dict_leagues.keys():
        for ye in dict_leagues[le]:
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
    train_data = aggr_data({'F1': range(2002, 2020), 'E0': range(1995, 2021), 'SP1': range(2000, 2021)})
    test_data = aggr_data({'F1': range(2020, 2021)})
    tdata = {'x_train': train_data['x_data'], 'y_train': train_data['y_data'],
             'x_test': test_data['x_data'], 'y_test': test_data['y_data']}
    return tdata


# tt = process_data(year=2020, league='F1')
# bb = aggr_data(years=[1993, 1995], leagues=['F1', 'E0'])
# td = get_training_data()
