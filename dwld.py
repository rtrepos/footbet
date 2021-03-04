import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

all_data = pd.DataFrame()

for season in range(1993, 2021):
    seas_y1 = str(season)[-2:]
    seas_y2 = str(season + 1)[-2:]
    print(seas_y1)
    print(seas_y2)
    url = "http://www.football-data.co.uk/mmz4281/" + str(seas_y1) + str(seas_y2) + "/data.zip"
    print(url)
    os.system("wget -c --read-timeout=5 --tries=0 --directory-prefix tmpdata/ " + url)
    os.system("unzip -d tmpdata tmpdata/data.zip ")
    tt = pd.read_csv("tmpdata/F1.csv", usecols=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    tt = tt.loc[:, ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
    tt.dropna(subset=["Date"], inplace=True)
    tt["Season"] = season
    if all_data.empty:
        all_data = tt
    else:
        all_data = all_data.append(tt)
    os.system("rm tmpdata/data.zip ; rm tmpdata/*csv")

all_data[["FTHG", "FTAG"]] = all_data[["FTHG", "FTAG"]].astype(int)

scores = np.zeros((np.max(all_data.FTHG) + 1, np.max(all_data.FTAG) + 1))

for i in range(0, np.max(all_data.FTHG) + 1):
    for j in range(0, np.max(all_data.FTAG) + 1):
        scores[j, i] = np.log(1 + len(all_data[(all_data.FTHG == i) & (all_data.FTAG == j)].index))

fig, ax = plt.subplots()
im = ax.imshow(scores)
for i in range(np.max(all_data.FTHG) + 1):
    for j in range(np.max(all_data.FTAG) + 1):
        text = ax.text(j, i, "{:.2f}".format(scores[i, j]),
                       ha="center", va="center", color="w")
ax.set_title("Log nb scores in French League 1")
fig.tight_layout()
plt.show()

all_data.head(5)

nb_all = all_data.shape[0]
#nb_all = 60
nb_train = sum(all_data.Season < 2020)
nb_test = sum(all_data.Season >= 2020)
x_train = np.zeros(shape=(nb_train, 5, 8))
x_train[:, :, :] = np.nan
y_train = np.zeros(shape=(nb_train, 1))
y_train[:, :] = np.nan
x_test = np.zeros(shape=(nb_test, 5, 8))
x_test[:, :, :] = np.nan
y_test = np.zeros(shape=(nb_test, 1))
y_test[:, :] = np.nan

curr_train = 0
curr_test = 0

classement = pd.DataFrame(columns=["team", "points", "win", "draw", "lost", "matchs", "BP", "BC", "DB"])
current_season = -1
dict_teams = dict()
for i in range(0, nb_all):
    # print(i)
    gameline = all_data.iloc[i, :]
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
        if i < nb_train:
            x_train[curr_train, :, 0] = dict_teams[hometeam]["BP"]
            x_train[curr_train, :, 1] = dict_teams[hometeam]["BC"]
            x_train[curr_train, :, 2] = dict_teams[hometeam]["cl"]
            x_train[curr_train, :, 3] = dict_teams[hometeam]["loc"]
            x_train[curr_train, :, 4] = dict_teams[awayteam]["BP"]
            x_train[curr_train, :, 5] = dict_teams[awayteam]["BC"]
            x_train[curr_train, :, 6] = dict_teams[awayteam]["cl"]
            x_train[curr_train, :, 7] = dict_teams[awayteam]["loc"]
            if gameline.FTHG > gameline.FTAG:
                y_train[curr_train, 0] = 0
            elif gameline.FTHG == gameline.FTAG:
                y_train[curr_train, 0] = 1
            else:
                y_train[curr_train, 0] = 2
            curr_train = curr_train + 1
        else:
            x_test[curr_test, :, 0] = dict_teams[hometeam]["BP"]
            x_test[curr_test, :, 1] = dict_teams[hometeam]["BC"]
            x_test[curr_test, :, 2] = dict_teams[hometeam]["cl"]
            x_test[curr_test, :, 3] = dict_teams[hometeam]["loc"]
            x_test[curr_test, :, 4] = dict_teams[awayteam]["BP"]
            x_test[curr_test, :, 5] = dict_teams[awayteam]["BC"]
            x_test[curr_test, :, 6] = dict_teams[awayteam]["cl"]
            x_test[curr_test, :, 7] = dict_teams[awayteam]["loc"]
            if gameline.FTHG > gameline.FTAG:
                y_test[curr_test, 0] = 0
            elif gameline.FTHG == gameline.FTAG:
                y_test[curr_test, 0] = 1
            else:
                y_test[curr_test, 0] = 2
            curr_test = curr_test + 1

    # add team into the ranking if necesseray
    clhome = classement.loc[classement.team == hometeam, :]
    if clhome.empty:
        classement.loc[classement.shape[0]] = [hometeam, 0, 0, 0, 0, 0, 0, 0, 0]
    claway = classement.loc[classement.team == awayteam, :]
    if claway.empty:
        classement.loc[classement.shape[0]] = [awayteam, 0, 0, 0, 0, 0, 0, 0, 0]

    # update rank
    if gameline.FTHG > gameline.FTAG:
        classement.loc[classement.team == hometeam, "win"] = classement.loc[classement.team == hometeam, "win"] + 1
        classement.loc[classement.team == awayteam, "lost"] = classement.loc[classement.team == awayteam, "lost"] + 1
    elif gameline.FTHG == gameline.FTAG:
        classement.loc[classement.team == hometeam, "draw"] = classement.loc[classement.team == hometeam, "draw"] + 1
        classement.loc[classement.team == awayteam, "draw"] = classement.loc[classement.team == awayteam, "draw"] + 1
    else:
        classement.loc[classement.team == hometeam, "lost"] = classement.loc[classement.team == hometeam, "lost"] + 1
        classement.loc[classement.team == awayteam, "win"] = classement.loc[classement.team == awayteam, "win"] + 1
    classement.loc[classement.team == hometeam, "BP"] = classement.loc[
                                                            classement.team == hometeam, "BP"] + gameline.FTHG
    classement.loc[classement.team == hometeam, "BC"] = classement.loc[
                                                            classement.team == hometeam, "BC"] + gameline.FTAG
    classement.loc[classement.team == awayteam, "BP"] = classement.loc[
                                                            classement.team == awayteam, "BP"] + gameline.FTAG
    classement.loc[classement.team == awayteam, "BC"] = classement.loc[
                                                            classement.team == awayteam, "BC"] + gameline.FTHG

    if current_season < 1994:
        classement.points = classement.win * 2 + classement.draw
    else:
        classement.points = classement.win * 3 + classement.draw
    classement.matchs = classement.win + classement.draw + classement.lost
    classement.DB = classement.BP - classement.BC
    classement = classement.sort_values(by=["points", "DB"], ascending=False)
    classement = classement.reset_index(drop=True)

    # fill hometeam dict
    clhomeAfter = classement.loc[classement.team == hometeam, :]
    clawayAfter = classement.loc[classement.team == awayteam, :]
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
        dict_teams[hometeam]['cl'] = np.concatenate(([clhomeAfter.index[0] + 1], dict_teams[hometeam]['cl'][:-1]))
        dict_teams[hometeam]['loc'] = np.concatenate(([0], dict_teams[hometeam]['loc'][:-1]))
    else:
        dict_teams[hometeam]['BP'][len(nanv) - 1] = gameline.FTHG
        dict_teams[hometeam]['BC'][len(nanv) - 1] = gameline.FTAG
        dict_teams[hometeam]['cl'][len(nanv) - 1] = clhomeAfter.index[0] + 1
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
        dict_teams[awayteam]['cl'] = np.concatenate(([clawayAfter.index[0] + 1], dict_teams[awayteam]['cl'][:-1]))
        dict_teams[awayteam]['loc'] = np.concatenate(([1], dict_teams[awayteam]['loc'][:-1]))
    else:
        dict_teams[awayteam]['BP'][len(nanv) - 1] = gameline.FTAG
        dict_teams[awayteam]['BC'][len(nanv) - 1] = gameline.FTHG
        dict_teams[awayteam]['cl'][len(nanv) - 1] = clawayAfter.index[0] + 1
        dict_teams[awayteam]['loc'][len(nanv) - 1] = 1

index_data_train = np.where(np.isnan(x_train[:, 0, 0]))[0][0]
x_train = x_train[range(index_data_train), :, :]
y_train = y_train[range(index_data_train), :]
index_data_test = np.where(np.isnan(x_test[:, 0, 0]))[0][0]
x_test = x_test[range(index_data_test), :, :]
y_test = y_test[range(index_data_test), :]


import tensorflow as tf
import tensorflow.keras as tfk

model_y = tfk.Sequential([
    tfk.layers.InputLayer(batch_input_shape=(None, 5, 8), name='in'),
    tfk.layers.LSTM(units=5, name='lstm'),
    tfk.layers.Dense(3, activation=tf.nn.softmax, name='dense')
])
model_y.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model_y.fit(
    x_train,
    y_train,
    batch_size=24,
    epochs=100)

model_y.evaluate(x_train, y_train, verbose=2)
predict_train = tf.math.argmax(model_y.predict(x_train), 1)
conf_train = tf.math.confusion_matrix(y_train, predict_train)
(conf_train[0,0]+conf_train[1,1]+conf_train[2,2])/np.sum(conf_train)
conf_train

model_y.evaluate(x_test, y_test, verbose=2)
predict_test = tf.math.argmax(model_y.predict(x_test), 1)
conf_test = tf.math.confusion_matrix(y_test, predict_test)
(conf_test[0,0]+conf_test[1,1]+conf_test[2,2])/np.sum(conf_test)


home_teams = ["Bordeaux", "Metz","Reims","Rennes","Strasbourg","Dijon","Lens","Monaco","Paris SG","St Etienne"]
away_teams = ["Lille","Montpellier","Angers","Lorient","Brest","Lyon","Marseille","Nice","Nimes","Nantes"]
x_predict = np.zeros(shape=(1, 5, 8))
for i in range(len(home_teams)):
    home_team = home_teams[i]
    away_team = away_teams[i]
    x_predict[0, :, 0] = dict_teams[home_team]["BP"]
    x_predict[0, :, 1] = dict_teams[home_team]["BC"]
    x_predict[0, :, 2] = dict_teams[home_team]["cl"]
    x_predict[0, :, 3] = dict_teams[home_team]["loc"]
    x_predict[0, :, 4] = dict_teams[away_team]["BP"]
    x_predict[0, :, 5] = dict_teams[away_team]["BC"]
    x_predict[0, :, 6] = dict_teams[away_team]["cl"]
    x_predict[0, :, 7] = dict_teams[away_team]["loc"]
    prediction = tf.math.argmax(model_y.predict(x_predict), 1)
    print(home_team + " vs. " + away_team +" : "+ str(float(prediction[0])))
