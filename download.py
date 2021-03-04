import os
import pandas as pd
import mutex_utils as mu


def download_scores(year, league):
    filename = 'data/'+str(league)+'_'+str(year)+'_scores.pkl'
    if mu.mutex_process(filename):
        data_scores = pd.DataFrame()
        seas_y1 = str(year)[-2:]
        seas_y2 = str(year + 1)[-2:]
        url = "http://www.football-data.co.uk/mmz4281/" + str(seas_y1) + str(seas_y2) + "/data.zip"
        print(url)
        os.system("wget -c --read-timeout=5 --tries=0 --directory-prefix tmpdata/ " + url)
        os.system("unzip -d tmpdata tmpdata/data.zip ")
        csvfile = "tmpdata/"+str(league)+".csv"
        tt = pd.read_csv(csvfile, usecols=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        tt = tt.loc[:, ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
        tt.dropna(subset=["Date"], inplace=True)
        tt["Season"] = year
        if data_scores.empty:
            data_scores = tt
        else:
            data_scores = data_scores.append(tt)
        os.system("rm tmpdata/data.zip ; rm tmpdata/*csv")
        data_scores[["FTHG", "FTAG"]] = data_scores[["FTHG", "FTAG"]].astype(int)
        mu.mutex_save(data_scores, filename)
    scores_year_league = mu.mutex_load(filename)
    return scores_year_league

#ss = download_scores(2020, 'F1')