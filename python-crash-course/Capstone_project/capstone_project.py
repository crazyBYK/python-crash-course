import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

all_sites = pd.read_csv("all_sites_scores.csv")
fandango = pd.read_csv("fandango_scrape.csv")
pd.set_option("display.max_columns", 3000)
pd.options.display.width = None
pd.options.display.max_columns = None


def part_one():
    # print(fandango)

    # print(fandango.info())
    # print(fandango.describe())
    # sns.scatterplot(data=fandango, x="RATING", y="VOTES")
    # plt.savefig("part_one_scatter.jpg")
    # correlation = fandango.corr(numeric_only=True)
    # print(correlation)
    # fandango['YEAR'] = fandango['FILM'].apply(get_years)
    # print(fandango)
    # series_year = fandango.groupby('YEAR')['YEAR'].count().sort_values(ascending=False)
    # dataframe_year = series_year.to_frame(name='COUNT')
    # sns.barplot(data=dataframe_year, x='YEAR', y='COUNT')
    # plt.show()

    # sns.barplot(data=test_185, x='YEAR', y='count')
    # plt.show()

    # DOTO: ini 181
    fandango["YEAR"] = fandango["FILM"].apply(get_years)
    # print((fandango['VOTES'] == 0).sum())
    # none_zero_votes = (fandango[fandango['VOTES'] != 0])
    none_zero_votes = fandango[fandango["VOTES"] != 0]
    # print(none_zero_votes)
    none_zero_votes["STARS_DIFF"] = none_zero_votes["STARS"] - none_zero_votes["RATING"]
    none_zero_votes["STARS_DIFF"] = none_zero_votes["STARS_DIFF"].round(2)

    # plt.figure(figsize=(12,4), dpi=150)
    # sns.countplot(data=none_zero_votes, x="STARS_DIFF", palette='magma')
    # plt.show()

    # print(none_zero_votes[none_zero_votes['STARS_DIFF'] == 1.0])

    # plt.figure(figsize=(10,4), dpi=150)
    # sns.kdeplot(data=none_zero_votes, x='RATING', fill=True, clip=[0,5], label='True Rating')
    # sns.kdeplot(data=none_zero_votes, x='STARS', fill=True, clip=[0,5], label='Stars Displayed')

    # plt.legend(loc=(1.05, 0.5))
    # plt.show()

    # print(all_sites.head())
    # print(all_sites.info())
    # print(all_sites.describe())
    # print(all_sites.corr(numeric_only=True))

    # sns.scatterplot(data=all_sites, x='RottenTomatoes', y='RottenTomatoes_User')
    # plt.xlim(0, 100)
    # plt.ylim(0, 100)
    # plt.show()

    all_sites["Rotten_Diff"] = (
        all_sites["RottenTomatoes"] - all_sites["RottenTomatoes_User"]
    )
    all_sites["Rotten_Diff"] = all_sites["Rotten_Diff"].round(2)
    # print(all_sites.head())

    # print(all_sites['Rotten_Diff'].apply(abs).mean())
    # plt.figure(figsize=(10,4), dpi=200)
    # sns.histplot(data=all_sites, x='Rotten_Diff', kde=True, bins=25)
    # plt.title('RT Critics Score minus RT User Score')
    # plt.show()

    # print(all_sites)

    # plt.figure(figsize=(10,4), dpi=150)
    # sns.histplot(data=all_sites['Rotten_Diff'].apply(abs), bins=25, kde=True )
    # plt.show()

    # print(all_sites.nsmallest(5, 'Rotten_Diff')[['FILM', 'Rotten_Diff']])
    # print(all_sites.nlargest(5, 'Rotten_Diff')[['FILM','Rotten_Diff']])
    # print(all_sites)

    # plt.figure(figsize=(10,4), dpi=200)
    # sns.scatterplot(data=all_sites, x='Metacritic', y='Metacritic_User')
    # plt.xlim(0, 100)
    # plt.ylim(0, 10)
    # plt.show()

    # plt.figure(figsize=(10, 4), dpi=200)
    # sns.scatterplot(data=all_sites, x='Metacritic_user_vote_count', y='IMDB_user_vote_count')
    # plt.show()

    # print(all_sites.nlargest(1, 'IMDB_user_vote_count'))
    # print(all_sites.nlargest(1, 'Metacritic_user_vote_count'))
    # result = pd.concat([fandango, all_sites])
    # print(result.head())

    result = pd.merge(fandango, all_sites, on=["FILM"], how="inner")

    # result['RT_Norm'] = np.round(result['RottenTomatoes'] / 20, 1)
    result["RT_Norm"] = result["RottenTomatoes"].apply(get_normal)
    result["RTU_Norm"] = result["RottenTomatoes_User"].apply(get_normal)
    result["Meta_Norm"] = result["Metacritic"].apply(get_normal)
    result["Meta_U_Norm"] = result["Metacritic_User"].apply(get_normal)
    result["IMDB_Norm"] = result["IMDB"].apply(get_normal)
    # print(result.head())

    # norm_score = result[['STARS', 'RATING', 'RT_Norm', 'RTU_Norm', 'Meta_Norm', 'Meta_U_Norm', 'IMDB_Norm']]
    norm_score = result[
        [
            "STARS",
            "RATING",
            "RT_Norm",
            "RTU_Norm",
            "Meta_Norm",
            "Meta_U_Norm",
            "IMDB_Norm",
            "FILM",
        ]
    ]
    # print(norm_score)
    #

    # plt.figure(figsize=(12,4), dpi=200)
    # sns.kdeplot(data=result, x='STARS', fill=True, label='STARS', clip=[0, 5])
    # sns.kdeplot(data=result, x='RATING', fill=True, label='RATING', clip=[0, 5])
    # sns.kdeplot(data=result, x='RI_Norm', fill=True, label='RT_Norm',clip=[0,5])
    # sns.kdeplot(data=result, x='RTU_Norm', fill=True, label='RTU_Norm', clip=[0, 5])
    # sns.kdeplot(data=result, x='Meta_Norm', fill=True, label='Meta_Norm', clip=[0, 5])
    # sns.kdeplot(data=result, x='Meta_U_Norm', fill=True, label='Meta_U_Norm', clip=[0, 5])
    # sns.kdeplot(data=result, x='IMDB_Norm', fill=True, label='IMDB_Norm', clip=[0, 5])
    # plt.legend(loc='upper left')
    # plt.show()

    # plt.figure(figsize=(10,4), dpi=200)
    # sns.kdeplot(data=result, x='RI_Norm', fill=True, label='RT_Norm', clip=[0, 5])
    # sns.kdeplot(data=result, x='STARS', fill=True, label='STARS', clip=[0, 5])
    # plt.legend(loc='upper left')
    # plt.show()
    # fig, ax = plt.subplots(figsize=(15,6), dpi=150)
    # plt.figure(figsize=(10, 4), dpi=200)
    # sns.kdeplot(data=norm_score, clip=[0, 5], fill=True, palette='Set1')
    # move_legend(ax, 'upper left')
    # plt.legend(loc='upper left')
    # plt.show()

    # plt.figure(figsize=(15,6), dpi=200)
    # plt.subplots(figsize=(15,6), dpi=150)
    # sns.histplot(data=norm_score, bins=50)
    # plt.show()

    # sns.clustermap(data=norm_score, col_cluster=False)
    # plt.show()

    # print(norm_score.nsmallest(10, 'RT_Norm'))
    # worst_10 = norm_score.nsmallest(10, 'RT_Norm')
    # sns.kdeplot(data=worst_10, fill=True)
    # plt.show()

    # plt.figure(figsize=(15,6), dpi=150)
    # worst_films = norm_score.nsmallest(10, 'RT_Norm').drop('FILM',axis=1)
    # sns.kdeplot(data=worst_films, clip=[0, 5], fill=True, palette='Set1')
    # plt.title("Rating for RT Critic's 10 Worst Reviewed Films")
    # plt.show()

    print(norm_score.iloc[25])

    pass


def part_solution():
    print(fandango.info())
    print(fandango.describe())
    fandango.corr(numeric_only=True)
    fandango["YEAR"] = fandango["FILM"].apply(lambda title: title.split("(")[-1])
    fandango["YEAR"].value_counts()
    sns.countplot(data=fandango, x="YEAR")
    fandango.nlargest(10, "vOTES")
    no_votes = fandango["VOTES"] == 0
    no_votes.sum()
    len(fandango[fandango["VOTES"] == 0])
    fan_reviewed = fandango[fandango["VOTES"] > 0]
    sns.kdeplot(
        data=fan_reviewed, x="RATING", fill=True, clip=[0, 5], label="True Rating"
    )
    sns.kdeplot(
        data=fan_reviewed, y="STARS", fill=True, clip=[0, 5], label="Stars Displayed"
    )


def move_legend(ax, new_loc, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    label = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, label, loc=new_loc, title=title, **kws)


def get_years(film):
    year = film.split()[-1].replace("(", "").replace(")", "")
    return year


def get_normal(score):
    # print(type(score))
    if type(score) == int:
        return np.round(score / 20, 1)
    elif type(score) == float:
        return np.round(score / 2, 1)


def part_two():
    pass


def part_three():
    pass


def test():
    s = pd.Series(["a", "b", "c"], name="vals")
    print(s)


def part_three_solution():
    all_sites = pd.read_csv("all_sites.csv")
    all_sites.head()
    all_sites.info()
    all_sites.describe()
    sns.scatterplot(data=all_sites, x="RottenTomatoes", y="RottenTomatoes_User")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    all_sites["Rotten_Diff"] = (
        all_sites["RottenTomatoes"] - all_sites["RottenTomatoes_User"]
    )
    all_sites["Rotten_Diff"].apply(abs).mean()

    plt.figure(figsize=(10, 4), dpi=200)
    sns.histplot(data=all_sites, x="Rotten_Diff", kde=True, bins=25)

    sns.histplot(x=all_sites["Rotten_Diff"].apply(abs), bins=25, kde=True)

    all_sites.nsmallest(5, "Rotten_Diff")["FILM"]

    all_sites.nlargest(5, "Rotten_Diff")["FILM"]

    plt.figure(figsize=(10, 4), dpi=200)
    sns.scatterplot(data=all_sites, x="Metacritic", y="Metacritic_User", bins=25)
    plt.xlim(0, 100)
    plt.ylim(0, 10)
    all_sites.nlargest(1, "IMDB_user_vote_count")
    all_sites.nlargest(1, "Metacritic_user_vote_count")


def part_four_solution():
    df = pd.merge(fandango, all_sites, on="FILM", how="inner")
    df.info()
    print(df.describe().transpose()["max"])

    df["RT_Norm"] = np.round(df["RottenTomatoes"] / 20, 1)
    df["RTU_Norm"] = np.round(df["RottenTomatoes_User"] / 20, 1)
    df["Meta_Norm"] = np.round(df["Metacritic"] / 20, 1)
    df["Meta_U_Norm"] = np.round(df["Metacritic_User"] / 2, 1)
    df["IMDB_Norm"] = np.round(df["IMDB"] / 20, 1)

    print(df.columns())

    norm_scores = df[
        [
            "STARS",
            "RATING",
            "RT_Norm",
            "RTU_Norm",
            "Meta_Norm",
            "Meta_U_Norm",
            "IMDB_Norm",
        ]
    ]

    plt.figure(figsize=(15, 6), dpi=150)
    sns.kdeplot(data=norm_scores, fill=True, clip=[0, 5], palette="Set1")
    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    sns.kdeplot(
        data=norm_scores[["RT_Norm", "STARS"]],
        clip=[0, 5],
        fill=True,
        palette="Set1",
        ax=ax,
    )

    sns.histplot(data=norm_scores, bins=50)

    norm_films = df[
        [
            "FILM",
            "STARS",
            "RATING",
            "RT_Norm",
            "RTU_Norm",
            "Meta_Norm",
            "Meta_U_Norm",
            "IMDB_Norm",
        ]
    ]


if __name__ == "__main__":
    # test()
    # part_one()
    # part_two().
    # part_three()
    # part_solution()
    # part_three_solution()
    part_four_solution()
