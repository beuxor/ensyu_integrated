import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# データの準備
data_file = "./../titanic/train.csv"
df = pd.read_csv(data_file)


def plot(chart_config, title = "title", ctype = "buble", is_show = False):
    # プロットの設定
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    if ctype == "bar":
        # bar chart の描画
        pass
    elif ctype == "stacked_bar":
         # 棒グラフで表示
        survived_count = df.groupby(chart_config["target"])['Survived'].value_counts()            
        if chart_config["ratio"]:
            # 各Pclassの乗客数で正規化して割合を計算
            survived_ratio = survived_count.unstack().apply(lambda x: x/x.sum(), axis=1)
            survived_ratio.plot(kind='bar', stacked=True, color = ["green", "red"])
        else:
            survived_count.unstack().plot(kind='bar', stacked=True)
    elif ctype == "bubble":
        # bubble chartの描画
        sns.scatterplot(x = chart_config["x"], y = chart_config["y"], s = chart_config["sizes"], 
                        color = chart_config["color"], alpha = chart_config["alpha"])
    elif ctype == "beeswarm":
        sns.swarmplot(x = chart_config["x"], # group
                      y = chart_config["y"], # data 
                      palette = chart_config["palette"], alpha = chart_config["alpha"])
    
    # グラフのタイトルと軸ラベルの設定
    plt.title(title, fontsize=16)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.savefig("./../results/{}.png".format(title))
    if is_show:
        plt.show()

    return 


# データの準備
cc = {
    "x" : [1, 2, 3, 4, 5],
    "y" : [10, 20, 30, 40, 50],
    "sizes" : [100, 200, 300, 400, 500],
    "color" : "green",
    "alpha" : 0.7
}
# グラフの表示
#plot(cc, ctype = "bubble", title = "hoge", is_show = True)

# データの準備
cc = {
    "x" : ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
    "y" : [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    "palette" : {"A": "orange", "B": "green"},
    "alpha" : 0.7
}
# グラフの表示
#plot(cc, ctype = "beeswarm", title = "hogebs", is_show = True)

cc = {
    "target" : "SibSp",
    "ratio" : True
}
plot(cc, ctype = "stacked_bar", is_show = True)

