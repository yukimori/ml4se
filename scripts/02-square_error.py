# -*- coding: utf-8 -*-
#
# 誤差関数（最小二乗法）による回帰分析
# 計算が簡単で厳密に答えを求められるので二乗誤差Edを判断基準として採用
# 2015/04/22 ver1.0
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from numpy.random import normal

#------------#
# Parameters #
#------------#
#N=10            # サンプルを取得する位置 x の個数
N=100 #サンプル数を増やすと次数が多くてもoverfittingを防げる
M=[0,1,3,9]     # 多項式の次数
# M=9であれば任意の10個の点を通過する曲線を作り出せる
# 従って訓練データが10である今回の例ではM=9とすれば
# 全てのデータに誤差0で一致する曲線を作り出すことができる
# テストの識別率は悪化する（過学習）

# データセット {x_n,y_n} (n=1...N) を用意
def create_dataset(num):
    # DataFrameはpandasのライブラリ
    dataset = DataFrame(columns=['x','y'])
    for i in range(num):
        x = float(i)/float(num-1)
        y = np.sin(2*np.pi*x) + normal(scale=0.3)
        # Seriesは順序保持される点で1次元配列に似ている
        # ただし、辞書のようにインデックスを文字列にできる
        print Series([x,y], index=['x', 'y'])
        """
        x    1.000000
        y    0.093874
        dtype: float64
        """
        dataset = dataset.append(Series([x,y], index=['x','y']),
                                 ignore_index=True)
#    print dataset
    return dataset

# 平方根平均二乗誤差（Root mean square error）を計算
# 多項式から予想される値とトレーニングセットの値が
# 平均的にどの程度異なっているかを示す
# 
# Edは多項式から予測される値と訓練データの値の差の2乗を合計したものの半分
# 2*Ed/Nで多項式から予測される値と訓練データの値の差の2乗の平均値
# sqrt(2*Ed/N)で多項式から予測される値と訓練データの平均値
#  平方根をとっているのでEdの2乗の効果を取り除いている
def rms_error(dataset, f):
    err = 0.0
    for index, line in dataset.iterrows():
        x, y = line.x, line.y
        err += 0.5 * (y - f(x))**2
    return np.sqrt(2 * err / len(dataset))

# 最小二乗法で解を求める
def resolve(dataset, m):
    t = dataset.y
    phi = DataFrame()
    for i in range(0,m+1):
        p = dataset.x**i
        p.name="x**%d" % i
        phi = pd.concat([phi,p], axis=1)
    tmp = np.linalg.inv(np.dot(phi.T, phi))
    ws = np.dot(np.dot(tmp, phi.T), t)

    def f(x):
        y = 0
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    return (f, ws)

# Main
if __name__ == '__main__':
    # train:N個,test:N個のデータを作成する
    train_set = create_dataset(N)
    test_set = create_dataset(N)
    df_ws = DataFrame()

    # 多項式近似の曲線を求めて表示
    fig = plt.figure()
    for c, m in enumerate(M):
        f, ws = resolve(train_set, m)
        df_ws = df_ws.append(Series(ws,name="M=%d" % m))

        subplot = fig.add_subplot(2,2,c+1)
        subplot.set_xlim(-0.05,1.05)
        subplot.set_ylim(-1.5,1.5)
        subplot.set_title("M=%d" % m)

        # トレーニングセットを表示
        subplot.scatter(train_set.x, train_set.y, marker='o', color='blue')

        # 真の曲線を表示
        linex = np.linspace(0,1,101)
        liney = np.sin(2*np.pi*linex)
        subplot.plot(linex, liney, color='green', linestyle='--')

        # 多項式近似の曲線を表示
        linex = np.linspace(0,1,101)
        liney = f(linex)
        label = "E(RMS)=%.2f" % rms_error(train_set, f)
        subplot.plot(linex, liney, color='red', label=label)
        subplot.legend(loc=1)

    # 係数の値を表示
    print "Table of the coefficients"
    print df_ws.transpose()
    fig.show()

    # トレーニングセットとテストセットでの誤差の変化を表示
    df = DataFrame(columns=['Training set','Test set'])
    for m in range(0,10):   # 多項式の次数
        f, ws = resolve(train_set, m)
        train_error = rms_error(train_set, f)
        test_error = rms_error(test_set, f)
        df = df.append(
                Series([train_error, test_error],
                    index=['Training set','Test set']),
                ignore_index=True)
    df.plot(title='RMS Error', style=['-','--'], grid=True, ylim=(0,0.9))
    plt.show()
