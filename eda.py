from parser import get_args
from lib.utils.IO import *
from lib.preprocessing.dataframe import *

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocessing(ser):
    if ser.dtype == float:
        is_int = lambda x: True if math.modf(x)[0] == 0 else False
        if ser.map(is_int).all():
            ser = ser.astype(int)
    return ser

def draw_histogram(data, title, graph_path):
    p01 = np.quantile(data, 0.01, axis=0)
    p25 = np.quantile(data, 0.25, axis=0)
    p50 = np.quantile(data, 0.50, axis=0)
    p75 = np.quantile(data, 0.75, axis=0)
    p99 = np.quantile(data, 0.99, axis=0)

    h = plt.hist(data,range=(p01,p99),bins=20,ec='black')[0].max()
    plt.vlines(p25, 0, h, colors='orange', linestyle='dashed')
    plt.vlines(p50, 0, h, colors='orange', linestyle='solid' )
    plt.vlines(p75, 0, h, colors='orange', linestyle='dashed')
    plt.title(title)
    plt.grid()
    plt.savefig(graph_path)
    plt.clf()

def main():
    initialize('output/output_eda')
    args = get_args()
    df = read_csv(args.eda_path)

    md = '# csv概要\n'
    md += f'- filepath: {args.eda_path}\n'
    md += f'- shape: {df.shape}\n'

    md += '\n# 特徴量説明\n'
    # TODO: trainとtestの分布の比較
    for column in df.columns:
        note = ''
        N = len(df[column])
        md += f'## {column}\n'
        md += f'- 説明: \n'
        md += f'- dtype: {df[column].dtype}\n'
        md += f'- null: あり({np.round(100 * df[column].isnull().sum()/ N,1)}%)\n' if df[column].isnull().any() else '- null: なし\n'
        ser = preprocessing(df[column].dropna())
        if len(set(ser.values.tolist())) == len(ser.values.tolist()):
            note += '一意'
        elif len(set(ser.values.tolist())) <= 10:
            # TODO: ラベルごとのyの分布図
            md += f'- 項目:\n'
            vc = ser.value_counts()
            plt.pie(vc.values, labels=vc.index.values, startangle=90, counterclock=False, autopct='%.1f%%')
            plt.title(column)
            plt.legend(loc='lower left', bbox_to_anchor=(0.4+0.5/1.4, 0.4+0.5/1.4))
            plt.savefig(f'output/output_eda/pie_{column}.pdf')
            plt.clf()
            md += f'- 円グラフ: \n![円グラフ](output_eda/pie_{column}.pdf)\n'
        elif ser.dtype in (int, float):
            graph_path = f'output/output_eda/hist_{column}.pdf'
            draw_histogram(ser.values, column, graph_path)
            md += f'- ヒストグラム: \n![ヒストグラム](output_eda/hist_{column}.pdf)\n'
        if column in args.eda_log_column_list:# 対数変換
            graph_path = f'output/output_eda/hist_log_{column}.pdf'
            draw_histogram(log(ser.values), column, graph_path)
            md += f'- ヒストグラム_log: \n![ヒストグラム](output_eda/hist_log_{column}.pdf)\n'

        md += f'- 例0: {df[column].iloc[0]}\n'
        md += f'- 例-1: {df[column].iloc[-1]}\n'
        md += f'- 備考: {note}\n\n'

    write_to_txt('output/EDA_Report.md',md)