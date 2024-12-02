import os, shutil
import pickle
import pandas as pd

def initialize(dir_path):
    '''ディレクトリ初期化'''
    os.makedirs(dir_path, exist_ok=True)
    shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def read_csv(filepath, usecols=None, nrows=None, encoding='utf-8'):
    '''csvの読み込み'''
    df = pd.read_csv(
        filepath,
        header='infer', # header操作:int, list of int, default ‘infer’
        # names=(), # 列名:array-like, optional
        usecols=usecols, # 読み込む列を限定するとき, リストを指定
        dtype = 'object',
        nrows = nrows,
        encoding = encoding # 'cp932'
    )
    return df

def write_to_csv(filepath, df):
    '''csvに書き出し'''
    df.to_csv(filepath, index=False)

def read_pickle(filepath):
    '''pickleの読み込み'''
    with open(filepath,'rb') as f:
        data = pickle.load(f)
    return data

def write_to_pickle(filepath, data):
    '''pickleに書き出し'''
    with open(filepath,'wb') as f:
        pickle.dump(data,f)

def write_to_txt(filepath,text):
    '''txtに書き込み'''
    with open(filepath,'a') as f:
        f.write(text)