import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from servise_ds import okno

def read_csv_loc(file):
    dtypes = {"session_id": 'int64',
              "index": np.int16,
              "elapsed_time": np.int32,
              "event_name": 'category',
              "name": 'category',
              "level": np.int8,
              "page": np.float16,
              "room_coor_x": np.float16,
              "room_coor_y": np.float16,
              # "screen_coor_x": np.float16,
              # "screen_coor_y": np.float16,
              "hover_duration": np.float32,
              "text": 'category',
              "fqid": 'category',
              "room_fqid": 'category',
              "text_fqid": 'category',
              # "fullscreen": np.int8,
              # "hq": np.int8,
              # "music": np.int8,
              "level_group": 'category'
              }
    train = pd.read_csv(file, dtype=dtypes)
    return train

def deftarget():
    global targets
    targets = pd.read_csv('C:\\kaggle\\ОбучИгра\\train_labels.csv')
    targets['q'] = targets['session_id'].apply(lambda x: int(x.split('_')[-1][1:]))
    targets['session'] = targets.session_id.apply(lambda x: int(x.split('_')[0]))

# создаем солонку 'delt_time' в train
def def_delt_time():
    global train
    train.sort_values(by=['session_id', 'elapsed_time'], inplace=True)
    train['delt_time'] = train['elapsed_time'].diff(1)
    train['delt_time'].fillna(0, inplace=True)
    train['delt_time'].clip(0, 100000, inplace=True)
    train['delt_time_next'] = train['delt_time'].shift(-1)

def def_pravilnie():
    deftarget()
    prav = targets.groupby(['session'])['correct'].mean()
    pravilniy = prav[prav == 1]
    # pravilniy2 = prav[prav >= 0.9443]
    pravilniy = pravilniy.index
    return pravilniy

# # формируем датафрейм правильных последлвательностей
def create_pravilniy_posl():
    global train
    # список колонок по которым формируем правильные последовательности
    l_col =['fqid', 'room_fqid', 'text_fqid']

    # датафрейм в котором сохраняем инфу о правильных последовательностях
    df_pravilno = pd.DataFrame(columns=['level', 'columns', 'val', 'last_val', 'pravilnost'])
    su = train.session_id.unique()
    print('len(su) =', len(su))
    i = 0
    train['last_col']=0
    for session_id in su: # цикл по всем уникальным session_id
        i += 1
        print('i=',i)
        df = train[train.session_id == session_id]
        df.reset_index(inplace=True, drop=True)
        for level in train.level.unique(): # цикл по уровням игры
            df1 = df[df.level==level]
            for col in l_col: # цикл по списку колонок по которым формируем правильные последовательности
                df1 = df1[df[col].notna()]
                df1['last_val'] = df1.shift(-1)
                maska = df1[col] != df1['last_val']
                df1 = df1[maska][[col,'last_val']]
                df1['level'] = level
                df1['columns'] = col


#             df_pravilno = pd.concat([df_serii, df_seria])
#
#     df_serii.to_csv("C:\\kaggle\\ОбучИгра\\df_serii.csv", index=False)
# create_serii()



pravilniy = def_pravilnie()
train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_5_12t.csv")
train = train[train['session_id'].isin(pravilniy)]

pass