import pandas as pd, numpy as np
from catboost import CatBoostClassifier
import pickle
import sys

dtypes = {"session_id": 'int64',
          "index": np.int16,
          "elapsed_time": np.int32,
          "event_name": 'category',
          "name": 'category',
          "level": np.int8,
          "page": np.float16,
          # "room_coor_x": np.float16,
          # "room_coor_y": np.float16,
          # "screen_coor_x": np.float16,
          # "screen_coor_y": np.float16,
          "hover_duration": np.float32,
          "text": 'category',
          "fqid": 'category',
          "room_fqid": 'category',
          "text_fqid": 'category',
          "fullscreen": np.int8,
          "hq": np.int8,
          "music": np.int8,
          "level_group": 'category'
          }
use_col = ['session_id', 'index', 'elapsed_time', 'event_name', 'name', 'level', 'page',
           'room_coor_x', 'room_coor_y', 'hover_duration', 'text', 'fqid', 'room_fqid', 'text_fqid', 'level_group']

def deftarget():
    global targets
    targets = pd.read_csv('C:\\kaggle\\ОбучИгра\\train_labels.csv')
    targets['q'] = targets['session_id'].apply(lambda x: int(x.split('_')[-1][1:]))
    targets['session'] = targets.session_id.apply(lambda x: int(x.split('_')[0]))

# создаем солонку 'delt_time' в train
def def_delt_time(df):
    df.sort_values(by=['session_id', 'elapsed_time'], inplace=True)
    df['d_time'] = df['elapsed_time'].diff(1)
    df['d_time'].fillna(0, inplace=True)
    df['delt_time'] = df['d_time'].clip(0, 103000)
    df['delt_time_next'] = df['delt_time'].shift(-1)
    return df


def feature_engineer(train, kol_f):
    global kol_col, kol_col_max
    kol_col = 9
    kol_col_max = 11 + kol_f * 2
    col = [i for i in range(0, kol_col_max)]
    new_train = pd.DataFrame(index=train['session_id'].unique(), columns=col, dtype=np.float16)
    new_train[10] = new_train.index  # "session_id"

    new_train[0] = train.groupby(['session_id'])['d_time'].quantile(q=0.3)
    new_train[1] = train.groupby(['session_id'])['d_time'].quantile(q=0.8)
    new_train[2] = train.groupby(['session_id'])['d_time'].quantile(q=0.5)
    new_train[3] = train.groupby(['session_id'])['d_time'].quantile(q=0.65)
    new_train[4] = train.groupby(['session_id'])['hover_duration'].agg('mean')
    new_train[5] = train.groupby(['session_id'])['hover_duration'].agg('std')
    new_train[6] = new_train[10].apply(lambda x: int(str(x)[:2])).astype(np.uint8)  # "year"
    new_train[7] = new_train[10].apply(lambda x: int(str(x)[2:4]) + 1).astype(np.uint8)  # "month"
    new_train[8] = new_train[10].apply(lambda x: int(str(x)[4:6])).astype(np.uint8)  # "day"
    new_train[9] = new_train[10].apply(lambda x: int(str(x)[6:8])).astype(np.uint8) + new_train[10].apply(
        lambda x: int(str(x)[8:10])).astype(np.uint8) / 60
    new_train[10] = 0
    new_train = new_train.fillna(-1)

    return new_train

def feature_next_t(row_f, new_train, train, gran_1, gran_2, i):
    global kol_col
    kol_col +=1
    col1 = row_f['col1']
    val1 = row_f['val1']
    maska = (train[col1] == val1)
    if row_f['kol_col'] == 1:
        if row_f['kol_col'] == 1:
            new_train[kol_col] = train[maska].groupby(['session_id'])['delt_time_next'].sum()
        if gran_1:
            kol_col +=1
            new_train[kol_col] = train[maska].groupby(['session_id'])['delt_time'].mean()
        if gran_2:
            kol_col +=1
            new_train[kol_col] = train[maska].groupby(['session_id'])['index'].count()
    elif row_f['kol_col'] == 2:
        col2 = row_f['col2']
        val2 = row_f['val2']
        maska = maska & (train[col2] == val2)
        new_train[kol_col] = train[maska].groupby(['session_id'])['delt_time_next'].sum()
        if gran_1:
            kol_col +=1
            new_train[kol_col] = train[maska].groupby(['session_id'])['delt_time'].mean()
        if gran_2:
            kol_col +=1
            new_train[kol_col] = train[maska].groupby(['session_id'])['index'].count()
    return new_train

def feature_next_t_otvet(row_f, new_train, train, gran_1, gran_2, i):
    global kol_col
    kol_col +=1
    col1 = row_f['col1']
    val1 = row_f['val1']
    maska = (train[col1] == val1)
    if row_f['kol_col'] == 1:
        new_train[kol_col] = train[maska]['delt_time_next'].sum()
        if gran_1:
            kol_col +=1
            new_train[kol_col] = train[maska]['delt_time'].mean()
        if gran_2:
            kol_col +=1
            new_train[kol_col] = train[maska]['index'].count()
    elif row_f['kol_col'] == 2:
        col2 = row_f['col2']
        val2 = row_f['val2']
        maska = maska & (train[col2] == val2)
        new_train[kol_col] = train[maska]['delt_time_next'].sum()
        if gran_1:
            kol_col +=1
            new_train[kol_col] = train[maska]['delt_time'].mean()
        if gran_2:
            kol_col +=1
            new_train[kol_col] = train[maska]['index'].count()
    return new_train

def experiment_feature_next_t_otvet(row_f, new_train, train, gran_1, gran_2, i):
    global kol_col
    # kol_col +=1
    # if row_f['kol_col'] == 1:
    #     maska = train[row_f['col1']] == row_f['val1']
    #     new_train[kol_col] = train[maska]['delt_time_next'].sum()
    #     if gran_1:
    #         kol_col +=1
    #         new_train[kol_col] = train[maska]['delt_time'].mean()
    #     if gran_2:
    #         kol_col +=1
    #         new_train[kol_col] = train[maska]['index'].count()
    # elif row_f['kol_col'] == 2:
    #     col2 = row_f['col2']
    #     val2 = row_f['val2']
    #     maska = (train[col1] == val1) & (train[col2] == val2)
    #     new_train[kol_col] = train[maska]['delt_time_next'].sum()
    #     if gran_1:
    #         kol_col +=1
    #         new_train[kol_col] = train[maska]['delt_time'].mean()
    #     if gran_2:
    #         kol_col +=1
    #         new_train[kol_col] = train[maska]['index'].count()
    return new_train


def feature_quest_otvet(new_train, train, quest, kol_f):
    # global kol_col
    # kol_col = 9
    # g1 = 0.7
    # g2 = 0.3
    #
    # q_feature = df_feature[df_feature['quest'] == quest].copy()
    # q_feature.reset_index(drop=True, inplace=True)
    #
    # for i in range(0, kol_f):
    #     f_row = q_feature.loc[i]
    #     new_train = feature_next_t_otvet(row_f, new_train, train, i)
    # col = [i for i in range(0, kol_col + 1)]
    return new_train#[col]

from datetime import datetime
def feature_engineer_new(new_train, train, q_feature, kol_f):
    start_time = datetime.now()
    # ПЕРВАЯ КОЛОНКА
    col1 = train[q_feature['col1']] # матрица имен первой колонки
    val1 = q_feature['val1']        # матрица значений первой колонки

    col1.columns = range(col1.columns.size) # делаем названия колонок цифрами
    maska1 = col1 == val1                   # маска соответствия значения колонок в трайне и q_feature
    one_col = q_feature['kol_col'] == 1     # если используется одна, а не 2 колонки
    maska_one_col = maska1 & one_col

    # prom = q_feature['col2']
    # prom[prom=='0']='nan'

    # ЕСЛИ КОЛОНКИ ДВЕ
    col2 = train[q_feature['col2']]     # матрица имен второй колонки колонки
    val2 = q_feature['val2']            # матрица значений второй колонки колонки

    col2.columns = range(col2.columns.size) # делаем названия колонок цифрами
    maska2 = maska1 & (col2 == val2)        # маска соответствия значений первой и второй колонки
    maska_two_col = maska2 & (~ one_col)    # если колонки 2

    maska = maska_one_col | maska_two_col

    # Если столбец типа 1 то суммируем 'delt_time_next'
    delt_time_next = train['delt_time_next']
    delt_time_next = delt_time_next.mul(q_feature['tip']==1)

    nechto = maska.mul(delt_time_next, axis=0)
    row_new_train = nechto.sum()

    # Если столбец типа 2 то среднее от 'delt_time'
    delt_time = train['delt_time']
    delt_time = delt_time.mul(q_feature['tip'] == 2)

    nechto = maska.mul(delt_time, axis=0)
    row_new_train = row_new_train + nechto.mean()

    # Если столбец типа 3, то количество строк 'delt_time'
    kol_index = train['index']
    kol_index = kol_index.mul(q_feature['tip'] == 3)

    nechto = maska.mul(kol_index, axis=0)
    row_new_train = row_new_train + nechto.count()
    print('по новому', datetime.now() - start_time)

    row_new_train2 = pd.Series(data=0, index=range(row_new_train.size))

    start_time = datetime.now()
    for i in range(0, kol_f):
        f_row = q_feature.loc[i]
        col1 = f_row['col1']
        val1 = f_row['val1']
        maska = (train[col1] == val1)
        if f_row['kol_col'] == 1:

            if f_row['tip'] == 1:
                row_new_train2[i] = train[maska]['delt_time_next'].sum()
            elif f_row['tip'] == 2:
                row_new_train2[i] = train[maska]['delt_time'].mean()
            else:
                row_new_train2[i] = train[maska]['index'].count()

        elif f_row['kol_col'] == 2:
            col2 = f_row['col2']
            val2 = f_row['val2']
            maska = maska & (train[col2] == val2)

            if f_row['tip'] == 1:
                row_new_train2[i] = train[maska]['delt_time_next'].sum()
            elif f_row['tip'] == 2:
                row_new_train2[i] = train[maska]['delt_time'].mean()
            else:
                row_new_train2[i] = train[maska]['index'].count()

    print('с циклом', datetime.now() - start_time)

    for i in range(0, len(row_new_train)):
        if row_new_train[i] != row_new_train2[i]:
            print('i=',i, row_new_train[i], row_new_train2[i])
    input()

    return row_new_train

def feature_quest(new_train, train, quest, kol_f):
    global kol_col, df_feature
    kol_col = 9
    q_feature = df_feature[df_feature['quest'] == quest].copy()
    q_feature.reset_index(drop=True, inplace=True)
    list_session_id = train.session_id.unique()
    for session_id in list_session_id:
        df = train[train.session_id == session_id]
        new_train = feature_engineer_new(new_train, df, q_feature, kol_f)
    col = [i for i in range(0,kol_col+1)]
    return new_train[col]


def create_model(train, quests, models, list_kol_f):
    kol_quest = len(quests)
    train['0'] = np.nan
    # ITERATE THRU QUESTIONS
    for q in quests:
        print('### quest ', q, end='')
        new_train = feature_engineer(train, list_kol_f[q])
        train_x = feature_quest(new_train, train, q, list_kol_f[q])
        print(' ---- ', 'train_q.shape = ', train_x.shape)

        # TRAIN DATA
        train_users = train_x.index.values
        train_y = targets.loc[targets.q == q].set_index('session').loc[train_users]

        # TRAIN MODEL

        model = CatBoostClassifier(
            n_estimators=300,
            learning_rate=0.045,
            depth=6
        )

        model.fit(train_x.astype('float32'), train_y['correct'], verbose=False)

        # SAVE MODEL, PREDICT VALID OOF
        models[f'{q}'] = model
    print('***')

    return models

models = {}
best_threshold = 0.63

list_kol_f = {
    1:300
    ,3:110,
    4:110, 5:220, 6:120, 7:110, 8:110, 9:100, 10:120, 11:120,
    14: 110, 15:160, 16:105, 17:140
             }
df_feature = pd.read_csv('C:\\kaggle\\ОбучИгра\\new_feature_sort.csv')



df0_4 = pd.read_csv('C:\\kaggle\\ОбучИгра\\train_0_4t.csv', dtype=dtypes)
test = df0_4 .groupby(['session_id'])['level'].agg('nunique')
kol_lvl = (df0_4 .groupby(['session_id'])['level'].agg('nunique') < 5)
list_session = kol_lvl[kol_lvl].index
df0_4  = df0_4 [~df0_4 ['session_id'].isin(list_session)]
df0_4 = def_delt_time(df0_4)

quests_0_4 = [1, 3]
# list_kol_f = {1:140,3:110}

models = create_model(df0_4, quests_0_4, models, list_kol_f)
del df0_4



