from servise_ds import okno
import pandas as pd
import numpy as np

#сверяем сходство 2-х файлов
def sverka():
    id = 'row_id'
    stolbec = 'microbusiness_density' # столбец, который сверяем
    file1 = pd.read_csv("C:\\kaggle\\МикроБизнес\\submission (2).csv")
    # file2 = pd.read_csv("C:\\kaggle\\МикроБизнес\\submission_1.3821.csv") # 283.8127077335253
    file2 = pd.read_csv("C:\\kaggle\\МикроБизнес\\submission_best.csv") # близко 3.933763435380734
    file1.sort_values(by=id, inplace=True)
    file2.sort_values(by=id, inplace=True)
    file1.reset_index(inplace=True)
    file2.reset_index(inplace=True)
    file1['raznost'] = file1[stolbec] - file2[stolbec]
    file1['file2'] = file2[stolbec]
    file3 = file1[file1['row_id'].str[-5:] == '01-01'].copy()
    file3.sort_values(by='raznost', inplace=True)
    print(file3['raznost'].sum())
    okno.vewdf(file1)
# sverka()


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

# Поиск строки по её началу
def melk2():
    train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_5_12t.csv")
    # train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_0_4t.csv")
    # ТЕСТИРУЕМЫЕ ПАРАМЕТРЫ
    col = 'text_fqid' # перебор колонок для трайна
    ls = train[col].unique() # список значений колонки
    for param in ls:
        if type(param) == 'str':
            if param.find('tunic.historicalsociety.closet_dirty.door_bloc') >= 0:
                print(param)
        print(param)

melk2()

# ПЕРЕСЕЧЕНИЯ ЗНАЧЕНИЙ КОЛОНОК. какие варианты колонки 2 есть при определенных значениях колонки 1
def perecechenie():
    train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_5_12.csv")
    # ТЕСТИРУЕМЫЕ ПАРАМЕТРЫ
    col = 'fqid' # перебор колонок для трайна
    ls = train[col].unique() # список значений колонки
    for param in ls:
        print('fqid =', param)
        rr = train[train[col] == param]['text_fqid'].unique()
        print(rr)

# perecechenie()

# какие варианты колонки 2 есть при определенных значениях колонки 1
def melk3():
    train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_5_12.csv")
    col = 'event_name' # колонка
    ls = train[col].unique() # список значений колонки
    df = train[train['event_name'] == 'checkpoint']
    tmp = df.groupby(['session_id'])['event_name'].count()
    tmp = tmp[tmp != 1].index
    train = train[train['session_id'].isin(tmp)]

    for param in range(1,10,1):
        # df = train[train['event_name'] == 'checkpoint']
        # tmp = df.groupby(['session_id'])['event_name'].count()
        rrr = (tmp == param).sum()

        # rrr = df['session_id'].unique()
        # print(col,'=', param)
        print('param =', param)
        print('кол =', rrr)

# melk3()

# вероятности ответа без модели
def melk4():

    targets = pd.read_csv('C:\\kaggle\\ОбучИгра\\train_labels.csv')
    targets['q'] = targets['session_id'].apply(lambda x: int(x.split('_')[-1][1:]))
    targets['session'] = targets.session_id.apply(lambda x: int(x.split('_')[0]))
    # 
    print('ВСЕ')
    question_means = targets.groupby('q').correct.agg('mean')
    porog = 0.62
    df = pd.DataFrame({'veroiatnoct':question_means})#, 'preds': (question_means > porog).astype('int32'), 'tochnost' : (question_means-porog).abs()})
    print(df)
    # 
    # print('C checkpoint != 1')
    # targets = targets[targets['session'].isin(tmp)]
    # question_means = targets.groupby('q').correct.agg('mean')
    # porog = 0.62
    # df = pd.DataFrame({'veroiatnoct':question_means, 'preds': (question_means > porog).astype('int32'),
    #                    'tochnost' : (question_means-porog).abs()})
    # print(df)
    # for param in range(1,10):
    #     print('param =', param)
    #     tmp = (param == train.groupby(['session_id'])[col].agg('nunique'))
    #     print(tmp.sum())

# melk4()

# создание модели из 3-х вариантов
def melk1():
    rez = pd.read_csv("C:\\kaggle\\Гравитация\\sub3.csv")
    # rez.sort_values(by='tm', inplace=True)
    # rez['tmm']=range(len(rez))
    # rez.sort_values(by='seq_id', inplace=True)
    sub0 = pd.read_csv("C:\\kaggle\\Гравитация\\sub_best.csv")
    sub1 = pd.read_csv("C:\\kaggle\\Гравитация\\sub2.csv")
    newdf = pd.DataFrame(columns=['id','target'])
    for i in range(len(rez)):
        newdf.loc[i, 'id'] = rez.loc[i,'id']
        if abs(sub0.loc[i,'target'] - rez.loc[i,'target']) > abs(sub1.loc[i,'target'] - rez.loc[i,'target']):
            newdf.loc[i, 'target'] = sub1.loc[i, 'target']
        else:
            newdf.loc[i, 'target'] = sub0.loc[i, 'target']
    newdf.to_csv("C:\\kaggle\\Гравитация\\reztest.csv", index=False)

#melk1()

