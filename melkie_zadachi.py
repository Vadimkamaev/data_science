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
              # "text": 'category',
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

def melk2():
    train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_13_22.csv")

    # ТЕСТИРУЕМЫЕ ПАРАМЕТРЫ
    col = 'text_fqid' # перебор колонок для трайна
    ls = train[col].unique() # список значений колонки
    for param in ls:
        print(param)

melk2()


# rez = pd.read_csv("C:\\kaggle\\белки\\train1.csv")
# rez1 = rez[(rez['lenstr']> 60)&(rez['lenstr']< 300) ]
# pass
