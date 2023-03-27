import numpy as np
import pandas as pd
from servise_ds import okno


# Reduce Memory Usage Функция уменьшения занимаемой датафреймом памяти
def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
            if (col_type != 'object'):
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)

                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
            else:
                df[col] = df[col].astype('category')
    mem_usg = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage became: ", mem_usg, " MB")
    return df

dtypes = {"session_id": 'int64',
          "index": np.int16,
          "elapsed_time": np.int32,
          "event_name": 'category',
          "name": 'category',
          "level": np.int8,
          "page": np.float16,
          "room_coor_x": np.float16,
          "room_coor_y": np.float16,
          "screen_coor_x": np.float16,
          "screen_coor_y": np.float16,
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
           'room_coor_x', 'room_coor_y', 'hover_duration', 'fqid', 'room_fqid', 'text_fqid', 'level_group']


train = pd.read_csv("C:\\kaggle\\ОбучИгра\\train.csv", dtype=dtypes, usecols=use_col)
# train = reduce_memory_usage(train)
# missed_columns = ["fullscreen", "hq", "music"]
# train = train.drop(missed_columns, axis=1)
# train.to_csv("C:\\kaggle\\ОбучИгра\\train_sokr.csv", index=False)

# train = pd.read_csv("C:\\kaggle\\ОбучИгра\\train_sokr.csv")

df = train[train['level_group'] == '0-4']
df.to_csv("C:\\kaggle\\ОбучИгра\\train_0_4.csv", index=False)

df = train[train['level_group'] == '5-12']
df.to_csv("C:\\kaggle\\ОбучИгра\\train_5_12.csv", index=False)

df = train[train['level_group'] == '13-22']
df.to_csv("C:\\kaggle\\ОбучИгра\\train_13_22.csv", index=False)


# ['session_id' - 11779 nunique, 'index' от 0 до 20473 - 20348 nunique,
# 'elapsed_time', 'event_name' - 11 nunique, 'name' - 6 nunique, 'level' - от 0 до 22, 'page' - 97.83 % null,
# 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y', 'hover_duration' - 92.4 % null,
# 'text' - 594 nunique, 'fqid' - 127 nunique, 'room_fqid' - 19 - nunique, 'text_fqid' - 126 nunique,
# 'level_group' - 3 nunique]



# okno.vewdf(train)
#
# missed_columns = ["page", "hover_duration", "text"]
# train = train.drop(missed_columns, axis=1)
# print(train.info())

# okno.vewdf(train)