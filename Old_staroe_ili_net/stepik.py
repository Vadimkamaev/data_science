from servise_ds import okno
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
pd.options.display.width = 0 # вывод данных во всю ширину окна
#okno.vewdf(df)

#работа с файлом event_data_train
def func_events_data():
    # работа с events_data - файл со всеми действиями в степике
    events_data = pd.read_csv('C:\\kaggle\\Степик\\event_data_train.csv')
    events_data['data'] = pd.to_datetime(events_data['timestamp'], unit='s')
    events_data['day'] = events_data['data'].dt.date
    users_events_data = events_data.pivot_table(index='user_id',
                                                columns='action',
                                                aggfunc='count',
                                                values='step_id', fill_value=0)
    users_day = events_data.groupby('user_id')['day'].nunique()
    user_data = events_data.groupby('user_id', as_index=False).\
    agg({'timestamp':'max'}).rename(columns={'timestamp':'last_timestamp'})
    # последнее время в датафрейме в формате таймстемп
    now = 1526772811
    # время отсутствия на платформе при котором мы считаем, что чел бросил занятия
    drop_out = 30 * 24 * 3600 # 30 дней в формате таймстемп
    user_data['is_gone_user'] = (now - user_data['last_timestamp']) > drop_out
    return user_data, users_events_data, users_day

#работа с файлом submissions_data_train
def func_submissions_data():
    # работа с submissions_data - файл с действиями с задачами в степике
    submissions_data = pd.read_csv('C:\\kaggle\\Степик\\submissions_data_train.csv')
    #создание новых полей дата и дэй
    submissions_data['data'] = pd.to_datetime(submissions_data['timestamp'], unit='s')
    submissions_data['day'] = submissions_data['data'].dt.date
    print(submissions_data.head())
    # количество корректных и ошибочных решений
    user_scores = submissions_data.pivot_table(index='user_id',
                                       columns='submission_status',
                                       aggfunc='count',
                                       values='step_id',
                                       fill_value=0)
    return user_scores

# формируем датафрейм обобщающий данные о прохождении курса в целом
def vcelom():
    user_data, users_events_data, users_day = func_events_data() # все действия в степике
    #print(users_day.head())
    #print(user_data.head())
    #print(users_events_data.head())
    user_scores = func_submissions_data() # решение задач в степике
    print(user_scores.head())
    user_data = user_data.join(user_scores, on='user_id')
    #print(user_data.head())
    user_data = user_data.fillna(0)
    #print(user_data.head())
    user_data = user_data.join(users_events_data, on='user_id')
    #print(user_data.head())
    user_data = user_data.join(users_day, on='user_id')
    #print(user_data.head())
    user_data['passed_course'] = user_data['passed'] > 170 # прошёл курс
    print('__________!!!!!!!!!!!!_____________')
    print(user_data.head())

#vcelom()

#работа с файлом submissions_data_train
def func_submissions_data():
    # работа с submissions_data - файл с действиями с задачами в степике
    submissions_data = pd.read_csv('C:\\kaggle\\Степик\\submissions_data_train.csv')
    #создание новых полей дата и дэй
    submissions_data['data'] = pd.to_datetime(submissions_data['timestamp'], unit='s')
    submissions_data['day'] = submissions_data['data'].dt.date
    print(submissions_data.head(30))
    # количество корректных и ошибочных решений
    # user_scores = submissions_data.pivot_table(index='user_id',
    #                                    columns='submission_status',
    #                                    aggfunc='count',
    #                                    values='step_id',
    #                                    fill_value=0)
    nemogu = submissions_data[submissions_data['submission_status']=='wrong']
    rez = nemogu.groupby('user_id')['step_id'].max()
    mod = rez.mode()
    #rez = submissions_data.groupby('user_id').agg({'timestamp':'max'})
    #maska =
    print(rez)
    print(mod)
    print('________________________')
    print(submissions_data.mode())

func_submissions_data()
data = pd.read_csv('https://stepik.org/media/attachments/course/4852/submissions_data_train.zip')
rrr = data[data.submission_status == "wrong"].groupby(['user_id', 'step_id'], as_index=False).agg({'timestamp':'max'}).step_id.value_counts().keys()[0]
print(rrr)

