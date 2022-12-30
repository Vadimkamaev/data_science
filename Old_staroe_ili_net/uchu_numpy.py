from servise_ds import okno
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
pd.options.display.width = 0 # вывод данных во всю ширину окна
#df = pd.read_csv('C:\\Users\\vadim\\Desktop\\Downloads\\StudentsPerformance.csv')
#df = pd.read_csv('C:\\Users\\vadim\\Desktop\\Downloads\\accountancy.csv')
#df = pd.read_csv('C:\\Users\\vadim\\Desktop\\Downloads\\dota_hero_stats.csv')
#concentrations = pd.read_csv('C:\\Users\\vadim\\Desktop\\Downloads\\algae.csv')
#df = pd.read_csv('C:\\Users\\vadim\\Desktop\\Downloads\\income.csv')
#df = pd.read_csv('C:\\Users\\vadim\\Desktop\\Downloads\\genome_matrix.csv', index_col=0)
#df = pd.read_csv('C:\\Users\\vadim\\Desktop\\Downloads\\iris.csv')
#my_stat = pd.read_csv('C:\\Users\\vadim\\Desktop\\Downloads\\my_stat_1.csv')
#my_stat.loc[my_stat['session_value'].isnull, 'session_value'] = 0
#okno.vewdf(df)
events_data = pd.read_csv('C:\\kaggle\\Степик\\event_data_train.csv')
events_data['data'] = pd.to_datetime(events_data['timestamp'], unit='s')
events_data['day'] = events_data['data'].dt.date
events_data = events_data.sort_values('timestamp')
print(events_data.head())
#events_data.sort_values(by='user_id')
#vdf = events_data.pivot_table(index='user_id', columns='action', aggfunc='count', values='step_id', fill_value=0)
gdbh = events_data.groupby('user_id').count()['step_id']
max_col_step = gdbh.max()
print(max_col_step)
print('__________________________________________')
#print(vdf.head())
#display(events_data.head)











