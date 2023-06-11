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


# _________________ ДАЛЕЕ ТЕСТИРОВАНИЕ И СОРТИРОВКА ФИЧЕЙ ПО КАЧЕСТВУ ______________

# формируем трайн из 1 колонки - из одной строки датафрейма feature_df
def feature_engineer(row_f, new_train):
    global train
    # tip = row_f['tip']
    col1 = row_f['col1']
    val1 = row_f['val1']
    l = len(new_train.columns)
    if row_f['kol_col'] == 1: # если фича одна
        maska = (train[col1] == val1)
        new_train[f'{l}'] = train[maska].groupby(['session_id'])['delt_time'].sum()
        new_train[f'{l+1}'] = train[maska].groupby(['session_id'])['index'].count()
        new_train[f'{l+2}'] = train[maska].groupby(['session_id'])['delt_time_next'].sum()
    elif row_f['kol_col'] == 2: # если фичей две
        col2 = row_f['col2']
        val2 = row_f['val2']
        maska = (train[col1] == val1) & (train[col2] == val2)
        new_train[f'{l}'] = train[maska].groupby(['session_id'])['delt_time'].sum()
        new_train[f'{l+1}'] = train[maska].groupby(['session_id'])['index'].count()
    return new_train

# формируем трайн из строк датафрейма feature_df, которые имеют 'rez' > 0
def feature_engineer_new(feature_chast_strok):
    global train, new_train
    for i, row_f in feature_chast_strok.iterrows():  # цикл по строкам feature_df относящимся к вопросу
        new_train = feature_engineer(row_f, new_train)
        if i % 50 == 0:
            new_train = new_train.copy()
        new_train[f'{l+2}'] = train[maska].groupby(['session_id'])['delt_time_next'].sum()
    return new_train


# формируем трайн из 1 колонки - из одной строки датафрейма feature_df
def feature_next_t(row_f, new_train, gran_1, gran_2):
    global train
    col1 = row_f['col1']
    val1 = row_f['val1']
    l = len(new_train.columns)
    if row_f['kol_col'] == 1: # если фича одна
        maska = (train[col1] == val1)
        new_train[f'{l}'] = train[maska].groupby(['session_id'])['delt_time_next'].sum()
        new_train[f'{l + 1}'] = train[maska].groupby(['session_id'])['delt_time_next'].mean()
        if gran_1:
            new_train[f'{l+1}'] = train[maska].groupby(['session_id'])['delt_time'].sum()
            new_train[f'{l + 1}'] = train[maska].groupby(['session_id'])['delt_time'].mean()
        if gran_2:
            new_train[f'{l+2}'] = train[maska].groupby(['session_id'])['index'].count()
    elif row_f['kol_col'] == 2: # если фичей две
        col2 = row_f['col2']
        val2 = row_f['val2']
        maska = (train[col1] == val1) & (train[col2] == val2)
        new_train[f'{l}'] = train[maska].groupby(['session_id'])['delt_time_next'].sum()
        new_train[f'{l + 1}'] = train[maska].groupby(['session_id'])['delt_time_next'].mean()
        if gran_1:
            new_train[f'{l+1}'] = train[maska].groupby(['session_id'])['delt_time'].sum()
            new_train[f'{l + 1}'] = train[maska].groupby(['session_id'])['delt_time'].mean()
        if gran_2:
            new_train[f'{l+2}'] = train[maska].groupby(['session_id'])['index'].count()
    return new_train


# формируем трайн из 1 колонки - из одной строки датафрейма feature_df
# предсказание по фичам - свойствам серии
def feature_engineer_serii(row_f, new_train):
    global train, df_serii
    # tip = row_f['tip']
    col1 = row_f['col1']
    val1 = row_f['val1']
    l = len(new_train.columns)
    if row_f['kol_col'] == 1: # если фича одна
        maska = (df_serii['columns'] == col1) & (df_serii['val'] ==  val1)
        new_train[f'{l}'] = df_serii[maska].groupby(['session_id'])['delt_time'].mean()
        new_train[f'{l+1}'] = df_serii[maska].groupby(['session_id'])['kol'].mean()
        # new_train[f'{l + 2}'] = df_serii[maska].groupby(['session_id'])['index'].count()
    return new_train

# формируем трайн из строк датафрейма feature_df, которые имеют 'rez' > 0
def feature_engineer_new(feature_q, kol_f):
    global train, new_train, k2t
    g1 = 0.7 # коэфициент - насколько меньше оптимально 0.6 - 0.7 но не сильно меняется
    g2 = 0.3 # неубедительно хорошо
    gran1 = round(kol_f * g1)
    gran2 = round(kol_f * g2)
    for i in range(0, kol_f): # цикл по строкам feature_df относящимся к вопросу
        row_f = feature_q.loc[i]
        # предсказание по фичам времени и количества
        new_train = feature_next_t(row_f, new_train, i < gran1, i <  gran2, )         # new_train = feature_engineer_serii(row_f, new_train)  # предсказание по фичам - свойствам серии
        if i % 20 == 0:
            new_train = new_train.copy()

def one_vopros(train_index, test_index):
    global new_train, targets, quest, pred_skaz
    # TRAIN DATA
    train_x = new_train.iloc[train_index]
    train_users = train_x.index.values
    train_y = targets.loc[targets.q == quest].set_index('session').loc[train_users]

    # VALID DATA
    valid_x = new_train.iloc[test_index]
    valid_users = valid_x.index.values
    valid_y = targets.loc[targets.q == quest].set_index('session').loc[valid_users]

    # TRAIN MODEL
    model = CatBoostClassifier(
        n_estimators = 200, # 200 лучше чем 300 до 200 фичей. От 200 до 300 ~ похоже
        learning_rate= 0.1,
        depth = 6
    )

    X = train_x.astype('float32')
    Y = train_y['correct']
    model.fit(X, Y, verbose=False)

    # SAVE MODEL, PREDICT VALID OOF
    pred_skaz.loc[valid_users, quest] = model.predict_proba(valid_x.astype('float32'))[:, 1]
    return pred_skaz

def preds():
    global quest, new_train, targets, pred_skaz, true
    ALL_USERS = new_train.index.unique()
    # print('В трайне', len(ALL_USERS), 'пользователей')
    gkf = KFold(n_splits=5)
    pred_skaz = pd.DataFrame(data=np.zeros((len(ALL_USERS), 1)), index=ALL_USERS)
    true = pred_skaz.copy()  # истинные значения

    # ВЫЧИСЛИТЕ РЕЗУЛЬТАТ С 5-ГРУППОВЫМ K FOLD
    for i, (train_index, test_index) in enumerate(gkf.split(X=new_train)):
        # print(' ', i + 1, end='')
        pred_skaz = one_vopros(train_index, test_index)
    # print()

    # GET TRUE LABELS
    tmp = targets.loc[targets.q == quest].set_index('session').loc[ALL_USERS]
    true[quest] = tmp.correct.values
    return true

def otvet():
    global quest, pred_skaz, true, best_thresholds, kol_stuk, b_thresholds
    best_threshold = 0.61
    # Считаем F1 SCORE для каждого вопроса
    tru = true[quest].values
    y_pred = pred_skaz[quest].values
    m = f1_score(tru, (y_pred > best_threshold).astype('int'), average='macro')

   # ЭКСПЕРИМЕНТ
    best_score = 0
    best_threshold = 0
    for threshold in np.arange(best_thresholds[quest-1], best_thresholds[quest-1] + 2, 0.01):
        preds = (y_pred > threshold).astype('int')
        m = f1_score(tru, preds, average='macro')
        if m > best_score:
            best_score = m
            best_threshold = threshold
    print('средний best_threshold =', best_threshold, end = ' ')
    return best_score

    # print('Результат для вопроса:', quest, 'F1 =', m)
    # return m

def main_ml(feature_df):
    global train,  best_thresholds, kol_stuk, b_thresholds, new_train
    best_thresholds = [0.6, 0.85, 0.80]
    new_train = pd.DataFrame(index=train['session_id'].unique(), columns=[])
    feature_engineer_new(feature_df)
    preds()  # предсказание / модель
    m = otvet()
    print('результат =', m)

# feature_df = pd.read_csv("C:\\kaggle\\ОбучИгра\\feature1.csv")
# train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_0_4t.csv")
# deftarget()
# def_delt_time()
# main_ml()

# СОРТИРОВКА СТОЛБЦОВ ПО КАЧЕСТВУ

def main_ml(feature_q, f_0, kol_f):
    global train,  best_thresholds, kol_stuk, b_thresholds, new_train
    best_thresholds = [0.6, 0.85, 0.80]
    best_thresholds += [0.6, 0.55, 0.65, 0.65, 0.55, 0.65, 0.55, 0.55, 0.65, 0.4]
    best_thresholds += [0.55, 0.5, 0.6, 0.6, 0.7]
    new_train = pd.DataFrame(index=train['session_id'].unique(), columns=[])

    feature_engineer_new(feature_q, kol_f)

    preds()  # предсказание / модель
    m = otvet()
    return m
    rezult.loc[len(rezult.index)] = [kol_feature, quest, m]

# СОРТИРОВКА СТОЛБЦОВ ПО КАЧЕСТВУ
def sort_kachestvo( feature_df):
    feature_df['kach'] = feature_df['kach1'] +feature_df['kach1'] + feature_df['kach2'] + feature_df['kach3']\
                         + feature_df['kach4'] + feature_df['kach5'] + feature_df['kach6'] + feature_df['kach7']\
                         + feature_df['kach8'] + feature_df['kach9'] + feature_df['kach10'] + feature_df['kach11']\
                         + feature_df['kach12'] + feature_df['kach13']
    feature_df.loc[(feature_df['rez'] > 0.5) & (feature_df['quest'] < 4), 'kach'] += 100
    feature_q = feature_df[feature_df['quest'] == quest].copy()
    feature_q.sort_values(by=['kach'], ascending=False, inplace=True, )
    feature_q.reset_index(drop = True , inplace = True)
    return feature_q

def vse_quest(feature_df):
    global quest
    l_quest = [1, 2, 3]
    for quest in l_quest: # цикл по вопросам
        print('ВОПРОС №', quest)
        feature_q = sort_kachestvo(quest, feature_df)
        for kol_f in range(0,300,50): # цикл по количеству применяемых фичей
            print('Количестко фичей =', kol_f, end = ' ')
            # main_ml(feature_q.head(kol_f))
            maska = feature_q.index > kol_f
            main_ml(feature_q[maska].head(50))

feature_df = pd.read_csv("C:\\kaggle\\ОбучИгра\\feature2.csv")
train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_0_4t.csv")
def_delt_time()

# deftarget()
# vse_quest(feature_df)

# формируем датафрейм серий
def create_serii():
    global train
    # список колонок по которым делаем серии
    l_col =['name', 'event_name', 'fqid', 'room_fqid', 'text_fqid']

    train.sort_values(by='elapsed_time', inplace=True)
    # датафрейм в котором сохраняем инфу о сериях
    df_serii = pd.DataFrame(columns=['columns', 'val', 'elapsed_time', 'delt_time', 'session_id', 'index', 'kol'])
    su = train.session_id.unique()
    print('len(su) =', len(su))
    i = 0
    for session_id in train.session_id.unique(): # цикл по всем уникальным session_id
        i += 1
        print('i=',i)
        df = train[train.session_id == session_id]
        df.reset_index(inplace=True, drop=True)
        df2 = df.shift(-1)
        for col in l_col: # цикл по списку формирующихся серий
            maska = ((df[col] != df2[col]) & df[col].notna()) | (df[col].isna() & df2[col].notna())
            df1 = df[maska]
            df_seria = pd.DataFrame(columns=['columns', 'val', 'elapsed_time', 'delt_time', 'session_id', 'index', 'kol'])
            df_seria['val'] = df1[col]
            df_seria['elapsed_time'] = df1['elapsed_time']
            df_seria['index'] = df1.index
            df_seria['kol'] = df_seria['index'].diff(1)
            df_seria['session_id'] = session_id
            df_seria['delt_time'] = df_seria['elapsed_time'].diff(1)
            df_seria['columns'] = col
            df_seria.iloc[0, 3] = df_seria.iloc[0, 2]
            df_seria.iloc[0, 6] = df_seria.iloc[0, 5]
            df_serii = pd.concat([df_serii, df_seria])

    df_serii.to_csv("C:\\kaggle\\ОбучИгра\\df_serii.csv", index=False)
    feature_df.loc[feature_df['rez'] > 0.5, 'kach'] += 100
    feature_df.sort_values(by=['quest','kach'], ascending=False, inplace=True, )
    feature_df.reset_index(drop = True , inplace = True)
    return feature_df

# def vse_quest(feature_df):
#     global quest
#     l_quest = [1, 2, 3]
#     for quest in l_quest: # цикл по вопросам
#         print('ВОПРОС №', quest)
#         feature_q = sort_kachestvo(quest, feature_df)
#         for kol_f in range(0,300,50): # цикл по количеству применяемых фичей
#             print('Количество фичей =', kol_f, end = ' ')
#             # main_ml(feature_q.head(kol_f))
#             maska = feature_q.index > kol_f
#             main_ml(feature_q[maska].head(50))

def vse_quest2(feature_df):
    global quest, k2t, kl
    l_quest = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    k2t = 0.7 # коэфициент - насколько меньше оптимально 0.6 - 0.7 но не сильно меняется
    kl = 0.3 # неубедительно хорошо
    rezult = pd.DataFrame(columns=['kol_feature', 'quest', 'rezultat'])
    for quest in l_quest: # цикл по вопросам
        print('ВОПРОС №', quest)
        # feature_q = sort_kachestvo(quest, feature_df)
        feature_q = feature_df[feature_df['quest'] == quest].copy()
        feature_q.reset_index(drop=True, inplace=True)
        for kol_f in range(250,450,50): # цикл по количеству применяемых фичей (оптимально 140 - 180)
            print('Количество фичей =', kol_f, end = ' ')
            m = main_ml(feature_q, 0, kol_f)
            print('результат =', m)
            rezult.loc[len(rezult.index)] = [kol_f, quest, m]
        print('ПО ВСЕМ ВОПРОСАМ В СРЕДНЕМ')
        for kol_feature in rezult.kol_feature.unique():
            print('Количество фичей =', kol_feature, 'результат =', rezult[rezult.kol_feature==kol_feature]['rezultat'].mean() )


feature_df = pd.read_csv("C:\\kaggle\\ОбучИгра\\feature_sort.csv")
# feature_df = sort_kachestvo(feature_df)



train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_5_12t.csv")
# df_serii = read_csv_loc("C:\\kaggle\\ОбучИгра\\df_serii.csv")
def_delt_time()

deftarget()
vse_quest2(feature_df)

# # формируем датафрейм серий
# def create_serii():
#     global train
#     # список колонок по которым делаем серии
#     l_col =['name', 'event_name', 'fqid', 'room_fqid', 'text_fqid']
#
#     train.sort_values(by='elapsed_time', inplace=True)
#     # датафрейм в котором сохраняем инфу о сериях
#     df_serii = pd.DataFrame(columns=['columns', 'val', 'elapsed_time', 'delt_time', 'session_id', 'index', 'kol'])
#     su = train.session_id.unique()
#     print('len(su) =', len(su))
#     i = 0
#     for session_id in train.session_id.unique(): # цикл по всем уникальным session_id
#         i += 1
#         print('i=',i)
#         df = train[train.session_id == session_id]
#         df.reset_index(inplace=True, drop=True)
#         df2 = df.shift(-1)
#         for col in l_col: # цикл по списку формирующихся серий
#             maska = ((df[col] != df2[col]) & df[col].notna()) | (df[col].isna() & df2[col].notna())
#             df1 = df[maska]
#             df_seria = pd.DataFrame(columns=['columns', 'val', 'elapsed_time', 'delt_time', 'session_id', 'index', 'kol'])
#             df_seria['val'] = df1[col]
#             df_seria['elapsed_time'] = df1['elapsed_time']
#             df_seria['index'] = df1.index
#             df_seria['kol'] = df_seria['index'].diff(1)
#             df_seria['session_id'] = session_id
#             df_seria['delt_time'] = df_seria['elapsed_time'].diff(1)
#             df_seria['columns'] = col
#             df_seria.iloc[0, 3] = df_seria.iloc[0, 2]
#             df_seria.iloc[0, 6] = df_seria.iloc[0, 5]
#             df_serii = pd.concat([df_serii, df_seria])
#
#     df_serii.to_csv("C:\\kaggle\\ОбучИгра\\df_serii.csv", index=False)
# create_serii()


# train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_5_12t.csv")
# l_quest = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#
#

# вопрос
# 4 -- 300, 250
# 5 -- 250, 300
# 6 -- 400, 250, 300
# 7 -- 350, 250, 200
# 8 -- 250, 150
# 9 -- 250, 300
# 10 -- 350, 100, 150 -- 0.6275881429849036
# 11 -- 400, 350, 200, 100 -- 0.6058749470525528
# 12 -- 350, 200, 250 -- 0.5848681495411421
# 13 -- 250, 200

# train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_13_22t.csv")
# l_quest = [14, 15, 16, 17, 18]