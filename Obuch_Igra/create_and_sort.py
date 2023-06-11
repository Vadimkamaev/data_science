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

# ___________ СОЗДАНИЕ ДАТАФРЕЙМА СО СПИСКОМ ФИЧЕЙ _________________

# фичи из 1 колонки
def fich1():
    global train, feature_df, l_quest, nabor
    CATS = ['name', 'event_name', 'fqid', 'room_fqid', 'text_fqid', 'level', 'page', 'text']
    nabor = 0
    for cat in CATS:
        nabor += 1
        for quest in l_quest:
            for val in train[cat].unique():
                # feature_df.loc[len(feature_df.index)] = (nabor, 't', quest, 1, cat, val, 0, 0, 0, 0, 0)
                feature_df.loc[len(feature_df.index)] = (nabor, ' ', quest, 1, cat, val, 0, 0, 0, 0, 0)

# фичи из 2 колонок
def fich2():
    global train, feature_df, l_quest, nabor
    CATS = [['room_fqid', 'level'],['text_fqid', 'level'], ['fqid', 'level'], ['room_fqid', 'fqid']]
    for cat in CATS:
        nabor += 1
        for quest in l_quest:
            lcet0 = train[cat[0]]
            for val0 in lcet0.unique():
                l_cat1 = train[lcet0==val0][cat[1]].unique()
                if len(l_cat1) > 1:
                    for val1 in l_cat1:
                        # feature_df.loc[len(feature_df.index)]=(nabor,'t',quest,2,cat[0],val0,cat[1],val1,0,0,0)
                        feature_df.loc[len(feature_df.index)]=(nabor,' ',quest,2,cat[0],val0,cat[1],val1,0,0,0)

# создание датафрейма с фичами
def create_df_feature():
    global train, feature_df, l_quest
    # 'nabor' - номер набора колонок; 'tip' - тип фичи: 't' - время, 'l' - количество
    # 'kach' - чем больше тем лучше фича; rez - если не 0, то принята, место в трайне
    feature_df = pd.DataFrame(columns=['nabor', 'tip', 'quest', 'kol_col', 'col1', 'val1', 'col2', 'val2',
                                       'col3', 'val3', 'kach', 'rez'])

    train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_0_4t.csv")
    l_quest = [1,2,3]
    fich1()
    fich2()

    train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_5_12t.csv")
    l_quest = [4,5,6,7,8,9,10,11,12,13]
    fich1()
    fich2()

    train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_13_22t.csv")
    l_quest = [14,15,16,17,18]
    fich1()
    fich2()
    feature_df.sort_values(by = 'quest', inplace=True)
    feature_df.to_csv("C:\\kaggle\\ОбучИгра\\feature.csv", index=False)

# 'nabor' - номер набора колонок; 'tip' - тип фичи: 't' - время, 'l' - количество
# 'kach' - чем больше тем лучше фича; rez - если не 0, то принята, место в трайне
# feature_df = pd.DataFrame(columns=['nabor', 'tip', 'quest', 'kol_col', 'col1', 'val1', 'col2', 'val2',
#                                    'col3', 'val3', 'kach', 'rez'])

# create_df_feature()

# ЗДЕСЬ ЗАКОНЧЕНО ПЕРВИЧНОЕ ФОРМИРОВАНИЕ ДАТАФРЕЙМА feature_df

def deftarget():
    global targets
    targets = pd.read_csv('C:\\kaggle\\ОбучИгра\\train_labels.csv')
    targets['q'] = targets['session_id'].apply(lambda x: int(x.split('_')[-1][1:]))
    targets['session'] = targets.session_id.apply(lambda x: int(x.split('_')[0]))

# _________________ ДАЛЕЕ ТЕСТИРОВАНИЕ И СОРТИРОВКА ФИЧЕЙ ПО КАЧЕСТВУ ______________

# создаем солонку 'delt_time' в train
def def_delt_time():
    global train
    train.sort_values(by=['session_id', 'elapsed_time'], inplace=True)
    train['delt_time'] = train['elapsed_time'].diff(1)
    train['delt_time'].fillna(0, inplace=True)
    train['delt_time'].clip(0, 100000, inplace=True)

# формируем трайн из 2 колонок - из одной строки датафрейма feature_df
def feature_engineer(row_f):
    global train, new_train
    # tip = row_f['tip']
    col1 = row_f['col1']
    val1 = row_f['val1']
    l = len(new_train.columns)
    if row_f['kol_col'] == 1: # если фича одна
        maska = (train[col1] == val1)
        new_train[f'{l}'] = train[maska].groupby(['session_id'])['delt_time'].sum()
        new_train[f'{l+1}'] = train[maska].groupby(['session_id'])['index'].count()
    elif row_f['kol_col'] == 2: # если фичей две
        col2 = row_f['col2']
        val2 = row_f['val2']
        maska = (train[col1] == val1) & (train[col2] == val2)
        new_train[f'{l}'] = train[maska].groupby(['session_id'])['delt_time'].sum()
        new_train[f'{l+1}'] = train[maska].groupby(['session_id'])['index'].count()

# формируем трайн из строк датафрейма feature_df, которые имеют 'rez' > 0
def feature_engineer_new(feature_rez_not_0):
    global train, new_train
    for i, row_f in feature_rez_not_0.iterrows():  # цикл по строкам feature_df относящимся к вопросу
        feature_engineer(row_f)

    # tip = row_f['tip']
    # col1 = row_f['col1']
    # val1 = row_f['val1']
    # if row_f['kol_col'] == 1: # если фича одна
    #     maska = (train[col1] == val1)
    #     new_train['1'] = train[maska].groupby(['session_id'])['delt_time'].sum()
    #     new_train['2'] = train[maska].groupby(['session_id'])['index'].count()
    # elif row_f['kol_col'] == 2: # если фичей две
    #     col2 = row_f['col2']
    #     val2 = row_f['val2']
    #     maska = (train[col1] == val1) & (train[col2] == val2)
    #     new_train['1'] = train[maska].groupby(['session_id'])['delt_time'].sum()
    #     new_train['2'] = train[maska].groupby(['session_id'])['index'].count()

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
        n_estimators = 40,
        learning_rate= 0.05,
        depth = 6,
        l2_leaf_reg = 1.4,
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
    for threshold in np.arange(best_thresholds[quest-1], best_thresholds[quest-1] + 2.5, 0.1):
        preds = (y_pred > threshold).astype('int')
        m = f1_score(tru, preds, average='macro')
        if m > best_score:
            best_score = m
            best_threshold = threshold
    b_thresholds += best_threshold
    kol_stuk += 1
    print('Результат для вопроса:', quest, 'F1 =', best_score, 'best_threshold =',
          best_threshold, 'средний best_threshold =', b_thresholds/kol_stuk)
    return best_score

    # print('Результат для вопроса:', quest, 'F1 =', m)
    # return m


def main_ml():
    global train, feature_df, quest, new_train, best_thresholds, kol_stuk, b_thresholds
    best_thresholds = [0.6, 0.85, 0.80]
    best_thresholds += [0.6, 0.55, 0.65, 0.65, 0.55, 0.65, 0.55, 0.55, 0.65, 0.4]
    best_thresholds += [0.6, 0.5, 0.6, 0.6, 0.7]
    best_thresholds += [0.55, 0.5, 0.6, 0.6, 0.7]

    # l_quest = [1, 2, 3]
    # l_quest = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    l_quest = [14, 15, 16, 17, 18]
    col = feature_df.columns
    if not ('kach1' in col):
        feature_df['kach1'] = 0
        feature_df['rez'] = 0
    for quest in l_quest: # цикл по вопросам
        b_thresholds = 0
        kol_stuk = 0
        maska = feature_df['quest'] == quest
        for i, row_f in feature_df[maska].iterrows(): # цикл по строкам feature_df относящимся к вопросу
            print('i=', i, 'kol_col=', row_f['kol_col'], 'col1=', row_f['col1'], row_f['val1'])
            new_train = pd.DataFrame(index=train['session_id'].unique(), columns=[])
            feature_engineer(row_f) # формируем трайн из 2 колонок - из одной строки датафрейма feature_df
            preds() # предсказание / модель
            m = otvet()
            feature_df.loc[i,'kach'] = m
            feature_df.loc[i, 'kach1'] = m
        feature_df.sort_values(by=['quest','kach'], ascending=False, inplace=True,)
        old_max = feature_df[maska]['kach'].max()
        feature_df.loc[maska & (feature_df['kach']==old_max), 'rez'] = 1
        print(feature_df[feature_df['quest']==quest].head(50))
        feature_df.to_csv("C:\\kaggle\\ОбучИгра\\feature7.csv", index=False)

    for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:#range(5,14,):

    # if not ('kach1' in col):
    #     feature_df['kach1'] = 0
        ###### feature_df['rez'] = 0
    # for quest in l_quest: # цикл по вопросам
    #     b_thresholds = 0
    #     kol_stuk = 0
    #     maska = feature_df['quest'] == quest
    #     for i, row_f in feature_df[maska].iterrows(): # цикл по строкам feature_df относящимся к вопросу
    #         print('i=', i, 'kol_col=', row_f['kol_col'], 'col1=', row_f['col1'], row_f['val1'])
    #         new_train = pd.DataFrame(index=train['session_id'].unique(), columns=[])
    #         feature_engineer(row_f) # формируем трайн из 2 колонок - из одной строки датафрейма feature_df
    #         preds() # предсказание / модель
    #         m = otvet()
    #         feature_df.loc[i,'kach'] = m
    #         feature_df.loc[i, 'kach1'] = m
    #     feature_df.sort_values(by=['quest','kach'], ascending=False, inplace=True,)
    #     old_max = feature_df[maska]['kach'].max()
    #     feature_df.loc[maska & (feature_df['kach']==old_max), 'rez'] = 1
    #     print(feature_df[feature_df['quest']==quest].head(50))
    #     feature_df.to_csv("C:\\kaggle\\ОбучИгра\\feature9.csv", index=False)

    # for k in []:#range(5,14,):
        kach = f'kach{k}'
        # if kach in col:
        #     continue
        # feature_df[kach] = 0
        for quest in l_quest:  # цикл по вопросам
            b_thresholds = 0
            kol_stuk = 0
            maska = feature_df['quest'] == quest
            new_train = pd.DataFrame(index=train['session_id'].unique(), columns=[])
            feature_engineer_new(feature_df[maska & (feature_df['rez'] > 0.5)])
            new_train0 = new_train.copy()
            for i, row_f in feature_df[maska].iterrows():  # цикл по строкам feature_df относящимся к вопросу
                if row_f [kach] < 0.001:
                    print('i=', i, 'kol_col=', row_f['kol_col'], 'col1=', row_f['col1'], row_f['val1'])
                    feature_engineer(row_f)  # формируем трайн из 2 колонок - из одной строки датафрейма feature_df
                    preds()  # предсказание / модель
                    m = otvet()
                    feature_df.loc[i, kach] = m
                    new_train = new_train0.copy()
            feature_df.sort_values(by=['quest', kach], ascending=False, inplace=True, )
            # old_max = feature_df[maska][kach].max()
            # feature_df[maska & (feature_df['rez'] < 0.5)].iloc[0,'rez'] = k
            ind = feature_df[maska & (feature_df['rez'] < 0.5)].index
            feature_df.loc[ind[0],'rez'] = k
            print(feature_df[feature_df['quest'] == quest].head(50))
            feature_df.to_csv("C:\\kaggle\\ОбучИгра\\feature7.csv", index=False)

# 1-Я СОРТИРОВКА СТОЛБЦОВ ПО КАЧЕСТВУ
feature_df = pd.read_csv("C:\\kaggle\\ОбучИгра\\feature6.csv")
# feature_df.to_csv("C:\\kaggle\\ОбучИгра\\feature9.csv", index=False)

# 1-Я СОРТИРОВКА СТОЛБЦОВ ПО КАЧЕСТВУ
feature_df = pd.read_csv("C:\\kaggle\\ОбучИгра\\feature8.csv")
# train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_0_4t.csv")
train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_13_22t.csv")
deftarget()
def_delt_time()
main_ml()



# train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_5_12t.csv")
# l_quest = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#
#
# train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_13_22t.csv")
# l_quest = [14, 15, 16, 17, 18]
