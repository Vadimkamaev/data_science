import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import f1_score
from servise_ds import okno

# считывание таргета из файла train_labels.csv
def deftarget():
    targets = pd.read_csv('C:\\kaggle\\ОбучИгра\\train_labels.csv')
    targets['q'] = targets['session_id'].apply(lambda x: int(x.split('_')[-1][1:]))
    targets['session'] = targets.session_id.apply(lambda x: int(x.split('_')[0]))
    question_means = targets.groupby('q').correct.agg('mean')
    return targets

def feature_engineer(train):
    # имена, используемых в модели, категориальных полей трайна
    CATS = ['event_name', 'fqid', 'room_fqid', 'text_fqid', 'level', 'page']
    # имена, используемых в модели, числовых полей трайна
    # NUMS = ['delt_time', 'room_coor_x', 'room_coor_y', 'hover_duration']
    NUMS = ['delt_time', 'hover_duration']
    EV_NAME = ['checkpoint', 'observation_click', 'cutscene_click', 'notification_click', 'person_click',
               'object_click', 'map_click', 'object_hover']
    new_train = pd.DataFrame(index=train['session_id'].unique(), columns=[])
    for c in EV_NAME:
        new_train['l_ev_name_' + c] = train[train['event_name'] == c].groupby(['session_id'])['index'].count()
        new_train['t_ev_name_' + c] = train[train['event_name'] == c].groupby(['session_id'])['delt_time'].sum()
    train = train[train['name'] == 'basic']
    new_train['finish'] = train.groupby(['session_id'])['elapsed_time'].last(1)  # ? надо ли?
    new_train['len'] = train.groupby(['session_id'])['index'].count()
    for c in CATS:
        tmp = train.groupby(['session_id'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        new_train = new_train.join(tmp)
    for c in NUMS:
        tmp = train.groupby(['session_id'])[c].agg('mean')
        new_train = new_train.join(tmp)
    for c in NUMS:
        tmp = train.groupby(['session_id'])[c].agg('std')
        tmp.name = tmp.name + '_std'
        new_train = new_train.join(tmp)
    new_train = new_train.fillna(-1)
    return new_train

def dop_feature(new_train, train, col, param):
    new_train['l_'+col+' '+param] = train[train[col]==param].groupby(['session_id'])['index'].count()
    new_train['t_' + col + ' ' + param] = train[train[col] == param].groupby(['session_id'])['delt_time'].sum()
    return new_train

def one_vopros(df, train_index, targets, test_index, models, oof, t, FEATURES, param):
    global gb_param
    # TRAIN DATA
    train_x = df.iloc[train_index]
    train_users = train_x.index.values
    train_y = targets.loc[targets.q == t].set_index('session').loc[train_users]

    # VALID DATA
    valid_x = df.iloc[test_index]
    valid_users = valid_x.index.values
    valid_y = targets.loc[targets.q == t].set_index('session').loc[valid_users]

    # TRAIN MODEL
    model = xgb.XGBClassifier(
        tree_method="hist",
        objective= 'binary:logistic',
        n_estimators = gb_param[t][0],
        max_depth = gb_param[t][1],
        learning_rate = gb_param[t][2], #param, #0.057,
        alpha=8,
        subsample= 0.4,
        colsample_bytree=0.8,
        seed = 40,
        # max_bin=4096,
        n_jobs=2
    )
    X = train_x[FEATURES].astype('float32')
    Y = train_y['correct']
    model.fit(X, Y)

    # SAVE MODEL, PREDICT VALID OOF
    models[f'{t}'] = model
    oof.loc[valid_users, t] = model.predict_proba(valid_x[FEATURES].astype('float32'))[:, 1]
    return models, oof

def preds(df, targets, param):
    global quests
    FEATURES = [c for c in df.columns if c != 'level_group']
    ALL_USERS = df.index.unique()
    print('В трайне', len(FEATURES), 'колонок')
    print('В трайне', len(ALL_USERS), 'пользователей')

    gkf = GroupKFold(n_splits=5)
    oof = pd.DataFrame(data=np.zeros((len(ALL_USERS), 19)), index=ALL_USERS)
    models = {}
    # ВОПРОСЫ
    for q in quests:
        print('### quest', q, '==> Fold ==>', end='')
        # ВЫЧИСЛИТЕ РЕЗУЛЬТАТ С 5-ГРУППОВЫМ K FOLD
        for i, (train_index, test_index) in enumerate(gkf.split(X=df, groups=df.index)):
            print(' ', i + 1, end='')
            models, oof = one_vopros(df, train_index, targets, test_index, models, oof, q, FEATURES, param)
        print()

    # ВСТАВЬТЕ ИСТИННЫЕ МЕТКИ В ФРЕЙМ ДАННЫХ С 18 СТОЛБЦАМИ
    true = oof.copy()
    for k in quests:
        # GET TRUE LABELS
        tmp = targets.loc[targets.q == k].set_index('session').loc[ALL_USERS]
        true[k] = tmp.correct.values
    return oof, true

def otvet(oof, true, param, param2, rezult):
    global quests
    #ВЫЧИСЛЕНИЕ ПОРОГА ДЛЯ КАЖДОГО ВОПРОСА ОТДЕЛЬНО
    scores = []; thresholds = []
    # for k in quests:
    #     best_score = 0
    #     best_threshold = 0
    #     for threshold in np.arange(-0.01, 1.01, 0.01):
    #         # print(f'{threshold:.02f}, ',end='')
    #         preds = (oof[k].values > threshold).astype('int')
    #         tru = true[k].values.reshape((-1))
    #         m = f1_score(tru, preds, average='macro')
    #         if m > best_score:
    #             best_score = m
    #             best_threshold = threshold
    #     thresholds.append(best_threshold)
    #     print('')
    #     print(f'Вопрос № {k+1} best_score = {best_score} best_threshold = {best_threshold}')

    best_threshold = 0.61

    print('Результат для каждого вопроса с общим порогом:')
    for k in quests:
        # Считаем F1 SCORE для каждого вопроса
        tru = true[k].values
        y_pred = oof[k].values
        m = f1_score(tru, (y_pred > best_threshold).astype('int'), average='macro')
        print(f'Q{k}: F1 =', m)

    # Считаем F1 SCORE для всех вопросов
    tru3 = true[quests]
    tru = tru3.values.reshape((-1))
    oof3 = oof[quests]
    y_pred = oof3.values.reshape((-1))
    m = f1_score(tru, (y_pred > best_threshold).astype('int'), average='macro')
    print('==> Для всех вопросов =', m)


    rezult.loc[len(rezult.index)] = [param, param2, m]

    # print('Результат для каждого вопроса с индивидуальным порогом:')
    # for k in quests:
    #     # Считаем F1 SCORE для каждого вопроса
    #     tru = true[k].values
    #     y_pred = oof[k].values
    #     m = f1_score(tru, (y_pred > thresholds[k-quests[0]]).astype('int'), average='macro')
    #     print(f'Q{k}: F1 =', m)

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


def main():
    global quests, gb_param
    vse_quests = [4, 5, 6, 7, 8, 9, 10, 11]#, 12, 13]
    gb_param = {4: [300, 5, 0.07],
                5: [200, 3, 0.06],
                6: [360, 3, 0.065],
                7: [410, 5, 0.055],
                8: [100, 4, 0.065],
                9: [250, 5, 0.06],
                10:[510, 5, 0.075],
                11:[150, 5, 0.045],
                12:[660, 7, 0.095],
                13:[670, 11, 0.085]
              }      #1-й элемент списка n_estimators, 2-й max_depth,
    rezult = pd.DataFrame(columns=['param', 'quest', 'rezultat'])
    train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_5_12.csv")
    targets = deftarget()
    # ТЕСТИРУЕМЫЕ ПАРАМЕТРЫ
    # col = 'event_name' # перебор колонок для трайна
    # ls = train[col].unique() # список значений колонки

    # for quest in vse_quests:
    #     quests = [quest]

    if True:
        quests = vse_quests
        quest = 0

        for param in range(5, 6, 5):
            param /= 1000
            train.sort_values(by=['session_id', 'elapsed_time'], inplace=True)
            train['delt_time'] = train['elapsed_time'].diff(1)
            train['delt_time'].fillna(0, inplace=True)
            train['delt_time'].clip(0, 103000, inplace=True)
            new_train = feature_engineer(train)

            # new_train = dop_feature(new_train, train, col, param)

            oof, true = preds(new_train, targets, param)
            otvet(oof, true, param, quest, rezult)
            rezult.sort_values(by = 'rezultat', inplace=True, ascending=False)
            print(rezult.head(22))
            for q in rezult.quest.unique():
                print('вопрос =', q)
                print(rezult[rezult.quest==q].head())


quests = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

if __name__ == "__main__":
    main()