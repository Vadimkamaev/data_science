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
    return targets


def feature_engineer(train):
    # имена, используемых в модели, категориальных полей трайна
    CATS = ['event_name', 'fqid', 'room_fqid', 'text_fqid', 'level', 'page']
    # имена, используемых в модели, числовых полей трайна
    NUMS = ['delt_time', 'room_coor_x', 'room_coor_y', 'hover_duration']
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
    new_train['l_' + col + ' ' + param] = train[train[col] == param].groupby(['session_id'])['index'].count()
    new_train['t_' + col + ' ' + param] = train[train[col] == param].groupby(['session_id'])['delt_time'].sum()
    return new_train


def one_vopros(df, train_index, targets, test_index, models, t, FEATURES, param):
    print(t, ', ', end='')
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
        objective='binary:logistic',
        n_estimators=510,
        max_depth=5,  # param,
        learning_rate=0.057,
        alpha=8,
        subsample=0.4,
        colsample_bytree=0.8,
        seed=40,
        # max_bin=4096,
        n_jobs=2
    )
    X = train_x[FEATURES].astype('float32')
    Y = train_y['correct']
    model.fit(X, Y)

    # SAVE MODEL, PREDICT VALID preds_user_x_vopros
    models[f'{t}'] = model
    preds_user_x_vopros.loc[valid_users, t] = model.predict_proba(valid_x[FEATURES].astype('float32'))[:, 1]
    return models


def preds(train, targets, param):
    global quests, preds_user_x_vopros
    FEATURES = [c for c in train.columns if c != 'level_group']
    ALL_USERS = train.index.unique()
    print('В трайне', len(FEATURES), 'колонок')
    print('В трайне', len(ALL_USERS), 'пользователей')

    gkf = GroupKFold(n_splits=5)
    models = {}

    # ВЫЧИСЛИТЕ РЕЗУЛЬТАТ С 5-ГРУППОВЫМ K FOLD
    for i, (train_index, test_index) in enumerate(gkf.split(X=train, groups=train.index)):
        print('### Fold', i + 1, '==>', end='')
        # ПРОВЕРЬТЕ ВОПРОСЫ С 1 ПО 3
        for t in quests:
            models = one_vopros(train, train_index, targets, test_index, models, t, FEATURES, param)
        print()

    # ВСТАВЬТЕ ИСТИННЫЕ МЕТКИ В ФРЕЙМ ДАННЫХ С 18 СТОЛБЦАМИ
    true = preds_user_x_vopros.copy()
    for k in quests:
        # GET TRUE LABELS
        tmp = targets.loc[targets.q == k].set_index('session').loc[ALL_USERS]
        true[k] = tmp.correct.values
    return true


def otvet(true, param, rezult):
    global quests, preds_user_x_vopros
    # ВЫЧИСЛЕНИЕ ПОРОГА ДЛЯ КАЖДОГО ВОПРОСА ОТДЕЛЬНО
    scores = [];
    thresholds = []
    for k in quests:
        best_score = 0
        best_threshold = 0
        for threshold in np.arange(-0.01, 1.01, 0.01):
            # print(f'{threshold:.02f}, ',end='')
            pred_s = (preds_user_x_vopros[k].values > threshold).astype('int')
            tru = true[k].values.reshape((-1))
            m = f1_score(tru, pred_s, average='macro')
            if m > best_score:
                best_score = m
                best_threshold = threshold
        thresholds.append(best_threshold)
        print('')
        print(f'Вопрос № {k + 1} best_score = {best_score} best_threshold = {best_threshold}')

    best_threshold = 0.61

    print('Результат для каждого вопроса с общим порогом:')
    for k in quests:
        # Считаем F1 SCORE для каждого вопроса
        tru = true[k].values
        y_pred = preds_user_x_vopros[k].values
        m = f1_score(tru, (y_pred > best_threshold).astype('int'), average='macro')
        print(f'Q{k}: F1 =', m)

    # Считаем F1 SCORE для всех вопросов
    tru3 = true[quests]
    tru = tru3.values.reshape((-1))
    oof3 = preds_user_x_vopros[quests]
    y_pred = oof3.values.reshape((-1))
    m = f1_score(tru, (y_pred > best_threshold).astype('int'), average='macro')
    print('==> Для всех вопросов =', m)

    rezult.loc[len(rezult.index)] = [param, 0, m]

    print('Результат для каждого вопроса с индивидуальным порогом:')
    for k in quests:
        # Считаем F1 SCORE для каждого вопроса
        tru = true[k].values
        y_pred = preds_user_x_vopros[k].values
        m = f1_score(tru, (y_pred > thresholds[k - quests[0]]).astype('int'), average='macro')
        print(f'Q{k}: F1 =', m)


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
               'hover_duration', 'fqid', 'room_fqid', 'text_fqid', 'level_group']
    train = pd.read_csv(file, dtype=dtypes, usecols=use_col)
    return train


def main():
    global quests, preds_user_x_vopros
    rezult = pd.DataFrame(columns=['param', 'param2', 'rezultat'])
    train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train.csv")
    targets = deftarget()

    # ТЕСТИРУЕМЫЕ ПАРАМЕТРЫ
    col = 'event_name'  # перебор колонок для трайна
    ls = train[col].unique()  # список значений колонки

    train.sort_values(by=['session_id', 'elapsed_time'], inplace=True)
    train['delt_time'] = train['elapsed_time'].diff(1)
    train['delt_time'].fillna(0, inplace=True)
    train['delt_time'].clip(0, 103000, inplace=True)

    ALL_USERS = train.session_id.unique()

    for param in [1]:  # ls: #range(1, 2, 1):

        preds_user_x_vopros = pd.DataFrame(data=np.zeros((len(ALL_USERS), 19)), index=ALL_USERS)

        quests = [1]
        new_train = feature_engineer(train[train['level_group'] == '0-4'])
        # new_train = dop_feature(new_train, train[train['level_group'] == '0-4'], col, param)

        true = preds(new_train, targets, param)

        quests = [4,5,6,7,8,9,10,11]
        new_train = feature_engineer(train[train['level_group'] == '5-12'])
        # new_train = dop_feature(new_train, train[train['level_group'] == '5-12'], col, param)
        true = preds(new_train, targets, param)

        quests = [14, 15, 16, 17]
        new_train = feature_engineer(train[train['level_group'] == '13-22'])
        # new_train = dop_feature(new_train, train[train['level_group'] == '13-22'], col, param)
        true = preds(new_train, targets, param)

        quests = [1] + [4,5,6,7,8,9,10,11] + [14, 15, 16, 17]
        otvet(true, param, rezult)
    rezult.sort_values(by='rezultat', inplace=True, ascending=False)
    print(rezult.head(22))


# quests = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
kol_quest = len(quests)
if __name__ == "__main__":
    main()