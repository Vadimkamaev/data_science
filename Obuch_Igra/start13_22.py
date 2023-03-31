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

def feature_quest(new_train, train, q):
    rooms = {14: ['tunic.historicalsociety.closet_dirty'],  # 0.611685
             15: ['tunic.historicalsociety.stacks', 'tunic.flaghouse.entry',
             'tunic.historicalsociety.frontdesk'],  # 0.580154
             16: ['tunic.historicalsociety.closet_dirty', 'tunic.library.microfiche',
                  'tunic.historicalsociety.cage'],  # 0.514063
             17: ['tunic.kohlcenter.halloffame'],  # 0.544310
             18: ['tunic.drycleaner.frontdesk']  # 0.503757
             }

    train_q = new_train.copy()
    for room in rooms[q]:
        train_q['l_room_' + room] = train[train['room_fqid'] == room].groupby(['session_id'])['index'].count()
        train_q['t_room_' + room] = train[train['room_fqid'] == room].groupby(['session_id'])['delt_time'].sum()
    return train_q

def dop_feature(new_train, train, col, param):
    new_train['l_'+col+' '+str(param)] = train[train[col]==param].groupby(['session_id'])['index'].count()
    new_train['t_' + col + ' ' + str(param)] = train[train[col] == param].groupby(['session_id'])['delt_time'].sum()
    return new_train

def one_vopros(df, train_index, targets, test_index, models, oof, t, param):
    global gb_param
    # print(t, ', ', end='')
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
        n_estimators = gb_param[t][0],#param,
        max_depth = gb_param[t][1],#5,
        learning_rate = gb_param[t][2],# 0.057,
        alpha=8,
        subsample= 0.4,
        colsample_bytree=0.8,
        seed = 40,
        # max_bin=4096,
        n_jobs=2
    )
    X = train_x.astype('float32')
    Y = train_y['correct']
    model.fit(X, Y)

    # SAVE MODEL, PREDICT VALID OOF
    models[f'{t}'] = model
    oof.loc[valid_users, t] = model.predict_proba(valid_x.astype('float32'))[:, 1]
    return models, oof

def preds(new_train, train, targets, param):
    global quests
    ALL_USERS = new_train.index.unique()
    # print('В трайне', len(train.columns), 'колонок')
    print('В трайне', len(ALL_USERS), 'пользователей')
    gkf = GroupKFold(n_splits=5)
    oof = pd.DataFrame(data=np.zeros((len(ALL_USERS), 19)), index=ALL_USERS)
    models = {}
    # ВОПРОСЫ
    for q in quests:
        print('### quest', q, '==> Fold ==>', end='')
        train_q = feature_quest(new_train, train, q)

        # ВЫЧИСЛИТЕ РЕЗУЛЬТАТ С 5-ГРУППОВЫМ K FOLD
        for i, (train_index, test_index) in enumerate(gkf.split(X=train_q, groups=train_q.index)):
            print(' ', i + 1, end='')
            models, oof = one_vopros(train_q, train_index, targets, test_index, models, oof, q, param)
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
    vse_quests = [14, 15, 16, 17, 18]
    gb_param = {14: [275, 5, 0.057], # 0.6105
                15: [540, 5, 0.05], # 0.5768
                16: [695, 6, 0.07], # 0.5067
                17: [325, 5, 0.06], # 0.5429
                18: [505, 5, 0.07] # 0.501
              }      #1-й элемент списка n_estimators, 2-й max_depth,
    rooms = {14: ['tunic.historicalsociety.closet_dirty', # 0.611685
                  'tunic.historicalsociety.collection_flag', # 0.610912
                  'tunic.library.frontdesk'], # 0.610788
             15: ['tunic.historicalsociety.stacks', # 0.579155
                  'tunic.flaghouse.entry', # 0.578392
                  'tunic.historicalsociety.frontdesk'], # 0.577933
             16: ['tunic.historicalsociety.closet_dirty', # 0.511145
                  'tunic.library.microfiche', # 0.510837
                  'tunic.historicalsociety.cage'], # 0.509886
             17: ['tunic.kohlcenter.halloffame'], # 0.544310
             18: ['tunic.drycleaner.frontdesk', # 0.506152
                  'tunic.library.microfiche'] # 0.503757
            }
    text_fqids = {

    }
    rezult = pd.DataFrame(columns=['param', 'quest', 'rezultat'])
    train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_13_22.csv")
    targets = deftarget()
    # ТЕСТИРУЕМЫЕ ПАРАМЕТРЫ
    col = 'text_fqid' # перебор колонок для трайна
    ls = train[col].unique() # список значений колонки

    for quest in vse_quests:
        quests = [quest]

    # if True:
    #     quests = vse_quests
    #     quest = 0

        for param in ls:
            # param = str(param)
            train.sort_values(by=['session_id', 'elapsed_time'], inplace=True)
            train['delt_time'] = train['elapsed_time'].diff(1)
            train['delt_time'].fillna(0, inplace=True)
            train['delt_time'].clip(0, 103000, inplace=True)
            new_train = feature_engineer(train)

            new_train = dop_feature(new_train, train, col, param)

            oof, true = preds(new_train, train, targets, param)
            otvet(oof, true, param, quest, rezult)
            rezult.sort_values(by = 'rezultat', inplace=True, ascending=False)
            print(rezult.head(22))
            for q in rezult.quest.unique():
                print('вопрос =', q)
                print(rezult[rezult.quest==q].head(10))


quests = [14,15,16,17,18]

if __name__ == "__main__":
    main()