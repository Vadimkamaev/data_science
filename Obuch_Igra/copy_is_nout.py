import pandas as pd, numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb

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

df13_22 = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_13_22.csv")
print(df13_22.shape)

targets = pd.read_csv('C:\\kaggle\\ОбучИгра\\train_labels.csv')
targets['session'] = targets.session_id.apply(lambda x: int(x.split('_')[0]) )
targets['q'] = targets.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )
print( targets.shape )
targets.head()

def delt_time_def(df):
    df.sort_values(by=['session_id', 'elapsed_time'], inplace=True)
    df['delt_time'] = df['elapsed_time'].diff(1)
    df['delt_time'].fillna(0, inplace=True)
    df['delt_time'].clip(0, 103000, inplace=True)
    return df


def feature_engineer(train):
    # имена, используемых в модели, категориальных полей трайна
    CATS = ['event_name', 'fqid', 'room_fqid', 'text_fqid', 'level', 'page']
    # имена, используемых в модели, числовых полей трайна
    #     NUMS = ['delt_time', 'room_coor_x', 'room_coor_y', 'hover_duration']
    NUMS = ['delt_time', 'hover_duration']
    EV_NAME = ['checkpoint', 'observation_click', 'cutscene_click', 'notification_click', 'person_click',
               'object_click', 'map_click', 'object_hover']
    new_train = pd.DataFrame(index=train['session_id'].unique(), columns=[])
    for c in EV_NAME:
        new_train['l_ev_name_' + c] = train[train['event_name'] == c].groupby(['session_id'])['index'].count()
        new_train['t_ev_name_' + c] = train[train['event_name'] == c].groupby(['session_id'])['delt_time'].sum()
    maska = train['name'] == 'basic'

    new_train['finish'] = train[maska].groupby(['session_id'])['elapsed_time'].last(1)  # ? надо ли?
    new_train['len'] = train[maska].groupby(['session_id'])['index'].count()
    for c in CATS:
        tmp = train[maska].groupby(['session_id'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        new_train = new_train.join(tmp)
    for c in NUMS:
        tmp = train[maska].groupby(['session_id'])[c].agg('mean')
        new_train = new_train.join(tmp)
    for c in NUMS:
        tmp = train[maska].groupby(['session_id'])[c].agg('std')
        tmp.name = tmp.name + '_std'
        new_train = new_train.join(tmp)
    new_train = new_train.fillna(-1)
    return new_train


def feature_quest(new_train, train, q):
    rooms = {14: ['tunic.historicalsociety.closet_dirty'],
             15: ['tunic.historicalsociety.stacks', 'tunic.flaghouse.entry',
                  'tunic.historicalsociety.frontdesk'],
             16: ['tunic.historicalsociety.closet_dirty', 'tunic.library.microfiche',
                  'tunic.historicalsociety.cage'],
             17: ['tunic.kohlcenter.halloffame'],
             18: ['tunic.drycleaner.frontdesk']
             }

    train_q = new_train.copy()
    for room in rooms[q]:
        train_q['l_room_' + room] = train[train['room_fqid'] == room].groupby(['session_id'])['index'].count()
        train_q['t_room_' + room] = train[train['room_fqid'] == room].groupby(['session_id'])['delt_time'].sum()

    text_fqids = {
        14: ['tunic.historicalsociety.frontdesk.archivist_glasses.confrontation_recap'],  # 0.6182001353192074
        15: ['tunic.historicalsociety.entry.groupconvo_flag',
             'tunic.flaghouse.entry.colorbook',  # F1 = 0.5904442045737962
             'tunic.historicalsociety.entry.boss.flag'],  # 0.5982452417971184
        16: ['tunic.historicalsociety.entry.boss.flag',  # 0.508959523084263
             'tunic.historicalsociety.entry.wells.flag',  # 0.5115978995166816
             'tunic.historicalsociety.cage.teddy.trapped',  # 0.5138155041323106
             'tunic.historicalsociety.basement.savedteddy'],
        17: ['tunic.historicalsociety.frontdesk.key'],  # 0.5432409135067107
        18: ['tunic.capitol_2.hall.boss.haveyougotit',  # 0.5030042
             'tunic.flaghouse.entry.flag_girl.symbol_recap',  # ] # 0.50779819
             'tunic.library.frontdesk.worker.flag']
    }
    for text_fqid in text_fqids[q]:
        train_q['l_text_fqid_' + text_fqid] = train[train['text_fqid'] == text_fqid].groupby(['session_id'])[
            'index'].count()
        train_q['t_text_fqid_' + text_fqid] = train[train['text_fqid'] == text_fqid].groupby(['session_id'])[
            'delt_time'].sum()
    #         print('1')
    return train_q


def create_model(train, old_train, quests, models):
    kol_quest = len(quests)
    ALL_USERS = train.index.unique()
    print('We will train with', len(ALL_USERS), 'users info')

    gb_param = {4: [300, 5, 0.07],
                5: [200, 3, 0.06],
                6: [360, 3, 0.065],
                7: [410, 5, 0.055],
                8: [100, 4, 0.065],
                9: [250, 5, 0.06],
                10: [510, 5, 0.075],
                11: [150, 5, 0.045],
                12: [660, 7, 0.095],
                13: [670, 11, 0.085],

                14: [275, 5, 0.057],
                15: [510, 5, 0.057],
                16: [510, 5, 0.057],
                17: [510, 5, 0.057],
                18: [510, 5, 0.057],
                }

    gkf = GroupKFold(n_splits=5)
    oof = pd.DataFrame(data=np.zeros((len(ALL_USERS), kol_quest)), columns=quests, index=ALL_USERS)

    # ITERATE THRU QUESTIONS
    for q in quests:
        print('### quest', q, '==> Fold ==>', end='')

        train_q = feature_quest(train, old_train, q)
        #         train_q = train
        train_q.to_csv('C:\\kaggle\\ОбучИгра\\train_pycarm.csv', index=False)

        # ВЫЧИСЛИТЕ РЕЗУЛЬТАТ CV С 5-ГРУППОВЫМ K-СКЛАДОМ
        for i, (train_index, test_index) in enumerate(gkf.split(X=train_q, groups=train_q.index)):
            print(' ', i + 1, end='')

            # TRAIN DATA
            train_x = train_q.iloc[train_index]

            train_users = train_x.index.values
            train_y = targets.loc[targets.q == q].set_index('session').loc[train_users]

            # VALID DATA
            valid_x = train_q.iloc[test_index]

            valid_users = valid_x.index.values
            #             valid_y = targets.loc[targets.q==q].set_index('session').loc[valid_users]

            # TRAIN MODEL
            model = xgb.XGBClassifier(
                tree_method="hist",
                objective='binary:logistic',
                n_estimators=gb_param[q][0],  # 510,
                max_depth=gb_param[q][1],  # 5,
                learning_rate=gb_param[q][2],  # 0.057,
                alpha=8,
                subsample=0.4,
                colsample_bytree=0.8,
                seed=40,
                # max_bin=4096,
            )

            model.fit(train_x.astype('float32'), train_y['correct'])

            # SAVE MODEL, PREDICT VALID OOF
            models[f'{q}'] = model
            oof.loc[valid_users, q] = model.predict_proba(valid_x.astype('float32'))[:, 1]
        print()

    # PUT TRUE LABELS INTO DATAFRAME
    true = oof.copy()
    for q in quests:  # range(kol_quest):
        # GET TRUE LABELS
        tmp = targets.loc[targets.q == q].set_index('session').loc[ALL_USERS]
        true[q] = tmp.correct.values

    print('When using optimal threshold...')
    for q in quests:  # range(kol_quest):

        # COMPUTE F1 SCORE PER QUESTION
        #         m = f1_score(true[k].values, (oof[q].values>best_thresholds[q]).astype('int'), average='macro')
        m = f1_score(true[q].values, (oof[q].values > best_threshold).astype('int'), average='macro')
        print(f'Q{q}: F1 =', m)

    # COMPUTE F1 SCORE OVERALL
    tru3 = true[quests]
    tru = tru3.values.reshape((-1))
    oof3 = oof[quests]
    y_pred = oof3.values.reshape((-1))
    m = f1_score(tru, (y_pred > best_threshold).astype('int'), average='macro')

    # m = f1_score(true[quests].values.reshape((-1)), (oof[quests].values.reshape((-1))>best_threshold).astype('int'), average='macro')
    print('==> Overall F1 =', m)

    return models

models = {}
best_threshold = 0.61

df13_22 = delt_time_def(df13_22)
train= feature_engineer(df13_22)
print(train.shape)
quests = [14]
models = create_model(train, df13_22, quests, models)
