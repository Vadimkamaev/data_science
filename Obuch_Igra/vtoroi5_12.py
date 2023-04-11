import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
import xgboost as xgb
from catboost import CatBoostClassifier
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
    CATS = ['event_name', 'fqid', 'room_fqid', 'text_fqid', 'page']
    # имена, используемых в модели, числовых полей трайна
    # NUMS = ['delt_time', 'room_coor_x', 'room_coor_y', 'hover_duration']
    NUMS = ['delt_time', 'hover_duration']
    EV_NAME = ['observation_click', 'cutscene_click', 'notification_click', 'person_click',
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
    train_q = new_train.copy()
    # rooms = {4: [], # 0.645516
    #          5: ['tunic.historicalsociety.frontdesk'],  # 0.610148
    #          6: ['tunic.drycleaner.frontdesk'],  # 0.614074
    #          7: ['tunic.historicalsociety.frontdesk'], # 0.593498
    #          8: [], # 0.559517
    #          9: ['tunic.historicalsociety.entry'], #0.607111
    #          10:['tunic.library.frontdesk', # 0.571301
    #              'tunic.historicalsociety.frontdesk'], # 0.5729109   ХОРОШО
    #          11:[] # 0.596917
    #         }
    # for room in rooms[q]:
    #     train_q['l_room_' + room] = train[train['room_fqid'] == room].groupby(['session_id'])['index'].count()
    #     train_q['t_room_' + room] = train[train['room_fqid'] == room].groupby(['session_id'])['delt_time'].sum()
    #
    # levels ={4: [12], #0.6464931109093858
    #          5: [], #
    #          6: [9], # 0.6163670083090893
    #          7: [11, 8], #0.5955192949527838
    #          8: [12], # 0.5589346712660985
    #          9: [9], # 0.6059824858104659
    #          10:[7], # 0.5790869708333823
    #          11:[7]  # 0.5948890284013294
    #          }
    # for level in levels[q]:
    #     train_q['l_level_' + str(level)] = train[train['level'] == level].groupby(['session_id'])['index'].count()
    #     train_q['t_level_' + str(level)] = train[train['level'] == level].groupby(['session_id'])['delt_time'].sum()
    #
    text_fqids = {
        4: ['tunic.historicalsociety.frontdesk.archivist.newspaper',
            'tunic.humanecology.frontdesk.worker.intro',
            'tunic.library.frontdesk.worker.wells', # 0.6666325627660743
            'tunic.library.frontdesk.worker.hello'], # 0.6678694174620372
        5: ['tunic.humanecology.frontdesk.worker.intro',
            'tunic.historicalsociety.closet_dirty.gramps.helpclean',
            'tunic.historicalsociety.closet_dirty.gramps.news'],     # 0.6225926406619734
        6: ['tunic.humanecology.frontdesk.worker.intro',
            'tunic.historicalsociety.frontdesk.archivist.foundtheodora',
            'tunic.historicalsociety.closet_dirty.trigger_coffee', # 0.6298310348680769
            'tunic.historicalsociety.closet_dirty.gramps.archivist'], # 0.6320710506038789
        7: ['tunic.historicalsociety.closet_dirty.door_block_talk',
            'tunic.drycleaner.frontdesk.worker.hub',
            'tunic.historicalsociety.closet_dirty.trigger_coffee', # 0.6133391068274732
            'tunic.library.frontdesk.block_badge_2'],              # 0.6137352401264455
        8: ['tunic.humanecology.frontdesk.worker.intro',
            'tunic.historicalsociety.frontdesk.magnify', # 0.565820361974706
            'tunic.historicalsociety.closet_dirty.trigger_coffee'], # 0.5666316505932836
        9: ['tunic.historicalsociety.frontdesk.archivist.hello',
            'tunic.library.frontdesk.worker.wells', # 0.6123618449755199
            'tunic.historicalsociety.frontdesk.archivist.foundtheodora'], # 0.6165404455938354
        10: ['tunic.library.frontdesk.worker.wells',
            'tunic.historicalsociety.frontdesk.archivist.have_glass_recap',
             'tunic.historicalsociety.closet_dirty.gramps.news'], # 0.5829876555092278
        11: ['tunic.historicalsociety.frontdesk.archivist.newspaper_recap',
             'tunic.historicalsociety.closet_dirty.gramps.archivist'] # 0.5990726954508437
    }
    for text_fqid in text_fqids[q]:
        maska = train['text_fqid'] == text_fqid
        train_q['l_text_fqid_' + text_fqid] = train[maska].groupby(['session_id'])['index'].count()
        train_q['t_text_fqid_' + text_fqid] = train[maska].groupby(['session_id'])['delt_time'].sum()
        # train_q['x_' + text_fqid] = train[maska].groupby(['session_id'])['room_coor_x'].mean()
        # train_q['y_' + text_fqid] = train[maska].groupby(['session_id'])['room_coor_y'].mean()
        # maska = maska & (train['name'] == 'basic')
        # train_q['l1_' + text_fqid] = train[maska].groupby(['session_id'])['index'].count()
        # train_q['t1_' + text_fqid] = train[maska].groupby(['session_id'])['delt_time'].sum()

    room_lvls = {
         4: [['tunic.historicalsociety.frontdesk',12], # 0.6597916057422418
             ['tunic.historicalsociety.stacks',7]], # 0.6601234742102491
         5: [['tunic.historicalsociety.stacks',12]],  # 0.6224656333644798
             # ['',]],
         6: [['tunic.drycleaner.frontdesk',8],  # 0.6213566138088428
             ['tunic.library.microfiche',9]], # 0.6230008195007571
         7: [['tunic.drycleaner.frontdesk',8], # 0.5996180477326464
             ['tunic.library.frontdesk',10]], # 0.602582704782065
         8: [['tunic.kohlcenter.halloffame', 11], # 0.5643078050912897
             ['tunic.kohlcenter.halloffame',6]], # 0.5649049719407627
         9: [['tunic.capitol_1.hall', 12], # 0.6126018162800113
             ['tunic.historicalsociety.collection',12]],
         10:[['tunic.kohlcenter.halloffame',5], # 0.5736735291344757
             ['tunic.humanecology.frontdesk',7]], # 0.580172783522924
         11:[['tunic.drycleaner.frontdesk',9], #0.5982855685463792
             ['tunic.historicalsociety.collection',6]] # 0.5985825159801804
        }
    for rl in room_lvls[q]:
        nam = rl[0]+str(rl[1])
        maska = (train['room_fqid'] == rl[0])&(train['level'] == rl[1])
        train_q['l_' + nam] = train[maska].groupby(['session_id'])['index'].count()
        train_q['t_' + nam] = train[maska].groupby(['session_id'])['delt_time'].sum()

        # train_q['hd_' + nam] = train[maska].groupby(['session_id'])['hover_duration'].sum()

        # train_q['x_' + nam] = train[maska].groupby(['session_id'])['room_coor_x'].mean()
        # train_q['y_' + nam] = train[maska].groupby(['session_id'])['room_coor_y'].mean()
    return train_q



def dop_feature(new_train, train, col, param):
    new_train['l_'+col+' '+str(param)] = train[train[col]==param].groupby(['session_id'])['index'].count()
    new_train['t_' + col + ' ' + str(param)] = train[train[col] == param].groupby(['session_id'])['delt_time'].sum()
    return new_train

def dop_feature2(new_train, train, col, param, col2, param2):
    new_train['l_'+str(param)+str(param2)] = \
        train[(train[col]==param)&(train[col2]==param2)].groupby(['session_id'])['index'].count()
    new_train['t_'+str(param)+str(param2)] = \
        train[(train[col]==param)&(train[col2]==param2)].groupby(['session_id'])['delt_time'].sum()
    return new_train

def one_vopros(df, train_index, targets, test_index, models, oof, t, param, param2):
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
    model = CatBoostClassifier(
        n_estimators = 280,#param
        learning_rate= 0.045,#param2,
        depth = 3
    )

    #     xgb.XGBClassifier(
    #     tree_method="hist",
    #     objective= 'binary:logistic',
    #     n_estimators = gb_param[t][0],#,
    #     max_depth = gb_param[t][1],#5,
    #     learning_rate = gb_param[t][2],# 0.057,
    #     alpha=8,
    #     subsample= 0.4,
    #     colsample_bytree=0.8,
    #     seed = 40,
    #     # max_bin=4096,
    #     n_jobs=2
    # )
    X = train_x.astype('float32')
    Y = train_y['correct']
    model.fit(X, Y, verbose=False)

    # SAVE MODEL, PREDICT VALID OOF
    models[f'{t}'] = model
    oof.loc[valid_users, t] = model.predict_proba(valid_x.astype('float32'))[:, 1]
    return models, oof

def preds(new_train, train, targets, param, param2):
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

        # train_q = feature_quest(new_train, train, q)
        train_q = new_train

        # ВЫЧИСЛИТЕ РЕЗУЛЬТАТ С 5-ГРУППОВЫМ K FOLD
        for i, (train_index, test_index) in enumerate(gkf.split(X=train_q, groups=train_q.index)):
            print(' ', i + 1, end='')
            models, oof = one_vopros(train_q, train_index, targets, test_index, models, oof, q, param, param2)
        print()

    # ВСТАВЬТЕ ИСТИННЫЕ МЕТКИ В ФРЕЙМ ДАННЫХ С 18 СТОЛБЦАМИ
    true = oof.copy()
    for k in quests:
        # GET TRUE LABELS
        tmp = targets.loc[targets.q == k].set_index('session').loc[ALL_USERS]
        true[k] = tmp.correct.values
    return oof, true

def otvet(oof, true, param, param2, quest, rezult):
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


    rezult.loc[len(rezult.index)] = [param, param2, quest, m]

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
    pd.options.display.width = 0  # для печати
    vse_quests = [4, 5, 6, 7, 8, 9, 10, 11]#, 12, 13]
    gb_param = {4: [550, 5, 0.045],
                5: [450, 5, 0.045],
                6: [650, 3, 0.045],
                7: [850, 4, 0.045],
                8: [350, 3, 0.045],
                9: [550, 5, 0.045],
                10:[750, 5, 0.045],
                11:[250, 4, 0.045],
                12:[660, 7, 0.045],
                13:[670, 11, 0.045]
              }     #1-й элемент списка n_estimators, 2-й max_depth,
    # gb_param = {4: [160, 5, 0.005],
    #             5: [660, 5, 0.01],
    #             6: [140, 5, 0.005],
    #             7: [180, 5, 0.005],
    #             8: [220, 5, 0.02],
    #             9: [200, 5, 0.005],
    #             10:[740, 5, 0.02],
    #             11:[850, 5, 0.01]
    #           }     #1-й элемент списка n_estimators, 2-й max_depth,

    rooms = {14: ['tunic.historicalsociety.closet_dirty', # 0.611685
                  'tunic.historicalsociety.collection_flag', # 0.610912
                  'tunic.library.frontdesk'], # 0.610788
            }
    text_fqids = {

    }
    rezult = pd.DataFrame(columns=['param', 'param2', 'quest', 'rezultat'])
    train = read_csv_loc("C:\\kaggle\\ОбучИгра\\train_5_12.csv")

    # удаление строк с session_id в которых пройдены не все уровни
    tmp = (train.groupby(['session_id'])['level'].agg('nunique') == 8)
    tmp = tmp[tmp].index
    train = train[train['session_id'].isin(tmp)]
    # удаление строк с не = 1 'checkpoint'
    tmp = train[train['event_name'] == 'checkpoint']
    tmp = tmp.groupby(['session_id'])['event_name'].count()
    tmp = tmp[tmp == 1].index
    train = train[train['session_id'].isin(tmp)]

    targets = deftarget()
    # ТЕСТИРУЕМЫЕ ПАРАМЕТРЫ
    col = 'room_fqid' # перебор колонок для трайна
    ls = train[col].unique() # список значений колонки

    #второй столбец перебора для трайна

    # for quest in vse_quests:
    #     quests = [quest]

    if True:
        quests = vse_quests
        quest = 0
        param = 1
        param2 = 0

        # for param in ls:
        # for param in range(150, 1000, 100):
        #     maska = train[col] == param
        #     col2 = 'level'
        #     rrr = train[maska][col2].unique()
            # for param2 in range(3,6,1):
            #     param2 = param2 / 1000
        train.sort_values(by=['session_id', 'elapsed_time'], inplace=True)
        train['delt_time'] = train['elapsed_time'].diff(1)
        train['delt_time'].fillna(0, inplace=True)
        train['delt_time'].clip(0, 103000, inplace=True)
        new_train = feature_engineer(train)

        # new_train = dop_feature(new_train, train, col, param)

        # new_train = dop_feature2(new_train, train, col, param, col2, param2)

        oof, true = preds(new_train, train, targets, param, param2)
        otvet(oof, true, param, param2, quest, rezult)
        rezult.sort_values(by = 'rezultat', inplace=True, ascending=False)
        print(rezult.head(22))
        for q in rezult.quest.unique():
            print('вопрос =', q)
            print(rezult[rezult.quest==q].head(10))


quests = [4, 5, 6, 7, 8, 9, 10, 11]#, 12, 13]

if __name__ == "__main__":
    main()