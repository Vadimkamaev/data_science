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
    CATS = ['event_name', 'room_fqid', 'text_fqid', 'page']

    EV_NAME1 = ['observation_click', 'cutscene_click', 'notification_click', 'person_click',
               'object_click', 'object_hover']
    EV_NAME2 = ['observation_click', 'cutscene_click', 'notification_click', 'person_click',
                'object_click', 'map_click', 'object_hover']
    new_train = pd.DataFrame(index=train['session_id'].unique(), columns=[])

    # train = train[train['name'] == 'basic']

    # new_train['finish'] = train.groupby(['session_id'])['elapsed_time'].last(1)  # ? надо ли?
    # new_train['len'] = train.groupby(['session_id'])['index'].count()
    for c in CATS:
        tmp = train.groupby(['session_id'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        new_train = new_train.join(tmp)

    tmp = train.groupby(['session_id'])['delt_time'].agg('mean')
    new_train = new_train.join(tmp)

    tmp = train.groupby(['session_id'])['delt_time'].agg('std')
    tmp.name = tmp.name + '_std'
    new_train = new_train.join(tmp)

    tmp = train.groupby(['session_id'])['hover_duration'].agg('mean')
    new_train = new_train.join(tmp)

    for c in EV_NAME1:
        new_train['l_ev_name_' + c] = train[train['event_name'] == c].groupby(['session_id'])['index'].count()
    for c in EV_NAME2:
        new_train['t_ev_name_' + c] = train[train['event_name'] == c].groupby(['session_id'])['delt_time'].sum()

    # ДОБАВЛЯЕМ КВАНТИЛИ
    qvant = train.groupby(['session_id'])['d_time'].quantile(q=0.3)
    qvant.name = 'qvant1_0_3'
    new_train = new_train.join(qvant)

    qvant = train.groupby(['session_id'])['d_time'].quantile(q=0.8)
    qvant.name = 'qvant2_0_8'
    new_train = new_train.join(qvant)

    qvant = train.groupby(['session_id'])['d_time'].quantile(q=0.5)
    qvant.name = 'qvant3_0_5'
    new_train = new_train.join(qvant)

    qvant = train.groupby(['session_id'])['d_time'].quantile(q=0.65)
    qvant.name = 'qvant4_0_65'
    new_train = new_train.join(qvant)

    # квантиль
    new_train['q_person_click'] = \
        train[train['event_name'] == 'person_click'].groupby(['session_id'])['d_time'].quantile(q=0.3)

    # комната / уровень
    new_train['t_historicalsociety.frontdesk_12'] = \
        train[(train['room_fqid']=='tunic.historicalsociety.frontdesk')&(train['level']==12)].groupby(['session_id'])['delt_time'].sum()

    new_train['t_historicalsociety.stacks_12'] = \
        train[(train['room_fqid']=='tunic.historicalsociety.stacks')&(train['level']==12)].groupby(['session_id'])['delt_time'].sum()

    new_train['t_historicalsociety.stacks_11'] = \
        train[(train['room_fqid'] == 'tunic.historicalsociety.stacks') & (train['level'] == 11)].groupby(
            ['session_id'])['delt_time'].sum()

    new_train['t_humanecology.frontdesk_11'] = \
        train[(train['room_fqid'] == 'tunic.humanecology.frontdesk') & (train['level'] == 11)].groupby(
            ['session_id'])['delt_time'].sum()

    new_train['t_historicalsociety.frontdesk_7'] = \
        train[(train['room_fqid']=='tunic.historicalsociety.frontdesk')&(train['level']==7)].groupby(['session_id'])['delt_time'].sum()

    # в этой строке ухудшение результата
    new_train['t_kohlcenter.halloffame_12'] = \
        train[(train['room_fqid'] == 'tunic.kohlcenter.halloffame') & (train['level'] == 12)].groupby(
            ['session_id'])['delt_time'].sum()

    new_train['t_historicalsociety.entry_6'] = \
        train[(train['room_fqid'] == 'tunic.historicalsociety.entry') & (train['level'] == 6)].groupby(
            ['session_id'])['delt_time'].sum()

    new_train['t_historicalsociety.stacks_7'] = \
        train[(train['room_fqid'] == 'tunic.historicalsociety.stacks') & (train['level'] == 7)].groupby(
            ['session_id'])['delt_time'].sum()

    new_train['t_historicalsociety.closet_dirty_7'] = \
        train[(train['room_fqid'] == 'tunic.historicalsociety.closet_dirty') & (train['level'] == 7)].groupby(
            ['session_id'])['delt_time'].sum()

    # fqid
    new_train['t_fqid_logbook.page.bingo'] = \
        train[train['fqid'] == 'logbook.page.bingo'].groupby(['session_id'])['delt_time'].sum()

    new_train['t_fqid_tobasement'] = \
        train[train['fqid'] == 'tobasement'].groupby(['session_id'])['delt_time'].sum()

    new_train['t_fqid_wellsbadge'] = \
        train[train['fqid'] == 'wellsbadge'].groupby(['session_id'])['delt_time'].sum()

    new_train['t_fqid_boss'] = \
        train[train['fqid'] == 'boss'].groupby(['session_id'])['delt_time'].sum()

    new_train['t_fqid_businesscards.card_bingo.bingo'] = \
        train[train['fqid'] == 'businesscards.card_bingo.bingo'].groupby(['session_id'])['delt_time'].sum()
    # 0.6473945817868101 далее хуже
    new_train['t_fqid_reader'] = \
        train[train['fqid'] == 'reader'].groupby(['session_id'])['delt_time'].sum() # 0.647236

    new_train['t_fqid_tocloset_dirty'] = \
        train[train['fqid'] == 'tocloset_dirty'].groupby(['session_id'])['delt_time'].sum() # 0.647350406310234

    new_train['t_fqid_tunic.capitol_0'] = \
        train[train['fqid'] == 'tunic.capitol_0'].groupby(['session_id'])['delt_time'].sum()  # 0.647534

    return new_train

def dop_feature(new_train, train, col, param):
    #new_train['l_'+col+' '+str(param)] = train[train[col]==param].groupby(['session_id'])['index'].count()
    new_train['t_' + col + ' ' + str(param)] = train[train[col] == param].groupby(['session_id'])['delt_time'].sum()
    # new_train['q_' + col + ' ' + str(param)] = train[train[col] == param].groupby(['session_id'])['d_time'].quantile(q=0.3)
    return new_train

def dop_feature2(new_train, train, col, param, col2, param2):
    # new_train['l_'+str(param)+str(param2)] = \
    #     train[(train[col]==param)&(train[col2]==param2)].groupby(['session_id'])['index'].count()
    new_train['t_'+str(param)+str(param2)] = \
        train[(train[col]==param)&(train[col2]==param2)].groupby(['session_id'])['delt_time'].sum()
    return new_train

def one_fold(df, train_users, targets, test_users, models, param, param2):
    global gb_param, quests
    # TRAIN DATA
    train_x = df[df['session_id'].isin(train_users)]
    maska = targets['q'].isin(quests)
    train_y = targets[maska]
    train_y = train_y[train_y['session'].isin(train_users)]
    train_y.sort_values(by=['q','session'], inplace=True)

    # train_y = train_y.set_index('session')
    # train_y = train_y.loc[train_users]


    # VALID DATA
    maska = df['session_id'].isin(test_users)
    valid_x = df[maska]
    # valid_users = valid_x.index.values
    # valid_y = targets.loc[targets.q == t].set_index('session').loc[valid_users]

    # TRAIN MODEL
    model = CatBoostClassifier(
        n_estimators = 300,#900, #param
        learning_rate= 0.05,
        depth = 3,
        cat_features = ['q'],
        ignored_features = ['session_id']
    )
    X = train_x#.astype('float32')
    Y = train_y['correct']
    model.fit(X, Y, verbose=False)

    # SAVE MODEL, PREDICT VALID preds
    models['5_12'] = model
    preds = model.predict_proba(valid_x)
    preds = preds[:, 1]
    preds = preds.reshape((len(quests),-1))
    preds = preds.T
    return models, preds

def preds(new_train, train, targets, param, param2):
    global quests
    ALL_USERS = new_train.index.unique()
    # print('В трайне', len(train.columns), 'колонок')
    print('В трайне', len(ALL_USERS), 'пользователей')
    k_fold = KFold(n_splits=5)
    preds_np = pd.DataFrame(data=np.zeros((len(ALL_USERS), len(quests))), index=ALL_USERS, columns=quests)
    models = {}
    # СОЗДАЕМ ТРАЙН ДЛЯ ВСЕХ ВОПРОСОВ
    new_train['q'] = quests[0]
    new_train.reset_index(inplace=True, names='session_id')
    train_q = new_train.copy()
    print('### quest', end='')
    for q in quests[1:]:
        print(q, ' ', end='')
        new_train['q'] = q
        train_q = pd.concat([train_q,new_train])
    train_q.reset_index(inplace=True, drop = True)
    # train_q = feature_quest(new_train, train, q)
    # ВЫЧИСЛИТЕ РЕЗУЛЬТАТ С 5-ГРУППОВЫМ K FOLD
    print('')
    print('k_fold ==>', end='')
    for (train_users, test_users) in k_fold.split(ALL_USERS):
        train_users = ALL_USERS[train_users]
        test_users = ALL_USERS[test_users]
        print(' .', end='')
        models, preds = one_fold(train_q, train_users, targets, test_users, models, param, param2)
        preds_np.loc[test_users] = preds
    print()

    # ВСТАВЬТЕ ИСТИННЫЕ МЕТКИ В ФРЕЙМ ДАННЫХ С 18 СТОЛБЦАМИ
    true = preds_np.copy()
    for k in quests:
        # GET TRUE LABELS
        tmp = targets.loc[targets.q == k].set_index('session').loc[ALL_USERS]
        true[k] = tmp.correct.values
    return preds_np, true

def otvet(preds_np, true, param, param2, quest, rezult):
    global quests, group_quest
    #ВЫЧИСЛЕНИЕ ПОРОГА ДЛЯ КАЖДОГО ВОПРОСА ОТДЕЛЬНО
    scores = []; thresholds = []
    # for k in quests:
    #     best_score = 0
    #     best_threshold = 0
    #     for threshold in np.arange(-0.01, 1.01, 0.01):
    #         # print(f'{threshold:.02f}, ',end='')
    #         preds = (preds_np[k].values > threshold).astype('int')
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
        y_pred = preds_np[k].values
        m = f1_score(tru, (y_pred > best_threshold).astype('int'), average='macro')
        print(f'Q{k}: F1 =', m)

    print('Результат для групп вопросов с общим порогом:')
    for gq in group_quest:
        print('Группа вопросов:', gq)
        tru = true[gq].values.reshape((-1))
        y_pred = preds_np[gq].values.reshape((-1))
        m = f1_score(tru, (y_pred > best_threshold).astype('int'), average='macro')
        print('F1 =', m)

    # Считаем F1 SCORE для всех вопросов
    tru3 = true[quests]
    tru = tru3.values.reshape((-1))
    preds_np3 = preds_np[quests]
    y_pred = preds_np3.values.reshape((-1))
    m = f1_score(tru, (y_pred > best_threshold).astype('int'), average='macro')
    print('==> Для всех вопросов =', m)


    rezult.loc[len(rezult.index)] = [param, param2, quest, m]

    # print('Результат для каждого вопроса с индивидуальным порогом:')
    # for k in quests:
    #     # Считаем F1 SCORE для каждого вопроса
    #     tru = true[k].values
    #     y_pred = preds_np[k].values
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
    vse_quests = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

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
    col = 'fqid' # перебор колонок для трайна
    ls = train[col].unique() # список значений колонки

    # СПИСОК СЛОВ РАЗДЕЛЕННЫХ ТОЧКОЙ В КОЛОНКЕ
    # ss = []
    # for param in ls:
    #     if isinstance(param, str):
    #         ss += param.split('.')
    # ss = set(ss)


    #второй столбец перебора для трайна
    quest = 0
    param = 1
    param2 = 0

    # for param in ls:
    # # # if param in EV_NAME2:
    # # #     continue
    # # # for param in range(3, 4, 1):
    #     maska = train[col] == param
    #     col2 = 'level'
    #     rrr = train[maska][col2].unique()
    #     for param2 in rrr:
            #  param2 = param2 / 1000
    train.sort_values(by=['session_id', 'elapsed_time'], inplace=True)
    train['d_time'] = train['elapsed_time'].diff(1)

    train['d_time'].fillna(0, inplace=True)
    train['delt_time'] = train['d_time'].clip(0, 19000)
    new_train = feature_engineer(train)

    # ПРОВЕРКА НЕТ ЛИ ЛИШНИХ КОЛОНОК
    # print('columns:', new_train.columns)
    # new_train.drop(columns = param, inplace=True)

    new_train = dop_feature(new_train, train, col, param)

    # new_train = dop_feature2(new_train, train, col, param, col2, param2)

    oof, true = preds(new_train, train, targets, param, param2)
    otvet(oof, true, param, param2, quest, rezult)
    rezult.sort_values(by = 'rezultat', inplace=True, ascending=False)
    print(rezult.head(30))


quests = [4, 5, 6, 7, 8, 9, 10, 11]#, 12, 13]
group_quest = [[4, 5, 6, 7, 8, 9, 10, 11]]
if __name__ == "__main__":
    main()