import numpy as np, pandas as pd

import gc # сборщик мусора
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm # прогресс бар

from servise_ds import okno
import obrabotka_filtr

blacklist = [
    'North Dakota', 'Iowa', 'Kansas', 'Nebraska', 'South Dakota', 'New Mexico',
    'Alaska', 'Vermont'
]
blacklistcfips = [
    1019, 1027, 1029, 1035, 1039, 1045, 1049, 1057, 1067, 1071, 1077, 1085, 1091, 1099, 1101, 1123, 1131, 1133,
    4001, 4012, 4013, 4021, 4023, 5001, 5003, 5005, 5017, 5019, 5027, 5031, 5035, 5047, 5063, 5065, 5071, 5081,
    5083, 5087, 5091, 5093, 5107, 5109, 5115, 5121, 5137, 5139, 5141, 5147, 6003, 6015, 6027, 6033, 6053, 6055,
    6057, 6071, 6093, 6097, 6103, 6105, 6115, 8003, 8007, 8009, 8019, 8021, 8023, 8047, 8051, 8053, 8055, 8057,
    8059, 8061, 8065, 8067, 8069, 8071, 8073, 8075, 8085, 8091, 8093, 8097, 8099, 8103, 8105, 8107, 8109, 8111,
    8115, 8117, 8121, 9007, 9009, 9015, 12009, 12017, 12019, 12029, 12047, 12055, 12065, 12075, 12093, 12107, 12127,
    13005, 13007, 13015, 13017, 13019, 13027, 13035, 13047, 13065, 13081, 13083, 13099, 13107, 13109, 13117, 13119,
    13121, 13123, 13125, 13127, 13135, 13143, 13147, 13161, 13165, 13171, 13175, 13181, 13193, 13201, 13221, 13225,
    13229, 13231, 13233, 13245, 13247, 13249, 13257, 13279, 13281, 13287, 13289, 13293, 13301, 13319, 15001, 15005,
    15007, 16001, 16003, 16005, 16007, 16013, 16015, 16017, 16023, 16025, 16029, 16031, 16033, 16035, 16037, 16043,
    16045, 16049, 16061, 16063, 16067, 17001, 17003, 17007, 17009, 17013, 17015, 17023, 17025, 17031, 17035, 17045,
    17051, 17059, 17061, 17063, 17065, 17067, 17069, 17075, 17077, 17081, 17085, 17087, 17103, 17105, 17107, 17109,
    17115, 17117, 17123, 17127, 17133, 17137, 17141, 17143, 17147, 17153, 17167, 17169, 17171, 17177, 17179, 17181,
    17185, 17187, 17193, 18001, 18007, 18009, 18013, 18015, 18019, 18021, 18025, 18035, 18037, 18039, 18041, 18053,
    18061, 18075, 18079, 18083, 18087, 18099, 18103, 18111, 18113, 18115, 18137, 18139, 18145, 18153, 18171, 18179,
    21001, 21003, 21013, 21017, 21023, 21029, 21035, 21037, 21039, 21045, 21047, 21055, 21059, 21065, 21075, 21077,
    21085, 21091, 21093, 21097, 21099, 21101, 21103, 21115, 21125, 21137, 21139, 21141, 21149, 21155, 21157, 21161,
    21165, 21179, 21183, 21191, 21197, 21199, 21215, 21217, 21223, 21227, 21237, 21239, 22019, 22021, 22031, 22039,
    22041, 22047, 22069, 22085, 22089, 22101, 22103, 22109, 22111, 22115, 22119, 22121, 23003, 23009, 23021, 23027,
    23029, 24011, 24027, 24029, 24031, 24035, 24037, 24039, 24041, 25011, 25015, 26003, 26007, 26011, 26019, 26021,
    26025, 26027, 26033, 26037, 26041, 26043, 26051, 26053, 26057, 26059, 26061, 26065, 26071, 26077, 26079, 26083,
    26089, 26097, 26101, 26103, 26109, 26111, 26115, 26117, 26119, 26127, 26129, 26131, 26135, 26141, 26143, 26155,
    26161, 26165, 27005, 27011, 27013, 27015, 27017, 27021, 27023, 27025, 27029, 27047, 27051, 27055, 27057, 27065,
    27069, 27073, 27075, 27077, 27079, 27087, 27091, 27095, 27101, 27103, 27105, 27107, 27109, 27113, 27117, 27119,
    27123, 27125, 27129, 27131, 27133, 27135, 27141, 27147, 27149, 27155, 27159, 27167, 27169, 28017, 28019, 28023,
    28025, 28035, 28045, 28049, 28061, 28063, 28093, 28097, 28099, 28125, 28137, 28139, 28147, 28159, 29001, 29015,
    29019, 29031, 29033, 29041, 29049, 29051, 29055, 29057, 29063, 29065, 29069, 29075, 29085, 29089, 29101, 29103,
    29111, 29121, 29123, 29125, 29135, 29137, 29139, 29143, 29157, 29159, 29161, 29167, 29171, 29173, 29175, 29177,
    29183, 29195, 29197, 29199, 29203, 29205, 29207, 29209, 29213, 29215, 29217, 29223, 29227, 29229, 30005, 30009,
    30025, 30027, 30033, 30035, 30037, 30039, 30045, 30049, 30051, 30053, 30055, 30057, 30059, 30069, 30071, 30073,
    30077, 30079, 30083, 30085, 30089, 30091, 30093, 30101, 30103, 30105, 30107, 30109, 32005, 32009, 32017, 32023,
    32027, 32029, 32510, 33005, 33007, 34021, 34027, 34033, 34035, 36011, 36017, 36023, 36033, 36043, 36047, 36049,
    36051, 36057, 36061, 36067, 36083, 36091, 36097, 36103, 36107, 36113, 36115, 36121, 36123, 37005, 37009, 37011,
    37017, 37023, 37029, 37031, 37049, 37061, 37075, 37095, 37117, 37123, 37131, 37137, 37151, 37187, 37189, 37197,
    39005, 39009, 39015, 39017, 39019, 39023, 39037, 39039, 39043, 39049, 39053, 39057, 39063, 39067, 39071, 39077,
    39085, 39087, 39091, 39097, 39105, 39107, 39113, 39117, 39119, 39125, 39127, 39129, 39135, 39137, 39151, 39153,
    39157, 40003, 40013, 40015, 40023, 40025, 40027, 40035, 40039, 40043, 40045, 40053, 40055, 40057, 40059, 40065,
    40067, 40073, 40077, 40079, 40099, 40105, 40107, 40111, 40115, 40123, 40127, 40129, 40133, 40141, 40147, 40151,
    40153, 41001, 41007, 41013, 41015, 41017, 41021, 41025, 41031, 41033, 41037, 41051, 41055, 41063, 41067, 41069,
    42005, 42007, 42011, 42013, 42015, 42019, 42027, 42029, 42031, 42035, 42053, 42057, 42067, 42071, 42083, 42085,
    42093, 42097, 42105, 42111, 42113, 42115, 42123, 42125, 42127, 42129, 44005, 44007, 44009, 45001, 45009, 45021,
    45025, 45031, 45059, 45067, 45071, 45073, 45089, 47001, 47005, 47013, 47015, 47019, 47021, 47023, 47027, 47035,
    47039, 47041, 47047, 47055, 47057, 47059, 47061, 47069, 47073, 47075, 47077, 47083, 47087, 47099, 47105, 47121,
    47127, 47131, 47133, 47135, 47137, 47147, 47151, 47153, 47159, 47161, 47163, 47169, 47177, 47183, 47185, 48001,
    48011, 48017, 48019, 48045, 48057, 48059, 48063, 48065, 48073, 48077, 48079, 48081, 48083, 48087, 48095, 48101,
    48103, 48107, 48109, 48115, 48117, 48119, 48123, 48125, 48129, 48149, 48151, 48153, 48155, 48159, 48161, 48165,
    48175, 48189, 48191, 48195, 48197, 48211, 48221, 48229, 48233, 48235, 48237, 48239, 48241, 48243, 48245, 48255,
    48261, 48263, 48265, 48267, 48269, 48275, 48277, 48283, 48293, 48299, 48305, 48311, 48313, 48319, 48321, 48323,
    48327, 48333, 48345, 48347, 48355, 48369, 48377, 48379, 48383, 48387, 48389, 48401, 48403, 48413, 48417, 48431,
    48433, 48437, 48443, 48447, 48453, 48455, 48457, 48461, 48463, 48465, 48469, 48471, 48481, 48483, 48485, 48487,
    48495, 48499, 49001, 49009, 49013, 49019, 49027, 49031, 49045, 51005, 51017, 51025, 51029, 51031, 51036, 51037,
    51043, 51057, 51059, 51065, 51071, 51073, 51077, 51079, 51083, 51091, 51095, 51097, 51101, 51111, 51115, 51119,
    51121, 51127, 51135, 51147, 51155, 51159, 51165, 51167, 51171, 51173, 51181, 51183, 51191, 51197, 51530, 51590,
    51610, 51620, 51670, 51678, 51720, 51735, 51750, 51770, 51810, 51820, 53013, 53019, 53023, 53031, 53033, 53037,
    53039, 53041, 53047, 53065, 53069, 53071, 53075, 54013, 54019, 54025, 54031, 54033, 54041, 54049, 54055, 54057,
    54063, 54067, 54071, 54077, 54079, 54085, 54089, 54103, 55001, 55003, 55005, 55007, 55011, 55017, 55021, 55025,
    55029, 55037, 55043, 55047, 55049, 55051, 55061, 55065, 55067, 55075, 55077, 55091, 55097, 55101, 55103, 55109,
    55117, 55123, 55125, 55127, 56007, 56009, 56011, 56015, 56017, 56019, 56021, 56027, 56031, 56037, 56043, 56045,
    12061, 6095, 49025, 18073, 29029, 29097, 48419, 51830, 30067, 26095, 18159, 32001, 54065, 54027, 13043, 48177,
    55069, 48137, 30087, 29007, 13055, 48295, 28157, 29037, 45061, 22053, 13199, 47171, 53001, 55041, 51195, 18127,
    29151, 48307, 51009, 16047, 29133, 5145, 17175, 21027, 48357, 29179, 13023, 16077, 48371, 21057, 16039, 21143,
    48435, 48317, 48475, 5129, 36041, 48075, 29017, 47175, 39167, 47109, 17189, 17173, 28009, 39027, 48133, 18129,
    48217, 40081, 36021, 6005, 42099, 18051, 36055, 53051, 6109, 21073, 27019, 6051, 48055, 8083, 48503, 17021,
    10003, 41061, 22001, 22011, 21205, 48223, 51103, 51047, 16069, 17033, 41011, 6035, 47145, 27083, 18165, 36055,
    12001, 26159, 8125, 34017,
    28141, 55119, 48405, 40029, 18125, 21135, 29073, 55115, 37149, 55039, 26029, 12099, 13251, 48421, 39007, 41043,
    22015, 37115, 54099, 51137, 22049, 55131, 17159, 56001, 40005, 18017, 28091, 47101, 27037, 29005, 13239, 21019,
    55085, 48253, 51139, 40101, 13283, 18049, 39163, 45049, 51113,
]
# С этими блек листами XGB. Ошибка SMAPE: 1.0786937804504582, хуже чем моделью = 33

print(len(blacklistcfips))

#blacklist =[]
#blacklistcfips =[]
# без блек листов XGB. Ошибка SMAPE: 1.081203276846456, хуже чем моделью = 1250

# БЛЕК ЛИСТ СОХРАНЯЮЩИЙСЯ В ФАЙЛЕ 1.0879. Лучше чем совсем без блек листа - 1.0971
#список 'cfips' у которых ошибка модели больше чем ошибка модели '='
# bl = pd.read_csv("C:\\kaggle\\МикроБизнес\\blec_list.csv")
# bl = bl.loc[bl['miss']>=0.50]
# vbl = list(bl['cfips'])
# blacklistcfips = vbl

# определение среднего размера ошибки
def smape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    pos_ind = (y_true != 0) | (y_pred != 0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    return 100 * np.mean(smap)

# получение вектора ошибоk
def vsmape(y_true, y_pred):
    smap = np.zeros(len(y_true))
    num = np.abs(y_true - y_pred)
    dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
    pos_ind = (y_true != 0) | (y_pred != 0)
    smap[pos_ind] = num[pos_ind] / dem[pos_ind]
    return 100 * smap

# загрузка файлов
def start():
    train = pd.read_csv("C:\\kaggle\\МикроБизнес\\train.csv")
    train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])
    train = train.sort_values(['cfips','first_day_of_month']).reset_index(drop=True)
    test = pd.read_csv('C:\\kaggle\\МикроБизнес\\test.csv')
    test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])
    test = test.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)
    census = pd.read_csv("C:\\kaggle\\МикроБизнес\\census_starter.csv")
    return train, test, census

# объединенный массив трейна и теста, создание объединенного raw
def maceraw(train, test):
    train['istest'] = 0
    test['istest'] = 1
    # объединенный массив трейна и теста
    raw = pd.concat((train, test)).sort_values(['cfips','row_id']).reset_index(drop=True)
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    raw['county'] = raw.groupby('cfips')['county'].ffill()
    raw['state'] = raw.groupby('cfips')['state'].ffill()
    raw["year"] = raw["first_day_of_month"].dt.year
    raw["month"] = raw["first_day_of_month"].dt.month
    # перенумерация внутри группы - номер месяца
    raw["dcount"] = raw.groupby(['cfips'])['row_id'].cumcount()
    # кодирование целыми числами
    raw['county_i'] = (raw['county'] + raw['state']).factorize()[0]
    raw['state_i'] = raw['state'].factorize()[0]
    return raw

# обработка файла census
def censusdef(census, raw):
    columns = ['pct_bb', 'pct_college', 'pct_foreign_born', 'pct_it_workers', 'median_hh_inc']
    raw[columns] = 0
    for index, row_census in census.iterrows():
        for year in range(2017, 2022):
            for col in columns:
                ncol = col+'_'+str(year)
                maska = (raw['year']==year+2)&(raw['cfips']==row_census['cfips'])
                raw.loc[maska, col] = row_census[ncol]
    return raw

# УДАЛЕНИЕ ВЫБРОСОВ. Применется изменение всех значений до выброса
def del_outliers(raw):
    # цикл по уникальным cfips
    for o in tqdm(raw.cfips.unique()):  # tqdm - прогресс бар
        indices = (raw['cfips'] == o)  # маска отфильтрованная по cfips
        tmp = raw.loc[indices].copy().reset_index(drop=True)  # df фильтрован по cfips
        # массив значений microbusiness_density
        var = tmp.microbusiness_density.values.copy()
        # цикл назад от предпоследнего до 2-го элемента
        for i in range(37, 2, -1):
            # среднее значение microbusiness_density с 0-го по i-й элемент * 0.2
            thr = 0.10 * np.mean(var[:i]) # 1.0863 - 22 место
            difa = var[i] - var[i - 1] #XGB. Ошибка SMAPE: 1.161117561342498
            if (difa >= thr) or (difa <= -thr):  # если microbusiness_density изменился больше чем на 20%
                if difa > 0:
                    var[:i] += difa - 0.0045 # 0.0045 лучше чем 0.003 и чем 0.006
                else:
                    var[:i] += difa + 0.0043 # 0.0043 лучше чем 0.003 и чем 0.006
        var[0] = var[1] * 0.99
        raw.loc[indices, 'microbusiness_density'] = var
    return raw

#SMAPE — это относительная метрика, поэтому цель raw['target'] преобразована и равна отношению
# 'microbusiness_density' за следующий месяц к текущему значению 'microbusiness_density' - 1
def chang_target(raw):
    # 'target' ='microbusiness_density' следующего месяца при том же 'cfips'
    raw['target'] = raw.groupby('cfips')['microbusiness_density'].shift(-1)
    # -1, чтобы при не изменении значения 'microbusiness_density' - 'target'==0
    raw['target'] = raw['target'] / raw['microbusiness_density'] - 1
    # в этих 'cfips' значение 'active' аномально маленькие
    raw.loc[raw['cfips'] == 28055, 'target'] = 0.0
    raw.loc[raw['cfips'] == 48269, 'target'] = 0.0
    return raw

# создаем новые столбцы 'lastactive' и 'lasttarget'
def mace_fold(raw):
    # 'lastactive' содержит значение 'active' одинаковое для всех строк с одним 'cfips'
    # и равное 'active' за последний месяц (сейчас 2022-10-01)
    raw['lastactive'] = raw.groupby('cfips')['active'].transform('last')
    # серия из 'cfips' и 'microbusiness_density' 28-го месяца из трайн
    dt = raw.loc[raw.dcount == 28].groupby('cfips')['microbusiness_density'].agg('last')
    # 'lasttarget' содержит значение 'microbusiness_density' 28-го месяца из трайн
    # одинаковое для всех строк с одним 'cfips'
    raw['lasttarget'] = raw['cfips'].map(dt)
    # print(raw[['row_id','microbusiness_density','lasttarget']].head(50))
    # в пределах от 0 до 8000
    raw['lastactive'].clip(0, 8000).hist(bins=30)
    return raw

# создаем 4 лага 'mbd_lag_{lag}' - (отношение 'microbusiness_density' следующего
# месяца к текущему - 1); 4 лага 'act_lag_{lag} - (разность 'active' следующей
# через лаг строки и текущей строки)' и столбцы 'mbd_rollmea{window}_{lag}' -
# суммы значений в окнах
def build_features(raw, target='target', target_act='active', lags=4):
    # 'target' ='microbusiness_density' следующего месяца деленное на
    # 'microbusiness_density' текущего месяца - 1 при том же 'cfips'
    lags = 4 # lags > 4 стабильно хуже.
    feats = []  # список имен лагов и сумм окон
    # создаем лаги target='microbusiness_density' от 1 до lags1
    for lag in range(1, lags):   #XGB. Ошибка SMAPE: 1.07924633283354
        # shift - сдвиг на определеное кол. позиций
        raw[f'mbd_lag_{lag}'] = raw.groupby('cfips')[target].shift(lag)
        # из значения 'active' следующей через лаг строки вычитается значение
        # текущей строки. diff - разность элемента df с элементом через lag
        raw[f'act_lag_{lag}'] = raw.groupby('cfips')[target_act].diff(lag)
        feats.append(f'mbd_lag_{lag}')
        feats.append(f'act_lag_{lag}')
    lag = 1
    # создаем значения сумм окон для lag = 1
    for window in [2, 4, 6, 8, 10]: # XGB. Ошибка SMAPE: 1.0786478967227329
        # сгруппированно по 'cfips' 1-й лаг трансформируем - считаем сумму в окнах
        # размером [2, 4, 6]
        raw[f'mbd_rollmea{window}_{lag}'] = raw.groupby('cfips')[f'mbd_lag_{lag}'].transform(
            lambda s: s.rolling(window, min_periods=1).sum())
        # raw[f'mbd_rollmea{window}_{lag}'] = raw[f'mbd_lag_{lag}'] - raw[f'mbd_rollmea{window}_{lag}']
        feats.append(f'mbd_rollmea{window}_{lag}')
    features = ['state_i']  # номера штатов
    # Проверенно бесполезные колонки "month",
    features += feats  # список имен лагов и сумм окон
    # без этого Ошибка SMAPE: 1.079043460453389, хуже чем моделью = 24
    print(features)
    return raw, features


# создание модели градиентного бустинга
def model_XGBRegressor():
    model = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        # objective='reg:squarederror',
        tree_method="hist",
        n_estimators=4999,
        learning_rate=0.0075,
        max_leaves=17,
        subsample=0.50,
        colsample_bytree=0.50,
        max_bin=4096,
        n_jobs=2,
        eval_metric='mae',
        early_stopping_rounds=70,
    )
    return model

# обучение модели
def learn_model(model, raw, TS, ACT_THR, ABS_THR):
    # маска тренировочной выборки
    train_indices = (raw.istest == 0) & (raw.dcount < TS) & (raw.dcount >= 1) & (raw.lastactive > ACT_THR) & (
            raw.lasttarget > ABS_THR)
    # маска валидационной выборки. Валидация по 1 месяцу
    valid_indices = (raw.istest == 0) & (raw.dcount == TS)
    # обучение модели
    model.fit(
        # Трайн. features - список имен столбцов
        raw.loc[train_indices, features],
        # y(игрик) модели
        raw.loc[train_indices, 'target'].clip(-0.0043, 0.0045),
        # валидационная выборка из 1 месяца
        eval_set=[(raw.loc[valid_indices, features], raw.loc[valid_indices, 'target'])],
        verbose=500,
    )
    return model, valid_indices

# предсказание на основе модели
def predsk_model(raw, valid_indices, model):
    # предсказание модели на основе валидационной выборки и списка имен столбцов
    ypred = model.predict(raw.loc[valid_indices, features])
    # получаем из значения предсказанного ypred, который есть 'target'
    # предсказанное значение 'microbusiness_density' и помещаем его в 'k'
    raw.loc[valid_indices, 'k'] = ypred + 1
    raw.loc[valid_indices, 'k'] = raw.loc[valid_indices, 'k'] * raw.loc[valid_indices, 'microbusiness_density']
    return raw

# Валидация
def valid_rez(raw, TS, ACT_THR, ABS_THR, blacklist, blacklistcfips):
    # Валидация
    # создаем словари 'microbusiness_density' и 'k'
    lastval = raw.loc[raw.dcount == TS, ['cfips', 'microbusiness_density']].set_index('cfips').to_dict()[
        'microbusiness_density']
    dt = raw.loc[raw.dcount == TS, ['cfips', 'k']].set_index('cfips').to_dict()['k']
    # создаем новый датафрейм с данными TS+1 месяца
    df = raw.loc[raw.dcount == (TS + 1), ['cfips', 'microbusiness_density', 'state',
                                          'lastactive', 'mbd_lag_1']].reset_index(drop=True)
    # создаем колонку предсказание
    df['pred'] = df['cfips'].map(dt)
    # создаем колонку реальность 'microbusiness_density' TS-го месяца
    df['lastval'] = df['cfips'].map(lastval)
    # если active маленький то предсказание равно реальности TS-го месяца
    df.loc[df['lastactive'] <= ACT_THR, 'pred'] = df.loc[df['lastactive'] <= ACT_THR, 'lastval']
    # если 'microbusiness_density' ниже порога, то предсказание равно реальности
    df.loc[df['lastval'] <= ABS_THR, 'pred'] = df.loc[df['lastval'] <= ABS_THR, 'lastval']
    # если штат в черном списке то предсказание равно реальности TS-го месяца
    df.loc[df['state'].isin(blacklist), 'pred'] = df.loc[df['state'].isin(blacklist), 'lastval']
    # если 'cfips' в черном списке то предсказание равно реальности TS-го месяца
    df.loc[df['cfips'].isin(blacklistcfips), 'pred'] = df.loc[df['cfips'].isin(blacklistcfips), 'lastval']
    # создаем колонку предсказаний
    raw.loc[raw.dcount == (TS + 1), 'ypred'] = df['pred'].values
    # создаем колонку предыдущих значений
    raw.loc[raw.dcount == (TS + 1), 'ypred_last'] = df['lastval'].values
    print(f'Месяц валидации - TS: {TS}')
    print('Равенство последнему значению. Ошибка SMAPE:',
          smape(df['microbusiness_density'], df['lastval']))
    print('Предсказано XGB. Ошибка SMAPE:',
          smape(df['microbusiness_density'], df['pred']))
    print()
    return raw

def model(raw):
    global blacklist, blacklistcfips
    ACT_THR = 5 # 1.0773034636263392
    ABS_THR = 0.7 # 1.0773034636263392
    # для теста на сайте лучше старые значения
    # ACT_THR = 1.8  # условие попадания в обучающую выборку raw.lastactive>ACT_THR
    # ABS_THR = 1
    ABS_THR = 100.00  # результат на сайте 1.0862 мало меняется при ABS_THR от 5 до 100
    raw['ypred_last'] = np.nan
    raw['ypred'] = np.nan
    raw['k'] = 1.
    df = raw[(raw.istest == 0) &  (raw.dcount >= 1)]
    df1 = df[df['cfips'].isin(blacklistcfips)]
    df4 = df[df['state'].isin(blacklist)]
    df2 = df[(raw.lastactive <= ACT_THR)]
    df3 = df[(raw.lasttarget <= ABS_THR)]
    BEST_ROUNDS = []  # лучшие раунды
    # TS - номер месяца валидацииции. Месяцы то TS это трайн.
    for TS in range(29, 38):
        print(TS)
        # создание модели
        model = model_XGBRegressor()
        # обучение модели
        model, valid_indices = learn_model(model, raw, TS, ACT_THR, ABS_THR)
        BEST_ROUNDS.append(model.best_iteration)  # добавляем в список лучших раундов
        # предсказание на основе модели
        raw = predsk_model(raw, valid_indices, model)
        raw = valid_rez(raw, TS, ACT_THR, ABS_THR, blacklist, blacklistcfips)
    # маска при которой вероятно все хорошо
    ind = (raw.dcount >= 30) & (raw.dcount <= 38)
    print('Предсказано XGB. Ошибка SMAPE:',
          smape(raw.loc[ind, 'microbusiness_density'], raw.loc[ind, 'ypred']))
    print('Равенство последнему значению. Ошибка SMAPE:',
          smape(raw.loc[ind, 'microbusiness_density'], raw.loc[ind, 'ypred_last']))

def kol_error(raw):
    # Выводы
    raw['error'] = vsmape(raw['microbusiness_density'], raw['ypred'])
    raw['error_last'] = vsmape(raw['microbusiness_density'], raw['ypred_last'])
    #raw.loc[(raw.dcount == 30), ['microbusiness_density', 'ypred', 'error', 'error_last']]
    # создаем датафрейм со столбцами error и error_last
    dt = raw.loc[(raw.dcount >= 30) & (raw.dcount <= 38)].groupby(['cfips', 'dcount'])['error', 'error_last'].last()
    # преобразуем dt в серию средних значений 'miss'
    dt['miss'] = dt['error'] > dt['error_last'] # ошибка модели > ошибки модели '='
    dt = dt.groupby('cfips')['miss'].mean()
    df_dt = pd.DataFrame({'cfips': dt.index, 'miss': dt})
    dt = dt.loc[dt>=0.50] # оставляем только те, где ['miss'].mean() > 0.5
    print('количство cfips предсказанных хуже чем моделью =', len(dt))
    # список 'cfips' у которых ошибка модели больше чем ошибка модели '='
    # li = ','.join([str(i) for i in dt.index])
    # li = list(dt.index)
    df_dt.to_csv("C:\\kaggle\\МикроБизнес\\blec_list.csv", index = False)
    pass


def rezultat(raw):
    global blacklist, blacklistcfips
    ACT_THR = 1.8  # условие попадания в обучающую выборку raw.lastactive>ACT_THR
    ABS_THR = 1.00  # условие попадания в обучающую выборку raw.lasttarget>ABS_THR
    best_rounds = 794
    TS = 38
    print(TS)
    # создание моделей
    model0 = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        # objective='reg:squarederror',
        tree_method="hist",
        n_estimators=best_rounds,
        learning_rate=0.0075,
        max_leaves=31,
        subsample=0.60,
        colsample_bytree=0.50,
        max_bin=4096,
        n_jobs=2,
        eval_metric='mae',
    )
    model1 = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        # objective='reg:squarederror',
        tree_method="hist",
        n_estimators=best_rounds,
        learning_rate=0.0075,
        max_leaves=31,
        subsample=0.60,
        colsample_bytree=0.50,
        max_bin=4096,
        n_jobs=2,
        eval_metric='mae',
    )
    # тренировочная выборка
    train_indices = (raw.istest == 0) & (raw.dcount < TS) & (raw.dcount >= 1) & (raw.lastactive > ACT_THR) & (
                raw.lasttarget > ABS_THR)
    # валидационная выборка
    valid_indices = (raw.dcount == TS)
    # обучение моделей
    model0.fit(
        raw.loc[train_indices, features],
        raw.loc[train_indices, 'target'].clip(-0.0044, 0.0046),
    )
    model1.fit(
        raw.loc[train_indices, features],
        raw.loc[train_indices, 'target'].clip(-0.0044, 0.0046),
    )
    # среднее предсказание моделей
    ypred = (model0.predict(raw.loc[valid_indices, features]) + model1.predict(raw.loc[valid_indices, features])) / 2
    # преобразуем 'k' к 'microbusiness_density'
    raw.loc[valid_indices, 'k'] = ypred + 1.
    raw.loc[valid_indices, 'k'] = raw.loc[valid_indices, 'k'] * raw.loc[valid_indices, 'microbusiness_density']
    # Валидация
    # Приводим 'microbusiness_density' и 'k' к виду словарей
    lastval = raw.loc[raw.dcount == TS, ['cfips', 'microbusiness_density']].set_index('cfips').to_dict()[
        'microbusiness_density']
    dt = raw.loc[raw.dcount == TS, ['cfips', 'k']].set_index('cfips').to_dict()['k']
    # создаем новый датафрейм с данными TS+1 месяца
    df = raw.loc[
        raw.dcount == (TS + 1), ['cfips', 'microbusiness_density', 'state', 'lastactive', 'mbd_lag_1']].reset_index(
        drop=True)
    # создаем колонку - предсказание
    df['pred'] = df['cfips'].map(dt)
    # создаем колонку реальность 'microbusiness_density' TS-го месяца
    df['lastval'] = df['cfips'].map(lastval)
    # если active маленький то предсказание равно реальности TS-го месяца
    df.loc[df['lastactive'] <= ACT_THR, 'pred'] = df.loc[df['lastactive'] <= ACT_THR, 'lastval']
    # если 'microbusiness_density' ниже порога, то предсказание равно реальности
    df.loc[df['lastval'] <= ABS_THR, 'pred'] = df.loc[df['lastval'] <= ABS_THR, 'lastval']
    # если штат в черном списке то предсказание равно реальности TS-го месяца
    df.loc[df['state'].isin(blacklist), 'pred'] = df.loc[df['state'].isin(blacklist), 'lastval']
    # если 'cfips' в черном списке то предсказание равно реальности TS-го месяца
    df.loc[df['cfips'].isin(blacklistcfips), 'pred'] = df.loc[df['cfips'].isin(blacklistcfips), 'lastval']
    raw.loc[raw.dcount == (TS + 1), 'ypred'] = df['pred'].values
    raw.loc[raw.dcount == (TS + 1), 'ypred_last'] = df['lastval'].values

    raw.loc[raw['cfips'] == 28055, 'microbusiness_density'] = 0
    raw.loc[raw['cfips'] == 48269, 'microbusiness_density'] = 1.762115

    dt = raw.loc[raw.dcount == 39, ['cfips', 'ypred']].set_index('cfips').to_dict()['ypred']
    test = raw.loc[raw.istest == 1, ['row_id', 'cfips', 'microbusiness_density']].copy()
    test['microbusiness_density'] = test['cfips'].map(dt)

    test[['row_id', 'microbusiness_density']].to_csv('C:\\kaggle\\МикроБизнес\\sub_xgb.csv', index=False)


# train, test, census = start() # загрузка файлов
# raw = maceraw(train, test) # объединенный массив трейна и теста, создание объединенного raw
# raw = del_outliers(raw) # УДАЛЕНИЕ ВЫБРОСОВ. Применется изменение всех значений до выброса
# raw = chang_target(raw) # изменение цели
# raw = mace_fold(raw) # создаем новые столбцы 'lastactive' и 'lasttarget'

raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw1_covid.csv")
# XGB. Ошибка SMAPE: 1.079043460453389, хуже чем моделью = 24

raw, features = build_features(raw, 'target', 'active', lags = 4) # создаем лаги и суммы в окнах
model(raw)
kol_error(raw)
rezultat(raw)


