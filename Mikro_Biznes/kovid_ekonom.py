# скачиание данных по ковид-19, S&P500 и других
import pandas as pd
import numpy as np
from servise_ds import okno

# первичная загрузка файлов микробизнеса
def start0():
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

def start_covid():
    # ковид - смертность
    covid_deaths = pd.read_csv("C:\\kaggle\\МикроБизнес\\time_series_covid19_deaths_US.csv")
    # ковид - заболеваемость
    covid_ill = pd.read_csv("C:\\kaggle\\МикроБизнес\\time_series_covid19_confirmed_US.csv")
    # безработца
    unemploy = pd.read_csv("C:\\kaggle\\МикроБизнес\\UNEMPLOY.csv")
    # S&P500
    sp500 = pd.read_csv("C:\\kaggle\\МикроБизнес\\sp500_index.csv")
    return covid_deaths, covid_ill

def glavnaia_covid(raw):
    covid_deaths, covid_ill = start_covid()
    # удаляем ненужные колонки
    covid_deaths.drop(['UID','iso2','iso3','code3','Country_Region','Lat', 'Long_', 'Combined_Key'],
                      axis=1, inplace=True)
    covid_ill.drop(['UID', 'iso2', 'iso3', 'code3', 'Country_Region', 'Lat', 'Long_', 'Combined_Key'],
                   axis=1, inplace=True)

    # создаем новые колонки
    raw['covill'] = 0 # количество заболевших
    raw['covdeat'] = 0 # количество жертв
    raw['Population'] = 0 # численность населения из файла жертв ковида
    #raw['naselenie'] = raw['active']/raw['microbusiness_density']*100
    raw['proc_covill'] = 0 # процент заболевших
    raw['proc_covdeat'] = 0 # процент жертв

    oll_coll = covid_ill.columns # список колонок в датафрейме заболевших

    for cfips in raw.cfips.unique(): # цикл по уникальным cfips
        print(cfips)
        last_ill = 0 # кол. заболевших в предыдущем месяце
        last_deaths = 0 # кол. жертв в предыдущем месяце
        maska_cfips = raw['cfips']==cfips
        df_cfips = raw[maska_cfips]

        maska_deaths = (covid_deaths['FIPS'] == cfips)  # маска 1 строки из датафрейма жертв
        population = covid_deaths.loc[maska_deaths, 'Population']  # население из датафрейма жертв
        population.reset_index(drop=True, inplace=True)
        population = population[0]  # население из датафрейма жертв

        maska_ill = (covid_ill['FIPS'] == cfips)  # маска 1 строки из датафрейма заболевших

        for mes in range(0, 47): # цикл по месяцам
            maska_mes = raw['dcount']==mes
            raw.loc[maska_cfips & maska_mes, 'Population'] = population
            row = df_cfips[maska_mes]
            row.reset_index(drop= True , inplace= True )
            ye = row["year"][0] # год в датафрейме микробизнеса
            mon = row["month"][0] # месяц в датафрейме микробизнеса
            ye = str(ye) # год в датафрейме микробизнеса
            mon = str(mon) # месяц в датафрейме микробизнеса
            colonka = mon + '/1/'+ ye[2:] # формируем строку - имя колонки из датафреймов ковид19
            if colonka in oll_coll: # если эта строка присутствует в списке названий колонок датафрейма ковид19
                # жертвы
                deaths = covid_deaths.loc[maska_deaths, colonka]
                deaths.reset_index(drop= True , inplace= True )
                deaths = deaths[0] # количество жертв
                raw.loc[maska_cfips & maska_mes, 'covdeat'] = deaths - last_deaths # количество жертв за месяц
                raw.loc[maska_cfips & maska_mes, 'proc_covdeat'] = (deaths - last_deaths)/population*100
                last_deaths = deaths
                # заболевшие
                ill = covid_ill.loc[maska_ill, colonka]
                ill.reset_index(drop= True , inplace= True )
                ill = ill[0] # количество заболевших
                raw.loc[maska_cfips & maska_mes, 'covill'] = ill - last_ill # количество заболевших за месяц
                raw.loc[maska_cfips & maska_mes, 'proc_covill'] = (ill - last_ill)/population*100
                last_ill = ill
    return raw

def start_ekonom():
    # безработца
    unemploy = pd.read_csv("C:\\kaggle\\МикроБизнес\\UNEMPLOY.csv")
    # S&P500
    sp500 = pd.read_csv("C:\\kaggle\\МикроБизнес\\sp500_index.csv")
    return unemploy, sp500

def glavnaia_ekonom(raw, unemploy, sp500):
    unemploy['data_new'] = pd.to_datetime(unemploy["DATE"])
    sp500['data_new'] = pd.to_datetime(sp500["Date"])
    raw['first_day_of_month'] = pd.to_datetime(raw["first_day_of_month"])
    # создаем новые колонки
    raw['unemploy'] = 0
    raw['sp500'] = 0
    predun = 0 # предыдущее значение 'UNEMPLOY'

    for i in raw["first_day_of_month"].unique(): # цикл по месяцам
        un = unemploy.loc[unemploy['data_new'] == i, 'UNEMPLOY'] # безработица в нужном месяце
        un.reset_index(drop=True, inplace=True)
        if len(un) != 0:
            un = un[0]
            predun = un
        else:
            un = predun
        raw.loc[raw["first_day_of_month"] == i, 'unemploy'] = un
        l = 0
        datt = i
        while l == 0: # цикл на случай если на 1-е число нет sp500, ищем предыдущее значение
            sp = sp500.loc[sp500['data_new'] == datt, 'S&P500']
            datt = datt - np.timedelta64(1, 'D')
            l = len(sp)
            pass
        sp.reset_index(drop=True, inplace=True)
        sp = sp[0]
        raw.loc[raw["first_day_of_month"] == i, 'sp500'] = sp
    return raw


if __name__ == "__main__":
    # train, test, census = start0() # загрузка файлов
    # raw = maceraw(train, test) # объединенный массив трейна и теста, создание объединенного raw
    # raw = censusdef(census, raw) # объединение census и raw
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw0.csv", index = False)

    # raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw0.csv")
    # raw = glavnaia_covid(raw)
    # raw.to_csv("C:\\kaggle\\МикроБизнес\\raw0_covid.csv", index=False)

    raw = pd.read_csv("C:\\kaggle\\МикроБизнес\\raw0_covid.csv")
    unemploy, sp500 = start_ekonom()
    raw = glavnaia_ekonom(raw, unemploy, sp500)
    raw.to_csv("C:\\kaggle\\МикроБизнес\\raw0_cov_econ.csv", index=False)