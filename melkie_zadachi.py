from servise_ds import okno
import pandas as pd

def melk():
    train = pd.read_csv("C:\\kaggle\\МикроБизнес\\train.csv")
    train.sort_values(by='microbusiness_density', ascending=False, inplace=True)
    subideal = pd.read_csv("C:\\kaggle\\МикроБизнес\\subideal.csv")
    subideal.sort_values(by='microbusiness_density', ascending=False, inplace=True)
    rrrr = train[train['cfips'] == 56033]

    train["first_day_of_month"] = pd.to_datetime(train["first_day_of_month"])
    train = train.sort_values(['cfips','first_day_of_month']).reset_index(drop=True)
    test = pd.read_csv('C:\\kaggle\\МикроБизнес\\test.csv')
    test["first_day_of_month"] = pd.to_datetime(test["first_day_of_month"])
    test = test.sort_values(['cfips', 'first_day_of_month']).reset_index(drop=True)

    train10 = train[train["first_day_of_month"] == '2022-10-01 00:00:00']
    target10 = train10[["cfips", "microbusiness_density"]]
    target10.set_index("cfips", inplace=True)
    sub_delta = pd.read_csv("C:\\kaggle\\МикроБизнес\\sub_delta1_114.csv")
    # sub_delta = sub_delta1_114[sub_delta1_114["first_day_of_month"] == '2022-11-01 00:00:00']
    test = test.join(target10, on="cfips")
    test = test[["row_id", "microbusiness_density"]]
    test.set_index("row_id", inplace=True)
    sub_delta.set_index("row_id", inplace=True)
    test = (test * 14 + sub_delta) / 15
    test.to_csv("C:\\kaggle\\МикроБизнес\\submiks3.csv")
    #test.to_csv("C:\\kaggle\\МикроБизнес\\submiks.csv", index=False)




def melk0():
    rez = pd.read_csv("C:\\kaggle\\белки\\reztest2.csv")
    rez.drop(columns='Unnamed: 0',inplace=True)
    rez['seq_id'] = rez['seq_id'].astype(int)
    rez.to_csv("C:\\kaggle\\белки\\reztest4.csv", index=False)
#

def melk1():
    rez = pd.read_csv("C:\\kaggle\\Гравитация\\sub3.csv")
    # rez.sort_values(by='tm', inplace=True)
    # rez['tmm']=range(len(rez))
    # rez.sort_values(by='seq_id', inplace=True)
    sub0 = pd.read_csv("C:\\kaggle\\Гравитация\\sub_best.csv")
    sub1 = pd.read_csv("C:\\kaggle\\Гравитация\\sub2.csv")
    newdf = pd.DataFrame(columns=['id','target'])
    for i in range(len(rez)):
        newdf.loc[i, 'id'] = rez.loc[i,'id']
        if abs(sub0.loc[i,'target'] - rez.loc[i,'target']) > abs(sub1.loc[i,'target'] - rez.loc[i,'target']):
            newdf.loc[i, 'target'] = sub1.loc[i, 'target']
        else:
            newdf.loc[i, 'target'] = sub0.loc[i, 'target']
    newdf.to_csv("C:\\kaggle\\Гравитация\\reztest.csv", index=False)

    pass
    # rez.drop(columns='Unnamed: 0',inplace=True)
    # rez['seq_id'] = rez['seq_id'].astype(int)
    # rez.to_csv("C:\\kaggle\\белки\\reztest4.csv", index=False)
melk1()


def erunda():
    rez = pd.read_csv("C:\\kaggle\\белки\\su0603experim.csv")
    print(rez[rez['tm'] < 5])

def umnoshitdf():
    rez1 = pd.read_csv("C:\\kaggle\\белки\\submiss1.csv")
    rez = pd.read_csv("C:\\kaggle\\белки\\submission0603.csv")
    rez2=rez.copy()
    rez2['tm2'] = rez1['tm']
    rez2['razn'] = rez2['tm2']-rez2['tm']
    maska = rez2['razn'].abs() > 1
    print(maska.sum())
    maska1 = maska & (maska.index >= 600)& (maska.index < 800)
    print(maska1.sum())
    rez1.loc[maska1] = rez.loc[maska1]
    rez1.to_csv("C:\\kaggle\\белки\\submiss2.csv", index=False)
    # базовым взят submission1_0603.csv.
    # замена 1-х 200 строк на строки из submission0603.csv подняла с 417 на 175 место - submiss1.csv
    # замена cтрок c 200 до 400 на строки из submission0603.csv вредна
    # замена cтрок c 400 до 600 на строки из submission0603.csv вредна


#umnoshitdf()




# rez = pd.read_csv("C:\\kaggle\\белки\\train1.csv")
# rez1 = rez[(rez['lenstr']> 60)&(rez['lenstr']< 300) ]
# pass
