from servise_ds import okno
import pandas as pd

#сверяем сходство 2-х файлов
def sverka():
    id = 'row_id'
    stolbec = 'microbusiness_density' # столбец, который сверяем
    file1 = pd.read_csv("C:\\kaggle\\МикроБизнес\\sub_xgb.csv")
    file2 = pd.read_csv("C:\\kaggle\\МикроБизнес\\dlia_sverki.csv")
    file1['raznost'] = file1[stolbec] - file2[stolbec]
    okno.vewdf(file1)

sverka()




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
#melk1()






# rez = pd.read_csv("C:\\kaggle\\белки\\train1.csv")
# rez1 = rez[(rez['lenstr']> 60)&(rez['lenstr']< 300) ]
# pass
