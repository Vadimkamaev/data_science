import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
import xgboost
from tabpfn import TabPFNClassifier
import warnings
warnings.filterwarnings("ignore")

# LOAD THE DATA
maindf = pd.read_csv('C:\\kaggle\\Возраст\\train.csv')
testdf = pd.read_csv('C:\\kaggle\\Возраст\\test.csv')
greeksdf = pd.read_csv('C:\\kaggle\\Возраст\\greeks.csv')

maindf['EJ'] = maindf['EJ'].replace({'A': 0, 'B': 1})
testdf['EJ']  = testdf['EJ'].replace({'A': 0, 'B': 1})


# Greeks содержат информацию о времени, которую мы можем использовать, нам просто нужно разобрать ее на int/nan.
times = greeksdf.Epsilon.copy()
times[greeksdf.Epsilon != 'Unknown'] = greeksdf.Epsilon[greeksdf.Epsilon != 'Unknown'].map(lambda x: datetime.strptime(x,'%m/%d/%Y').toordinal())
times[greeksdf.Epsilon == 'Unknown'] = np.nan

# Set predictor and target columns

predictors = [n for n in maindf.columns if n != 'Class' and n != 'Id']

class WeightedEns(BaseEstimator):
    def __init__(self):
        self.classifiers = [xgboost.XGBClassifier(), TabPFNClassifier(N_ensemble_configurations=64, device='cpu')]
        self.imp = SimpleImputer(missing_values=np.nan, strategy='median')

    def fit(self, X, y):
        cls, y = np.unique(y, return_inverse=True)
        self.classes_ = cls
        X = self.imp.fit_transform(X)
        for cl in self.classifiers:
            cl.fit(X, y)

    def predict_proba(self, X):
        X = self.imp.transform(X)
        ps = np.stack([cl.predict_proba(X) for cl in self.classifiers])
        p = np.mean(ps, axis=0)
        class_0_est_instances = p[:, 0].sum()
        others_est_instances = p[:, 1:].sum()
        # we reweight the probs, since the loss is also balanced like this
        # our models out of the box optimize CE
        # with these changes they optimize balanced CE
        new_p = p * np.array(
            [[1 / (class_0_est_instances if i == 0 else others_est_instances) for i in range(p.shape[1])]])
        return new_p / np.sum(new_p, axis=1, keepdims=1)

pred_and_time = pd.concat((maindf[predictors], times), 1)

test_predictors = np.array(testdf[predictors])
test_pred_and_time = np.concatenate((test_predictors, np.zeros((len(test_predictors),1)) + pred_and_time.Epsilon.max()+1),1)

m = WeightedEns()
m.fit(np.array(pred_and_time),np.array(greeksdf['Alpha']))
p = m.predict_proba(test_pred_and_time)
assert (m.classes_[0] == 'A')
p0 = p[:,:1]
p0[p0 > 0.95] = 1
p0[p0 < 0.05] = 0
submit=pd.DataFrame(testdf["Id"], columns=["Id"])
submit["class_0"] = p0
submit["class_1"] = 1 - p0
submit.to_csv('submission.csv',index=False)

# p = np.concatenate((p[:,:1],np.sum(p[:,1:],1,keepdims=True)), 1)
# # p[p[0]>0.9]=[1,0]
# result_df = pd.concat((testdf['Id'],pd.DataFrame(p, columns=('class_0', 'class_1'))),axis=1)
# result_df.to_csv('submission.csv',index=False)

pd.read_csv('submission.csv')