import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class LogNormalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, attr_list=None): # no *args **kwargs
        self.attr_list = attr_list

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # for attr in self.attr_list:
        #     X[attr] = np.log1p(X[attr])
        return np.log1p(X)

class AddingContFeatures(BaseEstimator, TransformerMixin):

    def __init__(self): # no *args **kwargs
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["OverallFlrSF"] = X["1stFlrSF"] + X["2ndFlrSF"]
        return X

class CatNomImputer(BaseEstimator, TransformerMixin):

    def __init__(self,  impute_with, attr_list=None,): # no *args **kwargs
        self.attr_list = attr_list
        self.impute_with = impute_with

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # for attr in self.attr_list:
        #     X[attr] = X[attr].fillna(self.impute_with)
        return X.fillna(self.impute_with)

class CatNomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, attr_list=None, drop_first=True): # no *args **kwargs
        self.attr_list = attr_list
        self.drop_first = drop_first

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X1 = X[[attr for attr in X.columns if attr not in self.attr_list]]
        # X2 = pd.get_dummies(X[self.attr_list], drop_first=self.drop_first)
        return pd.get_dummies(X, drop_first=self.drop_first)


class CatOrdImputer(BaseEstimator, TransformerMixin):

    def __init__(self, attr_list=None):
        self.attr_list = attr_list
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for attr in X.columns:
            X[attr] = X[attr].fillna(X[attr].mode()[0])
        return X

class CatOrdEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['MSZoning'] = X['MSZoning'].replace({'C (all)':1,'RM':2,'RH':3,'RL':4,'FV':5})
        X['Condition1'] = X['Condition1'].replace({'Artery':1,
                                                'RRAe':1,
                                                'RRNn':1,
                                                'Feedr':1,
                                                'RRNe':1,
                                                'RRAn':1,
                                                'Norm':2,
                                                'PosA':3,
                                                'PosN':3})
        X['Condition2'] = X['Condition2'].replace({'RRNn':1,
                                                'Artery':2, 
                                                'Feedr':2,
                                                'RRAn':2,
                                                'RRAe':2,    
                                                'Norm':2,
                                                'PosA':3,
                                                'PosN':3})
        X['HouseStyle'] = X['HouseStyle'].apply(self.HouseStyleToInt)
        X['MasVnrType'] = X['MasVnrType'].apply(self.fit_transformMasVnrTypeToInt)
        X['Foundation'] = X['Foundation'].replace({'Slab':1,'BrkTil':2,'Stone':2,'CBlock':3,'Wood':4,'PConc':5})
        X['GarageType'] = X['GarageType'].replace({'CarPort':1,'Basment':2,'Detchd':2,'Attchd':3,'2Types':3,'BuiltIn':4})
        X['GarageFinish'] = X['GarageFinish'].replace({'Unf':1,'RFn':2,'Fin':3})
        X['PavedDrive'] = X['PavedDrive'].replace({'N':1,'P':2,'Y':3})
        X['SaleCondition'] = X['SaleCondition'].replace({'AdjLand':1,'Abnorml':2,'Family':2,'Alloca':2,'Normal':3,'Partial':4})
        ext_lable = {'AsbShng':1,'AsphShn':1,
                    'MetalSd':2,'Wd Sdng':2,'WdShing':2, 'Wd Shng':2,'Stucco':2,'CBlock':2,
                    'HdBoard':3,'BrkFace':3,'Plywood':3,'Other':3,
                    'VinylSd':4,'CemntBd':4,'BrkComm':4,'CmentBd':4,'Brk Cmn':4,
                    'Stone':5,'ImStucc':5 }
        X['Exterior1st'] = X['Exterior1st'].replace(ext_lable)
        X['Exterior2nd'] = X['Exterior2nd'].replace(ext_lable)
        X['BsmtExposure'] = X['BsmtExposure'].apply(self.BsmtExposureToInt)
        X['BsmtFinType1'] = X['BsmtFinType1'].apply(self.BsmtFinType1ToInt)
        quality_label = {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

        X['ExterQual'] = X['ExterQual'].replace(quality_label)
        X['ExterCond'] = X['ExterCond'].replace(quality_label)
        X['KitchenQual'] = X['KitchenQual'].replace(quality_label)
        X['HeatingQC'] = X['HeatingQC'].replace(quality_label)
        X['BsmtQual'] = X['BsmtQual'].replace(quality_label)
        X['BsmtCond'] = X['BsmtCond'].replace(quality_label)
        X['FireplaceQu'] = X['FireplaceQu'].replace(quality_label)
        X['GarageQual'] = X['GarageQual'].replace(quality_label)
        X['PoolQC'] = X['PoolQC'].replace(quality_label)

        return X

    def HouseStyleToInt(self, x):
        if(x=='1.5Unf'):
            r = 0
        elif(x=='SFoyer'):
            r = 1
        elif(x=='1.5Fin'):
            r = 2
        elif(x=='2.5Unf'):
            r = 3
        elif(x=='SLvl'):
            r = 4
        elif(x=='1Story'):
            r = 5
        elif(x=='2Story'):
            r = 6  
        elif(x==' 2.5Fin'):
            r = 7          
        else:
            r = 8
        return r

    def MasVnrTypeToInt(self, x):
        if(x=='Stone'):
            r = 3
        elif(x=='BrkFace'):
            r = 2
        elif(x=='BrkCmn'):
            r = 1        
        else:
            r = 0
        return r

    def BsmtExposureToInt(self, x):
        if(x=='Gd'):
            r = 4
        elif(x=='Av'):
            r = 3
        elif(x=='Mn'):
            r = 2
        elif(x=='No'):
            r = 1
        else:
            r = 0
        return r

    def BsmtFinType1ToInt(self, x):
        if(x=='GLQ'):
            r = 6
        elif(x=='ALQ'):
            r = 5
        elif(x=='BLQ'):
            r = 4
        elif(x=='Rec'):
            r = 3   
        elif(x=='LwQ'):
            r = 2
        elif(x=='Unf'):
            r = 1        
        else:
            r = 0
        return r



dis_pipeline = [("dis_imputer", SimpleImputer(strategy="median")),
                ("std_scale", StandardScaler())]

cont_pipeline = [("cont_imputer", SimpleImputer(strategy="median")),
                ("add_feature", AddingContFeatures()),
                ("log_normal", LogNormalTransformer())]

date_time_pipeline = [("std_scaler"), StandardScaler()]

cat_nom_pipeline = [("nom_imputer", CatNomImputer("missing")),
                    ("nom_encoding", CatNomEncoder())]

cat_ord_pipeline = ["ord_imputer", CatOrdImputer(), 
                    "ord_encoding", CatOrdEncoder()]

label_pipeline = [("log_normal", LogNormalTransformer())]