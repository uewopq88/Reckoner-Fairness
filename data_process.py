import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
import torchvision
import os
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

def load_COMPAS():
    data_name = "compas-scores-two-years"
    sensitive = 'race'
    predict_attr = "is_recid"
    df = pd.read_csv("./data/{}.csv".format(data_name))
    df = df.query("race in ('African-American','Caucasian')")
    intertsted_columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                'age',
                'c_charge_degree',
                'c_charge_desc',
                'age_cat',
                'sex', 'race', 'is_recid']

    header = list(intertsted_columns)
    header.remove(predict_attr)

    X = df[header]
    groundtruths = df[predict_attr].values

    # use FeatureHasher to encode categorical features
    categorical_features = X[['c_charge_desc']].apply(lambda x: x.astype(str).tolist(), axis=1)
    hasher = FeatureHasher(n_features=20, input_type='string')
    hashed_features = hasher.transform(categorical_features)
    X = X[['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
                'age','sex', 'race', 'c_charge_degree', 'age_cat']]
    # add new hashed features to DataFrame
    for i in range(hashed_features.shape[1]):
        X[f'hashed_feature_{i+1}'] = hashed_features.toarray()[:, i]


    X = pd.get_dummies(X)
    X.reset_index(drop=True, inplace=True)
    X = X.sort_index(axis=1)

    def f(row):
        if row['race_African-American'] == 1:
            val = 0
        if row['race_Caucasian'] == 1:
            val = 1
        return val

    X['sensitive_info'] = X.apply(f, axis=1)
    X = X[X.columns.drop(list(X.filter(regex = sensitive)))]

    return X, groundtruths

def load_newAdult():
    data_name = "new_adult"
    sensitive = 'race'
    predict_attr = "income_label"
    df = pd.read_csv("./data/{}.csv".format(data_name))
    header = list(df.columns)
    header.remove(predict_attr)
    X = df[header]
    groundtruths = df[predict_attr].values

    categorical_features = X[['workclass', 'education', 'marital-status',
                              'relationship', 'gender', 'native-country',
                              'occupation']].apply(lambda x: x.astype(str).tolist(), axis=1)
    hasher = FeatureHasher(n_features=45, input_type='string')
    hashed_features = hasher.transform(categorical_features)
    X = X[['hours-per-week', 'age', 'capital-gain', 'capital-loss', 'education-num', 'race']]
    for i in range(hashed_features.shape[1]):
        X[f'hashed_feature_{i+1}'] = hashed_features.toarray()[:, i]

    X['sensitive_info'] = X['race'].apply(lambda x: 1 if x == 'White' else 0)
    X = X[X.columns.drop(list(X.filter(regex = sensitive)))]

    return X, groundtruths


def load_train_test_valid(X, y, f, train_ratio, valid_ratio):
    class A_C(Dataset):
        def __init__(self, attrs, labels):
            self.attrs = attrs
            self.labels = labels

        def __getitem__(self, idx):
            return [self.attrs[idx], self.labels[idx]]

        def __len__(self):
            return len(self.labels)

    data_size = len(X)
    true_test_ratio = 1 - (train_ratio + valid_ratio)
    rest_data = data_size * (1 - true_test_ratio)
    true_valid_ratio = (rest_data - data_size * train_ratio) / rest_data

    train, valid, train_labels, valid_labels = train_test_split(X, y, test_size=true_valid_ratio, random_state=7)
    if f != 'highConf' and f != 'lowConf':
        header = list(X.columns)
        header = header[:-1]
        ct = ColumnTransformer([('_', StandardScaler(), header)], remainder='passthrough')
        X = ct.fit_transform(X)
        print('after scalling: ', X.shape)
        train, test, train_labels, test_labels = train_test_split(X, y, test_size=true_test_ratio, random_state=7)
        train, valid, train_labels, valid_labels = train_test_split(train, train_labels, test_size=true_valid_ratio, random_state=7)

    train_set = A_C(train, train_labels)
    print("len of train labels: ", train_set.__len__())
    valid_set = A_C(valid, valid_labels)
    print("len of valid labels: ", valid_set.__len__())
    train_size = len(train)
    valid_size = len(valid)
    test_size = 0
    test_set = []
    if f != 'highConf' and f != 'lowConf':
        test_set = A_C(test, test_labels)
        test_size = len(test)
        print("len of test labels: ", test_set.__len__())

    print("The size of training set is: {}".format(train_size))
    print("The size of testing set is: {}".format(test_size))
    print("The size of valid set is: {}".format(valid_size))

    data_size = [train_size, test_size, valid_size]
    return train_set, valid_set, test_set, data_size
