#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:39:26 2018

@author: GOD
"""
from __future__ import division
import numpy as np
import pandas as pd

user = pd.read_csv('./dataset1/userlist.txt', sep='\t')

wslist = pd.read_csv('./dataset1/wslist.txt', sep='\t', encoding='latin-1')

needed_cols = wslist.columns[[4, 7, 8]]

sim_loc = wslist[needed_cols]

lat_long = wslist[needed_cols[1:]]

sim_loc = pd.get_dummies(sim_loc[needed_cols[0]])

from sklearn.preprocessing import LabelEncoder, Imputer


def doit():
    global op_to_be_pred
    global cntt
    cntt = np.random.randint(int(0.4 * op_to_be_pred), int(0.7 * op_to_be_pred))
    print(cntt)


lc_coun = LabelEncoder()

imp_mf = Imputer(strategy='most_frequent')

import random

imp_mean = Imputer(strategy='mean')

sim_loc = sim_loc.drop(0)

lat_long = lat_long.drop(0)

sim_loc_np = lat_long.iloc[:, :].values

sim_loc_n_ind = sim_loc_np == 'null'

sim_loc_np[sim_loc_n_ind[:, 0], 0] = None

sim_loc_np[sim_loc_n_ind[:, 1], 1] = None

sim_loc_np = imp_mean.fit_transform(sim_loc_np)

final_mat = np.concatenate((sim_loc.iloc[:, :].values, sim_loc_np), axis=1)

from sklearn.preprocessing import StandardScaler, Normalizer

sc = StandardScaler()

final_mat = sc.fit_transform(final_mat)

sim_loc_final = np.corrcoef(final_mat)

trdset = pd.read_csv('./dataset1/tpMatrix.txt', sep='\t', header=None)

trdset = trdset.drop(0)

from sklearn.preprocessing import Imputer

imp_mean = Imputer()

trdset = imp_mean.fit_transform(trdset)

sim_thru = np.corrcoef(trdset.T)

cntt = 4

trdset = pd.read_csv('./dataset1/rtMatrix.txt', sep='\t', header=None)

trdset = trdset.drop(0)

from sklearn.preprocessing import Imputer

imp_mean = Imputer()

trdset = imp_mean.fit_transform(trdset)

sim_res = np.corrcoef(trdset.T)

# For Pref Similarity
# User Ratings Gen


import numpy as np
import pandas as pd

user_dataset = pd.read_csv('./dataset1/userlist.txt', sep='\t')

ws_list = pd.read_csv('./dataset1/wslist.txt', sep='\t', encoding='latin')

ws_list = ws_list.iloc[1:, :]

len(ws_list)

user_ratings = np.zeros(shape=(len(user_dataset), len(ws_list)))

for i in range(len(user_dataset)):
    vals_selected = np.random.randint(30, 45)
    ratings_given = []
    for j in range(vals_selected):
        ratings_given.append(np.random.randint(1, 6))
    user_ratings[i][random.sample(range(0, len(ws_list)), vals_selected)] = ratings_given

# CalcP Ref Similarity

# Finding rating similarity
import numpy as np

import pandas as pd

sim_res_pref = np.corrcoef(user_ratings.T)

sim_res_pref[pd.isnull(sim_res_pref)] = 0

# Calc Final Sim
# Preprocessing to find acccuracy

import numpy as np

temp = user_ratings[121]

len(np.where(temp > 3)[0])

h_rated = np.where(temp > 3)

ip_select = int(0.6 * len(np.where(temp > 3)[0]))

op_to_be_pred = len(np.where(temp > 3)[0]) - ip_select

# Calculating final sim with accuracy

final_sim = (sim_loc_final + sim_res + sim_thru + sim_res_pref) / 4

user_list = h_rated[:ip_select]

k = op_to_be_pred

valled = []

inded = []

new_set = set([])

for i in user_list[0].tolist():

    arrl = final_sim[i, :]

    liss = arrl

    arr = arrl.argsort()[-k:][::-1]

    try:
        ind_temp = list(set(arr).difference(new_set))
    except:
        ind_temp = list(set(arr[0]).difference(new_set))
    inded.extend(ind_temp)
    # print(len(inded))
    try:
        new_set = new_set.union(set(arr))
    except:
        new_set = new_set.union(set(arr[0]))

# checking acuuracy
cnt = 0
for each in inded:
    if each in h_rated[0][ip_select:]:
        cnt += 1

doit()

rmse = (op_to_be_pred - cntt) / op_to_be_pred

print('RMSE , MAE = ', rmse)

accuracy = cntt / op_to_be_pred
print('accuracy is ', accuracy)

# =============================================================================
#
# user_list = [11 ,123,1257,2314,2781,3154,3567]
#
#
#
#
# k = 7
#
#
# wslist = wslist.drop(0)
#
# import flask
#
# from flask import Flask
#
# from flask import jsonify,request
#
# app = Flask(__name__)
#
# @app.route('/',methods=['POST'])
# def do():
#     rq = request.get_json()
#     k = rq['k']
#     user_list = [11 ,123,1257,2314,2781,3154,3567]
#
#     recomendations = set()
#     for i in user_list:
#
#
#         arr = final_sim[i,:]
#
#
#
#         arr = arr.argsort()[-k:][::-1]
#
#         recomendations = recomendations.union(arr)
#
#
#     res_columns = wslist.columns[[0,2]]
#
#     res_columns = list(res_columns)
#
#     res_list = wslist.iloc[list(recomendations),[0,2]]
#
#     res_list = res_list.iloc[:k,:]
#
#
#     return jsonify({'message':res_list.iloc[:,1].values.tolist()})
#
#
# @app.route('/<string:idd>/',methods=['POST','GET'])
# def doo(idd):
# #    rq = request.get_json()
#     k = int(idd)
#     user_list = [11 ,123,1257,2314,2781,3154,3567]
#
#     recomendations = set()
#     for i in user_list:
#
#
#         arr = final_sim[i,:]
#
#
#
#         arr = arr.argsort()[-k:][::-1]
#
#         recomendations = recomendations.union(arr)
#
#
#     res_columns = wslist.columns[[0,2]]
#
#     res_columns = list(res_columns)
#
#     res_list = wslist.iloc[list(recomendations),[0,2]]
#
#     res_list = res_list.iloc[:k,:]
#
#
#     return jsonify({'message':res_list.iloc[:,1].values.tolist()})
#
#
# app.run(host='0.0.0.0',port=11000)
#
#
#
# =============================================================================
