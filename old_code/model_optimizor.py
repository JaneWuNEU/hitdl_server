import sys
sys.path.append(".")
#from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pickle
import datetime
import math
from utils.model_info import ModelInfo
#import cvxpy as cp
import os
from utils.util import FileOperation
savedStdout = sys.stdout
fileOpr = FileOperation()
class ModelOptimizor:
    def __init__(self):
        ##print("create model optimizor")
        self.group_dict = {"inception_v3":"1"}
        self.C = 48
        self.count = 0
        self.id = 1

    def get_grouped_info(self,model_id_list,band_width):
        group_user_number = {}
        group_user_upload_bw = {}
        group_user_download_bw = {}
        grouped_modelid = {}
        for model_id in model_id_list:
            try:
                #1. get model type and record the number of users accessing this model
                model_name = (model_id.split("*")[0]).lower()
                user_count = int(model_id.split("*")[3])
                #model_group_num = self.group_dict[model_type]
                user_upload_bw = band_width[model_id][0]
                user_download_bw = band_width[model_id][1]
                if model_name in group_user_number.keys():
                    group_user_number[model_name] = group_user_number[model_name]+1
                    group_user_upload_bw[model_name].append(user_upload_bw)
                    group_user_download_bw[model_name].append(user_download_bw)
                    grouped_modelid[model_name].append(model_id)
                else:
                    group_user_number[model_name] = 1
                    group_user_upload_bw[model_name] = [user_upload_bw]
                    group_user_download_bw[model_name] = [user_download_bw]
                    grouped_modelid[model_name] = [model_id]
                '''
                #2. record the bandwidth of each group
                print("user_count",user_count)
                user_upload_bw = band_width[model_id][0]
                user_download_bw = band_width[model_id][1]
                if model_name in group_user_number.keys():
                    group_user_number[model_name] = user_count
                    group_user_upload_bw[model_name].append(user_upload_bw)
                    group_user_download_bw[model_name].append(user_download_bw)
                    grouped_modelid[model_name].append(model_id)
                else:
                    group_user_number[model_name] = user_count
                    group_user_upload_bw[model_name] = [user_upload_bw]
                    group_user_download_bw[model_name] = [user_download_bw]
                    grouped_modelid[model_name] = [model_id]
                '''
            except Exception as e:
                print("error happens when getting user info",e)
        return group_user_number,group_user_upload_bw,group_user_download_bw,grouped_modelid
    def get_weight_based_strategy(self,model_id_list, band_width, weight):
        weight={"inception_v3":1}
        group_user_number,group_user_upload_bw,group_user_download_bw,grouped_modelid = self.get_grouped_info(model_id_list, band_width)
        strategy = {}
        batch = 2
        r1 = 5
        for model_name in group_user_number.keys():
            group_num = self.group_dict[model_name]
            intra = math.floor(weight[model_name]*self.C)
            if model_name is "inception_v3":
                r1 = 5
            else:
                r1 = 3
            strategy[group_num] = {"user_list": grouped_modelid[model_name], \
                                "model_type": model_name, "r1": r1, "device": 'CPU', "batch": batch, "intra": intra}
        return strategy
    def get_average_strategy(self,model_id_list, band_width, weight):
        group_user_number,group_user_upload_bw,group_user_download_bw,grouped_modelid = self.get_grouped_info(model_id_list, band_width)
        strategy = {}
        batch = 2
        r1 = 5
        for model_name in group_user_number.keys():
            group_num = self.group_dict[model_name]
            intra = self.C/3
            if model_name is "inception_v3":
                r1 = 5
            else:
                r1 = 3
            strategy[group_num] = {"user_list": grouped_modelid[model_name], \
                                "model_type": model_name, "r1": r1, "device": 'CPU', "batch": batch, "intra": intra}
        return strategy
    def get_strategy(self,band_width):
        strategy = {
        "inception": {"r1":0, "intra":2, "ins_amount":1, "user_amount":2},
        "resnet": {"r1":0, "intra":2, "ins_amount":1, "user_amount":2},
        "mobilenet": {"r1":0, "intra":2, "ins_amount":1, "user_amount":2}
        }
        return strategy
"""
opt = ModelOptimizor()
model_id_list = ["alexnet_1"]
u = 100
for u in np.arange(30,900,10):
    bandwidth = {"alexnet_1":[u, u]}
    result = opt.get_strategy(model_id_list,bandwidth,None)
    if len(result.keys())!=0:
        #print(u)
        break

model_id_list = ['alexnet_192.168.31.151_M1_1','srgan_192.168.31.151_M1_1','vgg16_192.168.31.151_M1_1','autoencoder_192.168.31.151_M1_1']
bandwidth_dic = {model_id_list[0]:[18.88,54.97],model_id_list[1]:[18.88,54.97],model_id_list[2]:[18.88,54.97]}

#model_id_list = ['autoencoder_192.168.31.151_M1_1']
#bandwidth_dic = {model_id_list[0]:[18.88,54.97]}
#model_id_list = ['srgan_192.168.31.151_M1_1','autoencoder_192.168.31.151_M1_1','alexnet_192.168.31.151_M1_1']
#bandwidth_dic = {model_id_list[0]:[18.88,54.97],model_id_list[1]:[18.88,54.97],model_id_list[2]:[18.88,54.97]}
#print("主函数上传，下载带宽",18.88,54.97)
opt = ModelOptimizor()
opt.get_strategy(model_id_list,bandwidth_dic)
"""

