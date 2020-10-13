import xml.dom.minidom
import joblib
import sys
import pandas as pd
sys.path.append(".")
from utils.util import ModelInfo
import math
import copy
import numpy as np
import os
import cvxpy as cp
import datetime
class Background:
    def __init__(self):
        self.mo  = ModelOptimizor()
    def model_partition(self):
        model_name = "inception"
        '''
        #=========model partition 
        k_list = [4,7]
        intra_list = [2,4]
        net_list = [105,93]
        SLA = self.mo.model_mobile_time[model_name]*0.9
        '''
        k_list = [0]
        intra_list = [2,8]
        net_list = [105]
        result = {"k": [], "intra": [], "throughput": [], "net": [],"e2e":[],"SLA":[],"net_throughput":[],"edge_throughput":[]}
        SLA = 0.205
        #result_writer = pd.ExcelWriter("background/model_partition/inception_model_partition.xlsx")
        result_writer = pd.ExcelWriter("background/resource_allocation/inception_ins_size.xlsx")
        for net in net_list:
            bd = net*1024*1024/8
            for k in k_list:
                for intra in intra_list:
                    # 1. get e2e
                    e2e,edge_latency,upload_latency = self.mo.get_model_e2e_no_inter(model_name,k,intra,bd)
                    result["k"].append(k)
                    result["intra"].append(intra)
                    result["e2e"].append(e2e)
                    result["net"].append(net)
                    result["throughput"].append(1.0/edge_latency/min(self.mo.model_frame_rate[model_name],1/upload_latency))
                    result["net_throughput"].append(1/upload_latency)
                    result["edge_throughput"].append(1 / edge_latency)
                    result["SLA"].append(SLA)
        #print(result)
        result_pd = pd.DataFrame(data=result,index=range(len(result["k"])))
        result_pd.to_excel(excel_writer=result_writer)
        result_writer.save()
        result_writer.close()

    def resource_allocation(self):
        """
        overall throughput when models are partitioned at Input_layer without any interference.

        :return:
        """
        net = 93
        bandwidth = {"inception":net*1024*1024/8,"resnet":net*1024*1024/8,"mobilenet":net*1024*1024/8}
        SLA = 0.205
        mo = ModelOptimizor()
        ins_size = mo.get_baseline_ins_size_no_inter(bandwidth,SLA)
        result = {"inc_num":[],"res_num":[],"mobile_num":[],"inc_th":[],"res_th":[],"mobile_th":[],"sys_th":[],"net":[],"inc_I":[],"res_I":[],"mobile_I":[]}
        for inc_num in range(1,12):
            for res_num in range(1,12):
                if inc_num+res_num>=12:
                    continue
                else:
                    mobile_num = 12-inc_num-res_num
                    sys_throughput = ins_size["inception"]["edge_throughput"]*inc_num\
                                     + ins_size["resnet"]["edge_throughput"]*res_num\
                                     + ins_size["mobilenet"]["edge_throughput"]*mobile_num
                    result["inc_num"].append(inc_num)
                    result["res_num"].append(res_num)
                    result["mobile_num"].append(mobile_num)
                    result["inc_th"].append(ins_size["inception"]["edge_throughput"]*inc_num)
                    result["res_th"].append(ins_size["resnet"]["edge_throughput"]*res_num)
                    result["mobile_th"].append(ins_size["mobilenet"]["edge_throughput"] * mobile_num)
                    result["sys_th"].append(sys_throughput)
                    result["net"].append(net)
                    result["inc_I"].append(ins_size["inception"]["intra"])
                    result["res_I"].append(ins_size["resnet"]["intra"])
                    result["mobile_I"].append(ins_size["mobilenet"]["intra"])
        result_pd = pd.DataFrame(data = result,index=range(len(result["net"])))
        #result_pd.to_excel("background/resource_allocation/resource_allocation.xlsx")
class ModelPartition:
    def __init__(self):
        self.model_list = ["inception","resnet","mobilenet"]
        self.wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx")
        self.model_wifi_trace = {"inception":"trace1","resnet":"trace2","mobilenet":"trace3"}
        self.opt = ModelOptimizor()
        self.layer_nums = {"inception": 20, "resnet": 21, "mobilenet": 16}
        self.trace_num = len(self.wifi_trace["trace1"].values)

    def get_partition_points_real_trace(self):
        I_trace = self.wifi_trace["trace1"]
        R_trace = self.wifi_trace["trace2"]
        M_trace = self.wifi_trace["trace3"]
        basic_unit = {"I_net":[],"R_net":[],"M_net":[],"R_k":[],"R_intra":[], "R_E":[],"M_k":[],"M_intra":[],"M_E":[],"I_k":[],"I_intra":[], "I_E":[]}
        result_hitdl = {"I_net":[],"R_net":[],"M_net":[],"R_k":[],"R_intra":[],
                  "R_E":[],"M_k":[],"M_intra":[],"M_E":[],"I_k":[],"I_intra":[],
                  "I_E":[],"I_e2e_no_queue":[],"R_e2e_no_queue":[],"M_e2e_no_queue":[],"I_num_per_ins":[],"R_num_per_ins":[],"M_num_per_ins":[]}
        result_baseline = {"I_net":[],"R_net":[],"M_net":[],"R_k":[],"R_intra":[],
                  "R_E":[],"M_k":[],"M_intra":[],"M_E":[],"I_k":[],"I_intra":[],
                  "I_E":[],"I_e2e_no_queue":[],"R_e2e_no_queue":[],"M_e2e_no_queue":[],"I_num_per_ins":[],"R_num_per_ins":[],"M_num_per_ins":[]}
        result_neuro = {"I_net":[],"R_net":[],"M_net":[],"R_k":[],"R_intra":[],
                  "R_E":[],"M_k":[],"M_intra":[],"M_E":[],"I_k":[],"I_intra":[],
                  "I_E":[],"I_e2e_no_queue":[],"R_e2e_no_queue":[],"M_e2e_no_queue":[],"I_num_per_ins":[],"R_num_per_ins":[],"M_num_per_ins":[]}
        for i in range(self.trace_num):
            inception_network = I_trace[i] * 1024 * 1024 / 8
            resnet_network = R_trace[i] * 1024 * 1024 / 8
            mobilenet_network = M_trace[i] * 1024 * 1024 / 8

            bandwidth = {"inception": inception_network, "resnet": resnet_network,
                         "mobilenet": mobilenet_network}

            baseline = self.opt.get_baseline_ins_size_queue(bandwidth,max_intra=4)
            neuro = self.opt.get_neurosurgeon_ins_size_queue(bandwidth,max_intra=4)
            hitdl = self.opt.get_ins_size_queue(bandwidth,max_intra=4)


            result_hitdl["I_net"].append(I_trace[i])
            result_hitdl["R_net"].append(R_trace[i])
            result_hitdl["M_net"].append(M_trace[i])

            result_baseline["I_net"].append(I_trace[i])
            result_baseline["R_net"].append(R_trace[i])
            result_baseline["M_net"].append(M_trace[i])

            result_neuro["I_net"].append(I_trace[i])
            result_neuro["R_net"].append(R_trace[i])
            result_neuro["M_net"].append(M_trace[i])

            for model_name in self.model_list:
                hitdl_plan = hitdl[model_name]
                result_hitdl[model_name[0].capitalize()+"_k"].append(hitdl_plan["k"])
                result_hitdl[model_name[0].capitalize() + "_intra"].append(hitdl_plan["intra"])
                result_hitdl[model_name[0].capitalize() + "_E"].append(hitdl_plan["efficiency"])
                result_hitdl[model_name[0].capitalize() + "_e2e_no_queue"].append(hitdl_plan["e2e_no_queue"])
                result_hitdl[model_name[0].capitalize() + "_num_per_ins"].append(hitdl_plan["user_num_per_ins"])

                baseline_plan = baseline[model_name]
                result_baseline[model_name[0].capitalize()+"_k"].append(baseline_plan["k"])
                result_baseline[model_name[0].capitalize() + "_intra"].append(baseline_plan["intra"])
                result_baseline[model_name[0].capitalize() + "_E"].append(baseline_plan["efficiency"])
                result_baseline[model_name[0].capitalize() + "_e2e_no_queue"].append(baseline_plan["e2e_no_queue"])
                result_baseline[model_name[0].capitalize() + "_num_per_ins"].append(baseline_plan["user_num_per_ins"])

                neuro_plan = neuro[model_name]
                result_neuro[model_name[0].capitalize()+"_k"].append(neuro_plan["k"])
                result_neuro[model_name[0].capitalize() + "_intra"].append(neuro_plan["intra"])
                result_neuro[model_name[0].capitalize() + "_E"].append(neuro_plan["efficiency"])
                result_neuro[model_name[0].capitalize() + "_e2e_no_queue"].append(neuro_plan["e2e_no_queue"])
                result_neuro[model_name[0].capitalize() + "_num_per_ins"].append(neuro_plan["user_num_per_ins"])

        file_path = "experiment/model_partition/interference"
        hitdl_pd = pd.DataFrame(data = result_hitdl,index = range(len(result_hitdl["I_net"])))
        hitdl_pd.to_excel(file_path+"/hitdl_partition.xlsx")

        neuro_pd = pd.DataFrame(data = result_neuro,index = range(len(result_neuro["I_net"])))
        neuro_pd.to_excel(file_path+"/neuro_partition.xlsx")

        baseline_pd = pd.DataFrame(data = result_baseline,index = range(len(result_baseline["I_net"])))
        baseline_pd.to_excel(file_path+"/baseline_partition.xlsx")

    def get_partition_points_SLA(self):

        wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx", sheet_name="model_partition")
        I_trace = wifi_trace["trace1"].values
        R_trace = wifi_trace["trace2"].values
        M_trace = wifi_trace["trace3"].values
        basic_unit = {"I_net":[],"R_net":[],"M_net":[],"R_k":[],"R_intra":[], "R_E":[],"M_k":[],"M_intra":[],"M_E":[],"I_k":[],"I_intra":[], "I_E":[]}
        result_hitdl = {"I_net":[],"R_net":[],"M_net":[],"R_k":[],"R_intra":[],
                  "R_E":[],"M_k":[],"M_intra":[],"M_E":[],"I_k":[],"I_intra":[],
                  "I_E":[],"I_e2e_no_queue":[],"R_e2e_no_queue":[],"M_e2e_no_queue":[],"I_num_per_ins":[],"R_num_per_ins":[],"M_num_per_ins":[]}
        result_baseline = {"I_net":[],"R_net":[],"M_net":[],"R_k":[],"R_intra":[],
                  "R_E":[],"M_k":[],"M_intra":[],"M_E":[],"I_k":[],"I_intra":[],
                  "I_E":[],"I_e2e_no_queue":[],"R_e2e_no_queue":[],"M_e2e_no_queue":[],"I_num_per_ins":[],"R_num_per_ins":[],"M_num_per_ins":[]}
        result_neuro = {"I_net":[],"R_net":[],"M_net":[],"R_k":[],"R_intra":[],
                  "R_E":[],"M_k":[],"M_intra":[],"M_E":[],"I_k":[],"I_intra":[],
                  "I_E":[],"I_e2e_no_queue":[],"R_e2e_no_queue":[],"M_e2e_no_queue":[],"I_num_per_ins":[],"R_num_per_ins":[],"M_num_per_ins":[]}
        SLA_range = np.arange(0.7, 1, 0.01)
        for i in [0]:#range(len(I_trace)):
            inception_network = I_trace[i] * 1024 * 1024 / 8
            resnet_network = R_trace[i] * 1024 * 1024 / 8
            mobilenet_network = M_trace[i] * 1024 * 1024 / 8

            bandwidth = {"inception": inception_network, "resnet": resnet_network,
                         "mobilenet": mobilenet_network}
            for SLA_factor_value in SLA_range:
                baseline = self.opt.get_baseline_ins_size_queue(bandwidth,max_intra=4,SLA_factor = SLA_factor_value)
                neuro = self.opt.get_neurosurgeon_ins_size_queue(bandwidth,max_intra=4,SLA_factor = SLA_factor_value)
                hitdl = self.opt.get_ins_size_queue(bandwidth,max_intra=4,SLA_factor = SLA_factor_value)


                result_hitdl["I_net"].append(I_trace[i])
                result_hitdl["R_net"].append(R_trace[i])
                result_hitdl["M_net"].append(M_trace[i])

                result_baseline["I_net"].append(I_trace[i])
                result_baseline["R_net"].append(R_trace[i])
                result_baseline["M_net"].append(M_trace[i])

                result_neuro["I_net"].append(I_trace[i])
                result_neuro["R_net"].append(R_trace[i])
                result_neuro["M_net"].append(M_trace[i])

                for model_name in self.model_list:
                    hitdl_plan = hitdl[model_name]
                    result_hitdl[model_name[0].capitalize()+"_k"].append(hitdl_plan["k"])
                    result_hitdl[model_name[0].capitalize() + "_intra"].append(hitdl_plan["intra"])
                    result_hitdl[model_name[0].capitalize() + "_E"].append(hitdl_plan["efficiency"])
                    result_hitdl[model_name[0].capitalize() + "_e2e_no_queue"].append(hitdl_plan["e2e_no_queue"])
                    result_hitdl[model_name[0].capitalize() + "_num_per_ins"].append(hitdl_plan["user_num_per_ins"])

                    baseline_plan = baseline[model_name]
                    result_baseline[model_name[0].capitalize()+"_k"].append(baseline_plan["k"])
                    result_baseline[model_name[0].capitalize() + "_intra"].append(baseline_plan["intra"])
                    result_baseline[model_name[0].capitalize() + "_E"].append(baseline_plan["efficiency"])
                    result_baseline[model_name[0].capitalize() + "_e2e_no_queue"].append(baseline_plan["e2e_no_queue"])
                    result_baseline[model_name[0].capitalize() + "_num_per_ins"].append(baseline_plan["user_num_per_ins"])

                    neuro_plan = neuro[model_name]
                    result_neuro[model_name[0].capitalize()+"_k"].append(neuro_plan["k"])
                    result_neuro[model_name[0].capitalize() + "_intra"].append(neuro_plan["intra"])
                    result_neuro[model_name[0].capitalize() + "_E"].append(neuro_plan["efficiency"])
                    result_neuro[model_name[0].capitalize() + "_e2e_no_queue"].append(neuro_plan["e2e_no_queue"])
                    result_neuro[model_name[0].capitalize() + "_num_per_ins"].append(neuro_plan["user_num_per_ins"])
        ''''''
        file_path = "experiment/model_partition/SLA/max_intra=4"
        hitdl_pd = pd.DataFrame(data = result_hitdl,index = SLA_range) #range(len(result_hitdl["I_net"]))
        hitdl_pd.to_excel(file_path+"/hitdl_partition.xlsx")

        neuro_pd = pd.DataFrame(data = result_neuro,index = SLA_range)#range(len(result_neuro["I_net"]))
        neuro_pd.to_excel(file_path+"/neuro_partition.xlsx")

        baseline_pd = pd.DataFrame(data = result_baseline,index = SLA_range) #range(len(result_baseline["I_net"]))
        baseline_pd.to_excel(file_path+"/baseline_partition.xlsx")

    def get_partition_points_simulate_trace(self):
        wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx", sheet_name="model_partition")
        I_trace = wifi_trace["trace1"].values
        R_trace = wifi_trace["trace1"].values
        M_trace = wifi_trace["trace1"].values
        basic_unit = {"I_net":[],"R_net":[],"M_net":[],"R_k":[],"R_intra":[], "R_E":[],"M_k":[],"M_intra":[],"M_E":[],"I_k":[],"I_intra":[], "I_E":[]}
        result_hitdl = {"I_net":[],"R_net":[],"M_net":[],"R_k":[],"R_intra":[],
                  "R_E":[],"M_k":[],"M_intra":[],"M_E":[],"I_k":[],"I_intra":[],
                  "I_E":[],"I_e2e_no_queue":[],"R_e2e_no_queue":[],"M_e2e_no_queue":[],"I_num_per_ins":[],"R_num_per_ins":[],"M_num_per_ins":[]}
        result_baseline = {"I_net":[],"R_net":[],"M_net":[],"R_k":[],"R_intra":[],
                  "R_E":[],"M_k":[],"M_intra":[],"M_E":[],"I_k":[],"I_intra":[],
                  "I_E":[],"I_e2e_no_queue":[],"R_e2e_no_queue":[],"M_e2e_no_queue":[],"I_num_per_ins":[],"R_num_per_ins":[],"M_num_per_ins":[]}
        result_neuro = {"I_net":[],"R_net":[],"M_net":[],"R_k":[],"R_intra":[],
                  "R_E":[],"M_k":[],"M_intra":[],"M_E":[],"I_k":[],"I_intra":[],
                  "I_E":[],"I_e2e_no_queue":[],"R_e2e_no_queue":[],"M_e2e_no_queue":[],"I_num_per_ins":[],"R_num_per_ins":[],"M_num_per_ins":[]}
        for i in range(len(I_trace)):
            inception_network = I_trace[i] * 1024 * 1024 / 8
            resnet_network = R_trace[i] * 1024 * 1024 / 8
            mobilenet_network = M_trace[i] * 1024 * 1024 / 8

            bandwidth = {"inception": inception_network, "resnet": resnet_network,
                         "mobilenet": mobilenet_network}

            baseline = self.opt.get_baseline_ins_size_queue(bandwidth,max_intra=4)
            neuro = self.opt.get_neurosurgeon_ins_size_queue(bandwidth,max_intra=4)
            hitdl = self.opt.get_ins_size_queue(bandwidth,max_intra=4)


            result_hitdl["I_net"].append(I_trace[i])
            result_hitdl["R_net"].append(R_trace[i])
            result_hitdl["M_net"].append(M_trace[i])

            result_baseline["I_net"].append(I_trace[i])
            result_baseline["R_net"].append(R_trace[i])
            result_baseline["M_net"].append(M_trace[i])

            result_neuro["I_net"].append(I_trace[i])
            result_neuro["R_net"].append(R_trace[i])
            result_neuro["M_net"].append(M_trace[i])

            for model_name in self.model_list:
                hitdl_plan = hitdl[model_name]
                result_hitdl[model_name[0].capitalize()+"_k"].append(hitdl_plan["k"])
                result_hitdl[model_name[0].capitalize() + "_intra"].append(hitdl_plan["intra"])
                result_hitdl[model_name[0].capitalize() + "_E"].append(hitdl_plan["efficiency"])
                result_hitdl[model_name[0].capitalize() + "_e2e_no_queue"].append(hitdl_plan["e2e_no_queue"])
                result_hitdl[model_name[0].capitalize() + "_num_per_ins"].append(hitdl_plan["user_num_per_ins"])

                baseline_plan = baseline[model_name]
                result_baseline[model_name[0].capitalize()+"_k"].append(baseline_plan["k"])
                result_baseline[model_name[0].capitalize() + "_intra"].append(baseline_plan["intra"])
                result_baseline[model_name[0].capitalize() + "_E"].append(baseline_plan["efficiency"])
                result_baseline[model_name[0].capitalize() + "_e2e_no_queue"].append(baseline_plan["e2e_no_queue"])
                result_baseline[model_name[0].capitalize() + "_num_per_ins"].append(baseline_plan["user_num_per_ins"])

                neuro_plan = neuro[model_name]
                result_neuro[model_name[0].capitalize()+"_k"].append(neuro_plan["k"])
                result_neuro[model_name[0].capitalize() + "_intra"].append(neuro_plan["intra"])
                result_neuro[model_name[0].capitalize() + "_E"].append(neuro_plan["efficiency"])
                result_neuro[model_name[0].capitalize() + "_e2e_no_queue"].append(neuro_plan["e2e_no_queue"])
                result_neuro[model_name[0].capitalize() + "_num_per_ins"].append(neuro_plan["user_num_per_ins"])

        file_path = "experiment/model_partition/interference/simulate_trace/max_intra=4"
        hitdl_pd = pd.DataFrame(data = result_hitdl,index = range(len(result_hitdl["I_net"])))
        hitdl_pd.to_excel(file_path+"/hitdl_partition.xlsx")

        neuro_pd = pd.DataFrame(data = result_neuro,index = range(len(result_neuro["I_net"])))
        neuro_pd.to_excel(file_path+"/neuro_partition.xlsx")

        baseline_pd = pd.DataFrame(data = result_baseline,index = range(len(result_baseline["I_net"])))
        baseline_pd.to_excel(file_path+"/baseline_partition.xlsx")

    def get_e2e(self):
        wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx", sheet_name="model_partition")
        model_trace = {"inception":wifi_trace["trace1"].values,"resnet":wifi_trace["trace2"].values,"mobilenet":wifi_trace["trace3"].values}
        for model_name in self.model_list:
            feasible_writer = pd.ExcelWriter("experiment/model_partition/e2e_no_queue/"+model_name+"_feasible_plans.xlsx")
            result = {"net":[]}
            result_select = {"net":[],"layer":[],"e2e":[]}

            for i in range(len(model_trace[model_name])):
                Handwdith = model_trace[model_name][i] * 1024 * 1024 / 8
                plans, select_layer,min_e2e = self.opt.get_partition_e2e_under_fix_intra_no_queue(model_name, Handwdith)

                result_select["net"].append(model_trace[model_name][i])
                result_select["layer"].append(select_layer)
                result_select["e2e"].append(min_e2e)
                j = 0
                result["net"].append(model_trace[model_name][i])
                for e2e in plans:
                    if str(j) in list(result.keys()):
                        result[str(j)].append(e2e)
                    else:
                        result[str(j)] = [e2e]
                    j = j+1

            result_pd = pd.DataFrame(data=result, index=range(len(result["net"])))
            result_pd.to_excel(feasible_writer, sheet_name="feasible")

            result_pd = pd.DataFrame(data=result_select, index=range(len(result_select["net"])))
            result_pd.to_excel(feasible_writer, sheet_name="select")
            feasible_writer.save()
            feasible_writer.close()

class MinMaxGreedySearch:
    def __init__(self):
        self.TOTAL_CPU_Cores = 12
        self.model_name_list = ["inception","resnet","mobilenet"]
    def sort_model_by_effeciency_asc(self, x):
        """
        sorted_model[0]["efficiency"]>sorted_model[1]["efficiency"]>sorted_model[2]["efficiency"]
        :param ins_size:
        :return:
        """
        #print("aaaaaaaaa",x)
        sorted_model = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
        return list(sorted_model.keys())
    def sort_model_by_effeciency(self, origial_ins_size):
        """
        sorted_model[0]["efficiency"]>sorted_model[1]["efficiency"]>sorted_model[2]["efficiency"]
        :param ins_size:
        :return:
        """
        ins_size = copy.deepcopy(origial_ins_size)
        # print("+++++++++",ins_size)
        for model_name in self.model_name_list:
            if model_name not in ins_size.keys():
                ins_size[model_name] = {"efficiency": 0}

        sorted_model = [None, None, None]
        if ins_size["inception"]["efficiency"] > ins_size["resnet"]["efficiency"]:
            sorted_model[0] = "inception"
            sorted_model[1] = "resnet"
        else:
            sorted_model[0] = "resnet"
            sorted_model[1] = "inception"
        if ins_size["mobilenet"]["efficiency"] > ins_size[sorted_model[0]]["efficiency"]:
            sorted_model[2] = sorted_model[1]
            sorted_model[1] = sorted_model[0]
            sorted_model[0] = "mobilenet"
        elif ins_size["mobilenet"]["efficiency"] > ins_size[sorted_model[1]]["efficiency"]:
            sorted_model[2] = sorted_model[1]
            sorted_model[1] = "mobilenet"
        else:
            sorted_model[2] = "mobilenet"
        return sorted_model
    def check_feasible(self,feasible_ins_num,ins_num,F,ins_size,model_utility):
        def is_feasible(model_utility):
            result = True
            total_utility = np.sum(list(model_utility.values()))
            if round(total_utility,2)>0:
                for model_name in self.model_name_list:
                    utility_ratio = round(model_utility[model_name] / total_utility, 2)
                    #print(model_name,utility_ratio)
                    if utility_ratio > F:
                        result = False
                        break
            return result
        #print(feasible_ins_num)
        if feasible_ins_num != None:
            return feasible_ins_num
        else:

            #1. find the model that fails to meet F and put them into X, put those meeting F into Y
            while True:
                if is_feasible(model_utility):
                    feasible_ins_num = copy.deepcopy(ins_num)
                    break
                else:
                    total_utility = np.sum(list(model_utility.values()))
                    X = []
                    X_U = 0
                    Y = []
                    for model_name in self.model_name_list:
                        utility_ratio = round(model_utility[model_name]/total_utility,2)
                        #print("model_u", model_name, model_utility[model_name],utility_ratio)
                        if utility_ratio>F:
                            Y.append(model_name)
                        else:
                            X.append(model_name)
                            X_U = X_U+model_utility[model_name]
                            #
                    gap = round(total_utility - X_U/len(X)/F,2)
                    #print("X", X, "Y", Y, X_U,"gap",gap)
                    diff = []
                    candi_model_size = {}
                    for model_name in Y:
                        candi_model_size[model_name] = ins_size[model_name]["efficiency"]

                    sorted_model = self.sort_model_by_effeciency_asc(candi_model_size)
                    for model_name in sorted_model:
                        m = ins_size[model_name]["efficiency"]
                        temp = round(abs(gap - m),2)
                        diff.append(temp)
                    # 2. sort the diff and find the model, the ins num of which is not zero
                    if round(np.sum(np.array(diff)),2)!=0:
                        diff = np.around(np.array(diff)/np.sum(np.array(diff)),3)
                    model_index = np.around(np.arange(1,len(sorted_model)+1,1)/len(sorted_model),3)
                    overall = diff+model_index
                    for i in np.argsort(overall):
                        model_name = sorted_model[i]
                        if ins_num[model_name]["ins_num"]>0:
                            temp = copy.deepcopy(model_utility)
                            temp[model_name] = temp[model_name] - ins_size[model_name]["efficiency"] * ins_size[model_name]["intra"]
                            if is_feasible(temp):
                                model_utility[model_name] = model_utility[model_name] - ins_size[model_name]["efficiency"] * ins_size[model_name]["intra"]
                                ins_num[model_name]["ins_num"] = ins_num[model_name]["ins_num"]-1
                                print("selected model",model_name)
                                #break

                    print(ins_num)
                    print()
            return feasible_ins_num

    def get_greedy_ins_number_modified(self,ins_size,F):
        def meet_F(model_name):
            max_u = None
            total_utility = np.sum(list(model_utility.values()))
            result = True
            if ins_size[model_name]["intra"]<0 or ins_size[model_name]["intra"]>A_cores:
                #print("fail create",model_name,ins_size[model_name]["intra"],A_cores)
                result = False
            else:
                total_utility = np.sum(list(model_utility.values()))+model_intra[model_name]*ins_size[model_name]["efficiency"]
                for temp in self.model_name_list:
                    if temp == model_name:
                        model_U_ratio = (model_utility[temp] + model_intra[model_name]*ins_size[model_name]["efficiency"])/total_utility
                    else:
                        model_U_ratio = model_utility[temp] / total_utility
                    #print("&&&&", temp, model_U_ratio)
                    if max_u == None:
                        max_u = model_U_ratio
                    elif round(model_U_ratio,2) > max_u:
                        max_u = model_U_ratio

                    if round(model_U_ratio,2)>F:
                        result = False
                        #print("fail F")
            return result,max_u

        model_utility = {}
        model_CPU_cores = {}
        model_intra = {}
        ins_num = {}
        feasible_ins_num = None
        for model_name in ins_size.keys():
            model_utility[model_name] = 0
            model_CPU_cores[model_name] = 0
            model_intra[model_name] = ins_size[model_name]["intra"]
            ins_num[model_name] = {"ins_num":0}
            #feasible_ins_num[model_name] = {"ins_num": 0}
        sorted_model_name = self.sort_model_by_effeciency(ins_size)
        init = True
        A_cores = self.TOTAL_CPU_Cores
        while True:
            if A_cores < min(list(model_intra.values())):
                break
            else:
                flag = True
                select_model = None
                model_max_utlity = {"inception":0,"resnet":0,"mobilenet":0}
                for model_name in sorted_model_name:
                    test_model = meet_F(model_name)
                    model_max_utlity[model_name] = test_model[1]
                    if init or test_model[0]:
                        select_model = model_name
                        ins_num[select_model]["ins_num"] = ins_num[select_model]["ins_num"] + 1
                        flag = False
                        if not init:
                            feasible_ins_num=copy.deepcopy(ins_num)
                        else:
                            init = False
                        break
                if flag:
                    min_U_ratio = None
                    for model_name in sorted_model_name:
                        if model_max_utlity[model_name]!=None:
                            if min_U_ratio == None:
                                min_U_ratio = model_max_utlity[model_name]
                                select_model = model_name
                            elif model_max_utlity[model_name] < min_U_ratio:
                                min_U_ratio = model_max_utlity[model_name]
                                select_model = model_name
                    if select_model != None:
                        ins_num[select_model]["ins_num"] = ins_num[select_model]["ins_num"] + 1
                if select_model != None:
                    A_cores = A_cores - model_intra[select_model]
                    model_utility[select_model] = model_utility[select_model] + model_intra[select_model] * ins_size[select_model]["efficiency"]
                else:
                    break
        print(feasible_ins_num,ins_num)
        feasible_ins_num = self.check_feasible(feasible_ins_num,ins_num,F,ins_size,model_utility)
        return feasible_ins_num
    def get_greedy_ins_number(self,ins_size,F):
        def meet_F(model_name):

            #print("++==")
            total_utility = np.sum(list(model_utility.values()))
            result = True
            if ins_size[model_name]["intra"]<0 or ins_size[model_name]["intra"]>A_cores:
                #print("fail create",model_name,ins_size[model_name]["intra"],A_cores)
                result = False
            else:
                total_utility = np.sum(list(model_utility.values()))+model_intra[model_name]*ins_size[model_name]["efficiency"]
                for temp in self.model_name_list:
                    if temp == model_name:
                        model_U_ratio = (model_utility[temp] + model_intra[model_name]*ins_size[model_name]["efficiency"])/total_utility
                    else:
                        model_U_ratio = model_utility[temp] / total_utility
                    if round(model_U_ratio,2)>F:
                        result = False
                        #print("fail F")
                        break
            return result
        def check_U_ratio(model_name):
            max_u = None
            #print("test=======",model_name,ins_size[model_name])
            if ins_size[model_name]["intra"]<0 or ins_size[model_name]["intra"]>A_cores:
               #print("fail to put in")
               pass
            else:
                total_utility = np.sum(list(model_utility.values()))+model_intra[model_name]*ins_size[model_name]["efficiency"]
                for temp in self.model_name_list:
                    if temp == model_name:
                        model_U_ratio = (model_utility[model_name] + model_intra[model_name]*ins_size[model_name]["efficiency"])/total_utility
                    else:
                        model_U_ratio = model_utility[temp] / total_utility
                    if max_u == None:
                        max_u = model_U_ratio
                    elif round(model_U_ratio,2) > max_u:
                        max_u = model_U_ratio
                    #print("check u",temp,model_U_ratio)
                    #print()
            #print(max_u)
            #print()
            return max_u
        model_utility = {}
        model_CPU_cores = {}
        model_intra = {}
        ins_num = {}
        feasible_ins_num = {}
        for model_name in ins_size.keys():
            model_utility[model_name] = 0
            model_CPU_cores[model_name] = 0
            model_intra[model_name] = ins_size[model_name]["intra"]
            ins_num[model_name] = {"ins_num":0}
            feasible_ins_num[model_name] = {"ins_num": 0}
        sorted_model_name = self.sort_model_by_effeciency(ins_size)
        #print("=========ins_size========", model_intra)
        init = True
        A_cores = self.TOTAL_CPU_Cores
        while True:
            if A_cores < min(list(model_intra.values())):
                break
            else:
                flag = True
                select_model = None
                for model_name in sorted_model_name:
                    if init or meet_F(model_name):
                        select_model = model_name
                        ins_num[select_model]["ins_num"] = ins_num[select_model]["ins_num"] + 1
                        flag = False
                        init = False
                        feasible_ins_num=copy.deepcopy(ins_num)
                        break
                if flag:
                    min_U_ratio = None
                    for model_name in sorted_model_name:
                        model_U = check_U_ratio(model_name)
                        if model_U != None:
                            if min_U_ratio == None:
                                min_U_ratio = model_U
                                select_model = model_name
                            elif model_U < min_U_ratio:
                                min_U_ratio = model_U
                                select_model = model_name
                    if select_model != None:
                        ins_num[select_model]["ins_num"] = ins_num[select_model]["ins_num"] + 1
                if select_model != None:
                    A_cores = A_cores - model_intra[select_model]
                    model_utility[select_model] = model_utility[select_model] + model_intra[select_model] * ins_size[select_model]["efficiency"]
                else:
                    break
        return feasible_ins_num

class GreedySearch:
    def __init__(self):
        self.TOTAL_CPU_Cores = 12
        self.model_name_list = ["inception","resnet","mobilenet"]
    def sort_model_by_effeciency(self,origial_ins_size):
        """
        sorted_model[0]["efficiency"]>sorted_model[1]["efficiency"]>sorted_model[2]["efficiency"]
        :param ins_size:
        :return:
        """
        ins_size = copy.deepcopy(origial_ins_size)
        #print("+++++++++",ins_size)
        for model_name in self.model_name_list:
            if model_name not in ins_size.keys():
                ins_size[model_name] = {"efficiency":0}

        sorted_model = [None, None, None]
        if ins_size["inception"]["efficiency"] > ins_size["resnet"]["efficiency"]:
            sorted_model[0] = "inception"
            sorted_model[1] = "resnet"
        else:
            sorted_model[0] = "resnet"
            sorted_model[1] = "inception"
        if ins_size["mobilenet"]["efficiency"] > ins_size[sorted_model[0]]["efficiency"]:
            sorted_model[2] = sorted_model[1]
            sorted_model[1] = sorted_model[0]
            sorted_model[0] = "mobilenet"
        elif ins_size["mobilenet"]["efficiency"] > ins_size[sorted_model[1]]["efficiency"]:
            sorted_model[2] = sorted_model[1]
            sorted_model[1] = "mobilenet"
        else:
            sorted_model[2] = "mobilenet"
        return sorted_model
    def meet_fairness(self, strategy, fairness):
        # 1. compute each models' utility ratio

        total_utility = 0
        #CPU_used = 0
        for model_name in ["inception", "resnet", "mobilenet"]:
            total_utility = total_utility + strategy[model_name]["ins_num"] * strategy[model_name]["efficiency"] * \
                            strategy[model_name]["intra"]
            #CPU_used = CPU_used + strategy[model_name]["ins_num"] * strategy[model_name]["intra"]
        result = True
        ratios = {}
        # print(model_name, ratio,total_utility)
        ratio = None
        for model_name in ["inception", "resnet", "mobilenet"]:
            # print("hehrehre",model_name)
            if total_utility != 0:
                ratio = strategy[model_name]["ins_num"] * strategy[model_name]["efficiency"] * strategy[model_name]["intra"] / total_utility
            else:
                ratio = 0
            ratios[model_name] = round(ratio,2)
            # print(round(ratio,4),round(fairness,4))
            if round(ratio, 2) > round(fairness, 2):
                result = False
        return result, ratios

    def make_least_E_meet_F(self,strategy,model_name,model_intra,sorted_model,fairness):
        """
        调整E最小的模型，以使策略满足F
        :param strategy:
        :return:
        """
        find_feasible = False
        new_violate_model = None
        while strategy[model_name]["ins_num"] > 0:
            strategy[model_name]["ins_num"] = strategy[model_name]["ins_num"] - 1
            used_cores = strategy[model_name]["ins_num"] * model_intra
            for i in [0, 1]:
                used_cores = used_cores + strategy[sorted_model[i]]["intra"] * strategy[sorted_model[i]]["ins_num"]
            result = self.meet_fairness(strategy,fairness)
            if result[0]:
                find_feasible = True
                break
            else:
                #
                new_select_model, new_select_model_index = self.find_fail_models(result[1], fairness,sorted_model)
                if new_select_model_index!=2:
                    break
                    #pass

        return find_feasible

    def find_fail_models(self,ratios, fairness,sorted_model):
        select_model = None
        max_ratio = 0
        for model_name in self.model_name_list:
            ratio = round(ratios[model_name], 3)
            if ratio > fairness:
                if ratio > max_ratio:
                    max_ratio = ratio
                    select_model = model_name
        select_model_index = sorted_model.index(select_model)
        return select_model, select_model_index

    def get_greedy_ins_number_deprecated(self, ins_size, fairness):
        def greedy_Hased_allocation(availaHle_CPU_cores, ins_size):
            ins_num = {}
            sorted_model = self.sort_model_by_effeciency(ins_size)
            for model_name in sorted_model:
                if model_name in list(ins_size.keys()):
                    # depend on the greedy-strategy to allocate CPU cores to the model as much as possiHle
                    if availaHle_CPU_cores > 0:
                        model_ins_num = math.floor(availaHle_CPU_cores / ins_size[model_name]["intra"])
                        ins_num[model_name] = {"ins_num": model_ins_num}
                        availaHle_CPU_cores = availaHle_CPU_cores - model_ins_num * ins_size[model_name]["intra"]
                    else:
                        ins_num[model_name] = {"ins_num": 0}
            return ins_num

        """
        Hased on the greedy-strategy to determine the numHer of instances for each model.
        Meanwhile, note that the percent of some model's utility must He smaller than the faireness factor
        :param ins_size:
        :return:
        """
        result = {}
        # Step 1: find out the model that has the highest efficiency
        sorted_model = self.sort_model_by_effeciency(ins_size)
        availaHle_CPU_cores = self.TOTAL_CPU_Cores
        strategy = {"inception": {}, "resnet": {}, "mobilenet": {}}
        for model_name in self.model_name_list:
            strategy[model_name].update(ins_size[model_name])

        # while True:
        ins_num = greedy_Hased_allocation(availaHle_CPU_cores, ins_size)
        # 1. check whether the allocation meets the fairness
        #print("初始分配",ins_num,ins_size)
        for model_name in self.model_name_list:
            strategy[model_name].update(ins_num[model_name])
        while strategy[sorted_model[0]]["ins_num"] >= 0:
            result = self.meet_fairness(strategy, fairness)
            '''
            for model_name in self.model_name_list:
                print(model_name,strategy[model_name]["ins_num"])
            #print("当前的策略"strategy)
            print(result[1])
            print()
            time.sleep(0.5)
            '''
            if result[0]:
                break
            else:  # fail to meet fairness
                select_model,select_model_index = self.find_fail_models(result[1],fairness,sorted_model)
                find_feasible = False
                model_Hefore = select_model_index-1
                if select_model_index == 2:
                    # strategy,model_name,model_intra,sorted_model,fairness
                    find_feasible= self.make_least_E_meet_F(strategy,select_model,ins_size[select_model]["intra"],sorted_model,fairness)
                elif select_model_index == 1:
                    while strategy[select_model]["ins_num"] >0:
                        strategy[select_model]["ins_num"] = strategy[select_model]["ins_num"] - 1
                        used_cores = strategy[select_model]["ins_num"]*ins_size[select_model]["intra"]+strategy[sorted_model[0]]["intra"] * strategy[sorted_model[0]]["ins_num"]
                        other_availaHle_CPU_cores = availaHle_CPU_cores - used_cores
                        other_model_name = sorted_model[2]
                        other_ins_num = greedy_Hased_allocation(other_availaHle_CPU_cores,{other_model_name: ins_size[other_model_name]})
                        strategy[sorted_model[2]]["ins_num"] = other_ins_num[other_model_name]["ins_num"]
                        result = self.meet_fairness(strategy, fairness)
                        #print("小E的初始结果", other_ins_num,result[1])
                        if result[0]:
                            find_feasible = True
                            break
                        else:  #
                            result = self.make_least_E_meet_F(strategy,other_model_name,ins_size[other_model_name]["intra"],sorted_model,fairness)
                            if result:
                                find_feasible = True
                                break
                            else:
                                pass
                if find_feasible:
                    break
                else:
                    if model_Hefore<0:
                        model_Hefore = 0

                    strategy[sorted_model[model_Hefore]]["ins_num"] = strategy[sorted_model[model_Hefore]]["ins_num"] - 1

                    used_cores = strategy[sorted_model[model_Hefore]]["intra"] * strategy[sorted_model[model_Hefore]]["ins_num"]
                    other_availaHle_CPU_cores = availaHle_CPU_cores - used_cores
                    other_ins_size = {}
                    flag = False
                    if ins_size[sorted_model[1]]["intra"]>0:
                        other_ins_size[sorted_model[1]]=ins_size[sorted_model[1]]
                        flag = True
                    if ins_size[sorted_model[2]]["intra"]>0:
                        other_ins_size[sorted_model[2]]=ins_size[sorted_model[2]]
                        flag = True
                    if flag:
                        other_ins_num = greedy_Hased_allocation(other_availaHle_CPU_cores, other_ins_size)
                        for model_name in other_ins_size.keys():
                            strategy[model_name]["ins_num"] = other_ins_num[model_name]["ins_num"]
        ins_num = {}
        for model_name in self.model_name_list:
            ins_num[model_name] = {"ins_num":strategy[model_name]["ins_num"]}
        #print("greedy",ins_num)
        return ins_num

    def get_greedy_ins_number_advanced_deprecated(self,ins_size,F):
        """
        非最后一轮加入的ins无需保证F，最后一轮加入的ins必须保证F，最大限度的降低整个策略违例的可能
        :param ins_size:
        :param F:
        :return:
        """
        def get_model_meet_F(sorted_model_name,model_utility):
            select_model = []
            model_meet_F = []
            total_utility = np.sum(list(model_utility.values()))
            # get the models that violate F
            available_cores = self.TOTAL_CPU_Cores-np.sum(list(model_CPU_cores.values()))
            if np.around(total_utility,3)>0:
                for model_name in sorted_model_name:
                    #print(model_name,round(model_utility[model_name]/total_utility,2))
                    if round(model_utility[model_name]/total_utility,2)>F or ins_size[model_name]["intra"]<0 or ins_size[model_name]["intra"]>available_cores:
                        model_meet_F.append(0)
                    else:
                        model_meet_F.append(1)
                if len(np.where(np.array(model_meet_F)==1)[0])>0:
                    # find out the one which has the highest efficiency without violating F
                    #select_model = sorted_model_name[np.where(np.array(model_meet_F)==1)[0][0]]
                    #print("========+++++++++=====",np.where(np.array(model_meet_F)==1)[0])
                    for i in np.where(np.array(model_meet_F) == 1)[0]:
                        select_model.append(sorted_model_name[i])
                else:
                    # all the models fail to meet F
                    select_model = None
            else:
                select_model.append(sorted_model_name[0])
            return select_model
        model_utility = {}
        model_CPU_cores = {}
        model_intra = {}
        ins_num = {}
        for model_name in ins_size.keys():
            model_utility[model_name] = 0
            model_CPU_cores[model_name] = 0
            model_intra[model_name] = ins_size[model_name]["intra"]
            ins_num[model_name] = {"ins_num": 0}
        sorted_model_name = self.sort_model_by_effeciency(ins_size)
        # print(" 模型efficiency排序",sorted_model_name,ins_size)
        init = True
        while True:
            # 2.find the model with the highest efficiency and without violaiting fairness
            select_models = get_model_meet_F(sorted_model_name, model_utility)
            # print("===**候选的models***====",select_models)
            # time.sleep(0.5)
            if select_models == None:
                break
            else:
                # ("开始尝试分配",model_name,model_utility_ratio)
                for select_model in select_models:
                    model_utility_ratio = np.around((model_utility[select_model] + ins_size[select_model]["efficiency"] * ins_size[select_model]["intra"]) /
                                    (np.sum(list(model_utility.values())) + ins_size[select_model]["efficiency"] * ins_size[select_model]["intra"]), 2)
                    if self.TOTAL_CPU_Cores - np.sum(list(model_CPU_cores.values())) <= min(model_intra.values()): # 最后一轮分配
                        if init or model_utility_ratio<=F:
                            model_CPU_cores[select_model] = model_CPU_cores[select_model] + ins_size[select_model]["intra"]
                            model_utility[select_model] = model_utility[select_model] + ins_size[select_model]["efficiency"] * \
                                                          ins_size[select_model]["intra"]
                            ins_num[select_model]["ins_num"] = ins_num[select_model]["ins_num"] + 1
                            init = False
                            break
                    else:
                        model_CPU_cores[select_model] = model_CPU_cores[select_model] + ins_size[select_model]["intra"]
                        model_utility[select_model] = model_utility[select_model] + ins_size[select_model]["efficiency"]*ins_size[select_model]["intra"]
                        ins_num[select_model]["ins_num"] = ins_num[select_model]["ins_num"] + 1
                        init = False
                        break

                if self.TOTAL_CPU_Cores - np.sum(list(model_CPU_cores.values())) < min(model_intra.values()):
                    # there are no efficienct CPU cores to create any instance
                    # print("there are no efficienct CPU cores to create any instance")
                    break
                else:
                    #  there are available CPU cores to create some instance,
                    #   but the instance will violate F after being created when
                    violate_number = 0
                    # ("===========come here==========")
                    for model_name in ins_size.keys():
                        if self.TOTAL_CPU_Cores - np.sum(list(model_CPU_cores.values())) >= ins_size[model_name]["intra"]:
                            model_utility_ratio = np.around(
                                (model_utility[model_name] + ins_size[model_name]["efficiency"] * ins_size[model_name]["intra"]) /
                                (np.sum(list(model_utility.values())) + ins_size[model_name]["efficiency"] * ins_size[model_name][
                                    "intra"]), 2)
                            if model_utility_ratio > F:
                                # print(model_name, model_utility_ratio)
                                violate_number = violate_number + 1
                        else:
                            violate_number = violate_number + 1
                    if violate_number == 3:
                        # 模型只要创建就违例
                        break
        return ins_num

    def get_greedy_ins_number(self,ins_size,F):
        """
        每轮分配都保证F
        :param ins_size:
        :param F:
        :return:
        """
        def get_model_not_meet_F(sorted_model_name,model_utility):
            """
            选择能让系统最大效用比例最小化的模型放入
            （系统最大效用比例：所有模型效用比例的最大值）
            :param sorted_model_name:
            :param model_utility:
            :return:
            """
            select_model = []
            feasible_model = []
            total_utility = np.sum(list(model_utility.values()))
            available_cores = self.TOTAL_CPU_Cores-np.sum(list(model_CPU_cores.values()))
            if np.around(total_utility,3)>0:
                for model_name in sorted_model_name:
                    #print(model_name,round(model_utility[model_name]/total_utility,2))
                    if ins_size[model_name]["intra"]<0 or ins_size[model_name]["intra"]>available_cores:
                        feasible_model.append(0)
                    else:
                        feasible_model.append(1)
                if len(np.where(np.array(feasible_model)==1)[0])>0:
                    for i in np.where(np.array(feasible_model) == 1)[0]:

                        select_model.append(sorted_model_name[i])
                else:
                    # all the models fail to meet F
                    select_model = None
            else:
                select_model.append(sorted_model_name[0])
            #print("==========select model======",select_model)
            return select_model

        def get_model_meet_F(sorted_model_name,model_utility):
            select_model = []
            model_meet_F = []
            total_utility = np.sum(list(model_utility.values()))
            # get the models that violate F
            available_cores = self.TOTAL_CPU_Cores-np.sum(list(model_CPU_cores.values()))
            if np.around(total_utility,3)>0:
                for model_name in sorted_model_name:
                    #print(model_name,round(model_utility[model_name]/total_utility,2))
                    if round(model_utility[model_name]/total_utility,2)>F or ins_size[model_name]["intra"]<0 or ins_size[model_name]["intra"]>available_cores:
                        model_meet_F.append(0)
                    else:
                        model_meet_F.append(1)
                if len(np.where(np.array(model_meet_F)==1)[0])>0:
                    # find out the one which has the highest efficiency without violating F
                    #select_model = sorted_model_name[np.where(np.array(model_meet_F)==1)[0][0]]
                    #print("========+++++++++=====",np.where(np.array(model_meet_F)==1)[0])
                    for i in np.where(np.array(model_meet_F) == 1)[0]:
                        select_model.append(sorted_model_name[i])
                else:
                    # all the models fail to meet F
                    select_model = None
            else:
                select_model.append(sorted_model_name[0])
            #print("==========select model======",select_model)
            return select_model
        """
        
        :param ins_size: 
        :param F: fairness
        :return:         """
        # 1. define model utility and model resource
        model_utility = {}
        model_CPU_cores = {}
        model_intra = {}
        ins_num = {}
        for model_name in ins_size.keys():
            model_utility[model_name] = 0
            model_CPU_cores[model_name] = 0
            model_intra[model_name] = ins_size[model_name]["intra"]
            ins_num[model_name] = {"ins_num":0}
        sorted_model_name = self.sort_model_by_effeciency(ins_size)
        #print(" 模型efficiency排序",sorted_model_name,ins_size)
        init = True
        while True:
            # 2.find the model with the highest efficiency and without violaiting fairness
            select_models = get_model_meet_F(sorted_model_name,model_utility)
            # print("===**候选的models***====",select_models)
            # time.sleep(0.5)
            flag = True
            if select_models == None:
                """
                adopt min-max 
                """
                break
            else:
                for select_model in select_models: # 结束一轮的分配
                    model_utility_ratio = np.around((model_utility[select_model] + ins_size[select_model]["efficiency"] * ins_size[select_model]["intra"]) /
                                    (np.sum(list(model_utility.values())) + ins_size[select_model]["efficiency"] * ins_size[select_model]["intra"]), 2)
                    if ins_size[select_model]["intra"]+np.sum(list(model_CPU_cores.values()))<=self.TOTAL_CPU_Cores:
                        if init or model_utility_ratio<=F:
                            model_CPU_cores[select_model] = model_CPU_cores[select_model]+ins_size[select_model]["intra"]
                            model_utility[select_model] = model_utility[select_model]+ins_size[select_model]["efficiency"]*ins_size[select_model]["intra"]
                            ins_num[select_model]["ins_num"] = ins_num[select_model]["ins_num"]+1
                            init = False
                            #print("选中的models",select_model)
                            break
            # there are no efficiency cores to allocate to the selected model.
            if self.TOTAL_CPU_Cores-np.sum(list(model_CPU_cores.values())) < min(model_intra.values()) or self.TOTAL_CPU_Cores-np.sum(list(model_CPU_cores.values()))<=0:
                # there are no efficienct CPU cores to create any instance
                #print("there are no efficienct CPU cores to create any instance")
                break
            else:
                #  there are available CPU cores to create some instance,
                #   but the instance will violate F after being created when
                violate_number = 0
                #("===========come here==========")
                for model_name in ins_size.keys():
                    if self.TOTAL_CPU_Cores-np.sum(list(model_CPU_cores.values()))>=ins_size[model_name]["intra"]:
                        model_utility_ratio = np.around((model_utility[model_name] + ins_size[model_name]["efficiency"] * ins_size[model_name]["intra"]) /
                            (np.sum(list(model_utility.values())) + ins_size[model_name]["efficiency"] * ins_size[model_name]["intra"]), 2)
                        if model_utility_ratio>F:
                            #print(model_name, model_utility_ratio)
                            violate_number = violate_number+1
                    else:
                        violate_number = violate_number + 1
                if violate_number == 3:
                    # 模型只要创建就违例
                    break
            #print("ins_num", ins_num, "used_cores", np.sum(list(model_CPU_cores.values())))
            #print()
        return ins_num

    def greedy_based_allocation_deprecated(self,availaHle_CPU_cores, ins_size):
        ins_num = {}
        sorted_model = self.sort_model_by_effeciency(ins_size)
        for model_name in sorted_model:
            if model_name not in ins_size.keys():
                continue
            else:
                # depend on the greedy-strategy to allocate CPU cores to the model as much as possiHle
                if availaHle_CPU_cores > 0:
                    model_ins_num = math.floor(availaHle_CPU_cores / ins_size[model_name]["intra"])
                    ins_num[model_name] = {"ins_num": model_ins_num}
                    availaHle_CPU_cores = availaHle_CPU_cores - model_ins_num * ins_size[model_name]["intra"]
                else:
                    ins_num[model_name] = {"ins_num": 0}
        return ins_num

class MCKPAllocation:
    def __init__(self,CPU_Cores,model_name_list,F):
        self.CPU_Cores = CPU_Cores
        self.model_name_list = model_name_list
    def cpu_const(self,ins_size):
        total_plans = 0
        for model_name in ins_size.keys():
            total_plans = total_plans + len(ins_size["inception"]["intra"])+len(ins_size["resnet"]["intra"])+len(ins_size["mobilenet"]["intra"])
        overall_cons = np.zeros(total_plans)
        overall_E = np.zeros(total_plans)
        model_cons = [np.zeros(total_plans),np.zeros(total_plans),np.zeros(total_plans)]#{"inception":np.zeros(total_plans),"resnet":np.zeros(total_plans),"mobilenet":np.zeros(total_plans)}
        cons_start = {"inception":0,"resnet":len(ins_size["inception"]["intra"]),"mobilenet":len(ins_size["resnet"]["intra"])+len(ins_size["inception"]["intra"])}
        i = 0
        ins_num_upper = np.zeros(total_plans)
        ins_num_lower = [np.zeros(total_plans),np.zeros(total_plans),np.zeros(total_plans)]
        for model_name in ["inception","resnet","mobilenet"]:
            model_cons[i][cons_start[model_name]:cons_start[model_name]+len(ins_size[model_name]["intra"])] = ins_size[model_name]["intra"]
            overall_cons[cons_start[model_name]:cons_start[model_name]+len(ins_size[model_name]["intra"])] = ins_size[model_name]["intra"]
            overall_E[cons_start[model_name]:cons_start[model_name] + len(ins_size[model_name]["intra"])] = np.array(ins_size[model_name]["efficiency"])*np.array(ins_size[model_name]["intra"])
            if len(np.array(ins_size[model_name]["intra"]))==1 and ins_size[model_name]["intra"][0]==0:
                ins_num_upper[cons_start[model_name]:cons_start[model_name] + len(ins_size[model_name]["intra"])] = 0
            else:
                ins_num_upper[cons_start[model_name]:cons_start[model_name] + len(ins_size[model_name]["intra"])] = np.floor(self.C_upper/np.array(ins_size[model_name]["intra"]))
            ins_num_lower[i][cons_start[model_name]:cons_start[model_name] + len(ins_size[model_name]["intra"])] =np.ones(len(ins_size[model_name]["intra"]))
            i = i+1
        return overall_cons,model_cons,overall_E,ins_num_upper.reshape((total_plans,1)),ins_num_lower
    def resource_allocation(self,ins_size,F):
        #print("++++++++++",ins_size)
        self.C_upper = round(self.CPU_Cores*F)
        result = self.cpu_const(ins_size)
        overall_cons = result[0]
        model_cons = result[1]
        overall_E = result[2]
        ins_num_upper = result[3]
        ins_num_lower = result[4]
        Z = cp.Variable((len(overall_cons),1),integer=True)
        obj = cp.Maximize(overall_E@Z)
        prob = cp.Problem(obj,[overall_cons@Z<=self.CPU_Cores,
                               model_cons[0]@Z<=self.C_upper,model_cons[1]@Z<=self.C_upper,model_cons[2]@Z<=self.C_upper,Z<=ins_num_upper,
                               Z>=np.zeros(shape=(len(overall_cons),1))])
        #ins_num_lower[0] @ Z >=1, ins_num_lower[1] @ Z >=1, ins_num_lower[2] @ Z >=1,
        prob.solve(solver=cp.GLPK_MI)
        ins_num = {"inception":None,"resnet":None,"mobilenet":None}
        ins_num_value = np.array(Z.value).flatten()
        print("ins_num_value",ins_num_value)
        cons_start = {"inception": 0, "resnet": len(ins_size["inception"]["intra"]),
                      "mobilenet": len(ins_size["resnet"]["intra"]) + len(ins_size["inception"]["intra"])}
        for model_name in self.model_name_list:
            i = 0
            result = []

            for elem in ins_num_value[cons_start[model_name]:cons_start[model_name] + len(ins_size[model_name]["intra"])]:
                if elem>0:
                    result.append({"ins_num": int(elem),"plan_index": i})
                i = i+1
            if len(result)==0:
                ins_num[model_name] = [{"ins_num": 0,"plan_index": 0}]
            else:
                ins_num[model_name] = result
        return ins_num
class ModelOptimizor:
    """
    depend on the resource allocation strategy of the paper to confirm the instance size
    and instance numHer of Inception, ResNet and mobilenet.
    """
    def __init__(self):
        # 1. load the prediction models for inference latency
        self.complete_latency_model = {}
        self.complete_throughput_model = {}
        self.interfer_factor_model={}
        self.model_latency_ratio = {}
        self.model_name_list = ["inception","resnet","mobilenet"]
        self.model_latency_interference_factor = pd.read_excel("model_zoo/edge_model_interfernce_factor.xlsx",index_col=0)

        self.layer_min_baseline ={"inception":0,"resnet":19,"mobilenet":0}
        for model_name in ["inception","resnet","mobilenet"]:
            model = joblib.load("model_zoo/performance_model/latency/"+model_name+"_complete_latency_model.pkl")
            self.complete_latency_model[model_name] = model
            #print("************",self.test_predicted_model(model))
            self.model_latency_ratio[model_name] = pd.read_excel("model_zoo/model_ratio_info.xlsx",sheet_name=model_name)

        self.model_frame_rate = {"inception":4,"resnet":5,"mobilenet":6}
        # 4. load the mobile user's fee.
        self.model_weights = {"inception":1,"resnet":1,"mobilenet":1}
        # 5. load the configuration info of the system
        self.TOTAL_CPU_Cores = 12
        self.layer_nums = {"inception":20,"resnet":21,"mobilenet":16}

        self.model_mobile_time_no_inter = {"inception":0.166,"resnet":0.145,"mobilenet":0.164}
        self.model_mobile_time = {"inception":0.221,"resnet":0.193,"mobilenet":0.164}
        self.model_mobile_interfer_factor = {"resnet": 1.331, "inception": 1.3313, "mobilenet": 1}

        self.model_SLA_factor = 0.9
        '''
        self.model_SLA = {}
        self.model_SLA_no_mobile_inter = {}
        for model_name in self.model_name_list:
            self.model_SLA[model_name] = self.model_mobile_time[model_name]*self.model_SLA_factor
            self.model_SLA_no_mobile_inter[model_name] = self.model_mobile_time_no_inter[model_name] * self.model_SLA_factor
        '''

        self.model_info = ModelInfo()
        self.fairness_factor = 1
    def get_edge_latency(self,model_name,intra,k):
        edge_latency = self.get_complete_latency(model_name, intra) * self.get_latency_ratio(model_name, intra, k)*self.get_interference_factor(model_name,intra)
        return edge_latency

    def get_complete_latency(self,model_name,intra):
        """
        :param model_name:
        :param intra:
        :return:
        """
        intra = [1.0/intra,1.0/intra]
        latency = self.complete_latency_model[model_name].predict([intra])[0]
        return latency

    def get_feasible_plan_by_name(self,model_name,bandwidth,SLA_factor=None,fairness = None):
        """
       :param Handwidth: dict.e.g.{"inception":X,"resnet":X,"mobilenet":X}
       :return:result dict. e.g.{"inception":{"intra":,"k":,"efficiency":},"resnet":{...},"mobilenet":{...}}
       tips: 0<=k<=$layer_num. When k = 0, it means that the model runs on the edge completely.
        when k = $layer_num means the model runs on the mobile completely.
        Otherwise, it runs in a byHrid way.
        求解过程：
        1. 根据不同切点下的网络延迟和mobile端延迟之和，过滤掉不满足SLA的切点，找出候选切点
        2. 评估候选切点对应的最小intra值和吞吐，确定可行切点
        3. 从可行切点中选出效率最高的切点及相应的资源
       """
        mobile_latency_list = self.model_info.get_mobile_latency(model_name)
        layer_size_list = self.model_info.get_layer_size(model_name)  # the unit if byte
        if SLA_factor == None:
            SLA = self.model_mobile_time[model_name]*0.9
        else:
            SLA = self.model_mobile_time[model_name] * SLA_factor
        if fairness == None:
            CPU_upper = 8
        else:
            CPU_upper = int(min(round(self.TOTAL_CPU_Cores*fairness),8))
        # step 1: find out all the feasible partition plans for #model_name
        feasible_partition_plan = []
        for k in range(self.layer_nums[model_name] - 1):  # can not select the last prediction layer
            # 1.1 evaluate the sum of the upload latency and the mobile latency
            layer_size = layer_size_list[k]
            mobile_latency = mobile_latency_list[k] * self.model_mobile_interfer_factor[model_name]
            upload_latency = layer_size / bandwidth
            e2e = mobile_latency+upload_latency

            if round(mobile_latency + upload_latency, 3) > round(SLA, 3):
                continue  # this partition point k is unaHle to meet ths SLA
            else:
                # 2. find out the minimun intra that can make the k a feasible partition piont
                SLA_gap = SLA - (mobile_latency + upload_latency)
                for intra in range(1, CPU_upper + 1):
                    if k==0:
                        edge_latency = self.get_edge_latency(model_name, intra, 0)
                    else:
                        edge_latency = self.get_edge_latency(model_name, intra, k+1)
                    queue_len = math.floor(SLA_gap / edge_latency)  # 这里可以乘以一个系数
                    if k == 7 and model_name == "mobilenet":
                        print("单独 intra", intra,k, edge_latency, SLA,SLA_gap, math.floor(SLA_gap / edge_latency),
                              mobile_latency, upload_latency, layer_size,bandwidth, math.floor(SLA_gap / edge_latency))
                    if queue_len > 0:
                        user_num_per_ins = queue_len
                        efficienty = round(1.0 / edge_latency/min(self.model_frame_rate[model_name],1.0 / upload_latency)/intra,3)
                        feasible_partition_plan.append((k, intra, efficienty,user_num_per_ins))
                        break
                        # print("队列长度",queue_len)

        # Step 2: select the most efficient partition plan for $model_name
        return feasible_partition_plan

    def get_model_e2e_no_inter(self,model_name,k,intra,bandwidth):
        mobile_latency_list = self.model_info.get_mobile_latency(model_name)
        layer_size_list = self.model_info.get_layer_size(model_name)  # the unit if byte
        layer_size = layer_size_list[k]
        mobile_latency = mobile_latency_list[k] * self.model_mobile_interfer_factor[model_name]
        upload_latency = layer_size / bandwidth
        edge_latency = self.get_complete_latency(model_name, intra) * self.get_latency_ratio(model_name, intra, k)
        e2e = mobile_latency+upload_latency+edge_latency
        return e2e,edge_latency,upload_latency

    def get_feasible_plan_by_name_no_inter(self,model_name, bandwidth,SLA_factor=None):
        """
        获取模型在特定带宽下可行的切分方案和效用最大的切分方案
        :param model_name:
        :param bandwidth:
        :return:
        """
        result = {}
        mobile_latency_list = self.model_info.get_mobile_latency(model_name)
        layer_size_list = self.model_info.get_layer_size(model_name)  # the unit if byte
        if SLA_factor == None:
            SLA = round(self.model_mobile_time[model_name]*0.9,3)
        else:
            SLA = round(self.model_mobile_time[model_name] * SLA_factor, 3)
        # step 1: find out all the feasible partition plans for #model_name
        #print(SLA)
        feasible_partition_plan = []
        for k in range(self.layer_nums[model_name] - 2):  # can not select the last prediction layer
            # 1.1 evaluate the sum of the upload latency and the mobile latency
            layer_size = layer_size_list[k]
            mobile_latency = mobile_latency_list[k]*self.model_mobile_interfer_factor[model_name]
            upload_latency = layer_size / bandwidth
            print(mobile_latency+upload_latency,round(SLA,3))
            if round(mobile_latency + upload_latency, 3) > round(SLA, 3):
                continue  # this partition point k is unaHle to meet ths SLA
            else:
                # 2. find out the minimun intra that can make the k a feasible partition piont
                SLA_gap = SLA - (mobile_latency + upload_latency)
                # print('k', k, "m", mobile_latency, "u", upload_latency, "gap",SLA_gap )
                if k == 0:
                    intra, edge_latency = self.find_matched_intra_no_inter(model_name, 0, SLA_gap)
                else:
                    intra, edge_latency = self.find_matched_intra_no_inter(model_name, k + 1, SLA_gap)
                if intra != None:
                    user_num_per_ins = math.floor(1.0 / edge_latency / min(self.model_frame_rate[model_name], 1.0 / upload_latency))
                    efficienty = round(1.0 / edge_latency / min(self.model_frame_rate[model_name], 1.0 / upload_latency) / intra, 3)
                    feasible_partition_plan.append((k, intra, efficienty, user_num_per_ins))
                    print(k, intra, efficienty, user_num_per_ins)
                    print()
        select_plan = None
        max_efficiency = 0
        for plan in feasible_partition_plan:
            # 2.1 efficiency = edge throughput / min(frame rate, net throughput)* model weights /
            efficienty = plan[2]
            if round(efficienty, 3) > max_efficiency:
                select_plan = plan
                max_efficiency = round(efficienty, 3)

        # Step3: generate the final
        if select_plan == None:
            result = {"intra": -1, "k": -1, "efficiency": 0,"user_num_per_ins":0}
        else:
            result = {"intra": select_plan[1], "k": select_plan[0], "efficiency": max_efficiency,"user_num_per_ins":select_plan[3]}

        return feasible_partition_plan,result

    def get_latency_ratio(self,model_name,intra,k):
        ratio = None
        if intra==1:
            #print("k",k,model_name)
            ratio = (self.model_latency_ratio[model_name].iloc[0,1:].values)[k]
        else:
            ratio = (self.model_latency_ratio[model_name].iloc[1, 1:].values)[k]
        return ratio

    def get_interference_factor(self,model_name,intra):
        factor = self.model_latency_interference_factor.loc["intra="+str(intra),model_name]
        if model_name == "mobilenet" and intra>1:
            pass
            #print("mobilenet ask for the interference factor of intra >1 ")
        #print("干扰因子",model_name,intra,factor)
        return factor

    def find_matched_intra(self,model_name,k,SLA_gap):
        """
        find out the minimun intra that can make the k a feasible partition piont
        :param model_name:
        :param k:
        :param SLA_gap:(seconds)
        :return:
        """
        # 1. get the solo latency of the model under $k with different intra.
        target_intra = None
        edge_latency = None
        for intra in range(1,8+1):
            # 1.1 get the complete latency
            solo_latency = self.get_complete_latency(model_name,intra)*self.get_latency_ratio(model_name,intra,k)
            #print(k,intra,"solo_latency",round(solo_latency,3))
            if round(solo_latency,3)>round(SLA_gap,3):
                continue # this intra is unaHle to ensure SLA.
            else:
                # 2.1 consider the inference factor to get the interference latency
                co_located_latency = solo_latency*self.get_interference_factor(model_name,intra)
                #print("co_located_latency",co_located_latency)
                if round(co_located_latency,3)>round(SLA_gap,3):
                    continue
                else:
                    target_intra = intra
                    edge_latency = co_located_latency
                    break
        return target_intra,edge_latency

    def find_matched_intra_no_inter(self,model_name,k,SLA_gap):
        """
        find out the minimun intra that can make the k a feasible partition piont
        :param model_name:
        :param k:
        :param SLA_gap:(seconds)
        :return:
        """
        # 1. get the solo latency of the model under $k with different intra.
        target_intra = None
        edge_latency = None
        for intra in range(1,8+1):
            # 1.1 get the complete latency
            solo_latency = self.get_complete_latency(model_name,intra)*self.get_latency_ratio(model_name,intra,k)
            if round(solo_latency,3)>round(SLA_gap,3):
                continue # this intra is unaHle to ensure SLA.
            else:
                # 2.1 consider the inference factor to get the interference latency
                #print("co_located_latency",co_located_latency)
                if round(solo_latency,3)>round(SLA_gap,3):
                    continue
                else:
                    target_intra = intra
                    edge_latency = solo_latency
                    break
        return target_intra,edge_latency

    def find_model_feasible_partition_points_no_inter(self,model_list = ["inception","resnet","mobilenet"]):
        """
        求出baseline所有可行的intra，找出吞吐最大的
        :param Handwidth:
        :return:
        """
        result = {}
        #wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx", index_col=0)
        #model_trace={"inception":wifi_trace["trace1"].values,"resnet":wifi_trace["trace2"].values,"mobilenet":wifi_trace["trace3"].values}
        wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx", index_col=0,sheet_name="background")
        model_trace={"inception":wifi_trace["bandwidth"].values}
        for model_name in model_list:
            '''
            feasible_writer = pd.ExcelWriter(
                "experiment/model_partition/hitdl_feasible_plans/" + model_name + "_feasible_plans.xlsx")
            '''
            feasible_writer = pd.ExcelWriter("background/model_partition/" + model_name + "_feasible_plans.xlsx")
            result = {"k": [], "intra": [], "efficiency": [], "user_num_per_ins": [], "net": []}
            result_select = {"k": [], "intra": [], "efficiency": [], "user_num_per_ins": [], "net": []}
            for i in [0,1,2]:#range(len(model_trace[model_name])):
                bd = model_trace[model_name][i] * 1024 * 1024 / 8
                plans, select = self.get_feasible_plan_by_name_no_inter(model_name, bd)
                result_select["k"].append(select["k"])
                result_select["intra"].append(select["intra"])
                result_select["efficiency"].append(select["efficiency"])
                result_select["user_num_per_ins"].append(select["user_num_per_ins"])
                result_select["net"].append(model_trace[model_name][i])
                for plan in plans:
                    result["k"].append(plan[0])
                    result["intra"].append(plan[1])
                    result["efficiency"].append(plan[2])
                    result["user_num_per_ins"].append(plan[3])
                    result["net"].append(model_trace[model_name][i])
            result_pd = pd.DataFrame(data=result, index=range(len(result["k"])))
            result_pd.to_excel(feasible_writer, sheet_name="feasible")

            result_pd = pd.DataFrame(data=result_select, index=range(len(result_select["k"])))
            result_pd.to_excel(feasible_writer, sheet_name="select")
            feasible_writer.save()
            feasible_writer.close()

    def get_baseline_intra_with_max_efficiency_no_inter(self,Handwidth):
        """
        求出baseline所有可行的intra，找出吞吐最大的
        :param Handwidth:
        :return:
        """
        result = {}
        for model_name in ["inception","resnet","mobilenet"]:
            Hd = Handwidth[model_name] #H/s
            mobile_latency_list = self.model_info.get_mobile_latency(model_name)
            layer_size_list = self.model_info.get_layer_size(model_name) # the unit if byte
            SLA = self.model_SLA_no_mobile_inter[model_name]
            # step 1: find out all the feasible partition plans for #model_name
            feasible_partition_plan = []
            for k in [0]: #can not select the last prediction layer
                # 1.1 evaluate the sum of the upload latency and the mobile latency
                layer_size = layer_size_list[k]
                mobile_latency = mobile_latency_list[k]
                upload_latency = layer_size/Hd
                #print(mobile_latency, upload_latency,round(SLA,3))
                if round(mobile_latency+upload_latency,3) > round(SLA,3):
                    continue # this partition point k is unaHle to meet ths SLA
                else:
                    # 2. find out the minimun intra that can make the k a feasible partition piont
                    SLA_gap = SLA - (mobile_latency+upload_latency)
                    #print('k', k, "m", mobile_latency, "u", upload_latency, "gap",SLA_gap )
                    intra,edge_latency = self.find_matched_intra_no_inter(model_name,0,SLA_gap)
                    if intra != None:
                        user_num_per_ins = math.floor(1.0/edge_latency/min(self.model_frame_rate[model_name],1.0/upload_latency))
                        feasible_partition_plan.append((k,intra,1.0/edge_latency,user_num_per_ins,1.0/upload_latency))

            # Step 2: select plan that ensure 最大的吞吐ame
            select_plan = None

            select_plan = None
            max_efficiency = 0
            for plan in feasible_partition_plan:
                # 2.1 efficiency = edge throughput / min(frame rate, net throughput)* model weights /intra
                efficienty = plan[2] / min(self.model_frame_rate[model_name], plan[4]) / plan[1]
                if round(efficienty, 3) > max_efficiency:
                    select_plan = plan
                    max_efficiency = round(efficienty, 3)
            # Step3: generate the final
            if select_plan == None:
                result[model_name] = {"intra": -1, "k": -1, "efficiency": 0}
            else:
                efficienty = select_plan[2] / select_plan[1]
                result[model_name] = {"intra": select_plan[1], "k": select_plan[0], "efficiency": max_efficiency}

        return result

    def get_neurosurgeon_ins_size_queue(self,Handwidth,max_intra=8,SLA_factor = None):
        """

               :param Handwidth: dict.e.g.{"inception":X,"resnet":X,"mobilenet":X}
               :return:result dict. e.g.{"inception":{"intra":,"k":,"efficiency":},"resnet":{...},"mobilenet":{...}}
               tips: 0<=k<=$layer_num. When k = 0, it means that the model runs on the edge completely.
                when k = $layer_num means the model runs on the mobile completely.
                Otherwise, it runs in a byHrid way.
                求解过程：
                1. 根据不同切点下的网络延迟和mobile端延迟之和，过滤掉不满足SLA的切点，找出候选切点
                2. 评估候选切点对应的最小intra值和吞吐，确定可行切点
                3. 从可行切点中选出效率最高的切点及相应的资源
               """
        result = {}
        for model_name in ["inception", "resnet", "mobilenet"]:
            Hd = Handwidth[model_name]  # H/s
            mobile_latency_list = self.model_info.get_mobile_latency(model_name)
            layer_size_list = self.model_info.get_layer_size(model_name)  # the unit if byte
            if SLA_factor == None:
                SLA = self.model_mobile_time[model_name]
            else:
                SLA = SLA_factor*self.model_mobile_time[model_name]
            min_e2e = None
            select_plan = None
            # step 1: find out all the feasible partition plans for #model_name
            for k in range(self.layer_nums[model_name]):  # can not select the last prediction layer
                # 1.1 evaluate the sum of the upload latency and the mobile latency

                if k<self.layer_nums[model_name]-1:
                    layer_size = layer_size_list[k]
                    mobile_latency = mobile_latency_list[k] * self.model_mobile_interfer_factor[model_name]
                    upload_latency = layer_size / Hd
                else:
                    upload_latency = 0
                    mobile_latency = mobile_latency_list[self.layer_nums[model_name]-1] * self.model_mobile_interfer_factor[model_name]
                e2e = mobile_latency + upload_latency
                # print(mobile_latency, upload_latency,round(SLA,3))
                if round(mobile_latency + upload_latency, 3) > round(SLA, 3):
                    continue  # this partition point k is unaHle to meet ths SLA
                else:

                    # 2. find out the minimun intra that can make the k a feasible partition piont
                    SLA_gap = SLA - (mobile_latency + upload_latency)
                    intra = max_intra
                    if k == 0:
                        edge_latency = self.get_edge_latency(model_name, intra, 0)
                    elif k<self.layer_nums[model_name]-2:
                        edge_latency = self.get_edge_latency(model_name, intra, k + 1)
                    else:
                        edge_latency = 0
                    e2e = e2e + edge_latency
                    if min_e2e == None or round(e2e,3)< min_e2e:
                        if edge_latency!=0:
                            queue_len = math.floor(SLA_gap / edge_latency)  # 这里可以乘以一个系数
                            if queue_len > 0:
                                min_e2e = e2e
                                user_num_per_ins = queue_len
                                efficienty = 1.0 / edge_latency/min(self.model_frame_rate[model_name],1.0 / upload_latency)/intra
                                select_plan = {"intra": intra, "k": k, "efficiency": efficienty,"user_num_per_ins": user_num_per_ins,"e2e_no_queue":e2e}
                        else:
                            select_plan = {"intra": 0, "k": self.layer_nums[model_name]-1, "efficiency": 0,
                                           "user_num_per_ins": 0,"e2e_no_queue":e2e}
            if select_plan == None:
                select_plan = {"intra": -1, "k": -1*self.layer_nums[model_name], "efficiency": -1,"user_num_per_ins": 0,"e2e_no_queue":0}
            result[model_name] = select_plan
        return result

    def get_neurosurgeon_ins_size_no_inter(self,Handwidth):
        model_info = ModelInfo()
        """
        :param Handwidth: dict.e.g.{"inception":X,"resnet":X,"mobilenet":X}
        :return:result dict. e.g.{"inception":{"intra":,"k":,"efficiency":,user_num_per_ins:},"resnet":{...},"mobilenet":{...}}
        tips: 0<=k<=$layer_num. When k = 0, it means that the model runs on the edge completely.
         when k = $layer_num means the model runs on the mobile completely.
         Otherwise, it runs in a byHrid way.
         求解过程：
         1. 根据不同切点下的网络延迟和mobile端延迟之和，过滤掉不满足SLA的切点，找出候选切点
         2. 评估候选切点对应的最小intra值和吞吐，确定可行切点
         3. 从可行切点中选出效率最高的切点及相应的资源
        """
        result = {}
        for model_name in ["inception","resnet","mobilenet"]:
            Hd = Handwidth[model_name] #H/s
            mobile_latency_list = self.model_info.get_mobile_latency(model_name) # without any interference
            layer_size_list = self.model_info.get_layer_size(model_name) # the unit if byte
            SLA = self.model_SLA_no_mobile_inter[model_name]
            # step 1: find out all the feasible partition plans for #model_name
            feasible_partition_plan = []
            for k in range(self.layer_nums[model_name] - 1): #can not select the last prediction layer
                # 1.1 evaluate the sum of the upload latency and the mobile latency
                layer_size = layer_size_list[k]
                mobile_latency = mobile_latency_list[k]
                upload_latency = layer_size/Hd
                if round(mobile_latency+upload_latency,3) > round(SLA,3):
                    continue # this partition point k is unaHle to meet ths SLA
                else:
                    # 2. find out the minimun intra that can make the k a feasible partition piont
                    SLA_gap = SLA - (mobile_latency+upload_latency)
                    #print('k', k, "m", mobile_latency, "u", upload_latency, "gap",SLA_gap )
                    intra,edge_latency = self.find_matched_intra_no_inter(model_name,k,SLA_gap)
                    if intra != None:
                        user_num_per_ins = math.floor(1.0/edge_latency/min(self.model_frame_rate[model_name],1.0/upload_latency))
                        feasible_partition_plan.append((k,intra,1.0/edge_latency,user_num_per_ins,1.0/upload_latency))

            # Step 2: select the most efficient partition plan for $model_name
            select_plan = None
            max_efficiency = 0
            #print("k,intra,edge_throughput,user_num_per_ins,network_throughput")
            #print(model_name,feasible_partition_plan)
            for plan in feasible_partition_plan:
                # 2.1 efficiency = edge throughput / min(frame rate, net throughput)* model weights /
                efficienty = self.model_weights[model_name]*plan[2]/min(self.model_frame_rate[model_name],plan[4])/plan[1]
                #print(efficienty)
                if round(efficienty,3)>max_efficiency:
                    select_plan = plan
                    max_efficiency = round(efficienty,3)

            # Step3: generate the final
            if select_plan == None:
                result[model_name] = {"intra": -1, "k": -1, "efficiency": 0,"user_num_per_ins": 0}
            else:
                result[model_name] = {"intra":select_plan[1],"k":select_plan[0],"efficiency":max_efficiency,"user_num_per_ins":select_plan[3]}

        return result

    def get_baseline_ins_size_no_inter(self,Handwidth,SLA=None):
        model_info = ModelInfo()
        """
        :param Handwidth: dict.e.g.{"inception":X,"resnet":X,"mobilenet":X}
        :return:result dict. e.g.{"inception":{"intra":,"k":,"efficiency":,user_num_per_ins:},"resnet":{...},"mobilenet":{...}}
        tips: 0<=k<=$layer_num. When k = 0, it means that the model runs on the edge completely.
         when k = $layer_num means the model runs on the mobile completely.
         Otherwise, it runs in a byHrid way.
         求解过程：
         1. 根据不同切点下的网络延迟和mobile端延迟之和，过滤掉不满足SLA的切点，找出候选切点
         2. 评估候选切点对应的最小intra值和吞吐，确定可行切点
         3. 从可行切点中选出效率最高的切点及相应的资源
        """
        result = {}
        for model_name in ["inception","resnet","mobilenet"]:
            Hd = Handwidth[model_name] #H/s
            mobile_latency_list = self.model_info.get_mobile_latency(model_name)
            layer_size_list = self.model_info.get_layer_size(model_name) # the unit if byte
            if SLA == None:
                SLA = self.model_mobile_time[model_name]*self.model_SLA_factor
            # step 1: find out all the feasible partition plans for #model_name
            k=0
            layer_size = layer_size_list[k]
            mobile_latency = mobile_latency_list[k]
            upload_latency = layer_size/Hd
            if round(mobile_latency+upload_latency,3) > round(SLA,3):
                result[model_name] = {"intra": -1, "k": -1, "efficiency": 0,"user_num_per_ins": 0,"edge_throughput":0}
            else:
                # 2. find out the minimun intra that can make the k a feasible partition piont
                SLA_gap = SLA - (mobile_latency+upload_latency)
                #print('k', k, "m", mobile_latency, "u", upload_latency, "gap",SLA_gap )
                intra,edge_latency = self.find_matched_intra_no_inter(model_name,k,SLA_gap)
                if intra != None:
                    user_num_per_ins = math.floor(1.0/edge_latency/min(self.model_frame_rate[model_name],1.0/upload_latency))
                    efficiency = round(self.model_weights[model_name]*(1.0/edge_latency/min(self.model_frame_rate[model_name],1.0/upload_latency))/intra,3)
                    result[model_name] =  {"intra":intra,"k":k,"efficiency":efficiency,"user_num_per_ins":user_num_per_ins,"edge_throughput":1.0/edge_latency}
                else:
                    result[model_name] = {"intra": -1, "k": -1, "efficiency": 0, "user_num_per_ins": 0,"edge_throughput":0}

        return result

    def get_ins_size_queue(self,Handwidth,max_intra=4,SLA_factor = None):
        """

               :param Handwidth: dict.e.g.{"inception":X,"resnet":X,"mobilenet":X}
               :return:result dict. e.g.{"inception":{"intra":,"k":,"efficiency":},"resnet":{...},"mobilenet":{...}}
               tips: 0<=k<=$layer_num. When k = 0, it means that the model runs on the edge completely.
                when k = $layer_num means the model runs on the mobile completely.
                Otherwise, it runs in a byHrid way.
                求解过程：
                1. 根据不同切点下的网络延迟和mobile端延迟之和，过滤掉不满足SLA的切点，找出候选切点
                2. 评估候选切点对应的最小intra值和吞吐，确定可行切点
                3. 从可行切点中选出效率最高的切点及相应的资源
               """
        result = {}
        for model_name in ["inception", "resnet", "mobilenet"]:
            Hd = Handwidth[model_name]  # H/s
            mobile_latency_list = self.model_info.get_mobile_latency(model_name)
            layer_size_list = self.model_info.get_layer_size(model_name)  # the unit if byte
            if SLA_factor == None:
                SLA = self.model_mobile_time[model_name]*0.9
            else:
                SLA = SLA_factor*self.model_mobile_time[model_name]

            # step 1: find out all the feasible partition plans for #model_name
            feasible_partition_plan = []
            efficienty_list = []
            for k in range(self.layer_nums[model_name]):  # can not select the last prediction layer
                # 1.1 evaluate the sum of the upload latency and the mobile latency
                if k<self.layer_nums[model_name]-1:
                    layer_size = layer_size_list[k]
                    upload_latency = layer_size / Hd
                else:
                    upload_latency = 0
                mobile_latency = mobile_latency_list[k] * self.model_mobile_interfer_factor[model_name]
                e2e = mobile_latency+upload_latency
                # print(mobile_latency, upload_latency,round(SLA,3))
                if round(mobile_latency + upload_latency, 3) > round(SLA, 3):
                    continue  # this partition point k is unaHle to meet ths SLA
                else:
                    # 2. find out the minimun intra that can make the k a feasible partition piont
                    #print("model name == sla",SLA, mobile_latency,upload_latency)
                    SLA_gap = SLA - (mobile_latency + upload_latency)
                    for intra in range(1, max_intra + 1):
                        if k==0:
                            edge_latency = self.get_edge_latency(model_name, intra, 0)
                        elif k<self.layer_nums[model_name]-2:
                            edge_latency = self.get_edge_latency(model_name, intra, k+1)
                        else:
                            edge_latency = 0
                        if k ==7 and model_name =="mobilenet":
                            print(model_name,"intra ins_queue_size",intra,edge_latency,SLA_gap,mobile_latency,upload_latency,layer_size_list[k],Hd)
                        if edge_latency !=0:
                            queue_len = math.floor(SLA_gap / edge_latency)  # 这里可以乘以一个系数
                            if queue_len > 0:
                                user_num_per_ins = queue_len
                                e2e = e2e+edge_latency
                                efficienty_list.append([k,intra,round(1.0 / edge_latency/min(self.model_frame_rate[model_name], 1.0 / upload_latency) / intra,2)])
                                feasible_partition_plan.append((k, intra, round(1.0 / edge_latency,2), user_num_per_ins, 1.0 / upload_latency,edge_latency,(queue_len-1)*edge_latency,e2e))
                        else:
                            feasible_partition_plan.append((k, 1,np.inf, 0,1, 0, 0, 0))

            # Step 2: select the most efficient partition plan for $model_name
            select_plan = None
            max_efficiency = 0
            print(model_name)
            print(efficienty_list)
            for plan in feasible_partition_plan:
                # 2.1 efficiency = edge throughput / min(frame rate, net throughput)* model weights / intra
                efficienty = plan[2] / min(self.model_frame_rate[model_name],plan[4]) / plan[1]
                # print(efficienty)
                if round(efficienty, 3) > max_efficiency:
                    select_plan = plan
                    max_efficiency = round(efficienty, 3)

            # Step3: generate the final solution
            if select_plan != None:
                result[model_name] = {"intra": select_plan[1], "k": select_plan[0], "efficiency": max_efficiency,
                                  "user_num_per_ins": select_plan[3],"e2e_no_queue":select_plan[7]}
            else:
                result[model_name] = {"intra": -1, "k": -1*self.layer_nums[model_name], "efficiency": -1,
                                  "user_num_per_ins":0,"e2e_no_queue":0}
        print("++++++++++++++++",result)
        return result

    def get_baseline_ins_size_queue(self,Handwidth,max_intra=8,SLA_factor = None):
        """
               :param Handwidth: dict.e.g.{"inception":X,"resnet":X,"mobilenet":X}
               :return:result dict. e.g.{"inception":{"intra":,"k":,"efficiency":},"resnet":{...},"mobilenet":{...}}
               tips: 0<=k<=$layer_num. When k = 0, it means that the model runs on the edge completely.
                when k = $layer_num means the model runs on the mobile completely.
                Otherwise, it runs in a byHrid way.
                求解过程：
                1. 根据不同切点下的网络延迟和mobile端延迟之和，过滤掉不满足SLA的切点，找出候选切点
                2. 评估候选切点对应的最小intra值和吞吐，确定可行切点
                3. 从可行切点中选出效率最高的切点及相应的资源
               """
        result = {}
        for model_name in ["inception", "resnet", "mobilenet"]:
            Hd = Handwidth[model_name]  # H/s
            mobile_latency_list = self.model_info.get_mobile_latency(model_name)
            layer_size_list = self.model_info.get_layer_size(model_name)  # the unit if byte
            if SLA_factor == None:
                SLA = self.model_mobile_time[model_name]*0.9
            else:
                SLA = SLA_factor*self.model_mobile_time[model_name]

            # step 1: find out all the feasible partition plans for #model_name
            feasible_partition_plan = []
            for k in [0]:  # can not select the last prediction layer
                # 1.1 evaluate the sum of the upload latency and the mobile latency
                layer_size = layer_size_list[k]
                mobile_latency = mobile_latency_list[k] * self.model_mobile_interfer_factor[model_name]
                upload_latency = layer_size / Hd
                e2e = mobile_latency+upload_latency
                # print(mobile_latency, upload_latency,round(SLA,3))
                if round(mobile_latency + upload_latency, 3) > round(SLA, 3):
                    continue  # this partition point k is unaHle to meet ths SLA
                else:
                    # 2. find out the minimun intra that can make the k a feasible partition piont
                    SLA_gap = SLA - (mobile_latency + upload_latency)

                    for intra in range(1, max_intra + 1):
                        edge_latency = self.get_edge_latency(model_name, intra, 0)
                        #print(SLA_gap,edge_latency)
                        queue_len = math.floor(SLA_gap / edge_latency)  # 这里可以乘以一个系数
                        if queue_len > 0:
                            user_num_per_ins = queue_len
                            e2e = e2e + edge_latency
                            feasible_partition_plan.append(
                                (k, intra, 1.0 / edge_latency, user_num_per_ins, 1.0 / upload_latency,e2e))
                            # print("队列长度",queue_len)
            # Step 2: select the most efficient partition plan for $model_name
            select_plan = None
            max_efficiency = 0
            # print("k,intra,edge_throughput,user_num_per_ins,network_throughput")

            for plan in feasible_partition_plan:
                # 2.1 efficiency = edge throughput / min(frame rate, net throughput)* model weights /
                efficienty = plan[2] / min(self.model_frame_rate[model_name], plan[4]) / plan[1]
                if round(efficienty, 3) > max_efficiency:
                    select_plan = plan
                    max_efficiency = round(efficienty, 3)

            # Step3: generate the final solution
            if select_plan == None:
                result[model_name] = {"intra":-1, "k": -1*self.layer_nums[model_name], "efficiency": -1,
                                      "user_num_per_ins": 0, "e2e_no_queue": 0}
            else:
                result[model_name] = {"intra": select_plan[1], "k": select_plan[0], "efficiency": max_efficiency,
                                      "user_num_per_ins": select_plan[3],"e2e_no_queue":select_plan[5]}
        # print(result)
        return result

    def get_model_utility_ratio(self,model_name,ins_num,model_efficiency,min_efficiency,availaHle_CPU_cores,sys_current_utility,model_intra):
        # 1. get the utilify of model $model_name
        # model_utility = model_weights*ins_num*user_num_per_ins
        # ratio = model_utility/(model_utility + availaHle_CPU_Cores*min_efficiency)
        model_utility = model_efficiency*ins_num*model_intra
        #print(self.model_weights[model_name],user_num_per_ins,ins_num)
        try:
            ratio = round(model_utility/(model_utility+min_efficiency*availaHle_CPU_cores+sys_current_utility),2)
        except Exception as e:
            pass
        #print("model_name,ins_num,model_utility,model_utility+min_efficiency*availaHle_CPU_cores,ratio")
        #print(model_name,ins_num,model_utility,model_utility+min_efficiency*availaHle_CPU_cores,ratio)
        return ratio

    def get_edge_throughput_no_inter(self,model_name,intra,k):
        edge_latency = self.get_complete_latency(model_name, intra) * self.get_latency_ratio(model_name, intra, k)
        return 1/edge_latency

    def get_edge_throughput(self,model_name,intra,k):
        edge_latency = self.get_complete_latency(model_name, intra) * self.get_latency_ratio(model_name, intra, k)*self.get_interference_factor(model_name,intra)
        return 1/edge_latency

    def get_used_Cores(self,strategy):
        used_Cores = 0
        for model_name in ["inception", "resnet", "mobilenet"]:
            model_cores = strategy[model_name]["intra"] * strategy[model_name]["ins_num"]
            used_Cores = used_Cores + model_cores
        return used_Cores

    def evalaute(self,strategy):
        """

        :param strategy:
        :return:
        """
        sys_throughput = 0
        sys_users = 0
        used_Cores = 0
        model_throughput = {"inception":0,"resnet":0,"mobilenet":0}
        model_cores = {"inception":0,"resnet":0,"mobilenet":0}
        model_ins = {"inception":0,"resnet":0,"mobilenet":0}
        model_users = {"inception":0,"resnet":0,"mobilenet":0}
        #print(strategy)
        for model_name in ["inception","resnet","mobilenet"]:
            if strategy[model_name]["k"]>=0:
                ins_throughput = self.get_edge_throughput(model_name,strategy[model_name]["intra"],strategy[model_name]["k"])
                ins_num = strategy[model_name]["ins_num"]
                model_throughput[model_name] = ins_throughput*ins_num
                model_cores[model_name] = strategy[model_name]["intra"]*ins_num
                model_users[model_name] = strategy[model_name]["user_num_per_ins"]*ins_num
                model_ins[model_name] = ins_num
                sys_throughput = sys_throughput+model_throughput[model_name]
                sys_users = sys_users+model_users[model_name]
                used_Cores = used_Cores+model_cores[model_name]
            else:
                ins_throughput = 0
                ins_num = 0
                model_throughput[model_name] = 0
                model_cores[model_name] = 0
                model_users[model_name] = 0
                model_ins[model_name] = 0
                sys_throughput = sys_throughput + model_throughput[model_name]
                sys_users = sys_users + model_users[model_name]
                used_Cores = used_Cores + model_cores[model_name]
        result = {"sys_h":sys_throughput,"used_cores":used_Cores,"sys_users":sys_users,
         "I_u":round(model_ins["inception"]*strategy["inception"]["efficiency"]*strategy["inception"]["intra"],2),
          "R_u":round(model_ins["resnet"]*strategy["resnet"]["efficiency"]*strategy["resnet"]["intra"],2),
           "M_u":round(model_ins["mobilenet"]*strategy["mobilenet"]["efficiency"]*strategy["mobilenet"]["intra"],2),
          "I_intra": round(strategy["inception"]["intra"]),
          "R_intra": round(strategy["resnet"]["intra"]),
          "M_intra": round(strategy["mobilenet"]["intra"]),
          "I_ins_num":model_ins["inception"],"R_ins_num":model_ins["resnet"],"M_ins_num":model_ins["mobilenet"],
          "I_cores":model_cores["inception"],"R_cores":model_cores["resnet"],"M_cores":model_cores["mobilenet"],
          'I_users':model_users["inception"],"R_users":model_users["resnet"],"M_users":model_users["mobilenet"]}
        #print("评估函数里每个应用的总体的效用",round(model_ins["inception"]*strategy["inception"]["efficiency"],2),round(model_ins["resnet"]*strategy["resnet"]["efficiency"],2),
        #round(model_ins["mobilenet"] * strategy["mobilenet"]["efficiency"], 2) )
        return result
    def meet_fairness(self,model_cores,fairness):
        cores_upper = round(self.TOTAL_CPU_Cores*fairness)

        result = True
        for model_name in self.model_name_list:
            if model_cores[model_name]>cores_upper:
                result = False
                break
        #print(result,model_cores,cores_upper)
        return result

    def meet_fairness_utility(self,strategy,fairness):
        # 1. compute each models' utility ratio
        total_utility = 0
        #CPU_used = 0
        for model_name in ["inception","resnet","mobilenet"]:
            if strategy[model_name]["intra"]>0:
                total_utility = total_utility+strategy[model_name]["ins_num"]*strategy[model_name]["efficiency"]*strategy[model_name]["intra"]
            #CPU_used = CPU_used +  total_utility+strategy[model_name]["ins_num"]*strategy[model_name]["intra"]
        result = True
        ratios = {}
        for model_name in  ["inception","resnet","mobilenet"]:
            if total_utility !=0 and strategy[model_name]["intra"]>0:
                ratio = strategy[model_name]["ins_num"]*strategy[model_name]["efficiency"]*strategy[model_name]["intra"]/total_utility
            else:
                ratio = 0
            ratios[model_name] = ratio
            #print(round(ratio,4),round(fairness,4))
            if round(ratio,2)>round(fairness,2):
                result = False
        return result,ratios
    def get_optimal_strategy(self,ins_size):
        fairness_factor = self.fairness_factor
        # 1. 为3个模型迭代所有可能的分配值
        inception_intra = ins_size["inception"]["intra"]
        resnet_intra = ins_size["resnet"]["intra"]
        mobilenet_intra = ins_size["mobilenet"]["intra"]
        optimal_allocation = None
        sys_max_throughput = None
        meet_fairness = False
        if inception_intra>0:
            inception_ins_num_max = math.floor(12 / inception_intra)
        else:
            inception_ins_num_max = 0
        if resnet_intra>0:
            resnet_ins_num_max = math.floor(12 / resnet_intra)
        else:
            resnet_ins_num_max = 0
        if mobilenet_intra>0:
            mobilenet_ins_num_max = math.floor(12 / mobilenet_intra)
        else:
            mobilenet_ins_num_max = 0
        #print("*************",inception_ins_num_max,resnet_ins_num_max,mobilenet_ins_num_max)
        for inception_num in range(0,inception_ins_num_max+1):
            for resnet_num in range(0,resnet_ins_num_max+1):
                for mobilenet_num in range(0,mobilenet_ins_num_max+1):
                    #print("*************",inception_num,resnet_num,mobilenet_num)
                    if inception_num*inception_intra+resnet_num*resnet_intra+mobilenet_num*mobilenet_intra>12: #没有核给mobilenet了
                        continue
                    else:
                        pass
                    strategy = {"inception": {}, "resnet": {}, "mobilenet": {}}
                    ins_num = {"inception": {}, "resnet": {}, "mobilenet": {}}
                    #mobilenet_num = math.floor(mobilenet_cores/mobilenet_intra)

                    ins_num["inception"]["ins_num"] = inception_num
                    ins_num["resnet"]["ins_num"] = resnet_num
                    ins_num["mobilenet"]["ins_num"] = mobilenet_num

                    for model_name in ["inception", "resnet", "mobilenet"]:
                        strategy[model_name].update(ins_size[model_name])
                        strategy[model_name].update(ins_num[model_name])
                    #print("++++++++++++",strategy,self.get_sys_utility(strategy))

                    if self.meet_fairness(strategy,fairness_factor)[0]:
                        meet_fairness = True
                        if optimal_allocation == None:
                            optimal_allocation = strategy
                            sys_max_throughput = round(self.get_sys_utility(strategy),3)

                        if round(self.get_sys_utility(strategy),3)>=sys_max_throughput:
                            #print("+++++++++++++++",self.meet_fairness(strategy, fairness_factor)[1],inception_ins_num,resnet_ins_num,mobilenet_ins_num)
                            optimal_allocation = strategy
                            sys_max_throughput = self.get_sys_utility(strategy)
        if meet_fairness: #没有满足条件的解
            ins_num = {"inception": {}, "resnet": {}, "mobilenet": {}}
            for model_name in  ["inception", "resnet", "mobilenet"]:
                ins_num[model_name]["ins_num"] = optimal_allocation[model_name]["ins_num"]
            return ins_num
        else:
            ins_num = {"inception": {}, "resnet": {}, "mobilenet": {}}
            for model_name in  ["inception", "resnet", "mobilenet"]:
                ins_num[model_name]["ins_num"] = 0
            return ins_num

    def get_weighted_strategy(self,ins_size,model_weights):
        model_cores = {"inception": 0, "resnet": 0, "mobilenet": 0}
        if model_weights == None:
            model_cores = {"inception": 7, "resnet": 4, "mobilenet": 1}
        else:
            for model_name in  ["resnet","inception"]:
                model_cores[model_name] = round(self.TOTAL_CPU_Cores*model_weights[model_name])
            model_cores["mobilenet"] = self.TOTAL_CPU_Cores-model_cores["resnet"]-model_cores["inception"]

        ins_num = {"inception": {}, "resnet": {}, "mobilenet": {}}
        for model_name in ["inception","resnet","mobilenet"]:
            #print("model_name",each_model_cores)
            model_ins_size = ins_size[model_name]["intra"]
            ins_num[model_name]["ins_num"] = math.floor(model_cores[model_name]/model_ins_size)
        return ins_num

    def get_average_strategy(self,ins_size):
        # 1. decide the instance size
        #self.fairness_factor = fariness_factor
        ins_num = {"inception":{},"resnet":{},"mobilenet":{}}
        each_model_cores = 4 # 3个模型个分4个核
        for model_name in ["inception","resnet","mobilenet"]:
            model_ins_size = ins_size[model_name]["intra"]
            ins_num[model_name]["ins_num"] = math.floor(each_model_cores/model_ins_size)
            #print("============",model_ins_size)
        return ins_num

    def get_strategy_background(self):
        result_ours = { "sys_h": [], "used_cores": [], "sys_users": [],
                           "I_h": [], "R_h": [], "M_h": [], "I_ins_num": [], "R_ins_num": [], "M_ins_num": [],
                           "I_cores": [],"R_cores": [], "M_cores": [], 'I_users': [], "R_users": [],
                        "M_users": [],'I_k': [], "R_k": [], "M_k": [], 'I_E': [], "R_E": [], "M_E": []}
        network_Hw = 86.7
        for i in [0]:  # range(len(I_trace)):
            Handwidth = {"inception": network_Hw*1024*1024/8, "resnet": network_Hw*1024*1024/8, "mobilenet": network_Hw*1024*1024/8}
            for fariness_factor in [1]:  # [0.9]:#np.arange(0.5,1.01,0.1):
                for inception_weight in [0.33]:  # [0.65]:#np.arange(0.01,1.01,0.05):#np.linspace(0.1,1,num=10,endpoint=False):
                    for resnet_weight in [0.33]:  # [0.34]:#np.arange(0.01,1.01,0.05):#np.linspace(0.1,1,num=10,endpoint=False):
                        inception_weight = round(inception_weight, 2)
                        resnet_weight = round(resnet_weight, 2)
                        if inception_weight + resnet_weight >= round(1, 2):
                            continue
                        else:
                            # mobilenet_weight = round(1 - inception_weight - resnet_weight, 2)
                            mobilenet_weight = 0.33
                            self.fairness_factor = round(fariness_factor, 2)
                            ins_size_ours = self.__get_baseline_ins_size_no_inter__(Handwidth)
                            #print(ins_size_ours)

                            ins_num_greedy = self.__get_ins_numHer(ins_size_ours)
                            ins_num_weighted = self.get_weighted_strategy(ins_size_ours,
                                                                          {"inception": inception_weight, "resnet": resnet_weight, "mobilenet": mobilenet_weight})
                            ins_num_average= self.get_average_strategy(ins_size_ours)
                            ins_num_optimal = self.get_optimal_strategy(ins_size_ours)

                            greedy_strategy = {"inception": {}, "resnet": {}, "mobilenet": {}}
                            weighted_strategy = {"inception": {}, "resnet": {}, "mobilenet": {}}
                            average_strategy = {"inception": {}, "resnet": {}, "mobilenet": {}}
                            optimal_strategy = {"inception": {}, "resnet": {}, "mobilenet": {}}

                            for model_name in ["inception", "resnet", "mobilenet"]:
                                greedy_strategy[model_name].update(ins_size_ours[model_name])
                                weighted_strategy[model_name].update(ins_size_ours[model_name])
                                average_strategy[model_name].update(ins_size_ours[model_name])
                                optimal_strategy[model_name].update(ins_size_ours[model_name])

                                greedy_strategy[model_name].update(ins_num_greedy[model_name])
                                weighted_strategy[model_name].update(ins_num_weighted[model_name])
                                average_strategy[model_name].update(ins_num_average[model_name])
                                optimal_strategy[model_name].update(ins_num_optimal[model_name])

                                result_ours[model_name[0].capitalize()+"_k"].append(ins_size_ours[model_name]["k"])
                                result_ours[model_name[0].capitalize() + "_E"].append(ins_size_ours[model_name]["efficiency"])

                                greedy_strategy[model_name].update(ins_num_greedy[model_name])
                                weighted_strategy[model_name].update(ins_num_weighted[model_name])
                                average_strategy[model_name].update(ins_num_average[model_name])
                                optimal_strategy[model_name].update(optimal_strategy[model_name])

                            greedy = self.evalaute(greedy_strategy)
                            average = self.evalaute(average_strategy)
                            weighted = self.evalaute(weighted_strategy)
                            optimal = self.evalaute(optimal_strategy)

                            strategy_kind = ["greedy","average","weighted","optimal"]


                            for item_names in greedy.keys():
                                result_ours[item_names].append(greedy[item_names])
                                result_ours[item_names].append(average[item_names])
                                result_ours[item_names].append(weighted[item_names])
                                result_ours[item_names].append(optimal[item_names])
                            result_pd = pd.DataFrame(data = result_ours,index = strategy_kind)
                            result_pd.to_excel("experiment/Hackground/Hackground_compare_resource_allocation_no_inter.xlsx")

    def get_throughput_by_partition_plan(self,k,intra,network):
        """
        this function is used to prove that more small models result in a Higger throughput.
        :param k:
        :param intra:
        :return:
        """
        model_name = "inception"
        edge_latency = self.get_complete_latency(model_name, intra) * self.get_latency_ratio(model_name, intra, k)
        edge_throughput = 1/edge_latency
        e2e_latency = round(self.model_info.get_layer_size_by_index(model_name,k)/network+1/edge_throughput,3)
        result = {"edge_throughput":edge_throughput,"e2e_latency":e2e_latency}
        return result

    def get_sys_utility(self,strategy,name = None):
        sys_utility = 0
        for model_name in ["inception","mobilenet","resnet"]:
            if strategy[model_name]["intra"]>0:
                sys_utility = sys_utility + strategy[model_name]["efficiency"]*strategy[model_name]["ins_num"]*strategy[model_name]["intra"]
        return sys_utility

    def get_strategy_search_param_baseline(self,partition,item):
        SLA_list = [0.9]
        I_W = [0.38]
        R_W = [0.38]
        F_list = [0.45]
        if item == "sim_network":
            wifi_trace = pd.read_excel("./experiment/wifi/experiment_wifi.xlsx", sheet_name="model_partition",index_col=0)
            I_trace = wifi_trace["trace0"].values
            R_trace = wifi_trace["trace0"].values
            M_trace = wifi_trace["trace0"].values
            #print(I_trace)
        elif item == "real_network":
            wifi_trace = pd.read_excel("./experiment/wifi/experiment_wifi.xlsx",index_col=0,sheet_name="experiment")
            #wifi_trace = wifi_trace.dropna()
            I_trace = wifi_trace["trace1"].values#[wifi_trace["trace1"].values[12]]#[12:14]
            R_trace = wifi_trace["trace2"].values
            M_trace = wifi_trace["trace3"].values
            #print(I_trace,R_trace,)
        else:
            I_trace = [92.525]
            R_trace = [91.9]
            M_trace = [90.8]

        if item == "fairness":
            F_list = np.arange(0.34,1.01,0.03)
        if item == "model_weight":
            I_W = np.arange(0.1,1,0.02)
            R_W = np.arange(0.1,1,0.02)
        if item == "SLA":
            SLA_list = np.arange(0.1,1.01,0.05)

        for strategy_kind in ["weighted"]:#,"average","optimal","weighted","optimal",,"weighted"
            result_baseline = {"sys_u":[],"I_net":[],"R_net":[],"M_net":[],"I_W": [], "R_W": [], "M_W": [], "sys_h": [], "used_cores": [], "sys_users": [],
                            "I_u": [], "R_u": [], "M_u": [], "I_intra":[],"R_intra":[],"M_intra":[],
                           "I_ins_num": [], "R_ins_num": [], "M_ins_num": [], "I_cores": [],
                            "R_cores": [], "M_cores": [], 'I_users': [], "R_users": [], "M_users": [],
                            "F": [],'I_k': [], "R_k": [], "M_k": [],'I_E': [], "R_E": [], "M_E": [],"meet_fairness":[]}
            result_greedy = {"sys_u":[],"I_net":[],"R_net":[],"M_net":[],"I_W": [], "R_W": [], "M_W": [], "sys_h": [], "used_cores": [], "sys_users": [],
                            "I_u": [], "R_u": [], "M_u": [], "I_intra":[],"R_intra":[],"M_intra":[],
                           "I_ins_num": [], "R_ins_num": [], "M_ins_num": [], "I_cores": [],
                            "R_cores": [], "M_cores": [], 'I_users': [], "R_users": [], "M_users": [],
                            "F": [],'I_k': [], "R_k": [], "M_k": [],'I_E': [], "R_E": [], "M_E": [],"meet_fairness":[]}
            result_weighted = {"sys_u":[],"I_net":[],"R_net":[],"M_net":[],"I_W": [], "R_W": [], "M_W": [], "sys_h": [], "used_cores": [], "sys_users": [],
                            "I_u": [], "R_u": [], "M_u": [], "I_intra":[],"R_intra":[],"M_intra":[],
                           "I_ins_num": [], "R_ins_num": [], "M_ins_num": [], "I_cores": [],
                            "R_cores": [], "M_cores": [], 'I_users': [], "R_users": [], "M_users": [],
                            "F": [],'I_k': [], "R_k": [], "M_k": [],'I_E': [], "R_E": [], "M_E": [],"meet_fairness":[]}
            result_ours = {"sys_u":[],"I_net":[],"R_net":[],"M_net":[],"I_W": [], "R_W": [], "M_W": [], "sys_h": [], "used_cores": [], "sys_users": [],
                            "I_u": [], "R_u": [], "M_u": [], "I_intra":[],"R_intra":[],"M_intra":[],
                           "I_ins_num": [], "R_ins_num": [], "M_ins_num": [], "I_cores": [],
                            "R_cores": [], "M_cores": [], 'I_users': [], "R_users": [], "M_users": [],
                            "F": [],'I_k': [], "R_k": [], "M_k": [],'I_E': [], "R_E": [], "M_E": [],"meet_fairness":[],'SLA': []}
            for SLA_factor in SLA_list:
                for fairness_factor in F_list:
                    for inception_weight in I_W:
                        for resnet_weight in R_W:
                            inception_weight = round(inception_weight, 2)
                            resnet_weight = round(resnet_weight, 2)
                            self.fairness_factor = round(fairness_factor, 2)
                            if inception_weight + resnet_weight >= round(1, 2):
                                continue
                            else:
                                mobilenet_weight = round(1 - resnet_weight - inception_weight, 2)
                                # mobilenet_weight = 0.33
                            fariness_factor = round(fairness_factor, 2)
                            for i in range(len(I_trace)):#ins_size_ours
                                #print(i,"==============")
                                inception_network = I_trace[i] * 1024 * 1024 / 8
                                resnet_network = R_trace[i] * 1024 * 1024 / 8
                                mobilenet_network = M_trace[i] * 1024 * 1024 / 8
                                bandwidth = {"inception": inception_network, "resnet": resnet_network,
                                             "mobilenet": mobilenet_network}
                                max_intra = int(min(8,round(self.TOTAL_CPU_Cores * fairness_factor)))
                                if partition=="E":
                                    ins_size_ours = self.get_ins_size_queue(bandwidth,max_intra=max_intra,SLA_factor = SLA_factor)
                                    print(i,ins_size_ours)
                                elif partition=="NS": #NS
                                    ins_size_ours = self.get_neurosurgeon_ins_size_queue(bandwidth,max_intra=max_intra,SLA_factor=SLA_factor)
                                elif partition=="I":
                                    ins_size_ours = self.get_baseline_ins_size_queue(bandwidth, max_intra=max_intra,SLA_factor=SLA_factor)
                                #print(i,ins_size_ours)
                                ins_size_ours["inception"]["efficiency"] = ins_size_ours["inception"]["efficiency"]*inception_weight
                                ins_size_ours["resnet"]["efficiency"] = ins_size_ours["resnet"]["efficiency"] * resnet_weight
                                ins_size_ours["mobilenet"]["efficiency"] = ins_size_ours["mobilenet"]["efficiency"] * mobilenet_weight
                                '''
                                #print(ins_size_ours)
                                ins_num_greedy = greedy.get_greedy_ins_numHer(ins_size_ours, fairness_factor)
                                ins_num_weighted = self.get_weighted_strategy(ins_size_ours, {"inception": inception_weight,
                                                                                          "resnet": resnet_weight,
                                                                                          "mobilenet": mobilenet_weight})
                                greedy_strategy = {"inception": {}, "resnet": {}, "mobilenet": {}}
                                weighted_strategy = {"inception": {}, "resnet": {}, "mobilenet": {}}
                                for model_name in ["inception", "resnet", "mobilenet"]:
                                    greedy_strategy[model_name].update(ins_size_ours[model_name])
                                    greedy_strategy[model_name].update(ins_num_greedy[model_name])
                                    weighted_strategy[model_name].update(ins_size_ours[model_name])
                                    weighted_strategy[model_name].update(ins_num_weighted[model_name])
                                result_greedy["meet_fairness"].append(self.meet_fairness(greedy_strategy, fariness_factor)[0])
                                result_weighted["meet_fairness"].append(self.meet_fairness(weighted_strategy, fariness_factor)[0])
                                
                                greedy = self.evalaute(greedy_strategy)
                                for item_names in greedy.keys():
                                    result_greedy[item_names].append(greedy[item_names])
                                for model_name in ["inception", "resnet", "mobilenet"]:
                                    result_greedy[model_name[0].capitalize() + "_k"].append(ins_size_ours[model_name]["k"])
                                    result_greedy[model_name[0].capitalize() + "_E"].append(round(ins_size_ours[model_name]["efficiency"],3))
                                result_greedy["sys_u"].append(round(self.get_sys_utility(greedy_strategy), 3))
                                
                                greedy_u = round(self.get_sys_utility(greedy_strategy), 3)
                                weighted_u = round(self.get_sys_utility(weighted_strategy), 3)
                                #print(greedy_u)
                                #print(inception_weight,resnet_weight,mobilenet_weight,i,greedy_u,weighted_u,greedy_u/weighted_u)
                                for model_name in self.model_name_list:
                                    print(model_name,ins_size_ours[model_name]["k"],ins_size_ours[model_name]["intra"])
                                print(i, greedy_u, weighted_u,greedy_u /weighted_u)
                                print(self.meet_fairness(greedy_strategy,fariness_factor)[1])
                                print(self.meet_fairness(weighted_strategy,fariness_factor)[1])
                                print()
                                # =======use greedy strategy
                                '''
                                result_ours["I_W"].append(inception_weight)
                                result_ours["R_W"].append(resnet_weight)
                                result_ours["M_W"].append(mobilenet_weight)
                                result_ours["I_net"].append(I_trace[i])
                                result_ours["R_net"].append(R_trace[i])
                                result_ours["M_net"].append(M_trace[i])
                                result_ours["F"].append(fariness_factor)
                                result_ours["SLA"].append(SLA_factor)
                                ins_num_ours = self.get_weighted_strategy(ins_size_ours,{"inception":inception_weight,"resnet":resnet_weight,"mobilenet":mobilenet_weight})
                                ours_strategy = {"inception": {}, "resnet": {}, "mobilenet": {}}
                                for model_name in ["inception", "resnet", "mobilenet"]:
                                    ours_strategy[model_name].update(ins_size_ours[model_name])
                                    ours_strategy[model_name].update(ins_num_ours[model_name])

                                ours = self.evalaute(ours_strategy)
                                for item_names in ours.keys():
                                    result_ours[item_names].append(ours[item_names])
                                for model_name in ["inception", "resnet", "mobilenet"]:
                                    result_ours[model_name[0].capitalize() + "_k"].append(ins_size_ours[model_name]["k"])
                                    result_ours[model_name[0].capitalize() + "_E"].append(round(ins_size_ours[model_name]["efficiency"],3))
                                model_cores = {}
                                for model_name in self.model_name_list:
                                    model_cores[model_name] = ours[model_name[0].capitalize() + "_cores"]
                                fairness_result = self.meet_fairness(model_cores, fariness_factor)
                                result_ours["meet_fairness"].append(fairness_result)
                                if fairness_result:
                                    result_ours["sys_u"].append(round(self.get_sys_utility(ours_strategy), 3))
                                else:
                                    result_ours["sys_u"].append(0)

                                '''
                                result_baseline["I_W"].append(inception_weight)
                                result_baseline["R_W"].append(resnet_weight)
                                result_baseline["M_W"].append(mobilenet_weight)
                                result_baseline["I_net"].append(I_trace[i])
                                result_baseline["R_net"].append(R_trace[i])
                                result_baseline["M_net"].append(M_trace[i])
                                result_baseline["F"].append(fariness_factor)
                                
                                ins_size_baseline = self.__get_baseline_ins_size_queue__(Handwidth)
                                ins_size_baseline["inception"]["efficiency"] = round(ins_size_baseline["inception"]["efficiency"]*inception_weight,3)
                                ins_size_baseline["resnet"]["efficiency"] = round(ins_size_baseline["resnet"]["efficiency"] * resnet_weight,3)
                                ins_size_baseline["mobilenet"]["efficiency"] = round(ins_size_baseline["mobilenet"]["efficiency"] * mobilenet_weight,3)
                                if strategy_kind == "greedy":
                                    ins_num_baseline = self.__get_ins_numHer(ins_size_baseline)
                                elif strategy_kind =="average":
                                    ins_num_baseline = self.get_average_strategy(ins_size_baseline)
                                elif strategy_kind =="weighted":
                                    ins_num_baseline = self.get_weighted_strategy(ins_size_baseline,{"inception":inception_weight,"resnet":resnet_weight,"mobilenet":mobilenet_weight})
                                else:
                                    ins_num_baseline = self.get_optimal_strategy(ins_size_baseline)
                                
                                baseline_strategy = {"inception": {}, "resnet": {}, "mobilenet": {}}
                                for model_name in ["inception", "resnet", "mobilenet"]:
                                    baseline_strategy[model_name].update(ins_size_baseline[model_name])
                                    baseline_strategy[model_name].update(ins_num_baseline[model_name])
    
                                for model_name in ["inception", "resnet", "mobilenet"]:
                                    result_baseline[model_name[0].capitalize() + "_k"].append(ins_size_baseline[model_name]["k"])
                                    result_baseline[model_name[0].capitalize() + "_E"].append(round(ins_size_baseline[model_name]["efficiency"],3))
                                result_baseline["meet_fairness"].append(self.meet_fairness(baseline_strategy, fariness_factor)[0])
    
                                baseline = self.evalaute(baseline_strategy)
                                for item_names in baseline.keys():
                                    result_baseline[item_names].append(baseline[item_names])
                                result_baseline["sys_u"].append(round(self.get_sys_utility(baseline_strategy), 3))
                                '''
                            #print(fairness_result)
                            print("=============finish============",strategy_kind,fariness_factor,inception_weight,resnet_weight)
            file_path = "experiment/resource_allocation/"+item
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            #baseline_pd = pd.DataFrame(data=result_baseline, index=range(len(result_ours["I_W"])))
            #baseline_pd.to_excel(file_path+"/baseline_"+strategy_kind+".xlsx")
            #baseline_pd.to_excel(file_path + "/baseline_partition_interference.xlsx")
            #print(result_ours)
            #ours_pd = pd.DataFrame(data=result_ours, index=range(len(result_ours["I_W"])))
            #ours_pd.to_excel(file_path+"/weight_"+partition+"_"+item+".xlsx")

            #ours_pd.to_excel(file_path + "/search/hitdl_" + strategy_kind + "_modified_75_wifi.xlsx")

    def find_model_feasible_partition_points(self):
        opt = ModelOptimizor()
        # opt.find_feasible_partition_points()
        #wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx", sheet_name="model_partition")
        wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx",index_col=0,sheet_name="experiment")
        #wifi_trace = wifi_trace.dropna()
        model_trace = {"inception":wifi_trace["trace1"].values,"resnet":wifi_trace["trace2"].values,"mobilenet":wifi_trace["trace3"].values}
        #model_trace = {"inception": wifi_trace["trace0"].values, "resnet": wifi_trace["trace0"].values,
        #               "mobilenet": wifi_trace["trace0"].values}
        #print("++++++++++++++--------------",model_trace)
        print(model_trace["mobilenet"])
        item = "50_slots"
        feasible_writer = pd.ExcelWriter("experiment/model_partition/hitdl_feasible_plans/feasible_plans_"+item+".xlsx")
        for model_name in self.model_name_list:
            result = {"k": [], "intra": [], "efficiency": [], "user_num_per_ins": [],"net":[],"F":[],"SLA":[],"net_index":[]}
            #result_select = {"k": [], "intra": [], "efficiency": [], "user_num_per_ins": [],"net":[]}
            for fairness in [0.45]:#np.arange(0.1,1.01,0.05):
                for SLA_factor in [0.9]:#np.arange(0.4, 1.01, 0.01):
                    for i in range(len(model_trace[model_name])):
                        print("----------------",i)
                        bandwdith = model_trace[model_name][i] * 1024 * 1024 / 8
                        SLA_factor = round(SLA_factor,2)
                        fairness = round(fairness,2)
                        plans= opt.get_feasible_plan_by_name(model_name, bandwdith,SLA_factor,fairness)
                        print()
                        #print(i, "model name=========", model_name, plans)
                        '''
                        result_select["k"].append(select["k"])
                        result_select["intra"].append(select["intra"])
                        result_select["efficiency"].append(select["efficiency"])
                        result_select["user_num_per_ins"].append(select["user_num_per_ins"])
                        result_select["net"].append(model_trace[model_name][i])
                        '''
                        
                        for plan in plans:
                            result["k"].append(plan[0])
                            result["intra"].append(plan[1])
                            result["efficiency"].append(plan[2])
                            result["user_num_per_ins"].append(plan[3])
                            result["net"].append(model_trace[model_name][i])
                            result["F"].append(fairness)
                            result["SLA"].append(SLA_factor)
                            result["net_index"].append(i)

            result_pd = pd.DataFrame(data=result, index=range(len(result["k"])))
            result_pd.to_excel(feasible_writer, sheet_name=model_name)
        feasible_writer.save()
        feasible_writer.close()

    def get_partition_e2e_under_fix_intra_no_queue(self,model_name,bandwidth):
        intra = 8
        mobile_latency_list = self.model_info.get_mobile_latency(model_name)
        layer_size_list = self.model_info.get_layer_size(model_name)  # the unit if byte
        partition_e2e = []
        for k in range(self.layer_nums[model_name]-1):  # can not select the last prediction layer
            # 1.1 evaluate the sum of the upload latency and the mobile latency
            mobile_latency = mobile_latency_list[k] * self.model_mobile_interfer_factor[model_name]
            if k<self.layer_nums[model_name]-2:
                layer_size = layer_size_list[k]
                upload_latency = layer_size / bandwidth
            else:
                upload_latency = 0
            e2e = mobile_latency+upload_latency
            if k==0:
                edge_latency = self.get_edge_latency(model_name, intra, 0)
                efficienty = round(1.0 / edge_latency / min(self.model_frame_rate[model_name], 1.0 / upload_latency) / intra, 3)
            elif k<self.layer_nums[model_name]-2:
                edge_latency = self.get_edge_latency(model_name, intra, k+1)
                efficienty = round(1.0 / edge_latency / min(self.model_frame_rate[model_name], 1.0 / upload_latency) / intra, 3)
            else:
                edge_latency = 0
                efficienty = 0
            e2e = round(e2e + edge_latency,3)
            partition_e2e.append(e2e)

        # Step 2: select the most efficient partition plan for $model_name
        select_plan = None
        min_e2e = None
        i = 0
        for e2e in partition_e2e:
            if min_e2e == None or e2e< min_e2e:
                select_plan = i
                min_e2e = e2e
            i = i+1
        return partition_e2e,select_plan,min_e2e

    def get_all_feasbile_ins_size_mckp(self, bandwidth, weight, fairness_factor,SLA_factor=None):
        ins_size = {}
        model_feasible_plans = {}
        for model_name in self.model_name_list:
            intra_list = []
            efficiency_list = []
            feasible_plans = self.get_feasible_plan_by_name(model_name, bandwidth[model_name],
                                                            SLA_factor=SLA_factor, fairness=fairness_factor)
            print(model_name,feasible_plans)
            print()

            model_feasible_plans[model_name] = feasible_plans
            for plan in feasible_plans:
                intra_list.append(plan[1])  # (k, intra, efficienty,user_num_per_ins)
                efficiency_list.append(plan[2] * weight[model_name])
            if len(intra_list)>0:
                ins_size[model_name] = {"intra": intra_list, "efficiency": efficiency_list}
            else:
                ins_size[model_name] = {"intra": [0], "efficiency": [0]}
                model_feasible_plans[model_name] = [(-1, 0, 0,0)]

        return ins_size, model_feasible_plans

    def get_ins_size_mckp(self,partition,bandwidth, fairness_factor,SLA_factor):
        max_intra = int(min(8,round(self.TOTAL_CPU_Cores * fairness_factor)))
        if partition=="E":
            ins_size_org = self.get_ins_size_queue(bandwidth, max_intra=max_intra,SLA_factor=SLA_factor)
        elif partition=="NS":
            ins_size_org = self.get_neurosurgeon_ins_size_queue(bandwidth, max_intra=max_intra,SLA_factor=SLA_factor)
        elif partition=="I":
            ins_size_org = self.get_baseline_ins_size_queue(bandwidth, max_intra=max_intra,SLA_factor=SLA_factor)
        ins_size = {}
        model_feasible_plans = {}  # (k, intra, efficienty,user_num_per_ins)
        for model_name in self.model_name_list:
            content = ins_size_org[model_name]
            model_feasible_plans[model_name] = [(content["k"], ins_size_org[model_name]["intra"], content["efficiency"], content["user_num_per_ins"])]
            ins_size[model_name] = {"intra":[int(max(0,ins_size_org[model_name]["intra"]))],
                                    "efficiency": [ins_size_org[model_name]["efficiency"]]}

        return ins_size, model_feasible_plans

    def get_strategy_search_param_hitdl(self, partition, item, result_ours=None):
        SLA_list = [0.9]
        I_W = [0.38]
        R_W = [0.38]
        F_list = [0.45]

        if item == "sim_network":
            wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx", sheet_name="model_partition",index_col=0)
            #wifi_trace = wifi_trace.dropna()
            I_trace = wifi_trace["trace0"].values
            R_trace = wifi_trace["trace0"].values
            M_trace = wifi_trace["trace0"].values
        elif item == "real_network":
            wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx",sheet_name="experiment",index_col=0)
            I_trace = [wifi_trace["trace1"].values[0]]
            R_trace = [wifi_trace["trace2"].values[0]]
            M_trace = [wifi_trace["trace3"].values[0]]
        else:
            I_trace = [92.525]
            R_trace = [91.9]
            M_trace = [90.8]

        if item == "fairness":
            F_list = np.arange(0.34,1.01,0.03)
        if item == "model_weight":
            I_W = np.arange(0.1,1,0.02)
            R_W = np.arange(0.1,1,0.02)
        if item == "SLA":
            SLA_list = np.arange(0.1,1.01,0.05)

        mckp = MCKPAllocation(self.TOTAL_CPU_Cores,self.model_name_list,self.fairness_factor)
        time = datetime.datetime.now().strftime("%H_%M_%S")
        file_path = "experiment/resource_allocation/"+item
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        with open(file_path+"/mckp_"+time+"_"+partition+"_"+item+".txt","a") as f:
                print(file_path+"/mckp_"+time+"_"+partition+"_"+item+".txt","a")
                result_ours = {"sys_u": [], "I_net": [], "R_net": [], "M_net": [], "I_W": [], "R_W": [], "M_W": [],
                               "used_cores": [], "sys_users": [], 'I_users': [], "R_users": [], "M_users": [],
                                 'I_cores': [], "R_cores": [], "M_cores": [],
                               "F": [],"meet_fairness": [],'SLA': []}
                for SLA_factor in SLA_list:
                    for fairness_factor in F_list:#:
                        for inception_weight in I_W:  #np.arange(0.1,1,0.02):#[0.12]:#
                            for resnet_weight in R_W:  # np.arange(0.1,1,0.02):#[0.1]: #[0.46]:  #
                                inception_weight = round(inception_weight, 2)
                                resnet_weight = round(resnet_weight, 2)
                                self.fairness_factor = round(fairness_factor, 2)
                                if inception_weight + resnet_weight >= round(1, 2):
                                    continue
                                else:
                                    mobilenet_weight = round(1 - resnet_weight - inception_weight, 2)
                                    # mobilenet_weight = 0.33
                                fairness_factor = round(fairness_factor, 2)
                                for i in range(len(I_trace)):  # ins_size_ours
                                    #print(i,"==============")
                                    inception_network = I_trace[i] * 1024 * 1024 / 8
                                    resnet_network = R_trace[i] * 1024 * 1024 / 8
                                    mobilenet_network = M_trace[i] * 1024 * 1024 / 8
                                    bandwidth = {"inception": inception_network, "resnet": resnet_network,
                                                 "mobilenet": mobilenet_network}
                                    #ins_size_ours = self.get_ins_size_queue(bandwidth, max_intra=4, SLA_factor=0.9)
                                    weight = {"inception": inception_weight, "resnet": resnet_weight,
                                                 "mobilenet": mobilenet_weight}
                                    if partition == "M":
                                        ins_size, model_feasible_plans = self.get_all_feasbile_ins_size_mckp(bandwidth,weight,fairness_factor,SLA_factor=SLA_factor)
                                    else:
                                        ins_size, model_feasible_plans = self.get_ins_size_mckp(partition,bandwidth, fairness_factor,SLA_factor=SLA_factor)
                                    result_ours["I_W"].append(inception_weight)
                                    result_ours["R_W"].append(resnet_weight)
                                    result_ours["M_W"].append(mobilenet_weight)
                                    result_ours["I_net"].append(I_trace[i])
                                    result_ours["R_net"].append(R_trace[i])
                                    result_ours["M_net"].append(M_trace[i])
                                    result_ours["F"].append(fairness_factor)
                                    result_ours["SLA"].append(SLA_factor)
                                    #print(ins_size)
                                    ins_num_ours_dict = {}
                                    if ins_size["inception"]["intra"][0] ==0 and ins_size["resnet"]["intra"][0] ==0 and ins_size["mobilenet"]["intra"][0] ==0:
                                        ''' there is no feasible plans for any models'''
                                        result_ours["used_cores"].append(0)
                                        result_ours["sys_users"].append(0)
                                        result_ours["sys_u"].append(-1)
                                        result_ours["I_users"].append(0)
                                        result_ours["R_users"].append(0)
                                        result_ours["M_users"].append(0)
                                        result_ours["I_cores"].append(0)
                                        result_ours["R_cores"].append(0)
                                        result_ours["M_cores"].append(0)
                                        result_ours["meet_fairness"].append(False)
                                        for model_name in self.model_name_list:
                                            ins_num_ours_dict[model_name] = [{"intra": -1, "k": -1 * self.layer_nums[model_name], "efficiency": -1,"ins_num": 0, "U": 0}]
                                    else:
                                        print("ins_size",ins_size)
                                        ins_num_ours_dict = mckp.resource_allocation(ins_size,fairness_factor)
                                        print("ins_num_ours_dict",ins_num_ours_dict)
                                        used_cores = 0
                                        sys_users = 0
                                        I_users = 0
                                        R_users = 0
                                        M_users = 0
                                        I_cores = 0
                                        R_cores = 0
                                        M_cores = 0
                                        sys_u = 0
                                        for model_name in ins_num_ours_dict.keys():
                                                k = 0
                                                #print(model_feasible_plans)
                                                for plan in ins_num_ours_dict[model_name]:
                                                    #print("===========",plan)
                                                    ins_num = plan["ins_num"]
                                                    select_ins_size = model_feasible_plans[model_name][plan["plan_index"]]
                                                    E = round(select_ins_size[2] * weight[model_name], 2)
                                                    utility = E * select_ins_size[1] * ins_num
                                                    ins_size_ours = {"intra":select_ins_size[1],"k":select_ins_size[0],
                                                                     "efficiency":E,"user_num_per_ins":math.floor(select_ins_size[3]),
                                                                     "ins_num":ins_num,"U":round(utility,2)}
                                                    used_cores = ins_num * select_ins_size[1]+used_cores

                                                    sys_users = sys_users+ins_num *ins_size_ours["user_num_per_ins"]
                                                    utility = ins_size_ours["efficiency"]*ins_size_ours["intra"]*ins_num
                                                    sys_u = sys_u + utility
                                                    if model_name == "inception":
                                                        I_users = ins_num * ins_size_ours["user_num_per_ins"]+I_users
                                                        I_cores = I_cores + ins_num * select_ins_size[1]
                                                    elif model_name == "resnet":
                                                        R_users = ins_num * ins_size_ours["user_num_per_ins"] + R_users
                                                        R_cores = R_cores + ins_num * select_ins_size[1]
                                                    else:
                                                        M_users = ins_num * ins_size_ours["user_num_per_ins"] + M_users
                                                        M_cores = M_cores + ins_num * select_ins_size[1]
                                                    #==========update the model info=============
                                                    #ins_num_ours_dict
                                                    ins_num_ours_dict[model_name][k]=ins_size_ours
                                                    k = k+1

                                        #print(sys_u)
                                        result_ours["used_cores"].append(used_cores)
                                        result_ours["sys_users"].append(sys_users)
                                        result_ours["sys_u"].append(sys_u)
                                        result_ours["I_users"].append(I_users)
                                        result_ours["R_users"].append(R_users)
                                        result_ours["M_users"].append(M_users)
                                        result_ours["I_cores"].append(I_cores)
                                        result_ours["R_cores"].append(R_cores)
                                        result_ours["M_cores"].append(M_cores)
                                        result_ours["meet_fairness"].append(self.meet_fairness({"inception":I_cores,"resnet":R_cores,"mobilenet":M_cores},self.fairness_factor))
                                    plan_str = "#" + str(ins_num_ours_dict) + "#"
                                    param_str = '&{"F":%.2f,I_W":%.2f,"R_W":%.2f,"M_W":%.2f,"I_net":%.3f,"R_net":%.3f,"M_net":%.3f}&' % \
                                                (fairness_factor, inception_weight, resnet_weight, mobilenet_weight,
                                                 I_trace[i], R_trace[i], M_trace[i])
                                    #f.write(param_str + "\n")
                                    #f.write(plan_str + "\n")
                                    #break
                            #break
                        #break

                                print("==========finish=======",fairness_factor,inception_weight,resnet_weight)
                    # baseline_pd = pd.DataFrame(data=result_baseline, index=range(len(result_ours["I_W"])))
                    # baseline_pd.to_excel(file_path+"/baseline_"+strategy_kind+".xlsx")
                    # baseline_pd.to_excel(file_path + "/baseline_partition_interference.xlsx")
                #ours_pd = pd.DataFrame(data=result_ours, index=range(len(result_ours["I_W"])))
                #ours_pd.to_excel(file_path+"/mckp_"+partition+"_"+item+".xlsx")
                #ours_pd.to_excel(file_path + "/search/hitdl_" + strategy_kind + "_modified_75_wifi.xlsx")
    def feasible_partition_plans_stats(self):
        writer = pd.ExcelWriter("experiment/model_partition/hitdl_feasible_plans/feasible_plans_50_slots_stats.xlsx")
        hybrid_model_weight = {"inception":0.38,"resnet":0.38,"mobilenet":0.24}
        for model_name in self.model_name_list:
            file_data = pd.read_excel("experiment/model_partition/hitdl_feasible_plans/feasible_plans_50_slots.xlsx",
                                      sheet_name=model_name,index_col=0)
            max_E = []
            min_E = []
            counts = []
            input_E = {}
            max_E_intra = []
            input_intra = {}
            k = -1
            #print(file_data["net_index"].values)
            row_index = 0
            flag = True
            for j in file_data["net_index"].values:
                if file_data.iloc[row_index,0] == 0:
                    input_E[j]= file_data.iloc[row_index,2]
                    input_intra[j]= file_data.iloc[row_index, 1]
                if j==k:
                    print(round(file_data.iloc[row_index,2],3),max_E[k])
                    if round(file_data.iloc[row_index,2],3)>max_E[k]:
                        max_E[k] = round(file_data.iloc[row_index,2],3)
                        max_E_intra[k] = file_data.iloc[row_index, 1]
                    if round(file_data.iloc[row_index,2],3)<min_E[k]:
                        min_E[k] = round(file_data.iloc[row_index, 2], 3)
                    counts[k] = counts[k]+1
                else:
                    k = j
                    max_E.append(file_data.iloc[row_index,2])
                    max_E_intra.append(file_data.iloc[row_index,1])
                    min_E.append(file_data.iloc[row_index,2])
                    counts.append(1)
                row_index = row_index + 1
            #print(input_E,input_E.values())
            #print(input_intra,input_intra.values())
            input_E = list(input_E.values())
            input_intra = list(input_intra.values())
            max_E = np.around(np.array(max_E)*hybrid_model_weight[model_name],3)
            min_E = np.around(np.array(min_E)*hybrid_model_weight[model_name], 3)
            input_E = np.around(np.array(input_E)* hybrid_model_weight[model_name], 3)
            #print(len(counts),max_E.shape,min_E.shape,input_E.shape,len(max_E_intra),len(input_intra))
            result_pd = pd.DataFrame(data = {"counts":counts,"max_E":max_E,"min_E":min_E,"input_E":input_E,
                                             "max_E_intra":max_E_intra,"input_intra":input_intra},index=range(len(min_E)))
            result_pd.to_excel(writer,sheet_name=model_name)
        writer.save()
        writer.close()

def process_data():
    data = pd.read_excel("experiment/resource_allocation/search/hitdl_weighted.xlsx", index_col=0)
    result = {}
    ratio = data["sys_u"] / data["sys_u(M)"]
    result["W/G"] = ratio
    result.update(data)
    result_pd = pd.DataFrame(result)
    result_pd.to_excel("experiment/resource_allocation/model_weight/hitdl_weighted_all.xlsx")

mo = ModelOptimizor()
item = "real_network"
for partition in ["E"]:#
    #mo.get_strategy_search_param_baseline(partition,item)
    pass
result = mo.find_model_feasible_partition_points()
#print(result)
mo.feasible_partition_plans_stats()
