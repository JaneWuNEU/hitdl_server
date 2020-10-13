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
class MCKPAllocation_passed:
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
        #print("ins_num_upper",ins_num_upper)
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
class MCKPAllocation:
    def __init__(self,CPU_Cores,model_name_list,F):
        self.CPU_Cores = CPU_Cores
        self.model_name_list = model_name_list
    def cpu_const(self,ins_size):
        total_plans = 0
        for model_name in ins_size.keys():
            #total_plans = total_plans + len(ins_size["inception"]["intra"])+len(ins_size["resnet"]["intra"])+len(ins_size["mobilenet"]["intra"])
            total_plans = total_plans + len(ins_size[model_name]["intra"])
        #print("+++++++++=ins size",ins_size,"total plans",total_plans)
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
        #print("ins_num_value",ins_num_value)
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
        self.model_weights = {"inception":0.38,"resnet":0.38,"mobilenet":0.24}
        # 5. load the configuration info of the system
        self.TOTAL_CPU_Cores = 12
        self.layer_nums = {"inception":20,"resnet":21,"mobilenet":16}

        self.model_mobile_time_no_inter = {"inception":0.166,"resnet":0.145,"mobilenet":0.164}
        self.model_mobile_time = {"inception":0.221,"resnet":0.193,"mobilenet":0.164}
        self.model_mobile_interfer_factor = {"resnet": 1.331, "inception": 1.3313, "mobilenet": 1}


        self.SLA_factor = 0.9
        '''
        self.model_SLA = {}
        self.model_SLA_no_mobile_inter = {}
        for model_name in self.model_name_list:
            self.model_SLA[model_name] = self.model_mobile_time[model_name]*self.model_SLA_factor
            self.model_SLA_no_mobile_inter[model_name] = self.model_mobile_time_no_inter[model_name] * self.model_SLA_factor
        '''

        self.model_info = ModelInfo()
        self.fairness_factor = 0.45
        self.mckp = MCKPAllocation(self.TOTAL_CPU_Cores, self.model_name_list, self.fairness_factor)
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

    def get_feasible_plan_by_name(self, model_name, bandwidth, SLA_factor=None):
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
            SLA = round(self.model_mobile_time[model_name] * 0.9, 3)
        else:
            SLA = round(self.model_mobile_time[model_name] * SLA_factor, 3)


        CPU_upper = int(min(round(self.TOTAL_CPU_Cores * self.fairness_factor), 8))
        # step 1: find out all the feasible partition plans for #model_name
        feasible_partition_plan = []
        for k in range(self.layer_nums[model_name] - 1):  # can not select the last prediction layer
            # 1.1 evaluate the sum of the upload latency and the mobile latency
            layer_size = layer_size_list[k]
            mobile_latency = mobile_latency_list[k] * self.model_mobile_interfer_factor[model_name]
            upload_latency = layer_size / bandwidth
            e2e = mobile_latency + upload_latency

            if round(mobile_latency + upload_latency, 3) > round(SLA, 3):
                continue  # this partition point k is unaHle to meet ths SLA
            else:
                # 2. find out the minimun intra that can make the k a feasible partition piont
                SLA_gap = SLA - (mobile_latency + upload_latency)
                for intra in range(1, CPU_upper + 1):
                    if k == 0:
                        edge_latency = self.get_edge_latency(model_name, intra, 0)
                    else:
                        edge_latency = self.get_edge_latency(model_name, intra, k + 1)
                    queue_len = math.floor(SLA_gap / edge_latency)  # 这里可以乘以一个系数
                    if queue_len > 0:
                        user_num_per_ins = queue_len
                        efficienty = round(
                            1.0 / edge_latency / min(self.model_frame_rate[model_name], 1.0 / upload_latency) / intra,
                            3)
                        feasible_partition_plan.append((k, intra, efficienty, user_num_per_ins))
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
            #print(mobile_latency+upload_latency,round(SLA,3))
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
            ratio = (self.model_latency_ratio[model_name].iloc[0,1:].values)[k]
            #print("$$$$$$$$latency ratio model_name,k",model_name,k)
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
                SLA = self.model_mobile_time[model_name]*self.SLA_factor
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
        print("**********",Handwidth)
        for model_name in ["inception", "resnet", "mobilenet"]:
            Hd = Handwidth[model_name]  # H/s
            mobile_latency_list = self.model_info.get_mobile_latency(model_name)
            layer_size_list = self.model_info.get_layer_size(model_name)  # the unit if byte
            if SLA_factor == None:
                SLA = self.model_mobile_time[model_name]
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

                if round(mobile_latency + upload_latency, 3) > round(SLA, 3):
                    continue  # this partition point k is unaHle to meet ths SLA
                else:
                    # 2. find out the minimun intra that can make the k a feasible partition piont
                    #print(SLA, mobile_latency,upload_latency)
                    SLA_gap = SLA - (mobile_latency + upload_latency)
                    for intra in range(1, max_intra + 1):
                        if k==0:
                            edge_latency = self.get_edge_latency(model_name, intra, 0)
                        elif k<self.layer_nums[model_name]-2:
                            edge_latency = self.get_edge_latency(model_name, intra, k+1)
                        else:
                            edge_latency = 0
                        if k == 7 and model_name == "mobilenet":
                            print("MO intra",intra,edge_latency,SLA_gap,math.floor(SLA_gap / edge_latency),mobile_latency,upload_latency,layer_size,Hd,math.floor(SLA_gap / edge_latency))
                        if edge_latency !=0:

                            queue_len = math.floor(SLA_gap / edge_latency)  # 这里可以乘以一个系数
                            #print("queue_len",queue_len)
                            if queue_len > 0:
                                user_num_per_ins = queue_len
                                e2e = e2e+edge_latency
                                efficienty_list.append([k,intra,round(1.0 / edge_latency/min(self.model_frame_rate[model_name], 1.0 / upload_latency) / intra,2)])

                                feasible_partition_plan.append((k, intra, 1.0 / edge_latency, user_num_per_ins, 1.0 / upload_latency,edge_latency,(queue_len-1)*edge_latency,e2e))
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
                SLA = self.model_mobile_time[model_name]
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

    def meet_fairness(self,model_cores,fairness):
        cores_upper = round(self.TOTAL_CPU_Cores*fairness)
        result = True
        for model_name in self.model_name_list:
            if model_cores[model_name]>cores_upper:
                result = False
                break
        return result

    def get_weighted_ins_num(self,ins_size,model_weights):

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

    def get_optimal_ins_num(self,ins_size):
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

                    if self.meet_fairness(strategy,fairness_factor):
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

    def find_model_feasible_partition_points(self):
        opt = ModelOptimizor()
        # opt.find_feasible_partition_points()
        wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx", sheet_name="Sheet1")
        model_trace = {"inception":wifi_trace["trace1"].values,"resnet":wifi_trace["trace2"].values,"mobilenet":wifi_trace["trace3"].values}
        for model_name in self.model_name_list:
            feasible_writer = pd.ExcelWriter("experiment/model_partition/hitdl_feasible_plans/"+model_name+"_feasible_plans.xlsx")
            result = {"k": [], "intra": [], "efficiency": [], "user_num_per_ins": [],"net":[]}
            result_select = {"k": [], "intra": [], "efficiency": [], "user_num_per_ins": [],"net":[]}

            for i in range(len(model_trace[model_name])):
                Handwdith = model_trace[model_name][i] * 1024 * 1024 / 8
                plans, select = opt.get_feasible_plans(model_name, Handwdith)

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

    def get_hitdl_strategy(self,bandwidth):
        strategy ={"inception":[], "resnet":[], "mobilenet":[]}
        ins_size = {}
        model_feasible_plans = {}
        #========get all the feasible partition plans=========
        for model_name in self.model_name_list:
            intra_list = []
            efficiency_list = []
            feasible_plans = self.get_feasible_plan_by_name(model_name, bandwidth[model_name])
            # print("*****",model_name,feasible_plans)
            model_feasible_plans[model_name] = feasible_plans
            for plan in feasible_plans:
                intra_list.append(plan[1])  # (k, intra, efficienty,user_num_per_ins)
                efficiency_list.append(plan[2] * self.model_weights[model_name])
            ins_size[model_name] = {"intra": intra_list, "efficiency": efficiency_list}

        #======== 2. get the allocation results ==========
        ins_num_ours_dict = self.mckp.resource_allocation(ins_size, self.fairness_factor)
        for model_name in ["inception", "resnet", "mobilenet"]:
            # print(ins_num_ours_dict[model_name])
            for plan in ins_num_ours_dict[model_name]:
                # print("===========",plan)
                ins_num = plan["ins_num"]
                select_ins_size = model_feasible_plans[model_name][plan["plan_index"]]
                E = round(select_ins_size[2] * self.model_weights[model_name], 2)
                utility = E * select_ins_size[1] * ins_num
                ins_size_ours = {"intra": select_ins_size[1], "k": select_ins_size[0],
                                 "user_num_per_ins": math.floor(select_ins_size[3]),"ins_num": ins_num}
                strategy[model_name].append(ins_size_ours)
        return strategy

    def get_weighted_strategy(self,ins_size):
        model_cores = {"inception": 0, "resnet": 0, "mobilenet": 0}
        for model_name in  ["resnet","inception"]:
            model_cores[model_name] = round(self.TOTAL_CPU_Cores*self.model_weights[model_name])
        model_cores["mobilenet"] = self.TOTAL_CPU_Cores-model_cores["resnet"]-model_cores["inception"]

        ins_num = {"inception": {}, "resnet": {}, "mobilenet": {}}
        for model_name in ["inception","resnet","mobilenet"]:
            #print("model_name",each_model_cores)
            model_ins_size = ins_size[model_name]["intra"]
            ins_num[model_name]["ins_num"] = math.floor(model_cores[model_name]/model_ins_size)
        return ins_num

    def get_ins_size_mckp(self,partition,bandwidth ):
        max_intra = int(min(8,round(self.TOTAL_CPU_Cores * self.fairness_factor)))
        if partition=="E":
            ins_size_org = self.get_ins_size_queue(bandwidth, max_intra=max_intra,SLA_factor=self.SLA_factor)
        elif partition=="NS":
            ins_size_org = self.get_neurosurgeon_ins_size_queue(bandwidth, max_intra=max_intra,SLA_factor=self.SLA_factor)
        elif partition=="I":
            ins_size_org = self.get_baseline_ins_size_queue(bandwidth, max_intra=max_intra,SLA_factor=self.SLA_factor)
        ins_size = {}
        model_feasible_plans = {}  # (k, intra, efficienty,user_num_per_ins)
        for model_name in self.model_name_list:
            content = ins_size_org[model_name]
            model_feasible_plans[model_name] = [(content["k"], ins_size_org[model_name]["intra"], content["efficiency"], content["user_num_per_ins"])]
            ins_size[model_name] = {"intra":[int(max(0,ins_size_org[model_name]["intra"]))],
                                    "efficiency": [ins_size_org[model_name]["efficiency"]]}
        return ins_size, model_feasible_plans

    def get_all_feasbile_ins_size_mckp(self, bandwidth):
        ins_size = {}
        model_feasible_plans = {}
        for model_name in self.model_name_list:
            intra_list = []
            efficiency_list = []
            feasible_plans = self.get_feasible_plan_by_name(model_name, bandwidth[model_name],SLA_factor=self.SLA_factor)
            #print(model_name, feasible_plans)
            model_feasible_plans[model_name] = feasible_plans
            for plan in feasible_plans:
                intra_list.append(plan[1])  # (k, intra, efficienty,user_num_per_ins)
                efficiency_list.append(plan[2] * self.model_weights[model_name])
            if len(intra_list)>0:
                ins_size[model_name] = {"intra": intra_list, "efficiency": efficiency_list}
            else:
                ins_size[model_name] = {"intra": [0], "efficiency": [0]}
                model_feasible_plans[model_name] = [(-1, 0, 0,0)]

        return ins_size, model_feasible_plans

    def get_strategy(self,bandwidth,strategy_kind,partition):
       #weight
       #I,E,NS,M
       strategy = {}
       if strategy_kind == "weight":
           #print("==================",strategy_kind)
           max_intra = int(min(8, round(self.TOTAL_CPU_Cores * self.fairness_factor)))
           if partition == "E":
               ins_size = self.get_ins_size_queue(bandwidth, max_intra=max_intra, SLA_factor=self.SLA_factor)
           elif partition == "NS":  # NS
               ins_size = self.get_neurosurgeon_ins_size_queue(bandwidth, max_intra=max_intra,SLA_factor=self.SLA_factor)
           elif partition == "I":
               ins_size = self.get_baseline_ins_size_queue(bandwidth, max_intra=max_intra, SLA_factor=self.SLA_factor)

           ("==========ins size", ins_size)

           ins_size["inception"]["efficiency"] = ins_size["inception"]["efficiency"] * self.model_weights["inception"]
           ins_size["resnet"]["efficiency"] = ins_size["resnet"]["efficiency"] * self.model_weights["resnet"]
           ins_size["mobilenet"]["efficiency"] = ins_size["mobilenet"]["efficiency"]* self.model_weights["mobilenet"]
           ins_num = self.get_weighted_strategy(ins_size)
           for model_name in self.model_name_list:
               strategy[model_name] = [{"intra": ins_size[model_name]["intra"], "k": ins_size[model_name]["k"],
                                        "user_num_per_ins": ins_size[model_name]["user_num_per_ins"],
                                        "ins_num": ins_num[model_name]["ins_num"]}]
       elif strategy_kind =="mckp":
           if partition == "M":
               ins_size, model_feasible_plans = self.get_all_feasbile_ins_size_mckp(bandwidth)
           else:
               ins_size, model_feasible_plans = self.get_ins_size_mckp(partition, bandwidth)
           ins_num_ours_dict = {}


           if ins_size["inception"]["intra"][0] == 0 and ins_size["resnet"]["intra"][0] == 0 and \
                   ins_size["mobilenet"]["intra"][0] == 0:
               for model_name in self.model_name_list:
                   ins_num_ours_dict[model_name] = [{"intra": -1, "k": -1 * self.layer_nums[model_name], "efficiency": -1, "ins_num": 0, "U": 0}]
           else:
               ins_num_ours_dict = self.mckp.resource_allocation(ins_size,self.fairness_factor)
               #print("初始解",ins_num_ours_dict)
               for model_name in ins_num_ours_dict.keys():
                   k = 0
                   for plan in ins_num_ours_dict[model_name]:
                       # print("===========",plan)
                       ins_num = plan["ins_num"]
                       select_ins_size = model_feasible_plans[model_name][plan["plan_index"]]
                       E = round(select_ins_size[2] * self.model_weights[model_name], 2)
                       utility = E * select_ins_size[1] * ins_num
                       ins_size_ours = {"intra": select_ins_size[1], "k": select_ins_size[0],
                                        "user_num_per_ins": math.floor(select_ins_size[3]),
                                        "ins_num": ins_num}
                       ins_num_ours_dict[model_name][k] = ins_size_ours
                       k = k + 1
           strategy = ins_num_ours_dict
       return strategy
''''''

def compare_module_mckp():
    item = "real_network"
    file_path = "./experiment/resource_allocation/real_network"
    opt_files = os.listdir("./experiment/resource_allocation/real_network/module_test")
    remove_ins_files = os.listdir("experiment/resource_allocation/real_network/50_slots")
    for strategy_kind in ["mckp"]:
        for partition in ["M"]:
            opt_file = None
            remove_ins_file = None
            for file in opt_files:
                temp = partition+"_real_network_50_slots.txt"
                if temp in file:
                    opt_file = file_path+"/module_test/"+file
                    break
            for file in remove_ins_files:
                temp = partition+"_real_network_50_slots.txt"
                if temp in file:
                    remove_ins_file = file_path+"/50_slots/"+file
                    break
            opt_lines = None
            remove_ins_lines = None
            print(opt_file)
            print(remove_ins_file)
            with open(opt_file,"r") as f:
                opt_lines = f.readlines()
            with open(remove_ins_file,"r") as f:
                remove_ins_lines = f.readlines()
            result = True
            #print("++++++++++", len(opt_lines),len(remove_ins_lines))
            #if len(opt_lines)!=len(remove_ins_lines):
            #    result = False
            #    break

            for i in range(len(opt_lines)):
                opt_line = str(opt_lines[i])
                remove_ins_line = str(remove_ins_lines[i])
                if opt_line[0]=="#" and opt_line[0:opt_line.index(":")]==remove_ins_line[0:remove_ins_line.index(":")]:
                    opt = eval(opt_line[opt_line.index(":")+1:opt_line.rindex("#")])
                    #print(remove_ins_line,i)
                    rem = eval(remove_ins_line[remove_ins_line.index(":") + 1:remove_ins_line.rindex("#")])
                    # 比较plan的个数
                    if len(opt)!=len(rem):
                        result = False
                        break
                    #比较每个plan的具体内容
                    for j in range(len(opt)):
                        for name in ["intra","k","ins_num","user_num_per_ins"]:
                            if opt[j][name] != rem[j][name]:
                                result = False
                                print(opt_line,remove_ins_line,name)
                                break
            print(partition,result)
def compare_module_weight():
    mo = ModelOptimizor()
    item = "real_network"
    file_path = "./experiment/resource_allocation/real_network"
    opt_files = os.listdir("./experiment/resource_allocation/real_network/module_test")
    remove_ins_files = os.listdir("experiment/resource_allocation/real_network/50_slots")
    for strategy_kind in ["weight"]:
        for partition in ["E","I","NS"]:
            opt_file = None
            remove_ins_file = None
            for file in opt_files:
                temp = partition+"_real_network_50_slots.xlsx"
                if temp in file and strategy_kind in file:
                    opt_file = file_path+"/module_test/"+file
                    break
            for file in remove_ins_files:
                temp = partition+"_real_network_50_slots.xlsx"
                if temp in file and strategy_kind in file:
                    remove_ins_file = file_path+"/50_slots/"+file
                    break
            opt_pd = pd.read_excel(opt_file,index_col=0)
            remove_ins_pd = pd.read_excel(remove_ins_file, index_col=0)
            result = True
            for name in ["intra","k","ins_num"]:
                for model_name in mo.model_name_list:
                    if (opt_pd[model_name[0].capitalize()+"_"+name].values != remove_ins_pd[model_name[0].capitalize()+"_"+name].values).all():
                        result = False
                        print(model_name,name,partition)
                        print(opt_pd[model_name[0].capitalize()+"_"+name].values)
                        print(remove_ins_pd[model_name[0].capitalize()+"_"+name].values)
                        break
            print(partition,result)
def get_opt_file():
    mo = ModelOptimizor()
    #result_ours = {"sys_u": [], "I_net": [], "R_net": [], 'I_users': [], "R_users": [], "M_users": [],
    #               'I_cores': [], "R_cores": [], "M_cores": []}

    wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx", index_col=0, sheet_name="experiment")
    I_trace = wifi_trace["trace1"].values  # [86.9,109.9]#.valuesnp.arange(90,100,1)
    R_trace = wifi_trace["trace2"].values  # [86.9,109.9]#wifi_trace["trace1"].valuesnp.arange(90,100,1)
    M_trace = wifi_trace["trace3"].values  # [86.9,109.9]#np.arange(90,100,1)
    item="real_network"
    for strategy_kind in ["mckp"]:#["mckp"]:
        for partition in ["E"]:#["E","M","I","NS"]:
            time = datetime.datetime.now().strftime("%H_%M_%S")
            file_path = "experiment/resource_allocation/" + item+"/module_test"
            result_weight = {"I_net": [], "R_net": [], "M_net": [], "I_W": [], "R_W": [], "M_W": [], "I_intra": [],
                             "R_intra": [], "M_intra": [],
                             "I_ins_num": [], "R_ins_num": [], "M_ins_num": [], 'I_k': [], "R_k": [], "M_k": []}
            with open(file_path + "/mckp_" + time + "_" + partition +"_"+item+ "_50_slots.txt", "a") as f:
            #if True:
                for i in range(len(I_trace)):
                    print("************", i)
                    inception_network = I_trace[i] * 1024 * 1024 / 8
                    resnet_network = R_trace[i] * 1024 * 1024 / 8
                    mobilenet_network = M_trace[i] * 1024 * 1024 / 8
                    bandwidth = {"inception": inception_network, "resnet": resnet_network,
                                 "mobilenet": mobilenet_network}
                    result = mo.get_strategy(bandwidth,strategy_kind,partition)
                    param_str = '&{"index":%.2f,"F":%.2f,I_W":%.2f,"R_W":%.2f,"M_W":%.2f,"I_net":%.3f,"R_net":%.3f,"M_net":%.3f}&' % \
                                (i, 0.45, 0.38, 0.38, 0.24,I_trace[i], R_trace[i], M_trace[i])
                    result_weight["I_net"].append(I_trace[i])
                    result_weight["R_net"].append(R_trace[i])
                    result_weight["M_net"].append(M_trace[i])
                    result_weight["I_W"].append(0.38)
                    result_weight["R_W"].append(0.38)
                    result_weight["M_W"].append(0.24)
                    for model_name in mo.model_name_list:
                        result_weight[model_name[0].capitalize()+"_intra"] = result[model_name][0]["intra"]
                        result_weight[model_name[0].capitalize() + "_ins_num"] = result[model_name][0]["ins_num"]
                        result_weight[model_name[0].capitalize() + "_k"] = result[model_name][0]["k"]

                    f.write(param_str + "\n")
                    for model_name in mo.model_name_list:
                        plan_str = "#" + model_name + ":" + str(result[model_name]) + "#"
                        f.write(plan_str + "\n")
                    f.write("\n")

            result_pd = pd.DataFrame(data=result_weight, index=range(len(result_weight["I_W"])))
            result_pd.to_excel(file_path + "/"+strategy_kind+"_"+ time + "_" + partition +"_"+item+ "_50_slots.xlsx")
#get_opt_file()
#compare_module_weight()
#mo = ModelOptimizor()
#strategy=mo.get_strategy({"inception":84.4*1024*1024/8,"resnet":93.6*1024*1024/8,"mobilenet":67.900*1024*1024/8},
#                         strategy_kind="mckp",partition="E")
#print(strategy)