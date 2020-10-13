import sys
sys.path.append(".")
import numpy as np
import pandas as pd
#from model_optimizor_networu import ModelOptimizor
import math

class ResourceAllocation:
    def get_weights_meet_fairs(self):
        HG = pd.read_excel("experiment/resource_allocation/modified_utility/ours_greedy.xlsx",index_col=0)
        HO = pd.read_excel("experiment/resource_allocation/modified_utility/ours_optimal.xlsx",index_col=0)
        HW = pd.read_excel("experiment/resource_allocation/modified_utility/ours_weighted.xlsx",index_col=0)
        HO_meet_fair_index = np.argwhere(HO["meet_fairness"].values== True).flatten()
        HW_meet_fair_index = np.argwhere(HW["meet_fairness"].values == True).flatten()
        HG_meet_fair_index = np.argwhere(HG["meet_fairness"].values == True).flatten()
        inter_section = np.intersect1d(HO_meet_fair_index,HW_meet_fair_index)
        trip_intersec = np.intersect1d(HG_meet_fair_index,inter_section) #
        print(trip_intersec)
        result = {"I_net":[],"R_net":[],"M_net":[],"HG_sys_utility":[],"HO_sys_utility":[],"HW_sys_utility":[],"fair":[],
                  "HG_I_cores":[],"HG_R_cores":[],"HG_M_cores":[],
                  "HO_I_cores": [], "HO_R_cores": [], "HO_M_cores": [],
                  "HW_I_cores": [], "HW_R_cores": [], "HW_M_cores": [],
                  "HG_U/HO_U":[],"I_W":[],"R_W":[],"M_W":[],
                  "HG_M_users":[],"HO_M_users":[],"HW_M_users":[],
                  "I_K": [], "R_K": [], "M_K": [],
                  "HW_I_U": [], "HW_R_U": [], "HW_M_U": [],
                  "HG_I_U": [], "HG_R_U": [], "HG_M_U": [],
                  "HO_I_U": [], "HO_R_U": [], "HO_M_U": [],
                  "I_intra": [], "R_intra": [], "M_intra": [],
                  "I_E": [], "R_E": [], "M_E": [],
                  "HG_I_num": [], "HG_R_num": [], "HG_M_num": [],
                  "HO_I_num": [], "HO_R_num": [], "HO_M_num": [],
                  "HW_I_num": [], "HW_R_num": [], "HW_M_num": []
                  }
        for i in trip_intersec:
            result["I_net"].append(HG.loc[i,"I_net"])
            result["R_net"].append(HG.loc[i, "R_net"])
            result["M_net"].append(HG.loc[i, "M_net"])
            result["I_W"].append(HG.loc[i,"I_W"])
            result["R_W"].append(HG.loc[i, "R_W"])
            result["M_W"].append(HG.loc[i, "M_W"])

            result["I_E"].append(HG.loc[i,"I_E"])
            result["R_E"].append(HG.loc[i, "R_E"])
            result["M_E"].append(HG.loc[i, "M_E"])

            result["I_K"].append(HG.loc[i,"I_k"])
            result["R_K"].append(HG.loc[i, "R_k"])
            result["M_K"].append(HG.loc[i, "M_k"])
            
            result["I_intra"].append(HG.loc[i,"I_intra"])
            result["R_intra"].append(HG.loc[i, "R_intra"])
            result["M_intra"].append(HG.loc[i, "M_intra"])

            result["HG_sys_utility"].append(HG.loc[i, "sys_u"])
            result["HO_sys_utility"].append(HO.loc[i, "sys_u"])
            result["HW_sys_utility"].append(HW.loc[i, "sys_u"])
            result["fair"].append(HW.loc[i, "F"])
            result["HG_I_cores"].append(HG.loc[i, "I_cores"])
            result["HG_R_cores"].append(HG.loc[i, "R_cores"])
            result["HG_M_cores"].append(HG.loc[i, "M_cores"])
            result["HO_I_cores"].append(HO.loc[i, "I_cores"])
            result["HO_R_cores"].append(HO.loc[i, "R_cores"])
            result["HO_M_cores"].append(HO.loc[i, "M_cores"])
            result["HW_I_cores"].append(HW.loc[i, "I_cores"])
            result["HW_R_cores"].append(HW.loc[i, "R_cores"])
            result["HW_M_cores"].append(HW.loc[i, "M_cores"])
            result["HW_M_users"].append(HW.loc[i, "M_users"])
            result["HG_M_users"].append(HG.loc[i, "M_users"])

            result["HO_I_U"].append(HO.loc[i, "I_u"])
            result["HO_R_U"].append(HO.loc[i, "R_u"])
            result["HO_M_U"].append(HO.loc[i, "M_u"])
            result["HG_I_U"].append(HG.loc[i, "I_u"])
            result["HG_R_U"].append(HG.loc[i, "R_u"])
            result["HG_M_U"].append(HG.loc[i, "M_u"])
            result["HW_I_U"].append(HW.loc[i, "I_u"])
            result["HW_R_U"].append(HW.loc[i, "R_u"])
            result["HW_M_U"].append(HW.loc[i, "M_u"])
            

            result["HO_I_num"].append(HO.loc[i, "I_ins_num"])
            result["HO_R_num"].append(HO.loc[i, "R_ins_num"])
            result["HO_M_num"].append(HO.loc[i, "M_ins_num"])
            result["HG_I_num"].append(HG.loc[i, "I_ins_num"])
            result["HG_R_num"].append(HG.loc[i, "R_ins_num"])
            result["HG_M_num"].append(HG.loc[i, "M_ins_num"])
            result["HW_I_num"].append(HW.loc[i, "I_ins_num"])
            result["HW_R_num"].append(HW.loc[i, "R_ins_num"])
            result["HW_M_num"].append(HW.loc[i, "M_ins_num"])
            

            result["HO_M_users"].append(HO.loc[i, "M_users"])
            result["HG_U/HO_U"].append(round((HG.loc[i, "sys_u"]/HW.loc[i, "sys_u"]),3))
            #breau
        result_pd = pd.DataFrame(data = result,index=trip_intersec)
        result_pd.to_excel("experiment/resource_allocation/modified_utility/all_fairt_strategy_model_info.xlsx")
    def get_efficiency(self):
        weight1_writer = pd.ExcelWriter("experiment/resource_allocation/more_networu/more_info_weight1.xlsx")
        weight2_writer = pd.ExcelWriter("experiment/resource_allocation/more_networu/more_info_weight2.xlsx")
        weight1_index = [53654,53655,53656,53657,53658,53644,53695,53696,53646,53650]
        wegiht2_index = [53714,53715,53716,53717,53718,53704,53755,53756,53706,53710]
        for strategy in ["optimal","weighted","greedy"]:
            print("strategy",strategy)
            result_w1 = {"I_E":[],"R_E":[],"M_E":[],"I_intra":[],"R_intra":[],"M_intra":[]}
            result_w2 = { "I_E": [], "R_E": [], "M_E": [],"I_intra":[],"R_intra":[],"M_intra":[]}
            efficiency_info = pd.read_excel("experiment/resource_allocation/more_networu/ours_param_search_networu_"+strategy+".xlsx",index_col=0)
            for i in weight1_index:
                #result_w1["index"].append(i)
                result_w1["I_E"].append(efficiency_info.loc[i,"I_E"])
                result_w1["R_E"].append(efficiency_info.loc[i, "R_E"])
                result_w1["M_E"].append(efficiency_info.loc[i, "M_E"])
                result_w1["I_intra"].append(efficiency_info.loc[i, "I_intra"])
                result_w1["R_intra"].append(efficiency_info.loc[i, "R_intra"])
                result_w1["M_intra"].append(efficiency_info.loc[i, "M_intra"])
            for j in wegiht2_index:
                #result_w2["index"].append(i)
                result_w2["I_E"].append(efficiency_info.loc[j,"I_E"])
                result_w2["R_E"].append(efficiency_info.loc[j, "R_E"])
                result_w2["M_E"].append(efficiency_info.loc[j, "M_E"])
                result_w2["I_intra"].append(efficiency_info.loc[j, "I_intra"])
                result_w2["R_intra"].append(efficiency_info.loc[j, "R_intra"])
                result_w2["M_intra"].append(efficiency_info.loc[j, "M_intra"])

            result_w1_pd = pd.DataFrame(data = result_w1,index=weight1_index)
            result_w1_pd.to_excel(weight1_writer,sheet_name=strategy)
            result_w2_pd = pd.DataFrame(data = result_w2,index=wegiht2_index)
            result_w2_pd.to_excel(weight2_writer,sheet_name=strategy)
        weight1_writer.save()
        weight1_writer.close()
        weight2_writer.save()
        weight2_writer.close()

def get_hybrid_resource_allocation():
    for strategy in ["greedy","optimal","weighted"]:
        for partition in ["baseline","ours"]:
            file_name = "experiment/weight_search/hybrid_test/wifi_trace/utility_based/" \
                        +strategy+"_allocation/"+partition+"_wifi_"+strategy+"_allocation_modify.xlsx"
            data = pd.read_excel(file_name,index_col=0)
            I_U = data["I_ins_num"]*data["I_E"]
            R_U = data["R_ins_num"]*data["R_E"]
            M_U = data["M_ins_num"]*data["M_E"]
            data["I_U"] = I_U
            data["R_U"] = R_U
            data["M_U"] = M_U
            data["sys_utility"] = I_U+R_U+M_U
            data.to_excel("experiment/weight_search/hybrid_test/wifi_trace/utility_based/" \
                        +strategy+"_allocation/"+partition+"_wifi_"+strategy+"_allocation.xlsx",index=range(10))

def find_feasible_partition_points(self):
    wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx", index_col=0)
    strategy_uind = "average"  # "weighted","optimal","average",greedy
    # for model_name in ["inception", "resnet", "mobilenet"]:
    I_trace = wifi_trace["trace3"].values[40:50]
    opt = ModelOptimizor()
    # 2. 求出三个模型的base size
    result = {"I_K": [], "I_intra": [], "I_E": [],
              "R_K": [], "R_intra": [], "R_E": [],
              "M_K": [], "M_intra": [], "M_E": [],
              "net":[]}
    for i in range(10):

        bd = I_trace[i]*1024*1024/8
        bandwidth = {"inception":bd,"mobilenet":bd,"resnet":bd}
        for model_name in ["inception","resnet","mobilenet"]:
            # 获取baselse能满足SLA且能实现吞吐最大化的intra
            result = opt.get_hitdl_intra_with_max_throughput(bandwidth)

        result["I_K"].append(result["inception"]["u"])
        result["I_intra"].append(result["inception"]["intra"])
        result["I_E"].append(result["inception"]["efficiency"])

        result["R_K"].append(result["resnet"]["u"])
        result["R_intra"].append(result["resnet"]["intra"])
        result["R_E"].append(result["resnet"]["efficiency"])

        result["M_K"].append(result["mobilenet"]["u"])
        result["M_intra"].append(result["mobilenet"]["intra"])
        result["M_E"].append(result["mobilenet"]["efficiency"])

        result["net"] = I_trace[i]
    result_pd = pd.DataFrame(data= result,index = range(len(result["net"])))
    result_pd.to_excel("experiment/model_partition/input_partition.xlsx")

ra = ResourceAllocation()
#ra.get_efficiency()
#get_hybrid_resource_allocation()
ra.get_weights_meet_fairs()