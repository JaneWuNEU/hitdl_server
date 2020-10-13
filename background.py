import sys
sys.path.append(".")
import numpy as np
import pandas as pd
from model_optimizor_network import ModelOptimizor
import math
def find_worst_allocation(inception_intra,resnet_intra,mobilenet_intra):
    opt = ModelOptimizor()
    sys_max = 0
    sys_min_all = [] #inceptin,resnet,mobilenet
    sys_min = None
    model_sys = []
    for inception_cores in range(inception_intra, 12 + 1):
        for resnet_cores in range(resnet_intra, 12 + 1):
            for mobilenet_cores in range(mobilenet_intra, 12+1):
                # print("*************",inception_num,resnet_num,mobilenet_num)
                if inception_cores + resnet_cores + mobilenet_cores> 12:  # 没有核给mobilenet了
                    continue
                else:
                    pass

                # 评估系统当前的吞吐
                inc_sys = opt.get_edge_throughput_no_inter("inception",inception_cores,0)
                res_sys = opt.get_edge_throughput_no_inter("resnet", inception_cores, 0)
                mobilenet_sys = opt.get_edge_throughput_no_inter("inception", mobilenet_cores,0)
                total = round(inc_sys+res_sys+mobilenet_sys,3)
                if sys_min == None:
                    sys_min =total
                if total<=sys_min:
                    sys_min = total
                    sys_min_all = [inception_cores,resnet_cores,mobilenet_cores]
                    model_sys = [inc_sys,res_sys,mobilenet_sys]
    return sys_min,sys_min_all,model_sys

def find_best_allocation(inception_intra,resnet_intra,mobilenet_intra):
    opt = ModelOptimizor()
    inception_ins_num_max = math.floor(12 / inception_intra)
    resnet_ins_num_max = math.floor(12 / resnet_intra)
    mobilenet_ins_num_max = math.floor(12 / mobilenet_intra)
    sys_max = 0
    sys_max_all = [] #inceptin,resnet,mobilenet
    model_sys = []
    # print("*************",inception_ins_num_max,resnet_ins_num_max,mobilenet_ins_num_max)
    for inception_num in range(1, inception_ins_num_max + 1):
        for resnet_num in range(1, resnet_ins_num_max + 1):
            for mobilenet_num in range(1, mobilenet_ins_num_max + 1):
                # print("*************",inception_num,resnet_num,mobilenet_num)
                if inception_num * inception_intra + resnet_num * resnet_intra + mobilenet_num * mobilenet_intra > 12:  # 没有核给mobilenet了

                    continue
                else:
                    pass
                inc_sys = opt.get_edge_throughput_no_inter("inception",inception_intra,0)
                res_sys = opt.get_edge_throughput_no_inter("resnet", resnet_intra, 0)
                mobilenet_sys = opt.get_edge_throughput_no_inter("inception", mobilenet_intra,0)
                total = round(inc_sys*inception_num+res_sys*resnet_num+mobilenet_sys*mobilenet_num,3)
                if total>sys_max:
                    sys_max = total
                    sys_max_all = [inception_intra*inception_num,resnet_intra*resnet_num,mobilenet_intra*mobilenet_num]
                    model_sys = [inc_sys*inception_num,res_sys*resnet_num,mobilenet_sys*mobilenet_num]
    return sys_max,sys_max_all,model_sys



def background_resource_allocation():
    wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx", index_col=0)
    strategy_kind = "average"  # "weighted","optimal","average",greedy
    # for model_name in ["inception", "resnet", "mobilenet"]:
    I_trace = wifi_trace["trace3"].values[40:50]
    opt = ModelOptimizor
    # 2. 求出三个模型的base size
    for i in range(1):
        bd = I_trace[i]*1024*1024/8
        bandwidth = {"inception":bd,"mobilenet":bd,"resnet":bd}
        inception_intra = 2
        resnet_intra = 2
        mobilenet_intra = 1
        sys_min,min_all,sys_w = find_worst_allocation(inception_intra,resnet_intra,mobilenet_intra)
        sys_max, max_all,sys_b = find_best_allocation(inception_intra, resnet_intra, mobilenet_intra)
        print(sys_max,sys_min,sys_max/sys_min)
        print(bd)
        if sys_max/sys_min>1:
            print(min_all)
            print(max_all)
            print("==================================",i)
            print(sys_w)
            print(sys_b)

background_resource_allocation()

