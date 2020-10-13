import sys
sys.path.append("./")
import numpy as np
from utils.util import Static_Info,ControlBandwidth,SocketCommunication
from threading import Thread,Timer

from device_manager import BandWidthManager
from model_optimizor import ModelOptimizor
from model_adaptor import ModelAdaptor
import socket
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import signal
import itertools
import gc
gc.disable()
device_manager = BandWidthManager()
model_optimizor = ModelOptimizor()
model_adaptor = ModelAdaptor()
bandwidth_contrl = ControlBandwidth()
sock_tool = SocketCommunication()
import time
strategy_kind = "mckp"
partition ="E"#"E"
def notify_user(strategy):
    def get_plan_index(user_index,plan_start):
        plan_index = 0
        for j in range(len(plan_start)):
            if user_index >= plan_start[j]:
                plan_index = j
                break
        return plan_index

    i = 0
    try:
        # notify stack server to create resnet and inception
        print("+++++++++++++++++",strategy)
        #mobilenet_strategy = strategy.pop("mobilenet")
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        #client.bind(("192.168.1.14",local_port))
        client.settimeout(1)
        client.connect((Static_Info.MOBILE_IP, Static_Info.MOBILE_CONTROLLER_PORT))
        sock_tool.send_data(client,strategy)
    except Exception as e:
        print("error happens when notifying stack server",e)
        client.close()
            #i = i+1

    user_count = 0
    plan_start = []
    for mobilenet_details in  strategy["model_details"]["mobilenet"]:
        plan_start.append(user_count)
        user_count = user_count + mobilenet_details["ins_num"]*mobilenet_details["user_num_per_ins"]
    print("==================mobile user count=============",user_count)
    if strategy["type"] == "activate":
        port_info = list(itertools.chain.from_iterable(strategy["port_details"]["mobilenet"]))
    else:
        port_info = None
    user_index = 0
    for user_ip in Static_Info.MOBILENET_USER_IP:
        plan_index = get_plan_index(user_index,plan_start)

        if user_index<user_count:
            if strategy["type"] =="activate":
                rasp_strategy = {"type":"activate",
                                 "model_details":{"mobilenet":strategy["model_details"]["mobilenet"][plan_index]},
                                 "port_details":{"mobilenet":[port_info[user_index]]},
                                 "bandwidth":{"mobilenet":strategy["bandwidth"]["mobilenet"]}}
            elif strategy["type"] =="create":
                rasp_strategy = {"type":"create",'model_details':{"mobilenet":strategy["model_details"]["mobilenet"][plan_index]}}
            try:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.settimeout(0.05)
                print((user_ip, Static_Info.RASP_CONTROLLER_PORT))
                client.connect((user_ip, Static_Info.RASP_CONTROLLER_PORT))
                sock_tool.send_data(client, rasp_strategy)
            except Exception as e:
                print("error happens when notifying mobilenet",user_ip,e)
        elif user_index>=user_count:
            '''
            send message to tell the rasp users to release all the resource
            '''
            try:
                strategy["type"] = "remove_rasp"
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.settimeout(0.01)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.connect((user_ip, Static_Info.RASP_CONTROLLER_PORT))
                sock_tool.send_data(client, strategy)
            except Exception as e:
                print("error happens when notifying mobilenet",user_ip,e)
        user_index = user_index+1
def update(release_flag_list,model_ins_pid_list,bandwidth):
    """
    update the strategy
    :return: void
    """
    time.sleep(Static_Info.REFRESH_INTERVEL)
    # 0. release the resources.
    if len(release_flag_list)>0:
        i = 0
        for release_flag in release_flag_list:
            ins_port = release_flag
            try:
                #print("#############stop ins_port==========",ins_port)
                os.system('fuser -k -n tcp ' + str(ins_port))

            except Exception as e:
                print("error happens when releasing the recv port",ins_port,e)
            i = i+1

    for i in range(len(model_ins_pid_list)):
        try:
            #print("_))))))))))))))))))))))kill pid_____________",model_ins_pid_list[i][1])
            os.kill(model_ins_pid_list[i][0], signal.SIGKILL)
        except Exception as e:
            print("error happens when releasing kill the process", model_ins_pid_list[i], e)

    device_manager.reset_user_bandwidth()

    # 1. get strategy
    strategy = {}
    #bandwidth = device_manager.get_default_model_bandwidth()

    model_details = model_optimizor.get_strategy({"inception":bandwidth["inception"]*1024*1024/8,
                                                  "resnet": bandwidth["resnet"] * 1024 * 1024 / 8,
                                                  "mobilenet": bandwidth["mobilenet"] * 1024 * 1024 / 8},strategy_kind,partition)
    # 2. notify users of the strategy
    print(model_details)
    strategy.update({"type":'create',"model_details":model_details})
    strategy["bandwidth"] = bandwidth

    notify_user(strategy)

    # control download bandwidth
    #control_download_bandwidth(strategy) #

    # 3. depend on the strategy to deploy models and allocate ports
    port_details,release_flag_list,model_ins_pid_list,model_ins_list = model_adaptor.deploy_model(strategy,device_manager)

    # 4. notify users of the ports
    strategy.update({"type":'activate',"port_details":port_details})
    strategy["bandwidth"] = bandwidth
    notify_user(strategy) # activate users
    return release_flag_list,model_ins_pid_list

if __name__ == '__main__':
    import pandas as pd
    #freeze_support()
    interval = Static_Info.REFRESH_INTERVEL
    run_times = 0
    release_flag_list = []
    model_ins_pid_list = []
    bandwidth = {}
    wifi_trace = pd.read_excel("experiment/wifi/experiment_wifi.xlsx", index_col=0,sheet_name="experiment")
    I_trace = wifi_trace["trace1"].values  # [86.9,109.9]#.valuesnp.arange(90,100,1)
    R_trace = wifi_trace["trace2"].values  # [86.9,109.9]#wifi_trace["trace1"].valuesnp.arange(90,100,1)
    M_trace = wifi_trace["trace3"].values  # [86.9,109.9]#np.arange(90,100,1)
    for i in range(len(I_trace)):
        bandwidth["inception"] = I_trace[i]#wifi_trace[i][0]
        bandwidth["resnet"] = R_trace[i]
        bandwidth["mobilenet"] = M_trace[i]
        try:
            print("++++++++带宽数据索引+++++++++++++",i,strategy_kind,partition)
            print(bandwidth)
            release_flag_list,model_ins_pid_list = update(release_flag_list,model_ins_pid_list,bandwidth)
        except Exception  as e:
            pass
            #print("!!!!!!!!!unexpected exceptions!!!!!!!!!",e)
    #sys.exit(0)
