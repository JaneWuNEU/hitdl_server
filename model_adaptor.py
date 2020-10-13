from utils.util import Static_Info
from model_ins import ModelInstance
from multiprocessing import Process,Value,Manager
import time
import os
manager = Manager()
class ModelAdaptor:

    def deploy_model(self,strategy,device_manager):
        """
        1. according to the $strategy to allocate its cores to different models.
        2. initialize instances, run them in a subprocess ,and bind them to the specific CPU cores.
        3. assign a unique port to each instance, and start a thread to listen to the port
        """
        port_details = {}
        print("--------strategy-------",strategy)
        available_core_id = 0
        release_flag_list = []
        model_ins_pid_list = []
        # 在主进程里创建的局部变量
        model_ins_list = []
        ins_id = 0
        for model_name in ["inception","resnet","mobilenet"]:
            ins_ports = []
            for model_details in strategy['model_details'][model_name]:
                for j in range(model_details["ins_num"]):
                    #for user_index in range(model_details["user_num_per_ins"]):
                    core_id = []
                    # 1.1 find out the available core id for the ins.
                    for i in range(model_details["intra"]):
                        core_id.append(available_core_id)
                        available_core_id = available_core_id + 1
                    # 1.2 create the model instance
                    #print(model_details)
                    model_ins = ModelInstance(model_details["k"],model_details["intra"],model_name,ins_id,None,core_id,model_details["user_num_per_ins"],device_manager,ins_id+1,model_details["ins_num"])
                    # 1.3 start a new process to run the instance
                    #print("%%%%%%%%%%%%%%",model_name,core_id)
                    #release_flag = Value("i",0)
                    release_flag_dict = manager.dict()
                    p = Process(target=model_ins.run_model,args=[release_flag_dict])
                    p.start()
                    model_ins_pid_list.append([p.pid,model_name+"_"+str(ins_id)])
                    ''''''
                    while len(release_flag_dict.keys())!= model_details["user_num_per_ins"]:
                        print("assigh ports has not finished")
                        time.sleep(0.01)

                    ports_info = release_flag_dict.values()
                    ins_ports.append(ports_info)
                    #print("当前实例给实例分配的端口",ports_info)
                    release_flag_list.extend(ports_info)
                    model_ins_list.append(model_ins)
                #print("******模型所有实例下的用户端口********",release_flag_list)
                port_details[model_name] = ins_ports
        return port_details,release_flag_list,model_ins_pid_list,model_ins_list
