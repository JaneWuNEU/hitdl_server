import os
import sys
import socket
import struct
import numpy as np
import xml.dom.minidom
import time
class ControlBandwidth:
    def reset_bandwidth(self):
        del_root = "tc qdisc del root dev em4"
        create_root = "tc qdisc add dev em4 root handle 1: htb default 10"
        create_root_class = "tc class add dev em4 parent 1: classid 1:10 htb rate 1000mbit"
        self.__excecute__(del_root)
        self.__excecute__(create_root)
        self.__excecute__(create_root_class)
    def change_bandwidth_demo(self,ports_list,total_bd):
        self.reset_bandwidth()
        clsss_id = 20

        print("===================================")
        for dport in ports_list:
            # 1. create a class under the root
            create_class = "tc class add dev em4 parent 1: classid 1:" + str(class_id) + " htb rate " + str(
                total_bd) + "mbit ceil " + str(total_bd) + "mbit"
            create_branch = "tc qdisc add dev em4 parent 1:" + str(class_id) + " handle " + str(
                class_id) + ": sfq perturb 10"
            create_filter = "tc filter add dev em4 protocol ip parent 1: prio 1 u32 match ip dport " + str(
                dport) + " 0xffff flowid 1:" + str(class_id)
            print("create_class",create_class)
            print("create_branch", create_branch)
            print("create_filter", create_filter)
            class_id = class_id + 1
            self.__excecute__(create_class)
            self.__excecute__(create_branch)
            self.__excecute__(create_filter)

    def change_bandwidth(self,edge_notice):
        """
        each model instance has an individual port which has limited bandwidth.
        This bandwidth is calcuated as $MODEL_USER_BANDWIDHT * $user_num_per_ins
        input:
        ports_details: Dict. e.g.{ inception:[], #ports list assigned for each model instance  resnet:[], mobilenet:[] }
        model_details: dict. e.g. {inception:{k:,ins_num: X,user_num_per_ins:Y} resnet:{...},mobilenet:{...}},
        """
        ports_details = edge_notice["port_details"]
        model_details = edge_notice["model_details"]
        self.reset_bandwidth()
        class_id = 20
        for model_name in ["inception","mobilenet","resnet"]:
            ports_list = ports_details[model_name]
            total_bd = Static_Info[model_name.capitalize()+"USER_BANDWIDTH"]*model_details[model_name]["user_num_per_ins"]
            for dport in ports_list:
                # 1. create a class under the root
                create_class = "tc class add dev em4 parent 1: classid 1:"+str(class_id)+" htb rate "+str(total_bd)+"mbit ceil "+str(total_bd)+"mbit"
                create_branch = "tc qdisc add dev em4 parent 1:"+str(class_id)+" handle "+str(class_id)+": sfq perturb 10"
                create_filter = "tc filter add dev em4 protocol ip parent 1: prio 1 u32 match ip dport "+str(dport)+" 0xffff flowid 1:"+str(class_id)
                class_id = class_id+1
                self.__excecute__(create_class)
                self.__excecute__(create_branch)
                self.__excecute__(create_filter)

    def control_download_bandwidth(self):
        """
        rely
        :return:
        """
        pass
    def __excecute__(self,command):
        sudoPassword = "wujing123"
        p = os.system('echo %s|sudo -S %s' % (sudoPassword, command))

class SocketCommunication:
   def recvall(self, conn, n):
      # Helper function to recv n bytes or return None if EOF is hit
      data = b''
      while len(data) < n:
         packet = conn.recv(n - len(data))
         if not packet:
            return None
         data += packet
      return data

   def send_data(self,conn, content):
      # Prefix each message with a 4-byte length (network byte order)

      # Send data to the server
      try:
          content = bytes(str(content),encoding="utf-8")
          msg = struct.pack('>I', len(content)) + content
          conn.send(msg)
      except Exception as e:
          print("error happens when sending data in socket tools",e)

   def recv_data_bytes(self, conn):
       # Receive data from the server
       # Return result(an numpy array)
       try:
           a = time.time()
           resp_len = self.recvall(conn, 4)
           b = time.time()
           if not resp_len:
               return None
           c = time.time()
           resp_len = struct.unpack('>I', resp_len)[0]

           if not resp_len:
               return None
           d = time.time()
           result = self.recvall(conn, resp_len)
           e = time.time()
           #print("读取头部",round(b-a,3),"解析长度",round(d-c,3),"接收数据",round(e-d,4))
           #result = eval(str(result.decode("utf8")))
       except Exception as e:
           print("error happens in socket utill when receiving data as eval(string)", e)
       return result

   def recv_data_str(self,conn):
      # Receive data from the server
      # Return result(an numpy array)
      try:
          resp_len = self.recvall(conn, 4)
          if not resp_len:
             return None
          resp_len = struct.unpack('>I', resp_len)[0]
          if not resp_len:
             return None
          result = self.recvall(conn, resp_len)
          #result = eval(str(result.decode("utf8")))
      except Exception as e:
          print("error happens in socket utill when receiving data as eval(string)",e)
      return result

   def recv_data_str(self,conn):
      # Receive data from the server
      # Return result(an numpy array)
      result =  None
      try:
          a = time.time()
          resp_len = self.recvall(conn, 4)
          if not resp_len:
             return None
          resp_len = struct.unpack('>I', resp_len)[0]
          if not resp_len:
             return None
          a1 = time.time()
          result = self.recvall(conn, resp_len)
          b = time.time()
          result = result.decode("utf8")
          c = time.time()
          #print("recv head",a1-a,"recv_body",b-a1,"decode",c-b,"total",c-a)
      except Exception as e:
          print("error happens in socket utils when receiving data",e)
      return result

class Static_Info:
    EDGE_IP_STACK = "192.168.1.25"
    EDGE_IP_RASP = "192.168.1.16"
    MOBILE_IP="192.168.1.14"
    MOBILE_CONTROLLER_PORT=10980
    RASP_CONTROLLER_PORT=10970
    REFRESH_INTERVEL= 20  # the edge refresh itself every $REFRESH_INTERVEL seconds.
    EDGE_CORE_UPPER = 8
    MAX_QUEUE_LEN = 50

    # ======= model instance receiving port
    RECV_PORT_INTERVEL = 20
    INCEPTION_RECV_PORT_START = 1100
    RESNET_RECV_PORT_START = 2100
    MOBILENET_RECV_PORT_START = 3100
    CYCLE=5
    MOBILENET_USER_IP =["192.168.1.36","192.168.1.35","192.168.1.34", "192.168.1.31", "192.168.1.32",
                        "192.168.1.30","192.168.1.37","192.168.1.38","192.168.1.40","192.168.1.20"]

    """["192.168.1.35","192.168.1.36","192.168.1.34","192.168.1.31"
        , "192.168.1.32", "192.168.1.30","192.168.1.37","192.168.1.38"]#,"192.168.1.35","192.168.1.36""192.168.1.34","192.168.1.34","192.168.1.34"
    #["192.168.1.32","192.168.1.31","192.168.1.30"]
    """
class ModelInfo():

    def get_layer_name_by_index(self,model_name,layer_index):
        """
        :param model_name:
        :param layer_index: [0,layer_num]
        layer_index==0 means the input layer.
        :return:
        """
        layer_name = None
        dom = xml.dom.minidom.parse('utils/model_info.xml')
        # root is an document element
        root = dom.documentElement
        # model is an element of each model
        model = root.getElementsByTagName(model_name)[0]
        layer_name_str = model.getElementsByTagName('model_layer_name')[0].firstChild.data
        layer_name_list = eval(layer_name_str.replace(" ","").replace("\n",""))
        layer_name = layer_name_list[layer_index]
        return layer_name
    def get_layer_shape_by_index(self,model_name,layer_index):
        """
        :param model_name:
        :param layer_index: [0,layer_num]
        layer_index==0 means the input layer.
        :return:
        """
        layer_shape = None
        dom = xml.dom.minidom.parse('utils/model_info.xml')
        # root is an document element
        root = dom.documentElement
        # model is an element of each model
        model = root.getElementsByTagName(model_name)[0]
        layer_shape_str = model.getElementsByTagName('model_layer_shape')[0].firstChild.data
        layer_shape_list = eval(layer_shape_str.replace(" ","").replace("\n",""))
        layer_shape = layer_shape_list[layer_index]
        return layer_shape


    def get_layer_size(self,model_name):
        dom = xml.dom.minidom.parse('utils/model_info.xml')
        # root is an document element
        root = dom.documentElement
        # model is an element of each model
        model = root.getElementsByTagName(model_name)[0]
        layer_size_str = model.getElementsByTagName('model_layer_size')[0].firstChild.data
        layer_size = eval(layer_size_str.replace(" ","").replace("\n",""))
        return layer_size


    def get_layer_size_by_index(self,model_name,layer_index):
        """
        :param model_name:
        :param layer_index: [0,layer_num]
        layer_index==0 means the input layer.
        :return:
        """
        dom = xml.dom.minidom.parse('utils/model_info.xml')
        # root is an document element
        root = dom.documentElement
        # model is an element of each model
        model = root.getElementsByTagName(model_name)[0]
        layer_size_str = model.getElementsByTagName('model_layer_size')[0].firstChild.data
        layer_size_list = eval(layer_size_str.replace(" ","").replace("\n",""))
        layer_size = layer_size_list[layer_index]
        return layer_size

    def get_layer_nums(self,model_name):
        dom = xml.dom.minidom.parse('utils/model_info.xml')
        # root is an document element
        root = dom.documentElement
        # model is an element of each model
        model = root.getElementsByTagName(model_name)[0]
        layer_nums_str = model.getElementsByTagName('model_layer_nums')[0].firstChild.data
        layer_nums = eval(layer_nums_str.replace(" ","").replace("\n",""))
        return layer_nums
    def get_mobile_latency(self,model_name):
        dom = xml.dom.minidom.parse('utils/model_info.xml')
        # root is an document element
        root = dom.documentElement
        # model is an element of each model
        model = root.getElementsByTagName(model_name)[0]
        mobile_latency_str = model.getElementsByTagName('mobile_latency')[0].firstChild.data
        mobile_latency = eval(mobile_latency_str.replace(" ","").replace("\n",""))
        return mobile_latency
import pandas as pd
import heapq
import sys
sys.path.append(".")
def process_search_weight_data():
    #file_path = ""
    # 1. 读取每个网络点的sys_h倍数，fairness，I_W,R_W,M_W的数据
    # 2. 去每个网络点sys_h ratio最高的10中配置。
    max_element = 100
    file_path = "../experiment/weight_search/has_queue/weight_fairness_search/WIFI_trace_param_search.xlsx"
    sys_h_writer = pd.ExcelWriter("../experiment/weight_search/weight_fairness_search/WIFI_trace_Top_"+str(max_element)+".xlsx")
    result = {}
    result_str = []
    for i in range(0,10):
        if i==4 or i==5:
            continue
        #1. read the network data
        #print(file_path)
        network_data = pd.read_excel(file_path,sheet_name="network="+str(i),index_col=0)
        # 2. read the sys_h ratio
        sys_h = network_data["Our_h/baseline_h"].values
        network_result = {"index":[],"Our_h/baseline_h":[],"fairness": [],"I_W": [], "R_W": [], "M_W": []}
        index = heapq.nlargest(max_element, range(len(sys_h)), sys_h.take)
        result_str.append(index)
        '''
        print(index)
        temp_str = []
        for j in index:
            data = network_data.iloc[j,:].values
            network_result["index"].append(j)
            network_result["Our_h/baseline_h"].append(round(data[0],2))
            network_result["fairness"].append(round(data[1],2))
            network_result["I_W"].append(round(data[2],2))
            network_result["R_W"].append(round(data[3],2))
            network_result["M_W"].append(round(data[4],2))
            temp_str.append("F_"+str(round(data[1],2))+"_I_"+str(round(data[2],2))+"_R_"+str(round(data[3],2))+"_M_"+str(round(data[4],2))+"_S_"+str(round(data[0],2)))
        result_str.append(temp_str)
        result["network="+str(i)] = network_result
        network_result_pd = pd.DataFrame(data= network_result,index = range(len(network_result["index"])))
        network_result_pd.to_excel(sys_h_writer,sheet_name="network="+str(i))
        '''
    #sys_h_writer.save()
    #sys_h_writer.close()
    #print(result_str)
    intersection = set.intersection(*map(set, result_str))
    print(intersection)


#process_search_weight_data()





