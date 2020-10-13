import socket
import sys
sys.path.append(".")
from utils.util import Static_Info
from threading import Event,Thread
from queue import Queue
from utils.util import SocketCommunication
import time
import tensorflow as tf
import model_zoo.net.resnet_v2 as resnet_v2
import model_zoo.net.inception_v3 as inception_v3
import model_zoo.net.mobilenet_v1 as mobilenet_v1
import tensorflow.contrib.slim as slim
from utils.util import Static_Info,ModelInfo
import numpy as np
import codecs
import pickle
import os
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import datetime
import copy
from multiprocessing import Manager,Process
import zlib
import threading
class ModelInstance:
    """
    ModelInstance is one of the most important component in the edge.
    It first records the necessary attributes of a model instance, like ins_id, model_name.
    Then its $run_model methods runs in a independent subprocess where a sub-thread is started to listen to the reccv_port,
    store the mobile requests, and share it with the subprocess.
    Note that the subprocess is bound with specific CPU cores so as to control the interference among instances.
    The subprocess is killed when the new round resource allocation generates.
    """
    def __init__(self,k,intra,model_name,ins_id,ins_port,core_id,user_num_per_ins,device_manager,cpu_id,ins_num):
        '''
        initialize member variables
        '''
        # ==== create necessary member variables ===
        self.k = k
        self.model_name = model_name
        self.intra = intra
        self.ins_id = ins_id
        self.ins_port = ins_port
        self.core_id = core_id
        self.request_records = {}
        self.model_info = ModelInfo()
        self.user_num_per_ins = user_num_per_ins
        self.device_manager = device_manager
        self.cpu_id = cpu_id
        self.ins_num = ins_num

    def process_image(self,sess, out, endpoints,input_images,data_queue):
        '''
        process images and add records the process
        tips:
        '''
        def return_request(request_result,user_ip,user_port):
            try:
                a = time.time()
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.settimeout(0.01)
                client.connect((user_ip,user_port))
                a = time.time()
                sock_tools.send_data(client,request_result)
                b = time.time()
                #print("returning time cost",b-a)
            except Exception as e:
                print("@",e)
        sock_tools = SocketCommunication()

        while True:
            # 1. process the image
            # 1.1 read data from the recv_queue

            try:
                #print("####run#########",self.model_name,self.ins_id)
                data = data_queue.get()  # {user_ip:,recv_port:,data:{'data':,'pic_num':},"enqueue_time":,"edge_recv_time":}
                #data = data_queue.get()
                dequeue_time = time.time()
                enqueue_time = data["enqueue_time"]
                if self.model_name == "inception":
                    a = time.time()
                    test  = data["data"]["data"]
                    b = time.time()
                    print("读取数据的时间",b-a)
                a = time.time()
                result = sess.run(out, feed_dict={input_images: data["data"]["data"]})
                b = time.time()
                if self.model_name=="inception":
                    print("out",out)
                    print("%%%%%%%%%%%%%运行模型l%%%%%%%%%%%%",self.model_name,self.ins_id,b-a,dequeue_time-enqueue_time,input_images.shape)
                image_id = np.argmax(result[0])

                # {"edge_run_time":X,"pic_num":Y,"image_id":Z,"queue_time":x,"edge_queue_time":x,"edge_recv_time":}
                # 1.2 return the results
                reqeust_result = {"edge_run_time": b - a, "pic_num": data['data']["pic_num"],
                                  "image_id": image_id, "queue_time": dequeue_time - enqueue_time,
                                  "edge_recv_time": data["edge_recv_time"],"bandwidth":data["bandwidth"],"queue_size":data_queue.qsize()}
                return_request(reqeust_result, data["user_ip"], data["recv_port"])

            except Exception as e:
                #pass
                print("@（（（（（（（（（（（（（（（（（（",e,self.ins_id,self.model_name)
                sys.exit(0)

    def bound_pid(self,pid):
        core_str = ''
        for i in self.core_id:
            core_str = core_str+str(i)+","

        core_str = core_str[:core_str.rindex(",")]
        #print("%%%%%%%%bound pid %%%%%%%",self.model_name,self.ins_id,self.core_id,core_str)
        os.system("taskset -cp " + core_str + " " + str(pid))
    def run_model(self,release_flag_dict):
        """
        The user should run the complete model in specific CPU cores defined by $core_id after being created.
        Only the the edge activates the user can it run the model in a hybrird way.
        """
        # 0. bind pid
        #print("&&&&&&&&&&&&&&&&&",os.getpid())
        pid = os.getpid()
        self.bound_pid(pid)

        # 1. initialize the model
        sess_config = tf.ConfigProto(intra_op_parallelism_threads=self.intra, log_device_placement=False,
                                 allow_soft_placement=True, device_count={"CPU":1},isolate_session_state = True)
        model_info = ModelInfo()

        layer_name = model_info.get_layer_name_by_index(self.model_name,self.k)
        #print("%%%%%%%%%%%%%layer name%%%%%%%%%%%",layer_name)
        layer_shape = model_info.get_layer_shape_by_index(self.model_name,self.k)

        #2. start a thread to receive data
        #manager = Manager()
        #data_queue = manager.Queue()
        data_queue = Queue(self.user_num_per_ins+10)#Static_Info.MAX_QUEUE_LEN
        user_port_dict = {}
        for i in range(self.user_num_per_ins):
            #release_flag_dict[self.model_name+"_"+str(self.ins_id)+"_"+str(i)]=-1
            Thread(target=self.recv_data,args=[data_queue,release_flag_dict,user_port_dict,i]).start()
        input_images = tf.placeholder(dtype=tf.float32, shape=[None, layer_shape[0], layer_shape[1], layer_shape[2]],name='input')
        #with tf.device("CPU:"+str(self.cpu_id)):
        if True:
            if self.model_name == "inception":
                with tf.device("CPU:0"):
                    model_path = "model_zoo/weights/inception_v3_quant.ckpt"
                    with tf.Session(config=sess_config) as sess:
                            with tf.contrib.slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                                out, endpoints = inception_v3.inception_v3(inputs=input_images,final_endpoint = "Predictions",partition_layer= layer_name)
                                print("++++++++++++++++++++",sess_config)
                                sess.run(tf.global_variables_initializer())
                                saver = tf.train.Saver()
                                saver.restore(sess, model_path)
                                self.process_image(sess, out, endpoints, input_images, data_queue)

            elif self.model_name == "resnet":
                model_path = "model_zoo/weights/resnet_v2_50.ckpt"
                input_images = tf.placeholder(dtype=tf.float32, shape=[None,layer_shape[0], layer_shape[1], layer_shape[2]], name='input')
                with tf.device("CPU:0"):
                    with tf.Session(config=sess_config,graph=tf.get_default_graph()) as sess:
                        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                            out, endpoints = resnet_v2.resnet_v2_50(inputs=input_images,final_endpoints="predictions", partition_layer = layer_name)
                            #print("&&&&&&&resnet&&&&&&&&&",)
                            sess.run(tf.global_variables_initializer())
                            saver = tf.train.Saver()
                            saver.restore(sess, model_path)
                            self.process_image(sess, out, endpoints, input_images,data_queue)
                self.process_image(None,None,None,None,None)
            else:
                model_path = "model_zoo/weights/mobilenet_v1_1.0_224.ckpt"
                input_images = tf.placeholder(dtype=tf.float32, shape=[None,layer_shape[0], layer_shape[1], layer_shape[2]], name='input')
                with tf.device("CPU:0"):
                    with tf.Session(config=sess_config) as sess:
                        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
                            #print("&&&&&&&&&&&&&&&&&&&&&&&&",sess_config)
                            out, endpoints = mobilenet_v1.mobilenet_v1(inputs=input_images,partition_layer=layer_name)
                            #print("&&&&&&&&&&&&&&&&&&&&&&&&",out,endpoints)
                            sess.run(tf.global_variables_initializer())
                            saver = tf.train.Saver()
                            saver.restore(sess, model_path)
                            self.process_image(sess, out, endpoints, input_images,data_queue)

    def recv_data(self,data_queue,release_flag_dict,user_port_dict,user_index):
        def assign_port(model_name,user_index,ins_id,user_num_per_ins):
            from utils.util import Static_Info
            """
            Different types of users have different methods to assign ports.
            SERVER_USER as 1000+$user_id*$RECV_PORT_INTERVEL, RASP_USER as 2000+$user_id*$RECV_PORT_INTERVEL.
            Note that the exception may be thrown during opening a specific port marked as $wrong_port, when try a new port
            as $(wrong_port++) until an available port appears.
            """
            if model_name == 'mobilenet':
                temp_revc_port = Static_Info.MOBILENET_RECV_PORT_START + (
                            ins_id + user_index) * Static_Info.RECV_PORT_INTERVEL
            elif model_name == 'resnet':
                temp_revc_port = Static_Info.RESNET_RECV_PORT_START + (
                            ins_id + user_index) * Static_Info.RECV_PORT_INTERVEL
            else:
                temp_revc_port = Static_Info.INCEPTION_RECV_PORT_START + (
                            ins_id + user_index) * Static_Info.RECV_PORT_INTERVEL

            # 1. open a socket to receive the date from the users.

            while True:
                try:
                    recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    recv_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    if model_name == "mobilenet":
                        recv_socket.bind((Static_Info.EDGE_IP_RASP, temp_revc_port))
                    else:
                        recv_socket.bind((Static_Info.EDGE_IP_STACK, temp_revc_port))
                        #rint("###############recv info##############",model_name,ins_id,temp_revc_port)
                    recv_socket.listen(user_num_per_ins + 3)
                    ins_port = temp_revc_port
                    recv_socket = recv_socket
                    #print("+=====assign ports", temp_revc_port, self.model_name, self.ins_id)
                    break
                except Exception as e:
                    #print("error happens when assigning recv ports ", temp_revc_port, "to model ins ", self.ins_id, " ",model_name, e)
                    # os.system("fuser -k -n tcp "+str(temp_revc_port))
                    # os.system("fuser -k -n tcp " + str(temp_revc_port))
                    temp_revc_port = temp_revc_port + 1
            return ins_port, recv_socket

        def listenToClient(conn,add,port):
            # receive data
            try:
                a = time.time()
                request = comm_sock.recv_data_bytes(conn)
                b = time.time()
                conn.close()
                c = time.time()
                recv_time = b - a
                close_time = c-b
                # calculate bandwidth
                data_size = len(request)
                bandwidth = len(request) * 8 / 1024.0 / 1024.0 / (b - a)  # Mbits/s
                a = time.time()
                request = pickle.loads(request)
                b = time.time()
                #if self.model_name=="mobilenet":
                #    print("============带宽=========",bandwidth,"转化时间",b-a)
                enqueue_time = time.time()
                #print("接收时间",round(recv_time,3),"关闭socket",round(close_time,3),"数据解析",round(b-a,3),3)
                #print()
                #print("close time",self.model_name,request["user_id"],close_time)
                request.update({'enqueue_time': enqueue_time, "edge_recv_time": recv_time, "bandwidth": bandwidth,"close_time":close_time})
                data_queue.put(request, block=False)
            except Exception as e:
                pass
                #print('error happens when the thread receive the requests',e,self.model_name,self.ins_id)
        """
        assgin a port to the instances, and port is recalled when a new strategy is generated
        :param data_queue:
        :param release_flag:
        :return:
        """
        # 2. receive the results until the instance is released
        comm_sock = SocketCommunication()
        ins_port, recv_socket = assign_port(self.model_name,user_index,self.ins_id,self.user_num_per_ins)

        user_port_dict[self.model_name+"_"+str(self.ins_id)+"_"+str(user_index)] = ins_port
        #print("$$$$$$$$$$$$$$$",self.model_name+"_"+str(self.ins_id)+"_"+str(user_index),ins_port,release_flag_dict.keys())
        release_flag_dict[self.model_name+"_"+str(self.ins_id)+"_"+str(user_index)] = ins_port
        while True:
            try:
                #print("#############port listening", ins_port, self.ins_id,self.model_name,'$$$$$$$$$$$$$$$')
                ''''''
                conn,add = recv_socket.accept()
                listenToClient(conn,add,ins_port)
                #time.sleep(1)
                #print("------%recv data------", self.model_name, self.ins_id)
                #print(os.getpid(),threading.current_thread().name,self.model_name,self.ins_id)
            except Exception as e:
                print("error happens when receiving data",e,self.model_name)

