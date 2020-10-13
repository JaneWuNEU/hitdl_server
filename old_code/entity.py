import pickle
import threading
import socket
import time
import datetime
import codecs
import sys
from utils.util import SocketCommunication, FileOperation, deal_input_image,read_image
from multiprocessing import Process
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from queue import Full
from utils.model_info import ModelInfo
import numpy as np
import model_zoo.models.inception_v3 as inception_v3
import os
import tensorflow.contrib.slim as slim
sys.path.append("./")
fileOper = FileOperation()
standrad = 1576842000

class ModelRecvThread(threading.Thread):
    """
    receive the requests from clients.
    """
    def __init__(self, recv_queue, model_id, start_event, stop_event):
        #super(ModelSocket, self).__init__()
        super().__init__()
        self.socket = None
        # initialize the attributes
        self.ip = None
        self.port = None
        self.recv_queue = recv_queue
        self.model_id = model_id
        self.stop = stop_event
        self.socket_tool = SocketCommunication()
        self.start_event = start_event
    def assgin_port(self,ip, port):
        """
        create a socket with specific ip and port to the model.
        If assign the socket sucessfully, return True, otherwise return false.
        :param ip:
        :param port:
        :param backlog:
        :return:
        """
        result = True
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((ip, port))
            self.socket.listen(100)
            self.ip = ip
            self.port = port
        except OSError:
            print("error happens when assigning a port to model socket")
            self.ip = None
            self.port = None
            result = False
        return result
    def stop_socket(self):
        self.stop.set()
    def run(self):
        # 1. wait for the ModelProcess finishing its initialization
        if self.start_event.is_set():
            time.sleep(7)
        while self.start_event.is_set():
            time.sleep(0.5)

        # 2. start to receive requests
        try:
            conn, addr = self.socket.accept()
            while not self.stop.isSet():
            # 1. recv data
                try:
                    receive_data_start = time.time()
                    t2 = time.time()
                    data = self.socket_tool.recv_data(conn)
                    if data is None:
                        continue
                    #t3 = datetime.datetime.now()
                    t3 = time.time()
                    receive_time = round((t3-t2)*1000)
                    data = data.split('_')
                    #ip,user_num,pic_num
                    user = addr[0] + '_' + data[0] + '_' + data[1]
                    r1 = data[3]
                    temp = codecs.decode(data[2].encode(), "base64")
                    if r1 == '0':#user sends the original picture
                        model_name = self.model_id.split("*")[0]
                        data = deal_input_image(model_name,temp,normalization=True)
                    else:
                        data = pickle.loads(temp)
                        data = np.asarray(data[0],dtype=np.float32)
                    post_receive = round((time.time()-t3)*1000)
                    queue_start = time.time()
                    # post_receive, time spent in receiving data
                    data = [post_receive,receive_data_start,data,receive_time,queue_start,user]
                    self.recv_queue.put(data)
                    #print("requests in queue",self.recv_queue.qsize(),self.recv_queue)
                except Full as e:
                    print("error happens because queue if full",e)
                except Exception as e:
                    print("model socket receive errors",e)
        except Exception as e:
            print("Model Socket is closed",e)
            self.stop_socket()

class RegisterSocket(threading.Thread):
        def __init__(self,ip, port, backlog, dbManager):
            """
            initialize ModelSocket
            :param load_balancer: LoadBalancer
            :param ip: str
            :param port: int
            :param backlog: str
            :param model_id: str
            """
            threading.Thread.__init__(self)
            # initialize the attributes
            self.ip = ip
            self.port = port
            self.backlog = backlog
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self.socket.bind((ip, port))
                self.socket.listen(self.backlog)
                print("open port",ip,port)
            except Exception as e:
                print("bind register|deregister port fail",e)
                os.system('fuser -k -n tcp 10990')
            self.dbManager = dbManager
            self.sock_comm = SocketCommunication()
            self.dbManager = dbManager
        def run(self):
            """
            listen to the port and receive the data.
            Then pass the data to DeviceManager to deal with it, and return the result
            :return:
            """
            while True:
                #try:
                print("ready to accept",self.ip,self.port)
                conn, addr = self.socket.accept()
                # receive data
                data = self.sock_comm.recv_data(conn)
                model_id = eval(data)
                print("receive register info from ",model_id)
                response = None
                if self.port == 10990:
                    result = self.dbManager.register(model_id["model_id"])
                    if result == 0:
                        response = "Have Registered"
                    elif result == -1:
                        response = "Register Unsuccessfully"
                    else:
                        response = "Register Successfully"
                elif self.port == 10980:
                    result = self.dbManager.deregister(model_id["model_id"])
                    if result == 0:
                        response = "Deregister Unsuccessfully"
                    else:
                        response = "Deregister Successfully"
                self.sock_comm.send_data(conn,response)
                # print("response",response,conn)
                conn.close()
                #except Exception as e:
                    #print("open register | deregister failed",e)

class ModelProcess(Process):
    def __init__(self, recv_queue, model_param, loadbalancer, model_name,mobile_type, start_event):
        """

        :param recv_queue:
        :param model_param:
        :param loadbalancer:
        :param model_name:
        :param config: system_config
        """
        super().__init__()
        self.recv_queue = recv_queue
        self.model_param = model_param
        self.loadBalancer = loadbalancer
        self.model_name = model_name
        self.mobile_type = mobile_type
        config = fileOper.get_module_config()
        self.total_CPUs = config["device_manager"]["CPU"]
        self.total_GPUs = config["device_manager"]["GPU"]
        self.sys_available_device = {"CPU":self.total_CPUs,"GPU":self.total_GPUs}
        self.start_event = start_event
        self.socket_tool = SocketCommunication()
        self.stop = False
        self.model_info = ModelInfo()
        self.user_return_socket={}

    def close_model_process(self):
        self.stop = True

    def return_result(self,result,local_image,edge_time):
        i = 0
        device_info = self.model_param["device_info"]
        r1 = self.model_param["r1"]
        intra = self.model_param["intra"]
        batch = self.model_param["batch"]
        memory = self.model_param["memory"]
        for processed_image in local_image:
            # 1. get the returned request
            user = processed_image[5]
            # 2. record the process time
            processed_image.append(round(edge_time))  # edge处理时间
            # 3. record the returning time
            return_result_start = time.time()
            addr = user.split('_')  # ip,user_num,pic_num
            IP = addr[0]
            PORT = 30000 + int(addr[1])  # 客户端端口
            #try:
            DATA = codecs.encode(pickle.dumps(result[i]), "base64").decode()
            msg = str(addr[2]) + '_' + DATA + "_" + str(time.time())
            client = None
            if IP in self.user_return_socket.keys():
                client = self.user_return_socket[IP]
            else:
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.connect((IP, PORT))
                self.user_return_socket[IP] = client
            self.socket_tool.send_data(client, msg)
            #print("return result")
            #except Exception as e:
                #print((IP, PORT), "return result errors", e)

    def run(self):
        device_info = "CPU:0"
        r1 = self.model_param["r1"]
        intra = self.model_param["intra"]
        batch = self.model_param["batch"]
        model_path = "model_zoo/weights/"+self.model_name+"/"+self.model_name+".ckpt"
        if r1 != -1:
            tf.reset_default_graph()
            sess_config = tf.ConfigProto(device_count=self.sys_available_device, #use_per_session_threads=True,
                                         intra_op_parallelism_threads=intra,inter_op_parallelism_threads=2,
                                         log_device_placement=False,
                                         session_inter_op_thread_pool=[config_pb2.ThreadPoolOptionProto(num_threads=2)])
            with tf.device(device_info):
                with tf.Session(config=sess_config) as sess:
                    if self.model_name=="inception_v3":
                        # read the config input of input data or middle data
                        layer_info = self.model_info.get_layer_info_by_layer_index(self.model_name,r1)
                        data_size = layer_info[0]
                        input_images = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None, data_size[0], data_size[1], data_size[2]])
                        partition_layer = self.model_info.get_layer_name_by_index(self.model_name,r1)

                        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                            out, endpoints = inception_v3.inception_v3(inputs=input_images,partition_layer=partition_layer)
                            #print("endpoints",endpoints)
                            sess.run(tf.global_variables_initializer())
                            saver = tf.train.Saver()
                            saver.restore(sess, model_path)

                            # preheat the model
                            temp = np.random.rand(batch,data_size[0],data_size[1],data_size[2])
                            a = time.time()
                            sess.run(out,feed_dict={input_images:temp})
                            print("preheat the model",time.time()-a)
                            save_input = True
                            self.start_event.set()
                            while not self.stop:
                                item_count = 0
                                input_data = []
                                local_img = []
                                while item_count < batch:
                                    try:
                                        #[post_receive,receive_data_start,data,receive_time,queue_start,user]
                                        processed_image = self.recv_queue.get(block=False)
                                        queue_start = processed_image[1]
                                        queue_end = np.around(time.time() - standrad, decimals=3)
                                        processed_image.append(queue_end)
                                        queue = np.around((queue_end - queue_start) * 1000, decimals=3)
                                        processed_image.append(queue)
                                        input_data.append(processed_image[2])
                                        local_img.append(processed_image)
                                        item_count += 1
                                    except Exception as e:
                                        print("no requests",e)
                                        if len(local_img) != 0:
                                            break
                                        else:
                                            time.sleep(0.001)
                                for processed_image in local_img:
                                    processed_image.append(round(time.time() - standrad, 3))
                                edge_process_start = datetime.datetime.now()
                                if save_input:
                                    print(input_data)
                                    save_input = False
                                result = sess.run(out,feed_dict={input_images:np.array(input_data)})
                                #print("========finish precessing========",np.argmax(result[0]))
                                #print(result[0])
                                edge_time = round((datetime.datetime.now() - edge_process_start).total_seconds() * 1000)
                                if result is not None:
                                   self.return_result(result,local_img,edge_time)

#stratety = {'1': {'user_list': ['alexnet_192.168.31.151_M1_1'], 'model_type': 'alexnet', 'r1': 0, 'device': 'CPU', 'batch': 10, 'intra': 12}}
#temp = ModelSocket(None, model_device_mapping, group, batch, None,flag)
