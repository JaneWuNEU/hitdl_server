import socket
import threading
from old_code.entity import RegisterSocket, ModelRecvThread,ModelProcess
from multiprocessing import Event
import multiprocessing
from utils.util import SocketCommunication
class LoadBalancer:
    def __init__(self, config, dbmanager):
        """
        initialize LoadBalancer
        :param config:
        :param devicemanager:
        :param dbmanager:
        """
        print("create load balancer")
        self.dbmanager = dbmanager

        self.start_port = config["load_balancer"]["start_port"]
        self.current_port = self.start_port
        self.edge_ip = config["load_balancer"]["edge_ip"]
        self.socket_backlog = config["load_balancer"]["socket_backlog"]
        self.recv_queue_len = config["model_socket"]["max_queue"]
        self.cycle = 0
        self.socket_tool = SocketCommunication()

        # Queue{model_name: queue instance} -> store the requests from the client
        self.model_recv_queue = {}

        # thread: {model_id:thread instance} -> receive the requests from the client
        self.user_recv_thread = {}

        # Process: {model_name: Process instance}
        self.model_process = {}
        # register port
        self.reg_port = 10990
        # ip, port, backlog, model_id, dbManager):
        self.reg_socket = RegisterSocket(self.edge_ip, self.reg_port, self.socket_backlog, self.dbmanager)
        self.reg_socket.start()
        # deregister port
        self.dereg_port = 10980
        self.dereg_socket = RegisterSocket( self.edge_ip, self.dereg_port, self.socket_backlog, self.dbmanager)
        self.dereg_socket.start()
        self.group_dict = {"inception_v3":"1"}
    def remove_inactive_user(self):
        # 1. get inactive_model_id_list
        model_id_list = self.dbmanager.get_inactive_userlist()
        for model_id in model_id_list:
            #try:
            # 2. close corresponding recv_thread
            self.user_recv_thread[model_id].stop_socket()
            # 3. close corresponding model_process
            model_name = model_id.split("*")[0]
            if model_name in self.model_process.keys():
                self.model_process[model_name].close_model_process()
            #except Exception as e:
            #    print("remove_inactive_user errors",e)

    def notify_user(self,model_id_list,strategy):
        """
        notify users with the recv port info and partition point
        :param model_id_list:
        :return:
        """
        for model_id in model_id_list:
            #try:
            # 1. get r1 and port for each user
            model_name = model_id.split("*")[0]
            group_id = self.group_dict[model_name]
            r1 = strategy[group_id]["r1"]
            print("model name list",self.user_recv_thread.keys())
            recv_port = self.user_recv_thread[model_id].port
            data = {"r1": r1,"port":recv_port,"group":int(group_id)}
            # 2. create sockets to return results
            user_num = int(model_id.split("*")[3])
            return_port = 20000 + user_num
            user_ip = model_id.split("*")[1]
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.connect((user_ip, return_port))
            self.socket_tool.send_data(client,str(data))
            #except Exception as e:
            #    print("notify user errors",e)
    def allocate_recv_port(self,strategy):
        """
         create a Process bound with a independent port for each model instance to
         listen its requests, and then store the requests into @model_recv_queue.
         Note that the users who access to the same instance share the recv_queue
        :param model_id_list: new users' model_id list
        :return:
        """
        # 1.create shared recv_queue and model process
        for model_name in ["inception","resnet","mobilenet"]:
            strategy_details = strategy[model_name]
            for ins_num in range(0,strategy["ins_number"]):
                # 1.1 create a model instance
                    # 1.1 create recv queue
                m = multiprocessing.Manager()
                recv_queue = m.Queue(self.recv_queue_len)
                print("create shared queue")
                self.model_recv_queue[model_name] = recv_queue
                # 1.2 create and start model process
                group_id = self.group_dict[model_name]
                print("strategy", strategy)
                model_param = strategy[group_id]
                start_event = Event()
                self.model_process[model_name] = ModelProcess(self.model_recv_queue[model_name], model_param, self, model_name, mobile_type,
                                                              start_event)
                self.model_process[model_name].start()
            # 2. allocate a port and start the corresponding thread to receive users' requests.
            if model_id in self.user_recv_thread.keys():
                # old users
                continue
            else:
                # new users
                model_recv_thread = ModelRecvThread(self.model_recv_queue[model_name], model_id,
                                                    self.model_process[model_name].start_event, threading.Event())
                while not model_recv_thread.assgin_port(self.edge_ip, self.current_port):
                    self.current_port = self.current_port + 1
                model_recv_thread.start()
                self.user_recv_thread[model_id] = model_recv_thread
        s

    def pause_user(self, model_id_list):
        '''
        Notify users to stop sending images while reallocating strategy
        :param model_id_list:
        :return:
        '''
        for model_id in model_id_list:
            # 1. get r1 and port for each user
            #try:
            model_name = model_id.split("*")[0]
            group_id = self.group_dict[model_name]
            r1 = -1
            recv_port = 0
            data = str({"r1":r1,"port":recv_port,"group":int(group_id)})
            # 2. create sockets to return results
            user_num = int(model_id.split("*")[3])
            return_port = 20000 + user_num
            user_ip = model_id.split("*")[1]
            result = False
            i = 0
            while (not result) and i<2:
                try:
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    print("pause user", user_ip, return_port)
                    client.connect((user_ip, return_port))
                    self.socket_tool.send_data(client,str(data))
                    result = True
                except Exception as e:
                    print("error happens when notifying users",e)
                    i = i + 1
            #except Exception as e:
            #    print("pause user errors",e)

