import os
import sys
import socket
import struct
import numpy as np
from utils.model_info import ModelInfo
import cv2
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

def read_image(filename, resize_height, resize_width,normalization=False):
    '''
       读取图片数据,默认返回的是uint8,[0,255]
       :param filename:
       :param resize_height:
       :param resize_width:
       :param normalization:是否归一化到[0.,1.0]
       :return: 返回的图片数据
       '''

    bgr_image = cv2.imread(filename)
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    if resize_height > 0 and resize_width > 0:
        rgb_image = cv2.resize(rgb_image, (resize_width, resize_height))
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image = (rgb_image-128) / 127.0
    # show_image("src resize image",image)
    return rgb_image

def deal_input_image(model_name,img_str,normalization=False):
    # get the shape of the input data
    #print("=========",model_name)
    model_info = ModelInfo()
    shape = model_info.get_input_info(model_name)[0]
    resize_height = shape[0]
    resize_width = shape[1]
    # 2. convert the type of the image from string to numpy
    nparr = np.fromstring(img_str, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 3. conver the channel of image from BGR to RGB
    rgb_image = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    if resize_height>0 and resize_width>0:
        rgb_image=cv2.resize(rgb_image,(resize_width,resize_height))
    # 4. change the dtype
    rgb_image=np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image=(rgb_image-128)/127.0
    return rgb_image

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
      content = bytes(content,encoding="utf-8")
      msg = struct.pack('>I', len(content)) + content
      conn.send(msg)

   def recv_data(self,conn):
      # Receive data from the server
      # Return result(an numpy array)
      resp_len = self.recvall(conn, 4)
      if not resp_len:
         return None
      resp_len = struct.unpack('>I', resp_len)[0]
      if not resp_len:
         return None
      result = self.recvall(conn, resp_len)
      result = result.decode("utf8")
      return result
class Static_Info:
    EDGE_IP = "192.168.1.16"
    MOBILE_IP="192.168.1.14"
    MOBILE_CONTROLLER_PORT="10990"
    REFRESH_INTERVEL=60 #the edge refresh itself every $REFRESH_INTERVEL seconds.


class FileOperation:
    def get_module_config(self):
        """
        read system config info
        :return: dict
        """
        fileOpr = FileOperation()
        config = fileOpr.readFile("./utils/system_config", "r")
        #print(config)
        config = eval(config)
        return config
    # read data from the file
    # openModel could be r or rb
    def printLogFile(self, filePath):
        self.createFile(filePath)
        sys.stdout = open(filePath, "a+")

    def readFile(self, filePath, openMode):
        result = ""
        if os.path.exists(filePath):
            with open(filePath, openMode,encoding='UTF-8') as f:
                result = f.read()
                f.close()
        else:
            print(filePath, "does not exist")
        return result

    def isFileExist(self, filePath):
        result = True
        if not os.path.exists(filePath):
            result = False
        return result

    def listFile(self,dirPath):
        """
        list the name of all the files under this dirPath
        :param dirPath
        :return: the file's name under this dir
        """
        file_list = None
        if os.path.isdir(dirPath):
            file_list = os.listdir(dirPath)
        return file_list

    def createFile(self, filePath):
        if self.isFileExist(filePath):
            return
        else:
            filePath = str(filePath)
            foldPath = filePath[:filePath.rindex("/")]
            # print("foldpath => ",foldPath)
            if not os.path.isdir(foldPath):
                os.makedirs(foldPath)
            with open(filePath, "w+") as f:
                f.write('')
                f.close()

    # write data into a file
    # writeMode could be w  w+ wb, and it first checks whether the file exists
    # if it not exists, create the file,the put into the content.
    def writeFile(self, filePath, writeMode, content):
        self.createFile(filePath)
        # print("filePath = >", filePath)
        with open(filePath, writeMode) as f:
            f.write(content)
            f.close()
