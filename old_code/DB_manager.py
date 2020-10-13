import sys
sys.path.append("./model_zoo")
import numpy as np

class DBmanager(object):
    model_id_list = []

    def __init__(self):
        """
        # init _ establish connection
         config: dic {
        'host': '115.156.197.239',  # mysql ip
        'port': 3306,         # mysql port
        'user': 'root',       # mysql account
        'passwd':'wujing123'     # mysql password
        }
        """
        self.usernum_modeltype = {}
        self.experiment_count = None
        self.uploade_ex_num()

        self.table_path = "./log/"
        self.request_table = []
        self.user_table = {}
        '''
        dic = {
            'model_id'：
            {model_id,
            'model_type': model_type,
            'user_ip': user_ip,
            'mobile_type': mobile_type,
            'user_num' : user_num ,
            'state': state,
            'upload_bandwidth' :upload_bandwidth,
            'download_bandwidth':download_bandwidth
            }
        }
        '''
        self.deploy_table = []

    def register(self,model_id):
        """
        register a user based on the data ,then add the record into user_info table
        If there is already a user with the same ip in the database, the registration is no longer repeated.

        :param user_ip: str
        :param mobile_type: str
        :param model_type: str
        :return: result == 0 > register successfully
        result == -1 -> register failing
        result == 1 -> have registered
        """
        print("db record registering info",model_id)
        list_ = model_id.split("*")
        model_name = list_[0]
        user_ip=list_[1]
        mobile_type = list_[2]
        user_num = int(list_[3])
        result = 0
        flag = False
        try:
            for user in self.user_table.keys():
                if user == str(model_id):
                    flag = True
                    break
            if flag:       # list is not empty, there is already a user with the same ip and num  in the database
                result = 1
            else:
                if model_name == 'inception_v3':
                    upload_bandwidth = 300
                    download_bandwidth = 500
                    # upload_bandwidth = 30
                    # download_bandwidth = 80
                    # upload_bandwidth = 18.88
                    # download_bandwidth = 54.97
                state = 0
                self.user_table[str(model_id)] = {
                    'model_type': str(model_name),
                    'user_ip': str(user_ip),
                    'mobile_type': str(mobile_type),
                    'user_num': str(user_num),
                    'state': str(state),
                    'upload_bandwidth': upload_bandwidth,
                    'download_bandwidth': download_bandwidth
                }
        except Exception as e:
            result = -1
            print("error happens when registering",e)
        return result
    def deregister(self, model_id):
        """
        deregister a user according to user_ip  , change the specific recording's item 'state' from 1 to -1
        If there is no user with the  ip , do nothing
        :param user_ip: str
        :return: 0  or 1
        0-> deregister successfully
        1-> deregister unsuccessfully
        """
        result = True
        try:
            self.user_table[str(model_id)]["state"] = str(-1)
        except Exception as e:
            result = False
            print("error happens when deregistering",e)

        return result

    def get_inactive_userlist(self):
        """
        get all the records from user_info , that have a 'state' value 1
        then return the records' model_id list
        :return: [ records ]  # 'state == 1'
        """
        inactive_userlist = []
        try:
            for model_id in self.user_table.keys():
                if self.user_table[model_id]["state"] == "-1":
                   inactive_userlist.append(model_id)
        except Exception as e:
            print("error happens when getting inactive_userlist",e)
        return inactive_userlist
    def get_active_userlist(self):
        """
        get all the records from user_info , that have a 'state' value 1
        then return the records' model_id list
        :return: [ records ]  # 'state == 1'
        """
        active_userlist = []
        try:
            for model_id in self.user_table.keys():
                if self.user_table[model_id]["state"] == "0" \
                        or self.user_table[model_id]["state"] == "1" :
                   active_userlist.append(model_id)
        except Exception as e:
            print("error happens when getting active_userlist",e)
        return active_userlist

    def activate_user(self):
        """
         # Activate all users : make all users'  'state' = 1
        :return:
        True （there are some users hadn't been active)
        False（all users were active,there is no need to activate any one）
        """
        result = True
        try:
            for model_id in self.user_table.keys():
                if self.user_table[model_id]["state"] == "0":
                   self.user_table[model_id]["state"] == "1"
        except Exception as e:
            print("error happens when getting active user",e)
            result = False
        return result

    def add_deploy_info(self,deploy_dic):
        """
        :param deploy_dic: {
        time : str
        model_id_list : str
        bandwidth :str
        port_dict : str
        strategy : str}
        :return:
        """
        time = str(deploy_dic["time"])
        model_id_list = str(deploy_dic["model_id_list"])
        bandwidth = str(deploy_dic["bandwidth"])
        port_dict = str(deploy_dic["part_dict"])
        strategy = str(deploy_dic["strategy"])
        result = True
        content = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"|"
        try:
             self.deploy_table.append(time + "|" + model_id_list + "|" + bandwidth + "|" + port_dict + "|" + strategy)

        except Exception as e:
            print("error happens when recording deploying strategy")
            result = False
        return result
    def update_down_bandwidth(self ,user_num,new_down_bw):
        '''
        this founction can upgrade the  bandwidth of a user
        :param model_id:
        :param new_bandwidth:   the new band
        :return:
        '''
        #print("download user-----num",user_num)
        #print("user_table",self.user_table.keys())
        result = True
        try:
            for model_id in self.user_table.keys():
                if self.user_table[model_id]["user_num"]==str(user_num):
                    self.user_table[model_id]["download_bandwidth"] = 30#new_down_bw
        except Exception as e:
            print("error happens when updating bandwidth",e)
            result = False
        return result
    def update_upload_bandwidth(self ,user_num,new_up_bw):
        '''
        this founction can upgrade the  bandwidth of a user
        :param model_id:
        :param new_bandwidth:   the new band
        :return:
        '''
        result = True
        #print("user-----num",user_num)
        #print("user_table",self.user_table.keys())
        try:
            for model_id in self.user_table.keys():
                if self.user_table[model_id]["user_num"]==str(user_num):
                    self.user_table[model_id]["upload_bandwidth"] = 40#new_up_bw
                    print("**************更新带宽=========",new_up_bw)
        except Exception as e:
            print("error happens when updating bandwidth",e)
            result = False
        return result
    def get_default_bandwidth(self):
        bandwidth_dic = {"inception":{"upload_bandwidth":1,"download_bandwidth":1},
                         "resnet":{"upload_bandwidth":1,"download_bandwidth":1},
                         "mobilenet":{"upload_bandwidth":1,"download_bandwidth":1}}
        return bandwidth_dic
    def get_bandwidth_dic(self):
        """
        get all the bandwidth from user_info , that have a 'state' value 1 or '0'

        :return: bandwidth  : dic { model_id : bandwidth}  # 'state == 1' or '0'
        """
        bandwidth_dic = {"inception":{"upload_bandwidth":1,"download_bandwidth":1},
                         "resnet":{"upload_bandwidth":1,"download_bandwidth":1},
                         "mobilenet":{"upload_bandwidth":1,"download_bandwidth":1}}
        inception_upload = []
        inception_download = []
        resnet_upload = []
        resnet_download = []
        mobilenet_upload = []
        mobilenet_download = []
        try:
            for model_id in self.user_table.keys():
                if self.user_table[model_id]["state"] ==str(0) or self.user_table[model_id]["state"]==str(1):
                    if "resnet" is model_id:
                        resnet_upload.append(self.user_table[model_id]["upload_bandwidth"])
                        resnet_download.append(self.user_table[model_id]["download_bandwidth"])
                    elif "inception" is model_id:
                        inception_upload.append(self.user_table[model_id]["upload_bandwidth"])
                        inception_download.append(self.user_table[model_id]["download_bandwidth"])
                    else:
                        mobilenet_upload.append(self.user_table[model_id]["upload_bandwidth"])
                        mobilenet_download.append(self.user_table[model_id]["download_bandwidth"])
            bandwidth_dic["inception"]["upload_bandwidth"] = np.average(np.array(inception_upload))
            bandwidth_dic["inception"]["download_bandwidth"] =np.average(np.array(inception_download))

            bandwidth_dic["resnet"]["upload_bandwidth"] = np.average(np.array(resnet_upload))
            bandwidth_dic["resnet"]["download_bandwidth"] = np.average(np.array(resnet_download))

            bandwidth_dic["mobilenet"]["upload_bandwidth"] = np.average(np.array(mobilenet_upload))
            bandwidth_dic["mobilenet"]["download_bandwidth"] = np.average(np.array(mobilenet_download))

        except Exception as e:
            print("error happens when getting bandwidth dict",e)
        return bandwidth_dic

    def uploade_ex_num(self):
        '''
        更新 实验次数标记  从文件 “count_experiments.txt”
        :return:
        '''
        f = open("count_experiments.txt", "r")
        str1 = f.readline()
        self.experiment_count = int(str1)
        f.close()
        next_count = self.experiment_count+1
        f = open("count_experiments.txt", "w+")
        f.write(str(next_count))









