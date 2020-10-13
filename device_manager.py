class BandWidthManager:
    def __init__(self):
        self.model_bandwidth = None
        self.user_bandwidth = {}
    def reset_user_bandwidth(self):
        self.user_bandwidth.clear()
    def get_default_model_bandwidth(self):
        bandwidth_dic = {"inception":{"upload_bandwidth":1,"download_bandwidth":1},
                         "resnet":{"upload_bandwidth":1,"download_bandwidth":1},
                         "mobilenet":{"upload_bandwidth":1,"download_bandwidth":1}}
        return bandwidth_dic
    def get_dynamic_model_bandwidth(self):
        bandwidth = None
        if self.model_bandwidth == None:
            bandwidth = self.get_default_model_bandwidth()
        else:
            pass
            #bandwidth = self.bandwidth
        return bandwidth

    def update_model_bandwidth(self):
        pass
    def update_user_bandwidth(self,user_mark,upload_banwidth):
        """
        It is in the moment when a user sends data to the edge does the edge update this user's upload bandwidth.
        :param user_mark: $model_name_$user_ip_$user_id
        :param upload_banwidth: Mbits/s
        :return:
        """
        self.user_bandwidth.update({user_mark:{"upload_bandwidth":upload_banwidth}})
