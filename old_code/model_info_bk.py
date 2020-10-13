# -*- coding: utf-8 -*-
import xml.dom.minidom
import numpy as np
import sys
sys.path.append(".")
class ModelInfo():
    def get_layer_name_by_index(self,model_name,layer_index):
        """
        :param model_name:
        :param layer_index: [0,layer_num]
        layer_index==0 means the input layer.
        :return:
        """
        layer_name = None
        if layer_index>0:
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
        layer_name = None
        if layer_index>0:
            dom = xml.dom.minidom.parse('utils/model_info.xml')
            # root is an document element
            root = dom.documentElement
            # model is an element of each model
            model = root.getElementsByTagName(model_name)[0]
            layer_name_str = model.getElementsByTagName('model_layer_shape')[0].firstChild.data
            layer_name_list = eval(layer_name_str.replace(" ","").replace("\n",""))
            layer_name = layer_name_list[layer_index]
        return layer_name
