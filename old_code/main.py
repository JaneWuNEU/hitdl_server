import os
from multiprocessing import freeze_support
import threading
from old_code.load_balancer import LoadBalancer
from utils.util import FileOperation
from old_code.model_optimizor import ModelOptimizor
from old_code.DB_manager import DBmanager
fileOpr = FileOperation()

os.system('fuser -k -n tcp 1097,1098,1099')   # kill 1097,1098,1099 ports which we will use later
modelOptimizor = ModelOptimizor()
dbManager = DBmanager()
config = fileOpr.get_module_config()

loadBalancer = LoadBalancer(config, dbManager)
def init():
    """
    initialize the system
    :return:void
    """

    band_width = dbManager.get_default_bandwidth()
    strategy = modelOptimizor.get_strategy(band_width)
    print("===init strategy===",strategy)
    loadBalancer.allocate_recv_port(strategy)
    loadBalancer.notify_user(strategy)
    dbManager.activate_user()

def update(interval):
    """
    update the strategy
    :return: void
    """
    model_id_list = dbManager.get_active_userlist()
    print("update get activate userlist", model_id_list)
    if len(model_id_list) > 0:
        band_width = dbManager.get_bandwidth_dic()
        result = modelOptimizor.get_strategy(model_id_list, band_width,loadBalancer.cycle)
        strategy = result[0]
        print(strategy)
        loadBalancer.pause_user(model_id_list)
        loadBalancer.remove_inactive_user()
        loadBalancer.allocate_recv_port(model_id_list,strategy)
        loadBalancer.notify_user(model_id_list,strategy)
        dbManager.activate_user()
    timer = threading.Timer(interval, update,[interval])
    timer.start()

if __name__ == '__main__':
    freeze_support()
    f = open("count_experiments.txt", "r")
    str1 = f.readline()
    f.close()
    init()
    update(config["system_config"]["strategy_update"])

