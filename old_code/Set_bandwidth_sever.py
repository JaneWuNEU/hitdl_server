import sys
import os
import time

class Set_sever_bandwidth(object):

    def __init__(self):
        self.sudoPassword = 'wujing123'   # 旧工作站 sudo 密码

    def init_tc(self, usernum,start_bw=40):
        '''
        根据用户num , 生成该用户的数据发送管道 。
        在用户加入系统时调用； 或在系统初始化时，从（1 ， 该次实验用户总数）循环，预先生成一系列管道

        初始带宽 40
        :param usernum:
        :return:
        '''

        if usernum == 1:
            self.reset_tc_iptables()   # 新一轮实验开始



        init_tc1 = "tc qdisc add dev eno1 root handle 1: htb default 24"
        init_tc2 = "tc class add dev eno1 parent 1:1 classid 1:2 htb rate 1000mbit prio 3"
        init_tc3 = "tc class add dev eno1 parent 1:1 classid 1:3 htb rate 1000mbit prio 3"

        user_port_down = 30000 + usernum * 100
        user_port_up = 30000 + usernum * 100 + 99

        test_bw_port_down = 40000 + usernum * 100
        test_bw_port_up = 40000 + usernum * 100 + 99


        init_user_class_com= 'tc class add dev eno1 parent 1:2 classid 1:2{} htb rate {}mbit ceil {}mbit'.format(usernum,start_bw ,start_bw)

        init_bw_class_com = "tc class add dev eno1 parent 1:3 classid 1:3{} htb rate {}mbit ceil {}mbit".format(usernum,start_bw, start_bw)


        init_user_sfq_com = "tc qdisc add dev eno1 parent 1:2{} handle 12{}:0 sfq perturb 5".format(usernum,usernum)


        init_bw_sfq_com = "tc qdisc add dev eno1 parent 1:3{} handle 13{}:0 sfq perturb 5".format(usernum,usernum)


        init_user_filter_com = "tc filter add dev eno1 parent 1:0 protocol ip prio 3 handle 12{} fw classid 1:2{}".format(usernum,usernum)
        init_bw_filter_com = 'tc filter add dev eno1 parent 1:0 protocol ip prio 3 handle 13{} fw classid 1:3{}'.format(usernum,usernum)


        init_iptables1 = "iptables -t mangle -A OUTPUT -p tcp --match tcp --sport {}:{} -j MARK --set-mark 12{}".format(user_port_down,user_port_up,usernum)
        init_iptables2 = "iptables -t mangle -A OUTPUT -p tcp --match tcp --sport {}:{} -j RETURN".format(user_port_down,user_port_up)

        init_iptables3 = "iptables -t mangle -A OUTPUT -p tcp --match tcp --sport {}:{} -j MARK --set-mark 13{}".format(test_bw_port_down,test_bw_port_up , usernum)
        init_iptables4 = "iptables -t mangle -A OUTPUT -p tcp --match tcp --sport {}:{} -j RETURN".format(test_bw_port_down,test_bw_port_up ,)

        # print(init_tc1)
        # print(init_tc2)
        # print(init_tc3)
        # print(init_user_class_com)
        # print(init_bw_class_com)
        # print(init_user_sfq_com)
        # print(init_bw_sfq_com)
        # print(init_user_filter_com)
        # print(init_bw_filter_com)
        # print(init_iptables1)
        # print(init_iptables2)
        # print(init_iptables3)
        # print(init_iptables4)

        self.exshell(init_tc1)
        self.exshell(init_tc2)
        self.exshell(init_tc3)

        self.exshell(init_user_class_com)
        self.exshell(init_bw_class_com)
        self.exshell(init_user_sfq_com)
        self.exshell(init_bw_sfq_com)
        self.exshell(init_user_filter_com)
        self.exshell(init_bw_filter_com)
        self.exshell(init_iptables1)
        self.exshell(init_iptables2)
        self.exshell(init_iptables3)
        self.exshell(init_iptables4)


    def exshell(self, command):
        '''
        以root 身份执行
        :param command:
        :return:
        '''
        r = os.popen('echo %s|sudo -S %s' % (self.sudoPassword, command))
        text = r.read()
        print(text)
        r.close()

    def reset_tc_iptables(self):
        '''
        若 TC 操作输出结果 “找不到设备”  说明已删除tc 规则

        若 iptables 操作输出结果找不到 “iptablesRules文件”  则说明当前路径不存在重载文件 "Reset_iptables",
        需按以下方法生成手动删除iptables规则 (Chain OUTPUT 下的规则 )，并生成原始的iptablesRules文件:
            “批量删除 iptables 规则 ：　
            你先用iptables-save > iptablesRules将所有的iptables规则导出到iptableRules文件 ，
            然后你用文件编辑器修改这个文件，将你不想要的所有规则都删掉，保存
            修改完之后运行 iptables-restore < iptablesRules”

        :return:
        '''
        del_tc = "tc qdisc del dev eno1 root"  # 删除根队列 根队列删除后，所有tc class 都会删除
        del_iptables = "iptables-restore < iptablesRules"  # 重载 iptables 规则 ，

        # 先必须将原始的iptables规则 保存到文件   iptables-store > Reset_iptables
        # reset 调用  iptables-restore < Reset_iptables  ，用原始设置覆盖改变的规则

        self.exshell(del_tc)
        self.exshell(del_iptables)

    def change_bandwidth(self, user_num, bandwidth):
        '''
        系统运行时每隔10s / 5s 调用
        :param user_num:
        :param bandwidth:   想要设定的带宽值 单位:mbit/s
        :return:
        '''

        set_user = 'tc class change dev eno1 parent 1:2 classid 1:2{} htb rate {}mbit ceil {}mbit'.format(user_num,bandwidth,bandwidth)
        set_testw = "tc class change dev eno1 parent 1:3 classid 1:3{} htb rate {}mbit ceil {}mbit".format(user_num,bandwidth,bandwidth)

        self.exshell(set_user)
        self.exshell(set_testw)
        print("bandwidth_control :  user_num : {}  up_load bandwidth: {} mbit/s".format(user_num, bandwidth))


if __name__ == '__main__':
    bw_control = Set_sever_bandwidth()
    bw_control.reset_tc_iptables()
    """"""
    usertotle = 10                                       # 用户总数
    for user in range(1,usertotle+1):                    # 初始化管道s
        bw_control.init_tc(user)

    time.sleep(5)

    while(True):

        for user in range (1,usertotle+1):               # 更改每个用户的带宽  ，带宽数据来源： 文件 、 硬数据..
            bw_control.change_bandwidth(user_num=user, bandwidth=30)

        time.sleep(10)                                   # 每隔十秒变动一次

