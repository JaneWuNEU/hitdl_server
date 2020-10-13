import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
import sys
sys.path.append(".")
legend_font = {'family': 'Arial',
               'weight': 'normal',
               'size': 11,

               }

label_font = {'family': "Arial",
              'weight': 'normal',
              'size': 15,
              }

ticks = {'family': 'Arial',
         'weight': 'normal',
         'size': 20,
         }
marker_size = 5
line_width = 3
class OutDate:
    def network_selection(self):
        # 1. read the latency data
        fig, ax = plt.subplots()
        fig.set_size_inches(800 * 1 / 72, 450 / 72)
        '''
        data = pd.read_excel("./draw_experiment.xlsx",sheet_name="network")
        network = data["model_partition"]
        CPU_cores = np.arange(1, 11)
        ax = axes[0]
        ax.plot(CPU_cores, network, "*-", markersize=marker_size)
        ax.set_ylabel('Bandwidth (Mbits/s)', label_font)
        ax.set_xlabel("Model Partition",label_font)
    
        y_major_locator = MultipleLocator(2)
        x_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)
    
        ax.set_xticks(CPU_cores)#set the location of xticks
        #ax.set_xticklabels(CPU_cores) # set the name of each xtick
        #for tick in ax.get_xticklabels(): # to rotate the xticks
            #tick.set_rotation(45)
        ax.tick_params(labelsize=20)
        ax.grid(linestyle="-.")
    
        ax = axes[1]
        CPU_cores = np.arange(1, 7)
        '''
        data = pd.read_excel("./draw_experiment.xlsx", sheet_name="network")
        I_n = data["I_net"]
        R_n = data["R_net"]
        M_n = data["M_net"]
        CPU_cores = np.arange(1, len(I_n) + 1)
        ax.plot(CPU_cores, I_n, "*-", markersize=marker_size, label="Inception")
        ax.plot(CPU_cores, R_n, "s-", markersize=marker_size, label="ResNet")
        ax.plot(CPU_cores, M_n, "^-", markersize=marker_size, label="MobileNet")
        ax.set_ylabel('Bandwidth (Mbits/s)', label_font)
        ax.set_xlabel("Time Slot", label_font)

        y_major_locator = MultipleLocator(2)
        x_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        ax.set_xticks(np.arange(1, len(I_n) + 1))  # set the location of xticks
        # ax.set_xticklabels(CPU_cores) # set the name of each xtick
        # for tick in ax.get_xticklabels(): # to rotate the xticks
        # tick.set_rotation(45)
        ax.tick_params(labelsize=20)
        ax.legend(prop=legend_font, loc='upper right')
        ax.grid(linestyle="-.")
        fig.tight_layout()
        # plt.savefig("./experiment_network.pdf")
        plt.savefig("./experiment_network.png")


    def compare_k(self):
        # 1. read the latency data
        fig, axes = plt.subplots(1, 3)
        fig.set_size_inches(800 * 1 / 72, 450 / 72)
        # data = pd.read_excel("./draw_experiment.xlsx",sheet_name="model_partition_interference")
        data = pd.read_excel("./draw_experiment.xlsx", sheet_name="model_partition")

        # network = data["net"]/ np.min(data["net"])*8
        hitdl_I = data["I_K"]
        hitdl_R = data["R_K"]
        hitdl_M = data["M_K"]

        base_I = data["BI_K"]
        base_R = data["BR_K"]
        base_M = data["BM_K"]

        ax = axes[0]
        CPU_cores = np.arange(1, 11)
        ax.plot(CPU_cores, hitdl_I, "^--", markersize=marker_size, label="HiTDL")
        ax.plot(CPU_cores, base_I, "*--", markersize=marker_size, label="Input")
        # ax01 = ax.twinx()
        # ax.plot(CPU_cores, network, ":", markersize=marker_size, label=" Normalized"+"\n"+"Bandwidth")
        ax.set_ylabel('Partition Layer', label_font)
        ax.set_xlabel('Inception', label_font)

        y_major_locator = MultipleLocator(1)
        x_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        ax.set_xticks(CPU_cores)  # set the location of xticks
        # ax.set_xticklabels(CPU_cores) # set the name of each xtick
        # for tick in ax.get_xticklabels(): # to rotate the xticks
        # tick.set_rotation(45)
        ax.tick_params(labelsize=20)

        # plt.ylim(0, 17)
        # ax.xlim(0.8,10.2)

        ax.legend(prop=legend_font, loc='center right', ncol=1)
        fig.tight_layout()
        # plt.show()
        ax.grid(linestyle="-.")

        ax = axes[1]
        CPU_cores = np.arange(1, 11)
        ax.plot(CPU_cores, hitdl_R, "^--", markersize=marker_size, label="HiTDL")
        ax.plot(CPU_cores, base_R, "*--", markersize=marker_size, label="Input")
        ax.set_ylabel('Partition Layer', label_font)
        ax.set_xlabel('ResNet', label_font)

        y_major_locator = MultipleLocator(3)
        x_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        ax.set_xticks(CPU_cores)  # set the location of xticks
        # ax.set_xticklabels(CPU_cores) # set the name of each xtick
        # for tick in ax.get_xticklabels(): # to rotate the xticks
        # tick.set_rotation(45)
        ax.tick_params(labelsize=20)

        # plt.ylim(0, 16)
        # ax.xlim(0.8,10.2)

        ax.legend(prop=legend_font, loc='center right')
        fig.tight_layout()
        # plt.show()
        ax.grid(linestyle="-.")

        ax = axes[2]
        # temp = ax.twinx()
        CPU_cores = np.arange(1, 11)
        ax.plot(CPU_cores, hitdl_M, "^--", markersize=marker_size, label="HiTDL")
        ax.plot(CPU_cores, base_M, "*--", markersize=marker_size, label="Input")
        # temp.plot(CPU_cores, base_M, "*--", markersize=marker_size, label="Input")
        ax.set_ylabel('Partition Layer', label_font)
        ax.set_xlabel('MobileNet', label_font)

        y_major_locator = MultipleLocator(2)
        x_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        ax.set_xticks(CPU_cores)  # set the location of xticks
        # ax.set_xticklabels(CPU_cores) # set the name of each xtick
        # for tick in ax.get_xticklabels(): # to rotate the xticks
        # tick.set_rotation(45)
        plt.tick_params(labelsize=20)

        # plt.ylim(0, 10)
        # ax.xlim(0.8,10.2)

        ax.legend(prop=legend_font, loc='center right')
        # fig.tight_layout()
        # plt.show()
        plt.grid(linestyle="-.")

        # plt.savefig("./compare_model_partition_k_interfernce.pdf")
        # plt.savefig("./compare_model_partition_k_interference.png")
        plt.savefig("./compare_model_partition_k.pdf")
        plt.savefig("./compare_model_partition_k.png")


    def compare_efficiency(self):
        # 1. read the latency data
        fig, axes = plt.subplots(1, 3)
        fig.set_size_inches(800 * 1 / 72, 450 / 72)
        # data = pd.read_excel("./draw_experiment.xlsx",sheet_name="model_partition_interference")
        data = pd.read_excel("./draw_experiment.xlsx", sheet_name="model_partition")

        hitdl_I = data["I_E"]
        hitdl_R = data["R_E"]
        hitdl_M = data["M_E"]

        base_I = data["BI_E"]
        base_R = data["BR_E"]
        base_M = data["BM_E"]

        ax = axes[0]
        CPU_cores = np.arange(1, 11)
        ax.plot(CPU_cores, hitdl_I, "s--", markersize=marker_size, label="HiTDL")
        ax.plot(CPU_cores, base_I, "*--", markersize=marker_size, label="Input")
        ax.set_ylabel('efficiency', label_font)
        ax.set_xlabel('Inception', label_font)

        y_major_locator = MultipleLocator(5)  # MultipleLocator(5)
        x_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        ax.set_xticks(CPU_cores)  # set the location of xticks
        # ax.set_xticklabels(CPU_cores) # set the name of each xtick
        # for tick in ax.get_xticklabels(): # to rotate the xticks
        # tick.set_rotation(45)
        ax.tick_params(labelsize=20)

        # ax.ylim(0, 3)
        # ax.xlim(0.8,10.2)

        ax.legend(prop=legend_font, loc='lower right')
        fig.tight_layout()
        # plt.show()
        ax.grid(linestyle="-.")

        ax = axes[1]
        CPU_cores = np.arange(1, 11)
        ax.plot(CPU_cores, hitdl_R, "s--", markersize=marker_size, label="HiTDL")
        ax.plot(CPU_cores, base_R, "*--", markersize=marker_size, label="Input")
        ax.set_ylabel('efficiency', label_font)
        ax.set_xlabel('ResNet', label_font)

        y_major_locator = MultipleLocator(0.5)  # MultipleLocator(1)
        x_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        ax.set_xticks(CPU_cores)  # set the location of xticks
        # ax.set_xticklabels(CPU_cores) # set the name of each xtick
        # for tick in ax.get_xticklabels(): # to rotate the xticks
        # tick.set_rotation(45)
        ax.tick_params(labelsize=20)

        # ax.ylim(0, 2)
        # ax.xlim(0.8,10.2)

        ax.legend(prop=legend_font, loc='center right')
        fig.tight_layout()
        # plt.show()
        ax.grid(linestyle="-.")

        ax = axes[2]
        CPU_cores = np.arange(1, 11)
        ax.plot(CPU_cores, hitdl_M, "s--", markersize=marker_size, label="HiTDL")
        ax.plot(CPU_cores, base_M, "*--", markersize=marker_size, label="Input")
        ax.set_ylabel('efficiency', label_font)
        ax.set_xlabel('MobileNet', label_font)

        y_major_locator = MultipleLocator(2)  # MultipleLocator(1)
        x_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        ax.set_xticks(CPU_cores)  # set the location of xticks
        # ax.set_xticklabels(CPU_cores) # set the name of each xtick
        # for tick in ax.get_xticklabels(): # to rotate the xticks
        # tick.set_rotation(45)
        plt.tick_params(labelsize=20)

        # ax.ylim(0, 8)
        # ax.xlim(0.8,10.2)

        ax.legend(prop=legend_font, loc='center right')
        fig.tight_layout()
        # plt.show()
        ax.grid(linestyle="-.")
        # plt.savefig("./compare_model_partition_efficiency_interference.pdf")
        # plt.savefig("./compare_model_partition_efficiency_interference.png")

        plt.savefig("./compare_model_partition_efficiency.pdf")
        plt.savefig("./compare_model_partition_efficiency.png")


    def compare_intra(self):
        # 1. read the latency data
        fig, axes = plt.subplots(1, 3)
        fig.set_size_inches(800 * 1 / 72, 450 / 72)
        # data = pd.read_excel("./draw_experiment.xlsx",sheet_name="model_partition_interference")
        data = pd.read_excel("./draw_experiment.xlsx", sheet_name="model_partition")

        hitdl_I = data["I_intra"]
        hitdl_R = data["R_intra"]
        hitdl_M = data["M_intra"]

        base_I = data["BI_intra"]
        base_R = data["BR_intra"]
        base_M = data["BM_intra"]

        ax = axes[0]
        CPU_cores = np.arange(1, 11)
        ax.plot(CPU_cores, hitdl_I, "o--", markersize=marker_size, label="HiTDL")
        ax.plot(CPU_cores, base_I, "*--", markersize=marker_size, label="Input")
        ax.set_ylabel('CPU Cores', label_font)
        ax.set_xlabel('Inception', label_font)

        y_major_locator = MultipleLocator(1)
        x_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        ax.set_xticks(CPU_cores)  # set the location of xticks
        # ax.set_xticklabels(CPU_cores) # set the name of each xtick
        # for tick in ax.get_xticklabels(): # to rotate the xticks
        # tick.set_rotation(45)
        ax.tick_params(labelsize=20)

        # ax.ylim(0, 4)
        # ax.xlim(0.8,10.2)
        # plt.ylim(0, 3)
        ax.legend(prop=legend_font, loc='upper right')
        fig.tight_layout()
        # plt.show()
        ax.grid(linestyle="-.")

        ax = axes[1]
        CPU_cores = np.arange(1, 11)
        ax.plot(CPU_cores, hitdl_R, "o--", markersize=marker_size, label="HiTDL")
        ax.plot(CPU_cores, base_R, "*--", markersize=marker_size, label="Input")
        ax.set_ylabel('CPU Cores', label_font)
        ax.set_xlabel('ResNet', label_font)

        y_major_locator = MultipleLocator(1)
        x_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        ax.set_xticks(CPU_cores)  # set the location of xticks
        # ax.set_xticklabels(CPU_cores) # set the name of each xtick
        # for tick in ax.get_xticklabels(): # to rotate the xticks
        # tick.set_rotation(45)
        ax.tick_params(labelsize=20)
        # plt.ylim(0, 3)
        # ax.ylim(0, 4)
        # ax.xlim(0.8,10.2)

        ax.legend(prop=legend_font, loc='upper right')
        fig.tight_layout()
        # plt.show()
        ax.grid(linestyle="-.")

        ax = axes[2]
        CPU_cores = np.arange(1, 11)
        ax.plot(CPU_cores, hitdl_M, "o--", markersize=marker_size, label="HiTDL")
        ax.plot(CPU_cores, base_M, "*--", markersize=marker_size, label="Input")
        ax.set_ylabel('CPU Cores', label_font)
        ax.set_xlabel('MobileNet', label_font)

        y_major_locator = MultipleLocator(1)
        x_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        ax.set_xticks(CPU_cores)  # set the location of xticks
        # ax.set_xticklabels(CPU_cores) # set the name of each xtick
        # for tick in ax.get_xticklabels(): # to rotate the xticks
        # tick.set_rotation(45)
        plt.tick_params(labelsize=20)

        # plt.ylim(0, 3)
        # ax.xlim(0.8,10.2)

        ax.legend(prop=legend_font, loc='lower right')
        fig.tight_layout()
        # plt.show()
        ax.grid(linestyle="-.")

        plt.savefig("./compare_model_partition_intra.pdf")
        plt.savefig("./compare_model_partition_intra.png")
        # plt.savefig("./compare_model_partition_intra_interference.pdf")
        # plt.savefig("./compare_model_partition_intra_interference.png")


    def plot_latency_intra(self):
        # 1. read the latency data
        fig, ax = plt.subplots()
        fig.set_size_inches(800 * 1 / 72, 450 / 72)

        inception_latency_avg_name = "../layer_time/inception_CPU=100/2_sigma_inception_CPU=100_model_time_avg_intra_from_1_8_CPU=100.xlsx"
        inception_latency_avg = pd.read_excel(inception_latency_avg_name)["input"].values

        resnet_latency_avg_name = "../layer_time/resnet_CPU=100/2_sigma_resnet_CPU=100_model_time_avg_intra_from_1_8_CPU=100.xlsx"
        resnet_latency_avg = pd.read_excel(resnet_latency_avg_name)["input"].values

        mobilenet_latency_avg_name = "../layer_time/mobilenet_CPU=100/2_sigma_mobilenet_CPU=100_model_time_avg_intra_from_1_8_CPU=100.xlsx"
        mobilenet_latency_avg = pd.read_excel(mobilenet_latency_avg_name)["input"].values

        CPU_cores = np.arange(1, 9)
        plt.plot(CPU_cores, inception_latency_avg, "o--", markersize=marker_size, label="Inception")
        plt.plot(CPU_cores, resnet_latency_avg, "*--", markersize=marker_size, label="ResNet")
        plt.plot(CPU_cores, mobilenet_latency_avg, "s--", markersize=marker_size, label="MobileNet")
        ax.set_ylabel('Latency (s)', label_font)
        ax.set_xlabel('CPU Cores', label_font)

        y_major_locator = MultipleLocator(0.02)
        x_major_locator = MultipleLocator(1)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.xaxis.set_major_locator(x_major_locator)

        ax.set_xticks(CPU_cores)  # set the location of xticks
        ax.set_xticklabels(CPU_cores)  # set the name of each xtick
        # for tick in ax.get_xticklabels(): # to rotate the xticks
        # tick.set_rotation(45)
        plt.tick_params(labelsize=20)

        plt.ylim(0, 0.22)
        plt.xlim(0.8, 8.2)

        ax.legend(prop=legend_font, loc='upper right')
        fig.tight_layout()
        plt.show()
        plt.grid(linestyle="-.")
        plt.savefig("../figures/latency_intra.pdf")

class PlotModelPartition:
    def __init__(self):
        self.layer_nums = {"inception": 20, "resnet": 21, "mobilenet": 16}
    def model_partition_SLA(self,axes):
        hitdl = pd.read_excel("../experiment/model_partition/SLA/max_intra=4/hitdl_partition.xlsx",index_col=0)
        baseline = pd.read_excel("../experiment/model_partition/SLA/max_intra=4/baseline_partition.xlsx",index_col=0)
        neuro = pd.read_excel("../experiment/model_partition/SLA/max_intra=4/neuro_partition.xlsx",index_col=0)
        I_E = axes[0][0]
        I_intra = axes[1][0]
        I_k = axes[2][0]
        I_E.grid(linestyle="-.")
        I_intra.grid(linestyle="-.")
        I_k.grid(linestyle="-.")

        #画出不同切分策略下Inception的切分情况
        x = np.arange(0.7, 1, 0.01)

        I_E.set_xlim(0.7-0.02,1+0.02)
        I_E.set_ylim(0,1.5)
        I_E.xaxis.set_major_locator(MultipleLocator(0.1))
        I_E.yaxis.set_major_locator(MultipleLocator(0.3))
        I_E.plot(x, hitdl.loc[:,"I_E"], "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        I_E.plot(x, baseline.loc[:,"I_E"], ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        I_E.plot(x, neuro.loc[:, "I_E"], "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        I_E.set_ylabel('Efficiency', label_font)
        I_E.set_title("Inception")

        I_intra.set_xlim(0.7-0.02,1+0.02)
        I_intra.set_ylim(0,4.5)
        I_intra.xaxis.set_major_locator(MultipleLocator(0.1))
        I_intra.yaxis.set_major_locator(MultipleLocator(1))
        I_intra.plot(x, hitdl.loc[:,"I_intra"], "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        I_intra.plot(x, baseline.loc[:,"I_intra"], ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        I_intra.plot(x, neuro.loc[:, "I_intra"], "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        I_intra.set_ylabel('CPU Cores', label_font)

        I_k.set_xlim(0.7-0.02,1+0.02)
        I_k.xaxis.set_major_locator(MultipleLocator(0.1))
        I_k.set_ylim(0, 1)
        I_k.yaxis.set_major_locator(MultipleLocator(0.2))
        I_k.plot(x, hitdl.loc[:,"I_k"]/(self.layer_nums["inception"]-1), "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        I_k.plot(x, baseline.loc[:,"I_k"]/(self.layer_nums["inception"]-1), ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        I_k.plot(x, neuro.loc[:, "I_k"]/(self.layer_nums["inception"]-1), "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        I_k.set_ylabel('Partition Index', label_font)
        #I_k.set_xlabel("Network speed (Mbps)",label_font)

        #I_k.legend(prop=legend_font, loc='upper left')

        R_E = axes[0][1]
        R_intra = axes[1][1]
        R_k = axes[2][1]
        R_E.grid(linestyle="-.")
        R_intra.grid(linestyle="-.")
        R_k.grid(linestyle="-.")

        #画出不同切分策略下Inception的切分情况
        R_E.set_xlim(0.7-0.02,1+0.02)
        R_E.xaxis.set_major_locator(MultipleLocator(0.1))
        R_E.set_ylim(0,6)
        R_E.yaxis.set_major_locator(MultipleLocator(1))
        R_E.plot(x, hitdl.loc[:,"R_E"], "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        R_E.plot(x, baseline.loc[:,"R_E"], ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        R_E.plot(x, neuro.loc[:, "R_E"], "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        R_E.set_title("ResNet")
        R_E.legend(prop=legend_font, framealpha=0.1, loc="upper center", bbox_to_anchor=(0, 1.3, 1, 0.2), ncol=3)

        R_intra.set_xlim(0.7-0.02,1+0.02)
        R_intra.xaxis.set_major_locator(MultipleLocator(0.1))
        R_intra.set_ylim(0,4.5)
        R_intra.yaxis.set_major_locator(MultipleLocator(1))
        R_intra.plot(x, hitdl.loc[:,"R_intra"], "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        R_intra.plot(x, baseline.loc[:,"R_intra"], ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        R_intra.plot(x, neuro.loc[:, "R_intra"], "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        #R_intra.set_ylabel('C', label_font)

        R_k.set_xlim(0.7-0.02,1+0.02)
        R_k.xaxis.set_major_locator(MultipleLocator(0.1))
        R_k.set_ylim(0, 1)
        R_k.yaxis.set_major_locator(MultipleLocator(0.2))
        R_k.plot(x, hitdl.loc[:,"R_k"]/(self.layer_nums["resnet"]-1), "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        R_k.plot(x, baseline.loc[:,"R_k"]/(self.layer_nums["resnet"]-1), ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        R_k.plot(x, neuro.loc[:, "R_k"]/(self.layer_nums["resnet"]-1), "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        R_k.set_xlabel("Latency SLA", label_font)


        M_E = axes[0][2]
        M_intra = axes[1][2]
        M_k = axes[2][2]
        M_E.grid(linestyle="-.")
        M_intra.grid(linestyle="-.")
        M_k.grid(linestyle="-.")
        #画出不同切分策略下Inception的切分情况
        M_E.set_xlim(0.7-0.02,1+0.02)
        M_E.xaxis.set_major_locator(MultipleLocator(0.1))
        M_E.set_ylim(0,22)
        M_E.yaxis.set_major_locator(MultipleLocator(4))
        M_E.plot(x, hitdl.loc[:,"M_E"], "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        M_E.plot(x, baseline.loc[:,"M_E"], ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        M_E.plot(x, neuro.loc[:, "M_E"], "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        M_E.set_title('MobileNet')

        M_intra.set_xlim(0.7-0.02,1+0.02)
        M_intra.xaxis.set_major_locator(MultipleLocator(0.1))
        M_intra.set_ylim(0,4.5)
        M_intra.yaxis.set_major_locator(MultipleLocator(1))
        M_intra.plot(x, hitdl.loc[:,"M_intra"], "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        M_intra.plot(x, baseline.loc[:,"M_intra"], ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        M_intra.plot(x, neuro.loc[:, "M_intra"], "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)


        M_k.set_xlim(0.7-0.02,1+0.02)
        M_k.xaxis.set_major_locator(MultipleLocator(0.1))
        M_k.set_ylim(0, 1)
        M_k.yaxis.set_major_locator(MultipleLocator(0.2))
        M_k.plot(x, hitdl.loc[:,"M_k"]/(self.layer_nums["mobilenet"]-1), "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        M_k.plot(x, baseline.loc[:,"M_k"]/(self.layer_nums["mobilenet"]-1), ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        M_k.plot(x, neuro.loc[:, "M_k"]/(self.layer_nums["mobilenet"]-1), "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        #M_k.set_xlabel("Latency SLA", label_font)
        plt.savefig("../figures/model_partition_SLA.pdf")
        plt.savefig("../figures/model_partition_SLA.png")
        plt.show()

    def model_partition_network(self,axes):
        hitdl = pd.read_excel("../experiment/model_partition/network/simulate_trace/max_intra=4/hitdl_partition.xlsx",index_col=0)
        baseline = pd.read_excel("../experiment/model_partition/network/simulate_trace/max_intra=4/baseline_partition.xlsx",index_col=0)
        neuro = pd.read_excel("../experiment/model_partition/network/simulate_trace/max_intra=4/neuro_partition.xlsx",index_col=0)
        I_E = axes[0][0]
        I_intra = axes[1][0]
        I_k = axes[2][0]
        I_E.grid(linestyle="-.")
        I_intra.grid(linestyle="-.")
        I_k.grid(linestyle="-.")

        #画出不同切分策略下Inception的切分情况
        I_E.set_xlim(10 ** 0-0.2, 10 ** 3+0.2)
        I_E.set_xscale('log')
        I_E.set_ylim(0,5)
        I_E.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        I_E.yaxis.set_major_locator(MultipleLocator(1))
        I_E.plot(range(1,1001), hitdl.loc[:,"I_E"], "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        I_E.plot(range(1,1001), baseline.loc[:,"I_E"], ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        I_E.plot(range(1,1001), neuro.loc[:, "I_E"], "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        I_E.set_ylabel('Efficiency', label_font)
        I_E.set_title("Inception")


        I_intra.set_xlim(10 ** 0, 10 ** 3)
        I_intra.set_xscale('log')
        I_intra.set_ylim(0, 4.5)
        I_intra.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        I_intra.yaxis.set_major_locator(MultipleLocator(1))
        I_intra.plot(range(1,1001), hitdl.loc[:,"I_intra"], "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        I_intra.plot(range(1,1001), baseline.loc[:,"I_intra"], ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        I_intra.plot(range(1,1001), neuro.loc[:, "I_intra"], "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        I_intra.set_ylabel('CPU Cores', label_font)


        I_k.set_xlim(10 ** 0-0.2, 10 ** 3+0.2)
        I_k.set_xscale('log')
        I_k.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        I_k.set_ylim(0, 1)
        I_k.yaxis.set_major_locator(MultipleLocator(0.2))
        I_k.plot(range(1,1001), hitdl.loc[:,"I_k"]/(self.layer_nums["inception"]-1), "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        I_k.plot(range(1,1001), baseline.loc[:,"I_k"]/(self.layer_nums["inception"]-1), ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        I_k.plot(range(1,1001), neuro.loc[:, "I_k"]/(self.layer_nums["inception"]-1), "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        I_k.set_ylabel('Partition Index', label_font)


        #I_k.legend(prop=legend_font, loc='upper left')

        R_E = axes[0][1]
        R_intra = axes[1][1]
        R_k = axes[2][1]
        R_E.grid(linestyle="-.")
        R_intra.grid(linestyle="-.")
        R_k.grid(linestyle="-.")

        #画出不同切分策略下Inception的切分情况
        R_E.set_xlim(10 ** 0, 10 ** 3)
        R_E.set_xscale('log')
        R_E.set_ylim(0,6)
        R_E.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        R_E.yaxis.set_major_locator(MultipleLocator(1))
        R_E.plot(range(1,1001), hitdl.loc[:,"R_E"], "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        R_E.plot(range(1,1001), baseline.loc[:,"R_E"], ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        R_E.plot(range(1,1001), neuro.loc[:, "R_E"], "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        R_E.set_title("ResNet")
        R_E.legend(prop=legend_font, framealpha=0.1, loc="upper center", bbox_to_anchor=(0, 1.3, 1, 0.2), ncol=3)


        R_intra.set_xlim(10 ** 0, 10 ** 3)
        R_intra.set_xscale('log')
        R_intra.set_ylim(0,4.5)
        R_intra.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        R_intra.yaxis.set_major_locator(MultipleLocator(1))
        R_intra.plot(range(1,1001), hitdl.loc[:,"R_intra"], "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        R_intra.plot(range(1,1001), baseline.loc[:,"R_intra"], ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        R_intra.plot(range(1,1001), neuro.loc[:, "R_intra"], "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)



        R_k.set_xlim(10 ** 0-0.2, 10 ** 3+0.2)
        R_k.set_xscale('log')
        R_k.set_ylim(0, 1)
        R_k.yaxis.set_major_locator(MultipleLocator(0.2))
        R_k.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        R_k.plot(range(1,1001), hitdl.loc[:,"R_k"]/(self.layer_nums["resnet"]-1), "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        R_k.plot(range(1,1001), baseline.loc[:,"R_k"]/(self.layer_nums["resnet"]-1), ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        R_k.plot(range(1,1001), neuro.loc[:, "R_k"]/(self.layer_nums["resnet"]-1), "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        R_k.set_xlabel("Network Bandwidth (Mbps)", label_font)




        M_E = axes[0][2]
        M_intra = axes[1][2]
        M_k = axes[2][2]
        M_E.grid(linestyle="-.")
        M_intra.grid(linestyle="-.")
        M_k.grid(linestyle="-.")
        #画出不同切分策略下Inception的切分情况
        M_E.set_xlim(10 ** 0, 10 ** 3)
        M_E.set_ylim(0,37)
        M_E.set_xscale('log')
        M_E.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        M_E.yaxis.set_major_locator(MultipleLocator(6))
        M_E.plot(range(1,1001), hitdl.loc[:,"M_E"], "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        M_E.plot(range(1,1001), baseline.loc[:,"M_E"], ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        M_E.plot(range(1,1001), neuro.loc[:, "M_E"], "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)
        M_E.set_title('MobileNet')

        M_intra.set_xlim(10 ** 0, 10 ** 3)
        M_intra.set_ylim(0,4.5)
        M_intra.set_xscale('log')
        M_intra.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        M_intra.yaxis.set_major_locator(MultipleLocator(1))
        M_intra.plot(range(1,1001), hitdl.loc[:,"M_intra"], "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        M_intra.plot(range(1,1001), baseline.loc[:,"M_intra"], ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        M_intra.plot(range(1,1001), neuro.loc[:, "M_intra"], "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)



        M_k.set_xlim(10 ** 0-0.2, 10 ** 3+0.2)
        M_k.set_xscale('log')
        M_k.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        M_k.set_ylim(0, 1)
        M_k.yaxis.set_major_locator(MultipleLocator(0.2))
        M_k.plot(range(1,1001), hitdl.loc[:,"M_k"]/(self.layer_nums["mobilenet"]-1), "-",markersize=marker_size, label="HiTDL",linewidth=line_width)
        M_k.plot(range(1,1001), baseline.loc[:,"M_k"]/(self.layer_nums["mobilenet"]-1), ":", markersize=marker_size, label="Server-only",linewidth=line_width)
        M_k.plot(range(1,1001), neuro.loc[:, "M_k"]/(self.layer_nums["mobilenet"]-1), "--", markersize=marker_size, label="Neurosurgeon",linewidth=line_width)

        #M_k.set_xlabel("Network speed (Mbps)", label_font)
        plt.savefig("../figures/model_partition_network.pdf")
        plt.savefig("../figures/model_partition_network.png")
        plt.show()

    def model_partition_factors(self):
        fig, axes = plt.subplots(3,27,sharex='all')
        self.model_partition_SLA(axes[:,0:3])
        self.model_partition_network(axes[:,3:6])
        self.model_partition_fairness(axes[:, 9:])
pmp = PlotModelPartition()
#pmp.model_partition_I_k_ax.set_yticks(np.linspace(0,1,5))()
pmp.model_partition_factors()