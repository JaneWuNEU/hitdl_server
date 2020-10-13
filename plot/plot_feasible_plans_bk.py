import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'
import math
import pandas as pd
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
import sys
sys.path.append(".")
legend_font = {'family': 'Arial',
               'weight': 'normal',
               'size': 18,

               }

label_font = {'family': "Arial",
              'weight': 'normal',
              'size': 22,
              }

tick_font = {'family': 'Arial',
         'weight': 'normal',
         'size': 20
         }
title_font= {'family': 'Arial',
         'weight': 'normal',
         'size': 22
         }
marker_size = 6
line_width =2

class PlotFeasiblePlan:
    def __init__(self):
        self.layer_nums = {"inception": 20, "resnet": 21, "mobilenet": 16}
    def model_partition_fairness(self):
        def get_info(data,model_name):
            F_data = data["F"]
            F_value = np.around(np.arange(0.1,1.01,0.05), 2)
            F_value_amount = (F_value.shape)[0]
            result = {"k":np.zeros((F_value_amount,self.layer_nums[model_name])),"E":np.zeros((F_value_amount,self.layer_nums[model_name])),
                      "intra":np.zeros((F_value_amount,self.layer_nums[model_name]))}
            plans = np.zeros(F_value_amount)
            F_value_index = 0
            for i in range(len(F_data)):
                if F_data[i]!=F_value[F_value_index]:
                    F_value_index = F_value_index + 1
                result["k"][F_value_index,data.iloc[i,0]]=1
                result["E"][F_value_index, data.iloc[i, 0]] = data.iloc[i, 2]
                result["intra"][F_value_index, data.iloc[i, 0]] = data.iloc[i,1]
                plans[F_value_index] = plans[F_value_index]+1
            #print(result["k"].shape)
            return result
        file_path ="../experiment/model_partition/hitdl_feasible_plans/feasible_plans_fairness.xlsx"
        inception = pd.read_excel(file_path,index_col=0,sheet_name="inception")
        resnet = pd.read_excel(file_path,index_col=0,sheet_name="resnet")
        mobilenet = pd.read_excel(file_path,index_col=0,sheet_name="mobilenet")
        fig,axes = plt.subplots(3,3,sharex="col",sharey="col")
        plt.subplots_adjust(hspace=0.2,wspace=0.5)

        I_grid_info = get_info(inception, "inception")
        R_grid_info = get_info(resnet, "resnet")
        M_grid_info= get_info(mobilenet, "mobilenet")

        x1 = np.around(np.arange(0.1,1.01,0.05), 2)
        x_ticks = np.around(np.arange(0.1, 1.01, 0.3), 1)
        y_I = np.linspace(0, self.layer_nums["inception"],self.layer_nums["inception"])
        x2_I, y2_I = np.meshgrid(x1, y_I)
        y_R = np.linspace(0, self.layer_nums["resnet"],self.layer_nums["resnet"])
        x2_R, y2_R = np.meshgrid(x1, y_R)
        y_M = np.linspace(0, self.layer_nums["mobilenet"],self.layer_nums["mobilenet"])
        x2_M, y2_M = np.meshgrid(x1, y_M)

        I_k_ax = axes[0][0]
        k_max = 1

        c = I_k_ax.pcolormesh(x2_I, y2_I, I_grid_info["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        I_k_ax.set_title("Inception",title_font)
        I_k_ax.set_ylabel("Split Index",label_font)
        I_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],4)))
        I_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 4)).astype(int),tick_font)


        R_k_ax = axes[0][1]
        c = R_k_ax.pcolormesh(x2_R, y2_R, R_grid_info["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        R_k_ax.set_title("ResNet", title_font)
        R_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)
        R_k_ax.tick_params(axis="y",labelsize=tick_font["size"])
        R_k_ax.set_xticks(x_ticks)
        R_k_ax.set_xticklabels(x_ticks,tick_font)#tick_font["size"]

        M_k_ax = axes[0][2]
        c = M_k_ax.pcolormesh(x2_M, y2_M, M_grid_info["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        M_k_ax.set_title("MobileNet", title_font)
        M_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["mobilenet"],4)))
        M_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["mobilenet"], 4)).astype(int),tick_font)
        temp = fig.colorbar(c, ax=axes[0, :],ticks=[0,1])
        temp.ax.set_yticklabels([0,1])
        temp.ax.tick_params(labelsize=tick_font["size"])

        I_intra_ax = axes[1][0]
        I_intra_ax.set_ylabel("Cores", label_font)
        intra_max = 8
        intra_min=0
        c = I_intra_ax.pcolormesh(x2_I, y2_I, I_grid_info["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        I_intra_ax.tick_params(axis="y",labelsize=tick_font["size"])

        R_intra_ax = axes[1][1]
        c = R_intra_ax.pcolormesh(x2_R, y2_R, R_grid_info["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        R_intra_ax.tick_params(axis="y", labelsize=tick_font["size"])
        #R_intra_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        #R_intra_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)

        M_intra_ax = axes[1][2]
        c = M_intra_ax.pcolormesh (x2_M, y2_M, M_grid_info["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        temp = fig.colorbar(c, ax=axes[1,:],ticks=np.arange(intra_min,intra_max+1,4))
        temp.ax.set_yticklabels(np.arange(intra_min,intra_max+1,4))
        temp.ax.tick_params(labelsize=tick_font["size"])
        M_intra_ax.set_ylabel("Layer Index",size=title_font["size"]-4)
        M_intra_ax.yaxis.set_label_position("right")
        M_intra_ax.tick_params(axis="y", labelsize=tick_font["size"])

        I_E_ax = axes[2][0]
        E_max = 7
        I_E_ax.set_ylabel("Efficiency", label_font)
        I_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],3)))
        I_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 3)).astype(int),tick_font)
        c = I_E_ax.pcolormesh(x2_I, y2_I, I_grid_info["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        I_E_ax.set_xticks(x_ticks)
        I_E_ax.set_xticklabels(x_ticks,tick_font)

        R_E_ax = axes[2][1]
        c = R_E_ax.pcolormesh(x2_R, y2_R, R_grid_info["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        R_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)
        R_E_ax.set_xlabel("fairness(%)",title_font)
        R_E_ax.set_xticks(x_ticks)
        R_E_ax.set_xticklabels(x_ticks,tick_font)

        M_E_ax = axes[2][2]
        c = M_E_ax.pcolormesh(x2_M, y2_M, M_grid_info["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        M_E_ax.set_xticks(x_ticks)
        M_E_ax.set_xticklabels(x_ticks,tick_font)
        M_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["mobilenet"],4)))
        M_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["mobilenet"], 4)).astype(int),tick_font)
        temp = fig.colorbar(c, ax=axes[2,:],ticks=np.linspace(0, E_max,3))
        temp.ax.set_yticklabels(np.linspace(0,E_max,5))
        temp.ax.tick_params(axis="both",labelsize=tick_font["size"])
        plt.savefig("../figures/feasible_plans_F.pdf",bbox_inches = 'tight',pad_inches = 0)
        plt.show()
    def model_partition_SLA(self):
        def get_inSLAo(data,model_name):
            SLA_data = data["SLA"]
            SLA_value = np.around(np.arange(0.4, 1.01, 0.01), 2)
            SLA_value_amount = (SLA_value.shape)[0]
            result = {"k":np.zeros((SLA_value_amount,self.layer_nums[model_name])),"E":np.zeros((SLA_value_amount,self.layer_nums[model_name])),
                      "intra":np.zeros((SLA_value_amount,self.layer_nums[model_name]))}
            plans = np.zeros(SLA_value_amount)
            SLA_value = np.around(np.arange(0.4, 1.01, 0.01),2)
            SLA_value_index = 0
            for i in range(len(SLA_data)):
                if SLA_data[i]!=SLA_value[SLA_value_index]:
                    SLA_value_index = SLA_value_index + 1
                result["k"][SLA_value_index,data.iloc[i,0]]=1
                result["E"][SLA_value_index, data.iloc[i, 0]] = data.iloc[i, 2]
                result["intra"][SLA_value_index, data.iloc[i, 0]] = data.iloc[i,1]
                plans[SLA_value_index] = plans[SLA_value_index]+1
            #print(result["k"].shape)
            return result
        SLAile_path ="../experiment/model_partition/hitdl_feasible_plans/feasible_plans_SLA.xlsx"
        inception = pd.read_excel(SLAile_path,index_col=0,sheet_name="inception")
        resnet = pd.read_excel(SLAile_path,index_col=0,sheet_name="resnet")
        mobilenet = pd.read_excel(SLAile_path,index_col=0,sheet_name="mobilenet")
        fig,axes = plt.subplots(3,3,sharex="col",sharey="col")
        plt.subplots_adjust(hspace=0.2,wspace=0.5)

        I_grid_inSLAo = get_inSLAo(inception, "inception")
        R_grid_inSLAo = get_inSLAo(resnet, "resnet")
        M_grid_inSLAo= get_inSLAo(mobilenet, "mobilenet")
        x1 = np.around(np.arange(0.4, 1.01, 0.01), 2)
        x_ticks = np.around(np.arange(0.4, 1.01, 0.2), 2)
        y_I = np.linspace(0, self.layer_nums["inception"],self.layer_nums["inception"])
        x2_I, y2_I = np.meshgrid(x1, y_I)
        y_R = np.linspace(0, self.layer_nums["resnet"],self.layer_nums["resnet"])
        x2_R, y2_R = np.meshgrid(x1, y_R)
        y_M = np.linspace(0, self.layer_nums["mobilenet"],self.layer_nums["mobilenet"])
        x2_M, y2_M = np.meshgrid(x1, y_M)

        I_k_ax = axes[0][0]
        k_max = 1

        c = I_k_ax.pcolormesh(x2_I, y2_I, I_grid_inSLAo["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        I_k_ax.set_title("Inception",label_font)
        I_k_ax.set_ylabel("Split Index",label_font)
        I_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],4)))
        I_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 4)).astype(int),tick_font)


        R_k_ax = axes[0][1]
        c = R_k_ax.pcolormesh(x2_R, y2_R, R_grid_inSLAo["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        R_k_ax.set_title("ResNet", label_font)
        R_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)
        R_k_ax.tick_params(axis="y",labelsize=tick_font["size"])
        R_k_ax.set_xticks(x_ticks)
        R_k_ax.set_xticklabels(x_ticks,tick_font)#tick_font["size"]

        M_k_ax = axes[0][2]
        c = M_k_ax.pcolormesh(x2_M, y2_M, M_grid_inSLAo["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        M_k_ax.set_title("MobileNet", label_font)
        M_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["mobilenet"],4)))
        M_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["mobilenet"], 4)).astype(int),tick_font)
        #temp = fig.colorbar(c, ax=axes[0, :],ticks=[0,1])
        #temp.ax.set_yticklabels([0,1])
        #temp.ax.tick_params(labelsize=tick_font["size"])

        I_intra_ax = axes[1][0]
        I_intra_ax.set_ylabel("Cores", label_font)
        intra_max = 8
        intra_min=0
        c = I_intra_ax.pcolormesh(x2_I, y2_I, I_grid_inSLAo["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        I_intra_ax.tick_params(axis="y",labelsize=tick_font["size"])

        R_intra_ax = axes[1][1]
        c = R_intra_ax.pcolormesh(x2_R, y2_R, R_grid_inSLAo["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        R_intra_ax.tick_params(axis="y", labelsize=tick_font["size"])
        #R_intra_ax.set_yticks(np.floor(np.linspace(0,selSLA.layer_nums["resnet"],3)))
        #R_intra_ax.set_yticklabels(np.floor(np.linspace(0, selSLA.layer_nums["resnet"], 3)).astype(int),tick_font)

        M_intra_ax = axes[1][2]
        c = M_intra_ax.pcolormesh (x2_M, y2_M, M_grid_inSLAo["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        #temp = fig.colorbar(c, ax=axes[1,:],ticks=np.arange(intra_min,intra_max+1,4))
        #temp.ax.set_yticklabels(np.arange(intra_min,intra_max+1,4))
        #temp.ax.tick_params(labelsize=tick_font["size"])
        #M_intra_ax.set_ylabel("Layer Index",size=title_font["size"]-4)
        M_intra_ax.yaxis.set_label_position("right")
        M_intra_ax.tick_params(axis="y", labelsize=tick_font["size"])

        I_E_ax = axes[2][0]
        E_max = 7
        I_E_ax.set_ylabel("Efficiency", label_font)
        I_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],3)))
        I_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 3)).astype(int),tick_font)
        c = I_E_ax.pcolormesh(x2_I, y2_I, I_grid_inSLAo["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        I_E_ax.set_xticks(x_ticks)
        I_E_ax.set_xticklabels(x_ticks,tick_font)

        R_E_ax = axes[2][1]
        c = R_E_ax.pcolormesh(x2_R, y2_R, R_grid_inSLAo["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        R_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)
        R_E_ax.set_xlabel("SLA (%)",title_font)
        R_E_ax.set_xticks(x_ticks)
        R_E_ax.set_xticklabels(x_ticks,tick_font)

        M_E_ax = axes[2][2]
        c = M_E_ax.pcolormesh(x2_M, y2_M, M_grid_inSLAo["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        M_E_ax.set_xticks(x_ticks)
        M_E_ax.set_xticklabels(x_ticks,tick_font)
        M_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["mobilenet"],4)))
        M_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["mobilenet"], 4)).astype(int),tick_font)
        #temp = fig.colorbar(c, ax=axes[2,:],ticks=np.linspace(0, E_max,3))
        #temp.ax.set_yticklabels(np.linspace(0,E_max,5))
        #temp.ax.tick_params(axis="both",labelsize=tick_font["size"])
        plt.savefig("../figures/SLAeasible_plans_SLA.pdf",bbox_inches = 'tight',pad_inches = 0)
        plt.show()

    def model_partition_network(self):
        def get_info(data,model_name):
            net_data = data["net"]
            result = {"k":np.zeros((1000,self.layer_nums[model_name])),"E":np.zeros((1000,self.layer_nums[model_name])),
                      "intra":np.zeros((1000,self.layer_nums[model_name]))}
            plans = np.zeros(1000)
            for i in range(len(net_data)):
                result["k"][net_data[i]-1,data.iloc[i,0]]=1
                result["E"][net_data[i] - 1, data.iloc[i, 0]] = data.iloc[i, 2]
                result["intra"][net_data[i] - 1, data.iloc[i, 0]] = data.iloc[i,1]
                plans[net_data[i]-1] = plans[net_data[i]-1]+1
            return result,plans
        file_path ="../experiment/model_partition/hitdl_feasible_plans/feasible_plans_network.xlsx"
        inception = pd.read_excel(file_path,index_col=0,sheet_name="inception")
        resnet = pd.read_excel(file_path,index_col=0,sheet_name="resnet")
        mobilenet = pd.read_excel(file_path,index_col=0,sheet_name="mobilenet")
        fig,axes = plt.subplots(3,3,sharex="all",sharey="col")
        plt.subplots_adjust(hspace=0.2,wspace=0.5)

        plt.tick_params(labelsize=tick_font["size"])

        I_grid_info, I_plans = get_info(inception, "inception")
        R_grid_info, R_plans = get_info(resnet, "resnet")
        M_grid_info, M_plans = get_info(mobilenet, "mobilenet")

        I_k_ax = axes[0][0]
        k_max = 1
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["inception"],self.layer_nums["inception"])
        x2, y2 = np.meshgrid(x1, y1)
        c = I_k_ax.pcolormesh (x2, y2, I_grid_info["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        I_k_ax.set_title("Inception",label_font)
        #I_k_ax.set_ylabel("Split Index",label_font)
        I_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],4)))
        I_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 4)).astype(int),tick_font)
        I_k_ax.set_xlim(10 ** 0-0.2, 10 ** 3+0.2)
        I_k_ax.set_xscale('log')
        I_k_ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        I_k_ax.tick_params(axis="x",labelsize=tick_font["size"])
        #I_k_ax.grid(linestyle="-.")

        R_k_ax = axes[0][1]
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["resnet"],self.layer_nums["resnet"])
        x2, y2 = np.meshgrid(x1, y1)
        c = R_k_ax.pcolormesh (x2, y2, R_grid_info["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        R_k_ax.set_title("ResNet", label_font)
        R_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)
        R_k_ax.tick_params(axis="y",labelsize=tick_font["size"])

        M_k_ax = axes[0][2]
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["mobilenet"],self.layer_nums["mobilenet"])
        x2, y2 = np.meshgrid(x1, y1)
        c = M_k_ax.pcolormesh (x2, y2, M_grid_info["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        M_k_ax.set_title("MobileNet", label_font)
        M_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["mobilenet"],4)))
        M_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["mobilenet"], 4)).astype(int),tick_font)
        temp = fig.colorbar(c, ax=axes[0, :],ticks=[0,1])
        #temp.mappable.set_clim(0,k_max)
        temp.ax.set_yticklabels([0,1])
        temp.ax.tick_params(labelsize=tick_font["size"])

        I_intra_ax = axes[1][0]
        I_intra_ax.set_ylabel("Cores", label_font)
        intra_max = 8
        intra_min=0
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["inception"],self.layer_nums["inception"])
        x2, y2 = np.meshgrid(x1, y1)
        c = I_intra_ax.pcolormesh(x2, y2, I_grid_info["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        I_intra_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],3)))
        I_intra_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 3)).astype(int),tick_font)

        R_intra_ax = axes[1][1]
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["resnet"],self.layer_nums["resnet"])
        x2, y2 = np.meshgrid(x1, y1)
        c = R_intra_ax.pcolormesh(x2, y2, R_grid_info["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        R_intra_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_intra_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)

        M_intra_ax = axes[1][2]
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["mobilenet"],self.layer_nums["mobilenet"])
        x2, y2 = np.meshgrid(x1, y1)
        c = M_intra_ax.pcolormesh (x2, y2, M_grid_info["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        temp = fig.colorbar(c, ax=axes[1,:],ticks=np.arange(intra_min,intra_max+1,4))
        temp.ax.set_yticklabels(np.arange(intra_min,intra_max+1,4))
        temp.ax.tick_params(labelsize=tick_font["size"])
        M_intra_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["mobilenet"],3)))
        M_intra_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["mobilenet"], 3)).astype(int),tick_font)
        M_intra_ax.set_ylabel("Layer Index",size=title_font["size"]-4)
        M_intra_ax.yaxis.set_label_position("right")

        I_E_ax = axes[2][0]
        E_max = 7
        I_E_ax.set_ylabel("Efficiency", label_font)
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["inception"],self.layer_nums["inception"])
        x2, y2 = np.meshgrid(x1, y1)
        c = I_E_ax.pcolormesh(x2, y2, I_grid_info["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        I_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],3)))
        I_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 3)).astype(int),tick_font)
        I_E_ax.tick_params(axis="y", labelsize=tick_font["size"])

        R_E_ax = axes[2][1]
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["resnet"],self.layer_nums["resnet"])
        x2, y2 = np.meshgrid(x1, y1)
        c = R_E_ax.pcolormesh(x2, y2, R_grid_info["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        R_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)
        R_E_ax.tick_params(axis="y", labelsize=tick_font["size"])
        R_E_ax.set_xlabel("Network Bandwidth (Mbps)",title_font)

        M_E_ax = axes[2][2]
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["mobilenet"],self.layer_nums["mobilenet"])
        x2, y2 = np.meshgrid(x1, y1)
        c = M_E_ax.pcolormesh(x2, y2, M_grid_info["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        temp = fig.colorbar(c, ax=axes[2,:],ticks=np.linspace(0, E_max,3))
        temp.ax.set_yticklabels(np.linspace(0,E_max,5))
        temp.ax.tick_params(labelsize=tick_font["size"])
        M_E_ax.tick_params(axis="y", labelsize=tick_font["size"])
        plt.savefig("../figures/feasible_plans_network.pdf",bbox_inches = 'tight',pad_inches = 0)
        #plt.show()

class PlotFeasiblePlanNew:
    def __init__(self):
        self.layer_nums = {"inception": 20, "resnet": 21, "mobilenet": 16}
    def model_partition_fairness(self):
        def get_info(data,model_name):
            F_data = data["F"]
            F_value = np.around(np.arange(0.1,1.01,0.05), 2)
            F_value_amount = (F_value.shape)[0]
            result = {"k":np.zeros((F_value_amount,self.layer_nums[model_name])),"E":np.zeros((F_value_amount,self.layer_nums[model_name])),
                      "intra":np.zeros((F_value_amount,self.layer_nums[model_name]))}
            plans = np.zeros(F_value_amount)
            F_value_index = 0
            for i in range(len(F_data)):
                if F_data[i]!=F_value[F_value_index]:
                    F_value_index = F_value_index + 1
                result["k"][F_value_index,data.iloc[i,0]]=1
                result["E"][F_value_index, data.iloc[i, 0]] = data.iloc[i, 2]
                result["intra"][F_value_index, data.iloc[i, 0]] = data.iloc[i,1]
                plans[F_value_index] = plans[F_value_index]+1
            #print(result["k"].shape)
            return result
        file_path ="../experiment/model_partition/hitdl_feasible_plans/feasible_plans_fairness.xlsx"
        inception = pd.read_excel(file_path,index_col=0,sheet_name="inception")
        resnet = pd.read_excel(file_path,index_col=0,sheet_name="resnet")
        mobilenet = pd.read_excel(file_path,index_col=0,sheet_name="mobilenet")
        fig,axes = plt.subplots(3,3,sharex="col",sharey="col")
        plt.rcParams["figure.figsize"] = [8, 6.5]
        plt.subplots_adjust(wspace=0.4,hspace=0.2)

        I_grid_info = get_info(inception, "inception")
        R_grid_info = get_info(resnet, "resnet")
        M_grid_info= get_info(mobilenet, "mobilenet")

        x1 = np.around(np.arange(0.1,1.01,0.05), 2)
        x_ticks = np.around(np.arange(0.1, 1.01, 0.4), 1)
        y_I = np.linspace(0, self.layer_nums["inception"],self.layer_nums["inception"])
        x2_I, y2_I = np.meshgrid(x1, y_I)
        y_R = np.linspace(0, self.layer_nums["resnet"],self.layer_nums["resnet"])
        x2_R, y2_R = np.meshgrid(x1, y_R)
        y_M = np.linspace(0, self.layer_nums["mobilenet"],self.layer_nums["mobilenet"])
        x2_M, y2_M = np.meshgrid(x1, y_M)

        I_k_ax = axes[0][0]
        k_max = 1

        c = I_k_ax.pcolormesh(x2_I, y2_I, I_grid_info["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        I_k_ax.set_title("Inception",title_font)
        #I_k_ax.set_yticks([])
        #I_k_ax.set_ylabel("Split Index",label_font)
        I_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],4)))
        I_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 4)).astype(int),tick_font)


        R_k_ax = axes[0][1]
        c = R_k_ax.pcolormesh(x2_R, y2_R, R_grid_info["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        R_k_ax.set_title("ResNet", title_font)
        #R_k_ax.set_yticks([])
        R_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)
        #R_k_ax.tick_params(axis="y",labelsize=tick_font["size"])
        #R_k_ax.set_xticks(x_ticks)
        #R_k_ax.set_xticklabels(x_ticks,tick_font)#tick_font["size"]

        M_k_ax = axes[0][2]
        c = M_k_ax.pcolormesh(x2_M, y2_M, M_grid_info["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        M_k_ax.set_title("MobileNet", title_font)
        M_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["mobilenet"],4)))
        M_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["mobilenet"], 4)).astype(int),tick_font)
        temp = fig.colorbar(c, ax=axes[0, :],ticks=[0,1])
        temp.ax.set_yticklabels([0,1])
        temp.ax.tick_params(labelsize=tick_font["size"])

        I_intra_ax = axes[1][0]
        #I_intra_ax.set_ylabel("Cores", label_font)
        intra_max = 8
        intra_min=0
        c = I_intra_ax.pcolormesh(x2_I, y2_I, I_grid_info["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        I_intra_ax.tick_params(axis="y",labelsize=tick_font["size"])

        R_intra_ax = axes[1][1]
        c = R_intra_ax.pcolormesh(x2_R, y2_R, R_grid_info["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        R_intra_ax.tick_params(axis="y", labelsize=tick_font["size"])
        #R_intra_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        #R_intra_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)

        M_intra_ax = axes[1][2]
        c = M_intra_ax.pcolormesh (x2_M, y2_M, M_grid_info["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        temp = fig.colorbar(c, ax=axes[1,:],ticks=np.arange(intra_min,intra_max+1,4))
        temp.ax.set_yticklabels(np.arange(intra_min,intra_max+1,4))
        temp.ax.tick_params(labelsize=tick_font["size"])
        M_intra_ax.set_ylabel("Layer Index",size=title_font["size"]-2)
        M_intra_ax.yaxis.set_label_position("right")
        M_intra_ax.tick_params(axis="y", labelsize=tick_font["size"])

        I_E_ax = axes[2][0]
        E_max = 7
        #I_E_ax.set_ylabel("Efficiency", label_font)
        I_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],3)))
        I_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 3)).astype(int),tick_font)
        c = I_E_ax.pcolormesh(x2_I, y2_I, I_grid_info["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        I_E_ax.set_xticks(x_ticks)
        I_E_ax.set_xticklabels(x_ticks,tick_font)

        R_E_ax = axes[2][1]
        c = R_E_ax.pcolormesh(x2_R, y2_R, R_grid_info["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        R_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)
        R_E_ax.set_xlabel("fairness(%)",title_font)
        R_E_ax.set_xticks(x_ticks)
        R_E_ax.set_xticklabels(x_ticks,tick_font)

        M_E_ax = axes[2][2]
        c = M_E_ax.pcolormesh(x2_M, y2_M, M_grid_info["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        M_E_ax.set_xticks(x_ticks)
        M_E_ax.set_xticklabels(x_ticks,tick_font)
        M_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["mobilenet"],4)))
        M_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["mobilenet"], 4)).astype(int),tick_font)
        temp = fig.colorbar(c, ax=axes[2,:],ticks=np.linspace(0, E_max,3))
        temp.ax.set_yticklabels(np.linspace(0,E_max,5))
        temp.ax.tick_params(axis="both",labelsize=tick_font["size"])
        #plt.tight_layout()
        plt.savefig("../figures/feasible_plans_F.pdf",bbox_inches = 'tight',pad_inches = 0)#D:/current_system/server/figures/feasible_plans_SLA.pdf
        plt.show()
    def model_partition_SLA(self):
        def get_inSLAo(data,model_name):
            SLA_data = data["SLA"]
            SLA_value = np.around(np.arange(0.4, 1.01, 0.01), 2)
            SLA_value_amount = (SLA_value.shape)[0]
            result = {"k":np.zeros((SLA_value_amount,self.layer_nums[model_name])),"E":np.zeros((SLA_value_amount,self.layer_nums[model_name])),
                      "intra":np.zeros((SLA_value_amount,self.layer_nums[model_name]))}
            plans = np.zeros(SLA_value_amount)
            SLA_value = np.around(np.arange(0.4, 1.01, 0.01),2)
            SLA_value_index = 0
            for i in range(len(SLA_data)):
                if SLA_data[i]!=SLA_value[SLA_value_index]:
                    SLA_value_index = SLA_value_index + 1
                result["k"][SLA_value_index,data.iloc[i,0]]=1
                result["E"][SLA_value_index, data.iloc[i, 0]] = data.iloc[i, 2]
                result["intra"][SLA_value_index, data.iloc[i, 0]] = data.iloc[i,1]
                plans[SLA_value_index] = plans[SLA_value_index]+1
            #print(result["k"].shape)
            return result
        SLAile_path ="../experiment/model_partition/hitdl_feasible_plans/feasible_plans_SLA.xlsx"
        inception = pd.read_excel(SLAile_path,index_col=0,sheet_name="inception")
        resnet = pd.read_excel(SLAile_path,index_col=0,sheet_name="resnet")
        mobilenet = pd.read_excel(SLAile_path,index_col=0,sheet_name="mobilenet")
        fig,axes = plt.subplots(3,3,sharex="col",sharey="col")
        plt.rcParams["figure.figsize"] = [8, 6.5]
        plt.subplots_adjust(wspace=0.4,hspace=0.2)

        I_grid_inSLAo = get_inSLAo(inception, "inception")
        R_grid_inSLAo = get_inSLAo(resnet, "resnet")
        M_grid_inSLAo= get_inSLAo(mobilenet, "mobilenet")
        x1 = np.around(np.arange(0.4, 1.01, 0.01), 2)
        x_ticks = np.around(np.arange(0.4, 1.01, 0.3), 1)
        y_I = np.linspace(0, self.layer_nums["inception"],self.layer_nums["inception"])
        x2_I, y2_I = np.meshgrid(x1, y_I)
        y_R = np.linspace(0, self.layer_nums["resnet"],self.layer_nums["resnet"])
        x2_R, y2_R = np.meshgrid(x1, y_R)
        y_M = np.linspace(0, self.layer_nums["mobilenet"],self.layer_nums["mobilenet"])
        x2_M, y2_M = np.meshgrid(x1, y_M)

        I_k_ax = axes[0][0]
        k_max = 1

        c = I_k_ax.pcolormesh(x2_I, y2_I, I_grid_inSLAo["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        I_k_ax.set_title("Inception",label_font)
        I_k_ax.set_ylabel("Split Index",label_font)
        I_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],4)))
        I_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 4)).astype(int),tick_font)


        R_k_ax = axes[0][1]
        c = R_k_ax.pcolormesh(x2_R, y2_R, R_grid_inSLAo["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        R_k_ax.set_title("ResNet", label_font)
        R_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)
        R_k_ax.tick_params(axis="y",labelsize=tick_font["size"])
        R_k_ax.set_xticks(x_ticks)
        R_k_ax.set_xticklabels(x_ticks,tick_font)#tick_font["size"]

        M_k_ax = axes[0][2]
        c = M_k_ax.pcolormesh(x2_M, y2_M, M_grid_inSLAo["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        M_k_ax.set_title("MobileNet", label_font)
        M_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["mobilenet"],4)))
        M_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["mobilenet"], 4)).astype(int),tick_font)
        #temp = fig.colorbar(c, ax=axes[0, :],ticks=[0,1])
        #temp.ax.set_yticklabels([0,1])
        #temp.ax.tick_params(labelsize=tick_font["size"])

        I_intra_ax = axes[1][0]
        I_intra_ax.set_ylabel("Cores", label_font)
        intra_max = 8
        intra_min=0
        c = I_intra_ax.pcolormesh(x2_I, y2_I, I_grid_inSLAo["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        I_intra_ax.tick_params(axis="y",labelsize=tick_font["size"])

        R_intra_ax = axes[1][1]
        c = R_intra_ax.pcolormesh(x2_R, y2_R, R_grid_inSLAo["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        R_intra_ax.tick_params(axis="y", labelsize=tick_font["size"])
        #R_intra_ax.set_yticks(np.floor(np.linspace(0,selSLA.layer_nums["resnet"],3)))
        #R_intra_ax.set_yticklabels(np.floor(np.linspace(0, selSLA.layer_nums["resnet"], 3)).astype(int),tick_font)

        M_intra_ax = axes[1][2]
        c = M_intra_ax.pcolormesh (x2_M, y2_M, M_grid_inSLAo["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        #temp = fig.colorbar(c, ax=axes[1,:],ticks=np.arange(intra_min,intra_max+1,4))
        #temp.ax.set_yticklabels(np.arange(intra_min,intra_max+1,4))
        #temp.ax.tick_params(labelsize=tick_font["size"])
        #M_intra_ax.set_ylabel("Layer Index",size=title_font["size"]-4)
        M_intra_ax.yaxis.set_label_position("right")
        M_intra_ax.tick_params(axis="y", labelsize=tick_font["size"])

        I_E_ax = axes[2][0]
        E_max = 7
        I_E_ax.set_ylabel("Efficiency", label_font)
        I_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],3)))
        I_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 3)).astype(int),tick_font)
        c = I_E_ax.pcolormesh(x2_I, y2_I, I_grid_inSLAo["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        I_E_ax.set_xticks(x_ticks)
        I_E_ax.set_xticklabels(x_ticks,tick_font)

        R_E_ax = axes[2][1]
        c = R_E_ax.pcolormesh(x2_R, y2_R, R_grid_inSLAo["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        R_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)
        R_E_ax.set_xlabel("SLA (%)",title_font)
        R_E_ax.set_xticks(x_ticks)
        R_E_ax.set_xticklabels(x_ticks,tick_font)

        M_E_ax = axes[2][2]
        c = M_E_ax.pcolormesh(x2_M, y2_M, M_grid_inSLAo["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        M_E_ax.set_xticks(x_ticks)
        M_E_ax.set_xticklabels(x_ticks,tick_font)
        M_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["mobilenet"],4)))
        M_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["mobilenet"], 4)).astype(int),tick_font)
        #temp = fig.colorbar(c, ax=axes[2,:],ticks=np.linspace(0, E_max,3))
        #temp.ax.set_yticklabels(np.linspace(0,E_max,5))
        #temp.ax.tick_params(axis="both",labelsize=tick_font["size"])
        plt.tight_layout()
        plt.savefig("../figures/feasible_plans_SLA.pdf",bbox_inches = 'tight',pad_inches = 0)#
        plt.show()

    def model_partition_network(self):
        def get_info(data,model_name):
            net_data = data["net"]
            result = {"k":np.zeros((1000,self.layer_nums[model_name])),"E":np.zeros((1000,self.layer_nums[model_name])),
                      "intra":np.zeros((1000,self.layer_nums[model_name]))}
            plans = np.zeros(1000)
            for i in range(len(net_data)):
                result["k"][net_data[i]-1,data.iloc[i,0]]=1
                result["E"][net_data[i] - 1, data.iloc[i, 0]] = data.iloc[i, 2]
                result["intra"][net_data[i] - 1, data.iloc[i, 0]] = data.iloc[i,1]
                plans[net_data[i]-1] = plans[net_data[i]-1]+1
            return result,plans
        file_path ="../experiment/model_partition/hitdl_feasible_plans/feasible_plans_network.xlsx"
        inception = pd.read_excel(file_path,index_col=0,sheet_name="inception")
        resnet = pd.read_excel(file_path,index_col=0,sheet_name="resnet")
        mobilenet = pd.read_excel(file_path,index_col=0,sheet_name="mobilenet")
        fig,axes = plt.subplots(3,3,sharex="all",sharey="col")
        plt.rcParams["figure.figsize"] = [8, 6.5]
        plt.subplots_adjust(wspace=0.4,hspace=0.2)

        plt.tick_params(labelsize=tick_font["size"])

        I_grid_info, I_plans = get_info(inception, "inception")
        R_grid_info, R_plans = get_info(resnet, "resnet")
        M_grid_info, M_plans = get_info(mobilenet, "mobilenet")

        I_k_ax = axes[0][0]
        k_max = 1
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["inception"],self.layer_nums["inception"])
        x2, y2 = np.meshgrid(x1, y1)
        c = I_k_ax.pcolormesh (x2, y2, I_grid_info["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        I_k_ax.set_title("Inception",label_font)
        #I_k_ax.set_ylabel("Split Index",label_font)
        I_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],4)))
        I_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 4)).astype(int),tick_font)
        I_k_ax.set_xlim(10 ** 0-0.2, 10 ** 3+0.2)
        I_k_ax.set_xscale('log')
        I_k_ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        I_k_ax.tick_params(axis="x",labelsize=tick_font["size"])
        #I_k_ax.grid(linestyle="-.")

        R_k_ax = axes[0][1]
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["resnet"],self.layer_nums["resnet"])
        x2, y2 = np.meshgrid(x1, y1)
        c = R_k_ax.pcolormesh (x2, y2, R_grid_info["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        R_k_ax.set_title("ResNet", label_font)
        R_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)
        R_k_ax.tick_params(axis="y",labelsize=tick_font["size"])

        M_k_ax = axes[0][2]
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["mobilenet"],self.layer_nums["mobilenet"])
        x2, y2 = np.meshgrid(x1, y1)
        c = M_k_ax.pcolormesh (x2, y2, M_grid_info["k"].T, cmap='Greens', vmin=0, vmax=k_max)
        M_k_ax.set_title("MobileNet", label_font)
        M_k_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["mobilenet"],4)))
        M_k_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["mobilenet"], 4)).astype(int),tick_font)
        #temp = fig.colorbar(c, ax=axes[0, :],ticks=[0,1])
        #temp.mappable.set_clim(0,k_max)
        #temp.ax.set_yticklabels([0,1])
        #temp.ax.tick_params(labelsize=tick_font["size"])

        I_intra_ax = axes[1][0]
        #I_intra_ax.set_ylabel("Cores", label_font)
        intra_max = 8
        intra_min=0
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["inception"],self.layer_nums["inception"])
        x2, y2 = np.meshgrid(x1, y1)
        c = I_intra_ax.pcolormesh(x2, y2, I_grid_info["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        I_intra_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],3)))
        I_intra_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 3)).astype(int),tick_font)

        R_intra_ax = axes[1][1]
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["resnet"],self.layer_nums["resnet"])
        x2, y2 = np.meshgrid(x1, y1)
        c = R_intra_ax.pcolormesh(x2, y2, R_grid_info["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        R_intra_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_intra_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)

        M_intra_ax = axes[1][2]
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["mobilenet"],self.layer_nums["mobilenet"])
        x2, y2 = np.meshgrid(x1, y1)
        c = M_intra_ax.pcolormesh (x2, y2, M_grid_info["intra"].T, cmap='Blues', vmin=intra_min, vmax=intra_max)
        #temp = fig.colorbar(c, ax=axes[1,:],ticks=np.arange(intra_min,intra_max+1,4))
        #temp.ax.set_yticklabels(np.arange(intra_min,intra_max+1,4))
        #temp.ax.tick_params(labelsize=tick_font["size"])
        M_intra_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["mobilenet"],3)))
        M_intra_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["mobilenet"], 3)).astype(int),tick_font)
        #M_intra_ax.set_ylabel("Layer Index",size=title_font["size"]-4)
        M_intra_ax.yaxis.set_label_position("right")

        I_E_ax = axes[2][0]
        E_max = 7
        #I_E_ax.set_ylabel("Efficiency", label_font)
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["inception"],self.layer_nums["inception"])
        x2, y2 = np.meshgrid(x1, y1)
        c = I_E_ax.pcolormesh(x2, y2, I_grid_info["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        I_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["inception"],3)))
        I_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["inception"], 3)).astype(int),tick_font)
        I_E_ax.tick_params(axis="both", labelsize=tick_font["size"])

        R_E_ax = axes[2][1]
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["resnet"],self.layer_nums["resnet"])
        x2, y2 = np.meshgrid(x1, y1)
        c = R_E_ax.pcolormesh(x2, y2, R_grid_info["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        R_E_ax.set_yticks(np.floor(np.linspace(0,self.layer_nums["resnet"],3)))
        R_E_ax.set_yticklabels(np.floor(np.linspace(0, self.layer_nums["resnet"], 3)).astype(int),tick_font)
        R_E_ax.tick_params(axis="both", labelsize=tick_font["size"])
        R_E_ax.set_xlabel("Network Bandwidth (Mbps)",title_font)

        M_E_ax = axes[2][2]
        x1 = np.linspace(1, 1000, 1000)
        y1 = np.linspace(0, self.layer_nums["mobilenet"],self.layer_nums["mobilenet"])
        x2, y2 = np.meshgrid(x1, y1)
        c = M_E_ax.pcolormesh(x2, y2, M_grid_info["E"].T, cmap='Oranges', vmin=0, vmax=E_max)
        #temp = fig.colorbar(c, ax=axes[2,:],ticks=np.linspace(0, E_max,3))
        #temp.ax.set_yticklabels(np.linspace(0,E_max,5))
        #temp.ax.tick_params(labelsize=tick_font["size"])
        M_E_ax.tick_params(axis="y", labelsize=tick_font["size"])
        plt.savefig("../figures/feasible_plans_network.pdf",bbox_inches = 'tight',pad_inches = 0)
        #plt.show()
pmp = PlotFeasiblePlanNew()
#pmp.model_partition_SLA()
pmp.model_partition_network()
#pmp.model_partition_fairness()