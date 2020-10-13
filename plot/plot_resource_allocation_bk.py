from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as ticker
legend_font = {'family': 'Arial',
               'weight': 'normal',
               'size': 18,

               }

label_font = {'family': "Arial",
              'weight': 'normal',
              'size': 20,
              }

tick_font = {'family': 'Arial',
         'weight': 'normal',
         'size': 18
         }
title_font= {'family': 'Arial',
         'weight': 'normal',
         'size': 16
         }
marker_size = 6
line_width =2

marker_shape = ['o','v','^','<','>','s','p','*']
color_style = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
line_style = ["-", "--", "-.", ":",  "-", "--", "-."]
marker_style=[marker_shape[0],"",marker_shape[2],"",marker_shape[4],"",marker_shape[6]]
class PlotResourceAllocation:
    def __init__(self):
        self.legend_name=["MCKP + Multi-partitions","MCKP + Efficient-partition","MCKP + Neurosurgeon","MCKP + Input-partition",
                "Weighted + Efficient-partition","Weighted + Input-partition","Weighted + Neurosurgeon"]
    def plot_weight_scatter(self):
        greedy = pd.read_excel("../experiment/resource_allocation/model_weight/hitdl_greedy.xlsx",sheet_name="model_weight")
        optimal = pd.read_excel("../experiment/resource_allocation/model_weight/hitdl_optimal.xlsx",sheet_name="model_weight")
        weighted = pd.read_excel("../experiment/resource_allocation/model_weight/hitdl_weighted.xlsx",sheet_name="model_weight")

        # 读取数据
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = greedy["I_W"].values #I_W
        y = greedy["R_W"].values #R_W
        z = 1-x-y # M_W
        c = greedy["sys_u"].values
        c2 = weighted["sys_u"].values
        img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
        fig.colorbar(img)
        #ax.scatter(x, y, z, c=c2, cmap=plt.get_cmap('Blues'))

        plt.show()

    def plot_weight_surface(self):
        greedy = pd.read_excel("../experiment/resource_allocation/model_weight/hitdl_greedy.xlsx",sheet_name="model_weight")
        optimal = pd.read_excel("../experiment/resource_allocation/model_weight/hitdl_optimal.xlsx",sheet_name="model_weight")
        weighted = pd.read_excel("../experiment/resource_allocation/model_weight/hitdl_weighted.xlsx",sheet_name="model_weight")

        # 读取数据
        x = greedy["I_W"].values #I_W
        y = greedy["R_W"].values #R_W
        z = 1-x-y # M_W
        c = greedy["sys_u"].values
        x1 = np.linspace(x.min(), x.max(), len(np.unique(x)))
        y1 = np.linspace(y.min(), y.max(), len(np.unique(y)))
        x2, y2 = np.meshgrid(x1, y1)
        z2 = griddata((x, y), z, (x2, y2), method='cubic', fill_value=0)
        z2[z2 < z.min()] = z.min()

        c2 = griddata((x, y), c, (x2, y2), method='cubic', fill_value=0)
        c2[c2 < c.min()] = c.min()

        # --------
        color_dimension = c2  # It must be in 2D - as for "X, Y, Z".
        minn, maxx = color_dimension.min(), color_dimension.max()
        norm = matplotlib.colors.Normalize(minn, maxx)
        m = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Blues'))
        m.set_array([])
        fcolors = m.to_rgba(color_dimension)

        # At this time, X-Y-Z-C are all 2D and we can use "plot_surface".
        fig = plt.figure();
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x2, y2, z2, facecolors=fcolors, linewidth=0, rstride=1, cstride=1,
                               antialiased=False)
        cbar = fig.colorbar(m, shrink=0.5, aspect=5)
        cbar.ax.get_yaxis().labelpad = 15
        plt.show()

    def plot_sys_utility_SLA(self,axes):
        item = "SLA"
        H_M = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_M_"+item+".xlsx",index_col=0)
        H_E = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_E_"+item+".xlsx",index_col=0)
        H_N = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_NS_"+item+".xlsx",index_col=0)
        H_I = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_I_"+item+".xlsx", index_col=0)
        W_E = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_E_"+item+".xlsx",index_col=0)
        W_N = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_NS_"+item+".xlsx",index_col=0)
        W_I = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_I_"+item+".xlsx", index_col=0)
        x = np.arange(0.1,1.01,0.05)
        axes.set_xlim(0.1,1.01)
        axes.xaxis.set_major_locator(MultipleLocator(0.1))
        axes.set_ylim(0-1, 40)
        axes.yaxis.set_major_locator(MultipleLocator(10))
        axes.grid(linestyle="-.")
        axes.plot(x, H_M.loc[:, "sys_u"], marker_shape[0]+"--", markersize=marker_size,  linewidth=line_width,markerfacecolor='none')
        axes.plot(x, H_E.loc[:, "sys_u"], "--", markersize=marker_size, 
                     linewidth=line_width)
        axes.plot(x, H_N.loc[:, "sys_u"], marker_shape[2]+"-", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        axes.plot(x, H_I.loc[:, "sys_u"], "-", markersize=marker_size,  linewidth=line_width)
        axes.plot(x, W_E.loc[:, "sys_u"], marker_shape[4]+"--", markersize=marker_size,
                     linewidth=line_width,markerfacecolor='none')
        axes.plot(x, W_I.loc[:, "sys_u"], ":", markersize=marker_size, 
                     linewidth=line_width)
        axes.plot(x, W_N.loc[:, "sys_u"], marker_shape[6]+"-.", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        axes.tick_params(axis='both', labelsize=tick_font["size"])
        axes.set_xlabel('SLA', label_font)
        axes.set_ylabel('Utility', label_font)
        axes.legend(prop=legend_font, loc="upper left", framealpha=0.05,labelspacing=0.01,ncol=2)


    def plot_sys_utility_factor(self,axs):
        axes = axs[0]
        self.plot_sys_utility_SLA(axes)
        axes = axs[1]
        self.plot_sys_utility_network(axes)
        axes = axs[2]
        self.plot_sys_utility_fairness(axes)
    def plot_model_resource_SLA(self,axes):
        item = "SLA"
        H_M = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_M_"+item+".xlsx",index_col=0)
        H_E = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_E_"+item+".xlsx",index_col=0)
        H_N = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_NS_"+item+".xlsx",index_col=0)
        H_I = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_I_"+item+".xlsx", index_col=0)
        W_E = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_E_"+item+".xlsx",index_col=0)
        W_N = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_NS_"+item+".xlsx",index_col=0)
        W_I = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_I_"+item+".xlsx", index_col=0)
        x = np.arange(0.1, 1.01, 0.05)


        I_cores = axes[0]
        I_cores.grid(linestyle="-.")
        I_cores.set_ylim(0 - 0.5, 6 + 1)
        I_cores.yaxis.set_major_locator(MultipleLocator(2))
        I_cores.set_xlim(0.1, 1.01)
        I_cores.xaxis.set_major_locator(MultipleLocator(0.1))
        I_cores.plot(x, H_M.loc[:, "I_cores"], marker_shape[0]+"--", markersize=marker_size, linewidth=line_width,markerfacecolor='none')
        I_cores.plot(x, H_E.loc[:, "I_cores"], "--", markersize=marker_size,
                     linewidth=line_width)
        I_cores.plot(x, H_N.loc[:, "I_cores"], marker_shape[2]+"-", markersize=marker_size,
                     linewidth=line_width,markerfacecolor='none')
        I_cores.plot(x, H_I.loc[:, "I_cores"], "-", markersize=marker_size,  linewidth=line_width)
        I_cores.plot(x, W_E.loc[:, "I_cores"], marker_shape[4]+"--", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        I_cores.plot(x, W_I.loc[:, "I_cores"], ":", markersize=marker_size, 
                     linewidth=line_width)
        I_cores.plot(x, W_N.loc[:, "I_cores"], marker_shape[6]+"-.", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        I_cores.tick_params(axis='y', labelsize=tick_font["size"])
        I_cores.set_ylabel("Inception"+"\n"+"(Cores)",label_font)
        #I_cores.legend(fontsize=legend_font["size"]-10, ncol=2,labelspacing=0.01,framealpha=0.05)

        R_cores = axes[1]
        R_cores.grid(linestyle="-.")
        R_cores.set_ylim(0 - 0.5, 6 + 1)
        R_cores.yaxis.set_major_locator(MultipleLocator(2))
        R_cores.plot(x, H_M.loc[:, "R_cores"], marker_shape[0]+"--", markersize=marker_size,  
                     linewidth=line_width,markerfacecolor='none')
        R_cores.plot(x, H_E.loc[:, "R_cores"], "--", markersize=marker_size, 
                     linewidth=line_width)
        R_cores.plot(x, H_N.loc[:, "R_cores"], marker_shape[2]+"-", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        R_cores.plot(x, H_I.loc[:, "R_cores"], "-", markersize=marker_size, 
                     linewidth=line_width)
        R_cores.plot(x, W_E.loc[:, "R_cores"], marker_shape[4]+"--", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        R_cores.plot(x, W_I.loc[:, "R_cores"], ":", markersize=marker_size, 
                     linewidth=line_width)
        R_cores.plot(x, W_N.loc[:, "R_cores"], marker_shape[6]+"-.", markersize=marker_size,
                     linewidth=line_width,markerfacecolor='none')
        R_cores.tick_params(axis='y', labelsize=tick_font["size"])
        R_cores.set_ylabel("ResNet"+"\n"+"(Cores)",label_font)

        M_cores = axes[2]
        M_cores.grid(linestyle="-.")
        M_cores.set_ylim(0 - 0.5, 6 + 1)
        M_cores.yaxis.set_major_locator(MultipleLocator(2))
        M_cores.plot(x, H_M.loc[:, "M_cores"], marker_shape[0]+"--", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        M_cores.plot(x, H_E.loc[:, "M_cores"], "--", markersize=marker_size, 
                     linewidth=line_width)
        M_cores.plot(x, H_N.loc[:, "M_cores"], marker_shape[2]+"-", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        M_cores.plot(x, H_I.loc[:, "M_cores"], "-", markersize=marker_size, 
                     linewidth=line_width)
        M_cores.plot(x, W_E.loc[:, "M_cores"], marker_shape[4]+"--", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        M_cores.plot(x, W_I.loc[:, "M_cores"], ":", markersize=marker_size, 
                     linewidth=line_width)
        M_cores.plot(x, W_N.loc[:, "M_cores"], marker_shape[6]+"-.", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        M_cores.tick_params(axis='both', labelsize=tick_font["size"])
        M_cores.set_xlim(0.3,1.01)
        M_cores.xaxis.set_major_locator(MultipleLocator(0.1))
        #M_cores.set_xlabel('SLA (%)', label_font)
        M_cores.set_ylabel('MobileNet'+"\n"+"(Cores)", label_font)

    def plot_model_resource_factor(self):
        plt.rcParams["figure.figsize"] = [16, 8]
        fig, axs = plt.subplots(4,3,sharex='col')
        axes = axs[0:3,0]
        self.plot_model_resource_SLA(axes)
        axes = axs[0:3,1]
        self.plot_model_resource_network(axes)
        axes = axs[0:3,2]
        self.plot_model_resource_fairness(axes)
        axes = axs[3,:]
        self.plot_sys_utility_factor(axes)
        fig.legend(loc="upper center",ncol=4,prop=legend_font,framealpha=0.05)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1, top=0.88)
        plt.savefig("../figures/resource_allocation_model_resource_factors.pdf", bbox_inches='tight', pad_inches=0)
        plt.show()


    def get_sys_u(self,data,I_weight_arr,R_weight_arr):
        result = -1*np.ones((I_weight_arr.shape[0],R_weight_arr.shape[0]))
        print(result.shape)
        for index in data.index.values:
            record= data.loc[index,:]
            I_W_index = np.where(I_weight_arr==record["I_W"])[0][0]
            R_W_index = np.where(I_weight_arr==record["R_W"])[0][0]
            sys_u = record["sys_u"]
            result[R_W_index][I_W_index] = sys_u
        return result

    def get_model_resource(self,data,I_weight_arr,R_weight_arr,model_name):
        result = -2*np.ones((I_weight_arr.shape[0],R_weight_arr.shape[0]))
        print(result.shape)
        for index in data.index.values:
            record= data.loc[index,:]
            I_W_index = np.where(I_weight_arr==record["I_W"])[0][0]
            R_W_index = np.where(I_weight_arr==record["R_W"])[0][0]
            if record["meet_fairness"]==False:
                cores = -1
            else:
                if model_name == "Inception":
                    cores = record["I_cores"]
                elif model_name == "ResNet":
                    cores = record["R_cores"]
                else:
                    cores = record["M_cores"]
            result[R_W_index][I_W_index] = cores
        return result

    def get_model_efficiency(self,data,I_weight_arr,R_weight_arr,model_name):
        result = -1*np.ones((I_weight_arr.shape[0],R_weight_arr.shape[0]))
        for index in data.index.values:
            record= data.loc[index,:]
            I_W_index = np.where(I_weight_arr==record["I_W"])[0][0]
            R_W_index = np.where(I_weight_arr==record["R_W"])[0][0]
            if model_name == "Inception":
                efficiency = record["I_E"]
            elif model_name == "ResNet":
                efficiency = record["R_E"]
            else:
                efficiency = record["M_E"]
            result[R_W_index][I_W_index] = efficiency
            #print(I_W_index,R_W_index,efficiency)
        return result

    def plot_model_resource_weight(self):
        H_M = pd.read_excel("../experiment/resource_allocation/model_weight/mckp_M_model_weight.xlsx",index_col=0)
        H_E = pd.read_excel("../experiment/resource_allocation/model_weight/mckp_E_model_weight.xlsx",index_col=0)
        H_N = pd.read_excel("../experiment/resource_allocation/model_weight/mckp_NS_model_weight.xlsx",index_col=0)
        H_I = pd.read_excel("../experiment/resource_allocation/model_weight/mckp_I_model_weight.xlsx", index_col=0)
        W_E = pd.read_excel("../experiment/resource_allocation/model_weight/weight_E_model_weight.xlsx",index_col=0)
        W_N = pd.read_excel("../experiment/resource_allocation/model_weight/weight_NS_model_weight.xlsx",index_col=0)
        W_I = pd.read_excel("../experiment/resource_allocation/model_weight/weight_I_model_weight.xlsx", index_col=0)
        fig, axs = plt.subplots(3,7,sharex='all',sharey='row')
        x = H_M["I_W"].values #I_W
        y = H_M["R_W"].values #R_W
        z = H_M["sys_u"].values

        z_min, z_max = -2,10
        ax = axs[0][0]
        x1 = np.around(np.linspace(x.min(), x.max(), len(np.unique(x))),2)
        y1 = np.around(np.linspace(y.min(), y.max(), len(np.unique(y))),2)
        x2, y2 = np.meshgrid(x1, y1)
        cores = self.get_model_resource(H_M,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.set_ylim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.3))
        ax.set_title('H_M',title_font)
        ax.set_ylabel("Inception", label_font)
        ax.tick_params(labelsize=tick_font["size"])
        temp = fig.colorbar(c,ax=axs.ravel().tolist())
        temp.set_label(label="CPU Cores", size=label_font["size"])
        temp.ax.tick_params(labelsize=tick_font["size"])

        ax = axs[0][1]
        cores = self.get_model_resource(H_E,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('H_E',title_font)


        ax = axs[0][2]
        cores = self.get_model_resource(H_N,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('H_N',title_font)

        ax = axs[0][3]
        cores = self.get_model_resource(H_I,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('H_I',title_font)

        ax = axs[0][4]
        cores = self.get_model_resource(W_E,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('W_E',title_font)

        ax = axs[0][5]
        cores = self.get_model_resource(W_N,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('W_N',title_font)

        ax = axs[0][6]
        cores = self.get_model_resource(W_N,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('W_I',title_font)

        ax = axs[1][0]
        cores = self.get_model_resource(H_M,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_ylabel("ResNet", label_font)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.set_ylim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.3))
        ax.tick_params(labelsize=tick_font["size"])
        #fig.colorbar(c,ax=axs.ravel().tolist()).set_label(label="CPU Cores", size=label_font["size"])

        ax = axs[1][1]
        cores = self.get_model_resource(H_E,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)

        ax = axs[1][2]
        cores = self.get_model_resource(H_N,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        ax = axs[1][3]
        cores = self.get_model_resource(H_I,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        ax = axs[1][4]
        cores = self.get_model_resource(W_E,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        ax = axs[1][5]
        cores = self.get_model_resource(W_N,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        ax = axs[1][6]
        cores = self.get_model_resource(W_N,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_ylabel("ResNet Weight",size=label_font["size"]-5)
        ax.yaxis.set_label_position("right")

        ax = axs[2][0]
        cores = self.get_model_resource(H_M, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_ylabel("MobileNet", label_font)
        ax.set_ylim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.3))
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[2][1]
        cores = self.get_model_resource(H_E, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[2][2]
        cores = self.get_model_resource(H_N, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[2][3]
        cores = self.get_model_resource(H_I, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_xlabel("Inception Weight", size=label_font["size"]-5)
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[2][4]
        cores = self.get_model_resource(W_E, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[2][5]
        cores = self.get_model_resource(W_N, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[2][6]
        cores = self.get_model_resource(W_I, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(labelsize=tick_font["size"])


        plt.savefig("../figures/resource_allocation_model_resource_weight.pdf",bbox_inches = 'tight',pad_inches = 0)
        #plt.savefig("../figures/resource_allocation_model_resource_weight.png")
        plt.show()

    def plot_model_efficiency_weight(self):
        greedy = pd.read_excel("../experiment/resource_allocation/model_weight/hitdl_greedy.xlsx",index_col=0)
        optimal = pd.read_excel("../experiment/resource_allocation/model_weight/hitdl_optimal.xlsx",index_col=0)
        weighted = pd.read_excel("../experiment/resource_allocation/model_weight/hitdl_weighted.xlsx",index_col=0)
        fig, axs = plt.subplots(3,1,sharex='col',sharey="col")
        x = greedy["I_W"].values #I_W
        y = greedy["R_W"].values #R_W

        ax = axs[0]
        x1 = np.around(np.linspace(x.min(), x.max(), len(np.unique(x))),2)
        y1 = np.around(np.linspace(y.min(), y.max(), len(np.unique(y))),2)
        x2, y2 = np.meshgrid(x1, y1)
        E = self.get_model_efficiency(greedy,x1,y1,"Inception")
        z_min, z_max = -1,2.5
        c = ax.pcolor(x2, y2, E, cmap='Blues', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.85 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_title('Inception',title_font)
        ax.tick_params(labelsize=tick_font["size"])
        #fig.colorbar(c, ax=axs.ravel().tolist()).set_label(label="Efficiency", size=15)

        ax = axs[1]
        E = self.get_model_efficiency(weighted,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, E, cmap='Blues', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.85 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_title('ResNet',title_font)
        ax.set_ylabel("ResNet Weight", label_font)
        ax.tick_params(labelsize=tick_font["size"])


        ax = axs[2]
        E = self.get_model_efficiency(optimal,x1,y1,"MobileNet")
        #z_min, z_max = np.min(E),np.max(E)
        c = ax.pcolor(x2, y2, E, cmap='Blues', vmin=z_min, vmax=z_max)
        #c.tick_params(labelsize=tick_font["size"])
        ax.set_xlim(0.1 - 0.02, 0.85 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_title('MobileNet',title_font)
        ax.set_xlabel("Inception Weight",label_font)
        ax.tick_params(labelsize=tick_font["size"])

        plt.subplots_adjust(hspace=0.4)
        temp = fig.colorbar(c, ax=axs.ravel().tolist())
        temp.set_label(label="Efficiency", size=label_font["size"])
        temp.ax.tick_params(labelsize=tick_font["size"])
        plt.savefig("../figures/resource_allocation_efficiency_weight.pdf",bbox_inches = 'tight',pad_inches = 0)
        plt.savefig("../figures/resource_allocation_efficiency_weight.png",bbox_inches = 'tight',pad_inches = 0)
        plt.show()

    def plot_sys_utility_weight(self):
        H_M = pd.read_excel("../experiment/resource_allocation/model_weight/mckp_M_model_weight.xlsx",index_col=0)
        H_E = pd.read_excel("../experiment/resource_allocation/model_weight/mckp_E_model_weight.xlsx",index_col=0)
        H_N = pd.read_excel("../experiment/resource_allocation/model_weight/mckp_NS_model_weight.xlsx",index_col=0)
        H_I = pd.read_excel("../experiment/resource_allocation/model_weight/mckp_I_model_weight.xlsx", index_col=0)
        W_E = pd.read_excel("../experiment/resource_allocation/model_weight/weight_E_model_weight.xlsx",index_col=0)
        W_N = pd.read_excel("../experiment/resource_allocation/model_weight/weight_NS_model_weight.xlsx",index_col=0)
        W_I = pd.read_excel("../experiment/resource_allocation/model_weight/weight_I_model_weight.xlsx", index_col=0)
        fig, axs = plt.subplots(1,7,sharey='row')

        x = H_M["I_W"].values #I_W
        y = H_M["R_W"].values #R_W
        x1 = np.around(np.linspace(x.min(), x.max(), len(np.unique(x))),2)
        y1 = np.around(np.linspace(y.min(), y.max(), len(np.unique(y))),2)
        x2, y2 = np.meshgrid(x1, y1)
        z_min, z_max =0,20
        ax = axs[0]
        x1 = np.around(np.linspace(x.min(), x.max(), len(np.unique(x))),2)
        y1 = np.around(np.linspace(y.min(), y.max(), len(np.unique(y))),2)
        x2, y2 = np.meshgrid(x1, y1)
        cores = self.get_sys_u(H_M,x1,y1)
        c = ax.pcolor(x2, y2, cores, cmap='Blues', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.set_ylim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.3))
        ax.set_title('H_M',title_font)
        ax.set_ylabel("ResNet Weight", label_font)
        ax.tick_params(labelsize=tick_font["size"])
        temp = fig.colorbar(c,ax=axs.ravel().tolist())
        temp.set_label(label="System Utility", size=label_font["size"]-5)
        temp.ax.tick_params(labelsize=tick_font["size"])

        ax = axs[1]
        cores = self.get_sys_u(H_E,x1,y1)
        c = ax.pcolor(x2, y2, cores, cmap='Blues', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('H_E',title_font)
        ax.tick_params(labelsize=tick_font["size"])


        ax = axs[2]
        cores = self.get_sys_u(H_N,x1,y1)
        c = ax.pcolor(x2, y2, cores, cmap='Blues', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('H_N',title_font)
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[3]
        cores = self.get_sys_u(H_I,x1,y1)
        c = ax.pcolor(x2, y2, cores, cmap='Blues', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('H_I',title_font)
        ax.set_xlabel("Inception Weight",label_font)
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[4]
        cores = self.get_sys_u(W_E,x1,y1)
        c = ax.pcolor(x2, y2, cores, cmap='Blues', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('W_E',title_font)
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[5]
        cores = self.get_sys_u(W_N,x1,y1)
        c = ax.pcolor(x2, y2, cores, cmap='Blues', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('W_N',title_font)
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[6]
        cores = self.get_sys_u(W_N,x1,y1)
        c = ax.pcolor(x2, y2, cores, cmap='Blues', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('W_I',title_font)
        ax.tick_params(labelsize=tick_font["size"])
        plt.savefig("../figures/resource_allocation_sys_utility_weight.pdf", bbox_inches='tight', pad_inches=0)
        plt.show()

    def plot_sys_utility_weight_pass(self):
        H_M = pd.read_excel("../experiment/resource_allocation/model_weight/mckp_M_model_weight.xlsx",index_col=0)
        H_E = pd.read_excel("../experiment/resource_allocation/model_weight/mckp_E_model_weight.xlsx",index_col=0)
        H_N = pd.read_excel("../experiment/resource_allocation/model_weight/mckp_NS_model_weight.xlsx",index_col=0)
        H_I = pd.read_excel("../experiment/resource_allocation/model_weight/mckp_I_model_weight.xlsx", index_col=0)
        W_E = pd.read_excel("../experiment/resource_allocation/model_weight/weight_E_model_weight.xlsx",index_col=0)
        W_N = pd.read_excel("../experiment/resource_allocation/model_weight/weight_NS_model_weight.xlsx",index_col=0)
        W_I = pd.read_excel("../experiment/resource_allocation/model_weight/weight_I_model_weight.xlsx", index_col=0)
        fig, axs = plt.subplots(3,7,sharex='all',sharey='row')
        x = H_M["I_W"].values #I_W
        y = H_M["R_W"].values #R_W
        z = H_M["sys_u"].values

        z_min, z_max = -2,10
        ax = axs[0][0]
        x1 = np.around(np.linspace(x.min(), x.max(), len(np.unique(x))),2)
        y1 = np.around(np.linspace(y.min(), y.max(), len(np.unique(y))),2)
        x2, y2 = np.meshgrid(x1, y1)
        cores = self.get_sys_u(H_M,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.set_ylim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.3))
        ax.set_title('H_M',title_font)
        ax.set_ylabel("Inception", label_font)
        ax.tick_params(labelsize=tick_font["size"])
        temp = fig.colorbar(c,ax=axs.ravel().tolist())
        temp.set_label(label="CPU Cores", size=label_font["size"])
        temp.ax.tick_params(labelsize=tick_font["size"])

        ax = axs[0][1]
        cores = self.get_model_resource(H_E,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('H_E',title_font)


        ax = axs[0][2]
        cores = self.get_model_resource(H_N,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('H_N',title_font)

        ax = axs[0][3]
        cores = self.get_model_resource(H_I,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('H_I',title_font)

        ax = axs[0][4]
        cores = self.get_model_resource(W_E,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('W_E',title_font)

        ax = axs[0][5]
        cores = self.get_model_resource(W_N,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('W_N',title_font)

        ax = axs[0][6]
        cores = self.get_model_resource(W_N,x1,y1,"Inception")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_title('W_I',title_font)

        ax = axs[1][0]
        cores = self.get_model_resource(H_M,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_ylabel("ResNet", label_font)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.set_ylim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.3))
        ax.tick_params(labelsize=tick_font["size"])
        #fig.colorbar(c,ax=axs.ravel().tolist()).set_label(label="CPU Cores", size=label_font["size"])

        ax = axs[1][1]
        cores = self.get_model_resource(H_E,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)

        ax = axs[1][2]
        cores = self.get_model_resource(H_N,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        ax = axs[1][3]
        cores = self.get_model_resource(H_I,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        ax = axs[1][4]
        cores = self.get_model_resource(W_E,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        ax = axs[1][5]
        cores = self.get_model_resource(W_N,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

        ax = axs[1][6]
        cores = self.get_model_resource(W_N,x1,y1,"ResNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_ylabel("ResNet Weight",size=label_font["size"]-5)
        ax.yaxis.set_label_position("right")

        ax = axs[2][0]
        cores = self.get_model_resource(H_M, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_ylabel("MobileNet", label_font)
        ax.set_ylim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.3))
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[2][1]
        cores = self.get_model_resource(H_E, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[2][2]
        cores = self.get_model_resource(H_N, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[2][3]
        cores = self.get_model_resource(H_I, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.set_xlabel("Inception Weight", size=label_font["size"]-5)
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[2][4]
        cores = self.get_model_resource(W_E, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[2][5]
        cores = self.get_model_resource(W_N, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(labelsize=tick_font["size"])

        ax = axs[2][6]
        cores = self.get_model_resource(W_I, x1, y1, "MobileNet")
        c = ax.pcolor(x2, y2, cores, cmap='Greens', vmin=z_min, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.88 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(labelsize=tick_font["size"])


        plt.savefig("../figures/resource_allocation_model_resource_weight.pdf",bbox_inches = 'tight',pad_inches = 0)
        #plt.savefig("../figures/resource_allocation_model_resource_weight.png")
        plt.show()
    def plot_sys_utility_weight_depected(self):
        greedy = pd.read_excel("../experiment/resource_allocation/model_weight/hitdl_greedy.xlsx",index_col=0)
        optimal = pd.read_excel("../experiment/resource_allocation/model_weight/hitdl_optimal.xlsx",index_col=0)
        weighted = pd.read_excel("../experiment/resource_allocation/model_weight/hitdl_weighted.xlsx",index_col=0)
        fig, axs = plt.subplots(3,1,sharex='col')
        x = optimal["I_W"].values #I_W
        y = optimal["R_W"].values #R_W
        z = optimal["sys_u"].values

        ax = axs[0]
        x1 = np.around(np.linspace(x.min(), x.max(), len(np.unique(x))),2)
        y1 = np.around(np.linspace(y.min(), y.max(), len(np.unique(y))),2)
        x2, y2 = np.meshgrid(x1, y1)
        sys_u = self.get_sys_u(optimal, x1, y1)
        z_min, z_max = np.min(sys_u),np.max(sys_u)
        ax.set_title("HiTDL+Optimal", label_font)
        ax.set_xlim(0.1 - 0.02, 0.85 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        c = ax.pcolor(x2, y2, sys_u, cmap='coolwarm', vmin=0, vmax=z_max)

        ax = axs[2]
        #c = ax.pcolor(x2, y2, sys_u, cmap='coolwarm', vmin=0, vmax=z_max)
        ax.set_xlim(0.1 - 0.02, 0.85 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_title("HiTDL", label_font)
        ax.set_xlabel("Inception Weight", label_font)
        sys_u = self.get_sys_u(greedy, x1, y1)
        c = ax.pcolor(x2, y2, sys_u, cmap='coolwarm', vmin=0, vmax=z_max)


        ax = axs[1]
        sys_u = self.get_sys_u(weighted, x1, y1)
        #z_min, z_max = np.min(sys_u),np.max(sys_u)
        c = ax.pcolor(x2, y2, sys_u, cmap='coolwarm', vmin=0, vmax=z_max)
        ax.set_ylabel('ResNet Weight',label_font)
        ax.set_xlim(0.1 - 0.02, 0.85 + 0.02)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_title("HiTDL+Weighted", label_font)



        fig.tight_layout(pad=0.01)
        fig.colorbar(c, ax=axs.ravel().tolist()).set_label(label="System Utility", size=15)
        plt.savefig("../figures/resource_allocation_sys_utility_weight.pdf")
        plt.savefig("../figures/resource_allocation_sys_utility_weight.png")
        plt.show()

    def plot_model_resource_fairness(self,axes):
        item = "fairness"
        H_M = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_M_"+item+".xlsx",index_col=0)
        H_E = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_E_"+item+".xlsx",index_col=0)
        H_N = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_NS_"+item+".xlsx",index_col=0)
        H_I = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_I_"+item+".xlsx", index_col=0)
        W_E = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_E_"+item+".xlsx",index_col=0)
        W_N = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_NS_"+item+".xlsx",index_col=0)
        W_I = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_I_"+item+".xlsx", index_col=0)
        x = np.arange(0.34,1.01,0.03)

        I_cores = axes[0]
        I_cores.grid(linestyle="-.")
        I_cores.set_ylim(0 - 0.5, 12 + 1)
        I_cores.yaxis.set_major_locator(MultipleLocator(4))
        I_cores.plot(x, H_M.loc[:, "I_cores"], marker_shape[0]+"--", markersize=marker_size, label=self.legend_name[0], linewidth=line_width,markerfacecolor='none')
        I_cores.plot(x, H_E.loc[:, "I_cores"], "--", markersize=marker_size, label=self.legend_name[1],
                     linewidth=line_width)
        I_cores.plot(x, H_N.loc[:, "I_cores"], marker_shape[2]+"-", markersize=marker_size, label=self.legend_name[2],
                     linewidth=line_width,markerfacecolor='none')
        I_cores.plot(x, H_I.loc[:, "I_cores"], "-", markersize=marker_size, label=self.legend_name[3], linewidth=line_width)
        I_cores.plot(x, W_E.loc[:, "I_cores"], marker_shape[4]+"--", markersize=marker_size, label=self.legend_name[4],
                     linewidth=line_width,markerfacecolor='none')
        I_cores.plot(x, W_I.loc[:, "I_cores"], ":", markersize=marker_size, label=self.legend_name[5],
                     linewidth=line_width)
        I_cores.plot(x, W_N.loc[:, "I_cores"], marker_shape[6]+"-.", markersize=marker_size, label=self.legend_name[6],
                     linewidth=line_width,markerfacecolor='none')
        I_cores.tick_params(axis='y', labelsize=tick_font["size"])

        R_cores = axes[1]
        R_cores.grid(linestyle="-.")
        R_cores.set_ylim(0 - 0.5, 6 + 1)
        R_cores.yaxis.set_major_locator(MultipleLocator(2))
        R_cores.plot(x, H_M.loc[:, "R_cores"], marker_shape[0]+"--", markersize=marker_size,  linewidth=line_width,markerfacecolor='none')
        R_cores.plot(x, H_E.loc[:, "R_cores"], "--", markersize=marker_size, 
                     linewidth=line_width)
        R_cores.plot(x, H_N.loc[:, "R_cores"], marker_shape[2]+"-", markersize=marker_size,
                     linewidth=line_width,markerfacecolor='none')
        R_cores.plot(x, H_I.loc[:, "R_cores"], "-", markersize=marker_size,  linewidth=line_width)
        R_cores.plot(x, W_E.loc[:, "R_cores"], marker_shape[4]+"--", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        R_cores.plot(x, W_I.loc[:, "R_cores"], ":", markersize=marker_size, 
                     linewidth=line_width)
        R_cores.plot(x, W_N.loc[:, "R_cores"], marker_shape[6]+"-.", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        R_cores.tick_params(axis='y', labelsize=tick_font["size"])

        M_cores = axes[2]
        M_cores.grid(linestyle="-.")
        M_cores.set_ylim(0 - 0.5, 12 + 1)
        M_cores.yaxis.set_major_locator(MultipleLocator(4))
        M_cores.plot(x, H_M.loc[:, "M_cores"], marker_shape[0]+"--", markersize=marker_size,linewidth=line_width,markerfacecolor='none')
        M_cores.plot(x, H_E.loc[:, "M_cores"], "--", markersize=marker_size, 
                     linewidth=line_width)
        M_cores.plot(x, H_N.loc[:, "M_cores"], marker_shape[2]+"-", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        M_cores.plot(x, H_I.loc[:, "M_cores"], "-", markersize=marker_size,  linewidth=line_width)
        M_cores.plot(x, W_E.loc[:, "M_cores"], marker_shape[4]+"--", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        M_cores.plot(x, W_I.loc[:, "M_cores"], ":", markersize=marker_size, 
                     linewidth=line_width)
        M_cores.plot(x, W_N.loc[:, "M_cores"], marker_shape[6]+"-.", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        M_cores.tick_params(axis='both', labelsize=tick_font["size"])
        M_cores.set_xlim(0.3,1.01)
        M_cores.xaxis.set_major_locator(MultipleLocator(0.1))
        #M_cores.set_xlabel('Fairness (%)', label_font)


    def plot_sys_utility_fairness(self,axes):
        item = "fairness"
        H_M = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_M_"+item+".xlsx",index_col=0)
        H_E = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_E_"+item+".xlsx",index_col=0)
        H_N = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_NS_"+item+".xlsx",index_col=0)
        H_I = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_I_"+item+".xlsx", index_col=0)
        W_E = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_E_"+item+".xlsx",index_col=0)
        W_N = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_NS_"+item+".xlsx",index_col=0)
        W_I = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_I_"+item+".xlsx", index_col=0)
        x = np.arange(0.34,1.01,0.03)
        axes.set_ylim(0-1, 36+1)
        axes.yaxis.set_major_locator(MultipleLocator(9))
        axes.set_xlim(0.3,1.01)
        axes.xaxis.set_major_locator(MultipleLocator(0.1))
        axes.grid(linestyle="-.")
        axes.plot(x, H_M.loc[:, "sys_u"], marker_shape[0]+"--", markersize=marker_size, linewidth=line_width,markerfacecolor='none')
        axes.plot(x, H_E.loc[:, "sys_u"], "--", markersize=marker_size, 
                     linewidth=line_width)
        axes.plot(x, H_N.loc[:, "sys_u"], marker_shape[2]+"-", markersize=marker_size,
                     linewidth=line_width,markerfacecolor='none')
        axes.plot(x, H_I.loc[:, "sys_u"], "-", markersize=marker_size,  linewidth=line_width)
        axes.plot(x, W_E.loc[:, "sys_u"], marker_shape[4]+"--", markersize=marker_size,
                     linewidth=line_width,markerfacecolor='none')
        axes.plot(x, W_I.loc[:, "sys_u"], ":", markersize=marker_size,
                     linewidth=line_width)
        axes.plot(x, W_N.loc[:, "sys_u"], marker_shape[6]+"-.", markersize=marker_size,
                     linewidth=line_width,markerfacecolor='none')
        axes.tick_params(axis='both', labelsize=tick_font["size"])

        axes.set_xlabel('Fairness', label_font)
        #axes.legend(prop=legend_font, loc="upper left", framealpha=0.05,labelspacing=0.01,ncol=2)


    def plot_sys_utility_network(self,axes):
        item = "sim_network"
        H_M = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_M_"+item+".xlsx",index_col=0)
        H_E = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_E_"+item+".xlsx",index_col=0)
        H_N = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_NS_"+item+".xlsx",index_col=0)
        H_I = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_I_"+item+".xlsx", index_col=0)
        W_E = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_E_"+item+".xlsx",index_col=0)
        W_N = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_NS_"+item+".xlsx",index_col=0)
        W_I = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_I_"+item+".xlsx", index_col=0)
        x = H_M["I_net"].values

        axes.grid(linestyle="-.")
        axes.set_xlim(10 ** 0-0.02, 10 ** 3+0.02)
        axes.set_ylim(0-1, 60)
        axes.yaxis.set_major_locator(MultipleLocator(20))
        axes.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        axes.set_xscale('log')

        axes.grid(linestyle="-.")
        axes.plot(x, H_M.loc[:, "sys_u"], marker_shape[0]+"--", markersize=marker_size,  linewidth=line_width,markerfacecolor='none')
        axes.plot(x, H_E.loc[:, "sys_u"], "--", markersize=marker_size,
                     linewidth=line_width)
        axes.plot(x, H_N.loc[:, "sys_u"], marker_shape[2]+"-", markersize=marker_size,
                     linewidth=line_width,markerfacecolor='none')
        axes.plot(x, H_I.loc[:, "sys_u"], "-", markersize=marker_size,  linewidth=line_width)
        axes.plot(x, W_E.loc[:, "sys_u"], marker_shape[4]+"--", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        axes.plot(x, W_I.loc[:, "sys_u"], ":", markersize=marker_size, 
                     linewidth=line_width)
        axes.plot(x, W_N.loc[:, "sys_u"], marker_shape[6]+"-.", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        axes.tick_params(axis='both', labelsize=tick_font["size"])

        #axes.legend(prop=legend_font, loc="upper left", framealpha=0.05,labelspacing=0.01,ncol=2)
        axes.set_xlabel("Network Bandwidth (Mbps)",label_font)


    def plot_model_efficiency_network(self):
        def filter_data(data, model_name):
            for i in range(len(data)):
                if round(data[i],3)<0:
                    data[i]=0
            return data
        greedy = pd.read_excel("../experiment/resource_allocation/network/simulate_trace/hitdl_greedy.xlsx",index_col=0)
        optimal = pd.read_excel("../experiment/resource_allocation/network/simulate_trace/hitdl_optimal.xlsx",index_col=0)
        weighted = pd.read_excel("../experiment/resource_allocation/network/simulate_trace/hitdl_weighted.xlsx",index_col=0)
        fig, I_E = plt.subplots()

        I_E.grid(linestyle="-.")
        I_E.set_xlim(10 ** 0-0.2, 10 ** 3+0.2)
        I_E.set_ylim(0,5.5)
        I_E.set_xscale('log')
        I_E.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        I_E.yaxis.set_major_locator(MultipleLocator(0.5))
        I_E.plot(range(1,1001), filter_data(greedy.loc[:,"I_E"],"Inception"), ".-",markersize=marker_size, label="Inception")
        I_E.plot(range(1,1001), filter_data(optimal.loc[:,"R_E"],"ResNet"), "*-", markersize=marker_size, label="ResNet")
        I_E.plot(range(1,1001), filter_data(weighted.loc[:, "M_E"],"MobileNet"), "<-", markersize=marker_size, label="MobileNet")
        I_E.set_ylabel('Efficiency', label_font)
        I_E.set_xlabel("Network Bandwdith (Mbps)")
        I_E.legend(prop=legend_font, loc="upper left", framealpha=0.1)

        plt.savefig("../figures/resource_allocation_efficiency_network.pdf")
        plt.savefig("../figures/resource_allocation_efficiency_network.png")
        plt.show()

    def plot_model_resource_network(self,axes):
        def filter_data(data,model_name):
            for i in data.index.values:
                record = data.loc[i,:]
                if record["meet_fairness"] == False:
                    data.loc[i,model_name[0]+"_cores"] = 0
            return data.loc[:,model_name[0]+"_cores"]
        item = "sim_network"
        H_M = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_M_"+item+".xlsx",index_col=0)
        H_E = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_E_"+item+".xlsx",index_col=0)
        H_N = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_NS_"+item+".xlsx",index_col=0)
        H_I = pd.read_excel("../experiment/resource_allocation/"+item+"/mckp_I_"+item+".xlsx", index_col=0)
        W_E = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_E_"+item+".xlsx",index_col=0)
        W_N = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_NS_"+item+".xlsx",index_col=0)
        W_I = pd.read_excel("../experiment/resource_allocation/"+item+"/weight_I_"+item+".xlsx", index_col=0)
        x = H_M["I_net"].values
        lines = []
        I_cores = axes[0]
        I_cores.grid(linestyle="-.")
        I_cores.set_xlim(10 ** 0-0.02, 10 ** 3+0.02)
        I_cores.set_xscale('log')
        I_cores.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        I_cores.set_ylim(0-0.5,6+1)
        I_cores.yaxis.set_major_locator(MultipleLocator(2))
        I_cores.plot(x, H_M.loc[:, "I_cores"], marker_shape[0]+"--", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')

        I_cores.plot(x, H_E.loc[:, "I_cores"], "--", markersize=marker_size, 
                     linewidth=line_width)
        I_cores.plot(x, H_N.loc[:, "I_cores"], marker_shape[2]+"-", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        I_cores.plot(x, H_I.loc[:, "I_cores"], "-", markersize=marker_size, 
                     linewidth=line_width)

        I_cores.plot(x, W_E.loc[:, "I_cores"], marker_shape[4]+"--", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        I_cores.plot(x, W_I.loc[:, "I_cores"], ":", markersize=marker_size, 
                     linewidth=line_width)

        I_cores.plot(x, W_N.loc[:, "I_cores"], marker_shape[6]+"-.", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')

        #I_cores.legend(fontsize=legend_font["size"]-10, loc="upper left", ncol=2,labelspacing=0.01,framealpha=0.05)
        I_cores.tick_params(axis='y', labelsize=tick_font["size"])

        R_cores = axes[1]
        R_cores.grid(linestyle="-.")
        R_cores.set_ylim(0 - 0.5, 6 + 1)
        R_cores.yaxis.set_major_locator(MultipleLocator(2))
        R_cores.plot(x, H_M.loc[:, "R_cores"], marker_shape[0]+"--", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        R_cores.plot(x, H_E.loc[:, "R_cores"], "--", markersize=marker_size, 
                     linewidth=line_width)
        R_cores.plot(x, H_N.loc[:, "R_cores"], marker_shape[2]+"-", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        R_cores.plot(x, H_I.loc[:, "R_cores"], "-", markersize=marker_size, linewidth=line_width)
        R_cores.plot(x, W_E.loc[:, "R_cores"], marker_shape[4]+"--", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        R_cores.plot(x, W_I.loc[:, "R_cores"], ":", markersize=marker_size, 
                     linewidth=line_width)
        R_cores.plot(x, W_N.loc[:, "R_cores"], marker_shape[6]+"-.", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        R_cores.tick_params(axis='y', labelsize=tick_font["size"])
        #R_cores.set_ylabel("CPU Cores", label_font)

        M_cores = axes[2]
        M_cores.grid(linestyle="-.")
        M_cores.set_ylim(0 - 0.5, 6 + 1)
        M_cores.yaxis.set_major_locator(MultipleLocator(2))
        M_cores.plot(x, H_M.loc[:, "M_cores"], marker_shape[0]+"--", markersize=marker_size,  linewidth=line_width,markerfacecolor='none')
        M_cores.plot(x, H_E.loc[:, "M_cores"], "--", markersize=marker_size, 
                     linewidth=line_width)
        M_cores.plot(x, H_N.loc[:, "M_cores"], marker_shape[2]+"-", markersize=marker_size,
                     linewidth=line_width,markerfacecolor='none')
        M_cores.plot(x, H_I.loc[:, "M_cores"], "-", markersize=marker_size,  linewidth=line_width)
        M_cores.plot(x, W_E.loc[:, "M_cores"], marker_shape[4]+"--", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        M_cores.plot(x, W_I.loc[:, "M_cores"], ":", markersize=marker_size, 
                     linewidth=line_width)
        M_cores.plot(x, W_N.loc[:, "M_cores"], marker_shape[6]+"-.", markersize=marker_size, 
                     linewidth=line_width,markerfacecolor='none')
        M_cores.tick_params(axis='x', labelsize=tick_font["size"])
        M_cores.tick_params(axis='y', labelsize=tick_font["size"])
        #M_cores.set_xlabel("Network Bandwidth (Mbps)", label_font)
        return lines

    def plot_sys_utility_model_efficiency_network(self):
        def filter_data(data, model_name,index):
            for i in range(len(data)):
                if round(data[i],3)<0:
                    data[i]=0
            return np.take(data,index)

        '''
        greedy = pd.read_excel("../experiment/resource_allocation/network/simulate_trace/hitdl_greedy.xlsx",index_col=0)
        optimal = pd.read_excel("../experiment/resource_allocation/network/simulate_trace/hitdl_optimal.xlsx",index_col=0)
        weighted = pd.read_excel("../experiment/resource_allocation/network/simulate_trace/hitdl_weighted.xlsx",index_col=0)
        '''
        greedy = pd.read_excel("../experiment/resource_allocation/network/wifi_trace/hitdl_greedy.xlsx",index_col=0)
        optimal = pd.read_excel("../experiment/resource_allocation/network/wifi_trace/hitdl_optimal.xlsx",index_col=0)
        weighted = pd.read_excel("../experiment/resource_allocation/network/wifi_trace/hitdl_weighted.xlsx",index_col=0)

        fig, axes = plt.subplots(2,1,sharex="col")

        index = np.arange(0,len(greedy["I_net"].values),1)
        x = np.take(greedy["I_net"].values,index)

        sys_utility = axes[0]
        sys_utility.grid(linestyle="-.")
        sys_utility.set_xscale('log')
        sys_utility.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        sys_utility.yaxis.set_major_locator(MultipleLocator(10))

        sys_utility.plot(x, np.take(greedy.loc[:, "sys_u"],indices=index), "-", markersize=marker_size,label="HiTDL",linewidth=line_width)
        sys_utility.plot(x, np.take(optimal.loc[:, "sys_u"], indices=index), "--", markersize=marker_size,label="HiTDL+Optimal", linewidth=line_width)
        sys_utility.plot(x, np.take(weighted.loc[:, "sys_u"], indices=index), ":", markersize=marker_size,
                         label="HiTDL+Weighted", linewidth=line_width)
        sys_utility.set_ylabel('System Utility', label_font)
        sys_utility.legend(prop=legend_font, loc="upper center", framealpha=0.1,bbox_to_anchor=(0,1.01,1,0.2),ncol=3)

        efficiency = axes[1]
        efficiency.grid(linestyle="-.")
        efficiency.set_xlim(10 ** 0-0.2, 10 ** 3+0.2)
        efficiency.set_ylim(0,5.5)
        efficiency.set_xscale('log')
        efficiency.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
        efficiency.yaxis.set_major_locator(MultipleLocator(1))
        efficiency.plot(x, filter_data(greedy.loc[:,"I_E"],"Inception",index), "-",markersize=marker_size, label="Inception",linewidth=line_width)
        efficiency.plot(x, filter_data(weighted.loc[:, "M_E"],"MobileNet",index), "--", markersize=marker_size, label="MobileNet",linewidth=line_width)
        efficiency.plot(x, filter_data(optimal.loc[:, "R_E"], "ResNet", index), ":",
                        markersize=marker_size, label="ResNet", linewidth=line_width)
        efficiency.set_ylabel('Efficiency', label_font)
        efficiency.set_xlabel("Network Bandwdith (Mbps)")
        efficiency.legend(prop=legend_font, loc="upper left", framealpha=0.1,bbox_to_anchor=(0,1.01,1,0.2),ncol=3)

        #plt.savefig("../figures/resouce_allocation_utitliy_efficiency_network_wifi.pdf")
        #plt.savefig("../figures/resouce_allocation_utitliy_efficiency_network_wifi.png")
        plt.show()

    def plot_resource_utility_weight(self):
        pass


pra = PlotResourceAllocation()
pra.plot_model_resource_factor()
#pra.plot_sys_utility_factor()
#pra.plot_model_resource_weight()
#pra.plot_model_efficiency_weight()
#pra.plot_sys_utility_weight()
#pra.plot_model_resource_fairness()
#pra.plot_sys_utility_fairness()
#pra.plot_model_resource_network()
#pra.plot_sys_utility_model_efficiency_network()
#pra.plot_resource_allocation_model_resource()
#pra.plot_resource_allocation_model_efficiency()