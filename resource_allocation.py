import cvxpy as cp
import numpy as np
c1 = np.array([1,3,5])
e1 = np.array([1.505,1.351,1.27])*c1

c2 = np.array([2,5,7])
e2 = np.array([1.844,1.502,1.843])*c2

c3 = np.array([1,5])
e3 = np.array([1.505,1.148])*c3

C = 12
'''
x1 = cp.Variable(name="inception",shape=(len(c1),1),integer=True,pos=True)
y1 = cp.Variable(shape=(len(c1),1),integer=True,pos=True)
x2 = cp.Variable(name="mobilenet",shape=(len(c2),1),integer=True,pos=True)
y2 = cp.Variable(shape=(len(c2),1),integer=True,pos=True)
x3= cp.Variable(name="resnet",shape=(len(c3),1),integer=True,pos=True)
y3 = cp.Variable(shape=(len(c3),1),integer=True,pos=True)
x1 = cp.Variable(name="inception",shape=(len(c1),1),pos=True)
y1 = cp.Variable(shape=(len(c1),1),pos=True)
x2 = cp.Variable(name="mobilenet",shape=(len(c2),1),pos=True)
y2 = cp.Variable(shape=(len(c2),1),pos=True)
x3= cp.Variable(name="resnet",shape=(len(c3),1),pos=True)
y3 = cp.Variable(shape=(len(c3),1),pos=True)

exp1 = e1@cp.multiply(x1-np.ones((len(c1),1)),y1-np.ones((len(c1),1)))
exp2 = e2@cp.multiply(x2-np.ones((len(c2),1)),y2-np.ones((len(c2),1)))
exp3 = e3@cp.multiply(x3-np.ones((len(c3),1)),y3-np.ones((len(c3),1)))

exp4 = c1@cp.multiply(x1-np.ones((len(c1),1)),y1-np.ones((len(c1),1)))
exp5 = c2@cp.multiply(x2-np.ones((len(c2),1)),y2-np.ones((len(c2),1)))
exp6 = c3@cp.multiply(x3-np.ones((len(c3),1)),y3-np.ones((len(c3),1)))

obj = exp1+exp2+exp3
print(obj.shape)
cores_cons = exp4+exp5+exp6
prob1 = cp.Problem(cp.Maximize(obj),
                [cp.sum(y1-np.ones((len(c1),1)))==1,
                cp.sum(y2-np.ones((len(c2),1)))==1,
                cp.sum(y3-np.ones((len(c3),1)))==1,
                cores_cons <= C])
result = prob1.solve(gp=True)
'''

class MCKPAllocation:
    def __init__(self,CPU_Cores,F):
        self.CPU_Cores = CPU_Cores
        self.F = F
        self.C_upper = round(C*F)
    def cpu_const(self):
        ins_size = {"inception":{"intra":[1,3,5],"efficiency":[2.605,1.351,1.27]},
                    "resnet":{"intra":[2,5,7],"efficiency":[1.844,1.502,1.843]},
                    "mobilenet":{"intra":[1,5],"efficiency":[1.505,1.148]}}
        total_plans = len(ins_size["inception"]["intra"])+len(ins_size["resnet"]["intra"])+len(ins_size["mobilenet"]["intra"])
        overall_cons = np.zeros(total_plans)
        overall_E = np.zeros(total_plans)
        model_cons = [np.zeros(total_plans),np.zeros(total_plans),np.zeros(total_plans)]#{"inception":np.zeros(total_plans),"resnet":np.zeros(total_plans),"mobilenet":np.zeros(total_plans)}
        cons_start = {"inception":0,"resnet":len(ins_size["inception"]["intra"]),"mobilenet":len(ins_size["resnet"]["intra"])+len(ins_size["inception"]["intra"])}
        i = 0
        ins_num_upper = np.zeros(total_plans)
        ins_num_lower = [np.zeros(total_plans),np.zeros(total_plans),np.zeros(total_plans)]
        for model_name in ["inception","resnet","mobilenet"]:
            model_cons[i][cons_start[model_name]:cons_start[model_name]+len(ins_size[model_name]["intra"])] = ins_size[model_name]["intra"]
            overall_cons[cons_start[model_name]:cons_start[model_name]+len(ins_size[model_name]["intra"])] = ins_size[model_name]["intra"]
            overall_E[cons_start[model_name]:cons_start[model_name] + len(ins_size[model_name]["intra"])] = ins_size[model_name]["efficiency"]
            ins_num_upper[cons_start[model_name]:cons_start[model_name] + len(ins_size[model_name]["intra"])] = np.floor(C_upper/np.array(ins_size[model_name]["intra"]))
            ins_num_lower[i][cons_start[model_name]:cons_start[model_name] + len(ins_size[model_name]["intra"])] =np.ones(len(ins_size[model_name]["intra"]))
            i = i+1
        return overall_cons,model_cons,overall_E,ins_num_upper.reshape((total_plans,1)),ins_num_lower
    def resource_allocation(self):
        result = self.cpu_const()
        overall_cons = result[0]
        model_cons = result[1]
        overall_E = result[2]
        ins_num_upper = result[3]
        ins_num_lower = result[4]

        Z = cp.Variable((len(overall_cons),1),integer=True)
        obj = cp.Maximize(overall_E@Z)
        prob = cp.Problem(obj,[overall_cons@Z<=C,
                               model_cons[0]@Z<=C_upper,model_cons[1]@Z<=C_upper,model_cons[2]@Z<=C_upper,Z<=ins_num_upper,
                               ins_num_lower[0] @ Z >=1, ins_num_lower[1] @ Z >=1, ins_num_lower[2] @ Z >=1,
                               Z <= ins_num_upper,
                               Z>=np.zeros(shape=(len(overall_cons),1))])
        print(prob.solve(),prob.status)
        print(Z.value)

resource_allocation()




