
#part-lib

import pandapower.plotting as plot
import pandapower.networks as nw
import pandapower as pp
from scipy.io import loadmat
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# load example net (IEEE 9 buses)
import sys
import torch
from torch import nn, optim
import argparse
# from model import GraphEncoder,GraphEncoder_with_attention
from sklearn.metrics.cluster import normalized_mutual_info_score
from tqdm import tqdm
from sklearn import preprocessing
import pandapower.networks as pn
import sys
import matplotlib.pyplot as plt
from scipy import sparse
import utlize
import os
# import plotly.io as pio
# pio.renderers.default = "browser"
try:
    import seaborn
    colors = seaborn.color_palette(n_colors=50)
except:
    colors = ["b", "g", "r", "c", "y"]
# from pandapower.plotting.plotly import simple_plotly,simple_plotly_gen
from utlize import simple_plotly_gen

import json

import sys
startingDir = os.getcwd() # save our current directory

# project path through relative path, parent(parent(current_directory))
proj_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..'))

# This is so Django knows where to find stuff.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iim_dashboard.settings")
sys.path.append(proj_path)

# This is so my local_settings.py gets loaded.
os.chdir(proj_path)
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
from islanding.models import IslandingScheme

from pandapower.plotting.plotly import simple_plotly

# SET VARIABLES

# epochs  = 100
# epochs  = 1000
epochs  = 2000
# epochs  = 4000
# epochs  = 10000
# epochs  = 35000
# kappa   = 1
# kappa   = 2 
# kappa   = 3 
kappa   = 4 
# kappa   = 5 
# kappa   = 6
# kappa   = 7
# kappa   = 8
# kappa   = 9
# kappa   = 10

# net = pp.from_json('CaseIEEE9_GridModel.json')
# net = pp.from_json('MV_CIGRE_GridModel.json')
# net = pp.from_json('MV_Oberrhein_GridModelv2.json')


# pp.diagnostic(new_net, report_style='detailed', warnings_only=False, return_result_dict=True, overload_scaling_factor=0.001, min_r_ohm=0.001, min_x_ohm=0.001, min_r_pu=1e-05, min_x_pu=1e-05, nom_voltage_tolerance=0.3, numba_tolerance=1e-05)

# print('#######################################')


# new_net = pp.networks.case14()
# new_net = pp.networks.case_ieee30()


# def evaluation_mini_imblance(X_hat,output_predict,bus_inc):
#     bus_inc=torch.from_numpy(bus_inc).to(device).unsqueeze(1).float()
#     # output_predict = torch.argmax(X_hat, dim=1)
#     predict_binary_matrix=torch.zeros_like(X_hat)
#     for i  in range(len(output_predict)):
#         predict_binary_matrix[i,output_predict[i]]=1
#     island_inc = torch.abs(torch.mm(predict_binary_matrix.t(), bus_inc))
#     sum_inc=torch.sum(island_inc)

#     return sum_inc,island_inc

def evaluation_mini_imblance(X_hat,output_predict,bus_inc):
    bus_inc=torch.from_numpy(bus_inc).to(device).unsqueeze(1).float()
    # output_predict = torch.argmax(X_hat, dim=1)
    predict_binary_matrix=torch.zeros_like(X_hat)
    for i  in range(len(output_predict)):
        predict_binary_matrix[i,output_predict[i]]=1
    island_inc = torch.mm(predict_binary_matrix.t(), bus_inc)
    # sum_inc=torch.abs(torch.sum(island_inc))
    sum_inc=torch.sum(island_inc)

    return sum_inc,island_inc

def plot_net_with_label(net,file_path='',save_fig=True):
    lc = plot.create_line_collection(net, net.line.index, color=colors[-1], zorder=0) #create lines
    # bc = plot.create_bus_collection(net, net.bus.index, size=80, color=colors[0], zorder=1) #create buses
    labels=list(range(k))
    color_colection_c=[]
    count=1
    for lable in labels:
        count+=1
        single_color=np.where(output_predict==lable)
        color_index=np.array(net.bus.index[single_color])
        color_colection_c+= [plot.create_bus_collection(net, color_index,size=0.2, color=colors[count], zorder=count)]
    if len(h_index_trans)>0:
        lc_trans,pc_trans = plot.create_trafo_collection(net, net.trafo.index, size=0.2,color=colors[-2], zorder=1) #create lines

        c=[lc]+[lc_trans]+[pc_trans]+color_colection_c
    else:
        c = [lc] + color_colection_c
    plot.draw_collections(c, figsize=(10,8))
    if save_fig==True:
        plt.savefig(file_path+'.png')
    # plt.show()

def euclidean_distances(x, all_vec):
    dist=np.zeros(len(all_vec))
    for i in range(len(all_vec)):
        y=all_vec[i]
        dist[i]=np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
    return dist
#part-test case


# Here replace with AIDB connection
# net= new_net
# net.bus.drop(0, inplace=True)
# pp.set_element_status(net, 0, in_service=False)
# pp.plotting.to_html(new_net, filename='islanding/iim_mlst/static/grid_after_islanding/initial.html', show_tables=False)
# simple_plotly_gen(new_net, file_name='islanding/iim_mlst/static/grid_after_islanding/initial2.html')

# net=pp.networks.case5()
# net=pp.networks.case9()
# net=pp.networks.case14()
# net=pp.networks.case24_ieee_rts()
# net=pp.networks.case30()
# net=pp.networks.case39()
# net=pp.networks.case57()
# net=pp.networks.case89pegase()
net=pp.networks.case_ieee30()

# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')

# net=pp.networks.case118()
# net=pp.networks.case_illinois200()
# net=pp.networks.case300()

# print(net.res_load)

# net=pp.networks.case145()
print('#######################################')
print('#######################################')
print(net)
# pp.estimation.estimate(net, algorithm='wls', init='flat', tolerance=1e-06, maximum_iterations=10, calculate_voltage_angles=True, zero_injection='aux_bus', fuse_buses_with_bb_switch='all', **opt_vars)
print('#######################################')
print('#######################################')

print('#######################################')
print('#######################################')
print(pp.rundcpp(net))
# pp.estimation.estimate(net, algorithm='wls', init='flat', tolerance=1e-06, maximum_iterations=10, calculate_voltage_angles=True, zero_injection='aux_bus', fuse_buses_with_bb_switch='all', **opt_vars)
print('#######################################')
print('#######################################')


# ########## Save binary, json, excel pandapower network file ##########
# pp.to_json(net, "net.json")
# # pp.to_pickle(net, "case14.p")
# pp.to_excel(net, "net.xlsx")
# ########## - ##########

# pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
# plt.savefig('islanding/iim_mlst/static/grid_initial/grid.png')

# input("1 Press a key to continue...")

# simple_plotly_gen(net, file_name='islanding/iim_mlst/static/grid_after_islanding/interactive-plotTEST.html')

# ########## Import binary pandapower network file ##########
# net2 = pp.from_pickle("case14_after.p") #relative path
# ########## - ##########

# print('##### Grid before islanding #####')

# net=pp.networks.case39()
# net=pp.networks.case57()
# net=pp.networks.case89pegase()
# net=pp.networks.case118()
# net=pp.networks.case145()
# net=pp.networks.case_illinois200()

# test_file_path='C:/Users/70474/Desktop/meeting_phd/project_va/GridModels_examples/CIGRE_MV_modified.json'
# net= pp.from_json(test_file_path)

######################################################
######################################################
######################################################
######################################################
######################################################
################## FOR GAP METHOD ####################
######################################################
######################################################
######################################################
######################################################
######################################################

# input("2 Press a key to continue...")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wine', help='Dataset to use')
parser.add_argument('-l', '--layers', nargs='+', type=int, default=[128, 64, 128], help='Sparsity Penalty Parameter')
parser.add_argument('-b', '--beta', type=float, default=0.01, help='Sparsity Penalty Parameter')
parser.add_argument('-p', '--rho', type=float, default=0.5, help='Prior rho')
parser.add_argument('-lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('-weight_decay', type=float, default=1e-4, help='weight_decay')#0.00001
parser.add_argument('-epoch', type=int, default=200, help='Number of Training Epochs')
parser.add_argument('-device', type=str, default='cpu', help='Train on GPU or CPU')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# input("3 Press a key to continue...")

# k=2
# num_epoch=500   #number of training time
# num_epoch=3000   #number of training time
num_epoch=epochs   #number of training time

def search_1d(a,lines):
    """find the corresponding index that a=lines[i]"""
    count=0
    result_search=0
    for item in lines:

        if (a == item).all():
            result_search=1
            break
        count += 1
    return count,result_search

#panpower powerflow
# print('#######################################')
# print('#######################################')
pp.runpp(net)
# print('#######################################')
# print('#######################################')

# print('#######################################')
# print('#######################################')
# print(pp.runpp(net))
# print('#######################################')
# print('#######################################')

# input("4 Press a key to continue...")


case_num=len(net.res_bus)
# print(case_num)

# save_fig_path='./save_fig/case_'+str(case_num)
save_fig_path='static/case_'+str(case_num)
# if not os.path.exists(save_fig_path):
#     os.mkdir(save_fig_path)
#generator
gen=net.gen



# print('#######################################')
# print('#######################################')
# simple_plotly_gen(net, file_name='islanding/iim_mlst/static/grid_after_islanding/initial2.html')
# print(gen)
# print('#######################################')
# print('#######################################')
# input("5 Press a key to continue...")

# k=3     #k is number of partitions, should be mannually point.
k=kappa    #k is number of partitions, should be mannually point.
# 
#s_generator
# gen=net.sgen
# k=len(gen)

# # extra_grid
# gen=net.ext_grid
# k=len(gen)

#part-shuffle the index of generator, will be fixed when coherent generator is used
index_store=np.zeros(len(gen))
for i in range(len(gen)):
    inndex_sb=net.gen.bus.values[i]
    a=net.bus.loc[inndex_sb].values
    # b=net.res_bus.loc[to_num].values
    a_index,a_succ=search_1d(a,net.bus.values)
    index_store[i]=a_index
    # b_index, b_succ = search_1d(b, net.res_bus.values)
np.random.shuffle(index_store)
gen_index=torch.tensor(np.int64(index_store))


# # extra_grid
gen=net.ext_grid
# k=len(gen)
index_store=np.zeros(len(gen))
for i in range(len(gen)):
    inndex_sb=net.ext_grid.bus.values[i]
    a=net.bus.loc[inndex_sb].values
    # b=net.res_bus.loc[to_num].values
    a_index,a_succ=search_1d(a,net.bus.values)
    index_store[i]=a_index
    # b_index, b_succ = search_1d(b, net.res_bus.values)
extra_grid_index=torch.tensor(np.int64(index_store))

#part-extract information from pandapower lib, for graph representation purpose
res_transfrom=net.res_trafo.values
h_index_trans=net.trafo.hv_bus.values
l_index_trans=net.trafo.lv_bus.values
nodes=net.res_bus.values   #0 is voltage magnitude
bus_inc=nodes[:,2]
lines=net.line.values   #2,3 are adjacent
res_lines=net.res_line.values  #0,1 are active and reactive power flow

# print('#######################################')
# print('#######################################')
# print('nodes:', nodes) #This is a 4 x n matrix containing the 4 parameters of each bus
# print('#######################################')
# print('bus_inc:', bus_inc) # Keep only the p_mw: resulting active power demand [MW]
# print('#######################################')
# print('lines:', lines)
# print('#######################################')
# print('res_lines:', res_lines)
# print('#######################################')
# print('#######################################')
# input("6 Press a key to continue...")

#######################################
#######################################
#######################################
# The lines are taken to calculate the adjacency matrix (most probably)
#######################################
#######################################
#######################################

min_max_scaler = preprocessing.MinMaxScaler()
nodes = min_max_scaler.fit_transform(nodes)

# print('#######################################')
# print('#######################################')
# print('nodes scaled:', nodes)
# print('#######################################')
# print('#######################################')
# input("7 Press a key to continue...")

S = cosine_similarity(nodes, nodes)
#part-construct the adjacent matrix from lines and transformer
adjacent_matrix=np.zeros_like(S)
# print('#######################################')
# print('#######################################')
# print('adjacent_matrix:', adjacent_matrix)
# print('#######################################')
# print('#######################################')
# input("7b Press a key to continue...")
from_bus_index=np.where(net.line.keys()=='from_bus')[0]
to_bus_index=np.where(net.line.keys()=='to_bus')[0]
# hv_bus_index=np.where(net.trafo.keys()=='hv_bus')[0]
# lv_bus_index=np.where(net.trafo.keys()=='lv_bus')[0]
for i in range(len(lines)):
    from_num=lines[i][from_bus_index]    #2,3

    to_num=lines[i][to_bus_index]  #2 -3
    # if to_num==10:
    #     a=1
    a=net.bus.loc[from_num].values
    b=net.bus.loc[to_num].values
    a_index,a_succ=search_1d(a,net.bus.values)
    b_index, b_succ = search_1d(b, net.bus.values)
    if a_succ and b_succ:
        absolut_value=np.sqrt(res_lines[i][0]**2+res_lines[i][1]**2)+ 1e-7

        # absolut_value = (np.abs(res_lines[i][0]) + np.abs(res_lines[i][2]) )
            # (np.abs(res_lines[i][0]) + np.abs(res_lines[i][2]) +5*lines[i][4])
        # absolut_value =np.abs((res_lines[i][0] + res_lines[i][2]) / 2)
        adjacent_matrix[b_index,a_index]=absolut_value
        adjacent_matrix[a_index, b_index] = absolut_value

    else:
        print('search fail')
        print('current item is '+str(i))
        sys,exit()
    c=1

# adjacent_matrix[gen_index,:]=0
# adjacent_matrix[:,gen_index]=0
# adjacent_matrix[extra_grid_index,:]=0
# adjacent_matrix[:,extra_grid_index]=0

for i in range(len(adjacent_matrix)):
    adjacent_matrix[i, i] = 1
    # a=net.bus[from_num]
    # a=np.where(net.bus['name']==('bus_'+str(from_num)))
    # b = np.where(net.bus['name'] == ('bus_' + str(to_num)))
# #transformer
for i in range(len(h_index_trans)):
    from_num=h_index_trans[i]
    to_num=l_index_trans[i]# switch to transormer
    a = net.bus.loc[from_num].values
    b = net.bus.loc[to_num].values
    a_index, a_succ = search_1d(a, net.bus.values)
    b_index, b_succ = search_1d(b, net.bus.values)
    if a_succ and b_succ:
        # absolut_value=np.sqrt(res_lines[i][0]**2+res_lines[i][1]**2)
        # absolut_value = (np.abs(net.res_trafo.values[i][0]) + np.abs(net.res_trafo.values[i][2]) )
        absolut_value = np.abs((net.res_trafo.values[i][0] + net.res_trafo.values[i][2]) / 2)+ 1e-7
        adjacent_matrix[b_index, a_index] = absolut_value
        adjacent_matrix[a_index, b_index] = absolut_value
# # simple plot of net with existing geocoordinates or generated artificial geocoordinates

#part-convert to Laplcian matrix and degree matrix
#to binary
indices_ad=np.where(adjacent_matrix>0)
tor_adjacent_matrix = torch.from_numpy(adjacent_matrix).to(device)#.cuda()
new_adj=torch.zeros_like(tor_adjacent_matrix)
# indices_ad
new_adj[indices_ad]=1
for i in range(len(new_adj)):
    new_adj[i,i]=0
adjacent_matrix=new_adj.cpu().numpy()
# print('adjacent_matrix:', adjacent_matrix)
# input("adjacent_matrix Press a key to continue...")
#part- construct the input feature of Graph CNN: bus=node, edge=line+transformer;

csc_adjacent_matrix=sparse.csr_matrix(adjacent_matrix)
# print('csc_adjacent_matrix:', csc_adjacent_matrix)
# input("csc_adjacent_matrix Press a key to continue...")
La=utlize.laplacian(csc_adjacent_matrix, normalized=True)
# print('La:', La)
# input("La1 Press a key to continue...")
# La=sparse.csr_matrix(La)
La=utlize.csc2sparsetensor(La).to(device)
# print('La:', La)
# input("La2 Press a key to continue...")
spars_Adj=utlize.csc2sparsetensor(csc_adjacent_matrix).to(device)
D = np.diag(1.0 / np.sqrt(adjacent_matrix.sum(axis=1)))
# print('D:', D)
# input("D Press a key to continue...")
# x_train_numpy=D.dot(S).dot(D) 
# d=np.isnan(x_train_numpy)
X_train=torch.tensor(nodes).float().to(device)
# X_train = torch.tensor(D.dot(S).dot(D)).float().to(device)
layers = [len(X_train)] + args.layers + [k]

# print('#######################################')
# print('#######################################')
# print('X_train:', X_train)
# print('layers:', layers)
# print('k:', k)
# print('La:', La)
# print('#######################################')
# print('#######################################')
# input("8 Press a key to continue...")

#part- initialize graph CNN model
import Graph_cnn_model
model = Graph_cnn_model.GraphEncoder(layers, k,La,len(X_train[0]),device=device,cluster_num=k).to(device)
# import GAP_DLmodel
# model = GAP_DLmodel.GraphEncoder(layers, k).to(device)
# model = GraphEncoder(layers, k).to(device)
# model = GraphEncoder_with_attention(layers, k).to(device)
# if optimzer_potion == 'Adam':


# print('#######################################')
# print('#######################################')
# print('model:', model)
# print('#######################################')
# print('#######################################')
# input("9 Press a key to continue...")

optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0
                       )#weight_decay=args.weight_decay
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=0)
#part- start training

with tqdm(total=num_epoch) as tq:
    for epoch in range(1, num_epoch + 1):

        # n = np.int(epoch / 2000)
        # global lr
        # # if epoch == 0:
        # #     lr = 0.002
        # lr = args.lr * (0.5 ** n)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        # model.train()
        optimizer.zero_grad()
        # X_hat = model(X_train,len(X_train),k)  #normal nn
        X_hat = model(X_train)     #graph CNN

        #loss function define, GAP and min_cut loss
        loss,Loss_cut,loss_balance,loss_bus_inc,loss_gen=model.gap_loss(X_hat,adjacent_matrix,bus_inc,gen_index=gen_index)
        # loss=model.min_cut_loss(adjacent_matrix,X_hat,bus_inc,gen_index=gen_index)

        # loss,rank_loss_score = model.loss(X_hat, X_train, args.beta, args.rho,gen_index=gen_index)
        # nmi = normalized_mutual_info_score(model.get_cluster(), Y)  # , average_method='arithmetic'
        # loss=Graph_cnn_model.CutLoss.apply(X_hat,spars_Adj)

        loss.backward()
        # print(grads['y'])
        optimizer.step()
        tq.set_postfix(loss='{:.3f}'.format(loss), nmi='{:.3f}'.format(1), rank_loss='{:.4f}'.format(1))
        # tq.set_postfix(loss='{:.3f}'.format(loss),loss_cut='{:.3f}'.format(Loss_cut), loss_balance='{:.3f}'.format(loss_balance), loss_bus_inc='{:.4f}'.format(loss_bus_inc),
        #                loss_gen='{:.4f}'.format(loss_gen))
        # tq.set_postfix(loss='{:.3f}'.format(loss), loss_cut='{:.3f}'.format(Loss_cut),
        #                loss_balance='{:.3f}'.format(loss_balance), loss_bus_inc='{:.4f}'.format(loss_bus_inc))
        # tq.set_postfix(loss='{:.3f}'.format(loss[0]), nmi='{:.3f}'.format(1),rank_loss='{:.4f}'.format(1))
        tq.update()

# plot predict
# print('#######################################')
# print('#######################################')
# print('X_hat:', X_hat)
# print('#######################################')
# print('#######################################')
# input("10 Press a key to continue...")

output_predict=torch.argmax(X_hat,dim=1)
output_predict=output_predict.cpu().numpy()
# generator_results=output_predict[gen_index]
# print(gen_index)

# print('#######################################')
# print('#######################################')
# print('output_predict2:', output_predict)
# print('#######################################')
# print('#######################################')
# input("11 Press a key to continue...")



#part- evaluation

# Initiate here variables to hold the values that will be stored in db #
island_imbalance_initial = []
total_imbalance_initial = 0
# lines = 0
##

# print('|###########################################|')

# print(type(X_hat))
# print(X_hat)
# print(type(output_predict))
# print(output_predict)
# print(type(bus_inc))
# print(bus_inc)

# print('|###########################################|')

sum_inc,island_inc=evaluation_mini_imblance(X_hat,output_predict,bus_inc)
for i in range(len(island_inc)):
    print('island imbalance ' +str(i)+': '+str(island_inc[i]))
    island_imbalance_initial.append( island_inc[i].numpy()[0] )
print('total island imbalance= '+str(sum_inc) )

total_imbalance_initial = sum_inc
#part- plot

# plot_net_with_label(net,file_path=save_fig_path+'/case_'+str(case_num)+'_cluster')
#first step, merge all

#modify transformer
# for i in range(len(h_index_trans)):
#     from_num=h_index_trans[i]
#     to_num=l_index_trans[i]# switch to transormer
#     a = net.bus.loc[from_num].values
#     b = net.bus.loc[to_num].values
#     a_index, a_succ = search_1d(a, net.bus.values)
#     b_index, b_succ = search_1d(b, net.bus.values)
#     if a_succ and b_succ:
#         # absolut_value=np.sqrt(res_lines[i][0]**2+res_lines[i][1]**2)
#         # absolut_value = (np.abs(net.res_trafo.values[i][0]) + np.abs(net.res_trafo.values[i][2]) )
#         # absolut_value = np.abs((net.res_trafo.values[i][0] + net.res_trafo.values[i][2]) / 2)
#         adjacent_matrix[b_index, a_index] = 1
#         adjacent_matrix[a_index, b_index] = 1
# #part- merge vertex to their nearest
# features=model.outputs['lin2'].detach().cpu().numpy().squeeze()
# merge_iter=4
# for j  in range(merge_iter):
#     for i in range(len(adjacent_matrix)):
#         current_index=i
#         current_line=adjacent_matrix[i]
#         index_adj=np.where(current_line>0)
#         deleted_index=np.delete(index_adj,np.where(current_index==index_adj[0]))

#         #search neighbor
#         if output_predict[current_index] in output_predict[deleted_index]:
#             continue
#         else:# search the nearest vertex
#             current_feature=features[current_index]
#             # for neighbor_vertex in features[index_adj]:
#             dist=euclidean_distances(current_feature,features[index_adj])
#             dist[np.where(current_index==index_adj[0])]=float("inf")  #ignore it self
#             min_dist_index=np.argmin(dist)
#             output_predict[current_index]=output_predict[index_adj][min_dist_index]



#part-cut line
line_to_cut=[]
for i in range(len(adjacent_matrix)):
    current_index=i
    current_line=adjacent_matrix[i]
    index_adj=np.where(current_line>0)
    for neibhor_vertex in index_adj[0]:
        #check  whether neighbor vertex have same label
        if output_predict[current_index]==output_predict[neibhor_vertex]:
            continue
        else:
            line_to_cut+=[[current_index,neibhor_vertex]]
number_lines=0
#apply cut on pandapower
lines_to_cut=[]

for single_line_tocut in line_to_cut:
    # tem_bus_va1 = net.bus.values[single_line_tocut[0]]   #bus_name
    # tem_bus1=search_1d(tem_bus_va1,net.bus.values)
    # # tem_bus1=net.bus.loc[single_line_tocut[0]]['name']
    # line_num1 = np.where(net.line['from_bus'] == tem_bus1)
    #
    # tem_bus_va2 = net.bus.values[single_line_tocut[1]]  # bus_name
    # tem_bus2 = search_1d(tem_bus_va2, net.bus.values)
    # # tem_bus2 = net.bus.values[single_line_tocut[1]][3]
    # line_num2 = np.where(net.line['to_bus'] == tem_bus2)
    #
    # #
    # # line_num1=np.where(net.line['from_bus']==single_line_tocut[0] )
    # # line_num2=np.where(net.line['to_bus']==single_line_tocut[1])
    tem_bus1=net.bus.index[single_line_tocut[0]]
    line_num1 = np.where(net.line['from_bus'] == tem_bus1)
    tem_bus2 = net.bus.index[single_line_tocut[1]]
    line_num2 = np.where(net.line['to_bus'] == tem_bus2)
    line_num =list(set(line_num1[0]) & set(line_num2[0]))
    line_num = np.array(line_num)


    if len(np.array(line_num))==0:
        # # tem_bus1 = net.bus.values[single_line_tocut[0]][3]
        # # tem_bus1 = net.bus.loc[single_line_tocut[0]]['name']
        # line_num1 = np.where(net.line['to_bus'] == tem_bus1)
        # # tem_bus2 = net.bus.loc[single_line_tocut[1]]['name']
        # line_num2 = np.where(net.line['from_bus'] == tem_bus2)
        #
        # #
        # # line_num1=np.where(net.line['from_bus']==single_line_tocut[0] )
        # # line_num2=np.where(net.line['to_bus']==single_line_tocut[1])
        # line_num = list(set(line_num1[0]) & set(line_num2[0]))
        # line_num = np.array(line_num)
        # if len(np.array(line_num)) == 0:
            continue
    else:
        # line_num=line_num#[0]
        if len(line_num)>1:
            for single_line_num in line_num:
                line_num = net.line.index[single_line_num]
                lines_to_cut += [line_num]
                number_lines += 1
        else:
            line_num = net.line.index[line_num[0]]
            lines_to_cut+=[line_num]
            number_lines += 1
        # line_num1 = np.where(net.line['from_bus'] == single_line_tocut[1]) & (net.line['to_bus'] == [0])
        # line_num1=np.array(line_num1)
    #
    # line_num=net.line.loc[line_num].name
# pp.drop_lines(net, [line_num])
pp.drop_lines(net, lines_to_cut)

#drop transformer
lines_to_cut=[]
for single_line_tocut in line_to_cut:

    tem_bus1=net.bus.index[single_line_tocut[0]]
    line_num1 = np.where(net.trafo['hv_bus'] == tem_bus1)
    tem_bus2 = net.bus.index[single_line_tocut[1]]
    line_num2 = np.where(net.trafo['lv_bus'] == tem_bus2)
    line_num =list(set(line_num1[0]) & set(line_num2[0]))
    line_num = np.array(line_num)


    if len(np.array(line_num))==0:
        # # tem_bus1 = net.bus.values[single_line_tocut[0]][3]
        # # tem_bus1 = net.bus.loc[single_line_tocut[0]]['name']
        # line_num1 = np.where(net.line['to_bus'] == tem_bus1)
        # # tem_bus2 = net.bus.loc[single_line_tocut[1]]['name']
        # line_num2 = np.where(net.line['from_bus'] == tem_bus2)
        #
        # #
        # # line_num1=np.where(net.line['from_bus']==single_line_tocut[0] )
        # # line_num2=np.where(net.line['to_bus']==single_line_tocut[1])
        # line_num = list(set(line_num1[0]) & set(line_num2[0]))
        # line_num = np.array(line_num)
        # if len(np.array(line_num)) == 0:
            continue
    else:
        # line_num=line_num#[0]
        line_num = net.trafo.index[line_num[0]]
        lines_to_cut+=[line_num]
        number_lines += 1
        # line_num1 = np.where(net.line['from_bus'] == single_line_tocut[1]) & (net.line['to_bus'] == [0])
        # line_num1=np.array(line_num1)
    #
    # line_num=net.line.loc[line_num].name
# pp.drop_lines(net, [line_num])
pp.drop_trafos(net, lines_to_cut)

#part- plot
# plot_net_with_label(net)
# plot_net_with_label(net,file_path=save_fig_path+'/case_'+str(case_num)+'_cut')

# pp.to_pickle(net, "case14_after.p")
# pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
# plt.savefig('grid_after.png')

#part- evaluation again on the result after cutting
sum_inc,island_inc=evaluation_mini_imblance(X_hat,output_predict,bus_inc)

# Initiate here variables to hold the values that will be stored in db #
island_imbalance_after = []
total_imbalance_after = 0
lines = 0
##

for i in range(len(island_inc)):
    print('island imbalance ' +str(i)+': '+str(island_inc[i]))
    island_imbalance_after.append( island_inc[i].numpy()[0] )
print('total island imbalance= '+str(sum_inc) )
print('number of lines to cut= '+str(number_lines))

total_imbalance_after = sum_inc
lines = number_lines

#part-run pp to test the result
pp.runpp(net)
#plot the result with generator
# simple_plotly_gen(net,file_name=save_fig_path+'/temp-plot.html')
# print('#######################################')
# print(net.res_bus)
# print('#######################################')
# print('powerflow is', pp.runpp(net, algorithm='nr', calculate_voltage_angles='auto', init='auto', max_iteration='auto', tolerance_mva=1e-08, trafo_model='t', trafo_loading='current', enforce_q_lims=False, check_connectivity=True, voltage_depend_loads=True, consider_line_temperature=False) )
# print('dc powerflow is', pp.rundcpp(net, trafo_model='t', trafo_loading='current', recycle=None, check_connectivity=True, switch_rx_ratio=2, trafo3w_losses='hv'))

# HERE
# print(island_imbalance_after)
# pp.runpp(net)

# print('#######################################')
# print('#######################################')
# print('pp.runpp(net)')
# print('#######################################')
# print(pp.runpp(net))
# print('#######################################')
# print('#######################################')

# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')

# print('#######################################')
# print('#######################################')
# print('pp.rundcpp(net)')
# print(pp.rundcpp(net))
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')

# print('get_connected_buses')
# print(pp.get_connected_buses(net,1))
# print(pp.get_connected_elements_dict(net,1))
# print(net.res_load)
# print(net.res_gen)
# print('#######################################')
# print('#######################################')

jnet=pp.to_json(net, filename=None, encryption_key=None)
method='GAP'
# IslandingScheme.objects.all().delete()
IslandingScheme.objects.create( method_name=method, island_imbalance=island_imbalance_initial, total_imbalance=total_imbalance_initial, island_imbalance_after_cut=island_imbalance_after, total_imbalance_after_cut=total_imbalance_after, lines_to_cut=lines, grid=jnet )

pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
plt.savefig('islanding/iim_mlst/static/grid_after_islanding/grid_after_'+method+'.png')

pp.plotting.to_html(net, filename='islanding/iim_mlst/static/grid_after_islanding/interactive-plot_'+method+'.html', show_tables=False)
simple_plotly_gen(net, file_name='islanding/iim_mlst/static/grid_after_islanding/interactive-plot2_'+method+'.html')

print('END OF GAP METHOD')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')

#################### END GAP METHOD ####################

# #################### FOR MIN-CUT METHOD ####################

# net= new_net
# net=pp.from_json('test.json')
# net = pp.from_json('CaseIEEE9_GridModel.json')
# net=pp.networks.case5()
# net=pp.networks.case9()
# net=pp.networks.case14()
# net=pp.networks.case24_ieee_rts()
# net=pp.networks.case30()
# net=pp.networks.case39()
# net=pp.networks.case57()
# net=pp.networks.case89pegase()
net=pp.networks.case_ieee30()

# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')

# net=pp.networks.case118()
# net=pp.networks.case_illinois200()
# net=pp.networks.case300()

# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')
# print('#######################################')

# net=pp.networks.case145()



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wine', help='Dataset to use')
parser.add_argument('-l', '--layers', nargs='+', type=int, default=[128, 64, 128], help='Sparsity Penalty Parameter')
parser.add_argument('-b', '--beta', type=float, default=0.01, help='Sparsity Penalty Parameter')
parser.add_argument('-p', '--rho', type=float, default=0.5, help='Prior rho')
parser.add_argument('-lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('-weight_decay', type=float, default=1e-4, help='weight_decay')#0.00001
parser.add_argument('-epoch', type=int, default=200, help='Number of Training Epochs')
parser.add_argument('-device', type=str, default='cpu', help='Train on GPU or CPU')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')
args = parser.parse_args()


# k=2

# num_epoch=500   #number of training time
num_epoch=epochs   #number of training time
# num_epoch=4500   #number of training time

def search_1d(a,lines):
    """find the corresponding index that a=lines[i]"""
    count=0
    result_search=0
    for item in lines:

        if (a == item).all():
            result_search=1
            break
        count += 1
    return count,result_search
#panpower powerflow
pp.runpp(net)


case_num=len(net.res_bus)
# save_fig_path='./save_fig/case_'+str(case_num)
save_fig_path='static/case_'+str(case_num)
# if not os.path.exists(save_fig_path):
#     os.mkdir(save_fig_path)
#generator
gen=net.gen

k=kappa    #k is number of partitions, should be mannually point.
# k=8  #k is number of partitions, should be mannually point.

#s_generator
# gen=net.sgen
# k=len(gen)

# # extra_grid
# gen=net.ext_grid
# k=len(gen)

#part-shuffle the index of generator, will be fixed when coherent generator is used
index_store=np.zeros(len(gen))
for i in range(len(gen)):
    inndex_sb=net.gen.bus.values[i]
    a=net.bus.loc[inndex_sb].values
    # b=net.res_bus.loc[to_num].values
    a_index,a_succ=search_1d(a,net.bus.values)
    index_store[i]=a_index
    # b_index, b_succ = search_1d(b, net.res_bus.values)
np.random.shuffle(index_store)
gen_index=torch.tensor(np.int64(index_store))


# # extra_grid
gen=net.ext_grid
# k=len(gen)
index_store=np.zeros(len(gen))
for i in range(len(gen)):
    inndex_sb=net.ext_grid.bus.values[i]
    a=net.bus.loc[inndex_sb].values
    # b=net.res_bus.loc[to_num].values
    a_index,a_succ=search_1d(a,net.bus.values)
    index_store[i]=a_index
    # b_index, b_succ = search_1d(b, net.res_bus.values)
extra_grid_index=torch.tensor(np.int64(index_store))

#part-extract information from pandapower lib, for graph representation purpose
res_transfrom=net.res_trafo.values
h_index_trans=net.trafo.hv_bus.values
l_index_trans=net.trafo.lv_bus.values
nodes=net.res_bus.values   #0 is voltage magnitude
bus_inc=nodes[:,2]
lines=net.line.values   #2,3 are adjacent
res_lines=net.res_line.values  #0,1 are active and reactive power flow


min_max_scaler = preprocessing.MinMaxScaler()
nodes = min_max_scaler.fit_transform(nodes)

S = cosine_similarity(nodes, nodes)
#part-construct the adjacent matrix from lines and transformer
adjacent_matrix=np.zeros_like(S)
from_bus_index=np.where(net.line.keys()=='from_bus')[0]
to_bus_index=np.where(net.line.keys()=='to_bus')[0]
# hv_bus_index=np.where(net.trafo.keys()=='hv_bus')[0]
# lv_bus_index=np.where(net.trafo.keys()=='lv_bus')[0]
for i in range(len(lines)):
    from_num=lines[i][from_bus_index]    #2,3

    to_num=lines[i][to_bus_index]  #2 -3
    # if to_num==10:
    #     a=1
    a=net.bus.loc[from_num].values
    b=net.bus.loc[to_num].values
    a_index,a_succ=search_1d(a,net.bus.values)
    b_index, b_succ = search_1d(b, net.bus.values)
    if a_succ and b_succ:
        absolut_value=np.sqrt(res_lines[i][0]**2+res_lines[i][1]**2)+ 1e-7

        # absolut_value = (np.abs(res_lines[i][0]) + np.abs(res_lines[i][2]) )
            # (np.abs(res_lines[i][0]) + np.abs(res_lines[i][2]) +5*lines[i][4])
        # absolut_value =np.abs((res_lines[i][0] + res_lines[i][2]) / 2)
        adjacent_matrix[b_index,a_index]=absolut_value
        adjacent_matrix[a_index, b_index] = absolut_value

    else:
        print('search fail')
        print('current item is '+str(i))
        sys,exit()
    c=1

# adjacent_matrix[gen_index,:]=0
# adjacent_matrix[:,gen_index]=0
# adjacent_matrix[extra_grid_index,:]=0
# adjacent_matrix[:,extra_grid_index]=0

for i in range(len(adjacent_matrix)):
    adjacent_matrix[i, i] = 1
    # a=net.bus[from_num]
    # a=np.where(net.bus['name']==('bus_'+str(from_num)))
    # b = np.where(net.bus['name'] == ('bus_' + str(to_num)))
# #transformer
for i in range(len(h_index_trans)):
    from_num=h_index_trans[i]
    to_num=l_index_trans[i]# switch to transormer
    a = net.bus.loc[from_num].values
    b = net.bus.loc[to_num].values
    a_index, a_succ = search_1d(a, net.bus.values)
    b_index, b_succ = search_1d(b, net.bus.values)
    if a_succ and b_succ:
        # absolut_value=np.sqrt(res_lines[i][0]**2+res_lines[i][1]**2)
        # absolut_value = (np.abs(net.res_trafo.values[i][0]) + np.abs(net.res_trafo.values[i][2]) )
        absolut_value = np.abs((net.res_trafo.values[i][0] + net.res_trafo.values[i][2]) / 2)+ 1e-7
        adjacent_matrix[b_index, a_index] = absolut_value
        adjacent_matrix[a_index, b_index] = absolut_value
# # simple plot of net with existing geocoordinates or generated artificial geocoordinates

#part-convert to Laplcian matrix and degree matrix
#to binary
indices_ad=np.where(adjacent_matrix>0)
tor_adjacent_matrix = torch.from_numpy(adjacent_matrix).to(device)#.cuda()
new_adj=torch.zeros_like(tor_adjacent_matrix)
# indices_ad
new_adj[indices_ad]=1
for i in range(len(new_adj)):
    new_adj[i,i]=0
adjacent_matrix=new_adj.cpu().numpy()
#part- construct the input feature of Graph CNN: bus=node, edge=line+transformer;

csc_adjacent_matrix=sparse.csr_matrix(adjacent_matrix)
La=utlize.laplacian(csc_adjacent_matrix, normalized=True)
# La=sparse.csr_matrix(La)
La=utlize.csc2sparsetensor(La).to(device)
spars_Adj=utlize.csc2sparsetensor(csc_adjacent_matrix).to(device)
D = np.diag(1.0 / np.sqrt(adjacent_matrix.sum(axis=1)))
# x_train_numpy=D.dot(S).dot(D)
# d=np.isnan(x_train_numpy)
X_train=torch.tensor(nodes).float().to(device)
# X_train = torch.tensor(D.dot(S).dot(D)).float().to(device)
layers = [len(X_train)] + args.layers + [k]


#part- initialize graph CNN model
import Graph_cnn_model
model = Graph_cnn_model.GraphEncoder(layers, k,La,len(X_train[0]),device=device,cluster_num=k).to(device)
# import GAP_DLmodel
# model = GAP_DLmodel.GraphEncoder(layers, k).to(device)
# model = GraphEncoder(layers, k).to(device)
# model = GraphEncoder_with_attention(layers, k).to(device)
# if optimzer_potion == 'Adam':



optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0
                       )#weight_decay=args.weight_decay
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,weight_decay=0)
#part- start training

with tqdm(total=num_epoch) as tq:
    for epoch in range(1, num_epoch + 1):

        # n = np.int(epoch / 2000)
        # global lr
        # # if epoch == 0:
        # #     lr = 0.002
        # lr = args.lr * (0.5 ** n)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        # model.train()
        optimizer.zero_grad()
        # X_hat = model(X_train,len(X_train),k)  #normal nn
        X_hat = model(X_train)     #graph CNN

        #loss function define, GAP and min_cut loss
        # loss,Loss_cut,loss_balance,loss_bus_inc,loss_gen=model.gap_loss(X_hat,adjacent_matrix,bus_inc,gen_index=gen_index)
        loss=model.min_cut_loss(adjacent_matrix,X_hat,bus_inc,gen_index=gen_index)

        # loss,rank_loss_score = model.loss(X_hat, X_train, args.beta, args.rho,gen_index=gen_index)
        # nmi = normalized_mutual_info_score(model.get_cluster(), Y)  # , average_method='arithmetic'
        # loss=Graph_cnn_model.CutLoss.apply(X_hat,spars_Adj)

        loss.backward()
        # print(grads['y'])
        optimizer.step()
        tq.set_postfix(loss='{:.3f}'.format(loss), nmi='{:.3f}'.format(1), rank_loss='{:.4f}'.format(1))
        # tq.set_postfix(loss='{:.3f}'.format(loss),loss_cut='{:.3f}'.format(Loss_cut), loss_balance='{:.3f}'.format(loss_balance), loss_bus_inc='{:.4f}'.format(loss_bus_inc),
        #                loss_gen='{:.4f}'.format(loss_gen))
        # tq.set_postfix(loss='{:.3f}'.format(loss), loss_cut='{:.3f}'.format(Loss_cut),
        #                loss_balance='{:.3f}'.format(loss_balance), loss_bus_inc='{:.4f}'.format(loss_bus_inc))
        # tq.set_postfix(loss='{:.3f}'.format(loss[0]), nmi='{:.3f}'.format(1),rank_loss='{:.4f}'.format(1))
        tq.update()
# plot predict
output_predict=torch.argmax(X_hat,dim=1)
output_predict=output_predict.cpu().numpy()
# generator_results=output_predict[gen_index]
print(gen_index)

#part- evaluation

# Initiate here variables to hold the values that will be stored in db #
island_imbalance_initial = []
total_imbalance_initial = 0
# lines = 0
##


sum_inc,island_inc=evaluation_mini_imblance(X_hat,output_predict,bus_inc)
for i in range(len(island_inc)):
    print('island imbalance ' +str(i)+': '+str(island_inc[i]))
    # island_imbalance_initial.append( int(island_inc[i].numpy()[0]) )
    island_imbalance_initial.append( island_inc[i].numpy()[0] )
print('total island imbalance= '+str(sum_inc) )

total_imbalance_initial = sum_inc
#part- plot

# plot_net_with_label(net,file_path=save_fig_path+'/case_'+str(case_num)+'_cluster')
#first step, merge all

#modify transformer
for i in range(len(h_index_trans)):
    from_num=h_index_trans[i]
    to_num=l_index_trans[i]# switch to transormer
    a = net.bus.loc[from_num].values
    b = net.bus.loc[to_num].values
    a_index, a_succ = search_1d(a, net.bus.values)
    b_index, b_succ = search_1d(b, net.bus.values)
    if a_succ and b_succ:
        # absolut_value=np.sqrt(res_lines[i][0]**2+res_lines[i][1]**2)
        # absolut_value = (np.abs(net.res_trafo.values[i][0]) + np.abs(net.res_trafo.values[i][2]) )
        # absolut_value = np.abs((net.res_trafo.values[i][0] + net.res_trafo.values[i][2]) / 2)
        adjacent_matrix[b_index, a_index] = 1
        adjacent_matrix[a_index, b_index] = 1
#part- merge vertex to their nearest
features=model.outputs['lin2'].detach().cpu().numpy().squeeze()
merge_iter=4
for j  in range(merge_iter):
    for i in range(len(adjacent_matrix)):
        current_index=i
        current_line=adjacent_matrix[i]
        index_adj=np.where(current_line>0)
        deleted_index=np.delete(index_adj,np.where(current_index==index_adj[0]))

        #search neighbor
        if output_predict[current_index] in output_predict[deleted_index]:
            continue
        else:# search the nearest vertex
            current_feature=features[current_index]
            # for neighbor_vertex in features[index_adj]:
            dist=euclidean_distances(current_feature,features[index_adj])
            dist[np.where(current_index==index_adj[0])]=float("inf")  #ignore it self
            min_dist_index=np.argmin(dist)
            output_predict[current_index]=output_predict[index_adj][min_dist_index]



#part-cut line
line_to_cut=[]
for i in range(len(adjacent_matrix)):
    current_index=i
    current_line=adjacent_matrix[i]
    index_adj=np.where(current_line>0)
    for neibhor_vertex in index_adj[0]:
        #check  whether neighbor vertex have same label
        if output_predict[current_index]==output_predict[neibhor_vertex]:
            continue
        else:
            line_to_cut+=[[current_index,neibhor_vertex]]
number_lines=0
#apply cut on pandapower
lines_to_cut=[]

for single_line_tocut in line_to_cut:
    # tem_bus_va1 = net.bus.values[single_line_tocut[0]]   #bus_name
    # tem_bus1=search_1d(tem_bus_va1,net.bus.values)
    # # tem_bus1=net.bus.loc[single_line_tocut[0]]['name']
    # line_num1 = np.where(net.line['from_bus'] == tem_bus1)
    #
    # tem_bus_va2 = net.bus.values[single_line_tocut[1]]  # bus_name
    # tem_bus2 = search_1d(tem_bus_va2, net.bus.values)
    # # tem_bus2 = net.bus.values[single_line_tocut[1]][3]
    # line_num2 = np.where(net.line['to_bus'] == tem_bus2)
    #
    # #
    # # line_num1=np.where(net.line['from_bus']==single_line_tocut[0] )
    # # line_num2=np.where(net.line['to_bus']==single_line_tocut[1])
    tem_bus1=net.bus.index[single_line_tocut[0]]
    line_num1 = np.where(net.line['from_bus'] == tem_bus1)
    tem_bus2 = net.bus.index[single_line_tocut[1]]
    line_num2 = np.where(net.line['to_bus'] == tem_bus2)
    line_num =list(set(line_num1[0]) & set(line_num2[0]))
    line_num = np.array(line_num)


    if len(np.array(line_num))==0:
        # # tem_bus1 = net.bus.values[single_line_tocut[0]][3]
        # # tem_bus1 = net.bus.loc[single_line_tocut[0]]['name']
        # line_num1 = np.where(net.line['to_bus'] == tem_bus1)
        # # tem_bus2 = net.bus.loc[single_line_tocut[1]]['name']
        # line_num2 = np.where(net.line['from_bus'] == tem_bus2)
        #
        # #
        # # line_num1=np.where(net.line['from_bus']==single_line_tocut[0] )
        # # line_num2=np.where(net.line['to_bus']==single_line_tocut[1])
        # line_num = list(set(line_num1[0]) & set(line_num2[0]))
        # line_num = np.array(line_num)
        # if len(np.array(line_num)) == 0:
            continue
    else:
        # line_num=line_num#[0]
        if len(line_num)>1:
            for single_line_num in line_num:
                line_num = net.line.index[single_line_num]
                lines_to_cut += [line_num]
                number_lines += 1
        else:
            line_num = net.line.index[line_num[0]]
            lines_to_cut+=[line_num]
            number_lines += 1
        # line_num1 = np.where(net.line['from_bus'] == single_line_tocut[1]) & (net.line['to_bus'] == [0])
        # line_num1=np.array(line_num1)
    #
    # line_num=net.line.loc[line_num].name
# pp.drop_lines(net, [line_num])
pp.drop_lines(net, lines_to_cut)

#drop transformer
lines_to_cut=[]
for single_line_tocut in line_to_cut:

    tem_bus1=net.bus.index[single_line_tocut[0]]
    line_num1 = np.where(net.trafo['hv_bus'] == tem_bus1)
    tem_bus2 = net.bus.index[single_line_tocut[1]]
    line_num2 = np.where(net.trafo['lv_bus'] == tem_bus2)
    line_num =list(set(line_num1[0]) & set(line_num2[0]))
    line_num = np.array(line_num)


    if len(np.array(line_num))==0:
        # # tem_bus1 = net.bus.values[single_line_tocut[0]][3]
        # # tem_bus1 = net.bus.loc[single_line_tocut[0]]['name']
        # line_num1 = np.where(net.line['to_bus'] == tem_bus1)
        # # tem_bus2 = net.bus.loc[single_line_tocut[1]]['name']
        # line_num2 = np.where(net.line['from_bus'] == tem_bus2)
        #
        # #
        # # line_num1=np.where(net.line['from_bus']==single_line_tocut[0] )
        # # line_num2=np.where(net.line['to_bus']==single_line_tocut[1])
        # line_num = list(set(line_num1[0]) & set(line_num2[0]))
        # line_num = np.array(line_num)
        # if len(np.array(line_num)) == 0:
            continue
    else:
        # line_num=line_num#[0]
        line_num = net.trafo.index[line_num[0]]
        lines_to_cut+=[line_num]
        number_lines += 1
        # line_num1 = np.where(net.line['from_bus'] == single_line_tocut[1]) & (net.line['to_bus'] == [0])
        # line_num1=np.array(line_num1)
    #
    # line_num=net.line.loc[line_num].name
# pp.drop_lines(net, [line_num])
pp.drop_trafos(net, lines_to_cut)

#part- plot
# plot_net_with_label(net)
# plot_net_with_label(net,file_path=save_fig_path+'/case_'+str(case_num)+'_cut')

# pp.to_pickle(net, "case14_after.p")
# pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
# plt.savefig('grid_after.png')

#part- evaluation again on the result after cutting
sum_inc,island_inc=evaluation_mini_imblance(X_hat,output_predict,bus_inc)

# Initiate here variables to hold the values that will be stored in db #
island_imbalance_after = []
total_imbalance_after = 0
lines = 0
##

for i in range(len(island_inc)):
    print('island imbalance ' +str(i)+': '+str(island_inc[i]))
    island_imbalance_after.append( island_inc[i].numpy()[0] )
print('total island imbalance= '+str(sum_inc) )
print('number of lines to cut= '+str(number_lines))

total_imbalance_after = sum_inc
lines = number_lines

#part-run pp to test the result
pp.runpp(net)
#plot the result with generator
# simple_plotly_gen(net, file_name='islanding/iim_mlst/static/grid_after_islanding/interactive-plot.html')
# print(net.res_bus)

# print('powerflow is', pp.runpp(net, algorithm='nr', calculate_voltage_angles='auto', init='auto', max_iteration='auto', tolerance_mva=1e-08, trafo_model='t', trafo_loading='current', enforce_q_lims=False, check_connectivity=True, voltage_depend_loads=True, consider_line_temperature=False) )
# print('dc powerflow is', pp.rundcpp(net, trafo_model='t', trafo_loading='current', recycle=None, check_connectivity=True, switch_rx_ratio=2, trafo3w_losses='hv'))

# HERE
# print(island_imbalance_after)
jnet=pp.to_json(net, filename=None, encryption_key=None)
method='MIN-CUT'
# IslandingScheme.objects.all().delete()
IslandingScheme.objects.create( method_name=method, island_imbalance=island_imbalance_initial, total_imbalance=total_imbalance_initial, island_imbalance_after_cut=island_imbalance_after, total_imbalance_after_cut=total_imbalance_after, lines_to_cut=lines, grid=jnet )

pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
plt.savefig('islanding/iim_mlst/static/grid_after_islanding/grid_after_'+method+'.png')

pp.plotting.to_html(net, filename='islanding/iim_mlst/static/grid_after_islanding/interactive-plot_'+method+'.html', show_tables=False)
# simple_plotly_gen(net, file_name='islanding/iim_mlst/static/grid_after_islanding/interactive-plot2_'+method+'.html') 

print('END OF MIN-CUT METHOD')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')
print('#######################################')

#################### END MIN-CUT METHOD ####################

# otsc_netpp = pp.from_json('Case5_Ring_0_results_0.json', convert=True, encryption_key=None)
# method='OTSC'

# pp.plotting.to_html(otsc_netpp, filename='islanding/iim_mlst/static/grid_after_islanding/interactive-plot_'+method+'.html', show_tables=False)
# simple_plotly_gen(otsc_netpp, file_name='islanding/iim_mlst/static/grid_after_islanding/interactive-plot2_'+method+'.html')


# otsc_net = pp.to_json(otsc_netpp, filename=None, encryption_key=None)

# # with open('test.json') as f:
# #         try:
# #             otsc_net = json.load(f)
# #         except:
# #             otsc_net = {}
            
# # otsc_net = open('test.json',"a")

# IslandingScheme.objects.create( method_name='OTSC', island_imbalance=island_imbalance_initial, total_imbalance=total_imbalance_initial, island_imbalance_after_cut=island_imbalance_after, total_imbalance_after_cut=total_imbalance_after, lines_to_cut=lines, grid=otsc_net )

# pp.plotting.to_html(otsc_net, filename='islanding/iim_mlst/static/grid_after_islanding/interactive-plot_'+method+'.html', show_tables=False)
# simple_plotly_gen(otsc_net, file_name='islanding/iim_mlst/static/grid_after_islanding/interactive-plot2_'+method+'.html')

# IslandingScheme.objects.all().delete()


########## The plot should be on the views.py. ##########
# mygrid = IslandingScheme.objects.all().last().grid

# with open('islanding/iim_mlst/static/grid_after_islanding/tmp_grid.txt', 'w') as f:
#     f.write(mygrid)

# net_after = pp.from_json('islanding/iim_mlst/static/grid_after_islanding/tmp_grid.txt')

# pp.plotting.simple_plot(net_after, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
# plt.savefig('islanding/iim_mlst/static/grid_after_islanding/grid_after.png')

# os.remove('islanding/iim_mlst/static/grid_after_islanding/tmp_grid.txt')
######################################################################

print('Done.')