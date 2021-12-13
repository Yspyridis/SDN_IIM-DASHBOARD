import pandapower.plotting as plot
import pandapower.networks as nw
import pandapower as pp
from scipy.io import loadmat
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import torch
from torch import nn, optim
import argparse
from sklearn.metrics.cluster import normalized_mutual_info_score
from tqdm import tqdm
from sklearn import preprocessing
import pandapower.networks as pn
import sys
import matplotlib.pyplot as plt
from scipy import sparse
import utilities
import os

try:
    import seaborn
    colors = seaborn.color_palette(n_colors=50)
except:
    colors = ["b", "g", "r", "c", "y"]

from utilities import simple_plotly_gen
import json
import sys

import http.client
import pika
import uuid

from datetime import timezone
import datetime


startingDir = os.getcwd() # save our current directory

# project path through relative path, parent(parent(current_directory))
proj_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..'))

# This is so Django knows where to find stuff.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iim_dashboard.settings")
sys.path.append(proj_path)

# This is so local_settings.py gets loaded.
os.chdir(proj_path)
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
from islanding.models import IslandingScheme
from pandapower.plotting.plotly import simple_plotly


#################### VARIABLES ##########################

# training epochs
#################### training epochs ####################
epochs  = 1000
#########################################################

#################### number of clusters #################
kappa   = 2 
#########################################################


#################### rmq json format ####################
rmq_json = {
    "messageId": "",
    "name": "islanding_result",
    "payload": "",
    "payloadType": "Payload",
    "source": "PublisherWithPipeline",
    "uc": "UC0",
    "utcTimestamp": ""
}
#########################################################


# START

########################### calculate total imbalance ###########################
# def evaluation_mini_imblance(X_hat,output_predict,bus_inc):
#     bus_inc=torch.from_numpy(bus_inc).to(device).unsqueeze(1).float()
#     # output_predict = torch.argmax(X_hat, dim=1)
#     predict_binary_matrix=torch.zeros_like(X_hat)
#     for i  in range(len(output_predict)):
#         predict_binary_matrix[i,output_predict[i]]=1
#     island_inc = torch.abs(torch.mm(predict_binary_matrix.t(), bus_inc))
#     sum_inc=torch.sum(island_inc)

#     return sum_inc,island_inc

########################### calculate total imbalance ###########################
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
##################################################################################

def euclidean_distances(x, all_vec):
    dist=np.zeros(len(all_vec))
    for i in range(len(all_vec)):
        y=all_vec[i]
        dist[i]=np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
    return dist


########################### load grid model #####################################

# get AIDB token ##################################################
# conn = http.client.HTTPSConnection("api.prod.gridpilot.tech", 8085)
# payload = 'username=0INF.UC1&password=0INFUC1&grant_type=password'
# headers = {
#   'Content-Type': 'application/x-www-form-urlencoded',
#   'Authorization': 'Basic Z3JpZHBpbG90Oi1xMy1zRmtud0o='
# }
# conn.request("POST", "/oauth/token", payload, headers)
# res = conn.getresponse()
# data = res.read()
# print(data.decode("utf-8"))
# aidb_token = data.decode("utf-8")

# input("Token received. Press enter to continue...")
###################################################################

# get AIDB asset ##################################################
# conn = http.client.HTTPSConnection("api.prod.gridpilot.tech", 8085)
# payload = ''
# headers = {
#   'Content-Type': 'application/json',
#   'Authorization': aidb_token
# }
# conn.request("GET", "/assetInventory/search?installations=NO&elements=YES&downstream=YES&topology=YES", payload, headers)
# res = conn.getresponse()
# data = res.read()
# print(data.decode("utf-8"))

# input("Grid received. Press enter to continue...")
###################################################################

# manual grid model
net = pp.from_json('gridmodel-alkyonis-2.json')

# net=pp.networks.case_ieee30()
# net=pp.networks.case5()

pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
# plt.savefig('islanding/iim_mlst/static/grid_initial/grid.png')
plt.savefig('grid.png')

print('#######################################')
print('#######################################')
print(net)
print('#######################################')
print('#######################################')

print('#######################################')
print('#######################################')
# print(pp.rundcpp(net))
print('#######################################')
print('#######################################')
##################################################################################


######################################################
######################################################
################## FOR GAP METHOD ####################
######################################################
######################################################


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
num_epoch = epochs

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


pp.runpp(net)
case_num=len(net.res_bus)
save_fig_path='static/case_'+str(case_num)
# if not os.path.exists(save_fig_path):
#     os.mkdir(save_fig_path)
gen=net.gen

k = kappa  

index_store=np.zeros(len(gen))
for i in range(len(gen)):
    inndex_sb=net.gen.bus.values[i]
    a=net.bus.loc[inndex_sb].values
    a_index,a_succ=search_1d(a,net.bus.values)
    index_store[i]=a_index
np.random.shuffle(index_store)
gen_index=torch.tensor(np.int64(index_store))

gen = net.ext_grid
index_store=np.zeros(len(gen))
for i in range(len(gen)):
    inndex_sb=net.ext_grid.bus.values[i]
    a=net.bus.loc[inndex_sb].values
    a_index,a_succ=search_1d(a,net.bus.values)
    index_store[i]=a_index
extra_grid_index=torch.tensor(np.int64(index_store))

########################### extract information from pandapower lib #####################################
# for graph representation purpose ######################################################################
res_transfrom=net.res_trafo.values
h_index_trans=net.trafo.hv_bus.values
l_index_trans=net.trafo.lv_bus.values
nodes=net.res_bus.values
bus_inc=nodes[:,2]
lines=net.line.values
res_lines=net.res_line.values

min_max_scaler = preprocessing.MinMaxScaler()
nodes = min_max_scaler.fit_transform(nodes)

S = cosine_similarity(nodes, nodes)

########################### construct the adjacent matrix from lines and transformer #####################################
adjacent_matrix=np.zeros_like(S)
from_bus_index=np.where(net.line.keys()=='from_bus')[0]
to_bus_index=np.where(net.line.keys()=='to_bus')[0]

for i in range(len(lines)):
    from_num=lines[i][from_bus_index]

    to_num=lines[i][to_bus_index]
    a=net.bus.loc[from_num].values
    b=net.bus.loc[to_num].values
    a_index,a_succ=search_1d(a,net.bus.values)
    b_index, b_succ = search_1d(b, net.bus.values)
    if a_succ and b_succ:
        absolut_value=np.sqrt(res_lines[i][0]**2+res_lines[i][1]**2)+ 1e-7
        adjacent_matrix[b_index,a_index]=absolut_value
        adjacent_matrix[a_index, b_index] = absolut_value
    else:
        print('search fail')
        print('current item is '+str(i))
        sys,exit()
    c=1

for i in range(len(adjacent_matrix)):
    adjacent_matrix[i, i] = 1

for i in range(len(h_index_trans)):
    from_num=h_index_trans[i]
    to_num=l_index_trans[i]
    a = net.bus.loc[from_num].values
    b = net.bus.loc[to_num].values
    a_index, a_succ = search_1d(a, net.bus.values)
    b_index, b_succ = search_1d(b, net.bus.values)
    if a_succ and b_succ:
        absolut_value = np.abs((net.res_trafo.values[i][0] + net.res_trafo.values[i][2]) / 2)+ 1e-7
        adjacent_matrix[b_index, a_index] = absolut_value
        adjacent_matrix[a_index, b_index] = absolut_value


########################### convert to Laplcian matrix and degree matrix #####################################
indices_ad=np.where(adjacent_matrix>0)
tor_adjacent_matrix = torch.from_numpy(adjacent_matrix).to(device)#.cuda()
new_adj=torch.zeros_like(tor_adjacent_matrix)
new_adj[indices_ad]=1
for i in range(len(new_adj)):
    new_adj[i,i]=0
adjacent_matrix=new_adj.cpu().numpy()
csc_adjacent_matrix=sparse.csr_matrix(adjacent_matrix)
La=utilities.laplacian(csc_adjacent_matrix, normalized=True)
La=utilities.csc2sparsetensor(La).to(device)
spars_Adj=utilities.csc2sparsetensor(csc_adjacent_matrix).to(device)
D = np.diag(1.0 / np.sqrt(adjacent_matrix.sum(axis=1)))
X_train=torch.tensor(nodes).float().to(device)
layers = [len(X_train)] + args.layers + [k]


########################### initialize GCN model #####################################
import gcn_model
model = gcn_model.GraphEncoder(layers, k,La,len(X_train[0]),device=device,cluster_num=k).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0)
with tqdm(total=num_epoch) as tq:
    for epoch in range(1, num_epoch + 1):
        optimizer.zero_grad()
        X_hat = model(X_train)

        loss,Loss_cut,loss_balance,loss_bus_inc,loss_gen=model.gap_loss(X_hat,adjacent_matrix,bus_inc,gen_index=gen_index)
        loss.backward()
        optimizer.step()
        tq.set_postfix(loss='{:.3f}'.format(loss), nmi='{:.3f}'.format(1), rank_loss='{:.4f}'.format(1))
        tq.update()

output_predict=torch.argmax(X_hat,dim=1)
output_predict=output_predict.cpu().numpy()


########################### evaluation #####################################

# Initiate here variables to hold the values that will be stored in db #####
island_imbalance_initial = []
total_imbalance_initial = 0
island_imbalance_after = []
total_imbalance_after = 0
lines = 0
############################################################################

sum_inc,island_inc=evaluation_mini_imblance(X_hat,output_predict,bus_inc)
for i in range(len(island_inc)):
    print('island imbalance ' +str(i)+': '+str(island_inc[i]))
    island_imbalance_initial.append( island_inc[i].numpy()[0] )
print('total island imbalance= '+str(sum_inc) )

total_imbalance_initial = sum_inc


########################### cut lines #####################################
line_to_cut=[]
for i in range(len(adjacent_matrix)):
    current_index=i
    current_line=adjacent_matrix[i]
    index_adj=np.where(current_line>0)
    for neibhor_vertex in index_adj[0]:
        if output_predict[current_index]==output_predict[neibhor_vertex]:
            continue
        else:
            line_to_cut+=[[current_index,neibhor_vertex]]
number_lines=0
lines_to_cut=[]

for single_line_tocut in line_to_cut:
    tem_bus1=net.bus.index[single_line_tocut[0]]
    line_num1 = np.where(net.line['from_bus'] == tem_bus1)
    tem_bus2 = net.bus.index[single_line_tocut[1]]
    line_num2 = np.where(net.line['to_bus'] == tem_bus2)
    line_num =list(set(line_num1[0]) & set(line_num2[0]))
    line_num = np.array(line_num)

    if len(np.array(line_num))==0:
        continue
    else:
        if len(line_num)>1:
            for single_line_num in line_num:
                line_num = net.line.index[single_line_num]
                lines_to_cut += [line_num]
                number_lines += 1
        else:
            line_num = net.line.index[line_num[0]]
            lines_to_cut+=[line_num]
            number_lines += 1

pp.drop_lines(net, lines_to_cut)

########################### drop transformers #####################################
lines_to_cut=[]
for single_line_tocut in line_to_cut:

    tem_bus1=net.bus.index[single_line_tocut[0]]
    line_num1 = np.where(net.trafo['hv_bus'] == tem_bus1)
    tem_bus2 = net.bus.index[single_line_tocut[1]]
    line_num2 = np.where(net.trafo['lv_bus'] == tem_bus2)
    line_num =list(set(line_num1[0]) & set(line_num2[0]))
    line_num = np.array(line_num)

    if len(np.array(line_num))==0:
        continue
    else:
        line_num = net.trafo.index[line_num[0]]
        lines_to_cut+=[line_num]
        number_lines += 1

pp.drop_trafos(net, lines_to_cut)

sum_inc,island_inc=evaluation_mini_imblance(X_hat,output_predict,bus_inc)

for i in range(len(island_inc)):
    print('island imbalance ' +str(i)+': '+str(island_inc[i]))
    island_imbalance_after.append( island_inc[i].numpy()[0] )
print('total island imbalance= '+str(sum_inc) )
print('number of lines to cut= '+str(number_lines))

total_imbalance_after = sum_inc
lines = number_lines

########################### run pp to test result #####################################

pp.runpp(net)

jnet=pp.to_json(net, filename=None, encryption_key=None)
gap_net=pp.to_json(net, filename='gap_net.json', encryption_key=None)
method='GAP'
# IslandingScheme.objects.all().delete()
IslandingScheme.objects.create( method_name=method, island_imbalance=island_imbalance_initial, total_imbalance=total_imbalance_initial, island_imbalance_after_cut=island_imbalance_after, total_imbalance_after_cut=total_imbalance_after, lines_to_cut=lines, grid=jnet )

pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
plt.savefig('islanding/iim_mlst/static/grid_after_islanding/grid_after_'+method+'.png')

pp.plotting.to_html(net, filename='islanding/iim_mlst/static/grid_after_islanding/interactive-plot_'+method+'.html', show_tables=False)
# simple_plotly_gen(net, file_name='islanding/iim_mlst/static/grid_after_islanding/interactive-plot2_'+method+'.html')

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


# ######################################################
# ######################################################
# ################## FOR MINCUT METHOD #################
# ######################################################
# ######################################################


# ########################### load grid model #####################################
# replace with AIDB connection ##################################################

# net=pp.networks.case_ieee30()
net = pp.from_json('gridmodel-alkyonis-2.json')

print('#######################################')
print('#######################################')
print(net)
print('#######################################')
print('#######################################')

print('#######################################')
print('#######################################')
print(pp.rundcpp(net))
print('#######################################')
print('#######################################')
##################################################################################

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
num_epoch = epochs

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


pp.runpp(net)
case_num=len(net.res_bus)
save_fig_path='static/case_'+str(case_num)
# if not os.path.exists(save_fig_path):
#     os.mkdir(save_fig_path)
gen=net.gen

k = kappa  

index_store=np.zeros(len(gen))
for i in range(len(gen)):
    inndex_sb=net.gen.bus.values[i]
    a=net.bus.loc[inndex_sb].values
    a_index,a_succ=search_1d(a,net.bus.values)
    index_store[i]=a_index
np.random.shuffle(index_store)
gen_index=torch.tensor(np.int64(index_store))

gen = net.ext_grid
index_store=np.zeros(len(gen))
for i in range(len(gen)):
    inndex_sb=net.ext_grid.bus.values[i]
    a=net.bus.loc[inndex_sb].values
    a_index,a_succ=search_1d(a,net.bus.values)
    index_store[i]=a_index
extra_grid_index=torch.tensor(np.int64(index_store))

########################### extract information from pandapower lib #####################################
# for graph representation purpose ######################################################################
res_transfrom=net.res_trafo.values
h_index_trans=net.trafo.hv_bus.values
l_index_trans=net.trafo.lv_bus.values
nodes=net.res_bus.values
bus_inc=nodes[:,2]
lines=net.line.values
res_lines=net.res_line.values

min_max_scaler = preprocessing.MinMaxScaler()
nodes = min_max_scaler.fit_transform(nodes)

S = cosine_similarity(nodes, nodes)

########################### construct the adjacent matrix from lines and transformer #####################################
adjacent_matrix=np.zeros_like(S)
from_bus_index=np.where(net.line.keys()=='from_bus')[0]
to_bus_index=np.where(net.line.keys()=='to_bus')[0]

for i in range(len(lines)):
    from_num=lines[i][from_bus_index]

    to_num=lines[i][to_bus_index]
    a=net.bus.loc[from_num].values
    b=net.bus.loc[to_num].values
    a_index,a_succ=search_1d(a,net.bus.values)
    b_index, b_succ = search_1d(b, net.bus.values)
    if a_succ and b_succ:
        absolut_value=np.sqrt(res_lines[i][0]**2+res_lines[i][1]**2)+ 1e-7
        adjacent_matrix[b_index,a_index]=absolut_value
        adjacent_matrix[a_index, b_index] = absolut_value
    else:
        print('search fail')
        print('current item is '+str(i))
        sys,exit()
    c=1

for i in range(len(adjacent_matrix)):
    adjacent_matrix[i, i] = 1

for i in range(len(h_index_trans)):
    from_num=h_index_trans[i]
    to_num=l_index_trans[i]
    a = net.bus.loc[from_num].values
    b = net.bus.loc[to_num].values
    a_index, a_succ = search_1d(a, net.bus.values)
    b_index, b_succ = search_1d(b, net.bus.values)
    if a_succ and b_succ:
        absolut_value = np.abs((net.res_trafo.values[i][0] + net.res_trafo.values[i][2]) / 2)+ 1e-7
        adjacent_matrix[b_index, a_index] = absolut_value
        adjacent_matrix[a_index, b_index] = absolut_value


########################### convert to Laplcian matrix and degree matrix #####################################
indices_ad=np.where(adjacent_matrix>0)
tor_adjacent_matrix = torch.from_numpy(adjacent_matrix).to(device)#.cuda()
new_adj=torch.zeros_like(tor_adjacent_matrix)
new_adj[indices_ad]=1
for i in range(len(new_adj)):
    new_adj[i,i]=0
adjacent_matrix=new_adj.cpu().numpy()
csc_adjacent_matrix=sparse.csr_matrix(adjacent_matrix)
La=utilities.laplacian(csc_adjacent_matrix, normalized=True)
La=utilities.csc2sparsetensor(La).to(device)
spars_Adj=utilities.csc2sparsetensor(csc_adjacent_matrix).to(device)
D = np.diag(1.0 / np.sqrt(adjacent_matrix.sum(axis=1)))
X_train=torch.tensor(nodes).float().to(device)
layers = [len(X_train)] + args.layers + [k]


########################### initialize GCN model #####################################
import gcn_model
model = gcn_model.GraphEncoder(layers, k,La,len(X_train[0]),device=device,cluster_num=k).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0)
with tqdm(total=num_epoch) as tq:
    for epoch in range(1, num_epoch + 1):
        optimizer.zero_grad()
        X_hat = model(X_train)

        loss = model.min_cut_loss(adjacent_matrix,X_hat,bus_inc,gen_index=gen_index)

        loss.backward()
        optimizer.step()
        tq.set_postfix(loss='{:.3f}'.format(loss), nmi='{:.3f}'.format(1), rank_loss='{:.4f}'.format(1))
        tq.update()

output_predict=torch.argmax(X_hat,dim=1)
output_predict=output_predict.cpu().numpy()


########################### evaluation #####################################

# Initiate here variables to hold the values that will be stored in db #####
island_imbalance_initial = []
total_imbalance_initial = 0
island_imbalance_after = []
total_imbalance_after = 0
lines = 0
############################################################################

sum_inc,island_inc=evaluation_mini_imblance(X_hat,output_predict,bus_inc)
for i in range(len(island_inc)):
    print('island imbalance ' +str(i)+': '+str(island_inc[i]))
    island_imbalance_initial.append( island_inc[i].numpy()[0] )
print('total island imbalance= '+str(sum_inc) )

total_imbalance_initial = sum_inc


########################### cut lines #####################################
line_to_cut=[]
for i in range(len(adjacent_matrix)):
    current_index=i
    current_line=adjacent_matrix[i]
    index_adj=np.where(current_line>0)
    for neibhor_vertex in index_adj[0]:
        if output_predict[current_index]==output_predict[neibhor_vertex]:
            continue
        else:
            line_to_cut+=[[current_index,neibhor_vertex]]
number_lines=0
lines_to_cut=[]

for single_line_tocut in line_to_cut:
    tem_bus1=net.bus.index[single_line_tocut[0]]
    line_num1 = np.where(net.line['from_bus'] == tem_bus1)
    tem_bus2 = net.bus.index[single_line_tocut[1]]
    line_num2 = np.where(net.line['to_bus'] == tem_bus2)
    line_num =list(set(line_num1[0]) & set(line_num2[0]))
    line_num = np.array(line_num)

    if len(np.array(line_num))==0:
        continue
    else:
        if len(line_num)>1:
            for single_line_num in line_num:
                line_num = net.line.index[single_line_num]
                lines_to_cut += [line_num]
                number_lines += 1
        else:
            line_num = net.line.index[line_num[0]]
            lines_to_cut+=[line_num]
            number_lines += 1

pp.drop_lines(net, lines_to_cut)

########################### drop transformers #####################################
lines_to_cut=[]
for single_line_tocut in line_to_cut:

    tem_bus1=net.bus.index[single_line_tocut[0]]
    line_num1 = np.where(net.trafo['hv_bus'] == tem_bus1)
    tem_bus2 = net.bus.index[single_line_tocut[1]]
    line_num2 = np.where(net.trafo['lv_bus'] == tem_bus2)
    line_num =list(set(line_num1[0]) & set(line_num2[0]))
    line_num = np.array(line_num)

    if len(np.array(line_num))==0:
        continue
    else:
        line_num = net.trafo.index[line_num[0]]
        lines_to_cut+=[line_num]
        number_lines += 1

pp.drop_trafos(net, lines_to_cut)

sum_inc,island_inc=evaluation_mini_imblance(X_hat,output_predict,bus_inc)

for i in range(len(island_inc)):
    print('island imbalance ' +str(i)+': '+str(island_inc[i]))
    island_imbalance_after.append( island_inc[i].numpy()[0] )
print('total island imbalance= '+str(sum_inc) )
print('number of lines to cut= '+str(number_lines))

total_imbalance_after = sum_inc
lines = number_lines

########################### run pp to test result #####################################

pp.runpp(net)

jnet=pp.to_json(net, filename=None, encryption_key=None)
mc_net=pp.to_json(net, filename='mc_net.json', encryption_key=None)
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

# #################### END MINCUT METHOD ####################

########################### create result json ########################################

# print(jnet)
# nnet  = pp.networks.case5()
# net  = pp.networks.case4gs()
# jnnet = pp.to_json(nnet, filename=None, encryption_key=None)
# snet = pp.to_json(net, filename='case4.json', encryption_key=None)

# print('####################################### gapnnet')
# print(jnet)
# print('#######################################')
# print('#######################################')
# print('#######################################')

# knnet = json.dumps(jnnet, separators=(',', ":"))

# print('####################################### knnet')
# print(knnet)
# print('#######################################')
# print('#######################################')
# print('#######################################')


# rmq_json['messageId'] = str(uuid.uuid4())
rmq_json['messageId'] = 'integration_test_iim_mlst_real_9_Dec'
# rmq_json['payload']   = jnet # messes up the format, like twice dumps does
# rmq_json['payload']['timestamp'] = str(datetime.datetime.now())
rmq_json['utcTimestamp'] = str(datetime.datetime.now(timezone.utc))

# input("Press enter to load minified json...")

with open('jsonminifier.json') as f:
  json_data = json.load(f)

rmq_json['payload']   = json_data # messes up the format, like twice dumps does
# input("Press enter to print rmqjson...")
# print(rmq_json)

final_json = json.dumps(rmq_json)
# final_json = json.dumps(json_data)
print(final_json)

# saved_rmqjson = pp.to_json(rmq_json, filename='rmq_json.json', encryption_key=None)
# print(rmq_json)
input("Press enter to send json to rabbitmq...")

# with open('data1.json', 'w', encoding='utf-8') as f:
#     json.dump(jnet, f, ensure_ascii=False, indent=4)
    
# with open('data2.json', 'w', encoding='utf-8') as f:
#     json.dump(rmq_json, f, ensure_ascii=False, indent=4)    
       
########################## test CLOUDAMQP_URL rabbit mq ################################

# print(" [x] Trying rabbitmq")
# url = os.environ.get('CLOUDAMQP_URL', 'amqps://wkpfxuzn:JiZuXbbWZQpBQY8FjTihmTjefnHQrnKb@rattlesnake.rmq.cloudamqp.com/wkpfxuzn')
# params = pika.URLParameters(url)
# connection = pika.BlockingConnection(params)
# channel = connection.channel()
# channel.queue_declare(queue='mlst_iim')
# # channel.basic_publish(exchange='', routing_key='mlst_iim', body=json.dumps(json_data))
# channel.basic_publish(exchange='', routing_key='mlst_iim', body=final_json, properties=pika.BasicProperties(delivery_mode = 2,))

# print(" [x] Sent test")
# connection.close()
#######################################################################################

########################## connect to rabbitmq AIDB gridpilot #########################

# credentials = pika.PlainCredentials('iim-guest', 'iimguest')
# # parameters = pika.ConnectionParameters('3.120.35.154', 5672, 'iim', credentials)
# parameters = pika.ConnectionParameters('rabbit.prod.gridpilot.tech', 5672, 'iim', credentials)
# connection = pika.BlockingConnection(parameters)

# channel = connection.channel()
# channel.queue_declare(queue='IIM#IIM')

# channel.basic_publish(exchange='Islanding_Exchange.headers', routing_key='IIM#IIM', body=final_json, properties=pika.BasicProperties(delivery_mode = 2,))

# print(" [x] Sent 'Islanding scheme!'")
# connection.close()
#######################################################################################

print('End.')