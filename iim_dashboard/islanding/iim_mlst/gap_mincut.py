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
epochs  = 100
#########################################################

#################### number of clusters #################
kappa   = 3 
#########################################################


#################### rmq json format ####################
rmq_json = {
    "messageId": "",
    "name": "islanding_result",
    "payload": {
        "data": "",
        "timestamp": ""
    },
    "payloadType": "Payload",
    "source": "PublisherWithPipeline",
    "uc": "UC1",
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
# net=pp.networks.case_ieee30()
net=pp.networks.case118()

pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
plt.savefig('islanding/iim_mlst/static/grid_initial/grid.png')

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


######################################################
######################################################
################## FOR MINCUT METHOD #################
######################################################
######################################################


########################### load grid model #####################################
# replace with AIDB connection ##################################################

net=pp.networks.case_ieee30()

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
method='MIN-CUT'
# IslandingScheme.objects.all().delete()
IslandingScheme.objects.create( method_name=method, island_imbalance=island_imbalance_initial, total_imbalance=total_imbalance_initial, island_imbalance_after_cut=island_imbalance_after, total_imbalance_after_cut=total_imbalance_after, lines_to_cut=lines, grid=jnet )

pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
plt.savefig('islanding/iim_mlst/static/grid_after_islanding/grid_after_'+method+'.png')

pp.plotting.to_html(net, filename='islanding/iim_mlst/static/grid_after_islanding/interactive-plot_'+method+'.html', show_tables=False)
simple_plotly_gen(net, file_name='islanding/iim_mlst/static/grid_after_islanding/interactive-plot2_'+method+'.html')

########################### create result json ########################################

# print(jnet)
net  = pp.networks.case_ieee30()
jnet = pp.to_json(net, filename=None, encryption_key=None)

rrmq_json = {
    "messageId": "e8793bb7-a4ea-4a5b-99b2-dc9caac24720",
    "name": "islanding_result",
    "payload": {
        "_module": "pandapower.auxiliary",
        "_class": "pandapowerNet",
        "_object": {
            "bus": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"in_service\",\"max_vm_pu\",\"min_vm_pu\",\"name\",\"type\",\"vn_kv\",\"zone\"],\"index\":[0,1,10,100,101,102,103,104,105,106,107,108,109,11,110,111,112,113,114,115,116,117,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,42,43,44,45,46,47,48,49,5,50,51,52,53,54,55,56,57,58,59,6,60,61,62,63,64,65,66,67,68,69,7,70,71,72,73,74,75,76,77,78,79,8,80,81,82,83,84,85,86,87,88,89,9,90,91,92,93,94,95,96,97,98,99],\"data\":[[true,1.06,0.94,1,\"b\",138.0,1.0],[true,1.06,0.94,2,\"b\",138.0,1.0],[true,1.06,0.94,11,\"b\",138.0,1.0],[true,1.06,0.94,101,\"b\",138.0,1.0],[true,1.06,0.94,102,\"b\",138.0,1.0],[true,1.06,0.94,103,\"b\",138.0,1.0],[true,1.06,0.94,104,\"b\",138.0,1.0],[true,1.06,0.94,105,\"b\",138.0,1.0],[true,1.06,0.94,106,\"b\",138.0,1.0],[true,1.06,0.94,107,\"b\",138.0,1.0],[true,1.06,0.94,108,\"b\",138.0,1.0],[true,1.06,0.94,109,\"b\",138.0,1.0],[true,1.06,0.94,110,\"b\",138.0,1.0],[true,1.06,0.94,12,\"b\",138.0,1.0],[true,1.06,0.94,111,\"b\",138.0,1.0],[true,1.06,0.94,112,\"b\",138.0,1.0],[true,1.06,0.94,113,\"b\",138.0,1.0],[true,1.06,0.94,114,\"b\",138.0,1.0],[true,1.06,0.94,115,\"b\",138.0,1.0],[true,1.06,0.94,116,\"b\",345.0,1.0],[true,1.06,0.94,117,\"b\",138.0,1.0],[true,1.06,0.94,118,\"b\",138.0,1.0],[true,1.06,0.94,13,\"b\",138.0,1.0],[true,1.06,0.94,14,\"b\",138.0,1.0],[true,1.06,0.94,15,\"b\",138.0,1.0],[true,1.06,0.94,16,\"b\",138.0,1.0],[true,1.06,0.94,17,\"b\",138.0,1.0],[true,1.06,0.94,18,\"b\",138.0,1.0],[true,1.06,0.94,19,\"b\",138.0,1.0],[true,1.06,0.94,20,\"b\",138.0,1.0],[true,1.06,0.94,3,\"b\",138.0,1.0],[true,1.06,0.94,21,\"b\",138.0,1.0],[true,1.06,0.94,22,\"b\",138.0,1.0],[true,1.06,0.94,23,\"b\",138.0,1.0],[true,1.06,0.94,24,\"b\",138.0,1.0],[true,1.06,0.94,25,\"b\",138.0,1.0],[true,1.06,0.94,26,\"b\",345.0,1.0],[true,1.06,0.94,27,\"b\",138.0,1.0],[true,1.06,0.94,28,\"b\",138.0,1.0],[true,1.06,0.94,29,\"b\",138.0,1.0],[true,1.06,0.94,30,\"b\",345.0,1.0],[true,1.06,0.94,4,\"b\",138.0,1.0],[true,1.06,0.94,31,\"b\",138.0,1.0],[true,1.06,0.94,32,\"b\",138.0,1.0],[true,1.06,0.94,33,\"b\",138.0,1.0],[true,1.06,0.94,34,\"b\",138.0,1.0],[true,1.06,0.94,35,\"b\",138.0,1.0],[true,1.06,0.94,36,\"b\",138.0,1.0],[true,1.06,0.94,37,\"b\",138.0,1.0],[true,1.06,0.94,38,\"b\",345.0,1.0],[true,1.06,0.94,39,\"b\",138.0,1.0],[true,1.06,0.94,40,\"b\",138.0,1.0],[true,1.06,0.94,5,\"b\",138.0,1.0],[true,1.06,0.94,41,\"b\",138.0,1.0],[true,1.06,0.94,42,\"b\",138.0,1.0],[true,1.06,0.94,43,\"b\",138.0,1.0],[true,1.06,0.94,44,\"b\",138.0,1.0],[true,1.06,0.94,45,\"b\",138.0,1.0],[true,1.06,0.94,46,\"b\",138.0,1.0],[true,1.06,0.94,47,\"b\",138.0,1.0],[true,1.06,0.94,48,\"b\",138.0,1.0],[true,1.06,0.94,49,\"b\",138.0,1.0],[true,1.06,0.94,50,\"b\",138.0,1.0],[true,1.06,0.94,6,\"b\",138.0,1.0],[true,1.06,0.94,51,\"b\",138.0,1.0],[true,1.06,0.94,52,\"b\",138.0,1.0],[true,1.06,0.94,53,\"b\",138.0,1.0],[true,1.06,0.94,54,\"b\",138.0,1.0],[true,1.06,0.94,55,\"b\",138.0,1.0],[true,1.06,0.94,56,\"b\",138.0,1.0],[true,1.06,0.94,57,\"b\",138.0,1.0],[true,1.06,0.94,58,\"b\",138.0,1.0],[true,1.06,0.94,59,\"b\",138.0,1.0],[true,1.06,0.94,60,\"b\",138.0,1.0],[true,1.06,0.94,7,\"b\",138.0,1.0],[true,1.06,0.94,61,\"b\",138.0,1.0],[true,1.06,0.94,62,\"b\",138.0,1.0],[true,1.06,0.94,63,\"b\",345.0,1.0],[true,1.06,0.94,64,\"b\",345.0,1.0],[true,1.06,0.94,65,\"b\",345.0,1.0],[true,1.06,0.94,66,\"b\",138.0,1.0],[true,1.06,0.94,67,\"b\",138.0,1.0],[true,1.06,0.94,68,\"b\",161.0,1.0],[true,1.06,0.94,69,\"b\",138.0,1.0],[true,1.06,0.94,70,\"b\",138.0,1.0],[true,1.06,0.94,8,\"b\",345.0,1.0],[true,1.06,0.94,71,\"b\",138.0,1.0],[true,1.06,0.94,72,\"b\",138.0,1.0],[true,1.06,0.94,73,\"b\",138.0,1.0],[true,1.06,0.94,74,\"b\",138.0,1.0],[true,1.06,0.94,75,\"b\",138.0,1.0],[true,1.06,0.94,76,\"b\",138.0,1.0],[true,1.06,0.94,77,\"b\",138.0,1.0],[true,1.06,0.94,78,\"b\",138.0,1.0],[true,1.06,0.94,79,\"b\",138.0,1.0],[true,1.06,0.94,80,\"b\",138.0,1.0],[true,1.06,0.94,9,\"b\",345.0,1.0],[true,1.06,0.94,81,\"b\",345.0,1.0],[true,1.06,0.94,82,\"b\",138.0,1.0],[true,1.06,0.94,83,\"b\",138.0,1.0],[true,1.06,0.94,84,\"b\",138.0,1.0],[true,1.06,0.94,85,\"b\",138.0,1.0],[true,1.06,0.94,86,\"b\",138.0,1.0],[true,1.06,0.94,87,\"b\",161.0,1.0],[true,1.06,0.94,88,\"b\",138.0,1.0],[true,1.06,0.94,89,\"b\",138.0,1.0],[true,1.06,0.94,90,\"b\",138.0,1.0],[true,1.06,0.94,10,\"b\",345.0,1.0],[true,1.06,0.94,91,\"b\",138.0,1.0],[true,1.06,0.94,92,\"b\",138.0,1.0],[true,1.06,0.94,93,\"b\",138.0,1.0],[true,1.06,0.94,94,\"b\",138.0,1.0],[true,1.06,0.94,95,\"b\",138.0,1.0],[true,1.06,0.94,96,\"b\",138.0,1.0],[true,1.06,0.94,97,\"b\",138.0,1.0],[true,1.06,0.94,98,\"b\",138.0,1.0],[true,1.06,0.94,99,\"b\",138.0,1.0],[true,1.06,0.94,100,\"b\",138.0,1.0]]}",
                "orient": "split",
                "dtype": {
                    "in_service": "bool",
                    "max_vm_pu": "float64",
                    "min_vm_pu": "float64",
                    "name": "object",
                    "type": "object",
                    "vn_kv": "float64",
                    "zone": "object"
                }
            },
            "load": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"bus\",\"const_i_percent\",\"const_z_percent\",\"controllable\",\"in_service\",\"name\",\"p_mw\",\"q_mvar\",\"scaling\",\"sn_mva\",\"type\"],\"index\":[0,1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,42,43,44,45,46,47,48,49,5,50,51,52,53,54,55,56,57,58,59,6,60,61,62,63,64,65,66,67,68,69,7,70,71,72,73,74,75,76,77,78,79,8,80,81,82,83,84,85,86,87,88,89,9,90,91,92,93,94,95,96,97,98],\"data\":[[0,0.0,0.0,false,true,null,51.0,27.0,1.0,null,null],[1,0.0,0.0,false,true,null,20.0,9.0,1.0,null,null],[13,0.0,0.0,false,true,null,14.0,1.0,1.0,null,null],[14,0.0,0.0,false,true,null,90.0,30.0,1.0,null,null],[15,0.0,0.0,false,true,null,25.0,10.0,1.0,null,null],[16,0.0,0.0,false,true,null,11.0,3.0,1.0,null,null],[17,0.0,0.0,false,true,null,60.0,34.0,1.0,null,null],[18,0.0,0.0,false,true,null,45.0,25.0,1.0,null,null],[19,0.0,0.0,false,true,null,18.0,3.0,1.0,null,null],[20,0.0,0.0,false,true,null,14.0,8.0,1.0,null,null],[21,0.0,0.0,false,true,null,10.0,5.0,1.0,null,null],[22,0.0,0.0,false,true,null,7.0,3.0,1.0,null,null],[2,0.0,0.0,false,true,null,39.0,10.0,1.0,null,null],[23,0.0,0.0,false,true,null,13.0,0.0,1.0,null,null],[26,0.0,0.0,false,true,null,71.0,13.0,1.0,null,null],[27,0.0,0.0,false,true,null,17.0,7.0,1.0,null,null],[28,0.0,0.0,false,true,null,24.0,4.0,1.0,null,null],[30,0.0,0.0,false,true,null,43.0,27.0,1.0,null,null],[31,0.0,0.0,false,true,null,59.0,23.0,1.0,null,null],[32,0.0,0.0,false,true,null,23.0,9.0,1.0,null,null],[33,0.0,0.0,false,true,null,59.0,26.0,1.0,null,null],[34,0.0,0.0,false,true,null,33.0,9.0,1.0,null,null],[35,0.0,0.0,false,true,null,31.0,17.0,1.0,null,null],[3,0.0,0.0,false,true,null,39.0,12.0,1.0,null,null],[38,0.0,0.0,false,true,null,27.0,11.0,1.0,null,null],[39,0.0,0.0,false,true,null,66.0,23.0,1.0,null,null],[40,0.0,0.0,false,true,null,37.0,10.0,1.0,null,null],[41,0.0,0.0,false,true,null,96.0,23.0,1.0,null,null],[42,0.0,0.0,false,true,null,18.0,7.0,1.0,null,null],[43,0.0,0.0,false,true,null,16.0,8.0,1.0,null,null],[44,0.0,0.0,false,true,null,53.0,22.0,1.0,null,null],[45,0.0,0.0,false,true,null,28.0,10.0,1.0,null,null],[46,0.0,0.0,false,true,null,34.0,0.0,1.0,null,null],[47,0.0,0.0,false,true,null,20.0,11.0,1.0,null,null],[5,0.0,0.0,false,true,null,52.0,22.0,1.0,null,null],[48,0.0,0.0,false,true,null,87.0,30.0,1.0,null,null],[49,0.0,0.0,false,true,null,17.0,4.0,1.0,null,null],[50,0.0,0.0,false,true,null,17.0,8.0,1.0,null,null],[51,0.0,0.0,false,true,null,18.0,5.0,1.0,null,null],[52,0.0,0.0,false,true,null,23.0,11.0,1.0,null,null],[53,0.0,0.0,false,true,null,113.0,32.0,1.0,null,null],[54,0.0,0.0,false,true,null,63.0,22.0,1.0,null,null],[55,0.0,0.0,false,true,null,84.0,18.0,1.0,null,null],[56,0.0,0.0,false,true,null,12.0,3.0,1.0,null,null],[57,0.0,0.0,false,true,null,12.0,3.0,1.0,null,null],[6,0.0,0.0,false,true,null,19.0,2.0,1.0,null,null],[58,0.0,0.0,false,true,null,277.0,113.0,1.0,null,null],[59,0.0,0.0,false,true,null,78.0,3.0,1.0,null,null],[61,0.0,0.0,false,true,null,77.0,14.0,1.0,null,null],[65,0.0,0.0,false,true,null,39.0,18.0,1.0,null,null],[66,0.0,0.0,false,true,null,28.0,7.0,1.0,null,null],[69,0.0,0.0,false,true,null,66.0,20.0,1.0,null,null],[71,0.0,0.0,false,true,null,12.0,0.0,1.0,null,null],[72,0.0,0.0,false,true,null,6.0,0.0,1.0,null,null],[73,0.0,0.0,false,true,null,68.0,27.0,1.0,null,null],[74,0.0,0.0,false,true,null,47.0,11.0,1.0,null,null],[7,0.0,0.0,false,true,null,28.0,0.0,1.0,null,null],[75,0.0,0.0,false,true,null,68.0,36.0,1.0,null,null],[76,0.0,0.0,false,true,null,61.0,28.0,1.0,null,null],[77,0.0,0.0,false,true,null,71.0,26.0,1.0,null,null],[78,0.0,0.0,false,true,null,39.0,32.0,1.0,null,null],[79,0.0,0.0,false,true,null,130.0,26.0,1.0,null,null],[81,0.0,0.0,false,true,null,54.0,27.0,1.0,null,null],[82,0.0,0.0,false,true,null,20.0,10.0,1.0,null,null],[83,0.0,0.0,false,true,null,11.0,7.0,1.0,null,null],[84,0.0,0.0,false,true,null,24.0,15.0,1.0,null,null],[85,0.0,0.0,false,true,null,21.0,10.0,1.0,null,null],[10,0.0,0.0,false,true,null,70.0,23.0,1.0,null,null],[87,0.0,0.0,false,true,null,48.0,10.0,1.0,null,null],[89,0.0,0.0,false,true,null,163.0,42.0,1.0,null,null],[90,0.0,0.0,false,true,null,10.0,0.0,1.0,null,null],[91,0.0,0.0,false,true,null,65.0,10.0,1.0,null,null],[92,0.0,0.0,false,true,null,12.0,7.0,1.0,null,null],[93,0.0,0.0,false,true,null,30.0,16.0,1.0,null,null],[94,0.0,0.0,false,true,null,42.0,31.0,1.0,null,null],[95,0.0,0.0,false,true,null,38.0,15.0,1.0,null,null],[96,0.0,0.0,false,true,null,15.0,9.0,1.0,null,null],[97,0.0,0.0,false,true,null,34.0,8.0,1.0,null,null],[11,0.0,0.0,false,true,null,47.0,10.0,1.0,null,null],[98,0.0,0.0,false,true,null,42.0,0.0,1.0,null,null],[99,0.0,0.0,false,true,null,37.0,18.0,1.0,null,null],[100,0.0,0.0,false,true,null,22.0,15.0,1.0,null,null],[101,0.0,0.0,false,true,null,5.0,3.0,1.0,null,null],[102,0.0,0.0,false,true,null,23.0,16.0,1.0,null,null],[103,0.0,0.0,false,true,null,38.0,25.0,1.0,null,null],[104,0.0,0.0,false,true,null,31.0,26.0,1.0,null,null],[105,0.0,0.0,false,true,null,43.0,16.0,1.0,null,null],[106,0.0,0.0,false,true,null,50.0,12.0,1.0,null,null],[107,0.0,0.0,false,true,null,2.0,1.0,1.0,null,null],[12,0.0,0.0,false,true,null,34.0,16.0,1.0,null,null],[108,0.0,0.0,false,true,null,8.0,3.0,1.0,null,null],[109,0.0,0.0,false,true,null,39.0,30.0,1.0,null,null],[111,0.0,0.0,false,true,null,68.0,13.0,1.0,null,null],[112,0.0,0.0,false,true,null,6.0,0.0,1.0,null,null],[113,0.0,0.0,false,true,null,8.0,3.0,1.0,null,null],[114,0.0,0.0,false,true,null,22.0,7.0,1.0,null,null],[115,0.0,0.0,false,true,null,184.0,0.0,1.0,null,null],[116,0.0,0.0,false,true,null,20.0,8.0,1.0,null,null],[117,0.0,0.0,false,true,null,33.0,15.0,1.0,null,null]]}",
                "orient": "split",
                "dtype": {
                    "bus": "uint32",
                    "const_i_percent": "float64",
                    "const_z_percent": "float64",
                    "controllable": "bool",
                    "in_service": "bool",
                    "name": "object",
                    "p_mw": "float64",
                    "q_mvar": "float64",
                    "scaling": "float64",
                    "sn_mva": "float64",
                    "type": "object"
                }
            },
            "sgen": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"p_mw\",\"q_mvar\",\"sn_mva\",\"scaling\",\"in_service\",\"type\",\"current_source\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "name": "object",
                    "bus": "int64",
                    "p_mw": "float64",
                    "q_mvar": "float64",
                    "sn_mva": "float64",
                    "scaling": "float64",
                    "in_service": "bool",
                    "type": "object",
                    "current_source": "bool"
                }
            },
            "motor": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"pn_mech_mw\",\"loading_percent\",\"cos_phi\",\"cos_phi_n\",\"efficiency_percent\",\"efficiency_n_percent\",\"lrc_pu\",\"vn_kv\",\"scaling\",\"in_service\",\"rx\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "name": "object",
                    "bus": "int64",
                    "pn_mech_mw": "float64",
                    "loading_percent": "float64",
                    "cos_phi": "float64",
                    "cos_phi_n": "float64",
                    "efficiency_percent": "float64",
                    "efficiency_n_percent": "float64",
                    "lrc_pu": "float64",
                    "vn_kv": "float64",
                    "scaling": "float64",
                    "in_service": "bool",
                    "rx": "float64"
                }
            },
            "asymmetric_load": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"p_a_mw\",\"q_a_mvar\",\"p_b_mw\",\"q_b_mvar\",\"p_c_mw\",\"q_c_mvar\",\"sn_mva\",\"scaling\",\"in_service\",\"type\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "name": "object",
                    "bus": "uint32",
                    "p_a_mw": "float64",
                    "q_a_mvar": "float64",
                    "p_b_mw": "float64",
                    "q_b_mvar": "float64",
                    "p_c_mw": "float64",
                    "q_c_mvar": "float64",
                    "sn_mva": "float64",
                    "scaling": "float64",
                    "in_service": "bool",
                    "type": "object"
                }
            },
            "asymmetric_sgen": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"p_a_mw\",\"q_a_mvar\",\"p_b_mw\",\"q_b_mvar\",\"p_c_mw\",\"q_c_mvar\",\"sn_mva\",\"scaling\",\"in_service\",\"type\",\"current_source\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "name": "object",
                    "bus": "int64",
                    "p_a_mw": "float64",
                    "q_a_mvar": "float64",
                    "p_b_mw": "float64",
                    "q_b_mvar": "float64",
                    "p_c_mw": "float64",
                    "q_c_mvar": "float64",
                    "sn_mva": "float64",
                    "scaling": "float64",
                    "in_service": "bool",
                    "type": "object",
                    "current_source": "bool"
                }
            },
            "storage": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"p_mw\",\"q_mvar\",\"sn_mva\",\"soc_percent\",\"min_e_mwh\",\"max_e_mwh\",\"scaling\",\"in_service\",\"type\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "name": "object",
                    "bus": "int64",
                    "p_mw": "float64",
                    "q_mvar": "float64",
                    "sn_mva": "float64",
                    "soc_percent": "float64",
                    "min_e_mwh": "float64",
                    "max_e_mwh": "float64",
                    "scaling": "float64",
                    "in_service": "bool",
                    "type": "object"
                }
            },
            "gen": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"bus\",\"controllable\",\"in_service\",\"name\",\"p_mw\",\"scaling\",\"sn_mva\",\"type\",\"vm_pu\",\"slack\",\"max_p_mw\",\"min_p_mw\",\"max_q_mvar\",\"min_q_mvar\"],\"index\":[0,1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,42,43,44,45,46,47,48,49,5,50,51,52,6,7,8,9],\"data\":[[0,true,true,null,0.0,1.0,null,null,0.955,false,100.0,0.0,15.0,-5.0],[3,true,true,null,0.0,1.0,null,null,0.998,false,100.0,0.0,300.0,-300.0],[24,true,true,null,220.0,1.0,null,null,1.05,false,320.0,0.0,140.0,-47.0],[25,true,true,null,314.0,1.0,null,null,1.015,false,414.0,0.0,1000.0,-1000.0],[26,true,true,null,0.0,1.0,null,null,0.968,false,100.0,0.0,300.0,-300.0],[30,true,true,null,7.0,1.0,null,null,0.967,false,107.0,0.0,300.0,-300.0],[31,true,true,null,0.0,1.0,null,null,0.963,false,100.0,0.0,42.0,-14.0],[33,true,true,null,0.0,1.0,null,null,0.984,false,100.0,0.0,24.0,-8.0],[35,true,true,null,0.0,1.0,null,null,0.98,false,100.0,0.0,24.0,-8.0],[39,true,true,null,0.0,1.0,null,null,0.97,false,100.0,0.0,300.0,-300.0],[41,true,true,null,0.0,1.0,null,null,0.985,false,100.0,0.0,300.0,-300.0],[45,true,true,null,19.0,1.0,null,null,1.005,false,119.0,0.0,100.0,-100.0],[5,true,true,null,0.0,1.0,null,null,0.99,false,100.0,0.0,50.0,-13.0],[48,true,true,null,204.0,1.0,null,null,1.025,false,304.0,0.0,210.0,-85.0],[53,true,true,null,48.0,1.0,null,null,0.955,false,148.0,0.0,300.0,-300.0],[54,true,true,null,0.0,1.0,null,null,0.952,false,100.0,0.0,23.0,-8.0],[55,true,true,null,0.0,1.0,null,null,0.954,false,100.0,0.0,15.0,-8.0],[58,true,true,null,155.0,1.0,null,null,0.985,false,255.0,0.0,180.0,-60.0],[60,true,true,null,160.0,1.0,null,null,0.995,false,260.0,0.0,300.0,-100.0],[61,true,true,null,0.0,1.0,null,null,0.998,false,100.0,0.0,20.0,-20.0],[64,true,true,null,391.0,1.0,null,null,1.005,false,491.0,0.0,200.0,-67.0],[65,true,true,null,392.0,1.0,null,null,1.05,false,492.0,0.0,200.0,-67.0],[69,true,true,null,0.0,1.0,null,null,0.984,false,100.0,0.0,32.0,-10.0],[7,true,true,null,0.0,1.0,null,null,1.015,false,100.0,0.0,300.0,-300.0],[71,true,true,null,0.0,1.0,null,null,0.98,false,100.0,0.0,100.0,-100.0],[72,true,true,null,0.0,1.0,null,null,0.991,false,100.0,0.0,100.0,-100.0],[73,true,true,null,0.0,1.0,null,null,0.958,false,100.0,0.0,9.0,-6.0],[75,true,true,null,0.0,1.0,null,null,0.943,false,100.0,0.0,23.0,-8.0],[76,true,true,null,0.0,1.0,null,null,1.006,false,100.0,0.0,70.0,-20.0],[79,true,true,null,477.0,1.0,null,null,1.04,false,577.0,0.0,280.0,-165.0],[84,true,true,null,0.0,1.0,null,null,0.985,false,100.0,0.0,23.0,-8.0],[86,true,true,null,4.0,1.0,null,null,1.015,false,104.0,0.0,1000.0,-100.0],[88,true,true,null,607.0,1.0,null,null,1.005,false,707.0,0.0,300.0,-210.0],[89,true,true,null,0.0,1.0,null,null,0.985,false,100.0,0.0,300.0,-300.0],[9,true,true,null,450.0,1.0,null,null,1.05,false,550.0,0.0,200.0,-147.0],[90,true,true,null,0.0,1.0,null,null,0.98,false,100.0,0.0,100.0,-100.0],[91,true,true,null,0.0,1.0,null,null,0.99,false,100.0,0.0,9.0,-3.0],[98,true,true,null,0.0,1.0,null,null,1.01,false,100.0,0.0,100.0,-100.0],[99,true,true,null,252.0,1.0,null,null,1.017,false,352.0,0.0,155.0,-50.0],[102,true,true,null,40.0,1.0,null,null,1.01,false,140.0,0.0,40.0,-15.0],[103,true,true,null,0.0,1.0,null,null,0.971,false,100.0,0.0,23.0,-8.0],[104,true,true,null,0.0,1.0,null,null,0.965,false,100.0,0.0,23.0,-8.0],[106,true,true,null,0.0,1.0,null,null,0.952,false,100.0,0.0,200.0,-200.0],[109,true,true,null,0.0,1.0,null,null,0.973,false,100.0,0.0,23.0,-8.0],[110,true,true,null,36.0,1.0,null,null,0.98,false,136.0,0.0,1000.0,-100.0],[11,true,true,null,85.0,1.0,null,null,0.99,false,185.0,0.0,120.0,-35.0],[111,true,true,null,0.0,1.0,null,null,0.975,false,100.0,0.0,1000.0,-100.0],[112,true,true,null,0.0,1.0,null,null,0.993,false,100.0,0.0,200.0,-100.0],[115,true,true,null,0.0,1.0,null,null,1.005,false,100.0,0.0,1000.0,-1000.0],[14,true,true,null,0.0,1.0,null,null,0.97,false,100.0,0.0,30.0,-10.0],[17,true,true,null,0.0,1.0,null,null,0.973,false,100.0,0.0,50.0,-16.0],[18,true,true,null,0.0,1.0,null,null,0.962,false,100.0,0.0,24.0,-8.0],[23,true,true,null,0.0,1.0,null,null,0.992,false,100.0,0.0,300.0,-300.0]]}",
                "orient": "split",
                "dtype": {
                    "bus": "uint32",
                    "controllable": "bool",
                    "in_service": "bool",
                    "name": "object",
                    "p_mw": "float64",
                    "scaling": "float64",
                    "sn_mva": "float64",
                    "type": "object",
                    "vm_pu": "float64",
                    "slack": "bool",
                    "max_p_mw": "float64",
                    "min_p_mw": "float64",
                    "max_q_mvar": "float64",
                    "min_q_mvar": "float64"
                }
            },
            "switch": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"bus\",\"element\",\"et\",\"type\",\"closed\",\"name\",\"z_ohm\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "bus": "int64",
                    "element": "int64",
                    "et": "object",
                    "type": "object",
                    "closed": "bool",
                    "name": "object",
                    "z_ohm": "float64"
                }
            },
            "shunt": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"bus\",\"name\",\"q_mvar\",\"p_mw\",\"vn_kv\",\"step\",\"max_step\",\"in_service\"],\"index\":[0,1,10,11,12,13,2,3,4,5,6,7,8,9],\"data\":[[4,null,40.0,0.0,138.0,1,1,true],[33,null,-14.0,0.0,138.0,1,1,true],[82,null,-10.0,0.0,138.0,1,1,true],[104,null,-20.0,0.0,138.0,1,1,true],[106,null,-6.0,0.0,138.0,1,1,true],[109,null,-6.0,0.0,138.0,1,1,true],[36,null,25.0,0.0,138.0,1,1,true],[43,null,-10.0,0.0,138.0,1,1,true],[44,null,-10.0,0.0,138.0,1,1,true],[45,null,-10.0,0.0,138.0,1,1,true],[47,null,-15.0,0.0,138.0,1,1,true],[73,null,-12.0,0.0,138.0,1,1,true],[78,null,-20.0,0.0,138.0,1,1,true],[81,null,-20.0,0.0,138.0,1,1,true]]}",
                "orient": "split",
                "dtype": {
                    "bus": "uint32",
                    "name": "object",
                    "q_mvar": "float64",
                    "p_mw": "float64",
                    "vn_kv": "float64",
                    "step": "uint32",
                    "max_step": "uint32",
                    "in_service": "bool"
                }
            },
            "ext_grid": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"bus\",\"in_service\",\"name\",\"va_degree\",\"vm_pu\",\"max_p_mw\",\"min_p_mw\",\"max_q_mvar\",\"min_q_mvar\"],\"index\":[0],\"data\":[[68,true,null,30.0,1.035,805.200000000000045,0.0,300.0,-300.0]]}",
                "orient": "split",
                "dtype": {
                    "bus": "uint32",
                    "in_service": "bool",
                    "name": "object",
                    "va_degree": "float64",
                    "vm_pu": "float64",
                    "max_p_mw": "float64",
                    "min_p_mw": "float64",
                    "max_q_mvar": "float64",
                    "min_q_mvar": "float64"
                }
            },
            "line": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"c_nf_per_km\",\"df\",\"from_bus\",\"g_us_per_km\",\"in_service\",\"length_km\",\"max_i_ka\",\"max_loading_percent\",\"name\",\"parallel\",\"r_ohm_per_km\",\"std_type\",\"to_bus\",\"type\",\"x_ohm_per_km\"],\"index\":[0,1,10,100,101,102,103,104,105,106,107,108,109,11,110,111,112,113,114,115,116,117,118,119,12,120,121,122,123,124,125,126,127,128,129,13,130,131,132,133,134,135,136,137,138,139,14,140,141,142,143,144,145,146,147,148,149,15,150,151,152,153,154,155,156,157,158,159,16,160,161,162,163,164,165,166,167,168,169,17,170,171,172,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,42,43,44,45,46,47,48,49,5,50,51,52,53,54,55,56,57,58,59,6,60,61,62,63,64,65,66,67,68,69,7,70,71,72,73,74,75,76,77,78,79,8,80,81,82,83,84,85,86,87,88,89,9,90,91,92,93,94,95,96,97,98,99],\"data\":[[353.789080947117327,1.0,0,0.0,true,1.0,41.418606267951418,100.0,null,1,5.770332,null,1,\"ol\",19.024956],[150.708577001882276,1.0,0,0.0,true,1.0,41.418606267951418,100.0,null,1,2.456676,null,2,\"ol\",8.074655999999999],[69.922093950965717,1.0,10,0.0,true,1.0,41.418606267951418,100.0,null,1,1.133118,null,11,\"ol\",3.732624],[122.294020894318521,1.0,69,0.0,true,1.0,41.418606267951418,100.0,null,1,1.6796808,null,70,\"ol\",6.76062],[679.720753945642855,1.0,23,0.0,true,1.0,41.418606267951418,100.0,null,1,9.293472,null,71,\"ol\",37.326239999999998],[618.991604617712483,1.0,70,0.0,true,1.0,41.418606267951418,100.0,null,1,8.493624,null,71,\"ol\",34.279200000000003],[164.080132817206419,1.0,70,0.0,true,1.0,41.418606267951418,100.0,null,1,1.6492104,null,72,\"ol\",8.645975999999999],[469.118749854287955,1.0,69,0.0,true,1.0,41.418606267951418,100.0,null,1,7.636644,null,73,\"ol\",25.195212000000002],[501.433343074654545,1.0,69,0.0,true,1.0,41.418606267951418,100.0,null,1,8.150831999999999,null,74,\"ol\",26.852039999999999],[1727.15929281269905,1.0,68,0.0,true,1.0,41.418606267951418,100.0,null,1,7.71282,null,74,\"ol\",23.23368],[144.022799094220204,1.0,73,0.0,true,1.0,41.418606267951418,100.0,null,1,2.342412,null,74,\"ol\",7.731864],[512.576306254091378,1.0,75,0.0,true,1.0,41.418606267951418,100.0,null,1,8.455536,null,76,\"ol\",28.185120000000001],[1445.799472531920628,1.0,68,0.0,true,1.0,41.418606267951418,100.0,null,1,5.884596,null,76,\"ol\",19.234439999999999],[218.959226475932468,1.0,1,0.0,true,1.0,41.418606267951418,100.0,null,1,3.561228,null,11,\"ol\",11.731104],[693.370883840452848,1.0,74,0.0,true,1.0,41.418606267951418,100.0,null,1,11.445444,null,76,\"ol\",38.068956],[176.058818235100944,1.0,76,0.0,true,1.0,41.418606267951418,100.0,null,1,0.7160544,null,77,\"ol\",2.361456],[90.25800175343781,1.0,77,0.0,true,1.0,41.418606267951418,100.0,null,1,1.0398024,null,78,\"ol\",4.646736],[657.434827586769188,1.0,76,0.0,true,1.0,41.418606267951418,100.0,null,1,3.23748,null,79,\"ol\",9.23634],[317.574450613947931,1.0,76,0.0,true,1.0,41.418606267951418,100.0,null,1,5.598936,null,79,\"ol\",19.996200000000002],[260.466764319334459,1.0,78,0.0,true,1.0,41.418606267951418,100.0,null,1,2.970864,null,79,\"ol\",13.406976],[1138.532262858951754,1.0,76,0.0,true,1.0,41.418606267951418,100.0,null,1,5.675112,null,81,\"ol\",16.244532],[528.733602864274644,1.0,81,0.0,true,1.0,41.418606267951418,100.0,null,1,2.132928,null,82,\"ol\",6.979626],[359.360562536835744,1.0,82,0.0,true,1.0,41.418606267951418,100.0,null,1,11.9025,null,83,\"ol\",25.138079999999999],[484.718898305499408,1.0,82,0.0,true,1.0,41.418606267951418,100.0,null,1,8.18892,null,84,\"ol\",28.185120000000001],[565.50538135641591,1.0,2,0.0,true,1.0,41.418606267951418,100.0,null,1,9.217295999999999,null,11,\"ol\",30.470400000000002],[171.880207042812117,1.0,83,0.0,true,1.0,41.418606267951418,100.0,null,1,5.751288,null,84,\"ol\",12.207204000000001],[384.432229690568477,1.0,84,0.0,true,1.0,41.418606267951418,100.0,null,1,6.6654,null,85,\"ol\",23.424119999999998],[384.432229690568477,1.0,84,0.0,true,1.0,41.418606267951418,100.0,null,1,3.8088,null,87,\"ol\",19.424880000000002],[654.649086791910122,1.0,84,0.0,true,1.0,41.418606267951418,100.0,null,1,4.551516,null,88,\"ol\",32.94612],[269.381134862883869,1.0,87,0.0,true,1.0,41.418606267951418,100.0,null,1,2.647116,null,88,\"ol\",13.559328000000001],[735.435569842826681,1.0,88,0.0,true,1.0,41.418606267951418,100.0,null,1,9.864792,null,89,\"ol\",35.802720000000001],[1476.442621275371494,1.0,88,0.0,true,1.0,41.418606267951418,100.0,null,1,4.532472,null,89,\"ol\",18.986868000000001],[298.074265049933501,1.0,89,0.0,true,1.0,41.418606267951418,100.0,null,1,4.837176,null,90,\"ol\",15.920783999999999],[763.292977791418593,1.0,88,0.0,true,1.0,41.418606267951418,100.0,null,1,1.885356,null,91,\"ol\",9.61722],[576.648344535852743,1.0,88,0.0,true,1.0,41.418606267951418,100.0,null,1,7.484292,null,91,\"ol\",30.108564000000001],[121.736872735346694,1.0,6,0.0,true,1.0,41.418606267951418,100.0,null,1,1.6415928,null,11,\"ol\",6.47496],[455.190045879991942,1.0,90,0.0,true,1.0,41.418606267951418,100.0,null,1,7.370028,null,91,\"ol\",24.223967999999999],[303.645746639651918,1.0,91,0.0,true,1.0,41.418606267951418,100.0,null,1,4.913352,null,92,\"ol\",16.149311999999998],[565.50538135641591,1.0,91,0.0,true,1.0,41.418606267951418,100.0,null,1,9.160164,null,93,\"ol\",30.08952],[261.302486557792236,1.0,92,0.0,true,1.0,41.418606267951418,100.0,null,1,4.246812,null,93,\"ol\",13.940208],[154.608614114685167,1.0,93,0.0,true,1.0,41.418606267951418,100.0,null,1,2.513808,null,94,\"ol\",8.265096],[688.077976330220395,1.0,79,0.0,true,1.0,41.418606267951418,100.0,null,1,6.779664,null,95,\"ol\",34.660080000000001],[757.72149620170012,1.0,81,0.0,true,1.0,41.418606267951418,100.0,null,1,3.085128,null,95,\"ol\",10.09332],[320.360191408807054,1.0,93,0.0,true,1.0,41.418606267951418,100.0,null,1,5.122836,null,95,\"ol\",16.549236],[353.789080947117327,1.0,79,0.0,true,1.0,41.418606267951418,100.0,null,1,3.485052,null,96,\"ol\",17.787095999999998],[398.360933664864433,1.0,79,0.0,true,1.0,41.418606267951418,100.0,null,1,4.532472,null,97,\"ol\",20.567519999999998],[261.302486557792236,1.0,10,0.0,true,1.0,41.418606267951418,100.0,null,1,4.23729,null,12,\"ol\",13.921163999999999],[760.507236996559413,1.0,79,0.0,true,1.0,41.418606267951418,100.0,null,1,8.645975999999999,null,98,\"ol\",39.230639999999987],[657.434827586769188,1.0,91,0.0,true,1.0,41.418606267951418,100.0,null,1,12.340512,null,99,\"ol\",56.179799999999993],[841.293720047475972,1.0,93,0.0,true,1.0,41.418606267951418,100.0,null,1,3.389832,null,99,\"ol\",11.04552],[205.309096581122446,1.0,94,0.0,true,1.0,41.418606267951418,100.0,null,1,3.256524,null,95,\"ol\",10.417068],[334.288895383103011,1.0,95,0.0,true,1.0,41.418606267951418,100.0,null,1,3.294612,null,96,\"ol\",16.853940000000002],[663.006309176487889,1.0,97,0.0,true,1.0,41.418606267951418,100.0,null,1,7.560468,null,99,\"ol\",34.088760000000001],[300.860005844792738,1.0,98,0.0,true,1.0,41.418606267951418,100.0,null,1,3.42792,null,99,\"ol\",15.482772000000001],[456.861490356907552,1.0,99,0.0,true,1.0,41.418606267951418,100.0,null,1,5.275188,null,100,\"ol\",24.033528],[203.916226183692856,1.0,91,0.0,true,1.0,41.418606267951418,100.0,null,1,2.342412,null,101,\"ol\",10.645595999999999],[409.50389684430121,1.0,100,0.0,true,1.0,41.418606267951418,100.0,null,1,4.684824,null,101,\"ol\",21.329280000000001],[252.94526417321461,1.0,11,0.0,true,1.0,41.418606267951418,100.0,null,1,4.09446,null,13,\"ol\",13.464108],[746.578533022263514,1.0,99,0.0,true,1.0,41.418606267951418,100.0,null,1,3.04704,null,102,\"ol\",9.998100000000001],[753.54288500941152,1.0,99,0.0,true,1.0,41.418606267951418,100.0,null,1,8.588844,null,103,\"ol\",38.849760000000003],[566.898251753845557,1.0,102,0.0,true,1.0,41.418606267951418,100.0,null,1,8.874504,null,103,\"ol\",30.165696],[568.291122151275204,1.0,102,0.0,true,1.0,41.418606267951418,100.0,null,1,10.18854,null,104,\"ol\",30.9465],[863.579646406349525,1.0,99,0.0,true,1.0,41.418606267951418,100.0,null,1,11.52162,null,105,\"ol\",43.610759999999999],[137.337021186558161,1.0,103,0.0,true,1.0,41.418606267951418,100.0,null,1,1.8929736,null,104,\"ol\",7.198632],[199.737614991404058,1.0,104,0.0,true,1.0,41.418606267951418,100.0,null,1,2.66616,null,105,\"ol\",10.417068],[657.434827586769188,1.0,104,0.0,true,1.0,41.418606267951418,100.0,null,1,10.09332,null,106,\"ol\",34.850520000000003],[256.845301286017502,1.0,104,0.0,true,1.0,41.418606267951418,100.0,null,1,4.970484,null,107,\"ol\",13.387931999999999],[657.434827586769188,1.0,105,0.0,true,1.0,41.418606267951418,100.0,null,1,10.09332,null,106,\"ol\",34.850520000000003],[873.051165108870691,1.0,12,0.0,true,1.0,41.418606267951418,100.0,null,1,14.168736000000001,null,14,\"ol\",46.543536000000003],[105.858150204649292,1.0,107,0.0,true,1.0,41.418606267951418,100.0,null,1,1.99962,null,108,\"ol\",5.484672],[642.113253215043756,1.0,102,0.0,true,1.0,41.418606267951418,100.0,null,1,7.4385864,null,109,\"ol\",34.526771999999987],[281.359820280778365,1.0,108,0.0,true,1.0,41.418606267951418,100.0,null,1,5.294232,null,109,\"ol\",14.511528],[278.574079485919185,1.0,109,0.0,true,1.0,41.418606267951418,100.0,null,1,4.18968,null,110,\"ol\",14.378220000000001],[863.579646406349525,1.0,109,0.0,true,1.0,41.418606267951418,100.0,null,1,4.703868,null,111,\"ol\",12.18816],[106.972446522592961,1.0,16,0.0,true,1.0,41.418606267951418,100.0,null,1,1.7387172,null,112,\"ol\",5.732244],[721.506865868530667,1.0,31,0.0,true,1.0,41.418606267951418,100.0,null,1,11.712059999999999,null,112,\"ol\",38.659320000000001],[226.759300701538194,1.0,31,0.0,true,1.0,41.418606267951418,100.0,null,1,2.57094,null,113,\"ol\",11.654928],[274.674042373116322,1.0,26,0.0,true,1.0,41.418606267951418,100.0,null,1,3.123216,null,114,\"ol\",14.111604],[38.443222969056848,1.0,113,0.0,true,1.0,41.418606267951418,100.0,null,1,0.438012,null,114,\"ol\",1.980576],[699.220939509657114,1.0,13,0.0,true,1.0,41.418606267951418,100.0,null,1,11.33118,null,14,\"ol\",37.135800000000003],[498.647602279795365,1.0,11,0.0,true,1.0,41.418606267951418,100.0,null,1,6.265476,null,116,\"ol\",26.6616],[166.86587361206557,1.0,74,0.0,true,1.0,41.418606267951418,100.0,null,1,2.76138,null,117,\"ol\",9.160164],[188.873225891453217,1.0,75,0.0,true,1.0,41.418606267951418,100.0,null,1,3.123216,null,117,\"ol\",10.359935999999999],[298.074265049933501,1.0,11,0.0,true,1.0,41.418606267951418,100.0,null,1,4.037328,null,15,\"ol\",15.882695999999999],[618.434456458740669,1.0,14,0.0,true,1.0,41.418606267951418,100.0,null,1,2.513808,null,16,\"ol\",8.322228000000001],[29.25027834602151,1.0,3,0.0,true,1.0,41.418606267951418,100.0,null,1,0.3351744,null,4,\"ol\",1.5197112],[649.077605202191762,1.0,15,0.0,true,1.0,41.418606267951418,100.0,null,1,8.645975999999999,null,16,\"ol\",34.298243999999997],[180.794577586361555,1.0,16,0.0,true,1.0,41.418606267951418,100.0,null,1,2.342412,null,17,\"ol\",9.61722],[159.065799386459844,1.0,17,0.0,true,1.0,41.418606267951418,100.0,null,1,2.1310236,null,18,\"ol\",9.388692000000001],[415.075378434019626,1.0,18,0.0,true,1.0,41.418606267951418,100.0,null,1,4.799088,null,19,\"ol\",22.281479999999998],[140.679910140389182,1.0,14,0.0,true,1.0,41.418606267951418,100.0,null,1,2.28528,null,18,\"ol\",7.503336],[300.860005844792738,1.0,19,0.0,true,1.0,41.418606267951418,100.0,null,1,3.485052,null,20,\"ol\",16.168355999999999],[342.646117767680607,1.0,20,0.0,true,1.0,41.418606267951418,100.0,null,1,3.980196,null,21,\"ol\",18.47268],[562.71964056155673,1.0,21,0.0,true,1.0,41.418606267951418,100.0,null,1,6.513048,null,22,\"ol\",30.279959999999999],[693.649457919938868,1.0,22,0.0,true,1.0,41.418606267951418,100.0,null,1,2.57094,null,23,\"ol\",9.369648],[1203.440023379170952,1.0,22,0.0,true,1.0,41.418606267951418,100.0,null,1,2.970864,null,24,\"ol\",15.235200000000001],[395.57519287000531,1.0,2,0.0,true,1.0,41.418606267951418,100.0,null,1,4.589604,null,4,\"ol\",20.567519999999998],[2457.023381065807371,1.0,24,0.0,true,1.0,41.418606267951418,100.0,null,1,6.055992,null,26,\"ol\",31.041720000000002],[300.860005844792738,1.0,26,0.0,true,1.0,41.418606267951418,100.0,null,1,3.6431172,null,27,\"ol\",16.282620000000001],[331.503154588243945,1.0,27,0.0,true,1.0,41.418606267951418,100.0,null,1,4.513428,null,28,\"ol\",17.958492],[1145.496614846099646,1.0,7,0.0,true,1.0,16.567442507180569,100.0,null,1,5.1299775,null,29,\"ol\",59.988599999999998],[2023.562113385716884,1.0,25,0.0,true,1.0,16.567442507180569,100.0,null,1,9.510097500000001,null,29,\"ol\",102.361499999999992],[555.755288574408723,1.0,16,0.0,true,1.0,41.418606267951418,100.0,null,1,9.026856,null,30,\"ol\",29.765771999999998],[115.608242986656464,1.0,28,0.0,true,1.0,41.418606267951418,100.0,null,1,2.056752,null,30,\"ol\",6.303564],[1633.836976184916011,1.0,22,0.0,true,1.0,41.418606267951418,100.0,null,1,6.036948,null,31,\"ol\",21.957732],[349.610469754828557,1.0,30,0.0,true,1.0,41.418606267951418,100.0,null,1,5.675112,null,31,\"ol\",18.75834],[268.266838544940128,1.0,26,0.0,true,1.0,41.418606267951418,100.0,null,1,4.361076,null,31,\"ol\",14.378220000000001],[198.623318673460403,1.0,4,0.0,true,1.0,41.418606267951418,100.0,null,1,2.266236,null,5,\"ol\",10.283759999999999],[444.882804939012999,1.0,14,0.0,true,1.0,41.418606267951418,100.0,null,1,7.23672,null,32,\"ol\",23.690736000000001],[880.294091175504832,1.0,18,0.0,true,1.0,41.418606267951418,100.0,null,1,14.321088,null,33,\"ol\",47.038679999999999],[37.328926651113171,1.0,34,0.0,true,1.0,41.418606267951418,100.0,null,1,0.4265856,null,35,\"ol\",1.942488],[183.580318381220764,1.0,34,0.0,true,1.0,41.418606267951418,100.0,null,1,2.09484,null,36,\"ol\",9.464867999999999],[509.790565459232084,1.0,32,0.0,true,1.0,41.418606267951418,100.0,null,1,7.90326,null,36,\"ol\",27.042480000000001],[79.115038574001062,1.0,33,0.0,true,1.0,41.418606267951418,100.0,null,1,1.6587324,null,35,\"ol\",5.103792],[137.058447107072226,1.0,33,0.0,true,1.0,41.418606267951418,100.0,null,1,0.4875264,null,36,\"ol\",1.790136],[376.07500730599088,1.0,36,0.0,true,1.0,41.418606267951418,100.0,null,1,6.113124,null,38,\"ol\",20.186640000000001],[585.005566920430397,1.0,36,0.0,true,1.0,41.418606267951418,100.0,null,1,11.293092,null,39,\"ol\",31.993919999999999],[940.466092344463164,1.0,29,0.0,true,1.0,16.567442507180569,100.0,null,1,5.52276,null,37,\"ol\",64.273499999999999],[76.607871858627774,1.0,5,0.0,true,1.0,41.418606267951418,100.0,null,1,0.8741196,null,6,\"ol\",3.961152],[216.173485681073316,1.0,38,0.0,true,1.0,41.418606267951418,100.0,null,1,3.504096,null,39,\"ol\",11.52162],[170.208762565896649,1.0,39,0.0,true,1.0,41.418606267951418,100.0,null,1,2.76138,null,40,\"ol\",9.274428],[649.077605202191762,1.0,39,0.0,true,1.0,41.418606267951418,100.0,null,1,10.569419999999999,null,41,\"ol\",34.850520000000003],[479.147416715780992,1.0,40,0.0,true,1.0,41.418606267951418,100.0,null,1,7.80804,null,41,\"ol\",25.709399999999999],[845.193757160278892,1.0,42,0.0,true,1.0,41.418606267951418,100.0,null,1,11.578752,null,43,\"ol\",46.733975999999998],[588.627029953747296,1.0,33,0.0,true,1.0,41.418606267951418,100.0,null,1,7.865172,null,42,\"ol\",32.012963999999997],[312.002969024229515,1.0,43,0.0,true,1.0,41.418606267951418,100.0,null,1,4.265856,null,44,\"ol\",17.158643999999999],[462.432971946625855,1.0,44,0.0,true,1.0,41.418606267951418,100.0,null,1,7.6176,null,45,\"ol\",25.823664000000001],[440.147045587752416,1.0,45,0.0,true,1.0,41.418606267951418,100.0,null,1,7.23672,null,46,\"ol\",24.185880000000001],[657.434827586769188,1.0,45,0.0,true,1.0,41.418606267951418,100.0,null,1,11.445444,null,47,\"ol\",35.993160000000003],[2589.62464290110438,1.0,7,0.0,true,1.0,16.567442507180569,100.0,null,1,2.90421,null,8,\"ol\",36.302624999999999],[223.416411747707201,1.0,46,0.0,true,1.0,41.418606267951418,100.0,null,1,3.637404,null,48,\"ol\",11.9025],[1197.868541789452365,1.0,41,0.0,true,1.0,41.418606267951418,100.0,null,1,13.61646,null,48,\"ol\",61.512120000000003],[1197.868541789452365,1.0,41,0.0,true,1.0,41.418606267951418,100.0,null,1,13.61646,null,48,\"ol\",61.512120000000003],[618.434456458740669,1.0,44,0.0,true,1.0,41.418606267951418,100.0,null,1,13.026096000000001,null,48,\"ol\",35.421840000000003],[175.223095996643138,1.0,47,0.0,true,1.0,41.418606267951418,100.0,null,1,3.408876,null,48,\"ol\",9.61722],[261.023912478306272,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,5.084748,null,49,\"ol\",14.321088],[476.361675920921868,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,9.255383999999999,null,50,\"ol\",26.09028],[194.444707481171577,1.0,50,0.0,true,1.0,41.418606267951418,100.0,null,1,3.865932,null,51,\"ol\",11.197872],[565.226807276930117,1.0,51,0.0,true,1.0,41.418606267951418,100.0,null,1,7.71282,null,52,\"ol\",31.136939999999999],[431.789823203174762,1.0,52,0.0,true,1.0,41.418606267951418,100.0,null,1,5.008572,null,53,\"ol\",23.23368],[2741.168942141444859,1.0,8,0.0,true,1.0,16.567442507180569,100.0,null,1,3.070845,null,9,\"ol\",38.326050000000002],[1027.93835330304205,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,13.90212,null,53,\"ol\",55.037159999999993],[1016.795390123604875,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,16.549236,null,53,\"ol\",55.418039999999998],[281.359820280778365,1.0,53,0.0,true,1.0,41.418606267951418,100.0,null,1,3.218436,null,54,\"ol\",13.464108],[101.958113091846428,1.0,53,0.0,true,1.0,41.418606267951418,100.0,null,1,0.52371,null,55,\"ol\",1.818702],[52.09335286386689,1.0,54,0.0,true,1.0,41.418606267951418,100.0,null,1,0.9293472,null,55,\"ol\",2.875644],[337.074636177962248,1.0,55,0.0,true,1.0,41.418606267951418,100.0,null,1,6.532092,null,56,\"ol\",18.396504],[462.432971946625855,1.0,49,0.0,true,1.0,41.418606267951418,100.0,null,1,9.026856,null,56,\"ol\",25.51896],[337.074636177962248,1.0,55,0.0,true,1.0,41.418606267951418,100.0,null,1,6.532092,null,57,\"ol\",18.396504],[249.045227060411776,1.0,50,0.0,true,1.0,41.418606267951418,100.0,null,1,4.85622,null,57,\"ol\",13.692636],[832.936497662898319,1.0,53,0.0,true,1.0,41.418606267951418,100.0,null,1,9.579132,null,58,\"ol\",43.667892000000002],[243.473745470693387,1.0,3,0.0,true,1.0,41.418606267951418,100.0,null,1,3.980196,null,10,\"ol\",13.102271999999999],[792.543256137440153,1.0,55,0.0,true,1.0,41.418606267951418,100.0,null,1,15.7113,null,58,\"ol\",47.800440000000002],[746.578533022263514,1.0,55,0.0,true,1.0,41.418606267951418,100.0,null,1,15.292332,null,58,\"ol\",45.515159999999987],[786.41462638874998,1.0,54,0.0,true,1.0,41.418606267951418,100.0,null,1,9.0249516,null,58,\"ol\",41.096951999999988],[523.719269433528098,1.0,58,0.0,true,1.0,41.418606267951418,100.0,null,1,6.036948,null,59,\"ol\",27.613800000000001],[540.433714202683177,1.0,58,0.0,true,1.0,41.418606267951418,100.0,null,1,6.246432,null,60,\"ol\",28.565999999999999],[202.801929865749173,1.0,59,0.0,true,1.0,41.418606267951418,100.0,null,1,0.5027616,null,60,\"ol\",2.57094],[204.473374342664698,1.0,59,0.0,true,1.0,41.418606267951418,100.0,null,1,2.342412,null,61,\"ol\",10.683684],[136.501298948100384,1.0,60,0.0,true,1.0,41.418606267951418,100.0,null,1,1.5692256,null,61,\"ol\",7.160544],[481.376009351668358,1.0,62,0.0,true,1.0,16.567442507180569,100.0,null,1,2.04723,null,63,\"ol\",23.805],[2331.10789713817212,1.0,37,0.0,true,1.0,16.567442507180569,100.0,null,1,10.724152500000001,null,64,\"ol\",117.358649999999997],[242.080875073263741,1.0,4,0.0,true,1.0,41.418606267951418,100.0,null,1,3.865932,null,10,\"ol\",12.988008000000001],[846.865201637194332,1.0,63,0.0,true,1.0,16.567442507180569,100.0,null,1,3.2017725,null,64,\"ol\",35.945549999999997],[345.431858562539844,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,3.42792,null,65,\"ol\",17.501436000000002],[345.431858562539844,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,3.42792,null,65,\"ol\",17.501436000000002],[805.079089714306519,1.0,61,0.0,true,1.0,41.418606267951418,100.0,null,1,9.179207999999999,null,65,\"ol\",41.515920000000001],[431.789823203174762,1.0,61,0.0,true,1.0,41.418606267951418,100.0,null,1,4.913352,null,66,\"ol\",22.281479999999998],[373.567840590617664,1.0,65,0.0,true,1.0,41.418606267951418,100.0,null,1,4.265856,null,66,\"ol\",19.32966],[987.823685857069336,1.0,46,0.0,true,1.0,41.418606267951418,100.0,null,1,16.073136000000002,null,68,\"ol\",52.904232],[1153.296689071705487,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,18.75834,null,68,\"ol\",61.702559999999998],[1699.301884864107024,1.0,68,0.0,true,1.0,41.418606267951418,100.0,null,1,5.7132,null,69,\"ol\",24.185880000000001],[1420.449231298701989,1.0,23,0.0,true,1.0,41.418606267951418,100.0,null,1,0.4208724,null,69,\"ol\",78.36605999999999]]}",
                "orient": "split",
                "dtype": {
                    "c_nf_per_km": "float64",
                    "df": "float64",
                    "from_bus": "uint32",
                    "g_us_per_km": "float64",
                    "in_service": "bool",
                    "length_km": "float64",
                    "max_i_ka": "float64",
                    "max_loading_percent": "float64",
                    "name": "object",
                    "parallel": "uint32",
                    "r_ohm_per_km": "float64",
                    "std_type": "object",
                    "to_bus": "uint32",
                    "type": "object",
                    "x_ohm_per_km": "float64"
                }
            },
            "trafo": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"df\",\"hv_bus\",\"i0_percent\",\"in_service\",\"lv_bus\",\"max_loading_percent\",\"name\",\"parallel\",\"pfe_kw\",\"shift_degree\",\"sn_mva\",\"std_type\",\"tap_max\",\"tap_neutral\",\"tap_min\",\"tap_phase_shifter\",\"tap_pos\",\"tap_side\",\"tap_step_degree\",\"tap_step_percent\",\"vn_hv_kv\",\"vn_lv_kv\",\"vk_percent\",\"vkr_percent\"],\"index\":[0,1,10,11,12,2,3,4,5,6,7,8,9],\"data\":[[1.0,7,0.0,true,4,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,1.5,345.0,138.0,264.329999999999984,0.0],[1.0,25,0.0,true,24,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,4.0,345.0,138.0,378.180000000000007,0.0],[1.0,80,0.0,true,79,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,6.49999999999999,345.0,138.0,366.300000000000011,0.0],[1.0,86,-0.04494949494949,true,85,100.0,null,1,0.0,0.0,9900.0,null,null,null,null,false,null,null,null,null,161.0,138.0,2072.25986507098105,279.97199999999998],[1.0,115,-0.16565656565657,true,67,100.0,null,1,0.0,0.0,9900.0,null,null,null,null,false,null,null,null,null,345.0,161.0,40.236040821631541,3.366],[1.0,29,0.0,true,16,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,4.0,345.0,138.0,384.120000000000005,0.0],[1.0,37,0.0,true,36,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,6.49999999999999,345.0,138.0,371.25,0.0],[1.0,62,0.0,true,58,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,4.0,345.0,138.0,382.139999999999986,0.0],[1.0,63,0.0,true,60,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,1.5,345.0,138.0,265.319999999999993,0.0],[1.0,64,0.0,true,65,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,6.49999999999999,345.0,138.0,366.300000000000011,0.0],[1.0,64,-0.64444444444444,true,67,100.0,null,1,0.0,0.0,9900.0,null,null,null,null,false,null,null,null,null,345.0,161.0,158.988082081645359,13.662000000000001],[1.0,67,0.0,true,68,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,6.49999999999999,161.0,138.0,366.300000000000011,0.0],[1.0,80,-0.81616161616162,true,67,100.0,null,1,0.0,0.0,9900.0,null,null,null,null,false,null,null,null,null,345.0,161.0,200.72906123678257,17.324999999999999]]}",
                "orient": "split",
                "dtype": {
                    "df": "float64",
                    "hv_bus": "uint32",
                    "i0_percent": "float64",
                    "in_service": "bool",
                    "lv_bus": "uint32",
                    "max_loading_percent": "float64",
                    "name": "object",
                    "parallel": "uint32",
                    "pfe_kw": "float64",
                    "shift_degree": "float64",
                    "sn_mva": "float64",
                    "std_type": "object",
                    "tap_max": "float64",
                    "tap_neutral": "float64",
                    "tap_min": "float64",
                    "tap_phase_shifter": "bool",
                    "tap_pos": "float64",
                    "tap_side": "object",
                    "tap_step_degree": "float64",
                    "tap_step_percent": "float64",
                    "vn_hv_kv": "float64",
                    "vn_lv_kv": "float64",
                    "vk_percent": "float64",
                    "vkr_percent": "float64"
                }
            },
            "trafo3w": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"std_type\",\"hv_bus\",\"mv_bus\",\"lv_bus\",\"sn_hv_mva\",\"sn_mv_mva\",\"sn_lv_mva\",\"vn_hv_kv\",\"vn_mv_kv\",\"vn_lv_kv\",\"vk_hv_percent\",\"vk_mv_percent\",\"vk_lv_percent\",\"vkr_hv_percent\",\"vkr_mv_percent\",\"vkr_lv_percent\",\"pfe_kw\",\"i0_percent\",\"shift_mv_degree\",\"shift_lv_degree\",\"tap_side\",\"tap_neutral\",\"tap_min\",\"tap_max\",\"tap_step_percent\",\"tap_step_degree\",\"tap_pos\",\"tap_at_star_point\",\"in_service\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "name": "object",
                    "std_type": "object",
                    "hv_bus": "uint32",
                    "mv_bus": "uint32",
                    "lv_bus": "uint32",
                    "sn_hv_mva": "float64",
                    "sn_mv_mva": "float64",
                    "sn_lv_mva": "float64",
                    "vn_hv_kv": "float64",
                    "vn_mv_kv": "float64",
                    "vn_lv_kv": "float64",
                    "vk_hv_percent": "float64",
                    "vk_mv_percent": "float64",
                    "vk_lv_percent": "float64",
                    "vkr_hv_percent": "float64",
                    "vkr_mv_percent": "float64",
                    "vkr_lv_percent": "float64",
                    "pfe_kw": "float64",
                    "i0_percent": "float64",
                    "shift_mv_degree": "float64",
                    "shift_lv_degree": "float64",
                    "tap_side": "object",
                    "tap_neutral": "int32",
                    "tap_min": "int32",
                    "tap_max": "int32",
                    "tap_step_percent": "float64",
                    "tap_step_degree": "float64",
                    "tap_pos": "int32",
                    "tap_at_star_point": "bool",
                    "in_service": "bool"
                }
            },
            "impedance": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"from_bus\",\"to_bus\",\"rft_pu\",\"xft_pu\",\"rtf_pu\",\"xtf_pu\",\"sn_mva\",\"in_service\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "name": "object",
                    "from_bus": "uint32",
                    "to_bus": "uint32",
                    "rft_pu": "float64",
                    "xft_pu": "float64",
                    "rtf_pu": "float64",
                    "xtf_pu": "float64",
                    "sn_mva": "float64",
                    "in_service": "bool"
                }
            },
            "dcline": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"from_bus\",\"to_bus\",\"p_mw\",\"loss_percent\",\"loss_mw\",\"vm_from_pu\",\"vm_to_pu\",\"max_p_mw\",\"min_q_from_mvar\",\"min_q_to_mvar\",\"max_q_from_mvar\",\"max_q_to_mvar\",\"in_service\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "name": "object",
                    "from_bus": "uint32",
                    "to_bus": "uint32",
                    "p_mw": "float64",
                    "loss_percent": "float64",
                    "loss_mw": "float64",
                    "vm_from_pu": "float64",
                    "vm_to_pu": "float64",
                    "max_p_mw": "float64",
                    "min_q_from_mvar": "float64",
                    "min_q_to_mvar": "float64",
                    "max_q_from_mvar": "float64",
                    "max_q_to_mvar": "float64",
                    "in_service": "bool"
                }
            },
            "ward": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"ps_mw\",\"qs_mvar\",\"qz_mvar\",\"pz_mw\",\"in_service\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "name": "object",
                    "bus": "uint32",
                    "ps_mw": "float64",
                    "qs_mvar": "float64",
                    "qz_mvar": "float64",
                    "pz_mw": "float64",
                    "in_service": "bool"
                }
            },
            "xward": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"ps_mw\",\"qs_mvar\",\"qz_mvar\",\"pz_mw\",\"r_ohm\",\"x_ohm\",\"vm_pu\",\"in_service\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "name": "object",
                    "bus": "uint32",
                    "ps_mw": "float64",
                    "qs_mvar": "float64",
                    "qz_mvar": "float64",
                    "pz_mw": "float64",
                    "r_ohm": "float64",
                    "x_ohm": "float64",
                    "vm_pu": "float64",
                    "in_service": "bool"
                }
            },
            "measurement": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"measurement_type\",\"element_type\",\"element\",\"value\",\"std_dev\",\"side\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "name": "object",
                    "measurement_type": "object",
                    "element_type": "object",
                    "element": "uint32",
                    "value": "float64",
                    "std_dev": "float64",
                    "side": "object"
                }
            },
            "pwl_cost": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"power_type\",\"element\",\"et\",\"points\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "power_type": "object",
                    "element": "object",
                    "et": "object",
                    "points": "object"
                }
            },
            "poly_cost": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"element\",\"et\",\"cp0_eur\",\"cp1_eur_per_mw\",\"cp2_eur_per_mw2\",\"cq0_eur\",\"cq1_eur_per_mvar\",\"cq2_eur_per_mvar2\"],\"index\":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53],\"data\":[[0.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[1.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[10.0,\"gen\",0.0,20.0,0.0454545,0.0,0.0,0.0],[11.0,\"gen\",0.0,20.0,0.0318471,0.0,0.0,0.0],[12.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[13.0,\"gen\",0.0,20.0,1.42857,0.0,0.0,0.0],[14.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[15.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[16.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[17.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[18.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[19.0,\"gen\",0.0,20.0,0.526316,0.0,0.0,0.0],[2.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[20.0,\"gen\",0.0,20.0,0.0490196,0.0,0.0,0.0],[21.0,\"gen\",0.0,20.0,0.208333,0.0,0.0,0.0],[22.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[23.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[24.0,\"gen\",0.0,20.0,0.0645161,0.0,0.0,0.0],[25.0,\"gen\",0.0,20.0,0.0625,0.0,0.0,0.0],[26.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[27.0,\"gen\",0.0,20.0,0.0255754,0.0,0.0,0.0],[28.0,\"gen\",0.0,20.0,0.0255102,0.0,0.0,0.0],[0.0,\"ext_grid\",0.0,20.0,0.0193648,0.0,0.0,0.0],[3.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[29.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[30.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[31.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[32.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[33.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[34.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[35.0,\"gen\",0.0,20.0,0.0209644,0.0,0.0,0.0],[36.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[37.0,\"gen\",0.0,20.0,2.5,0.0,0.0,0.0],[38.0,\"gen\",0.0,20.0,0.0164745,0.0,0.0,0.0],[4.0,\"gen\",0.0,20.0,0.0222222,0.0,0.0,0.0],[39.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[40.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[41.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[42.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[43.0,\"gen\",0.0,20.0,0.0396825,0.0,0.0,0.0],[44.0,\"gen\",0.0,20.0,0.25,0.0,0.0,0.0],[45.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[46.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[47.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[48.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[5.0,\"gen\",0.0,20.0,0.117647,0.0,0.0,0.0],[49.0,\"gen\",0.0,20.0,0.277778,0.0,0.0,0.0],[50.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[51.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[52.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[6.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[7.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[8.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[9.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0]]}",
                "orient": "split",
                "dtype": {
                    "element": "object",
                    "et": "object",
                    "cp0_eur": "float64",
                    "cp1_eur_per_mw": "float64",
                    "cp2_eur_per_mw2": "float64",
                    "cq0_eur": "float64",
                    "cq1_eur_per_mvar": "float64",
                    "cq2_eur_per_mvar2": "float64"
                }
            },
            "controller": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"object\",\"in_service\",\"order\",\"level\",\"recycle\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "object": "object",
                    "in_service": "bool",
                    "order": "float64",
                    "level": "object",
                    "recycle": "bool"
                }
            },
            "line_geodata": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"coords\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "coords": "object"
                }
            },
            "bus_geodata": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"x\",\"y\",\"coords\"],\"index\":[0,1,10,100,101,102,103,104,105,106,107,108,109,11,110,111,112,113,114,115,116,117,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,42,43,44,45,46,47,48,49,5,50,51,52,53,54,55,56,57,58,59,6,60,61,62,63,64,65,66,67,68,69,7,70,71,72,73,74,75,76,77,78,79,8,80,81,82,83,84,85,86,87,88,89,9,90,91,92,93,94,95,96,97,98,99],\"data\":[[-2.2753708781,2.8543413351,null],[-2.9368186836,2.2121792656,null],[-1.8084809456,1.078348466,null],[-7.6657645056,-8.7967993378,null],[-7.9822420097,-9.888120989300001,null],[-8.236522503,-7.2232594029,null],[-7.9601004239,-7.6514408832,null],[-9.083981855399999,-7.9345527343,null],[-8.1036220416,-8.232699824599999,null],[-9.060039011900001,-8.827258800499999,null],[-10.2222637563,-8.3101899758,null],[-10.4378672891,-7.4563425184,null],[-9.358739758900001,-6.8228628497,null],[-2.4521165899,1.188776931,null],[-10.440240663200001,-6.5781548081,null],[-10.0488510549,-5.9288876624,null],[0.0210809519,-1.4982181621,null],[2.2630604919,-2.7868829987,null],[3.1644478736,-2.3853607175,null],[-4.1528539003,-4.9348933434,null],[-3.5172125372,1.7774067282,null],[-2.1457808291,-8.546699179499999,null],[-2.5339886632,0.2882488225,null],[-3.0867743311,0.3407195505,null],[-2.3297472058,-0.6794711196,null],[-1.4955187998,0.1152182093,null],[-1.0489390134,-1.0911455168,null],[-1.8356962694,-0.8292516322,null],[-2.6758265845,-1.2562696807,null],[-1.271723021,-1.462800663,null],[-1.8344312496,1.7094451782,null],[0.07347017,-2.1572792461,null],[0.9345466389,-3.4895133702,null],[0.3017014198,-3.799814658,null],[-0.2778569786,-5.1101840996,null],[0.9631168234,-2.9810635525,null],[-0.198421303,-2.3375045516,null],[1.9445958496,-2.2280444796,null],[2.4540519038,-1.2219755328,null],[1.4972464461,-0.7253664385,null],[-1.2374314539,-2.0300347993,null],[-0.8886268958,1.5532705585,null],[0.3831112001,-1.275137263,null],[0.9036426682,-2.4715943587,null],[-2.9833939473,-1.6491727215,null],[-3.6197042214,-2.0172857852,null],[-4.3347899304,-2.2080206011,null],[-4.7428784027,-1.3946055976,null],[-2.9559373581,-2.7113176762,null],[-1.9467507717,-3.1510525007,null],[-3.641503222,-3.1155712489,null],[-2.805657193,-3.7660095235,null],[-0.9632829393,0.694729907,null],[-3.152467354,-4.4158860294,null],[-2.274592477,-5.0451097798,null],[-4.1875091242,-3.1292654848,null],[-3.5914723889,-4.4209512352,null],[-2.6299875318,-5.6039710603,null],[-3.1343589918,-6.5469756868,null],[-2.3186784101,-6.714154807,null],[-2.0246271797,-7.1400534325,null],[-1.7242222011,-6.2584123784,null],[-1.1556470812,-7.0861765912,null],[-0.4109732139,1.7675722497,null],[-1.2192198707,-7.6503732093,null],[-0.7485947859,-8.8465566854,null],[0.1459019855,-8.3262367561,null],[-0.4574089283,-6.9830212906,null],[0.861150583,-7.4175967649,null],[0.3692720954,-7.6867284542,null],[-0.3863224274,-8.201226779600001,null],[-0.096485424,-8.545959205300001,null],[0.6559375961,-6.6866792823,null],[1.4936494437,-6.25030887,null],[-1.4836496595,2.2620419749,null],[0.7929941546,-5.649416175,null],[0.0834611333,-5.7403617415,null],[0.5977433777,-5.3868498213,null],[-0.6251705072,-4.7706623304,null],[-1.9502435677,-4.5114830894,null],[-1.2078613144,-5.3568578774,null],[-0.1247860314,-5.0525544148,null],[-3.181060934,-5.4511075483,null],[-2.6512661211,-6.4304476843,null],[-1.2850990413,-6.3036420672,null],[-0.6792997874,-0.5813210368,null],[0.0746177813,-7.0344963198,null],[1.023717234,-5.910201308,null],[1.174663034,-8.1088356138,null],[-1.7822188996,-7.1334058675,null],[-2.5699674814,-7.3039427852,null],[-3.0946467463,-8.595026854,null],[-3.6187253582,-7.4081044036,null],[-3.6927835541,-8.1636937876,null],[-4.7266433485,-7.7222452258,null],[-4.8540470344,-6.9946893219,null],[0.492210326,0.0395449742,null],[-4.4266115974,-5.9494061119,null],[-4.1989304816,-8.569851999000001,null],[-4.1284060755,-9.8585501443,null],[-4.0286420276,-10.845582416899999,null],[-4.9872676968,-10.7009377309,null],[-4.6398857601,-11.9115838943,null],[-4.4754585089,-13.005245318,null],[-5.7173489254,-11.3183526816,null],[-6.2786501696,-10.590002866800001,null],[-7.0445216565,-11.514366900500001,null],[1.5086493002,0.7537990673,null],[-7.5500105531,-10.5464275695,null],[-7.0121455802,-9.433431104,null],[-6.479922409,-9.6952112759,null],[-6.4131486749,-8.5909140285,null],[-5.719463754,-8.846278912800001,null],[-5.3339595297,-8.0606533179,null],[-5.4897495419,-6.9547678609,null],[-6.0865230031,-7.3112763353,null],[-6.2121103608,-6.9919515492,null],[-7.0667004418,-7.9857230104,null]]}",
                "orient": "split",
                "dtype": {
                    "x": "float64",
                    "y": "float64",
                    "coords": "object"
                }
            },
            "version": "2.4.0",
            "converged": "True",
            "name": "",
            "f_hz": 60,
            "sn_mva": 1.0,
            "std_types": {
                "line": {
                    "NAYY 4x150 SE": {
                        "type": "cs",
                        "r_ohm_per_km": 0.208,
                        "q_mm2": 150,
                        "x_ohm_per_km": 0.08,
                        "c_nf_per_km": 261.0,
                        "max_i_ka": 0.27
                    },
                    "70-AL1/11-ST1A 20.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.4132,
                        "q_mm2": 70,
                        "x_ohm_per_km": 0.36,
                        "c_nf_per_km": 9.7,
                        "max_i_ka": 0.29
                    },
                    "NA2XS2Y 1x70 RM/25 6/10 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.443,
                        "q_mm2": 70,
                        "x_ohm_per_km": 0.123,
                        "c_nf_per_km": 280.0,
                        "max_i_ka": 0.217
                    },
                    "N2XS(FL)2Y 1x300 RM/35 64/110 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.06,
                        "q_mm2": 300,
                        "x_ohm_per_km": 0.144,
                        "c_nf_per_km": 144.0,
                        "max_i_ka": 0.588
                    },
                    "NA2XS2Y 1x120 RM/25 6/10 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.253,
                        "q_mm2": 120,
                        "x_ohm_per_km": 0.113,
                        "c_nf_per_km": 340.0,
                        "max_i_ka": 0.28
                    },
                    "149-AL1/24-ST1A 10.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.194,
                        "q_mm2": 149,
                        "x_ohm_per_km": 0.315,
                        "c_nf_per_km": 11.25,
                        "max_i_ka": 0.47
                    },
                    "15-AL1/3-ST1A 0.4": {
                        "type": "ol",
                        "r_ohm_per_km": 1.8769,
                        "q_mm2": 16,
                        "x_ohm_per_km": 0.35,
                        "c_nf_per_km": 11.0,
                        "max_i_ka": 0.105
                    },
                    "NA2XS2Y 1x185 RM/25 6/10 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.161,
                        "q_mm2": 185,
                        "x_ohm_per_km": 0.11,
                        "c_nf_per_km": 406.0,
                        "max_i_ka": 0.358
                    },
                    "NA2XS2Y 1x240 RM/25 6/10 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.122,
                        "q_mm2": 240,
                        "x_ohm_per_km": 0.105,
                        "c_nf_per_km": 456.0,
                        "max_i_ka": 0.416
                    },
                    "N2XS(FL)2Y 1x240 RM/35 64/110 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.075,
                        "q_mm2": 240,
                        "x_ohm_per_km": 0.149,
                        "c_nf_per_km": 135.0,
                        "max_i_ka": 0.526
                    },
                    "NAYY 4x120 SE": {
                        "type": "cs",
                        "r_ohm_per_km": 0.225,
                        "q_mm2": 120,
                        "x_ohm_per_km": 0.08,
                        "c_nf_per_km": 264.0,
                        "max_i_ka": 0.242
                    },
                    "48-AL1/8-ST1A 10.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.5939,
                        "q_mm2": 48,
                        "x_ohm_per_km": 0.35,
                        "c_nf_per_km": 10.1,
                        "max_i_ka": 0.21
                    },
                    "94-AL1/15-ST1A 10.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.306,
                        "q_mm2": 94,
                        "x_ohm_per_km": 0.33,
                        "c_nf_per_km": 10.75,
                        "max_i_ka": 0.35
                    },
                    "NA2XS2Y 1x70 RM/25 12/20 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.443,
                        "q_mm2": 70,
                        "x_ohm_per_km": 0.132,
                        "c_nf_per_km": 190.0,
                        "max_i_ka": 0.22
                    },
                    "243-AL1/39-ST1A 20.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.1188,
                        "q_mm2": 243,
                        "x_ohm_per_km": 0.32,
                        "c_nf_per_km": 11.0,
                        "max_i_ka": 0.645
                    },
                    "NA2XS2Y 1x150 RM/25 6/10 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.206,
                        "q_mm2": 150,
                        "x_ohm_per_km": 0.11,
                        "c_nf_per_km": 360.0,
                        "max_i_ka": 0.315
                    },
                    "184-AL1/30-ST1A 110.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.1571,
                        "q_mm2": 184,
                        "x_ohm_per_km": 0.4,
                        "c_nf_per_km": 8.8,
                        "max_i_ka": 0.535
                    },
                    "149-AL1/24-ST1A 110.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.194,
                        "q_mm2": 149,
                        "x_ohm_per_km": 0.41,
                        "c_nf_per_km": 8.75,
                        "max_i_ka": 0.47
                    },
                    "NA2XS2Y 1x240 RM/25 12/20 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.122,
                        "q_mm2": 240,
                        "x_ohm_per_km": 0.112,
                        "c_nf_per_km": 304.0,
                        "max_i_ka": 0.421
                    },
                    "122-AL1/20-ST1A 20.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.2376,
                        "q_mm2": 122,
                        "x_ohm_per_km": 0.344,
                        "c_nf_per_km": 10.3,
                        "max_i_ka": 0.41
                    },
                    "48-AL1/8-ST1A 20.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.5939,
                        "q_mm2": 48,
                        "x_ohm_per_km": 0.372,
                        "c_nf_per_km": 9.5,
                        "max_i_ka": 0.21
                    },
                    "34-AL1/6-ST1A 10.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.8342,
                        "q_mm2": 34,
                        "x_ohm_per_km": 0.36,
                        "c_nf_per_km": 9.7,
                        "max_i_ka": 0.17
                    },
                    "24-AL1/4-ST1A 0.4": {
                        "type": "ol",
                        "r_ohm_per_km": 1.2012,
                        "q_mm2": 24,
                        "x_ohm_per_km": 0.335,
                        "c_nf_per_km": 11.25,
                        "max_i_ka": 0.14
                    },
                    "184-AL1/30-ST1A 20.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.1571,
                        "q_mm2": 184,
                        "x_ohm_per_km": 0.33,
                        "c_nf_per_km": 10.75,
                        "max_i_ka": 0.535
                    },
                    "94-AL1/15-ST1A 20.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.306,
                        "q_mm2": 94,
                        "x_ohm_per_km": 0.35,
                        "c_nf_per_km": 10.0,
                        "max_i_ka": 0.35
                    },
                    "NAYY 4x50 SE": {
                        "type": "cs",
                        "r_ohm_per_km": 0.642,
                        "q_mm2": 50,
                        "x_ohm_per_km": 0.083,
                        "c_nf_per_km": 210.0,
                        "max_i_ka": 0.142
                    },
                    "490-AL1/64-ST1A 380.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.059,
                        "q_mm2": 490,
                        "x_ohm_per_km": 0.253,
                        "c_nf_per_km": 11.0,
                        "max_i_ka": 0.96
                    },
                    "48-AL1/8-ST1A 0.4": {
                        "type": "ol",
                        "r_ohm_per_km": 0.5939,
                        "q_mm2": 48,
                        "x_ohm_per_km": 0.3,
                        "c_nf_per_km": 12.2,
                        "max_i_ka": 0.21
                    },
                    "NA2XS2Y 1x95 RM/25 6/10 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.313,
                        "q_mm2": 95,
                        "x_ohm_per_km": 0.123,
                        "c_nf_per_km": 315.0,
                        "max_i_ka": 0.249
                    },
                    "NA2XS2Y 1x120 RM/25 12/20 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.253,
                        "q_mm2": 120,
                        "x_ohm_per_km": 0.119,
                        "c_nf_per_km": 230.0,
                        "max_i_ka": 0.283
                    },
                    "34-AL1/6-ST1A 20.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.8342,
                        "q_mm2": 34,
                        "x_ohm_per_km": 0.382,
                        "c_nf_per_km": 9.15,
                        "max_i_ka": 0.17
                    },
                    "94-AL1/15-ST1A 0.4": {
                        "type": "ol",
                        "r_ohm_per_km": 0.306,
                        "q_mm2": 94,
                        "x_ohm_per_km": 0.29,
                        "c_nf_per_km": 13.2,
                        "max_i_ka": 0.35
                    },
                    "NA2XS2Y 1x185 RM/25 12/20 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.161,
                        "q_mm2": 185,
                        "x_ohm_per_km": 0.117,
                        "c_nf_per_km": 273.0,
                        "max_i_ka": 0.362
                    },
                    "NA2XS2Y 1x150 RM/25 12/20 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.206,
                        "q_mm2": 150,
                        "x_ohm_per_km": 0.116,
                        "c_nf_per_km": 250.0,
                        "max_i_ka": 0.319
                    },
                    "243-AL1/39-ST1A 110.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.1188,
                        "q_mm2": 243,
                        "x_ohm_per_km": 0.39,
                        "c_nf_per_km": 9.0,
                        "max_i_ka": 0.645
                    },
                    "490-AL1/64-ST1A 220.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.059,
                        "q_mm2": 490,
                        "x_ohm_per_km": 0.285,
                        "c_nf_per_km": 10.0,
                        "max_i_ka": 0.96
                    },
                    "N2XS(FL)2Y 1x185 RM/35 64/110 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.099,
                        "q_mm2": 185,
                        "x_ohm_per_km": 0.156,
                        "c_nf_per_km": 125.0,
                        "max_i_ka": 0.457
                    },
                    "N2XS(FL)2Y 1x120 RM/35 64/110 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.153,
                        "q_mm2": 120,
                        "x_ohm_per_km": 0.166,
                        "c_nf_per_km": 112.0,
                        "max_i_ka": 0.366
                    },
                    "NA2XS2Y 1x95 RM/25 12/20 kV": {
                        "type": "cs",
                        "r_ohm_per_km": 0.313,
                        "q_mm2": 95,
                        "x_ohm_per_km": 0.132,
                        "c_nf_per_km": 216.0,
                        "max_i_ka": 0.252
                    },
                    "122-AL1/20-ST1A 10.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.2376,
                        "q_mm2": 122,
                        "x_ohm_per_km": 0.323,
                        "c_nf_per_km": 11.1,
                        "max_i_ka": 0.41
                    },
                    "149-AL1/24-ST1A 20.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.194,
                        "q_mm2": 149,
                        "x_ohm_per_km": 0.337,
                        "c_nf_per_km": 10.5,
                        "max_i_ka": 0.47
                    },
                    "70-AL1/11-ST1A 10.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.4132,
                        "q_mm2": 70,
                        "x_ohm_per_km": 0.339,
                        "c_nf_per_km": 10.4,
                        "max_i_ka": 0.29
                    },
                    "305-AL1/39-ST1A 110.0": {
                        "type": "ol",
                        "r_ohm_per_km": 0.0949,
                        "q_mm2": 305,
                        "x_ohm_per_km": 0.38,
                        "c_nf_per_km": 9.2,
                        "max_i_ka": 0.74
                    }
                },
                "trafo": {
                    "0.4 MVA 20/0.4 kV": {
                        "shift_degree": 150,
                        "vector_group": "Dyn5",
                        "vn_hv_kv": 20.0,
                        "pfe_kw": 1.35,
                        "i0_percent": 0.3375,
                        "vn_lv_kv": 0.4,
                        "sn_mva": 0.4,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -2,
                        "vkr_percent": 1.425,
                        "tap_step_percent": 2.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 2,
                        "vk_percent": 6.0
                    },
                    "63 MVA 110/20 kV v1.4.3 and older": {
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "vn_hv_kv": 110.0,
                        "pfe_kw": 33.0,
                        "i0_percent": 0.086,
                        "vn_lv_kv": 20.0,
                        "sn_mva": 63.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.322,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 11.2
                    },
                    "63 MVA 110/10 kV v1.4.3 and older": {
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "vn_hv_kv": 110.0,
                        "pfe_kw": 31.51,
                        "i0_percent": 0.078,
                        "vn_lv_kv": 10.0,
                        "sn_mva": 63.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.31,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 10.04
                    },
                    "25 MVA 110/20 kV v1.4.3 and older": {
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "vn_hv_kv": 110.0,
                        "pfe_kw": 29.0,
                        "i0_percent": 0.071,
                        "vn_lv_kv": 20.0,
                        "sn_mva": 25.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.282,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 11.2
                    },
                    "40 MVA 110/20 kV v1.4.3 and older": {
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "vn_hv_kv": 110.0,
                        "pfe_kw": 31.0,
                        "i0_percent": 0.08,
                        "vn_lv_kv": 20.0,
                        "sn_mva": 40.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.302,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 11.2
                    },
                    "0.25 MVA 20/0.4 kV": {
                        "shift_degree": 150,
                        "vector_group": "Yzn5",
                        "vn_hv_kv": 20.0,
                        "pfe_kw": 0.8,
                        "i0_percent": 0.32,
                        "vn_lv_kv": 0.4,
                        "sn_mva": 0.25,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -2,
                        "vkr_percent": 1.44,
                        "tap_step_percent": 2.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 2,
                        "vk_percent": 6.0
                    },
                    "25 MVA 110/10 kV v1.4.3 and older": {
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "vn_hv_kv": 110.0,
                        "pfe_kw": 28.51,
                        "i0_percent": 0.073,
                        "vn_lv_kv": 10.0,
                        "sn_mva": 25.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.276,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 10.04
                    },
                    "0.25 MVA 10/0.4 kV": {
                        "shift_degree": 150,
                        "vector_group": "Dyn5",
                        "vn_hv_kv": 10.0,
                        "pfe_kw": 0.6,
                        "i0_percent": 0.24,
                        "vn_lv_kv": 0.4,
                        "sn_mva": 0.25,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -2,
                        "vkr_percent": 1.2,
                        "tap_step_percent": 2.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 2,
                        "vk_percent": 4.0
                    },
                    "160 MVA 380/110 kV": {
                        "shift_degree": 0,
                        "vector_group": "Yy0",
                        "vn_hv_kv": 380.0,
                        "pfe_kw": 60.0,
                        "i0_percent": 0.06,
                        "vn_lv_kv": 110.0,
                        "sn_mva": 160.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.25,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 12.2
                    },
                    "63 MVA 110/10 kV": {
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "vn_hv_kv": 110.0,
                        "pfe_kw": 22.0,
                        "i0_percent": 0.04,
                        "vn_lv_kv": 10.0,
                        "sn_mva": 63.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.32,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 18.0
                    },
                    "0.63 MVA 20/0.4 kV": {
                        "shift_degree": 150,
                        "vector_group": "Dyn5",
                        "vn_hv_kv": 20.0,
                        "pfe_kw": 1.65,
                        "i0_percent": 0.2619,
                        "vn_lv_kv": 0.4,
                        "sn_mva": 0.63,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -2,
                        "vkr_percent": 1.206,
                        "tap_step_percent": 2.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 2,
                        "vk_percent": 6.0
                    },
                    "0.4 MVA 10/0.4 kV": {
                        "shift_degree": 150,
                        "vector_group": "Dyn5",
                        "vn_hv_kv": 10.0,
                        "pfe_kw": 0.95,
                        "i0_percent": 0.2375,
                        "vn_lv_kv": 0.4,
                        "sn_mva": 0.4,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -2,
                        "vkr_percent": 1.325,
                        "tap_step_percent": 2.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 2,
                        "vk_percent": 4.0
                    },
                    "0.63 MVA 10/0.4 kV": {
                        "shift_degree": 150,
                        "vector_group": "Dyn5",
                        "vn_hv_kv": 10.0,
                        "pfe_kw": 1.18,
                        "i0_percent": 0.1873,
                        "vn_lv_kv": 0.4,
                        "sn_mva": 0.63,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -2,
                        "vkr_percent": 1.0794,
                        "tap_step_percent": 2.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 2,
                        "vk_percent": 4.0
                    },
                    "63 MVA 110/20 kV": {
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "vn_hv_kv": 110.0,
                        "pfe_kw": 22.0,
                        "i0_percent": 0.04,
                        "vn_lv_kv": 20.0,
                        "sn_mva": 63.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.32,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 18.0
                    },
                    "100 MVA 220/110 kV": {
                        "shift_degree": 0,
                        "vector_group": "Yy0",
                        "vn_hv_kv": 220.0,
                        "pfe_kw": 55.0,
                        "i0_percent": 0.06,
                        "vn_lv_kv": 110.0,
                        "sn_mva": 100.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.26,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 12.0
                    },
                    "25 MVA 110/10 kV": {
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "vn_hv_kv": 110.0,
                        "pfe_kw": 14.0,
                        "i0_percent": 0.07,
                        "vn_lv_kv": 10.0,
                        "sn_mva": 25.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.41,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 12.0
                    },
                    "40 MVA 110/20 kV": {
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "vn_hv_kv": 110.0,
                        "pfe_kw": 18.0,
                        "i0_percent": 0.05,
                        "vn_lv_kv": 20.0,
                        "sn_mva": 40.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.34,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 16.2
                    },
                    "40 MVA 110/10 kV v1.4.3 and older": {
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "vn_hv_kv": 110.0,
                        "pfe_kw": 30.45,
                        "i0_percent": 0.076,
                        "vn_lv_kv": 10.0,
                        "sn_mva": 40.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.295,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 10.04
                    },
                    "25 MVA 110/20 kV": {
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "vn_hv_kv": 110.0,
                        "pfe_kw": 14.0,
                        "i0_percent": 0.07,
                        "vn_lv_kv": 20.0,
                        "sn_mva": 25.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.41,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 12.0
                    },
                    "40 MVA 110/10 kV": {
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "vn_hv_kv": 110.0,
                        "pfe_kw": 18.0,
                        "i0_percent": 0.05,
                        "vn_lv_kv": 10.0,
                        "sn_mva": 40.0,
                        "tap_step_degree": 0,
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "vkr_percent": 0.34,
                        "tap_step_percent": 1.5,
                        "tap_side": "hv",
                        "tap_phase_shifter": "False",
                        "tap_max": 9,
                        "vk_percent": 16.2
                    }
                },
                "trafo3w": {
                    "63/25/38 MVA 110/10/10 kV": {
                        "vector_group": "YN0yn0yn0",
                        "vn_mv_kv": 10,
                        "vn_lv_kv": 10,
                        "shift_lv_degree": 0,
                        "shift_mv_degree": 0,
                        "pfe_kw": 35,
                        "vn_hv_kv": 110,
                        "i0_percent": 0.89,
                        "sn_lv_mva": 38.0,
                        "sn_hv_mva": 63.0,
                        "sn_mv_mva": 25.0,
                        "vkr_lv_percent": 0.35,
                        "tap_neutral": 0,
                        "tap_min": -10,
                        "vk_mv_percent": 10.4,
                        "vkr_hv_percent": 0.28,
                        "vk_lv_percent": 10.4,
                        "tap_max": 10,
                        "vkr_mv_percent": 0.32,
                        "tap_step_percent": 1.2,
                        "tap_side": "hv",
                        "vk_hv_percent": 10.4
                    },
                    "63/25/38 MVA 110/20/10 kV": {
                        "vector_group": "YN0yn0yn0",
                        "vn_mv_kv": 20,
                        "vn_lv_kv": 10,
                        "shift_lv_degree": 0,
                        "shift_mv_degree": 0,
                        "pfe_kw": 35,
                        "vn_hv_kv": 110,
                        "i0_percent": 0.89,
                        "sn_lv_mva": 38.0,
                        "sn_hv_mva": 63.0,
                        "sn_mv_mva": 25.0,
                        "vkr_lv_percent": 0.35,
                        "tap_neutral": 0,
                        "tap_min": -10,
                        "vk_mv_percent": 10.4,
                        "vkr_hv_percent": 0.28,
                        "vk_lv_percent": 10.4,
                        "tap_max": 10,
                        "vkr_mv_percent": 0.32,
                        "tap_step_percent": 1.2,
                        "tap_side": "hv",
                        "vk_hv_percent": 10.4
                    }
                }
            },
            "res_bus": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"vm_pu\",\"va_degree\",\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "vm_pu": "float64",
                    "va_degree": "float64",
                    "p_mw": "float64",
                    "q_mvar": "float64"
                }
            },
            "res_line": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_from_mw\",\"q_from_mvar\",\"p_to_mw\",\"q_to_mvar\",\"pl_mw\",\"ql_mvar\",\"i_from_ka\",\"i_to_ka\",\"i_ka\",\"vm_from_pu\",\"va_from_degree\",\"vm_to_pu\",\"va_to_degree\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_from_mw": "float64",
                    "q_from_mvar": "float64",
                    "p_to_mw": "float64",
                    "q_to_mvar": "float64",
                    "pl_mw": "float64",
                    "ql_mvar": "float64",
                    "i_from_ka": "float64",
                    "i_to_ka": "float64",
                    "i_ka": "float64",
                    "vm_from_pu": "float64",
                    "va_from_degree": "float64",
                    "vm_to_pu": "float64",
                    "va_to_degree": "float64",
                    "loading_percent": "float64"
                }
            },
            "res_trafo": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_hv_mw\",\"q_hv_mvar\",\"p_lv_mw\",\"q_lv_mvar\",\"pl_mw\",\"ql_mvar\",\"i_hv_ka\",\"i_lv_ka\",\"vm_hv_pu\",\"va_hv_degree\",\"vm_lv_pu\",\"va_lv_degree\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_hv_mw": "float64",
                    "q_hv_mvar": "float64",
                    "p_lv_mw": "float64",
                    "q_lv_mvar": "float64",
                    "pl_mw": "float64",
                    "ql_mvar": "float64",
                    "i_hv_ka": "float64",
                    "i_lv_ka": "float64",
                    "vm_hv_pu": "float64",
                    "va_hv_degree": "float64",
                    "vm_lv_pu": "float64",
                    "va_lv_degree": "float64",
                    "loading_percent": "float64"
                }
            },
            "res_trafo3w": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_hv_mw\",\"q_hv_mvar\",\"p_mv_mw\",\"q_mv_mvar\",\"p_lv_mw\",\"q_lv_mvar\",\"pl_mw\",\"ql_mvar\",\"i_hv_ka\",\"i_mv_ka\",\"i_lv_ka\",\"vm_hv_pu\",\"va_hv_degree\",\"vm_mv_pu\",\"va_mv_degree\",\"vm_lv_pu\",\"va_lv_degree\",\"va_internal_degree\",\"vm_internal_pu\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_hv_mw": "float64",
                    "q_hv_mvar": "float64",
                    "p_mv_mw": "float64",
                    "q_mv_mvar": "float64",
                    "p_lv_mw": "float64",
                    "q_lv_mvar": "float64",
                    "pl_mw": "float64",
                    "ql_mvar": "float64",
                    "i_hv_ka": "float64",
                    "i_mv_ka": "float64",
                    "i_lv_ka": "float64",
                    "vm_hv_pu": "float64",
                    "va_hv_degree": "float64",
                    "vm_mv_pu": "float64",
                    "va_mv_degree": "float64",
                    "vm_lv_pu": "float64",
                    "va_lv_degree": "float64",
                    "va_internal_degree": "float64",
                    "vm_internal_pu": "float64",
                    "loading_percent": "float64"
                }
            },
            "res_impedance": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_from_mw\",\"q_from_mvar\",\"p_to_mw\",\"q_to_mvar\",\"pl_mw\",\"ql_mvar\",\"i_from_ka\",\"i_to_ka\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_from_mw": "float64",
                    "q_from_mvar": "float64",
                    "p_to_mw": "float64",
                    "q_to_mvar": "float64",
                    "pl_mw": "float64",
                    "ql_mvar": "float64",
                    "i_from_ka": "float64",
                    "i_to_ka": "float64"
                }
            },
            "res_ext_grid": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64"
                }
            },
            "res_load": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64"
                }
            },
            "res_motor": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64"
                }
            },
            "res_sgen": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64"
                }
            },
            "res_storage": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64"
                }
            },
            "res_shunt": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\",\"vm_pu\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64",
                    "vm_pu": "float64"
                }
            },
            "res_gen": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\",\"va_degree\",\"vm_pu\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64",
                    "va_degree": "float64",
                    "vm_pu": "float64"
                }
            },
            "res_ward": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\",\"vm_pu\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64",
                    "vm_pu": "float64"
                }
            },
            "res_xward": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\",\"vm_pu\",\"va_internal_degree\",\"vm_internal_pu\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64",
                    "vm_pu": "float64",
                    "va_internal_degree": "float64",
                    "vm_internal_pu": "float64"
                }
            },
            "res_dcline": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_from_mw\",\"q_from_mvar\",\"p_to_mw\",\"q_to_mvar\",\"pl_mw\",\"vm_from_pu\",\"va_from_degree\",\"vm_to_pu\",\"va_to_degree\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_from_mw": "float64",
                    "q_from_mvar": "float64",
                    "p_to_mw": "float64",
                    "q_to_mvar": "float64",
                    "pl_mw": "float64",
                    "vm_from_pu": "float64",
                    "va_from_degree": "float64",
                    "vm_to_pu": "float64",
                    "va_to_degree": "float64"
                }
            },
            "res_bus_est": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"vm_pu\",\"va_degree\",\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "vm_pu": "float64",
                    "va_degree": "float64",
                    "p_mw": "float64",
                    "q_mvar": "float64"
                }
            },
            "res_line_est": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_from_mw\",\"q_from_mvar\",\"p_to_mw\",\"q_to_mvar\",\"pl_mw\",\"ql_mvar\",\"i_from_ka\",\"i_to_ka\",\"i_ka\",\"vm_from_pu\",\"va_from_degree\",\"vm_to_pu\",\"va_to_degree\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_from_mw": "float64",
                    "q_from_mvar": "float64",
                    "p_to_mw": "float64",
                    "q_to_mvar": "float64",
                    "pl_mw": "float64",
                    "ql_mvar": "float64",
                    "i_from_ka": "float64",
                    "i_to_ka": "float64",
                    "i_ka": "float64",
                    "vm_from_pu": "float64",
                    "va_from_degree": "float64",
                    "vm_to_pu": "float64",
                    "va_to_degree": "float64",
                    "loading_percent": "float64"
                }
            },
            "res_trafo_est": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_hv_mw\",\"q_hv_mvar\",\"p_lv_mw\",\"q_lv_mvar\",\"pl_mw\",\"ql_mvar\",\"i_hv_ka\",\"i_lv_ka\",\"vm_hv_pu\",\"va_hv_degree\",\"vm_lv_pu\",\"va_lv_degree\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_hv_mw": "float64",
                    "q_hv_mvar": "float64",
                    "p_lv_mw": "float64",
                    "q_lv_mvar": "float64",
                    "pl_mw": "float64",
                    "ql_mvar": "float64",
                    "i_hv_ka": "float64",
                    "i_lv_ka": "float64",
                    "vm_hv_pu": "float64",
                    "va_hv_degree": "float64",
                    "vm_lv_pu": "float64",
                    "va_lv_degree": "float64",
                    "loading_percent": "float64"
                }
            },
            "res_trafo3w_est": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_hv_mw\",\"q_hv_mvar\",\"p_mv_mw\",\"q_mv_mvar\",\"p_lv_mw\",\"q_lv_mvar\",\"pl_mw\",\"ql_mvar\",\"i_hv_ka\",\"i_mv_ka\",\"i_lv_ka\",\"vm_hv_pu\",\"va_hv_degree\",\"vm_mv_pu\",\"va_mv_degree\",\"vm_lv_pu\",\"va_lv_degree\",\"va_internal_degree\",\"vm_internal_pu\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_hv_mw": "float64",
                    "q_hv_mvar": "float64",
                    "p_mv_mw": "float64",
                    "q_mv_mvar": "float64",
                    "p_lv_mw": "float64",
                    "q_lv_mvar": "float64",
                    "pl_mw": "float64",
                    "ql_mvar": "float64",
                    "i_hv_ka": "float64",
                    "i_mv_ka": "float64",
                    "i_lv_ka": "float64",
                    "vm_hv_pu": "float64",
                    "va_hv_degree": "float64",
                    "vm_mv_pu": "float64",
                    "va_mv_degree": "float64",
                    "vm_lv_pu": "float64",
                    "va_lv_degree": "float64",
                    "va_internal_degree": "float64",
                    "vm_internal_pu": "float64",
                    "loading_percent": "float64"
                }
            },
            "res_bus_sc": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                "orient": "split"
            },
            "res_line_sc": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                "orient": "split"
            },
            "res_trafo_sc": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                "orient": "split"
            },
            "res_trafo3w_sc": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                "orient": "split"
            },
            "res_ext_grid_sc": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                "orient": "split"
            },
            "res_gen_sc": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                "orient": "split"
            },
            "res_sgen_sc": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                "orient": "split"
            },
            "res_bus_3ph": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"vm_a_pu\",\"va_a_degree\",\"vm_b_pu\",\"va_b_degree\",\"vm_c_pu\",\"va_c_degree\",\"p_a_mw\",\"q_a_mvar\",\"p_b_mw\",\"q_b_mvar\",\"p_c_mw\",\"q_c_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "vm_a_pu": "float64",
                    "va_a_degree": "float64",
                    "vm_b_pu": "float64",
                    "va_b_degree": "float64",
                    "vm_c_pu": "float64",
                    "va_c_degree": "float64",
                    "p_a_mw": "float64",
                    "q_a_mvar": "float64",
                    "p_b_mw": "float64",
                    "q_b_mvar": "float64",
                    "p_c_mw": "float64",
                    "q_c_mvar": "float64"
                }
            },
            "res_line_3ph": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_a_from_mw\",\"q_a_from_mvar\",\"p_b_from_mw\",\"q_b_from_mvar\",\"q_c_from_mvar\",\"p_a_to_mw\",\"q_a_to_mvar\",\"p_b_to_mw\",\"q_b_to_mvar\",\"p_c_to_mw\",\"q_c_to_mvar\",\"p_a_l_mw\",\"q_a_l_mvar\",\"p_b_l_mw\",\"q_b_l_mvar\",\"p_c_l_mw\",\"q_c_l_mvar\",\"i_a_from_ka\",\"i_a_to_ka\",\"i_b_from_ka\",\"i_b_to_ka\",\"i_c_from_ka\",\"i_c_to_ka\",\"i_a_ka\",\"i_b_ka\",\"i_c_ka\",\"i_n_from_ka\",\"i_n_to_ka\",\"i_n_ka\",\"loading_a_percent\",\"loading_b_percent\",\"loading_c_percent\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_a_from_mw": "float64",
                    "q_a_from_mvar": "float64",
                    "p_b_from_mw": "float64",
                    "q_b_from_mvar": "float64",
                    "q_c_from_mvar": "float64",
                    "p_a_to_mw": "float64",
                    "q_a_to_mvar": "float64",
                    "p_b_to_mw": "float64",
                    "q_b_to_mvar": "float64",
                    "p_c_to_mw": "float64",
                    "q_c_to_mvar": "float64",
                    "p_a_l_mw": "float64",
                    "q_a_l_mvar": "float64",
                    "p_b_l_mw": "float64",
                    "q_b_l_mvar": "float64",
                    "p_c_l_mw": "float64",
                    "q_c_l_mvar": "float64",
                    "i_a_from_ka": "float64",
                    "i_a_to_ka": "float64",
                    "i_b_from_ka": "float64",
                    "i_b_to_ka": "float64",
                    "i_c_from_ka": "float64",
                    "i_c_to_ka": "float64",
                    "i_a_ka": "float64",
                    "i_b_ka": "float64",
                    "i_c_ka": "float64",
                    "i_n_from_ka": "float64",
                    "i_n_to_ka": "float64",
                    "i_n_ka": "float64",
                    "loading_a_percent": "float64",
                    "loading_b_percent": "float64",
                    "loading_c_percent": "float64"
                }
            },
            "res_trafo_3ph": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_a_hv_mw\",\"q_a_hv_mvar\",\"p_b_hv_mw\",\"q_b_hv_mvar\",\"p_c_hv_mw\",\"q_c_hv_mvar\",\"p_a_lv_mw\",\"q_a_lv_mvar\",\"p_b_lv_mw\",\"q_b_lv_mvar\",\"p_c_lv_mw\",\"q_c_lv_mvar\",\"p_a_l_mw\",\"q_a_l_mvar\",\"p_b_l_mw\",\"q_b_l_mvar\",\"p_c_l_mw\",\"q_c_l_mvar\",\"i_a_hv_ka\",\"i_a_lv_ka\",\"i_b_hv_ka\",\"i_b_lv_ka\",\"i_c_hv_ka\",\"i_c_lv_ka\",\"loading_a_percent\",\"loading_b_percent\",\"loading_c_percent\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_a_hv_mw": "float64",
                    "q_a_hv_mvar": "float64",
                    "p_b_hv_mw": "float64",
                    "q_b_hv_mvar": "float64",
                    "p_c_hv_mw": "float64",
                    "q_c_hv_mvar": "float64",
                    "p_a_lv_mw": "float64",
                    "q_a_lv_mvar": "float64",
                    "p_b_lv_mw": "float64",
                    "q_b_lv_mvar": "float64",
                    "p_c_lv_mw": "float64",
                    "q_c_lv_mvar": "float64",
                    "p_a_l_mw": "float64",
                    "q_a_l_mvar": "float64",
                    "p_b_l_mw": "float64",
                    "q_b_l_mvar": "float64",
                    "p_c_l_mw": "float64",
                    "q_c_l_mvar": "float64",
                    "i_a_hv_ka": "float64",
                    "i_a_lv_ka": "float64",
                    "i_b_hv_ka": "float64",
                    "i_b_lv_ka": "float64",
                    "i_c_hv_ka": "float64",
                    "i_c_lv_ka": "float64",
                    "loading_a_percent": "float64",
                    "loading_b_percent": "float64",
                    "loading_c_percent": "float64",
                    "loading_percent": "float64"
                }
            },
            "res_ext_grid_3ph": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_a_mw\",\"q_a_mvar\",\"p_b_mw\",\"q_b_mvar\",\"p_c_mw\",\"q_c_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_a_mw": "float64",
                    "q_a_mvar": "float64",
                    "p_b_mw": "float64",
                    "q_b_mvar": "float64",
                    "p_c_mw": "float64",
                    "q_c_mvar": "float64"
                }
            },
            "res_shunt_3ph": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                "orient": "split"
            },
            "res_load_3ph": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64"
                }
            },
            "res_sgen_3ph": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64"
                }
            },
            "res_storage_3ph": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64"
                }
            },
            "res_asymmetric_load_3ph": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_a_mw\",\"q_a_mvar\",\"p_b_mw\",\"q_b_mvar\",\"p_c_mw\",\"q_c_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_a_mw": "float64",
                    "q_a_mvar": "float64",
                    "p_b_mw": "float64",
                    "q_b_mvar": "float64",
                    "p_c_mw": "float64",
                    "q_c_mvar": "float64"
                }
            },
            "res_asymmetric_sgen_3ph": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_a_mw\",\"q_a_mvar\",\"p_b_mw\",\"q_b_mvar\",\"p_c_mw\",\"q_c_mvar\"],\"index\":[],\"data\":[]}",
                "orient": "split",
                "dtype": {
                    "p_a_mw": "float64",
                    "q_a_mvar": "float64",
                    "p_b_mw": "float64",
                    "q_b_mvar": "float64",
                    "p_c_mw": "float64",
                    "q_c_mvar": "float64"
                }
            },
            "user_pf_options": {

            },
            "OPF_converged": {
                "_module": "pandapower.auxiliary",
                "_class": "pandapowerNet",
                "_object": {
                    "bus": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"in_service\",\"max_vm_pu\",\"min_vm_pu\",\"name\",\"type\",\"vn_kv\",\"zone\"],\"index\":[0,1,10,100,101,102,103,104,105,106,107,108,109,11,110,111,112,113,114,115,116,117,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,42,43,44,45,46,47,48,49,5,50,51,52,53,54,55,56,57,58,59,6,60,61,62,63,64,65,66,67,68,69,7,70,71,72,73,74,75,76,77,78,79,8,80,81,82,83,84,85,86,87,88,89,9,90,91,92,93,94,95,96,97,98,99],\"data\":[[true,1.06,0.94,1,\"b\",138.0,1.0],[true,1.06,0.94,2,\"b\",138.0,1.0],[true,1.06,0.94,11,\"b\",138.0,1.0],[true,1.06,0.94,101,\"b\",138.0,1.0],[true,1.06,0.94,102,\"b\",138.0,1.0],[true,1.06,0.94,103,\"b\",138.0,1.0],[true,1.06,0.94,104,\"b\",138.0,1.0],[true,1.06,0.94,105,\"b\",138.0,1.0],[true,1.06,0.94,106,\"b\",138.0,1.0],[true,1.06,0.94,107,\"b\",138.0,1.0],[true,1.06,0.94,108,\"b\",138.0,1.0],[true,1.06,0.94,109,\"b\",138.0,1.0],[true,1.06,0.94,110,\"b\",138.0,1.0],[true,1.06,0.94,12,\"b\",138.0,1.0],[true,1.06,0.94,111,\"b\",138.0,1.0],[true,1.06,0.94,112,\"b\",138.0,1.0],[true,1.06,0.94,113,\"b\",138.0,1.0],[true,1.06,0.94,114,\"b\",138.0,1.0],[true,1.06,0.94,115,\"b\",138.0,1.0],[true,1.06,0.94,116,\"b\",345.0,1.0],[true,1.06,0.94,117,\"b\",138.0,1.0],[true,1.06,0.94,118,\"b\",138.0,1.0],[true,1.06,0.94,13,\"b\",138.0,1.0],[true,1.06,0.94,14,\"b\",138.0,1.0],[true,1.06,0.94,15,\"b\",138.0,1.0],[true,1.06,0.94,16,\"b\",138.0,1.0],[true,1.06,0.94,17,\"b\",138.0,1.0],[true,1.06,0.94,18,\"b\",138.0,1.0],[true,1.06,0.94,19,\"b\",138.0,1.0],[true,1.06,0.94,20,\"b\",138.0,1.0],[true,1.06,0.94,3,\"b\",138.0,1.0],[true,1.06,0.94,21,\"b\",138.0,1.0],[true,1.06,0.94,22,\"b\",138.0,1.0],[true,1.06,0.94,23,\"b\",138.0,1.0],[true,1.06,0.94,24,\"b\",138.0,1.0],[true,1.06,0.94,25,\"b\",138.0,1.0],[true,1.06,0.94,26,\"b\",345.0,1.0],[true,1.06,0.94,27,\"b\",138.0,1.0],[true,1.06,0.94,28,\"b\",138.0,1.0],[true,1.06,0.94,29,\"b\",138.0,1.0],[true,1.06,0.94,30,\"b\",345.0,1.0],[true,1.06,0.94,4,\"b\",138.0,1.0],[true,1.06,0.94,31,\"b\",138.0,1.0],[true,1.06,0.94,32,\"b\",138.0,1.0],[true,1.06,0.94,33,\"b\",138.0,1.0],[true,1.06,0.94,34,\"b\",138.0,1.0],[true,1.06,0.94,35,\"b\",138.0,1.0],[true,1.06,0.94,36,\"b\",138.0,1.0],[true,1.06,0.94,37,\"b\",138.0,1.0],[true,1.06,0.94,38,\"b\",345.0,1.0],[true,1.06,0.94,39,\"b\",138.0,1.0],[true,1.06,0.94,40,\"b\",138.0,1.0],[true,1.06,0.94,5,\"b\",138.0,1.0],[true,1.06,0.94,41,\"b\",138.0,1.0],[true,1.06,0.94,42,\"b\",138.0,1.0],[true,1.06,0.94,43,\"b\",138.0,1.0],[true,1.06,0.94,44,\"b\",138.0,1.0],[true,1.06,0.94,45,\"b\",138.0,1.0],[true,1.06,0.94,46,\"b\",138.0,1.0],[true,1.06,0.94,47,\"b\",138.0,1.0],[true,1.06,0.94,48,\"b\",138.0,1.0],[true,1.06,0.94,49,\"b\",138.0,1.0],[true,1.06,0.94,50,\"b\",138.0,1.0],[true,1.06,0.94,6,\"b\",138.0,1.0],[true,1.06,0.94,51,\"b\",138.0,1.0],[true,1.06,0.94,52,\"b\",138.0,1.0],[true,1.06,0.94,53,\"b\",138.0,1.0],[true,1.06,0.94,54,\"b\",138.0,1.0],[true,1.06,0.94,55,\"b\",138.0,1.0],[true,1.06,0.94,56,\"b\",138.0,1.0],[true,1.06,0.94,57,\"b\",138.0,1.0],[true,1.06,0.94,58,\"b\",138.0,1.0],[true,1.06,0.94,59,\"b\",138.0,1.0],[true,1.06,0.94,60,\"b\",138.0,1.0],[true,1.06,0.94,7,\"b\",138.0,1.0],[true,1.06,0.94,61,\"b\",138.0,1.0],[true,1.06,0.94,62,\"b\",138.0,1.0],[true,1.06,0.94,63,\"b\",345.0,1.0],[true,1.06,0.94,64,\"b\",345.0,1.0],[true,1.06,0.94,65,\"b\",345.0,1.0],[true,1.06,0.94,66,\"b\",138.0,1.0],[true,1.06,0.94,67,\"b\",138.0,1.0],[true,1.06,0.94,68,\"b\",161.0,1.0],[true,1.06,0.94,69,\"b\",138.0,1.0],[true,1.06,0.94,70,\"b\",138.0,1.0],[true,1.06,0.94,8,\"b\",345.0,1.0],[true,1.06,0.94,71,\"b\",138.0,1.0],[true,1.06,0.94,72,\"b\",138.0,1.0],[true,1.06,0.94,73,\"b\",138.0,1.0],[true,1.06,0.94,74,\"b\",138.0,1.0],[true,1.06,0.94,75,\"b\",138.0,1.0],[true,1.06,0.94,76,\"b\",138.0,1.0],[true,1.06,0.94,77,\"b\",138.0,1.0],[true,1.06,0.94,78,\"b\",138.0,1.0],[true,1.06,0.94,79,\"b\",138.0,1.0],[true,1.06,0.94,80,\"b\",138.0,1.0],[true,1.06,0.94,9,\"b\",345.0,1.0],[true,1.06,0.94,81,\"b\",345.0,1.0],[true,1.06,0.94,82,\"b\",138.0,1.0],[true,1.06,0.94,83,\"b\",138.0,1.0],[true,1.06,0.94,84,\"b\",138.0,1.0],[true,1.06,0.94,85,\"b\",138.0,1.0],[true,1.06,0.94,86,\"b\",138.0,1.0],[true,1.06,0.94,87,\"b\",161.0,1.0],[true,1.06,0.94,88,\"b\",138.0,1.0],[true,1.06,0.94,89,\"b\",138.0,1.0],[true,1.06,0.94,90,\"b\",138.0,1.0],[true,1.06,0.94,10,\"b\",345.0,1.0],[true,1.06,0.94,91,\"b\",138.0,1.0],[true,1.06,0.94,92,\"b\",138.0,1.0],[true,1.06,0.94,93,\"b\",138.0,1.0],[true,1.06,0.94,94,\"b\",138.0,1.0],[true,1.06,0.94,95,\"b\",138.0,1.0],[true,1.06,0.94,96,\"b\",138.0,1.0],[true,1.06,0.94,97,\"b\",138.0,1.0],[true,1.06,0.94,98,\"b\",138.0,1.0],[true,1.06,0.94,99,\"b\",138.0,1.0],[true,1.06,0.94,100,\"b\",138.0,1.0]]}",
                        "orient": "split",
                        "dtype": {
                            "in_service": "bool",
                            "max_vm_pu": "float64",
                            "min_vm_pu": "float64",
                            "name": "object",
                            "type": "object",
                            "vn_kv": "float64",
                            "zone": "object"
                        }
                    },
                    "load": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"bus\",\"const_i_percent\",\"const_z_percent\",\"controllable\",\"in_service\",\"name\",\"p_mw\",\"q_mvar\",\"scaling\",\"sn_mva\",\"type\"],\"index\":[0,1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,42,43,44,45,46,47,48,49,5,50,51,52,53,54,55,56,57,58,59,6,60,61,62,63,64,65,66,67,68,69,7,70,71,72,73,74,75,76,77,78,79,8,80,81,82,83,84,85,86,87,88,89,9,90,91,92,93,94,95,96,97,98],\"data\":[[0,0.0,0.0,false,true,null,51.0,27.0,1.0,null,null],[1,0.0,0.0,false,true,null,20.0,9.0,1.0,null,null],[13,0.0,0.0,false,true,null,14.0,1.0,1.0,null,null],[14,0.0,0.0,false,true,null,90.0,30.0,1.0,null,null],[15,0.0,0.0,false,true,null,25.0,10.0,1.0,null,null],[16,0.0,0.0,false,true,null,11.0,3.0,1.0,null,null],[17,0.0,0.0,false,true,null,60.0,34.0,1.0,null,null],[18,0.0,0.0,false,true,null,45.0,25.0,1.0,null,null],[19,0.0,0.0,false,true,null,18.0,3.0,1.0,null,null],[20,0.0,0.0,false,true,null,14.0,8.0,1.0,null,null],[21,0.0,0.0,false,true,null,10.0,5.0,1.0,null,null],[22,0.0,0.0,false,true,null,7.0,3.0,1.0,null,null],[2,0.0,0.0,false,true,null,39.0,10.0,1.0,null,null],[23,0.0,0.0,false,true,null,13.0,0.0,1.0,null,null],[26,0.0,0.0,false,true,null,71.0,13.0,1.0,null,null],[27,0.0,0.0,false,true,null,17.0,7.0,1.0,null,null],[28,0.0,0.0,false,true,null,24.0,4.0,1.0,null,null],[30,0.0,0.0,false,true,null,43.0,27.0,1.0,null,null],[31,0.0,0.0,false,true,null,59.0,23.0,1.0,null,null],[32,0.0,0.0,false,true,null,23.0,9.0,1.0,null,null],[33,0.0,0.0,false,true,null,59.0,26.0,1.0,null,null],[34,0.0,0.0,false,true,null,33.0,9.0,1.0,null,null],[35,0.0,0.0,false,true,null,31.0,17.0,1.0,null,null],[3,0.0,0.0,false,true,null,39.0,12.0,1.0,null,null],[38,0.0,0.0,false,true,null,27.0,11.0,1.0,null,null],[39,0.0,0.0,false,true,null,66.0,23.0,1.0,null,null],[40,0.0,0.0,false,true,null,37.0,10.0,1.0,null,null],[41,0.0,0.0,false,true,null,96.0,23.0,1.0,null,null],[42,0.0,0.0,false,true,null,18.0,7.0,1.0,null,null],[43,0.0,0.0,false,true,null,16.0,8.0,1.0,null,null],[44,0.0,0.0,false,true,null,53.0,22.0,1.0,null,null],[45,0.0,0.0,false,true,null,28.0,10.0,1.0,null,null],[46,0.0,0.0,false,true,null,34.0,0.0,1.0,null,null],[47,0.0,0.0,false,true,null,20.0,11.0,1.0,null,null],[5,0.0,0.0,false,true,null,52.0,22.0,1.0,null,null],[48,0.0,0.0,false,true,null,87.0,30.0,1.0,null,null],[49,0.0,0.0,false,true,null,17.0,4.0,1.0,null,null],[50,0.0,0.0,false,true,null,17.0,8.0,1.0,null,null],[51,0.0,0.0,false,true,null,18.0,5.0,1.0,null,null],[52,0.0,0.0,false,true,null,23.0,11.0,1.0,null,null],[53,0.0,0.0,false,true,null,113.0,32.0,1.0,null,null],[54,0.0,0.0,false,true,null,63.0,22.0,1.0,null,null],[55,0.0,0.0,false,true,null,84.0,18.0,1.0,null,null],[56,0.0,0.0,false,true,null,12.0,3.0,1.0,null,null],[57,0.0,0.0,false,true,null,12.0,3.0,1.0,null,null],[6,0.0,0.0,false,true,null,19.0,2.0,1.0,null,null],[58,0.0,0.0,false,true,null,277.0,113.0,1.0,null,null],[59,0.0,0.0,false,true,null,78.0,3.0,1.0,null,null],[61,0.0,0.0,false,true,null,77.0,14.0,1.0,null,null],[65,0.0,0.0,false,true,null,39.0,18.0,1.0,null,null],[66,0.0,0.0,false,true,null,28.0,7.0,1.0,null,null],[69,0.0,0.0,false,true,null,66.0,20.0,1.0,null,null],[71,0.0,0.0,false,true,null,12.0,0.0,1.0,null,null],[72,0.0,0.0,false,true,null,6.0,0.0,1.0,null,null],[73,0.0,0.0,false,true,null,68.0,27.0,1.0,null,null],[74,0.0,0.0,false,true,null,47.0,11.0,1.0,null,null],[7,0.0,0.0,false,true,null,28.0,0.0,1.0,null,null],[75,0.0,0.0,false,true,null,68.0,36.0,1.0,null,null],[76,0.0,0.0,false,true,null,61.0,28.0,1.0,null,null],[77,0.0,0.0,false,true,null,71.0,26.0,1.0,null,null],[78,0.0,0.0,false,true,null,39.0,32.0,1.0,null,null],[79,0.0,0.0,false,true,null,130.0,26.0,1.0,null,null],[81,0.0,0.0,false,true,null,54.0,27.0,1.0,null,null],[82,0.0,0.0,false,true,null,20.0,10.0,1.0,null,null],[83,0.0,0.0,false,true,null,11.0,7.0,1.0,null,null],[84,0.0,0.0,false,true,null,24.0,15.0,1.0,null,null],[85,0.0,0.0,false,true,null,21.0,10.0,1.0,null,null],[10,0.0,0.0,false,true,null,70.0,23.0,1.0,null,null],[87,0.0,0.0,false,true,null,48.0,10.0,1.0,null,null],[89,0.0,0.0,false,true,null,163.0,42.0,1.0,null,null],[90,0.0,0.0,false,true,null,10.0,0.0,1.0,null,null],[91,0.0,0.0,false,true,null,65.0,10.0,1.0,null,null],[92,0.0,0.0,false,true,null,12.0,7.0,1.0,null,null],[93,0.0,0.0,false,true,null,30.0,16.0,1.0,null,null],[94,0.0,0.0,false,true,null,42.0,31.0,1.0,null,null],[95,0.0,0.0,false,true,null,38.0,15.0,1.0,null,null],[96,0.0,0.0,false,true,null,15.0,9.0,1.0,null,null],[97,0.0,0.0,false,true,null,34.0,8.0,1.0,null,null],[11,0.0,0.0,false,true,null,47.0,10.0,1.0,null,null],[98,0.0,0.0,false,true,null,42.0,0.0,1.0,null,null],[99,0.0,0.0,false,true,null,37.0,18.0,1.0,null,null],[100,0.0,0.0,false,true,null,22.0,15.0,1.0,null,null],[101,0.0,0.0,false,true,null,5.0,3.0,1.0,null,null],[102,0.0,0.0,false,true,null,23.0,16.0,1.0,null,null],[103,0.0,0.0,false,true,null,38.0,25.0,1.0,null,null],[104,0.0,0.0,false,true,null,31.0,26.0,1.0,null,null],[105,0.0,0.0,false,true,null,43.0,16.0,1.0,null,null],[106,0.0,0.0,false,true,null,50.0,12.0,1.0,null,null],[107,0.0,0.0,false,true,null,2.0,1.0,1.0,null,null],[12,0.0,0.0,false,true,null,34.0,16.0,1.0,null,null],[108,0.0,0.0,false,true,null,8.0,3.0,1.0,null,null],[109,0.0,0.0,false,true,null,39.0,30.0,1.0,null,null],[111,0.0,0.0,false,true,null,68.0,13.0,1.0,null,null],[112,0.0,0.0,false,true,null,6.0,0.0,1.0,null,null],[113,0.0,0.0,false,true,null,8.0,3.0,1.0,null,null],[114,0.0,0.0,false,true,null,22.0,7.0,1.0,null,null],[115,0.0,0.0,false,true,null,184.0,0.0,1.0,null,null],[116,0.0,0.0,false,true,null,20.0,8.0,1.0,null,null],[117,0.0,0.0,false,true,null,33.0,15.0,1.0,null,null]]}",
                        "orient": "split",
                        "dtype": {
                            "bus": "uint32",
                            "const_i_percent": "float64",
                            "const_z_percent": "float64",
                            "controllable": "bool",
                            "in_service": "bool",
                            "name": "object",
                            "p_mw": "float64",
                            "q_mvar": "float64",
                            "scaling": "float64",
                            "sn_mva": "float64",
                            "type": "object"
                        }
                    },
                    "sgen": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"name\",\"bus\",\"p_mw\",\"q_mvar\",\"sn_mva\",\"scaling\",\"in_service\",\"type\",\"current_source\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "name": "object",
                            "bus": "int64",
                            "p_mw": "float64",
                            "q_mvar": "float64",
                            "sn_mva": "float64",
                            "scaling": "float64",
                            "in_service": "bool",
                            "type": "object",
                            "current_source": "bool"
                        }
                    },
                    "motor": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"name\",\"bus\",\"pn_mech_mw\",\"loading_percent\",\"cos_phi\",\"cos_phi_n\",\"efficiency_percent\",\"efficiency_n_percent\",\"lrc_pu\",\"vn_kv\",\"scaling\",\"in_service\",\"rx\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "name": "object",
                            "bus": "int64",
                            "pn_mech_mw": "float64",
                            "loading_percent": "float64",
                            "cos_phi": "float64",
                            "cos_phi_n": "float64",
                            "efficiency_percent": "float64",
                            "efficiency_n_percent": "float64",
                            "lrc_pu": "float64",
                            "vn_kv": "float64",
                            "scaling": "float64",
                            "in_service": "bool",
                            "rx": "float64"
                        }
                    },
                    "asymmetric_load": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"name\",\"bus\",\"p_a_mw\",\"q_a_mvar\",\"p_b_mw\",\"q_b_mvar\",\"p_c_mw\",\"q_c_mvar\",\"sn_mva\",\"scaling\",\"in_service\",\"type\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "name": "object",
                            "bus": "uint32",
                            "p_a_mw": "float64",
                            "q_a_mvar": "float64",
                            "p_b_mw": "float64",
                            "q_b_mvar": "float64",
                            "p_c_mw": "float64",
                            "q_c_mvar": "float64",
                            "sn_mva": "float64",
                            "scaling": "float64",
                            "in_service": "bool",
                            "type": "object"
                        }
                    },
                    "asymmetric_sgen": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"name\",\"bus\",\"p_a_mw\",\"q_a_mvar\",\"p_b_mw\",\"q_b_mvar\",\"p_c_mw\",\"q_c_mvar\",\"sn_mva\",\"scaling\",\"in_service\",\"type\",\"current_source\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "name": "object",
                            "bus": "int64",
                            "p_a_mw": "float64",
                            "q_a_mvar": "float64",
                            "p_b_mw": "float64",
                            "q_b_mvar": "float64",
                            "p_c_mw": "float64",
                            "q_c_mvar": "float64",
                            "sn_mva": "float64",
                            "scaling": "float64",
                            "in_service": "bool",
                            "type": "object",
                            "current_source": "bool"
                        }
                    },
                    "storage": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"name\",\"bus\",\"p_mw\",\"q_mvar\",\"sn_mva\",\"soc_percent\",\"min_e_mwh\",\"max_e_mwh\",\"scaling\",\"in_service\",\"type\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "name": "object",
                            "bus": "int64",
                            "p_mw": "float64",
                            "q_mvar": "float64",
                            "sn_mva": "float64",
                            "soc_percent": "float64",
                            "min_e_mwh": "float64",
                            "max_e_mwh": "float64",
                            "scaling": "float64",
                            "in_service": "bool",
                            "type": "object"
                        }
                    },
                    "gen": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"bus\",\"controllable\",\"in_service\",\"name\",\"p_mw\",\"scaling\",\"sn_mva\",\"type\",\"vm_pu\",\"slack\",\"max_p_mw\",\"min_p_mw\",\"max_q_mvar\",\"min_q_mvar\"],\"index\":[0,1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,42,43,44,45,46,47,48,49,5,50,51,52,6,7,8,9],\"data\":[[0,true,true,null,0.0,1.0,null,null,0.955,false,100.0,0.0,15.0,-5.0],[3,true,true,null,0.0,1.0,null,null,0.998,false,100.0,0.0,300.0,-300.0],[24,true,true,null,220.0,1.0,null,null,1.05,false,320.0,0.0,140.0,-47.0],[25,true,true,null,314.0,1.0,null,null,1.015,false,414.0,0.0,1000.0,-1000.0],[26,true,true,null,0.0,1.0,null,null,0.968,false,100.0,0.0,300.0,-300.0],[30,true,true,null,7.0,1.0,null,null,0.967,false,107.0,0.0,300.0,-300.0],[31,true,true,null,0.0,1.0,null,null,0.963,false,100.0,0.0,42.0,-14.0],[33,true,true,null,0.0,1.0,null,null,0.984,false,100.0,0.0,24.0,-8.0],[35,true,true,null,0.0,1.0,null,null,0.98,false,100.0,0.0,24.0,-8.0],[39,true,true,null,0.0,1.0,null,null,0.97,false,100.0,0.0,300.0,-300.0],[41,true,true,null,0.0,1.0,null,null,0.985,false,100.0,0.0,300.0,-300.0],[45,true,true,null,19.0,1.0,null,null,1.005,false,119.0,0.0,100.0,-100.0],[5,true,true,null,0.0,1.0,null,null,0.99,false,100.0,0.0,50.0,-13.0],[48,true,true,null,204.0,1.0,null,null,1.025,false,304.0,0.0,210.0,-85.0],[53,true,true,null,48.0,1.0,null,null,0.955,false,148.0,0.0,300.0,-300.0],[54,true,true,null,0.0,1.0,null,null,0.952,false,100.0,0.0,23.0,-8.0],[55,true,true,null,0.0,1.0,null,null,0.954,false,100.0,0.0,15.0,-8.0],[58,true,true,null,155.0,1.0,null,null,0.985,false,255.0,0.0,180.0,-60.0],[60,true,true,null,160.0,1.0,null,null,0.995,false,260.0,0.0,300.0,-100.0],[61,true,true,null,0.0,1.0,null,null,0.998,false,100.0,0.0,20.0,-20.0],[64,true,true,null,391.0,1.0,null,null,1.005,false,491.0,0.0,200.0,-67.0],[65,true,true,null,392.0,1.0,null,null,1.05,false,492.0,0.0,200.0,-67.0],[69,true,true,null,0.0,1.0,null,null,0.984,false,100.0,0.0,32.0,-10.0],[7,true,true,null,0.0,1.0,null,null,1.015,false,100.0,0.0,300.0,-300.0],[71,true,true,null,0.0,1.0,null,null,0.98,false,100.0,0.0,100.0,-100.0],[72,true,true,null,0.0,1.0,null,null,0.991,false,100.0,0.0,100.0,-100.0],[73,true,true,null,0.0,1.0,null,null,0.958,false,100.0,0.0,9.0,-6.0],[75,true,true,null,0.0,1.0,null,null,0.943,false,100.0,0.0,23.0,-8.0],[76,true,true,null,0.0,1.0,null,null,1.006,false,100.0,0.0,70.0,-20.0],[79,true,true,null,477.0,1.0,null,null,1.04,false,577.0,0.0,280.0,-165.0],[84,true,true,null,0.0,1.0,null,null,0.985,false,100.0,0.0,23.0,-8.0],[86,true,true,null,4.0,1.0,null,null,1.015,false,104.0,0.0,1000.0,-100.0],[88,true,true,null,607.0,1.0,null,null,1.005,false,707.0,0.0,300.0,-210.0],[89,true,true,null,0.0,1.0,null,null,0.985,false,100.0,0.0,300.0,-300.0],[9,true,true,null,450.0,1.0,null,null,1.05,false,550.0,0.0,200.0,-147.0],[90,true,true,null,0.0,1.0,null,null,0.98,false,100.0,0.0,100.0,-100.0],[91,true,true,null,0.0,1.0,null,null,0.99,false,100.0,0.0,9.0,-3.0],[98,true,true,null,0.0,1.0,null,null,1.01,false,100.0,0.0,100.0,-100.0],[99,true,true,null,252.0,1.0,null,null,1.017,false,352.0,0.0,155.0,-50.0],[102,true,true,null,40.0,1.0,null,null,1.01,false,140.0,0.0,40.0,-15.0],[103,true,true,null,0.0,1.0,null,null,0.971,false,100.0,0.0,23.0,-8.0],[104,true,true,null,0.0,1.0,null,null,0.965,false,100.0,0.0,23.0,-8.0],[106,true,true,null,0.0,1.0,null,null,0.952,false,100.0,0.0,200.0,-200.0],[109,true,true,null,0.0,1.0,null,null,0.973,false,100.0,0.0,23.0,-8.0],[110,true,true,null,36.0,1.0,null,null,0.98,false,136.0,0.0,1000.0,-100.0],[11,true,true,null,85.0,1.0,null,null,0.99,false,185.0,0.0,120.0,-35.0],[111,true,true,null,0.0,1.0,null,null,0.975,false,100.0,0.0,1000.0,-100.0],[112,true,true,null,0.0,1.0,null,null,0.993,false,100.0,0.0,200.0,-100.0],[115,true,true,null,0.0,1.0,null,null,1.005,false,100.0,0.0,1000.0,-1000.0],[14,true,true,null,0.0,1.0,null,null,0.97,false,100.0,0.0,30.0,-10.0],[17,true,true,null,0.0,1.0,null,null,0.973,false,100.0,0.0,50.0,-16.0],[18,true,true,null,0.0,1.0,null,null,0.962,false,100.0,0.0,24.0,-8.0],[23,true,true,null,0.0,1.0,null,null,0.992,false,100.0,0.0,300.0,-300.0]]}",
                        "orient": "split",
                        "dtype": {
                            "bus": "uint32",
                            "controllable": "bool",
                            "in_service": "bool",
                            "name": "object",
                            "p_mw": "float64",
                            "scaling": "float64",
                            "sn_mva": "float64",
                            "type": "object",
                            "vm_pu": "float64",
                            "slack": "bool",
                            "max_p_mw": "float64",
                            "min_p_mw": "float64",
                            "max_q_mvar": "float64",
                            "min_q_mvar": "float64"
                        }
                    },
                    "switch": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"bus\",\"element\",\"et\",\"type\",\"closed\",\"name\",\"z_ohm\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "bus": "int64",
                            "element": "int64",
                            "et": "object",
                            "type": "object",
                            "closed": "bool",
                            "name": "object",
                            "z_ohm": "float64"
                        }
                    },
                    "shunt": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"bus\",\"name\",\"q_mvar\",\"p_mw\",\"vn_kv\",\"step\",\"max_step\",\"in_service\"],\"index\":[0,1,10,11,12,13,2,3,4,5,6,7,8,9],\"data\":[[4,null,40.0,0.0,138.0,1,1,true],[33,null,-14.0,0.0,138.0,1,1,true],[82,null,-10.0,0.0,138.0,1,1,true],[104,null,-20.0,0.0,138.0,1,1,true],[106,null,-6.0,0.0,138.0,1,1,true],[109,null,-6.0,0.0,138.0,1,1,true],[36,null,25.0,0.0,138.0,1,1,true],[43,null,-10.0,0.0,138.0,1,1,true],[44,null,-10.0,0.0,138.0,1,1,true],[45,null,-10.0,0.0,138.0,1,1,true],[47,null,-15.0,0.0,138.0,1,1,true],[73,null,-12.0,0.0,138.0,1,1,true],[78,null,-20.0,0.0,138.0,1,1,true],[81,null,-20.0,0.0,138.0,1,1,true]]}",
                        "orient": "split",
                        "dtype": {
                            "bus": "uint32",
                            "name": "object",
                            "q_mvar": "float64",
                            "p_mw": "float64",
                            "vn_kv": "float64",
                            "step": "uint32",
                            "max_step": "uint32",
                            "in_service": "bool"
                        }
                    },
                    "ext_grid": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"bus\",\"in_service\",\"name\",\"va_degree\",\"vm_pu\",\"max_p_mw\",\"min_p_mw\",\"max_q_mvar\",\"min_q_mvar\"],\"index\":[0],\"data\":[[68,true,null,30.0,1.035,805.200000000000045,0.0,300.0,-300.0]]}",
                        "orient": "split",
                        "dtype": {
                            "bus": "uint32",
                            "in_service": "bool",
                            "name": "object",
                            "va_degree": "float64",
                            "vm_pu": "float64",
                            "max_p_mw": "float64",
                            "min_p_mw": "float64",
                            "max_q_mvar": "float64",
                            "min_q_mvar": "float64"
                        }
                    },
                    "line": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"c_nf_per_km\",\"df\",\"from_bus\",\"g_us_per_km\",\"in_service\",\"length_km\",\"max_i_ka\",\"max_loading_percent\",\"name\",\"parallel\",\"r_ohm_per_km\",\"std_type\",\"to_bus\",\"type\",\"x_ohm_per_km\"],\"index\":[0,1,10,100,101,102,103,104,105,106,107,108,109,11,110,111,112,113,114,115,116,117,118,119,12,120,121,122,123,124,125,126,127,128,129,13,130,131,132,133,134,135,136,137,138,139,14,140,141,142,143,144,145,146,147,148,149,15,150,151,152,153,154,155,156,157,158,159,16,160,161,162,163,164,165,166,167,168,169,17,170,171,172,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,42,43,44,45,46,47,48,49,5,50,51,52,53,54,55,56,57,58,59,6,60,61,62,63,64,65,66,67,68,69,7,70,71,72,73,74,75,76,77,78,79,8,80,81,82,83,84,85,86,87,88,89,9,90,91,92,93,94,95,96,97,98,99],\"data\":[[353.789080947117327,1.0,0,0.0,true,1.0,41.418606267951418,100.0,null,1,5.770332,null,1,\"ol\",19.024956],[150.708577001882276,1.0,0,0.0,true,1.0,41.418606267951418,100.0,null,1,2.456676,null,2,\"ol\",8.074655999999999],[69.922093950965717,1.0,10,0.0,true,1.0,41.418606267951418,100.0,null,1,1.133118,null,11,\"ol\",3.732624],[122.294020894318521,1.0,69,0.0,true,1.0,41.418606267951418,100.0,null,1,1.6796808,null,70,\"ol\",6.76062],[679.720753945642855,1.0,23,0.0,true,1.0,41.418606267951418,100.0,null,1,9.293472,null,71,\"ol\",37.326239999999998],[618.991604617712483,1.0,70,0.0,true,1.0,41.418606267951418,100.0,null,1,8.493624,null,71,\"ol\",34.279200000000003],[164.080132817206419,1.0,70,0.0,true,1.0,41.418606267951418,100.0,null,1,1.6492104,null,72,\"ol\",8.645975999999999],[469.118749854287955,1.0,69,0.0,true,1.0,41.418606267951418,100.0,null,1,7.636644,null,73,\"ol\",25.195212000000002],[501.433343074654545,1.0,69,0.0,true,1.0,41.418606267951418,100.0,null,1,8.150831999999999,null,74,\"ol\",26.852039999999999],[1727.15929281269905,1.0,68,0.0,true,1.0,41.418606267951418,100.0,null,1,7.71282,null,74,\"ol\",23.23368],[144.022799094220204,1.0,73,0.0,true,1.0,41.418606267951418,100.0,null,1,2.342412,null,74,\"ol\",7.731864],[512.576306254091378,1.0,75,0.0,true,1.0,41.418606267951418,100.0,null,1,8.455536,null,76,\"ol\",28.185120000000001],[1445.799472531920628,1.0,68,0.0,true,1.0,41.418606267951418,100.0,null,1,5.884596,null,76,\"ol\",19.234439999999999],[218.959226475932468,1.0,1,0.0,true,1.0,41.418606267951418,100.0,null,1,3.561228,null,11,\"ol\",11.731104],[693.370883840452848,1.0,74,0.0,true,1.0,41.418606267951418,100.0,null,1,11.445444,null,76,\"ol\",38.068956],[176.058818235100944,1.0,76,0.0,true,1.0,41.418606267951418,100.0,null,1,0.7160544,null,77,\"ol\",2.361456],[90.25800175343781,1.0,77,0.0,true,1.0,41.418606267951418,100.0,null,1,1.0398024,null,78,\"ol\",4.646736],[657.434827586769188,1.0,76,0.0,true,1.0,41.418606267951418,100.0,null,1,3.23748,null,79,\"ol\",9.23634],[317.574450613947931,1.0,76,0.0,true,1.0,41.418606267951418,100.0,null,1,5.598936,null,79,\"ol\",19.996200000000002],[260.466764319334459,1.0,78,0.0,true,1.0,41.418606267951418,100.0,null,1,2.970864,null,79,\"ol\",13.406976],[1138.532262858951754,1.0,76,0.0,true,1.0,41.418606267951418,100.0,null,1,5.675112,null,81,\"ol\",16.244532],[528.733602864274644,1.0,81,0.0,true,1.0,41.418606267951418,100.0,null,1,2.132928,null,82,\"ol\",6.979626],[359.360562536835744,1.0,82,0.0,true,1.0,41.418606267951418,100.0,null,1,11.9025,null,83,\"ol\",25.138079999999999],[484.718898305499408,1.0,82,0.0,true,1.0,41.418606267951418,100.0,null,1,8.18892,null,84,\"ol\",28.185120000000001],[565.50538135641591,1.0,2,0.0,true,1.0,41.418606267951418,100.0,null,1,9.217295999999999,null,11,\"ol\",30.470400000000002],[171.880207042812117,1.0,83,0.0,true,1.0,41.418606267951418,100.0,null,1,5.751288,null,84,\"ol\",12.207204000000001],[384.432229690568477,1.0,84,0.0,true,1.0,41.418606267951418,100.0,null,1,6.6654,null,85,\"ol\",23.424119999999998],[384.432229690568477,1.0,84,0.0,true,1.0,41.418606267951418,100.0,null,1,3.8088,null,87,\"ol\",19.424880000000002],[654.649086791910122,1.0,84,0.0,true,1.0,41.418606267951418,100.0,null,1,4.551516,null,88,\"ol\",32.94612],[269.381134862883869,1.0,87,0.0,true,1.0,41.418606267951418,100.0,null,1,2.647116,null,88,\"ol\",13.559328000000001],[735.435569842826681,1.0,88,0.0,true,1.0,41.418606267951418,100.0,null,1,9.864792,null,89,\"ol\",35.802720000000001],[1476.442621275371494,1.0,88,0.0,true,1.0,41.418606267951418,100.0,null,1,4.532472,null,89,\"ol\",18.986868000000001],[298.074265049933501,1.0,89,0.0,true,1.0,41.418606267951418,100.0,null,1,4.837176,null,90,\"ol\",15.920783999999999],[763.292977791418593,1.0,88,0.0,true,1.0,41.418606267951418,100.0,null,1,1.885356,null,91,\"ol\",9.61722],[576.648344535852743,1.0,88,0.0,true,1.0,41.418606267951418,100.0,null,1,7.484292,null,91,\"ol\",30.108564000000001],[121.736872735346694,1.0,6,0.0,true,1.0,41.418606267951418,100.0,null,1,1.6415928,null,11,\"ol\",6.47496],[455.190045879991942,1.0,90,0.0,true,1.0,41.418606267951418,100.0,null,1,7.370028,null,91,\"ol\",24.223967999999999],[303.645746639651918,1.0,91,0.0,true,1.0,41.418606267951418,100.0,null,1,4.913352,null,92,\"ol\",16.149311999999998],[565.50538135641591,1.0,91,0.0,true,1.0,41.418606267951418,100.0,null,1,9.160164,null,93,\"ol\",30.08952],[261.302486557792236,1.0,92,0.0,true,1.0,41.418606267951418,100.0,null,1,4.246812,null,93,\"ol\",13.940208],[154.608614114685167,1.0,93,0.0,true,1.0,41.418606267951418,100.0,null,1,2.513808,null,94,\"ol\",8.265096],[688.077976330220395,1.0,79,0.0,true,1.0,41.418606267951418,100.0,null,1,6.779664,null,95,\"ol\",34.660080000000001],[757.72149620170012,1.0,81,0.0,true,1.0,41.418606267951418,100.0,null,1,3.085128,null,95,\"ol\",10.09332],[320.360191408807054,1.0,93,0.0,true,1.0,41.418606267951418,100.0,null,1,5.122836,null,95,\"ol\",16.549236],[353.789080947117327,1.0,79,0.0,true,1.0,41.418606267951418,100.0,null,1,3.485052,null,96,\"ol\",17.787095999999998],[398.360933664864433,1.0,79,0.0,true,1.0,41.418606267951418,100.0,null,1,4.532472,null,97,\"ol\",20.567519999999998],[261.302486557792236,1.0,10,0.0,true,1.0,41.418606267951418,100.0,null,1,4.23729,null,12,\"ol\",13.921163999999999],[760.507236996559413,1.0,79,0.0,true,1.0,41.418606267951418,100.0,null,1,8.645975999999999,null,98,\"ol\",39.230639999999987],[657.434827586769188,1.0,91,0.0,true,1.0,41.418606267951418,100.0,null,1,12.340512,null,99,\"ol\",56.179799999999993],[841.293720047475972,1.0,93,0.0,true,1.0,41.418606267951418,100.0,null,1,3.389832,null,99,\"ol\",11.04552],[205.309096581122446,1.0,94,0.0,true,1.0,41.418606267951418,100.0,null,1,3.256524,null,95,\"ol\",10.417068],[334.288895383103011,1.0,95,0.0,true,1.0,41.418606267951418,100.0,null,1,3.294612,null,96,\"ol\",16.853940000000002],[663.006309176487889,1.0,97,0.0,true,1.0,41.418606267951418,100.0,null,1,7.560468,null,99,\"ol\",34.088760000000001],[300.860005844792738,1.0,98,0.0,true,1.0,41.418606267951418,100.0,null,1,3.42792,null,99,\"ol\",15.482772000000001],[456.861490356907552,1.0,99,0.0,true,1.0,41.418606267951418,100.0,null,1,5.275188,null,100,\"ol\",24.033528],[203.916226183692856,1.0,91,0.0,true,1.0,41.418606267951418,100.0,null,1,2.342412,null,101,\"ol\",10.645595999999999],[409.50389684430121,1.0,100,0.0,true,1.0,41.418606267951418,100.0,null,1,4.684824,null,101,\"ol\",21.329280000000001],[252.94526417321461,1.0,11,0.0,true,1.0,41.418606267951418,100.0,null,1,4.09446,null,13,\"ol\",13.464108],[746.578533022263514,1.0,99,0.0,true,1.0,41.418606267951418,100.0,null,1,3.04704,null,102,\"ol\",9.998100000000001],[753.54288500941152,1.0,99,0.0,true,1.0,41.418606267951418,100.0,null,1,8.588844,null,103,\"ol\",38.849760000000003],[566.898251753845557,1.0,102,0.0,true,1.0,41.418606267951418,100.0,null,1,8.874504,null,103,\"ol\",30.165696],[568.291122151275204,1.0,102,0.0,true,1.0,41.418606267951418,100.0,null,1,10.18854,null,104,\"ol\",30.9465],[863.579646406349525,1.0,99,0.0,true,1.0,41.418606267951418,100.0,null,1,11.52162,null,105,\"ol\",43.610759999999999],[137.337021186558161,1.0,103,0.0,true,1.0,41.418606267951418,100.0,null,1,1.8929736,null,104,\"ol\",7.198632],[199.737614991404058,1.0,104,0.0,true,1.0,41.418606267951418,100.0,null,1,2.66616,null,105,\"ol\",10.417068],[657.434827586769188,1.0,104,0.0,true,1.0,41.418606267951418,100.0,null,1,10.09332,null,106,\"ol\",34.850520000000003],[256.845301286017502,1.0,104,0.0,true,1.0,41.418606267951418,100.0,null,1,4.970484,null,107,\"ol\",13.387931999999999],[657.434827586769188,1.0,105,0.0,true,1.0,41.418606267951418,100.0,null,1,10.09332,null,106,\"ol\",34.850520000000003],[873.051165108870691,1.0,12,0.0,true,1.0,41.418606267951418,100.0,null,1,14.168736000000001,null,14,\"ol\",46.543536000000003],[105.858150204649292,1.0,107,0.0,true,1.0,41.418606267951418,100.0,null,1,1.99962,null,108,\"ol\",5.484672],[642.113253215043756,1.0,102,0.0,true,1.0,41.418606267951418,100.0,null,1,7.4385864,null,109,\"ol\",34.526771999999987],[281.359820280778365,1.0,108,0.0,true,1.0,41.418606267951418,100.0,null,1,5.294232,null,109,\"ol\",14.511528],[278.574079485919185,1.0,109,0.0,true,1.0,41.418606267951418,100.0,null,1,4.18968,null,110,\"ol\",14.378220000000001],[863.579646406349525,1.0,109,0.0,true,1.0,41.418606267951418,100.0,null,1,4.703868,null,111,\"ol\",12.18816],[106.972446522592961,1.0,16,0.0,true,1.0,41.418606267951418,100.0,null,1,1.7387172,null,112,\"ol\",5.732244],[721.506865868530667,1.0,31,0.0,true,1.0,41.418606267951418,100.0,null,1,11.712059999999999,null,112,\"ol\",38.659320000000001],[226.759300701538194,1.0,31,0.0,true,1.0,41.418606267951418,100.0,null,1,2.57094,null,113,\"ol\",11.654928],[274.674042373116322,1.0,26,0.0,true,1.0,41.418606267951418,100.0,null,1,3.123216,null,114,\"ol\",14.111604],[38.443222969056848,1.0,113,0.0,true,1.0,41.418606267951418,100.0,null,1,0.438012,null,114,\"ol\",1.980576],[699.220939509657114,1.0,13,0.0,true,1.0,41.418606267951418,100.0,null,1,11.33118,null,14,\"ol\",37.135800000000003],[498.647602279795365,1.0,11,0.0,true,1.0,41.418606267951418,100.0,null,1,6.265476,null,116,\"ol\",26.6616],[166.86587361206557,1.0,74,0.0,true,1.0,41.418606267951418,100.0,null,1,2.76138,null,117,\"ol\",9.160164],[188.873225891453217,1.0,75,0.0,true,1.0,41.418606267951418,100.0,null,1,3.123216,null,117,\"ol\",10.359935999999999],[298.074265049933501,1.0,11,0.0,true,1.0,41.418606267951418,100.0,null,1,4.037328,null,15,\"ol\",15.882695999999999],[618.434456458740669,1.0,14,0.0,true,1.0,41.418606267951418,100.0,null,1,2.513808,null,16,\"ol\",8.322228000000001],[29.25027834602151,1.0,3,0.0,true,1.0,41.418606267951418,100.0,null,1,0.3351744,null,4,\"ol\",1.5197112],[649.077605202191762,1.0,15,0.0,true,1.0,41.418606267951418,100.0,null,1,8.645975999999999,null,16,\"ol\",34.298243999999997],[180.794577586361555,1.0,16,0.0,true,1.0,41.418606267951418,100.0,null,1,2.342412,null,17,\"ol\",9.61722],[159.065799386459844,1.0,17,0.0,true,1.0,41.418606267951418,100.0,null,1,2.1310236,null,18,\"ol\",9.388692000000001],[415.075378434019626,1.0,18,0.0,true,1.0,41.418606267951418,100.0,null,1,4.799088,null,19,\"ol\",22.281479999999998],[140.679910140389182,1.0,14,0.0,true,1.0,41.418606267951418,100.0,null,1,2.28528,null,18,\"ol\",7.503336],[300.860005844792738,1.0,19,0.0,true,1.0,41.418606267951418,100.0,null,1,3.485052,null,20,\"ol\",16.168355999999999],[342.646117767680607,1.0,20,0.0,true,1.0,41.418606267951418,100.0,null,1,3.980196,null,21,\"ol\",18.47268],[562.71964056155673,1.0,21,0.0,true,1.0,41.418606267951418,100.0,null,1,6.513048,null,22,\"ol\",30.279959999999999],[693.649457919938868,1.0,22,0.0,true,1.0,41.418606267951418,100.0,null,1,2.57094,null,23,\"ol\",9.369648],[1203.440023379170952,1.0,22,0.0,true,1.0,41.418606267951418,100.0,null,1,2.970864,null,24,\"ol\",15.235200000000001],[395.57519287000531,1.0,2,0.0,true,1.0,41.418606267951418,100.0,null,1,4.589604,null,4,\"ol\",20.567519999999998],[2457.023381065807371,1.0,24,0.0,true,1.0,41.418606267951418,100.0,null,1,6.055992,null,26,\"ol\",31.041720000000002],[300.860005844792738,1.0,26,0.0,true,1.0,41.418606267951418,100.0,null,1,3.6431172,null,27,\"ol\",16.282620000000001],[331.503154588243945,1.0,27,0.0,true,1.0,41.418606267951418,100.0,null,1,4.513428,null,28,\"ol\",17.958492],[1145.496614846099646,1.0,7,0.0,true,1.0,16.567442507180569,100.0,null,1,5.1299775,null,29,\"ol\",59.988599999999998],[2023.562113385716884,1.0,25,0.0,true,1.0,16.567442507180569,100.0,null,1,9.510097500000001,null,29,\"ol\",102.361499999999992],[555.755288574408723,1.0,16,0.0,true,1.0,41.418606267951418,100.0,null,1,9.026856,null,30,\"ol\",29.765771999999998],[115.608242986656464,1.0,28,0.0,true,1.0,41.418606267951418,100.0,null,1,2.056752,null,30,\"ol\",6.303564],[1633.836976184916011,1.0,22,0.0,true,1.0,41.418606267951418,100.0,null,1,6.036948,null,31,\"ol\",21.957732],[349.610469754828557,1.0,30,0.0,true,1.0,41.418606267951418,100.0,null,1,5.675112,null,31,\"ol\",18.75834],[268.266838544940128,1.0,26,0.0,true,1.0,41.418606267951418,100.0,null,1,4.361076,null,31,\"ol\",14.378220000000001],[198.623318673460403,1.0,4,0.0,true,1.0,41.418606267951418,100.0,null,1,2.266236,null,5,\"ol\",10.283759999999999],[444.882804939012999,1.0,14,0.0,true,1.0,41.418606267951418,100.0,null,1,7.23672,null,32,\"ol\",23.690736000000001],[880.294091175504832,1.0,18,0.0,true,1.0,41.418606267951418,100.0,null,1,14.321088,null,33,\"ol\",47.038679999999999],[37.328926651113171,1.0,34,0.0,true,1.0,41.418606267951418,100.0,null,1,0.4265856,null,35,\"ol\",1.942488],[183.580318381220764,1.0,34,0.0,true,1.0,41.418606267951418,100.0,null,1,2.09484,null,36,\"ol\",9.464867999999999],[509.790565459232084,1.0,32,0.0,true,1.0,41.418606267951418,100.0,null,1,7.90326,null,36,\"ol\",27.042480000000001],[79.115038574001062,1.0,33,0.0,true,1.0,41.418606267951418,100.0,null,1,1.6587324,null,35,\"ol\",5.103792],[137.058447107072226,1.0,33,0.0,true,1.0,41.418606267951418,100.0,null,1,0.4875264,null,36,\"ol\",1.790136],[376.07500730599088,1.0,36,0.0,true,1.0,41.418606267951418,100.0,null,1,6.113124,null,38,\"ol\",20.186640000000001],[585.005566920430397,1.0,36,0.0,true,1.0,41.418606267951418,100.0,null,1,11.293092,null,39,\"ol\",31.993919999999999],[940.466092344463164,1.0,29,0.0,true,1.0,16.567442507180569,100.0,null,1,5.52276,null,37,\"ol\",64.273499999999999],[76.607871858627774,1.0,5,0.0,true,1.0,41.418606267951418,100.0,null,1,0.8741196,null,6,\"ol\",3.961152],[216.173485681073316,1.0,38,0.0,true,1.0,41.418606267951418,100.0,null,1,3.504096,null,39,\"ol\",11.52162],[170.208762565896649,1.0,39,0.0,true,1.0,41.418606267951418,100.0,null,1,2.76138,null,40,\"ol\",9.274428],[649.077605202191762,1.0,39,0.0,true,1.0,41.418606267951418,100.0,null,1,10.569419999999999,null,41,\"ol\",34.850520000000003],[479.147416715780992,1.0,40,0.0,true,1.0,41.418606267951418,100.0,null,1,7.80804,null,41,\"ol\",25.709399999999999],[845.193757160278892,1.0,42,0.0,true,1.0,41.418606267951418,100.0,null,1,11.578752,null,43,\"ol\",46.733975999999998],[588.627029953747296,1.0,33,0.0,true,1.0,41.418606267951418,100.0,null,1,7.865172,null,42,\"ol\",32.012963999999997],[312.002969024229515,1.0,43,0.0,true,1.0,41.418606267951418,100.0,null,1,4.265856,null,44,\"ol\",17.158643999999999],[462.432971946625855,1.0,44,0.0,true,1.0,41.418606267951418,100.0,null,1,7.6176,null,45,\"ol\",25.823664000000001],[440.147045587752416,1.0,45,0.0,true,1.0,41.418606267951418,100.0,null,1,7.23672,null,46,\"ol\",24.185880000000001],[657.434827586769188,1.0,45,0.0,true,1.0,41.418606267951418,100.0,null,1,11.445444,null,47,\"ol\",35.993160000000003],[2589.62464290110438,1.0,7,0.0,true,1.0,16.567442507180569,100.0,null,1,2.90421,null,8,\"ol\",36.302624999999999],[223.416411747707201,1.0,46,0.0,true,1.0,41.418606267951418,100.0,null,1,3.637404,null,48,\"ol\",11.9025],[1197.868541789452365,1.0,41,0.0,true,1.0,41.418606267951418,100.0,null,1,13.61646,null,48,\"ol\",61.512120000000003],[1197.868541789452365,1.0,41,0.0,true,1.0,41.418606267951418,100.0,null,1,13.61646,null,48,\"ol\",61.512120000000003],[618.434456458740669,1.0,44,0.0,true,1.0,41.418606267951418,100.0,null,1,13.026096000000001,null,48,\"ol\",35.421840000000003],[175.223095996643138,1.0,47,0.0,true,1.0,41.418606267951418,100.0,null,1,3.408876,null,48,\"ol\",9.61722],[261.023912478306272,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,5.084748,null,49,\"ol\",14.321088],[476.361675920921868,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,9.255383999999999,null,50,\"ol\",26.09028],[194.444707481171577,1.0,50,0.0,true,1.0,41.418606267951418,100.0,null,1,3.865932,null,51,\"ol\",11.197872],[565.226807276930117,1.0,51,0.0,true,1.0,41.418606267951418,100.0,null,1,7.71282,null,52,\"ol\",31.136939999999999],[431.789823203174762,1.0,52,0.0,true,1.0,41.418606267951418,100.0,null,1,5.008572,null,53,\"ol\",23.23368],[2741.168942141444859,1.0,8,0.0,true,1.0,16.567442507180569,100.0,null,1,3.070845,null,9,\"ol\",38.326050000000002],[1027.93835330304205,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,13.90212,null,53,\"ol\",55.037159999999993],[1016.795390123604875,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,16.549236,null,53,\"ol\",55.418039999999998],[281.359820280778365,1.0,53,0.0,true,1.0,41.418606267951418,100.0,null,1,3.218436,null,54,\"ol\",13.464108],[101.958113091846428,1.0,53,0.0,true,1.0,41.418606267951418,100.0,null,1,0.52371,null,55,\"ol\",1.818702],[52.09335286386689,1.0,54,0.0,true,1.0,41.418606267951418,100.0,null,1,0.9293472,null,55,\"ol\",2.875644],[337.074636177962248,1.0,55,0.0,true,1.0,41.418606267951418,100.0,null,1,6.532092,null,56,\"ol\",18.396504],[462.432971946625855,1.0,49,0.0,true,1.0,41.418606267951418,100.0,null,1,9.026856,null,56,\"ol\",25.51896],[337.074636177962248,1.0,55,0.0,true,1.0,41.418606267951418,100.0,null,1,6.532092,null,57,\"ol\",18.396504],[249.045227060411776,1.0,50,0.0,true,1.0,41.418606267951418,100.0,null,1,4.85622,null,57,\"ol\",13.692636],[832.936497662898319,1.0,53,0.0,true,1.0,41.418606267951418,100.0,null,1,9.579132,null,58,\"ol\",43.667892000000002],[243.473745470693387,1.0,3,0.0,true,1.0,41.418606267951418,100.0,null,1,3.980196,null,10,\"ol\",13.102271999999999],[792.543256137440153,1.0,55,0.0,true,1.0,41.418606267951418,100.0,null,1,15.7113,null,58,\"ol\",47.800440000000002],[746.578533022263514,1.0,55,0.0,true,1.0,41.418606267951418,100.0,null,1,15.292332,null,58,\"ol\",45.515159999999987],[786.41462638874998,1.0,54,0.0,true,1.0,41.418606267951418,100.0,null,1,9.0249516,null,58,\"ol\",41.096951999999988],[523.719269433528098,1.0,58,0.0,true,1.0,41.418606267951418,100.0,null,1,6.036948,null,59,\"ol\",27.613800000000001],[540.433714202683177,1.0,58,0.0,true,1.0,41.418606267951418,100.0,null,1,6.246432,null,60,\"ol\",28.565999999999999],[202.801929865749173,1.0,59,0.0,true,1.0,41.418606267951418,100.0,null,1,0.5027616,null,60,\"ol\",2.57094],[204.473374342664698,1.0,59,0.0,true,1.0,41.418606267951418,100.0,null,1,2.342412,null,61,\"ol\",10.683684],[136.501298948100384,1.0,60,0.0,true,1.0,41.418606267951418,100.0,null,1,1.5692256,null,61,\"ol\",7.160544],[481.376009351668358,1.0,62,0.0,true,1.0,16.567442507180569,100.0,null,1,2.04723,null,63,\"ol\",23.805],[2331.10789713817212,1.0,37,0.0,true,1.0,16.567442507180569,100.0,null,1,10.724152500000001,null,64,\"ol\",117.358649999999997],[242.080875073263741,1.0,4,0.0,true,1.0,41.418606267951418,100.0,null,1,3.865932,null,10,\"ol\",12.988008000000001],[846.865201637194332,1.0,63,0.0,true,1.0,16.567442507180569,100.0,null,1,3.2017725,null,64,\"ol\",35.945549999999997],[345.431858562539844,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,3.42792,null,65,\"ol\",17.501436000000002],[345.431858562539844,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,3.42792,null,65,\"ol\",17.501436000000002],[805.079089714306519,1.0,61,0.0,true,1.0,41.418606267951418,100.0,null,1,9.179207999999999,null,65,\"ol\",41.515920000000001],[431.789823203174762,1.0,61,0.0,true,1.0,41.418606267951418,100.0,null,1,4.913352,null,66,\"ol\",22.281479999999998],[373.567840590617664,1.0,65,0.0,true,1.0,41.418606267951418,100.0,null,1,4.265856,null,66,\"ol\",19.32966],[987.823685857069336,1.0,46,0.0,true,1.0,41.418606267951418,100.0,null,1,16.073136000000002,null,68,\"ol\",52.904232],[1153.296689071705487,1.0,48,0.0,true,1.0,41.418606267951418,100.0,null,1,18.75834,null,68,\"ol\",61.702559999999998],[1699.301884864107024,1.0,68,0.0,true,1.0,41.418606267951418,100.0,null,1,5.7132,null,69,\"ol\",24.185880000000001],[1420.449231298701989,1.0,23,0.0,true,1.0,41.418606267951418,100.0,null,1,0.4208724,null,69,\"ol\",78.36605999999999]]}",
                        "orient": "split",
                        "dtype": {
                            "c_nf_per_km": "float64",
                            "df": "float64",
                            "from_bus": "uint32",
                            "g_us_per_km": "float64",
                            "in_service": "bool",
                            "length_km": "float64",
                            "max_i_ka": "float64",
                            "max_loading_percent": "float64",
                            "name": "object",
                            "parallel": "uint32",
                            "r_ohm_per_km": "float64",
                            "std_type": "object",
                            "to_bus": "uint32",
                            "type": "object",
                            "x_ohm_per_km": "float64"
                        }
                    },
                    "trafo": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"df\",\"hv_bus\",\"i0_percent\",\"in_service\",\"lv_bus\",\"max_loading_percent\",\"name\",\"parallel\",\"pfe_kw\",\"shift_degree\",\"sn_mva\",\"std_type\",\"tap_max\",\"tap_neutral\",\"tap_min\",\"tap_phase_shifter\",\"tap_pos\",\"tap_side\",\"tap_step_degree\",\"tap_step_percent\",\"vn_hv_kv\",\"vn_lv_kv\",\"vk_percent\",\"vkr_percent\"],\"index\":[0,1,10,11,12,2,3,4,5,6,7,8,9],\"data\":[[1.0,7,0.0,true,4,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,1.5,345.0,138.0,264.329999999999984,0.0],[1.0,25,0.0,true,24,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,4.0,345.0,138.0,378.180000000000007,0.0],[1.0,80,0.0,true,79,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,6.49999999999999,345.0,138.0,366.300000000000011,0.0],[1.0,86,-0.04494949494949,true,85,100.0,null,1,0.0,0.0,9900.0,null,null,null,null,false,null,null,null,null,161.0,138.0,2072.25986507098105,279.97199999999998],[1.0,115,-0.16565656565657,true,67,100.0,null,1,0.0,0.0,9900.0,null,null,null,null,false,null,null,null,null,345.0,161.0,40.236040821631541,3.366],[1.0,29,0.0,true,16,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,4.0,345.0,138.0,384.120000000000005,0.0],[1.0,37,0.0,true,36,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,6.49999999999999,345.0,138.0,371.25,0.0],[1.0,62,0.0,true,58,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,4.0,345.0,138.0,382.139999999999986,0.0],[1.0,63,0.0,true,60,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,1.5,345.0,138.0,265.319999999999993,0.0],[1.0,64,0.0,true,65,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,6.49999999999999,345.0,138.0,366.300000000000011,0.0],[1.0,64,-0.64444444444444,true,67,100.0,null,1,0.0,0.0,9900.0,null,null,null,null,false,null,null,null,null,345.0,161.0,158.988082081645359,13.662000000000001],[1.0,67,0.0,true,68,100.0,null,1,0.0,0.0,9900.0,null,null,0.0,null,false,-1.0,\"hv\",null,6.49999999999999,161.0,138.0,366.300000000000011,0.0],[1.0,80,-0.81616161616162,true,67,100.0,null,1,0.0,0.0,9900.0,null,null,null,null,false,null,null,null,null,345.0,161.0,200.72906123678257,17.324999999999999]]}",
                        "orient": "split",
                        "dtype": {
                            "df": "float64",
                            "hv_bus": "uint32",
                            "i0_percent": "float64",
                            "in_service": "bool",
                            "lv_bus": "uint32",
                            "max_loading_percent": "float64",
                            "name": "object",
                            "parallel": "uint32",
                            "pfe_kw": "float64",
                            "shift_degree": "float64",
                            "sn_mva": "float64",
                            "std_type": "object",
                            "tap_max": "float64",
                            "tap_neutral": "float64",
                            "tap_min": "float64",
                            "tap_phase_shifter": "bool",
                            "tap_pos": "float64",
                            "tap_side": "object",
                            "tap_step_degree": "float64",
                            "tap_step_percent": "float64",
                            "vn_hv_kv": "float64",
                            "vn_lv_kv": "float64",
                            "vk_percent": "float64",
                            "vkr_percent": "float64"
                        }
                    },
                    "trafo3w": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"name\",\"std_type\",\"hv_bus\",\"mv_bus\",\"lv_bus\",\"sn_hv_mva\",\"sn_mv_mva\",\"sn_lv_mva\",\"vn_hv_kv\",\"vn_mv_kv\",\"vn_lv_kv\",\"vk_hv_percent\",\"vk_mv_percent\",\"vk_lv_percent\",\"vkr_hv_percent\",\"vkr_mv_percent\",\"vkr_lv_percent\",\"pfe_kw\",\"i0_percent\",\"shift_mv_degree\",\"shift_lv_degree\",\"tap_side\",\"tap_neutral\",\"tap_min\",\"tap_max\",\"tap_step_percent\",\"tap_step_degree\",\"tap_pos\",\"tap_at_star_point\",\"in_service\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "name": "object",
                            "std_type": "object",
                            "hv_bus": "uint32",
                            "mv_bus": "uint32",
                            "lv_bus": "uint32",
                            "sn_hv_mva": "float64",
                            "sn_mv_mva": "float64",
                            "sn_lv_mva": "float64",
                            "vn_hv_kv": "float64",
                            "vn_mv_kv": "float64",
                            "vn_lv_kv": "float64",
                            "vk_hv_percent": "float64",
                            "vk_mv_percent": "float64",
                            "vk_lv_percent": "float64",
                            "vkr_hv_percent": "float64",
                            "vkr_mv_percent": "float64",
                            "vkr_lv_percent": "float64",
                            "pfe_kw": "float64",
                            "i0_percent": "float64",
                            "shift_mv_degree": "float64",
                            "shift_lv_degree": "float64",
                            "tap_side": "object",
                            "tap_neutral": "int32",
                            "tap_min": "int32",
                            "tap_max": "int32",
                            "tap_step_percent": "float64",
                            "tap_step_degree": "float64",
                            "tap_pos": "int32",
                            "tap_at_star_point": "bool",
                            "in_service": "bool"
                        }
                    },
                    "impedance": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"name\",\"from_bus\",\"to_bus\",\"rft_pu\",\"xft_pu\",\"rtf_pu\",\"xtf_pu\",\"sn_mva\",\"in_service\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "name": "object",
                            "from_bus": "uint32",
                            "to_bus": "uint32",
                            "rft_pu": "float64",
                            "xft_pu": "float64",
                            "rtf_pu": "float64",
                            "xtf_pu": "float64",
                            "sn_mva": "float64",
                            "in_service": "bool"
                        }
                    },
                    "dcline": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"name\",\"from_bus\",\"to_bus\",\"p_mw\",\"loss_percent\",\"loss_mw\",\"vm_from_pu\",\"vm_to_pu\",\"max_p_mw\",\"min_q_from_mvar\",\"min_q_to_mvar\",\"max_q_from_mvar\",\"max_q_to_mvar\",\"in_service\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "name": "object",
                            "from_bus": "uint32",
                            "to_bus": "uint32",
                            "p_mw": "float64",
                            "loss_percent": "float64",
                            "loss_mw": "float64",
                            "vm_from_pu": "float64",
                            "vm_to_pu": "float64",
                            "max_p_mw": "float64",
                            "min_q_from_mvar": "float64",
                            "min_q_to_mvar": "float64",
                            "max_q_from_mvar": "float64",
                            "max_q_to_mvar": "float64",
                            "in_service": "bool"
                        }
                    },
                    "ward": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"name\",\"bus\",\"ps_mw\",\"qs_mvar\",\"qz_mvar\",\"pz_mw\",\"in_service\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "name": "object",
                            "bus": "uint32",
                            "ps_mw": "float64",
                            "qs_mvar": "float64",
                            "qz_mvar": "float64",
                            "pz_mw": "float64",
                            "in_service": "bool"
                        }
                    },
                    "xward": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"name\",\"bus\",\"ps_mw\",\"qs_mvar\",\"qz_mvar\",\"pz_mw\",\"r_ohm\",\"x_ohm\",\"vm_pu\",\"in_service\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "name": "object",
                            "bus": "uint32",
                            "ps_mw": "float64",
                            "qs_mvar": "float64",
                            "qz_mvar": "float64",
                            "pz_mw": "float64",
                            "r_ohm": "float64",
                            "x_ohm": "float64",
                            "vm_pu": "float64",
                            "in_service": "bool"
                        }
                    },
                    "measurement": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"name\",\"measurement_type\",\"element_type\",\"element\",\"value\",\"std_dev\",\"side\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "name": "object",
                            "measurement_type": "object",
                            "element_type": "object",
                            "element": "uint32",
                            "value": "float64",
                            "std_dev": "float64",
                            "side": "object"
                        }
                    },
                    "pwl_cost": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"power_type\",\"element\",\"et\",\"points\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "power_type": "object",
                            "element": "object",
                            "et": "object",
                            "points": "object"
                        }
                    },
                    "poly_cost": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"element\",\"et\",\"cp0_eur\",\"cp1_eur_per_mw\",\"cp2_eur_per_mw2\",\"cq0_eur\",\"cq1_eur_per_mvar\",\"cq2_eur_per_mvar2\"],\"index\":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53],\"data\":[[0.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[1.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[10.0,\"gen\",0.0,20.0,0.0454545,0.0,0.0,0.0],[11.0,\"gen\",0.0,20.0,0.0318471,0.0,0.0,0.0],[12.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[13.0,\"gen\",0.0,20.0,1.42857,0.0,0.0,0.0],[14.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[15.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[16.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[17.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[18.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[19.0,\"gen\",0.0,20.0,0.526316,0.0,0.0,0.0],[2.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[20.0,\"gen\",0.0,20.0,0.0490196,0.0,0.0,0.0],[21.0,\"gen\",0.0,20.0,0.208333,0.0,0.0,0.0],[22.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[23.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[24.0,\"gen\",0.0,20.0,0.0645161,0.0,0.0,0.0],[25.0,\"gen\",0.0,20.0,0.0625,0.0,0.0,0.0],[26.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[27.0,\"gen\",0.0,20.0,0.0255754,0.0,0.0,0.0],[28.0,\"gen\",0.0,20.0,0.0255102,0.0,0.0,0.0],[0.0,\"ext_grid\",0.0,20.0,0.0193648,0.0,0.0,0.0],[3.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[29.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[30.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[31.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[32.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[33.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[34.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[35.0,\"gen\",0.0,20.0,0.0209644,0.0,0.0,0.0],[36.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[37.0,\"gen\",0.0,20.0,2.5,0.0,0.0,0.0],[38.0,\"gen\",0.0,20.0,0.0164745,0.0,0.0,0.0],[4.0,\"gen\",0.0,20.0,0.0222222,0.0,0.0,0.0],[39.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[40.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[41.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[42.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[43.0,\"gen\",0.0,20.0,0.0396825,0.0,0.0,0.0],[44.0,\"gen\",0.0,20.0,0.25,0.0,0.0,0.0],[45.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[46.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[47.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[48.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[5.0,\"gen\",0.0,20.0,0.117647,0.0,0.0,0.0],[49.0,\"gen\",0.0,20.0,0.277778,0.0,0.0,0.0],[50.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[51.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[52.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[6.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[7.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[8.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0],[9.0,\"gen\",0.0,40.0,0.01,0.0,0.0,0.0]]}",
                        "orient": "split",
                        "dtype": {
                            "element": "object",
                            "et": "object",
                            "cp0_eur": "float64",
                            "cp1_eur_per_mw": "float64",
                            "cp2_eur_per_mw2": "float64",
                            "cq0_eur": "float64",
                            "cq1_eur_per_mvar": "float64",
                            "cq2_eur_per_mvar2": "float64"
                        }
                    },
                    "controller": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"object\",\"in_service\",\"order\",\"level\",\"recycle\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "object": "object",
                            "in_service": "bool",
                            "order": "float64",
                            "level": "object",
                            "recycle": "bool"
                        }
                    },
                    "line_geodata": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"coords\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "coords": "object"
                        }
                    },
                    "bus_geodata": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"x\",\"y\",\"coords\"],\"index\":[0,1,10,100,101,102,103,104,105,106,107,108,109,11,110,111,112,113,114,115,116,117,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,42,43,44,45,46,47,48,49,5,50,51,52,53,54,55,56,57,58,59,6,60,61,62,63,64,65,66,67,68,69,7,70,71,72,73,74,75,76,77,78,79,8,80,81,82,83,84,85,86,87,88,89,9,90,91,92,93,94,95,96,97,98,99],\"data\":[[-2.2753708781,2.8543413351,null],[-2.9368186836,2.2121792656,null],[-1.8084809456,1.078348466,null],[-7.6657645056,-8.7967993378,null],[-7.9822420097,-9.888120989300001,null],[-8.236522503,-7.2232594029,null],[-7.9601004239,-7.6514408832,null],[-9.083981855399999,-7.9345527343,null],[-8.1036220416,-8.232699824599999,null],[-9.060039011900001,-8.827258800499999,null],[-10.2222637563,-8.3101899758,null],[-10.4378672891,-7.4563425184,null],[-9.358739758900001,-6.8228628497,null],[-2.4521165899,1.188776931,null],[-10.440240663200001,-6.5781548081,null],[-10.0488510549,-5.9288876624,null],[0.0210809519,-1.4982181621,null],[2.2630604919,-2.7868829987,null],[3.1644478736,-2.3853607175,null],[-4.1528539003,-4.9348933434,null],[-3.5172125372,1.7774067282,null],[-2.1457808291,-8.546699179499999,null],[-2.5339886632,0.2882488225,null],[-3.0867743311,0.3407195505,null],[-2.3297472058,-0.6794711196,null],[-1.4955187998,0.1152182093,null],[-1.0489390134,-1.0911455168,null],[-1.8356962694,-0.8292516322,null],[-2.6758265845,-1.2562696807,null],[-1.271723021,-1.462800663,null],[-1.8344312496,1.7094451782,null],[0.07347017,-2.1572792461,null],[0.9345466389,-3.4895133702,null],[0.3017014198,-3.799814658,null],[-0.2778569786,-5.1101840996,null],[0.9631168234,-2.9810635525,null],[-0.198421303,-2.3375045516,null],[1.9445958496,-2.2280444796,null],[2.4540519038,-1.2219755328,null],[1.4972464461,-0.7253664385,null],[-1.2374314539,-2.0300347993,null],[-0.8886268958,1.5532705585,null],[0.3831112001,-1.275137263,null],[0.9036426682,-2.4715943587,null],[-2.9833939473,-1.6491727215,null],[-3.6197042214,-2.0172857852,null],[-4.3347899304,-2.2080206011,null],[-4.7428784027,-1.3946055976,null],[-2.9559373581,-2.7113176762,null],[-1.9467507717,-3.1510525007,null],[-3.641503222,-3.1155712489,null],[-2.805657193,-3.7660095235,null],[-0.9632829393,0.694729907,null],[-3.152467354,-4.4158860294,null],[-2.274592477,-5.0451097798,null],[-4.1875091242,-3.1292654848,null],[-3.5914723889,-4.4209512352,null],[-2.6299875318,-5.6039710603,null],[-3.1343589918,-6.5469756868,null],[-2.3186784101,-6.714154807,null],[-2.0246271797,-7.1400534325,null],[-1.7242222011,-6.2584123784,null],[-1.1556470812,-7.0861765912,null],[-0.4109732139,1.7675722497,null],[-1.2192198707,-7.6503732093,null],[-0.7485947859,-8.8465566854,null],[0.1459019855,-8.3262367561,null],[-0.4574089283,-6.9830212906,null],[0.861150583,-7.4175967649,null],[0.3692720954,-7.6867284542,null],[-0.3863224274,-8.201226779600001,null],[-0.096485424,-8.545959205300001,null],[0.6559375961,-6.6866792823,null],[1.4936494437,-6.25030887,null],[-1.4836496595,2.2620419749,null],[0.7929941546,-5.649416175,null],[0.0834611333,-5.7403617415,null],[0.5977433777,-5.3868498213,null],[-0.6251705072,-4.7706623304,null],[-1.9502435677,-4.5114830894,null],[-1.2078613144,-5.3568578774,null],[-0.1247860314,-5.0525544148,null],[-3.181060934,-5.4511075483,null],[-2.6512661211,-6.4304476843,null],[-1.2850990413,-6.3036420672,null],[-0.6792997874,-0.5813210368,null],[0.0746177813,-7.0344963198,null],[1.023717234,-5.910201308,null],[1.174663034,-8.1088356138,null],[-1.7822188996,-7.1334058675,null],[-2.5699674814,-7.3039427852,null],[-3.0946467463,-8.595026854,null],[-3.6187253582,-7.4081044036,null],[-3.6927835541,-8.1636937876,null],[-4.7266433485,-7.7222452258,null],[-4.8540470344,-6.9946893219,null],[0.492210326,0.0395449742,null],[-4.4266115974,-5.9494061119,null],[-4.1989304816,-8.569851999000001,null],[-4.1284060755,-9.8585501443,null],[-4.0286420276,-10.845582416899999,null],[-4.9872676968,-10.7009377309,null],[-4.6398857601,-11.9115838943,null],[-4.4754585089,-13.005245318,null],[-5.7173489254,-11.3183526816,null],[-6.2786501696,-10.590002866800001,null],[-7.0445216565,-11.514366900500001,null],[1.5086493002,0.7537990673,null],[-7.5500105531,-10.5464275695,null],[-7.0121455802,-9.433431104,null],[-6.479922409,-9.6952112759,null],[-6.4131486749,-8.5909140285,null],[-5.719463754,-8.846278912800001,null],[-5.3339595297,-8.0606533179,null],[-5.4897495419,-6.9547678609,null],[-6.0865230031,-7.3112763353,null],[-6.2121103608,-6.9919515492,null],[-7.0667004418,-7.9857230104,null]]}",
                        "orient": "split",
                        "dtype": {
                            "x": "float64",
                            "y": "float64",
                            "coords": "object"
                        }
                    },
                    "version": "2.4.0",
                    "converged": "True",
                    "name": "",
                    "f_hz": 60,
                    "sn_mva": 1.0,
                    "std_types": {
                        "line": {
                            "NAYY 4x150 SE": {
                                "type": "cs",
                                "r_ohm_per_km": 0.208,
                                "q_mm2": 150,
                                "x_ohm_per_km": 0.08,
                                "c_nf_per_km": 261.0,
                                "max_i_ka": 0.27
                            },
                            "70-AL1/11-ST1A 20.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.4132,
                                "q_mm2": 70,
                                "x_ohm_per_km": 0.36,
                                "c_nf_per_km": 9.7,
                                "max_i_ka": 0.29
                            },
                            "NA2XS2Y 1x70 RM/25 6/10 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.443,
                                "q_mm2": 70,
                                "x_ohm_per_km": 0.123,
                                "c_nf_per_km": 280.0,
                                "max_i_ka": 0.217
                            },
                            "N2XS(FL)2Y 1x300 RM/35 64/110 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.06,
                                "q_mm2": 300,
                                "x_ohm_per_km": 0.144,
                                "c_nf_per_km": 144.0,
                                "max_i_ka": 0.588
                            },
                            "NA2XS2Y 1x120 RM/25 6/10 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.253,
                                "q_mm2": 120,
                                "x_ohm_per_km": 0.113,
                                "c_nf_per_km": 340.0,
                                "max_i_ka": 0.28
                            },
                            "149-AL1/24-ST1A 10.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.194,
                                "q_mm2": 149,
                                "x_ohm_per_km": 0.315,
                                "c_nf_per_km": 11.25,
                                "max_i_ka": 0.47
                            },
                            "15-AL1/3-ST1A 0.4": {
                                "type": "ol",
                                "r_ohm_per_km": 1.8769,
                                "q_mm2": 16,
                                "x_ohm_per_km": 0.35,
                                "c_nf_per_km": 11.0,
                                "max_i_ka": 0.105
                            },
                            "NA2XS2Y 1x185 RM/25 6/10 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.161,
                                "q_mm2": 185,
                                "x_ohm_per_km": 0.11,
                                "c_nf_per_km": 406.0,
                                "max_i_ka": 0.358
                            },
                            "NA2XS2Y 1x240 RM/25 6/10 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.122,
                                "q_mm2": 240,
                                "x_ohm_per_km": 0.105,
                                "c_nf_per_km": 456.0,
                                "max_i_ka": 0.416
                            },
                            "N2XS(FL)2Y 1x240 RM/35 64/110 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.075,
                                "q_mm2": 240,
                                "x_ohm_per_km": 0.149,
                                "c_nf_per_km": 135.0,
                                "max_i_ka": 0.526
                            },
                            "NAYY 4x120 SE": {
                                "type": "cs",
                                "r_ohm_per_km": 0.225,
                                "q_mm2": 120,
                                "x_ohm_per_km": 0.08,
                                "c_nf_per_km": 264.0,
                                "max_i_ka": 0.242
                            },
                            "48-AL1/8-ST1A 10.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.5939,
                                "q_mm2": 48,
                                "x_ohm_per_km": 0.35,
                                "c_nf_per_km": 10.1,
                                "max_i_ka": 0.21
                            },
                            "94-AL1/15-ST1A 10.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.306,
                                "q_mm2": 94,
                                "x_ohm_per_km": 0.33,
                                "c_nf_per_km": 10.75,
                                "max_i_ka": 0.35
                            },
                            "NA2XS2Y 1x70 RM/25 12/20 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.443,
                                "q_mm2": 70,
                                "x_ohm_per_km": 0.132,
                                "c_nf_per_km": 190.0,
                                "max_i_ka": 0.22
                            },
                            "243-AL1/39-ST1A 20.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.1188,
                                "q_mm2": 243,
                                "x_ohm_per_km": 0.32,
                                "c_nf_per_km": 11.0,
                                "max_i_ka": 0.645
                            },
                            "NA2XS2Y 1x150 RM/25 6/10 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.206,
                                "q_mm2": 150,
                                "x_ohm_per_km": 0.11,
                                "c_nf_per_km": 360.0,
                                "max_i_ka": 0.315
                            },
                            "184-AL1/30-ST1A 110.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.1571,
                                "q_mm2": 184,
                                "x_ohm_per_km": 0.4,
                                "c_nf_per_km": 8.8,
                                "max_i_ka": 0.535
                            },
                            "149-AL1/24-ST1A 110.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.194,
                                "q_mm2": 149,
                                "x_ohm_per_km": 0.41,
                                "c_nf_per_km": 8.75,
                                "max_i_ka": 0.47
                            },
                            "NA2XS2Y 1x240 RM/25 12/20 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.122,
                                "q_mm2": 240,
                                "x_ohm_per_km": 0.112,
                                "c_nf_per_km": 304.0,
                                "max_i_ka": 0.421
                            },
                            "122-AL1/20-ST1A 20.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.2376,
                                "q_mm2": 122,
                                "x_ohm_per_km": 0.344,
                                "c_nf_per_km": 10.3,
                                "max_i_ka": 0.41
                            },
                            "48-AL1/8-ST1A 20.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.5939,
                                "q_mm2": 48,
                                "x_ohm_per_km": 0.372,
                                "c_nf_per_km": 9.5,
                                "max_i_ka": 0.21
                            },
                            "34-AL1/6-ST1A 10.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.8342,
                                "q_mm2": 34,
                                "x_ohm_per_km": 0.36,
                                "c_nf_per_km": 9.7,
                                "max_i_ka": 0.17
                            },
                            "24-AL1/4-ST1A 0.4": {
                                "type": "ol",
                                "r_ohm_per_km": 1.2012,
                                "q_mm2": 24,
                                "x_ohm_per_km": 0.335,
                                "c_nf_per_km": 11.25,
                                "max_i_ka": 0.14
                            },
                            "184-AL1/30-ST1A 20.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.1571,
                                "q_mm2": 184,
                                "x_ohm_per_km": 0.33,
                                "c_nf_per_km": 10.75,
                                "max_i_ka": 0.535
                            },
                            "94-AL1/15-ST1A 20.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.306,
                                "q_mm2": 94,
                                "x_ohm_per_km": 0.35,
                                "c_nf_per_km": 10.0,
                                "max_i_ka": 0.35
                            },
                            "NAYY 4x50 SE": {
                                "type": "cs",
                                "r_ohm_per_km": 0.642,
                                "q_mm2": 50,
                                "x_ohm_per_km": 0.083,
                                "c_nf_per_km": 210.0,
                                "max_i_ka": 0.142
                            },
                            "490-AL1/64-ST1A 380.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.059,
                                "q_mm2": 490,
                                "x_ohm_per_km": 0.253,
                                "c_nf_per_km": 11.0,
                                "max_i_ka": 0.96
                            },
                            "48-AL1/8-ST1A 0.4": {
                                "type": "ol",
                                "r_ohm_per_km": 0.5939,
                                "q_mm2": 48,
                                "x_ohm_per_km": 0.3,
                                "c_nf_per_km": 12.2,
                                "max_i_ka": 0.21
                            },
                            "NA2XS2Y 1x95 RM/25 6/10 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.313,
                                "q_mm2": 95,
                                "x_ohm_per_km": 0.123,
                                "c_nf_per_km": 315.0,
                                "max_i_ka": 0.249
                            },
                            "NA2XS2Y 1x120 RM/25 12/20 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.253,
                                "q_mm2": 120,
                                "x_ohm_per_km": 0.119,
                                "c_nf_per_km": 230.0,
                                "max_i_ka": 0.283
                            },
                            "34-AL1/6-ST1A 20.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.8342,
                                "q_mm2": 34,
                                "x_ohm_per_km": 0.382,
                                "c_nf_per_km": 9.15,
                                "max_i_ka": 0.17
                            },
                            "94-AL1/15-ST1A 0.4": {
                                "type": "ol",
                                "r_ohm_per_km": 0.306,
                                "q_mm2": 94,
                                "x_ohm_per_km": 0.29,
                                "c_nf_per_km": 13.2,
                                "max_i_ka": 0.35
                            },
                            "NA2XS2Y 1x185 RM/25 12/20 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.161,
                                "q_mm2": 185,
                                "x_ohm_per_km": 0.117,
                                "c_nf_per_km": 273.0,
                                "max_i_ka": 0.362
                            },
                            "NA2XS2Y 1x150 RM/25 12/20 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.206,
                                "q_mm2": 150,
                                "x_ohm_per_km": 0.116,
                                "c_nf_per_km": 250.0,
                                "max_i_ka": 0.319
                            },
                            "243-AL1/39-ST1A 110.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.1188,
                                "q_mm2": 243,
                                "x_ohm_per_km": 0.39,
                                "c_nf_per_km": 9.0,
                                "max_i_ka": 0.645
                            },
                            "490-AL1/64-ST1A 220.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.059,
                                "q_mm2": 490,
                                "x_ohm_per_km": 0.285,
                                "c_nf_per_km": 10.0,
                                "max_i_ka": 0.96
                            },
                            "N2XS(FL)2Y 1x185 RM/35 64/110 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.099,
                                "q_mm2": 185,
                                "x_ohm_per_km": 0.156,
                                "c_nf_per_km": 125.0,
                                "max_i_ka": 0.457
                            },
                            "N2XS(FL)2Y 1x120 RM/35 64/110 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.153,
                                "q_mm2": 120,
                                "x_ohm_per_km": 0.166,
                                "c_nf_per_km": 112.0,
                                "max_i_ka": 0.366
                            },
                            "NA2XS2Y 1x95 RM/25 12/20 kV": {
                                "type": "cs",
                                "r_ohm_per_km": 0.313,
                                "q_mm2": 95,
                                "x_ohm_per_km": 0.132,
                                "c_nf_per_km": 216.0,
                                "max_i_ka": 0.252
                            },
                            "122-AL1/20-ST1A 10.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.2376,
                                "q_mm2": 122,
                                "x_ohm_per_km": 0.323,
                                "c_nf_per_km": 11.1,
                                "max_i_ka": 0.41
                            },
                            "149-AL1/24-ST1A 20.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.194,
                                "q_mm2": 149,
                                "x_ohm_per_km": 0.337,
                                "c_nf_per_km": 10.5,
                                "max_i_ka": 0.47
                            },
                            "70-AL1/11-ST1A 10.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.4132,
                                "q_mm2": 70,
                                "x_ohm_per_km": 0.339,
                                "c_nf_per_km": 10.4,
                                "max_i_ka": 0.29
                            },
                            "305-AL1/39-ST1A 110.0": {
                                "type": "ol",
                                "r_ohm_per_km": 0.0949,
                                "q_mm2": 305,
                                "x_ohm_per_km": 0.38,
                                "c_nf_per_km": 9.2,
                                "max_i_ka": 0.74
                            }
                        },
                        "trafo": {
                            "0.4 MVA 20/0.4 kV": {
                                "shift_degree": 150,
                                "vector_group": "Dyn5",
                                "vn_hv_kv": 20.0,
                                "pfe_kw": 1.35,
                                "i0_percent": 0.3375,
                                "vn_lv_kv": 0.4,
                                "sn_mva": 0.4,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -2,
                                "vkr_percent": 1.425,
                                "tap_step_percent": 2.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 2,
                                "vk_percent": 6.0
                            },
                            "63 MVA 110/20 kV v1.4.3 and older": {
                                "shift_degree": 150,
                                "vector_group": "YNd5",
                                "vn_hv_kv": 110.0,
                                "pfe_kw": 33.0,
                                "i0_percent": 0.086,
                                "vn_lv_kv": 20.0,
                                "sn_mva": 63.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.322,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 11.2
                            },
                            "63 MVA 110/10 kV v1.4.3 and older": {
                                "shift_degree": 150,
                                "vector_group": "YNd5",
                                "vn_hv_kv": 110.0,
                                "pfe_kw": 31.51,
                                "i0_percent": 0.078,
                                "vn_lv_kv": 10.0,
                                "sn_mva": 63.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.31,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 10.04
                            },
                            "25 MVA 110/20 kV v1.4.3 and older": {
                                "shift_degree": 150,
                                "vector_group": "YNd5",
                                "vn_hv_kv": 110.0,
                                "pfe_kw": 29.0,
                                "i0_percent": 0.071,
                                "vn_lv_kv": 20.0,
                                "sn_mva": 25.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.282,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 11.2
                            },
                            "40 MVA 110/20 kV v1.4.3 and older": {
                                "shift_degree": 150,
                                "vector_group": "YNd5",
                                "vn_hv_kv": 110.0,
                                "pfe_kw": 31.0,
                                "i0_percent": 0.08,
                                "vn_lv_kv": 20.0,
                                "sn_mva": 40.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.302,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 11.2
                            },
                            "0.25 MVA 20/0.4 kV": {
                                "shift_degree": 150,
                                "vector_group": "Yzn5",
                                "vn_hv_kv": 20.0,
                                "pfe_kw": 0.8,
                                "i0_percent": 0.32,
                                "vn_lv_kv": 0.4,
                                "sn_mva": 0.25,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -2,
                                "vkr_percent": 1.44,
                                "tap_step_percent": 2.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 2,
                                "vk_percent": 6.0
                            },
                            "25 MVA 110/10 kV v1.4.3 and older": {
                                "shift_degree": 150,
                                "vector_group": "YNd5",
                                "vn_hv_kv": 110.0,
                                "pfe_kw": 28.51,
                                "i0_percent": 0.073,
                                "vn_lv_kv": 10.0,
                                "sn_mva": 25.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.276,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 10.04
                            },
                            "0.25 MVA 10/0.4 kV": {
                                "shift_degree": 150,
                                "vector_group": "Dyn5",
                                "vn_hv_kv": 10.0,
                                "pfe_kw": 0.6,
                                "i0_percent": 0.24,
                                "vn_lv_kv": 0.4,
                                "sn_mva": 0.25,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -2,
                                "vkr_percent": 1.2,
                                "tap_step_percent": 2.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 2,
                                "vk_percent": 4.0
                            },
                            "160 MVA 380/110 kV": {
                                "shift_degree": 0,
                                "vector_group": "Yy0",
                                "vn_hv_kv": 380.0,
                                "pfe_kw": 60.0,
                                "i0_percent": 0.06,
                                "vn_lv_kv": 110.0,
                                "sn_mva": 160.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.25,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 12.2
                            },
                            "63 MVA 110/10 kV": {
                                "shift_degree": 150,
                                "vector_group": "YNd5",
                                "vn_hv_kv": 110.0,
                                "pfe_kw": 22.0,
                                "i0_percent": 0.04,
                                "vn_lv_kv": 10.0,
                                "sn_mva": 63.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.32,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 18.0
                            },
                            "0.63 MVA 20/0.4 kV": {
                                "shift_degree": 150,
                                "vector_group": "Dyn5",
                                "vn_hv_kv": 20.0,
                                "pfe_kw": 1.65,
                                "i0_percent": 0.2619,
                                "vn_lv_kv": 0.4,
                                "sn_mva": 0.63,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -2,
                                "vkr_percent": 1.206,
                                "tap_step_percent": 2.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 2,
                                "vk_percent": 6.0
                            },
                            "0.4 MVA 10/0.4 kV": {
                                "shift_degree": 150,
                                "vector_group": "Dyn5",
                                "vn_hv_kv": 10.0,
                                "pfe_kw": 0.95,
                                "i0_percent": 0.2375,
                                "vn_lv_kv": 0.4,
                                "sn_mva": 0.4,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -2,
                                "vkr_percent": 1.325,
                                "tap_step_percent": 2.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 2,
                                "vk_percent": 4.0
                            },
                            "0.63 MVA 10/0.4 kV": {
                                "shift_degree": 150,
                                "vector_group": "Dyn5",
                                "vn_hv_kv": 10.0,
                                "pfe_kw": 1.18,
                                "i0_percent": 0.1873,
                                "vn_lv_kv": 0.4,
                                "sn_mva": 0.63,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -2,
                                "vkr_percent": 1.0794,
                                "tap_step_percent": 2.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 2,
                                "vk_percent": 4.0
                            },
                            "63 MVA 110/20 kV": {
                                "shift_degree": 150,
                                "vector_group": "YNd5",
                                "vn_hv_kv": 110.0,
                                "pfe_kw": 22.0,
                                "i0_percent": 0.04,
                                "vn_lv_kv": 20.0,
                                "sn_mva": 63.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.32,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 18.0
                            },
                            "100 MVA 220/110 kV": {
                                "shift_degree": 0,
                                "vector_group": "Yy0",
                                "vn_hv_kv": 220.0,
                                "pfe_kw": 55.0,
                                "i0_percent": 0.06,
                                "vn_lv_kv": 110.0,
                                "sn_mva": 100.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.26,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 12.0
                            },
                            "25 MVA 110/10 kV": {
                                "shift_degree": 150,
                                "vector_group": "YNd5",
                                "vn_hv_kv": 110.0,
                                "pfe_kw": 14.0,
                                "i0_percent": 0.07,
                                "vn_lv_kv": 10.0,
                                "sn_mva": 25.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.41,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 12.0
                            },
                            "40 MVA 110/20 kV": {
                                "shift_degree": 150,
                                "vector_group": "YNd5",
                                "vn_hv_kv": 110.0,
                                "pfe_kw": 18.0,
                                "i0_percent": 0.05,
                                "vn_lv_kv": 20.0,
                                "sn_mva": 40.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.34,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 16.2
                            },
                            "40 MVA 110/10 kV v1.4.3 and older": {
                                "shift_degree": 150,
                                "vector_group": "YNd5",
                                "vn_hv_kv": 110.0,
                                "pfe_kw": 30.45,
                                "i0_percent": 0.076,
                                "vn_lv_kv": 10.0,
                                "sn_mva": 40.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.295,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 10.04
                            },
                            "25 MVA 110/20 kV": {
                                "shift_degree": 150,
                                "vector_group": "YNd5",
                                "vn_hv_kv": 110.0,
                                "pfe_kw": 14.0,
                                "i0_percent": 0.07,
                                "vn_lv_kv": 20.0,
                                "sn_mva": 25.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.41,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 12.0
                            },
                            "40 MVA 110/10 kV": {
                                "shift_degree": 150,
                                "vector_group": "YNd5",
                                "vn_hv_kv": 110.0,
                                "pfe_kw": 18.0,
                                "i0_percent": 0.05,
                                "vn_lv_kv": 10.0,
                                "sn_mva": 40.0,
                                "tap_step_degree": 0,
                                "tap_neutral": 0,
                                "tap_min": -9,
                                "vkr_percent": 0.34,
                                "tap_step_percent": 1.5,
                                "tap_side": "hv",
                                "tap_phase_shifter": "False",
                                "tap_max": 9,
                                "vk_percent": 16.2
                            }
                        },
                        "trafo3w": {
                            "63/25/38 MVA 110/10/10 kV": {
                                "vector_group": "YN0yn0yn0",
                                "vn_mv_kv": 10,
                                "vn_lv_kv": 10,
                                "shift_lv_degree": 0,
                                "shift_mv_degree": 0,
                                "pfe_kw": 35,
                                "vn_hv_kv": 110,
                                "i0_percent": 0.89,
                                "sn_lv_mva": 38.0,
                                "sn_hv_mva": 63.0,
                                "sn_mv_mva": 25.0,
                                "vkr_lv_percent": 0.35,
                                "tap_neutral": 0,
                                "tap_min": -10,
                                "vk_mv_percent": 10.4,
                                "vkr_hv_percent": 0.28,
                                "vk_lv_percent": 10.4,
                                "tap_max": 10,
                                "vkr_mv_percent": 0.32,
                                "tap_step_percent": 1.2,
                                "tap_side": "hv",
                                "vk_hv_percent": 10.4
                            },
                            "63/25/38 MVA 110/20/10 kV": {
                                "vector_group": "YN0yn0yn0",
                                "vn_mv_kv": 20,
                                "vn_lv_kv": 10,
                                "shift_lv_degree": 0,
                                "shift_mv_degree": 0,
                                "pfe_kw": 35,
                                "vn_hv_kv": 110,
                                "i0_percent": 0.89,
                                "sn_lv_mva": 38.0,
                                "sn_hv_mva": 63.0,
                                "sn_mv_mva": 25.0,
                                "vkr_lv_percent": 0.35,
                                "tap_neutral": 0,
                                "tap_min": -10,
                                "vk_mv_percent": 10.4,
                                "vkr_hv_percent": 0.28,
                                "vk_lv_percent": 10.4,
                                "tap_max": 10,
                                "vkr_mv_percent": 0.32,
                                "tap_step_percent": 1.2,
                                "tap_side": "hv",
                                "vk_hv_percent": 10.4
                            }
                        }
                    },
                    "res_bus": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"vm_pu\",\"va_degree\",\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "vm_pu": "float64",
                            "va_degree": "float64",
                            "p_mw": "float64",
                            "q_mvar": "float64"
                        }
                    },
                    "res_line": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_from_mw\",\"q_from_mvar\",\"p_to_mw\",\"q_to_mvar\",\"pl_mw\",\"ql_mvar\",\"i_from_ka\",\"i_to_ka\",\"i_ka\",\"vm_from_pu\",\"va_from_degree\",\"vm_to_pu\",\"va_to_degree\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_from_mw": "float64",
                            "q_from_mvar": "float64",
                            "p_to_mw": "float64",
                            "q_to_mvar": "float64",
                            "pl_mw": "float64",
                            "ql_mvar": "float64",
                            "i_from_ka": "float64",
                            "i_to_ka": "float64",
                            "i_ka": "float64",
                            "vm_from_pu": "float64",
                            "va_from_degree": "float64",
                            "vm_to_pu": "float64",
                            "va_to_degree": "float64",
                            "loading_percent": "float64"
                        }
                    },
                    "res_trafo": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_hv_mw\",\"q_hv_mvar\",\"p_lv_mw\",\"q_lv_mvar\",\"pl_mw\",\"ql_mvar\",\"i_hv_ka\",\"i_lv_ka\",\"vm_hv_pu\",\"va_hv_degree\",\"vm_lv_pu\",\"va_lv_degree\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_hv_mw": "float64",
                            "q_hv_mvar": "float64",
                            "p_lv_mw": "float64",
                            "q_lv_mvar": "float64",
                            "pl_mw": "float64",
                            "ql_mvar": "float64",
                            "i_hv_ka": "float64",
                            "i_lv_ka": "float64",
                            "vm_hv_pu": "float64",
                            "va_hv_degree": "float64",
                            "vm_lv_pu": "float64",
                            "va_lv_degree": "float64",
                            "loading_percent": "float64"
                        }
                    },
                    "res_trafo3w": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_hv_mw\",\"q_hv_mvar\",\"p_mv_mw\",\"q_mv_mvar\",\"p_lv_mw\",\"q_lv_mvar\",\"pl_mw\",\"ql_mvar\",\"i_hv_ka\",\"i_mv_ka\",\"i_lv_ka\",\"vm_hv_pu\",\"va_hv_degree\",\"vm_mv_pu\",\"va_mv_degree\",\"vm_lv_pu\",\"va_lv_degree\",\"va_internal_degree\",\"vm_internal_pu\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_hv_mw": "float64",
                            "q_hv_mvar": "float64",
                            "p_mv_mw": "float64",
                            "q_mv_mvar": "float64",
                            "p_lv_mw": "float64",
                            "q_lv_mvar": "float64",
                            "pl_mw": "float64",
                            "ql_mvar": "float64",
                            "i_hv_ka": "float64",
                            "i_mv_ka": "float64",
                            "i_lv_ka": "float64",
                            "vm_hv_pu": "float64",
                            "va_hv_degree": "float64",
                            "vm_mv_pu": "float64",
                            "va_mv_degree": "float64",
                            "vm_lv_pu": "float64",
                            "va_lv_degree": "float64",
                            "va_internal_degree": "float64",
                            "vm_internal_pu": "float64",
                            "loading_percent": "float64"
                        }
                    },
                    "res_impedance": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_from_mw\",\"q_from_mvar\",\"p_to_mw\",\"q_to_mvar\",\"pl_mw\",\"ql_mvar\",\"i_from_ka\",\"i_to_ka\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_from_mw": "float64",
                            "q_from_mvar": "float64",
                            "p_to_mw": "float64",
                            "q_to_mvar": "float64",
                            "pl_mw": "float64",
                            "ql_mvar": "float64",
                            "i_from_ka": "float64",
                            "i_to_ka": "float64"
                        }
                    },
                    "res_ext_grid": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_mw": "float64",
                            "q_mvar": "float64"
                        }
                    },
                    "res_load": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_mw": "float64",
                            "q_mvar": "float64"
                        }
                    },
                    "res_motor": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_mw": "float64",
                            "q_mvar": "float64"
                        }
                    },
                    "res_sgen": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_mw": "float64",
                            "q_mvar": "float64"
                        }
                    },
                    "res_storage": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_mw": "float64",
                            "q_mvar": "float64"
                        }
                    },
                    "res_shunt": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_mw\",\"q_mvar\",\"vm_pu\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_mw": "float64",
                            "q_mvar": "float64",
                            "vm_pu": "float64"
                        }
                    },
                    "res_gen": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_mw\",\"q_mvar\",\"va_degree\",\"vm_pu\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_mw": "float64",
                            "q_mvar": "float64",
                            "va_degree": "float64",
                            "vm_pu": "float64"
                        }
                    },
                    "res_ward": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_mw\",\"q_mvar\",\"vm_pu\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_mw": "float64",
                            "q_mvar": "float64",
                            "vm_pu": "float64"
                        }
                    },
                    "res_xward": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_mw\",\"q_mvar\",\"vm_pu\",\"va_internal_degree\",\"vm_internal_pu\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_mw": "float64",
                            "q_mvar": "float64",
                            "vm_pu": "float64",
                            "va_internal_degree": "float64",
                            "vm_internal_pu": "float64"
                        }
                    },
                    "res_dcline": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_from_mw\",\"q_from_mvar\",\"p_to_mw\",\"q_to_mvar\",\"pl_mw\",\"vm_from_pu\",\"va_from_degree\",\"vm_to_pu\",\"va_to_degree\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_from_mw": "float64",
                            "q_from_mvar": "float64",
                            "p_to_mw": "float64",
                            "q_to_mvar": "float64",
                            "pl_mw": "float64",
                            "vm_from_pu": "float64",
                            "va_from_degree": "float64",
                            "vm_to_pu": "float64",
                            "va_to_degree": "float64"
                        }
                    },
                    "res_bus_est": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"vm_pu\",\"va_degree\",\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "vm_pu": "float64",
                            "va_degree": "float64",
                            "p_mw": "float64",
                            "q_mvar": "float64"
                        }
                    },
                    "res_line_est": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_from_mw\",\"q_from_mvar\",\"p_to_mw\",\"q_to_mvar\",\"pl_mw\",\"ql_mvar\",\"i_from_ka\",\"i_to_ka\",\"i_ka\",\"vm_from_pu\",\"va_from_degree\",\"vm_to_pu\",\"va_to_degree\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_from_mw": "float64",
                            "q_from_mvar": "float64",
                            "p_to_mw": "float64",
                            "q_to_mvar": "float64",
                            "pl_mw": "float64",
                            "ql_mvar": "float64",
                            "i_from_ka": "float64",
                            "i_to_ka": "float64",
                            "i_ka": "float64",
                            "vm_from_pu": "float64",
                            "va_from_degree": "float64",
                            "vm_to_pu": "float64",
                            "va_to_degree": "float64",
                            "loading_percent": "float64"
                        }
                    },
                    "res_trafo_est": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_hv_mw\",\"q_hv_mvar\",\"p_lv_mw\",\"q_lv_mvar\",\"pl_mw\",\"ql_mvar\",\"i_hv_ka\",\"i_lv_ka\",\"vm_hv_pu\",\"va_hv_degree\",\"vm_lv_pu\",\"va_lv_degree\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_hv_mw": "float64",
                            "q_hv_mvar": "float64",
                            "p_lv_mw": "float64",
                            "q_lv_mvar": "float64",
                            "pl_mw": "float64",
                            "ql_mvar": "float64",
                            "i_hv_ka": "float64",
                            "i_lv_ka": "float64",
                            "vm_hv_pu": "float64",
                            "va_hv_degree": "float64",
                            "vm_lv_pu": "float64",
                            "va_lv_degree": "float64",
                            "loading_percent": "float64"
                        }
                    },
                    "res_trafo3w_est": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_hv_mw\",\"q_hv_mvar\",\"p_mv_mw\",\"q_mv_mvar\",\"p_lv_mw\",\"q_lv_mvar\",\"pl_mw\",\"ql_mvar\",\"i_hv_ka\",\"i_mv_ka\",\"i_lv_ka\",\"vm_hv_pu\",\"va_hv_degree\",\"vm_mv_pu\",\"va_mv_degree\",\"vm_lv_pu\",\"va_lv_degree\",\"va_internal_degree\",\"vm_internal_pu\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_hv_mw": "float64",
                            "q_hv_mvar": "float64",
                            "p_mv_mw": "float64",
                            "q_mv_mvar": "float64",
                            "p_lv_mw": "float64",
                            "q_lv_mvar": "float64",
                            "pl_mw": "float64",
                            "ql_mvar": "float64",
                            "i_hv_ka": "float64",
                            "i_mv_ka": "float64",
                            "i_lv_ka": "float64",
                            "vm_hv_pu": "float64",
                            "va_hv_degree": "float64",
                            "vm_mv_pu": "float64",
                            "va_mv_degree": "float64",
                            "vm_lv_pu": "float64",
                            "va_lv_degree": "float64",
                            "va_internal_degree": "float64",
                            "vm_internal_pu": "float64",
                            "loading_percent": "float64"
                        }
                    },
                    "res_bus_sc": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                        "orient": "split"
                    },
                    "res_line_sc": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                        "orient": "split"
                    },
                    "res_trafo_sc": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                        "orient": "split"
                    },
                    "res_trafo3w_sc": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                        "orient": "split"
                    },
                    "res_ext_grid_sc": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                        "orient": "split"
                    },
                    "res_gen_sc": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                        "orient": "split"
                    },
                    "res_sgen_sc": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                        "orient": "split"
                    },
                    "res_bus_3ph": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"vm_a_pu\",\"va_a_degree\",\"vm_b_pu\",\"va_b_degree\",\"vm_c_pu\",\"va_c_degree\",\"p_a_mw\",\"q_a_mvar\",\"p_b_mw\",\"q_b_mvar\",\"p_c_mw\",\"q_c_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "vm_a_pu": "float64",
                            "va_a_degree": "float64",
                            "vm_b_pu": "float64",
                            "va_b_degree": "float64",
                            "vm_c_pu": "float64",
                            "va_c_degree": "float64",
                            "p_a_mw": "float64",
                            "q_a_mvar": "float64",
                            "p_b_mw": "float64",
                            "q_b_mvar": "float64",
                            "p_c_mw": "float64",
                            "q_c_mvar": "float64"
                        }
                    },
                    "res_line_3ph": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_a_from_mw\",\"q_a_from_mvar\",\"p_b_from_mw\",\"q_b_from_mvar\",\"q_c_from_mvar\",\"p_a_to_mw\",\"q_a_to_mvar\",\"p_b_to_mw\",\"q_b_to_mvar\",\"p_c_to_mw\",\"q_c_to_mvar\",\"p_a_l_mw\",\"q_a_l_mvar\",\"p_b_l_mw\",\"q_b_l_mvar\",\"p_c_l_mw\",\"q_c_l_mvar\",\"i_a_from_ka\",\"i_a_to_ka\",\"i_b_from_ka\",\"i_b_to_ka\",\"i_c_from_ka\",\"i_c_to_ka\",\"i_a_ka\",\"i_b_ka\",\"i_c_ka\",\"i_n_from_ka\",\"i_n_to_ka\",\"i_n_ka\",\"loading_a_percent\",\"loading_b_percent\",\"loading_c_percent\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_a_from_mw": "float64",
                            "q_a_from_mvar": "float64",
                            "p_b_from_mw": "float64",
                            "q_b_from_mvar": "float64",
                            "q_c_from_mvar": "float64",
                            "p_a_to_mw": "float64",
                            "q_a_to_mvar": "float64",
                            "p_b_to_mw": "float64",
                            "q_b_to_mvar": "float64",
                            "p_c_to_mw": "float64",
                            "q_c_to_mvar": "float64",
                            "p_a_l_mw": "float64",
                            "q_a_l_mvar": "float64",
                            "p_b_l_mw": "float64",
                            "q_b_l_mvar": "float64",
                            "p_c_l_mw": "float64",
                            "q_c_l_mvar": "float64",
                            "i_a_from_ka": "float64",
                            "i_a_to_ka": "float64",
                            "i_b_from_ka": "float64",
                            "i_b_to_ka": "float64",
                            "i_c_from_ka": "float64",
                            "i_c_to_ka": "float64",
                            "i_a_ka": "float64",
                            "i_b_ka": "float64",
                            "i_c_ka": "float64",
                            "i_n_from_ka": "float64",
                            "i_n_to_ka": "float64",
                            "i_n_ka": "float64",
                            "loading_a_percent": "float64",
                            "loading_b_percent": "float64",
                            "loading_c_percent": "float64"
                        }
                    },
                    "res_trafo_3ph": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_a_hv_mw\",\"q_a_hv_mvar\",\"p_b_hv_mw\",\"q_b_hv_mvar\",\"p_c_hv_mw\",\"q_c_hv_mvar\",\"p_a_lv_mw\",\"q_a_lv_mvar\",\"p_b_lv_mw\",\"q_b_lv_mvar\",\"p_c_lv_mw\",\"q_c_lv_mvar\",\"p_a_l_mw\",\"q_a_l_mvar\",\"p_b_l_mw\",\"q_b_l_mvar\",\"p_c_l_mw\",\"q_c_l_mvar\",\"i_a_hv_ka\",\"i_a_lv_ka\",\"i_b_hv_ka\",\"i_b_lv_ka\",\"i_c_hv_ka\",\"i_c_lv_ka\",\"loading_a_percent\",\"loading_b_percent\",\"loading_c_percent\",\"loading_percent\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_a_hv_mw": "float64",
                            "q_a_hv_mvar": "float64",
                            "p_b_hv_mw": "float64",
                            "q_b_hv_mvar": "float64",
                            "p_c_hv_mw": "float64",
                            "q_c_hv_mvar": "float64",
                            "p_a_lv_mw": "float64",
                            "q_a_lv_mvar": "float64",
                            "p_b_lv_mw": "float64",
                            "q_b_lv_mvar": "float64",
                            "p_c_lv_mw": "float64",
                            "q_c_lv_mvar": "float64",
                            "p_a_l_mw": "float64",
                            "q_a_l_mvar": "float64",
                            "p_b_l_mw": "float64",
                            "q_b_l_mvar": "float64",
                            "p_c_l_mw": "float64",
                            "q_c_l_mvar": "float64",
                            "i_a_hv_ka": "float64",
                            "i_a_lv_ka": "float64",
                            "i_b_hv_ka": "float64",
                            "i_b_lv_ka": "float64",
                            "i_c_hv_ka": "float64",
                            "i_c_lv_ka": "float64",
                            "loading_a_percent": "float64",
                            "loading_b_percent": "float64",
                            "loading_c_percent": "float64",
                            "loading_percent": "float64"
                        }
                    },
                    "res_ext_grid_3ph": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_a_mw\",\"q_a_mvar\",\"p_b_mw\",\"q_b_mvar\",\"p_c_mw\",\"q_c_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_a_mw": "float64",
                            "q_a_mvar": "float64",
                            "p_b_mw": "float64",
                            "q_b_mvar": "float64",
                            "p_c_mw": "float64",
                            "q_c_mvar": "float64"
                        }
                    },
                    "res_shunt_3ph": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[],\"index\":[],\"data\":[]}",
                        "orient": "split"
                    },
                    "res_load_3ph": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_mw": "float64",
                            "q_mvar": "float64"
                        }
                    },
                    "res_sgen_3ph": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_mw": "float64",
                            "q_mvar": "float64"
                        }
                    },
                    "res_storage_3ph": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_mw": "float64",
                            "q_mvar": "float64"
                        }
                    },
                    "res_asymmetric_load_3ph": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_a_mw\",\"q_a_mvar\",\"p_b_mw\",\"q_b_mvar\",\"p_c_mw\",\"q_c_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_a_mw": "float64",
                            "q_a_mvar": "float64",
                            "p_b_mw": "float64",
                            "q_b_mvar": "float64",
                            "p_c_mw": "float64",
                            "q_c_mvar": "float64"
                        }
                    },
                    "res_asymmetric_sgen_3ph": {
                        "_module": "pandas.core.frame",
                        "_class": "DataFrame",
                        "_object": "{\"columns\":[\"p_a_mw\",\"q_a_mvar\",\"p_b_mw\",\"q_b_mvar\",\"p_c_mw\",\"q_c_mvar\"],\"index\":[],\"data\":[]}",
                        "orient": "split",
                        "dtype": {
                            "p_a_mw": "float64",
                            "q_a_mvar": "float64",
                            "p_b_mw": "float64",
                            "q_b_mvar": "float64",
                            "p_c_mw": "float64",
                            "q_c_mvar": "float64"
                        }
                    },
                    "user_pf_options": {

                    },
                    "OPF_converged": "False"
                }
            }
        }
    },
    "payloadType": "Payload",
    "source": "PublisherWithPipeline",
    "uc": "UC0",
    "utcTimestamp": "2021-11-09T13:31:07"
}

# net = pp.networks.case_ieee30()
# jnet = pp.to_json(net, filename=None, encryption_key=None)

rmq_json['messageId'] = str(uuid.uuid4())
# rmq_json['payload']['data'] = jnet
rmq_json['payload']['timestamp'] = str(datetime.datetime.now())
rmq_json['utcTimestamp'] = str(datetime.datetime.now(timezone.utc))

# rmq_json['messageId']            = str(uuid.uuid4())
# rmq_json['payload']['data']      = jnet
# # rmq_json['payload']['data']      = {"module":"test",
# #                                         "name":"test"
# #                                         }
# rmq_json['payload']['timestamp'] = str(datetime.datetime.now())
# rmq_json['utcTimestamp']         = str(datetime.datetime.now(timezone.utc))

# print(rmq_json)
# input("Press enter to continue...")

# with open('data1.json', 'w', encoding='utf-8') as f:
#     json.dump(jnet, f, ensure_ascii=False, indent=4)
    
# with open('data2.json', 'w', encoding='utf-8') as f:
#     json.dump(rmq_json, f, ensure_ascii=False, indent=4)    
    
########################## connect ro rabbitmq AIDB gridpilot #########################

print(" [x] Trying rabbitmq")
url = os.environ.get('CLOUDAMQP_URL', 'amqps://jdzlpput:5ny6ANo8vdhwr8iYkwVXd_8sRwyIKLBi@rattlesnake.rmq.cloudamqp.com/jdzlpput')
params = pika.URLParameters(url)
connection = pika.BlockingConnection(params)
channel = connection.channel()
channel.queue_declare(queue='mlst_iim')
channel.basic_publish(exchange='', routing_key='mlst_iim', body=json.dumps(rmq_json))

print(" [x] Sent test")
connection.close()

# credentials = pika.PlainCredentials('iim-guest', 'iimguest')
# # parameters = pika.ConnectionParameters('3.120.35.154', 5672, 'iim', credentials)
# parameters = pika.ConnectionParameters('rabbit.prod.gridpilot.tech', 5672, 'iim', credentials)
# connection = pika.BlockingConnection(parameters)

# channel = connection.channel()
# channel.queue_declare(queue='IIM#IIM')

# channel.basic_publish(exchange='Islanding_Exchange.headers', routing_key ='IIM#IIM', body = json.dumps(rmq_json))

# print(" [x] Sent 'Islanding scheme!'")
# connection.close()


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

#################### END MINCUT METHOD ####################





print('End.')