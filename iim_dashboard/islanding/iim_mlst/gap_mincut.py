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
net=pp.networks.case_ieee30()

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

rmq_json = {
    "messageId": "e8793bb7-a4ea-4a5b-99b2-dc9caac24720",
    "name": "islanding_result",
    "payload": {
        "_module": "pandapower.auxiliary",
        "_class": "pandapowerNet",
        "_object": {
            "bus": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"vn_kv\",\"type\",\"zone\",\"in_service\"],\"index\":[0,1,2],\"data\":[[\"Bus 1\",20.0,\"b\",null,true],[\"Bus 2\",0.4,\"b\",null,true],[\"Bus 3\",0.4,\"b\",null,true]]}",
                "dtype": {
                    "name": "object",
                    "vn_kv": "float64",
                    "type": "object",
                    "zone": "object",
                    "in_service": "bool"
                },
                "orient": "split"
            },
            "load": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"p_mw\",\"q_mvar\",\"const_z_percent\",\"const_i_percent\",\"sn_mva\",\"scaling\",\"in_service\",\"type\",\"priority\"],\"index\":[0],\"data\":[[\"Load\",2,0.1,0.05,0.0,0.0,null,1.0,true,null,10]]}",
                "dtype": {
                    "name": "object",
                    "bus": "uint32",
                    "p_mw": "float64",
                    "q_mvar": "float64",
                    "const_z_percent": "float64",
                    "const_i_percent": "float64",
                    "sn_mva": "float64",
                    "scaling": "float64",
                    "in_service": "bool",
                    "type": "object",
                    "priority": "int64"
                },
                "orient": "split"
            },
            "sgen": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"p_mw\",\"q_mvar\",\"sn_mva\",\"scaling\",\"in_service\",\"type\",\"current_source\"],\"index\":[0],\"data\":[[\"static generator\",2,2.0,-0.5,null,1.0,true,null,true]]}",
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
                },
                "orient": "split"
            },
            "storage": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"p_mw\",\"q_mvar\",\"sn_mva\",\"soc_percent\",\"min_e_mwh\",\"max_e_mwh\",\"scaling\",\"in_service\",\"type\"],\"index\":[],\"data\":[]}",
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
                },
                "orient": "split"
            },
            "gen": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"p_mw\",\"vm_pu\",\"sn_mva\",\"min_q_mvar\",\"max_q_mvar\",\"scaling\",\"slack\",\"in_service\",\"type\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "name": "object",
                    "bus": "uint32",
                    "p_mw": "float64",
                    "vm_pu": "float64",
                    "sn_mva": "float64",
                    "min_q_mvar": "float64",
                    "max_q_mvar": "float64",
                    "scaling": "float64",
                    "slack": "bool",
                    "in_service": "bool",
                    "type": "object"
                },
                "orient": "split"
            },
            "switch": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"bus\",\"element\",\"et\",\"type\",\"closed\",\"name\",\"z_ohm\"],\"index\":[0],\"data\":[[2,0,\"l\",null,true,\"S1\",0.0]]}",
                "dtype": {
                    "bus": "int64",
                    "element": "int64",
                    "et": "object",
                    "type": "object",
                    "closed": "bool",
                    "name": "object",
                    "z_ohm": "float64"
                },
                "orient": "split"
            },
            "shunt": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"bus\",\"name\",\"q_mvar\",\"p_mw\",\"vn_kv\",\"step\",\"max_step\",\"in_service\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "bus": "uint32",
                    "name": "object",
                    "q_mvar": "float64",
                    "p_mw": "float64",
                    "vn_kv": "float64",
                    "step": "uint32",
                    "max_step": "uint32",
                    "in_service": "bool"
                },
                "orient": "split"
            },
            "ext_grid": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"vm_pu\",\"va_degree\",\"in_service\"],\"index\":[0],\"data\":[[\"Grid Connection\",0,1.02,0.0,true]]}",
                "dtype": {
                    "name": "object",
                    "bus": "uint32",
                    "vm_pu": "float64",
                    "va_degree": "float64",
                    "in_service": "bool"
                },
                "orient": "split"
            },
            "line": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"std_type\",\"from_bus\",\"to_bus\",\"length_km\",\"r_ohm_per_km\",\"x_ohm_per_km\",\"c_nf_per_km\",\"g_us_per_km\",\"max_i_ka\",\"df\",\"parallel\",\"type\",\"in_service\"],\"index\":[0],\"data\":[[\"L1\",\"NAYY 4x50 SE\",1,2,0.1,0.642,0.083,210.0,0.0,0.142,1.0,1,\"cs\",true]]}",
                "dtype": {
                    "name": "object",
                    "std_type": "object",
                    "from_bus": "uint32",
                    "to_bus": "uint32",
                    "length_km": "float64",
                    "r_ohm_per_km": "float64",
                    "x_ohm_per_km": "float64",
                    "c_nf_per_km": "float64",
                    "g_us_per_km": "float64",
                    "max_i_ka": "float64",
                    "df": "float64",
                    "parallel": "uint32",
                    "type": "object",
                    "in_service": "bool"
                },
                "orient": "split"
            },
            "trafo": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"std_type\",\"hv_bus\",\"lv_bus\",\"sn_mva\",\"vn_hv_kv\",\"vn_lv_kv\",\"vk_percent\",\"vkr_percent\",\"pfe_kw\",\"i0_percent\",\"shift_degree\",\"tap_side\",\"tap_neutral\",\"tap_min\",\"tap_max\",\"tap_step_percent\",\"tap_step_degree\",\"tap_pos\",\"tap_phase_shifter\",\"parallel\",\"df\",\"in_service\"],\"index\":[0],\"data\":[[\"Trafo\",\"0.4 MVA 20\\/0.4 kV\",0,1,0.4,20.0,0.4,6.0,1.425,1.35,0.3375,150.0,\"hv\",0,-2,2,2.5,0.0,0,False,1,1.0,true]]}",
                "dtype": {
                    "name": "object",
                    "std_type": "object",
                    "hv_bus": "uint32",
                    "lv_bus": "uint32",
                    "sn_mva": "float64",
                    "vn_hv_kv": "float64",
                    "vn_lv_kv": "float64",
                    "vk_percent": "float64",
                    "vkr_percent": "float64",
                    "pfe_kw": "float64",
                    "i0_percent": "float64",
                    "shift_degree": "float64",
                    "tap_side": "object",
                    "tap_neutral": "int32",
                    "tap_min": "int32",
                    "tap_max": "int32",
                    "tap_step_percent": "float64",
                    "tap_step_degree": "float64",
                    "tap_pos": "int32",
                    "tap_phase_shifter": "bool",
                    "parallel": "uint32",
                    "df": "float64",
                    "in_service": "bool"
                },
                "orient": "split"
            },
            "trafo3w": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"std_type\",\"hv_bus\",\"mv_bus\",\"lv_bus\",\"sn_hv_mva\",\"sn_mv_mva\",\"sn_lv_mva\",\"vn_hv_kv\",\"vn_mv_kv\",\"vn_lv_kv\",\"vk_hv_percent\",\"vk_mv_percent\",\"vk_lv_percent\",\"vkr_hv_percent\",\"vkr_mv_percent\",\"vkr_lv_percent\",\"pfe_kw\",\"i0_percent\",\"shift_mv_degree\",\"shift_lv_degree\",\"tap_side\",\"tap_neutral\",\"tap_min\",\"tap_max\",\"tap_step_percent\",\"tap_step_degree\",\"tap_pos\",\"tap_at_star_point\",\"in_service\"],\"index\":[],\"data\":[]}",
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
                },
                "orient": "split"
            },
            "impedance": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"from_bus\",\"to_bus\",\"rft_pu\",\"xft_pu\",\"rtf_pu\",\"xtf_pu\",\"sn_mva\",\"in_service\"],\"index\":[],\"data\":[]}",
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
                },
                "orient": "split"
            },
            "dcline": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"from_bus\",\"to_bus\",\"p_mw\",\"loss_percent\",\"loss_mw\",\"vm_from_pu\",\"vm_to_pu\",\"max_p_mw\",\"min_q_from_mvar\",\"min_q_to_mvar\",\"max_q_from_mvar\",\"max_q_to_mvar\",\"in_service\"],\"index\":[],\"data\":[]}",
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
                },
                "orient": "split"
            },
            "ward": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"ps_mw\",\"qs_mvar\",\"qz_mvar\",\"pz_mw\",\"in_service\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "name": "object",
                    "bus": "uint32",
                    "ps_mw": "float64",
                    "qs_mvar": "float64",
                    "qz_mvar": "float64",
                    "pz_mw": "float64",
                    "in_service": "bool"
                },
                "orient": "split"
            },
            "xward": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"bus\",\"ps_mw\",\"qs_mvar\",\"qz_mvar\",\"pz_mw\",\"r_ohm\",\"x_ohm\",\"vm_pu\",\"in_service\"],\"index\":[],\"data\":[]}",
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
                },
                "orient": "split"
            },
            "measurement": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"name\",\"measurement_type\",\"element_type\",\"element\",\"value\",\"std_dev\",\"side\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "name": "object",
                    "measurement_type": "object",
                    "element_type": "object",
                    "element": "uint32",
                    "value": "float64",
                    "std_dev": "float64",
                    "side": "object"
                },
                "orient": "split"
            },
            "pwl_cost": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"power_type\",\"element\",\"et\",\"points\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "power_type": "object",
                    "element": "object",
                    "et": "object",
                    "points": "object"
                },
                "orient": "split"
            },
            "poly_cost": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"element\",\"et\",\"cp0_eur\",\"cp1_eur_per_mw\",\"cp2_eur_per_mw2\",\"cq0_eur\",\"cq1_eur_per_mvar\",\"cq2_eur_per_mvar2\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "element": "object",
                    "et": "object",
                    "cp0_eur": "float64",
                    "cp1_eur_per_mw": "float64",
                    "cp2_eur_per_mw2": "float64",
                    "cq0_eur": "float64",
                    "cq1_eur_per_mvar": "float64",
                    "cq2_eur_per_mvar2": "float64"
                },
                "orient": "split"
            },
            "line_geodata": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"coords\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "coords": "object"
                },
                "orient": "split"
            },
            "bus_geodata": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"x\",\"y\",\"coords\"],\"index\":[0,1,2],\"data\":[[10.0,20.0,null],[10.0,10.0,null],[10.0,0.0,null]]}",
                "dtype": {
                    "x": "float64",
                    "y": "float64",
                    "coords": "object"
                },
                "orient": "split"
            },
            "version": "2.1.0",
            "converged": False,
            "name": "GMexample",
            "f_hz": 50,
            "sn_mva": 1.0,
            "std_types": {
                "line": {
                    "NAYY 4x50 SE": {
                        "c_nf_per_km": 210.0,
                        "r_ohm_per_km": 0.642,
                        "x_ohm_per_km": 0.083,
                        "max_i_ka": 0.142,
                        "type": "cs",
                        "q_mm2": 50,
                        "alpha": 0.00403
                    },
                    "NAYY 4x120 SE": {
                        "c_nf_per_km": 264.0,
                        "r_ohm_per_km": 0.225,
                        "x_ohm_per_km": 0.08,
                        "max_i_ka": 0.242,
                        "type": "cs",
                        "q_mm2": 120,
                        "alpha": 0.00403
                    },
                    "NAYY 4x150 SE": {
                        "c_nf_per_km": 261.0,
                        "r_ohm_per_km": 0.208,
                        "x_ohm_per_km": 0.08,
                        "max_i_ka": 0.27,
                        "type": "cs",
                        "q_mm2": 150,
                        "alpha": 0.00403
                    },
                    "NA2XS2Y 1x95 RM/25 12/20 kV": {
                        "c_nf_per_km": 216.0,
                        "r_ohm_per_km": 0.313,
                        "x_ohm_per_km": 0.132,
                        "max_i_ka": 0.252,
                        "type": "cs",
                        "q_mm2": 95,
                        "alpha": 0.00403
                    },
                    "NA2XS2Y 1x185 RM/25 12/20 kV": {
                        "c_nf_per_km": 273.0,
                        "r_ohm_per_km": 0.161,
                        "x_ohm_per_km": 0.117,
                        "max_i_ka": 0.362,
                        "type": "cs",
                        "q_mm2": 185,
                        "alpha": 0.00403
                    },
                    "NA2XS2Y 1x240 RM/25 12/20 kV": {
                        "c_nf_per_km": 304.0,
                        "r_ohm_per_km": 0.122,
                        "x_ohm_per_km": 0.112,
                        "max_i_ka": 0.421,
                        "type": "cs",
                        "q_mm2": 240,
                        "alpha": 0.00403
                    },
                    "NA2XS2Y 1x95 RM/25 6/10 kV": {
                        "c_nf_per_km": 315.0,
                        "r_ohm_per_km": 0.313,
                        "x_ohm_per_km": 0.123,
                        "max_i_ka": 0.249,
                        "type": "cs",
                        "q_mm2": 95,
                        "alpha": 0.00403
                    },
                    "NA2XS2Y 1x185 RM/25 6/10 kV": {
                        "c_nf_per_km": 406.0,
                        "r_ohm_per_km": 0.161,
                        "x_ohm_per_km": 0.11,
                        "max_i_ka": 0.358,
                        "type": "cs",
                        "q_mm2": 185,
                        "alpha": 0.00403
                    },
                    "NA2XS2Y 1x240 RM/25 6/10 kV": {
                        "c_nf_per_km": 456.0,
                        "r_ohm_per_km": 0.122,
                        "x_ohm_per_km": 0.105,
                        "max_i_ka": 0.416,
                        "type": "cs",
                        "q_mm2": 240,
                        "alpha": 0.00403
                    },
                    "NA2XS2Y 1x150 RM/25 12/20 kV": {
                        "c_nf_per_km": 250.0,
                        "r_ohm_per_km": 0.206,
                        "x_ohm_per_km": 0.116,
                        "max_i_ka": 0.319,
                        "type": "cs",
                        "q_mm2": 150,
                        "alpha": 0.00403
                    },
                    "NA2XS2Y 1x120 RM/25 12/20 kV": {
                        "c_nf_per_km": 230.0,
                        "r_ohm_per_km": 0.253,
                        "x_ohm_per_km": 0.119,
                        "max_i_ka": 0.283,
                        "type": "cs",
                        "q_mm2": 120,
                        "alpha": 0.00403
                    },
                    "NA2XS2Y 1x70 RM/25 12/20 kV": {
                        "c_nf_per_km": 190.0,
                        "r_ohm_per_km": 0.443,
                        "x_ohm_per_km": 0.132,
                        "max_i_ka": 0.22,
                        "type": "cs",
                        "q_mm2": 70,
                        "alpha": 0.00403
                    },
                    "NA2XS2Y 1x150 RM/25 6/10 kV": {
                        "c_nf_per_km": 360.0,
                        "r_ohm_per_km": 0.206,
                        "x_ohm_per_km": 0.11,
                        "max_i_ka": 0.315,
                        "type": "cs",
                        "q_mm2": 150,
                        "alpha": 0.00403
                    },
                    "NA2XS2Y 1x120 RM/25 6/10 kV": {
                        "c_nf_per_km": 340.0,
                        "r_ohm_per_km": 0.253,
                        "x_ohm_per_km": 0.113,
                        "max_i_ka": 0.28,
                        "type": "cs",
                        "q_mm2": 120,
                        "alpha": 0.00403
                    },
                    "NA2XS2Y 1x70 RM/25 6/10 kV": {
                        "c_nf_per_km": 280.0,
                        "r_ohm_per_km": 0.443,
                        "x_ohm_per_km": 0.123,
                        "max_i_ka": 0.217,
                        "type": "cs",
                        "q_mm2": 70,
                        "alpha": 0.00403
                    },
                    "N2XS(FL)2Y 1x120 RM/35 64/110 kV": {
                        "c_nf_per_km": 112.0,
                        "r_ohm_per_km": 0.153,
                        "x_ohm_per_km": 0.166,
                        "max_i_ka": 0.366,
                        "type": "cs",
                        "q_mm2": 120,
                        "alpha": 0.00393
                    },
                    "N2XS(FL)2Y 1x185 RM/35 64/110 kV": {
                        "c_nf_per_km": 125.0,
                        "r_ohm_per_km": 0.099,
                        "x_ohm_per_km": 0.156,
                        "max_i_ka": 0.457,
                        "type": "cs",
                        "q_mm2": 185,
                        "alpha": 0.00393
                    },
                    "N2XS(FL)2Y 1x240 RM/35 64/110 kV": {
                        "c_nf_per_km": 135.0,
                        "r_ohm_per_km": 0.075,
                        "x_ohm_per_km": 0.149,
                        "max_i_ka": 0.526,
                        "type": "cs",
                        "q_mm2": 240,
                        "alpha": 0.00393
                    },
                    "N2XS(FL)2Y 1x300 RM/35 64/110 kV": {
                        "c_nf_per_km": 144.0,
                        "r_ohm_per_km": 0.06,
                        "x_ohm_per_km": 0.144,
                        "max_i_ka": 0.588,
                        "type": "cs",
                        "q_mm2": 300,
                        "alpha": 0.00393
                    },
                    "15-AL1/3-ST1A 0.4": {
                        "c_nf_per_km": 11.0,
                        "r_ohm_per_km": 1.8769,
                        "x_ohm_per_km": 0.35,
                        "max_i_ka": 0.105,
                        "type": "ol",
                        "q_mm2": 16,
                        "alpha": 0.00403
                    },
                    "24-AL1/4-ST1A 0.4": {
                        "c_nf_per_km": 11.25,
                        "r_ohm_per_km": 1.2012,
                        "x_ohm_per_km": 0.335,
                        "max_i_ka": 0.14,
                        "type": "ol",
                        "q_mm2": 24,
                        "alpha": 0.00403
                    },
                    "48-AL1/8-ST1A 0.4": {
                        "c_nf_per_km": 12.2,
                        "r_ohm_per_km": 0.5939,
                        "x_ohm_per_km": 0.3,
                        "max_i_ka": 0.21,
                        "type": "ol",
                        "q_mm2": 48,
                        "alpha": 0.00403
                    },
                    "94-AL1/15-ST1A 0.4": {
                        "c_nf_per_km": 13.2,
                        "r_ohm_per_km": 0.306,
                        "x_ohm_per_km": 0.29,
                        "max_i_ka": 0.35,
                        "type": "ol",
                        "q_mm2": 94,
                        "alpha": 0.00403
                    },
                    "34-AL1/6-ST1A 10.0": {
                        "c_nf_per_km": 9.7,
                        "r_ohm_per_km": 0.8342,
                        "x_ohm_per_km": 0.36,
                        "max_i_ka": 0.17,
                        "type": "ol",
                        "q_mm2": 34,
                        "alpha": 0.00403
                    },
                    "48-AL1/8-ST1A 10.0": {
                        "c_nf_per_km": 10.1,
                        "r_ohm_per_km": 0.5939,
                        "x_ohm_per_km": 0.35,
                        "max_i_ka": 0.21,
                        "type": "ol",
                        "q_mm2": 48,
                        "alpha": 0.00403
                    },
                    "70-AL1/11-ST1A 10.0": {
                        "c_nf_per_km": 10.4,
                        "r_ohm_per_km": 0.4132,
                        "x_ohm_per_km": 0.339,
                        "max_i_ka": 0.29,
                        "type": "ol",
                        "q_mm2": 70,
                        "alpha": 0.00403
                    },
                    "94-AL1/15-ST1A 10.0": {
                        "c_nf_per_km": 10.75,
                        "r_ohm_per_km": 0.306,
                        "x_ohm_per_km": 0.33,
                        "max_i_ka": 0.35,
                        "type": "ol",
                        "q_mm2": 94,
                        "alpha": 0.00403
                    },
                    "122-AL1/20-ST1A 10.0": {
                        "c_nf_per_km": 11.1,
                        "r_ohm_per_km": 0.2376,
                        "x_ohm_per_km": 0.323,
                        "max_i_ka": 0.41,
                        "type": "ol",
                        "q_mm2": 122,
                        "alpha": 0.00403
                    },
                    "149-AL1/24-ST1A 10.0": {
                        "c_nf_per_km": 11.25,
                        "r_ohm_per_km": 0.194,
                        "x_ohm_per_km": 0.315,
                        "max_i_ka": 0.47,
                        "type": "ol",
                        "q_mm2": 149,
                        "alpha": 0.00403
                    },
                    "34-AL1/6-ST1A 20.0": {
                        "c_nf_per_km": 9.15,
                        "r_ohm_per_km": 0.8342,
                        "x_ohm_per_km": 0.382,
                        "max_i_ka": 0.17,
                        "type": "ol",
                        "q_mm2": 34,
                        "alpha": 0.00403
                    },
                    "48-AL1/8-ST1A 20.0": {
                        "c_nf_per_km": 9.5,
                        "r_ohm_per_km": 0.5939,
                        "x_ohm_per_km": 0.372,
                        "max_i_ka": 0.21,
                        "type": "ol",
                        "q_mm2": 48,
                        "alpha": 0.00403
                    },
                    "70-AL1/11-ST1A 20.0": {
                        "c_nf_per_km": 9.7,
                        "r_ohm_per_km": 0.4132,
                        "x_ohm_per_km": 0.36,
                        "max_i_ka": 0.29,
                        "type": "ol",
                        "q_mm2": 70,
                        "alpha": 0.00403
                    },
                    "94-AL1/15-ST1A 20.0": {
                        "c_nf_per_km": 10.0,
                        "r_ohm_per_km": 0.306,
                        "x_ohm_per_km": 0.35,
                        "max_i_ka": 0.35,
                        "type": "ol",
                        "q_mm2": 94,
                        "alpha": 0.00403
                    },
                    "122-AL1/20-ST1A 20.0": {
                        "c_nf_per_km": 10.3,
                        "r_ohm_per_km": 0.2376,
                        "x_ohm_per_km": 0.344,
                        "max_i_ka": 0.41,
                        "type": "ol",
                        "q_mm2": 122,
                        "alpha": 0.00403
                    },
                    "149-AL1/24-ST1A 20.0": {
                        "c_nf_per_km": 10.5,
                        "r_ohm_per_km": 0.194,
                        "x_ohm_per_km": 0.337,
                        "max_i_ka": 0.47,
                        "type": "ol",
                        "q_mm2": 149,
                        "alpha": 0.00403
                    },
                    "184-AL1/30-ST1A 20.0": {
                        "c_nf_per_km": 10.75,
                        "r_ohm_per_km": 0.1571,
                        "x_ohm_per_km": 0.33,
                        "max_i_ka": 0.535,
                        "type": "ol",
                        "q_mm2": 184,
                        "alpha": 0.00403
                    },
                    "243-AL1/39-ST1A 20.0": {
                        "c_nf_per_km": 11.0,
                        "r_ohm_per_km": 0.1188,
                        "x_ohm_per_km": 0.32,
                        "max_i_ka": 0.645,
                        "type": "ol",
                        "q_mm2": 243,
                        "alpha": 0.00403
                    },
                    "48-AL1/8-ST1A 110.0": {
                        "c_nf_per_km": 8.0,
                        "r_ohm_per_km": 0.5939,
                        "x_ohm_per_km": 0.46,
                        "max_i_ka": 0.21,
                        "type": "ol",
                        "q_mm2": 48,
                        "alpha": 0.00403
                    },
                    "70-AL1/11-ST1A 110.0": {
                        "c_nf_per_km": 8.4,
                        "r_ohm_per_km": 0.4132,
                        "x_ohm_per_km": 0.45,
                        "max_i_ka": 0.29,
                        "type": "ol",
                        "q_mm2": 70,
                        "alpha": 0.00403
                    },
                    "94-AL1/15-ST1A 110.0": {
                        "c_nf_per_km": 8.65,
                        "r_ohm_per_km": 0.306,
                        "x_ohm_per_km": 0.44,
                        "max_i_ka": 0.35,
                        "type": "ol",
                        "q_mm2": 94,
                        "alpha": 0.00403
                    },
                    "122-AL1/20-ST1A 110.0": {
                        "c_nf_per_km": 8.5,
                        "r_ohm_per_km": 0.2376,
                        "x_ohm_per_km": 0.43,
                        "max_i_ka": 0.41,
                        "type": "ol",
                        "q_mm2": 122,
                        "alpha": 0.00403
                    },
                    "149-AL1/24-ST1A 110.0": {
                        "c_nf_per_km": 8.75,
                        "r_ohm_per_km": 0.194,
                        "x_ohm_per_km": 0.41,
                        "max_i_ka": 0.47,
                        "type": "ol",
                        "q_mm2": 149,
                        "alpha": 0.00403
                    },
                    "184-AL1/30-ST1A 110.0": {
                        "c_nf_per_km": 8.8,
                        "r_ohm_per_km": 0.1571,
                        "x_ohm_per_km": 0.4,
                        "max_i_ka": 0.535,
                        "type": "ol",
                        "q_mm2": 184,
                        "alpha": 0.00403
                    },
                    "243-AL1/39-ST1A 110.0": {
                        "c_nf_per_km": 9.0,
                        "r_ohm_per_km": 0.1188,
                        "x_ohm_per_km": 0.39,
                        "max_i_ka": 0.645,
                        "type": "ol",
                        "q_mm2": 243,
                        "alpha": 0.00403
                    },
                    "305-AL1/39-ST1A 110.0": {
                        "c_nf_per_km": 9.2,
                        "r_ohm_per_km": 0.0949,
                        "x_ohm_per_km": 0.38,
                        "max_i_ka": 0.74,
                        "type": "ol",
                        "q_mm2": 305,
                        "alpha": 0.00403
                    },
                    "490-AL1/64-ST1A 110.0": {
                        "c_nf_per_km": 9.75,
                        "r_ohm_per_km": 0.059,
                        "x_ohm_per_km": 0.37,
                        "max_i_ka": 0.96,
                        "type": "ol",
                        "q_mm2": 490,
                        "alpha": 0.00403
                    },
                    "679-AL1/86-ST1A 110.0": {
                        "c_nf_per_km": 9.95,
                        "r_ohm_per_km": 0.042,
                        "x_ohm_per_km": 0.36,
                        "max_i_ka": 0.115,
                        "type": "ol",
                        "q_mm2": 679,
                        "alpha": 0.00403
                    },
                    "490-AL1/64-ST1A 220.0": {
                        "c_nf_per_km": 10.0,
                        "r_ohm_per_km": 0.059,
                        "x_ohm_per_km": 0.285,
                        "max_i_ka": 0.96,
                        "type": "ol",
                        "q_mm2": 490,
                        "alpha": 0.00403
                    },
                    "679-AL1/86-ST1A 220.0": {
                        "c_nf_per_km": 11.7,
                        "r_ohm_per_km": 0.042,
                        "x_ohm_per_km": 0.275,
                        "max_i_ka": 0.115,
                        "type": "ol",
                        "q_mm2": 679,
                        "alpha": 0.00403
                    },
                    "490-AL1/64-ST1A 380.0": {
                        "c_nf_per_km": 11.0,
                        "r_ohm_per_km": 0.059,
                        "x_ohm_per_km": 0.253,
                        "max_i_ka": 0.96,
                        "type": "ol",
                        "q_mm2": 490,
                        "alpha": 0.00403
                    },
                    "679-AL1/86-ST1A 380.0": {
                        "c_nf_per_km": 14.6,
                        "r_ohm_per_km": 0.042,
                        "x_ohm_per_km": 0.25,
                        "max_i_ka": 0.115,
                        "type": "ol",
                        "q_mm2": 679,
                        "alpha": 0.00403
                    }
                },
                "trafo": {
                    "160 MVA 380/110 kV": {
                        "i0_percent": 0.06,
                        "pfe_kw": 60.0,
                        "vkr_percent": 0.25,
                        "sn_mva": 160.0,
                        "vn_lv_kv": 110.0,
                        "vn_hv_kv": 380,
                        "vk_percent": 12.2,
                        "shift_degree": 0,
                        "vector_group": "Yy0",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "tap_max": 9,
                        "tap_step_degree": 0,
                        "tap_step_percent": 1.5,
                        "tap_phase_shifter": False
                    },
                    "100 MVA 220/110 kV": {
                        "i0_percent": 0.06,
                        "pfe_kw": 55.0,
                        "vkr_percent": 0.26,
                        "sn_mva": 100.0,
                        "vn_lv_kv": 110.0,
                        "vn_hv_kv": 220,
                        "vk_percent": 12.0,
                        "shift_degree": 0,
                        "vector_group": "Yy0",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "tap_max": 9,
                        "tap_step_degree": 0,
                        "tap_step_percent": 1.5,
                        "tap_phase_shifter": False
                    },
                    "63 MVA 110/20 kV": {
                        "i0_percent": 0.04,
                        "pfe_kw": 22.0,
                        "vkr_percent": 0.32,
                        "sn_mva": 63.0,
                        "vn_lv_kv": 20.0,
                        "vn_hv_kv": 110,
                        "vk_percent": 18.0,
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "tap_max": 9,
                        "tap_step_degree": 0,
                        "tap_step_percent": 1.5,
                        "tap_phase_shifter": False
                    },
                    "40 MVA 110/20 kV": {
                        "i0_percent": 0.05,
                        "pfe_kw": 18.0,
                        "vkr_percent": 0.34,
                        "sn_mva": 40.0,
                        "vn_lv_kv": 20.0,
                        "vn_hv_kv": 110,
                        "vk_percent": 16.2,
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "tap_max": 9,
                        "tap_step_degree": 0,
                        "tap_step_percent": 1.5,
                        "tap_phase_shifter": False
                    },
                    "25 MVA 110/20 kV": {
                        "i0_percent": 0.07,
                        "pfe_kw": 14.0,
                        "vkr_percent": 0.41,
                        "sn_mva": 25.0,
                        "vn_lv_kv": 20.0,
                        "vn_hv_kv": 110,
                        "vk_percent": 12.0,
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "tap_max": 9,
                        "tap_step_degree": 0,
                        "tap_step_percent": 1.5,
                        "tap_phase_shifter": False
                    },
                    "63 MVA 110/10 kV": {
                        "i0_percent": 0.04,
                        "pfe_kw": 22.0,
                        "vkr_percent": 0.32,
                        "sn_mva": 63.0,
                        "vn_lv_kv": 10.0,
                        "vn_hv_kv": 110,
                        "vk_percent": 18.0,
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "tap_max": 9,
                        "tap_step_degree": 0,
                        "tap_step_percent": 1.5,
                        "tap_phase_shifter": False
                    },
                    "40 MVA 110/10 kV": {
                        "i0_percent": 0.05,
                        "pfe_kw": 18.0,
                        "vkr_percent": 0.34,
                        "sn_mva": 40.0,
                        "vn_lv_kv": 10.0,
                        "vn_hv_kv": 110,
                        "vk_percent": 16.2,
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "tap_max": 9,
                        "tap_step_degree": 0,
                        "tap_step_percent": 1.5,
                        "tap_phase_shifter": False
                    },
                    "25 MVA 110/10 kV": {
                        "i0_percent": 0.07,
                        "pfe_kw": 14.0,
                        "vkr_percent": 0.41,
                        "sn_mva": 25.0,
                        "vn_lv_kv": 10.0,
                        "vn_hv_kv": 110,
                        "vk_percent": 12.0,
                        "shift_degree": 150,
                        "vector_group": "YNd5",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -9,
                        "tap_max": 9,
                        "tap_step_degree": 0,
                        "tap_step_percent": 1.5,
                        "tap_phase_shifter": False
                    },
                    "0.25 MVA 20/0.4 kV": {
                        "i0_percent": 0.32,
                        "pfe_kw": 0.8,
                        "vkr_percent": 1.44,
                        "sn_mva": 0.25,
                        "vn_lv_kv": 0.4,
                        "vn_hv_kv": 20,
                        "vk_percent": 6.0,
                        "shift_degree": 150,
                        "vector_group": "Yzn5",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -2,
                        "tap_max": 2,
                        "tap_step_degree": 0,
                        "tap_step_percent": 2.5,
                        "tap_phase_shifter": False
                    },
                    "0.4 MVA 20/0.4 kV": {
                        "i0_percent": 0.3375,
                        "pfe_kw": 1.35,
                        "vkr_percent": 1.425,
                        "sn_mva": 0.4,
                        "vn_lv_kv": 0.4,
                        "vn_hv_kv": 20,
                        "vk_percent": 6.0,
                        "shift_degree": 150,
                        "vector_group": "Dyn5",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -2,
                        "tap_max": 2,
                        "tap_step_degree": 0,
                        "tap_step_percent": 2.5,
                        "tap_phase_shifter": False
                    },
                    "0.63 MVA 20/0.4 kV": {
                        "i0_percent": 0.2619,
                        "pfe_kw": 1.65,
                        "vkr_percent": 1.206,
                        "sn_mva": 0.63,
                        "vn_lv_kv": 0.4,
                        "vn_hv_kv": 20,
                        "vk_percent": 6.0,
                        "shift_degree": 150,
                        "vector_group": "Dyn5",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -2,
                        "tap_max": 2,
                        "tap_step_degree": 0,
                        "tap_step_percent": 2.5,
                        "tap_phase_shifter": False
                    },
                    "0.25 MVA 10/0.4 kV": {
                        "i0_percent": 0.24,
                        "pfe_kw": 0.6,
                        "vkr_percent": 1.2,
                        "sn_mva": 0.25,
                        "vn_lv_kv": 0.4,
                        "vn_hv_kv": 10,
                        "vk_percent": 4.0,
                        "shift_degree": 150,
                        "vector_group": "Dyn5",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -2,
                        "tap_max": 2,
                        "tap_step_degree": 0,
                        "tap_step_percent": 2.5,
                        "tap_phase_shifter": False
                    },
                    "0.4 MVA 10/0.4 kV": {
                        "i0_percent": 0.2375,
                        "pfe_kw": 0.95,
                        "vkr_percent": 1.325,
                        "sn_mva": 0.4,
                        "vn_lv_kv": 0.4,
                        "vn_hv_kv": 10,
                        "vk_percent": 4.0,
                        "shift_degree": 150,
                        "vector_group": "Dyn5",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -2,
                        "tap_max": 2,
                        "tap_step_degree": 0,
                        "tap_step_percent": 2.5,
                        "tap_phase_shifter": False
                    },
                    "0.63 MVA 10/0.4 kV": {
                        "i0_percent": 0.1873,
                        "pfe_kw": 1.18,
                        "vkr_percent": 1.0794,
                        "sn_mva": 0.63,
                        "vn_lv_kv": 0.4,
                        "vn_hv_kv": 10,
                        "vk_percent": 4.0,
                        "shift_degree": 150,
                        "vector_group": "Dyn5",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -2,
                        "tap_max": 2,
                        "tap_step_degree": 0,
                        "tap_step_percent": 2.5,
                        "tap_phase_shifter": False
                    }
                },
                "trafo3w": {
                    "63/25/38 MVA 110/20/10 kV": {
                        "sn_hv_mva": 63,
                        "sn_mv_mva": 25,
                        "sn_lv_mva": 38,
                        "vn_hv_kv": 110,
                        "vn_mv_kv": 20,
                        "vn_lv_kv": 10,
                        "vk_hv_percent": 10.4,
                        "vk_mv_percent": 10.4,
                        "vk_lv_percent": 10.4,
                        "vkr_hv_percent": 0.28,
                        "vkr_mv_percent": 0.32,
                        "vkr_lv_percent": 0.35,
                        "pfe_kw": 35,
                        "i0_percent": 0.89,
                        "shift_mv_degree": 0,
                        "shift_lv_degree": 0,
                        "vector_group": "YN0yn0yn0",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -10,
                        "tap_max": 10,
                        "tap_step_percent": 1.2
                    },
                    "63/25/38 MVA 110/10/10 kV": {
                        "sn_hv_mva": 63,
                        "sn_mv_mva": 25,
                        "sn_lv_mva": 38,
                        "vn_hv_kv": 110,
                        "vn_mv_kv": 10,
                        "vn_lv_kv": 10,
                        "vk_hv_percent": 10.4,
                        "vk_mv_percent": 10.4,
                        "vk_lv_percent": 10.4,
                        "vkr_hv_percent": 0.28,
                        "vkr_mv_percent": 0.32,
                        "vkr_lv_percent": 0.35,
                        "pfe_kw": 35,
                        "i0_percent": 0.89,
                        "shift_mv_degree": 0,
                        "shift_lv_degree": 0,
                        "vector_group": "YN0yn0yn0",
                        "tap_side": "hv",
                        "tap_neutral": 0,
                        "tap_min": -10,
                        "tap_max": 10,
                        "tap_step_percent": 1.2
                    }
                }
            },
            "res_bus": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"vm_pu\",\"va_degree\",\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "vm_pu": "float64",
                    "va_degree": "float64",
                    "p_mw": "float64",
                    "q_mvar": "float64"
                },
                "orient": "split"
            },
            "res_line": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_from_mw\",\"q_from_mvar\",\"p_to_mw\",\"q_to_mvar\",\"pl_mw\",\"ql_mvar\",\"i_from_ka\",\"i_to_ka\",\"i_ka\",\"vm_from_pu\",\"va_from_degree\",\"vm_to_pu\",\"va_to_degree\",\"loading_percent\"],\"index\":[],\"data\":[]}",
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
                },
                "orient": "split"
            },
            "res_trafo": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_hv_mw\",\"q_hv_mvar\",\"p_lv_mw\",\"q_lv_mvar\",\"pl_mw\",\"ql_mvar\",\"i_hv_ka\",\"i_lv_ka\",\"vm_hv_pu\",\"va_hv_degree\",\"vm_lv_pu\",\"va_lv_degree\",\"loading_percent\"],\"index\":[],\"data\":[]}",
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
                },
                "orient": "split"
            },
            "res_trafo3w": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_hv_mw\",\"q_hv_mvar\",\"p_mv_mw\",\"q_mv_mvar\",\"p_lv_mw\",\"q_lv_mvar\",\"pl_mw\",\"ql_mvar\",\"i_hv_ka\",\"i_mv_ka\",\"i_lv_ka\",\"vm_hv_pu\",\"va_hv_degree\",\"vm_mv_pu\",\"va_mv_degree\",\"vm_lv_pu\",\"va_lv_degree\",\"va_internal_degree\",\"vm_internal_pu\",\"loading_percent\"],\"index\":[],\"data\":[]}",
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
                },
                "orient": "split"
            },
            "res_impedance": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_from_mw\",\"q_from_mvar\",\"p_to_mw\",\"q_to_mvar\",\"pl_mw\",\"ql_mvar\",\"i_from_ka\",\"i_to_ka\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "p_from_mw": "float64",
                    "q_from_mvar": "float64",
                    "p_to_mw": "float64",
                    "q_to_mvar": "float64",
                    "pl_mw": "float64",
                    "ql_mvar": "float64",
                    "i_from_ka": "float64",
                    "i_to_ka": "float64"
                },
                "orient": "split"
            },
            "res_ext_grid": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64"
                },
                "orient": "split"
            },
            "res_load": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64"
                },
                "orient": "split"
            },
            "res_sgen": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64"
                },
                "orient": "split"
            },
            "res_storage": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64"
                },
                "orient": "split"
            },
            "res_shunt": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\",\"vm_pu\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64",
                    "vm_pu": "float64"
                },
                "orient": "split"
            },
            "res_gen": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\",\"va_degree\",\"vm_pu\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64",
                    "va_degree": "float64",
                    "vm_pu": "float64"
                },
                "orient": "split"
            },
            "res_ward": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\",\"vm_pu\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64",
                    "vm_pu": "float64"
                },
                "orient": "split"
            },
            "res_xward": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_mw\",\"q_mvar\",\"vm_pu\",\"va_internal_degree\",\"vm_internal_pu\"],\"index\":[],\"data\":[]}",
                "dtype": {
                    "p_mw": "float64",
                    "q_mvar": "float64",
                    "vm_pu": "float64",
                    "va_internal_degree": "float64",
                    "vm_internal_pu": "float64"
                },
                "orient": "split"
            },
            "res_dcline": {
                "_module": "pandas.core.frame",
                "_class": "DataFrame",
                "_object": "{\"columns\":[\"p_from_mw\",\"q_from_mvar\",\"p_to_mw\",\"q_to_mvar\",\"pl_mw\",\"vm_from_pu\",\"va_from_degree\",\"vm_to_pu\",\"va_to_degree\"],\"index\":[],\"data\":[]}",
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
                },
                "orient": "split"
            },
            "user_pf_options": {

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

# print(" [x] Trying rabbitmq")
# url = os.environ.get('CLOUDAMQP_URL', 'amqps://jdzlpput:5ny6ANo8vdhwr8iYkwVXd_8sRwyIKLBi@rattlesnake.rmq.cloudamqp.com/jdzlpput')
# params = pika.URLParameters(url)
# connection = pika.BlockingConnection(params)
# channel = connection.channel()
# channel.queue_declare(queue='mlst_iim')
# channel.basic_publish(exchange='', routing_key='mlst_iim', body=json.dumps(rmq_json))

# print(" [x] Sent test")
# connection.close()

credentials = pika.PlainCredentials('iim-guest', 'iimguest')
# parameters = pika.ConnectionParameters('3.120.35.154', 5672, 'iim', credentials)
parameters = pika.ConnectionParameters('rabbit.prod.gridpilot.tech', 5672, 'iim', credentials)
connection = pika.BlockingConnection(parameters)

channel = connection.channel()
channel.queue_declare(queue='IIM#IIM')

channel.basic_publish(exchange='Islanding_Exchange.headers', routing_key ='IIM#IIM', body = json.dumps(rmq_json))

print(" [x] Sent 'Islanding scheme!'")
connection.close()


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