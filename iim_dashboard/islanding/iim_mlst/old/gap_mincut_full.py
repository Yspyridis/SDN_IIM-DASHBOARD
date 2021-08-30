
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
# import plotly.io as pio
# pio.renderers.default = "browser"
try:
    import seaborn
    colors = seaborn.color_palette(n_colors=50)
except:
    colors = ["b", "g", "r", "c", "y"]
# from pandapower.plotting.plotly import simple_plotly,simple_plotly_gen
from utlize import simple_plotly_gen

import csv


def evaluation_mini_imblance(X_hat,output_predict,bus_inc):
    bus_inc=torch.from_numpy(bus_inc).to(device).unsqueeze(1).float()
    # output_predict = torch.argmax(X_hat, dim=1)
    predict_binary_matrix=torch.zeros_like(X_hat)
    for i  in range(len(output_predict)):
        predict_binary_matrix[i,output_predict[i]]=1
    # island_inc = torch.abs(torch.mm(predict_binary_matrix.t(), bus_inc))
    island_inc = torch.mm(predict_binary_matrix.t(), bus_inc)
    sum_inc=torch.sum(island_inc)
    # sum_inc=torch.abs(torch.sum(island_inc))

    return sum_inc,island_inc





def plot_net_with_label(net):
    lc = plot.create_line_collection(net, net.line.index, color='black', zorder=0) #create lines
    # bc = plot.create_bus_collection(net, net.bus.index, size=80, color=colors[0], zorder=1) #create buses
    labels=list(range(k))
    color_colection_c=[]
    count=1
    for lable in labels:
        single_color=np.where(output_predict==lable)
        color_index=np.array(net.bus.index[single_color])
        color_colection_c = color_colection_c + [plot.create_bus_collection(net, color_index, size=0.1, color=colors[count], zorder=count)]
        count+=1
    if len(h_index_trans)>0:
        lc_trans,pc_trans = plot.create_trafo_collection(net, net.trafo.index, size=0.1,color=colors[-2], zorder=1) #create lines

        c=[lc]+[lc_trans]+[pc_trans]+color_colection_c
    else:
        c = [lc] + color_colection_c
    plot.draw_collections(c, figsize=(5,8))
    plt.show()
    plt.savefig('colored.png')

def euclidean_distances(x, all_vec):
    dist=np.zeros(len(all_vec))
    for i in range(len(all_vec)):
        y=all_vec[i]
        dist[i]=np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
    return dist
#part-test case


# net=pp.networks.case9()
# net=pp.networks.case14()
# net=pp.networks.case24_ieee_rts()
# net=pp.networks.case30()
net=pp.networks.case_ieee30()

# net=pp.networks.case39()
# net=pp.networks.case57()
# net=pp.networks.case89pegase()
# net=pp.networks.case118()
# net=pp.networks.case145()
# net=pp.networks.case_illinois200()


# plot_net_with_label(net)
# test_file_path='C:/Users/70474/Desktop/meeting_phd/project_va/GridModels_examples/CIGRE_MV_modified.json'
# net= pp.from_json(test_file_path)

# net = pp.from_json('/home/yannis/Documents/Code/IIM_Dashboard/SDN_IIM-DASHBOARD/iim_dashboard/CaseIEEE9_GridModel.json')
# net = pp.from_json('/home/yannis/Documents/Code/IIM_Dashboard/SDN_IIM-DASHBOARD/iim_dashboard/MV_CIGRE_GridModel.json')
# net = pp.from_json('/home/yannis/Documents/Code/IIM_Dashboard/SDN_IIM-DASHBOARD/iim_dashboard/MV_Oberrhein_GridModelv2.json')

# print('pp.rundcpp(net)')
# print(pp.rundcpp(net))
# print(net.res_load)
# print(net.res_sgen)
# print('#######################################')
# print('#######################################')



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


# k=2
# num_epoch=11000   #number of training time
# num_epoch=1000   #number of training time



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



#generator
gen=net.gen

num_epoch=1000
k = 4

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
# model = Graph_cnn_model.GraphEncoder(layers, k,La,len(X_train[0]),cluster_num=k).to(device)
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
output_predict=torch.argmax(X_hat,dim=1)
output_predict=output_predict.cpu().numpy()
# generator_results=output_predict[gen_index]
# print(gen_index)


#part- evaluation
sum_inc,island_inc=evaluation_mini_imblance(X_hat,output_predict,bus_inc)
for i in range(len(island_inc)):
    print('island imbalance ' +str(i)+': '+str(island_inc[i]))
print('total island imbalance= '+str(sum_inc) )

#part- plot
# plot_net_with_label(net)
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
plot_net_with_label(net)

#part- evaluation again on the result after cutting
sum_inc,island_inc=evaluation_mini_imblance(X_hat,output_predict,bus_inc)
for i in range(len(island_inc)):
    print('island imbalance ' +str(i)+': '+str(island_inc[i]))
print('total island imbalance= '+str(sum_inc) )
print('number of lines to cut= '+str(number_lines))
#part-run pp to test the result
pp.runpp(net)
#plot the result with generator
# simple_plotly_gen(net)
# print(net.res_bus)

# pp.plotting.simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
# plt.savefig('net.png')

# print('pp.rundcpp(net)')
# print(pp.rundcpp(net))
# print(net.res_load)
# print(net.res_sgen)
# net.res_load.to_csv('load.csv')
# net.res_sgen.to_csv('sgen.csv')
# print('#######################################')
# print('#######################################')


