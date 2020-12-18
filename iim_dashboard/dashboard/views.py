from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

from islanding.models import IslandingScheme
import os
# path = os.path.dirname(os.path.realpath(__file__))
# menu = str(path) + "/visualization/Models/PPC/analytics_menu.csv"
import pandapower as pp
import pandapower.plotting as plot
import matplotlib.pyplot as plt

# import utlize
# from utlize import simple_plotly_gen

from pandapower.plotting.plotly import simple_plotly

from django.core import serializers
import pandas as pd
import json


def index(request):

    context = {
    'app_name': "dashboard",
    # 'page_name': "overview",
    }
    template = loader.get_template('dashboard/index.html')
    return HttpResponse(template.render(context, request))

def compare(request):

    context = {
    'app_name': "dashboard",
    'page_name': "compare",
    }
    template = loader.get_template('dashboard/compare.html')
    return HttpResponse(template.render(context, request))


# def islanding_result(request):

#     method = request.GET['method']
#     response = 'islanding/iim_mlst/static/grid_after_islanding/grid_after_'+method+'.png'

#     return HttpResponse(response)


# def save_islanding_result(request):
    
#     grid = request.POST



# def islanding_result(request):

# def islanding_result(request):
#     # method = request.GET['method']
#     # print(method)
    
#     ########## Get grid json and plot image of result ##########
#     # mygrid = IslandingScheme.objects.filter(method=method).last().grid

#     # with open('islanding/iim_mlst/static/grid_after_islanding/tmp_grid.txt', 'w') as f:
#         # f.write(mygrid)

#     # net_after = pp.from_json('islanding/iim_mlst/static/grid_after_islanding/tmp_grid.txt')

#     # pp.plotting.simple_plot(net_after, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
#     # plt.savefig('islanding/iim_mlst/static/grid_after_islanding/grid_after_'+method+'.png')

#     # pp.plotting.to_html(net_after, filename='islanding/iim_mlst/static/grid_after_islanding/interactive-plot2.html', show_tables=False)

#     # os.remove('islanding/iim_mlst/static/grid_after_islanding/tmp_grid.txt')
#     ############################################################


# def islanding_result(request):

#     ################ Get the numerical results #################
#     # Get all fields amd remove grid because it is huge and not needed here
#     fields = [f.name for f in IslandingScheme._meta.get_fields()]
#     fields.remove('grid')

#     # Now query the results without the grid
#     # results = IslandingScheme.objects.filter(method=method).order_by('-id')[:1].values(*fields)
#     results = IslandingScheme.objects.all().order_by('-id')[:2].values(*fields)
#     df = pd.DataFrame(data=results)
#     response_json = df.to_json(orient='records')
#     ############################################################

#     return HttpResponse(response_json, content_type='application/json')