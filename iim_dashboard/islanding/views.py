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

from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser 
from rest_framework import status
 
from islanding.serializers import IslandingSchemeSerializer
from rest_framework.decorators import api_view


@api_view(['GET'])
def islanding_plot(request):
                      
    method = request.GET['method']
    print(method)

    # ######### Get grid json and plot image of result ##########
    mygrid = IslandingScheme.objects.filter(method_name=method).last().grid

    with open('islanding/iim_mlst/static/grid_after_islanding/tmp_grid.txt', 'w') as f:
        f.write(mygrid)

    net_after = pp.from_json('islanding/iim_mlst/static/grid_after_islanding/tmp_grid.txt')

    pp.plotting.simple_plot(net_after, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0, trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0, switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True, bus_color='b', line_color='grey', trafo_color='k', ext_grid_color='y', switch_color='k', library='igraph', show_plot=False, ax=None)
    plt.savefig('islanding/iim_mlst/static/grid_after_islanding/grid_after_'+method+'.png')

    pp.plotting.to_html(net_after, filename='islanding/iim_mlst/static/grid_after_islanding/interactive-plot_'+method+'.html', show_tables=False)

    os.remove('islanding/iim_mlst/static/grid_after_islanding/tmp_grid.txt')
    ###########################################################
    
    return HttpResponse("DONE")

# @api_view(['GET', 'POST', 'DELETE'])
@api_view(['GET', 'POST'])
def islanding_result(request):
        
    if request.method == 'GET':
        
        ################ Get the numerical results #################
        # Get all fields amd remove grid because it is huge and not needed here
        fields = [f.name for f in IslandingScheme._meta.get_fields()]
        fields.remove('grid')
    
        # Now query the results without the grid
        # results = IslandingScheme.objects.filter(method=method).order_by('-id')[:1].values(*fields)
        results = IslandingScheme.objects.all().order_by('-id')[:2].values(*fields)
        df = pd.DataFrame(data=results)
        response_json = df.to_json(orient='records')
        ############################################################
    
        return HttpResponse(response_json, content_type='application/json')
        
    elif request.method == 'POST':
                
        islanding_data = JSONParser().parse(request)
        islanding_serializer = IslandingSchemeSerializer(data=islanding_data)
        if islanding_serializer.is_valid():
            islanding_serializer.save()
            return JsonResponse(islanding_serializer.data, status=status.HTTP_201_CREATED) 
        return JsonResponse(islanding_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
    # elif request.method == 'DELETE':
    #     count = Islanding.objects.all().delete()
    #     return JsonResponse({'message': '{} Tutorials were deleted successfully!'.format(count[0])}, status=status.HTTP_204_NO_CONTENT)
 