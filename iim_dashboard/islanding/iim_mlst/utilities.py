import numpy as np
import torch
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import pandas as pd
from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapower.plotting.plotly.traces import create_bus_trace, create_line_trace, \
    create_trafo_trace, draw_traces, version_check
from pandapower.plotting.plotly.mapbox_plot import *

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


def get_hoverinfo(net, element, precision=3, sub_index=None):
    hover_index = net[element].index
    if element == "bus":
        load_str, sgen_str = [], []
        for ln in [net.load.loc[net.load.bus == b, "p_mw"].sum() for b in net.bus.index]:
            load_str.append("Load: {:.3f} MW<br />".format(ln) if ln != 0. else "")
        for s in [net.sgen.loc[net.sgen.bus == b, "p_mw"].sum() for b in net.bus.index]:
            sgen_str.append("Static generation: {:.3f} MW<br />".format(s) if s != 0. else "")
        hoverinfo = (
                "Index: " + net.bus.index.astype(str) + '<br />' +
                "Name: " + net.bus['name'].astype(str) + '<br />' +
                'V_n: ' + net.bus['vn_kv'].round(precision).astype(str) + ' kV' + '<br />' + load_str + sgen_str)\
            .tolist()
    elif element == "line":
        hoverinfo = (
                "Index: " + net.line.index.astype(str) + '<br />' +
                "Name: " + net.line['name'].astype(str) + '<br />' +
                'Length: ' + net.line['length_km'].round(precision).astype(str) + ' km' + '<br />' +
                'R: ' + (net.line['length_km'] * net.line['r_ohm_per_km']).round(precision).astype(str)
                + ' Ohm' + '<br />'
                + 'X: ' + (net.line['length_km'] * net.line['x_ohm_per_km']).round(precision).astype(str)
                + ' Ohm' + '<br />').tolist()
    elif element == "trafo":
        hoverinfo = (
                "Index: " + net.trafo.index.astype(str) + '<br />' +
                "Name: " + net.trafo['name'].astype(str) + '<br />' +
                'V_n HV: ' + net.trafo['vn_hv_kv'].round(precision).astype(str) + ' kV' + '<br />' +
                'V_n LV: ' + net.trafo['vn_lv_kv'].round(precision).astype(str) + ' kV' + '<br />' +
                'Tap pos.: ' + net.trafo['tap_pos'].astype(str) + '<br />').tolist()
    elif element == "ext_grid":
        hoverinfo = (
                "Index: " + net.ext_grid.index.astype(str) + '<br />' +
                "Name: " + net.ext_grid['name'].astype(str) + '<br />' +
                'V_m: ' + net.ext_grid['vm_pu'].round(precision).astype(str) + ' p.u.' + '<br />' +
                'V_a: ' + net.ext_grid['va_degree'].round(precision).astype(str) + ' °' + '<br />').tolist()
        hover_index = net.ext_grid.bus.tolist()
    else:
        return None
    hoverinfo = pd.Series(index=hover_index, data=hoverinfo)
    if sub_index is not None:
        hoverinfo = hoverinfo.loc[list(sub_index)]
    return hoverinfo


def simple_plotly(net, respect_switches=True, use_line_geodata=None, on_map=False,
                  projection=None, map_style='basic', figsize=1, aspectratio='auto', line_width=1,
                  bus_size=10, ext_grid_size=20.0, bus_color="blue", line_color='grey',
                  trafo_color='green', ext_grid_color="yellow"):
    """
    Plots a pandapower network as simple as possible in plotly.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the tutorial

    INPUT:
        **net** - The pandapower format network. If none is provided, mv_oberrhein() will be
            plotted as an example

    OPTIONAL:
        **respect_switches** (bool, True) - Respect switches when artificial geodata is created

        *use_line_geodata** (bool, True) - defines if lines patches are based on net.line_geodata of the lines (True)
            or on net.bus_geodata of the connected buses (False)

        **on_map** (bool, False) - enables using mapbox plot in plotly.
            If provided geodata are not real geo-coordinates in lon/lat form, on_map will be set to False.

        **projection** (String, None) - defines a projection from which network geo-data will be transformed to
            lat-long. For each projection a string can be found at http://spatialreference.org/ref/epsg/


        **map_style** (str, 'basic') - enables using mapbox plot in plotly

            - 'streets'
            - 'bright'
            - 'light'
            - 'dark'
            - 'satellite'

        **figsize** (float, 1) - aspectratio is multiplied by it in order to get final image size

        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the network geodata;
            any custom aspectration can be given as a tuple, e.g. (1.2, 1)

        **line_width** (float, 1.0) - width of lines

        **bus_size** (float, 10.0) -  size of buses to plot.

        **ext_grid_size** (float, 20.0) - size of ext_grids to plot.

            See bus sizes for details. Note: ext_grids are plotted as rectangles

        **bus_color** (String, "blue") - Bus Color. Init as first value of color palette.

        **line_color** (String, 'grey') - Line Color. Init is grey

        **trafo_color** (String, 'green') - Trafo Color. Init is green

        **ext_grid_color** (String, 'yellow') - External Grid Color. Init is yellow

    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object
    """
    version_check()
    # create geocoord if none are available
    if 'line_geodata' not in net:
        net.line_geodata = pd.DataFrame(columns=['coords'])
    if 'bus_geodata' not in net:
        net.bus_geodata = pd.DataFrame(columns=["x", "y"])
    if len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time...")
        create_generic_coordinates(net, respect_switches=respect_switches)
        if on_map:
            logger.warning("Map plots not available with artificial coordinates and will be disabled!")
            on_map = False

    # check if geodata are real geographycal lat/lon coordinates using geopy
    if on_map and projection is not None:
        geo_data_to_latlong(net, projection=projection)

    # ----- Buses ------
    # initializating bus trace
    hoverinfo = get_hoverinfo(net, element="bus")
    bus_trace = create_bus_trace(net, net.bus.index, size=bus_size, color=bus_color, infofunc=hoverinfo)

    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    if use_line_geodata is None:
        use_line_geodata = False if len(net.line_geodata) == 0 else True
    elif use_line_geodata and len(net.line_geodata) == 0:
        logger.warning("No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geodata = False

    hoverinfo = get_hoverinfo(net, element="line")
    line_traces = create_line_trace(net, net.line.index, respect_switches=respect_switches,
                                    color=line_color, width=line_width,
                                    use_line_geodata=use_line_geodata, infofunc=hoverinfo)

    # ----- Trafos ------
    hoverinfo = get_hoverinfo(net, element="trafo")
    trafo_trace = create_trafo_trace(net, color=trafo_color, width=line_width * 5, infofunc=hoverinfo,
                                     use_line_geodata=use_line_geodata)

    # ----- Ext grid ------
    # get external grid from create_bus_trace
    marker_type = 'circle' if on_map else 'square'  # workaround because doesn't appear on mapbox if square
    hoverinfo = get_hoverinfo(net, element="ext_grid")
    ext_grid_trace = create_bus_trace(net, buses=net.ext_grid.bus,
                                      color=ext_grid_color, size=ext_grid_size,
                                      patch_type=marker_type, trace_name='external_grid', infofunc=hoverinfo)

    return draw_traces(line_traces + trafo_trace + ext_grid_trace + bus_trace,
                       aspectratio=aspectratio, figsize=figsize, on_map=on_map, map_style=map_style)


def simple_plotly_gen(net, respect_switches=True, use_line_geodata=None, on_map=False,
                  projection=None, map_style='basic', figsize=1, aspectratio='auto', line_width=1,
                  bus_size=10, ext_grid_size=20.0, bus_color="blue", line_color='grey',
                  trafo_color='green', ext_grid_color="yellow",file_name=''):
    """
    Plots a pandapower network as simple as possible in plotly.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the tutorial

    INPUT:
        **net** - The pandapower format network. If none is provided, mv_oberrhein() will be
            plotted as an example

    OPTIONAL:
        **respect_switches** (bool, True) - Respect switches when artificial geodata is created

        *use_line_geodata** (bool, True) - defines if lines patches are based on net.line_geodata of the lines (True)
            or on net.bus_geodata of the connected buses (False)

        **on_map** (bool, False) - enables using mapbox plot in plotly.
            If provided geodata are not real geo-coordinates in lon/lat form, on_map will be set to False.

        **projection** (String, None) - defines a projection from which network geo-data will be transformed to
            lat-long. For each projection a string can be found at http://spatialreference.org/ref/epsg/


        **map_style** (str, 'basic') - enables using mapbox plot in plotly

            - 'streets'
            - 'bright'
            - 'light'
            - 'dark'
            - 'satellite'

        **figsize** (float, 1) - aspectratio is multiplied by it in order to get final image size

        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the network geodata;
            any custom aspectration can be given as a tuple, e.g. (1.2, 1)

        **line_width** (float, 1.0) - width of lines

        **bus_size** (float, 10.0) -  size of buses to plot.

        **ext_grid_size** (float, 20.0) - size of ext_grids to plot.

            See bus sizes for details. Note: ext_grids are plotted as rectangles

        **bus_color** (String, "blue") - Bus Color. Init as first value of color palette.

        **line_color** (String, 'grey') - Line Color. Init is grey

        **trafo_color** (String, 'green') - Trafo Color. Init is green

        **ext_grid_color** (String, 'yellow') - External Grid Color. Init is yellow

    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object
    """
    version_check()
    # create geocoord if none are available
    if 'line_geodata' not in net:
        net.line_geodata = pd.DataFrame(columns=['coords'])
    if 'bus_geodata' not in net:
        net.bus_geodata = pd.DataFrame(columns=["x", "y"])
    if len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time...")
        create_generic_coordinates(net, respect_switches=respect_switches)
        if on_map:
            logger.warning("Map plots not available with artificial coordinates and will be disabled!")
            on_map = False

    # check if geodata are real geographycal lat/lon coordinates using geopy
    if on_map and projection is not None:
        geo_data_to_latlong(net, projection=projection)

    # ----- Buses ------
    # initializating bus trace
    hoverinfo = get_hoverinfo(net, element="bus")
    bus_trace = create_bus_trace(net, net.bus.index, size=bus_size, color=bus_color, infofunc=hoverinfo)

    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    if use_line_geodata is None:
        use_line_geodata = False if len(net.line_geodata) == 0 else True
    elif use_line_geodata and len(net.line_geodata) == 0:
        logger.warning("No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geodata = False

    hoverinfo = get_hoverinfo(net, element="line")
    line_traces = create_line_trace(net, net.line.index, respect_switches=respect_switches,
                                    color=line_color, width=line_width,
                                    use_line_geodata=use_line_geodata, infofunc=hoverinfo)

    # ----- Trafos ------
    hoverinfo = get_hoverinfo(net, element="trafo")
    trafo_trace = create_trafo_trace(net, color=trafo_color, width=line_width * 5, infofunc=hoverinfo,
                                     use_line_geodata=use_line_geodata)

    # ----- Ext grid ------
    # get external grid from create_bus_trace
    marker_type = 'circle' if on_map else 'square'  # workaround because doesn't appear on mapbox if square
    hoverinfo = get_hoverinfo(net, element="gen")
    ext_grid_trace = create_bus_trace(net, buses=net.gen.bus,
                                      color=ext_grid_color, size=ext_grid_size,
                                      patch_type=marker_type, trace_name='external_grid', infofunc=hoverinfo)

    return draw_traces(line_traces + trafo_trace + ext_grid_trace + bus_trace,
                       aspectratio=aspectratio, figsize=figsize, on_map=on_map, map_style=map_style,filename=file_name)

def total_Progressive(net,gen_ids,switch_id,features,adj_matrix, number_clusters=None):

    init_centers=features[gen_ids]
    number_clusters=len(gen_ids)
    unlabel_index=list(range(len(features)))
    labeled_index=[]
    label_result=np.zeros(len(gen_ids),len(features))
    current
    while len(unlabel_index)!=0:
        for i in range(len(gen_ids)):
            label_result[i]


def csc2sparsetensor(csc):
    output_matrix=[]
    # for csc in csc_matrix:
    coo=csc.tocoo()
    values = coo.data
    indices = np.int64(np.vstack((coo.row, coo.col)))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    output_matrix=torch.sparse.FloatTensor(i, v, torch.Size(shape))#.to_dense()

    return output_matrix
def weighted_KMeans(data,generator_index, k=2,theta_gen=0.7,max_time_step=100):
    def _distance(p1, p2):
        """
        Return Eclud distance between two points.
        p1 = np.array([0,0]), p2 = np.array([1,1]) => 1.414
        """
        tmp = np.sum((p1 - p2) ** 2)
        return np.sqrt(tmp)

    def _rand_center(data, k):
        """Generate k center within the range of data set."""
        n = data.shape[1]  # features
        centroids = np.zeros((k, n))  # init with (0,0)....
        for i in range(n):
            dmin, dmax = np.min(data[:, i]), np.max(data[:, i])
            centroids[:, i] = dmin + (dmax - dmin) * np.random.rand(k)
        return centroids
    def generator_center(data,gen_index):
        centroids=data[gen_index,:]
        return centroids
    def _converged(centroids1, centroids2):

        # if centroids not changed, we say 'converged'
        set1 = set([tuple(c) for c in centroids1])
        set2 = set([tuple(c) for c in centroids2])
        return (set1 == set2)
    k=len(generator_index)
    n = data.shape[0]  # number of entries
    rest_importance=(1-theta_gen)/np.float(n-1)
    # centroids = _rand_center(data, k)
    centroids=generator_center(data,generator_index)
    label = np.zeros(n, dtype=np.int)  # track the nearest centroid
    assement = np.zeros(n)  # for the assement of our model
    converged = False

    while not converged:
        old_centroids = np.copy(centroids)
        for i in range(n):
            # determine the nearest centroid and track it with label
            min_dist, min_index = np.inf, -1
            for j in range(k):
                dist=_distance(data[i], centroids[j])
                if dist < min_dist:
                    min_dist, min_index = dist, j
                    label[i] = j
        label[generator_index]=np.arange(0,k)

        # update centroid
        for m in range(k):
            centroids[m]=((theta_gen-rest_importance)*data[generator_index[m]] + rest_importance*np.sum(data[label == m],axis=0))/(1.0*n)
        converged = _converged(old_centroids, centroids)
    return centroids, label


def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L