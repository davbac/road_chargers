import geopandas as gpd
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import igraph
import matplotlib as mpl

def prepare_network(mf, cf, nf, display = False):
    """
    arguments: 
        mf: str         file (.shp) -- read
        cf: str         file (.csv) -- read
        nf: str         network file (pickle) -- write
        display: bool   enables plotting graphs
    """
    
    shapefile = gpd.read_file(mf) ## read map
    chargedata = pd.read_csv(cf)  ## read charger info
    
    """ ### show data """
    if display:
        ## shows map of the roads
        for l in shapefile.geometry:
            _ = plt.plot(*l.coords.xy)
        
        plt.plot("X","Y", marker="o", lw=0, data=chargedata, color="black")
        
        plt.show()
    
    
    """ ### find min distances between chargers 
    l = len(chargedata)
    dist_mat = [[((chargedata["X"][i]-chargedata["X"][j])**2 + (chargedata["Y"][i]-chargedata["Y"][j])**2)**0.5 for i in range(l)] for j in range(l)]
    for i in range(l):
        dist_mat[i][i] = float("inf") 
        ## set self-distance to infinity so it doesn't register as shortest distance
        
    mindists = np.min(dist_mat,axis=0)
    plt.hist(mindists)
    plt.show()
    """
    
    """ ### convert to graph """
    starts = []
    ends = []
    matches = []
    lengths = []
    ids = []
    for l in shapefile.geometry:
        ## cycle through the roads
        
        xy = l.coords.xy
        dists = [0]
        dists.extend(np.cumsum((np.diff(xy, axis=1)**2).sum(axis=0)**0.5))
        ## going through the road, we calculate eacch point's distance from the road's origin 
        ##      by cumulatively summing the distance found with pythagoras theorem
        
        match_ind = []
        
        for i in range(len(chargedata)):
            ## check if any chargers are on this road
            
            if i in matches:
                ## skip, already assigned to another road
                continue
                
            cx = chargedata["X"][i]
            cy = chargedata["Y"][i]
            d = ((xy[0]-cx)**2 + (xy[1]-cy)**2)**0.5
            
            if np.any(d<50): ## allow 50m tolerance when assigning charger to road
                # matches.append([i, len(starts)])
                matches.append(i)
                match_ind.append((d.tolist().index(d.min()), i)) ## still match to closest point
                # print(i, len(starts))
                
        prevd = 0
        match_ind = sorted(match_ind) 
        ## will sort using the first element, ie distance from the street's origin
        
        ## now add the street's start and endpoints to the global list, but making sure to break the road into segments where we have chargers
        s = [[xy[0][0], xy[1][0], False]]
        e = []
        for i in range(len(match_ind)):
            ind = match_ind[i][1]
            s.append([chargedata["X"][ind],chargedata["Y"][ind], True])
            e.append([chargedata["X"][ind],chargedata["Y"][ind], True])
            lengths.append(dists[match_ind[i][0]]-prevd)
            prevd = dists[match_ind[i][0]]
            ids.append(shapefile[shapefile.geometry == l]["objectid"].values[0])
        e.append([xy[0][-1], xy[1][-1], False])
        lengths.append(dists[-1]-prevd)
        starts.extend(s)
        ends.extend(e)
        #print(shapefile[shapefile.geometry == l]["objectid"])
        ids.append(shapefile[shapefile.geometry == l]["objectid"].values[0])
    
    ## truncate with a 2m precision
    starts = (np.ceil(np.array(starts)/2)*2).tolist()
    ends = (np.ceil(np.array(ends)/2)*2).tolist()
    
    node_list = np.vstack((starts, ends))
    node_list = np.unique(node_list, axis=0).tolist()
    
    # print(len(starts), len(node_list))
    edges = []
    for i in range(len(starts)):
        edges.append([node_list.index(starts[i]), node_list.index(ends[i])])
    
    node_locations = [[i[0],i[1]] for i in node_list]
    node_is_charger = [bool(i[2]) for i in node_list]
    graph = igraph.Graph(edges)
    graph.vs.set_attribute_values("locations", node_locations)
    graph.vs.set_attribute_values("is_charger", node_is_charger)
    graph.es.set_attribute_values("weight", lengths)
    graph.es.set_attribute_values("id", ids)
    graph.save(nf, "pickle") ## save network info
    
    if display:
        colors = ["red", "blue"]
        layout = igraph.Layout(node_locations)
        layout.mirror(1) ## puts the origin on bottom-left corner as expected in the map 
        pl = igraph.plot(graph, layout=layout,
                        vertex_size = [10 if i else 5 for i in node_is_charger],
                        vertex_color = [colors[int(i)] for i in node_is_charger])
        pl.save("prova_colors.png")
    


def set_distance_info(nf, nf2, display = False):
    """
    arguments:
        nf: str         network file (pickle) -- read
        nf2: str        network file with distances (pickle) -- write
        display: bool   enables plotting graphs
    """
    graph = igraph.Graph.Read_Pickle(nf)
    graph.vs.set_attribute_values("distance", [float("inf") for i in range(len(graph.vs))])
    for i in range(len(graph.vs)):
        if graph.vs[i]["is_charger"] == 1:
            graph.vs[i]["distance"] = 0
    
    ## the distance value is now set up to be infinite at all nodes except the chargers;
    ## at the chargers it is 0
    
    charger_inds = np.arange(len(graph.vs))[np.bool_(graph.vs["is_charger"])] 
    ## get indices where it's true
    
    ## spread the distance information, breadth-first
    active = set(charger_inds)
    while len(active)>0:
        newactive = set()
        for i in active:
            edges = graph.es.select(_source=i) 
            ## in an undirected graph it is the same as using _target=i
            
            tospread = []
            for e in edges:
                j = e.target if e.target != i else e.source
                if graph.vs[i]["distance"] + e["weight"] < graph.vs[j]["distance"]:
                    graph.vs[j]["distance"] = graph.vs[i]["distance"] + e["weight"]
                    tospread.append(j)
            newactive = newactive.union(tospread)
        
        # print(newactive)
        active = newactive
        
    
    graph.save(nf2, "pickle")
    
    if display:
        layout = igraph.Layout(graph.vs["locations"])
        layout.mirror(1)
        cw = mpl.colormaps.get("coolwarm_r")
        max_dist = np.max(graph.vs["distance"], where=np.logical_not(np.isinf(graph.vs["distance"])), initial=0)
        vert_cols = cw(graph.vs["distance"]/max_dist).tolist()
        for i in np.arange(len(graph.vs))[np.isinf(graph.vs["distance"])]:
            vert_cols[i] = [0,1,0]
        
        pl = igraph.plot(graph, layout=layout, bbox=(4000,4000),
                            vertex_size = [30 if i else 10 for i in graph.vs["is_charger"]],
                            vertex_color = vert_cols)
        
        pl.save("prova_dist.png")
    

def analyze(mf,nf,pf, display = False):
    """
    arguments:
        mf: str         map file (.shp) -- read
        nf: str         network file with distances (pickle) -- read
        pf: str         points file (.csv) -- write
        display: bool   enables plotting graphs
    """
    
    graph = igraph.Graph.Read_Pickle(nf)
    shapefile = gpd.read_file(mf)
    
    """ ### per-road statistics 
    shapefile["mindist"] = None
    shapefile["maxdist"] = None
    
    mindists = []
    maxdists = []
    for e in graph.es:
        if e["weight"]<100:
            ## ignore roads less than 100m long
            continue
        
        sd = graph.vs[e.source]["distance"]
        td = graph.vs[e.target]["distance"]
        if np.isinf(sd) or np.isinf(td):
            continue
        
        
        
        
        l = e["weight"]
        mindists.append(min(sd, td))
        x=0.5 + (td-sd)/(2*l)
        #print(sd + x*l == td + (1-x)*l) ## should be true
        
        if x>1: ## shouldnt happen, but to be 100% sure
            x=1
        elif x<0:
            x=0
        maxdists.append(sd + x*l)
        
        for i in range(len(shapefile)):
            if shapefile["objectid"][i] == e["id"]:
                shapefile["maxdist"][i] = maxdists[-1]
                shapefile["mindist"][i] = mindists[-1]
                break
            
    shapefile.to_file("min_max_distance_roads.shp")
    
    if display:
        M = max(maxdists)
        bins = [i*M/20 for i in range(20)] + [M] 
        plt.hist(maxdists, bins = bins, color = "#dd1111cc", label="max distance")
        plt.hist(mindists, bins = bins, color = "#1111ddcc", label="min distance")
        plt.xlabel("distance (m)")
        plt.legend(loc="upper right")
        plt.title("Distribution of minimum and maximum distance on a per-road basis")
        plt.show()
        plt.hist(np.array(maxdists)-np.array(mindists), bins=15)
        plt.title("Distribution of the difference between maximum\n and minimum distance on a per-road basis")
        plt.xlabel("distance (m)")
        plt.show()
    """
    
    """ ### find statistic for points every <scale> meters on each road """
    dists = (np.array(graph.vs["distance"])[np.logical_not(np.isinf(graph.vs["distance"]))]).tolist()
    locs = (np.array(graph.vs["locations"])[np.logical_not(np.isinf(graph.vs["distance"]))]).tolist()
    scale = 500 ## space scale
    for e in graph.es:
        ## for every road, split into <scale> length segments and find distance at those points
        sd = graph.vs[e.source]["distance"]
        td = graph.vs[e.target]["distance"]
        l = e["weight"]
        
        if np.isinf(sd) or np.isinf(td) or l<scale:
            continue
        
        shape = None
        for i in range(len(shapefile)):
            if shapefile["objectid"][i] == e["id"]:
                shape = shapefile["geometry"][i]
                break
        
        if shape is not None:
            shplen = [0]
            shplen.extend(np.cumsum((np.diff(shape.coords.xy, axis=1)**2).sum(axis=0)**0.5))
            shplen = np.array(shplen)
        
        N = int(l/scale)
        #print(l, scale, N)
        for i in range(N):
            x = (i+1)*scale
            dists.append(min(sd + x, td + (l-x)))
            if shape is not None:
                ## find approximate location of points 
                ind = (shplen>x).tolist().index(False)-1
                rl = x-shplen[ind]
                tl = shplen[ind+1]-shplen[ind]
                locs.append([shape.coords.xy[0][ind]*(rl/tl) + shape.coords.xy[0][ind]*(1-(rl/tl)),
                             shape.coords.xy[1][ind]*(rl/tl) + shape.coords.xy[1][ind]*(1-(rl/tl))])
            else:
                locs.append([None, None])
            
    
    if display:
        plt.hist(dists, bins=20)
        plt.xlabel("distance (m)")
        plt.title("Distribution of distances using points every "+str(scale)+"m on each road")
        plt.show()
    
    """
    dists_s = pd.Series(dists)
    dists_s.name = "distance"
    dists_s.to_csv("distance.csv")
    """
    data = pd.DataFrame([dists,[loc[0] for loc in locs],[loc[1] for loc in locs]], ["dists", "X","Y"]).transpose()
    data.to_csv(pf)

if __name__=="__main__":
    data_folder = "../dati_trentino/"
    
    roads = "p029_l_pup_v.shp"
    chargers = "colonnine_trentino_xy.csv"
    savefile = "trentino_graph"
    save_dist = "trentino_graph_dists"
    points = "points.csv"
    
    # prepare_network(data_folder + roads, data_folder + chargers, data_folder + savefile)
    # print("network prepared")
    # set_distance_info(data_folder + savefile, data_folder + save_dist)
    # print("distances set")
    analyze(data_folder + roads, data_folder + save_dist, data_folder + points)
    
