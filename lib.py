import matplotlib as mpl
import networkx as nx
import random
from graphviz import *
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib.image as mpimg
import math

def get_func(alpha,k,d,h):
    def f(x):
        return sigma(x,alpha,k,d,h)
    return f

def sigma(x,alpha,k,d,h):
    if -k*(x - d) > 100:
            return h
    return alpha/(1 + math.exp(-k*(x - d))) + h


mpl.rcParams['figure.dpi'] = 500



def dfs_image(tree,pos,scale,diffx, diffy,out,out_mask,mask=None):
    adj = tree.adj
    mode = False 
    def node_to_point(index):
        point_v = (round(pos[index][0] * scale) + diffx,round(pos[index][1]*scale) + diffy)
        return point_v
    n = out.shape[0]
    m = out.shape[1]
    transparency = {}
    color = np.array([158,86,73])
    all_points = []
    is_v = np.zeros(out.shape)
    out = out.astype(int)
    is_v = is_v.astype(bool)
    visited = [False for i in tree.nodes]
    beta = 1
    def dfs(node,ls,parent,number_of_v):
        visited[node] = True
        parent_point = None
        point = node_to_point(node)
        if parent != -1:
            parent_point = node_to_point(parent)
            transparency[point] = 0.9 * transparency[parent_point]
        else:
            transparency[point] = 1
        point = node_to_point(node)
        ls.append(point)
        not_child =True
        polynom = None
        ls_in = ls[:]
        new_parent = parent
        if len(adj[node]) > 2:
            new_parent = node
            number_of_v = 0
        for i in adj[node]:
            if visited[i] == True:continue
            not_child = False
            dfs(i,ls_in,new_parent,number_of_v + 1)
        if not_child and number_of_v >= 2:
            data = np.array(ls)
            try:
                data= np.array(ls)
                tck,u = interpolate.splprep(data.transpose(), s=0)
                end = 0.0
                count = 0
                index = -1
                for i in range(len(ls)):
                    try:
                        if is_v[ls[i][0]][ls[i][1]] == True:
                            end = u[i] + 0.001
                            index = i
                    except Exception as e:
                        continue
                end = 0.0
                unew = np.arange(end, 1.001, 0.001)
                result = interpolate.splev(unew, tck)
                result = [(round (result[0][i]),round(result[1][i])) for i in range(len(result[1]))]
                beta =0.03
                act_points = []
                for i in result:
                    try:
                        po = i
                        if po[0] < 0 or po[1] < 0:
                            po = list(po)
                            if mode:
                                if po[0] < 0:
                                    po[0] = n + po[0]
                                if po[1] < 0:
                                    po[1] = m + po[1]
                                po = tuple(po)
                            else:
                                continue
                        if mask[po[0]][po[1]] == 0:
                            continue
                        is_v[po[0]][po[1]] = True
                        out_mask[po[0]][po[1]] = True
                        transparency[po] = transparency[point]
                        act_points.append(po)
                        count+=1
                        all_points.append(po)
                        out[po[0]][po[1]] = beta *color + (1 - beta)*out[po[0]][po[1]]
                    except Exception as e:
                        continue
            except Exception as e:
                print(e)
                l = None
        
        
        ls.pop()
    dfs(0,[],-1,0)
    all_points = list(set(all_points))
    print(len(all_points))
    return out,all_points,transparency

arr = [
    get_func(46.95069173,1.14086519,1.96669375, -4.50232485),
    get_func(38.49476413, 0.79133508,2.35083377,-5.18394511),
    get_func(16.66643923,0.93984572  ,1.8621735,-2.46709811),
    get_func(7.24971584, 0.95235099, 0.61697073,-2.58954652),
]
    
    

def generate_capillaries(scale,diffx, diffy,out,out_mask,mask=None):
    options = {
        'node_color': 'black',
        'node_size': 10,
        'width': 1,
    }
    tree = nx.random_tree(700)
    pos = nx.kamada_kawai_layout(tree,dim = 2,scale= 50)
    #nx.draw(tree, pos,**options)
    
    out,ls_points_image,tr_image = dfs_image(tree,pos,scale,diffx, diffy,out,out_mask,mask)
    
    points = set()
    image = out
    ls_points = ls_points_image
    tr = tr_image
    r = 6  #10
    n = out.shape[0]
    m = out.shape[1]
    dp_index = [[[] for i in range(m)] for j in range(n)]
    dp_dist = [[[] for i in range(m)] for j in range(n)]
    for i in range(len(ls_points)):
        ind_point = i
        point = ls_points[ind_point]
        first_point = (point[0],point[1])
        for i in range(-int(r) - 1,int(r) + 2):
            for j in range(0,int(r) + 2):
                dist =(i**2 + j ** 2)**0.5 
                if dist > r:break
                for j1 in [j,-j]:
                    if point[0] + i < 0 or point[1] + j1 < 0 or point[0] + i >= len(image) or point[1] + j1 >= len(image[0]):
                        continue
                    dp_dist[point[0] + i][point[1] + j1].append(dist)
                    dp_index[point[0] + i][point[1] + j1].append(ind_point)
                    points.add((point[0] + i,point[1] + j1))

    points -= set(ls_points)
    for pnt in points:
        i = pnt[0]
        j = pnt[1]
        if len(dp_dist[i][j]) != 0:
            sm_dist = 0 
            RGB_pixel_act = out[i][j]
            CIELAB_pixel_act = rgb_to_cielab(RGB_pixel_act).get_value_tuple()
            CIELAB_new = np.array(list(CIELAB_pixel_act))
            sm_dst = 0.0
            for k in dp_dist[i][j]:
                sm_dst += 1/(k)
            for z in range(len(dp_index[i][j])):
                scel_point = ls_points[dp_index[i][j][z]]
                dst = dp_dist[i][j][z]
                RGB_pixel_scel = out[scel_point[0]][scel_point[1]]
                func = arr[0]
                if tr[scel_point] > 0.5:
                    func = arr[0]
                elif tr[scel_point] > 0.3:
                    func = arr[0]
                else:
                    func = arr[0]
                func = arr[0]
                alpha = 1 - func(dp_dist[i][j][z])/func(12)
                CIELAB_pixal_scel = np.array(rgb_to_cielab(RGB_pixel_scel).get_value_tuple())
                l_val =  alpha*((1/dst)/sm_dst)
                CIELAB_new = CIELAB_new*(1 - l_val) + CIELAB_pixal_scel*l_val
            RGB_new =  list(cielab_to_rgb(CIELAB_new))
            out[i][j] = RGB_new
    return out,out_mask

def generate_(points_with_repeats,mask,image,sz=6):
    """
        points_with_repeats - ((x-coord,y-coord),n-repeats)
    """
    out_mask = np.zeros(image.shape)
    out = image.copy()
    for point in points_with_repeats:
        for _ in range(point[1]):
            out,out_mask =  generate_capillaries(sz,point[0][0],point[0][1],out,out_mask,mask)
    return out.astype(np.uint8),out_mask.astype(np.uint8)
    
def rgb_to_cielab(a):
    """
    a is a pixel with RGB coloring
    """
    a1,a2,a3 = a/255
    color1_rgb = sRGBColor(a1, a2, a3);
    color1_lab = convert_color(color1_rgb, LabColor);
    return color1_lab

def cielab_to_rgb(color):
    """
    color is a color in CIELAB color space
    """
    # convert CIELAB to sRGB
    color_lab = LabColor(color[0], color[1], color[2])
    color_rgb = convert_color(color_lab, sRGBColor)
    
    # convert sRGB to RGB
    r = round(color_rgb.rgb_r * 255)
    g = round(color_rgb.rgb_g * 255)
    b = round(color_rgb.rgb_b * 255)
    return (r, g, b)