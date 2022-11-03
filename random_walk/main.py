import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
import multiprocessing

df = pd.read_csv("data/int_net.csv", index_col=0)
edges = list(zip(df.iloc[:,0], df.iloc[:,1], [{'p': df.iloc[i, 2]} for i in range(len(df))]))
G = nx.Graph()
G.add_edges_from(edges)
sg = max(nx.connected_components(G), key=len)
sub_G = G.subgraph(sg)

df_subg = pd.DataFrame(sub_G.edges())
df_subg.to_csv("data/max_connect_net.csv")

tfs = list(map(lambda x: x.strip('\n'), open("data/allTFs.txt", 'r').readlines()))
net = pd.read_csv("data/max_connect_net.csv", index_col=0, header=0)

net_val = net.values
nodes = np.unique(net_val.flatten())

tf_count = 0
tg_count = 0
map_dict = dict()
fp = open("data/map_file.txt", "w", encoding="utf8")
for _ in nodes:
    if _ in tfs:
        tf_count += 1
        tf_name = f"F{tf_count}"
        map_dict[_] = tf_name
    else:
        tg_count += 1
        tg_name = f"G{tg_count}"
        map_dict[_] = tg_name

line = ""
for k, v in map_dict.items():
    line += k + ',' + v + '\n'
fp.write(line)
fp.close()

G = nx.relabel_nodes(sub_G, map_dict)

def node_neib_p(G:nx.Graph):
    p_dict = dict()
    for node in list(G.nodes()):
        curr_node_nbrs = list(G.neighbors(node))
        p = [abs(G[node][i]['p']) for i in curr_node_nbrs]
        p = list(map(lambda x: x / sum(p), p))
        p_dict[node] = p
    return p_dict

p_dict = node_neib_p(G)

import random
def deep_walk(G:nx.Graph, snode:str):
    tfs = list()
    tgs = list()
    walk_seq = snode
    while len(tfs)<10 or len(tgs) < 90:
        rnd = random.random()
        if rnd <= 0.5:
            # walk_seq.append(snode) # 重启
            curr_node = snode
        else:
            curr_node = walk_seq
        curr_node_nbrs = list(G.neighbors(curr_node))
        #p = [abs(G[curr_node][i]['p']) for i in curr_node_nbrs]
        #p = list(map(lambda x: x/sum(p), p))
        next_node = np.random.choice(curr_node_nbrs, p = p_dict[curr_node])
        if next_node.startswith('G'):
            tgs.append(next_node)
        else:
            tfs.append(next_node)
        walk_seq = next_node
    return tfs[0:10], tgs[0:90]

def node2Vector(node:str, G:nx.Graph):
    fp = open(f"ndvct/{node}.txt", "w")
    TFS = []
    TGS = []
    for step in range(10):
        tfs, tgs = deep_walk(G, node)
        TFS.extend(tfs)
        TGS.extend(tgs)
    C_TF = Counter(TFS)
    C_TG = Counter(TGS)
    maxTFs = sorted(C_TF.keys(), key=lambda x:C_TF[x], reverse=True)[0:2]
    maxTGs = sorted(C_TG.keys(), key=lambda x:C_TG[x], reverse=True)[0:8]
    fp.write(",".join(maxTFs) + "," + ",".join(maxTGs))
    fp.close()

if __name__ == '__main__':
    def pp(nodes:list):
        for n in nodes:
            print(n)
            node2Vector(n, G)
    n = 2858
    nodes = list(G.nodes())
    output=[nodes[i:i + n] for i in range(0, len(nodes), n)]
    cpu_work_num = len(output)
    with multiprocessing.Pool(cpu_work_num) as p:
        p.map(pp, output)
