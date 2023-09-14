"""import numpy as np

def _compute_affinity(affs, dists, wss):
    new_aff = np.einsum('njk,ij->kn', (affs * (dists ** 1)).T, wss)
    naff_sum = new_aff.sum(axis=1)
    new_aff = np.nan_to_num(new_aff / naff_sum[:, np.newaxis], nan=0.0, posinf=0.0, neginf=0.0)
    return new_aff

# Use this function to check the correct output shape
affs = np.random.rand(49, 565, 3)
dists = np.random.rand(49, 565, 3)
wss = np.random.rand(49, 565)

result = _compute_affinity(affs, dists, wss)
print(result.shape)

sys.exit()"""

import networkit as nxk
import pandas as pd
import networkx as nx
import time
import numpy as np
from multiprocessing import Pool
import multiprocessing
from itertools import chain
import polars as pl
import os
import math
from copy import copy, deepcopy
import datetime
#from numba import njit, jit
#from numba.typed import List
import random
import sys
from spreadAnalysis.persistence.schemas import Spread
from spreadAnalysis.persistence.mongo import MongoSpread
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.io.config_io import Config
import spreadAnalysis.analysis.batching as bsc
import psutil
import time
import Levenshtein
from multiprocessing import Process, Queue
import pickle
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore", category=DeprecationWarning)

def timer(func):
	def wrapper(*args, **kwargs):
			# start the timer
			start_time = time.time()
			# call the decorated function
			result = func(*args, **kwargs)
			# remeasure the time
			end_time = time.time()
			# compute the elapsed time and print it
			execution_time = end_time - start_time
			print(f"Execution time: {execution_time} seconds")
			# return the result of the decorated function execution
			return result
	# return reference to the wrapper function
	return wrapper

def _stlp_get_n_idx_ws_sorted(wneighs,ns):

	sorted_n_idx = defaultdict(list)
	sorted_ws = defaultdict(list)
	sorted_ns = defaultdict(list)
	for wneigh,nn in zip(wneighs,ns):
		n_idx = []
		ws = []
		for n,w in wneigh:
			n_idx.append(n)
			ws.append(w)
		nlen = len(n_idx)
		sorted_n_idx[nlen].append(n_idx)
		sorted_ws[nlen].append(ws)
		sorted_ns[nlen].append(nn)
	sorted_ws = {nlen:np.array(v) for nlen,v in sorted_ws.items()}
	sorted_n_idx = {nlen:np.array(v,dtype='int32') for nlen,v in sorted_n_idx.items()}
	return sorted_n_idx,sorted_ws,sorted_ns

def _stlp_get_n_idx_ws_chunk(n_idxs, ws, aff_map, dist_map):
    ii, jj = np.meshgrid(np.arange(n_idxs.shape[0]), np.arange(n_idxs.shape[1]), indexing='ij')
    affs = aff_map[n_idxs[ii, jj], :]
    dists = dist_map[n_idxs[ii, jj], :]
    wss = ws

    return affs, dists, wss

def _compute_affinity(affs, dists, wss):
    modified_affs = affs * (dists ** 1)
    new_aff = np.zeros((modified_affs.shape[0], modified_affs.shape[2]))
    for i in range(modified_affs.shape[2]):
        new_aff[:, i] = np.sum(wss * modified_affs[:, :, i], axis=1)
    naff_sum = new_aff.sum(axis=1)
    new_aff = np.nan_to_num(new_aff / naff_sum[:, np.newaxis], nan=0.0)
    return new_aff

def get_djik_distances(g,org_nodes,labels):

	all_dists = {v:[] for v in labels.values()}
	for nl,lab in labels.items():
		print (nl)
		djik = nxk.distance.MultiTargetDijkstra(nxk.graphtools.toUndirected(g), nl, org_nodes)
		djik.run()
		dists = djik.getDistances()
		all_dists[lab].append(dists)
	return all_dists

@timer
def stlp(g,labels,net_idx,title="test",num_cores=1,its=5,epochs=3,stochasticity=1,verbose=False):

	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	#max_deg = nxk.graphtools.maxDegree(g)
	org_nodes = list([n for n in g.iterNodes()])
	affinities = []
	distances = []
	all_dists = {}
	org_labels = {net_idx[n]:l for n,l in labels.items() if n in net_idx}
	labels = {}

	print ("generating neighbour idxs")
	nn_idx, nn_ws, nn_ns = _stlp_get_n_idx_ws_sorted([g.iterNeighborsWeights(n) for n in org_nodes],org_nodes)

	for epoch in range(epochs):
		print ("getting distances")
		if len(all_dists) > 0:
			for n,_ in noise_labels.items():
				del labels[n]
			noise_labels = {n:"noise" for n in random.sample([n for n in net_idx.values() if n not in org_labels and g.hasNode(n)],int(1.0*len(org_labels)))}
			labels.update(noise_labels)
			all_dists["noise"]=get_djik_distances(g,org_nodes,noise_labels)["noise"]
		else:
			noise_labels = {n:"noise" for n in random.sample([n for n in net_idx.values() if n not in org_labels and g.hasNode(n)],int(1.0*len(org_labels)))}
			labels.update(org_labels)
			labels.update(noise_labels)
			all_dists = get_djik_distances(g,org_nodes,labels)
		print ("distances set")

		aff_cats = {c:j for j,c in enumerate(sorted(list(set(labels.values()))))}
		dist_map = np.array([np.mean(((np.array(v)+1)**-1)**((np.array(v)+1)**(0.5)), axis=0) for k,v in sorted(all_dists.items())]).T
		aff_map = np.zeros((len(net_idx),len(all_dists)))
		for n,l in labels.items():
			aff_map[n][aff_cats[l]]=1.0
		n_labels = len(aff_cats)

		start_time = time.time()
		for it in range(its):
			new_aff_stack = []
			ncount = 0
			save_all_counts = {}
			nlens = list(nn_ns.keys())
			random.shuffle(nlens)
			for nlen in nlens:
				affs,dists,wss = _stlp_get_n_idx_ws_chunk(nn_idx[nlen],nn_ws[nlen],aff_map,dist_map)
				for i,n in enumerate(nn_ns[nlen]):
					save_all_counts[n]=ncount
					ncount+=1
				new_aff = _compute_affinity(affs,dists,wss)
				new_aff_stack.append(new_aff)

			new_aff_stack = np.concatenate(new_aff_stack,axis=0)
			new_aff_stack = new_aff_stack[[v for k,v in sorted(save_all_counts.items())]]
			#new_aff_stack = np.nan_to_num(new_aff_stack / new_aff_stack.sum(axis=1)[:, np.newaxis], nan=0.0, posinf=0.0, neginf=0.0)
			for n,l in labels.items():
				new_aff_stack[n]=np.zeros(n_labels)
				new_aff_stack[n][aff_cats[l]]=1.0
			aff_map = new_aff_stack
			if verbose:
				print ("OSCAR")
				print (aff_map[net_idx["oscarlagansson7_Twitter"]])
				print (new_aff_stack[net_idx["oscarlagansson7_Twitter"]])
				print ()
				print ("KIMBO")
				print (aff_map[net_idx["Kimbo_Twitter"]])
				print (new_aff_stack[net_idx["Kimbo_Twitter"]])
				print ()
				print ("VANSTER")
				print (aff_map[net_idx["tatillbakavalfarden_Facebook Page"]])
				print (new_aff_stack[net_idx["tatillbakavalfarden_Facebook Page"]])
				print ()

		if verbose:
			print (aff_map[net_idx["Aktuellt Fokus_Facebook Page"]])
			print (aff_map[net_idx["Kimbo_Twitter"]])
			print (aff_map[net_idx["NordfrontSE_Telegram"]])
			print (aff_map[net_idx["oscarlagansson7_Twitter"]])
			end_time = time.time()
			print (end_time - start_time)
		affinities.append(aff_map)
		distances.append(dist_map)
	affinities = np.mean(np.array(affinities),axis=0)
	distances = np.mean(np.array(distances),axis=0)
	#out_affinities = {k:{title+"_"+lab:0.0 for lab in set(labels.values())} for k in net_idx.keys()}
	out_affinities = {title+"_"+lab:{k:0.0 for k in net_idx.keys()} for lab in set(labels.values())}
	out_distances = {k:[] for k in net_idx.keys()}
	for lab in set(labels.values()):
		for i,n in enumerate(affinities):
			#out_affinities[rev_net_ids[i]][title+"_"+lab]=n[aff_cats[lab]]
			out_affinities[title+"_"+lab][rev_net_ids[i]]=n[aff_cats[lab]]
	for lab in set(labels.values()):
		if "noise" not in lab:
			for i,d in enumerate(distances):
				out_distances[rev_net_ids[i]].append(np.log(d+2))
	return {title:out_affinities},{title+"_"+"dist":{k:np.mean(np.array(v)) for k,v in out_distances.items()}}


def set_affinities(g,affinities):

	for n,affs in affinities.items():
		for aff,val in affs.items():
			if n in g:
				nx.set_node_attributes(g,{n:val},aff)
	return g

@timer
def get_labels_from_affinities(affinities, approach="prob"):
	labels = {}		
	#labels = {n:sorted(affs.items(), key=lambda x:x[1], reverse=True)[0][0] for n,affs in affinities.items()}
	for title,labs in list(affinities.items()):
		labels[title]={}
		#lab_series = {lab:pd.Series(list(vlab.values())) for lab,vlab in labs.items()}
		lab_series = {}
		for lab,vlab in labs.items():
			lab_series[lab]={}
			array_len = len(vlab)
			for i,val in enumerate(sorted(list(vlab.values()))):
				if val not in lab_series[lab]:
					lab_series[lab][val]=i/array_len

		#lab_stds = {lab:np.std(data) for lab,data in lab_series.items()}
		for n in list(labs[list(labs.keys())[0]].keys()):
			for label in list(labs.keys()):
				if approach == "simple":
					affs = {label:affinities[title][label][n] for label in list(labs.keys())}
				if approach == "prob":
					affs = {label:lab_series[label][affinities[title][label][n]] for label in list(labs.keys())}
				labels[title][n]=sorted(affs.items(), key=lambda x:x[1], reverse=True)[0][0]
	return labels

def set_labels(g,labels,title="test"):

	nx.set_node_attributes(g,labels,title)
	return g

def remap_nodes_based_on_category(df,node_mapping,keep_self_loops=False):

	df = df.with_columns(pl.col("o").map_dict(node_mapping).alias("o"))
	df = df.with_columns(pl.col("e").map_dict(node_mapping).alias("e"))
	if not keep_self_loops:
		df = df.filter(pl.col("e")!=pl.col("o"))
	return df

def chunks(l, n):
	for i in range(0, n):
			yield l[i::n]

def filter_df_on_degrees(df,col,mind=2,only=None):

	if only is not None:
		df.filter(~pl.col(col).is_in(df.groupby([col]).agg(pl.count()).filter((pl.col("count")<mind) & (pl.col(col).is_in(only)))[col].to_list()))
	else:
		df = df.filter(~pl.col(col).is_in(df.groupby([col]).agg(pl.count()).filter(pl.col("count")<mind)[col].to_list()))
	return df

def filter_df_on_edge_weight(df,col,minw=2):

	df = df.filter((pl.col(col)>=minw))
	return df

def filter_on_degrees(g,mind=2):

	to_remove = []
	for n in list(g.iterNodes()):
			if g.degree(n) < mind:
				to_remove.append(n)
	for n in to_remove:
		g.removeNode(n)
	return g

def filter_on_selected(g,keep_nodes=[]):

	to_remove = []
	new_g = nxk.Graph(g,weighted=g.isWeighted(), directed=g.isDirected(), edgesIndexed=False)
	for n in list(new_g.iterNodes()):
		if n not in keep_nodes:
			to_remove.append(n)
	for n in to_remove:
		if new_g.hasNode(n):
			new_g.removeNode(n)
	return new_g

def filter_on_gc(g,is_nxg=False):
	
	if is_nxg:
		gc = set(sorted(nx.connected_components(g.to_undirected()), key = len, reverse=True)[0])
		for node in list(g.nodes()):
			if not node in gc:
				g.remove_node(node)
	else:
		cug = set(list(nxk.components.ConnectedComponents.extractLargestConnectedComponent(nxk.graphtools.toUndirected(g)).iterNodes()))
		g = filter_on_selected(g,keep_nodes=cug)

	return g

def filter_on_edge_weight(nxg,minw=2):

	for o,e,d in list(nxg.edges(data=True)):
		if d["weight"]<minw:
			nxg.remove_edge(o,e)
	return nxg

def filter_on_metric(g,df,net_idx,metrics,metric="partisan_dist",keep_prop=0.2):

	current_total = g.numberOfNodes()
	keep_nodes = set([k for k,v in sorted(metrics[metric].items(), key=lambda x:x[1], reverse=True)[:int(current_total*keep_prop)]])
	keep_nodes.update(set([k for k,v in metrics["custom"].items() if v < 2]))
	keep_nodes = set([net_idx[n] for n in keep_nodes if n in net_idx])
	g = filter_on_selected(g,keep_nodes=keep_nodes)
	df = df.filter((pl.col("o").is_in(keep_nodes)) & (pl.col("e").is_in(keep_nodes)))

	return g,df,net_idx


def filter_based_on_com(nxg,max_nodes=1000,com_var="com",preferred_metric="pagerank"):

	print (nxg.number_of_nodes())
	print (nxg.number_of_edges())
	print ("filtering based on com")
	nnodes = len(nxg.nodes())
	if nnodes > max_nodes:
		keep_percent = max_nodes/nnodes
		df = []
		for node,dat in nxg.nodes(data=True):
			dat["actor_platform"]=node
			df.append(dat)
		df = pd.DataFrame(df)
		df = df.sort_values([com_var,preferred_metric],ascending=False)
		com_counts = df[[com_var]].groupby(com_var).size().reset_index()
		df = df[df[com_var].isin(set(list(com_counts[com_counts[0]>int(0.003*max_nodes)][com_var])))]
		filtered_dfs = []
		for group,vals in df.groupby(com_var):
			filtered_dfs.append(vals.head(int(keep_percent*len(vals))))
		filtered_dfs = pd.concat(filtered_dfs)
		keep_nodes=set(list(filtered_dfs["actor_platform"]))
		keep_nodes.update(set([n for n,d in nxg.nodes(data=True) if "custom" in d and d["custom"]==0]))
		for n in list(nxg.nodes()):
			if n not in keep_nodes:
				nxg.remove_node(n)
		print (nxg.number_of_nodes())
		print (nxg.number_of_edges())
		gc = set(sorted(nx.connected_components(nxg.to_undirected()), key = len, reverse=True)[0])
		for node in list(nxg.nodes()):
			if not node in gc:
				nxg.remove_node(node)
		print (nxg.number_of_nodes())
		print (nxg.number_of_edges())
		if nxg.number_of_edges() > 2000000:
			nxg = filter_on_edge_weight(nxg,minw=3)
			nxg = filter_on_gc(nxg,is_nxg=True)
		print (nxg.number_of_nodes())
		print (nxg.number_of_edges())

		return nxg
	else:
		return nxg

def net_to_compact(g,net_idx):

	tmp_node_map = nxk.graphtools.getContinuousNodeIds(g)
	g = nxk.graphtools.getCompactedGraph(g,tmp_node_map)
	new_net_idx = {k:tmp_node_map[v] for k,v in net_idx.items() if v in tmp_node_map}
	return g,new_net_idx

def create_rev_net_idx(g=None,net_idx=None):
	
	if g is not None:
		rev_net_idx = {v:k for k,v in net_idx.items() if g.hasNode(v)}
	else:
		rev_net_idx = {v:k for k,v in net_idx.items()}
	return rev_net_idx

@timer
def to_uni_matrix(g,ntc,di=False,ncores=-1):

	bi_neighbors = []
	for n in ntc:
		if di:
			ns = np.array(list(g.iterNeighbors(n)),dtype=np.int32)
		else:
			ns = np.array(sorted(list(g.iterNeighbors(n))),dtype=np.int32)
		bi_neighbors.append(ns)
		g.removeNode(n)
	random.shuffle(bi_neighbors)
	edge_m = uni_neighbors(bi_neighbors,ncores=ncores)

	return edge_m

@timer
def edge_df_to_graph(g,edge_m,ncores=-1):
	
	new_g = nxk.graphtools.copyNodes(g)
	for row in edge_m.to_numpy():
		#g.increaseWeight(int(row[0]), int(row[1]), int(row[2]))
		new_g.addEdge(int(row[0]), int(row[1]), w=int(row[2]))
	return new_g

@timer
def create_edge_tuples(df):

	if isinstance(df,pd.DataFrame):
			df = df.groupby(["url","actor_platform"]).size().reset_index()
			#df.sort_values("url",inplace=True)
			e_tups = zip(list(df.iloc[:,0]),list(df.iloc[:,1]),list(df.iloc[:,2]))
	else:
			#e_tups = df.groupby(["url", "actor_platform"]).agg(pl.count())
			#e_tups = e_tups.sort("actor_platform")
			#e_tups = zip(e_tups["url"].to_list(),e_tups["actor_platform"].to_list(),e_tups["count"].to_list())
			e_tups = zip(df["o"].to_list(),df["e"].to_list(),df["w"].to_list())

	return e_tups

@timer
def get_collapse_node_list(g,df,net_idx,col="o"):

	do_not_collapse = set(df["e"].to_list())
	to_collapse = set(df[col].to_list())
	nodes_to_collapse = [net_idx[str(n)] for n in to_collapse if g.hasNode(net_idx[str(n)]) and n not in do_not_collapse]	
	return nodes_to_collapse

@timer
def df_to_nxk(s_t,di=False):

	g = nxk.graph.Graph(n=0, weighted=True, directed=di, edgesIndexed=False)
	net_idx = {}
	for o,e,w in s_t:
		o = str(o)
		e = str(e)
		if o not in net_idx:
			o_i = g.addNode()
			net_idx[o]=o_i
		else:
			o_i = net_idx[o]
		if e not in net_idx:
			e_i = g.addNode()
			net_idx[e]=e_i
		else:
			e_i = net_idx[e]
		#g.addEdge(o_i,e_i,w=int(w))
		g.increaseWeight(o_i, e_i, w)

	return g, net_idx

def uni_neighbors(X,ncores=-1):

	if ncores < 1:
		ncores = multiprocessing.cpu_count()-2
	#Y = chain.from_iterable(Pool(ncores).map(_bi_permutation,chunks(X,ncores)))
	#Y = pl.concat(Pool(ncores).map(_bi_permutation_NEW,chunks(X,ncores)))
	Y = np.concatenate(Pool(ncores).map(_bi_permutation,chunks(X,ncores)),axis=0)
	
	return Y

def _bi_permutation(X):

	#Y = np.empty(shape = [0,2],dtype=np.int32)
	Y = []
	for i in range(len(X)):
		x = X.pop()
		y = x[np.stack(np.triu_indices(len(x), k=1), axis=-1)]
		Y.append(y)
	Y = np.vstack(Y)
	return Y

@timer
def edge_m_to_df(edge_m):
	#df = df.groupby(["o", "e"]).agg(weight=pl.count())
	return pl.DataFrame(edge_m,schema=["o","e"],schema_overrides={"o":pl.Int32,"e":pl.Int32}).groupby(["o", "e"]).agg(weight=pl.count())
	#return (pl.DataFrame(edge_m,schema=["o","e"],schema_overrides={"o":pl.Int32,"e":pl.Int32}).lazy().groupby(["o", "e"]).agg(weight=pl.count())).collect(streaming=True)

def extract_subgraph(graph, partition, community):
    subgraph = nxk.Graph(directed=graph.isDirected())
    nodes_in_community = {node for node, comm in enumerate(partition.getVector()) if comm == community}
    
    # Create a mapping from original node IDs to new subgraph node IDs
    node_mapping = {original_node: subgraph.addNode() for original_node in nodes_in_community}
    
    for u in nodes_in_community:
        for v in graph.iterNeighbors(u):
            if v in nodes_in_community:
                subgraph.addEdge(node_mapping[u], node_mapping[v])
    
    return subgraph

@timer
def noise_corrected(df, undirected = True,weight="weight"):
	 
	src_sum = df.groupby("o").agg(o_sum=pl.sum(weight))
	trg_sum = df.groupby("e").agg(e_sum=pl.sum(weight))
	df = df.join(src_sum,how="left",on="o")
	df = df.join(trg_sum,how="left",on="e")
	df = df.with_columns(df.select(pl.sum(weight))[weight].alias("n.."))
	df = df.with_columns((((pl.col("o_sum") * pl.col("e_sum")) / pl.col("n..")) * (1 / pl.col("n..")) ).alias("mean_prior_probability"))
	df = df.with_columns((pl.col("n..")/(pl.col("o_sum")*pl.col("e_sum"))).alias("kappa"))
	df = df.with_columns((((pl.col("kappa")*pl.col(weight))-1)/((pl.col("kappa")*pl.col(weight))+1)).alias("score"))
	df = df.with_columns(((1/(pl.col("n..")**2))*(pl.col("o_sum")*pl.col("e_sum")*(pl.col("n..")-pl.col("o_sum"))*(pl.col("n..")-pl.col("e_sum")))/((pl.col("n..")**2)*(pl.col("n..")-1))).alias("var_prior_probability"))
	df = df.with_columns((((pl.col("mean_prior_probability")**2)/pl.col("var_prior_probability"))*(1-pl.col("mean_prior_probability"))-pl.col("mean_prior_probability")).alias("alpha_prior"))
	df = df.with_columns(((pl.col("mean_prior_probability")/pl.col("var_prior_probability"))*(1-(pl.col("mean_prior_probability")**2))-(1-pl.col("mean_prior_probability"))).alias("beta_prior"))
	df.drop_in_place("mean_prior_probability")
	df = df.with_columns((pl.col("alpha_prior")+pl.col(weight)).alias("alpha_post"))
	df.drop_in_place("alpha_prior")
	df = df.with_columns((pl.col("n..")-pl.col(weight)+pl.col("beta_prior")).alias("beta_post"))
	df.drop_in_place("beta_prior")
	df = df.with_columns((pl.col("alpha_post")/(pl.col("alpha_post")+pl.col("beta_post"))).alias("expected_pij"))
	df.drop_in_place("alpha_post")
	df.drop_in_place("beta_post")
	df = df.with_columns((pl.col("expected_pij")*(1-pl.col("expected_pij"))*pl.col("n..")).alias("variance_nij"))
	df.drop_in_place("expected_pij")
	df = df.with_columns(((1.0/(pl.col("o_sum")*pl.col("e_sum")))-(pl.col("n..")*((pl.col("o_sum")+pl.col("e_sum")) / ((pl.col("o_sum")*pl.col("e_sum"))**2)))).alias("d"))
	df = df.with_columns((pl.col("variance_nij")*(((2*(pl.col("kappa")+(pl.col(weight)*pl.col("d")))) / (((pl.col("kappa")*pl.col(weight))+1)**2))**2)).alias("variance_cij"))
	df = df.with_columns((pl.col("variance_cij")**.5).alias("sdev_cij"))
	if undirected:
		df = df.filter(pl.col("o") <= pl.col("e"))
	return df.select(pl.col(["o", "e", weight, "score", "sdev_cij"]))

@timer
def filter_on_backbone(df,threshold=1.0,max_edges=-1,tol=0.2,skip_nodes=[],weight="weight",remove_only={}):
	
	if max_edges > 1:
		new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | (pl.col("o").is_in(skip_nodes)) | (pl.col("e").is_in(skip_nodes)))
		while len(new_df) > max_edges:
			threshold = threshold*(tol*np.log(len(new_df)))
			new_df = new_df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | (pl.col("o").is_in(skip_nodes)) | (pl.col("e").is_in(skip_nodes)))
			print (len(new_df))
	elif len(remove_only) > 0:
		sp_df = df.filter((pl.col("o").is_in(remove_only)) | (pl.col("e").is_in(remove_only)))
		bf_filter = len(sp_df) 
		new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | (pl.col("o").is_in(skip_nodes)) | (pl.col("e").is_in(skip_nodes)))
		org_new_df = len(new_df)
		if max_edges > 0:
			while len(new_df) > org_new_df-int(max_edges*bf_filter):
				threshold = threshold*(tol*np.log(len(new_df)))
				new_df = new_df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | (pl.col("o").is_in(skip_nodes)) | (pl.col("e").is_in(skip_nodes)))
				print (len(new_df))
	else:
		if skip_nodes:
			new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0) | (pl.col("o").is_in(skip_nodes)) | (pl.col("e").is_in(skip_nodes)))
		else:
			new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0))
	return new_df.select(pl.col(["o", "e", weight]))

def _get_node_coms(g,coms,net_idx=None,reverse=True):

	node_coms = {}
	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	for subset in coms.getSubsetIds():
		if len(coms.getMembers(subset)) > 9:
			for member in coms.getMembers(subset):
				if reverse:
					if subset not in node_coms: node_coms[subset]=set([])
					node_coms[subset].add(member)
				else:
					if net_idx is not None:
						node_coms[rev_net_ids[member]]=subset
					else:
						node_coms[member]=subset
	return node_coms

def incremental_conductance(g, community,community_idx, node, node_impact_dict, community_weights):
    
    # Calculate the change in the weight of internal edges and cut edges
	internal_edge_weight_change = sum([w for neighbor,w in g.iterNeighborsWeights(node) if neighbor in community])
	cut_edge_weight_change = sum([w for neighbor,w in g.iterNeighborsWeights(node)]) - internal_edge_weight_change
    
    # Calculate the change in conductance
	delta_conductance = (cut_edge_weight_change - internal_edge_weight_change) / community_weights[community_idx]

	# Record the impact of the node on the change in conductance
	node_impact_dict[community_idx] =  delta_conductance
	return node_impact_dict

def _find_best_com_fit(g1,gam,its=10):

	all_coms = {}
	coms_scores = defaultdict(list)
	for r in range(its):
		coms = nxk.community.detectCommunities(g1, algo=nxk.community.PLM(g1, refine=True, gamma=gam+random.uniform(-1, 1)*0.25, par='balanced', maxIter=128, turbo=True, recurse=True),inspect=False)
		all_coms[r]=coms
	if its > 1:
		for r1,coms1 in list(all_coms.items()):
			for r2,coms2 in list(all_coms.items()):
				if r1 != r2:
					jac_score = nxk.community.JaccardMeasure().getDissimilarity(g1, coms1, coms2)
					coms_scores[r1].append(jac_score)
		coms_scores = {k:np.mean(np.array(v)) for k,v in coms_scores.items()}
		coms = all_coms[sorted(coms_scores.items(), key=lambda x:x[1], reverse=False)[0][0]]
	print (f"Best partition: (average similarity: {round((1-sorted(coms_scores.items(), key=lambda x:x[1], reverse=False)[0][1])*100,3)}%)")
	nxk.community.inspectCommunities(coms, g1)
	return coms

def _find_best_modularity_gamma(g1,gammas=None):

	if gammas is None: gammas = np.linspace(0.75, 4.0, num=20)
	com_mods = {}
	prev_score = 0
	for gam in gammas:
		mod = nxk.community.Modularity().getQuality(nxk.community.detectCommunities(g1, algo=nxk.community.PLM(g1, refine=True, gamma=gam, par='balanced', maxIter=128, turbo=True, recurse=True),inspect=False),g1)
		score = (np.log(gam+1))*mod
		com_mods[gam]=score
		if score < prev_score:
			break
		prev_score = score
		print (str(gam)+" : "+str(mod)+" - "+str(score))
	return sorted(com_mods.items(), key=lambda x:x[1], reverse=True)[0][0]

@timer
def _get_coms(g,org_deg_df,net_idx=None,base_deg=1):

	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	final_node_coms = {}
	final_node_coms_conds = {}
	if g.isDirected(): g = nxk.graphtools.toUndirected(g)
	base_nodes = set([net_idx[n] for n,c in org_deg_df.items() if n in net_idx and c == base_deg])
	base_g = filter_on_selected(g,base_nodes)
	gam = _find_best_modularity_gamma(base_g)
	coms = _find_best_com_fit(base_g,gam,its=12)
	#coms = nxk.community.detectCommunities(base_g, algo=nxk.community.PLM(base_g, refine=True, gamma=_find_best_modularity_gamma(base_g), par='balanced', maxIter=128, turbo=True, recurse=True),inspect=True)
	com_nodes = _get_node_coms(base_g,coms)
	node_coms = {n:set([]) for n in g.iterNodes()}
	for com, nodes in com_nodes.items():
		new_assigned = set(nxk.scd.LocalTightnessExpansion(g, alpha=1.0).expandOneCommunity(list(nodes)))
		com_nodes[com].update(set([n for n in new_assigned if (n in nodes) or (n not in nodes and n not in base_nodes)]))
		for n in com_nodes[com]:
			node_coms[n].add(com)

	community_weights = {community:sum([sum([w for neighbor,w in g.iterNeighborsWeights(n)]) for n in com_nodes[community]]) for community in com_nodes.keys()}
	new_assigned_coms = {k:0 for k in com_nodes.keys()}
	new_assigned_coms[-1]=0
	for n,coms in node_coms.items():
		if len(coms) == 0:
			real_com = -1
		elif len(coms) == 1:
			real_com = list(coms)[0]
		else:
			com_changes = {}
			for com in coms:
				com_changes = incremental_conductance(g,com_nodes[com],com,n,com_changes,community_weights)
			real_com = sorted(com_changes.items(), key=lambda x:x[1], reverse=False)[0][0]
		final_node_coms[n]=real_com
		if n not in base_nodes: new_assigned_coms[real_com]+=1
	
	for com in com_nodes.keys():
		org_com_cond = nxk.scd.SetConductance(g,com_nodes[com])
		org_com_cond.run()
		org_com_cond = org_com_cond.getConductance()
		for n in com_nodes[com]:
			final_node_coms_conds[n]=org_com_cond
	
	final_node_coms = {rev_net_ids[k]:v for k,v in final_node_coms.items()}
	final_node_coms_conds = {rev_net_ids[k]:v for k,v in final_node_coms_conds.items()}
	print (len({k for k,v in final_node_coms.items() if v != -1}))
	print (len({k for k,v in final_node_coms.items() if v == -1}))
	print (new_assigned_coms)

	return final_node_coms,final_node_coms_conds

def get_communities(g,net_idx=None,its=10,res=1.0,recursive=True,parent_com=None):
	
	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	node_coms = {}
	node_com_conds = {}
	inner_node_coms = {}
	inner_node_com_conds = {}
	if g.isDirected(): g = nxk.graphtools.toUndirected(g)
	all_coms = {}
	coms_scores = defaultdict(list)
	#gams = [int(res*0.5) for r in range(int(its*0.5))]
	gams = []
	#gams.extend([float(i) for i in np.linspace(0.4, res, int(its*1))])
	for r in range(its):
		gam = res
		#gam = gams[r]
		#if r < (0.4*its)-1 and its>1:
			#coms = nxk.community.detectCommunities(g, algo=nxk.community.ParallelLeiden(g, randomize=True, iterations=3, gamma=res),inspect=False)
		#else:
		coms = nxk.community.detectCommunities(g, algo=nxk.community.PLM(g, refine=True, gamma=gam, par='balanced', maxIter=128, turbo=True, recurse=True),inspect=False)
		#coms = nxk.community.detectCommunities(g, algo=nxk.community.SpectralPartitioner(g, int(g.numberOfNodes()/1300), balanced=True),inspect=False)
		#coms = nxk.community.detectCommunities(g, algo=nxk.community.ParallelLeiden(g, randomize=True, iterations=3, gamma=res),inspect=False)
		#coms = nxk.community.detectCommunities(g, algo=nxk.community.LFM(g, scd = nxk.scd.LocalTightnessExpansion(g)),inspect=False)
			#coms = nxk.community.detectCommunities(g, algo=nxk.community.LouvainMapEquation(g, hierarchical=True, maxIterations=32, parallelizationStrategy='relaxmap'),inspect=False)
		all_coms[r]=coms
	if its > 1:
		for r1,coms1 in list(all_coms.items()):
			for r2,coms2 in list(all_coms.items()):
				if r1 != r2:
					#nmid_score = nxk.community.NMIDistance().getDissimilarity(g, coms1, coms2)
					jac_score = nxk.community.JaccardMeasure().getDissimilarity(g, coms1, coms2)
					#rand_score = nxk.community.NodeStructuralRandMeasure().getDissimilarity(g, coms1, coms2)
					#coms_scores[r1].append(nmid_score)
					coms_scores[r1].append(jac_score)
					#coms_scores[r1].append(rand_score)
		coms_scores = {k:np.mean(np.array(v)) for k,v in coms_scores.items()}
		coms = all_coms[sorted(coms_scores.items(), key=lambda x:x[1], reverse=False)[0][0]]
		#print (sorted(coms_scores.items(), key=lambda x:x[1], reverse=False))
	else:
		coms = all_coms[0]
		nxk.community.inspectCommunities(coms, g)
	if its > 1:
		print (f"Best partition: (average similarity: {round((1-sorted(coms_scores.items(), key=lambda x:x[1], reverse=False)[0][1])*100,3)}%)")
		nxk.community.inspectCommunities(coms, g)
	conds = get_conductance_scores(g,coms)
	for subset in coms.getSubsetIds():
		for member in coms.getMembers(subset):
			com = subset
			if parent_com is not None: com = str(parent_com)+"_"+str(com)
			if net_idx is not None:
				node_coms[rev_net_ids[member]]=com
				node_com_conds[rev_net_ids[member]]=conds[subset]
			else:
				node_coms[member]=com
				node_com_conds[member]=conds[subset]

	if recursive:
		for subset in coms.getSubsetIds():
			if len(coms.getMembers(subset)) > int(g.numberOfNodes()*0.005):
				subgraph = nxk.graphtools.subgraphFromNodes(g, coms.getMembers(subset))
				temp_node_coms, temp_node_com_conds,_,_ = get_communities(subgraph,net_idx=net_idx,its=1,res=1.5,recursive=False,parent_com=subset)
				inner_node_coms.update(temp_node_coms)
				inner_node_com_conds.update(temp_node_com_conds)
		for member,com in node_coms.items():
			if member not in inner_node_coms:
				inner_node_coms[member]=str(com)+"_"+str(-1)
				inner_node_com_conds[member]=node_com_conds[member]
	else:
		inner_node_coms = copy(node_coms)
		inner_node_com_conds = copy(node_com_conds)

	return node_coms, node_com_conds, inner_node_coms, inner_node_com_conds

def get_actor_fields(nodes,add_actor_fields=None,net_idx=None,use_num_id=False):

	if add_actor_fields is None: add_actor_fields = ["platform","actor_name","lang","link_to_actor","n_posts","shares_mean","comments_mean","reactions_mean","followers_mean"]
	if net_idx is not None: rev_net_ids = create_rev_net_idx(g=None,net_idx=net_idx)
	mdb = MongoSpread()
	node_data = {f:{} for f in add_actor_fields}
	projection = {f:1 for f in add_actor_fields}
	projection["actor_platform"]=1
	for a_chunk in chunks(list(set([n for n in nodes])),100):
		if net_idx is not None: a_chunk = [rev_net_ids[n] for n in a_chunk]
		for doc in mdb.database["actor_metric"].find({"actor_platform":{"$in":list(a_chunk)}},projection):
			for f in add_actor_fields:
				if net_idx is not None and use_num_id: node_data[f][net_idx[doc["actor_platform"]]]=doc[f]
				else: node_data[f][doc["actor_platform"]]=doc[f]
	mdb.close()
	return node_data

@timer
def get_metrics(g,net_idx=None):

	metrics = {}
	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	metrics["deg_c"]=nxk.centrality.DegreeCentrality(g, normalized=True)
	metrics["pagerank"]=nxk.centrality.PageRank(g,tol=1e-3, normalized=True)
	#metrics["eig_c"]=nxk.centrality.EigenvectorCentrality(g,tol=1e-3)
	metrics["btw_c"]=nxk.centrality.EstimateBetweenness(g, 10, normalized=True, parallel=True)
	out_metrics = {k:{} for k in metrics.keys()}
	out_metrics.update({"in_degree":{},"out_degree":{}})
	for m,f in metrics.items():
		print (m)
		f.run()
		print (m)
		for n in g.iterNodes():
			score = f.score(n)
			if net_idx is not None:
				n = rev_net_ids[n]
			out_metrics[m][n]=score
	for n in g.iterNodes():
		in_deg = g.degreeIn(n)
		out_deg = g.degreeOut(n)
		if net_idx is not None:
			n = rev_net_ids[n]
		out_metrics["in_degree"][n]=in_deg
		out_metrics["out_degree"][n]=out_deg
	return out_metrics

def get_enriched_metrics(g,metrics,net_idx=None,enga_trans_path=True):

	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	mdb = MongoSpread()
	if enga_trans_path is not None:
		enga_trans_path = Config().PROJECT_PATH+"/full_test/enga_trans"
		if os.path.isfile(enga_trans_path+"_scores.p"):
			enga_trans_scores = pickle.load(open(enga_trans_path+"_scores.p","rb"))
		else:
			pickle.dump({},open(enga_trans_path+"_scores.p","wb"))
			enga_trans_scores = pickle.load(open(enga_trans_path+"_scores.p","rb"))
		if os.path.isfile(enga_trans_path+".p"):
			enga_trans = pickle.load(open(enga_trans_path+".p","rb"))
		else:
			pickle.dump(bsc.create_enga_transformer_actor(mdb,n_sample=30000),open(enga_trans_path+".p","wb"))
			enga_trans = pickle.load(open(enga_trans_path+".p","rb"))
	projection = {"reactions_mean":1,"shares_mean":1,"comments_mean":1,"platform":1,"followers_mean":1,"actor_platform":1}
	adocs = {}
	nodes_to_search = set([rev_net_ids[n] for n in g.iterNodes() if rev_net_ids[n] not in enga_trans_scores])
	for a_chunk in chunks(list(nodes_to_search),100):
		for doc in mdb.database["actor_metric"].find({"actor_platform":{"$in":list(a_chunk)}},projection):
			adocs[doc["actor_platform"]]=doc
	a_engas = {}
	a_engas.update(enga_trans_scores)
	added_one = False
	for n in list(adocs.keys()):
		if n not in a_engas:
			data = []
			actor_doc = adocs[n]
			data_doc = {}
			for f in ["reactions_mean","shares_mean","comments_mean","platform"]:
				data_doc[f]=actor_doc[f]
				if actor_doc["platform"] == "reddit" and f == "reactions_mean":
					data_doc[f]=0.0
				if actor_doc["platform"] == "twitter" and f == "reactions_mean":
					data_doc[f]=actor_doc[f]+actor_doc["followers_mean"]
			data.append(data_doc)
			df = pd.DataFrame(data)
			df["engagement"]=df["comments_mean"]+df["reactions_mean"]+df["shares_mean"]
			scalers = enga_trans[actor_doc["platform"]]
			new_enga = scalers[0].transform(df[["engagement"]])
			df["engagement_yj"]=new_enga
			n_enga = scalers[1].transform(df[["engagement_yj"]])
			n_enga = float(n_enga[0])
			a_engas[n]=n_enga
			added_one = True
	if added_one: pickle.dump(a_engas,open(enga_trans_path+"_scores.p","wb"))
	enriched_metrics = {"norm_engagement":{}}
	for met,vals in metrics.items():
		if met in set(["deg_c","btw_c","pagerank"]):
			enriched_metrics[met+"_wEnga"]={}
			enriched_metrics[met+"_wEnga_sq"]={}
			enriched_metrics[met+"_wEnga^2"]={}
			for n,val in vals.items():
				if n in a_engas:
					n_enga = a_engas[n]
					enriched_metrics[met+"_wEnga"][n]=val*(n_enga*10000)
					enriched_metrics[met+"_wEnga_sq"][n]=float(np.sqrt((n_enga*10000)))
					enriched_metrics[met+"_wEnga^2"][n]=val*((n_enga*1000)**2)
					if n not in enriched_metrics["norm_engagement"]:
						enriched_metrics["norm_engagement"][n]=n_enga*100
	
	mdb.close()
	return enriched_metrics

def get_norm_edge_weights_out(nxg):

	no_e_w = {}
	for o,e,w in nxg.edges(data=True):
		no_e_w[(o,e)]=w["weight"]/nxg.out_degree(o,"weight")
	return no_e_w

def get_norm_edge_weights_full(nxg):

	no_e_w = {}
	max_indeg = max([nxg.in_degree(n,"weight") for n in nxg.nodes()])
	max_outdeg = max([nxg.out_degree(n,"weight") for n in nxg.nodes()])
	for o,e,w in nxg.edges(data=True):
		no_e_w[(o,e)]=(w["weight"]/(nxg.out_degree(o,"weight")))*(w["weight"]/(nxg.in_degree(e,"weight")))
	return no_e_w

def conductance(graph, community_nodes):
    # Sum the weights of edges between the community and its complement
	cut_weight = sum(graph.weight(u, v) for u in community_nodes for v in graph.iterNeighbors(u) if v not in community_nodes)
    # Sum the weights of edges with one endpoint in the community
	community_edge_weight = sum(graph.weight(u, v) for u in community_nodes for v in graph.iterNeighbors(u))

    # Compute conductance
	return cut_weight / community_edge_weight

def get_conductance_scores(g,coms,verbose=False):

	scores = {}
	for com in set(coms.getSubsetIds()):
		nodes_in_community = {node for node in coms.getMembers(com)}
		comm_conductance = conductance(g, nodes_in_community)
		if verbose:
			print(f"Conductance of community {com}:", comm_conductance)
		scores[com]=comm_conductance
	return scores

def set_metrics(g,metrics,enga_trans=True,enrich=True):

	for met,vals in metrics.items():
		nx.set_node_attributes(g,vals,met)

	return g

def export_net_data(g,metrics):

	pass

def _group_text_similar_urls(url_ts):

	ut_count = 0
	ut_groups = []
	for url,ts in url_ts:
		same = {}
		same_idx = {}
		for t1 in ts:
			if t1 is not None: t1 = t1.replace(url,"").split("http")[0]
			if len(same) < 1 or t1 is None or len(t1) < 5:
				ut_count+=1
				ut_groups.append(ut_count)
				if t1 is not None:
					same[t1]=set([])
					same_idx[t1]=ut_count
			else:
				found = False
				for t2 in list(same.keys()):
					sim = Levenshtein.ratio(t1, t2)
					if (sim > 0.69) or (len(same[t2])>0 and Levenshtein.ratio(t1,random.choice(list(same[t2]))) > 0.69):
						same[t2].add(t1)
						found = True
						ut_groups.append(same_idx[t2])
						break
				if not found:
					ut_count+=1
					ut_groups.append(ut_count)
					same[t1]=set([])
					same_idx[t1]=ut_count
	return ut_groups

@timer
def filter_urls_based_on_texts(df,num_cores=1):
	
	ut_groups = []
	df = df.sort(['o','text'],descending=False) 
	url_g_df = df.groupby("o").agg(pl.col("text")).sort('o',descending=False)
	url_ts = list(zip(url_g_df["o"].to_list(),url_g_df["text"].to_list()))
	if num_cores > 1:
		results = Pool(num_cores).map(_group_text_similar_urls,chunks(url_ts,num_cores))
	else:
		results = [_group_text_similar_urls(url_ts)]
	for result in results:
		ut_groups.extend(result)
	df = df.with_columns(pl.Series(name="text_group", values=ut_groups))
	df = df.with_columns((pl.col("o")+"_"+pl.col("text_group").cast(pl.Utf8)).alias("text_based_o"))
	df.replace("o",pl.Series(df["text_based_o"].to_list()))
	df.drop_in_place("text")
	df.drop_in_place("text_based_o")
	df.drop_in_place("text_group")
	return df

def _mid_to_dt(args):

	UNI_DATE_FIELDS = {"date":1,"created_at":1,"created_utc":1,"snippet.publishedAt":1,"timestamp":1}
	mdb = MongoSpread()
	new_dfs = []
	dfs = args[0]
	incl_text = args[1]
	dts = {}
	texts = {}
	for df_i in dfs:
		all_mids = df_i["dt"].to_list()
		print (len(all_mids))
		if len(all_mids) < 150000:
			all_mids = [all_mids]
		else:
			all_mids = chunks(all_mids,20)
		#query = mdb.database["post"].find({"message_id":{"$in":mids}},{"message_id":1,"method":1,**UNI_DATE_FIELDS},batch_size=5000)
		i = 0
		for mids in all_mids:
			query = mdb.database["post"].find({"message_id":{"$in":mids}},batch_size=10)
			for post in query:
				#if i % 1000 == 0: print (i)
				dts[post["message_id"]]=Spread()._get_date(data=post,method=post["method"])
				if incl_text:
					texts[post["message_id"]]=Spread()._get_post_message(data=post,method=post["method"])
				i+=1
		#new_df_i_0 = df_i.with_columns(pl.Series(name="text", values=[v for k,v in sorted(texts.items())]))
		#new_df_i = new_df_i_0.replace("dt",pl.Series([v for k,v in sorted(dts.items())]))
		#new_dfs.append(new_df_i)
	mdb.close()
	return (dfs,dts,texts)

def _chunk_neighbors(args):

	chunked_entities = args[0]
	pls = args[1]
	entity_type = args[2]
	exclude = args[3]
	url_min_occur = args[4]
	mdb = MongoSpread()
	url_is_dom = args[6]
	is_directed = args[7]
	dfs = []
	for i,entity_c in enumerate(chunked_entities):
		insert_data = []
		if entity_type == "url":
			if exclude:
				query = mdb.database["url_bi_network"].find({"url":{"$in":list(entity_c)},"platform":{"$in":pls},"actor_platform":{"$nin":exclude},"occurences":{"$gte":url_min_occur}},{"url":1,"actor_platform":1,"message_ids":1})
			else:
				query = mdb.database["url_bi_network"].find({"url":{"$in":list(entity_c)},"platform":{"$in":pls},"occurences":{"$gte":url_min_occur}},{"url":1,"actor_platform":1,"message_ids":1})
		if entity_type == "actor_platform":
			query = mdb.database["url_bi_network"].find({"actor_platform":{"$in":list(entity_c)},"platform":{"$in":pls},"occurences":{"$gte":url_min_occur}},{"url":1,"actor_platform":1,"message_ids":1})
		if entity_type == "domain":
			query = mdb.database["url_bi_network"].find({"domain":{"$in":list(entity_c)},"platform":{"$in":pls},"occurences":{"$gte":url_min_occur}},{"domain":1,"url":1,"message_ids":1,"actor_platform":1})
		
		if is_directed:
			mids = []
			for doc in query:
				for mid in doc["message_ids"]:
					if entity_type == "domain":
						if url_is_dom:
							insert_data.append({"o":doc["url"],"e":str(doc["actor_platform"]),"w":1,"dt":mid})
						else:
							insert_data.append({"o":doc["url"],"e":doc["domain"],"w":1,"dt":mid})
					else:
						insert_data.append({"o":doc["url"],"e":str(doc["actor_platform"]),"w":1,"dt":mid})
		else:
			for doc in query:
				if entity_type == "domain":
					if url_is_dom:
						insert_data.append({"o":doc["url"],"e":str(doc["actor_platform"]),"w":len(doc["message_ids"])})
					else:
						insert_data.append({"o":doc["url"],"e":doc["domain"],"w":len(doc["message_ids"])})
				else:
					insert_data.append({"o":doc["url"],"e":str(doc["actor_platform"]),"w":len(doc["message_ids"])})
		if is_directed:
			if len(insert_data) > 0: dfs.append(pl.DataFrame(insert_data))
		else:
			if len(insert_data) > 0: dfs.append(pl.DataFrame(insert_data))
	mdb.close()
	return dfs

def output_graph_simple(g,net_idx,save_as,add_custom=None):

	rev_net_idx = create_rev_net_idx(g,net_idx)
	g = nxk.nxadapter.nk2nx(g)
	g = nx.relabel_nodes(g,rev_net_idx)
	if add_custom is not None: nx.set_node_attributes(g,add_custom,"custom")
	nx.write_gexf(g,save_as)

def output_graph(g,net_idx,save_as,add_custom=None,metrics=None,affinity_map=None):

	def map_sigmoid(input):

		return 1 / (1 + math.exp(0.000005 * (input - 95000)))

	CMAX = 16000
	rev_net_idx = create_rev_net_idx(g,net_idx)
	g = nxk.nxadapter.nk2nx(g)
	g = nx.relabel_nodes(g,rev_net_idx)
	#nx.set_node_attributes(g,rev_net_idx,"Label")
	if add_custom is not None: nx.set_node_attributes(g,add_custom,"custom")
	if metrics is not None: g = set_metrics(g,metrics,enrich=False,enga_trans=True)
	if g.number_of_nodes() <= CMAX:
		g = filter_based_on_com(g,max_nodes=CMAX,com_var="com_1.0")
	if g.number_of_nodes() > CMAX and g.number_of_nodes() < 1000000:
		g = filter_based_on_com(g,max_nodes=int(g.number_of_nodes()*map_sigmoid(g.number_of_nodes())),com_var="com_1.0")
	print ("writing graph file")
	nx.write_gexf(g,save_as)

@timer
def create_com_output_data(df,com_var="com",strict=True,include_doms=None,projects=None):

	def create_com_rank_df(coms,rank_dfs,com_var,per_com=10):

		wdata = []
		com_rank_dfs = {com:[rdf.filter((pl.col(com_var)==com)).head(per_com+1).to_pandas() for rdf in rank_dfs] for com in coms}
		for com in coms:
			rank_dfs = com_rank_dfs[com]
			for r in range(per_com):
				doc = {"rank":r+1}
				for rdf in rank_dfs:
					if len(rdf) > 0:
						if len(rdf) > r:
							doc.update(rdf.iloc[[r]].to_dict(orient="records")[0])
				wdata.append(doc)
		
		wdf = pd.DataFrame(wdata)
		wdf = pl.from_pandas(wdf)
		return wdf
	
	print ("Creating data export")
	com_sizes = defaultdict(int)
	for node,com in zip(df["actor_platform"].to_list(),df[com_var].to_list()):
		com_sizes[com]+=1
	df = df.filter(~pl.col("link_to_actor").is_in(set(df.filter((pl.col("platform")=="twitter")&(pl.col("shares_mean")<1))["link_to_actor"].to_list()))) 
	df = df.sort([com_var,"pagerank_wEnga^2"], descending=True)
	com_df = df.groupby(com_var).agg(
		size=pl.count(),
		mean_pagerank_enga_squared=pl.mean("pagerank_wEnga^2"),
	)
	com_top2_lang_df = df.select(pl.col([com_var,"lang"])).groupby([com_var,"lang"]).agg(lang_count=pl.count()).sort([com_var,"lang_count"],descending=True).groupby(com_var).head(5)
	com_top3_pl_df = df.select(pl.col([com_var,"platform"])).groupby([com_var,"platform"]).agg(platform_count=pl.count()).sort([com_var,"platform_count"],descending=True).groupby(com_var).head(5)
	com_top10_a_df = df.select(pl.col([com_var,"actor_name","actor_platform","link_to_actor"])).groupby([com_var]).head(20)
	com_top10_0a_df = df.filter((pl.col("custom")==0)).select(pl.col([com_var,"actor_name","actor_platform","link_to_actor"])).groupby([com_var]).head(20)
	com_top10_0a_df = com_top10_0a_df.rename({"actor_name":"0_actor_name","actor_platform":"0_actor_platform","link_to_actor":"0_link_to_actor"})
	com_top10_1a_df = df.filter((pl.col("custom")==1)).select(pl.col([com_var,"actor_name","actor_platform","link_to_actor"])).groupby([com_var]).head(100)
	com_top10_1a_df = com_top10_1a_df.rename({"actor_name":"1_actor_name","actor_platform":"1_actor_platform","link_to_actor":"1_link_to_actor"})
	com_top10_2a_df = df.filter((pl.col("custom")==3)).select(pl.col([com_var,"actor_name","actor_platform","link_to_actor"])).groupby([com_var]).head(100)
	com_top10_2a_df = com_top10_2a_df.rename({"actor_name":"3_actor_name","actor_platform":"3_actor_platform","link_to_actor":"3_link_to_actor"})
	#df = df.with_columns(pl.col("dt").map_dict(texts).alias("text"))

	out_df = pl.from_dict({com_var:[k for k,v in sorted(com_sizes.items())],"size":[v for k,v in sorted(com_sizes.items())]})
	out_df = out_df.join(df.select(pl.col([com_var,"pagerank_wEnga^2","n_posts","pagerank","norm_engagement","in_degree","out_degree","btw_c","custom",f"conductance_{com_var}"]))\
		      .groupby(com_var)\
				.agg(mean_pagerank_enga_squared=pl.mean("pagerank_wEnga^2"),
			    mean_n_posts=pl.mean("n_posts"),
				mean_pagerank=pl.mean("pagerank"),
				mean_norm_engagement=pl.mean("norm_engagement"),
				mean_in_degrees=pl.mean("in_degree"),
				mean_out_degress=pl.mean("out_degree"),
				mean_btw_c=pl.mean("btw_c"),
				conductance=pl.mean(f"conductance_{com_var}"),
				mean_iteration=pl.mean("custom")), on=com_var)
	for col in df.columns:
		if "partisan" in col:
			out_df = out_df.join(df.select(pl.col([com_var,col]))\
				.groupby(com_var)\
					.agg(
					pl.mean(col)), on=com_var)

	if "alt_share" in df.columns:
		out_df = out_df.join(df.select(pl.col([com_var,"alt_share"]))\
			.groupby(com_var)\
				.agg(
				pl.mean("alt_share")), on=com_var)

	out_df = out_df.join(df.filter((pl.col("custom")==0)).select(pl.col([com_var,"custom"])).groupby(com_var).agg(count_0=pl.count()), on=com_var,how="outer")
	out_df = out_df.join(df.filter((pl.col("custom")==1)).select(pl.col([com_var,"custom"])).groupby(com_var).agg(count_1=pl.count()), on=com_var,how="outer")
	out_df = out_df.with_columns((pl.col("count_0")/pl.col("size")).alias("0_prop"))
	out_df = out_df.with_columns((pl.col("count_1")/pl.col("size")).alias("1_prop"))
	out_df.drop_in_place("count_0")
	out_df.drop_in_place("count_1")
	out_df = out_df.with_columns(pl.col("0_prop").fill_null(strategy="zero"))
	out_df = out_df.with_columns(pl.col("1_prop").fill_null(strategy="zero"))
	out_df = out_df.join(com_top3_pl_df.groupby(com_var).head(1), on=com_var)
	out_df = out_df.join(com_top2_lang_df.groupby(com_var).head(1), on=com_var)
	if "inner" in com_var:
		out_df = out_df.join(df.select(pl.col([com_var,com_var.replace("_inner","")])).groupby([com_var]).head(1),on=com_var)
	out_df = out_df.join(com_top10_0a_df.groupby(com_var).agg(pl.col('0_actor_name')), on=com_var, how="outer")
	out_df = out_df.with_columns(pl.format("[{}]",pl.col("0_actor_name").cast(pl.List(pl.Utf8)).list.join(", ")).alias("0_actor"))
	out_df.drop_in_place("0_actor_name")
	if include_doms is not None:
		dom_per_actor = include_doms
		dom_per_actor = dom_per_actor.join(df.select(pl.col(["actor_platform",com_var])),on="actor_platform",how="left")
		dom_per_com = dom_per_actor.groupby([com_var,"domain"]).agg(idf=pl.mean("idf"),count=pl.sum("count"))
		
		dom_per_com = dom_per_com.with_columns((pl.col("count")*pl.col("idf")).alias("tfidf"))
		dom_per_com = dom_per_com.with_columns(np.log((pl.col("count")+1)*(pl.col("idf")**2.5)).alias("tfidf_lognorm"))
		dom_per_com = dom_per_com.sort([com_var,"tfidf_lognorm"], descending=True)
		
		dom_per_com_total = dom_per_com.sort([com_var,"count"], descending=True)
		dom_per_com_total = dom_per_com_total.select(pl.col(com_var,"domain","count"))
		dom_per_com_total = dom_per_com_total.rename({"count":"domain_total_count","domain":"domain_total"})

		if projects is not None:
			p_actros = set([actor for k,v in import_labels(projects=projects).items() for actor in v.keys()])
			dom_per_com_anm = dom_per_com.filter((pl.col("domain").is_in(p_actros))).sort([com_var,"count"], descending=True)
			dom_per_com_anm = dom_per_com_anm.select(pl.col(com_var,"domain","count"))
			dom_per_com_anm = dom_per_com_anm.rename({"count":"domain_anm_count","domain":"domain_anm"})
			out_df = out_df.join(dom_per_com_anm.select(pl.col(com_var,"domain_anm")).groupby(com_var).head(5).groupby(com_var).agg(pl.col('domain_anm')), on=com_var, how="outer")
			out_df = out_df.with_columns(pl.format("[{}]",pl.col("domain_anm").cast(pl.List(pl.Utf8)).list.join(", ")).alias("domain_anms"))
			out_df.drop_in_place("domain_anm")
		
	else:
		dom_per_com, dom_per_com_total,dom_per_com_anm = pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
	if strict:
		com_rank_df = create_com_rank_df(set([k for k,v in com_sizes.items() if v > 8]),[com_top10_a_df,com_top2_lang_df,com_top3_pl_df,com_top10_0a_df,com_top10_1a_df,com_top10_2a_df,dom_per_com,dom_per_com_total,dom_per_com_anm],com_var,per_com=20)
	else:
		com_rank_df = pl.DataFrame()

	print ("Exporting data")
	return com_df,com_top2_lang_df,com_top3_pl_df,com_top10_a_df,com_sizes,out_df,com_rank_df

def output_data(metrics,net_idx,main_path,title,path_step):

	docs = []
	mets = list(metrics.keys())
	nodes = set(list(metrics[mets[0]].keys()))
	print (len(nodes))
	#nodes = nodes.intersection(*[set(list(v.keys())) for k,v in metrics.items()])
	nodes.union(*[set(list(v.keys())) for k,v in metrics.items()])
	print (len(nodes))
	for node in nodes:
		if node in net_idx:
			doc = {"actor_platform":node}
			for met in mets:
				if node not in metrics[met]:
					doc[met]=0
				else:
					doc[met]=metrics[met][node]
			docs.append(doc)
	df = pl.from_dicts(docs)
	df.write_csv(main_path+f"/{title}_{path_step}steps_data.csv")

def coms_data_to_textfile(df,langs,pls,actors,com_sizes,most_sim_com,most_sim_val,com_var,save_as):

	with open(save_as, 'w') as out:
		for com,size in sorted(com_sizes.items(), key=lambda x:x[1], reverse=True):
			if size < 3: continue
			out.write(f"Com: {com} - size: {size}")
			out.write("\n\n")
			out.write(str(df.filter((pl.col(com_var)==com))))
			out.write("\n\n")
			out.write("Top Actors:\n")
			for a in actors.filter((pl.col(com_var)==com))["actor_name"].to_list():
				out.write(a)
				out.write("\n")
			out.write("\n\n")
			out.write("Top Langs:\n")
			out.write(str(langs.filter((pl.col(com_var)==com)).with_columns((pl.col("lang_count")/size).alias("%"))))
			out.write("\n\n")
			out.write("Top Platforms:\n")
			out.write(str(pls.filter((pl.col(com_var)==com)).with_columns((pl.col("platform_count")/size).alias("%"))))
			out.write("\n\n")
			if int(com) in most_sim_com:
				out.write(f"Most similar com in previous iteration: {most_sim_com[int(com)]} ({most_sim_val[int(com)]*100}%)")
			else:
				out.write(f"Most similar com in previous iteration: {None} ({None}%)")
			out.write("\n")
			out.write("\n")
			out.write("--------------------------------------------------------------------------------")
			out.write("\n\n\n\n")

def get_clean_actor_info(mdb,pls,projects=["altmed_denmark"],iterations=[0,0.0],add_fields=[]):
	
	data = []
	for doc in mdb.database["actor"].find({"org_project_title":{"$in":projects},"Iteration":{"$in":iterations}}):
		dom = LinkCleaner().extract_special_url(doc["Website"])
		pro = doc["org_project_title"]
		a = doc["Actor"]
		new_doc = {"source":pro,"domain":dom,"actor":a}
		for pl in pls:
			if pl == "facebook":
				new_doc[pl]=LinkCleaner().extract_username(doc["Facebook Page"])
			if pl.capitalize() in doc:
				new_doc[pl]=LinkCleaner().extract_username(doc[pl.capitalize()])
		for field in add_fields:
			new_doc[field]=doc[field]
		data.append(new_doc)
	return pd.DataFrame(data)

@timer
def get_domain_share(projects=[]):

	print ("getting domain share")
	pls = ["twitter","facebook","instagram","telegram","gab","youtube","fourchan","vkontakte","reddit"]
	mdb = MongoSpread()
	out_data = defaultdict(int)
	if len(projects) > 0:
		data = get_clean_actor_info(mdb,pls,projects=projects)
		doms = set(list(data["domain"]))
		for doc in mdb.database["url_bi_network"].aggregate([{"$match":{"domain":{"$in":list(doms)}}},{"$group":{"_id":{"actor_platform":"$actor_platform", "domain":"$domain"}, "total":{"$sum":{"$size":"$message_ids"}}}}]):
			data_doc = {"actor_platform":doc["_id"]["actor_platform"],"domain":doc["_id"]["domain"],"count":doc["total"]}
			out_data[data_doc["actor_platform"]]+=data_doc["count"]
	return {"alt_share":out_data}

def get_actor_platform_from_actor_info(mdb,actor_info,pls,strict=True,verbose=True):

	aps = {}
	supposed_to_find = set([])
	pls_in_data = set(actor_info.columns)
	for i,actor_row in actor_info.iterrows():
		for pl in pls:
			if pl in actor_row and actor_row[pl] is not None and not isinstance(actor_row[pl],float):
				a = actor_row["actor"]
				if pl == "facebook":
					ap_ = actor_row["actor"]+"_"+"Facebook Page"
					ap__ = actor_row[pl]+"_"+"Facebook Page"
				else:
					ap_ = actor_row["actor"]+"_"+pl.capitalize()
					ap__ = actor_row[pl]+"_"+pl.capitalize()
				supposed_to_find.add(ap_)
				if a is not None:
					if strict == True:
						query = mdb.database["actor_platform_post"].find({"$or":[{"actor_platform":ap_},{"actor_platform":ap__}]})
					else:
						query = mdb.database["actor_platform_post"].find({"$or":[{"actor":a},{"actor_platform":ap_},{"actor_platform":ap__}]})
					for ap_doc in query:
						ap = ap_doc["actor_platform"]
						aps[ap]=a
	for a in supposed_to_find:
		if a not in aps:
			if verbose:
				print (a)
	if verbose:
		print (f"Supposed to find: {len(supposed_to_find)}")
		print (f"Found: {len(aps)}")
	return aps

def import_labels(projects=None,label_vars=["Partisan"]):

	#df = pd.read_excel(Config().PROJECT_PATH+f"/{project}/Actors.xlsx")
	pls = ["twitter","facebook","instagram","telegram","gab","youtube","fourchan","vkontakte","reddit"]
	mdb = MongoSpread()
	actor_labels = {l:{} for l in label_vars}
	if projects is not None:
		seed_actors = get_clean_actor_info(mdb,pls,projects=projects,add_fields=label_vars)
		for i,row in seed_actors.iterrows():
			for l in label_vars:
				if row[l] is not None and len(str(row[l])) > 0:
					label = row[l]
					for ap in get_actor_platform_from_actor_info(mdb,seed_actors[seed_actors["actor"]==row["actor"]],pls,verbose=False):
						actor_labels[l][ap]=label.lower()
					actor_labels[l][row["domain"]]=label.lower()
	mdb.close()
	return actor_labels

def import_data(projects=None,actors=[],domains=[],num_cores=8,platforms=None,steps=1,directed=False,save_as=None,new=False,keep_org_degree=False,zero_deg_domain_constraint=[],keep_doms_actors=False,incl_text=False):

	@timer
	def get_neighbors(entities,entity_type="url",mdb=None,format_="polars",pls=[],verbose=True,url_min_occur=2,exclude=[],num_cores=1,url_is_dom=False,is_directed=False,incl_text=False):

		if not pls:
			pls = ["twitter","facebook","instagram","telegram","gab","youtube","fourchan","vkontakte","reddit"]
		dfs = []
		if len(entities) > 20*num_cores:
			n_chunks = 20*num_cores
		else:
			n_chunks = num_cores
		entities = list(entities)
		random.shuffle(entities)
		chunked_entities = chunks(entities,n_chunks)
		if num_cores < 2:
			results = [_chunk_neighbors([chunked_entities,pls,entity_type,exclude,url_min_occur,None,url_is_dom,is_directed])]
		else:
			results = Pool(num_cores).map(_chunk_neighbors,[[cc,pls,entity_type,exclude,url_min_occur,None,url_is_dom,is_directed] for cc in chunks(list(chunked_entities),num_cores)])
		for result in results:
			if is_directed:
				dfs.extend(result)
			else:
				dfs.extend(result)
		if is_directed:
			if num_cores > 1:
				results = Pool(num_cores).map(_mid_to_dt,[(c,incl_text) for c in chunks(dfs,num_cores)])
			else:
				results = [_mid_to_dt((dfs,incl_text))]
			dfs = []
			texts = {}
			dts = {}
			for result in results:
				texts.update(result[2])
				dts.update(result[1])
				dfs.extend(result[0])
			df = pl.concat(dfs)
			if incl_text:
				df = df.with_columns(pl.col("dt").map_dict(texts).alias("text"))
			df = df.replace("dt",df["dt"].map_dict(dts))
			df = df.with_columns(pl.col("dt").str.to_datetime("%Y-%m-%d %H:%M:%S"))
		else:
			df = pl.concat(dfs)

		return df

	if os.path.isfile(save_as) and not new:
		return pl.read_csv(save_as)

	mdb = MongoSpread()
	final_graph = None
	finalized = False

	if platforms is None:
		pls = ["twitter","facebook","instagram","telegram","gab","youtube","fourchan","vkontakte","reddit"]
	else:
		pls = platforms
	if projects is not None:
		seed_actors = get_clean_actor_info(mdb,pls,projects=projects)
		if actors:
			seed_actors = seed_actors[seed_actors["actor"].isin(set(actors))]
		seed_actor_platform = get_actor_platform_from_actor_info(mdb,seed_actors,pls)
		seed_actor_platform = list(seed_actor_platform.keys())
		seed_domains = list(set(list(seed_actors["domain"])))
		zero_deg_domain_constraint = seed_domains
	else:
		seed_actor_platform = actors
		seed_domains = domains
	mdb.close()

	print ("Getting zero degs...")
	if seed_actor_platform:
		shared_by_alt = get_neighbors(seed_actor_platform,entity_type="actor_platform",pls=pls,is_directed=directed,incl_text=incl_text,num_cores=num_cores)
		print (len(shared_by_alt))
	if seed_domains:
		if projects is not None:
			alt_shared_dom = get_neighbors(seed_domains,entity_type="domain",pls=pls,url_is_dom=False,is_directed=directed,incl_text=incl_text,num_cores=num_cores)
		else:
			alt_shared_dom = get_neighbors(seed_domains,entity_type="domain",pls=pls,url_is_dom=False,is_directed=directed,incl_text=incl_text,num_cores=num_cores)
			#alt_shared_dom = get_neighbors(seed_domains,entity_type="domain",pls=pls,url_is_dom=True,is_directed=directed)
		print (len(alt_shared_dom))

	if projects is not None:
		alt_shared_dom = alt_shared_dom.filter(~pl.col('o').is_in(shared_by_alt["o"].to_list()))
		print (len(alt_shared_dom))
		zero_deg = pl.concat([shared_by_alt,alt_shared_dom])
	elif seed_actor_platform and seed_domains:
		zero_deg = pl.concat([shared_by_alt,alt_shared_dom])
		zero_deg = zero_deg.unique(subset=["o","e","w"])
	elif seed_actor_platform and not seed_domains:
		zero_deg = shared_by_alt
	elif not seed_actor_platform and seed_domains:
		zero_deg = alt_shared_dom
	else:
		print ("Something went wrong with data inputs... No data")
		sys.exit()
	
	if zero_deg_domain_constraint:
		keep_urls = []
		for dom in zero_deg_domain_constraint:
			for url in zero_deg["o"].to_list():
				if str(dom) in str(url):
					keep_urls.append(url)
		zero_deg = zero_deg.filter(pl.col("o").is_in(set(keep_urls)))
			
	if keep_org_degree:
		zero_deg = zero_deg.with_columns(pl.lit(0).alias("org_degree"))

	if steps > 1:
		print ("Getting first degs...")
		alt_shared_by_others = get_neighbors(zero_deg.unique(subset=["o"])["o"].to_list(),entity_type="url",pls=pls,is_directed=directed,num_cores=num_cores,incl_text=incl_text)
		print (len(alt_shared_by_others))
		first_deg = alt_shared_by_others.filter(~pl.col('e').is_in(zero_deg["e"].to_list()))
		print (len(first_deg))
		if keep_org_degree:
			first_deg = first_deg.with_columns(pl.lit(1).alias("org_degree"))
			first_deg = pl.concat([zero_deg,first_deg]).sort(["org_degree"], descending=False).unique(subset=["o","e","w"],keep="first")
		else:
			first_deg =  pl.concat([zero_deg,first_deg]).unique(subset=["o","e","w"])
		if not keep_doms_actors:
			first_deg = first_deg.filter(~pl.col("e").is_in(list(seed_domains)))
		if True:
			first_deg = filter_df_on_degrees(first_deg,"e")
		print (len(first_deg))
		first_deg = first_deg.filter(pl.col("e")!=pl.col("o"))
		print (len(first_deg))
	else:
		final_graph = zero_deg
		finalized = True
	
	if steps > 2 and not finalized:
		first_deg_search = list(first_deg.unique(subset=["e"])["e"].to_list())
		print (f"Getting first degs - n = {len(first_deg_search)}")
		shared_by_first_deg = get_neighbors(first_deg_search,entity_type="actor_platform",num_cores=num_cores,pls=pls,is_directed=directed,incl_text=incl_text)
		print (len(shared_by_first_deg))
		if keep_org_degree:
			shared_by_first_deg = shared_by_first_deg.with_columns(pl.lit(2).alias("org_degree"))
			second_deg = pl.concat([first_deg,shared_by_first_deg]).sort(["org_degree"], descending=False).unique(subset=["o","e","w"],keep="first")
		else:
			second_deg = pl.concat([first_deg,shared_by_first_deg]).unique(subset=["o","e","w"])
		print (len(second_deg))
	else:
		if not finalized:
			final_graph = first_deg
			finalized = True
		
	if steps > 3 and not finalized:
		if projects is not None or (seed_actor_platform and seed_domains):
			second_deg_search = list(second_deg.filter(~pl.col("o").is_in(zero_deg["o"].to_list())).unique(subset=["o"])["o"].to_list())
		else:
			#second_deg_search = random.sample(second_deg.unique(subset=["o"])["o"].to_list(),10000)
			second_deg_search = second_deg.unique(subset=["o"])["o"].to_list()
		print (f"Getting second degs - n = {len(second_deg_search)}")
		coshared_second_deg = get_neighbors(second_deg_search,entity_type="url",num_cores=num_cores,pls=pls,is_directed=directed,incl_text=incl_text)
		if keep_org_degree:
			coshared_second_deg = coshared_second_deg.with_columns(pl.lit(3).alias("org_degree"))
			full = pl.concat([second_deg,coshared_second_deg]).sort(["org_degree"], descending=False).unique(subset=["o","e","w"],keep="first")
		else:
			full = pl.concat([second_deg,coshared_second_deg]).unique(subset=["o","e","w"])
		print (len(full))
		full = full.unique(subset=["o","e","w"])
		print (len(full))
		full = full.filter(~pl.col("e").is_in(full.groupby(["e"]).agg(pl.count()).filter(pl.col("count")<2)["e"].to_list()))
		print (len(full))
		final_graph = full
	else:
		if not finalized:
			final_graph = second_deg
	if incl_text:
		final_graph = filter_urls_based_on_texts(final_graph,num_cores=num_cores)
	if save_as is not None:
		final_graph.write_csv(save_as)

	return final_graph

