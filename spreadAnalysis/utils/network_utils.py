import community
import networkx as nx
import math
import numpy as np
from operator import itemgetter
from multiprocessing import Pool, Manager
import random
import pandas as pd
from spreadAnalysis.utils import helpers as hlp
from scipy import integrate

def disparity_filter_alpha_cut_multi(G,weight='weight',alpha_t=0.4, cut_mode='or'):

	B = nx.MultiGraph()#Undirected case:
	for u, v in list(G.edges()):
		for k in list(G[u][v].keys()):
			try:
				alpha = G[u][v][k]['alpha']
			except KeyError: #there is no alpha, so we assign 1. It will never pass the cut
				alpha = 1
			if alpha<alpha_t:
				B.add_edge(u,v,key=k, weight=G[u][v][k]['weight'])
	return B

def disparity_filter_multi(G,keys=[], weight='weight'):

	B = nx.MultiGraph()
	for u in G:
		k = len(G[u])
		if k > 1:
			for ke in keys:
				sum_w = sum(np.absolute(G[u][v][ke][weight]) for v in G[u] if ke in G[u][v])
				for v in G[u]:
					if ke in G[u][v]:
						w = G[u][v][ke][weight]
						p_ij = float(np.absolute(w))/sum_w
						alpha_ij = 1 - (k-1) * integrate.quad(lambda x: (1-x)**(k-2), 0, p_ij)[0]
						B.add_edge(u, v, key=ke, weight = w, alpha=float('%.4f' % alpha_ij))
	return B

def disparity_filter_alpha_cut(G,weight='weight',alpha_t=0.4, cut_mode='or'):

	if nx.is_directed(G):#Directed case:
		B = nx.DiGraph()
		for u, v, w in G.edges(data=True):
			try:
				alpha_in =  w['alpha_in']
			except KeyError: #there is no alpha_in, so we assign 1. It will never pass the cut
				alpha_in = 1
			try:
				alpha_out =  w['alpha_out']
			except KeyError: #there is no alpha_out, so we assign 1. It will never pass the cut
				alpha_out = 1

			if cut_mode == 'or':
				if alpha_in<alpha_t or alpha_out<alpha_t:
					B.add_edge(u,v, weight=w[weight])
			elif cut_mode == 'and':
				if alpha_in<alpha_t and alpha_out<alpha_t:
					B.add_edge(u,v, weight=w[weight])
		return B

	else:
		B = nx.Graph()#Undirected case:
		for u, v, w in G.edges(data=True):

			try:
				alpha = w['alpha']
			except KeyError: #there is no alpha, so we assign 1. It will never pass the cut
				alpha = 1

			if alpha<alpha_t:
				B.add_edge(u,v, weight=w[weight])
		return B

def disparity_filter(G, weight='weight'):

	if nx.is_directed(G): #directed case
		N = nx.DiGraph()
		for u in G:

			k_out = G.out_degree(u)
			k_in = G.in_degree(u)

			if k_out > 1:
				sum_w_out = sum(np.absolute(G[u][v][weight]) for v in G.successors(u))
				for v in G.successors(u):
					w = G[u][v][weight]
					p_ij_out = float(np.absolute(w))/sum_w_out
					alpha_ij_out = 1 - (k_out-1) * integrate.quad(lambda x: (1-x)**(k_out-2), 0, p_ij_out)[0]
					N.add_edge(u, v, weight = w, alpha_out=float('%.4f' % alpha_ij_out))

			elif k_out == 1 and G.in_degree(G.successors(u)[0]) == 1:
				#we need to keep the connection as it is the only way to maintain the connectivity of the network
				v = G.successors(u)[0]
				w = G[u][v][weight]
				N.add_edge(u, v, weight = w, alpha_out=0., alpha_in=0.)
				#there is no need to do the same for the k_in, since the link is built already from the tail

			if k_in > 1:
				sum_w_in = sum(np.absolute(G[v][u][weight]) for v in G.predecessors(u))
				for v in G.predecessors(u):
					w = G[v][u][weight]
					p_ij_in = float(np.absolute(w))/sum_w_in
					alpha_ij_in = 1 - (k_in-1) * integrate.quad(lambda x: (1-x)**(k_in-2), 0, p_ij_in)[0]
					N.add_edge(v, u, weight = w, alpha_in=float('%.4f' % alpha_ij_in))
		return N

	else: #undirected case
		B = nx.Graph()
		for u in G:
			k = len(G[u])
			if k > 1:
				sum_w = sum(np.absolute(G[u][v][weight]) for v in G[u])
				for v in G[u]:
					w = G[u][v][weight]
					p_ij = float(np.absolute(w))/sum_w
					alpha_ij = 1 - (k-1) * integrate.quad(lambda x: (1-x)**(k-2), 0, p_ij)[0]
					B.add_edge(u, v, weight = w, alpha=float('%.4f' % alpha_ij))
		return B

def add_score_based_label(g,att_name,att_vals):

	def get_node_score_medians(g,atts):

		df = pd.DataFrame([d for n,d in list(g.nodes(data=True))])[[a+"_score" for a in atts]]
		medians = df.median()
		return dict(medians)

	def _is_all_low(scores):

		return all(s <= 0.0 for s in scores)

	if isinstance(g,str):
		g = nx.read_gexf(g)
	medians = get_node_score_medians(g,att_vals)
	try:
		default_grey_label = [a for a in medians.keys() if "grey" in a][0]
	except:
		default_grey_label = None
	for n,d in list(g.nodes(data=True)):
		stdzied = {a+"_score":d[a+"_score"]-medians[a+"_score"] for a in att_vals}
		if _is_all_low(list(stdzied.values())) and default_grey_label is not None:
			label = default_grey_label
		else:
			label = sorted(stdzied.items(), key = itemgetter(1), reverse=True)[0][0]
		nx.set_node_attributes(g,{n:label},att_name)
	return g

def approximate_eccentricity(g):

	try:
		eccs = [nx.eccentricity(g,n) for n in random.sample(list(g.nodes()),100)]
	except:
		eccs = [3.0]

	return float(np.mean(np.array(eccs)))

def set_neighbour_affinity(args):

	new_node_atts = {}
	node_edges = args[0]
	node_atts = args[1]
	unique_atts = args[2]
	diam = args[3]
	grey_number = args[4]
	for n,edges in node_edges.items():
		n_edges = 0
		new_atts = {a:0.0 for a in unique_atts}
		avg_dists = []
		min_dists = []
		dist_weights = []
		for org,edg,w in edges:
			main_node = org
			avg_dists.append(node_atts_gl[edg]["dist_to_0g{0}".format(grey_number)])
			dist_weights.append(w["weight"])
			for att in unique_atts:
				new_atts[att]+=np.log(w["weight"]+2)*node_atts_gl[edg][att]*(node_atts_gl[edg]["dist_to_0g{0}".format(grey_number)]+1)**-1
		atts_sum = float(np.sum(np.array(list(new_atts.values()))))
		if atts_sum > 0:
			new_atts = {a:float(v)/atts_sum for a,v in list(new_atts.items())}
		mean_dist = float(np.average(np.array(avg_dists),weights=np.array(dist_weights)))
		min_dist = float(np.amin(np.array(avg_dists)))
		dist_score = mean_dist*float((np.log((mean_dist-min_dist)+1.0)+1.0)**-1)+1.0
		if dist_score < 1.0:
			dist_score = 1.0
		new_atts["dist_to_0g{0}".format(grey_number)]=dist_score
		new_node_atts[main_node]=new_atts

	return new_node_atts

def set_neighbour_affinity_score(args):

	new_node_atts = {}
	node_edges = args[0]
	node_atts = args[1]
	unique_atts = args[2]
	grey_number = args[3]
	for n,edges in node_edges.items():
		new_atts = dict(node_atts_gl[n])
		avg_dist = new_atts["dist_to_0g{0}".format(grey_number)]
		new_atts.update({str(a)+"_score":float(v)*np.log(len(edges)+1)*((avg_dist+1)**-1) for a,v in list(new_atts.items()) if a in unique_atts})
		new_node_atts[n]=new_atts
	return new_node_atts

def add_affinity_scores(g,actor_mapping,num_cores=12,grey_number=0,batch_size=2,noise=True,exclude_from_grey=None):

	manager = Manager()
	global node_atts_gl
	node_atts_gl = manager.dict()
	actor_mapping = {k:v for k,v in actor_mapping.items() if k in g}
	n_actors = len(actor_mapping)
	not_mapped_actors = [a for a in g.nodes() if a not in actor_mapping]
	if exclude_from_grey is not None:
		not_mapped_actors = [a for a,d in g.nodes(data=True) if a not in actor_mapping and d[exclude_from_grey[0]]!=exclude_from_grey[1]]
	if noise:
		if len(not_mapped_actors) > 2*n_actors:
			grey_actors = {a:"grey{}".format(grey_number) for a in random.sample(not_mapped_actors,n_actors)}
		else:
			grey_actors = {a:"grey{}".format(grey_number) for a in random.sample(not_mapped_actors,int(n_actors*0.5))}
		actor_mapping.update(grey_actors)
	else:
		grey_actors = {}
	unique_atts = set([v for k,v in actor_mapping.items()])
	node_atts_gl = {n:{ua:0.0 for ua in unique_atts} for n in g.nodes()}
	#diam = float(nx.diameter(g))
	diam = approximate_eccentricity(g)-1.0
	for n in list(g.nodes()):
		if n in actor_mapping:
			node_atts_gl[n][actor_mapping[n]]=1.0
			node_atts_gl[n]["dist_to_0g{0}".format(grey_number)]=0.0
		else:
			node_atts_gl[n]["dist_to_0g{0}".format(grey_number)]=diam
	pool = Pool(num_cores)
	its = 10
	for r in range(its):
		print ("Adding affinities. Iteration {0} out of {1}".format(r,its))
		net_nodes = list(g.nodes())
		random.shuffle(net_nodes)
		for nodes_chunk in hlp.chunks(net_nodes,int(len(net_nodes)/batch_size)+1):
			temp_docs = {n:list(g.edges(n,data=True)) for n in nodes_chunk if n not in actor_mapping}
			multi_inputs = [(l,dict({}),unique_atts,diam,grey_number) for l in list(hlp.chunks_optimized(temp_docs,n_chunks=num_cores))]
			#print (dict(psutil.virtual_memory()._asdict()))
			results = pool.map(set_neighbour_affinity,multi_inputs)
			for result in results:
				node_atts_gl.update(result)
	temp_docs = {n:list(g.edges(n,data=True)) for n in list(g.nodes()) if n in grey_actors}
	node_atts_gl.update(set_neighbour_affinity((temp_docs,{},unique_atts,diam,grey_number)))
	temp_docs = {n:list(g.edges(n,data=True)) for n in list(g.nodes())}
	node_atts_gl.update(set_neighbour_affinity_score((temp_docs,{},unique_atts,grey_number)))
	for att in unique_atts:
		nx.set_node_attributes(g,{n:node_atts_gl[n][att] for n in list(g.nodes())} ,att)
		nx.set_node_attributes(g,{n:node_atts_gl[n][att+"_score"] for n in list(g.nodes())} ,att+"_score")
	nx.set_node_attributes(g,{n:node_atts_gl[n]["dist_to_0g{0}".format(grey_number)] for n in list(g.nodes())} ,"dist_to_0g{0}".format(grey_number))
	return g

class NetworkUtils:

	@classmethod
	def revalue_edge_weights(cls,g):

		def max_log(val):
			return np.log((float(val)+1.0))

		print ("Inverting edge weigths...")
		max_edge_weight = float(np.amax(np.array(list([float(dat["weight"]) for u,v,dat in list(g.edges(data=True))]))))
		print ("Max edge weight is: {0}".format(max_edge_weight))
		for u,v,dat in list(g.edges(data=True)):
			if g.has_edge(u,v) and dat["weight"] > 0:
				if max_edge_weight > 10000:
					g.get_edge_data(u,v)['weight']=(max_log(max_edge_weight)+(max_log(dat["weight"])/max_log(max_edge_weight)))
				else:
					g.get_edge_data(u,v)['weight']=(max_edge_weight+(dat["weight"]/max_edge_weight))
		return g

	@classmethod
	def add_communities(cls,g,get_com_sizes=False):
		com_nodes = {}
		int_node_mapping = dict(zip(g.nodes(),range(0,len(g.nodes()))))
		g = nx.relabel_nodes(g,int_node_mapping)
		partition = community.best_partition(g.to_undirected(),resolution=0.5)
		print("Number of communities: " + str(len(set(partition.values()))))
		for node, com in partition.items():
			if not com in com_nodes: com_nodes[com]=[]
			com_nodes[com].append(node)
			g.nodes[node]["modularity"]=com
		com_sizes = {com:len(nodes) for com,nodes in com_nodes.items()}
		g = nx.relabel_nodes(g,{y:x for x,y in int_node_mapping.items()})
		if get_com_sizes:
			return g, com_sizes
		else:
			return g


	@classmethod
	def filter_on_most_important_edges(cls,g,keep_percent=0.25,max_fixed=100000,tfidf=False,weighted=True,skip_nodes={}):

		print("Removing less important edges.")
		total_docs = {}
		edges_count = {}
		approved_edges = {}
		edges = g.copy().edges(data=True)
		for u,v,dat in edges:
			e1 = u
			e2 = v
			if e1 not in total_docs: total_docs[e1]=0.0
			if e2 not in total_docs: total_docs[e2]=0.0
			total_docs[e1]+=1.0
			total_docs[e2]+=1.0
			if e1 not in edges_count:
				edges_count[e1]={}
			if e2 not in edges_count[e1]:
				if weighted:
					edges_count[e1][e2]=dat["weight"]
				else:
					edges_count[e1][e2]=1.0

			if len(skip_nodes) > 0:
				if u in skip_nodes:
					approved_edges[u,v]=True
				if v in skip_nodes:
					approved_edges[u,v]=True

		if tfidf == True:
			new_edges_count = {}
			total_docs = len(total_docs)
			for edge,con in edges_count.items():
				new_edges_count[edge]={}
				inverse_doc_freqs = math.log(total_docs/float(len(con)))
				for con_edge,weight in con.items():
					tfidf_score = inverse_doc_freqs*math.log(weight)
					if float(tfidf_score) > 0.0:
						new_edges_count[edge][con_edge]=float(weight)/float(tfidf_score)
					else:
						new_edges_count[edge][con_edge]=0.0
			edges_count = new_edges_count

		for edge,con in edges_count.items():
			con_count = 0
			if keep_percent >= 1.0:
				cut_ = math.log(len(con))*keep_percent
			else:
				cut_ = len(con)*keep_percent
			for con_edge,weight in sorted(con.items(), key=itemgetter(1), reverse=True)[:int(cut_)]:
				con_count+=1
				if con_count < max_fixed:
					approved_edges[edge,con_edge]=True

		for u,v,dat in edges:
			if (u,v) not in approved_edges:
				g.remove_edge(u,v)

		return g

	@classmethod
	def giant_component(cls,g):
		gc = set(sorted(nx.connected_components(g.to_undirected()), key = len, reverse=True)[0])
		for node in list(g.nodes()):
			if not node in gc:
				g.remove_node(node)
		return g

	@classmethod
	def add_eigenvector_centrality(cls,g):
		print ("Adding eigenvector centralities...")
		eigs = nx.eigenvector_centrality(g,max_iter=6900)
		nx.set_node_attributes(g, eigs, 'eigenvector_centrality')
		return g

	@classmethod
	def add_degrees(cls,g):
		print ("Adding degrees...")
		degs = {node:g.degree[node] for node in list(g.nodes())}
		nx.set_node_attributes(g, degs, 'degrees')
		return g

	@classmethod
	def add_pagerank(cls,g,skip_nodes={}):
		print ("Adding page ranks...")
		pgs = nx.pagerank(g)
		if len(skip_nodes) > 0: pgs = {k:v if k not in skip_nodes else 0.0 for k,v in pgs.items()}
		nx.set_node_attributes(g, pgs,'pagerank')
		return g

	@classmethod
	def filter_by_degrees(cls,g,degree=1,skip_nodes={},preserve_skip_node_edges=True,extra=None):

		deg = nx.degree(g)
		if preserve_skip_node_edges:
			new_skip_nodes = set([])
			for node,d in list(g.nodes(data=True)):
				if node in skip_nodes:
					for org, edge in list(g.edges(node)):
						new_skip_nodes.add(org)
						new_skip_nodes.add(edge)
				if extra is not None and extra in d and d[extra] in skip_nodes:
					for org, edge in list(g.edges(node)):
						new_skip_nodes.add(org)
						new_skip_nodes.add(edge)
			skip_nodes = new_skip_nodes

		for node,d in list(g.nodes(data=True)):
			if deg[node] < degree:
				if node in skip_nodes:
					pass
				elif extra is not None and extra in d and d[extra] in skip_nodes:
					pass
				else:
					g.remove_node(node)
		return g

	@classmethod
	def filter_by_kcore(cls,g,kcore):
		print ("Shrinking down to k-core.")
		g.remove_edges_from(nx.selfloop_edges(g))
		g = nx.k_core(g,k=kcore)
		return g

	@classmethod
	def add_edge_percentage(cls,g,att_name,att_val,exclude_att_vals=None):

		new_att_name = str(att_val)+"_%"
		nx.set_node_attributes(g, 0.0, new_att_name)
		nodes_to_exclude = set({})
		nodes_of_interest = set({node for node, ndat in g.nodes(data=True) if att_name in ndat and ndat[att_name]==att_val})
		if exclude_att_vals is not None:
			for _val in exclude_att_vals:
				nodes_to_exclude.update(set({node for node, ndat in g.nodes(data=True) if att_name in ndat and ndat[att_name]==_val}))
		for node in list(g.nodes()):
			edge_weight_sum = 0.0
			noi_weight_sum = 0.0
			for org, edge, edat in g.edges(node,data=True):
				if edge in nodes_of_interest:
					noi_weight_sum+=float(edat["weight"])
				if node not in nodes_to_exclude and g.nodes[edge][att_name] is not None and g.nodes[edge][att_name] != "None" and len(str(g.nodes[edge][att_name])) > 0:
					edge_weight_sum+=float(edat["weight"])
			if edge_weight_sum > 0:
				g.nodes[node][new_att_name]=noi_weight_sum/edge_weight_sum
			else:
				g.nodes[node][new_att_name]=np.nan
		return g

	@classmethod
	def normalize_edge_percentages(cls,g,att_name,att_vals=None):

		#print ("Normalizing network attribute percentages")
		if att_vals is None:
			att_vals = set([str(ndat[att_name]) for n,ndat in g.nodes(data=True) if ndat[att_name] is not None and len(str(ndat[att_name])) > 0 and str(ndat[att_name]) != "None"])
		for node, ndat in list(g.nodes(data=True)):
			att_sum = 0.0
			for att_val in att_vals:
				new_att_name = str(att_val)+"_%"
				att_sum += float(ndat[new_att_name])
			if att_sum > 0.0:
				for att_val in att_vals:
					new_att_name = str(att_val)+"_%"
					g.nodes[node][new_att_name]=ndat[new_att_name]/att_sum

		return g

	@classmethod
	def add_edge_percentage_cont(cls,g,att_name,exclude_att_vals=None,max_steps=10,normalize=True,max_iter=5):

		att_vals = set([str(ndat[att_name]) for n,ndat in g.nodes(data=True) if ndat[att_name] is not None and len(str(ndat[att_name])) > 0 and str(ndat[att_name]) != "None"])
		org_nodes = set([])
		for att_val in att_vals:
			new_att_name = str(att_val)+"_%"
			nx.set_node_attributes(g, -1, new_att_name)
			for node, ndat in list(g.nodes(data=True)):
				if att_name in ndat and ndat[att_name]==att_val:
					g.nodes[node][new_att_name]=1.0
					org_nodes.add(node)
		iter = 0
		first_iter = True
		while iter < max_iter:
			all_nodes = set([])
			if not first_iter:
				max_steps=1
				if normalize:
					g = cls.normalize_edge_percentages(g,att_name,att_vals=att_vals)
			for r in range(max_steps):
				nodes_affected = 0
				att_val_dif = 0.0
				for node, ndat in list(g.nodes(data=True)):
					has_touched=False
					if node not in all_nodes:
						if node not in org_nodes:
							for att_val in att_vals:
								prev_att_val = g.nodes[node][new_att_name]
								new_att_name = str(att_val)+"_%"
								edge_weight_sum = 0.0
								noi_weight_sum = 0.0
								for org, edge, edat in list(g.edges(node,data=True)):
									if g.nodes[edge][new_att_name] > 0.0:
										has_touched=True
										noi_weight_sum+=float(edat["weight"])*g.nodes[edge][new_att_name]
									edge_weight_sum+=float(edat["weight"])
								if has_touched:
									g.nodes[node][new_att_name]=noi_weight_sum/edge_weight_sum
									all_nodes.add(node)
									nodes_affected+=1
									att_val_dif+=abs(g.nodes[node][new_att_name]-prev_att_val)
				#print (nodes_affected)
				#print (att_val_dif)
			if first_iter:
				for node, ndat in list(g.nodes(data=True)):
					for att_val in att_vals:
						new_att_name = str(att_val)+"_%"
						if ndat[new_att_name] < 0.0:
							g.nodes[node][new_att_name]=0.0
						if new_att_name not in dict(ndat):
							g.nodes[node][new_att_name]=0.0
			iter+=1
			first_iter=False
		if normalize:
			g = cls.normalize_edge_percentages(g,att_name,att_vals=att_vals)
		return g

	@classmethod
	def label_cont_attributes(cls,g,att_name):

		att_vals = set([str(ndat[att_name]) for n,ndat in g.nodes(data=True) if ndat[att_name] is not None and len(str(ndat[att_name])) > 0 and str(ndat[att_name]) != "None"])
		for node,ndat in list(g.nodes(data=True)):
			vals = {att:float(val) for att,val in dict(ndat).items() if att.replace("_%","") in att_vals}
			if np.sum(np.array(list(vals.values()))) > 0:
				g.nodes[node][att_name]=sorted(vals.items(), key=itemgetter(1), reverse=True)[0][0].replace("_%","")

		return g

	@classmethod
	def label_cont_range_attributes(cls,g,att_vals,lim=0.8):

		new_att_name = "-".join([a.replace("_%","") for a in att_vals])+"_BRIDGE"
		nx.set_node_attributes(g, False, new_att_name)
		lim = float(lim/float(len(att_vals)))
		for node,ndat in list(g.nodes(data=True)):
			cond_met = True
			for att_val in att_vals:
				if "%" not in str(att_val): att_val = str(att_val)+"_%"
				if ndat[att_val] < lim:
					cond_met = False
			g.nodes[node][new_att_name]=cond_met

		return g

	@classmethod
	def get_homophily_score(cls,g2,att_name,att_vals,only_node_types=set([]),weighted=True):

		n_edges = 0
		h_score = 0.0
		if weighted:
			base_val = int(g2.size()*0.01)
		else:
			base_val = int(g2.number_of_edges()*0.01)
		same_edges = dict({a:base_val for a in att_vals})
		all_edges = dict({a:base_val for a in att_vals})
		n_edges = base_val*len(att_vals)
		g = g2.copy()
		for u,v,d in g.edges(data=True):
			u_val = g.nodes[u][att_name]
			v_val = g.nodes[v][att_name]
			if weighted:
				edge_val = np.log(d["weight"]+1)
			else:
				edge_val = 1
			if u_val in att_vals and v_val in att_vals:
				if u_val == v_val:
					same_edges[v_val]+=edge_val
					all_edges[v_val]+=edge_val
				else:
					all_edges[u_val]+=edge_val
					all_edges[v_val]+=edge_val
				n_edges+=edge_val

		if n_edges > 0:
			qual_term = np.sum(np.array([(v/n_edges) for k,v in sorted(same_edges.items())]))
			end_term = np.sum(np.array([(v/n_edges)**2 for k,v in sorted(all_edges.items())]))
			h_score = (qual_term - end_term)/(1.0-end_term)
		if h_score > 1 or np.isnan(h_score):
			h_score = 1.0
		if h_score < -1:
			h_score = -1.0
		return h_score


	@classmethod
	def get_homophily_score_soft(cls,g,att_name,att_vals,only_node_types=set([])):

		n_edges = 0
		same_edges = dict({a:0.0 for a in att_vals})
		all_edges = dict({a:0.0 for a in att_vals})
		for ego_node, ndat in list(g.nodes(data=True)):
			if len(only_node_types) > 0 and ndat["node_type"] not in only_node_types:
				continue
			for ego_org, edge, edat in list(g.edges(ego_node,data=True)):
				edge_att_val = g.nodes[edge][att_name]
				if edge_att_val in att_vals:
					if ndat[att_name] == edge_att_val:
						same_edges[edge_att_val]+=float(edat["weight"])*g.nodes[edge][edge_att_val+"_%"]
					all_edges[edge_att_val]+=float(edat["weight"])
					n_edges+=float(edat["weight"])
		end_term = np.sum(np.array([(v/n_edges)**2 for k,v in sorted(all_edges.items())]))
		qual_term = np.sum(np.array([(v/n_edges) for k,v in sorted(same_edges.items())]))
		h_score = (qual_term - end_term)/(1.0-end_term)
		return h_score

	@classmethod
	def add_homophily_score(cls,g,att_name,score_name="homophily",only_nodes=set([]),only_node_types=set([])):

		att_vals = set([str(ndat[att_name]) for n,ndat in g.nodes(data=True) if ndat[att_name] is not None and len(str(ndat[att_name])) > 0 and str(ndat[att_name]) != "None"])
		nx.set_node_attributes(g, 0.0, score_name)
		n_nodes = g.number_of_nodes()
		node_count = 0
		for node, node_data in list(g.nodes(data=True)):
			node_count += 1
			if len(only_node_types) > 0 and node_data["node_type"] not in only_node_types:
				continue
			if len(only_nodes) > 0 and node not in only_nodes:
				continue
			ego = nx.generators.ego.ego_graph(g,node,radius=2)
			h_score = cls.get_homophily_score(ego,att_name,att_vals)
			g.nodes[node][score_name]=h_score

			if node_count % 1000 == 0:
				pass
				#print (score_name + " -> " + str(node_count)+" out of "+str(n_nodes))

		return g

	@classmethod
	def add_affinity_scores(cls,g,actor_mapping,num_cores=12):

		n_actors = len(actor_mapping)
		not_mapped_actors = [a for a in g.nodes() if a not in actor_mapping]
		actor_mapping.update( {a:"grey" for a in random.sample(not_mapped_actors,n_actors)} )
		unique_atts = set([v for k,v in actor_mapping.items()])
		node_atts = {n:{ua:0.0 for ua in unique_atts} for n in g.nodes()}
		diam = float(nx.diameter(g))-1.0
		for n in list(g.nodes()):
			if n in actor_mapping:
				node_atts[n][actor_mapping[n]]=1.0
				node_atts["dist_to_0"]=0.0
			else:
				node_atts["dist_to_0"]=diam
