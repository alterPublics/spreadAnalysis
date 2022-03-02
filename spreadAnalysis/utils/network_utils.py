import community
import networkx as nx
import math
import numpy as np
from operator import itemgetter
from multiprocessing import Pool, Manager
import random

def set_neighbour_affinity(args):

	new_node_atts = {}
	node_edges = args[0]
	node_atts = args[1]
	unique_atts = args[2]
	for edges in node_edges:
		n_edges = 0
		new_atts = {a:0.0 for a in unique_atts}
		avg_dists = []
		for org,edg,w in edges:
			avg_dists.append(node_atts[edg]["dist_to_0"]+1.0)
			for att in unique_atts:
				new_atts[att]+=np.log(w+2)*node_atts[edg][att]*(node_atts[edg]["dist_to_0"]+1)**-1
		atts_sum = float(np.sum(np.array(list(new_atts.values()))))
		if atts_sum > 0:
			new_atts = {a:float(v)/atts_sum for a,v in list(new_atts.items())}
		new_atts["dist_to_0"]=float(np.mean(np.array(avg_dists)))
	return new_node_atts

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
		partition = community.best_partition(g.to_undirected())
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
	def filter_by_degrees(cls,g,degree=1,skip_nodes={},preserve_skip_node_edges=True):

		deg = nx.degree(g)
		if preserve_skip_node_edges:
			new_skip_nodes = set([])
			for node in list(g.nodes()):
				if node in skip_nodes:
					for org, edge in list(g.edges(node)):
						new_skip_nodes.add(org)
						new_skip_nodes.add(edge)
			skip_nodes = new_skip_nodes

		for node in list(g.nodes()):
			if deg[node] < degree:
				if node in skip_nodes:
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
