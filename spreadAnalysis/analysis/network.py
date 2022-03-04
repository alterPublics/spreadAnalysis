import sys
import time
import random
import networkx as nx
import community
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from spreadAnalysis.utils.network_utils import NetworkUtils
from spreadAnalysis.utils.link_utils import LinkCleaner
import re

class Net(NetworkUtils):

	def __init__(self,
				title,
				data):

		self.title = title
		self.data = data

		self.g = None

		self.export_path = "/Users/jakobbk/Documents/postdoc/mainProject/exploration/sweden/data_analysis"

	def load_net(self):

		self.g = nx.read_gexf(self.export_path+"/"+self.title+".gexf")

	def write_net(self):

		#self.recursive_data_check()
		nx.write_gexf(self.g, self.export_path+"/"+self.title+".gexf")

	def recursive_data_check(self):
		cols = {}
		for n,dat in self.g.nodes(data=True):
			cols.update(dict(dat))
		for n,dat in self.g.nodes(data=True):
			#if len(dat) < len(cols):
			for col in list(cols.keys()):
				if col not in dat:
					if isinstance(cols[col],str):
						self.g.nodes[n][col]=""
					elif isinstance(cols[col],float):
						self.g.nodes[n][col]=0.0
				elif dat[col] is None:
					if isinstance(cols[col],str):
						self.g.nodes[n][col]=""
					elif isinstance(cols[col],float):
						self.g.nodes[n][col]=0.0
					else:
						self.g.nodes[n][col]=False

	def add_data_to_graph(self,var_name="random_var",type="str",datadict={}):
		if type == "str" and var_name not in self.g.nodes[list(self.g.nodes())[0]]:
			nx.set_node_attributes(self.g, "", var_name)
		if type == "float" and var_name not in self.g.nodes[list(self.g.nodes())[0]]:
			nx.set_node_attributes(self.g, 0.0, var_name)
		if type == "boolean" and var_name not in self.g.nodes[list(self.g.nodes())[0]]:
			nx.set_node_attributes(self.g, False, var_name)

		banned_nodes = set(["mortenoesterlun twitter","q8een_ twitter","e2d3org twitter",
		"AlekoNedjalkow twitter"])

		pat = re.compile("[åäöÅÄÖüåæøÅÆØA-Za-z0-9_\s]+")
		if len(datadict) > 0:
			for n in list(self.g.nodes()):
				if n in datadict:
					if isinstance(datadict[n],str):
						if var_name == "Label":
							_val = pat.match(str(datadict[n]))
							if _val is not None and n not in banned_nodes:
								_val = _val.group()
							else:
								_val = str(n)
						else:
							_val = str(datadict[n])
						#_val = str(datadict[n].encode('ascii', errors='ignore').decode())
						#_val = re.sub('[^!-~]+',' ',str(datadict[n])).strip()
					else:
						_val = datadict[n]
					self.g.nodes[n][var_name]=_val

	def net_to_df(self):

		self.recursive_data_check()
		new_data = []
		for node,dat in self.g.nodes(data=True):
			data_row = dict(dat)
			cols = list(sorted(data_row.keys()))
			new_row = list([v for k,v in sorted(data_row.items())])
			new_data.append(new_row)
		df = pd.DataFrame(new_data,columns=cols)
		return df

	def net_to_dict(self):

		self.recursive_data_check()
		new_data = defaultdict(dict)
		for node,dat in self.g.nodes(data=True):
			data_row = dict(dat)
			new_data[node]=data_row
		return new_data

	def add_node_and_edges(self,node0,node1,node_type0="node",node_type1="node",weight=1,node0_extra=None):

		if node0 not in self.g:
			if node0_extra is not None:
				self.g.add_node(node0,node_type=node_type0,extra=node0_extra)
			else:
				self.g.add_node(node0,node_type=node_type0)
		if node1 not in self.g: self.g.add_node(node1,node_type=node_type1)
		if self.g.has_edge(node0,node1):
			self.g.get_edge_data(node0,node1)['weight'] += weight
		else:
			self.g.add_edge(node0,node1,weight=weight,)

class BipartiteNet(Net):

	def __init__(self,
				title,
				data,
				links_to_own_website=None):
		Net.__init__(self,title,data)

		self.links_to_own_website = links_to_own_website
		self.title = self.title + "_BI"
		self.g = nx.Graph()

	def create_net(self,node0_col,node1_col,skip_domains=False):

		self.g = nx.Graph()
		for i,row in self.data.iterrows():
			if skip_domains and LinkCleaner().is_url_domain(row[node1_col]):
				continue
			if row[node0_col] is not None:
				node0 = row[node0_col]
				node1 = row[node1_col]
				self.add_node_and_edges(node0,node1,node_type0=node0_col,node_type1=node1_col)

		if self.links_to_own_website is not None:
			for actor_username, links in self.links_to_own_website.items():
				for link in links:
					node0 = actor_username
					node1 = link
					if skip_domains and LinkCleaner().is_url_domain(node1):
						continue
					self.add_node_and_edges(node0,node1)

class UnipartiteNet(Net):

	def __init__(self,
				title,
				data,
				links_to_own_website=None):
		Net.__init__(self,title,data)

		self.links_to_own_website = links_to_own_website
		self.title = self.title + "_UNI"

	def create_net(self,node_col,mutual_col,skip_domains=True,max_constrain=5000,skip_nodes=[]):

		self.g = nx.Graph()
		unique_mutuals = len(set(list(self.data[mutual_col])))
		mutual_count = 0
		for mutual,nodes in self.data[[node_col,mutual_col]].groupby(mutual_col):
			mutual_count+=1
			if mutual is not None:
				if skip_domains and LinkCleaner().is_url_domain(mutual):
					continue
				all_nodes = list(nodes[node_col])
				if len(skip_nodes) > 0:
					all_nodes = [n for n in all_nodes if n not in skip_nodes]
				if max_constrain:
					random.shuffle(all_nodes)
					all_nodes = all_nodes[:max_constrain]
				#print (mutual)
				#print (len(all_nodes))
				for node0 in all_nodes:
					for node1 in all_nodes:
						if node0 is not None and node1 is not None and node0 != node1:
							self.add_node_and_edges(node0,node1)
			if mutual_count % 100 == 0:
				print ("Building unipartite: {0} of out {1}".format(mutual_count,unique_mutuals))

		if self.links_to_own_website is not None:
			for actor_username, links in self.links_to_own_website.items():
				for link in links:
					node0 = actor_username
					node1 = link
					self.add_node_and_edges(node0,node1)
