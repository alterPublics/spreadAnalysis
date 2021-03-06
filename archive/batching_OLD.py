from spreadAnalysis.persistence.mongo import MongoSpread
from spreadAnalysis.analysis.network import BipartiteNet
from spreadAnalysis.utils import helpers as hlp
from spreadAnalysis.persistence.schemas import Spread
from spreadAnalysis.utils.link_utils import LinkCleaner
import pandas as pd
import time
import random
import numpy as np
import sys
import gc
from multiprocessing import Pool, Manager
import networkx as nx

def bi_to_uni(data):

	rep_data = {}
	for k,v in data.items():
		for n1 in v:
			for n2 in v:
				if n1[0] != n2[0]:
					if (n1[0],n2[0]) in rep_data:
						rep_data[(n1[0],n2[0])]+=(n1[1]+n2[1])/2
					elif (n2[0],n1[0]) in rep_data:
						rep_data[(n2[0],n1[0])]+=(n1[1]+n2[1])/2
					else:
						rep_data[(n1[0],n2[0])]=0
	return rep_data

	"""
		net_cols = ["url","actor","weight"]
		data = pd.DataFrame(list([vv for v in data.values() for vv in v]),columns=net_cols)
		G_affil=nx.from_pandas_edgelist(data, 'actor', 'url', edge_attr='weight')
		actors = data.drop_duplicates(subset=['actor'])
		actors = actors['actor']
		g = nx.bipartite.projected_graph(G_affil, actors, multigraph=False)
	"""

def filter_docs(args):

	docs = args[0]
	degree_type = args[1]
	between_dates = args[2]
	new_docs = []
	if between_dates is None:
		for doc in docs:
			doc["message_ids"]=len(doc["message_ids"])
			new_docs.append(doc)
	else:
		mdb = MongoSpread()
		for doc in docs:
			if doc[degree_type] is not None and doc[degree_type] != "None":
				tmp_mids = []
				for mid in doc["message_ids"]:
					if between_dates is not None:
						post = mdb.database["post"].find_one({"message_id":mid})
						post_date = Spread._get_date(data=post,method=post["method"])
						if post_date is not None:
							if hlp.date_is_between_dates(post_date,between_dates["start_date"],between_dates["end_date"]):
								tmp_mids.append(mid)
			doc["message_ids"]=len(tmp_mids)
			new_docs.append(doc)
		mdb.close()
	return new_docs

def normalize_nb(args):

	nb_vals = args[0]
	entity_type = args[1]
	cat = args[2]
	distance = args[3]
	write_docs = []
	for nb,vals in nb_vals.items():
		edge_sum = float(np.sum(np.array(list(vals.values()))))
		tmp_ed = {k:float(v)/edge_sum for k,v in vals.items()}
		new_doc = {"uentity":nb,"entity_type":entity_type,
					cat:dict(tmp_ed),"is_fixed":False,"category":cat,
					"n_degrees":edge_sum}
		if distance is not None:
			new_doc["distance_to_0"]=distance
		write_docs.append(new_doc)
	return write_docs

def assign_values_to_neighbours(args):

	mdb = MongoSpread()
	entities = args
	only_platforms = ["facebook","twitter","vkontakte","reddit","youtube",
				"telegram","tiktok","gab","instagram","web"]
	entity_type_conv = {"actor":"url","url":"actor"}
	nb_vals = {}
	for ent in entities:
		n_data = mdb.database["url_bi_network"].find({ent["entity_type"]:ent["uentity"],"platform":{"$in":only_platforms}})
		for edge in n_data:
			nb = edge[entity_type_conv[ent["entity_type"]]]
			if nb not in nb_vals: nb_vals[nb]=dict({k:0.0 for k in dict(ent[ent["category"]]).keys()})
			for k,v in dict(ent[ent["category"]]).items():
				nb_vals[nb][k]+=v*np.log(len(edge["message_ids"])+2)

	mdb.close()
	return nb_vals

def assign_neighbour_values_to_self(args):

	mdb = MongoSpread()
	entities = args[0]
	entity_type = args[1]
	cat = args[2]
	only_platforms = ["facebook","twitter","vkontakte","reddit","youtube",
				"telegram","tiktok","gab","instagram","web"]
	entity_type_conv = {"actor":"url","url":"actor"}
	inserts = []
	updates = []
	for ent in entities:
		n_degrees = 0
		ent_vals = {}
		seen_distances = set([])
		ent_coded = mdb.database["url_bi_network_coded"].find_one({ "uentity": ent})
		if ent_coded is not None and ent_coded["is_fixed"]: continue
		n_data = mdb.database["url_bi_network"].aggregate([{ "$match": { entity_type: ent,"platform":{"$in":only_platforms}}  },{"$lookup":{"from":"url_bi_network_coded","localField": entity_type_conv[entity_type],"foreignField": "uentity","as":"entity_data"}}])
		for edge in n_data:
			n_degrees += 1
			if "entity_data" in edge:
				for scheme in edge["entity_data"]:
					if cat in scheme:
						_vals = dict(scheme[cat])
						if len(ent_vals) < 1:
							ent_vals=dict({k:0.0 for k in _vals.keys()})
						seen_distances.add(int(scheme["distance_to_0"]))
						for k,v in _vals.items():
							ent_vals[k]+=v*np.log(len(edge["message_ids"])+2)*(np.log(scheme["distance_to_0"]+2)**-1)
						break
		if len(seen_distances) > 0:
			ent_sum = float(np.sum(np.array(list(ent_vals.values()))))
			ent_vals = {k:float(v)/ent_sum for k,v in ent_vals.items()}
			new_doc = {"uentity":ent,"entity_type":entity_type,
						cat:ent_vals,"is_fixed":False,"category":cat,
						"n_degrees":n_degrees,"distance_to_0":min(seen_distances)+1}
			if ent_coded is not None:
				updates.append(new_doc)
			else:
				inserts.append(new_doc)

	return inserts,updates

def query_multi(args):

	db = args[0]
	queries = args[1]
	mdb = MongoSpread()
	docs = []
	for query in queries:
		if db == "url_bi_network":
			docs.extend(list(mdb.database[db].find(query,{"url":1,"actor":1,"message_ids":1})))
		else:
			docs.extend(list(mdb.database[db].find(query)))
	mdb.close()
	return docs

def update_tmp_degree_data(degree_data,net_doc,degree_type,edge_type):

	if net_doc[degree_type] not in degree_data:
		degree_data[net_doc[degree_type]]=[]
	degree_data[net_doc[degree_type]].append(net_doc[edge_type])

	return degree_data

def find_actors(selection={},actors=[]):

	if len(selection) == 0 and len(actors) == 0:
		return []
	else:
		cur = MongoSpread().database["actor"].find(selection)
		if len(actors) > 0:
			actors = [d["Actor"] for d in cur if d["Actor"] in set(actors)]
		else:
			actors = [d["Actor"] for d in cur]
		return actors

def find_urls(selection={},urls=[]):

	mdb = MongoSpread()
	final_urls = []
	if len(selection) == 0 and len(urls) == 0:
		return final_urls
	elif len(selection) == 0:
		org_urls = list(urls)
	else:
		cur = mdb.database["url"].find(selection)
		if len(urls):
			org_urls = [d["Url"] for d in cur if d["Url"] in set(urls)]
		else:
			org_urls = [d["Url"] for d in cur]
	url_post_db = mdb.database["url_post"]
	post_db = mdb.database["post"]
	for org_url in org_urls:
		cur = url_post_db.find({"input":str(org_url)})
		next_url_post = True
		while next_url_post is not None:
			next_url_post = next(cur, None)
			if next_url_post is not None:
				post = post_db.find_one({"message_id":next_url_post["message_id"]})
				url = Spread._get_message_link(data=post,method=post["method"])
				url = LinkCleaner().single_clean_url(url)
				url = LinkCleaner().sanitize_url_prefix(url)
				final_urls.append(url)
	#final_urls = {u:[] for u in final_urls}
	return set(final_urls)

def get_actor_query_overlap(strs=[],message_ids=[]):

	mdb = MongoSpread()
	if len(message_ids) > 0:
		for mid in message_ids:
			post = post_db.find_one({"message_id":next_url_post["message_id"]})

def iteration_test():

	mdb = MongoSpread()
	cur = mdb.database["post"].find()
	next_url_post = True
	count = 0
	start_time = time.time()
	while next_url_post is not None:
		count += 1
		next_url_post = next(cur, None)
		if count % 100000 == 0:
			print("--- {0} seconds --- total for {1}".format((time.time() - start_time),str(count)))

def create_bi_ego_graph(selection_types=["actor"],actor_selection={},url_selection={},actors=[],urls=[],between_dates=None,only_platforms=[],title="test"):

	def add_data_to_net(docs,binet,has_been_queried,org_type):

		for doc in docs:
			if doc[org_type] not in has_been_queried:
				if doc["message_ids"] > 0:
					if doc["actor"] is not None and doc["url"] is not None:
						binet.add_node_and_edges(doc["actor"],doc["url"],node_type0="actor",node_type1="url",weight=doc["message_ids"])
		return binet

	if len(only_platforms) < 1:
		only_platforms = ["facebook","twitter","web","vkontakte","reddit","youtube",
					"telegram","tiktok","gab","instagram"]

	mdb = MongoSpread()
	net_db = mdb.database["url_bi_network"]
	num_cores = 12
	pool = Pool(num_cores)
	has_been_queried = set([])
	binet = BipartiteNet(title,{})
	first_degree_urls = set([])
	print ("finding core nodes.")
	if "actor" in selection_types:
		print ("actors...")
		actors = find_actors(selection=actor_selection,actors=actors)
		aliases = []
		temp_docs = []
		for actor in actors:
			temp_docs.extend(list(mdb.database["url_bi_network"].find({"actor":actor,"platform":{"$in":only_platforms}})))
			aliases.extend(list(mdb.database["alias"].find({"actor":actor})))
			has_been_queried.add(actor)
		results = pool.map(filter_docs,[(l,"url",between_dates) for l in hlp.chunks(temp_docs,int(len(temp_docs)/num_cores)+1)])
		for result in results:
			binet = add_data_to_net(result,binet,{},"actor")
		alias_queries = []
		for alias_doc in aliases:
			alias_queries.append({"actor":alias_doc["alias"],"platform":{"$in":only_platforms}})
			has_been_queried.add(alias_doc["alias"])
		results = pool.map(query_multi,[("url_bi_network",l) for l in hlp.chunks(alias_queries,int(len(alias_queries)/num_cores)+1)])
		for result in results:
			for fdocs in pool.map(filter_docs,[(l,"url",between_dates) for l in hlp.chunks(list(result),int(len(list(result))/num_cores)+1)]):
				binet = add_data_to_net(fdocs,binet,{},"actor")
	if "url" in selection_types:
		print ("urls...")
		urls = find_urls(selection=url_selection,urls=urls)
		first_degree_urls.update(urls)
	first_degree_urls.update(set([n for n,d in binet.g.nodes(data=True) if d["node_type"]=="url"]))
	del actors
	del urls
	del results
	gc.collect()

	print ("building first degree connections.")
	fucount = 0
	first_degree_queries = []
	has_been_queried_first_degree = set([])
	for furl in first_degree_urls:
		fucount+=1
		if furl is not None and furl != "None":
			first_degree_queries.append({"url":furl,"platform":{"$in":only_platforms}})
			has_been_queried_first_degree.add(furl)
		if fucount % 10000 == 0: print ("{0} out of {1}".format(fucount,str(len(first_degree_urls))))
	results = pool.map(query_multi,[("url_bi_network",l) for l in hlp.chunks(first_degree_queries,int(len(first_degree_queries)/num_cores)+1)])
	for result in results:
		for fdocs in pool.map(filter_docs,[(l,"actor",between_dates) for l in hlp.chunks(list(result),int(len(list(result))/num_cores)+1)]):
			binet = add_data_to_net(fdocs,binet,has_been_queried,"actor")

	del results
	gc.collect()
	print ("Nodes in net before shave {0} for second degree actors".format(len(list(binet.g.nodes()))))
	binet.g = binet.filter_by_degrees(binet.g,degree=2,skip_nodes=has_been_queried,preserve_skip_node_edges=False)
	print ("Nodes in net after shave {0} for second degree actors".format(len(list(binet.g.nodes()))))
	second_degree_actors = set([n for n,d in binet.g.nodes(data=True) if d["node_type"]=="actor" and n not in has_been_queried])
	print (len(second_degree_actors))

	print ("searching for second degree interconnections.")
	fucount = 0
	for _factor_chunk in hlp.chunks(list(second_degree_actors),int(len(list(second_degree_actors))/num_cores)+1):
		second_degree_queries = []
		for _factor in _factor_chunk:
			fucount+=1
			second_degree_queries.append({"actor":_factor,"platform":{"$in":only_platforms}})
		results = pool.map(query_multi,[("url_bi_network",l) for l in hlp.chunks(second_degree_queries,int(len(second_degree_queries)/num_cores)+1)])
		for result in results:
			for fdocs in pool.map(filter_docs,[(l,"url",between_dates) for l in hlp.chunks(list(result),int(len(list(result))/num_cores)+1)]):
				binet = add_data_to_net(fdocs,binet,has_been_queried_first_degree,"url")
		print (fucount)
	del results
	gc.collect()
	print ("Nodes in net before shave {0} for second degree urls".format(len(list(binet.g.nodes()))))
	binet.g = binet.filter_by_degrees(binet.g,degree=2,skip_nodes=has_been_queried,preserve_skip_node_edges=False)
	print ("Nodes in net before shave {0} for second degree urls".format(len(list(binet.g.nodes()))))
	print ("done")

	return binet

def populate_net_with_scheme(scheme_title,incl_cats=["main_category"]):

	def set_code_blueprints(entities,codes,incl_cats):

		new_docs = []
		ce = entities
		for cat in incl_cats:
			for e in list(ce.keys()):
				tmp_ed = {c:0.0 for c in codes[cat]}
				if ce[e][cat] in tmp_ed:
					tmp_ed[ce[e][cat]]=1.0
				new_doc = {"uentity":e,"entity_type":ce[e]["entity_type"],
							cat:dict(tmp_ed),"is_fixed":True,"category":cat,
							"distance_to_0":0}
				new_docs.append(new_doc)
		return new_docs

	print ("Getting coded entities.")
	mdb = MongoSpread()
	coding_db = mdb.database["coding"]
	codes = {}
	for cat in incl_cats:
		codes.update({cat:set([d[cat] for d in coding_db.find({"scheme_title":scheme_title})])})
	coded_actors = {d["uentity"]:d for d in coding_db.find({"scheme_title":scheme_title,
					"entity_type":"actor"})}
	coded_urls = {}
	for d in coding_db.find({"entity_type":"url","scheme_title":scheme_title}):
		net_urls = find_urls(urls=[d["uentity"]])
		for nurl in list(net_urls):
			coded_urls[nurl]=d

	print ("Setting blueprints for coded entities.")
	coded_actors = set_code_blueprints(coded_actors,codes,incl_cats)
	coded_urls = set_code_blueprints(coded_urls,codes,incl_cats)
	all_fixed = set([d["uentity"] for d in coded_actors])
	all_fixed.update(set([d["uentity"] for d in coded_urls]))

	print ("Writing to database.")
	mdb.write_many(mdb.database["url_bi_network_coded"],coded_actors,("uentity","category"))
	mdb.write_many(mdb.database["url_bi_network_coded"],coded_urls,("uentity","category"))

def propagate_scheme_values(cat,max_steps=20,low_memory=False,start_step=0):

	def tabulate_nb_results(results,nb_vals={},skip_nodes={}):

		for result in results:
			for nb,vals in result.items():
				if nb in skip_nodes: continue
				if nb not in nb_vals:
					nb_vals[nb]=vals
				else:
					for k,v in vals.items():
						nb_vals[nb][k]+=v
		return nb_vals

	mdb = MongoSpread()
	if not low_memory:
		all_fixed_nodes = set([d["uentity"] for d in mdb.database["url_bi_network_coded"].find({"distance_to_0":{ "$lte": start_step },"category":cat},{"uentity":1})])
		mdb.close()
	for distance in range(start_step,max_steps):
		print (len(all_fixed_nodes))
		for entity_type in ["actor","url"]:
			batch_size = 20000
			if entity_type == "url": batch_size = batch_size*4
			ent_dis_count = 0
			nb_vals = {}
			mu_nodes = {}
			next_d = True
			mdb = MongoSpread()
			est_doc_count = mdb.database["url_bi_network_coded"].count_documents({"distance_to_0":distance,"category":cat,"entity_type":entity_type})
			cur = mdb.database["url_bi_network_coded"].aggregate([{"$match":{"distance_to_0":distance,"category":cat,"entity_type":entity_type}},{ "$sample" : { "size": est_doc_count } }],allowDiskUse=True)
			while next_d is not None:
				ent_dis_count+=1
				next_d = next(cur, None)
				if next_d is not None:
					if not low_memory and next_d["uentity"] not in all_fixed_nodes:
						mu_nodes[next_d["uentity"]]=next_d
						all_fixed_nodes.add(next_d["uentity"])
					else:
						mu_nodes[next_d["uentity"]]=next_d

				if len(mu_nodes) >= batch_size or next_d is None:
					print ("Processing batch at {}".format(ent_dis_count))
					entity_type_conv = {"actor":"url","url":"actor"}
					num_cores = 8
					pool = Pool(num_cores)
					mu_nodes = [d for k,d in mu_nodes.items() if d["entity_type"] == entity_type]
					if len(mu_nodes) > 0:
						mu_nodes = list(hlp.chunks(mu_nodes,int(len(mu_nodes)/num_cores)+1))
						print ("initializing values for {0} at distance {1}".format(entity_type,distance))
						results = pool.map(assign_values_to_neighbours,mu_nodes)
						print ("tabulating values for {0} at distance {1}".format(entity_type,distance))
						nb_vals = tabulate_nb_results(results,nb_vals=nb_vals,skip_nodes=all_fixed_nodes)
						del results
						gc.collect()
					mu_nodes = {}
			if len(nb_vals) > 0:
				mdb.close()
				print ("normalizing values for {0} at distance {1}".format(entity_type,distance))
				all_fixed_nodes.update(set(nb_vals.keys()))
				nb_vals = list(hlp.chunks_dict(nb_vals,int(len(nb_vals)/num_cores)+1))
				nb_vals = [(l,entity_type_conv[entity_type],cat,distance+1) for l in nb_vals]
				pool = Pool(int(num_cores/2))
				all_write_docs = pool.map(normalize_nb,nb_vals)
				del nb_vals
				del mu_nodes
				gc.collect()
				mdb = MongoSpread()
				print ("writing to database for {0} at distance {1}".format(entity_type,distance))
				for write_docs in all_write_docs:
					if low_memory:
						mdb.write_many(mdb.database["url_bi_network_coded"],
												write_docs,("uentity","category"),only_insert=True)
					else:
						mdb.insert_many(mdb.database["url_bi_network_coded"],write_docs)
				mdb.close()
				del all_write_docs
				gc.collect()

def stochastic_update_scheme_values(cat,batch_size=None):

	mdb = MongoSpread()
	num_cores = 8
	inner_batch_size = 5000*num_cores
	pool = Pool(num_cores)
	full_count = 0
	#print (assign_neighbour_values_to_self((["MajorpatriotQ45"],"actor","main_category")))
	for entity_type in ["url","actor"]:
		ents = set([])
		if batch_size is None:
			real_batch_size = mdb.database["url_bi_network"].aggregate([{"$group": {"_id": "${0}".format(entity_type),"count": { "$sum": 1 }}},{"$group": {"_id": "${0}".format(entity_type),"totalCount": { "$sum": "$count" },"distinctCount": { "$sum": 1 }}}],allowDiskUse=True)
			real_batch_size = list(real_batch_size)[0]["distinctCount"]
		next_d = True
		cur = mdb.database["url_bi_network"].aggregate([{"$group": {"_id": "${0}".format(entity_type)}},{ "$sample" : { "size": real_batch_size } }],allowDiskUse=True)
		while next_d is not None:
			full_count+=1
			next_d = next(cur, None)
			if next_d is not None:
				ents.add(next_d["_id"])
			if len(ents) >= inner_batch_size or next_d is None:
				if len(ents) > 0:
					chunked_ents = [(l,entity_type,cat) for l in list(hlp.chunks(list(ents),int(len(list(ents))/num_cores)+1))]
					results = pool.map(assign_neighbour_values_to_self,chunked_ents)
					for result in results:
						inserts = result[0]
						updates = result[1]
						mdb.insert_many(mdb.database["url_bi_network_coded"],inserts)
						mdb.update_many(mdb.database["url_bi_network_coded"],updates,"uentity")
				ents = set([])
				print (full_count)

	mdb.close()

def bi_to_uni_net(data,node0="actor",node1="url"):

	def add_node_and_edges(g,node0,node1,weight):

		if node0 not in g: g.add_node(node0)
		if node1 not in g: g.add_node(node1)
		if g.has_edge(node0,node1):
			g.get_edge_data(node0,node1)['weight'] += weight
		elif g.has_edge(node1,node0):
			g.get_edge_data(node1,node0)['weight'] += weight
		else:
			g.add_edge(node0,node1,weight=1,)

		return g

	net_data = {}
	if isinstance(data,nx.Graph):
		for n,nd in list(data.nodes(data=True)):
			if nd["node_type"]==node1:
				net_data[n]=[]
				for on,e,ed in list(data.edges(n,data=True)):
					net_data[n].append([e,ed["weight"]])
	del data
	gc.collect()
	start_time = time.time()
	num_cores = 12
	N = num_cores
	S = int(len(net_data)/N)
	net_data = list(hlp.chunks_optimized(net_data,n_chunks=num_cores))
	#net_data = [ net_data.iloc[i*S:(i+1)*S] for i in range(N) ]
	pool = Pool(num_cores)
	rep_data = {}
	results = pool.map(bi_to_uni,net_data)
	del net_data
	gc.collect()
	print("--- %s seconds --- for num cores {0} to reproject data".format(num_cores) % (time.time() - start_time))
	start_time = time.time()
	g = nx.Graph()
	for result in results:
		for k_tup, w in result.items():
			g = add_node_and_edges(g,k_tup[0],k_tup[1],w)
	print (len(g.nodes()))
	print (len(g.edges()))
	print("--- %s seconds --- to build network" % (time.time() - start_time))

	return g

if __name__ == "__main__":

	only_platforms = ["facebook","twitter","vkontakte","reddit","youtube",
				"telegram","tiktok","gab","instagram","web"]

	"""create_bi_ego_graph(actor_selection={"org_project_title":"altmed_denmark"},
						selection_types=["actor"],actors=['Free Observer'],
						between_dates={"start_date":"2021-08-01","end_date":"2022-01-01"},
						only_platforms=only_platforms)"""
	create_bi_ego_graph(url_selection={"Url" : {"$regex" : "freeobserver.org"}},
						selection_types=["url","actor"],urls=[],
						between_dates={"start_date":"2019-01-01","end_date":"2022-01-10"},
						only_platforms=only_platforms,actors=['Free Observer'],actor_selection={"org_project_title":"altmed_denmark"})
	"""create_bi_ego_graph(actor_selection={"org_project_title":"altmed_denmark","Iteration": { "$in": [0] }},
						selection_types=["actor"],actors=[],
						between_dates={"start_date":"2019-01-01","end_date":"2020-04-01"},
						only_platforms=only_platforms)"""
	"""create_bi_ego_graph(actor_selection={"org_project_title":"altmed_denmark","Iteration": { "$in": [0] }},
						selection_types=["actor"],actors=["Arbejderen"],
						between_dates={"start_date":"2020-01-01","end_date":"2021-01-01"},
						only_platforms=only_platforms)"""
	#create_bi_ego_graph(selection={"org_project_title":"altmed_denmark","Iteration": { "$in": [0] }},between_dates={"start_date":"2021-07-01","end_date":"2022-01-01"})
	#get_actor_query_overlap()
	#iteration_test()

	sys.stderr.write("Calculating NC score...\n")
	table = table.copy()
	src_sum = table.groupby(by = "src").sum()[["nij"]]
	table = table.merge(src_sum, left_on = "src", right_index = True, suffixes = ("", "_src_sum"))
	trg_sum = table.groupby(by = "trg").sum()[["nij"]]
	table = table.merge(trg_sum, left_on = "trg", right_index = True, suffixes = ("", "_trg_sum"))
	table.rename(columns = {"nij_src_sum": "ni.", "nij_trg_sum": "n.j"}, inplace = True)
	table["n.."] = table["nij"].sum()
	table["mean_prior_probability"] = ((table["ni."] * table["n.j"]) / table["n.."]) * (1 / table["n.."])
	if calculate_p_value:
		 table["score"] = binom.cdf(table["nij"], table["n.."], table["mean_prior_probability"])
		 return table[["src", "trg", "nij", "score"]]
	table["kappa"] = table["n.."] / (table["ni."] * table["n.j"])
	table["score"] = ((table["kappa"] * table["nij"]) - 1) / ((table["kappa"] * table["nij"]) + 1)
	table["var_prior_probability"] = (1 / (table["n.."] ** 2)) * (table["ni."] * table["n.j"] * (table["n.."] - table["ni."]) * (table["n.."] - table["n.j"])) / ((table["n.."] ** 2) * ((table["n.."] - 1)))
	table["alpha_prior"] = (((table["mean_prior_probability"] ** 2) / table["var_prior_probability"]) * (1 - table["mean_prior_probability"])) - table["mean_prior_probability"]
	table["beta_prior"] = (table["mean_prior_probability"] / table["var_prior_probability"]) * (1 - (table["mean_prior_probability"] ** 2)) - (1 - table["mean_prior_probability"])
	table["alpha_post"] = table["alpha_prior"] + table["nij"]
	table["beta_post"] = table["n.."] - table["nij"] + table["beta_prior"]
	table["expected_pij"] = table["alpha_post"] / (table["alpha_post"] + table["beta_post"])
	table["variance_nij"] = table["expected_pij"] * (1 - table["expected_pij"]) * table["n.."]
	table["d"] = (1.0 / (table["ni."] * table["n.j"])) - (table["n.."] * ((table["ni."] + table["n.j"]) / ((table["ni."] * table["n.j"]) ** 2)))
	table["variance_cij"] = table["variance_nij"] * (((2 * (table["kappa"] + (table["nij"] * table["d"]))) / (((table["kappa"] * table["nij"]) + 1) ** 2)) ** 2)
	table["sdev_cij"] = table["variance_cij"] ** .5
	if not return_self_loops:
		table = table[table["src"] != table["trg"]]
	if undirected:
		table = table[table["src"] <= table["trg"]]
	return table[["src", "trg", "nij", "score", "sdev_cij"]]
