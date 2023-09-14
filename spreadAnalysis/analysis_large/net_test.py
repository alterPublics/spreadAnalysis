import networkit as nxk
import pandas as pd
import time
import numpy as np
from multiprocessing import Pool
import multiprocessing
from itertools import chain
import polars as pl
import datetime
import psutil
from spreadAnalysis.persistence.schemas import Spread
import random
import sys

try:
	from spreadAnalysis.persistence.mongo import MongoSpread
	from spreadAnalysis.utils.link_utils import LinkCleaner
except:
	pass

def chunks(l, n):
		for i in range(0, n):
				yield l[i::n]


def _get_posts_from_ids(ids):

	mdb = MongoSpread()
	dts = {}
	new_dfs = []
	all_mids = ids["dt"].to_list()
	for post in mdb.database["post"].find({"message_id":{"$in":all_mids}},{"message_id":1,"method":1,**UNI_DATE_FIELDS},batch_size=5000):
		dt = Spread()._get_date(data=post,method=post["method"])
		print (dt)
		dts[post["message_id"]]=dt
	new_dfs.append(ids.replace("dt",pl.Series(all_mids).map_dict(dts)).with_columns(pl.col("dt").str.to_datetime("%Y-%m-%d %H:%M:%S")))
	return new_dfs 

ncores = 4
mdb = MongoSpread()
post_ids = pl.DataFrame([d["message_id"] for d in mdb.database["post"].find({"method":"crowdtangle"},{"message_id":1}).skip(1).limit(50000)],schema=["dt"])
results = Pool(ncores).map(_get_posts_from_ids,chunks(post_ids,ncores))
#results = [_get_posts_from_ids(post_ids)]
post_info = []
for result in results:
	post_info.extend(result)

print (post_info)
print (len(post_info))
sys.exit()

"""
@timer
def uni_m_to_graph(g,edge_m,ncores=-1):
		
		if ncores < 1:
				ncores = multiprocessing.cpu_count()
		inputs = []
		for chunk in chunks(list(edge_m),ncores):
				inputs.append((nxk.graphtools.copyNodes(g),chunk))
		ngs = Pool(ncores).map(_multi_insert_edges,inputs)
		g = nxk.graph.Graph(n=0, weighted=True, directed=False, edgesIndexed=False)
		for ng in ngs:
			 nxk.graphtools.merge(g,ng)
		
		return g
def _multi_insert_edges(g_m_edges):

g = g_m_edges[0]
edge_m = g_m_edges[1]
for row in edge_m:
		for comb in row:
				o = comb[0]
				e = comb[1]
				g.increaseWeight(o, e, 1)
return g
"""

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

def chunks(l, n):
		for i in range(0, n):
				yield l[i::n]

def filter_on_degrees(g,mind=2):

		to_remove = []
		for n in list(g.iterNodes()):
			if g.degree(n) < mind:
				to_remove.append(n)
		for n in to_remove:
			g.removeNode(n)
			
		return g

def create_rev_net_idx(g,net_idx):
		
		rev_net_idx = {v:k for k,v in net_idx.items() if g.hasNode(v)}
		return rev_net_idx

@timer
def to_uni_matrix(g,ntc):

	bi_neighbors = []
	for n in ntc:
		ns = np.array(sorted(list(g.iterNeighbors(n))))
		bi_neighbors.append(ns)
		g.removeNode(n)
	edge_m = uni_neighbors(bi_neighbors)

	return edge_m

@timer
def edge_df_to_graph(g,edge_m,ncores=-1):
		
		for row in edge_m.to_numpy():
				#g.increaseWeight(int(row[0]), int(row[1]), int(row[2]))
				g.addEdge(int(row[0]), int(row[1]), w=int(row[2]))
		return g

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
def df_to_nxk(s_t):

	g = nxk.graph.Graph(n=0, weighted=True, directed=False, edgesIndexed=False)
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
	Y = np.concatenate(Pool(ncores).map(_bi_permutation,chunks(X,ncores)),axis=0)
	
	return Y

def _bi_permutation(X):

	#Y = np.empty((0,2))
	Y = []
	for i in range(len(X)):
		x = X.pop()
		y = x[np.stack(np.triu_indices(len(x), k=1), axis=-1)]
		#Y = np.vstack((y,Y))
		Y.append(y)
	Y = np.vstack(Y)
	return Y

@timer
def _test_iter():
	 
		s = 0
		for r in range(50000000):
				s+=1
		print (s)

@timer
def edge_m_to_df(edge_m):
		#df = pd.DataFrame(edge_m,columns=["o","e"])
		#df = df.groupby(["o","e"]).size().reset_index()
		df = pl.DataFrame(edge_m,schema=["o","e"],schema_overrides={"o":pl.Int64,"e":pl.Int64})
		df = df.groupby(["o", "e"]).agg(weight=pl.count())
		return df

@timer
def noise_corrected(df, undirected = True):
	 
	src_sum = df.groupby("o").agg(o_sum=pl.sum("weight"))
	trg_sum = df.groupby("e").agg(e_sum=pl.sum("weight"))
	#src_sum = table.groupby(by = "src").sum()[["nij"]]
	#table = table.merge(src_sum, left_on = "src", right_index = True, suffixes = ("", "_src_sum"))
	#trg_sum = table.groupby(by = "trg").sum()[["nij"]]
	df = df.join(src_sum,how="left",on="o")
	df = df.join(trg_sum,how="left",on="e")
	#table = table.merge(trg_sum, left_on = "trg", right_index = True, suffixes = ("", "_trg_sum"))
	#table.rename(columns = {"nij_src_sum": "ni.", "nij_trg_sum": "n.j"}, inplace = True)
	df = df.with_columns(df.select(pl.sum("weight"))["weight"].alias("n.."))
	df = df.with_columns((((pl.col("o_sum") * pl.col("e_sum")) / pl.col("n..")) * (1 / pl.col("n..")) ).alias("mean_prior_probability"))
	df = df.with_columns((pl.col("n..")/(pl.col("o_sum")*pl.col("e_sum"))).alias("kappa"))
	df = df.with_columns((((pl.col("kappa")*pl.col("weight"))-1)/((pl.col("kappa")*pl.col("weight"))+1)).alias("score"))
	df = df.with_columns(((1/(pl.col("n..")**2))*(pl.col("o_sum")*pl.col("e_sum")*(pl.col("n..")-pl.col("o_sum"))*(pl.col("n..")-pl.col("e_sum")))/((pl.col("n..")**2)*(pl.col("n..")-1))).alias("var_prior_probability"))
	df = df.with_columns((((pl.col("mean_prior_probability")**2)/pl.col("var_prior_probability"))*(1-pl.col("mean_prior_probability"))-pl.col("mean_prior_probability")).alias("alpha_prior"))
	df = df.with_columns(((pl.col("mean_prior_probability")/pl.col("var_prior_probability"))*(1-(pl.col("mean_prior_probability")**2))-(1-pl.col("mean_prior_probability"))).alias("beta_prior"))
	df.drop_in_place("mean_prior_probability")
	df = df.with_columns((pl.col("alpha_prior")+pl.col("weight")).alias("alpha_post"))
	df.drop_in_place("alpha_prior")
	df = df.with_columns((pl.col("n..")-pl.col("weight")+pl.col("beta_prior")).alias("beta_post"))
	df.drop_in_place("beta_prior")
	df = df.with_columns((pl.col("alpha_post")/(pl.col("alpha_post")+pl.col("beta_post"))).alias("expected_pij"))
	df.drop_in_place("alpha_post")
	df.drop_in_place("beta_post")
	df = df.with_columns((pl.col("expected_pij")*(1-pl.col("expected_pij"))*pl.col("n..")).alias("variance_nij"))
	df.drop_in_place("expected_pij")
	df = df.with_columns(((1.0/(pl.col("o_sum")*pl.col("e_sum")))-(pl.col("n..")*((pl.col("o_sum")+pl.col("e_sum")) / ((pl.col("o_sum")*pl.col("e_sum"))**2)))).alias("d"))
	df = df.with_columns((pl.col("variance_nij")*(((2*(pl.col("kappa")+(pl.col("weight")*pl.col("d")))) / (((pl.col("kappa")*pl.col("weight"))+1)**2))**2)).alias("variance_cij"))
	df = df.with_columns((pl.col("variance_cij")**.5).alias("sdev_cij"))
	#table["mean_prior_probability"] = ((table["ni."] * table["n.j"]) / table["n.."]) * (1 / table["n.."])
	#table["kappa"] = table["n.."] / (table["ni."] * table["n.j"])
	#table["score"] = ((table["kappa"] * table["nij"]) - 1) / ((table["kappa"] * table["nij"]) + 1)
	#table["var_prior_probability"] = (1 / (table["n.."] ** 2)) * (table["ni."] * table["n.j"] * (table["n.."] - table["ni."]) * (table["n.."] - table["n.j"])) / ((table["n.."] ** 2) * ((table["n.."] - 1)))
	#table["alpha_prior"] = (((table["mean_prior_probability"] ** 2) / table["var_prior_probability"]) * (1 - table["mean_prior_probability"])) - table["mean_prior_probability"]
	#table["beta_prior"] = (table["mean_prior_probability"] / table["var_prior_probability"]) * (1 - (table["mean_prior_probability"] ** 2)) - (1 - table["mean_prior_probability"])
	#table["alpha_post"] = table["alpha_prior"] + table["nij"]
	#table["beta_post"] = table["n.."] - table["nij"] + table["beta_prior"]
	#table["expected_pij"] = table["alpha_post"] / (table["alpha_post"] + table["beta_post"])
	#table["variance_nij"] = table["expected_pij"] * (1 - table["expected_pij"]) * table["n.."]
	#table["d"] = (1.0 / (table["ni."] * table["n.j"])) - (table["n.."] * ((table["ni."] + table["n.j"]) / ((table["ni."] * table["n.j"]) ** 2)))
	#table["variance_cij"] = table["variance_nij"] * (((2 * (table["kappa"] + (table["nij"] * table["d"]))) / (((table["kappa"] * table["nij"]) + 1) ** 2)) ** 2) 
	#table["sdev_cij"] = table["variance_cij"] ** .5
	if undirected:
		df = df.filter(pl.col("o") <= pl.col("e"))
	return df.select(pl.col(["o", "e", "weight", "score", "sdev_cij"]))

@timer
def filter_on_backbone(df,threshold=1.0,max_edges=-1):
	
	if max_edges > 0:
		new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0))
		while len(new_df) > max_edges:
			threshold = threshold*1.05
			new_df = new_df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0))
	new_df = df.filter((pl.col("score")-(float(threshold)*pl.col("sdev_cij"))>0))
	return new_df.select(pl.col(["o", "e", "weight"]))

def _chunk_neighbors(args):

	chunked_entities = args[0]
	pls = args[1]
	entity_type = args[2]
	exclude = args[3]
	url_min_occur = args[4]
	mdb = MongoSpread()
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
			query = mdb.database["url_bi_network"].find({"domain":{"$in":list(entity_c)},"platform":{"$in":pls},"occurences":{"$gte":url_min_occur}},{"domain":1,"url":1,"message_ids":1})
		for doc in query:
			if entity_type == "domain":
				insert_data.append({"o":doc["url"],"e":doc["domain"],"w":len(doc["message_ids"])})
			else:
				insert_data.append({"o":doc["url"],"e":doc["actor_platform"],"w":len(doc["message_ids"])})
		if len(insert_data) > 0: dfs.append(pl.DataFrame(insert_data))

	return dfs

def import_data(project=None,num_cores=32):
	 
	def get_clean_actor_info(mdb,pls,projects=["altmed_denmark"],iterations=[0,0.0]):
			
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
					data.append(new_doc)
			return pd.DataFrame(data)
	
	def get_actor_platform_from_actor_info(mdb,actor_info,pls,strict=True):

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
				print (a)
		print (f"Supposed to find: {len(supposed_to_find)}")
		print (f"Found: {len(aps)}")
		return aps

	@timer
	def get_neighbors(entities,entity_type="url",mdb=None,format_="polars",pls=[],verbose=True,url_min_occur=2,exclude=[],num_cores=1):

		if not pls:
			pls = ["twitter","facebook","instagram","telegram","gab","youtube","fourchan","vkontakte","reddit"]
		dfs = []
		if len(entities) > 1000:
			n_chunks = 10*(1+int(len(entities)/1000))
		else:
			n_chunks = 1
		entities = list(entities)
		random.shuffle(entities)
		chunked_entities = chunks(entities,n_chunks)
		if num_cores < 2:
			results = [_chunk_neighbors([chunked_entities,pls,entity_type,exclude,url_min_occur])]
		else:
			results = Pool(num_cores).map(_chunk_neighbors,[[cc,pls,entity_type,exclude,url_min_occur] for cc in chunks(list(chunked_entities),num_cores)])
		for result in results:
			dfs.extend(result)
		df = pl.concat(dfs)

		return df

	mdb = MongoSpread()

	pls = ["twitter","facebook","instagram","telegram","gab","youtube","fourchan","vkontakte","reddit"]
	if project is not None:
		seed_actors = get_clean_actor_info(mdb,pls,projects=[project])
	#seed_actors = seed_actors[seed_actors["actor"]=="Solidaritet"]
	seed_actor_platform = get_actor_platform_from_actor_info(mdb,seed_actors,pls)
	print ("Getting zero degs...")
	shared_by_alt = get_neighbors(list(seed_actor_platform.keys()),entity_type="actor_platform")
	print (len(shared_by_alt))
	alt_shared_dom = get_neighbors(list(set(list(seed_actors["domain"]))),entity_type="domain")
	print (len(alt_shared_dom))
	alt_shared_dom = alt_shared_dom.filter(~pl.col('o').is_in(shared_by_alt["o"].to_list()))
	print (len(alt_shared_dom))
	zero_deg = pl.concat([shared_by_alt,alt_shared_dom])
	print (len(zero_deg))
	print (len(zero_deg.unique(subset=["o"])))
	print ("Getting first degs...")
	alt_shared_by_others = get_neighbors(zero_deg.unique(subset=["o"])["o"].to_list(),entity_type="url")
	print (len(alt_shared_by_others))
	first_deg = alt_shared_by_others.filter(~pl.col('e').is_in(zero_deg["e"].to_list()))
	print (len(first_deg))
	first_deg = first_deg.unique(subset=["o","e","w"])
	print (len(first_deg))
	first_deg_search = list(first_deg.unique(subset=["e"])["e"].to_list())
	print (f"Getting first degs - n = {len(first_deg_search)}")
	shared_by_first_deg = get_neighbors(first_deg_search,entity_type="actor_platform",num_cores=num_cores)
	print (len(shared_by_first_deg))
	first_deg = pl.concat([first_deg,shared_by_first_deg])
	print (len(first_deg))
	first_deg = first_deg.unique(subset=["o","e","w"])
	print (len(first_deg))
	second_deg_search = list(first_deg.filter(~pl.col("o").is_in(zero_deg["o"].to_list())).unique(subset=["o"])["o"].to_list())
	#print (f"Getting second degs - n = {len(second_deg_search)}")
	#coshared_first_deg = get_neighbors(second_deg_search,entity_type="url",num_cores=num_cores)
	#print (len(coshared_first_deg))
	#full = pl.concat([zero_deg,first_deg,coshared_first_deg])
	full = pl.concat([zero_deg,first_deg])
	print (len(full))
	full = full.unique(subset=["o","e","w"])
	print (len(full))
	full = full.filter(~pl.col("e").is_in(full.groupby(["e"]).agg(pl.count()).filter(pl.col("count")<2)["e"].to_list()))
	print (len(full))
	
	return full


main_path = "/work/JakobBækKristensen#1091/alterpublics/projects/altmed"

"""for p in ["altmed_denmark","altmed_sweden","altmed_austria","altmed_germany"][2:]:
	data = import_data(project=p)
	data.write_csv(main_path+f"/{p}_2deg.csv", separator=",")
sys.exit()"""

bf_load = datetime.datetime.now()
#df = pd.read_csv(main_path+"/"+"test_large.csv")
#df = pd.read_csv(main_path+"/"+"solidaritet.csv")
#df = pl.read_csv(main_path+"/"+"solidaritet.csv",columns=["url","actor_platform"],dtypes={"url":pl.Utf8,"actor_platform":pl.Utf8})
#df = pl.read_csv(main_path+"/"+"test_vlarge.csv",columns=["url","actor_platform"],dtypes={"url":pl.Utf8,"actor_platform":pl.Utf8})
df = pl.read_csv(main_path+"/"+"altmed_denmark_2deg.csv",columns=["o","e","w"],dtypes={"url":pl.Utf8,"actor_platform":pl.Utf8})
#df = pl.from_pandas(df)
#print (len(df.unique(subset="actor_platform")))
#print (len(df.unique(subset="url")))
print ()
print (f"Total time to load data = {(datetime.datetime.now()-bf_load).total_seconds()} seconds")

"""print (df.groupby(["url","actor_platform"]).size().reset_index())
print (df1.groupby(["url", "actor_platform"]).agg(pl.count()))
e_tups = create_edge_tuples(df)
e_tups1 = create_edge_tuples(df1)
e_tups = list(e_tups)
e_tups1 = list(e_tups1)
print (list(e_tups)[2])
print (list(e_tups1)[3])
print (type(list(e_tups)[0][1]))
print (type(list(e_tups1)[0][1]))
g, net_idx = df_to_nxk(e_tups)
g1, net_idx1 = df_to_nxk(e_tups1)
g2, net_idx2 = df_to_nxk(e_tups1)
g3, net_idx3 = df_to_nxk(e_tups1)
print (len(net_idx))
print (len(net_idx1))
print (g.numberOfEdges())
print (g1.numberOfEdges())
print (g2.numberOfEdges())
print (g3.numberOfEdges())
g = filter_on_degrees(g,mind=2)
g1 = filter_on_degrees(g1,mind=2)
g2 = filter_on_degrees(g1,mind=2)
g3 = filter_on_degrees(g,mind=2)
print (g.numberOfEdges())
print (g1.numberOfEdges())
print (g2.numberOfEdges())
print (g3.numberOfEdges())
rev_net_idx = {v:k for k,v in net_idx.items()}
rev_net_idx1 = {v:k for k,v in net_idx1.items()}
for o,e in g1.iterEdges():
	 ro = rev_net_idx1[o]
	 re = rev_net_idx1[e]
	 if not g.hasEdge(net_idx[ro], net_idx[re]):
			print (ro)
			print (re)
	 
sys.exit()"""

"""g = nxk.graph.Graph(n=0, weighted=True, directed=False, edgesIndexed=False)
g.addNode()
g.addNode()
g.increaseWeight(0, 1, 1)
g.increaseWeight(1, 0, 1)
g.increaseWeight(1, 0, 1)
print (g.totalEdgeWeight())
print (g.numberOfEdges())
sys.exit()"""

mems = []
bf_net = datetime.datetime.now()
e_tups = create_edge_tuples(df)
mems.append(psutil.Process().memory_info().rss / (1024 * 1024))
g, net_idx = df_to_nxk(e_tups)
e_tups = None
mems.append(psutil.Process().memory_info().rss / (1024 * 1024))
print (g.numberOfNodes())
print (g.numberOfEdges())
g = filter_on_degrees(g,mind=2)
nodes_to_collapse = get_collapse_node_list(g,df,net_idx,col="o")
#nodes_to_collapse = [net_idx[n] for n in set(list(df["url"])) if g.hasNode(net_idx[n])]
mems.append(psutil.Process().memory_info().rss / (1024 * 1024))
print (g.numberOfNodes())
print (g.numberOfEdges())
print (len(nodes_to_collapse))
edge_m = to_uni_matrix(g,nodes_to_collapse)
mems.append(psutil.Process().memory_info().rss / (1024 * 1024))
print (edge_m.shape)
edge_df = edge_m_to_df(edge_m)
edge_m = None
mems.append(psutil.Process().memory_info().rss / (1024 * 1024))
print (edge_df.shape)
edge_df = noise_corrected(edge_df)
edge_df = filter_on_backbone(edge_df,threshold=1.0)
print (edge_df.shape)
ug = edge_df_to_graph(g,edge_df)
rev_net_idx = create_rev_net_idx(ug,net_idx)
#print (nxk.centrality.DegreeCentrality(g, normalized=True, outDeg=True))
mems.append(psutil.Process().memory_info().rss / (1024 * 1024))
print (ug.numberOfNodes())
print (ug.numberOfEdges())
print ()
print (f"Total time to construct nets = {(datetime.datetime.now()-bf_load).total_seconds()} seconds")
print (f"Using at most {round(max(mems)/1024,3)}GB of memory")


@timer
def stlp(g,labels,net_idx,title="test",num_cores=1,its=5,epochs=3,stochasticity=1,verbose=False):

	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	max_deg = nxk.graphtools.maxDegree(g)
	org_nodes = list([n for n in g.iterNodes()])
	affinities = []
	all_dists = {}
	org_labels = {net_idx[n]:l for n,l in labels.items() if n in net_idx}
	labels = {}

	print ("generating neighbour idxs")
	nn_idx = {n:_stlp_get_n_idx_ws([g.iterNeighborsWeights(n)]) for n in org_nodes}

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
		dist_map = np.array([np.mean(((np.array(v)+1)**-1)**(np.array(v)+1), axis=0) for k,v in sorted(all_dists.items())]).T
		aff_map = np.zeros((len(net_idx),len(all_dists)))
		for n,l in labels.items():
			aff_map[n][aff_cats[l]]=1.0

		start_time = time.time()
		stochastics = result = [True] * int(stochasticity*its) + [False] * int((1-stochasticity)*its)
		if len(stochastics) < its:
			stochastics.append(False)
		random.shuffle(stochastics)

		if num_cores > 1:
			n_batches = int(np.ceil((len(org_nodes)/25)/(num_cores*25)))
		else:
			n_batches = int(np.ceil((len(org_nodes)/50)))
		for it in range(its):
			
			#Stochastiscism
			use_nodes = copy(org_nodes)
			if stochastics[it]:
				random.shuffle(use_nodes)
			
			new_aff_stack = []
			#multi_pre_chunks = np.array_split(org_nodes,4)
			#multi_pre_chunks = [(nodes,nxk.graphtools.subgraphAndNeighborsFromNodes(g, nodes, includeOutNeighbors=True, includeInNeighbors=True),max_deg,aff_cats,aff_map,dist_map) for nodes in multi_pre_chunks]
			#results = Pool(4).map(_batch_compute_affinity,multi_pre_chunks)
			ncount = 0
			save_all_counts = {}
			for pre_chunk in np.array_split(use_nodes,n_batches):
				batch_size = len(pre_chunk)
				n_labels = len(aff_cats)
				affs = np.zeros((batch_size,max_deg,n_labels))
				dists = np.zeros((batch_size,max_deg,n_labels))
				wss = np.zeros((batch_size,max_deg))
				for i,n in enumerate(pre_chunk):
					for n_idx, ws in nn_idx[n]:
						affs[i,:len(n_idx), :] = aff_map[n_idx, :]
						dists[i,:len(n_idx), :] = dist_map[n_idx, :]
						wss[i,:len(ws)] = ws
					save_all_counts[n]=ncount
					ncount+=1

				if num_cores > 1:
					for result in Pool(num_cores).map(_compute_affinity_multi,[(affs[idx_],dists[idx_],wss[idx_]) for idx_ in np.array_split([i for i in range(affs.shape[0])],num_cores)]):
						new_aff_stack.append(result)
				else:
					new_aff = _compute_affinity(affs,dists,wss)
					new_aff_stack.append(new_aff)

			new_aff_stack = np.concatenate(new_aff_stack,axis=0)
			if stochastics[it]:
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
	affinities = np.mean(np.array(affinities),axis=0)
	#out_affinities = {k:{title+"_"+lab:0.0 for lab in set(labels.values())} for k in net_idx.keys()}
	out_affinities = {title+"_"+lab:{k:0.0 for k in net_idx.keys()} for lab in set(labels.values())}
	for lab in set(labels.values()):
		for i,n in enumerate(affinities):
			#out_affinities[rev_net_ids[i]][title+"_"+lab]=n[aff_cats[lab]]
			out_affinities[title+"_"+lab][rev_net_ids[i]]=n[aff_cats[lab]]
	return {title:out_affinities}