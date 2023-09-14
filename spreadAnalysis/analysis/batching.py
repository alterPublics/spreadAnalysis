from spreadAnalysis.persistence.mongo import MongoSpread
from spreadAnalysis.analysis.network import BipartiteNet
from spreadAnalysis.utils import helpers as hlp
from spreadAnalysis.persistence.schemas import Spread
from spreadAnalysis.utils.link_utils import LinkCleaner
import pandas as pd
import time
import random
import numpy as np
from operator import itemgetter
import sys
import gc
import multiprocessing
from multiprocessing import Pool, Manager, set_start_method, get_context
import networkx as nx
from collections import defaultdict,Counter
from spreadAnalysis.io.config_io import Config
from datetime import datetime, timedelta
import warnings
import polars as pl
from sklearn import preprocessing
from newspaper import Article
from pymongo import InsertOne, DeleteOne, ReplaceOne, UpdateOne, UpdateMany

def bi_to_uni(data):

	rep_data = {}
	for k,v in data.items():
		for n1 in v:
			for n2 in v:
				if n1[0] != n2[0]:
					if (n1[0],n2[0]) in rep_data:
						rep_data[(n1[0],n2[0])]+=n1[1]
					elif (n2[0],n1[0]) in rep_data:
						rep_data[(n2[0],n1[0])]+=n2[1]
					else:
						rep_data[(n1[0],n2[0])]=n1[1]
	return rep_data

def update_actor_message(new=False):

	mdb = MongoSpread()
	actor_db = mdb.database["actor_platform_post"]
	try:
		actor_db.drop_index('actor_-1')
		actor_db.drop_index('platform_-1')
	except:
		pass
	if new:
		actor_db.drop()
		actor_db = mdb.database["actor_platform_post"]
		mdb.create_indexes()
	max_net_date = list(actor_db.find().sort("inserted_at",-1).limit(1))
	if max_net_date is None or len(max_net_date) < 1:
		max_net_date = datetime(2000,1,1)
	else:
		if "updated_at" in max_net_date[0]:
			max_net_date = max_net_date[0]["updated_at"]
		elif "inserted_at" in max_net_date[0]:
			max_net_date = max_net_date[0]["inserted_at"]
		max_net_date = max_net_date-timedelta(days=2)

	aliases = mdb.get_aliases()
	actor_aliases = mdb.get_actor_aliases(platform_sorted=False)
	post_db = mdb.database["post"]
	cur = post_db.find({"$or":[ {"updated_at": {"$gt": max_net_date}}, {"inserted_at": {"$gt": max_net_date}}]})
	next_post = True
	batch_insert = {}
	seen_ids = set([])
	count = 0
	while next_post is not None:
		count += 1
		next_post = next(cur, None)
		if next_post is not None:
			message_id = next_post["message_id"]
			post_obj_id = next_post["_id"]
			if not message_id in seen_ids:
				post = next_post
				platform = Spread._get_platform(data=post,method=post["method"])
				actor_id = Spread._get_actor_id(data=post,method=post["method"])
				platform_type = Spread._get_platform_type(data=post,method=post["method"])
				actor_username = Spread._get_actor_username(data=post,method=post["method"])
				actor_label = str(Spread._get_actor_name(data=post,method=post["method"]))+" ({0})".format(str(platform_type))
				url = Spread._get_message_link(data=post,method=post["method"])
				domain = Spread._get_message_link_domain(data=post,method=post["method"])
				if actor_username in actor_aliases and platform_type in actor_aliases[actor_username]:
					actor = actor_aliases[actor_username][platform_type]
					actor_platform = str(actor)+"_"+str(platform_type)
					actor_label = str(Spread._get_actor_name(data=post,method=post["method"]))+" ({0})".format(str(platform_type))
				elif actor_id in actor_aliases and platform_type in actor_aliases[actor_id]:
					actor = actor_aliases[actor_id][platform_type]
					actor_platform = str(actor)+"_"+platform_type
					actor_label = str(Spread._get_actor_name(data=post,method=post["method"]))+" ({0})".format(str(platform_type))
				else:
					actor = actor_username
					actor_platform = str(actor_username)+"_"+platform_type
					actor_label = str(Spread._get_actor_name(data=post,method=post["method"]))+" ({0})".format(str(platform_type))
				if actor_platform is not None and actor_platform not in batch_insert:
					batch_insert[actor_platform]={"url":url,"actor_username":actor_username,
										"actor":actor,"actor_label":actor_label,"platform":platform,
										"post_obj_ids":[],"actor_platform":actor_platform,"domain":domain}
				batch_insert[actor_platform]["post_obj_ids"].append(post_obj_id)
				if len(batch_insert) >= 100000:
					mdb.write_many(actor_db,list(batch_insert.values()),key_col="actor_platform",sub_mapping="post_obj_ids")
					batch_insert = {}
				if count % 100000 == 0:
					print ("actor loop " + str(count))
	if len(batch_insert) > 0:
		mdb.write_many(actor_db,list(batch_insert.values()),key_col="actor_platform",sub_mapping="post_obj_ids")
	actor_db.create_index([ ("actor", -1) ])
	actor_db.create_index([ ("platform", -1) ])

def create_enga_transformer_actor(mdb,n_sample=10000):

	def transform_per_platform_multi(df,vals,cat_var,appr=False):
    
		dfs = []
		lambdas = {}
		for cat in set(list(df[cat_var])):
			new_df = df.copy()[df[cat_var]==cat]
			x = new_df[vals] #returns a numpy array
			min_max_scaler = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True, copy=True)
			x_scaled = min_max_scaler.fit_transform(x)
			x_scaled = pd.DataFrame(x_scaled,columns=[val+"_yj" for val in vals],index=new_df.index)
			new_df[[val+"_yj" for val in vals]] = x_scaled
			dfs.append(new_df)
			lambdas[cat]=min_max_scaler
		return pd.concat(dfs,axis=0), lambdas

	def normalize_per_platform_multi(df,vals,cat_var,appr=False):
		
		dfs = []
		lambdas = {}
		for cat in set(list(df[cat_var])):
			new_df = df.copy()[df[cat_var]==cat]
			x = new_df[vals] #returns a numpy array
			min_max_scaler = preprocessing.MinMaxScaler()
			x_scaled = min_max_scaler.fit_transform(x)
			x_scaled = pd.DataFrame(x_scaled,columns=[val+"_norm" for val in vals],index=new_df.index)
			new_df[[val+"_norm" for val in vals]] = x_scaled
			dfs.append(new_df),min_max_scaler
			lambdas[cat]=min_max_scaler
		return pd.concat(dfs,axis=0),lambdas

	def get_scaler_per_group(df,val,group,scaler=preprocessing.MinMaxScaler()):
		
		scaler_per_group = {}
		df_enga,lams = transform_per_platform_multi(df,["engagement"],"platform")
		df_enga,lams1 = normalize_per_platform_multi(df_enga,["engagement_yj"],"platform")
		for cat in set(list(df[group])):
			scaler_per_group[cat]=[lams[cat],lams1[cat]]
		return scaler_per_group
	
	pls = ["twitter","facebook","instagram","telegram","gab","youtube","fourchan","vkontakte","reddit"]
	data = []
	for pl in pls:
		docs = mdb.database["actor_metric"].aggregate([{ "$match": { "platform":{"$eq":pl}}} ,{"$sample":{"size":n_sample}}])
		for doc in docs:
			data_doc = {}
			for f in ["reactions_mean","shares_mean","comments_mean","platform"]:
				data_doc[f]=doc[f]
				if pl == "reddit" and f == "reactions_mean":
					data_doc[f]=0.0
				if pl == "twitter" and f == "reactions_mean":
					data_doc[f]=doc[f]+doc["followers_mean"]
			data.append(data_doc)
	df = pd.DataFrame(data)
	df["engagement"]=df["comments_mean"]+df["reactions_mean"]+df["shares_mean"]
	s_p_g = get_scaler_per_group(df,"engagement","platform",scaler=[preprocessing.PowerTransformer(method='yeo-johnson', standardize=True, copy=True),preprocessing.MinMaxScaler()])
	"""data = pd.DataFrame([{'reactions_mean': 61.111111111111114, 'shares_mean': 0.3333333333333333, 'comments_mean': 0.1111111111111111, 'platform': 'twitter'}])
	data["engagement"]=data["comments_mean"]+data["reactions_mean"]+data["shares_mean"]
	e = s_p_g["twitter"][0].transform(data[["engagement"]])
	print (df[df["platform"]=="twitter"].sort_values("engagement",ascending=False))
	print (e)
	print (s_p_g["twitter"][1].transform(e))
	sys.exit()"""

	return s_p_g

def create_enga_transformer(mdb,n_sample=(25000,10000)):

	def get_scaler_per_group(df,val,group,scaler=preprocessing.MinMaxScaler()):
		
		scaler_per_group = {}
		for cat in set(list(df[group])):
			new_df = df.copy()[df[group]==cat]
			x = new_df[[val]]
			if isinstance(scaler,list):
				all_scalers = []
				for scale in scaler:
					scale.fit(x)
					all_scalers.append(scale)
				scaler_per_group[cat]=all_scalers
			else:
				scaler.fit(x)
				scaler_per_group[cat]=scaler
		return scaler_per_group
	
	pls = ["twitter","facebook","instagram","telegram","gab","youtube","fourchan","vkontakte","reddit"]
	n_actors = n_sample[0]
	n_posts = n_sample[1]
	aps = {k:[] for k in pls}
	for pl in pls:
		docs = mdb.database["actor_platform_post"].aggregate([{ "$match": { "platform":{"$eq":pl}}} ,{"$sample":{"size":n_actors}},{"$project": { "actor_platform": 1}}])
		for doc in docs:
			aps[pl].append(doc["actor_platform"])
	pl_sampls = {k:[] for k in pls}
	for pl in pls:
		while len(pl_sampls[pl]) < n_posts:
			doc = mdb.database["actor_platform_post"].find_one({"actor_platform":random.choice(aps[pl])})
			if doc is not None:
				oids = list(doc["post_obj_ids"])
				if pl == "twitter":
					is_not_rt_count = 0
					random.shuffle(oids)
					for oid in oids:
						post = mdb.database["post"].find_one({"_id":oid})
						if not Spread._get_is_retweet(data=post,method=post["method"]) and not Spread._get_is_reply(data=post,method=post["method"]):
							pl_sampls[pl].append(post)
							is_not_rt_count+=1
						if is_not_rt_count>=2:
							break
				else:
					for r in range(2):
						pl_sampls[pl].append(mdb.database["post"].find_one({"_id":random.choice(oids)}))
	data = []
	for k,v in pl_sampls.items():
		for d in v:
			doc = {}
			if k == "twitter":
				doc.update({"engagement":(Spread._get_engagement(data=d,method=d["method"]))})
			else:
				doc.update({"engagement":Spread._get_engagement(data=d,method=d["method"])})
			doc.update({"platform":Spread._get_platform(data=d,method=d["method"])})
			data.append(doc)
	df = pd.DataFrame(data)
	return get_scaler_per_group(df,"engagement","platform",scaler=[preprocessing.PowerTransformer(method='yeo-johnson', standardize=True, copy=True),preprocessing.MinMaxScaler()])

def aggregate_actor_data(args):

	actor_batch = args[0]
	actor_info = args[1]
	process_id = args[2]
	enga_trans = args[3]
	actor_data = []
	mdb = MongoSpread()
	actor_count = 0
	for actor, poids in actor_batch.items():
		actor_count+=1
		print (actor + " - " + str(actor_count) + " - " + str(len(poids)) + " for process " + str(process_id))
		most_popular_url_shared = defaultdict(int)
		unique_domains = set([])
		most_often_shared_domain = []
		posts = set([])
		langs = []
		interactions = []
		engagement = []
		reactions = []
		comments = []
		shares = []
		text_lengths = []
		post_dates = []
		followers = []
		#print ("iterating posts")
		for post_doc in mdb.database["post"].find({"_id":{"$in":list(poids)}}):
			posts.add(post_doc["_id"])
			#posts.add(poid)
			#post_doc = mdb.database["post"].find_one({"_id":poid})
			langs.append(Spread._get_lang(data=post_doc,method=post_doc["method"],model=None))
			platform = Spread._get_platform(data=post_doc,method=post_doc["method"])
			interactions.append(Spread._get_interactions(data=post_doc,method=post_doc["method"]))
			if platform == "twitter":
				engagement.append((Spread._get_engagement(data=post_doc,method=post_doc["method"])))
			else:
				engagement.append(Spread._get_engagement(data=post_doc,method=post_doc["method"]))
			reactions.append(Spread._get_reactions(data=post_doc,method=post_doc["method"]))
			comments.append(Spread._get_comments(data=post_doc,method=post_doc["method"]))
			shares.append(Spread._get_shares(data=post_doc,method=post_doc["method"]))
			text_lengths.append(len(Spread._get_message_text(data=post_doc,method=post_doc["method"])))
			post_dates.append(Spread._get_date(data=post_doc,method=post_doc["method"]))
			followers.append(Spread._get_followers(data=post_doc,method=post_doc["method"]))
			account_type = Spread._get_account_type(data=post_doc,method=post_doc["method"])
			account_category = Spread._get_account_category(data=post_doc,method=post_doc["method"])
			link_to_actor = Spread._get_link_to_actor(data=post_doc,method=post_doc["method"])

			url = Spread._get_message_link(data=post_doc,method=post_doc["method"])
			if url is not None:
				url = LinkCleaner().single_standardize_url(url)
				url = LinkCleaner().single_standardize_url(url)
				url = LinkCleaner().single_standardize_url(url)
				domain = Spread._get_message_link_domain(data=post_doc,method=post_doc["method"])
				most_popular_url_shared[url]+=1
				unique_domains.add(domain)
				most_often_shared_domain.append(domain)
		if len(most_popular_url_shared) > 0:
			most_poluar_url = sorted(most_popular_url_shared.items(), key = itemgetter(1), reverse=True)[0][0]
			most_shared_domain = sorted(dict(Counter(most_often_shared_domain)).items(), key = itemgetter(1), reverse=True)[0][0]
		else:
			most_poluar_url = None
			most_shared_domain = None
		real_post_dates = [hlp.to_default_date_format(d) for d in post_dates if d is not None]
		#print ("aggregating")
		for scaler in enga_trans[platform]:
			new_enga = scaler.transform(np.array(engagement).reshape(-1, 1))
			new_enga = scaler.transform(new_enga)
		actor_doc = {  "actor_name":actor_info[actor]["actor_label"],
						"actor":actor_info[actor]["actor"],
						"actor_username":actor_info[actor]["actor_username"],
						"actor_platform":actor,
						"most_popular_url_shared":most_poluar_url,
						"n_unique_domains_shared":len(unique_domains),
						"most_often_shared_domain":most_shared_domain,
						"n_posts":len(posts),
						"lang":sorted(dict(Counter(langs)).items(), key = itemgetter(1), reverse=True)[0][0],
						"interactions_mean":np.nanmean(np.array(interactions,dtype=np.float64)),
						"interactions_std":np.nanstd(np.array(interactions,dtype=np.float64)),
						"engagement_trans_mean":np.nanmean(new_enga,dtype=np.float64),
						"engagement_trans_std":np.nanstd(new_enga,dtype=np.float64),
						"engagement_mean":np.nanmean(engagement,dtype=np.float64),
						"engagement_std":np.nanstd(engagement,dtype=np.float64),
						"reactions_mean":np.nanmean(reactions,dtype=np.float64),
						"reactions_std":np.nanstd(reactions,dtype=np.float64),
						"comments_mean":np.nanmean(comments,dtype=np.float64),
						"comments_std":np.nanstd(comments,dtype=np.float64),
						"shares_mean":np.nanmean(shares,dtype=np.float64),
						"shares_std":np.nanstd(shares,dtype=np.float64),
						"message_length_mean":np.nanmean(np.array(text_lengths,dtype=np.float64)),
						"message_length_std":np.nanstd(np.array(text_lengths,dtype=np.float64)),
						"first_post_observed":min(real_post_dates),
						"last_post_observed":max(real_post_dates),
						"followers_mean":np.nanmean(np.array(followers,dtype=np.float64)),
						"followers_max":np.nanmax(np.array(followers,dtype=np.float64)),
						"platform":platform,
						"account_type":account_type,
						"account_category":account_category,
						"link_to_actor":link_to_actor,
						}
		actor_data.append(actor_doc)
	#print ("closing")
	mdb.close()
	#print ("returning " + str(process_id))
	return actor_data

def update_agg_actor_metrics(num_cores=1,new=False,skip_existing=False,missing=False):

	warnings.filterwarnings('ignore')
	np.seterr(all="ignore")
	mdb = MongoSpread()
	pls = ["twitter","facebook","instagram","telegram","gab","youtube","fourchan","vkontakte","reddit"]
	batch_size = 1000*num_cores
	actor_count = 0
	actor_metric_db = mdb.database["actor_metric"]
	actor_platform_db = mdb.database["actor_platform_post"]
	#actor_metric_db.drop()
	actor_platform = True
	batch_insert = {}
	actor_info = {}
	if not missing:
		if new:
			actor_metric_db.drop()
			actor_metric_db = mdb.database["actor_metric"]
			mdb.create_indexes()
			actor_obj_ids = [d["_id"] for d in actor_platform_db.find({"platform":{"$in":pls}},{"_id":1})]
		else:
			actor_metric_db.drop_index('actor_-1')
			actor_metric_db.drop_index('platform_-1')
			max_upd_date = list(actor_metric_db.find().sort("updated_at",-1).limit(1))
			if max_upd_date is None or len(max_upd_date) < 1 or "updated_at" not in max_upd_date[0]:
				max_upd_date = list(actor_metric_db.find().sort("inserted_at",-1).limit(1))
				if max_upd_date is None or len(max_upd_date) < 1:
					max_upd_date = datetime(2000,1,1)
				else:
					max_upd_date = max_upd_date[0]["inserted_at"]
			else:
				max_upd_date = max_upd_date[0]["updated_at"]
			max_upd_date = max_upd_date-timedelta(days=2)
			if skip_existing: max_upd_date = datetime(2000,1,1)
			actor_obj_ids = [d["_id"] for d in actor_platform_db.find({"platform":{"$in":pls}},{"$or":[ {"updated_at": {"$gt": max_upd_date}}, {"inserted_at": {"$gt": max_upd_date}}]},{"_id":1})]
	else:
		actor_platform_ids = set([])
		#already_updated = set(set([d["actor_platform"] for d in actor_metric_db.find({},{"actor_platform":1})]))
		#actor_platform_ids.update(set(set([d["actor_platform"] for d in actor_platform_db.find({"platform":{"$in":pls}},{"actor_platform":1}) if d["actor_platform"] not in already_updated])))
		print (len(actor_platform_ids))
		#actor_platform_ids.update(set(set([d["actor_platform"] for d in actor_metric_db.find({"actor_name":{"$exists":False}},{"actor_platform":1})])))
		actor_platform_ids.update(set(set([d["actor_platform"] for d in actor_metric_db.find({"platform":{"$ne":"twitter"}},{"actor_platform":1})])))
		print (len(actor_platform_ids))
		actor_obj_ids = [d["_id"] for d in actor_platform_db.find({"platform":{"$in":pls}},{"actor_platform":1,"_id":1}) if d["actor_platform"] in actor_platform_ids]
		print (len(actor_obj_ids))
	print ("Creating engagement transformation")
	enga_trans = create_enga_transformer(mdb)
	random.shuffle(actor_obj_ids)
	for a_obj in actor_obj_ids:
		actor_count += 1
		if actor_count % 10000 == 0:
			print ("actor loop " + str(actor_count))
		actor_platform = actor_platform_db.find_one({"_id":a_obj})
		if skip_existing:
			actor_metric = actor_metric_db.find_one({"actor_platform":actor_platform["actor_platform"]})
			if actor_metric is not None:
				continue
		if actor_platform is not None:
			unique_actor = actor_platform["actor_platform"]
			batch_insert[unique_actor]=list(actor_platform["post_obj_ids"])
			actor_info[unique_actor]={"actor_username":actor_platform["actor_username"],
										"actor_label":actor_platform["actor_label"],
										"actor":actor_platform["actor"]}
		if len(batch_insert) >= batch_size or actor_platform is None:
			if num_cores > 1:
				chunked_batches = [(l,actor_info,i,enga_trans) for i,l in enumerate(hlp.chunks_optimized(batch_insert,num_cores,verbose=False))]
				pool = Pool(num_cores)
				#with get_context("spawn").Pool(num_cores) as pool:
				results = pool.map(aggregate_actor_data,chunked_batches)
				pool.close()
				pool.join()
			else:
				results = [aggregate_actor_data((batch_insert,actor_info,0,enga_trans))]
			#print ("inserting")
			for result in results:
				mdb.write_many(actor_metric_db,result,key_col="actor_platform")
			batch_insert = {}
			actor_info = {}
	#cur.close()
	mdb.write_many(actor_metric_db,aggregate_actor_data((batch_insert,actor_info,0,enga_trans)),key_col="actor_platform")
	actor_metric_db.create_index([ ("actor", -1) ])
	actor_metric_db.create_index([ ("platform", -1) ])
	mdb.close()

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
			docs.extend(list(mdb.database[db].find(query,{"url":1,"actor":1,"message_ids":1,"actor_platform":1})))
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

def multi_find_urls(org_urls):

	prev_urls = set(find_urls(selection={},urls=org_urls))
	return prev_urls

def _find_domains_per_actor(actors):

	mdb = MongoSpread()
	data = []
	for doc in mdb.database["url_bi_network"].aggregate([{"$match":{"actor_platform":{"$in":list(actors)}}},{"$group":{"_id":{"actor_platform":"$actor_platform", "domain":"$domain"}, "total":{"$sum":{"$size":"$message_ids"}}}}]):
		data_doc = {"actor_platform":doc["_id"]["actor_platform"],"domain":doc["_id"]["domain"],"count":doc["total"]}
		if data_doc["domain"] is not None:
			if "youtube.com" in data_doc["domain"]: data_doc["domain"]="youtube.com"
			data.append(data_doc)
	return data

def multi_find_domains_per_actor(actors,ncores=-1):

	def nchunks(l, n):
		for i in range(0, n):
				yield l[i::n]

	url_shorts = set(list(set(list(pd.read_csv("/work/JakobBækKristensen#1091/alterpublics/projects/full_test"+"/"+"url_shorteners.csv")["domain"]))))
	min10_doms = pl.read_csv(Config().PROJECT_PATH+"/full_test/domain_per_actor_min10.csv")
	min10_doms_set = set(min10_doms["domain"].to_list())
	min10_doms_set.add("youtube.com")
	if ncores == -1:
		ncores = multiprocessing.cpu_count()-2
	all_results = []
	outer_chunk_size = 1
	if len(actors) > 60000:
		outer_chunk_size = 20
	for outer_chunk in nchunks(actors,outer_chunk_size):
		results = Pool(ncores).map(_find_domains_per_actor,nchunks(list(outer_chunk),ncores))
		for result in results:
			all_results.extend(result)
	df = pl.from_dicts(all_results)
	df = df.filter((pl.col("domain").is_in(min10_doms_set))&(~pl.col("domain").is_in(url_shorts)))
	df = df.join(min10_doms.select(pl.col(["domain","idf"])),on="domain",how="left")
	return df

def find_urls(selection={},urls=[]):

	mdb = MongoSpread()
	final_urls = []
	if len(selection) == 0 and len(urls) == 0:
		return set(final_urls)
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

def create_bi_ego_graph_OLD(selection_types=["actor"],actor_selection={},url_selection={},actors=[],urls=[],between_dates=None,only_platforms=[],title="test",num_cores=12,direct_urls=set([]),actor_domains=[],batch_size=2):

	def add_domains_as_actors(binet,actor_domains):

		actor_domains = set(actor_domains)
		for n,d in list(binet.g.nodes(data=True)):
			if d["node_type"] == "url":
				dom = LinkCleaner().remove_url_prefix(n).split("/")[0]
				if len(actor_domains) > 0:
					if dom in actor_domains:
						binet.add_node_and_edges(dom,n,node_type0="actor",node_type1="url",weight=1)
				else:
					binet.add_node_and_edges(dom,n,node_type0="actor",node_type1="url",weight=1)
		return binet

	def add_data_to_net(docs,binet,has_been_queried,org_type,extra=None,verbose=False):

		for doc in docs:
			if doc[org_type] not in has_been_queried:
				if doc["message_ids"] > 0:
					if doc["actor"] is not None and doc["url"] is not None and "actor_platform" in doc:
						if extra is not None:
							binet.add_node_and_edges(doc["actor_platform"],doc["url"],node_type0="actor",node_type1="url",weight=doc["message_ids"],node0_extra=doc[extra])
						else:
							binet.add_node_and_edges(doc["actor_platform"],doc["url"],node_type0="actor",node_type1="url",weight=doc["message_ids"])
					else:
						if verbose:
							print (doc)
						else:
							pass
		return binet

	if len(only_platforms) < 1:
		only_platforms = ["facebook","twitter","web","vkontakte","reddit","youtube",
					"telegram","tiktok","gab","instagram","fourchan"]

	mdb = MongoSpread()
	net_db = mdb.database["url_bi_network"]
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
			binet = add_data_to_net(result,binet,{},"actor",extra="actor")
		alias_queries = []
		for alias_doc in aliases:
			alias_queries.append({"actor":alias_doc["alias"],"platform":{"$in":only_platforms}})
			has_been_queried.add(alias_doc["alias"])
		results = pool.map(query_multi,[("url_bi_network",l) for l in hlp.chunks(alias_queries,int(len(alias_queries)/num_cores)+1)])
		for result in results:
			for fdocs in pool.map(filter_docs,[(l,"url",between_dates) for l in hlp.chunks(list(result),int(len(list(result))/num_cores)+1)]):
				binet = add_data_to_net(fdocs,binet,{},"actor",extra="actor")
	if "url" in selection_types:
		print ("urls...")
		if len(urls) > 0 and len(url_selection) == 0:
			results = pool.map(multi_find_urls,[l for l in hlp.chunks(urls,int(len(urls)/num_cores)+1)])
			for result in results:
				first_degree_urls.update(result)
		else:
			urls = find_urls(selection=url_selection,urls=urls)
			first_degree_urls.update(urls)
	first_degree_urls.update(direct_urls)
	first_degree_urls.update(set([n for n,d in binet.g.nodes(data=True) if d["node_type"]=="url"]))
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
			binet = add_data_to_net(fdocs,binet,has_been_queried,"actor",extra="actor")
	del results
	gc.collect()
	print ("Nodes in net before shave {0} for second degree actors".format(len(list(binet.g.nodes()))))
	binet.g = binet.filter_by_degrees(binet.g,degree=2,skip_nodes=has_been_queried,preserve_skip_node_edges=False,extra="actor")
	print ("Nodes in net after shave {0} for second degree actors".format(len(list(binet.g.nodes()))))
	second_degree_actors = set([d["extra"] for n,d in binet.g.nodes(data=True) if d["node_type"]=="actor" and d["extra"] not in has_been_queried])
	print (len(second_degree_actors))

	print ("searching for second degree interconnections.")
	fucount = 0
	fchunks_size = batch_size
	for _factor_chunk in hlp.chunks(list(second_degree_actors),int(len(list(second_degree_actors))/fchunks_size)+1):
		second_degree_queries = []
		for _factor in _factor_chunk:
			fucount+=1
			second_degree_queries.append({"actor":_factor,"platform":{"$in":only_platforms}})
		results = pool.map(query_multi,[("url_bi_network",l) for l in hlp.chunks(second_degree_queries,int(len(second_degree_queries)/num_cores)+1)])
		for result in results:
			for fdocs in pool.map(filter_docs,[(l,"url",between_dates) for l in hlp.chunks(list(result),int(len(list(result))/num_cores)+1)]):
				binet = add_data_to_net(fdocs,binet,has_been_queried_first_degree,"url",extra="actor")
	del results
	gc.collect()
	print ("Nodes in net before shave {0} for second degree urls".format(len(list(binet.g.nodes()))))
	binet.g = binet.filter_by_degrees(binet.g,degree=2,skip_nodes=has_been_queried,preserve_skip_node_edges=False,extra="actor")
	print ("Nodes in net after shave {0} for second degree urls".format(len(list(binet.g.nodes()))))
	if actor_domains is None:
		binet = add_domains_as_actors(binet,[])
	elif len(actor_domains) > 0:
		binet = add_domains_as_actors(binet,actor_domains)
	else:
		pass
	print ("Nodes in net after domains added as actors {0}".format(len(list(binet.g.nodes()))))
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

def stochastic_update_scheme_values(cat,batch_size=None,num_cores=12):

	mdb = MongoSpread()
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

def bi_to_uni_net(data,node0="actor",node1="url",output="net",num_cores=12,batch_size=1):

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
		node_key_map = {}
		node_counter = 0
		for n,nd in list(data.nodes(data=True)):
			if nd["node_type"]==node1:
				net_data[n]=[]
				for on,e,ed in list(data.edges(n,data=True)):
					if e not in node_key_map:
						node_key_map[e]=node_counter
						node_counter+=1
					net_data[n].append([node_key_map[e],ed["weight"]])
	del data
	gc.collect()
	start_time = time.time()
	pool = Pool(num_cores)
	if batch_size is not None:
		edge_dict = {}
		if batch_size > 10:
			net_data = list(hlp.chunks_optimized(net_data,n_chunks=num_cores*batch_size,semi_opti=True))
		else:
			net_data = list(hlp.chunks_optimized(net_data,n_chunks=num_cores*batch_size,semi_opti=False))
		for i,net_data_batch in enumerate(hlp.chunks(net_data,num_cores)):
			results = pool.map(bi_to_uni,net_data_batch)
			print("--- %s seconds --- for num cores {0} to reproject data for batch {1}".format(num_cores,i) % (time.time() - start_time))
			start_time = time.time()
			for result in results:
				for k_tup, w in result.items():
					edge_tup = (k_tup[0],k_tup[1])
					if edge_tup in edge_dict:
						edge_dict[edge_tup]+=w
					else:
						edge_dict[edge_tup]=w
			print("--- %s seconds --- to tabulate data for batch {0}".format(i) % (time.time() - start_time))
			start_time = time.time()
		del results
		gc.collect()
		del net_data
		gc.collect()
	if output == "net":
		g = nx.Graph()
		for result in results:
			for k_tup, w in result.items():
				g = add_node_and_edges(g,k_tup[0],k_tup[1],w)
		print (len(g.nodes()))
		print (len(g.edges()))
		print("--- %s seconds --- to build network" % (time.time() - start_time))
		return g,node_key_map
	elif output == "pandas":
		edge_list = []
		cols = ["src","trg","weight"]
		for k_tup, w in edge_dict.items():
			edge_list.append([k_tup[0],k_tup[1],float(w)])
		edge_df = pd.DataFrame(edge_list, columns=cols)
		print("--- %s seconds --- to build network" % (time.time() - start_time))
		return edge_df, node_key_map
	else:
		print("--- %s seconds --- to build network" % (time.time() - start_time))
		return edge_dict,node_key_map

def update_actor_data(actors=[],extra_data=None,extra_data_key="actor"):

	mdb = MongoSpread()
	num_cores = 12
	pool = Pool(num_cores)
	chunked_actors = hlp.chunks(actors,int(len(actors)/num_cores)+1)
	results = pool.map(get_agg_actor_metrics,chunked_actors)
	all_actor_data = []
	for result in results:
		for uactor,doc in result.items():
			if extra_data is not None and doc[extra_data_key] in extra_data:
				doc.update(extra_data[doc[extra_data_key]])
			all_actor_data.append(doc)

	all_actor_data = pd.DataFrame(all_actor_data)
	return all_actor_data

def update_cleaned_urls():

	import urllib.parse

	def clean_the_clean(url):
		new_url = str(url)
		if "facebook" in new_url and "login" in new_url and "next=" in new_url:
			new_url = urllib.parse.unquote(new_url.split("next=")[-1])
		if "instagram" in new_url and "login" in new_url and "next=" in new_url:
			new_url = "instagram.com" + new_url.split("next=")[-1]
		return new_url

	print ("updating cleaned urls")
	mdb = MongoSpread()
	e_cleaned = []
	e_non_cleaned = []
	identical = []
	clean_version = {}
	rest = []
	no_dups = set([])
	count = 0
	"""for doc in mdb.database["clean_url"].aggregate([{"$group":{"_id":"$clean_url","count":{"$sum":1}}},{"$sort": { "count": -1 }},{"$limit":10000000000}]):
		if "facebook" in str(doc["_id"]) and "login" in str(doc["_id"]):
			print (clean_the_clean(str(doc["_id"])))
		if "instagram" in str(doc["_id"]) and "login" in str(doc["_id"]):
			print (clean_the_clean(str(doc["_id"])))
	sys.exit()"""

	url_shorts = set(list(set(list(pd.read_csv("/work/JakobBækKristensen#1091/alterpublics/projects/full_test"+"/"+"url_shorteners.csv")["domain"]))))
	for url_cl in mdb.database["clean_url"].find().limit(102222000):
		count+=1
		if LinkCleaner().extract_special_url(url_cl["url"]) in url_shorts:
			url_cl["url"]=LinkCleaner().remove_url_prefix(url_cl["url"])
			if url_cl["clean_url"] is not None:
				if url_cl["url"] not in no_dups:
					u_cl = mdb.database["url_bi_network"].find_one({"url":url_cl["clean_url"]})
					if u_cl is not None and "cleaned" in u_cl and u_cl["cleaned"]==True:
						e_cleaned.append(u_cl["url"])
					if u_cl is None:
						u = mdb.database["url_bi_network"].find_one({"url":url_cl["url"]})
						if u is not None:
							cleaned_v = url_cl["clean_url"]
							cleaned_v = clean_the_clean(cleaned_v)
							cleaned_v = LinkCleaner().single_standardize_url(cleaned_v)
							cleaned_v = LinkCleaner().single_standardize_url(cleaned_v)
							cleaned_v = LinkCleaner().single_standardize_url(cleaned_v)
							if u["url"] != cleaned_v:
								e_non_cleaned.append(u["url"])
								clean_version[u["url"]]=cleaned_v
							else:
								identical.append(u["url"])
						else:
							rest.append(url_cl["url"])
					no_dups.add(url_cl["url"])
		if count % 1000 == 0:
			print (count)


	for uc in e_non_cleaned:
		print (str(uc)+"  -  "+str(clean_version[uc]))
	print (len(e_cleaned))
	print (len(e_non_cleaned))
	print (len(identical))
	print (len(rest))

	print ("writing cleaned urls to database")
	bulks = []
	for uc in e_non_cleaned:
		bulks.append(UpdateMany({"url":uc},
									{'$set': {"url":str(clean_version[uc]),"domain":LinkCleaner().extract_special_url(str(clean_version[uc])),"cleaned":True,"before_clean":uc}}))
	mdb.database["url_bi_network"].bulk_write(bulks)
	print ("done writing to database!")

def get_articles(url):

	print ()
	print (url)
	print ()
	try:
		article = Article(url)
		article.download()
		article.parse()
		print (article.text)
		print ()
		print ()
		print ()
		doc = {"url":url,"text":str(article.text),"succes":True}
	except Exception as e:
		print ()
		print (e)
		print ()
		doc = {"url":url,"text":str(e),"succes":False}

	return doc

def update_url_texts(num_cores=2):

	from newspaper import Config as NConfig
	user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
	config = Config()
	config.browser_user_agent = user_agent
	mdb = MongoSpread()
	project = "rt_sputnik"
	prev_urls = set([d["url"] for d in mdb.database["url_texts"].find({},{"url"})])
	pri_domains = ['de.sputniknews.com', 'sputniknews.com',"de.rt.com","rt.com"]
	domains = list(set([LinkCleaner().strip_backslash( LinkCleaner().remove_url_prefix(d["Website"])) for d in mdb.database["actor"].find({"org_project_title":"rt_sputnik","Iteration":0})]))
	domains = [d for d in domains if d is not None and d not in pri_domains]
	pri_urls = list(set([d["url"] for d in mdb.database["url_bi_network"].find({"domain":{"$in":pri_domains}})]))
	urls = list(set([d["url"] for d in mdb.database["url_bi_network"].find({"domain":{"$in":domains}})]))
	print (len(domains))
	print (len(pri_domains))
	print (len(urls))
	print (len(pri_urls))
	"""pri_urls = ['https://de.rt.com/live/114494-live-pressekonferenz-von-sebastian-kurz', 'https://rt.com/newsline/497774-eu-ministers-belarus-sanctions', 'https://rt.com/op-ed/525478-china-cnn-wolf-warrior-diplomacy', 'https://rt.com/news/449550-france-putin-children-disinfo', 'https://sputniknews.com/asia/202001291078167767-significant-breakthrough-australian-scientists-succeeded-to-recreate-coronavirus-outside-china', 'https://sputniknews.com/20211014/full-scale-us-thai-military-drills-to-be-conducted-next-year-us-military-says-1089917386.html', 'https://sputniknews.com/radio-political-misfits/202104201082674671-new-wave-of-protests-after-more-police-killings-the-long-fight-at-amazon', 'https://rt.com/rtmobile/news/latest/451020/html', 'https://rt.com/rtmobile/news/latest/461646/html', 'https://de.rt.com/2bod', 'https://de.rt.com/inland/115746-streit-um-ausgangssperren-lauterbach-kritisiert', 'https://rt.com/usa/473631-san-diego-shooting-children', 'https://sputniknews.com/russia/201907171076276602-russian-universities', 'https://sputniknews.com/videoclub/202108271083718277-mummy-im-your-little-helper-golden-retriever-puppy-steals-cleaning-cloths', 'https://sputniknews.com/photo/202004221079048263-lenin-monuments-around-the-world', 'https://sputniknews.com/20220101/shedding-of-the-soul-new-book-reveals-aviciis-journal-entries-on-inner-demons-health-issues---1091954582.html', 'https://de.rt.com/europa/130923-junkers-im-sturzflug-wie-sechsjahrige-stalingrader-schlacht-okkupation-ueberlebte', 'https://de.rt.com/inland/111882-erster-wahlgang-merz-und-laschet-vorne/Laschet', 'https://de.sputniknews.com/politik/20190411324661444-wikileaks-assange-london-festnahme', 'https://rt.com/news/429282-trump-kim-singapore-impersonation', 'https://de.rt.com/meinung/119385-studie-die-meisten-positiv-getesteten-sind-nicht-infektioes/amp/?__twitter_impression=true', 'https://rt.com/news/548727-canada-covid-mandates-saskatchewan/?utm_referrer=https%3A%2F%2Fzen.yandex.com%2F%3Ffromzen%3Dabro', 'https://sputniknews.com/latam/202104101082591267-argentine-government-to-launch-legal-action-against-ex-president-over-imf-loan', 'https://sputniknews.com/middleeast/201909201076845495-militants-block-humanitarian-corridor-to-idlibs-abu-al-duhur-checkpoint--syrian-army', 'https://sputniknews.com/military/201904241074410203-norway-frigate-losses', 'https://sputniknews.com/news/20160512/1039528652/911-saudi-obama-terror-fbi.html', 'https://sputniknews.com/amp/radio_the_critical_hour/202009051080370875-covid-19-model-predicts-us-death-toll-will-surpass-410000-by-january-1/?__twitter_impression=true', 'https://rt.com/news/513795-hong-kong-ambush-lockdowns-covid', 'https://sputniknews.com/viral/201909271076907025-following-in-ivankas-footsteps-jennifer-lopez-attends-public-event-bra-less', 'https://sputniknews.com/20211210/ufc-269-closes-out-2021-pay-per-view-schedule-after-record-year-of-profits-for-company-1091426972.html', 'https://rt.com/usa/529309-georgia-highway-bridge-truck', 'https://sputniknews.com/20220308/ex-ukrainian-president-yanukovich-urges-zelensky-to-stop-bloodshed-at-any-price-reach-peace-deal-1093678034.html', 'https://sputniknews.com/photo/201903251073512085-kazakhstan-beauty-nowruz', 'https://rt.com/pop-culture/540942-wheel-of-time-woke', 'https://sputniknews.com/middleeast/201907281076382657-intra-afghan-talks-usa-deal-taliban', 'https://de.sputniknews.com/panorama/20200407326818063-klinische-tests-praeparate-gegen-coronavirus-russland', 'https://sputniknews.com/us/202105051082811083-gop-heavyweights-slam-big-tech-over-trump-facebook-ban-fueling-2024-campaign-rumors', 'https://rt.com/business/449508-bear-market-risk-shiller', 'https://de.rt.com/international/116909-gesprache-zur-rettung-atomdeals-es', 'https://sputniknews.com/news/201905141074980145-US-Law-Enforcement-Officials-Raid-Venezuelan-Embassy-Washington-DC']"""
	temp_urls = []
	for url_p in [pri_urls,urls]:
		for url in url_p:
			if url not in prev_urls:
				temp_urls.append(url)
				if num_cores < 2:
					results = [get_articles(temp_urls[0])]
					mdb.insert_many(mdb.database["url_texts"],list(results))
					temp_urls = []
				if len(temp_urls) == num_cores:
					pool = Pool(num_cores)
					results = pool.map(get_articles,temp_urls)
					pool.close()
					mdb.insert_many(mdb.database["url_texts"],list(results))
					temp_urls = []

def batch_domain_per_actor():

	batch_path = Config().PROJECT_PATH+"/full_test/domain_per_actor"

	"""df = pl.read_csv(batch_path+".csv")
	print (len(df))
	df = df.filter((pl.col("t_freq")>9))
	df.write_csv(batch_path+"_min10.csv")
	print (len(df))
	sys.exit()"""

	mdb = MongoSpread()
	all_docs = {"domain":[],"d_freq":[],"t_freq":[],"idf":[],"tfidf":[],"logn_tfidf":[]}
	N_DOCS = 10259859
	query = mdb.database["url_bi_network"].aggregate([{"$group":{"_id":"$domain", "uniqueValuesOfB":{"$addToSet":"$actor_platform"}, "totalLengthOfC":{"$sum":{"$size":"$message_ids"}}}},{"$project":{"countOfUniqueB":{"$size":"$uniqueValuesOfB"}, "totalLengthOfC":1}}])
	q_count = 0
	for doc in query:
		if doc is not None and doc["_id"] is not None:
			all_docs["domain"].append(doc["_id"])
			all_docs["d_freq"].append(doc["countOfUniqueB"])
			all_docs["t_freq"].append(doc["totalLengthOfC"])
			idf = np.log(N_DOCS/(1+doc["countOfUniqueB"]))+1
			all_docs["idf"].append(idf)
			tfidf = doc["totalLengthOfC"]*idf
			logn_tfidf = np.log(doc["totalLengthOfC"]+1)*idf
			all_docs["tfidf"].append(tfidf)
			all_docs["logn_tfidf"].append(logn_tfidf)
		q_count+=1
		if q_count % 1000 == 0:
			print (q_count)
	df = pl.from_dict(all_docs)
	df = df.filter((pl.col("t_freq")>9))
	df.write_csv(batch_path+"_min10.csv")

if __name__ == "__main__":
	#update_actor_message()
	#update_agg_actor_metrics(skip_existing=True)
	#update_url_texts(num_cores=1)
	#update_cleaned_urls()
	batch_domain_per_actor()
	sys.exit()
