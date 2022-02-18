# MONGODB COLLECTIONS

"""
platform
pull
query
actor
url
post
actor_website_url
actor_url
query_url
url_post
actor_post
query_post

"""

# MONGODB MANDATORY FIELDS

"""
ALL COLLECTIONS:

- _id
- insert_time
- update_time

"""

"""
platform

- platform_name
- platform_api_url
- available

pull

- input
- input_type
- method
- last_run
- last_run_issue
- attempts ~isList
	- issue

queries

- query
- input_file
- title

actors

- actor
- input_file
- title

urls

- url
- input_file
- title

posts

- message_id

"""

import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from spreadAnalysis.utils.link_utils import LinkCleaner
from pymongo import InsertOne, DeleteOne, ReplaceOne, UpdateOne
from spreadAnalysis.persistence.schemas import Spread
from spreadAnalysis.utils import helpers as hlp
from multiprocessing import Pool, Manager
import networkx as nx
import time
import sys
import random

class MongoDatabase:

	def __init__(self):

		self.client = MongoClient('mongodb://localhost:27017/')
		self.database = self.client["spreadAnalysis"]

	def close(self):

		self.client.close()

	def unpack_docs(self,cursor):

		return [doc for doc in cursor]

	def get_ids_from_db(self,db,use_col=None):

		ids = {d[use_col]:d["_id"] for d in db.find({},{ "_id": 1, use_col: 1})}
		return ids

	def get_keys_from_db(self,db,use_col=None):

		ks = set([d[use_col] for d in db.find({},{use_col: 1})])
		return ks

	def get_keys_from_nested(self,db,top_col,nest_col):

		all_keys = set([])
		comb_key = f"{top_col}.{nest_col}"
		for d in db.find({},{comb_key: 1}):
			for dd in d[top_col]:
				all_keys.add(dd[nest_col])

		ks = set([d[use_col] for d in db.find({},{use_col: 1})])
		return ks

	def get_key_pairs_from_db(self,db,col1,col2):

		kps = {}
		for d in db.find({},{col1: 1, col2: 1}):
			if not d[col1] in kps: kps[d[col1]]=set([])
			kps[d[col1]].add(d[col2])
		return kps

	def get_data_from_db(self,db):

		data = self.unpack_docs(db.find({}))
		return data

	def delete_with_key_val(self,db,key,val):

		if isinstance(key,tuple):
			db.delete_one({key[0]:val[0],key[1]:val[1]})
		else:
			db.delete_one({key:val})

	def test_update(self):

		db = self.database["post"]
		#db.bulk_write([InsertOne({'_id': 111, "foo":"bar"})])
		db.bulk_write([UpdateOne({'_id': 111}, {'$set': {'foo': 'sup'}})])

	def update_many(self,db,docs,key_col):

		bulks = []
		if len(docs) > 0:
			for doc in docs:
				doc["updated_at"]=datetime.now()
				if isinstance(key_col,tuple):
					bulks.append(UpdateOne({key_col[0]:doc[key_col[0]],key_col[1]: doc[key_col[1]]},
								{'$set': doc}))
				else:
					bulks.append(UpdateOne({key_col:doc[key_col]},
								{'$set': doc}))
			db.bulk_write(bulks)

	def insert_many(self,db,docs):

		bulks = []
		if len(docs) > 0:
			for doc in docs:
				doc["inserted_at"]=datetime.now()
				bulks.append(InsertOne(doc))
			db.bulk_write(bulks)

	def insert_one(self,db,doc):

		doc["inserted_at"]=datetime.now()
		db.insert_one(doc)

	def write_many(self,db,docs,key_col,sub_mapping=None,only_insert=False):

		bulks = []
		seen_ids = set([])
		if len(docs) > 0:
			if isinstance(key_col,tuple):
				for doc in docs:
					if (doc[key_col[0]],doc[key_col[1]]) in seen_ids: continue
					if db.count_documents({ key_col[0]: doc[key_col[0]],key_col[1]: doc[key_col[1]] }, limit = 1) != 0:
						if only_insert: continue
						doc["updated_at"]=datetime.now()
						if sub_mapping is not None:
							old_doc = db.find_one({ key_col[0]: doc[key_col[0]],key_col[1]: doc[key_col[1]] })
							new_mapping = set(doc[sub_mapping])
							new_mapping.update(set(old_doc[sub_mapping]))
							doc[sub_mapping]=list(new_mapping)
						bulks.append(UpdateOne({key_col[0]:doc[key_col[0]],key_col[1]: doc[key_col[1]]},
									{'$set': doc}))
					else:
						doc["inserted_at"]=datetime.now()
						bulks.append(InsertOne(doc))
					seen_ids.add((doc[key_col[0]],doc[key_col[1]]))
			else:
				for doc in docs:
					if doc[key_col] in seen_ids: continue
					if db.count_documents({ key_col: doc[key_col] }, limit = 1) != 0:
						if only_insert: continue
						doc["updated_at"]=datetime.now()
						if sub_mapping is not None:
							old_doc = db.find_one({key_col:doc[key_col]})
							new_mapping = set(doc[sub_mapping])
							new_mapping.update(set(old_doc[sub_mapping]))
							doc[sub_mapping]=list(new_mapping)
						bulks.append(UpdateOne({key_col:doc[key_col] },
									{'$set': doc}))
					else:
						doc["inserted_at"]=datetime.now()
						bulks.append(InsertOne(doc))
					seen_ids.add(doc[key_col])
			if len(bulks) > 0:
				db.bulk_write(bulks)

	def insert_into_nested(self,db,doc,nest_keys):

		doc["updated_at"]=datetime.now()
		if isinstance(nest_keys[0],tuple):
			db.update(
				{ nest_keys[0][0]: nest_keys[0][1], nest_keys[1][0]: nest_keys[1][1]},
				{ "$push": {"attempts":doc}})
		else:
			db.update(
				{ nest_keys[0]: nest_keys[1]},
				{ "$push": {"attempts":doc}})


class MongoSpread(MongoDatabase):

	custom_file_schema = {  "actor":("Actors.xlsx","Actor"),
							"query":("Queries.xlsx","Query"),
							"url":("Urls.xlsx","Url")  }
	del custom_file_schema["query"]

	avlb_platforms = {"Facebook Page":"crowdtangle",
						"Facebook Group":"crowdtangle",
						"Twitter":"twitter2",
						"Instagram":"crowdtangle",
						"Reddit":"reddit",
						"Youtube":"youtube",
						"Tiktok":"tiktok",
						"Vkontakte":"vkontakte",
						"CrowdtangleApp":"crowdtangle_app",
						"Google":"google",
						"Majestic":"majestic",
						"Telegram":"telegram",
						"Gab":"gab"}

	avlb_endpoints = {  "Facebook Page":["actor","url","domain","query"],
						"Facebook Group":["actor","url","domain","query"],
						"Twitter":["actor","url","domain","query"],
						"Instagram":["actor","url","domain","query"],
						"Reddit":["url","domain","query"],
						"Youtube":["actor"],
						"Tiktok":["actor","query"],
						"Vkontakte":["actor","url","domain","query"],
						"CrowdtangleApp":["url"],
						"Google":["url","query"],
						"Majestic":["url","domain"],
						"Telegram":["actor"],
						"Gab":["actor"] }

	table_key_cols = {  "post":"message_id",
						"pull":"input",
						"url":"Url",
						"actor":"Actor",
						"url_post":("input","message_id"),
						"actor_post":("input","message_id"),
						"domain_url":("input","url"),
						"alias":"actor",
						"url_bi_network":("url","actor_platform"),
						"url_bi_network2":("url","actor_platform"),
						"url_bi_network_coded":"uentity"  }

	table_idx_cols = {   }

	def __init__(self,custom_file_schema=None):
		MongoDatabase.__init__(self)

		if custom_file_schema is not None:
			self.custom_file_schema = custom_file_schema
		self.create_indexes()

	def create_indexes(self):

		for table,key_col in self.table_key_cols.items():
			if isinstance(key_col,str):
				self.database[table].create_index([ (key_col, -1) ])
			elif isinstance(key_col,tuple):
				self.database[table].create_index([ (key_col[0], -1), (key_col[1], -1)])

	def get_custom_file_as_df(self,file_path):

		if ".xlsx" in file_path:
			df = pd.read_excel(file_path)
		elif ".csv" in file_path:
			df = pd.read_csv(file_path)
		df = df.where(pd.notnull(df), None)
		df = df.replace({np.nan: None})
		return df

	def delete_pulls(self,title=None,methods=None,iterations=[0,1]):

		bulks = []
		url_inputs = self.database["url"].find({ "Iteration": { "$in": iterations },
												"org_project_title":{"$eq":title},
												"Domain":0})
		domain_inputs = self.database["url"].find({ "Iteration": { "$in": iterations },
												"org_project_title":{"$eq":title},
												"Domain":1})
		actor_inputs = self.database["actor"].find({ "Iteration": { "$in": iterations },
												"org_project_title":{"$eq":title}})
		for endpoint, inputs in [("domain",domain_inputs),("url",url_inputs),("actor",actor_inputs)][:1]:
			if endpoint == "domain": call_endpoint = "url"
			else: call_endpoint = endpoint
			for input_doc in inputs:
				if methods is None:
					bulks.append(DeleteOne({"input_type":endpoint,
					"input":input_doc[self.custom_file_schema[call_endpoint][1]]}))
				else:
					for method in methods:
						bulks.append(DeleteOne({"input_type":endpoint,
						"input":input_doc[self.custom_file_schema[call_endpoint][1]],
						"method":method}))
		self.database["pull"].bulk_write(bulks)
		print (len(bulks))

	def get_aliases(self):

		aliases = {}
		for d in self.get_data_from_db(self.database["alias"]):
			if not d["actor"] in aliases: aliases[d["actor"]]=[]
			aliases[d["actor"]].append(d)
		return aliases

	def get_actor_aliases(self,platform_sorted=True):

		aliases = {}
		if platform_sorted:
			for d in self.get_data_from_db(self.database["alias"]):
				if not d["platform"] in aliases: aliases[d["platform"]]={}
				if not d["alias"] in aliases[d["platform"]]:
					aliases[d["platform"]][d["alias"]]=[]
				aliases[d["platform"]][d["alias"]].append(d["actor"])
		else:
			for d in self.get_data_from_db(self.database["alias"]):
				if not d["alias"] in aliases: aliases[d["alias"]]={}
				if not d["platform"] in aliases[d["alias"]]:
					aliases[d["alias"]][d["platform"]]=d["actor"]
				else:
					if d["actor"] != d["alias"] and str(d["actor"]).strip() != str(d["alias"]).strip():
						aliases[d["alias"]][d["platform"]]=d["actor"]
		return aliases

	def update_custom_data(self,custom_path,custom_title=None,with_del=True,files_key_cols=None):

		# !!! Function is missing a check for unique actors in Actor column.
		# Import requires a unique Actor designation in the column
		if files_key_cols is None:
			files_key_cols = self.custom_file_schema

		if custom_title is None:
			custom_title = custom_path.split("/")[-1]

		for database, files in files_key_cols.items():
			db = self.database[database]
			file_path = files[0]
			key_col = files[1]
			prev_ids = self.get_ids_from_db(db,key_col)
			prev_data = self.get_data_from_db(db)
			actor_df = self.get_custom_file_as_df(custom_path+"/"+file_path)
			actor_df["org_project_title"]=custom_title
			actor_df["org_project_path"]=custom_path
			actor_df["Iteration"]=pd.to_numeric(actor_df["Iteration"],downcast='integer')
			if "Domain" in actor_df.columns:
				actor_df["Domain"]=pd.to_numeric(actor_df["Domain"],downcast='integer')
			cols = actor_df.columns
			actor_docs = [{col:row[col] for col in cols} for i,row in actor_df.iterrows()]

			inserts, updates = [d for d in actor_docs if d[key_col] not in prev_ids],\
				[d for d in actor_docs if d[key_col] in prev_ids]
			self.insert_many(db,inserts)
			self.update_many(db,updates,key_col)

			if with_del:
				new_actor_keys = set([d[key_col] for d in actor_docs])
				for k,_id in prev_ids.items():
					if k not in new_actor_keys:
						self.delete_with_key_val(db,(key_col,"org_project_title"),(k,custom_title))

	def update_platform_info(self,with_test=False):

		db = self.database["platform"]
		db.drop()
		prev_ids = self.get_ids_from_db(db,"platform_dest")
		not_working_platforms = set(["CrowdtangleApp","Google"])

		inserts, updates = [], []
		for platform_dest, source in self.avlb_platforms.items():
			platform_doc = {}
			platform_doc["working"]=True
			if platform_dest in not_working_platforms: platform_doc["working"]=False
			platform_doc["platform_dest"]=platform_dest
			platform_doc["platform_source"]=source
			for endp in ["url","actor","domain","query"]:
				if endp in self.avlb_endpoints[platform_dest]:
					platform_doc[endp+"_endpoint"]=True
				else:
					platform_doc[endp+"_endpoint"]=False
			if platform_dest not in prev_ids:
				inserts.append(platform_doc)

		self.insert_many(db,inserts)

	def update_aliases(self,custom_path,no_check=False):

		prev_aliases = self.get_key_pairs_from_db(self.database["alias"],"actor","platform")
		actor_info = self.get_data_from_db(self.database["actor"])

		inserts, updates = [], []
		for doc in actor_info:
			for platform in list(self.avlb_platforms.keys()):
				if platform in doc and doc[platform] is not None:
					try:
						alias = LinkCleaner().extract_username(doc[platform],never_none=no_check)
					except:
						print (doc[platform])
					if alias is not None and len(alias) > 1:
						alias_doc = {"actor":str(doc["Actor"]),"platform":platform,"alias":alias}
						if doc["Actor"] in prev_aliases and platform in prev_aliases[doc["Actor"]]:
							updates.append(alias_doc)
						else:
							inserts.append(alias_doc)
							prev_aliases[doc["Actor"]]=platform
		self.insert_many(self.database["alias"],inserts)
		self.update_many(self.database["alias"],updates,("actor","platform"))

	def update_coding_schemes(self,custom_path,scheme_title,with_del=True):

		fixed_cols = ["uentity","entity_type","scheme_title"]
		coding_db_key = "uentity"
		db = self.database["coding"]
		prev_entity_keys = set([d[coding_db_key] for d in db.find({"scheme_title":scheme_title})])
		duplicate_cats = set([])
		for doc in db.find({'scheme_title': {"$ne" : scheme_title}}):
			for k in dict(doc).keys():
				duplicate_cats.add(k)
		file_path = custom_path+"/Coding.xlsx"
		coding_df = self.get_custom_file_as_df(file_path)
		cols = coding_df.columns
		for col in cols:
			if col not in fixed_cols and col in duplicate_cats:
				print ("category exists for other scheme. Exiting.")
				sys.exit()
		coding_docs = [{col:row[col] for col in cols} for i,row in coding_df.iterrows()]

		if with_del:
			new_entity_keys = set([d[coding_db_key] for d in coding_docs])
			for k in prev_entity_keys:
				if k not in new_entity_keys:
					self.delete_with_key_val(db,(coding_db_key,"scheme_title"),(k,scheme_title))
		self.write_many(db,coding_docs,(coding_db_key,"scheme_title"),sub_mapping=None)

	def update_url_bi_network(self,new=False):

		net_db = self.database["url_bi_network"]
		try:
			net_db.drop_index('actor_-1')
		except:
			pass
		if new:
			net_db.drop()
			net_db = self.database["url_bi_network"]
			self.create_indexes()
		max_net_date = list(net_db.find().sort("inserted_at",-1).limit(1))
		if max_net_date is None or len(max_net_date) < 1:
			max_net_date = datetime(2000,1,1)
		else:
			if "updated_at" in max_net_date[0]:
				max_net_date = max_net_date[0]["updated_at"]
			elif "inserted_at" in max_net_date[0]:
				max_net_date = max_net_date[0]["inserted_at"]
			max_net_date = max_net_date-timedelta(days=1)

		aliases = self.get_aliases()
		actor_aliases = self.get_actor_aliases(platform_sorted=False)
		url_post_db = self.database["url_post"]
		post_db = self.database["post"]
		cur = url_post_db.find({"$or":[ {"updated_at": {"$gt": max_net_date}}, {"inserted_at": {"$gt": max_net_date}}]}).sort("input",-1)
		#print (url_post_db.count_documents({"$or":[ {"updated_at": {"$gt": max_net_date}}, {"inserted_at": {"$gt": max_net_date}}]}))
		next_url_post = True
		batch_insert = {}
		seen_ids = set([])
		count = 0
		while next_url_post is not None:
			count += 1
			next_url_post = next(cur, None)
			if next_url_post is not None:
				url = next_url_post["input"]
				seen_ids.add(next_url_post["message_id"])
				post = post_db.find_one({"message_id":next_url_post["message_id"]})
				url = Spread._get_message_link(data=post,method=post["method"])
				if url is not None:
					url = LinkCleaner().single_clean_url(url)
					url = LinkCleaner().sanitize_url_prefix(url)
					actor_username = Spread._get_actor_username(data=post,method=post["method"])
					actor_id = Spread._get_actor_id(data=post,method=post["method"])
					platform = Spread._get_platform(data=post,method=post["method"])
					platform_type = Spread._get_platform_type(data=post,method=post["method"])
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
					if (url,actor) not in batch_insert:
						batch_insert[(url,actor)]={"url":url,"actor_username":actor_username,
											"actor":actor,"actor_label":actor_label,"platform":platform,
											"message_ids":[],"actor_platform":actor_platform,"domain":domain}
					batch_insert[(url,actor)]["message_ids"].append(post["message_id"])
					if len(batch_insert) >= 10000:
						self.write_many(net_db,list(batch_insert.values()),key_col=("url","actor_platform"),sub_mapping="message_ids")
						batch_insert = {}
					if count % 1000 == 0:
						print ("url loop " + str(count))
		if len(batch_insert) > 0:
			self.write_many(net_db,list(batch_insert.values()),key_col=("url","actor_platform"),sub_mapping="message_ids")

		actor_post_db = self.database["actor_post"]
		cur = actor_post_db.find({"$or":[ {"updated_at": {"$gt": max_net_date}}, {"inserted_at": {"$gt": max_net_date}}]}).sort("input",-1)
		next_actor_post = True
		batch_insert = {}
		count = 0
		while next_actor_post is not None:
			count += 1
			next_actor_post = next(cur, None)
			if next_actor_post is not None:
				message_id = next_actor_post["message_id"]
				actor = next_actor_post["input"]
				if not message_id in seen_ids:
					post = post_db.find_one({"message_id":message_id})
					platform = Spread._get_platform(data=post,method=post["method"])
					platform_type = Spread._get_platform_type(data=post,method=post["method"])
					actor_username = Spread._get_actor_username(data=post,method=post["method"])
					actor_platform = str(actor)+"_"+str(platform_type)
					actor_label = str(Spread._get_actor_name(data=post,method=post["method"]))+" ({0})".format(str(platform_type))
					url = Spread._get_message_link(data=post,method=post["method"])
					domain = Spread._get_message_link_domain(data=post,method=post["method"])
					if url is not None:
						url = LinkCleaner().single_clean_url(url)
						url = LinkCleaner().sanitize_url_prefix(url)
						if (url,actor) not in batch_insert:
							batch_insert[(url,actor)]={"url":url,"actor_username":actor_username,
												"actor":actor,"actor_label":actor_label,"platform":platform,
												"message_ids":[],"actor_platform":actor_platform,"domain":domain}
						batch_insert[(url,actor)]["message_ids"].append(post["message_id"])
						if len(batch_insert) >= 10000:
							self.write_many(net_db,list(batch_insert.values()),key_col=("url","actor_platform"),sub_mapping="message_ids")
							batch_insert = {}
						if count % 1000 == 0:
							print ("actor loop " + str(count))
		if len(batch_insert) > 0:
			self.write_many(net_db,list(batch_insert.values()),key_col=("url","actor_platform"),sub_mapping="message_ids")
		net_db.create_index([ ("actor", -1) ])

	def bi_to_uni_net_OLD(self,node1,node2):

		net_db = self.database["url_bi_network"]
		cur = net_db.find({}).sort("url",-1).limit(1000000)
		url_actor = True
		net_data = {}
		while url_actor is not None:
			url_actor = next(cur, None)
			if url_actor is not None:
				if url_actor["actor"] is not None and url_actor["url"] is not None:
					if url_actor["url"] not in net_data:
						net_data[url_actor["url"]]=[]
					net_data[url_actor["url"]].append([url_actor["actor"],len(url_actor["message_ids"])])
		start_time = time.time()
		num_cores = 6
		N = num_cores
		S = int(len(net_data)/N)
		net_data = list(hlp.chunks_optimized(net_data,n_chunks=num_cores))
		#net_data = [ net_data.iloc[i*S:(i+1)*S] for i in range(N) ]
		pool = Pool(num_cores)
		rep_data = {}
		results = pool.map(bi_to_uni,net_data)
		print("--- %s seconds --- for num cores {0} to reproject data".format(num_cores) % (time.time() - start_time))
		start_time = time.time()
		g = nx.Graph()
		for result in results:
			for k_tup, w in result.items():
				g = add_node_and_edges(g,k_tup[0],k_tup[1],w)
		print (len(g.nodes()))
		print (len(g.edges()))
		print("--- %s seconds --- to build network" % (time.time() - start_time))

def test():

	m = MongoSpread()
	#m.update_url_bi_network(new=False)
	m.bi_to_uni_net("url","actor")

if __name__ == '__main__':
	args = sys.argv
	if args[1] == "test":
		test()
