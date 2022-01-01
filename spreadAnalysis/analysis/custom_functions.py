import pandas as pd
from urllib.parse import urlparse
from collections import Counter
from operator import itemgetter
import numpy as np

def create_link_to_actor(meta_data,ct_data):

	def clean_url(url):
		new_url = str(url).strip().rstrip()
		if str(new_url)[-1] == "/": new_url = str(url)[:-1]
		new_url = str(new_url).split("/")[-1]
		if len(new_url) < 2: new_url = str(url).split("/")[-2]
		if "-" in new_url: new_url = new_url.split("-")[-1].strip()

		return new_url

	id_and_username_to_id = {}
	for i,row in ct_data.iterrows():
		id_and_username_to_id[str(row["User Name"]).strip()]=str(row["Facebook Id"]).strip()
		id_and_username_to_id[str(row["Facebook Id"]).strip()]=str(row["Facebook Id"]).strip()
	actor_to_meta = {}
	for i,row in meta_data.iterrows():
		actor = clean_url(row["Facebook"])
		if actor in id_and_username_to_id:
			actor = id_and_username_to_id[actor]
			actor_to_meta[actor]=row
		else:
			pass
			#print (actor)
	link_to_actor = {}
	for i,row in ct_data.iterrows():
		if str(row["Facebook Id"]).strip() in actor_to_meta:
			link_to_actor[str(row["Link"])]=actor_to_meta[str(row["Facebook Id"]).strip()]
		else:
			pass

	return link_to_actor

def get_domain_count_per_actor(df,url_col="url",actor_col="actor_id"):

	def _get_domain(url):
		return str(urlparse(url).netloc).replace("www.","")

	actor_domain_counts = {}
	for i,row in df.iterrows():
		if not row[actor_col] in actor_domain_counts:
			actor_domain_counts[row[actor_col]]=set([])
		actor_domain_counts[row[actor_col]].add(_get_domain(str(row[url_col])))
	actor_domain_counts = {a:int(len(d)) for a,d in actor_domain_counts.items()}

	return actor_domain_counts

def get_metric_per_actor(df,metric_col="clicks",actor_col="actor_id"):

	#df[metric_col] = df[metric_col].astype(float)
	grouped = df[[metric_col, actor_col]].groupby([actor_col]).agg(['sum'])
	return list(grouped.to_dict().values())[0]

def get_max_metric_per_actor(df,metric_col="clicks",actor_col="actor_id"):

	df[metric_col] = list([val if not isinstance(val,str) else None for val in list(df[metric_col])])
	#df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce').astype('Int64')
	#df.replace({pd.NaT: None})
	grouped = df[[metric_col, actor_col]].groupby([actor_col]).agg(['max'])
	return list(grouped.to_dict().values())[0]

def get_max_value_count_per_actor(df,metric_col="clicks",actor_col="actor_id"):

	grouped = df[[metric_col, actor_col]].groupby([actor_col]).agg(['value_counts'])
	actor_grouped = {}
	actor_max_counts = {}
	for i,row in pd.DataFrame(grouped).iterrows():
		actor = i[0]
		lang = i[1]
		count = int(row[0])
		if actor not in actor_grouped:
			actor_max_counts[actor]=count
			actor_grouped[actor]=lang
		elif count > actor_max_counts[actor]:
			actor_max_counts[actor]=count
			actor_grouped[actor]=lang

	return actor_grouped
