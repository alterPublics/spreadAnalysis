from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.utils.network_utils import NetworkUtils
from spreadAnalysis.persistence.schemas import Spread
from spreadAnalysis.scraper.scraper import Scraper
from difflib import SequenceMatcher
from spreadAnalysis.wrangling import wranglers as wrang
from spreadAnalysis.utils import helpers as hlp
from operator import itemgetter
import numpy as np
import time
import math
import sys
import pickle
from collections import defaultdict
import pandas as pd
import requests
from os import listdir
from os.path import isfile, join
from spreadAnalysis.persistence.mongo import MongoSpread
from spreadAnalysis.wrangling.viewers import export_actor_to_pandas
from spreadAnalysis.export.export_exploration import get_actor_net_exploration
from spreadAnalysis.export.export_exploration import get_actor_net_cat_dist, get_post_from_actor_and_link
import scipy.spatial
import networkx as nx
from spreadAnalysis.export.export import *
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from spreadAnalysis.utils.network_utils import NetworkUtils,disparity_filter,disparity_filter_alpha_cut
from spreadAnalysis.utils.network_utils import disparity_filter_multi,disparity_filter_alpha_cut_multi


def explore_domain_profiles(main_path,com_var="inner_com"):

	def tf_idf(row,idf_vecs,t_to_idx):

		tf_idf_score = (np.log(row["n_shared"]+1)*np.log10(row["n_actors"]+1))*idf_vecs.idf_[t_to_idx[row["domain"]]]
		return tf_idf_score

	def idf(row,idf_vecs,t_to_idx):

		return idf_vecs.idf_[t_to_idx[row["domain"]]]

	def to_count_matrix(docs,vocab):

		t_to_idx = {}
		for i,v in enumerate(vocab):
			t_to_idx[v]=i
		t_mat = []
		for d in docs:
			d_vec = [0] * len(vocab)
			for t in d:
				d_vec[t_to_idx[t[0]]]=t[1]
			t_mat.append(d_vec)

		return t_mat,t_to_idx

	idf_vecs = TfidfTransformer()
	print ("Loading data.")
	df = pd.read_csv(main_path+"/"+"{0}_domains_per_{1}.csv".format("all_actors",com_var))
	df=df.dropna(axis=0)
	print (len(df))
	#vocab = sorted(list(set(list(df["domain"].astype(str).dropna()))))
	vocab = df.groupby("domain")
	vocab = vocab.filter(lambda x: x['n_shared'].sum() > 19)
	vocab = sorted(list(set(list(vocab["domain"].astype(str).dropna()))))
	df = df[df["domain"].isin(set(vocab))]
	print (len(vocab))
	print (len(df))
	print ("Creating docs.")
	#pre_docs = [(row["domain"],row["n_shared"]) for i,row in df.iterrows()]
	pre_docs = [list(df[df[com_var]==com][["domain","n_shared"]].to_records(index=False)) for com in set(list(df[com_var]))]
	print ("Creating count matrix.")
	docs,t_to_idx = to_count_matrix(pre_docs,vocab)
	print ("Fitting.")
	idf_vecs.fit(docs)
	df["idf"]=df.apply(idf,args=[idf_vecs,t_to_idx],axis=1)
	df["tf_idf"]=df.apply(tf_idf,args=[idf_vecs,t_to_idx],axis=1)

	print (df.sort_values("tf_idf",inplace=False,ascending=False))
	df.to_csv(main_path+"/"+"{0}_domains_per_{1}_tfidf40.csv".format("all_actors",com_var),index=False)

def explore_actor_net_content(main_path,title="altmed_germany",actor_p="PParzival_Twitter",url=None):

	g = nx.read_gexf(main_path+"/inner_com_net_test.gexf")
	if actor_p is not None:
		for o,e,d in g.edges(actor_p,data=True):
			post = get_post_from_actor_and_link(actor_p,e)
			if post is not None:
				print (e+"\t"+post["text"])
				print ()
				print ()
	if url is not None:
		for o,e,d in list(g.edges(url,data=True)):
			post = get_post_from_actor_and_link(e,url)
			if post is not None:
				print (g.nodes[e]["actor_name"]+"\t"+post["text"])
				print ()
				print ("-------------------------------------------")
				print ()


def explore_actor_net(main_path,title="altmed_germany"):

	df = pd.read_csv(main_path+"/"+"all_actors.csv")
	df=df[df["title"]==title]
	#actors = set(list(df[(df["inner_com"]=="germanyi_45.0") & df["show_label"]==True]["actor"]))
	actors = list(set(list(df[(df["inner_com"]=="germanyi_45.0")]["actor"])))
	g = get_actor_net_exploration(actors=actors)
	nx.set_node_attributes(g,"Null",name="pol_main_fringe_sharp")
	nx.set_node_attributes(g,df.set_index("actor_platform")[["pol_main_fringe_sharp"]].to_dict()["pol_main_fringe_sharp"],name="pol_main_fringe_sharp")
	nx.set_node_attributes(g,df.set_index("actor_platform")[["link_to_actor"]].to_dict()["link_to_actor"],name="link_to_actor")
	nx.set_node_attributes(g,df.set_index("actor_platform")[["main_fringe_sharp"]].to_dict()["main_fringe_sharp"],name="main_fringe_sharp")
	nx.set_node_attributes(g,df.set_index("actor_platform")[["show_label"]].to_dict()["show_label"],name="show_label")
	nx.set_node_attributes(g,df.set_index("actor_platform")[["actor_name"]].to_dict()["actor_name"],name="actor_name")
	main_fringe_dist = get_actor_net_cat_dist(g,node_type="url",cat="main_fringe_sharp",vals=["grey2","mainstream"])
	for n,vals in main_fringe_dist.items():
		for v,vv in vals.items():
			nx.set_node_attributes(g, {n:vv}, v)
	show_label_dist = get_actor_net_cat_dist(g,node_type="url",cat="actor_name",vals=list(set(list(df[(df["inner_com"]=="germanyi_45.0") & df["show_label"]==True]["actor_name"]))))
	for n,vals in show_label_dist.items():
		for v,vv in vals.items():
			if "distance" in v:
				nx.set_node_attributes(g, {n:vv}, v)
	nx.write_gexf(g,main_path+"/inner_com_net_test.gexf")

def posts_export_russian_dk(main_path,title,net_title,new=False):

	#df = create_actor_data(title,new=False,main_path=main_path,net_title=net_title)
	df = pd.read_csv(main_path+"/"+"{0}_final.csv".format(title))
	df = df[df["platform"]!="web"]
	df["is_central_country_actor"]=df["actor_platform"].map({row["actor_platform"]:True if (row["is_native_lang"]==1 or row["pol_main_fringe_sharp"]!="grey2_grey") else False for i,row in df.iterrows()})
	mdb = MongoSpread()
	actors = list(set(list(df["actor"])))
	actors.extend([d["actor"] for d in mdb.database["actor_metric"].find({"lang":"da"})])
	add_actor_langs = {d["actor_platform"]:d["lang"] for d in mdb.database["actor_metric"].find({"lang":"da"})}

	yt_actors = {}
	yt_actors.update({LinkCleaner().extract_username(str(d["Youtube"])):"second_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"Second Pillar"})})
	yt_actors.update({LinkCleaner().extract_username(str(d["Youtube"])):"first_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"First Pillar"})})
	yt_actors.update({LinkCleaner().extract_username(str(d["Youtube"])):"third_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"Third Pillar"})})
	yt_actors = {LinkCleaner().extract_username(str(d["Youtube"])):"rt_sputnik" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"rt_sputnik"})}
	yt_actors.update({str(d["Actor"]):"second_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"Second Pillar"})})
	yt_actors.update({str(d["Actor"]):"first_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"First Pillar"})})
	yt_actors.update({str(d["Actor"]):"third_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"Third Pillar"})})
	yt_actors.update({str(d["Actor"]):"rt_sputnik" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"rt_sputnik"})})

	del yt_actors[None]
	yt_doms = []
	yt_all_doms = {}
	for d in mdb.database["url_bi_network"].find({"actor":{"$in":list(set(list(yt_actors.keys())))},"platform":"youtube"}):
		yt_doms.append(d["url"])
		if "youtube." in str(d["domain"]) or "youtu.be" in str(d["domain"]):
			yt_all_doms[d["domain"]]=yt_actors[d["actor"]]
	print (len(yt_doms))
	print (len(actors))

	domain_cats = {LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://",""):"rt_sputnik" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"rt_sputnik"})}
	domain_cats.update({LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://",""):"second_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"Second Pillar"})})
	domain_cats.update({LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://",""):"first_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"First Pillar"})})
	domain_cats.update({LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://",""):"third_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"Third Pillar"})})
	domain_cats.update(yt_all_doms)
	del domain_cats["odysee.com"]
	#del domain_cats["bit.ly"]
	#del domain_cats["fb.me"]
	print (len(domain_cats))

	if isfile(main_path+"/"+"{0}_posts_with_rus.csv".format(title)) and not new:
		print ("found previous file.")
		posts = pd.read_csv(main_path+"/"+"{0}_posts_with_rus.csv".format(title))
	else:
		dom_posts = []
		dom_posts.extend(get_posts_from_actors(actors,only_urls=yt_doms))
		print (len(dom_posts))
		dom_posts.extend(get_posts_from_actors(actors,only_domains=list(set(list(domain_cats.keys())))))
		print (len(dom_posts))
		posts = pd.DataFrame(dom_posts)
		posts.to_csv(main_path+"/"+"{0}_posts_with_rus.csv".format(title),index=False)

	posts = posts[~posts["link"].str.contains("odysee.com")]

	posts["interactions_sqrt"]=np.sqrt(posts["interactions"])
	interactions_platform_norm = {group:name["interactions_sqrt"] for group,name in posts[["interactions_sqrt","platform"]].groupby("platform").mean().iterrows()}
	posts["interactions_pl_norm"] = posts.apply(add_platform_norm,args=["interactions",interactions_platform_norm],axis=1)

	posts["pillar"]=posts["domain"].map(domain_cats)
	posts["main_fringe_sharp"]=posts["actor_platform"].map(df.set_index("actor_platform")[["main_fringe_sharp"]].to_dict()["main_fringe_sharp"])
	posts["political_sharp"]=posts["actor_platform"].map(df.set_index("actor_platform")[["political_sharp"]].to_dict()["political_sharp"])
	posts["right"]=posts["actor_platform"].map(df.set_index("actor_platform")[["right"]].to_dict()["right"])
	posts["left"]=posts["actor_platform"].map(df.set_index("actor_platform")[["left"]].to_dict()["left"])
	posts["mainstream"]=posts["actor_platform"].map(df.set_index("actor_platform")[["mainstream"]].to_dict()["mainstream"])
	posts["russian"]=posts["actor_platform"].map(df.set_index("actor_platform")[["russian"]].to_dict()["russian"])
	posts["alternative"]=posts["actor_platform"].map(df.set_index("actor_platform")[["alternative"]].to_dict()["alternative"])
	posts["grey1"]=posts["actor_platform"].map(df.set_index("actor_platform")[["grey1"]].to_dict()["grey1"])
	posts["is_central_country_actor"]=posts["actor_platform"].map(df.set_index("actor_platform")[["is_central_country_actor"]].to_dict()["is_central_country_actor"])
	actor_langs = df.set_index("actor_platform")[["lang"]].to_dict()["lang"]
	actor_langs.update(add_actor_langs)
	posts["actor_lang"]=posts["actor_platform"].map(actor_langs)
	posts = posts.drop_duplicates(subset='message_id', keep="first")
	posts = posts[posts["domain"]!="bit.ly"]
	posts = posts[posts["domain"]!="fb.me"]
	posts.to_csv(main_path+"/"+"{0}_posts_final_with_rus.csv".format(title),index=False)

def actor_export_russian_dk(main_path,title,net_title,nat_langs=["de","de-AT","de-DE"]):

	mdb = MongoSpread()
	print ("Creating data for title: {0}...".format(title))
	df = create_actor_data(title,new=False,main_path=main_path,net_title=net_title)
	df = df[df["platform"]!="web"]

	#print ("Loading net...")
	#net = nx.read_gexf(main_path+"/"+"{0}_BACKBONED_filtered.gexf".format(net_title))

	domain_cats = {LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://",""):"second_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"rt_sputnik"})}
	domain_cats.update({LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://",""):"second_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"Second Pillar"})})
	domain_cats.update({LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://",""):"first_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"First Pillar"})})
	domain_cats.update({LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://",""):"third_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"Third Pillar"})})
	del domain_cats["odysee.com"]

	domain_cat_counts = get_actor_domain_category_shares(list(set(list(df["actor"]))),domain_cats,filepath=main_path+"/"+"{0}_domain_cats_share.p".format(title))
	for cat,val_map in domain_cat_counts.items():
		df[cat]=df["actor_platform"].map(val_map)
		df[cat]=df[cat]-0.5

	print ("Adding special coms.")
	df = add_inner_outer_com(df,title,inner_base="com_res=25_small")

	print ("Adding calculated fields.")
	df["interactions_mean_sqrt"]=np.sqrt(df["interactions_mean"])
	interactions_platform_norm = {group:name["interactions_mean_sqrt"] for group,name in df[["interactions_mean_sqrt","platform"]].groupby("platform").mean().iterrows()}
	df["interactions_mean_pl_norm"] = df.apply(add_platform_norm,args=["interactions_mean",interactions_platform_norm],axis=1)

	df["fringeness"]=df["grey2"]-df["mainstream"]
	df["fringeness_score"]=df["grey2_score"]-df["mainstream_score"]
	df["left_vs_right"] = df.apply(add_collapsed_binary_value,args=[["left","right"]],axis=1)

	for cat,val_map in domain_cat_counts.items():
		df[cat+"_prop"]=df[cat]/df["links_shared_total"]

	df = add_actor_native_language(df,nat_langs)
	print ("Setting language per community.")
	actors_i0 = {d["Actor"]:d["org_project_title"] for d in mdb.database["actor"].find({"Iteration":0})}
	df["iteration0_title"]=df["actor"].map(actors_i0)
	df.loc[df["actor"] == "RT Deutschland main", 'iteration0_title'] = "altmed_germany"
	df.loc[df["actor"] == "TichysEinblick", 'iteration0_title'] = "altmed_germany"
	df.loc[df["actor"] == "tichyseinblick", 'iteration0_title'] = "altmed_germany"

	if len(nat_langs) > 0:
		#df["com_res=40_nat_lang_count"]=df["com_res=40"].map(get_custom_value_count_grouped(df,"com_res=40",vals={"lang":set(nat_langs)}))
		#df["com_res=40_nat_it0_count"]=df["com_res=40"].map(get_custom_value_count_grouped(df,"com_res=40",vals={"iteration0_title":set([title.replace("_small","")])}))
		#df["com_res=25_nat_lang_count"]=df["com_res=25"].map(get_custom_value_count_grouped(df,"com_res=25",vals={"lang":set(nat_langs)}))
		#df["com_res=25_nat_it0_count"]=df["com_res=25"].map(get_custom_value_count_grouped(df,"com_res=25",vals={"iteration0_title":set([title.replace("_small","")])}))
		#df["com_res=25_small_nat_lang_count"]=df["com_res=25_small"].map(get_custom_value_count_grouped(df,"com_res=25_small",vals={"lang":set(nat_langs)}))
		#df["com_res=25_small_nat_it0_count"]=df["com_res=25_small"].map(get_custom_value_count_grouped(df,"com_res=25_small",vals={"iteration0_title":set([title.replace("_small","")])}))
		#df["com_res=09_small_nat_lang_count"]=df["com_res=09_small"].map(get_custom_value_count_grouped(df,"com_res=09_small",vals={"lang":set(nat_langs)}))
		#df["com_res=09_small_nat_it0_count"]=df["com_res=09_small"].map(get_custom_value_count_grouped(df,"com_res=09_small",vals={"iteration0_title":set([title.replace("_small","")])}))
		#df["com_res=70_nat_lang_count"]=df["com_res=70"].map(get_custom_value_count_grouped(df,"com_res=70",vals={"lang":set(nat_langs)}))
		df["outer_com_nat_it0_count"]=df["outer_com"].map(get_custom_value_count_grouped(df,"outer_com",vals={"iteration0_title":set([title.replace("_small","")])}))
		df["outer_com_nat_lang_count"]=df["outer_com"].map(get_custom_value_count_grouped(df,"outer_com",vals={"lang":set(nat_langs)}))
		df["inner_com_nat_it0_count"]=df["inner_com"].map(get_custom_value_count_grouped(df,"inner_com",vals={"iteration0_title":set([title.replace("_small","")])}))
		df["inner_com_nat_lang_count"]=df["inner_com"].map(get_custom_value_count_grouped(df,"inner_com",vals={"lang":set(nat_langs)}))
		df["inner_com_grey2"]=df["inner_com"].map(get_mean_grouped(df,"inner_com","grey2"))
		df["inner_com_mainstream"]=df["inner_com"].map(get_mean_grouped(df,"inner_com","mainstream"))

	print ("Adding additional label data.")
	df = add_score_based_label(df,"pol_main_fringe_score",[("grey2_score","mainstream_score"),("right","left","other")])
	df = add_score_based_label(df,"pol_main_fringe",[("grey2","mainstream"),("right","left","other")])
	df = add_score_based_label(df,"pol_main_fringe_sharp",[("grey2","mainstream"),("right","left","other")],sharp=True)
	df = add_score_based_label(df,"political_sharp",[("right","left","other")],sharp=True)
	df = add_score_based_label(df,"main_fringe_sharp",[("mainstream","grey2")],sharp=True)
	df = add_score_based_label(df,"pillar_sharp",[("links_shared_first_pillar_prop","links_shared_second_pillar_prop","links_shared_third_pillar_prop")],sharp=True)
	df = add_score_based_label(df,"pillar",[("links_shared_first_pillar_prop","links_shared_second_pillar_prop","links_shared_third_pillar_prop")],sharp=False)
	#net = load_net(title,main_path=main_path)

	df["is_central_country_actor"]=df["actor_platform"].map({row["actor_platform"]:True if (row["is_native_lang"]==1 or row["pol_main_fringe_sharp"]!="grey2_grey") else False for i,row in df.iterrows()})

	print ("Adding com primary categories.")
	icl, icl_overlap = get_primary_category_grouped(df,"inner_com","lang")
	pmfs, pmfs_overlap = get_primary_category_grouped(df,"inner_com","pol_main_fringe_sharp")
	pmf, pmf_overlap = get_primary_category_grouped(df,"inner_com","pol_main_fringe")
	ps, ps_overlap = get_primary_category_grouped(df,"inner_com","political_sharp")
	fm, fm_overlap = get_primary_category_grouped(df,"inner_com","main_fringe_sharp")
	df["inner_com_lang"]=df["inner_com"].map(icl)
	df["inner_com_pol_main_fringe_sharp"]=df["inner_com"].map(pmfs)
	df["inner_com_pol_main_fringe"]=df["inner_com"].map(pmf)
	df["inner_com_political_sharp"]=df["inner_com"].map(ps)
	df["inner_com_main_fringe_sharp"]=df["inner_com"].map(fm)
	df["inner_com_lang_overlap"]=df["inner_com"].map(icl_overlap)
	df["inner_com_pol_main_fringe_sharp_overlap"]=df["inner_com"].map(pmfs_overlap)
	df["inner_com_pol_main_fringe_overlap"]=df["inner_com"].map(pmf_overlap)
	df["inner_com_political_sharp_overlap"]=df["inner_com"].map(ps_overlap)
	df["inner_com_main_fringe_sharp_overlap"]=df["inner_com"].map(ps_overlap)

	print ("Adding distance metrics.")
	df = add_value_distances(df,"political_polar_lvl",["right","left","other"])
	df = add_value_distances(df,"political_bipolar_lvl",["right","left"])
	df = add_value_distances(df,"fringemain_bipolar_lvl",["grey2","mainstream"])
	df = add_value_distances(df,"inner_com_fringemain_bipolar_lvl",["inner_com_grey2","inner_com_mainstream"])

	#print ("Adding ranks.")
	#df = add_rank_binary(df,"inner_com",net=net)

	df.to_csv(main_path+"/"+"{0}_final_with_rus.csv".format(title),index=False)

	print ("Loading net...")
	net = nx.read_gexf(main_path+"/"+"{0}_BACKBONED_filtered.gexf".format(net_title))

	nx.set_node_attributes(net,df.set_index("actor_platform")[["inner_com"]].to_dict()["inner_com"],name="inner_com")
	nx.set_node_attributes(net,df.set_index("actor_platform")[["outer_com"]].to_dict()["outer_com"],name="outer_com")
	nx.set_node_attributes(net,df.set_index("actor_platform")[["pillar"]].to_dict()["pillar"],name="pillar")
	nx.set_node_attributes(net,df.set_index("actor_platform")[["pillar_sharp"]].to_dict()["pillar_sharp"],name="pillar_sharp")
	nx.set_node_attributes(net,df.set_index("actor_platform")[["is_central_country_actor"]].to_dict()["is_central_country_actor"],name="is_central_country_actor")

	nx.write_gexf(net,main_path+"/"+"{0}_net_final_with_rus.gexf".format(net_title))

def export_net_slice(main_path,titles=["altmed_denmark","altmed_sweden"],new=False):

	def add_node_and_edges(g,node0,node1,weight=1,node0_atts={},node1_atts={}):

		if node0 not in g:
			g.add_node(node0,**node0_atts)
		if node1 not in g:
			g.add_node(node1,**node1_atts)
		if g.has_edge(node0,node1):
			g.get_edge_data(node0,node1)['weight'] += weight
		elif g.has_edge(node1,node0):
			g.get_edge_data(node0,node1)['weight'] += weight
		else:
			g.add_edge(node0,node1,weight=weight,)

		return g

	NODE = "actor"
	if not new and isfile(main_path+"/{0}_net_{1}.gexf".format(NODE,"slice_test")):
		all_actors = pd.read_csv(main_path+"/all_actors.csv")
		all_actors['iteration0_title'] = all_actors['iteration0_title'].fillna("")
		all_actors['rank_score_alt'] = np.where( (all_actors['iteration0_title'] == 'altmed_denmark') | (all_actors['iteration0_title'] == 'altmed_sweden' ), 0.0, all_actors['rank_score_alt'])
		all_actors['denmark_sweden'] = ""
		all_actors['denmark_sweden'] = np.where( (all_actors['title'] == 'altmed_denmark') & (all_actors['dk_sv_count'] == 1 ), "denmark", all_actors['denmark_sweden'])
		all_actors['denmark_sweden'] = np.where( (all_actors['title'] == 'altmed_sweden') & (all_actors['dk_sv_count'] == 1 ), "sweden", all_actors['denmark_sweden'])
		all_actors['denmark_sweden'] = np.where( all_actors['dk_sv_count'] > 1 , "both_countries", all_actors['denmark_sweden'])
		all_actors['inner_com_adj'] = np.where( all_actors['inner_com_pol_main_fringe_sharp'] == "grey2_grey" , "null", all_actors['inner_com'])
		all_actors['inner_com_pol_main_fringe_sharp_adj'] = "null"
		all_actors['inner_com_pol_main_fringe_sharp_adj'] = np.where( all_actors['inner_com_pol_main_fringe_sharp'] == "grey2_grey" , "null", all_actors['inner_com_pol_main_fringe_sharp'])
		max_rank_score_alt_dk = max(list(all_actors[all_actors["denmark_sweden"]=="denmark"]["rank_score_alt"]))
		max_rank_score_alt_sv = max(list(all_actors[all_actors["denmark_sweden"]=="sweden"]["rank_score_alt"]))
		max_alt_share_dk = max(list(np.log(all_actors[all_actors["denmark_sweden"]=="denmark"]["alt_share_count"]+1)))
		max_alt_share_sv = max(list(np.log(all_actors[(all_actors["denmark_sweden"]=="sweden") | (all_actors["denmark_sweden"]=="both_countries")]["alt_share_count"]+1)))
		only_dk = set(list(all_actors[all_actors["denmark_sweden"]=="denmark"]["actor_platform"]))
		sv_and_both = set(list(all_actors[all_actors["denmark_sweden"]=="sweden"]["actor_platform"]))
		sv_and_both.update(set(list(all_actors[all_actors["denmark_sweden"]=="both_countries"]["actor_platform"])))
		max_rank_score_alt = max(list(all_actors["rank_score_alt"]))
		all_actors["rank_score_alt_index"]=all_actors["rank_score_alt"]/max_rank_score_alt
		all_actors['rank_score_alt_index'] = np.where( all_actors['denmark_sweden'] == 'denmark', all_actors["rank_score_alt_index"]/max_rank_score_alt_dk, all_actors["rank_score_alt_index"])
		all_actors['rank_score_alt_index'] = np.where( (all_actors['denmark_sweden'] == 'sweden') | (all_actors['denmark_sweden'] == "both_countries" ), all_actors["rank_score_alt_index"]/max_rank_score_alt_sv, all_actors["rank_score_alt_index"])
		print ("loading net")
		dknet = nx.read_gexf(main_path+"/{0}_net_{1}.gexf".format(NODE,"altmed_denmark"))
		svnet = nx.read_gexf(main_path+"/{0}_net_{1}.gexf".format(NODE,"altmed_sweden"))
		print ("re-filtering")
		svnet = disparity_filter_alpha_cut(disparity_filter(svnet),alpha_t=0.07)
		net = nx.compose(svnet,dknet)
		net = NetworkUtils().add_communities(net)
		alt_net = nx.read_gexf(main_path+"/{0}_net_{1}_alt.gexf".format(NODE,"slice_test"))
		alt_edge_weight = {}
		for org,edg,aw in alt_net.edges(data=True):
			if org in only_dk and edg in only_dk:
				index_max = max_alt_share_dk
			else:
				index_max = max_alt_share_sv
			alt_edge_weight[(org,edg)]=np.log(aw["weight"]+1)/index_max
		print (len(net.nodes()))
		print (len(net.edges()))

		nx.set_edge_attributes(net, alt_edge_weight, name="alt_weight")

		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["pol_main_fringe_sharp"]].to_dict()["pol_main_fringe_sharp"],name="pol_main_fringe_sharp")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["inner_com_adj"]].to_dict()["inner_com_adj"],name="inner_com_adj")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["platform"]].to_dict()["platform"],name="platform")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["lang"]].to_dict()["lang"],name="lang")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["main_fringe_sharp"]].to_dict()["main_fringe_sharp"],name="main_fringe_sharp")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["iteration0_title"]].to_dict()["iteration0_title"],name="iteration0_title")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["rank_score_alt_index"]].to_dict()["rank_score_alt_index"],name="rank_score_alt_index")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["degrees"]].to_dict()["degrees"],name="degrees")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["dk_sv_count"]].to_dict()["dk_sv_count"],name="dk_sv_count")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["denmark_sweden"]].to_dict()["denmark_sweden"],name="denmark_sweden")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["actor_name"]].to_dict()["actor_name"],name="Label")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["alt_share_count"]].to_dict()["alt_share_count"],name="alt_share_count")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["political_sharp_safe"]].to_dict()["political_sharp_safe"],name="political_sharp_safe")
		nx.set_node_attributes(net,all_actors.set_index("actor_platform")[["inner_com_pol_main_fringe_sharp_adj"]].to_dict()["inner_com_pol_main_fringe_sharp_adj"],name="inner_com_pol_main_fringe_sharp_adj")

		for n, d in list(net.nodes(data=True)):
			if "rank_score_alt_index" not in d:
				net.remove_node(n)
		print (len(net.nodes()))
		print (len(net.edges()))
		nx.write_gexf(net,main_path+"/{0}_net_{1}_atts.gexf".format(NODE,titles[0]))
		all_actors["dk_sv_com"]=all_actors["actor_platform"].map({n:d["modularity"] for n,d in net.nodes(data=True)})
		all_actors[all_actors["title"].isin(set(["altmed_denmark","altmed_sweden"]))]
		all_actors.to_csv(main_path+"/all_actors.csv",index=False)
	else:
		mdb = MongoSpread()
		search_using = "actor"
		net_data = {}
		g = nx.Graph()
		all_actors = pd.read_csv(main_path+"/all_actors.csv")
		all_actors = all_actors[all_actors["title"].isin(set(titles))]
		all_actors = all_actors[all_actors["alt_share_count"]>3]
		alt_websites = []
		for title in titles:
			alt_websites.extend([LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://","") for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":title.replace("_small","")})])
		actor_labels = {}
		if search_using =="domain":
			search_for = alt_websites
		elif search_using == "actor":
			search_for = list(set(list(all_actors["actor"])))
		for aw in search_for:
			if search_using =="domain":
				query = {"domain":aw,"actor_platform":{"$in":list(set(list(all_actors["actor_platform"])))}}
			elif search_using == "actor":
				query = {"actor":aw}
			for doc in mdb.database["url_bi_network"].find(query):
				ap = doc["actor_platform"]
				pl = doc["platform"]
				url = doc["url"]
				actor_labels[ap]=doc["actor_label"]
				if not url in net_data: net_data[url]={}
				if not ap in net_data[url]: net_data[url][ap]=0.0
				net_data[url][ap]+=1.0

		print ("building net...")
		print (len(net_data))
		url_count = 0
		pl_dict = {}
		for aw,aps in net_data.items():
			url_count+=1
			if url_count % 100 == 0:
				print (str(url_count) + " out of "+ str(len(net_data)))
			pl = ap.split("_")[-1].lower().split(" ")[0]
			dup_tups = set([])
			if len(aps) > 60:
				print (aw + "  " + str(len(aps)))
			for aw1,w1 in list(aps.items()):
				for aw2,w2 in list(aps.items()):
					pl1 = aw1.split("_")[-1].lower().split(" ")[0]
					pl2 = aw2.split("_")[-1].lower().split(" ")[0]
					pl_dict[aw1]=pl1
					pl_dict[aw2]=pl2
					if aw1 != aw2:
						if (aw1,aw2) not in dup_tups and (aw2,aw1) not in dup_tups:
							g = add_node_and_edges(g,aw1,aw2,weight=(w1+w2),node0_atts={"platform":pl1},node1_atts={"platform":pl2})
							dup_tups.add((aw1,aw2))
		print ("filtering")
		#NetworkUtils().filter_by_degrees(g,degree=3,skip_nodes={},preserve_skip_node_edges=True,extra=None)
		g = disparity_filter_alpha_cut(disparity_filter(g),alpha_t=0.32)
		print ("writing")
		print (len(g.nodes()))
		print (len(g.edges()))
		nx.set_node_attributes(g,pl_dict,name="platform")
		nx.set_node_attributes(g,actor_labels,name="Label")
		if search_using == "actor":
			nx.write_gexf(g,main_path+"/{0}_net_{1}.gexf".format(NODE,titles[0]))
		elif search_using == "domain":
			nx.write_gexf(g,main_path+"/{0}_net_{1}_alt.gexf".format(NODE,titles[0]))
