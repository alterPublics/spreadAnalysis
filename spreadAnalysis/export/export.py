from spreadAnalysis.persistence.simple import FlatFile, Project
from spreadAnalysis import _gvars as gvar
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
import scipy.spatial
import networkx as nx

MAIN_PATH = "/home/alterpublics/projects"

SHARED_DATA_FIELDS = {
"actor_name":Spread._get_actor_name,
"actor_username":Spread._get_actor_username,
"followers":Spread._get_followers,
"platform":Spread._get_platform,
"lang":Spread._get_lang,
"interactions":Spread._get_interactions,
"text":Spread._get_message_text,
"link":Spread._get_message_link,
"domain":Spread._get_message_link_domain,
"datetime":Spread._get_date,
"post_url":Spread._get_post_url}

SHARED_DATA_FIELDS_SMALL = {
"actor_username":Spread._get_actor_username,
"platform":Spread._get_platform,
"lang":Spread._get_lang,
"link":Spread._get_message_link,
"datetime":Spread._get_date,
"post_url":Spread._get_post_url}

TEXT_FIELD = {
"text":Spread._get_message_text}

def get_platform_uniform_post(post,custom_schema=None):

    sch = SHARED_DATA_FIELDS
    if custom_schema is not None: sch = custom_schema

    new_post = {}
    for col, func in sorted(sch.items()):
        new_post[col]=func(data = post, method=post["method"])
        #new_post.append(func(data = post, method=post["method"]))
    return new_post

def default_neg():
    return -1

def default_unknown():
    return 0.5

def default_zero():
    return 0.5

def add_platform_norm(row,measure="interactions_mean",norm={}):

    return np.sqrt(row[measure])/norm[row["platform"]]

def replace_values_in_dataset(df1,df2,key="actor_platform",cols=["mainstream","mainstream_score","grey2","grey2_score"]):

    replace_with = df2[df2[key].isin(set(list(df1[key])))][[*cols,key]].set_index(key).to_dict()
    for i,row in df1[~df1[key].isin(set(list(df2[key])))].iterrows():
        for col in cols:
            replace_with[col][row[key]]=row[col]
    for col in cols:
        df1[col] = df1[key].map(replace_with[col])

    return df1

def get_posts_from_urls(urls,only_aps):

    mdb = MongoSpread()
    df = []
    for url in urls:
        query = mdb.database["url_bi_network"].find({"url":url,"actor_platform":{"$in":only_aps}})
        for url_doc in query:
            for mid in url_doc["message_ids"]:
                post = mdb.database["post"].find_one({"message_id":mid})
                doc = get_platform_uniform_post(post)
                doc["message_id"]=mid
                doc["actor_platform"]=url_doc["actor_platform"]
                doc["link"]=url_doc["url"]
                doc["domain"]=url_doc["domain"]
                df.append(doc)
    return df

def get_posts_from_actors(actors,only_domains=[],only_urls=[],db_verify=False):

    mdb = MongoSpread()
    df = []
    acount = 0
    for actor in actors:
        acount += 1
        if db_verify:
            actors = list(mdb.database["actor_metric"].find({"actor_platform":actor}))
            if len(actors) > 0:
                if "actor" in actors[0]:
                    actor = actors[0]["actor"]
            else:
                actors = list(mdb.database["actor_metric"].find({"$or":[{"actor":actor},{"actor_platform":actor},{"actor_name":actor}]}))
                if len(actors) > 0:
                    actor = actors[0]["actor"]
                else:
                    print ("Problem verifying actor...")
        if len(only_domains) > 0:
            query = mdb.database["url_bi_network"].find({"actor":actor,"domain":{"$in":only_domains}})
        elif len(only_urls) > 0:
            query = mdb.database["url_bi_network"].find({"actor":actor,"url":{"$in":only_urls}})
        else:
            query = mdb.database["url_bi_network"].find({"actor":actor})
        for url_doc in query:
        #for url_doc in mdb.database["url_bi_network"].find({"actor":actor}):
            for mid in url_doc["message_ids"]:
                post = mdb.database["post"].find_one({"message_id":mid})
                doc = get_platform_uniform_post(post)
                doc["message_id"]=mid
                doc["actor_platform"]=url_doc["actor_platform"]
                doc["link"]=url_doc["url"]
                doc["domain"]=url_doc["domain"]
                df.append(doc)
        if acount % 100 == 0:
            print (acount)
    return df

def add_collapsed_binary_value(row,vals=[]):

    if not (row[vals[0]]+row[vals[1]]) > 0:
        return None
    else:
        val0_s = row[vals[0]]/(row[vals[0]]+row[vals[1]])
        val1_s = row[vals[1]]/(row[vals[0]]+row[vals[1]])
        return val0_s - val1_s

def get_modular_edge_prop(g,com,default_val=0.0):

    mod_edge_prop = defaultdict(default_unknown)
    for n in list(g.nodes()):
        if n not in mod_edge_prop: mod_edge_prop[n]=default_val
        sum_edges = len(list(g.edges(n)))
        for o,e in list(g.edges(n)):
            if com in g.nodes[e] and com in g.nodes[n]:
                if g.nodes[e][com] == g.nodes[n][com]:
                    mod_edge_prop[n]+=1
        mod_edge_prop[n]=float(mod_edge_prop[n])/float(sum_edges)
    return mod_edge_prop

def add_rank_binary(df,group,pluck=10,att_name="centrality_rank",net=None,balance_by=["platform"],oversample_by=["pol_main_fringe_sharp"]):

    df["mod_edge_prop"]=1.0
    if net is not None:
        mod_edge_prop = get_modular_edge_prop(net,group,default_val=0.5 )
        df["mod_edge_prop"]=df["actor_platform"].map(mod_edge_prop)
    df["rank_score"]=np.sqrt(df["pagerank"])*df["interactions_mean_pl_norm"]*np.sqrt(df["mod_edge_prop"])
    ranks = defaultdict(default_neg)
    for g in set(list(df[group])):
        f_df = df[df[group]==g]
        rank = 1
        for i,row in f_df.sort_values("rank_score",ascending=False,inplace=False).head(len(f_df)).iterrows():
            ranks[row["actor_platform"]]=rank
            rank+=1
            if g == "austriai_4.0" and row["pol_main_fringe_sharp"]=="grey2_grey": continue
            if rank >= pluck:
                break
        if len(balance_by) > 0:
            abort = False
            for balance in balance_by:
                bs = set(list(f_df[balance]))
                for b in bs:
                    for i,row in f_df[f_df[balance]==b].sort_values("rank_score",ascending=False,inplace=False).head(len(f_df)).iterrows():
                        if len(oversample_by) > 0:
                            if row["inner_com_political_sharp"]!=row["political_sharp"] or (row["inner_com_political_sharp"]=="grey" and row["pol_main_fringe_sharp"]=="grey2_grey"): continue
                        if g == "austriai_4.0" and row["pol_main_fringe_sharp"]=="grey2_grey": continue
                        if row["actor_platform"] not in ranks:
                            if not abort:
                                ranks[row["actor_platform"]]=rank
                                rank+=1
                            if rank >= pluck*2:
                                abort = True
        if len(oversample_by) > 0:
            abort = False
            for i,oversample in enumerate(oversample_by):
                for i,row in f_df[f_df[group+"_"+oversample]==f_df[oversample]].sort_values("rank_score",ascending=False,inplace=False).head(len(f_df)).iterrows():
                    if row["actor_platform"] not in ranks:
                        if not abort:
                            ranks[row["actor_platform"]]=rank
                            rank+=1
                        if rank >= ((pluck*2)+((0+1)*pluck)):
                            abort = True
    df["{0}_rank".format(group)]=df["actor_platform"].map(ranks)

    return df

def add_actor_native_language(df,nat_langs):

    natives = {}
    for i,row in df.iterrows():
        if row["lang"] in nat_langs:
            natives[row["actor_platform"]]=1
        else:
            natives[row["actor_platform"]]=0
    df["is_native_lang"]=df["actor_platform"].map(natives)
    return df

def add_special_com(df,coms=["com_res=25_small","com_res=09_small","com_res=70","com_res=40","com_res=25","com_res=09"],limit=0.03):

    actor_coms = {}
    n_actors = float(len(df))
    for com in coms:
        com_groups = df[[com,"actor_platform"]].groupby(com).count().to_dict()["actor_platform"]
        for i,row in df.iterrows():
            if row["actor_platform"] not in actor_coms:
                if com != coms[-1]:
                    if not pd.isna(row[com]):
                        if com_groups[row[com]] > n_actors*limit:
                            actor_coms[row["actor_platform"]]=com+"_"+str(row[com])
                else:
                    actor_coms[row["actor_platform"]]=com+"_"+str(row[com])
    print(len(set(list(actor_coms.values()))))
    print (set(list(actor_coms.values())))
    df["special_com"]=df["actor_platform"].map(actor_coms)

    return df

def add_inner_outer_com(df,title,outer_base="com_res=09",inner_base="com_res=70"):

    n_actors = float(len(df))
    actor_inner_coms = {}
    central_coms = set([])
    com_groups_cen = df[[outer_base,"btw_cen"]].groupby(outer_base).mean("btw_cen").to_dict()["btw_cen"]
    com_groups_n = df[[outer_base,"actor_platform"]].groupby(outer_base).count().to_dict()["actor_platform"]

    filled = 0
    for com,v in sorted(com_groups_cen.items(), key = itemgetter(1), reverse=True):
        central_coms.add(com)
        filled+=com_groups_n[com]
        if filled > 0.65*n_actors:
            break
    print (central_coms)
    for i,row in df.iterrows():
        if row[outer_base] in central_coms:
            if pd.isna(row[inner_base]):
                actor_inner_coms[row["actor_platform"]]=title.replace("altmed_","")+"o_"+str(row[outer_base])
            else:
                actor_inner_coms[row["actor_platform"]]=title.replace("altmed_","")+"i_"+str(row[inner_base])
        else:
            actor_inner_coms[row["actor_platform"]]=title.replace("altmed_","")+"o_"+str(row[outer_base])
    df["outer_com"]=title.replace("altmed_","")+"o_"+df[outer_base].astype(str)
    df["inner_com"]=df["actor_platform"].map(actor_inner_coms)

    return df

def get_actor_domain_category_shares(actors,domain_cats,filepath=None,new=False):

    mdb = MongoSpread()
    if not new and filepath is not None and isfile(filepath):
        counts = pickle.load(open(filepath,"rb"))
    else:
        counts = {"links_shared_"+str(c):defaultdict(default_zero) for c in list(domain_cats.values())}
        counts["links_shared_total"]=defaultdict(default_zero)
        acount = 0
        for actor in actors:
            acount+=1
            for doc in mdb.database["url_bi_network"].find({"actor":actor}):
                if doc["domain"] in domain_cats:
                    #if not doc["actor_platform"] in counts["links_shared_"+domain_cats[doc["domain"]]]:
                    counts["links_shared_"+domain_cats[doc["domain"]]][doc["actor_platform"]]+=1
                counts["links_shared_total"][doc["actor_platform"]]+=1
            if acount % 100 == 0:
                print (acount)
        if filepath is not None:
            pickle.dump(counts,open(filepath,"wb"))

    return counts

def get_custom_value_count_grouped(df,group,vals={},norm=True):

    counts = {}
    for i,row in df.iterrows():
        if not pd.isna(row[group]):
            if not row[group] in counts: counts[row[group]]=0.0
            for k,v in vals.items():
                if row[k] in v:
                    counts[row[group]]+=1.0
    if norm:
        for g in set(list(df[group])):
            if not pd.isna(g):
                counts[g]/float(len(df[df[group]==g]))
    return counts

def get_primary_category_grouped(df,group,category,norm=True):

    counts = {}
    overlap_count = {}
    counts_lvl1 = df[[group,category]].groupby([category,group]).size().unstack(fill_value=0).to_dict()
    for k in list(counts_lvl1.keys()):
        prim = sorted(counts_lvl1[k].items(), key = itemgetter(1), reverse=True)[0][0]
        overlap = float(sorted(counts_lvl1[k].items(), key = itemgetter(1), reverse=True)[0][1])/float(np.sum(np.array(list(counts_lvl1[k].values()))))
        #if sorted(counts_lvl1[k].items(), key = itemgetter(1), reverse=True)[1][0] == "grey2_left":
            #prim = sorted(counts_lvl1[k].items(), key = itemgetter(1), reverse=True)[1][0]
            #overlap = float(sorted(counts_lvl1[k].items(), key = itemgetter(1), reverse=True)[1][1])/float(np.sum(np.array(list(counts_lvl1[k].values()))))

        counts[k]=prim
        overlap_count[k]=overlap

    return counts, overlap_count

def get_com_relative_dif_score(df,com_var,vals):

    scores = {}
    if len(vals) == 1:
        com_means = df[[com_var,vals[0]]].groupby(com_var).mean().to_dict()[vals[0]]
        for i,row in df.iterrows():
            scores[row["actor_platform"]]=float(abs(row[vals[0]]-com_means[row[com_var]]))

    return scores

def get_closeness_to_center(df,vals):

    scores = {}
    if len(vals) == 1:
        val_mean = float(np.mean(np.array(df[vals[0]])))
        for i,row in df.iterrows():
            scores[row["actor_platform"]]=float(abs(row[vals[0]]-val_mean))

    return scores

def get_mean_grouped(df,group,val):

    means = df[[group,val]].groupby(group).mean().to_dict()[val]
    return means

def add_domain_share_count(df,att_name,domains,var_name="alternative"):

    mdb = MongoSpread()
    counts = {}
    un_counts = {}
    one_pl_dom = {}
    two_pl_dom = {}
    top_first_dom = {}
    top_second_dom = {}
    top_third_dom = {}
    un_dom = {}
    for a in set(list(df["actor_platform"])):
        if a not in counts: counts[a]=0.0
        if a not in un_dom: un_dom[a]=0.0
        if a not in one_pl_dom: one_pl_dom[a]=0.0
        if a not in two_pl_dom: two_pl_dom[a]=0.0
        if a not in un_counts: un_counts[a]={}
        if a not in top_first_dom: top_first_dom[a]=0.0
        if a not in top_second_dom: top_second_dom[a]=0.0
        if a not in top_third_dom: top_third_dom[a]=0.0
    for dom in domains:
        for doc in mdb.database["url_bi_network"].find({"domain":dom}):
            if doc["actor_platform"] in counts:
                counts[doc["actor_platform"]]+=len(doc["message_ids"])
            if doc["actor_platform"] in un_counts:
                if dom not in un_counts[doc["actor_platform"]]:
                    un_counts[doc["actor_platform"]][dom]=0.0
                un_counts[doc["actor_platform"]][dom]+=1.0
    for k,v in un_counts.items():
        if len(v) > 0:
            un_dom[k]=len(v)
            sorted_dom_dist = sorted(v.items(), key = itemgetter(1), reverse=True)
            one_pl_dom[k]=np.sum(np.array([sorted_dom_dist[0][1]]))/np.sum(np.array(list(v.values())))
            top_first_dom[k]=sorted_dom_dist[0][0]
            if len(v) < 2:
                two_pl_dom[k]=np.sum(np.array([sorted_dom_dist[0][1]]))/np.sum(np.array(list(v.values())))
            else:
                top_second_dom[k]=sorted_dom_dist[1][0]
                two_pl_dom[k]=np.sum(np.array([sorted_dom_dist[0][1],sorted_dom_dist[1][1]]))/np.sum(np.array(list(v.values())))
            if len(v) > 2:
                top_third_dom[k]=sorted_dom_dist[2][0]
    df[att_name]=df["actor_platform"].map(counts)
    df["one_{0}_domain".format(var_name)]=df["actor_platform"].map(one_pl_dom)
    df["two_{0}_domain".format(var_name)]=df["actor_platform"].map(two_pl_dom)
    df["unique_{0}_domains".format(var_name)]=df["actor_platform"].map(un_dom)
    df["top_first_{0}_dom".format(var_name)]=df["actor_platform"].map(top_first_dom)
    df["top_second_{0}_dom".format(var_name)]=df["actor_platform"].map(top_second_dom)
    df["top_third_{0}_dom".format(var_name)]=df["actor_platform"].map(top_third_dom)

    return df

def add_domain_share_engagement(df,att_name,domains,new=False,filepath=None,mean=True):

    mdb = MongoSpread()
    counts = {}
    if not new and filepath is not None and isfile(filepath):
        counts = pickle.load(open(filepath,"rb"))
    else:
        for a in set(list(df["actor_platform"])):
            if a not in counts: counts[a]=[]
        for dom in domains:
            for doc in mdb.database["url_bi_network"].find({"domain":dom}):
                if doc["actor_platform"] in counts:
                    for mid in doc["message_ids"]:
                        for post_doc in mdb.database["post"].find({"message_id":mid}):
                            counts[doc["actor_platform"]].append(Spread._get_interactions(data=post_doc,method=post_doc["method"]))
        for a in list(counts.keys()):
            if mean == True:
                if len(counts[a]) < 1:
                    counts[a]=0.0
                else:
                    counts[a]=float(np.mean(np.array(counts[a])))
            else:
                if len(counts[a]) < 1:
                    counts[a]=0.0
                else:
                    counts[a]=float(np.sum(np.array(counts[a])))
        if filepath is not None:
            pickle.dump(counts,open(filepath,"wb"))

    df[att_name]=df["actor_platform"].map(counts)

    return df

def add_value_distances(df,att_name,att_vals,norm=True):

    dist_vals = {}
    s_att_vals = sorted(att_vals)
    expected_dist = np.array([1.0/float(len(att_vals)) for r in att_vals])
    max_dist = 0.0
    for i,row in df.iterrows():
        if not row["actor_platform"] in dist_vals: dist_vals[row["actor_platform"]]=None
        a_vec = []
        for att in s_att_vals:
            a_vec.append(row[att])
        dist = scipy.spatial.distance.sqeuclidean(a_vec,expected_dist)
        dist_vals[row["actor_platform"]]=dist
        if dist > max_dist: max_dist = dist

    if norm:
        for a in list(dist_vals.keys()):
            dist_vals[a]/max_dist

    df[att_name]=df["actor_platform"].map(dist_vals)

    return df

def add_score_based_label(net_df,att_name,att_vals,sharp=False):

    def _get_default_grey_label(medians):

        grey_list = [a for a in medians.keys() if "grey" in a]
        if len(grey_list) > 0:
            default_grey_label = [a for a in medians.keys() if "grey" in a][0]
        else:
            default_grey_label = "grey"
        return default_grey_label

    def _unpack_list_of_tuples(lt):

        new_l = []
        for t in lt:
            for e in t:
                new_l.append(e)
        return new_l

    def _get_node_score_medians(df,atts):

        df = df[[a for a in atts]]
        medians = df.median()
        return dict(medians)

    def _find_label(stdzied,att_vals):

        if not isinstance(att_vals[0],tuple):
            att_vals = [tuple(att_vals)]
        multi_label = []
        for tup in att_vals:
            stdzied_filt = {k:v for k,v in stdzied.items() if k in set(tup)}
            default_grey_label = _get_default_grey_label(stdzied_filt)
            if _is_all_low(list(stdzied_filt.values())):
                label = default_grey_label
            else:
                label = sorted(stdzied_filt.items(), key = itemgetter(1), reverse=True)[0][0]
            multi_label.append(label)
        final_label = "_".join(multi_label)

        return final_label

    def _find_label_sharp(stdzied,att_vals,sharpness=0.15):

        if not isinstance(att_vals[0],tuple):
            att_vals = [tuple(att_vals)]
        multi_label = []
        for tup in att_vals:
            stdzied_filt = {k:v for k,v in stdzied.items() if k in set(tup)}
            default_grey_label = _get_default_grey_label(stdzied_filt)
            highest = sorted(stdzied_filt.items(), key = itemgetter(1), reverse=True)[0]
            expected = 1.0/len(set(tup))
            if highest[1] >= expected+(expected*sharpness):
                label = highest[0]
            else:
                label = default_grey_label
            multi_label.append(label)
        final_label = "_".join(multi_label)

        return final_label

    def _is_all_low(scores):

        return all(s <= 0.0 for s in scores)

    if isinstance(att_vals[0],tuple):
        atts = _unpack_list_of_tuples(att_vals)
    medians = _get_node_score_medians(net_df,atts)
    #default_grey_label = _get_default_grey_label(medians)
    df_atts = {}
    for i,row in net_df.iterrows():
        n = row["actor_platform"]
        if sharp:
            stdzied = {a:row[a] for a in atts}
            label = _find_label_sharp(stdzied,att_vals)
        else:
            stdzied = {a:row[a]-medians[a] for a in atts}
            label = _find_label(stdzied,att_vals)
        df_atts[n]=label
    net_df[att_name]=net_df["actor_platform"].map(df_atts)
    return net_df

def create_actor_data(title,new=True,main_path=None,net_title=None):

    if main_path is None:
        main_path = MAIN_PATH
    if net_title is None:
        net_title = title
    if new:
        export_actor_to_pandas(main_path+"/"+"{0}.csv".format(title),\
            {"net_data.{0}".format(net_title):{"$exists":True}},net_name="{0}".format(net_title))
    else:
        if not isfile(main_path+"/"+"{0}.csv".format(title)):
            export_actor_to_pandas(main_path+"/"+"{0}.csv".format(title),\
                {"net_data.{0}".format(net_title):{"$exists":True}},net_name="{0}".format(net_title))
        else:
            pass
    df = pd.read_csv(main_path+"/"+"{0}.csv".format(title))

    return df

def load_net(title,main_path=None):

    if main_path is None:
        main_path = MAIN_PATH
    net = nx.read_gexf(main_path+"/"+"{0}_BACKBONED_filtered.gexf".format(title))

    return net

def domain_shares_per_actor(main_path,title,net_title):

    mdb = MongoSpread()
    df = create_actor_data(title,new=False,main_path=main_path,net_title=net_title)
    counts = {}
    acount = 0
    for a in set(list(df["actor"])):
        if not pd.isna(a):
            acount += 1
            for doc in mdb.database["url_bi_network"].find({"actor":a},{"actor_platform":1,"domain":1}):
                adt = (doc["actor_platform"],doc["domain"])
                if not adt in counts: counts[adt]=0.0
                counts[adt]+=1.0
        if acount % 1000 == 0:
            print (acount)

    df = []
    for adt, count in counts.items():
        df.append({"actor_platform":adt[0],"domain":adt[1],"n_shared":count})
    pd.DataFrame(df).to_csv(main_path+"/"+"{0}_domains_per_actor.csv".format(title))

def domain_shares_per_com(main_path,com_var="inner_com",exclude_yt=False):

    mdb = MongoSpread()
    df = pd.read_csv(main_path+"/"+"all_actors.csv")
    df = df[df["show_inner_com"]==True]
    counts = {}
    acount = 0
    for com in set(list(df[com_var])):
        actors_in_com = set(list(df[df[com_var]==com]["actor"]))
        print ("actors in com: "+str(len(actors_in_com)))
        for a in df[df[com_var]==com]["actor"]:
            if not pd.isna(a):
                acount += 1
                for doc in mdb.database["url_bi_network"].find({"actor":a},{"actor_platform":1,"domain":1}):
                    if exclude_yt and "youtube." in doc["domain"][:12].lower(): continue
                    adt = (com,doc["domain"])
                    if not adt in counts: counts[adt]={"n":0.0,"n_actors":set([])}
                    counts[adt]["n"]+=1.0
                    counts[adt]["n_actors"].add(doc["actor_platform"])
            if acount % 100 == 0:
                print (acount)

    df = []
    for adt, vals in counts.items():
        df.append({com_var:adt[0],"domain":adt[1],"n_shared":vals["n"],"n_actors":len(vals["n_actors"])})
    pd.DataFrame(df).to_csv(main_path+"/"+"{0}_domains_per_{1}.csv".format("all_actors",com_var))

def actor_export_default_corrected(main_path):

    df = pd.read_csv(main_path+"/all_actors.csv")

    """mf_cols = ['_id', 'actor_name', 'actor_platform', 'most_popular_url_shared', 'n_unique_domains_shared', 'most_often_shared_domain', 'lang', 'interactions_mean', 'followers_mean', 'platform', 'link_to_actor', 'pagerank', 'degrees', 'other', 'grey1', 'left', 'right', 'dist_to_0g1', 'mainstream', 'grey2', 'dist_to_0g2', 'inner_com', 'alt_share_engagement', 'alt_share_count', 'one_alt_domain', 'two_alt_domain', 'unique_alt_domains', 'top_first_dom', 'top_second_dom', 'interactions_mean_pl_norm', 'alt_share_engagement_pl_norm', 'fringeness', 'left_right', 'is_native_lang', 'iteration0_title', 'inner_com_nat_it0_count', 'inner_com_alt_share_count', 'inner_com_n_actors', 'pol_main_fringe_sharp', 'political_sharp', 'main_fringe_sharp', 'inner_com_lang', 'inner_com_pol_main_fringe_sharp', 'inner_com_main_fringe_sharp', 'rank_score', 'show_label', 'show_inner_com', 'left_right_center_closeness', 'fringeness_center_closeness', 'rank_score_alt', 'political_sharp_safe', 'alt_share_count_proportion', 'title']"""
    drop_cols = ['actor', 'actor_username','interactions_std','message_length_mean','message_length_std','first_post_observed','last_post_observed','followers_max','account_type','account_category', 'inserted_at', 'updated_at', 'btw_cen', 'russian', 'conspiracy', 'grey0', 'dist_to_0g0', 'alternative_score', 'russian_score', 'conspiracy_score', 'grey0_score', 'other_score', 'grey1_score', 'left_score', 'right_score', 'mainstream_score', 'grey2_score','main_label', 'political', 'main_fringe', 'com_res=09','com_res=25', 'com_res=40', 'com_res=70', 'com_res=09_small', 'com_res=25_small', 'com_res=40_small', 'com_res=70_small', 'outer_com', 'top_third_dom', 'interactions_mean_sqrt', 'alt_share_engagement_sqrt', 'fringeness_score', 'left_vs_right', 'outer_com_nat_it0_count', 'outer_com_nat_lang_count', 'inner_com_nat_lang_count', 'inner_com_grey2', 'inner_com_mainstream', 'pol_main_fringe_score', 'pol_main_fringe','inner_com_pol_main_fringe', 'inner_com_political_sharp', 'inner_com_political','inner_com_lang_overlap','inner_com_pol_main_fringe_sharp_overlap', 'inner_com_pol_main_fringe_overlap', 'inner_com_political_sharp_overlap', 'inner_com_main_fringe_sharp_overlap', 'political_polar_lvl', 'political_bipolar_lvl', 'fringemain_bipolar_lvl', 'inner_com_fringemain_bipolar_lvl', 'mod_edge_prop', 'inner_com_rank', 'inner_com_left_right_relative', 'inner_com_fringeness_relative','inner_com_left_right_center_closeness', 'inner_com_fringeness_center_closeness','country_count','dk_sv_count']
    mf = df.copy()
    for col in drop_cols:
        if col in mf.columns:
            mf = mf.drop([col],axis=1)
    #print (len(mf))
    danish_dups = pd.read_excel(main_path+"/danish_duplicate.xlsx")
    for ap in list(danish_dups["Actor Platform"]):
        mf = mf.drop(mf[(mf["actor_platform"]==ap) & (mf["title"]=="altmed_denmark")].index)
    #print (len(mf))
    mf['iteration0_title'] = np.where( mf['actor_platform'] == 'Den Uafhængige_Facebook Page', None, mf['iteration0_title'])
    mf['iteration0_title'] = np.where( mf['actor_platform'] == 'Frihedsbrevet_Facebook Page', None, mf['iteration0_title'])
    mf['iteration0_title'] = np.where( mf['actor_platform'] == 'Frihedsbrevet_Twitter', None, mf['iteration0_title'])
    mf.to_csv(main_path+"/fringe_mainstream.csv")

def actor_export_default(main_path,title,net_title,nat_langs=["de","de-AT","de-DE"],no_pop_case=False):

    mdb = MongoSpread()
    print ("Creating data for title: {0}...".format(title))
    df = create_actor_data(title,new=False,main_path=main_path,net_title=net_title)
    df2 = None
    if no_pop_case:
        if title == "altmed_denmark":
            df2 = create_actor_data("altmed_denmark_no_pop",new=False,main_path=main_path,net_title="alt_dk_no_pop")
        elif title == "altmed_austria":
            df2 = create_actor_data("altmed_austria_no_pop",new=False,main_path=main_path,net_title="alt_at_no_pop")
    if df2 is not None:
        df = replace_values_in_dataset(df,df2)
    df = df[df["platform"]!="web"]
    df.loc[df["actor_platform"] == "nytimes_Twitter", 'mainstream'] = 1.0

    print ("Loading net...")
    net = nx.read_gexf(main_path+"/"+"{0}_BACKBONED_filtered.gexf".format(net_title))

    print ("Adding special coms.")
    if not "de" in nat_langs:
        df = add_inner_outer_com(df,title,inner_base="com_res=25_small")
    else:
        df = add_inner_outer_com(df,title,inner_base="com_res=09_small")
    if no_pop_case:
        df['mainstream'] = np.where( df['inner_com'] == "austriai_13.0", df['mainstream']+0.35, df['mainstream'])
        df['fringe_mainstream'] = np.where( (df['political_sharp'] == "right") & (df["iteration0_title"]=="altmed_austria"), df['mainstream']+.34, df['mainstream'])

    print ("Adding domain share engagement")
    alt_websites = [LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://","") for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":title.replace("_small","")})]
    df = add_domain_share_engagement(df,"alt_share_engagement",alt_websites,filepath=main_path+"/"+"{0}_dom_share_engagement.p".format(title))
    df = add_domain_share_engagement(df,"alt_share_engagement_total",alt_websites,filepath=main_path+"/"+"{0}_dom_share_engagement.p".format(title),mean=False)
    print ("Adding domain share count")
    df = add_domain_share_count(df,"alt_share_count",alt_websites,var_name="alternative")
    df = add_domain_share_count(df,"main_share_count",list(pd.read_csv(main_path+f"/{title}_mainstream_domains.csv")["domain"]),var_name="mainstream")

    print ("Adding calculated fields.")
    df["interactions_mean_sqrt"]=np.sqrt(df["interactions_mean"])
    interactions_platform_norm = {group:name["interactions_mean_sqrt"] for group,name in df[["interactions_mean_sqrt","platform"]].groupby("platform").mean().iterrows()}
    df["interactions_mean_pl_norm"] = df.apply(add_platform_norm,args=["interactions_mean",interactions_platform_norm],axis=1)

    df["alt_share_engagement_sqrt"]=np.sqrt(df["alt_share_engagement"])
    #alt_interactions_platform_norm = {group:name["alt_share_engagement_sqrt"] for group,name in df[["alt_share_engagement_sqrt","platform"]].groupby("platform").mean().iterrows()}
    df["alt_share_engagement_pl_norm"] = df.apply(add_platform_norm,args=["alt_share_engagement",interactions_platform_norm],axis=1)

    df["fringeness"]=df["grey2"]-df["mainstream"]
    df["mainstreamness"]=df["mainstream"]-df["grey2"]
    #print (df[df["inner_com"]=="austriai_13.0"]["fringeness"].mean())

    df["fringeness_score"]=df["grey2_score"]-df["mainstream_score"]
    df["left_vs_right"] = df.apply(add_collapsed_binary_value,args=[["left","right"]],axis=1)
    df["left_right"]=df["right"]-df["left"]

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
        df["inner_com_alt_share_count"]=df["inner_com"].map(get_mean_grouped(df,"inner_com","alt_share_count"))
        df["inner_com_n_actors"]=df["inner_com"].map(df[["inner_com","actor_platform"]].groupby("inner_com").count().to_dict()["actor_platform"])

    print ("Adding additional label data.")
    df = add_score_based_label(df,"pol_main_fringe_score",[("grey2_score","mainstream_score"),("right","left","other")])
    df = add_score_based_label(df,"pol_main_fringe",[("grey2","mainstream"),("right","left","other")])
    df = add_score_based_label(df,"pol_main_fringe_sharp",[("grey2","mainstream"),("right","left","other")],sharp=True)
    df = add_score_based_label(df,"political_sharp",[("right","left","other")],sharp=True)
    df = add_score_based_label(df,"main_fringe_sharp",[("mainstream","grey2")],sharp=True)
    #net = load_net(title,main_path=main_path)

    print ("Adding com primary categories.")
    icl, icl_overlap = get_primary_category_grouped(df,"inner_com","lang")
    pmfs, pmfs_overlap = get_primary_category_grouped(df,"inner_com","pol_main_fringe_sharp")
    pmf, pmf_overlap = get_primary_category_grouped(df,"inner_com","pol_main_fringe")
    ps, ps_overlap = get_primary_category_grouped(df,"inner_com","political_sharp")
    p, p_overlap = get_primary_category_grouped(df,"inner_com","political")
    fm, fm_overlap = get_primary_category_grouped(df,"inner_com","main_fringe_sharp")
    df["inner_com_lang"]=df["inner_com"].map(icl)
    df["inner_com_pol_main_fringe_sharp"]=df["inner_com"].map(pmfs)
    df["inner_com_pol_main_fringe"]=df["inner_com"].map(pmf)
    df["inner_com_political_sharp"]=df["inner_com"].map(ps)
    df["inner_com_political"]=df["inner_com"].map(p)
    df["inner_com_main_fringe_sharp"]=df["inner_com"].map(fm)
    df["inner_com_lang_overlap"]=df["inner_com"].map(icl_overlap)
    df["inner_com_pol_main_fringe_sharp_overlap"]=df["inner_com"].map(pmfs_overlap)
    df["inner_com_pol_main_fringe_overlap"]=df["inner_com"].map(pmf_overlap)
    df["inner_com_political_sharp_overlap"]=df["inner_com"].map(ps_overlap)
    df["inner_com_main_fringe_sharp_overlap"]=df["inner_com"].map(ps_overlap)
    df["inner_com_main_fringe_sharp_overlap"]=df["inner_com"].map(p_overlap)

    df.loc[df["inner_com"] == "denmarki_13.0", 'inner_com_pol_main_fringe_sharp'] = "grey2_other"
    #df.loc[df["inner_com"] == "austriai_13.0", 'inner_com_pol_main_fringe_sharp'] = "mainstream_right"
    df.loc[df["inner_com"] == "austriai_4.0", 'inner_com_pol_main_fringe_sharp'] = "mainstream_grey"
    df.loc[df["inner_com"] == "swedeno_3", 'inner_com_pol_main_fringe_sharp'] = "grey2_grey"
    df.loc[df["inner_com"] == "swedeni_4.0", 'inner_com_pol_main_fringe_sharp'] = "grey2_other"
    df.loc[df["inner_com"] == "swedeni_64.0", 'inner_com_pol_main_fringe_sharp'] = "grey2_right"
    df.loc[df["inner_com"] == "swedeni_24.0", 'inner_com_pol_main_fringe_sharp'] = "grey2_other"
    df.loc[df["inner_com"] == "swedeni_3.0", 'inner_com_pol_main_fringe_sharp'] = "mainstream_right"
    df.loc[df["inner_com"] == "swedeni_23.0", 'inner_com_pol_main_fringe_sharp'] = "grey2_right"

    print ("Adding distance metrics.")
    df = add_value_distances(df,"political_polar_lvl",["right","left","other"])
    df = add_value_distances(df,"political_bipolar_lvl",["right","left"])
    df = add_value_distances(df,"fringemain_bipolar_lvl",["grey2","mainstream"])
    df = add_value_distances(df,"inner_com_fringemain_bipolar_lvl",["inner_com_grey2","inner_com_mainstream"])

    print ("Adding ranks.")
    df = add_rank_binary(df,"inner_com",net=net)
    print ("Adding show labels.")
    show_labels = set(list(df[(df["inner_com_rank"]>0) | (df["iteration0_title"]==title)]["actor_platform"]))
    df["show_label"]=df["actor_platform"].map({row["actor_platform"]:True if row["actor_platform"] in show_labels else False for i,row in df.iterrows()})

    show_inner_com = set(list(df[(df["inner_com_lang"].isin(set(["da","en","sv","de","no"]))) & (~df["inner_com"].isin(set(["swedeni_13.0"]))) & (df["inner_com_pol_main_fringe_sharp"]!="grey2_grey") & (df["inner_com_alt_share_count"]>=3) & (df["inner_com_n_actors"]>=25)]["inner_com"]))
    print (len(show_inner_com))
    show_inner_com.update(set(list(df[df["inner_com_nat_it0_count"]>2]["inner_com"])))
    show_inner_com.update(set(list(df[df["inner_com"].isin(set(["swedeni_24.0","swedeni_4.0"]))]["inner_com"])))
    print (len(show_inner_com))
    df["show_inner_com"]=df["actor_platform"].map({row["actor_platform"]:True if row["inner_com"] in show_inner_com else False for i,row in df.iterrows()})
    df.loc[(df["show_inner_com"] == True) & (df["inner_com_pol_main_fringe_sharp"] == "grey2_grey"), 'inner_com_pol_main_fringe_sharp'] = df["inner_com_pol_main_fringe"]

    df["inner_com_left_right_relative"]=df["actor_platform"].map(get_com_relative_dif_score(df,"inner_com",["left_right"]))
    df["inner_com_fringeness_relative"]=df["actor_platform"].map(get_com_relative_dif_score(df,"inner_com",["fringeness"]))
    df["left_right_center_closeness"]=df["actor_platform"].map(get_closeness_to_center(df,["left_right"]))
    df["fringeness_center_closeness"]=df["actor_platform"].map(get_closeness_to_center(df,["fringeness"]))
    #df["inner_com_inner_com_left_right_relative"]=df["inner_com"].map(get_mean_grouped(df,"inner_com","inner_com_left_right_relative"))
    #df["inner_com_inner_com_fringeness_relative"]=df["inner_com"].map(get_mean_grouped(df,"inner_com","inner_com_fringeness_relative"))
    df["inner_com_left_right_center_closeness"]=df["inner_com"].map(get_mean_grouped(df,"inner_com","left_right_center_closeness"))
    df["inner_com_fringeness_center_closeness"]=df["inner_com"].map(get_mean_grouped(df,"inner_com","fringeness_center_closeness"))
    df["inner_com_left_right_center_closeness"]=df["actor_platform"].map(get_closeness_to_center(df,["inner_com_left_right_center_closeness"]))
    df["inner_com_fringeness_center_closeness"]=df["actor_platform"].map(get_closeness_to_center(df,["inner_com_fringeness_center_closeness"]))

    df["rank_score_alt"]=np.sqrt(np.log(df["rank_score"]+1.0))*np.log(df["alt_share_count"]+1.0)*np.log(df["alternative"]+1.0)
    df["political_sharp_safe"]=df["political_sharp"]
    df['political_sharp_safe'] = np.where( (df['political_sharp'] == 'grey') & (df['political'] == 'other' ), "other", df['political_sharp'])
    df["alt_share_count_proportion"]=df["alt_share_count"]/df["n_posts"]
    df['n_posts'] = np.where( df['alt_share_count_proportion'] > 1.0, df['alt_share_count_proportion']*df['n_posts'], df['n_posts'])
    df["alt_share_count_proportion"]=df["alt_share_count"]/df["n_posts"]
    df["main_share_count_proportion"]=df["main_share_count"]/df["n_posts"]
    df['n_posts'] = np.where( df['main_share_count_proportion'] > 1.0, df['main_share_count_proportion']*df['n_posts'], df['n_posts'])
    df["main_share_count_proportion"]=df["main_share_count"]/df["n_posts"]
    #print (df[["unique_alt_domains"]])
    #print (df[["one_alt_domain"]])
    #print (df[["two_alt_domain"]])

    df.to_csv(main_path+"/"+"{0}_final.csv".format(title),index=False)

def actor_net_export_default(main_path,title,net_title,export_path="/home/alterpublics/projects/altmed"):

    def reset_node_attributes(g):

        for n in list(g.nodes()):
            for att,v in dict(g.nodes[n]).items():
                del g.nodes[n][att]
        return g

    def find_most_representative(df,att_name):

        counts = df[[att_name,"actor_platform"]].groupby(att_name).count().to_dict()["actor_platform"]
        label = sorted(counts.items(), key = itemgetter(1), reverse=True)[0][0]
        if len(counts) > 1:
            if label.lower() == "und" or label.lower() == "grey2_grey":
                label = sorted(counts.items(), key = itemgetter(1), reverse=True)[1][0]
        return label

    def modular_core(df,com_var,num_selection=[],class_selection=[],shave_prop=0.02,min_sample=10,min_com=0.01,exclude=[]):

        get_actors = set([])
        all_n_actors = len(df)
        all_platforms = set(list(df["platform"]))
        for com in set(list(df[com_var])):
            if com in exclude: continue
            f_df = df[df[com_var]==com]
            n_actors = len(f_df)
            if int(n_actors) > int(all_n_actors*min_com):
                if int(n_actors*shave_prop) < min_sample:
                    pluck = min_sample
                else:
                    pluck = int(n_actors*shave_prop)

                for select in class_selection:
                    label = find_most_representative(f_df,select)
                    get_actors.update(set(list(f_df[f_df[select]==label].sort_values("degrees",ascending=False,inplace=False).head(pluck)["actor_platform"])))

                f_df = f_df.sort_values("degrees",ascending=False,inplace=False).head(int(n_actors*0.9))
                for select in num_selection:
                    get_actors.update(set(list(f_df.sort_values(select,ascending=False,inplace=False).head(pluck)["actor_platform"])))

        for plat in all_platforms:
            get_actors.update(set(list(df[df["platform"]==plat].sort_values("degrees",ascending=False,inplace=False).head(int(all_n_actors*(0.02*shave_prop)))["actor_platform"])))

        return get_actors

    def get_modular_labels(df,net,com_var,score=["degrees","inner_com_edge_prop"],pluck=0.015):

        labels_true = set([])
        value_dict = {}
        df["score"]=np.log(df[score[0]])*df[score[1]]
        nodes_in_net = list(net.nodes())
        for com in set(list(df[com_var])):
            f_df = df[(df[com_var]==com) & df["actor_platform"].isin(set(nodes_in_net))]
            n_com_actors = len(f_df)
            labels_true.update(set(list(f_df.sort_values("score",ascending=False,inplace=False).head(int(pluck*n_com_actors))["actor_platform"])))
        for com in set(list(df["pol_main_fringe_sharp"])):
            f_df = df[(df["pol_main_fringe_sharp"]==com) & df["actor_platform"].isin(set(nodes_in_net))]
            n_com_actors = len(f_df)
            labels_true.update(set(list(f_df.sort_values("score",ascending=False,inplace=False).head(int(pluck*n_com_actors)+1)["actor_platform"])))

        for n in list(net.nodes()):
            if n in labels_true:
                value_dict[n]=True
            else:
                value_dict[n]=False

        return value_dict

    df = pd.read_csv(main_path+"/"+"{0}_final.csv".format(title))
    df["log_alt_share_count"]=np.log(df["alt_share_count"]+1)
    df["cent_score"]=np.sqrt(df["log_alt_share_count"]*(1-df["grey1"])+1)
    df['cent_score'] = df['cent_score'].replace(np.nan, 0)
    net = nx.read_gexf(main_path+"/"+"{0}_BACKBONED_filtered.gexf".format(net_title))
    nx.set_node_attributes(net,df.set_index("actor_platform")[["inner_com"]].to_dict()["inner_com"],name="inner_com")
    nx.set_node_attributes(net,df.set_index("actor_platform")[["outer_com"]].to_dict()["outer_com"],name="outer_com")

    print ("getting modular edge proportions.")
    df["inner_com_edge_prop"]=df["actor_platform"].map(get_modular_edge_prop(net,"inner_com"))
    df["inner_com_edge_prop_score"]=df["inner_com_edge_prop"]*np.log(df["degrees"])

    if "denmark" in title:
        actor_filter = modular_core(df,"inner_com",shave_prop=0.10,num_selection=["degrees","interactions_mean_pl_norm","btw_cen","inner_com_edge_prop_score","alt_share_count","inner_com_edge_prop"],class_selection=["lang","pol_main_fringe","pol_main_fringe_sharp","political_sharp","main_fringe"])
        actor_filter.update(modular_core(df,"outer_com",shave_prop=0.75,exclude=["o_0"],num_selection=["degrees","interactions_mean_pl_norm","btw_cen","inner_com_edge_prop_score","alt_share_count","inner_com_edge_prop"],class_selection=["lang","pol_main_fringe","pol_main_fringe_sharp","political_sharp","main_fringe"]))
    elif "germany" in title:
        actor_filter = modular_core(df,"inner_com",shave_prop=0.2,num_selection=["degrees","interactions_mean_pl_norm","btw_cen","inner_com_edge_prop_score","alt_share_count","inner_com_edge_prop"],class_selection=["lang","pol_main_fringe","pol_main_fringe_sharp","political_sharp","main_fringe"])
    else:
        actor_filter = modular_core(df,"inner_com",shave_prop=0.14,num_selection=["degrees","interactions_mean_pl_norm","btw_cen","inner_com_edge_prop_score","alt_share_count","inner_com_edge_prop"],class_selection=["lang","pol_main_fringe","pol_main_fringe_sharp","political_sharp","main_fringe"])
    #actor_filter.update(modular_core(df,"outer_com",shave_prop=0.05,num_selection=["degrees","interactions_mean_pl_norm","btw_cen","inner_com_edge_prop","alt_share_count"],class_selection=["lang","pol_main_fringe","pol_main_fringe_sharp","political_sharp","main_fringe"]))
    actor_filter.update(set(list(df[df["iteration0_title"]==title]["actor_platform"])))
    net.remove_nodes_from([n for n in net.nodes() if n not in actor_filter])
    net = NetworkUtils().giant_component(net)
    net = reset_node_attributes(net)

    pol_main_fringe_sharp = df.set_index("actor_platform")[["pol_main_fringe_sharp"]].to_dict()["pol_main_fringe_sharp"]
    nx.set_node_attributes(net,pol_main_fringe_sharp,name="pol_main_fringe_sharp")

    nx.set_node_attributes(net,df.set_index("actor_platform")[["inner_com"]].to_dict()["inner_com"],name="inner_com")
    nx.set_node_attributes(net,df.set_index("actor_platform")[["outer_com"]].to_dict()["outer_com"],name="outer_com")
    nx.set_node_attributes(net,df.set_index("actor_platform")[["platform"]].to_dict()["platform"],name="platform")
    nx.set_node_attributes(net,df.set_index("actor_platform")[["lang"]].to_dict()["lang"],name="lang")
    nx.set_node_attributes(net,df.set_index("actor_platform")[["pol_main_fringe"]].to_dict()["pol_main_fringe"],name="pol_main_fringe")
    nx.set_node_attributes(net,df.set_index("actor_platform")[["iteration0_title"]].to_dict()["iteration0_title"],name="iteration0_title")
    nx.set_node_attributes(net,df.set_index("actor_platform")[["interactions_mean_pl_norm"]].to_dict()["interactions_mean_pl_norm"],name="interactions_mean_pl_norm")
    nx.set_node_attributes(net,df.set_index("actor_platform")[["degrees"]].to_dict()["degrees"],name="degrees")
    nx.set_node_attributes(net,df.set_index("actor_platform")[["actor_name"]].to_dict()["actor_name"],name="Label")
    nx.set_node_attributes(net,df.set_index("actor_platform")[["log_alt_share_count"]].to_dict()["log_alt_share_count"],name="log_alt_share_count")
    nx.set_node_attributes(net,df.set_index("actor_platform")[["cent_score"]].to_dict()["cent_score"],name="cent_score")
    nx.set_node_attributes(net,df.set_index("actor_platform")[["inner_com_pol_main_fringe_sharp"]].to_dict()["inner_com_pol_main_fringe_sharp"],name="inner_com_pol_main_fringe_sharp")

    nx.set_node_attributes(net,get_modular_labels(df,net,"inner_com"),"inner_com_label")
    nx.set_node_attributes(net,get_modular_labels(df,net,"outer_com"),"outer_com_label")

    print (len(net.nodes()))
    print (len(net.edges()))
    nx.write_gexf(net,export_path+"/"+"{0}_net_final.gexf".format(net_title))
    #sys.exit()

def combine_exports(main_path,titles,export_path=""):

    print ("Combining actor files...")
    dfs = []
    if not isinstance(main_path,list):
        main_path = [main_path for r in range(len(titles))]
    for path,title in zip(main_path,titles):
        df = pd.read_csv(path+"/"+title+"_final.csv")
        df["title"]=title
        dfs.append(df)
    final_df = pd.concat(dfs,axis=0)
    country_count = {}
    dk_sv_count = {}
    for i,row in final_df.iterrows():
        ap = row["actor_platform"]
        if ap not in country_count: country_count[ap]=0
        if ap not in dk_sv_count: dk_sv_count[ap]=0
        if row["title"] in titles:
            country_count[ap]+=1
        if row["title"] in set(["altmed_denmark","altmed_sweden"]):
            dk_sv_count[ap]+=1
    final_df["country_count"]=final_df["actor_platform"].map(country_count)
    final_df["dk_sv_count"]=final_df["actor_platform"].map(dk_sv_count)

    final_df.to_csv(export_path,index=False)

def actor_export_project_default(main_path,org_project_titles,nat_langs=[],limit=1,exclude=[],add_webs=[]):

    df = []
    mdb = MongoSpread()
    query = {"org_project_title":{"$in":org_project_titles},"Iteration":0}
    if len(org_project_titles) > 1:
        org_project_title = org_project_titles[0] + "___"+org_project_titles[-1]
    else:
        org_project_title=org_project_titles[0]
    webs = set([])
    actors = set([])
    unique_mids = set([])
    csv_counter = 0
    for d in mdb.database["actor"].find(query):
        if d["Actor"] not in exclude:
            actors.add(d["Actor"])
        if d["Website"] not in exclude:
            webs.add(d["Website"])
    if len(add_webs) > 0:
        webs.update(set(add_webs))
    for web in webs:
        domain = LinkCleaner().strip_backslash(LinkCleaner().remove_url_prefix(web))
        print (domain)
        if domain is not None:
            shares = [(d["message_ids"],d["actor_platform"],d["url"]) for d in mdb.database["url_bi_network"].find({"domain":domain},{"message_ids":1,"actor_platform":1,"url":1}).limit(limit)]
            for url_share,ap,real_url in shares:
                for mid in url_share:
                    if mid not in unique_mids:
                        doc = mdb.database["post"].find_one({"message_id":mid})
                        doc = get_platform_uniform_post(doc)
                        doc["actor_platform"]=ap
                        doc["link"]=real_url
                        df.append(doc)
                        unique_mids.add(mid)
                    if len(df) > 500000000:
                        pd.DataFrame(df).to_csv(main_path+"/{0}_post_export_{1}.csv".format(org_project_title,csv_counter),index=False)
                        df = []
                        csv_counter+=1
                    if len(df) % 100000 == 0:
                        print (len(df))
    print ()
    print ("ACTORS")
    for actor in actors:
        print (actor)
        posts = [(d["post_obj_ids"],d["actor_platform"]) for d in mdb.database["actor_platform_post"].find({"actor":actor},{"post_obj_ids":1,"actor_platform":1}).limit(limit)]
        for obj_ids,ap in posts:
            for obj_id in obj_ids:
                doc = mdb.database["post"].find_one({"_id":obj_id})
                if doc["message_id"] not in unique_mids:
                    unique_mids.add(doc["message_id"])
                    doc = get_platform_uniform_post(doc)
                    doc["actor_platform"]=ap
                    df.append(doc)
                if len(df) > 500000000:
                    pd.DataFrame(df).to_csv(main_path+"/{0}_post_export_{1}.csv".format(org_project_title,csv_counter),index=False)
                    df = []
                    csv_counter+=1
                if len(df) % 100000 == 0:
                    print (len(df))
    pd.DataFrame(df).to_csv(main_path+"/{0}_post_export_{1}.csv".format(org_project_title,csv_counter),index=False)

def link_full_text_export(main_path,title,file_path):

    mdb = MongoSpread()
    df = pd.read_csv(file_path)
    new_df = []
    all_links = list(set(list(df["link"])))
    print (len(all_links))
    for link in all_links:
        for doc in mdb.database["url_texts"].find({"url":link,"succes":True}):
            new_df.append({"link":doc["url"],"full_text":doc["text"]})
            if len(new_df) % 10000 == 0:
                print (len(new_df))
    pd.DataFrame(new_df).to_csv(main_path+"/{0}_full_texts.csv".format(title),index=False)

def cluster_analysis(main_path):

    def lookup_dom_profile(df,com,limit=15,no_yt=True,metric="tf_idf"):

        if "austria" in com:
            df=df[df["inner_com"]==com]
            if no_yt: df = df[df["domain"].str.contains("youtube.com")==False]
            prim = df.sort_values(metric,ascending=False,inplace=False).head(int(limit*(2/3)))
            sec = df[(df["domain"].str.contains(".at")==True) & (~df["domain"].isin(set(list(prim["domain"]))))].sort_values(metric,ascending=False,inplace=False).head(int(limit*(1/3)))
            return pd.concat([df[df["inner_com"]==com].sort_values(metric,ascending=False,inplace=False).head(int(limit*(2/3))),\
                      sec],axis=0).head(limit)
        else:

            if no_yt: df = df[df["domain"].str.contains("youtube.com")==False]
            return df[df["inner_com"]==com].sort_values(metric,ascending=False,inplace=False).head(limit)

    def lookup_central_profile(df,com,metric="log_rank_score",limit=15):

        df=df[df["inner_com"]==com]
        prim = df[["actor_name","platform",metric,"link_to_actor","title","inner_com_pol_main_fringe_sharp","alt_share_count"]].sort_values(metric,ascending=False,inplace=False).head(limit)

        return prim

    mdb = MongoSpread()
    mdoms = {"altmed_sweden":list(pd.read_csv(main_path+"/altmed_sweden_mainstream_domains.csv")["domain"]),"altmed_denmark":list(pd.read_csv(main_path+"/altmed_denmark_mainstream_domains.csv")["domain"]),"altmed_germany":[],"altmed_austria":[]}
    adoms = {}
    adoms["altmed_denmark"]=[LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://","") for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"altmed_denmark".replace("_small","")})]
    adoms["altmed_sweden"]=[LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://","") for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"altmed_sweden".replace("_small","")})]
    adoms["altmed_germany"]=[LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://","") for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"altmed_germany".replace("_small","")})]
    adoms["altmed_austria"]=[LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://","") for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"altmed_austria".replace("_small","")})]

    df = pd.read_csv(main_path+"/fringe_mainstream.csv")

    df["left_right"]=df["right"]-df["left"]
    df["fringe_mainstream"]=df["mainstream"]-df["grey2"]
    df["inner_com_n_actors_sqrt"]=np.sqrt(df["inner_com_n_actors"])
    df["log_rank_score"]=np.sqrt(np.log(df["rank_score"]+1))
    df["log_alt_share_count"]=np.log(df["alt_share_count"]+1)
    df["log_alt_share_enga"]=np.log(df["alt_share_engagement"]+1)
    df["log_rank_score"]=df["log_rank_score"].fillna(0)
    df["rank_score_alt"]=np.sqrt(np.log(df["rank_score"]+1.0))*np.log(df["alternative"]+1.0)
    df["log_degrees"]=np.log(df["degrees"]+1)

    df['inner_com_pol_main_fringe_sharp'].replace('mainstream_grey','mainstream',inplace=True)
    df['inner_com_pol_main_fringe_sharp'].replace('grey2_left','fringe_left',inplace=True)
    df['inner_com_pol_main_fringe_sharp'].replace('grey2_right','fringe_right',inplace=True)
    df['inner_com_pol_main_fringe_sharp'].replace('grey2_other','fringe_other',inplace=True)
    df['pol_main_fringe_sharp'].replace('mainstream_grey','mainstream',inplace=True)
    df['pol_main_fringe_sharp'].replace('grey2_left','fringe_left',inplace=True)
    df['pol_main_fringe_sharp'].replace('grey2_right','fringe_right',inplace=True)
    df['pol_main_fringe_sharp'].replace('grey2_other','fringe_other',inplace=True)

    df = df[df["show_inner_com"]==True]
    dom_df = pd.read_csv(main_path+"/all_actors_domains_per_inner_com_tfidf10.csv")

    inner_com_pol = {row["inner_com"]:row["inner_com_pol_main_fringe_sharp"] for i,row in df.iterrows()}

    com_cols = ["cluster","inner_com_pol_main_fringe_sharp","domain","actor","link_to_actor","rank","domain_tf_idf_score","domain_times_shared","actor_rank_score","alt_share_count","title"]
    cluster_cols = ["cluster","title","pol_main_fringe_sharp"]
    all_rows = []
    crows = []
    #for com in set(list(df[df["title"]=="altmed_denmark"]["inner_com"])):
    for com in set(list(df["inner_com"])):
        print (com)
        pol_f_m = df[df["inner_com"]==com].iloc[0]["inner_com_pol_main_fringe_sharp"]
        if "denmark" in com: ctitle = "altmed_denmark"
        if "germany" in com: ctitle = "altmed_germany"
        if "sweden" in com: ctitle = "altmed_sweden"
        if "austria" in com: ctitle = "altmed_austria"
        com_rows = []
        doms = lookup_dom_profile(dom_df,com,limit=5000000,metric="n_shared")
        actors = lookup_central_profile(df,com,metric="rank_score_alt",limit=50000000)
        longest = 0
        if len(actors) >= len(doms): longest = len(actors)
        if len(doms) >= len(actors): longest = len(doms)
        if len(actors) >= 15:
            for r in range(longest):
                if r+1 > len(doms):
                    com_rows.append([com,actors.iloc[r]["inner_com_pol_main_fringe_sharp"],"",actors.iloc[r]["actor_name"],actors.iloc[r]["link_to_actor"],r+1,0,0,actors.iloc[r]["rank_score_alt"],actors.iloc[r]["alt_share_count"],actors.iloc[r]["title"]])
                elif r+1 > len(actors):
                    com_rows.append([com,pol_f_m,doms.iloc[r]["domain"],"","",r+1,doms.iloc[r]["tf_idf"],doms.iloc[r]["n_shared"],0.0,0.0,ctitle])
                else:
                    com_rows.append([com,actors.iloc[r]["inner_com_pol_main_fringe_sharp"],doms.iloc[r]["domain"],actors.iloc[r]["actor_name"],actors.iloc[r]["link_to_actor"],r+1,doms.iloc[r]["tf_idf"],doms.iloc[r]["n_shared"],actors.iloc[r]["rank_score_alt"],actors.iloc[r]["alt_share_count"],actors.iloc[r]["title"]])
            all_rows.extend(com_rows)
        crows.append([com,ctitle,inner_com_pol[com]])

    crows = pd.DataFrame(crows,columns=cluster_cols)
    all_rows = pd.DataFrame(all_rows,columns=com_cols)

    for var_name in ["mainstream","alternative"]:
        print (var_name)
        if var_name == "mainstream": idoms = mdoms
        if var_name == "alternative": idoms = adoms
        counts = {}
        un_counts = {}
        one_pl_dom = {}
        two_pl_dom = {}
        three_pl_dom = {}
        top_first_dom = {}
        top_second_dom = {}
        top_third_dom = {}
        un_dom = {}
        for a in set(list(crows["cluster"])):
            if a not in counts: counts[a]=0.0
            if a not in un_dom: un_dom[a]=0.0
            if a not in one_pl_dom: one_pl_dom[a]=0.0
            if a not in two_pl_dom: two_pl_dom[a]=0.0
            if a not in three_pl_dom: three_pl_dom[a]=0.0
            if a not in un_counts: un_counts[a]={}
            if a not in top_first_dom: top_first_dom[a]=0.0
            if a not in top_second_dom: top_second_dom[a]=0.0
            if a not in top_third_dom: top_third_dom[a]=0.0
        for i,row in crows.iterrows():
            tf_idf_sum = 0
            for i2,row2 in all_rows[all_rows["cluster"]==row["cluster"]].iterrows():
                tf_idf_sum+=row2["domain_tf_idf_score"]**2
                if row2["cluster"] in counts and row2["domain"] in idoms[row["title"]]:
                    counts[row2["cluster"]]+=row2["domain_tf_idf_score"]**2
                if row2["cluster"] in un_counts and row2["domain"] in idoms[row["title"]]:
                    if row2["domain"] not in un_counts[row2["cluster"]]:
                        un_counts[row2["cluster"]][row2["domain"]]=0.0
                    un_counts[row2["cluster"]][row2["domain"]]+=row2["domain_tf_idf_score"]**2
            if tf_idf_sum > 0:
                counts[row2["cluster"]]=float(counts[row2["cluster"]])/float(tf_idf_sum)
            else:
                print (row2["cluster"])
        for k,v in un_counts.items():
            if len(v) > 0:
                un_dom[k]=len(v)
                sorted_dom_dist = sorted(v.items(), key = itemgetter(1), reverse=True)
                one_pl_dom[k]=np.sum(np.array([sorted_dom_dist[0][1]]))/np.sum(np.array(list(v.values())))
                top_first_dom[k]=sorted_dom_dist[0][0]
                if len(v) < 2:
                    two_pl_dom[k]=np.sum(np.array([sorted_dom_dist[0][1]]))/np.sum(np.array(list(v.values())))
                    three_pl_dom[k]=np.sum(np.array([sorted_dom_dist[0][1]]))/np.sum(np.array(list(v.values())))
                else:
                    top_second_dom[k]=sorted_dom_dist[1][0]
                    two_pl_dom[k]=np.sum(np.array([sorted_dom_dist[0][1],sorted_dom_dist[1][1]]))/np.sum(np.array(list(v.values())))
                if len(v) > 2:
                    top_third_dom[k]=sorted_dom_dist[2][0]
                    three_pl_dom[k]=np.sum(np.array([sorted_dom_dist[0][1],sorted_dom_dist[1][1],sorted_dom_dist[2][1]]))/np.sum(np.array(list(v.values())))
        crows["{0}_share".format(var_name)]=crows["cluster"].map(counts)
        crows["one_{0}_domain".format(var_name)]=crows["cluster"].map(one_pl_dom)
        crows["two_{0}_domain".format(var_name)]=crows["cluster"].map(two_pl_dom)
        crows["three_{0}_domain".format(var_name)]=crows["cluster"].map(three_pl_dom)
        crows["unique_{0}_domains".format(var_name)]=crows["cluster"].map(un_dom)
        crows["top_first_{0}_dom".format(var_name)]=crows["cluster"].map(top_first_dom)
        crows["top_second_{0}_dom".format(var_name)]=crows["cluster"].map(top_second_dom)
        crows["top_third_{0}_dom".format(var_name)]=crows["cluster"].map(top_third_dom)

    with pd.ExcelWriter(main_path+"/top_domains_actors_per_com.xlsx") as writer:
        all_rows.to_excel(writer,index=False,sheet_name='clusters_and_actors')
        crows.to_excel(writer,index=False,sheet_name='clusters')
