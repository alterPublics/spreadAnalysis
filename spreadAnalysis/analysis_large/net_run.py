import networkit as nxk
import networkx as nx
import pandas as pd
import time
import numpy as np
np.seterr(all="ignore")
from multiprocessing import Pool
import multiprocessing
from itertools import chain
import polars as pl
import datetime
import psutil
import random
from collections import Counter
#from pyvis.network import Network
import os
import sys
from spreadAnalysis.analysis_large import net_tools as nt
import spreadAnalysis.analysis.batching as bsc
import plotly.graph_objects as go
import distinctipy
from collections import defaultdict
from scipy.stats import iqr, kurtosis

def get_alt_spread_actor(main_path,title,path_step):

    dom_per_actor = pl.read_csv(main_path+f"/{title}_{path_step}steps_dom_per_actor.csv")
    df = pl.read_csv(main_path+f"/{title}_{path_step}steps_data.csv")
    rank_df = pl.read_csv(main_path+f"/{title}_{path_step}steps_com_1.0_rank_data.csv")

    u_anm_doms = set(rank_df["domain_anm"].to_list())
    dom_per_actor = dom_per_actor.filter(pl.col("domain").is_in(u_anm_doms))

    df = df.with_columns((np.log(df["pagerank_wEnga^2"]+2)*np.log(df["alt_share"]+2)).alias("alt_influence"))
    df = df.sort(["alt_influence"], descending=True)
    df = df.filter((pl.col("custom")>0) & (pl.col("actor_platform").is_in(set(dom_per_actor["actor_platform"].to_list()))))
    df = df.head(100)
    80
    38

    unique_anm_per_actor = {}
    anm_spread_1 = {}
    anm_spread_2 = {}
    anm_spread_3 = {}
    anm_spread = {}
    for actor in df["actor_platform"].to_list():
        filtered = dom_per_actor.filter(pl.col("actor_platform")==actor)
        filtered = filtered.sort(["count"], descending=True)
        anm_shared = sum(filtered["count"].to_list())
        unique_anm_per_actor[actor]=len(filtered)
        anm_spread_1[actor]=sum(filtered.head(1)["count"].to_list())/anm_shared
        anm_spread_2[actor]=sum(filtered.head(2)["count"].to_list())/anm_shared
        anm_spread_3[actor]=sum(filtered.head(3)["count"].to_list())/anm_shared
        arr = filtered["count"].to_list()
        anm_spread[actor]=np.std((arr - np.min(arr)) / (np.max(arr) - np.min(arr)))
    df = df.with_columns(pl.col("actor_platform").map_dict(unique_anm_per_actor).alias("unique_anm_per_actor"))
    df = df.with_columns(pl.col("actor_platform").map_dict(anm_spread_1).alias("anm_spread_1"))
    df = df.with_columns(pl.col("actor_platform").map_dict(anm_spread_2).alias("anm_spread_2"))
    df = df.with_columns(pl.col("actor_platform").map_dict(anm_spread_3).alias("anm_spread_3"))
    df = df.with_columns(pl.col("actor_platform").map_dict(anm_spread).alias("anm_spread"))
    
    print (df)
    print (np.nanmean(df["anm_spread"].to_numpy()))
    df.write_csv(main_path+f"/{title}_{path_step}steps_top_influ.csv")


def create_multi_step_node_colors(main_path,title,path_steps,com_var="com",add_to_all=True):

    pls = ["twitter","facebook","instagram","telegram","gab","youtube","fourchan","vkontakte","reddit"]
    step_gs = {}
    step_dats = {}
    step_g_com_cols = {s:{} for s in path_steps}
    step_g_com_sim_vals = {s:{} for s in path_steps}
    step_g_com_sim_coms = {s:{} for s in path_steps}
    for path_step in path_steps:
        g = nx.read_gexf(main_path+f"/{title}_{path_step}steps_{com_var}.gexf")
        dat = pl.read_csv(main_path+f"/{title}_{path_step}steps_data.csv").filter(pl.col(com_var).is_in(set([int(n) for n in list(g.nodes())])))
        step_gs[path_step]=g
        step_dats[path_step]=dat
    n_init_com_cols = sorted([len(list(step_gs[path_step].nodes())) for path_step in path_steps])[-1]+10
    langs = set(list([d["lang"] for n,d in step_gs[max(path_steps)].nodes(data=True)]))
    for path_step in path_steps:
        langs.update(set(list([d["lang"] for n,d in step_gs[path_step].nodes(data=True)])))
    langs = list(langs)
    init_com_cols = distinctipy.get_colors(n_init_com_cols)
    pl_cols = [c for c in distinctipy.get_colors(len(pls)+2,init_com_cols) if not distinctipy.get_hex(c).lower() in ["#FFFFFF".lower(),"#000000".lower()]]
    lang_cols = [c for c in distinctipy.get_colors(len(langs)+2,init_com_cols+pl_cols) if not distinctipy.get_hex(c).lower() in ["#FFFFFF".lower(),"#000000".lower()]]

    pl_cols = {pls[i]:pl_cols[i] for i in range(len(pls))}
    lang_cols = {langs[i]:lang_cols[i] for i in range(len(langs))}

    step_g_com_cols[min(path_steps)]={int(n):init_com_cols.pop(0) for i,n in enumerate(list(step_gs[min(path_steps)].nodes()))}

    step_dats[min(path_steps)]=step_dats[min(path_steps)].groupby(com_var, maintain_order=True).agg(pl.col("actor_platform"))
    rev_sim = defaultdict(dict)
    for path_step in path_steps[1:]:
        sim_to_prev = defaultdict(dict)
        step_dats[path_step]=step_dats[path_step].groupby(com_var, maintain_order=True).agg(pl.col("actor_platform"))
        c_l = list(zip(step_dats[path_step-1][com_var].to_list(),step_dats[path_step-1]["actor_platform"].to_list()))
        p_l = list(zip(step_dats[path_step][com_var].to_list(),step_dats[path_step]["actor_platform"].to_list()))
        c_l_size = sum([len(l) for c,l in c_l])
        p_l_size = sum([len(l) for c,l in p_l])
        for com,l in c_l:
            for pcom,pl_ in p_l:
                overlap = float(float((len(set(l).intersection(set(pl_)))))/float(c_l_size))/float(float((len(set(l).union(set(pl_)))))/float(p_l_size))
                sim_to_prev[com].update({pcom:overlap})
        has_seen_com = set([])
        while True:
            added_one = False
            for com in list(sim_to_prev.keys()):
                for sim_com,sim_val in sorted(sim_to_prev[com].items(), key=lambda x:x[1], reverse=True):
                    if com not in has_seen_com and sim_com not in step_g_com_cols[path_step]:
                        step_g_com_cols[path_step][sim_com]=step_g_com_cols[path_step-1][com]
                        has_seen_com.add(com)
                        added_one = True

                    rev_sim[sim_com][com]=sim_val
            if not added_one: break
        #print (len(list(sim_to_prev.keys())))
        #print (len(step_g_com_cols[path_step]))
        #print (len(init_com_cols))
        for sim_com,coms in rev_sim.items():
            if sorted(coms.items(), key=lambda x:x[1], reverse=True)[0][1] > 0:
                step_g_com_sim_coms[path_step][sim_com]=sorted(coms.items(), key=lambda x:x[1], reverse=True)[0][0]
                step_g_com_sim_vals[path_step][sim_com]=sorted(coms.items(), key=lambda x:x[1], reverse=True)[0][1]
            else:
                step_g_com_sim_coms[path_step][sim_com]=-1
                step_g_com_sim_vals[path_step][sim_com]=0.0
        if len(step_g_com_cols[path_step]) < len(p_l):
            for rest_com in step_dats[path_step][com_var].to_list():
                if rest_com not in step_g_com_cols[path_step]:
                    step_g_com_cols[path_step][rest_com]=init_com_cols.pop(0)

    for path_step in path_steps:
        com_data = {}
        for node,d in step_gs[path_step].nodes(data=True):
            node = int(node)
            com_data[str(node)]={"master color":distinctipy.get_hex(step_g_com_cols[path_step][node]),
                            "com_col":distinctipy.get_hex(step_g_com_cols[path_step][node]),
                            "platform_col":distinctipy.get_hex(pl_cols[d["platform"]]),
                            "lang_col":distinctipy.get_hex(lang_cols[d["lang"]])}
            if path_step != min(path_steps):
                com_data[str(node)]["most_similar_prev_com"]=step_g_com_sim_coms[path_step][node]
                com_data[str(node)]["most_similar_overlap"]=step_g_com_sim_vals[path_step][node]
                if path_step != min(path_steps):
                    if step_g_com_sim_coms[path_step][node] == -1 or step_g_com_sim_coms[path_step][node] not in step_g_com_cols[path_step-1]:
                        com_data[str(node)]["most_similar_prev_com_color"]="#c0c0c0"
                    else:
                        com_data[str(node)]["most_similar_prev_com_color"]=distinctipy.get_hex(step_g_com_cols[path_step-1][step_g_com_sim_coms[path_step][node]])
        nx.set_node_attributes(step_gs[path_step],com_data)
        nx.write_gexf(step_gs[path_step],main_path+f"/{title}_{path_step}steps_{com_var}.gexf")

    if add_to_all:
        for path_step in path_steps:
            a_g = nx.read_gexf(main_path+f"/{title}_{path_step}steps.gexf")
            for n,d in list(a_g.nodes(data=True)):
                if "platform" in d:
                    nx.set_node_attributes(a_g,{n:distinctipy.get_hex(pl_cols[d["platform"]])},"platform_col")
                if "lang" in d:
                    if d["lang"] in lang_cols:
                        nx.set_node_attributes(a_g,{n:distinctipy.get_hex(lang_cols[d["lang"]])},"lang_col")
                    else:
                        nx.set_node_attributes(a_g,{n:"#808080"},"lang_col")
                if com_var in d:
                    nx.set_node_attributes(a_g,{n:distinctipy.get_hex(step_g_com_cols[path_step][int(d[com_var])])},"com_col")
            nx.write_gexf(a_g,main_path+f"/{title}_{path_step}steps.gexf")
            com_df,com_top2_lang_df,com_top3_pl_df,com_top10_a_df,com_sizes,out_df,com_rank_df = nt.create_com_output_data(pl.read_csv(main_path+f"/{title}_{path_step}steps_data.csv").filter(pl.col(com_var).is_in(set([int(n) for n in list(g.nodes())]))),com_var=com_var)
            #out_df.write_csv(main_path+f"/{title}_{path_step}steps_{com_var}_data.csv")
            nt.coms_data_to_textfile(com_df,com_top2_lang_df,com_top3_pl_df,com_top10_a_df,com_sizes,step_g_com_sim_coms[path_step],step_g_com_sim_vals[path_step],com_var,main_path+f"/{title}_{path_step}steps_{com_var}_analysis.txt")
    sys.exit()

def create_remapped_nets(edge_df,net_idx,metrics,main_path,title,path_step,com_res=[1.0],include_doms=False,projects=None,remapping=True):
    
    print ("creating new projections")
    new_doms = False
    df = pl.read_csv(main_path+f"/{title}_{path_step}steps_data.csv")
    if include_doms:
        if os.path.isfile(main_path+f"/{title}_{path_step}steps_dom_per_actor.csv") and not new_doms:
            dom_per_actor = pl.read_csv(main_path+f"/{title}_{path_step}steps_dom_per_actor.csv")
        else:
            if len(df) > 150000:
                com_sizes = defaultdict(int)
                for node,com in zip(df["actor_platform"].to_list(),df["com_{}".format(com_res[0])].to_list()):
                    com_sizes[com]+=1
                dom_per_actor = bsc.multi_find_domains_per_actor(pl.concat([df.filter(pl.col("com_{}".format(com_res[0]))==k).head(int(v*0.55)) for k,v in com_sizes.items()])["actor_platform"].to_list())
            else:
                dom_per_actor = bsc.multi_find_domains_per_actor(df["actor_platform"].to_list())
            dom_per_actor.write_csv(main_path+f"/{title}_{path_step}steps_dom_per_actor.csv")
    else:
        dom_per_actor = None
    for res in com_res:
        for o_i in ["","_inner"]:
            com_var="com_{0}{1}".format(res,o_i)
            if com_var not in metrics: continue
            com_counts = dict(Counter(list(metrics[com_var].values())))
            all_coms = [c for c in sorted(set(list(metrics[com_var].values()))) if com_counts[c] > 2]
            new_net_idx = {v:i for i,v in enumerate(all_coms)}
            node_mapping = {net_idx[n]:new_net_idx[v] for n,v in metrics[com_var].items() if n in net_idx and v in new_net_idx}
            re_df = nt.remap_nodes_based_on_category(edge_df.filter((pl.col('e').is_in(set(list(node_mapping.keys())))) & (pl.col('o').is_in(set(list(node_mapping.keys()))))),node_mapping)
            re_g = nxk.graph.Graph(n=len(new_net_idx), weighted=True, directed=True, edgesIndexed=False)
            com_df,com_top2_lang_df,com_top3_pl_df,com_top10_a_df,com_sizes,out_df,com_rank_df = nt.create_com_output_data(df,com_var=com_var,include_doms=dom_per_actor,projects=projects)
            out_df.write_csv(main_path+f"/{title}_{path_step}steps_{com_var}_data.csv")
            com_rank_df.write_csv(main_path+f"/{title}_{path_step}steps_{com_var}_rank_data.csv")
            if remapping:
                com_sizes = {k:{"size":v} for k,v in com_sizes.items()}
                for com in list(com_sizes.keys()):
                    com_sizes[com].update({"lang":com_top2_lang_df.filter((pl.col(com_var)==com)).head(1)["lang"].to_list()[0]})
                    com_sizes[com].update({"platform":com_top3_pl_df.filter((pl.col(com_var)==com)).head(1)["platform"].to_list()[0]})
                #print (re_g.numberOfNodes())
                #print (len(set(re_df["o"].to_list()).union(set(re_df["e"].to_list()))))
                #print (set(re_df["o"].to_list()).union(set(re_df["e"].to_list())))
                re_df = re_df.groupby(["o", "e"]).agg(weight=pl.sum("weight"))
                re_df_bc = nt.noise_corrected(re_df,undirected=False)
                bc_scores = {(o,e):s for o,e,s in zip(re_df_bc["o"].to_list(),re_df_bc["e"].to_list(),re_df_bc["score"].to_list())}
                #re_df = nt.filter_on_backbone(re_df,threshold=1.0,tol=0.2)
                for row in re_df.to_numpy():
                    #re_g.addEdge(int(row[0]), int(row[1]), w=0.0, addMissing=True, checkMultiEdge=True)
                    re_g.increaseWeight(int(row[0]), int(row[1]), int(row[2]))
                    #re_g.addEdge(int(row[0]), int(row[1]), w=int(row[2]))
                re_g = nt.filter_on_gc(re_g)
                rev_net_idx = nt.create_rev_net_idx(g=None,net_idx=new_net_idx)
                bc_scores = {(rev_net_idx[k[0]],rev_net_idx[k[1]]):(v+1)*0.5 for k,v in bc_scores.items()}
                re_g = nxk.nxadapter.nk2nx(re_g)
                re_g = nx.relabel_nodes(re_g,rev_net_idx)
                norm_edge_weights_out = nt.get_norm_edge_weights_out(re_g)
                norm_edge_weights_full = nt.get_norm_edge_weights_full(re_g)
                nx.set_edge_attributes(re_g,norm_edge_weights_out,"norm_weight_out")
                nx.set_edge_attributes(re_g,norm_edge_weights_full,"norm_weight_full")
                nx.set_edge_attributes(re_g,bc_scores,"bc_score")
                nx.set_node_attributes(re_g,com_sizes)
                nx.write_gexf(re_g,main_path+f"/{title}_{path_step}steps_{com_var}.gexf")

main_path = "/work/JakobBækKristensen#1091/alterpublics/projects/altmed"

bf_load = datetime.datetime.now()
"""df = pl.read_csv(main_path+"/"+"altmed_denmark_2deg.csv",columns=["o","e","w"],dtypes={"url":pl.Utf8,"actor_platform":pl.Utf8})
print ()
print (f"Total time to load data = {(datetime.datetime.now()-bf_load).total_seconds()} seconds")"""

mems = []
bf_net = datetime.datetime.now()

#df = nt.import_data(domains=["aktuelltfokus.se","friatider.se"],platforms=["facebook"])
#df = nt.import_data(domains=["sameksistens.dk"],platforms=["facebook"],directed=True,save_as=main_path+"/sam_test.csv",new=True,keep_org_degree=True)
#df = nt.import_data(domains=["sameksistens.dk"],directed=True,save_as=main_path+"/sam_test2.csv",new=True,keep_org_degree=True)
#df = nt.import_data(domains=["sameksistens.dk"],directed=True,save_as=main_path+"/sam_test3.csv",new=False,keep_org_degree=True,steps=4,num_cores=16)
#df = nt.import_data(actors=["Sameksistens_Twitter"],directed=True,save_as=main_path+"/sam_actor_test.csv",new=True,keep_org_degree=True,steps=2,num_cores=1,zero_deg_domain_constraint=["sameksistens.dk"])
#df = nt.import_data(actors=["Sameksistens_Twitter"],directed=True,save_as=main_path+"/sam_actor_test2.csv",new=True,keep_org_degree=True,steps=3,num_cores=8,zero_deg_domain_constraint=["sameksistens.dk"])
#df = nt.import_data(actors=["Sameksistens_Twitter"],directed=True,save_as=main_path+"/sam_actor_test3.csv",new=False,keep_org_degree=True,steps=4,num_cores=8,zero_deg_domain_constraint=["sameksistens.dk"])
#df = nt.import_data(projects=None,actors=["Aktuellt Fokus_Facebook Page","NordfrontSE_Telegram","NordfrontSE_Twitter","NordfrontSE_Vkontakte"],domains=["nordfront.se","aktuelltfokus.se"],directed=True,save_as=main_path+"/sv_test.csv",new=True,keep_org_degree=True,steps=2,num_cores=8)
#df = nt.import_data(projects=["altmed_sweden"],actors=["Aktuellt Fokus","NordfrontSE"],directed=True,save_as=main_path+"/sv_test1.csv",new=False,keep_org_degree=True,steps=2,num_cores=8)
#df = nt.import_data(projects=["altmed_sweden"],actors=["Aktuellt Fokus","NordfrontSE"],directed=True,save_as=main_path+"/sv_test2.csv",new=False,keep_org_degree=True,steps=4,num_cores=8)
#df = nt.import_data(projects=["altmed_denmark"],directed=True,save_as=main_path+"/alt_dk_4steps_di.csv",new=False,keep_org_degree=True,steps=4,num_cores=16)
#df = nt.import_data(projects=["altmed_sweden"],actors=["Aktuellt Fokus","NordfrontSE"],directed=True,save_as=main_path+"/stupid_test.csv",new=True,keep_org_degree=True,steps=2,num_cores=24)
#df = nt.import_data(projects=["altmed_sweden"],actors=["Aktuellt Fokus","NordfrontSE"],directed=True,save_as=main_path+"/sv_test_texts_4steps.csv",new=True,keep_org_degree=True,steps=4,num_cores=16,incl_text=True)
#sys.exit()
#com_var = "com_1.0"
#dom_per_actor = pl.read_csv(main_path+f"/{title}_{path_step}steps_dom_per_actor.csv")
#df = pl.read_csv(main_path+f"/{title}_{path_step}steps_data.csv")
#com_df,com_top2_lang_df,com_top3_pl_df,com_top10_a_df,com_sizes,out_df,com_rank_df = nt.create_com_output_data(df,com_var=com_var,include_doms=dom_per_actor,projects=projects)
#out_df.write_csv(main_path+f"/{title}_{path_step}steps_{com_var}_data.csv")
#com_rank_df.write_csv(main_path+f"/{title}_{path_step}steps_{com_var}_rank_data.csv")
#continue
#title = "respons_test"
#title = "sv_undi"
#title = "sv_test_texts"
#title = "sv_test_di"
#title = "sv_di"
#title = "dk_di"
directed = True
path_steps = [4]
com_res = [1.0]
com_detect_its = 1
com_recursive = False
affinity_epochs = 7
#projects = ["altmed_denmark"]
projects = ["altmed_sweden"]
#create_multi_step_node_colors(main_path,title,path_steps)
#get_alt_spread_actor(main_path,"dk_di",4)
#get_alt_spread_actor(main_path,"sv_di",4)
#sys.exit()
#for title,projects in list(zip(["dk_di","sv_di"],[["altmed_denmark"],["altmed_sweden"]])):
for title,projects in list(zip(["de_di","au_di"],[["altmed_germany"],["altmed_austria"]])):
    for path_step in path_steps:
        #title = "sv_test_di"
        #path_step = 2
        #df = nt.import_data(projects=["altmed_sweden"],actors=["Aktuellt Fokus","NordfrontSE"],directed=directed,save_as=main_path+f"/{title}_{path_step}steps.csv",new=False,keep_org_degree=True,steps=path_step,num_cores=24,incl_text=False)
        df = nt.import_data(projects=projects,actors=[],directed=directed,save_as=main_path+f"/{title}_{path_step}steps.csv",new=False,keep_org_degree=True,steps=path_step,num_cores=48,incl_text=False,keep_doms_actors=True)
        #df = nt.import_data(projects=["altmed_denmark"],actors=["Respons"],directed=directed,save_as=main_path+f"/{title}_{path_step}steps.csv",new=False,keep_org_degree=True,steps=path_step,num_cores=16,incl_text=False)
        #df.write_csv(main_path+"/nxk_di_test.csv")
        plabels = nt.import_labels(projects=projects)
        if directed:
            df = df.sort(["o", "dt"], descending=True)
        org_deg_df = df.select(pl.col(["e","org_degree"])).sort(["org_degree"], descending=False).unique(subset=["e"],keep="first").to_dict(as_series=False)
        org_deg_df = {k:v for k,v in zip(org_deg_df["e"],org_deg_df["org_degree"]) if "_" in k}
        df = df.select(pl.col(["o", "e", "w"]))
        print (len(df))
        #df = nt.filter_df_on_degrees(df,"e",mind=20,only=set([k for k,v in org_deg_df.items() if v==3]))
        #df = nt.filter_df_on_degrees(df,"e",mind=2)
        print (len(df))
        df = nt.noise_corrected(df,undirected=False,weight="w")
        if path_step == 4:
            df = nt.filter_on_backbone(df,threshold=1.0,tol=0.24,max_edges=0.6,remove_only=set([k for k,v in org_deg_df.items() if v>2]),weight="w",skip_nodes=set([k for k,v in org_deg_df.items() if v<3]))
        #df = nt.filter_on_backbone(df,threshold=1.0,tol=0.2,max_edges=-1,weight="w",skip_nodes=set([k for k,v in org_deg_df.items() if v<3]))
        df = nt.filter_df_on_degrees(df,"e",mind=2)
        print (len(df))
        e_tups = nt.create_edge_tuples(df)
        g, net_idx = nt.df_to_nxk(e_tups,di=directed)
        print (g.numberOfNodes())
        print (g.numberOfEdges())

        #nt.output_graph_simple(g,net_idx,main_path+f"/{title}_{path_step}steps_BI.gexf",add_custom=org_deg_df)
        #sys.exit()
        #g = nt.filter_on_degrees(g,mind=2)
        nodes_to_collapse = nt.get_collapse_node_list(g,df,net_idx,col="o")
        edge_m = nt.to_uni_matrix(g,nodes_to_collapse,di=directed,ncores=-1)
        edge_df = nt.edge_m_to_df(edge_m)
        print (edge_df.shape)
        if path_step == 4:
            edge_df = nt.filter_df_on_edge_weight(edge_df,"weight",minw=3)
        else:
            edge_df = nt.filter_df_on_edge_weight(edge_df,"weight",minw=2)
        print (edge_df.shape)
        edge_df = nt.noise_corrected(edge_df,undirected=False)
        if path_step > 3:
            #edge_df = nt.filter_on_backbone(edge_df,threshold=1.0,tol=0.1,max_edges=(edge_df.shape[0]*0.5),skip_nodes=set([net_idx[k] for k,v in org_deg_df.items() if v==0]))
            edge_df = nt.filter_on_backbone(edge_df,threshold=1.0,tol=0.1,skip_nodes=set([net_idx[k] for k,v in org_deg_df.items() if v==0]))
        else:
            edge_df = nt.filter_on_backbone(edge_df,threshold=1.0,tol=0.1,skip_nodes=set([net_idx[k] for k,v in org_deg_df.items() if v==0]))
        print (edge_df.shape)
        temp_graph = nt.edge_df_to_graph(g,edge_df)
        #temp_graph = nt.filter_on_gc(nt.edge_df_to_graph(g,edge_df))
        print (temp_graph.numberOfNodes())
        print (temp_graph.numberOfEdges())
        temp_graph = nt.filter_on_degrees(temp_graph,mind=1)
        temp_graph = nt.filter_on_gc(temp_graph)
        print (temp_graph.numberOfNodes())
        print (temp_graph.numberOfEdges())
        temp_graph, temp_graph_net_idx = nt.net_to_compact(temp_graph,net_idx)
        plabels["Partisan"] = {k:v for k,v in plabels["Partisan"].items() if k in org_deg_df and org_deg_df[k]==0}
        if "test" in title:
            labels = {k:"right" for k,v in org_deg_df.items() if v==0}
            labels["Aktuellt Fokus_Facebook Page"]="left"
            plabels["Partisan"]=labels
        #pd.DataFrame([{"partisan":v,"e":k} for k,v in plabels["Partisan"].items()]).to_csv(main_path+"/affinity_labels_abdul.csv",index=False)
        #sys.exit()
        #affinity_map=None
        metrics = nt.get_metrics(temp_graph,net_idx=temp_graph_net_idx)
        sys.exit()
        if affinity_epochs > 0:
            affinity_map, distances_map = nt.stlp(temp_graph,plabels["Partisan"],temp_graph_net_idx,title="partisan",num_cores=1,verbose=False,epochs=affinity_epochs,its=7)
            metrics.update(affinity_map["partisan"])
            metrics.update(distances_map)
            metrics.update(nt.get_labels_from_affinities(affinity_map))
        metrics["custom"]={k:v for k,v in org_deg_df.items() if k in net_idx}
        #temp_graph,edge_df,temp_graph_net_idx = nt.filter_on_metric(temp_graph,edge_df,temp_graph_net_idx,metrics,metric="partisan_dist",keep_prop=0.9)

        for res in com_res:
            use_res = {2:res,3:res*1,4:res*1}[path_step]
            if path_step == 4:
                coms, com_conds = nt._get_coms(temp_graph,org_deg_df,net_idx=temp_graph_net_idx,base_deg=path_step-3)
            else:
                coms, com_conds, inner_coms, inner_conds = nt.get_communities(temp_graph,net_idx=temp_graph_net_idx,its=com_detect_its,res=use_res,recursive=com_recursive)
            metrics["com_{}".format(res)]=coms
            metrics["conductance_{}".format("com_{}".format(res))]=com_conds
            if com_recursive:
                metrics["com_{}_inner".format(res)]=inner_coms
                metrics["conductance_{}_inner".format("com_{}".format(res))]=inner_conds
        metrics.update(nt.get_actor_fields(list(temp_graph.iterNodes()),net_idx=temp_graph_net_idx))
        metrics.update(nt.get_domain_share(projects))
        metrics.update(nt.get_enriched_metrics(temp_graph,metrics,net_idx=temp_graph_net_idx))
        net_idx = {k:v for k,v in net_idx.items() if "_" in k and k in temp_graph_net_idx}
        print (edge_df.shape)
        edge_df = edge_df.filter((pl.col("o").is_in(set(net_idx.values()))) | (pl.col("e").is_in(set(net_idx.values()))))
        print (edge_df.shape)
        nt.output_data(metrics,net_idx,main_path,title,path_step)
        create_remapped_nets(edge_df,net_idx,metrics,main_path,title,path_step,com_res=com_res,include_doms=True,projects=projects,remapping=True)
        continue
        edge_df = nt.noise_corrected(edge_df,undirected=False)
        if path_step > 3:
            edge_df = nt.filter_on_backbone(edge_df,threshold=1.0,tol=0.2,max_edges=temp_graph.numberOfNodes()*50,skip_nodes=set([net_idx[k] for k,v in org_deg_df.items() if v==0]))
        else:
            edge_df = nt.filter_on_backbone(edge_df,threshold=1.0,tol=0.2,max_edges=temp_graph.numberOfNodes()*50,skip_nodes=set([net_idx[k] for k,v in org_deg_df.items() if v==0]))
        temp_graph = None
        #edge_df = nt.noise_corrected(edge_df,undirected=False)
        #edge_df = nt.filter_on_backbone(edge_df,threshold=1.0,max_edges=10000*(path_step**2.75),tol=0.1,skip_nodes=set([net_idx[k] for k,v in org_deg_df.items() if v==0]))
        print (edge_df.shape)
        ug = nt.edge_df_to_graph(g,edge_df)
        print (ug.numberOfNodes())
        print (ug.numberOfEdges())
        ug = nt.filter_on_degrees(ug,mind=1)
        ug = nt.filter_on_gc(ug)
        print (ug.numberOfNodes())
        print (ug.numberOfEdges())
        nt.output_graph(ug,net_idx,main_path+f"/{title}_{path_step}steps.gexf",add_custom=org_deg_df,metrics=metrics,affinity_map=affinity_map)
        mems.append(psutil.Process().memory_info().rss / (1024 * 1024))
        print ()
        print (f"Total time to construct nets = {(datetime.datetime.now()-bf_load).total_seconds()} seconds")
        print (f"Using at most {round(max(mems)/1024,3)}GB of memory")
    #for res in com_res:
        #create_multi_step_node_colors(main_path,title,path_steps,com_var="com_{}".format(res))
        #create_multi_step_node_colors(main_path,title,path_steps,com_var="com_{}_inner".format(res))