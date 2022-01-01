from spreadAnalysis.persistence.simple import FlatFile, Project
from spreadAnalysis import _gvars as gvar
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.persistence.schemas import Spread
from difflib import SequenceMatcher
from spreadAnalysis.wrangling import wranglers as wrang
from spreadAnalysis.utils import helpers as hlp
import operator
import numpy as np
import pandas as pd

some_prefixes = ["facebook.","twitter.","vk.com","t.me"]

def view_data_structure_examples(main_path,data_type="referal_data"):

    project = Project(main_path,init=True,actor_meta_file="Actors.xlsx").get_project(format="dict")
    if data_type == "referal_data": sources = set(gvar.SOURCES.values())
    if data_type == "actor_data": sources = set(gvar.SOURCES.keys())
    for source in sources:
        for url,dat in project[data_type].data.items():
            if "data" in dat:
                dat = dat["data"]
            if source in dat:
                if len(dat[source]["output"]) > 1:
                    print (source.upper())
                    print ("\n")
                    print (dat[source]["output"][0])
                    print (dat[source]["output"][1])
                    print ("\n")
                    break

def print_urls_from_actor_data(main_path,data_type="referal_data"):

    unique_urls = set([])
    project = Project(main_path,init=True,actor_meta_file="Actors.xlsx").get_project(format="dict")
    for url,dat in project["referal_data"].data.items():
        for source in set(gvar.SOURCES.values()):
            unique_urls.add(Spread._get_message_link(data=dat["data"][source]["output"],method=dat["data"][source]["method"]))

    for url in unique_urls:
        print (url)

def print_urls_from_urls_data(main_path,data_type="referal_data"):

    sorted_urls = []
    project = Project(main_path,init=True,actor_meta_file="Actors.xlsx").get_project(format="dict")
    for url,dat in project[data_type].data.items():
        hits = 0
        source_list = []
        source_hits = 0
        for source in set(gvar.SOURCES.values()):
            if source in dat["data"]:
                hits += len(dat["data"][source]["output"])
                source_hits += 1
                source_list.append(source)
        if source_hits > 0:
            sorted_urls.append((hits,url,str(source_list)))

    print (len(sorted_urls))

    for hits,url,source_list in sorted(sorted_urls):
        #print (str(hits)+"   "+str(url)+"   "+source_list)
        print (url)

    sys.exit()
    for hits,url,source_list in list(sorted(sorted_urls)):
        for hits2,url2,source_list2 in list(sorted(sorted_urls)):
            if url != url2:
                sim_score = SequenceMatcher(None, url, url2).ratio()
                if sim_score > 0.7:
                    print (sim_score)
                    print (url+" : "+source_list)
                    print (url2+" : "+source_list)

def print_urls_from_all_data(main_path):

    #wrang.connect_actors_to_urls(main_path)
    start_date = "2019-01-01"
    end_date = "2020-01-01"
    actor_data_path = f'{main_path}/actor_data.p'
    referal_data_path = f'{main_path}/referal_data.p'
    actor_data = FlatFile(actor_data_path)
    referal_data = FlatFile(referal_data_path)
    already_printed = set([])
    already_printed = set(list(pd.read_excel(main_path+"/Urls.xlsx",engine="openpyxl")["Url"]))
    sorted_urls = {}
    seen_sources = set([])
    actor_sources = {}
    #project = Project(main_path,init=True,actor_meta_file="Actors.xlsx").get_project(format="dict")
    actors_and_links = {}
    print ("first loop")
    for actor,dat in actor_data.data.items():
        for url in dat["links_to_own_website"]:
            if url not in actors_and_links: actors_and_links[url]=actor
    for url,dat in referal_data.data.items():
        #continue
        if url in already_printed or len(str(url)) < 5 or LinkCleaner().is_url_domain(url): continue
        sorted_urls[url]={"url":url,"source_list":[],"hits":0,"seen":0,"source_hits":0,"in_both":False,"actors":set([]),"actor_source":set([])}
        hits = 0
        source_list = set([])
        source_hits = 0
        for source in set(gvar.SOURCES.values()):
            if source in dat["data"]:
                hits += len(dat["data"][source]["output"])
                source_hits += 1
                source_list.add(source)
                seen_sources.add(source)
                for row in dat["data"][source]["output"]:
                    row_date = Spread._get_date(data=row,method=source)
                    #if start_date is not None:
                        #if hlp.to_default_date_format(end_date) < hlp.to_default_date_format(row_date):
                            #continue
                    print (row_date)
                    actor = Spread._get_actor_username(data=row,method=source)
                    sorted_urls[url]["actors"].add(actor)
                    if actor not in actor_sources: actor_sources[actor]=set([])
                    actor_sources[actor].add(source)
                if url in actors_and_links:
                    actor = actors_and_links[url]
                    sorted_urls[url]["actors"].add(actor)
                    sorted_urls[url]["actor_source"].add((actor,source))
                    if actor not in actor_sources: actor_sources[actor]=set([])
                    actor_sources[actor].add(source)
        sorted_urls[url]["hits"]=hits
        sorted_urls[url]["source_list"]=source_list
        sorted_urls[url]["seen"]=len(source_list)*np.log(hits+1)
        sorted_urls[url]["source_hits"]=source_hits

    print ("second loop")
    true_actors = set({})
    for actor,dat in actor_data.data.items():
        if str(dat["meta"]["Iteration"]) != "0" and str(dat["meta"]["Iteration"]) != "0.0":
            continue
        for source in set(gvar.SOURCES.keys()):
            if source in dat["data"]:
                for row in dat["data"][source]["output"]:
                    if row is not None:
                        row_date = Spread._get_date(data=row,method=gvar.SOURCES[source])
                        #if start_date is not None:
                            #if hlp.to_default_date_format(end_date) < hlp.to_default_date_format(row_date):
                                #continue
                        print (row_date)
                        #actor_username = Spread._get_actor_username(data=row,method=gvar.SOURCES[source])
                        true_actors.add(actor)
                        if actor not in actor_sources: actor_sources[actor]=set([])
                        actor_sources[actor].add(gvar.SOURCES[source])
                        url = Spread._get_message_link(data=row,method=gvar.SOURCES[source])
                        print (url)
                        is_some_prefix = False
                        if url in already_printed or len(str(url)) < 5 or LinkCleaner().is_url_domain(url): continue
                        for som in some_prefixes:
                            if som in str(url): is_some_prefix = True
                        if is_some_prefix: continue
                        interacts = Spread._get_interactions(data=row,method=gvar.SOURCES[source])
                        if url in sorted_urls:
                            if interacts is None: interacts=0
                            if gvar.SOURCES[source] not in sorted_urls[url]["source_list"]:
                                sorted_urls[url]["source_hits"]+=1
                            sorted_urls[url]["hits"]+=1
                            sorted_urls[url]["seen"]+=interacts
                            sorted_urls[url]["source_list"].add(gvar.SOURCES[source])
                            sorted_urls[url]["in_both"]=True
                            sorted_urls[url]["actors"].add(actor)
                            sorted_urls[url]["actor_source"].add((actor,gvar.SOURCES[source]))
                            seen_sources.add(gvar.SOURCES[source])
                        else:
                            sorted_urls[url]={"url":url,"source_list":set([gvar.SOURCES[source]]),"hits":1,"seen":interacts,"source_hits":1,"in_both":False,"actors":set([actor]),"actor_source":set([(actor,gvar.SOURCES[source])])}
    sorted_urls = [tuple(d.values()) for d in list(sorted_urls.values())]
    for t in sorted(sorted_urls, key=operator.itemgetter(3), reverse=False):
        print (" - ".join([str(t[0]),str(t[1]),str(t[2]),str(t[3]),str(t[4]),str(t[5])]))
    #sys.exit()

    final_urls = {}
    for actor, sources in actor_sources.items():
        final_urls[actor]={source:[] for source in sources}
    for bol in [True,False]:
        bol_count=0
        for t in sorted(sorted_urls, key=operator.itemgetter(3), reverse=True):
            bol_count+=1
            if bol == t[5] or bol != t[5]:
                for actor,source in t[-1]:
                    #print (final_urls[actor])
                    #print (actor_sources[actor])
                    if len(final_urls[actor][source]) < int((48000/len(actor_data.data.items()))/len(actor_sources[actor])):
                        if t[0] is not None:
                            final_urls[actor][source].append((str(t[0])," - ".join([str(t[0]),str(t[1]),str(t[2]),str(t[3]),str(t[4]),str(t[5])])))
            if bol == True and bol_count > int(len([u for u in sorted_urls if u[5] == True])/2):
                break
    final_url_count = 0
    csv_rows = []
    seen_urls = set([])
    for actor,sources in final_urls.items():
        print (actor.upper())
        print ()
        for source, urls in sources.items():
            print ("\t"+source)
            for url in urls:
                if not url[0] in seen_urls:
                    print ("\t\t"+url[1])
                    final_url_count+=1
                    csv_rows.append([url[0],"0","0",actor,source])
                    seen_urls.add(url[0])
        print ()
        print ()
    print (final_url_count)
    print (len(sorted_urls))
    print (len([u for u in sorted_urls if u[5] == True]))
    pd.DataFrame(csv_rows, columns=["Url","Iteration","Domain","Org Actor","Org Source"]).to_csv(main_path+"/url_print.csv")
