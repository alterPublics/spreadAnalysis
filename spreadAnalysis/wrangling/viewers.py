from spreadAnalysis.persistence.simple import FlatFile, Project
from spreadAnalysis import _gvars as gvar
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.persistence.schemas import Spread
from spreadAnalysis.scraper.scraper import Scraper
from difflib import SequenceMatcher
from spreadAnalysis.wrangling import wranglers as wrang
from spreadAnalysis.utils import helpers as hlp
import operator
import numpy as np
import time
import sys
from bs4 import BeautifulSoup
import pandas as pd
import requests
from os import listdir
from os.path import isfile, join
from spreadAnalysis.persistence.mongo import MongoSpread

some_prefixes = ["facebook.","twitter.","vk.com","t.me"]

def show_url_disparities():

    mdb = MongoSpread()
    url_post_db = mdb.database["url_post"]
    post_db = mdb.database["post"]
    cur = url_post_db.find().limit(1000)
    next_url_post = True
    while next_url_post is not None:
        next_url_post = next(cur, None)
        post = post_db.find_one({"message_id":next_url_post["message_id"]})
        post_url = Spread._get_message_link(data=post,method=post["method"])
        org_url = next_url_post["input"]

        post_url = LinkCleaner().single_clean_url(post_url)
        post_url = LinkCleaner().sanitize_url_prefix(post_url)
        org_url = LinkCleaner().single_clean_url(org_url)
        org_url = LinkCleaner().sanitize_url_prefix(org_url)

        if post_url != org_url:
            print ()
            print (org_url)
            print (post_url)
            print ()

def show_domains_in_url_db(title):

    mdb = MongoSpread()
    urls = mdb.database["url"].find({"org_project_title":title})
    df =  pd.DataFrame(list(urls))
    df["domain"]=df['Url'].apply(lambda x: str(LinkCleaner().extract_domain(x)))
    grouped = df[["Url","domain"]].groupby(['domain']) \
                             .count() \
                             .reset_index() \
                             .sort_values(['Url'], ascending=False)
    print (grouped[["domain"]].head(100).to_string(index=False))

def show_some_accounts_in_db(main_path,some="Telegram"):

    mdb = MongoSpread()
    new_data = []
    post_db = mdb.database["post"]
    cur = post_db.find()
    next_url_post = True
    #prev_telegram_actors = set([d["actor_username"] for d in mdb.database["url_bi_network"].find({"platform":"telegram"})])
    prev_telegram_actors = set([LinkCleaner().extract_username(d[some]) for d in mdb.database["actor"].find() if some in d and d[some] is not None])
    row_count = 0
    while next_url_post is not None:
        next_url_post = next(cur, None)
        if next_url_post is not None:
            if some == "Telegram":
                tel_mentions = Spread._get_message_telegram_mention(data=next_url_post,method=next_url_post["method"])
            elif some == "TikTok":
                tel_mentions = Spread._get_message_tiktok_mention(data=next_url_post,method=next_url_post["method"])
            elif some == "Gab":
                tel_mentions = Spread._get_message_gab_mention(data=next_url_post,method=next_url_post["method"])
            elif some == "Youtube":
                tel_mentions = Spread._get_message_yt_mention(data=next_url_post,method=next_url_post["method"])
            if tel_mentions is not None:
                tel_mentions = tel_mentions.split(",")
                if len(tel_mentions) > 0 and tel_mentions[0] != "":
                    for tm in tel_mentions:
                        print (tm)
                        if len(tm) > 2:
                            try:
                                tel_username = LinkCleaner()._recursive_trim( LinkCleaner().extract_username(tm) )
                            except:
                                print ("ERROR")
                                #print (tm)
                            print (tel_username)
                            if tel_username not in prev_telegram_actors:
                                row_count+=1
                                new_data.append({some:tel_username,"row":row_count})
    df = pd.DataFrame(new_data)
    grouped = df.groupby([some]) \
                             .count() \
                             .reset_index() \
                             .sort_values(['row'], ascending=False)
    print (grouped.head(10000).to_string(index=False))
    grouped.to_csv(main_path+"/{0}_exports.csv".format(some))

def rip_majestic_exports(main_path,iteration=1):

    all_backlinks = []
    dir_path = main_path + "/majestic_exports"
    url_file_path = main_path + "/Urls.xlsx"
    prev_urls = set(list(MongoSpread().get_custom_file_as_df(url_file_path)["Url"]))
    for csvfile in listdir(dir_path):
        if ".csv" in str(csvfile):
            load_csvfile = dir_path + "/" + csvfile
            current_urls = set(list(MongoSpread().get_custom_file_as_df(load_csvfile)["Source URL"]))
            for url in current_urls:
                if not url in prev_urls:
                    is_dom = int(LinkCleaner().is_url_domain(url))
                    print (url + ";" + str(iteration) + ";" + str(is_dom))

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

def choose_domains_from_list(inputs,export_path):

    def save_data(prev_df,new_doms):

        if prev_df is not None:
            out_df = pd.concat([pd.DataFrame(new_doms),prev_df], axis=0)
        out_df.to_csv(export_path,index=False)


    try:
        prev_df = pd.read_csv(export_path)
        prev = set(list(prev_df["input_domain"]))
    except:
        prev_df = None
        prev = set([])

    if isinstance(inputs,str):
        pass
    else:
        doms = inputs

    scrp = Scraper(settings={"machine":"local"})
    scrp.browser_init()
    new_doms = []
    for dom in doms:
        if "merlins-tagebuch.com" in dom: continue
        if not dom in prev:
            call_url = "https://"+LinkCleaner()._recursive_trim(dom)
            try:
                response = requests.get(call_url,timeout=10)
            except:
                print ("skipping "+str(dom))
                new_doms.append({"input_domain":dom,"ouput_domain":""})
                save_data(prev_df,new_doms)
                continue
            if response.ok:
                try:
                    scrp.browser.get(call_url)
                except KeyboardInterrupt:
                    print ("skipping "+str(dom))
                    new_doms.append({"input_domain":dom,"ouput_domain":""})
                    save_data(prev_df,new_doms)
                    continue
                current_url = scrp.browser.current_url
                answer = input("Press a to add or d to remove or exit to exit")
                if answer == "a":
                    new_doms.append({"input_domain":dom,"ouput_domain":current_url})
                    save_data(prev_df,new_doms)
                elif answer == "d":
                    new_doms.append({"input_domain":dom,"ouput_domain":""})
                elif answer == "exit":
                    break

    save_data(prev_df,new_doms)
    scrp.browser_quit()

def scrape_majestic_ref_dlinks(domains,main_path):

    scrp = Scraper(settings={"machine":"local","cookie_path":"/Users/jakobbk/Documents/user_cookies/local_maj","cookie_user":"local_maj"})
    scrp.browser_init()
    #scrp.browser.get("https://majestic.com/account/login")
    time.sleep(1)
    unique_doms = set([])
    all_doms = []
    for dom in domains:
        try:
            dom = LinkCleaner().extract_domain(dom)
            dom = LinkCleaner().remove_url_prefix(dom)
            dom = LinkCleaner()._recursive_trim(dom)
            print (dom)
            indx = 0
            for r in range(10):
                scrp.browser.get("https://majestic.com/reports/site-explorer/referring-domains?q={0}&oq=https%3A%2F%2F{0}%2F&IndexDataSource=F&s={1}#key".format(dom,indx))
                table_html = BeautifulSoup(str(scrp.browser.page_source),scrp.default_soup_parser).find("table",{"id":"vue-ref-domain-table"})
                per_range_count = 0
                if table_html is not None:
                    for row in BeautifulSoup(str(table_html),scrp.default_soup_parser).find_all("tr"):
                        for a in BeautifulSoup(str(table_html),scrp.default_soup_parser).find_all("a"):
                            if str(a["href"])[0] != "/" and "javascript" not in str(a["href"]):
                                new_dom = str(a["href"])
                                per_range_count+=1
                                if new_dom not in unique_doms:
                                    print (new_dom)
                                    all_doms.append({"domain":new_dom})
                                    unique_doms.add(new_dom)
                        if per_range_count > 45:
                            break

                else:
                    break
                indx+=50
        except:
            pass
    scrp.browser_quit()
    all_doms = pd.DataFrame(all_doms)
    all_doms.to_csv(main_path+"/output_maj_dom.csv",index=False)
