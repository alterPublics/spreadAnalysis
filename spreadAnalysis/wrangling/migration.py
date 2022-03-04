import sys
import spreadAnalysis.utils.helpers as hlp
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.persistence.schemas import Spread
from spreadAnalysis.persistence.simple import FlatFile, Project
from spreadAnalysis.collect.collect_mongo import CollectMongo

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

def migrate_refs(main_path):

    referal_data_path = f'{main_path}/referal_data.p'
    referal_data = FlatFile(referal_data_path)

    total_docs_inserted = 0
    counts = set([])
    data_rows = referal_data.data
    referal_data = None

    #data_rows = dict(list(data_rows.items())[len(data_rows)//2:])
    #data_rows = dict(list(data_rows.items())[:len(data_rows)//2])

    col = CollectMongo(main_path)
    #prev_post_ids = col.mdb.get_keys_from_db(col.mdb.database["post"],use_col="message_id")
    #prev_url_ids_post_ids = col.mdb.get_key_pairs_from_db(col.mdb.database["url_post"],"input","message_id")
    prev_post_ids = None
    prev_url_ids_post_ids = None
    prev_pulls = col.get_pulls("url")
    prev_pulls_here = set([])

    print (len(data_rows))
    for ref, ref_dat in data_rows.items():
        ref_dat = ref_dat["data"]
        for method, source_dat in ref_dat.items():
            if len(source_dat["output"]) > 0:
                attempt_d = [hlp.to_default_date_format(Spread._get_date(data=a,method=method)) for a in source_dat["output"]]
                docs = source_dat["output"]
                min_date = min(attempt_d)
                max_date = max(attempt_d)
                prev_post_ids = col.save_data(col.mdb.database["post"],
                    docs,method,prev_post_ids,Spread._get_message_id,update_key_col="message_id",
                    skip_update=False)
                if source_dat["input"] not in prev_pulls and source_dat["input"] not in prev_pulls_here:
                    col.process_pull(source_dat["input"],method,"url",docs,str(min_date)[:10],str(max_date)[:10],prev_pulls,extra_fields={"from_backup":True})
                    prev_pulls_here.add(source_dat["input"])

                prev_url_ids_post_ids = col.save_data(col.mdb.database["url_post"],
                    [{"input":source_dat["input"],"message_id":Spread._get_message_id(method=method,data=doc)} for doc in docs],
                    method,prev_url_ids_post_ids,("input","message_id"),
                    skip_update=False,update_key_col=("input","message_id"))
                total_docs_inserted+=len(docs)
            if round(total_docs_inserted, -4) % 10000 == 0 and round(total_docs_inserted, -4) not in counts:
                print ("ref docs inserted: " + str(total_docs_inserted))
                counts.add(round(total_docs_inserted, -4))
    col = None
    referal_data = None

def migrate_actors(main_path):

    actor_data_path = f'{main_path}/actor_data.p'
    actor_data = FlatFile(actor_data_path)

    col = CollectMongo(main_path)
    #prev_post_ids = col.mdb.get_keys_from_db(col.mdb.database["post"],use_col="message_id")
    #prev_actor_ids_post_ids = col.mdb.get_key_pairs_from_db(col.mdb.database["actor_post"],"input","message_id")
    prev_post_ids = None
    prev_actor_ids_post_ids = None
    prev_pulls = col.get_pulls("actor")
    prev_pulls_here = set([])

    total_docs_inserted = 0
    counts = set([])
    for actor, adat in actor_data.data.items():
        adat = adat["data"]
        for method, source_dat in adat.items():
            if method in avlb_platforms: method = avlb_platforms[method]
            if len(source_dat["output"]) > 0:
                attempt_d = [hlp.to_default_date_format(Spread._get_date(data=a,method=method)) for a in source_dat["output"] if hlp.to_default_date_format(Spread._get_date(data=a,method=method)) is not None]
                docs = source_dat["output"]
                if len(attempt_d) > 0:
                    min_date = min(attempt_d)
                    max_date = max(attempt_d)
                    if actor not in prev_pulls and actor not in prev_pulls_here:
                        col.process_pull(actor,method,"actor",docs,str(min_date)[:10],str(max_date)[:10],prev_pulls,extra_fields={"from_backup":True})
                        prev_pulls_here.add(actor)
                prev_post_ids = col.save_data(col.mdb.database["post"],
                    docs,method,prev_post_ids,Spread._get_message_id,update_key_col="message_id",
                    skip_update=False)
                prev_actor_ids_post_ids = col.save_data(col.mdb.database["actor_post"],
                    [{"input":actor,"message_id":Spread._get_message_id(method=method,data=doc)} for doc in docs],
                    method,prev_actor_ids_post_ids,("input","message_id"),
                    skip_update=False,update_key_col=("input","message_id"))
                total_docs_inserted+=len(docs)
            if round(total_docs_inserted, -4) % 10000 == 0 and round(total_docs_inserted, -4) not in counts:
                print ("actor docs inserted: " + str(total_docs_inserted))
                counts.add(round(total_docs_inserted, -4))
    col = None
    actor_data = None

def migrate_domains(main_path):

    referal_data_path = f'{main_path}/referal_data.p'
    domain_data_path = f'{main_path}/domain_data.p'
    referal_data = FlatFile(referal_data_path)
    domain_data = FlatFile(domain_data_path)

    col = CollectMongo(main_path)
    prev_pulls = col.get_pulls("actor")
    prev_pulls_here = set([])
    #prev_domain_ids_url_ids = col.mdb.get_key_pairs_from_db(col.mdb.database["domain_url"],"input","url")
    prev_domain_ids_url_ids = None

    total_docs_inserted = 0
    for dom, url_dat in domain_data.data.items():
        for outer_method, urls in url_dat.items():
            attempt_d = []
            for url in urls:
                if url in referal_data.data:
                    for method, source_dat in referal_data.data[url]["data"].items():
                        if len(source_dat["output"]) > 0:
                            for a in source_dat["output"]:
                                attempt_d.append(hlp.to_default_date_format(Spread._get_date(data=a,method=method)))
            prev_domain_ids_url_ids = col.save_data(col.mdb.database["domain_url"],
                [{"input":dom,"url":url} for url in urls],
                outer_method,prev_domain_ids_url_ids,("input","url"),update_key_col=("input","url"))
            if len(attempt_d) > 0:
                min_date = min(attempt_d)
                max_date = max(attempt_d)
                if dom not in prev_pulls and dom not in prev_pulls_here:
                    col.process_pull(dom,outer_method,"domain",attempt_d,str(min_date)[:10],str(max_date)[:10],prev_pulls,extra_fields={"from_backup":True})
                    prev_pulls_here.add(dom)
            else:
                print (dom)
                print (attempt_d)
        total_docs_inserted+=1
        print ("domain urls inserted: " + str(total_docs_inserted))
    col = None
    domain_data = None
    referal_data = None

mains = []
titles = []
#mains.append("/home/alterpublics/downloaded_spread_data/fivepillar/euvsdisinfo")
#mains.append("/home/alterpublics/downloaded_spread_data/fivepillar/midtifleisen")
#mains.append("/home/alterpublics/downloaded_spread_data/fivepillar/fivepillar")
#mains.append("/home/alterpublics/downloaded_spread_data/fivepillar/no_gen5")
#mains.append("/home/alterpublics/downloaded_spread_data/mainmed/norway")
#mains.append("/home/alterpublics/downloaded_spread_data/NorgeDis/danskdis_exploration")
#mains.append("/home/alterpublics/downloaded_spread_data/NorgeDis/p10_exploration")
#mains.append("/home/alterpublics/downloaded_spread_data/NorgeDis/peacedata")
#mains.append("/home/alterpublics/downloaded_spread_data/NorgeDis/sott")
#mains.append("/home/alterpublics/downloaded_spread_data/altmed_germany")
mains.append("/home/alterpublics/downloaded_spread_data/altmed_austria")
#mains.append("/home/alterpublics/downloaded_spread_data/altmed_norway")
#mains.append("/home/alterpublics/downloaded_spread_data/altmed_denmark")
#mains.append("/home/alterpublics/downloaded_spread_data/altmed_sweden")

#mains.append("/home/alterpublics/downloaded_spread_data/altmed_norway")
#mains.append("/home/alterpublics/downloaded_spread_data/mainmed_norway")

for main_path in mains:
    print (main_path)
    col = CollectMongo(main_path)
    #print (len(col.mdb.get_keys_from_db(col.mdb.database["post"],use_col="message_id")))
    #sys.exit()
    col.mdb.update_custom_data(main_path)
    #col.mdb.update_platform_info()
    #col.mdb.update_aliases(main_path)
    #col = None
    #migrate_domains(main_path)
    #migrate_refs(main_path)
    #migrate_actors(main_path)
