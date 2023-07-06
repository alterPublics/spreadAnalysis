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

def get_non_collected_actors(main_path,net_title):

	mdb = MongoSpread()
	platforms = [("twitter","Twitter"),("facebook","Facebook Page"),("vkontakte","Vkontakte"),("instagram","Instagram")]
	out_df = []
	actor_aliases = mdb.get_actor_aliases(platform_sorted=True)
	for platform,avl_platform in platforms:
		actor_query = {"net_data.{0}".format(net_title):{"$exists":True},"platform":platform}
		for adoc in mdb.database["actor_metric"].find(actor_query):
			if adoc["actor_username"] not in actor_aliases[avl_platform]:
				if "grey" in adoc["net_data"][net_title]["political"]: continue
				out_df.append(dict(adoc))

	out_df = pd.DataFrame(out_df)
	out_df[["actor_username","interactions_mean","n_posts","lang","platform","followers_mean","n_unique_domains_shared"]].to_csv(main_path+"/not_yet_actors_{0}.csv".format(net_title),index=False)

	langs = set(["und"])
	out_df = out_df[out_df["lang"].isin(langs)]
	#out_df = out_df[(out_df["n_posts"]>50) & (out_df["n_posts"] < 1500)]
	out_df[["actor_username","interactions_mean","n_posts","lang","platform","followers_mean","n_unique_domains_shared"]].to_csv(main_path+"/not_yet_actors_{0}_filtered.csv".format(net_title),index=False)

def get_links_to_actors(main_path,net_title):

	mdb = MongoSpread()
	final_data = []
	actor_aliases = mdb.get_actor_aliases(platform_sorted=True)
	if net_title is None:
		with_net_actors = {d["actor"] for d in mdb.database["actor_metric"].find() if "actor" in d}
	else:
		with_net_actors = {d["actor"] for d in mdb.database["actor_metric"].find({"net_data.{0}".format(net_title):{"$exists":True}}) if "actor" in d}
	platforms = ["Tiktok","Instagram","Facebook Page","Twitter","Youtube","Vkontakte","Telegram","Gab"]
	some_prefixes = ["tiktok.","instagram.","facebook.","twitter.","youtube.","vk.com/","t.me/","gab.com/"]
	fb_page_count = {}
	for pre,platform in list(zip(some_prefixes,platforms)):
		if not platform == "Telegram":continue
		print (platform)
		new_accounts = set([])
		if len(with_net_actors) < 15000:
			links = mdb.database["url_bi_network"].find({"actor":{"$in":list(with_net_actors)},"domain":{"$regex":pre}})
		else:
			links = mdb.database["url_bi_network"].find({"domain":{"$regex":pre}})
		for link in links:
			if pre in str(link["domain"]):
				if platform == "Youtube":
					if "user/" in str(link["url"]) or "channel/" in str(link["url"]) or "/c/" in str(link["url"]):
						pass
					else:
						continue
				if link["actor"] in with_net_actors:
					try:
						username = LinkCleaner().extract_username(link["url"],never_none=False,with_unpack=False)
					except:
						continue
					if username is not None and len(str(username)) > 2:
						if username not in new_accounts and username not in actor_aliases[platform]:
							new_accounts.add(username)
							pdoc = {"Actor":username}
							pdoc.update({p:None for p in platforms})
							pdoc["Iteration"]=0
							if platform == "Facebook Page":
								pdoc[platform]="https://www.facebook.com/{0}".format(username)
								fb_page_count["https://www.facebook.com/{0}".format(username)]=0
							else:
								pdoc[platform]=username
							final_data.append(pdoc)
							#print (username)
							#print (username + "     " + str(link["url"]))
						if username in new_accounts and platform == "Facebook Page":
							fb_page_count["https://www.facebook.com/{0}".format(username)]+=1
		print (len(new_accounts))
	out_df = pd.DataFrame(final_data)
	out_df["FB_counts"]=out_df["Facebook Page"].map(fb_page_count)
	out_df.to_csv(main_path+"/links_to_actors_{0}_only_tel.csv".format(net_title),index=False)

def get_domain_refs(main_path,net_title=None):

	if net_title is None:
		net_title = "no_net"
	mdb = MongoSpread()
	final_data = []
	some_prefixes = ["tiktok.","instagram.","facebook.","twitter.","youtube.","vk.com","t.me","gab.com"]
	#actor_aliases = mdb.get_actor_aliases(platform_sorted=True)
	#platforms = ["Tiktok","Instagram","Facebook Page","Twitter","Youtube","Vkontakte","Telegram","Gab"]
	#for actor in with_net_actors:
		#for row in mdb.database["url_bi_network"].find({""})
	prev_pulls = set([str(LinkCleaner().extract_special_url(d["input"])).lower() for d in mdb.database["pull"].find({"input_type":"domain"},{"input":1})])
	shorten_domains = pd.read_csv("/home/alterpublics/projects/full_test/url_shorteners.csv")
	shorten_domains = set([row["domain"] for i,row in shorten_domains.iterrows()])
	domain_count = {}
	domain_platform_query = [{"$group":{"_id":{"domain":"$domain","platform":"$platform"},"count":{"$sum":1}}}]
	if net_title != "no_net":
		with_net_actors = {d["actor"] for d in mdb.database["actor_metric"].find({"net_data.{0}".format(net_title):{"$exists":True}}) if "actor" in d}
		by_net_query = {"$match": { "actor": { "$in": list(with_net_actors) } }}
		domain_platform_query.insert(0, by_net_query)
	for row in mdb.database["url_bi_network"].aggregate(domain_platform_query,allowDiskUse=True):
		dom = row["_id"]["domain"]
		pl = row["_id"]["platform"]
		cnt = int(row["count"])
		if dom not in shorten_domains:
			if dom not in domain_count:
				domain_count[dom]={"all_platforms":0}
			if pl not in domain_count[dom]:
				domain_count[dom][pl]=0
			domain_count[dom][pl]+=cnt
			domain_count[dom]["all_platforms"]+=cnt

	for dom,pls in domain_count.items():
		dom = LinkCleaner().remove_url_prefix(dom)
		skipdom = False
		dom_low = str(dom).lower()
		if dom_low not in prev_pulls:
			for pref in some_prefixes:
				if pref in dom_low[:len(pref)]:
					skipdom = True
					break
			if not skipdom:
				temp_doc = dict(pls)
				temp_doc["n_unique_platforms"]=len(pls)-1
				temp_doc["domain"]=dom
				final_data.append(temp_doc)

	pd.DataFrame(final_data).to_csv(main_path+"/domains_shared_{0}.csv".format(net_title),index=False)


def export_actor_to_pandas(file_path,query,net_name=None):

	df = []
	mdb = MongoSpread()
	for actor_doc in mdb.database["actor_metric"].find(query):
		if net_name is not None:
			if "net_data" in actor_doc and net_name in actor_doc["net_data"]:
				actor_doc.update(actor_doc["net_data"][net_name])
		if "net_data" in actor_doc:
			del actor_doc["net_data"]
		if "actor_name" not in actor_doc or actor_doc["actor_name"] is None or str(actor_doc["actor_name"]) == "":
			#print (actor_doc["actor_platform"])
			actor_doc["actor_name"]=actor_doc["actor_platform"]
			actor_doc["platform"]="web"
		df.append(actor_doc)

	pd.DataFrame(df).to_csv(file_path,index=False)

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
							if some == "TikTok" and len(tel_username) >= 30: continue
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
	all_doms = []
	for dom in domains:
		unique_doms = set([])
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
