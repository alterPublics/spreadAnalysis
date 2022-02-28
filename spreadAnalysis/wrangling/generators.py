from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.persistence.mongo import MongoSpread
from spreadAnalysis.io.config_io import Config
from spreadAnalysis.persistence.schemas import Spread
from spreadAnalysis.wrangling.searching import *
import spreadAnalysis.utils.helpers as hlp
from spreadAnalysis.scraper.scraper import Scraper
from datetime import datetime, timedelta
import sys
import pandas as pd
import numpy as np
from newspaper import Article

std_some_domains = ["t.me/","facebook.","twitter.","gab.com","youtube.com","instagram.","reddit.","vk.com/"]

def net_based_domain_collection():

	mdb = MongoSpread()
	prev_domains = mdb.database["pull"].find({"input_type":"domain"})
	prev_domains = set([LinkCleaner().remove_url_prefix(LinkCleaner().extract_special_url(d["input"])) for d in prev_domains])
	prev_domains = set([d.lower() for d in prev_domains if d is not None])
	new_domains = mdb.database["url_bi_network"].aggregate([{ "$match": {"domain": {"$regex": "\.dk" } }},{"$limit":1000000000000},{"$lookup":{"from":"url_bi_network_coded","localField":"url","foreignField":"uentity","as":"entity_data"}},{"$unwind": "$entity_data"},{"$group":{"_id":"$domain","alt":{"$avg":"$entity_data.main_category.Alternative"},"deg":{"$sum":"$entity_data.n_degrees"},"dist":{"$avg":"$entity_data.distance_to_0"}}}],allowDiskUse=True)
	#new_domains = mdb.database["url_bi_network"].aggregate([{"$limit":100000000000},{"$lookup":{"from":"url_bi_network_coded","localField":"url","foreignField":"uentity","as":"entity_data"}},{"$unwind": "$entity_data"},{"$group":{"_id":"$domain","alt":{"$avg":"$entity_data.main_category.Alternative"},"deg":{"$sum":"$entity_data.n_degrees"},"dist":{"$avg":"$entity_data.distance_to_0"}}}],allowDiskUse=True)

	all_doms = []
	for dom_doc in new_domains:
		go_on = True
		dom = LinkCleaner().remove_url_prefix(dom_doc["_id"])
		if dom is not None:
			dom = dom.lower()
			if dom not in prev_domains:
				for sdom in std_some_domains:
					if sdom in str(dom):
						go_on = False
				if go_on:
					if dom_doc["dist"] < 3 and dom_doc["alt"] > 0.12:
						new_dom_doc = {"domain":dom,
										"score":(dom_doc["alt"]**2)*np.log(dom_doc["deg"]+1)*np.log(dom_doc["dist"]+2)**-1,
										"deg":dom_doc["deg"],
										"dist":dom_doc["dist"],
										"alt":dom_doc["alt"],
										"dom_space":str(dom).split(".")[-1]}
						all_doms.append(new_dom_doc)
	all_doms = pd.DataFrame(all_doms)
	count = 0
	for i,row in all_doms.sort_values("score", axis=0, ascending=False).iterrows():
		count += 1
		print (str(row["domain"])+" : "+str(row["score"])+" - "+str(row["alt"])+" - "+str(row["deg"])+" - "+str(row["dist"]))
		if count > 200:
			break
	print (all_doms)
	mdb.close()
	return all_doms

def net_based_url_collection():

	all_urls = []
	prev_urls = get_dom_sorted_urls(full=True)
	#prev_urls = []
	mdb = MongoSpread()
	cur = mdb.database["url_bi_network"].aggregate([{"$limit":10000000000},{"$lookup":{"from":"url_bi_network_coded","localField":"url","foreignField":"uentity","as":"entity_data"}},{"$unwind": "$entity_data"},{"$group":{"_id":{"url":"$url","platform":"$platform"},"alt":{"$avg":"$entity_data.main_category.Alternative"},"deg":{"$sum":"$entity_data.n_degrees"},"dist":{"$avg":"$entity_data.distance_to_0"}}}],allowDiskUse=True)
	nurl = True
	while nurl is not None:
		try:
			nurl = next(cur, None)
			if nurl is not None:
				url = nurl["_id"]["url"]
				platform = nurl["_id"]["platform"]
				if url is not None:
					if "facebook." not in url and "instagram." not in url and "twitter." not in url and "t.me/" not in url:
						if nurl["dist"] < 3.3 and nurl["alt"] > 0.42:
							dom_url,url_sim = check_url_similarities(prev_urls,[url])
							if url_sim < .98:
								new_dom_doc = {"url":url,
												"score":(nurl["alt"]**2)*np.log(nurl["deg"]+1)*np.log(nurl["dist"]+2)**-1,
												"deg":nurl["deg"],
												"dist":nurl["dist"],
												"alt":nurl["alt"],
												"platform":platform,
												"url_sim":url_sim}
								all_urls.append(new_dom_doc)
		except:
			pass
	all_urls = pd.DataFrame(all_urls)
	print (all_urls)
	print (set(list(all_urls["platform"])))
	return all_urls
