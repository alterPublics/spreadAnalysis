def get_agg_actor_metrics(actors):

	agg_data = {}
	mdb = MongoSpread()
	first_it = True
	for actor in actors:
		try:
			actor_url_docs = mdb.database["url_bi_network"].find({"actor":actor},no_cursor_timeout=True).sort("actor_platform",-1)
			for actor_url in actor_url_docs:
				unique_actor = actor_url["actor_platform"]
				if unique_actor not in agg_data or first_it:
					if not first_it:

						actor_data = {  "actor_name":prev_ua_doc["actor_label"],
										"actor_unique":prev_ua_doc["actor_platform"],
										"actor_username":prev_ua_doc["actor_username"],
										"actor":actor,
										"most_popular_url_shared":sorted(most_popular_url_shared.items(), key = itemgetter(1), reverse=True)[0][0],
										"n_unique_domains_shared":len(unique_domains),
										"most_often_shared_domain":max(most_often_shared_domain, key = most_often_shared_domain.count),
										"n_posts":len(posts),
										"lang":max(langs, key = langs.count),
										"interactions_mean":np.nanmean(np.array(interactions,dtype=np.float)),
										"interactions_std":np.nanstd(np.array(interactions,dtype=np.float)),
										"message_length_mean":np.nanmean(np.array(text_lengths,dtype=np.float)),
										"message_length_std":np.nanstd(np.array(text_lengths,dtype=np.float)),
										"first_post_observed":min([hlp.to_default_date_format(d) for d in post_dates if d is not None]),
										"last_post_observed":max([hlp.to_default_date_format(d) for d in post_dates if d is not None]),
										"followers_mean":np.nanmean(np.array(followers,dtype=np.float)),
										"followers_max":np.nanmax(np.array(followers,dtype=np.float)),
										"platform":platform,
										"account_type":account_type,
										"account_category":account_category,
										"link_to_actor":link_to_actor,
										}

						net_label_data = mdb.database["url_bi_network_coded"].find_one({"uentity":actor,"entity_type":"actor"})
						if net_label_data is not None:
							actor_data.update(dict(net_label_data["main_category"]))
							actor_data["min_distance_to_0"]=net_label_data["distance_to_0"]
						agg_data[prev_ua_doc["actor_platform"]]=actor_data
					else:
						pass

					most_popular_url_shared = defaultdict(int)
					unique_domains = set([])
					most_often_shared_domain = []

					posts = set([])
					langs = []
					interactions = []
					text_lengths = []
					post_dates = []
					followers = []

				most_popular_url_shared[actor_url["url"]]+=1
				unique_domains.add(actor_url["domain"])
				most_often_shared_domain.append(actor_url["domain"])
				for pid in list(actor_url["message_ids"]):
					posts.add(pid)
					post_doc = mdb.database["post"].find_one({"message_id":pid})
					langs.append(Spread._get_lang(data=post_doc,method=post_doc["method"],model=lang_model))
					interactions.append(Spread._get_interactions(data=post_doc,method=post_doc["method"]))
					text_lengths.append(len(Spread._get_message_text(data=post_doc,method=post_doc["method"])))
					post_dates.append(Spread._get_date(data=post_doc,method=post_doc["method"]))
					followers.append(Spread._get_followers(data=post_doc,method=post_doc["method"]))
					platform = Spread._get_platform(data=post_doc,method=post_doc["method"])
					account_type = Spread._get_account_type(data=post_doc,method=post_doc["method"])
					account_category = Spread._get_account_category(data=post_doc,method=post_doc["method"])
					link_to_actor = Spread._get_link_to_actor(data=post_doc,method=post_doc["method"])

				prev_ua_doc = actor_url
				first_it = False
		except:
			print ("CURSOR FAIL!")
	mdb.close()
	return agg_data
