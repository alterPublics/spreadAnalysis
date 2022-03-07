	def update_url_bi_network(self,new=False):

		net_db = self.database["url_bi_network"]
		try:
			net_db.drop_index('actor_-1')
		except:
			pass
		if new:
			net_db.drop()
			net_db = self.database["url_bi_network"]
			self.create_indexes()
		max_net_date = list(net_db.find().sort("inserted_at",-1).limit(1))
		if max_net_date is None or len(max_net_date) < 1:
			max_net_date = datetime(2000,1,1)
		else:
			if "updated_at" in max_net_date[0]:
				max_net_date = max_net_date[0]["updated_at"]
			elif "inserted_at" in max_net_date[0]:
				max_net_date = max_net_date[0]["inserted_at"]
			max_net_date = max_net_date-timedelta(days=1)

		aliases = self.get_aliases()
		actor_aliases = self.get_actor_aliases(platform_sorted=False)
		url_post_db = self.database["url_post"]
		post_db = self.database["post"]
		cur = url_post_db.find({"$or":[ {"updated_at": {"$gt": max_net_date}}, {"inserted_at": {"$gt": max_net_date}}]}).sort("input",-1)
		#print (url_post_db.count_documents({"$or":[ {"updated_at": {"$gt": max_net_date}}, {"inserted_at": {"$gt": max_net_date}}]}))
		next_url_post = True
		batch_insert = {}
		seen_ids = set([])
		count = 0
		while next_url_post is not None:
			count += 1
			next_url_post = next(cur, None)
			if next_url_post is not None:
				url = next_url_post["input"]
				seen_ids.add(next_url_post["message_id"])
				post = post_db.find_one({"message_id":next_url_post["message_id"]})
				url = Spread._get_message_link(data=post,method=post["method"])
				if url is not None:
					url = LinkCleaner().single_clean_url(url)
					url = LinkCleaner().sanitize_url_prefix(url)
					actor_username = Spread._get_actor_username(data=post,method=post["method"])
					actor_id = Spread._get_actor_id(data=post,method=post["method"])
					platform = Spread._get_platform(data=post,method=post["method"])
					platform_type = Spread._get_platform_type(data=post,method=post["method"])
					domain = Spread._get_message_link_domain(data=post,method=post["method"])
					if actor_username in actor_aliases and platform_type in actor_aliases[actor_username]:
						actor = actor_aliases[actor_username][platform_type]
						actor_platform = str(actor)+"_"+str(platform_type)
						actor_label = str(Spread._get_actor_name(data=post,method=post["method"]))+" ({0})".format(str(platform_type))
					elif actor_id in actor_aliases and platform_type in actor_aliases[actor_id]:
						actor = actor_aliases[actor_id][platform_type]
						actor_platform = str(actor)+"_"+platform_type
						actor_label = str(Spread._get_actor_name(data=post,method=post["method"]))+" ({0})".format(str(platform_type))
					else:
						actor = actor_username
						actor_platform = str(actor_username)+"_"+platform_type
						actor_label = str(Spread._get_actor_name(data=post,method=post["method"]))+" ({0})".format(str(platform_type))
					if (url,actor) not in batch_insert:
						batch_insert[(url,actor)]={"url":url,"actor_username":actor_username,
											"actor":actor,"actor_label":actor_label,"platform":platform,
											"message_ids":[],"actor_platform":actor_platform,"domain":domain}
					batch_insert[(url,actor)]["message_ids"].append(post["message_id"])
					if len(batch_insert) >= 10000:
						self.write_many(net_db,list(batch_insert.values()),key_col=("url","actor_platform"),sub_mapping="message_ids")
						batch_insert = {}
					if count % 1000 == 0:
						print ("url loop " + str(count))
		if len(batch_insert) > 0:
			self.write_many(net_db,list(batch_insert.values()),key_col=("url","actor_platform"),sub_mapping="message_ids")

		actor_post_db = self.database["actor_post"]
		cur = actor_post_db.find({"$or":[ {"updated_at": {"$gt": max_net_date}}, {"inserted_at": {"$gt": max_net_date}}]}).sort("input",-1)
		next_actor_post = True
		batch_insert = {}
		count = 0
		while next_actor_post is not None:
			count += 1
			next_actor_post = next(cur, None)
			if next_actor_post is not None:
				message_id = next_actor_post["message_id"]
				actor = next_actor_post["input"]
				if not message_id in seen_ids:
					post = post_db.find_one({"message_id":message_id})
					platform = Spread._get_platform(data=post,method=post["method"])
					platform_type = Spread._get_platform_type(data=post,method=post["method"])
					actor_username = Spread._get_actor_username(data=post,method=post["method"])
					actor_platform = str(actor)+"_"+str(platform_type)
					actor_label = str(Spread._get_actor_name(data=post,method=post["method"]))+" ({0})".format(str(platform_type))
					url = Spread._get_message_link(data=post,method=post["method"])
					domain = Spread._get_message_link_domain(data=post,method=post["method"])
					if url is not None:
						url = LinkCleaner().single_clean_url(url)
						url = LinkCleaner().sanitize_url_prefix(url)
						if (url,actor) not in batch_insert:
							batch_insert[(url,actor)]={"url":url,"actor_username":actor_username,
												"actor":actor,"actor_label":actor_label,"platform":platform,
												"message_ids":[],"actor_platform":actor_platform,"domain":domain}
						batch_insert[(url,actor)]["message_ids"].append(post["message_id"])
						if len(batch_insert) >= 10000:
							self.write_many(net_db,list(batch_insert.values()),key_col=("url","actor_platform"),sub_mapping="message_ids")
							batch_insert = {}
						if count % 1000 == 0:
							print ("actor loop " + str(count))
		if len(batch_insert) > 0:
			self.write_many(net_db,list(batch_insert.values()),key_col=("url","actor_platform"),sub_mapping="message_ids")
		net_db.create_index([ ("actor", -1) ])
