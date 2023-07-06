def posts_export_russian_dk(main_path,title,net_title,new=False):

	#df = create_actor_data(title,new=False,main_path=main_path,net_title=net_title)
	df = pd.read_csv(main_path+"/"+"{0}_final.csv".format(title))
	df = df[df["platform"]!="web"]
	df["is_central_country_actor"]=df["actor_platform"].map({row["actor_platform"]:True if (row["is_native_lang"]==1 or row["pol_main_fringe_sharp"]!="grey2_grey") else False for i,row in df.iterrows()})
	mdb = MongoSpread()
	actors = list(set(list(df["actor"])))

	domain_cats = {LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://",""):"second_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"rt_sputnik"})}
	domain_cats.update({LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://",""):"second_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"Second Pillar"})})
	domain_cats.update({LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://",""):"first_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"First Pillar"})})
	domain_cats.update({LinkCleaner().strip_backslash(LinkCleaner().extract_domain(str(d["Website"]))).replace("https://",""):"third_pillar" for d in mdb.database["actor"].find({"Iteration":0,"org_project_title":"random_russian","Type":"Third Pillar"})})
	del domain_cats["odysee.com"]

	if isfile(main_path+"/"+"{0}_posts_with_rus.csv".format(title)) or not new:
		posts = pd.read_csv(main_path+"/"+"{0}_posts_with_rus.csv".format(title))
	else:
		posts = pd.DataFrame(get_posts_from_actors(actors,only_domains=list(set(list(domain_cats.keys())))))
		posts.to_csv(main_path+"/"+"{0}_posts_with_rus.csv".format(title),index=False)

	posts = posts[~posts["link"].str.contains("odysee.com")]

	posts["pillar"]=posts["domain"].map(domain_cats)
	posts["main_fringe_sharp"]=posts["actor_platform"].map(df.set_index("actor_platform")[["main_fringe_sharp"]].to_dict()["main_fringe_sharp"])
	posts["is_central_country_actor"]=posts["actor_platform"].map(df.set_index("actor_platform")[["is_central_country_actor"]].to_dict()["is_central_country_actor"])
	posts["actor_lang"]=posts["actor_platform"].map(df.set_index("actor_platform")[["lang"]].to_dict()["lang"])
	posts.to_csv(main_path+"/"+"{0}_posts_final_with_rus.csv".format(title),index=False)
