def create_bi_ego_graph(selection_types=["actor"],actor_selection={},url_selection={},actors=[],urls=[],between_dates=None,only_platforms=[]):

    def filter_degree_data(degree_data,cut_off=0.5):

        cut_off_val = sorted(list(degree_data.values())\
            ,reverse=False)[int(len(degree_data)*cut_off)]
        degree_data = {f:c for f,c in degree_data.items() if c > cut_off_val}

        return degree_data

    def update_final_degree_data(degree_data):

        degree_data = {d:len(set(dl))*(np.log(len(dl)+1)) for d,dl in degree_data.items()}
        return degree_data

    def update_net_doc(mdb,data,all_data,actor_url_ids,degree_data,degree_type="url",edge_type="actor",between_dates=None):

        for net_doc in data:
            if not net_doc["_id"] in actor_url_ids:
                if net_doc[degree_type] is not None and net_doc[degree_type] != "None":
                    for mid in net_doc["message_ids"]:
                        if between_dates is not None:
                            post = mdb.database["post"].find_one({"message_id":mid})
                            post_date = Spread._get_date(data=post,method=post["method"])
                            if post_date is not None:
                                if hlp.date_is_between_dates(post_date,between_dates["start_date"],between_dates["end_date"]):
                                    actor_url_ids.add(net_doc["_id"])
                                    all_data.append({"actor":net_doc["actor"],"url":net_doc["actor"]})
                                    degree_data = update_tmp_degree_data(degree_data,net_doc,degree_type,edge_type)
                        else:
                            actor_url_ids.add(net_doc["_id"])
                            all_data.append({"actor":net_doc["actor"],"url":net_doc["actor"]})
                            degree_data = update_tmp_degree_data(degree_data,net_doc,degree_type,edge_type)

        return all_data, actor_url_ids, degree_data

    if len(only_platforms) < 1:
        only_platforms = ["facebook","twitter","web","vkontakte","reddit","youtube",
                    "telegram","tiktok","gab","instagram"]
    mdb = MongoSpread()
    net_db = mdb.database["url_bi_network"]
    num_cores = 12
    pool = Pool(num_cores)
    final_net_data = []
    first_degree_urls = {}
    actor_url_ids = set([])
    print ("finding core nodes.")
    if "actor" in selection_types:
        print ("actors...")
        actors = find_actors(selection=actor_selection,actors=actors)
        for actor in actors:
            actor_data = mdb.database["url_bi_network"].find({"actor":actor,"platform":{"$in":only_platforms}})
            final_net_data, actor_url_ids, first_degree_urls = update_net_doc(mdb,actor_data,final_net_data,actor_url_ids,first_degree_urls,degree_type="url",edge_type="actor",between_dates=between_dates)
            aliases = mdb.database["alias"].find({"actor":actor})
            for alias_doc in aliases:
                actor_data = mdb.database["url_bi_network"].find({"actor":alias_doc["alias"],"platform":{"$in":only_platforms}})
                final_net_data, actor_url_ids, first_degree_urls = update_net_doc(mdb,actor_data,final_net_data,actor_url_ids,first_degree_urls,degree_type="url",edge_type="actor",between_dates=between_dates)
    if "url" in selection_types:
        print ("urls...")
        urls = find_urls(selection=url_selection,urls=urls)
        first_degree_urls.update(urls)
    del actors
    del actor_data
    del urls
    gc.collect()

    print ("building first degree connections.")
    first_degree_actors = {}
    fucount = 0
    url_data = {}
    for furl,furl_count in first_degree_urls.items():
        fucount+=1
        if furl is not None and furl != "None":
            url_data[furl]=mdb.database["url_bi_network"].find({"url":furl,"platform":{"$in":only_platforms}})
        if fucount % 100 == 0: print ("{0} out of {1}".format(fucount,str(len(first_degree_urls))))
    print ("tabulating data from first degree connections")
    url_data_chunked = hlp.chunks_dict(url_data,int(len(url_data)/num_cores)+1)
    url_data = []
    for chunk in url_data_chunked:
        new_list = []
        for k,v in chunk.items():
            new_list.extend(list(v))
        url_data.append(new_list)
    url_data = [(l,set(actor_url_ids),"actor","url",between_dates) for l in url_data]
    print ("processing...")
    results = pool.map(update_net_doc_multi,url_data)
    for result in results:
        final_net_data.extend(result[0])
        actor_url_ids.update(result[1])
        first_degree_actors.update(result[2])
    first_degree_actors = update_final_degree_data(first_degree_actors)
    del results
    del url_data
    del url_data_chunked
    gc.collect()

    print ("searching for second degree interconnections.")
    fucount = 0
    net_docs_by_url_count = {}
    net_docs_by_actor_count = {}
    before_filter_count = 0
    before_shaving = len(first_degree_actors)
    first_degree_actors = filter_degree_data(first_degree_actors,cut_off=0.5)
    print ("shaving from {0} to {1} for degree connections".format(before_shaving,len(first_degree_actors)))
    all_queries = []
    for _factor,facount in first_degree_actors.items():
        fucount+=1
        all_queries.append({"actor":_factor,"platform":{"$in":only_platforms}})
        #if fucount % 100 == 0: print ("{0} out of {1}".format(fucount,str(len(first_degree_actors))))
    print ("processing search")
    results = pool.map(query_multi,[("url_bi_network",l) for l in hlp.chunks(all_queries,num_cores)])
    for result in results:
        for net_doc in result:
            if net_doc["_id"] in actor_url_ids: continue
            actor_url_ids.add(net_doc["_id"])
            if net_doc["url"] not in net_docs_by_url_count:
                net_docs_by_url_count[net_doc["url"]]=[[],0]
            net_docs_by_url_count[net_doc["url"]][0].append(net_doc)
            net_docs_by_url_count[net_doc["url"]][1]+=1
            before_filter_count+=1
    del results
    del all_queries
    gc.collect()
    print ("number of new actor/url pairs before filter: {0}".format(str(before_filter_count)))
    fucount = 0
    final_net_docs = []
    for url,udat in net_docs_by_url_count.items():
        if udat[1] >= 2:
            final_net_docs.extend(udat[0])
    del net_docs_by_url_count
    gc.collect()
    print ("number of actor/url pairs after filter: {0}".format(str(len(final_net_docs))))
    print ("tabulating data for interconnected second degree actors.")
    chunked_net_docs = hlp.chunks(final_net_docs,num_cores)
    final_net_docs = [(l,set([]),"url","actor",between_dates) for l in chunked_net_docs]
    results = pool.map(update_net_doc_multi,final_net_docs)
    for result in results:
        final_net_data.extend(result[0])
    del results
    del chunked_net_docs
    del final_net_docs
    gc.collect()
    #for net_docs in final_net_docs:
        #fucount+=1
        #final_net_data, actor_url_ids, _ = update_net_doc(mdb,net_docs,final_net_data,actor_url_ids,dict({}),degree_type="url",edge_type="actor",between_dates=between_dates)
        #if fucount % 1000 == 0: print ("{0} out of {1}".format(fucount,str(len(final_net_docs))))

    print (len(final_net_data))
    final_net_data = pd.DataFrame(final_net_data)
    return final_net_data


def update_net_doc_multi(args):

    mdb = MongoSpread()
    all_data = []
    degree_data = {}
    data = args[0]
    actor_url_ids = args[1]
    degree_type = args[2]
    edge_type = args[3]
    between_dates = args[4]
    for net_doc in data:
        if not net_doc["_id"] in actor_url_ids:
            if net_doc[degree_type] is not None and net_doc[degree_type] != "None":
                for mid in net_doc["message_ids"]:
                    if between_dates is not None:
                        post = mdb.database["post"].find_one({"message_id":mid})
                        post_date = Spread._get_date(data=post,method=post["method"])
                        if post_date is not None:
                            if hlp.date_is_between_dates(post_date,between_dates["start_date"],between_dates["end_date"]):
                                actor_url_ids.add(net_doc["_id"])
                                all_data.append({"actor":net_doc["actor"],"url":net_doc["actor"]})
                                degree_data = update_tmp_degree_data(degree_data,net_doc,degree_type,edge_type)
                    else:
                        actor_url_ids.add(net_doc["_id"])
                        all_data.append({"actor":net_doc["actor"],"url":net_doc["actor"]})
                        degree_data = update_tmp_degree_data(degree_data,net_doc,degree_type,edge_type)

    return all_data, actor_url_ids, degree_data
