    def propagate_loop(nodes,all_fixed_nodes,cat,distance):

        entity_type_conv = {"actor":"url","url":"actor"}
        num_cores = 8
        pool = Pool(num_cores)
        for entity_type in ["url","actor"]:
            coded_entities = [d for k,d in nodes.items() if d["entity_type"] == entity_type]
            if len(coded_entities) > 0:
                chunked_nodes = list(hlp.chunks(coded_entities,num_cores))
                print ("initializing values for {0} at distance {1}".format(entity_type,distance))
                results = pool.map(assign_values_to_neighbours,chunked_nodes)
                print ("tabulating values for {0} at distance {1}".format(entity_type,distance))
                nb_vals = tabulate_nb_results(results,skip_nodes=set(all_fixed_nodes))
                print ("normalizing values for {0} at distance {1}".format(entity_type,distance))
                norm_func_args = [(l,entity_type_conv[entity_type],cat,distance) for l in list(hlp.chunks_dict(nb_vals,int(len(nb_vals)/num_cores)+1))]
                all_write_docs = pool.map(normalize_nb,norm_func_args)
                mdb = MongoSpread()
                print ("writing to database for {0} at distance {1}".format(entity_type,distance))
                for write_docs in all_write_docs:
                    if low_memory:
                        mdb.write_many(mdb.database["url_bi_network_coded"],
                                                write_docs,("uentity","category"),only_insert=True)
                    else:
                        mdb.insert_many(mdb.database["url_bi_network_coded"],write_docs)
                mdb = None

    #result_test = list(MongoSpread().database["url_bi_network"].aggregate([{ "$sample" : { "size": 10 } },{"$group": {"_id": "$actor"}}]))
    #print (len(result_test))
    #sys.exit()
    all_fixed_nodes = {d["uentity"]:d for d in MongoSpread().database["url_bi_network_coded"].find({"is_fixed":True,"category":cat})}
    #propagate_loop(all_fixed_nodes,set(all_fixed_nodes.keys()),cat,1)
