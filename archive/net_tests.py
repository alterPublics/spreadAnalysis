
def _bi_permutation_BWAH(X):

	Y = np.empty(shape = [0,2],dtype=np.int32)
	for i in range(len(X)):
		x = X.pop()
		y = x[np.stack(np.triu_indices(len(x), k=1), axis=-1)]
		#Y.append(y)
		Y = np.vstack([Y,y])
	#Y = np.vstack(Y)
	Y = pl.DataFrame(Y,schema=["o","e"],schema_overrides={"o":pl.Int32,"e":pl.Int32})
	return Y

def _bi_permutation_WA(X,batching=True):

	Yl = []
	Y = np.empty(shape = [0,2],dtype=np.int32)
	for i in range(len(X)):
		x = X.pop()
		y = x[np.stack(np.triu_indices(len(x), k=1), axis=-1)]
		Yl.append(y)
		if batching:
			if i % 1 == 0 and i > 0:
				Y = np.vstack([Y,*Yl])
				if i % 10000 == 0 and i > 0: print (psutil.Process().memory_info().rss / (1024 * 1024))
				Yl = []
	if batching:
		print (psutil.Process().memory_info().rss / (1024 * 1024))
		Y = np.vstack([Y,*Yl])
	else:
		Y = np.vstack(Yl)
	Y = pl.DataFrame(Y,schema=["o","e"],schema_overrides={"o":pl.Int32,"e":pl.Int32})
	return Y

def _bi_permutation_NEW(X):

	Y = pl.DataFrame(np.empty(shape = [0,2],dtype=np.int32),schema=["o","e"],schema_overrides={"o":pl.Int32,"e":pl.Int32})
	for i in range(len(X)):
		x = X.pop()
		y = pl.DataFrame(x[np.stack(np.triu_indices(len(x), k=1), axis=-1)],schema=["o","e"],schema_overrides={"o":pl.Int32,"e":pl.Int32})
		Y.vstack(y,in_place=True)
	return Y




def _compute_affinity_multi(args):
    affs,dists,wss = args[0],args[1],args[2]
    modified_affs = affs * (dists ** 1)
    new_aff = np.zeros((modified_affs.shape[0], modified_affs.shape[2]))
    for i in range(modified_affs.shape[2]):
        new_aff[:, i] = np.sum(wss * modified_affs[:, :, i], axis=1)
    naff_sum = new_aff.sum(axis=1)
    new_aff = np.nan_to_num(new_aff / naff_sum[:, np.newaxis], nan=0.0, posinf=0.0, neginf=0.0)
    return new_aff

def _compute_affinity_multi_OLD(args):

	affs,dists,wss = args[0],args[1],args[2]
	new_aff = np.einsum('njk,ij->kn', (affs*(dists)).T, np.sqrt(wss+2))
	naff_sum = new_aff.sum(axis=1)
	new_aff =np.nan_to_num(new_aff / naff_sum[:, np.newaxis], nan=0.0, posinf=0.0, neginf=0.0)
	return new_aff



def _compute_affinity_OLD(affs,dists,wss):
	
	new_aff = np.einsum('njk,ij->kn', (affs*(dists**1)).T, wss)
	naff_sum = new_aff.sum(axis=1)
	new_aff =np.nan_to_num(new_aff / naff_sum[:, np.newaxis], nan=0.0, posinf=0.0, neginf=0.0)
	print (new_aff.shape)
	sys.exit()
	return new_aff


def _stlp_get_n_idx_ws_chunk_OLD(n_idxs,aff_map,dist_map,max_deg,nlabels,batch_size):

	affs = []
	dists = []
	wss = []
	for n_idx,ws in n_idxs:
		length = len(n_idx)
		affs.append(np.concatenate([aff_map[n_idx, :],np.zeros((max_deg-length,nlabels))]))
		dists.append(np.concatenate([dist_map[n_idx, :],np.zeros((max_deg-length,nlabels))]))
		wss.append(np.concatenate([np.array(ws),np.zeros(max_deg-length)]))
	return np.array(affs),np.array(dists),np.array(wss)

def _stlp_get_n_idx_ws_chunk_NOT_AS_OLD(n_idxs, aff_map, dist_map, max_deg, nlabels, batch_size):
    
    affs = np.zeros((batch_size, max_deg, nlabels))
    dists = np.zeros((batch_size, max_deg, nlabels))
    wss = np.zeros((batch_size, max_deg))
    
    for i, (n_idx, ws) in enumerate(n_idxs):
        length = len(n_idx)
        
        affs[i, :length, :] = aff_map[n_idx, :]
        dists[i, :length, :] = dist_map[n_idx, :]
        wss[i, :length] = ws

    return affs, dists, wss


#@timer
def _stlp_get_n_idx_ws(wneighs):

	result = []
	for wneigh in wneighs:
		n_idx = []
		ws = []
		for n,w in wneigh:
			n_idx.append(n)
			ws.append(w)
		result.append((n_idx,np.array(ws)))
	return result


@timer
def stlp2(g,labels,net_idx,title="test",num_cores=1,its=5,epochs=3,stochasticity=1,verbose=False):

	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	max_deg = nxk.graphtools.maxDegree(g)
	org_nodes = list([n for n in g.iterNodes()])
	affinities = []
	all_dists = {}
	org_labels = {net_idx[n]:l for n,l in labels.items() if n in net_idx}
	labels = {}

	print ("generating neighbour idxs")
	nn_idx = {n:_stlp_get_n_idx_ws([g.iterNeighborsWeights(n)]) for n in org_nodes}

	for epoch in range(epochs):
		print ("getting distances")
		if len(all_dists) > 0:
			for n,_ in noise_labels.items():
				del labels[n]
			noise_labels = {n:"noise" for n in random.sample([n for n in net_idx.values() if n not in org_labels and g.hasNode(n)],int(1.0*len(org_labels)))}
			labels.update(noise_labels)
			all_dists["noise"]=get_djik_distances(g,org_nodes,noise_labels)["noise"]
		else:
			noise_labels = {n:"noise" for n in random.sample([n for n in net_idx.values() if n not in org_labels and g.hasNode(n)],int(1.0*len(org_labels)))}
			labels.update(org_labels)
			labels.update(noise_labels)
			all_dists = get_djik_distances(g,org_nodes,labels)
		print ("distances set")

		aff_cats = {c:j for j,c in enumerate(sorted(list(set(labels.values()))))}
		dist_map = np.array([np.mean(((np.array(v)+1)**-1)**(np.array(v)+1), axis=0) for k,v in sorted(all_dists.items())]).T
		aff_map = np.zeros((len(net_idx),len(all_dists)))
		for n,l in labels.items():
			aff_map[n][aff_cats[l]]=1.0

		start_time = time.time()
		stochastics = result = [True] * int(stochasticity*its) + [False] * int((1-stochasticity)*its)
		if len(stochastics) < its:
			stochastics.append(False)
		random.shuffle(stochastics)

		if num_cores > 1:
			n_batches = int(np.ceil((len(org_nodes)/25)/(num_cores*25)))
		else:
			n_batches = int(np.ceil((len(org_nodes)/50)))
		for it in range(its):
			
			#Stochastiscism
			use_nodes = copy(org_nodes)
			if stochastics[it]:
				random.shuffle(use_nodes)
			
			new_aff_stack = []
			#multi_pre_chunks = np.array_split(org_nodes,4)
			#multi_pre_chunks = [(nodes,nxk.graphtools.subgraphAndNeighborsFromNodes(g, nodes, includeOutNeighbors=True, includeInNeighbors=True),max_deg,aff_cats,aff_map,dist_map) for nodes in multi_pre_chunks]
			#results = Pool(4).map(_batch_compute_affinity,multi_pre_chunks)
			ncount = 0
			save_all_counts = {}
			for pre_chunk in np.array_split(use_nodes,n_batches):
				batch_size = len(pre_chunk)
				n_labels = len(aff_cats)
				affs = np.zeros((batch_size, max_deg, n_labels))
				dists = np.zeros((batch_size, max_deg, n_labels))
				wss = np.zeros((batch_size, max_deg))

				for i, n in enumerate(pre_chunk):
					n_idx_list, ws_list = zip(*nn_idx[n])
					
					# Concatenate all the indices and weights
					n_idx_concat = np.concatenate(n_idx_list)
					ws_concat = np.concatenate(ws_list)

					# Compute the lengths of the original lists
					lengths = [len(n_idx) for n_idx in n_idx_list]

					# Compute the start and end indices for each segment in the concatenated arrays
					start_indices = np.cumsum([0] + lengths[:-1])
					end_indices = np.cumsum(lengths)
					
					for start, end in zip(start_indices, end_indices):
						len_segment = end - start
						affs[i, :len_segment, :] = aff_map[n_idx_concat[start:end], :]
						dists[i, :len_segment, :] = dist_map[n_idx_concat[start:end], :]
						wss[i, :len_segment] = ws_concat[start:end]
					save_all_counts[n]=ncount
					ncount+=1

				if num_cores > 1:
					for result in Pool(num_cores).map(_compute_affinity_multi,[(affs[idx_],dists[idx_],wss[idx_]) for idx_ in np.array_split([i for i in range(affs.shape[0])],num_cores)]):
						new_aff_stack.append(result)
				else:
					new_aff = _compute_affinity(affs,dists,wss)
					new_aff_stack.append(new_aff)

			new_aff_stack = np.concatenate(new_aff_stack,axis=0)
			if stochastics[it]:
				new_aff_stack = new_aff_stack[[v for k,v in sorted(save_all_counts.items())]]
			#new_aff_stack = np.nan_to_num(new_aff_stack / new_aff_stack.sum(axis=1)[:, np.newaxis], nan=0.0, posinf=0.0, neginf=0.0)
			for n,l in labels.items():
				new_aff_stack[n]=np.zeros(n_labels)
				new_aff_stack[n][aff_cats[l]]=1.0
			aff_map = new_aff_stack
			if verbose:
				print ("OSCAR")
				print (aff_map[net_idx["oscarlagansson7_Twitter"]])
				print (new_aff_stack[net_idx["oscarlagansson7_Twitter"]])
				print ()
				print ("KIMBO")
				print (aff_map[net_idx["Kimbo_Twitter"]])
				print (new_aff_stack[net_idx["Kimbo_Twitter"]])
				print ()
				print ("VANSTER")
				print (aff_map[net_idx["tatillbakavalfarden_Facebook Page"]])
				print (new_aff_stack[net_idx["tatillbakavalfarden_Facebook Page"]])
				print ()

		if verbose:
			print (aff_map[net_idx["Aktuellt Fokus_Facebook Page"]])
			print (aff_map[net_idx["Kimbo_Twitter"]])
			print (aff_map[net_idx["NordfrontSE_Telegram"]])
			print (aff_map[net_idx["oscarlagansson7_Twitter"]])
			end_time = time.time()
			print (end_time - start_time)
		affinities.append(aff_map)
	affinities = np.mean(np.array(affinities),axis=0)
	#out_affinities = {k:{title+"_"+lab:0.0 for lab in set(labels.values())} for k in net_idx.keys()}
	out_affinities = {title+"_"+lab:{k:0.0 for k in net_idx.keys()} for lab in set(labels.values())}
	for lab in set(labels.values()):
		for i,n in enumerate(affinities):
			#out_affinities[rev_net_ids[i]][title+"_"+lab]=n[aff_cats[lab]]
			out_affinities[title+"_"+lab][rev_net_ids[i]]=n[aff_cats[lab]]
	return {title:out_affinities}



@timer
def stlp(g,labels,net_idx,title="test",num_cores=1,its=5,epochs=3,stochasticity=1,verbose=False):

	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	max_deg = nxk.graphtools.maxDegree(g)
	org_nodes = list([n for n in g.iterNodes()])
	affinities = []
	distances = []
	all_dists = {}
	org_labels = {net_idx[n]:l for n,l in labels.items() if n in net_idx}
	labels = {}

	print ("generating neighbour idxs")
	nn_idx, nn_ws, nn_ns = _stlp_get_n_idx_ws_sorted([g.iterNeighborsWeights(n) for n in org_nodes],org_nodes)

	for epoch in range(epochs):
		print ("getting distances")
		if len(all_dists) > 0:
			for n,_ in noise_labels.items():
				del labels[n]
			noise_labels = {n:"noise" for n in random.sample([n for n in net_idx.values() if n not in org_labels and g.hasNode(n)],int(1.0*len(org_labels)))}
			labels.update(noise_labels)
			all_dists["noise"]=get_djik_distances(g,org_nodes,noise_labels)["noise"]
		else:
			noise_labels = {n:"noise" for n in random.sample([n for n in net_idx.values() if n not in org_labels and g.hasNode(n)],int(1.0*len(org_labels)))}
			labels.update(org_labels)
			labels.update(noise_labels)
			all_dists = get_djik_distances(g,org_nodes,labels)
		print ("distances set")

		aff_cats = {c:j for j,c in enumerate(sorted(list(set(labels.values()))))}
		dist_map = np.array([np.mean(((np.array(v)+1)**-1)**((np.array(v)+1)**(0.5)), axis=0) for k,v in sorted(all_dists.items())]).T
		aff_map = np.zeros((len(net_idx),len(all_dists)))
		for n,l in labels.items():
			aff_map[n][aff_cats[l]]=1.0
		n_labels = len(aff_cats)

		start_time = time.time()
		for it in range(its):
			new_aff_stack = []
			ncount = 0
			save_all_counts = {}
			nlens = list(nn_ns.keys())
			random.shuffle(nlens)
			for nlen in nlens:
				affs,dists,wss = _stlp_get_n_idx_ws_chunk(nn_idx[nlen],nn_ws[nlen],aff_map,dist_map)
				for i,n in enumerate(nn_ns[nlen]):
					save_all_counts[n]=ncount
					ncount+=1
				new_aff = _compute_affinity(affs,dists,wss)
				new_aff_stack.append(new_aff)

			new_aff_stack = np.concatenate(new_aff_stack,axis=0)
			new_aff_stack = new_aff_stack[[v for k,v in sorted(save_all_counts.items())]]
			#new_aff_stack = np.nan_to_num(new_aff_stack / new_aff_stack.sum(axis=1)[:, np.newaxis], nan=0.0, posinf=0.0, neginf=0.0)
			for n,l in labels.items():
				new_aff_stack[n]=np.zeros(n_labels)
				new_aff_stack[n][aff_cats[l]]=1.0
			aff_map = new_aff_stack
			if verbose:
				print ("OSCAR")
				print (aff_map[net_idx["oscarlagansson7_Twitter"]])
				print (new_aff_stack[net_idx["oscarlagansson7_Twitter"]])
				print ()
				print ("KIMBO")
				print (aff_map[net_idx["Kimbo_Twitter"]])
				print (new_aff_stack[net_idx["Kimbo_Twitter"]])
				print ()
				print ("VANSTER")
				print (aff_map[net_idx["tatillbakavalfarden_Facebook Page"]])
				print (new_aff_stack[net_idx["tatillbakavalfarden_Facebook Page"]])
				print ()

		if verbose:
			print (aff_map[net_idx["Aktuellt Fokus_Facebook Page"]])
			print (aff_map[net_idx["Kimbo_Twitter"]])
			print (aff_map[net_idx["NordfrontSE_Telegram"]])
			print (aff_map[net_idx["oscarlagansson7_Twitter"]])
			end_time = time.time()
			print (end_time - start_time)
		affinities.append(aff_map)
		distances.append(dist_map)
	affinities = np.mean(np.array(affinities),axis=0)
	distances = np.mean(np.array(distances),axis=0)
	#out_affinities = {k:{title+"_"+lab:0.0 for lab in set(labels.values())} for k in net_idx.keys()}
	out_affinities = {title+"_"+lab:{k:0.0 for k in net_idx.keys()} for lab in set(labels.values())}
	out_distances = {k:[] for k in net_idx.keys()}
	for lab in set(labels.values()):
		for i,n in enumerate(affinities):
			#out_affinities[rev_net_ids[i]][title+"_"+lab]=n[aff_cats[lab]]
			out_affinities[title+"_"+lab][rev_net_ids[i]]=n[aff_cats[lab]]
	for lab in set(labels.values()):
		for i,d in enumerate(distances):
			out_distances[rev_net_ids[i]].append(d)
	return {title:out_affinities},{title+"_"+"dist":{k:np.log(np.mean(np.array(v))) for k,v in out_distances.items()}}

def set_affinities(g,affinities):

	for n,affs in affinities.items():
		for aff,val in affs.items():
			if n in g:
				nx.set_node_attributes(g,{n:val},aff)
	return g





def _get_distances(args):

	print (1)
	g,org_nodes,net_idx,org_labels,labels,all_dists,noise_labels = args
	print ("getting distances")
	if len(all_dists) > 0:
		for n,_ in noise_labels.items():
			del labels[n]
		noise_labels = {n:"noise" for n in random.sample([n for n in net_idx.values() if n not in org_labels and g.hasNode(n)],int(1.0*len(org_labels)))}
		labels.update(noise_labels)
		all_dists["noise"]=get_djik_distances(g,org_nodes,noise_labels)["noise"]
	else:
		noise_labels = {n:"noise" for n in random.sample([n for n in net_idx.values() if n not in org_labels and g.hasNode(n)],int(1.0*len(org_labels)))}
		labels.update(org_labels)
		labels.update(noise_labels)
		all_dists = get_djik_distances(g,org_nodes,labels)
	print ("distances set")
	
	return all_dists,labels,noise_labels

def _stlp(net_idx,sorted_nns,labels,all_dists):

	nn_idx, nn_ws, nn_ns = sorted_nns
	its=8
	aff_cats = {c:j for j,c in enumerate(sorted(list(set(labels.values()))))}
	dist_map = np.array([np.mean(((np.array(v)+1)**-1)**((np.array(v)+1)**(0.5)), axis=0) for k,v in sorted(all_dists.items())]).T
	aff_map = np.zeros((len(net_idx),len(all_dists)))
	for n,l in labels.items():
		aff_map[n][aff_cats[l]]=1.0
	n_labels = len(aff_cats)

	start_time = time.time()
	for it in range(its):
		new_aff_stack = []
		ncount = 0
		save_all_counts = {}
		nlens = list(nn_ns.keys())
		random.shuffle(nlens)
		for nlen in nlens:
			affs,dists,wss = _stlp_get_n_idx_ws_chunk(nn_idx[nlen],nn_ws[nlen],aff_map,dist_map)
			for i,n in enumerate(nn_ns[nlen]):
				save_all_counts[n]=ncount
				ncount+=1
			new_aff = _compute_affinity(affs,dists,wss)
			new_aff_stack.append(new_aff)

		new_aff_stack = np.concatenate(new_aff_stack,axis=0)
		new_aff_stack = new_aff_stack[[v for k,v in sorted(save_all_counts.items())]]
		#new_aff_stack = np.nan_to_num(new_aff_stack / new_aff_stack.sum(axis=1)[:, np.newaxis], nan=0.0, posinf=0.0, neginf=0.0)
		for n,l in labels.items():
			new_aff_stack[n]=np.zeros(n_labels)
			new_aff_stack[n][aff_cats[l]]=1.0
		aff_map = new_aff_stack
	return aff_map,dist_map

@timer
def stlp_multi(g,labels,net_idx,title="test",num_cores=1,its=8,epochs=3,stochasticity=1,verbose=False):

	ncores = 2
	if net_idx is not None: rev_net_ids = create_rev_net_idx(g,net_idx)
	max_deg = nxk.graphtools.maxDegree(g)
	org_nodes = list([n for n in g.iterNodes()])
	affinities = []
	distances = []
	all_dists = {}
	org_labels = {net_idx[n]:l for n,l in labels.items() if n in net_idx}
	labels = {}

	print ("generating neighbour idxs")
	sorted_nns = _stlp_get_n_idx_ws_sorted([g.iterNeighborsWeights(n) for n in org_nodes],org_nodes)
	subgs = [nxk.graphtools.subgraphFromNodes(g,org_nodes) for r in range(ncores)]
	#subgs = [None for r in range(5)]
	all_dists,labels,noise_labels = _get_distances((g,org_nodes,net_idx,org_labels,labels,all_dists,{}))
	for epoch in range(epochs):
		dist_results = Pool(2).map(_get_distances,[(subgs[r],copy(org_nodes),copy(net_idx),deepcopy(org_labels),deepcopy(labels),deepcopy(all_dists),deepcopy(noise_labels)) for r in range(ncores)])
		print ("dsADsa")
		results = Pool(ncores).map(_stlp,[(net_idx,sorted_nns,dres[1],dres[0]) for dres in dist_results ])
		for aff_map,dist_map in results:
			affinities.append(aff_map)
			distances.append(dist_map)
	affinities = np.mean(np.array(affinities),axis=0)
	distances = np.mean(np.array(distances),axis=0)
	#out_affinities = {k:{title+"_"+lab:0.0 for lab in set(labels.values())} for k in net_idx.keys()}
	out_affinities = {title+"_"+lab:{k:0.0 for k in net_idx.keys()} for lab in set(labels.values())}
	out_distances = {k:[] for k in net_idx.keys()}
	for lab in set(labels.values()):
		for i,n in enumerate(affinities):
			#out_affinities[rev_net_ids[i]][title+"_"+lab]=n[aff_cats[lab]]
			out_affinities[title+"_"+lab][rev_net_ids[i]]=n[aff_cats[lab]]
	for lab in set(labels.values()):
		for i,d in enumerate(distances):
			out_distances[rev_net_ids[i]].append(d)
	return {title:out_affinities},{title+"_"+"dist":{k:np.log(np.mean(np.array(v))) for k,v in out_distances.items()}}
