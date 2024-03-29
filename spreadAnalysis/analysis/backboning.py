import sys, warnings
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from scipy.stats import binom
from multiprocessing import Pool, Manager

def read(filename, column_of_interest, triangular_input = False, consider_self_loops = True, undirected = False, drop_zeroes = True, sep = ","):
	"""Reads a field separated input file into the internal backboning format (a Pandas Dataframe).
	The input file should have three or more columns (default separator: tab).
	The input file must have a one line header with the column names.
	There must be two columns called 'src' and 'trg', indicating the origin and destination of the interaction.
	All other columns must contain integer or floats, indicating the edge weight.
	In case of undirected network, the edges have to be present in both directions with the same weights, or set triangular_input to True.

	Args:
	filename (str): The path to the file containing the edges.
	column_of_interest (str): The column name identifying the weight that will be used for the backboning.

	KWArgs:
	triangular_input (bool): Is the network undirected and are the edges present only in one direction? default: False
	consider_self_loops (bool): Do you want to consider self loops when calculating the backbone? default: True
	undirected (bool): Is the network undirected? default: False
	drop_zeroes (bool): Do you want to keep zero weighted connections in the network? Important: it affects methods based on degree, like disparity_filter. default: False
	sep (char): The field separator of the inout file. default: tab

	Returns:
	The parsed network data, the number of nodes in the network and the number of edges.
	"""
	if isinstance(filename,str):
		table = pd.read_csv(filename, sep = sep)
	else:
		table = filename
	table = table[["src", "trg", column_of_interest]]
	table.rename(columns = {column_of_interest: "nij"}, inplace = True)
	if drop_zeroes:
		table = table[table["nij"] > 0]
	if not consider_self_loops:
		table = table[table["src"] != table["trg"]]
	if triangular_input:
		table2 = table.copy()
		table2["new_src"] = table["trg"]
		table2["new_trg"] = table["src"]
		table2.drop("src", 1, inplace = True)
		table2.drop("trg", 1, inplace = True)
		table2 = table2.rename(columns = {"new_src": "src", "new_trg": "trg"})
		table = pd.concat([table, table2], axis = 0)
		table = table.drop_duplicates(subset = ["src", "trg"])
	original_nodes = len(set(table["src"]) | set(table["trg"]))
	original_edges = table.shape[0]
	if undirected:
		return table, original_nodes, original_edges / 2
	else:
		return table, original_nodes, original_edges

def thresholding(table, threshold, keep_percent=None, visualize_max=1000000, skip_actors=None):
	"""Reads a preprocessed edge table and returns only the edges supassing a significance threshold.

	Args:
	table (pandas.DataFrame): The edge table.
	threshold (float): The minimum significance to include the edge in the backbone.

	Returns:
	The network backbone.
	"""
	print ("thresholding...")
	table = table.copy()
	if keep_percent is not None:
		table["score"] = table["score"] - (float(threshold) * table["sdev_cij"])
		keep_table = table.nlargest(int(len(table)*keep_percent),"score")[["src", "trg", "nij", "score"]]
		if visualize_max is not None and len(keep_table) > visualize_max:
			while len(keep_table) > visualize_max:
				keep_percent = keep_percent*0.5
				keep_table = table.nlargest(int(len(table)*keep_percent),"score")[["src", "trg", "nij", "score"]]
		if skip_actors is not None:
			wrong_skips = [a for a in skip_actors if a not in set(list(keep_table["src"])) and a not in set(list(keep_table["trg"]))]
			new_table = table[(table["trg"].isin(wrong_skips)) | (table["src"].isin(wrong_skips))]
			keep_table = pd.concat([keep_table,new_table.nlargest(int(len(new_table)*keep_percent),"score")[["src", "trg", "nij", "score"]]])
		return keep_table
	if "sdev_cij" in table:
		return table[(table["score"] - (float(threshold) * table["sdev_cij"])) > 0][["src", "trg", "nij", "score"]]
	else:
		return table[table["score"] > threshold][["src", "trg", "nij", "score"]]

def write(table, network, method, folder):
	if not table.empty and "src" in table:
		table.to_csv("%s/%s_%s.csv" % (folder, network, method), sep = "\t", index = False)
	else:
		warnings.warn("Incorrect/empty output. Nothing written on disk", RuntimeWarning)

def stability_jac(table1, table2):
	table1_edges = set(zip(table1["src"], table1["trg"]))
	table2_edges = set(zip(table2["src"], table2["trg"]))
	return float(len(table1_edges & table2_edges)) / len(table1_edges | table2_edges)

def stability_corr(table1, table2, method = "spearman", log = False, what = "nij"):
	corr_table = table1.merge(table2, on = ["src", "trg"])
	corr_table = corr_table[["%s_x" % what, "%s_y" % what]]
	if log:
		corr_table["%s_x" % what] = np.log(corr_table["%s_x" % what])
		corr_table["%s_y" % what] = np.log(corr_table["%s_y" % what])
	return corr_table["%s_x" % what].corr(corr_table["%s_y" % what], method = method)

def test_densities(table, start, end, step):
	if start > end:
		raise ValueError("start must be lower than end")
	steps = []
	x = start
	while x <= end:
		steps.append(x)
		x += step
	onodes = len(set(table["src"]) | set(table["trg"]))
	oedges = table.shape[0]
	oavgdeg = (2.0 * oedges) / onodes
	for s in steps:
		edge_table = thresholding(table, s)
		nodes = len(set(edge_table["src"]) | set(edge_table["trg"]))
		edges = edge_table.shape[0]
		avgdeg = (2.0 * edges) / nodes
		yield (s, nodes, (100.0 * nodes) / onodes, edges, (100.0 * edges) / oedges, avgdeg, avgdeg / oavgdeg)

def noise_corrected(table, undirected = False, return_self_loops = False, calculate_p_value = False, num_cores=12):
	if num_cores > 1:
		sys.stderr.write("Calculating NC score...\n")
		#table = table.copy()
		#trg_sum = table.groupby(["trg"]).apply_parallel(_by_sum, num_processes=num_cores)
		#src_sum = table.groupby(["src"]).apply_parallel(_by_sum, num_processes=num_cores)
		#trg_sum = table.groupby(by = "trg").sum()[["nij"]]
		#src_sum = table.groupby(by = "src").sum()[["nij"]]
		trg_sum, src_sum = _multi_group_by(table,["trg","src"],"nij")
		table = table.merge(trg_sum, left_on = "trg", right_index = True, suffixes = ("", "_trg_sum"))
		table = table.merge(src_sum, left_on = "src", right_index = True, suffixes = ("", "_src_sum"))
		table.rename(columns = {"nij_src_sum": "ni.", "nij_trg_sum": "n.j"}, inplace = True)
		table["n.."] = table["nij"].sum()
		#table["mean_prior_probability"] = ((table["ni."] * table["n.j"]) / table["n.."]) * (1 / table["n.."])
		#table["kappa"] = table["n.."] / (table["ni."] * table["n.j"])
		table = _multi_funcs(table,["mean_prior_probability","kappa"],[_mean_prior_prob,_kappa])
		table = _multi_funcs(table,["score","var_prior_probability"],[_score,_var_prior_probability])
		table = _multi_funcs(table,["alpha_prior","beta_prior"],[_alpha_prior,_beta_prior])
		table = _multi_funcs(table,["alpha_post","beta_post"],[_alpha_post,_beta_post])
		table["expected_pij"] = table["alpha_post"] / (table["alpha_post"] + table["beta_post"])
		table["variance_nij"] = table["expected_pij"] * (1 - table["expected_pij"]) * table["n.."]
		#table = _multi_funcs(table,["d","variance_cij"],[_d,_variance_cij])
		table["d"] = (1.0 / (table["ni."] * table["n.j"])) - (table["n.."] * ((table["ni."] + table["n.j"]) / ((table["ni."] * table["n.j"]) ** 2)))
		table["variance_cij"] = table["variance_nij"] * (((2 * (table["kappa"] + (table["nij"] * table["d"]))) / (((table["kappa"] * table["nij"]) + 1) ** 2)) ** 2)
		table["sdev_cij"] = table["variance_cij"] ** .5
		if not return_self_loops:
			table = table[table["src"] != table["trg"]]
		if undirected:
			table = table[table["src"] <= table["trg"]]
	else:
		sys.stderr.write("Calculating NC score...\n")
		table = table.copy()
		print ("Grouping...")
		src_sum = table.groupby(by = "src").sum()[["nij"]]
		table = table.merge(src_sum, left_on = "src", right_index = True, suffixes = ("", "_src_sum"))
		trg_sum = table.groupby(by = "trg").sum()[["nij"]]
		table = table.merge(trg_sum, left_on = "trg", right_index = True, suffixes = ("", "_trg_sum"))
		table.rename(columns = {"nij_src_sum": "ni.", "nij_trg_sum": "n.j"}, inplace = True)
		print ("Getting Means...")
		table["n.."] = table["nij"].sum()
		table["mean_prior_probability"] = ((table["ni."] * table["n.j"]) / table["n.."]) * (1 / table["n.."])
		if calculate_p_value:
			table["score"] = binom.cdf(table["nij"], table["n.."], table["mean_prior_probability"])
			return table[["src", "trg", "nij", "score"]]
		print ("Getting Kappa...")
		table["kappa"] = table["n.."] / (table["ni."] * table["n.j"])
		print ("Getting score...")
		table["score"] = ((table["kappa"] * table["nij"]) - 1) / ((table["kappa"] * table["nij"]) + 1)
		print ("Getting var_prior...")
		table["var_prior_probability"] = (1 / (table["n.."] ** 2)) * (table["ni."] * table["n.j"] * (table["n.."] - table["ni."]) * (table["n.."] - table["n.j"])) / ((table["n.."] ** 2) * ((table["n.."] - 1)))
		print ("Getting alpha prior...")
		table["alpha_prior"] = (((table["mean_prior_probability"] ** 2) / table["var_prior_probability"]) * (1 - table["mean_prior_probability"])) - table["mean_prior_probability"]
		print ("Getting beta_prior...")
		table["beta_prior"] = (table["mean_prior_probability"] / table["var_prior_probability"]) * (1 - (table["mean_prior_probability"] ** 2)) - (1 - table["mean_prior_probability"])
		table.drop('mean_prior_probability', axis=1, inplace=True)
		table.drop('var_prior_probability', axis=1, inplace=True)
		print ("Getting alpha post...")
		table["alpha_post"] = table["alpha_prior"] + table["nij"]
		table.drop('alpha_prior', axis=1, inplace=True)
		print ("Getting beta post...")
		table["beta_post"] = table["n.."] - table["nij"] + table["beta_prior"]
		table.drop('beta_prior', axis=1, inplace=True)
		print ("Getting pij...")
		table["expected_pij"] = table["alpha_post"] / (table["alpha_post"] + table["beta_post"])
		table.drop('alpha_post', axis=1, inplace=True)
		table.drop('beta_post', axis=1, inplace=True)
		print ("Getting nij...")
		table["variance_nij"] = table["expected_pij"] * (1 - table["expected_pij"]) * table["n.."]
		table.drop('expected_pij', axis=1, inplace=True)
		print ("Getting d...")
		table["d"] = (1.0 / (table["ni."] * table["n.j"])) - (table["n.."] * ((table["ni."] + table["n.j"]) / ((table["ni."] * table["n.j"]) ** 2)))
		print ("Getting cij...")
		table["variance_cij"] = table["variance_nij"] * (((2 * (table["kappa"] + (table["nij"] * table["d"]))) / (((table["kappa"] * table["nij"]) + 1) ** 2)) ** 2)
		print ("Getting sdev_cij...")
		table["sdev_cij"] = table["variance_cij"] ** .5
		if not return_self_loops:
			table = table[table["src"] != table["trg"]]
		if undirected:
			table = table[table["src"] <= table["trg"]]

	return table[["src", "trg", "nij", "score", "sdev_cij"]]

def _multi_get_result(args):

	df = args[0]
	func = args[1]
	return func(df)

def _multi_funcs(df,vars,funcs):

	results = Pool(len(vars)).map(_multi_get_result,[(df,func) for func in funcs])
	for var,result in zip(vars,results):
		df[var]=result
	return df

def _multi_get_group(args):

	by = args[0]
	s = args[1]
	df = args[2]
	return df.groupby(by = by).sum()[[s]]

def _multi_group_by(df,bys,ss):

	results = Pool(len(bys)).map(_multi_get_group,[(by,ss,df) for by in bys])
	return [r for r in results]

def noise_corrected_NEW2(table, undirected = False, return_self_loops = False, calculate_p_value = False, num_cores=12):
	sys.stderr.write("Calculating NC score...\n")
	#table = table.copy()
	#trg_sum = table.groupby(["trg"]).apply_parallel(_by_sum, num_processes=num_cores)
	#src_sum = table.groupby(["src"]).apply_parallel(_by_sum, num_processes=num_cores)
	trg_sum = table.groupby(by = "trg").sum()[["nij"]]
	src_sum = table.groupby(by = "src").sum()[["nij"]]
	table = table.merge(trg_sum, left_on = "trg", right_index = True, suffixes = ("", "_trg_sum"))
	table = table.merge(src_sum, left_on = "src", right_index = True, suffixes = ("", "_src_sum"))
	table.rename(columns = {"nij_src_sum": "ni.", "nij_trg_sum": "n.j"}, inplace = True)
	table["n.."] = table["nij"].sum()
	table["mean_prior_probability"] = table.apply_parallel(_mean_prior_prob, num_processes=num_cores, axis=0)
	if calculate_p_value:
		table["score"] = binom.cdf(table["nij"], table["n.."], table["mean_prior_probability"])
		return table[["src", "trg", "nij", "score"]]
	table["kappa"]  = table.apply_parallel(_kappa, num_processes=num_cores, axis=0)
	table["score"]  = table.apply_parallel(_score, num_processes=num_cores, axis=0)
	table["var_prior_probability"]  = table.apply_parallel(_var_prior_probability, num_processes=num_cores, axis=0)
	table["alpha_prior"]  = table.apply_parallel(_alpha_prior, num_processes=num_cores, axis=0)
	table["beta_prior"]  = table.apply_parallel(_beta_prior, num_processes=num_cores, axis=0)
	table["alpha_post"]  = table.apply_parallel(_alpha_post, num_processes=num_cores, axis=0)
	table["beta_post"] = table.apply_parallel(_beta_post, num_processes=num_cores, axis=0)
	table["expected_pij"]  = table.apply_parallel(_expected_pij, num_processes=num_cores, axis=0)
	table["variance_nij"]  = table.apply_parallel(_variance_nij, num_processes=num_cores, axis=0)
	table["d"] = table.apply_parallel(_d, num_processes=num_cores, axis=0)
	table["variance_cij"]  = table.apply_parallel(_variance_cij, num_processes=num_cores, axis=0)
	table["sdev_cij"]  = table.apply_parallel(_sdev_cij, num_processes=num_cores, axis=0)
	if not return_self_loops:
		table = table[table["src"] != table["trg"]]
	if undirected:
		table = table[table["src"] <= table["trg"]]
	return table[["src", "trg", "nij", "score", "sdev_cij"]]

def _mean_prior_prob(data_row):

	return ((data_row["ni."] * data_row["n.j"]) / data_row["n.."]) * (1 / data_row["n.."])

def _kappa(data_row):

	return data_row["n.."] / (data_row["ni."] * data_row["n.j"])

def _score(data_row):

	return ((data_row["kappa"] * data_row["nij"]) - 1) / ((data_row["kappa"] * data_row["nij"]) + 1)

def _var_prior_probability(data_row):

	return (1 / (data_row["n.."] ** 2)) * (data_row["ni."] * data_row["n.j"] * (data_row["n.."] - data_row["ni."]) * (data_row["n.."] - data_row["n.j"])) / ((data_row["n.."] ** 2) * ((data_row["n.."] - 1)))

def _alpha_prior(data_row):

	return (((data_row["mean_prior_probability"] ** 2) / data_row["var_prior_probability"]) * (1 - data_row["mean_prior_probability"])) - data_row["mean_prior_probability"]

def _beta_prior(data_row):

	return (data_row["mean_prior_probability"] / data_row["var_prior_probability"]) * (1 - (data_row["mean_prior_probability"] ** 2)) - (1 - data_row["mean_prior_probability"])

def _alpha_post(data_row):

	return data_row["alpha_prior"] + data_row["nij"]

def _beta_post(data_row):

	return data_row["n.."] - data_row["nij"] + data_row["beta_prior"]

def _expected_pij(data_row):

	return data_row["alpha_post"] / (data_row["alpha_post"] + data_row["beta_post"])

def _variance_nij(data_row):

	return data_row["expected_pij"] * (1 - data_row["expected_pij"]) * data_row["n.."]

def _d(data_row):

	return (1.0 / (data_row["ni."] * data_row["n.j"])) - (data_row["n.."] * ((data_row["ni."] + data_row["n.j"]) / ((data_row["ni."] * data_row["n.j"]) ** 2)))

def _variance_cij(data_row):

	return data_row["variance_nij"] * (((2 * (data_row["kappa"] + (data_row["nij"] * data_row["d"]))) / (((data_row["kappa"] * data_row["nij"]) + 1) ** 2)) ** 2)

def _sdev_cij(data_row):

	return data_row["variance_cij"] ** .5

def _by_sum(df):
	return pd.Series([df['nij'].sum()])

def noise_corrected_NEW(table, undirected = False, return_self_loops = False, calculate_p_value = False, num_cores=12):
	sys.stderr.write("Calculating NC score...\n")
	#table = table.copy()

	trg_sum = table.groupby(by = "trg").apply_parallel(_by_sum, num_processes=num_cores)
	src_sum = table.groupby(by = "src").apply_parallel(_by_sum, num_processes=num_cores)
	table = table.merge(trg_sum, left_on = "trg", right_index = True, suffixes = ("", "_trg_sum"))
	table = table.merge(src_sum, left_on = "src", right_index = True, suffixes = ("", "_src_sum"))
	table.rename(columns = {"nij_src_sum": "ni.", "nij_trg_sum": "n.j"}, inplace = True)
	table["n.."] = table["nij"].sum()
	table["mean_prior_probability"] = multi_process(func=_mean_prior_prob,data=table,num_process=num_cores,verbose=False)["mean_prior_probability"]
	if calculate_p_value:
		table["score"] = binom.cdf(table["nij"], table["n.."], table["mean_prior_probability"])
		return table[["src", "trg", "nij", "score"]]
	table["kappa"] = multi_process(func=_kappa,data=table,num_process=num_cores,verbose=False)["kappa"]
	table["score"] = multi_process(func=_score,data=table,num_process=num_cores,verbose=False)["score"]
	table["var_prior_probability"] = multi_process(func=_var_prior_probability,data=table,num_process=num_cores,verbose=False)["var_prior_probability"]
	table["alpha_prior"] = multi_process(func=_alpha_prior,data=table,num_process=num_cores,verbose=False)["alpha_prior"]
	table["beta_prior"] = multi_process(func=_beta_prior,data=table,num_process=num_cores,verbose=False)["beta_prior"]
	table["alpha_post"] = multi_process(func=_alpha_post,data=table,num_process=num_cores,verbose=False)["alpha_post"]
	table["beta_post"] = multi_process(func=_beta_post,data=table,num_process=num_cores,verbose=False)["beta_post"]
	table["expected_pij"] = multi_process(func=_expected_pij,data=table,num_process=num_cores,verbose=False)["expected_pij"]
	table["variance_nij"] = multi_process(func=_variance_nij,data=table,num_process=num_cores,verbose=False)["variance_nij"]
	table["d"] = multi_process(func=_d,data=table,num_process=num_cores,verbose=False)["d"]
	table["variance_cij"] = multi_process(func=_variance_cij,data=table,num_process=num_cores,verbose=False)["variance_cij"]
	table["sdev_cij"] = multi_process(func=_sdev_cij,data=table,num_process=num_cores,verbose=False)["sdev_cij"]
	if not return_self_loops:
		table = table[table["src"] != table["trg"]]
	if undirected:
		table = table[table["src"] <= table["trg"]]
	return table[["src", "trg", "nij", "score", "sdev_cij"]]

def disparity_filter(table, undirected = False, return_self_loops = False):
	sys.stderr.write("Calculating DF score...\n")
	table = table.copy()
	table_sum = table.groupby(table["src"]).sum().reset_index()
	table_deg = table.groupby(table["src"]).count()["trg"].reset_index()
	table = table.merge(table_sum, on = "src", how = "left", suffixes = ("", "_sum"))
	table = table.merge(table_deg, on = "src", how = "left", suffixes = ("", "_count"))
	table["score"] = 1.0 - ((1.0 - (table["nij"] / table["nij_sum"])) ** (table["trg_count"] - 1))
	table["variance"] = (table["trg_count"] ** 2) * (((20 + (4.0 * table["trg_count"])) / ((table["trg_count"] + 1.0) * (table["trg_count"] + 2) * (table["trg_count"] + 3))) - ((4.0) / ((table["trg_count"] + 1.0) ** 2)))
	if not return_self_loops:
		table = table[table["src"] != table["trg"]]
	if undirected:
		table["edge"] = table.apply(lambda x: "%s-%s" % (min(x["src"], x["trg"]), max(x["src"], x["trg"])), axis = 1)
		table_maxscore = table.groupby(by = "edge")["score"].max().reset_index()
		table_minvar = table.groupby(by = "edge")["variance"].min().reset_index()
		table = table.merge(table_maxscore, on = "edge", suffixes = ("_min", ""))
		table = table.merge(table_minvar, on = "edge", suffixes = ("_max", ""))
		table = table.drop_duplicates(subset = ["edge"])
		table = table.drop("edge", 1)
		table = table.drop("score_min", 1)
		table = table.drop("variance_max", 1)
	return table[["src", "trg", "nij", "score", "variance"]]
