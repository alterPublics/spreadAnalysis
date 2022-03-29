from spreadAnalysis.persistence.mongo import MongoSpread
from spreadAnalysis.analysis import batching as btc
import time
from datetime import datetime


if __name__ == "__main__":

	print ("starting update")
	start_time = time.time()
	all_time = time.time()
	mdb = MongoSpread()
	mdb.update_url_bi_network(new=False)
	#mdb.repart_url_bi_net()
	#sys.exit()
	print("--- %s seconds --- to update url network" % (time.time() - start_time))
	start_time = time.time()
	btc.update_actor_message(new=False)
	print("--- %s seconds --- to update actor messages" % (time.time() - start_time))
	start_time = time.time()
	btc.update_agg_actor_metrics(skip_existing=False,num_cores=8)
	print("--- %s seconds --- to update actor metrics" % (time.time() - start_time))
	print("--- %s seconds --- to run entire update" % (time.time() - all_time))
	print (datetime.today())
