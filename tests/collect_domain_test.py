from spreadAnalysis.some.crowdtangle import Crowdtangle
from spreadAnalysis.some.twitter2 import Twitter2
from spreadAnalysis.persistence.simple import FlatFile
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.io.config_io import Config
from spreadAnalysis.scraper.scraper import Scraper
from spreadAnalysis.utils.pd_utils import *
from spreadAnalysis.utils import helpers as hlp

import pandas as pd
import time

# SWEDEN
"""main_path = "/Users/jakobbk/Documents/postdoc/mainProject/exploration/sweden/data_collection"
input_file = main_path+"/{0}".format("Actorlist_SWE.xlsx")
output_file = main_path+"/{0}".format("referal_data.p")
backup_file = main_path+"/{0}".format("referal_data_backup.p")
url_match_file = main_path+"/{0}".format("cleaned_urls.p")"""

# DENMARK
main_path = "/Users/jakobbk/Documents/postdoc/mainProject/exploration/denmark/data_collection"
input_file = main_path+"/{0}".format("Actorlist_DK.xlsx")
output_file = main_path+"/{0}".format("referal_data.p")
backup_file = main_path+"/{0}".format("referal_data_backup.p")
url_match_file = main_path+"/{0}".format("cleaned_urls.p")

main_data = FlatFile(output_file)
backup_data = FlatFile(backup_file)
url_data = FlatFile(url_match_file)

conf = Config()
ct = Crowdtangle(conf.get_auth()["crowdtangle"])
tw = Twitter2(conf.get_auth()["twitter2"])
include_iterations = set(["0"])

input_list = pd.read_excel(input_file)
input_list = input_list.where(pd.notnull(input_list), None)
data_to_update = {}
for i,row in list(input_list.iterrows()):
    if row["Iteration"] is not None and str(int(row["Iteration"])) in include_iterations:
        domain = row["Website"]
        if domain is not None:
            ct_data = ct.domain_referals(domain,start_date="2019-01-01",max_results=None)
            url_count = 0
            for url_dat in sorted(ct_data, key=lambda x: len(x["output"]), reverse=True)[:400]:
                if url_dat["input"] not in main_data.data and url_dat["input"]+"/" not in main_data.data:
                    try:
                        url_count += 1
                        tw_data = tw.url_referals(url_dat["input"],start_date="2019-01-01")
                        temp_dat = {url_dat["input"]:{"crowdtangle":url_dat,"twitter":tw_data,
                                    "total_referals":len(url_dat["output"])+len(tw_data["output"])}}
                        data_to_update.update(temp_dat)
                        if url_count % 5 == 0:
                            main_data.simple_update(data_to_update)
                            print ("Referals collected so far: {0} from {1}".format(sum([ref["total_referals"]
                                for ref in list(data_to_update.values())]),len(data_to_update)))
                    except:
                        print ("ERROR!")
                        time.sleep(2)
            main_data.simple_update(data_to_update)
            backup_data.simple_update(data_to_update)
