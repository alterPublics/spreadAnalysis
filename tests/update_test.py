from spreadAnalysis.some.crowdtangle_app import CrowdtangleApp
from spreadAnalysis.some.twitter2 import Twitter2
from spreadAnalysis.persistence.simple import FlatFile
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.io.config_io import Config
from spreadAnalysis.scraper.scraper import Scraper
from spreadAnalysis.utils.pd_utils import *
from spreadAnalysis.utils import helpers as hlp
from spreadAnalysis.some.google import Google
import sys
import time
from operator import itemgetter

import pandas as pd

# SWEDEN
"""main_path = "/Users/jakobbk/Documents/postdoc/mainProject/exploration/sweden/data_collection"
input_file = main_path+"/{0}".format("2021-05-26-18-45-58-NZST-Historical-Report-swedeninput-2018-12-31--2021-05-26.csv")
output_file = main_path+"/{0}".format("referal_data.p")
backup_file = main_path+"/{0}".format("referal_data_backup.p")
url_match_file = main_path+"/{0}".format("cleaned_urls.p")"""

# DENMARK
main_path = "/Users/jakobbk/Documents/postdoc/mainProject/exploration/denmark/data_collection"
output_file = main_path+"/{0}".format("referal_data.p")
backup_file = main_path+"/{0}".format("referal_data_backup.p")
url_match_file = main_path+"/{0}".format("cleaned_urls.p")

conf = Config()
ctapp = CrowdtangleApp(conf.get_auth()["crowdtangle_app"])
goo = Google(conf.get_auth()["google"])

main_data = FlatFile(output_file)
backup_data = FlatFile(backup_file)
url_data = FlatFile(url_match_file)

#del main_data.data["total_referals"]
#main_data.simple_update({})

new_referals = 0
url_count = 1
input_data = [(url,dat) for url,dat in sorted(list(main_data.data.items()), key=lambda x: len(x[1]["crowdtangle"]["output"])+len(x[1]["twitter"]["output"]), reverse=True)]
for inp,dat in input_data:
    if inp in url_data.data and url_data.data[inp] is not None:
        cleaned_url = url_data.data[inp]["unpacked"]
    else:
        cleaned_url = inp
    print (cleaned_url)
    try:
        if not "crowdtangle_app" in dat:
            url_count+=1
            ct_app_data = ctapp.url_referals(cleaned_url)
            main_data.data[inp]["crowdtangle_app"]=ct_app_data
            main_data.data[inp]["total_referals"]+=len(ct_app_data["output"])
            new_referals+=len(ct_app_data["output"])
        if not "google" in dat:
            google_data = goo.url_referals(cleaned_url)
            main_data.data[inp]["google"]=google_data
            main_data.data[inp]["total_referals"]+=len(google_data["output"])
            new_referals+=len(google_data["output"])
    except:
        print ("ERROR")
        time.sleep(4)

    if url_count % 20 == 0:
        print ("new referals: {0} - url count: {1}".format(str(new_referals),str(url_count)))
        main_data.simple_update({})
        url_count+=1
    if url_count % 100 == 0:
        backup_data.data = {}
        backup_data.simple_update(main_data.data)
        #print ("Exiting")
        #sys.exit()
main_data.simple_update({})

for inp,dat in list(main_data.data.items()):
    n_methods = 0
    if "crowdtangle_app" in dat:
        if len(dat["crowdtangle_app"]["output"]) > 0:
            n_methods+=1
    if "google" in dat:
        if len(dat["google"]["output"]) > 1:
            n_methods+=1
    if "crowdtangle" in dat:
        if len(dat["crowdtangle"]["output"]) > 1:
            n_methods+=1
    if "twitter" in dat:
        if len(dat["crowdtangle"]["output"]) > 1:
            n_methods+=1

    if n_methods >= 4:
        for method in ["crowdtangle_app","google","crowdtangle","twitter"]:
            print (dat[method]["output"][0])
            print ()
        sys.exit()
