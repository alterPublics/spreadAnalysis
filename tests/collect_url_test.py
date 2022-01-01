from spreadAnalysis.some.crowdtangle import Crowdtangle
from spreadAnalysis.some.twitter2 import Twitter2
from spreadAnalysis.persistence.simple import FlatFile
from spreadAnalysis.utils.link_utils import LinkCleaner
from spreadAnalysis.io.config_io import Config
from spreadAnalysis.scraper.scraper import Scraper
from spreadAnalysis.utils.pd_utils import *

import pandas as pd

# SWEDEN
#main_path = "/Users/jakobbk/Documents/postdoc/mainProject/exploration/sweden/data_collection"
#input_file = main_path+"/{0}".format("2021-05-26-18-45-58-NZST-Historical-Report-swedeninput-2018-12-31--2021-05-26.csv")
#output_file = main_path+"/{0}".format("referal_data.p")
#backup_file = main_path+"/{0}".format("referal_data_backup.p")
#url_match_file = main_path+"/{0}".format("cleaned_urls.p")

"""# DENMARK
main_path = "/Users/jakobbk/Documents/postdoc/mainProject/exploration/denmark/data_collection"
input_file = main_path+"/{0}".format("2021-06-01-19-32-48-NZST-Historical-Report-denmarkinput-2018-12-31--2021-06-01.csv")
output_file = main_path+"/{0}".format("referal_data.p")
backup_file = main_path+"/{0}".format("referal_data_backup.p")
url_match_file = main_path+"/{0}".format("cleaned_urls.p")"""

# AUSTRIA
main_path = "/Users/jakobbk/Documents/postdoc/mainProject/exploration/austria/data_collection"
input_file = main_path+"/{0}".format("2021-06-01-19-32-48-NZST-Historical-Report-denmarkinput-2018-12-31--2021-06-01.csv")
output_file = main_path+"/{0}".format("referal_data.p")
backup_file = main_path+"/{0}".format("referal_data_backup.p")
url_match_file = main_path+"/{0}".format("cleaned_urls.p")

main_data = FlatFile(output_file)
backup_data = FlatFile(backup_file)
url_data = FlatFile(url_match_file)

input_df = pd.read_csv(input_file)
input_df["Shares"] = pd.to_numeric(input_df["Shares"], downcast="float")
input_df = input_df[(input_df["Shares"] >= 1.0)]
input_df = input_df[~(pd.isna(input_df["Link"]))]
input_df = filter_on_overperform(input_df,"Page Name","Shares",factor=0.75)
input_df.sort_values("Post Created Date", axis=0, ascending=True, inplace=True)

conf = Config()
ct = Crowdtangle(conf.get_auth()["crowdtangle"])
tw = Twitter2(conf.get_auth()["twitter2"])
scrp = Scraper(settings={"change_user_agent":True,"exe_path":conf.CHROMEDRIVER})
scrp.browser_init()
lc = LinkCleaner(scraper=scrp)

url_list = list(input_df["Link"])

url_count = 0
iterate_data = {}
for org_url in url_list:
    if org_url not in main_data.data:
        url_count+=1

        if org_url in url_data.data and url_data.data[org_url] is not None:
            cleaned_url = url_data.data[org_url]["unpacked"]
        else:
            clean_data = lc.clean_url(org_url)
            url_data.simple_update({org_url:clean_data})
            if clean_data is not None:
                cleaned_url = url_data.data[org_url]["unpacked"]
            else:
                cleaned_url = None

        if cleaned_url is not None:
            print (cleaned_url)
            try:
                ct_ref_data = ct.url_referals(cleaned_url,start_date="2010-01-01")
                tw_ref_data = tw.url_referals(cleaned_url,start_date="2010-01-01")
                iterate_data.update({org_url:{"crowdtangle":ct_ref_data,
                    "twitter":tw_ref_data,
                    "total_referals":len(ct_ref_data["output"])+len(tw_ref_data["output"])}})
            except:
                print ("ERROR")

        else:
            print ("**** NONE **** : "+str(org_url))

        if url_count % 10 == 0:
            main_data.simple_update(iterate_data)
            print ("Referals collected so far: {0} from {1}".format(sum([ref["total_referals"]
                for ref in list(main_data.data.values())]),len(main_data.data)))
        if url_count % 100 == 0:
            backup_data.simple_update(iterate_data)
