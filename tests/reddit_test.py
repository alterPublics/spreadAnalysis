from spreadAnalysis.some.reddit import Reddit
from spreadAnalysis.io.config_io import Config

conf = Config()
rdit= Reddit()

url = "https://konfront.dk/et-hvidt-tryghedshold-kalder-den-hvide-del-af-venstrefloejen-ind/"
url = "https://rt.com"
url = "https://piopio.dk"
url = "https://faktum-magazin.de"
#data = rdit.url_referals(url,start_date="2017-06-01")
data = rdit.domain_referals(url,start_date="2021-06-01")
#print (data)
#print (len(data))
#print (len(data["output"]))

#query = '"danske værdier"'
#data = rdit.query_content(query,start_date="2021-12-01")
print (data)
print (len(data["output"]))
