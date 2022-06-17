from spreadAnalysis.some.reddit import Reddit
from spreadAnalysis.io.config_io import Config

conf = Config()
rdit= Reddit()

url = "https://konfront.dk/et-hvidt-tryghedshold-kalder-den-hvide-del-af-venstrefloejen-ind/"
url = "https://rt.com"
url = "https://piopio.dk"
url = "https://faktum-magazin.de"
url = "https://www.youtube.com/watch?v=pQLcFs6H9NU"
url = "thelancet.com"
#data = rdit.url_referals(url,start_date="2000-06-01")
data = rdit.domain_referals(url,start_date="2022-03-01")
#print (data)
#print (len(data))
#print (len(data["output"]))

#query = '"danske værdier"'
#data = rdit.query_content(query,start_date="2021-12-01")
print (data)
print (len(data["output"]))
