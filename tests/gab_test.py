from spreadAnalysis.some.gab import Gab
from spreadAnalysis.io.config_io import Config

conf = Config()
gab = Gab(conf.get_auth()["gab"])

url = "https://konfront.dk/et-hvidt-tryghedshold-kalder-den-hvide-del-af-venstrefloejen-ind/"
url = "https://rt.com"
actor = "https://gab.com/MorKarins"
#actor = "NaturalNews"
#actor = "RT"
data = gab.actor_content(actor,start_date="2019-08-10")
#data = rdit.domain_referals(url,start_date="2017-06-01")
#print (len(data))
print (data)
print (len(data["output"]))
