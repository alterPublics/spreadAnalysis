from spreadAnalysis.some.twitter2 import Twitter2
from spreadAnalysis.io.config_io import Config
import sys

actor_urls = ["https://twitter.com/samhallsnytt","https://twitter.com/sdriks"]
actor_urls = ["https://twitter.com/Mikkel_oerum"]
actor_urls = ["https://twitter.com/Exakt24"]
conf = Config()
tw = Twitter2(conf.get_auth()["twitter2"])

query = '"danske værdier"'
data = tw.query_content(query,start_date="2021-12-12")
print (data)
print (len(data["output"]))
sys.exit()

#actors = tw._user_urls_to_ID(actor_urls)
actors = ["https://twitter.com/Alpenschau"]
actors = ["https://twitter.com/netavisen180"]
actors = ["https://twitter.com/SpiegelAnti"]

url = "https://dagensblaeser.net/"
#url = "https://dagensblaeser.net/2020/07/13/boligformand-i-rotterede-skuffet-over-hans-rotter"

data = tw.domain_referals(url,start_date="2019-01-01")
print (data)
sys.exit()

for actor in actors:
    data = tw.actor_content(actor,start_date="2021-09-01")

    print (data)
    print (len(data["output"]))
    break