from spreadAnalysis.some.vkontakte import Vkontakte
from spreadAnalysis.io.config_io import Config

actor = "anonymousnews_org"
conf = Config()
vk = Vkontakte(conf.get_auth()["vkontakte"])
data = vk.actor_content("352484777",start_date="2022-03-01")
print (data)
print (len(data["output"]))
sys.exit()

query = 'støjberg'
data = vk.query_content(query,start_date="2021-12-01")
print (data)
print (len(data["output"]))
sys.exit()
#print (vk.update_actor_info(["-202864751"],verbose=True))
#sys.exit()
data = vk.actor_content("https://vk.com/nordfront_danmark",start_date="2021-01-01")
#data = vk.url_referals("https://report24.news")
print (data)
print (len(data))
print (len(data["output"]))
