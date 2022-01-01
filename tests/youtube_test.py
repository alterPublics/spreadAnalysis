from spreadAnalysis.some.youtube import Youtube
from spreadAnalysis.io.config_io import Config

#actor = "https://www.youtube.com/channel/UCO2ikjgXF6tVqaiwZeDDltw"
actor = "UC03g0H9JfNI97CRDSSvIs3w"
conf = Config()
yt = Youtube(conf.get_auth()["youtube"])
actor_data = yt.actor_content(actor)
print (actor_data)
print (len(actor_data["output"]))
