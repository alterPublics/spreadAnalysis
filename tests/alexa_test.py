from spreadAnalysis.some.alexa import Alexa
from spreadAnalysis.io.config_io import Config

actor = "friatider.se"
conf = Config()
yt = Alexa(conf.get_auth()["alexa"])
yt.url_referals(actor)
