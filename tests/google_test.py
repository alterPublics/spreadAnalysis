from spreadAnalysis.some.google import Google
from spreadAnalysis.io.config_io import Config

url = "https://redox.dk/nyheder/nye-borgerlige-nynazister-er-velkomne/"
conf = Config()
goo = Google(conf.get_auth()["google"])
#ref_data = goo.url_referals(url)
query = '"danske værdier"'
ref_data = goo.query_content(query,start_date="2021-12-12")
print (ref_data)
print (len(ref_data["output"]))
