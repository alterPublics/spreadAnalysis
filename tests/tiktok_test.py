from spreadAnalysis.some.tiktok import Tiktok
from spreadAnalysis.io.config_io import Config

tt = Tiktok()

actor = "https://www.tiktok.com/@infodirekt?"

data = tt.actor_content(actor,start_date="2022-01-10")
print (data)
print (len(data["output"]))
sys.exit()

query = 'fravær'
data = tt.query_content(query,start_date="2021-10-01")
print (data)
print (len(data["output"]))
