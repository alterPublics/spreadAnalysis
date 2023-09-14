"""from telethon import TelegramClient, events, sync
from spreadAnalysis.io.config_io import Config

conf = Config()
api_id = conf.get_auth()["telegram"]["tokens"][0]["api_id"]
api_hash = conf.get_auth()["telegram"]["tokens"][0]["api_hash"]
client = TelegramClient('main_session', api_id, api_hash)
client.start()

async def get_messages():
    counter = 0
    async for message in client.iter_messages(None,search="Siegfried"):
        print (message.to_dict())
        print ()
        counter += 1
        if counter > 30:
            break

with client:
    client.loop.run_until_complete(get_messages())"""


from telethon import TelegramClient, events, sync
from spreadAnalysis.io.config_io import Config
from spreadAnalysis.some.telegram import Telegram

class Teltest:

    def __init__(self):
        pass

    def run(self):

        conf = Config()
        tg = Telegram(conf.get_auth()["telegram"])
        #print (tg.actor_content("https://t.me/alpenschau_aktuell",start_date="2021-12-01"))
        try:
            print (tg.actor_content("https://t.me/wokedealerpinkarisa",start_date="2022-01-01"))
        except Exception as e:
            print (e)
        query = 'tjeneste'
        #data = tg.query_content(query,start_date="2021-12-01")
        #print (data)
        #print (len(data["output"]))

def main():
    Teltest().run()

main()
