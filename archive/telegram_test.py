from spreadAnalysis.some.telegram import Telegram
from spreadAnalysis.io.config_io import Config
import time

"""query = {'@type': 'getAuthorizationState', '@extra': {'request_id': 'getAuthorizationState'}}

tel = Telegram(Config().TELEGRAM_BINARY)
tel.send_to_api(query)
time.sleep(0.2)
result = tel.get_from_api(query)
print (result)
result = tel.get_from_api(query)
print (result)

setParams = {'@type': 'setTdlibParameters', 'parameters': {'use_test_dc': False, 'api_id': '5084073', 'api_hash': 'b5a39aebffdf27fbe4fa81e7ebde634c', 'device_model': 'spreadAnalysis-telegram', 'system_version': 'unknown', 'application_version': '0.2.0', 'system_language_code': 'en', 'database_directory': '/tmp/.tdlib_files/bb3429cb2180e6677d4c7e8909c2e279/database', 'use_message_database': True, 'files_directory': '/tmp/.tdlib_files/bb3429cb2180e6677d4c7e8909c2e279/files', 'use_secret_chats': True}, '@extra': {'request_id': 'updateAuthorizationState'}}

query = {'@type': 'getAuthorizationState', '@extra': {'request_id': 'getAuthorizationState'}}"""

kontrast_gid = 1266517652

conf = Config()
tel = Telegram(conf.TELEGRAM_BINARY,conf.get_auth()["telegram"])
tel.authorize()
#tel.send_to_api({'@type': "messages.searchGlobal","q":"covid"})
#tel.send_to_api({'@type': "searchPublicChats","query_":"covid"})
#tel.send_to_api({'@type': "getChats","offset_order":0,"offset_chat_id":0,"limit":100})
#tel.send_to_api({'@type': "searchMessages","chat_list_":"chatListMain","query":"donation","offset_date_":0,"offset_chat_id":0,"limit":100,"offset_message_id_":0,"min_date_":0,"max_date_":0})
#tel.send_to_api({'@type': "searchPublicChats","query_":"politics"})
tel.send_to_api({'@type': "getSupergroup","supergroup_id":kontrast_gid})
#tel.send_to_api({'@type': "getMe"})
#tel.send_to_api({'@type': "getChats"})
#1088417529
me = tel.get_from_api(None)
print (me)
while me is not None:
    me = tel.get_from_api(None)
    print (me)
    print ("")
    print ("")
    """if "'title': " in str(me) and "'supergroup_id': " in str(me):
        gid = str(me).split("'supergroup_id': ")[-1].split(",")[0]
        tit = str(me).split("'title': ")[-1].split(",")[0]
        print (tit + " : " + gid)
    if True:
        print (str(me))"""
tel.send_to_api({'@type': "close"})
