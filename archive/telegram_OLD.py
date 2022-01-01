from spreadAnalysis.reqs.req import Req
import spreadAnalysis.utils.helpers as hlp
from ctypes import CDLL, CFUNCTYPE, c_int, c_char_p, c_double, c_void_p, c_longlong
import pkg_resources
import json
import random
import sys

class Telegram:

    def __init__(self,lib_path,auth):

        self.lib_path = lib_path
        self.auth = random.choice(auth["tokens"])
        self._load_tdjson_binary()

        self.start_query = {'@type': 'getAuthorizationState',
            '@extra': {'request_id': 'getAuthorizationState'}}
        self.tdlib_query = {'@type': 'setTdlibParameters', 'parameters': {'use_test_dc': False,
         'api_id': str(self.auth["api_id"]),
         'api_hash': str(self.auth["api_hash"]),
         'device_model': 'spreadAnalysis-telegram',
         'system_version': 'unknown',
         'application_version': '0.2.0',
         'system_language_code': 'en',
         'database_directory': '/tmp/.tdlib_files/bb3429cb2180e6677d4c7e8909c2e279/database',
         'use_message_database': True, 'files_directory': '/tmp/.tdlib_files/bb3429cb2180e6677d4c7e8909c2e279/files',
         'use_secret_chats': True},
            '@extra': {'request_id': 'updateAuthorizationState'}}
        self.crypt_query = {'@type': 'checkDatabaseEncryptionKey',
            'encryption_key': str(self.auth["encryption_key"]),
            '@extra': {'request_id': 'updateAuthorizationState'}}
        self.phone_query = {
            '@type': 'setAuthenticationPhoneNumber',
            'phone_number': str(self.auth["phone_number"]),
            'allow_flash_call': False,
            'is_current_phone_number': True,
        }
        self.code_query = {'@type': 'checkAuthenticationCode', 'code': str(self.auth["code"])}

    def _load_tdjson_binary(self):

        self._tdjson = CDLL(self.lib_path)
        self._td_json_client_create = self._tdjson.td_json_client_create
        self._td_json_client_create.restype = c_void_p
        self._td_json_client_create.argtypes = []
        self._td_json_client_send = self._tdjson.td_json_client_send
        self._td_json_client_send.restype = None
        self._td_json_client_send.argtypes = [c_void_p, c_char_p]
        self._td_json_client_receive = self._tdjson.td_json_client_receive
        self._td_json_client_receive.restype = c_char_p
        self._td_json_client_receive.argtypes = [c_void_p, c_double]
        self._td_set_log_verbosity_level = self._tdjson.td_set_log_verbosity_level
        self._td_set_log_verbosity_level.restype = None
        self._td_set_log_verbosity_level.argtypes = [c_int]

        self._td_set_log_verbosity_level(2)
        self.td_json_client = self._td_json_client_create()

    def authorize(self):

        auth_ready = False
        for query in [self.start_query,self.tdlib_query,self.crypt_query,self.phone_query,self.code_query]:
            print (query)
            self.send_to_api(query)
            result = "empty"
            while result is not None:
                result = self.get_from_api(query)
                print (result)
                if result is not None:
                    if "authorization_state" in result and result["authorization_state"]["@type"] == "authorizationStateReady":
                        auth_ready = True
                        break

                    if "authorization_state" in result and result["authorization_state"]["@type"] == "authorizationStateWaitCode":
                        print ("Telegram code needs to be updated!")
                        break
            if auth_ready:
                break
        if not auth_ready:
            print ("update info and try again.")

    def send_to_api(self,query):

        self._td_json_client_send(self.td_json_client, json.dumps(query).encode('utf-8'))

    def get_from_api(self,query):

        result = None
        res = self._td_json_client_receive(self.td_json_client, 1.0)
        if res is not None:
            result = json.loads(res.decode('utf-8'))
        return result

    def get_url_referals(self):

        pass
