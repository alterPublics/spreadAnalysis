import requests
from urllib.parse import quote_plus,unquote
from requests.exceptions import HTTPError
import time
import sys

class Req:

    def __init__(self):

        pass

    @staticmethod
    def get_response(url,timeout=30,max_tries=3,wait_time=0,fail_wait_time=10,params=None,auth=None,headers=None,retry_except=[]):

        def try_again(wait_time,try_n):
            time.sleep(wait_time+5)
            print (f"Trying again... attempt no. {try_n}")

        response = None
        i = 0
        while i < max_tries:
            i += 1
            time.sleep(wait_time)
            try:
                response = requests.get(url,params=params,auth=auth,headers=headers,timeout=timeout)
            except:
                pass
            try:
                response.raise_for_status()
                break
            except HTTPError as http_err:
                if int(response.status_code) in retry_except:
                    break
                elif response.status_code == 404:
                    print (str(response.status_code) + " SOMETHING MISSING IN CALL URL")
                    break
                elif response.status_code != 404:
                    try:
                        if "code" in response.json() and int(response.json()["code"]) in retry_except:
                            break
                    except Exception as e:
                        print (e)
                else:
                    print (http_err)
                    try_again(fail_wait_time,i)
            except Exception as e:
                print (e)
                try_again(fail_wait_time,i)
        return response
