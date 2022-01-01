from spreadAnalysis.utils import helpers as hlp
import selenium
import timeout_decorator
from timeout_decorator import TimeoutError
import random

class ScraperHelpers:

    def __init__(self):
        self.browser = None

    def _match_element(self,elem,attr_match_pairs):

        elems_found = []
        try:
            elem.get_attribute('innerHTML')
        except selenium.common.exceptions.StaleElementReferenceException:
            return elems_found
        for attr_pairs in attr_match_pairs:
            for attr_v, attr_k in attr_pairs.items():
                if attr_k is not None and hasattr(elem,attr_k):
                    if attr_v in getattr(elem,attr_k):
                        elems_found.append(elem)
                elif attr_k == "text":
                    if attr_v in str(elem.text):
                        elems_found.append(elem)
                else:
                    #print (str(elem.get_attribute('innerHTML')))
                    #print (attr_v)
                    if attr_v in str(elem.get_attribute('innerHTML')):
                        #print ("ELEM FOUND")
                        elems_found.append(elem)
        return elems_found

    @timeout_decorator.timeout(seconds=30)
    def go_to_url(self,url):

        try:
            self.browser.get(url)
            hlp.random_wait(between=(1,3))
        except TimeoutError as te:
            print ("TIMED OUT")
            return None
        except Exception as e:
            print (e)

    @timeout_decorator.timeout(seconds=30)
    def get_page_source(self):

        try:
            page_source = self.browser.page_source
            #hlp.random_wait(between=(1,2))
        except TimeoutError as te:
            print ("TIMED OUT")
            page_source = None
        except Exception as e:
            print (e)
            page_source = None

        return page_source

    def get_elements(self,type="class",attr="box"):

        if type == "class":
            elems = self.browser.find_elements_by_class_name(attr)
        if type == "id":
            try:
                elems = [self.browser.find_element_by_id(attr)]
            except:
                elems = []
        if type == "link_text":
            elems = self.browser.find_elements_by_partial_link_text(attr)
        if type == "xpath":
            elems = self.browser.find_elements_by_xpath(attr)
        elif type == "tag":
            elems = self.browser.find_elements_by_tag_name(attr)
        if type == "css":
            elems = self.browser.find_elements_by_css_selector(attr)

        return elems

    def click_elem_until_disappear(self,type="class",attr="box",child_tag=None,attr_match_pairs=[],max_attempts=None):

        elems = self.get_elements(type=type,attr=attr)
        times_clicked = 0
        tried_times = 0
        while len(elems) > 0:
            for elem in elems:
                #print (str(elem.get_attribute('innerHTML')))
                if child_tag is not None:
                    if isinstance(child_tag, list):
                        for cht in child_tag:
                            elem = elem.find_element_by_tag_name(cht)
                    else:
                        elem = elem.find_element_by_tag_name(child_tag)
                if len(attr_match_pairs) > 0:
                    elems_found = self._match_element(elem,attr_match_pairs)
                    if len(elems_found) == len(attr_match_pairs):
                        self.click_single_element(elems_found[0])
                        times_clicked+=1
                        hlp.random_wait(between=(2,4))
                        elems = self.get_elements(type=type,attr=attr)
                        break
            tried_times+=1
            if times_clicked < 1:
                print ("no elements found to click")
                return
            if max_attempts is not None and tried_times >= max_attempts:
                print ("max_attempts reached")
                return

    def get_to_page_bottom(self,wait_=1.9):

        wait_ = hlp.random_wait(between=(wait_,wait_*2.0))
        lenOfPage = self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
        match=False
        while(match==False):
            time.sleep(wait_)
            lastCount = lenOfPage
            lenOfPage = self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
            if lastCount==lenOfPage:
                match=True

    def scroll_down(self,wait_=(1,3)):

        lenOfPage = self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
        hlp.random_wait(between=wait_)
        return lenOfPage

    def click_single_element(self,elem):

        elem.click()
        #self.browser.execute_script("arguments[0].click();", elem)

    def click_elements(self,type="class",attr="box",child_tag=None,attr_match_pairs=[],max_click=None):

        elems = self.get_elements(type=type,attr=attr)
        click_count = 0
        for elem in elems:
            if child_tag is not None:
                elem = elem.find_element_by_tag_name(child_tag)
            if len(attr_match_pairs) > 0:
                elems_found = self._match_element(elem,attr_match_pairs)
                if len(elems_found) == len(attr_match_pairs):
                    if max_click is not None and click_count > max_click:
                        break
                    #print (str(elem.get_attribute('innerHTML')))
                    self.click_single_element(elems_found[0])
                    click_count += 1
                    hlp.random_wait((0,1),skip_wait=0.6)
        hlp.random_wait(between=(2,4))
