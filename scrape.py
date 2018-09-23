from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import re
import json
import sys

tag = sys.argv[1]
pages=[int(sys.argv[2]), int(sys.argv[3])]

def simple_get(url):
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        print('Error during requests to {0} : {1}'.format(url, str(e)))
        return None

def is_good_response(resp):
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)

for p in range(pages[0], pages[1]+1):
    #cnt = simple_get('https://konachan.com/post?page=%d&tags=%s' % (p, tag))
    cnt = simple_get('https://yande.re/post?page=%d&tags=%s' % (p, tag))
    soup = BeautifulSoup(cnt, 'html.parser')
    for an in soup.select('a.thumb'):
        #cnt_img = simple_get('https://konachan.com/' + an.attrs['href'])
        cnt_img = simple_get('https://yande.re/' + an.attrs['href'])
        soup_img = BeautifulSoup(cnt_img, 'html.parser')
        print(soup_img.select_one('#image').attrs['src'], flush=True)
