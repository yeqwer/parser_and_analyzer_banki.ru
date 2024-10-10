from bs4 import BeautifulSoup
from os import path
import lxml
import requests
import asyncio
import aiohttp
import time
import json
import os 



main_url = "https://www.banki.ru/services/responses/list/?type=all"
review_url = "https://www.banki.ru/services/responses/bank/response/"
json_file = ""
# proxy_list = "http://147.45.104.252:80"
proxies_list = {
  "http" : "http://81.177.160.200:80",
  # "https" : "https://67.43.228.254:2679"
}
remove_file_before_start = False 
parse_mode_all = True #if this variable true "parse_count_all" was ignored
parse_count_all = 10000
parse_count_step = 50



def getResponseTitle(soup):
  return soup.find_all("h1", {"class":"text-header-0 le856f50c"})[0].text 

def getResponseRating(soup):
  try:
    return soup.find("div", {"class":"text-size-4 text-weight-bold"}).findNext("div").text
  except:
    return "Без оценки"

def getResponseText(soup):
  text_module = soup.find("p")
  full_response_text = text_module.text
  try: 
    text_module = text_module.find_next_siblings("p")
    for tm in text_module:
      full_response_text = full_response_text + "\n" + tm.text
  finally:
    return full_response_text

def getResponseBankName(soup):
  img = soup.find("img", alt=True)
  return img['alt']

def getResponseDate(soup):
  return soup.find_all("span", {"class":"l10fac986"})[0].text.split(" ")[0]

def getResponseTime(soup):
  return soup.find_all("span", {"class":"l10fac986"})[0].text.split(" ")[1]

def getResponseViews(soup):
  return soup.find_all("span", {"class":"l10fac986"})[1].text

def getResponseAllInfo(soup, response, url):
  if response.ok:
    return {
      "ID" : "" + url.split("/")[7],
      "STATUS CODE" : "" + str(response.status),
      "BANK NAME" : " ".join(getResponseBankName(soup).split()),
      "POST DATE" : " ".join(getResponseDate(soup).split()),
      "POST TIME" : " ".join(getResponseTime(soup).split()),
      "TITLE" : " ".join(getResponseTitle(soup).split()),
      "TEXT" : " ".join(getResponseText(soup).split()),
      "RATING" : " ".join(getResponseRating(soup).split()),
      "VIEWS" : " ".join(getResponseViews(soup).split())
    }
  else:
    return {
      "ID" : "" + url.split("/")[7],
      "STATUS CODE" : "" + str(response.status),
    }

def createJson(file_name="data.json"):
  start_info = {"responses": [ ]}
  with open(file_name, 'w') as fp:
    json.dump(start_info, fp, indent = 2, ensure_ascii=False)
  return os.path.abspath(file_name)

def writeJson(new_data, json_file="data.json"):
  with open(json_file,'r+') as file:
    file_data = json.load(file)
    file_data["responses"].append(new_data)
    file.seek(0)
    json.dump(file_data, file, indent = 2, ensure_ascii=False)

async def fetch(session, url):
  async with session.get(url, proxy = "http://51.89.134.69:80") as response:
    html = await response.text()
    soup = BeautifulSoup(html, "lxml")
    writeJson(getResponseAllInfo(soup, response, url), json_file)

async def get_html(urls):
  start_time = time.time()
  #
  async with aiohttp.ClientSession() as session:
    # async with session.get(urls[0], proxy = "http://81.177.160.200:80") as response:
    #   print(response.json())
    tasks = []
    for url in urls:
      tasks.append(fetch(session, url))

    return await asyncio.gather(*tasks)
  
  #
  end_time = time.time()
  print(f"Time taken: {end_time - start_time} seconds")

async def main():
  last_review_id = BeautifulSoup(requests.get(main_url, proxies = proxies_list).text, "lxml").find("a", {"class":"link-simple"}, href=True)["href"].split("/")[5]
  id_urls = []
  _parse_count_all = parse_count_all
  if parse_mode_all: 
    _parse_count_all = last_review_id
    print("Parse all - True, total = "+str(_parse_count_all)) #debug
  else:
    print("Parse all - False, total = "+str(_parse_count_all)) #debug
  
  count = 0 #debug
  for step in range(int(last_review_id), int(last_review_id) - int(_parse_count_all), -parse_count_step):
    
    for id in range(step, step - parse_count_step, -1):
      id_urls.append(review_url+str(id)+"/")
      count = count + 1
      print(f"Already finish: {count}") #debug
    task = asyncio.create_task(get_html(id_urls))
    await task
    id_urls.clear()



# if path.isfile(json_file) is False:
#   json_file = createJson("data.json")
#   asyncio.run(main())
# else:
#   if remove_file_before_start:
#     os.remove(json_file)
#   asyncio.run(main())

proxy_id = requests.get(main_url, proxies = proxies_list, stream=True)
print(proxy_id.raw._connection.sock.getsockname())