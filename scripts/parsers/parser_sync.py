import os
import json
import requests
from bs4 import BeautifulSoup
from os import path

json_file = './responses.json'
last_response_dom = BeautifulSoup(requests.get("https://www.banki.ru/services/responses/list/?type=all").text, "lxml")
last_response_object = last_response_dom.find("a", {"class":"link-simple"}, href=True)
response_id = last_response_object["href"].split("/")[5]

# parse_count = int(input("How much responces you need? \n"))
parse_count = 0

class Response(): 
  def __init__(sl,current_id):
    sl.current_id = current_id
    sl.req = requests.get("https://www.banki.ru/services/responses/bank/response/"+current_id+"/")
    sl.soap = BeautifulSoup((sl.req).text,"lxml")

  def getResponseId(self):
    return self.current_id

  def getResponseTitle(self):
    return self.soap.find_all("h1", {"class":"text-header-0 le856f50c"})[0].text 

  def getResponseRating(self):
    try:
      return self.soap.find("div", {"class":"text-size-4 text-weight-bold"}).findNext("div").text
    except:
      return "Без оценки"

  def getResponseText(self):
    text_module = self.soap.find("p")
    full_response_text = text_module.text
    try: 
      text_module = text_module.find_next_siblings("p")
      for tm in text_module:
        full_response_text = full_response_text + "\n" + tm.text
    finally:
      return full_response_text
  
  def getResponseBankName(self):
    img = self.soap.find("img", alt=True)
    return img['alt']

  def getResponseDateAndTime(self, dateOrTime):
      
    date_time = self.soap.find_all("span", {"class":"l10fac986"})[0].text
    match dateOrTime:
      case "date":
        return date_time.split(" ")[0]
      case "time": 
        return date_time.split(" ")[1]
      case _:
        return "CAN'T FIND DATE OR TIME"

  def getResponseViews(self):
    return self.soap.find_all("span", {"class":"l10fac986"})[1].text

  def getResponseAllInfo(self):
    if  self.req.ok:
      return {
      "ID" : ""+self.getResponseId(),
      "STATUS CODE" : "" + str(self.req.status_code),
      "BANK NAME" : " ".join(self.getResponseBankName().split()),
      "POST DATE" : " ".join(self.getResponseDateAndTime("date").split()),
      "POST TIME" : " ".join(self.getResponseDateAndTime("time").split()),
      "TITLE" : " ".join(self.getResponseTitle().split()),
      "TEXT" : " ".join(self.getResponseText().split()),
      "RATING" : " ".join(self.getResponseRating().split()),
      "VIEWS" : " ".join(self.getResponseViews().split())
      }
    else: 
      print("miss")

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

def main(parse_count):
  print("start")
  for current_id in range(int(response_id), 0, -1):
    print("Parsed: " + str(parse_count+1))
    writeJson(Response(str(current_id)).getResponseAllInfo(), json_file)
    parse_count = parse_count + 1

if path.isfile(json_file) is False:
  json_file = createJson("data.json")

main(parse_count)