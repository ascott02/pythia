import requests
import os

filename = "./COCO_val2014_000000391895.jpg"
page = 'http://localhost:8080/upload'
token = ""

def send_data_to_server():
    multipart_form_data = {
        'token': ('', str(token)),
        'img_file': (os.path.basename(filename),  open(filename, 'rb')),
    }
 
    response = requests.post(page, files=multipart_form_data)
    print(response.status_code, response.text)

send_data_to_server()


