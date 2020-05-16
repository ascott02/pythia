import requests
image_url = "https://github.com/ascott02/vsepp/raw/master/COCO_val2014_000000391895.jpg"
page = 'http://localhost:8080/api'
token = 'J6FLgLawuqzbEHsGzxm35GumUNGk4gbZAQ2WrcWdet4zLDFFesMERbgT4LqHwrGK'

def send_data_to_server():
    data = {"token": str(token), "image_url": str(image_url)}

    response = requests.post(page, data)
    print(response.status_code, response.text)

send_data_to_server()

