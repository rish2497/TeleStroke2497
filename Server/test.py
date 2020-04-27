import requests
import time

'''
your localhost url. If running on port 5000
'''
###Test for PainDetectionModel
url = "http://localhost:5000/processface"
# Path to image file
filess = {"img": open("PainClassificationModel/PAIN-samples/4.jpg", "rb")}
starttime = time.time()
results = requests.post(url, files=filess)
print("time taken:", time.time() - starttime)
print(results.text)

###Test for HandDetectionModel
url = "http://localhost:5000/processhand"
# Path to image file
filess = {"img": open("HandClassificationModel/Stretched/0.jpg", "rb")}
starttime = time.time()
results = requests.post(url, files=filess)
print("time taken:", time.time() - starttime)
print(results.text)

###Test for HandDetectionModel
url = "http://localhost:5000/analysisreport"
# Path to image file
filess = {"img": open("HandClassificationModel/Stretched/0.jpg", "rb")}
starttime = time.time()
results = requests.post(url, files=filess)
print("time taken:", time.time() - starttime)
print(results.text)
