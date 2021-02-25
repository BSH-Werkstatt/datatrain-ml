import os
from django.http import HttpResponse
from django.http import JsonResponse
from django.http import FileResponse
import urllib
import json
import uuid 
from django.views.decorators.csrf import csrf_exempt

import model_setup as m
from multiprocessing import Process, Queue

processes = []

@csrf_exempt # avoid cookies check in Postman
def predict(request):
	if request.method == 'POST':
		print("Starting process for " + str(request.body))
		randomName = str(uuid.uuid4()) + '.jpg'

		urlTrack = request.build_absolute_uri() # URL where request was sent
		filename = urlTrack + randomName # assign random filename to image 

		p = Process(target=predict_process, args=(request, filename, ))
		processes.append(p)
		print("---PROCESS COUNT:" + str(len(processes)) + "---")

		p.start()
		p.join()
		processes.remove(p)
		print("---PROCESS COUNT:" + str(len(processes)) + "---")
		
		return JsonResponse({'predictionURL': filename})

def predict_process(request, randomName):
	body = json.loads(request.body)
	url = body['imageURL']

	model_name = body['campaignId'] + ".h5"
	campaign_class_names = ['BG']
	for class_name in body['campaignTaxonomy']:
		campaign_class_names.append(class_name)
	rcnn, class_names = m.define_model(model_name, campaign_class_names)
	prediction, image = m.predict_img(url, rcnn)
	m.visualize(prediction, image, class_names, randomName)

	exit(1)


@csrf_exempt
def predictReturnFile(request):
	if request.method == 'GET':
		#body = json.loads(request.body)
		#url = body['predictionURL']
		url = request.build_absolute_uri()
		filename = url.rsplit('/', 1)[-1]
		with open(filename, 'rb') as f:
			fileContent = f.read()
		responseImg =  HttpResponse(fileContent, content_type='image/jpeg') #FileResponse(open(filename, 'rb')) 
		os.remove(filename)
		return responseImg
