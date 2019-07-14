from django.http import HttpResponse
from django.http import JsonResponse
from django.http import FileResponse
import urllib
import json
import uuid 
from django.views.decorators.csrf import csrf_exempt

import model_setup as m

@csrf_exempt # avoid cookies check in Postman
def predict(request):
	if request.method == 'POST':
		body = json.loads(request.body)
		url = body['imageURL']
		urlTrack = request.build_absolute_uri() # URL where request was sent

		model_name = body['campaignId'] + ".h5"
		campaign_class_names = ['BG']
		for class_name in body['campaignTaxonomy']:
			campaign_class_names.append(class_name)

		rcnn, class_names = m.define_model(model_name, campaign_class_names)
		prediction, image = m.predict_img(url, rcnn)
		randomName = str(uuid.uuid4()) + '.jpg'
		filename = urlTrack + randomName # assign random filename to image 
		m.visualize(prediction, image, class_names, randomName)
		return JsonResponse({'predictionURL':filename})

@csrf_exempt
def predictReturnFile(request):
	if request.method == 'GET':
		#body = json.loads(request.body)
		#url = body['predictionURL']
		url = request.build_absolute_uri()
		filename = url.rsplit('/', 1)[-1]
		return FileResponse(open(filename, 'rb'))  #response
