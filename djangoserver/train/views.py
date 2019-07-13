from django.http import HttpResponse
from django.http import JsonResponse
from django.http import FileResponse
import requests
import urllib
import json
import uuid 
from django.views.decorators.csrf import csrf_exempt

import model_setup as m

@csrf_exempt # avoid cookies check in Postman
def train(request):
	if request.method == 'POST':
		body = json.loads(request.body)
		print(body)
		classes = body['taxonomy']
		campaignId = body['campaignId']
		#url = request.build_absolute_uri()
		#campaignId = url.rsplit('/', 1)[-2] # url is ...<campaignId>/train
		campaignInfoUrl = 'http://ios19bsh.ase.in.tum.de/dev/api/campaigns/' + campaignId + '/images'
		result = requests.get(campaignInfoUrl) #('http://ios19bsh.ase.in.tum.de/dev/api/campaigns/5d276e02d3ad9e10b8864893/images')
		imagesInfo = json.loads(result.text)
		print('Image count: ', len([a for a in imagesInfo if a['annotations']]))
		from run import run_main
		#campaignId = '5d276e02d3ad9e10b8864893'
		#classes = ["Tomato","Lime","Kohlrabi","Kiwi","Iceberg Lettuce","Ginger","Eggplant","Cucumber","Cauliflower","Banana"]
		run_main('train', 'coco', campaignId, classes, imagesInfo, dataset='http://ios19bsh.ase.in.tum.de/dev/api/campaigns/' + campaignId + '/')
		return JsonResponse({'training':1})

