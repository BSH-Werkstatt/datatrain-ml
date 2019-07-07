from django.http import JsonResponse
import urllib
import json
import uuid 
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt # avoid cookies check in Postman
def healthcheck(request):
	if request.method == 'GET':
		return JsonResponse({'healthy': True})
