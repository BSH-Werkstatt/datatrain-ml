from django.http import HttpResponse
from django.http import JsonResponse
from django.http import FileResponse
import requests
import urllib
import json
import uuid
from django.views.decorators.csrf import csrf_exempt

from . import runtrain
import model_setup as m
import shutil
import os
from threading import Thread

threads = []


@csrf_exempt  # avoid cookies check in Postman
def train(request):

    if request.method == 'POST':
        body = json.loads(request.body)

        classes = body['taxonomy']
        campaignId = body['campaignId']
        campaignInfoUrl = 'http://datatrain-api-736295320.eu-central-1.elb.amazonaws.com/campaigns/' + campaignId + '/images'

        result = requests.get(campaignInfoUrl)
        imagesInfo = json.loads(result.text)

        print('Image samples: ', len(
            [a for a in imagesInfo if a['annotations']]))

        campaign_link = 'http://datatrain-api-736295320.eu-central-1.elb.amazonaws.com/campaigns/' + campaignId + '/'
        t = Thread(target=start_train_thread, args=('train', 'coco',
                                                    campaignId, classes, imagesInfo, campaign_link, ))
        threads.append(t)
        print("---THREAD COUNT:" + str(len(threads)) + "---")
        t.start()

        return JsonResponse({'training': 1, 'thread_name': t.getName()})


def start_train_thread(cmd, base_model, campaignId, classes, imagesInfo, campaign_link):
    runtrain.train_main(cmd, base_model, campaignId,
                        classes, imagesInfo, campaign_link)

    campaign_dir = os.getcwd() + '/campaigns/' + campaignId + '/'
    shutil.rmtree(campaign_dir)
