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
from multiprocessing import Process, Queue

processes = []


@csrf_exempt  # avoid cookies check in Postman
def train(request): 

    if request.method == 'POST':
        body = json.loads(request.body)

        classes = body['taxonomy']
        campaignId = body['campaignId']
        campaignInfoUrl = 'http://api.url:port/campaigns/' + campaignId + '/images'

        result = requests.get(campaignInfoUrl)
        imagesInfo = json.loads(result.text)

        print('Image samples: ', len(
            [a for a in imagesInfo if a['annotations']]))

        campaign_link = 'http://api.url:port/campaigns/' + campaignId + '/'
        p = Process(target=start_train_process, args=('train', 'coco',
                                                    campaignId, classes, imagesInfo, campaign_link, ))
        processes.append(p)
        print("---PROCESS COUNT:" + str(len(processes)) + "---")
        p.start()

        return JsonResponse({'training': 1, 'process_name': p.name})


def start_train_process(cmd, base_model, campaignId, classes, imagesInfo, campaign_link):
    runtrain.train_main(cmd, base_model, campaignId,
                        classes, imagesInfo, campaign_link)

    # remove downloaded images
    campaign_dir = os.getcwd() + '/campaigns/' + campaignId + '/'
    shutil.rmtree(campaign_dir)
    
    # end process
    exit(1)
