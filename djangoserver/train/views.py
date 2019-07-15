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

from threading import Thread

threads = []


@csrf_exempt  # avoid cookies check in Postman
def train(request):

    if request.method == 'POST':
        body = json.loads(request.body)
        classes = body['taxonomy']
        campaignId = body['campaignId']
        #url = request.build_absolute_uri()
        # campaignId = url.rsplit('/', 1)[-2] # url is ...<campaignId>/train
        campaignInfoUrl = 'http://ios19bsh.ase.in.tum.de/dev/api/campaigns/' + \
            campaignId + '/images'
        # ('http://ios19bsh.ase.in.tum.de/dev/api/campaigns/5d276e02d3ad9e10b8864893/images')
        result = requests.get(campaignInfoUrl)
        imagesInfo = json.loads(result.text)
        print('Image samples: ', len(
            [a for a in imagesInfo if a['annotations']]))
        #campaignId = '5d276e02d3ad9e10b8864893'
        #classes = ["Tomato","Lime","Kohlrabi","Kiwi","Iceberg Lettuce","Ginger","Eggplant","Cucumber","Cauliflower","Banana"]

        campaign_link = 'http://ios19bsh.ase.in.tum.de/dev/api/campaigns/' + campaignId + '/'
        t = Thread(target=start_train_thread, args=('train', 'coco',
                                                    campaignId, classes, imagesInfo, campaign_link, ))
        threads.append(t)
        print("---THREAD COUNT:" + str(len(threads)) + "---")
        t.start()

        return JsonResponse({'training': 1, 'thread_name': t.getName()})

def start_train_thread(cmd, base_model, campaignId, classes, imagesInfo, campaign_link):
    runtrain.train_main(cmd, base_model, campaignId,
                        classes, imagesInfo, campaign_link)
