FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

COPY . .


CMD python -u djangoserver/manage.py runserver 0.0.0.0:5000