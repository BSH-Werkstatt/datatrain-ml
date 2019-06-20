from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'^$', views.predict, name='predict'),
	re_path(r'^', views.predictReturnFile, name='predict')
]