# from django.conf.urls import url, include
# from . import views

from django.conf.urls import url
from islanding import views

# from django.urls import path

app_name = 'islanding'

urlpatterns = [
    # url(r'^islanding_result$', views.islanding_result),
    url('islanding_result/', views.islanding_result, name='islanding_result'),
    url('islanding_plot/', views.islanding_plot, name='islanding_plot')
]
 