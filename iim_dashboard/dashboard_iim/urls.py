from django.conf.urls import url, include
from . import views
from django.views.generic import RedirectView

# from django.urls import path

app_name = 'dashboard_iim'

urlpatterns = [
    # url('^$', views.index, name='index'),
    url('/home', views.index, name='index'),
    url('/compare', views.compare, name='compare'),
    url(r'^favicon\.ico$',RedirectView.as_view(url='/static/assets/favicon.ico')),
]
 