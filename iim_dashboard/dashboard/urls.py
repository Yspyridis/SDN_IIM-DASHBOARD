from django.conf.urls import url, include
from . import views
from django.views.generic import RedirectView

# from django.urls import path

app_name = 'dashboard'

urlpatterns = [
    url('^$', views.index, name='index'),
    url(r'^favicon\.ico$',RedirectView.as_view(url='/static/assets/favicon.ico')),

    url('/compare', views.compare, name='compare'),
    # url('/login', views.login, name='login'),
    # url(r'^login/$', views.login, name='login')

]
 