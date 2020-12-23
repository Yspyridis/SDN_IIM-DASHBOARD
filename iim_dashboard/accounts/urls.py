# from django.conf.urls import url, include
from . import views
# from django.views.generic import RedirectView

from django.urls import path
from django.contrib.auth.views import LoginView, LogoutView

# app_name = 'accounts'

urlpatterns = [
    # url('login', views.login, name='login')
    # path('', views.indexView, name = "home"),
    # path('', dashboard.dashboardView, name = "dashboard"),
    path('login/', LoginView.as_view(), name='login_url'),
    path('logout/', LogoutView.as_view(next_page='login_url'), name='logout'),
]
 