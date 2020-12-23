from django.shortcuts import render, redirect, reverse
from django.http import HttpResponse
from django.template import loader

from django.contrib.auth.decorators import login_required


@login_required
def index(request):

    context = {
    'app_name': "dashboard",
    }
    template = loader.get_template('dashboard/index.html')
    return HttpResponse(template.render(context, request))

@login_required
def compare(request):

    context = {
    'app_name': "dashboard",
    'page_name': "compare",
    }
    template = loader.get_template('dashboard/compare.html')
    return HttpResponse(template.render(context, request))
    