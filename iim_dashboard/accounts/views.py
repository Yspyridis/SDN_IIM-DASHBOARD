from django.shortcuts import render, redirect, reverse
from django.http import HttpResponse
from django.template import loader

from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

# def login(request):




# def login(request):

    # if request.method == 'POST':
    #     form = AuthenticationForm(data = request.POST)
    #     if form.is_valid():
    #         return redirect('/dashboard')
        # username = request.POST['username']
        # password = request.POST['password']
        
        # user = auth.authenticate(username=username, password=password)
        # if user is not None:
        #     auth.login(request, user)
        #     return redirect('/')
    # else:
    #     form = AuthenticationForm()
    # return render(request, 'accounts/login.html', {'form':form})
    # return render(request, 'dashboard/compare.html', {'form':form})