# -*- coding: utf-8 -*-
from django.urls import path

from .views import Views

urlpatterns = [
    path('', Views.run, name='run_exe'),
]
