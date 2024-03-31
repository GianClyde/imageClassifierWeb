# urls.py
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("classify/<path:image_path>/", views.classify, name="classify"),  # Updated URL pattern
    path("results/", views.results, name="results"),  # Updated URL pattern
    path("CNN/<path:image_path>/", views.bridge, name="bridge"),  # Updated URL pattern
]
