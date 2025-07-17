from django.contrib import admin
from django.urls import path
from voice_agent_app.views import start_fastapi

urlpatterns = [
    path('api/start-fastapi/', start_fastapi, name='start_fastapi'),
]