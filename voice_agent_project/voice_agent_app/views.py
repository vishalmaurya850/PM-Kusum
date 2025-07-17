from django.http import HttpResponse
from django.core.management import call_command
from django.core.management.base import CommandError

def start_fastapi(request):
    try:
        call_command('runfastapi')
        return HttpResponse("FastAPI server started")
    except CommandError as e:
        return HttpResponse(f"Failed to start FastAPI server: {str(e)}", status=500)