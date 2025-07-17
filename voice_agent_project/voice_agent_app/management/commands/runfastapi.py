import uvicorn
from django.core.management.base import BaseCommand
from voice_agent_app.voice_agent_system import app

class Command(BaseCommand):
    help = "Run FastAPI server for voice agent system"

    def add_arguments(self, parser):
        parser.add_argument('--port', type=int, default=8001, help='Port to run FastAPI server on')

    def handle(self, *args, **options):
        port = options['port']
        self.stdout.write(self.style.SUCCESS(f"Starting FastAPI server on port {port}..."))
        uvicorn.run(app, host="0.0.0.0", port=port)