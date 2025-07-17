import os
import json
import asyncio
import logging
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from django.core.management.base import BaseCommand
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Say
from deepgram import DeepgramClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from prisma import Prisma
from datetime import datetime
import base64
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import torch
from cachetools import TTLCache
from prometheus_client import Counter, Histogram, make_asgi_app
import time
from dotenv import load_dotenv
import PyPDF2
import re
from sklearn.preprocessing import MultiLabelBinarizer

load_dotenv()

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
api_latency = Histogram("api_latency_seconds", "API request latency", ["endpoint"])
call_success = Counter("call_success_total", "Total successful calls", ["outcome"])
websocket_duration = Histogram("websocket_duration_seconds", "WebSocket connection duration")
websocket_messages = Counter("websocket_messages_total", "Total WebSocket messages processed")
fine_tune_duration = Histogram("fine_tune_duration_seconds", "Fine-tuning duration")

# API credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
NGROK_URL = os.getenv("NGROK_URL")  # e.g., https://your-ngrok-domain.ngrok.io
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

# IndicBERT model and tokenizer
model_name = "./models/fine_tuned_indicbert"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
except Exception as e:
    logger.warning(f"Fine-tuned IndicBERT tokenizer not found: {str(e)}. Using pre-trained model.")
    try:
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert", use_fast=False)
    except Exception as e2:
        logger.error(f"Pre-trained IndicBERT tokenizer not found: {str(e2)}. Using default multilingual tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=False)

try:
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
except Exception as e:
    logger.warning(f"Fine-tuned IndicBERT model not found: {str(e)}. Using pre-trained model.")
    try:
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert", use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=3)
    except Exception as e2:
        logger.error(f"Pre-trained IndicBERT model not found: {str(e2)}. Using default multilingual model.")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)

# Prisma setup for NeonDB
prisma = Prisma()

# In-memory cache for ElevenLabs prompts (TTL: 1 hour)
prompt_cache = TTLCache(maxsize=100, ttl=3600)

# Hindi prompt
PROMPT = """
नमस्ते, यह PM-KUSUM योजना टीम से सोलरबॉट है। हम आप जैसे किसानों को सौर ऊर्जा अपनाने में मदद करने के लिए संपर्क कर रहे हैं। 
PM-KUSUM के साथ, आप कम लागत पर सौर पैनल स्थापित कर सकते हैं, बिजली बिल कम कर सकते हैं, और अतिरिक्त बिजली बेचकर कमाई कर सकते हैं। 
क्या आप आवेदन करने के बारे में और जानना चाहेंगे?
"""

# ElevenLabs voice ID and settings for Hindi
VOICE_ID = "gHu9GtaHOXcSqFTK06ux"  # Ravi
VOICE_SETTINGS = {
    "stability": 0.7,
    "similarity_boost": 0.8
}

# Pydantic models
class CallData(BaseModel):
    call_id: str
    phone_number: str

class AnalysisResult(BaseModel):
    call_id: str
    sentiment: str
    interest_level: bool
    intro_clarity: bool
    objections: list
    outcome: str

class CallResponse(BaseModel):
    call_id: str
    transcript: str
    analysis: AnalysisResult
    updates: dict

class FineTuneResponse(BaseModel):
    status: str
    message: str
    metrics: dict

# PDF text extraction
def extract_pdf_text(pdf_path: str) -> str:
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            # Clean text: remove extra spaces, newlines
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        return ""

# Create fine-tuning dataset
async def create_fine_tune_dataset(pdf_path: str) -> Dataset:
    try:
        # Extract PDF text
        pdf_text = extract_pdf_text(pdf_path)
        pdf_data = [
            {"text": sentence, "sentiment": "neutral"}
            for sentence in pdf_text.split('. ')
            if sentence.strip()
        ]

        # Fetch transcriptions from NeonDB
        await prisma.connect()
        calls = await prisma.callanalysis.find_many()
        await prisma.disconnect()

        transcript_data = [
            {
                "text": call.transcript,
                "sentiment": call.sentiment
            }
            for call in calls if call.transcript and call.transcript.strip() and call.sentiment != "pending"
        ]

        # Combine datasets
        combined_data = pdf_data + transcript_data

        if not combined_data:
            raise ValueError("No valid data found for fine-tuning.")

        # Create dataset for sentiment
        dataset = Dataset.from_dict({
            "text": [item["text"] for item in combined_data],
            "labels": [0 if item["sentiment"] == "negative" else 1 if item["sentiment"] == "neutral" else 2 for item in combined_data]
        })

        return dataset
    except Exception as e:
        logger.error(f"Error creating fine-tune dataset: {str(e)}")
        raise

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Fine-tune IndicBERT
async def fine_tune_model():
    try:
        start_time = time.time()
        dataset = await create_fine_tune_dataset(os.path.join(os.getcwd(), "data", "PM_KUSUM_Knowledge_Base.pdf"))
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Split dataset
        train_test = tokenized_dataset.train_test_split(test_size=0.2)
        train_dataset = train_test["train"]
        eval_dataset = train_test["test"]

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./models/fine_tuned_indicbert",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_dir="./logs",
            logging_steps=10,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,  # Updated from tokenizer
        )

        # Fine-tune
        trainer.train()
        trainer.save_model("./models/fine_tuned_indicbert")
        tokenizer.save_pretrained("./models/fine_tuned_indicbert")

        metrics = trainer.evaluate()
        fine_tune_duration.observe(time.time() - start_time)
        return metrics
    except Exception as e:
        logger.error(f"Error fine-tuning model: {str(e)}")
        raise

# Sentiment analysis with IndicBERT
def analyze_sentiment(text: str) -> str:
    try:
        start_time = time.time()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        sentiment_id = torch.argmax(logits, dim=1).item()
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map.get(sentiment_id, "neutral")
        api_latency.labels(endpoint="sentiment_analysis").observe(time.time() - start_time)
        return sentiment
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        return "neutral"

# ElevenLabs TTS with caching and fallback
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(aiohttp.ClientError))
async def generate_tts(text: str) -> bytes:
    cache_key = f"{text}_hi"
    if cache_key in prompt_cache:
        logger.info(f"Using cached audio for prompt: {text[:50]}...")
        return prompt_cache[cache_key]
    
    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            headers = {"xi-api-key": ELEVENLABS_API_KEY}
            payload = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": VOICE_SETTINGS
            }
            async with session.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"ElevenLabs API error {response.status}: {error_text}")
                    raise aiohttp.ClientError(f"ElevenLabs API error: {response.status} - {error_text}")
                audio_data = await response.read()
                prompt_cache[cache_key] = audio_data
                api_latency.labels(endpoint="elevenlabs_tts").observe(time.time() - start_time)
                return audio_data
    except aiohttp.ClientError as e:
        logger.error(f"ElevenLabs TTS error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected TTS error: {str(e)}")
        # Fallback to Twilio Polly TTS
        twiml = VoiceResponse()
        twiml.say(text, voice="Polly.Aditi")
        with open(os.path.join(os.getcwd(), "audio", f"fallback_hi_{hash(text)}.mp3"), "wb") as f:
            f.write(b"")  # Placeholder; Twilio handles audio
        return b""

# Twilio call initiation
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(Exception))
async def initiate_twilio_call(call_id: str, phone_number: str) -> str:
    AUDIO_DIR = os.path.join(os.getcwd(), "audio")
    os.makedirs(AUDIO_DIR, exist_ok=True)
    try:
        start_time = time.time()
        twiml = VoiceResponse()
        prompt_audio = await generate_tts(PROMPT)
        
        if not prompt_audio:
            twiml.say(PROMPT, voice="Polly.Aditi")
        else:
            with open(os.path.join(AUDIO_DIR, f"prompt_{call_id}.mp3"), "wb") as f:
                f.write(prompt_audio)
            twiml.play(url=f"{NGROK_URL}/api/audio/prompt_{call_id}.mp3")
        
        connect = Connect()
        connect.stream(url=f"wss://{NGROK_URL.replace('https://', '')}/ws/{call_id}")
        twiml.append(connect)
        
        call = twilio_client.calls.create(
            to=phone_number,
            from_=os.getenv("TWILIO_PHONE_NUMBER"),
            twiml=str(twiml)
        )
        logger.info(f"Initiated Twilio call {call.sid} for call_id {call_id}")
        api_latency.labels(endpoint="twilio_call").observe(time.time() - start_time)
        return call.sid
    except Exception as e:
        logger.error(f"Twilio call initiation error: {str(e)}")
        raise

# Call Analysis
async def analyze_call(transcript: str, call_id: str) -> AnalysisResult:
    try:
        start_time = time.time()
        sentiment = analyze_sentiment(transcript)
        interest_keywords = ["बताओ", "हाँ", "और", "रुचि"]
        confusion_keywords = ["समझा नहीं", "क्या बोल"]
        objection_keywords = {
            "cost": ["खर्चा", "कितना"],
            "eligibility": ["पात्र", "कौन"],
            "time": ["कब", "कितना समय"]
        }
        
        interest_level = any(keyword in transcript.lower() for keyword in interest_keywords)
        intro_clarity = not any(keyword in transcript.lower() for keyword in confusion_keywords)
        objections = [key for key, keywords in objection_keywords.items() if any(kw in transcript.lower() for kw in keywords)]
        
        if "दोबारा कॉल" in transcript.lower():
            outcome = "follow-up"
        elif interest_level:
            outcome = "success"
        else:
            outcome = "failure"
        
        call_success.labels(outcome=outcome).inc()
        api_latency.labels(endpoint="call_analysis").observe(time.time() - start_time)
        
        return AnalysisResult(
            call_id=call_id,
            sentiment=sentiment,
            interest_level=interest_level,
            intro_clarity=intro_clarity,
            objections=objections,
            outcome=outcome
        )
    except Exception as e:
        logger.error(f"Call analysis error for call {call_id}: {str(e)}")
        return AnalysisResult(
            call_id=call_id,
            sentiment="neutral",
            interest_level=False,
            intro_clarity=True,
            objections=[],
            outcome="failure"
        )

# Reinforcement Loop
async def reinforce_agent(analysis: AnalysisResult) -> dict:
    global PROMPT
    updates = {}
    
    try:
        start_time = time.time()
        if not analysis.intro_clarity:
            updates["intro"] = "सरल परिचय: हम किसानों को पैसे बचाने के लिए सरकारी सौर योजना के बारे में बता रहे हैं।"
            PROMPT = updates["intro"] + PROMPT[PROMPT.index("\n"):]
        
        if analysis.sentiment == "negative":
            updates["tone"] = "Using more polite tone."
            PROMPT = PROMPT.replace("संपर्क कर रहे हैं", "आपकी मदद करना चाहेंगे")
        
        if analysis.objections:
            updates["faq"] = f"Added FAQ for {', '.join(analysis.objections)}."
            PROMPT += f"\nFAQ: {updates['faq']}"
        
        if analysis.outcome == "follow-up":
            updates["follow_up"] = "Flagged for follow-up call."
            await prisma.followup.create(data={
                "call_id": analysis.call_id,
                "status": "pending",
                "created_at": datetime.now()
            })
        
        if not analysis.interest_level:
            updates["cta"] = "पुनर्लिखित CTA: 'क्या हम आपकी मदद के लिए और विवरण साझा कर सकते हैं?'"
            PROMPT = PROMPT.rsplit("क्या आप", 1)[0] + updates["cta"]
        
        api_latency.labels(endpoint="reinforce_agent").observe(time.time() - start_time)
        return updates
    except Exception as e:
        logger.error(f"Reinforcement error for call {analysis.call_id}: {str(e)}")
        return updates

# FastAPI Endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to the PM-KUSUM Voice Agent API. Use /api/start-call with POST to initiate a call."}

@app.post("/api/start-call", response_model=CallResponse)
async def start_call(call: CallData):
    try:
        start_time = time.time()
        await prisma.connect()
        twilio_call_sid = await initiate_twilio_call(call.call_id, call.phone_number)
        
        await prisma.callanalysis.create(
            data={
                "call_id": call.call_id,
                "sentiment": "pending",
                "interest_level": False,
                "intro_clarity": True,
                "objections": json.dumps([]),
                "outcome": "pending",
                "language": "hi",
                "transcript": "",
                "created_at": datetime.now()
            }
        )
        
        api_latency.labels(endpoint="start_call").observe(time.time() - start_time)
        return CallResponse(
            call_id=call.call_id,
            transcript="कॉल शुरू हुआ, बातचीत की प्रतीक्षा है।",
            analysis=AnalysisResult(
                call_id=call.call_id,
                sentiment="pending",
                interest_level=False,
                intro_clarity=True,
                objections=[],
                outcome="pending"
            ),
            updates={}
        )
    except Exception as e:
        logger.error(f"Error in start_call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await prisma.disconnect()

@app.get("/api/audio/{filename}")
async def serve_audio(filename: str):
    AUDIO_DIR = os.path.join(os.getcwd(), "audio")
    try:
        with open(os.path.join(AUDIO_DIR, filename), "rb") as f:
            return StreamingResponse(f, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"Audio serve error for {filename}: {str(e)}")
        raise HTTPException(status_code=404, detail="Audio file not found")

@app.post("/api/fine-tune", response_model=FineTuneResponse)
async def fine_tune_endpoint():
    try:
        metrics = await fine_tune_model()
        return FineTuneResponse(
            status="success",
            message="Model fine-tuning completed successfully.",
            metrics=metrics
        )
    except Exception as e:
        logger.error(f"Fine-tuning endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    await websocket.accept()
    start_time = time.time()
    transcript = ""
    message_count = 0
    
    try:
        dg_live = deepgram.listen.asynclive({
            "model": "general",
            "punctuate": True,
            "language": "hi"
        })
        
        async def process_audio(audio_data):
            nonlocal transcript, message_count
            if audio_data.get("media"):
                audio = base64.b64decode(audio_data["media"]["payload"])
                await dg_live.send(audio)
                message_count += 1
                websocket_messages.inc()
        
        async def receive_transcript():
            nonlocal transcript
            async for event in dg_live:
                if event.get("is_final"):
                    transcript += event.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "") + " "
        
        transcription_task = asyncio.create_task(receive_transcript())
        
        while True:
            message = await websocket.receive_json()
            if message.get("event") == "media":
                await process_audio(message)
            elif message.get("event") == "stop":
                break
        
        await dg_live.finish()
        transcription_task.cancel()
        
        analysis = await analyze_call(transcript, call_id)
        updates = await reinforce_agent(analysis)
        
        await prisma.connect()
        await prisma.callanalysis.update(
            where={"call_id": call_id},
            data={
                "sentiment": analysis.sentiment,
                "interest_level": analysis.interest_level,
                "intro_clarity": analysis.intro_clarity,
                "objections": json.dumps(analysis.objections),
                "outcome": analysis.outcome,
                "transcript": transcript.strip()
            }
        )
        await prisma.disconnect()
        
        await websocket.send_json({
            "event": "analysis",
            "transcript": transcript,
            "analysis": analysis.dict(),
            "updates": updates
        })
        
        websocket_duration.observe(time.time() - start_time)
    
    except Exception as e:
        logger.error(f"WebSocket error for call {call_id}: {str(e)}")
    finally:
        await websocket.close()

# Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())

# Django Command to Run FastAPI
class Command(BaseCommand):
    help = "Run FastAPI server for voice agent system"

    def add_arguments(self, parser):
        parser.add_argument('--port', type=int, default=8001, help='Port to run FastAPI server on')

    def handle(self, *args, **options):
        port = options['port']
        logger.info(f"Starting FastAPI server on port {port}...")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)