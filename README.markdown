# PM-KUSUM Voice Agent

The PM-KUSUM Voice Agent is a conversational AI system designed to assist farmers in understanding and applying for the PM-KUSUM scheme, a government initiative to promote solar energy adoption in rural India. The application supports Hindi-only interactions, leveraging real-time speech transcription (Deepgram), text-to-speech (ElevenLabs with Twilio fallback), sentiment analysis (IndicBERT), and a reinforcement loop to improve conversation prompts based on call outcomes. The backend is built with FastAPI and Django, integrated with NeonDB for data storage, while the frontend is a Next.js application providing a dashboard for call analysis and fine-tuning.

## Features
- **Voice Interaction**: Initiates calls via Twilio, delivers Hindi prompts, and transcribes responses using Deepgram.
- **Sentiment Analysis**: Uses a fine-tuned IndicBERT model to classify call sentiments (positive, neutral, negative).
- **Objection Detection**: Identifies objections (e.g., cost, eligibility, time) based on keyword matching.
- **Reinforcement Loop**: Dynamically updates the conversation prompt based on call analysis (e.g., improving clarity or tone).
- **Data Storage**: Stores call transcripts and analysis in NeonDB using Prisma ORM.
- **Frontend Dashboard**: Built with Next.js, displays call metrics and allows triggering model fine-tuning.
- **Monitoring**: Prometheus metrics for API latency, call success, and fine-tuning duration.

## Project Structure
```
voice_agent_project/
├── data/
│   └── PM_KUSUM_Knowledge_Base.pdf  # Knowledge base for fine-tuning
├── models/
│   └── fine_tuned_indicbert/        # Fine-tuned IndicBERT model
├── audio/                           # Generated audio files
├── logs/                            # Training logs
├── voice_agent_app/
│   ├── __init__.py
│   ├── voice_agent_system.py        # FastAPI backend logic
│   └── manage.py                    # Django management script
├── prisma/
│   └── schema.prisma                # Prisma schema for NeonDB
├── .env                             # Environment variables
└── requirements.txt                 # Backend dependencies
```

## Prerequisites
- **Python**: 3.8+
- **Node.js**: 16+ (for Next.js frontend)
- **NeonDB**: PostgreSQL database instance
- **Ngrok**: For exposing the local server to Twilio and Deepgram
- **Accounts and APIs**:
  - Twilio (for calls)
  - Deepgram (for transcription)
  - ElevenLabs (for text-to-speech)
- **System Requirements**:
  - Windows/Linux/Mac
  - Minimum 8GB RAM (16GB recommended for fine-tuning)
  - Optional: GPU for faster model training

## Dependencies
### Backend
Add the following to `requirements.txt`:
```
django
fastapi
uvicorn
prisma
deepgram-sdk
twilio
python-dotenv
transformers
torch
bentoml
tenacity
aiohttp
cachetools
prometheus-client
psycopg2
tiktoken 
protobuf
setuptools
requests
PyPDF2
datasets
scikit-learn
huggingface_hub[hf_xet]
transformers[torch]
```

## Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/vishalmaurya850/PM-Kusum
cd voice_agent_project
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+your_twilio_phone_number
DEEPGRAM_API_KEY=your_deepgram_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
NGROK_URL=https://your-ngrok-domain.ngrok.io
DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname>?sslmode=require
```

For the frontend, create `voice-agent-frontend/.env`:
```
DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/<dbname>?sslmode=require
NEXT_PUBLIC_API_URL=http://127.0.0.1:8001  # Update to NGROK_URL if using Ngrok
```

### 3. Install Backend Dependencies
```bash
python -m venv venv
venv/Scripts/activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies
```bash
cd voice-agent-frontend
npm install
```

### 5. Set Up NeonDB
1. Create a NeonDB PostgreSQL instance and note the connection string.
2. Update `prisma/schema.prisma`:
```prisma
generator client {
  provider = "prisma-client-py"
  interface = "asyncio"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model CallAnalysis {
  id            Int      @id @default(autoincrement())
  call_id       String   @unique
  sentiment     String
  interest_level Boolean
  intro_clarity Boolean
  objections    Json
  outcome       String
  language      String
  transcript    String?  @default("")
  created_at    DateTime @default(now())
}

model Followup {
  id         Int      @id @default(autoincrement())
  call_id    String   @unique
  status     String
  created_at DateTime @default(now())
}
```
3. Generate Prisma client:
```bash
cd voice_agent_project
venv/Scripts/activate
prisma generate
```

### 6. Prepare Data
- Place `PM_KUSUM_Knowledge_Base.pdf` in `voice_agent_project/data/`.
- Ensure the PDF contains PM-KUSUM details (e.g., costs, eligibility) in readable text format.
- Generate initial call data for fine-tuning:
  - Run the application (see below) and initiate test calls.
  - Respond verbally in Hindi (e.g., “हाँ” for success, “दोबारा कॉल” for follow-up, “नहीं” for rejection).
  - Aim for 50+ calls to create a robust dataset.

## Running the Application
### 1. Run the Backend
```bash
cd voice_agent_project
.\venv\Scripts\activate
python manage.py runserver
```

### 2. Run the Backend
```bash
cd voice_agent_project
.\venv\Scripts\activate
python manage.py runfastapi --port 8001
```

### 3. Start Ngrok
Expose the local server for Twilio and Deepgram:
```bash
ngrok http 8001
```

- Copy the Ngrok URL (e.g., `https://your-ngrok-domain.ngrok.io`) and update `NGROK_URL` in `.env` and `NEXT_PUBLIC_API_URL` in `voice-agent-frontend/.env`.
### 4. Run the Frontend
```bash
cd voice-agent-frontend
npm run dev
```
- Access the frontend at `http://localhost:3000`.

## Usage
### Initiating a Call
1. Open `http://localhost:3000`.
2. Enter:
   - **Phone Number**: Recipient’s number (e.g., `+1234567890`).
   - **Twilio Number**: Your Twilio number (e.g., `+0987654321`).
   - **Call ID**: Unique identifier (e.g., `test_call_1`).
3. Click “Start Call” to initiate a Twilio call with the Hindi prompt.
4. Respond verbally in Hindi. The system transcribes responses, analyzes sentiment/objections, and updates the database.

### Fine-Tuning the Model
1. Navigate to `http://localhost:3000/dashboard`.
2. Click “मॉडल को फाइन-ट्यून करें” (Fine-Tune Model).
3. The backend:
   - Extracts text from `PM_KUSUM_Knowledge_Base.pdf` and transcripts from `CallAnalysis`.
   - Fine-tunes IndicBERT for sentiment classification (positive, neutral, negative).
   - Saves the model to `voice_agent_project/models/fine_tuned_indicbert`.
4. Monitor progress in FastAPI logs:
   ```
   INFO:__main__:Starting fine-tuning...
   ```
5. View metrics (e.g., `eval_loss`) on the dashboard.

### Viewing Call Analysis
- On `http://localhost:3000/dashboard`, view call transcripts, sentiments, objections (cost, eligibility, time), and outcomes (success, failure, follow-up).
- Use Prisma Studio to inspect the database:
  ```bash
  prisma studio
  ```
  Check the `transcript` column in `CallAnalysis`.

### Monitoring Metrics
- Access Prometheus metrics at `http://127.0.0.1:8001/metrics` or `NGROK_URL/metrics`.
- Key metrics:
  - `api_latency_seconds`: API request latency.
  - `call_success_total`: Call outcomes.
  - `fine_tune_duration_seconds`: Fine-tuning duration.

## Testing
1. **Generate Test Calls**:
   - Initiate calls with varied responses:
     - Success: “हाँ, और बताओ”
     - Follow-up: “दोबारा कॉल करें”
     - Rejection: “नहीं, मुझे रुचि नहीं है”
     - Objections: “खर्चा कितना है?” (cost), “पात्रता क्या है?” (eligibility), “कितना समय लगेगा?” (time)
   - Verify analysis on the dashboard.

2. **Verify Fine-Tuning**:
   - Trigger fine-tuning and check logs for completion.
   - Test sentiment analysis on new calls to ensure improved accuracy.
   - Sample dataset check:
     ```bash
     psql "<your-database-url>" -c "SELECT call_id, transcript, sentiment FROM CallAnalysis WHERE transcript != '' LIMIT 5;"
     ```

3. **Check PDF Extraction**:
   ```python
   from PyPDF2 import PdfReader
   reader = PdfReader("data/PM_KUSUM_Knowledge_Base.pdf")
   print(reader.pages[0].extract_text())
   ```

## Troubleshooting
1. **Fine-Tuning Errors**:
   - **Loss Error**: If “model did not return a loss” persists, verify dataset:
     ```python
     from datasets import Dataset
     dataset = Dataset.from_dict({"text": ["test"], "labels": [1]})
     tokenized = dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512))
     print(tokenized[0])  # Ensure 'labels' is present
     ```
   - **Empty Dataset**: Ensure `PM_KUSUM_Knowledge_Base.pdf` is readable and `CallAnalysis` has 50+ transcripts.

2. **Pin Memory Warning**:
   - If PyTorch warns about `pin_memory`, confirm CPU training:
     ```bash
     python -c "import torch; print(torch.cuda.is_available())"
     ```
   - The warning is harmless and should be minimal with the updated code.

3. **Database Issues**:
   - Verify NeonDB connection:
     ```bash
     psql "<your-database-url>" -c "SELECT 1;"
     ```
   - Ensure `schema.prisma` matches the database schema.

4. **Frontend Issues**:
   - Check `NEXT_PUBLIC_API_URL` matches `NGROK_URL` or `http://127.0.0.1:8001`.
   - Inspect browser console for CORS or 500 errors.

5. **Call Failures**:
   - Verify Twilio, Deepgram, and ElevenLabs API keys.
   - Check Ngrok connectivity and FastAPI logs for errors.

## Future Enhancements
- **Multi-Label Objection Classification**: Add a separate model or custom loss function for detecting objections (cost, eligibility, time).
- **Fine-Tuning Progress UI**: Show real-time training progress on the dashboard.
- **GPU Support**: Optimize for GPU if available to speed up fine-tuning.
- **Multilingual Support**: Extend beyond Hindi using `bert-base-multilingual-cased`.

## Contributing
- Submit pull requests to `https://github.com/vishalmaurya850/PM-Kusum`.
- Report issues via GitHub Issues, including logs and error details.

## License
MIT License. See `LICENSE` for details.

## Contact
For support, contact `vishalmaurya850@gmail.com` or raise an issue on GitHub.
