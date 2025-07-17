import requests
import os
import argparse
import logging
from dotenv import load_dotenv
from twilio.rest import Client

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initiate_test_call(test_number: str, twilio_number: str, call_id: str):
    """
    Send a POST request to initiate a Twilio call and test the voice agent service.
    
    Args:
        test_number (str): The phone number to receive the call (E.164 format, e.g., +1234567890).
        twilio_number (str): The Twilio phone number to initiate the call (E.164 format).
        call_id (str): Unique identifier for the call.
    """
    # Ensure phone numbers are in E.164 format
    if not test_number.startswith('+'):
        test_number = f"+{test_number}"
    if not twilio_number.startswith('+'):
        twilio_number = f"+{twilio_number}"

    # API endpoint
    url = "http://127.0.0.1:8001/api/start-call"
    # If using Ngrok, update to NGROK_URL
    ngrok_url = os.getenv("NGROK_URL")
    if ngrok_url:
        url = f"{ngrok_url}api/start-call"

    # Payload for POST request
    payload = {
        "call_id": call_id,
        "phone_number": test_number
    }
    headers = {"Content-Type": "application/json"}

    try:
        # Send POST request to start-call endpoint
        logger.info(f"Sending POST request to {url} with call_id: {call_id}")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for non-200 status codes
        
        logger.info(f"API Response: {response.status_code} - {response.json()}")
        
        # Verify Twilio call initiation
        twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        calls = twilio_client.calls.list(to=test_number, limit=1)
        if calls:
            logger.info(f"Twilio call initiated: SID={calls[0].sid}, Status={calls[0].status}")
        else:
            logger.warning("No recent Twilio calls found for the test number.")
        
        return response.json()
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending POST request: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Test the PM-KUSUM Voice Agent by initiating a call.")
    parser.add_argument("--test-number", required=True, help="Phone number to receive the call (E.164 format, e.g., +1234567890)")
    parser.add_argument("--twilio-number", required=True, help="Twilio phone number to initiate the call (E.164 format)")
    parser.add_argument("--call-id", default="test_call_1", help="Unique call identifier (default: test_call_1)")
    
    args = parser.parse_args()
    
    try:
        # Initiate the test call
        result = initiate_test_call(args.test_number, args.twilio_number, args.call_id)
        logger.info(f"Test call initiated successfully: {result}")
        
        # Instructions for testing
        print("\nNext Steps to Test the Service:")
        print("1. Answer the call on your test number.")
        print("2. Listen to the Hindi prompt from SolarBot about PM-KUSUM.")
        print("3. Respond verbally (e.g., say 'हाँ' or 'बताओ' to show interest, or 'दोबारा कॉल' for follow-up).")
        print("4. Check FastAPI logs for transcription and analysis.")
        print(f"5. Monitor Prometheus metrics at http://127.0.0.1:8001/metrics or {os.getenv('NGROK_URL')}/metrics.")
        print("6. Verify call data in NeonDB using Prisma (e.g., `prisma studio`).")
        
    except Exception as e:
        logger.error(f"Test call failed: {str(e)}")
        print(f"Error: {str(e)}")
        print("Check FastAPI logs, Twilio dashboard, and ensure .env credentials are correct.")

if __name__ == "__main__":
    main()