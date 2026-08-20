#dedicated to sending alerts when/if trading bot fails and for when the bot runs its trading process

import requests

def send_alert(subject, body, env_path=None):
    #send message of activity to discord server that only i am in

    import os 
    from dotenv import load_dotenv
    if env_path:
        load_dotenv(env_path)

    webhook_url = os.getenv('DISCORD_WEBHOOK')

    if not webhook_url:
        print("alerting not configured, skipping...")

    try:
        response = requests.post(webhook_url, json={"content": f"**{subject}**\n{body}"})
        response.raise_for_status()
        print(f"Alert sent: {subject}")
    except Exception as e:
        print(f"Alert failed: {e}")

