# backend/mcp_servers/alert_mcp.py
from mcp.server.fastmcp import FastMCP
from datetime import datetime
import os
import json

mcp = FastMCP("KinnectAlertServer")

@mcp.tool()
def format_alert_email(user_id: str, cognitive_score: float, anomalies: list, summary: str, transcript_excerpt: str) -> str:
    """
    Formats a professional, elderly-care focused alert email.
    
    Returns:
        A formatted email body string.
    """
    subject = f"⚠️ Alert: Cognitive Health Concern Detected for Patient {user_id}"
    
    anomalies_str = "\n".join([f"- {a}" for a in anomalies]) if anomalies else "None"
    
    email_body = f"""
Subject: {subject}
Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
Recipient: Family Caregiver / Medical Contact

Dear Caregiver,

This is an automated cognitive health update from Kinnect AI regarding patient: {user_id}.

During today's scheduled check-in call, our cognitive screening models identified patterns that may warrant attention.

SUMMARY assessment:
----------------------------------------
{summary}

KEY METRICS:
----------------------------------------
- Cognitive Score: {cognitive_score}/100
- Concerns Identified:
{anomalies_str}

TRANSCRIPT EXCERPT (Concerning segment):
----------------------------------------
{transcript_excerpt}

RECOMMENDED ACTION:
----------------------------------------
We recommend calling the patient to verify their wellbeing, checking if they have taken their prescribed medication, or scheduling a visit.

If you have questions or would like to view full conversation history, please log in to the Kinnect Caregiver Portal.

Warm regards,
Kinnect AI Care Team
"""
    return email_body

@mcp.tool()
def send_email_alert(to_email: str, subject: str, body: str) -> str:
    """
    Sends an email alert to the caregiver.
    STUB ONLY: Saves the alert details to a file inside the alerts/ folder.
    """
    try:
        os.makedirs("alerts", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alerts/sent_email_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"To: {to_email}\n")
            f.write(f"Subject: {subject}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n")
            f.write(body)
            
        print(f"📧 [STUB] Alert email saved to file: {filename}")
        return f"Successfully queued email (saved to file: {filename})"
    except Exception as e:
        return f"Error queuing email to file: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport='stdio')
