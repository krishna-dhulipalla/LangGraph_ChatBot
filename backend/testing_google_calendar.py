#!/usr/bin/env python3
"""
Create a Google Calendar event (optionally with a Google Meet link).

Usage examples:
  python test_gcal_create_event.py \
    --title "Intro call" \
    --start "2025-08-14 10:00" \
    --duration 30 \
    --tz "America/Chicago" \
    --attendees "recruiter@example.com,hm@example.com" \
    --meet

  python test_gcal_create_event.py \
    --title "Pairing session" \
    --start-rfc3339 "2025-08-14T10:00:00-05:00" \
    --end-rfc3339   "2025-08-14T10:45:00-05:00" \
    --calendar "primary" \
    --description "Quick screen"
"""

import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Py3.9+
from typing import List, Optional

# --- Google libs ---
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# -------- Config --------
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/calendar.events.owned"]
CREDS_DIR = Path("credentials")  # change if you want
CREDS_DIR.mkdir(parents=True, exist_ok=True)
CLIENT_SECRET_FILE = CREDS_DIR / "credentials.json"  # download from Google Cloud
TOKEN_FILE = CREDS_DIR / "token.json"


def rfc3339(dt: datetime) -> str:
    """Return RFC3339 string with timezone offset, e.g. 2025-08-14T10:00:00-05:00"""
    # Ensure aware datetime
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("datetime must be timezone-aware")
    # isoformat() produces RFC3339-compatible output for aware datetimes
    return dt.isoformat(timespec="seconds")


def get_service():
    if not CLIENT_SECRET_FILE.exists():
        raise FileNotFoundError(
            f"Missing OAuth client file: {CLIENT_SECRET_FILE}\n"
            "Create an OAuth 2.0 Client ID (Desktop) and download JSON."
        )

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), GOOGLE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CLIENT_SECRET_FILE), GOOGLE_SCOPES
            )
            # Opens a local browser once to authorize your Google account
            creds = flow.run_local_server(port=8765, prompt="consent")
        TOKEN_FILE.write_text(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def parse_args():
    p = argparse.ArgumentParser(description="Create a Google Calendar event.")
    # Option A: local date/time + duration
    p.add_argument("--start", help="Local start time: 'YYYY-MM-DD HH:MM'", type=str)
    p.add_argument("--duration", help="Duration in minutes", type=int)
    p.add_argument("--tz", help="IANA timezone, e.g. America/Chicago", default="America/Chicago")

    # Option B: direct RFC3339
    p.add_argument("--start-rfc3339", help="RFC3339 start, e.g. 2025-08-14T10:00:00-05:00")
    p.add_argument("--end-rfc3339", help="RFC3339 end, e.g. 2025-08-14T10:30:00-05:00")

    p.add_argument("--title", required=True, help="Event title")
    p.add_argument("--description", default="", help="Event description")
    p.add_argument("--location", default="", help="Event location")
    p.add_argument("--attendees", default="", help="Comma-separated emails")
    p.add_argument("--calendar", default="primary", help="Calendar ID (default 'primary')")
    p.add_argument("--meet", action="store_true", help="Create a Google Meet link")
    return p.parse_args()


def main():
    args = parse_args()

    # Compute start/end
    if args.start_rfc3339 and args.end_rfc3339:
        start_rfc3339 = args.start_rfc3339
        end_rfc3339 = args.end_rfc3339
    else:
        if not args.start:
            raise SystemExit("Provide --start 'YYYY-MM-DD HH:MM' (and --duration) or use --start-rfc3339/--end-rfc3339.")
        if not args.duration and not args.end_rfc3339:
            raise SystemExit("Provide --duration (minutes) when using --start.")
        tz = ZoneInfo(args.tz)
        start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M").replace(tzinfo=tz)
        if args.duration:
            end_dt = start_dt + timedelta(minutes=args.duration)
        else:
            # rare branch if user supplied --end-rfc3339 with --start
            raise SystemExit("If you use --start, prefer --duration. Otherwise use both --start-rfc3339 and --end-rfc3339.")
        start_rfc3339 = rfc3339(start_dt)
        end_rfc3339 = rfc3339(end_dt)

    attendees_list: List[dict] = []
    if args.attendees.strip():
        for email in [e.strip() for e in args.attendees.split(",") if e.strip()]:
            attendees_list.append({"email": email})

    svc = get_service()

    body = {
        "summary": args.title,
        "description": args.description,
        "location": args.location,
        "start": {"dateTime": start_rfc3339},
        "end": {"dateTime": end_rfc3339},
        "attendees": attendees_list,
    }

    params = {}
    if args.meet:
        body["conferenceData"] = {"createRequest": {"requestId": str(uuid4())}}
        params["conferenceDataVersion"] = 1  # required to create Meet link

    event = svc.events().insert(calendarId=args.calendar, body=body, **params).execute()

    meet_url = None
    if args.meet:
        conf = event.get("conferenceData") or {}
        entry_points = conf.get("entryPoints") or []
        # find the video entry point if available
        for ep in entry_points:
            if ep.get("entryPointType") == "video":
                meet_url = ep.get("uri")
                break

    print("✅ Event created")
    print("  Title   :", event.get("summary"))
    print("  When    :", event["start"]["dateTime"], "→", event["end"]["dateTime"])
    print("  With    :", ", ".join([a["email"] for a in attendees_list]) or "—")
    print("  Meet    :", meet_url or "—")
    print("  Calendar:", args.calendar)
    print("  EventID :", event.get("id"))


if __name__ == "__main__":
    main()