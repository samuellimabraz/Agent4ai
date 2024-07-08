import datetime
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class GoogleCalendarTool:
    SCOPES = [
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/userinfo.email",
        "openid",
    ]
    CREDENTIALS_FILE = "credentials.json"
    TOKEN_FILE = "token.json"

    def __init__(self):
        self.creds = self.authenticate()
        self.service = build("calendar", "v3", credentials=self.creds)
        self.user_info_service = build("oauth2", "v2", credentials=self.creds)

    def authenticate(self):
        """
        Authenticate with google account and obtain token.json
        """
        creds = None
        if os.path.exists(self.TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(self.TOKEN_FILE, self.SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.CREDENTIALS_FILE, self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(self.TOKEN_FILE, "w") as token:
                token.write(creds.to_json())
        return creds

    def get_user_info(self) -> dict:
        """
        Get the use info in a dict:
        User ID: {
          'id': 'xxx',
          'email': 'xxxx',
          'verified_email': bool,
          'picture': url
          }
        """
        return self.user_info_service.userinfo().v2().me().get().execute()

    def list_events(self, max_results=10) -> list:
        try:
            now = datetime.datetime.now().isoformat() + "Z"  # 'Z' indicates UTC time
            print("Getting the upcoming {} events".format(max_results))
            events_result = (
                self.service.events()
                .list(
                    calendarId="primary",
                    timeMin=now,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            events = events_result.get("items", [])
            cleaned_events = []
            for event in events:
                cleaned_event = {
                    "htmlLink": event.get("htmlLink"),
                    "creator": event.get("creator", {}).get("email"),
                    "summary": event.get("summary", {}),
                    "start": event.get("start", {}).get("dateTime"),
                    "end": event.get("end", {}).get("dateTime"),
                    "description": event.get("description"),
                }
                cleaned_events.append(cleaned_event)

            return {"events": cleaned_events, "total": len(cleaned_events)}

        except HttpError as error:
            print("An error occurred: %s" % error)
            return []

    def create_event(self, summary, start_time, end_time, location="", description=""):
        event = {
            "summary": summary,
            "location": location,
            "description": description,
            "start": {
                "dateTime": start_time,
                "timeZone": "America/Belem",
            },
            "end": {
                "dateTime": end_time,
                "timeZone": "America/Belem",
            },
        }
        try:
            event = (
                self.service.events().insert(calendarId="primary", body=event).execute()
            )
            print("Event created: %s" % (event.get("htmlLink")))
            return event
        except HttpError as error:
            print("An error occurred: %s" % error)
            return None


if __name__ == "__main__":
    calendar_tool = GoogleCalendarTool()

    l = calendar_tool.list_events()
    print(l)

    # e = calendar_tool.create_event(
    #     summary="Meeting",
    #     description="Discuss project",
    #     start_time="2024-07-07T21:00:00-03:00",
    #     end_time="2024-07-07T21:30:00-03:00",
    # )
    # print(e)
    # print(type(e))
    # # Obter informações do usuário
    user_info = calendar_tool.get_user_info()

    # # Imprimir informações do usuário
    print(f"User ID: {user_info}")
