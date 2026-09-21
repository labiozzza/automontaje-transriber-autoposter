import argparse
import shutil
import json
import os
import sys
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

CURL = shutil.which("curl") or "curl"




BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLIENT_SECRET_FILE = os.path.join(BASE_DIR, "client_secret.json")
TOKEN_FILE = os.path.join(BASE_DIR, "token.json")

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload"
]


def get_credentials():
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES
        )

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if not os.path.exists(CLIENT_SECRET_FILE):
            raise FileNotFoundError(
                f"Не найден файл OAuth:\n{CLIENT_SECRET_FILE}"
            )

        if os.path.exists(TOKEN_FILE):
            os.unlink(TOKEN_FILE)

        creds = run_local_oauth()

    return creds


def run_local_oauth():
    if not os.path.exists(CLIENT_SECRET_FILE):
        raise FileNotFoundError(
            f"Не найден файл OAuth:\n{CLIENT_SECRET_FILE}"
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        SCOPES,
        redirect_uri="http://localhost"
    )

    creds = flow.run_local_server(
        port=0,
        access_type="offline",
        prompt="consent",
        authorization_prompt_message="",
        success_message="Авторизация YouTube завершена! Это окно можно закрыть."
    )

    with open(TOKEN_FILE, "w", encoding="utf-8") as token:
        token.write(creds.to_json())

    return creds


def get_youtube():
    creds = get_credentials()

    return build(
        "youtube",
        "v3",
        credentials=creds,
        cache_discovery=False
    )


def upload_video(
    video_path: str,
    title: str,
    description: str,
    privacy: str,
    category_id: str,
    tags: list[str],
    made_for_kids: bool
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Видео не найдено:\n{video_path}"
        )

    youtube = get_youtube()

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": category_id,
            "tags": tags,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": made_for_kids,
        },
    }

    media = MediaFileUpload(
        video_path,
        chunksize=8 * 1024 * 1024,
        resumable=True
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
        notifySubscribers=False
    )

    print()
    print("Загрузка началась...")
    print()

    response = None

    while response is None:
        try:
            status, response = request.next_chunk()

            if status:
                progress = int(status.progress() * 100)

                print(
                    f"\rЗагружено: {progress}%",
                    end="",
                    flush=True
                )

        except HttpError as e:
            print()
            print("YouTube API error:")
            print(e)
            raise

    print()
    print()

    video_id = response["id"]

    print("SUCCESS")
    print(f"Video ID: {video_id}")
    print(f"Watch URL: https://www.youtube.com/watch?v={video_id}")
    print(f"Shorts URL: https://www.youtube.com/shorts/{video_id}")

    return response


def main():
    parser = argparse.ArgumentParser(
        description="Upload YouTube Short"
    )

    parser.add_argument(
        "video",
        help="Путь до видео"
    )

    parser.add_argument(
        "--title",
        required=True,
        help="Название видео"
    )

    parser.add_argument(
        "--description",
        default="",
        help="Описание"
    )

    parser.add_argument(
        "--description-file",
        help="TXT-файл с описанием"
    )

    parser.add_argument(
        "--privacy",
        choices=["public", "private", "unlisted"],
        default="public"
    )

    parser.add_argument(
        "--category",
        default="22",
        help="YouTube categoryId. По умолчанию 22 = People & Blogs"
    )

    parser.add_argument(
        "--tags",
        default="",
        help="Теги через запятую"
    )

    parser.add_argument(
        "--made-for-kids",
        action="store_true"
    )

    args = parser.parse_args()

    description = args.description

    if args.description_file:
        if not os.path.exists(args.description_file):
            print(
                f"Не найден description-file: "
                f"{args.description_file}"
            )
            sys.exit(1)

        with open(
            args.description_file,
            "r",
            encoding="utf-8"
        ) as f:
            description = f.read()

    tags = [
        x.strip()
        for x in args.tags.split(",")
        if x.strip()
    ]

    try:
        upload_video(
            video_path=args.video,
            title=args.title,
            description=description,
            privacy=args.privacy,
            category_id=args.category,
            tags=tags,
            made_for_kids=args.made_for_kids
        )

    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
        sys.exit(1)

    except Exception as e:
        print()
        print("ERROR:")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()