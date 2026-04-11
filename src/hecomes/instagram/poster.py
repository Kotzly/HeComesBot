"""Instagram posting via the Meta Graph API and Cloudinary for file hosting.

Usage::

    from hecomes.instagram.poster import InstagramPoster, load_credentials

    creds = load_credentials()          # reads ~/.hecomes_instagram.json or env vars
    poster = InstagramPoster(**creds)
    poster.post_image("output.png", caption="Generated art")
    poster.post_reel("videos/video-1.mp4", caption="Generated animation")
"""

import json
import os
import time
from pathlib import Path

import requests

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".ogg", ".mpeg", ".flv"}

_CREDS_FILE = Path.home() / ".hecomes_instagram.json"

_ENV_KEYS = {
    "ig_user_id":         "HECOMES_IG_USER_ID",
    "access_token":       "HECOMES_IG_ACCESS_TOKEN",
    "cloudinary_cloud":   "HECOMES_CLOUDINARY_CLOUD",
    "cloudinary_key":     "HECOMES_CLOUDINARY_KEY",
    "cloudinary_secret":  "HECOMES_CLOUDINARY_SECRET",
}


def load_credentials(credentials_path=None) -> dict:
    """Load Instagram + Cloudinary credentials.

    Load order:
    1. Environment variables (``HECOMES_IG_USER_ID``, etc.)
    2. JSON credentials file (``~/.hecomes_instagram.json`` or *credentials_path*)

    Missing keys from the file are filled from env vars.
    Raises ``RuntimeError`` if any required key is still missing.
    """
    creds = {}

    # File first (env vars can override individual keys below)
    path = Path(credentials_path) if credentials_path else _CREDS_FILE
    if path.exists():
        with open(path) as f:
            creds.update(json.load(f))

    # Env vars override / fill gaps
    for key, env_var in _ENV_KEYS.items():
        val = os.environ.get(env_var)
        if val:
            creds[key] = val

    missing = [k for k in _ENV_KEYS if k not in creds]
    if missing:
        raise RuntimeError(
            f"Missing credentials: {missing}.\n"
            f"Set them in {_CREDS_FILE} or as environment variables "
            f"({', '.join(_ENV_KEYS[k] for k in missing)})."
        )
    return creds


class InstagramPoster:
    """Post images and Reels to Instagram via the Meta Graph API.

    Requires a Business or Creator Instagram account connected to a Meta
    developer app with ``instagram_content_publish`` permission.

    Parameters
    ----------
    ig_user_id:
        Numeric Instagram user ID (string).
    access_token:
        Long-lived access token (valid 60 days; refresh manually via the
        Meta token debugger before expiry).
    cloudinary_cloud / cloudinary_key / cloudinary_secret:
        Cloudinary credentials — used to host files publicly so Instagram
        can fetch them.  Free tier is sufficient for most use cases.
    """

    BASE = "https://graph.instagram.com/v21.0"

    def __init__(
        self,
        ig_user_id: str,
        access_token: str,
        cloudinary_cloud: str,
        cloudinary_key: str,
        cloudinary_secret: str,
    ):
        self.ig_user_id = ig_user_id
        self.access_token = access_token

        try:
            import cloudinary
            import cloudinary.uploader
        except ImportError:
            raise ImportError(
                "cloudinary package is required: pip install 'hecomes[instagram]'"
            )
        cloudinary.config(
            cloud_name=cloudinary_cloud,
            api_key=cloudinary_key,
            api_secret=cloudinary_secret,
        )
        self._cloudinary_uploader = cloudinary.uploader

    # ── Public API ────────────────────────────────────────────────────────────

    def post_image(self, local_path: str, caption: str = "") -> dict:
        """Upload *local_path* to Cloudinary and post it as an Instagram image.

        Image must have an aspect ratio between 4:5 and 1.91:1 (square works).
        """
        url = self._upload(local_path)
        print(f"Uploaded to Cloudinary: {url}")
        cid = self._create_container(image_url=url, caption=caption)
        self._wait(cid)
        result = self._publish(cid)
        print(f"Posted image — media ID: {result.get('id')}")
        return result

    def post_story_image(self, local_path: str) -> dict:
        """Upload *local_path* to Cloudinary and post it as an Instagram Story image.

        Image should be 9:16 aspect ratio (e.g. 1080×1920). Stories have no caption.
        """
        url = self._upload(local_path)
        print(f"Uploaded to Cloudinary: {url}")
        cid = self._create_container(image_url=url, media_type="STORIES")
        self._wait(cid)
        result = self._publish(cid)
        print(f"Posted Story — media ID: {result.get('id')}")
        return result

    def post_story_video(self, local_path: str) -> dict:
        """Upload *local_path* to Cloudinary and post it as an Instagram Story video.

        Video should be MP4, H.264, 9:16, max 60 seconds. Stories have no caption.
        """
        url = self._upload(local_path, resource_type="video")
        print(f"Uploaded to Cloudinary: {url}")
        cid = self._create_container(video_url=url, media_type="STORIES")
        self._wait(cid)
        result = self._publish(cid)
        print(f"Posted Story — media ID: {result.get('id')}")
        return result

    def post_reel(self, local_path: str, caption: str = "") -> dict:
        """Upload *local_path* to Cloudinary and post it as an Instagram Reel.

        Video must be MP4/MOV, H.264, 9:16 aspect ratio, 5–90 seconds.
        Generate with ``hecomes-instagram --type reel -W 540 -H 960``.
        """
        url = self._upload(local_path, resource_type="video")
        print(f"Uploaded to Cloudinary: {url}")
        cid = self._create_container(video_url=url, caption=caption, media_type="REELS")
        self._wait(cid)
        result = self._publish(cid)
        print(f"Posted Reel — media ID: {result.get('id')}")
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    def _upload(self, local_path: str, resource_type: str = None) -> str:
        """Upload file to Cloudinary and return its public HTTPS URL."""
        if resource_type is None:
            resource_type = (
                "video"
                if Path(local_path).suffix.lower() in _VIDEO_EXTENSIONS
                else "image"
            )
        result = self._cloudinary_uploader.upload(
            local_path, resource_type=resource_type
        )
        return result["secure_url"]

    def _create_container(self, caption: str = "", **kwargs) -> str:
        r = requests.post(
            f"{self.BASE}/{self.ig_user_id}/media",
            params={"access_token": self.access_token, "caption": caption, **kwargs},
        )
        self._raise(r)
        return r.json()["id"]

    def _wait(self, container_id: str, timeout: int = 300, poll: int = 5):
        """Poll container status until FINISHED or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = requests.get(
                f"{self.BASE}/{container_id}",
                params={"fields": "status_code", "access_token": self.access_token},
            )
            self._raise(r)
            status = r.json().get("status_code")
            if status == "FINISHED":
                return
            if status == "ERROR":
                raise RuntimeError(f"Container processing failed: {r.json()}")
            print(f"  Container status: {status} — waiting {poll}s…")
            time.sleep(poll)
        raise TimeoutError(f"Container {container_id} did not finish within {timeout}s")

    def _publish(self, container_id: str) -> dict:
        r = requests.post(
            f"{self.BASE}/{self.ig_user_id}/media_publish",
            params={"creation_id": container_id, "access_token": self.access_token},
        )
        self._raise(r)
        return r.json()

    @staticmethod
    def _raise(response: requests.Response):
        if not response.ok:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(
                f"Instagram API error {response.status_code}: {detail}"
            )
