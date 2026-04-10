import json
import logging
import re
from pathlib import Path

import click
import yt_dlp

logger = logging.getLogger(__name__)

SEARCH_QUERIES: list[str] = [
    "40 yard dash",
    "40 yard dash combine",
    "NFL combine 40",
    "high school 40 yard dash",
    "college 40 yard dash",
]

# Realistic 40-yard dash time range (seconds)
_TIME_MIN = 3.8
_TIME_MAX = 5.5
_TIME_RE = re.compile(r"\b([345]\.\d{1,3})\b")

# Clips longer than this are treated as compilations
MAX_CLIP_DURATION = 120  # seconds


def _sanitize_filename(name: str) -> str:
    """Strip unsafe filesystem characters and truncate to 80 chars."""
    cleaned = re.sub(r"[^\w\s\-.]", "_", name).strip()
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned[:80]


def _extract_metadata(title: str, description: str = "") -> dict:
    """Extract athlete name and known 40-yard dash time from title/description.

    Returns a dict with keys: title, athlete_name, known_time.
    """
    meta: dict = {"title": title, "athlete_name": None, "known_time": None}

    # Search title first, then description
    for text in (title, description):
        times = _TIME_RE.findall(text)
        valid = [t for t in times if _TIME_MIN <= float(t) <= _TIME_MAX]
        if valid:
            meta["known_time"] = float(valid[0])
            break

    # Heuristic: "Firstname Lastname …" at the start of the title
    name_match = re.match(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)", title)
    if name_match:
        meta["athlete_name"] = name_match.group(1)

    return meta


def search_and_download(
    queries: list[str],
    count: int,
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Search YouTube and download individual 40-yard dash clips.

    Skips compilations (duration > 120 s).  Prefers 720p–1080p.
    Saves metadata to output_dir/metadata.json.

    Returns a tuple of (downloaded, failed_downloads) where each entry in
    failed_downloads has keys: video_id, url, title, error.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    existing: dict = {}
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            existing = {}

    downloaded: list[dict] = []
    failed_downloads: list[dict] = []
    seen_ids: set[str] = set(existing.keys())

    # Flat search: get playlist of video IDs without downloading
    search_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
    }

    # Full info fetch (no download) to confirm duration
    info_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }

    # Prefer mp4 at 720p–1080p; fall back gracefully
    download_opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "format": (
            "bestvideo[height<=1080][height>=720][ext=mp4]+bestaudio[ext=m4a]"
            "/bestvideo[height<=1080][height>=720]+bestaudio"
            "/best[height<=1080][height>=720]"
            "/best"
        ),
        "noplaylist": True,
        "ignoreerrors": True,
    }

    for query in queries:
        if len(downloaded) >= count:
            break

        click.echo(f"Searching YouTube: {query!r}")

        try:
            with yt_dlp.YoutubeDL(search_opts) as ydl:
                result = ydl.extract_info(f"ytsearch50:{query}", download=False)
        except (yt_dlp.utils.DownloadError, urllib.error.URLError) as exc:
            logger.warning("Search failed for query %r: %s", query, exc)
            click.echo(f"  Search error: {exc}", err=True)
            continue

        if not result or "entries" not in result:
            continue

        for entry in result["entries"]:
            if len(downloaded) >= count:
                break
            if entry is None:
                continue

            video_id = entry.get("id")
            if not video_id or video_id in seen_ids:
                continue

            duration = entry.get("duration")
            full_info: dict | None = None

            # Fetch full metadata when duration is unknown from flat search
            if duration is None:
                try:
                    with yt_dlp.YoutubeDL(info_opts) as ydl:
                        full_info = ydl.extract_info(
                            f"https://www.youtube.com/watch?v={video_id}",
                            download=False,
                        )
                    if full_info:
                        duration = full_info.get("duration")
                except Exception as exc:
                    logger.warning(
                        "Failed to fetch full metadata for video %s: %s",
                        video_id,
                        exc,
                    )

            if duration is not None and duration > MAX_CLIP_DURATION:
                title = (full_info or entry).get("title", video_id)
                click.echo(f"  Skipping compilation ({duration:.0f}s): {title[:60]}")
                seen_ids.add(video_id)
                continue

            title = (full_info or entry).get("title", "")
            description = (full_info or {}).get("description", "") or ""
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            dl_opts = dict(download_opts)
            dl_opts["outtmpl"] = str(
                output_dir / f"{video_id}_{_sanitize_filename(title)}.%(ext)s"
            )

            click.echo(f"  [{len(downloaded) + 1}/{count}] {title[:70]}")
            try:
                with yt_dlp.YoutubeDL(dl_opts) as ydl:
                    ydl.download([video_url])
            except Exception as exc:
                logger.warning("Download failed for %s (%s): %s", video_id, title[:60], exc)
                click.echo(f"  Download failed: {exc}", err=True)
                failed_downloads.append(
                    {"video_id": video_id, "url": video_url, "title": title, "error": str(exc)}
                )
                seen_ids.add(video_id)
                continue

            meta = _extract_metadata(title, description)
            meta.update(
                video_id=video_id,
                url=video_url,
                duration=duration,
                query=query,
            )

            existing[video_id] = meta
            seen_ids.add(video_id)
            downloaded.append(meta)

    metadata_path.write_text(json.dumps(existing, indent=2))
    click.echo(f"Saved metadata -> {metadata_path}")

    return downloaded, failed_downloads
