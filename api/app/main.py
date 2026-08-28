from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field, HttpUrl
from playwright.async_api import async_playwright
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions

# CREATIVE_LOOP_V0_18_10_9_6_DETAIL_IDENTITY_CRAWLER
APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Shanghai")
WORKSPACE_DIR = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
BROWSER_STATE_DIR = Path(os.getenv("BROWSER_STATE_DIR", "/browser-state"))
LOG_DIR = Path(os.getenv("LOG_DIR", "/logs"))

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CACHE_DIR = Path(os.getenv("WHISPER_CACHE_DIR", "/model-cache"))
LANGUAGE_CONFIDENCE_THRESHOLD = float(os.getenv("LANGUAGE_CONFIDENCE_THRESHOLD", "0.80"))
MIN_SPEECH_SECONDS = float(os.getenv("MIN_SPEECH_SECONDS", "1.5"))

SELENIUM_REMOTE_URL = os.getenv("SELENIUM_REMOTE_URL", "http://insight-browser:4444")
INSIGHT_HOME_URL = os.getenv("INSIGHT_HOME_URL", "https://data.insightrackr.com/")
INSIGHT_BROWSER_VNC_URL = os.getenv(
    "INSIGHT_BROWSER_VNC_URL",
    "http://localhost:7900/?autoconnect=1&resize=scale",
)

RAW_DIR = WORKSPACE_DIR / "raw"
STAGING_DIR = RAW_DIR / "_staging"
REVIEW_DIR = WORKSPACE_DIR / "review"
CONFIG_DIR = WORKSPACE_DIR / "config"
DIAG_DIR = REVIEW_DIR / "diagnostics"
INSIGHT_DIAG_DIR = DIAG_DIR / "insight"

for p in [
    WORKSPACE_DIR,
    BROWSER_STATE_DIR,
    LOG_DIR,
    WHISPER_CACHE_DIR,
    RAW_DIR,
    STAGING_DIR,
    REVIEW_DIR,
    CONFIG_DIR,
    DIAG_DIR,
    INSIGHT_DIAG_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)

DB_PATH = WORKSPACE_DIR / "asset_index.db"
PRODUCT_MAP_PATH = CONFIG_DIR / "product_map.json"
BRAND_REGISTRY_DIR = CONFIG_DIR / "brand_registry"

_MODEL: WhisperModel | None = None
_MODEL_LOCK = threading.RLock()

_INSIGHT_DRIVER = None
_INSIGHT_DRIVER_LOCK = threading.RLock()

LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "tr": "Turkish",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "hi": "Hindi",
    "bn": "Bengali",
    "tl": "Filipino",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "uk": "Ukrainian",
    "el": "Greek",
    "he": "Hebrew",
    "fa": "Persian",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
}


def _init_product_map():
    if PRODUCT_MAP_PATH.exists():
        return
    PRODUCT_MAP_PATH.write_text(
        json.dumps(
            {
                "package_names": {
                    "com.storymatrix.drama": "DramaBox"
                },
                "app_titles": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_product_map() -> dict:
    _init_product_map()
    data = json.loads(PRODUCT_MAP_PATH.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise RuntimeError("product_map.json must contain a JSON object")
    data.setdefault("package_names", {})
    data.setdefault("app_titles", {})
    return data


def _detail_identity_icon_url(value: Any) -> str | None:
    """Accept only an explicit HTTP(S) app-icon field, never creative artwork."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or not re.match(r"^https?://", value, flags=re.IGNORECASE):
        return None
    return value


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assets (
                sha256 TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                source_url TEXT,
                original_filename TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


_init_product_map()
_init_db()

app = FastAPI(
    title="Creative Loop API",
    version="0.19.0",
    description="Floatboat-driven creative-material workflow API with spoken-language detection.",
)


class EchoRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)


class HashRequest(BaseModel):
    relative_path: str


class CrawlUrlRequest(BaseModel):
    urls: list[HttpUrl] = Field(min_length=1, max_length=50)
    mode: Literal["download"] = "download"
    wait_ms: int = Field(default=6000, ge=0, le=20000)
    target_product_name: str | None = None
    target_language: str | None = None
    download_cover: bool = False


class CrawlItem(BaseModel):
    url: HttpUrl
    product_name: str | None = None
    app_title: str | None = None
    package_name: str | None = None
    countries: list[str] = Field(default_factory=list)
    language: str | None = None
    material_type: str | None = None
    source_filename: str | None = None
    source_media_url: str | None = None
    impression: float | int | None = None
    search_rank: int | None = None


class CrawlItemsRequest(BaseModel):
    items: list[CrawlItem] = Field(min_length=1, max_length=100)
    wait_ms: int = Field(default=5000, ge=0, le=20000)
    download_cover: bool = False
    auto_detect_language: bool = True


class LanguageDetectRequest(BaseModel):
    relative_path: str = Field(
        description="Existing media path relative to /workspace, typically review/missing_language/..."
    )
    promote: bool = True
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class InsightAuthOpenRequest(BaseModel):
    url: HttpUrl | None = None


def _safe_workspace_path(relative_path: str) -> Path:
    candidate = (WORKSPACE_DIR / relative_path).resolve()
    workspace = WORKSPACE_DIR.resolve()
    if workspace not in candidate.parents and candidate != workspace:
        raise HTTPException(status_code=400, detail="Path must stay inside workspace")
    return candidate


def _display_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    except Exception:
        return url


def _folder_name(text: str, fallback: str) -> str:
    text = (text or "").strip()
    if not text:
        return fallback
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text)
    text = re.sub(r"\s+", " ", text).strip().strip(".")
    return text[:100] or fallback


def _run_folder_name() -> str:
    return datetime.now(ZoneInfo(APP_TIMEZONE)).strftime("%Y-%m-%d_%H%M")


def _extract_filename(url: str, fallback_prefix: str = "asset") -> str:
    name = Path(urlparse(url).path).name
    return name or f"{fallback_prefix}.bin"


def _normalize_language(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    invalid = {"未找到", "unknown", "unknown_language", "n/a", "na", "null", "none"}
    if value.lower() in invalid or value in invalid:
        return None
    if len(value) == 2 and value.lower() in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[value.lower()]
    return value


def _resolve_product_name(
    explicit_product_name: str | None,
    package_name: str | None,
    app_title: str | None,
) -> tuple[str | None, str | None]:
    if explicit_product_name and explicit_product_name.strip():
        return explicit_product_name.strip(), "explicit"

    # An app-title map is valid only when the current material response itself
    # supplied that App title. Do not resolve a package-only search record from
    # historical cache: package IDs can be shared by unrelated creatives.
    # Exact v2 Brand Registry mappings take priority while product_map remains
    # a compatibility source for legacy crawl and reporting paths.
    try:
        try:
            from .services.brand_registry import BrandRegistry
        except ImportError:
            from services.brand_registry import BrandRegistry
        profile = BrandRegistry(WORKSPACE_DIR).resolve(
            app_title=app_title,
            package_name=package_name,
        )
        if profile and profile.get("product_name"):
            return str(profile["product_name"]).strip(), "brand_registry_exact"
    except Exception:
        pass

    mapping = _load_product_map()
    if app_title and app_title in mapping["app_titles"]:
        return str(mapping["app_titles"][app_title]).strip(), "app_title_map"

    return None, None


def _guess_product_name_from_base_info(base_info: dict) -> str | None:
    for key in ("appList", "sourceAppList"):
        value = base_info.get(key)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict):
                for name_key in ("appName", "name", "title"):
                    if first.get(name_key):
                        return str(first[name_key]).strip()
            elif isinstance(first, str) and first.strip():
                return first.strip()
    return None


def _guess_language_from_base_info(base_info: dict) -> str | None:
    languages = base_info.get("languages") or []
    if isinstance(languages, list) and languages:
        first = languages[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return None


def _choose_media_urls(base_info: dict) -> tuple[str | None, str | None]:
    video_url = base_info.get("videoUrl")
    cover_url = base_info.get("thumbnailConverUrl") or base_info.get("converUrl")
    video = video_url.strip() if isinstance(video_url, str) and video_url.strip() else None
    cover = cover_url.strip() if isinstance(cover_url, str) and cover_url.strip() else None
    return video, cover


def _db_lookup_by_sha(sha256_value: str) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT file_path FROM assets WHERE sha256 = ?",
            (sha256_value,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _db_insert(
    sha256_value: str,
    file_path: str,
    source_url: str | None,
    original_filename: str | None,
):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO assets
            (sha256, file_path, source_url, original_filename, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                sha256_value,
                file_path,
                source_url,
                original_filename,
                datetime.now(ZoneInfo(APP_TIMEZONE)).isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _db_update_path(sha256_value: str, new_path: str):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "UPDATE assets SET file_path = ? WHERE sha256 = ?",
            (new_path, sha256_value),
        )
        conn.commit()
    finally:
        conn.close()


def _ensure_unique_path(folder: Path, filename: str) -> Path:
    base = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = folder / filename
    n = 1
    while candidate.exists():
        candidate = folder / f"{base}_{n:03d}{suffix}"
        n += 1
    return candidate


def _download_to_staging(url: str, run_folder: str, detail_id: str, filename: str) -> dict:
    item_staging = STAGING_DIR / run_folder / _folder_name(detail_id, "unknown_detail")
    item_staging.mkdir(parents=True, exist_ok=True)
    temp_path = _ensure_unique_path(item_staging, filename)

    digest = hashlib.sha256()
    size_bytes = 0

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with temp_path.open("wb") as f:
                for chunk in response.iter_bytes(1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)

    sha256_value = digest.hexdigest()
    existing = _db_lookup_by_sha(sha256_value)
    if existing:
        temp_path.unlink(missing_ok=True)
        return {
            "status": "duplicate",
            "sha256": sha256_value,
            "size_bytes": size_bytes,
            "existing_path": existing,
            "staging_path": None,
        }

    return {
        "status": "staged",
        "sha256": sha256_value,
        "size_bytes": size_bytes,
        "existing_path": None,
        "staging_path": str(temp_path.relative_to(WORKSPACE_DIR)),
    }


def _review_reason(product_name: str | None, language: str | None) -> str | None:
    if not product_name and not language:
        return "missing_product_and_language"
    if not product_name:
        return "missing_product"
    if not language:
        return "missing_language"
    return None


def _move_staged_file(
    staged_relative_path: str,
    run_folder: str,
    filename: str,
    product_name: str | None,
    language: str | None,
    review_reason: str | None,
) -> str:
    staged_path = _safe_workspace_path(staged_relative_path)

    if review_reason:
        destination_folder = REVIEW_DIR / review_reason / run_folder
        if product_name:
            destination_folder = destination_folder / _folder_name(product_name, "unknown_product")
    else:
        destination_folder = (
            RAW_DIR
            / run_folder
            / _folder_name(product_name or "", "unknown_product")
            / _folder_name(language or "", "unknown_language")
        )

    destination_folder.mkdir(parents=True, exist_ok=True)
    final_path = _ensure_unique_path(destination_folder, filename)
    shutil.move(str(staged_path), str(final_path))
    return str(final_path.relative_to(WORKSPACE_DIR))


def _metadata_sidecar_path(media_path: Path) -> Path:
    return media_path.with_name(media_path.name + ".metadata.json")


def _read_metadata_sidecar(media_path: Path) -> dict:
    sidecar = _metadata_sidecar_path(media_path)
    if not sidecar.is_file():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_metadata_sidecar(file_relative_path: str, metadata: dict) -> str:
    file_path = _safe_workspace_path(file_relative_path)
    sidecar = _metadata_sidecar_path(file_path)
    sidecar.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(sidecar.relative_to(WORKSPACE_DIR))


def _materialize_source_identity_snapshot(
    file_relative_path: str,
    brand_profile: dict | None,
) -> dict:
    """Store the acquired product identity beside its source video.

    A brand registry is useful for deduplicating future acquisitions, but it
    must not be the only location of a source video's icon/template.  Moving
    the project or deploying a fresh machine would otherwise turn a valid
    source identity into an OCR-only watermark scan.  The sidecar references
    this self-contained snapshot first.
    """
    profile = dict(brand_profile or {})
    media_path = _safe_workspace_path(file_relative_path)
    if not media_path.is_file() or not profile.get("brand_id"):
        return {"ok": False, "reason": "media_or_brand_identity_unavailable"}

    snapshot_dir = media_path.parent / (media_path.name + ".identity")
    source_dir = snapshot_dir / "source"
    templates_dir = snapshot_dir / "templates"
    source_dir.mkdir(parents=True, exist_ok=True)
    templates_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, str] = {}

    def copy_asset(relative_path: str | None, destination: Path) -> str | None:
        if not relative_path:
            return None
        try:
            original = _safe_workspace_path(str(relative_path))
        except Exception:
            return None
        if not original.is_file():
            return None
        try:
            shutil.copy2(original, destination)
            rel = destination.relative_to(WORKSPACE_DIR).as_posix()
            copied[str(relative_path)] = rel
            return rel
        except OSError:
            return None

    snapshot = json.loads(json.dumps(profile, ensure_ascii=False))
    icon = dict(snapshot.get("icon") or {})
    icon_rel = copy_asset(icon.get("relative_path"), source_dir / "icon.png")
    if icon_rel:
        icon["relative_path"] = icon_rel
    snapshot["icon"] = icon

    templates = dict(snapshot.get("templates") or {})
    template_assets = dict(templates.get("assets") or {})
    copied_templates: dict[str, str] = {}
    for name, relative_path in template_assets.items():
        safe_name = Path(str(name)).name
        if not safe_name:
            continue
        copied_rel = copy_asset(relative_path, templates_dir / safe_name)
        if copied_rel:
            copied_templates[str(name)] = copied_rel
    if copied_templates:
        templates["assets"] = {
            **template_assets,
            **copied_templates,
        }
    snapshot["templates"] = templates
    snapshot["snapshot"] = {
        "version": 1,
        "kind": "material_bound_source_identity",
        "source_video_relative_path": str(file_relative_path).replace("\\", "/"),
        "copied_asset_count": len(copied),
    }
    snapshot_path = snapshot_dir / "profile.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "profile_relative_path": snapshot_path.relative_to(WORKSPACE_DIR).as_posix(),
        "copied_asset_count": len(copied),
        "icon_relative_path": icon_rel,
        "brand_id": snapshot.get("brand_id"),
    }


def _get_whisper_model() -> WhisperModel:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = WhisperModel(
                WHISPER_MODEL_NAME,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                download_root=str(WHISPER_CACHE_DIR),
            )
        return _MODEL


def _detect_spoken_language(
    media_path: Path,
    confidence_threshold: float | None = None,
) -> dict:
    threshold = (
        LANGUAGE_CONFIDENCE_THRESHOLD
        if confidence_threshold is None
        else confidence_threshold
    )

    try:
        # Keep inference serialized on the starter setup. It avoids CPU/memory
        # spikes if Floatboat submits multiple jobs at once.
        with _MODEL_LOCK:
            model = _get_whisper_model()
            segments, info = model.transcribe(
                str(media_path),
                beam_size=1,
                vad_filter=True,
                language=None,
                task="transcribe",
                condition_on_previous_text=False,
                language_detection_segments=3,
            )

            transcript_parts = []
            for idx, segment in enumerate(segments):
                text = (segment.text or "").strip()
                if text:
                    transcript_parts.append(text)
                if idx >= 2:
                    break

        code = info.language
        probability = float(info.language_probability or 0.0)
        speech_seconds = float(info.duration_after_vad or 0.0)
        language_name = LANGUAGE_NAMES.get(code, code)

        accepted = (
            probability >= threshold
            and speech_seconds >= MIN_SPEECH_SECONDS
            and bool(language_name)
        )

        if speech_seconds < MIN_SPEECH_SECONDS:
            reason = "insufficient_speech"
        elif probability < threshold:
            reason = "low_confidence"
        else:
            reason = None

        return {
            "ok": True,
            "accepted": accepted,
            "language_code": code,
            "language": language_name if accepted else None,
            "candidate_language": language_name,
            "probability": round(probability, 4),
            "threshold": threshold,
            "speech_seconds": round(speech_seconds, 3),
            "minimum_speech_seconds": MIN_SPEECH_SECONDS,
            "reason": reason,
            "transcript_preview": " ".join(transcript_parts)[:500],
            "model": WHISPER_MODEL_NAME,
            "device": WHISPER_DEVICE,
            "compute_type": WHISPER_COMPUTE_TYPE,
        }
    except Exception as exc:
        return {
            "ok": False,
            "accepted": False,
            "language": None,
            "reason": "detection_error",
            "error": str(exc),
            "model": WHISPER_MODEL_NAME,
        }



def _insight_driver_alive(driver) -> bool:
    if driver is None:
        return False
    try:
        _ = driver.current_url
        return True
    except Exception:
        return False


def _get_or_create_insight_driver(start_url: str | None = None):
    global _INSIGHT_DRIVER

    with _INSIGHT_DRIVER_LOCK:
        if _insight_driver_alive(_INSIGHT_DRIVER):
            driver = _INSIGHT_DRIVER
        else:
            options = ChromeOptions()
            # Dedicated, non-default profile. The Docker volume persists it.
            options.add_argument("--user-data-dir=/home/seluser/insight-profile")
            options.add_argument("--profile-directory=Default")
            options.add_argument("--window-size=1440,1100")
            options.add_argument("--disable-notifications")
            # This profile is exclusively owned by the Selenium-controlled
            # visible Chrome. Its transient Singleton* locks are cleared by
            # the profile-init service before the browser starts, so a prior
            # container restart cannot make Chrome exit before session creation.
            options.set_capability("se:name", "creative-loop-insight-auth")

            driver = webdriver.Remote(
                command_executor=SELENIUM_REMOTE_URL,
                options=options,
            )
            _INSIGHT_DRIVER = driver

        # Do not hold the shared driver lock during navigation. A slow external
        # page load must not block the Console's read-only status probe and make
        # it falsely report that the browser is unavailable.
        try:
            driver.set_page_load_timeout(30)
        except Exception:
            pass

    target = start_url or INSIGHT_HOME_URL
    if target and driver.current_url != target:
        try:
            driver.get(target)
        except Exception:
            # The visible browser can still finish loading after Selenium's
            # bounded page-load wait. Its actual state is determined by the
            # later authenticated-session probe, not this navigation timeout.
            pass

    return driver


def _insight_auth_probe(driver) -> dict:
    """
    Read-only authenticated-session probe.

    Insight's frontend may store its access token in Web Storage and attach it
    through an application request interceptor. A plain browser fetch therefore
    can return "token empty" even when the visible UI is already logged in.

    This probe searches only inside the current browser session for token-like
    values and tries the common header forms internally. It never returns token,
    cookie, password, or credential values to the API caller.
    """
    script = r"""
        const done = arguments[arguments.length - 1];

        function collectCandidates() {
          const candidates = [];
          const seen = new Set();

          function add(value, path) {
            if (typeof value !== 'string') return;
            const v = value.trim();
            if (v.length < 12 || seen.has(v)) return;
            seen.add(v);
            candidates.push({ value: v, path });
          }

          function walk(value, path, depth) {
            if (depth > 5 || value === null || value === undefined) return;

            if (typeof value === 'string') {
              add(value, path);
              try {
                const parsed = JSON.parse(value);
                walk(parsed, path + '.__json__', depth + 1);
              } catch (e) {}
              return;
            }

            if (Array.isArray(value)) {
              value.slice(0, 50).forEach((v, i) => walk(v, `${path}[${i}]`, depth + 1));
              return;
            }

            if (typeof value === 'object') {
              Object.entries(value).slice(0, 100).forEach(([k, v]) => {
                const next = path ? `${path}.${k}` : k;
                if (/token|authorization|jwt|access[_-]?key|session/i.test(k)) {
                  if (typeof v === 'string') add(v, next);
                }
                walk(v, next, depth + 1);
              });
            }
          }

          for (const [storageName, storage] of [
            ['localStorage', window.localStorage],
            ['sessionStorage', window.sessionStorage]
          ]) {
            for (let i = 0; i < storage.length; i++) {
              const key = storage.key(i);
              const value = storage.getItem(key);
              const tokenishKey = /token|authorization|jwt|access|session|common|user/i.test(key || '');
              if (tokenishKey && typeof value === 'string') {
                add(value, `${storageName}.${key}`);
              }
              try {
                const parsed = JSON.parse(value);
                walk(parsed, `${storageName}.${key}`, 0);
              } catch (e) {}
            }
          }

          return candidates.slice(0, 40);
        }

        async function callFindSelf(headers) {
          try {
            const response = await fetch('/cas/api/user/v2/findSelf', {
              method: 'GET',
              credentials: 'include',
              headers
            });

            let body = null;
            try { body = await response.json(); } catch (e) {}

            return {
              http_status: response.status,
              code: body && body.code !== undefined ? body.code : null,
              message: body && (body.message || body.msg) ? (body.message || body.msg) : null,
              has_data: !!(body && body.data)
            };
          } catch (err) {
            return {
              http_status: null,
              code: null,
              message: String(err),
              has_data: false
            };
          }
        }

        (async () => {
          // First try normal cookie/session behaviour.
          let probe = await callFindSelf({});
          if (probe.http_status === 200 && probe.code === 200 && probe.has_data) {
            done({
              ...probe,
              auth_method: 'browser_session',
              token_candidate_count: 0
            });
            return;
          }

          const candidates = collectCandidates();

          const headerFactories = [
            ['token', v => ({ 'token': v })],
            ['Token', v => ({ 'Token': v })],
            ['x-token', v => ({ 'x-token': v })],
            ['X-Token', v => ({ 'X-Token': v })],
            ['Authorization bearer', v => ({ 'Authorization': `Bearer ${v}` })],
            ['Authorization raw', v => ({ 'Authorization': v })],
            ['satoken', v => ({ 'satoken': v })]
          ];

          for (const candidate of candidates) {
            for (const [label, makeHeaders] of headerFactories) {
              probe = await callFindSelf(makeHeaders(candidate.value));
              if (probe.http_status === 200 && probe.code === 200 && probe.has_data) {
                done({
                  ...probe,
                  auth_method: label,
                  token_candidate_count: candidates.length,
                  token_source_path: candidate.path
                });
                return;
              }
            }
          }

          done({
            ...probe,
            auth_method: null,
            token_candidate_count: candidates.length,
            token_source_path: null
          });
        })();
    """

    try:
        result = driver.execute_async_script(script)
        return result if isinstance(result, dict) else {
            "http_status": None,
            "code": None,
            "message": "Unexpected auth probe result",
            "has_data": False,
            "auth_method": None,
            "token_candidate_count": 0,
        }
    except Exception as exc:
        return {
            "http_status": None,
            "code": None,
            "message": str(exc),
            "has_data": False,
            "auth_method": None,
            "token_candidate_count": 0,
        }

def _insight_status_sync() -> dict:
    global _INSIGHT_DRIVER

    with _INSIGHT_DRIVER_LOCK:
        if not _insight_driver_alive(_INSIGHT_DRIVER):
            return {
                "browser_session_active": False,
                "logged_in": False,
                "current_url": None,
                "title": None,
                "auth_probe": None,
            }

        driver = _INSIGHT_DRIVER
        probe = _insight_auth_probe(driver)
        logged_in = (
            probe.get("http_status") == 200
            and probe.get("code") == 200
            and bool(probe.get("has_data"))
        )

        return {
            "browser_session_active": True,
            "logged_in": logged_in,
            "current_url": driver.current_url,
            "title": driver.title,
            "auth_probe": probe,
        }


def _close_insight_driver_sync() -> dict:
    global _INSIGHT_DRIVER

    with _INSIGHT_DRIVER_LOCK:
        if _INSIGHT_DRIVER is None:
            return {"closed": True, "had_session": False}

        try:
            _INSIGHT_DRIVER.quit()
        except Exception:
            pass
        finally:
            _INSIGHT_DRIVER = None

        return {"closed": True, "had_session": True}


def _detail_identity_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set)):
        return None
    text_value = str(value).strip()
    if not text_value:
        return None
    if text_value.casefold() in {
        "none", "null", "unknown", "n/a", "na", "-", "--"
    }:
        return None
    return text_value


def _detail_identity_package_from_url(value: str | None) -> str | None:
    raw = _detail_identity_text(value)
    if not raw:
        return None
    match = re.search(
        r"(?:play\.google\.com/store/apps/details\?[^#]*\bid=)([A-Za-z0-9._-]+)",
        raw,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else None


def _detail_id_from_url(value: str | None) -> str | None:
    """Return the Insight material id from a canonical detail-page URL."""
    raw = _detail_identity_text(value)
    if not raw:
        return None
    path = urlparse(raw).path.rstrip("/")
    marker = "/creative/material/detail/"
    if marker not in path:
        return None
    detail_id = path.rsplit(marker, 1)[-1].strip()
    return detail_id or None


def _identity_response_binding(
    expected_detail_id: str | None,
    request,
    response_url: str,
    response_text: str,
) -> str | None:
    """Classify whether an Insight response is a trustworthy identity source.

    `/imagevideo/distribute/app` is the detail page's dedicated *投放应用*
    endpoint. Its payload need not repeat the material id, but it is still the
    authoritative app title for the page currently being loaded. Conversely,
    recents/recommendation endpoints are page-adjacent and must not contribute
    App/Product fields unless they explicitly bind to the requested material.
    """
    response_path = urlparse(str(response_url or "")).path.casefold()
    if "/cas/api/v3/imagevideo/distribute/app" in response_path:
        return "detail_distribute_app_endpoint"

    # These endpoints can be rendered alongside a detail page but their app
    # fields belong to historical or recommended materials, not this creative.
    if any(marker in response_path for marker in (
        "/remember/",
        "/recommend",
        "/related",
        "/search",
        "/list",
    )):
        return None

    expected = str(expected_detail_id or "").strip().casefold()
    if not expected:
        return None

    candidates = [("response_url", response_url), ("response_body", response_text)]
    try:
        candidates.append(("request_body", request.post_data or ""))
    except Exception:
        pass

    for source, value in candidates:
        if expected in str(value or "").casefold():
            return source
    return None


def _merge_detail_identity(
    base: dict,
    extra: dict,
    *,
    prefer_extra: bool = False,
) -> dict:
    merged = dict(base or {})
    extra = dict(extra or {})

    for key in (
        "product_name",
        "product_name_source",
        "app_title",
        "package_name",
        "icon_url",
        "icon_source",
        "source_filename",
        "source_media_url",
    ):
        value = extra.get(key)
        if value not in (None, "") and (
            merged.get(key) in (None, "") or prefer_extra
        ):
            merged[key] = value

    countries = []
    for source in (merged.get("countries") or [], extra.get("countries") or []):
        if isinstance(source, str):
            source = [source]
        if isinstance(source, list):
            for country in source:
                country = _detail_identity_text(country)
                if country and country not in countries:
                    countries.append(country)
    if countries:
        merged["countries"] = countries

    sources = []
    for source in (merged.get("identity_sources") or [], extra.get("identity_sources") or []):
        if isinstance(source, str) and source and source not in sources:
            sources.append(source)
    if sources:
        merged["identity_sources"] = sources

    return merged


def _extract_detail_identity(payload, response_url: str | None = None) -> dict:
    """
    Extract only explicit App/Product/Package identity from Insight JSON.

    This intentionally does NOT infer product from country, spoken language,
    creative title, or package-name text. Generic name/title fields are used
    only when they occur inside an app/application context or a response whose
    endpoint is clearly app/distribution related.
    """
    result = {
        "product_name": None,
        "product_name_source": None,
        "app_title": None,
        "package_name": None,
        "icon_url": None,
        "icon_source": None,
        "countries": [],
        "source_filename": None,
        "source_media_url": None,
        "identity_sources": [],
    }

    endpoint = str(response_url or "")
    endpoint_lower = endpoint.casefold()
    endpoint_app_context = (
        "distribute/app" in endpoint_lower
        or "/app/" in endpoint_lower
        or endpoint_lower.endswith("/app")
        or "application" in endpoint_lower
    )

    product_keys = {
        "productname", "product_name", "appproductname", "app_product_name"
    }
    app_title_keys = {
        "apptitle", "app_title", "appname", "app_name",
        "applicationname", "application_name"
    }
    package_keys = {
        "packagename", "package_name", "bundleid", "bundle_id",
        "apppackage", "app_package"
    }
    country_keys = {
        "countries", "countrylist", "country_list",
        "countrycodes", "country_codes"
    }
    filename_keys = {
        "sourcefilename", "source_filename", "originalfilename",
        "original_filename", "filename", "file_name"
    }
    media_url_keys = {
        "sourcemediaurl", "source_media_url", "videourl", "video_url",
        "mediaurl", "media_url", "downloadurl", "download_url"
    }
    icon_url_keys = {
        "icon", "iconurl", "icon_url", "appicon", "app_icon",
        "appiconurl", "app_icon_url", "logo", "logourl", "logo_url",
    }

    def set_once(key, value, source=None):
        value = _detail_identity_text(value)
        if value and not result.get(key):
            result[key] = value
            if source and source not in result["identity_sources"]:
                result["identity_sources"].append(source)

    def visit(value, parent_key="", app_context=False):
        parent_lower = str(parent_key or "").casefold()
        local_app_context = (
            app_context
            or endpoint_app_context
            or "app" in parent_lower
            or "application" in parent_lower
        )

        if isinstance(value, dict):
            # Strong explicit fields.
            for key, nested in value.items():
                lower = str(key).casefold()

                if lower in product_keys:
                    before = result.get("product_name")
                    set_once("product_name", nested, "explicit_product_field")
                    if not before and result.get("product_name"):
                        result["product_name_source"] = "insight_detail_product_name"

                elif lower in app_title_keys:
                    set_once("app_title", nested, "explicit_app_field")

                elif lower in package_keys:
                    set_once("package_name", nested, "explicit_package_field")

                elif lower in filename_keys:
                    # filename is media metadata, never a product signal.
                    set_once("source_filename", nested, "explicit_filename_field")

                elif lower in media_url_keys:
                    set_once("source_media_url", nested, "explicit_media_url_field")

                elif lower in icon_url_keys and local_app_context:
                    icon_url = _detail_identity_icon_url(nested)
                    if icon_url and not result.get("icon_url"):
                        result["icon_url"] = icon_url
                        result["icon_source"] = "insight_app_record"
                        if "explicit_app_icon" not in result["identity_sources"]:
                            result["identity_sources"].append("explicit_app_icon")

                elif lower in country_keys:
                    countries = nested if isinstance(nested, list) else [nested]
                    for country in countries:
                        country = _detail_identity_text(country)
                        if country and country not in result["countries"]:
                            result["countries"].append(country)

            # Generic name/title is allowed only inside an application context.
            if local_app_context and not result.get("app_title"):
                for key in ("name", "title"):
                    if key in value:
                        set_once("app_title", value.get(key), "app_context_name")

            # Store/package links can explicitly yield package id.
            if not result.get("package_name"):
                for nested in value.values():
                    if isinstance(nested, str):
                        package = _detail_identity_package_from_url(nested)
                        if package:
                            set_once("package_name", package, "store_url_package")
                            break

            for key, nested in value.items():
                visit(nested, str(key), local_app_context)

        elif isinstance(value, list):
            for nested in value:
                visit(nested, parent_key, local_app_context)

        elif isinstance(value, str) and not result.get("package_name"):
            package = _detail_identity_package_from_url(value)
            if package:
                set_once("package_name", package, "store_url_package")

    visit(payload)

    # An explicit detail-page App name is the product identity when the API
    # does not separately expose productName. This is not inference from a
    # creative title: app_title is captured only from explicit app contexts.
    if not result.get("product_name") and result.get("app_title"):
        result["product_name"] = result["app_title"]
        result["product_name_source"] = "insight_detail_app_title"

    if result.get("product_name") and not result.get("product_name_source"):
        result["product_name_source"] = "insight_detail_explicit"

    return result


async def _load_insight_base_info(
    page,
    url: str,
    wait_ms: int,
) -> tuple[dict, dict, int | None, list[str], dict]:
    base_info_response: dict = {}
    detail_identity: dict = {}
    media_hits: list[str] = []
    identity_endpoints: list[dict] = []
    ignored_identity_endpoints: list[str] = []
    expected_detail_id = _detail_id_from_url(url)

    async def capture_response(response):
        try:
            req = response.request
            if req.resource_type == "media":
                media_hits.append(response.url)

            content_type = response.headers.get("content-type", "")
            response_url = str(response.url or "")

            if "application/json" not in content_type.casefold():
                return

            # Limit identity inspection to Insight's own API responses.
            parsed_url = urlparse(response_url)
            if not parsed_url.netloc.casefold().endswith("insightrackr.com"):
                return
            if "/cas/api/" not in parsed_url.path.casefold():
                return

            raw_response_text = await response.text()
            parsed_json = json.loads(raw_response_text)

            # Never borrow product/app fields from a recommendation, recents,
            # or another material rendered on the same detail page.
            binding = _identity_response_binding(
                expected_detail_id,
                req,
                response_url,
                raw_response_text,
            )
            if not binding:
                if parsed_url.path not in ignored_identity_endpoints:
                    ignored_identity_endpoints.append(parsed_url.path)
                return

            if (
                "/cas/api/v3/imagevideo/detail/baseinfo"
                in response_url.casefold()
                and parsed_json.get("code") == 200
                and isinstance(parsed_json.get("data"), dict)
            ):
                base_info_response.clear()
                base_info_response.update(parsed_json.get("data") or {})

            if parsed_json.get("code") not in (None, 200):
                return

            identity_payload = (
                parsed_json.get("data")
                if isinstance(parsed_json, dict) and "data" in parsed_json
                else parsed_json
            )

            extracted = _extract_detail_identity(
                identity_payload,
                response_url=response_url,
            )

            found_fields = [
                key
                for key in (
                    "product_name",
                    "app_title",
                    "package_name",
                    "source_filename",
                    "source_media_url",
                )
                if extracted.get(key)
            ]

            if found_fields:
                nonlocal detail_identity
                detail_identity = _merge_detail_identity(
                    detail_identity,
                    extracted,
                    prefer_extra=(binding == "detail_distribute_app_endpoint"),
                )
                identity_endpoints.append(
                    {
                        "path": parsed_url.path,
                        "fields": found_fields,
                        "binding": binding,
                    }
                )

        except Exception:
            # Network capture is diagnostic enrichment. One malformed optional
            # response must not block the base media acquisition.
            pass

    page.on("response", capture_response)
    nav = await page.goto(url, wait_until="domcontentloaded", timeout=45000)
    await page.wait_for_timeout(wait_ms)

    dom_media = await page.eval_on_selector_all(
        "video, source",
        """els => els.map(el => el.currentSrc || el.src || el.getAttribute('src') || '')
            .filter(Boolean)"""
    )

    # baseInfo is material-bound when its request/response includes the detail
    # id. Otherwise its appList/sourceAppList can be page-level adjacent data,
    # so retain only the media fields already collected above.
    base_info_identity = _extract_detail_identity(
        base_info_response,
        response_url="/cas/api/v3/imagevideo/detail/baseInfo",
    )
    if expected_detail_id:
        detail_identity = _merge_detail_identity(
            detail_identity,
            base_info_identity,
        )

    diagnostics = {
        "auth_injected": bool(
            await page.evaluate(
                """() => {
                    try { return !!window.localStorage.getItem('token'); }
                    catch (e) { return false; }
                }"""
            )
        ),
        "identity_response_count": len(identity_endpoints),
        "identity_binding_required": True,
        "identity_endpoints": identity_endpoints[:20],
        "identity_ignored_endpoint_count": len(ignored_identity_endpoints),
        "identity_ignored_endpoints": ignored_identity_endpoints[:20],
        "identity_fields": [
            key
            for key in (
                "product_name",
                "app_title",
                "package_name",
                "source_filename",
                "source_media_url",
            )
            if detail_identity.get(key)
        ],
    }

    return (
        dict(base_info_response),
        dict(detail_identity),
        nav.status if nav else None,
        list(dict.fromkeys(media_hits + dom_media)),
        diagnostics,
    )


@app.get("/")
async def root():
    return {"service": "creative-loop-api", "version": "0.18.10.9.6", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "creative-loop-api",
        "version": "0.18.10.9.6",
        "workspace": str(WORKSPACE_DIR),
        "timezone": APP_TIMEZONE,
        "asset_index_db": str(DB_PATH),
        "product_map": str(PRODUCT_MAP_PATH),
        "authenticated_browser": {
            "selenium_remote_url": SELENIUM_REMOTE_URL,
            "vnc_url": INSIGHT_BROWSER_VNC_URL,
            "profile": "persistent Docker volume",
        },
        "language_detection": {
            "engine": "faster-whisper",
            "model": WHISPER_MODEL_NAME,
            "device": WHISPER_DEVICE,
            "compute_type": WHISPER_COMPUTE_TYPE,
            "confidence_threshold": LANGUAGE_CONFIDENCE_THRESHOLD,
            "min_speech_seconds": MIN_SPEECH_SECONDS,
        },
    }


@app.get("/config/products")
async def get_product_map():
    return {
        "ok": True,
        "path": str(PRODUCT_MAP_PATH.relative_to(WORKSPACE_DIR)),
        "mapping": _load_product_map(),
    }


@app.post("/tools/echo")
async def echo(payload: EchoRequest):
    return {
        "ok": True,
        "echo": payload.message,
        "received_at": datetime.now().isoformat(timespec="seconds"),
    }


@app.post("/tools/hash")
async def hash_file(payload: HashRequest):
    path = _safe_workspace_path(payload.relative_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {payload.relative_path}")

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return {
        "ok": True,
        "relative_path": payload.relative_path,
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }



@app.post("/auth/insight/open")
async def auth_insight_open(payload: InsightAuthOpenRequest):
    """
    Start or reuse the dedicated visible Chrome session.
    The user completes login manually through noVNC.
    """
    target = str(payload.url) if payload.url else INSIGHT_HOME_URL

    try:
        driver = await asyncio.to_thread(_get_or_create_insight_driver, target)
        status = await asyncio.to_thread(_insight_status_sync)
        return {
            "ok": True,
            "message": "Insight browser is ready. Complete login manually in the noVNC window if needed.",
            "vnc_url": INSIGHT_BROWSER_VNC_URL,
            "current_url": driver.current_url,
            "status": status,
            "security_note": "Do not paste credentials, cookies or tokens into the API. Enter credentials only in the visible browser.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not start authenticated Insight browser: {exc}",
        ) from exc


@app.get("/auth/insight/status")
async def auth_insight_status():
    status = await asyncio.to_thread(_insight_status_sync)
    return {
        "ok": True,
        **status,
        "vnc_url": INSIGHT_BROWSER_VNC_URL,
    }


@app.post("/auth/insight/close")
async def auth_insight_close():
    result = await asyncio.to_thread(_close_insight_driver_sync)
    return {"ok": True, **result}


@app.post("/language/detect")
async def language_detect(payload: LanguageDetectRequest):
    media_path = _safe_workspace_path(payload.relative_path)
    if not media_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {payload.relative_path}")

    metadata = _read_metadata_sidecar(media_path)
    detection = await asyncio.to_thread(
        _detect_spoken_language,
        media_path,
        payload.confidence_threshold,
    )

    result = {
        "ok": detection.get("ok", False),
        "relative_path": payload.relative_path,
        "detection": detection,
        "promoted": False,
    }

    # Save detection result even if confidence is insufficient.
    metadata["language_detection"] = detection
    metadata["language_detection_checked_at"] = datetime.now(
        ZoneInfo(APP_TIMEZONE)
    ).isoformat(timespec="seconds")

    if not payload.promote or not detection.get("accepted"):
        if metadata:
            _write_metadata_sidecar(payload.relative_path, metadata)
        return result

    product_name = metadata.get("product_name")
    if not product_name:
        result["promotion_reason"] = "missing_product"
        _write_metadata_sidecar(payload.relative_path, metadata)
        return result

    language = detection["language"]

    # Preserve the original run folder when the file is under:
    # review/<reason>/<run-folder>/<product>/<filename>
    parts = Path(payload.relative_path).parts
    if len(parts) >= 4 and parts[0] == "review":
        run_folder = parts[2]
    else:
        run_folder = _run_folder_name()

    destination_folder = (
        RAW_DIR
        / run_folder
        / _folder_name(product_name, "unknown_product")
        / _folder_name(language, "unknown_language")
    )
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination_path = _ensure_unique_path(destination_folder, media_path.name)

    old_sidecar = _metadata_sidecar_path(media_path)
    shutil.move(str(media_path), str(destination_path))

    metadata["language"] = language
    metadata["language_source"] = "spoken_language_whisper"
    metadata["routing"] = "raw"
    metadata["review_reason"] = None
    metadata["promoted_at"] = datetime.now(ZoneInfo(APP_TIMEZONE)).isoformat(timespec="seconds")

    new_relative_path = str(destination_path.relative_to(WORKSPACE_DIR))
    new_sidecar = _metadata_sidecar_path(destination_path)
    new_sidecar.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if old_sidecar.exists():
        old_sidecar.unlink(missing_ok=True)

    sha256_value = metadata.get("sha256")
    if sha256_value:
        _db_update_path(sha256_value, new_relative_path)

    result.update(
        {
            "promoted": True,
            "product_name": product_name,
            "language": language,
            "new_relative_path": new_relative_path,
            "metadata_sidecar": str(new_sidecar.relative_to(WORKSPACE_DIR)),
        }
    )
    return result


@app.post("/crawl/items")
async def crawl_items(payload: CrawlItemsRequest):
    run_folder = _run_folder_name()
    results = []

    # Reuse the currently logged-in Insight browser token for the detail-page
    # Playwright context. Search already uses this same browser login.
    auth_token = None
    auth_user_agent = None
    auth_diag = {
        "available": False,
        "injected": False,
        "error": None,
    }

    auth_getter = globals().get("_cl_auth_context")
    if callable(auth_getter):
        try:
            auth_token, auth_user_agent = await asyncio.to_thread(auth_getter)
            auth_diag["available"] = bool(auth_token)
        except Exception as exc:
            # Keep the baseInfo/media path available, but expose a non-sensitive
            # diagnostic because missing auth can explain missing App metadata.
            auth_diag["error"] = type(exc).__name__

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        context_kwargs = {}
        if auth_user_agent:
            context_kwargs["user_agent"] = str(auth_user_agent)

        context = await browser.new_context(**context_kwargs)

        if auth_token:
            # Insight's frontend request interceptor reads localStorage.token.
            # The token never leaves the local container or API response.
            token_literal = json.dumps(str(auth_token))
            await context.add_init_script(
                script=f"""
                    (() => {{
                        try {{
                            window.localStorage.setItem('token', {token_literal});
                        }} catch (e) {{}}
                    }})();
                """
            )
            auth_diag["injected"] = True

        try:
            for item in payload.items:
                url = str(item.url)
                detail_id = urlparse(url).path.rstrip("/").split("/")[-1]
                page = await context.new_page()

                try:
                    (
                        base_info,
                        detail_identity,
                        http_status,
                        media_hits,
                        detail_diag,
                    ) = await _load_insight_base_info(
                        page, url, payload.wait_ms
                    )

                    # Explicit request metadata is preserved, but detail-page
                    # identity fills any missing fields.
                    effective_app_title = (
                        detail_identity.get("app_title")
                        or item.app_title
                    )
                    effective_package_name = (
                        detail_identity.get("package_name")
                        or item.package_name
                    )
                    effective_icon_url = detail_identity.get("icon_url")
                    effective_countries = (
                        detail_identity.get("countries")
                        or item.countries
                    )

                    detail_product = detail_identity.get("product_name")
                    # Current, material-bound detail metadata outranks a search
                    # row. Search fields are discovery hints only and can be
                    # stale or incomplete after a material is reassigned.
                    if detail_product:
                        product_name = str(detail_product).strip()
                        product_source = (
                            detail_identity.get("product_name_source")
                            or "insight_detail"
                        )
                    elif item.product_name:
                        product_name, product_source = _resolve_product_name(
                            item.product_name,
                            effective_package_name,
                            effective_app_title,
                        )
                    else:
                        base_product = _guess_product_name_from_base_info(base_info)
                        # baseInfo may contain a page-wide app list. Use it
                        # only when material-bound detail identity was captured;
                        # otherwise preserve missing_product for review.
                        if detail_identity.get("app_title") or detail_identity.get("package_name"):
                            product_name, product_source = _resolve_product_name(
                                base_product,
                                effective_package_name,
                                effective_app_title,
                            )
                        else:
                            product_name, product_source = None, None

                    video_url, cover_url = _choose_media_urls(base_info)

                    if not video_url:
                        detail_media = detail_identity.get("source_media_url")
                        if isinstance(detail_media, str) and detail_media.strip():
                            video_url = detail_media.strip()

                    if not video_url and item.source_media_url:
                        video_url = str(item.source_media_url).strip()

                    if not video_url:
                        for candidate in media_hits:
                            if urlparse(candidate).path.lower().endswith(".mp4"):
                                video_url = candidate
                                break

                    media_url = video_url or cover_url
                    if not media_url:
                        results.append(
                            {
                                "url": url,
                                "ok": False,
                                "status": "no_media",
                                "detail_id": detail_id,
                                "product_name": product_name,
                                "product_name_source": product_source,
                                "app_title": effective_app_title,
                                "package_name": effective_package_name,
                                "detail_identity": {
                                    "fields": detail_diag.get("identity_fields"),
                                    "identity_endpoints": detail_diag.get("identity_endpoints"),
                                    "auth_injected": detail_diag.get("auth_injected"),
                                },
                                "error": "No downloadable media URL found.",
                            }
                        )
                        continue

                    language = (
                        _normalize_language(item.language)
                        or _normalize_language(_guess_language_from_base_info(base_info))
                    )
                    language_source = "provided_or_insight" if language else None

                    filename = (
                        _detail_identity_text(item.source_filename)
                        or _detail_identity_text(detail_identity.get("source_filename"))
                        or _extract_filename(media_url, "asset")
                    )

                    staged = _download_to_staging(
                        media_url,
                        run_folder,
                        detail_id,
                        filename,
                    )

                    identity_summary = {
                        "app_title": effective_app_title,
                        "package_name": effective_package_name,
                        "icon_url": effective_icon_url,
                        "product_name": product_name,
                        "product_name_source": product_source,
                        "countries": effective_countries,
                        "source_filename": filename,
                        "detail_auth_available": auth_diag.get("available"),
                        "detail_auth_injected": detail_diag.get("auth_injected"),
                        "detail_identity_fields": detail_diag.get("identity_fields"),
                        "detail_identity_endpoints": detail_diag.get("identity_endpoints"),
                    }

                    # P0: persist explicit Insight identity as a reusable Brand
                    # Profile. It is non-blocking: absent/bad optional icon data
                    # must never break media acquisition.
                    brand_identity_profile = None
                    try:
                        try:
                            from .services.brand_identity import identity_from_detail
                            from .services.brand_registry import BrandRegistry
                        except ImportError:
                            from services.brand_identity import identity_from_detail
                            from services.brand_registry import BrandRegistry
                        brand_identity = identity_from_detail(
                            {
                                "product_name": product_name,
                                "app_title": effective_app_title,
                                "package_name": effective_package_name,
                                "icon_url": effective_icon_url,
                                "icon_source": detail_identity.get("icon_source"),
                                "detail_id": detail_id,
                            },
                            _load_product_map(),
                        )
                        if brand_identity.product_name:
                            icon_bytes = None
                            # The URL is material-bound Insight metadata.  Fetch
                            # the small app icon at acquisition time so the
                            # production watermark detector receives product
                            # icon + logo-template evidence with the source.
                            if effective_icon_url:
                                try:
                                    with httpx.Client(
                                        timeout=20,
                                        follow_redirects=True,
                                    ) as icon_client:
                                        icon_response = icon_client.get(
                                            str(effective_icon_url)
                                        )
                                        icon_response.raise_for_status()
                                        candidate_bytes = icon_response.content
                                        if 64 <= len(candidate_bytes) <= 8 * 1024 * 1024:
                                            icon_bytes = candidate_bytes
                                except Exception:
                                    # Icon acquisition augments detection but
                                    # must not block the underlying video crawl.
                                    icon_bytes = None
                            brand_identity_profile = BrandRegistry(WORKSPACE_DIR).upsert(
                                brand_identity,
                                icon_bytes=icon_bytes,
                                source_url=effective_icon_url,
                            )
                    except Exception as exc:
                        identity_summary["brand_registry_error"] = type(exc).__name__
                    if brand_identity_profile:
                        identity_summary["brand_id"] = brand_identity_profile.get("brand_id")
                        identity_summary["brand_profile"] = "config/brand_registry/{}/profile.json".format(
                            brand_identity_profile.get("brand_id")
                        )

                    if staged["status"] == "duplicate":
                        existing_path = staged["existing_path"]
                        existing_metadata = {}

                        if existing_path:
                            try:
                                existing_media_path = _safe_workspace_path(existing_path)
                                existing_metadata = _read_metadata_sidecar(existing_media_path)
                            except Exception:
                                existing_metadata = {}

                        # A fresh, material-bound detail acquisition is newer
                        # evidence than a historical SHA-duplicate sidecar.
                        # Keep the archived source's identity only when the
                        # current detail page genuinely has no product value.
                        effective_product_name = (
                            product_name
                            or existing_metadata.get("product_name")
                        )
                        effective_product_source = (
                            product_source
                            or existing_metadata.get("product_name_source")
                        )
                        effective_language = (
                            existing_metadata.get("language")
                            or language
                        )
                        effective_language_source = (
                            existing_metadata.get("language_source")
                            or language_source
                        )
                        effective_detection = (
                            existing_metadata.get("language_detection")
                        )
                        duplicate_app_title = (
                            effective_app_title
                            or existing_metadata.get("app_title")
                        )
                        duplicate_package = (
                            effective_package_name
                            or existing_metadata.get("package_name")
                        )

                        # Repair the existing sidecar in place when the current
                        # authenticated detail acquisition recovered better identity.
                        if existing_path and (
                            effective_product_name
                            or duplicate_app_title
                            or duplicate_package
                        ):
                            identity_snapshot = _materialize_source_identity_snapshot(
                                existing_path,
                                brand_identity_profile,
                            )
                            repaired_metadata = dict(existing_metadata or {})
                            repaired_metadata.update(
                                {
                                    key: value
                                    for key, value in {
                                        "detail_id": detail_id,
                                        "source_detail_url": url,
                                        "source_media_url": _display_url(media_url),
                                        "app_title": duplicate_app_title,
                                        "package_name": duplicate_package,
                                        "icon_url": effective_icon_url,
                                        "brand_id": (brand_identity_profile or {}).get("brand_id"),
                                        # Always prefer the per-video snapshot.
                                        # The registry location is kept only as
                                        # diagnostic provenance and is never a
                                        # deployment prerequisite for matching.
                                        "brand_profile": identity_snapshot.get("profile_relative_path"),
                                        "brand_registry_profile": (identity_summary.get("brand_profile")),
                                        "brand_identity_snapshot": identity_snapshot,
                                        "product_name": effective_product_name,
                                        "product_name_source": effective_product_source,
                                        "countries": (
                                            effective_countries
                                            or existing_metadata.get("countries")
                                        ),
                                        "source_filename": (
                                            existing_metadata.get("source_filename")
                                            or filename
                                        ),
                                        "detail_identity_diagnostic": identity_summary,
                                    }.items()
                                    if value not in (None, "")
                                }
                            )
                            try:
                                _write_metadata_sidecar(
                                    existing_path,
                                    repaired_metadata,
                                )
                            except Exception:
                                pass

                        results.append(
                            {
                                "url": url,
                                "ok": True,
                                "status": "duplicate",
                                "detail_id": detail_id,
                                "product_name": effective_product_name,
                                "product_name_source": effective_product_source,
                                "app_title": duplicate_app_title,
                                "package_name": duplicate_package,
                                "brand_id": (brand_identity_profile or {}).get("brand_id"),
                                "source_filename": filename,
                                "source_media_url": _display_url(media_url),
                                "language": effective_language,
                                "language_source": effective_language_source,
                                "language_detection": effective_detection,
                                "countries": (
                                    effective_countries
                                    or existing_metadata.get("countries")
                                ),
                                "sha256": staged["sha256"],
                                "existing_path": existing_path,
                                "metadata_complete": bool(
                                    effective_product_name and effective_language
                                ),
                                "detail_identity": identity_summary,
                                "metadata_sidecar": (
                                    str(
                                        _metadata_sidecar_path(
                                            _safe_workspace_path(existing_path)
                                        ).relative_to(WORKSPACE_DIR)
                                    )
                                    if existing_path
                                    and _metadata_sidecar_path(
                                        _safe_workspace_path(existing_path)
                                    ).is_file()
                                    else None
                                ),
                            }
                        )
                        continue

                    language_detection = None
                    if not language and payload.auto_detect_language and video_url:
                        staged_path = _safe_workspace_path(staged["staging_path"])
                        language_detection = await asyncio.to_thread(
                            _detect_spoken_language,
                            staged_path,
                            None,
                        )
                        if language_detection.get("accepted"):
                            language = language_detection["language"]
                            language_source = "spoken_language_whisper"

                    common_metadata = {
                        "detail_id": detail_id,
                        "source_detail_url": url,
                        "source_media_url": _display_url(media_url),
                        "source_cover_url": _display_url(cover_url),
                        "source_filename": filename,
                        "app_title": effective_app_title,
                        "package_name": effective_package_name,
                        "icon_url": effective_icon_url,
                        "product_name": product_name,
                        "product_name_source": product_source,
                        "countries": effective_countries,
                        "language": language,
                        "language_source": language_source,
                        "language_detection": language_detection,
                        "material_type": item.material_type,
                        "insight_material_type": base_info.get("materialType"),
                        "size": base_info.get("size"),
                        "video_duration_seconds": base_info.get("videoTimeSpan"),
                        "width": base_info.get("width"),
                        "height": base_info.get("height"),
                        "global_first_time": base_info.get("globalFirstTime"),
                        "global_last_time": base_info.get("globalLastTime"),
                        "search_impression": item.impression,
                        "search_rank": item.search_rank,
                        "detail_identity_diagnostic": identity_summary,
                        "brand_id": (brand_identity_profile or {}).get("brand_id"),
                    }

                    reason = _review_reason(product_name, language)
                    final_relative_path = _move_staged_file(
                        staged["staging_path"],
                        run_folder,
                        filename,
                        product_name,
                        language,
                        reason,
                    )

                    identity_snapshot = _materialize_source_identity_snapshot(
                        final_relative_path,
                        brand_identity_profile,
                    )

                    metadata = {
                        **common_metadata,
                        "sha256": staged["sha256"],
                        "size_bytes": staged["size_bytes"],
                        "routing": "review" if reason else "raw",
                        "review_reason": reason,
                        # This path is inside the source video's own archive
                        # directory.  Watermark detection must use it before
                        # consulting any global registry cache.
                        "brand_profile": identity_snapshot.get("profile_relative_path"),
                        "brand_registry_profile": identity_summary.get("brand_profile"),
                        "brand_identity_snapshot": identity_snapshot,
                        "saved_at": datetime.now(
                            ZoneInfo(APP_TIMEZONE)
                        ).isoformat(timespec="seconds"),
                    }
                    sidecar_path = _write_metadata_sidecar(
                        final_relative_path,
                        metadata,
                    )

                    _db_insert(
                        staged["sha256"],
                        final_relative_path,
                        media_url,
                        filename,
                    )

                    results.append(
                        {
                            "url": url,
                            "ok": True,
                            "status": "needs_review" if reason else "downloaded",
                            "detail_id": detail_id,
                            "product_name": product_name,
                            "product_name_source": product_source,
                            "app_title": effective_app_title,
                            "package_name": effective_package_name,
                            "source_filename": filename,
                            "language": language,
                            "language_source": language_source,
                            "language_detection": language_detection,
                            "countries": effective_countries,
                            "metadata_complete": reason is None,
                            "review_reason": reason,
                            "sha256": staged["sha256"],
                            "size_bytes": staged["size_bytes"],
                            "relative_path": final_relative_path,
                            "metadata_sidecar": sidecar_path,
                            "source_media_url": _display_url(media_url),
                            "detail_identity": identity_summary,
                        }
                    )

                except Exception as exc:
                    results.append(
                        {
                            "url": url,
                            "ok": False,
                            "status": "error",
                            "detail_id": detail_id,
                            "error": str(exc),
                        }
                    )
                finally:
                    await page.close()
        finally:
            await context.close()
            await browser.close()

    return {
        "ok": all(item.get("ok") for item in results),
        "mode": "items",
        "run_folder": run_folder,
        "count": len(results),
        "results": results,
        "detail_auth": {
            "available": auth_diag.get("available"),
            "injected": auth_diag.get("injected"),
            "error": auth_diag.get("error"),
        },
        "rules": {
            "final_directory": "raw/<run minute>/<product>/<language>/<original filename>",
            "missing_metadata": "review/<reason>/<run minute>/...",
            "language_policy": "Insight explicit language first; otherwise spoken-language detection; never infer from countries.",
            "product_policy": "Use explicit Insight detail/search App/Product identity only; never infer product from language/country.",
            "auto_detection_threshold": LANGUAGE_CONFIDENCE_THRESHOLD,
        },
    }


@app.post("/crawl/url")
async def crawl_url(payload: CrawlUrlRequest):
    # Compatibility wrapper: production use should prefer /crawl/items.
    wrapped = CrawlItemsRequest(
        items=[
            CrawlItem(
                url=url,
                product_name=payload.target_product_name,
                language=payload.target_language,
            )
            for url in payload.urls
        ],
        wait_ms=payload.wait_ms,
        download_cover=payload.download_cover,
        auto_detect_language=True,
    )
    return await crawl_items(wrapped)


# === CREATIVE_LOOP_V0_8_SEARCH_BEGIN ===
# Added by creative-loop v0.18.8.1 stabilized search block.
# Purpose: Insight keyword search API -> optional reuse of existing /crawl/items pipeline.
# No credentials/tokens are returned by these endpoints.

import asyncio as _cl_asyncio
import os as _cl_search_os
import json as _cl_json
import re as _cl_re
import urllib.request as _cl_urlrequest
import urllib.error as _cl_urlerror
from typing import List as _CLList
from pydantic import BaseModel as _CLBaseModel, Field as _CLField
from fastapi import HTTPException as _CLHTTPException

_CL_INSIGHT_SEARCH_URL = "https://data.insightrackr.com/cas/api/v2/imagevideo/search"
_CL_DETAIL_URL_BASE = "https://data.insightrackr.com/creative/material/detail"

class _CLSearchRequest(_CLBaseModel):
    keyword: str = _CLField(..., min_length=1)
    start_date: str
    end_date: str
    media_ids: _CLList[str] = _CLField(default_factory=list)
    ad_media_type_ids: _CLList[str] = _CLField(default_factory=list)
    industry_ids: _CLList[str] = _CLField(default_factory=list)
    material_types: _CLList[str] = _CLField(default_factory=lambda: ["video"])
    top_n: int = _CLField(default=10, ge=1, le=300)
    page_size: int = _CLField(default=60, ge=1, le=100)
    download: bool = False
    auto_detect_language: bool = True
    wait_ms: int = _CLField(default=5000, ge=0, le=30000)


def _cl_driver_globals():
    """Return module globals that look like Selenium WebDriver instances."""
    _items = []
    for _name, _value in list(globals().items()):
        if _name.startswith("_cl_"):
            continue
        try:
            _sid = getattr(_value, "session_id", None)
            _exec = getattr(_value, "execute_script", None)
            if _sid and callable(_exec):
                _items.append((_name, _value, str(_sid)))
        except Exception:
            continue
    return _items


def _cl_find_active_webdriver():
    """
    Find an actually usable Selenium WebDriver stored by the existing auth code.
    A session object can survive after its Chrome window is gone, so merely
    having session_id is not sufficient.
    """
    # The primary Insight session is kept in _INSIGHT_DRIVER. It starts with an
    # underscore, so the legacy globals scanner below deliberately skips it.
    # Check it first to keep the Console's status / recovery path attached to
    # the same browser that noVNC displays.
    global _INSIGHT_DRIVER
    if _insight_driver_alive(_INSIGHT_DRIVER):
        return _INSIGHT_DRIVER

    for _name, _value, _sid in _cl_driver_globals():
        try:
            _handles = _value.window_handles
            if not _handles:
                continue
            _ = _value.current_url
            return _value
        except Exception:
            continue
    return None


def _cl_auth_context():
    _driver = _cl_find_active_webdriver()
    if _driver is None:
        raise _CLHTTPException(
            status_code=409,
            detail={
                "stage": "auth_context",
                "error": "no_usable_browser_session",
                "message": "No usable Insight browser session is attached to the API process."
            }
        )

    try:
        _token = _driver.execute_script(
            "return window.localStorage.getItem('token');"
        )
        _ua = _driver.execute_script("return navigator.userAgent;")
    except Exception as _exc:
        raise _CLHTTPException(
            status_code=409,
            detail={
                "stage": "auth_context",
                "error": "stale_browser_session",
                "message": str(_exc),
            }
        )

    if not _token:
        raise _CLHTTPException(
            status_code=401,
            detail={
                "stage": "auth_context",
                "error": "login_required",
                "message": "Insight login token was not found. Complete login in the visible noVNC browser, then retry."
            }
        )

    return _token, (_ua or "Mozilla/5.0")


def _cl_grid_base_url():
    return str(
        _cl_search_os.environ.get("SELENIUM_REMOTE_URL")
        or "http://insight-browser:4444"
    ).rstrip("/")


def _cl_grid_graphql_sessions(_timeout: int = 10):
    """List active sessions from the dedicated Insight Selenium Grid."""
    _url = _cl_grid_base_url() + "/graphql"
    _payload = {
        "query": "{ sessionsInfo { sessions { id } } grid { sessionCount sessionQueueSize } }"
    }
    _body = _cl_json.dumps(_payload).encode("utf-8")
    _request = _cl_urlrequest.Request(
        _url,
        data=_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _cl_urlrequest.urlopen(_request, timeout=_timeout) as _response:
        _raw = _response.read().decode("utf-8", errors="replace")
        _data = _cl_json.loads(_raw)

    _root = (_data or {}).get("data") or {}
    _sessions = (((_root.get("sessionsInfo") or {}).get("sessions")) or [])
    _ids = []
    for _item in _sessions:
        if isinstance(_item, dict) and _item.get("id"):
            _ids.append(str(_item["id"]))

    _grid = _root.get("grid") or {}
    return {
        "session_ids": _ids,
        "session_count": _grid.get("sessionCount"),
        "session_queue_size": _grid.get("sessionQueueSize"),
    }


def _cl_grid_delete_session(_session_id: str, _timeout: int = 10):
    _request = _cl_urlrequest.Request(
        _cl_grid_base_url() + "/session/" + str(_session_id),
        data=None,
        method="DELETE",
    )
    try:
        with _cl_urlrequest.urlopen(_request, timeout=_timeout) as _response:
            _response.read()
            return True
    except Exception:
        return False


def _cl_grid_clear_queue(_timeout: int = 10):
    # Best effort. Some Grid configurations protect this endpoint with a
    # registration secret; recovery does not fail if queue clearing is denied.
    _request = _cl_urlrequest.Request(
        _cl_grid_base_url() + "/se/grid/newsessionqueue/queue",
        data=None,
        method="DELETE",
    )
    try:
        with _cl_urlrequest.urlopen(_request, timeout=_timeout) as _response:
            _response.read()
            return True
    except Exception:
        return False


def _cl_clear_local_driver_globals():
    """
    Quit and detach Selenium driver objects known to this API process.
    This intentionally does not touch Playwright objects.
    """
    global _INSIGHT_DRIVER
    _cleared = []

    # _INSIGHT_DRIVER is intentionally excluded from _cl_driver_globals()
    # because of its underscore name. Clear it explicitly during recovery so
    # the single Selenium slot is released exactly once.
    if _INSIGHT_DRIVER is not None:
        try:
            _INSIGHT_DRIVER.quit()
        except Exception:
            pass
        finally:
            _INSIGHT_DRIVER = None
            _cleared.append("_INSIGHT_DRIVER")

    for _name, _driver, _sid in _cl_driver_globals():
        try:
            _driver.quit()
        except Exception:
            pass
        try:
            globals()[_name] = None
            _cleared.append(_name)
        except Exception:
            pass
    return _cleared


def _cl_open_auth_browser_via_local_api(_timeout: int = 90):
    """
    Reuse the existing tested /auth/insight/open endpoint rather than duplicate
    its Chrome profile/options logic.
    """
    _body = b"{}"
    _request = _cl_urlrequest.Request(
        "http://127.0.0.1:8000/auth/insight/open",
        data=_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _cl_urlrequest.urlopen(_request, timeout=_timeout) as _response:
            _raw = _response.read().decode("utf-8", errors="replace")
            try:
                _json = _cl_json.loads(_raw)
            except Exception:
                _json = {}
            return {
                "ok": 200 <= int(_response.status) < 300,
                "http_status": int(_response.status),
                "response": _json if isinstance(_json, dict) else {},
            }
    except _cl_urlerror.HTTPError as _exc:
        try:
            _raw = _exc.read().decode("utf-8", errors="replace")
            _json = _cl_json.loads(_raw)
        except Exception:
            _json = {}
        return {
            "ok": False,
            "http_status": int(getattr(_exc, "code", 0) or 0),
            "response": _json if isinstance(_json, dict) else {},
        }
    except Exception as _exc:
        return {
            "ok": False,
            "http_status": None,
            "response": {},
            "error": type(_exc).__name__,
            "message": str(_exc),
        }


def _cl_recover_auth_session_sync():
    """
    Recover the dedicated Insight Selenium browser without restarting Docker.

    Because insight-browser is a dedicated single-purpose Grid, any session on
    that Grid belongs to the Insight auth browser and can be safely retired.
    The persistent Chrome profile is not deleted.
    """
    _diag = {
        "attempted": True,
        "local_driver_globals_cleared": [],
        "grid_sessions_found": 0,
        "grid_sessions_deleted": 0,
        "queue_clear_attempted": True,
        "queue_clear_succeeded": False,
        "browser_reopen_attempted": True,
        "browser_reopen_http_status": None,
        "browser_reopen_ok": False,
    }

    _diag["local_driver_globals_cleared"] = _cl_clear_local_driver_globals()

    try:
        _grid = _cl_grid_graphql_sessions()
        _ids = _grid.get("session_ids") or []
        _diag["grid_sessions_found"] = len(_ids)
        for _sid in _ids:
            if _cl_grid_delete_session(_sid):
                _diag["grid_sessions_deleted"] += 1
    except Exception as _exc:
        _diag["grid_inspection_error"] = type(_exc).__name__

    _diag["queue_clear_succeeded"] = _cl_grid_clear_queue()

    _opened = _cl_open_auth_browser_via_local_api()
    _diag["browser_reopen_http_status"] = _opened.get("http_status")
    _diag["browser_reopen_ok"] = bool(_opened.get("ok"))

    # Never return credentials, cookies, token values, or raw Selenium session IDs.
    return _diag


async def _cl_auth_context_with_recovery():
    """
    Get Insight auth. A stale/missing browser session gets one automatic repair.
    A genuine login-required condition is not hidden and still returns 401.
    """
    try:
        _token, _ua = _cl_auth_context()
        return _token, _ua, {
            "attempted": False,
            "succeeded": False,
        }
    except _CLHTTPException as _exc:
        if getattr(_exc, "status_code", None) != 409:
            raise

    _diag = await _cl_asyncio.to_thread(_cl_recover_auth_session_sync)

    try:
        _token, _ua = _cl_auth_context()
        _diag["succeeded"] = True
        return _token, _ua, _diag
    except _CLHTTPException as _exc:
        # If the persistent profile is no longer logged in, surface a clean
        # login-required response rather than another Selenium stacktrace.
        if getattr(_exc, "status_code", None) == 401:
            raise
        raise _CLHTTPException(
            status_code=409,
            detail={
                "stage": "auth_recovery",
                "error": "automatic_recovery_failed",
                "message": "Automatic Insight browser recovery did not produce a usable authenticated session.",
                "recovery": _diag,
                "hint": "Open the noVNC browser and verify the Insight login state. A Docker restart should no longer be the first recovery step."
            }
        )

@app.post("/auth/insight/recover", tags=["auth"])
async def creative_loop_auth_recover():
    _diag = await _cl_asyncio.to_thread(_cl_recover_auth_session_sync)
    _logged_in = False
    try:
        _token, _ua = _cl_auth_context()
        _logged_in = bool(_token)
    except Exception:
        _logged_in = False
    return {
        "ok": bool(_diag.get("browser_reopen_ok")),
        "version": "0.18.8.1",
        "browser_session_active": _cl_find_active_webdriver() is not None,
        "logged_in": _logged_in,
        "recovery": _diag,
        "security_note": "No token, cookie, credential, raw Selenium session id, or Grid URL is returned."
    }

def _cl_build_insight_payload(_req: _CLSearchRequest, _page_index: int):
    # Field mapping verified from the user's captured Insight requests.
    return {
        "keyWord": _req.keyword,
        "keyWordType": "0,1,2,3,4,6,8",
        "keyWordList": [],
        "keyWordListType": True,
        "isNew": False,
        "creativeList": [],
        "appealTypeList": [],
        "interactionList": [],
        "languages": [],
        "productIds": [],
        "productOption": {
            "productType": [],
            "selling": [],
            "monetization": [],
            "payType": [],
            "companyLocation": [],
            "campaignList": []
        },
        "baseOption": {
            "permission": False,
            "putOverseaInland": None,
            "tradeLevel1": [],
            "tradeLevel2": [],
            "tradeLevel3": list(_req.industry_ids),
            "subjectType": [],
            "countryLevel2": [],
            "adfactionIds": [],
            "mediaIds": list(_req.media_ids),
            "device": [],
            "topicType": [],
            "dayMode": "DY",
            "productModel": [],
            "startTime": _req.start_date,
            "endTime": _req.end_date,
            "compareEndDate": "",
            "compareStartDate": "",
            "pageIndex": _page_index,
            "pageSize": _req.page_size,
            "sortField": "15",
            "sortRule": "desc",
            "gptSearch": False,
            "globalSearch": True
        },
        "classIds": [],
        "seelTargets": [],
        "webTools": [],
        "demoadFormats": [],
        "adMediaType": list(_req.ad_media_type_ids),
        "materialRemovalRepeat": False
    }

def _cl_post_json(_url: str, _payload: dict, _headers: dict, _timeout: int = 40):
    _body = _cl_json.dumps(_payload, ensure_ascii=False).encode("utf-8")
    _request = _cl_urlrequest.Request(
        _url,
        data=_body,
        headers=_headers,
        method="POST"
    )
    try:
        with _cl_urlrequest.urlopen(_request, timeout=_timeout) as _response:
            _raw = _response.read().decode("utf-8", errors="replace")
            return _response.status, _cl_json.loads(_raw)
    except _cl_urlerror.HTTPError as _exc:
        try:
            _detail = _exc.read().decode("utf-8", errors="replace")
        except Exception:
            _detail = str(_exc)
        raise _CLHTTPException(
            status_code=502,
            detail=f"Insight HTTP {_exc.code}: {_detail[:1000]}"
        )
    except Exception as _exc:
        raise _CLHTTPException(
            status_code=502,
            detail=f"Insight request failed: {_exc}"
        )

def _cl_extract_package_name(_row: dict):
    _urls = _row.get("nonLocalDemoad") or []
    for _url in _urls:
        if not isinstance(_url, str):
            continue
        _m = _cl_re.search(
            r"(?:play\.google\.com/store/apps/details\?[^#]*\bid=)([A-Za-z0-9._-]+)",
            _url
        )
        if _m:
            return _m.group(1)
    return None

def _cl_material_type(_row: dict):
    if _row.get("videoUrl"):
        return "video"
    _image_urls = _row.get("imageUrl") or []
    if any(isinstance(_x, str) and _x.strip() for _x in _image_urls):
        return "image"
    if _row.get("converUrl"):
        return "image"
    return "unknown"


_CL_ALLOWED_MATERIAL_TYPES = {"video", "image", "unknown"}

def _cl_requested_material_types(_req: _CLSearchRequest):
    _requested = []
    for _value in (_req.material_types or []):
        _value = str(_value or "").strip().lower()
        if not _value:
            continue
        if _value not in _CL_ALLOWED_MATERIAL_TYPES:
            raise _CLHTTPException(
                status_code=422,
                detail={
                    "message": "Unsupported material type.",
                    "value": _value,
                    "allowed": sorted(_CL_ALLOWED_MATERIAL_TYPES),
                }
            )
        if _value not in _requested:
            _requested.append(_value)

    # Empty list means no local material-type filter.
    return _requested

def _cl_media_filename(_row: dict):
    _url = _row.get("videoUrl") or ""
    if _url:
        _base = _url.split("?", 1)[0].rstrip("/")
        if "/" in _base:
            return _base.rsplit("/", 1)[-1]
    return None


_CL_RESULT_TITLE_MAX_CHARS = 600

def _cl_normalize_search_row(_row: dict, _rank: int, _start_date: str, _end_date: str):
    _search_identity = _extract_detail_identity(_row, response_url="insight_search_payload")
    _item_id = str(_row.get("id") or "")
    _title_raw = _row.get("title")
    if _title_raw is None:
        _title_text = None
        _title_length = 0
        _title_truncated = False
    else:
        _title_full = str(_title_raw)
        _title_length = len(_title_full)
        _title_truncated = _title_length > _CL_RESULT_TITLE_MAX_CHARS
        _title_text = (
            _title_full[:_CL_RESULT_TITLE_MAX_CHARS] + "…"
            if _title_truncated
            else _title_full
        )

    return {
        "rank": _rank,
        "id": _item_id,
        "title": _title_text,
        "title_truncated": _title_truncated,
        "title_length": _title_length,
        "product_name": _search_identity.get("product_name"),
        "app_title": _search_identity.get("app_title"),
        "package_name": (
            _search_identity.get("package_name")
            or _cl_extract_package_name(_row)
        ),
        "material_type": _cl_material_type(_row),
        "impression": _row.get("impression"),
        "show_cnt": _row.get("showCnt"),
        "first_time": _row.get("firstTime"),
        "global_first_time": _row.get("globalFirstTime"),
        "global_last_time": _row.get("globalLastTime"),
        "width": _row.get("width"),
        "height": _row.get("height"),
        "duration_seconds": _row.get("videoTimeSpan"),
        "video_url_present": bool(_row.get("videoUrl")),
        "source_filename": _cl_media_filename(_row),
        "detail_url": (
            f"{_CL_DETAIL_URL_BASE}/{_item_id}?dayMode={_start_date}~{_end_date}"
            if _item_id else None
        ),
    }


def _cl_impression_sort_diagnostic(_rows: list):
    _vals = []
    for _row in _rows:
        _v = _row.get("impression")
        if isinstance(_v, (int, float)):
            _vals.append(_v)

    if len(_vals) < 2:
        return {
            "verified": None,
            "status": "insufficient_numeric_values",
            "numeric_count": len(_vals),
            "distinct_count": len(set(_vals)),
            "min": min(_vals) if _vals else None,
            "max": max(_vals) if _vals else None,
        }

    _distinct = len(set(_vals))
    if _distinct == 1:
        return {
            "verified": None,
            "status": "all_equal",
            "numeric_count": len(_vals),
            "distinct_count": 1,
            "min": _vals[0],
            "max": _vals[0],
        }

    _descending = all(
        _vals[_i] >= _vals[_i + 1]
        for _i in range(len(_vals) - 1)
    )
    return {
        "verified": bool(_descending),
        "status": "descending" if _descending else "not_descending",
        "numeric_count": len(_vals),
        "distinct_count": _distinct,
        "min": min(_vals),
        "max": max(_vals),
    }


def _cl_is_impression_desc(_rows: list):
    # Backward-compatible helper used by any older code.
    return _cl_impression_sort_diagnostic(_rows)["verified"]

def _cl_call_existing_crawl_items(_items: list, _req: _CLSearchRequest):
    # Reuse the already-tested v0.6.x pipeline instead of duplicating
    # download / SHA256 / product mapping / Whisper / folder logic.
    _payload = {
        "items": _items,
        "wait_ms": _req.wait_ms,
        "download_cover": False,
        "auto_detect_language": _req.auto_detect_language,
    }
    _headers = {"Content-Type": "application/json"}
    return _cl_post_json(
        "http://127.0.0.1:8000/crawl/items",
        _payload,
        _headers,
        _timeout=max(60, 30 + len(_items) * 15)
    )

@app.post("/crawl/search", tags=["crawl"])
async def creative_loop_crawl_search(_req: _CLSearchRequest):
    """
    Search Insight using the authenticated browser token.

    v0.18.8.1 behavior:
    - Results remain ordered by Insight estimated-impression descending.
    - material_types is applied BEFORE counting toward top_n.
    - If a page contains excluded material types, pagination continues until
      top_n matching results are collected or Insight is exhausted.
    - download=true currently supports video-only selections, so image results
      cannot accidentally enter the Whisper/video archive pipeline.
    """
    if _req.start_date > _req.end_date:
        raise _CLHTTPException(
            status_code=422,
            detail="start_date must be <= end_date"
        )

    _material_types = _cl_requested_material_types(_req)

    if _req.download:
        _unsupported_for_download = [
            _t for _t in _material_types if _t != "video"
        ]
        if _unsupported_for_download:
            raise _CLHTTPException(
                status_code=422,
                detail={
                    "message": "download=true currently supports video material only.",
                    "requested_material_types": _material_types,
                    "unsupported_for_download": _unsupported_for_download,
                    "hint": "Use material_types=['video'] for downloading. Image download/archive will be added as a separate pipeline.",
                }
            )

    _token, _ua, _auth_recovery = await _cl_auth_context_with_recovery()
    _headers = {
        "Authorization": _token,
        "Content-Type": "application/json;charset=UTF-8",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://data.insightrackr.com",
        "Referer": "https://data.insightrackr.com/",
        "User-Agent": _ua,
    }

    _selected_raw = []
    _seen_ids = set()
    _page_index = 1
    _pages_fetched = 0
    _api_page_sizes = []

    _scanned_unique_count = 0
    _filtered_out_count = 0
    _material_type_counts = {
        "video": 0,
        "image": 0,
        "unknown": 0,
    }

    while len(_selected_raw) < _req.top_n:
        _payload = _cl_build_insight_payload(_req, _page_index)
        _http_status, _json = await _cl_asyncio.to_thread(
            _cl_post_json,
            _CL_INSIGHT_SEARCH_URL,
            _payload,
            _headers,
            40,
        )

        if not isinstance(_json, dict):
            raise _CLHTTPException(
                status_code=502,
                detail="Insight returned a non-object JSON response."
            )
        if _json.get("code") != 200:
            raise _CLHTTPException(
                status_code=502,
                detail={
                    "message": "Insight search returned a non-200 business code.",
                    "code": _json.get("code"),
                    "msg": _json.get("message") or _json.get("msg"),
                }
            )

        _data = _json.get("data") or {}
        _rows = _data.get("list") or []
        if not isinstance(_rows, list):
            raise _CLHTTPException(
                status_code=502,
                detail="Insight data.list is not an array."
            )

        _pages_fetched += 1
        _api_page_sizes.append(len(_rows))

        if not _rows:
            break

        for _row in _rows:
            if not isinstance(_row, dict):
                continue

            _id = str(_row.get("id") or "")
            if _id and _id in _seen_ids:
                continue
            if _id:
                _seen_ids.add(_id)

            _scanned_unique_count += 1
            _row_material_type = _cl_material_type(_row)
            if _row_material_type not in _material_type_counts:
                _row_material_type = "unknown"
            _material_type_counts[_row_material_type] += 1

            if _material_types and _row_material_type not in _material_types:
                _filtered_out_count += 1
                continue

            _selected_raw.append(_row)
            if len(_selected_raw) >= _req.top_n:
                break

        # A short page means Insight has no next full page to rely on.
        if len(_rows) < _req.page_size:
            break

        if len(_selected_raw) >= _req.top_n:
            break

        _page_index += 1
        if _page_index > 100:
            break

    _selected_raw = _selected_raw[:_req.top_n]
    _normalized = [
        _cl_normalize_search_row(
            _row,
            _rank=_i + 1,
            _start_date=_req.start_date,
            _end_date=_req.end_date,
        )
        for _i, _row in enumerate(_selected_raw)
    ]

    _result = {
        "ok": True,
        "mode": "search",
        "keyword": _req.keyword,
        "filters": {
            "start_date": _req.start_date,
            "end_date": _req.end_date,
            "media_ids": _req.media_ids,
            "ad_media_type_ids": _req.ad_media_type_ids,
            "industry_ids": _req.industry_ids,
            "material_types": _material_types,
        },
        "sort": {
            "field": "15",
            "rule": "desc",
            "impression_desc_verified": _cl_impression_sort_diagnostic(_selected_raw)["verified"],
            "diagnostic": _cl_impression_sort_diagnostic(_selected_raw),
        },
        "auth_recovery": _auth_recovery,
        "requested_top_n": _req.top_n,
        "selected_count": len(_normalized),
        "selection_complete": len(_normalized) >= _req.top_n,
        "pages_fetched": _pages_fetched,
        "api_page_sizes": _api_page_sizes,
        "scan": {
            "unique_results_scanned": _scanned_unique_count,
            "filtered_out_by_material_type": _filtered_out_count,
            "material_type_counts_seen": _material_type_counts,
        },
        "download_requested": _req.download,
        "results": _normalized,
        "security_note": "Insight authentication was used internally; no token/cookie value is returned.",
    }

    if not _req.download:
        return _result

    _crawl_items = []
    for _row, _norm in zip(_selected_raw, _normalized):
        if not _norm.get("detail_url"):
            continue

        # Defensive check even though the request-level validation above already
        # prevents non-video download selections.
        if _norm.get("material_type") != "video":
            continue

        _search_identity = _extract_detail_identity(
            _row,
            response_url="insight_search_payload",
        )

        _crawl_items.append({
            "url": _norm["detail_url"],
            "product_name": _search_identity.get("product_name"),
            "app_title": _search_identity.get("app_title"),
            "package_name": (
                _search_identity.get("package_name")
                or _norm.get("package_name")
            ),
            "countries": (
                _search_identity.get("countries")
                or _row.get("countryCodes")
                or []
            ),
            "language": None,
            "material_type": "video",
            "source_filename": _norm.get("source_filename"),
            "source_media_url": (
                _row.get("videoUrl")
                if isinstance(_row.get("videoUrl"), str)
                else None
            ),
            "impression": _norm.get("impression"),
            "search_rank": _norm.get("rank"),
        })

    if not _crawl_items:
        _result["crawl_items"] = {
            "ok": False,
            "reason": "No selected video results contained a usable detail id."
        }
        return _result

    _crawl_http_status, _crawl_json = await _cl_asyncio.to_thread(
        _cl_call_existing_crawl_items,
        _crawl_items,
        _req,
    )
    _result["crawl_items_http_status"] = _crawl_http_status
    _result["crawl_items"] = _crawl_json
    return _result

@app.get("/crawl/search/mapping", tags=["crawl"])
def creative_loop_search_mapping():
    return {
        "ok": True,
        "version": "0.18.8.1",
        "endpoint": _CL_INSIGHT_SEARCH_URL,
        "verified_mapping": {
            "keyword": "keyWord",
            "start_date": "baseOption.startTime",
            "end_date": "baseOption.endTime",
            "traffic_channel_ids": "baseOption.mediaIds",
            "ad_type_ids": "adMediaType",
            "industry_ids": "baseOption.tradeLevel3",
            "page_index": "baseOption.pageIndex",
            "page_size": "baseOption.pageSize",
            "impression_sort_field": "15",
            "sort_rule": "desc",
            "video_url": "data.list[].videoUrl",
            "impression": "data.list[].impression",
            "material_id": "data.list[].id",
            "title": "data.list[].title",
        },
        "local_selection": {
            "material_types_field": "material_types",
            "default": ["video"],
            "allowed": ["video", "image", "unknown"],
            "empty_list_means": "no material-type filter",
            "top_n_semantics": "Count matching material types only; continue pagination until top_n matches or Insight is exhausted.",
            "download_true_currently_supports": ["video"],
        },
        "notes": [
            "download=false previews search results only.",
            "download=true reuses the existing /crawl/items pipeline.",
            "Material type filtering happens before top_n is counted.",
            "Stale/missing Selenium auth sessions get one automatic Grid cleanup + browser reopen attempt.",
            "Very long result titles are truncated in API responses; the raw Insight title remains available internally for the existing crawl pipeline.",
            "If all returned impression values are equal, impression_desc_verified is null with diagnostic.status=all_equal.",
            "totalSize is not relied upon; pagination stops on empty/short page, once top_n matching results are reached, or at the 100-page safety limit."
        ],
    }

# === CREATIVE_LOOP_V0_8_SEARCH_END ===





# === CREATIVE_LOOP_V0_8_1_TIMEOUT_FIX_BEGIN ===
# v0.18.8.1:
# - Distinguish Insight-search failures from internal /crawl/items failures.
# - Give the existing download/crawl pipeline substantially more time.
# - Keep tokens/cookies internal.

import socket as _cl_socket
import time as _cl_time

def _cl_post_json(_url: str, _payload: dict, _headers: dict, _timeout: int = 40):
    _stage = "crawl_items" if "/crawl/items" in _url else "insight_search"
    _body = _cl_json.dumps(_payload, ensure_ascii=False).encode("utf-8")
    _request = _cl_urlrequest.Request(
        _url,
        data=_body,
        headers=_headers,
        method="POST"
    )
    _started = _cl_time.monotonic()

    try:
        with _cl_urlrequest.urlopen(_request, timeout=_timeout) as _response:
            _raw = _response.read().decode("utf-8", errors="replace")
            return _response.status, _cl_json.loads(_raw)

    except (_cl_socket.timeout, TimeoutError) as _exc:
        _elapsed = round(_cl_time.monotonic() - _started, 2)
        raise _CLHTTPException(
            status_code=504,
            detail={
                "stage": _stage,
                "error": "timed_out",
                "timeout_seconds": _timeout,
                "elapsed_seconds": _elapsed,
                "message": (
                    "Insight search request timed out."
                    if _stage == "insight_search"
                    else "The existing /crawl/items download pipeline timed out."
                ),
            }
        )

    except _cl_urlerror.HTTPError as _exc:
        try:
            _detail = _exc.read().decode("utf-8", errors="replace")
        except Exception:
            _detail = str(_exc)
        raise _CLHTTPException(
            status_code=502,
            detail={
                "stage": _stage,
                "error": "http_error",
                "http_status": _exc.code,
                "message": _detail[:1000],
            }
        )

    except Exception as _exc:
        _elapsed = round(_cl_time.monotonic() - _started, 2)
        raise _CLHTTPException(
            status_code=502,
            detail={
                "stage": _stage,
                "error": type(_exc).__name__,
                "elapsed_seconds": _elapsed,
                "message": str(_exc),
            }
        )

def _cl_call_existing_crawl_items(_items: list, _req: _CLSearchRequest):
    """
    Reuse the existing /crawl/items pipeline.

    v0.8 used a ~60s timeout. That is too short because /crawl/items can include
    detail-page probing, media transfer, hashing, Whisper inference and file moves.
    Give one item at least 300s, then scale moderately for batches.
    """
    _payload = {
        "items": _items,
        "wait_ms": _req.wait_ms,
        "download_cover": False,
        "auto_detect_language": _req.auto_detect_language,
    }
    _headers = {"Content-Type": "application/json"}

    _timeout = min(
        1800,
        max(300, 180 + len(_items) * 120)
    )

    return _cl_post_json(
        "http://127.0.0.1:8000/crawl/items",
        _payload,
        _headers,
        _timeout=_timeout,
    )

@app.get("/crawl/search/runtime", tags=["crawl"])
def creative_loop_search_runtime():
    return {
        "ok": True,
        "version": "0.18.8.1",
        "timeouts": {
            "insight_search_seconds": 40,
            "crawl_items_min_seconds": 300,
            "crawl_items_formula": "min(1800, max(300, 180 + item_count * 120))",
        },
        "diagnostics": {
            "timeout_errors_include_stage": True,
            "stages": ["insight_search", "crawl_items"],
        },
    }

# === CREATIVE_LOOP_V0_8_1_TIMEOUT_FIX_END ===



# === CREATIVE_LOOP_V0_8_2_BROWSER_ROUTING_FIX_BEGIN ===
# Keep the dedicated authenticated Insight browser on Selenium,
# but force legacy Playwright tasks (/crawl/items, probes, diagnostics)
# to launch Chromium locally inside creative-loop-api.
#
# Why:
# Playwright automatically uses Selenium Grid whenever SELENIUM_REMOTE_URL
# exists in the environment. That made /crawl/items compete for the same
# single Selenium slot already occupied by the persistent auth browser.

import os as _cl_os
import asyncio as _cl_asyncio_v082

# Preserve the real Playwright factory imported by the existing application.
_cl_real_async_playwright = async_playwright
_cl_playwright_spawn_lock = _cl_asyncio_v082.Lock()

class _CLLocalPlaywrightContext:
    def __init__(self):
        self._inner = _cl_real_async_playwright()

    async def _start_without_selenium_env(self, starter):
        # SELENIUM_REMOTE_URL only needs to be absent while Playwright's
        # driver process is spawned. Restore it immediately afterwards so
        # the application's Selenium auth configuration remains untouched.
        async with _cl_playwright_spawn_lock:
            _old = _cl_os.environ.pop("SELENIUM_REMOTE_URL", None)
            try:
                return await starter()
            finally:
                if _old is not None:
                    _cl_os.environ["SELENIUM_REMOTE_URL"] = _old

    async def __aenter__(self):
        return await self._start_without_selenium_env(self._inner.__aenter__)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._inner.__aexit__(exc_type, exc_val, exc_tb)

    async def start(self):
        return await self._start_without_selenium_env(self._inner.start)

# Existing route functions resolve this global name at call time.
# From v0.18.8.1 onward, Playwright-based routes therefore launch locally.
def async_playwright():
    return _CLLocalPlaywrightContext()

@app.get("/diagnostics/browser-routing", tags=["diagnostics"])
def creative_loop_browser_routing():
    return {
        "ok": True,
        "version": "0.18.8.1",
        "routing": {
            "authenticated_insight_browser": "Selenium -> http://insight-browser:4444",
            "legacy_playwright_tasks": "local Chromium inside creative-loop-api",
            "playwright_forced_local": True,
        },
        "environment": {
            "selenium_remote_url_configured": bool(_cl_os.environ.get("SELENIUM_REMOTE_URL")),
        },
        "security_note": "The Selenium URL value and all authentication values are intentionally not returned.",
    }

# === CREATIVE_LOOP_V0_8_2_BROWSER_ROUTING_FIX_END ===





# === CREATIVE_LOOP_V0_8_4_1_AUTH_RECOVERY_HOTFIX ===

# === CREATIVE_LOOP_V0_8_5_FILTER_DICTIONARY_PROBE_BEGIN ===
# Read Insight's own filter dictionary and locate human-readable context for
# machine IDs. This endpoint never returns authentication material.

import time as _cl_v085_time

_CL_FILTER_OPTIONS_URL = "https://data.insightrackr.com/cas/api/home/common/screen/query/type"

class _CLFilterLookupRequest(_CLBaseModel):
    ids: _CLList[str] = _CLField(
        default_factory=lambda: ["1006", "1076682150", "6040205"]
    )


def _cl_v085_compact(_value, _depth=0):
    if _depth > 4:
        return "<max-depth>"
    if isinstance(_value, dict):
        _out = {}
        for _i, (_k, _v) in enumerate(_value.items()):
            if _i >= 40:
                _out["<truncated_fields>"] = len(_value) - 40
                break
            _out[str(_k)] = _cl_v085_compact(_v, _depth + 1)
        return _out
    if isinstance(_value, list):
        _items = [_cl_v085_compact(_v, _depth + 1) for _v in _value[:20]]
        if len(_value) > 20:
            _items.append({"<truncated_items>": len(_value) - 20})
        return _items
    if isinstance(_value, str):
        return _value if len(_value) <= 300 else _value[:300] + "…"
    if isinstance(_value, (int, float, bool)) or _value is None:
        return _value
    return str(_value)[:300]


def _cl_v085_find_contexts(_root, _target_ids):
    _targets = {str(_x) for _x in _target_ids if str(_x).strip()}
    _matches = {str(_x): [] for _x in _targets}

    def _walk(_node, _path="$"):
        if isinstance(_node, dict):
            # If any direct scalar field equals a target ID, return the entire
            # surrounding option object. Human-readable labels are typically
            # siblings of id/value/code fields.
            _hit_ids = set()
            for _k, _v in _node.items():
                if isinstance(_v, (str, int, float)) and str(_v) in _targets:
                    _hit_ids.add(str(_v))

            for _tid in _hit_ids:
                if len(_matches[_tid]) < 20:
                    _matches[_tid].append({
                        "path": _path,
                        "context": _cl_v085_compact(_node),
                    })

            for _k, _v in _node.items():
                _walk(_v, f"{_path}.{_k}")

        elif isinstance(_node, list):
            for _i, _v in enumerate(_node):
                _walk(_v, f"{_path}[{_i}]")

    _walk(_root)
    return _matches


def _cl_v085_request_options(_method: str, _token: str, _ua: str, _timeout: int = 25):
    _headers = {
        "Authorization": _token,
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://data.insightrackr.com",
        "Referer": "https://data.insightrackr.com/",
        "User-Agent": _ua,
    }

    _data = None
    if _method == "POST":
        _data = b"{}"
        _headers["Content-Type"] = "application/json;charset=UTF-8"

    _req = _cl_urlrequest.Request(
        _CL_FILTER_OPTIONS_URL,
        data=_data,
        headers=_headers,
        method=_method,
    )

    _started = _cl_v085_time.monotonic()
    try:
        with _cl_urlrequest.urlopen(_req, timeout=_timeout) as _resp:
            _raw = _resp.read().decode("utf-8", errors="replace")
            try:
                _json = _cl_json.loads(_raw)
            except Exception:
                _json = None
            return {
                "method": _method,
                "ok": True,
                "http_status": int(_resp.status),
                "elapsed_seconds": round(_cl_v085_time.monotonic() - _started, 3),
                "json": _json,
                "raw_preview": None if _json is not None else _raw[:300],
            }
    except _cl_urlerror.HTTPError as _exc:
        try:
            _raw = _exc.read().decode("utf-8", errors="replace")
        except Exception:
            _raw = ""
        return {
            "method": _method,
            "ok": False,
            "http_status": int(_exc.code),
            "elapsed_seconds": round(_cl_v085_time.monotonic() - _started, 3),
            "error": "http_error",
            "message": _raw[:300],
            "json": None,
        }
    except Exception as _exc:
        return {
            "method": _method,
            "ok": False,
            "http_status": None,
            "elapsed_seconds": round(_cl_v085_time.monotonic() - _started, 3),
            "error": type(_exc).__name__,
            "message": str(_exc)[:300],
            "json": None,
        }


@app.post("/crawl/search/options/lookup", tags=["crawl"])
async def creative_loop_search_options_lookup(_req: _CLFilterLookupRequest):
    _ids = []
    for _value in (_req.ids or []):
        _value = str(_value or "").strip()
        if _value and _value not in _ids:
            _ids.append(_value)

    if not _ids:
        raise _CLHTTPException(
            status_code=422,
            detail="ids must contain at least one non-empty ID."
        )
    if len(_ids) > 50:
        raise _CLHTTPException(
            status_code=422,
            detail="At most 50 IDs may be looked up at once."
        )

    _token, _ua, _auth_recovery = await _cl_auth_context_with_recovery()

    _attempts = []
    _best = None
    _best_match_count = -1

    # We captured this endpoint from Insight XHR traffic but the earlier
    # diagnostic did not preserve its HTTP method. Probe the two safe,
    # read-oriented forms and choose the response containing the most target IDs.
    for _method in ("GET", "POST"):
        _attempt = await _cl_asyncio.to_thread(
            _cl_v085_request_options,
            _method,
            _token,
            _ua,
        )

        _json = _attempt.pop("json", None)
        _summary = {
            "method": _attempt.get("method"),
            "ok": _attempt.get("ok"),
            "http_status": _attempt.get("http_status"),
            "elapsed_seconds": _attempt.get("elapsed_seconds"),
            "error": _attempt.get("error"),
            "message": _attempt.get("message"),
            "business_code": _json.get("code") if isinstance(_json, dict) else None,
            "top_level_keys": list(_json.keys())[:30] if isinstance(_json, dict) else None,
        }

        if _json is not None:
            _matches = _cl_v085_find_contexts(_json, _ids)
            _count = sum(len(_v) for _v in _matches.values())
            _summary["target_match_count"] = _count

            if _count > _best_match_count:
                _best_match_count = _count
                _best = {
                    "method": _method,
                    "matches": _matches,
                    "response_shape": {
                        "type": type(_json).__name__,
                        "top_level_keys": list(_json.keys())[:30] if isinstance(_json, dict) else None,
                        "business_code": _json.get("code") if isinstance(_json, dict) else None,
                    },
                }

        _attempts.append(_summary)

    if _best is None:
        raise _CLHTTPException(
            status_code=502,
            detail={
                "stage": "filter_dictionary_probe",
                "error": "no_json_response",
                "attempts": _attempts,
            }
        )

    return {
        "ok": True,
        "version": "0.18.8.1",
        "endpoint": "/cas/api/home/common/screen/query/type",
        "requested_ids": _ids,
        "auth_recovery": _auth_recovery,
        "attempts": _attempts,
        "selected_method": _best["method"],
        "matches": _best["matches"],
        "response_shape": _best["response_shape"],
        "security_note": "Only filter-option context is returned. Authentication tokens, cookies and raw Selenium session IDs are never returned.",
    }

# === CREATIVE_LOOP_V0_8_5_FILTER_DICTIONARY_PROBE_END ===



# === CREATIVE_LOOP_V0_8_5_1_CAPTURE_FILTER_REQUEST_BEGIN ===
# Capture the real Insight /screen/query/type request from the authenticated
# Chrome instead of guessing its method/body.

import time as _cl_v0851_time
from typing import Optional as _CLV0851Optional

class _CLFilterCaptureRequest(_CLBaseModel):
    wait_seconds: float = 6.0
    target_url: _CLV0851Optional[str] = None


_CL_FILTER_CAPTURE_NEEDLE = "/cas/api/home/common/screen/query/type"


_CL_FILTER_CAPTURE_SCRIPT = r"""
(() => {
  if (window.__creativeLoopFilterCaptureInstalled) return;
  window.__creativeLoopFilterCaptureInstalled = true;
  window.__creativeLoopNetlog = [];

  const push = (entry) => {
    try {
      entry.ts = Date.now();
      window.__creativeLoopNetlog.push(entry);
      if (window.__creativeLoopNetlog.length > 500) {
        window.__creativeLoopNetlog = window.__creativeLoopNetlog.slice(-500);
      }
    } catch (_) {}
  };

  const originalFetch = window.fetch;
  window.fetch = function(input, init) {
    try {
      const url = typeof input === 'string' ? input : (input && input.url) || '';
      const method =
        (init && init.method) ||
        (input && input.method) ||
        'GET';
      const body = init && Object.prototype.hasOwnProperty.call(init, 'body')
        ? init.body
        : null;

      const idx = window.__creativeLoopNetlog.length;
      push({
        transport: 'fetch',
        url: String(url || ''),
        method: String(method || 'GET').toUpperCase(),
        body: typeof body === 'string' ? body : (body == null ? null : String(body)),
        status: null
      });

      const p = originalFetch.apply(this, arguments);
      Promise.resolve(p).then(
        (resp) => {
          try {
            if (window.__creativeLoopNetlog[idx]) {
              window.__creativeLoopNetlog[idx].status = resp.status;
            }
          } catch (_) {}
        },
        () => {}
      );
      return p;
    } catch (_) {
      return originalFetch.apply(this, arguments);
    }
  };

  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;

  XMLHttpRequest.prototype.open = function(method, url) {
    try {
      this.__clCaptureMeta = {
        transport: 'xhr',
        url: String(url || ''),
        method: String(method || 'GET').toUpperCase()
      };
    } catch (_) {}
    return origOpen.apply(this, arguments);
  };

  XMLHttpRequest.prototype.send = function(body) {
    try {
      const meta = this.__clCaptureMeta || {};
      const entry = {
        transport: 'xhr',
        url: meta.url || '',
        method: meta.method || 'GET',
        body: typeof body === 'string' ? body : (body == null ? null : String(body)),
        status: null
      };
      const idx = window.__creativeLoopNetlog.length;
      push(entry);
      this.addEventListener('loadend', () => {
        try {
          if (window.__creativeLoopNetlog[idx]) {
            window.__creativeLoopNetlog[idx].status = this.status;
          }
        } catch (_) {}
      });
    } catch (_) {}
    return origSend.apply(this, arguments);
  };
})();
"""


def _cl_v0851_safe_body(_body):
    """
    Return a compact request body without exposing credential-like fields.
    The captured hook does not record headers, but redact common sensitive
    JSON keys defensively in case Insight puts them in a body.
    """
    if _body is None:
        return None

    _text = str(_body)
    if len(_text) > 12000:
        _text = _text[:12000] + "…"

    try:
        _obj = _cl_json.loads(_text)
    except Exception:
        return _text

    _sensitive = {
        "authorization", "cookie", "token", "accesstoken", "access_token",
        "refresh_token", "refreshtoken", "password", "passwd", "secret"
    }

    def _redact(_v):
        if isinstance(_v, dict):
            _o = {}
            for _k, _x in _v.items():
                if str(_k).lower() in _sensitive:
                    _o[_k] = "<redacted>"
                else:
                    _o[_k] = _redact(_x)
            return _o
        if isinstance(_v, list):
            return [_redact(_x) for _x in _v]
        return _v

    return _redact(_obj)


def _cl_v0851_capture_sync(_driver, _target_url: str, _wait_seconds: float):
    _diag = {
        "cdp_injection": False,
        "fallback_injection": False,
        "navigated_url": _target_url,
        "wait_seconds": _wait_seconds,
    }

    # Best path: install before the next document is created so requests made
    # during initial app bootstrap are captured.
    try:
        _driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": _CL_FILTER_CAPTURE_SCRIPT},
        )
        _diag["cdp_injection"] = True
    except Exception as _exc:
        _diag["cdp_error"] = type(_exc).__name__

    # Navigate/reload after registering the script.
    try:
        _driver.get(_target_url)
    except Exception as _exc:
        return {
            "ok": False,
            "diag": _diag,
            "error": "navigation_failed",
            "message": str(_exc)[:1000],
            "calls": [],
        }

    # If CDP injection is unavailable, installing after navigation may miss
    # bootstrap traffic, but it still lets us capture requests triggered by
    # delayed app initialization.
    if not _diag["cdp_injection"]:
        try:
            _driver.execute_script(_CL_FILTER_CAPTURE_SCRIPT)
            _diag["fallback_injection"] = True
        except Exception as _exc:
            _diag["fallback_error"] = type(_exc).__name__

    _cl_v0851_time.sleep(max(1.0, min(float(_wait_seconds), 20.0)))

    try:
        _calls = _driver.execute_script(
            "return (window.__creativeLoopNetlog || []).filter("
            "x => String(x.url || '').includes(arguments[0]));",
            _CL_FILTER_CAPTURE_NEEDLE,
        ) or []
    except Exception as _exc:
        return {
            "ok": False,
            "diag": _diag,
            "error": "read_capture_failed",
            "message": str(_exc)[:1000],
            "calls": [],
        }

    _clean = []
    for _call in _calls[:50]:
        if not isinstance(_call, dict):
            continue
        _clean.append({
            "transport": _call.get("transport"),
            "method": _call.get("method"),
            "url": _call.get("url"),
            "body": _cl_v0851_safe_body(_call.get("body")),
            "status": _call.get("status"),
            "ts": _call.get("ts"),
        })

    return {
        "ok": True,
        "diag": _diag,
        "calls": _clean,
    }


@app.post("/crawl/search/options/capture", tags=["crawl"])
async def creative_loop_search_options_capture(_req: _CLFilterCaptureRequest):
    # Auth recovery is deliberately reused here so API-only rebuilds do not
    # require manual Selenium cleanup.
    _token, _ua, _auth_recovery = await _cl_auth_context_with_recovery()
    _driver = _cl_find_active_webdriver()

    if _driver is None:
        raise _CLHTTPException(
            status_code=409,
            detail={
                "stage": "filter_request_capture",
                "error": "no_usable_browser_after_auth",
            }
        )

    try:
        _current_url = str(_driver.current_url or "")
    except Exception:
        _current_url = ""

    _target = str(_req.target_url or _current_url or "https://data.insightrackr.com/").strip()
    if not _target.startswith("https://data.insightrackr.com/"):
        raise _CLHTTPException(
            status_code=422,
            detail="target_url must be an https://data.insightrackr.com/ URL."
        )

    _capture = await _cl_asyncio.to_thread(
        _cl_v0851_capture_sync,
        _driver,
        _target,
        _req.wait_seconds,
    )

    if not _capture.get("ok"):
        raise _CLHTTPException(
            status_code=502,
            detail={
                "stage": "filter_request_capture",
                "error": _capture.get("error"),
                "message": _capture.get("message"),
                "diagnostic": _capture.get("diag"),
            }
        )

    _calls = _capture.get("calls") or []

    return {
        "ok": True,
        "version": "0.18.8.1",
        "target_url": _target,
        "auth_recovery": _auth_recovery,
        "capture": _capture.get("diag"),
        "matching_call_count": len(_calls),
        "calls": _calls,
        "next_step": (
            "Use the captured method/query/body to replay Insight's filter dictionary endpoint."
            if _calls
            else "No matching request was observed on this page load. The endpoint may be triggered only on a specific Insight screen."
        ),
        "security_note": "Request headers are not captured. Credential-like JSON body fields are redacted.",
    }

# === CREATIVE_LOOP_V0_8_5_1_CAPTURE_FILTER_REQUEST_END ===



# === CREATIVE_LOOP_V0_8_5_2_REPLAY_FILTER_DICTIONARY_BEGIN ===
# Replay the real filter-dictionary requests captured from Insight:
# POST /cas/api/home/common/screen/query/type
# {"elementType": 1004}
# {"elementType": 2006}

class _CLFilterReplayRequest(_CLBaseModel):
    ids: _CLList[str] = _CLField(
        default_factory=lambda: ["1006", "1076682150", "6040205"]
    )
    element_types: _CLList[int] = _CLField(
        default_factory=lambda: [1004, 2006]
    )


def _cl_v0852_post_element_type(_element_type: int, _token: str, _ua: str, _timeout: int = 25):
    _payload = {"elementType": int(_element_type)}
    _headers = {
        "Authorization": _token,
        "Content-Type": "application/json;charset=UTF-8",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://data.insightrackr.com",
        "Referer": "https://data.insightrackr.com/",
        "User-Agent": _ua,
    }

    _body = _cl_json.dumps(_payload, ensure_ascii=False).encode("utf-8")
    _request = _cl_urlrequest.Request(
        _CL_FILTER_OPTIONS_URL,
        data=_body,
        headers=_headers,
        method="POST",
    )

    try:
        with _cl_urlrequest.urlopen(_request, timeout=_timeout) as _response:
            _raw = _response.read().decode("utf-8", errors="replace")
            try:
                _json = _cl_json.loads(_raw)
            except Exception:
                _json = None
            return {
                "ok": True,
                "http_status": int(_response.status),
                "json": _json,
                "raw_preview": None if _json is not None else _raw[:500],
            }

    except _cl_urlerror.HTTPError as _exc:
        try:
            _raw = _exc.read().decode("utf-8", errors="replace")
        except Exception:
            _raw = ""
        return {
            "ok": False,
            "http_status": int(_exc.code),
            "json": None,
            "error": "http_error",
            "message": _raw[:500],
        }

    except Exception as _exc:
        return {
            "ok": False,
            "http_status": None,
            "json": None,
            "error": type(_exc).__name__,
            "message": str(_exc)[:500],
        }


def _cl_v0852_preview_data(_json):
    """
    Return a compact structural preview. This is intentionally bounded so a
    large Insight option tree does not flood the API response.
    """
    if not isinstance(_json, dict):
        return _cl_v085_compact(_json)

    _data = _json.get("data")
    if _data is None:
        return {
            "top_level": _cl_v085_compact(_json),
        }

    if isinstance(_data, list):
        return {
            "data_type": "list",
            "data_length": len(_data),
            "sample": _cl_v085_compact(_data[:8]),
        }

    if isinstance(_data, dict):
        return {
            "data_type": "dict",
            "data_keys": list(_data.keys())[:50],
            "sample": _cl_v085_compact(_data),
        }

    return {
        "data_type": type(_data).__name__,
        "sample": _cl_v085_compact(_data),
    }


@app.post("/crawl/search/options/replay", tags=["crawl"])
async def creative_loop_search_options_replay(_req: _CLFilterReplayRequest):
    _ids = []
    for _value in (_req.ids or []):
        _value = str(_value or "").strip()
        if _value and _value not in _ids:
            _ids.append(_value)

    if not _ids:
        raise _CLHTTPException(
            status_code=422,
            detail="ids must contain at least one non-empty ID."
        )

    _element_types = []
    for _value in (_req.element_types or []):
        try:
            _value = int(_value)
        except Exception:
            continue
        if _value not in _element_types:
            _element_types.append(_value)

    if not _element_types:
        raise _CLHTTPException(
            status_code=422,
            detail="element_types must contain at least one integer."
        )
    if len(_element_types) > 20:
        raise _CLHTTPException(
            status_code=422,
            detail="At most 20 element_types may be replayed at once."
        )

    _token, _ua, _auth_recovery = await _cl_auth_context_with_recovery()

    _responses = []
    _merged = {str(_id): [] for _id in _ids}

    for _element_type in _element_types:
        _reply = await _cl_asyncio.to_thread(
            _cl_v0852_post_element_type,
            _element_type,
            _token,
            _ua,
        )

        _json = _reply.pop("json", None)
        _entry = {
            "element_type": _element_type,
            "ok": bool(_reply.get("ok")),
            "http_status": _reply.get("http_status"),
            "error": _reply.get("error"),
            "message": _reply.get("message"),
        }

        if isinstance(_json, dict):
            _entry["business_code"] = _json.get("code")
            _entry["message"] = _json.get("message") or _json.get("msg")
            _entry["response_preview"] = _cl_v0852_preview_data(_json)

            _matches = _cl_v085_find_contexts(_json, _ids)
            _entry["matches"] = _matches

            for _target_id, _items in _matches.items():
                for _item in _items:
                    _merged[_target_id].append({
                        "element_type": _element_type,
                        **_item,
                    })
        else:
            _entry["raw_preview"] = _reply.get("raw_preview")

        _responses.append(_entry)

    _found_ids = [
        _id for _id, _items in _merged.items()
        if _items
    ]
    _missing_ids = [
        _id for _id, _items in _merged.items()
        if not _items
    ]

    return {
        "ok": True,
        "version": "0.18.8.1",
        "endpoint": "/cas/api/home/common/screen/query/type",
        "request_method": "POST",
        "requested_ids": _ids,
        "element_types": _element_types,
        "auth_recovery": _auth_recovery,
        "found_ids": _found_ids,
        "missing_ids": _missing_ids,
        "matches": _merged,
        "responses": _responses,
        "next_step": (
            "Human-readable contexts were found for all requested IDs."
            if not _missing_ids
            else "Some IDs were not present in these captured element types. Use the response previews to identify these dictionaries, then capture additional elementType requests from the relevant Insight filter/search screen."
        ),
        "security_note": "Authentication is used internally only. No token, cookie, credential, request header, or raw Selenium session ID is returned.",
    }

# === CREATIVE_LOOP_V0_8_5_2_REPLAY_FILTER_DICTIONARY_END ===



# === CREATIVE_LOOP_V0_8_5_3_LIVE_FILTER_CAPTURE_BEGIN ===
# Persistent Selenium-side capture for Insight filter dictionary requests.
# Start capture, interact with the real Insight UI, then read captured calls.

_CL_V0853_CAPTURE_SCRIPT = r"""
(() => {
  if (window.__creativeLoopFilterCaptureInstalled) {
    window.__creativeLoopNetlog = [];
    return;
  }

  window.__creativeLoopFilterCaptureInstalled = true;
  window.__creativeLoopNetlog = [];

  const push = (entry) => {
    try {
      entry.ts = Date.now();
      window.__creativeLoopNetlog.push(entry);
      if (window.__creativeLoopNetlog.length > 1000) {
        window.__creativeLoopNetlog = window.__creativeLoopNetlog.slice(-1000);
      }
    } catch (_) {}
  };

  const originalFetch = window.fetch;
  window.fetch = function(input, init) {
    let idx = -1;
    try {
      const url = typeof input === 'string' ? input : (input && input.url) || '';
      const method =
        (init && init.method) ||
        (input && input.method) ||
        'GET';
      const body = init && Object.prototype.hasOwnProperty.call(init, 'body')
        ? init.body
        : null;

      idx = window.__creativeLoopNetlog.length;
      push({
        transport: 'fetch',
        url: String(url || ''),
        method: String(method || 'GET').toUpperCase(),
        body: typeof body === 'string' ? body : (body == null ? null : String(body)),
        status: null
      });
    } catch (_) {}

    const p = originalFetch.apply(this, arguments);
    if (idx >= 0) {
      Promise.resolve(p).then(
        (resp) => {
          try {
            if (window.__creativeLoopNetlog[idx]) {
              window.__creativeLoopNetlog[idx].status = resp.status;
            }
          } catch (_) {}
        },
        () => {}
      );
    }
    return p;
  };

  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;

  XMLHttpRequest.prototype.open = function(method, url) {
    try {
      this.__clCaptureMeta = {
        transport: 'xhr',
        url: String(url || ''),
        method: String(method || 'GET').toUpperCase()
      };
    } catch (_) {}
    return origOpen.apply(this, arguments);
  };

  XMLHttpRequest.prototype.send = function(body) {
    let idx = -1;
    try {
      const meta = this.__clCaptureMeta || {};
      idx = window.__creativeLoopNetlog.length;
      push({
        transport: 'xhr',
        url: meta.url || '',
        method: meta.method || 'GET',
        body: typeof body === 'string' ? body : (body == null ? null : String(body)),
        status: null
      });

      this.addEventListener('loadend', () => {
        try {
          if (idx >= 0 && window.__creativeLoopNetlog[idx]) {
            window.__creativeLoopNetlog[idx].status = this.status;
          }
        } catch (_) {}
      });
    } catch (_) {}

    return origSend.apply(this, arguments);
  };
})();
"""


def _cl_v0853_install_capture(_driver):
    _diag = {
        "cdp_registered": False,
        "current_document_injected": False,
    }

    # Register for all future navigations in this Chrome session.
    try:
        _driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": _CL_V0853_CAPTURE_SCRIPT},
        )
        _diag["cdp_registered"] = True
    except Exception as _exc:
        _diag["cdp_error"] = type(_exc).__name__

    # Also capture immediately in the currently loaded page.
    try:
        _driver.execute_script(_CL_V0853_CAPTURE_SCRIPT)
        _diag["current_document_injected"] = True
    except Exception as _exc:
        _diag["injection_error"] = type(_exc).__name__

    try:
        _diag["current_url"] = str(_driver.current_url or "")
    except Exception:
        _diag["current_url"] = None

    return _diag


def _cl_v0853_decode_body(_body):
    if _body is None:
        return None
    _text = str(_body)
    if len(_text) > 12000:
        _text = _text[:12000] + "…"
    try:
        return _cl_json.loads(_text)
    except Exception:
        return _text


def _cl_v0853_read_capture(_driver):
    try:
        _current_url = str(_driver.current_url or "")
    except Exception:
        _current_url = None

    try:
        _raw_calls = _driver.execute_script(
            "return (window.__creativeLoopNetlog || []).filter("
            "x => String(x.url || '').includes('/cas/api/home/common/screen/query/type'));"
        ) or []
    except Exception as _exc:
        return {
            "ok": False,
            "current_url": _current_url,
            "error": type(_exc).__name__,
            "message": str(_exc)[:1000],
            "calls": [],
        }

    _calls = []
    _element_types = []

    for _call in _raw_calls[-100:]:
        if not isinstance(_call, dict):
            continue

        _body = _cl_v0853_decode_body(_call.get("body"))
        _element_type = None
        if isinstance(_body, dict):
            try:
                if _body.get("elementType") is not None:
                    _element_type = int(_body.get("elementType"))
            except Exception:
                _element_type = _body.get("elementType")

        if _element_type is not None and _element_type not in _element_types:
            _element_types.append(_element_type)

        _calls.append({
            "transport": _call.get("transport"),
            "method": _call.get("method"),
            "url": _call.get("url"),
            "body": _body,
            "element_type": _element_type,
            "status": _call.get("status"),
            "ts": _call.get("ts"),
        })

    return {
        "ok": True,
        "current_url": _current_url,
        "calls": _calls,
        "element_types_seen": _element_types,
    }


@app.post("/crawl/search/options/live/start", tags=["crawl"])
async def creative_loop_search_options_live_start():
    _token, _ua, _auth_recovery = await _cl_auth_context_with_recovery()
    _driver = _cl_find_active_webdriver()
    if _driver is None:
        raise _CLHTTPException(
            status_code=409,
            detail={
                "stage": "filter_live_capture",
                "error": "no_usable_browser_after_auth",
            }
        )

    _diag = await _cl_asyncio.to_thread(_cl_v0853_install_capture, _driver)

    return {
        "ok": True,
        "version": "0.18.8.1",
        "capture_active": bool(
            _diag.get("cdp_registered") or _diag.get("current_document_injected")
        ),
        "auth_recovery": _auth_recovery,
        "capture": _diag,
        "instructions": [
            "Open the existing noVNC Insight browser.",
            "Navigate to the Creative/Material search screen.",
            "Open or change the Ad Type filter once.",
            "Do not close the Chrome window.",
            "Then call POST /crawl/search/options/live/read."
        ],
        "security_note": "Request headers, tokens, cookies and credentials are not captured."
    }


@app.post("/crawl/search/options/live/read", tags=["crawl"])
async def creative_loop_search_options_live_read():
    # Do not force a browser recreation here because that would erase the
    # in-page captured log. Only use the currently attached usable driver.
    _driver = _cl_find_active_webdriver()
    if _driver is None:
        raise _CLHTTPException(
            status_code=409,
            detail={
                "stage": "filter_live_capture",
                "error": "capture_browser_not_usable",
                "message": "The browser used for live capture is no longer usable. Start the capture again."
            }
        )

    _result = await _cl_asyncio.to_thread(_cl_v0853_read_capture, _driver)
    if not _result.get("ok"):
        raise _CLHTTPException(
            status_code=502,
            detail={
                "stage": "filter_live_capture",
                "error": _result.get("error"),
                "message": _result.get("message"),
            }
        )

    return {
        "ok": True,
        "version": "0.18.8.1",
        "current_url": _result.get("current_url"),
        "matching_call_count": len(_result.get("calls") or []),
        "element_types_seen": _result.get("element_types_seen") or [],
        "calls": _result.get("calls") or [],
        "known_element_types": {
            "1004": "Industry dictionary (confirmed)",
            "2006": "Media/traffic-channel dictionary (confirmed)"
        },
        "next_step": (
            "Replay any newly observed elementType values and search them for ad_media_type_id 1076682150."
            if any(str(x) not in {"1004", "2006"} for x in (_result.get("element_types_seen") or []))
            else "No new elementType was observed yet. On the Insight search screen, open/change the Ad Type filter and read again."
        ),
        "security_note": "Request headers, tokens, cookies and credentials are not captured."
    }

# === CREATIVE_LOOP_V0_8_5_3_LIVE_FILTER_CAPTURE_END ===



# === CREATIVE_LOOP_V0_8_6_HUMAN_READABLE_FILTERS_BEGIN ===
# Human-readable filter facade for Floatboat / operators.
#
# Confirmed Insight dictionaries:
#   elementType 2006 -> media / traffic channel
#   elementType 5001 -> ad type
#   elementType 1004 -> industry
#
# Names are resolved dynamically from Insight instead of being hard-coded.

import re as _cl_v086_re
import time as _cl_v086_time

_CL_V086_FILTER_SPECS = {
    "traffic_channels": {
        "element_type": 2006,
        "target_field": "media_ids",
        "label": "traffic channel",
    },
    "ad_types": {
        "element_type": 5001,
        "target_field": "ad_media_type_ids",
        "label": "ad type",
    },
    "industries": {
        "element_type": 1004,
        "target_field": "industry_ids",
        "label": "industry",
    },
}

_CL_V086_DICT_CACHE = {}
_CL_V086_DICT_CACHE_TTL_SECONDS = 600


class _CLNamedSearchRequest(_CLBaseModel):
    keyword: str
    start_date: str
    end_date: str

    traffic_channels: _CLList[str] = _CLField(default_factory=list)
    ad_types: _CLList[str] = _CLField(default_factory=list)
    industries: _CLList[str] = _CLField(default_factory=list)

    material_types: _CLList[str] = _CLField(default_factory=lambda: ["video"])
    top_n: int = 10
    page_size: int = 60
    download: bool = False
    auto_detect_language: bool = True
    wait_ms: int = 5000


class _CLResolveFilterRequest(_CLBaseModel):
    traffic_channels: _CLList[str] = _CLField(default_factory=list)
    ad_types: _CLList[str] = _CLField(default_factory=list)
    industries: _CLList[str] = _CLField(default_factory=list)


def _cl_v086_normalize(_value):
    _text = str(_value or "").strip().casefold()
    _text = _cl_v086_re.sub(r"[\s_\-–—/]+", "", _text)
    _text = _cl_v086_re.sub(r"[()\[\]{}（）【】]", "", _text)
    return _text


def _cl_v086_get_dictionary_sync(_element_type: int, _token: str, _ua: str):
    _now = _cl_v086_time.monotonic()
    _cached = _CL_V086_DICT_CACHE.get(int(_element_type))

    if (
        isinstance(_cached, dict)
        and (_now - float(_cached.get("loaded_at", 0))) < _CL_V086_DICT_CACHE_TTL_SECONDS
    ):
        return {
            "ok": True,
            "cached": True,
            "items": _cached.get("items") or [],
        }

    _reply = _cl_v0852_post_element_type(
        int(_element_type),
        _token,
        _ua,
        25,
    )

    if not _reply.get("ok"):
        return {
            "ok": False,
            "cached": False,
            "http_status": _reply.get("http_status"),
            "error": _reply.get("error"),
            "message": _reply.get("message"),
            "items": [],
        }

    _json = _reply.get("json")
    if not isinstance(_json, dict) or _json.get("code") != 200:
        return {
            "ok": False,
            "cached": False,
            "http_status": _reply.get("http_status"),
            "error": "unexpected_dictionary_response",
            "message": (
                (_json or {}).get("message")
                if isinstance(_json, dict)
                else "Insight filter dictionary did not return JSON."
            ),
            "items": [],
        }

    _items = _json.get("data") or []
    if not isinstance(_items, list):
        _items = []

    _CL_V086_DICT_CACHE[int(_element_type)] = {
        "loaded_at": _now,
        "items": _items,
    }

    return {
        "ok": True,
        "cached": False,
        "items": _items,
    }


def _cl_v086_item_aliases(_item):
    """
    Exact aliases only.

    Do NOT split elementCode on underscores. For example:
      Facebook_Ads_104
      Facebook_Audience_Network_SDK_1017

    Splitting those strings created a synthetic alias "Facebook" for multiple
    records and made the valid searchName "Facebook" on ccode=1006 appear
    ambiguous.

    Full elementCode remains accepted, and exact ccode / visible names /
    searchName remain accepted.
    """
    _values = []
    for _key in (
        "nameCn",
        "nameEn",
        "nameJp",
        "nameKr",
        "searchName",
        "ccode",
        "elementCode",
        "abbreviation",
    ):
        _value = _item.get(_key)
        if _value is None:
            continue
        _value = str(_value).strip()
        if _value and _value not in _values:
            _values.append(_value)

    return _values


def _cl_v086_public_option(_item):
    return {
        "id": str(_item.get("ccode") or _item.get("elementCode") or ""),
        "name_cn": _item.get("nameCn"),
        "name_en": _item.get("nameEn"),
        "search_name": _item.get("searchName"),
        "level": _item.get("level"),
        "parent_element_code": _item.get("parentElementCode"),
        "element_code": _item.get("elementCode"),
    }


def _cl_v086_resolve_one(_input_value, _items, _label):
    _raw = str(_input_value or "").strip()
    if not _raw:
        raise _CLHTTPException(
            status_code=422,
            detail=f"Empty {_label} value is not allowed."
        )

    _needle = _cl_v086_normalize(_raw)
    _matches = []

    for _item in _items:
        if not isinstance(_item, dict):
            continue

        _aliases = _cl_v086_item_aliases(_item)
        _normalized_aliases = {
            _cl_v086_normalize(_alias)
            for _alias in _aliases
            if _cl_v086_normalize(_alias)
        }

        if _needle in _normalized_aliases:
            _matches.append(_item)

    # Exact normalized name/ID should be unique. If Insight has duplicate labels,
    # fail loudly instead of guessing which ccode the operator intended.
    _unique = {}
    for _item in _matches:
        _id = str(_item.get("ccode") or _item.get("elementCode") or "")
        if _id:
            _unique[_id] = _item

    _matches = list(_unique.values())

    if len(_matches) == 1:
        _item = _matches[0]
        return {
            "input": _raw,
            **_cl_v086_public_option(_item),
        }

    if len(_matches) > 1:
        raise _CLHTTPException(
            status_code=422,
            detail={
                "stage": "filter_name_resolution",
                "error": "ambiguous_filter_name",
                "filter_type": _label,
                "input": _raw,
                "candidates": [
                    _cl_v086_public_option(_item)
                    for _item in _matches[:20]
                ],
            }
        )

    # Give useful nearby candidates without silently fuzzy-matching.
    _suggestions = []
    for _item in _items:
        if not isinstance(_item, dict):
            continue
        _aliases = _cl_v086_item_aliases(_item)
        for _alias in _aliases:
            _norm = _cl_v086_normalize(_alias)
            if not _norm:
                continue
            if _needle in _norm or _norm in _needle:
                _pub = _cl_v086_public_option(_item)
                if _pub not in _suggestions:
                    _suggestions.append(_pub)
                break
        if len(_suggestions) >= 10:
            break

    raise _CLHTTPException(
        status_code=422,
        detail={
            "stage": "filter_name_resolution",
            "error": "filter_name_not_found",
            "filter_type": _label,
            "input": _raw,
            "suggestions": _suggestions,
            "hint": "Use an Insight-visible Chinese name, English name, search name, or exact ID.",
        }
    )


async def _cl_v086_resolve_filters(
    _traffic_channels,
    _ad_types,
    _industries,
):
    _token, _ua, _auth_recovery = await _cl_auth_context_with_recovery()

    _inputs = {
        "traffic_channels": list(_traffic_channels or []),
        "ad_types": list(_ad_types or []),
        "industries": list(_industries or []),
    }

    _resolved = {
        "traffic_channels": [],
        "ad_types": [],
        "industries": [],
    }

    _ids = {
        "media_ids": [],
        "ad_media_type_ids": [],
        "industry_ids": [],
    }

    _dictionary_status = {}

    for _input_key, _spec in _CL_V086_FILTER_SPECS.items():
        _values = _inputs[_input_key]
        if not _values:
            continue

        _element_type = int(_spec["element_type"])
        _dictionary = await _cl_asyncio.to_thread(
            _cl_v086_get_dictionary_sync,
            _element_type,
            _token,
            _ua,
        )

        _dictionary_status[str(_element_type)] = {
            "ok": bool(_dictionary.get("ok")),
            "cached": bool(_dictionary.get("cached")),
            "item_count": len(_dictionary.get("items") or []),
        }

        if not _dictionary.get("ok"):
            raise _CLHTTPException(
                status_code=502,
                detail={
                    "stage": "filter_name_resolution",
                    "error": "dictionary_load_failed",
                    "filter_type": _spec["label"],
                    "element_type": _element_type,
                    "http_status": _dictionary.get("http_status"),
                    "message": _dictionary.get("message"),
                }
            )

        _items = _dictionary.get("items") or []

        for _value in _values:
            _one = _cl_v086_resolve_one(
                _value,
                _items,
                _spec["label"],
            )
            _resolved[_input_key].append(_one)

            _id = str(_one["id"])
            _target_field = _spec["target_field"]
            if _id not in _ids[_target_field]:
                _ids[_target_field].append(_id)

    return {
        "auth_recovery": _auth_recovery,
        "resolved": _resolved,
        "ids": _ids,
        "dictionary_status": _dictionary_status,
    }


@app.post("/crawl/search/options/resolve", tags=["crawl"])
async def creative_loop_search_options_resolve(_req: _CLResolveFilterRequest):
    _resolution = await _cl_v086_resolve_filters(
        _req.traffic_channels,
        _req.ad_types,
        _req.industries,
    )

    return {
        "ok": True,
        "version": "0.18.8.1",
        "input": {
            "traffic_channels": list(_req.traffic_channels or []),
            "ad_types": list(_req.ad_types or []),
            "industries": list(_req.industries or []),
        },
        "resolved": _resolution["resolved"],
        "ids": _resolution["ids"],
        "dictionary_status": _resolution["dictionary_status"],
        "auth_recovery": _resolution["auth_recovery"],
        "confirmed_dictionary_types": {
            "traffic_channels": 2006,
            "ad_types": 5001,
            "industries": 1004,
        },
        "security_note": "Filter dictionaries are read from Insight dynamically. No token, cookie, credential, or raw Selenium session ID is returned.",
    }


@app.post("/crawl/search/named", tags=["crawl"])
async def creative_loop_crawl_search_named(_req: _CLNamedSearchRequest):
    """
    Human-readable facade over the already-verified /crawl/search pipeline.

    Example:
      traffic_channels = ["Facebook"]
      ad_types          = ["Native Ads"]
      industries        = ["Audiobooks"]

    These names are dynamically resolved against Insight's own dictionaries,
    then the existing ID-based search route is called directly.
    """
    _resolution = await _cl_v086_resolve_filters(
        _req.traffic_channels,
        _req.ad_types,
        _req.industries,
    )

    _ids = _resolution["ids"]

    # Construct the already-tested ID-based request model so all existing
    # validation and crawler behavior remain unchanged.
    _inner_req = _CLSearchRequest(
        keyword=_req.keyword,
        start_date=_req.start_date,
        end_date=_req.end_date,
        media_ids=_ids["media_ids"],
        ad_media_type_ids=_ids["ad_media_type_ids"],
        industry_ids=_ids["industry_ids"],
        material_types=list(_req.material_types or []),
        top_n=_req.top_n,
        page_size=_req.page_size,
        download=_req.download,
        auto_detect_language=_req.auto_detect_language,
        wait_ms=_req.wait_ms,
    )

    _result = await creative_loop_crawl_search(_inner_req)

    if isinstance(_result, dict):
        _result = dict(_result)
        _result["mode"] = "search_named"
        _result["human_filters"] = {
            "traffic_channels": list(_req.traffic_channels or []),
            "ad_types": list(_req.ad_types or []),
            "industries": list(_req.industries or []),
        }
        _result["resolved_filters"] = _resolution["resolved"]
        _result["resolution_dictionary_status"] = _resolution["dictionary_status"]
        _result["resolution_auth_recovery"] = _resolution["auth_recovery"]
        _result["version"] = "0.18.8.1"

    return _result

# === CREATIVE_LOOP_V0_8_6_HUMAN_READABLE_FILTERS_END ===



# === CREATIVE_LOOP_V0_8_6_1_FILTER_RESOLVER_HOTFIX ===

# === CREATIVE_LOOP_V0_9_BRANDING_ROUTES ===
try:
    from .branding_v09 import register_branding_routes as _cl_register_branding_routes
except ImportError:
    from branding_v09 import register_branding_routes as _cl_register_branding_routes

_cl_register_branding_routes(app)

# === CREATIVE_LOOP_V0_19_WATERMARK_ARCHITECTURE ===
# New v2 modules are intentionally separate from branding_v09.py. Legacy routes
# stay available as compatibility wrappers while callers migrate to /brands and
# /watermark endpoints.
try:
    from .services.brand_registry import BrandRegistry as _CLBrandRegistry
    from .routers.brands import register_brand_routes as _cl_register_brand_routes
    from .routers.watermark import register_watermark_routes as _cl_register_watermark_routes
    from .routers.agent import register_agent_watermark_routes as _cl_register_agent_watermark_routes
except ImportError:
    from services.brand_registry import BrandRegistry as _CLBrandRegistry
    from routers.brands import register_brand_routes as _cl_register_brand_routes
    from routers.watermark import register_watermark_routes as _cl_register_watermark_routes
    from routers.agent import register_agent_watermark_routes as _cl_register_agent_watermark_routes

_cl_brand_registry = _CLBrandRegistry(WORKSPACE_DIR)
try:
    _cl_brand_registry.import_legacy_map(_load_product_map())
except Exception:
    # A malformed compatibility file cannot prevent the service from starting.
    pass
_cl_register_brand_routes(app, _cl_brand_registry)
_cl_register_watermark_routes(app, WORKSPACE_DIR, _cl_brand_registry)
_cl_register_agent_watermark_routes(app, _cl_brand_registry)

# CREATIVE_LOOP_V0_19_0_MOVING_WATERMARK_INSIGHT
# Insightrackr 式移动水印跟随处理（自包含模块，默认关闭、按需启用；不修改
# branding_v09 既有渲染管线）。POST /process/branding/moving-watermark-follow
try:
    from .branding_moving_watermark_insight import register_moving_watermark_routes as _cl_register_moving_watermark_routes
except ImportError:
    from branding_moving_watermark_insight import register_moving_watermark_routes as _cl_register_moving_watermark_routes

_cl_register_moving_watermark_routes(app)

# CREATIVE_LOOP_V0_19_0_FLOATBOAT_CONSOLE_CONTROL
# Keep Floatboat-facing routes in a small adapter module.  The existing
# Operator Console remains the only production queue for download, rendering,
# review and Mintegral upload.
try:
    from .agent_v019 import register_agent_routes as _cl_register_agent_routes
except ImportError:
    from agent_v019 import register_agent_routes as _cl_register_agent_routes

_cl_register_agent_routes(app)







# CREATIVE_LOOP_V0_9_7_1_VERSION_HOTFIX

# CREATIVE_LOOP_V0_9_8_TEMPORAL_OVERLAY

# CREATIVE_LOOP_V0_9_9_TEMPORAL_CANDIDATE_PREVIEW

# CREATIVE_LOOP_V0_9_10_DIAGONAL_GRID

# CREATIVE_LOOP_V0_9_11_MANUAL_SEED_GRID

# CREATIVE_LOOP_V0_9_12_THREE_ANCHOR_GRID

# CREATIVE_LOOP_V0_9_13_WATERMARK_ROUTER

# CREATIVE_LOOP_V0_9_14_ROUTER_AUTO_PREVIEW

# CREATIVE_LOOP_V0_9_15_WATERMARK_REMEDIATION

# CREATIVE_LOOP_V0_9_16_TILED_ALPHA_REBUILD

# CREATIVE_LOOP_V0_9_17_TILED_SHAPE_DYNAMIC

# CREATIVE_LOOP_V0_9_18_STRONG_LOGO_TEMPLATE

# CREATIVE_LOOP_V0_10_0_BRAND_REPLACEMENT

# CREATIVE_LOOP_V0_10_1_BOUNDARY_PLACEMENT

# CREATIVE_LOOP_V0_10_3_1_STRUCTURE_ROUTER_HOTFIX

# CREATIVE_LOOP_V0_10_5_MULTIMODAL_STRUCTURE

# CREATIVE_LOOP_V0_10_6_PAIRED_REGIME

# CREATIVE_LOOP_V0_11_0_SEMANTIC_STRUCTURE

# CREATIVE_LOOP_V0_11_1_WHISPER_EPISODES

# CREATIVE_LOOP_V0_11_2_SEMANTIC_ENVELOPE

# CREATIVE_LOOP_V0_11_4_SEMANTIC_PLAN

# CREATIVE_LOOP_V0_11_5_COMPOSED_TOP_BRAND

# CREATIVE_LOOP_V0_11_6_LOGO_TRANSITION

# CREATIVE_LOOP_V0_12_0_FULL_RENDER

# CREATIVE_LOOP_V0_12_1_POLICY_FIXES

# CREATIVE_LOOP_V0_12_2_TOP_ICON_COVER

# CREATIVE_LOOP_V0_12_3_CANONICAL_TOP_SLOT

# CREATIVE_LOOP_V0_12_4_DIAGONAL_PREVIEW

# CREATIVE_LOOP_V0_13_0_BRAND_ASSETS

# CREATIVE_LOOP_V0_14_0_LANGUAGE

# CREATIVE_LOOP_V0_16_0_MINTEGRAL

# CREATIVE_LOOP_V0_16_1_MINTEGRAL_UPLOAD_ONLY

# CREATIVE_LOOP_V0_16_2_PROCESS_UPLOAD

# CREATIVE_LOOP_V0_16_3_MINTEGRAL_DUPLICATE_REUSE

# CREATIVE_LOOP_V0_16_4_RENDER_CACHE

# CREATIVE_LOOP_V0_17_0_BATCH_QUEUE

# CREATIVE_LOOP_V0_18_0_OPERATOR_CONSOLE

# CREATIVE_LOOP_V0_18_1_SEARCH_DATE_RANGE

# CREATIVE_LOOP_V0_18_2_LOGIN_WINDOW_FIX

# CREATIVE_LOOP_V0_18_3_INSIGHT_SESSION_RECOVERY

# CREATIVE_LOOP_V0_18_4_PROGRESS_CANCEL

# CREATIVE_LOOP_V0_18_5_METADATA_PROCESSED_LAYOUT

# CREATIVE_LOOP_V0_18_6_BRAND_ASSETS_PREFLIGHT

# CREATIVE_LOOP_V0_18_7_DEMO_LANGUAGE_ASSETS

# CREATIVE_LOOP_V0_18_8_IMP_RANKING

# CREATIVE_LOOP_V0_18_8_1_NATIVE_IMPRESSION

# CREATIVE_LOOP_V0_18_8_2_PRODUCT_MAP

# CREATIVE_LOOP_V0_18_9_AUTO_PRODUCT_REGISTRY

# CREATIVE_LOOP_V0_18_9_1_REVIEW_DUPLICATE_RECOVERY

# CREATIVE_LOOP_V0_18_9_2_PATH_RECONCILIATION

# CREATIVE_LOOP_V0_18_9_3_IMP_PARSER_HOTFIX

# CREATIVE_LOOP_V0_18_9_4_STALE_SOURCE_RELOCATOR

# CREATIVE_LOOP_V0_18_9_5_BRAND_VALIDATOR_LANGUAGE_FIX

# CREATIVE_LOOP_V0_18_10_OPERATOR_UX_SPEED

# CREATIVE_LOOP_V0_18_10_1_PROCESSING_OBSERVABILITY

# CREATIVE_LOOP_V0_18_10_2_MEDIA_HANDOFF_RECOVERY

# CREATIVE_LOOP_V0_18_10_3_SEMANTIC_FAST_PATH

# CREATIVE_LOOP_V0_18_10_4_BRANDING_REVIEW_CORRECTNESS

# CREATIVE_LOOP_V0_18_10_5_TOP_BRAND_ENFORCEMENT

# CREATIVE_LOOP_V0_18_10_6_TASK_LIFECYCLE_RECOVERY

# CREATIVE_LOOP_V0_18_10_7_CONSOLE_INSIGHT_ISOLATION

# CREATIVE_LOOP_V0_18_10_8_DETAIL_METADATA_HANDOFF

# CREATIVE_LOOP_V0_18_10_8_1_PRODUCT_HELPER_RESTORE

# CREATIVE_LOOP_V0_18_10_8_2_BOM_JSON_COMPAT
# Install after route/function definitions is intentional: module import
# finishes before Uvicorn serves requests, and JSONDecoder patch also protects
# earlier `from json import loads` aliases.
try:
    from app.bom_json_compat import install_bom_json_compat as _install_bom_json_compat
except Exception:
    from .bom_json_compat import install_bom_json_compat as _install_bom_json_compat

_install_bom_json_compat()

# CREATIVE_LOOP_V0_18_10_8_3_NATIVE_IMPRESSION_SORT_HANDSHAKE

# CREATIVE_LOOP_V0_18_10_9_2_DETAIL_DOWNLOAD_MODE_HANDOFF

# CREATIVE_LOOP_V0_18_10_9_3_DETAIL_IDENTITY_RECOVERY

# CREATIVE_LOOP_V0_18_10_9_4_PRODUCT_IDENTITY_CONTINUATION
