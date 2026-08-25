"""
Creative Loop UTF-8 BOM JSON compatibility shim.

Insight/detail responses and Windows-edited JSON files may occasionally begin
with the UTF-8 BOM bytes EF BB BF. Python's stdlib json.loads() rejects a BOM
when its input is already decoded to str, producing:

    Unexpected UTF-8 BOM (decode using utf-8-sig)

This shim strips ONLY a leading BOM. Normal JSON input is unchanged.
"""

from __future__ import annotations

import json
from typing import Any

_INSTALLED = False
_ORIGINAL_LOADS = None
_ORIGINAL_DECODER_DECODE = None
_ORIGINAL_REQUESTS_RESPONSE_JSON = None


def _strip_json_bom(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("\ufeff"):
            return value[1:]
        return value

    if isinstance(value, bytes):
        if value.startswith(b"\xef\xbb\xbf"):
            return value[3:]
        return value

    if isinstance(value, bytearray):
        raw = bytes(value)
        if raw.startswith(b"\xef\xbb\xbf"):
            return raw[3:]
        return value

    return value


def install_bom_json_compat() -> dict[str, Any]:
    global _INSTALLED
    global _ORIGINAL_LOADS
    global _ORIGINAL_DECODER_DECODE
    global _ORIGINAL_REQUESTS_RESPONSE_JSON

    if _INSTALLED:
        return {
            "ok": True,
            "already_installed": True,
        }

    # Patch JSONDecoder.decode as well as json.loads. This also protects code
    # that imported `loads` before this shim was installed, because the old
    # stdlib loads function still dispatches through the default JSONDecoder.
    _ORIGINAL_DECODER_DECODE = json.JSONDecoder.decode

    def _bom_safe_decoder_decode(self, s, *args, **kwargs):
        return _ORIGINAL_DECODER_DECODE(
            self,
            _strip_json_bom(s),
            *args,
            **kwargs,
        )

    json.JSONDecoder.decode = _bom_safe_decoder_decode

    _ORIGINAL_LOADS = json.loads

    def _bom_safe_loads(s, *args, **kwargs):
        return _ORIGINAL_LOADS(
            _strip_json_bom(s),
            *args,
            **kwargs,
        )

    _bom_safe_loads._creative_loop_bom_safe = True
    json.loads = _bom_safe_loads

    # requests.Response.json() can use requests' own complexjson binding.
    # Give it an explicit BOM fallback so detail API responses are protected
    # even if requests is using simplejson instead of stdlib json.
    try:
        import requests

        _ORIGINAL_REQUESTS_RESPONSE_JSON = requests.Response.json

        def _bom_safe_response_json(self, **kwargs):
            try:
                return _ORIGINAL_REQUESTS_RESPONSE_JSON(
                    self,
                    **kwargs,
                )
            except Exception:
                raw = bytes(
                    self.content
                    or b""
                )

                if not raw.startswith(
                    b"\xef\xbb\xbf"
                ):
                    raise

                return json.loads(
                    raw.decode(
                        "utf-8-sig"
                    ),
                    **kwargs,
                )

        _bom_safe_response_json._creative_loop_bom_safe = True
        requests.Response.json = _bom_safe_response_json

    except Exception:
        # requests may not be imported/installed in a narrow test environment.
        # stdlib json protection above remains active.
        pass

    _INSTALLED = True

    return {
        "ok": True,
        "already_installed": False,
        "stdlib_json": True,
        "requests_response_json": (
            _ORIGINAL_REQUESTS_RESPONSE_JSON
            is not None
        ),
    }
