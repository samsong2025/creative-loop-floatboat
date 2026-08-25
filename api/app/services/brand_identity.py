from __future__ import annotations

import re
from typing import Any

from ..models.brand import BrandIdentity


_NULLS = {"", "none", "null", "unknown", "n/a", "na", "-", "--"}


def normalize_identity_text(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    value = str(value).strip()
    return None if value.casefold() in _NULLS else (value or None)


def normalize_brand_id(value: str | None) -> str | None:
    value = normalize_identity_text(value)
    if not value:
        return None
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or None


def merge_explicit_identity(base: dict[str, Any] | None, extra: dict[str, Any] | None) -> dict[str, Any]:
    """Merge only explicit fields; a later null never erases a known identity."""
    merged = dict(base or {})
    for key in ("product_name", "app_title", "package_name", "icon_url", "icon_source", "detail_id"):
        value = normalize_identity_text((extra or {}).get(key))
        if value and not normalize_identity_text(merged.get(key)):
            merged[key] = value
    aliases: list[str] = []
    for values in (merged.get("aliases", []), (extra or {}).get("aliases", [])):
        values = [values] if isinstance(values, str) else values
        for value in values or []:
            value = normalize_identity_text(value)
            if value and value not in aliases:
                aliases.append(value)
    merged["aliases"] = aliases
    if not merged.get("product_name") and merged.get("app_title"):
        merged["product_name"] = merged["app_title"]
        merged["identity_source"] = "insight_explicit_app_title"
    return merged


def identity_from_detail(detail: dict[str, Any], legacy_map: dict[str, Any] | None = None) -> BrandIdentity:
    """Resolve explicit Insight identity, then exact legacy map entries only."""
    merged = merge_explicit_identity({}, detail)
    mapping = legacy_map or {}
    package_name = merged.get("package_name")
    app_title = merged.get("app_title")
    product_name = merged.get("product_name")
    mapped = mapping.get("package_names", {}).get(package_name) if package_name else None
    mapped = mapped or (mapping.get("app_titles", {}).get(app_title) if app_title else None)
    if not product_name and mapped:
        product_name = normalize_identity_text(mapped)
        merged["identity_source"] = "legacy_product_map_exact"
    if product_name and not merged.get("identity_source"):
        merged["identity_source"] = "insight_explicit"
    brand_id = normalize_brand_id(product_name)
    aliases = list(dict.fromkeys([x for x in [product_name, app_title, *merged.get("aliases", [])] if x]))
    return BrandIdentity(
        brand_id=brand_id,
        product_name=product_name,
        app_title=app_title,
        package_name=package_name,
        aliases=aliases,
        icon_url=merged.get("icon_url"),
        icon_source=merged.get("icon_source"),
        detail_id=merged.get("detail_id"),
        identity_confidence=1.0 if product_name else 0.0,
        identity_source=merged.get("identity_source", "unresolved"),
    )