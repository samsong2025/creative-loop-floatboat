from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..models.brand import BrandIdentity, BrandProfile
from .brand_identity import identity_from_detail, normalize_brand_id, normalize_identity_text


class BrandRegistry:
    """File-backed registry with exact package/title resolution and icon dedupe."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.root = self.workspace_dir / "config" / "brand_registry"
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_index({"schema_version": 1, "packages": {}, "app_titles": {}, "brands": {}})

    def _read_index(self) -> dict[str, Any]:
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8-sig"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {"schema_version": 1, "packages": {}, "app_titles": {}, "brands": {}}

    def _write_index(self, value: dict[str, Any]) -> None:
        self.index_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:80] or "brand"

    def list_profiles(self) -> list[dict[str, Any]]:
        index = self._read_index()
        profiles = []
        for brand_id, relative in index.get("brands", {}).items():
            path = self.root / Path(str(relative))
            if path.is_file():
                try:
                    profiles.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
        return profiles

    def get_profile(self, brand_id: str) -> dict[str, Any] | None:
        index = self._read_index()
        relative = index.get("brands", {}).get(brand_id)
        if not relative:
            return None
        path = self.root / Path(str(relative))
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def resolve(self, *, product_name: str | None = None, app_title: str | None = None, package_name: str | None = None) -> dict[str, Any] | None:
        index = self._read_index()
        brand_id = (index.get("packages", {}).get(package_name) if package_name else None)
        brand_id = brand_id or (index.get("app_titles", {}).get(app_title) if app_title else None)
        brand_id = brand_id or normalize_brand_id(product_name)
        return self.get_profile(brand_id) if brand_id else None

    def upsert(self, identity: BrandIdentity, *, icon_bytes: bytes | None = None, source_url: str | None = None) -> dict[str, Any]:
        if not identity.product_name:
            raise ValueError("Cannot register unresolved brand identity")
        brand_id = identity.brand_id or normalize_brand_id(identity.product_name)
        if not brand_id:
            raise ValueError("Cannot derive brand_id from explicit product identity")
        brand_dir = self.root / self._safe_name(brand_id)
        source_dir = brand_dir / "source"
        templates_dir = brand_dir / "templates"
        source_dir.mkdir(parents=True, exist_ok=True)
        templates_dir.mkdir(parents=True, exist_ok=True)
        existing = self.get_profile(brand_id) or {}
        package_names = list(dict.fromkeys([*(existing.get("package_names") or []), *([identity.package_name] if identity.package_name else [])]))
        aliases = list(dict.fromkeys([*(existing.get("aliases") or []), *identity.aliases]))
        icon = dict(existing.get("icon") or {})
        if icon_bytes:
            digest = hashlib.sha256(icon_bytes).hexdigest()
            icon_path = source_dir / "icon.png"
            if not icon_path.exists() or icon.get("sha256") != digest:
                icon_path.write_bytes(icon_bytes)
            icon = {"relative_path": str(icon_path.relative_to(self.workspace_dir)).replace("\\", "/"), "sha256": digest, "source_url": source_url or identity.icon_url, "source": identity.icon_source or "insight_detail", "size_bytes": len(icon_bytes), "updated_at": datetime.now(timezone.utc).isoformat()}
            templates = self._generate_templates(icon_path, templates_dir)
        elif identity.icon_url and not icon:
            icon = {"source_url": identity.icon_url, "source": identity.icon_source or "insight_detail", "status": "pending"}
            templates = existing.get("templates") or {}
        else:
            templates = existing.get("templates") or {}
        profile = BrandProfile(brand_id=brand_id, product_name=identity.product_name, app_title=identity.app_title, package_names=package_names, aliases=aliases, icon=icon, templates=templates, updated_at=datetime.now(timezone.utc).isoformat())
        (brand_dir / "profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        index = self._read_index()
        index.setdefault("packages", {})
        index.setdefault("app_titles", {})
        index.setdefault("brands", {})
        for package in package_names:
            index["packages"][package] = brand_id
        for title in [identity.app_title, identity.product_name, *aliases]:
            if title:
                index["app_titles"][title] = brand_id
        index["brands"][brand_id] = f"{self._safe_name(brand_id)}/profile.json"
        self._write_index(index)
        return profile.model_dump(mode="json")

    def _generate_templates(self, icon_path: Path, templates_dir: Path) -> dict[str, Any]:
        """Build detector-ready icon templates for Dynamic Watermark Engine v1.

        The detector loads the canonical gray and edge templates, then evaluates
        them at several image scales.  Fixed-size copies are also retained for
        debugging and future ROI-specific detector policies; they are not a
        substitute for multi-scale matching.
        """
        image = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return {"status": "generation_failed", "version": 1}
        if image.ndim == 3 and image.shape[2] == 4:
            alpha = image[:, :, 3]
            bgr = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2BGRA)
            background = np.full(image.shape[:2], 255, dtype=np.uint8)
            gray_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2GRAY)
            gray = np.where(alpha > 0, gray_rgb, background).astype(np.uint8)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edge = cv2.Canny(gray, 64, 160)
        # A lightly dilated edge image is materially more tolerant of a small
        # alpha-blended logo than a one-pixel reference edge map.
        edge_dilated = cv2.dilate(edge, np.ones((2, 2), dtype=np.uint8), iterations=1)
        written: dict[str, str] = {}
        for name, data in (("icon-gray.png", gray), ("icon-edge.png", edge), ("icon-edge-dilated.png", edge_dilated)):
            path = templates_dir / name
            cv2.imwrite(str(path), data)
            written[name] = str(path.relative_to(self.workspace_dir)).replace("\\", "/")
        for size in (32, 48, 64, 96):
            path = templates_dir / f"icon-{size}.png"
            cv2.imwrite(str(path), cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA))
            written[f"icon-{size}.png"] = str(path.relative_to(self.workspace_dir)).replace("\\", "/")
        average_hash = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        average_hash = (average_hash > average_hash.mean()).astype(np.uint8)
        descriptor = {
            "version": 2,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "average_hash": "".join(str(int(bit)) for bit in average_hash.flatten()),
            "edge_histogram": cv2.calcHist([edge], [0], None, [16], [0, 256]).flatten().astype(int).tolist(),
            "assets": written,
        }
        features_path = templates_dir.parent / "features.json"
        features_path.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "generated_at": descriptor["generated_at"],
            "version": 2,
            "features_relative_path": str(features_path.relative_to(self.workspace_dir)).replace("\\", "/"),
            "assets": written,
            "detector_assets": {
                "grayscale": written.get("icon-gray.png"),
                "edge": written.get("icon-edge.png"),
                "edge_dilated": written.get("icon-edge-dilated.png"),
                "multi_scale_reference_sizes": [32, 48, 64, 96],
            },
        }

    def import_legacy_map(self, mapping: dict[str, Any]) -> dict[str, int]:
        created = 0
        for package, product in (mapping.get("package_names") or {}).items():
            identity = identity_from_detail({"product_name": product, "package_name": package}, mapping)
            if identity.product_name and not self.get_profile(identity.brand_id or ""):
                self.upsert(identity)
                created += 1
        for title, product in (mapping.get("app_titles") or {}).items():
            identity = identity_from_detail({"product_name": product, "app_title": title}, mapping)
            self.upsert(identity)
        return {"created": created, "profiles": len(self.list_profiles())}