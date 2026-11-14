#!/usr/bin/env python3
"""P3 Nepal-first data discovery and approved-asset ingest on persistent Fuzzball volume.

Stdlib only. Discovery never claims that an STAC listing is a downloaded image.
Downloads require a reviewed manifest of actual HTTPS asset URLs; no token URLs in git.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

STAC = "https://stac.dataspace.copernicus.eu/v1/search"
BBOX = [84.405, 27.682, 85.70, 28.35]  # Broad provisional corridor, NOT an authoritative AOI
WINDOWS = [("pre_event", "2026-07-01T00:00:00Z/2026-08-25T23:59:59Z"),
           ("post_event", "2026-08-26T00:00:00Z/2026-09-10T23:59:59Z")]
COLLECTIONS = ["sentinel-2-l1c", "sentinel-1-grd"]


def get_json(url, payload=None):
    headers = {"User-Agent": "pravah-team12-data-audit/0.1", "Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=90) as res:
        return json.load(res)


def discover(root):
    out = root / "manifests" / "stac"
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    for collection in COLLECTIONS:
        for window, interval in WINDOWS:
            name = f"{collection}_{window}"
            query = {"collections": [collection], "bbox": BBOX, "datetime": interval, "limit": 100}
            try:
                response = get_json(STAC, query)
                features = response.get("features", [])
                # Only serialize metadata. Full asset URLs may include authorization info;
                # do not commit these manifests without review.
                (out / (name + "_raw_private.json")).write_text(json.dumps(response, indent=2) + "\n")
                items = []
                for item in features:
                    props = item.get("properties") or {}
                    items.append({
                        "id": item.get("id"), "collection": item.get("collection"),
                        "datetime": props.get("datetime"), "created": props.get("created"),
                        "updated": props.get("updated"),
                        "bbox": item.get("bbox"),
                        "assets": [{"key": key, "type": asset.get("type"), "roles": asset.get("roles"),
                                    "title": asset.get("title")}
                                   for key, asset in (item.get("assets") or {}).items()],
                    })
                links = response.get("links") or []
                more = any(l.get("rel") == "next" for l in links)
                (out / (name + "_index.json")).write_text(
                    json.dumps({"query": query, "items": items, "results_this_page": len(items),
                                "has_more_pages": more, "collection_complete": not more,
                                "downloaded": False}, indent=2) + "\n")
                print(f"{name}: {len(items)} STAC items in first page; more_pages={more}")
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                print(f"ERROR {name}: {exc}", file=sys.stderr)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "data_root": str(root), "bbox_is_authoritative": False,
              "status": "discovery_only", "errors": errors}
    (out / "discovery_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 1 if errors else 0


def download(root, manifest, max_gb):
    assets = json.loads(Path(manifest).read_text())
    if not isinstance(assets, list) or not assets:
        raise ValueError("Reviewed manifest must be a nonempty list of assets")
    max_bytes = int(max_gb * 1024**3)
    written = 0
    root = root.resolve()
    report = []
    for item in assets:
        url, relative = item["url"], item["destination"]
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Only HTTPS asset URLs without embedded userinfo are supported")
        if parsed.query:
            raise ValueError("Do not put signed/query-bearing asset URLs in manifests; use secrets")
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError("Destination outside data volume")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"EXISTS {relative} (not overwritten)")
            report.append({"destination": relative, "status": "already_exists_unverified"})
            continue
        staging = target.with_name(target.name + ".part")
        # Restart partial files safely; no assumption of HTTP range support.
        if staging.exists():
            staging.unlink()
        sha = hashlib.sha256()
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": "pravah-team12-ingest/0.1"}), timeout=90) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) + written > max_bytes:
                    raise ValueError("Job download budget exceeded")
                with staging.open("wb") as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        written += len(block)
                        if written > max_bytes:
                            raise ValueError("Job download budget exceeded")
                        sha.update(block)
                        stream.write(block)
            actual = sha.hexdigest()
            expected = item.get("sha256")
            if expected and actual.lower() != expected.lower():
                raise ValueError(f"SHA256 mismatch for {relative}")
            staging.replace(target)
            report.append({"destination": relative, "status": "downloaded", "bytes": target.stat().st_size,
                           "sha256": actual, "provider": item.get("provider"),
                           "acquisition_at": item.get("acquisition_at"),
                           "available_at": item.get("available_at")})
            print(f"DOWNLOADED {relative}: {target.stat().st_size} bytes")
        except Exception:
            staging.unlink(missing_ok=True)
            raise
    report_path = root / "manifests" / "approved_ingest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["discover", "download"])
    p.add_argument("--root", type=Path, default=Path(os.getenv("PRAVAH_DATA_ROOT", "/data")))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--max-gb", type=float, default=25)
    args = p.parse_args()
    if not args.root.is_dir():
        raise ValueError(f"Persistent volume mount missing: {args.root}")
    if args.command == "discover":
        return discover(args.root)
    if not args.manifest:
        p.error("--manifest is required for download")
    return download(args.root, args.manifest, args.max_gb)


if __name__ == "__main__":
    sys.exit(main())
