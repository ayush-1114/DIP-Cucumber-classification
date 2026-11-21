"""Fetch a bounded, labelled subset of WMC-Leafset v4 (DOI 10.17632/8m2ytxd4dg.4)."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://data.mendeley.com/public-api/datasets/8m2ytxd4dg/files"
DEST = Path(__file__).resolve().parents[1] / "external_data/wmc4"


def request(url, **kwargs):
    with requests.Session() as session:
        session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=4, backoff_factor=1, status_forcelist=[429, 502, 503, 504]
                )
            ),
        )
        response = session.get(url, timeout=50, **kwargs)
        response.raise_for_status()
        return response


def listing(folder):
    items = []
    while True:
        batch = request(
            BASE,
            params={
                "folder_id": folder,
                "version": 4,
                "$start": len(items),
                "$limit": 1000,
            },
        ).json()
        items.extend(batch)
        if len(batch) < 1000:
            return {Path(i["filename"]).stem: i for i in items}


def fetch(item, folder):
    info = item["content_details"]
    path = DEST / folder / Path(item["filename"]).name
    path.parent.mkdir(parents=True, exist_ok=True)
    if (
        path.exists()
        and hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256_hash"]
    ):
        return path
    content = request(info["download_url"]).content
    if hashlib.sha256(content).hexdigest() != info["sha256_hash"]:
        raise ValueError(f"Hash mismatch: {path.name}")
    path.write_bytes(content)
    return path


def main():
    labels = listing("e577c7a6-5a66-4f56-b6a0-a2b24bd774c4")
    images = listing("9afd429b-94d9-4515-ac29-510dd1d944ff")
    healthy = listing("e49123ce-83ec-4574-b9b5-fcc0e837a517")

    def download(stem):
        p = fetch(labels[stem], "labels")
        if any(line.split()[:1] == ["0"] for line in p.read_text().splitlines()):
            fetch(images[stem], "miner")

    with ThreadPoolExecutor(5) as workers:
        list(workers.map(download, sorted(labels.keys() & images.keys())))
        list(
            workers.map(
                lambda item: fetch(item, "healthy"),
                sorted(healthy.values(), key=lambda x: x["filename"])[::3],
            )
        )
    print(
        "Downloaded",
        len(list((DEST / "miner").glob("*.jpg"))),
        "annotated miner photos",
        flush=True,
    )


if __name__ == "__main__":
    main()
