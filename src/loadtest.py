"""
Latency and throughput benchmark for the serving API.

    python -m src.loadtest --url http://127.0.0.1:8000
    python -m src.loadtest --cold-start-image cfpb-classifier:latest

For each endpoint and each concurrency level (1, 4, 16), a fixed number of
requests is issued by that many concurrent workers, each drawing real test-set
narratives. Reported: p50/p95/p99 latency and sustained QPS (completed / wall).

/predict and /explain are measured separately and deliberately. IG needs
N_STEPS forward+backward passes per request, so /explain is far slower -- and
that gap is the serving cost of explainability, the trade-off the audit is about.

Cold start is time from `docker run` to the first successful /predict: image
start, interpreter, imports, weight load, first forward pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
import urllib.error
import urllib.request

import httpx
import numpy as np
import pandas as pd

from src.utils import RESULTS_DIR

OUT_PATH = RESULTS_DIR / "serving_benchmark.json"
CONCURRENCY = (1, 4, 16)
SEED = 42


def _texts(n: int) -> list[str]:
    df = pd.read_parquet(RESULTS_DIR / "predictions_test.parquet", columns=["narrative"])
    return df["narrative"].sample(n=n, random_state=SEED).tolist()


async def _run(url: str, endpoint: str, texts: list[str], concurrency: int, n: int, body: dict):
    queue: asyncio.Queue[str] = asyncio.Queue()
    for i in range(n):
        queue.put_nowait(texts[i % len(texts)])
    latencies, errors = [], 0

    async def worker(client: httpx.AsyncClient):
        nonlocal errors
        while True:
            try:
                text = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            t0 = time.perf_counter()
            r = await client.post(f"{url}{endpoint}", json={"narrative": text, **body})
            if r.status_code == 200:
                latencies.append((time.perf_counter() - t0) * 1000)
            else:
                errors += 1

    async with httpx.AsyncClient(timeout=600) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        wall = time.perf_counter() - t0

    lat = np.array(latencies)
    return {
        "concurrency": concurrency,
        "requests": n,
        "errors": errors,
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "p99_ms": round(float(np.percentile(lat, 99)), 1),
        "mean_ms": round(float(lat.mean()), 1),
        "qps": round(len(lat) / wall, 2),
    }


async def benchmark(url: str, n_predict: int, n_explain: int, n_steps: int) -> dict:
    texts = _texts(200)
    async with httpx.AsyncClient(timeout=600) as client:
        health = (await client.get(f"{url}/health")).json()
        for t in texts[:5]:  # warm-up: first calls pay allocator / kernel setup
            await client.post(f"{url}/predict", json={"narrative": t})

    out = {"url": url, "health": health, "predict": [], "explain": []}
    for c in CONCURRENCY:
        r = await _run(url, "/predict", texts, c, n_predict, {})
        out["predict"].append(r)
        print(f"/predict  c={c:<3} {r}")
    for c in CONCURRENCY:
        r = await _run(url, "/explain", texts, c, max(n_explain, c), {"n_steps": n_steps})
        out["explain"].append(r)
        print(f"/explain  c={c:<3} {r}")
    out["explain_n_steps"] = n_steps
    return out


def cold_start(image: str, port: int = 8001, timeout: float = 180) -> dict:
    """Seconds from `docker run` to the first successful /predict."""
    text = _texts(1)[0]
    payload = json.dumps({"narrative": text}).encode()
    name = "cfpb-coldstart"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    t0 = time.perf_counter()
    subprocess.run(
        ["docker", "run", "-d", "--name", name, "-p", f"{port}:8000", image],
        check=True,
        capture_output=True,
    )
    try:
        while time.perf_counter() - t0 < timeout:
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/predict",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=30) as r:
                    if r.status == 200:
                        return {"image": image, "cold_start_s": round(time.perf_counter() - t0, 2)}
            except urllib.error.URLError, ConnectionError:
                time.sleep(0.1)
        raise TimeoutError(f"no successful /predict within {timeout}s")
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _image_size(image: str) -> str:
    out = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Size}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return f"{int(out.stdout.strip()) / 1e9:.2f} GB"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.loadtest")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--predict-requests", type=int, default=200)
    ap.add_argument("--explain-requests", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--cold-start-image")
    ap.add_argument("--cold-start-runs", type=int, default=3)
    ap.add_argument("--label", default="container")
    args = ap.parse_args(argv)

    results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    if args.cold_start_image:
        runs = [
            cold_start(args.cold_start_image)["cold_start_s"] for _ in range(args.cold_start_runs)
        ]
        results["cold_start"] = {
            "image": args.cold_start_image,
            "image_size": _image_size(args.cold_start_image),
            "runs_s": runs,
            "median_s": float(np.median(runs)),
        }
        print(json.dumps(results["cold_start"], indent=2))
    else:
        results[args.label] = asyncio.run(
            benchmark(args.url, args.predict_requests, args.explain_requests, args.n_steps)
        )
    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
