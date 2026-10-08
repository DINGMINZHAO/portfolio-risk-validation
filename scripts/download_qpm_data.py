from pathlib import Path
from urllib.request import Request, urlopen
from datetime import datetime, timezone
import hashlib
import json


REPO = "AhmadMujtaba200210/qpm-project"
COMMIT = "96ab45ff7bdc12ee9e220d1ade5138e571f520e4"

FILES = [
    {
        "repo_path": "data/raw/raw_adj_close_prices.csv",
        "local_path": "data/raw/qpm/raw_adj_close_prices.csv",
        "role": "raw_market_input",
        "git_blob_sha": "e940e483acdab1219e74ac08a56ee72149d724c7",
    },
    {
        "repo_path": "data/raw/raw_risk_free_rate.csv",
        "local_path": "data/raw/qpm/raw_risk_free_rate.csv",
        "role": "raw_market_input",
        "git_blob_sha": "a01547dae76fe920edb139cd3c0bb8f0ea62ccc8",
    },
    {
        "repo_path": "data/processed/portfolio_weights.csv",
        "local_path": "data/external/qpm/upstream/portfolio_weights.csv",
        "role": "upstream_reference",
        "git_blob_sha": "21cbe91a7ab736f7c77c9d0f74609675a6f6d6f2",
    },
    {
        "repo_path": "data/processed/monthly_rebal_dates.csv",
        "local_path": "data/external/qpm/upstream/monthly_rebal_dates.csv",
        "role": "upstream_reference",
        "git_blob_sha": "9c663e52e64c1d0784f2ad1e5c96e9e55b259cf5",
    },
    {
        "repo_path": "data/processed/regime_labels.csv",
        "local_path": "data/external/qpm/upstream/regime_labels.csv",
        "role": "upstream_reference",
        "git_blob_sha": "3f1595d68b91ecf983b4656ce39f1957b3035747",
    },
    {
    "repo_path": "data/processed/backtest_results.csv",
    "local_path": "data/external/qpm/upstream/backtest_results.csv",
    "role": "upstream_reference",
    "git_blob_sha": "5a7ca73ab58e2ee48eacdc394e64a2e5759b844c",
    },
]


def git_blob_sha(data):
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


project_root = Path(__file__).resolve().parents[1]

manifest = {
    "repository": REPO,
    "commit": COMMIT,
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    "files": [],
}

for item in FILES:
    url = (
        f"https://raw.githubusercontent.com/"
        f"{REPO}/{COMMIT}/{item['repo_path']}"
    )

    target = project_root / item["local_path"]
    target.parent.mkdir(parents=True, exist_ok=True)

    request = Request(url, headers={"User-Agent": "quant-risk-project"})

    with urlopen(request, timeout=60) as response:
        data = response.read()

    actual_git_sha = git_blob_sha(data)

    if actual_git_sha != item["git_blob_sha"]:
        raise RuntimeError(
            f"Git blob SHA mismatch for {item['repo_path']}"
        )

    target.write_bytes(data)

    sha256 = hashlib.sha256(data).hexdigest()

    manifest["files"].append(
        {
            "repo_path": item["repo_path"],
            "local_path": item["local_path"],
            "role": item["role"],
            "source_url": url,
            "bytes": len(data),
            "git_blob_sha": actual_git_sha,
            "sha256": sha256,
        }
    )

    print(
        f"OK | {item['local_path']} "
        f"| {len(data):,} bytes "
        f"| sha256={sha256[:12]}..."
    )

manifest_path = (
    project_root
    / "data/external/qpm/upstream/source_manifest.json"
)

manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print()
print(f"Manifest written to: {manifest_path.relative_to(project_root)}")