"""
smoke_test.py — validates /health, /metrics, /predict match the agreed contract.

Checks for new response fields: model_version, serving_version, latency_ms
"""

import argparse
import json
import sys
import requests


def check(label, condition, detail=""):
    marker = "PASS" if condition else "FAIL"
    print(f"  [{marker}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--input", default="sample_data/input_sample.json")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    print(f"Smoke testing {args.url}...")
    all_ok = True

    # /health
    try:
        r = requests.get(f"{args.url}/health", timeout=args.timeout)
        all_ok &= check("GET /health returns 200",
                         r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            body = r.json()
            all_ok &= check("  has 'status'", "status" in body)
            all_ok &= check("  has 'model_version'",
                             "model_version" in body)
            all_ok &= check("  has 'serving_version'",
                             "serving_version" in body)
    except Exception as e:
        all_ok &= check("GET /health", False, str(e))

    # /metrics
    try:
        r = requests.get(f"{args.url}/metrics", timeout=args.timeout)
        all_ok &= check("GET /metrics returns 200",
                         r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            all_ok &= check("  has subst_top1_embedding_score",
                             "subst_top1_embedding_score" in r.text)
            all_ok &= check("  has http auto-instrumentation",
                             "http_request" in r.text
                             or "http_requests" in r.text)
    except Exception as e:
        all_ok &= check("GET /metrics", False, str(e))

    # /predict
    try:
        with open(args.input) as f:
            payload = json.load(f)
        r = requests.post(f"{args.url}/predict",
                          json=payload, timeout=args.timeout)
        all_ok &= check("POST /predict returns 200",
                         r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            body = r.json()
            for field in ["recipe_id", "missing_ingredient", "request_id",
                          "substitutions", "model_version",
                          "serving_version", "latency_ms"]:
                all_ok &= check(f"  has '{field}'", field in body)
            subs = body.get("substitutions", [])
            all_ok &= check(f"  {len(subs)} substitutions (>= 1)",
                             len(subs) >= 1)
            if subs:
                first = subs[0]
                all_ok &= check("    has 'ingredient'",
                                 "ingredient" in first)
                all_ok &= check("    has 'embedding_score'",
                                 "embedding_score" in first)
                all_ok &= check("    rank == 1",
                                 first.get("rank") == 1)
                print(f"         -> {first['ingredient']} "
                      f"(score={first.get('embedding_score')})")
    except Exception as e:
        all_ok &= check("POST /predict", False, str(e))

    print()
    if all_ok:
        print("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("ONE OR MORE CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
