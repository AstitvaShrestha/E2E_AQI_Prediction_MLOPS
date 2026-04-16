# scripts/find_sensor_ids.py
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY", "")
if not API_KEY:
    raise ValueError("OPENAQ_API_KEY not set in .env")

headers = {"X-API-Key": API_KEY}
BASE    = "https://api.openaq.org/v3"

CITIES = {
    # "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    # "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
    # "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    # "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
}


def make_bbox(lat, lon, delta=0.5):
    return f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"


def safe_get(url, params=None, retries=3) -> dict | None:
    """
    GET with automatic 429 retry.
    Waits 60s on rate limit then retries up to 3 times.
    """
    for attempt in range(retries):
        try:
            resp = requests.get(
                url, params=params, headers=headers, timeout=20
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            # Any other error — return None silently
            return None
        except Exception:
            return None
    return None


def find_locations(lat, lon) -> list:
    data = safe_get(
        f"{BASE}/locations",
        {"bbox": make_bbox(lat, lon), "limit": 100}
    )
    if not data:
        return []
    results = data.get("results", [])
    print(f"  Found {len(results)} locations")
    return results


def get_pm25_sensors(location_id: int) -> list:
    data = safe_get(f"{BASE}/locations/{location_id}/sensors")
    if not data:
        return []
    return [
        s["id"] for s in data.get("results", [])
        if s.get("parameter", {}).get("name", "").lower() == "pm25"
    ]


def get_latest_reading(sensor_id: int) -> dict:
    """
    Try descending order first.
    If that gives old date, also try without sort (API default).
    Take the more recent of the two.
    """
    def fetch(params):
        data = safe_get(f"{BASE}/sensors/{sensor_id}/hours", params)
        if not data:
            return None
        results = data.get("results", [])
        if not results:
            return None
        utc = results[0]["period"]["datetimeTo"]["utc"]
        return {"latest": utc[:10], "year": int(utc[:4])}

    # Try desc
    r1 = fetch({"limit": 1, "order_by": "datetime", "sort_order": "desc"})
    # Try default (some endpoints ignore sort params)
    r2 = fetch({"limit": 1})

    # Return whichever has the more recent date
    candidates = [r for r in [r1, r2] if r]
    if not candidates:
        return {}
    return max(candidates, key=lambda x: x["latest"])


def scan_location(loc: dict) -> dict | None:
    """Returns result if location has PM2.5 data from 2024+."""
    loc_id   = loc["id"]
    loc_name = loc.get("name", "Unknown")
    provider = loc.get("provider", {}).get("name", "Unknown")

    sensor_ids = get_pm25_sensors(loc_id)
    if not sensor_ids:
        return None

    for sensor_id in sensor_ids:
        info = get_latest_reading(sensor_id)
        if info and info.get("year", 0) >= 2024:
            return {
                "loc_id":   loc_id,
                "name":     loc_name,
                "provider": provider,
                "sensor":   sensor_id,
                "latest":   info["latest"],
            }
    return None


# ── Main ───────────────────────────────────────────────────────────────────

all_city_ids = {}

for city_idx, (city, coords) in enumerate(CITIES.items()):
    print(f"\n{'='*60}")
    print(f"  {city}")
    print(f"{'='*60}")

    # Wait between cities to avoid rate limit carryover
    if city_idx > 0:
        print("  Waiting 30s before next city (rate limit protection)...")
        time.sleep(30)

    locations = find_locations(coords["lat"], coords["lon"])
    if not locations:
        print(f"  No locations found")
        all_city_ids[city] = []
        continue

    print(f"  Scanning with 4 workers (reduced to avoid 429)...\n")

    results = []

    # Reduced to 4 workers (was 6/8) — prevents 429 during parallel scan
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(scan_location, loc): loc
            for loc in locations
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                results.append(result)
            if done % 25 == 0:
                print(f"  ... {done}/{len(locations)} done, "
                      f"{len(results)} active so far")
            # Small sleep between completed futures to space out requests
            time.sleep(0.1)

    results.sort(key=lambda x: x["loc_id"])

    working_ids = []
    for r in results:
        print(f"  ✓ Loc {r['loc_id']:8d} | {r['name'][:45]}")
        print(f"    Provider : {r['provider']}")
        print(f"    Sensor   : {r['sensor']}")
        print(f"    Latest   : {r['latest']}")
        print()
        working_ids.append(r["loc_id"])

    working_ids = list(set(working_ids))
    all_city_ids[city] = working_ids
    print(f"  → {len(working_ids)} IDs: {working_ids}")

# ── Output ─────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  PASTE INTO src/config.py")
print(f"{'='*60}\n")
for city, ids in all_city_ids.items():
    print(f'    "{city}": {{')
    print(f'        "openaq_location_ids": {ids},')
    print(f'    }},')

with open("data/sensor_ids_final.txt", "w") as f:
    for city, ids in all_city_ids.items():
        f.write(f'"{city}": {ids}\n')
print("\n  Saved to data/sensor_ids_final.txt")