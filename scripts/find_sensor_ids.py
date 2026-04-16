import requests
from dotenv import load_dotenv
import os

load_dotenv()


API_KEY = os.getenv("OPENAQ_API_KEY")
headers = {"X-API-Key": API_KEY}

cities = {
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
}

for city, coords in cities.items():
    url = "https://api.openaq.org/v3/locations"
    params = {
        "coordinates": f"{coords['lat']},{coords['lon']}",
        "radius":       25000,   # 25km radius around city centre
        "limit":        20,
    }
    resp = requests.get(url, params=params, headers=headers).json()

    print(f"\n=== {city} ===")
    for loc in resp.get("results", []):
        # Check if this location has PM2.5 sensors
        sensors = loc.get("sensors", [])
        has_pm25 = any(
            s.get("parameter", {}).get("name") == "pm25"
            for s in sensors
        )
        if has_pm25:
            print(f"  ID: {loc['id']}")
            print(f"  Name: {loc.get('name', 'Unknown')}")
            print(f"  Provider: {loc.get('provider', {}).get('name', 'Unknown')}")
            print(f"  Sensors: {[s['parameter']['name'] for s in sensors]}")
            print()