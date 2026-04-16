"""
City configuration — single source of truth.
All pipeline, training, and API code imports from here.
"""

CITIES = {
    "Delhi": {
        "lat": 28.6139,
        "lon": 77.2090,
        "country": "India",
        "timezone": "Asia/Kolkata",
        "openaq_location_ids": [5570, 5541, 5610, 235, 11603, 5616, 17, 50, 5586, 5588, 5404, 5598],   # CPCB stations
        "aqi_profile": "severe_winter",               
    },
    "Mumbai": {
        "lat": 19.0760,
        "lon": 72.8777,
        "country": "India",
        "timezone": "Asia/Kolkata",
        "openaq_location_ids": [2598, 5593, 6927, 6943, 6945, 6948, 6956, 6959, 6965, 6967, 6987, 7850, 8039, 11602, 11604, 11606, 11611, 11612, 12024, 12039],
        "aqi_profile": "coastal_moderate",
    },
    "Kolkata": {
        "lat": 22.5726,
        "lon": 88.3639,
        "country": "India",
        "timezone": "Asia/Kolkata",
        "openaq_location_ids":  [716, 910, 2460, 5614, 6310, 6946, 6950, 6981, 8172, 10633, 10844, 10851, 10904, 10918, 1236037, 3409320, 3409509, 3409524, 3409530],
        "aqi_profile": "industrial_high",
    },
    "Chennai": {
        "lat": 13.0827,
        "lon": 80.2707,
        "country": "India",
        "timezone": "Asia/Kolkata",
        "openaq_location_ids": [378, 2461, 2549, 2586, 5655, 8558, 10780, 11578, 11579, 11581, 12046],
        "aqi_profile": "coastal_low",
    },
    "Bengaluru": {
        "lat": 12.9716,
        "lon": 77.5946,
        "country": "India",
        "timezone": "Asia/Kolkata",
        "openaq_location_ids": [412, 594, 2592, 5547, 5548, 5607, 6973, 6974, 6975, 6983, 6984, 229473, 2498781, 3409312, 3409385, 3409388],
        "aqi_profile": "traffic_moderate",
    },
}

# India National AQI categories (CPCB standard)
AQI_CATEGORIES = [
    (0,   50,  "Good",           "#00b050", "No health implications."),
    (51,  100, "Satisfactory",   "#92d050", "Minor breathing discomfort to sensitive people."),
    (101, 200, "Moderate",       "#ffff00", "Breathing discomfort to people with lung/heart disease."),
    (201, 300, "Poor",           "#ff9900", "Breathing discomfort to most people on prolonged exposure."),
    (301, 400, "Very Poor",      "#ff0000", "Respiratory illness on prolonged exposure."),
    (401, 500, "Severe",         "#c00000", "Affects healthy people. Serious impact on those with disease."),
]


def get_aqi_category(aqi):
    """Return the AQI category for a given AQI value."""
    for low, high, label, color, advice in AQI_CATEGORIES:
        if low <= aqi <= high:
            return {"label": label, "color": color, "advice": advice}
        
    return {"label":"Severe", "color":"#c00000", 
            "advice":"Affects healthy people. Serious impact on those with disease."}

# EPA PM2.5 -> AQI breakpoints
PM25_BREAKPOINTS = [
    (0.0,   12.0,   0,   50),
    (12.1,  35.4,  51,  100),
    (35.5,  55.4, 101,  150),
    (55.5, 150.4, 151,  200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]


def pm25_to_aqi(pm25):
    """Convert PM2.5 concentration (µg/m³) to AQI using EPA formula."""

    if pm25<0:
        return 0
    
    for c_lo, c_hi, i_lo, i_hi in PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            aqi = ((i_hi - i_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + i_lo
            return round(aqi)
        
    return 500  # Above max breakpoint, cap at AQI 500


# if __name__ == "__main__":
#     # Quick test of PM2.5 to AQI conversion
#     print('Cities:', list(CITIES.keys()))
#     print('Delhi coords:', CITIES['Delhi']['lat'], CITIES['Delhi']['lon'])

#     # Test PM2.5 to AQI conversion
#     test_values = [5, 20, 45, 100, 200, 350]
#     for pm25 in test_values:
#         aqi = pm25_to_aqi(pm25)
#         cat = get_aqi_category(aqi)
#         print(f'PM2.5={pm25:6.1f} -> AQI={aqi:4d} -> {cat["label"]}')