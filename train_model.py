"""
RouteWiseAI — ML Model Training Script
=======================================
Fetches real route data from TomTom + OpenWeatherMap APIs for 30 Indian cities,
then augments across time-of-day, day-of-week, and weather scenarios to create
a robust dataset (~50,000+ samples). Trains a tuned RandomForestRegressor on
all 9 features consumed by delay_predictor.py.

Features: distance_km, traffic_level, weather_encoded, hour_of_day, day_of_week,
          is_rush_hour, traffic_weather_interaction, rush_hour_traffic,
          distance_weather_risk

Usage:
    python train_model.py
"""

import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import time
import random
import json
from datetime import datetime

# ─── API Keys ────────────────────────────────────────────────────────────────
TOMTOM_API_KEY = "y3lqXrAZjVCThGRsEFVLiiJb5GSUpmI1"
WEATHER_API_KEY = "81b66e6697efb2b8adaa3f99f877b664"

# ─── 30 Indian Cities (up from 10) ──────────────────────────────────────────
cities = [
    # Metros
    ("Delhi", 28.6139, 77.2090),
    ("Mumbai", 19.0760, 72.8777),
    ("Bangalore", 12.9716, 77.5946),
    ("Chennai", 13.0827, 80.2707),
    ("Kolkata", 22.5726, 88.3639),
    ("Hyderabad", 17.3850, 78.4867),
    # Tier 1
    ("Pune", 18.5204, 73.8567),
    ("Ahmedabad", 23.0225, 72.5714),
    ("Jaipur", 26.9124, 75.7873),
    ("Lucknow", 26.8467, 80.9462),
    # Tier 2
    ("Chandigarh", 30.7333, 76.7794),
    ("Indore", 22.7196, 75.8577),
    ("Bhopal", 23.2599, 77.4126),
    ("Nagpur", 21.1458, 79.0882),
    ("Patna", 25.6093, 85.1376),
    ("Kochi", 9.9312, 76.2673),
    ("Coimbatore", 11.0168, 76.9558),
    ("Visakhapatnam", 17.6868, 83.2185),
    ("Vadodara", 22.3072, 73.1812),
    ("Surat", 21.1702, 72.8311),
    # Tier 3 / Important Hubs
    ("Mysore", 12.2958, 76.6394),
    ("Varanasi", 25.3176, 82.9739),
    ("Amritsar", 31.6340, 74.8723),
    ("Guwahati", 26.1445, 91.7362),
    ("Dehradun", 30.3165, 78.0322),
    ("Ranchi", 23.3441, 85.3096),
    ("Bhubaneswar", 20.2961, 85.8245),
    ("Thiruvananthapuram", 8.5241, 76.9366),
    ("Jodhpur", 26.2389, 73.0243),
    ("Agra", 27.1767, 78.0081),
]

# ─── Weather Severity Mapping (matches delay_predictor.py) ──────────────────
weather_severity_map = {
    "Clear": 1,
    "Clouds": 2,
    "Drizzle": 4,
    "Rain": 4,
    "Mist": 5,
    "Haze": 5,
    "Fog": 5,
    "Dust": 5,
    "Smoke": 5,
    "Snow": 7,
    "Thunderstorm": 9,
    "Squall": 9,
    "Tornado": 9,
}

# All possible weather conditions for augmentation
ALL_WEATHER_CONDITIONS = list(weather_severity_map.keys())

# ─── Hour & Day Definitions ─────────────────────────────────────────────────
RUSH_HOURS = {7, 8, 9, 16, 17, 18}
SAMPLE_HOURS = [0, 3, 6, 7, 8, 9, 11, 13, 15, 16, 17, 18, 20, 22]
SAMPLE_DAYS = list(range(7))  # Mon(0) to Sun(6)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Fetch Real Route Data from TomTom API
# ═══════════════════════════════════════════════════════════════════════════════

def get_weather(lat, lon):
    """Fetch current weather condition from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        weather_main = data["weather"][0]["main"]
        temp = data["main"].get("temp", 30)
        humidity = data["main"].get("humidity", 50)
        wind_speed = data.get("wind", {}).get("speed", 0)
        return weather_main, temp, humidity, wind_speed
    except Exception as e:
        print(f"  ⚠️  Weather API error for ({lat},{lon}): {e}")
        return "Clear", 30, 50, 0


def fetch_route_data(lat1, lon1, lat2, lon2):
    """Fetch route info from TomTom Routing API."""
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{lat1},{lon1}:{lat2},{lon2}/json"
    params = {"traffic": "true", "travelMode": "car", "key": TOMTOM_API_KEY}
    try:
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        route = res.json()["routes"][0]["summary"]
        return {
            "distance_km": round(route["lengthInMeters"] / 1000, 2),
            "travel_time_sec": route["travelTimeInSeconds"],
            "traffic_delay_sec": route.get("trafficDelayInSeconds", 0),
        }
    except Exception as e:
        return None


def fetch_all_routes():
    """Fetch real route data for all city pairs (sampled for API limits)."""
    print("📡 Fetching real route data from TomTom API...")
    print(f"   Cities: {len(cities)} | Max pairs: {len(cities) * (len(cities) - 1)}")

    routes = []
    total_pairs = len(cities) * (len(cities) - 1)

    # For 30 cities = 870 pairs. Sample ~150 representative pairs to stay within API limits.
    # We'll augment these into 50,000+ training samples.
    all_pairs = [(i, j) for i in range(len(cities)) for j in range(len(cities)) if i != j]
    max_api_calls = min(len(all_pairs), 200)
    sampled_pairs = random.sample(all_pairs, max_api_calls) if len(all_pairs) > max_api_calls else all_pairs

    print(f"   Fetching {len(sampled_pairs)} route pairs via API...")

    for idx, (i, j) in enumerate(sampled_pairs):
        from_name, lat1, lon1 = cities[i]
        to_name, lat2, lon2 = cities[j]

        route_data = fetch_route_data(lat1, lon1, lat2, lon2)
        if route_data is None:
            continue

        # Get real weather at origin
        weather_label, temp, humidity, wind_speed = get_weather(lat1, lon1)

        routes.append({
            "from_city": from_name,
            "to_city": to_name,
            "from_lat": lat1,
            "from_lon": lon1,
            "to_lat": lat2,
            "to_lon": lon2,
            "distance_km": route_data["distance_km"],
            "base_travel_time_sec": route_data["travel_time_sec"],
            "traffic_delay_sec": route_data["traffic_delay_sec"],
            "weather": weather_label,
            "temperature": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
        })

        if (idx + 1) % 20 == 0:
            print(f"   ✅ Fetched {idx + 1}/{len(sampled_pairs)} routes...")
        time.sleep(1.0)  # Rate limit

    print(f"\n   📊 Successfully fetched {len(routes)} routes from APIs")
    return routes


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Data Augmentation — Expand to 50,000+ Samples
# ═══════════════════════════════════════════════════════════════════════════════

def compute_traffic_level(base_delay_sec, distance_km, hour, day_of_week, weather_severity):
    """
    Compute a realistic traffic level (1-10) based on route characteristics.
    Uses the real traffic delay as a base, then adjusts by time/weather.
    """
    # Base traffic from real delay data (normalized to 1-10)
    if distance_km > 0:
        delay_ratio = base_delay_sec / max(distance_km * 60, 1)  # seconds delay per km
    else:
        delay_ratio = 0

    base_level = min(10, max(1, int(delay_ratio * 2) + 1))

    # Time-of-day multiplier
    if hour in RUSH_HOURS:
        time_mult = random.uniform(1.3, 1.8)
    elif hour in {10, 11, 12, 13, 14, 15}:
        time_mult = random.uniform(0.9, 1.2)
    elif hour in {19, 20, 21}:
        time_mult = random.uniform(0.7, 1.0)
    else:  # late night / early morning
        time_mult = random.uniform(0.3, 0.6)

    # Weekend reduction
    if day_of_week >= 5:
        time_mult *= random.uniform(0.6, 0.85)

    # Weather impact on traffic
    weather_mult = 1.0 + (weather_severity - 1) * 0.08

    level = base_level * time_mult * weather_mult
    # Add noise
    level += random.gauss(0, 0.5)

    return round(min(10, max(1, level)), 1)


def compute_delay_minutes(distance_km, traffic_level, weather_severity, hour, day_of_week, base_travel_sec):
    """
    Compute realistic delay in minutes based on all features.
    This is the TARGET variable for training.
    """
    is_rush = 1 if hour in RUSH_HOURS else 0

    # Base delay from traffic level
    base_delay = (traffic_level - 1) * distance_km * 0.03  # minutes per km scaled by traffic

    # Rush hour surge
    rush_surge = is_rush * traffic_level * random.uniform(0.5, 1.5)

    # Weather delay component
    weather_delay = weather_severity * distance_km * 0.005 * random.uniform(0.5, 1.5)

    # Interaction effects (what the model needs to learn)
    interaction_delay = (traffic_level * weather_severity) * 0.02 * random.uniform(0.5, 1.5)

    # Weekend tends to have less delay
    weekend_factor = 0.7 if day_of_week >= 5 else 1.0

    total_delay = (base_delay + rush_surge + weather_delay + interaction_delay) * weekend_factor

    # Add realistic noise (±15%)
    noise = random.gauss(1.0, 0.15)
    total_delay *= max(0.3, noise)

    return round(max(0, total_delay), 2)


def augment_data(raw_routes):
    """
    Augment each real route across multiple hours, days, and weather conditions.
    Creates 50,000+ diverse training samples from ~150-200 real routes.
    """
    print("\n🔄 Augmenting data across time/day/weather scenarios...")
    augmented = []

    # For each real route, generate variations
    for route in raw_routes:
        distance_km = route["distance_km"]
        base_travel_sec = route["base_travel_time_sec"]
        base_delay_sec = route["traffic_delay_sec"]

        # Sample diverse combinations of hour × day × weather
        weather_scenarios = random.sample(
            ALL_WEATHER_CONDITIONS,
            min(len(ALL_WEATHER_CONDITIONS), 6)
        )

        for hour in SAMPLE_HOURS:
            for day in SAMPLE_DAYS:
                for weather in weather_scenarios:
                    weather_enc = weather_severity_map.get(weather, 1)
                    is_rush = 1 if hour in RUSH_HOURS else 0

                    # Compute realistic traffic level
                    traffic_level = compute_traffic_level(
                        base_delay_sec, distance_km, hour, day, weather_enc
                    )

                    # Engineered features (must match delay_predictor.py exactly)
                    traffic_weather_interaction = traffic_level * weather_enc
                    rush_hour_traffic = traffic_level * is_rush
                    distance_weather_risk = distance_km * weather_enc

                    # Compute target: delay in minutes
                    delay_min = compute_delay_minutes(
                        distance_km, traffic_level, weather_enc,
                        hour, day, base_travel_sec
                    )

                    augmented.append({
                        "from_city": route["from_city"],
                        "to_city": route["to_city"],
                        "distance_km": distance_km,
                        "traffic_level": traffic_level,
                        "weather_encoded": weather_enc,
                        "hour_of_day": hour,
                        "day_of_week": day,
                        "is_rush_hour": is_rush,
                        "traffic_weather_interaction": traffic_weather_interaction,
                        "rush_hour_traffic": rush_hour_traffic,
                        "distance_weather_risk": distance_weather_risk,
                        "weather_label": weather,
                        "delay_minutes": delay_min,
                    })

    df = pd.DataFrame(augmented)
    print(f"   📊 Generated {len(df):,} training samples")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Model Training with Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "distance_km",
    "traffic_level",
    "weather_encoded",
    "hour_of_day",
    "day_of_week",
    "is_rush_hour",
    "traffic_weather_interaction",
    "rush_hour_traffic",
    "distance_weather_risk",
]

TARGET_COL = "delay_minutes"


def train_model(df):
    """Train RandomForestRegressor with hyperparameter tuning and evaluation."""
    print("\n🧠 Training ML Model...")
    print(f"   Features: {FEATURE_COLS}")
    print(f"   Target: {TARGET_COL}")
    print(f"   Samples: {len(df):,}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # ── Train / Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Hyperparameter Grid Search ──
    print("\n   🔍 Running GridSearchCV for best hyperparameters...")
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [15, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"   ✅ Best params: {grid_search.best_params_}")

    # ── Evaluation ──
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n   📈 Model Evaluation (Test Set):")
    print(f"      MAE  : {mae:.3f} minutes")
    print(f"      RMSE : {rmse:.3f} minutes")
    print(f"      R²   : {r2:.4f}")

    # ── Feature Importance ──
    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(f"\n   🏆 Feature Importance:")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"      {row['feature']:35s} {row['importance']:.4f}  {bar}")

    return best_model, {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "best_params": grid_search.best_params_,
        "feature_importance": importance_df.to_dict("records"),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_cities": len(cities),
        "trained_at": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Save Everything
# ═══════════════════════════════════════════════════════════════════════════════

def save_outputs(model, df, metrics):
    """Save the model, dataset, and metrics to disk."""
    # Ensure directories exist
    os.makedirs("model", exist_ok=True)
    os.makedirs("backend/model", exist_ok=True)

    # ── Save Dataset ──
    df.to_csv("model/dataset.csv", index=False)
    print(f"\n💾 Saved dataset → model/dataset.csv ({len(df):,} rows)")

    # ── Save Model (both locations — root and backend) ──
    model_paths = ["model/delay_model.pkl", "backend/model/delay_model.pkl"]
    for path in model_paths:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"💾 Saved model → {path} ({size_mb:.1f} MB)")

    # ── Save Metrics ──
    with open("model/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"💾 Saved metrics → model/training_metrics.json")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Synthetic Route Generator (for --quick / Docker builds)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_routes():
    """Generate synthetic routes for ALL city pairs using haversine distance.
    No API calls needed — runs in seconds. Used for Docker/Render builds."""
    print("⚡ Generating synthetic routes (no API calls)...")
    raw_routes = []
    for i in range(len(cities)):
        for j in range(len(cities)):
            if i == j:
                continue
            from_name, lat1, lon1 = cities[i]
            to_name, lat2, lon2 = cities[j]

            # Haversine distance
            import math
            R = 6371
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
            dist_km = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            dist_km *= random.uniform(1.2, 1.5)  # road multiplier

            # Realistic synthetic values
            avg_speed = random.uniform(40, 70)  # km/h
            base_travel_sec = int(dist_km / avg_speed * 3600)
            traffic_delay_sec = random.randint(0, int(dist_km * 25))

            raw_routes.append({
                "from_city": from_name,
                "to_city": to_name,
                "from_lat": lat1, "from_lon": lon1,
                "to_lat": lat2, "to_lon": lon2,
                "distance_km": round(dist_km, 2),
                "base_travel_time_sec": base_travel_sec,
                "traffic_delay_sec": traffic_delay_sec,
                "weather": random.choice(ALL_WEATHER_CONDITIONS),
                "temperature": random.uniform(15, 42),
                "humidity": random.randint(20, 95),
                "wind_speed": random.uniform(0, 15),
            })

    print(f"   📊 Generated {len(raw_routes)} synthetic route pairs")
    return raw_routes


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RouteWiseAI ML Training Pipeline")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: use synthetic data only (no API calls). For Docker/Render builds."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  RouteWiseAI — ML Model Training Pipeline")
    print(f"  Mode: {'⚡ QUICK (synthetic)' if args.quick else '🌐 FULL (real API data)'}")
    print("=" * 70)
    start_time = time.time()

    if args.quick:
        # Quick mode: synthetic data only, no API calls (~30 seconds)
        raw_routes = generate_synthetic_routes()
    else:
        # Full mode: fetch real data from TomTom + OpenWeatherMap APIs
        raw_routes = fetch_all_routes()

        if len(raw_routes) < 10:
            print("\n⚠️  Too few routes fetched. Check API keys / network.")
            print("    Falling back to synthetic data...\n")
            raw_routes = generate_synthetic_routes()

    # Step 2: Augment into large training dataset
    df = augment_data(raw_routes)

    # Step 3: Train model
    model, metrics = train_model(df)

    # Step 4: Save outputs
    save_outputs(model, df, metrics)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print(f"     {len(cities)} cities | {len(df):,} samples | R² = {metrics['r2']:.4f}")
    print(f"{'=' * 70}")
