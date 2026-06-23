import os
import io
import requests
from datetime import datetime, timedelta, date
from notion_client import Client
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed in CI
import matplotlib.pyplot as plt

# =========================================================
# AUTH
# =========================================================
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
PAGE_ID = os.environ["NOTION_PAGE_ID"]

notion = Client(auth=NOTION_TOKEN)

# =========================================================
# SITE CONSTANTS — Herschel Island / Qikiqtaruk
# =========================================================
LAT = 69.590
LON = -139.099
BBOX_WMS = "-141,68,-136,71"          # for GIBS WMS (minlon,minlat,maxlon,maxlat)
BBOX_STAC = [-141.0, 68.0, -136.0, 71.0]  # for STAC search (same order)

now = datetime.utcnow()


# =========================================================
# HELPERS — Notion block builders (kept tiny to reduce repetition)
# =========================================================
def heading(text, level=2):
    tag = f"heading_{level}"
    return {"object": "block", "type": tag, tag: {"rich_text": [{"type": "text", "text": {"content": text}}]}}


def paragraph(text):
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]}}


def divider():
    return {"object": "block", "type": "divider", "divider": {}}


def upload_image_to_notion(image_bytes, filename="image.png"):
    """
    Uploads raw image bytes to Notion's file upload API and returns the
    upload id. We use this instead of external image URLs because Notion's
    external-URL fetcher is unreliable for query-string-based image
    services (no file extension, content negotiated at request time).
    """
    create_resp = requests.post(
        "https://api.notion.com/v1/file_uploads",
        headers={
            "Authorization": f"Bearer {NOTION_TOKEN}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        },
        json={},
        timeout=20,
    )
    create_resp.raise_for_status()
    upload_id = create_resp.json()["id"]

    send_resp = requests.post(
        f"https://api.notion.com/v1/file_uploads/{upload_id}/send",
        headers={
            "Authorization": f"Bearer {NOTION_TOKEN}",
            "Notion-Version": "2022-06-28",
        },
        files={"file": (filename, image_bytes, "image/png")},
        timeout=30,
    )
    send_resp.raise_for_status()
    return upload_id


def image_block_from_upload(upload_id):
    return {
        "object": "block",
        "type": "image",
        "image": {"type": "file_upload", "file_upload": {"id": upload_id}},
    }


def fig_to_png_bytes(fig):
    """Renders a matplotlib figure to PNG bytes in memory, then closes it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# =========================================================
# MODULE 1 — WEATHER (temperature, wind, humidity, pressure)
# Source: Open-Meteo current_weather + hourly (free, no key needed)
# =========================================================
def get_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LAT,
            "longitude": LON,
            "current_weather": True,
            "hourly": "relativehumidity_2m,pressure_msl",
            "timezone": "UTC",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        cw = data["current_weather"]
        current_time = cw["time"]  # e.g. "2026-06-22T14:00"

        # Match the current hour in the hourly arrays to current_weather's timestamp
        humidity = None
        pressure = None
        try:
            idx = data["hourly"]["time"].index(current_time)
            humidity = data["hourly"]["relativehumidity_2m"][idx]
            pressure = data["hourly"]["pressure_msl"][idx]
        except (ValueError, KeyError, IndexError) as e:
            print("WEATHER: could not align hourly index:", e)

        return {
            "temperature_c": cw.get("temperature"),
            "windspeed_kmh": cw.get("windspeed"),
            "humidity_pct": humidity,
            "pressure_hpa": pressure,
            "status": "ok",
        }
    except Exception as e:
        print("WEATHER FETCH FAILED:", e)
        return {
            "temperature_c": None,
            "windspeed_kmh": None,
            "humidity_pct": None,
            "pressure_hpa": None,
            "status": "missing",
        }


weather = get_weather()

if weather["status"] == "ok":
    weather_text = (
        f"Air temperature: {weather['temperature_c']} °C\n"
        f"Wind speed: {weather['windspeed_kmh']} km/h\n"
        f"Humidity: {weather['humidity_pct']} %\n"
        f"Pressure: {weather['pressure_hpa']} hPa\n"
        f"Source: Open-Meteo (ERA5-based forecast/analysis)"
    )
else:
    weather_text = "Weather data unavailable — fetch failed. Check Action logs."


# =========================================================
# MODULE 1b — SUN: sunrise, sunset, day length
# Source: sunrise-sunset.org (free, no key). Herschel Island is above
# 69°N, so polar day / polar night periods are expected and handled
# explicitly rather than treated as errors.
# =========================================================
def get_sun_info():
    try:
        url = "https://api.sunrise-sunset.org/json"
        params = {"lat": LAT, "lng": LON, "formatted": 0}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        print("SUN: raw response status:", data.get("status"))

        if data.get("status") != "OK":
            # status can be e.g. INVALID_REQUEST; treat as no data rather than crash
            return {"status": "no_data", "raw_status": data.get("status")}

        results = data["results"]
        sunrise = datetime.fromisoformat(results["sunrise"].replace("Z", "+00:00"))
        sunset = datetime.fromisoformat(results["sunset"].replace("Z", "+00:00"))
        day_length_s = results.get("day_length")

        return {
            "status": "ok",
            "sunrise": sunrise,
            "sunset": sunset,
            "day_length_s": day_length_s,
        }
    except Exception as e:
        print("SUN FETCH FAILED:", e)
        return {"status": "error"}


sun_info = get_sun_info()

# At this latitude in summer, sunrise/sunset can come back as the same
# instant or with a day_length very close to 24h/0h — this is polar day,
# not a bug, so we detect and label it rather than show a misleading time.
if sun_info["status"] == "ok":
    day_length_s = sun_info["day_length_s"]
    hours = int(day_length_s // 3600)
    minutes = int((day_length_s % 3600) // 60)

    if day_length_s >= 23 * 3600 + 30 * 60:
        sun_text = (
            f"Day length: ~{hours}h {minutes}min — consistent with continuous "
            f"daylight (midnight sun) at this latitude.\n"
            f"Source: sunrise-sunset.org"
        )
    elif day_length_s <= 30 * 60:
        sun_text = (
            f"Day length: ~{hours}h {minutes}min — consistent with polar night "
            f"at this latitude.\n"
            f"Source: sunrise-sunset.org"
        )
    else:
        sun_text = (
            f"Sunrise (UTC): {sun_info['sunrise'].strftime('%H:%M')}\n"
            f"Sunset (UTC): {sun_info['sunset'].strftime('%H:%M')}\n"
            f"Day length: {hours}h {minutes}min\n"
            f"Source: sunrise-sunset.org"
        )
elif sun_info["status"] == "no_data":
    sun_text = f"Sun data unavailable ({sun_info.get('raw_status')}) — may be a polar-day/polar-night edge case the API can't resolve at this latitude."
else:
    sun_text = "Sun data fetch failed — check Action logs."


# =========================================================
# SHARED HELPER — Open-Meteo historical archive
# Used by both the 10-day temperature chart and thawing degree days.
# =========================================================
def fetch_daily_temps(start_date, end_date):
    """
    Fetches daily mean temperature for [start_date, end_date] (inclusive)
    from Open-Meteo's historical archive (ERA5 reanalysis).
    Returns a dict {date_str: temp_c} or {} on failure.
    """
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": "temperature_2m_mean",
            "timezone": "UTC",
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        daily = data.get("daily", {})
        times = daily.get("time", [])
        temps = daily.get("temperature_2m_mean", [])
        return dict(zip(times, temps))
    except Exception as e:
        print(f"HISTORICAL FETCH FAILED for {start_date} to {end_date}:", e)
        return {}


# =========================================================
# MODULE 1c — TEMPERATURE CHART: last 10 days vs 30-year daily normal
# =========================================================
def build_temperature_chart():
    """
    Builds a chart of the last 10 days of mean daily temperature against
    the 30-year (1996-2025) average for the same calendar days.

    The 30-year normal is computed here by pulling the same 10-day
    calendar window from each of the past 30 years and averaging —
    Open-Meteo has no pre-computed "climate normal" endpoint, so this
    is done as 30 separate small historical queries.
    """
    end = (now - timedelta(days=1)).date()  # yesterday, since today's mean isn't final yet
    start = end - timedelta(days=9)

    recent = fetch_daily_temps(start, end)
    if not recent:
        return None, "No recent historical temperature data returned."

    day_labels = sorted(recent.keys())
    recent_values = [recent[d] for d in day_labels]

    # Build 30-year normal for the same month/day combinations
    normals_by_day = {d: [] for d in day_labels}
    current_year = now.year

    for years_back in range(1, 31):
        hist_start = start.replace(year=start.year - years_back)
        hist_end = end.replace(year=end.year - years_back)
        hist_data = fetch_daily_temps(hist_start, hist_end)

        if not hist_data:
            continue

        # Map historical dates back onto this year's day labels by month/day
        for hist_date_str, temp in hist_data.items():
            hist_date = datetime.strptime(hist_date_str, "%Y-%m-%d").date()
            matching_label = next(
                (d for d in day_labels if datetime.strptime(d, "%Y-%m-%d").date().strftime("%m-%d") == hist_date.strftime("%m-%d")),
                None,
            )
            if matching_label and temp is not None:
                normals_by_day[matching_label].append(temp)

    normal_values = []
    years_used_counts = []
    for d in day_labels:
        vals = normals_by_day[d]
        years_used_counts.append(len(vals))
        normal_values.append(sum(vals) / len(vals) if vals else None)

    min_years_used = min(years_used_counts) if years_used_counts else 0
    print(f"TEMP CHART: normal built from {min_years_used}-{max(years_used_counts) if years_used_counts else 0} years of data per day")

    if min_years_used < 15:
        print("TEMP CHART: WARNING — fewer than 15 years of data available for the normal, treat with caution")

    # Render chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_labels = [datetime.strptime(d, "%Y-%m-%d").strftime("%b %d") for d in day_labels]

    ax.plot(x_labels, recent_values, marker="o", linewidth=2, label=f"{current_year} observed", color="#c0392b")
    ax.plot(x_labels, normal_values, marker="o", linewidth=2, linestyle="--", label="1996-2025 average", color="#7f8c8d")

    ax.set_ylabel("Mean daily temperature (°C)")
    ax.set_title("Herschel Island — last 10 days vs. 30-year average")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    png_bytes = fig_to_png_bytes(fig)
    caption = f"Daily mean temperature, last 10 days vs. 30-year (1996-2025) average. Normal computed from {min_years_used}-{max(years_used_counts)} years of ERA5 data per calendar day."
    return png_bytes, caption


temp_chart_bytes, temp_chart_caption = build_temperature_chart()


# =========================================================
# MODULE 1d — THAWING DEGREE DAYS (TDD)
# Cumulative sum of mean daily temps above 0°C, from Jan 1 of the
# current year, compared against the same cumulative curve averaged
# over the past 30 years.
# =========================================================
def compute_tdd_from_series(daily_temp_dict, year_start, up_to_date):
    """
    Given {date_str: temp_c}, computes cumulative thawing degree days
    from year_start through up_to_date (inclusive).
    Days with missing data are skipped (not treated as 0).
    """
    cumulative = 0.0
    d = year_start
    while d <= up_to_date:
        d_str = d.strftime("%Y-%m-%d")
        temp = daily_temp_dict.get(d_str)
        if temp is not None and temp > 0:
            cumulative += temp
        d += timedelta(days=1)
    return cumulative


def build_tdd_comparison():
    end = (now - timedelta(days=1)).date()
    year_start_current = date(end.year, 1, 1)

    current_series = fetch_daily_temps(year_start_current, end)
    if not current_series:
        return "No data available to compute current-year thawing degree days."

    current_tdd = compute_tdd_from_series(current_series, year_start_current, end)

    # 30-year average TDD-to-date (same day-of-year cutoff each year)
    historical_tdds = []
    for years_back in range(1, 31):
        hist_year = end.year - years_back
        hist_start = date(hist_year, 1, 1)
        try:
            hist_end = end.replace(year=hist_year)
        except ValueError:
            # Feb 29 in a non-leap year; fall back to Feb 28
            hist_end = date(hist_year, 2, 28)

        hist_series = fetch_daily_temps(hist_start, hist_end)
        if hist_series:
            historical_tdds.append(compute_tdd_from_series(hist_series, hist_start, hist_end))

    if not historical_tdds:
        return f"Current TDD: {current_tdd:.0f} °C-days (Jan 1 – {end.strftime('%b %d')}). No historical comparison available."

    avg_tdd = sum(historical_tdds) / len(historical_tdds)
    diff = current_tdd - avg_tdd
    direction = "above" if diff > 0 else "below"

    return (
        f"Current TDD ({end.year}, Jan 1 – {end.strftime('%b %d')}): {current_tdd:.0f} °C-days\n"
        f"30-year average TDD for the same period: {avg_tdd:.0f} °C-days\n"
        f"Difference: {abs(diff):.0f} °C-days {direction} average (based on {len(historical_tdds)} years of data)\n"
        f"Source: derived from Open-Meteo ERA5 historical archive"
    )


tdd_text = build_tdd_comparison()


# =========================================================
# MODULE 2 — SATELLITE: MODIS true color via GIBS WMS
# =========================================================
def build_gibs_url(date_str):
    bbox = BBOX_WMS
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "STYLES": "",
        "FORMAT": "image/png",
        "TRANSPARENT": "false",
        "WIDTH": "1024",
        "HEIGHT": "768",
        "SRS": "EPSG:4326",
        "BBOX": bbox,
        "TIME": date_str,
    }
    base = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{base}?{query}"


def fetch_modis_image(max_days_back=5):
    for days_back in range(1, max_days_back + 1):
        date_str = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = build_gibs_url(date_str)
        try:
            resp = requests.get(url, timeout=20)
        except Exception as e:
            print(f"MODIS request failed for {date_str}:", e)
            continue

        content_type = resp.headers.get("Content-Type", "")
        is_real_png = resp.content[:8] == b"\x89PNG\r\n\x1a\n"
        print(f"MODIS {date_str}: HTTP {resp.status_code}, type={content_type}, bytes={len(resp.content)}")

        if resp.status_code == 200 and "image/png" in content_type and is_real_png and len(resp.content) >= 5000:
            return resp.content, date_str
        print("  -> rejected (not a usable image for this date)")

    return None, None


modis_bytes, modis_date = fetch_modis_image()


# =========================================================
# MODULE 3 — SATELLITE: Sentinel-2 via Earth Search STAC API
# (Element84, free, no auth, AWS Open Data)
# =========================================================
def fetch_latest_sentinel2(max_lookback_days=20, max_cloud_cover=70):
    """
    Queries Earth Search STAC for the most recent, least-cloudy Sentinel-2
    scene over Herschel Island, then downloads its visual (true-color)
    thumbnail/preview asset.

    Returns (image_bytes, scene_date_str, cloud_cover) or (None, None, None).
    """
    search_url = "https://earth-search.aws.element84.com/v1/search"
    date_to = now.strftime("%Y-%m-%d")
    date_from = (now - timedelta(days=max_lookback_days)).strftime("%Y-%m-%d")

    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": BBOX_STAC,
        "datetime": f"{date_from}T00:00:00Z/{date_to}T23:59:59Z",
        "limit": 20,
        "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }

    try:
        resp = requests.post(search_url, json=body, timeout=30)
        resp.raise_for_status()
        results = resp.json()
    except Exception as e:
        print("SENTINEL-2 STAC SEARCH FAILED:", e)
        return None, None, None

    features = results.get("features", [])
    print(f"SENTINEL-2: {len(features)} scenes found in lookback window")

    if not features:
        return None, None, None

    # Features are sorted newest-first; take the first usable one
    for feature in features:
        props = feature.get("properties", {})
        scene_date = props.get("datetime", "")[:10]
        cloud_cover = props.get("eo:cloud_cover")

        assets = feature.get("assets", {})
        # Prefer a small rendered preview over the full COG (much faster, and
        # still a real true-color image suitable for a dashboard thumbnail).
        thumb_asset = assets.get("thumbnail") or assets.get("visual")
        if not thumb_asset:
            continue
        thumb_url = thumb_asset.get("href")
        if not thumb_url:
            continue

        try:
            img_resp = requests.get(thumb_url, timeout=30)
            img_resp.raise_for_status()
        except Exception as e:
            print(f"SENTINEL-2 thumbnail download failed for {scene_date}:", e)
            continue

        content = img_resp.content
        # thumbnail asset is usually JPEG; visual COG would be TIFF (skip those,
        # not embeddable directly in Notion without conversion)
        is_jpeg = content[:3] == b"\xff\xd8\xff"
        is_png = content[:8] == b"\x89PNG\r\n\x1a\n"

        if (is_jpeg or is_png) and len(content) > 2000:
            return content, scene_date, cloud_cover
        else:
            print(f"  -> rejected: asset for {scene_date} not a usable JPEG/PNG ({len(content)} bytes)")

    return None, None, None


sentinel_bytes, sentinel_date, sentinel_cloud = fetch_latest_sentinel2()


# =========================================================
# MODULE 4 — SEA ICE CONCENTRATION (OSI SAF)
# =========================================================
# OSI SAF distributes sea ice concentration as daily gridded NetCDF files,
# not via a simple JSON/REST API. We pull the most recent available daily
# file from the OSI SAF Interim CDR (OSI-430-a, fast-track ~2 day latency)
# and extract the value at the grid cell nearest Herschel Island.
#
# Requires: xarray, netCDF4 (added to requirements.txt — see note at bottom
# of this file / README)
def fetch_sea_ice_concentration(max_days_back=10):
    try:
        import xarray as xr
    except ImportError:
        print("SEA ICE: xarray not installed — see requirements.txt note")
        return None, None

    base = "https://thredds.met.no/thredds/dodsC/osisaf/met.no/ice/conc_crb/nh"

    for days_back in range(2, max_days_back + 2):
        # OSI SAF fast-track latency is ~2 days, so start lookback at day 2
        date = now - timedelta(days=days_back)
        # OSI SAF filename convention: ice_conc_nh_polstere-100_multi_YYYYMMDD1200.nc
        date_str = date.strftime("%Y%m%d")
        url = f"{base}/{date.strftime('%Y')}/{date.strftime('%m')}/ice_conc_nh_polstere-100_multi_{date_str}1200.nc"

        try:
            ds = xr.open_dataset(url)
        except Exception as e:
            print(f"SEA ICE: no file for {date_str} ({e})")
            continue

        try:
            sic_value = float(ds["ice_conc"].sel(lat=LAT, lon=LON, method="nearest").values)
            ds.close()
            return sic_value, date.strftime("%Y-%m-%d")
        except Exception as e:
            print(f"SEA ICE: extraction failed for {date_str}:", e)
            ds.close()
            continue

    return None, None


sea_ice_pct, sea_ice_date = fetch_sea_ice_concentration()

if sea_ice_pct is not None:
    sea_ice_text = f"Sea ice concentration: {sea_ice_pct:.1f} %\nDate: {sea_ice_date}\nSource: EUMETSAT OSI SAF (OSI-430-a)"
else:
    sea_ice_text = (
        "Sea ice concentration unavailable for the recent lookback window.\n"
        "This module reads gridded NetCDF files from OSI SAF's THREDDS server; "
        "if this keeps failing, check the Action log — the file naming convention "
        "or server path may have changed, since OSI SAF does not offer a simple REST API."
    )


# =========================================================
# MODULE 5 — TIDES & SEA LEVEL (DFO Canadian Hydrographic Service)
# =========================================================
# DFO's SPINE API takes lat/lon directly (no need to know a station ID ahead
# of time), but its documented coverage is concentrated in southern Canadian
# waters. Arctic coverage near Herschel Island is not guaranteed — this is
# left as a graceful "no data" case rather than assumed to work.
def fetch_tide_level():
    url = "https://api-spine.dfo-mpo.gc.ca/rest/v1/waterLevel"
    params = {
        "lat": LAT,
        "lon": LON,
        "t": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        print("TIDES: raw response:", data)
    except Exception as e:
        print("TIDES FETCH FAILED:", e)
        return None

    items = data.get("responseItems", [])
    if not items:
        return None

    item = items[0]
    if item.get("status") != "OK":
        print("TIDES: status not OK:", item.get("status"))
        return None

    return item.get("waterLevel")


tide_level_m = fetch_tide_level()

if tide_level_m is not None:
    tide_text = f"Water level: {tide_level_m:.2f} m (relative to chart datum)\nSource: DFO/CHS SPINE service"
else:
    tide_text = (
        "No tide/water level data returned for this location.\n"
        "DFO's SPINE service has sparse coverage in the Arctic — the nearest "
        "real gauge may be Tuktoyaktuk or Inuvik rather than Herschel Island itself. "
        "Manual substitution of a specific station ID may be needed; see DFO's "
        "Tidal Station Inventory to find the closest one."
    )


# =========================================================
# UPLOAD ANY VALID IMAGES TO NOTION
# =========================================================
modis_block = None
modis_caption = "No valid MODIS image found in the last 5 days (cloud cover or processing delay)."
if modis_bytes:
    try:
        uid = upload_image_to_notion(modis_bytes, "modis.png")
        modis_block = image_block_from_upload(uid)
        modis_caption = f"MODIS Terra true color — {modis_date}"
    except Exception as e:
        print("MODIS NOTION UPLOAD FAILED:", e)
        modis_caption = "MODIS image found but upload to Notion failed — see Action logs."

sentinel_block = None
sentinel_caption = "No recent low-cloud Sentinel-2 scene found in the lookback window."
if sentinel_bytes:
    try:
        uid = upload_image_to_notion(sentinel_bytes, "sentinel2.png")
        sentinel_block = image_block_from_upload(uid)
        cc_text = f", cloud cover {sentinel_cloud:.0f}%" if sentinel_cloud is not None else ""
        sentinel_caption = f"Sentinel-2 L2A — {sentinel_date}{cc_text}"
    except Exception as e:
        print("SENTINEL-2 NOTION UPLOAD FAILED:", e)
        sentinel_caption = "Sentinel-2 scene found but upload to Notion failed — see Action logs."

temp_chart_block = None
if temp_chart_bytes:
    try:
        uid = upload_image_to_notion(temp_chart_bytes, "temp_chart.png")
        temp_chart_block = image_block_from_upload(uid)
    except Exception as e:
        print("TEMP CHART NOTION UPLOAD FAILED:", e)
        temp_chart_caption = "Chart generated but upload to Notion failed — see Action logs."


# =========================================================
# ASSEMBLE DASHBOARD BLOCKS
# =========================================================
blocks = [
    heading("Herschel Island Environmental Dashboard", level=1),
    paragraph(f"Last update (UTC): {now.strftime('%Y-%m-%d %H:%M')}"),
    divider(),

    heading("🛰 Satellite — MODIS True Color"),
]
if modis_block:
    blocks.append(modis_block)
blocks.append(paragraph(modis_caption))

blocks.append(heading("🛰 Satellite — Sentinel-2 (higher resolution)"))
if sentinel_block:
    blocks.append(sentinel_block)
blocks.append(paragraph(sentinel_caption))

blocks += [
    heading("🌡 Weather"),
    paragraph(weather_text),

    heading("☀️ Sunrise / Sunset"),
    paragraph(sun_text),

    heading("📈 Temperature — last 10 days vs. 30-year average"),
]
if temp_chart_block:
    blocks.append(temp_chart_block)
blocks.append(paragraph(temp_chart_caption if temp_chart_bytes else "Chart could not be generated — see Action logs."))

blocks += [
    heading("🌱 Thawing Degree Days"),
    paragraph(tdd_text),

    heading("🧊 Sea Ice Concentration"),
    paragraph(sea_ice_text),

    heading("🌊 Tides & Sea Level"),
    paragraph(tide_text),

    heading("🧊 Permafrost (boreholes)"),
    paragraph("Placeholder — no live data source configured yet. Add borehole logger endpoint here when available."),

    heading("📷 Time-lapse Cameras"),
    paragraph("Placeholder — no live data source configured yet. Add camera feed/FTP endpoint here when available."),

    heading("🌬 Eddy Covariance Tower"),
    paragraph("Placeholder — no live data source configured yet. Add tower data endpoint here when available."),
]

# =========================================================
# CLEAR PAGE
# =========================================================
existing = notion.blocks.children.list(block_id=PAGE_ID)
print("EXISTING BLOCK COUNT:", len(existing["results"]))

for b in existing["results"]:
    notion.blocks.delete(block_id=b["id"])

# =========================================================
# UPDATE PAGE
# =========================================================
response = notion.blocks.children.append(block_id=PAGE_ID, children=blocks)
print("APPEND RESPONSE BLOCK COUNT:", len(response.get("results", [])))
print("Dashboard updated successfully")

# =========================================================
# NOTE ON DEPENDENCIES
# =========================================================
# requirements.txt should now include, in addition to what you already have:
#   xarray
#   netCDF4
#   notion-client
#   requests
