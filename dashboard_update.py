"""
dashboard_update.py — Shingle Point entrypoint.

This file is intentionally thin: it fetches data (using functions from
dashboard_lib, the shared library checked out alongside this repo by the
GitHub Actions workflow) and assembles the page using this site's
config.py. Almost nothing here is Shingle-Point-specific logic — that
all lives in dashboard_lib.py. What IS specific to this site is:
  - the values in config.py
  - the ORDER and CHOICE of which build_*_section() calls appear below

To add a new section type that doesn't exist yet in dashboard_lib.py
(e.g. a genuinely new kind of chart), add it there, then call it here —
the new section becomes available to every other site for free.

To add or remove an EXISTING section type for this site (e.g. one more
hydrometric station, or skip the tide card because there's no nearby
IWLS station), edit THIS file and/or config.py — never dashboard_lib.py
for a single-site change.
"""

import sys
import os
from datetime import datetime, timezone

# dashboard-lib is checked out as a sibling directory by the workflow
# (see .github/workflows/update.yml's "Checkout shared dashboard-lib"
# step) — add it to the import path before importing from it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard-lib"))

import dashboard_lib as lib
import config

now_utc = datetime.utcnow()
now_local = lib.to_local_time(now_utc, config.TZ_NAME)
print(f"SCRIPT STARTED: {now_utc.isoformat()} UTC")

temp_cache = lib.load_temp_cache()
temp_cache_dirty = False

# =========================================================
# FETCH — GEM/GDPS forecast (primary), Open-Meteo as fallback
# =========================================================
gem_forecast = lib.fetch_gem_forecast(config.LAT, config.LON, now_utc, tz_name=config.TZ_NAME)
if gem_forecast:
    print(f"GEM FORECAST: using source={gem_forecast['source']}")
else:
    print("GEM FORECAST: all sources failed, forecast sections will show fallback text")

# =========================================================
# FETCH — weather, wind, forecast
# =========================================================
weather = lib.get_weather(config.LAT, config.LON)

if weather["status"] == "ok":
    compass = lib.degrees_to_compass(weather["winddirection_deg"])
    wind_dir_text = f"{compass} ({weather['winddirection_deg']:.0f}°)" if compass else "—"
    weather_text = [
        ("Air temperature: ", f"{weather['temperature_c']} °C"),
        ("Humidity: ", f"{weather['humidity_pct']} %"),
        ("Pressure: ", f"{weather['pressure_hpa']} hPa"),
    ]
    weather_source_text = "Source: Open-Meteo (ERA5-based current analysis)"
    wind_now_text = [
        ("Wind speed: ", f"{weather['windspeed_kmh']} km/h"),
        ("Wind direction: ", wind_dir_text),
    ]
    _gem_src = "GDPS (GEM-seamless)" if (gem_forecast and gem_forecast.get("source") == "gem_seamless") else "Open-Meteo (ECMWF fallback)"
    wind_source_text = f"Current conditions: Open-Meteo (ERA5).  48-hour forecast: {_gem_src}."
else:
    weather_text = "Weather data unavailable — fetch failed. Check Action logs."
    weather_source_text = ""
    wind_now_text = "Wind data unavailable — fetch failed. Check Action logs."
    wind_source_text = ""

weather_icon_bytes = lib.render_weather_icon(weather.get("weathercode")) if weather["status"] == "ok" else None
wind_icon_bytes = (
    lib.render_wind_icon(weather["winddirection_deg"], weather["windspeed_kmh"])
    if weather["status"] == "ok" and weather.get("winddirection_deg") is not None and weather.get("windspeed_kmh") is not None
    else None
)
weather_icon_big_bytes = (
    lib.render_icon_with_big_number(
        weather_icon_bytes, lib.fmt_temp(weather['temperature_c']), "°C",
        number_color=lib.temperature_to_color(weather["temperature_c"]),
    )
    if weather_icon_bytes and weather.get("temperature_c") is not None
    else weather_icon_bytes
)
_wind_color, _beaufort_label = lib.windspeed_to_beaufort_color(weather.get("windspeed_kmh"))
if weather["status"] == "ok" and isinstance(wind_now_text, list):
    wind_now_text.append(("Beaufort force: ", _beaufort_label))
wind_icon_big_bytes = (
    lib.render_icon_with_big_number(
        wind_icon_bytes, f"{weather['windspeed_kmh']:.0f}", "km/h",
        number_color=_wind_color,
    )
    if wind_icon_bytes and weather.get("windspeed_kmh") is not None
    else wind_icon_bytes
)

weather_icon_block = None
if weather_icon_big_bytes:
    try:
        uid = lib.upload_image_to_notion(weather_icon_big_bytes, "weather_icon.png")
        weather_icon_block = lib.image_block_from_upload(uid)
    except Exception as e:
        print("WEATHER ICON NOTION UPLOAD FAILED:", e)

wind_icon_block = None
if wind_icon_big_bytes:
    try:
        uid = lib.upload_image_to_notion(wind_icon_big_bytes, "wind_icon.png")
        wind_icon_block = lib.image_block_from_upload(uid)
    except Exception as e:
        print("WIND ICON NOTION UPLOAD FAILED:", e)

# Forecast strips: GEM daily → strip format, fall back to Open-Meteo
mini_forecast_strip_block = None
large_forecast_strip_bytes = None
land_forecast_caption = "Land forecast unavailable — fetch failed. Check Action logs."

if gem_forecast:
    land_forecast_days = lib.gem_daily_to_land_forecast_days(gem_forecast["daily"])
    land_forecast_source = "GDPS (GEM-seamless)" if gem_forecast["source"] == "gem_seamless" else "Open-Meteo (ECMWF fallback)"
else:
    land_forecast_days = lib.get_land_forecast(config.LAT, config.LON)
    land_forecast_source = "Open-Meteo"

if land_forecast_days:
    mini_strip_days = []
    for d in land_forecast_days[:5]:
        day_compass = lib.degrees_to_compass(d["wind_dir_deg"])
        wind_label = f"{d['wind_max_kmh']:.0f} km/h {day_compass or ''}".strip()
        precip_label = f"{d['precip_mm']:.1f} mm" + (f" ({d['precip_prob_pct']:.0f}%)" if d.get("precip_prob_pct") is not None else "")
        mini_strip_days.append({
            "day_label": datetime.strptime(d["date"], "%Y-%m-%d").strftime("%a"),
            "weathercode": d["weathercode"],
            "temp_min": d["temp_min"],
            "temp_max": d["temp_max"],
            "wind_label": wind_label,
            "precip_label": precip_label,
        })
    mini_forecast_strip_bytes = lib.build_mini_forecast_strip(mini_strip_days)
    large_forecast_strip_bytes = lib.build_large_forecast_strip(mini_strip_days)
    land_forecast_caption = f"Source: {land_forecast_source}"
    if mini_forecast_strip_bytes:
        try:
            uid = lib.upload_image_to_notion(mini_forecast_strip_bytes, "mini_forecast_strip.png")
            mini_forecast_strip_block = lib.image_block_from_upload(uid)
        except Exception as e:
            print("MINI FORECAST STRIP NOTION UPLOAD FAILED:", e)

# Wind forecast mini chart: prefer GEM hourly, fall back to Open-Meteo
gem_wind_hourly = lib.gem_hourly_wind_forecast(gem_forecast["hourly"], now_utc, tz_name=config.TZ_NAME) if gem_forecast else None
wind_forecast_chart_bytes, wind_forecast_chart_caption = lib.build_wind_forecast_mini_chart(
    gem_wind_hourly if gem_wind_hourly else (weather.get("hourly_wind_forecast") if weather["status"] == "ok" else None)
)
wind_forecast_chart_block = None
if wind_forecast_chart_bytes:
    try:
        uid = lib.upload_image_to_notion(wind_forecast_chart_bytes, "wind_forecast_chart.png")
        wind_forecast_chart_block = lib.image_block_from_upload(uid)
    except Exception as e:
        print("WIND FORECAST CHART NOTION UPLOAD FAILED:", e)

# =========================================================
# FETCH — marine forecast, weather alerts
# =========================================================
marine_entries = lib.get_marine_forecast(config.MARINE_ZONE_ID)
marine_text, marine_source_text = lib.format_marine_forecast_text(marine_entries, config.MARINE_ZONE_NAME)

weather_alert_entries = lib.get_weather_alerts(config.LAT, config.LON)
active_alerts = lib.filter_active_alerts(weather_alert_entries)
if active_alerts:
    print(f"WEATHER ALERTS: {len(active_alerts)} active alert(s) found")
else:
    print("WEATHER ALERTS: none active (or fetch failed) — section will be hidden")

# =========================================================
# FETCH — sun
# =========================================================
sun_info = lib.get_sun_info(config.LAT, config.LON)
sun_text = lib.classify_sun_text(sun_info, config.LAT, config.LON, now_utc, config.TZ_NAME)
sun_chart_bytes, sun_chart_caption = lib.build_sun_curve_chart(config.LAT, config.LON, now_utc, now_local, config.TZ_NAME)

# =========================================================
# FETCH — temperature chart, TDD histogram (30-year historical prefetch)
# =========================================================
print("STARTING: temperature chart (30-year historical prefetch)")
temp_chart_bytes, temp_chart_caption = lib.build_temperature_chart(config.LAT, config.LON, now_utc, temp_cache)

print("STARTING: TDD histogram (25-year historical prefetch)")
tdd_histogram_bytes, tdd_histogram_caption = lib.build_tdd_histogram(config.LAT, config.LON, now_utc, temp_cache)

temp_cache_dirty = True  # the prefetch calls above mutate temp_cache in place when new years are fetched

# =========================================================
# FETCH — wind vector chart
# =========================================================
wind_chart_bytes, wind_rose_bytes, wind_chart_caption = lib.build_wind_charts_combined(config.LAT, config.LON, now_utc)

# =========================================================
# FETCH — logo
# =========================================================
logo_png_bytes = lib.fetch_and_convert_logo_to_png(config.LOGO_URL) if config.LOGO_URL else None

# =========================================================
# FETCH — tides (DFO IWLS)
# =========================================================
station_id = lib.find_iwls_station_id(config.TIDE_STATION_CODE)
tide_points = lib.fetch_tide_predictions(station_id, now_utc, hours_ahead=24 * 7) if station_id else None
tide_text = lib.format_tide_text(tide_points, now_utc, config.TIDE_STATION_CODE, config.TIDE_STATION_NAME)
tide_chart_bytes, tide_chart_caption = lib.build_tide_chart(tide_points, now_utc, config.TZ_NAME)

# =========================================================
# PARALLEL FETCH — MODIS, total water level, Sentinel-1, hydrometric
# stations (all independent, slow, network-bound — run concurrently)
# =========================================================
from concurrent.futures import ThreadPoolExecutor

print("STARTING: parallel fetch of MODIS, water level, Sentinel-1, and hydrometric stations")
with ThreadPoolExecutor(max_workers=3 + len(config.HYDROMETRIC_STATIONS)) as executor:
    modis_future = executor.submit(
        lib.fetch_and_process_modis,
        bbox_3413=config.MODIS_BBOX_3413, center_x=config.MODIS_CENTER_X, center_y=config.MODIS_CENTER_Y,
        rotation_deg=config.MODIS_ROTATION_DEG, points=config.MAP_POINTS, now_utc=now_utc,
        tz_name=config.TZ_NAME, reference_lines=config.MAP_REFERENCE_LINES,
    )
    water_level_future = executor.submit(
        lib.fetch_copernicus_water_level,
        lat=config.LAT, lon=config.LON, now_utc=now_utc, site_label=config.SITE_DISPLAY_NAME,
        yearly_mean=config.WATER_LEVEL_YEARLY_MEAN,
    )
    sentinel1_future = executor.submit(
        lib.fetch_and_process_sentinel1,
        lat=config.LAT, lon=config.LON, site_label=config.SITE_DISPLAY_NAME,
        utm_zone=config.UTM_ZONE, utm_epsg=config.UTM_EPSG,
        center_x=config.UTM_CENTER_X, center_y=config.UTM_CENTER_Y,
        points=config.MAP_POINTS, tz_name=config.TZ_NAME,
        reference_lines=config.MAP_REFERENCE_LINES,
        coastline_geojson_path=config.COASTLINE_GEOJSON_PATH, now_utc=now_utc,
    )
    hydrometric_futures = [
        (station, executor.submit(lib.fetch_hydrometric_water_level, station["station_id"], station["provterr"]))
        for station in config.HYDROMETRIC_STATIONS
    ]

    try:
        modis_bytes, modis_date = modis_future.result()
    except Exception as e:
        print("MODIS PARALLEL FETCH FAILED:", e)
        modis_bytes, modis_date = None, None

    try:
        copernicus_times, copernicus_values, copernicus_yearly_mean = water_level_future.result()
    except Exception as e:
        print("WATER LEVEL PARALLEL FETCH FAILED:", e)
        copernicus_times, copernicus_values, copernicus_yearly_mean = None, None, None

    try:
        sentinel1_bytes, sentinel1_caption = sentinel1_future.result()
    except Exception as e:
        print("SENTINEL-1 PARALLEL FETCH FAILED:", e)
        sentinel1_bytes = None
        sentinel1_caption = "Sentinel-1 SAR image unavailable — fetch failed. Check Action logs."

    hydrometric_results = []
    for station, future in hydrometric_futures:
        try:
            h_times, h_values = future.result()
        except Exception as e:
            print(f"HYDROMETRIC[{station['station_id']}] PARALLEL FETCH FAILED:", e)
            h_times, h_values = None, None
        hydrometric_results.append((station, h_times, h_values))

modis_block, _ = lib._upload_chart_or_caption(modis_bytes, "modis.png", None)
modis_caption = f"NASA MODIS Terra, true color, {modis_date}." if modis_date else "MODIS image unavailable."

water_level_chart_bytes, water_level_chart_caption = lib.build_water_level_chart(
    copernicus_times, copernicus_values, config.TZ_NAME, copernicus_yearly_mean,
)
water_level_text = (
    [("Latest forecast value: ", f"{copernicus_values[0]:.2f} m (above geoid, TOPAZ6 model reference level)")] if copernicus_values
    else "Total water level forecast unavailable — fetch failed. Check Action logs."
)

# =========================================================
# ASSEMBLE PAGE
# This is the part that varies per site: which sections, in what order.
# To add/remove a block for THIS site only, edit the list below — no
# changes to dashboard_lib.py or config.py needed for that alone.
# =========================================================
blocks = []
blocks += lib.build_header_blocks(now_local, logo_url=config.LOGO_URL, logo_png_bytes=logo_png_bytes,
                                    institution_text=config.INSTITUTION_TEXT, tz_name=config.TZ_NAME)
blocks += lib.build_todays_conditions_section(
    weather_text, weather_source_text, weather_icon_block, mini_forecast_strip_block,
    config.LAT, config.LON, wind_now_text, wind_source_text, wind_icon_block, wind_forecast_chart_block,
    tide_text, tide_chart_bytes, tide_chart_caption, config.TIDE_STATION_CODE,
    sun_text, sun_chart_bytes, sun_chart_caption,
)
blocks += lib.build_active_alerts_section(active_alerts)
blocks += lib.build_gem_forecast_section(gem_forecast, config.TZ_NAME, now_utc=now_utc)
blocks += lib.build_marine_forecast_section(marine_text, marine_source_text, config.MARINE_ZONE_NAME, config.MARINE_ZONE_ID)
blocks += lib.build_total_water_level_section(water_level_text, water_level_chart_bytes, water_level_chart_caption)
for station, h_times, h_values in hydrometric_results:
    h_chart_bytes, h_chart_caption = lib.build_hydrometric_chart(h_times, h_values, station["station_id"], station["river_name"])
    blocks += lib.build_hydrometric_section(h_chart_bytes, h_chart_caption, station["heading"])
blocks += lib.build_modis_section(modis_block, modis_caption, modis_date, now_utc, config.MODIS_BBOX_3413, config.SITE_DISPLAY_NAME)
sentinel1_explore_url = f"https://apps.sentinel-hub.com/eo-browser/?zoom=11&lat={config.LAT}&lng={config.LON}&themeId=DEFAULT-THEME"
blocks += lib.build_sentinel1_section(sentinel1_bytes, sentinel1_caption, sentinel1_explore_url, config.SITE_DISPLAY_NAME)
blocks += lib.build_temperature_chart_section(temp_chart_bytes, temp_chart_caption)
blocks += lib.build_tdd_histogram_section(tdd_histogram_bytes, tdd_histogram_caption)
blocks += lib.build_wind_chart_section(wind_chart_bytes, wind_chart_caption, rose_bytes=wind_rose_bytes)
blocks += lib.build_disclaimer_section()

lib.publish_blocks_to_notion(blocks)

if temp_cache_dirty:
    lib.save_temp_cache(temp_cache)
    print(f"CACHE: saved {len(temp_cache)} years to {lib.CACHE_FILE_PATH}")
else:
    print("CACHE: no new years added this run, skipping save")
