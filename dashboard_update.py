
import os
import io
import math
import requests
from datetime import datetime, timedelta, date, timezone
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
 
now = datetime.utcnow()
 
 
# =========================================================
# HELPERS — Notion block builders (kept tiny to reduce repetition)
# =========================================================
def heading(text, level=2):
    tag = f"heading_{level}"
    return {"object": "block", "type": tag, tag: {"rich_text": [{"type": "text", "text": {"content": text}}]}}
 
 
 
def divider():
    return {"object": "block", "type": "divider", "divider": {}}
 
 
def _line_to_segments(line):
    """
    Converts one line specification into a list of rich_text segments
    (no trailing newline added here — that's handled by the caller).
 
    A line can be:
      - a plain string: rendered as-is, no bolding.
      - a (label, value) tuple: label in normal text, value in bold.
      - a list of strings/tuples: each rendered in sequence on the same
        line, mixing plain and bold segments freely. Use this when a
        single line has several distinct values, e.g. a forecast day with
        a temperature range AND a wind speed on one line.
    """
    if isinstance(line, list):
        segments = []
        for piece in line:
            segments.extend(_line_to_segments(piece))
        return segments
 
    if isinstance(line, tuple):
        label, value = line
        segments = []
        if label:
            segments.append({"type": "text", "text": {"content": label}})
        segments.append({
            "type": "text",
            "text": {"content": str(value)},
            "annotations": {"bold": True},
        })
        return segments
 
    return [{"type": "text", "text": {"content": line}}]
 
 
def build_bolded_lines(lines):
    """
    Builds a single rich_text array from a list of lines (see
    _line_to_segments for what a line can be). Lines are separated by a
    newline appended to the end of the last segment of the previous line,
    rather than as a standalone segment, since a lone "\\n"-only text
    object with no preceding content can be dropped by Notion in practice.
    """
    segments = []
    for i, line in enumerate(lines):
        if i > 0 and segments:
            segments[-1]["text"]["content"] += "\n"
        segments.extend(_line_to_segments(line))
    return segments
 
 
def callout(lines, emoji="📌", color="gray_background", children=None):
    """
    Builds a callout block from a list of lines (see build_bolded_lines).
    Accepts either a plain string (backward compatible, no bolding) or a
    list of strings/tuples/mixed-segment-lists for selective bolding.
 
    children: optional list of child blocks (e.g. an image block) to nest
    inside the callout, placed below the text. Notion callouts support
    nested child blocks the same way column blocks do — the children must
    be included in the same create call, not patched in afterward.
    """
    if isinstance(lines, str):
        lines = [lines]
    callout_obj = {
        "rich_text": build_bolded_lines(lines),
        "icon": {"type": "emoji", "emoji": emoji},
        "color": color,
    }
    if children:
        callout_obj["children"] = children
    return {
        "object": "block",
        "type": "callout",
        "callout": callout_obj,
    }
 
 
def paragraph(lines):
    """
    Builds a paragraph block from a list of lines (see build_bolded_lines).
    Accepts either a plain string (backward compatible) or a list for
    selective bolding.
    """
    if isinstance(lines, str):
        lines = [lines]
    return {"object": "block", "paragraph": {"rich_text": build_bolded_lines(lines)}, "type": "paragraph"}
 
 
def link_paragraph(label, url):
    """Paragraph block containing a clickable link."""
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": label, "link": {"url": url}}}]
        },
    }
 
 
def columns(*column_block_lists):
    """
    Builds a column_list block with N columns, each containing the given
    list of child blocks. Notion requires all column content to be created
    in the same request as the column_list itself — content cannot be
    patched into columns afterward the way top-level blocks can.
    """
    return {
        "object": "block",
        "type": "column_list",
        "column_list": {
            "children": [
                {"object": "block", "type": "column", "column": {"children": blocks}}
                for blocks in column_block_lists
            ]
        },
    }
 
 
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
def degrees_to_compass(deg):
    """Converts wind direction in degrees to a 16-point compass label."""
    if deg is None:
        return None
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return directions[idx]
 
 
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
        current_time = cw["time"]  # e.g. "2026-06-23T07:15" — can be off the hour
 
        # current_weather's timestamp can fall on a quarter-hour, but the
        # hourly arrays are always on the hour — round down before matching,
        # or exact-match .index() fails whenever the minutes aren't ":00".
        current_dt = datetime.strptime(current_time, "%Y-%m-%dT%H:%M")
        current_hour = current_dt.replace(minute=0).strftime("%Y-%m-%dT%H:%M")
 
        humidity = None
        pressure = None
        try:
            idx = data["hourly"]["time"].index(current_hour)
            humidity = data["hourly"]["relativehumidity_2m"][idx]
            pressure = data["hourly"]["pressure_msl"][idx]
        except (ValueError, KeyError, IndexError) as e:
            print("WEATHER: could not align hourly index:", e)
 
        return {
            "temperature_c": cw.get("temperature"),
            "windspeed_kmh": cw.get("windspeed"),
            "winddirection_deg": cw.get("winddirection"),
            "weathercode": cw.get("weathercode"),
            "humidity_pct": humidity,
            "pressure_hpa": pressure,
            "status": "ok",
        }
    except Exception as e:
        print("WEATHER FETCH FAILED:", e)
        return {
            "temperature_c": None,
            "windspeed_kmh": None,
            "winddirection_deg": None,
            "weathercode": None,
            "humidity_pct": None,
            "pressure_hpa": None,
            "status": "missing",
        }
 
 
weather = get_weather()
 
if weather["status"] == "ok":
    compass = degrees_to_compass(weather["winddirection_deg"])
    wind_dir_text = f"{compass} ({weather['winddirection_deg']:.0f}°)" if compass else "—"
    weather_text = [
        ("Air temperature: ", f"{weather['temperature_c']} °C"),
        ("Wind speed: ", f"{weather['windspeed_kmh']} km/h"),
        ("Wind direction: ", wind_dir_text),
        ("Humidity: ", f"{weather['humidity_pct']} %"),
        ("Pressure: ", f"{weather['pressure_hpa']} hPa"),
        "Source: Open-Meteo (ERA5-based forecast/analysis)",
    ]
else:
    weather_text = "Weather data unavailable — fetch failed. Check Action logs."
 
 
# =========================================================
# MODULE 1a-1b — WEATHER PICTOGRAM
# Draws a simple icon matching the current WMO weathercode (returned by
# Open-Meteo's current_weather) rather than depending on an external icon
# server staying available — same reasoning as the MODIS image annotation:
# self-contained drawing is more robust than a third-party image URL.
#
# WMO weathercode reference (subset relevant to Arctic conditions):
# 0-1: clear/mainly clear, 2: partly cloudy, 3: cloudy, 45/48: fog,
# 51-57: drizzle, 61-67: rain, 71-77: snow, 80-82: showers,
# 85-86: snow showers, 95-99: thunderstorm
# =========================================================
NOTION_ICON_YELLOW = (231, 179, 71)
NOTION_ICON_GRAY = (155, 154, 151)
NOTION_ICON_DARK_GRAY = (120, 119, 116)
NOTION_ICON_BLUE = (51, 126, 169)
NOTION_ICON_LIGHT_GRAY = (227, 226, 224)
NOTION_ICON_WHITE = (255, 255, 255)
 
 
def _icon_draw_sun(draw, cx, cy, r=18, color=None):
    color = color or NOTION_ICON_YELLOW
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    for i in range(8):
        angle = i * math.pi / 4
        x1 = cx + math.cos(angle) * (r + 4)
        y1 = cy + math.sin(angle) * (r + 4)
        x2 = cx + math.cos(angle) * (r + 11)
        y2 = cy + math.sin(angle) * (r + 11)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=4)
 
 
def _icon_draw_cloud(draw, cx, cy, scale=1.0, color=None):
    color = color or NOTION_ICON_LIGHT_GRAY
    r = 16 * scale
    bumps = [(-1.5, 0.1, 0.85), (-0.6, -0.5, 1.0), (0.4, -0.4, 1.05),
             (1.3, 0.1, 0.8), (-0.9, 0.35, 0.9), (0.9, 0.35, 0.9)]
    for dx, dy, s in bumps:
        rr = r * s
        draw.ellipse([cx + dx * r - rr, cy + dy * r - rr, cx + dx * r + rr, cy + dy * r + rr], fill=color)
 
 
def _icon_partly_cloudy(draw, cx, cy):
    _icon_draw_sun(draw, cx - 10, cy - 10, r=13)
    _icon_draw_cloud(draw, cx + 8, cy + 6, scale=0.85)
 
 
def _icon_rain(draw, cx, cy, heavy=False):
    _icon_draw_cloud(draw, cx, cy, color=NOTION_ICON_DARK_GRAY if heavy else NOTION_ICON_GRAY)
    offsets = [-18, -6, 6, 18] if heavy else [-14, 0, 14]
    for dx in offsets:
        draw.line([(cx + dx, cy + 22), (cx + dx - 4, cy + 36)], fill=NOTION_ICON_BLUE, width=3)
 
 
def _icon_snow(draw, cx, cy):
    _icon_draw_cloud(draw, cx, cy, color=NOTION_ICON_LIGHT_GRAY)
    for dx in [-14, 0, 14]:
        for dy in [26, 38]:
            r = 2.5
            draw.ellipse([cx + dx - r, cy + dy - r, cx + dx + r, cy + dy + r],
                         fill=NOTION_ICON_WHITE, outline=NOTION_ICON_GRAY, width=1)
 
 
def _icon_fog(draw, cx, cy):
    for i, dy in enumerate([-10, 2, 14, 26]):
        w = 28 - i * 1.5
        draw.line([(cx - w, cy + dy), (cx + w, cy + dy)], fill=NOTION_ICON_GRAY, width=4)
 
 
def _icon_thunder(draw, cx, cy):
    _icon_draw_cloud(draw, cx, cy, color=NOTION_ICON_DARK_GRAY)
    pts = [(cx - 4, cy + 18), (cx + 6, cy + 18), (cx - 2, cy + 34), (cx + 8, cy + 34), (cx - 6, cy + 50)]
    draw.line(pts, fill=NOTION_ICON_YELLOW, width=3, joint="curve")
 
 
def render_weather_icon(weathercode):
    """
    Renders a small PNG icon matching the given WMO weathercode.
    Returns PNG bytes, or None if rendering fails for any reason (so a
    drawing bug never blocks the rest of the dashboard from updating).
    """
    try:
        from PIL import Image, ImageDraw
        import io as _io
 
        size = 100
        img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        cx, cy = size // 2, size // 2 - 5
 
        code = weathercode if weathercode is not None else -1
 
        if code in (0, 1):
            _icon_draw_sun(draw, cx, cy)
        elif code == 2:
            _icon_partly_cloudy(draw, cx, cy)
        elif code == 3:
            _icon_draw_cloud(draw, cx, cy)
        elif code in (45, 48):
            _icon_fog(draw, cx, cy)
        elif code in (51, 53, 55, 56, 57):
            _icon_rain(draw, cx, cy, heavy=False)
        elif code in (61, 63, 65, 66, 67):
            _icon_rain(draw, cx, cy, heavy=(code in (65, 67)))
        elif code in (71, 73, 75, 77):
            _icon_snow(draw, cx, cy)
        elif code in (80, 81, 82):
            _icon_rain(draw, cx, cy, heavy=(code == 82))
        elif code in (85, 86):
            _icon_snow(draw, cx, cy)
        elif code in (95, 96, 99):
            _icon_thunder(draw, cx, cy)
        else:
            # Unrecognized code: fall back to a plain cloud rather than
            # guessing, since an unknown code shouldn't be shown as sunny.
            _icon_draw_cloud(draw, cx, cy)
 
        out_buf = _io.BytesIO()
        img.save(out_buf, format="PNG")
        return out_buf.getvalue()
 
    except Exception as e:
        print("WEATHER ICON RENDER FAILED:", e)
        return None
 
 
weather_icon_bytes = render_weather_icon(weather.get("weathercode")) if weather["status"] == "ok" else None
 
 
# =========================================================
# MODULE 1a-2 — LAND WEATHER FORECAST (next 5 days)
# Extends the existing Open-Meteo call with a daily forecast block —
# separate API parameters (daily=...) from current_weather, so this is
# an additional, independent request rather than reusing the same payload.
# =========================================================
def get_land_forecast(days=5):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LAT,
            "longitude": LON,
            "daily": "temperature_2m_max,temperature_2m_min,windspeed_10m_max,winddirection_10m_dominant,precipitation_sum",
            "forecast_days": days,
            "timezone": "UTC",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        daily = data.get("daily", {})
 
        days_list = []
        for i, day_str in enumerate(daily.get("time", [])):
            days_list.append({
                "date": day_str,
                "temp_max": daily["temperature_2m_max"][i],
                "temp_min": daily["temperature_2m_min"][i],
                "wind_max_kmh": daily["windspeed_10m_max"][i],
                "wind_dir_deg": daily["winddirection_10m_dominant"][i],
                "precip_mm": daily["precipitation_sum"][i],
            })
        return days_list
    except Exception as e:
        print("LAND FORECAST FETCH FAILED:", e)
        return []
 
 
land_forecast_days = get_land_forecast()
 
if land_forecast_days:
    lines = []
    for d in land_forecast_days:
        day_label = datetime.strptime(d["date"], "%Y-%m-%d").strftime("%a %b %d")
        compass = degrees_to_compass(d["wind_dir_deg"])
        # Each day is ONE line containing multiple bold value segments
        # mixed with plain connecting text (commas, "wind up to", etc).
        lines.append([
            f"{day_label}: ",
            ("", f"{d['temp_min']:.0f}–{d['temp_max']:.0f} °C"),
            ", wind up to ",
            ("", f"{d['wind_max_kmh']:.0f} km/h {compass or ''}".strip()),
            ", precip ",
            ("", f"{d['precip_mm']:.1f} mm"),
        ])
    land_forecast_text = lines + ["Source: Open-Meteo"]
else:
    land_forecast_text = "Land forecast unavailable — fetch failed. Check Action logs."
 
 
# =========================================================
# MODULE 1a-3 — MARINE FORECAST (Environment Canada, Yukon Coast)
# Source: Environment Canada Atom feed for marine zone 16000, which
# covers Herschel Island / Yukon Coast. The feed returns natural-language
# forecast text per period (e.g. "Wind light becoming southeast 15 knots"),
# not structured numeric fields, so we display the text as published
# rather than trying to parse specific values out of free-form wording.
# =========================================================
def get_marine_forecast():
    try:
        import xml.etree.ElementTree as ET
 
        url = "https://weather.gc.ca/rss/marine/16000_e.xml"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
 
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.content)
 
        entries = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            title = title_el.text if title_el is not None else ""
            summary = summary_el.text if summary_el is not None else ""
            entries.append({"title": title, "summary": summary})
        return entries
    except Exception as e:
        print("MARINE FORECAST FETCH FAILED:", e)
        return []
 
 
marine_entries = get_marine_forecast()
 
if marine_entries:
    # The feed mixes forecast periods with warnings/synopsis entries; show
    # the first several as-is, since titles already summarize each one
    # (e.g. "Wind", "Waves", "Extended Forecast", "Ice Forecast"). We bold
    # the section title (a clean label) but leave the forecaster's
    # free-form prose unbolded — there's no single "value" to highlight in
    # a sentence like "Wind light becoming southeast 15 knots", and trying
    # to extract just the number would risk mangling the wording.
    lines = []
    for e in marine_entries[:6]:
        title = e["title"].strip()
        summary = e["summary"].strip() if e["summary"] else ""
        if summary and summary != title:
            lines.append([("", title), ": ", summary])
        else:
            lines.append(title)
    marine_text = lines + ["Source: Environment Canada (Yukon Coast marine zone)"]
else:
    marine_text = "Marine forecast unavailable — fetch failed. Check Action logs."
 
 
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
        sun_text = [
            ("Day length: ", f"~{hours}h {minutes}min"),
            "Consistent with continuous daylight (midnight sun) at this latitude.",
            "Source: sunrise-sunset.org",
        ]
    elif day_length_s <= 30 * 60:
        sun_text = [
            ("Day length: ", f"~{hours}h {minutes}min"),
            "Consistent with polar night at this latitude.",
            "Source: sunrise-sunset.org",
        ]
    else:
        sun_text = [
            ("Sunrise (UTC): ", sun_info['sunrise'].strftime('%H:%M')),
            ("Sunset (UTC): ", sun_info['sunset'].strftime('%H:%M')),
            ("Day length: ", f"{hours}h {minutes}min"),
            "Source: sunrise-sunset.org",
        ]
elif sun_info["status"] == "no_data":
    sun_text = f"Sun data unavailable ({sun_info.get('raw_status')}) — may be a polar-day/polar-night edge case the API can't resolve at this latitude."
else:
    sun_text = "Sun data fetch failed — check Action logs."
 
 
# =========================================================
# MODULE 1b-2 — SUN POSITION CURVE (solar elevation over 24h)
# Computed directly with the standard solar elevation formula (declination
# + hour angle), not pulled from an external API — this is well-documented
# astronomical math accurate to a fraction of a degree, which is more than
# enough for a dashboard chart, and avoids depending on another service.
# At this latitude the curve naturally shows polar day (never crosses
# zero) or polar night (always negative) without any special-case logic —
# the same formula handles both automatically.
# =========================================================
def solar_elevation_deg(lat_deg, lon_deg, dt_utc):
    lat = math.radians(lat_deg)
    day_of_year = dt_utc.timetuple().tm_yday
    hour_utc = dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600
 
    decl = math.radians(23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81))))
    b = math.radians(360 / 365 * (day_of_year - 81))
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
 
    time_correction = 4 * lon_deg + eot  # minutes
    solar_time = hour_utc + time_correction / 60
    hour_angle = math.radians(15 * (solar_time - 12))
 
    elevation = math.asin(
        math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    )
    return math.degrees(elevation)
 
 
def build_sun_curve_chart():
    """
    Renders a 24-hour solar elevation curve (today, UTC) for Herschel
    Island, with the current moment marked. Returns (png_bytes, caption).
    """
    try:
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        times = [day_start + timedelta(minutes=15 * i) for i in range(96)]
        elevations = [solar_elevation_deg(LAT, LON, t) for t in times]
        hour_floats = [t.hour + t.minute / 60 for t in times]
        current_elevation = solar_elevation_deg(LAT, LON, now)
        current_hour_float = now.hour + now.minute / 60
 
        NOTION_YELLOW = "#E7B347"
        NOTION_RED = "#E16259"
        NOTION_TEXT_GRAY = "#787774"
        NOTION_LIGHT_GRID = "#EDECEC"
        NOTION_HORIZON = "#D4A72C"
 
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
 
        ax.fill_between(hour_floats, elevations, 0, where=[e > 0 for e in elevations],
                         color=NOTION_YELLOW, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(hour_floats, elevations, linewidth=2.5, color=NOTION_YELLOW, zorder=2)
        ax.axhline(0, color=NOTION_HORIZON, linewidth=1.2, alpha=0.6, zorder=1)
 
        ax.plot([current_hour_float], [current_elevation], marker="o", markersize=8,
                 color=NOTION_RED, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
 
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(NOTION_LIGHT_GRID)
 
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=10, color=NOTION_TEXT_GRAY)
        ax.tick_params(axis="y", labelsize=10, colors=NOTION_TEXT_GRAY, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=NOTION_LIGHT_GRID, linewidth=1, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_ylabel("Elevation (°)", fontsize=10, color=NOTION_TEXT_GRAY)
 
        fig.tight_layout()
        png_bytes = fig_to_png_bytes(fig)
 
        max_elev = max(elevations)
        min_elev = min(elevations)
        if min_elev > 0:
            note = "Sun stays above the horizon all day (midnight sun)."
        elif max_elev < 0:
            note = "Sun stays below the horizon all day (polar night)."
        else:
            note = "Sun crosses the horizon today."
        caption = f"Solar elevation today (UTC), {day_start.strftime('%b %d')}. {note} Computed from standard solar position formulas, not measured."
        return png_bytes, caption
 
    except Exception as e:
        print("SUN CURVE CHART FAILED:", e)
        return None, "Sun position chart could not be generated — see Action logs."
 
 
sun_chart_bytes, sun_chart_caption = build_sun_curve_chart()
 
 
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
 
    # Render chart — styled to read more like a clean Notion-native graphic
    # than a default matplotlib plot: no box border, light horizontal-only
    # gridlines, soft shaded band for the historical normal (instead of a
    # second competing line), and a muted, Notion-like color palette.
    NOTION_TEXT_GRAY = "#787774"
    NOTION_BLUE = "#337EA9"
    NOTION_RED = "#E16259"
    NOTION_LIGHT_GRID = "#EDECEC"
 
    plt.rcParams["font.family"] = "DejaVu Sans"
 
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
 
    x_labels = [datetime.strptime(d, "%Y-%m-%d").strftime("%b %d") for d in day_labels]
    x = range(len(day_labels))
 
    # Shaded band: from the lowest to the highest observed normal across the
    # window, giving a sense of "typical range" rather than a single line
    # competing visually with the current-year line.
    ax.fill_between(x, [v - 1.5 if v is not None else math.nan for v in normal_values],
                     [v + 1.5 if v is not None else math.nan for v in normal_values],
                     color=NOTION_BLUE, alpha=0.10, linewidth=0, zorder=1)
    ax.plot(x, normal_values, linewidth=1.5, color=NOTION_BLUE, alpha=0.55,
             label="1996–2025 average", zorder=2)
 
    ax.plot(x, recent_values, marker="o", markersize=5, linewidth=2.5,
             color=NOTION_RED, label=f"{current_year} observed",
             markerfacecolor="white", markeredgewidth=1.5, markeredgecolor=NOTION_RED, zorder=3)
 
    # Remove chart border entirely except a faint baseline
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(NOTION_LIGHT_GRID)
 
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, fontsize=10, color=NOTION_TEXT_GRAY)
    ax.tick_params(axis="y", labelsize=10, colors=NOTION_TEXT_GRAY, length=0)
    ax.tick_params(axis="x", length=0)
 
    ax.yaxis.grid(True, color=NOTION_LIGHT_GRID, linewidth=1, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
 
    ax.set_ylabel("°C", fontsize=10, color=NOTION_TEXT_GRAY)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=NOTION_TEXT_GRAY)
 
    fig.tight_layout()
 
    png_bytes = fig_to_png_bytes(fig)
    caption = f"Daily mean temperature, last 10 days vs. 30-year (1996-2025) average (shaded band ±1.5°C). Normal computed from {min_years_used}-{max(years_used_counts)} years of ERA5 data per calendar day."
    return png_bytes, caption
 
 
temp_chart_bytes, temp_chart_caption = build_temperature_chart()
 
 
# =========================================================
# MODULE 2 — SATELLITE: MODIS true color via GIBS WMS
# =========================================================
MODIS_WIDTH_PX = 1024
MODIS_HEIGHT_PX = 768
 
 
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
        "WIDTH": str(MODIS_WIDTH_PX),
        "HEIGHT": str(MODIS_HEIGHT_PX),
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
 
 
def annotate_modis_image(png_bytes, points=None, scale_km=50):
    """
    Draws label markers at the given coordinates and a scale bar on the
    MODIS image. Pixel position is computed from the same bbox/width/height
    used to request the image from GIBS (an equirectangular/EPSG:4326
    projection, so lon maps linearly to x and lat maps linearly to y).
 
    points: list of (lat, lon, label) tuples. Defaults to Herschel Island
    and Shingle Point if not given.
 
    The scale bar uses the latitude-adjusted km-per-degree-longitude value
    (1 degree of longitude is shorter in real km the further from the
    equator you are), since this image is not an equal-distance projection.
    This is a standard simple-map approximation, accurate along the
    horizontal/east-west direction at this image's center latitude — not
    survey-grade, but appropriate for a regional reference image like this.
 
    Returns annotated PNG bytes, or the original bytes unchanged if
    annotation fails for any reason (so a drawing bug never blocks the
    underlying satellite image from being shown).
    """
    if points is None:
        points = [
            (69.575, -139.083, "Herschel Island"),
            (68.989, -137.345, "Shingle Point"),
        ]
 
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io as _io
        import math
 
        img = Image.open(_io.BytesIO(png_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        width_px, height_px = img.size
 
        minlon, minlat, maxlon, maxlat = map(float, BBOX_WMS.split(","))
 
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
 
        # --- Label markers ---
        for lat, lon, label_text in points:
            x_frac = (lon - minlon) / (maxlon - minlon)
            y_frac = 1 - (lat - minlat) / (maxlat - minlat)  # y=0 is top=maxlat
            x_px = x_frac * width_px
            y_px = y_frac * height_px
 
            marker_radius = 6
            draw.ellipse(
                [x_px - marker_radius, y_px - marker_radius, x_px + marker_radius, y_px + marker_radius],
                fill=(255, 60, 60), outline=(255, 255, 255), width=2,
            )
 
            text_x, text_y = x_px + 12, y_px - 10
            for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                draw.text((text_x + dx, text_y + dy), label_text, font=font, fill=(0, 0, 0))
            draw.text((text_x, text_y), label_text, font=font, fill=(255, 255, 255))
 
        # --- Scale bar (bottom-left corner) ---
        # Latitude correction uses the center of the bbox, not any one
        # point's latitude, since the scale bar describes the whole image.
        center_lat = (minlat + maxlat) / 2
        km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))
        total_km = (maxlon - minlon) * km_per_deg_lon
        px_per_km = width_px / total_km
 
        bar_px = scale_km * px_per_km
        margin = 30
        bar_x0 = margin
        bar_y0 = height_px - margin - 10
        bar_x1 = bar_x0 + bar_px
 
        draw.line([(bar_x0, bar_y0), (bar_x1, bar_y0)], fill=(255, 255, 255), width=4)
        draw.line([(bar_x0, bar_y0 - 6), (bar_x0, bar_y0 + 6)], fill=(255, 255, 255), width=4)
        draw.line([(bar_x1, bar_y0 - 6), (bar_x1, bar_y0 + 6)], fill=(255, 255, 255), width=4)
        draw.text((bar_x0, bar_y0 + 8), f"{scale_km} km", font=font, fill=(255, 255, 255))
 
        out_buf = _io.BytesIO()
        img.save(out_buf, format="PNG")
        return out_buf.getvalue()
 
    except Exception as e:
        print("MODIS ANNOTATION FAILED (showing unannotated image instead):", e)
        return png_bytes
 
 
modis_bytes, modis_date = fetch_modis_image()
if modis_bytes:
    modis_bytes = annotate_modis_image(modis_bytes)
 
 
# =========================================================
# MODULE 4 — TIDES & SEA LEVEL (DFO Canadian Hydrographic Service, IWLS API)
# =========================================================
# IWLS station 06525 = Herschel Island. Unlike the old SPINE API (which only
# covers the St. Lawrence and never had Arctic coverage), IWLS hosts real
# tide-table stations across Canada including the Arctic. Station IDs are
# internal UUIDs, not the public 5-digit code, so we resolve the code to an
# ID first, then request water level predictions (wlp) for that station.
HERSCHEL_STATION_CODE = "06525"
 
 
def find_iwls_station_id(code):
    try:
        resp = requests.get("https://api-iwls.dfo-mpo.gc.ca/api/v1/stations", timeout=30)
        resp.raise_for_status()
        stations = resp.json()
    except Exception as e:
        print("TIDES: failed to fetch IWLS station list:", e)
        return None
 
    for s in stations:
        if s.get("code") == code:
            return s.get("id")
 
    print(f"TIDES: station code {code} not found in IWLS station list")
    return None
 
 
def fetch_tide_predictions(station_id, hours_ahead=24):
    from_dt = now
    to_dt = now + timedelta(hours=hours_ahead)
    url = f"https://api-iwls.dfo-mpo.gc.ca/api/v1/stations/{station_id}/data"
    params = {
        "time-series-code": "wlp",
        "from": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print("TIDES: failed to fetch predictions:", e)
        return None
 
 
station_id = find_iwls_station_id(HERSCHEL_STATION_CODE)
tide_points = fetch_tide_predictions(station_id) if station_id else None
 
if tide_points:
    # Find the prediction point closest to right now
    closest = min(
        tide_points,
        key=lambda p: abs(datetime.fromisoformat(p["eventDate"].replace("Z", "+00:00")) - now.replace(tzinfo=timezone.utc)),
    )
    current_level = closest.get("value")
    event_time = closest.get("eventDate", "")
 
    # Find next high and low in the window for a bit more useful context
    sorted_points = sorted(tide_points, key=lambda p: p["eventDate"])
    levels = [p["value"] for p in sorted_points]
    next_max = max(levels) if levels else None
    next_min = min(levels) if levels else None
 
    tide_text = [
        ("Predicted water level (now): ", f"{current_level:.2f} m"),
        ["Next 24h range: ", ("", f"{next_min:.2f} m"), " to ", ("", f"{next_max:.2f} m")],
        "Reference: chart datum, Herschel Island station (06525)",
        "Source: DFO/CHS Integrated Water Level System (IWLS)",
    ]
else:
    tide_text = (
        "Tide prediction data unavailable for Herschel Island station (06525).\n"
        "Check Action logs — this uses DFO's IWLS API, which requires resolving "
        "the station code to an internal station ID first; if DFO changes that "
        "station's status or the API shape, this lookup may need adjustment."
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
 
temp_chart_block = None
if temp_chart_bytes:
    try:
        uid = upload_image_to_notion(temp_chart_bytes, "temp_chart.png")
        temp_chart_block = image_block_from_upload(uid)
    except Exception as e:
        print("TEMP CHART NOTION UPLOAD FAILED:", e)
        temp_chart_caption = "Chart generated but upload to Notion failed — see Action logs."
 
weather_icon_block = None
if weather_icon_bytes:
    try:
        uid = upload_image_to_notion(weather_icon_bytes, "weather_icon.png")
        weather_icon_block = image_block_from_upload(uid)
    except Exception as e:
        print("WEATHER ICON NOTION UPLOAD FAILED:", e)
 
sun_chart_block = None
if sun_chart_bytes:
    try:
        uid = upload_image_to_notion(sun_chart_bytes, "sun_chart.png")
        sun_chart_block = image_block_from_upload(uid)
    except Exception as e:
        print("SUN CHART NOTION UPLOAD FAILED:", e)
        sun_chart_caption = "Sun chart generated but upload to Notion failed — see Action logs."
 
 
# =========================================================
# ASSEMBLE DASHBOARD BLOCKS
# =========================================================
blocks = [
    paragraph(f"Last update (UTC): {now.strftime('%Y-%m-%d %H:%M')}"),
    divider(),
 
    heading("🛰 Satellite — MODIS True Color"),
]
if modis_block:
    blocks.append(modis_block)
blocks.append(paragraph(modis_caption))
blocks.append(divider())
 
# --- Row 1: current conditions (weather) + sun, side by side ---
weather_column = [
    heading("🌡 Weather", level=3),
    callout(
        weather_text,
        emoji="🌡",
        color="blue_background",
        children=[weather_icon_block] if weather_icon_block else None,
    ),
]
 
sun_column = [
    heading("☀️ Sunrise / Sunset", level=3),
    callout(sun_text, emoji="☀️", color="yellow_background"),
]
blocks.append(columns(weather_column, sun_column))
 
blocks.append(divider())
 
# --- Sun position curve, full width since a chart needs more room than a column ---
blocks.append(heading("📈 Sun Position — today's solar elevation"))
if sun_chart_block:
    blocks.append(sun_chart_block)
blocks.append(paragraph(sun_chart_caption if sun_chart_bytes else "Sun position chart could not be generated — see Action logs."))
 
blocks.append(divider())
 
# --- Row 2: land forecast + marine forecast, side by side ---
land_column = [
    heading("📅 Weather Forecast — next 5 days", level=3),
    callout(land_forecast_text, emoji="📅", color="green_background"),
]
marine_column = [
    heading("⚓ Marine Forecast — Yukon Coast", level=3),
    callout(marine_text, emoji="⚓", color="purple_background"),
]
blocks.append(columns(land_column, marine_column))
 
blocks.append(divider())
 
# --- Temperature chart (full width, needs room for the image) ---
blocks.append(heading("📈 Temperature — last 10 days vs. 30-year average"))
if temp_chart_block:
    blocks.append(temp_chart_block)
blocks.append(paragraph(temp_chart_caption if temp_chart_bytes else "Chart could not be generated — see Action logs."))
 
blocks.append(divider())
 
# --- Row 3: tides + permafrost, side by side ---
tide_column = [
    heading("🌊 Tides", level=3),
    callout(tide_text, emoji="🌊", color="blue_background"),
]
permafrost_column = [
    heading("🧊 Permafrost (boreholes)", level=3),
    callout(
        "Placeholder — no live data source configured yet. Add borehole logger endpoint here when available.",
        emoji="🧊",
        color="gray_background",
    ),
    link_paragraph("→ GTN-P Global Terrestrial Network for Permafrost database", "https://data.gtn-p.org/"),
]
blocks.append(columns(tide_column, permafrost_column))
 
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
