import os
import io
import json
import math
import time
import requests
import numpy as np
from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo
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
# HISTORICAL DATA CACHE
# Historical (complete, past) years of daily temperature never change once
# fetched — only the current (in-progress) year needs refreshing. This
# cache stores one full calendar year of daily mean temperatures per
# entry, keyed by year, in a JSON file committed back to the repo by the
# workflow after each run (see the "Commit cache" step in the workflow
# YAML). A year is only ever fetched from Open-Meteo once; every run
# after that reads it from this file instead, removing most of this
# script's exposure to Open-Meteo's demonstrated intermittent timeouts.
# =========================================================
CACHE_FILE_PATH = "cache/daily_temps_cache.json"
 
 
def load_temp_cache():
    """Loads the historical temperature cache from disk, or {} if missing/corrupt."""
    try:
        with open(CACHE_FILE_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"CACHE: could not load {CACHE_FILE_PATH} ({e}), starting with empty cache")
        return {}
 
 
def save_temp_cache(cache):
    """Writes the cache back to disk. Creates the cache/ directory if needed."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE_PATH), exist_ok=True)
        with open(CACHE_FILE_PATH, "w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"CACHE: failed to save {CACHE_FILE_PATH}: {e}")
 
 
_temp_cache = load_temp_cache()
_temp_cache_dirty = False  # tracks whether anything new was added this run, so we only write if needed
 
# =========================================================
# SITE CONSTANTS — Herschel Island / Qikiqtaruk
# =========================================================
LAT = 68.933333
LON = -137.2
 
# Computed once via compute_yearly_mean_once.py (real run, not a
# placeholder) — see that script for methodology (single nearest grid
# point, daily-sampled across the past 365 days). Refresh by re-running
# that script roughly once a year; the value barely changes year to
# year, so recomputing it more often isn't necessary.
WATER_LEVEL_YEARLY_MEAN = -0.2668  # computed 2026-06-25 via compute_yearly_mean_once.py
 
# 'now' stays naive UTC throughout the script — every API call, date
# arithmetic ("yesterday", "last 30 days", etc.) and historical fetch
# depends on this being UTC, so it is never converted in place. For
# DISPLAY purposes only, INUVIK_TZ and to_inuvik_time() convert a UTC
# datetime to local Mountain Time for Inuvik, NT — America/Inuvik
# genuinely observes MST/MDT (unlike Yukon, which abandoned the
# twice-yearly switch), so this is a real DST-aware conversion, not a
# fixed offset. Verified against known transition dates: MDT (UTC-6) in
# summer, MST (UTC-7) in winter.
INUVIK_TZ = ZoneInfo("America/Inuvik")
 
 
def to_inuvik_time(utc_dt):
    """Converts a naive UTC datetime to Inuvik local time (DST-aware)."""
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(INUVIK_TZ)
 
 
now = datetime.utcnow()
now_inuvik = to_inuvik_time(now)
 
print(f"SCRIPT STARTED: {now.isoformat()} UTC")
 
 
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
 
 
def disclaimer_paragraph(text):
    """
    Builds a paragraph block in gray, italicized text — used for the
    bottom-of-page disclaimer, visually distinct (de-emphasized) from the
    rest of the dashboard's normal-weight content.
    """
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{
                "type": "text",
                "text": {"content": text},
                "annotations": {"color": "gray", "italic": True},
            }]
        },
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
 
 
def gray_caption(text):
    """
    Builds a paragraph block with gray text — the closest real equivalent
    to 'small/de-emphasized font' for an individual block, since Notion's
    API has no per-block font-size control (only a page-wide 'Small text'
    toggle, which would affect everything, not just this caption).
    """
    if not text:
        return paragraph("")
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}, "annotations": {"color": "gray"}}]},
    }
 
 
def link_paragraph(label, url, prefix=None, prefix_gray=False):
    """
    Paragraph block containing a clickable link, optionally preceded by
    plain (non-linked) text on the same line — e.g. a caption followed by
    an "Explore here" link, without a line break between them.
 
    prefix_gray: if True, renders the prefix text in gray (matching
    gray_caption's de-emphasized styling), so a source line and its
    "Explore here" link can share one line/block instead of being two
    separate paragraph blocks, which always render on separate lines
    regardless of content — that's inherent to Notion's block structure,
    not something a styling change alone could fix.
    """
    rich_text = []
    if prefix:
        prefix_segment = {"type": "text", "text": {"content": prefix}}
        if prefix_gray:
            prefix_segment["annotations"] = {"color": "gray"}
        rich_text.append(prefix_segment)
    rich_text.append({"type": "text", "text": {"content": label, "link": {"url": url}}})
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": rich_text},
    }
 
 
def table_row(cell_lines_list):
    """
    Builds a single table_row block. cell_lines_list is a list of "lines"
    specs (one per column, same format accepted by build_bolded_lines —
    plain string, (label, value) tuple, or list of mixed segments), so
    individual values within a cell can be bolded the same way as
    elsewhere in this script.
    """
    return {
        "object": "block",
        "type": "table_row",
        "table_row": {
            "cells": [build_bolded_lines([cell]) for cell in cell_lines_list]
        },
    }
 
 
def table(header_cells, rows, has_column_header=True):
    """
    Builds a table block with a fixed column count (table_width), which
    per Notion's API can only be set at creation time. All rows — header
    included — must be supplied as nested children in the same create
    call; rows cannot be patched in afterward.
 
    header_cells: list of plain strings/line-specs for the header row.
    rows: list of cell_lines_list, one per data row (see table_row).
    """
    width = len(header_cells)
    all_rows = [table_row(header_cells)] + [table_row(r) for r in rows]
    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": width,
            "has_column_header": has_column_header,
            "has_row_header": False,
            "children": all_rows,
        },
    }
 
 
def columns(*column_block_lists, width_ratios=None):
    """
    Builds a column_list block with N columns, each containing the given
    list of child blocks. Notion requires all column content to be created
    in the same request as the column_list itself — content cannot be
    patched into columns afterward the way top-level blocks can.
 
    width_ratios: optional list of floats (one per column, same order as
    column_block_lists) between 0 and 1, summing to 1, to give columns
    unequal widths. If omitted, Notion defaults to equal widths.
    """
    column_objs = []
    for i, blocks in enumerate(column_block_lists):
        column_data = {"children": blocks}
        if width_ratios:
            column_data["width_ratio"] = width_ratios[i]
        column_objs.append({"object": "block", "type": "column", "column": column_data})
 
    return {
        "object": "block",
        "type": "column_list",
        "column_list": {"children": column_objs},
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
 
 
def external_image_block(url):
    """
    Image block referencing a directly-hosted external URL (not something
    we fetch and re-upload ourselves). Per Notion's API docs, the URL must
    be directly hosted — not a URL that points to a service that retrieves
    the image — and .svg is among the supported image types.
    """
    return {
        "object": "block",
        "type": "image",
        "image": {"type": "external", "external": {"url": url}},
    }
 
 
def fetch_and_convert_logo_to_png(svg_url, output_width=120):
    """
    Fetches an SVG logo and converts it to a fixed-pixel-width PNG, rather
    than embedding the SVG directly. The AWI logo, embedded as a raw SVG
    via external_image_block(), rendered far too large on mobile while
    looking correct on desktop — Notion has no per-block image size
    control, and scales images to fill the container, which apparently
    interacts unpredictably with an SVG's own internal viewBox sizing
    across different device renderers. A fixed-pixel PNG, uploaded via
    the same pipeline as every chart on this page, gives genuine, precise
    size control independent of any of that.
 
    Returns PNG bytes, or None on failure (e.g. network issue, or the
    cairosvg dependency not being installed — this is wrapped so a
    problem here never blocks the rest of the dashboard from updating).
    """
    try:
        import cairosvg
 
        resp = requests.get(svg_url, timeout=15)
        resp.raise_for_status()
 
        png_bytes = cairosvg.svg2png(bytestring=resp.content, output_width=output_width)
        return png_bytes
 
    except Exception as e:
        print("LOGO SVG-TO-PNG CONVERSION FAILED:", e)
        return None
 
 
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
def get_with_retry(url, params=None, timeout=20, retries=1, backoff_seconds=3):
    """
    Wraps requests.get with one automatic retry on timeout or connection
    errors. archive-api.open-meteo.com has shown widespread transient
    timeouts across many sequential historical calls in a single run
    (seen in practice — sometimes 10+ of 30 calls fail in one run), so a
    single retry meaningfully improves the odds of getting real data
    without dramatically increasing total run time when things are
    already working normally.
    """
    last_exception = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exception = e
            if attempt < retries:
                time.sleep(backoff_seconds)
                continue
            raise
        except Exception:
            raise
    raise last_exception
 
 
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
            "hourly": "relativehumidity_2m,pressure_msl,windspeed_10m,winddirection_10m",
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
        hourly_wind_forecast = None
        try:
            idx = data["hourly"]["time"].index(current_hour)
            humidity = data["hourly"]["relativehumidity_2m"][idx]
            pressure = data["hourly"]["pressure_msl"][idx]
            # Slice the next 48h of wind forecast starting from now, for
            # the Wind card's compact forecast chart — reuses this same
            # request rather than a separate API call.
            hourly_wind_forecast = {
                "time": data["hourly"]["time"][idx:idx+49],
                "windspeed_10m": data["hourly"]["windspeed_10m"][idx:idx+49],
                "winddirection_10m": data["hourly"]["winddirection_10m"][idx:idx+49],
            }
        except (ValueError, KeyError, IndexError) as e:
            print("WEATHER: could not align hourly index:", e)
 
        return {
            "temperature_c": cw.get("temperature"),
            "windspeed_kmh": cw.get("windspeed"),
            "winddirection_deg": cw.get("winddirection"),
            "weathercode": cw.get("weathercode"),
            "humidity_pct": humidity,
            "pressure_hpa": pressure,
            "hourly_wind_forecast": hourly_wind_forecast,
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
            "hourly_wind_forecast": None,
            "status": "missing",
        }
 
 
weather = get_weather()
 
if weather["status"] == "ok":
    compass = degrees_to_compass(weather["winddirection_deg"])
    wind_dir_text = f"{compass} ({weather['winddirection_deg']:.0f}°)" if compass else "—"
    weather_text = [
        ("Air temperature: ", f"{weather['temperature_c']} °C"),
        ("Humidity: ", f"{weather['humidity_pct']} %"),
        ("Pressure: ", f"{weather['pressure_hpa']} hPa"),
    ]
    weather_source_text = "Source: Open-Meteo (ERA5-based forecast/analysis)"
    wind_now_text = [
        ("Wind speed: ", f"{weather['windspeed_kmh']} km/h"),
        ("Wind direction: ", wind_dir_text),
    ]
    wind_source_text = "Source: Open-Meteo (ERA5-based forecast/analysis)"
else:
    weather_text = "Weather data unavailable — fetch failed. Check Action logs."
    weather_source_text = ""
    wind_now_text = "Wind data unavailable — fetch failed. Check Action logs."
    wind_source_text = ""
 
 
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
from PIL import Image, ImageDraw, ImageFont
 
NOTION_ICON_SIZE = 140
 
 
def _icon_new_canvas():
    return Image.new("RGBA", (NOTION_ICON_SIZE, NOTION_ICON_SIZE), (0, 0, 0, 0))
 
 
def _icon_cloud_bumps(r):
    return [(-1.4, 0.1, 0.8), (-0.5, -0.5, 1.0), (0.5, -0.45, 1.05),
            (1.4, 0.1, 0.75), (-0.9, 0.3, 0.85), (0.9, 0.3, 0.85)]
 
 
def _icon_cloud_with_shadow(cx, cy, r, fill, highlight=None):
    """
    Builds a cloud shape with a soft drop shadow and a subtle highlight on
    the upper lobes, for a gentler, more dimensional look than a flat
    single-color fill.
    """
    from PIL import ImageFilter
 
    img = _icon_new_canvas()
    bumps = _icon_cloud_bumps(r)
 
    shadow = _icon_new_canvas()
    sd = ImageDraw.Draw(shadow)
    for dx, dy, s in bumps:
        rr = r * s
        sd.ellipse([cx + dx * r - rr, cy + dy * r - rr + 4, cx + dx * r + rr, cy + dy * r + rr + 4], fill=(0, 0, 0, 55))
    shadow = shadow.filter(ImageFilter.GaussianBlur(4))
    img = Image.alpha_composite(img, shadow)
 
    draw = ImageDraw.Draw(img)
    for dx, dy, s in bumps:
        rr = r * s
        draw.ellipse([cx + dx * r - rr, cy + dy * r - rr, cx + dx * r + rr, cy + dy * r + rr], fill=fill)
    if highlight:
        for dx, dy, s in [(-0.5, -0.5, 1.0), (0.5, -0.45, 1.05)]:
            rr = r * s * 0.55
            draw.ellipse([cx + dx * r - rr, cy + dy * r - rr - 3, cx + dx * r + rr, cy + dy * r + rr - 3], fill=highlight)
 
    return img
 
 
def _icon_sun(cx=None, cy=None, r=30):
    """Layered gradient sun (three concentric circles, light to dark) with rays."""
    img = _icon_new_canvas()
    if cx is None:
        cx, cy = NOTION_ICON_SIZE // 2, NOTION_ICON_SIZE // 2
    draw = ImageDraw.Draw(img)
 
    for i in range(12):
        angle = i * math.pi / 6
        x1 = cx + math.cos(angle) * (r + 8)
        y1 = cy + math.sin(angle) * (r + 8)
        x2 = cx + math.cos(angle) * (r + 18)
        y2 = cy + math.sin(angle) * (r + 18)
        draw.line([(x1, y1), (x2, y2)], fill=(255, 200, 60, 255), width=5)
 
    for rad, color in [(r, (255, 196, 61, 255)), (r - 6, (255, 179, 46, 255)), (r - 14, (255, 213, 107, 255))]:
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=color)
 
    return img
 
 
def _icon_cloud(cx=None, cy=None, r=22, dark=False):
    if cx is None:
        cx, cy = NOTION_ICON_SIZE // 2, NOTION_ICON_SIZE // 2 + 5
    fill = (158, 165, 175, 255) if dark else (255, 255, 255, 255)
    highlight = (190, 196, 204, 255) if dark else (245, 248, 250, 255)
    return _icon_cloud_with_shadow(cx, cy, r, fill, highlight)
 
 
def _icon_partly_cloudy():
    sun = _icon_sun(cx=NOTION_ICON_SIZE // 2 - 15, cy=NOTION_ICON_SIZE // 2 - 15, r=18)
    cloud = _icon_cloud(cx=NOTION_ICON_SIZE // 2 + 13, cy=NOTION_ICON_SIZE // 2 + 15, r=18)
    return Image.alpha_composite(sun, cloud)
 
 
def _icon_rain(heavy=False):
    cx, cy = NOTION_ICON_SIZE // 2, NOTION_ICON_SIZE // 2 - 5
    r = 20
    fill = (158, 165, 175, 255) if heavy else (200, 206, 213, 255)
    highlight = (180, 186, 194, 255) if heavy else (225, 229, 233, 255)
    img = _icon_cloud_with_shadow(cx, cy, r, fill, highlight)
    draw = ImageDraw.Draw(img)
    offsets = [-18, -6, 6, 18] if heavy else [-13, 0, 13]
    for dx in offsets:
        x0, y0 = cx + dx, cy + 20
        x1, y1 = cx + dx - 4, cy + 34
        draw.line([(x0, y0), (x1, y1)], fill=(64, 131, 217, 120), width=8)  # soft glow halo
        draw.line([(x0, y0), (x1, y1)], fill=(64, 131, 217, 255), width=4)
    return img
 
 
def _icon_snow():
    cx, cy = NOTION_ICON_SIZE // 2, NOTION_ICON_SIZE // 2 - 5
    r = 20
    img = _icon_cloud_with_shadow(cx, cy, r, (255, 255, 255, 255), (245, 248, 250, 255))
    draw = ImageDraw.Draw(img)
    for dx in [-13, 0, 13]:
        for dy in [24, 36]:
            cx2, cy2 = cx + dx, cy + dy
            rad = 4
            for i in range(3):
                angle = i * math.pi / 3
                x1 = cx2 + math.cos(angle) * rad
                y1 = cy2 + math.sin(angle) * rad
                x2 = cx2 - math.cos(angle) * rad
                y2 = cy2 - math.sin(angle) * rad
                draw.line([(x1, y1), (x2, y2)], fill=(170, 195, 220, 255), width=2)
    return img
 
 
def _icon_fog():
    img = _icon_new_canvas()
    draw = ImageDraw.Draw(img)
    cx, cy = NOTION_ICON_SIZE // 2, NOTION_ICON_SIZE // 2
    for i, dy in enumerate([-22, -6, 10, 26]):
        w = 32 - i * 2
        alpha = 255 - i * 15
        draw.line([(cx - w, cy + dy), (cx + w, cy + dy)], fill=(150, 165, 180, alpha), width=7)
    return img
 
 
def _icon_thunder():
    cx, cy = NOTION_ICON_SIZE // 2, NOTION_ICON_SIZE // 2 - 10
    r = 20
    img = _icon_cloud_with_shadow(cx, cy, r, (148, 158, 170, 255), (172, 181, 192, 255))
    draw = ImageDraw.Draw(img)
    pts = [(cx - 5, cy + 16), (cx + 6, cy + 16), (cx - 2, cy + 30), (cx + 8, cy + 30), (cx - 6, cy + 46)]
    draw.line(pts, fill=(255, 200, 40, 90), width=10, joint="curve")  # glow
    draw.line(pts, fill=(255, 196, 40, 255), width=5, joint="curve")
    return img
 
 
def _icon_wind_arrow(direction_from_deg, speed_kmh):
    """
    Draws an arrow icon showing wind direction and relative strength, in
    the same gradient/shadow visual style as the weather icons. Points in
    the direction the wind blows TOWARD (180° from the meteorological
    "from" convention), with both length and color intensity scaling
    with speed.
    """
    from PIL import ImageFilter
 
    size = NOTION_ICON_SIZE
    cx, cy = size // 2, size // 2
 
    img = _icon_new_canvas()
 
    min_len, max_len = 45, 65
    length = min_len + min(speed_kmh / 40, 1.0) * (max_len - min_len)
 
    angle_rad = math.radians(direction_from_deg + 180)
    dx = math.sin(angle_rad) * length
    dy = -math.cos(angle_rad) * length
    tail = (cx - dx, cy - dy)
    tip = (cx + dx, cy + dy)
 
    import matplotlib
    # Same colormap and 0-40 km/h normalization as the 30-day wind vector
    # chart, so this single current-conditions arrow is genuinely
    # color-consistent with that chart, not just a similar-looking but
    # separately hand-picked scheme.
    _norm_speed = max(0, min(speed_kmh, 40)) / 40
    _plasma_rgba = matplotlib.colormaps["plasma"](_norm_speed)
    color = tuple(round(c * 255) for c in _plasma_rgba[:3]) + (255,)
 
    shadow = _icon_new_canvas()
    sd = ImageDraw.Draw(shadow)
    sd.line([tail, tip], fill=(0, 0, 0, 70), width=16)
    shadow = shadow.filter(ImageFilter.GaussianBlur(4))
    img = Image.alpha_composite(img, shadow)
 
    draw = ImageDraw.Draw(img)
 
    head_len = 22
    head_angle = math.radians(28)
    back_angle = angle_rad + math.pi
 
    # The shaft stops at the arrowhead's BASE (a point head_len back from
    # the true tip along the arrow's own axis), not at the tip itself —
    # otherwise the thick flat-capped shaft end pokes through the
    # narrower triangle base, looking like two mismatched pieces glued
    # together rather than one continuous arrow.
    shaft_end = (tip[0] + head_len * 0.6 * math.sin(back_angle),
                 tip[1] - head_len * 0.6 * math.cos(back_angle))
    draw.line([tail, shaft_end], fill=color, width=12)
 
    left = (tip[0] + head_len * math.sin(back_angle + head_angle),
            tip[1] - head_len * math.cos(back_angle + head_angle))
    right = (tip[0] + head_len * math.sin(back_angle - head_angle),
             tip[1] - head_len * math.cos(back_angle - head_angle))
    draw.polygon([tip, left, right], fill=color)
 
    return img
 
 
def render_wind_icon(direction_from_deg, speed_kmh):
    """
    Renders the wind direction/strength arrow as PNG bytes, or None if
    rendering fails for any reason (so a drawing bug never blocks the
    rest of the dashboard from updating).
    """
    try:
        import io as _io
        img = _icon_wind_arrow(direction_from_deg, speed_kmh)
        out_buf = _io.BytesIO()
        img.save(out_buf, format="PNG")
        return out_buf.getvalue()
    except Exception as e:
        print("WIND ICON RENDER FAILED:", e)
        return None
 
 
def render_icon_with_big_number(icon_bytes, number_text, unit_text, icon_size=90, number_color=(40, 40, 40)):
    """
    Combines a small icon on the left with a large number + unit on the
    right, rendered as a single image, cropped TIGHTLY to actual content
    rather than a fixed oversized canvas.
 
    Why tight cropping matters: Notion has no per-block image width
    control — it always scales the whole image to fill the available
    column width. A canvas with a lot of empty transparent margin around
    the real content (icon + number) meant that margin got scaled along
    with everything else, so the actual number ended up much smaller on
    screen than its own font size would suggest, especially in a
    half-width column. Cropping to content means nearly every pixel of
    the final image is meaningful, so the same display width shows
    dramatically larger text.
 
    number_color: RGB tuple for the big number specifically, letting
    callers color-code it (e.g. by temperature or wind force) while the
    unit text stays a neutral gray.
 
    Returns PNG bytes, or None on failure.
    """
    try:
        import io as _io
 
        icon = Image.open(_io.BytesIO(icon_bytes)).convert("RGBA") if icon_bytes else None
 
        try:
            font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
            font_unit = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        except Exception:
            font_big = font_unit = ImageFont.load_default()
 
        # Measure actual text size first, on a throwaway canvas, so the
        # real canvas can be sized exactly to fit (plus small margins).
        _tmp = Image.new("RGBA", (10, 10))
        _tmp_draw = ImageDraw.Draw(_tmp)
        num_bbox = _tmp_draw.textbbox((0, 0), number_text, font=font_big)
        num_w, num_h = num_bbox[2] - num_bbox[0], num_bbox[3] - num_bbox[1]
        unit_bbox = _tmp_draw.textbbox((0, 0), unit_text, font=font_unit)
        unit_w, unit_h = unit_bbox[2] - unit_bbox[0], unit_bbox[3] - unit_bbox[1]
 
        margin = 8
        gap_icon_text = 14
        gap_num_unit = 6
 
        canvas_h = max(icon_size, num_h) + margin * 2
        canvas_w = margin + icon_size + gap_icon_text + num_w + gap_num_unit + unit_w + margin
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
 
        if icon:
            icon_resized = icon.resize((icon_size, icon_size), Image.LANCZOS)
            paste_y = (canvas_h - icon_size) // 2
            canvas.paste(icon_resized, (margin, paste_y), icon_resized)
 
        draw = ImageDraw.Draw(canvas)
        text_x = margin + icon_size + gap_icon_text
        text_y = (canvas_h - num_h) // 2 - num_bbox[1]
        draw.text((text_x, text_y), number_text, font=font_big, fill=number_color + (255,))
 
        unit_x = text_x + num_w + gap_num_unit
        unit_y = text_y + num_h - unit_h + num_bbox[1] - unit_bbox[1] - 2
        draw.text((unit_x, unit_y), unit_text, font=font_unit, fill=(90, 90, 90, 255))
 
        out_buf = _io.BytesIO()
        canvas.save(out_buf, format="PNG")
        return out_buf.getvalue()
 
    except Exception as e:
        print("ICON WITH BIG NUMBER RENDER FAILED:", e)
        return None
 
 
def temperature_to_color(temp_c):
    """
    Maps a temperature in Celsius to a cold-to-hot color gradient. There
    is no single official WMO color standard for temperature display
    (confirmed — meteorological organizations each use their own
    convention), so this follows the common, widely-used blue-to-red
    convention rather than claiming a specific authoritative standard.
    """
    if temp_c is None:
        return (40, 40, 40)
    stops = [
        (-30, (84, 130, 217)),
        (-15, (107, 174, 230)),
        (0, (140, 200, 230)),
        (10, (90, 160, 110)),
        (20, (220, 170, 50)),
        (30, (220, 100, 50)),
        (40, (180, 40, 40)),
    ]
    if temp_c <= stops[0][0]:
        return stops[0][1]
    if temp_c >= stops[-1][0]:
        return stops[-1][1]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= temp_c <= t1:
            frac = (temp_c - t0) / (t1 - t0)
            return tuple(round(c0[j] + (c1[j] - c0[j]) * frac) for j in range(3))
    return (40, 40, 40)
 
 
def windspeed_to_beaufort_color(speed_kmh):
    """
    Maps a wind speed in km/h to a color based on the Beaufort wind force
    scale — a real, internationally standardized scale (defined by the
    WMO, with identical speed-range definitions worldwide; only the
    preferred display unit varies by country). Color progresses from
    pale (calm) through yellow/orange (gale) to dark red (storm+).
    Returns (color_rgb_tuple, beaufort_description).
    """
    if speed_kmh is None:
        return (40, 40, 40), "—"
    scale = [
        (1, (180, 200, 215), "Calm"),
        (5, (150, 190, 210), "Light air"),
        (11, (120, 180, 190), "Light breeze"),
        (19, (100, 170, 140), "Gentle breeze"),
        (28, (140, 170, 80), "Moderate breeze"),
        (38, (200, 170, 50), "Fresh breeze"),
        (49, (220, 140, 40), "Strong breeze"),
        (61, (220, 100, 40), "Moderate gale"),
        (74, (200, 60, 40), "Gale"),
        (88, (170, 40, 40), "Strong gale"),
        (102, (140, 20, 60), "Storm"),
        (117, (110, 10, 80), "Violent storm"),
        (9999, (80, 0, 80), "Hurricane force"),
    ]
    for max_kmh, color, label in scale:
        if speed_kmh <= max_kmh:
            return color, label
    return scale[-1][1], scale[-1][2]
 
 
def render_weather_icon(weathercode):
    """
    Renders a small PNG icon matching the given WMO weathercode.
    Returns PNG bytes, or None if rendering fails for any reason (so a
    drawing bug never blocks the rest of the dashboard from updating).
    """
    try:
        from PIL import Image, ImageDraw
        import io as _io
 
        code = weathercode if weathercode is not None else -1
 
        if code in (0, 1):
            img = _icon_sun()
        elif code == 2:
            img = _icon_partly_cloudy()
        elif code == 3:
            img = _icon_cloud()
        elif code in (45, 48):
            img = _icon_fog()
        elif code in (51, 53, 55, 56, 57):
            img = _icon_rain(heavy=False)
        elif code in (61, 63, 65, 66, 67):
            img = _icon_rain(heavy=(code in (65, 67)))
        elif code in (71, 73, 75, 77):
            img = _icon_snow()
        elif code in (80, 81, 82):
            img = _icon_rain(heavy=(code == 82))
        elif code in (85, 86):
            img = _icon_snow()
        elif code in (95, 96, 99):
            img = _icon_thunder()
        else:
            # Unrecognized code: fall back to a plain cloud rather than
            # guessing, since an unknown code shouldn't be shown as sunny.
            img = _icon_cloud()
 
        out_buf = _io.BytesIO()
        img.save(out_buf, format="PNG")
        return out_buf.getvalue()
 
    except Exception as e:
        print("WEATHER ICON RENDER FAILED:", e)
        return None
 
 
def weathercode_to_emoji(code):
    """
    Maps a WMO weathercode to a representative emoji, using the same code
    groupings as render_weather_icon, for use in contexts where an actual
    image can't be embedded — e.g. Notion table cells, which only support
    rich text, not nested image blocks.
    """
    if code is None:
        return "—"
    if code in (0, 1):
        return "☀️"
    elif code == 2:
        return "🌤️"
    elif code == 3:
        return "☁️"
    elif code in (45, 48):
        return "🌫️"
    elif code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82):
        return "🌧️"
    elif code in (71, 73, 75, 77, 85, 86):
        return "❄️"
    elif code in (95, 96, 99):
        return "⛈️"
    else:
        return "☁️"
 
 
weather_icon_bytes = render_weather_icon(weather.get("weathercode")) if weather["status"] == "ok" else None
wind_icon_bytes = (
    render_wind_icon(weather["winddirection_deg"], weather["windspeed_kmh"])
    if weather["status"] == "ok" and weather.get("winddirection_deg") is not None and weather.get("windspeed_kmh") is not None
    else None
)
 
weather_icon_big_bytes = (
    render_icon_with_big_number(
        weather_icon_bytes, f"{weather['temperature_c']:.0f}", "°C",
        number_color=temperature_to_color(weather["temperature_c"]),
    )
    if weather_icon_bytes and weather.get("temperature_c") is not None
    else weather_icon_bytes
)
 
_wind_color, _beaufort_label = windspeed_to_beaufort_color(weather.get("windspeed_kmh"))
if weather["status"] == "ok" and isinstance(wind_now_text, list):
    wind_now_text.append(("Beaufort force: ", _beaufort_label))
wind_icon_big_bytes = (
    render_icon_with_big_number(
        wind_icon_bytes, f"{weather['windspeed_kmh']:.0f}", "km/h",
        number_color=_wind_color,
    )
    if wind_icon_bytes and weather.get("windspeed_kmh") is not None
    else wind_icon_bytes
)
 
 
def build_mini_forecast_strip(days_data):
    """
    Renders a compact horizontal strip: one small weather icon per day,
    with a day label above and a temperature range below — a scannable
    visual summary for the Weather card, reusing the same icon family as
    the rest of the dashboard rather than introducing a new visual style.
 
    days_data: list of dicts with 'day_label', 'weathercode', 'temp_min',
    'temp_max' keys. Returns PNG bytes, or None on failure.
    """
    try:
        import io as _io
 
        n = len(days_data)
        cell_w, cell_h = 110, 175
        canvas = Image.new("RGBA", (cell_w * n, cell_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
 
        try:
            font_day = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
            font_temp = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except Exception:
            font_day = font_temp = ImageFont.load_default()
 
        for i, d in enumerate(days_data):
            x0 = i * cell_w
            icon_bytes = render_weather_icon(d["weathercode"])
            if icon_bytes:
                icon_img = Image.open(_io.BytesIO(icon_bytes)).convert("RGBA")
                icon_img = icon_img.resize((76, 76), Image.LANCZOS)
                canvas.paste(icon_img, (x0 + (cell_w - 76) // 2, 42), icon_img)
 
            day_label = d["day_label"]
            temp_label = f"{d['temp_min']:.0f}–{d['temp_max']:.0f}°"
 
            day_bbox = draw.textbbox((0, 0), day_label, font=font_day)
            draw.text((x0 + (cell_w - (day_bbox[2] - day_bbox[0])) // 2, 4), day_label, font=font_day, fill=(40, 40, 40))
 
            temp_bbox = draw.textbbox((0, 0), temp_label, font=font_temp)
            draw.text((x0 + (cell_w - (temp_bbox[2] - temp_bbox[0])) // 2, 128), temp_label, font=font_temp, fill=(60, 60, 60))
 
        out_buf = _io.BytesIO()
        canvas.save(out_buf, format="PNG")
        return out_buf.getvalue()
 
    except Exception as e:
        print("MINI FORECAST STRIP RENDER FAILED:", e)
        return None
 
 
def build_large_forecast_strip(days_data):
    """
    A larger, more detailed version of build_mini_forecast_strip, sized
    for the full-width 5-day forecast detail section rather than a
    half-width card. Notion table cells have no font-size control, which
    is why the previous emoji-in-table approach couldn't be made bigger
    no matter how the table itself was sized — rendering real icon
    images at a larger fixed size sidesteps that limitation entirely.
 
    days_data: list of dicts with 'day_label', 'weathercode', 'temp_min',
    'temp_max', 'wind_label', 'precip_label' keys.
    Returns PNG bytes, or None on failure.
    """
    try:
        import io as _io
 
        n = len(days_data)
        cell_w, cell_h = 170, 230
        canvas = Image.new("RGBA", (cell_w * n, cell_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
 
        try:
            font_day = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
            font_temp = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_detail = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            font_day = font_temp = font_detail = ImageFont.load_default()
 
        icon_size = 110
 
        for i, d in enumerate(days_data):
            x0 = i * cell_w
            icon_bytes = render_weather_icon(d["weathercode"])
            if icon_bytes:
                icon_img = Image.open(_io.BytesIO(icon_bytes)).convert("RGBA")
                icon_img = icon_img.resize((icon_size, icon_size), Image.LANCZOS)
                canvas.paste(icon_img, (x0 + (cell_w - icon_size) // 2, 36), icon_img)
 
            day_label = d["day_label"]
            day_bbox = draw.textbbox((0, 0), day_label, font=font_day)
            draw.text((x0 + (cell_w - (day_bbox[2] - day_bbox[0])) // 2, 4), day_label, font=font_day, fill=(50, 50, 50))
 
            temp_label = f"{d['temp_min']:.0f}–{d['temp_max']:.0f}°"
            temp_bbox = draw.textbbox((0, 0), temp_label, font=font_temp)
            draw.text((x0 + (cell_w - (temp_bbox[2] - temp_bbox[0])) // 2, 152), temp_label, font=font_temp, fill=(40, 40, 40))
 
            for j, detail_label in enumerate([d.get("wind_label", ""), d.get("precip_label", "")]):
                detail_bbox = draw.textbbox((0, 0), detail_label, font=font_detail)
                draw.text((x0 + (cell_w - (detail_bbox[2] - detail_bbox[0])) // 2, 184 + j * 22), detail_label, font=font_detail, fill=(90, 90, 90))
 
        out_buf = _io.BytesIO()
        canvas.save(out_buf, format="PNG")
        return out_buf.getvalue()
 
    except Exception as e:
        print("LARGE FORECAST STRIP RENDER FAILED:", e)
        return None
 
 
def build_wind_forecast_mini_chart(hourly_wind_forecast):
    """
    Builds a compact 48h wind speed forecast chart for the Wind card,
    sized for a half-width column rather than the full-width 30-day
    historical vector chart elsewhere on the page. Uses data already
    fetched as part of get_weather's existing hourly request — no
    separate API call. Returns (png_bytes, caption).
    """
    if not hourly_wind_forecast or not hourly_wind_forecast.get("time"):
        return None, "Wind forecast unavailable."
 
    try:
        times = hourly_wind_forecast["time"]
        speeds = hourly_wind_forecast["windspeed_10m"]
        hours = list(range(len(times)))
 
        NOTION_BLUE = "#337EA9"
        NOTION_TEXT_GRAY = "#787774"
        NOTION_LIGHT_GRID = "#EDECEC"
 
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(4.2, 2.4), dpi=150)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
 
        ax.fill_between(hours, speeds, 0, color=NOTION_BLUE, alpha=0.15, linewidth=0)
        ax.plot(hours, speeds, color=NOTION_BLUE, linewidth=3)
 
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(NOTION_LIGHT_GRID)
 
        max_h = max(hours) if hours else 48
        tick_positions = [h for h in [0, 12, 24, 36, 48] if h <= max_h]
        tick_labels = ["now" if h == 0 else f"+{h}h" for h in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=16, color=NOTION_TEXT_GRAY)
        ax.tick_params(axis="y", labelsize=16, colors=NOTION_TEXT_GRAY, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=NOTION_LIGHT_GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_ylabel("km/h", fontsize=17, color=NOTION_TEXT_GRAY)
        ax.set_ylim(0, max(speeds) * 1.2 if speeds else 10)
 
        fig.tight_layout()
        png_bytes = fig_to_png_bytes(fig)
        caption = "Wind speed, next 48h. Source: Open-Meteo."
        return png_bytes, caption
 
    except Exception as e:
        print("WIND FORECAST MINI CHART FAILED:", e)
        return None, "Wind forecast chart could not be generated."
 
 
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
            "daily": "temperature_2m_max,temperature_2m_min,windspeed_10m_max,winddirection_10m_dominant,precipitation_sum,precipitation_probability_max,weathercode",
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
                "precip_prob_pct": daily.get("precipitation_probability_max", [None]*len(daily.get("time", [])))[i],
                "weathercode": daily.get("weathercode", [None]*len(daily.get("time", [])))[i],
            })
        return days_list
    except Exception as e:
        print("LAND FORECAST FETCH FAILED:", e)
        return []
 
 
land_forecast_days = get_land_forecast()
 
if land_forecast_days:
    forecast_table_rows = []
    mini_strip_days = []
    for d in land_forecast_days:
        day_label = datetime.strptime(d["date"], "%Y-%m-%d").strftime("%a %b %d")
        compass = degrees_to_compass(d["wind_dir_deg"])
        wind_label = f"{d['wind_max_kmh']:.0f} km/h {compass or ''}".strip()
        forecast_table_rows.append([
            weathercode_to_emoji(d["weathercode"]),
            day_label,
            ("", f"{d['temp_min']:.0f}–{d['temp_max']:.0f} °C"),
            ("", wind_label),
            ("", f"{d['precip_mm']:.1f} mm" + (f" ({d['precip_prob_pct']:.0f}%)" if d.get("precip_prob_pct") is not None else "")),
        ])
        mini_strip_days.append({
            "day_label": datetime.strptime(d["date"], "%Y-%m-%d").strftime("%a"),
            "weathercode": d["weathercode"],
            "temp_min": d["temp_min"],
            "temp_max": d["temp_max"],
            "wind_label": wind_label,
            "precip_label": f"{d['precip_mm']:.1f} mm" + (f" ({d['precip_prob_pct']:.0f}%)" if d.get("precip_prob_pct") is not None else ""),
        })
    land_forecast_table_block = table(
        header_cells=["", "Day", "Temp", "Wind", "Precip (chance)"],
        rows=forecast_table_rows,
    )
    land_forecast_caption = "Source: Open-Meteo"
    mini_forecast_strip_bytes = build_mini_forecast_strip(mini_strip_days)
    large_forecast_strip_bytes = build_large_forecast_strip(mini_strip_days)
else:
    land_forecast_table_block = None
    land_forecast_caption = "Land forecast unavailable — fetch failed. Check Action logs."
    mini_forecast_strip_bytes = None
    large_forecast_strip_bytes = None
 
 
# =========================================================
# MODULE 1a-3 — MARINE FORECAST (Environment Canada, Yukon Coast)
# Source: Environment Canada Atom feed for marine zone 16000, which
# covers Herschel Island / Yukon Coast. The feed returns natural-language
# forecast text per period (e.g. "Wind light becoming southeast 15 knots"),
# not structured numeric fields, so we display the text as published
# rather than trying to parse specific values out of free-form wording.
# =========================================================
def _strip_html_to_text(html_str):
    """
    Converts Environment Canada's HTML-formatted summary text into plain
    text suitable for a Notion paragraph: <br/> tags become newlines, and
    any other HTML tags are stripped using Python's built-in HTML parser
    rather than naive string replacement (more robust to whatever markup
    variations the feed actually contains).
    """
    if not html_str:
        return ""
 
    from html.parser import HTMLParser
 
    class _TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
 
        def handle_data(self, data):
            self.parts.append(data)
 
        def handle_starttag(self, tag, attrs):
            if tag.lower() == "br":
                self.parts.append("\n")
 
    parser = _TextExtractor()
    parser.feed(html_str)
    text = "".join(parser.parts)
 
    # Collapse repeated whitespace within lines, but preserve the
    # intentional newlines from <br/> tags.
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)
 
 
def get_marine_forecast():
    try:
        import xml.etree.ElementTree as ET
 
        url = "https://weather.gc.ca/rss/marine/16000_e.xml"
        # weather.gc.ca's RSS endpoints have shown occasional transient
        # connection timeouts (seen in practice on a real run), so this
        # uses the same retry helper as the historical weather fetches
        # above, rather than a single unprotected attempt.
        r = get_with_retry(url, timeout=15, retries=2, backoff_seconds=5)
 
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.content)
 
        entries = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            title = _strip_html_to_text(title_el.text) if title_el is not None else ""
            summary = _strip_html_to_text(summary_el.text) if summary_el is not None else ""
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
    #
    # The feed's own title text includes "- Yukon Coast" (e.g. "Forecast
    # for Today Tonight and Wednesday - Yukon Coast"), which is redundant
    # since the section heading above already says "Yukon Coast" — strip
    # it here rather than leave it duplicated in the body text.
    import re
 
    def _strip_yukon_coast(text):
        return re.sub(r"\s*[-–—]\s*Yukon Coast\s*$", "", text, flags=re.IGNORECASE).strip()
 
    # The feed repeats an "Issued HH:MM AM/PM <timezone> <date>" line inside
    # every entry's summary (Wind, Waves, Extended Forecast, etc). Extract
    # it once and strip it from each individual entry, rather than show the
    # same issuance timestamp three or more times.
    issued_pattern = re.compile(r"Issued\s+\d{1,2}:\d{2}\s*[AP]M\s+\w+\s+\d{1,2}\s+\w+\s+\d{4}\.?", re.IGNORECASE)
 
    def _extract_and_strip_issued(text):
        match = issued_pattern.search(text)
        issued_text = match.group(0).strip() if match else None
        cleaned = issued_pattern.sub("", text).strip()
        # Collapse any double spaces/newlines left behind after removal
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned, issued_text
 
    lines = []
    issued_line = None
    for e in marine_entries[:6]:
        title = _strip_yukon_coast(e["title"].strip())
        summary = e["summary"].strip() if e["summary"] else ""
        summary, found_issued = _extract_and_strip_issued(summary)
        if found_issued and not issued_line:
            issued_line = found_issued
        if summary and summary != title:
            lines.append([("", title), ": ", summary])
        else:
            lines.append(title)
 
    if issued_line:
        lines.append(issued_line)
    marine_text = lines
    marine_source_text = "Source: Environment Canada (Yukon Coast marine zone)"
else:
    marine_text = "Marine forecast unavailable — fetch failed. Check Action logs."
    marine_source_text = ""
 
 
# =========================================================
# MODULE — WEATHER & COASTAL FLOOD ALERTS (only shown if active)
# Environment Canada publishes a per-location Atom feed (the same kind
# of link found at the bottom of any location's Current Conditions page)
# covering watches, warnings, and special statements — including coastal
# flooding alerts — for that specific point. When nothing is active, the
# feed contains a single boilerplate entry with the well-documented
# wording "No watches or warnings in effect" (confirmed against a
# third-party parser's real output, which recognizes this exact phrase
# as a structured "inEffect: false" signal) — we treat that phrase as
# the reliable signal to show nothing, rather than guess from absence of
# entries alone (the feed may have zero or one boilerplate entry even
# when nothing is active, depending on the exact location).
# =========================================================
def get_weather_alerts():
    try:
        import xml.etree.ElementTree as ET
 
        url = f"https://weather.gc.ca/rss/alerts/{LAT}_{LON}_e.xml"
        # Same retry treatment as the marine forecast and Napoiak fetches
        # — weather.gc.ca and dd.weather.gc.ca have shown transient
        # connection timeouts together on the same run in practice.
        r = get_with_retry(url, timeout=15, retries=2, backoff_seconds=5)
 
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.content)
 
        entries = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            link_el = entry.find("atom:link", ns)
            title = (title_el.text or "").strip() if title_el is not None else ""
            summary = _strip_html_to_text(summary_el.text) if summary_el is not None else ""
            link = link_el.get("href") if link_el is not None else None
            entries.append({"title": title, "summary": summary, "link": link})
        return entries
    except Exception as e:
        print("WEATHER ALERTS FETCH FAILED:", e)
        return None  # None (fetch failed) is distinct from [] (fetched OK, no entries)
 
 
weather_alert_entries = get_weather_alerts()
 
active_alerts = []
if weather_alert_entries is not None:
    for e in weather_alert_entries:
        title_lower = e["title"].lower()
        if "no watches or warnings" in title_lower or "no alerts" in title_lower:
            continue  # the documented boilerplate "nothing active" entry
        active_alerts.append(e)
 
if active_alerts:
    print(f"WEATHER ALERTS: {len(active_alerts)} active alert(s) found")
else:
    print("WEATHER ALERTS: none active (or fetch failed) — section will be hidden")
 
 
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
 
 
def solar_elevation_deg(lat_deg, lon_deg, dt_utc):
    """
    Standard solar elevation angle formula (declination + hour angle).
    Returns elevation in degrees above the horizon (negative = below).
    Verified against known physical cases: ~90° at the equator/equinox
    noon, positive at Herschel Island summer midnight (midnight sun),
    negative at Herschel Island winter noon (polar night boundary).
    """
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
 
 
sun_info = get_sun_info()
 
# At this latitude, sunrise-sunset.org's reported day_length appears to
# behave unexpectedly during polar day — it returned ~0 instead of ~24h in
# practice, likely because its internal sunrise/sunset timestamps become
# degenerate when the sun never sets. Rather than guess at that API's
# internal edge-case behavior, we classify polar day/night directly using
# our own solar elevation formula (already verified correct against known
# physical cases), which doesn't depend on day_length at all.
if sun_info["status"] == "ok":
    # Scan the full day at 15-minute resolution to find the true minimum
    # and maximum elevation — checking only fixed clock-hour samples (e.g.
    # UTC noon/midnight) isn't reliable, since the actual highest/lowest
    # points of the day are offset from UTC by this location's longitude.
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_elevations = [solar_elevation_deg(LAT, LON, day_start + timedelta(minutes=15 * i)) for i in range(96)]
    min_elevation_today = min(day_elevations)
    max_elevation_today = max(day_elevations)
 
    day_length_s = sun_info["day_length_s"]
    hours = int(day_length_s // 3600) if day_length_s else 0
    minutes = int((day_length_s % 3600) // 60) if day_length_s else 0
 
    if min_elevation_today > 0:
        # Sun never sets: above the horizon at every point in the day.
        sun_text = "Sun stays above the horizon all day (midnight sun) at this latitude."
    elif max_elevation_today < 0:
        # Sun never rises: below the horizon at every point in the day.
        sun_text = "Sun stays below the horizon all day (polar night) at this latitude."
    else:
        sun_text = [
            ("Sunrise: ", sun_info['sunrise'].astimezone(INUVIK_TZ).strftime('%H:%M %Z')),
            ("Sunset: ", sun_info['sunset'].astimezone(INUVIK_TZ).strftime('%H:%M %Z')),
            ("Day length: ", f"{hours}h {minutes}min"),
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
#
# (solar_elevation_deg itself is defined earlier in the file, right before
# the sunrise/sunset text module, since that module now also depends on it
# for robust polar-day/polar-night detection.)
# =========================================================
 
def build_sun_curve_chart():
    """
    Renders a 24-hour solar elevation curve for today (Inuvik local time)
    for Herschel Island, with the current moment marked. The data window
    spans one Inuvik calendar day (midnight to midnight local time); the
    underlying solar position formula still operates on UTC internally
    (as it must, since it's tied to real longitude/UTC physics), but the
    window boundaries and all displayed labels are Inuvik-local so the
    axis and the data genuinely correspond to the same local day.
    Returns (png_bytes, caption).
    """
    try:
        inuvik_day_start_local = now_inuvik.replace(hour=0, minute=0, second=0, microsecond=0)
        times_inuvik = [inuvik_day_start_local + timedelta(minutes=15 * i) for i in range(96)]
        times_utc = [t.astimezone(timezone.utc).replace(tzinfo=None) for t in times_inuvik]
 
        elevations = [solar_elevation_deg(LAT, LON, t) for t in times_utc]
        hour_floats = [t.hour + t.minute / 60 for t in times_inuvik]
 
        current_elevation = solar_elevation_deg(LAT, LON, now)
        current_hour_float = now_inuvik.hour + now_inuvik.minute / 60
 
        NOTION_YELLOW = "#E7B347"
        NOTION_RED = "#E16259"
        NOTION_TEXT_GRAY = "#787774"
        NOTION_LIGHT_GRID = "#EDECEC"
        NOTION_HORIZON = "#D4A72C"
 
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(4.5, 2.8), dpi=150)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
 
        ax.fill_between(hour_floats, elevations, 0, where=[e > 0 for e in elevations],
                         color=NOTION_YELLOW, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(hour_floats, elevations, linewidth=3, color=NOTION_YELLOW, zorder=2)
        ax.axhline(0, color=NOTION_HORIZON, linewidth=1.2, alpha=0.6, zorder=1)
 
        ax.plot([current_hour_float], [current_elevation], marker="o", markersize=18,
                 color=NOTION_YELLOW, markeredgecolor=NOTION_RED, markeredgewidth=2.5, zorder=3)
 
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(NOTION_LIGHT_GRID)
 
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 6))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 6)], fontsize=16, color=NOTION_TEXT_GRAY)
        ax.tick_params(axis="y", labelsize=16, colors=NOTION_TEXT_GRAY, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=NOTION_LIGHT_GRID, linewidth=1, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_ylabel("Elevation (°)", fontsize=17, color=NOTION_TEXT_GRAY)
 
        fig.tight_layout()
        png_bytes = fig_to_png_bytes(fig)
 
        caption = (
            f"Solar elevation today, {inuvik_day_start_local.strftime('%b %d')} (Inuvik local time). "
            f"Computed from standard solar position formulas, not measured."
        )
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
        r = get_with_retry(url, params=params, timeout=20, retries=1)
        data = r.json()
        daily = data.get("daily", {})
        times = daily.get("time", [])
        temps = daily.get("temperature_2m_mean", [])
        return dict(zip(times, temps))
    except Exception as e:
        print(f"HISTORICAL FETCH FAILED for {start_date} to {end_date}:", e)
        return {}
 
 
def prefetch_years_concurrently(years, max_workers=8):
    """
    Pre-fetches any of the given years not already in the on-disk cache,
    running up to max_workers requests concurrently rather than one at a
    time. This is the main lever for cutting this script's total run
    time: the 30-year temperature normal and 25-year TDD histogram were
    previously fetching years one after another, each waiting for the
    full HTTP round-trip (including any retries) before starting the
    next. Open-Meteo's documented rate limit is 600 calls/minute, 5000/
    hour — our worst case of ~50 calls across both loops is trivial
    against that, so this concurrency is safe, not just faster.
 
    Populates the shared _temp_cache (and persists newly-fetched complete
    years to disk via the existing fetch_full_year_cached/save_temp_cache
    machinery) — callers then read from the now-populated cache via their
    normal sequential loop, unchanged.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
 
    years_to_fetch = [y for y in years if str(y) not in _temp_cache]
    if not years_to_fetch:
        return
 
    print(f"PREFETCH: fetching {len(years_to_fetch)} uncached years concurrently (max {max_workers} at once)")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_full_year_cached, year): year for year in years_to_fetch}
        for future in as_completed(futures):
            year = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"PREFETCH FAILED for {year}:", e)
 
 
def fetch_full_year_cached(year):
    """
    Returns the complete Jan 1 - Dec 31 daily temperature dict for the
    given (necessarily past, complete) year — from the on-disk cache if
    already present, otherwise fetched fresh and added to the cache for
    future runs. Never used for the current year, which is always
    in-progress and must be fetched fresh every time.
    """
    global _temp_cache_dirty
 
    cache_key = str(year)
    if cache_key in _temp_cache:
        return _temp_cache[cache_key]
 
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)
    temps = fetch_daily_temps(year_start, year_end)
 
    days_in_year = (year_end - year_start).days + 1
    if len(temps) >= days_in_year * 0.95:
        # Only cache genuinely (near-)complete years — caching a partial
        # year from a bad fetch would permanently "freeze in" an
        # incomplete result, defeating the point of retrying later.
        _temp_cache[cache_key] = temps
        _temp_cache_dirty = True
        print(f"CACHE: fetched and cached {year} ({len(temps)}/{days_in_year} days)")
    else:
        print(f"CACHE: {year} fetch incomplete ({len(temps)}/{days_in_year} days), not caching")
 
    return temps
 
 
# =========================================================
# MODULE 1c — TEMPERATURE CHART: last 30 days vs 30-year daily normal
# =========================================================
def build_temperature_chart():
    """
    Builds a chart of the last 30 days of mean daily temperature against
    the 30-year average for the same calendar days.
 
    The 30-year normal is computed here by pulling the same 30-day
    calendar window from each of the past 30 years and averaging —
    Open-Meteo has no pre-computed "climate normal" endpoint, so this
    is done as 30 separate historical queries (one per year, each
    covering the full 30-day window in a single request).
    """
    end = (now - timedelta(days=1)).date()  # yesterday, since today's mean isn't final yet
    start = end - timedelta(days=29)
 
    recent = fetch_daily_temps(start, end)
    if not recent:
        return None, "No recent historical temperature data returned."
 
    day_labels = sorted(recent.keys())
    recent_values = [recent[d] for d in day_labels]
 
    # Build 30-year normal for the same month/day combinations
    normals_by_day = {d: [] for d in day_labels}
    current_year = now.year
 
    # Pre-fetch all 30 years concurrently (rather than one at a time) —
    # the loop below then reads from the now-populated cache.
    prefetch_years_concurrently([end.year - yb for yb in range(1, 31)])
 
    years_with_data = []  # tracks which specific years actually returned data
 
    for years_back in range(1, 31):
        hist_year = end.year - years_back
        hist_start = start.replace(year=hist_year)
        hist_end = end.replace(year=hist_year)
        full_year_data = fetch_full_year_cached(hist_year)
 
        if not full_year_data:
            continue
 
        # Slice out just the needed window from the full year's data
        hist_data = {
            d: t for d, t in full_year_data.items()
            if hist_start.strftime("%Y-%m-%d") <= d <= hist_end.strftime("%Y-%m-%d")
        }
        if not hist_data:
            continue
 
        years_with_data.append(hist_year)
 
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
    max_years_used = max(years_used_counts) if years_used_counts else 0
    print(f"TEMP CHART: normal built from {min_years_used}-{max_years_used} years of data per day")
    print(f"TEMP CHART: years with at least some data: {sorted(years_with_data)}")
 
    if min_years_used < 15:
        print("TEMP CHART: WARNING — fewer than 15 years of data available for the normal, treat with caution")
 
    # Build a label describing the actual years used, not the theoretical
    # 30-year window — if fetches timed out (a real, recurring issue with
    # this many sequential historical API calls), the years actually
    # contributing data may be fewer than 30, and may have gaps rather
    # than being a clean contiguous range. The label should reflect what
    # actually went into the average, not what was merely intended.
    if years_with_data:
        years_sorted = sorted(years_with_data)
        is_contiguous = years_sorted == list(range(years_sorted[0], years_sorted[-1] + 1))
        if len(years_sorted) == 30 and is_contiguous:
            normal_label = f"{years_sorted[0]}–{years_sorted[-1]} average"
        elif is_contiguous:
            normal_label = f"{years_sorted[0]}–{years_sorted[-1]} average ({len(years_sorted)} years)"
        else:
            normal_label = f"{len(years_sorted)}-year average ({years_sorted[0]}–{years_sorted[-1]}, with gaps)"
    else:
        normal_label = "historical average (no data)"
 
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
             label=normal_label, zorder=2)
 
    ax.plot(x, recent_values, marker="o", markersize=4, linewidth=2,
             color=NOTION_RED, label=f"{current_year} observed",
             markerfacecolor="white", markeredgewidth=1.2, markeredgecolor=NOTION_RED, zorder=3)
 
    # Remove chart border entirely except a faint baseline
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(NOTION_LIGHT_GRID)
 
    # With 30 days instead of 10, show every other date label to avoid
    # overcrowding the x-axis.
    tick_positions = list(x)[::2]
    tick_labels = x_labels[::2]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9, color=NOTION_TEXT_GRAY, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=10, colors=NOTION_TEXT_GRAY, length=0)
    ax.tick_params(axis="x", length=0)
 
    ax.yaxis.grid(True, color=NOTION_LIGHT_GRID, linewidth=1, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
 
    ax.set_ylabel("°C", fontsize=10, color=NOTION_TEXT_GRAY)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=NOTION_TEXT_GRAY)
 
    fig.tight_layout()
 
    png_bytes = fig_to_png_bytes(fig)
 
    # Caption now uses the same real year-range as the legend, so the two
    # can never contradict each other the way "1996-2025" + "18 years" did.
    if min_years_used == max_years_used:
        years_phrase = f"{min_years_used} years" if min_years_used != 30 else "the full 30 years"
    else:
        years_phrase = f"{min_years_used} to {max_years_used} years (varies by day)"
 
    caption = (
        f"Daily mean temperature, last 30 days vs. {normal_label.replace(' average', '')} "
        f"(shaded band ±1.5°C). Normal computed from {years_phrase} of ERA5 data per calendar day."
    )
    return png_bytes, caption
 
 
print("STARTING: temperature chart (30-year historical prefetch)")
temp_chart_bytes, temp_chart_caption = build_temperature_chart()
 
 
# =========================================================
# THAWING DEGREE DAYS HISTOGRAM (one bar per year, full year totals)
# Thawing degree days = cumulative sum of mean daily temperatures above
# 0°C, from Jan 1 through Dec 31 for past years (a real annual total), or
# Jan 1 through yesterday for the current year (necessarily partial,
# highlighted in a different color so it's not mistaken for a complete
# year). Uses the same fetch_daily_temps function already verified for
# the temperature chart, so failures/retries behave identically.
# =========================================================
def compute_tdd_from_temps(daily_temps, start_date, end_date):
    """
    Sums mean daily temps above 0°C from start_date through end_date
    (inclusive). Missing days are skipped, not treated as 0 — using real
    date arithmetic (not hand-rolled year shifting) so leap years are
    handled correctly automatically.
    """
    total = 0.0
    days_counted = 0
    d = start_date
    while d <= end_date:
        temp = daily_temps.get(d.strftime("%Y-%m-%d"))
        if temp is not None:
            days_counted += 1
            if temp > 0:
                total += temp
        d += timedelta(days=1)
    return total, days_counted
 
 
def build_tdd_histogram(num_years=25):
    """
    Builds a bar chart of annual thawing degree days for the past
    num_years complete years, plus the current (partial) year in a
    different color. Returns (png_bytes, caption).
 
    Past complete years are fetched via fetch_full_year_cached(), the
    same on-disk cache used by the temperature chart's 30-year normal —
    so a year already fetched for one chart doesn't need to be fetched
    again for the other. A finished year's data never changes, so this
    avoids re-fetching decades of data from Open-Meteo on every single
    hourly run.
    """
    today = now.date()
    current_year = today.year
 
    tdd_by_year = {}
 
    # Pre-fetch all needed years concurrently — for any year the
    # temperature chart's own prefetch already cached, this is a fast
    # no-op; only genuinely new years trigger new concurrent fetches.
    prefetch_years_concurrently([current_year - yb for yb in range(1, num_years + 1)])
 
    # Past complete years: Jan 1 - Dec 31, via the shared full-year cache
    for years_back in range(1, num_years + 1):
        year = current_year - years_back
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        temps = fetch_full_year_cached(year)
        if not temps:
            print(f"TDD HISTOGRAM: no data for {year}, skipping (will retry next run)")
            continue
        tdd, days_counted = compute_tdd_from_temps(temps, year_start, year_end)
        # Require a reasonable fraction of the year's days to have data,
        # or a single short outage could badly understate that year's TDD.
        # (fetch_full_year_cached already applies a 95% completeness bar
        # before caching, but we double-check here in case a non-cached,
        # genuinely incomplete fetch came back from a failed cache write.)
        days_in_year = (year_end - year_start).days + 1
        if days_counted < days_in_year * 0.8:
            print(f"TDD HISTOGRAM: {year} only has {days_counted}/{days_in_year} days, skipping (too incomplete, will retry next run)")
            continue
        tdd_by_year[year] = tdd
 
    # Current year: Jan 1 - yesterday (today's mean isn't final yet) —
    # always fetched fresh, never cached, since it's still accumulating.
    current_start = date(current_year, 1, 1)
    current_end = today - timedelta(days=1)
    current_temps = fetch_daily_temps(current_start, current_end)
    if current_temps:
        current_tdd, current_days = compute_tdd_from_temps(current_temps, current_start, current_end)
        tdd_by_year[current_year] = current_tdd
 
    if not tdd_by_year:
        return None, "Thawing degree days data unavailable — all fetches failed. Check Action logs."
 
    print(f"TDD HISTOGRAM: years with data: {sorted(tdd_by_year.keys())}")
 
    try:
        NOTION_TEXT_GRAY = "#787774"
        NOTION_BLUE = "#337EA9"
        NOTION_RED = "#E16259"
        NOTION_LIGHT_GRID = "#EDECEC"
 
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
 
        full_year_range = list(range(current_year - num_years, current_year + 1))
        plotted_values = [tdd_by_year.get(y, 0) for y in full_year_range]
        colors = [
            NOTION_RED if y == current_year
            else NOTION_LIGHT_GRID if y not in tdd_by_year  # gap year: shown as a faint placeholder, not a real 0-value bar
            else NOTION_BLUE
            for y in full_year_range
        ]
 
        x = range(len(full_year_range))
        ax.bar(x, plotted_values, color=colors, width=0.7)
 
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(NOTION_LIGHT_GRID)
 
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(y) for y in full_year_range], fontsize=9, color=NOTION_TEXT_GRAY, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=9, colors=NOTION_TEXT_GRAY, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=NOTION_LIGHT_GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel("Thawing degree days (°C·days)", fontsize=10, color=NOTION_TEXT_GRAY)
 
        fig.tight_layout()
        png_bytes = fig_to_png_bytes(fig)
 
        gap_years = [y for y in full_year_range if y not in tdd_by_year and y != current_year]
        gap_note = f" Years with incomplete or missing data ({', '.join(str(y) for y in gap_years)}) are shown empty." if gap_years else ""
        caption = (
            f"Annual thawing degree days (sum of mean daily temperatures above 0°C, Jan 1–Dec 31), "
            f"{full_year_range[0]}–{full_year_range[-2]}. "
            f"Current year ({current_year}, in red) is partial: Jan 1 through {current_end.strftime('%b %d')} only, "
            f"not directly comparable to complete-year totals.{gap_note} Source: Open-Meteo (ERA5)."
        )
        return png_bytes, caption
 
    except Exception as e:
        print("TDD HISTOGRAM RENDER FAILED:", e)
        return None, "Thawing degree days chart could not be generated — see Action logs."
 
 
print("STARTING: TDD histogram (25-year historical prefetch)")
tdd_histogram_bytes, tdd_histogram_caption = build_tdd_histogram()
 
 
# =========================================================
# MODULE 1e — WIND VECTOR CHART (last 30 days)
# Fetches hourly wind speed/direction from the same Open-Meteo historical
# archive used for temperature, aggregates to one vector per day (using
# proper vector averaging — not naive angle averaging, which is wrong
# near the 0/360 boundary), and renders as color-graded direction arrows.
# =========================================================
def wind_to_uv(speed, direction_deg):
    """
    Converts meteorological wind speed/direction (direction = where wind
    comes FROM, standard convention) to u (eastward) / v (northward)
    vector components, for correct vector-based averaging.
    """
    direction_rad = math.radians(direction_deg)
    u = -speed * math.sin(direction_rad)
    v = -speed * math.cos(direction_rad)
    return u, v
 
 
def uv_to_wind(u, v):
    speed = math.hypot(u, v)
    direction_rad = math.atan2(-u, -v)
    direction_deg = math.degrees(direction_rad) % 360
    return speed, direction_deg
 
 
def fetch_hourly_wind_chunk(start_date, end_date):
    """
    Fetches hourly wind speed and direction for a single [start_date,
    end_date] window from Open-Meteo's historical archive. Returns a dict
    {date_str: [(speed, direction), ...]} grouped by calendar day, or {}
    on failure for just this chunk.
    """
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": LAT,
            "longitude": LON,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": "windspeed_10m,winddirection_10m",
            "timezone": "UTC",
        }
        r = get_with_retry(url, params=params, timeout=30, retries=2)
        data = r.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        speeds = hourly.get("windspeed_10m", [])
        directions = hourly.get("winddirection_10m", [])
 
        by_day = {}
        for t, s, d in zip(times, speeds, directions):
            if s is None or d is None:
                continue
            day = t[:10]
            by_day.setdefault(day, []).append((s, d))
        return by_day
    except Exception as e:
        print(f"WIND VECTOR CHUNK FETCH FAILED for {start_date} to {end_date}:", e)
        return {}
 
 
def fetch_hourly_wind(start_date, end_date, chunk_days=10):
    """
    Fetches hourly wind speed/direction across [start_date, end_date] by
    splitting the request into smaller chunks (default 10 days each),
    fetched concurrently rather than one after another. This is more
    resilient than one large request: archive-api.open-meteo.com has
    shown frequent timeouts on this project's larger historical calls,
    and a failure in one chunk only loses that chunk's days rather than
    the entire requested window. Fetching the (typically 3) chunks
    concurrently rather than sequentially is a small but free speed win.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
 
    chunk_ranges = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end_date)
        chunk_ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(days=1)
 
    combined = {}
    with ThreadPoolExecutor(max_workers=len(chunk_ranges) or 1) as executor:
        futures = {executor.submit(fetch_hourly_wind_chunk, s, e): (s, e) for s, e in chunk_ranges}
        for future in as_completed(futures):
            s, e = futures[future]
            try:
                chunk_data = future.result()
                combined.update(chunk_data)
            except Exception as ex:
                print(f"WIND VECTOR CHUNK FAILED for {s} to {e}:", ex)
 
    if not combined:
        print("WIND VECTOR FETCH FAILED: all chunks returned no data")
    elif len(combined) < (end_date - start_date).days:
        print(f"WIND VECTOR FETCH: partial data only — got {len(combined)} of {(end_date - start_date).days + 1} expected days")
 
    return combined
 
 
def build_wind_vector_chart():
    """
    Builds a 30-day wind vector chart: one arrow per day, pointing in the
    direction the wind blows TOWARD (so arrows visually show flow
    direction), colored by speed. Returns (png_bytes, caption).
    """
    end = (now - timedelta(days=1)).date()
    start = end - timedelta(days=29)
 
    by_day = fetch_hourly_wind(start, end)
    if not by_day:
        return None, "Wind vector data unavailable — fetch failed. Check Action logs."
 
    day_labels = sorted(by_day.keys())
    daily_speed = []
    daily_dir = []
    for day in day_labels:
        readings = by_day[day]
        us, vs = [], []
        for s, d in readings:
            u, v = wind_to_uv(s, d)
            us.append(u)
            vs.append(v)
        avg_u, avg_v = sum(us) / len(us), sum(vs) / len(vs)
        speed, direction = uv_to_wind(avg_u, avg_v)
        daily_speed.append(speed)
        daily_dir.append(direction)
 
    try:
        NOTION_TEXT_GRAY = "#787774"
        NOTION_LIGHT_GRID = "#EDECEC"
 
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
 
        x = list(range(len(day_labels)))
        # Arrow components: direction is where wind comes FROM, so the
        # arrow should point in the direction the wind blows TOWARD —
        # that's 180 degrees from the "from" direction. Length is now
        # scaled by speed, so faster days show longer arrows on top of
        # the existing color-by-speed encoding.
        u_arrows = [-math.sin(math.radians(d)) * s for d, s in zip(daily_dir, daily_speed)]
        v_arrows = [-math.cos(math.radians(d)) * s for d, s in zip(daily_dir, daily_speed)]
 
        # Faint baseline showing the shared line all arrow tails sit on —
        # drawn first (zorder=1) so the arrows render on top of it.
        ax.axhline(0, color=NOTION_LIGHT_GRID, linewidth=1.2, zorder=1)
 
        quiv = ax.quiver(
            x, [0] * len(x), u_arrows, v_arrows,
            daily_speed, cmap="plasma", scale=220, width=0.005,
            pivot="tail", clim=(0, 40), zorder=2,  # 0-40 km/h covers typical regional range without making calm days look artificially extreme
        )
 
        cbar = fig.colorbar(quiv, ax=ax, orientation="vertical", pad=0.02, fraction=0.04)
        cbar.set_label("Wind speed (km/h)", fontsize=9, color=NOTION_TEXT_GRAY)
        cbar.ax.tick_params(labelsize=8, colors=NOTION_TEXT_GRAY)
        cbar.outline.set_visible(False)
 
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(NOTION_LIGHT_GRID)
 
        tick_positions = x[::3]
        tick_labels = [datetime.strptime(day_labels[i], "%Y-%m-%d").strftime("%b %d") for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=9, color=NOTION_TEXT_GRAY, rotation=45, ha="right")
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(-1.5, 1.5)
        # Wider left/right margin than before — arrows are anchored at
        # their tail (pivot="tail") and can extend a full speed-scaled
        # length in any direction from each day's x position, so the
        # first/last days' arrows need more room than a simple -1/+1
        # margin to avoid being clipped at the plot edge.
        ax.set_xlim(-2, len(x) + 1)
 
        fig.tight_layout()
        png_bytes = fig_to_png_bytes(fig)
        caption = (
            f"Daily-average wind vectors, last 30 days. Arrows point in the direction "
            f"the wind blows toward; color shows speed. Source: Open-Meteo (ERA5)."
        )
        return png_bytes, caption
 
    except Exception as e:
        print("WIND VECTOR CHART RENDER FAILED:", e)
        return None, "Wind vector chart could not be generated — see Action logs."
 
 
wind_chart_bytes, wind_chart_caption = build_wind_vector_chart()
wind_forecast_chart_bytes, wind_forecast_chart_caption = build_wind_forecast_mini_chart(
    weather.get("hourly_wind_forecast") if weather["status"] == "ok" else None
)
 
 
# =========================================================
# MODULE 2 — SATELLITE: MODIS true color via GIBS WMS
# =========================================================
# Final displayed image size (after rotation + crop).
MODIS_FINAL_SIZE_PX = 1024
 
# GIBS's polar stereographic image is only north-up exactly along its
# central meridian (-45°). Herschel Island is far from that meridian, so
# the raw image comes back rotated relative to true north — the further a
# point is from -45° longitude, the more it's rotated. We fix this by
# fetching a larger image than needed, rotating it so true north points up
# at Herschel Island's location, then cropping back to the final size.
# The oversize factor below covers the worst-case corner loss from
# rotating a square image by any angle, with extra margin for safety.
MODIS_OVERSIZE_FACTOR = 1.2
MODIS_FETCH_SIZE_PX = int(MODIS_FINAL_SIZE_PX * MODIS_OVERSIZE_FACTOR)
 
# Rotation angle (degrees, clockwise) needed to make true north point up
# at Herschel Island. Computed from the meridian convergence: the angle
# between Herschel Island's local meridian and the EPSG:3413 central
# meridian (-45°), found geometrically as the direction from Herschel
# Island's projected position toward the pole (the projection's origin).
MODIS_ROTATION_DEG = 92.0  # PROVISIONAL for Shingle Point — verify visually after first run, adjust if satellite image looks rotated
 
# Polar stereographic bbox (EPSG:3413, meters), centered on Herschel Island,
# sized to the oversized fetch dimensions above (so after rotation and
# crop, the final 1024x1024 frame is fully covered by real imagery, with
# no blank corners introduced by the rotation).
_HERSCHEL_X, _HERSCHEL_Y = -2305418, 88565  # Shingle Point center, verified against pyproj
_half_width_m = 150_000 * MODIS_OVERSIZE_FACTOR
BBOX_3413 = (
    f"{_HERSCHEL_X - _half_width_m:.0f},{_HERSCHEL_Y - _half_width_m:.0f},"
    f"{_HERSCHEL_X + _half_width_m:.0f},{_HERSCHEL_Y + _half_width_m:.0f}"
)
 
 
def build_gibs_url(date_str):
    params = {
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.1.1",
        "LAYERS": "MODIS_Terra_CorrectedReflectance_TrueColor,Coastlines",
        "STYLES": "",
        "FORMAT": "image/png",
        "TRANSPARENT": "false",
        "WIDTH": str(MODIS_FETCH_SIZE_PX),
        "HEIGHT": str(MODIS_FETCH_SIZE_PX),
        "SRS": "EPSG:3413",
        "BBOX": BBOX_3413,
        "TIME": date_str,
    }
    base = "https://gibs.earthdata.nasa.gov/wms/epsg3413/best/wms.cgi"
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{base}?{query}"
 
 
def rotate_to_north_up(png_bytes):
    """
    Rotates the fetched (oversized) polar stereographic image so true
    north points up at Herschel Island, then center-crops to the final
    display size. Returns the original bytes unchanged if this fails for
    any reason, so a rotation bug never blocks the image from displaying.
    """
    try:
        from PIL import Image
        import io as _io
 
        img = Image.open(_io.BytesIO(png_bytes)).convert("RGB")
 
        # Empirically verified (by placing a known due-north test point in a
        # simulated raw image and checking both signs): PIL's rotate() needs
        # the POSITIVE angle here to bring true north to the top. The
        # earlier comment about "PIL rotates counter-clockwise for positive
        # angles, so negate it" was correct in isolation but led to the
        # wrong overall result once combined with how the raw image's pixel
        # rows map to projected y — verified empirically instead of by
        # further sign algebra, since that's what actually caught the bug.
        rotated = img.rotate(MODIS_ROTATION_DEG, resample=Image.BICUBIC, expand=False)
 
        # Center-crop to the final size
        w, h = rotated.size
        left = (w - MODIS_FINAL_SIZE_PX) // 2
        top = (h - MODIS_FINAL_SIZE_PX) // 2
        cropped = rotated.crop((left, top, left + MODIS_FINAL_SIZE_PX, top + MODIS_FINAL_SIZE_PX))
 
        out_buf = _io.BytesIO()
        cropped.save(out_buf, format="PNG")
        return out_buf.getvalue()
    except Exception as e:
        print("MODIS ROTATION FAILED (showing unrotated image instead):", e)
        return png_bytes
 
 
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
 
 
def latlon_to_3413(lat_deg, lon_deg):
    """
    Converts WGS84 lat/lon to EPSG:3413 (Arctic polar stereographic)
    projected meters, using the standard Snyder polar stereographic
    variant B forward formula. Verified against pyproj to sub-meter
    precision for this project's coordinates (Herschel Island, Shingle
    Point, plus the pole and standard-parallel edge cases).
    """
    a = 6378137.0           # WGS84 semi-major axis (meters)
    f = 1 / 298.257223563   # WGS84 flattening
    e2 = 2 * f - f ** 2
    e = math.sqrt(e2)
 
    lat_ts = math.radians(70)    # EPSG:3413 standard parallel (latitude of true scale)
    lon0 = math.radians(-45)     # EPSG:3413 central meridian
 
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
 
    t_c = math.tan(math.pi / 4 - lat_ts / 2) / (
        ((1 - e * math.sin(lat_ts)) / (1 + e * math.sin(lat_ts))) ** (e / 2)
    )
    m_c = math.cos(lat_ts) / math.sqrt(1 - e2 * math.sin(lat_ts) ** 2)
 
    t = math.tan(math.pi / 4 - lat / 2) / (
        ((1 - e * math.sin(lat)) / (1 + e * math.sin(lat))) ** (e / 2)
    )
    rho = a * m_c * (t / t_c)
 
    x = rho * math.sin(lon - lon0)
    y = -rho * math.cos(lon - lon0)
    return x, y
 
 
# UTM zone 7N (EPSG:32607) — genuinely north-up at Herschel Island's
# longitude (-139°, well within zone 7's -138° to -132° range), unlike
# EPSG:3413's polar stereographic projection, which is only north-up at
# its own central meridian (-45°) and is rotated ~94° from true north at
# our longitude. This was the actual cause of the Sentinel-1 image not
# being north-up: requesting EPSG:3413 directly (assuming Sentinel Hub's
# server-side reprojection would automatically be north-up) carries
# exactly the same fundamental issue MODIS has, which is why MODIS needs
# its separate rotate-after-fetch step. UTM avoids that problem entirely
# for Sentinel-1, since we don't need server-side reprojection AND a
# rotation step — just the right CRS choice.
SENTINEL1_UTM_ZONE = 7
SENTINEL1_UTM_EPSG = "32607"
 
 
def latlon_to_utm(lat_deg, lon_deg, zone=SENTINEL1_UTM_ZONE):
    """
    Standard UTM forward projection (WGS84), verified against pyproj to
    sub-meter precision for this project's coordinates.
    """
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    e4 = e2 ** 2
    e6 = e2 ** 3
    k0 = 0.9996
 
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
 
    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    T = math.tan(lat) ** 2
    C = e2 / (1 - e2) * math.cos(lat) ** 2
    A = (lon - lon0) * math.cos(lat)
 
    M = a * (
        (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * lat
        - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * lat)
        + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * lat)
        - (35 * e6 / 3072) * math.sin(6 * lat)
    )
 
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 +
                  (5 - 18 * T + T ** 2 + 72 * C - 58 * (e2 / (1 - e2))) * A ** 5 / 120) + 500000.0
    y = k0 * (M + N * math.tan(lat) * (A ** 2 / 2 +
              (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24 +
              (61 - 58 * T + T ** 2 + 600 * C - 330 * (e2 / (1 - e2))) * A ** 6 / 720))
 
    if lat_deg < 0:
        y += 10000000.0
 
    return x, y


 
 
_HERSCHEL_UTM_X, _HERSCHEL_UTM_Y = latlon_to_utm(LAT, LON)
 
 
def annotate_modis_image(png_bytes, points=None, scale_km=50):
    """
    Draws label markers at the given coordinates and a scale bar on the
    MODIS image. The image itself has already been rotated to north-up
    and cropped to MODIS_FINAL_SIZE_PX by rotate_to_north_up() before this
    function runs. To place points consistently with that same rotation,
    each point's (x, y) offset from Herschel Island's center, in EPSG:3413
    projected meters, is rotated by the same angle used for the image,
    then mapped onto the final square frame (which is centered on
    Herschel Island by construction).
 
    points: list of (lat, lon, label) or (lat, lon, label, text_dy_offset)
    tuples. Defaults to Shingle Point, Herschel Island, Aklavik, and Inuvik if not given.
 
    The scale bar uses a uniform meters-per-pixel value, valid since
    rotation preserves distances and the frame is centered consistently.
 
    Returns annotated PNG bytes, or the original bytes unchanged if
    annotation fails for any reason (so a drawing bug never blocks the
    underlying satellite image from being shown).
    """
    if points is None:
        points = [
            (68.933333, -137.2, "Shingle Point", -28),
            (69.568861, -138.911754, "Qikiqtaruk Herschel Island", -10),
            (68.226653, -135.003294, "Aklavik", -10),
            (68.360741, -133.723022, "Inuvik", -10, -90),
        ]
 
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io as _io
 
        img = Image.open(_io.BytesIO(png_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        width_px, height_px = img.size  # should be MODIS_FINAL_SIZE_PX square
 
        # Meters-per-pixel for the FINAL (post-crop) frame, not the
        # oversized fetch — this is the actual resolution of what's shown.
        meters_per_px = (150_000 * 2) / MODIS_FINAL_SIZE_PX  # final frame covers the original ±150km
 
        rotation_rad = math.radians(MODIS_ROTATION_DEG)
 
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
 
        # --- Label markers ---
        for point in points:
            if len(point) == 5:
                lat, lon, label_text, text_dy, text_dx = point
            elif len(point) == 4:
                lat, lon, label_text, text_dy = point
                text_dx = 12
            else:
                lat, lon, label_text = point
                text_dy = -10
                text_dx = 12
 
            x_m, y_m = latlon_to_3413(lat, lon)
            dx_m = x_m - _HERSCHEL_X
            dy_m = y_m - _HERSCHEL_Y
 
            # Rotate this offset by the same angle used to rotate the
            # image (clockwise by MODIS_ROTATION_DEG), so the point lands
            # in the same relative position it would in the rotated image.
            cos_r, sin_r = math.cos(rotation_rad), math.sin(rotation_rad)
            dx_rot = dx_m * cos_r - dy_m * sin_r
            dy_rot = dx_m * sin_r + dy_m * cos_r
 
            # Map onto the final frame: center of frame = Herschel Island,
            # +x = right, and projected +y = north so -dy_rot = down in
            # image space (image y increases downward).
            x_px = width_px / 2 + dx_rot / meters_per_px
            y_px = height_px / 2 - dy_rot / meters_per_px
 
            marker_radius = 6
            draw.ellipse(
                [x_px - marker_radius, y_px - marker_radius, x_px + marker_radius, y_px + marker_radius],
                fill=(255, 60, 60), outline=(255, 255, 255), width=2,
            )
 
            text_x, text_y = x_px + text_dx, y_px + text_dy
            for tdx, tdy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                draw.text((text_x + tdx, text_y + tdy), label_text, font=font, fill=(0, 0, 0))
            draw.text((text_x, text_y), label_text, font=font, fill=(255, 255, 255))
 
        # --- Yukon/Alaska international border ---
        # The border follows the 141st meridian west exactly (a verified
        # geographic fact, not drawn from an unconfirmed GIBS reference
        # layer — meridians are straight lines radiating from the pole in
        # this projection, so two points fully determine the line).
        def project_point(lat, lon):
            x_m, y_m = latlon_to_3413(lat, lon)
            dx_m = x_m - _HERSCHEL_X
            dy_m = y_m - _HERSCHEL_Y
            cos_r, sin_r = math.cos(rotation_rad), math.sin(rotation_rad)
            dx_rot = dx_m * cos_r - dy_m * sin_r
            dy_rot = dx_m * sin_r + dy_m * cos_r
            x_px = width_px / 2 + dx_rot / meters_per_px
            y_px = height_px / 2 - dy_rot / meters_per_px
            return x_px, y_px
 
        border_lon = -141.0
        p1 = project_point(60.0, border_lon)    # well south of the visible frame
        p2 = project_point(69.65, border_lon)  # the border's actual starting point at the Beaufort Sea coast — verified (69°39'N, 141°W); north of this is open ocean, not a land border
 
        # Dashed line: draw only the "on" segments of a repeating
        # on/off pattern along the p1->p2 line.
        num_dashes = 120
        for i in range(num_dashes):
            if i % 2 != 0:
                continue  # skip every other segment to create the dash gap
            t0, t1 = i / num_dashes, (i + 1) / num_dashes
            seg_p1 = (p1[0] + (p2[0] - p1[0]) * t0, p1[1] + (p2[1] - p1[1]) * t0)
            seg_p2 = (p1[0] + (p2[0] - p1[0]) * t1, p1[1] + (p2[1] - p1[1]) * t1)
            draw.line([seg_p1, seg_p2], fill=(255, 255, 255), width=2)
 
        # --- Scale bar (bottom-left corner) ---
        px_per_km = 1000 / meters_per_px
 
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
 
 
def annotate_plain_image(png_bytes, points=None, scale_km=50, half_width_m=150_000,
                           project_fn=None, center_x=None, center_y=None):
    """
    Draws label markers and a scale bar on an already north-up,
    non-rotated image — unlike annotate_modis_image, which additionally
    rotates each point's offset to match MODIS's own rotate-then-crop
    workflow. Applying that MODIS-specific rotation to an image that was
    never actually rotated was the cause of a real bug.
 
    project_fn/center_x/center_y let the caller specify a different
    projection than the EPSG:3413 default — e.g. Sentinel-1 uses UTM zone
    7N, since that CRS is genuinely north-up at Herschel Island's
    longitude (EPSG:3413 polar stereographic is not, except at its own
    central meridian, which was the root cause of the Sentinel-1
    rotation bug — requesting it directly carries the same fundamental
    issue MODIS has, just without MODIS's compensating rotation step).
 
    points: list of (lat, lon, label) or (lat, lon, label, text_dy_offset)
    tuples. Defaults to Shingle Point, Herschel Island, Aklavik, and Inuvik if not given.
 
    Returns annotated PNG bytes, or the original bytes unchanged if
    annotation fails for any reason.
    """
    if points is None:
        points = [
            (68.933333, -137.2, "Shingle Point", -28),
            (69.568861, -138.911754, "Qikiqtaruk Herschel Island", -10),
            (68.226653, -135.003294, "Aklavik", -10),
            (68.360741, -133.723022, "Inuvik", -10, -90),
        ]
 
    if project_fn is None:
        project_fn = latlon_to_3413
        center_x, center_y = _HERSCHEL_X, _HERSCHEL_Y
 
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io as _io
 
        img = Image.open(_io.BytesIO(png_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        width_px, height_px = img.size
 
        meters_per_px = (half_width_m * 2) / width_px
 
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
 
        def project_point(lat, lon):
            x_m, y_m = project_fn(lat, lon)
            dx_m = x_m - center_x
            dy_m = y_m - center_y
            # NO rotation applied here — the image is already north-up.
            x_px = width_px / 2 + dx_m / meters_per_px
            y_px = height_px / 2 - dy_m / meters_per_px
            return x_px, y_px
 
        # --- Coastline overlay, in white (annotate_plain_image only --
        # i.e. the Sentinel-1/radar image, where seeing the true
        # coastline against the SAR backscatter is genuinely useful,
        # unlike MODIS's true-color photo where the coastline is usually
        # already visible).
        #
        # Uses a small, pre-filtered local extract of OpenStreetMap's
        # natural=coastline data (coastline_data.geojson, committed to
        # this repo -- see README for how it was generated), rather than
        # fetching a global coastline dataset fresh every run.
        #
        # This replaced two earlier approaches, in order:
        #   1. Natural Earth's 1:50m-scale global coastline -- too
        #      coarse for this delta's small channels and bars.
        #   2. NASA GIBS's rasterized Coastlines WMS layer, reprojected
        #      into this image's UTM-7N frame -- technically worked, but
        #      produced visibly pixelated/blocky lines (a rasterize-then
        #      resample pipeline is inherently lossier than drawing
        #      vector lines directly), and GIBS's generic "Coastlines"
        #      layer includes river/delta channel boundaries alongside
        #      the true outer coast, which looked cluttered and busy in
        #      this specific low-relief, multi-channel delta environment.
        #
        # OSM's natural=coastline tag specifically marks the outer
        # ocean coast, distinct from inland water features, which avoids
        # the clutter problem, and drawing it as vector line segments
        # (like Natural Earth before it) avoids the pixelation problem --
        # both fixed by switching data source AND going back to direct
        # vector drawing rather than image compositing.
        try:
            coastline_lon_min = LON - (half_width_m / 111_000) * 1.5
            coastline_lon_max = LON + (half_width_m / 111_000) * 1.5
            coastline_lat_min = LAT - (half_width_m / 111_000) * 1.5
            coastline_lat_max = LAT + (half_width_m / 111_000) * 1.5

            coastline_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "coastline_data.geojson"
            )
            with open(coastline_path) as _f:
                coast_geojson = json.load(_f)

            segments_drawn = 0
            for feature in coast_geojson.get("features", []):
                geom = feature.get("geometry", {})
                if geom.get("type") != "LineString":
                    continue

                coords = geom.get("coordinates", [])
                prev_px = None
                for coord_lon, coord_lat in coords:
                    if not (coastline_lon_min <= coord_lon <= coastline_lon_max and
                            coastline_lat_min <= coord_lat <= coastline_lat_max):
                        prev_px = None  # break the line when it exits our extent
                        continue
                    px = project_point(coord_lat, coord_lon)
                    if prev_px is not None:
                        draw.line([prev_px, px], fill=(255, 255, 255), width=2)
                        segments_drawn += 1
                    prev_px = px

            print(f"SENTINEL-1 COASTLINE: drew {segments_drawn} line segments from local OSM extract")
        except Exception as e:
            print("SENTINEL-1 COASTLINE OVERLAY FAILED (continuing without it):", e)

        for point in points:
            if len(point) == 5:
                lat, lon, label_text, text_dy, text_dx = point
            elif len(point) == 4:
                lat, lon, label_text, text_dy = point
                text_dx = 12
            else:
                lat, lon, label_text = point
                text_dy = -10
                text_dx = 12
 
            x_px, y_px = project_point(lat, lon)
 
            marker_radius = 6
            draw.ellipse(
                [x_px - marker_radius, y_px - marker_radius, x_px + marker_radius, y_px + marker_radius],
                fill=(255, 60, 60), outline=(255, 255, 255), width=2,
            )
 
            text_x, text_y = x_px + text_dx, y_px + text_dy
            for tdx, tdy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                draw.text((text_x + tdx, text_y + tdy), label_text, font=font, fill=(0, 0, 0))
            draw.text((text_x, text_y), label_text, font=font, fill=(255, 255, 255))
 
        # --- Yukon/Alaska international border (141st meridian) ---
        border_lon = -141.0
        p1 = project_point(60.0, border_lon)
        p2 = project_point(69.65, border_lon)
 
        num_dashes = 120
        for i in range(num_dashes):
            if i % 2 != 0:
                continue
            t0, t1 = i / num_dashes, (i + 1) / num_dashes
            seg_p1 = (p1[0] + (p2[0] - p1[0]) * t0, p1[1] + (p2[1] - p1[1]) * t0)
            seg_p2 = (p1[0] + (p2[0] - p1[0]) * t1, p1[1] + (p2[1] - p1[1]) * t1)
            draw.line([seg_p1, seg_p2], fill=(255, 255, 255), width=2)
 
 
        # --- Scale bar (bottom-left corner) ---
        px_per_km = 1000 / meters_per_px
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
        print("PLAIN IMAGE ANNOTATION FAILED (showing unannotated image instead):", e)
        return png_bytes
 
 
def stamp_timestamp(png_bytes, dt_inuvik, label="Acquired"):
    """
    Draws a timestamp in the upper-right corner of a satellite image, in
    Inuvik local time, so it's visually obvious the image is not
    real-time. Returns the stamped PNG bytes, or the original bytes
    unchanged if stamping fails for any reason.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io as _io
 
        img = Image.open(_io.BytesIO(png_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        width_px, height_px = img.size
 
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
 
        text = f"{label}: {dt_inuvik.strftime('%Y-%m-%d %H:%M %Z')}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        margin = 18
        text_x = width_px - text_w - margin
        text_y = margin
 
        for tdx, tdy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            draw.text((text_x + tdx, text_y + tdy), text, font=font, fill=(0, 0, 0))
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))
 
        out_buf = _io.BytesIO()
        img.save(out_buf, format="PNG")
        return out_buf.getvalue()
 
    except Exception as e:
        print("TIMESTAMP STAMP FAILED (showing unstamped image instead):", e)
        return png_bytes
 
 
def fetch_and_process_modis():
    """
    Wraps the full MODIS fetch-rotate-annotate-stamp chain as a single
    function, so it can run concurrently with the other independent
    top-level data fetches (water level, Sentinel-1) via a thread pool,
    rather than running sequentially before them — these three modules
    have no data dependency on each other, so there's no reason for one
    to wait on another to finish.
    """
    modis_bytes, modis_date = fetch_modis_image()
    if modis_bytes:
        modis_bytes = rotate_to_north_up(modis_bytes)
    if modis_bytes:
        modis_bytes = annotate_modis_image(modis_bytes)
    if modis_bytes and modis_date:
        try:
            modis_dt_utc = datetime.strptime(modis_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            modis_bytes = stamp_timestamp(modis_bytes, to_inuvik_time(modis_dt_utc.replace(tzinfo=None)), label="Acquired")
        except Exception as e:
            print("MODIS TIMESTAMP STAMP FAILED:", e)
    return modis_bytes, modis_date
 
 
def fetch_and_process_sentinel1():
    """
    Wraps the full Sentinel-1 token-catalog-image-annotate-stamp chain as
    a single function, for the same concurrent-top-level-fetch reason as
    fetch_and_process_modis(). The internal steps stay sequential (each
    needs the previous step's result — token before catalog search,
    catalog search before image request), but the whole chain runs
    concurrently with MODIS and water level.
 
    Returns (sentinel1_bytes, sentinel1_caption).
    """
    sentinel1_bytes = None
    sentinel1_caption = "Sentinel-1 SAR image unavailable — credentials missing or fetch failed. Check Action logs."
 
    sh_token = get_sentinel_hub_token()
    if sh_token:
        s1_date, s1_full_datetime = find_latest_sentinel1_date(sh_token)
        if s1_date:
            s1_raw = fetch_sentinel1_image(sh_token, s1_date)
            if s1_raw:
                from PIL import Image
                import io as _io
                rgba_img = Image.open(_io.BytesIO(s1_raw)).convert("RGBA")
                background = Image.new("RGBA", rgba_img.size, (50, 50, 50, 255))
                composited = Image.alpha_composite(background, rgba_img).convert("RGB")
                buf = _io.BytesIO()
                composited.save(buf, format="PNG")
                sentinel1_bytes = annotate_plain_image(
                    buf.getvalue(), project_fn=latlon_to_utm,
                    center_x=_HERSCHEL_UTM_X, center_y=_HERSCHEL_UTM_Y,
                )
                try:
                    s1_dt_utc = datetime.strptime(s1_full_datetime[:19], "%Y-%m-%dT%H:%M:%S")
                    sentinel1_bytes = stamp_timestamp(sentinel1_bytes, to_inuvik_time(s1_dt_utc), label="Acquired")
                except Exception as e:
                    print("SENTINEL-1 TIMESTAMP STAMP FAILED:", e)
                sentinel1_caption = (
                    f"Sentinel-1 SAR, VV decibel gamma0 (orthorectified), {s1_date}. "
                    f"Dark gray areas are outside that day's satellite swath coverage. "
                    f"Source: Copernicus Sentinel-1 via Sentinel Hub."
                )
    return sentinel1_bytes, sentinel1_caption
 
 
# =========================================================
# MODULE 4 — TIDES & SEA LEVEL (DFO Canadian Hydrographic Service, IWLS API)
# =========================================================
# IWLS station 06525 = Herschel Island. Unlike the old SPINE API (which only
# covers the St. Lawrence and never had Arctic coverage), IWLS hosts real
# tide-table stations across Canada including the Arctic. Station IDs are
# internal UUIDs, not the public 5-digit code, so we resolve the code to an
# ID first, then request water level predictions (wlp) for that station.
HERSCHEL_STATION_CODE = "06505"  # Shingle Point IWLS station — variable name kept for simplicity, value updated
 
 
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
tide_points = fetch_tide_predictions(station_id, hours_ahead=24*7) if station_id else None
 
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
        "Reference: chart datum, Shingle Point station (06505)",
    ]
else:
    tide_text = (
        "Tide prediction data unavailable for Shingle Point station (06505).\n"
        "Check Action logs — this uses DFO's IWLS API, which requires resolving "
        "the station code to an internal station ID first; if DFO changes that "
        "station's status or the API shape, this lookup may need adjustment."
    )
 
 
# =========================================================
# TIDE FORECAST CURVE
# Renders the same 24h water level predictions already fetched above as a
# chart, styled consistently with the temperature and sun position charts.
# =========================================================
def build_tide_chart(tide_points):
    if not tide_points:
        return None, "Tide chart unavailable — no prediction data."
 
    try:
        sorted_points = sorted(tide_points, key=lambda p: p["eventDate"])
        times = [datetime.fromisoformat(p["eventDate"].replace("Z", "+00:00")) for p in sorted_points]
        levels = [p["value"] for p in sorted_points]
 
        # Hours since the start of the window, for a clean numeric x-axis
        t0 = times[0]
        hours = [(t - t0).total_seconds() / 3600 for t in times]
 
        current_idx = min(range(len(times)), key=lambda i: abs((times[i] - now.replace(tzinfo=timezone.utc)).total_seconds()))
        current_hour = hours[current_idx]
        current_level = levels[current_idx]
 
        NOTION_BLUE = "#337EA9"
        NOTION_RED = "#E16259"
        NOTION_TEXT_GRAY = "#787774"
        NOTION_LIGHT_GRID = "#EDECEC"
 
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(4.8, 3.0), dpi=150)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
 
        ax.fill_between(hours, levels, min(levels), color=NOTION_BLUE, alpha=0.12, linewidth=0, zorder=1)
        ax.plot(hours, levels, linewidth=3, color=NOTION_BLUE, zorder=2)
        ax.plot([current_hour], [current_level], marker="o", markersize=10,
                 color=NOTION_RED, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
 
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(NOTION_LIGHT_GRID)
 
        ax.set_xlim(0, max(hours))
 
        # Build tick positions at clean clock hours (e.g. 22:00, 04:00),
        # not fixed offsets from t0's exact minute (which previously
        # produced labels like "22:36" instead of a round hour). Steps
        # daily (24h) rather than every 6h, since the window is now 7
        # days — 6h ticks would produce 28 crowded labels.
        first_tick_time = t0.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        if first_tick_time < t0:
            first_tick_time += timedelta(hours=1)
        tick_times = []
        t = first_tick_time
        while (t - t0).total_seconds() / 3600 <= max(hours):
            tick_times.append(t)
            t += timedelta(hours=24)
        tick_hours = [(t - t0).total_seconds() / 3600 for t in tick_times]
        tick_labels = [t.astimezone(INUVIK_TZ).strftime("%b %d") for t in tick_times]
        ax.set_xticks(tick_hours)
        ax.set_xticklabels(tick_labels, fontsize=15, color=NOTION_TEXT_GRAY, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=15, colors=NOTION_TEXT_GRAY, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=NOTION_LIGHT_GRID, linewidth=1, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_ylabel("Water level (m)", fontsize=16, color=NOTION_TEXT_GRAY)
 
        # Label the current-time marker directly in red, so "now" is
        # unambiguous rather than relying on the reader to infer it from
        # the dot's position alone. Offset to the right (not straight up)
        # since current_hour sits near the start of the chart (x=0), where
        # an upward offset would crowd the y-axis; a horizontal offset
        # also more reliably clears the curve regardless of its local slope.
        x_range = max(hours) if hours else 24
        x_offset = x_range * 0.035
        ax.annotate(
            "now", xy=(current_hour, current_level),
            xytext=(current_hour + x_offset, current_level),
            color=NOTION_RED, fontsize=16, fontweight="bold",
            ha="left", va="center",
        )
 
        fig.tight_layout()
        png_bytes = fig_to_png_bytes(fig)
        caption = f"Predicted water level, next 7 days, starting {t0.astimezone(INUVIK_TZ).strftime('%b %d, %H:%M %Z')}. Source: DFO/CHS IWLS."
        return png_bytes, caption
 
    except Exception as e:
        print("TIDE CHART FAILED:", e)
        return None, "Tide chart could not be generated — see Action logs."
 
 
tide_chart_bytes, tide_chart_caption = build_tide_chart(tide_points)
 
 
# =========================================================
# MODULE — TOTAL WATER LEVEL (TOPAZ6 Arctic model, tide + storm surge)
# Unlike DFO IWLS (pure astronomical tide prediction from a station),
# this product is a 3km HYCOM model that includes both tides AND storm
# surge — i.e. the actual "total" water level signal, not just the
# predictable tidal component. Dataset and variable verified directly
# from Copernicus's own Product User Manual: dataset
# "dataset-topaz6-arc-15min-3km-be", variable "zos" (meters). Fetched via
# plain xarray against the public THREDDS OPeNDAP endpoint (see
# fetch_copernicus_water_level for why — copernicusmarine's own
# open_dataset() reported no subset-compatible service for this dataset).
# Update frequency is daily (forecast published ~00:30 UTC the next day),
# not hourly — so this won't refresh every run the way other blocks do,
# but the forecast curve itself remains valid and useful between updates.
# =========================================================
 
 
def fetch_copernicus_water_level():
    """
    Fetches the total water level (sea surface height, tide + storm surge)
    forecast for the next ~24h near Shingle Point from the TOPAZ6
    Arctic tide/surge model.
 
    Uses plain xarray against the public THREDDS OPeNDAP endpoint
    (no Copernicus Marine authentication needed for this specific access
    path) rather than the copernicusmarine library's open_dataset(),
    since that wrapper reported "No service available for dataset with
    command subset" for this specific dataset — its Marine Data Store
    catalog apparently doesn't expose a subset-compatible service for
    this dataset, even though the same underlying data is genuinely
    accessible via plain OPeNDAP (confirmed via an independent published
    usage example from the OpenDrift project, a third-party tool, using
    this exact URL).
 
    Returns (times, values_m) as parallel lists, or (None, None) on
    failure — network error or any other issue — so a problem here
    never blocks the rest of the dashboard.
    """
    try:
        import xarray as xr
 
        thredds_url = "https://thredds.met.no/thredds/dodsC/cmems/topaz6/dataset-topaz6-arc-15min-3km-be.ncml"
 
        # The grid is polar stereographic (x/y in meters, same pole and
        # central meridian convention as our existing EPSG:3413 pipeline)
        # rather than plain longitude/latitude, so we convert Herschel
        # Island's coordinates the same way already verified for MODIS.
        target_x_m, target_y_m = latlon_to_3413(LAT, LON)
 
        ds = xr.open_dataset(thredds_url)
 
        # CONFIRMED ROOT CAUSE (after several incorrect prior guesses):
        # this .ncml file's x/y coordinate variables are NOT in plain
        # meters — they're in units of 100km (i.e. meters / 100,000).
        # Verified by cross-checking our own observed coordinate range
        # (-36 to 38) against an independent, real source: OpenDrift's
        # own debug log for this exact file reports real coverage of
        # -3,600,000 to 3,798,000 (meters), and -3,600,000 / 100,000 =
        # -36.0, 3,798,000 / 100,000 = 37.98 — an exact match. Converting
        # our target point and search radius into this same native unit
        # before slicing is the actual fix; everything else (slice
        # direction, neighborhood search for valid cells) was already
        # correct and is unaffected by this unit conversion.
        UNIT_SCALE = 100_000  # meters per native coordinate unit in this file
        target_x = target_x_m / UNIT_SCALE
        target_y = target_y_m / UNIT_SCALE
 
        # The TOPAZ6 grid is 3km resolution; near a coastline like
        # Herschel Island's, the single geometrically-nearest cell can
        # land on a masked/land grid point, which comes back as NaN
        # rather than a clean error. Search a small neighborhood (a few
        # grid cells in each direction) and use the nearest cell that
        # actually has valid data, rather than trusting the nearest
        # match blindly.
        search_radius_m = 50_000
        search_radius = search_radius_m / UNIT_SCALE
        x_coords = ds["x"].values
        y_coords = ds["y"].values
        x_ascending = x_coords[0] < x_coords[-1] if len(x_coords) > 1 else True
        y_ascending = y_coords[0] < y_coords[-1] if len(y_coords) > 1 else True
 
        # xarray's .sel(slice(...)) requires the slice bounds in the SAME
        # order as the underlying coordinate array — slice(low, high) on
        # a descending coordinate silently returns an empty selection,
        # which was the actual cause of "no grid cells found" here.
        x_slice = (
            slice(target_x - search_radius, target_x + search_radius) if x_ascending
            else slice(target_x + search_radius, target_x - search_radius)
        )
        y_slice = (
            slice(target_y - search_radius, target_y + search_radius) if y_ascending
            else slice(target_y + search_radius, target_y - search_radius)
        )
 
        # Diagnostic logging: print the dataset's REAL coordinate ranges
        # alongside what we're requesting, so a failure's actual cause
        # (coordinate mismatch vs. time-range mismatch vs. something else)
        # is visible in the log instead of needing another guess.
        print(f"COPERNICUS WATER LEVEL DEBUG: target (meters): x={target_x_m:.0f}, y={target_y_m:.0f}")
        print(f"COPERNICUS WATER LEVEL DEBUG: target (native 100km units): x={target_x:.3f}, y={target_y:.3f}, search_radius={search_radius:.3f}")
        print(f"COPERNICUS WATER LEVEL DEBUG: dataset x range: {x_coords.min():.0f} to {x_coords.max():.0f} ({'ascending' if x_ascending else 'descending'})")
        print(f"COPERNICUS WATER LEVEL DEBUG: dataset y range: {y_coords.min():.0f} to {y_coords.max():.0f} ({'ascending' if y_ascending else 'descending'})")
        print(f"COPERNICUS WATER LEVEL DEBUG: requested x_slice={x_slice}, y_slice={y_slice}")
 
        nearby = ds["zos"].sel(x=x_slice, y=y_slice)
        print(f"COPERNICUS WATER LEVEL DEBUG: after x/y selection, nearby size={nearby.size}, dims={dict(nearby.sizes)}")
 
        start = now
        end = now + timedelta(days=10)  # full 10-day TOPAZ6 forecast horizon
 
        time_coords = nearby["time"].values
        time_ascending = time_coords[0] < time_coords[-1] if len(time_coords) > 1 else True
        time_slice = slice(start, end) if time_ascending else slice(end, start)
        print(f"COPERNICUS WATER LEVEL DEBUG: dataset time range: {time_coords.min()} to {time_coords.max()} ({'ascending' if time_ascending else 'descending'})")
        print(f"COPERNICUS WATER LEVEL DEBUG: requested time range: {start} to {end}")
        nearby = nearby.sel(time=time_slice)
        print(f"COPERNICUS WATER LEVEL DEBUG: after time selection, nearby size={nearby.size}, dims={dict(nearby.sizes)}")
 
        if nearby.size == 0:
            print("COPERNICUS WATER LEVEL: no grid cells found near Shingle Point in this window")
            return None, None, None
 
        # Vectorized validity check across the WHOLE x/y grid at once,
        # instead of 1,000+ individual .sel() calls in a Python loop (each
        # of which has real xarray indexing overhead — label lookup,
        # bounds checking — that adds up badly when repeated this many
        # times just to check "is there any valid value here"). This
        # computes "has any non-NaN value over time" for every cell in
        # one bulk operation, collapsing the time dimension, leaving a
        # small 2D (y, x) boolean array that's cheap to scan in Python.
        has_valid_data = nearby.notnull().any(dim="time").values  # shape (y, x)
        xs = nearby["x"].values
        ys = nearby["y"].values
 
        best_point = None
        best_dist = None
        for yi_idx, yi in enumerate(ys):
            for xi_idx, xi in enumerate(xs):
                if has_valid_data[yi_idx, xi_idx]:
                    dist = math.hypot(xi - target_x, yi - target_y)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_point = (xi, yi)
 
        if best_point is None:
            print("COPERNICUS WATER LEVEL: no valid (non-NaN) grid cells found near Shingle Point")
            return None, None, None
 
        print(f"COPERNICUS WATER LEVEL: using grid cell at distance {best_dist:.0f}m from Shingle Point")
        point = nearby.sel(x=best_point[0], y=best_point[1])
 
        times = [str(t) for t in point["time"].values]
        raw_values = [float(v) for v in point.values.flatten()]
 
        # Drop any remaining individual NaN time steps (a cell can be
        # valid overall but still have occasional missing time steps)
        # rather than letting a partial-NaN series through silently.
        times_clean = []
        values_clean = []
        for t, v in zip(times, raw_values):
            if not math.isnan(v):
                times_clean.append(t)
                values_clean.append(v)
 
        if not values_clean:
            print("COPERNICUS WATER LEVEL: selected cell had no valid values in this time window")
            return None, None
 
        # Compute a real yearly mean from the past 365 days at this SAME
        # validated good cell (not the 10-day forecast window above) —
        # the dataset's confirmed historical range starts 2018, so a full
        # past year of real data is genuinely available here, from the
        # same already-open dataset, no new data source needed.
        #
        # IMPORTANT: slice by TIME first, then by point — not point first.
        # Yearly mean is a hardcoded constant (see WATER_LEVEL_YEARLY_MEAN
        # near the top of this file), computed once via the separate
        # compute_yearly_mean_once.py script rather than live on every
        # run — this value was confirmed, via real test runs, to
        # sometimes take 30+ seconds (and was never observed to actually
        # finish within that window) against the remote THREDDS server,
        # for a value that barely changes year to year. No live
        # computation, no cache file, no timeout machinery needed.
        yearly_mean = WATER_LEVEL_YEARLY_MEAN
 
        return times_clean, values_clean, yearly_mean
 
    except Exception as e:
        print("COPERNICUS WATER LEVEL FETCH FAILED:", e)
        return None, None, None
 
 
def build_water_level_chart(times, values, yearly_mean=None):
    if not times or not values:
        return None, "Total water level chart unavailable — no data."
 
    try:
        parsed_times = [datetime.fromisoformat(t.split(".")[0]) for t in times]
        t0 = parsed_times[0]
        hours = [(t - t0).total_seconds() / 3600 for t in parsed_times]
 
        NOTION_PURPLE = "#9065B0"
        NOTION_RED = "#E16259"
        NOTION_GRAY_LINE = "#9B9A97"
        NOTION_TEXT_GRAY = "#787774"
        NOTION_LIGHT_GRID = "#EDECEC"
 
        # When the yearly mean is available, plot relative to it (meters
        # above/below the past-year average) rather than absolute meters
        # — the comparison to "typical" is the actually useful number,
        # and a y-axis genuinely labeled this way communicates it more
        # clearly than an absolute-meter axis plus a separately-labeled
        # dashed reference line competing for space with the curve.
        if yearly_mean is not None:
            plot_values = [v - yearly_mean for v in values]
            ylabel = "Water level vs. yearly average (m)"
        else:
            plot_values = values
            ylabel = "Total water level (m)"
 
        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=150)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
 
        ax.fill_between(hours, plot_values, min(plot_values), color=NOTION_PURPLE, alpha=0.12, linewidth=0, zorder=1)
        ax.plot(hours, plot_values, linewidth=2.5, color=NOTION_PURPLE, zorder=2)
        ax.plot([hours[0]], [plot_values[0]], marker="o", markersize=8,
                 color=NOTION_RED, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
 
        # Reference line at zero (= the yearly average itself) when
        # plotting relative values, so "typical" is a clean, simple
        # horizontal line rather than needing its own value label.
        if yearly_mean is not None:
            ax.axhline(0, color=NOTION_GRAY_LINE, linewidth=1.5, linestyle="--", zorder=1.5)
            # The label's Y position uses axes-fraction coordinates (a
            # fixed spot near the top of the plot area) rather than the
            # actual data value 0 -- the relative-to-mean curve regularly
            # crosses zero, including near the right edge where the label
            # sits horizontally, which previously made the label overlap
            # and become unreadable against the curve. Axes-fraction
            # positioning is independent of the data's actual shape, so
            # this can't happen regardless of how the forecast looks.
            ax.text(0.99, 0.97, f"yearly avg ({yearly_mean:.2f}m)",
                    color=NOTION_GRAY_LINE, fontsize=8, va="top", ha="right",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75))
 
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(NOTION_LIGHT_GRID)
 
        ax.set_xlim(0, max(hours))
        tick_hours = list(range(0, int(max(hours)) + 1, 24))
        tick_labels = [(t0 + timedelta(hours=h)).replace(tzinfo=timezone.utc).astimezone(INUVIK_TZ).strftime("%b %d") for h in tick_hours]
        ax.set_xticks(tick_hours)
        ax.set_xticklabels(tick_labels, fontsize=9, color=NOTION_TEXT_GRAY, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=9, colors=NOTION_TEXT_GRAY, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=NOTION_LIGHT_GRID, linewidth=1, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_ylabel(ylabel, fontsize=10, color=NOTION_TEXT_GRAY)
 
        fig.tight_layout()
        png_bytes = fig_to_png_bytes(fig)
        forecast_days = round(max(hours) / 24)
        start_label = (t0.replace(tzinfo=timezone.utc).astimezone(INUVIK_TZ)).strftime('%b %d, %H:%M %Z')
        caption = f"Total water level (tide + storm surge), {forecast_days}-day forecast, starting {start_label}. Source: TOPAZ6 (Copernicus Marine)."
        return png_bytes, caption
 
    except Exception as e:
        print("WATER LEVEL CHART FAILED:", e)
        return None, "Water level chart could not be generated — see Action logs."
 
 
# =========================================================
# MODULE — NAPOIAK CHANNEL WATER LEVEL (Mackenzie River, Napoiak Channel
# above Shallow Bay, ECCC Water Survey of Canada station 10MC023)
# Distinct from the Total Water Level module above: that's a model
# forecast of tide + storm surge at the coast, while this is a real,
# directly-measured river water level at a station well upstream in the
# Mackenzie Delta, from ECCC's public real-time hydrometric CSV datamart
# (no authentication required). The "daily" frequency file already
# contains exactly the last 30 complete days plus the current incomplete
# day, so no separate historical-archive request is needed for a 30-day
# view, unlike the temperature/wind charts above.
#
# NOTE: this station (10MC023) reports Water Level only, not Discharge —
# confirmed empirically (the Discharge column came back empty for every
# row on a real run) rather than assumed from documentation alone. Not
# every WSC hydrometric station measures both products.
#
# Source format confirmed against ECCC's own documentation:
# https://eccc-msc.github.io/open-data/msc-data/obs_hydrometric/readme_hydrometric-datamart_en/
# Column layout: ID,Date,Water Level (m),Grade,Symbol,QA/QC,Discharge (cms),Grade,Symbol,QA/QC
# =========================================================
NAPOIAK_STATION_ID = "10MC023"
NAPOIAK_PROVTERR = "NT"
NAPOIAK_URL = (
    f"https://dd.weather.gc.ca/today/hydrometric/csv/{NAPOIAK_PROVTERR}/daily/"
    f"{NAPOIAK_PROVTERR}_{NAPOIAK_STATION_ID}_daily_hydrometric.csv"
)


def fetch_napoiak_water_level():
    """
    Fetches the last ~30 days of daily water level (m) for the Napoiak
    Channel station from ECCC's public hydrometric CSV datamart.

    Returns (times, values_m) as parallel lists (naive local datetimes,
    UTC-offset suffix stripped since a once-per-day value doesn't need
    timezone math for a 30-day chart), or (None, None) on failure, so a
    problem here never blocks the rest of the dashboard.
    """
    try:
        # dd.weather.gc.ca and weather.gc.ca have both shown transient
        # connection timeouts together on the same run (seen in
        # practice), suggesting shared underlying infrastructure -- same
        # retry treatment as the marine forecast fetch above.
        resp = get_with_retry(NAPOIAK_URL, timeout=20, retries=2, backoff_seconds=5)

        import csv as _csv

        # First line is a bilingual header, not data — skip it.
        lines = resp.text.splitlines()
        print(f"NAPOIAK DEBUG: fetched {len(resp.content)} bytes, {len(lines)} lines total")
        if lines:
            print(f"NAPOIAK DEBUG: header line: {lines[0]!r}")
        for sample_row in lines[1:4]:
            print(f"NAPOIAK DEBUG: sample data row: {sample_row!r}")

        reader = _csv.reader(lines[1:])

        times = []
        values_m = []
        rows_seen = 0
        rows_too_short = 0
        rows_empty_level = 0
        rows_parse_failed = 0
        for row in reader:
            rows_seen += 1
            if len(row) < 3:
                rows_too_short += 1
                continue
            date_str = row[1].strip()
            level_str = row[2].strip()
            if not level_str:
                rows_empty_level += 1
                continue
            try:
                # Timestamps look like "2026-06-24T00:00:00-07:00" — keep
                # only the naive local date/time portion (first 19 chars,
                # "YYYY-MM-DDTHH:MM:SS"), dropping the UTC-offset suffix.
                t = datetime.fromisoformat(date_str[:19])
                v = float(level_str)
            except Exception:
                rows_parse_failed += 1
                continue
            times.append(t)
            values_m.append(v)

        print(f"NAPOIAK DEBUG: rows_seen={rows_seen}, rows_too_short={rows_too_short}, "
              f"rows_empty_level={rows_empty_level}, rows_parse_failed={rows_parse_failed}, "
              f"rows_kept={len(values_m)}")

        if not values_m:
            print("NAPOIAK: fetch succeeded but no usable water level values found in CSV "
                  "— see NAPOIAK DEBUG lines above for the actual column layout")
            return None, None

        print(f"NAPOIAK: parsed {len(values_m)} daily values from {NAPOIAK_URL}")
        return times, values_m

    except Exception as e:
        print("NAPOIAK FETCH FAILED:", e)
        return None, None


def build_napoiak_chart(times, values_m):
    if not times or not values_m:
        return None, "Napoiak Channel water level chart unavailable — no data."

    try:
        t0 = times[0]
        hours = [(t - t0).total_seconds() / 3600 for t in times]

        NOTION_GREEN = "#4F9768"
        NOTION_RED = "#E16259"
        NOTION_TEXT_GRAY = "#787774"
        NOTION_LIGHT_GRID = "#EDECEC"

        plt.rcParams["font.family"] = "DejaVu Sans"
        fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=150)
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        ax.fill_between(hours, values_m, min(values_m), color=NOTION_GREEN, alpha=0.12, linewidth=0, zorder=1)
        ax.plot(hours, values_m, linewidth=2.5, color=NOTION_GREEN, zorder=2)
        ax.plot([hours[-1]], [values_m[-1]], marker="o", markersize=8,
                 color=NOTION_RED, markeredgecolor="white", markeredgewidth=1.5, zorder=3)

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(NOTION_LIGHT_GRID)

        ax.set_xlim(0, max(hours))
        tick_hours = list(range(0, int(max(hours)) + 1, 5 * 24))  # every 5 days, to avoid crowding over a 30-day span
        tick_labels = [(t0 + timedelta(hours=h)).strftime("%b %d") for h in tick_hours]
        ax.set_xticks(tick_hours)
        ax.set_xticklabels(tick_labels, fontsize=9, color=NOTION_TEXT_GRAY, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=9, colors=NOTION_TEXT_GRAY, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=NOTION_LIGHT_GRID, linewidth=1, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_ylabel("Water level (m)", fontsize=10, color=NOTION_TEXT_GRAY)

        fig.tight_layout()
        png_bytes = fig_to_png_bytes(fig)
        span_days = round(max(hours) / 24)
        end_label = times[-1].strftime("%b %d, %Y")
        caption = (
            f"Mackenzie River water level, Napoiak Channel above Shallow Bay, "
            f"past {span_days} days, ending {end_label}. "
            f"Source: ECCC Water Survey of Canada, station {NAPOIAK_STATION_ID} (real-time, preliminary/unreviewed)."
        )
        return png_bytes, caption

    except Exception as e:
        print("NAPOIAK CHART FAILED:", e)
        return None, "Napoiak Channel water level chart could not be generated — see Action logs."

# water_level_chart_bytes/caption are built later, right after the
# parallel executor block produces copernicus_times/copernicus_values.
 
 
# =========================================================
# UPLOAD ANY VALID IMAGES TO NOTION
# =========================================================
# modis_block/modis_caption are built later, right after the parallel
# executor block produces modis_bytes/modis_date — see below.
 
temp_chart_block = None
if temp_chart_bytes:
    try:
        uid = upload_image_to_notion(temp_chart_bytes, "temp_chart.png")
        temp_chart_block = image_block_from_upload(uid)
    except Exception as e:
        print("TEMP CHART NOTION UPLOAD FAILED:", e)
        temp_chart_caption = "Chart generated but upload to Notion failed — see Action logs."
 
tdd_histogram_block = None
if tdd_histogram_bytes:
    try:
        uid = upload_image_to_notion(tdd_histogram_bytes, "tdd_histogram.png")
        tdd_histogram_block = image_block_from_upload(uid)
    except Exception as e:
        print("TDD HISTOGRAM NOTION UPLOAD FAILED:", e)
        tdd_histogram_caption = "Thawing degree days chart generated but upload to Notion failed — see Action logs."
 
wind_chart_block = None
if wind_chart_bytes:
    try:
        uid = upload_image_to_notion(wind_chart_bytes, "wind_chart.png")
        wind_chart_block = image_block_from_upload(uid)
    except Exception as e:
        print("WIND CHART NOTION UPLOAD FAILED:", e)
        wind_chart_caption = "Wind chart generated but upload to Notion failed — see Action logs."
 
weather_icon_block = None
if weather_icon_big_bytes:
    try:
        uid = upload_image_to_notion(weather_icon_big_bytes, "weather_icon.png")
        weather_icon_block = image_block_from_upload(uid)
    except Exception as e:
        print("WEATHER ICON NOTION UPLOAD FAILED:", e)
 
wind_icon_block = None
if wind_icon_big_bytes:
    try:
        uid = upload_image_to_notion(wind_icon_big_bytes, "wind_icon.png")
        wind_icon_block = image_block_from_upload(uid)
    except Exception as e:
        print("WIND ICON NOTION UPLOAD FAILED:", e)
 
mini_forecast_strip_block = None
if mini_forecast_strip_bytes:
    try:
        uid = upload_image_to_notion(mini_forecast_strip_bytes, "mini_forecast_strip.png")
        mini_forecast_strip_block = image_block_from_upload(uid)
    except Exception as e:
        print("MINI FORECAST STRIP NOTION UPLOAD FAILED:", e)
 
large_forecast_strip_block = None
if large_forecast_strip_bytes:
    try:
        uid = upload_image_to_notion(large_forecast_strip_bytes, "large_forecast_strip.png")
        large_forecast_strip_block = image_block_from_upload(uid)
    except Exception as e:
        print("LARGE FORECAST STRIP NOTION UPLOAD FAILED:", e)
 
wind_forecast_chart_block = None
if wind_forecast_chart_bytes:
    try:
        uid = upload_image_to_notion(wind_forecast_chart_bytes, "wind_forecast_chart.png")
        wind_forecast_chart_block = image_block_from_upload(uid)
    except Exception as e:
        print("WIND FORECAST CHART NOTION UPLOAD FAILED:", e)
 
sun_chart_block = None
if sun_chart_bytes:
    try:
        uid = upload_image_to_notion(sun_chart_bytes, "sun_chart.png")
        sun_chart_block = image_block_from_upload(uid)
    except Exception as e:
        print("SUN CHART NOTION UPLOAD FAILED:", e)
        sun_chart_caption = "Sun chart generated but upload to Notion failed — see Action logs."
 
tide_chart_block = None
if tide_chart_bytes:
    try:
        uid = upload_image_to_notion(tide_chart_bytes, "tide_chart.png")
        tide_chart_block = image_block_from_upload(uid)
    except Exception as e:
        print("TIDE CHART NOTION UPLOAD FAILED:", e)
        tide_chart_caption = "Tide chart generated but upload to Notion failed — see Action logs."
 
water_level_chart_block = None
# (built later, right after the executor block produces copernicus_times/copernicus_values — see below)
 
 
# =========================================================
# ASSEMBLE DASHBOARD BLOCKS
# =========================================================
AWI_LOGO_URL = "https://www.awi.de/_assets/978631966794c5093250775de182779d/Images/AWI/awi_logo.svg"
 
logo_png_bytes = fetch_and_convert_logo_to_png(AWI_LOGO_URL, output_width=120)
logo_block = None
if logo_png_bytes:
    try:
        uid = upload_image_to_notion(logo_png_bytes, "awi_logo.png")
        logo_block = image_block_from_upload(uid)
    except Exception as e:
        print("AWI LOGO NOTION UPLOAD FAILED:", e)
 
# Fall back to the original external SVG embed only if the fetch/convert
# step itself failed — better to show something (even if oversized on
# mobile) than nothing at all.
logo_column = [logo_block] if logo_block else [external_image_block(AWI_LOGO_URL)]
attribution_column = [
    paragraph(
        "This dashboard is provided by the Alfred Wegener Institute Helmholtz Centre "
        "for Polar and Marine Research."
    )
]
 
blocks = [
    columns(logo_column, attribution_column, width_ratios=[0.2, 0.8]),
    divider(),
    paragraph(f"Last update: {now_inuvik.strftime('%Y-%m-%d %H:%M %Z')}"),
    paragraph(
        "All times shown on this page are local Mountain Time for Inuvik, NT "
        "(automatically adjusts for daylight saving)."
    ),
    divider(),
]
 
# --- TODAY'S CONDITIONS — compact snapshot cards, 2x2 grid ---
# This is the most important, fastest-scanning part of the page, so it
# comes first, before any imagery — each card shows only the current
# value, no charts. The fuller 30-day/multi-day charts for wind and tide
# remain further down the page in their own detailed sections, unchanged.
blocks.append(heading("📍 Today's Conditions"))
 
weather_card = [
    heading("🌡 Weather", level=3),
    callout(
        weather_text,
        emoji="🌡",
        color="blue_background",
        children=[b for b in [weather_icon_block, mini_forecast_strip_block] if b] or None,
    ),
    link_paragraph(
        "Full weather data →",
        f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=temperature_2m,relative_humidity_2m,pressure_msl&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto",
        prefix=f"{weather_source_text}  ", prefix_gray=True,
    ),
]
 
wind_card = [
    heading("🧭 Wind", level=3),
    callout(
        wind_now_text,
        emoji="🧭",
        color="blue_background",
        children=[b for b in [wind_icon_block, wind_forecast_chart_block] if b] or None,
    ),
    link_paragraph(
        "Full wind data →",
        f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=windspeed_10m,winddirection_10m&timezone=auto",
        prefix=f"{wind_source_text}  ", prefix_gray=True,
    ),
]
 
tide_card = [
    heading("🌊 Tide", level=3),
    callout(
        tide_text,
        emoji="🌊",
        color="blue_background",
        children=[tide_chart_block] if tide_chart_block else None,
    ),
    link_paragraph(
        "Full station data →",
        f"https://www.tides.gc.ca/en/stations/{HERSCHEL_STATION_CODE}",
        prefix=f"{tide_chart_caption if tide_chart_bytes else 'Tide chart could not be generated — see Action logs.'}  ",
        prefix_gray=True,
    ),
]
 
sun_card = [
    heading("☀️ Sun", level=3),
    callout(
        sun_text,
        emoji="☀️",
        color="blue_background",
        children=[sun_chart_block] if sun_chart_block else None,
    ),
    link_paragraph(
        "Full sun data →",
        f"https://api.sunrise-sunset.org/json?lat={LAT}&lng={LON}&formatted=0",
        prefix=f"{sun_chart_caption if sun_chart_bytes else 'Sun position chart could not be generated — see Action logs.'}  ",
        prefix_gray=True,
    ),
]
 
blocks.append(columns(weather_card, wind_card))
blocks.append(columns(tide_card, sun_card))
blocks.append(divider())
 
# --- Weather / coastal flood alerts: ONLY shown when something is active ---
if active_alerts:
    alert_lines = []
    for a in active_alerts[:5]:
        if a["summary"] and a["summary"] != a["title"]:
            alert_lines.append([("", a["title"]), ": ", a["summary"]])
        else:
            alert_lines.append(a["title"])
    alert_lines.append("Source: Environment Canada")
 
    blocks.append(heading("⚠️ Active Weather Alerts"))
    blocks.append(callout(alert_lines, emoji="⚠️", color="yellow_background"))
    blocks.append(
        link_paragraph("See full alert details →", active_alerts[0]["link"])
        if active_alerts[0].get("link")
        else paragraph("")
    )
    blocks.append(divider())
 
blocks.append(divider())
 
# --- Weather Forecast and Marine Forecast — moved here, right after
# Today's Conditions, before the satellite imagery. ---
blocks.append(heading("📅 Weather Forecast — next 5 days", level=3))
blocks.append(callout(
    "5-day outlook:",
    emoji="📅",
    color="purple_background",
    children=[large_forecast_strip_block] if large_forecast_strip_block else None,
))
blocks.append(gray_caption(land_forecast_caption))
 
blocks.append(divider())
 
blocks.append(heading("⚓ Marine Forecast — Yukon Coast", level=3))
blocks.append(callout(marine_text, emoji="⚓", color="purple_background"))
blocks.append(link_paragraph(
    "Explore here →", "https://weather.gc.ca/marine/forecast_e.html?mapID=07&siteID=16000",
    prefix=f"{marine_source_text}  ", prefix_gray=True,
))
 
blocks.append(divider())
 
_water_level_insertion_index = len(blocks)
 
# --- MODIS section's blocks.append() calls happen later, right after the
# parallel executor block produces modis_block/modis_date/modis_caption
# — see below, near the Sentinel-1 section. The two are appended to
# `blocks` in the right order there (MODIS before Sentinel-1), so the
# page's visual layout is unaffected by this relocation. ---
 
# --- Sentinel-1 SAR (VV decibel gamma0, orthorectified) ---
SENTINEL1_LAYER_ID = "IW-DV-VV-DECIBEL-GAMMA0-ORTHORECTIFIED"
SENTINEL1_DATASET_ID = "S1_CDAS_IW_VVVH"
 
sentinel1_date_str = now.strftime("%Y-%m-%d")
sentinel1_from = f"{sentinel1_date_str}T00%3A00%3A00.000Z"
sentinel1_to = f"{sentinel1_date_str}T23%3A59%3A59.999Z"
sentinel1_url = (
    f"https://browser.dataspace.copernicus.eu/?zoom=10"
    f"&lat={LAT}&lng={LON}"
    f"&themeId=DEFAULT-THEME"
    f"&datasetId={SENTINEL1_DATASET_ID}"
    f"&fromTime={sentinel1_from}&toTime={sentinel1_to}"
    f"&layerId={SENTINEL1_LAYER_ID}"
    f"&cloudCoverage=30&dateMode=TIME%20RANGE"
)
 
 
def get_sentinel_hub_token():
    """
    Obtains an OAuth2 access token from Copernicus Data Space Ecosystem's
    identity service using client credentials. Returns the token string,
    or None on failure.
    """
    client_id = os.environ.get("SENTINEL_HUB_CLIENT_ID")
    client_secret = os.environ.get("SENTINEL_HUB_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("SENTINEL-1: credentials not found in environment, skipping")
        return None
 
    try:
        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        resp = requests.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]
    except Exception as e:
        print("SENTINEL-1 TOKEN REQUEST FAILED:", e)
        return None
 
 
def find_latest_sentinel1_date(token, lookback_days=10):
    """
    Searches the Catalog API for the most recent Sentinel-1 GRD scene that
    actually covers Herschel Island (not just anywhere in a broad search
    area) within the lookback window. Returns a date string (YYYY-MM-DD)
    or None if nothing was found / the search failed.
 
    Two layers of filtering: the search bbox itself is tight (matching
    our actual ~150km display half-width, not an arbitrarily larger
    area), and each candidate result's own bbox is additionally checked
    to confirm it genuinely contains Herschel Island's coordinates with a
    safety margin — since a scene can overlap a search box at a corner
    without actually covering the specific point we care about.
    """
    try:
        url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
        date_to = now
        date_from = now - timedelta(days=lookback_days)
 
        # Tight search box matching our actual display extent (~150km
        # half-width), converted from EPSG:3413 meters to an approximate
        # WGS84 lat/lon box for the Catalog API.
        half_width_km = 150
        lat_buffer = half_width_km / 111
        lon_buffer = half_width_km / (111 * math.cos(math.radians(LAT)))
        search_bbox = [LON - lon_buffer, LAT - lat_buffer, LON + lon_buffer, LAT + lat_buffer]
 
        body = {
            "bbox": search_bbox,
            "datetime": f"{date_from.strftime('%Y-%m-%dT%H:%M:%SZ')}/{date_to.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "collections": ["sentinel-1-grd"],
            "limit": 20,
        }
        resp = requests.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            print("SENTINEL-1: no scenes found in catalog search window")
            return None, None
 
        # Require each candidate's own bbox to genuinely contain Herschel
        # Island with a real safety margin (~30km), not just barely touch
        # the point — the marker dot and its label text need actual image
        # data underneath them, not the gray "uncovered" background. Now
        # uses the ACTUAL display half-width (150km, matching the real
        # image extent requested later), not an arbitrary smaller margin
        # — testing only a small bubble around the bare center point let
        # through scenes that covered the center but left large parts of
        # the actual visible 300km-square frame in the gray, uncovered
        # area (the original bug you reported).
        display_half_width_km = 150
        half_width_deg_lat = display_half_width_km / 111
        half_width_deg_lon = display_half_width_km / (111 * math.cos(math.radians(LAT)))
 
        def _point_in_ring(lon, lat, ring):
            """
            Standard ray-casting point-in-polygon test against a single
            linear ring (list of [lon, lat] coordinate pairs). No new
            dependency (e.g. shapely) needed for this — it's a compact,
            well-known algorithm. Capped at 1000 vertices as a defensive
            bound, since this processes externally-controlled API data
            with no guaranteed upper bound on complexity.
            """
            if len(ring) > 1000:
                ring = ring[:1000]
            n = len(ring)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = ring[i][0], ring[i][1]
                xj, yj = ring[j][0], ring[j][1]
                if ((yi > lat) != (yj > lat)) and (
                    lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-15) + xi
                ):
                    inside = not inside
                j = i
            return inside
 
        def _point_covered_by_geometry(lon, lat, geometry, half_width_deg_lon, half_width_deg_lat, grid_n=5):
            """
            Checks coverage with two real, distinct requirements:
            1. The exact center point (where the Herschel Island marker
               and its label actually render) must be covered, with a
               small real margin around it — this is the requirement
               that actually matters for "is the dot on real data."
            2. At least some reasonable minimum of the wider display
               frame must also be covered (so the image isn't almost
               entirely gray), but NOT a strict majority — a long narrow
               Sentinel-1 swath can legitimately leave large parts of a
               300km square frame uncovered while still giving a
               perfectly usable image with the dot clearly on real data.
 
            The earlier version required 70% of the full grid covered,
            which was too strict and rejected scenes that would have
            looked fine — the dot on real data, just with some genuine
            gray area elsewhere in the frame (which is honest, correct
            behavior to show, not a reason to reject the whole scene).
            """
            if not geometry:
                return False
            gtype = geometry.get("type")
            coords = geometry.get("coordinates")
            if gtype == "Polygon":
                rings = [coords[0]]  # outer ring only; ignoring holes is fine for a coverage check
            elif gtype == "MultiPolygon":
                rings = [poly[0] for poly in coords]
            else:
                return False
 
            # Requirement 1: center point + a small real margin (~15% of
            # the half-width, i.e. ~22km at our 150km half-width) must be
            # covered, so the marker dot and its label render on real data.
            center_margin_frac = 0.15
            center_test_points = [
                (lon, lat),
                (lon - half_width_deg_lon * center_margin_frac, lat),
                (lon + half_width_deg_lon * center_margin_frac, lat),
                (lon, lat - half_width_deg_lat * center_margin_frac),
                (lon, lat + half_width_deg_lat * center_margin_frac),
            ]
            for tlon, tlat in center_test_points:
                if not any(_point_in_ring(tlon, tlat, ring) for ring in rings):
                    return False
 
            # Requirement 2: a low minimum fraction of the wider frame
            # also needs real data, just to avoid an almost-entirely-gray
            # image — not a strict majority.
            offsets = [-1.0, -0.5, 0.0, 0.5, 1.0][:grid_n] if grid_n == 5 else \
                [i / (grid_n - 1) * 2 - 1 for i in range(grid_n)]
 
            test_points = [
                (lon + dx * half_width_deg_lon, lat + dy * half_width_deg_lat)
                for dx in offsets for dy in offsets
            ]
 
            covered_count = sum(
                1 for tlon, tlat in test_points
                if any(_point_in_ring(tlon, tlat, ring) for ring in rings)
            )
            return covered_count / len(test_points) >= 0.20
 
        covering_features = []
        for f in features:
            geometry = f.get("geometry")
            if geometry and _point_covered_by_geometry(LON, LAT, geometry, half_width_deg_lon, half_width_deg_lat):
                covering_features.append(f)
                continue
            # Fallback: bbox-coverage approximation, only used if geometry
            # is missing from this particular feature's response. Uses
            # the same smaller center-margin requirement as the geometry
            # check above, not the full display half-width.
            if not geometry:
                fbbox = f.get("bbox")
                if fbbox and len(fbbox) >= 4:
                    fminx, fminy, fmaxx, fmaxy = fbbox[0], fbbox[1], fbbox[2], fbbox[3]
                    bbox_margin_lon = half_width_deg_lon * 0.15
                    bbox_margin_lat = half_width_deg_lat * 0.15
                    if (fminx + bbox_margin_lon <= LON <= fmaxx - bbox_margin_lon and
                            fminy + bbox_margin_lat <= LAT <= fmaxy - bbox_margin_lat):
                        covering_features.append(f)
 
        if not covering_features:
            print("SENTINEL-1: scenes found nearby, but none actually cover Shingle Point with margin")
            return None, None
 
        covering_features.sort(key=lambda f: f["properties"]["datetime"], reverse=True)
        latest_datetime = covering_features[0]["properties"]["datetime"]
        print(f"SENTINEL-1: latest scene covering Shingle Point: {latest_datetime}")
        return latest_datetime[:10], latest_datetime  # (YYYY-MM-DD for the request, full datetime for display)
 
    except Exception as e:
        print("SENTINEL-1 CATALOG SEARCH FAILED:", e)
        return None, None
 
 
def fetch_sentinel1_image(token, date_str):
    """
    Requests a VV decibel gamma0 orthorectified Sentinel-1 image for the
    given date, directly reprojected to EPSG:3413 at the same extent used
    for the MODIS image — no separate rotation step needed here, since
    Sentinel Hub reprojects server-side (unlike GIBS/WMS for MODIS, which
    only serves north-up-at-its-own-central-meridian and needed the
    oversized-fetch-then-rotate workaround).
 
    The dB conversion and 0-255 grayscale scaling are both done inside the
    evalscript (server-side), so the response is a ready-to-use 8-bit PNG
    with alpha — avoiding any need for a GeoTIFF-reading library like
    rasterio/GDAL, which would be a much heavier dependency than anything
    else in this project. Pixels outside the swath (dataMask == 0) are
    fully transparent, correctly left "unmapped" rather than faked.
 
    Returns PNG bytes, or None on failure.
    """
    try:
        # Plain extent matching MODIS's actual FINAL display area
        # (±150km around Herschel Island) — NOT the oversized BBOX_3413
        # used for MODIS's own fetch-then-rotate workflow. Sentinel Hub
        # reprojects server-side and returns an already north-up image at
        # exactly the requested bbox — but ONLY if the requested CRS is
        # actually north-up at this longitude. EPSG:3413 (polar
        # stereographic) is NOT north-up here (only at its own central
        # meridian, -45°), which was the real cause of the rotation bug.
        # UTM zone 7N (EPSG:32607) genuinely is north-up at Herschel
        # Island's longitude, so it's used here instead.
        plain_half_width_m = 150_000
        minx = _HERSCHEL_UTM_X - plain_half_width_m
        maxx = _HERSCHEL_UTM_X + plain_half_width_m
        miny = _HERSCHEL_UTM_Y - plain_half_width_m
        maxy = _HERSCHEL_UTM_Y + plain_half_width_m
 
        # dB range -25 to 0 mapped to 0-255 grayscale, a typical display
        # stretch for VV gamma0 (matches the documented layer's general
        # appearance). Pixels outside the swath get alpha=0 (transparent).
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["VV", "dataMask"],
            output: { bands: 2, sampleType: "UINT8" }
          };
        }
        function evaluatePixel(samples) {
          if (samples.dataMask == 0) {
            return [0, 0];
          }
          var db = 10 * Math.log(samples.VV) / Math.LN10;
          var clipped = Math.max(-25, Math.min(0, db));
          var gray = Math.round((clipped + 25) / 25 * 255);
          return [gray, 255];
        }
        """
 
        request_body = {
            "input": {
                "bounds": {
                    "bbox": [minx, miny, maxx, maxy],
                    "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{SENTINEL1_UTM_EPSG}"},
                },
                "data": [
                    {
                        "type": "sentinel-1-grd",
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{date_str}T00:00:00Z",
                                "to": f"{date_str}T23:59:59Z",
                            },
                            "acquisitionMode": "IW",
                            "polarization": "DV",
                        },
                        "processing": {
                            "backCoeff": "GAMMA0_ELLIPSOID",
                            "orthorectify": "true",
                        },
                    }
                ],
            },
            "output": {
                "width": MODIS_FINAL_SIZE_PX,
                "height": MODIS_FINAL_SIZE_PX,
                "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
            },
            "evalscript": evalscript,
        }
 
        resp = requests.post(
            "https://sh.dataspace.copernicus.eu/api/v1/process",
            json=request_body,
            headers={"Authorization": f"Bearer {token}", "Accept": "image/png"},
            timeout=60,
        )
        resp.raise_for_status()
 
        if resp.content[:8] != b"\x89PNG\r\n\x1a\n":
            print("SENTINEL-1: response was not a valid PNG")
            return None
 
        return resp.content
 
    except Exception as e:
        print("SENTINEL-1 IMAGE FETCH FAILED:", e)
        return None
 
 
sentinel1_bytes = None
sentinel1_caption = "Sentinel-1 SAR image unavailable — credentials missing or fetch failed. Check Action logs."
 
# Run the three independent, slow, network-bound top-level fetches
# (MODIS, total water level, Sentinel-1) concurrently rather than one
# after another — none of them depend on each other's output, so this
# is a straightforward, safe speed win for the slowest part of the
# script after the historical-data fetches. Placed here (not earlier,
# nearer MODIS's own code) because fetch_and_process_sentinel1's inner
# calls (get_sentinel_hub_token, find_latest_sentinel1_date,
# fetch_sentinel1_image) and fetch_copernicus_water_level all need to
# already be defined by the time this block actually calls them.
from concurrent.futures import ThreadPoolExecutor as _TopLevelExecutor
 
print("STARTING: parallel fetch of MODIS, water level, Sentinel-1, and Napoiak Channel water level")
with _TopLevelExecutor(max_workers=4) as _top_level_executor:
    _modis_future = _top_level_executor.submit(fetch_and_process_modis)
    _water_level_future = _top_level_executor.submit(fetch_copernicus_water_level)
    _sentinel1_future = _top_level_executor.submit(fetch_and_process_sentinel1)
    _napoiak_future = _top_level_executor.submit(fetch_napoiak_water_level)
 
    try:
        modis_bytes, modis_date = _modis_future.result()
    except Exception as e:
        print("MODIS PARALLEL FETCH FAILED:", e)
        modis_bytes, modis_date = None, None
 
    try:
        copernicus_times, copernicus_values, copernicus_yearly_mean = _water_level_future.result()
    except Exception as e:
        print("WATER LEVEL PARALLEL FETCH FAILED:", e)
        copernicus_times, copernicus_values, copernicus_yearly_mean = None, None, None
 
    try:
        sentinel1_bytes, sentinel1_caption = _sentinel1_future.result()
    except Exception as e:
        print("SENTINEL-1 PARALLEL FETCH FAILED:", e)
        sentinel1_bytes = None
        sentinel1_caption = "Sentinel-1 SAR image unavailable — fetch failed. Check Action logs."

    try:
        napoiak_times, napoiak_values = _napoiak_future.result()
    except Exception as e:
        print("NAPOIAK PARALLEL FETCH FAILED:", e)
        napoiak_times, napoiak_values = None, None
 
if copernicus_times and copernicus_values:
    current_level_total = copernicus_values[0]
    max_level_total = max(copernicus_values)
    min_level_total = min(copernicus_values)
    water_level_text = [
        ("Total water level (now): ", f"{current_level_total:.2f} m"),
        ["Next 24h range: ", ("", f"{min_level_total:.2f} m"), " to ", ("", f"{max_level_total:.2f} m")],
        "Includes tide + storm surge (not just astronomical tide).",
    ]
else:
    water_level_text = (
        "Total water level data unavailable — the THREDDS server may be temporarily "
        "unreachable or slow. Check Action logs. (This is separate from the Tides block "
        "above, which uses DFO's astronomical tide predictions.)"
    )
 
water_level_chart_bytes, water_level_chart_caption = build_water_level_chart(copernicus_times, copernicus_values, copernicus_yearly_mean)
napoiak_chart_bytes, napoiak_chart_caption = build_napoiak_chart(napoiak_times, napoiak_values)
 
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
 
if water_level_chart_bytes:
    try:
        uid = upload_image_to_notion(water_level_chart_bytes, "water_level_chart.png")
        water_level_chart_block = image_block_from_upload(uid)
    except Exception as e:
        print("WATER LEVEL CHART NOTION UPLOAD FAILED:", e)
        water_level_chart_caption = "Water level chart generated but upload to Notion failed — see Action logs."
 
# Built as a standalone list rather than appended directly to `blocks`,
# since this section needs to appear visually ABOVE "Today's Conditions"
# (per explicit request — full width, above the Tide card, with the full
# 10-day forecast), even though the underlying data isn't ready until
# this much later point in the script's actual execution order. Spliced
# into `blocks` at the right position once both this list and the
# Today's-Conditions insertion point both exist — see below.
_total_water_level_blocks = [
    heading("🌊 Total Water Level — 10-Day Forecast (tide + storm surge)"),
    callout(water_level_text, emoji="🌊", color="purple_background"),
]
if water_level_chart_block:
    _total_water_level_blocks.append(water_level_chart_block)
_total_water_level_blocks.append(
    paragraph(water_level_chart_caption if water_level_chart_bytes else "Water level chart could not be generated — see Action logs.")
)
_total_water_level_blocks.append(divider())

napoiak_chart_block = None
if napoiak_chart_bytes:
    try:
        uid = upload_image_to_notion(napoiak_chart_bytes, "napoiak_chart.png")
        napoiak_chart_block = image_block_from_upload(uid)
    except Exception as e:
        print("NAPOIAK CHART NOTION UPLOAD FAILED:", e)
        napoiak_chart_caption = "Napoiak Channel water level chart generated but upload to Notion failed — see Action logs."

_napoiak_blocks = [
    heading("💧 Napoiak Channel Water Level — Mackenzie River above Shallow Bay"),
]
if napoiak_chart_block:
    _napoiak_blocks.append(napoiak_chart_block)
_napoiak_blocks.append(
    paragraph(napoiak_chart_caption if napoiak_chart_bytes else "Napoiak Channel water level chart could not be generated — see Action logs.")
)
_napoiak_blocks.append(divider())

_total_water_level_blocks.extend(_napoiak_blocks)
 
blocks.append(heading("🛰 Satellite View of Shingle Point"))
if modis_block:
    blocks.append(modis_block)
blocks.append(paragraph(f"A real satellite photo of Shingle Point, taken on {modis_date if modis_date else 'a recent date'}."))
 
# Link to explore the same date/location/layers interactively in NASA's
# own Worldview tool, using its documented permalink parameters:
# p=projection, v=viewport extent (minX,minY,maxX,maxY), l=layer list, t=date.
worldview_date = modis_date if modis_date else now.strftime("%Y-%m-%d")
worldview_url = (
    f"https://worldview.earthdata.nasa.gov/?p=arctic"
    f"&l=MODIS_Terra_CorrectedReflectance_TrueColor,Coastlines"
    f"&t={worldview_date}"
    f"&v={BBOX_3413}"
)
blocks.append(gray_caption(modis_caption))
blocks.append(link_paragraph("Explore here →", worldview_url))
blocks.append(divider())
 
blocks.append(heading("🛰 Radar View of Shingle Point"))
sentinel1_block = None
if sentinel1_bytes:
    try:
        uid = upload_image_to_notion(sentinel1_bytes, "sentinel1.png")
        sentinel1_block = image_block_from_upload(uid)
    except Exception as e:
        print("SENTINEL-1 NOTION UPLOAD FAILED:", e)
        sentinel1_caption = "Sentinel-1 image generated but upload to Notion failed — see Action logs."
if sentinel1_block:
    blocks.append(sentinel1_block)
blocks.append(paragraph(
    "A radar image of Shingle Point, which can see through cloud and darkness — useful when "
    "the regular satellite photo above is blocked by weather. Gray areas were outside the "
    "satellite's path that day."
))
blocks.append(gray_caption(sentinel1_caption))
blocks.append(link_paragraph("Explore here →", sentinel1_url))
blocks.append(divider())
 
# --- Weather Forecast and Marine Forecast relocated to appear right
# after "Today's Conditions" and before the satellite imagery — see
# above, inserted right after the Today's Conditions card grid.
 
# --- Total Water Level moved to appear after Marine Forecast and before
# the satellite imagery — see _total_water_level_blocks built later (once
# the data is ready) and spliced into `blocks` at _water_level_insertion_index. ---
 
# =========================================================
# HISTORICAL / TREND SECTIONS (past data, distinct from the forecast
# sections above) — grouped together so forward-looking content (5-day
# weather, marine forecast, total water level) is never interleaved
# with backward-looking trends (30-day temperature, annual thawing
# degree days, 30-day wind history).
# =========================================================
 
# --- Temperature chart (full width, needs room for the image) ---
blocks.append(heading("📈 Temperature — last 30 days vs. 30-year average"))
if temp_chart_block:
    blocks.append(temp_chart_block)
blocks.append(paragraph(temp_chart_caption if temp_chart_bytes else "Chart could not be generated — see Action logs."))
 
blocks.append(divider())
 
# --- Thawing degree days histogram (full width, needs room for the image) ---
blocks.append(heading("🌡 Thawing Degree Days — annual totals"))
if tdd_histogram_block:
    blocks.append(tdd_histogram_block)
blocks.append(paragraph(tdd_histogram_caption if tdd_histogram_bytes else "Thawing degree days chart could not be generated — see Action logs."))
 
blocks.append(divider())
 
# --- Wind vector chart (full width, needs room for the image) ---
blocks.append(heading("🧭 Wind — last 30 days"))
if wind_chart_block:
    blocks.append(wind_chart_block)
blocks.append(paragraph(wind_chart_caption if wind_chart_bytes else "Wind vector chart could not be generated — see Action logs."))
 
blocks.append(divider())
 
blocks.append(disclaimer_paragraph(
    "Disclaimer: All data and imagery on this page are collated from external third-party sources "
    "(including NASA GIBS/EOSDIS, Open-Meteo, sunrise-sunset.org, Environment Canada, DFO/CHS, "
    "the Norwegian Meteorological Institute, Copernicus Sentinel Hub/Data Space Ecosystem, and "
    "Natural Earth) and are displayed here for general informational purposes only. We hold no "
    "responsibility for the accuracy, completeness, or timeliness of this data, and this page is "
    "not a substitute for official sources. Do not use this information for navigation, "
    "safety-critical decisions, or any other purpose where inaccurate or delayed data could cause harm."
))
 
# =========================================================
# CLEAR PAGE
# =========================================================
existing = notion.blocks.children.list(block_id=PAGE_ID)
print("EXISTING BLOCK COUNT:", len(existing["results"]))
 
for b in existing["results"]:
    notion.blocks.delete(block_id=b["id"])
 
# Splice the Total Water Level section in at the position recorded
# earlier (right before "Today's Conditions"), now that both the section
# itself and the insertion index are available. List slicing here is
# safe and simple — `blocks` is a plain Python list, and inserting a
# sub-list at a saved index doesn't care that the section's underlying
# data was only fetched much later in the script's actual execution
# order than where it now visually appears.
blocks = blocks[:_water_level_insertion_index] + _total_water_level_blocks + blocks[_water_level_insertion_index:]
 
# =========================================================
# UPDATE PAGE
# =========================================================
response = notion.blocks.children.append(block_id=PAGE_ID, children=blocks)
print("APPEND RESPONSE BLOCK COUNT:", len(response.get("results", [])))
print("Dashboard updated successfully")
 
# Persist the historical temperature cache to disk if anything new was
# added this run, so future runs can skip re-fetching years already
# validated as complete — this call was previously missing, meaning the
# cache loaded correctly but was never actually saved back, silently
# providing no benefit across runs.
if _temp_cache_dirty:
    save_temp_cache(_temp_cache)
    print(f"CACHE: saved {len(_temp_cache)} years to {CACHE_FILE_PATH}")
else:
    print("CACHE: no new years added this run, skipping save")
 
# =========================================================
# NOTE ON DEPENDENCIES
# =========================================================
# requirements.txt should now include, in addition to what you already have:
#   xarray
#   netCDF4
#   notion-client
#   requests
