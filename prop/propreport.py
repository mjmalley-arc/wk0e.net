#!/usr/bin/env python3
"""
QRP CW POTA propagation report.

Default home: EN11xe (central Nebraska, 41.19N -96.04W).
Override with --grid <maidenhead>.

Usage:
    python3 propreport.py                          # writes report.html + report.txt
    python3 propreport.py --grid EM48              # different grid
    python3 propreport.py --html out.html --txt out.txt
    python3 propreport.py --json raw.json          # also dump raw fetched JSON

Outputs:
    report.html  -- visual dashboard (open in browser)
    report.txt   -- field-ready monospace summary

Data sources (all free, no auth):
    OpenHamClock public API   /api/propagation, /api/solar-indices, /api/n0nbh
    NOAA SWPC                 alerts, scales, 3-day forecast
    POTA                      api.pota.app/spot/activator
    Sunrise-Sunset.org        local sun events for greyline

Author: built collaboratively with Claude. License: do whatever you want.
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

USER_AGENT = "QRP-POTA-Propreport/1.0 (https://github.com/yourcall; contact@example.com)"
HOME_GRID_DEFAULT = "EN11xe"
BANDS = ["40m", "30m", "20m", "17m", "15m", "12m", "10m"]
BAND_FREQS = {"40m": 7.0, "30m": 10.1, "20m": 14.0, "17m": 18.1,
              "15m": 21.0, "12m": 24.9, "10m": 28.0}

# NA centroids we average propagation predictions across.
NA_CENTROIDS = [
    ("Northeast",      42.36, -71.06),   # Boston
    ("Southeast",      33.75, -84.39),   # Atlanta
    ("Mid-Atlantic",   38.90, -77.04),   # DC
    ("South-Central",  30.27, -97.74),   # Austin
    ("Mountain",       39.74, -104.99),  # Denver
    ("West Coast",     37.77, -122.42),  # SF
    ("Pacific NW",     47.61, -122.33),  # Seattle
]

# POTA band frequency edges (kHz)
POTA_BAND_KHZ = {
    "40m": (7000, 7300),  "30m": (10100, 10150), "20m": (14000, 14350),
    "17m": (18068, 18168), "15m": (21000, 21450), "12m": (24890, 24990),
    "10m": (28000, 29700),
}
NA_PREFIXES = ("US-", "CA-", "MX-")


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def grid_to_latlon(grid: str) -> tuple[float, float]:
    """Maidenhead grid (4 or 6 char) -> (lat, lon) center of square."""
    g = grid.upper().strip()
    if len(g) < 4:
        raise ValueError(f"Grid must be 4 or 6 characters: {grid}")
    lon = (ord(g[0]) - ord("A")) * 20 - 180
    lat = (ord(g[1]) - ord("A")) * 10 - 90
    lon += int(g[2]) * 2
    lat += int(g[3]) * 1
    if len(g) >= 6:
        lon += (ord(g[4]) - ord("A")) * (2.0 / 24)
        lat += (ord(g[5]) - ord("A")) * (1.0 / 24)
        lon += (2.0 / 24) / 2
        lat += (1.0 / 24) / 2
    else:
        lon += 1.0
        lat += 0.5
    return (lat, lon)


def http_get(url: str, *, timeout: int = 20, kind: str = "json") -> Any:
    """GET a URL, return parsed JSON or text. Raises on error."""
    r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    if kind == "json":
        return r.json()
    return r.text


def fmt_local(t_utc: dt.datetime, tz_offset_hours: float) -> str:
    return (t_utc + dt.timedelta(hours=tz_offset_hours)).strftime("%H:%M")


def round_half(x: float | None) -> int | None:
    return None if x is None else int(round(x))


# -------------------------------------------------------------------------
# Data fetchers
# -------------------------------------------------------------------------

def fetch_propagation_path(de: tuple[float, float], dx: tuple[str, float, float],
                           mode: str = "CW", power: int = 5) -> dict:
    """Hit OpenHamClock /api/propagation for one path."""
    url = (f"https://openhamclock.com/api/propagation"
           f"?deLat={de[0]}&deLon={de[1]}&dxLat={dx[1]}&dxLon={dx[2]}"
           f"&mode={mode}&power={power}&antenna=isotropic")
    return {"name": dx[0], "lat": dx[1], "lon": dx[2], "data": http_get(url)}


def fetch_all(de: tuple[float, float]) -> dict:
    """Pull every data source we need. Uses thread pool for parallelism."""
    out: dict[str, Any] = {}
    fetched = dt.datetime.now(dt.timezone.utc)
    out["meta"] = {"fetched_utc": fetched.isoformat(), "home_lat": de[0], "home_lon": de[1]}

    # Fan out the static endpoints + per-path predictions in parallel.
    jobs: list[tuple[str, callable, tuple]] = [
        ("solar",  http_get, ("https://openhamclock.com/api/solar-indices",)),
        ("n0nbh",  http_get, ("https://openhamclock.com/api/n0nbh",)),
        ("pota",   http_get, ("https://api.pota.app/spot/activator",)),
        ("scales", http_get, ("https://services.swpc.noaa.gov/products/noaa-scales.json",)),
        ("alerts", http_get, ("https://services.swpc.noaa.gov/products/alerts.json",)),
        ("forecast3day", http_get, ("https://services.swpc.noaa.gov/text/3-day-forecast.txt",), {"kind": "text"}),
        ("sun", http_get,
         (f"https://api.sunrise-sunset.org/json?lat={de[0]}&lng={de[1]}&formatted=0",)),
    ]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {}
        for entry in jobs:
            name, fn, args = entry[0], entry[1], entry[2]
            kw = entry[3] if len(entry) > 3 else {}
            futs[ex.submit(fn, *args, **kw)] = name
        # Per-path predictions
        for dx in NA_CENTROIDS:
            futs[ex.submit(fetch_propagation_path, de, dx)] = ("path", dx[0])
        for fut in as_completed(futs):
            label = futs[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {"error": f"{type(exc).__name__}: {exc}"}
            if isinstance(label, tuple) and label[0] == "path":
                out.setdefault("paths", []).append(res)
            else:
                out[label] = res
    return out


# -------------------------------------------------------------------------
# Aggregation
# -------------------------------------------------------------------------

def aggregate(raw: dict) -> dict:
    """Reduce raw fetched data into the structured payload the renderers expect."""
    paths = [p for p in raw.get("paths", []) if isinstance(p.get("data"), dict)]

    # Average reliability per band per hour across paths
    avg_grid: dict[str, dict[int, int | None]] = {}
    for b in BANDS:
        avg_grid[b] = {}
        for h in range(24):
            vals = []
            for p in paths:
                hp = (p["data"].get("hourlyPredictions") or {}).get(b)
                if not hp:
                    continue
                cell = next((c for c in hp if c.get("hour") == h), None)
                if cell and cell.get("reliability") is not None:
                    vals.append(cell["reliability"])
            avg_grid[b][h] = round(sum(vals) / len(vals)) if vals else None

    # Current band averages
    current_by_band: dict[str, int | None] = {}
    for b in BANDS:
        vals = []
        for p in paths:
            cb = next((x for x in (p["data"].get("currentBands") or []) if x.get("band") == b), None)
            if cb and cb.get("reliability") is not None:
                vals.append(cb["reliability"])
        current_by_band[b] = round(sum(vals) / len(vals)) if vals else None

    # Per-path summaries
    path_summaries = [
        {"name": p["name"], "distance": p["data"].get("distance"),
         "muf": p["data"].get("muf"), "luf": p["data"].get("luf")}
        for p in paths
    ]

    current_hour_utc = paths[0]["data"].get("currentHour") if paths else dt.datetime.utcnow().hour

    # Solar context
    solar = raw.get("solar") or {}
    cur_solar = {
        "sfi": (solar.get("sfi") or {}).get("current"),
        "ssn": (solar.get("ssn") or {}).get("current"),
        "kp":  (solar.get("kp") or {}).get("current"),
        "bz":  (solar.get("bz") or {}).get("current"),
    }
    kp_forecast = (solar.get("kp") or {}).get("forecast") or []

    # N0NBH bits
    n = raw.get("n0nbh") or {}
    nsd = n.get("solarData") or {}
    n0nbh_summary = {
        "xray": nsd.get("xray"),
        "aIndex": nsd.get("aIndex"),
        "aurora": nsd.get("aurora"),
        "solarWind": nsd.get("solarWind"),
        "magneticField": nsd.get("magneticField"),
        "geomag": n.get("geomagField"),
        "signalNoise": n.get("signalNoise"),
        "updated": n.get("updated"),
    }

    # POTA spot density
    pota = raw.get("pota") or []
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=30))
    pota_counts = {b: 0 for b in BANDS}
    pota_modes = {b: {"CW": 0, "SSB": 0, "FT8": 0, "FT4": 0, "Other": 0} for b in BANDS}
    en11_spots = []
    home_grid_prefix = "EN11"  # caller may swap; for now driven by HOME_GRID
    for s in pota:
        try:
            t = dt.datetime.fromisoformat(s["spotTime"]).replace(tzinfo=dt.timezone.utc)
        except Exception:
            continue
        if t < cutoff:
            continue
        loc = s.get("locationDesc") or ""
        is_na = any(loc.startswith(p) for p in NA_PREFIXES)
        if not is_na:
            continue
        try:
            f = float(s.get("frequency") or 0)
        except Exception:
            continue
        if f <= 0:
            continue
        for band, (lo, hi) in POTA_BAND_KHZ.items():
            if lo <= f <= hi:
                pota_counts[band] += 1
                m = (s.get("mode") or "").upper()
                if m == "CW":
                    pota_modes[band]["CW"] += 1
                elif m in ("SSB", "USB", "LSB"):
                    pota_modes[band]["SSB"] += 1
                elif m == "FT8":
                    pota_modes[band]["FT8"] += 1
                elif m == "FT4":
                    pota_modes[band]["FT4"] += 1
                else:
                    pota_modes[band]["Other"] += 1
                break
        # Track spots from same grid (home grid 4-char prefix)
        if (s.get("grid4") or "") == home_grid_prefix:
            en11_spots.append({
                "activator": s.get("activator"), "frequency": s.get("frequency"),
                "mode": s.get("mode"), "ref": s.get("reference"),
                "name": s.get("name"), "comments": s.get("comments"),
                "spotTime": s.get("spotTime"), "grid": s.get("grid6"),
            })

    # NOAA scales
    scales = raw.get("scales") or {}

    # Find any X-class flare alert in last 72 hours
    alerts = raw.get("alerts") or []
    flare_alert = None
    cutoff72 = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=72)
    for a in alerts:
        msg = a.get("message") or ""
        if extract_flare_class(msg) is None:
            continue
        try:
            ts = dt.datetime.fromisoformat(a["issue_datetime"]).replace(tzinfo=dt.timezone.utc)
        except Exception:
            continue
        if ts >= cutoff72:
            flare_alert = {"date": a["issue_datetime"], "msg": msg[:400]}
            break

    # Sunrise/sunset
    sun = (raw.get("sun") or {}).get("results") or {}

    # 3-day Kp forecast text excerpt
    forecast_text = raw.get("forecast3day") or ""

    return {
        "fetched": raw.get("meta", {}).get("fetched_utc"),
        "currentHourUTC": current_hour_utc,
        "current": cur_solar,
        "n0nbh": n0nbh_summary,
        "paths": path_summaries,
        "currentByBand": current_by_band,
        "avgGrid": avg_grid,
        "kpForecast": kp_forecast,
        "potaCounts": pota_counts,
        "potaModes": pota_modes,
        "en11Spots": en11_spots,
        "scales": {
            "today":      scales.get("0"),
            "tomorrow":   scales.get("1"),
            "dayAfter":   scales.get("2"),
            "yesterday":  scales.get("-1"),
        },
        "flareAlert": flare_alert,
        "sun": sun,
        "forecast3day": forecast_text,
    }


# -------------------------------------------------------------------------
# Decision logic: 6-hour go/no-go grid in 2-hour blocks
# -------------------------------------------------------------------------

def reliability_to_status(r: int | None) -> str:
    """Map reliability % to a 3-level go/no-go category."""
    if r is None:
        return "unk"
    if r >= 55:
        return "good"
    if r >= 25:
        return "fair"
    return "poor"


def block_avg(avg_grid: dict, band: str, h_start_utc: int) -> int | None:
    """Average reliability across a 2-hour block starting at h_start_utc (mod 24)."""
    vals = []
    for off in range(2):
        h = (h_start_utc + off) % 24
        v = avg_grid[band].get(h)
        if v is not None:
            vals.append(v)
    return round(sum(vals) / len(vals)) if vals else None


def six_hour_grid(avg_grid: dict, current_hour_utc: int) -> list[tuple[int, int]]:
    """Return [(start_utc, end_utc), ...] — three 2h blocks starting at current hour."""
    blocks = []
    h = current_hour_utc
    for _ in range(3):
        blocks.append((h % 24, (h + 2) % 24))
        h += 2
    return blocks


def rest_of_day_blocks(avg_grid: dict, start_utc: int) -> list[tuple[str, list[int]]]:
    """Three coarse blocks (morning / afternoon / evening) covering the next ~12-18h after the 6h grid."""
    # Hours after the 6h block
    base = (start_utc + 6) % 24
    blocks = [
        ("Next 6h", [(base + i) % 24 for i in range(6)]),
        ("Then 6h", [(base + 6 + i) % 24 for i in range(6)]),
        ("Overnight",  [(base + 12 + i) % 24 for i in range(6)]),
    ]
    return blocks


# -------------------------------------------------------------------------
# Renderers
# -------------------------------------------------------------------------

def utc_to_local_offset_hours(home_lon: float) -> float:
    """Rough local-time offset from longitude. Good enough for a header."""
    return round(home_lon / 15)


def parse_iso_utc(s: str) -> dt.datetime | None:
    """Parse ISO 8601 string into a tz-aware UTC datetime. Naive strings are assumed UTC."""
    if not s:
        return None
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d


def extract_flare_class(msg: str) -> str | None:
    """Pull the actual flare magnitude (e.g. 'X2.5') from a SWPC alert message.

    SWPC messages contain message codes like 'SUMX01' which look like X-class flares
    to a naive regex, so we look for explicit context: 'X-ray Class:' or 'exceeded X<n>'.
    """
    if not msg:
        return None
    # Preferred: 'X-ray Class: X2.5'
    m = re.search(r"X[- ]?ray Class:\s*X([0-9]+(?:\.[0-9]+)?)", msg, re.IGNORECASE)
    if m:
        return f"X{m.group(1)}"
    # Fallback: 'exceeded X1' / 'exceeded X3.4'
    m = re.search(r"exceeded\s+X([0-9]+(?:\.[0-9]+)?)", msg, re.IGNORECASE)
    if m:
        return f"X{m.group(1)}"
    return None


def render_text(p: dict, grid: str) -> str:
    """Field-ready monospace summary."""
    cur = p["current"]
    n = p["n0nbh"]
    cbb = p["currentByBand"]
    home_lon = -96.04
    # Home longitude: best-effort from grid square
    try:
        _, home_lon = grid_to_latlon(grid)
    except Exception:
        pass
    tz_off = utc_to_local_offset_hours(home_lon)
    now_utc = dt.datetime.now(dt.timezone.utc)
    now_local = now_utc + dt.timedelta(hours=tz_off)

    sun = p.get("sun") or {}
    sunrise = parse_iso_utc(sun.get("sunrise"))
    sunset  = parse_iso_utc(sun.get("sunset"))
    sunset_local = (sunset + dt.timedelta(hours=tz_off)).strftime("%H:%M") if sunset else "?"

    # Best band right now (highest reliability)
    ranked = sorted(
        [(b, v) for b, v in cbb.items() if v is not None],
        key=lambda x: -x[1]
    )
    best = ranked[0][0] if ranked else "?"
    second = ranked[1][0] if len(ranked) > 1 else None
    third = ranked[2][0] if len(ranked) > 2 else None

    # Storm watch from Kp forecast
    kp_warn = ""
    for f in p.get("kpForecast") or []:
        if f.get("value", 0) >= 5:
            t = parse_iso_utc(f.get("time"))
            if t and t > now_utc:
                hours_until = (t - now_utc).total_seconds() / 3600
                if hours_until < 24:
                    local_t = (t + dt.timedelta(hours=tz_off)).strftime("%H:%M")
                    kp_warn = f"G1 storm watch starting ~{local_t} L (Kp {f['value']:.1f})"
                    break

    # Active flare watchout
    flare = ""
    if p.get("flareAlert"):
        cls = extract_flare_class(p["flareAlert"]["msg"])
        if cls:
            flare = f"Recent {cls} flare in last 72h — region may still be active."

    pota_top = ", ".join([f"{b} ({c})" for b, c in
                          sorted(p["potaCounts"].items(), key=lambda x: -x[1])
                          if c > 0][:4]) or "no recent NA spots"

    lines = []
    lines.append(f"{grid} · {now_local.strftime('%d %b')} · {now_local.strftime('%H:%M')} L  ({now_utc.strftime('%H:%MZ')})")
    lines.append(f"SFI {cur['sfi']} · K {cur['kp']} · X-ray {n['xray']} · {n['geomag']}")
    lines.append("")
    lines.append(f"NOW: {best.upper()} is the play.")
    if second or third:
        alts = " / ".join([x.upper() for x in [second, third] if x])
        lines.append(f"  primary: {best.upper()} ({cbb[best]}% reliability NA-avg)")
        if second:
            lines.append(f"  alts:    {alts}")
    lines.append(f"  POTA NA last 30m: {pota_top}")
    lines.append("")
    lines.append(f"GREYLINE: sunset {sunset_local} L · window roughly 30 min around")
    if kp_warn:
        lines.append("")
        lines.append(f"WATCHOUT: {kp_warn}")
    if flare:
        lines.append(f"          {flare}")
    lines.append("")
    lines.append("BANDS NOW (NA-avg reliability %):")
    for b in BANDS:
        v = cbb.get(b)
        bar = "█" * (v // 10) if v is not None else ""
        lines.append(f"  {b:4s}  {v if v is not None else '--':>3}%  {bar}")
    lines.append("")
    lines.append(f"3-day Kp peaks: see report.html for detail")
    lines.append(f"data: openhamclock.com + NOAA SWPC + POTA")
    return "\n".join(lines)


def render_html(p: dict, grid: str) -> str:
    """Visual dashboard, single self-contained HTML file."""
    cur = p["current"]
    n = p["n0nbh"]
    cbb = p["currentByBand"]
    avg = p["avgGrid"]
    cur_hour = p["currentHourUTC"]

    try:
        home_lat, home_lon = grid_to_latlon(grid)
    except Exception:
        home_lat, home_lon = 41.19, -96.04
    tz_off = utc_to_local_offset_hours(home_lon)
    now_utc = dt.datetime.now(dt.timezone.utc)
    now_local = now_utc + dt.timedelta(hours=tz_off)

    sun = p.get("sun") or {}
    sunrise = parse_iso_utc(sun.get("sunrise"))
    sunset  = parse_iso_utc(sun.get("sunset"))
    sunset_local = (sunset + dt.timedelta(hours=tz_off)).strftime("%H:%M") if sunset else "?"

    # Build 6-hour grid (3 columns x 2-hour blocks, starting at current UTC hour)
    blocks_6h = six_hour_grid(avg, cur_hour)
    grid_rows = []
    for b in BANDS:
        cells = []
        for (start, _end) in blocks_6h:
            v = block_avg(avg, b, start)
            status = reliability_to_status(v)
            cls = {"good": "good", "fair": "fair", "poor": "poor", "unk": "unk"}[status]
            label = {"good": "good", "fair": "fair", "poor": "poor", "unk": "—"}[status]
            disp = f"{v}%" if v is not None else "—"
            cells.append((cls, label, disp))
        grid_rows.append((b, cells))

    # Column headers in local time
    col_headers = []
    for (start, end) in blocks_6h:
        utc_label_h = (now_utc + dt.timedelta(hours=(start - now_utc.hour) % 24)).hour
        local_h = (utc_label_h + tz_off) % 24
        local_h2 = (local_h + 2) % 24
        col_headers.append(f"{int(local_h):02d}–{int(local_h2):02d}")

    # Rest of day strip
    rod_blocks = rest_of_day_blocks(avg, cur_hour)
    rod_data = []
    for label, hours in rod_blocks:
        # For each band compute avg over those hours, then take top non-poor band names
        band_avgs = []
        for b in BANDS:
            vals = [avg[b][h] for h in hours if avg[b].get(h) is not None]
            if vals:
                band_avgs.append((b, sum(vals) / len(vals)))
        band_avgs.sort(key=lambda x: -x[1])
        good = [b for b, v in band_avgs if v >= 55]
        fair = [b for b, v in band_avgs if 25 <= v < 55]
        msg_parts = []
        if good:
            msg_parts.append("good: " + ", ".join(good))
        if fair:
            msg_parts.append("fair: " + ", ".join(fair[:3]))
        if not msg_parts:
            msg_parts.append("nothing reliable")
        rod_data.append((label, " · ".join(msg_parts)))

    # POTA bars
    pota_max = max(p["potaCounts"].values()) or 1
    pota_rows = []
    for b in sorted(BANDS, key=lambda x: -p["potaCounts"][x]):
        c = p["potaCounts"][b]
        modes = p["potaModes"][b]
        mode_summary = []
        if modes["CW"]:  mode_summary.append(f"{modes['CW']} CW")
        if modes["SSB"]: mode_summary.append(f"{modes['SSB']} SSB")
        if modes["FT8"]: mode_summary.append(f"{modes['FT8']} FT8")
        if modes["FT4"]: mode_summary.append(f"{modes['FT4']} FT4")
        mode_str = " · ".join(mode_summary) if mode_summary else "—"
        width_pct = round(100 * c / pota_max) if c > 0 else 0
        pota_rows.append((b, c, mode_str, width_pct))

    # Alert banner: storm + flare logic
    alert_html = ""
    storm_warn_text = ""
    for f in p.get("kpForecast") or []:
        if f.get("value", 0) >= 5:
            t = parse_iso_utc(f.get("time"))
            if t and t > now_utc and (t - now_utc).total_seconds() < 24 * 3600:
                local_t = (t + dt.timedelta(hours=tz_off)).strftime("%H:%M")
                storm_warn_text = f"G1 minor geomagnetic storm watch starting ~{local_t} local (Kp {f['value']:.1f})"
                break
    flare_warn_text = ""
    if p.get("flareAlert"):
        cls = extract_flare_class(p["flareAlert"]["msg"])
        if cls:
            flare_warn_text = f"{cls} flare within last 72h — sudden HF fades possible if region fires again."
    alert_bits = [x for x in [storm_warn_text, flare_warn_text] if x]
    if alert_bits:
        alert_html = (
            f'<div class="alert">'
            + " ".join([f'<strong>{a}</strong>' for a in alert_bits[:1]])
            + (f" {alert_bits[1]}" if len(alert_bits) > 1 else "")
            + '</div>'
        )

    # 3-day forecast strip — get peak Kp per day from forecast
    kp_by_day = {}
    for fc in p.get("kpForecast") or []:
        t = parse_iso_utc(fc.get("time"))
        if not t:
            continue
        day = t.strftime("%a")
        kp_by_day.setdefault(day, []).append(fc.get("value", 0))
    forecast_strip = []
    today_label = now_local.strftime("%a · %b %d")
    forecast_strip.append((today_label, _scales_label(p["scales"].get("today")), p["current"]["sfi"]))
    if p["scales"].get("tomorrow"):
        d = (now_local + dt.timedelta(days=1)).strftime("%a · %b %d")
        forecast_strip.append((d, _scales_label(p["scales"].get("tomorrow")), None))
    if p["scales"].get("dayAfter"):
        d = (now_local + dt.timedelta(days=2)).strftime("%a · %b %d")
        forecast_strip.append((d, _scales_label(p["scales"].get("dayAfter")), None))

    # Field-text rendered as <pre>
    field = render_text(p, grid).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Decide best band sentence (for header narrative)
    ranked = sorted([(b, v) for b, v in cbb.items() if v is not None], key=lambda x: -x[1])
    best_band = ranked[0][0] if ranked else "—"
    best_pct = ranked[0][1] if ranked else 0

    # ---------------- HTML ----------------
    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HF propagation · {grid} · {now_local.strftime("%d %b %H:%M")}</title>
<style>
  :root {{
    --bg: #fafaf7; --fg: #2a2a28; --muted: #73726c; --tert: #b4b2a9;
    --surface: #fff; --border: rgba(0,0,0,0.08);
    --good-bg: #c0dd97; --good-fg: #173404;
    --fair-bg: #fac775; --fair-fg: #412402;
    --poor-bg: #f4c0d1; --poor-fg: #4b1528;
    --warn-bg: #faeeda; --warn-fg: #633806;
    --info-bg: #e6f1fb; --info-fg: #0c447c;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #1a1a18; --fg: #e8e6dd; --muted: #a8a69d; --tert: #6a6963;
      --surface: #25241f; --border: rgba(255,255,255,0.08);
      --good-bg: #3b6d11; --good-fg: #eaf3de;
      --fair-bg: #854f0b; --fair-fg: #faeeda;
      --poor-bg: #791f1f; --poor-fg: #fcebeb;
      --warn-bg: #633806; --warn-fg: #faeeda;
      --info-bg: #0c447c; --info-fg: #e6f1fb;
    }}
  }}
  html, body {{ background: var(--bg); color: var(--fg); margin: 0; padding: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; line-height: 1.5; }}
  .wrap {{ max-width: 920px; margin: 0 auto; padding: 24px 20px 40px; }}
  h1 {{ font-size: 22px; font-weight: 500; margin: 0 0 4px; }}
  h2 {{ font-size: 13px; font-weight: 500; color: var(--muted); margin: 28px 0 10px;
    text-transform: uppercase; letter-spacing: 0.5px; }}
  .subtitle {{ font-size: 13px; color: var(--tert); font-family: ui-monospace, monospace;
    margin-bottom: 4px; }}
  .alert {{ background: var(--warn-bg); color: var(--warn-fg); padding: 10px 14px;
    border-radius: 8px; font-size: 13px; margin: 16px 0 0; border-left: 3px solid var(--warn-fg); }}
  .alert strong {{ font-weight: 500; }}
  table.grid {{ width: 100%; border-collapse: separate; border-spacing: 0;
    font-size: 13px; table-layout: fixed; }}
  table.grid th {{ text-align: center; font-weight: 500; padding: 6px 4px;
    color: var(--muted); font-size: 11px; border-bottom: 0.5px solid var(--border); }}
  table.grid th:first-child {{ text-align: left; padding-left: 8px; width: 56px; }}
  table.grid td {{ padding: 0; height: 32px; vertical-align: middle; }}
  table.grid td:first-child {{ font-weight: 500; padding-left: 8px;
    font-family: ui-monospace, monospace; font-size: 13px; }}
  .cell {{ display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 28px; margin: 2px; border-radius: 4px; font-size: 11px; font-weight: 500; }}
  .cell .pct {{ font-size: 10px; font-weight: 400; opacity: 0.7; }}
  .cell-good {{ background: var(--good-bg); color: var(--good-fg); }}
  .cell-fair {{ background: var(--fair-bg); color: var(--fair-fg); }}
  .cell-poor {{ background: var(--poor-bg); color: var(--poor-fg); }}
  .cell-unk {{ background: var(--surface); color: var(--tert); border: 0.5px dashed var(--border); }}
  .legend {{ display: flex; gap: 14px; font-size: 11px; color: var(--muted);
    margin-top: 6px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; }}
  .strip {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
  .strip .cell-strip {{ background: var(--surface); padding: 10px 12px; border-radius: 8px;
    border: 0.5px solid var(--border); }}
  .strip .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.5px; }}
  .strip .val {{ font-size: 13px; font-weight: 500; margin-top: 4px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }}
  .stat {{ background: var(--surface); padding: 12px 14px; border-radius: 8px;
    border: 0.5px solid var(--border); }}
  .stat .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 4px; }}
  .stat .val {{ font-size: 22px; font-weight: 500; line-height: 1; }}
  .stat .sub {{ font-size: 11px; color: var(--tert); margin-top: 4px; }}
  .pota-row {{ display: flex; align-items: center; padding: 8px 12px;
    background: var(--surface); border-radius: 8px; margin-bottom: 6px; font-size: 13px;
    border: 0.5px solid var(--border); }}
  .pota-band {{ font-family: ui-monospace, monospace; font-weight: 500; min-width: 40px; }}
  .pota-bar {{ flex: 1; margin: 0 12px; height: 8px; background: var(--bg); border-radius: 4px;
    overflow: hidden; position: relative; border: 0.5px solid var(--border); }}
  .pota-fill {{ position: absolute; left: 0; top: 0; bottom: 0; background: #1d9e75;
    border-radius: 4px; }}
  .pota-cnt {{ font-size: 11px; color: var(--muted); font-family: ui-monospace, monospace;
    min-width: 130px; text-align: right; }}
  .field {{ background: var(--surface); border: 0.5px solid var(--border);
    border-radius: 8px; padding: 14px 16px; font-family: ui-monospace, monospace;
    font-size: 12px; line-height: 1.7; white-space: pre-wrap; }}
  .footer {{ font-size: 11px; color: var(--tert); margin-top: 32px; text-align: center; }}
  .meta {{ font-size: 11px; color: var(--tert); }}
</style>
</head>
<body>
<div class="wrap">

  <h1>{grid} · {now_local.strftime("%a %d %b %Y")}</h1>
  <div class="subtitle">data fetched {now_utc.strftime("%H:%M UTC")} · {now_local.strftime("%H:%M local")} · sunset {sunset_local} L</div>
  {alert_html}

  <h2>Next 6 hours · 2-hour blocks · NA average</h2>
  <table class="grid">
    <thead><tr>
      <th>band</th>
      <th>{col_headers[0]}<br><span style="font-weight:400;color:var(--tert)">L</span></th>
      <th>{col_headers[1]}<br><span style="font-weight:400;color:var(--tert)">L</span></th>
      <th>{col_headers[2]}<br><span style="font-weight:400;color:var(--tert)">L</span></th>
    </tr></thead>
    <tbody>
"""
    for band, cells in grid_rows:
        html += f"      <tr><td>{band}</td>"
        for cls, lbl, pct in cells:
            html += f'<td><div class="cell cell-{cls}"><div>{lbl}</div><div class="pct">{pct}</div></div></td>'
        html += "</tr>\n"
    html += f"""    </tbody>
  </table>
  <div class="legend">
    <span class="legend-item"><span class="legend-swatch" style="background:var(--good-bg)"></span>good ≥55%</span>
    <span class="legend-item"><span class="legend-swatch" style="background:var(--fair-bg)"></span>fair 25–54%</span>
    <span class="legend-item"><span class="legend-swatch" style="background:var(--poor-bg)"></span>poor &lt;25%</span>
    <span style="margin-left:auto">best band right now: <strong>{best_band.upper()}</strong> ({best_pct}%)</span>
  </div>

  <h2>Rest of day · coarse outlook</h2>
  <div class="strip">
"""
    for label, msg in rod_data:
        html += f'    <div class="cell-strip"><div class="label">{label}</div><div class="val">{msg}</div></div>\n'
    html += """  </div>

  <h2>Current space weather</h2>
  <div class="stats">
"""
    html += f"""    <div class="stat"><div class="label">SFI</div><div class="val">{cur['sfi']}</div><div class="sub">solar flux 10.7cm</div></div>
    <div class="stat"><div class="label">Kp</div><div class="val">{cur['kp']}</div><div class="sub">{n['geomag']}</div></div>
    <div class="stat"><div class="label">X-ray</div><div class="val">{n['xray']}</div><div class="sub">A-index {n['aIndex']} · aurora {n['aurora']}</div></div>
    <div class="stat"><div class="label">SSN</div><div class="val">{cur['ssn']}</div><div class="sub">solar wind {n['solarWind']} km/s · Bz {n['magneticField']}</div></div>
"""
    html += """  </div>

  <h2>Live POTA activity · last 30 min · NA only</h2>
"""
    for band, count, mode_str, w in pota_rows:
        spot_label = "spot" if count == 1 else "spots"
        html += (f'  <div class="pota-row">'
                 f'<span class="pota-band">{band}</span>'
                 f'<span class="pota-bar"><span class="pota-fill" style="width:{w}%"></span></span>'
                 f'<span class="pota-cnt">{count} {spot_label} · {mode_str}</span>'
                 f'</div>\n')

    # EN11 spots callout
    en11 = p.get("en11Spots") or []
    if en11:
        s = en11[0]
        html += (f'<div class="alert" style="background:var(--info-bg);color:var(--info-fg);border-left-color:var(--info-fg);margin-top:8px">'
                 f'Same-grid signal: <strong>{s["activator"]}</strong> in {s.get("grid") or "?"} on {s.get("frequency")} {s.get("mode") or ""} '
                 f'({s.get("ref")} {s.get("name") or ""})</div>')

    html += """  <h2>3-day forecast</h2>
  <div class="strip">
"""
    for day, txt, sfi_val in forecast_strip:
        sub = f" · SFI ~{sfi_val}" if sfi_val else ""
        html += f'    <div class="cell-strip"><div class="label">{day}</div><div class="val">{txt}{sub}</div></div>\n'
    html += """  </div>

  <h2>Field-ready summary</h2>
  <div class="field">""" + field + """</div>

  <div class="footer">data: openhamclock.com /api/propagation + /api/solar-indices · NOAA SWPC · POTA · sunrise-sunset.org · model: ITU-R P.533 (when available) or built-in heuristic from solar indices</div>
</div>
</body></html>
"""
    return html


def _scales_label(scale: dict | None) -> str:
    """Convert a NOAA scales entry to a short label."""
    if not scale:
        return "—"
    parts = []
    g = scale.get("G") or {}
    r = scale.get("R") or {}
    s = scale.get("S") or {}
    if g.get("Scale") and g["Scale"] != "0":
        parts.append(f"G{g['Scale']} {g.get('Text','')}")
    elif r.get("MinorProb") or r.get("MajorProb"):
        bits = []
        if r.get("MinorProb"):
            bits.append(f"R1 {r['MinorProb']}%")
        if r.get("MajorProb"):
            bits.append(f"R3 {r['MajorProb']}%")
        if bits:
            parts.append("/".join(bits))
    if r.get("Scale") and r["Scale"] != "0":
        parts.append(f"R{r['Scale']} blackout")
    if s.get("Prob"):
        parts.append(f"S {s['Prob']}%")
    return " · ".join(parts) if parts else "quiet"


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid", default=HOME_GRID_DEFAULT,
                    help=f"Maidenhead grid (4 or 6 char, default {HOME_GRID_DEFAULT})")
    ap.add_argument("--html", default="report.html", help="HTML output path")
    ap.add_argument("--txt",  default="report.txt",  help="text output path")
    ap.add_argument("--json", default=None, help="optional path to dump raw fetched JSON")
    ap.add_argument("--quiet", action="store_true", help="suppress progress messages")
    ap.add_argument("--data-json", default=None, help="write dashboard data.json")
    args = ap.parse_args(argv)

    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr))

    try:
        lat, lon = grid_to_latlon(args.grid)
    except Exception as e:
        sys.exit(f"bad grid: {e}")
    log(f"# grid {args.grid} -> {lat:.3f},{lon:.3f}")
    log("# fetching live data...")
    t0 = time.time()
    raw = fetch_all((lat, lon))
    log(f"# fetched in {time.time()-t0:.1f}s")

    if args.json:
        Path(args.json).write_text(json.dumps(raw, indent=2, default=str))
        log(f"# wrote {args.json}")

    payload = aggregate(raw)
    Path(args.html).write_text(render_html(payload, args.grid))
    Path(args.txt).write_text(render_text(payload, args.grid))
    log(f"# wrote {args.html} and {args.txt}")
    if args.data_json:
        import datetime as _dt
        dj = {
            "generated": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "grid": args.grid,
            "payload": payload,
        }
        Path(args.data_json).write_text(json.dumps(dj, indent=2, default=str))
        log(f"# wrote {args.data_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
