"""
GPS extraction and validation from EXIF data.
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Optional, Tuple

from PIL import Image

from config import GPS_DISTANCE_THRESHOLD_KM
from schemas.prediction import GPSStatus


def _read_exif(image: Image.Image) -> dict | None:
    """Return the EXIF dict if present."""
    exif = getattr(image, "getexif", None)
    if exif is None:
        return None
    try:
        data = exif()
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _gps_tag_value(exif: dict, tag: int) -> Optional[Fraction | tuple] | None:
    """Safely read a GPS tag."""
    try:
        val = exif.get(tag)
    except Exception:  # noqa: BLE001
        return None
    return val


def _parse_degrees(value) -> Optional[float]:
    """Parse an EXIF GPS degrees value into a float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, tuple) and len(value) == 3:
        try:
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + m / 60.0 + s / 3600.0
        except Exception:  # noqa: BLE001
            return None
    if isinstance(value, Fraction):
        return float(value)
    return None


def extract_gps(image: Image.Image) -> Tuple[Optional[float], Optional[float]]:
    """Extract (latitude, longitude) from image EXIF if available."""
    exif = _read_exif(image)
    if not exif:
        return None, None

    # EXIF GPS tags
    GPS_LATITUDE = 34853
    GPS_LATITUDE_REF = 34854
    GPS_LONGITUDE = 34855
    GPS_LONGITUDE_REF = 34856

    lat_raw = _gps_tag_value(exif, GPS_LATITUDE)
    lat_ref_raw = _gps_tag_value(exif, GPS_LATITUDE_REF)
    lon_raw = _gps_tag_value(exif, GPS_LONGITUDE)
    lon_ref_raw = _gps_tag_value(exif, GPS_LONGITUDE_REF)

    if lat_raw is None or lon_raw is None:
        return None, None

    latitude = _parse_degrees(lat_raw)
    longitude = _parse_degrees(lon_raw)

    if latitude is None or longitude is None:
        return None, None

    # Apply ref sign
    if lat_ref_raw is not None:
        try:
            if str(lat_ref_raw).upper() == "S":
                latitude = -abs(latitude)
        except Exception:  # noqa: BLE001
            pass
    if lon_ref_raw is not None:
        try:
            if str(lon_ref_raw).upper() == "W":
                longitude = -abs(longitude)
        except Exception:  # noqa: BLE001
            pass

    return latitude, longitude


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate distance in km between two GPS points."""
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def check_gps(
    image: Image.Image,
    object_latitude: Optional[float],
    object_longitude: Optional[float],
) -> dict:
    """Compare image GPS with the specified object coordinates.

    Returns a dict with the GPS status and optional distance.
    """
    lat, lon = extract_gps(image)
    result: dict = {
        "gps_status": GPSStatus.GPS_NOT_AVAILABLE,
        "image_latitude": lat,
        "image_longitude": lon,
        "object_latitude": object_latitude,
        "object_longitude": object_longitude,
        "distance_km": None,
    }

    if lat is None or lon is None:
        return result

    result["gps_status"] = GPSStatus.GPS_FOUND

    if object_latitude is None or object_longitude is None:
        return result

    dist = haversine_distance(lat, lon, object_latitude, object_longitude)
    result["distance_km"] = round(dist, 2)

    if dist <= GPS_DISTANCE_THRESHOLD_KM:
        result["gps_status"] = GPSStatus.GPS_MATCH
    else:
        result["gps_status"] = GPSStatus.GPS_MISMATCH

    return result
