"""Shared policy for resources the MCP server can actually query."""

from pathlib import PurePosixPath
from typing import Any, Dict
from urllib.parse import urlparse


QUERYABLE_FILE_FORMATS = {
    "CSV",
    "JSON",
    "GEOJSON",
    "PARQUET",
    "PDF",
    "TXT",
    "ZIP",
    "XLS",
    "XLSX",
    "TEXT/CSV",
    "APPLICATION/JSON",
    "APPLICATION/GEO+JSON",
    "APPLICATION/VND.GEO+JSON",
    "APPLICATION/PARQUET",
    "APPLICATION/VND.APACHE.PARQUET",
    "APPLICATION/VND.MS-EXCEL",
    "APPLICATION/VND.OPENXMLFORMATS-OFFICEDOCUMENT.SPREADSHEETML.SHEET",
}

QUERYABLE_SUFFIXES = {
    ".csv", ".json", ".geojson", ".parquet", ".pdf", ".txt", ".xls", ".xlsx", ".zip"
}
EXCLUDED_FORMATS = {"SHP", "SHAPEFILE", "WMS", "WFS", "HTML", "HTM", "KML", "KMZ", "GPKG"}
GEOSPATIAL_ZIP_HINTS = ("shapefile", "shape file", ".shp", "geodatabase", ".gdb")


def resource_format(resource: Dict[str, Any]) -> str:
    """Return a stable uppercase resource format label."""
    return str(resource.get("format") or resource.get("mimetype") or "").strip().upper()


def is_queryable_resource(resource: Dict[str, Any]) -> bool:
    """True only when an existing MCP query path can read the resource."""
    if bool(resource.get("datastore_active")):
        return True

    fmt = resource_format(resource)
    if fmt in EXCLUDED_FORMATS:
        return False
    name = str(resource.get("name") or "").lower()
    if fmt == "ZIP" and any(hint in name for hint in GEOSPATIAL_ZIP_HINTS):
        return False
    if fmt in QUERYABLE_FILE_FORMATS:
        return True

    url = str(resource.get("url") or "").strip()
    path = PurePosixPath(urlparse(url).path.lower())
    if path.suffix in QUERYABLE_SUFFIXES:
        if path.suffix == ".zip" and any(
            hint in path.name for hint in GEOSPATIAL_ZIP_HINTS
        ):
            return False
        return True

    return False
