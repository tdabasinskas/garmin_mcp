"""
Course management functions for Garmin Connect MCP Server.

Adds support for uploading GPX files as Garmin Connect Courses, listing courses,
getting detailed course waypoints/metadata, downloading course GPX files, and deleting courses.
"""

import datetime
from decimal import Decimal
import io
import json
import math
import os
import pathlib
import random
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from xml.sax.saxutils import escape

from garmin_mcp.activity_analysis import _resolve_download_dir

# The garmin_client will be set by the main file
garmin_client = None


def configure(client):
    """Configure the module with the Garmin client instance"""
    global garmin_client
    garmin_client = client


def _gpx_decimal(value: Any) -> str:
    """GPX uses xsd:decimal; expand exponent notation without rounding."""
    return format(Decimal(str(value)), "f")


_EARTH_RADIUS_M = 6371000.0


def _haversine(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    lat1, lon1 = math.radians(p1["latitude"]), math.radians(p1["longitude"])
    lat2, lon2 = math.radians(p2["latitude"]), math.radians(p2["longitude"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _initial_bearing(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    lat1, lat2 = math.radians(p1["latitude"]), math.radians(p2["latitude"])
    dlon = math.radians(p2["longitude"] - p1["longitude"])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    return (initial_bearing + 360) % 360


# Garmin Connect activity typeIds (see get_activity_types).
_ACTIVITY_TYPE_IDS = {
    "running": 1,
    "cycling": 2,
    "hiking": 3,
    "walking": 9,
    "trail_running": 6,
    "mountain_biking": 5,
    "road_biking": 10,
    "gravel_cycling": 143,
}


def _build_course_payload(
    parsed_skeleton: Dict[str, Any],
    course_name: str,
    activity_type_id: int,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    geo_points = parsed_skeleton.get("geoPoints", [])
    if not geo_points:
        raise ValueError("GPX parsed skeleton contains no geoPoints")

    total_distance = 0.0
    for i in range(1, len(geo_points)):
        total_distance += _haversine(geo_points[i - 1], geo_points[i])

    for p in geo_points:
        if p.get("elevation") is None:
            p["elevation"] = 0.0

    lats = [p["latitude"] for p in geo_points]
    lons = [p["longitude"] for p in geo_points]

    bbox = {
        "center": {
            "latitude": (min(lats) + max(lats)) / 2,
            "longitude": (min(lons) + max(lons)) / 2,
        },
        "lowerLeft": {"latitude": min(lats), "longitude": min(lons)},
        "upperRight": {"latitude": max(lats), "longitude": max(lons)},
        "lowerLeftLatIsSet": True,
        "lowerLeftLongIsSet": True,
        "upperRightLatIsSet": True,
        "upperRightLongIsSet": True,
    }

    start_point = {
        "latitude": geo_points[0]["latitude"],
        "longitude": geo_points[0]["longitude"],
        "elevation": geo_points[0].get("elevation") or 0.0,
        "distance": None,
        "timestamp": None,
    }

    bearing = _initial_bearing(geo_points[0], geo_points[-1])

    return {
        "courseName": course_name,
        "description": description,
        "openStreetMap": False,
        "matchedToSegments": False,
        "userProfilePk": None,
        "userGroupPk": None,
        "rulePK": 2,  # private
        "geoRoutePk": None,
        "sourceTypeId": 3,  # GPX
        "sourcePk": None,
        "distanceMeter": total_distance,
        "elevationGainMeter": 0.0,
        "elevationLossMeter": 0.0,
        "startPoint": start_point,
        "coursePoints": [],
        "boundingBox": bbox,
        "hasShareableEvent": False,
        "hasTurnDetectionDisabled": False,
        "activityTypePk": activity_type_id,
        "virtualPartnerId": None,
        "includeLaps": False,
        "elapsedSeconds": None,
        "startBearing": bearing,
        "endBearing": None,
        "geoPoints": geo_points,
    }


# --- Course Creator routing -------------------------------------------------
#
# The Course Creator on connect.garmin.com routes over Garmin's own map with
# these course-service endpoints; the tools below call them the same way.

# Route types as named in the Course Creator, mapped to the API's routeType.
_ROUTE_TYPES = {
    "popular": "POPULARITY",  # "Most Popular": Garmin's popularity routing
    "most_direct": "FOLLOW_ROADS",  # "Most Direct": shortest along roads
    "least_elevation": "MIN_ASCENT",
    "free_flow": "FREE_FLOW",  # cycling only: fewer stops and turns
}
_ROUND_TRIP_ROUTE_TYPES = ("popular", "least_elevation", "free_flow")
_CYCLING_COURSE_TYPES = {"ROAD_CYCLING", "GRAVEL_BIKING", "MOUNTAIN_BIKING"}

# activity_type -> routing courseType (the routing profile).  Garmin has no
# walking profile, so walking routes as hiking.
_ROUTING_COURSE_TYPES = {
    "running": "RUNNING",
    "trail_running": "TRAIL_RUNNING",
    "hiking": "HIKING",
    "walking": "HIKING",
    "cycling": "ROAD_CYCLING",
    "road_biking": "ROAD_CYCLING",
    "mountain_biking": "MOUNTAIN_BIKING",
    "gravel_cycling": "GRAVEL_BIKING",
}

_ROUND_TRIP_DIRECTIONS = ("north", "south", "east", "west", "tailwind")
_MAX_WAYPOINTS = 25


def _avoid_params(
    avoid_unpaved_roads: bool, avoid_high_traffic_roads: bool, avoid_road_closures: bool
) -> Dict[str, str]:
    params = {}
    if avoid_road_closures:
        params["avoidRoadClosures"] = "true"
    if avoid_unpaved_roads:
        params["avoidUnpavedRoads"] = "true"
    if avoid_high_traffic_roads:
        params["avoidHighTrafficRoads"] = "true"
    return params


def _check_route_options(activity_type: str, route_type: str) -> tuple:
    """(activity_type_id, routing course type) or raise ValueError."""
    activity_type = activity_type.lower()
    activity_type_id = _ACTIVITY_TYPE_IDS.get(activity_type)
    if activity_type_id is None:
        raise ValueError(
            f"unknown activity_type '{activity_type}'. "
            f"Supported: {', '.join(sorted(_ACTIVITY_TYPE_IDS))}."
        )
    course_type = _ROUTING_COURSE_TYPES[activity_type]
    if route_type == "free_flow" and course_type not in _CYCLING_COURSE_TYPES:
        raise ValueError("route_type 'free_flow' is only available for cycling activity types.")
    return activity_type_id, course_type


def _parse_waypoints(waypoints: List[Any]) -> List[Dict[str, float]]:
    """Accept [[lat, lon], ...] or [{"latitude"/"lat", "longitude"/"lon"}, ...]."""
    points = []
    for item in waypoints:
        if isinstance(item, dict):
            lat = item.get("latitude", item.get("lat"))
            lon = item.get("longitude", item.get("lon", item.get("lng")))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lat, lon = item[0], item[1]
        else:
            raise ValueError(f"waypoint {item!r} is not [lat, lon]")
        lat, lon = float(lat), float(lon)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"waypoint ({lat}, {lon}) is out of range")
        points.append({"latitude": lat, "longitude": lon})
    if len(points) < 2:
        raise ValueError("at least two waypoints (start and end) are needed")
    if len(points) > _MAX_WAYPOINTS:
        raise ValueError(f"at most {_MAX_WAYPOINTS} waypoints are supported")
    return points


def _route_leg(client, start, end, route_type: str, course_type: str, avoid: Dict[str, str]):
    """Routed points for one leg, as [{"latitude", "longitude"}, ...]."""
    query = urlencode({"routeType": route_type, "courseType": course_type, **avoid})
    data = client.client.post(
        "connectapi",
        f"/course-service/course/route/v2?{query}",
        json={
            "startLat": start["latitude"],
            "startLon": start["longitude"],
            "endLat": end["latitude"],
            "endLon": end["longitude"],
            "deviceType": "WEB",
        },
        api=True,
    )
    points = [
        {"latitude": p[0], "longitude": p[1]}
        for segment in (data or {}).get("route") or []
        for p in segment
    ]
    if not points:
        raise ValueError(
            "Garmin found no route between "
            f"({start['latitude']}, {start['longitude']}) and ({end['latitude']}, {end['longitude']})"
        )
    return points


def _with_elevation(client, points: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Fill in elevations from Garmin's elevation service (as the Creator does)."""
    data = client.client.post(
        "connectapi",
        "/course-service/course/elevation?smoothingEnabled=true",
        json=[[p["latitude"], p["longitude"], None] for p in points],
        api=True,
    )
    if isinstance(data, list) and len(data) == len(points):
        for p, row in zip(points, data):
            if isinstance(row, (list, tuple)) and len(row) >= 3 and row[2] is not None:
                p["elevation"] = row[2]
    return points


def _route_summary(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    distance = sum(_haversine(points[i - 1], points[i]) for i in range(1, len(points)))
    gain = loss = 0.0
    for i in range(1, len(points)):
        a, b = points[i - 1].get("elevation"), points[i].get("elevation")
        if a is not None and b is not None:
            gain += max(b - a, 0.0)
            loss += max(a - b, 0.0)
    return {
        "distance_m": round(distance, 1),
        "elevation_gain_m": round(gain, 1),
        "elevation_loss_m": round(loss, 1),
        "points_count": len(points),
    }


def _save_course(client, points, course_name, activity_type_id, description) -> Dict[str, Any]:
    payload = _build_course_payload(
        {"geoPoints": points},
        course_name=course_name,
        activity_type_id=activity_type_id,
        description=description,
    )
    saved = client.client.post("connectapi", "/course-service/course", json=payload, api=True)
    return {
        "status": "success",
        "course_id": saved.get("courseId"),
        "name": saved.get("courseName"),
        "distance_m": saved.get("distanceMeter"),
        "elevation_gain_m": saved.get("elevationGainMeter"),
        "elevation_loss_m": saved.get("elevationLossMeter"),
        "activity_type_id": saved.get("activityTypePk"),
        "url": f"https://connect.{client.client.domain}/modern/course/{saved.get('courseId')}",
    }


def _resolve_gpx_output_path(course_id: int, output_path: Optional[str] = None) -> str:
    """Resolve the destination file path for downloading a course GPX.

    A directory output_path gets {course_id}.gpx; a file output_path must end
    in .gpx so the tool cannot overwrite arbitrary files. Without output_path,
    use the same directory as download_activity_file (GARMIN_FIT_DOWNLOAD_DIR,
    then the one saved by set_fit_download_dir), falling back to ./courses.
    """
    if output_path:
        p = os.path.abspath(os.path.expanduser(output_path))
        if os.path.isdir(p) or output_path.endswith("/") or output_path.endswith("\\"):
            return os.path.join(p, f"{course_id}.gpx")
        if not p.lower().endswith(".gpx"):
            raise ValueError(
                f"output_path must be a directory or a .gpx file, got: {output_path}"
            )
        return p

    base_dir = _resolve_download_dir(None) or os.path.abspath("./courses")
    return os.path.join(base_dir, f"{course_id}.gpx")


def _course_preview(client: Any, course_id: int) -> Dict[str, Any]:
    """Build a compact preview of a course before deleting it."""
    preview: Dict[str, Any] = {"course_id": course_id}

    course = None
    try:
        course = client.client.connectapi(f"/course-service/course/{course_id}")
    except Exception:
        course = None

    if not (isinstance(course, dict) and course):
        # Fall back to the course list if the details call fails.
        try:
            items = client.client.connectapi("/course-service/course")
        except Exception:
            items = None
        if not isinstance(items, list):
            return preview
        course = next(
            (
                item
                for item in items
                if isinstance(item, dict) and str(item.get("courseId")) == str(course_id)
            ),
            None,
        )
        if course is None:
            return preview

    activity_type = course.get("activityType")
    activity_type = activity_type if isinstance(activity_type, dict) else {}
    preview.update(
        {
            "name": course.get("courseName"),
            "activity": activity_type.get("typeKey"),
            "distance_m": course.get("distanceInMeters") or course.get("distanceMeter"),
            "created": course.get("createdDateFormatted"),
        }
    )
    return {key: value for key, value in preview.items() if value is not None}


def register_tools(app):
    """Register course management tools"""

    @app.tool()
    async def get_courses() -> str:
        """List all courses saved on Garmin Connect.

        Returns a curated list of courses with id, name, distance, activity type
        and creation date.
        """
        try:
            data = garmin_client.client.connectapi("/course-service/course")

            if not isinstance(data, list):
                return json.dumps(data, indent=2)

            curated = [
                {
                    "course_id": c.get("courseId"),
                    "name": c.get("courseName"),
                    "distance_m": c.get("distanceInMeters"),
                    "elevation_gain_m": c.get("elevationGainInMeters"),
                    "elevation_loss_m": c.get("elevationLossInMeters"),
                    "activity": (c.get("activityType") or {}).get("typeKey"),
                    "has_pace_band": c.get("hasPaceBand"),
                    "created": c.get("createdDateFormatted"),
                }
                for c in data
            ]
            return json.dumps({"count": len(curated), "courses": curated}, indent=2)
        except Exception as e:
            return f"Error listing courses: {str(e)}"

    @app.tool()
    async def get_course_details(course_id: int) -> str:
        """Get full details of a Garmin Connect course by ID.

        Returns course metadata, elevation gain/loss, total distance,
        and all custom course waypoints (shops, water, food, campgrounds, hazards).

        Args:
            course_id: ID of the course (from get_courses).
        """
        try:
            data = garmin_client.client.connectapi(f"/course-service/course/{course_id}")
            if not isinstance(data, dict):
                return json.dumps(data, indent=2)

            course_points = [
                {
                    "name": cp.get("name"),
                    "type": cp.get("pointType"),
                    "lat": cp.get("lat"),
                    "lon": cp.get("lon"),
                    "distance_m": cp.get("distance"),
                }
                for cp in (data.get("coursePoints") or [])
            ]
            geo_points = data.get("geoPoints") or []

            result = {
                "course_id": data.get("courseId"),
                "name": data.get("courseName"),
                "distance_m": data.get("distanceInMeters") or data.get("distanceMeter"),
                "elevation_gain_m": data.get("elevationGainInMeters") or data.get("elevationGainMeter"),
                "elevation_loss_m": data.get("elevationLossInMeters") or data.get("elevationLossMeter"),
                "activity": (data.get("activityType") or {}).get("typeKey"),
                "waypoints_count": len(course_points),
                "waypoints": course_points,
                "geo_points_count": len(geo_points),
                "url": f"https://connect.{garmin_client.client.domain}/modern/course/{course_id}",
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error fetching course details: {str(e)}"

    @app.tool()
    async def download_course_gpx(
        course_id: int,
        output_path: Optional[str] = None,
    ) -> str:
        """Download a Garmin Connect course as a GPX file.

        The GPX is rebuilt from the course's track points (lat/lon/elevation)
        and course points (as waypoints); it is not the originally uploaded file.

        Args:
            course_id: ID of the course to download.
            output_path: Optional destination directory (saved as {course_id}.gpx)
                or file path ending in .gpx; any other file path is rejected.
                Defaults to the download_activity_file directory
                (GARMIN_FIT_DOWNLOAD_DIR, then set_fit_download_dir), else
                ./courses/{course_id}.gpx.
        """
        try:
            target_path = _resolve_gpx_output_path(course_id, output_path)

            data = garmin_client.client.connectapi(f"/course-service/course/{course_id}")
            if not isinstance(data, dict) or not data.get("geoPoints"):
                return f"Error: course {course_id} not found or missing geoPoints."

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            name = data.get("courseName") or f"Garmin Course {course_id}"
            geo_points = data.get("geoPoints") or []
            course_points = data.get("coursePoints") or []

            lines = [
                "<?xml version='1.0' encoding='UTF-8'?>",
                "<gpx version='1.1' creator='GarminConnectMCP' xmlns='http://www.topografix.com/GPX/1/1'>",
                "  <metadata>",
                f"    <name>{escape(name)}</name>",
                "  </metadata>",
            ]
            for cp in course_points:
                lat, lon = cp.get("lat"), cp.get("lon")
                if lat is not None and lon is not None:
                    cp_name = cp.get("name") or cp.get("pointType") or "Waypoint"
                    cp_type = cp.get("pointType") or "GENERIC"
                    lines.append(f"  <wpt lat='{_gpx_decimal(lat)}' lon='{_gpx_decimal(lon)}'>")
                    lines.append(f"    <name>{escape(cp_name)}</name>")
                    lines.append(f"    <type>{escape(cp_type)}</type>")
                    lines.append("  </wpt>")

            lines.append("  <trk>")
            lines.append(f"    <name>{escape(name)}</name>")
            lines.append("    <trkseg>")
            for p in geo_points:
                lat = p.get("latitude")
                lon = p.get("longitude")
                ele = p.get("elevation")
                if lat is not None and lon is not None:
                    ele_tag = f"<ele>{_gpx_decimal(ele)}</ele>" if ele is not None else ""
                    lines.append(f"      <trkpt lat='{_gpx_decimal(lat)}' lon='{_gpx_decimal(lon)}'>{ele_tag}</trkpt>")
            lines.append("    </trkseg>")
            lines.append("  </trk>")
            lines.append("</gpx>")

            with open(target_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            return json.dumps(
                {
                    "status": "success",
                    "course_id": course_id,
                    "name": name,
                    "gpx_path": target_path,
                    "waypoints_count": len(course_points),
                    "track_points_count": len(geo_points),
                },
                indent=2,
            )
        except Exception as e:
            return f"Error downloading course GPX: {str(e)}"

    @app.tool()
    async def upload_course(
        gpx_path: str,
        course_name: Optional[str] = None,
        activity_type: str = "running",
        description: Optional[str] = None,
    ) -> str:
        """Upload a GPX file as a Garmin Connect Course.

        The course can then be loaded onto the watch (sync or "Send to Device")
        and used as a navigation course or to build a PacePro strategy.

        Args:
            gpx_path: Absolute path to the .gpx file on disk.
            course_name: Override the course name. Defaults to the name parsed
                from the GPX file. Max 128 characters; Garmin rejects longer.
            activity_type: One of running, cycling, hiking, walking, trail_running,
                mountain_biking, road_biking, gravel_cycling. Defaults to running.
            description: Optional description shown on the course detail page.
                Max 512 characters; Garmin rejects longer.
        """
        try:
            _p = pathlib.Path(gpx_path)
            if _p.suffix.lower() != ".gpx":
                return f"Error: only .gpx files are allowed, got: {_p.suffix or '(no extension)'}"
            gpx_path = str(_p.resolve())
            if not os.path.isfile(gpx_path):
                return f"Error: GPX file not found: {gpx_path}"

            activity_type_id = _ACTIVITY_TYPE_IDS.get(activity_type.lower())
            if activity_type_id is None:
                return (
                    f"Error: unknown activity_type '{activity_type}'. "
                    f"Supported: {', '.join(sorted(_ACTIVITY_TYPE_IDS))}."
                )

            with open(gpx_path, "rb") as f:
                gpx_bytes = f.read()

            # Step 1: parse the GPX server-side
            parsed = garmin_client.client.post(
                "connectapi",
                "/course-service/course/import",
                files={
                    "file": (
                        os.path.basename(gpx_path),
                        io.BytesIO(gpx_bytes),
                        "application/gpx+xml",
                    )
                },
                api=True,
            )

            effective_name = (
                course_name
                or parsed.get("courseName")
                or os.path.splitext(os.path.basename(gpx_path))[0]
            )

            # Step 2: build the create payload and save
            payload = _build_course_payload(
                parsed,
                course_name=effective_name,
                activity_type_id=activity_type_id,
                description=description,
            )

            saved = garmin_client.client.post(
                "connectapi", "/course-service/course", json=payload, api=True,
            )
            return json.dumps(
                {
                    "status": "success",
                    "course_id": saved.get("courseId"),
                    "name": saved.get("courseName"),
                    "distance_m": saved.get("distanceMeter"),
                    "elevation_gain_m": saved.get("elevationGainMeter"),
                    "elevation_loss_m": saved.get("elevationLossMeter"),
                    "activity_type_id": saved.get("activityTypePk"),
                    "url": f"https://connect.{garmin_client.client.domain}/modern/course/{saved.get('courseId')}",
                },
                indent=2,
            )

        except Exception as e:
            return f"Error uploading course: {str(e)}"

    @app.tool()
    async def create_course_from_waypoints(
        waypoints: List[Any],
        course_name: str,
        activity_type: str = "running",
        route_type: str = "popular",
        avoid_unpaved_roads: bool = False,
        avoid_high_traffic_roads: bool = False,
        avoid_road_closures: bool = False,
        description: Optional[str] = None,
        save: bool = True,
    ) -> str:
        """Create a Garmin Connect Course routed through waypoints

        Works like the Course Creator on connect.garmin.com: each leg between
        consecutive waypoints is routed over Garmin's map, and elevations come
        from Garmin's elevation service.

        Args:
            waypoints: Start, optional via points and end, as [[lat, lon], ...]
                or [{"latitude": .., "longitude": ..}, ...]; 2 to 25 points.
                Repeat the start at the end for a loop.
            course_name: Name of the course. Max 128 characters.
            activity_type: One of running, cycling, hiking, walking,
                trail_running, mountain_biking, road_biking, gravel_cycling.
                It also picks the routing profile (walking routes as hiking).
            route_type: popular (Garmin's popularity routing, the Creator's
                default), most_direct (shortest along roads), least_elevation,
                free_flow (cycling only: fewer stops and turns) or freehand
                (straight lines between the waypoints, no routing).
            avoid_unpaved_roads: Avoid unpaved roads.
            avoid_high_traffic_roads: Avoid high-traffic roads.
            avoid_road_closures: Avoid known road closures.
            description: Optional description. Max 512 characters.
            save: False returns the route's distance and climb without
                creating the course.
        """
        try:
            route_type = route_type.lower()
            if route_type != "freehand" and route_type not in _ROUTE_TYPES:
                return (
                    f"Error: unknown route_type '{route_type}'. "
                    f"Supported: {', '.join([*_ROUTE_TYPES, 'freehand'])}."
                )
            activity_type_id, course_type = _check_route_options(activity_type, route_type)
            stops = _parse_waypoints(waypoints)

            if route_type == "freehand":
                points = [dict(p) for p in stops]
            else:
                avoid = _avoid_params(
                    avoid_unpaved_roads, avoid_high_traffic_roads, avoid_road_closures
                )
                points = []
                for start, end in zip(stops, stops[1:]):
                    leg = _route_leg(
                        garmin_client, start, end, _ROUTE_TYPES[route_type], course_type, avoid
                    )
                    if points and leg and points[-1] == leg[0]:
                        leg = leg[1:]
                    points.extend(leg)
            points = _with_elevation(garmin_client, points)

            if not save:
                return json.dumps(
                    {"status": "preview", "name": course_name, **_route_summary(points)},
                    indent=2,
                )
            return json.dumps(
                _save_course(garmin_client, points, course_name, activity_type_id, description),
                indent=2,
            )
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error creating course: {str(e)}"

    @app.tool()
    async def create_round_trip_course(
        start_latitude: float,
        start_longitude: float,
        distance_km: float,
        course_name: str,
        activity_type: str = "running",
        direction: str = "random",
        route_type: str = "popular",
        avoid_unpaved_roads: bool = False,
        avoid_high_traffic_roads: bool = False,
        avoid_road_closures: bool = False,
        alternative: int = 1,
        description: Optional[str] = None,
        save: bool = True,
    ) -> str:
        """Create a loop course of about a given length from a start point

        This is the Course Creator's automatic (round trip) mode. Garmin
        proposes up to three loops; call with save=false to compare them, then
        again with the chosen alternative and the direction the preview
        reports.

        Args:
            start_latitude: Latitude of the start (and finish).
            start_longitude: Longitude of the start (and finish).
            distance_km: Wanted length in kilometres; Garmin's loops come out
                near it, not exactly on it.
            course_name: Name of the course. Max 128 characters.
            activity_type: One of running, cycling, hiking, walking,
                trail_running, mountain_biking, road_biking, gravel_cycling.
            direction: north, south, east, west, tailwind (head out against
                the wind, come back with it) or random.
            route_type: popular, least_elevation or free_flow (cycling only).
            avoid_unpaved_roads: Avoid unpaved roads.
            avoid_high_traffic_roads: Avoid high-traffic roads.
            avoid_road_closures: Avoid known road closures.
            alternative: Which of Garmin's loops to save, 1-based.
            description: Optional description. Max 512 characters.
            save: False lists the proposed loops without creating a course.
        """
        try:
            route_type = route_type.lower()
            if route_type not in _ROUND_TRIP_ROUTE_TYPES:
                return (
                    f"Error: unknown route_type '{route_type}' for a round trip. "
                    f"Supported: {', '.join(_ROUND_TRIP_ROUTE_TYPES)}."
                )
            direction = direction.lower()
            if direction == "random":
                direction = random.choice(_ROUND_TRIP_DIRECTIONS[:4])
            if direction not in _ROUND_TRIP_DIRECTIONS:
                return (
                    f"Error: unknown direction '{direction}'. "
                    f"Supported: {', '.join(_ROUND_TRIP_DIRECTIONS)}, random."
                )
            if not 0 < distance_km <= 500:
                return "Error: distance_km must be between 0 and 500."
            activity_type_id, course_type = _check_route_options(activity_type, route_type)
            _parse_waypoints([[start_latitude, start_longitude]] * 2)

            query = urlencode(
                {
                    "startLat": start_latitude,
                    "startLon": start_longitude,
                    "distance": int(distance_km * 1000),
                    "direction": direction.upper(),
                    "courseType": course_type,
                    "routeType": _ROUTE_TYPES[route_type],
                    **_avoid_params(
                        avoid_unpaved_roads, avoid_high_traffic_roads, avoid_road_closures
                    ),
                }
            )
            data = garmin_client.client.connectapi(
                f"/course-service/course/roundTripRoutes/multiple?{query}"
            )
            loops = [
                c for c in (data or {}).get("roundTripRouteCourses") or [] if c.get("geoPoints")
            ]
            if not loops:
                return "Error: Garmin proposed no loop from this start point."

            if not save:
                return json.dumps(
                    {
                        "status": "preview",
                        "direction": direction,
                        "alternatives": [
                            {
                                "alternative": i,
                                "distance_m": round(c.get("distanceMeter") or 0, 1),
                                "elevation_gain_m": round(c.get("elevationGainMeter") or 0, 1),
                                "elevation_loss_m": round(c.get("elevationLossMeter") or 0, 1),
                                "points_count": len(c["geoPoints"]),
                            }
                            for i, c in enumerate(loops, 1)
                        ],
                    },
                    indent=2,
                )

            if not 1 <= alternative <= len(loops):
                return f"Error: alternative must be between 1 and {len(loops)}."
            points = [
                {
                    "latitude": p["latitude"],
                    "longitude": p["longitude"],
                    "elevation": p.get("elevation"),
                }
                for p in loops[alternative - 1]["geoPoints"]
            ]
            result = _save_course(garmin_client, points, course_name, activity_type_id, description)
            result["direction"] = direction
            return json.dumps(result, indent=2)
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error creating round trip course: {str(e)}"

    @app.tool()
    async def delete_course(course_id: int, confirm: bool = False) -> str:
        """Delete a course from Garmin Connect.

        This cannot be undone. Call once with confirm=false (the default) to
        preview the course, then call again with confirm=true to delete it.

        Args:
            course_id: ID of the course to delete (get IDs from get_courses).
            confirm: Must be true to actually delete. Default false returns a preview.
        """
        try:
            preview = _course_preview(garmin_client, course_id)
            if not confirm:
                return json.dumps(
                    {
                        "status": "needs_confirmation",
                        "course": preview,
                        "message": (
                            "This permanently deletes the course from Garmin Connect "
                            "and cannot be undone. Call again with confirm=true to proceed."
                        ),
                    },
                    indent=2,
                )

            garmin_client.client.delete(
                "connectapi", f"/course-service/course/{course_id}"
            )
            print(
                "delete_course: deleted course "
                f"{course_id} name={preview.get('name')!r} "
                f"at {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
                file=sys.stderr,
            )
            return json.dumps(
                {
                    "status": "success",
                    "course_id": course_id,
                    "course": preview,
                    "message": f"Course {course_id} deleted",
                },
                indent=2,
            )
        except Exception as e:
            return f"Error deleting course: {str(e)}"

    return app
