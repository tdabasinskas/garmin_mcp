"""
Integration tests for the courses module MCP tools.

Covers get_courses, get_course_details, download_course_gpx, upload_course, and delete_course
using FastMCP integration with a mocked Garmin client. No real Garmin account or network access is used.
"""
import json
import os
import xml.etree.ElementTree as ET

import pytest
from mcp.server.fastmcp import FastMCP

from garmin_mcp import courses
from garmin_mcp.courses import _build_course_payload, _haversine, _resolve_gpx_output_path


@pytest.fixture
def app_with_courses(mock_garmin_client):
    """Create a FastMCP app with the courses tools registered."""
    courses.configure(mock_garmin_client)
    app = FastMCP("Test Courses")
    app = courses.register_tools(app)
    return app


def _result_text(result):
    """Extract the text payload from a FastMCP call_tool result."""
    return result[0][0].text


# --- get_courses ----------------------------------------------------------

@pytest.mark.asyncio
async def test_get_courses_curates_fields(app_with_courses, mock_garmin_client):
    """get_courses curates the raw Garmin list into a compact shape."""
    mock_garmin_client.client.connectapi.return_value = [
        {
            "courseId": 111,
            "courseName": "River Loop",
            "distanceInMeters": 10250.5,
            "elevationGainInMeters": 120.0,
            "elevationLossInMeters": 118.0,
            "activityType": {"typeKey": "running"},
            "hasPaceBand": False,
            "createdDateFormatted": "2024-03-01",
        }
    ]

    result = await app_with_courses.call_tool("get_courses", {})

    data = json.loads(_result_text(result))
    assert data["count"] == 1
    course = data["courses"][0]
    assert course["course_id"] == 111
    assert course["name"] == "River Loop"
    assert course["distance_m"] == 10250.5
    assert course["activity"] == "running"
    mock_garmin_client.client.connectapi.assert_called_once_with("/course-service/course")


@pytest.mark.asyncio
async def test_get_courses_empty(app_with_courses, mock_garmin_client):
    """An empty course list returns count 0 and an empty list."""
    mock_garmin_client.client.connectapi.return_value = []

    result = await app_with_courses.call_tool("get_courses", {})

    data = json.loads(_result_text(result))
    assert data["count"] == 0
    assert data["courses"] == []


@pytest.mark.asyncio
async def test_get_courses_error_is_caught(app_with_courses, mock_garmin_client):
    """A client error is surfaced as a clean message, not a traceback."""
    mock_garmin_client.client.connectapi.side_effect = Exception("boom")

    result = await app_with_courses.call_tool("get_courses", {})

    assert "Error listing courses" in _result_text(result)


# --- get_course_details ---------------------------------------------------

@pytest.mark.asyncio
async def test_get_course_details_success(app_with_courses, mock_garmin_client):
    """get_course_details returns full course structure with custom waypoints."""
    mock_garmin_client.client.connectapi.return_value = {
        "courseId": 777,
        "courseName": "Bikepack Day 1",
        "distanceInMeters": 45000.0,
        "elevationGainInMeters": 350.0,
        "elevationLossInMeters": 300.0,
        "activityType": {"typeKey": "gravel_cycling"},
        "coursePoints": [
            {
                "name": "Water Fountain",
                "pointType": "WATER",
                "lat": 52.2,
                "lon": 21.0,
                "distance": 12000.0,
            }
        ],
        "geoPoints": [
            {"latitude": 52.1, "longitude": 20.9},
            {"latitude": 52.2, "longitude": 21.0},
        ],
    }

    result = await app_with_courses.call_tool("get_course_details", {"course_id": 777})

    data = json.loads(_result_text(result))
    assert data["course_id"] == 777
    assert data["name"] == "Bikepack Day 1"
    assert data["distance_m"] == 45000.0
    assert data["waypoints_count"] == 1
    assert data["waypoints"][0]["type"] == "WATER"
    assert data["geo_points_count"] == 2
    mock_garmin_client.client.connectapi.assert_called_once_with("/course-service/course/777")


@pytest.mark.asyncio
async def test_get_course_details_treats_null_course_points_as_empty(
    app_with_courses, mock_garmin_client
):
    """A course with no waypoints still returns its valid geo-point count."""
    mock_garmin_client.client.connectapi.return_value = {
        "courseId": 778,
        "courseName": "Track Only",
        "coursePoints": None,
        "geoPoints": [
            {"latitude": 52.1, "longitude": 20.9},
            {"latitude": 52.2, "longitude": 21.0},
        ],
    }

    result = await app_with_courses.call_tool("get_course_details", {"course_id": 778})

    data = json.loads(_result_text(result))
    assert data["waypoints_count"] == 0
    assert data["waypoints"] == []
    assert data["geo_points_count"] == 2


@pytest.mark.asyncio
async def test_get_course_details_treats_null_geo_points_as_empty(
    app_with_courses, mock_garmin_client
):
    """A course with no route points reports zero geo points cleanly."""
    mock_garmin_client.client.connectapi.return_value = {
        "courseId": 779,
        "courseName": "Metadata Only",
        "coursePoints": None,
        "geoPoints": None,
    }

    result = await app_with_courses.call_tool("get_course_details", {"course_id": 779})

    data = json.loads(_result_text(result))
    assert data["waypoints_count"] == 0
    assert data["geo_points_count"] == 0


@pytest.mark.asyncio
async def test_get_course_details_error_is_caught(app_with_courses, mock_garmin_client):
    """Client error in get_course_details is caught cleanly."""
    mock_garmin_client.client.connectapi.side_effect = Exception("Not found")

    result = await app_with_courses.call_tool("get_course_details", {"course_id": 999})

    assert "Error fetching course details" in _result_text(result)


# --- download_course_gpx ---------------------------------------------------

@pytest.mark.asyncio
async def test_download_course_gpx_expands_small_numbers_to_decimal(
    app_with_courses, mock_garmin_client, tmp_path
):
    """GPX 1.1 lat/lon/elevation are xsd:decimal, which excludes exponents."""
    mock_garmin_client.client.connectapi.return_value = {
        "courseName": "Near the equator and prime meridian",
        "geoPoints": [{"latitude": 0.00005, "longitude": -0.00005, "elevation": 0.000001}],
        "coursePoints": [{"lat": -0.00000001, "lon": 0.00000001}],
    }
    path = tmp_path / "small_numbers.gpx"
    result = await app_with_courses.call_tool(
        "download_course_gpx", {"course_id": 888, "output_path": str(path)}
    )
    assert json.loads(_result_text(result))["status"] == "success"
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    root = ET.parse(path).getroot()
    waypoint = root.find("gpx:wpt", ns)
    point = root.find("gpx:trk/gpx:trkseg/gpx:trkpt", ns)
    assert waypoint.attrib == {"lat": "-0.00000001", "lon": "0.00000001"}
    assert point.attrib == {"lat": "0.00005", "lon": "-0.00005"}
    assert point.find("gpx:ele", ns).text == "0.000001"


@pytest.mark.asyncio
async def test_download_course_gpx_success(app_with_courses, mock_garmin_client, tmp_path):
    """download_course_gpx formats a valid GPX 1.1 file and writes to disk."""
    mock_garmin_client.client.connectapi.return_value = {
        "courseId": 888,
        "courseName": "Mountain Pass",
        "geoPoints": [
            {"latitude": 46.1, "longitude": 8.1, "elevation": 1200.0},
            {"latitude": 46.2, "longitude": 8.2, "elevation": 1400.0},
        ],
        "coursePoints": [
            {"name": "Summit Rest", "pointType": "SUMMIT", "lat": 46.2, "lon": 8.2}
        ],
    }

    out_file = str(tmp_path / "mountain_pass.gpx")
    result = await app_with_courses.call_tool(
        "download_course_gpx", {"course_id": 888, "output_path": out_file}
    )

    data = json.loads(_result_text(result))
    assert data["status"] == "success"
    assert data["course_id"] == 888
    assert data["waypoints_count"] == 1
    assert data["track_points_count"] == 2
    assert os.path.isfile(out_file)

    with open(out_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert "<gpx version='1.1'" in content
    assert "<name>Mountain Pass</name>" in content
    assert "<wpt lat='46.2' lon='8.2'>" in content
    assert "<name>Summit Rest</name>" in content
    assert "<trkpt lat='46.1' lon='8.1'><ele>1200.0</ele></trkpt>" in content


@pytest.mark.asyncio
async def test_download_course_gpx_escapes_xml_special_characters(
    app_with_courses, mock_garmin_client, tmp_path
):
    """Names containing &, < or > still produce well-formed GPX."""
    mock_garmin_client.client.connectapi.return_value = {
        "courseId": 891,
        "courseName": "Rock & Roll <Loop>",
        "geoPoints": [
            {"latitude": 46.1, "longitude": 8.1, "elevation": 1200.0},
            {"latitude": 46.2, "longitude": 8.2, "elevation": 1400.0},
        ],
        "coursePoints": [
            {"name": "Café & Water", "pointType": "WATER", "lat": 46.2, "lon": 8.2}
        ],
    }
    out_file = tmp_path / "rock_and_roll.gpx"

    result = await app_with_courses.call_tool(
        "download_course_gpx", {"course_id": 891, "output_path": str(out_file)}
    )

    data = json.loads(_result_text(result))
    assert data["status"] == "success"
    assert data["name"] == "Rock & Roll <Loop>"

    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    root = ET.parse(out_file).getroot()
    assert root.find("gpx:metadata/gpx:name", ns).text == "Rock & Roll <Loop>"
    assert root.find("gpx:trk/gpx:name", ns).text == "Rock & Roll <Loop>"
    assert root.find("gpx:wpt/gpx:name", ns).text == "Café & Water"
    assert root.find("gpx:wpt/gpx:type", ns).text == "WATER"
    assert len(root.findall("gpx:trk/gpx:trkseg/gpx:trkpt", ns)) == 2


@pytest.mark.asyncio
async def test_download_course_gpx_null_course_name_uses_default(
    app_with_courses, mock_garmin_client, tmp_path
):
    """A null courseName falls back to a default instead of writing 'None'."""
    mock_garmin_client.client.connectapi.return_value = {
        "courseId": 892,
        "courseName": None,
        "geoPoints": [{"latitude": 46.1, "longitude": 8.1, "elevation": None}],
        "coursePoints": [{"name": None, "pointType": None, "lat": 46.1, "lon": 8.1}],
    }
    out_file = tmp_path / "unnamed.gpx"

    result = await app_with_courses.call_tool(
        "download_course_gpx", {"course_id": 892, "output_path": str(out_file)}
    )

    data = json.loads(_result_text(result))
    assert data["name"] == "Garmin Course 892"
    content = out_file.read_text(encoding="utf-8")
    assert "None" not in content

    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    root = ET.fromstring(content)
    assert root.find("gpx:metadata/gpx:name", ns).text == "Garmin Course 892"
    assert root.find("gpx:wpt/gpx:name", ns).text == "Waypoint"
    assert root.find("gpx:wpt/gpx:type", ns).text == "GENERIC"


@pytest.mark.asyncio
async def test_download_course_gpx_preserves_track_when_course_points_are_null(
    app_with_courses, mock_garmin_client, tmp_path
):
    """Null waypoints do not prevent exporting the valid track."""
    mock_garmin_client.client.connectapi.return_value = {
        "courseId": 889,
        "courseName": "Track Only",
        "geoPoints": [
            {"latitude": 46.1, "longitude": 8.1, "elevation": 1200.0},
            {"latitude": 46.2, "longitude": 8.2, "elevation": 1400.0},
        ],
        "coursePoints": None,
    }
    out_file = str(tmp_path / "track_only.gpx")

    result = await app_with_courses.call_tool(
        "download_course_gpx", {"course_id": 889, "output_path": out_file}
    )

    data = json.loads(_result_text(result))
    assert data["status"] == "success"
    assert data["waypoints_count"] == 0
    assert data["track_points_count"] == 2
    assert "<trkpt lat='46.1' lon='8.1'><ele>1200.0</ele></trkpt>" in (
        open(out_file, encoding="utf-8").read()
    )


@pytest.mark.asyncio
async def test_download_course_gpx_reports_null_geo_points_without_writing(
    app_with_courses, mock_garmin_client, tmp_path
):
    """Null geo points are reported as missing route data, not iterated."""
    mock_garmin_client.client.connectapi.return_value = {
        "courseId": 890,
        "courseName": "Metadata Only",
        "geoPoints": None,
        "coursePoints": None,
    }
    out_file = tmp_path / "metadata_only.gpx"

    result = await app_with_courses.call_tool(
        "download_course_gpx", {"course_id": 890, "output_path": str(out_file)}
    )

    assert "Error: course 890 not found or missing geoPoints" in _result_text(result)
    assert not out_file.exists()


@pytest.mark.asyncio
async def test_download_course_gpx_missing_course(app_with_courses, mock_garmin_client):
    """Missing course surfaces a clean error message."""
    mock_garmin_client.client.connectapi.return_value = {}

    result = await app_with_courses.call_tool("download_course_gpx", {"course_id": 404})

    assert "Error: course 404 not found" in _result_text(result)


_TRACK_ONLY_COURSE = {
    "courseId": 893,
    "courseName": "Track Only",
    "geoPoints": [{"latitude": 46.1, "longitude": 8.1, "elevation": 1200.0}],
    "coursePoints": None,
}


@pytest.mark.asyncio
async def test_download_course_gpx_rejects_non_gpx_output_file(
    app_with_courses, mock_garmin_client, tmp_path
):
    """A non-.gpx file path is rejected and the existing file is left untouched."""
    mock_garmin_client.client.connectapi.return_value = dict(_TRACK_ONLY_COURSE)
    target = tmp_path / "x" / ".zshrc"
    target.parent.mkdir()
    target.write_text("export PATH=/usr/bin\n")

    result = await app_with_courses.call_tool(
        "download_course_gpx", {"course_id": 893, "output_path": str(target)}
    )

    assert "output_path must be a directory or a .gpx file" in _result_text(result)
    assert target.read_text() == "export PATH=/usr/bin\n"
    mock_garmin_client.client.connectapi.assert_not_called()


@pytest.mark.asyncio
async def test_download_course_gpx_accepts_gpx_output_file(
    app_with_courses, mock_garmin_client, tmp_path
):
    """A file path ending in .gpx (any case) is written as given."""
    mock_garmin_client.client.connectapi.return_value = dict(_TRACK_ONLY_COURSE)
    target = tmp_path / "Route.GPX"

    result = await app_with_courses.call_tool(
        "download_course_gpx", {"course_id": 893, "output_path": str(target)}
    )

    data = json.loads(_result_text(result))
    assert data["gpx_path"] == str(target)
    assert ET.parse(target).getroot().tag == "{http://www.topografix.com/GPX/1/1}gpx"


@pytest.mark.asyncio
async def test_download_course_gpx_defaults_to_configured_download_dir(
    app_with_courses, mock_garmin_client, tmp_path, monkeypatch
):
    """Without output_path, the set_fit_download_dir directory is used."""
    mock_garmin_client.client.connectapi.return_value = dict(_TRACK_ONLY_COURSE)
    download_dir = tmp_path / "downloads"
    config_path = tmp_path / "fit_config.json"
    config_path.write_text(json.dumps({"download_dir": str(download_dir)}))
    monkeypatch.setenv("GARMIN_FIT_CONFIG", str(config_path))
    monkeypatch.delenv("GARMIN_FIT_DOWNLOAD_DIR", raising=False)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    result = await app_with_courses.call_tool("download_course_gpx", {"course_id": 893})

    data = json.loads(_result_text(result))
    assert data["gpx_path"] == str(download_dir / "893.gpx")
    assert (download_dir / "893.gpx").is_file()
    assert not (cwd / "courses").exists()


# --- upload_course --------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_course_rejects_non_gpx(app_with_courses, mock_garmin_client):
    """Only .gpx files are accepted; nothing is uploaded otherwise."""
    result = await app_with_courses.call_tool(
        "upload_course", {"gpx_path": "/tmp/route.tcx"}
    )

    assert "only .gpx files are allowed" in _result_text(result)
    mock_garmin_client.client.post.assert_not_called()


@pytest.mark.asyncio
async def test_upload_course_missing_file(app_with_courses, mock_garmin_client):
    """A missing file path returns an error message before hitting Garmin."""
    result = await app_with_courses.call_tool(
        "upload_course", {"gpx_path": "/tmp/nonexistent-route-12345.gpx"}
    )

    assert "GPX file not found" in _result_text(result)
    mock_garmin_client.client.post.assert_not_called()


@pytest.mark.asyncio
async def test_upload_course_rejects_unknown_activity_type(
    app_with_courses, mock_garmin_client, tmp_path
):
    """Unknown activity_type strings are rejected with supported list."""
    gpx_file = tmp_path / "valid.gpx"
    gpx_file.write_text("<gpx></gpx>")

    result = await app_with_courses.call_tool(
        "upload_course",
        {"gpx_path": str(gpx_file), "activity_type": "paragliding"},
    )

    assert "unknown activity_type 'paragliding'" in _result_text(result)
    mock_garmin_client.client.post.assert_not_called()


@pytest.mark.asyncio
async def test_upload_course_success(app_with_courses, mock_garmin_client, tmp_path):
    """Valid GPX file triggers step 1 import and step 2 save, returning metadata."""
    gpx_file = tmp_path / "test.gpx"
    gpx_file.write_text("<gpx><trk><trkseg></trkseg></trk></gpx>")

    mock_garmin_client.client.post.side_effect = [
        # Step 1: /course-service/course/import response skeleton
        {
            "courseName": "River Loop",
            "geoPoints": [
                {"latitude": 40.0, "longitude": -105.0, "elevation": 1600.0},
                {"latitude": 40.01, "longitude": -105.01, "elevation": 1610.0},
            ],
        },
        # Step 2: /course-service/course save response
        {
            "courseId": 999,
            "courseName": "River Loop",
            "distanceMeter": 1250.0,
            "elevationGainMeter": 10.0,
            "elevationLossMeter": 0.0,
            "activityTypePk": 1,
        },
    ]

    result = await app_with_courses.call_tool(
        "upload_course",
        {
            "gpx_path": str(gpx_file),
            "course_name": "River Loop",
            "activity_type": "running",
        },
    )

    data = json.loads(_result_text(result))
    assert data["status"] == "success"
    assert data["course_id"] == 999
    assert data["name"] == "River Loop"
    assert data["activity_type_id"] == 1
    assert "course/999" in data["url"]

    # Two-step flow: POST /import then POST /course.
    assert mock_garmin_client.client.post.call_count == 2
    import_call, create_call = mock_garmin_client.client.post.call_args_list
    assert import_call.args[1] == "/course-service/course/import"
    assert create_call.args[1] == "/course-service/course"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "activity_type,type_id",
    [
        ("running", 1),
        ("cycling", 2),
        ("hiking", 3),
        ("walking", 9),
        ("mountain_biking", 5),
        ("trail_running", 6),
        ("road_biking", 10),
        ("gravel_cycling", 143),
    ],
)
async def test_upload_course_posts_garmin_activity_type_id(
    app_with_courses, mock_garmin_client, tmp_path, activity_type, type_id
):
    """Each supported activity_type maps to Garmin's real activity typeId."""
    gpx_file = tmp_path / "route.gpx"
    gpx_file.write_text("<gpx><trk><trkseg></trkseg></trk></gpx>")

    mock_garmin_client.client.post.side_effect = [
        {
            "courseName": "Route",
            "geoPoints": [
                {"latitude": 40.0, "longitude": -105.0, "elevation": 1600.0},
                {"latitude": 40.01, "longitude": -105.01, "elevation": 1610.0},
            ],
        },
        {"courseId": 1000, "courseName": "Route", "activityTypePk": type_id},
    ]

    result = await app_with_courses.call_tool(
        "upload_course",
        {"gpx_path": str(gpx_file), "activity_type": activity_type},
    )

    assert json.loads(_result_text(result))["status"] == "success"
    create_call = mock_garmin_client.client.post.call_args_list[1]
    assert create_call.kwargs["json"]["activityTypePk"] == type_id


# --- delete_course --------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_course_requires_confirmation(app_with_courses, mock_garmin_client):
    """delete_course returns a preview and deletes nothing unless confirm=true."""
    mock_garmin_client.client.connectapi.return_value = {
        "courseId": 555,
        "courseName": "River Loop",
        "distanceMeter": 10250.5,
        "activityType": {"typeKey": "running"},
    }

    result = await app_with_courses.call_tool("delete_course", {"course_id": 555})

    data = json.loads(_result_text(result))
    assert data["status"] == "needs_confirmation"
    assert data["course"]["name"] == "River Loop"
    assert data["course"]["distance_m"] == 10250.5
    mock_garmin_client.client.connectapi.assert_called_once_with("/course-service/course/555")
    mock_garmin_client.client.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_course_success(app_with_courses, mock_garmin_client):
    """delete_course with confirm=true hits the right endpoint and reports success."""
    mock_garmin_client.client.connectapi.return_value = {
        "courseId": 555,
        "courseName": "River Loop",
    }

    result = await app_with_courses.call_tool(
        "delete_course", {"course_id": 555, "confirm": True}
    )

    data = json.loads(_result_text(result))
    assert data["status"] == "success"
    assert data["course_id"] == 555
    assert data["course"]["name"] == "River Loop"
    mock_garmin_client.client.delete.assert_called_once_with(
        "connectapi", "/course-service/course/555"
    )


@pytest.mark.asyncio
async def test_delete_course_error_is_caught(app_with_courses, mock_garmin_client):
    """A client error is surfaced as a clean message, not a traceback."""
    mock_garmin_client.client.connectapi.return_value = {"courseId": 555}
    mock_garmin_client.client.delete.side_effect = Exception("nope")

    result = await app_with_courses.call_tool(
        "delete_course", {"course_id": 555, "confirm": True}
    )

    assert "Error deleting course" in _result_text(result)


# --- pure helpers ---------------------------------------------------------

def test_build_course_payload_computes_distance_and_defaults():
    """Distances accumulate, missing elevation defaults to 0, bbox is derived."""
    parsed = {
        "geoPoints": [
            {"latitude": 40.0, "longitude": -105.0, "elevation": 1600.0},
            {"latitude": 40.0, "longitude": -105.0, "elevation": None},
        ],
    }

    payload = _build_course_payload(parsed, "X", 1, None)

    assert payload["courseName"] == "X"
    assert payload["activityTypePk"] == 1
    # Identical points -> zero total distance.
    assert payload["distanceMeter"] == 0.0
    # Missing elevation is backfilled to 0.0.
    assert payload["geoPoints"][1]["elevation"] == 0.0
    assert payload["boundingBox"]["lowerLeft"]["latitude"] == 40.0


def test_build_course_payload_rejects_too_few_points():
    """A GPX skeleton without geoPoints raises ValueError."""
    with pytest.raises(ValueError, match="no geoPoints"):
        _build_course_payload({}, "X", 1, None)


def test_resolve_gpx_output_path():
    """Output path resolution precedence and directory handling."""
    # 1. Custom file path
    assert _resolve_gpx_output_path(123, "/tmp/custom.gpx") == "/tmp/custom.gpx"

    # 2. Custom directory path
    assert _resolve_gpx_output_path(123, "/tmp/dir/") == "/tmp/dir/123.gpx"

    # 3. Default path
    resolved_default = _resolve_gpx_output_path(123)
    assert resolved_default.endswith("courses/123.gpx") or "123.gpx" in resolved_default


# --- create_course_from_waypoints / create_round_trip_course ---------------

def _leg(*points):
    return {"route": [[list(p) for p in points]], "routeMessages": []}


def _elevations(*values):
    def respond(*args, **kwargs):
        return [[lat, lon, values[i]] for i, (lat, lon, _) in enumerate(kwargs["json"])]
    return respond


def _post_router(legs, elevations, saved):
    """Answer the routing, elevation and save calls in the order they arrive."""
    legs = list(legs)

    def post(_service, path, **kwargs):
        if path.startswith("/course-service/course/route/v2"):
            return legs.pop(0)
        if path.startswith("/course-service/course/elevation"):
            return _elevations(*elevations)(**kwargs)
        if path == "/course-service/course":
            return saved
        raise AssertionError(f"unexpected POST {path}")

    return post


@pytest.mark.asyncio
async def test_create_course_from_waypoints_routes_each_leg(app_with_courses, mock_garmin_client):
    """Each leg is routed, legs are joined without repeating the shared point,
    elevations are filled in and the course is saved."""
    mock_garmin_client.client.post.side_effect = _post_router(
        legs=[
            _leg((54.0, 23.0), (54.001, 23.001), (54.002, 23.002)),
            _leg((54.002, 23.002), (54.003, 23.003)),
        ],
        elevations=(10.0, 11.0, 12.0, 15.0),
        saved={"courseId": 77, "courseName": "Old Town", "activityTypePk": 3},
    )

    result = await app_with_courses.call_tool(
        "create_course_from_waypoints",
        {
            "waypoints": [[54.0, 23.0], {"lat": 54.002, "lon": 23.002}, [54.003, 23.003]],
            "course_name": "Old Town",
            "activity_type": "hiking",
            "avoid_unpaved_roads": True,
        },
    )

    data = json.loads(_result_text(result))
    assert data["status"] == "success"
    assert data["course_id"] == 77

    calls = mock_garmin_client.client.post.call_args_list
    route_calls = [c for c in calls if c.args[1].startswith("/course-service/course/route/v2")]
    assert len(route_calls) == 2
    assert "routeType=POPULARITY" in route_calls[0].args[1]
    assert "courseType=HIKING" in route_calls[0].args[1]
    assert "avoidUnpavedRoads=true" in route_calls[0].args[1]
    assert route_calls[1].kwargs["json"]["startLat"] == 54.002

    payload = calls[-1].kwargs["json"]
    assert payload["courseName"] == "Old Town"
    assert payload["activityTypePk"] == 3
    assert [p["elevation"] for p in payload["geoPoints"]] == [10.0, 11.0, 12.0, 15.0]
    assert payload["elevationGainMeter"] == 5.0
    assert payload["elevationLossMeter"] == 0.0


@pytest.mark.asyncio
async def test_create_course_from_waypoints_preview_does_not_save(app_with_courses, mock_garmin_client):
    """save=false reports distance and climb and creates nothing."""
    mock_garmin_client.client.post.side_effect = _post_router(
        legs=[_leg((54.0, 23.0), (54.01, 23.0))],
        elevations=(10.0, 14.0),
        saved=None,
    )

    result = await app_with_courses.call_tool(
        "create_course_from_waypoints",
        {"waypoints": [[54.0, 23.0], [54.01, 23.0]], "course_name": "Test", "save": False},
    )

    data = json.loads(_result_text(result))
    assert data["status"] == "preview"
    assert 1100 < data["distance_m"] < 1120
    assert data["elevation_gain_m"] == 4.0
    paths = [c.args[1] for c in mock_garmin_client.client.post.call_args_list]
    assert "/course-service/course" not in paths


@pytest.mark.asyncio
async def test_create_course_from_waypoints_freehand_skips_routing(app_with_courses, mock_garmin_client):
    """freehand uses the waypoints as they are."""
    mock_garmin_client.client.post.side_effect = _post_router(
        legs=[], elevations=(1.0, 2.0, 3.0), saved={"courseId": 5}
    )

    await app_with_courses.call_tool(
        "create_course_from_waypoints",
        {
            "waypoints": [[54.0, 23.0], [54.1, 23.1], [54.2, 23.2]],
            "course_name": "Straight",
            "route_type": "freehand",
        },
    )

    paths = [c.args[1] for c in mock_garmin_client.client.post.call_args_list]
    assert not any(p.startswith("/course-service/course/route/v2") for p in paths)
    assert len(mock_garmin_client.client.post.call_args_list[-1].kwargs["json"]["geoPoints"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args,message",
    [
        ({"waypoints": [[54.0, 23.0]]}, "at least two waypoints"),
        ({"waypoints": [[54.0, 23.0]] * 26}, "at most 25 waypoints"),
        ({"waypoints": [[95.0, 23.0], [54.0, 23.0]]}, "out of range"),
        ({"waypoints": [[54.0, 23.0], [54.1, 23.1]], "route_type": "scenic"}, "unknown route_type"),
        ({"waypoints": [[54.0, 23.0], [54.1, 23.1]], "route_type": "free_flow"}, "only available for cycling"),
        ({"waypoints": [[54.0, 23.0], [54.1, 23.1]], "activity_type": "swimming"}, "unknown activity_type"),
    ],
)
async def test_create_course_from_waypoints_rejects_bad_input(
    app_with_courses, mock_garmin_client, args, message
):
    result = await app_with_courses.call_tool(
        "create_course_from_waypoints", {"course_name": "X", **args}
    )

    assert message in _result_text(result)
    mock_garmin_client.client.post.assert_not_called()


@pytest.mark.asyncio
async def test_create_course_from_waypoints_no_route(app_with_courses, mock_garmin_client):
    mock_garmin_client.client.post.side_effect = _post_router(
        legs=[{"route": []}], elevations=(), saved=None
    )

    result = await app_with_courses.call_tool(
        "create_course_from_waypoints",
        {"waypoints": [[54.0, 23.0], [54.1, 23.1]], "course_name": "X"},
    )

    assert "Garmin found no route" in _result_text(result)


_LOOPS = {
    "roundTripRouteCourses": [
        {
            "distanceMeter": 4501.3,
            "elevationGainMeter": 84.6,
            "elevationLossMeter": 84.0,
            "geoPoints": [
                {"latitude": 54.0, "longitude": 23.0, "elevation": 30.0, "distance": 0.0},
                {"latitude": 54.01, "longitude": 23.01, "elevation": 40.0, "distance": 1.0},
                {"latitude": 54.0, "longitude": 23.0, "elevation": 30.0, "distance": 2.0},
            ],
        },
        {
            "distanceMeter": 4687.0,
            "elevationGainMeter": 120.6,
            "elevationLossMeter": 120.0,
            "geoPoints": [
                {"latitude": 54.0, "longitude": 23.0, "elevation": 30.0},
                {"latitude": 53.99, "longitude": 22.99, "elevation": 25.0},
                {"latitude": 54.0, "longitude": 23.0, "elevation": 30.0},
            ],
        },
    ]
}


@pytest.mark.asyncio
async def test_create_round_trip_course_preview(app_with_courses, mock_garmin_client):
    """save=false lists Garmin's loops with the resolved direction."""
    mock_garmin_client.client.connectapi.return_value = _LOOPS

    result = await app_with_courses.call_tool(
        "create_round_trip_course",
        {
            "start_latitude": 54.0,
            "start_longitude": 23.0,
            "distance_km": 5,
            "course_name": "Loop",
            "direction": "tailwind",
            "activity_type": "road_biking",
            "route_type": "free_flow",
            "save": False,
        },
    )

    data = json.loads(_result_text(result))
    assert data["status"] == "preview"
    assert data["direction"] == "tailwind"
    assert [a["distance_m"] for a in data["alternatives"]] == [4501.3, 4687.0]
    path = mock_garmin_client.client.connectapi.call_args.args[0]
    assert path.startswith("/course-service/course/roundTripRoutes/multiple?")
    for part in ("distance=5000", "direction=TAILWIND", "courseType=ROAD_CYCLING", "routeType=FREE_FLOW"):
        assert part in path
    mock_garmin_client.client.post.assert_not_called()


@pytest.mark.asyncio
async def test_create_round_trip_course_saves_chosen_alternative(app_with_courses, mock_garmin_client):
    mock_garmin_client.client.connectapi.return_value = _LOOPS
    mock_garmin_client.client.post.return_value = {"courseId": 9, "courseName": "Loop"}

    result = await app_with_courses.call_tool(
        "create_round_trip_course",
        {
            "start_latitude": 54.0,
            "start_longitude": 23.0,
            "distance_km": 5,
            "course_name": "Loop",
            "direction": "random",
            "alternative": 2,
        },
    )

    data = json.loads(_result_text(result))
    assert data["course_id"] == 9
    assert data["direction"] in ("north", "south", "east", "west")
    payload = mock_garmin_client.client.post.call_args.kwargs["json"]
    assert [p["latitude"] for p in payload["geoPoints"]] == [54.0, 53.99, 54.0]
    assert payload["elevationGainMeter"] == 120.6
    assert payload["elevationLossMeter"] == 120.0
    assert payload["activityTypePk"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args,message",
    [
        ({"route_type": "most_direct"}, "for a round trip"),
        ({"direction": "up"}, "unknown direction"),
        ({"distance_km": 0}, "distance_km must be between"),
        ({"route_type": "free_flow"}, "only available for cycling"),
        ({"start_latitude": 100}, "out of range"),
    ],
)
async def test_create_round_trip_course_rejects_bad_input(
    app_with_courses, mock_garmin_client, args, message
):
    base = {"start_latitude": 54.0, "start_longitude": 23.0, "distance_km": 5, "course_name": "X"}
    result = await app_with_courses.call_tool("create_round_trip_course", {**base, **args})

    assert message in _result_text(result)
    mock_garmin_client.client.connectapi.assert_not_called()


@pytest.mark.asyncio
async def test_create_round_trip_course_alternative_out_of_range(app_with_courses, mock_garmin_client):
    mock_garmin_client.client.connectapi.return_value = _LOOPS

    result = await app_with_courses.call_tool(
        "create_round_trip_course",
        {"start_latitude": 54.0, "start_longitude": 23.0, "distance_km": 5, "course_name": "X", "alternative": 3},
    )

    assert "alternative must be between 1 and 2" in _result_text(result)
    mock_garmin_client.client.post.assert_not_called()
