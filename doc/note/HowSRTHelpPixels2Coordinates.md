# To help myself with understanding RQ2 (converting pixels 2 real-world coordinates) I made some notes for easier comprehension

---

The object here is that how can we use SRT telemetry data recorded by DJI drones to reconstruct the actual geographic coordinates (ground footprint) of each frame of the camera's field of view (FOV)?

A example record of SRT telemetry data is as follows:

```cmd
1
00:00:00,000 --> 00:00:00,198
<font size="28">FrameCnt: 1, DiffTime: 198ms
2025-04-09 09:15:02.655
[iso: 170] [shutter: 1/99.77] [fnum: 4.4] [ev: 0] [color_md : default] [ae_meter_md: 1] [focal_len: 284.80] [dzoom_ratio: 1.77], [latitude: 53.388452] [longitude: 5.361783] [rel_alt: 55.774 abs_alt: 98.364] [gb_yaw: -83.6 gb_pitch: -9.1 gb_roll: 0.0] </font>

2
00:00:00,198 --> 00:00:00,396
<font size="28">FrameCnt: 2, DiffTime: 198ms
2025-04-09 09:15:02.837
[iso: 170] [shutter: 1/99.77] [fnum: 4.4] [ev: 0] [color_md : default] [ae_meter_md: 1] [focal_len: 284.80] [dzoom_ratio: 1.77], [latitude: 53.388452] [longitude: 5.361783] [rel_alt: 55.759 abs_alt: 98.349] [gb_yaw: -83.6 gb_pitch: -9.1 gb_roll: 0.0] </font>
```

First line is the frame index which starts at one. Second line is the timeline from start to end with millisecond level precision. Third line is the html label + frame indexes and the time interval. Fourth line is the UTC timestamp with millisecond level precision.

The fifth line includes all telemetry parameters:

| Field       | Example   | Unit |                                                              |
| ----------- | --------- | ---- | ------------------------------------------------------------ |
| latitude    | 53.388452 |      |                                                              |
| longitude   | 5.361783  |      |                                                              |
| rel_alt     | 55.759    | m    | Relative altitude is used to estimate the camera's altitude (H) above the ground. It is 0 at takeoff and positive during ascent. |
| abs_alt     | 98.349    | m    | Absolute altitude is the altitude relative to the WGS84 ellipsoid. Absolute altitude = rel_alt + altitude of the takeoff point. |
|             |           |      |                                                              |
|             |           |      | The attitude of a drone can be described using a right-handed coordinate system: The Z-axis points upwards, representing the direction of lift. The Y-axis represents the longitudinal (front-to-back) direction. The X-axis represents the lateral (left-to-right) direction. |
| gb_yaw      | -83.6     | °    | Yaw is the angle at which a drone rotates around the Z-axis, which determines the drone's heading. **North is defined as 0°, and clockwise is positive.** |
| gb_pitch    | -9.1      | °    | Pitch is the angle at which a drone rotates around the Y-axis, which determines the drone's forward and backward tilt. **0° = horizontal, -90° = directly below (nadir)** |
| gb_roll     | 0.0       | °    | Roll is the angle at which a drone rotates around the X-axis, which determines the drone's left and right tilt. **Should be 0 all the time.** |
|             |           |      |                                                              |
| focal_len   | 284.80    |      |                                                              |
| dzoom_ratio | 1.77      | null | Digital zoom.                                                |
|             |           |      |                                                              |



3840x 2160 pixel per frame (16:9)

- 1/2" CMOS，有效像素 1200 万

- 视角：15°
  等效焦距：162 毫米
  光圈：f/4.4
  对焦点：3 米至无穷远
- 4K：3840×2160@30fps
- 俯仰：-135° 至 100°
  横滚：-45° 至 45°
  平移：-27° 至 27°

```
长焦摄像头传感器（1/2" CMOS，本项目实际使用）：
  原生传感器：6.4mm × 4.8mm（4:3）← 标准 1/2" 规格
  视频裁切为 16:9：有效高度 = 6.4 × (9/16) = 3.6mm
  传感器宽度：SENSOR_W = 6.4mm
  传感器有效高度：SENSOR_H = 3.6mm（16:9 裁切后）
  传感器对角线：SENSOR_DIAG = √(6.4² + 3.6²) ≈ 7.34mm
  35mm 全画幅对角线：FF_DIAG = √(36² + 24²) ≈ 43.27mm
  裁切系数：43.27 / 8.0 ≈ 5.41×
  物理焦距：162 / 5.41 ≈ 29.9mm（实际光学焦距）
```



Some prompts for schematic figures:

```
 ---
  Sub-figure 1: Sensor Size Comparison & Crop Factor

  A clean technical diagram on a dark navy background (#0d1b2a) showing three
  rectangles side by side to compare sensor sizes to scale:

  Left rectangle: Full-frame 35mm sensor, labeled "Full-Frame 35mm",
  dimensions 36×24mm, drawn large, filled with a subtle blue gradient,
  white border. Show diagonal line with label "43.27mm diagonal".

  Middle rectangle: 4/3" sensor (Four Thirds), labeled "Wide Camera 4/3\"",
  dimensions 17.3×13mm, proportionally smaller than full-frame, filled with
  green gradient. Show diagonal "21.64mm". Below it show the 16:9 crop region
  as a dashed inner rectangle (17.3×9.74mm) with label "16:9 video crop".

  Right rectangle: 1/2" sensor, labeled "Tele Camera 1/2\" (THIS PROJECT)",
  dimensions 6.4×4.8mm, much smaller, filled with orange/amber gradient.
  Show diagonal "8.0mm". Below it show the 16:9 crop region as a dashed inner
  rectangle (6.4×3.6mm) with label "16:9 video crop, SENSOR_DIAG=7.34mm".

  Below each sensor, show the crop factor: "1.0×", "2.0×", "5.41×" in large
  bold text. Add a scale bar at the bottom. Use monospace font for all labels.
  Style: scientific illustration, flat design, no shadows.

  ---
  Sub-figure 2: Equivalent Focal Length vs Physical Focal Length

  A two-panel side-by-side diagram on dark background explaining the difference
  between equivalent and physical focal length.

  Left panel titled "35mm Equivalent Focal Length (SRT focal_len)":
  Draw a full-frame sensor rectangle (36×24mm) with a lens cone projecting
  from it. Label the cone angle as "FOV = 15°". Mark the focal length as
  "162mm" along the optical axis. Add label "What the SRT reports".

  Right panel titled "Physical Focal Length (actual glass)":
  Draw the 1/2" sensor rectangle (6.4×4.8mm) with the same lens cone showing
  identical FOV angle of 15°. Mark the physical focal length as "≈29.9mm"
  along the optical axis. Add label "Actual lens length".

  In the center between panels, show the formula:
    Physical FL = Equivalent FL ÷ Crop Factor
    29.9mm = 162mm ÷ 5.41

  Add a note: "Same field of view angle (15°), different physical focal length
  because sensor size differs." Use cyan (#00ffcc) for formula text, white for
  labels, dark navy background. Clean technical illustration style.

  ---
  Sub-figure 3: Digital Zoom Removal

  A horizontal flow diagram on dark background showing how digital zoom is
  removed from the SRT focal_len value.

  Step 1 (leftmost box):
    Label "SRT Raw Value"
    Large text: focal_len = 284.80mm
    Subtitle: "35mm equivalent, includes digital zoom"
    Box color: dark red border

  Step 2 (middle, with ÷ symbol between):
    Label "Digital Zoom Factor"
    Large text: dzoom_ratio = 1.77×
    Subtitle: "Sensor center crop, narrows FOV"
    Show a small diagram: a rectangle with a smaller dashed rectangle inside
    it, arrow pointing inward, labeled "1.77× crop"
    Box color: orange border

  Step 3 (rightmost box, with = symbol):
    Label "Optical Equivalent Focal Length"
    Large text: focalOpt = 160.9mm ≈ 162mm
    Subtitle: "Pure optical FOV, no digital zoom"
    Box color: green border (#00c896)

  Below the flow, add a note in small text:
    "focalOpt = focal_len / dzoom_ratio = 284.80 / 1.77 ≈ 160.9mm"
    "This matches the tele camera's optical spec: 162mm equivalent"

  Style: flat design, dark navy background, monospace font, connected by
  thick arrows with labels.

  ---
  Sub-figure 4: Camera Coordinate System & Ray Direction

  A 3D perspective diagram showing the camera coordinate system and the four
  corner rays of the image sensor.

  Draw a drone viewed from slightly above and to the side, hovering in the air.
  The drone body is a simple gray rectangle with four arms and propellers.

  From the camera gimbal (bottom of drone), draw a coordinate system:
    - Y axis (cyan): pointing forward (camera front direction)
    - Z axis (green): pointing upward
    - X axis (red): pointing right

  Show the camera sensor as a small rectangle at the gimbal, tilted at
  gb_pitch = -9.1° from horizontal (nearly horizontal, slightly downward).

  From the four corners of the sensor, draw four thin colored lines (rays)
  extending outward:
    - Top-left corner ray: labeled "[-halfW, +halfH]"
    - Top-right corner ray: labeled "[+halfW, +halfH]"
    - Bottom-right corner ray: labeled "[+halfW, -halfH]"
    - Bottom-left corner ray: labeled "[-halfW, -halfH]"

  Label the center ray as "optical axis, pitch = -9.1°".

  Show the half-angle annotations:
    halfW = 6.7° (horizontal half-angle)
    halfH = 3.8° (vertical half-angle)

  Add a small inset showing the sensor rectangle with the four corners labeled
  and the half-angles marked. Dark background, technical illustration style.

  ---
  Sub-figure 5: Yaw Rotation — Camera to World Coordinates

  A top-down (bird's eye view) diagram showing the yaw rotation transformation.

  Draw a compass rose in the background with N/S/E/W labels. North is up.

  In the center, draw a small drone icon (top-down view, simple cross shape
  with four arms). The drone is facing a direction indicated by gb_yaw = -83.6°
  (nearly due West, slightly North).

  Show two coordinate systems overlaid:
    1. Camera coordinate system (before yaw rotation):
       - cy axis (dashed cyan): pointing in camera-forward direction
       - cx axis (dashed red): pointing camera-right
       Label: "Camera Frame"

    2. World coordinate system (after yaw rotation):
       - Y axis (solid green): pointing North
       - X axis (solid orange): pointing East
       Label: "World Frame (ENU)"

  Draw the rotation arc from camera-forward to world-north, labeled
  "yaw = -83.6° (clockwise from North)".

  Show the transformation formula in a box:
    wx = cx·cos(yaw) + cy·sin(yaw)  → East
    wy = -cx·sin(yaw) + cy·cos(yaw) → North

  Add a note: "DJI convention: 0° = North, clockwise positive"
  Dark navy background, clean flat design.

  ---
  Sub-figure 6: Ray-Ground Intersection Geometry

  A side-view (cross-section) technical diagram showing how a camera ray
  intersects the ground plane.

  Draw a horizontal ground plane (green line) at the bottom, labeled "Ground
  plane (Z = 0)".

  Above it, draw the drone as a small icon at height H = 55.774m, with a
  vertical dashed line down to the ground labeled "rel_alt = 55.774m".

  From the drone camera, draw the optical axis ray going downward at a shallow
  angle (gb_pitch = -9.1° from horizontal, so nearly horizontal).

  Show the ray hitting the ground far in the distance. Mark the intersection
  point with a red dot labeled "Ground footprint point".

  Draw the geometry annotations:
    - Angle α = 9.1° (depression angle from horizontal) at the drone
    - Vertical component: wz (must be < 0, pointing downward)
    - Horizontal component: wx (East), wy (North)
    - Parameter t = H / (-wz), shown as the ray length scale factor
    - east_m = wx × t (horizontal East offset in meters)
    - north_m = wy × t (horizontal North offset in meters)

  Add a right-angle marker at the ground intersection showing the
  perpendicular relationship.

  Show a small inset formula box:
    "t = H / (-wz)"
    "east_m = wx × t"
    "north_m = wy × t"
    "Condition: wz < 0 (ray must point downward)"

  Note: at pitch = -9.1°, the ground intersection is very far away (~343m
  from directly below). Show this distance on the diagram.

  Style: engineering cross-section diagram, white lines on dark navy,
  dimension arrows with labels.

  ---
  Sub-figure 7: Meter Offset to Geographic Coordinates

  A map-view diagram showing the conversion from meter offsets to
  latitude/longitude coordinates.

  Draw a curved Earth surface segment to suggest the spherical Earth.
  Show a small area of the Wadden Sea / Terschelling island (Netherlands)
  as a simple coastline sketch.

  Mark the drone GPS position as a blue dot labeled:
    "Drone position
     lat = 53.388452°N
     lon = 5.361783°E"

  From this point, draw two perpendicular arrows:
    - North arrow (green): labeled "north_m offset"
    - East arrow (orange): labeled "east_m offset"

  Mark the resulting ground point as a red dot.

  Show the conversion formula in a clean box:
    lat_ground = lat + (north_m / R) × (180/π)
    lon_ground = lon + (east_m / R) × (180/π) / cos(lat × π/180)
    where R = 6,378,137m (Earth radius)

  Add a small inset showing the Earth as a sphere with the local tangent
  plane approximation highlighted, explaining why cos(lat) appears in the
  longitude formula (longitude degrees are shorter at higher latitudes).

  Add annotation: "At lat=53.4°N, 1° longitude ≈ 66.8km (not 111.3km)"

  Style: cartographic illustration, dark background, cyan grid lines for
  the local coordinate system.

  ---
  Sub-figure 8: Complete Ground Footprint Quadrilateral

  A combined top-down map view showing the complete ground footprint polygon
  for one video frame.

  Background: satellite-style imagery of flat green/brown terrain
  (Terschelling island, Netherlands). Slightly stylized, not photorealistic.

  Draw the drone as a small icon at the center-top of the image,
  facing West (yaw = -83.6°). Add a small rotation indicator.

  From the drone, draw four thin dashed lines going to the four ground
  corner points (the four rays hitting the ground).

  Connect the four ground points with a solid cyan polygon (#00ffcc),
  semi-transparent fill (opacity 0.25). Label the corners:
    - Top-left: "[-halfW, +halfH] → far-left"
    - Top-right: "[+halfW, +halfH] → far-right"
    - Bottom-right: "[+halfW, -halfH] → near-right"
    - Bottom-left: "[-halfW, -halfH] → near-left"

  Note: because pitch = -9.1° (nearly horizontal), the footprint is very
  elongated — much longer in the forward direction than wide. Show this
  clearly: the footprint should be a very long thin trapezoid extending
  far in the westward direction.

  Add dimension annotations:
    "~520m depth (forward)"
    "~60m width"
    "Drone altitude: 55.8m"

  Add a north arrow and scale bar.
  Style: GIS map overlay style, dark basemap, glowing cyan polygon.

  ---
  Sub-figure 9: Full Pipeline Overview (Summary Diagram)

  A vertical flowchart summarizing the entire projection pipeline,
  designed as a poster-style infographic on dark navy background.

  Use 6 connected stages, each as a rounded rectangle with icon:

  Stage 1 — "SRT Telemetry Input" (icon: document/file)
    Color: dark blue border
    Content: focal_len, dzoom_ratio, gb_yaw, gb_pitch, rel_alt, lat, lon

  Stage 2 — "Sensor Parameters" (icon: camera chip)
    Color: purple border
    Content: 1/2" sensor → 6.4×3.6mm (16:9), SENSOR_DIAG=7.34mm

  Stage 3 — "Focal Length Conversion" (icon: lens)
    Color: teal border
    Content: focalOpt = 284.80/1.77 = 160.9mm
             fReal = 160.9 × (7.34/43.27) = 27.3mm
             halfW = 6.7°, halfH = 3.8°

  Stage 4 — "Camera Ray Directions" (icon: 3D axes)
    Color: cyan border
    Content: 4 corner rays [cx, cy, cz] in camera frame
             Yaw rotation → world frame [wx, wy, wz]

  Stage 5 — "Ray-Ground Intersection" (icon: crosshair)
    Color: green border
    Content: t = H/(-wz), east_m = wx×t, north_m = wy×t

  Stage 6 — "Geographic Coordinates" (icon: map pin)
    Color: orange border
    Content: (lat_ground, lon_ground) × 4 corners
             → Ground footprint polygon

  Connect stages with thick downward arrows. On the right side of the
  diagram, add a vertical label: "INTRINSICS → EXTRINSICS → GEOMETRY".

  At the bottom, add a small thumbnail of the resulting footprint polygon
  on a map. Style: modern infographic, monospace font, glowing borders.

  ---
  Sub-figure 10: FOOTPRINT NULL Failure Cases

  A three-panel diagram showing the three conditions that cause projection
  failure (FOOTPRINT NULL), each panel showing a side-view geometry sketch.

  Panel 1 — "Case 1: rel_alt ≤ 0"
    Draw a drone on the ground (altitude = 0).
    Show rays from camera going in all directions but no ground intersection
    possible. Red X mark.
    Label: "rel_alt = 0m — drone on ground or GPS error"
    Border: red

  Panel 2 — "Case 2: gb_pitch ≥ 0° (camera pointing up or horizontal)"
    Draw a drone in the air with camera pointing upward (pitch = +10°).
    Show rays going upward, away from ground. Red X mark.
    Label: "gb_pitch = +10° — camera faces sky, rays never hit ground"
    Border: red

  Panel 3 — "Case 3: Corner ray wz ≥ 0 (extreme shallow angle)"
    Draw a drone in the air with camera nearly horizontal (pitch = -2°).
    Show that the center ray barely hits the ground far away, but the
    top-corner ray (halfH upward) actually points slightly above horizontal
    and goes to infinity. Red X on that corner ray.
    Label: "gb_pitch = -2° — upper corner ray escapes to horizon"
    Show the math: cz = -sin(2°) + tan(halfH)·cos(2°) ≥ 0
    Border: red

  Below all three panels, add a green panel showing the SUCCESS condition:
    "FOOTPRINT OK: rel_alt > 0 AND gb_pitch < 0 AND all corner wz < 0"
    Show a drone at 55m altitude with pitch = -9.1°, all four rays hitting
    the ground, green checkmark.

  Style: technical diagram, dark background, red for failure cases,
  green for success.

```



