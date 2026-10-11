import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import folium
from streamlit_folium import st_folium


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Traffic Congestion Predictor",
    page_icon="🚦",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --ink: #eef3f1;
    --muted: #87938f;
    --panel: #121917;
    --panel-soft: #18211e;
    --line: rgba(211, 234, 222, 0.11);
    --lime: #c7f36b;
    --cyan: #79e5d0;
    --orange: #ffb86b;
    --red: #ff7272;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 82% 0%, rgba(90, 158, 133, .12), transparent 28rem),
        linear-gradient(135deg, #0b100f 0%, #0d1412 52%, #111b17 100%);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stAppViewContainer"] > .main {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] {
    background: #0a0f0e;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.35rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: var(--ink) !important;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: .73rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--muted) !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="checkbox"] {
    background: var(--panel);
    border-color: var(--line);
    border-radius: 10px;
}

.main-title {
    margin: .4rem 0 .15rem;
    font-size: clamp(2rem, 4vw, 3.65rem);
    line-height: 1.02;
    letter-spacing: -.07em;
    font-weight: 700;
    color: var(--ink);
    max-width: 760px;
}

.sub-title {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.6;
    margin: .8rem 0 1.6rem;
    max-width: 620px;
}

.eyebrow {
    color: var(--lime);
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    letter-spacing: .16em;
    text-transform: uppercase;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: .75rem 0 1rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 2.3rem;
}

.live-pill {
    display: inline-flex;
    align-items: center;
    gap: .55rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: .7rem;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--lime);
    box-shadow: 0 0 0 5px rgba(199, 243, 107, .1);
    animation: pulse 1.8s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: .45; transform: scale(.78); }
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(25, 36, 32, .94), rgba(17, 25, 22, .94));
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.15rem 1.2rem;
    box-shadow: 0 14px 35px rgba(0, 0, 0, .18);
}

div[data-testid="stMetric"] label {
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    letter-spacing: .08em;
    text-transform: uppercase;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--ink);
    font-size: 1.65rem;
}

.section-kicker {
    color: var(--cyan);
    font-family: 'DM Mono', monospace;
    font-size: .7rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin: 2rem 0 .75rem;
}

.high-box {
    background: rgba(255, 114, 114, .1);
    border: 1px solid rgba(255, 114, 114, .35);
    border-left: 4px solid var(--red);
    color: var(--ink);
    padding: 1rem 1.2rem;
    border-radius: 12px;
}

.medium-box {
    background: rgba(255, 184, 107, .1);
    border: 1px solid rgba(255, 184, 107, .35);
    border-left: 4px solid var(--orange);
    color: var(--ink);
    padding: 1rem 1.2rem;
    border-radius: 12px;
}

.low-box {
    background: rgba(121, 229, 208, .1);
    border: 1px solid rgba(121, 229, 208, .35);
    border-left: 4px solid var(--cyan);
    color: var(--ink);
    padding: 1rem 1.2rem;
    border-radius: 12px;
}

.info-box {
    background: var(--panel);
    border: 1px solid var(--line);
    color: var(--ink);
    padding: 1.1rem 1.25rem;
    border-radius: 12px;
}

.empty-state {
    min-height: 330px;
    display: grid;
    place-items: center;
    text-align: center;
    padding: 2rem;
    border: 1px dashed rgba(121, 229, 208, .25);
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(22, 34, 29, .8), rgba(14, 21, 18, .8));
}

.empty-state-icon {
    font-size: 3.5rem;
    filter: drop-shadow(0 0 18px rgba(199, 243, 107, .22));
}

.empty-state h3 {
    color: var(--ink);
    margin: .7rem 0 .35rem;
}

.empty-state p {
    color: var(--muted);
    max-width: 430px;
    margin: 0 auto;
}

[data-testid="stButton"] button {
    background: var(--lime);
    border: 0;
    border-radius: 10px;
    color: #132016;
    font-weight: 700;
    min-height: 2.75rem;
    transition: transform .18s ease, box-shadow .18s ease;
}

[data-testid="stButton"] button:hover {
    background: #d9ff83;
    box-shadow: 0 8px 24px rgba(199, 243, 107, .18);
    transform: translateY(-2px);
}

.stAlert {
    background: var(--panel);
    border: 1px solid var(--line);
    color: var(--ink);
    border-radius: 12px;
}

iframe[title="streamlit_folium.st_folium"] {
    border: 1px solid var(--line);
    border-radius: 15px;
    overflow: hidden;
}

@media (max-width: 760px) {
    .topbar { margin-bottom: 1.5rem; }
    .main-title { font-size: 2.35rem; }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATASET
# =========================================================

DATA_FILE = (
    Path(__file__).resolve().parent
    / "project_traffic_dataset-v4.csv"
)

try:
    df = pd.read_csv(DATA_FILE)

except FileNotFoundError:

    st.error(
        f"❌ {DATA_FILE.name} not found in the app folder.\n\n"
        "CSV file ko app.py ke same folder mein rakho."
    )

    st.stop()

if df.empty:
    st.error(
        f"{DATA_FILE.name} is empty. Add traffic records before starting the app."
    )
    st.stop()


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "route_name",
    "source_city",
    "destination_city",
    "connected_routes",
    "intermediate_traffic_spots",
    "time_slot",
    "is_festival_holiday",
    "is_weekend",
    "current_vehicle_count",
    "avg_historical_vehicle_count",
    "avg_speed_kmh",
    "potholes_count",
    "construction_active",
    "accident_reported",
    "primary_congestion_reason",
    "congestion_level"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Dataset mein ye columns missing hain:\n\n{missing_columns}"
    )

    st.stop()

if df["congestion_level"].dropna().nunique() < 2:
    st.error(
        "The dataset needs at least two congestion levels "
        "(for example, Low and High) to train the predictor."
    )
    st.stop()


# =========================================================
# CLEAN DATA
# =========================================================

numeric_columns = [
    "is_festival_holiday",
    "is_weekend",
    "current_vehicle_count",
    "avg_historical_vehicle_count",
    "avg_speed_kmh",
    "potholes_count",
    "construction_active",
    "accident_reported"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    df[col] = df[col].fillna(
        df[col].median()
    )


df["route_name"] = (
    df["route_name"]
    .astype(str)
    .str.strip()
)

df["time_slot"] = (
    df["time_slot"]
    .astype(str)
    .str.strip()
)

df["congestion_level"] = (
    df["congestion_level"]
    .astype(str)
    .str.strip()
    .str.title()
)


# =========================================================
# MODEL FEATURES
# =========================================================

MODEL_FEATURES = [

    "route_name",
    "time_slot",

    "is_festival_holiday",
    "is_weekend",

    "current_vehicle_count",
    "avg_historical_vehicle_count",

    "avg_speed_kmh",

    "potholes_count",
    "construction_active",
    "accident_reported"
]


X = df[MODEL_FEATURES].copy()

y = df["congestion_level"].copy()


# =========================================================
# ENCODING
# =========================================================

X_encoded = pd.get_dummies(
    X,
    columns=[
        "route_name",
        "time_slot"
    ],
    dtype=int
)

model_columns = X_encoded.columns.tolist()


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X_encoded,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# =========================================================
# RANDOM FOREST
# =========================================================

model = RandomForestClassifier(

    n_estimators=150,

    random_state=42,

    class_weight="balanced"
)


model.fit(
    X_train,
    y_train
)


# =========================================================
# MODEL ACCURACY
# =========================================================

test_prediction = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    test_prediction
)


# =========================================================
# ROUTE COORDINATES
# Approximate visualization coordinates
# =========================================================

route_coordinates = {

    "Delhi to Noida": {

        "start": (28.6139, 77.2090),

        "end": (28.5355, 77.3910),

        "spots": [

            (
                "Mayur Vihar Pocket 1",
                28.6100,
                77.2950
            ),

            (
                "DND Toll Plaza",
                28.5820,
                77.3200
            ),

            (
                "Sector 16 Noida",
                28.5780,
                77.3205
            )
        ]
    },


    "Noida to Greater Noida": {

        "start": (28.5355, 77.3910),

        "end": (28.4744, 77.5040),

        "spots": [

            (
                "Sector 37 Underpass",
                28.5700,
                77.3500
            ),

            (
                "Knowledge Park",
                28.4590,
                77.4830
            ),

            (
                "Pari Chowk",
                28.4595,
                77.5000
            )
        ]
    },


    "Delhi to Gurugram": {

        "start": (28.6139, 77.2090),

        "end": (28.4595, 77.0266),

        "spots": [

            (
                "Dhaula Kuan",
                28.5921,
                77.1610
            ),

            (
                "Sirhaul Toll",
                28.4910,
                77.0750
            ),

            (
                "IFFCO Chowk",
                28.4800,
                77.0950
            )
        ]
    },


    "Gurugram to Faridabad": {

        "start": (28.4595, 77.0266),

        "end": (28.4089, 77.3178),

        "spots": [

            (
                "Gwal Pahari",
                28.4210,
                77.1420
            ),

            (
                "Pali Chowk",
                28.4030,
                77.2400
            ),

            (
                "Sainik Colony",
                28.4140,
                77.2910
            )
        ]
    },


    "Modinagar to Ghaziabad": {

        "start": (28.9845, 77.5880),

        "end": (28.6692, 77.4538),

        "spots": [

            (
                "Muradnagar Canal",
                28.7800,
                77.4980
            ),

            (
                "Duhai Flyover",
                28.7130,
                77.4730
            ),

            (
                "Raj Nagar Extension",
                28.7000,
                77.4100
            )
        ]
    },


    "Meerut to Delhi": {

        "start": (28.9845, 77.7064),

        "end": (28.6139, 77.2090),

        "spots": [

            (
                "Partapur Toll",
                28.9360,
                77.6550
            ),

            (
                "Bhojpur Interchange",
                28.8500,
                77.5800
            ),

            (
                "Dasna Flyover",
                28.6810,
                77.5170
            )
        ]
    },


    "Mathura to Delhi": {

        "start": (27.4924, 77.6737),

        "end": (28.6139, 77.2090),

        "spots": [

            (
                "Kosi Kalan Toll",
                27.7980,
                77.4640
            ),

            (
                "Palwal Bypass",
                28.1487,
                77.3320
            ),

            (
                "Badarpur Border",
                28.5020,
                77.3020
            )
        ]
    },


    "Delhi to Agra": {

        "start": (28.6139, 77.2090),

        "end": (27.1767, 78.0081),

        "spots": [

            (
                "Jewar Toll Plaza",
                28.1310,
                77.5550
            ),

            (
                "Mathura Cut",
                27.6500,
                77.6500
            ),

            (
                "Agra Inner Ring Road",
                27.2100,
                77.9500
            )
        ]
    },


    "Agra to Aligarh": {

        "start": (27.1767, 78.0081),

        "end": (27.8974, 78.0880),

        "spots": [

            (
                "Khandauli Bypass",
                27.2900,
                78.0200
            ),

            (
                "Hathras Junction",
                27.5900,
                78.0600
            ),

            (
                "Sasni Border",
                27.6900,
                78.0700
            )
        ]
    },


    "Delhi to Bulandshahr": {

        "start": (28.6139, 77.2090),

        "end": (28.4069, 77.8498),

        "spots": [

            (
                "Lal Kuan Ghaziabad",
                28.6540,
                77.4530
            ),

            (
                "Dadri Bypass",
                28.5520,
                77.5560
            ),

            (
                "Sikandrabad Chowk",
                28.4520,
                77.6940
            )
        ]
    }
}


# =========================================================
# BYPASS ROUTES
# =========================================================

bypass_routes = {

    "Delhi to Noida":
        "Mayur Vihar Link Road / Kalindi Kunj Bridge",

    "Noida to Greater Noida":
        "Sector 137 Inner Arterial Road",

    "Delhi to Gurugram":
        "NH-48 Alternate Service Road",

    "Gurugram to Faridabad":
        "Gurugram-Faridabad Highway Alternate",

    "Modinagar to Ghaziabad":
        "Delhi-Meerut Expressway Bypass",

    "Meerut to Delhi":
        "Delhi-Meerut Road via Muradnagar",

    "Mathura to Delhi":
        "Yamuna Expressway / NH-19 Bypass",

    "Delhi to Agra":
        "NH-19 Parallel Connector via Palwal",

    "Agra to Aligarh":
        "Sasni Bypass Road",

    "Delhi to Bulandshahr":
        "GT Road Bypass via Dadri"
}


# =========================================================
# FIND TRAFFIC SPOT
# =========================================================

def get_traffic_spot(
    route_name,
    prediction,
    selected_row
):

    route_info = route_coordinates.get(
        route_name
    )

    if not route_info:
        return None

    spots = route_info["spots"]

    reason = str(
        selected_row.get(
            "primary_congestion_reason",
            ""
        )
    ).lower()


    # Accident
    if "accident" in reason:

        return spots[0]


    # Construction
    if "construction" in reason:

        return spots[1]


    # Pothole / road condition
    if "pothole" in reason:

        return spots[-1]


    # High traffic
    if prediction == "High":

        return spots[
            len(spots) // 2
        ]


    # Medium traffic
    if prediction == "Medium":

        return spots[0]


    return None


# =========================================================
# CREATE MAP
# =========================================================

def create_traffic_map(
    route_name,
    prediction,
    selected_row
):

    route_info = route_coordinates.get(
        route_name
    )

    if route_info is None:
        return None


    start = route_info["start"]

    end = route_info["end"]

    spots = route_info["spots"]


    # =====================================================
    # MAP
    # =====================================================

    traffic_map = folium.Map(

        location=[
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2
        ],

        zoom_start=10,

        tiles="OpenStreetMap",

        control_scale=True
    )


    # =====================================================
    # MAIN ROUTE
    # =====================================================

    folium.PolyLine(

        locations=[
            start,
            end
        ],

        color="#444444",

        weight=7,

        opacity=0.85,

        tooltip=f"Main Route: {route_name}"

    ).add_to(
        traffic_map
    )


    # =====================================================
    # START MARKER
    # =====================================================

    folium.Marker(

        location=start,

        tooltip="START",

        popup=f"""
        <b>START</b><br>
        {selected_row['source_city']}
        """,

        icon=folium.Icon(
            color="blue",
            icon="play"
        )

    ).add_to(
        traffic_map
    )


    # =====================================================
    # DESTINATION MARKER
    # =====================================================

    folium.Marker(

        location=end,

        tooltip="DESTINATION",

        popup=f"""
        <b>DESTINATION</b><br>
        {selected_row['destination_city']}
        """,

        icon=folium.Icon(
            color="blue",
            icon="flag"
        )

    ).add_to(
        traffic_map
    )


    # =====================================================
    # INTERMEDIATE TRAFFIC SPOTS
    # =====================================================

    for spot in spots:

        spot_name, lat, lon = spot


        # Small point
        folium.CircleMarker(

            location=[
                lat,
                lon
            ],

            radius=5,

            color="#666666",

            fill=True,

            fill_color="#666666",

            fill_opacity=0.8,

            tooltip=spot_name

        ).add_to(
            traffic_map
        )


        # Spot name
        folium.Marker(

            location=[
                lat,
                lon
            ],

            icon=folium.DivIcon(

                html=f"""

                <div style="
                    font-size:12px;
                    font-weight:bold;
                    color:#222;
                    background:white;
                    border:1px solid #555;
                    border-radius:4px;
                    padding:3px 6px;
                    white-space:nowrap;
                    box-shadow:1px 1px 3px #888;
                ">

                    {spot_name}

                </div>

                """

            )

        ).add_to(
            traffic_map
        )


    # =====================================================
    # TRAFFIC PROBLEM SPOT
    # =====================================================

    traffic_spot = get_traffic_spot(

        route_name,

        prediction,

        selected_row
    )


    if traffic_spot:

        spot_name, spot_lat, spot_lon = traffic_spot


        # -----------------------------------------------
        # COLOR
        # -----------------------------------------------

        if prediction == "High":

            color = "red"

            status = "HIGH TRAFFIC"

        elif prediction == "Medium":

            color = "orange"

            status = "MEDIUM TRAFFIC"

        else:

            color = "green"

            status = "LOW TRAFFIC"


        # -----------------------------------------------
        # TRAFFIC AREA
        # -----------------------------------------------

        folium.Circle(

            location=[
                spot_lat,
                spot_lon
            ],

            radius=1000,

            color=color,

            fill=True,

            fill_color=color,

            fill_opacity=0.18,

            weight=3

        ).add_to(
            traffic_map
        )


        # -----------------------------------------------
        # TRAFFIC MARKER
        # -----------------------------------------------

        reason = selected_row.get(
            "primary_congestion_reason",
            "Traffic congestion"
        )


        folium.Marker(

            location=[
                spot_lat,
                spot_lon
            ],

            tooltip=f"🚨 {status}",

            popup=f"""

            <div style="width:220px">

            <h4>🚨 Traffic Problem</h4>

            <b>Location:</b><br>
            {spot_name}

            <br><br>

            <b>Traffic Level:</b><br>
            {status}

            <br><br>

            <b>Reason:</b><br>
            {reason}

            </div>

            """,

            icon=folium.Icon(

                color=color,

                icon="warning-sign"
            )

        ).add_to(
            traffic_map
        )


        # -----------------------------------------------
        # BIG TRAFFIC LABEL
        # -----------------------------------------------

        folium.Marker(

            location=[
                spot_lat + 0.012,
                spot_lon
            ],

            icon=folium.DivIcon(

                html=f"""

                <div style="
                    background:{color};
                    color:white;
                    font-size:14px;
                    font-weight:bold;
                    padding:6px 10px;
                    border-radius:6px;
                    border:2px solid white;
                    box-shadow:0px 2px 5px #555;
                    white-space:nowrap;
                ">

                    🚨 {status}<br>
                    {spot_name}

                </div>

                """

            )

        ).add_to(
            traffic_map
        )


    # =====================================================
    # BYPASS
    # =====================================================

    if prediction == "High":

        bypass_name = bypass_routes.get(

            route_name,

            selected_row[
                "connected_routes"
            ]
        )


        if traffic_spot:

            _, traffic_lat, traffic_lon = (
                traffic_spot
            )


            bypass_points = [

                start,

                (
                    traffic_lat + 0.035,
                    traffic_lon - 0.035
                ),

                (
                    traffic_lat + 0.045,
                    traffic_lon + 0.035
                ),

                end
            ]

        else:

            bypass_points = [

                start,

                (
                    (start[0] + end[0]) / 2 + 0.04,

                    (start[1] + end[1]) / 2 + 0.04
                ),

                end
            ]


        # -----------------------------------------------
        # BLUE BYPASS LINE
        # -----------------------------------------------

        folium.PolyLine(

            locations=bypass_points,

            color="#0066ff",

            weight=8,

            opacity=0.95,

            dash_array="12, 8",

            tooltip=(
                f"Recommended Bypass: "
                f"{bypass_name}"
            )

        ).add_to(
            traffic_map
        )


        # -----------------------------------------------
        # BYPASS LABEL
        # -----------------------------------------------

        bypass_label = bypass_points[
            len(bypass_points) // 2
        ]


        folium.Marker(

            location=bypass_label,

            icon=folium.DivIcon(

                html="""

                <div style="
                    background:#0066ff;
                    color:white;
                    font-size:13px;
                    font-weight:bold;
                    padding:7px 12px;
                    border-radius:6px;
                    border:2px solid white;
                    box-shadow:0px 2px 5px #555;
                    white-space:nowrap;
                ">

                    🔵 RECOMMENDED BYPASS

                </div>

                """

            )

        ).add_to(
            traffic_map
        )


    # =====================================================
    # AUTOMATIC ZOOM
    # =====================================================

    all_points = [

        start,

        end

    ]


    for spot in spots:

        all_points.append(
            (
                spot[1],
                spot[2]
            )
        )


    traffic_map.fit_bounds(

        all_points,

        padding=(30, 30)
    )


    # =====================================================
    # LEGEND
    # =====================================================

    legend = """

    <div style="

        position:fixed;

        bottom:25px;

        right:25px;

        z-index:9999;

        background:white;

        border:2px solid #555;

        border-radius:8px;

        padding:12px;

        width:220px;

        font-size:13px;

        box-shadow:0px 2px 8px #777;

    ">

        <b style="font-size:15px;">
            🗺️ Traffic Map
        </b>

        <hr>

        <div>

            <span style="
                display:inline-block;
                width:25px;
                height:5px;
                background:#444;
                margin-right:7px;
            "></span>

            Main Route

        </div>

        <br>

        <div>

            <span style="
                display:inline-block;
                width:25px;
                height:5px;
                background:#0066ff;
                margin-right:7px;
            "></span>

            Recommended Bypass

        </div>

        <br>

        <div>
            🔴 Traffic Problem Spot
        </div>

        <br>

        <div>
            📍 Start / Destination
        </div>

    </div>

    """


    traffic_map.get_root().html.add_child(

        folium.Element(
            legend
        )
    )


    return traffic_map


# =========================================================
# HEADER
# =========================================================

st.markdown(

    """
    <div class="topbar">
        <div class="eyebrow">Urban mobility intelligence / 01</div>
        <div class="live-pill"><span class="live-dot"></span> Control room online</div>
    </div>
    <div class="main-title">
        Know the road<br>before you take it.
    </div>
    """,

    unsafe_allow_html=True
)


st.markdown(

    """
    <div class="sub-title">
        A practical traffic control desk for forecasting congestion,
        locating the pressure point, and choosing a calmer route.
    </div>
    """,

    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# SAME USER INPUTS
# =========================================================

st.sidebar.markdown(
    """
    <div class="eyebrow">Traffic control desk</div>
    <h2 style="margin:.35rem 0 .5rem; color:#eef3f1;">Run a forecast</h2>
    <p style="color:#87938f; font-size:.85rem; line-height:1.5;">
        Set the live context. The model fills in the road conditions from historical route data.
    </p>
    """,
    unsafe_allow_html=True
)


# Route
route_list = sorted(
    df["route_name"]
    .dropna()
    .unique()
    .tolist()
)


selected_route = st.sidebar.selectbox(

    "Select Route",

    route_list
)


# Time
route_df = df[
    df["route_name"] == selected_route
]


time_slots = sorted(
    route_df["time_slot"]
    .dropna()
    .unique()
    .tolist()
)


selected_time = st.sidebar.selectbox(

    "Select Time Slot",

    time_slots
)


# Festival
festival = st.sidebar.checkbox(

    "Festival / Holiday"
)


# Weekend
weekend = st.sidebar.checkbox(

    "Weekend"
)


# Accident
accident = st.sidebar.checkbox(

    "Accident Reported"
)


# Predict
predict_button = st.sidebar.button(

    "Run traffic forecast  →",

    use_container_width=True
)

if predict_button:
    st.session_state["forecast_requested"] = True


# =========================================================
# DEFAULT MESSAGE
# =========================================================

if not st.session_state.get("forecast_requested", False):

    st.markdown(
        """
        <div class="empty-state">
            <div>
                <div class="empty-state-icon">⌁</div>
                <h3>Forecast workspace ready</h3>
                <p>Choose a route and scenario from the control desk, then run a forecast to reveal congestion risk and the recommended response.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:


    # =====================================================
    # HISTORICAL DATA
    # =====================================================

    historical = df[

        (df["route_name"] == selected_route) &

        (df["time_slot"] == selected_time)

    ].copy()


    # If exact time not available
    if historical.empty:

        historical = df[
            df["route_name"] == selected_route
        ].copy()


    if historical.empty:

        st.error(
            "Selected route ka data dataset mein available nahi hai."
        )

        st.stop()


    # =====================================================
    # AUTOMATIC VALUES
    # User doesn't enter these
    # =====================================================

    vehicle_count = int(

        historical[
            "current_vehicle_count"
        ].median()
    )


    historical_vehicle_count = int(

        historical[
            "avg_historical_vehicle_count"
        ].median()
    )


    avg_speed = float(

        historical[
            "avg_speed_kmh"
        ].median()
    )


    potholes = int(

        historical[
            "potholes_count"
        ].median()
    )


    construction_mode = (

        historical[
            "construction_active"
        ].mode()
    )


    if construction_mode.empty:

        construction = 0

    else:

        construction = int(
            construction_mode.iloc[0]
        )


    # =====================================================
    # MODEL INPUT
    # =====================================================

    prediction_input = pd.DataFrame({

        "route_name": [
            selected_route
        ],

        "time_slot": [
            selected_time
        ],

        "is_festival_holiday": [
            int(festival)
        ],

        "is_weekend": [
            int(weekend)
        ],

        "current_vehicle_count": [
            vehicle_count
        ],

        "avg_historical_vehicle_count": [
            historical_vehicle_count
        ],

        "avg_speed_kmh": [
            avg_speed
        ],

        "potholes_count": [
            potholes
        ],

        "construction_active": [
            construction
        ],

        "accident_reported": [
            int(accident)
        ]

    })


    # =====================================================
    # ENCODE
    # =====================================================

    input_encoded = pd.get_dummies(

        prediction_input,

        columns=[
            "route_name",
            "time_slot"
        ],

        dtype=int
    )


    input_encoded = input_encoded.reindex(

        columns=model_columns,

        fill_value=0
    )


    # =====================================================
    # PREDICT
    # =====================================================

    prediction = model.predict(

        input_encoded
    )[0]


    probabilities = model.predict_proba(

        input_encoded
    )[0]


    confidence = (
        max(probabilities) * 100
    )


    # Keep the explanation and map anchored to the selected
    # route/time context instead of a random row with the
    # same predicted class.
    selected_row = historical.iloc[0]


    # =====================================================
    # RESULT METRICS
    # =====================================================

    st.subheader(
        "📊 Traffic Prediction"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "Predicted Congestion",

            prediction
        )


    with col2:

        st.metric(

            "Model Confidence",

            f"{confidence:.1f}%"
        )


    with col3:

        st.metric(

            "Test Accuracy",

            f"{accuracy * 100:.1f}%"
        )
    




    # =====================================================
    # TRAFFIC PREDICTION RESULT
    # =====================================================

    if prediction == "High":
        st.error("🔴 HIGH TRAFFIC CONGESTION")

    elif prediction == "Medium":
        st.warning("🟠 MEDIUM TRAFFIC")

    else:
        st.success("🟢 LOW TRAFFIC")
    # =====================================================
    # TRAFFIC SPOT
    # =====================================================

    traffic_spot = get_traffic_spot(

        selected_route,

        prediction,

        selected_row
    )


    st.subheader(
        "📍 Traffic Problem Location"
    )


    if traffic_spot:

        spot_name = traffic_spot[0]


        c1, c2 = st.columns(2)


        with c1:

            st.metric(

                "Traffic Spot",

                spot_name
            )


        with c2:

            st.metric(

                "Traffic Level",

                prediction
            )


    else:

        st.success(
            "No specific traffic problem spot highlighted."
        )


    # =====================================================
    # MAP
    # =====================================================

    st.subheader(
        "🗺️ Traffic Location Map"
    )


    traffic_map = create_traffic_map(

        selected_route,

        prediction,

        selected_row
    )


    if traffic_map:

        st_folium(

            traffic_map,

            width=None,

            height=650,

            returned_objects=[]
        )


    # =====================================================
    # BYPASS
    # =====================================================

    if prediction == "High":

        bypass_name = bypass_routes.get(

            selected_route,

            selected_row[
                "connected_routes"
            ]
        )


        st.subheader(
            "🔵 Recommended Bypass"
        )


        st.markdown(

            f"""
            <div class="info-box">

            <b>Current Route:</b><br>
            {selected_route}

            <br><br>

            <b>🔴 Traffic Spot:</b><br>
            {traffic_spot[0]
             if traffic_spot
             else "Congestion detected"}

            <br><br>

            <b>🔵 Recommended Bypass:</b><br>
            {bypass_name}

            </div>
            """,

            unsafe_allow_html=True
        )


    elif prediction == "Medium":

        st.info(

            "🟠 Medium congestion: "
            "Traffic monitoring and signal timing "
            "adjustment recommended."
        )


    else:

        st.success(

            "🟢 Low congestion: "
            "Normal traffic operation can continue."
        )


    # =====================================================
    # AUTOMATIC TRAFFIC DATA
    # =====================================================

    with st.expander(
        "🔎 Automatically Estimated Traffic Data"
    ):

        st.write(

            "Ye technical values user se manually "
            "input nahi li ja rahi hain. "
            "System selected route aur time slot ke "
            "historical dataset data se estimate karta hai."
        )


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(

                "Vehicle Count",

                vehicle_count
            )


        with c2:

            st.metric(

                "Average Speed",

                f"{avg_speed:.1f} km/h"
            )


        with c3:

            st.metric(

                "Potholes",

                potholes
            )


        with c4:

            st.metric(
                "Construction",
                "Yes" if construction else "No"
            )


    # =====================================================
    # CONGESTION REASON
    # =====================================================

    st.subheader(
        "🧠 Congestion Reason"
    )


    reason = selected_row.get(

        "primary_congestion_reason",

        "Not Available"
    )


    st.info(
        f"Dataset-based primary reason: **{reason}**"
    )


    # =====================================================
    # MUNICIPAL ACTION
    # =====================================================

    st.subheader(
        "🏛️ Suggested Municipal Action"
    )


    if prediction == "High":

        st.write(
            "🔴 Alert traffic control team"
        )

        st.write(
            "🔴 Monitor the highlighted traffic spot"
        )

        st.write(
            "🔵 Consider the recommended bypass"
        )

        st.write(
            "🟠 Review signal timing"
        )


    elif prediction == "Medium":

        st.write(
            "🟠 Monitor traffic condition"
        )

        st.write(
            "🟠 Review signal timing"
        )


    else:

        st.write(
            "🟢 Continue normal traffic monitoring"
        )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.caption(

        "Note: Traffic spot coordinates shown on the map "
        "are approximate visualization coordinates because "
        "the provided CSV does not contain GPS latitude/"
        "longitude for each intermediate traffic spot. "
        "The map is therefore a project visualization, "
        "not a live navigation/traffic map."
    )