import streamlit as st
import pandas as pd
import numpy as np

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

.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #222;
}

.sub-title {
    font-size: 16px;
    color: #666;
    margin-bottom: 20px;
}

.high-box {
    background-color: #ffe6e6;
    border-left: 6px solid #d00000;
    padding: 18px;
    border-radius: 8px;
}

.medium-box {
    background-color: #fff3d6;
    border-left: 6px solid #ff9800;
    padding: 18px;
    border-radius: 8px;
}

.low-box {
    background-color: #e5f7e5;
    border-left: 6px solid #159447;
    padding: 18px;
    border-radius: 8px;
}

.info-box {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATASET
# =========================================================

DATA_FILE = "project_traffic_dataset-v4.csv"

try:
    df = pd.read_csv(DATA_FILE)

except FileNotFoundError:

    st.error(
        f"❌ {DATA_FILE} not found.\n\n"
        "CSV file ko app.py ke same folder mein rakho."
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
    <div class="main-title">
        🚦 AI-Driven Road Traffic Congestion Prediction
    </div>
    """,

    unsafe_allow_html=True
)


st.markdown(

    """
    <div class="sub-title">
        AI-based traffic congestion prediction,
        traffic spot detection and bypass advisory system.
    </div>
    """,

    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# SAME USER INPUTS
# =========================================================

st.sidebar.header(
    "🚗 Traffic Input"
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

    "🔍 Predict Traffic",

    use_container_width=True
)


# =========================================================
# DEFAULT MESSAGE
# =========================================================

if not predict_button:

    st.info(

        "👈 Select Route, Time Slot, Festival/Holiday, "
        "Weekend and Accident status, then click "
        "**Predict Traffic**."
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


    # =====================================================
    # REPRESENTATIVE DATASET ROW
    # =====================================================

    matching_prediction = historical[

        historical[
            "congestion_level"
        ] == prediction

    ]


    if matching_prediction.empty:

        selected_row = historical.iloc[0]

    else:

        selected_row = (
            matching_prediction.iloc[0]
        )


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