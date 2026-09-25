
import math
import io
import textwrap
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tatermaxxer_basketengine_v5 import OptimizerConfig, optimize as run_optimizer

st.set_page_config(page_title="Tatermaxxer", layout="wide")

st.title("Tatermaxxer v5.10")


# ----------------------------
# Primary UI layout
# ----------------------------
optimal_stats_output = st.container()
input_col, cross_section_col = st.columns([1.0, 1.55], gap="large")


def persistent_number_input(label, saved_key, default_value, min_value, step):
    """Keep a value even while its temporary Streamlit widget is hidden."""
    widget_key = f"_{saved_key}"
    if saved_key not in st.session_state:
        st.session_state[saved_key] = float(default_value)
    st.session_state[widget_key] = st.session_state[saved_key]

    def save_value():
        st.session_state[saved_key] = float(st.session_state[widget_key])

    return st.number_input(
        label,
        min_value=float(min_value),
        step=float(step),
        key=widget_key,
        on_change=save_value,
    )


def basket_class_controls(bag_class, flat_width_default, squished_width_default):
    saved_shape_key = f"shape_{bag_class}_saved"
    widget_shape_key = f"_shape_{bag_class}_widget"
    if saved_shape_key not in st.session_state:
        st.session_state[saved_shape_key] = "Squished"
    st.session_state[widget_shape_key] = st.session_state[saved_shape_key]

    def save_shape(key=saved_shape_key, widget_key=widget_shape_key):
        st.session_state[key] = st.session_state[widget_key]

    shape = st.radio(
        f"{bag_class}-bag shape",
        ["Flat", "Squished"],
        horizontal=True,
        key=widget_shape_key,
        on_change=save_shape,
    )
    shape_key = shape.lower()
    width_default = flat_width_default if shape == "Flat" else squished_width_default
    height_default = 1.5 if shape == "Flat" else 2.0
    selected_width = persistent_number_input(
        f"{bag_class}-bag {shape_key} minimum width (in)",
        f"{shape_key}_minimum_{bag_class}_bag_width_saved",
        width_default,
        0.0,
        0.05,
    )
    selected_height = persistent_number_input(
        f"{bag_class}-bag {shape_key} inside height (in)",
        f"{shape_key}_{bag_class}_bag_inside_height_saved",
        height_default,
        0.01,
        0.05,
    )
    return shape, selected_width, selected_height

with input_col:
    st.subheader("Input parameters for 2D model")
    with st.expander("Vessel parameters", expanded=False):
        diameter = st.number_input(
            "Vessel diameter (in)", min_value=0.01, value=39.0, step=0.25
        )
        clearance = st.number_input(
            "Minimum clearance from vessel wall (in)",
            min_value=0.0, value=0.40, step=0.05
        )
        heater_height = st.number_input(
            "Height of internal heater plus clearance (in)",
            min_value=0.0, value=5.00, step=0.25
        )

    if "bag_length_saved" not in st.session_state:
        st.session_state["bag_length_saved"] = 16.0
    if "bag_width_saved" not in st.session_state:
        st.session_state["bag_width_saved"] = 9.9

    def save_bag_length():
        st.session_state["bag_length_saved"] = float(
            st.session_state["_bag_length_widget"]
        )

    def save_bag_width():
        st.session_state["bag_width_saved"] = float(
            st.session_state["_bag_width_widget"]
        )

    st.session_state["_bag_length_widget"] = st.session_state["bag_length_saved"]
    st.session_state["_bag_width_widget"] = st.session_state["bag_width_saved"]
    bag_length = st.number_input(
        "Bag length (in)",
        min_value=0.01,
        step=0.05,
        key="_bag_length_widget",
        on_change=save_bag_length,
        help="Bag dimension in the Y direction.",
    )
    bag_width = st.number_input(
        "Bag width (in)",
        min_value=0.01,
        step=0.05,
        key="_bag_width_widget",
        on_change=save_bag_width,
        help="Bag dimension in the X direction.",
    )
    basket_wall_thickness = st.number_input(
        "Basket wall thickness (in)", min_value=0.0, value=0.25, step=0.05
    )

    heater_suggestion_output = st.empty()

    st.markdown("**Fitting basket widths**")
    st.caption(
        "Choose Flat or Squished independently for each basket class. "
        "Only the selected minimum width and inside height are sent to the optimizer."
    )
    shape_3, minimum_3_bag_basket_width, inside_height_3 = (
        basket_class_controls(3, 42.0, 29.75)
    )
    shape_2, minimum_2_bag_basket_width, inside_height_2 = (
        basket_class_controls(2, 28.0, 19.8)
    )
    shape_1, minimum_1_bag_basket_width, inside_height_1 = (
        basket_class_controls(1, 14.0, 9.9)
    )
    if not (
        minimum_1_bag_basket_width <= minimum_2_bag_basket_width
        <= minimum_3_bag_basket_width
    ):
        st.error("Basket width thresholds must satisfy 1-bag <= 2-bag <= 3-bag.")
        st.stop()
    class_specs = {
        1: {
            "min_width": float(minimum_1_bag_basket_width),
            "inside_height": float(inside_height_1),
            "height": float(inside_height_1 + basket_wall_thickness),
        },
        2: {
            "min_width": float(minimum_2_bag_basket_width),
            "inside_height": float(inside_height_2),
            "height": float(inside_height_2 + basket_wall_thickness),
        },
        3: {
            "min_width": float(minimum_3_bag_basket_width),
            "inside_height": float(inside_height_3),
            "height": float(inside_height_3 + basket_wall_thickness),
        },
    }
    row_settings_output = st.container()

with cross_section_col:
    st.subheader("Cross-Sectional Layout")
    cross_section_output = st.container()
    manufacturing_settings_output = st.container()

R = diameter / 2.0
usable_R = R - clearance
lower_bottom = max(clearance, heater_height)
top_limit = diameter - clearance

if heater_height > diameter:
    st.error("The internal heater height cannot exceed the circle diameter.")
    st.stop()
if usable_R <= 0:
    st.error("Clearance must be smaller than half the circle diameter.")
    st.stop()

# These controls render beneath the 2D drawing, to the right of the basket
# fitting controls, without changing their established behavior.
with manufacturing_settings_output:
    simplify_manufacturing = st.toggle(
        "Simplify manufacturing",
        value=True,
        help=(
            "When enabled, reruns the layout optimization using at most one common "
            "manufacturing width for each basket category."
        ),
    )
    if simplify_manufacturing:
        manufacturing_increment = st.number_input(
            "Manufacturing width increment (in)",
            min_value=0.01,
            value=0.25,
            step=0.05,
            format="%.2f",
        )
    else:
        manufacturing_increment = 0.25

min_class_height = min(spec["height"] for spec in class_specs.values())
available_vertical_span = max(0.0, top_limit - lower_bottom)
max_rows = int(math.floor(available_vertical_span / min_class_height + 1e-12))

with row_settings_output:
    row_mode = st.radio(
        "Row count",
        ["Optimize automatically", "Specify row count"],
        horizontal=True,
    )
    if row_mode == "Specify row count":
        if max_rows < 1:
            st.error("No row can fit with the specified vessel and carrier heights.")
            st.stop()
        requested_rows = st.number_input(
            "Number of rows",
            min_value=1,
            max_value=max_rows,
            value=1,
            step=1,
        )
    else:
        requested_rows = None
if max_rows < 1:
    st.error("No row can fit with the specified vessel and carrier heights.")
    st.stop()

# ----------------------------
# Variable-height branch optimizer
# ----------------------------
# The mathematical solver remains isolated in the unchanged basket engine.
solver_config = OptimizerConfig(
    diameter=diameter,
    clearance=clearance,
    basket_wall_thickness=basket_wall_thickness,
    heater_height=heater_height,
    minimum_3_bag_basket_width=minimum_3_bag_basket_width,
    minimum_2_bag_basket_width=minimum_2_bag_basket_width,
    minimum_1_bag_basket_width=minimum_1_bag_basket_width,
    inside_height_3=inside_height_3,
    inside_height_2=inside_height_2,
    inside_height_1=inside_height_1,
    simplify_manufacturing=simplify_manufacturing,
    manufacturing_increment=manufacturing_increment,
    row_mode=row_mode,
    requested_rows=int(requested_rows) if requested_rows is not None else None,
)
solver_result = run_optimizer(solver_config)
optimized = None if solver_result is None else (solver_result["score"], solver_result["rows"])


if optimized is None:
    st.error(
        "No carrier arrangement can fit the vessel constraints using the configured geometry."
    )
    st.stop()

_, row_details = optimized
rows = [(row["bottom_z"], row["width"]) for row in row_details]
n = len(row_details)
total_bags = sum(row["bags"] for row in row_details)
total_area = sum(row["height"] * row["width"] for row in row_details)
active_wall_thickness = basket_wall_thickness

usable_area = math.pi * usable_R**2
heater_vertical_distance = abs(heater_height - R)
heater_chord_q = usable_R**2 - heater_vertical_distance**2
heater_chord = (
    2.0 * math.sqrt(max(0.0, heater_chord_q))
    if heater_chord_q >= -1e-10
    else 0.0
)
heater_area = heater_height * heater_chord
usable_area -= heater_area

utilization = 100.0 * total_area / usable_area if usable_area > 0 else 0.0
physical_circle_area = math.pi * R**2
physical_utilization = 100.0 * total_area / physical_circle_area if physical_circle_area else 0.0

# ----------------------------
# Visualization
# ----------------------------

fig, ax = plt.subplots(figsize=(6.8, 6.8))
ax.set_aspect("equal", adjustable="box")

# Display coordinates: bottom of physical circle = Z=0, top = Z=diameter.
physical = Circle((0, R), R, fill=False, linewidth=2.0, label="Circle")
usable = Circle((0, R), usable_R, fill=False, linestyle="--", linewidth=1.5,
                label="Clearance boundary")
ax.add_patch(physical)
ax.add_patch(usable)

for idx, row in enumerate(row_details, start=1):
    bottom_z = row["bottom_z"]
    width = row["width"]
    row_height = row["height"]
    bag_class = row["bags"]
    display_z = bottom_z + row_height / 2.0
    basket_color = {3: "green", 2: "blue", 1: "purple"}.get(bag_class, "gray")
    ax.add_patch(Rectangle(
        (-width / 2.0, bottom_z),
        width, row_height,
        fill=True, alpha=1.0, facecolor=basket_color,
        edgecolor="black", linewidth=0.8, zorder=3
    ))
    ax.text(0, display_z, f'{idx}: {width:.2f}"',
            ha="center", va="center", fontsize=8, zorder=4)

# Heater starts at the physical bottom and extends straight upward.
if heater_height > 0:
    ax.plot(
        [0, 0], [0, heater_height],
        linewidth=4.0, solid_capstyle="butt", color="black",
        label="Internal heater", zorder=5
    )

lim = R + 1.5
ax.set_xlim(-lim, lim)
ax.set_ylim(0, diameter)
ax.set_xlabel("X (in)")
ax.set_ylabel("Z (in)")
# Keep gridlines at 1-inch spacing, but label the axes every 5 inches.
x_grid_positions = np.arange(-R, R + 0.001, 1.0)
x_display_values = x_grid_positions + R
ax.set_xticks(x_grid_positions, minor=True)
x_label_values = np.arange(0.0, diameter + 0.001, 5.0)
x_label_positions = x_label_values - R
ax.set_xticks(x_label_positions)
ax.set_xticklabels([f"{x:.0f}" for x in x_label_values])

z_grid_positions = np.arange(0.0, diameter + 0.001, 1.0)
ax.set_yticks(z_grid_positions, minor=True)
z_label_values = np.arange(0.0, diameter + 0.001, 5.0)
ax.set_yticks(z_label_values)
ax.set_yticklabels([f"{z:.0f}" for z in z_label_values])

# Draw the grid beneath all basket/heater geometry.
ax.set_axisbelow(True)
ax.grid(True, which="minor", alpha=0.2, zorder=0)
ax.grid(True, which="major", alpha=0.2, zorder=0)

green_basket = Rectangle((0, 0), 1, 1, facecolor="green", alpha=1.0, edgecolor="black")
blue_basket = Rectangle((0, 0), 1, 1, facecolor="blue", alpha=1.0, edgecolor="black")
purple_basket = Rectangle((0, 0), 1, 1, facecolor="purple", alpha=1.0, edgecolor="black")
ax.legend(handles=[green_basket, blue_basket, purple_basket],
          labels=["3 bag basket", "2 bag basket", "1 bag basket"],
          loc="upper right")

# Populate the full-width Stage 1 summary in its placeholder above both columns.
with optimal_stats_output:
    st.markdown("**Optimal arrangement**")
    summary_cols = st.columns(5)
    summary_cols[0].metric("Baskets", f"{n}")
    summary_cols[1].metric("Number of bags", f"{total_bags}")
    summary_cols[2].metric("Total basket area", f"{total_area:,.2f} in²")
    summary_cols[3].metric("Clearance-adjusted area utilization", f"{utilization:.2f}%")
    summary_cols[4].metric("Full vessel utilization", f"{physical_utilization:.2f}%")

with cross_section_output:
    st.pyplot(fig, use_container_width=False)

if simplify_manufacturing and rows:
    lowest_basket_bottom_z = min(row["bottom_z"] for row in row_details)
    if lowest_basket_bottom_z > heater_height + 1e-9:
        with heater_suggestion_output:
            st.caption(
                f"Suggested internal heater height: {lowest_basket_bottom_z:.2f} in "
                "(would touch the lowest basket without changing the current heater input)."
            )

# ----------------------------
# Stage 2: axial vessel capacity / deferred 3D model
# ----------------------------
# Stage 1 remains live. The 3D model uses a snapshot of the most recently applied
# Stage 1 result so normal input changes do not rebuild the expensive Plotly scene.

def make_3d_snapshot():
    return {
        "rows": [
            (float(row["bottom_z"]), float(row["width"]), float(row["height"]), int(row["bags"]))
            for row in row_details
        ],
        "diameter": float(diameter),
        "clearance": float(clearance),
        "heater_height": float(heater_height),
        "minimum_3": float(minimum_3_bag_basket_width),
        "minimum_2": float(minimum_2_bag_basket_width),
        "minimum_1": float(minimum_1_bag_basket_width),
        "total_bags": int(total_bags),
        "n": int(n),
        "basket_inside_length": float(basket_inside_length),
        "basket_wall_thickness": float(active_wall_thickness),
        "basket_outside_length": float(basket_outside_length),
        "axial_stack_gap": float(axial_stack_gap),
        "vessel_length_m": float(vessel_length_m),
        "vessel_length_in": float(vessel_length_in),
        "vessel_dead_volume_m": float(vessel_dead_volume_m),
        "dead_volume_in": float(dead_volume_in),
        "usable_axial_length_in": float(usable_axial_length_in),
        "axial_stack_pitch": float(axial_stack_pitch),
        "axial_stack_count": int(axial_stack_count),
        "axial_start_y": float(axial_start_y),
    }


def build_3d_figure(snapshot):
    rows_3d = snapshot["rows"]
    diameter_3d = snapshot["diameter"]
    clearance_3d = snapshot["clearance"]
    heater_height_3d = snapshot["heater_height"]
    minimum_3_3d = snapshot["minimum_3"]
    minimum_2_3d = snapshot["minimum_2"]
    minimum_1_3d = snapshot["minimum_1"]

    R_3d = diameter_3d / 2.0
    usable_R_3d = R_3d - clearance_3d
    view_length_y = snapshot["vessel_length_in"]
    basket_length_3d = snapshot["basket_outside_length"]
    stack_gap_3d = snapshot["axial_stack_gap"]
    stack_pitch_3d = snapshot["axial_stack_pitch"]
    stack_count_3d = snapshot["axial_stack_count"]
    axial_start_y_3d = snapshot["axial_start_y"]
    fig_obj = go.Figure()

    # Lower-resolution vessel shell: visually smooth enough while reducing WebGL work.
    theta = np.linspace(0.0, 2.0 * math.pi, 36)
    y_shell = np.linspace(0.0, view_length_y, 2)
    theta_grid, y_grid = np.meshgrid(theta, y_shell)
    x_shell = R_3d * np.cos(theta_grid)
    z_shell = R_3d + R_3d * np.sin(theta_grid)
    fig_obj.add_trace(go.Surface(
        x=x_shell, y=y_grid, z=z_shell,
        surfacecolor=np.zeros_like(x_shell),
        colorscale=[[0.0, "lightgray"], [1.0, "lightgray"]],
        cmin=0, cmax=1, showscale=False, opacity=0.13,
        contours=dict(
            x=dict(show=False, highlight=False),
            y=dict(show=False, highlight=False),
            z=dict(show=False, highlight=False),
        ),
        hoverinfo="skip", name="Vessel", showlegend=False,
    ))

    # Physical vessel end rings only. The clearance boundary is intentionally
    # not drawn in 3D; clearance still remains fully active in Stage 1 placement.
    ring_x, ring_y, ring_z = [], [], []
    for y_end in (0.0, view_length_y):
        ring_x.extend((R_3d * np.cos(theta)).tolist() + [None])
        ring_y.extend(np.full_like(theta, y_end).tolist() + [None])
        ring_z.extend((R_3d + R_3d * np.sin(theta)).tolist() + [None])
    fig_obj.add_trace(go.Scatter3d(
        x=ring_x, y=ring_y, z=ring_z, mode="lines",
        line=dict(color="gray", width=4), hoverinfo="skip", showlegend=False,
    ))

    # Basket dead space is the leftover usable Y-length after the final complete
    # axial stack. Show it as a translucent red cylindrical segment. This is
    # display-only and does not affect axial stack count or basket placement.
    usable_axial_length_3d = snapshot["usable_axial_length_in"]
    used_axial_length_3d = (
        stack_count_3d * basket_length_3d
        + max(0, stack_count_3d - 1) * stack_gap_3d
    )
    basket_dead_space_3d = max(0.0, usable_axial_length_3d - used_axial_length_3d)
    dead_space_y0 = axial_start_y_3d + used_axial_length_3d
    dead_space_y1 = dead_space_y0 + basket_dead_space_3d

    if basket_dead_space_3d > 1e-10:
        dead_theta = np.linspace(0.0, 2.0 * math.pi, 36)
        dead_y = np.array([dead_space_y0, dead_space_y1])
        dead_theta_grid, dead_y_grid = np.meshgrid(dead_theta, dead_y)
        dead_x_shell = R_3d * np.cos(dead_theta_grid)
        dead_z_shell = R_3d + R_3d * np.sin(dead_theta_grid)
        fig_obj.add_trace(go.Surface(
            x=dead_x_shell, y=dead_y_grid, z=dead_z_shell,
            surfacecolor=np.zeros_like(dead_x_shell),
            colorscale=[[0.0, "red"], [1.0, "red"]],
            cmin=0, cmax=1, showscale=False, opacity=0.10,
            contours=dict(
                x=dict(show=False, highlight=False),
                y=dict(show=False, highlight=False),
                z=dict(show=False, highlight=False),
            ),
            hovertemplate=f"Basket dead space: {basket_dead_space_3d:.2f} in<extra></extra>", name="Basket dead space", showlegend=False,
        ))

        # Circular end faces make the dead-space region read as a shaded
        # cross-sectional volume rather than only a red outline.
        cap_x = [0.0] + (R_3d * np.cos(dead_theta[:-1])).tolist()
        cap_z = [R_3d] + (R_3d + R_3d * np.sin(dead_theta[:-1])).tolist()
        n_cap = len(cap_x)
        cap_i, cap_j, cap_k = [], [], []
        for idx in range(1, n_cap):
            nxt = 1 if idx == n_cap - 1 else idx + 1
            cap_i.append(0); cap_j.append(idx); cap_k.append(nxt)
        for cap_y in (dead_space_y0, dead_space_y1):
            fig_obj.add_trace(go.Mesh3d(
                x=cap_x, y=[cap_y] * n_cap, z=cap_z,
                i=cap_i, j=cap_j, k=cap_k,
                color="red", opacity=0.10, flatshading=False,
                hoverinfo="skip", showscale=False, showlegend=False,
            ))

    edge_x, edge_y, edge_z = [], [], []
    z_edge_x, z_edge_y, z_edge_z = [], [], []
    mesh_groups = {
        "green": {"x": [], "y": [], "z": [], "i": [], "j": [], "k": [], "customdata": []},
        "blue": {"x": [], "y": [], "z": [], "i": [], "j": [], "k": [], "customdata": []},
        "purple": {"x": [], "y": [], "z": [], "i": [], "j": [], "k": [], "customdata": []},
        "gray": {"x": [], "y": [], "z": [], "i": [], "j": [], "k": [], "customdata": []},
    }

    def add_box_to_group(x0, x1, z0, z1, y0, y1, color, stack_number):
        x = [x0, x1, x1, x0, x0, x1, x1, x0]
        y = [y0, y0, y0, y0, y1, y1, y1, y1]
        z = [z0, z0, z1, z1, z0, z0, z1, z1]
        # Twelve triangles (two per rectangular face) with consistent outward winding.
        # Vertex layout: 0-3 are the Y- face; 4-7 are the Y+ face.
        # Faces: Y-, Y+, X-, X+, Z-, Z+.
        local_i = [0, 0, 4, 4, 0, 0, 1, 1, 0, 0, 3, 3]
        local_j = [3, 2, 5, 6, 4, 7, 2, 6, 1, 5, 7, 6]
        local_k = [2, 1, 6, 7, 7, 3, 6, 5, 5, 4, 6, 2]
        group = mesh_groups[color]
        offset = len(group["x"])
        group["x"].extend(x); group["y"].extend(y); group["z"].extend(z)
        group["i"].extend(v + offset for v in local_i)
        group["j"].extend(v + offset for v in local_j)
        group["k"].extend(v + offset for v in local_k)
        group["customdata"].extend([stack_number] * 8)
        # Draw Z-parallel edges darker/thicker to make axial stack boundaries easier to read.
        z_parallel = {(1,2), (3,0), (5,6), (7,4)}
        for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            if (a, b) in z_parallel:
                z_edge_x.extend([x[a], x[b], None])
                z_edge_y.extend([y[a], y[b], None])
                z_edge_z.extend([z[a], z[b], None])
            else:
                edge_x.extend([x[a], x[b], None])
                edge_y.extend([y[a], y[b], None])
                edge_z.extend([z[a], z[b], None])

    for stack_idx in range(stack_count_3d):
        y0 = axial_start_y_3d + stack_idx * stack_pitch_3d
        y1 = y0 + basket_length_3d
        for bottom_z, width, row_height, bag_class in rows_3d:
            basket_color = {3: "green", 2: "blue", 1: "purple"}.get(bag_class, "gray")
            add_box_to_group(
                -width/2.0, width/2.0, bottom_z, bottom_z + row_height,
                y0, y1, basket_color, stack_idx + 1
            )

    # One Mesh3d trace per basket category instead of one trace per basket.
    for basket_color, group in mesh_groups.items():
        if not group["x"]:
            continue
        fig_obj.add_trace(go.Mesh3d(
            x=group["x"], y=group["y"], z=group["z"],
            i=group["i"], j=group["j"], k=group["k"],
            customdata=group["customdata"],
            color=basket_color, opacity=1.0, flatshading=True,
            hovertemplate="Axial stack %{customdata}<extra></extra>",
            showscale=False, showlegend=False,
        ))

    fig_obj.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode="lines",
        line=dict(color="rgba(10,10,10,0.95)", width=4),
        hoverinfo="skip", showlegend=False,
    ))
    fig_obj.add_trace(go.Scatter3d(
        x=z_edge_x, y=z_edge_y, z=z_edge_z, mode="lines",
        line=dict(color="rgba(0,0,0,1.0)", width=6),
        hoverinfo="skip", showlegend=False,
    ))

    if heater_height_3d > 0:
        fig_obj.add_trace(go.Mesh3d(
            x=[0.0, 0.0, 0.0, 0.0],
            y=[0.0, view_length_y, view_length_y, 0.0],
            z=[0.0, 0.0, heater_height_3d, heater_height_3d],
            i=[0, 0], j=[1, 2], k=[2, 3], color="black", opacity=0.35,
            name="Internal heater", hoverinfo="skip", showlegend=False,
        ))

    # True 3D compass outside the vessel. Consolidated to one trace per axis/color.
    compass_len = max(diameter_3d * 0.16, 1.0)
    ox = R_3d + compass_len * 1.55
    oy = view_length_y * 0.06
    oz = diameter_3d * 0.12
    compass_axes = [
        ("red", [("X-", (ox - compass_len, oy, oz)), ("X+", (ox + compass_len, oy, oz))]),
        ("green", [("Y-", (ox, max(0.0, oy - compass_len), oz)), ("Y+", (ox, oy + compass_len, oz))]),
        ("blue", [("Z-", (ox, oy, max(0.0, oz - compass_len))), ("Z+", (ox, oy, oz + compass_len))]),
    ]
    for axis_color, endpoints in compass_axes:
        neg_label, neg = endpoints[0]
        pos_label, pos = endpoints[1]
        fig_obj.add_trace(go.Scatter3d(
            x=[neg[0], ox, pos[0]], y=[neg[1], oy, pos[1]], z=[neg[2], oz, pos[2]],
            mode="lines+markers+text", line=dict(color=axis_color, width=5),
            marker=dict(size=[5, 3, 5], color=axis_color),
            text=[neg_label, "", pos_label], textposition="top center",
            textfont=dict(color=axis_color, size=11), hoverinfo="skip", showlegend=False,
        ))

    # Draw a custom numbered X axis on the Z- side of the vessel. Plotly does not
    # provide a way to relocate the native 3D X axis independently, so the native
    # X axis is hidden and this display-only axis is positioned just below the vessel.
    axis_z = -max(diameter_3d * 0.055, 0.75)
    x_ticks = np.linspace(-R_3d, R_3d, 7)
    fig_obj.add_trace(go.Scatter3d(
        x=x_ticks, y=np.zeros_like(x_ticks), z=np.full_like(x_ticks, axis_z),
        mode="lines+markers+text",
        line=dict(color="rgba(80,80,80,0.85)", width=3),
        marker=dict(size=3, color="rgba(80,80,80,0.85)"),
        text=[f"{value:.0f}" for value in x_ticks],
        textposition="bottom center",
        textfont=dict(color="rgba(80,80,80,0.95)", size=10),
        hoverinfo="skip", showlegend=False,
    ))

    x_max = ox + compass_len * 1.25
    fig_obj.update_layout(
        margin=dict(l=0, r=0, t=10, b=0), height=560,
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="", range=[-R_3d - 1.5, x_max], visible=False, showspikes=False),
            yaxis=dict(title="", range=[0.0, view_length_y + 1.0], showspikes=False),
            zaxis=dict(title="", range=[axis_z - 0.5, diameter_3d], visible=False, showspikes=False),
            camera=dict(eye=dict(x=1.35, y=-2.15, z=0.70), up=dict(x=0.0, y=0.0, z=1.0)),
        ),
        legend=dict(orientation="h"),
    )
    return fig_obj


st.subheader("3D Visualization")
st.markdown("**3D inputs**")
input_3d_1, input_3d_2, input_3d_3, input_3d_4 = st.columns(4)

with input_3d_1:
    basket_inside_length = persistent_number_input(
        "Basket inside length (in)",
        "basket_inside_length_3d_saved",
        16.0,
        0.01,
        0.25,
    )
with input_3d_2:
    axial_stack_gap = persistent_number_input(
        "Gap between stacks (in)",
        "axial_stack_gap_3d_saved",
        0.1,
        0.0,
        0.05,
    )
with input_3d_3:
    specify_axial_stacks = st.toggle("Specify number of stacks", value=False)
with input_3d_4:
    vessel_dead_volume_m = persistent_number_input(
        "Vessel dead volume (m)",
        "vessel_dead_volume_3d_saved",
        0.3,
        0.0,
        0.1,
    )

basket_outside_length = basket_inside_length + (2.0 * active_wall_thickness)
axial_stack_pitch = basket_outside_length + axial_stack_gap

if specify_axial_stacks:
    requested_axial_stacks = persistent_number_input(
        "Number of stacks",
        "requested_axial_stacks_3d_saved",
        15.0,
        1.0,
        1.0,
    )
    axial_stack_count = int(requested_axial_stacks)
    required_usable_length_in = (
        axial_stack_count * basket_outside_length
        + max(0, axial_stack_count - 1) * axial_stack_gap
    )
    vessel_length_m = vessel_dead_volume_m + required_usable_length_in / 39.37
    st.metric("Ideal vessel length", f"{vessel_length_m:.2f} m")
else:
    vessel_length_m = persistent_number_input(
        "Vessel length (m)",
        "vessel_length_3d_saved",
        7.0,
        0.01,
        0.1,
    )

# Stage 2 axial dimensions are controlled by the UI inputs above.
# Basket inside length is converted to outside basket length using the shared basket wall thickness input.
# Dead volume reserves the front (Y-) end and reduces usable axial basket length;
# the 3D cylinder still represents the full physical vessel length.
vessel_length_in = vessel_length_m * 39.37
dead_volume_m_effective = min(vessel_dead_volume_m, vessel_length_m)
dead_volume_in = dead_volume_m_effective * 39.37
usable_vessel_length_m = max(0.0, vessel_length_m - dead_volume_m_effective)
usable_vessel_length_in = usable_vessel_length_m * 39.37
usable_axial_length_in = usable_vessel_length_in
if not specify_axial_stacks:
    axial_stack_count = (
        int(math.floor((usable_axial_length_in + axial_stack_gap + 1e-12) / axial_stack_pitch))
        if axial_stack_pitch > 0 else 0
    )
# Display all dead volume at the front (Y-) side of the vessel.
axial_start_y = dead_volume_in

# Build the live one-stack trolley view from the same geometry and rendering
# function used by the full payload view.
trolley_snapshot = make_3d_snapshot()
trolley_snapshot.update({
    "vessel_length_m": basket_outside_length / 39.37,
    "vessel_length_in": basket_outside_length,
    "vessel_dead_volume_m": 0.0,
    "dead_volume_in": 0.0,
    "usable_axial_length_in": basket_outside_length,
    "axial_stack_pitch": basket_outside_length,
    "axial_stack_count": 1,
    "axial_start_y": 0.0,
})
trolley_figure = build_3d_figure(trolley_snapshot)

if "model_3d_snapshot" not in st.session_state:
    st.session_state.model_3d_snapshot = make_3d_snapshot()
    st.session_state.model_3d_figure = build_3d_figure(st.session_state.model_3d_snapshot)

def refresh_3d_model():
    """Refresh only the deferred 3D snapshot and figure."""
    st.session_state.model_3d_snapshot = make_3d_snapshot()
    st.session_state.model_3d_figure = build_3d_figure(st.session_state.model_3d_snapshot)

snapshot_3d = st.session_state.model_3d_snapshot
fig3d = st.session_state.model_3d_figure

# Live Stage 2 outputs. These recalculate immediately from the current inputs and
# current 2D result. The expensive Plotly geometry remains deferred until Refresh.
stack_count_display = int(axial_stack_count)
basket_outside_length_display = float(basket_outside_length)
stack_gap_display = float(axial_stack_gap)
vessel_length_m_display = float(vessel_length_m)
dead_volume_m_display = float(vessel_dead_volume_m)
dead_volume_in_display = float(dead_volume_in)
usable_axial_length_display = float(usable_axial_length_in)
axial_used_length = (
    stack_count_display * basket_outside_length_display
    + max(0, stack_count_display - 1) * stack_gap_display
)
basket_dead_space = max(0.0, usable_axial_length_display - axial_used_length)
total_vessel_baskets = n * stack_count_display
total_vessel_bags = total_bags * stack_count_display

# 3D utilization metrics are volume analogs of the existing 2D metrics, using live values.
stack_basket_area_3d = total_area
total_basket_volume_3d = stack_basket_area_3d * basket_outside_length_display * stack_count_display
R_live = diameter / 2.0
usable_R_live = R_live - clearance
heater_vertical_distance_3d = abs(heater_height - R_live)
heater_chord_q_3d = usable_R_live**2 - heater_vertical_distance_3d**2
heater_chord_3d = (
    2.0 * math.sqrt(max(0.0, heater_chord_q_3d))
    if heater_chord_q_3d >= -1e-10 else 0.0
)
usable_cross_section_area_3d = max(
    0.0,
    math.pi * usable_R_live**2 - heater_height * heater_chord_3d,
)
clearance_adjusted_vessel_volume_3d = usable_cross_section_area_3d * usable_axial_length_display
full_vessel_volume_3d = math.pi * R_live**2 * vessel_length_in
clearance_adjusted_utilization_3d = (
    100.0 * total_basket_volume_3d / clearance_adjusted_vessel_volume_3d
    if clearance_adjusted_vessel_volume_3d > 0 else 0.0
)
full_vessel_utilization_3d = (
    100.0 * total_basket_volume_3d / full_vessel_volume_3d
    if full_vessel_volume_3d > 0 else 0.0
)

# Flag only geometry-relevant changes. Cycle time is intentionally excluded.
def current_3d_signature():
    return (
        tuple(
            (round(float(row["bottom_z"]), 10), round(float(row["width"]), 10),
             round(float(row["height"]), 10), int(row["bags"]))
            for row in row_details
        ),
        round(float(diameter), 10), round(float(clearance), 10),
        round(float(heater_height), 10), round(float(basket_inside_length), 10),
        round(float(active_wall_thickness), 10), round(float(basket_outside_length), 10),
        round(float(axial_stack_gap), 10), round(float(vessel_length_m), 10),
        round(float(vessel_dead_volume_m), 10),
        int(axial_stack_count), int(total_bags), int(n),
    )

def snapshot_3d_signature(snapshot):
    return (
        tuple(
            (round(float(z), 10), round(float(w), 10), round(float(h), 10), int(b))
            for z, w, h, b in snapshot["rows"]
        ),
        round(float(snapshot["diameter"]), 10),
        round(float(snapshot["clearance"]), 10), round(float(snapshot["heater_height"]), 10),
        round(float(snapshot["basket_inside_length"]), 10),
        round(float(snapshot["basket_wall_thickness"]), 10),
        round(float(snapshot["basket_outside_length"]), 10),
        round(float(snapshot["axial_stack_gap"]), 10), round(float(snapshot["vessel_length_m"]), 10),
        round(float(snapshot["vessel_dead_volume_m"]), 10),
        int(snapshot["axial_stack_count"]), int(snapshot["total_bags"]), int(snapshot["n"]),
    )

payload_needs_refresh = current_3d_signature() != snapshot_3d_signature(snapshot_3d)

a1, a2, a3, a4, a5, a6, a7 = st.columns(7)
a1.metric("Stacks", f"{stack_count_display}")
a2.metric("Baskets per payload", f"{total_vessel_baskets}")
a3.metric("Bags per payload", f"{total_vessel_bags}")
a4.metric("Basket outside length", f"{basket_outside_length_display:.2f} in")
a5.metric("Clearance-adjusted area utilization", f"{clearance_adjusted_utilization_3d:.2f}%")
a6.metric("Full vessel utilization", f"{full_vessel_utilization_3d:.2f}%")
a7.metric("Vessel dead space", f"{basket_dead_space:.2f} in")

capacity_input_col, capacity_output_col, view_toggle_col, capacity_spacer = st.columns(
    [1.0, 1.0, 1.45, 2.55]
)
with capacity_input_col:
    cycle_time_minutes = persistent_number_input(
        "Cycle time (min)",
        "cycle_time_minutes_saved",
        34.0,
        0.01,
        0.5,
    )
capacity_mt_per_hr = (
    (60.0 / cycle_time_minutes)
    * ((stack_count_display * 2.2 * total_bags) / 1000.0)
)
with capacity_output_col:
    st.metric("Capacity (MT/hr)", f"{capacity_mt_per_hr:.3f}")
with view_toggle_col:
    view_mode = st.radio(
        "3D view",
        ["Trolley", "Payload"],
        horizontal=True,
        help=(
            "Trolley shows one optimized stack inside its matching vessel section. "
            "Payload shows every axial stack in the full vessel."
        ),
    )

view_needs_refresh = view_mode == "Payload" and payload_needs_refresh

view_title_col, refresh_col = st.columns([0.78, 0.22])
with view_title_col:
    st.markdown(
        "**Interactive trolley view**"
        if view_mode == "Trolley"
        else "**Interactive payload view**"
    )
with refresh_col:
    if view_mode == "Payload":
        st.button("Refresh 3D model", type="primary", on_click=refresh_3d_model, use_container_width=True)
if view_needs_refresh:
    st.caption("3D view needs refresh to match the current inputs and live outputs.")
# Center the 3D viewport and trim its horizontal footprint for easier page scrolling.
plot_left, plot_center, plot_right = st.columns([0.10, 0.80, 0.10])
with plot_center:
    st.plotly_chart(
        trolley_figure if view_mode == "Trolley" else fig3d,
        use_container_width=True,
        config={"scrollZoom": True},
        key="combined_3d_visualization",
    )


plt.close(fig)

# ----------------------------
# Detailed table
# ----------------------------
st.subheader("Basket dimensions and positions")

data = []
for i, row in enumerate(row_details, start=1):
    data.append({
        "Basket #": i,
        "Width (in)": round(row["width"], 4),
        "Length (in)": round(basket_outside_length, 4),
        "Inside height (in)": round(row["inside_height"], 4),
        "Height (in)": round(row["height"], 4),
        "Bags": int(row["bags"]),
        "Basket bottom Z (in)": round(row["bottom_z"], 4),
    })

st.dataframe(
    data,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Basket #": st.column_config.NumberColumn(
            "Basket #", width="small"
        ),
        "Width (in)": st.column_config.NumberColumn(width="small", format="%.2f"),
        "Length (in)": st.column_config.NumberColumn(width="small", format="%.2f"),
        "Inside height (in)": st.column_config.NumberColumn(width="small", format="%.2f"),
        "Height (in)": st.column_config.NumberColumn(width="small", format="%.2f"),
        "Bags": st.column_config.NumberColumn("Bags", width="small"),
        "Basket bottom Z (in)": st.column_config.NumberColumn(
            "Basket bottom Z (in)",
            width="small", format="%.2f"
        ),
    },
)

# ----------------------------
# PDF configuration export
# ----------------------------
def render_3d_snapshot_png(snapshot):
    """Render a dependency-free static 3D image for the PDF with Matplotlib."""
    rows_3d = snapshot["rows"]
    diameter_3d = float(snapshot["diameter"])
    radius_3d = diameter_3d / 2.0
    vessel_length_3d = float(snapshot["vessel_length_in"])
    carrier_length_3d = float(snapshot["basket_outside_length"])
    stack_pitch_3d = float(snapshot["axial_stack_pitch"])
    stack_count_3d = int(snapshot["axial_stack_count"])
    axial_start_3d = float(snapshot["axial_start_y"])

    static_fig = plt.figure(figsize=(11.2, 6.4), facecolor="white")
    static_ax = static_fig.add_subplot(111, projection="3d")

    theta = np.linspace(0.0, 2.0 * math.pi, 64)
    y_shell = np.array([0.0, vessel_length_3d])
    theta_grid, y_grid = np.meshgrid(theta, y_shell)
    x_shell = radius_3d * np.cos(theta_grid)
    z_shell = radius_3d + radius_3d * np.sin(theta_grid)
    static_ax.plot_surface(
        x_shell, y_grid, z_shell,
        color="lightgray", alpha=0.10, linewidth=0, shade=False,
    )
    for y_end in (0.0, vessel_length_3d):
        static_ax.plot(
            radius_3d * np.cos(theta),
            np.full_like(theta, y_end),
            radius_3d + radius_3d * np.sin(theta),
            color="gray", linewidth=1.8,
        )

    def add_box(x0, x1, y0, y1, z0, z1, color):
        vertices = [
            (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
            (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1),
        ]
        faces = [
            [vertices[i] for i in face]
            for face in (
                (0, 1, 2, 3), (4, 5, 6, 7),
                (0, 1, 5, 4), (3, 2, 6, 7),
                (0, 3, 7, 4), (1, 2, 6, 5),
            )
        ]
        static_ax.add_collection3d(Poly3DCollection(
            faces,
            facecolors=color,
            edgecolors="black",
            linewidths=1.35,
            alpha=1.0,
        ))

    for stack_index in range(stack_count_3d):
        y0 = axial_start_3d + stack_index * stack_pitch_3d
        y1 = y0 + carrier_length_3d
        for bottom_z, width, row_height, bag_class in rows_3d:
            color = {3: "green", 2: "blue", 1: "purple"}.get(bag_class, "gray")
            add_box(
                -width / 2.0, width / 2.0, y0, y1,
                bottom_z, bottom_z + row_height, color,
            )

    heater_height_3d = float(snapshot["heater_height"])
    if heater_height_3d > 0.0:
        heater_faces = [[
            (0.0, 0.0, 0.0),
            (0.0, vessel_length_3d, 0.0),
            (0.0, vessel_length_3d, heater_height_3d),
            (0.0, 0.0, heater_height_3d),
        ]]
        static_ax.add_collection3d(Poly3DCollection(
            heater_faces, facecolors="black", edgecolors="black",
            linewidths=1.0, alpha=0.22,
        ))

    static_ax.set_xlim(-radius_3d * 1.08, radius_3d * 1.08)
    static_ax.set_ylim(0.0, max(vessel_length_3d, 1.0))
    static_ax.set_zlim(0.0, diameter_3d)
    static_ax.set_xlabel("X width (in)")
    static_ax.set_ylabel("Y length (in)")
    static_ax.set_zlabel("Z height (in)")
    static_ax.view_init(elev=20, azim=-62)
    try:
        static_ax.set_box_aspect((diameter_3d, max(vessel_length_3d, 1.0), diameter_3d))
    except AttributeError:
        pass
    static_fig.tight_layout()
    image_buffer = io.BytesIO()
    static_fig.savefig(image_buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(static_fig)
    image_buffer.seek(0)
    return image_buffer.getvalue()


def build_configuration_pdf(include_3d=True, notes=""):
    """Build a PDF summary in memory using a Chrome-free static 3D renderer."""
    pdf_buffer = io.BytesIO()
    three_d_png = None
    three_d_note = None

    if include_3d:
        try:
            three_d_png = render_3d_snapshot_png(snapshot_3d)
        except Exception as exc:
            three_d_note = (
                f"3D image omitted: {type(exc).__name__}: {exc}. "
                "The configuration data remains complete."
            )

    with PdfPages(pdf_buffer) as pdf:
        # Page 1 - configuration and results
        page = plt.figure(figsize=(8.5, 11))
        page.patch.set_facecolor("white")
        page.text(0.07, 0.955, "Tatermaxxer - Configuration Summary", fontsize=17, weight="bold")
        opening_sentence = (
            f"Vessel diameter of {diameter:.2f} in, with a minimum wall clearance of {clearance:.2f} in "
            f"and an internal heater space budget of {heater_height:.2f} in."
        )
        page.text(0.07, 0.920, opening_sentence, fontsize=9.5, wrap=True)

        notes_text = str(notes).strip()
        if notes_text:
            wrapped_note_lines = []
            for raw_line in notes_text.splitlines():
                if raw_line.strip():
                    wrapped_note_lines.extend(
                        textwrap.wrap(raw_line, width=100) or [""]
                    )
                else:
                    wrapped_note_lines.append("")
            page.text(
                0.07, 0.885, "Notes", fontsize=10.5, weight="bold", va="top"
            )
            page.text(
                0.07,
                0.861,
                "\n".join(wrapped_note_lines),
                fontsize=9.0,
                va="top",
                linespacing=1.25,
            )
            y = 0.825 - 0.016 * max(0, len(wrapped_note_lines) - 1)
        else:
            y = 0.870

        def section(title, items):
            nonlocal y
            page.text(0.07, y, title, fontsize=11.5, weight="bold")
            y -= 0.030
            label_x = 0.08
            value_x = 0.49
            for label, value in items:
                label_lines = textwrap.wrap(str(label), width=32) or [""]
                value_lines = textwrap.wrap(str(value), width=38) or [""]
                row_lines = max(len(label_lines), len(value_lines))
                label_text = "\n".join(
                    f"{line}:" if i == len(label_lines) - 1 else line
                    for i, line in enumerate(label_lines)
                )
                page.text(label_x, y, label_text, fontsize=8.7, weight="bold", va="top", linespacing=1.15)
                page.text(value_x, y, "\n".join(value_lines), fontsize=8.7, va="top", linespacing=1.15)
                y -= 0.019 * row_lines + 0.006
            y -= 0.014

        geometry_items = [
            ("Load carrier", "Baskets"),
            ("Bag length (Y)", f"{bag_length:.2f} in"),
            ("Bag width (X)", f"{bag_width:.2f} in"),
            ("Basket wall thickness", f"{basket_wall_thickness:.2f} in"),
            ("3-bag inside height", f"{inside_height_3:.2f} in"),
            ("3-bag outside height", f"{class_specs[3]['height']:.2f} in"),
            ("2-bag inside height", f"{inside_height_2:.2f} in"),
            ("2-bag outside height", f"{class_specs[2]['height']:.2f} in"),
            ("1-bag inside height", f"{inside_height_1:.2f} in"),
            ("1-bag outside height", f"{class_specs[1]['height']:.2f} in"),
            ("Minimum 3-bag basket width", f"{minimum_3_bag_basket_width:.2f} in"),
            ("Minimum 2-bag basket width", f"{minimum_2_bag_basket_width:.2f} in"),
            ("Minimum 1-bag basket width", f"{minimum_1_bag_basket_width:.2f} in"),
            ("Simplify manufacturing", "On" if simplify_manufacturing else "Off"),
            ("Manufacturing width increment", f"{manufacturing_increment:.2f} in"),
        ]
        section("2D configuration", [
            ("Vessel diameter", f"{diameter:.2f} in"),
            ("Minimum wall clearance", f"{clearance:.2f} in"),
            ("Internal heater height + clearance", f"{heater_height:.2f} in"),
            *geometry_items,
            ("Row count mode", row_mode),
            ("Requested row count", int(requested_rows) if row_mode == "Specify row count" else "N/A"),
        ])
        section("Results", [
            ("Baskets per stack", n),
            ("Bags per stack", total_bags),
            ("Stacks", stack_count_display),
            ("Baskets per payload", total_vessel_baskets),
            ("Bags per payload", total_vessel_bags),
            ("Clearance-adjusted utilization", f"{clearance_adjusted_utilization_3d:.2f}%"),
            ("Cycle time", f"{cycle_time_minutes:.2f} min"),
            ("Capacity (MT/hr)", f"{capacity_mt_per_hr:.3f}"),
        ])
        section("3D / payload configuration", [
            ("Basket inside length", f"{basket_inside_length:.2f} in"),
            ("Basket outside length", f"{basket_outside_length_display:.2f} in"),
            ("Gap between stacks", f"{axial_stack_gap:.2f} in"),
            ("Specify number of stacks", "On" if specify_axial_stacks else "Off"),
            ("Requested stacks", int(requested_axial_stacks) if specify_axial_stacks else "N/A"),
            ("Vessel length", f"{vessel_length_m:.3f} m"),
            ("Vessel dead volume", f"{vessel_dead_volume_m:.3f} m"),
            ("Cycle time", f"{cycle_time_minutes:.2f} min"),
            ("Basket dead space", f"{basket_dead_space:.2f} in"),
        ])
        note_y = max(y, 0.055)
        if payload_needs_refresh:
            page.text(
                0.07, note_y,
                "Note: the 3D screenshot reflects the last refreshed 3D view, while the numeric summary above reflects the live configuration.",
                fontsize=8.3, style="italic", wrap=True,
            )
        elif three_d_note:
            page.text(0.07, note_y, three_d_note, fontsize=8.3, style="italic", wrap=True)
        pdf.savefig(page, bbox_inches="tight")
        plt.close(page)

        # Page 2 - visual layouts side by side when the 3D capture is available.
        two_d_buffer = io.BytesIO()
        fig.savefig(two_d_buffer, format="png", dpi=170, bbox_inches="tight")
        two_d_buffer.seek(0)
        two_d_img = plt.imread(two_d_buffer, format="png")

        visuals_page = plt.figure(figsize=(11, 8.5))
        if three_d_png is not None:
            ax_2d = visuals_page.add_axes([0.035, 0.10, 0.43, 0.80])
            ax_3d = visuals_page.add_axes([0.515, 0.10, 0.45, 0.80])
            ax_2d.axis("off")
            ax_3d.axis("off")
            ax_2d.set_title("2D Cross-Sectional Layout", fontsize=13, pad=10)
            ax_3d.set_title("3D Payload Layout - Last Refreshed View", fontsize=13, pad=10)
            ax_2d.imshow(two_d_img)
            img_3d = plt.imread(io.BytesIO(three_d_png), format="png")
            ax_3d.imshow(img_3d)
        else:
            ax_2d = visuals_page.add_axes([0.14, 0.08, 0.72, 0.84])
            ax_2d.axis("off")
            ax_2d.set_title("2D Cross-Sectional Layout", fontsize=14, pad=10)
            ax_2d.imshow(two_d_img)
        pdf.savefig(visuals_page, bbox_inches="tight")
        plt.close(visuals_page)

        # Final page - basket dimensions table
        table_page = plt.figure(figsize=(8.5, 11))
        table_page.text(0.06, 0.96, "Basket Dimensions and Positions", fontsize=14, weight="bold")
        ax_table = table_page.add_axes([0.05, 0.07, 0.90, 0.84])
        ax_table.axis("off")
        headers = ["Basket #", "Width", "Length", "Inside H", "Height", "Bags", "Bottom Z"]
        rows_pdf = [[
            row["Basket #"],
            f'{row["Width (in)"]:.2f}',
            f'{row["Length (in)"]:.2f}',
            f'{row["Inside height (in)"]:.2f}',
            f'{row["Height (in)"]:.2f}',
            row["Bags"],
            f'{row["Basket bottom Z (in)"]:.2f}',
        ] for row in data]
        table = ax_table.table(
            cellText=rows_pdf, colLabels=headers, cellLoc="center", colLoc="center", loc="upper center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.25)
        pdf.savefig(table_page, bbox_inches="tight")
        plt.close(table_page)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue(), three_d_png is not None, three_d_note

st.markdown("**Configuration export**")
pdf_filename = st.text_input(
    "PDF file name",
    value="tatermaxxer_configuration",
    help="Enter the desired download name. The .pdf extension is added automatically if omitted.",
).strip()
if not pdf_filename:
    pdf_filename = "tatermaxxer_configuration"
if not pdf_filename.lower().endswith(".pdf"):
    pdf_filename += ".pdf"

pdf_notes = st.text_area(
    "Notes",
    value="",
    placeholder="Enter optional notes to include near the top of the report.",
    help="Plain text entered here is included near the top of page one.",
)

include_3d_pdf = st.checkbox(
    "Include 3D screenshot",
    value=True,
    help=(
        "Creates a static Matplotlib rendering of the last refreshed payload. "
        "This does not require Kaleido, Chrome, or Chromium."
    ),
)

# Generate only when requested so normal app reruns do not build the static PDF image.
if st.button("Prepare configuration PDF"):
    pdf_bytes, included_3d, pdf_3d_note = build_configuration_pdf(
        include_3d=include_3d_pdf,
        notes=pdf_notes,
    )
    st.session_state["configuration_pdf_bytes"] = pdf_bytes
    st.session_state["configuration_pdf_has_3d"] = included_3d
    st.session_state["configuration_pdf_note"] = pdf_3d_note
    st.session_state["configuration_pdf_notes_source"] = pdf_notes

if "configuration_pdf_bytes" in st.session_state:
    if pdf_notes != st.session_state.get("configuration_pdf_notes_source", ""):
        st.caption(
            "Notes have changed. Prepare the configuration PDF again to include the revised notes."
        )
    elif st.session_state.get("configuration_pdf_note"):
        st.caption(st.session_state["configuration_pdf_note"])
    elif st.session_state.get("configuration_pdf_has_3d"):
        st.caption("PDF prepared with the current configuration and a static rendering of the last refreshed payload. PDF must be prepared after each revision.")
    else:
        st.caption("PDF prepared without a 3D screenshot.")

    st.download_button(
        "Download configuration PDF",
        data=st.session_state["configuration_pdf_bytes"],
        file_name=pdf_filename,
        mime="application/pdf",
        disabled=(
            pdf_notes
            != st.session_state.get("configuration_pdf_notes_source", "")
        ),
    )
