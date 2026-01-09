import numpy as np
import networkx as nx
from typing import Optional, Sequence
from numpydantic import NDArray
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, LinearRing
from shapely.geometry.base import BaseGeometry
import matplotlib.pyplot as plt
import os

from pdm4ar.exercises.ex14.global_config import GlobalConfig
from pdm4ar.exercises.ex14.agent_state import AgentState
from dg_commons.sim.models.obstacles import StaticObstacle


def plot_graph(
    G: Optional[nx.Graph] = None,
    filename: Optional[str] = None,
    sub_dir: Optional[str] = None,
    planned_path: Optional[list[tuple[float, float]]] = None,
    actual_path: Optional[list[tuple[float, float]]] = None,
    obstacles: Optional[Sequence[StaticObstacle]] = None,
    traversable_space: Optional[BaseGeometry] = None,
    other_agents: Optional[dict[str, "AgentState"]] = None,
    own_state: Optional["AgentState"] = None,
):

    fig, ax = plt.subplots(figsize=(10, 10))

    # Helper function to prevent duplicate legend entries
    def plot_coords(x, y, color, linestyle, linewidth, label, fill_color=None, alpha=None):
        current_labels = [l for l in ax.get_legend_handles_labels()[1]]
        lbl = label if label not in current_labels else ""
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=lbl, zorder=3)
        if fill_color:
            ax.fill(x, y, color=fill_color, alpha=alpha, zorder=2)

    if obstacles:
        # --- 1. Physical World (Solid Black Lines) ---
        boundary_wrapper = obstacles[-1]
        boundary_shape = boundary_wrapper.shape

        if isinstance(boundary_shape, LinearRing):
            x, y = boundary_shape.xy
        else:
            x, y = boundary_shape.exterior.xy

        plot_coords(x, y, "black", "-", 2.0, "Physical Obstacles/Bounds")

        for obs in obstacles[:-1]:
            poly = obs.shape
            if poly.is_empty:
                continue
            geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
            for p in geoms:
                x, y = p.exterior.xy
                plot_coords(x, y, "black", "-", 2.0, "Physical Obstacles/Bounds", fill_color="gray", alpha=0.5)

    if traversable_space:
        # --- 2. Navigable Space Limits (Dashed Black Lines) ---
        if not traversable_space.is_empty:
            geoms = traversable_space.geoms if isinstance(traversable_space, MultiPolygon) else [traversable_space]

            for p in geoms:
                x, y = p.exterior.xy
                plot_coords(x, y, "black", "--", 1.5, "Inflated Obstacles/Bounds")
                for interior in p.interiors:
                    xi, yi = interior.xy
                    plot_coords(xi, yi, "black", "--", 1.5, "Inflated Obstacles/Bounds")
                ax.fill(x, y, color="lightgreen", alpha=0.1, zorder=0)

    if G:
        # --- 3. Graph ---
        edge_x, edge_y = [], []
        for u, v in G.edges():
            edge_x.extend([u[0], v[0], None])
            edge_y.extend([u[1], v[1], None])
        ax.plot(edge_x, edge_y, "b-", alpha=0.2, linewidth=0.5, label="Graph Edges")

        # --- MODIFIED SECTION: Split nodes by 'type' attribute ---
        grid_x, grid_y = [], []
        goal_x, goal_y = [], []
        collection_x, collection_y = [], []

        for node, data in G.nodes(data=True):
            # Check the 'type' attribute string instead of boolean flags
            node_type = data.get("type")

            if node_type == "goal":
                goal_x.append(node[0])
                goal_y.append(node[1])
            elif node_type == "collection":
                collection_x.append(node[0])
                collection_y.append(node[1])
            else:
                grid_x.append(node[0])
                grid_y.append(node[1])

        if grid_x:
            ax.scatter(grid_x, grid_y, c="blue", s=4, alpha=1, label="Nodes", zorder=4)

        if goal_x:
            ax.scatter(
                goal_x, goal_y, c="yellow", s=150, edgecolors="black", marker="*", alpha=1, label="Goals", zorder=5
            )

        if collection_x:
            ax.scatter(collection_x, collection_y, c="blue", s=70, marker="o", alpha=1, label="Collections", zorder=5)

    if planned_path:
        # --- 4. Path ---
        px, py = zip(*planned_path)
        ax.plot(px, py, "r-", linewidth=2.5, zorder=5, label="Planned Path")
        ax.plot(px[0], py[0], "go", markersize=8, zorder=6, label="Start")
        ax.plot(px[-1], py[-1], "rx", markersize=8, zorder=6, label="Goal")

    if actual_path:
        # --- 4. Path ---
        px, py = zip(*actual_path)
        ax.plot(px, py, "b-", linewidth=2.5, zorder=5, label="Actual Path")
        ax.plot(px[0], py[0], "go", markersize=8, zorder=6, label="Start")
        ax.plot(px[-1], py[-1], "rx", markersize=8, zorder=6, label="Goal")

    if other_agents:
        for name, state in other_agents.items():
            # Legend deduplication
            current_labels = [l for l in ax.get_legend_handles_labels()[1]]
            lbl = "Other Agents" if "Other Agents" not in current_labels else ""

            # Plot Body (Circle)
            # Using darkorange/brown color to distinguish from goals (orange stars) and path (red)
            circle = plt.Circle((state.x, state.y), state.radius, color="saddlebrown", alpha=0.7, zorder=6, label=lbl)
            ax.add_patch(circle)

            # Plot Heading (Line)
            arrow_len = state.radius
            heading = state.heading
            end_x = state.x + arrow_len * np.cos(heading)
            end_y = state.y + arrow_len * np.sin(heading)
            ax.plot([state.x, end_x], [state.y, end_y], "k-", linewidth=1.5, zorder=7)

    if own_state:
        current_labels = [l for l in ax.get_legend_handles_labels()[1]]
        lbl = "Other Agents" if "Other Agents" not in current_labels else ""

        # Plot Body (Circle)
        # Using darkorange/brown color to distinguish from goals (orange stars) and path (red)
        circle = plt.Circle(
            (own_state.x, own_state.y), own_state.radius, color="orange", alpha=0.7, zorder=6, label=lbl
        )
        ax.add_patch(circle)

        # Plot Heading (Line)
        arrow_len = own_state.radius
        heading = own_state.heading
        end_x = own_state.x + arrow_len * np.cos(heading)
        end_y = own_state.y + arrow_len * np.sin(heading)
        ax.plot([own_state.x, end_x], [own_state.y, end_y], "k-", linewidth=1.5, zorder=7)

    # --- Formatting & Saving ---
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize="small", framealpha=0.9)

    try:
        file_dir = GlobalConfig.SESSION_FOLDER

        if not filename:
            filename = f"{np.random.random_sample():.3f}.png"

        if sub_dir:
            file_dir = os.path.join(file_dir, sub_dir)

        file_path = os.path.join(file_dir, f"{filename}.png")
        os.makedirs(file_dir, exist_ok=True)
        plt.savefig(file_path, dpi=300)

        if GlobalConfig.OUTPUT_VERBOSE:
            print(f"Graph visualization saved to {file_path}")
    except Exception as e:
        print(f"  [Exception]  Whoopsie Dasie  ({e})")

    plt.close(fig)
