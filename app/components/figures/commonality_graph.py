"""
Commonality visualizations for sample group overlaps.

Provides two representations:
- ``supervenn``: Supervenn diagram for up to ~10 sets
- ``common_heatmap``: Jaccard-like overlap heatmap for larger numbers
"""
from __future__ import annotations
import io
import base64
import matplotlib as mpl
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from supervenn import supervenn as svenn
from dash.dcc import Graph
from dash.html import Img
from pandas import DataFrame, Index
from plotly.express import imshow
from numpy import nan as NA

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Set, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class Intersection:
    """Representation of an intersection used for plotting."""

    bits: Tuple[bool, ...]
    label: str
    size: int


def _prepare_intersections(
    df: pd.DataFrame,
    membership_columns: Sequence[str],
    top_n: int | None,
    min_size: int,
) -> List[Intersection]:
    """
    Aggregate intersection sizes and return the most prominent entries.

    Args:
        df: Input dataframe where each membership column is boolean-like.
        membership_columns: Ordered set labels used to build combinations.
        top_n: Maximum number of intersections to keep (after filtering).
        min_size: Drop intersections smaller than this threshold.
    """
    normalized = df.loc[:, membership_columns].astype(bool)
    counts = (
        normalized.groupby(list(normalized.columns), dropna=False)
        .size()
        .sort_values(ascending=False)
    )

    intersections: List[Intersection] = []
    for bits, size in counts.items():
        if size < min_size:
            continue
        label = " & ".join(
            col for include, col in zip(bits, membership_columns) if include
        )
        # Avoid empty labels for all-zero combinations.
        label = label or "∅"
        intersections.append(Intersection(tuple(bits), label, int(size)))

    if top_n is not None:
        intersections = intersections[:top_n]

    return intersections


def _build_matrix_traces(
    intersections: Sequence[Intersection],
    membership_columns: Sequence[str],
) -> Iterable[go.Scatter]:
    """
    Draw the boolean matrix that indicates which sets participate per bar.
    """
    col_index = {name: idx for idx, name in enumerate(membership_columns)}
    xs: List[int] = []
    ys: List[int] = []
    marker_colors: List[str] = []
    line_traces: List[go.Scatter] = []

    for x, inter in enumerate(intersections):
        participating_indices: List[int] = []
        for name, include in zip(membership_columns, inter.bits):
            if not include:
                continue
            xs.append(x)
            ys.append(col_index[name])
            marker_colors.append("#222")
            participating_indices.append(col_index[name])

        if len(participating_indices) >= 2:
            line_traces.append(
                go.Scatter(
                    x=[x, x],
                    y=[min(participating_indices), max(participating_indices)],
                    mode="lines",
                    line=dict(color="#222", width=3),
                    hoverinfo="skip",
                )
            )

    scatter = go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(size=8, color=marker_colors, line=dict(color="#000", width=1)),
        hoverinfo="skip",
    )
    return [scatter, *line_traces]


def create_upset_figure(
    set_membership: Mapping[str, Iterable[object]],
    *,
    top_n: int | None = 15,
    min_size: int = 1,
    title: str = "UpSet Plot",
) -> go.Figure:
    """
    Build a Plotly UpSet graph representing set intersections.

    Args:
        set_membership: Mapping of set label to an iterable of hashable entries.
        top_n: Optional limit for the number of intersection bars to show.
        min_size: Minimum intersection size; smaller entries are dropped.
        title: Figure title.

    Returns:
        plotly.graph_objects.Figure configured with bar and matrix sections.
    """
    if not set_membership:
        raise ValueError("At least one set is required.")

    ordered_sets: List[str] = list(set_membership.keys())
    normalized_sets: dict[str, Set[object]] = {
        name: set(values) for name, values in set_membership.items()
    }

    entry_order: List[object] = []
    seen_entries: Set[object] = set()
    for values in normalized_sets.values():
        for entry in values:
            if entry not in seen_entries:
                seen_entries.add(entry)
                entry_order.append(entry)

    if not entry_order:
        raise ValueError("Provided sets are empty; cannot build UpSet plot.")

    df = pd.DataFrame(
        {
            name: [entry in normalized_sets[name] for entry in entry_order]
            for name in ordered_sets
        },
    )

    intersections = _prepare_intersections(
        df, ordered_sets, top_n=top_n, min_size=min_size
    )
    if not intersections:
        raise ValueError("No intersections left after filtering; relax criteria.")

    base_counts = [len(normalized_sets[name]) for name in ordered_sets]

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.2, 0.8],
        row_heights=[0.6, 0.4],
        specs=[[None, {}], [{}, {}]],
        shared_xaxes=False,
        vertical_spacing=0.03,
        horizontal_spacing=0.05,
    )

    fig.add_trace(
        go.Bar(
            x=[inter.label for inter in intersections],
            y=[inter.size for inter in intersections],
            marker_color="#444",
        hovertemplate="Intersection size: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    matrix_traces = _build_matrix_traces(intersections, ordered_sets)
    for trace in matrix_traces:
        fig.add_trace(trace, row=2, col=2)

    fig.add_trace(
        go.Bar(
            x=base_counts,
            y=list(range(len(ordered_sets))),
            orientation="h",
            marker_color="#888",
            opacity=0.35,
            customdata=ordered_sets,
            hovertemplate="%{customdata}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Intersection size", row=1, col=2)
    fig.update_yaxes(
        row=2,
        col=2,
        tickmode="array",
        tickvals=list(range(len(ordered_sets))),
        ticktext=ordered_sets,
        tickfont=dict(size=14),
        showticklabels=True,
    )
    fig.update_yaxes(
        row=2,
        col=1,
        tickmode="array",
        tickvals=list(range(len(ordered_sets))),
        ticktext=["" for _ in ordered_sets],
    )
    fig.update_xaxes(
        showticklabels=True,
        row=2,
        col=1,
        title_text="Number of proteins",
        autorange="reversed",
    )
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(
        row=2,
        col=2,
        showticklabels=False,
        tickmode="linear",
        dtick=1,
        showgrid=True,
        gridcolor="#e3e3e3",
        gridwidth=1,
        zeroline=False,
    )
    fig.update_xaxes(matches="x4", row=1, col=2)
    fig.update_layout(
        title=title,
        bargap=0.25,
        hovermode="closest",
        template="plotly_white",
        showlegend=False,
    )

    return fig

def make_graph(group_sets: dict, id_str: str, defaults: dict) -> tuple:
    """Choose and build an overlap visualization based on group count.

    :param group_sets: Mapping of group name to a set of identifiers.
    :param id_str: Component ID for the output figure.
    :param use_supervenn: If ``True``, prefer Supervenn when feasible.
    :param defaults: Figure defaults (expects ``height``, ``width``, ``config``).
    :returns: Tuple of (component, aux_data) where aux_data may be a base64 PDF.
    """
    upset_figure = create_upset_figure(group_sets)
    return (
        Graph(
            id=id_str,
            figure=upset_figure,
            config=defaults['config']
        ),
        ''
    )
