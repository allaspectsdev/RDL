"""
Visualization Builder Module - 22+ interactive chart types with full customization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CHART_TYPES = [
    "Scatter Plot", "Line Chart", "Bar Chart", "Histogram",
    "Box Plot", "Violin Plot", "Strip Plot", "Heatmap",
    "Bubble Chart", "Area Chart", "Pie / Donut", "Treemap",
    "Sunburst", "Radar Chart", "Parallel Coordinates",
    "3D Scatter", "3D Surface", "Contour Plot",
    "Funnel Chart", "Waterfall Chart", "Candlestick (OHLC)",
    "Joint Plot",
]


def render_visualization(df: pd.DataFrame):
    """Render the visualization builder."""
    if df is None or df.empty:
        st.warning("No data loaded.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = df.columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    chart_type = st.selectbox("Chart Type:", CHART_TYPES, key="viz_chart_type")

    # Common options
    with st.expander("Chart Settings", expanded=False):
        title = st.text_input("Title:", chart_type, key="viz_title")
        height = st.slider("Height:", 300, 1000, 500, 50, key="viz_height")
        color_scale = st.selectbox("Color palette:", [
            "plotly", "Set1", "Set2", "Pastel1", "Dark2",
            "Viridis", "Plasma", "Inferno", "Blues", "Reds", "RdBu",
        ], key="viz_palette")
        opacity = st.slider("Opacity:", 0.1, 1.0, 0.8, 0.05, key="viz_opacity")

    # ── Scatter Plot ──
    if chart_type == "Scatter Plot":
        c1, c2 = st.columns(2)
        x = c1.selectbox("X:", all_cols, key="sc_x")
        y = c2.selectbox("Y:", [c for c in num_cols if c != x] if x in num_cols else num_cols, key="sc_y")
        c1, c2, c3 = st.columns(3)
        color = c1.selectbox("Color:", [None] + all_cols, key="sc_color")
        size = c2.selectbox("Size:", [None] + num_cols, key="sc_size")
        facet = c3.selectbox("Facet:", [None] + cat_cols, key="sc_facet")
        trendline = st.selectbox("Trendline:", [None, "ols", "lowess"], key="sc_trend")
        marginal = st.selectbox("Marginals:", [None, "histogram", "box", "violin", "rug"], key="sc_marg")

        fig = px.scatter(df, x=x, y=y, color=color, size=size,
                         facet_col=facet, trendline=trendline,
                         marginal_x=marginal, marginal_y=marginal,
                         title=title, opacity=opacity,
                         color_continuous_scale=color_scale if color and color in num_cols else None)
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)

    # ── Line Chart ──
    elif chart_type == "Line Chart":
        x = st.selectbox("X:", all_cols, key="ln_x")
        y_cols = st.multiselect("Y (multiple):", num_cols, default=num_cols[:1], key="ln_y")
        color = st.selectbox("Group/Color:", [None] + cat_cols, key="ln_color")
        markers = st.checkbox("Show markers", value=False, key="ln_markers")

        if y_cols:
            if len(y_cols) == 1 and color:
                fig = px.line(df, x=x, y=y_cols[0], color=color, markers=markers,
                              title=title)
            else:
                fig = go.Figure()
                for yc in y_cols:
                    fig.add_trace(go.Scatter(x=df[x], y=df[yc], name=yc,
                                             mode="lines+markers" if markers else "lines"))
                fig.update_layout(title=title, xaxis_title=x)
            fig.update_layout(height=height)
            st.plotly_chart(fig, use_container_width=True)

    # ── Bar Chart ──
    elif chart_type == "Bar Chart":
        x = st.selectbox("X:", all_cols, key="bar_x")
        y = st.selectbox("Y:", [None] + num_cols, key="bar_y")
        color = st.selectbox("Color:", [None] + cat_cols, key="bar_color")
        c1, c2 = st.columns(2)
        orientation = c1.selectbox("Orientation:", ["vertical", "horizontal"], key="bar_orient")
        barmode = c2.selectbox("Mode:", ["group", "stack", "relative"], key="bar_mode")

        orient = "v" if orientation == "vertical" else "h"
        if y:
            fig = px.bar(df, x=x if orient == "v" else y, y=y if orient == "v" else x,
                         color=color, barmode=barmode, title=title,
                         orientation=orient, opacity=opacity)
        else:
            counts = df[x].value_counts().reset_index()
            counts.columns = [x, "count"]
            fig = px.bar(counts, x=x if orient == "v" else "count",
                         y="count" if orient == "v" else x,
                         title=title, orientation=orient, opacity=opacity)
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)

    # ── Histogram ──
    elif chart_type == "Histogram":
        x = st.selectbox("Column:", num_cols, key="hist_x")
        c1, c2, c3 = st.columns(3)
        nbins = c1.slider("Bins:", 5, 200, 30, key="hist_bins")
        color = c2.selectbox("Color by:", [None] + cat_cols, key="hist_color")
        barmode = c3.selectbox("Mode:", ["overlay", "stack", "group"], key="hist_mode")
        cumulative = st.checkbox("Cumulative", value=False, key="hist_cum")
        show_kde = st.checkbox("KDE overlay", value=False, key="hist_kde")

        fig = px.histogram(df, x=x, nbins=nbins, color=color, barmode=barmode,
                           cumulative=cumulative, title=title, opacity=opacity,
                           marginal="rug")
        if show_kde and not cumulative:
            data = df[x].dropna()
            kde_x = np.linspace(data.min(), data.max(), 200)
            try:
                kde = stats.gaussian_kde(data)
                bin_width = (data.max() - data.min()) / nbins
                kde_y = kde(kde_x) * len(data) * bin_width
                fig.add_trace(go.Scatter(x=kde_x, y=kde_y, name="KDE",
                                         line=dict(color="red", width=2)))
            except Exception:
                pass
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)

    # ── Box Plot ──
    elif chart_type == "Box Plot":
        y = st.selectbox("Value:", num_cols, key="box_y")
        x = st.selectbox("Group by:", [None] + cat_cols, key="box_x")
        color = st.selectbox("Color:", [None] + cat_cols, key="box_color")
        c1, c2 = st.columns(2)
        points = c1.selectbox("Points:", ["outliers", "all", False], key="box_pts")
        notched = c2.checkbox("Notched", value=False, key="box_notch")

        fig = px.box(df, x=x, y=y, color=color, points=points, notched=notched,
                     title=title)
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)

    # ── Violin Plot ──
    elif chart_type == "Violin Plot":
        y = st.selectbox("Value:", num_cols, key="vio_y")
        x = st.selectbox("Group by:", [None] + cat_cols, key="vio_x")
        color = st.selectbox("Color:", [None] + cat_cols, key="vio_color")
        box = st.checkbox("Show box", value=True, key="vio_box")
        points = st.selectbox("Points:", [False, "all", "outliers"], key="vio_pts")

        fig = px.violin(df, x=x, y=y, color=color, box=box, points=points,
                        title=title)
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)

    # ── Strip Plot ──
    elif chart_type == "Strip Plot":
        y = st.selectbox("Value:", num_cols, key="strip_y")
        x = st.selectbox("Group by:", [None] + cat_cols, key="strip_x")
        color = st.selectbox("Color:", [None] + cat_cols, key="strip_color")

        fig = px.strip(df, x=x, y=y, color=color, title=title)
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap ──
    elif chart_type == "Heatmap":
        selected = st.multiselect("Columns:", num_cols, default=num_cols[:6], key="heat_cols")
        if len(selected) >= 2:
            method = st.selectbox("Values:", ["Correlation", "Raw Mean", "Raw Data"], key="heat_method")
            if method == "Correlation":
                matrix = df[selected].corr()
            elif method == "Raw Mean":
                matrix = df[selected].describe().loc[["mean", "std", "min", "max"]]
            else:
                matrix = df[selected].head(50)

            fig = px.imshow(matrix, text_auto=".2f", color_continuous_scale=color_scale,
                            title=title, aspect="auto")
            fig.update_layout(height=height)
            st.plotly_chart(fig, use_container_width=True)

    # ── Bubble Chart ──
    elif chart_type == "Bubble Chart":
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X:", num_cols, key="bub_x")
        y = c2.selectbox("Y:", [c for c in num_cols if c != x], key="bub_y")
        size = c3.selectbox("Size:", num_cols, key="bub_size")
        color = st.selectbox("Color:", [None] + all_cols, key="bub_color")

        fig = px.scatter(df, x=x, y=y, size=size, color=color,
                         title=title, opacity=opacity, size_max=60)
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)

    # ── Area Chart ──
    elif chart_type == "Area Chart":
        x = st.selectbox("X:", all_cols, key="area_x")
        y_cols = st.multiselect("Y:", num_cols, default=num_cols[:2], key="area_y")
        stacked = st.checkbox("Stacked", value=True, key="area_stack")

        if y_cols:
            fig = px.area(df, x=x, y=y_cols, title=title,
                          groupnorm="percent" if stacked else None)
            fig.update_layout(height=height)
            st.plotly_chart(fig, use_container_width=True)

    # ── Pie / Donut ──
    elif chart_type == "Pie / Donut":
        names = st.selectbox("Labels:", cat_cols if cat_cols else all_cols, key="pie_names")
        values = st.selectbox("Values:", [None] + num_cols, key="pie_vals")
        hole = st.slider("Hole (0=pie, >0=donut):", 0.0, 0.7, 0.0, 0.05, key="pie_hole")

        if values:
            fig = px.pie(df, names=names, values=values, title=title, hole=hole)
        else:
            counts = df[names].value_counts()
            fig = px.pie(values=counts.values, names=counts.index.astype(str),
                         title=title, hole=hole)
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)

    # ── Treemap ──
    elif chart_type == "Treemap":
        path_cols = st.multiselect("Hierarchy (outer → inner):", cat_cols, key="tree_path")
        values = st.selectbox("Values:", [None] + num_cols, key="tree_vals")

        if path_cols:
            fig = px.treemap(df, path=path_cols, values=values, title=title)
            fig.update_layout(height=height)
            st.plotly_chart(fig, use_container_width=True)

    # ── Sunburst ──
    elif chart_type == "Sunburst":
        path_cols = st.multiselect("Hierarchy:", cat_cols, key="sun_path")
        values = st.selectbox("Values:", [None] + num_cols, key="sun_vals")

        if path_cols:
            fig = px.sunburst(df, path=path_cols, values=values, title=title)
            fig.update_layout(height=height)
            st.plotly_chart(fig, use_container_width=True)

    # ── Radar Chart ──
    elif chart_type == "Radar Chart":
        selected = st.multiselect("Variables:", num_cols, default=num_cols[:5], key="radar_cols")
        group_col = st.selectbox("Group by:", [None] + cat_cols, key="radar_group")

        if len(selected) >= 3:
            fig = go.Figure()
            if group_col:
                for name, group in df.groupby(group_col):
                    means = group[selected].mean()
                    # Normalize to 0-1 for radar
                    mins = df[selected].min()
                    maxs = df[selected].max()
                    ranges = maxs - mins
                    ranges = ranges.replace(0, 1)
                    normalized = (means - mins) / ranges
                    fig.add_trace(go.Scatterpolar(
                        r=normalized.tolist() + [normalized.tolist()[0]],
                        theta=selected + [selected[0]],
                        name=str(name), fill="toself", opacity=0.5,
                    ))
            else:
                means = df[selected].mean()
                mins = df[selected].min()
                maxs = df[selected].max()
                ranges = maxs - mins
                ranges = ranges.replace(0, 1)
                normalized = (means - mins) / ranges
                fig.add_trace(go.Scatterpolar(
                    r=normalized.tolist() + [normalized.tolist()[0]],
                    theta=selected + [selected[0]],
                    fill="toself", name="Mean",
                ))
            fig.update_layout(title=title, height=height,
                              polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
            st.plotly_chart(fig, use_container_width=True)

    # ── Parallel Coordinates ──
    elif chart_type == "Parallel Coordinates":
        selected = st.multiselect("Variables:", num_cols, default=num_cols[:5], key="pc_cols")
        color_col = st.selectbox("Color by:", [None] + num_cols + cat_cols, key="pc_color")

        if len(selected) >= 2:
            if color_col and color_col in cat_cols:
                df_temp = df.copy()
                df_temp[f"{color_col}_code"] = df_temp[color_col].astype("category").cat.codes
                fig = px.parallel_coordinates(df_temp, dimensions=selected,
                                              color=f"{color_col}_code", title=title)
            else:
                fig = px.parallel_coordinates(df, dimensions=selected, color=color_col,
                                              title=title, color_continuous_scale=color_scale)
            fig.update_layout(height=height)
            st.plotly_chart(fig, use_container_width=True)

    # ── 3D Scatter ──
    elif chart_type == "3D Scatter":
        if len(num_cols) < 3:
            st.warning("Need at least 3 numeric columns.")
            return
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X:", num_cols, key="3d_x")
        y = c2.selectbox("Y:", [c for c in num_cols if c != x], key="3d_y")
        z = c3.selectbox("Z:", [c for c in num_cols if c not in (x, y)], key="3d_z")
        color = st.selectbox("Color:", [None] + all_cols, key="3d_color")

        fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, title=title, opacity=opacity)
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)

    # ── 3D Surface ──
    elif chart_type == "3D Surface":
        if len(num_cols) < 3:
            st.warning("Need at least 3 numeric columns.")
            return
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X:", num_cols, key="surf_x")
        y = c2.selectbox("Y:", [c for c in num_cols if c != x], key="surf_y")
        z = c3.selectbox("Z:", [c for c in num_cols if c not in (x, y)], key="surf_z")

        # Create grid
        try:
            from scipy.interpolate import griddata
            data = df[[x, y, z]].dropna()
            xi = np.linspace(data[x].min(), data[x].max(), 50)
            yi = np.linspace(data[y].min(), data[y].max(), 50)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((data[x].values, data[y].values), data[z].values,
                          (xi, yi), method="linear")
            fig = go.Figure(go.Surface(x=xi, y=yi, z=zi, colorscale=color_scale))
            fig.update_layout(title=title, height=height,
                              scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title=z))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Surface plot error: {e}")

    # ── Contour Plot ──
    elif chart_type == "Contour Plot":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
            return
        c1, c2 = st.columns(2)
        x = c1.selectbox("X:", num_cols, key="cont_x")
        y = c2.selectbox("Y:", [c for c in num_cols if c != x], key="cont_y")

        fig = px.density_contour(df, x=x, y=y, title=title)
        fig.update_traces(contours_coloring="fill")
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)

    # ── Funnel Chart ──
    elif chart_type == "Funnel Chart":
        stage = st.selectbox("Stage:", cat_cols if cat_cols else all_cols, key="fun_stage")
        value = st.selectbox("Value:", num_cols, key="fun_val")

        agg_data = df.groupby(stage)[value].sum().sort_values(ascending=False).reset_index()
        fig = px.funnel(agg_data, x=value, y=stage, title=title)
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)

    # ── Waterfall Chart ──
    elif chart_type == "Waterfall Chart":
        cat = st.selectbox("Categories:", cat_cols if cat_cols else all_cols, key="wf_cat")
        val = st.selectbox("Values:", num_cols, key="wf_val")

        agg = df.groupby(cat)[val].sum().reset_index()
        fig = go.Figure(go.Waterfall(
            x=agg[cat].astype(str), y=agg[val],
            connector=dict(line=dict(color="rgb(63,63,63)")),
        ))
        fig.update_layout(title=title, height=height)
        st.plotly_chart(fig, use_container_width=True)

    # ── Candlestick ──
    elif chart_type == "Candlestick (OHLC)":
        date = st.selectbox("Date:", date_cols + all_cols, key="ohlc_date")
        c1, c2, c3, c4 = st.columns(4)
        o = c1.selectbox("Open:", num_cols, key="ohlc_o")
        h = c2.selectbox("High:", num_cols, key="ohlc_h")
        l = c3.selectbox("Low:", num_cols, key="ohlc_l")
        c = c4.selectbox("Close:", num_cols, key="ohlc_c")

        fig = go.Figure(go.Candlestick(x=df[date], open=df[o], high=df[h],
                                        low=df[l], close=df[c]))
        fig.update_layout(title=title, height=height, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Joint Plot ──
    elif chart_type == "Joint Plot":
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
            return
        c1, c2 = st.columns(2)
        x = c1.selectbox("X:", num_cols, key="joint_x")
        y = c2.selectbox("Y:", [c for c in num_cols if c != x], key="joint_y")
        color = st.selectbox("Color:", [None] + cat_cols, key="joint_color")

        fig = px.scatter(df, x=x, y=y, color=color, marginal_x="histogram",
                         marginal_y="histogram", title=title, opacity=opacity)
        fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)
