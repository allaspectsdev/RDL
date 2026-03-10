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
from modules.ui_helpers import grouped_chart_selector, empty_state


def render_visualization(df: pd.DataFrame):
    """Render the visualization builder."""
    if df is None or df.empty:
        empty_state("No data loaded.", "Upload a dataset from the sidebar to begin.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = df.columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    chart_type = grouped_chart_selector()

    # Common options
    with st.expander("Chart Settings", expanded=True):
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
                n_rows = len(df)
                matrix = df[selected].head(50)
                if n_rows > 50:
                    st.caption(f"Showing first 50 of {n_rows:,} rows.")

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

        n_unique = df[names].nunique()
        if n_unique > 30:
            st.warning(f"Too many categories ({n_unique}). Showing top 20 only.")
        if values:
            if n_unique > 30:
                top20 = df[names].value_counts().head(20).index
                plot_df = df[df[names].isin(top20)]
                fig = px.pie(plot_df, names=names, values=values, title=title, hole=hole)
            else:
                fig = px.pie(df, names=names, values=values, title=title, hole=hole)
        else:
            counts = df[names].value_counts()
            if n_unique > 30:
                counts = counts.head(20)
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

    # ── Mosaic Plot ──
    elif chart_type == "Mosaic Plot":
        if len(cat_cols) < 2:
            st.warning("Need at least 2 categorical columns for mosaic plot.")
            return
        c1, c2 = st.columns(2)
        var1 = c1.selectbox("Variable 1 (width):", cat_cols, key="mos_v1")
        var2 = c2.selectbox("Variable 2 (height):", [c for c in cat_cols if c != var1], key="mos_v2")

        ct = pd.crosstab(df[var1], df[var2])
        total = ct.values.sum()

        # Compute expected frequencies and Pearson residuals
        row_sums = ct.sum(axis=1)
        col_sums = ct.sum(axis=0)
        expected = np.outer(row_sums, col_sums) / total
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals = np.where(expected > 0, (ct.values - expected) / np.sqrt(expected), 0)

        # Build rectangles
        fig = go.Figure()
        x_start = 0
        categories_v1 = ct.index.tolist()
        categories_v2 = ct.columns.tolist()

        for i, cat1 in enumerate(categories_v1):
            width = row_sums.iloc[i] / total
            y_start = 0
            for j, cat2 in enumerate(categories_v2):
                height_val = ct.iloc[i, j] / row_sums.iloc[i] if row_sums.iloc[i] > 0 else 0
                res = residuals[i, j]

                # Color based on residual
                if res > 2:
                    color = "rgba(99, 102, 241, 0.8)"  # Strong positive
                elif res > 0:
                    color = "rgba(99, 102, 241, 0.4)"  # Mild positive
                elif res > -2:
                    color = "rgba(239, 68, 68, 0.4)"   # Mild negative
                else:
                    color = "rgba(239, 68, 68, 0.8)"   # Strong negative

                fig.add_shape(type="rect",
                              x0=x_start, x1=x_start + width,
                              y0=y_start, y1=y_start + height_val,
                              line=dict(color="white", width=1),
                              fillcolor=color)
                # Label
                if width > 0.05 and height_val > 0.05:
                    fig.add_annotation(
                        x=x_start + width / 2, y=y_start + height_val / 2,
                        text=f"{ct.iloc[i, j]}<br>({res:.1f})",
                        showarrow=False, font=dict(size=10),
                    )
                y_start += height_val
            # X-axis label
            fig.add_annotation(x=x_start + width / 2, y=-0.05,
                               text=str(cat1), showarrow=False, font=dict(size=11))
            x_start += width

        # Y-axis labels
        y_cum = 0
        overall_col_props = col_sums / total
        for j, cat2 in enumerate(categories_v2):
            prop = overall_col_props.iloc[j]
            fig.add_annotation(x=-0.03, y=y_cum + prop / 2,
                               text=str(cat2), showarrow=False, font=dict(size=11),
                               xanchor="right")
            y_cum += prop

        fig.update_layout(title=title, height=height,
                          xaxis=dict(range=[-0.1, 1], showgrid=False, zeroline=False,
                                     title=var1),
                          yaxis=dict(range=[-0.1, 1.05], showgrid=False, zeroline=False,
                                     title=var2))
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Colors: Blue = more than expected (positive residual), Red = less than expected (negative residual). Numbers show count and Pearson residual.")

    # ── Variability Chart ──
    elif chart_type == "Variability Chart":
        if not num_cols:
            st.warning("Need numeric columns.")
            return
        if not cat_cols:
            st.warning("Need at least one categorical grouping column.")
            return

        value = st.selectbox("Value:", num_cols, key="var_val")
        group1 = st.selectbox("Primary group:", cat_cols, key="var_g1")
        group2 = st.selectbox("Secondary group (optional):", [None] + [c for c in cat_cols if c != group1],
                               key="var_g2")

        data_clean = df[[value, group1] + ([group2] if group2 else [])].dropna()

        fig = go.Figure()
        if group2:
            # Nested grouping
            groups = data_clean.groupby([group1, group2])
            x_labels = []
            x_pos = 0
            group_positions = {}
            for (g1, g2), grp in groups:
                label = f"{g1}|{g2}"
                x_labels.append(label)
                vals = grp[value].values
                # Individual points
                fig.add_trace(go.Scatter(
                    x=[x_pos] * len(vals), y=vals, mode="markers",
                    marker=dict(size=5, opacity=0.5), showlegend=False,
                ))
                if g1 not in group_positions:
                    group_positions[g1] = []
                group_positions[g1].append((x_pos, grp[value].mean()))
                x_pos += 1

            # Connect means within primary groups
            for g1, positions in group_positions.items():
                xs, ys = zip(*positions)
                fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines+markers",
                                         marker=dict(size=8, symbol="diamond"),
                                         line=dict(width=2),
                                         name=str(g1)))

            fig.update_layout(xaxis=dict(tickvals=list(range(len(x_labels))),
                                         ticktext=x_labels, tickangle=45))
        else:
            groups = data_clean.groupby(group1)
            x_labels = []
            means = []
            for i, (g_name, grp) in enumerate(groups):
                x_labels.append(str(g_name))
                vals = grp[value].values
                fig.add_trace(go.Scatter(
                    x=[i] * len(vals), y=vals, mode="markers",
                    marker=dict(size=5, opacity=0.5), showlegend=False,
                ))
                means.append(grp[value].mean())

            fig.add_trace(go.Scatter(x=list(range(len(x_labels))), y=means,
                                     mode="lines+markers",
                                     marker=dict(size=10, symbol="diamond", color="red"),
                                     line=dict(width=2, color="red"),
                                     name="Mean"))
            fig.update_layout(xaxis=dict(tickvals=list(range(len(x_labels))),
                                         ticktext=x_labels))

        fig.update_layout(title=title, height=height,
                          xaxis_title=group1, yaxis_title=value)
        st.plotly_chart(fig, use_container_width=True)
