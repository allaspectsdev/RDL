"""
Visualization Builder Module - 34+ interactive chart types with full customization.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.ui_helpers import grouped_chart_selector, empty_state, section_header, _RDL_COLORWAY, rdl_plotly_chart


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
        rdl_plotly_chart(fig)

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
            rdl_plotly_chart(fig)

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
        rdl_plotly_chart(fig)

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
        rdl_plotly_chart(fig)

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
        rdl_plotly_chart(fig)

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
        rdl_plotly_chart(fig)

    # ── Strip Plot ──
    elif chart_type == "Strip Plot":
        y = st.selectbox("Value:", num_cols, key="strip_y")
        x = st.selectbox("Group by:", [None] + cat_cols, key="strip_x")
        color = st.selectbox("Color:", [None] + cat_cols, key="strip_color")

        fig = px.strip(df, x=x, y=y, color=color, title=title)
        fig.update_layout(height=height)
        rdl_plotly_chart(fig)

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
            rdl_plotly_chart(fig)

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
        rdl_plotly_chart(fig)

    # ── Area Chart ──
    elif chart_type == "Area Chart":
        x = st.selectbox("X:", all_cols, key="area_x")
        y_cols = st.multiselect("Y:", num_cols, default=num_cols[:2], key="area_y")
        stacked = st.checkbox("Stacked", value=True, key="area_stack")

        if y_cols:
            fig = px.area(df, x=x, y=y_cols, title=title,
                          groupnorm="percent" if stacked else None)
            fig.update_layout(height=height)
            rdl_plotly_chart(fig)

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
        rdl_plotly_chart(fig)

    # ── Treemap ──
    elif chart_type == "Treemap":
        path_cols = st.multiselect("Hierarchy (outer → inner):", cat_cols, key="tree_path")
        values = st.selectbox("Values:", [None] + num_cols, key="tree_vals")

        if path_cols:
            fig = px.treemap(df, path=path_cols, values=values, title=title)
            fig.update_layout(height=height)
            rdl_plotly_chart(fig)

    # ── Sunburst ──
    elif chart_type == "Sunburst":
        path_cols = st.multiselect("Hierarchy:", cat_cols, key="sun_path")
        values = st.selectbox("Values:", [None] + num_cols, key="sun_vals")

        if path_cols:
            fig = px.sunburst(df, path=path_cols, values=values, title=title)
            fig.update_layout(height=height)
            rdl_plotly_chart(fig)

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
            rdl_plotly_chart(fig)

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
            rdl_plotly_chart(fig)

    # ── 3D Scatter ──
    elif chart_type == "3D Scatter":
        if len(num_cols) < 3:
            empty_state("Need at least 3 numeric columns.")
            return
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X:", num_cols, key="3d_x")
        y = c2.selectbox("Y:", [c for c in num_cols if c != x], key="3d_y")
        z = c3.selectbox("Z:", [c for c in num_cols if c not in (x, y)], key="3d_z")
        color = st.selectbox("Color:", [None] + all_cols, key="3d_color")

        fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, title=title, opacity=opacity)
        fig.update_layout(height=height)
        rdl_plotly_chart(fig)

    # ── 3D Surface ──
    elif chart_type == "3D Surface":
        if len(num_cols) < 3:
            empty_state("Need at least 3 numeric columns.")
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
            rdl_plotly_chart(fig)
        except Exception as e:
            st.error(f"Surface plot error: {e}")

    # ── Contour Plot ──
    elif chart_type == "Contour Plot":
        if len(num_cols) < 2:
            empty_state("Need at least 2 numeric columns.")
            return
        c1, c2 = st.columns(2)
        x = c1.selectbox("X:", num_cols, key="cont_x")
        y = c2.selectbox("Y:", [c for c in num_cols if c != x], key="cont_y")

        fig = px.density_contour(df, x=x, y=y, title=title)
        fig.update_traces(contours_coloring="fill")
        fig.update_layout(height=height)
        rdl_plotly_chart(fig)

    # ── Funnel Chart ──
    elif chart_type == "Funnel Chart":
        stage = st.selectbox("Stage:", cat_cols if cat_cols else all_cols, key="fun_stage")
        value = st.selectbox("Value:", num_cols, key="fun_val")

        agg_data = df.groupby(stage)[value].sum().sort_values(ascending=False).reset_index()
        fig = px.funnel(agg_data, x=value, y=stage, title=title)
        fig.update_layout(height=height)
        rdl_plotly_chart(fig)

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
        rdl_plotly_chart(fig)

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
        rdl_plotly_chart(fig)

    # ── Joint Plot ──
    elif chart_type == "Joint Plot":
        if len(num_cols) < 2:
            empty_state("Need at least 2 numeric columns.")
            return
        c1, c2 = st.columns(2)
        x = c1.selectbox("X:", num_cols, key="joint_x")
        y = c2.selectbox("Y:", [c for c in num_cols if c != x], key="joint_y")
        color = st.selectbox("Color:", [None] + cat_cols, key="joint_color")

        fig = px.scatter(df, x=x, y=y, color=color, marginal_x="histogram",
                         marginal_y="histogram", title=title, opacity=opacity)
        fig.update_layout(height=height)
        rdl_plotly_chart(fig)

    # ── Mosaic Plot ──
    elif chart_type == "Mosaic Plot":
        if len(cat_cols) < 2:
            empty_state("Need at least 2 categorical columns for mosaic plot.")
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
        rdl_plotly_chart(fig)

        st.caption("Colors: Blue = more than expected (positive residual), Red = less than expected (negative residual). Numbers show count and Pearson residual.")

    # ── Variability Chart ──
    elif chart_type == "Variability Chart":
        if not num_cols:
            empty_state("Need numeric columns.")
            return
        if not cat_cols:
            empty_state("Need at least one categorical grouping column.")
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
        rdl_plotly_chart(fig)

    # ── Error Bar Chart ──
    elif chart_type == "Error Bar Chart":
        if not num_cols or not cat_cols:
            empty_state("Need at least one numeric and one categorical column.")
            return
        c1, c2 = st.columns(2)
        val_col = c1.selectbox("Value column:", num_cols, key="err_val")
        grp_col = c2.selectbox("Group column:", cat_cols, key="err_grp")
        c1, c2 = st.columns(2)
        error_type = c1.selectbox("Error type:", ["SE", "SD", "95% CI"], key="err_type")
        orientation = c2.selectbox("Orientation:", ["vertical", "horizontal"], key="err_orient")

        grouped = df.groupby(grp_col)[val_col]
        means = grouped.mean()
        sds = grouped.std()
        counts = grouped.count()

        if error_type == "SE":
            errors = sds / np.sqrt(counts)
        elif error_type == "SD":
            errors = sds
        else:  # 95% CI
            errors = 1.96 * sds / np.sqrt(counts)

        categories = means.index.astype(str).tolist()
        if orientation == "vertical":
            fig = go.Figure(go.Bar(
                x=categories, y=means.values, error_y=dict(type="data", array=errors.values),
                marker_color=_RDL_COLORWAY[0], opacity=opacity,
            ))
            fig.update_layout(xaxis_title=grp_col, yaxis_title=f"Mean {val_col}")
        else:
            fig = go.Figure(go.Bar(
                y=categories, x=means.values, error_x=dict(type="data", array=errors.values),
                marker_color=_RDL_COLORWAY[0], opacity=opacity, orientation="h",
            ))
            fig.update_layout(yaxis_title=grp_col, xaxis_title=f"Mean {val_col}")

        fig.update_layout(title=title, height=height)
        rdl_plotly_chart(fig)

    # ── Sankey Diagram ──
    elif chart_type == "Sankey Diagram":
        if not cat_cols:
            empty_state("Need at least two categorical columns for a Sankey diagram.",
                        "Upload data with categorical columns.")
            return
        stage_cols = st.multiselect("Stage columns (2-3, in order):", cat_cols,
                                     default=cat_cols[:min(2, len(cat_cols))], key="sankey_stages")
        value_col = st.selectbox("Value column (optional, auto-counts if None):",
                                  [None] + num_cols, key="sankey_val")

        if len(stage_cols) < 2:
            st.info("Select at least 2 stage columns.")
        else:
            # Build links across consecutive stages
            all_labels = []
            label_map = {}
            links_source = []
            links_target = []
            links_value = []

            for stage_col in stage_cols:
                for val in df[stage_col].dropna().unique():
                    label = f"{stage_col}: {val}"
                    if label not in label_map:
                        label_map[label] = len(all_labels)
                        all_labels.append(label)

            for i in range(len(stage_cols) - 1):
                src_col = stage_cols[i]
                tgt_col = stage_cols[i + 1]
                if value_col:
                    agg = df.groupby([src_col, tgt_col])[value_col].sum().reset_index()
                    for _, row in agg.iterrows():
                        src_label = f"{src_col}: {row[src_col]}"
                        tgt_label = f"{tgt_col}: {row[tgt_col]}"
                        if src_label in label_map and tgt_label in label_map:
                            links_source.append(label_map[src_label])
                            links_target.append(label_map[tgt_label])
                            links_value.append(row[value_col])
                else:
                    counts = df.groupby([src_col, tgt_col]).size().reset_index(name="count")
                    for _, row in counts.iterrows():
                        src_label = f"{src_col}: {row[src_col]}"
                        tgt_label = f"{tgt_col}: {row[tgt_col]}"
                        if src_label in label_map and tgt_label in label_map:
                            links_source.append(label_map[src_label])
                            links_target.append(label_map[tgt_label])
                            links_value.append(row["count"])

            # Auto-assign colors from RDL palette
            node_colors = [_RDL_COLORWAY[i % len(_RDL_COLORWAY)] for i in range(len(all_labels))]

            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=15, thickness=20, line=dict(color="black", width=0.5),
                    label=all_labels, color=node_colors,
                ),
                link=dict(source=links_source, target=links_target, value=links_value),
            ))
            fig.update_layout(title=title, height=height)
            rdl_plotly_chart(fig)

    # ── Ridgeline Plot ──
    elif chart_type == "Ridgeline Plot":
        if not num_cols or not cat_cols:
            empty_state("Need at least one numeric and one categorical column.")
            return
        c1, c2 = st.columns(2)
        val_col = c1.selectbox("Numeric column:", num_cols, key="ridge_val")
        grp_col = c2.selectbox("Group column:", cat_cols, key="ridge_grp")
        sort_by = st.selectbox("Sort groups by:", ["alphabetical", "median"], key="ridge_sort")

        data = df[[val_col, grp_col]].dropna()
        groups = data.groupby(grp_col)[val_col]

        if sort_by == "median":
            group_order = groups.median().sort_values().index.tolist()
        else:
            group_order = sorted(groups.groups.keys())

        n_groups = len(group_order)
        if n_groups < 1:
            st.warning("No groups found.")
        else:
            fig = go.Figure()
            spacing = 1.0
            x_min = data[val_col].min()
            x_max = data[val_col].max()
            kde_x = np.linspace(x_min, x_max, 200)

            for i, grp_name in enumerate(group_order):
                grp_data = groups.get_group(grp_name).values
                if len(grp_data) < 2:
                    continue
                try:
                    kde = stats.gaussian_kde(grp_data)
                    kde_y = kde(kde_x)
                    # Normalize so max height is ~0.8 of spacing
                    kde_y = kde_y / kde_y.max() * 0.8 * spacing
                    offset = i * spacing
                    color = _RDL_COLORWAY[i % len(_RDL_COLORWAY)]
                    fig.add_trace(go.Scatter(
                        x=kde_x, y=kde_y + offset, mode="lines",
                        fill="tozeroy" if i == 0 else None,
                        line=dict(color=color, width=1.5),
                        fillcolor=color.replace(")", ", 0.3)").replace("rgb", "rgba") if "rgb" in color else color,
                        name=str(grp_name), showlegend=True,
                    ))
                    # Fill to the offset baseline
                    fig.add_trace(go.Scatter(
                        x=kde_x, y=[offset] * len(kde_x), mode="lines",
                        line=dict(color="rgba(0,0,0,0)", width=0),
                        showlegend=False, hoverinfo="skip",
                    ))
                    # Use fill="tonexty" on the KDE trace relative to baseline
                    fig.data[-2].update(fill="tonexty", fillcolor=color + "4D" if color.startswith("#") else color)
                except Exception:
                    continue

            fig.update_layout(
                title=title, height=max(height, n_groups * 60 + 100),
                yaxis=dict(
                    tickvals=[i * spacing for i in range(n_groups)],
                    ticktext=[str(g) for g in group_order],
                    showgrid=False,
                ),
                xaxis_title=val_col,
            )
            rdl_plotly_chart(fig)

    # ── Hexbin Plot ──
    elif chart_type == "Hexbin Plot":
        if len(num_cols) < 2:
            empty_state("Need at least 2 numeric columns.")
            return
        c1, c2 = st.columns(2)
        x = c1.selectbox("X:", num_cols, key="hex_x")
        y = c2.selectbox("Y:", [c for c in num_cols if c != x], key="hex_y")
        c1, c2 = st.columns(2)
        nbins = c1.slider("Number of bins:", 10, 100, 30, key="hex_bins")
        hex_color_scale = c2.selectbox("Color scale:", [
            "Viridis", "Plasma", "Inferno", "Blues", "Reds", "YlOrRd", "RdBu",
        ], key="hex_cs")
        marginals = st.checkbox("Show marginal histograms", value=False, key="hex_marg")

        fig = px.density_heatmap(
            df, x=x, y=y, nbinsx=nbins, nbinsy=nbins,
            color_continuous_scale=hex_color_scale, title=title,
            marginal_x="histogram" if marginals else None,
            marginal_y="histogram" if marginals else None,
        )
        fig.update_layout(height=height)
        rdl_plotly_chart(fig)

    # ── Lollipop Chart ──
    elif chart_type == "Lollipop Chart":
        if not cat_cols:
            empty_state("Need at least one categorical column.")
            return
        c1, c2 = st.columns(2)
        cat_col = c1.selectbox("Category:", cat_cols, key="lolli_cat")
        val_col = c2.selectbox("Value:", [None] + num_cols, key="lolli_val")
        orientation = st.selectbox("Orientation:", ["horizontal", "vertical"], key="lolli_orient")

        if val_col:
            agg = df.groupby(cat_col)[val_col].mean().sort_values(ascending=True).reset_index()
            agg.columns = [cat_col, "value"]
        else:
            agg = df[cat_col].value_counts().sort_values(ascending=True).reset_index()
            agg.columns = [cat_col, "value"]

        fig = go.Figure()
        if orientation == "horizontal":
            for i, row in agg.iterrows():
                fig.add_trace(go.Scatter(
                    x=[0, row["value"]], y=[row[cat_col], row[cat_col]],
                    mode="lines", line=dict(color=_RDL_COLORWAY[0], width=2),
                    showlegend=False, hoverinfo="skip",
                ))
            fig.add_trace(go.Scatter(
                x=agg["value"], y=agg[cat_col], mode="markers",
                marker=dict(size=10, color=_RDL_COLORWAY[0]),
                name="Value",
            ))
            fig.update_layout(xaxis_title=val_col or "Count", yaxis_title=cat_col)
        else:
            for i, row in agg.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row[cat_col], row[cat_col]], y=[0, row["value"]],
                    mode="lines", line=dict(color=_RDL_COLORWAY[0], width=2),
                    showlegend=False, hoverinfo="skip",
                ))
            fig.add_trace(go.Scatter(
                x=agg[cat_col], y=agg["value"], mode="markers",
                marker=dict(size=10, color=_RDL_COLORWAY[0]),
                name="Value",
            ))
            fig.update_layout(xaxis_title=cat_col, yaxis_title=val_col or "Count")

        fig.update_layout(title=title, height=height)
        rdl_plotly_chart(fig)

    # ── Bump Chart ──
    elif chart_type == "Bump Chart":
        if not num_cols or not cat_cols:
            empty_state("Need numeric and categorical columns for a bump chart.")
            return
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X (time/category):", all_cols, key="bump_x")
        y_col = c2.selectbox("Y (numeric):", num_cols, key="bump_y")
        grp_col = c3.selectbox("Group:", cat_cols, key="bump_grp")

        data = df[[x_col, y_col, grp_col]].dropna()

        # Rank-transform Y within each X value (lower rank = higher value, 1 = top)
        ranked = data.copy()
        ranked["rank"] = data.groupby(x_col)[y_col].rank(ascending=False, method="min").astype(int)

        fig = go.Figure()
        for i, grp_name in enumerate(ranked[grp_col].unique()):
            grp_data = ranked[ranked[grp_col] == grp_name].sort_values(x_col)
            fig.add_trace(go.Scatter(
                x=grp_data[x_col], y=grp_data["rank"],
                mode="lines+markers", name=str(grp_name),
                line=dict(width=3, color=_RDL_COLORWAY[i % len(_RDL_COLORWAY)]),
                marker=dict(size=8),
            ))

        fig.update_layout(
            title=title, height=height,
            yaxis=dict(autorange="reversed", title="Rank (1 = top)", dtick=1),
            xaxis_title=x_col,
        )
        rdl_plotly_chart(fig)

    # ── Slope Chart ──
    elif chart_type == "Slope Chart":
        if len(num_cols) < 2:
            empty_state("Need at least 2 numeric columns for a slope chart.")
            return
        c1, c2 = st.columns(2)
        col_before = c1.selectbox("Before (column):", num_cols, key="slope_before")
        col_after = c2.selectbox("After (column):", [c for c in num_cols if c != col_before], key="slope_after")
        label_col = st.selectbox("Label column (optional):", [None] + cat_cols + [c for c in all_cols if c not in num_cols], key="slope_label")

        data = df[[col_before, col_after]].dropna()
        if label_col:
            labels = df.loc[data.index, label_col].astype(str).values
        else:
            labels = [f"Row {i}" for i in range(len(data))]

        fig = go.Figure()
        for i in range(len(data)):
            color = _RDL_COLORWAY[i % len(_RDL_COLORWAY)]
            before_val = data.iloc[i][col_before]
            after_val = data.iloc[i][col_after]
            fig.add_trace(go.Scatter(
                x=[col_before, col_after], y=[before_val, after_val],
                mode="lines+markers+text",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                text=[str(labels[i]), str(labels[i])],
                textposition=["middle left", "middle right"],
                textfont=dict(size=9),
                name=str(labels[i]),
                showlegend=False,
            ))

        fig.update_layout(
            title=title, height=max(height, len(data) * 15 + 100),
            xaxis=dict(tickvals=[col_before, col_after], range=[-0.3, 1.3]),
            yaxis_title="Value",
        )
        rdl_plotly_chart(fig)

    # ── Diverging Bar Chart ──
    elif chart_type == "Diverging Bar Chart":
        if not cat_cols or not num_cols:
            empty_state("Need at least one categorical and one numeric column.")
            return
        c1, c2 = st.columns(2)
        cat_col = c1.selectbox("Category:", cat_cols, key="div_cat")
        val_col = c2.selectbox("Value:", num_cols, key="div_val")

        agg = df.groupby(cat_col)[val_col].mean().sort_values().reset_index()
        agg.columns = [cat_col, "value"]

        colors = [_RDL_COLORWAY[4] if v >= 0 else _RDL_COLORWAY[3] for v in agg["value"]]

        fig = go.Figure(go.Bar(
            y=agg[cat_col].astype(str), x=agg["value"],
            orientation="h", marker_color=colors, opacity=opacity,
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(title=title, height=max(height, len(agg) * 25),
                          xaxis_title=val_col, yaxis_title=cat_col)
        rdl_plotly_chart(fig)

    # ── Bullet Chart ──
    elif chart_type == "Bullet Chart":
        if not cat_cols or len(num_cols) < 2:
            empty_state("Need at least one categorical column and two numeric columns (actual + target).")
            return
        c1, c2, c3 = st.columns(3)
        cat_col = c1.selectbox("Category:", cat_cols, key="bullet_cat")
        actual_col = c2.selectbox("Actual value:", num_cols, key="bullet_actual")
        target_col = c3.selectbox("Target value:", [c for c in num_cols if c != actual_col], key="bullet_target")
        range_col = st.selectbox("Range max (optional):", [None] + [c for c in num_cols if c not in (actual_col, target_col)], key="bullet_range")

        data = df.groupby(cat_col).agg({actual_col: "mean", target_col: "mean"}).reset_index()
        if range_col:
            range_data = df.groupby(cat_col)[range_col].mean()
        else:
            range_data = None

        fig = go.Figure()

        categories = data[cat_col].astype(str).tolist()
        actuals = data[actual_col].values
        targets = data[target_col].values

        max_val = max(actuals.max(), targets.max())
        if range_data is not None:
            max_val = max(max_val, range_data.max())

        # Background ranges (light gray bars representing scale)
        fig.add_trace(go.Bar(
            y=categories, x=[max_val * 1.1] * len(categories),
            orientation="h", marker_color="rgba(200,200,200,0.3)",
            name="Range", showlegend=True, width=0.6,
        ))
        fig.add_trace(go.Bar(
            y=categories, x=[max_val * 0.75] * len(categories),
            orientation="h", marker_color="rgba(200,200,200,0.5)",
            name="75%", showlegend=False, width=0.6,
        ))

        # Actual values (thinner bars)
        fig.add_trace(go.Bar(
            y=categories, x=actuals,
            orientation="h", marker_color=_RDL_COLORWAY[0],
            name="Actual", width=0.3,
        ))

        # Target markers (vertical lines)
        fig.add_trace(go.Scatter(
            y=categories, x=targets,
            mode="markers", marker=dict(symbol="line-ns", size=20,
                                         line=dict(width=3, color=_RDL_COLORWAY[3])),
            name="Target",
        ))

        fig.update_layout(
            title=title, height=max(height, len(categories) * 60),
            barmode="overlay", xaxis_title="Value", yaxis_title=cat_col,
        )
        rdl_plotly_chart(fig)

    # ── Calendar Heatmap ──
    elif chart_type == "Calendar Heatmap":
        if not date_cols:
            empty_state("Need at least one datetime column for a calendar heatmap.",
                        "Convert a column to datetime in Data Manager.")
            return
        c1, c2 = st.columns(2)
        date_col = c1.selectbox("Date column:", date_cols, key="cal_date")
        val_col = c2.selectbox("Value column:", num_cols if num_cols else [None], key="cal_val")
        cal_color_scale = st.selectbox("Color scale:", [
            "Viridis", "Plasma", "Blues", "Greens", "YlOrRd", "RdBu",
        ], key="cal_cs")

        data = df[[date_col]].copy()
        if val_col:
            data["value"] = df[val_col]
        else:
            data["value"] = 1

        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
        data = data.dropna(subset=[date_col])
        data = data.set_index(date_col)
        daily = data["value"].resample("D").sum().reset_index()
        daily.columns = ["date", "value"]

        daily["dow"] = daily["date"].dt.dayofweek  # Mon=0, Sun=6
        daily["week"] = daily["date"].dt.isocalendar().week.astype(int)
        daily["year"] = daily["date"].dt.year

        # Handle multi-year: offset week by year
        years = sorted(daily["year"].unique())
        if len(years) > 1:
            st.info(f"Showing {len(years)} years. Select a specific year for cleaner display.")
            year_filter = st.selectbox("Year:", years, index=len(years) - 1, key="cal_year")
            daily = daily[daily["year"] == year_filter]

        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig = go.Figure(go.Heatmap(
            x=daily["week"], y=daily["dow"], z=daily["value"],
            colorscale=cal_color_scale, showscale=True,
            hovertemplate="Week %{x}<br>%{text}<br>Value: %{z}<extra></extra>",
            text=[dow_labels[d] for d in daily["dow"]],
        ))
        fig.update_layout(
            title=title, height=height,
            yaxis=dict(tickvals=list(range(7)), ticktext=dow_labels,
                       autorange="reversed"),
            xaxis_title="Week of Year",
        )
        rdl_plotly_chart(fig)

    # ── Bland-Altman Plot ──
    elif chart_type == "Bland-Altman Plot":
        if len(num_cols) < 2:
            empty_state("Need at least 2 numeric columns for a Bland-Altman plot.")
            return
        c1, c2 = st.columns(2)
        method_a = c1.selectbox("Method A:", num_cols, key="ba_a")
        method_b = c2.selectbox("Method B:", [c for c in num_cols if c != method_a], key="ba_b")

        data = df[[method_a, method_b]].dropna()
        means = (data[method_a] + data[method_b]) / 2
        diffs = data[method_a] - data[method_b]

        mean_diff = diffs.mean()
        sd_diff = diffs.std()
        upper_loa = mean_diff + 1.96 * sd_diff
        lower_loa = mean_diff - 1.96 * sd_diff

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=means, y=diffs, mode="markers",
            marker=dict(color=_RDL_COLORWAY[0], opacity=0.7, size=6),
            name="Observations",
        ))
        fig.add_hline(y=mean_diff, line_dash="solid", line_color=_RDL_COLORWAY[1],
                       annotation_text=f"Mean diff: {mean_diff:.4f}")
        fig.add_hline(y=upper_loa, line_dash="dash", line_color=_RDL_COLORWAY[3],
                       annotation_text=f"+1.96 SD: {upper_loa:.4f}")
        fig.add_hline(y=lower_loa, line_dash="dash", line_color=_RDL_COLORWAY[3],
                       annotation_text=f"-1.96 SD: {lower_loa:.4f}")

        fig.update_layout(
            title=title, height=height,
            xaxis_title=f"Mean of {method_a} & {method_b}",
            yaxis_title=f"Difference ({method_a} - {method_b})",
        )
        rdl_plotly_chart(fig)

        # Display metrics
        section_header("Agreement Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Difference", f"{mean_diff:.4f}")
        c2.metric("SD of Differences", f"{sd_diff:.4f}")
        c3.metric("95% Limits of Agreement", f"({lower_loa:.4f}, {upper_loa:.4f})")
