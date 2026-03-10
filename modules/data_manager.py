"""
Data Manager Module - Upload, preview, clean, and transform datasets.
"""

import streamlit as st
import pandas as pd
import numpy as np


def render_upload():
    """Render file upload widget and return DataFrame or None."""
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx", "xls", "tsv", "json"],
        help="Supported: CSV, Excel, TSV, JSON",
    )
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".tsv"):
            return pd.read_csv(uploaded_file, sep="\t")
        elif uploaded_file.name.endswith(".json"):
            return pd.read_json(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return None


def render_data_manager(df: pd.DataFrame) -> pd.DataFrame:
    """Render full data management interface. Returns modified DataFrame."""
    if df is None or df.empty:
        st.warning("No data loaded.")
        return df

    tabs = st.tabs([
        "Preview", "Column Info", "Missing Values",
        "Transform", "Filter & Sort", "Column Operations",
    ])

    # ── Tab 1: Preview ──
    with tabs[0]:
        _render_preview(df)

    # ── Tab 2: Column Info ──
    with tabs[1]:
        _render_column_info(df)

    # ── Tab 3: Missing Values ──
    with tabs[2]:
        df = _render_missing_values(df)

    # ── Tab 4: Transform ──
    with tabs[3]:
        df = _render_transform(df)

    # ── Tab 5: Filter & Sort ──
    with tabs[4]:
        df = _render_filter_sort(df)

    # ── Tab 6: Column Operations ──
    with tabs[5]:
        df = _render_column_operations(df)

    return df


def _render_preview(df: pd.DataFrame):
    """Data preview tab."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]}")
    col3.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
    col4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    view_option = st.radio("View:", ["Head", "Tail", "Sample", "Full"], horizontal=True)
    n_rows = st.slider("Rows to display:", 5, min(200, len(df)), 20)

    if view_option == "Head":
        st.dataframe(df.head(n_rows), use_container_width=True)
    elif view_option == "Tail":
        st.dataframe(df.tail(n_rows), use_container_width=True)
    elif view_option == "Sample":
        st.dataframe(df.sample(min(n_rows, len(df))), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

    with st.expander("Data Types"):
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Null": df.isnull().sum().values,
            "Unique": df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)


def _render_column_info(df: pd.DataFrame):
    """Detailed column information."""
    selected_col = st.selectbox("Select column:", df.columns, key="col_info_select")
    col = df[selected_col]

    c1, c2, c3 = st.columns(3)
    c1.metric("Data Type", str(col.dtype))
    c2.metric("Non-Null Count", f"{col.notnull().sum():,}")
    c3.metric("Unique Values", f"{col.nunique():,}")

    if pd.api.types.is_numeric_dtype(col):
        st.markdown("#### Numeric Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{col.mean():.4g}")
        c2.metric("Median", f"{col.median():.4g}")
        c3.metric("Std Dev", f"{col.std():.4g}")
        c4.metric("Variance", f"{col.var():.4g}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min", f"{col.min():.4g}")
        c2.metric("Max", f"{col.max():.4g}")
        c3.metric("Skewness", f"{col.skew():.4g}")
        c4.metric("Kurtosis", f"{col.kurtosis():.4g}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Q1 (25%)", f"{col.quantile(0.25):.4g}")
        c2.metric("Q2 (50%)", f"{col.quantile(0.50):.4g}")
        c3.metric("Q3 (75%)", f"{col.quantile(0.75):.4g}")
    else:
        st.markdown("#### Categorical Summary")
        freq = col.value_counts()
        st.dataframe(
            pd.DataFrame({"Value": freq.index, "Count": freq.values,
                           "Percentage": (freq.values / len(col) * 100).round(2)}),
            use_container_width=True, hide_index=True,
        )


def _render_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success("No missing values in the dataset.")
        return df

    st.markdown("#### Missing Values Summary")
    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": (missing.values / len(df) * 100).round(2),
    })
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    st.markdown("#### Handle Missing Values")
    strategy = st.selectbox(
        "Strategy:",
        ["Drop rows with any missing", "Drop rows with all missing",
         "Drop columns above threshold", "Fill with value"],
        key="missing_strategy",
    )

    if strategy == "Drop rows with any missing":
        cols_to_check = st.multiselect("Check columns (all if empty):", df.columns, key="drop_rows_cols")
        if st.button("Apply", key="apply_drop_rows"):
            subset = cols_to_check if cols_to_check else None
            df = df.dropna(subset=subset).reset_index(drop=True)
            st.success(f"Dropped rows. New shape: {df.shape}")
            st.session_state["df"] = df
    elif strategy == "Drop rows with all missing":
        if st.button("Apply", key="apply_drop_all"):
            df = df.dropna(how="all").reset_index(drop=True)
            st.success(f"Dropped rows. New shape: {df.shape}")
            st.session_state["df"] = df
    elif strategy == "Drop columns above threshold":
        threshold = st.slider("Max missing %:", 0, 100, 50, key="drop_col_thresh")
        if st.button("Apply", key="apply_drop_cols"):
            max_missing = len(df) * threshold / 100
            df = df.loc[:, df.isnull().sum() <= max_missing]
            st.success(f"Dropped columns. New shape: {df.shape}")
            st.session_state["df"] = df
    elif strategy == "Fill with value":
        fill_col = st.selectbox("Column:", missing.index, key="fill_col")
        fill_method = st.selectbox(
            "Method:",
            ["Mean", "Median", "Mode", "Constant", "Forward Fill", "Backward Fill"],
            key="fill_method",
        )
        if fill_method == "Constant":
            fill_val = st.text_input("Value:", "0", key="fill_const")
        if st.button("Apply", key="apply_fill"):
            if fill_method == "Mean" and pd.api.types.is_numeric_dtype(df[fill_col]):
                df[fill_col] = df[fill_col].fillna(df[fill_col].mean())
            elif fill_method == "Median" and pd.api.types.is_numeric_dtype(df[fill_col]):
                df[fill_col] = df[fill_col].fillna(df[fill_col].median())
            elif fill_method == "Mode":
                mode_vals = df[fill_col].mode()
                if mode_vals.empty:
                    st.warning(f"No mode found for '{fill_col}' (all values unique or all NaN).")
                else:
                    df[fill_col] = df[fill_col].fillna(mode_vals.iloc[0])
            elif fill_method == "Constant":
                df[fill_col] = df[fill_col].fillna(fill_val)
            elif fill_method == "Forward Fill":
                df[fill_col] = df[fill_col].ffill()
            elif fill_method == "Backward Fill":
                df[fill_col] = df[fill_col].bfill()
            st.success(f"Filled missing values in '{fill_col}'.")
            st.session_state["df"] = df

    return df


def _render_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Data type conversion and transformations."""
    st.markdown("#### Convert Data Types")
    conv_col = st.selectbox("Column:", df.columns, key="conv_col")
    target_type = st.selectbox(
        "Convert to:",
        ["numeric", "string", "category", "datetime"],
        key="conv_type",
    )
    if st.button("Convert", key="apply_convert"):
        try:
            nans_before = df[conv_col].isna().sum()
            if target_type == "numeric":
                df[conv_col] = pd.to_numeric(df[conv_col], errors="coerce")
            elif target_type == "string":
                df[conv_col] = df[conv_col].astype(str)
            elif target_type == "category":
                df[conv_col] = df[conv_col].astype("category")
            elif target_type == "datetime":
                df[conv_col] = pd.to_datetime(df[conv_col], errors="coerce")
            nans_after = df[conv_col].isna().sum()
            new_nans = nans_after - nans_before
            if new_nans > 0:
                st.warning(f"{new_nans} value(s) could not be converted and became NaN.")
            st.success(f"Converted '{conv_col}' to {target_type}.")
            st.session_state["df"] = df
        except Exception as e:
            st.error(f"Conversion failed: {e}")

    st.divider()
    st.markdown("#### Rename Columns")
    rename_col = st.selectbox("Column to rename:", df.columns, key="rename_col")
    new_name = st.text_input("New name:", rename_col, key="rename_new")
    if st.button("Rename", key="apply_rename"):
        if new_name and new_name != rename_col:
            df = df.rename(columns={rename_col: new_name})
            st.success(f"Renamed '{rename_col}' to '{new_name}'.")
            st.session_state["df"] = df

    st.divider()
    st.markdown("#### Remove Duplicates")
    dup_count = df.duplicated().sum()
    st.write(f"Duplicate rows: **{dup_count}**")
    if dup_count > 0 and st.button("Remove Duplicates", key="apply_dedup"):
        df = df.drop_duplicates().reset_index(drop=True)
        st.success(f"Removed {dup_count} duplicates. New shape: {df.shape}")
        st.session_state["df"] = df

    return df


def _render_filter_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and sort data."""
    st.markdown("#### Filter Rows")
    filter_col = st.selectbox("Filter column:", df.columns, key="filter_col")

    if pd.api.types.is_numeric_dtype(df[filter_col]):
        col_min, col_max = float(df[filter_col].min()), float(df[filter_col].max())
        range_vals = st.slider(
            "Value range:", col_min, col_max, (col_min, col_max),
            key="filter_range",
        )
        if st.button("Apply Filter", key="apply_filter_num"):
            df = df[(df[filter_col] >= range_vals[0]) & (df[filter_col] <= range_vals[1])].reset_index(drop=True)
            st.success(f"Filtered. New shape: {df.shape}")
            st.session_state["df"] = df
    else:
        unique_vals = df[filter_col].dropna().unique().tolist()
        selected_vals = st.multiselect("Select values:", unique_vals, default=unique_vals, key="filter_cat")
        if st.button("Apply Filter", key="apply_filter_cat"):
            df = df[df[filter_col].isin(selected_vals)].reset_index(drop=True)
            st.success(f"Filtered. New shape: {df.shape}")
            st.session_state["df"] = df

    st.divider()
    st.markdown("#### Sort Data")
    sort_col = st.selectbox("Sort by:", df.columns, key="sort_col")
    sort_asc = st.checkbox("Ascending", value=True, key="sort_asc")
    if st.button("Sort", key="apply_sort"):
        df = df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)
        st.success(f"Sorted by '{sort_col}'.")
        st.session_state["df"] = df

    return df


def _render_column_operations(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level operations: transforms, binning, encoding."""
    st.markdown("#### Numeric Transformations")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if num_cols:
        trans_col = st.selectbox("Column:", num_cols, key="trans_col")
        transform = st.selectbox(
            "Transform:",
            ["Log (ln)", "Log10", "Square Root", "Z-Score (Standardize)",
             "Min-Max Normalize (0-1)", "Square", "Reciprocal", "Abs"],
            key="trans_type",
        )
        new_col_name = st.text_input("New column name:", f"{trans_col}_{transform.split()[0].lower()}", key="trans_name")

        if st.button("Apply Transform", key="apply_trans"):
            col_data = df[trans_col]
            if transform == "Log (ln)":
                if (col_data <= 0).any():
                    st.warning("Log requires positive values. Applying log(x + |min| + 1) for non-positive values.")
                    shift = abs(col_data.min()) + 1 if col_data.min() <= 0 else 0
                    df[new_col_name] = np.log(col_data + shift)
                else:
                    df[new_col_name] = np.log(col_data)
            elif transform == "Log10":
                if (col_data <= 0).any():
                    shift = abs(col_data.min()) + 1 if col_data.min() <= 0 else 0
                    df[new_col_name] = np.log10(col_data + shift)
                else:
                    df[new_col_name] = np.log10(col_data)
            elif transform == "Square Root":
                if (col_data < 0).any():
                    st.warning("Negative values present. Taking sqrt of absolute values.")
                    df[new_col_name] = np.sqrt(np.abs(col_data))
                else:
                    df[new_col_name] = np.sqrt(col_data)
            elif transform == "Z-Score (Standardize)":
                mean_val = col_data.mean()
                std_val = col_data.std()
                if std_val == 0:
                    st.error("Standard deviation is 0. Cannot standardize.")
                else:
                    df[new_col_name] = (col_data - mean_val) / std_val
            elif transform == "Min-Max Normalize (0-1)":
                min_val = col_data.min()
                max_val = col_data.max()
                if max_val == min_val:
                    st.error("All values are identical. Cannot normalize.")
                else:
                    df[new_col_name] = (col_data - min_val) / (max_val - min_val)
            elif transform == "Square":
                df[new_col_name] = col_data ** 2
            elif transform == "Reciprocal":
                if (col_data == 0).any():
                    st.warning("Zero values present. They will become NaN.")
                df[new_col_name] = 1.0 / col_data.replace(0, np.nan)
            elif transform == "Abs":
                df[new_col_name] = np.abs(col_data)
            st.success(f"Created column '{new_col_name}'.")
            st.session_state["df"] = df

    st.divider()
    st.markdown("#### Binning")
    if num_cols:
        bin_col = st.selectbox("Column to bin:", num_cols, key="bin_col")
        bin_method = st.selectbox("Method:", ["Equal Width", "Equal Frequency", "Custom"], key="bin_method")
        n_bins = st.number_input("Number of bins:", 2, 50, 5, key="n_bins")
        bin_col_name = st.text_input("New column name:", f"{bin_col}_binned", key="bin_name")

        if st.button("Apply Binning", key="apply_bin"):
            if bin_method == "Equal Width":
                df[bin_col_name] = pd.cut(df[bin_col], bins=n_bins, include_lowest=True).astype(str)
            elif bin_method == "Equal Frequency":
                df[bin_col_name] = pd.qcut(df[bin_col], q=n_bins, duplicates="drop").astype(str)
            st.success(f"Created binned column '{bin_col_name}'.")
            st.session_state["df"] = df

    st.divider()
    st.markdown("#### Encoding")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        enc_col = st.selectbox("Column to encode:", cat_cols, key="enc_col")
        enc_method = st.selectbox("Method:", ["One-Hot Encoding", "Label Encoding"], key="enc_method")

        if st.button("Apply Encoding", key="apply_enc"):
            if enc_method == "One-Hot Encoding":
                dummies = pd.get_dummies(df[enc_col], prefix=enc_col, dtype=int)
                df = pd.concat([df, dummies], axis=1)
                st.success(f"One-hot encoded '{enc_col}'. Added {len(dummies.columns)} columns.")
            elif enc_method == "Label Encoding":
                df[f"{enc_col}_encoded"] = df[enc_col].astype("category").cat.codes
                st.success(f"Label encoded '{enc_col}'.")
            st.session_state["df"] = df
    else:
        st.info("No categorical columns available for encoding.")

    st.divider()
    st.markdown("#### Computed Column")
    st.caption("Create a new column using a pandas expression. Use column names as variables.")
    st.caption("Examples: `col_a + col_b`, `col_a * 2`, `col_a / col_b`")
    expr = st.text_input("Expression:", key="computed_expr")
    computed_name = st.text_input("New column name:", "computed", key="computed_name")
    if expr and st.button("Create Column", key="apply_computed"):
        try:
            df[computed_name] = df.eval(expr)
            st.success(f"Created '{computed_name}' = {expr}")
            st.session_state["df"] = df
        except Exception as e:
            st.error(f"Expression error: {e}")

    return df
