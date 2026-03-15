"""
Data Manager Module - Upload, preview, clean, and transform datasets.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

from modules.ui_helpers import section_header, empty_state, help_tip


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
        empty_state("No data loaded.", "Upload a dataset from the sidebar to begin.")
        return df

    tabs = st.tabs([
        "Preview", "Column Info", "Missing Values",
        "Transform", "Filter & Sort", "Column Operations", "Export",
        "Reshape", "Merge & Join", "Sampling",
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

    # ── Tab 7: Export ──
    with tabs[6]:
        _render_export(df)

    # ── Tab 8: Reshape ──
    with tabs[7]:
        df = _render_reshape(df)

    # ── Tab 9: Merge & Join ──
    with tabs[8]:
        df = _render_merge_join(df)

    # ── Tab 10: Sampling ──
    with tabs[9]:
        df = _render_sampling(df)

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
        section_header("Numeric Summary")
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
        section_header("Categorical Summary")
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

    section_header("Missing Values Summary")
    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": (missing.values / len(df) * 100).round(2),
    })
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    section_header("Handle Missing Values")
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
    section_header("Convert Data Types")
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
    section_header("Rename Columns")
    rename_col = st.selectbox("Column to rename:", df.columns, key="rename_col")
    new_name = st.text_input("New name:", rename_col, key="rename_new")
    if st.button("Rename", key="apply_rename"):
        if new_name and new_name != rename_col:
            df = df.rename(columns={rename_col: new_name})
            st.success(f"Renamed '{rename_col}' to '{new_name}'.")
            st.session_state["df"] = df

    st.divider()
    section_header("Remove Duplicates")
    dup_count = df.duplicated().sum()
    st.write(f"Duplicate rows: **{dup_count}**")
    if dup_count > 0 and st.button("Remove Duplicates", key="apply_dedup"):
        df = df.drop_duplicates().reset_index(drop=True)
        st.success(f"Removed {dup_count} duplicates. New shape: {df.shape}")
        st.session_state["df"] = df

    return df


def _render_filter_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and sort data."""
    section_header("Filter Rows")
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
    section_header("Sort Data")
    sort_col = st.selectbox("Sort by:", df.columns, key="sort_col")
    sort_asc = st.checkbox("Ascending", value=True, key="sort_asc")
    if st.button("Sort", key="apply_sort"):
        df = df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)
        st.success(f"Sorted by '{sort_col}'.")
        st.session_state["df"] = df

    return df


def _render_column_operations(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level operations: transforms, binning, encoding."""
    section_header("Numeric Transformations")
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
    section_header("Binning")
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
    section_header("Encoding")
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
    section_header("String Operations")
    str_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if str_cols:
        str_col = st.selectbox("String column:", str_cols, key="str_op_col")
        str_op = st.selectbox(
            "Operation:",
            ["Split by Delimiter", "Regex Extract", "Find & Replace",
             "Strip Whitespace", "Uppercase", "Lowercase", "Title Case"],
            key="str_op_type",
        )

        if str_op == "Split by Delimiter":
            delimiter = st.text_input("Delimiter:", ",", key="str_split_delim")
            max_splits = st.number_input("Max splits (0 = unlimited):", 0, 100, 0, key="str_split_n")
            n_param = max_splits if max_splits > 0 else -1
            try:
                preview_split = df[str_col].astype(str).head(5).str.split(delimiter, n=n_param, expand=True)
                preview_split.columns = [f"{str_col}_part{i}" for i in range(preview_split.shape[1])]
                st.caption("**Preview (first 5 rows):**")
                st.dataframe(preview_split, use_container_width=True, hide_index=True)
            except Exception:
                pass
            if st.button("Apply Split", key="apply_str_split"):
                try:
                    split_result = df[str_col].astype(str).str.split(delimiter, n=n_param, expand=True)
                    split_result.columns = [f"{str_col}_part{i}" for i in range(split_result.shape[1])]
                    df = pd.concat([df, split_result], axis=1)
                    st.success(f"Split '{str_col}' into {split_result.shape[1]} columns.")
                    st.session_state["df"] = df
                except Exception as e:
                    st.error(f"Split failed: {e}")

        elif str_op == "Regex Extract":
            pattern = st.text_input("Regex pattern (use capturing group):", r"(\d+)", key="str_regex_pat")
            extract_name = st.text_input("New column name:", f"{str_col}_extracted", key="str_regex_name")
            if pattern:
                try:
                    preview_extract = df[str_col].astype(str).head(5).str.extract(pattern, expand=False)
                    st.caption("**Preview (first 5 rows):**")
                    st.dataframe(pd.DataFrame({extract_name: preview_extract}), use_container_width=True, hide_index=True)
                except Exception:
                    pass
            if st.button("Apply Extract", key="apply_str_regex"):
                try:
                    df[extract_name] = df[str_col].astype(str).str.extract(pattern, expand=False)
                    st.success(f"Created '{extract_name}' from regex extraction.")
                    st.session_state["df"] = df
                except Exception as e:
                    st.error(f"Regex extraction failed: {e}")

        elif str_op == "Find & Replace":
            find_str = st.text_input("Find:", key="str_find_val")
            replace_str = st.text_input("Replace with:", key="str_replace_val")
            use_regex = st.checkbox("Use regex", value=False, key="str_replace_regex")
            if find_str:
                try:
                    preview_replaced = df[str_col].astype(str).head(5).str.replace(
                        find_str, replace_str, regex=use_regex
                    )
                    st.caption("**Preview (first 5 rows):**")
                    st.dataframe(pd.DataFrame({str_col: preview_replaced}), use_container_width=True, hide_index=True)
                except Exception:
                    pass
            if st.button("Apply Replace", key="apply_str_replace"):
                try:
                    df[str_col] = df[str_col].astype(str).str.replace(find_str, replace_str, regex=use_regex)
                    st.success(f"Replaced '{find_str}' with '{replace_str}' in '{str_col}'.")
                    st.session_state["df"] = df
                except Exception as e:
                    st.error(f"Find & Replace failed: {e}")

        elif str_op in ("Strip Whitespace", "Uppercase", "Lowercase", "Title Case"):
            modify_mode = st.radio("Mode:", ["Create new column", "Modify in place"], key="str_case_mode", horizontal=True)
            new_col_label = st.text_input(
                "New column name:",
                f"{str_col}_{str_op.split()[0].lower()}",
                key="str_case_name",
                disabled=(modify_mode == "Modify in place"),
            )
            target_name = str_col if modify_mode == "Modify in place" else new_col_label
            # Preview
            try:
                sample = df[str_col].astype(str).head(5)
                if str_op == "Strip Whitespace":
                    preview_val = sample.str.strip()
                elif str_op == "Uppercase":
                    preview_val = sample.str.upper()
                elif str_op == "Lowercase":
                    preview_val = sample.str.lower()
                else:
                    preview_val = sample.str.title()
                st.caption("**Preview (first 5 rows):**")
                st.dataframe(pd.DataFrame({target_name: preview_val}), use_container_width=True, hide_index=True)
            except Exception:
                pass
            if st.button("Apply", key="apply_str_case"):
                try:
                    series = df[str_col].astype(str)
                    if str_op == "Strip Whitespace":
                        df[target_name] = series.str.strip()
                    elif str_op == "Uppercase":
                        df[target_name] = series.str.upper()
                    elif str_op == "Lowercase":
                        df[target_name] = series.str.lower()
                    else:
                        df[target_name] = series.str.title()
                    action = "Modified" if modify_mode == "Modify in place" else f"Created '{target_name}'"
                    st.success(f"{action} with {str_op}.")
                    st.session_state["df"] = df
                except Exception as e:
                    st.error(f"String operation failed: {e}")
    else:
        st.info("No string/object columns available for string operations.")

    st.divider()
    section_header("Date/Time Extraction")
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if not dt_cols:
        st.info("No datetime columns detected. Convert a column to datetime in the Transform tab first.")
    else:
        dt_col = st.selectbox("Datetime column:", dt_cols, key="dt_extract_col")
        extraction_options = [
            "Year", "Quarter", "Month", "Month Name", "Day", "Day of Week",
            "Day Name", "Hour", "Minute", "Second", "Week of Year", "Is Weekend",
        ]
        selected_extractions = st.multiselect(
            "Extractions to create:", extraction_options,
            default=["Year", "Month", "Day"],
            key="dt_extractions",
        )
        if selected_extractions:
            preview_data = {}
            dt_series = df[dt_col]
            for ext in selected_extractions:
                col_label = f"{dt_col}_{ext.lower().replace(' ', '_')}"
                try:
                    if ext == "Year":
                        preview_data[col_label] = dt_series.dt.year
                    elif ext == "Quarter":
                        preview_data[col_label] = dt_series.dt.quarter
                    elif ext == "Month":
                        preview_data[col_label] = dt_series.dt.month
                    elif ext == "Month Name":
                        preview_data[col_label] = dt_series.dt.month_name()
                    elif ext == "Day":
                        preview_data[col_label] = dt_series.dt.day
                    elif ext == "Day of Week":
                        preview_data[col_label] = dt_series.dt.dayofweek
                    elif ext == "Day Name":
                        preview_data[col_label] = dt_series.dt.day_name()
                    elif ext == "Hour":
                        preview_data[col_label] = dt_series.dt.hour
                    elif ext == "Minute":
                        preview_data[col_label] = dt_series.dt.minute
                    elif ext == "Second":
                        preview_data[col_label] = dt_series.dt.second
                    elif ext == "Week of Year":
                        preview_data[col_label] = dt_series.dt.isocalendar().week.astype(int)
                    elif ext == "Is Weekend":
                        preview_data[col_label] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
                except Exception:
                    preview_data[col_label] = pd.Series([None] * len(df))

            st.caption("**Preview (first 5 rows):**")
            st.dataframe(pd.DataFrame(preview_data).head(5), use_container_width=True, hide_index=True)

            if st.button("Apply Extractions", key="apply_dt_extract"):
                for col_label, values in preview_data.items():
                    df[col_label] = values
                st.success(f"Created {len(preview_data)} column(s) from '{dt_col}'.")
                st.session_state["df"] = df

        st.divider()
        section_header("Time Difference")
        if len(dt_cols) >= 2:
            c1, c2 = st.columns(2)
            dt_col_a = c1.selectbox("Start datetime:", dt_cols, key="dt_diff_a")
            dt_col_b = c2.selectbox("End datetime:", dt_cols, index=min(1, len(dt_cols) - 1), key="dt_diff_b")
            diff_unit = st.selectbox("Difference in:", ["Days", "Hours", "Minutes"], key="dt_diff_unit")
            diff_name = st.text_input("New column name:", f"diff_{diff_unit.lower()}", key="dt_diff_name")

            try:
                delta = df[dt_col_b] - df[dt_col_a]
                if diff_unit == "Days":
                    preview_diff = delta.dt.total_seconds() / 86400
                elif diff_unit == "Hours":
                    preview_diff = delta.dt.total_seconds() / 3600
                else:
                    preview_diff = delta.dt.total_seconds() / 60
                st.caption("**Preview (first 5 rows):**")
                st.dataframe(pd.DataFrame({diff_name: preview_diff.head(5)}), use_container_width=True, hide_index=True)
            except Exception:
                preview_diff = None

            if st.button("Apply Time Difference", key="apply_dt_diff"):
                try:
                    delta = df[dt_col_b] - df[dt_col_a]
                    if diff_unit == "Days":
                        df[diff_name] = delta.dt.total_seconds() / 86400
                    elif diff_unit == "Hours":
                        df[diff_name] = delta.dt.total_seconds() / 3600
                    else:
                        df[diff_name] = delta.dt.total_seconds() / 60
                    st.success(f"Created '{diff_name}' ({diff_unit} between '{dt_col_a}' and '{dt_col_b}').")
                    st.session_state["df"] = df
                except Exception as e:
                    st.error(f"Time difference failed: {e}")
        else:
            st.info("Need at least 2 datetime columns for time difference.")

    st.divider()
    section_header("Computed Column")
    help_tip("Formula Editor", """
Create new columns using expressions. Supported functions:
- **Math:** abs, round, sqrt, log, log10, exp, sin, cos, tan, pi
- **Stats:** mean, median, std, var, min, max, sum, cumsum
- **String:** str.upper, str.lower, str.len, str.contains
- **Conditional:** where(condition, true_val, false_val)
- **Column reference:** Use column names directly
""")

    # Function picker
    func_categories = {
        "Math": ["abs(col)", "np.sqrt(col)", "np.log(col)", "np.log10(col)", "np.exp(col)",
                 "np.round(col, 2)", "np.sin(col)", "col ** 2", "1 / col"],
        "Stats": ["col.mean()", "col.median()", "col.std()", "col.cumsum()",
                  "col.rank()", "(col - col.mean()) / col.std()"],
        "Conditional": ["np.where(col > 0, 'positive', 'negative')",
                        "np.where(col > col.median(), 'high', 'low')"],
        "Combine": ["col_a + col_b", "col_a * col_b", "col_a / col_b",
                     "col_a - col_b"],
    }

    with st.expander("Function Reference"):
        for cat, funcs in func_categories.items():
            st.markdown(f"**{cat}:** " + ", ".join([f"`{f}`" for f in funcs]))

    # Column inserter
    col_to_insert = st.selectbox("Insert column reference:", [""] + df.columns.tolist(),
                                  key="formula_col_insert")
    expr = st.text_input("Expression:", key="computed_expr",
                          placeholder="e.g. col_a + col_b, np.log(col_a)")
    computed_name = st.text_input("New column name:", "computed", key="computed_name")

    # Live preview
    if expr:
        try:
            preview = df.head(5).eval(expr) if "np." not in expr else eval(
                expr.replace("np.", "np."),
                {"np": np, "pd": pd, **{c: df[c] for c in df.columns}}
            ).head(5)
            st.caption("**Preview (first 5 rows):**")
            st.dataframe(pd.DataFrame({computed_name: preview}), use_container_width=True,
                          hide_index=True)
        except Exception:
            pass  # Don't show errors during typing

    if expr and st.button("Create Column", key="apply_computed"):
        try:
            if "np." in expr:
                # Use eval with numpy for complex expressions
                result = eval(
                    expr,
                    {"__builtins__": {}},
                    {"np": np, "pd": pd, **{c: df[c] for c in df.columns}},
                )
                df[computed_name] = result
            else:
                df[computed_name] = df.eval(expr)
            st.success(f"Created '{computed_name}' = {expr}")
            st.session_state["df"] = df
        except Exception as e:
            st.error(f"Expression error: {e}")

    return df


def _render_export(df: pd.DataFrame):
    """Export data in various formats."""
    section_header("Export Dataset")

    data_name = st.session_state.get("data_name", "data")
    base_name = data_name.rsplit(".", 1)[0] if "." in data_name else data_name

    c1, c2, c3 = st.columns(3)

    # CSV
    csv_data = df.to_csv(index=False)
    c1.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"{base_name}.csv",
        mime="text/csv",
        key="export_csv",
        use_container_width=True,
    )

    # Excel
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    c2.download_button(
        label="Download Excel",
        data=excel_buffer.getvalue(),
        file_name=f"{base_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="export_xlsx",
        use_container_width=True,
    )

    # JSON
    json_data = df.to_json(orient="records", indent=2)
    c3.download_button(
        label="Download JSON",
        data=json_data,
        file_name=f"{base_name}.json",
        mime="application/json",
        key="export_json",
        use_container_width=True,
    )

    st.divider()

    # Export subset
    section_header("Export Subset")
    export_cols = st.multiselect(
        "Select columns to export:", df.columns.tolist(),
        default=df.columns.tolist(), key="export_cols",
    )
    if export_cols:
        subset = df[export_cols]
        c1, c2 = st.columns(2)
        c1.write(f"**Rows:** {len(subset):,}  |  **Columns:** {len(export_cols)}")
        csv_subset = subset.to_csv(index=False)
        c2.download_button(
            label="Download Subset (CSV)",
            data=csv_subset,
            file_name=f"{base_name}_subset.csv",
            mime="text/csv",
            key="export_subset_csv",
        )


def _render_reshape(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape data: wide-to-long (melt) and long-to-wide (pivot)."""

    # ── Wide to Long (Unpivot) ──
    section_header("Wide to Long (Unpivot)", "Convert columns into rows using pd.melt().")
    help_tip("Unpivot / Melt", """
Converts selected value columns into two columns: a **variable** column (holding the original column names)
and a **value** column (holding the cell values). ID columns are kept as-is.
""")

    all_cols = df.columns.tolist()
    id_cols = st.multiselect("ID columns (kept as-is):", all_cols, key="melt_id_cols")
    remaining = [c for c in all_cols if c not in id_cols]
    value_cols = st.multiselect(
        "Value columns to unpivot:", remaining,
        default=remaining, key="melt_val_cols",
    )
    c1, c2 = st.columns(2)
    var_name = c1.text_input("Variable column name:", "variable", key="melt_var_name")
    val_name = c2.text_input("Value column name:", "value", key="melt_val_name")

    if value_cols:
        try:
            melted = pd.melt(
                df, id_vars=id_cols or None, value_vars=value_cols,
                var_name=var_name, value_name=val_name,
            )
            st.caption(f"**Preview** — {melted.shape[0]:,} rows x {melted.shape[1]} columns:")
            st.dataframe(melted.head(20), use_container_width=True, hide_index=True)

            if st.button("Apply Unpivot", key="apply_melt"):
                df = melted.reset_index(drop=True)
                st.success(f"Unpivoted. New shape: {df.shape}")
                st.session_state["df"] = df
        except Exception as e:
            st.error(f"Melt failed: {e}")
    else:
        st.info("Select at least one value column to unpivot.")

    st.divider()

    # ── Long to Wide (Pivot) ──
    section_header("Long to Wide (Pivot)", "Convert rows into columns using pd.pivot_table().")
    help_tip("Pivot Table", """
Creates a spreadsheet-style pivot. Choose:
- **Index**: row labels
- **Columns**: which column's unique values become new columns
- **Values**: what to aggregate
- **Aggregation**: how to combine duplicate entries
""")

    c1, c2, c3 = st.columns(3)
    pivot_index = c1.selectbox("Index (rows):", all_cols, key="pivot_index")
    pivot_columns = c2.selectbox("Columns:", all_cols, key="pivot_columns")
    pivot_values = c3.selectbox("Values:", all_cols, key="pivot_values")
    pivot_aggfunc = st.selectbox(
        "Aggregation function:",
        ["mean", "sum", "count", "first", "last"],
        key="pivot_aggfunc",
    )

    if pivot_index != pivot_columns:
        try:
            pivoted = pd.pivot_table(
                df, index=pivot_index, columns=pivot_columns,
                values=pivot_values, aggfunc=pivot_aggfunc,
            ).reset_index()
            # Flatten MultiIndex columns if present
            if isinstance(pivoted.columns, pd.MultiIndex):
                pivoted.columns = [
                    f"{a}_{b}" if b != "" else a
                    for a, b in pivoted.columns
                ]
            st.caption(f"**Preview** — {pivoted.shape[0]:,} rows x {pivoted.shape[1]} columns:")
            st.dataframe(pivoted.head(20), use_container_width=True, hide_index=True)

            if st.button("Apply Pivot", key="apply_pivot"):
                df = pivoted.reset_index(drop=True)
                st.success(f"Pivoted. New shape: {df.shape}")
                st.session_state["df"] = df
        except Exception as e:
            st.error(f"Pivot failed: {e}")
    else:
        st.warning("Index and Columns must be different.")

    return df


def _render_merge_join(df: pd.DataFrame) -> pd.DataFrame:
    """Merge/join with a secondary dataset."""

    # ── Upload secondary dataset ──
    section_header("Secondary Dataset", "Upload a second dataset to merge or concatenate.")
    uploaded = st.file_uploader(
        "Upload secondary dataset",
        type=["csv", "xlsx", "xls", "tsv", "json"],
        help="Supported: CSV, Excel, TSV, JSON",
        key="merge_uploader",
    )

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df2 = pd.read_csv(uploaded)
            elif uploaded.name.endswith((".xlsx", ".xls")):
                df2 = pd.read_excel(uploaded)
            elif uploaded.name.endswith(".tsv"):
                df2 = pd.read_csv(uploaded, sep="\t")
            elif uploaded.name.endswith(".json"):
                df2 = pd.read_json(uploaded)
            else:
                df2 = None
            if df2 is not None:
                st.session_state["df_secondary"] = df2
        except Exception as e:
            st.error(f"Error reading secondary file: {e}")

    df2 = st.session_state.get("df_secondary")
    if df2 is None:
        st.info("Upload a secondary dataset to enable merge and concatenation.")
        return df

    # Preview both datasets
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Primary dataset**")
        st.caption(f"{df.shape[0]:,} rows x {df.shape[1]} columns")
        st.dataframe(df.head(5), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Secondary dataset**")
        st.caption(f"{df2.shape[0]:,} rows x {df2.shape[1]} columns")
        st.dataframe(df2.head(5), use_container_width=True, hide_index=True)

    st.divider()

    # ── Join ──
    section_header("Join", "Merge datasets on key columns.")
    join_type = st.selectbox(
        "Join type:", ["inner", "left", "right", "outer", "cross"],
        key="merge_join_type",
    )

    if join_type != "cross":
        c1, c2 = st.columns(2)
        left_keys = c1.multiselect(
            "Key columns (primary):", df.columns.tolist(), key="merge_left_keys",
        )
        right_keys = c2.multiselect(
            "Key columns (secondary):", df2.columns.tolist(), key="merge_right_keys",
        )

        if left_keys and right_keys:
            if len(left_keys) != len(right_keys):
                st.warning("Number of key columns must match between primary and secondary datasets.")
            else:
                try:
                    merged = pd.merge(
                        df, df2, left_on=left_keys, right_on=right_keys,
                        how=join_type, suffixes=("", "_right"),
                    )
                    st.caption(f"**Preview** — {merged.shape[0]:,} rows x {merged.shape[1]} columns:")
                    st.dataframe(merged.head(20), use_container_width=True, hide_index=True)

                    if st.button("Apply Join", key="apply_merge"):
                        df = merged.reset_index(drop=True)
                        st.success(f"Joined. New shape: {df.shape}")
                        st.session_state["df"] = df
                except Exception as e:
                    st.error(f"Join failed: {e}")
        else:
            st.info("Select key columns from both datasets.")
    else:
        # Cross join
        try:
            merged = pd.merge(df, df2, how="cross", suffixes=("", "_right"))
            st.caption(f"**Preview** — {merged.shape[0]:,} rows x {merged.shape[1]} columns:")
            st.dataframe(merged.head(20), use_container_width=True, hide_index=True)
            if merged.shape[0] > 100_000:
                st.warning(f"Cross join produces {merged.shape[0]:,} rows. This may be very large.")
            if st.button("Apply Cross Join", key="apply_cross_join"):
                df = merged.reset_index(drop=True)
                st.success(f"Cross-joined. New shape: {df.shape}")
                st.session_state["df"] = df
        except Exception as e:
            st.error(f"Cross join failed: {e}")

    st.divider()

    # ── Concatenate ──
    section_header("Concatenate", "Stack datasets together vertically or horizontally.")
    concat_axis = st.radio(
        "Direction:", ["Vertical (stack rows)", "Horizontal (add columns)"],
        key="concat_axis", horizontal=True,
    )
    axis = 0 if "Vertical" in concat_axis else 1

    if axis == 0:
        # Warn about column mismatches
        primary_cols = set(df.columns)
        secondary_cols = set(df2.columns)
        only_primary = primary_cols - secondary_cols
        only_secondary = secondary_cols - primary_cols
        if only_primary:
            st.warning(f"Columns only in primary: {', '.join(sorted(only_primary))} — will be filled with NaN in secondary rows.")
        if only_secondary:
            st.warning(f"Columns only in secondary: {', '.join(sorted(only_secondary))} — will be filled with NaN in primary rows.")

    try:
        concatenated = pd.concat([df, df2], axis=axis, ignore_index=True)
        st.caption(f"**Preview** — {concatenated.shape[0]:,} rows x {concatenated.shape[1]} columns:")
        st.dataframe(concatenated.head(20), use_container_width=True, hide_index=True)

        if st.button("Apply Concatenation", key="apply_concat"):
            df = concatenated.reset_index(drop=True)
            st.success(f"Concatenated. New shape: {df.shape}")
            st.session_state["df"] = df
    except Exception as e:
        st.error(f"Concatenation failed: {e}")

    return df


def _render_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """Sampling methods: random, stratified, systematic."""

    method = st.radio(
        "Sampling method:", ["Random", "Stratified", "Systematic"],
        key="sampling_method", horizontal=True,
    )

    if method == "Random":
        section_header("Random Sampling", "Draw a random sample from the dataset.")
        size_mode = st.radio("Specify sample size as:", ["Number of rows", "Percentage"], key="rand_size_mode", horizontal=True)
        if size_mode == "Number of rows":
            n_sample = st.number_input("Number of rows:", 1, len(df), min(100, len(df)), key="rand_n")
            frac = None
        else:
            pct = st.slider("Percentage:", 1, 100, 50, key="rand_pct")
            n_sample = None
            frac = pct / 100.0
        replace = st.checkbox("With replacement", value=False, key="rand_replace")
        random_seed = st.number_input("Random seed (0 = none):", 0, 99999, 42, key="rand_seed")
        seed_val = random_seed if random_seed > 0 else None

        try:
            sampled = df.sample(n=n_sample, frac=frac, replace=replace, random_state=seed_val)
            st.caption(f"**Preview** — {sampled.shape[0]:,} rows sampled:")
            st.dataframe(sampled.head(20), use_container_width=True, hide_index=True)

            if st.button("Apply Random Sample", key="apply_rand_sample"):
                df = sampled.reset_index(drop=True)
                st.success(f"Sampled {len(df):,} rows.")
                st.session_state["df"] = df
        except Exception as e:
            st.error(f"Random sampling failed: {e}")

    elif method == "Stratified":
        section_header("Stratified Sampling", "Sample proportionally or with fixed count per group.")
        cat_cols = df.columns.tolist()
        strata_col = st.selectbox("Strata column:", cat_cols, key="strat_col")
        strata_counts = df[strata_col].value_counts()

        st.caption(f"**Strata distribution** ({strata_counts.shape[0]} groups):")
        st.dataframe(
            pd.DataFrame({"Stratum": strata_counts.index, "Count": strata_counts.values}),
            use_container_width=True, hide_index=True,
        )

        strat_mode = st.radio(
            "Sampling mode:", ["Proportional", "Fixed N per stratum"],
            key="strat_mode", horizontal=True,
        )
        random_seed = st.number_input("Random seed (0 = none):", 0, 99999, 42, key="strat_seed")
        seed_val = random_seed if random_seed > 0 else None

        if strat_mode == "Proportional":
            prop = st.slider("Proportion to sample from each stratum:", 0.01, 1.0, 0.5, 0.01, key="strat_prop")
            try:
                sampled = df.groupby(strata_col, group_keys=False).apply(
                    lambda x: x.sample(frac=prop, random_state=seed_val)
                )
                st.caption(f"**Preview** — {sampled.shape[0]:,} rows sampled:")
                st.dataframe(sampled.head(20), use_container_width=True, hide_index=True)
                sampled_counts = sampled[strata_col].value_counts()
                st.dataframe(
                    pd.DataFrame({"Stratum": sampled_counts.index, "Sampled": sampled_counts.values}),
                    use_container_width=True, hide_index=True,
                )

                if st.button("Apply Stratified Sample", key="apply_strat_prop"):
                    df = sampled.reset_index(drop=True)
                    st.success(f"Stratified sample: {len(df):,} rows.")
                    st.session_state["df"] = df
            except Exception as e:
                st.error(f"Stratified sampling failed: {e}")
        else:
            min_group = int(strata_counts.min())
            n_per = st.number_input(
                "N per stratum:", 1, min_group,
                min(min_group, 10), key="strat_fixed_n",
            )
            try:
                sampled = df.groupby(strata_col, group_keys=False).apply(
                    lambda x: x.sample(n=n_per, random_state=seed_val)
                )
                st.caption(f"**Preview** — {sampled.shape[0]:,} rows sampled:")
                st.dataframe(sampled.head(20), use_container_width=True, hide_index=True)

                if st.button("Apply Stratified Sample", key="apply_strat_fixed"):
                    df = sampled.reset_index(drop=True)
                    st.success(f"Stratified sample: {len(df):,} rows ({n_per} per stratum).")
                    st.session_state["df"] = df
            except Exception as e:
                st.error(f"Stratified sampling failed: {e}")

    elif method == "Systematic":
        section_header("Systematic Sampling", "Select every Nth row from the dataset.")
        step = st.number_input("Select every Nth row:", 2, max(2, len(df)), 10, key="sys_step")
        random_start = st.checkbox("Random start offset", value=False, key="sys_random_start")

        if random_start:
            start = np.random.randint(0, step)
            st.caption(f"Random start offset: {start}")
        else:
            start = 0

        indices = list(range(start, len(df), step))
        sampled = df.iloc[indices]
        st.caption(f"**Preview** — {sampled.shape[0]:,} rows sampled (every {step}th row, starting at row {start}):")
        st.dataframe(sampled.head(20), use_container_width=True, hide_index=True)

        if st.button("Apply Systematic Sample", key="apply_sys_sample"):
            df = sampled.reset_index(drop=True)
            st.success(f"Systematic sample: {len(df):,} rows.")
            st.session_state["df"] = df

    return df
