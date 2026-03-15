"""
Visual Dataset Editor - Spreadsheet-like editing with undo, find & replace,
row operations, validation rules, and edit history.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

from modules.ui_helpers import section_header, empty_state, help_tip


# ─── Constants ────────────────────────────────────────────────────────────

_MAX_HISTORY = 20
_MAX_HISTORY_LARGE = 5
_LARGE_DF_BYTES = 50_000_000  # 50 MB


# ─── Session State Helpers ────────────────────────────────────────────────

def _init_session_state(df: pd.DataFrame):
    """Initialize or reset all dse_ session state keys.

    Detects dataset switches via data_name and reinitializes when needed.
    """
    current_name = st.session_state.get("data_name", "")
    needs_init = (
        "dse_original_df" not in st.session_state
        or st.session_state.get("dse_original_name", "") != current_name
    )

    if needs_init:
        st.session_state["dse_original_df"] = df.copy()
        st.session_state["dse_original_name"] = current_name
        st.session_state["dse_history"] = [{
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "label": "Initial state",
            "df": df.copy(),
        }]
        st.session_state["dse_history_pointer"] = 0
        st.session_state["dse_validation_rules"] = st.session_state.get("dse_validation_rules", [])
        st.session_state["dse_edit_buffer"] = df.copy()
        st.session_state["dse_buffer_version"] = 0


def _push_history(label: str):
    """Deep-copy current session df into history, trim to cap."""
    df = st.session_state["df"]
    mem = df.memory_usage(deep=True).sum()
    cap = _MAX_HISTORY_LARGE if mem > _LARGE_DF_BYTES else _MAX_HISTORY

    pointer = st.session_state.get("dse_history_pointer", 0)
    history = st.session_state.get("dse_history", [])

    # Trim any forward states if we branched from an undo point
    history = history[: pointer + 1]

    history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "label": label,
        "df": df.copy(),
    })

    # Enforce cap
    if len(history) > cap:
        history = history[-cap:]

    st.session_state["dse_history"] = history
    st.session_state["dse_history_pointer"] = len(history) - 1


def _undo_to(index: int):
    """Restore df from history at given index."""
    history = st.session_state.get("dse_history", [])
    if 0 <= index < len(history):
        st.session_state["df"] = history[index]["df"].copy()
        st.session_state["dse_history_pointer"] = index
        st.session_state["dse_edit_buffer"] = st.session_state["df"].copy()
        st.session_state["dse_buffer_version"] = st.session_state.get("dse_buffer_version", 0) + 1


def _compute_change_summary(original: pd.DataFrame, edited: pd.DataFrame) -> dict:
    """Compare two DataFrames and return a change summary dict."""
    summary = {"cells_changed": 0, "rows_added": 0, "rows_deleted": 0}

    orig_len = len(original)
    edit_len = len(edited)

    if edit_len > orig_len:
        summary["rows_added"] = edit_len - orig_len
    elif edit_len < orig_len:
        summary["rows_deleted"] = orig_len - edit_len

    # Count cell-level changes in the overlapping portion
    common_rows = min(orig_len, edit_len)
    common_cols = original.columns.intersection(edited.columns)
    if common_rows > 0 and len(common_cols) > 0:
        orig_slice = original.iloc[:common_rows][common_cols].reset_index(drop=True)
        edit_slice = edited.iloc[:common_rows][common_cols].reset_index(drop=True)
        try:
            diff_mask = orig_slice.ne(edit_slice)
            # Handle NaN comparison (NaN != NaN is True, but we don't want to count both-NaN as a change)
            both_null = orig_slice.isna() & edit_slice.isna()
            diff_mask = diff_mask & ~both_null
            summary["cells_changed"] = int(diff_mask.sum().sum())
        except Exception:
            pass

    return summary


# ─── Tab 1: Edit Data ────────────────────────────────────────────────────

def _render_edit_data(df: pd.DataFrame):
    """Inline spreadsheet editor with add/delete rows and columns."""

    section_header("Inline Editor", "Click any cell to edit. Use the + button to add rows.")

    # Sync buffer if another module changed the df
    buf = st.session_state.get("dse_edit_buffer")
    if buf is None or buf.shape != df.shape or not buf.columns.equals(df.columns):
        st.session_state["dse_edit_buffer"] = df.copy()
        st.session_state["dse_buffer_version"] = st.session_state.get("dse_buffer_version", 0) + 1

    buffer = st.session_state["dse_edit_buffer"]
    version = st.session_state.get("dse_buffer_version", 0)

    # Build dynamic column config
    col_config = {}
    for col in buffer.columns:
        dtype = buffer[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            col_config[col] = st.column_config.CheckboxColumn(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            col_config[col] = st.column_config.DatetimeColumn(col)
        elif pd.api.types.is_integer_dtype(dtype):
            col_config[col] = st.column_config.NumberColumn(col, format="%d")
        elif pd.api.types.is_float_dtype(dtype):
            col_config[col] = st.column_config.NumberColumn(col)

    if len(df) > 10_000:
        st.caption("Large dataset detected. Editor performance may vary with 10K+ rows.")

    edited = st.data_editor(
        buffer,
        num_rows="dynamic",
        use_container_width=True,
        column_config=col_config,
        key=f"dse_editor_{version}",
    )

    # Change summary
    summary = _compute_change_summary(buffer, edited)
    total_changes = summary["cells_changed"] + summary["rows_added"] + summary["rows_deleted"]

    if total_changes > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Cells Modified", summary["cells_changed"])
        c2.metric("Rows Added", summary["rows_added"])
        c3.metric("Rows Deleted", summary["rows_deleted"])

    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Apply Changes", type="primary", key="dse_apply",
                 disabled=total_changes == 0):
        st.session_state["df"] = edited.copy()
        st.session_state["dse_edit_buffer"] = edited.copy()
        _push_history(f"Edit data ({summary['cells_changed']} cells, "
                      f"+{summary['rows_added']}/-{summary['rows_deleted']} rows)")
        st.rerun()
    if c2.button("Discard Changes", key="dse_discard"):
        st.session_state["dse_edit_buffer"] = st.session_state["df"].copy()
        st.session_state["dse_buffer_version"] = version + 1
        st.rerun()

    # ── Add Column ──
    st.divider()
    section_header("Add Column")
    ac1, ac2, ac3 = st.columns(3)
    new_col_name = ac1.text_input("Column name:", key="dse_new_col_name")
    new_col_type = ac2.selectbox("Type:", ["Text", "Integer", "Float", "Boolean", "Datetime"],
                                 key="dse_new_col_type")
    new_col_default = ac3.text_input("Default value:", key="dse_new_col_default",
                                     help="Leave blank for NaN/empty.")

    if st.button("Add Column", key="dse_add_col"):
        if not new_col_name:
            st.error("Column name cannot be empty.")
        elif new_col_name in st.session_state["df"].columns:
            st.error(f"Column '{new_col_name}' already exists.")
        else:
            current = st.session_state["df"].copy()
            default = new_col_default.strip() if new_col_default.strip() else None
            try:
                if new_col_type == "Integer":
                    current[new_col_name] = int(default) if default else pd.NA
                    current[new_col_name] = current[new_col_name].astype("Int64")
                elif new_col_type == "Float":
                    current[new_col_name] = float(default) if default else np.nan
                elif new_col_type == "Boolean":
                    current[new_col_name] = default.lower() in ("true", "1", "yes") if default else False
                elif new_col_type == "Datetime":
                    current[new_col_name] = pd.to_datetime(default) if default else pd.NaT
                else:
                    current[new_col_name] = default if default else ""
                st.session_state["df"] = current
                st.session_state["dse_edit_buffer"] = current.copy()
                st.session_state["dse_buffer_version"] = version + 1
                _push_history(f"Add column '{new_col_name}'")
                st.rerun()
            except (ValueError, TypeError) as e:
                st.error(f"Invalid default value: {e}")

    # ── Delete Columns ──
    st.divider()
    section_header("Delete Columns")
    del_cols = st.multiselect("Select columns to delete:", df.columns.tolist(),
                              key="dse_del_cols")
    if del_cols:
        st.caption(f"{len(del_cols)} column(s) selected for deletion.")
        confirm_del = st.checkbox("I confirm deletion of these columns", key="dse_confirm_del")
        if st.button("Delete Selected Columns", key="dse_del_btn", disabled=not confirm_del):
            current = st.session_state["df"].drop(columns=del_cols)
            st.session_state["df"] = current
            st.session_state["dse_edit_buffer"] = current.copy()
            st.session_state["dse_buffer_version"] = version + 1
            _push_history(f"Delete {len(del_cols)} column(s)")
            st.rerun()

    # ── Reorder Columns ──
    st.divider()
    section_header("Reorder Columns", "Drag items in the multiselect to set the new order.")
    current_order = st.session_state["df"].columns.tolist()
    new_order = st.multiselect("Column order:", current_order, default=current_order,
                               key="dse_col_order")
    if new_order and set(new_order) == set(current_order) and new_order != current_order:
        if st.button("Apply Column Order", key="dse_apply_order"):
            current = st.session_state["df"][new_order]
            st.session_state["df"] = current
            st.session_state["dse_edit_buffer"] = current.copy()
            st.session_state["dse_buffer_version"] = version + 1
            _push_history("Reorder columns")
            st.rerun()
    elif new_order and len(new_order) < len(current_order):
        st.caption("All columns must be included to reorder.")


# ─── Tab 2: Find & Replace ───────────────────────────────────────────────

def _render_find_replace(df: pd.DataFrame):
    """Cross-column find & replace with regex support and preview."""

    section_header("Find & Replace")

    col_options = ["All Columns"] + df.columns.tolist()
    c1, c2 = st.columns([2, 1])
    scope = c1.selectbox("Search in:", col_options, key="dse_fr_scope")
    search_term = c2.text_input("Search for:", key="dse_fr_search")

    c1, c2 = st.columns(2)
    case_sensitive = c1.checkbox("Case sensitive", value=False, key="dse_fr_case")
    use_regex = c2.checkbox("Use regex", value=False, key="dse_fr_regex")

    if not search_term:
        st.caption("Enter a search term to find matches.")
        return

    # Validate regex
    if use_regex:
        try:
            re.compile(search_term)
        except re.error as e:
            st.error(f"Invalid regex pattern: {e}")
            return

    # Search
    flags = 0 if case_sensitive else re.IGNORECASE
    search_cols = df.columns.tolist() if scope == "All Columns" else [scope]

    matches = []
    for col in search_cols:
        str_series = df[col].astype(str).fillna("")
        if use_regex:
            mask = str_series.str.contains(search_term, flags=flags, na=False, regex=True)
        else:
            mask = str_series.str.contains(re.escape(search_term), flags=flags, na=False, regex=True)

        matched_idx = df.index[mask]
        for idx in matched_idx:
            matches.append({
                "Row": idx,
                "Column": col,
                "Value": str(df.at[idx, col]),
            })

    if not matches:
        st.info("No matches found.")
        return

    match_df = pd.DataFrame(matches)
    total_matches = len(match_df)
    unique_rows = match_df["Row"].nunique()
    unique_cols = match_df["Column"].nunique()

    st.success(f"**{total_matches:,}** matches in **{unique_rows:,}** rows across **{unique_cols}** column(s)")

    display_limit = 500
    if total_matches > display_limit:
        st.caption(f"Showing first {display_limit} of {total_matches:,} matches.")
    st.dataframe(match_df.head(display_limit), use_container_width=True)

    # Replace
    st.divider()
    section_header("Replace")
    replacement = st.text_input("Replace with:", key="dse_fr_replace")

    # Warn about numeric columns
    numeric_cols_affected = [c for c in match_df["Column"].unique()
                             if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols_affected:
        st.warning(f"Replacing in numeric column(s) ({', '.join(numeric_cols_affected)}) "
                   f"will convert them to text.")

    if st.button("Replace All", type="primary", key="dse_fr_replace_all"):
        current = st.session_state["df"].copy()
        count = 0
        for col in search_cols:
            str_series = current[col].astype(str).fillna("")
            if use_regex:
                new_series = str_series.str.replace(search_term, replacement,
                                                    flags=flags, regex=True)
            else:
                new_series = str_series.str.replace(search_term, replacement,
                                                    case=case_sensitive, regex=False)
            changed = str_series != new_series
            count += changed.sum()
            if changed.any():
                current[col] = new_series

        st.session_state["df"] = current
        st.session_state["dse_edit_buffer"] = current.copy()
        st.session_state["dse_buffer_version"] = st.session_state.get("dse_buffer_version", 0) + 1
        _push_history(f"Replace '{search_term}' → '{replacement}' ({count} cells)")
        st.rerun()


# ─── Tab 3: Row Operations ───────────────────────────────────────────────

def _render_row_operations(df: pd.DataFrame):
    """Insert, duplicate, multi-sort, and conditional delete rows."""

    # ── Insert Rows ──
    section_header("Insert Rows")
    c1, c2 = st.columns(2)
    position = c1.radio("Position:", ["Top", "Bottom", "After Row Index"],
                        horizontal=True, key="dse_insert_pos")
    n_insert = c2.number_input("Number of rows:", min_value=1, value=1, key="dse_insert_n")

    after_idx = 0
    if position == "After Row Index":
        after_idx = st.number_input("After row index:", min_value=0,
                                    max_value=max(0, len(df) - 1), value=0,
                                    key="dse_insert_idx")

    if st.button("Insert Rows", key="dse_insert_btn"):
        current = st.session_state["df"]
        new_rows = pd.DataFrame(
            {col: [np.nan] * n_insert for col in current.columns}
        )
        if position == "Top":
            result = pd.concat([new_rows, current], ignore_index=True)
        elif position == "Bottom":
            result = pd.concat([current, new_rows], ignore_index=True)
        else:
            top = current.iloc[: after_idx + 1]
            bottom = current.iloc[after_idx + 1:]
            result = pd.concat([top, new_rows, bottom], ignore_index=True)
        st.session_state["df"] = result
        st.session_state["dse_edit_buffer"] = result.copy()
        st.session_state["dse_buffer_version"] = st.session_state.get("dse_buffer_version", 0) + 1
        _push_history(f"Insert {n_insert} row(s) at {position.lower()}")
        st.rerun()

    # ── Duplicate Rows ──
    st.divider()
    section_header("Duplicate Rows")
    row_spec = st.text_input("Row indices (e.g. 0,3,5 or 2-7):", key="dse_dup_spec")
    n_copies = st.number_input("Number of copies:", min_value=1, value=1, key="dse_dup_copies")

    if row_spec.strip():
        try:
            indices = _parse_row_indices(row_spec, len(df))
            st.caption(f"{len(indices)} row(s) selected.")
            st.dataframe(df.iloc[indices].head(20), use_container_width=True)
            if st.button("Duplicate", key="dse_dup_btn"):
                current = st.session_state["df"]
                duped = current.iloc[indices]
                parts = [current] + [duped] * n_copies
                result = pd.concat(parts, ignore_index=True)
                st.session_state["df"] = result
                st.session_state["dse_edit_buffer"] = result.copy()
                st.session_state["dse_buffer_version"] = st.session_state.get("dse_buffer_version", 0) + 1
                _push_history(f"Duplicate {len(indices)} row(s) ×{n_copies}")
                st.rerun()
        except ValueError as e:
            st.error(str(e))

    # ── Multi-Column Sort ──
    st.divider()
    section_header("Multi-Column Sort")
    sort_cols = []
    sort_asc = []
    for i in range(3):
        c1, c2 = st.columns([3, 1])
        col = c1.selectbox(f"Sort level {i + 1}:", ["(none)"] + df.columns.tolist(),
                           key=f"dse_sort_col_{i}")
        asc = c2.checkbox("Ascending", value=True, key=f"dse_sort_asc_{i}")
        if col != "(none)":
            sort_cols.append(col)
            sort_asc.append(asc)

    if sort_cols:
        if st.button("Apply Sort", key="dse_sort_btn"):
            current = st.session_state["df"].sort_values(
                by=sort_cols, ascending=sort_asc, ignore_index=True
            )
            st.session_state["df"] = current
            st.session_state["dse_edit_buffer"] = current.copy()
            st.session_state["dse_buffer_version"] = st.session_state.get("dse_buffer_version", 0) + 1
            _push_history(f"Sort by {', '.join(sort_cols)}")
            st.rerun()

    # ── Conditional Row Deletion ──
    st.divider()
    section_header("Conditional Row Deletion")
    c1, c2 = st.columns(2)
    cond_col = c1.selectbox("Column:", df.columns.tolist(), key="dse_cond_col")
    operators = ["equals", "not equals", "greater than", "less than",
                 "contains", "is null", "is not null"]
    cond_op = c2.selectbox("Operator:", operators, key="dse_cond_op")

    value_disabled = cond_op in ("is null", "is not null")
    cond_val = st.text_input("Value:", key="dse_cond_val", disabled=value_disabled)

    # Build mask
    mask = _build_condition_mask(df, cond_col, cond_op, cond_val)
    n_match = mask.sum() if mask is not None else 0

    if mask is not None and n_match > 0:
        st.caption(f"**{n_match:,}** row(s) match this condition.")
        with st.expander("Preview matching rows"):
            st.dataframe(df[mask].head(100), use_container_width=True)

        if n_match == len(df):
            st.warning("This would delete all rows. Operation blocked.")
        else:
            confirm = st.checkbox("I confirm deletion", key="dse_cond_confirm")
            if st.button("Delete Matching Rows", key="dse_cond_del",
                         disabled=not confirm, type="primary"):
                current = st.session_state["df"][~mask].reset_index(drop=True)
                st.session_state["df"] = current
                st.session_state["dse_edit_buffer"] = current.copy()
                st.session_state["dse_buffer_version"] = st.session_state.get("dse_buffer_version", 0) + 1
                _push_history(f"Delete {n_match} rows ({cond_col} {cond_op} {cond_val})")
                st.rerun()
    elif mask is not None:
        st.info("No rows match this condition.")


def _parse_row_indices(spec: str, max_len: int) -> list[int]:
    """Parse a comma/dash index specification like '0,3,5-10' into a list of ints."""
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start.strip()), int(end.strip())
            if start < 0 or end >= max_len:
                raise ValueError(f"Index range {start}-{end} out of bounds (max {max_len - 1}).")
            indices.update(range(start, end + 1))
        else:
            idx = int(part)
            if idx < 0 or idx >= max_len:
                raise ValueError(f"Index {idx} out of bounds (max {max_len - 1}).")
            indices.add(idx)
    return sorted(indices)


def _build_condition_mask(df: pd.DataFrame, col: str, op: str, val: str):
    """Return a boolean Series mask for the given condition."""
    try:
        series = df[col]
        if op == "is null":
            return series.isna()
        if op == "is not null":
            return series.notna()
        if op == "contains":
            return series.astype(str).str.contains(val, case=True, na=False, regex=False)

        # Attempt numeric comparison
        if pd.api.types.is_numeric_dtype(series):
            try:
                v = float(val)
            except ValueError:
                v = val
        else:
            v = val

        if op == "equals":
            if pd.api.types.is_numeric_dtype(series) and isinstance(v, (int, float)):
                return series == v
            return series.astype(str) == str(val)
        if op == "not equals":
            if pd.api.types.is_numeric_dtype(series) and isinstance(v, (int, float)):
                return series != v
            return series.astype(str) != str(val)
        if op == "greater than":
            return series > v
        if op == "less than":
            return series < v
    except Exception:
        return None
    return None


# ─── Tab 4: Validation Rules ─────────────────────────────────────────────

def _render_validation_rules(df: pd.DataFrame):
    """Define, run, and auto-fix validation rules per column."""

    # ── Define Rules ──
    section_header("Define Rule")
    c1, c2 = st.columns(2)
    rule_col = c1.selectbox("Column:", df.columns.tolist(), key="dse_vr_col")
    rule_types = ["Not Null", "Data Type", "Range (min/max)", "Allowed Values", "Regex Pattern"]
    rule_type = c2.selectbox("Rule type:", rule_types, key="dse_vr_type")

    params = {}
    if rule_type == "Data Type":
        expected = st.selectbox("Expected type:",
                                ["int64", "float64", "object", "bool", "datetime64[ns]"],
                                key="dse_vr_dtype")
        params["expected"] = expected
    elif rule_type == "Range (min/max)":
        rc1, rc2 = st.columns(2)
        rmin = rc1.text_input("Min (leave blank for none):", key="dse_vr_min")
        rmax = rc2.text_input("Max (leave blank for none):", key="dse_vr_max")
        params["min"] = float(rmin) if rmin.strip() else None
        params["max"] = float(rmax) if rmax.strip() else None
    elif rule_type == "Allowed Values":
        allowed = st.text_area("Allowed values (comma-separated):", key="dse_vr_allowed")
        params["values"] = [v.strip() for v in allowed.split(",") if v.strip()]
    elif rule_type == "Regex Pattern":
        pattern = st.text_input("Regex pattern:", key="dse_vr_regex")
        params["pattern"] = pattern

    if st.button("Add Rule", key="dse_vr_add"):
        if rule_type == "Regex Pattern":
            try:
                re.compile(params.get("pattern", ""))
            except re.error as e:
                st.error(f"Invalid regex: {e}")
                return
        if rule_type == "Allowed Values" and not params.get("values"):
            st.error("Enter at least one allowed value.")
            return

        rules = st.session_state.get("dse_validation_rules", [])
        rules.append({
            "id": str(uuid.uuid4())[:8],
            "column": rule_col,
            "rule_type": rule_type,
            "params": params,
            "enabled": True,
        })
        st.session_state["dse_validation_rules"] = rules
        st.rerun()

    # ── Rules List ──
    st.divider()
    section_header("Active Rules")
    rules = st.session_state.get("dse_validation_rules", [])

    if not rules:
        empty_state("No validation rules defined.", "Add rules above to check your data quality.")
        return

    for i, rule in enumerate(rules):
        c1, c2, c3, c4 = st.columns([3, 3, 1, 1])
        c1.markdown(f"**{rule['column']}** — {rule['rule_type']}")
        param_str = _format_rule_params(rule)
        c2.caption(param_str)
        new_enabled = c3.checkbox("On", value=rule["enabled"], key=f"dse_vr_en_{rule['id']}")
        rules[i]["enabled"] = new_enabled
        if c4.button("✕", key=f"dse_vr_del_{rule['id']}"):
            rules.pop(i)
            st.session_state["dse_validation_rules"] = rules
            st.rerun()

    st.session_state["dse_validation_rules"] = rules

    # ── Run Validation ──
    st.divider()
    if st.button("Run Validation", type="primary", key="dse_vr_run"):
        violations = _run_validation(df, rules)
        if violations.empty:
            st.success("All rules passed — no violations found.")
        else:
            st.error(f"**{len(violations):,}** violation(s) found across "
                     f"**{violations['Rule'].nunique()}** rule(s).")
            st.dataframe(violations, use_container_width=True)

            # Export
            csv = violations.to_csv(index=False)
            st.download_button("Download Violations CSV", csv,
                               "validation_violations.csv", "text/csv",
                               key="dse_vr_export")

            # Auto-fix
            st.divider()
            section_header("Auto-Fix")
            help_tip("Auto-Fix", """
Automatic fixes by rule type:
- **Not Null**: Fill with column mean (numeric), mode (categorical), or empty string.
- **Data Type**: Coerce to the expected type (invalid values become NaN).
- **Range**: Clip values to the specified min/max bounds.
- **Allowed Values**: Replace disallowed values with NaN.
- **Regex**: No auto-fix available (patterns are too ambiguous).
""")
            if st.button("Auto-Fix All", key="dse_vr_autofix"):
                fixed = _auto_fix(st.session_state["df"].copy(), rules)
                st.session_state["df"] = fixed
                st.session_state["dse_edit_buffer"] = fixed.copy()
                st.session_state["dse_buffer_version"] = st.session_state.get("dse_buffer_version", 0) + 1
                _push_history("Auto-fix validation violations")
                st.rerun()


def _format_rule_params(rule: dict) -> str:
    """Format rule parameters as a short summary string."""
    rt = rule["rule_type"]
    p = rule["params"]
    if rt == "Not Null":
        return "Must not be null"
    if rt == "Data Type":
        return f"Expected: {p.get('expected', '?')}"
    if rt == "Range (min/max)":
        parts = []
        if p.get("min") is not None:
            parts.append(f"min={p['min']}")
        if p.get("max") is not None:
            parts.append(f"max={p['max']}")
        return ", ".join(parts) if parts else "No bounds set"
    if rt == "Allowed Values":
        vals = p.get("values", [])
        if len(vals) <= 5:
            return f"Allowed: {', '.join(vals)}"
        return f"Allowed: {', '.join(vals[:5])}... (+{len(vals) - 5})"
    if rt == "Regex Pattern":
        return f"Pattern: {p.get('pattern', '')}"
    return ""


def _run_validation(df: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """Run all enabled rules and return a violations DataFrame."""
    records = []
    for rule in rules:
        if not rule["enabled"]:
            continue
        col = rule["column"]
        if col not in df.columns:
            records.append({
                "Row": "-", "Column": col, "Rule": rule["rule_type"],
                "Value": "-", "Expected": f"Column '{col}' not found",
            })
            continue

        series = df[col]
        rt = rule["rule_type"]
        p = rule["params"]

        if rt == "Not Null":
            for idx in series[series.isna()].index:
                records.append({
                    "Row": idx, "Column": col, "Rule": rt,
                    "Value": "NaN/NULL", "Expected": "Not null",
                })
        elif rt == "Data Type":
            expected = p.get("expected", "")
            if str(series.dtype) != expected:
                bad = series[pd.to_numeric(series, errors="coerce").isna()] if "int" in expected or "float" in expected else series
                for idx in bad.index[:100]:
                    records.append({
                        "Row": idx, "Column": col, "Rule": rt,
                        "Value": str(series.at[idx]),
                        "Expected": f"dtype={expected}",
                    })
        elif rt == "Range (min/max)":
            numeric = pd.to_numeric(series, errors="coerce")
            if p.get("min") is not None:
                mask = numeric < p["min"]
                for idx in series[mask].index:
                    records.append({
                        "Row": idx, "Column": col, "Rule": rt,
                        "Value": str(series.at[idx]),
                        "Expected": f">= {p['min']}",
                    })
            if p.get("max") is not None:
                mask = numeric > p["max"]
                for idx in series[mask].index:
                    records.append({
                        "Row": idx, "Column": col, "Rule": rt,
                        "Value": str(series.at[idx]),
                        "Expected": f"<= {p['max']}",
                    })
        elif rt == "Allowed Values":
            allowed = set(p.get("values", []))
            str_series = series.astype(str)
            mask = ~str_series.isin(allowed) & series.notna()
            for idx in series[mask].index:
                records.append({
                    "Row": idx, "Column": col, "Rule": rt,
                    "Value": str(series.at[idx]),
                    "Expected": f"One of: {', '.join(list(allowed)[:5])}",
                })
        elif rt == "Regex Pattern":
            pattern = p.get("pattern", "")
            if pattern:
                str_series = series.astype(str)
                mask = ~str_series.str.match(pattern, na=False) & series.notna()
                for idx in series[mask].index:
                    records.append({
                        "Row": idx, "Column": col, "Rule": rt,
                        "Value": str(series.at[idx]),
                        "Expected": f"Match /{pattern}/",
                    })

    return pd.DataFrame(records) if records else pd.DataFrame()


def _auto_fix(df: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """Apply automatic fixes for each enabled rule."""
    for rule in rules:
        if not rule["enabled"]:
            continue
        col = rule["column"]
        if col not in df.columns:
            continue

        rt = rule["rule_type"]
        p = rule["params"]

        if rt == "Not Null":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                mode = df[col].mode()
                fill = mode.iloc[0] if len(mode) > 0 else ""
                df[col] = df[col].fillna(fill)
        elif rt == "Data Type":
            expected = p.get("expected", "")
            try:
                if "int" in expected:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif "float" in expected:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif expected == "datetime64[ns]":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif expected == "bool":
                    df[col] = df[col].astype(bool)
                else:
                    df[col] = df[col].astype(str)
            except Exception:
                pass
        elif rt == "Range (min/max)":
            numeric = pd.to_numeric(df[col], errors="coerce")
            if p.get("min") is not None:
                numeric = numeric.clip(lower=p["min"])
            if p.get("max") is not None:
                numeric = numeric.clip(upper=p["max"])
            df[col] = numeric
        elif rt == "Allowed Values":
            allowed = set(p.get("values", []))
            mask = ~df[col].astype(str).isin(allowed) & df[col].notna()
            df.loc[mask, col] = np.nan
        # Regex: no auto-fix

    return df


# ─── Tab 5: Edit History ─────────────────────────────────────────────────

def _render_edit_history(df: pd.DataFrame):
    """View history, undo/redo, and reset to original."""

    history = st.session_state.get("dse_history", [])
    pointer = st.session_state.get("dse_history_pointer", 0)

    if not history:
        empty_state("No edit history yet.", "Make some changes to start tracking history.")
        return

    # Controls
    section_header("Controls")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Undo", key="dse_undo", disabled=pointer <= 0):
        _undo_to(pointer - 1)
        st.rerun()
    if c2.button("Redo", key="dse_redo", disabled=pointer >= len(history) - 1):
        _undo_to(pointer + 1)
        st.rerun()

    confirm_reset = c3.checkbox("Confirm", key="dse_reset_confirm")
    if c4.button("Reset to Original", key="dse_reset", disabled=not confirm_reset):
        original = st.session_state.get("dse_original_df")
        if original is not None:
            st.session_state["df"] = original.copy()
            st.session_state["dse_edit_buffer"] = original.copy()
            st.session_state["dse_buffer_version"] = st.session_state.get("dse_buffer_version", 0) + 1
            _push_history("Reset to original")
            st.rerun()

    # Memory estimate
    mem_total = sum(h["df"].memory_usage(deep=True).sum() for h in history)
    mem_mb = mem_total / (1024 * 1024)
    cap = _MAX_HISTORY_LARGE if mem_mb > 50 else _MAX_HISTORY
    st.caption(f"History: {len(history)} snapshots ({mem_mb:.1f} MB) — max {cap}")

    if st.button("Clear History", key="dse_clear_hist"):
        st.session_state["dse_history"] = [{
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "label": "Current state",
            "df": st.session_state["df"].copy(),
        }]
        st.session_state["dse_history_pointer"] = 0
        st.rerun()

    # Timeline
    st.divider()
    section_header("Timeline")
    for i, entry in enumerate(reversed(history)):
        real_idx = len(history) - 1 - i
        is_current = real_idx == pointer
        shape = entry["df"].shape

        marker = " **(current)**" if is_current else ""
        st.markdown(
            f"**{entry['timestamp']}** — {entry['label']}{marker}  \n"
            f"Shape: {shape[0]:,} rows × {shape[1]} cols"
        )

        if not is_current:
            # Diff summary
            diff = _compute_change_summary(entry["df"], df)
            parts = []
            if diff["cells_changed"]:
                parts.append(f"{diff['cells_changed']} cell(s) differ")
            if diff["rows_added"]:
                parts.append(f"{diff['rows_added']} row(s) added since")
            if diff["rows_deleted"]:
                parts.append(f"{diff['rows_deleted']} row(s) removed since")
            if parts:
                st.caption("vs current: " + ", ".join(parts))

            if st.button("Restore", key=f"dse_restore_{real_idx}"):
                _undo_to(real_idx)
                st.rerun()

        st.divider()


# ─── Public Entry Point ──────────────────────────────────────────────────

def render_dataset_editor(df: pd.DataFrame):
    """Main render function for the Visual Dataset Editor module."""

    if df is None or df.empty:
        empty_state(
            "No data loaded.",
            "Upload a dataset or select a sample from the sidebar.",
        )
        return None

    _init_session_state(df)

    help_tip("Visual Dataset Editor", """
This module provides spreadsheet-like editing for your dataset:
- **Edit Data**: Click cells to modify values, add/delete rows and columns.
- **Find & Replace**: Search across columns with regex support.
- **Row Operations**: Insert, duplicate, sort, and conditionally delete rows.
- **Validation Rules**: Define quality rules per column and auto-fix violations.
- **Edit History**: Undo any change or reset to the original dataset.
""")

    tabs = st.tabs([
        "Edit Data", "Find & Replace", "Row Operations",
        "Validation Rules", "Edit History",
    ])

    # Always read the latest df from session state
    current_df = st.session_state["df"]

    with tabs[0]:
        _render_edit_data(current_df)
    with tabs[1]:
        _render_find_replace(current_df)
    with tabs[2]:
        _render_row_operations(current_df)
    with tabs[3]:
        _render_validation_rules(current_df)
    with tabs[4]:
        _render_edit_history(current_df)

    return None
