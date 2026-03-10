"""
Analysis Templates Module - Save and load analysis configurations as JSON.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime

from modules.ui_helpers import section_header, empty_state, help_tip


# Keys to capture per module
_MODULE_STATE_KEYS = {
    "Descriptive Statistics": ["desc_"],
    "Visualization": ["viz_"],
    "Hypothesis Testing": ["ht_", "hyp_"],
    "Correlation & Multivariate": ["corr_", "pca_", "fa_", "tsne_", "mds_", "lda_qda_"],
    "Regression": ["reg_", "logistic_", "glm_", "robust_", "mixed_", "regularized_", "nonlinear_", "profiler_"],
    "ANOVA": ["anova_"],
    "Time Series": ["ts_"],
    "Machine Learning": ["ml_"],
    "Survival Analysis": ["surv_", "km_", "cox_"],
    "Quality & SPC": ["spc_", "qual_", "gage_", "accept_"],
    "DOE": ["doe_", "desg_", "rsm_", "desir_", "mix_"],
    "Text Analytics": ["te_", "sent_", "tf_"],
    "Monte Carlo Simulation": ["mc_", "ps_", "ra_"],
}


def render_templates():
    """Render the template save/load interface."""
    section_header("Analysis Templates")
    help_tip("Analysis Templates",
             "Save your current analysis settings as a template file (JSON) and reload them later.")

    tabs = st.tabs(["Save Template", "Load Template"])

    with tabs[0]:
        _render_save_template()
    with tabs[1]:
        _render_load_template()


def _render_save_template():
    section_header("Save Current Settings")

    template_name = st.text_input("Template name:", value="My Analysis", key="tmpl_name")
    description = st.text_area("Description (optional):", key="tmpl_desc", height=80)

    # Select which modules to save
    modules = st.multiselect(
        "Modules to include:",
        list(_MODULE_STATE_KEYS.keys()),
        default=list(_MODULE_STATE_KEYS.keys()),
        key="tmpl_modules",
    )

    if st.button("Generate Template", key="tmpl_save"):
        template = _capture_state(template_name, description, modules)

        if not template["settings"]:
            st.warning("No settings found for selected modules. Run some analyses first.")
            return

        json_str = json.dumps(template, indent=2, default=str)
        st.success(f"Template captured: {len(template['settings'])} settings from {len(modules)} module(s).")

        st.download_button(
            label="Download Template (JSON)",
            data=json_str,
            file_name=f"{template_name.lower().replace(' ', '_')}_template.json",
            mime="application/json",
            key="tmpl_download",
        )

        with st.expander("Preview template"):
            st.json(template)


def _render_load_template():
    section_header("Load Template")

    uploaded = st.file_uploader("Upload template JSON:", type=["json"], key="tmpl_upload")

    if uploaded is not None:
        try:
            template = json.loads(uploaded.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            st.error(f"Invalid template file: {e}")
            return

        st.markdown(f"**Name:** {template.get('name', 'Unknown')}")
        st.markdown(f"**Description:** {template.get('description', 'N/A')}")
        st.markdown(f"**Created:** {template.get('created', 'Unknown')}")
        st.markdown(f"**Settings:** {len(template.get('settings', {}))} parameters")

        with st.expander("Preview settings"):
            st.json(template.get("settings", {}))

        if st.button("Apply Template", key="tmpl_apply"):
            applied = _apply_state(template)
            if applied > 0:
                st.success(f"Applied {applied} settings. Rerun the relevant module to see changes.")
                st.rerun()
            else:
                st.warning("No settings could be applied.")


def _capture_state(name: str, description: str, modules: list) -> dict:
    """Capture relevant session state keys for the selected modules."""
    template = {
        "name": name,
        "description": description,
        "created": datetime.now().isoformat(),
        "version": "1.0",
        "settings": {},
    }

    prefixes = []
    for mod in modules:
        prefixes.extend(_MODULE_STATE_KEYS.get(mod, []))

    for key, value in st.session_state.items():
        if any(key.startswith(p) for p in prefixes):
            # Only save serializable types
            if isinstance(value, (str, int, float, bool, list)):
                template["settings"][key] = value
            elif isinstance(value, (np.integer,)):
                template["settings"][key] = int(value)
            elif isinstance(value, (np.floating,)):
                template["settings"][key] = float(value)

    return template


def _apply_state(template: dict) -> int:
    """Apply template settings to session state."""
    settings = template.get("settings", {})
    applied = 0
    for key, value in settings.items():
        st.session_state[key] = value
        applied += 1
    return applied
