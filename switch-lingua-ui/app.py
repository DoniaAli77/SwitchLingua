import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "Modified_Version" / "config"
OUTPUT_DIR = PROJECT_ROOT / "Modified_Version" / "output"
SAVED_CONFIG_DIR = BASE_DIR / "saved_configs"
SAVED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SMOKE_SCRIPT_PATH = PROJECT_ROOT / "Modified_Version" / "core" / "smoke_test_real_api.py"
FULL_SCRIPT_PATH = PROJECT_ROOT / "Modified_Version" / "core" / "run_french_ui.py"

st.set_page_config(
    page_title="Switch Lingua UI",
    page_icon="SL",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session state defaults ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # default to dark mode (light mode sidebar has render issues)
if "recent_runs" not in st.session_state:
    st.session_state.recent_runs = []  # list of {ts, kind, config, code}
if "last_run_outputs" not in st.session_state:
    st.session_state.last_run_outputs = []
if "preferred_output_name" not in st.session_state:
    st.session_state.preferred_output_name = None

_dark = st.session_state.dark_mode

if _dark:
    _bg      = "#1a1f2e"
    _bg2     = "#242938"
    _section = "#2b3044"
    _border  = "#3c4260"
    _text    = "#e8eaf6"
    _subtext = "#a0aec0"
    _input_bg   = "#1e2333"
    _input_text = "#e8eaf6"
    _input_border = "#3c4260"
    _sidebar_bg = "#161b27"
    _sidebar_text = "#c9d3e0"
    _sidebar_divider = "#2e3a50"
    _btn_bg = "#233149"
    _btn_text = "#e8eaf6"
    _tab_active = "#3aa6b9"
else:
    _bg      = "#f0f4f8"
    _bg2     = "#e8eef3"
    _section = "#ffffff"
    _border  = "#c8d8e4"
    _text    = "#1a2a3a"
    _subtext = "#4a6070"
    _input_bg   = "#ffffff"
    _input_text = "#1a2a3a"
    _input_border = "#a0bcc8"
    _sidebar_bg = "#dde8ef"
    _sidebar_text = "#1a2a3a"
    _sidebar_divider = "#b0c8d4"
    _btn_bg = "#e7eff5"
    _btn_text = "#12283a"
    _tab_active = "#0f4c5c"

st.markdown(
    f"""
    <style>
      /* ── Page background ── */
      .stApp {{
        background: linear-gradient(180deg, {_bg} 0%, {_bg2} 100%);
      }}
      .block-container {{
        padding-top: 1.4rem;
        padding-bottom: 2rem;
      }}

      /* ── Sidebar ── */
      section[data-testid="stSidebar"] {{
        background-color: {_sidebar_bg} !important;
        border-right: 1px solid {_sidebar_divider} !important;
      }}
      section[data-testid="stSidebar"] > div,
      section[data-testid="stSidebar"] > div:first-child,
      [data-testid="stSidebarContent"],
      [data-testid="stSidebarUserContent"] {{
        background-color: {_sidebar_bg} !important;
        border-right: 1px solid {_sidebar_divider};
      }}
      [data-testid="stSidebar"] *,
      [data-testid="stSidebarContent"] * {{
        color: {_sidebar_text} !important;
      }}
      /* Collapse button (inside sidebar) */
      [data-testid="stSidebarCollapseButton"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
      }}
      [data-testid="stSidebarCollapseButton"] button,
      [data-testid="stSidebarCollapseButton"] svg {{
        color: {_sidebar_text} !important;
        fill: {_sidebar_text} !important;
      }}
      /* Expand button (outside sidebar, shown when collapsed) — must be always visible */
      [data-testid="stSidebarCollapsedControl"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        position: fixed !important;
        top: 2.5rem !important;
        left: 0 !important;
        background-color: {_sidebar_bg} !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 0.4rem 0.3rem !important;
        box-shadow: 2px 0 6px rgba(0,0,0,0.35) !important;
      }}
      [data-testid="stSidebarCollapsedControl"] button,
      [data-testid="stSidebarCollapsedControl"] svg {{
        color: {_sidebar_text} !important;
        fill: {_sidebar_text} !important;
        visibility: visible !important;
        opacity: 1 !important;
      }}

      /* ── All text in main area ── */
            .stApp, .stApp p, .stApp label, .stApp span,
            .stApp li, .stApp h1, .stApp h2,
      .stApp h3, .stApp h4, .stApp small {{
        color: {_text};
      }}

      /* ── Streamlit native widgets: inputs, selects, textareas ── */
      .stTextInput input,
      .stNumberInput input,
      .stTextArea textarea,
      .stSelectbox select,
            [data-baseweb="select"] > div,
      [data-baseweb="input"] input,
      [data-baseweb="textarea"] textarea {{
        background-color: {_input_bg} !important;
        color: {_input_text} !important;
        border-color: {_input_border} !important;
      }}
      [data-baseweb="select"] [data-testid="stMarkdownContainer"],
      [data-baseweb="select"] span {{
        color: {_input_text} !important;
      }}

            /* Keep multiselect selected chips readable and consistent */
            [data-baseweb="tag"] {{
                background-color: #ff4b4b !important;
                color: #ffffff !important;
                border-radius: 10px !important;
            }}
            [data-baseweb="tag"] * {{
                color: #ffffff !important;
            }}

      /* ── Metric labels ── */
      [data-testid="stMetricLabel"] p,
      [data-testid="stMetricValue"] {{
        color: {_text} !important;
      }}

      /* ── Caption / markdown text ── */
      .stCaption, [data-testid="stCaptionContainer"] p {{
        color: {_subtext} !important;
      }}

      /* ── Hero banner ── */
      .sl-hero {{
        background: linear-gradient(120deg, #0f4c5c 0%, #1f7a8c 60%, #3aa6b9 100%);
        color: #ffffff;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
      }}
      .sl-hero h2 {{ margin: 0; font-size: 1.45rem; color: #fff !important; }}
      .sl-hero p  {{ margin: 0.3rem 0 0 0; opacity: 0.95; color: #fff !important; }}

      /* ── Chips ── */
      .sl-chip {{
        display: inline-block;
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.35);
        color: #fff;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        margin-right: 0.35rem;
        font-size: 0.8rem;
      }}

      /* ── Section cards ── */
      .sl-section {{
        background: {_section};
        border: 1px solid {_border};
        border-radius: 14px;
        padding: 0.9rem 1rem 0.8rem 1rem;
        margin-bottom: 0.9rem;
      }}
      .sl-section h3, .sl-section p {{
        color: {_text} !important;
        margin-top: 0.1rem;
      }}

      /* ── Tabs ── */
      .stTabs [data-baseweb="tab-list"] {{
        background: transparent;
      }}
      .stTabs [aria-selected="true"] {{
        color: {_tab_active} !important;
        border-bottom-color: {_tab_active} !important;
      }}

      /* ── Buttons ── */
      .stButton > button {{
        border-radius: 10px;
        border: 1px solid {_input_border};
                background-color: {_btn_bg};
                color: {_btn_text} !important;
            }}
            .stButton > button:hover {{
                filter: brightness(0.98);
            }}
            .stButton > button[kind="primary"] {{
                background-color: #ff4b4b !important;
                border-color: #ff4b4b !important;
                color: #ffffff !important;
            }}
            .stButton > button[kind="secondary"] {{
                background-color: {_btn_bg} !important;
                border-color: {_input_border} !important;
                color: {_btn_text} !important;
      }}

      /* ── Run badges ── */
      .run-badge-ok  {{ color: #22c55e; font-weight: 600; }}
      .run-badge-err {{ color: #ef4444; font-weight: 600; }}

            /* Hide clutter */
            [data-testid="stToolbar"],
            #MainMenu {{
                display: none !important;
            }}
            [data-testid="stHeader"],
            header {{
                background: transparent !important;
                border-bottom: none !important;
            }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Inject JS directly into the page DOM (not iframe) to clear Streamlit's sidebar localStorage key
st.markdown(
    """
    <script>
    (function() {
        try {
            for (var i = localStorage.length - 1; i >= 0; i--) {
                var k = localStorage.key(i);
                if (k && k.toLowerCase().includes('sidebar')) {
                    localStorage.removeItem(k);
                }
            }
        } catch(e) {}
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='sl-hero'>
      <h2>Switch Lingua Control Center</h2>
      <p>Edit configs, run smoke/full pipeline, and inspect outputs in one place.</p>
      <div style='margin-top:0.5rem;'>
        <span class='sl-chip'>Config Editor</span>
        <span class='sl-chip'>Smoke + Full Run</span>
        <span class='sl-chip'>Output Explorer</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def run_smoke_test(config_path: Path, max_scenarios: int = 2) -> tuple[int, str]:
    """Run smoke test with selected config and return (exit_code, combined_output)."""
    env = os.environ.copy()
    env["SWITCHLINGUA_CONFIG_PATH"] = str(config_path)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [
        sys.executable,
        str(SMOKE_SCRIPT_PATH),
        "--max-scenarios",
        str(max_scenarios),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 124, "Smoke test timed out after 60 minutes."


def run_full_pipeline(config_path: Path, max_scenarios: int = 0) -> tuple[int, str]:
    """Run full pipeline with selected config and return (exit_code, combined_output)."""
    env = os.environ.copy()
    env["SWITCHLINGUA_CONFIG_PATH"] = str(config_path)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [
        sys.executable,
        str(FULL_SCRIPT_PATH),
    ]
    if max_scenarios and max_scenarios > 0:
        cmd.extend(["--max-scenarios", str(max_scenarios)])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=14400,
        )
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 124, "Full pipeline run timed out after 4 hours."


def parse_jsonl_to_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (scenario_rows, sentence_rows) with full agent scores extracted."""
    scenario_rows: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                scenario_rows.append({"line": i, "error": "Invalid JSON"})
                continue

            # ── Agent results ──
            tv   = obj.get("task_validation_result") or {}
            flu  = obj.get("fluency_result") or {}
            nat  = obj.get("naturalness_result") or {}
            soc  = obj.get("social_cultural_result") or {}
            cs_list = obj.get("cs_ratio_results_per_instances") or []

            scen = {
                "line":                   i,
                "task":                   obj.get("task"),
                "label":                  obj.get("label"),
                "topic":                  obj.get("topic"),
                "tense":                  obj.get("tense"),
                "perspective":            obj.get("perspective"),
                "cs_ratio":               obj.get("cs_ratio"),
                "gender":                 obj.get("gender"),
                "age":                    obj.get("age"),
                "education_level":        obj.get("education_level"),
                "conversation_type":      obj.get("conversation_type"),
                # Agent scores
                "overall_score":          obj.get("score"),
                "refine_count":           obj.get("refine_count"),
                "fluency_score":          flu.get("fluency_score"),
                "fluency_errors":         len(flu.get("errors") or []),
                "fluency_summary":        flu.get("summary"),
                "naturalness_score":      nat.get("naturalness_score"),
                "naturalness_summary":    nat.get("summary"),
                "social_cultural_score":  soc.get("socio_cultural_score"),
                "social_cultural_summary": soc.get("summary"),
                "task_val_passed":        tv.get("passed"),
                "task_val_confidence":    tv.get("confidence"),
                "task_val_notes":         tv.get("notes"),
                "task_val_predicted":     tv.get("predicted_label"),
                "sentence_count":         len(obj.get("data_generation_result") or []),
            }
            scenario_rows.append(scen)

            # ── Per-sentence rows ──
            sents = obj.get("data_generation_result") or []
            per_inst = tv.get("per_instance_results") or []
            for idx, s in enumerate(sents):
                inst = per_inst[idx] if idx < len(per_inst) else {}
                cs   = cs_list[idx]  if idx < len(cs_list)  else {}
                sentence_rows.append({
                    "line":                    i,
                    "sentence_index":          idx + 1,
                    "task":                    obj.get("task"),
                    "label":                   obj.get("label"),
                    "topic":                   obj.get("topic"),
                    "sentence":                s,
                    # Scenario-level scores (same for every sentence in this record)
                    "overall_score":           obj.get("score"),
                    "fluency_score":           flu.get("fluency_score"),
                    "fluency_errors":          len(flu.get("errors") or []),
                    "fluency_summary":         flu.get("summary"),
                    "naturalness_score":       nat.get("naturalness_score"),
                    "naturalness_observations": nat.get("observations"),
                    "naturalness_summary":     nat.get("summary"),
                    "social_cultural_score":   soc.get("socio_cultural_score"),
                    "social_cultural_issues":  "; ".join(str(x) for x in (soc.get("issues") or [])),
                    "social_cultural_summary": soc.get("summary"),
                    # Task-validation per-instance
                    "tv_passed":               inst.get("passed"),
                    "tv_confidence":           inst.get("confidence"),
                    "tv_predicted":            inst.get("predicted_label"),
                    "tv_notes":                inst.get("notes"),
                    # CS ratio per-instance
                    "cs_ratio_score":          cs.get("ratio_score"),
                    "cs_ratio_computed":       cs.get("computed_ratio"),
                    "cs_ratio_notes":          cs.get("notes"),
                    "refine_count":            obj.get("refine_count"),
                })

    return scenario_rows, sentence_rows


def parse_pipeline_json(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sentence_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []

    for run in data.get("runs", []):
        state = run.get("state", {})
        task_validation = state.get("task_validation_result") or {}

        scenario_rows.append(
            {
                "task": state.get("task", run.get("task")),
                "label": state.get("label"),
                "topic": state.get("topic"),
                "overall_score": state.get("score"),
                "num_sentences": len(state.get("data_generation_result") or []),
                "task_passed": task_validation.get("passed"),
            }
        )

        sentences = state.get("data_generation_result") or []
        flu = state.get("fluency_results_per_instances") or []
        nat = state.get("naturalness_results_per_instances") or []
        csr = state.get("cs_ratio_results_per_instances") or []
        soc = state.get("social_cultural_results_per_instances") or []

        for i, sent in enumerate(sentences):
            flu_i = flu[i] if i < len(flu) and isinstance(flu[i], dict) else {}
            nat_i = nat[i] if i < len(nat) and isinstance(nat[i], dict) else {}
            csr_i = csr[i] if i < len(csr) and isinstance(csr[i], dict) else {}
            soc_i = soc[i] if i < len(soc) and isinstance(soc[i], dict) else {}

            sentence_rows.append(
                {
                    "task": state.get("task", run.get("task")),
                    "label": state.get("label"),
                    "topic": state.get("topic"),
                    "overall_score": state.get("score"),
                    "sentence_index": i,
                    "sentence": sent,
                    "fluency_score": flu_i.get("fluency_score"),
                    "naturalness_score": nat_i.get("naturalness_score"),
                    "cs_ratio_score": csr_i.get("ratio_score"),
                    "cs_ratio_computed": csr_i.get("computed_ratio"),
                    "socio_score": soc_i.get("socio_cultural_score"),
                }
            )

    return pd.DataFrame(sentence_rows), pd.DataFrame(scenario_rows)


def drop_embedded_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that look like repeated header values inside the data."""
    if df.empty:
        return df

    col_names = [str(c).strip().lower() for c in df.columns]
    keep_mask: list[bool] = []

    for _, row in df.iterrows():
        match_count = 0
        for col, col_name in zip(df.columns, col_names):
            val = row.get(col)
            # pd.notna returns an array for list/dict values — guard with a scalar check
            try:
                is_na = pd.isna(val)
                if isinstance(is_na, (list, tuple)) or hasattr(is_na, "__len__") and not isinstance(is_na, (str, bytes)):
                    continue  # non-scalar cell, can't be a header repeat
                if not is_na and str(val).strip().lower() == col_name:
                    match_count += 1
            except (TypeError, ValueError):
                continue

        # If many cells mirror column names, this is likely a duplicated header row.
        is_header_like = match_count >= max(3, len(col_names) // 2)
        keep_mask.append(not is_header_like)

    return df.loc[keep_mask].reset_index(drop=True)


def list_configs() -> list[Path]:
    project_configs = sorted(CONFIG_DIR.glob("*.yaml"))
    saved_configs = sorted(SAVED_CONFIG_DIR.glob("*.yaml"))
    return project_configs + saved_configs


def list_outputs() -> list[Path]:
    files = []
    files.extend(sorted(OUTPUT_DIR.glob("*.xlsx")))
    files.extend(sorted(OUTPUT_DIR.glob("*.json")))
    files.extend(sorted(OUTPUT_DIR.glob("*.jsonl")))
    return files


def snapshot_output_mtimes() -> dict[str, float]:
    return {str(p): p.stat().st_mtime for p in list_outputs() if p.exists()}


def detect_touched_outputs(before: dict[str, float]) -> list[Path]:
    touched: list[Path] = []
    for p in list_outputs():
        key = str(p)
        current_mtime = p.stat().st_mtime
        previous_mtime = before.get(key)
        if previous_mtime is None or current_mtime > previous_mtime + 1e-9:
            touched.append(p)
    touched.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return touched


def infer_expected_output_file(config_path: Path) -> Path | None:
    """Infer expected jsonl output file from config first language."""
    try:
        cfg = load_yaml(config_path)
        pre = cfg.get("pre_execute") or {}

        first_language = pre.get("first_language")
        if not first_language:
            shared = pre.get("shared") or {}
            char = shared.get("character_setting") or {}
            nationality = char.get("nationality") or {}
            first_language = nationality.get("first_language")

        if not first_language:
            return None

        expected = OUTPUT_DIR / f"{str(first_language).strip()}.jsonl"
        return expected if expected.exists() else None
    except Exception:
        return None


_ALL_TOPICS = ["tech", "finance", "business", "education", "health", "shopping", "medical", "sports", "social"]
_SENTIMENT_LABELS = ["positive", "negative", "neutral"]
_NER_TAGS = ["PER", "ORG", "LOC", "GPE", "MISC", "DATE", "TIME", "MONEY", "PERCENT", "PRODUCT", "EVENT"]
_TENSES = ["Present", "Past", "Future"]
_PERSPECTIVES = ["First Person", "Second Person", "Third Person"]
_CS_FUNCTIONS = ["Expressive", "Referential", "Directive", "Phatic", "Metalinguistic"]
_CS_TYPES = ["Intrasentential", "Intersentential", "Tag-switching"]
_CONV_TYPES = ["single_turn", "multi_turn"]
_AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "56+"]
_GENDERS = ["Male", "Female", "Non-binary"]
_EDUCATION_LEVELS = ["High School", "College", "Graduate", "PhD"]
_OUTPUT_FORMATS = ["json", "jsonl", "text"]


def _get_topics(pre: dict) -> list[str]:
    """Read topics from either config format."""
    # config.yaml format: pre_execute.topics
    if isinstance(pre.get("topics"), list):
        return pre["topics"]
    # config2.yaml format: pre_execute.shared.topic
    shared = pre.get("shared") or {}
    if isinstance(shared.get("topic"), list):
        return shared["topic"]
    return []


def _set_topics(pre: dict, topics: list[str]) -> None:
    """Write topics back to whichever location was present."""
    if "topics" in pre:
        pre["topics"] = topics
    elif isinstance(pre.get("shared"), dict):
        pre["shared"]["topic"] = topics
    else:
        pre["topics"] = topics


def _get_shared(pre: dict, key: str, default):
    """Read a field from flat pre_execute or pre_execute.shared."""
    if key in pre:
        v = pre[key]
        return v if isinstance(v, list) else [v] if v else default
    shared = pre.get("shared") or {}
    if key in shared:
        v = shared[key]
        return v if isinstance(v, list) else [v] if v else default
    return default


def _set_shared(pre: dict, key: str, value) -> None:
    """Write a field back to the location it was read from."""
    if key in pre:
        pre[key] = value
    elif isinstance(pre.get("shared"), dict):
        pre["shared"][key] = value
    else:
        pre[key] = value


def _get_char(pre: dict, key: str, default):
    """Read character_setting fields from flat or shared location."""
    char = pre.get("character_setting") or (pre.get("shared") or {}).get("character_setting") or {}
    v = char.get(key, default)
    return v if isinstance(v, list) else [v] if v else default


def _set_char(pre: dict, key: str, value) -> None:
    """Write character_setting field back."""
    if "character_setting" in pre:
        pre["character_setting"][key] = value
    elif isinstance(pre.get("shared"), dict):
        pre["shared"].setdefault("character_setting", {})[key] = value
    else:
        pre.setdefault("character_setting", {})[key] = value


def editable_pre_execute_form(config_data: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config_data)
    pre = cfg.get("pre_execute") or {}
    cfg["pre_execute"] = pre

    st.subheader("Pre Execute")

    tasks = pre.get("task", [])
    if not isinstance(tasks, list):
        tasks = [tasks] if tasks else []

    task_options = ["topic", "sentiment", "ner"]
    chosen_tasks = st.multiselect(
        "Tasks",
        options=task_options,
        default=[t for t in tasks if t in task_options],
    )
    pre["task"] = chosen_tasks

    if "sentiment" in chosen_tasks:
        sentiment_cfg = pre.get("sentiment") if isinstance(pre.get("sentiment"), dict) else {}
        current_labels = sentiment_cfg.get("labels", _SENTIMENT_LABELS)
        if not isinstance(current_labels, list):
            current_labels = [str(current_labels)] if current_labels else []
        extra_sentiment = [x for x in current_labels if x not in _SENTIMENT_LABELS]
        sentiment_options = _SENTIMENT_LABELS + extra_sentiment
        selected_labels = st.multiselect(
            "Sentiment Labels",
            options=sentiment_options,
            default=[x for x in current_labels if x in sentiment_options],
        )
        custom_sentiment = st.text_input("Add custom sentiment labels (comma separated)", value=", ".join(extra_sentiment) if extra_sentiment else "")
        if custom_sentiment.strip():
            selected_labels = list(dict.fromkeys(selected_labels + [x.strip() for x in custom_sentiment.split(",") if x.strip()]))
        sentiment_cfg["labels"] = selected_labels

        col_s1, col_s2 = st.columns(2)
        _INTENSITIES = ["low", "medium", "high"]
        _AMBIGUITIES = ["low", "medium", "high"]
        with col_s1:
            cur_int = sentiment_cfg.get("intensity", ["low", "medium"])
            if not isinstance(cur_int, list):
                cur_int = [cur_int] if cur_int else []
            sentiment_cfg["intensity"] = st.multiselect(
                "Intensity", options=_INTENSITIES,
                default=[x for x in cur_int if x in _INTENSITIES] or cur_int[:1],
                help="Intensity levels to include in sentiment generation.",
            )
        with col_s2:
            cur_amb = sentiment_cfg.get("ambiguity", ["low"])
            if not isinstance(cur_amb, list):
                cur_amb = [cur_amb] if cur_amb else []
            sentiment_cfg["ambiguity"] = st.multiselect(
                "Ambiguity", options=_AMBIGUITIES,
                default=[x for x in cur_amb if x in _AMBIGUITIES] or cur_amb[:1],
                help="Ambiguity levels for sentiment examples.",
            )

        pre["sentiment"] = sentiment_cfg

    if "ner" in chosen_tasks:
        ner_cfg = pre.get("ner") if isinstance(pre.get("ner"), dict) else {}
        current_entities = ner_cfg.get("entity_types", _NER_TAGS[:3])
        if not isinstance(current_entities, list):
            current_entities = [str(current_entities)] if current_entities else []
        extra_ner = [x for x in current_entities if x not in _NER_TAGS]
        ner_options = _NER_TAGS + extra_ner
        selected_entities = st.multiselect(
            "NER Entity Types",
            options=ner_options,
            default=[x for x in current_entities if x in ner_options],
        )
        custom_ner = st.text_input("Add custom NER tags (comma separated)", value=", ".join(extra_ner) if extra_ner else "")
        if custom_ner.strip():
            selected_entities = list(dict.fromkeys(selected_entities + [x.strip() for x in custom_ner.split(",") if x.strip()]))
        ner_cfg["entity_types"] = selected_entities

        # must_include_types — subset of entity_types
        cur_must = ner_cfg.get("must_include_types", [])
        if not isinstance(cur_must, list):
            cur_must = [cur_must] if cur_must else []
        must_options = selected_entities if selected_entities else ner_options
        ner_cfg["must_include_types"] = st.multiselect(
            "Must Include Types",
            options=must_options,
            default=[x for x in cur_must if x in must_options],
            help="Entity types that must appear in every generated sentence.",
        )

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            min_e_raw = ner_cfg.get("min_entities", [1])
            min_e = int(min_e_raw[0]) if isinstance(min_e_raw, list) else int(min_e_raw or 1)
            min_e_val = st.number_input("Min Entities", min_value=0, max_value=20, value=min_e, step=1)
            ner_cfg["min_entities"] = [min_e_val]
        with col_n2:
            max_e_raw = ner_cfg.get("max_entities", [3])
            max_e = int(max_e_raw[0]) if isinstance(max_e_raw, list) else int(max_e_raw or 3)
            max_e_val = st.number_input("Max Entities", min_value=1, max_value=20, value=max(max_e, min_e_val), step=1)
            ner_cfg["max_entities"] = [max_e_val]

        cs_ent_raw = ner_cfg.get("allow_code_switched_entities", [True])
        cs_ent = bool(cs_ent_raw[0]) if isinstance(cs_ent_raw, list) else bool(cs_ent_raw)
        ner_cfg["allow_code_switched_entities"] = [st.checkbox("Allow Code-Switched Entities", value=cs_ent)]

        pre["ner"] = ner_cfg

    # --- Topics ---
    current_topics = _get_topics(pre)
    extra_t = [t for t in current_topics if t not in _ALL_TOPICS]
    chosen_topics = st.multiselect("Topics", options=_ALL_TOPICS + extra_t, default=[t for t in current_topics if t in _ALL_TOPICS + extra_t])
    extra_topics_input = st.text_input("Add custom topics (comma separated)", value=", ".join(extra_t) if extra_t else "")
    if extra_topics_input.strip():
        chosen_topics = list(dict.fromkeys(chosen_topics + [t.strip() for t in extra_topics_input.split(",") if t.strip()]))
    _set_topics(pre, chosen_topics)

    st.divider()
    st.subheader("Language & Generation")

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        # first_language: flat or inside character_setting.nationality
        char = pre.get("character_setting") or (pre.get("shared") or {}).get("character_setting") or {}
        nat = char.get("nationality") or {}
        fl_default = pre.get("first_language") or nat.get("first_language") or "Arabic"
        fl_val = st.text_input("First Language", value=str(fl_default))
    with col_l2:
        sl_default = pre.get("second_language") or nat.get("second_language") or "English"
        sl_val = st.text_input("Second Language", value=str(sl_default))

    # Write language back to wherever it was
    if "first_language" in pre:
        pre["first_language"] = fl_val
    elif "character_setting" in pre:
        pre["character_setting"].setdefault("nationality", {})["first_language"] = fl_val
    elif isinstance(pre.get("shared"), dict):
        pre["shared"].setdefault("character_setting", {}).setdefault("nationality", {})["first_language"] = fl_val
    else:
        pre["first_language"] = fl_val

    if "second_language" in pre:
        pre["second_language"] = sl_val
    elif "character_setting" in pre:
        pre["character_setting"].setdefault("nationality", {})["second_language"] = sl_val
    elif isinstance(pre.get("shared"), dict):
        pre["shared"].setdefault("character_setting", {}).setdefault("nationality", {})["second_language"] = sl_val
    else:
        pre["second_language"] = sl_val

    cs_ratio_val = pre.get("cs_ratio", ["70%"])
    cs_ratio_text = st.text_input(
        "CS Ratio (comma separated, e.g. 70%, 50%)",
        value=", ".join(cs_ratio_val) if isinstance(cs_ratio_val, list) else str(cs_ratio_val),
    )
    pre["cs_ratio"] = [x.strip() for x in cs_ratio_text.split(",") if x.strip()]

    max_scenarios = int(pre.get("max_scenarios_per_task", 1) or 1)
    pre["max_scenarios_per_task"] = st.number_input("Max Scenarios Per Task", min_value=1, max_value=100, value=max_scenarios, step=1)

    pre["use_tools"] = st.checkbox("Use Tools", value=bool(pre.get("use_tools", False)))

    st.divider()
    st.subheader("Style & Character")

    col_a, col_b = st.columns(2)
    with col_a:
        cur_tense = _get_shared(pre, "tense", ["Present"])
        chosen_tense = st.multiselect("Tense", options=_TENSES, default=[t for t in cur_tense if t in _TENSES] or cur_tense[:1])
        _set_shared(pre, "tense", chosen_tense)

        cur_persp = _get_shared(pre, "perspective", ["First Person"])
        chosen_persp = st.multiselect("Perspective", options=_PERSPECTIVES, default=[p for p in cur_persp if p in _PERSPECTIVES] or cur_persp[:1])
        _set_shared(pre, "perspective", chosen_persp)

        cur_conv = _get_shared(pre, "conversation_type", ["single_turn"])
        chosen_conv = st.multiselect("Conversation Type", options=_CONV_TYPES, default=[c for c in cur_conv if c in _CONV_TYPES] or cur_conv[:1])
        _set_shared(pre, "conversation_type", chosen_conv)

    with col_b:
        cur_csf = _get_shared(pre, "cs_function", ["Expressive"])
        chosen_csf = st.multiselect("CS Function", options=_CS_FUNCTIONS, default=[f for f in cur_csf if f in _CS_FUNCTIONS] or cur_csf[:1])
        _set_shared(pre, "cs_function", chosen_csf)

        cur_cst = _get_shared(pre, "cs_type", ["Intrasentential"])
        chosen_cst = st.multiselect("CS Type", options=_CS_TYPES, default=[t for t in cur_cst if t in _CS_TYPES] or cur_cst[:1])
        _set_shared(pre, "cs_type", chosen_cst)

        cur_fmt = _get_shared(pre, "output_format", ["json"])
        fmt_default = cur_fmt[0] if cur_fmt else "json"
        chosen_fmt = st.selectbox("Output Format", options=_OUTPUT_FORMATS, index=_OUTPUT_FORMATS.index(fmt_default) if fmt_default in _OUTPUT_FORMATS else 0)
        _set_shared(pre, "output_format", chosen_fmt)

    st.markdown("**Character Settings**")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        cur_age = _get_char(pre, "age", ["18-25"])
        chosen_age = st.multiselect("Age", options=_AGE_GROUPS, default=[a for a in cur_age if a in _AGE_GROUPS] or cur_age[:1])
        _set_char(pre, "age", chosen_age)
    with col_c2:
        cur_gender = _get_char(pre, "gender", ["Male"])
        chosen_gender = st.multiselect("Gender", options=_GENDERS, default=[g for g in cur_gender if g in _GENDERS] or cur_gender[:1])
        _set_char(pre, "gender", chosen_gender)
    with col_c3:
        cur_edu = _get_char(pre, "education_level", ["College"])
        chosen_edu = st.multiselect("Education Level", options=_EDUCATION_LEVELS, default=[e for e in cur_edu if e in _EDUCATION_LEVELS] or cur_edu[:1])
        _set_char(pre, "education_level", chosen_edu)

    # --- on_execute ---
    on_ex = cfg.get("on_execute") or {}
    st.divider()
    st.subheader("On Execute")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        on_ex["round"] = st.number_input("Round", min_value=1, max_value=50, value=int(on_ex.get("round", 1)), step=1)
    with col_e2:
        on_ex["verbose"] = st.checkbox("Verbose", value=bool(on_ex.get("verbose", True)))
    cfg["on_execute"] = on_ex

    st.divider()
    st.subheader("Raw YAML Preview")
    edited_yaml_text = st.text_area(
        "You can directly edit YAML here",
        value=yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        height=300,
    )

    try:
        cfg = yaml.safe_load(edited_yaml_text) or {}
    except yaml.YAMLError as e:
        st.error(f"YAML parse error: {e}")

    return cfg


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def show_run_result(title: str, code: int, output: str, output_limit: int) -> None:
    if code == 0:
        st.success(f"{title} finished successfully.")
    elif code == 124:
        st.warning(output)
    else:
        st.error(f"{title} failed with exit code {code}.")

    if output:
        with st.expander(f"{title} logs", expanded=True):
            st.code(output[-output_limit:] if len(output) > output_limit else output, language="text")


with st.sidebar:
    # --- Theme toggle ---
    _label = "Switch to Light Mode" if _dark else "Switch to Dark Mode"
    if st.button(_label, width="stretch"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.divider()
    st.markdown("### Workspace")
    st.caption(relative_to_root(PROJECT_ROOT))
    cfg_count = len(list(CONFIG_DIR.glob("*.yaml"))) + len(list(SAVED_CONFIG_DIR.glob("*.yaml")))
    out_count = len(list_outputs())
    st.metric("Config Files", cfg_count)
    st.metric("Output Files", out_count)
    st.markdown("### Runners")
    st.caption(f"Smoke: {relative_to_root(SMOKE_SCRIPT_PATH)}")
    st.caption(f"Full: {relative_to_root(FULL_SCRIPT_PATH)}")

    # --- Recent runs history ---
    st.divider()
    st.markdown("### Recent Runs")
    if not st.session_state.recent_runs:
        st.caption("No runs yet this session.")
    else:
        for entry in reversed(st.session_state.recent_runs[-10:]):
            badge = "ok" if entry["code"] == 0 else "err"
            status = "✓" if entry["code"] == 0 else "✗"
            st.markdown(
                f"<span class='run-badge-{badge}'>{status}</span> "
                f"**{entry['kind']}** · `{entry['config']}`  \n"
                f"<small>{entry['ts']}</small>",
                unsafe_allow_html=True,
            )
        if st.button("Clear history", key="clear_history"):
            st.session_state.recent_runs = []
            st.rerun()


config_tab, output_tab = st.tabs(["Configuration", "Output Viewer"])

with config_tab:
    st.markdown("<div class='sl-section'><h3>Configuration Studio</h3><p>Choose a config, tweak it, save a UI copy, then run.</p></div>", unsafe_allow_html=True)

    configs = list_configs()
    if not configs:
        st.warning(f"No .yaml configs found in {CONFIG_DIR} or {SAVED_CONFIG_DIR}")
    else:
        cfg_map = {f"{p.name} ({'project' if p.parent == CONFIG_DIR else 'ui'})": p for p in configs}

        top_col1, top_col2 = st.columns([1.4, 1])
        with top_col1:
            selected_cfg_label = st.selectbox("Config File", options=list(cfg_map.keys()))
            selected_cfg_path = cfg_map[selected_cfg_label]
            st.caption(f"Current: {relative_to_root(selected_cfg_path)}")

        cfg_data = load_yaml(selected_cfg_path)
        edited_cfg = editable_pre_execute_form(cfg_data)

        with top_col2:
            st.markdown("#### Save & Commands")
            save_name = st.text_input("Save As", value=f"ui_{selected_cfg_path.name}")
            if st.button("Save To UI Folder", type="primary", width="stretch"):
                out_name = save_name.strip() or f"ui_{selected_cfg_path.name}"
                if not out_name.endswith(".yaml"):
                    out_name = f"{out_name}.yaml"
                out_path = SAVED_CONFIG_DIR / out_name
                save_yaml(out_path, edited_cfg)
                st.success(f"Saved: {relative_to_root(out_path)}")

            with st.expander("Show CLI commands"):
                st.code(
                    (
                        f"set SWITCHLINGUA_CONFIG_PATH={selected_cfg_path}\n"
                        f"python {relative_to_root(SMOKE_SCRIPT_PATH)} --max-scenarios 2\n"
                        f"python {relative_to_root(FULL_SCRIPT_PATH)} --max-scenarios 0"
                    ),
                    language="bash",
                )

        st.markdown("<div class='sl-section'><h3>Execution</h3><p>Run quick smoke tests or full generation directly from UI.</p></div>", unsafe_allow_html=True)

        run_col1, run_col2 = st.columns(2)

        with run_col1:
            st.markdown("#### Smoke Test")
            run_scenarios = st.number_input(
                "Max scenarios",
                min_value=1,
                max_value=20,
                value=2,
                step=1,
                key="smoke_run_max_scenarios",
            )
            st.caption(f"Runner: {relative_to_root(SMOKE_SCRIPT_PATH)}")
            run_clicked = st.button("Run Smoke Test", type="secondary", width="stretch")

            if run_clicked:
                before_outputs = snapshot_output_mtimes()
                with st.spinner("Running smoke test..."):
                    code, output = run_smoke_test(selected_cfg_path, int(run_scenarios))
                touched_outputs = detect_touched_outputs(before_outputs)
                st.session_state.last_run_outputs = [p.name for p in touched_outputs]
                st.session_state.preferred_output_name = touched_outputs[0].name if touched_outputs else None
                st.session_state.recent_runs.append({
                    "ts": datetime.now().strftime("%H:%M:%S"),
                    "kind": "Smoke",
                    "config": selected_cfg_path.name,
                    "code": code,
                })
                show_run_result("Smoke test", code, output, 12000)
                if touched_outputs:
                    st.info("Updated output file(s): " + ", ".join(p.name for p in touched_outputs[:3]))
                else:
                    expected = infer_expected_output_file(selected_cfg_path)
                    if expected is not None:
                        st.session_state.preferred_output_name = expected.name
                        st.session_state.last_run_outputs = [expected.name]
                        st.info(f"No timestamp change detected. Expected output file: {expected.name}")
                    else:
                        st.info("Run finished, but no output file timestamp changed in output/.")

        with run_col2:
            st.markdown("#### Full Pipeline")
            full_scenarios = st.number_input(
                "Max scenarios (0 = all)",
                min_value=0,
                max_value=1000,
                value=0,
                step=1,
                key="full_run_max_scenarios",
            )
            st.caption(f"Runner: {relative_to_root(FULL_SCRIPT_PATH)}")
            full_run_clicked = st.button("Run Full Pipeline", type="secondary", width="stretch")

            if full_run_clicked:
                before_outputs = snapshot_output_mtimes()
                with st.spinner("Running full pipeline... this may take longer."):
                    code, output = run_full_pipeline(selected_cfg_path, int(full_scenarios))
                touched_outputs = detect_touched_outputs(before_outputs)
                st.session_state.last_run_outputs = [p.name for p in touched_outputs]
                st.session_state.preferred_output_name = touched_outputs[0].name if touched_outputs else None
                st.session_state.recent_runs.append({
                    "ts": datetime.now().strftime("%H:%M:%S"),
                    "kind": "Full Pipeline",
                    "config": selected_cfg_path.name,
                    "code": code,
                })
                show_run_result("Full pipeline", code, output, 20000)
                if touched_outputs:
                    st.info("Updated output file(s): " + ", ".join(p.name for p in touched_outputs[:3]))
                else:
                    expected = infer_expected_output_file(selected_cfg_path)
                    if expected is not None:
                        st.session_state.preferred_output_name = expected.name
                        st.session_state.last_run_outputs = [expected.name]
                        st.info(f"No timestamp change detected. Expected output file: {expected.name}")
                    else:
                        st.info("Run finished, but no output file timestamp changed in output/.")

with output_tab:
    st.markdown("<div class='sl-section'><h3>Output Explorer</h3><p>Inspect JSON/JSONL/XLSX results with quick filters and score views.</p></div>", unsafe_allow_html=True)

    outputs = list_outputs()
    if not outputs:
        st.warning(f"No output files found in {OUTPUT_DIR}")
    else:
        out_map = {p.name: p for p in outputs}
        output_names = list(out_map.keys())
        preferred_name = st.session_state.get("preferred_output_name")
        default_index = output_names.index(preferred_name) if preferred_name in output_names else 0
        selected_name = st.selectbox("Output File", options=output_names, index=default_index)
        selected_output = out_map[selected_name]

        if st.session_state.last_run_outputs:
            visible = [n for n in st.session_state.last_run_outputs if n in out_map]
            if visible:
                st.caption("From last run: " + ", ".join(visible[:3]))

        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.metric("File Type", selected_output.suffix.lower())
        with meta_col2:
            st.metric("Folder", "output")
        with meta_col3:
            st.metric("Size (KB)", round(selected_output.stat().st_size / 1024, 1))

        st.caption(f"Selected: {relative_to_root(selected_output)}")

        if selected_output.suffix.lower() == ".xlsx":
            xls = pd.ExcelFile(selected_output)
            sheet = st.selectbox("Sheet", options=xls.sheet_names)
            df = pd.read_excel(xls, sheet_name=sheet)
            df = drop_embedded_header_rows(df)
            st.dataframe(df, width="stretch", height=460)

            if "overall_score" in df.columns:
                scores = pd.to_numeric(df["overall_score"], errors="coerce").dropna()
                if not scores.empty:
                    st.markdown("#### Overall Score Distribution")
                    st.bar_chart(scores)

        elif selected_output.suffix.lower() == ".json":
            sent_df, scen_df = parse_pipeline_json(selected_output)
            sent_df = drop_embedded_header_rows(sent_df)
            scen_df = drop_embedded_header_rows(scen_df)

            if not scen_df.empty and "overall_score" in scen_df.columns:
                s = pd.to_numeric(scen_df["overall_score"], errors="coerce").dropna()
                m1, m2, m3 = st.columns(3)
                m1.metric("Scenarios", len(scen_df))
                m2.metric("Sentences", len(sent_df))
                m3.metric("Avg Overall Score", round(float(s.mean()), 3) if len(s) else "N/A")

            st.markdown("#### Per Scenario")
            st.dataframe(scen_df, width="stretch", height=220)

            if "overall_score" in scen_df.columns:
                score_series = pd.to_numeric(scen_df["overall_score"], errors="coerce").dropna()
                if not score_series.empty:
                    st.bar_chart(score_series)

            st.markdown("#### Per Sentence")
            filter_col1, filter_col2 = st.columns([1, 1])
            with filter_col1:
                task_filter = st.multiselect(
                    "Filter by task",
                    options=sorted([x for x in sent_df["task"].dropna().unique().tolist()]) if "task" in sent_df.columns else [],
                )
            with filter_col2:
                min_score = st.slider("Min overall score", 0.0, 10.0, 0.0, 0.1)

            view_df = sent_df.copy()
            if task_filter and "task" in sent_df.columns:
                view_df = view_df[view_df["task"].isin(task_filter)]
            if "overall_score" in view_df.columns:
                numeric_score = pd.to_numeric(view_df["overall_score"], errors="coerce")
                view_df = view_df[numeric_score >= min_score]
            view_df = drop_embedded_header_rows(view_df)
            st.dataframe(view_df, width="stretch", height=420)

            # ── Export (JSON files) ──
            import io as _io
            st.markdown("**Export**")
            _ec1, _ec2, _ec3 = st.columns(3)
            with _ec1:
                st.download_button("Download Sentences CSV", data=sent_df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=selected_output.stem + "_sentences.csv", mime="text/csv", width="stretch")
            with _ec2:
                _xbuf = _io.BytesIO()
                with pd.ExcelWriter(_xbuf, engine="openpyxl") as _w:
                    scen_df.to_excel(_w, index=False, sheet_name="Scenarios")
                    sent_df.to_excel(_w, index=False, sheet_name="Sentences")
                st.download_button("Download XLSX", data=_xbuf.getvalue(),
                                   file_name=selected_output.stem + ".xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
            with _ec3:
                st.download_button("Download Scenarios CSV", data=scen_df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=selected_output.stem + "_scenarios.csv", mime="text/csv", width="stretch")

        elif selected_output.suffix.lower() == ".jsonl":
            scen_rows, sent_rows = parse_jsonl_to_rows(selected_output)
            scen_df = drop_embedded_header_rows(pd.DataFrame(scen_rows))
            sent_df = drop_embedded_header_rows(pd.DataFrame(sent_rows))

            # ── Summary metrics ──
            if not scen_df.empty:
                _s_ov  = pd.to_numeric(scen_df.get("overall_score",          pd.Series(dtype=float)), errors="coerce").dropna()
                _s_flu = pd.to_numeric(scen_df.get("fluency_score",          pd.Series(dtype=float)), errors="coerce").dropna()
                _s_nat = pd.to_numeric(scen_df.get("naturalness_score",      pd.Series(dtype=float)), errors="coerce").dropna()
                _s_soc = pd.to_numeric(scen_df.get("social_cultural_score",  pd.Series(dtype=float)), errors="coerce").dropna()
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Scenarios",          len(scen_df))
                m2.metric("Avg Overall",        round(float(_s_ov.mean()),  2) if len(_s_ov)  else "N/A")
                m3.metric("Avg Fluency",        round(float(_s_flu.mean()), 2) if len(_s_flu) else "N/A")
                m4.metric("Avg Naturalness",    round(float(_s_nat.mean()), 2) if len(_s_nat) else "N/A")
                m5.metric("Avg Socio-Cultural", round(float(_s_soc.mean()), 2) if len(_s_soc) else "N/A")

            # ── Export ──
            import io as _io
            st.markdown("**Export**")
            exp_c1, exp_c2, exp_c3 = st.columns(3)
            with exp_c1:
                st.download_button("Download Scenarios CSV",
                    data=scen_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=selected_output.stem + "_scenarios.csv", mime="text/csv", width="stretch")
            with exp_c2:
                _xbuf = _io.BytesIO()
                with pd.ExcelWriter(_xbuf, engine="openpyxl") as _w:
                    scen_df.to_excel(_w, index=False, sheet_name="Scenarios")
                    sent_df.to_excel(_w, index=False, sheet_name="Sentences")
                    for _col, _sht in [("fluency_score","Fluency"),("naturalness_score","Naturalness"),("social_cultural_score","SocioCultural")]:
                        if _col in scen_df.columns:
                            _tmp = scen_df[["task","label","topic",_col]].copy()
                            _tmp[_col] = pd.to_numeric(_tmp[_col], errors="coerce")
                            _tmp.groupby("task")[_col].agg(["count","mean","min","max"]).reset_index().to_excel(_w, index=False, sheet_name=_sht)
                st.download_button("Download XLSX (all sheets)",
                    data=_xbuf.getvalue(),
                    file_name=selected_output.stem + ".xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
            with exp_c3:
                st.download_button("Download Sentences CSV",
                    data=sent_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=selected_output.stem + "_sentences.csv", mime="text/csv", width="stretch")

            # ── Score charts ──
            with st.expander("Score Charts", expanded=False):
                _chart_cols = st.columns(2)
                for _ci, (_col, _lbl) in enumerate([
                    ("overall_score","Overall Score"), ("fluency_score","Fluency"),
                    ("naturalness_score","Naturalness"), ("social_cultural_score","Socio-Cultural")
                ]):
                    if _col in scen_df.columns:
                        with _chart_cols[_ci % 2]:
                            _sv = pd.to_numeric(scen_df[_col], errors="coerce").dropna()
                            if not _sv.empty:
                                st.markdown(f"**{_lbl}**")
                                _hist = _sv.value_counts(bins=8, sort=False).sort_index().reset_index()
                                _hist.columns = ["range","count"]
                                _hist["range"] = _hist["range"].astype(str)
                                st.bar_chart(_hist.set_index("range")["count"])

            # ── Scenario table ──
            st.markdown("#### Scenario Summary")
            _scen_cols = [c for c in [
                "line","task","label","topic","overall_score",
                "fluency_score","naturalness_score","social_cultural_score",
                "task_val_passed","task_val_confidence","task_val_predicted",
                "refine_count","sentence_count","fluency_errors",
                "fluency_summary","naturalness_summary","social_cultural_summary","task_val_notes"
            ] if c in scen_df.columns]
            st.dataframe(scen_df[_scen_cols], width="stretch", height=280)

            # ── Per-sentence table ──
            st.markdown("#### Per-Sentence Details")
            _fc1, _fc2, _fc3 = st.columns(3)
            with _fc1:
                _task_f = st.multiselect("Filter task",
                    options=sorted(sent_df["task"].dropna().unique().tolist()) if "task" in sent_df.columns else [],
                    key="sent_task_filter")
            with _fc2:
                _label_f = st.multiselect("Filter label",
                    options=sorted(sent_df["label"].dropna().unique().tolist()) if "label" in sent_df.columns else [],
                    key="sent_label_filter")
            with _fc3:
                _tv_f = st.selectbox("Task validation", options=["All","Passed","Failed"], index=0, key="sent_tv_filter")

            _view = sent_df.copy()
            if _task_f and "task" in _view.columns:
                _view = _view[_view["task"].isin(_task_f)]
            if _label_f and "label" in _view.columns:
                _view = _view[_view["label"].isin(_label_f)]
            if _tv_f == "Passed" and "tv_passed" in _view.columns:
                _view = _view[_view["tv_passed"] == True]
            elif _tv_f == "Failed" and "tv_passed" in _view.columns:
                _view = _view[_view["tv_passed"] == False]
            # ordered columns: identifiers → sentence → all agent scores → notes
            _sent_cols = [c for c in [
                "line","sentence_index","task","label","topic","sentence",
                "overall_score",
                "fluency_score","fluency_errors","fluency_summary",
                "naturalness_score","naturalness_observations","naturalness_summary",
                "social_cultural_score","social_cultural_issues","social_cultural_summary",
                "tv_passed","tv_confidence","tv_predicted","tv_notes",
                "cs_ratio_score","cs_ratio_computed","cs_ratio_notes",
                "refine_count",
            ] if c in _view.columns]
            _final = drop_embedded_header_rows(_view[_sent_cols]).copy()
            # Flatten any list/dict cells to strings so PyArrow can serialize the frame
            for _c in _final.columns:
                if _final[_c].apply(lambda x: isinstance(x, (list, dict))).any():
                    _final[_c] = _final[_c].apply(
                        lambda x: ", ".join(str(i) for i in x) if isinstance(x, list)
                        else json.dumps(x, ensure_ascii=False) if isinstance(x, dict)
                        else x
                    )
            st.dataframe(_final, width="stretch", height=460)

        else:
            st.info("Unsupported file type.")

