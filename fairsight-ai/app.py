from dotenv import load_dotenv
load_dotenv()  # loads .env into os.environ
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.bias_detector import BiasDetector
from core.bias_flagger import BiasFlagger
from core.debiaser import Debiaser
from ai.gemini_service import GeminiService

st.set_page_config(
    page_title="FairSight AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .stApp { background-color: #f8fafc; }
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 0.75rem;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0 4px;
    }
    .badge-blue { background: #dbeafe; color: #1d4ed8; }
    .badge-green { background: #dcfce7; color: #16a34a; }
    .badge-gray { background: #f1f5f9; color: #475569; }
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-top: 4px solid #2563eb;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .metric-card .value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
    }
    .score-display {
        text-align: center;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
    }
    .score-display .score-number {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-display .score-label {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    .score-display .score-message {
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    .verdict-box {
        text-align: center;
        padding: 1.25rem;
        border-radius: 12px;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .verdict-reason {
        font-size: 0.85rem;
        font-weight: 400;
        margin-top: 0.25rem;
    }
    .info-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 4px solid #2563eb;
        margin: 1rem 0;
    }
    .info-card-warning {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 4px solid #ea580c;
        margin: 1rem 0;
    }
    .explanation-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 5px solid #2563eb;
        margin: 1rem 0;
        line-height: 1.7;
    }
    .chat-bubble {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .chat-bubble-user {
        background: #dbeafe;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .flag-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        margin: 0.5rem 0;
    }
    .whatif-card {
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        border: 1px solid #bfdbfe;
    }
    .report-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        white-space: pre-wrap;
        line-height: 1.8;
        font-size: 0.95rem;
    }
    .step-indicator {
        display: inline-block;
        background: #2563eb;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        text-align: center;
        line-height: 28px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 8px;
    }
    div[data-testid="stSidebar"] {
        background: #ffffff;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

detector = BiasDetector()
flagger = BiasFlagger()
debiaser = Debiaser()

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')

SAMPLE_MAP = {
    '👔 Hiring Sample': 'hiring_sample.csv',
    '🏦 Loan Sample': 'loan_sample.csv',
    '🏥 Medical Sample': 'medical_sample.csv',
}

DOMAIN_DEFAULTS = {
    '👔 Hiring Sample': {'target': 'hired', 'sensitive': 'gender', 'domain': 'hiring'},
    '🏦 Loan Sample': {'target': 'loan_approved', 'sensitive': 'age_group', 'domain': 'loan'},
    '🏥 Medical Sample': {'target': 'treatment_received', 'sensitive': 'region', 'domain': 'medical'},
}


def init_session_state():
    defaults = {
        'df': None,
        'target': None,
        'sensitive': None,
        'results': None,
        'flag': None,
        'verdict': None,
        'ds_results': None,
        'group_flags': None,
        'fair_results': None,
        'fix_method': None,
        'X_tr': None, 'X_te': None,
        'y_tr': None, 'y_te': None,
        'sens_tr': None, 'sens_te': None,
        'model': None,
        'messages': [],
        'explanation': None,
        'fix_recommendation': None,
        'y_pred_original': None,
        'gemini_service': None,
        'gemini_connected': False,
        'gemini_key_saved': '',
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


def load_sample_data(filename):
    path = os.path.join(SAMPLE_DIR, filename)
    return pd.read_csv(path)


def auto_detect_columns(df):
    target = df.columns[-1]
    obj_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    sensitive = obj_cols[0] if obj_cols else df.columns[0]
    return target, sensitive


def make_metric_card(label, value, color='#2563eb'):
    border_color = f"border-top: 4px solid {color};"
    return f"""
    <div class="metric-card" style="{border_color}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """


def make_gauge_chart(score):
    if score >= 80:
        color = '#16a34a'
    elif score >= 65:
        color = '#ea580c'
    elif score >= 50:
        color = '#c2410c'
    else:
        color = '#dc2626'

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 48, 'color': color, 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#cbd5e1'},
            'bar': {'color': color, 'thickness': 0.35},
            'bgcolor': 'white',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 65], 'color': '#fed7aa'},
                {'range': [65, 80], 'color': '#fef3c7'},
                {'range': [80, 100], 'color': '#dcfce7'},
            ],
            'threshold': {
                'line': {'color': '#0f172a', 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def make_group_accuracy_chart(group_acc, avg_acc):
    groups = list(group_acc.keys())
    accs = list(group_acc.values())
    colors = ['#dc2626' if a < 0.70 else '#16a34a' for a in accs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=groups, y=accs,
        marker_color=colors,
        text=[f'{a:.1%}' for a in accs],
        textposition='outside',
        textfont=dict(size=12, color='#0f172a'),
    ))
    fig.add_hline(
        y=avg_acc, line_dash='dash',
        line_color='#64748b', line_width=2,
        annotation_text=f'Avg: {avg_acc:.1%}',
        annotation_position='top right',
        annotation_font=dict(size=11, color='#64748b')
    )
    fig.update_layout(
        title='Model Accuracy by Group',
        yaxis_title='Accuracy',
        yaxis_tickformat='.0%',
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='',
        showlegend=False,
    )
    return fig


def make_group_rate_chart(group_rates):
    groups = list(group_rates.keys())
    rates = list(group_rates.values())
    fig = go.Figure(go.Bar(
        x=groups, y=rates,
        marker_color=['#2563eb', '#ea580c', '#16a34a', '#7c3aed'][:len(groups)],
        text=[f'{r:.1%}' for r in rates],
        textposition='outside',
        textfont=dict(size=12, color='#0f172a'),
    ))
    fig.update_layout(
        title='Positive Outcome Rate by Group',
        yaxis_title='Rate',
        yaxis_tickformat='.0%',
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='',
        showlegend=False,
    )
    return fig


def make_before_after_chart(before_results, after_results):
    metrics = ['DPD', 'EOD', 'Acc Gap']
    before_vals = [
        before_results.get('demographic_parity_difference', 0),
        before_results.get('equalized_odds_difference', 0),
        before_results.get('accuracy_gap', 0),
    ]
    after_vals = [
        after_results.get('demographic_parity_difference', 0),
        after_results.get('equalized_odds_difference', 0),
        after_results.get('accuracy_gap', 0),
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Before',
        x=metrics, y=before_vals,
        marker_color='#FF6B6B',
        text=[f'{v:.4f}' for v in before_vals],
        textposition='outside',
        textfont=dict(size=11),
    ))
    fig.add_trace(go.Bar(
        name='After',
        x=metrics, y=after_vals,
        marker_color='#51CF66',
        text=[f'{v:.4f}' for v in after_vals],
        textposition='outside',
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title='Bias Metrics: Before vs After Debiasing',
        barmode='group',
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title='Value',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


def make_distribution_chart(df, sensitive, target):
    rate_df = df.groupby(sensitive)[target].mean().reset_index()
    rate_df.columns = [sensitive, 'rate']
    count_df = df.groupby(sensitive).size().reset_index(name='count')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=count_df[sensitive],
        y=count_df['count'],
        marker_color='#2563eb',
        name='Count',
        text=count_df['count'],
        textposition='outside',
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=f'Distribution of {sensitive}',
        yaxis_title='Count',
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    return fig


def run_analysis(df, target, sensitive):
    df_enc = df.copy()
    le_dict = {}
    for col in df_enc.select_dtypes(include=['object', 'string']).columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        le_dict[col] = le

    X = df_enc.drop(target, axis=1)
    y = df_enc[target]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    sens_tr = df.loc[X_tr.index, sensitive]
    sens_te = df.loc[X_te.index, sensitive]

    results = detector.analyze_model(y_te, y_pred, sens_te, sensitive)
    ds_results = detector.analyze_dataset(df, sensitive, target)
    flag = flagger.get_flag(results['fairness_score'])
    group_flags = flagger.flag_groups(ds_results['group_rates'])
    critical_flags = [f for f in group_flags if f['severity'] == 'HIGH']
    verdict = flagger.deploy_verdict(results['fairness_score'], critical_flags)
    results['model_accuracy'] = round(accuracy_score(y_te, y_pred), 4)

    st.session_state.update({
        'X_tr': X_tr, 'X_te': X_te,
        'y_tr': y_tr, 'y_te': y_te,
        'sens_tr': sens_tr, 'sens_te': sens_te,
        'model': model,
        'results': results,
        'ds_results': ds_results,
        'flag': flag,
        'group_flags': group_flags,
        'verdict': verdict,
        'y_pred_original': y_pred,
        'fair_results': None,
        'fix_method': None,
    })


# ─── HEADER ───
st.markdown("""
<div class="main-header">
    <h1>⚖️ FairSight AI</h1>
    <p>Measure • Flag • Fix AI Bias</p>
    <span class="badge badge-blue">Google Gemini AI</span>
    <span class="badge badge-green">Zero Coding</span>
    <span class="badge badge-gray">Any Domain</span>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown("#### 🔑 Google AI Setup")
    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your key here...",
        help="Get free key at ai.google.dev"
    )
    if gemini_key and gemini_key != st.session_state.gemini_key_saved:
        with st.spinner("Connecting to Gemini..."):
            _gs = GeminiService()
            _success, _msg = _gs.set_api_key(gemini_key)
            if _success:
                st.session_state.gemini_service = _gs
                st.session_state.gemini_connected = True
                st.session_state.gemini_key_saved = gemini_key
            else:
                st.session_state.gemini_service = None
                st.session_state.gemini_connected = False
                st.session_state.gemini_key_saved = ""
                st.error(f"❌ {_msg}")

    if st.session_state.gemini_connected:
        st.markdown(
            '<span style="color:#16a34a;font-weight:600;">' 
            '✅ Google AI Connected!</span>',
            unsafe_allow_html=True
        )
    else:
        st.caption("Get free key at [ai.google.dev](https://ai.google.dev)")

    st.markdown("---")
    st.markdown("#### 🏢 Domain Selection")
    domain = st.selectbox(
        "Domain",
        ['hiring', 'loan', 'medical', 'education', 'other'],
        label_visibility='collapsed'
    )

    st.markdown("---")
    st.markdown("#### 📂 Dataset Selection")
    data_choice = st.radio(
        "Choose dataset",
        ['👔 Hiring Sample', '🏦 Loan Sample',
         '🏥 Medical Sample', '📁 Upload Your CSV'],
        label_visibility='collapsed'
    )

    if data_choice == '📁 Upload Your CSV':
        uploaded = st.file_uploader(
            "Upload CSV", type=['csv'],
            label_visibility='collapsed'
        )
        if uploaded:
            st.session_state.df = pd.read_csv(uploaded)
            t, s = auto_detect_columns(st.session_state.df)
            st.session_state.target = t
            st.session_state.sensitive = s
    else:
        filename = SAMPLE_MAP[data_choice]
        st.session_state.df = load_sample_data(filename)
        defaults = DOMAIN_DEFAULTS[data_choice]
        st.session_state.target = defaults['target']
        st.session_state.sensitive = defaults['sensitive']
        if domain == defaults['domain'] or domain == 'other':
            pass

    st.markdown("---")
    st.markdown("#### ℹ️ About FairSight AI")
    st.caption(
        "FairSight AI helps organizations detect, understand, "
        "and fix bias in their AI models — no coding required. "
        "Upload your data, get instant fairness analysis, "
        "and generate professional audit reports."
    )

    st.markdown("---")
    if st.button("🔄 Reset Analysis", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ─── MAIN TABS ───
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset", "🔍 Detect & Flag", "🤖 AI Explain",
    "🔧 Fix Bias", "📄 Report"
])

# ═══════════════════════════════════════════
# TAB 1 — Dataset
# ═══════════════════════════════════════════
with tab1:
    if st.session_state.df is not None:
        df = st.session_state.df
        target = st.session_state.target
        sensitive = st.session_state.sensitive

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                make_metric_card('Records', f'{len(df):,}'),
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                make_metric_card('Features', len(df.columns) - 1),
                unsafe_allow_html=True
            )
        with col3:
            pos_rate = df[target].mean()
            st.markdown(
                make_metric_card(
                    'Positive Rate',
                    f'{pos_rate:.1%}',
                    '#16a34a' if 0.3 <= pos_rate <= 0.7 else '#ea580c'
                ),
                unsafe_allow_html=True
            )
        with col4:
            missing = df.isnull().sum().sum()
            st.markdown(
                make_metric_card(
                    'Missing Values',
                    missing,
                    '#16a34a' if missing == 0 else '#dc2626'
                ),
                unsafe_allow_html=True
            )

        st.markdown("#### Data Preview")
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.session_state.target = st.selectbox(
                "🎯 Target Column (what AI predicts)",
                df.columns.tolist(),
                index=df.columns.tolist().index(target) if target in df.columns else len(df.columns) - 1
            )
        with col_right:
            st.session_state.sensitive = st.selectbox(
                "🔒 Sensitive Column (gender/age/region)",
                df.columns.tolist(),
                index=df.columns.tolist().index(sensitive) if sensitive in df.columns else 0
            )

        target = st.session_state.target
        sensitive = st.session_state.sensitive

        st.markdown("#### Distribution Analysis")
        fig_dist = make_distribution_chart(df, sensitive, target)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.success(
            "✅ Dataset loaded! Go to **Detect & Flag** tab to analyze for bias →"
        )
    else:
        st.info(
            "👈 Select a sample dataset or upload your CSV in the sidebar to get started."
        )

# ═══════════════════════════════════════════
# TAB 2 — Detect & Flag
# ═══════════════════════════════════════════
with tab2:
    if st.session_state.df is not None:
        st.markdown("""
        <div class="info-card">
            <strong>How this works:</strong> We train a machine learning model on your data,
            then measure whether it treats different groups fairly.
            This detects hidden bias that humans often miss.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Analyze Now!", type="primary", use_container_width=True):
            with st.spinner("Analyzing your data for bias..."):
                progress = st.progress(0, text="Step 1/4: Preprocessing data...")
                try:
                    df = st.session_state.df.copy()
                    target = st.session_state.target
                    sensitive = st.session_state.sensitive

                    progress.progress(25, text="Step 2/4: Training model...")
                    import time
                    time.sleep(0.3)

                    progress.progress(50, text="Step 3/4: Calculating bias metrics...")
                    run_analysis(df, target, sensitive)

                    progress.progress(75, text="Step 4/4: Generating insights...")
                    time.sleep(0.3)

                    progress.progress(100, text="Analysis complete!")
                    time.sleep(0.3)
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.stop()

        if st.session_state.results is not None:
            results = st.session_state.results
            flag = st.session_state.flag
            verdict = st.session_state.verdict
            ds_results = st.session_state.ds_results
            group_flags = st.session_state.group_flags or []

            # Score display
            st.markdown("### Fairness Score")
            gauge_fig = make_gauge_chart(results['fairness_score'])
            st.plotly_chart(gauge_fig, use_container_width=True)

            score_color = flag['color']
            st.markdown(f"""
            <div class="score-display" style="background:{flag['bg']};">
                <div class="score-number" style="color:{score_color};">
                    {flag['emoji']} {results['fairness_score']}
                </div>
                <div class="score-label" style="color:{score_color};">
                    {flag['label']}
                </div>
                <div class="score-message" style="color:#64748b;">
                    {flag['message']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Deploy verdict
            st.markdown(f"""
            <div class="verdict-box" style="background:{verdict['bg']};color:{verdict['color']};">
                {verdict['verdict']}
                <div class="verdict-reason" style="color:#64748b;">
                    {verdict['reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 4 metric columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                dpd = results['demographic_parity_difference']
                dpd_color = '#dc2626' if dpd > 0.2 else '#ea580c' if dpd > 0.1 else '#16a34a'
                st.markdown(
                    make_metric_card('DPD', f'{dpd:.4f}', dpd_color),
                    unsafe_allow_html=True
                )
                st.caption(results['dpd_status'])
            with col2:
                eod = results['equalized_odds_difference']
                eod_color = '#dc2626' if eod > 0.2 else '#ea580c' if eod > 0.1 else '#16a34a'
                st.markdown(
                    make_metric_card('EOD', f'{eod:.4f}', eod_color),
                    unsafe_allow_html=True
                )
                st.caption(results['eod_status'])
            with col3:
                ag = results['accuracy_gap']
                ag_color = '#dc2626' if ag > 0.15 else '#ea580c' if ag > 0.08 else '#16a34a'
                st.markdown(
                    make_metric_card('Accuracy Gap', f'{ag:.4f}', ag_color),
                    unsafe_allow_html=True
                )
            with col4:
                ma = results.get('model_accuracy', 0)
                st.markdown(
                    make_metric_card('Model Accuracy', f'{ma:.1%}', '#2563eb'),
                    unsafe_allow_html=True
                )

            # Flagged groups
            if group_flags:
                st.markdown("### ⚠️ Flagged Groups")
                for gf in group_flags:
                    border = '#dc2626' if gf['severity'] == 'HIGH' else '#ea580c'
                    st.markdown(f"""
                    <div class="flag-card" style="border-left:4px solid {border};">
                        <strong>{gf['group']}</strong> —
                        <span style="color:{border};font-weight:600;">
                            {gf['severity']}
                        </span><br>
                        <span style="color:#64748b;">{gf['message']}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Group accuracy chart
            group_acc = results.get('group_accuracies', {})
            if group_acc:
                avg_acc = np.mean(list(group_acc.values()))
                fig_acc = make_group_accuracy_chart(group_acc, avg_acc)
                st.plotly_chart(fig_acc, use_container_width=True)

            # Group selection rate chart
            if ds_results:
                fig_rate = make_group_rate_chart(ds_results['group_rates'])
                st.plotly_chart(fig_rate, use_container_width=True)

            # What-if preview
            if ds_results and ds_results.get('most_favored') and ds_results.get('least_favored'):
                rates = ds_results['group_rates']
                favored = ds_results['most_favored']
                least = ds_results['least_favored']
                favored_rate = rates[favored]
                least_rate = rates[least]
                if favored_rate > 0:
                    ratio = least_rate / favored_rate
                    x_val = int(round(ratio * 100))
                    st.markdown(f"""
                    <div class="whatif-card">
                        <strong>💡 What does this mean?</strong><br>
                        If 100 people from the <strong>{favored}</strong> group
                        receive a positive outcome, only about
                        <strong style="color:#dc2626;font-size:1.2em;">{x_val}</strong>
                        people from the <strong>{least}</strong> group
                        get the same outcome.
                    </div>
                    """, unsafe_allow_html=True)

            st.success(
                "✅ Bias detected! Go to **AI Explain** tab for a plain-language explanation →"
            )
        else:
            st.info(
                "👆 Click **Analyze Now!** above to train a model and detect bias in your data."
            )
    else:
        st.info("👈 Load a dataset first from the sidebar.")

# ═══════════════════════════════════════════
# TAB 3 — AI Explain
# ═══════════════════════════════════════════
with tab3:
    if st.session_state.results is not None:
        st.markdown("""
        <span class="badge badge-blue">Powered by Google Gemini</span>
        """, unsafe_allow_html=True)

        col_main, col_chat = st.columns([2, 1])

        with col_main:
            st.markdown("#### 🧠 AI Bias Explanation")

            if st.button("✨ Explain in Simple Language", type="primary", use_container_width=True):
                with st.spinner("Gemini AI is analyzing the bias..."):
                    try:
                        gs = st.session_state.gemini_service

                        if gs is None:

                            raise Exception("Please add your Gemini API key in the sidebar and wait for the ✅ Connected status.")
                        explanation = gs.explain_bias(
                            st.session_state.results,
                            st.session_state.sensitive,
                            domain
                        )
                        st.session_state.explanation = explanation
                    except Exception as e:
                        st.session_state.explanation = f"Error: {str(e)}"

            if st.session_state.explanation:
                st.markdown(f"""
                <div class="explanation-card">
                    {st.session_state.explanation}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 💡 Fix Recommendation")

            if st.button("Get Fix Recommendation", type="primary", use_container_width=True):
                with st.spinner("Gemini AI is recommending a fix..."):
                    try:
                        gs = st.session_state.gemini_service

                        if gs is None:

                            raise Exception("Please add your Gemini API key in the sidebar and wait for the ✅ Connected status.")
                        results = st.session_state.results
                        prompt_context = f"Fairness score: {results['fairness_score']}, DPD: {results['demographic_parity_difference']}, EOD: {results['equalized_odds_difference']}"
                        fix_rec = gs.chat(
                            "What is the best method to fix this bias? "
                            "Should I use re-weighting or fairness constraints? "
                            "Explain in simple terms.",
                            prompt_context
                        )
                        st.session_state.fix_recommendation = fix_rec
                    except Exception as e:
                        st.session_state.fix_recommendation = f"Error: {str(e)}"

            if st.session_state.fix_recommendation:
                st.markdown(f"""
                <div class="explanation-card">
                    {st.session_state.fix_recommendation}
                </div>
                """, unsafe_allow_html=True)

        with col_chat:
            st.markdown("#### 💬 Ask FairSight AI")

            question = st.text_input(
                "Ask a question about your bias analysis:",
                placeholder="Type your question...",
                label_visibility='collapsed'
            )

            if question:
                with st.spinner("Thinking..."):
                    try:
                        gs = st.session_state.gemini_service

                        if gs is None:

                            raise Exception("Please add your Gemini API key in the sidebar and wait for the ✅ Connected status.")
                        results = st.session_state.results
                        bias_ctx = (
                            f"Score: {results['fairness_score']}, "
                            f"DPD: {results['demographic_parity_difference']}, "
                            f"Feature: {st.session_state.sensitive}"
                        )
                        answer = gs.chat(question, bias_ctx)
                        st.session_state.messages.append(
                            {'q': question, 'a': answer}
                        )
                    except Exception as e:
                        st.session_state.messages.append(
                            {'q': question, 'a': f"Error: {str(e)}"}
                        )

            if st.session_state.messages:
                for msg in st.session_state.messages[-4:]:
                    st.markdown(f"""
                    <div class="chat-bubble-user">
                        <strong>You:</strong> {msg['q']}
                    </div>
                    <div class="chat-bubble">
                        <strong>AI:</strong> {msg['a']}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("##### Quick Questions")
            quick_qs = [
                "Is this bias illegal?",
                "Who is most affected?",
                "How serious is this?"
            ]
            for qq in quick_qs:
                if st.button(qq, key=f"quick_{qq}", use_container_width=True):
                    with st.spinner("Thinking..."):
                        try:
                            gs = st.session_state.gemini_service

                            if gs is None:

                                raise Exception("Please add your Gemini API key in the sidebar and wait for the ✅ Connected status.")
                            results = st.session_state.results
                            bias_ctx = (
                                f"Score: {results['fairness_score']}, "
                                f"DPD: {results['demographic_parity_difference']}, "
                                f"Feature: {st.session_state.sensitive}"
                            )
                            answer = gs.chat(qq, bias_ctx)
                            st.session_state.messages.append(
                                {'q': qq, 'a': answer}
                            )
                        except Exception as e:
                            st.session_state.messages.append(
                                {'q': qq, 'a': f"Error: {str(e)}"}
                            )
                    st.rerun()
    else:
        st.info("Run bias analysis in the **Detect & Flag** tab first.")

# ═══════════════════════════════════════════
# TAB 4 — Fix Bias
# ═══════════════════════════════════════════
with tab4:
    if st.session_state.results is not None:
        current_score = st.session_state.results['fairness_score']
        score_color = '#dc2626' if current_score < 50 else '#ea580c' if current_score < 75 else '#16a34a'

        st.markdown(f"""
        <div class="info-card-warning">
            <strong>Current Fairness Score:</strong>
            <span style="color:{score_color};font-size:1.5rem;font-weight:700;">
                {current_score}/100
            </span><br>
            <span style="color:#64748b;">
                Select a debiasing method below to improve fairness.
            </span>
        </div>
        """, unsafe_allow_html=True)

        fix_method = st.radio(
            "Choose debiasing method:",
            [
                '⚡ Re-weighting — Fast (10 seconds)',
                '🎯 Fairness Constraint — Best Quality (30 seconds)'
            ],
            index=0
        )

        method_key = 'reweighting' if 'Re-weighting' in fix_method else 'fairness_constraint'

        if method_key == 'reweighting':
            st.markdown("""
            <div class="info-card">
                <strong>Re-weighting</strong> gives underrepresented groups
                more influence during model training. This balances the data
                without changing the model architecture. Fast and effective
                for most bias types.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card">
                <strong>Fairness Constraint</strong> adds a mathematical
                fairness rule during training. The model optimizes for both
                accuracy and fairness simultaneously. Slower but produces
                the most fair results.
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔧 Fix Bias Now!", type="primary", use_container_width=True):
            with st.spinner("Applying debiasing... This may take a moment."):
                try:
                    X_tr = st.session_state.X_tr
                    X_te = st.session_state.X_te
                    y_tr = st.session_state.y_tr
                    y_te = st.session_state.y_te
                    sens_tr = st.session_state.sens_tr
                    sens_te = st.session_state.sens_te
                    sensitive = st.session_state.sensitive

                    if method_key == 'reweighting':
                        fair_model = debiaser.reweighting(
                            X_tr, y_tr, sens_tr)
                    else:
                        fair_model = debiaser.fairness_constraint(
                            X_tr, y_tr, sens_tr)

                    y_pred_fair = fair_model.predict(X_te)

                    fair_results = detector.analyze_model(
                        y_te, y_pred_fair, sens_te, sensitive)
                    fair_results['model_accuracy'] = round(
                        accuracy_score(y_te, y_pred_fair), 4)

                    st.session_state.fair_results = fair_results
                    st.session_state.fix_method = method_key

                except Exception as e:
                    st.error(f"Debiasing failed: {str(e)}")
                    st.stop()

        if st.session_state.fair_results is not None:
            before = st.session_state.results
            after = st.session_state.fair_results
            improvement = after['fairness_score'] - before['fairness_score']

            col1, col2, col3 = st.columns(3)
            with col1:
                before_color = '#dc2626' if before['fairness_score'] < 50 else '#ea580c' if before['fairness_score'] < 75 else '#16a34a'
                st.markdown(
                    make_metric_card(
                        'Before Score',
                        f"{before['fairness_score']}/100",
                        before_color
                    ),
                    unsafe_allow_html=True
                )
            with col2:
                after_color = '#dc2626' if after['fairness_score'] < 50 else '#ea580c' if after['fairness_score'] < 75 else '#16a34a'
                st.markdown(
                    make_metric_card(
                        'After Score',
                        f"{after['fairness_score']}/100",
                        after_color
                    ),
                    unsafe_allow_html=True
                )
            with col3:
                imp_color = '#16a34a' if improvement > 0 else '#dc2626'
                sign = '+' if improvement > 0 else ''
                st.markdown(
                    make_metric_card(
                        'Improvement',
                        f"{sign}{improvement:.1f} pts",
                        imp_color
                    ),
                    unsafe_allow_html=True
                )

            if improvement > 15:
                st.balloons()

            # Comparison table
            st.markdown("### 📊 Detailed Comparison")
            metrics_data = {
                'Metric': ['DPD', 'EOD', 'Accuracy Gap', 'Fairness Score'],
                'Before': [
                    f"{before['demographic_parity_difference']:.4f}",
                    f"{before['equalized_odds_difference']:.4f}",
                    f"{before['accuracy_gap']:.4f}",
                    f"{before['fairness_score']}/100"
                ],
                'After': [
                    f"{after['demographic_parity_difference']:.4f}",
                    f"{after['equalized_odds_difference']:.4f}",
                    f"{after['accuracy_gap']:.4f}",
                    f"{after['fairness_score']}/100"
                ],
            }
            # Calculate changes
            changes = []
            for i, metric_name in enumerate(metrics_data['Metric']):
                if metric_name == 'Fairness Score':
                    change = after['fairness_score'] - before['fairness_score']
                    changes.append(f"{'+' if change > 0 else ''}{change:.0f} pts")
                else:
                    before_val = [before['demographic_parity_difference'],
                                  before['equalized_odds_difference'],
                                  before['accuracy_gap']][i]
                    after_val = [after['demographic_parity_difference'],
                                 after['equalized_odds_difference'],
                                 after['accuracy_gap']][i]
                    if before_val > 0:
                        pct = ((after_val - before_val) / before_val) * 100
                        changes.append(f"{pct:+.1f}%")
                    else:
                        changes.append("N/A")
            metrics_data['Change'] = changes

            comp_df = pd.DataFrame(metrics_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Before vs After chart
            fig_compare = make_before_after_chart(before, after)
            st.plotly_chart(fig_compare, use_container_width=True)

            # New deploy verdict
            new_flag = flagger.get_flag(after['fairness_score'])
            new_ds = detector.analyze_dataset(
                st.session_state.df,
                st.session_state.sensitive,
                st.session_state.target
            )
            new_group_flags = flagger.flag_groups(new_ds['group_rates'])
            new_critical = [f for f in new_group_flags if f['severity'] == 'HIGH']
            new_verdict = flagger.deploy_verdict(
                after['fairness_score'], new_critical)

            st.markdown(f"""
            <div class="verdict-box" style="background:{new_verdict['bg']};color:{new_verdict['color']};">
                {new_verdict['verdict']}
                <div class="verdict-reason" style="color:#64748b;">
                    {new_verdict['reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.success(
                "✅ Bias fixed! Go to **Report** tab to generate your audit report →"
            )
    else:
        st.info("Run bias analysis in the **Detect & Flag** tab first.")

# ═══════════════════════════════════════════
# TAB 5 — Report
# ═══════════════════════════════════════════
with tab5:
    if st.session_state.results is not None:
        st.markdown("### 📄 AI Bias Audit Report")

        org_name = st.text_input(
            "Organization Name",
            placeholder="Enter your organization name...",
            value=""
        )

        if st.button("📄 Generate AI Audit Report", type="primary", use_container_width=True):
            with st.spinner("Generating professional audit report..."):
                try:
                    gs = st.session_state.gemini_service

                    if gs is None:

                        raise Exception("Please add your Gemini API key in the sidebar and wait for the ✅ Connected status.")
                    fix_desc = "None"
                    if st.session_state.fix_method == 'reweighting':
                        fix_desc = "Re-weighting debiasing applied"
                    elif st.session_state.fix_method == 'fairness_constraint':
                        fix_desc = "Fairness constraint debiasing applied"

                    report_results = st.session_state.fair_results or st.session_state.results
                    report_text = gs.generate_report(
                        org_name or "Organization",
                        domain,
                        report_results,
                        fix_desc
                    )
                    st.session_state.report_text = report_text
                except Exception as e:
                    st.session_state.report_text = f"Report generation failed: {str(e)}"

        if hasattr(st.session_state, 'report_text') and st.session_state.report_text:
            st.markdown(f"""
            <div class="report-card">{st.session_state.report_text}</div>
            """, unsafe_allow_html=True)

            st.download_button(
                "📥 Download Report (.txt)",
                data=st.session_state.report_text,
                file_name=f"fairsight_audit_report_{domain}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("Click the button above to generate your audit report.")
    else:
        st.info("Run bias analysis in the **Detect & Flag** tab first.")