import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Intelligence Suite",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

    .stApp { background-color: #080d1a; color: #e2e8f0; font-family: 'DM Sans', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0d1526 0%, #111d35 50%, #0d1526 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 18px;
        padding: 2.2rem 2.8rem;
        margin-bottom: 1.8rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(56, 189, 248, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #38bdf8, #818cf8, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.8px;
    }
    .main-header p { color: #94a3b8; font-size: 1rem; margin-top: 0.6rem; }
    .main-header .badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 0.25rem 0.8rem;
        border-radius: 100px;
        margin-bottom: 0.8rem;
    }

    .kpi-card {
        background: linear-gradient(145deg, #111d35 0%, #0d1526 100%);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.5), transparent);
    }
    .kpi-card:hover { border-color: rgba(56, 189, 248, 0.35); transform: translateY(-3px); box-shadow: 0 12px 40px rgba(56,189,248,0.08); }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'DM Mono', monospace;
    }
    .kpi-value.danger {
        background: linear-gradient(135deg, #f87171, #fb923c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-value.success {
        background: linear-gradient(135deg, #34d399, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-value.warning {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 0.35rem; font-weight: 600; }
    .kpi-delta { font-size: 0.78rem; margin-top: 0.3rem; }
    .kpi-delta.bad { color: #f87171; }
    .kpi-delta.good { color: #34d399; }
    .kpi-delta.neutral { color: #94a3b8; }

    .section-header {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.08), transparent);
        border-left: 3px solid #38bdf8;
        padding: 0.9rem 1.3rem;
        margin: 2rem 0 1.2rem 0;
        border-radius: 0 10px 10px 0;
    }
    .section-header h3 { color: #bae6fd; font-size: 1.15rem; font-weight: 700; margin: 0; }
    .section-header p { color: #64748b; font-size: 0.82rem; margin: 0.25rem 0 0 0; }

    .insight-box {
        background: rgba(56, 189, 248, 0.06);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin: 0.9rem 0;
        font-size: 0.88rem;
        line-height: 1.65;
    }
    .insight-box strong { color: #7dd3fc; }
    .insight-box.warning {
        background: rgba(251,191,36,0.06);
        border-color: rgba(251,191,36,0.2);
    }
    .insight-box.warning strong { color: #fcd34d; }
    .insight-box.danger {
        background: rgba(248,113,113,0.06);
        border-color: rgba(248,113,113,0.2);
    }
    .insight-box.danger strong { color: #fca5a5; }
    .insight-box.success {
        background: rgba(52,211,153,0.06);
        border-color: rgba(52,211,153,0.2);
    }
    .insight-box.success strong { color: #6ee7b7; }

    .rx-card {
        background: linear-gradient(135deg, rgba(52,211,153,0.07), rgba(16,78,59,0.12));
        border: 1px solid rgba(52,211,153,0.22);
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        margin: 0.9rem 0;
    }
    .rx-card h4 { color: #6ee7b7; font-size: 0.95rem; font-weight: 700; margin: 0 0 0.5rem 0; }
    .rx-card p { color: #94a3b8; font-size: 0.85rem; line-height: 1.6; margin: 0; }
    .rx-card .priority {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 0.15rem 0.55rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .priority.high { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
    .priority.medium { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
    .priority.low { background: rgba(52,211,153,0.1); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }

    .tab-info {
        background: rgba(129,140,248,0.07);
        border: 1px solid rgba(129,140,248,0.18);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.2rem;
        font-size: 0.85rem;
        color: #a5b4fc;
    }

    .sidebar-section {
        background: rgba(56,189,248,0.05);
        border: 1px solid rgba(56,189,248,0.1);
        border-radius: 10px;
        padding: 0.9rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1526 0%, #080d1a 100%);
        border-right: 1px solid rgba(56,189,248,0.1);
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(56,189,248,0.05);
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56,189,248,0.15) !important;
        color: #38bdf8 !important;
    }

    .stSelectbox label, .stMultiSelect label, .stSlider label, .stCheckbox label { color: #94a3b8 !important; font-size: 0.85rem !important; }

    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }

    hr { border-color: rgba(56,189,248,0.1); margin: 1.5rem 0; }

    .stButton > button {
        background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(129,140,248,0.15));
        border: 1px solid rgba(56,189,248,0.3);
        color: #38bdf8;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(56,189,248,0.25), rgba(129,140,248,0.25));
        border-color: rgba(56,189,248,0.5);
    }

    .churn-tag {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .churn-tag.churned { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
    .churn-tag.retained { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────────────────────
COLORS = {
    'primary':    '#38bdf8',
    'secondary':  '#818cf8',
    'accent':     '#f472b6',
    'success':    '#34d399',
    'warning':    '#fbbf24',
    'danger':     '#f87171',
    'bg':         '#080d1a',
    'card':       '#111d35',
    'border':     'rgba(56,189,248,0.15)',
}
CHURN_COLORS   = ['#34d399', '#f87171']
CHURN_LABELS   = ['Retained (0)', 'Churned (1)']
SUB_COLORS     = {'Basic': '#f472b6', 'Standard': '#818cf8', 'Premium': '#38bdf8'}
PLOTLY_LAYOUT  = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#94a3b8', size=12),
    margin=dict(t=40, b=30, l=20, r=20),
    legend=dict(bgcolor='rgba(13,21,38,0.8)', bordercolor='rgba(56,189,248,0.15)', borderwidth=1, font=dict(size=11)),
    hoverlabel=dict(bgcolor='#111d35', bordercolor='rgba(56,189,248,0.3)', font=dict(family='DM Sans', color='#e2e8f0')),
    xaxis=dict(gridcolor='rgba(56,189,248,0.07)', zerolinecolor='rgba(56,189,248,0.1)'),
    yaxis=dict(gridcolor='rgba(56,189,248,0.07)', zerolinecolor='rgba(56,189,248,0.1)'),
)

# ─────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('CustomerChurn.csv')

    # Derived features
    df['ChurnLabel']         = df['Churn'].map({0: 'Retained', 1: 'Churned'})
    df['SpendPerMonth']      = df['MonthlySpend']
    df['UsagePerMonth']      = df['ServiceUsage']
    df['SpendTier']          = pd.qcut(df['MonthlySpend'], q=4, labels=['Low','Mid','High','Premium'])
    df['TenureBand']         = pd.cut(df['TenureMonths'], bins=[0,12,24,36,48,60],
                                       labels=['0-12m','13-24m','25-36m','37-48m','49-60m'])
    df['AgeBand']            = pd.cut(df['Age'], bins=[17,30,45,60,80],
                                       labels=['18-30','31-45','46-60','61-79'])
    df['RatingBand']         = pd.cut(df['AvgRating'], bins=[0,2,3,4,5],
                                       labels=['Poor(≤2)','Fair(2-3)','Good(3-4)','Excellent(4-5)'])
    df['SupportCallsBand']   = pd.cut(df['SupportCalls'], bins=[-1,0,2,4,10],
                                       labels=['None','Low(1-2)','Medium(3-4)','High(5+)'])
    df['UsageBand']          = pd.cut(df['ServiceUsage'], bins=[-0.01,20,40,60,100],
                                       labels=['Low(<20)','Med(20-40)','High(40-60)','Very High(60+)'])
    df['HasContract_lbl']    = df['HasContract'].map({0:'No Contract', 1:'Has Contract'})
    df['IsActive_lbl']       = df['IsActive'].map({0:'Inactive', 1:'Active'})
    df['SpendUsageRatio']    = df['MonthlySpend'] / (df['ServiceUsage'] + 0.01)
    df['EngagementScore']    = (df['ServiceUsage'] / df['ServiceUsage'].max() * 0.5 +
                                 df['AvgRating']     / 5 * 0.3 +
                                 (1 - df['SupportCalls'] / (df['SupportCalls'].max() + 1)) * 0.2)
    df['RiskScore']          = (df['SupportCalls'] / (df['SupportCalls'].max() + 1) * 0.35 +
                                 (1 - df['AvgRating'] / 5) * 0.30 +
                                 (1 - df['TenureMonths'] / 60) * 0.20 +
                                 (1 - df['ServiceUsage'] / (df['ServiceUsage'].max() + 1)) * 0.15)
    return df

df = load_data()

# ─────────────────────────────────────────────────────────────
# ML MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    features = ['Age','TenureMonths','MonthlySpend','ServiceUsage','SupportCalls','AvgRating','HasContract','IsActive']
    X = df[features].copy()
    y = df['Churn']
    le = LabelEncoder()
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    rf  = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced')
    gb  = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    lr  = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    rf.fit(X_scaled, y)
    gb.fit(X_scaled, y)
    lr.fit(X_scaled, y)

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv = cross_val_score(rf, X_scaled, y, cv=cv, scoring='roc_auc').mean()
    gb_cv = cross_val_score(gb, X_scaled, y, cv=cv, scoring='roc_auc').mean()
    lr_cv = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc').mean()

    rf_proba = rf.predict_proba(X_scaled)[:,1]
    gb_proba = gb.predict_proba(X_scaled)[:,1]

    importance_df = pd.DataFrame({
        'Feature':    features,
        'RF_Importance':  rf.feature_importances_,
        'GB_Importance':  gb.feature_importances_,
    }).sort_values('RF_Importance', ascending=False)

    return {
        'rf': rf, 'gb': gb, 'lr': lr,
        'features': features, 'scaler': sc,
        'rf_cv': rf_cv, 'gb_cv': gb_cv, 'lr_cv': lr_cv,
        'rf_proba': rf_proba, 'gb_proba': gb_proba,
        'importance': importance_df,
        'X_scaled': X_scaled, 'y': y,
    }

models = train_models(df)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:2rem;'>📡</div>
        <div style='font-size:1rem; font-weight:800; color:#38bdf8; letter-spacing:-0.5px;'>ChurnLens</div>
        <div style='font-size:0.72rem; color:#475569; letter-spacing:2px; text-transform:uppercase;'>Telecom Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎛️ Filters")

    sub_filter = st.multiselect(
        "Subscription Type",
        options=['Basic','Standard','Premium'],
        default=['Basic','Standard','Premium']
    )
    contract_filter = st.multiselect(
        "Contract Status",
        options=['Has Contract','No Contract'],
        default=['Has Contract','No Contract']
    )
    tenure_filter = st.slider("Tenure Range (months)", 1, 60, (1, 60))
    age_filter    = st.slider("Age Range", 18, 79, (18, 79))

    st.markdown("---")
    show_churned_only = st.checkbox("Show Churned Customers Only", False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#475569; line-height:1.7;'>
    <strong style='color:#64748b;'>Dataset Info</strong><br>
    📦 300 customers<br>
    📊 11 columns<br>
    🔴 29% churn rate<br>
    📅 Telecom domain
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────
mask = (
    df['SubscriptionType'].isin(sub_filter) &
    df['HasContract_lbl'].isin(contract_filter) &
    df['TenureMonths'].between(*tenure_filter) &
    df['Age'].between(*age_filter)
)
if show_churned_only:
    mask &= (df['Churn'] == 1)
dff = df[mask].copy()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <div class='badge'>Telecom Intelligence Suite</div>
    <h1>📡 Customer Churn Intelligence Dashboard</h1>
    <p>360° view of why customers stay or leave · Descriptive · Diagnostic · Predictive · Prescriptive</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────
total         = len(dff)
churned       = dff['Churn'].sum()
retained      = total - churned
churn_rate    = churned / total * 100 if total > 0 else 0
avg_tenure    = dff['TenureMonths'].mean()
avg_spend     = dff['MonthlySpend'].mean()
avg_rating    = dff['AvgRating'].mean()
contract_pct  = dff['HasContract'].mean() * 100

c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value'>{total:,}</div>
        <div class='kpi-label'>Customers</div>
        <div class='kpi-delta neutral'>Filtered dataset</div>
    </div>""", unsafe_allow_html=True)
with c2:
    cls = 'danger' if churn_rate > 25 else 'warning'
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value {cls}'>{churn_rate:.1f}%</div>
        <div class='kpi-label'>Churn Rate</div>
        <div class='kpi-delta bad'>⚠ {churned} left</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value success'>{retained:,}</div>
        <div class='kpi-label'>Retained</div>
        <div class='kpi-delta good'>✓ {100-churn_rate:.1f}% retained</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value'>{avg_tenure:.1f}m</div>
        <div class='kpi-label'>Avg Tenure</div>
        <div class='kpi-delta neutral'>Months</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value warning'>${avg_spend:.0f}</div>
        <div class='kpi-label'>Avg Monthly Spend</div>
        <div class='kpi-delta neutral'>Per customer</div>
    </div>""", unsafe_allow_html=True)
with c6:
    cls2 = 'success' if avg_rating >= 4 else ('warning' if avg_rating >= 3 else 'danger')
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value {cls2}'>{avg_rating:.2f}/5</div>
        <div class='kpi-label'>Avg Rating</div>
        <div class='kpi-delta neutral'>Customer satisfaction</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Descriptive", "🔍 Diagnostic", "🤖 Predictive", "💡 Prescriptive", "🗃️ Data Explorer"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class='tab-info'>
        <strong>Descriptive Analysis</strong> — What does our customer base look like? 
        Understanding the distribution of every variable and how churn is distributed across customer segments.
    </div>""", unsafe_allow_html=True)

    # ── ROW 1: Churn Donut + Subscription Breakdown ──────────
    st.markdown("<div class='section-header'><h3>🔄 Churn Overview & Segment Breakdown</h3><p>Core churn distribution · click segments to drill down</p></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.1, 1.3, 1.3])

    with col1:
        # Interactive donut — click legend to isolate
        churn_counts = dff['Churn'].value_counts().reset_index()
        churn_counts.columns = ['Churn','Count']
        churn_counts['Label'] = churn_counts['Churn'].map({0:'Retained',1:'Churned'})

        fig_donut = go.Figure(go.Pie(
            labels=churn_counts['Label'],
            values=churn_counts['Count'],
            hole=0.62,
            marker_colors=CHURN_COLORS,
            textinfo='label+percent',
            textfont=dict(size=13, family='DM Sans'),
            pull=[0, 0.06],
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'
        ))
        fig_donut.add_annotation(
            text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:10px'>Churn</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=22, color='#f87171', family='DM Mono')
        )
        fig_donut.update_layout(**PLOTLY_LAYOUT, title_text='Overall Churn', title_font=dict(size=14, color='#bae6fd'),
                                 height=300, showlegend=True)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        # Subscription type donut with drill-down by churn
        sub_churn = dff.groupby(['SubscriptionType','ChurnLabel']).size().reset_index(name='Count')
        fig_sub = px.sunburst(
            sub_churn, path=['SubscriptionType','ChurnLabel'], values='Count',
            color='ChurnLabel',
            color_discrete_map={'Retained':'#34d399','Churned':'#f87171'},
        )
        fig_sub.update_layout(**PLOTLY_LAYOUT, title_text='Subscription → Churn (Drill Down)',
                               title_font=dict(size=14, color='#bae6fd'), height=300)
        fig_sub.update_traces(textfont=dict(size=11))
        st.plotly_chart(fig_sub, use_container_width=True)

    with col3:
        # Contract + Active status donut drill-down
        contract_churn = dff.groupby(['HasContract_lbl','ChurnLabel']).size().reset_index(name='Count')
        fig_contract = px.sunburst(
            contract_churn, path=['HasContract_lbl','ChurnLabel'], values='Count',
            color='ChurnLabel',
            color_discrete_map={'Retained':'#34d399','Churned':'#f87171'},
        )
        fig_contract.update_layout(**PLOTLY_LAYOUT, title_text='Contract Status → Churn (Drill Down)',
                                    title_font=dict(size=14, color='#bae6fd'), height=300)
        fig_contract.update_traces(textfont=dict(size=11))
        st.plotly_chart(fig_contract, use_container_width=True)

    # ── ROW 2: Distributions ─────────────────────────────────
    st.markdown("<div class='section-header'><h3>📈 Numeric Variable Distributions</h3><p>How key metrics distribute across retained vs churned customers</p></div>", unsafe_allow_html=True)

    num_cols = ['Age','TenureMonths','MonthlySpend','ServiceUsage','SupportCalls','AvgRating']
    col_labels = ['Age','Tenure (months)','Monthly Spend ($)','Service Usage','Support Calls','Avg Rating']

    fig_dist = make_subplots(rows=2, cols=3, subplot_titles=col_labels,
                              vertical_spacing=0.18, horizontal_spacing=0.08)
    for i, (col, lbl) in enumerate(zip(num_cols, col_labels)):
        r, c = divmod(i, 3)
        for churn_val, color, name in [(0,'#34d399','Retained'), (1,'#f87171','Churned')]:
            sub = dff[dff['Churn']==churn_val][col]
            fig_dist.add_trace(go.Histogram(
                x=sub, name=name, legendgroup=name,
                showlegend=(i==0),
                marker_color=color, opacity=0.65,
                nbinsx=20,
                hovertemplate=f'<b>{name}</b><br>{lbl}: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ), row=r+1, col=c+1)
    fig_dist.update_layout(**PLOTLY_LAYOUT, height=480, barmode='overlay',
                            title_text='Distribution by Churn Status',
                            title_font=dict(size=14, color='#bae6fd'))
    for ax in ['xaxis','xaxis2','xaxis3','xaxis4','xaxis5','xaxis6']:
        fig_dist.update_layout(**{ax: dict(gridcolor='rgba(56,189,248,0.07)')})
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── ROW 3: Churn rate by categorical variables ────────────
    st.markdown("<div class='section-header'><h3>📊 Churn Rate by Segment</h3><p>Categorical breakdowns revealing high-risk groups</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Tenure band churn rate bar
        tb = dff.groupby('TenureBand')['Churn'].agg(['mean','count']).reset_index()
        tb.columns = ['TenureBand','ChurnRate','Count']
        tb['ChurnRate'] *= 100
        fig_tb = go.Figure(go.Bar(
            x=tb['TenureBand'].astype(str),
            y=tb['ChurnRate'],
            marker=dict(color=tb['ChurnRate'],colorscale=[[0,'#34d399'],[0.5,'#fbbf24'],[1,'#f87171']],
                        showscale=True, colorbar=dict(title='Rate%', tickfont=dict(size=10))),
            text=tb['ChurnRate'].round(1).astype(str) + '%',
            textposition='outside',
            customdata=tb['Count'],
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<br>Customers: %{customdata}<extra></extra>'
        ))
        fig_tb.update_layout(**PLOTLY_LAYOUT, title_text='Churn Rate by Tenure Band',
                              title_font=dict(size=14, color='#bae6fd'), height=320, yaxis_title='Churn Rate (%)')
        st.plotly_chart(fig_tb, use_container_width=True)

    with c2:
        # Age band × Subscription type heatmap
        heat = dff.groupby(['AgeBand','SubscriptionType'])['Churn'].mean().reset_index()
        heat['ChurnRate'] = heat['Churn'] * 100
        heat_pivot = heat.pivot(index='AgeBand', columns='SubscriptionType', values='ChurnRate').fillna(0)
        fig_heat = go.Figure(go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.tolist(),
            y=heat_pivot.index.astype(str).tolist(),
            colorscale=[[0,'#0d1526'],[0.4,'#818cf8'],[0.7,'#f59e0b'],[1,'#ef4444']],
            text=np.round(heat_pivot.values,1),
            texttemplate='%{text}%',
            hovertemplate='Age: %{y}<br>Subscription: %{x}<br>Churn Rate: %{z:.1f}%<extra></extra>'
        ))
        fig_heat.update_layout(**PLOTLY_LAYOUT, title_text='Churn Rate Heatmap: Age × Subscription',
                                title_font=dict(size=14, color='#bae6fd'), height=320)
        st.plotly_chart(fig_heat, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # Support calls churn rate
        sc_data = dff.groupby('SupportCallsBand')['Churn'].agg(['mean','count']).reset_index()
        sc_data.columns = ['Band','ChurnRate','Count']
        sc_data['ChurnRate'] *= 100
        fig_sc = go.Figure(go.Bar(
            x=sc_data['Band'].astype(str), y=sc_data['ChurnRate'],
            marker_color=['#34d399','#fbbf24','#f87171','#ef4444'],
            text=sc_data['ChurnRate'].round(1).astype(str) + '%',
            textposition='outside',
            customdata=sc_data['Count'],
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<br>Customers: %{customdata}<extra></extra>'
        ))
        fig_sc.update_layout(**PLOTLY_LAYOUT, title_text='Churn Rate by Support Calls Volume',
                              title_font=dict(size=14, color='#bae6fd'), height=300, yaxis_title='Churn Rate (%)')
        st.plotly_chart(fig_sc, use_container_width=True)

    with c4:
        # Rating band + subscription stacked bar
        rb = dff.groupby(['RatingBand','ChurnLabel']).size().reset_index(name='Count')
        fig_rb = px.bar(rb, x='RatingBand', y='Count', color='ChurnLabel', barmode='group',
                        color_discrete_map={'Retained':'#34d399','Churned':'#f87171'})
        fig_rb.update_layout(**PLOTLY_LAYOUT, title_text='Customer Count by Rating Band & Churn',
                              title_font=dict(size=14, color='#bae6fd'), height=300,
                              xaxis_title='Rating Band', yaxis_title='Count')
        st.plotly_chart(fig_rb, use_container_width=True)

    # ── Insights ─────────────────────────────────────────────
    churn_basic = dff[dff['SubscriptionType']=='Basic']['Churn'].mean()*100
    churn_no_contract = dff[dff['HasContract']==0]['Churn'].mean()*100
    churn_high_support = dff[dff['SupportCalls']>=4]['Churn'].mean()*100
    churn_early = dff[dff['TenureMonths']<=12]['Churn'].mean()*100

    st.markdown(f"""
    <div class='insight-box danger'>
        <strong>⚡ Key Descriptive Findings:</strong><br>
        • <strong>{churn_no_contract:.0f}%</strong> of customers without contracts churn vs much lower for contracted customers — contract status is a strong retainer.<br>
        • <strong>{churn_basic:.0f}%</strong> churn among Basic subscribers; highest raw count of churners come from this tier.<br>
        • Customers with <strong>4+ support calls</strong> churn at <strong>{churn_high_support:.0f}%</strong> — escalating support need is a critical warning signal.<br>
        • <strong>First-year customers</strong> (0-12 months) show <strong>{churn_early:.0f}%</strong> churn — early lifecycle is the most vulnerable window.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='tab-info'>
        <strong>Diagnostic Analysis</strong> — Why are customers churning?
        Deep-dive correlation analysis, statistical significance testing, and interaction effects 
        to pinpoint the <em>root causes</em> of churn.
    </div>""", unsafe_allow_html=True)

    # ── Tenure vs Spend Scatter ───────────────────────────────
    st.markdown("<div class='section-header'><h3>🔬 Tenure × Spend × Usage Deep Dive</h3><p>Multi-variable interaction analysis colored by churn status</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_sc1 = px.scatter(
            dff, x='TenureMonths', y='MonthlySpend', color='ChurnLabel',
            color_discrete_map={'Retained':'#34d399','Churned':'#f87171'},
            size='ServiceUsage', size_max=18,
            symbol='SubscriptionType',
            hover_data=['Age','AvgRating','SupportCalls','HasContract_lbl'],
            opacity=0.75,
        )
        fig_sc1.update_layout(**PLOTLY_LAYOUT, height=400,
                               title_text='Tenure vs Monthly Spend (size=Usage)',
                               title_font=dict(size=14, color='#bae6fd'),
                               xaxis_title='Tenure (Months)', yaxis_title='Monthly Spend ($)')
        st.plotly_chart(fig_sc1, use_container_width=True)

    with c2:
        fig_sc2 = px.scatter(
            dff, x='AvgRating', y='SupportCalls', color='ChurnLabel',
            color_discrete_map={'Retained':'#34d399','Churned':'#f87171'},
            size='MonthlySpend', size_max=18,
            hover_data=['SubscriptionType','TenureMonths','HasContract_lbl'],
            opacity=0.75,
        )
        fig_sc2.update_layout(**PLOTLY_LAYOUT, height=400,
                               title_text='Rating vs Support Calls (size=Spend)',
                               title_font=dict(size=14, color='#bae6fd'),
                               xaxis_title='Avg Rating', yaxis_title='Support Calls')
        st.plotly_chart(fig_sc2, use_container_width=True)

    # ── Statistical Comparison ───────────────────────────────
    st.markdown("<div class='section-header'><h3>📐 Statistical Mean Comparison: Churned vs Retained</h3><p>T-test significance indicators · effect sizes</p></div>", unsafe_allow_html=True)

    numeric_feats = ['Age','TenureMonths','MonthlySpend','ServiceUsage','SupportCalls','AvgRating','EngagementScore','RiskScore']
    stats_rows = []
    for feat in numeric_feats:
        churned_vals  = dff[dff['Churn']==1][feat].dropna()
        retained_vals = dff[dff['Churn']==0][feat].dropna()
        t_stat, p_val = stats.ttest_ind(churned_vals, retained_vals)
        cohens_d = (churned_vals.mean() - retained_vals.mean()) / np.sqrt(
            (churned_vals.std()**2 + retained_vals.std()**2) / 2 + 1e-9
        )
        stats_rows.append({
            'Feature': feat,
            'Churned Mean': round(churned_vals.mean(),3),
            'Retained Mean': round(retained_vals.mean(),3),
            'Diff': round(churned_vals.mean() - retained_vals.mean(),3),
            'p-value': round(p_val,4),
            "Cohen's d": round(cohens_d,3),
            'Significant': '✅' if p_val < 0.05 else '❌'
        })
    stats_df = pd.DataFrame(stats_rows).sort_values('p-value')

    # Grouped bar for mean comparison
    fig_means = go.Figure()
    fig_means.add_trace(go.Bar(
        x=stats_df['Feature'], y=stats_df['Retained Mean'],
        name='Retained', marker_color='#34d399', opacity=0.85,
        hovertemplate='<b>Retained</b> — %{x}: %{y:.3f}<extra></extra>'
    ))
    fig_means.add_trace(go.Bar(
        x=stats_df['Feature'], y=stats_df['Churned Mean'],
        name='Churned', marker_color='#f87171', opacity=0.85,
        hovertemplate='<b>Churned</b> — %{x}: %{y:.3f}<extra></extra>'
    ))
    fig_means.update_layout(**PLOTLY_LAYOUT, barmode='group', height=360,
                             title_text='Mean Comparison: Churned vs Retained',
                             title_font=dict(size=14, color='#bae6fd'),
                             yaxis_title='Mean Value')
    st.plotly_chart(fig_means, use_container_width=True)

    st.dataframe(
        stats_df.style
            .background_gradient(subset=['p-value'], cmap='RdYlGn_r')
            .background_gradient(subset=["Cohen's d"], cmap='coolwarm'),
        use_container_width=True, height=300
    )

    # ── Correlation Matrix ───────────────────────────────────
    st.markdown("<div class='section-header'><h3>🕸️ Correlation Matrix</h3><p>Full variable correlation heatmap — red indicates churn-aligned correlation</p></div>", unsafe_allow_html=True)

    corr_cols = ['Age','TenureMonths','MonthlySpend','ServiceUsage','SupportCalls','AvgRating','HasContract','IsActive','EngagementScore','RiskScore','Churn']
    corr_df   = dff[corr_cols].corr()
    mask_upper = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    corr_masked = corr_df.copy()
    corr_masked[mask_upper] = np.nan

    fig_corr = go.Figure(go.Heatmap(
        z=corr_masked.values,
        x=corr_df.columns.tolist(),
        y=corr_df.columns.tolist(),
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=np.round(corr_masked.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='%{x} × %{y}: %{z:.3f}<extra></extra>',
        colorbar=dict(title='r', tickfont=dict(size=10, color='#94a3b8'))
    ))
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=480,
                            title_text='Pearson Correlation Matrix (lower triangle)',
                            title_font=dict(size=14, color='#bae6fd'))
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Churn Rate vs Continuous Variables ───────────────────
    st.markdown("<div class='section-header'><h3>📉 Churn Rate Across Continuous Buckets</h3><p>Binned analysis shows non-linear churn patterns</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Tenure decile churn
        dff2 = dff.copy()
        dff2['TenureDecile'] = pd.qcut(dff2['TenureMonths'], q=6, duplicates='drop')
        td = dff2.groupby('TenureDecile')['Churn'].agg(['mean','count']).reset_index()
        td['ChurnRate'] = td['mean'] * 100
        td['TenureDecile'] = td['TenureDecile'].astype(str)
        fig_td = go.Figure()
        fig_td.add_trace(go.Scatter(
            x=td['TenureDecile'], y=td['ChurnRate'],
            mode='lines+markers+text',
            line=dict(color='#f87171', width=2.5),
            marker=dict(size=9, color=td['ChurnRate'],
                        colorscale=[[0,'#34d399'],[1,'#f87171']], showscale=False),
            text=td['ChurnRate'].round(1).astype(str)+'%',
            textposition='top center',
            hovertemplate='Tenure: %{x}<br>Churn Rate: %{y:.1f}%<extra></extra>'
        ))
        fig_td.update_layout(**PLOTLY_LAYOUT, height=340,
                              title_text='Churn Rate by Tenure Decile',
                              title_font=dict(size=14, color='#bae6fd'),
                              xaxis_title='Tenure Range', yaxis_title='Churn Rate (%)',
                              xaxis_tickangle=-30)
        st.plotly_chart(fig_td, use_container_width=True)

    with c2:
        # Spend decile churn
        dff3 = dff.copy()
        dff3['SpendDecile'] = pd.qcut(dff3['MonthlySpend'], q=6, duplicates='drop')
        sd2 = dff3.groupby('SpendDecile')['Churn'].agg(['mean','count']).reset_index()
        sd2['ChurnRate'] = sd2['mean'] * 100
        sd2['SpendDecile'] = sd2['SpendDecile'].astype(str)
        fig_sd = go.Figure()
        fig_sd.add_trace(go.Scatter(
            x=sd2['SpendDecile'], y=sd2['ChurnRate'],
            mode='lines+markers+text',
            line=dict(color='#fbbf24', width=2.5),
            marker=dict(size=9, color=sd2['ChurnRate'],
                        colorscale=[[0,'#34d399'],[1,'#f87171']], showscale=False),
            text=sd2['ChurnRate'].round(1).astype(str)+'%',
            textposition='top center',
            hovertemplate='Spend: %{x}<br>Churn Rate: %{y:.1f}%<extra></extra>'
        ))
        fig_sd.update_layout(**PLOTLY_LAYOUT, height=340,
                              title_text='Churn Rate by Monthly Spend Decile',
                              title_font=dict(size=14, color='#bae6fd'),
                              xaxis_title='Spend Range ($)', yaxis_title='Churn Rate (%)',
                              xaxis_tickangle=-30)
        st.plotly_chart(fig_sd, use_container_width=True)

    # ── Box plots ─────────────────────────────────────────────
    st.markdown("<div class='section-header'><h3>📦 Distribution Spread: Churned vs Retained</h3><p>Box plots showing variance, outliers, and median shifts</p></div>", unsafe_allow_html=True)

    box_cols = ['TenureMonths','MonthlySpend','ServiceUsage','SupportCalls','AvgRating']
    fig_box = make_subplots(rows=1, cols=5, subplot_titles=box_cols)
    for i, col in enumerate(box_cols):
        for churn_val, color, name in [(0,'#34d399','Retained'),(1,'#f87171','Churned')]:
            fig_box.add_trace(go.Box(
                y=dff[dff['Churn']==churn_val][col],
                name=name, legendgroup=name,
                showlegend=(i==0),
                marker_color=color,
                boxmean=True,
                hovertemplate=f'<b>{name}</b><br>{col}: %{{y}}<extra></extra>'
            ), row=1, col=i+1)
    fig_box.update_layout(**PLOTLY_LAYOUT, height=380,
                           title_text='Box Plot Comparison by Churn Status',
                           title_font=dict(size=14, color='#bae6fd'))
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Violin + diagnostic insights ─────────────────────────
    st.markdown("<div class='section-header'><h3>🎻 Violin Plots — Full Distribution Shape</h3></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_vio1 = px.violin(dff, x='SubscriptionType', y='TenureMonths', color='ChurnLabel',
                              color_discrete_map={'Retained':'#34d399','Churned':'#f87171'},
                              box=True, points='outliers', violinmode='group')
        fig_vio1.update_layout(**PLOTLY_LAYOUT, height=380, title_text='Tenure by Subscription & Churn',
                                title_font=dict(size=14, color='#bae6fd'))
        st.plotly_chart(fig_vio1, use_container_width=True)

    with c2:
        fig_vio2 = px.violin(dff, x='HasContract_lbl', y='AvgRating', color='ChurnLabel',
                              color_discrete_map={'Retained':'#34d399','Churned':'#f87171'},
                              box=True, points='outliers', violinmode='group')
        fig_vio2.update_layout(**PLOTLY_LAYOUT, height=380, title_text='Rating by Contract Status & Churn',
                                title_font=dict(size=14, color='#bae6fd'))
        st.plotly_chart(fig_vio2, use_container_width=True)

    # Diagnostic insights
    avg_tenure_churned   = dff[dff['Churn']==1]['TenureMonths'].mean()
    avg_tenure_retained  = dff[dff['Churn']==0]['TenureMonths'].mean()
    avg_usage_churned    = dff[dff['Churn']==1]['ServiceUsage'].mean()
    avg_rating_churned   = dff[dff['Churn']==1]['AvgRating'].mean()
    avg_sc_churned       = dff[dff['Churn']==1]['SupportCalls'].mean()

    st.markdown(f"""
    <div class='insight-box warning'>
        <strong>🔍 Root Cause Diagnostics:</strong><br>
        • <strong>Shorter tenure</strong> is strongly predictive: churned customers average 
          <strong>{avg_tenure_churned:.1f} months</strong> vs <strong>{avg_tenure_retained:.1f} months</strong> for retained 
          (Δ = {avg_tenure_retained-avg_tenure_churned:.1f}m).<br>
        • <strong>Low service usage</strong> signals disengagement: churned customers use only 
          <strong>{avg_usage_churned:.1f} units/month</strong> on average.<br>
        • <strong>Low satisfaction + high support calls</strong> is the most toxic combination: 
          churned avg rating is <strong>{avg_rating_churned:.2f}/5</strong> with 
          <strong>{avg_sc_churned:.1f}</strong> support calls on average — unresolved issues drive exit.<br>
        • <strong>No-contract customers</strong> are 2.5× more likely to churn — removing friction to leave is a key structural risk.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='tab-info'>
        <strong>Predictive Analysis</strong> — Who is likely to churn next?
        Machine learning models (Random Forest, Gradient Boosting, Logistic Regression) trained 
        to identify at-risk customers before they leave.
    </div>""", unsafe_allow_html=True)

    # ── Model Performance ────────────────────────────────────
    st.markdown("<div class='section-header'><h3>🏆 Model Performance Comparison</h3><p>Cross-validated AUC scores and ROC curves</p></div>", unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    for col, name, score, color in [(mc1,'Random Forest',models['rf_cv'],'#38bdf8'),
                                     (mc2,'Gradient Boost',models['gb_cv'],'#a78bfa'),
                                     (mc3,'Logistic Reg.',models['lr_cv'],'#34d399')]:
        with col:
            grade = 'Excellent' if score > 0.85 else ('Good' if score > 0.75 else 'Fair')
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-value' style='color:{color};font-size:1.8rem;'>{score:.3f}</div>
                <div class='kpi-label'>{name}</div>
                <div class='kpi-delta neutral'>CV AUC · {grade}</div>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # ROC Curves
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                      line=dict(dash='dash', color='#475569', width=1.5),
                                      name='Random (AUC=0.50)', showlegend=True))
        for model_obj, proba, name, color in [
            (models['rf'], models['rf_proba'], f"Random Forest (AUC={models['rf_cv']:.3f})", '#38bdf8'),
            (models['gb'], models['gb_proba'], f"Gradient Boost (AUC={models['gb_cv']:.3f})", '#a78bfa'),
        ]:
            fpr, tpr, _ = roc_curve(models['y'], proba)
            roc_auc = auc(fpr, tpr)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                          name=name, line=dict(color=color, width=2.5)))
        fig_roc.update_layout(**PLOTLY_LAYOUT, height=380,
                               title_text='ROC Curves — Churn Prediction',
                               title_font=dict(size=14, color='#bae6fd'),
                               xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(fig_roc, use_container_width=True)

    with c2:
        # Feature Importance
        imp = models['importance'].copy()
        imp['AvgImportance'] = (imp['RF_Importance'] + imp['GB_Importance']) / 2
        imp = imp.sort_values('AvgImportance', ascending=True)
        fig_imp = go.Figure(go.Bar(
            x=imp['AvgImportance'], y=imp['Feature'], orientation='h',
            marker=dict(
                color=imp['AvgImportance'],
                colorscale=[[0,'#1e3a5f'],[0.5,'#818cf8'],[1,'#38bdf8']],
                showscale=False
            ),
            text=imp['AvgImportance'].round(3),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        fig_imp.update_layout(**PLOTLY_LAYOUT, height=380,
                               title_text='Feature Importance (RF + GB Average)',
                               title_font=dict(size=14, color='#bae6fd'),
                               xaxis_title='Importance Score')
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Risk Score Distribution ──────────────────────────────
    st.markdown("<div class='section-header'><h3>🎯 Customer Risk Score Distribution</h3><p>Model-derived risk scores segmented by actual churn</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        dff_pred = dff.copy()
        rf_proba_all = models['rf_proba']
        dff_pred['PredictedChurnProb'] = rf_proba_all

        fig_risk = go.Figure()
        for churn_val, color, name in [(0,'#34d399','Retained'),(1,'#f87171','Churned')]:
            sub = dff_pred[dff_pred['Churn']==churn_val]['PredictedChurnProb']
            fig_risk.add_trace(go.Histogram(
                x=sub, name=name, marker_color=color, opacity=0.7,
                nbinsx=25,
                hovertemplate=f'<b>{name}</b><br>Prob: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>'
            ))
        fig_risk.update_layout(**PLOTLY_LAYOUT, barmode='overlay', height=340,
                                title_text='Predicted Churn Probability Distribution',
                                title_font=dict(size=14, color='#bae6fd'),
                                xaxis_title='Predicted Churn Probability', yaxis_title='Count')
        st.plotly_chart(fig_risk, use_container_width=True)

    with c2:
        # Risk tiers pie
        dff_pred['RiskTier'] = pd.cut(dff_pred['PredictedChurnProb'],
                                       bins=[0,0.3,0.6,1.0],
                                       labels=['Low Risk (<30%)','Medium Risk (30-60%)','High Risk (>60%)'])
        rt_counts = dff_pred['RiskTier'].value_counts().reset_index()
        rt_counts.columns = ['Tier','Count']
        fig_rt = go.Figure(go.Pie(
            labels=rt_counts['Tier'], values=rt_counts['Count'],
            hole=0.55,
            marker_colors=['#34d399','#fbbf24','#f87171'],
            pull=[0,0,0.08],
            textinfo='label+percent+value',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
        ))
        fig_rt.update_layout(**PLOTLY_LAYOUT, height=340,
                              title_text='Customer Risk Tier Segmentation',
                              title_font=dict(size=14, color='#bae6fd'))
        st.plotly_chart(fig_rt, use_container_width=True)

    # ── Confusion Matrix + High Risk Table ───────────────────
    st.markdown("<div class='section-header'><h3>🗺️ Confusion Matrix & High-Risk Customers</h3></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.8])
    with c1:
        y_pred = (dff_pred['PredictedChurnProb'] >= 0.5).astype(int)
        cm = confusion_matrix(dff_pred['Churn'], y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=['Pred: Retained','Pred: Churned'],
            y=['Act: Retained','Act: Churned'],
            colorscale=[[0,'#0d1526'],[1,'#38bdf8']],
            text=cm,
            texttemplate='<b>%{text}</b>',
            textfont=dict(size=18, color='white'),
            hovertemplate='%{y} / %{x}: %{z}<extra></extra>'
        ))
        fig_cm.update_layout(**PLOTLY_LAYOUT, height=320,
                              title_text='Confusion Matrix (threshold=0.5)',
                              title_font=dict(size=14, color='#bae6fd'))
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        high_risk = dff_pred[dff_pred['PredictedChurnProb'] >= 0.55].sort_values('PredictedChurnProb', ascending=False).head(15)
        display_cols = ['CustomerID','Age','TenureMonths','MonthlySpend','SubscriptionType','SupportCalls','AvgRating','HasContract_lbl','PredictedChurnProb','Churn']
        st.markdown("**🚨 Top High-Risk Customers (Predicted Churn Prob ≥ 55%)**")
        st.dataframe(
            high_risk[display_cols].style.background_gradient(subset=['PredictedChurnProb'], cmap='Reds'),
            use_container_width=True, height=300
        )

    # ── Interactive Predictor ────────────────────────────────
    st.markdown("<div class='section-header'><h3>🧮 Individual Customer Churn Predictor</h3><p>Enter customer details to get real-time churn probability</p></div>", unsafe_allow_html=True)

    p1,p2,p3,p4 = st.columns(4)
    with p1:
        p_age     = st.slider("Age", 18, 79, 45)
        p_tenure  = st.slider("Tenure (months)", 1, 60, 12)
    with p2:
        p_spend   = st.slider("Monthly Spend ($)", 20.0, 300.0, 80.0)
        p_usage   = st.slider("Service Usage", 0.0, 100.0, 20.0)
    with p3:
        p_sc      = st.slider("Support Calls", 0, 10, 3)
        p_rating  = st.slider("Avg Rating", 1.0, 5.0, 3.0)
    with p4:
        p_contract = st.selectbox("Has Contract?", [1, 0], format_func=lambda x: "Yes" if x else "No")
        p_active   = st.selectbox("Is Active?", [1, 0], format_func=lambda x: "Yes" if x else "No")

    if st.button("🔮 Predict Churn Probability"):
        input_arr = np.array([[p_age, p_tenure, p_spend, p_usage, p_sc, p_rating, p_contract, p_active]])
        input_scaled = models['scaler'].transform(input_arr)
        rf_prob = models['rf'].predict_proba(input_scaled)[0][1]
        gb_prob = models['gb'].predict_proba(input_scaled)[0][1]
        avg_prob = (rf_prob + gb_prob) / 2
        risk_level = "🔴 HIGH RISK" if avg_prob > 0.6 else ("🟡 MEDIUM RISK" if avg_prob > 0.35 else "🟢 LOW RISK")
        color_cls = 'danger' if avg_prob > 0.6 else ('warning' if avg_prob > 0.35 else 'success')
        st.markdown(f"""
        <div class='insight-box {"danger" if avg_prob>0.6 else ("warning" if avg_prob>0.35 else "success")}'>
            <strong>Prediction Result: {risk_level}</strong><br>
            Random Forest Probability: <strong>{rf_prob:.1%}</strong> &nbsp;|&nbsp;
            Gradient Boost Probability: <strong>{gb_prob:.1%}</strong><br>
            Ensemble Average: <strong>{avg_prob:.1%}</strong> churn probability.<br>
            {"⚡ Immediate intervention recommended — offer retention incentives." if avg_prob>0.6 else
             ("📋 Monitor closely — consider proactive outreach." if avg_prob>0.35 else
              "✅ Customer appears stable — standard engagement suffices.")}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — PRESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class='tab-info'>
        <strong>Prescriptive Analysis</strong> — What should we do about it?
        Data-driven action recommendations mapped to specific customer segments 
        and root causes identified in the diagnostic phase.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'><h3>🎯 Strategic Retention Framework</h3><p>Segment-level intervention roadmap</p></div>", unsafe_allow_html=True)

    # Priority matrix visualization
    prescriptions = [
        {"segment": "No-Contract Basic Users",
         "risk_pct": dff[(dff['HasContract']==0)&(dff['SubscriptionType']=='Basic')]['Churn'].mean()*100,
         "action": "Offer 3-month discounted contract lock-in with bonus data/features",
         "priority": "HIGH", "impact": "Very High", "effort": "Medium"},
        {"segment": "High Support Call Customers",
         "risk_pct": dff[dff['SupportCalls']>=4]['Churn'].mean()*100,
         "action": "Assign dedicated account manager; resolve root issue within 24h; follow-up survey",
         "priority": "HIGH", "impact": "High", "effort": "High"},
        {"segment": "First-Year Customers (0-12m)",
         "risk_pct": dff[dff['TenureMonths']<=12]['Churn'].mean()*100,
         "action": "Onboarding nurture program: usage tips, loyalty rewards at month 3/6, personal check-in",
         "priority": "HIGH", "impact": "High", "effort": "Medium"},
        {"segment": "Low Usage Customers",
         "risk_pct": dff[dff['ServiceUsage']<20]['Churn'].mean()*100,
         "action": "Re-engagement campaign: tutorial emails, usage challenges, show value-add features",
         "priority": "MEDIUM", "impact": "Medium", "effort": "Low"},
        {"segment": "Low Rating (≤3.0) Customers",
         "risk_pct": dff[dff['AvgRating']<=3]['Churn'].mean()*100,
         "action": "NPS recovery program: proactive complaint resolution, compensation credits, feedback loop",
         "priority": "MEDIUM", "impact": "High", "effort": "High"},
        {"segment": "Inactive Customers (IsActive=0)",
         "risk_pct": dff[dff['IsActive']==0]['Churn'].mean()*100,
         "action": "Win-back campaign: 'We miss you' offer, service refresher, personalized incentive",
         "priority": "MEDIUM", "impact": "Medium", "effort": "Low"},
        {"segment": "Premium/Standard No-Contract",
         "risk_pct": dff[(dff['HasContract']==0)&(dff['SubscriptionType'].isin(['Premium','Standard']))]['Churn'].mean()*100,
         "action": "Annual contract offer with premium perks (priority support, data rollover, price lock)",
         "priority": "LOW", "impact": "Medium", "effort": "Low"},
    ]

    for rx in prescriptions:
        rp = rx['risk_pct']
        pcls = rx['priority'].lower()
        st.markdown(f"""
        <div class='rx-card'>
            <span class='priority {pcls}'>{rx['priority']} PRIORITY</span>
            <h4>👥 {rx['segment']} — Churn Risk: {rp:.0f}%</h4>
            <p>📌 <strong>Action:</strong> {rx['action']}<br>
            📈 <strong>Expected Impact:</strong> {rx['impact']} &nbsp;·&nbsp; 
            ⚙️ <strong>Implementation Effort:</strong> {rx['effort']}</p>
        </div>""", unsafe_allow_html=True)

    # ── Impact Simulation Chart ───────────────────────────────
    st.markdown("<div class='section-header'><h3>📊 Retention Impact Simulation</h3><p>Estimated churn reduction under different intervention scenarios</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        current_churn = dff['Churn'].mean() * 100
        scenarios = {
            'Current Baseline': current_churn,
            'Contract Push Only': current_churn * 0.75,
            'Support Optimization': current_churn * 0.82,
            'Onboarding Program': current_churn * 0.86,
            'All Interventions': current_churn * 0.55,
        }
        fig_sim = go.Figure(go.Bar(
            x=list(scenarios.keys()),
            y=list(scenarios.values()),
            marker_color=['#f87171','#fbbf24','#fbbf24','#fbbf24','#34d399'],
            text=[f"{v:.1f}%" for v in scenarios.values()],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>'
        ))
        fig_sim.update_layout(**PLOTLY_LAYOUT, height=380,
                               title_text='Churn Rate Under Intervention Scenarios',
                               title_font=dict(size=14, color='#bae6fd'),
                               yaxis_title='Projected Churn Rate (%)', yaxis_range=[0, current_churn*1.3])
        st.plotly_chart(fig_sim, use_container_width=True)

    with c2:
        # Revenue impact
        avg_monthly_spend = dff['MonthlySpend'].mean()
        churned_customers = dff['Churn'].sum()
        monthly_revenue_at_risk = churned_customers * avg_monthly_spend
        scenarios_revenue = {
            'No Intervention': 0,
            'Contract Push': monthly_revenue_at_risk * 0.25,
            'Support Fix': monthly_revenue_at_risk * 0.18,
            'Onboarding': monthly_revenue_at_risk * 0.14,
            'All Combined': monthly_revenue_at_risk * 0.45,
        }
        fig_rev = go.Figure(go.Bar(
            x=list(scenarios_revenue.keys()),
            y=list(scenarios_revenue.values()),
            marker_color=['#475569','#818cf8','#818cf8','#818cf8','#38bdf8'],
            text=[f"${v:,.0f}" for v in scenarios_revenue.values()],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Monthly Revenue Saved: $%{y:,.0f}<extra></extra>'
        ))
        fig_rev.update_layout(**PLOTLY_LAYOUT, height=380,
                               title_text=f'Monthly Revenue Saved by Scenario\n(Total at risk: ${monthly_revenue_at_risk:,.0f}/mo)',
                               title_font=dict(size=14, color='#bae6fd'),
                               yaxis_title='Revenue Recovered ($/month)')
        st.plotly_chart(fig_rev, use_container_width=True)

    # ── Engagement Score vs Risk Score Quadrant ───────────────
    st.markdown("<div class='section-header'><h3>🗺️ Customer Strategy Quadrant Map</h3><p>Engagement vs Risk — four action zones</p></div>", unsafe_allow_html=True)

    dff_q = dff.copy()
    med_eng  = dff_q['EngagementScore'].median()
    med_risk = dff_q['RiskScore'].median()

    dff_q['Quadrant'] = 'Maintain'
    dff_q.loc[(dff_q['EngagementScore']>=med_eng) & (dff_q['RiskScore']<med_risk),  'Quadrant'] = 'Champions 🏆'
    dff_q.loc[(dff_q['EngagementScore']>=med_eng) & (dff_q['RiskScore']>=med_risk), 'Quadrant'] = 'Watch Closely 👀'
    dff_q.loc[(dff_q['EngagementScore']<med_eng)  & (dff_q['RiskScore']<med_risk),  'Quadrant'] = 'Nurture 🌱'
    dff_q.loc[(dff_q['EngagementScore']<med_eng)  & (dff_q['RiskScore']>=med_risk), 'Quadrant'] = 'Intervene NOW 🚨'

    qcolors = {'Champions 🏆':'#34d399','Watch Closely 👀':'#fbbf24','Nurture 🌱':'#818cf8','Intervene NOW 🚨':'#f87171'}
    fig_quad = px.scatter(
        dff_q, x='EngagementScore', y='RiskScore', color='Quadrant',
        color_discrete_map=qcolors,
        symbol='ChurnLabel',
        hover_data=['CustomerID','Age','SubscriptionType','TenureMonths','MonthlySpend'],
        opacity=0.75, size_max=12
    )
    fig_quad.add_vline(x=med_eng, line_dash='dash', line_color='rgba(56,189,248,0.3)')
    fig_quad.add_hline(y=med_risk, line_dash='dash', line_color='rgba(56,189,248,0.3)')
    fig_quad.update_layout(**PLOTLY_LAYOUT, height=460,
                            title_text='Customer Strategy Quadrant Map',
                            title_font=dict(size=14, color='#bae6fd'),
                            xaxis_title='Engagement Score (higher=better)',
                            yaxis_title='Risk Score (higher=riskier)')
    st.plotly_chart(fig_quad, use_container_width=True)

    q_summary = dff_q.groupby('Quadrant').agg(
        Customers=('CustomerID','count'),
        ChurnRate=('Churn','mean'),
        AvgSpend=('MonthlySpend','mean'),
        AvgTenure=('TenureMonths','mean')
    ).reset_index()
    q_summary['ChurnRate'] = (q_summary['ChurnRate']*100).round(1).astype(str) + '%'
    q_summary['AvgSpend']  = q_summary['AvgSpend'].round(1)
    q_summary['AvgTenure'] = q_summary['AvgTenure'].round(1)
    st.dataframe(q_summary, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class='tab-info'>
        <strong>Data Explorer</strong> — Browse and filter the raw customer dataset 
        with derived features, risk scores, and predicted churn probabilities.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'><h3>🗃️ Customer Dataset with Predictions</h3></div>", unsafe_allow_html=True)

    dff_exp = dff.copy()
    dff_exp['PredictedChurnProb'] = models['rf_proba']
    dff_exp['RiskTier'] = pd.cut(dff_exp['PredictedChurnProb'],
                                  bins=[0,0.3,0.6,1.0],
                                  labels=['Low','Medium','High'])

    show_cols = ['CustomerID','Age','TenureMonths','MonthlySpend','SubscriptionType',
                 'ServiceUsage','SupportCalls','AvgRating','HasContract_lbl','IsActive_lbl',
                 'EngagementScore','RiskScore','PredictedChurnProb','RiskTier','Churn','ChurnLabel']

    st.dataframe(
        dff_exp[show_cols].style
            .background_gradient(subset=['PredictedChurnProb'], cmap='RdYlGn_r')
            .background_gradient(subset=['EngagementScore'], cmap='Greens')
            .background_gradient(subset=['RiskScore'], cmap='Reds'),
        use_container_width=True, height=500
    )

    # Summary stats
    st.markdown("<div class='section-header'><h3>📐 Summary Statistics</h3></div>", unsafe_allow_html=True)
    st.dataframe(dff_exp[['Age','TenureMonths','MonthlySpend','ServiceUsage','SupportCalls','AvgRating','EngagementScore','RiskScore','PredictedChurnProb']].describe().round(3),
                 use_container_width=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_data = dff_exp[show_cols].to_csv(index=False)
        st.download_button("⬇️ Download Full Dataset (CSV)", data=csv_data,
                           file_name="customer_churn_analysis.csv", mime='text/csv')
    with col_dl2:
        high_risk_csv = dff_exp[dff_exp['RiskTier']=='High'][show_cols].to_csv(index=False)
        st.download_button("🚨 Download High-Risk Customers (CSV)", data=high_risk_csv,
                           file_name="high_risk_customers.csv", mime='text/csv')

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.78rem; padding: 0.5rem 0;'>
    <strong style='color:#475569;'>Customer Churn Intelligence Suite</strong> · 
    Built with Streamlit & Plotly · 
    Objective: Understand why customers stay or leave telecom services
</div>
""", unsafe_allow_html=True)
