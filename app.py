import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import json
import os
import random

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodBloom 🌸",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,600;1,300&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #fdf6f0 0%, #fef0f5 50%, #f0f4fe 100%);
}

#MainMenu, footer, header { visibility: hidden; }

h1 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    color: #2d1b4e !important;
    letter-spacing: -1px;
}
h2, h3 {
    font-family: 'Fraunces', serif !important;
    color: #2d1b4e !important;
}

.stButton > button {
    background: linear-gradient(135deg, #b06ab3, #4568dc) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 10px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 15px rgba(176,106,179,0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(176,106,179,0.4) !important;
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.75) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    border: 1px solid rgba(255,255,255,0.9) !important;
    box-shadow: 0 2px 12px rgba(180,140,200,0.08) !important;
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.6) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.8);
}

.insight-box {
    background: linear-gradient(135deg, rgba(176,106,179,0.1), rgba(69,104,220,0.1));
    border-left: 3px solid #b06ab3;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-style: italic;
    color: #4a3060;
}

.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #9b8ab8;
    font-family: 'Fraunces', serif;
}
.empty-state .icon { font-size: 64px; margin-bottom: 16px; }
.empty-state h3 { color: #9b8ab8 !important; font-size: 1.4rem; }
.empty-state p { font-family: 'DM Sans', sans-serif; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_FILE = "mood_data.json"

MOOD_EMOJIS = {
    "😄 Joyful": 5,
    "🙂 Good": 4,
    "😐 Neutral": 3,
    "😕 Low": 2,
    "😞 Struggling": 1,
}

ENERGY_LABELS = {
    1: "🪫 Drained",
    2: "😴 Tired",
    3: "⚡ Okay",
    4: "🔥 Energized",
    5: "🚀 Buzzing",
}

HABITS = [
    "💧 Drank water",
    "🏃 Exercised",
    "🥗 Ate well",
    "📵 Less screen time",
    "🧘 Meditated",
    "📚 Read",
    "🌿 Spent time outside",
    "🎵 Listened to music",
]

AFFIRMATIONS = [
    "You showed up today — that's everything. 🌸",
    "Small steps still move you forward. ✨",
    "Your feelings are valid. Be gentle with yourself. 💜",
    "Rest is productive too. 🌙",
    "You are doing better than you think. 🌱",
    "Every day is a new chapter. 📖",
    "You deserve the kindness you give others. 🤍",
]

# ── Data helpers ──────────────────────────────────────────────────────────────
def load_data():
    """Load user entries from a local JSON file. Returns [] if none exist yet."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_entry(entry):
    """Append one entry dict and write to disk."""
    data = load_data()
    data.append(entry)
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def clear_all_data():
    with open(DATA_FILE, "w") as f:
        json.dump([], f)

def entries_to_df(entries):
    """Turn list-of-dicts into a sorted DataFrame."""
    df = pd.DataFrame(entries)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def calc_streak(entries):
    """Count consecutive days up to today that have an entry."""
    logged = {e["date"] for e in entries}
    streak = 0
    for i in range(len(entries) + 30):
        check = (date.today() - pd.Timedelta(days=i)).isoformat()
        if check in logged:
            streak += 1
        else:
            break
    return streak

def already_logged_today(entries):
    return date.today().isoformat() in {e["date"] for e in entries}

# ── Sidebar ───────────────────────────────────────────────────────────────────
entries = load_data()

with st.sidebar:
    st.markdown("## 🌸 MoodBloom")
    st.markdown("*Your personal wellness journal*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📝 Log Today", "📊 My Trends", "🔍 Insights", "📅 History"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        f"<div class='insight-box'>{random.choice(AFFIRMATIONS)}</div>",
        unsafe_allow_html=True,
    )
    streak = calc_streak(entries)
    st.metric("🔥 Streak", f"{streak} day{'s' if streak != 1 else ''}")
    st.metric("📓 Total Entries", len(entries))

# ── Empty-state helper ────────────────────────────────────────────────────────
def show_empty(icon, title, msg):
    st.markdown(
        f"<div class='empty-state'><div class='icon'>{icon}</div>"
        f"<h3>{title}</h3><p>{msg}</p></div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Log Today
# ══════════════════════════════════════════════════════════════════════════════
if "📝 Log Today" in page:
    st.markdown("# 📝 How are you today?")
    st.markdown("A quick 2-minute check-in. Everything here comes from *you*.")

    if already_logged_today(entries):
        st.warning("✅ Already logged today! Come back tomorrow, or visit **History** to review.")
    else:
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown("### 😊 Mood")
            mood_choice = st.select_slider(
                "Mood", options=list(MOOD_EMOJIS.keys()),
                value="🙂 Good", label_visibility="collapsed",
            )

            st.markdown("### ⚡ Energy Level")
            energy = st.slider("Energy", 1, 5, 3, label_visibility="collapsed")
            st.markdown(f"**{ENERGY_LABELS[energy]}**")

            st.markdown("### 😴 Sleep Last Night")
            sleep = st.slider(
                "Sleep", 0.0, 12.0, 7.0, 0.5,
                format="%.1f hrs", label_visibility="collapsed",
            )

            st.markdown("### 😰 Stress Level")
            stress = st.slider(
                "Stress  (1 = calm · 5 = very stressed)", 1, 5, 2,
                label_visibility="collapsed",
            )

        with col2:
            st.markdown("### ✅ Healthy Habits Today")
            selected_habits = [h for h in HABITS if st.checkbox(h)]

            st.markdown("### 📝 Journal Note *(optional)*")
            note = st.text_area(
                "Note", placeholder="What's on your mind today?",
                height=110, label_visibility="collapsed",
            )

        st.markdown("")
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            if st.button("🌸 Save Entry", use_container_width=True):
                save_entry({
                    "date":        date.today().isoformat(),
                    "mood":        MOOD_EMOJIS[mood_choice],
                    "mood_label":  mood_choice,
                    "energy":      energy,
                    "sleep":       sleep,
                    "stress":      stress,
                    "habits":      selected_habits,
                    "habit_count": len(selected_habits),
                    "note":        note,
                })
                st.success(f"✨ Saved! {random.choice(AFFIRMATIONS)}")
                st.balloons()
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: My Trends
# ══════════════════════════════════════════════════════════════════════════════
elif "📊 My Trends" in page:
    st.markdown("# 📊 Your Wellness Trends")

    if len(entries) < 2:
        show_empty("🌱", "Not enough data yet",
                   "Log at least 2 days to see charts.")
    else:
        df = entries_to_df(entries)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("😊 Avg Mood",   f"{df['mood'].mean():.1f} / 5")
        m2.metric("⚡ Avg Energy", f"{df['energy'].mean():.1f} / 5")
        m3.metric("😴 Avg Sleep",  f"{df['sleep'].mean():.1f} hrs")
        m4.metric("✅ Avg Habits", f"{df['habit_count'].mean():.1f} / day")

        # Mood over time
        fig_mood = px.area(df, x="date", y="mood", title="Mood Over Time",
                           color_discrete_sequence=["#b06ab3"], template="plotly_white")
        fig_mood.update_traces(fill="tozeroy", fillcolor="rgba(176,106,179,0.15)", line_width=2.5)
        fig_mood.update_layout(
            font_family="DM Sans", title_font_family="Fraunces",
            yaxis=dict(range=[0, 5.5], tickvals=[1,2,3,4,5],
                       ticktext=["😞 1","😕 2","😐 3","🙂 4","😄 5"]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_mood, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_sc = px.scatter(df, x="sleep", y="mood", title="Sleep vs Mood",
                                color="energy", color_continuous_scale=["#4568dc","#b06ab3"],
                                size="habit_count", size_max=18,
                                hover_data=["date","mood_label"], template="plotly_white")
            fig_sc.update_layout(
                font_family="DM Sans", title_font_family="Fraunces",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0,5.5]), margin=dict(t=40,b=20),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with col_b:
            fig_es = go.Figure()
            fig_es.add_trace(go.Scatter(x=df["date"], y=df["energy"], name="Energy",
                                        mode="lines+markers",
                                        line=dict(color="#4568dc", width=2), marker=dict(size=6)))
            fig_es.add_trace(go.Scatter(x=df["date"], y=df["stress"], name="Stress",
                                        mode="lines+markers",
                                        line=dict(color="#f87171", width=2, dash="dot"),
                                        marker=dict(size=6)))
            fig_es.update_layout(
                title="Energy vs Stress", font_family="DM Sans", title_font_family="Fraunces",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0,5.5]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_es, use_container_width=True)

        # Habit bar
        all_habits = [h for e in entries for h in e.get("habits", [])]
        if all_habits:
            hc = pd.Series(all_habits).value_counts().reset_index()
            hc.columns = ["Habit", "Count"]
            fig_h = px.bar(hc, x="Count", y="Habit", orientation="h",
                           title="Most Practiced Habits",
                           color="Count", color_continuous_scale=["#b06ab3","#4568dc"],
                           template="plotly_white")
            fig_h.update_layout(
                font_family="DM Sans", title_font_family="Fraunces",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
                margin=dict(t=40,b=20),
            )
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Start checking habits in your daily log to see them here!")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Insights
# ══════════════════════════════════════════════════════════════════════════════
elif "🔍 Insights" in page:
    st.markdown("# 🔍 Personal Insights")
    st.markdown("Auto-generated from *your* logged entries — no external data used.")

    if len(entries) < 3:
        show_empty("🔭", "Need a bit more data",
                   "Log at least 3 days to unlock personalised insights.")
    else:
        df = entries_to_df(entries)
        insights = []

        # Sleep
        good_sleep = df[df["sleep"] >= 7]["mood"].mean()
        bad_sleep  = df[df["sleep"] <  7]["mood"].mean()
        if pd.notna(good_sleep) and pd.notna(bad_sleep):
            diff = good_sleep - bad_sleep
            if diff > 0.3:
                insights.append(f"🛌 **Sleep matters for you!** Mood is **{diff:.1f} pts higher** on 7+ hour nights.")
            else:
                insights.append("🛌 Your mood is fairly consistent regardless of how much you sleep — interesting!")

        # Stress
        hi_s = df[df["stress"] >= 4]["mood"].mean()
        lo_s = df[df["stress"] <= 2]["mood"].mean()
        if pd.notna(hi_s) and pd.notna(lo_s):
            insights.append(f"😰 High-stress days average **{hi_s:.1f}/5** mood vs **{lo_s:.1f}/5** on calm days.")

        # Habits
        active   = df[df["habit_count"] >= 3]["mood"].mean()
        inactive = df[df["habit_count"] <  3]["mood"].mean()
        if pd.notna(active) and pd.notna(inactive) and active > inactive:
            insights.append(f"✅ Days with 3+ habits completed average **{active:.1f}/5** mood vs **{inactive:.1f}/5** otherwise.")

        # Best day
        best = df.loc[df["mood"].idxmax()]
        insights.append(
            f"🏆 Your **best mood day** was **{best['date'].strftime('%B %d')}** "
            f"— {best['mood_label']} with {best['sleep']}h sleep."
        )

        # Trend
        if len(df) >= 5:
            recent  = df.tail(3)["mood"].mean()
            earlier = df.iloc[:-3]["mood"].mean()
            if recent > earlier + 0.2:
                insights.append("📈 **Great news!** Your mood has been trending **upward** recently! 🌟")
            elif recent < earlier - 0.2:
                insights.append("📉 Your mood has dipped lately. Try adding one small habit tomorrow 💪")
            else:
                insights.append("➡️ Your mood has been **steady** recently. Consistency is underrated!")

        for ins in insights:
            st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)

        # Donut
        mood_dist = df["mood_label"].value_counts().reset_index()
        mood_dist.columns = ["Mood", "Count"]
        fig_d = px.pie(mood_dist, values="Count", names="Mood",
                       title="Your Mood Distribution", hole=0.55,
                       color_discrete_sequence=["#b06ab3","#4568dc","#7dd3fc","#f9a8d4","#86efac"])
        fig_d.update_layout(font_family="DM Sans", title_font_family="Fraunces",
                            paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=40))
        st.plotly_chart(fig_d, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: History
# ══════════════════════════════════════════════════════════════════════════════
elif "📅 History" in page:
    st.markdown("# 📅 Entry History")

    if not entries:
        show_empty("📓", "No entries yet",
                   "Head to 'Log Today' and make your first entry!")
    else:
        df = entries_to_df(entries)
        df_show = df[["date","mood_label","energy","sleep","stress","habit_count","note"]].copy()
        df_show["date"] = df_show["date"].dt.strftime("%b %d, %Y")
        df_show.columns = ["Date","Mood","Energy","Sleep (h)","Stress","Habits ✅","Note"]
        df_show = df_show.iloc[::-1].reset_index(drop=True)

        st.dataframe(
            df_show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Mood":      st.column_config.TextColumn(width="medium"),
                "Note":      st.column_config.TextColumn(width="large"),
                "Energy":    st.column_config.ProgressColumn(min_value=0, max_value=5, format="%d ⚡"),
                "Sleep (h)": st.column_config.NumberColumn(format="%.1f h"),
                "Stress":    st.column_config.ProgressColumn(min_value=0, max_value=5, format="%d"),
            },
        )

        st.markdown("---")
        st.markdown("#### ⚠️ Danger Zone")
        if st.button("🗑️ Clear All My Data"):
            if st.session_state.get("confirm_clear"):
                clear_all_data()
                st.session_state["confirm_clear"] = False
                st.success("All entries deleted.")
                st.rerun()
            else:
                st.session_state["confirm_clear"] = True
                st.warning("Click again to permanently delete all entries.")
