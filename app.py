import streamlit as st
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import io
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from apscheduler.schedulers.background import BackgroundScheduler
import time
import threading
from openai import OpenAI
from twilio.rest import Client

# ===============================
# 🎓 App Title
# ===============================
st.markdown(
    """
    <h3 style='text-align: center; font-size: 4vw;'>
        🎓 AI-based Drop-out Prediction & Counseling System
    </h3>
    """,
    unsafe_allow_html=True
)

# ===============================
# Sidebar Upload Section
# ===============================
st.sidebar.header("📂 Upload Student Data")
attendance_file = st.sidebar.file_uploader("Upload Attendance CSV", type=["csv"])
tests_file = st.sidebar.file_uploader("Upload Test Scores CSV", type=["csv"])
fees_file = st.sidebar.file_uploader("Upload Fees CSV", type=["csv"])
mentor_email = st.sidebar.text_input("Enter Mentor Email")

# ===============================
# Data Processing Function
# ===============================
def process_data(attendance, tests, fees, model):
    df = attendance.merge(tests, on="StudentID", how="outer")
    df = df.merge(fees, on="StudentID", how="outer")

    df = df.groupby(["StudentID", "Name", "Guardian phone no"], as_index=False).agg({
        "Attendance": "mean",
        "TestScore": "mean",
        "PendingMonths": "max"
    })

    df.rename(columns={"TestScore": "AvgScore"}, inplace=True)

    # Rule-based risk points
    df["RiskPoints"] = 0
    df.loc[df["Attendance"] < 75, "RiskPoints"] += 1
    df.loc[df["AvgScore"] < 40, "RiskPoints"] += 1
    df.loc[df["PendingMonths"] > 1, "RiskPoints"] += 1

    def risk_level(points):
        if points == 0: return "Safe"
        elif points == 1: return "Warning"
        else: return "Critical"

    df["RiskLevel"] = df["RiskPoints"].apply(risk_level)

    # AI prediction
    df["AI_Dropout_Prob"] = model.predict_proba(
        df[["Attendance", "AvgScore", "PendingMonths"]]
    )[:, 1] * 100
    df["AI_Dropout_Prob"] = df["AI_Dropout_Prob"].round(2)

    return df

# ===============================
# Email Function
# ===============================
def send_email(report_df, recipient):
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    sender_email = "dropoutaishield@gmail.com"
    sender_pass = "lhuk ssov koog jqyi"  # App password

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient
    msg["Subject"] = "Weekly Student Risk Report"

    body = """Hello Mentor,

Attached is the latest student risk analysis report with AI predictions.

Regards,  
Drop-out Prediction System"""
    msg.attach(MIMEText(body, "plain"))

    part = MIMEBase("application", "octet-stream")
    part.set_payload(csv_data.encode("utf-8"))
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment; filename=risk_report.csv")
    msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, sender_pass)
    server.send_message(msg)
    server.quit()

# ===============================
# Twilio SMS Function
# ===============================
def send_sms_alert(to_number, message):
    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    client = Client(account_sid, auth_token)
    from_number = st.secrets["TWILIO_PHONE_NUMBER"]

    client.messages.create(
        body=message,
        from_=from_number,
        to=to_number
    )

# ===============================
# Train AI Model (Demo Data)
# ===============================
train_data = pd.DataFrame({
    "Attendance": [60, 80, 72, 45, 90, 55, 85],
    "AvgScore": [35, 70, 55, 20, 88, 30, 75],
    "PendingMonths": [2, 0, 1, 3, 0, 2, 0],
    "DroppedOut": [1, 0, 0, 1, 0, 1, 0]
})

X = train_data[["Attendance", "AvgScore", "PendingMonths"]]
y = train_data["DroppedOut"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test)) * 100

# ===============================
# Show Risk Analysis if Data Uploaded
# ===============================
if attendance_file and tests_file and fees_file:
    attendance = pd.read_csv(attendance_file)
    tests = pd.read_csv(tests_file)
    fees = pd.read_csv(fees_file)

    df = process_data(attendance, tests, fees, model)

    st.subheader("📊 Student Risk Analysis (Rule-based + AI)")
    st.dataframe(df[["StudentID", "Name", "Attendance", "AvgScore",
                     "PendingMonths", "RiskLevel", "AI_Dropout_Prob", "Guardian phone no"]])

    st.subheader("🤖 AI Model Performance")
    st.write(f"Model Accuracy on Test Data: {accuracy:.2f}%")

    # Mentor Email
    if mentor_email:
        if st.button("📧 Send Report to Mentor"):
            send_email(df, mentor_email)
            st.success(f"✅ Report sent successfully to {mentor_email}")

        # Weekly Auto-Email
        def start_scheduler():
            scheduler = BackgroundScheduler()
            scheduler.add_job(lambda: send_email(df, mentor_email), "cron", day_of_week="mon", hour=9, minute=0)
            scheduler.start()
            while True:
                time.sleep(1)

        if st.button("⏰ Enable Weekly Auto-Email"):
            thread = threading.Thread(target=start_scheduler, daemon=True)
            thread.start()
            st.success(f"✅ Auto-email scheduled every Monday 9 AM to {mentor_email}")

    # Guardian SMS Alerts
    if st.button("📲 Send SMS Alerts to Guardians"):
        risky_students = df[df["RiskLevel"] == "Critical"]
        if not risky_students.empty:
            for _, row in risky_students.iterrows():
                alert_msg = f"⚠️ Alert for {row['Name']}: Your child is at CRITICAL dropout risk."
                try:
                    send_sms_alert(row["Guardian phone no"], alert_msg)
                except Exception as e:
                    st.error(f"❌ Failed to send SMS to {row['Guardian phone no']}: {e}")
            st.success("✅ SMS alerts sent to all critical students' guardians.")
        else:
            st.info("No critical students to alert.")

# ===============================
# AI Counselor Chatbot
# ===============================
st.sidebar.markdown("---")
if st.sidebar.checkbox("💬 Talk to AI Counselor"):
    st.subheader("🧑‍🏫 AI Counseling Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    def send_message():
        user_input = st.session_state.chat_input.strip()
        if user_input == "":
            return
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a kind, supportive counselor for students."}] +
                     [{"role": "user", "content": msg} for msg in st.session_state.chat_history] +
                     [{"role": "user", "content": user_input}]
        )
        bot_reply = response.choices[0].message.content
        st.session_state.chat_history.append(user_input)
        st.session_state.chat_history.append(bot_reply)
        st.session_state.chat_input = ""

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f"*You:* {msg}")
        else:
            st.markdown(f"*AI Counselor:* {msg}")

    st.text_input("Type your message...", key="chat_input", on_change=send_message)