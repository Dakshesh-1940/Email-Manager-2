import email
from email.header import decode_header
import imaplib
import json
import os
import re
from google import genai
from google.genai import types
import psycopg2
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="College Mail Classifier & Task Extractor",
    page_icon="🎓",
    layout="wide",
)

# Load default secrets with environment variable fallbacks
DEFAULT_DB_URL = os.environ.get(
    "DATABASE_URL", st.secrets.get("DATABASE_URL", "")
)
DEFAULT_IMAP = os.environ.get(
    "IMAP_SERVER", st.secrets.get("IMAP_SERVER", "imap.gmail.com")
)
DEFAULT_USER = os.environ.get(
    "EMAIL_USER", st.secrets.get("EMAIL_USER", "")
)
DEFAULT_PASS = os.environ.get(
    "EMAIL_PASS", st.secrets.get("EMAIL_PASS", "")
)
DEFAULT_GEMINI_KEY = os.environ.get(
    "GOOGLE_API_KEY", st.secrets.get("GOOGLE_API_KEY", "")
)

# -----------------------------------------------------------------------------
# Helper Functions: Database Operations
# -----------------------------------------------------------------------------
def get_db_connection(db_url):
    if not db_url:
        return None
    return psycopg2.connect(db_url)


def save_todo_to_db(db_url, email_id, title, category, importance, deadline, details):
    try:
        conn = get_db_connection(db_url)
        if not conn:
            return
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO todos (email_id, title, category, importance, deadline, details)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (email_id, title, category, importance, deadline, details),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Failed to save task to database: {e}")


def load_todos_from_db(db_url):
    try:
        conn = get_db_connection(db_url)
        if not conn:
            return []
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, category, importance, deadline, details, completed 
                FROM todos 
                ORDER BY id DESC
                """
            )
            rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        st.warning(f"Database connection skipped: {e}")
        return []


def toggle_todo_in_db(db_url, todo_id, status):
    try:
        conn = get_db_connection(db_url)
        if not conn:
            return
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE todos SET completed = %s WHERE id = %s", (status, todo_id)
            )
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Failed to update task: {e}")


# -----------------------------------------------------------------------------
# Helper Functions: Email Fetching
# -----------------------------------------------------------------------------
def decode_str(header_value):
    if not header_value:
        return ""
    decoded_list = decode_header(header_value)
    text = ""
    for content, encoding in decoded_list:
        if isinstance(content, bytes):
            text += content.decode(encoding or "utf-8", errors="ignore")
        else:
            text += str(content)
    return text


def clean_body_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_emails(imap_server, email_user, email_pass, folder="INBOX", limit=10):
    emails = []
    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email_user, email_pass)
        mail.select(folder)

        status, messages = mail.search(None, "UNSEEN")
        email_ids = messages[0].split()

        if not email_ids:
            status, messages = mail.search(None, "ALL")
            email_ids = messages[0].split()

        latest_ids = email_ids[-limit:]

        for e_id in reversed(latest_ids):
            _, msg_data = mail.fetch(e_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_str(msg["Subject"])
                    sender = decode_str(msg["From"])
                    date = decode_str(msg["Date"])

                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(
                                part.get("Content-Disposition")
                            )
                            if (
                                content_type == "text/plain"
                                and "attachment" not in content_disposition
                            ):
                                payload = part.get_payload(decode=True)
                                if payload:
                                    body = payload.decode(
                                        "utf-8", errors="ignore"
                                    )
                                    break
                    else:
                        payload = msg.get_payload(decode=True)
                        if payload:
                            body = payload.decode("utf-8", errors="ignore")

                    emails.append(
                        {
                            "id": e_id.decode("utf-8"),
                            "subject": subject,
                            "sender": sender,
                            "date": date,
                            "body": clean_body_text(body[:2000]),
                        }
                    )

        mail.logout()
        return emails, None
    except Exception as e:
        return [], str(e)


# -----------------------------------------------------------------------------
# Helper Functions: AI Classification
# -----------------------------------------------------------------------------
def analyze_emails_with_ai(emails, api_key):
    try:
        client = genai.Client(api_key=api_key.strip())

        system_prompt = """
        You are an intelligent executive assistant for a university student.
        Your task is to analyze a list of college emails and return a structured JSON response.

        For each email, classify its importance level and category, and extract explicit actionable tasks (To-Dos) if any exist.

        CATEGORIES:
        - Academic (Assignments, Grades, Exam Schedules, Lectures)
        - Opportunity (Hackathons, Internships, Workshops, Clubs)
        - Administrative (Fee Payments, Circulars, Library, Portal Alerts)
        - General/Newsletter (Events, Spam, Non-urgent announcements)

        IMPORTANCE:
        - High (Immediate action/deadline required)
        - Medium (Action required soon)
        - Low (Informational only)

        OUTPUT FORMAT: Return ONLY a valid JSON object matching this schema:
        {
          "analyzed_emails": [
            {
              "id": "email_id",
              "subject": "Subject",
              "category": "Academic|Opportunity|Administrative|General",
              "importance": "High|Medium|Low",
              "summary": "1-sentence summary of the email",
              "has_actionable_task": true/false,
              "task": {
                "title": "Clear action phrase",
                "deadline": "Extracted date/time or 'Not specified'",
                "link_or_details": "Relevant link or brief next step instruction"
              }
            }
          ]
        }
        """

        user_content = json.dumps(emails, indent=2)

        response = client.models.generate_content(
            model="gemini-3.1-flash",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return json.loads(response.text)

    except Exception as e:
        st.error(f"⚠️ Gemini API Error Details: {e}")
        return None


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("🎓 College Email Assistant & Task Tracker")
st.caption("Automated classification, task extraction, and database tracking.")

# Sidebar Settings (Auto-populated from secrets)
st.sidebar.header("🔑 Configuration")

imap_server = st.sidebar.text_input("IMAP Host", value=DEFAULT_IMAP)
email_user = st.sidebar.text_input("College Email", value=DEFAULT_USER)
email_pass = st.sidebar.text_input("App Password", value=DEFAULT_PASS, type="password")
google_key = st.sidebar.text_input("Google API Key", value=DEFAULT_GEMINI_KEY, type="password")
db_url = st.sidebar.text_input("Database URL (Optional)", value=DEFAULT_DB_URL, type="password")

fetch_limit = st.sidebar.slider("Number of emails to fetch", 5, 25, 10)

# Main Execution Trigger
if st.button("📥 Fetch & Analyze Emails", type="primary"):
    if not email_user or not email_pass or not google_key:
        st.error("Please provide your Email ID, App Password, and Google API Key in secrets or sidebar.")
    else:
        with st.spinner("Connecting to mail server & fetching messages..."):
            emails, err = fetch_emails(
                imap_server, email_user, email_pass, limit=fetch_limit
            )

        if err:
            st.error(f"Failed to fetch emails: {err}")
        elif not emails:
            st.warning("No emails retrieved.")
        else:
            st.success(f"Retrieved {len(emails)} emails!")

            with st.spinner("Analyzing contents with Gemini AI..."):
                analysis = analyze_emails_with_ai(emails, google_key)
                if analysis and "analyzed_emails" in analysis:
                    extracted_emails = analysis.get("analyzed_emails", [])
                    st.session_state["analysis_data"] = extracted_emails
                    
                    # Persist extracted actionable tasks into database
                    if db_url:
                        for item in extracted_emails:
                            if item.get("has_actionable_task"):
                                task = item.get("task", {})
                                save_todo_to_db(
                                    db_url,
                                    item.get("id"),
                                    task.get("title", item.get("subject")),
                                    item.get("category", "General"),
                                    item.get("importance", "Medium"),
                                    task.get("deadline", "Not specified"),
                                    task.get("link_or_details", ""),
                                )
                else:
                    st.error("Failed to parse email analysis.")

# --- DISPLAY DATABASE PERSISTED TASKS ---
if db_url:
    db_todos = load_todos_from_db(db_url)
    if db_todos:
        st.markdown("---")
        st.header("🗄️ Saved Tasks (Database History)")
        pending_db = [t for t in db_todos if not t[6]]
        
        for t in pending_db:
            t_id, title, category, importance, deadline, details, comp = t
            badge = "🔴 High" if importance == "High" else ("🟡 Medium" if importance == "Medium" else "🟢 Low")
            c1, c2 = st.columns([0.05, 0.95])
            with c1:
                if st.checkbox("", key=f"db_t_{t_id}", value=False):
                    toggle_todo_in_db(db_url, t_id, True)
                    st.rerun()
            with c2:
                st.markdown(f"**{title}** &nbsp; `{badge}` &nbsp; `📁 {category}`")
                st.caption(f"🗓️ **Deadline:** {deadline} | ℹ️ **Details:** {details}")

# --- DISPLAY CURRENT SESSION RESULTS ---
if "analysis_data" in st.session_state and st.session_state["analysis_data"]:
    data = st.session_state["analysis_data"]
    todo_items = [item for item in data if item.get("has_actionable_task")]

    st.markdown("---")
    st.header("📋 Current Run Extracted Tasks")

    if todo_items:
        for idx, item in enumerate(todo_items):
            task = item["task"]
            imp_badge = "🔴 High" if item["importance"] == "High" else ("🟡 Medium" if item["importance"] == "Medium" else "🟢 Low")
            col1, col2 = st.columns([0.05, 0.95])
            with col1:
                st.checkbox("", key=f"todo_chk_{idx}")
            with col2:
                st.markdown(f"**{task['title']}** &nbsp; `{imp_badge}` &nbsp; `📁 {item['category']}`")
                st.caption(f"🗓️ **Deadline:** {task.get('deadline', 'N/A')} | ℹ️ **Details:** {task.get('link_or_details', 'None')}")
    else:
        st.info("No explicit actionable tasks identified in these emails.")
