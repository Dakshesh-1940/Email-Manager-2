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

DEFAULT_DB_URL = st.secrets.get("DATABASE_URL", "")
DEFAULT_IMAP = st.secrets.get("IMAP_SERVER", "imap.gmail.com")
DEFAULT_USER = st.secrets.get("EMAIL_USER", "")
DEFAULT_PASS = st.secrets.get("EMAIL_PASS", "")
DEFAULT_GEMINI_KEY = st.secrets.get("GOOGLE_API_KEY", "")

# -----------------------------------------------------------------------------
# Helper Functions: Database Operations
# -----------------------------------------------------------------------------
def get_db_connection(db_url):
    if not db_url:
        return None
    return psycopg2.connect(db_url)


def save_todo_to_db(db_url, email_id, title, category, importance, deadline, details, summary):
    try:
        conn = get_db_connection(db_url)
        if not conn:
            return
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM todos WHERE email_id = %s", (email_id,))
            if cur.fetchone() is None:
                # Dynamically insert summary if supported, or fallback safely
                try:
                    cur.execute(
                        """
                        INSERT INTO todos (email_id, title, category, importance, deadline, details, summary, completed)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE)
                        """,
                        (email_id, title, category, importance, deadline, details, summary),
                    )
                except psycopg2.errors.UndefinedColumn:
                    conn.rollback()
                    cur.execute(
                        """
                        INSERT INTO todos (email_id, title, category, importance, deadline, details, completed)
                        VALUES (%s, %s, %s, %s, %s, %s, FALSE)
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
            # Try selecting with summary first
            try:
                cur.execute(
                    """
                    SELECT id, title, category, importance, deadline, details, completed, summary 
                    FROM todos 
                    ORDER BY id DESC
                    """
                )
                rows = cur.fetchall()
            except psycopg2.errors.UndefinedColumn:
                # Fallback if summary column has not been added yet
                conn.rollback()
                cur.execute(
                    """
                    SELECT id, title, category, importance, deadline, details, completed, 'No summary available' as summary 
                    FROM todos 
                    ORDER BY id DESC
                    """
                )
                rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Database fetch error: {e}")
        return []


def toggle_todo_in_db(db_url, todo_id, status):
    conn = None
    try:
        conn = get_db_connection(db_url)
        if not conn:
            st.error("No DB connection available.")
            return
        
        with conn.cursor() as cur:
            bool_status = True if status else False
            cur.execute(
                "UPDATE todos SET completed = %s WHERE id = %s",
                (bool_status, todo_id),
            )
            conn.commit()
    except Exception as e:
        st.error(f"Failed to update task state in Database: {e}")
    finally:
        if conn:
            conn.close()


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
# Helper Functions: AI Classification & Summarization
# -----------------------------------------------------------------------------
def analyze_emails_with_ai(emails, api_key):
    try:
        client = genai.Client(api_key=api_key.strip())

        system_prompt = """
        You are an intelligent executive assistant for a university student.
        Your task is to analyze a list of college emails and return a structured JSON response.

        For each email:
        1. Classify its importance level and category.
        2. Generate a concise 3 to 4 sentence summary covering key points and explicit instructions.
        3. Extract explicit actionable tasks (To-Dos) and all important dates or deadlines.

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
              "summary": "3 to 4 sentence clear summary of the email content.",
              "has_actionable_task": true/false,
              "task": {
                "title": "Clear action phrase",
                "deadline": "Extracted deadline/important dates or 'Not specified'",
                "link_or_details": "Relevant link or next step details"
              }
            }
          ]
        }
        """

        user_content = json.dumps(emails, indent=2)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
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
st.caption("Automated classification, 3-4 line summarization, deadline extraction, and database tracking.")

imap_server = DEFAULT_IMAP
email_user = DEFAULT_USER
email_pass = DEFAULT_PASS
google_key = DEFAULT_GEMINI_KEY
db_url = DEFAULT_DB_URL

st.sidebar.header("⚙️ Settings")
fetch_limit = st.sidebar.slider("Number of emails to fetch", 5, 25, 10)

# Fetching Actions
if st.button("📥 Fetch & Parse Emails", type="primary"):
    with st.spinner("Connecting to mail server..."):
        emails, err = fetch_emails(
            imap_server, email_user, email_pass, limit=fetch_limit
        )

    if err:
        st.error(f"Failed to fetch emails: {err}")
    elif not emails:
        st.warning("No emails retrieved.")
    else:
        with st.spinner("Analyzing & summarizing emails with Gemini AI..."):
            analysis = analyze_emails_with_ai(emails, google_key)
            if analysis and "analyzed_emails" in analysis:
                extracted = analysis.get("analyzed_emails", [])
                for item in extracted:
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
                            item.get("summary", "No summary generated."),
                        )
                st.success("New tasks saved to database successfully!")
                st.rerun()

# --- DISPLAY DATABASE PERSISTED TASKS ---
if db_url:
    todos = load_todos_from_db(db_url)
    pending = [t for t in todos if not t[6]]
    completed = [t for t in todos if t[6]]

    col1, col2, col3 = st.columns(3)
    col1.metric("Pending Tasks", len(pending))
    col2.metric("High Priority", len([t for t in pending if t[3] == "High"]))
    col3.metric("Completed Tasks", len(completed))

    st.markdown("---")
    tab1, tab2 = st.tabs(["📋 Pending Tasks", "✅ Completed Tasks"])

    with tab1:
        if not pending:
            st.info("No pending tasks!")
        else:
            for t in pending:
                t_id, title, category, importance, deadline, details, comp, summary = t
                badge = (
                    "🔴 High"
                    if importance == "High"
                    else ("🟡 Medium" if importance == "Medium" else "🟢 Low")
                )

                c1, c2 = st.columns([0.05, 0.95])
                with c1:
                    if st.checkbox("", key=f"t_pend_{t_id}", value=False):
                        toggle_todo_in_db(db_url, t_id, True)
                        st.rerun()
                with c2:
                    st.markdown(
                        f"**{title}** &nbsp; `{badge}` &nbsp; `📁 {category}`"
                    )
                    st.markdown(f"🗓️ **Deadline / Key Dates:** `{deadline}`")
                    st.caption(f"📝 **Summary:** {summary or 'N/A'}")
                    if details:
                        st.caption(f"ℹ️ **Next Steps / Links:** {details}")
                    st.markdown("---")

    with tab2:
        if not completed:
            st.info("No completed tasks yet.")
        else:
            for t in completed:
                t_id, title, category, importance, deadline, details, comp, summary = t
                c1, c2 = st.columns([0.05, 0.95])
                with c1:
                    if not st.checkbox("", key=f"t_comp_{t_id}", value=True):
                        toggle_todo_in_db(db_url, t_id, False)
                        st.rerun()
                with c2:
                    st.markdown(f"~~{title}~~ &nbsp; `📁 {category}`")
                    st.caption(f"🗓️ **Deadline:** {deadline} | 📝 **Summary:** {summary or 'N/A'}")
