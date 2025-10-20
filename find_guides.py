import pandas as pd
from datetime import datetime, timezone
import hextoolkit as htk
import requests
import time
import io
import re
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import base64
import textwrap
import json
from typing import Set, Dict, List

# ============================================================================
# GENERAL CONFIGURATION
# ============================================================================

GEMINI_MODEL_EMBED = "gemini-embedding-001"
SIM_THRESHOLD = 0.90
CHUNK_SIZE = 25
DRY_RUN = False

VERTEX_CONFIG = {
    'project_id': "supabase-etl-prod-eu",
    'location': "europe-west1",
    'generate_model': "gemini-2.5-flash",
    'embed_model': "text-embedding-004"
}

# ============================================================================
# PROMPTS CONFIGURATION
# ============================================================================

PROMPTS = {
    #this prompt is used to clean up the support conversation for more efficient LLM analysis
    "filter_conversation": (
        "Clean this support thread. Keep only essential info: errors, logs, code, "
        "config, client versions, root cause, and resolution. Remove greetings, signatures, "
        "unsubscribe text, quoted forwards (lines starting with '>'), social links, PII and "
        "repetitive boilerplate. Preserve timestamps, sender labels, and separators. "
        "Do not summarize, only delete noise and potential PII.\n\n"
        "{conversation_text}"
    ),
    
    #this prompt is used to make sure that no important details were missed in the cleaned up conversation vs the original support conversation
    "improve_conversation": (
        "Compare ORIGINAL and FILTERED support conversations.\n"
        "- If FILTERED is already clear and complete, return FILTERED exactly.\n"
        "- Otherwise return an improved version that has all the technical substance required to depict the problem and solution.\n\n"
        "- Promise to only return the support conversation that is most correct. Add nothing else.\n\n"
        "ORIGINAL:\n{original}\n\n"
        "FILTERED:\n{filtered}\n"
    ),
    
    #this prompt is used to create the actual troubleshooting guide
    "generate_troubleshooting": (
        "You are a technical writer creating a Supabase troubleshooting guide. Write a concise, factual guide based on this support conversation. "
        "Focus on the confirmed solution in the conversation's resolution (typically the last few messages) that users can implement themselves\n\n"
        
        "GUIDELINES:\n"
        "- Promise to be concise and factual - no conversational fluff, greetings, or reassurances\n"
        "- Start with the specific error message or symptom users will see\n"
        "- Provide only actionable solutions users can do themselves\n"
        "- Use technical, matter-of-fact language - this is documentation, not chat\n"
        "- Avoid speculation about multiple causes unless clearly mapped to specific errors\n"
        "- Don't use phrases like 'reach out to support' as the primary solution\n"
        "- Keep the entire body under 200 words\n\n"
                
        "SUPPORT CONVERSATION:\n{conversation_text}\n\n---\n"
        
        "INITIAL_EXPRESSED_PROBLEM: <one-sentence description of the customer's initial problem in first-person>\n"
        "PROBLEM_EXPRESSED_AS_A_QUESTION: <customer's problem turned into a question>\n"
        "INSIGHTS: <one-sentence technical summary of root cause and solution>\n"
        "TROUBLESHOOTING_GUIDE_TITLE: <Question starting with Why/How/What describing the specific error or symptom>\n"
    ),
    
    #this prompt evaluates if the guide is truly self-serve
    "evaluate_self_serve": (
        "Evaluate if this is a self-serve troubleshooting guide.\n\n"
        "TROUBLESHOOTING GUIDE TITLE: {title}\n"
        "INSIGHTS: {insights}\n"
        "SUPPORT CONVERSATION: {conversation_text}\n\n"
        "Answer YES if the user can solve the issue with these instructions without contacting support, or NO if they need to contact support.\n"
        "Provide brief reasoning for your answer.\n\n"
        "SELF_SERVE_EVALUATION: <YES or NO, followed by brief reasoning>\n"
    ),
    
    #this prompt compares the generated troubleshooting guide with existing, good troubleshooting guides and generates a new version that matches the tone and concision
    "refine_guide": (
        "Compare the generated troubleshooting guide with these good examples and refine it to match their style, tone and concision.\n\n"
        
        "Do NOT introduce any new technical claims, facts, hypotheses, roles, tables, error causes, commands, SQL, or steps that are not already present in the GUIDE TO REFINE. Also, only use relative links.\n\n"
        "- Anonymize all customer-specific details: replace actual table and schema names with generic placeholders (e.g., 'example_table', 'example_schema'), "
        "and replace specific numeric values from the incident (row counts, sizes, XIDs, percentages) with generalized ranges or examples "
        "(e.g., 'millions of rows', 'hundreds of GBs', 'approaching the threshold'). Keep the guide issue-agnostic while preserving technical accuracy\n"

        "GOOD EXAMPLES:\n"
        "---\n"
        "Example 1:\n"
        "Title: An \"invalid response was received from the upstream server\" error when querying auth\n"
        "Body: If you are observing an \"invalid response was received from the upstream server\" error when making requests to Supabase Auth, it could mean that the respective service is down. One way to confirm this is to visit the [logs explorer](/dashboard/project/_/logs/explorer) and look at the auth logs to see if there are any errors with the following lines:\n"
        "\n"
        "- `running db migrations: error executing migrations/20221208132122_backfill_email_last_sign_in_at.up.sql`\n"
        "\n"
        "We're currently investigating an issue where the tables responsible for keeping track of migrations ran by Auth (`auth.schema_migrations`) are not being restored properly, which leads to the service(s) retrying those migrations. In such cases, migrations which are not idempotent will run into issues.\n"
        "\n"
        "We've documented some of the migrations that run into this issue and their corresponding fix here:\n"
        "\n"
        "### Auth: `operator does not exist: uuid = text`\n"
        "\n"
        "Temporary fix: Run `insert into auth.schema_migrations values ('20221208132122');` via the [SQL editor](/dashboard/project/_/sql/new) to fix the issue.\n"
        "\n"
        "If the migration error you're seeing looks different, reach out to [supabase.help](https://supabase.help/) for assistance.\n"
        "---\n\n"
        "---\n"
        "Example 2:\n"
        "Title: Auth error: {{401: invalid claim: missing sub}}\n"
        "Body: The missing sub claim error is returned when `supabase.auth.getUser()` is called with an invalid JWT in the session or when the user attempts to register/sign in but hasn't completed the sign in when the `getUser` call is made.\n"
        "\n"
        "A common pitfall is inadvertently using a Supabase API key (such as the anon or service_role keys) instead of the Supabase Auth access token.\n"
        "\n"
        "**Why Does This Happen?**\n"
        "- The Supabase API keys are designed for different purposes and, while they are recognized by the Supabase Auth system, they do not carry the sub claim. The sub claim is essential as it encodes the user ID, which is a mandatory field for authentication processes. This mistake leads to the \"missing sub claim\" error because the system expects a token that contains user identification information.\n"
        "\n"
        "**How to Avoid This Issue:**\n"
        "- Ensure that the token being passed to `supabase.auth.getUser()` is, indeed, an Auth access token and not one of the API keys.\n"
        "- Are you creating the client on a per-request basis or are you creating a global client to be shared? If you're creating the client on a per-request basis, then you need to pass the session with the user's JWT from the client to the server somehow. This can be done by sending the user's JWT in a header like an `Authorization: Bearer <user_jwt>`. You can then get this header and call `supabase.auth.getUser(user_jwt)` with the user's JWT.\n"
        "- Examine how the Supabase client is being initialized, especially in server-side scenarios.\n"
        "---\n\n"

        "GUIDE TO REFINE:\n"
        "Title: {title}\n"
        "Body: {body}\n\n"
        
        "REFINED_TITLE: <Ensure the title is specific like in the examples>\n"
        "REFINED_BODY: <Match the example styles in tone, structure and concision.>"
    ),


}

# Common English words that might also be names - don't replace these
COMMON_WORDS_BLACKLIST = {
    'will', 'may', 'mark', 'grace', 'hope', 'grant', 'bill', 'frank',
    'rose', 'joy', 'faith', 'art', 'max', 'sky', 'summer', 'winter',
    'spring', 'autumn', 'dawn', 'april', 'june', 'august', 'march',
    'page', 'link', 'admin', 'user', 'test', 'demo', 'example'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def call_gemini_with_retries(gemini_api_key, prompt, max_retries=3):
    """Call Gemini Vertex AI REST API with retry logic for transient failures"""
    
    url = f"https://{VERTEX_CONFIG['location']}-aiplatform.googleapis.com/v1/projects/{VERTEX_CONFIG['project_id']}/locations/{VERTEX_CONFIG['location']}/publishers/google/models/{VERTEX_CONFIG['generate_model']}:generateContent"
    
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }
    
    params = {"key": gemini_api_key}
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, params=params, json=payload, headers=headers)
            response.raise_for_status()
            
            response_json = response.json()
            
            if 'candidates' in response_json and len(response_json['candidates']) > 0:
                candidate = response_json['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    text = candidate['content']['parts'][0]['text']
                    class Response:
                        def __init__(self, text):
                            self.text = text
                    return Response(text)
            
            print(f"Attempt {attempt + 1}: Empty response from Gemini")
            
        except requests.exceptions.HTTPError as e:
            error_str = str(e).lower()
            if any(code in error_str for code in ['503', '500', '429', 'connection', 'timeout']):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Attempt {attempt + 1} failed with retryable error: {e}")
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
            print(f"API call failed: {e}")
            return None
        except Exception as e:
            print(f"API call failed: {e}")
            return None
    
    return None

def call_gemini_embed_with_retries(gemini_api_key, texts, max_retries=3):
    """Call Gemini Vertex AI embedding REST API with retry logic for transient failures"""
    
    url = f"https://{VERTEX_CONFIG['location']}-aiplatform.googleapis.com/v1/projects/{VERTEX_CONFIG['project_id']}/locations/{VERTEX_CONFIG['location']}/publishers/google/models/{VERTEX_CONFIG['embed_model']}:predict"
    
    instances = [{"content": text} for text in texts]
    
    payload = {
        "instances": instances
    }
    
    params = {"key": gemini_api_key}
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, params=params, json=payload, headers=headers)
            response.raise_for_status()
            
            response_json = response.json()
            
            if 'predictions' in response_json:
                class Embedding:
                    def __init__(self, values):
                        self.values = values
                
                class EmbedResult:
                    def __init__(self, predictions):
                        self.embeddings = [Embedding(pred['embeddings']['values']) for pred in predictions]
                
                return EmbedResult(response_json['predictions'])
            
            print(f"Attempt {attempt + 1}: Empty embedding response from Gemini")
            
        except requests.exceptions.HTTPError as e:
            error_str = str(e).lower()
            if any(code in error_str for code in ['503', '500', '429', 'connection', 'timeout']):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[EMBEDDINGS API] Attempt {attempt + 1} failed with retryable error: {e}")
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
            print(f"Embedding API call failed: {e}")
            return None
        except Exception as e:
            print(f"Embedding API call failed: {e}")
            return None
    
    return None

def extract_names_from_emails(email_list: str) -> Dict[str, Set[str]]:
    """Extract names from email addresses and categorize them"""
    support_names = set()
    customer_names = set()
    
    email_list = str(email_list) if email_list is not None else ''
    if not email_list or email_list == 'None':
        return {'support': support_names, 'customer': customer_names}
    
    email_pattern = r'[\w.+-]+@[\w-]+\.[\w.-]+'
    emails = re.findall(email_pattern, str(email_list), re.IGNORECASE)
    
    for email in emails:
        local_part = email.split('@')[0].lower()
        domain = email.split('@')[1].lower() if '@' in email else ''
        
        name_parts = re.split(r'[._\-+]', local_part)
        
        valid_names = {
            name for name in name_parts 
            if len(name) >= 3 and not name.isdigit() and name not in COMMON_WORDS_BLACKLIST
        }
        
        if 'supabase' in domain:
            support_names.update(valid_names)
        else:
            customer_names.update(valid_names)
    
    return {'support': support_names, 'customer': customer_names}

def anonymize_names_in_text(text: str, support_names: Set[str], customer_names: Set[str]) -> str:
    """Replace names in text with anonymized versions"""
    if not text:
        return ""
    
    all_names = [
        (name, '[SUPPORT]') for name in sorted(support_names, key=len, reverse=True)
    ] + [
        (name, '[CUSTOMER_NAME]') for name in sorted(customer_names, key=len, reverse=True)
    ]
    
    for name, replacement in all_names:
        pattern = r'\b' + re.escape(name) + r'\b'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def anonymize_text(text: str, support_names: Set[str] = None, customer_names: Set[str] = None, 
                   org_slug: str = None, project_ref: str = None) -> str:
    """Full anonymization including emails, URLs, names, and project/org references"""
    if not text:
        return ""
    
    if support_names is None or customer_names is None:
        names_from_text = extract_names_from_emails(text)
        support_names = support_names or set()
        customer_names = customer_names or set()
        support_names.update(names_from_text['support'])
        customer_names.update(names_from_text['customer'])
    
    if org_slug:
        text = text.replace(org_slug, '[ORGANIZATION_SLUG]')
    if project_ref:
        text = text.replace(project_ref, '[PROJECT_SLUG]')
    
    supabase_links = {}
    def _protect_supabase(match):
        key = f"__SUPABASE_PROTECTED_{len(supabase_links)}__"
        supabase_links[key] = match.group(0)
        return key

    text = re.sub(r'https?://[^\s)>\]"}]+supabase\.com[^\s)>\]"}]*', _protect_supabase, text)
    
    text = anonymize_names_in_text(text, support_names, customer_names)
    
    patterns = [
        (r'[\w.+-]+@[\w-]+\.[\w.-]+', '[EMAIL]'),
        (r'https?://[^\s)>\]"}]+', '[URL]'),
        (r'@\w+', '[HANDLE]'),
        (r'\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b', '[IP]'),
        (r'\b(?:[A-F0-9]{1,4}:){2,7}[A-F0-9]{1,4}\b', '[IP]'),
        (r'\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b', '[UUID]'),
        (r'\b[A-Za-z0-9-_]{10,}\.[A-Za-z0-9-_]{10,}\.[A-Za-z0-9-_]{10,}\b', '[JWT]'),
        (r'\bsk-[A-Za-z0-9]{16,}\b', '[API_KEY]'),
        (r'\bAKIA[0-9A-Z]{16}\b', '[AWS_KEY]'),
        (r'(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?){1,3}\d{3,4})(?=\D|$)', '[PHONE]'),
    ]
    
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    
    for key, link in supabase_links.items():
        text = text.replace(key, link)
    
    return text

def get_support_conversation(conversation_id):
    """Get and format a specific support conversation by conversation_id (fully anonymized)"""
    print(f"    Searching for conversation ID: {conversation_id}")
    
    query = f"""
    SELECT 
        ticket_id,
        conversation_id,
        conversation_subject,
        emitted_timestamp,
        message_text,
        comment_body,
        comment_author_username,
        message_is_inbound,
        message_recipient_handles,
        organization_slug,
        project_reference,
        LAST_VALUE(
            CASE 
                WHEN NOT message_is_inbound 
                     AND comment_author_username NOT IN ('front_rule__38696286','front_api_jwt_s71a')
                THEN comment_author_username 
            END IGNORE NULLS
        ) OVER (ORDER BY emitted_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_support_agent
    FROM `supabase-etl-prod-eu.dbt.refinery_front_messages_final`
    WHERE conversation_id = '{conversation_id}'
    ORDER BY emitted_timestamp ASC
    """

    connection = htk.get_data_connection("supabase-etl-prod-eu")
    df = connection.query(query)
    
    # print(f"    Query returned {len(df)} rows")

    if df.empty:
        return f"No conversation found for ID: {conversation_id}", None, None

    all_support_names = set()
    all_customer_names = set()

    for _, row in df.iterrows():
        handles = str(row.get('message_recipient_handles', ''))
        names_dict = extract_names_from_emails(handles)
        all_support_names.update(names_dict['support'])
        all_customer_names.update(names_dict['customer'])

        content = str(row.get('message_text', '') or row.get('comment_body', '') or '')
        names_from_content = extract_names_from_emails(content)
        all_support_names.update(names_from_content['support'])
        all_customer_names.update(names_from_content['customer'])

    result = []
    org_slug = df.iloc[0].get('organization_slug', '')
    project_ref = df.iloc[0].get('project_reference', '')
    subject = anonymize_text(
        str(df.iloc[0]['conversation_subject'] or ""),
        all_support_names, all_customer_names,
        org_slug, project_ref
    )

    result.append("=== SUPPORT THREAD ===")
    result.append(f"Subject: {subject}")
    result.append("Organization: [ORGANIZATION_SLUG]" if org_slug else "Organization: N/A")
    result.append("Project: [PROJECT_SLUG]" if project_ref else "Project: N/A")
    result.append("=" * 60)

    for _, row in df.iterrows():
        if (row['comment_author_username'] in ['front_rule__38696286','front_api_jwt_s71a'] or 
            (not row['message_is_inbound'] and pd.isna(row['comment_author_username']))):
            continue

        sender = "Customer" if row['message_is_inbound'] else "Support"
        content_raw = row['message_text'] or row['comment_body'] or ""
        content = anonymize_text(
            str(content_raw), all_support_names, all_customer_names,
            org_slug, project_ref
        )
        timestamp = pd.to_datetime(row['emitted_timestamp']).strftime('%Y-%m-%d %H:%M')

        result.append(f"\n{timestamp} | {sender}")
        result.append("-" * 60)
        result.append(textwrap.fill(content, width=60))
        result.append("-" * 60)

    last_support_agent = df.iloc[0].get('last_support_agent')
    ticket_id_val = df.iloc[0].get('ticket_id')

    return '\n'.join(result), last_support_agent, ticket_id_val

def filter_support_conversation(conversation_text, gemini_api_key):
    """Filter support conversation with Gemini: remove boilerplate, signatures, and duplicates"""
    if not conversation_text or "No conversation found" in conversation_text:
        return "No valid conversation found"
    
    prompt = PROMPTS["filter_conversation"].format(conversation_text=conversation_text)
    
    response = call_gemini_with_retries(gemini_api_key, prompt)
    
    if not response:
        return conversation_text  # Return original if API fails
    
    return response.text.strip()

def improve_support_conversation(original, filtered, gemini_api_key):
    """Compare original and filtered conversations to produce final improved version"""
    if "Error:" in filtered or "No valid conversation" in filtered:
        return filtered
    
    prompt = PROMPTS["improve_conversation"].format(original=original, filtered=filtered)
    
    response = call_gemini_with_retries(gemini_api_key, prompt)
    
    if not response:
        return filtered
    
    improved = response.text.strip()
    
    # Check if different from filtered
    if improved == filtered.strip():
        print("  → Kept filtered version")
    else:
        print("  → Created improved version")
    
    return improved

def generate_troubleshooting_content(conversation_text: str, gemini_api_key):
    """Generate troubleshooting guide content from support conversation using Gemini."""
    if not conversation_text or "No conversation found" in conversation_text:
        return None
    
    improved_prompt = PROMPTS["generate_troubleshooting"].format(conversation_text=conversation_text)
    
    response = call_gemini_with_retries(gemini_api_key, improved_prompt)
    if not response:
        return None
    
    return parse_improved_guide_response(response.text)


def parse_improved_guide_response(response_text):
    """Parse the improved troubleshooting guide response sections"""
    sections = {}
    
    initial_match = re.search(r'INITIAL_EXPRESSED_PROBLEM:\s*(.*?)(?=PROBLEM_EXPRESSED_AS_A_QUESTION:|$)', response_text, re.DOTALL | re.IGNORECASE)
    problem_question_match = re.search(r'PROBLEM_EXPRESSED_AS_A_QUESTION:\s*(.*?)(?=INSIGHTS:|$)', response_text, re.DOTALL | re.IGNORECASE)
    insights_match = re.search(r'INSIGHTS:\s*(.*?)(?=TROUBLESHOOTING_GUIDE_TITLE:|$)', response_text, re.DOTALL | re.IGNORECASE)
    title_match = re.search(r'TROUBLESHOOTING_GUIDE_TITLE:\s*(.*?)$', response_text, re.DOTALL | re.IGNORECASE)
    
    sections['initial_problem'] = initial_match.group(1).strip() if initial_match else ""
    sections['problem_question'] = problem_question_match.group(1).strip() if problem_question_match else ""
    sections['insights'] = insights_match.group(1).strip() if insights_match else ""
    sections['title'] = title_match.group(1).strip() if title_match else ""
    
    return sections

def evaluate_self_serve(title: str, insights: str, conversation_text: str, gemini_api_key):
    """Evaluate if the troubleshooting guide is self-serve."""
    if not title or not insights:
        return "Unable to evaluate - missing title or insights"
    
    prompt = PROMPTS["evaluate_self_serve"].format(
        title=title,
        insights=insights,
        conversation_text=conversation_text
    )
    
    response = call_gemini_with_retries(gemini_api_key, prompt)
    if not response:
        return "Evaluation failed - API error"
    
    # Parse the self-serve evaluation from the response
    self_serve_match = re.search(r'SELF_SERVE_EVALUATION:\s*(.*?)$', response.text, re.DOTALL | re.IGNORECASE)
    if self_serve_match:
        return self_serve_match.group(1).strip()
    
    return response.text.strip()

def refine_guide(title: str, body: str, gemini_api_key):
    """Refine the troubleshooting guide to match good examples' style and tone."""
    if not title or not body:
        return {"refined_title": title, "refined_body": body}
    
    prompt = PROMPTS["refine_guide"].format(title=title, body=body)
    
    response = call_gemini_with_retries(gemini_api_key, prompt)
    if not response:
        return {"refined_title": title, "refined_body": body}
    
    # Parse the refined title and body from the response
    refined_title_match = re.search(r'REFINED_TITLE:\s*(.*?)(?=REFINED_BODY:|$)', response.text, re.DOTALL | re.IGNORECASE)
    refined_body_match = re.search(r'REFINED_BODY:\s*(.*?)$', response.text, re.DOTALL | re.IGNORECASE)
    
    refined_title = refined_title_match.group(1).strip() if refined_title_match else title
    refined_body = refined_body_match.group(1).strip() if refined_body_match else body
    
    return {"refined_title": refined_title, "refined_body": refined_body}

def create_preview_gist(title, body, github_pat):
    """Create a secret GitHub Gist with MDX content for preview."""
    gist_data = {
        "description": f"Preview: {title}",
        "public": False,  # Secret gist
        "files": {
            "preview.md": {
                "content": f"# {title}\n\n{body}"
            }
        }
    }
    
    response = requests.post(
        "https://api.github.com/gists",
        headers={
            "Authorization": f"token {github_pat}",
            "Accept": "application/vnd.github.v3+json"
        },
        json=gist_data
    )
    
    return response.json()['html_url'] if response.status_code == 201 else None

def format_troubleshooting_mdx(sections, error_codes=None):
    """Create improved MDX troubleshooting guide from sections (Supabase docs format)."""
    title = (sections.get('title') or 'Untitled Troubleshooting Guide').strip('<>"').strip()
    body = (sections.get('body') or '').strip()

    content_text = body.lower()
    available_topics = [
        "ai", "auth", "branching", "cli", "database", "functions",
        "platform", "realtime", "self-hosting", "storage", "studio",
        "supavisor", "terraform",
    ]
    
    topics = []
    for topic in available_topics:
        if topic == "ai":
            if re.search(r'\b(ai|artificial intelligence|openai|gpt|llm|embedding|vector)\b', content_text, re.IGNORECASE):
                topics.append(topic)
        else:
            if topic in content_text:
                topics.append(topic)
    
    if not topics:
        topics = ["platform"]

    # Format error codes if provided
    errors_blocks = []
    if error_codes:
        for error in error_codes:
            code = error['code']
            message = error['message']
            
            # Determine if it's an HTTP status code
            if code.isdigit() and len(code) == 3 and code[0] in '45':
                errors_blocks.append(f'[[errors]]\nhttp_status_code = {code}\nmessage = "{message}"\n')
            else:
                errors_blocks.append(f'[[errors]]\ncode = "{code}"\nmessage = "{message}"\n')
    
    errors_section = ("\n" + "\n".join(errors_blocks)) if errors_blocks else ""

    mdx_content = f'''---
title = "{title}"
topics = {_format_topics_array_improved(topics)}
keywords = []{errors_section}
---

{body}'''.rstrip()

    return mdx_content

def _format_topics_array_improved(topics: list[str]) -> str:
    """Format topics into single-line array like in Supabase docs"""
    seen, ordered = set(), []
    for t in topics:
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    return '[ ' + ', '.join(f'"{t}"' for t in ordered) + ' ]'



# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_single_conversation(conversation_id, gemini_api_key):
    """Process one conversation through the simplified pipeline"""
    result = {
        'conversation_id': conversation_id,
        'status': 'failed',
        'error': None,
        'troubleshooting_content': None,
        'ticket_id': None,
        'last_support_agent': None
    }
    
    try:
        print(f"  Extracting conversation...")
        support_conversation, last_support_agent, ticket_id = get_support_conversation(conversation_id)
        
        if not support_conversation or "No conversation found" in support_conversation:
            result['error'] = 'Conversation not found'
            return result
        
        result['ticket_id'] = ticket_id
        result['last_support_agent'] = last_support_agent
        
        print(f"  Filtering with AI...")
        time.sleep(5)
        filtered_conversation = filter_support_conversation(support_conversation, gemini_api_key)
        
        print(f"  Improving conversation...")
        time.sleep(5)
        final_conversation = improve_support_conversation(support_conversation, filtered_conversation, gemini_api_key)
        
        print(f"  Generating troubleshooting content...")
        time.sleep(5)
        troubleshooting_content = generate_troubleshooting_content(final_conversation, gemini_api_key)

        if not troubleshooting_content or not troubleshooting_content.get('title'):
            result['error'] = 'Failed to generate troubleshooting content'
            return result
        
        print(f"  Evaluating if guide is self-serve...")
        time.sleep(5)
        self_serve_evaluation = evaluate_self_serve(
            troubleshooting_content.get('title', ''),
            troubleshooting_content.get('insights', ''),
            final_conversation,
            gemini_api_key
        )
        troubleshooting_content['self_serve_evaluation'] = self_serve_evaluation
        
        # Check if self-serve evaluation contains YES and run refinement
        if 'YES' in self_serve_evaluation.upper():
            print(f"  Refining guide to match good examples...")
            time.sleep(5)
            # Create a basic body from the insights for refinement
            basic_body = troubleshooting_content.get('insights', '')
            refined_guide = refine_guide(
                troubleshooting_content.get('title', ''),
                basic_body,
                gemini_api_key
            )
            troubleshooting_content['refined_title'] = refined_guide.get('refined_title', troubleshooting_content.get('title', ''))
            troubleshooting_content['refined_body'] = refined_guide.get('refined_body', basic_body)
        else:
            print(f"  Skipping refinement - guide is not self-serve")
            troubleshooting_content['refined_title'] = troubleshooting_content.get('title', '')
            troubleshooting_content['refined_body'] = troubleshooting_content.get('insights', '')
        
        # Generate MDX content using refined title and body
        mdx_sections = {
            'title': troubleshooting_content.get('refined_title', ''),
            'body': troubleshooting_content.get('refined_body', '')
        }
        mdx_content = format_troubleshooting_mdx(mdx_sections)
        troubleshooting_content['mdx_content'] = mdx_content
        
        result['troubleshooting_content'] = troubleshooting_content
        
        print("=" * 80)
        print(f"INITIAL_EXPRESSED_PROBLEM: {troubleshooting_content.get('initial_problem', 'N/A')}")
        print(f"PROBLEM_EXPRESSED_AS_A_QUESTION: {troubleshooting_content.get('problem_question', 'N/A')}")
        print(f"INSIGHTS: {troubleshooting_content.get('insights', 'N/A')}")
        print(f"TROUBLESHOOTING_GUIDE_TITLE: {troubleshooting_content.get('title', 'N/A')}")
        print(f"SELF_SERVE_EVALUATION: {troubleshooting_content.get('self_serve_evaluation', 'N/A')}")
        print(f"REFINED_TITLE: {troubleshooting_content.get('refined_title', 'N/A')}")
        print(f"REFINED_BODY: {troubleshooting_content.get('refined_body', 'N/A')}")
        print("=" * 80)
        
        result['status'] = 'success'
        
    except Exception as e:
        result['error'] = str(e)
        print(f"  Error: {e}")
    
    return result

def process_conversation_batch(conversation_ids, max_conversations=None, delay_seconds=20):
    """Main orchestrator function"""
    
    # Limit conversations if specified
    conversations_to_process = conversation_ids[:max_conversations] if max_conversations else conversation_ids
    
    print(f"Processing {len(conversations_to_process)} conversations...")
    print("=" * 50)
    
    results = []
    
    for i, conversation_id in enumerate(conversations_to_process, 1):
        print(f"\n[{i}/{len(conversations_to_process)}] Processing {conversation_id}...")
        
        result = process_single_conversation(
            conversation_id, 
            gemini_vertex_api_key
        )
        
        results.append(result)
        
        # Print status
        if result['status'] == 'success':
            print(f"✓ Troubleshooting content generated successfully")
        else:
            print(f"✗ Failed: {result['error']}")
        
        # Delay between conversations (except for last one)
        if i < len(conversations_to_process):
            print(f"Waiting {delay_seconds} seconds before next conversation...")
            time.sleep(delay_seconds)
    
    # Summary
    print("\n" + "="*50)
    print("BATCH PROCESSING SUMMARY")
    print("="*50)
    
    print(f"Total processed: {len(results)}")
    print(f"Successful: {sum(r['status'] == 'success' for r in results)}")
    print(f"Failed: {sum(r['status'] == 'failed' for r in results)}")
    
    return results

def create_results_dataframe(results):
    """Convert results to a pandas DataFrame with key fields"""
    data = []
    
    for result in results:
        row = {
            'conversation_id': result.get('conversation_id', ''),
            'status': result.get('status', ''),
            'ticket_id': result.get('ticket_id', ''),
            'last_support_agent': result.get('last_support_agent', ''),
            'error': result.get('error', ''),
        }
        
        # Extract troubleshooting content fields
        if result.get('troubleshooting_content'):
            content = result['troubleshooting_content']
            row['problem_expressed_as_question'] = content.get('problem_question', '')
            row['initial_expressed_problem'] = content.get('initial_problem', '')
            row['insights'] = content.get('insights', '')
            row['self_serve_evaluation'] = content.get('self_serve_evaluation', '')
            row['title'] = content.get('title', '')
            row['refined_title'] = content.get('refined_title', '')
            row['refined_body'] = content.get('refined_body', '')
            row['mdx_content'] = content.get('mdx_content', '')
        else:
            row['problem_expressed_as_question'] = ''
            row['initial_expressed_problem'] = ''
            row['insights'] = ''
            row['self_serve_evaluation'] = ''
            row['title'] = ''
            row['refined_title'] = ''
            row['refined_body'] = ''
            row['mdx_content'] = ''
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Reorder columns for better readability
    column_order = [
        'conversation_id',
        'status',
        'problem_expressed_as_question',
        'initial_expressed_problem',
        'insights',
        'self_serve_evaluation',
        'title',
        'refined_title',
        'refined_body',
        'mdx_content',
        'ticket_id',
        'last_support_agent',
        'error'
    ]
    
    return df[column_order]

# ============================================================================
# EXECUTION
# ============================================================================

# API Keys - These should be set in your Hex project variables
# gemini_vertex_api_key is available from Hex project variables

# Single conversation ID for testing

print(f"Processing single conversation: {SINGLE_CONVERSATION_ID}")
print("=" * 80)

# Process the single conversation
result = process_single_conversation(
    SINGLE_CONVERSATION_ID, 
    gemini_vertex_api_key
)

# Create GitHub Gist with MDX content for preview
if result['status'] == 'success' and result.get('troubleshooting_content', {}).get('mdx_content'):
    print("\n" + "=" * 80)
    print("Creating preview gist...")
    
    content = result['troubleshooting_content']
    gist_url = create_preview_gist(
        content.get('refined_title', 'Troubleshooting Guide'),
        content.get('mdx_content', ''),
        github_pat
    )
    
    if gist_url:
        print(f"✓ Preview gist created: {gist_url}")
    else:
        print("✗ Failed to create preview gist")
    print("=" * 80)