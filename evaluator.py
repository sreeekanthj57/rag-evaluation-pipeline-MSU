import re
import csv
import requests
from datetime import datetime, timedelta, timezone
from langfuse import get_client
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
load_dotenv()

langfuse = get_client()
openai_client = OpenAI()

FALLBACK_PATTERNS = (
    # English
    "Hmm, that's not something I can help with",
    "I don't have",
    # Malay
    "Hmm, itu bukan sesuatu yang boleh saya bantu."
    "Saya tiada"
)

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", 1))
TRACE_NAME = os.getenv("TRACE_NAME", "msu evaluator xxxxxxxxxxxxxxxxxxxxxxxxx")


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_retrieved_context(system_prompt: str) -> str | None:
    match = re.search(
        r"## Retrieved Context\n+"
        r"The following chunks.*?\n+"
        r"(.*?)"
        r"\n+---\n+"
        r"## Conversation History",
        system_prompt,
        re.DOTALL,
    )
    return match.group(1).strip() if match else None


def is_fallback(output: str) -> bool:
    return any(pattern in str(output) for pattern in FALLBACK_PATTERNS)


# ── Fetch ──────────────────────────────────────────────────────────────────────

def fetch_traces(limit=50, start_time=None, end_time=None, trace_name=None):
    all_data = []
    page = 1

    while True:
        traces_response = langfuse.api.trace.list(
            limit=limit,
            page=page,
            from_timestamp=start_time,
            to_timestamp=end_time,
            name=trace_name,
        )

        for trace in traces_response.data:
            full_trace = langfuse.api.trace.get(trace.id)
            observations_response = langfuse.api.legacy.observations_v1.get_many(
                trace_id=trace.id,
                type="GENERATION",
                limit=50,
            )

            for obs in observations_response.data:
                obs_input = obs.input
                if not isinstance(obs_input, list):
                    continue

                has_msu_prompt = any(
                    isinstance(m, dict) and
                    "You are a friendly student services assistant for Management and Science University" in str(m.get("content", ""))
                    for m in obs_input
                )
                if not has_msu_prompt:
                    continue

                student_input = getattr(full_trace, "input", None)
                ai_output = getattr(full_trace, "output", None)

                system_prompt = next(
                    (m.get("content", "") for m in obs_input if isinstance(m, dict) and m.get("role") == "assistant"),
                    None,
                )
                if not system_prompt:
                    continue

                retrieved_context = extract_retrieved_context(system_prompt)

                if not all([student_input, ai_output, retrieved_context]):
                    continue

                all_data.append({
                    "trace_id": trace.id,
                    "input": student_input,
                    "output": ai_output,
                    "retrieved_context": retrieved_context,
                })
                break

        if len(traces_response.data) < limit:
            break
        page += 1

    return all_data


# ── Evaluate ───────────────────────────────────────────────────────────────────

def evaluate_trace(input: str, output: str, retrieved_context: str) -> dict:
    prompt = f"""You are an evaluator for a university chatbot.
Your job is to assess whether the AI's response is grounded in the retrieved context.

IMPORTANT — MULTILINGUAL CONTEXT:
The retrieved context is in English. The AI may respond in Malay, English, or a mix of both.
When evaluating, compare meaning and semantics across languages — do NOT penalise a response
simply because it is in a different language from the context. If a Malay claim is a faithful
translation or restatement of content in the English context, it counts as supported.

STUDENT MESSAGE:
{input}

RETRIEVED CONTEXT (source of truth):
{retrieved_context}

AI RESPONSE:
{output}

Step 1 — Classify the student message into ONE of three types:

TYPE A — CONVERSATIONAL: No information is being sought. Examples: "It's okay", "It's been days",
"Thank you", "I see", "Alright", "Okay", general small talk, emotional expressions, or any statement
that is not requesting facts or university-related information.
→ Set faithfulness=1.0, correctness=1.0, answer_relevance=1.0. No further evaluation needed.

TYPE B — VAGUE: Single word, short affirmation, or filler with no clear question intent.
Examples: "ya", "ok", "yes", "iya", "sure", "noted", anything under 3 words.
→ Set answer_relevance=0.8. Evaluate ONLY faithfulness and correctness against retrieved context.

TYPE C — SUBSTANTIVE QUESTION: A clear question or request for university-related information.
→ Evaluate all three criteria normally.

Evaluate on:
1. FAITHFULNESS (0-1): Are all claims in the AI response grounded in the retrieved context?
   Penalise anything invented or not present in the context (hallucination detection).

2. ANSWER_RELEVANCE (0-1): Does the response actually address what the student asked?

3. CORRECTNESS (0-1): Did the AI interpret the context accurately and give the right answer?
   Penalise misread values, incomplete answers, or errors in reasoning based on the context.

Respond ONLY with a valid JSON object, no extra text:
{{
  "faithfulness": <float 0-1>,
  "answer_relevance": <float 0-1>,
  "correctness": <float 0-1>
}}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)


# ── Send to n8n ────────────────────────────────────────────────────────────────

def send_to_n8n(payload: dict):
    if not N8N_WEBHOOK_URL:
        print("⚠️  N8N_WEBHOOK_URL not set — skipping webhook.")
        return
    response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
    response.raise_for_status()
    print(f"📤 Sent to n8n — status: {response.status_code}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    run_date = end.strftime("%Y-%m-%d")

    print(f"🔄 [{run_date}] Fetching traces from Langfuse...")

    traces = fetch_traces(
        start_time=start,
        end_time=end,
        limit=50,
        trace_name=TRACE_NAME,
    )

    print(f"📦 Fetched {len(traces)} traces")

    faithfulness_scores = []
    relevance_scores = []
    correctness_scores = []
    skipped_fallback = 0
    evaluated = 0

    for trace in traces:
        if is_fallback(str(trace["output"])):
            skipped_fallback += 1
            continue

        print(f"🔍 Evaluating trace {trace['trace_id']}...")

        try:
            scores = evaluate_trace(
                input=str(trace["input"]),
                output=str(trace["output"]),
                retrieved_context=trace["retrieved_context"],
            )
            f_score = scores["faithfulness"]
            r_score = scores["answer_relevance"]
            c_score = scores["correctness"]
            overall = round((f_score + r_score + c_score) / 3, 2)

            faithfulness_scores.append(f_score)
            relevance_scores.append(r_score)
            correctness_scores.append(c_score)
            evaluated += 1

            print(f"✅ {trace['trace_id']} — F:{f_score} R:{r_score} C:{c_score} Overall:{overall}")

        except Exception as e:
            print(f"⚠️  {trace['trace_id']} — eval failed: {e}")

    if not evaluated:
        print("⚠️  No evaluable traces found.")
        return

    avg_faithfulness = round(sum(faithfulness_scores) / evaluated, 2)
    avg_relevance    = round(sum(relevance_scores)    / evaluated, 2)
    avg_correctness  = round(sum(correctness_scores)  / evaluated, 2)
    avg_overall      = round((avg_faithfulness + avg_relevance + avg_correctness) / 3, 2)

    summary = {
        "run_date": run_date,
        "total_traces": len(traces),
        "evaluated": evaluated,
        "skipped_fallback": skipped_fallback,
        "avg_faithfulness": avg_faithfulness,
        "avg_answer_relevance": avg_relevance,
        "avg_correctness": avg_correctness,
        "avg_overall": avg_overall,
        "trace_name": TRACE_NAME
    }

    print(f"\n📊 Summary for {run_date}")
    print(f"   Evaluated:        {evaluated} (skipped {skipped_fallback} fallbacks)")
    print(f"   Faithfulness:     {avg_faithfulness}")
    print(f"   Answer Relevance: {avg_relevance}")
    print(f"   Correctness:      {avg_correctness}")
    print(f"   Overall:          {avg_overall}")

    send_to_n8n({"summary": summary})


if __name__ == "__main__":
    run()