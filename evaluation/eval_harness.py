"""
RAGAS-style CI Evaluation Gate.

Runs against the live FastAPI server using a golden test set.
Each test case has a question and a known ground-truth answer.

USAGE:
  # Step 1: start the API server
  uvicorn customer_support_bot.api.app:app --reload

  # Step 2: generate an evaluation token
  python -c "
import jwt, time
token = jwt.encode({'tenant_id': 'acme', 'exp': int(time.time()) + 3600}, 'dev-secret-change-in-production', algorithm='HS256')
print(token)
"

  # Step 3: run evaluation
  python -m evaluation.eval_harness \\
    --golden tests/golden_test_set.json \\
    --api-url http://localhost:8000 \\
    --tenant-id acme \\
    --jwt-token <token from step 2>

  Exit code 0 = all metrics pass
  Exit code 1 = one or more metrics failed

WHAT THIS MEASURES:
  faithfulness      - are the answers grounded in the policy context?
  answer_relevancy  - do the answers actually address the question?
  api_success_rate  - fraction of questions that returned a valid response
  fallback_rate     - fraction of questions that hit the fallback message
                      (a high fallback rate means the FAISS index is too thin)
"""

import json
import sys
import argparse
import logging
import time
from typing import List, Dict, Any

import jwt
import httpx
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from customer_support_bot.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# CI gate thresholds.
# If any metric drops below its threshold the script exits with code 1.
THRESHOLDS = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.75,
    "api_success_rate": 1.00,
    # fallback_rate is a MAX threshold - high fallback = index too sparse
    "fallback_rate": 0.50,
}

# ⚠️ Updated from deprecated api_key= → groq_api_key= (latest as of 2025)
judge_llm = ChatGroq(
    model=CONFIG.llm_model,
    temperature=0.0,
    groq_api_key=CONFIG.groq_api_key,
)

ANSWER_RELEVANCY_PROMPT = """You are evaluating a customer-support bot answer.

Score how well the ANSWER addresses the QUESTION.

- 1.0 = directly and correctly answers the question
- 0.75 = mostly correct but missing a detail
- 0.50 = partially answers, incomplete or off-target
- 0.25 = weakly related, does not really answer
- 0.0 = does not answer the question at all

Respond ONLY with valid JSON:
{{"score": 0.0, "reason": "one sentence"}}

QUESTION: {question}
ANSWER: {answer}
GROUND_TRUTH: {ground_truth}
"""

FAITHFULNESS_EVAL_PROMPT = """You are evaluating whether a customer-support answer is grounded in context.

- 1.0 = all important claims are supported by the context
- 0.75 = mostly supported, minor paraphrasing allowed
- 0.50 = some supported, some unsupported
- 0.25 = mostly unsupported
- 0.0 = unsupported or fabricated

Respond ONLY with valid JSON:
{{"score": 0.0, "reason": "one sentence"}}

ANSWER: {answer}
CONTEXT: {context}
GROUND_TRUTH: {ground_truth}
"""


def load_golden_test_set(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def make_eval_jwt(tenant_id: str, secret: str) -> str:
    return jwt.encode(
        {"tenant_id": tenant_id, "exp": int(time.time()) + 3600},
        secret,
        algorithm="HS256",
    )


def call_api(question: str, tenant_id: str, api_url: str, jwt_token: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {jwt_token}"}
    payload = {"question": question, "session_id": "eval_harness"}
    response = httpx.post(f"{api_url}/query", json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def score_with_groq(prompt: str) -> Dict[str, Any]:
    try:
        response = judge_llm.invoke([HumanMessage(content=prompt)])
        return json.loads(response.content)
    except Exception as e:
        logger.warning("Groq judge failed: %s", e)
        return {"score": 0.0, "reason": "Judge failed to return valid JSON."}


def run_evaluation(
    golden_path: str,
    api_url: str,
    tenant_id: str,
    jwt_token: str,
) -> Dict[str, float]:
    test_cases = load_golden_test_set(golden_path)
    logger.info("Loaded %d test cases from %s", len(test_cases), golden_path)

    faithfulness_scores = []
    relevancy_scores = []
    success_count = 0
    fallback_count = 0

    for i, case in enumerate(test_cases, start=1):
        logger.info("Case %d/%d: %s", i, len(test_cases), case["question"][:70])
        try:
            result = call_api(case["question"], tenant_id, api_url, jwt_token)
            success_count += 1

            answer = result.get("answer", "")
            context = " | ".join(result.get("citations", ["No context retrieved"]))
            ground_truth = case["ground_truth"]

            # Detect fallback responses
            if (
                "please contact our support team" in answer.lower()
                or result.get("faithfulness_score", 1.0) == 0.0
            ):
                fallback_count += 1

            rel = score_with_groq(
                ANSWER_RELEVANCY_PROMPT.format(
                    question=case["question"],
                    answer=answer,
                    ground_truth=ground_truth,
                )
            )
            relevancy_scores.append(float(rel.get("score", 0.0)))

            faith = score_with_groq(
                FAITHFULNESS_EVAL_PROMPT.format(
                    answer=answer,
                    context=context,
                    ground_truth=ground_truth,
                )
            )
            faithfulness_scores.append(float(faith.get("score", 0.0)))

        except Exception as e:
            logger.error("Case %d failed: %s", i, e)
            relevancy_scores.append(0.0)
            faithfulness_scores.append(0.0)

    total = max(len(test_cases), 1)
    return {
        "faithfulness": sum(faithfulness_scores) / total,
        "answer_relevancy": sum(relevancy_scores) / total,
        "api_success_rate": success_count / total,
        "fallback_rate": fallback_count / total,
    }


def print_results(results: Dict[str, float]) -> bool:
    all_pass = True
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for metric, threshold in THRESHOLDS.items():
        score = results.get(metric, 0.0)
        if metric == "fallback_rate":
            passed = score <= threshold
            label = f"(max allowed: {threshold})"
        else:
            passed = score >= threshold
            label = f"(threshold:   {threshold})"

        status = "PASS" if passed else "FAIL"
        print(f"{metric:25s}: {score:.3f}  {label}  [{status}]")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL METRICS PASS - deployment allowed")
    else:
        print("METRICS FAILED - deployment blocked")
    print("=" * 60 + "\n")
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--golden", required=True, help="Path to golden_test_set.json")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--tenant-id", default="acme")
    parser.add_argument(
        "--jwt-token",
        default=None,
        help="JWT token for API auth. If not provided, auto-generated from JWT_SECRET in .env",
    )
    args = parser.parse_args()

    # Auto-generate JWT from config secret if no token was passed
    jwt_token = args.jwt_token or make_eval_jwt(args.tenant_id, CONFIG.jwt_secret)

    results = run_evaluation(args.golden, args.api_url, args.tenant_id, jwt_token)
    passed = print_results(results)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
