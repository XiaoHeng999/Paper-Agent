"""Generate offline answers to real PDF questions and check fact terms and citations."""
import argparse
import json
import re
from pathlib import Path


def parse_answer(output):
    try:
        response = json.loads(output.strip().removeprefix("```json").removesuffix("```").strip())
    except ValueError:
        response = None
    if isinstance(response, list):
        response = {"parts": response}
    if not isinstance(response, dict):
        response = {}
    return response


def grounded_parts(parts, evidence):
    return isinstance(parts, list) and len(parts) >= 2 and all(
        isinstance(part, dict) and type(part.get("source")) is int
        and 1 <= part["source"] <= len(evidence) and isinstance(part.get("quote"), str)
        and bool(part["quote"].strip()) and part["quote"] in evidence[part["source"]-1]["text"]
        and isinstance(part.get("answer"), str) and bool(part["answer"].strip()) for part in parts)


def score(case, output):
    response = parse_answer(output)
    answer = response.get("answer", "")
    citations = response.get("citations")
    evidence = case["evidence"]
    parts = response.get("parts", [])
    if isinstance(parts, list) and parts and all(isinstance(p, dict) for p in parts):
        answer = " ".join(p["answer"] for p in parts if isinstance(p.get("answer"), str))
        citations = [p.get("source") for p in parts]
    valid = isinstance(citations, list) and all(type(i) is int and 1 <= i <= len(evidence) for i in citations)
    cited = {evidence[i-1]["id"] for i in citations} if valid else set()
    compact = re.sub(r"[\W_]", "", answer.casefold()) if isinstance(answer, str) else ""
    facts = [any(re.sub(r"[\W_]", "", term.casefold()) in compact for term in alternatives)
             for alternatives in case["answer_terms"]]
    return {"paper_id": case["paper_id"], "output": output, "fact_term_matches": facts,
            "all_parts_have_verbatim_quotes": grounded_parts(parts, evidence),
            "both_sources_cited": valid and all(cited.intersection(ids) for ids in case["expected_sources"])}


def evaluate(inputs, model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype="auto").to(device).eval()
    results = []
    for case in inputs:
        excerpts = "\n".join(f"[{i}] Page {s['page']}: {s['text']}" for i, s in enumerate(case["evidence"], 1))
        prompt = (f"{excerpts}\n\n{case['question']}\nAnswer each clause separately in English. "
                  "For a location give the printed page or section. For a measured result give the reported value. "
                  "For each clause choose the excerpt that directly answers it and copy a short exact quote. "
                  'Return JSON only: {"parts":[{"answer":"...", "source":1, "quote":"..."}]}.')
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(2):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            tokens = tokenizer(text, return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = model.generate(**tokens, do_sample=False, max_new_tokens=384)
            output = tokenizer.decode(generated[0, tokens.input_ids.shape[1]:], skip_special_tokens=True)
            parts = parse_answer(output).get("parts", [])
            if grounded_parts(parts, case["evidence"]):
                break
            if attempt == 0:
                messages.extend([{"role":"assistant", "content":output},
                    {"role":"user", "content":"Check both clauses of the question. Give a separate answer for each. "
                        "Copy an exact short substring from each supporting numbered excerpt. Verify its number. "
                        "When asked where, give the page number. Return the corrected JSON."}])
        results.append({**score(case, output), "attempts": attempt + 1})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("model", type=Path, help="Local Transformers chat model directory")
    args = parser.parse_args()
    print(json.dumps(evaluate(json.loads(args.inputs.read_text()), args.model), ensure_ascii=False, indent=2))
