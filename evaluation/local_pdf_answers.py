"""Generate offline answers to real PDF questions and check fact terms and citations."""
import argparse
import json
import re
from pathlib import Path


def score(case, output):
    try:
        response = json.loads(output.strip().removeprefix("```json").removesuffix("```").strip())
    except ValueError:
        response = None
    if not isinstance(response, dict):
        response = {}
    answer = response.get("answer", "")
    citations = response.get("citations")
    evidence = case["evidence"]
    valid = (isinstance(citations, list) and all(type(i) is int and 1 <= i <= len(evidence) for i in citations))
    cited = {evidence[i-1]["id"] for i in citations} if valid else set()
    compact = re.sub(r"[\W_]", "", answer.casefold()) if isinstance(answer, str) else ""
    facts = [any(re.sub(r"[\W_]", "", term.casefold()) in compact for term in alternatives)
             for alternatives in case["answer_terms"]]
    return {"paper_id": case["paper_id"], "output": output, "fact_term_matches": facts,
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
        prompt = (f"{excerpts}\n\n{case['question']}\nAnswer in English using only the excerpts. "
                  'Return one JSON object with "answer" (two sentences) and "citations" '
                  '(an array of the supporting excerpt numbers).')
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        tokens = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(**tokens, do_sample=False, max_new_tokens=256)
        output = tokenizer.decode(generated[0, tokens.input_ids.shape[1]:], skip_special_tokens=True)
        results.append(score(case, output))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("model", type=Path, help="Local Transformers chat model directory")
    args = parser.parse_args()
    print(json.dumps(evaluate(json.loads(args.inputs.read_text()), args.model), ensure_ascii=False, indent=2))
