"""Real-paper evaluation, version-bound human annotation and separate score cohorts.

Run `uv run python -m src.research.evaluation --help`.
Human identity/expertise are declared by annotators, not authenticated by this CLI.
"""
import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from .models import Record, Claim, EvidenceSpan, ResearchOptions, version_of


class Source(Record):
    id: str
    paper_id: str
    title: str
    url: str
    locator: str
    text: str = Field(min_length=1)
    level: Literal["full_text", "abstract"]
    origin: Literal["real_paper", "controlled_fixture"] = "real_paper"


class Labels(Record):
    fidelity: Literal["supported", "qualified", "contradicted", "unknown"] | None = None
    experimental_support: Literal["supported", "qualified", "contradicted", "unknown"] | None = None
    relation: Literal["equivalent", "extends", "supports", "conflicts", "different_scope", "incomparable"] | None = None

    @model_validator(mode="after")
    def not_empty(self):
        if not any(v is not None for v in self.model_dump().values()):
            raise ValueError("评测标签不能为空")
        return self


class Case(Record):
    id: str
    split: Literal["development", "holdout"]
    source_ids: list[str] = Field(min_length=1)
    claims: list[dict] = Field(min_length=1, max_length=2)
    reference: Labels
    rationale: str = Field(min_length=1)
    annotation_status: Literal["source_grounded_draft", "controlled_fixture"] = "source_grounded_draft"


class Review(Record):
    case_id: str
    case_version: str
    reviewer: str = Field(min_length=1)
    expertise: str = Field(min_length=1)
    reviewed_at: str = Field(min_length=1)
    origin: Literal["human"]
    source_ids: list[str] = Field(min_length=1)
    labels: Labels
    rationale: str = Field(min_length=1)


def load_dataset(path):
    value = json.loads(Path(path).read_text())
    sources = [Source.model_validate(s) for s in value["sources"]]
    cases = [Case.model_validate(c) for c in value["cases"]]
    if len({s.id for s in sources}) != len(sources) or len({c.id for c in cases}) != len(cases):
        raise ValueError("重复的来源或样例 ID")
    lookup = {s.id: s for s in sources}
    for case in cases:
        if not set(case.source_ids) <= lookup.keys():
            raise ValueError("样例引用了不存在的来源")
        if any(lookup[sid].origin == "controlled_fixture" for sid in case.source_ids) and case.annotation_status != "controlled_fixture":
            raise ValueError("合成来源必须标记为 controlled_fixture，不能伪装真实论文")
        for i, draft in enumerate(case.claims):
            c = Claim(**draft, id=f"{case.id}:{i}", version=version_of(draft))
            if not set(c.evidence_ids) <= set(case.source_ids) or any(lookup[eid].paper_id != c.paper_id for eid in c.evidence_ids):
                raise ValueError("评测主张引用越界")
        if bool(case.reference.relation) != (len(case.claims) == 2):
            raise ValueError("关系评测必须包含两个主张")
    return sources, cases


def case_version(case, sources):
    return version_of({"case": case.model_dump(), "sources": [s.model_dump() for s in sources if s.id in case.source_ids]})


def human_labels(cases, sources, records):
    lookup = {c.id: c for c in cases}
    votes, identities = {}, set()
    for record in records:
        review = Review.model_validate(record)
        case = lookup.get(review.case_id)
        if case and case.annotation_status == "controlled_fixture":
            raise ValueError("受控合成样例不能进入真实论文专家指标")
        if not case or review.case_version != case_version(case, sources):
            raise ValueError("人工标注对应的样例或来源版本已过期")
        if set(review.source_ids) != set(case.source_ids):
            raise ValueError("专家需核查样例的全部来源")
        if {k for k, v in review.labels.model_dump().items() if v is not None} != {k for k, v in case.reference.model_dump().items() if v is not None}:
            raise ValueError("人工标注维度必须与样例一致")
        identity = (review.case_id, review.reviewer.strip().casefold())
        if identity in identities:
            raise ValueError("同一审阅者不能重复投票")
        identities.add(identity)
        votes.setdefault(case.id, []).append(review.labels.model_dump(exclude_none=True))
    agreed, unresolved = {}, []
    for case_id, labels in votes.items():
        if len(labels) >= 2 and all(label == labels[0] for label in labels):
            agreed[case_id] = labels[0]
        else:
            unresolved.append(case_id)
    return agreed, unresolved


def metrics(cases, predictions, labels):
    selected = [c for c in cases if c.id in labels]
    rows, correct, total, false_conflicts = [], Counter(), Counter(), 0
    for case in selected:
        expected = labels[case.id]
        predicted = predictions.get(case.id, {}).get("labels", {})
        for field, value in expected.items():
            total[field] += 1
            correct[field] += predicted.get(field) == value
            rows.append((field, value, predicted.get(field)))
        false_conflicts += predicted.get("relation") == "conflicts" and expected.get("relation") not in (None, "conflicts")
    macro_f1 = {}
    for field in total:
        values = {gold for f, gold, _ in rows if f == field}
        scores = []
        for label in values:
            tp = sum(f == field and gold == label and pred == label for f, gold, pred in rows)
            fp = sum(f == field and gold != label and pred == label for f, gold, pred in rows)
            fn = sum(f == field and gold == label and pred != label for f, gold, pred in rows)
            scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
        macro_f1[field] = sum(scores) / len(scores)
    decisive = [(field, gold, pred) for field, gold, pred in rows if gold not in {'unknown', 'incomparable'}]
    decisive_correct = sum(gold == pred for _, gold, pred in decisive)
    class_recall = {}
    for field in total:
        class_recall[field] = {label: sum(f == field and gold == label and pred == label for f, gold, pred in rows) /
            sum(f == field and gold == label for f, gold, pred in rows) for label in {g for f, g, _ in rows if f == field}}
    return {"decisive_cases": len(decisive), "decisive_accuracy": decisive_correct / len(decisive) if decisive else None,
        "unnecessary_abstentions": sum(pred in {'unknown', 'incomparable', None} for _, _, pred in decisive),
        "per_class_recall": class_recall, "cases": len(selected), "predicted_cases": sum(c.id in predictions and not predictions[c.id].get("error") for c in selected),
        "accuracy": {f: correct[f] / total[f] for f in total}, "macro_f1": macro_f1,
        "false_conflicts": false_conflicts, "missing_predictions_count_as_errors": True}


def score(sources, cases, predictions, reviews=()):
    lookup = {c.id: c for c in cases}
    records = {}
    for p in predictions:
        if p.get("mode") == "workflow_unaligned":
            raise ValueError("完整流程产物尚未与人工标签对齐，不能直接计算主张准确率")
        if p["case_id"] not in lookup or p["case_id"] in records or p.get("case_version") != case_version(lookup[p["case_id"]], sources):
            raise ValueError("预测样例 ID 重复、不存在或版本不一致")
        records[p["case_id"]] = p
    expert, unresolved = human_labels(cases, sources, reviews)
    real = [c for c in cases if c.annotation_status != 'controlled_fixture']
    controlled = [c for c in cases if c.annotation_status == 'controlled_fixture']
    expected = {c.id: c.reference.model_dump(exclude_none=True) for c in cases}
    baseline = {c.id: {'labels': {'fidelity': 'unknown', 'experimental_support': 'unknown', 'relation': 'incomparable'}} for c in cases}
    return {"dataset_audit": dataset_audit(cases), "real_paper_dataset_audit": dataset_audit(real),
        "abstention_baseline": metrics(cases, baseline, expected),
        "controlled_fixture": metrics(controlled, records, expected),
        "reference_draft": metrics(real, records, expected),
        "expert_consensus": metrics(cases, records, expert), "unresolved_human_reviews": unresolved,
        "expert_status": "available" if expert else "pending_human_annotation"}


def dataset_audit(cases):
    distribution = {field: dict(Counter(getattr(c.reference, field) for c in cases if getattr(c.reference, field) is not None))
                    for field in ('fidelity', 'experimental_support', 'relation')}
    missing_relations = sorted({'supports', 'conflicts', 'equivalent', 'extends', 'different_scope', 'incomparable'} - distribution['relation'].keys())
    missing_experiments = sorted({'supported', 'contradicted', 'unknown'} - distribution['experimental_support'].keys())
    return {'cohorts': dict(Counter(c.annotation_status for c in cases)), 'label_distribution': distribution, 'missing_relation_labels': missing_relations,
        'missing_experimental_labels': missing_experiments, 'balanced_coverage': not (missing_relations or missing_experiments),
        'research_quality_established': False,
        'note': '覆盖检查不是研究质量认证；合成场景与真实论文成绩分别报告，专家标注另行统计。'}


async def run_cases(sources, cases, llm):
    from .agents import ModelTasks, compare, verify
    from .context import ScopedEvidence
    records = []
    for case in cases:
        tasks = ModelTasks(llm, ResearchOptions(max_calls=8, max_repair_rounds=0))
        evidence = [EvidenceSpan(id=s.id, paper_id=s.paper_id, version=version_of(s.model_dump()), text=s.text,
            source_level=s.level, section=s.locator) for s in sources if s.id in case.source_ids]
        claims = [Claim(**c, id=f"{case.id}:{i}", version=version_of(c)) for i, c in enumerate(case.claims)]
        scope = ScopedEvidence(evidence, {c.paper_id for c in claims})
        record = {"case_id": case.id, "case_version": case_version(case, sources), "labels": {}, "model": getattr(llm, "signature", None)}
        try:
            # Expected labels/rationale/other cases are never included in the model context.
            if len(claims) == 2:
                relations = await compare(tasks, "比较给定主张的关系及适用条件", claims, scope)
                record["labels"] = {"relation": relations[0].relation if relations else "incomparable"}
                record["output"] = [r.model_dump() for r in relations]
            else:
                result = await verify(tasks, claims[0], scope)
                record["labels"] = {"fidelity": result.fidelity.status, "experimental_support": result.experimental_support.status}
                record["output"] = result.model_dump()
        except (ValueError, RuntimeError, TimeoutError) as exc:
            record["error"] = str(exc)
        record["usage"], record["calls"] = tasks.usage, tasks.calls
        records.append(record)
    return records


async def run_workflow_cases(sources, cases, llm):
    """Exercise extraction -> verification -> comparison -> synthesis, without gold claims.

    These are source-excerpt workflows. Their output requires human alignment to gold
    judgments; structural checks must not be interpreted as semantic accuracy.
    """
    import tempfile
    from src.agents.contracts import ResearchRequest
    from .workflow import run_research_node
    results = []
    for case in cases:
        with tempfile.TemporaryDirectory() as directory:
            selected = [s for s in sources if s.id in case.source_ids]
            items = []
            for pid in dict.fromkeys(s.paper_id for s in selected):
                paper_sources = [s for s in selected if s.paper_id == pid]
                paper = {'id': pid, 'paperId': pid, 'title': paper_sources[0].title}
                if all(s.level == 'abstract' for s in paper_sources):
                    paper['abstract'] = '\n\n'.join(s.text for s in paper_sources)
                    full_text = {}
                else:
                    folder = Path(directory) / version_of(pid)
                    folder.mkdir()
                    path = folder / 'paper.md'
                    path.write_text('\n\n'.join(s.text for s in paper_sources))
                    full_text = {'markdown_path': str(path)}
                items.append({'paper': paper, 'full_text': full_text})
            result = await run_research_node()({'request': ResearchRequest('阅读来源，明确前提、证据与局限，再比较各篇主张。',
                {'output_mode': 'review' if len(items) > 1 else 'paper_report', 'cross_paper_enabled': len(items) > 1, 'research_max_repair_rounds': 0}),
                'research_node_llm': llm, 'research_evidence_index': None, 'read_results': items})
            report = result['research_report']
            results.append({'case_id': case.id, 'case_version': case_version(case, sources), 'mode': 'workflow_unaligned',
                'semantic_accuracy': None, 'report': report,
                'structural_coverage': {'claims': sum(len(c['claims']) for c in report['cards']), 'relations': len(report['relations']), 'paragraphs': len(report['paragraphs'])}})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "review-template", "score", "run", "run-workflow"])
    parser.add_argument("--dataset", default="evaluation/real-papers.json")
    parser.add_argument("--predictions")
    parser.add_argument("--reviews")
    parser.add_argument("--split", choices=["development", "holdout", "all"], default="development")
    parser.add_argument("--output")
    args = parser.parse_args()
    sources, all_cases = load_dataset(args.dataset)
    cases = [c for c in all_cases if args.split == "all" or c.split == args.split]
    reviews = json.loads(Path(args.reviews).read_text()) if args.reviews else []
    if args.command == "validate":
        expert, unresolved = human_labels(all_cases, sources, reviews)
        result = {"sources": len(sources), "cases": len(cases), "expert_cases": len(expert), "unresolved": unresolved, "dataset_audit": dataset_audit(cases)}
    elif args.command == "review-template":
        result = [{"case_id": c.id, "case_version": case_version(c, sources), "reviewer": "", "expertise": "", "reviewed_at": "",
            "origin": "human", "source_ids": c.source_ids, "labels": {}, "rationale": ""} for c in cases]
    elif args.command == "score":
        if not args.predictions:
            parser.error("score 需要 --predictions")
        predictions = json.loads(Path(args.predictions).read_text())
        # Validate all records first, then select a split without miscounting missing cases.
        score(sources, all_cases, predictions, reviews)
        ids = {c.id for c in cases}
        result = score(sources, cases, [p for p in predictions if p["case_id"] in ids], [r for r in reviews if r["case_id"] in ids])
    else:
        from src.agents.readAgent import load_read_agent_llm
        async def execute():
            llm = load_read_agent_llm()
            if llm is None:
                parser.error("未配置可用研究模型；请先配置 config/model.json，离线 validate/score 无需模型")
            try:
                return await (run_workflow_cases(sources, cases, llm) if args.command == "run-workflow" else run_cases(sources, cases, llm))
            finally:
                await llm.aclose()
        result = asyncio.run(execute())
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as output:
            output.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
