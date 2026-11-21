export interface Argument { claim_id: string; thesis: string; inference: string; conditions: string; alternative_explanation: string; falsification: string; conclusion: string; evidence_ids: string[]; gaps: Array<{question: string; why_material: string; stopping_rule: string}> }
export interface Insight { statement: string; epistemic_status: string; derivation: string; uncertainty: string; distinguishing_test: string; evidence_ids: string[] }
export interface ReviewedCitation {
  sentence: string; scope: string; comparison: { baseline: string; metric: string } | null;
  overreach: string; evidence_id: string; excerpt: string; section: string;
  page_start: number | null; page_end: number | null;
}
export interface ResearchReasoning { paper_id: string; status: string; arguments: Argument[]; insights: Insight[]; external_support: Array<{argument_index: number; relation: string; comparability: string; contribution: string; evidence_ids: string[]}>; checks: Array<{target_type: string; index: number; verdict: string; reason: string; citation?: ReviewedCitation | null}>; warnings: string[]; omitted_evidence_ids: string[] }
export interface Evidence { modality?: string; id: string; paper_id: string; text: string; source_level: string; page_start: number | null; section: string }
export interface Claim { id: string; version: string; paper_id: string; statement: string; conditions: string; experiment: string; study?: Record<string, string>; evidence_ids: string[] }
export interface Check { status: string; reason: string; evidence_ids: string[] }
export interface Verification { visual_observations?: Array<{paper_id: string; page: number; locator: string; observation: string; value: string; unit: string; conditions: string; status: string}>; challenges?: Array<{premise: string; alternative_explanation: string; required_test: string; evidence_ids: string[]}>; target_id: string; target_version: string; verdict: string; reason: string; fidelity: Check; experimental_support: Check; counterevidence: Check }
export interface Card { paper_id: string; title: string; claims: Claim[]; verifications: Verification[]; warnings: string[] }
export interface Dimension { dimension: string; label: string; score: number | null; reason: string; evidence_ids: string[] }
export interface Assessment { paper_id: string; dimensions: Dimension[]; weighted_score: number | null; score_reason: string }
export interface Relation { relation_id: string; source: string; target: string; left_paper_id: string; right_paper_id: string; relation: string; verdict: string; rationale: string; evidence_ids: string[]; unresolved_questions: string[]; axes: Array<{axis: string; left: string; right: string; status: string; evidence_ids: string[]}> }
export interface ResearchReport { reasoning?: ResearchReasoning[]; incremental?: { enabled: boolean; compatible: boolean; reason: string; reused: Record<string, number> }; visual?: Array<{paper_id: string; status: string; pages: Array<{page: number}>; omitted_pages: number[]; warnings: string[]}>; cards: Card[]; evidence: Evidence[]; assessments?: Assessment[]; cross_paper_executed: boolean; relation_graph?: { nodes: Array<{id: string; paper_id: string; title: string; statement: string}>; edges: Relation[] }; revision_history?: unknown[]; external_search?: {status: string; added_paper_ids: string[]; warnings: string[]}; warnings: string[] }
