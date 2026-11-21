<script setup lang="ts">
import { computed, ref, watch } from "vue";
import MethodFlow from './MethodFlow.vue';
import type { Claim, Evidence, Relation, ResearchReport, ResearchReasoning } from "../../types/research";
import type { SessionRunAccepted } from "../../types/sessions";
const props = defineProps<{ sessionKey: string; artifactId: string; busy: boolean }>();
const emit = defineEmits<{ revision: [value: SessionRunAccepted] }>();
const report = ref<ResearchReport | null>(null);
const artifact = ref("");
const error = ref("");
const loading = ref(false);
const submitting = ref(false);
const evidence = ref<Evidence[]>([]);
const selectedRelation = ref<Relation | null>(null);
const editing = ref<Claim | null>(null);
const statement = ref("");
const conditions = ref("");
const reason = ref("");
let generation = 0;
watch(() => [props.sessionKey, props.artifactId], async () => {
  const current = ++generation;
  report.value = null; evidence.value = []; editing.value = null; selectedRelation.value = null; error.value = "";
  if (!props.sessionKey || !props.artifactId) return;
  loading.value = true;
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(props.sessionKey)}/research`);
    if (!response.ok) throw new Error("研究报告读取失败");
    const data = await response.json();
    if (current === generation) { report.value = data.report; artifact.value = data.artifact_id; }
  } catch (e) { if (current === generation) error.value = String(e); }
  finally { if (current === generation) loading.value = false; }
}, { immediate: true });
const nodes = computed(() => report.value?.relation_graph?.nodes ?? []);
const edges = computed(() => report.value?.relation_graph?.edges ?? []);
const labels: Record<string, string> = {exact: "定义或严格推导", approximation: "近似及其条件", empirical: "实验观察", equivalent: "同质", extends: "扩展", supports: "支持", conflicts: "冲突", different_scope: "条件不同", incomparable: "无法比较", verified: "当前证据支持", qualified: "需限定", contradicted: "与证据不符", insufficient_evidence: "证据不足", needs_human_review: "待人工复核", supported: "支持", unknown: "未证实", hypothesis: "待验证假设", proposal: "研究建议", supported_inference: "证据推论", unsupported: "不支持", unreviewed: "待审", inconclusive: "不足以判断", challenges: "挑战", different_conditions: "条件不同", matched: "匹配", different: "不同"};
function label(value: string) { return labels[value] ?? value; }
function reasoningCheck(result: ResearchReasoning, kind: string, index: number) {
  const check = result.checks.find(c => c.target_type === kind && c.index === index);
  return check ? `${label(check.verdict)}：${check.reason}` : '尚未完成独立核查';
}
function relatedEdges(result: ResearchReasoning, indices: number[] = []) {
  return (result.flow_edges ?? []).map((e, index) => ({...e, index}))
    .filter(e => indices.includes(e.index));
}
function relatedPassages(result: ResearchReasoning, claims: string[]) {
  return (result.reading_guide ?? []).map((p, index) => ({ ...p, index }))
    .filter(p => p.claim_ids.some(id => claims.includes(id)));
}
function showEvidence(ids: string[]) { evidence.value = (report.value?.evidence ?? []).filter(e => ids.includes(e.id)); }
function selectRelation(edge: Relation) { selectedRelation.value = edge; showEvidence(edge.evidence_ids); }
function position(id: string) {
  const index = nodes.value.findIndex(node => node.id === id);
  const angle = Math.PI * 2 * index / Math.max(1, nodes.value.length);
  return { x: 300 + Math.cos(angle) * 235, y: 180 + Math.sin(angle) * 130 };
}
function edit(claim: Claim) { editing.value = claim; statement.value = claim.statement; conditions.value = claim.conditions; reason.value = ""; }
async function revise() {
  if (!editing.value || !reason.value.trim() || submitting.value || props.busy) return;
  const session = props.sessionKey;
  const current = generation;
  submitting.value = true; error.value = "";
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(props.sessionKey)}/research/revisions`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({artifact_id: artifact.value, claim_id: editing.value.id, target_version: editing.value.version,
        reason: reason.value, changes: {statement: statement.value, conditions: conditions.value}}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message ?? data.detail ?? "修订失败");
    if (current === generation && session === props.sessionKey) { editing.value = null; emit("revision", data); }
  } catch (e) { if (current === generation) error.value = String(e); }
  finally { submitting.value = false; }
}
</script>

<template>
  <section class="research-panel" aria-label="论文判断与证据">
    <h3>判断、评价与证据</h3>
    <p v-if="loading">正在读取研究对象…</p><p v-if="error" role="alert">{{ error }}</p>
    <template v-if="report">
      <p>评价仅针对已读取来源；unknown 表示证据或核查不足。人工修订将生成新报告，原记录保留。</p>
      <section v-for="result in report.reasoning ?? []" :key="result.paper_id" class="reasoning">
        <h4>论文讲解与分析 · {{ result.paper_id }}</h4>
        <p>系统推论和假设不等于作者结论；核查支持不表示已经独立复现。</p>
        <p v-for="warning in result.warnings" :key="warning">{{ warning }}</p>
        <section v-for="(passage, index) in result.reading_guide" :id="result.paper_id + '-reading-' + index" :key="index">
          <h4>{{ index === 0 ? '论文总结' : '讲解 ' + index }} · {{ passage.title }}</h4>
          <p style="white-space: pre-line">{{ passage.explanation }}</p>
          <details><summary>讲解依据与核查</summary>
            <p>{{ reasoningCheck(result, 'reading_passage', index) }}</p>
            <button type="button" @click="showEvidence(passage.evidence_ids)">查看原文</button>
          </details>
          <figure v-for="fig in result.reading_figures?.filter(f => f.passage_index === index)" :key="fig.number">
            <figcaption>作者图 {{ fig.number }} · 第 {{ fig.page }} 页 · {{ fig.caption_excerpt }}</figcaption>
            <a v-if="fig.url" :href="fig.url" target="_blank" rel="noopener">
              <img :src="fig.url" :alt="'作者图 ' + fig.number + ' 所在原页，点击放大'" loading="lazy" style="max-width:100%;max-height:28rem" />
            </a>
          </figure>
          <MethodFlow v-if="index === 0 && result.flow_edges?.length" :edges="result.flow_edges" :partial="result.flow_scope === 'partial'" :paper="result.paper_id" @evidence="showEvidence"/>
        </section>
        <section v-if="result.method_steps?.length">
          <h4>算法如何工作</h4>
          <ol><li v-for="(step, index) in result.method_steps" :key="index">
            <strong>{{ step.operation }}</strong>
            <p>输入：{{ step.input }}</p><p>输出与流向：{{ step.output }}</p>
            <p>设计取舍：{{ step.design_reason }}</p>
            <p>{{ label(step.basis) }}：{{ step.justification }}</p>
            <p>失效条件或检验：{{ step.failure_or_test }}</p>
            <p>{{ reasoningCheck(result, 'method_step', index) }}</p>
            <button type="button" @click="showEvidence(step.evidence_ids)">查看步骤原文</button>
          </li></ol>
        </section>
        <article v-for="(argument, index) in result.arguments" :key="argument.claim_id">
          <h5>{{ argument.thesis }}</h5>
          <p v-for="e in relatedEdges(result, argument.flow_edge_indices)" :key="e.index">
            对应图中路径 · <a :href="'#' + result.paper_id + '-flow-' + e.index">{{ e.source }} → {{ e.target }}</a>
          </p>
          <p v-for="p in relatedPassages(result, [argument.claim_id])" :key="p.index">
            承接前文 · <a :href="'#' + result.paper_id + '-reading-' + p.index">{{ p.title }}</a>
          </p>
          <p>推理：{{ argument.inference }}</p><p>条件：{{ argument.conditions }}</p>
          <p>替代解释：{{ argument.alternative_explanation }}</p><p>证伪条件：{{ argument.falsification }}</p>
          <p>结论：{{ argument.conclusion }}</p><p>{{ reasoningCheck(result, 'argument', index) }}</p>
          <template v-for="check in result.checks.filter(c => c.target_type === 'argument' && c.index === index)" :key="check.index">
            <div v-if="check.citation">
              <p><strong>可引用转述</strong>　{{ check.citation.sentence }}</p>
              <p>引用边界：{{ check.citation.overreach }}</p>
              <blockquote>{{ check.citation.excerpt }}</blockquote>
              <button type="button" @click="showEvidence([check.citation.evidence_id])">查看决定性原文 {{ check.citation.page_start ? '· 第 ' + check.citation.page_start + ' 页' : check.citation.section }}</button>
            </div>
            <p v-else-if="check.citation === null">尚无通过核查的转述</p>
          </template>
          <p v-for="gap in argument.gaps" :key="gap.question">待补证：{{ gap.question }}；影响：{{ gap.why_material }}；停止标准：{{ gap.stopping_rule }}</p>
          <button type="button" @click="showEvidence(argument.evidence_ids)">查看论证原文</button>
        </article>
        <details v-for="q in result.evidence_questions" :key="q.gap_id">
          <summary>{{ q.question }}</summary>
          <p>假设：{{ q.hypothesis }}</p><p>预期观察：{{ q.prediction }}</p>
          <p>会削弱判断的观察：{{ q.weakening_observation }}</p>
          <p>比较条件：{{ q.comparable_conditions }}</p><p>外查停止条件：{{ q.stopping_rule }}</p>
          <p v-if="q.local_review.decision === 'external_needed' && !q.candidate_ids.length">该缺口尚无可用外部候选，保留当前判断。</p>
          <p>本文核查与外查决定：{{ q.local_review.reason }}</p>
          <button v-if="q.local_review.evidence_ids.length" @click="showEvidence(q.local_review.evidence_ids)">查看本文依据</button>
        </details>
        <article v-for="(support, index) in result.external_support" :key="`support-${index}`">
          <h5>论证 {{ support.argument_index + 1 }} 的外部证据：{{ label(support.relation) }}</h5>
          <p v-if="support.gap_id">对应问题：{{ result.evidence_questions?.find(q => q.gap_id === support.gap_id)?.question }}</p>
          <p>对判断的影响：{{ result.checks.find(c => c.target_type === 'external_support' && c.index === index)?.judgment_update ?? '待核查' }}</p>
          <p>{{ support.contribution }}</p><p>可比性：{{ support.comparability }}</p>
          <p>{{ reasoningCheck(result, 'external_support', index) }}</p>
          <button type="button" @click="showEvidence(support.evidence_ids)">查看外部证据</button>
        </article>
        <article v-for="(insight, index) in result.insights" :key="`insight-${index}`">
          <h5>{{ label(insight.epistemic_status) }}：{{ insight.statement }}</h5>
          <p v-for="e in relatedEdges(result, insight.flow_edge_indices)" :key="e.index">
            对应图中路径 · <a :href="'#' + result.paper_id + '-flow-' + e.index">{{ e.source }} → {{ e.target }}</a>
          </p>
          <p v-for="p in relatedPassages(result, insight.claim_ids ?? [])" :key="p.index">
            承接前文 · <a :href="'#' + result.paper_id + '-reading-' + p.index">{{ p.title }}</a>
          </p>
          <p v-for="q in result.evidence_questions?.filter(q => insight.gap_ids?.includes(q.gap_id))" :key="q.gap_id">对应证据缺口：{{ q.question }}</p>
          <p>推导：{{ insight.derivation }}</p><p>不确定性：{{ insight.uncertainty }}</p>
          <p v-if="insight.prediction">预期观察：{{ insight.prediction }}</p><p v-if="insight.weakening_observation">会削弱判断的观察：{{ insight.weakening_observation }}</p>
          <p>验证实验：{{ insight.distinguishing_test }}</p><p>{{ reasoningCheck(result, 'insight', index) }}</p>
          <button type="button" @click="showEvidence(insight.evidence_ids)">查看推导依据</button>
        </article>
        <p v-if="!result.insights.length">未形成可交付的 Insight，不强行补齐。</p>
        <p v-if="result.omitted_evidence_ids.length">本阶段尚未覆盖 {{ result.omitted_evidence_ids.length }} 个证据片段。</p>
      </section>
      <p v-if="report.revision_history?.length">已记录 {{ report.revision_history.length }} 次人工修订。</p>
      <p v-if="report.incremental?.enabled">本次更新复用了 {{ report.incremental.reused.cards ?? 0 }} 篇阅读结果、{{ report.incremental.reused.claim_checks ?? 0 }} 条主张核查、{{ report.incremental.reused.pairs ?? 0 }} 组关系及 {{ report.incremental.reused.paragraphs ?? 0 }} 段正文。{{ report.incremental.reason }}</p>
      <details v-if="report.visual?.some(v => v.status !== 'not_applicable')">
        <summary>PDF 视觉读取范围</summary>
        <div v-for="visual in report.visual.filter(v => v.status !== 'not_applicable')" :key="visual.paper_id">
          <p>{{ visual.paper_id }}：已准备 {{ visual.pages.map(p => p.page).join('、') || '无' }} 页原图；未处理 {{ visual.omitted_pages.length }} 页。准备原图不代表核查通过，请结合各项核查结论。</p>
          <p v-for="warning in visual.warnings" :key="warning">{{ warning }}</p>
        </div>
      </details>
      <article v-for="card in report.cards" :key="card.paper_id">
        <h4>{{ card.title }}</h4><p v-for="warning in card.warnings" :key="warning">{{ warning }}</p>
        <table v-if="report.assessments?.find(a => a.paper_id === card.paper_id)">
          <thead><tr><th>维度</th><th>分值（0–4）</th><th>理由</th><th>证据</th></tr></thead>
          <tbody><tr v-for="dimension in report.assessments.find(a => a.paper_id === card.paper_id)?.dimensions" :key="dimension.dimension">
            <td>{{ dimension.label }}</td><td>{{ dimension.score ?? 'unknown' }}</td><td>{{ dimension.reason }}</td>
            <td><button :disabled="!dimension.evidence_ids.length" @click="showEvidence(dimension.evidence_ids)">查看原文</button></td>
          </tr></tbody>
        </table>
        <p>{{ report.assessments?.find(a => a.paper_id === card.paper_id)?.score_reason }}</p>
        <p v-if="report.assessments?.find(a => a.paper_id === card.paper_id)?.weighted_score != null">加权分：{{ report.assessments.find(a => a.paper_id === card.paper_id)?.weighted_score }}</p>
        <details v-for="claim in card.claims" :key="claim.id">
          <summary>{{ claim.statement }}</summary><p>成立前提：{{ claim.conditions }}</p><p>实验：{{ claim.experiment }}</p>
          <p v-for="(value, key) in claim.study" :key="key">{{ key }}：{{ value }}</p>
          <template v-for="check in card.verifications.filter(v => v.target_id === claim.id && v.target_version === claim.version)" :key="check.target_version">
            <p>{{ label(check.verdict) }}：{{ check.reason }}</p>
            <p v-if="check.fidelity">忠实转述：{{ label(check.fidelity.status) }}；{{ check.fidelity.reason }}</p>
            <p v-if="check.experimental_support">实验支持：{{ label(check.experimental_support.status) }}；{{ check.experimental_support.reason }}</p>
            <p v-for="(observation, index) in check.visual_observations" :key="`visual-${index}`">原页观察：第 {{ observation.page }} 页，{{ observation.locator }}；{{ observation.observation }}；{{ observation.value }} {{ observation.unit }}；条件：{{ observation.conditions }}（{{ observation.status }}）</p>
            <p v-for="(challenge, index) in check.challenges" :key="`challenge-${index}`">待验证挑战：前提 {{ challenge.premise }}；可能的其他解释 {{ challenge.alternative_explanation }}；需检查 {{ challenge.required_test }}。这不代表已发现反证。</p>
            <p v-if="check.counterevidence">内部反证：{{ check.counterevidence.status === "supported" ? "发现反证" : check.counterevidence.status === "contradicted" ? "已检查的反例被排除" : label(check.counterevidence.status) }}；{{ check.counterevidence.reason }}</p>
          </template>
          <button @click="showEvidence(claim.evidence_ids)">查看原文</button>
          <button :disabled="busy || submitting" @click="edit(claim)">修订并重新核查</button>
        </details>
      </article>
      <section v-if="report.cross_paper_executed" aria-label="跨论文比较">
        <h4>主张关系图与比较矩阵</h4><p>边表示待核查或已核查的主张关系，请同时查看核查状态。</p>
        <svg v-if="nodes.length && nodes.length <= 60" viewBox="0 0 600 360" role="group" aria-label="点击节点查看主张证据，点击连线查看关系">
          <defs><marker id="research-arrow" markerWidth="7" markerHeight="7" refX="10" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#64748b" /></marker></defs>
          <line v-for="edge in edges" :key="edge.relation_id" :x1="position(edge.source).x" :y1="position(edge.source).y" :x2="position(edge.target).x" :y2="position(edge.target).y" :stroke="edge.verdict === 'verified' ? '#247b63' : '#9a7854'" stroke-width="3" :marker-end="['supports', 'extends'].includes(edge.relation) ? 'url(#research-arrow)' : undefined" tabindex="0" role="button" :aria-label="label(edge.relation)" @click="selectRelation(edge)" @keydown.enter="selectRelation(edge)"><title>{{ label(edge.relation) }} · {{ label(edge.verdict) }}</title></line>
          <g v-for="(node, index) in nodes" :key="node.id" tabindex="0" role="button" :aria-label="node.statement" @click="showEvidence(report.cards.flatMap(c => c.claims).find(c => c.id === node.id)?.evidence_ids ?? [])" @keydown.enter="showEvidence(report.cards.flatMap(c => c.claims).find(c => c.id === node.id)?.evidence_ids ?? [])">
            <circle :cx="position(node.id).x" :cy="position(node.id).y" r="15" fill="#eef4ff" stroke="#5066a2"/><text :x="position(node.id).x" :y="position(node.id).y + 5" text-anchor="middle">{{ index + 1 }}</text><title>{{ node.title }}：{{ node.statement }}</title>
          </g>
        </svg>
        <p v-else-if="nodes.length > 60">主张较多，使用下方矩阵查看，避免关系图拥挤。</p>
        <table><thead><tr><th>论文对</th><th>关系</th><th>核查</th><th>依据</th></tr></thead><tbody>
          <tr v-for="edge in edges" :key="edge.relation_id"><td>{{ edge.left_paper_id }} / {{ edge.right_paper_id }}</td><td>{{ label(edge.relation) }}</td><td>{{ label(edge.verdict) }}</td><td><button @click="selectRelation(edge)">比较条件与证据</button></td></tr>
        </tbody></table><p v-if="!edges.length">没有得到可交付关系，不能据此认定论文之间没有冲突。</p>
        <div v-if="selectedRelation"><p>{{ selectedRelation.rationale }}</p>
          <table><thead><tr><th>比较轴</th><th>左侧</th><th>右侧</th><th>状态</th><th>证据</th></tr></thead><tbody><tr v-for="axis in selectedRelation.axes" :key="axis.axis"><td>{{ axis.axis }}</td><td>{{ axis.left }}</td><td>{{ axis.right }}</td><td>{{ label(axis.status) }}</td><td><button :disabled="!axis.evidence_ids.length" @click="showEvidence(axis.evidence_ids)">原文</button></td></tr></tbody></table>
          <p v-for="question in selectedRelation.unresolved_questions" :key="question">待解决：{{ question }}</p>
        </div>
      </section>
      <form v-if="editing" @submit.prevent="revise"><h4>修订判断</h4>
        <label>判断<textarea v-model="statement" required maxlength="12000" /></label><label>成立前提<textarea v-model="conditions" maxlength="4000" /></label><label>修订理由<textarea v-model="reason" required maxlength="4000" /></label>
        <button type="submit" :disabled="busy || submitting">{{ submitting ? '提交中…' : '保存新版本并重新核查' }}</button><button type="button" @click="editing = null">取消</button>
      </form>
      <section v-if="evidence.length" aria-label="原文证据"><h4>原文证据</h4><blockquote v-for="source in evidence" :key="source.id"><small>{{ source.paper_id }} · {{ source.modality === "visual_transcription" ? "扫描页转录，需对照原图" : source.source_level }} · {{ source.page_start ? `第 ${source.page_start} 页` : source.section }}<br/>{{ source.id }}</small><pre>{{ source.text }}</pre></blockquote></section>
    </template>
  </section>
</template>

<style scoped>
.research-panel { padding: 20px; background: var(--panel, #fff); border: 1px solid #dedfe6; border-radius: 16px; margin-top: 16px; overflow-wrap: anywhere; }
article, form { padding: 14px 0; border-top: 1px solid #e5e7eb; }
table { width: 100%; border-collapse: collapse; font-size: 13px; display: block; overflow-x: auto; }
th, td { text-align: left; padding: 8px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }
button { cursor: pointer; padding: 6px 10px; margin: 4px; border: 1px solid #bdc7df; border-radius: 6px; background: #f4f6fb; }
button:disabled { cursor: default; opacity: .5; } details { padding: 8px 0; } summary { cursor: pointer; }
label { display: block; } textarea { display: block; width: 100%; min-height: 75px; padding: 8px; }
svg { width: 100%; max-height: 440px; } svg line, svg g { cursor: pointer; } svg text { font-size: 12px; }
blockquote { margin: 12px 0; padding: 12px; border-left: 3px solid #8096bb; background: #f6f8fb; } pre { white-space: pre-wrap; } small { color: #657087; }
</style>
