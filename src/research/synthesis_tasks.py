"""Model calls for generating and independently reviewing a complete paper draft."""
from typing import Literal

from pydantic import Field
from .models import Record
from .argumentation import CITATION_TASK, Synthesis, ask_scoped, check_summary


class ReasoningCheck(Record):
    citation_verdict: Literal['comparison', 'non_comparison', 'withhold'] = 'withhold'
    target_type: Literal['argument', 'insight', 'external_support', 'method_step', 'reading_passage', 'flow_edge']
    index: int = Field(ge=0)
    verdict: Literal['supported', 'qualified', 'unsupported']
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class ReasoningReview(Record):
    checks: list[ReasoningCheck] = Field(max_length=33)


async def generate_draft(tasks, question, card, plan, claims, supplied, question_map, external_ids):
    return await ask_scoped(tasks, 'research_synthesis',
        '按先理解、再分析的顺序写 reading_guide。第一段用连贯白话总结研究问题、已有做法卡在哪里、作者改了什么、'
        '在什么设置观察到什么结果和适用范围。后续按理解所需写若干有标题的详细讲解，每段承接上段，'
        '解释一个具体设计选择或实验：先给直觉，再解释机制、为什么这样设计，再用原文公式、实例或带基线的观测帮助读者复述。'
        '清楚区分作者实验、教学例子和你的推断。方法步骤承担可执行的演算，避免重复讲解段落。'
        '每段列出它所铺垫的本次论证 claim_ids 和主论文 evidence_ids。覆盖每条论证，'
        '每条 inference 和 insight.derivation 明确点出前文讲过的具体事实，并解释从它到判断还要用什么前提。'
        '仅摘要时用第一段说明可确认的内容和细节缺口。窄问题只讲该问题需要的背景。'
        '若用户明确只问特定数字等窄问题，遵循其范围。对一般阅读或方法问题，针对提出算法的主论文先用 method_steps 帮初学者重建作者的方法。用同一个具体输入按执行顺序走一遍，'
        '写清每步输入、实际操作、输出流向，朴素做法的瓶颈和为何选此设计。解释关键设计为什么有效：'
        '区分严格等价、带条件的近似、消融的实验证据和系统提出的假设。给出假设失效的具体输入或可区分的对照实验。'
        '若有可定位的算法执行结构，给出一张简明教学图 flow_edges（3至9条边），明确数据复用、分支条件和汇合。'
        'source/target用一致的简短中文节点名，transfer为具体数据或条件；每条边绑定全文证据，'
        'claim_ids仅标后续论证或洞见确实依赖此边的主张。每个新节点应连接到同一张图。'
        '对有相关作者原图的讲解填写figure_number，并在段落证据中引用含该图图注的原页。只用确切出现的图号。'
        '一般方法阅读在总结后展示图；狭窄数值问题、非方法论文、摘要或缺依据时flow_edges留空。'
        '每步只引用主论文已提供的全文。综述、仅摘要、或无法定位算法细节时返回空列表，不虚构组件。'
        '面向学生解释符号。洞见尽量从设计步骤的近似或取舍推导。更新主论文的论证并提出有推导依据的insight。只使用所给原文。外部候选不是已确认佐证：'
        '逐条记录它支持或挑战论证的哪一步及任务/指标/预算/条件可比性；摘要不能验证实验充分性。'
        '没有可比证据时写inconclusive或different_conditions，不要凑引用。'
        'insight区分supported_inference/hypothesis/proposal，给出推导、不确定性和区分替代解释的具体实验；'
        '每条insight写出可观察预测及哪种结果会削弱它。源自已规划缺口时填对应gap_ids。'
        '外部关系必须填写实际检索该候选的gap_id，说明观察如何改变判断。'
        '不要求凑数，不把常规摘要当insight，不把未知条件、阶段描述差异或数字舍入直接当矛盾。'
        'argument_indices和argument_index均指向本次返回arguments的零起始索引。' + CITATION_TASK,
        {'question': question, 'primary_paper_id': card.paper_id, 'arguments': plan['arguments'],
         'claims': [c.model_dump() for c in claims.values()], 'claim_verifications': [check_summary(v, supplied) for v in card.verifications if v.target_id in claims],
         'evidence_questions': list(question_map.values()), 'external_candidate_ids': sorted(external_ids), 'evidence': [e.model_dump() for e in supplied.values()]}, Synthesis, claims, supplied)



async def review_draft(tasks, card, draft, question_map, claims, supplied):
    return await ask_scoped(tasks, 'reasoning_verifier',
        '每条教学图箭头以flow_edge逐条核查，确认起点、终点、传递对象、条件和所绑定主张。'
        '对figure_number核对其图注和段落是否相关。'
        '以reading_passage逐段核查讲解、引用和绑定主张：总结可否由原文支持，讲解是否解释了对应论证实际依赖的前提，'
        '有无混淆教学例子和作者结果。评审argument和insight时检查推导能否接回所关联讲解中的具体事实。'
        '以method_step为target_type逐步核查算法重建：数据流、设计动机是否有原文依据，'
        '推导是否成立，近似是否说明条件，实验能否区分所称机制；对系统推测只评价可检验性。'
        '独立检查每条argument、insight及external_support。为每个对象返回且仅返回一条核查。'
        '对照缺口的可比条件检查外部关系；检查每条见解是否给出可观察且能区分替代解释的预测和削弱结果。'
        '检查证据→推理→结论是否跳步、替代解释、条件可比性、是否只是重复作者摘要。'
        'supported只表示证据支持相应推理；hypothesis/proposal的supported仅表示值得检验，不表示假设已证实。'
        '数字舍入、同一实验不同阶段描述不能直接认定矛盾。引文必须在所给证据内，证据不足返回qualified或unsupported。'
        '对每条argument的 citation_draft 单独判定 citation_verdict：语义核查通过时区分比较主张 comparison 和其他主张 non_comparison。逐字核对摘录，检验转述是否保留基线、指标、'
        '适用条件和贡献归因。比较型转述缺少comparison、反驳条件写成已验证、或限定语不能消除歧义时返回withhold。'
        '批准时将决定性的原文编号纳入本条核查的evidence_ids；其他对象返回withhold。',
        {'primary_paper_id': card.paper_id, 'synthesis': draft.model_dump(), 'evidence_questions': list(question_map.values()),
         'claim_verifications': [check_summary(v, supplied) for v in card.verifications if v.target_id in claims],
         'evidence': [e.model_dump() for e in supplied.values()]}, ReasoningReview, claims, supplied)
