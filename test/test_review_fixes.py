import asyncio
import base64
import hashlib
import io
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject, ArrayObject
from src.research.agents import ModelTasks, read_claims, verify, write_report
from src.research.assessment import assess_card
from src.research.claims import merge_duplicate
from src.research.context import ScopedEvidence
from src.research.evaluation import load_dataset, score, case_version, dataset_audit, run_workflow_cases
from src.research.incremental import IncrementalPlan
from src.research.models import Claim, EvidenceSpan, ResearchOptions, PaperCard, AnalysisRelation, VerificationResult
from src.research.multimodal import build_multimodal_catalog
from src.research.provenance import source_manifest, document_version
from src.research.scheduling import comparison_schedule
from src.research.visual import labels, prepare_visual, inspect_pdf, verification_images
from src.research.workflow import run_research_node
from src.research.supplement import supplement_sources
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.services.research import latest_report, revision_request
from src.llm.base import LLMResponse
from test.test_research_workflow import FakeModel, item, state
from test.test_research_completion import VisualModel, scanned_pdf


def add_header(path, text, index=0):
    writer = PdfWriter(clone_from=str(path))
    page = writer.pages[index]
    fonts = page['/Resources'].setdefault(NameObject('/Font'), DictionaryObject())
    fonts[NameObject('/HeaderFont')] = writer._add_object(DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')}))
    stream = DecodedStreamObject()
    stream.set_data(f'BT /HeaderFont 12 Tf 20 750 Td ({text}) Tj ET'.encode())
    existing = page.raw_get('/Contents')
    page[NameObject('/Contents')] = ArrayObject([existing, writer._add_object(stream)])
    with path.open('wb') as output:
        writer.write(output)


def two_figure_pdf(path):
    scanned_pdf(path)
    add_header(path, "Native text")
    writer = PdfWriter(clone_from=str(path))
    page = writer.pages[0]
    name = next(iter(page["/Resources"]["/XObject"]))
    stream = DecodedStreamObject()
    stream.set_data(f"""q 160 0 0 100 40 470 cm {name} Do Q
        BT /HeaderFont 12 Tf 40 450 Td (Figure 1: First experiment architecture) Tj ET
        q 200 0 0 80 40 240 cm {name} Do Q
        BT /HeaderFont 12 Tf 40 220 Td (Figure 2: Second experiment latency) Tj ET""".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as out:
        writer.write(out)
    return writer, stream


def check(target, **kwargs):
    return VerificationResult(target_id=target.id, target_version=target.version, verdict='verified', reason='source supports', evidence_ids=target.evidence_ids,
        scope='paper' if isinstance(target, Claim) else 'cross_paper', source_levels=['full_text'],
        fidelity={'status': 'supported', 'reason': 'source', 'evidence_ids': target.evidence_ids}, **kwargs)


def claim(pid, name='', kind='result'):
    return Claim(id=pid+name, paper_id=pid, version='v', statement='controlled effect'+name, kind=kind, evidence_ids=[pid+':e'])


def relation(a, b):
    return AnalysisRelation(id=a.id+'-'+b.id, version='v', claim_ids=[a.id,b.id], relation='conflicts', conditions='matched setting', rationale='Opposite effects', evidence_ids=[*a.evidence_ids,*b.evidence_ids])


class SourceRevisionTests(unittest.IsolatedAsyncioTestCase):
    async def prepare_revision(self, directory, target_paper):
        path = Path(directory)/'scan.pdf'
        scanned_pdf(path)
        first_paper = item()
        first_paper['full_text']['source_path'] = str(path)
        repo = SQLiteSessionRepository(storage_root=directory)
        session = repo.create()
        task = state(VisualModel(), [first_paper,item('p2')], visual_page_limit=1)
        task.update(session_repo=repo,session_key=session.key,turn_id='one')
        report = (await run_research_node()(task))['research_report']
        target = next(c for card in report['cards'] if card['paper_id']==target_paper for c in card['claims'])
        checkpoint = revision_request(repo,session.key,{'artifact_id':latest_report(repo,session.key)['artifact_id'], 'claim_id':target['id'], 'target_version':target['version'], 'reason':'scope', 'changes':{'statement':'revised'}})['read_resume_checkpoint']
        return path,repo,session,report,checkpoint

    async def test_appended_unread_page_invalidates_other_card(self):
        with tempfile.TemporaryDirectory() as d:
            path,repo,session,old,cp = await self.prepare_revision(d,'p2')
            scanned_pdf(path,2)
            task=state(VisualModel(),cp['read_results'],visual_page_limit=1)
            task.update(session_repo=repo,session_key=session.key,turn_id='two',read_resume_checkpoint=cp)
            report=(await run_research_node()(task))['research_report']
            self.assertIn('p1',report['incremental']['changed_source_papers'])
            self.assertIn('p1:unread-visual-page:2',report['cards'][0]['omitted_evidence_ids'])
            self.assertNotEqual(old['cards'][0]['version'],report['cards'][0]['version'])
            self.assertTrue(report['visual'][0]['pages'][0]['transcription_reused'])

    async def test_target_document_change_rejects_stale_human_revision(self):
        with tempfile.TemporaryDirectory() as d:
            path,repo,session,_,cp=await self.prepare_revision(d,'p1')
            scanned_pdf(path,2)
            task=state(VisualModel(),cp['read_results'],visual_page_limit=1)
            task.update(session_repo=repo,session_key=session.key,turn_id='two',read_resume_checkpoint=cp)
            with self.assertRaisesRegex(ValueError,'原文版本已变化'):
                await run_research_node()(task)

    def test_coverage_and_missing_source_change_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.pdf';path.write_bytes(b'original')
            paper=item();paper['full_text']['source_path']=str(path)
            audit={'status':'prepared','pages':[],'omitted_pages':[]}
            initial=source_manifest(document_version(paper),[],audit)
            audit['omitted_pages']=[2]
            self.assertNotEqual(initial['version'],source_manifest(document_version(paper),[],audit)['version'])
            path.unlink()
            self.assertNotEqual(initial['document'],document_version(paper))


class VisualReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_query_transcribes_only_matching_raster_and_verifies_page(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.pdf"
            writer, stream = two_figure_pdf(path)
            paper = item()
            paper["full_text"]["source_path"] = str(path)
            model = VisualModel()
            tasks = ModelTasks(model, ResearchOptions())
            tasks.question = "Read the values in Fig. 2"
            evidence, audit = await prepare_visual(tasks, paper, [])
            self.assertEqual(audit["warnings"], [])
            self.assertEqual(len(model.images), 1)
            crop = base64.b64decode(model.images[0][0]["image_url"]["url"].split(",")[1])
            self.assertEqual(Image.open(io.BytesIO(crop)).size, (400, 160))
            updated, objects = build_multimodal_catalog(evidence, tasks.visual_catalog["p1"], tasks.visual_pages)
            self.assertEqual([obj.label for obj in objects], ["figure:1", "figure:2"])
            obj = objects[1]
            self.assertEqual(obj.bbox, [40, 240, 240, 320])
            self.assertEqual(obj.crop_version, hashlib.sha256(crop).hexdigest())
            self.assertEqual(updated[0].visual_object_ids, [obj.id])
            target = claim("p1").model_copy(update={"evidence_ids": [updated[0].id]})
            images, context, missing = verification_images(tasks, updated, target)
            self.assertEqual(missing, [])
            self.assertEqual(images, [tasks.visual_pages["p1", 1]["url"]])
            self.assertEqual(obj.source_version, context[0]["source_version"])
            tasks.previous_visual_evidence = [span.model_dump() for span in updated]
            again, audit = await prepare_visual(tasks, paper, [])
            self.assertEqual(again[0].id, updated[0].id)
            self.assertTrue(audit["pages"][0]["crop_transcriptions"][0]["reused"])
            tasks.question = "Summarize this paper"
            unrequested, _ = await prepare_visual(tasks, paper, [])
            self.assertFalse(unrequested)
            self.assertEqual(len(model.images), 1)

            tasks.question = "Read the y-axis label in Figure 2"
            with patch("src.research.visual.label_views", return_value={"270": "Layer Index"}) as local_read:
                oriented, _ = await prepare_visual(tasks, paper, [])
                local_read.assert_called_once()
            self.assertEqual(oriented[0].transcription_profile, "visual_labels")
            self.assertEqual(model.requests[-1]["oriented_label_hints"], {"270": "Layer Index"})

            # A two-panel figure with only one caption has no reliable single-raster boundary.
            stream.set_data(stream.get_data().replace(b"Figure 1:", b"Results:"))
            with path.open("wb") as out:
                writer.write(out)
            _, _, catalog = inspect_pdf(path, 1)
            self.assertEqual(catalog[0]["object_bboxes"], {})

    def test_two_labelled_panels_share_a_figure_only_with_matching_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panels.pdf"
            writer, stream = two_figure_pdf(path)
            image = next(iter(writer.pages[0]["/Resources"]["/XObject"]))
            body = f"""q 100 0 0 80 40 240 cm {image} Do Q
                q 100 0 0 80 160 240 cm {image} Do Q
                BT /HeaderFont 10 Tf 42 304 Td ((a)) Tj ET
                BT /HeaderFont 10 Tf 162 304 Td ((b)) Tj ET
                BT /HeaderFont 12 Tf 40 220 Td (Figure 1: (a) Accuracy and (b) Latency across runs) Tj ET"""
            for marker, expected in [("(b)", True), ("(c)", False)]:
                stream.set_data(body.replace("162 304 Td ((b))", f"162 304 Td ({marker})").encode())
                with path.open("wb") as out:
                    writer.write(out)
                pages, _, _ = inspect_pdf(path, 1)
                self.assertEqual(bool(pages[0]["panel_bboxes"]), expected)
                if expected:
                    self.assertEqual(pages[0]["object_bboxes"]["figure:1"], [40,240,260,320])
                    self.assertEqual(set(pages[0]["panel_bboxes"]["figure:1"]), {"a", "b"})
                    self.assertEqual(pages[0]["object_bboxes"]["figure:1b"], [160, 240, 260, 320])
                    self.assertEqual(labels("Inspect Figure 1(b)"), {"figure:1b"})

    def test_framed_vector_diagram_rejects_outside_tick_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vector.pdf"
            writer, stream = two_figure_pdf(path)
            body = """40 240 200 80 re S
                50 250 30 30 re S
                120 250 30 30 re S
                BT /HeaderFont 12 Tf 40 214 Td (Figure 4: Vector diagram of the training process) Tj ET"""
            for extra, expected in [("", True), (" BT /HeaderFont 8 Tf 244 265 Td (ms) Tj ET", False)]:
                stream.set_data((body + extra).encode())
                with path.open("wb") as out:
                    writer.write(out)
                pages, _, _ = inspect_pdf(path, 1)
                self.assertEqual("figure:4" in pages[0]["pixel_bboxes"], expected)

    async def test_rotation_keeps_crop_and_native_text_in_the_same_pdf_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.pdf"
            writer, _ = two_figure_pdf(path)
            writer.pages[0].rotate(90)
            with path.open("wb") as out:
                writer.write(out)
            paper = item()
            paper["full_text"]["source_path"] = str(path)
            tasks = ModelTasks(VisualModel(), ResearchOptions())
            tasks.question = "Figure 2?"
            source = EvidenceSpan(id="caption", version="v", paper_id="p1", source_level="full_text",
                page_start=1, text="Figure 2: Second experiment latency")
            spans, audit = await prepare_visual(tasks, paper, [source])
            self.assertFalse(audit["warnings"])
            crop = base64.b64decode(tasks.llm.images[0][0]["image_url"]["url"].split(",")[1])
            with Image.open(io.BytesIO(crop)) as image:
                self.assertEqual(image.size, (160, 400))
            caption, transcription = spans
            self.assertEqual(caption.reading_order, transcription.reading_order)
            self.assertEqual(caption.source_version, transcription.source_version)
            self.assertLess(caption.bbox[3], transcription.bbox[1])
            self.assertEqual(transcription.bbox, [40, 240, 240, 320])

    async def test_precision_gap_selects_latency_crop_without_a_figure_number(self):
        from src.research.planning import build_reading_plan
        from src.research.selection import build_evidence_packets
        from src.research.visual import Transcription
        from src.research.models import VerificationDraft

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.pdf"
            two_figure_pdf(path)
            paper = item()
            paper["full_text"]["source_path"] = str(path)
            tasks = ModelTasks(VisualModel(), ResearchOptions())
            tasks.question = "论文的加速是否来自更低精度？"
            plan = build_reading_plan(tasks.question, "p1", "Latency experiment")
            tasks.ask = AsyncMock(return_value=Transcription(readable=True,
                text="Method: FP16, 10 ms. Baseline: 14 ms."))
            evidence, audit = await prepare_visual(tasks, paper, [], plan)
            tasks.ask.assert_awaited_once()
            self.assertEqual(tasks.ask.await_args.args[2]["object_label"], "figure:2")
            self.assertEqual(audit["pages"][0]["crop_transcriptions"][0]["trigger"], "requirement_gap")
            scope = ScopedEvidence(evidence, {"p1"})
            packets = await build_evidence_packets(scope, plan, None)
            packet = next(p for p in packets if p.question == plan.user_questions[0])
            self.assertIn("precision", packet.covered_requirements)
            self.assertIn("baseline_precision", packet.missing_requirements)
            eid = evidence[0].id
            target = Claim(id="c", version="v", paper_id="p1", kind="result",
                attribution="system_inference", statement="The speedup comes from precision.",
                evidence_ids=[eid], subquery_ids=[packet.subquery_id])
            tasks.ask = AsyncMock(return_value=VerificationDraft(verdict="verified", reason="reported",
                evidence_ids=[eid], fidelity={"status": "supported", "evidence_ids": [eid]},
                experimental_support={"status": "supported", "evidence_ids": [eid]},
                visual_observations=[{"paper_id": "p1", "page": 1, "image_version": evidence[0].image_version,
                    "locator": "Figure 2", "observation": "method row", "status": "supports"}]))
            result = await verify(tasks, target, scope, evidence_packets=packets)
            self.assertTrue(tasks.ask.await_args.args[2]["precision_comparison_required"])
            self.assertEqual(result.verdict, "qualified")
            self.assertIn("精度", result.repair_request)

            # When both settings are explicit in native text, skip the visual model.
            native = EvidenceSpan(id="native", paper_id="p1", version="v", source_level="full_text",
                text="Method uses FP16. Baseline uses FP32. Latency result improves.")
            tasks.ask.reset_mock()
            _, audit = await prepare_visual(tasks, paper, [native], plan)
            self.assertEqual(audit["visual_gaps"], [])
            tasks.ask.assert_not_awaited()


    async def test_mixed_scan_with_long_header_is_transcribed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'mixed.pdf';scanned_pdf(path)
            add_header(path,'A searchable running header that contains more than forty characters')
            self.assertGreater(len(PdfReader(path).pages[0].extract_text()),40)
            pages,_,_=inspect_pdf(path,1)
            self.assertTrue(pages[0]['scanned'])
            paper=item();paper['full_text']['source_path']=str(path)
            evidence,_=await prepare_visual(ModelTasks(VisualModel(),ResearchOptions()),paper,[])
            self.assertEqual(evidence[0].modality,'visual_transcription')

    def test_requested_caption_has_priority_over_page_order(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'two.pdf';scanned_pdf(path,2)
            add_header(path,'Table 9: ablation accuracy',1)
            pages,omitted,_=inspect_pdf(path,1,'Check Table 9')
            self.assertEqual(pages[0]['page'],2)
            self.assertIn(1,omitted)

    async def test_image_without_observation_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scan.pdf';scanned_pdf(path)
            paper=item();paper['full_text']['source_path']=str(path)
            tasks=ModelTasks(VisualModel(),ResearchOptions())
            evidence,_=await prepare_visual(tasks,paper,[])
            class MissingObservation(VisualModel):
                async def chat(self,messages,**kwargs):
                    response=await super().chat(messages,**kwargs)
                    value=json.loads(response.content);value.pop('visual_observations',None)
                    return LLMResponse(content=json.dumps(value))
            tasks.llm=MissingObservation()
            c=claim('p1').model_copy(update={'evidence_ids':[evidence[0].id]})
            result=await verify(tasks,c,ScopedEvidence(evidence,{'p1'}))
            self.assertEqual(result.verdict,'needs_human_review')
            self.assertIn('已附原图但缺少逐页可定位的视觉观察',result.unchecked)

    def test_reference_page_resolves_to_actual_caption_page(self):
        tasks=ModelTasks(FakeModel(),ResearchOptions())
        tasks.visual_catalog={'p':[{'page':1,'captions':[],'visual_content':False},{'page':3,'captions':['figure:2'],'visual_content':True}]}
        tasks.visual_pages={('p',3):{'url':'image','image_version':'i','source_version':'s'}}
        evidence=EvidenceSpan(id='e',paper_id='p',version='v',text='See Figure 2 for the result.',source_level='full_text',page_start=1)
        images,context,missing=verification_images(tasks,[evidence],claim('p').model_copy(update={'evidence_ids':['e']}))
        self.assertEqual(images,['image']);self.assertEqual(context[0]['page'],3);self.assertFalse(missing)

    async def test_visual_budget_leaves_pages_for_later_papers(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'scan.pdf';scanned_pdf(path,14)
            tasks=ModelTasks(VisualModel(),ResearchOptions(visual_page_limit=20))
            tasks.remaining_visual_papers=2
            p1=item();p1['full_text']['source_path']=str(path)
            _,first=await prepare_visual(tasks,p1,[])
            tasks.remaining_visual_papers=1
            p2=item('p2');p2['full_text']['source_path']=str(path)
            _,second=await prepare_visual(tasks,p2,[])
            self.assertEqual(len(first['pages']),12)
            self.assertEqual(len(second['pages']),12)
            self.assertEqual(len(tasks.visual_pages),24)


class RelationWritingTests(unittest.IsolatedAsyncioTestCase):
    async def test_writer_receives_relation_and_paragraph_binds_its_version(self):
        a,b=claim('a'),claim('b');r=relation(a,b);v=check(r)
        model=FakeModel();tasks=ModelTasks(model,ResearchOptions())
        paragraphs=await write_report(tasks,'compare',[a,b],relations=[r],relation_checks=[v])
        self.assertEqual(model.requests[0]['relations'][0]['relation'],'conflicts')
        self.assertEqual(paragraphs[0].relation_ids,[r.id])
        self.assertIn(r.id,paragraphs[0].relation_context)

    async def test_unverified_relations_never_enter_writer(self):
        a,b=claim('a'),claim('b');r=relation(a,b);v=check(r).model_copy(update={'verdict':'insufficient_evidence'})
        model=FakeModel()
        paragraphs=await write_report(ModelTasks(model,ResearchOptions()),'Q',[a,b],relations=[r],relation_checks=[v])
        self.assertFalse(model.requests[0]['relations']);self.assertFalse(paragraphs[0].relation_ids)

    async def test_unknown_relation_and_missing_endpoint_are_rejected(self):
        a,b=claim('a'),claim('b');r=relation(a,b)
        class InvalidWriter(FakeModel):
            async def chat(self,messages,**kwargs):
                return LLMResponse(content=json.dumps({'paragraphs':[{'text':'claim','claim_ids':[a.id],'relation_ids':[r.id]}]}))
        with self.assertRaisesRegex(ValueError,'双方主张'):
            await write_report(ModelTasks(InvalidWriter(),ResearchOptions()),'Q',[a,b],relations=[r],relation_checks=[check(r)])
        with self.assertRaisesRegex(ValueError,'未知'):
            await write_report(ModelTasks(InvalidWriter(),ResearchOptions()),'Q',[a,b])

    async def test_relation_change_invalidates_only_connected_paragraphs(self):
        a,b,c=claim('a'),claim('b'),claim('c');r=relation(a,b);v=check(r)
        paragraphs=await write_report(ModelTasks(FakeModel(),ResearchOptions()),'Q',[a,b],relations=[r],relation_checks=[v])
        independent=await write_report(ModelTasks(FakeModel(),ResearchOptions()),'Q',[c])
        cards=[PaperCard(paper_id=x.paper_id,title=x.paper_id,version='v',claims=[x],verifications=[check(x)]) for x in (a,b,c)]
        parent={'execution_policy':'p','cards':[x.model_dump() for x in cards],'evidence':[],
                'paragraphs':[{'paragraph':p.model_dump(),'verification':check(p).model_dump()} for p in [*paragraphs,*independent]]}
        plan=IncrementalPlan(parent,[],'p');plan.bind_cards(cards)
        changed=r.model_copy(update={'version':'new','relation':'supports'})
        retained=plan.paragraphs([a,b,c],[changed],[check(changed)])
        self.assertEqual(len(retained),1);self.assertEqual(retained[0]['paragraph']['claim_ids'],[c.id])


class ReadingScoringSchedulingTests(unittest.IsolatedAsyncioTestCase):
    def test_question_claim_does_not_block_reproducibility_score(self):
        c,q=claim('p'),claim('p','question','question')
        scored=check(c,dimensions=[{'dimension':'reproducibility','score':4,'reason':'Code, data and configuration reported','evidence_ids':c.evidence_ids}])
        card=PaperCard(paper_id='p',title='P',version='v',claims=[c,q],verifications=[scored,check(q)])
        assessment=assess_card(card,ResearchOptions())
        dimension=next(d for d in assessment['dimensions'] if d['dimension']=='reproducibility')
        self.assertEqual(dimension['score'],4);self.assertEqual(dimension['excluded_claim_ids'],[q.id])
        card.omitted_evidence_ids=['unread-page']
        self.assertIsNone(next(d for d in assess_card(card,ResearchOptions())['dimensions'] if d['dimension']=='reproducibility')['score'])

    async def test_reader_gets_prior_assumptions_without_other_paper_leakage(self):
        spans=[EvidenceSpan(id='p:e',paper_id='p',version='v',text='Result improves.',source_level='full_text'),
               EvidenceSpan(id='assumption',paper_id='p',version='v',text='We assume independent samples.',source_level='full_text'),
               EvidenceSpan(id='foreign',paper_id='other',version='v',text='We assume secret.',source_level='full_text')]
        model=FakeModel()
        await read_claims(ModelTasks(model,ResearchOptions()),'Q','p',ScopedEvidence(spans,{'p'}),['p:e'])
        self.assertEqual([e['id'] for e in model.requests[0]['context_evidence']],['assumption'])

    async def test_reader_reports_omissions_and_rejects_out_of_scope_gap(self):
        e=EvidenceSpan(id='p:e',paper_id='p',version='v',text='Result',source_level='full_text')
        class Gaps(FakeModel):
            gap='p:e'
            async def chat(self,messages,**kwargs):
                return LLMResponse(content=json.dumps({'claims':[],'uncovered_evidence_ids':[self.gap]}))
        tasks=ModelTasks(Gaps(),ResearchOptions())
        await read_claims(tasks,'Q','p',ScopedEvidence([e],{'p'}),['p:e'])
        self.assertEqual(tasks.reading_gaps['p'],{'p:e'})
        tasks.llm.gap='foreign'
        with self.assertRaises(ValueError):
            await read_claims(tasks,'Q','p',ScopedEvidence([e],{'p'}),['p:e'])

    def test_exact_duplicates_merge_sources_but_different_conditions_stay_separate(self):
        first=claim('p');second=first.model_copy(update={'id':'other','evidence_ids':['p:second']})
        spans=[EvidenceSpan(id=eid,paper_id='p',version='v',text='Result',source_level='full_text') for eid in ['p:e','p:second']]
        scope=ScopedEvidence(spans,{'p'})
        merged=merge_duplicate(first,second,scope)
        self.assertEqual(merged.evidence_ids,['p:e','p:second'])
        self.assertIsNone(merge_duplicate(first,second.model_copy(update={'conditions':'different population'}),scope))

    def test_comparisons_are_prioritized_and_round_robin_across_papers(self):
        cards=[PaperCard(paper_id=p,title=p,version='v',claims=[claim(p,'1'),claim(p,'2')]) for p in ['a','b','c']]
        for card in cards:
            card.claims[1].study.problem='same problem'
        scheduled=list(comparison_schedule(cards))
        self.assertEqual(len(scheduled),12)
        self.assertEqual({(a.paper_id,b.paper_id) for a,b,_,_ in scheduled[:3]},{('a','b'),('a','c'),('b','c')})
        self.assertTrue(all(first.id.endswith('2') and second.id.endswith('2') for _,_,first,second in scheduled[:3]))


class EvaluationAndChallengeTests(unittest.IsolatedAsyncioTestCase):
    def test_balanced_cases_penalize_constant_abstention_and_are_not_real_papers(self):
        sources,cases=load_dataset('evaluation/controlled-cases.json')
        predictions=[{'case_id':c.id,'case_version':case_version(c,sources),'labels':{'relation':'incomparable','experimental_support':'unknown'}} for c in cases]
        result=score(sources,cases,predictions)
        self.assertTrue(result['dataset_audit']['balanced_coverage'])
        self.assertEqual(result['controlled_fixture']['decisive_accuracy'],0)
        self.assertEqual(result['controlled_fixture']['unnecessary_abstentions'],7)
        self.assertEqual(result['reference_draft']['cases'],0)
        self.assertEqual(result['expert_consensus']['cases'],0)
        _,real=load_dataset('evaluation/real-papers.json')
        self.assertFalse(dataset_audit(real)['balanced_coverage'])

    async def test_full_workflow_runner_does_not_supply_gold_claims_or_labels(self):
        sources,cases=load_dataset('evaluation/controlled-cases.json')
        model=FakeModel()
        result=await run_workflow_cases(sources,cases[:1],model)
        self.assertTrue({'paper_reader','evidence_verifier','cross_paper_analyst','writer'} <= {p['role'] for p in model.requests})
        self.assertIsNone(result[0]['semantic_accuracy'])
        self.assertTrue(result[0]['report']['paragraphs'])
        self.assertFalse(any('reference' in p or 'rationale' in p for p in model.requests))
        with self.assertRaisesRegex(ValueError,'尚未与人工标签对齐'):
            score(sources,cases,result)

    async def test_external_search_focuses_on_actual_claim_challenge(self):
        model=FakeModel();task=state(model,[item()],cross_paper_enabled=True,allow_external_search=True)
        service=SimpleNamespace(async_search=AsyncMock(return_value=SimpleNamespace(errors={},papers=[])))
        task['research_search_service']=service
        report=(await run_research_node()(task))['research_report']
        query=service.async_search.call_args.kwargs['query']
        self.assertIn('有效',query)
        self.assertTrue(report['external_search']['challenged_claim_ids'])
        self.assertTrue(report['challenges'])

    async def test_fabricated_challenge_citations_are_rejected(self):
        class BadChallenge(FakeModel):
            async def chat(self,messages,**kwargs):
                response=await super().chat(messages,**kwargs);value=json.loads(response.content)
                value['challenges']=[{'premise':'P','alternative_explanation':'V','required_test':'test V','evidence_ids':['foreign']}]
                return LLMResponse(content=json.dumps(value))
        e=EvidenceSpan(id='p:e',paper_id='p',version='v',text='Result',source_level='full_text')
        result=await verify(ModelTasks(BadChallenge(),ResearchOptions()),claim('p'),ScopedEvidence([e],{'p'}))
        self.assertEqual(result.verdict,'insufficient_evidence')

class BoundaryRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_mutation_during_load_aborts_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.txt';path.write_text('original')
            paper=item();paper['full_text']['source_path']=str(path)
            def changing_source(_):
                path.write_text('changed while reading')
                return [EvidenceSpan(id='p1:e',paper_id='p1',version='v',text='original',source_level='full_text')]
            with patch('src.research.workflow.load_evidence',side_effect=changing_source):
                with self.assertRaisesRegex(ValueError,'读取期间原文发生变化'):
                    await run_research_node()(state(FakeModel(),[paper]))

    async def test_legacy_pdf_report_without_fingerprint_cannot_be_revised(self):
        with tempfile.TemporaryDirectory() as d:
            helper=SourceRevisionTests()
            _,repo,session,report,_=await helper.prepare_revision(d,'p1')
            latest=latest_report(repo,session.key)
            legacy=copy.deepcopy(report);legacy.pop('source_manifests')
            target=legacy['cards'][0]['claims'][0]
            with patch('src.services.research.latest_report',return_value={'artifact_id':latest['artifact_id'],'report':legacy}):
                with self.assertRaisesRegex(ValueError,'缺少原文件版本指纹'):
                    revision_request(repo,session.key,{'artifact_id':latest['artifact_id'],'claim_id':target['id'],
                        'target_version':target['version'],'reason':'scope','changes':{'statement':'changed'}})

    async def test_exhausted_budget_does_not_acquire_more_papers(self):
        tasks=ModelTasks(FakeModel(),ResearchOptions(cross_paper_enabled=True,allow_external_search=True,max_calls=1))
        tasks.calls=1
        service=SimpleNamespace(async_search=AsyncMock())
        task=state(FakeModel(),[item()],cross_paper_enabled=True,allow_external_search=True)
        task['research_search_service']=service
        _,audit=await supplement_sources(task,[item()],tasks)
        self.assertEqual(audit['status'],'budget_exhausted')
        service.async_search.assert_not_called()

    async def test_new_relation_invalidates_previously_independent_paragraph(self):
        a,b=claim('a'),claim('b')
        paragraphs=await write_report(ModelTasks(FakeModel(),ResearchOptions()),'Q',[a])
        cards=[PaperCard(paper_id=c.paper_id,title=c.paper_id,version='v',claims=[c],verifications=[check(c)]) for c in (a,b)]
        parent={'execution_policy':'p','evidence':[],'cards':[c.model_dump() for c in cards],
                'paragraphs':[{'paragraph':p.model_dump(),'verification':check(p).model_dump()} for p in paragraphs]}
        plan=IncrementalPlan(parent,[],'p');plan.bind_cards(cards)
        self.assertEqual(len(plan.paragraphs([a,b])),1)
        r=relation(a,b)
        self.assertEqual(plan.paragraphs([a,b],[r],[check(r)]),[])

    async def test_anchor_reservation_does_not_reject_a_maximum_size_source_span(self):
        span=EvidenceSpan(id='p1:e',paper_id='p1',version='v',text='x'*4000,source_level='full_text')
        with patch('src.research.workflow.load_evidence',return_value=[span]):
            report=(await run_research_node()(state(FakeModel(),research_context_chars=4000)))['research_report']
        self.assertTrue(report['cards'][0]['claims'])
        self.assertFalse(report['cards'][0]['omitted_evidence_ids'])
