import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from src.api.app import create_app

from src.graph.graph import run_graph
from src.repositories.sessions.sqlite import SQLiteSessionRepository
from src.research.reasoning import stage_versions
from src.research.storage import ResearchStore
from src.research.workflow import run_research_node
from src.services.research import latest_report, resume_review_request
from test.test_research_reasoning import ReasoningModel
from test.test_research_workflow import item, state


class InterruptFinalReview(ReasoningModel):
    interrupt = True

    async def chat(self, messages, **kwargs):
        payload = json.loads(messages[-1]['content'])
        if (self.interrupt and payload['role'] == 'reasoning_verifier'
                and payload['primary_paper_id'] == 'p1' and payload['synthesis']['reading_guide']):
            self.requests.append(payload)
            raise TimeoutError('模拟草稿落盘后的核查超时')
        return await super().chat(messages, **kwargs)


class ResearchRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_graph_and_change_only_one_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = SQLiteSessionRepository(storage_root=directory)
            session = repo.create().key
            model = InterruptFinalReview()
            task = state(model, papers=[item('p1'),item('p2')], research_reasoning_enabled=True)
            async def run(checkpoint):
                model.requests.clear()
                overrides = {'research_node_llm': model, 'read_resume_checkpoint': checkpoint}
                return (await run_graph(task['request'], session_repo=repo, session_key=session,
                                        state_overrides=overrides)).state['research_report']

            # Use the research node for the first call, then the normal graph resume entry.
            first = await run_research_node()({**task, 'session_repo':repo, 'session_key':session})
            pending = first['research_report']['reasoning'][0]
            self.assertEqual(pending['pending_review'], 'timeout')
            artifact_id = latest_report(repo, session)['artifact_id']
            client = TestClient(create_app(sessions_repo=repo))
            start = AsyncMock(return_value={'run_id':'recovery-test'})
            with patch('src.services.session_runs.SessionRunService.start_run', start):
                response = client.post(f'/api/sessions/{session}/research/resume-review',
                                       json={'artifact_id': artifact_id})
            self.assertEqual(response.status_code, 202)
            checkpoint = start.await_args.args[1]['read_resume_checkpoint']
            model.interrupt = False
            reviewed = await run(checkpoint)
            self.assertEqual([p['role'] for p in model.requests], ['reasoning_verifier'])
            self.assertEqual(reviewed['reasoning'][0]['draft_version'], pending['draft_version'])
            published = reviewed['reasoning'][0]
            self.assertEqual(published['status'], 'reviewed')
            with self.assertRaises(ValueError):
                resume_review_request(repo, session, {'artifact_id':artifact_id})

            again = await run(checkpoint)
            self.assertEqual(again['calls'], 0)
            self.assertEqual(again['reasoning'][0]['version'], published['version'])
            policies = stage_versions()
            with patch('src.research.reasoning.stage_versions', return_value={**policies,'publication':'new-rule'}):
                changed = await run(checkpoint)
                self.assertEqual(changed['calls'], 0)
                self.assertEqual(changed['reasoning'][0]['review_version'], published['review_version'])
            with patch('src.research.reasoning.stage_versions', return_value={**policies,'review':'new-standard'}):
                changed = await run(checkpoint)
                self.assertEqual([p['role'] for p in model.requests], ['reasoning_verifier'] * 2)
                self.assertNotEqual(changed['reasoning'][0]['review_version'], published['review_version'])
                self.assertEqual(changed['reasoning'][0]['draft_version'], pending['draft_version'])
            checkpoint['read_results'][0]['paper']['abstract'] += ' Updated experiment.'
            changed = await run(checkpoint)
            self.assertEqual([p['paper_id'] for p in model.requests if p['role'] == 'paper_reader'], ['p1'])
            self.assertEqual(changed['reasoning'][1]['version'], reviewed['reasoning'][1]['version'])
            self.assertNotEqual(changed['reasoning'][0]['draft_version'], pending['draft_version'])

    async def test_concurrent_attempts_accept_the_same_record(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = SQLiteSessionRepository(storage_root=directory)
            session = repo.create().key
            stores = [ResearchStore({'session_repo': repo, 'session_key': session, 'turn_id': str(i)}) for i in range(4)]
            attempts = await asyncio.gather(*(store.accept('reasoning_review', {'attempt': i}, 'same-version')
                                              for i, store in enumerate(stores)))
            self.assertTrue(all(value == attempts[0] for value in attempts))
            self.assertEqual(len(repo.get(session).artifacts), 1)

    async def test_prepared_paper_reports_material_and_failure_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)/'paper.md'
            md.write_text('Evidence from a controlled experiment.\n' * 10)
            for abstract, full_text, expected in [
                ('abstract', {}, 'abstract'),
                ('abstract', {'status':'parse_failed', 'reason':'PDF损坏'}, 'abstract'),
                ('', {'status':'parse_failed', 'reason':'PDF损坏'}, 'unavailable'),
                ('abstract', {'markdown_path':str(md), 'status':'parsed'}, 'full_text'),
            ]:
                paper = {'paper':{'id':'p1','title':'Method','abstract':abstract},'full_text':full_text}
                task = state(ReasoningModel(), papers=[paper])
                task['research_evidence_index'] = None
                report = (await run_research_node()(task))['research_report']
                prepared = report['prepared_papers'][0]
                self.assertEqual(prepared['material'], expected)
                self.assertEqual(prepared['failure_reason'], full_text.get('reason',''))
                self.assertEqual(prepared['evidence_ids'], [e['id'] for e in report['evidence']])
