import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from debug_agent_system.core.contracts import CheckNode, LockedSubgraph
from debug_agent_system.eval.debug_sim.build_broad_debug_scenarios import EXCLUDED_RAW_DIRS
from debug_agent_system.eval.debug_sim.chat_replay import render_markdown
from debug_agent_system.eval.debug_sim.gate import evaluate_gate
from debug_agent_system.eval.debug_sim.runner import build_smoke_scenarios, run_one, write_run
from debug_agent_system.eval.debug_sim.sag_regression import run_regression
from debug_agent_system.eval.debug_sim.scenario_v2 import RequiredCheck, ScenarioV2, load_scenarios
from debug_agent_system.eval.debug_sim.scorer import score_case


def test_industrial_pc_boot_scenario_file_loads():
    scenarios = load_scenarios('data/eval/scenarios/industrial_pc_boot_v1.json')
    assert len(scenarios) == 11
    first = scenarios[0]
    assert first.case_id.startswith('IPC_BOOT_')
    assert first.query
    assert first.source == 'industrial_pc_boot_manual'
    assert first.target_error_id
    assert first.required_checks
    assert first.evidence_key_facts


def test_smoke_scenarios_use_store_error_enumeration_contract():
    class Store:
        errors = [{"error_id": "err:one", "label": "蓝屏", "symptom": "工控机蓝屏"}]

        @staticmethod
        def load_locked_subgraph(error_id: str) -> LockedSubgraph:
            assert error_id == "err:one"
            return LockedSubgraph(
                error_id=error_id,
                label="蓝屏",
                checks=[CheckNode(check_id="check:one", label="检查内存", how_to_check="检查内存")],
            )

    scenarios = build_smoke_scenarios(SimpleNamespace(store=Store()), 1)

    assert len(scenarios) == 1
    assert scenarios[0].target_error_id == "err:one"
    assert scenarios[0].required_checks[0].id == "check:one"


def test_broad_debug_scenario_file_is_debug_only_gate_set():
    scenarios = load_scenarios('data/eval/scenarios/broad_debug_v1.json')
    assert len(scenarios) == 150
    assert len({s.case_id for s in scenarios}) == 150
    assert all(s.query_type == 'debug' for s in scenarios)
    assert not any(s.query_type in {'operation', 'open', 'log'} for s in scenarios)
    assert not any((s.metadata or {}).get('gate_scope') == 'report_only' for s in scenarios)
    assert all(s.expected_status == 'step' for s in scenarios)
    assert all(s.target_error_id for s in scenarios)
    assert all(s.required_checks for s in scenarios)
    assert all(s.evidence_key_facts for s in scenarios)
    assert all(s.expected_resolution_facts for s in scenarios)
    source_types = {str(s.metadata.get('source_type') or '') for s in scenarios}
    assert {'SOP', 'FAQ', 'manual', 'tech_support', 'chunks'} <= source_types


def test_broad_debug_rejected_report_and_raw_gitignore_contracts():
    gitignore = Path('.gitignore').read_text(encoding='utf-8')
    assert 'data/raw/' in gitignore
    assert {'knowledge-graph', 'bge_index', 'lightrag_index', 'results'} <= set(EXCLUDED_RAW_DIRS)
    report = json.loads(Path('data/eval/scenarios/broad_debug_v1_rejected.json').read_text(encoding='utf-8'))
    assert set(report['meta']['excluded_raw_dirs']) == set(EXCLUDED_RAW_DIRS)
    assert isinstance(report.get('rejected'), list)
    assert all('reason' in row for row in report['rejected'])


def test_chat_replay_seed_file_loads_as_review_only_debug_cases():
    scenarios = load_scenarios('data/eval/scenarios/chat_replay_seed_v1.json')
    assert len(scenarios) == 8
    assert len({s.case_id for s in scenarios}) == 8
    assert all(s.query_type == 'debug' for s in scenarios)
    assert all(s.source == 'w1_chat_replay' for s in scenarios)
    assert all(s.metadata.get('gate_scope') == 'seed_review' for s in scenarios)
    assert all(isinstance(s.metadata.get('replay_truth'), dict) for s in scenarios)
    assert any((s.metadata['replay_truth'].get('failure_path') or []) for s in scenarios)
    assert any((s.metadata['replay_truth'].get('missing_info_requests') or []) for s in scenarios)


def test_scorer_detects_target_error_miss():
    scenario = ScenarioV2(
        case_id='target-miss',
        query='相机失败',
        target_error_id='err:expected',
        expected_status='step',
        required_checks=[],
    )
    transcript = {
        'case_id': 'target-miss',
        'query': '相机失败',
        'final_status': 'step',
        'turns': [
            {'actor': 'agent', 'response': {'status': 'step', 'answer': 'x', 'observability': {'top_error_id': 'err:wrong'}}}
        ],
    }
    score = score_case(scenario, transcript)
    assert score['target_error_acc'] == 0.0


def test_gate_ignores_missing_judge_score_but_fails_unsafe():
    baseline = {'summary': {'failed': 0, 'unsafe_action_rate': 0.0, 'terminal_ok_rate': 1.0, 'check_recall': 0.5, 'evidence_recall': 0.5, 'composite_gated': 0.5}, 'details': []}
    current = {'summary': {'failed': 0, 'unsafe_action_rate': 1.0, 'terminal_ok_rate': 1.0, 'check_recall': 0.5, 'evidence_recall': 0.5, 'composite_gated': 0.5, 'judge_score': None}, 'details': []}
    report = evaluate_gate(current, baseline)
    assert report['status'] == 'FAIL'
    assert any('unsafe_action_rate' in x for x in report['failures'])


def test_write_run_isolates_named_latest_from_latest_real():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        real_run = {'run_id': 'real_run', 'summary': {}, 'details': []}
        ask_run = {'run_id': 'ask_info_run', 'summary': {}, 'details': []}
        real_path = write_run(real_run, tmp_path, real=True)
        ask_path = write_run(ask_run, tmp_path, real=True, latest_name='latest_ask_info.txt')

        assert (tmp_path / 'latest_real.txt').read_text(encoding='utf-8').strip() == str(real_path)
        assert (tmp_path / 'latest_ask_info.txt').read_text(encoding='utf-8').strip() == str(ask_path)


def test_runner_uses_non_interactive_for_single_turn_step_eval():
    class StubSystem:
        def __init__(self):
            self.payloads = []

        def start(self, payload):
            self.payloads.append(payload)
            return {
                'session_id': payload['session']['session_id'],
                'status': 'step',
                'answer': 'check',
                'current_check_id': 'check:1',
                'current_check': 'check',
                'metadata': {'presented_check_ids': ['check:1']},
            }

        def step(self, session_id, user):
            raise AssertionError('single-turn step eval should not call step')

    system = StubSystem()
    scenario = ScenarioV2(case_id='single-step', query='q', expected_status='step', required_checks=[])
    run_one(system, scenario)
    assert system.payloads[0]['interactive'] is False


def test_runner_prefers_replay_check_result_reply():
    class StubSystem:
        def __init__(self):
            self.user_messages = []

        def start(self, payload):
            return {
                'session_id': payload['session']['session_id'],
                'status': 'step',
                'answer': '请检查BIOS中的SATA模式。',
                'current_check_id': 'check:sata',
                'current_check': '检查BIOS中的SATA模式',
                'observability': {'top_error_id': 'err:industrial-pc-blue-screen'},
                'metadata': {'retrieval_trace': {'candidate_paths': [{'error_id': 'err:industrial-pc-blue-screen'}]}},
            }

        def step(self, session_id, user):
            self.user_messages.append(user)
            return {'session_id': session_id, 'status': 'resolved', 'answer': '已记录恢复。'}

    scenario = ScenarioV2(
        case_id='replay-check',
        query='蓝屏',
        expected_status='resolved',
        target_error_id='err:industrial-pc-blue-screen',
        metadata={'replay_truth': {'check_results': [{
            'check_text': '检查BIOS中的SATA模式',
            'result_type': 'effective',
            'user_reply': '切回AHCI后恢复正常。',
        }]}},
    )
    system = StubSystem()
    transcript = run_one(system, scenario)
    assert system.user_messages == ['切回AHCI后恢复正常。']
    assert transcript['replay_events'][0]['kind'] == 'check_result'
    assert transcript['replay_events'][0]['result_type'] == 'effective'
    assert transcript['top_error_id'] == 'err:industrial-pc-blue-screen'
    assert transcript['retrieval_trace_present'] is True
    assert transcript['latency_ms'] is not None
    assert transcript['trace_digest']['candidate_scores'][0]['error_id'] == 'err:industrial-pc-blue-screen'


def test_runner_replays_missing_info_request():
    class StubSystem:
        def __init__(self):
            self.user_messages = []

        def start(self, payload):
            return {
                'session_id': payload['session']['session_id'],
                'status': 'ask_info',
                'answer': '请提供DMP文件进一步确认蓝屏原因。',
                'required_data': ['DMP文件'],
                'observability': {'top_error_id': 'err:industrial-pc-blue-screen'},
            }

        def step(self, session_id, user):
            self.user_messages.append(user)
            return {'session_id': session_id, 'status': 'step', 'answer': '等待DMP分析。'}

    scenario = ScenarioV2(
        case_id='replay-info',
        query='蓝屏',
        expected_status='ask_info',
        target_error_id='err:industrial-pc-blue-screen',
        metadata={'replay_truth': {'missing_info_requests': [{
            'slot': 'log_package',
            'question': '请提供DMP文件进一步确认哪个硬件或驱动导致蓝屏。',
            'provided_later': True,
        }]}},
    )
    system = StubSystem()
    transcript = run_one(system, scenario)
    assert 'DMP文件' in system.user_messages[0]
    assert transcript['replay_events'][0]['kind'] == 'missing_info_request'
    assert transcript['replay_events'][0]['slot'] == 'log_package'


def test_runner_marks_unmatched_replay_step_without_simulator_gap():
    class StubSystem:
        def start(self, payload):
            return {
                'session_id': payload['session']['session_id'],
                'status': 'step',
                'answer': '请先检查完全无关的项目。',
                'current_check_id': 'check:wrong',
                'current_check': '完全无关的项目',
                'observability': {'top_error_id': 'err:industrial-pc-blue-screen'},
            }

        def step(self, session_id, user):
            raise AssertionError('unmatched replay step should stop without stepping')

    scenario = ScenarioV2(
        case_id='replay-unmatched',
        query='蓝屏',
        expected_status='resolved',
        target_error_id='err:industrial-pc-blue-screen',
        metadata={'replay_truth': {'check_results': [{
            'check_text': '检查BIOS中的SATA模式',
            'result_type': 'effective',
            'user_reply': '切回AHCI后恢复正常。',
        }]}},
    )
    transcript = run_one(StubSystem(), scenario)
    assert transcript['simulator_gap'] is False
    assert transcript['replay_events'][0]['kind'] == 'replay_unmatched_step'


def test_scorer_chat_replay_metrics_cover_failure_path_and_trace():
    scenario = ScenarioV2(
        case_id='replay-score',
        query='蓝屏',
        expected_status='resolved',
        target_error_id='err:industrial-pc-blue-screen',
        required_checks=[],
        metadata={'replay_truth': {
            'check_results': [
                {'check_text': 'PE系统还原、修复C盘引导', 'result_type': 'ineffective', 'user_reply': '无效'},
                {'check_text': 'BIOS SATA模式调整', 'result_type': 'effective', 'user_reply': '恢复'},
            ],
            'failure_path': [{
                'failed_check_text': 'PE系统还原、修复C盘引导',
                'expected_next_check_text': '检查BIOS SATA模式',
            }],
        }},
    )
    transcript = {
        'case_id': 'replay-score',
        'query': '蓝屏',
        'expected_status': 'resolved',
        'final_status': 'resolved',
        'checks_presented': [],
        'first_check_id': '',
        'first_check_text': 'PE系统还原、修复C盘引导',
        'top_error_id': 'err:industrial-pc-blue-screen',
        'retrieval_trace_present': True,
        'latency_ms': 12.5,
        'replay_events': [
            {'kind': 'check_result', 'check_text': 'PE系统还原、修复C盘引导', 'result_type': 'ineffective', 'user_turn_index': 1},
            {'kind': 'check_result', 'check_text': 'BIOS SATA模式调整', 'result_type': 'effective', 'user_turn_index': 3},
        ],
        'turns': [
            {'actor': 'agent', 'response': {'status': 'step', 'current_check': 'PE系统还原、修复C盘引导'}},
            {'actor': 'user', 'content': '无效'},
            {'actor': 'agent', 'response': {'status': 'step', 'current_check': '检查BIOS SATA模式'}},
            {'actor': 'user', 'content': '恢复'},
            {'actor': 'agent', 'response': {'status': 'resolved', 'answer': '恢复'}},
        ],
    }
    score = score_case(scenario, transcript)
    assert score['top_error_acc'] == 1.0
    assert score['first_check_acc'] == 1.0
    assert score['effective_result_covered'] == 1.0
    assert score['failure_path_acc'] == 1.0
    assert score['trace_coverage'] == 1.0
    assert score['latency_ms'] == 12.5
    assert score['chat_replay_composite'] == 1.0
    assert score['failure_stage'] == 'ok'
    assert score['trace_diagnosis']['primary_cause'] == 'no_failure_detected'


def test_scorer_first_check_accepts_diagnostic_anchor_match():
    scenario = ScenarioV2(
        case_id='first-anchor',
        query='显卡驱动蓝屏',
        expected_status='step',
        target_error_id='err:industrial-pc-blue-screen',
        required_checks=[RequiredCheck(text='卸载并重装显卡驱动')],
        metadata={'replay_truth': {'check_results': [{
            'check_text': '必要时回退版本验证',
            'result_type': 'diagnostic_method',
            'user_reply': '已处理显卡驱动。',
        }]}},
    )
    transcript = {
        'case_id': 'first-anchor',
        'query': '显卡驱动蓝屏',
        'expected_status': 'step',
        'final_status': 'step',
        'checks_presented': [],
        'first_check_id': '',
        'first_check_text': '故障模块指向显卡驱动时专项排查显卡驱动与供电散热',
        'top_error_id': 'err:industrial-pc-blue-screen',
        'retrieval_trace_present': True,
        'latency_ms': 1.0,
        'replay_events': [],
        'turns': [{'actor': 'agent', 'response': {'status': 'step', 'current_check': '故障模块指向显卡驱动时专项排查显卡驱动与供电散热'}}],
    }
    score = score_case(scenario, transcript)
    assert score['first_check_acc'] == 1.0


def test_chat_replay_markdown_report_contains_case_and_latency():
    report = {
        'mode': 'report-only',
        'meta': {'config': 'config/debug_agent_system_sag.yaml', 'scenario_file': 'data/eval/scenarios/chat_replay_seed_v1.json'},
        'summary': {'n': 1, 'chat_replay_composite': 1.0, 'latency_ms': 10.0, 'failure_stage_counts': {'ranking': 1}},
        'details': [{
            'case_id': 'case-1',
            'top_error_id': 'err:x',
            'first_check_text': '检查项',
            'final_status': 'step',
            'chat_replay_composite': 1.0,
            'trace_coverage': 1.0,
            'latency_ms': 10.0,
            'chat_replay_notes': [],
            'trace_diagnosis': {'primary_stage': 'ranking', 'primary_cause': 'target_not_top', 'next_debug_action': '检查候选分数'},
            'trace_digest': {
                'candidate_scores': [{'error_id': 'err:x', 'final_rank': 1, 'final_score': 12.0}],
                'selected_check_trace': {'check_id': 'check:1', 'source_error_id': 'err:x', 'source_tier': 'A'},
            },
        }],
        'transcripts': [{'case_id': 'case-1', 'replay_events': [{'kind': 'check_result', 'result_type': 'effective', 'check_text': '检查项', 'reply': '恢复'}]}],
    }
    md = render_markdown(report)
    assert '# Chat Replay Eval Report' in md
    assert 'case-1' in md
    assert 'Latency' in md
    assert 'Failure Diagnosis' in md
    assert 'ranking:target_not_top' in md


def test_scorer_diagnoses_target_error_miss_as_retrieval_failure():
    scenario = ScenarioV2(
        case_id='trace-diag-target-miss',
        query='相机失败',
        target_error_id='err:expected',
        expected_status='step',
        required_checks=[],
    )
    transcript = {
        'case_id': 'trace-diag-target-miss',
        'query': '相机失败',
        'final_status': 'step',
        'top_error_id': 'err:wrong',
        'first_check_id': 'check:wrong',
        'first_check_text': '错误检查项',
        'retrieval_trace_present': True,
        'trace_digest': {
            'candidate_ids': ['err:wrong'],
            'store_candidate_ids': ['err:wrong'],
            'candidate_scores': [{'error_id': 'err:wrong', 'final_rank': 1, 'final_score': 9.0}],
            'selected_check_trace': {'check_id': 'check:wrong', 'source_error_id': 'err:wrong', 'source_tier': 'B'},
        },
        'turns': [
            {'actor': 'agent', 'response': {'status': 'step', 'answer': 'x', 'current_check_id': 'check:wrong', 'current_check': '错误检查项', 'observability': {'top_error_id': 'err:wrong'}}}
        ],
    }
    score = score_case(scenario, transcript)
    assert score['failure_stage'] == 'retrieval'
    assert score['failure_cause'] == 'target_absent_from_candidates'


def test_sag_regression_detail_includes_trace_diagnosis():
    report = run_regression(
        config='config/debug_agent_system_sag.yaml',
        scenario_file='data/eval/scenarios/sag_regression_v1.json',
    )
    assert report['details']
    first = report['details'][0]
    assert 'trace_diagnosis' in first
    assert 'trace_digest' in first
    assert 'failure_stage' in first
