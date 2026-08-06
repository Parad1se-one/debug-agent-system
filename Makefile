.PHONY: test test-read-tool-harness test-write-trace-harness test-w7-multi-agent smoke eval eval-real eval-real-gate gate-real baseline-real generate-broad-debug-eval eval-broad-debug baseline-broad-debug eval-broad-debug-gate gate-broad-debug generate-ask-info-eval eval-ask-info eval-ask-info-gate sag-build sag-build-legacy eval-sag-retrieval eval-sag-regression eval-json-real-compare eval-json-broad-debug-compare eval-sag-real eval-sag-broad-debug gate-json-real-compare gate-json-broad-debug gate-sag-real gate-sag-broad-debug sag-comparison-report sag-comparison eval-chat-replay eval-write-rollback-audit w7-release-gate-build w7-human-review-init w7-human-review-serve w7-human-review-validate w7-human-review-export w7-multi-agent-shadow w7-multi-agent-session-shadow w7-multi-agent-calibration-build w7-multi-agent-calibration-run w7-multi-agent-calibration-score w7-multi-agent-fixed173-run w7-multi-agent-fixed173-gate w7-multi-agent-acceptance deepseek-trace-assembly-harness eval-write-manual-golden eval-write-batch-gate eval-conditional-branch coupling-scan w2-family-diagnostics w2-quality-diagnostics w2-quality-gate w2-postrun-report w2-postrun-compare w2-run-status w2-live-report kg-v2-build-curated kg-v2-overview kg-v2-terminology-build kg-v2-terminology-check kg-v2-terminology-review-build kg-v2-terminology-review-apply kg-v2-noun-discovery-build kg-v2-noun-discovery-apply gold-annotations-render gold-v1-verify gold-v2-verify gold-v1-ingest-kg-v2 gold-v1-standard-ingest gold-v1-baseline gold-v1-prompt-preview gold-v1-prompt-replay gold-v1-prompt-baseline blind-011-015-inputs blind-011-015-review blind-011-015-w1-baseline blind-011-015-prompt-preview blind-011-015-deepseek-pro blind-011-015-score-pro xing-candidate-library xing-heldout-probe xing-heldout-freeze build-kg-v2-read-eval validate-kg-v2-read-eval eval-kg-v2-read build-aoi-debug-benchmark validate-aoi-debug-benchmark score-aoi-debug-benchmark build-aoi-fae-report-benchmark validate-aoi-fae-report-benchmark build-aoi-document-qa-pilot validate-aoi-document-qa-pilot build-aoi-document-qa-extended validate-aoi-document-qa-extended formal-debug-benchmark-v1 validate-formal-debug-benchmark-v1 release-check-formal-debug-benchmark-v1 run-formal-debug-benchmark-v1-validation run-formal-debug-benchmark-v1-test score-formal-debug-benchmark-v1-validation score-formal-debug-benchmark-v1-test

GOLD_PROMPT_RESPONSES ?= data/results/gold-v1-prompt-responses.json
KG_V2_READ_EVAL ?= data/eval/scenarios/kg_v2_quality_v1.json
KG_V2_READ_EVAL_REPORT ?= data/eval/scenarios/kg_v2_quality_v1.report.json
KG_V2_READ_REPLAY ?= data/results/kg_v2_read_eval/latest.json
AOI_DEBUG_BENCHMARK ?= data/eval/benchmark/aoi_debug_benchmark_v1.json
AOI_DEBUG_BENCHMARK_REPORT ?= data/eval/benchmark/aoi_debug_benchmark_v1.report.json
AOI_DEBUG_BENCHMARK_MARKDOWN ?= docs/AOI_Debug_Benchmark_v1_Query与答案.md
AOI_DEBUG_BENCHMARK_PREDICTIONS ?= data/results/aoi_debug_benchmark/predictions.json
AOI_DEBUG_BENCHMARK_SCORE ?= data/results/aoi_debug_benchmark/latest_score.json
AOI_FAE_REPORT_BENCHMARK ?= data/eval/benchmark/aoi_fae_report_benchmark_v2.json
AOI_FAE_REPORT_BENCHMARK_REPORT ?= data/eval/benchmark/aoi_fae_report_benchmark_v2.report.json
AOI_FAE_REPORT_BENCHMARK_MARKDOWN ?= docs/AOI_FAE_Report_Benchmark_v2_Query与答案.md
AOI_DOCUMENT_QA_PILOT ?= data/eval/benchmark/aoi_document_qa_pilot_v1.json
AOI_DOCUMENT_QA_PILOT_REPORT ?= data/eval/benchmark/aoi_document_qa_pilot_v1.report.json
AOI_DOCUMENT_QA_PILOT_MARKDOWN ?= docs/AOI_Document_QA_Pilot_v1_Query与答案.md
AOI_DOCUMENT_QA_EXTENDED ?= data/eval/benchmark/aoi_document_qa_extended_v1.json
AOI_DOCUMENT_QA_EXTENDED_REPORT ?= data/eval/benchmark/aoi_document_qa_extended_v1.report.json
AOI_DOCUMENT_QA_EXTENDED_MARKDOWN ?= docs/AOI_Document_QA_Extended_v1_Query与答案.md
FORMAL_DEBUG_PREDICTIONS ?= data/eval/formal_debug_benchmark_v1/predictions.template.json
FORMAL_DEBUG_SCORE ?= data/results/formal_debug_benchmark_v1/latest_score.json
FORMAL_DEBUG_RUN_ROOT ?= data/results/formal_debug_benchmark_v1/latest_run
FORMAL_DEBUG_MODEL ?= gpt-5.6-luna
FORMAL_DEBUG_RUNTIME ?= codex_cli
XING_HELDOUT_SOURCE ?= data/results/xing_relation_context_final_20260717/messages.jsonl
XING_CANDIDATE_LIBRARY ?= data/annotations/goldcases/candidates/xing-lark-v1
XING_HELDOUT_OUTPUT ?= data/annotations/goldcases/heldout-021-025

kg-v2-terminology-build:
	PYTHONPATH=src python3 scripts/build_debug_terminology.py

kg-v2-terminology-check:
	PYTHONPATH=src python3 scripts/build_debug_terminology.py --check

kg-v2-terminology-review-build:
	PYTHONPATH=src python3 scripts/manage_debug_terminology_review.py

kg-v2-terminology-review-apply:
	PYTHONPATH=src python3 scripts/manage_debug_terminology_review.py --apply-approved

kg-v2-noun-discovery-build:
	PYTHONPATH=src python3 scripts/manage_debug_terminology_review.py --discover-nouns

kg-v2-noun-discovery-apply:
	PYTHONPATH=src python3 scripts/manage_debug_terminology_review.py --apply-approved-nouns

build-aoi-debug-benchmark:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.unified_benchmark \
	  --out $(AOI_DEBUG_BENCHMARK) \
	  --report-out $(AOI_DEBUG_BENCHMARK_REPORT) \
	  --markdown-out $(AOI_DEBUG_BENCHMARK_MARKDOWN)

validate-aoi-debug-benchmark:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.unified_benchmark \
	  --validate-only \
	  --out $(AOI_DEBUG_BENCHMARK) \
	  --report-out $(AOI_DEBUG_BENCHMARK_REPORT)

score-aoi-debug-benchmark: validate-aoi-debug-benchmark
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.unified_benchmark \
	  --validate-only \
	  --out $(AOI_DEBUG_BENCHMARK) \
	  --report-out $(AOI_DEBUG_BENCHMARK_REPORT) \
	  --score $(AOI_DEBUG_BENCHMARK_PREDICTIONS) \
	  --score-out $(AOI_DEBUG_BENCHMARK_SCORE)

build-aoi-fae-report-benchmark:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.fae_report_benchmark \
	  --out $(AOI_FAE_REPORT_BENCHMARK) \
	  --report-out $(AOI_FAE_REPORT_BENCHMARK_REPORT) \
	  --markdown-out $(AOI_FAE_REPORT_BENCHMARK_MARKDOWN)

validate-aoi-fae-report-benchmark:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.fae_report_benchmark \
	  --validate-only \
	  --out $(AOI_FAE_REPORT_BENCHMARK) \
	  --report-out $(AOI_FAE_REPORT_BENCHMARK_REPORT)

build-aoi-document-qa-pilot:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.document_qa_benchmark \
	  --out $(AOI_DOCUMENT_QA_PILOT) \
	  --report-out $(AOI_DOCUMENT_QA_PILOT_REPORT) \
	  --markdown-out $(AOI_DOCUMENT_QA_PILOT_MARKDOWN)

validate-aoi-document-qa-pilot:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.document_qa_benchmark \
	  --validate-only \
	  --out $(AOI_DOCUMENT_QA_PILOT) \
	  --report-out $(AOI_DOCUMENT_QA_PILOT_REPORT)

build-aoi-document-qa-extended:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.document_qa_extended_benchmark \
	  --out $(AOI_DOCUMENT_QA_EXTENDED) \
	  --report-out $(AOI_DOCUMENT_QA_EXTENDED_REPORT) \
	  --markdown-out $(AOI_DOCUMENT_QA_EXTENDED_MARKDOWN)

validate-aoi-document-qa-extended:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.document_qa_extended_benchmark \
	  --validate-only \
	  --out $(AOI_DOCUMENT_QA_EXTENDED) \
	  --report-out $(AOI_DOCUMENT_QA_EXTENDED_REPORT)

formal-debug-benchmark-v1:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.formal_debug_benchmark

validate-formal-debug-benchmark-v1:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.formal_debug_benchmark \
	  --validate-only

release-check-formal-debug-benchmark-v1:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.formal_debug_benchmark \
	  --validate-only --release-check

run-formal-debug-benchmark-v1-validation: formal-debug-benchmark-v1
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.formal_debug_runner \
	  --split validation \
	  --runtime $(FORMAL_DEBUG_RUNTIME) \
	  --model $(FORMAL_DEBUG_MODEL) \
	  --run-root $(FORMAL_DEBUG_RUN_ROOT) \
	  --score-out $(FORMAL_DEBUG_SCORE)

run-formal-debug-benchmark-v1-test: formal-debug-benchmark-v1
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.formal_debug_runner \
	  --split held_out_test --allow-held-out-test \
	  --runtime $(FORMAL_DEBUG_RUNTIME) \
	  --model $(FORMAL_DEBUG_MODEL) \
	  --run-root $(FORMAL_DEBUG_RUN_ROOT) \
	  --score-out $(FORMAL_DEBUG_SCORE)

score-formal-debug-benchmark-v1-validation: validate-formal-debug-benchmark-v1
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.formal_debug_benchmark \
	  --validate-only \
	  --predictions $(FORMAL_DEBUG_PREDICTIONS) \
	  --score-out $(FORMAL_DEBUG_SCORE) \
	  --split validation

score-formal-debug-benchmark-v1-test: validate-formal-debug-benchmark-v1
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.formal_debug_benchmark \
	  --validate-only \
	  --predictions $(FORMAL_DEBUG_PREDICTIONS) \
	  --score-out $(FORMAL_DEBUG_SCORE) \
	  --split held_out_test --allow-held-out-test

build-kg-v2-read-eval:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.kg_v2_quality_dataset \
	  --out $(KG_V2_READ_EVAL) \
	  --report-out $(KG_V2_READ_EVAL_REPORT)

validate-kg-v2-read-eval:
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.kg_v2_quality_dataset \
	  --validate-only \
	  --out $(KG_V2_READ_EVAL) \
	  --report-out $(KG_V2_READ_EVAL_REPORT)

eval-kg-v2-read: validate-kg-v2-read-eval
	PYTHONPATH=src python3 -m debug_agent_system.eval.read_side.kg_v2_quality_replay \
	  --dataset $(KG_V2_READ_EVAL) \
	  --out $(KG_V2_READ_REPLAY)

xing-candidate-library:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.build_xing_lark_candidate_library \
	  --source $(XING_HELDOUT_SOURCE) \
	  --out $(XING_CANDIDATE_LIBRARY)

xing-heldout-probe:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.freeze_xing_lark_heldout \
	  --source $(XING_HELDOUT_SOURCE) \
	  --library $(XING_CANDIDATE_LIBRARY)/candidates.json \
	  --probe

xing-heldout-freeze:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.freeze_xing_lark_heldout \
	  --source $(XING_HELDOUT_SOURCE) \
	  --library $(XING_CANDIDATE_LIBRARY)/candidates.json \
	  --out $(XING_HELDOUT_OUTPUT)

blind-011-015-inputs:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.blind_gold_set \
	  --root data/annotations/goldcases/review-v3 \
	  --manifest-name gold-011-015-review-v3.manifest.json

blind-011-015-review:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.render_blind_ground_truth_review

blind-011-015-w1-baseline:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.blind_011_015_w1_baseline

blind-011-015-prompt-preview:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.blind_011_015_prompt_preview

blind-011-015-deepseek-pro:
	set -a; . ./.env.local; set +a; \
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.blind_011_015_deepseek_pro --workers 2

blind-011-015-score-pro:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.score_blind_011_015_deepseek \
	  --prediction data/results/blind_runs/gold-011-015-review-v3/deepseek-v4-pro-two-stage-v4/predictions.json \
	  --prediction-manifest data/results/blind_runs/gold-011-015-review-v3/deepseek-v4-pro-two-stage-v4/predictions.manifest.json \
	  --out data/results/blind_runs/gold-011-015-review-v3/deepseek-v4-pro-two-stage-v4/score.json \
	  --md-out data/results/blind_runs/gold-011-015-review-v3/deepseek-v4-pro-two-stage-v4/score.md

gold-v1-verify:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.gold_set

gold-v2-verify:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.gold_v2_set

gold-annotations-render:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.render_gold_v1_annotations
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.render_blind_ground_truth_review

gold-v1-ingest-kg-v2: gold-v1-verify
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.ingest_gold_v1_to_kg_v2 \
	  --apply --authorization "user_request:2026-07-21:ingest_goldcase_001_010"

gold-v1-standard-ingest: gold-v1-verify
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.ingest_gold_v1_via_standard_pipeline \
	  --approve --authorization "user_request:2026-07-21:standard_pipeline_goldcase_001_010"

gold-v1-baseline: gold-v1-verify
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.kg_v2_gold_compare \
	  --gold-root data/annotations/goldcases/gold-v1 --kg-root data/kg --runner-mode native_v2 --with-w7-loo \
	  --out data/results/gold-v1-w1-w7-current.json \
	  --md-out data/results/gold-v1-w1-w7-current.md --quiet

gold-v1-prompt-preview: gold-v1-verify
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.gold_prompt_preview \
	  --gold-root data/annotations/goldcases/gold-v1 --kg-root data/kg \
	  --out data/results/gold-v1-prompt-preview.json \
	  --md-out data/results/gold-v1-prompt-preview.md \
	  --response-template-out data/results/gold-v1-prompt-responses.template.json

gold-v1-prompt-replay: gold-v1-verify
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.gold_prompt_replay \
	  $(GOLD_PROMPT_RESPONSES) \
	  --gold-root data/annotations/goldcases/gold-v1 --kg-root data/kg \
	  --out data/results/gold-v1-w1-w7-prompt-replayed.json \
	  --md-out data/results/gold-v1-w1-w7-prompt-replayed.md

gold-v1-prompt-baseline: gold-v1-verify
	DEBUG_AGENT_SYSTEM_W2_DEEPSEEK=1 PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.kg_v2_gold_compare \
	  --gold-root data/annotations/goldcases/gold-v1 --kg-root data/kg --runner-mode prompt_first --deepseek --with-w7-loo \
	  --out data/results/gold-v1-w1-w7-prompt-current.json \
	  --md-out data/results/gold-v1-w1-w7-prompt-current.md --quiet

test:
	PYTHONPATH=src python3 tests/run_tests.py

test-read-tool-harness:
	PYTHONPATH=src pytest -q \
	  tests/unit/test_read_tool_contracts.py \
	  tests/unit/test_evidence_gap_resolver.py \
	  tests/integration/test_read_codex_tool_harness.py

smoke:
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli diagnose "AOI主程序初始化失败，相机连接异常，请检查相机IP"

eval:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner --limit 10

eval-real:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner --scenario-file data/eval/scenarios/industrial_pc_boot_v1.json --limit 100 --judge report-only

eval-real-gate:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.gate --eval data/results/runs/latest_real.txt --baseline data/eval/baselines/real_diag_v1_baseline.json

gate-real: eval-real-gate

baseline-real:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.baseline --run data/results/runs/latest_real.txt --baseline data/eval/baselines/real_diag_v1_baseline.json

generate-broad-debug-eval:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.build_broad_debug_scenarios \
	  --raw-dir data/raw/aoi_debug_agent_sources \
	  --out data/eval/scenarios/broad_debug_v1.json \
	  --rejected-out data/eval/scenarios/broad_debug_v1_rejected.json \
	  --limit 150

eval-broad-debug:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner \
	  --scenario-file data/eval/scenarios/broad_debug_v1.json \
	  --limit 150 \
	  --judge report-only

baseline-broad-debug:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.baseline \
	  --run data/results/runs/latest_real.txt \
	  --baseline data/eval/baselines/broad_debug_v1_baseline.json

eval-broad-debug-gate:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.gate \
	  --eval data/results/runs/latest_real.txt \
	  --baseline data/eval/baselines/broad_debug_v1_baseline.json

gate-broad-debug: eval-broad-debug-gate

ASK_INFO_QUEUE ?= data/kg/review_queue/ask_info_candidates.json
ASK_INFO_SCENARIOS ?= data/eval/scenarios/ask_info_candidates_v1.json

generate-ask-info-eval:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.ask_info_candidates \
	  --queue $(ASK_INFO_QUEUE) \
	  --out $(ASK_INFO_SCENARIOS) \
	  --limit 30

eval-ask-info:
	@test -s $(ASK_INFO_SCENARIOS) || (echo "missing $(ASK_INFO_SCENARIOS); run make generate-ask-info-eval ASK_INFO_QUEUE=<review_queue_path> first"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner \
	  --scenario-file $(ASK_INFO_SCENARIOS) \
	  --limit 30 \
	  --judge report-only \
	  --latest-name latest_ask_info.txt

eval-ask-info-gate:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.ask_info_gate \
	  --eval data/results/runs/latest_ask_info.txt \
	  --min-cases 30 \
	  --min-required-info-acc 0.6

SAG_SQLITE ?= data/kg_sag/debug_agent.sqlite
SAG_REPORT ?= data/kg_sag/build_report.json
SAG_V2_SQLITE ?= data/kg_v2_sag/debug_agent_v2.sqlite
SAG_W1_ROOT ?= data/results/w1_full_20260703_061455
SAG_RETRIEVAL_OUT ?= data/results/sag_retrieval/latest.json
SAG_REGRESSION_OUT ?= data/results/sag_regression/latest.json
SAG_COMPARE_MD ?= docs/kg_sag_experiment_report.md
SAG_COMPARE_JSON ?= data/results/sag_comparison/latest.json
CHAT_REPLAY_SCENARIOS ?= data/eval/scenarios/chat_replay_seed_v1.json
CHAT_REPLAY_OUT_JSON ?= data/results/chat_replay/latest.json
CHAT_REPLAY_OUT_MD ?= data/results/chat_replay/latest.md
W7_REVIEW_ROOT ?= data/results/w7_release_gate_payload_candidate_20260724
W7_HUMAN_ANNOTATIONS ?= $(W7_REVIEW_ROOT)/human_annotations.json
W7_HUMAN_REPORT ?= $(W7_REVIEW_ROOT)/human_review_report.json
W7_HUMAN_DATASET ?= $(W7_REVIEW_ROOT)/adjudicated_dataset.json
W7_RELEASE_INPUT ?= data/results/xing_relation_context_payload_candidate_20260724/episodes.json
W7_FIXED173_RESULT ?= data/results/xing_fixed173_safety_v6_20260722/pipeline_result.json
W7_HUMAN_REVIEW_HOST ?= 127.0.0.1
W7_HUMAN_REVIEW_PORT ?= 8765
W7_MULTI_AGENT_INPUT ?= $(W7_RELEASE_INPUT)
W7_MULTI_AGENT_OUT ?= data/results/w7_multi_agent_shadow_current
W7_MULTI_AGENT_DEEPSEEK ?= 0
W7_MULTI_AGENT_SCOPE ?= episode
W7_MULTI_AGENT_RUN_W2 ?= 0
W7_MULTI_AGENT_W2_MODE ?= native_v2
W7_MULTI_AGENT_W2_WORKERS ?= 1
W7_MULTI_AGENT_DECISION_WORKERS ?= 1
W7_CALIBRATION_ROOT ?= data/results/w7_multi_agent_calibration_5
W7_CALIBRATION_INPUT ?= $(W7_CALIBRATION_ROOT)/source_input.json
W7_CALIBRATION_OUT ?= $(W7_CALIBRATION_ROOT)/shadow
W7_CALIBRATION_SCORE ?= $(W7_CALIBRATION_ROOT)/score.json
W7_CALIBRATION_LIMIT ?= 5
W7_FIXED173_SHADOW ?= data/results/w7_multi_agent_fixed173_shadow_current
W7_FIXED173_GATE ?= $(W7_FIXED173_SHADOW)/safety_gate.json
W7_ACCEPTANCE_HELDOUT ?=
W7_ACCEPTANCE_OUT ?= data/results/w7_multi_agent_acceptance_current.json
TRACE_HARNESS_MESSAGES ?= data/results/xing_relation_context_payload_candidate_20260724/messages.jsonl
TRACE_HARNESS_SOURCE_THREAD_ID ?=
TRACE_HARNESS_OUT ?= data/results/deepseek_trace_assembly_harness_current
W2_CANDIDATES ?= data/results/w2_native_v2_full_latest/w2_candidates.jsonl
W2_FAMILY_OUT ?= data/results/w2_native_v2_full_latest/family_diagnostics.json
W2_QUALITY_OUT ?= data/results/w2_native_v2_full_latest/quality_diagnostics.json
W2_POSTRUN_OUT ?= data/results/w2_native_v2_full_latest/postrun_report.json
W2_POSTRUN_BASE ?=
W2_POSTRUN_CANDIDATE ?=
KG_V2_OVERVIEW_SNAPSHOT ?= data/results/kg_v2_overview_snapshot.json
KG_V2_OVERVIEW_HTML ?= data/results/kg_v2_overview.html
KG_V2_OVERVIEW_RUN ?= data/results/w2_native_v2_full_pinned_20260708_010455

sag-build:
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli kg-v2-sag-build \
	  --kg-v2-root data/kg_v2 \
	  --out $(SAG_V2_SQLITE)

sag-build-legacy:
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli sag-build \
	  --out $(SAG_SQLITE) \
	  --raw-root data/raw/aoi_debug_agent_sources \
	  --kg-root data/kg \
	  --kg-v2-root data/kg_v2 \
	  --w1-root $(SAG_W1_ROOT) \
	  --report-out $(SAG_REPORT)

eval-sag-retrieval:
	@test -s $(SAG_SQLITE) || (echo "missing $(SAG_SQLITE); run make sag-build first"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.sag_retrieval \
	  --scenario-file data/eval/scenarios/broad_debug_v1.json \
	  --kg-root data/kg \
	  --sqlite-sag-path $(SAG_SQLITE) \
	  --limit 150 \
	  --top-k 5 \
	  --out $(SAG_RETRIEVAL_OUT)

eval-sag-regression:
	@test -s $(SAG_SQLITE) || (echo "missing $(SAG_SQLITE); run make sag-build first"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.sag_regression \
	  --config config/debug_agent_system_sag.yaml \
	  --scenario-file data/eval/scenarios/sag_regression_v1.json \
	  --out $(SAG_REGRESSION_OUT)

eval-json-real-compare:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner \
	  --config config/debug_agent_system_json.yaml \
	  --scenario-file data/eval/scenarios/industrial_pc_boot_v1.json \
	  --limit 100 \
	  --judge report-only \
	  --latest-name latest_json_real.txt

eval-json-broad-debug-compare:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner \
	  --config config/debug_agent_system_json.yaml \
	  --scenario-file data/eval/scenarios/broad_debug_v1.json \
	  --limit 150 \
	  --judge report-only \
	  --latest-name latest_json_broad_debug.txt

eval-sag-real:
	@test -s $(SAG_SQLITE) || (echo "missing $(SAG_SQLITE); run make sag-build first"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner \
	  --config config/debug_agent_system_sag.yaml \
	  --scenario-file data/eval/scenarios/industrial_pc_boot_v1.json \
	  --limit 100 \
	  --judge report-only \
	  --latest-name latest_sag_real.txt

eval-sag-broad-debug:
	@test -s $(SAG_SQLITE) || (echo "missing $(SAG_SQLITE); run make sag-build first"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner \
	  --config config/debug_agent_system_sag.yaml \
	  --scenario-file data/eval/scenarios/broad_debug_v1.json \
	  --limit 150 \
	  --judge report-only \
	  --latest-name latest_sag_broad_debug.txt

gate-json-real-compare:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.gate \
	  --eval data/results/runs/latest_json_real.txt \
	  --baseline data/eval/baselines/real_diag_v1_baseline.json

gate-json-broad-debug-compare:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.gate \
	  --eval data/results/runs/latest_json_broad_debug.txt \
	  --baseline data/eval/baselines/broad_debug_v1_baseline.json

gate-sag-real:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.gate \
	  --eval data/results/runs/latest_sag_real.txt \
	  --baseline data/eval/baselines/real_diag_v1_baseline.json

gate-sag-broad-debug:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.gate \
	  --eval data/results/runs/latest_sag_broad_debug.txt \
	  --baseline data/eval/baselines/broad_debug_v1_baseline.json

sag-comparison-report:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.sag_comparison \
	  --out-md $(SAG_COMPARE_MD) \
	  --out-json $(SAG_COMPARE_JSON)

sag-comparison: sag-build sag-build-legacy eval-sag-retrieval eval-sag-regression eval-json-real-compare gate-json-real-compare eval-json-broad-debug-compare gate-json-broad-debug-compare eval-sag-real gate-sag-real eval-sag-broad-debug gate-sag-broad-debug sag-comparison-report

eval-chat-replay:
	@test -s $(SAG_SQLITE) || (echo "missing $(SAG_SQLITE); run make sag-build first"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.chat_replay \
	  --config config/debug_agent_system_sag.yaml \
	  --scenario-file $(CHAT_REPLAY_SCENARIOS) \
	  --out-json $(CHAT_REPLAY_OUT_JSON) \
	  --out-md $(CHAT_REPLAY_OUT_MD)

eval-write-rollback-audit:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.approved_replay_rollback_audit \
	  --out data/results/write_rollback_audit/latest.json

w7-release-gate-build:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.w7_targeted_regression \
	  --input $(W7_RELEASE_INPUT) \
	  --out-dir $(W7_REVIEW_ROOT) \
	  --per-bucket 20 \
	  --include-pipeline-result $(W7_FIXED173_RESULT)
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.w7_human_review init \
	  $(W7_REVIEW_ROOT)/review_pack.json \
	  --out $(W7_HUMAN_ANNOTATIONS)

w7-human-review-init:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.w7_human_review init \
	  $(W7_REVIEW_ROOT)/review_pack.json \
	  --out $(W7_HUMAN_ANNOTATIONS)

w7-human-review-serve:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.w7_human_review_server \
	  $(W7_HUMAN_ANNOTATIONS) \
	  --host $(W7_HUMAN_REVIEW_HOST) \
	  --port $(W7_HUMAN_REVIEW_PORT)

w7-human-review-validate:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.w7_human_review validate \
	  $(W7_HUMAN_ANNOTATIONS) \
	  --min-sessions 50 \
	  --out $(W7_HUMAN_REPORT)

w7-human-review-export:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.w7_human_review export \
	  $(W7_HUMAN_ANNOTATIONS) \
	  --out $(W7_HUMAN_DATASET)

w7-multi-agent-shadow:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.w7_multi_agent_harness \
	  --input $(W7_MULTI_AGENT_INPUT) \
	  --out-dir $(W7_MULTI_AGENT_OUT) \
	  --env-file .env.local \
	  --batch-scope $(W7_MULTI_AGENT_SCOPE) \
	  --w2-mode $(W7_MULTI_AGENT_W2_MODE) \
	  --w2-workers $(W7_MULTI_AGENT_W2_WORKERS) \
	  --decision-workers $(W7_MULTI_AGENT_DECISION_WORKERS) \
	  $(if $(filter 1 true TRUE yes YES,$(W7_MULTI_AGENT_RUN_W2)),--run-w2,) \
	  $(if $(filter 1 true TRUE yes YES,$(W7_MULTI_AGENT_DEEPSEEK)),--deepseek,)

w7-multi-agent-session-shadow:
	$(MAKE) w7-multi-agent-shadow \
	  W7_MULTI_AGENT_SCOPE=thread \
	  W7_MULTI_AGENT_RUN_W2=1 \
	  W7_MULTI_AGENT_DECISION_WORKERS=4

w7-multi-agent-calibration-build:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.build_w7_calibration_input \
	  --annotations $(W7_HUMAN_ANNOTATIONS) \
	  --out $(W7_CALIBRATION_INPUT) \
	  --limit $(W7_CALIBRATION_LIMIT)

w7-multi-agent-calibration-run: w7-multi-agent-calibration-build
	$(MAKE) w7-multi-agent-shadow \
	  W7_MULTI_AGENT_INPUT=$(W7_CALIBRATION_INPUT) \
	  W7_MULTI_AGENT_OUT=$(W7_CALIBRATION_OUT) \
	  W7_MULTI_AGENT_SCOPE=chat \
	  W7_MULTI_AGENT_RUN_W2=1 \
	  W7_MULTI_AGENT_DEEPSEEK=1 \
	  W7_MULTI_AGENT_DECISION_WORKERS=4 \
	  W7_MULTI_AGENT_W2_WORKERS=4

w7-multi-agent-calibration-score:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.w7_multi_agent_score \
	  --manifest $(W7_CALIBRATION_OUT)/manifest.json \
	  --annotations $(W7_HUMAN_ANNOTATIONS) \
	  --session-limit $(W7_CALIBRATION_LIMIT) \
	  --out $(W7_CALIBRATION_SCORE)

w7-multi-agent-fixed173-run:
	$(MAKE) w7-multi-agent-shadow \
	  W7_MULTI_AGENT_INPUT=$(W7_FIXED173_RESULT) \
	  W7_MULTI_AGENT_OUT=$(W7_FIXED173_SHADOW) \
	  W7_MULTI_AGENT_SCOPE=chat \
	  W7_MULTI_AGENT_RUN_W2=1 \
	  W7_MULTI_AGENT_DEEPSEEK=1 \
	  W7_MULTI_AGENT_DECISION_WORKERS=6 \
	  W7_MULTI_AGENT_W2_WORKERS=6

w7-multi-agent-fixed173-gate:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.w7_multi_agent_safety_gate \
	  --manifest $(W7_FIXED173_SHADOW)/manifest.json \
	  --expected-episodes 173 \
	  --min-schema-valid-rate 1.0 \
	  --out $(W7_FIXED173_GATE)

w7-multi-agent-acceptance:
	@test -s $(W7_CALIBRATION_SCORE) || (echo "missing $(W7_CALIBRATION_SCORE); run make w7-multi-agent-calibration-score first"; exit 1)
	@test -s $(W7_FIXED173_GATE) || (echo "missing $(W7_FIXED173_GATE); run make w7-multi-agent-fixed173-gate first"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.w7_multi_agent_acceptance \
	  --calibration $(W7_CALIBRATION_SCORE) \
	  --fixed173-safety $(W7_FIXED173_GATE) \
	  $(if $(W7_ACCEPTANCE_HELDOUT),--heldout $(W7_ACCEPTANCE_HELDOUT),) \
	  --out $(W7_ACCEPTANCE_OUT)

test-w7-multi-agent:
	PYTHONPATH=src pytest -q \
	  tests/unit/test_w7_multi_agent.py \
	  tests/unit/test_w3_v2_bundle.py

deepseek-trace-assembly-harness:
	@test -n "$(TRACE_HARNESS_SOURCE_THREAD_ID)" || (echo "missing TRACE_HARNESS_SOURCE_THREAD_ID=<source_thread_id>"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.deepseek_trace_assembly_harness \
	  --messages $(TRACE_HARNESS_MESSAGES) \
	  --source-thread-id "$(TRACE_HARNESS_SOURCE_THREAD_ID)" \
	  --out-dir $(TRACE_HARNESS_OUT) \
	  --env-file .env.local

test-write-trace-harness:
	PYTHONPATH=src pytest -q \
	  tests/unit/test_blind_deepseek_pro.py \
	  tests/unit/test_deepseek_trace_assembly_harness.py

MANUAL_GOLDEN_IMPORT_ROOT ?= data/imports/lark_xing_crawl_output_xing_upload
MANUAL_GOLDEN_OUT ?= data/results/manual_golden_compare_latest.json
MANUAL_GOLDEN_EPISODES ?=
MANUAL_GOLDEN_DEEPSEEK ?= 0
WRITE_BATCH_RUN_DIR ?=
WRITE_BATCH_MIN_DEEPSEEK_USED_RATE ?= 0

eval-write-manual-golden:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.manual_golden_compare \
	  --manual-root data/kg/review_queue/manual_review_examples \
	  --import-root $(MANUAL_GOLDEN_IMPORT_ROOT) \
	  $(if $(MANUAL_GOLDEN_EPISODES),--episodes $(MANUAL_GOLDEN_EPISODES),) \
	  $(if $(filter 1 true TRUE yes YES,$(MANUAL_GOLDEN_DEEPSEEK)),--deepseek,) \
	  --out $(MANUAL_GOLDEN_OUT)

eval-write-batch-gate:
	@test -n "$(WRITE_BATCH_RUN_DIR)" || (echo "missing WRITE_BATCH_RUN_DIR=<data/results/write_batch_...>"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.batch_candidate_gate \
	  --run-dir $(WRITE_BATCH_RUN_DIR) \
	  --min-candidates 1 \
	  --min-deepseek-used-rate $(WRITE_BATCH_MIN_DEEPSEEK_USED_RATE)

eval-conditional-branch:
	PYTHONPATH=src python3 -m debug_agent_system.eval.debug_sim.runner \
	  --scenario-file data/eval/scenarios/conditional_branch_v1.json \
	  --limit 4 \
	  --judge report-only \
	  --out-dir data/results/runs/conditional_branch

coupling-scan:
	@if grep -R --exclude='*.pyc' --exclude-dir='__pycache__' "archive/debug_agent_system\|debugagent\|debug_kg/functions\|<home-dir>\|parents\[" src/debug_agent_system tests; then exit 1; fi

w2-family-diagnostics:
	@test -s $(W2_CANDIDATES) || (echo "missing $(W2_CANDIDATES); run extract-summaries-w2 first"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli w2-family-diagnostics \
	  $(W2_CANDIDATES) \
	  --out $(W2_FAMILY_OUT)

w2-quality-diagnostics:
	@test -s $(W2_CANDIDATES) || (echo "missing $(W2_CANDIDATES); run extract-summaries-w2 first"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli w2-quality-diagnostics \
	  $(W2_CANDIDATES) \
	  --out $(W2_QUALITY_OUT)

w2-quality-gate:
	@test -s $(W2_QUALITY_OUT) || (echo "missing $(W2_QUALITY_OUT); run make w2-quality-diagnostics first"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli w2-quality-gate \
	  --diagnostics $(W2_QUALITY_OUT)

w2-postrun-report:
	@test -s $(W2_CANDIDATES) || (echo "missing $(W2_CANDIDATES); run extract-summaries-w2 first"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli w2-postrun-report \
	  --run-dir $(dir $(W2_CANDIDATES)) \
	  --out $(W2_POSTRUN_OUT)

w2-postrun-compare:
	@test -n "$(W2_POSTRUN_BASE)" || (echo "missing W2_POSTRUN_BASE=<base postrun report>"; exit 1)
	@test -n "$(W2_POSTRUN_CANDIDATE)" || (echo "missing W2_POSTRUN_CANDIDATE=<candidate postrun report>"; exit 1)
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli w2-postrun-compare \
	  --base $(W2_POSTRUN_BASE) \
	  --candidate $(W2_POSTRUN_CANDIDATE)

w2-run-status:
	@RUN_DIR=$$(dirname $(W2_CANDIDATES)); \
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli w2-run-status --run-dir $$RUN_DIR

w2-live-report:
	@RUN_DIR=$$(dirname $(W2_CANDIDATES)); \
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli w2-live-report --run-dir $$RUN_DIR

kg-v2-overview:
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli kg-v2-overview \
	  --kg-v2-root data/kg_v2 \
	  --pinned-run-dir $(KG_V2_OVERVIEW_RUN) \
	  --snapshot-out $(KG_V2_OVERVIEW_SNAPSHOT) \
	  --html-out $(KG_V2_OVERVIEW_HTML)

SOP_DOC_ROOT ?= data/raw/aoi_debug_agent_sources
SOP_SYNC_OUT ?= data/results/sop_document_sync_latest.json

.PHONY: sop-doc-sync kg-v2-build-curated-legacy

# Compatibility name: the normal SOP "build" is now a versioned update scan
# that stops at W6. It never replaces data/kg_v2 directly.
kg-v2-build-curated: sop-doc-sync

sop-doc-sync:
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli sync-sop-docs \
	  $(SOP_DOC_ROOT) \
	  --kg-v2-root data/kg_v2 \
	  --out $(SOP_SYNC_OUT)

# Bootstrap/rollback only. The explicit flag prevents accidental use as a
# second production write authority.
kg-v2-build-curated-legacy:
	PYTHONPATH=src python3 -m debug_agent_system.adapters.cli kg-v2-build-curated \
	  --kg-v2-root data/kg_v2 \
	  --build-root data/kg_v2_sop_draft_build \
	  --gold-root data/kg_v2_sop_draft_build/gold_cases \
	  --summary-out data/results/kg_v2_write_side_build_summary.json \
	  --allow-active-rebuild

JIRA_SYNC_ROOT ?= data/imports/jira_offline/raw
JIRA_SYNC_WORKERS ?= 10

jira-sync-probe:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.sync_jira_offline_full \
	  --probe-only

jira-sync-full:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.sync_jira_offline_full \
	  --output-root $(JIRA_SYNC_ROOT) \
	  --workers $(JIRA_SYNC_WORKERS)

jira-sync-repair:
	PYTHONPATH=src python3 -m debug_agent_system.eval.write_side.sync_jira_offline_full \
	  --output-root $(JIRA_SYNC_ROOT) \
	  --repair-existing-only
