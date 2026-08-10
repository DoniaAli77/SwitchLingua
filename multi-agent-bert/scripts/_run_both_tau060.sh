set -u
cd "$(dirname "$0")/.."
O=experiments/outputs/multi_agent_bert
mkdir -p $O/experiment_twostage_agentic_tau060 $O/experiment_topic1080_agentic_tau060
echo "===== [1/2] two-stage (300 rows, ~50 routed) $(date +%H:%M:%S) ====="
python scripts/run_twostage_agentic_tau060.py > $O/experiment_twostage_agentic_tau060/run.log 2>&1
echo "  exit=$? $(date +%H:%M:%S)"
echo "===== [2/2] topic-1080 (1163 rows, ~266 routed) $(date +%H:%M:%S) ====="
python scripts/run_topic1080_agentic_tau060.py > $O/experiment_topic1080_agentic_tau060/run.log 2>&1
echo "  exit=$? $(date +%H:%M:%S)"
echo "===== BOTH TAU060 RUNS DONE $(date +%H:%M:%S) ====="
