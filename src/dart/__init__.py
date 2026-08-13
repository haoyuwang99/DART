"""DART --- Detect, Attribute, Redact: representation-drift monitoring for LLM-agent defense.

Package layout (by concern):
  agent       the agent + model: HiddenLM (generation, hidden states, steering hook), tool parsing
  monitor     detection: reading-direction fit + transition scoring + AUROC
  mitigation  interventions: stop / redact (+span attribution) / reminder (+attributed) / steer + gating
  datasets    executable benchmark adapters: AgentDojo + MT (rollouts, fits, harmful/benign pairs)
  baselines   non-monitor baselines: AgentSpec rule-based action enforcement
  eval        grid harness: record store (emit) + report + run entry
"""
