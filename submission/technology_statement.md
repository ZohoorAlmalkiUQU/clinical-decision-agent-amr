# Technology Statement (Optional) — Agentic AI & watsonx Orchestrate

**Project:** AMR Clinical Decision Support System
**Team:** GROUP-6 — UQU CS6117 (Advanced Topics in AI)

---

## How the project uses Agentic AI

The core of the system is the `ClinicalDecisionAgent`, which implements a **multi-step, tool-using agentic workflow** rather than a single model call. For each patient case, the agent autonomously executes:

1. **Tool call — ML prediction.** It invokes the calibrated ensemble model (`ResistancePredictor`) to obtain a resistance probability for every candidate antibiotic, given the patient's structured clinical data.
2. **Tool call — explainability.** It invokes a SHAP `TreeExplainer` to retrieve the top contributing features behind the leading prediction, grounding the reasoning step in concrete evidence rather than letting the LLM reason "blind."
3. **Decision / ranking step.** The agent ranks all candidates by predicted resistance and applies a **confidence gate** derived from the paper-validated optimal decision threshold (per model, per Table 4 of our accepted *Discover AI* paper).
4. **Conditional branching.** Based on the confidence gate, the agent autonomously chooses one of two paths:
   - **High confidence →** it constructs a structured prompt (patient context + ranked resistance scores + SHAP evidence) and calls an LLM (LLaMA 3.3 70B via Groq) to generate a natural-language clinical rationale.
   - **Low confidence →** it bypasses the LLM entirely and returns a safe-failure response that flags the case for Infectious Disease consultation — preventing the system from fabricating a recommendation it cannot support.

This is the defining property of an agentic system: the workflow's *path* (which tools run, in what order, and whether the LLM is invoked at all) is determined dynamically by intermediate results (the ML prediction and the confidence check), not hard-coded per request. The LLM is one reasoning tool among several in the pipeline, used only when the evidence justifies it.

---

## watsonx Orchestrate

The current implementation does **not** use IBM watsonx Orchestrate — the agent orchestration described above is implemented directly in Python (`src/agent/decision_agent.py`), and the LLM reasoning step calls LLaMA 3.3 70B via the Groq API for free, low-latency inference suitable for an academic prototype.

That said, the agent's architecture maps cleanly onto a watsonx Orchestrate deployment, which we see as a natural next step for productionising this system:

- The **ML prediction step** and **SHAP explanation step** could each be exposed as a custom **skill/tool** registered with a watsonx Orchestrate agent, callable through its tool-use interface.
- The **confidence-gate branching logic** (LLM reasoning vs. safe-failure escalation) maps directly to Orchestrate's conditional flow/agent-routing capabilities.
- The **LLM reasoning step** could be swapped from Groq/LLaMA to a **watsonx.ai foundation model** (e.g., Granite) invoked through Orchestrate, with no change to the surrounding agent logic, since the prompt-construction and response-handling are already isolated in `build_prompt()` / `recommend()`.
- The **Streamlit UI** could remain as the clinician-facing front end, or be replaced by an Orchestrate-hosted conversational interface that calls the same underlying agent.

We scoped the current build around an open, reproducible stack (scikit-learn-compatible models + Groq) to keep the prototype fully runnable without enterprise credentials, while keeping the agent's tool/decision structure compatible with a future watsonx Orchestrate migration.
