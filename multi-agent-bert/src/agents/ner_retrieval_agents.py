"""ner_retrieval_agents.py — two agent ROLES from the wider NER-agent literature
that target our two remaining failure causes:

* :class:`LLMNERVerifyAgent`  — GPT-NER-style SELF-VERIFICATION. Reads a draft's
  entities and asks the LLM to confirm each ("is this really a <type> here?"),
  KEEPING only confirmed ones. It can only REMOVE false positives → targets
  cause #3 (over-tagging).

* :class:`NERGazetteerAgent`  — retrieval/gazetteer lookup (KDR-Agent / RAG
  style) built deterministically from the TRAIN set. Matches known entity
  surface forms in the sentence → targets cause #4 (domain entities such as
  football clubs seen in training). Non-parametric — no fine-tuning.

Both reuse the deterministic span aligner and JSON helpers. NER-only.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.agents.llm_ner_agent import (
    _LLM_CONFIDENCE, _SLOT_ATTR, LLMNERParseError, _extract_json, coerce_to_valid,
)
from src.agents.llm_ner_span_agent import _norm, spans_from_tags
from src.llm.base_client import LLMClient
from src.state.schema import (
    AgentOutput, ModelOutput, PipelineState, SequenceLabelingOutput, TokenTag,
)

_SRC_ATTR = {**_SLOT_ATTR, "model": "ner_model_output"}


def _read_tags(state, slot):
    ao = getattr(state, _SRC_ATTR.get(slot, ""), None)
    if ao is None or ao.sequence_output is None:
        return None
    return [tt.tag for tt in ao.sequence_output.tags]


def _write(state, slot, tokens, tags, note, extra=None):
    seq = SequenceLabelingOutput(
        tags=[TokenTag(token=t, tag=g, confidence=_LLM_CONFIDENCE) for t, g in zip(tokens, tags)],
        notes=note, features=extra or {})
    setattr(state, _SLOT_ATTR[slot], AgentOutput(
        agent_name="", model_output=ModelOutput(), sequence_output=seq, notes=note))


# ---------------------------------------------------------------------------
# Verification agent (GPT-NER style)
# ---------------------------------------------------------------------------

class LLMNERVerifyAgent(BaseAgent[PipelineState]):
    """Confirm each drafted entity via the LLM; keep only approved ones."""

    def __init__(self, llm_client: LLMClient, source_slot="contextual",
                 output_slot="contextual", name=None, logger=None):
        super().__init__(name=name or "LLMNERVerifyAgent", logger=logger)
        self.llm_client = llm_client
        self.source_slot, self.output_slot = source_slot, output_slot

    def run(self, state: PipelineState) -> PipelineState:
        if state.task_config.task_type != "sequence_labeling":
            state.append_history(component=self.name, summary="skip"); return state
        tokens = state.extras.get("tokens") or state.input_text.split()
        cur = _read_tags(state, self.source_slot) or _read_tags(state, "model")
        if cur is None:
            return state
        cur = (cur + ["O"] * len(tokens))[:len(tokens)]
        ents = spans_from_tags(tokens, cur)
        if not ents:
            _write(state, self.output_slot, tokens, cur, "nothing to verify", {"removed": 0})
            state.append_history(component=self.name, summary="no entities to verify")
            return state

        raw = self.llm_client.generate(self._prompt(state.input_text, ents))
        try:
            kept = self._parse_kept(raw, ents)
            note = ""
        except LLMNERParseError as exc:
            kept, note = ents, f"verify parse fail ({exc}); kept all"

        # Rebuild tags keeping only confirmed entity spans.
        from src.agents.llm_ner_span_agent import align_entities_to_tokens
        new_tags = align_entities_to_tokens(tokens, kept, set(state.task_config.labels))
        removed = len(ents) - len(kept)
        _write(state, self.output_slot, tokens, new_tags, note or "verified",
               {"removed": removed, "kept": len(kept)})
        state.append_history(component=self.name,
                             summary=f"Verified: kept {len(kept)}/{len(ents)} entity(ies).",
                             outputs={"removed": removed})
        return state

    @staticmethod
    def _prompt(text, ents):
        lines = "\n".join(f"  {i}: \"{e['text']}\"  (claimed type: {e['type']})"
                          for i, e in enumerate(ents))
        return (
            "You verify candidate named entities in code-switched Arabic-English "
            "text. For EACH candidate, decide if it is genuinely a named entity of "
            "the claimed type IN THIS SENTENCE. Reject common words, prepositions, "
            "and mis-typed spans.\n\n"
            f"SENTENCE: {text}\n\nCANDIDATES:\n{lines}\n\n"
            "Return JSON ONLY with the candidates you CONFIRM (drop the rest):\n"
            '{"entities": [{"text": "<exact candidate text>", "type": "<TYPE>"}]}'
        )

    @staticmethod
    def _parse_kept(raw, ents):
        try:
            data = json.loads(_extract_json(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMNERParseError(str(exc)) from exc
        if not isinstance(data, dict) or not isinstance(data.get("entities"), list):
            raise LLMNERParseError("no entities list")
        return [e for e in data["entities"] if isinstance(e, dict) and e.get("text")]


# ---------------------------------------------------------------------------
# Gazetteer / retrieval agent (deterministic, built from train)
# ---------------------------------------------------------------------------

def build_gazetteer_from_conll(train_sents, tag_to_type) -> Dict[str, str]:
    """Build ``{normalized_entity_text -> majority_type}`` from train sentences."""
    counts = defaultdict(Counter)
    for s in train_sents:
        ents = spans_from_tags(s["tokens"], [tag_to_type(t) for t in s["tags"]])
        for e in ents:
            key = " ".join(_norm(x) for x in e["text"].split() if _norm(x))
            if key:
                counts[key][e["type"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in counts.items()}


class NERGazetteerAgent(BaseAgent[PipelineState]):
    """Tag known entity surface forms via a gazetteer built from train data.

    ``mode="augment"`` (default): only fill positions the source draft left as
    ``O`` (adds recall without overriding the draft). ``mode="overwrite"``:
    gazetteer wins everywhere it matches. Deterministic — no LLM.
    """

    def __init__(self, gazetteer: Dict[str, str], source_slot: Optional[str] = "contextual",
                 output_slot="contextual", mode="augment", name=None, logger=None):
        super().__init__(name=name or "NERGazetteerAgent", logger=logger)
        self.gazetteer = gazetteer
        self.source_slot, self.output_slot, self.mode = source_slot, output_slot, mode
        self._maxlen = max((len(k.split()) for k in gazetteer), default=1)

    def _lookup_tags(self, tokens, valid):
        norm = [_norm(t) for t in tokens]
        tags = ["O"] * len(tokens)
        i = 0
        while i < len(tokens):
            hit = False
            for L in range(min(self._maxlen, len(tokens) - i), 0, -1):
                key = " ".join(norm[i:i + L])
                if key in self.gazetteer:
                    ty = coerce_to_valid(self.gazetteer[key], valid)
                    if ty != "O":
                        for j in range(i, i + L):
                            tags[j] = ty
                        i += L; hit = True; break
            if not hit:
                i += 1
        return tags

    def run(self, state: PipelineState) -> PipelineState:
        if state.task_config.task_type != "sequence_labeling":
            state.append_history(component=self.name, summary="skip"); return state
        tokens = state.extras.get("tokens") or state.input_text.split()
        valid = set(state.task_config.labels)
        gaz_tags = self._lookup_tags(tokens, valid)

        base = None
        if self.source_slot:
            base = _read_tags(state, self.source_slot) or _read_tags(state, "model")
        if base is None:
            final = gaz_tags
        else:
            base = (base + ["O"] * len(tokens))[:len(tokens)]
            if self.mode == "overwrite":
                final = [g if g != "O" else b for g, b in zip(gaz_tags, base)]
            else:  # augment: only fill O positions of the draft
                final = [b if b != "O" else g for b, g in zip(base, gaz_tags)]
        n_hits = sum(1 for t in gaz_tags if t != "O")
        _write(state, self.output_slot, tokens, final, f"gazetteer {self.mode}",
               {"gazetteer_hits": n_hits})
        state.append_history(component=self.name,
                             summary=f"Gazetteer {self.mode}: {n_hits} token hit(s).")
        return state
