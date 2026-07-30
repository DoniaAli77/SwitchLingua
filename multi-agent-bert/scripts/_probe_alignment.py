import os, sys
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agents.llm_ner_reflection_agent import LLMNERReflectionAgent
from src.agents.llm_ner_panel_agents import LLMNERDebateAgent, LLMNERDisambiguationAgent
from src.evaluation.ner_conll_loader import TYPE_LABELS, load_conll, tag_to_type, to_type_dataset
from src.llm.openai_client import OpenAIClient
from src.models.transformer_ner_tagger import TransformerNERTagger
from src.pipeline.ner_agentic import agentic_ner_task_config
from src.state.schema import PipelineState, StateMetadata

p = Path(__file__).resolve().parent.parent / "../Modified_Version/.env"
for line in p.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

SEARCH = sys.argv[1] if len(sys.argv) > 1 else "الجنوب"
sents = to_type_dataset(load_conll(Path(__file__).resolve().parent.parent / "data/NER/Test_AR-EN_NER.txt"))
target = next(s for s in sents if SEARCH in s["tokens"])
tagger = TransformerNERTagger.from_pretrained(checkpoint="models/xlmr_sabty_ner", device="cpu", tag_normalizer=tag_to_type)
client = OpenAIClient(model="gpt-4o-mini", max_tokens=1500)
refl = LLMNERReflectionAgent(client, output_slot="contextual")
deb = LLMNERDebateAgent(client, source_a="model", source_b="contextual", output_slot="contextual")
dis = LLMNERDisambiguationAgent(client, source_slot="contextual", output_slot="contextual")

st = PipelineState(metadata=StateMetadata(sample_id="x"), input_text=target["text"],
    task_config=agentic_ner_task_config(TYPE_LABELS, threshold=0.90), extras={"tokens": target["tokens"]})
tagger.run(st)
prim = [t.tag for t in st.ner_model_output.sequence_output.tags]
refl.run(st); refl_tags = [t.tag for t in st.contextual_output.sequence_output.tags]
refl_raw = st.contextual_output.sequence_output.features.get("raw_llm_response", "")
deb.run(st); deb_tags = [t.tag for t in st.contextual_output.sequence_output.tags]
dis.run(st); dis_tags = [t.tag for t in st.contextual_output.sequence_output.tags]

print("N tokens:", len(target["tokens"]))
print("REFLECTOR RAW:\n", refl_raw[:600])
print(f"\n{'idx':<4}{'token':<16}{'gold':<7}{'prim':<7}{'refl':<7}{'deb':<7}{'disamb':<7}")
for i,(tok,g,pr,rf,db,ds) in enumerate(zip(target["tokens"], target["tags"], prim, refl_tags, deb_tags, dis_tags)):
    flag = "" if g==ds else "  <<"
    print(f"{i:<4}{tok[:15]:<16}{g:<7}{pr:<7}{rf:<7}{db:<7}{ds:<7}{flag}")
