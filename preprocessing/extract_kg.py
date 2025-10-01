# -*- coding: utf-8 -*-
import os
import re
import ujson as json
import unicodedata
from tqdm import tqdm
import requests
from types import SimpleNamespace

# =============== API Key ===============
#   export SILICONFLOW_API_KEY="your_real_api_key"
API_KEY = os.environ.get("SILICONFLOW_API_KEY")
# =======================================

# ---------------- SiliconFlow LLM ----------------
class SiliconFlowLLM:
    def __init__(self, model: str = "Qwen/Qwen3-8B",
                 api_key: str | None = None,
                 url: str = "https://api.siliconflow.cn/v1/chat/completions",
                 timeout: int = 120):
        """
        Lightweight wrapper for SiliconFlow Chat Completions.
        """
        self.api_key = api_key or os.environ.get("SILICONFLOW_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing SiliconFlow API key. Set env var SILICONFLOW_API_KEY.")
        self.model = model
        self.url = url
        self.timeout = timeout

    def complete(self, prompt: str):
        """
        Synchronous, non-streaming completion call.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "enable_thinking": False,
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 1024
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        resp = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return SimpleNamespace(text=content)
# -------------------------------------------------

# -------------- Utilities: normalization & matching --------------
def _normalize(s: str) -> str:
    """
    Remove accents, normalize whitespace/case.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _contains(a: str, b: str) -> bool:
    """
    Case/diacritic-insensitive containment check.
    """
    a, b = _normalize(a), _normalize(b)
    return (a and b) and (a in b or b in a)

def select_answer_edge(triplets, answer: str):
    """
    Select the single triplet that 'contains the correct answer' from `triplets`.

    Priority:
      1) t exactly equals answer
      2) h exactly equals answer
      3) h or t contains answer (substring)
      4) r contains answer (substring)

    Returns (h, r, t) or None.
    """
    if not triplets or not answer:
        return None
    ans_norm = _normalize(answer)

    # 1) t == answer
    for h, r, t in triplets:
        if _normalize(t) == ans_norm:
            return (h, r, t)
    # 2) h == answer
    for h, r, t in triplets:
        if _normalize(h) == ans_norm:
            return (h, r, t)
    # 3) head/tail contains
    for h, r, t in triplets:
        if _contains(h, answer) or _contains(t, answer):
            return (h, r, t)
    # 4) relation contains
    for h, r, t in triplets:
        if _contains(r, answer):
            return (h, r, t)
    return None
# -------------------------------------------------

# ---------------- Triplet extraction (keeps your original logic) ----------------
def extract_triplets(llm, ctx):
    """
    Prompt the LLM to extract <h##r##t> $$-separated triplets from the given context text.
    """
    prompt = (
        'Extract triplets informative from the text following the examples. '
        'Make sure the triplet texts are only directly from the given text! '
        'Complete directly and strictly following the instructions without any additional words, line break nor space!\n'
        + '-'*20 + '\n'
        'Text: Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.\n'
        'Triplets:<Scott Derrickson##born in##1966>$$<Scott Derrickson##nationality##America>$$<Scott Derrickson##occupation##director>$$<Scott Derrickson##occupation##screenwriter>$$<Scott Derrickson##occupation##producer>$$\n'
        + '-'*20 + '\n'
        'Text: A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. '
        'It stars Shirley Temple in her final starring role as well as her final film appearance. '
        'Shirley Temple was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.\n'
        'Triplets:<A Kiss for Corliss##cast member##Shirley Temple>$$<Shirley Temple##served as##Chief of Protocol>$$\n'
        + '-'*20 + f'\nText: {ctx}\nTriplets:'
    )
    try:
        resp = llm.complete(prompt).text
    except Exception:
        return []

    triplets = set()
    for chunk in resp.split('$$'):
        chunk = chunk.strip()
        if len(chunk) < 6 or not (chunk.startswith('<') and chunk.endswith('>')):
            continue
        chunk = chunk[1:-1]
        toks = [t.strip() for t in chunk.split('##')]
        if len(toks) != 3:
            continue
        h, r, t = toks
        # Basic filtering
        bad = any(s for s in ["no ", "unknown", "null", "NULL", "No ", "Null", "Unknown", "NO"] if s in h or s in t) \
              or ('NO' in r)
        if bad or h == t:
            continue
        triplets.add((h, r, t))
    return list(triplets)
# ------------------------------------------------------------

if __name__ == "__main__":
    # I/O paths
    data_path = '/path/to/kg_builder/hotpot_dev_distractor_v1.json'
    out_dir  = '/path/to/kg_builder/hotpotqa_answer_edges_txt'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "answer_edges_all.txt")  # single output file for the full dataset

    # Initialize LLM (reads key from env)
    llm = SiliconFlowLLM(model='Qwen/Qwen3-8B', api_key=API_KEY)

    # Load full dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lines = []  # collect all matched edge lines
    saved = 0
    missed = 0

    # Iterate over the entire dataset (no limit)
    for sample in tqdm(data, desc="extract"):
        question = sample.get('question', '')
        answer   = sample.get('answer', '')
        ctxs     = sample.get('context', [])

        found = None
        # Scan contexts; stop at the first hit and move to the next sample
        for ctx in ctxs:
            ent = ctx[0]
            paragraphs = ctx[1]

            # Extract per paragraph
            for i, para in enumerate(paragraphs):
                ctx_text = para if i == 0 else f'{ent}: {para}'
                triples = extract_triplets(llm, ctx_text)
                if not triples:
                    continue
                edge = select_answer_edge(triples, answer)
                if edge:
                    found = edge
                    break
            if found:
                break

        if found:
            h, r, t = found
            lines.append(f"{h}|{r}|{t}")
            saved += 1
        else:
            missed += 1

    # Write all results at once
    with open(out_file, "w", encoding="utf-8") as wf:
        wf.write("\n".join(lines))

    print(f"[done] processed={len(data)}, saved={saved}, missed={missed}, out_file={out_file}")
