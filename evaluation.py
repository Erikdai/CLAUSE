# -*- coding: utf-8 -*-
import os
import argparse
from typing import List, Dict, Any

import torch

from logikg_mappo.agents.enhanced_graph_builder import EnhancedGraphBuilder
from logikg_mappo.agents.traversal import EnhancedTraversalAgent
from logikg_mappo.agents.decoder import LogicAwareDecoder
from logikg_mappo.rl.lc_mappo import MultiHeadCentralCritic, create_lc_mappo_policies, create_lc_mappo_optimizers, lc_mappo_update

# ---------------- data ----------------
def split_multi_answers(s: str) -> List[str]:
    seps = ["||", "///", "|", ";", ","]
    parts = [s]
    for sp in seps:
        if any(sp in p for p in parts):
            nxt = []
            for p in parts:
                nxt.extend(p.split(sp))
            parts = nxt
    return [p.strip() for p in parts if p.strip()]

class QADataset:
    def __init__(self, path: str):
        self.items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 1:
                    self.items.append({"question": parts[0], "answers": []})
                else:
                    q, raw = parts[0], parts[1]
                    self.items.append({"question": q, "answers": split_multi_answers(raw)})

def em_match(pred: str, golds: List[str]) -> int:
    p = (pred or "").strip().lower()
    return 1 if any(p == g.strip().lower() for g in golds) else 0

def extract_candidates_from_action(action: Dict[str, Any], topk: int = 64) -> List[str]:
    cands: List[str] = []
    nodes = action.get("nodes", [])
    if isinstance(nodes, list):
        for x in nodes:
            if isinstance(x, str):
                cands.append(x)
    if not cands:
        edges = action.get("edges", [])
        if isinstance(edges, list):
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) == 3 and all(isinstance(v, str) for v in e):
                    h, _, t = e
                    cands += [h, t]
    seen, out = set(), []
    for x in cands:
        if x not in seen:
            seen.add(x)
            out.append(x)
        if len(out) >= topk:
            break
    return out


# ---------------- eval ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--ckpt_dir", default="outputs/lcmappo_txt_cache_only")
    ap.add_argument("--device", default="cuda")

    # cache-only KB（用户自定义）
    ap.add_argument("--kb_cache_dir", required=True, help="Path to your KB cache directory")
    ap.add_argument("--embedder", default="thenlper/gte-small")
    ap.add_argument("--disable_ner", action="store_true")

    # Decoder config
    ap.add_argument("--dec_vocab_size", type=int, default=2048)
    ap.add_argument("--dec_max_rules", type=int, default=4)
    ap.add_argument("--dec_disable_lm", action="store_true")
    ap.add_argument("--dec_lm_name", default="gpt2")

    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    ds = QADataset(args.test_file)
    print(f"[data] {len(ds.items)} samples")

    gb = EnhancedGraphBuilder(
        ttl_file=None,
        cache_dir=args.kb_cache_dir,
        embedder_name=args.embedder,
        device=args.device,
        use_cache_only=True,
        enable_ner=not args.disable_ner,
    )
    tr = EnhancedTraversalAgent(device=args.device)
    dec = LogicAwareDecoder(
        vocab_size=args.dec_vocab_size,
        max_rules=args.dec_max_rules,
        device=args.device,
        use_lm=not args.dec_disable_lm,
        lm_model_name=args.dec_lm_name,
    )

    # load checkpoints if present
    ck_gb = os.path.join(args.ckpt_dir, "gb-ep1.pt")
    ck_tr = os.path.join(args.ckpt_dir, "tr-ep1.pt")
    ck_dec = os.path.join(args.ckpt_dir, "dec-ep1.pt")
    if os.path.exists(ck_gb):
        gb.load_state_dict(torch.load(ck_gb, map_location=device))
        print("[ckpt] loaded gb")
    if os.path.exists(ck_tr):
        tr.load_state_dict(torch.load(ck_tr, map_location=device))
        print("[ckpt] loaded tr")
    if os.path.exists(ck_dec):
        dec.load_state_dict(torch.load(ck_dec, map_location=device))
        print("[ckpt] loaded dec")

    em, tot = 0, 0
    for ex in ds.items:
        q, golds = ex["question"], ex.get("answers", [])
        if hasattr(gb, "reset_subgraph"):
            gb.reset_subgraph()
        tr._maybe_reset_path({"reset": True})
        dec.set_question_text(q)

        last_gb_action = {}
        for _ in range(args.steps):
            gb_out = gb({"question": q})
            last_gb_action = gb_out.get("action", {}) or {}
            if (last_gb_action.get("operation_type") in (None, "stop")):
                break

        cands = extract_candidates_from_action(last_gb_action, topk=64)
        pred = ""
        if cands:
            out = dec(
                obs={"question": q},
                generate_answer=True,
                candidates=cands,
                rule_violations=[],
                gold_answer=golds[0] if golds else None,
            )
            pred = out.get("answer", "")

        em += em_match(pred, golds); tot += 1
        if args.verbose and (tot % 50 == 0):
            print(f"[{tot}] Q:{q}\n pred:{pred}\n gold(s):{golds[:5]}")

    print(f"[EM] {em}/{tot} = {em/max(1,tot):.3f}")

if __name__ == "__main__":
    main()
