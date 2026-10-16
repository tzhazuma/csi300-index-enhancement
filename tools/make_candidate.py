"""Assemble a submission package from a trained model dir plus the shared code and MLP member.

    ./venv/bin/python tools/make_candidate.py --model /tmp/cat63/model --out /tmp/cand \
        --weights lightgbm=0.5,catboost=0.30,mlp=0.20 [--no-mlp]

Renormalises the member weights so they sum to 1, copies the shipped inference/train/requirements,
and drops in the MLP member when asked for.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="trained model dir (holds metadata.json)")
    ap.add_argument("--out", required=True, help="output submission dir (will be replaced)")
    ap.add_argument("--weights", default="", help="kind=weight,... e.g. lightgbm=0.5,catboost=0.3,mlp=0.2")
    ap.add_argument("--mlp-npz", default=str(ROOT / "submission" / "model" / "mlp_member.npz"))
    ap.add_argument("--linear-npz", default=str(ROOT / "submission" / "model" / "linear_member.npz"))
    ap.add_argument("--linear", action="store_true",
                    help="also add the ridge member; it reuses the `mlp` kind because a 1-layer "
                         "npz evaluates to a plain linear map (no relu is applied to the output)")
    ap.add_argument("--mlp-train", default=None, help="train the MLP member if the npz is missing")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("inference.py", "train.py", "requirements.txt", "__init__.py"):
        shutil.copy2(ROOT / "submission" / name, out / name)
    shutil.copytree(Path(args.model), out / "model")

    meta_path = out / "model" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    members = list(meta.get("members", []))

    if args.mlp_train and not Path(args.mlp_npz).exists():
        print("[cand] training the MLP member ...")
        subprocess.run([sys.executable, str(ROOT / "tools" / "train_mlp_member.py"),
                        "--out", args.mlp_npz], check=True, cwd=ROOT)
    want_mlp = "mlp" in args.weights
    if want_mlp:
        shutil.copy2(args.mlp_npz, out / "model" / "mlp_member.npz")
        members.append({"kind": "mlp", "file": "mlp_member.npz", "weight": 1.0})
        meta["mlp_member"] = {"hidden_layer_sizes": [128, 64], "max_iter": 30, "alpha": 1e-4,
                              "preprocess": "cs_rank", "seed": 42, "target": "rank"}
    if args.linear:
        shutil.copy2(args.linear_npz, out / "model" / "linear_member.npz")
        # Kind `mlp` on purpose: a 1-layer npz is a plain linear map, because `_mlp_scores` only
        # applies relu between layers, never to the output.
        members.append({"kind": "mlp", "file": "linear_member.npz", "weight": 1.0})
        meta["linear_member"] = {"hidden_layer_sizes": [], "alpha": 100.0,
                                 "preprocess": "cs_rank", "target": "rank", "note": "ridge"}

    spec = {}
    for part in args.weights.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            spec[k.strip()] = float(v)
    if spec:
        # The MLP and the ridge member share kind `mlp`, so they must be weighted by FILE STEM
        # rather than by kind, or the two would silently split one weight between them.
        def group_key(member):
            stem = Path(member["file"]).stem
            if member["kind"] == "mlp":
                return "linear" if stem.startswith("linear") else "mlp"
            return member["kind"]
        counts = {}
        for m in members:
            counts[group_key(m)] = counts.get(group_key(m), 0) + 1
        for m in members:
            share = spec.get(group_key(m))
            if share is None:
                continue
            m["weight"] = share / counts[group_key(m)]
    total = sum(m["weight"] for m in members)
    for m in members:
        m["weight"] = m["weight"] / total
    meta["members"] = members
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # REQUIREMENTS MUST COVER EVERY MEMBER KIND.  The shipped requirements.txt lists numpy,
    # pandas, pyarrow, lightgbm and catboost only -- so a candidate carrying an XGBoost member
    # would import a package the judge was never asked to install, and the entire submission
    # would be scored as a FAILED EVALUATION rather than a weak one.  Derive the list from the
    # members actually present instead of trusting the template.  (`mlp` needs nothing extra:
    # the forward pass is pure numpy.)
    needed = {"xgboost": "xgboost>=2.0,<4"}
    req_path = out / "requirements.txt"
    text = req_path.read_text()
    added = []
    for kind in sorted({m["kind"] for m in members}):
        spec_line = needed.get(kind)
        if spec_line and spec_line.split(">")[0] not in text:
            text = text.rstrip("\n") + f"\n{spec_line}\n"
            added.append(spec_line)
    if added:
        req_path.write_text(text)
        print(f"[cand] requirements += {', '.join(added)}")

    print(f"[cand] {out}")
    for m in members:
        print(f"    {m['kind']:10s} {m['file']:24s} w={m['weight']:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
