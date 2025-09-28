
# DexBench: Dexterous Manipulation Benchmark (MuJoCo + Isaac + RLHF)

End-to-end kit for **teleop → BC → RLHF → RL** on a simple dexterous manipulation task.

## Install
```bash
pip install -r requirements.txt
pip install -e .[inputs]     # package + optional inputs (pygame, hidapi)
```

## Quickstart (MuJoCo)
```bash
python scripts/teleop_record.py --env three --len 600 --out demos/teleop
python bc_train.py --demo-glob 'demos/teleop/*.npz' --epochs 10 --save runs/bc.pt
python bc_eval.py --env three --model runs/bc.pt --render
```

## RLHF
```bash
python prefs_make_pairs.py --demo-glob 'demos/teleop/*.npz' --seg-len 150 --pairs 40 --out demos/preferences/pairs.json
python segment_render.py --pairs demos/preferences/pairs.json --outdir label_ui/static/segs
cd label_ui && python app.py    # http://localhost:5001
cd ..
python reward_model_train.py --pairs demos/preferences/pairs.json --labels label_ui/labels.json --out runs/reward.pt
python train_with_reward_model.py --env three --rm runs/reward.pt --total-steps 100000
```

## Browser teleop
```bash
python teleop_ws_server.py
# open label_ui/teleop_web/index.html in Chrome; connect to ws://localhost:8765
```

## Isaac (optional)
```bash
./isaac-sim.sh --no-window --python $PWD/isaac/create_dexbench_stage.py
./isaac-sim.sh --ext-folder . --python $PWD/isaac/play_dexbench_teleop_state.py -- --input keyboard --episodes 1 --len 1200 --save demos/isaac_state
```

### Set the real CI badge
```bash
python tools/set_ci_badge.py --owner YOUR_GITHUB_USER --repo YOUR_REPO
git add README.md && git commit -m "doc: set CI badge"
```

### Releases
Tag with SemVer to trigger a build and GitHub Release attachments:
```bash
git tag v0.1.0
git push origin v0.1.0
```

### Pre-commit hooks
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
