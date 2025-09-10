import sys, pathlib, yaml

p = pathlib.Path("backtest/config.yaml")
if not p.exists():
    print("FAIL: backtest/config.yaml not found")
    sys.exit(1)

try:
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
except Exception as e:
    print(f"FAIL: YAML parse error: {e}")
    sys.exit(1)

print("OK: parsed YAML")
print("Top-level keys:", list(cfg)[:10] if isinstance(cfg, dict) else type(cfg).__name__)

for k in ("strategy","strategycfg","engine","backtest"):
    sub = cfg.get(k) if isinstance(cfg, dict) else None
    if isinstance(sub, dict):
        print(f"Subkeys in '{k}':", list(sub)[:15])

required = ["start","end","max_positions","symbols","data","risk"]
if isinstance(cfg, dict):
    missing = [r for r in required if r not in cfg]
    print("Missing required (ignore if names differ):", missing or "None")
else:
    print("Note: YAML root is not a dict; skipping required-key check")
