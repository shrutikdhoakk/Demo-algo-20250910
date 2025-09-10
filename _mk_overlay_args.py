import argparse, yaml, pathlib
ap = argparse.ArgumentParser()
ap.add_argument("--buf", type=float, required=True)
ap.add_argument("--trail", type=float, required=True)
ap.add_argument("--atrp", type=float, required=True)
ap.add_argument("--base", default="backtest/config.yaml")
ap.add_argument("--out",  default="backtest/config_tmp.yaml")
a = ap.parse_args()

cfgp = pathlib.Path(a.base)
cfg  = yaml.safe_load(cfgp.read_text(encoding="utf-8"))
cfg.setdefault("strategycfg", {})
cfg["strategycfg"].update({
    "breakout_atr_buf": float(a.buf),
    "trail_atr_mult":   float(a.trail),
    "atr_pct_max":      float(a.atrp),
})
outp = pathlib.Path(a.out)
outp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(outp)
