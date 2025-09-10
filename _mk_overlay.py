import yaml, pathlib
cfgp = pathlib.Path("backtest/config.yaml")
cfg  = yaml.safe_load(cfgp.read_text(encoding="utf-8"))

cfg.setdefault("strategycfg", {})
cfg["strategycfg"].update({
    "breakout_atr_buf": 0.30,
    "trail_atr_mult":   1.40,
    "atr_pct_max":      0.10,
})

tmp = pathlib.Path("backtest/config_tmp.yaml")
tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print("Wrote", tmp)
