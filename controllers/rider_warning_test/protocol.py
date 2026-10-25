"""Runs the test and saves the results.

The sim needs:
  geom                 dict with car_half, bike_half, rear_to_crash (m)
  bikes                list of bike names
  start_pass(bike, speed_ms, gap_m)
  step()               returns dict(t, radar, image, boxes, truth) for one simulation step
  end_pass()
"""
import csv
import os
import time

import numpy as np

import config as C
from warning_logic import make_modes, pick_bike_box, score, t_req


def run_protocol(sim, light, speeds_kmh, passes, out_dir, detector=None, use_camera=True, log=print):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    modes = make_modes(sim.geom)
    if not use_camera:
        modes = [m for m in modes if m.kind != "camera"]
    names = [m.name for m in modes]
    run_id = 0
    started = time.time()
    for bike in sim.bikes:
        for i in range(passes):
            speed_kmh = speeds_kmh[i % len(speeds_kmh)]
            lines = [("danger", C.DANGER_GAP), ("safe", C.SAFE_GAP)]
            if C.HAND_TEST:
                lines.append(("nohand", C.DANGER_GAP))    # door zone but nobody at the door
            for path, gap in lines:
                run_id += 1
                if hasattr(sim, "on_pass_start"):
                    sim.on_pass_start(run_id, bike, speed_kmh, path)
                result = one_pass(sim, modes, bike, speed_kmh / 3.6, gap, detector, use_camera, path)
                for m in modes:
                    res = result[m.name]
                    rows.append(dict(run=run_id, bike=bike, light=light, speed_kmh=speed_kmh, path=path,
                                     mode=m.name, warned=res["warned"], lead_s=res["lead_s"],
                                     needed_s=res["needed_s"], in_time=res["in_time"],
                                     caution_warned=res["caution_warned"], caution_lead_s=res["caution_lead_s"],
                                     caution_in_time=res["caution_in_time"],
                                     updates=res["updates"], drift_m=result["drift_m"]))
                short = ", ".join(f"{m.name.split(' (')[0]}: {fmt(result[m.name], path)}" for m in modes)
                log(f"run {run_id:2d}  {bike}  {speed_kmh} km/h  {path:6s}  {short}")
    summary = score(rows, names)
    save(rows, summary, out_dir, light)
    log(f"done in {time.time() - started:.0f} s, results saved in {out_dir}")
    return rows, summary


def fmt(res, path):
    if path == "safe":
        return "false alarm" if res["warned"] or res["caution_warned"] else "quiet"
    if path == "nohand":
        return "Danger with nobody at the door" if res["warned"] else "no Danger"
    if not res["warned"]:
        return "no warning"
    return f"{'in time' if res['in_time'] else 'late'} {res['lead_s']:.2f} s"


def one_pass(sim, modes, bike, v, gap, detector, use_camera, path="danger"):
    for m in modes:
        m.reset()
    sim.start_pass(bike, v, gap)
    dt = C.FIXED_DT
    cam_every = {m.name: max(1, round(1 / (m.rate * dt))) for m in modes if m.kind == "camera"}
    detect_every = min(cam_every.values()) if cam_every else None
    cam30 = next((m for m in modes if m.kind == "camera" and m.rate >= 30), None)
    out = {m.name: dict(warned=False, lead_s=float("nan"), needed_s=t_req(v), in_time=False, updates=0,
                        caution_warned=False, caution_lead_s=float("nan"), caution_in_time=False) for m in modes}
    hand = path != "nohand"
    drift = 0.0
    tick = 0
    box = None
    while True:
        s = sim.step()
        tr = s["truth"]
        drift = max(drift, abs(tr["gap"] - gap))
        if tr["d"] <= C.END_DISTANCE:
            break
        t = s["t"]
        if use_camera and detect_every and tick % detect_every == 0:
            boxes = s["boxes"]
            if boxes is None and detector is not None and s["image"] is not None:
                boxes = detector(s["image"])
            box = pick_bike_box(boxes)
        for m in modes:
            if m.kind == "camera":
                if use_camera and tick % cam_every[m.name] == 0:
                    fired = m.update(t, box)
                else:
                    continue
            else:
                cam_gap = None
                if m.kind in ("fusion", "new") and cam30 is not None and cam30.last_seen is not None and 0 <= t - cam30.last_seen < 0.3:
                    cam_gap = cam30.last_gap
                fired = m.update(t, s["radar"], cam_gap, hand=hand)
            if getattr(m, "caution_now", False):
                lead = (tr["d"] - tr["v"] * m.proc) / tr["v"]
                ok = lead >= t_req(tr["v"]) + C.CAUTION_LEAD
                out[m.name].update(caution_warned=True, caution_lead_s=lead, caution_in_time=ok)
                if hasattr(sim, "on_caution"):
                    sim.on_caution(m.name, lead, ok, path)
            if fired:
                d_warn = tr["d"] - tr["v"] * m.proc
                lead = d_warn / tr["v"]
                out[m.name].update(warned=True, lead_s=lead, in_time=lead >= t_req(tr["v"]))
                if hasattr(sim, "on_warning"):
                    sim.on_warning(m.name, lead, lead >= t_req(tr["v"]), path)
        tick += 1
    for m in modes:
        out[m.name]["updates"] = m.updates
    out["drift_m"] = drift
    sim.end_pass()
    return out


def save(rows, summary, out_dir, light):
    with open(os.path.join(out_dir, f"runs_{light}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [f"Rider warning test, {light}, noise {'on' if C.NOISE else 'off'}", ""]
    for s in summary:
        lead = f"average Danger {s['mean_lead_s']:.2f} s before the crash point" if s["warned"] else "no warnings on door zone runs"
        line = (f"{s['mode']:28s} in time {s['in_time']}/{s['danger_runs']} ({s['in_time_pct']:.0f}%), "
                f"false alarms {s['false_alarms']}/{s['safe_runs']} ({s['false_alarm_pct']:.0f}%), ")
        if s["nohand_runs"]:
            line += f"Danger with nobody at the door {s['nohand_alarms']}/{s['nohand_runs']}, "
        line += f"recall {s['recall']:.2f}, {lead}"
        if s["has_caution"]:
            line += f", Caution in time {s['caution_in_time']}/{s['danger_runs']}"
        lines.append(line)
    text = "\n".join(lines)
    with open(os.path.join(out_dir, f"summary_{light}.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [s["mode"] for s in summary]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=150)
        w = 0.27 if any(s["nohand_runs"] for s in summary) else 0.4
        ax.bar(x - w, [s["in_time_pct"] for s in summary], w, color="#1F7A47", label="Warned in time (door zone passes)")
        ax.bar(x, [s["false_alarm_pct"] for s in summary], w, color="#9E1B22", label="False alarms (safe passes)")
        if w < 0.4:
            ax.bar(x + w, [s["nohand_alarm_pct"] for s in summary], w, color="#E08A00",
                   label="Danger with nobody at the door")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Percent of runs")
        ax.set_title(f"Webots rider warning test, {light}, noise {'on' if C.NOISE else 'off'}")
        ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"summary_{light}.png"))
        plt.close(fig)
    except ImportError:
        pass
