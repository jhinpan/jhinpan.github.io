#!/usr/bin/env python3
"""Build the amdpilot trajectory comparison site from result directories."""

import json
import os
import glob
from pathlib import Path

RESULTS_ROOT = Path("/home/jinpan12/amdpilot/results")

RUNS = {
    "kimi-k25": {
        "task": "sglang-kimi-k25-optimize",
        "metric_name": "decode_median_ms",
        "metric_unit": "ms",
        "metric_direction": "lower",
        "variants": {
            "base": {"path": "20260324_0342", "label": "Base Qwen3.5-397B-A17B", "color": "#6366f1"},
            "sft":  {"path": "20260324_2158", "label": "SFT Qwen3.5-397B-A17B-SFT-v4", "color": "#f59e0b"},
        }
    },
    "qwen-vl": {
        "task": "sglang-qwen-vl-optimize",
        "metric_name": "output_throughput_tok_s",
        "metric_unit": "tok/s",
        "metric_direction": "higher",
        "variants": {
            "base": {"path": "20260324_0342", "label": "Base Qwen3.5-397B-A17B", "color": "#6366f1"},
            "sft":  {"path": "20260324_2158", "label": "SFT Qwen3.5-397B-A17B-SFT-v4", "color": "#f59e0b"},
        }
    },
}


def load_summary(task_dir, timestamp):
    p = RESULTS_ROOT / task_dir / timestamp / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_scoreboard(task_dir, timestamp):
    p = RESULTS_ROOT / task_dir / timestamp / "scoreboard.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().strip().split("\n") if l.strip()]


def parse_context_jsonl(path):
    steps = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            role = d.get("role", "")
            if role.startswith("_"):
                if role == "_usage":
                    steps.append({"type": "usage", "tokens": d.get("token_count", 0)})
                continue
            if role == "user":
                content = d.get("content", "")
                steps.append({"type": "user", "content": content if isinstance(content, str) else content})
            elif role == "assistant":
                content = d.get("content", [])
                tool_calls = d.get("tool_calls", [])
                thinking_text, assistant_text = "", ""
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") == "thinking":
                                thinking_text += c.get("thinking", "")
                            elif c.get("type") == "text" and c.get("text", "").strip():
                                assistant_text += c.get("text", "")
                elif isinstance(content, str):
                    assistant_text = content
                tools = []
                for tc in (tool_calls or []):
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {"raw": fn.get("arguments", "")}
                    tools.append({"id": tc.get("id", ""), "name": fn.get("name", "?"), "args": args})
                steps.append({"type": "assistant", "thinking": thinking_text, "text": assistant_text, "tool_calls": tools})
            elif role == "tool":
                content = d.get("content", "")
                if isinstance(content, list):
                    text = "\n".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
                else:
                    text = str(content)
                steps.append({"type": "tool_result", "tool_call_id": d.get("tool_call_id", ""), "content": text})
    return steps


def load_trial_trajectories(task_dir, timestamp):
    base = RESULTS_ROOT / task_dir / timestamp / "agent_output"
    if not base.exists():
        return {}
    trials = {}
    for trial_dir in sorted(base.iterdir()):
        if trial_dir.is_dir() and trial_dir.name.endswith("_trajectory"):
            trial_num = trial_dir.name.replace("trial_", "").replace("_trajectory", "")
            sessions_dir = trial_dir / "sessions"
            if not sessions_dir.exists():
                continue
            all_contexts = sorted(sessions_dir.rglob("context*.jsonl"), key=lambda p: p.stat().st_mtime)
            session_data = {}
            for cf in all_contexts:
                uuid_parts = str(cf.relative_to(sessions_dir)).split("/")
                sk = uuid_parts[1] if len(uuid_parts) > 1 else uuid_parts[0]
                if sk not in session_data:
                    session_data[sk] = []
                session_data[sk].append({"file": cf.name, "lines": sum(1 for _ in open(cf)), "path": str(cf)})
            new_sessions = set()
            if len(session_data) > 1:
                max_key = max(session_data, key=lambda k: max(s["lines"] for s in session_data[k]))
                new_sessions = {max_key}
            elif session_data:
                new_sessions = set(session_data.keys())
            new_context_path = None
            for sk in new_sessions:
                for sf in session_data[sk]:
                    if sf["file"] == "context.jsonl":
                        new_context_path = sf["path"]
                        break
            steps = parse_context_jsonl(new_context_path) if new_context_path else []
            trials[trial_num] = {"step_count": len([s for s in steps if s["type"] == "assistant"]), "steps": steps}
    return trials


def truncate(text, max_len=50000):
    return text[:max_len] + f"\n... [{len(text)} chars total]" if len(text) > max_len else text


def build_data():
    data = {"tasks": {}}
    for task_key, task_info in RUNS.items():
        task_data = {"task": task_info["task"], "metric_name": task_info["metric_name"],
                     "metric_unit": task_info["metric_unit"], "metric_direction": task_info["metric_direction"], "variants": {}}
        for var_key, var_info in task_info["variants"].items():
            summary = load_summary(task_info["task"], var_info["path"])
            scoreboard = load_scoreboard(task_info["task"], var_info["path"])
            trajectories = load_trial_trajectories(task_info["task"], var_info["path"])
            for tdata in trajectories.values():
                for step in tdata["steps"]:
                    if step["type"] == "user":
                        c = step.get("content", "")
                        if isinstance(c, str):
                            step["content"] = truncate(c, 20000)
                    elif step["type"] == "assistant":
                        step["text"] = truncate(step.get("text", ""), 10000)
                        step["thinking"] = truncate(step.get("thinking", ""), 5000)
                        for tc in step.get("tool_calls", []):
                            for k, v in tc.get("args", {}).items():
                                if isinstance(v, str) and len(v) > 5000:
                                    tc["args"][k] = v[:5000] + "..."
                    elif step["type"] == "tool_result":
                        step["content"] = truncate(step.get("content", ""), 15000)
            task_data["variants"][var_key] = {
                "label": var_info["label"], "color": var_info["color"], "timestamp": var_info["path"],
                "summary": summary, "scoreboard": scoreboard, "trials": trajectories}
        data["tasks"][task_key] = task_data
    return data


if __name__ == "__main__":
    data = build_data()
    out = Path("/home/jinpan12/jhinpan.github.io/amdpilot/data.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data))
    print(f"Wrote {out} ({out.stat().st_size/1024/1024:.1f} MB)")
    for tk, td in data["tasks"].items():
        for vk, vd in td["variants"].items():
            print(f"  {tk}/{vk}: {len(vd['trials'])} trials, {sum(t['step_count'] for t in vd['trials'].values())} steps")
