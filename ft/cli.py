"""`ft` command line (spec 6)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import click
import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from ft.agent.session import checkpoint_summary, default_run_id, plan_folder, task_path_for
from ft.config import ConfigError
from ft.context import REPO_ROOT, RunContext, runs_dir
from ft.events import now_iso
from ft.io import read_json, write_json
from ft.pipeline import add_attempt, run_free_stages, run_pipeline, run_stage
from ft.providers.base import Endpoint, ProviderError
from ft.spend import SpendError, total_budget_usd, total_real_spend
from ft.stages import TooManyFailures, serve
from ft.stages import report as report_stage
from ft.stages.split import LeakageError
from ft.stages.train import TrainingFailed

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Auto fine-tuning pipeline.")
email_app = typer.Typer(help="Email cleaner (M5).")
app.add_typer(email_app, name="email")
console = Console()

# Failures with a known cause: reported as one `error:` line (plus how to resume), never a traceback.
# Anything else is a bug and keeps its traceback.
EXPECTED = (
    ConfigError,
    SpendError,
    ProviderError,  # the provider rejected a file, a deployment failed, a CLI call errored
    TrainingFailed,
    TooManyFailures,  # an outage during predictions
    LeakageError,
    TimeoutError,  # an endpoint that never became ready
    NotImplementedError,
    FileExistsError,
    FileNotFoundError,
    ValueError,
)


def _confirm(msg: str) -> bool:
    return typer.confirm(msg, default=False)


def _log(msg: str) -> None:
    console.print(f"  [dim]{msg}[/]")


def _task_path(task_dir: Path) -> Path:
    return task_dir if task_dir.is_file() else task_path_for(task_dir)


def _fail(e: BaseException, run_id: str | None = None) -> None:
    console.print(f"[bold]error:[/] {e}")
    if run_id:
        console.print(f"[dim]state saved; continue with `ft resume {run_id}` after fixing the cause.[/]")
    raise typer.Exit(1)


@app.callback()
def _main() -> None:
    load_dotenv(REPO_ROOT / ".env")


@app.command()
def run(
    task_dir: Annotated[
        Path, typer.Argument(help="Folder with task.yaml (or the task.yaml itself), or a raw data folder.")
    ],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmations (never skips the spend cap).")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Fake providers, no network, no spend.")] = False,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    from_stage: Annotated[str | None, typer.Option("--from", help="Re-run from this stage onward.")] = None,
    max_spend: Annotated[float | None, typer.Option("--max-spend", help="Override budget.max_spend_usd.")] = None,
) -> None:
    """Run the pipeline on a task folder. A folder without task.yaml goes to the agent first (M3)."""
    task_path = _task_path(task_dir)
    planned = False
    if not task_path.exists():
        if not task_dir.is_dir():
            _fail(FileNotFoundError(f"{task_dir} not found"))
        run_id = run_id or default_run_id(task_dir)
        _plan_or_exit(task_dir, run_id, dry_run)
        planned = True
    try:
        ctx = RunContext.create(
            task_path, dry_run=dry_run, yes=yes, run_id=run_id, max_spend=max_spend, confirm=_confirm
        )
    except EXPECTED as e:
        _fail(e)
    console.print(f"[bold]{ctx.run_id}[/] {'[dim](dry run)[/]' if dry_run else ''}")
    try:
        if planned and not yes:
            ctx = _checkpoint(ctx, task_dir)
        run_pipeline(ctx, console, from_stage=from_stage)
    except EXPECTED as e:
        _fail(e, ctx.run_id)
    console.print(f"\nreport: {ctx.run_dir / 'report.html'}")


def _hm(minutes: float) -> str:
    h, m = divmod(round(minutes), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def _plan_or_exit(data_dir: Path, run_id: str, dry_run: bool) -> None:
    """Agent plans the folder; writes <data_dir>/task.yaml on submit, exits 2 on a decline."""
    console.print(f"[bold]{run_id}[/] agent is looking at {data_dir}{' [dim](dry run)[/]' if dry_run else ''}")
    try:
        result, run_dir = plan_folder(data_dir, run_id, dry_run=dry_run, log=_log)
    except EXPECTED as e:
        _fail(e)
    cost = f"${result.usd:.3f}" + (" (simulated)" if dry_run else "")
    if result.outcome != "submitted":
        word = f"declined ({result.fit_label})" if result.outcome == "declined" else "gave up"
        console.print(f"\n[bold]agent {word}[/] after {result.turns} turns, {cost}: {result.reason}")
        if result.suggestion:
            console.print(f"suggestion: {result.suggestion}")
        console.print(f"[dim]transcript: {run_dir / 'agent' / 'transcript.jsonl'}[/]")
        raise typer.Exit(2)
    task_path_for(data_dir).write_text(result.task_yaml)
    console.print(f"\nagent wrote {task_path_for(data_dir)} after {result.turns} turns, {cost}")
    for k, v in result.rationale.items():
        console.print(f"  [dim]{k.replace('_', ' ')}:[/] {v}")


def _checkpoint(ctx: RunContext, data_dir: Path) -> RunContext:
    """Human checkpoint after the free local stages (spec 12): approve, edit (then revalidate), or quit."""
    while True:
        run_free_stages(ctx, console)
        s = checkpoint_summary(ctx, data_dir)
        est = s["estimate"]
        console.print(f"\n[bold]{s['task']}[/]\n  instruction: {s['instruction']}")
        console.print(
            f"  labels from {s['label_source']} · train {s['splits']['train']:,} / val {s['splits']['val']:,} / "
            f"test {s['splits']['test']:,} ({s['split_method']})" + (f" · {s['labels']} labels" if s["labels"] else "")
        )
        console.print(f"  base model {s['base_model']} · metric {s['metric']} · baselines {', '.join(s['baselines'])}")
        total = est.get("total_usd")
        cost = f"${total:.2f}" if total is not None else "unknown"
        mins = est.get("total_minutes")
        took = f", about {_hm(mins)} end to end" if mins else ""
        console.print(f"  estimated cost {cost}{took} (budget ${s['budget']:.2f})")
        for line in est.get("lines", []):
            when = f", ~{_hm(line['minutes'])}" if line.get("minutes") is not None else ""
            console.print(f"    [dim]{line['item']}: ${line['usd']}{when}[/]")
        for w in s["warnings"]:
            console.print(f"  [bold]warning:[/] {w}")
        console.print("\n  10 random test examples:")
        for ex in s["examples"]:
            console.print(f"    · {ex['input']}\n      [dim]→ {ex['expected']}[/]")
        choice = typer.prompt("\n[a]pprove / [e]dit / [q]uit", default="a").strip().lower()[:1]
        if choice == "a":
            ctx.events.emit("checkpoint", choice="approve")
            return ctx
        if choice == "q":
            ctx.events.emit("checkpoint", choice="quit")
            console.print(f"stopped before any paid stage; continue later with `ft resume {ctx.run_id}`")
            raise typer.Exit(0)
        if choice == "e":
            click.edit(filename=str(task_path_for(data_dir)))
            try:
                changed = RunContext.replace_task(ctx.run_id, task_path_for(data_dir))
            except EXPECTED as e:
                console.print(f"[bold]not valid:[/] {e}")
                continue
            ctx.events.emit("checkpoint", choice="edit", changed=changed)
            ctx = RunContext.open(ctx.run_id, yes=ctx.yes, confirm=_confirm)


def _checkpoint_if_pending(ctx: RunContext) -> RunContext:
    """A run quit (or edited) at the checkpoint hasn't been approved: show the checkpoint again before any
    paid stage. --yes approves it, as on `ft run`."""
    marks = ctx.events.of_type("checkpoint")
    if not marks or marks[-1]["data"].get("choice") == "approve":
        return ctx
    if ctx.yes:
        ctx.events.emit("checkpoint", choice="approve", via="--yes")
        return ctx
    return _checkpoint(ctx, Path(ctx.state.data["source_task_path"]).parent)


@app.command()
def resume(
    run_id: str,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    from_stage: Annotated[str | None, typer.Option("--from")] = None,
    max_spend: Annotated[float | None, typer.Option("--max-spend", help="Change this run's spend cap.")] = None,
    update_task: Annotated[
        Path | None,
        typer.Option("--update-task", help="Replace the run's task.yaml; only stages whose inputs change re-run."),
    ] = None,
) -> None:
    """Continue a run from its first incomplete (or changed) stage."""
    try:
        if update_task is not None:
            changed = RunContext.replace_task(run_id, update_task)
            console.print(f"task.yaml updated; changed sections: {', '.join(changed) or 'none'}")
        ctx = RunContext.open(run_id, yes=yes, confirm=_confirm)
        if max_spend is not None:
            ctx.state.data["max_spend_usd"] = max_spend
            ctx.state.save()
            ctx.spend.cap_usd = max_spend
        console.print(f"[bold]{run_id}[/] resuming")
        ctx = _checkpoint_if_pending(ctx)
        run_pipeline(ctx, console, from_stage=from_stage)
    except EXPECTED as e:
        _fail(e, run_id)
    console.print(f"\nreport: {ctx.run_dir / 'report.html'}")


@app.command()
def attempt(
    run_id: str,
    patch: Annotated[Path, typer.Option("--patch", help="YAML with the config changes, e.g. train: {epochs: 4}.")],
    note: Annotated[str, typer.Option("--note", help="Why this attempt; shown in the report.")],
    after_test: Annotated[
        bool, typer.Option("--after-test", help="Designed after seeing test results; the report marks it.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Add a hand-designed attempt to a run (same data, splits and baselines) and regenerate the report."""
    try:
        ctx = RunContext.open(run_id, yes=yes, confirm=_confirm)
        changes = yaml.safe_load(patch.read_text()) or {}
        if not isinstance(changes, dict):
            raise ConfigError(f"{patch}: expected a mapping of config changes")
        console.print(f"[bold]{run_id}[/] new attempt")
        n = add_attempt(ctx, console, changes, note, after_test)
    except EXPECTED as e:
        _fail(e, run_id)
    console.print(f"\nattempt {n} done; final attempt: {ctx.state.data['final_attempt']}")
    console.print(f"report: {ctx.run_dir / 'report.html'}")


@app.command()
def hybrid(
    run_id: str,
    fallback: Annotated[
        str | None, typer.Option("--fallback", help="Baseline to route to (default: few-shot).")
    ] = None,
) -> None:
    """Simulate hybrid routing on a finished run's saved predictions (free, no API calls): the tuned model
    answers unless its label's val F1 is below a threshold picked on val; then the fallback baseline does."""
    from ft.hybrid import HybridError, simulate

    run_dir = runs_dir() / run_id
    try:
        out = simulate(run_dir, fallback)
    except (HybridError, FileNotFoundError) as e:
        _fail(e, run_id)
    write_json(run_dir / "hybrid.json", out)
    t = out["test"]
    thr = out["threshold_val_f1"]
    routed = out["labels_routed"]
    rule = (
        "no routing"
        if not routed
        else f"route {len(routed)} answer(s) with val F1 "
        + ("of 0" if thr < 0.01 else f"below {thr:g}")
        + f" ({', '.join(routed[:5])}{', …' if len(routed) > 5 else ''})"
    )
    console.print(f"[bold]{run_id}[/] · {out['primary_metric']} · fallback {out['fallback']} · picked on val: {rule}")
    table = Table("model", "test accuracy", "test macro-F1", "cost / 1k", "routed to fallback")
    for name, key in (("tuned", "tuned"), (out["fallback"], "fallback"), ("hybrid", "hybrid")):
        r = t[key]
        share = f"{t['routed_share']:.1%}" if key == "hybrid" else ""
        cost = "" if r["cost_per_1k"] is None else f"${r['cost_per_1k']:.4f}"
        table.add_row(name, f"{r['accuracy']:.2f}", f"{r['macro_f1']:.2f}", cost, share)
    console.print(table)
    d, lo, hi = t["hybrid_minus_tuned"]
    console.print(f"hybrid − tuned ({out['primary_metric']}): {d:+.2f} [95% CI {lo:+.2f}, {hi:+.2f}]")
    console.print(f"[dim]{out['cost_note']}; details in {run_dir / 'hybrid.json'}[/]")


@app.command()
def status(run_id: str) -> None:
    """Stages, attempts, spend so far, live endpoints."""
    try:
        ctx = RunContext.open(run_id, yes=True)
    except EXPECTED as e:
        _fail(e)
    st = ctx.state.data
    t = Table(title=f"{run_id}{' (dry run)' if st['dry_run'] else ''}", show_edge=False)
    for col in ("stage", "status", "seconds", "note"):
        t.add_column(col)
    for k, v in st["stages"].items():
        t.add_row(k, v.get("status", ""), str(v.get("seconds", "")), v.get("error", "")[:80])
    console.print(t)
    for n, a in st["attempts"].items():
        console.print(f"attempt {n}: {a.get('external') or '{}'}")
    sim = " (simulated)" if st["dry_run"] else ""
    console.print(f"spend this run: ${ctx.state.spent():.4f}{sim} of ${ctx.spend.cap_usd:.2f}")
    console.print(f"real spend, all runs: ${total_real_spend(runs_dir()):.2f} of ${total_budget_usd():.2f}")
    live = ctx.state.live_endpoints()
    if live:
        for name, ep in live.items():
            console.print(f"[bold]LIVE ENDPOINT[/] {name} ({ep.get('id')}), billing since {ep.get('requested_at')}")
        console.print(f"tear down with `ft teardown {run_id}`")
    else:
        console.print("live endpoints (this run's records): none")
    if not st["dry_run"]:
        try:
            on_account = ctx.providers.server.list_ft_endpoints()
            names = ", ".join(e.name for e in on_account) or "none"
            provider = ctx.cfg.train.provider.capitalize()
            console.print(f"ft- endpoints on the {provider} account: {names}")
        except Exception as e:  # noqa: BLE001
            console.print(f"couldn't list the provider's endpoints: {e}")


def _teardown_recorded(ctx: RunContext, name: str, rec: dict) -> bool:
    """Tear down one endpoint recorded in a run's state; finds it by name if its id was never saved."""
    ep_id = rec.get("id")
    if not ep_id:
        found = serve.find_by_name(ctx, name)
        if found is None:
            ctx.state.record_endpoint(name, torn_down_at=now_iso(), note="never created on the provider")
            return True
        ep_id = found.id
    ep = Endpoint(
        id=ep_id,
        name=name,
        model=rec.get("model", ""),
        deployment_id=rec.get("deployment_id"),
        simulated=bool(rec.get("simulated")),
    )
    serve.teardown(ctx, ep, rec.get("attempt", 1))
    return bool(ctx.state.data["endpoints"][name].get("torn_down_at"))


@app.command()
def teardown(
    run_id: Annotated[str | None, typer.Argument()] = None,
    all_: Annotated[bool, typer.Option("--all", help="Every provider endpoint whose name starts with ft-.")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Delete endpoints created by ft."""
    if all_:
        _teardown_all(yes)
        return
    if not run_id:
        _fail(ValueError("give a run id or --all"))
    try:
        ctx = RunContext.open(run_id, yes=yes, confirm=_confirm)
    except EXPECTED as e:
        _fail(e)
    live = ctx.state.live_endpoints()
    if not live:
        console.print("no live endpoints recorded for this run")
        return
    for name, rec in live.items():
        if not yes and not _confirm(f"delete endpoint {name} ({rec.get('id', 'id unknown')})?"):
            continue
        ok = _teardown_recorded(ctx, name, rec)
        console.print(f"{name}: {'torn down' if ok else 'TEARDOWN FAILED'}")


def _account_servers() -> list:
    """A server for each training provider whose API key is set."""
    servers = []
    if os.environ.get("TOGETHER_API_KEY"):
        from ft.providers.together_provider import TogetherServer

        servers.append(TogetherServer())
    if os.environ.get("FIREWORKS_API_KEY"):
        from ft.providers.fireworks_provider import FireworksServer

        servers.append(FireworksServer())
    return servers


def _teardown_all(yes: bool) -> None:
    try:
        found = [(server, ep) for server in _account_servers() for ep in server.list_ft_endpoints()]
    except Exception as e:  # noqa: BLE001
        _fail(e)
    if not found:
        console.print("no ft- endpoints on the provider accounts")
        return
    owners = _endpoint_owners()
    for server, ep in found:
        run_id = owners.get(ep.name)
        if not yes and not _confirm(f"delete endpoint {ep.name} ({ep.id}, run {run_id or 'unknown'})?"):
            continue
        if run_id:  # tear down through the run so minutes billed and spend are recorded
            ctx = RunContext.open(run_id, yes=True)
            ok = _teardown_recorded(ctx, ep.name, {**ctx.state.data["endpoints"][ep.name], "id": ep.id})
        else:
            server.teardown(ep.id)
            ok = True
        console.print(f"{ep.name}: {'torn down' if ok else 'TEARDOWN FAILED'}")


def _endpoint_owners() -> dict[str, str]:
    owners = {}
    for p in runs_dir().glob("*/state.json"):
        try:
            data = read_json(p)
        except (OSError, ValueError):  # unreadable or half-written state: not this command's problem
            continue
        for name in data.get("endpoints", {}):
            owners[name] = data["run_id"]
    return owners


@app.command()
def report(run_id: str) -> None:
    """Regenerate the report from saved results."""
    try:
        ctx = RunContext.open(run_id, yes=True)
        run_stage(ctx, report_stage, None, console)
    except EXPECTED as e:
        _fail(e)


@app.command()
def plan(
    data_dir: Annotated[Path, typer.Argument(help="Raw data folder (no task.yaml yet).")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Heuristic fake agent, no network, no spend.")] = False,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing task.yaml.")] = False,
) -> None:
    """The agent looks at a data folder and writes task.yaml (or declines). Only agent calls are paid."""
    if not data_dir.is_dir():
        _fail(FileNotFoundError(f"{data_dir} is not a folder"))
    if task_path_for(data_dir).exists() and not force:
        _fail(FileExistsError(f"{task_path_for(data_dir)} exists; pass --force to replace it"))
    _plan_or_exit(data_dir, run_id or default_run_id(data_dir), dry_run)
    console.print(f"run it with `ft run {data_dir}`")


@app.command("simulate-logs")
def simulate_logs(
    task_dir: Annotated[Path, typer.Argument(help="Folder with a classify task.yaml (csv/jsonl source).")],
    out: Annotated[Path, typer.Option("--out", help="Where logs.jsonl and truth.jsonl go.")],
    model: Annotated[str, typer.Option(help="Frontier model that answers (the 'app' being logged).")] = "gpt-6-luna",
    n: Annotated[int, typer.Option(help="How many calls to log.")] = 5000,
    effort: Annotated[str, typer.Option(help="Reasoning effort.")] = "low",
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Fake model, no spend.")] = False,
) -> None:
    """Run a dataset's inputs through a frontier model with an app-style prompt and save the calls as logs,
    with the dataset's labels in a separate truth file (evaluation only). Paid: shows an estimate first."""
    from ft.simulate import simulate_logs as run_sim

    try:
        res = run_sim(
            _task_path(task_dir), model, n, out, effort=effort, dry_run=dry_run,
            confirm=(lambda _m: True) if yes else _confirm, log=_log,
        )  # fmt: skip
    except EXPECTED as e:
        _fail(e)
    console.print(
        f"wrote {res['written']:,} logs ({res['failed']} failed calls skipped) to {res['out']}, "
        f"${res['usd']:.4f}{' (simulated)' if dry_run else ''}; logged answers match truth on {res['logged_vs_truth']}%"
    )


@email_app.command("pull")
def email_pull(
    months: Annotated[int, typer.Option(help="How far back to read.")] = 12,
    min_age_days: Annotated[int, typer.Option(help="Skip mail younger than this (outcome not settled).")] = 7,
) -> None:
    """Export your Gmail history (read-only) to data/email/messages.jsonl, for training."""
    from ft.connectors import gmail

    out = gmail.export_path()
    try:
        meta = gmail.pull(months, min_age_days, out, log=lambda m: console.print(f"[dim]{m}[/]"))
    except gmail.GmailError as e:
        _fail(e)
    console.print(f"exported {meta['messages']} messages to {out} (stays on this machine)")


@email_app.command("apply")
def email_apply(
    run: str = typer.Option(..., "--run", help="Email run whose best model labels new mail."),
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the endpoint spend confirmation.")] = False,
    since: Annotated[str | None, typer.Option(help="ISO date to label from (default: last checkpoint).")] = None,
) -> None:
    """Label mail received since the last run: adds 'ft/Probably skip' and 'ft/Important'. Never archives,
    deletes or marks read. On demand only (no scheduling)."""
    from datetime import datetime

    from ft.connectors.gmail import GmailError
    from ft.email_apply import ApplyError, apply

    try:
        ctx = RunContext.open(run, yes=yes, confirm=_confirm)
        since_ms = int(datetime.fromisoformat(since).timestamp() * 1000) if since else None
        result = apply(ctx, since_ms=since_ms, log=lambda m: console.print(f"[dim]{m}[/]"))
    except (*EXPECTED, ApplyError, GmailError) as e:
        _fail(e, run)
    if result.get("messages"):
        console.print(f"labeled {result['messages']} messages: {result['predictions']}")


if __name__ == "__main__":
    app()
