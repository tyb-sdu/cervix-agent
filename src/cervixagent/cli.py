from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .audit import AuditError, create_p1_01_baseline, verify_audit_run
from .config import (
    load_compound_ingestion_schema,
    load_public_sources,
    load_workflow,
    workflow_checksum,
)
from .data import DataError, build_test_dataset, fetch_sources
from .doctor import checks_as_dicts, run_checks
from .ingest import IngestError, ingest_engineering_test, verify_ingestion_run
from .project import ProjectError, init_project, load_project
from .staging import StagingError, stage_public_snapshots, verify_staging_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cervixagent",
        description="CervixAgent 终端科研工作流执行器",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init", help="初始化一个本地研究项目")
    init_cmd.add_argument("path", type=Path)
    init_cmd.add_argument("--name", default="cervixagent-research")
    init_cmd.add_argument("--force", action="store_true")

    doctor_cmd = sub.add_parser("doctor", help="检查本地软件和计算环境")
    doctor_cmd.add_argument("--path", type=Path, default=Path.cwd())
    doctor_cmd.add_argument("--json", action="store_true")

    status_cmd = sub.add_parser("status", help="显示项目状态")
    status_cmd.add_argument("path", type=Path)
    status_cmd.add_argument("--json", action="store_true")

    workflow_cmd = sub.add_parser("workflow", help="显示锁定的三阶段工作流")
    workflow_cmd.add_argument("--json", action="store_true")

    data_cmd = sub.add_parser("data", help="获取并校验白名单内的公开科研数据")
    data_sub = data_cmd.add_subparsers(dest="data_command", required=True)
    sources_cmd = data_sub.add_parser("sources", help="显示允许使用的公开数据源")
    sources_cmd.add_argument("--json", action="store_true")
    fetch_cmd = data_sub.add_parser("fetch", help="下载结构或天然产物公开数据")
    fetch_cmd.add_argument(
        "selection",
        help="all、structures、natural-products 或具体数据源键",
    )
    fetch_cmd.add_argument("--project", type=Path, default=Path.cwd())
    fetch_cmd.add_argument("--force", action="store_true")
    fetch_cmd.add_argument("--json", action="store_true")
    test_cmd = data_sub.add_parser("build-test", help="生成可追溯的工程联调测试集")
    test_cmd.add_argument("--project", type=Path, default=Path.cwd())
    test_cmd.add_argument("--size", type=int, default=500)
    test_cmd.add_argument("--seed", type=int, default=20260715)
    test_cmd.add_argument("--json", action="store_true")

    audit_cmd = sub.add_parser("audit", help="建立或验证可检测篡改的本地审计记录")
    audit_sub = audit_cmd.add_subparsers(dest="audit_command", required=True)
    baseline_cmd = audit_sub.add_parser("baseline", help="封存 P1-01 环境基线")
    baseline_cmd.add_argument("--project", type=Path, default=Path.cwd())
    baseline_cmd.add_argument("--label", default="p1-01-baseline")
    baseline_cmd.add_argument("--json", action="store_true")
    verify_cmd = audit_sub.add_parser("verify", help="验证封存运行记录的完整性")
    verify_cmd.add_argument("--project", type=Path, default=Path.cwd())
    verify_cmd.add_argument("--run-id")
    verify_cmd.add_argument("--json", action="store_true")

    ingest_cmd = sub.add_parser("ingest", help="执行当前 P1-02 数据入库工作")
    ingest_sub = ingest_cmd.add_subparsers(dest="ingest_command", required=True)
    ingest_contract_cmd = ingest_sub.add_parser(
        "contract", help="显示 P1-02 入库数据契约和禁止操作"
    )
    ingest_contract_cmd.add_argument("--json", action="store_true")
    ingest_test_cmd = ingest_sub.add_parser(
        "test", help="对工程测试集做 RDKit 解析与规范 SMILES 序列化"
    )
    ingest_test_cmd.add_argument("--project", type=Path, default=Path.cwd())
    ingest_test_cmd.add_argument("--input", type=Path)
    ingest_test_cmd.add_argument("--label", default="engineering-test")
    ingest_test_cmd.add_argument("--json", action="store_true")
    ingest_verify_cmd = ingest_sub.add_parser("verify", help="验证 P1-02 入库输出完整性")
    ingest_verify_cmd.add_argument("--project", type=Path, default=Path.cwd())
    ingest_verify_cmd.add_argument("--run-id")
    ingest_verify_cmd.add_argument("--json", action="store_true")
    stage_cmd = ingest_sub.add_parser(
        "stage-public", help="流式暂存 COCONUT 与 LOTUS 公开快照"
    )
    stage_cmd.add_argument("--project", type=Path, default=Path.cwd())
    stage_cmd.add_argument("--label", default="coconut-lotus-staging")
    stage_cmd.add_argument("--batch-size", type=int, default=10000)
    stage_cmd.add_argument("--limit-per-source", type=int)
    stage_cmd.add_argument("--json", action="store_true")
    stage_verify_cmd = ingest_sub.add_parser(
        "stage-verify", help="验证两源暂存数据库及封存结果"
    )
    stage_verify_cmd.add_argument("--project", type=Path, default=Path.cwd())
    stage_verify_cmd.add_argument("--run-id")
    stage_verify_cmd.add_argument("--json", action="store_true")
    return parser


def _print_workflow(as_json: bool) -> None:
    workflow = load_workflow()
    if as_json:
        print(json.dumps(workflow, ensure_ascii=False, indent=2))
        return

    print(f"工作流：{workflow['name']}")
    print(f"协议锁定：{'是' if workflow['locked'] else '否'}")
    print(f"SHA-256：{workflow_checksum(workflow)}")
    for phase in workflow["phases"]:
        print(f"\n{phase['id']}  {phase['name']}")
        for step in phase["steps"]:
            gate = " [人工审批]" if step.get("human_gate") else ""
            print(f"  {step['id']}  {step['name']}{gate}")


def _print_checks(as_json: bool, path: Path) -> None:
    checks = run_checks(path)
    if as_json:
        print(json.dumps(checks_as_dicts(checks), ensure_ascii=False, indent=2))
        return
    for item in checks:
        print(f"{item.status:>13}  {item.name:<28} {item.detail} [{item.required_for}]")


def _print_sources(as_json: bool) -> None:
    registry = load_public_sources()
    if as_json:
        print(json.dumps(registry, ensure_ascii=False, indent=2))
        return
    print("仅允许以下公开数据源：")
    for key, source in registry["sources"].items():
        print(f"  {key:<26} {source['record_id']:<32} {source['license']}")


def _download_progress(source_key: str, byte_count: int) -> None:
    if byte_count % (10 * 1024 * 1024) < 1024 * 1024:
        print(f"  {source_key}: {byte_count / 1024 / 1024:.1f} MiB", file=sys.stderr)


def _staging_progress(source_name: str, record_count: int) -> None:
    print(f"  {source_name}: 已处理 {record_count:,} 条", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            state = init_project(args.path, args.name, force=args.force)
            print(f"项目已初始化：{state['project_root']}")
            print(f"当前步骤：{state['current_step']}")
            print("服务器状态：当前不可登录，已记录但不阻塞本地开发")
            return 0
        if args.command == "doctor":
            _print_checks(args.json, args.path)
            return 0
        if args.command == "status":
            state = load_project(args.path)
            if args.json:
                print(json.dumps(state, ensure_ascii=False, indent=2))
            else:
                print(f"项目：{state['project_name']}")
                print(f"状态：{state['status']}")
                print(f"当前：{state['current_phase']} / {state['current_step']}")
                print(f"协议锁定：{'是' if state['workflow_locked'] else '否'}")
                print(f"服务器：{state['server_access']}")
            return 0
        if args.command == "workflow":
            _print_workflow(args.json)
            return 0
        if args.command == "data":
            if args.data_command == "sources":
                _print_sources(args.json)
                return 0
            if args.data_command == "fetch":
                results = fetch_sources(
                    args.project,
                    args.selection,
                    force=args.force,
                    progress=None if args.json else _download_progress,
                )
                if args.json:
                    print(json.dumps(results, ensure_ascii=False, indent=2))
                else:
                    for item in results:
                        mib = item["bytes"] / 1024 / 1024
                        print(
                            f"{item['source_key']}: {item['status']}, "
                            f"{mib:.1f} MiB, SHA-256 {item['sha256']}"
                        )
                return 0
            if args.data_command == "build-test":
                manifest = build_test_dataset(args.project, size=args.size, seed=args.seed)
                if args.json:
                    print(json.dumps(manifest, ensure_ascii=False, indent=2))
                else:
                    print(f"测试集已生成：{manifest['output']['relative_path']}")
                    print(f"记录数：{manifest['record_count']}；来源：{manifest['source_counts']}")
                    print("用途边界：工程联调，不是正式筛选库或科学结论")
                return 0
        if args.command == "audit":
            if args.audit_command == "baseline":
                result = create_p1_01_baseline(args.project, label=args.label)
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print(f"P1-01 审计基线：{result['run_id']}")
                    print(f"封存聚合 SHA-256：{result['aggregate_sha256']}")
                    print(f"已进入：{result['current_step']}")
                    print("保证级别：本地可检测篡改，不等同于不可变存储")
                return 0 if result["ready_for_p1_02"] else 3
            if args.audit_command == "verify":
                result = verify_audit_run(args.project, run_id=args.run_id)
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print(f"运行记录：{result['run_id']}")
                    print(f"完整性：{'通过' if result['valid'] else '失败'}")
                    for error in result["errors"]:
                        print(f"  - {error}")
                return 0 if result["valid"] else 3
        if args.command == "ingest":
            if args.ingest_command == "contract":
                contract = load_compound_ingestion_schema()
                if args.json:
                    print(json.dumps(contract, ensure_ascii=False, indent=2))
                else:
                    print(f"契约：{contract['name']}")
                    print(f"范围：{contract['scope']}")
                    print(
                        "科学过滤："
                        + ("允许" if contract["scientific_filtering_allowed"] else "禁止")
                    )
                    print("正式完成所需来源：" + ", ".join(contract["source_requirements"]["required_sources"]))
                    print(f"ECNPDB：{contract['source_requirements']['ecnpdb_current_status']}")
                return 0
            if args.ingest_command == "test":
                result = ingest_engineering_test(
                    args.project,
                    input_path=args.input,
                    label=args.label,
                )
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    counts = result["counts"]
                    print(f"P1-02 工程入库试运行：{result['run_id']}")
                    print(
                        f"输入 {counts['input_records']}；有效 {counts['valid_records']}；"
                        f"无效 {counts['invalid_records']}；规范结构重复 {counts['canonical_duplicate_records']}"
                    )
                    print(f"输出：{result['relative_path']}")
                    print("P1-02 正式状态：未完成；P1-04 筛选：未启动")
                return 0
            if args.ingest_command == "verify":
                result = verify_ingestion_run(args.project, run_id=args.run_id)
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print(f"入库运行：{result['run_id']}")
                    print(f"完整性：{'通过' if result['valid'] else '失败'}")
                    print(
                        f"范围：{result.get('scope')}；"
                        f"P1-02 正式完成：{result.get('formal_p1_02_complete')}"
                    )
                    for error in result["errors"]:
                        print(f"  - {error}")
                return 0 if result["valid"] else 3
            if args.ingest_command == "stage-public":
                result = stage_public_snapshots(
                    args.project,
                    label=args.label,
                    batch_size=args.batch_size,
                    limit_per_source=args.limit_per_source,
                    progress=None if args.json else _staging_progress,
                )
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    counts = result["counts"]
                    print(f"P1-02 两源暂存：{result['run_id']}")
                    print(
                        f"输入 {counts['input_records']:,}；有效 {counts['valid_records']:,}；"
                        f"无效 {counts['invalid_records']:,}；"
                        f"规范结构重复 {counts['canonical_duplicate_records']:,}"
                    )
                    print(f"唯一有效结构：{counts['unique_valid_structures']:,}")
                    print(f"封存聚合 SHA-256：{result['aggregate_sha256']}")
                    print("范围：COCONUT+LOTUS 暂存；P1-02 正式状态仍为未完成")
                return 0
            if args.ingest_command == "stage-verify":
                result = verify_staging_run(args.project, run_id=args.run_id)
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print(f"两源暂存：{result['run_id']}")
                    print(f"完整性：{'通过' if result['valid'] else '失败'}")
                    print(
                        f"SQLite：{result.get('sqlite_quick_check')}；"
                        f"P1-02 正式完成：{result.get('formal_p1_02_complete')}"
                    )
                    for error in result["errors"]:
                        print(f"  - {error}")
                return 0 if result["valid"] else 3
    except (ProjectError, DataError, AuditError, IngestError, StagingError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    parser.error("未知命令")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
