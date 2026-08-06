from __future__ import annotations

from debug_agent_system.core.contracts import CheckNode, LockedSubgraph, SolutionNode


class DiagnosisGenerationAgent:
    """O-GEN: render locked-subgraph facts into user-facing text."""

    def render_check(self, subgraph: LockedSubgraph, check: CheckNode, *, compact: bool = True, check_ids: list[str] | None = None) -> str:
        checks = [check] if compact else self._checks_to_render(subgraph, check, check_ids=check_ids)
        lines = [
            f"命中故障：{subgraph.label}。",
            f"优先检查：{check.label}",
            check.how_to_check,
        ]
        if len(checks) > 1:
            lines.append("推荐排查清单（人工确认，不自动执行高风险动作）：")
            for idx, item in enumerate(checks, start=1):
                lines.append(f"{idx}. {item.label}：{item.how_to_check}")
                for hint in self._historical_outcome_hints(item):
                    lines.append(f"   - 历史结果：{hint}")
                for hint in self._checklist_hints(item):
                    lines.append(f"   - 检查项：{hint}")
                solutions = subgraph.solutions_by_check.get(item.check_id, [])
                for hint in self._manual_hints(item, solutions):
                    lines.append(f"   - 操作要点：{hint}")
                for solution in solutions[:2]:
                    if solution.content:
                        lines.append(f"   - 若确认相关现象：{solution.content}")
        return "\n".join(lines)

    def render_resolution(self, subgraph: LockedSubgraph, solution: SolutionNode | None) -> str:
        if solution and solution.content:
            hints = self._solution_outcome_hints(solution)
            suffix = "\n" + "\n".join(f"历史验证：{hint}" for hint in hints) if hints else ""
            return f"诊断闭环：{subgraph.label}\n建议处理：{solution.content}{suffix}"
        return f"诊断闭环：{subgraph.label}\n当前检查已确认解决，但 KG 中缺少结构化 Solution，建议补充入 review_queue。"

    def render_escalation(self, subgraph: LockedSubgraph, reason: str) -> str:
        return f"已走完当前 KG 检查链但未闭环：{subgraph.label}。原因：{reason}。建议升级并携带已排查步骤和日志。"

    def _checks_to_render(self, subgraph: LockedSubgraph, current: CheckNode, check_ids: list[str] | None = None, limit: int = 8) -> list[CheckNode]:
        if check_ids:
            by_id = {check.check_id: check for check in subgraph.checks}
            ordered = [by_id[check_id] for check_id in check_ids if check_id in by_id]
            if current.check_id not in {check.check_id for check in ordered}:
                ordered.insert(0, current)
            return ordered[:limit]
        ordered = [current]
        for check in subgraph.checks:
            if check.check_id == current.check_id:
                continue
            ordered.append(check)
            if len(ordered) >= limit:
                break
        return ordered

    def _historical_outcome_hints(self, check: CheckNode) -> list[str]:
        hints: list[str] = []
        for outcome in check.payload.get("_historical_outcomes") or []:
            if not isinstance(outcome, dict):
                continue
            action = str(outcome.get("action_label") or check.label)
            outcome_type = str(outcome.get("outcome_type") or "")
            if outcome_type == "verified_fix":
                hints.append(f"{action} 曾验证有效。")
            elif outcome_type == "ineffective":
                hints.append(f"{action} 曾在历史案例中无效；若用户已尝试，应降低该路径权重。")
            elif outcome_type == "partial_temporary":
                hints.append(f"{action} 只产生临时缓解，需继续验证是否复发。")
            elif outcome_type == "mitigation_observed":
                hints.append(f"{action} 仅观察到缓解，不能当最终根因。")
            elif outcome_type == "pending_validation":
                if outcome.get("high_cost") or outcome.get("destructive"):
                    hints.append(f"{action} 待验证且成本/风险高，只能建议人工确认。")
                else:
                    hints.append(f"{action} 待验证。")
            elif outcome_type == "diagnostic_method":
                hints.append(f"{action} 是诊断方法，不是修复方案。")
        return _dedupe(hints)

    def _solution_outcome_hints(self, solution: SolutionNode) -> list[str]:
        hints: list[str] = []
        for outcome in solution.payload.get("_historical_outcomes") or []:
            if not isinstance(outcome, dict):
                continue
            outcome_type = str(outcome.get("outcome_type") or "")
            action = str(outcome.get("action_label") or solution.content)
            if outcome_type == "verified_fix":
                hints.append(f"{action} 曾验证有效。")
            elif outcome_type == "ineffective":
                hints.append(f"{action} 在相似历史条件下曾无效，不能直接当最终解。")
            elif outcome_type == "partial_temporary":
                hints.append(f"{action} 曾短时有效后复发，需要复验。")
            elif outcome_type == "pending_validation":
                hints.append(f"{action} 尚待验证，高成本/高风险时必须人工确认。")
        return _dedupe(hints)

    def _manual_hints(self, check: CheckNode, solutions: list[SolutionNode]) -> list[str]:
        text = " ".join([check.label, check.how_to_check, *[s.content for s in solutions]]).lower()
        hints: list[str] = []
        if any(x in text for x in ("清 cmos", "清cmos", "cr2032", "cmos电池", "bios 设置", "bios设置")):
            hints.append("完全断开电源后再清除 CMOS；按主板手册短接 CLR_CMOS 跳线 5-10 秒，或取下纽扣电池等待约 5 分钟；恢复后重新设置启动顺序、通电启动策略和显示输出优先级。")
            hints.append("如果是更新 BIOS 或刷写后清 CMOS 仍无效，不继续盲目改设置，携带 BIOS 版本和现场现象升级技术服务处理。")
        if any(x in text for x in ("pwr_sw", "电源保护", "短路", "24pin", "供电链路", "风扇转一下就停")):
            hints.append("排查掉电保护时先拔掉 USB、网线、串口等非必要外设；断开硬盘电源线/数据线（包含 M.2 硬盘），只保留 PWR_SW 开机线并断开前面板 USB/音频线。")
            hints.append("重新插紧 CPU 辅助供电线和主板 24Pin 主供电线；检查主板铜柱、金属异物和接地异常；最后用确认完好的 PSU/电源交叉验证。")
        if any(x in text for x in ("交流输入电压", "市电", "电网", "ups", "接地", "电源线老化")):
            hints.append("使用万用表测量插座电压是否处于工控机电源额定范围；检查工控机接地是否可靠，电源线是否破损、老化或烧蚀。")
            hints.append("如果大功率设备启停导致电网波动，优先隔离干扰源，并配置 UPS 或工业级稳压器；潮湿粉尘环境下同步检查灰尘积累导致的短路风险。")
        if any(x in text for x in ("氧化", "接触不良", "板卡插槽", "扩展卡", "金手指", "插槽灰尘")):
            hints.append("清洁内存条、显卡和扩展卡金手指；清理 PCIe 插槽和内存插槽灰尘；重新插拔主板内部电源线、数据线、前面板连接线和 CPU 风扇接口，并确认所有连接插紧牢固。")
        if any(x in text for x in ("uefi", "legacy", "多硬盘", "启动顺序", "系统盘", "引导损坏", "pe", "m.2", "sata")):
            hints.append("多硬盘启动冲突时先断开所有非系统硬盘，只保留系统盘；检查 BIOS Boot Mode（UEFI/Legacy/Auto）是否匹配系统盘，并确认第一启动设备是系统硬盘。")
            hints.append("BIOS 不识别硬盘时重新插拔硬盘并更换 M.2/SATA 端口；能识别但不能启动时使用 Windows PE 或 DiskGenius 检查并修复引导记录，必要时禁用非系统硬盘引导分区。")
        if any(x in text for x in ("散热", "cpu 温度", "cpu温度", "风扇转速", "occt", "内存稳定", "主板电容")):
            hints.append("使用 OCCT、BIOS 或硬件监控工具查看 CPU/系统温度；检查 CPU 风扇和机箱风扇是否转动、转速是否过低，并清理散热器和机箱风道灰尘。")
            hints.append("间歇性黑屏/死机还要做 Windows 内存诊断或 MemTest86，并观察主板电容是否鼓包漏液；必要时用替换法验证电源负载升高时输出是否稳定。")
        return _dedupe(hints)

    def _checklist_hints(self, check: CheckNode) -> list[str]:
        """Deterministic manual checklist hints for non-interactive diagnosis.

        These are still user-visible instructions only.  They make the manual
        branch output concrete enough for field use and eval, without changing
        traversal, automation, or KG merge behavior.
        """

        mapping = {
            "check:industrial-pc-no-boot-step2a": [
                "确认插座有电。",
                "检查电源线两端插接牢固并尝试更换电源线。",
                "确认主机后面板电源开关已打开。",
                "检查主板前面板PWR_SW开机信号线是否松动或脱落。",
                "判断工控机电源PSU故障可能性。",
                "拔掉USB网线串口等所有非必要外部设备。",
                "断开硬盘电源线数据线包括M.2硬盘。",
                "只保留PWR_SW开机线并断开前面板USB音频线。",
                "重新插紧CPU辅助供电线和主板24Pin主供电线。",
                "检查主板铜柱金属异物和接地导致短路。",
                "替换确认完好的电源判断电源故障或主板短路。",
                "清洁内存条显卡扩展卡金手指。",
                "清理PCIe插槽和内存插槽灰尘。",
                "重新插拔主板内部电源线数据线前面板连接线和CPU风扇接口。",
                "确认所有连接插紧牢固。",
            ],
            "check:industrial-pc-no-boot-step2b": [
                "确认显示器电源指示灯亮起并处于开机状态。",
                "检查DP HDMI VGA视频线两端插紧。",
                "视频线连接到主板核显输出接口而不是独立显卡接口。",
                "选择正确的显示器信号输入源。",
                "更换视频线或显示器测试。",
            ],
            "check:industrial-pc-no-boot-step2c": [
                "观察主板Debug灯CPU DRAM VGA BOOT。",
                "CPU灯长亮检查CPU供电线并谨慎重新安装CPU。",
                "DRAM灯长亮时断电拔下内存擦拭金手指并单条测试。",
                "VGA灯长亮检查显卡供电重新插拔显卡或改接主板视频接口。",
                "BOOT灯长亮进入BIOS检查硬盘识别。",
                "等待至少1到2分钟排除大内存自检时间长。",
            ],
            "check:industrial-pc-no-boot-step2d": [
                "断开所有非系统硬盘只保留系统盘。",
                "检查BIOS Boot Mode启动模式为UEFI Legacy或Auto是否匹配系统盘。",
                "使用系统安装U盘或PE工具修复引导记录。",
                "确认非系统硬盘引导分区已清除或在BIOS禁用。",
            ],
            "check:industrial-pc-no-boot-step3": [
                "完全断电并拔掉电源线。",
                "短接CLR_CMOS跳线5到10秒或扣下纽扣电池等待5分钟。",
                "移除新添加的内存硬盘扩展卡等硬件再测试。",
                "更新BIOS后清除CMOS无效需联系技术服务处理BIOS版本或刷写失败。",
                "清除CMOS后重新设置通电启动策略和显示输出优先级。",
                "观察主板Debug灯CPU DRAM VGA BOOT。",
                "CPU灯长亮检查CPU供电线并谨慎重新安装CPU。",
                "DRAM灯长亮时断电拔下内存擦拭金手指并单条测试。",
                "VGA灯长亮检查显卡供电重新插拔显卡或改接主板视频接口。",
                "BOOT灯长亮进入BIOS检查硬盘识别。",
            ],
            "check:industrial-pc-blue-screen-step2boot": [
                "进入BIOS设置界面。",
                "检查Boot或Startup菜单第一启动设备是系统硬盘。",
                "无法识别硬盘时重新插拔硬盘并更换M.2或SATA端口。",
                "能识别硬盘时使用Windows PE或DiskGenius检查并修复系统引导。",
            ],
            "check:industrial-pc-freeze-black-screen-step3b1": [
                "使用OCCT查看CPU和系统温度是否过高。",
                "检查CPU风扇机箱风扇是否转动和转速是否过低。",
                "检查散热器和机箱风道灰尘并清理。",
                "替换法测试电源负载升高时是否输出不稳。",
                "运行Windows内存诊断测试内存稳定性。",
                "观察主板电容是否鼓包漏液。",
            ],
            "check:industrial-pc-unexpected-reboot-step3b2": [
                "使用万用表测量插座电压是否在工控机电源额定范围。",
                "检查工控机是否良好可靠接地。",
                "检查电源线是否破损老化烧蚀。",
                "电网波动大时配备UPS或工业级稳压器。",
                "检查内部灰尘是否在潮湿环境下引起短路。",
            ],
        }
        return list(mapping.get(check.check_id, []))


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out
