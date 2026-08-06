# KG_v2 Debug 名词术语总表与关系图清单

> 本表把正式术语层与发现审核层放在同一视图中。未批准条目不参与确定性诊断和安全等价扩展。

## 总览

| 层级 | 概念/候选 | 变体 | 关系 |
|---|---:|---:|---:|
| 正式层 | 308 | 63 | 295 |
| 发现审核队列 | 216（待审 0） | 见候选类型 | 见候选类型 |

## 正式名词、变体与出边

> `[类型模板, 需实例证据]` 表示允许的连接或组成模式，并非对每个现场实例都成立；实例化必须由 BOM、型号、日志、照片或原文连接说明等证据确认。

| 规范名 | 类型 | 已批准变体 | 已批准关系 |
|---|---|---|---|
| CPU | `component` | — | — |
| CXP线缆 | `component` | CXP线(colloquial_alias) | part_of → CXP采集连接 [类型模板, 需实例证据] |
| DisplayPort线缆 | `component` | — | part_of → DisplayPort显示连接 [类型模板, 需实例证据] |
| GPU | `component` | — | part_of → 显卡 |
| HDMI线缆 | `component` | — | part_of → HDMI显示连接 [类型模板, 需实例证据] |
| M.2网卡 | `component` | — | installed_in → M.2接口 [类型模板, 需实例证据]; is_a → 网卡 |
| PLC控制器 | `component` | PLC(abbreviation) | — |
| SATA数据线 | `component` | — | part_of → SATA存储连接 [类型模板, 需实例证据] |
| SMEMA信号线缆 | `component` | — | part_of → SMEMA握手连接 [类型模板, 需实例证据] |
| USB线缆 | `component` | — | part_of → USB外设连接 [类型模板, 需实例证据] |
| U盘 | `component` | — | connected_via → USB外设连接 [类型模板, 需实例证据] |
| 串口线缆 | `component` | — | part_of → 串口控制连接 [类型模板, 需实例证据] |
| 主板 | `component` | — | connected_via → SATA存储连接 [类型模板, 需实例证据]; has_interface → SATA接口 [类型模板, 需实例证据] |
| 主板电池 | `component` | — | part_of → 主板 |
| 以太网线 | `component` | 网线(colloquial_alias) | part_of → 以太网连接 [类型模板, 需实例证据] |
| 传感器 | `component` | 感应器(colloquial_alias) | — |
| 光控 | `component` | — | — |
| 光源 | `component` | — | connected_to → ARM连接; connected_to → 光控; connected_to → 光控链路; connected_to → 硬件链路; connected_to → 通信链路 |
| 光源控制器 | `component` | — | connected_via → 串口控制连接 [类型模板, 需实例证据] |
| 光源控制板 | `component` | 光控板(colloquial_alias) | — |
| 内存 | `component` | — | — |
| 内存条 | `component` | — | — |
| 加密狗 | `component` | — | connected_via → USB外设连接 [类型模板, 需实例证据] |
| 压块 | `component` | — | — |
| 固态硬盘 | `component` | SSD(abbreviation) | is_a → 硬盘 |
| 图像采集卡 | `component` | 采集卡(colloquial_alias) | connected_via → CXP采集连接 [类型模板, 需实例证据]; has_interface → CXP接口 [类型模板, 需实例证据]; installed_in → PCIe接口 [类型模板, 需实例证据] |
| 工作站电源模块 | `component` | — | — |
| 工控机CPU散热 | `component` | — | part_of → 工控机 |
| 工控机USB外设 | `component` | — | part_of → 工控机 |
| 散热风扇 | `component` | 风扇(colloquial_alias) | — |
| 无线网卡 | `component` | — | is_a → 网卡 |
| 显卡 | `component` | — | connected_to → CUDA链路 |
| 显示器 | `component` | — | connected_via → DisplayPort显示连接 [类型模板, 需实例证据]; connected_via → HDMI显示连接 [类型模板, 需实例证据] |
| 机械硬盘 | `component` | — | is_a → 硬盘 |
| 气缸 | `component` | — | — |
| 电机 | `component` | — | — |
| 硬盘 | `component` | — | — |
| 系统盘 | `component` | — | connected_via → SATA存储连接 [类型模板, 需实例证据] |
| 网卡 | `component` | — | has_interface → 以太网接口 [类型模板, 需实例证据]; installed_in → PCIe接口 [类型模板, 需实例证据] |
| 调宽轴 | `component` | — | — |
| 轨道 | `component` | — | connected_to → 宽度调节; connected_to → 挡块机构; connected_to → 皮带机构 |
| 输送滚轮 | `component` | 滚轮(colloquial_alias) | — |
| 输送皮带 | `component` | 皮带(colloquial_alias) | — |
| 运动控制卡 | `component` | 运控卡(abbreviation) | installed_in → PCIe接口 [类型模板, 需实例证据] |
| 进板挡块 | `component` | 挡块(colloquial_alias) | — |
| 键盘 | `component` | — | connected_via → USB外设连接 [类型模板, 需实例证据] |
| 鼠标 | `component` | — | connected_via → USB外设连接 [类型模板, 需实例证据] |
| SensorInfo.xml | `configuration_file` | — | — |
| app.cfg.toml | `configuration_file` | — | — |
| calibration.yml | `configuration_file` | — | — |
| machine.toml | `configuration_file` | — | — |
| machine_status.ini | `configuration_file` | — | — |
| machined.toml | `configuration_file` | — | configuration_of → Machine服务 |
| tgo_cfg.ini | `configuration_file` | — | — |
| user.cfg | `configuration_file` | — | — |
| user.cfg.toml | `configuration_file` | — | — |
| CXP采集连接 | `connection` | — | uses_protocol → CoaXPress协议 [类型模板] |
| DisplayPort显示连接 | `connection` | — | uses_protocol → DisplayPort协议 [类型模板] |
| HDMI显示连接 | `connection` | — | uses_protocol → HDMI协议 [类型模板] |
| M.2扩展连接 | `connection` | — | — |
| PCIe扩展连接 | `connection` | — | uses_protocol → PCIe协议 [类型模板] |
| SATA存储连接 | `connection` | — | uses_protocol → SATA协议 [类型模板] |
| SMEMA握手连接 | `connection` | — | uses_protocol → SMEMA握手协议 [类型模板] |
| USB外设连接 | `connection` | — | uses_protocol → USB协议 [类型模板] |
| 串口控制连接 | `connection` | — | uses_protocol → 串行通信协议 [类型模板] |
| 以太网连接 | `connection` | — | uses_protocol → 以太网协议 [类型模板] |
| BOM物料清单 | `data_artifact` | BOM(abbreviation), BOM文件(exact_synonym), 物料清单(exact_synonym) | — |
| CAD工程数据 | `data_artifact` | CAD(abbreviation), CAD文件(exact_synonym) | — |
| Gerber工程数据 | `data_artifact` | Gerber(exact_synonym), Gerber文件(exact_synonym) | — |
| PROJ工程数据 | `data_artifact` | PROJ(abbreviation), proj文件(colloquial_alias) | — |
| RGB图 | `data_artifact` | — | is_a → 图像产物 |
| 图像产物 | `data_artifact` | — | — |
| 整板大图 | `data_artifact` | — | is_a → 图像产物 |
| 白图 | `data_artifact` | — | is_a → 图像产物 |
| host.db | `database_file` | — | — |
| host_v4.db | `database_file` | — | — |
| stats.db | `database_file` | — | — |
| user.cfg.db | `database_file` | — | — |
| meta.json | `diagnostic_artifact` | — | — |
| sysinfo.json | `diagnostic_artifact` | — | — |
| Windows核心驱动 | `driver` | — | connected_to → 显卡驱动 |
| 显卡驱动 | `driver` | — | driver_of → 显卡; is_a → 设备驱动程序 |
| 相机驱动 | `driver` | — | driver_of → 相机 [类型模板, 需实例证据]; is_a → 设备驱动程序 |
| 网卡驱动 | `driver` | — | driver_of → 网卡; is_a → 设备驱动程序 |
| 设备驱动程序 | `driver` | 驱动(colloquial_alias) | — |
| 2D相机 | `equipment` | — | is_a → 相机 |
| 3D相机 | `equipment` | — | is_a → 相机 |
| AOI设备 | `equipment` | AOI(abbreviation) | connected_via → SMEMA握手连接 [类型模板, 需实例证据]; has_interface → SMEMA接口 [类型模板, 需实例证据]; signals_to → 产线下游设备 [类型模板, 需实例证据, 方向=aoi_to_downstream] |
| D052/SI2020D | `equipment` | — | context_member → D052; context_member → SI2020D |
| SI1020E/AOI-7209 | `equipment` | — | context_member → AOI-7209; context_member → SI1020E |
| SI2020T/工控机 | `equipment` | — | context_member → SI2020T; context_member → 工控机 |
| SPI设备 | `equipment` | — | — |
| 上板机 | `equipment` | — | — |
| 产线上游设备 | `equipment` | — | connected_via → SMEMA握手连接 [类型模板, 需实例证据]; signals_to → AOI设备 [类型模板, 需实例证据, 方向=upstream_to_aoi] |
| 产线下游设备 | `equipment` | — | connected_via → SMEMA握手连接 [类型模板, 需实例证据] |
| 回流焊设备 | `equipment` | 回流焊(colloquial_alias) | — |
| 工业相机 | `equipment` | — | is_a → 相机 |
| 工控机 | `equipment` | IPC(abbreviation), 工业电脑(colloquial_alias), 工业计算机(exact_synonym) | — |
| 扫码枪 | `equipment` | 扫码器(colloquial_alias), 条码枪(colloquial_alias), 码枪(colloquial_alias) | — |
| 接驳台 | `equipment` | — | — |
| 收板机 | `equipment` | — | — |
| 相机 | `equipment` | — | connected_via → CXP采集连接 [类型模板, 需实例证据]; connected_via → 以太网连接 [类型模板, 需实例证据]; has_interface → CXP接口 [类型模板, 需实例证据] |
| 缓存机 | `equipment` | — | — |
| Jira | `external_system` | — | — |
| MES | `external_system` | — | — |
| BIOS | `firmware` | — | firmware_of → 主板; is_a → 设备固件 |
| 相机固件 | `firmware` | — | firmware_of → 相机 [类型模板, 需实例证据]; is_a → 设备固件 |
| 设备固件 | `firmware` | 固件(colloquial_alias) | — |
| 二维码 | `identifier` | — | is_a → 条码 |
| 序列号 | `identifier` | SN(abbreviation) | — |
| 料号 | `identifier` | — | — |
| 条码 | `identifier` | Barcode(english_equivalent) | — |
| CHIP | `inspection_object` | — | is_a → 器件 |
| IC | `inspection_object` | — | is_a → 器件 |
| LED | `inspection_object` | — | is_a → 器件 |
| Mark点 | `inspection_object` | 标记点(exact_synonym) | — |
| 器件 | `inspection_object` | 元件(exact_synonym), 元器件(exact_synonym) | — |
| 焊盘 | `inspection_object` | — | — |
| ARM连接 | `interface` | — | — |
| CXP接口 | `interface` | CXP(abbreviation), CXP接口(exact_synonym) | endpoint_of → CXP采集连接 [类型模板] |
| CXP链路 | `interface` | — | part_of → 3D相机 |
| DisplayPort接口 | `interface` | — | endpoint_of → DisplayPort显示连接 [类型模板] |
| HDMI接口 | `interface` | HDMI(abbreviation) | endpoint_of → HDMI显示连接 [类型模板] |
| M.2接口 | `interface` | M.2(exact_synonym) | endpoint_of → M.2扩展连接 [类型模板, 需实例证据] |
| PCIe接口 | `interface` | PCIe(abbreviation) | endpoint_of → PCIe扩展连接 [类型模板, 需实例证据] |
| SATA接口 | `interface` | — | endpoint_of → SATA存储连接 [类型模板] |
| SMEMA接口 | `interface` | SMEMA(abbreviation) | endpoint_of → SMEMA握手连接 [类型模板] |
| USB接口 | `interface` | USB(abbreviation), USB口(colloquial_alias), USB接口(exact_synonym) | endpoint_of → USB外设连接 [类型模板] |
| 串行接口 | `interface` | 串口(colloquial_alias) | endpoint_of → 串口控制连接 [类型模板] |
| 以太网接口 | `interface` | 网口(colloquial_alias) | endpoint_of → 以太网连接 [类型模板] |
| 外部触发 | `interface` | — | part_of → 扫码枪 |
| 联网 | `interface` | — | part_of → 扫码枪 |
| 日志产物 | `log_artifact` | 日志(exact_synonym) | — |
| 锡膏 | `material` | — | — |
| BGA | `package_type` | — | — |
| QFN | `package_type` | — | — |
| QFP | `package_type` | — | — |
| 1020D | `product_model` | — | — |
| 2030T | `product_model` | — | — |
| AOI-7209 | `product_model` | — | — |
| D052 | `product_model` | — | — |
| SI1020E | `product_model` | SI-1020E(exact_synonym) | — |
| SI2020D | `product_model` | SI-2020D(exact_synonym) | — |
| SI2020T | `product_model` | SI-2020T(exact_synonym) | model_of → 工控机 |
| SI2030 | `product_model` | SI-2030(exact_synonym) | — |
| T81 | `product_model` | — | — |
| CoaXPress协议 | `protocol` | — | — |
| DisplayPort协议 | `protocol` | — | — |
| HDMI协议 | `protocol` | — | — |
| PCIe协议 | `protocol` | — | — |
| SATA协议 | `protocol` | — | — |
| SMEMA握手协议 | `protocol` | — | — |
| USB协议 | `protocol` | — | — |
| 串行通信协议 | `protocol` | — | — |
| 以太网协议 | `protocol` | — | — |
| explorer.exe | `runtime_process` | — | runs_on → Windows |
| 软件运行进程 | `runtime_process` | — | — |
| Machine SDK | `sdk` | — | is_a → SDK |
| SDK | `sdk` | — | — |
| 相机 SDK | `sdk` | — | is_a → SDK; sdk_for → 相机 [类型模板, 需实例证据] |
| AOI主程序 | `software` | — | communicates_with → MES [类型模板, 需实例证据]; runs_on → AOI主站工控机 |
| Buddy | `software` | — | part_of → 模板 |
| CUDA | `software` | — | compatible_with → GPU [类型模板, 需实例证据] |
| DL算法 | `software` | DL(abbreviation) | — |
| Dism++ | `software` | — | — |
| Display Driver Uninstaller | `software` | DDU(abbreviation) | — |
| Hercules调试工具 | `software` | — | — |
| MVS | `software` | — | associated_with → 相机 |
| Machine服务 | `software` | — | runs_on → AOI主站工控机 |
| Microsoft Defender | `software` | Defender(abbreviation) | part_of → Windows |
| OCR算法 | `software` | OCR(abbreviation) | — |
| ODA算法 | `software` | ODA(abbreviation) | — |
| ODB算法 | `software` | ODB(abbreviation) | — |
| SPC | `software` | — | — |
| Windows | `software` | — | runs_on → 工作站 [类型模板, 需实例证据]; runs_on → 工控机 |
| Windows PE | `software` | WinPE(abbreviation) | is_a → Windows |
| 主程序 | `software` | — | — |
| 主程序配置 | `software` | — | — |
| 复判站软件 | `software` | — | runs_on → 复判工作站; runs_on → 复判站 |
| 复判站配置 | `software` | — | part_of → 主程序配置 |
| 导出 | `software` | — | part_of → SPC |
| 工控机Windows 内核 | `software` | — | runs_on → 工控机 |
| 数据采集 | `software` | — | part_of → SPC |
| 模板 | `software` | — | — |
| 模板与冷存储 | `software` | — | part_of → Buddy |
| 程序 | `software` | — | — |
| 维护站软件 | `software` | — | — |
| 编程软件 | `software` | — | runs_on → 编程工作站 |
| 运控程序 | `software` | — | — |
| MotionPanel.exe | `software_artifact` | — | — |
| cublas64_11.dll | `software_artifact` | — | artifact_of → CUDA |
| hercules.exe | `software_artifact` | — | artifact_of → Hercules调试工具 |
| kernelbase.dll | `software_artifact` | — | artifact_of → Windows |
| machined.exe | `software_artifact` | — | — |
| ntdll.dll | `software_artifact` | — | artifact_of → Windows |
| smt-aoi-maintenance-station.exe | `software_artifact` | — | artifact_of → 维护站软件 |
| smt-aoi.exe | `software_artifact` | — | artifact_of → AOI主程序 |
| AOI主站 | `station` | — | — |
| 复判站 | `station` | 复盘站(typo_variant) | connected_to → AOI设备 [类型模板, 需实例证据] |
| 编程站 | `station` | — | connected_to → AOI设备 [类型模板, 需实例证据] |
| 2D相机/光学链路 | `subsystem` | — | context_member → 2D相机; context_member → 光学链路 |
| 3D相机/CXP链路 | `subsystem` | — | context_member → 3D相机; context_member → CXP链路 |
| BIOS启动 | `subsystem` | — | part_of → 工控机 |
| Buddy/模板与冷存储 | `subsystem` | — | context_member → Buddy; context_member → 模板与冷存储 |
| Buddy/模板管理 | `subsystem` | — | context_member → Buddy; context_member → 模板管理 |
| CAD | `subsystem` | — | connected_to → 程序导入 |
| CAD/程序导入 | `subsystem` | — | context_member → CAD; context_member → 程序导入 |
| CT | `subsystem` | — | — |
| CUDA链路 | `subsystem` | — | — |
| Mark | `subsystem` | — | connected_to → 定位对齐 |
| Mark/定位对齐 | `subsystem` | — | context_member → Mark; context_member → 定位对齐 |
| SPC/数据采集/导出 | `subsystem` | — | context_member → SPC; context_member → 导出; context_member → 数据采集 |
| Windows核心驱动/显卡驱动 | `subsystem` | — | context_member → Windows核心驱动; context_member → 显卡驱动 |
| 主程序/启动链路 | `subsystem` | — | context_member → 主程序; context_member → 主程序启动链路 |
| 主程序/运行稳定性 | `subsystem` | — | context_member → 主程序; context_member → 运行稳定性 |
| 主程序/配置链路 | `subsystem` | — | context_member → 主程序; context_member → 主程序配置链路 |
| 主程序启动链路 | `subsystem` | — | part_of → 主程序 |
| 主程序配置/复判站配置 | `subsystem` | — | context_member → 主程序配置; context_member → 复判站配置 |
| 主程序配置链路 | `subsystem` | — | part_of → 主程序 |
| 光学成像 | `subsystem` | — | part_of → 相机 |
| 光学链路 | `subsystem` | — | part_of → 2D相机 |
| 光控链路 | `subsystem` | — | — |
| 光源/光控/ARM连接 | `subsystem` | — | context_member → ARM连接; context_member → 光控; context_member → 光源 |
| 光源/光控链路 | `subsystem` | — | context_member → 光控链路; context_member → 光源 |
| 光源/硬件链路 | `subsystem` | — | context_member → 光源; context_member → 硬件链路 |
| 光源/通信链路 | `subsystem` | — | context_member → 光源; context_member → 通信链路 |
| 初始化链路 | `subsystem` | — | part_of → 相机 |
| 坏板标记 | `subsystem` | — | connected_to → 流程链路 |
| 坏板标记/流程链路 | `subsystem` | — | context_member → 坏板标记; context_member → 流程链路 |
| 复判 | `subsystem` | — | connected_to → 显示性能; connected_to → 板卡加载; connected_to → 结果保存 |
| 复判/显示性能 | `subsystem` | — | context_member → 复判; context_member → 显示性能 |
| 复判/板卡加载 | `subsystem` | — | context_member → 复判; context_member → 板卡加载 |
| 复判/结果保存 | `subsystem` | — | context_member → 复判; context_member → 结果保存 |
| 复判站/软件 | `subsystem` | — | context_member → 复判站; context_member → 复判站软件 |
| 外设链路 | `subsystem` | — | part_of → 扫码枪 |
| 存储链路 | `subsystem` | — | — |
| 定位对齐 | `subsystem` | — | — |
| 宽度调节 | `subsystem` | — | — |
| 工控机/BIOS启动 | `subsystem` | — | context_member → BIOS启动; context_member → 工控机 |
| 工控机/CPU散热 | `subsystem` | — | context_member → 工控机; context_member → 工控机CPU散热 |
| 工控机/USB外设 | `subsystem` | — | context_member → 工控机; context_member → 工控机USB外设 |
| 工控机/Windows 内核 | `subsystem` | — | context_member → 工控机; context_member → 工控机Windows 内核 |
| 工控机/启动链路 | `subsystem` | — | context_member → 工控机; context_member → 工控机启动链路 |
| 工控机/显示链路 | `subsystem` | — | context_member → 工控机; context_member → 工控机显示链路 |
| 工控机/系统运行稳定性 | `subsystem` | — | context_member → 工控机; context_member → 工控机系统运行稳定性 |
| 工控机/网络链路 | `subsystem` | — | context_member → 工控机; context_member → 工控机网络链路 |
| 工控机启动链路 | `subsystem` | — | part_of → 工控机 |
| 工控机显示链路 | `subsystem` | — | part_of → 工控机 |
| 工控机系统运行稳定性 | `subsystem` | — | part_of → 工控机 |
| 工控机网络链路 | `subsystem` | — | part_of → 工控机 |
| 扫码枪/外设链路 | `subsystem` | — | context_member → 外设链路; context_member → 扫码枪 |
| 扫码枪/联网/外部触发 | `subsystem` | — | context_member → 外部触发; context_member → 扫码枪; context_member → 联网 |
| 拼图 | `subsystem` | — | connected_to → 角度标定 |
| 拼图/角度标定 | `subsystem` | — | context_member → 拼图; context_member → 角度标定 |
| 拼图链路 | `subsystem` | — | part_of → 相机 |
| 挡块机构 | `subsystem` | — | — |
| 显卡/CUDA链路 | `subsystem` | — | context_member → CUDA链路; context_member → 显卡 |
| 显示性能 | `subsystem` | — | — |
| 板卡加载 | `subsystem` | — | part_of → 程序 |
| 检测框 | `subsystem` | — | connected_to → 模型输出 |
| 检测框/模型输出 | `subsystem` | — | context_member → 检测框; context_member → 模型输出 |
| 模型输出 | `subsystem` | — | — |
| 模板/Buddy | `subsystem` | — | context_member → Buddy; context_member → 模板 |
| 模板管理 | `subsystem` | — | part_of → Buddy |
| 气压链路 | `subsystem` | — | — |
| 气路 | `subsystem` | — | connected_to → 气压链路 |
| 气路/气压链路 | `subsystem` | — | context_member → 气压链路; context_member → 气路 |
| 流程链路 | `subsystem` | — | — |
| 皮带机构 | `subsystem` | — | — |
| 相机/光学成像 | `subsystem` | — | context_member → 光学成像; context_member → 相机 |
| 相机/初始化链路 | `subsystem` | — | context_member → 初始化链路; context_member → 相机 |
| 相机/拼图链路 | `subsystem` | — | context_member → 拼图链路; context_member → 相机 |
| 相机/采集链路 | `subsystem` | — | context_member → 相机; context_member → 相机采集链路 |
| 相机采集链路 | `subsystem` | — | part_of → 相机 |
| 硬件链路 | `subsystem` | — | — |
| 磁盘 | `subsystem` | — | connected_to → 存储链路 |
| 磁盘/存储链路 | `subsystem` | — | context_member → 存储链路; context_member → 磁盘 |
| 程序/板卡加载 | `subsystem` | — | context_member → 板卡加载; context_member → 程序 |
| 程序导入 | `subsystem` | — | — |
| 结果保存 | `subsystem` | — | — |
| 节拍 | `subsystem` | — | connected_to → CT |
| 节拍/CT | `subsystem` | — | context_member → CT; context_member → 节拍 |
| 角度标定 | `subsystem` | — | — |
| 调试 | `subsystem` | — | — |
| 轨道/宽度调节 | `subsystem` | — | context_member → 宽度调节; context_member → 轨道 |
| 轨道/挡块机构 | `subsystem` | — | context_member → 挡块机构; context_member → 轨道 |
| 轨道/皮带机构 | `subsystem` | — | context_member → 皮带机构; context_member → 轨道 |
| 轨道链路 | `subsystem` | — | — |
| 软件使用 | `subsystem` | — | connected_to → 调试 |
| 软件使用/调试 | `subsystem` | — | context_member → 调试; context_member → 软件使用 |
| 运控 | `subsystem` | — | connected_to → 运控启动链路 |
| 运控/启动链路 | `subsystem` | — | context_member → 运控; context_member → 运控启动链路 |
| 运控卡 | `subsystem` | — | connected_to → 初始化链路 |
| 运控卡/初始化链路 | `subsystem` | — | context_member → 初始化链路; context_member → 运控卡 |
| 运控启动链路 | `subsystem` | — | — |
| 运控程序/启动链路 | `subsystem` | — | context_member → 运控程序; context_member → 运控程序启动链路 |
| 运控程序启动链路 | `subsystem` | — | part_of → 运控程序 |
| 运行稳定性 | `subsystem` | — | part_of → 主程序 |
| 进出板 | `subsystem` | — | connected_to → 轨道链路 |
| 进出板/轨道链路 | `subsystem` | — | context_member → 轨道链路; context_member → 进出板 |
| 通信链路 | `subsystem` | — | — |
| PCB | `workpiece` | 线路板(colloquial_alias), PCB板(exact_synonym) | processed_by → AOI设备 |
| 标准金板 | `workpiece` | 金板(colloquial_alias) | — |
| 镂空板 | `workpiece` | — | is_a → PCB |
| AOI主站工控机 | `workstation` | — | deployed_at → AOI主站; is_a → 工作站; is_a → 工控机; part_of → AOI设备 [类型模板, 需实例证据] |
| 复判工作站 | `workstation` | — | deployed_at → 复判站; is_a → 工作站 |
| 工作站 | `workstation` | — | connected_via → DisplayPort显示连接 [类型模板, 需实例证据]; connected_via → HDMI显示连接 [类型模板, 需实例证据]; connected_via → USB外设连接 [类型模板, 需实例证据]; connected_via → 串口控制连接 [类型模板, 需实例证据]; connected_via → 以太网连接 [类型模板, 需实例证据]; has_component → CPU [类型模板, 需实例证据]; has_component → 主板 [类型模板, 需实例证据]; has_component → 内存 [类型模板, 需实例证据]; has_component → 系统盘 [类型模板, 需实例证据]; has_component → 网卡 [类型模板, 需实例证据]; has_interface → DisplayPort接口 [类型模板, 需实例证据]; has_interface → HDMI接口 [类型模板, 需实例证据]; has_interface → USB接口 [类型模板, 需实例证据]; has_interface → 串行接口 [类型模板, 需实例证据]; has_interface → 以太网接口 [类型模板, 需实例证据]; powered_by → 工作站电源模块 [类型模板, 需实例证据] |
| 编程工作站 | `workstation` | — | deployed_at → 编程站; is_a → 工作站 |

## 待审核名词与变体大表

| 类型 | 名称/叫法 | 建议目标 | 建议关系 | 语料次数 | 风险 | 状态 |
|---|---|---|---|---:|---|---|

详细语料证据与审核字段见 `../review_queue/noun_discovery_candidates.json`。
