# KG_v2 多源名词发现与审核清单

> 该清单来自群聊、文档 Chunk 和技术支持记录；全部条目在人工审核前均为非权威候选，不可锁定 Variant 或生成诊断动作。

## 汇总

| 项目 | 数量 |
|---|---:|
| 去重群聊记录 | 137043 |
| 文档 Chunk | 4711 |
| 去重支持记录 | 4563 |
| 新名词概念 | 27 |
| 变体叫法 | 31 |
| 名词关系 | 85 |
| 语料共现关联 | 73 |
| 总候选 | 216 |

## 新名词概念

| 名词 | 类型 | 总次数 | 群聊 | 文档 | 支持记录 | 来源种类 | 风险 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|---|
| 板卡 | `workpiece` | 8476 | 7745 | 386 | 345 | 3 | `high` | `deferred` |
| Qt | `software` | 1495 | 1479 | 8 | 8 | 3 | `high` | `deferred` |
| 大板 | `workpiece` | 751 | 711 | 21 | 19 | 3 | `high` | `deferred` |
| 电源 | `component` | 436 | 364 | 38 | 34 | 3 | `high` | `deferred` |
| 治具 | `inspection_object` | 435 | 377 | 29 | 29 | 3 | `medium` | `deferred` |
| 顶升 | `component` | 212 | 184 | 17 | 11 | 3 | `medium` | `deferred` |
| 弯板 | `workpiece` | 142 | 99 | 22 | 21 | 3 | `medium` | `deferred` |
| SI1020T | `product_model` | 137 | 135 | 1 | 1 | 3 | `medium` | `deferred` |
| SI252T | `product_model` | 126 | 118 | 4 | 4 | 3 | `medium` | `deferred` |
| 急停 | `component` | 112 | 92 | 10 | 10 | 3 | `medium` | `deferred` |
| 数据库 | `software` | 108 | 97 | 7 | 4 | 3 | `high` | `deferred` |
| EAP | `external_system` | 88 | 86 | 1 | 1 | 3 | `high` | `deferred` |
| SI2020C | `product_model` | 65 | 65 | 0 | 0 | 1 | `medium` | `deferred` |
| SI2030L | `product_model` | 57 | 57 | 0 | 0 | 1 | `medium` | `deferred` |
| SI1020 | `product_model` | 47 | 45 | 1 | 1 | 3 | `medium` | `deferred` |
| SI1020C | `product_model` | 45 | 43 | 1 | 1 | 3 | `medium` | `deferred` |
| DP | `interface` | 35 | 27 | 4 | 4 | 3 | `high` | `deferred` |
| SI252L | `product_model` | 26 | 26 | 0 | 0 | 1 | `medium` | `deferred` |
| SI2020 | `product_model` | 18 | 18 | 0 | 0 | 1 | `medium` | `deferred` |
| 薄板 | `workpiece` | 18 | 9 | 5 | 4 | 3 | `medium` | `deferred` |
| SY2600D | `product_model` | 10 | 10 | 0 | 0 | 1 | `medium` | `deferred` |
| SI2020L | `product_model` | 9 | 9 | 0 | 0 | 1 | `medium` | `deferred` |
| SI252 | `product_model` | 9 | 9 | 0 | 0 | 1 | `medium` | `deferred` |
| SI1030T | `product_model` | 8 | 8 | 0 | 0 | 1 | `medium` | `deferred` |
| SI1020L | `product_model` | 7 | 7 | 0 | 0 | 1 | `medium` | `deferred` |
| 1200D | `product_model` | 6 | 6 | 0 | 0 | 1 | `medium` | `deferred` |
| SI1030 | `product_model` | 3 | 3 | 0 | 0 | 1 | `medium` | `deferred` |

## 变体叫法

| 现场叫法 | 建议规范名 | 建议关系 | 总次数 | 风险 | 状态 |
|---|---|---|---:|---|---|
| RGB | RGB图 | `abbreviation` | 859 | `high` | `deferred` |
| 252T | SI252T | `abbreviation` | 83 | `medium` | `deferred` |
| 2020D | SI2020D | `abbreviation` | 82 | `medium` | `deferred` |
| 1020T | SI1020T | `abbreviation` | 77 | `medium` | `deferred` |
| SI-2030T | 2030T | `exact_synonym` | 45 | `medium` | `deferred` |
| SI-1020T | SI1020T | `exact_synonym` | 43 | `medium` | `deferred` |
| SI-1020 | SI1020 | `exact_synonym` | 42 | `medium` | `deferred` |
| 2030L | SI2030L | `abbreviation` | 40 | `medium` | `deferred` |
| SI-1020D | 1020D | `exact_synonym` | 38 | `medium` | `deferred` |
| SI-2020C | SI2020C | `exact_synonym` | 36 | `medium` | `deferred` |
| SI-252T | SI252T | `exact_synonym` | 32 | `medium` | `deferred` |
| 1020C | SI1020C | `abbreviation` | 31 | `medium` | `deferred` |
| 2020C | SI2020C | `abbreviation` | 26 | `medium` | `deferred` |
| 1020E | SI1020E | `abbreviation` | 24 | `medium` | `deferred` |
| 2020t | SI2020T | `abbreviation` | 23 | `medium` | `deferred` |
| 252L | SI252L | `abbreviation` | 21 | `medium` | `deferred` |
| SI-2020 | SI2020 | `exact_synonym` | 18 | `medium` | `deferred` |
| SI-2030L | SI2030L | `exact_synonym` | 13 | `medium` | `deferred` |
| SI1020D | 1020D | `exact_synonym` | 11 | `medium` | `deferred` |
| SI-252 | SI252 | `exact_synonym` | 8 | `medium` | `deferred` |
| 2020L | SI2020L | `abbreviation` | 7 | `medium` | `deferred` |
| SI-1020C | SI1020C | `exact_synonym` | 7 | `medium` | `deferred` |
| 1030T | SI1030T | `abbreviation` | 6 | `medium` | `deferred` |
| 2600D | SY2600D | `abbreviation` | 6 | `medium` | `deferred` |
| SI-1020L | SI1020L | `exact_synonym` | 6 | `medium` | `deferred` |
| SI-1030 | SI1030 | `exact_synonym` | 3 | `medium` | `deferred` |
| SI2030T | 2030T | `exact_synonym` | 3 | `medium` | `deferred` |
| SY-2600D | SY2600D | `abbreviation` | 3 | `medium` | `deferred` |
| SI-1030T | SI1030T | `exact_synonym` | 2 | `medium` | `deferred` |
| SI-252L | SI252L | `exact_synonym` | 2 | `medium` | `deferred` |
| SI_1020 | SI1020 | `exact_synonym` | 2 | `medium` | `deferred` |

## 名词关系

| 起点 | 关系 | 终点 | 总次数 | 风险 | 状态 |
|---|---|---|---:|---|---|
| `workpiece:板卡` | `is_a` | `workpiece:pcb` | 8476 | `high` | `rejected` |
| `inspection_object:器件` | `part_of` | `workpiece:pcb` | 7598 | `low` | `rejected` |
| `inspection_object:焊盘` | `part_of` | `workpiece:pcb` | 4013 | `low` | `rejected` |
| `external_system:mes` | `communicates_with` | `equipment:aoi设备` | 3276 | `medium` | `rejected` |
| `sdk:sdk` | `runs_on` | `equipment:工控机` | 2285 | `medium` | `rejected` |
| `inspection_object:mark点` | `part_of` | `workpiece:pcb` | 1874 | `low` | `rejected` |
| `software:buddy` | `runs_on` | `equipment:aoi设备` | 1851 | `medium` | `rejected` |
| `subsystem:cad` | `input_of` | `station:编程站` | 1768 | `high` | `rejected` |
| `component:加密狗` | `connected_to` | `equipment:工控机` | 1447 | `low` | `rejected` |
| `component:轨道` | `part_of` | `equipment:aoi设备` | 1338 | `low` | `rejected` |
| `identifier:料号` | `identifies` | `workpiece:pcb` | 1307 | `medium` | `rejected` |
| `identifier:序列号` | `identifies` | `workpiece:pcb` | 1267 | `high` | `rejected` |
| `identifier:条码` | `identifies` | `workpiece:pcb` | 1130 | `medium` | `rejected` |
| `data_artifact:整板大图` | `output_of` | `equipment:aoi设备` | 1086 | `high` | `rejected` |
| `driver:设备驱动程序` | `runs_on` | `equipment:工控机` | 1024 | `high` | `rejected` |
| `data_artifact:白图` | `output_of` | `equipment:aoi设备` | 986 | `medium` | `rejected` |
| `component:显卡` | `part_of` | `equipment:工控机` | 882 | `low` | `rejected` |
| `component:硬盘` | `part_of` | `equipment:工控机` | 869 | `low` | `rejected` |
| `equipment:接驳台` | `connected_to` | `equipment:aoi设备` | 853 | `low` | `rejected` |
| `workpiece:标准金板` | `is_a` | `workpiece:pcb` | 848 | `medium` | `rejected` |
| `component:传感器` | `part_of` | `equipment:aoi设备` | 813 | `medium` | `rejected` |
| `inspection_object:led` | `part_of` | `workpiece:pcb` | 804 | `medium` | `rejected` |
| `component:网卡` | `part_of` | `equipment:工控机` | 781 | `low` | `rejected` |
| `inspection_object:chip` | `part_of` | `workpiece:pcb` | 774 | `high` | `rejected` |
| `workpiece:大板` | `is_a` | `workpiece:pcb` | 751 | `high` | `rejected` |
| `component:输送皮带` | `part_of` | `equipment:aoi设备` | 695 | `low` | `rejected` |
| `inspection_object:ic` | `part_of` | `workpiece:pcb` | 573 | `medium` | `rejected` |
| `equipment:收板机` | `connected_to` | `equipment:aoi设备` | 491 | `low` | `rejected` |
| `component:内存条` | `part_of` | `equipment:工控机` | 439 | `low` | `rejected` |
| `component:cpu` | `part_of` | `equipment:工控机` | 436 | `low` | `rejected` |
| `component:电源` | `part_of` | `equipment:工控机` | 436 | `high` | `rejected` |
| `inspection_object:治具` | `connected_to` | `equipment:aoi设备` | 435 | `medium` | `rejected` |
| `data_artifact:rgb图` | `output_of` | `equipment:aoi设备` | 431 | `medium` | `rejected` |
| `component:主板` | `part_of` | `equipment:工控机` | 427 | `low` | `rejected` |
| `component:显示器` | `connected_to` | `equipment:工控机` | 359 | `medium` | `rejected` |
| `component:鼠标` | `connected_to` | `equipment:工控机` | 351 | `low` | `rejected` |
| `component:键盘` | `connected_to` | `equipment:工控机` | 326 | `low` | `rejected` |
| `component:进板挡块` | `part_of` | `equipment:aoi设备` | 295 | `low` | `rejected` |
| `data_artifact:gerber工程数据` | `input_of` | `station:编程站` | 275 | `low` | `rejected` |
| `package_type:qfn` | `part_of` | `workpiece:pcb` | 265 | `medium` | `rejected` |
| `package_type:bga` | `part_of` | `workpiece:pcb` | 227 | `medium` | `rejected` |
| `component:顶升` | `part_of` | `equipment:aoi设备` | 212 | `medium` | `rejected` |
| `component:气缸` | `part_of` | `equipment:aoi设备` | 202 | `low` | `rejected` |
| `interface:cxp接口` | `connected_to` | `equipment:相机` | 197 | `low` | `rejected` |
| `component:图像采集卡` | `connected_to` | `equipment:相机` | 159 | `medium` | `rejected` |
| `component:散热风扇` | `part_of` | `equipment:工控机` | 151 | `medium` | `rejected` |
| `workpiece:弯板` | `is_a` | `workpiece:pcb` | 142 | `medium` | `rejected` |
| `component:gpu` | `part_of` | `equipment:工控机` | 140 | `low` | `rejected` |
| `software:cuda` | `runs_on` | `equipment:工控机` | 139 | `medium` | `rejected` |
| `product_model:si1020t` | `model_of` | `equipment:aoi设备` | 137 | `medium` | `rejected` |
| `firmware:bios` | `runs_on` | `equipment:工控机` | 133 | `low` | `rejected` |
| `product_model:si252t` | `model_of` | `equipment:aoi设备` | 126 | `medium` | `rejected` |
| `component:急停` | `part_of` | `equipment:aoi设备` | 112 | `medium` | `rejected` |
| `package_type:qfp` | `part_of` | `workpiece:pcb` | 103 | `medium` | `rejected` |
| `component:电机` | `part_of` | `equipment:aoi设备` | 100 | `medium` | `rejected` |
| `external_system:eap` | `communicates_with` | `equipment:aoi设备` | 88 | `high` | `rejected` |
| `component:光源控制器` | `part_of` | `equipment:aoi设备` | 85 | `low` | `rejected` |
| `configuration_file:machine.toml` | `part_of` | `software:主程序` | 79 | `low` | `rejected` |
| `component:压块` | `part_of` | `equipment:aoi设备` | 78 | `medium` | `rejected` |
| `interface:smema接口` | `connected_to` | `equipment:aoi设备` | 67 | `medium` | `rejected` |
| `product_model:si2020c` | `model_of` | `equipment:aoi设备` | 65 | `medium` | `rejected` |
| `data_artifact:bom物料清单` | `input_of` | `station:编程站` | 57 | `medium` | `rejected` |
| `product_model:si2030l` | `model_of` | `equipment:aoi设备` | 57 | `medium` | `rejected` |
| `software:winpe` | `runs_on` | `equipment:工控机` | 47 | `medium` | `rejected` |
| `product_model:si1020` | `model_of` | `equipment:aoi设备` | 47 | `medium` | `rejected` |
| `product_model:si1020c` | `model_of` | `equipment:aoi设备` | 45 | `medium` | `rejected` |
| `equipment:上板机` | `connected_to` | `equipment:aoi设备` | 45 | `low` | `rejected` |
| `component:plc控制器` | `part_of` | `equipment:aoi设备` | 44 | `high` | `rejected` |
| `component:输送滚轮` | `part_of` | `equipment:aoi设备` | 34 | `medium` | `rejected` |
| `product_model:si252l` | `model_of` | `equipment:aoi设备` | 26 | `medium` | `rejected` |
| `software:display driver uninstaller` | `runs_on` | `equipment:工控机` | 20 | `medium` | `rejected` |
| `product_model:si2020` | `model_of` | `equipment:aoi设备` | 18 | `medium` | `rejected` |
| `workpiece:薄板` | `is_a` | `workpiece:pcb` | 18 | `medium` | `rejected` |
| `component:运动控制卡` | `part_of` | `equipment:aoi设备` | 18 | `low` | `rejected` |
| `component:光源控制板` | `part_of` | `equipment:aoi设备` | 17 | `low` | `rejected` |
| `software:microsoft defender` | `runs_on` | `equipment:工控机` | 16 | `medium` | `rejected` |
| `component:调宽轴` | `part_of` | `equipment:aoi设备` | 16 | `low` | `rejected` |
| `product_model:sy2600d` | `model_of` | `equipment:aoi设备` | 10 | `medium` | `rejected` |
| `product_model:si2020l` | `model_of` | `equipment:aoi设备` | 9 | `medium` | `rejected` |
| `product_model:si252` | `model_of` | `equipment:aoi设备` | 9 | `medium` | `rejected` |
| `product_model:si1030t` | `model_of` | `equipment:aoi设备` | 8 | `medium` | `rejected` |
| `product_model:si1020l` | `model_of` | `equipment:aoi设备` | 7 | `medium` | `rejected` |
| `product_model:1200d` | `model_of` | `equipment:aoi设备` | 6 | `medium` | `rejected` |
| `software:dism++` | `runs_on` | `equipment:工控机` | 3 | `low` | `rejected` |
| `product_model:si1030` | `model_of` | `equipment:aoi设备` | 3 | `medium` | `rejected` |

## 语料共现关联（非结构事实）

> 共现只说明两个名词经常出现在同一条记录中；不能直接推出`part_of`、`connected_to` 或因果关系。

| 名词 A | 名词 B | 同记录次数 | Jaccard | 来源种类 | 风险 | 状态 |
|---|---|---:|---:|---:|---|---|
| 程序 | 板卡 | 894 | 0.1571 | 3 | `high` | `rejected` |
| 器件 | 板卡 | 880 | 0.1569 | 3 | `high` | `rejected` |
| Jira | 器件 | 806 | 0.1486 | 3 | `high` | `rejected` |
| SPC | 导出 | 723 | 0.3178 | 3 | `high` | `rejected` |
| Jira | 板卡 | 697 | 0.1180 | 3 | `high` | `rejected` |
| Jira | 程序 | 678 | 0.1201 | 3 | `high` | `rejected` |
| 器件 | 焊盘 | 642 | 0.1630 | 3 | `high` | `rejected` |
| 模板 | 程序 | 470 | 0.1123 | 3 | `high` | `rejected` |
| RGB图 | 白图 | 295 | 0.3923 | 3 | `high` | `rejected` |
| Buddy | 主程序 | 227 | 0.1333 | 3 | `high` | `rejected` |
| Buddy | 复判站 | 204 | 0.0819 | 3 | `high` | `rejected` |
| 加密狗 | 复判站 | 187 | 0.0839 | 3 | `high` | `rejected` |
| MES | 条码 | 170 | 0.1426 | 3 | `high` | `rejected` |
| 主程序 | 复判站 | 165 | 0.0613 | 3 | `high` | `rejected` |
| MES | Buddy | 149 | 0.0901 | 3 | `high` | `rejected` |
| CHIP | 焊盘 | 145 | 0.0793 | 3 | `high` | `rejected` |
| IC | SPC | 141 | 0.0486 | 3 | `high` | `rejected` |
| Mark点 | 模板 | 139 | 0.0678 | 3 | `high` | `rejected` |
| SDK | PE | 135 | 0.1557 | 3 | `high` | `rejected` |
| SDK | Qt | 133 | 0.1752 | 3 | `high` | `rejected` |
| 相机 | 主程序 | 126 | 0.0692 | 3 | `high` | `rejected` |
| Qt | PE | 125 | 0.1452 | 3 | `high` | `rejected` |
| 扫码枪 | MES | 122 | 0.1089 | 3 | `high` | `rejected` |
| 焊盘 | 锡膏 | 115 | 0.0689 | 3 | `high` | `rejected` |
| LED | SDK | 110 | 0.1092 | 3 | `high` | `rejected` |
| LED | PE | 109 | 0.0989 | 3 | `high` | `rejected` |
| 网卡 | 相机 | 98 | 0.0954 | 3 | `high` | `rejected` |
| 扫码枪 | 条码 | 96 | 0.1424 | 3 | `high` | `rejected` |
| 加密狗 | IC | 93 | 0.0478 | 3 | `high` | `rejected` |
| 光源 | 相机 | 87 | 0.0602 | 3 | `high` | `rejected` |
| 键盘 | 鼠标 | 83 | 0.3018 | 3 | `high` | `rejected` |
| 料号 | 模板 | 82 | 0.0392 | 3 | `high` | `rejected` |
| 光源 | RGB图 | 81 | 0.0760 | 3 | `high` | `rejected` |
| LED | Qt | 68 | 0.0658 | 3 | `high` | `rejected` |
| 料号 | CHIP | 67 | 0.0636 | 3 | `high` | `rejected` |
| 料号 | IC | 65 | 0.0319 | 3 | `high` | `rejected` |
| 电源 | 工控机 | 60 | 0.1056 | 3 | `high` | `rejected` |
| 轨道 | Mark点 | 60 | 0.0467 | 3 | `high` | `rejected` |
| CPU | 内存 | 58 | 0.0870 | 3 | `high` | `rejected` |
| 内存 | 硬盘 | 56 | 0.0775 | 3 | `high` | `rejected` |
| 内存条 | 工控机 | 54 | 0.0993 | 3 | `high` | `rejected` |
| 内存 | Windows | 50 | 0.0791 | 3 | `high` | `rejected` |
| 主板 | 工控机 | 48 | 0.0823 | 3 | `high` | `rejected` |
| 内存条 | 显卡 | 47 | 0.1093 | 3 | `high` | `rejected` |
| 主板 | 显卡 | 43 | 0.0921 | 3 | `high` | `rejected` |
| 接驳台 | 缓存机 | 41 | 0.0930 | 3 | `high` | `rejected` |
| CPU | GPU | 39 | 0.1439 | 3 | `high` | `rejected` |
| 二维码 | 条码 | 38 | 0.0642 | 3 | `high` | `rejected` |
| 显卡 | 显卡驱动 | 35 | 0.1057 | 3 | `high` | `rejected` |
| 传感器 | 镂空板 | 34 | 0.1073 | 3 | `high` | `rejected` |
| 光源 | 白图 | 34 | 0.0302 | 3 | `high` | `rejected` |
| 传感器 | 轨道 | 33 | 0.0366 | 3 | `high` | `rejected` |
| Mark点 | 弯板 | 31 | 0.0436 | 3 | `high` | `rejected` |
| 气缸 | 顶升 | 28 | 0.1687 | 3 | `high` | `rejected` |
| 轨道 | 收板机 | 28 | 0.0333 | 3 | `high` | `rejected` |
| AOI设备 | 接驳台 | 25 | 0.0435 | 3 | `high` | `rejected` |
| 硬盘 | 大板 | 25 | 0.0423 | 2 | `high` | `rejected` |
| 网卡 | 网卡驱动 | 25 | 0.1037 | 3 | `high` | `rejected` |
| CPU | 电源 | 24 | 0.0558 | 2 | `high` | `rejected` |
| 主板 | BIOS | 24 | 0.0816 | 3 | `high` | `rejected` |
| 传感器 | 接驳台 | 24 | 0.0385 | 3 | `high` | `rejected` |
| CHIP | DL算法 | 20 | 0.0331 | 3 | `high` | `rejected` |
| 显卡驱动 | CUDA | 17 | 0.1241 | 2 | `high` | `rejected` |
| 显示器 | 电源 | 15 | 0.0426 | 3 | `high` | `rejected` |
| 机械硬盘 | 硬盘 | 15 | 0.0497 | 3 | `high` | `rejected` |
| BIOS | Windows | 14 | 0.0598 | 3 | `high` | `rejected` |
| 显卡驱动 | Windows | 14 | 0.0547 | 3 | `high` | `rejected` |
| 显示器 | DP | 14 | 0.0467 | 3 | `high` | `rejected` |
| 无线网卡 | 网卡 | 12 | 0.0469 | 3 | `high` | `rejected` |
| BGA | QFN | 10 | 0.0341 | 3 | `high` | `rejected` |
| 内存条 | BIOS | 10 | 0.0364 | 3 | `high` | `rejected` |
| 显示器 | 鼠标 | 10 | 0.0316 | 3 | `high` | `rejected` |
| 键盘 | USB接口 | 8 | 0.0417 | 3 | `high` | `rejected` |

完整上下文样例、来源路径、记录 ID、审核字段和稳定 `content_hash` 见 `../review_queue/noun_discovery_candidates.json`。
