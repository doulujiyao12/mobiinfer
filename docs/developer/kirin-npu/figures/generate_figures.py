#!/usr/bin/env python3
"""生成麒麟 NPU 学习手册的八张中文教学配图。

依赖 Matplotlib 和中文字体；--font 可指定 OTF / TTF / TTC 文件。
默认生成脚本目录中的 SVG；--preview-dir 生成检查用 PNG。
"""
import argparse
from pathlib import Path
from figure_primitives import (Figure, configure_font, INK, MUTED, LIGHT, PAPER,
                               TEAL, BLUE, PURPLE, GOLD, CORAL, COLORS, FILLS)


def software_stack():
    d = Figure(1, "从模型到硅片：NPU 软件栈", "沿编译产物与调用路径，理解应用、运行时和设备之间的边界。", "分层剖面", 980)
    d.tag(62, 154, 130, "应用可控制", "cpu")
    d.tag(208, 154, 150, "系统提供实现", "neutral")
    d.text(1140, 168, "示意分层 · 不代表物理核数", 12, MUTED, ha="right")
    layers = [(218, 92, "01", "应用与模型", "模型文件、输入组织、线程与生命周期", "cpu"),
              (342, 190, "02", "构图与模型导入", None, "tool"),
              (564, 84, "03", "设备运行时 / 系统服务", "准备执行资源、管理设备模型并提交任务", "neutral"),
              (680, 84, "04", "驱动与固件", "资源映射、任务调度与硬件执行控制", "neutral")]
    for y, h, n, title, body, kind in layers:
        d.poly([(72, y + h), (895, y + h), (914, y + h + 12), (91, y + h + 12)], "#E6ECF2")
        d.rect(72, y, 823, h, FILLS[kind], LIGHT, radius=8, z=3)
        d.text(94, y + 28, n, 13, COLORS[kind], weight="bold")
        d.text(135, y + 28, title, 21, weight="bold")
        if body:
            d.text(135, y + h - 24, body, 15, MUTED)
    d.file(664, 240, 60, 48, "file")
    d.graph(755, 252, 95, 28)
    d.rule(94, 389, 780)
    d.graph(121, 427, 92, 48)
    d.label(240, 423, "在线：MNN / HiAI IR", "设备端构图 → 编译 → 模型加载", PURPLE, 18)
    d.file(530, 419, 44, 69, "file")
    d.label(595, 423, "离线：OMC + 外置权重", "主机提前编译 → NNRT / HiAI 初始化", GOLD, 18)
    for x, color in [(317, PURPLE), (703, GOLD)]:
        d.arrow([(x, 310), (x, 342)], color)
        d.arrow([(x, 532), (x, 564)], color)
    d.arrow([(482, 648), (482, 680)], TEAL)
    d.chip(301, 795, 362, 92)
    for x in [322, 390, 458, 526]:
        d.matrix(x, 818, 3, 3, cell=11, gap=3, fill="#73B9A7")
    d.text(614, 841, "…", 28, "white", ha="center")
    d.text(94, 839, "麒麟 NPU", 24, TEAL, weight="bold")
    d.arrow([(482, 764), (482, 787)], TEAL)
    d.text(696, 837, "执行计算图中的\n受支持算子", 16, MUTED)
    d.line([(938, 224), (938, 522)], LIGHT, 2)
    d.label(966, 250, "应用侧接口", "MNN / HiAI / NNRT", BLUE, 18)
    d.label(966, 585, "运行时边界", "查询能力与返回码\n管理句柄和张量", MUTED, 18)
    d.label(966, 704, "硬件边界", "公开 API 提交任务\n设备完成实际计算", TEAL, 18)
    d.note(72, 901, 1074, "计算图可覆盖完整模型，也可按算子支持与内存工作集划分；编译方式与计算划分分别设计。", h=32)
    return d


def online_cache():
    d = Figure(2, "缓存命中后，哪些工作被省去了？", "三种状态沿相同执行阶段比较；条带长度仅示意步骤，不是实测耗时。", "状态与时间轴", 900)
    for x, label, kind in [(65, "模型 / 缓存读取", "cpu"), (250, "在线编译", "tool"), (435, "设备加载 / I/O", "file"), (680, "图执行", "npu")]:
        d.dot_label(x, 162, label, kind)
    names, kinds = ["读取模型", "构图 / 编译", "加载 / I/O", "Process"], ["cpu", "tool", "file", "npu"]
    xx, ww = [325, 530, 735, 940], 180
    for i, (y, title, sub, active) in enumerate([
            (244, "无缓存", "冷启动", [True, True, True, True]),
            (414, "磁盘命中", "新模型实例", [True, False, True, True]),
            (584, "设备驻留", "同一实例复用", [False, False, False, True])]):
        d.step(76, y + 10, i + 1)
        d.text(105, y + 5, title, 25, weight="bold")
        d.text(105, y + 40, sub, 15, MUTED)
        d.line([(316, y + 67), (1132, y + 67)], LIGHT, 1)
        for j, (x, label, kind) in enumerate(zip(xx, names, kinds)):
            fill = FILLS[kind] if active[j] else "#FAFBFC"
            d.rect(x, y - 18, ww, 69, fill, LIGHT if not active[j] else None, radius=5)
            if active[j]:
                d.rect(x, y - 18, ww, 5, COLORS[kind], radius=1, z=2)
                d.text(x + ww / 2, y + 17, "读缓存 / 模型" if i == 1 and j == 0 else label,
                       17, COLORS[kind], ha="center", weight="bold")
            else:
                d.line([(x + 18, y + 15), (x + ww - 18, y + 15)], LIGHT, 1, dashed=True)
                d.text(x + ww / 2, y + 16, "已省去", 14, "#8996A5", ha="center",
                       bbox={"facecolor": "#FAFBFC", "edgecolor": "none", "pad": 3})
        d.arrow([(1123, y + 67), (1140, y + 67)], MUTED, 1)
        if i == 0:
            d.text(325, y + 98, "编译结果落盘，供下一次模型加载复用", 15, MUTED)
            d.line([(620, y + 58), (620, y + 84), (670, y + 84)], PURPLE, 1)
        elif i == 1:
            d.text(325, y + 98, "普通 OM 命中仍可能读源模型；完整索引可进一步省去源权重准备。", 15, MUTED)
        else:
            d.text(325, y + 98, "输入准备与数据拷贝仍需执行；模型实例保持驻留", 15, MUTED)
    d.rect(60, 758, 1080, 78, PAPER, radius=8)
    d.text(84, 784, "预加载的作用", 18, TEAL, weight="bold")
    d.text(84, 813, "将设备加载 / 权重初始化前移到模型加载阶段。", 15, MUTED)
    d.text(646, 784, "磁盘命中", 18, GOLD, weight="bold")
    d.arrow([(754, 785), (806, 785)], MUTED)
    d.text(828, 784, "设备实例就绪", 18, TEAL, weight="bold")
    d.text(646, 813, "两者之间仍有模型加载和资源准备。", 15, MUTED)
    return d


def operator_routes():
    d = Figure(3, "算子适配与内核编程：从语义到数据", "先复用已有 IR 算子；需要自定义内核时，再设计分块、搬运与同步。", "分块与流水", 1030)
    d.rect(56, 160, 1090, 80, PAPER, radius=9)
    d.step(83, 200, 1)
    d.text(113, 185, "定义算子规格", 21, weight="bold")
    d.text(113, 217, "计算语义、shape、dtype、layout、广播与尾块边界", 16, MUTED)
    d.tag(852, 184, 262, "示例：相同形状的 FP16 Add", "npu", 32)
    d.text(62, 280, "A  框架适配", 22, BLUE, weight="bold")
    d.text(62, 312, "CPU 构图；复用厂商设备实现", 15, MUTED)
    d.rect(60, 343, 360, 185, FILLS["cpu"], radius=10)
    d.graph(90, 379, 110, 80, BLUE)
    d.text(233, 380, "MNN onResize", 18, BLUE, weight="bold")
    d.text(233, 400, "属性映射\n连边 / 输出槽位", 16, MUTED, va="top")
    d.rule(82, 476, 315)
    d.text(82, 502, "已有 IR 节点 → 整图编译", 17, BLUE)
    d.text(479, 280, "B  自定义设备内核", 22, TEAL, weight="bold")
    d.text(479, 312, "原型 / InferShape → Host Tiling → AscendC", 15, MUTED)
    d.text(479, 355, "Global Memory 中的输入", 16, weight="bold")
    for i, color in enumerate([BLUE, TEAL, PURPLE]):
        d.matrix(480 + i * 212, 382, 3, 9, cell=14, gap=3, fill=color)
        d.text(480 + i * 212, 458, f"分片 {i}", 14, color)
        d.arrow([(549 + i * 212, 435), (549 + i * 212, 492)], color)
        d.rect(482 + i * 212, 496, 154, 50, FILLS[["cpu", "npu", "tool"][i]], radius=6)
        d.text(559 + i * 212, 521, f"核 {i} 的局部缓冲", 14, color, ha="center")
    d.text(479, 579, "块数与核数为示意；Host 根据平台信息决定切分。", 14, MUTED)
    d.rule(60, 615, 1080)
    d.step(79, 658, 2)
    d.text(108, 658, "理解双缓冲排程", 21, weight="bold")
    d.text(108, 688, "逻辑依赖：搬入完成 → 计算完成 → 写回完成", 15, MUTED)
    start, tile_w, stride = 342, 154, 158
    for row, (name, color, offset) in enumerate([("CopyIn", BLUE, 0), ("Compute", TEAL, 1), ("CopyOut", PURPLE, 2)]):
        y = 730 + row * 61
        d.text(95, y + 18, name, 17, color, weight="bold")
        d.line([(269, y + 39), (1138, y + 39)], LIGHT, 1)
        for tile in range(3):
            x = start + (tile + offset) * stride
            d.rect(x, y, tile_w, 36, FILLS[["cpu", "npu", "tool"][row]], radius=4)
            d.text(x + tile_w / 2, y + 18, f"分块 {tile}", 14, color, ha="center")
    for k in range(5):
        d.line([(start + k * stride, 717), (start + k * stride, 908)], LIGHT, .7, dashed=True, z=0)
    d.note(60, 930, 1080, "仅为排程示意；搬运与计算能否重叠，需满足目标平台、缓冲区容量和同步约束。", h=38)
    return d


def offline_weights():
    d = Figure(4, "图与权重分离：文件变小，工作集仍在", "从磁盘文件到设备资源，分清存储组织、初始化与执行器驻留。", "文件解剖与内存视图", 990)
    d.tag(64, 157, 268, "主机编译：OMG + 目标芯片插件", "tool", 30)
    d.tag(814, 157, 322, "设备初始化：Build / Construct / InitWeights", "npu", 30)
    d.label(66, 232, "内嵌模型", "计算结构与常量保存在同一产物", size=21)
    d.file(76, 301, 252, 399, "file", "模型文件")
    d.graph(105, 373, 177, 52)
    d.text(103, 457, "图结构与执行元数据", 15, PURPLE)
    d.line([(94, 485), (309, 485)], GOLD, 1, dashed=True)
    d.matrix(103, 510, 8, 12, cell=13, gap=3, sparse=True)
    d.text(103, 671, "常量 / 权重载荷", 15, GOLD)
    d.arrow([(350, 498), (411, 498)], PURPLE, 2)
    d.text(380, 466, "分离", 17, PURPLE, ha="center", weight="bold")
    d.label(444, 232, "成套外置文件", "图与权重分别存储，整体发布", size=21)
    d.file(453, 301, 245, 156, "tool", "model.omc")
    d.graph(480, 365, 176, 48)
    d.text(480, 434, "计算结构与执行元数据", 14, PURPLE)
    d.file(453, 500, 245, 200, "file", "SubGraph_0.weight")
    d.matrix(478, 553, 8, 12, cell=13, gap=3, sparse=True)
    d.arrow([(717, 379), (776, 379)], PURPLE, 1.8)
    d.arrow([(717, 603), (776, 603)], GOLD, 1.8)
    d.text(745, 663, "配套\n初始化", 14, MUTED, ha="center")
    d.label(813, 232, "设备工作集", "按运行时与模型要求分配", size=21)
    d.rect(811, 300, 324, 401, PAPER, LIGHT, radius=12)
    memory = [("图 / 执行元数据", 50, "tool"), ("设备侧权重表示", 114, "file"),
              ("中间激活与工作空间", 85, "npu"), ("I/O 与持久状态", 72, "cpu")]
    yy = 321
    for label, h, kind in memory:
        d.rect(830, yy, 286, h, FILLS[kind], radius=4, z=2)
        d.rect(830, yy, 5, h, COLORS[kind], radius=0, z=3)
        d.text(851, yy + h / 2, label, 17, COLORS[kind], weight="bold")
        yy += h + 9
    d.text(973, 734, "面积仅作组成示意，不代表占用比例", 13, MUTED, ha="center")
    d.rule(62, 779, 1074)
    for x, n, title, body, kind in [(82, 1, "存储分离", "图与权重独立组织", "file"),
                                    (445, 2, "初始化", "配套权重进入设备执行资源", "tool"),
                                    (837, 3, "驻留复用", "保留执行器，减少重复加载", "npu")]:
        d.step(x, 823, n, COLORS[kind])
        d.label(x + 30, 822, title, body, COLORS[kind], 19)
    d.note(62, 903, 1074, "文件分离、磁盘去重与跨执行器设备权重共享是三种机制；共享能力需由 SDK 明确支持。", h=38)
    return d


def hybrid_deepstack():
    d = Figure(5, "DeepStack：视觉深度如何接入语言深度", "仓库折中案例：单图 ViT 在 NPU、LLM 在 CPU；保留主特征与三路深层特征。", "模型剖面与特征注入", 1100)
    d.rect(60, 164, 1080, 393, FILLS["npu"], radius=12)
    d.tag(78, 183, 366, "NPU / vision_full.omc + 外置 W4A16 权重", "npu", 30)
    d.text(1120, 198, "特征矩阵与分组条带为缩略示意", 12, MUTED, ha="right")
    d.text(98, 250, "图像块嵌入", 16, TEAL, weight="bold")
    d.matrix(94, 279, 4, 5, cell=13, gap=3, sparse=True)
    d.arrow([(178, 307), (213, 307)], TEAL)
    start, bw = 224, 29
    d.text(start, 229, "24 层视觉 Transformer（层号 0–23）", 15, TEAL)
    injections = [(5, 320, BLUE), (11, 550, PURPLE), (17, 780, CORAL)]
    for i in range(24):
        color = [BLUE, PURPLE, CORAL][[5, 11, 17].index(i)] if i in [5, 11, 17] else "#B9DBD0"
        d.rect(start + i * bw, 274, bw - 4, 64, color, radius=3, z=3)
        if i in [0, 5, 11, 17, 23]:
            d.text(start + i * bw + (bw - 4) / 2, 252, str(i), 13,
                   color if i in [5, 11, 17] else MUTED, ha="center")
    d.arrow([(924, 307), (972, 307)], TEAL)
    d.poly([(978, 278), (1063, 290), (1063, 325), (978, 338)], "#C5E3D6", TEAL)
    d.text(1021, 357, "主 merger", 15, TEAL, ha="center")
    d.arrow([(1064, 307), (1095, 307), (1095, 452)], TEAL)
    d.matrix(1052, 459, 3, 5, cell=12, gap=2, fill=TEAL)
    d.text(1019, 534, "152 个图像 token", 15, TEAL, ha="center")
    for layer, target, color in injections:
        source = start + layer * bw + (bw - 4) / 2
        d.arrow([(source, 339), (source, 393), (target, 393), (target, 425)], color, 1.8)
        d.poly([(target - 66, 426), (target + 66, 426), (target + 47, 484), (target - 47, 484)], "white", color)
        d.text(target, 448, "DeepStack", 16, color, ha="center", weight="bold")
        d.text(target, 471, "merger", 13, color, ha="center")
        d.text(target + 18, 525, f"视觉层 {layer} 的特征", 15, color)
        d.arrow([(target, 485), (target, 746)], color, 1.8, dashed=True)
    d.rect(60, 650, 1080, 275, FILLS["cpu"], radius=12)
    d.tag(78, 668, 155, "CPU / 语言模型", "cpu", 30)
    for i, (_, target, color) in enumerate(injections):
        d.rect(target - 83, 751, 166, 97, "white", LIGHT, radius=7, z=7)
        d.text(target, 776, f"语言层 {i}", 21, BLUE, ha="center", weight="bold")
        d.circle(target, 814, 14, FILLS["cpu"], color, z=8)
        d.text(target, 813, "+", 21, color, ha="center")
        if i < 2:
            d.arrow([(target + 83, 800), (target + 147, 800)], BLUE)
    d.arrow([(863, 800), (932, 800)], BLUE)
    d.rect(935, 751, 175, 97, "white", LIGHT, radius=7, z=7)
    d.text(1023, 776, "语言层 3–27", 18, BLUE, ha="center", weight="bold")
    for i in range(13):
        d.rect(952 + i * 11, 809, 7, 20, "#B9CCE1", radius=1, z=8)
    d.text(283, 888, "三路特征分别注入语言层 0 / 1 / 2，仅在图像 token 位置相加。", 16, BLUE)
    d.arrow([(1095, 501), (1168, 501), (1168, 960), (174, 960), (174, 799), (237, 799)], TEAL)
    d.text(690, 987, "主视觉特征 → 替换图像位置的词元嵌入", 16, TEAL, ha="center",
           bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    d.text(64, 1030, "实线：主特征 / 语言计算流    虚线：DeepStack 注入    本例高 × 宽 608 × 256 → 152 个图像 token", 14, MUTED)
    return d


def nnrt_lifecycle():
    d = Figure(6, "NNRT：调用顺序与资源存活区间", "应用串行组织准备、执行与卸载；右侧显示本仓库外置权重路径的资源持有策略。", "时序与所有权", 1080)
    d.text(124, 180, "应用线程", 20, BLUE, ha="center", weight="bold")
    d.text(690, 180, "NNRT / HiAI 接口", 20, PURPLE, ha="center", weight="bold")
    d.line([(124, 210), (124, 938)], "#C4D3E2", 1.5)
    d.line([(690, 210), (690, 938)], "#CCC3DF", 1.5)
    d.text(1004, 180, "持有资源", 20, weight="bold", ha="center")
    calls = [(249, "检查兼容性 / 选择设备", "文件与目标设备匹配"),
             (335, "ConstructWithOfflineModelFile / 配置选项", "创建并配置 compilation"),
             (421, "OH_NNCompilation_Build", "检查构建返回码"),
             (507, "OH_NNExecutor_Construct", "创建 executor"),
             (593, "HMS_HiAIExecutor_InitWeights", "初始化配套外置权重"),
             (679, "查询端口 / 分配并校验 I/O", "按实际描述建立张量与绑定"),
             (788, "OH_NNExecutor_RunSync", "检查执行完成与结果")]
    for i, (y, title, body) in enumerate(calls):
        d.step(61, y, i + 1, TEAL if i == 6 else PURPLE, r=13)
        d.circle(124, y, 4, BLUE)
        d.arrow([(124, y), (690, y)], TEAL if i == 6 else PURPLE, 1.5)
        d.text(151, y - 20, title, 16, weight="bold")
        d.text(151, y + 24, body, 14, MUTED)
        d.rect(681, y, 18, 30, FILLS["npu"] if i == 6 else FILLS["tool"], radius=2, z=7)
    d.rect(91, 739, 642, 101, "none", TEAL, radius=9, lw=1.2, z=0)
    d.tag(504, 724, 195, "同实例重复执行", "npu", 28)
    resources = [(858, 335, "编译实例", "tool"), (961, 507, "执行器", "npu"), (1064, 679, "I/O 张量", "cpu")]
    for x, y, label, kind in resources:
        d.text(x + 25, 223, label, 14, COLORS[kind], ha="center", weight="bold")
        d.line([(x + 25, 245), (x + 25, 883)], LIGHT, 1, dashed=True)
        d.rect(x, y, 50, 883 - y, FILLS[kind], radius=5, z=2)
        d.rect(x + 22, y, 6, 883 - y, COLORS[kind], radius=2, z=3)
        d.circle(x + 25, y, 6, COLORS[kind])
        d.line([(x + 14, 883), (x + 36, 883)], COLORS[kind], 2)
    d.text(1006, 919, "等待任务结束后逆序释放", 15, MUTED, ha="center")
    d.arrow([(1089, 946), (885, 946)], CORAL)
    d.text(985, 971, "张量 → 执行器 → 编译实例", 14, CORAL, ha="center")
    d.note(60, 986, 683, "任一步骤失败均需清理已获取资源；完成权重与 I/O 初始化后再发布就绪状态。", h=40)
    return d


def accuracy_validation():
    d = Figure(7, "精度验证：先理解误差，再缩小范围", "固定相同输入，分别检查量化损失、结构转换与运行时协议。", "数值示意与诊断阶梯", 1020)
    d.text(65, 167, "01  取整与截断会改变数值", 21, weight="bold")
    d.text(650, 167, "02  高相似度仍需核对幅值", 21, weight="bold")
    d.text(65, 201, "示意：scale = 0.5，重建值限制在 [−1, 1]", 14, MUTED)
    d.text(650, 201, "示意：候选值 = 1.6 × 参考值", 14, MUTED)
    x0, y0, w, h = 115, 274, 382, 246
    map1 = lambda x, y: (x0 + (x + 2) / 4 * w, y0 + (2 - y) / 4 * h)
    # round(x / 0.5) 超过整数范围 [-2, 2] 的区域从 |x| > 1.25 开始。
    d.rect(x0, y0, w * 3 / 16, h, FILLS["warn"], radius=0, z=0)
    d.rect(x0 + w * 13 / 16, y0, w * 3 / 16, h, FILLS["warn"], radius=0, z=0)
    for t in [-2, -1, 0, 1, 2]:
        d.line([map1(-2, t), map1(2, t)], LIGHT, .7)
        d.text(*map1(-2.19, t), str(t), 12, MUTED, ha="right")
        d.text(*map1(t, -2.25), str(t), 12, MUTED, ha="center")
    d.arrow([map1(-2, -2), map1(2.1, -2)], MUTED, 1)
    d.arrow([map1(-2, -2), map1(-2, 2.15)], MUTED, 1)
    d.line([map1(-2, -2), map1(2, 2)], BLUE, 1.8)
    steps, cuts, levels = [], [-2, -.75, -.25, .25, .75, 2], [-1, -.5, 0, .5, 1]
    for i, level in enumerate(levels):
        steps += [map1(cuts[i], level), map1(cuts[i + 1], level)]
    d.line(steps, TEAL, 2.5)
    d.text(306, 573, "原始输入", 14, MUTED, ha="center")
    d.text(78, 250, "重建值", 14, MUTED)
    d.text(151, 295, "截断", 13, CORAL, ha="center")
    d.text(461, 480, "截断", 13, CORAL, ha="center")
    d.dot_label(117, 612, "浮点恒等映射", "cpu")
    d.dot_label(329, 612, "量化后重建", "npu")
    xx, yy, ww, hh = 713, 274, 375, 246
    map2 = lambda x, y: (xx + (x + 2) / 4 * ww, yy + (3.2 - y) / 6.4 * hh)
    for t in [-2, 0, 2]:
        d.line([map2(-2, t), map2(2, t)], LIGHT, .7)
        d.text(*map2(-2.12, t), str(t), 12, MUTED, ha="right")
        d.text(*map2(t, -3.55), str(t), 12, MUTED, ha="center")
    d.line([map2(-2, -2), map2(2, 2)], MUTED, 1.4, dashed=True)
    d.line([map2(-2, -3.2), map2(2, 3.2)], CORAL, 1.7)
    for value in [-1.8, -1.3, -.8, -.3, .3, .8, 1.3, 1.8]:
        x, y = map2(value, value * 1.6)
        d.circle(x, y, 4.7, CORAL, "white", 1, z=8)
    d.arrow([map2(-2, -3.2), map2(2.1, -3.2)], MUTED, 1)
    d.arrow([map2(-2, -3.2), map2(-2, 3.4)], MUTED, 1)
    d.text(1080, 573, "参考值", 14, MUTED, ha="right")
    d.text(675, 250, "候选值", 14, MUTED)
    d.text(799, 239, "虚线：幅值一致", 12, MUTED)
    d.text(966, 239, "红线：放大 1.6 倍", 12, CORAL)
    d.text(715, 611, "余弦相似度 = 1", 21, CORAL, weight="bold")
    d.text(929, 611, "相对 L2 = 0.6", 20, CORAL, weight="bold")
    d.rule(60, 660, 1080)
    d.text(65, 698, "03  用分层参考定位偏差从哪一步开始", 21, weight="bold")
    labels = [("浮点参考", "原始基准"), ("仅权重量化", "权重误差"), ("完整量化", "激活误差"),
              ("结构 / ONNX", "导出一致性"), ("CPU 调度", "协议一致性"), ("设备 OMC", "设备侧核验")]
    for i, (title, body) in enumerate(labels):
        x, y = 73 + i * 180, 878 - i * 18
        d.rect(x, y, 160, 26 + i * 18, FILLS["npu"] if i == 5 else PAPER, radius=3)
        d.step(x + 13, y - 46, i + 1, TEAL if i == 5 else BLUE, r=12)
        d.text(x + 34, y - 46, title, 16, TEAL if i == 5 else INK, weight="bold")
        d.text(x + 13, y - 14, body, 14, MUTED)
        if i < 5:
            d.arrow([(x + 160, y + 8), (x + 178, y + 8)], MUTED, 1)
    d.text(65, 953, "上方曲线与散点是数学示例，不是模型测评；主机参考通过后，仍需用相同输入核对真机结果。", 14, MUTED)
    return d


def performance_kv():
    d = Figure(8, "性能边界与 KV 增量写入", "分别观察模型准备、图执行，以及历史状态的同步量。", "计时分区与张量切片", 1120)
    d.text(63, 170, "计时边界", 22, weight="bold")
    d.text(1139, 170, "阶段长度仅示意，不是实测耗时", 13, MUTED, ha="right")
    d.rect(62, 213, 518, 157, FILLS["tool"], radius=9)
    d.rect(600, 213, 540, 157, FILLS["npu"], radius=9)
    d.text(82, 239, "模型加载阶段", 19, PURPLE, weight="bold")
    d.text(622, 239, "图已驻留后的请求", 19, TEAL, weight="bold")
    for x, w, title, body, kind in [(82, 138, "读模型", "磁盘 / 主机", "file"),
                                   (230, 138, "按需编译", "缓存未命中", "tool"),
                                   (378, 180, "设备初始化", "加载 / 权重 / I/O", "tool"),
                                   (622, 132, "准备与拷贝", "输入与状态", "cpu"),
                                   (766, 180, "Process / RunSync", "图执行计时", "npu"),
                                   (958, 162, "结果处理", "输出与主机处理", "cpu")]:
        d.rect(x, 274, w, 67, "white", radius=4)
        d.text(x + w / 2, 295, title, 15, COLORS[kind], ha="center", weight="bold")
        d.text(x + w / 2, 322, body, 13, MUTED, ha="center")
    d.dimension(767, 945, 392, "单独报告 NPU 图执行时间", TEAL)
    d.rule(62, 446, 1076)
    d.text(63, 484, "KV Cache：同样的历史，写入范围不同", 23, weight="bold")
    d.text(63, 520, "示意矩阵压缩显示了 token 行与通道列；层数、头数与 K/V 两路作为外层维度。", 14, MUTED)
    rows, cols, size, gap = 13, 10, 15, 3
    for base, delta in [(175, False), (751, True)]:
        title = "全量重写" if not delta else "缓冲区同步后的增量写入"
        d.text(base - 92, 570, title, 20, CORAL if not delta else TEAL, weight="bold")
        d.text(base - 92, 607, "CPU → 设备输入缓冲区", 14, MUTED)
        xx, yy = base, 649
        colors = (["#C5DED5"] * 9 + [TEAL] + ["#EFF2F5"] * 3 if delta else
                  ["#E8B7A7"] * 9 + [CORAL] + ["#F3DED6"] * 3)
        d.matrix(xx, yy, rows, cols, cell=size, gap=gap, row_colors=colors)
        d.dimension(xx, xx + cols * (size + gap) - gap, 898, "通道维度（压缩示意）")
        if delta:
            d.arrow([(base - 100, yy + 9 * 18 + 7), (xx - 8, yy + 9 * 18 + 7)], TEAL, 2.5)
            d.text(base + 204, yy + 9 * 18 + 8, "新增行", 16, TEAL, weight="bold")
            d.text(base + 204, yy + 58, "历史行\n保留", 15, MUTED)
            d.text(base + 204, yy + 214, "空容量", 14, MUTED)
        else:
            d.line([(xx - 24, yy), (xx - 36, yy), (xx - 36, yy + 231), (xx - 24, yy + 231)], CORAL, 1.7)
            d.arrow([(base - 106, yy + 115), (xx - 41, yy + 115)], CORAL, 2.5)
            d.text(base + 204, yy + 58, "历史 + 新增\n+ 固定容量", 15, CORAL)
        d.text(base - 92, 955, "224 MiB / 步" if not delta else "112 KiB / 词元", 28,
               CORAL if not delta else TEAL, weight="bold")
    d.text(63, 1003, "例：28 层 × 2(K,V) × 8 头 × 容量 2048 × 头维度 128 × FP16；这里只比较主机写入量。", 14, MUTED)
    d.note(63, 1030, 1075, "注意力仍需读取历史 KV；执行器重建或缓冲区失同步后，需要先恢复完整状态。", h=34)
    return d


FIGURES = [("01-software-stack", software_stack), ("02-online-cache-lifecycle", online_cache),
           ("03-operator-development", operator_routes), ("04-offline-external-weights", offline_weights),
           ("05-hybrid-vit-deepstack", hybrid_deepstack), ("06-nnrt-lifecycle", nnrt_lifecycle),
           ("07-accuracy-validation", accuracy_validation), ("08-performance-kv", performance_kv)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent, help="SVG 输出目录")
    parser.add_argument("--preview-dir", type=Path, help="检查用 PNG 目录")
    parser.add_argument("--font", type=Path, help="中文字体文件")
    args = parser.parse_args()
    configure_font(args.font)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    for stem, build in FIGURES:
        build().save(args.output_dir / f"{stem}.svg", args.preview_dir / f"{stem}.png" if args.preview_dir else None)
        print(stem + ".svg")


if __name__ == "__main__":
    main()
