"""Author a factual, editable project handbook using the bundled document runtime."""
from pathlib import Path
import argparse
import json
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/project_handbook_20261011"
OUT.mkdir(parents=True, exist_ok=True)
QA = OUT / "qa"
QA.mkdir(exist_ok=True)
TARGET = OUT / "RK3588多模态垃圾识别系统项目详解.docx"
parser = argparse.ArgumentParser()
parser.add_argument("--refresh-owned-draft", action="store_true")
args = parser.parse_args()
if TARGET.exists() and not args.refresh_owned_draft:
    raise RuntimeError("preserve existing deliverable; use an explicit new version")


def architecture():
    im = Image.new("RGB", (2000, 680), "white")
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 31)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 25)
    def box(x, y, title, desc, width=420):
        draw.rounded_rectangle((x, y, x + width, y + 130), radius=12, fill="#EDF2F7", outline="#8D9BAA", width=2)
        draw.text((x + 18, y + 24), title, font=font, fill="black")
        draw.text((x + 18, y + 80), desc, font=small, fill="#334155")
    def arrow(a, b):
        draw.line([a, b], fill="#536273", width=5)
        x, y = b
        if a[1] == b[1]:
            direction = 1 if b[0] > a[0] else -1
            draw.polygon([(x, y), (x - direction * 15, y - 9), (x - direction * 15, y + 9)], fill="#536273")
        else:
            draw.polygon([(x, y), (x - 9, y - 15), (x + 9, y - 15)], fill="#536273")
    box(40, 50, "V4L2 采集", "帧引用与 DMA 描述符")
    box(535, 50, "有界帧队列", "满载时丢弃最旧待处理帧")
    box(1030, 50, "YOLO 工作线程", "RGA 预处理与检测")
    box(1525, 50, "ROI 与张量准备", "裁剪归一化后归还相机帧")
    for x in [460, 955, 1450]:
        arrow((x, 115), (x + 65, 115))
    arrow((1735, 180), (1735, 305))
    box(1525, 320, "CLIP 有界队列", "传递张量槽租约与帧信息")
    box(1030, 320, "CLIP 工作线程", "独立上下文与分核推理")
    box(535, 320, "文本特征匹配", "相似度与类别拒绝规则")
    box(40, 320, "输出与资源回收", "结果统计与张量槽归还")
    for x in [1525, 1030, 535]:
        arrow((x, 385), (x - 75, 385))
    draw.text((45, 505), "张量池负责缓冲复用    核准入控制协调 NPU    统计模块区分等待与执行", font=font, fill="black")
    draw.text((45, 570), "双线程池表示两个阶段的工作线程集合，并不是整个系统只有两个线程", font=small, fill="#334155")
    path = QA / "architecture.png"
    im.save(path)
    return path


doc = Document()
section = doc.sections[0]
section.page_width, section.page_height = Cm(21), Cm(29.7)
section.top_margin, section.bottom_margin = Cm(1.85), Cm(1.8)
section.left_margin, section.right_margin = Cm(2), Cm(2)
section.header_distance, section.footer_distance = Cm(.8), Cm(.8)
styles = doc.styles
for style in styles:
    for border in list(style.element.xpath(".//w:pBdr")):
        border.getparent().remove(border)
for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3", "Caption", "List Bullet"]:
    style = styles[name]
    style.font.name = "Calibri"
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
normal = styles["Normal"]
normal.font.size = Pt(10.5)
normal.paragraph_format.line_spacing = 1.28
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.widow_control = True
styles["Title"].font.size = Pt(24)
styles["Title"].paragraph_format.space_after = Pt(12)
styles["Subtitle"].font.size = Pt(11)
styles["Subtitle"].font.italic = False
styles["Subtitle"].paragraph_format.space_after = Pt(12)
for name, size in [("Heading 1", 16), ("Heading 2", 12), ("Heading 3", 11)]:
    styles[name].font.size = Pt(size)
    styles[name].font.bold = True
    styles[name].paragraph_format.space_before = Pt(13)
    styles[name].paragraph_format.space_after = Pt(6)
    styles[name].paragraph_format.keep_with_next = True
styles["Caption"].font.size = Pt(9)
styles["Caption"].paragraph_format.space_after = Pt(8)
doc.core_properties.title = "RK3588 多模态垃圾识别系统项目详解"
doc.core_properties.subject = "技术原理 实现细节 实测结果 面试准备"
doc.core_properties.author = "YOLO CLIP 项目"
doc.core_properties.keywords = "RK3588 V4L2 DMA-BUF RGA RKNN C++ CLIP"


def p(text, bold=False):
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = bold
    return paragraph


def h(text, level=1):
    return doc.add_heading(text, level=level)


def bullets(items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(4)
        paragraph.add_run(item)


def table(headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    pr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        node = OxmlElement("w:" + edge)
        for name, value in [("val", "single"), ("sz", "4"), ("color", "D9D9D9")]:
            node.set(qn("w:" + name), value)
        borders.append(node)
    pr.append(borders)
    margin = OxmlElement("w:tblCellMar")
    for edge in ["top", "bottom", "left", "right"]:
        node = OxmlElement("w:" + edge)
        node.set(qn("w:w"), "85")
        node.set(qn("w:type"), "dxa")
        margin.append(node)
    pr.append(margin)
    repeat = OxmlElement("w:tblHeader")
    t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row_index, values in enumerate([headers] + rows):
        row = t.rows[0] if row_index == 0 else t.add_row()
        cant_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(cant_split)
        for i, value in enumerate(values):
            cell = row.cells[i]
            cell.width = Cm(widths[i])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            shade = OxmlElement("w:shd")
            shade.set(qn("w:fill"), "E8EFF6" if row_index == 0 else ("F7F9FB" if row_index % 2 == 0 else "FFFFFF"))
            cell._tc.get_or_add_tcPr().append(shade)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(2)
            paragraph.paragraph_format.space_before = Pt(2)
            paragraph.paragraph_format.line_spacing = 1.15
            paragraph.paragraph_format.keep_with_next = (row_index == 0 or
                (len(rows) <= 6 and row_index < len(rows)))
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 0 and len(str(value)) < 34 else WD_ALIGN_PARAGRAPH.LEFT
            run = paragraph.add_run(str(value))
            run.font.size = Pt(9.2)
            run.bold = row_index == 0
            run.font.color.rgb = RGBColor(0, 0, 0)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return t


footer = section.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
footer.add_run("项目技术说明  ").font.size = Pt(9)
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)

doc.add_paragraph("RK3588 多模态垃圾识别系统项目详解", style="Title")
doc.add_paragraph("技术原理 实现细节 实测结果 面试准备", style="Subtitle")
p("版本日期 2026 年 10 月 11 日")
p("本项目把 YOLO 的目标定位与 CLIP 的图文特征匹配组合成垃圾识别流程，并围绕 RK3588 的摄像头采集、共享内存、硬件图像处理、并发推理和模型量化开展工程优化。阅读本说明后，应能够解释一帧数据从采集到分类的路径、各阶段的资源所有权，以及性能与精度数字成立的条件。")
p("当前已经跑通真实 V4L2 摄像头与 DMA-BUF、RGA、RKNN 的数据链路，并实现有界队列、张量池和多上下文调度。摄像头短测约 3.70 FPS；固定图片调度对照的 P95 降低约 33.45%，吞吐略降。非 YOLO 核验裁剪下，原部署 CLIP 为 88%，同源 FP16 为 87.5%，当前两种混合量化为 82.5%。这些结果来自不同实验，不能拼成同一完整系统的速度与准确率。")
p("阅读顺序：先看模型分工和系统架构，再理解 V4L2、DMA 与资源回收，随后阅读调度、量化和实测结果，最后使用面试问答检查自己的理解。")

h("1 项目目标与技术分工")
p("系统面向边缘设备上的视觉垃圾识别。CPU 负责采集控制、队列、结果解析和部分预处理，RGA 负责图像变换，NPU 执行模型计算。优化副本保留原模型和默认部署，实验不自动替换系统 Runtime 或驱动。")
table(["技术", "在项目中的作用"], [
    ["C++17 与 Linux", "线程池 队列 资源管理 设备控制及时间统计"],
    ["V4L2 与 DMA-BUF", "摄像头缓冲管理及跨设备共享图像内存"],
    ["RGA", "颜色转换 缩放 补边放置及 ROI 裁剪"],
    ["RKNN Toolkit2 与 Runtime", "模型转换 量化 上下文管理及 NPU 推理"],
    ["OpenCV", "离线图片路径 传统视觉裁剪及 CPU 对照路径"],
    ["ONNX 与 Python", "浮点参考 输入冻结 数据校验及数值分析"],
    ["PyTorch 与 CUDA", "官方模型参考及量化训练重建代理"],
], [4.3, 12.7])
p("RGA 不是神经网络计算器，DMA-BUF 也不是加速算法。它们主要改善图像变换的执行方式和内存传递。NPU 调度负责组织计算资源；量化改变数值表示，因此必须单独检查其精度影响。")

h("2 YOLO 与 CLIP 如何完成分类")
h("2 1 YOLO 负责找位置", 2)
p("YOLO 先在输入画面中寻找垃圾目标，返回候选框及置信度。项目中的垃圾检测模型为单类检测，板端输出形状为 1×5×8400，其中包含四个框参数和一个置信通道。程序选择候选框后，将坐标映射回原画面并产生 ROI。YOLO 不负责完成最终四类垃圾语义分类。")
p("当前板端 YOLO 预处理将画面放到边长为 max 高宽的正方形左上角，其余部分以 114 补边，再缩放到 640×640、转换 RGB 并归一化。它不是常见的居中补边方案，反算检测框时必须采用同一坐标约定，否则即使网络输出正常，裁剪也可能错位。")
h("2 2 CLIP 负责图文语义匹配", 2)
p("CLIP 使用 ViT-L/14 图像编码器，将 224×224 的 ROI 转为 768 维图像特征。文本编码器不必每帧运行：候选文本的特征预先计算并保存，板端读取固定 29 行标签和文本特征库。图像与文本特征分别做 L2 归一化，再通过点积计算余弦相似度，取最高分标签并映射到垃圾大类。")
p("图像按 14×14 patch 划分为 16×16，即 256 个 patch，加一个分类 token 后共 257 个 token。视觉塔有 24 个 Transformer block，隐藏宽度 1024，16 个注意力头、每头宽度 64，MLP 在 1024 与 4096 维之间进行上下投影。这些大矩阵和多层注意力解释了 CLIP 为什么比小检测模型更重；是否由计算或内存带宽限制，仍须测量而不能只看架构下结论。")
p("该视觉塔约 3.04 亿逻辑参数，研究 FP16 RKNN 文件约 623 MB，混合量化文件约 321 MB。多上下文共享权重能减少重复模型内存，但输入输出和工作区仍各自独立。224×224 RGB 的 FP16 输入为 301056 字节；29 条文本与 768 维图像特征的点积工作量相对图像编码小，因此优先分析视觉塔而不是每帧重新运行文本编码器。")
p("当前分类规则同时包含 empty background 等候选标签与 0.22 的拒绝阈值。最高相似度不足阈值时输出 unknown；匹配到空背景标签也可能映射为空。因此不能把相似度直接解释为经过校准的概率，更不能把最高分 0.3 表述为 30% 的正确概率。")
h("2 3 两个模型之间的边界", 2)
p("模型流程是画面、目标定位、裁剪、图像编码、文本匹配。检测失败与分类失败应分开记录。完整系统可以采用漏检后跳过分类或整图回退，但必须说明选用的策略；离线已知 empty 绕过检测的分支不等于在线系统已经实现自动判空。")

h("3 整体架构与数据流")
doc.add_picture(str(architecture()), width=Cm(17))
doc.add_paragraph("图 1 两阶段线程池与共享缓冲的数据流", style="Caption")
p("采集线程只负责取得帧、记录时间和投递任务。YOLO 工作线程消费帧队列，完成图像变换及检测，并把 ROI 编码为 CLIP 输入。CLIP 工作线程消费张量任务队列，执行独立推理上下文并进行文本匹配。统计模块在结果完成时记录延迟、吞吐与丢帧信息。")
p("双线程池是两组承担不同阶段的工作线程，不是固定只有两个线程。具体工作线程数由配置决定，例如历史摄像头测试为 2 个 YOLO 与 4 个 CLIP 线程，而低尾延迟对照采用 1 个 YOLO 与 3 个 CLIP 线程。不同配置不能混写成同一结果。")
p("每个阶段有自己的排队空间和资源容量。帧队列存放相机帧引用，CLIP 队列存放张量槽租约、帧编号和时间信息。把大图像变成受控的缓冲引用或槽编号，既减少额外复制，也让生命周期与积压数量可见。")

h("4 V4L2 摄像头采集原理")
h("4 1 缓冲区生命周期", 2)
p("初始化时打开采集设备，协商分辨率、像素格式和帧间隔，申请驱动缓冲区并逐个查询。DMA 路径通过 EXPBUF 获得可共享的文件描述符；CPU 对照路径可以映射缓冲区读取像素。把缓冲区排入驱动队列后启动采集，运行期先等待可读事件，再通过 DQBUF 取得完成帧。")
p("DQBUF 表示这个缓冲区暂时交给应用使用，QBUF 则表示应用已使用完毕，可以再次交给摄像头。只要 RGA 或预处理阶段还在读取这帧，就不能提前 QBUF，否则下一帧可能覆盖尚未使用完的数据，形成偶发花屏或错误分类。")
p("项目 Frame 对象持有采集对象的共享引用，帧引用销毁时归还相机缓冲。队列满时丢弃的帧也走同一归还路径。相机缓冲会保留到 ROI 预处理结束，此后 CLIP 只持有自己的输入张量，避免相机帧被长耗时推理长期占用。")
h("4 2 格式与异常边界", 2)
p("当前路径支持渐进式、单平面的 YUYV 或连续 NV12，并检查行跨度、有效字节数与实际接受的图像尺寸。摄像头报告的 30 FPS 不代表必然每秒送达 30 帧，历史测试实际送达约 10 FPS，应以 DQBUF 统计和时间戳为准。")
p("程序对超时、损坏帧、设备不可用和归还失败进行区分，异常时退出并回收资源。当前打开稳定设备路径能减少 /dev/video 编号变化的影响，但并没有实现已打开 FD 失效后的自动重连。该功能属于后续工作。")
p("原始 YUYV/NV12 不需要 MPP。只有 MJPEG、H.264 等压缩输入才需要增加解码步骤；未来可研究 MPP 解码帧 FD 接入 RGA，但不能将这一规划算作当前已实现的技术。")

h("5 DMA 共享内存与 RGA 处理")
h("5 1 FD 是共享内存的句柄", 2)
p("DMA-BUF FD 是内核中共享缓冲对象的引用，不是物理地址，也不是普通的 CPU 指针。用户态虚拟地址用于 CPU 访问，设备侧地址由驱动及 IOMMU 等机制建立映射。共享同一个缓冲区也不等于保证物理页一定连续，应让设备驱动处理映射和约束。")
p("相机驱动导出 FD，RGA 将其导入为图像句柄；程序从 DMA heap 申请工作缓冲，并让 RGA、CPU 和 RKNN 在受控时序下访问。导入句柄和工作图像尽可能复用，避免每帧重复申请大块内存或重新导入。")
h("5 2 实际的图像路径", 2)
p("摄像头 YUYV/NV12 缓冲先由 RGA 转换为工作线程私有的 BGR DMA 图像。YOLO 分支将画面放入可复用的方形补边图，再缩放转为 640×640 RGB；检测后，CLIP 分支从 BGR 图像裁剪 ROI 并生成 224×224 RGB。")
p("这条摄像头 DMA 路径不把采集帧转换为 cv::Mat，也不依赖 OpenCV 采集或 FFmpeg。离线图片与视频路径仍可能经过 OpenCV 解码、内存落地和复制，不能用文件输入测试证明摄像头全链路共享缓冲的收益。")
p("当前板端 RGA 的颜色填充操作未通过测试，因此常量补边由 CPU 在几何变化时初始化一次并复用。每帧的放置、缩放和裁剪仍由 RGA 完成。RGA 本身仍会读写内存；共享 FD 避免的是 CPU 额外复制，不是消除所有硬件内存流量。")
h("5 3 同步决定数据是否正确", 2)
p("CPU 缓存和设备看到的内存内容不一定自动一致。CPU 读取设备结果之前、写入设备输入之后，需要按照 DMA-BUF 的访问规则执行同步。项目通过成对的访问开始与结束操作表达 READ 或 WRITE 方向，并使用作用域对象保证异常路径也结束访问。")
p("缓存同步不等于等待设备任务完成。当前 RGA 操作采用同步调用，完成后才能进行依赖它的 CPU 读取或 NPU 提交。若以后改为异步设备操作，还需要处理设备完成事件或 fence，不能只增加一次缓存刷新。")

h("6 归一化与 RKNN 输入")
h("6 1 两个模型的数值契约", 2)
p("YOLO 使用像素值除以 255。CLIP 在像素值除以 255 后，按 RGB 通道减去均值再除以标准差。CLIP 均值为 0.48145466、0.4578275、0.40821073，标准差为 0.26862954、0.26130258、0.27577711。当前项目直接缩放到 224×224，这与某些官方短边缩放加中心裁剪流程不同，比较成绩时必须保持一致。")
p("转换校准输入已经标准化，因此 RKNN 实验配置使用 mean 0、std 1，避免重复归一化。ONNX 浮点参考输入是 NCHW，板端已验证的原生输入是 NHWC FP16；维度元素相同不代表布局相同，通道、行跨度、数据类型和字节数都需要检查。")
h("6 2 查表减少逐像素计算", 2)
p("RGB 每个通道的输入只有 256 个可能值，可以预先保存三张归一化后 FP16 结果表。运行期根据像素索引查表，直接写入最终张量。它与先生成大块 FP32 数组、再转换和复制的方式不同，主要减少浮点计算与中间缓冲。")
p("FP16 转换已经修正为 IEEE 舍入，显式处理尾数进位和特殊数值；旧模式仅用于复现历史结果。查表模式应与正确标量模式在同一输入上比较，不能因舍入实现变化而把结果差异误判为调度或量化收益。历史一个 CPU 核配置的微基准中，224 输入的 CLIP 标量归一化约 4.07 ms，查表约 0.37 ms；这不是摄像头完整预处理耗时。")
h("6 3 零拷贝输入的准确含义", 2)
p("CPU 从 RGB DMA 图像读取像素，并把归一化结果直接写入 DMA 支持的 FP16 张量。RKNN 根据这个张量 FD 绑定原生输入，不再额外复制整个输入张量。RGB 图像与 FP16 张量的表示不同，通常是不同的缓冲区。")
p("因此，当前可以称为 DMA 共享缓冲与零额外输入拷贝，但不能称为没有 CPU 像素处理。归一化是计算变换，本身不是一次纯 memcpy；RGA 输出新图像也同样包含必要写入。继续优化可比较 NEON、亲和性及前处理折叠，硬件侧归一化还需要验证后端支持和数值一致性。")

h("7 线程池 队列与资源所有权")
h("7 1 生产者消费者分阶段工作", 2)
p("采集、检测和分类的速度不同。如果每一帧都串行等待 CLIP，CPU 和图像处理器容易在 NPU 运行时空闲。分阶段队列允许上一帧在 CLIP 推理时，下一帧继续完成采集和 YOLO 预处理，但不保证两个模型一定同时占用同一个 NPU 核。")
p("每个工作线程持有自己的推理上下文和可复用预处理状态。上下文的输入输出存储不能被多个线程无保护地修改，尤其不能为了减少内存而共享正在写入的输入张量。共享模型权重与共享可变工作区是两回事。")
h("7 2 有界队列与过载丢帧", 2)
p("队列容量固定，满载时丢弃最旧的待处理任务，使系统避免无上限的积压与内存增长。丢弃的是排队任务，已经进入硬件执行的任务不能由这个队列机制抢占。队列容量越大，未必吞吐越高，可能只是增加结果陈旧程度。")
p("Push 与 Pop 的队列修改在互斥锁保护下完成，条件变量通知消费者；释放被丢弃任务的回调在队列锁外执行。这样可以避免资源归还再拿其他锁时形成锁顺序问题，也减少队列临界区时间。Close 唤醒等待者，队列中已有任务可以按照既定策略处理完，空队列消费者返回结束。")
h("7 3 张量池与 RAII 租约", 2)
p("CLIP 输入张量在启动时按固定数量预分配。YOLO 阶段从池中领取一个槽，写入 ROI 输入后把租约移动到 CLIP 队列；CLIP 完成后租约析构，槽回到空闲列表。领取不到槽时明确记录丢弃，不能覆盖仍在使用的槽。")
p("租约是不可复制、可移动对象，保证一个槽只有一个归还责任。队列拒绝任务、丢弃旧任务、线程异常以及正常完成都会触发相同的归还路径。租约持有池状态的共享引用，使状态不会在仍有未归还任务时提前析构。")
p("关闭流程应先停止或阻止新任务进入，再关闭队列、唤醒并等待工作线程，最后销毁 RKNN 内存封装、上下文和底层 DMA 资源。实际实现通过错误收集与 join 控制退出，必须维护封装先销毁、底层内存后释放的依赖关系。")

h("8 NPU 多核调度与性能边界")
h("8 1 单上下文多核与多上下文并发", 2)
p("单个上下文的一次推理可以请求全部 NPU 核，由 Runtime 按算子支持与计算图安排执行；这不等于每个算子都能同时平均分到三个核。多上下文分核则让不同帧由不同上下文同时处理，主要改善聚合吞吐。两种并行不能混为一谈。")
p("如果单核一次推理假设为 662 ms，理想三核聚合吞吐上限约为 4.53 FPS，对应结果平均输出间隔约 221 ms。这个间隔不是每个请求的模型延迟。不同上下文还可能共享内存带宽、驱动与其他资源，YOLO 也消耗核，因此实际吞吐不保证线性三倍。")
h("8 2 核准入控制", 2)
p("应用以三位占用掩码记录本进程已准入请求使用的核。固定请求等待目标核空闲，动态 YOLO 请求选择任一空闲核，并采用轮转方式避免一直选择同一个核。条件变量只在资源条件满足时继续执行，作用域 Permit 在推理结束或异常时释放占用。")
p("等待动态请求获得一定准入优先权，减少它被长 CLIP 请求持续挤压的机会。这个控制器不改变内核优先级、不管理其他进程、不抢占已经启动的 NPU 计算，也不提供硬实时期限保证。准入等待必须计入相关耗时或单独报告，不能把等待从计时中移走后宣称推理变快。")
h("8 3 已完成的固定图片对照", 2)
table(["配置", "完成吞吐 FPS", "各轮 P95 的均值 ms"], [
    ["2 YOLO 与 4 CLIP 轮转", "3.628", "2604.90"],
    ["1 YOLO 自适应与 3 CLIP 分核准入", "3.582", "1733.58"],
    ["2 YOLO 自适应与 3 CLIP 分核准入", "3.475", "3184.79"],
], [8.9, 3.7, 4.4])
p("该对照使用修正后的 IEEE FP16 输入，三轮轮换运行，每轮输入 12 秒，统计包含队列排空。第一种优化配置的 P95 降低约 33.45%，吞吐下降约 1.27%。因此收益是尾延迟，不是 FPS 增长；增加第二个 YOLO 工作线程反而加重了竞争。表中的 P95 是各轮 P95 的平均，不是全部请求合并后的 P95。")

h("9 摄像头实测与指标解释")
table(["指标", "已完成的 60 秒摄像头短测"], [
    ["采集与完成", "598 帧采集 231 帧完成 367 帧入口丢弃"],
    ["完整流水线吞吐", "3.696 FPS 包含排空时间"],
    ["端到端延迟", "P50 1941.553 ms P95 2606.735 ms"],
    ["CLIP 预处理均值", "4.156 ms"],
    ["CLIP run 均值", "1030.400 ms 当前并发配置下的测量"],
    ["资源短时采样", "RSS 约 1372.77 MiB FD 数量 58"],
], [5.4, 11.6])
p("摄像头为 640×480 YUYV，12 个驱动缓冲，2 个 YOLO 与 4 个 CLIP 线程；本次没有检测到有效 ROI，开启整图回退后全部完成帧仍运行了 CLIP。它证明采集、DMA、RGA 和推理链路可用，但不是带标注垃圾场景的准确率测试，也不是同条件 CPU-copy 对照的速度提升证明。")
p("598 等于 231 加 367，说明该轮入口帧账目闭合。短时 RSS 与 FD 无增长不等于长期没有泄漏；其他运行及采集控制测试出现 USB 断连，重复与长期稳定性没有通过。CPU 采样约 15.40% 以单 CPU 核满载为 100%，不能当作整块八核板的利用率百分比。")
p("FPS 表示单位时间完成量，平均 run 表示某个计时边界内的耗时，端到端延迟还包含排队和各阶段等待。摄像头计时从成功取出帧开始，不包含曝光及 USB 传输。单次独占 NPU 的约 400 ms 与并发流水线中的约 900 至 1000 ms，并不天然矛盾，应先核对核配置、竞争、计时范围和输入版本。")
p("独立 CLIP 饱和短基准中，FP16 三上下文约 3.8 FPS，KL 混合量化约 6.3 FPS。这不包含摄像头、YOLO、RGA 或队列端到端延迟，不能写成完整系统已达到 6.3 FPS。尚无直接 DDR 带宽分析，不能确定地把非线性扩展归因为 DDR 瓶颈。")

h("10 模型转换与输入一致性")
p("模型部署分为训练或原权重、ONNX 浮点导出、Toolkit 转换、RKNN 文件和 Runtime 执行几个阶段。Toolkit 是转换工具，Runtime 是设备运行库，驱动连接硬件，三者不是同一个版本号。当前默认 Runtime 为 2.3.0、驱动 0.9.6，实验使用隔离的 Toolkit 与 Runtime 2.3.2，不覆盖默认系统库。")
p("首先核验文件来源与 SHA，再固定解码、裁剪、RGB 顺序、缩放、归一化、FP16 舍入和布局。为隔离量化影响，浮点参考直接读取冻结的板端 FP16 输入值并转为 FP32，不重新生成另一套输入。否则输出差异可能来自预处理，而不是模型数值精度。")
p("已恢复原项目的 CLIP ONNX、YOLO 权重与训练记录，板端历史 RKNN 与原项目文件 SHA 一致。原 CLIP ONNX 输出单位特征，下载的官方图像塔输出原始特征；在之前 200 张精确输入上显式统一归一化后，两者平均余弦几乎为 1。跨输出契约的 raw RMSE 大，不能据此认定权重不同或量化损失巨大。")
p("历史部署 RKNN 对原项目浮点输出仍有方向偏差，约 0.973 的平均余弦。相同 200 张输入在 Runtime 2.3.0 与 2.3.2 下，历史模型输出逐值相同；更换 Runtime 没有解决这部分偏差，但该对照也不能证明所有模型、所有输入都与版本无关。")

h("11 量化方法与实际实现程度")
h("11 1 PTQ 与混合精度", 2)
p("PTQ 使用校准样本估计激活分布和量化范围，不需要重新训练全部网络。INT8 仿射表示通过尺度和零点把浮点数映射到有限整数，带来舍入和裁剪误差。范围太窄容易丢失极端值，范围太宽则步长增大、普通值分辨率下降。权重逐通道量化能缓解不同通道幅度不一致。")
p("项目比较过普通范围、KL 和 MMSE 校准。KL 倾向保留分布信息，MMSE 更直接关注重建误差；算法名称本身不能保证分类提升。同一 30 张校准的控制已完成，但旧公共六类任务分数不能替代真实四类加 empty 的结果。")
p("混合精度让部分敏感节点保持 FP16、其余采用 INT8。保护 LayerNorm、残差或某组矩阵并非越多越好：它可能降低加速收益，还受到编译融合和板端内核正确性的限制。当前候选保持 24 个融合注意力为 INT8，以避开已复现的异常路线。")
h("11 2 参数折叠与层重建", 2)
p("SmoothQuant 式处理通过通道尺度迁移缓解激活离群值，RepQ 式处理将归一化仿射参数与相邻线性变换重新组织。本项目使用不增减网络层的等价参数折叠，并先验证浮点方向基本一致，再重新校准和编译。它们是面向 RKNN 约束的改编，不是完整论文算法复现，尤其未实现论文中的全部非均匀 Softmax 量化。")
p("AdaRound 式重建在固定权重量化网格上学习向上或向下舍入的选择。项目对 24 个已有 MLP 下投影矩阵执行校准层重建，并分别保存仅偏置补偿、仅自适应舍入及两者结合的模型。训练使用实际 SDK 逐输出通道权重尺度，但激活 QDQ 与 NPU 融合算子仍可能不逐位一致。")
p("层重建的平均代理 MSE，AdaRound 下降约 5.76%，结合偏置补偿下降约 9.90%。真实 YOLO 候选 ROI 条件集上三版准确率分别为 81.5%、82.5%、81.0%，不能用代理损失下降替代完整模型收益，也不能称为已完成完整 BRECQ 块联合重建。")
h("11 3 QAT 与待验证方法", 2)
p("QAT 在训练时模拟量化，用 STE 等方法传递梯度。4060 上已完成 LayerNorm 和原投影矩阵等小范围参数训练，部分代理特征拟合改善，但未形成可部署的精度收益。代理算子与实际 NPU 融合算子不匹配时，继续训练也可能拟合错误目标。")
p("公共校准扩充到 240 张的转换尚没有完成证据。校准直方图已生成 percentile 和 MSE 范围候选，真实场景独立校准抽样工具也已准备；这些仍需要证明参数实际被后端采用并完成板端验证。不能把脚本准备或范围拟合成功写成优化已经生效。")

h("12 关键问题的定位过程")
h("12 1 融合注意力的正确性问题", 2)
p("某条混合量化路径将融合 SDPA 强制保护为 FP16 后，模拟器输出正常，但板端输出等效于对 V 做均匀平均，不再保持由 Q、K 决定的 query 差异。为区分普通量化误差与执行错误，项目逐段导出前缀、比较 QKV 和注意力边界，再使用仅有 Q/K/V 输入的约 0.45 MB 无权重模型复现。")
p("普通非量化 FP16 注意力对正确参考的余弦约 0.99992；异常混合 FP16 路线约 0.612；仅取消 SDPA 的 FP16 保护后，INT8 控制恢复 query 差异，局部余弦约 0.956。局部正确性恢复仍不是完整 CLIP 精度恢复。该定位证明了当前版本组合中的特定路线异常，但没有 SDK 内部源码证据，不能声称已找到或修复某一行驱动代码。")
h("12 2 不能直接归因为大量 CPU 回退", 2)
p("算子表应同时检查算子类型、精度、执行目标和时间。出现 CPU Input/Output 边界，不等于 LayerNorm、Softmax、GELU 大量回退到 CPU。在上述最小复现中，关键融合算子被标为 NPU。项目不把日志级别设置本身当成证据，也不把 CUDA FlashAttention 当作能直接加速 RK3588 NPU 的方法。")
h("12 3 SDK 私有 IR 与标准 ONNX 检查不同", 2)
p("重建候选首次转换在标准 ONNX checker 遇到 exDataConvert 私有算子时失败。修复没有简单删除检查，而是约束原图与修改图的连接、节点属性、输入输出契约及非目标参数保持不变，目标参数形状和类型不变，数值有限，并对保存的 external-data 文件重载校验。随后由实际 SDK step2、模拟器和板端承担各自验证职责。")
h("12 4 YOLO 漏检与模型来源", 2)
p("当前 200 张真实集上，160 张非空只有 5 张达到默认 YOLO 门槛。原 PyTorch、ONNX 和 NPU 在固定探针上同样低置信度，找到的已完成垃圾专用训练记录只有 3 个 epoch。因此不能单独归罪于 NPU 或 Runtime，也不能仅凭训练轮数认定全部原因。提高 raw 输入分数但框落在角落或补边，不构成输入修复证据。")

h("13 真实数据测试与数值保真")
h("13 1 三种成绩必须分开", 2)
p("严格检测门控成绩要求非空图先通过 YOLO 默认门槛；漏检计为失败。在已知 empty 绕过检测的离线协议下，历史部署模型为 45/200，即 22.5%，主要被检测失败限制。它仍不是完整在线自动判空系统的成绩。")
p("YOLO 候选 ROI 条件分类采用最高分有效框，不要求达到默认检测门槛。该诊断集上，FP16 为 86.5%、KL 为 79%、RepQ 全折叠为 83.5%。这些数值用于隔离分类能力，不能把条件成绩当作检测成功率或线上端到端正确率。")
p("最新 CLIP-only 测试完全不运行 YOLO。200 张中 132 张非空用固定绿幕 HSV 连通域裁剪，28 张透明或小目标等明显错误在查看分类分数之前，经视觉检查后改用同一个保守中央 ROI；40 张 empty 保留整图，仍由 CLIP 预测。视觉检查不是人工真值框标注，也未证明近重复或物体实例独立。")
h("13 2 最新非 YOLO 裁剪结果", 2)
eval_path = ROOT / "results/real_eval_200_20261011/nonyolo_reviewed_crop_v2/clip_only_evaluation.json"
evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
labels = {"onnx_fp32": "同源 FP32", "deployed_historical": "历史部署模型", "official_fp16": "同源 FP16", "official_kl120": "KL 混合量化", "repq_all": "RepQ 混合量化"}
rows = [[labels[v["variant"]], f'{v["task"]["correct"]}/200', f'{v["task"]["accuracy"] * 100:.1f}%',
    f'{v["nonempty_accuracy"] * 100:.2f}%', f'{v["empty_correct"]}/40'] for v in evaluation["variants"]]
table(["版本", "总正确数", "全部准确率", "非空准确率", "空图正确数"], rows, [4.6, 2.8, 3.2, 3.4, 3])
p("空图相对容易，因此同时列出 160 张非空成绩，避免只用总体分数掩盖物体分类能力。KL 和 RepQ 在本轮均比同源 FP16 低 5 个百分点，RepQ 没有复现上一组 YOLO 候选 ROI 的分类优势。测试输入、裁剪、29 标签和 0.22 阈值在评分后均未调整，默认部署未替换。")
h("13 3 为什么还需要数值指标", 2)
fidelity = [[labels[v["variant"]], f'{v["unit_contract_fp32_fidelity"]["cosine_mean"]:.6f}',
    str(v["unit_contract_fp32_fidelity"]["sqnr_db_global"]) if v["variant"] == "onnx_fp32"
    else f'{v["unit_contract_fp32_fidelity"]["sqnr_db_global"]:.2f}'] for v in evaluation["variants"] if v["variant"] != "onnx_fp32"]
table(["对同源 FP32", "单位特征平均余弦", "单位输出 SQNR dB"], fidelity, [6.8, 5.1, 5.1])
p("余弦衡量方向，RMSE 和 NRMSE 衡量数值误差，SQNR 比较信号能量与误差能量，范数比检查幅度，固定文本决策一致率检查实际分类规则。单位特征比较之前显式归一化，raw 指标则只在相同输出契约下使用。RepQ 特征方向优于 KL，但本轮分类正确数相同，说明这些指标不能互相替代。")
p("200 张来自同一绿幕数据集，按五类各 40 张抽样并做文件与像素精确去重，没有按物体实例或新场景严格划分。它是项目样本对照，不代表任意背景下的泛化准确率，也不是重新生成的一套独立盲测。")

h("14 代码结构与复现入口")
p("本地工作副本位于 F:/YOLO-CLIP/system_demo。下表帮助把技术讲解对应到真实代码；阅读时先看 pipeline.cpp 的任务流，再看资源与模型模块，不必按文件名机械背诵。")
table(["相对路径", "职责"], [
    ["src/pipeline.cpp", "参数校验 采集 线程编排 阶段衔接及关闭"],
    ["include/v4l2_capture.hpp", "相机队列 格式校验 帧引用与归还"],
    ["include/dma_buffer.hpp", "DMA 分配 映射 FD 与 CPU 访问同步"],
    ["include/rga_preprocess.hpp", "工作图像复用 转色 补边与 ROI 缩放"],
    ["include/model_runner.hpp", "上下文 输入契约 绑定 推理与输出管理"],
    ["include/normalization.hpp", "正确 FP16 舍入 标量与查表归一化"],
    ["include/pipeline_queue.hpp", "有界丢旧队列与关闭唤醒"],
    ["include/buffer_pool.hpp", "张量槽及不可复制可移动租约"],
    ["include/npu_arbiter.hpp", "核占用掩码 条件等待与准入释放"],
    ["include/text_bank.hpp", "固定文本特征 余弦匹配及拒绝规则"],
    ["include/metrics.hpp", "阶段等待 耗时 帧账目与延迟分布"],
], [8.4, 8.6])
p("硬件工程由 CMake 编译，便携测试覆盖归一化、队列与池、核准入等模块；硬件测试负责实际 DMA、RGA、RKNN 和摄像头行为。便携单元测试通过不能替代设备上的执行正确性。不同模型、输入契约和 Runtime 配置的结果分别存放，不能覆盖旧记录后只保留更好的一组。")
p("CLIP-only 复现入口依次是 prepare_nonyolo_crops.py、freeze_nonyolo_clip_inputs.cpp、run_nonyolo_clip_test.py 和 evaluate_nonyolo_clip.py。运行脚本拒绝覆盖已有输出；若状态中断，应先检查实际进程、模型和输出，再建立明确的新尝试目录，不因暂时连接失败自动重跑。")

h("15 当前边界与下一步优先级")
table(["状态", "具体内容"], [
    ["已实现", "真实 V4L2 DMA RGA RKNN 链路 线程池 张量池 核准入及分类"],
    ["已验证", "摄像头短测 调度尾延迟对照 固定输入数值及真实裁剪分类"],
    ["未达到", "完全无 CPU 前处理 量化无损 同时高 FPS 与高端到端精度"],
    ["尚待完成", "自动重连 MPP 压缩解码 同摄像头 CPU 与 DMA 消融 后端匹配重建"],
], [3.1, 13.9])
p("系统方向最先处理结果陈旧问题：帧队列和 CLIP 队列都增加基于帧龄的策略，在不抢占已执行硬件任务的前提下跳过过时待处理任务，并同时报告 P95、完成数与丢弃率。只加线程或扩大队列，不能保证实时性。")
p("第二步是同摄像头、同分辨率、同模型和同工作线程的 CPU-copy 与 DMA 对照，记录 CPU 时间、明确可计数的复制字节、同步等待及吞吐。当前数据证明链路可用，但尚不能量化 DMA 相对该基线的加速比例。")
p("设备方向完善断连恢复与分辨率变化后的重新初始化，检查帧引用、RGA 句柄和 RKNN 封装的销毁顺序，优先做短时故障注入而非长期烧机。图像方向比较 RGA 操作合并和中间缓冲减少，硬件内存流量的下降需要实际支持与测量。")
p("模型方向先让 YOLO 在真实图上可靠定位，再使用与测试集互斥的真实校准样本检查量化分布。MSE/Fisher 范围、块联合重建、后端匹配 QAT 及敏感性搜索仍未完成，当前不能宣布方法已耗尽。任何收益都应使用同输入、同任务和明确的速度范围再次验收。")

h("16 面试中的核心问题")
qa = [
    ("项目最核心的优化是什么", "把模型调用改造成可控的边缘流水线：V4L2 与 DMA-BUF 管理输入，RGA 承担图像变换，双线程池和张量池管理阶段衔接，多上下文和核准入管理 NPU 竞争。结果上强调实际链路、尾延迟和速度精度边界，而不是堆接口名称。"),
    ("零拷贝是否意味着没有 CPU 或内存写入", "不是。我们避免额外输入张量复制，但 CPU 仍做归一化并写入最终 DMA 张量，RGA 也产生必要的图像读写。准确说法是共享缓冲与零额外输入拷贝。去除 CPU 前处理还需要新的设备侧路径及正确性对照。"),
    ("为什么相机缓冲不能提前归还", "相机驱动下一次采集可能覆盖该缓冲。应用必须等所有读取者完成再 QBUF。项目把相机帧保留到 ROI 张量准备结束，此后 CLIP 使用独立张量槽，既保证正确性，也避免慢推理耗尽采集缓冲。"),
    ("共享 FD 为什么仍需要同步", "句柄共享解决的是访问同一对象，缓存一致性和设备任务完成是另外两件事。CPU 与设备访问顺序需要显式同步；若设备任务异步，还要等待完成事件，不能把一次缓存同步当成所有设备计算已经结束。"),
    ("一个 RKNN 上下文能否使用三个核", "可以请求多核，但 Runtime 具体如何执行由计算图和内核支持决定。多个上下文并发处理不同帧则是另一种并行。三核吞吐的倒数只是平均输出间隔，不能直接用作单帧推理延迟。"),
    ("为何线程更多反而更慢", "线程增加可能加重同核竞争、同步与排队，并共享驱动和内存资源。我们的 1 YOLO 加 3 CLIP 配置改善尾延迟，2 YOLO 配置反而更差；原因必须结合等待时间与资源证据，不直接断言是 DDR 瓶颈。"),
    ("队列丢任务如何避免泄漏", "相机帧和张量槽都由作用域对象持有归还责任，正常完成、异常、队列关闭或丢旧任务使用同一回收路径。任务租约不可复制，只能移动；丢弃回调在队列锁外执行，避免嵌套锁带来的风险。"),
    ("为什么模拟器正确但板端可能错误", "模拟器不覆盖全部设备融合与执行细节。我们在混合 FP16 SDPA 路线上发现 query 差异丢失，通过前缀差分和无权重小模型复现。软件数值验证、编译验证和硬件验证是不同层级，缺一不可。"),
    ("量化精度怎样衡量", "先保证输入、输出契约和文本规则相同，再同时看任务正确率、逐样本特征余弦、RMSE、NRMSE、SQNR 及决策一致率。校准集不能读取测试激活来拟合参数，raw 与单位输出也不能不加区分直接比较。"),
    ("88 百分比与 6 点 3 FPS 能否放在一起宣传", "不能。88% 来自已核验非 YOLO 裁剪的历史部署 CLIP，6.3 FPS 来自混合量化的独立 CLIP 聚合基准，最新该量化候选准确率是 82.5%。这不是完整系统同时达到的速度和精度。"),
    ("项目还缺少哪些工程能力", "主要是当前真实检测的可用性、过期帧处理、摄像头自动恢复，以及同条件 DMA 对照。量化还缺后端一致性和独立校准数据上的完整验收。回答时把已完成证据和建议明确分开，不把计划包装成成果。"),
    ("如何讲清自己的贡献", "按问题、设计、实现、验证、代价来讲一项工作，例如核竞争如何导致长尾，位图准入如何限制冲突，为什么使用 RAII 释放，以及同轮 P95 和吞吐发生了什么变化。只把自己实际参与、理解并能复现的部分写为个人贡献，不把工具辅助写成独立完成所有工作。"),
]
for question, answer in qa:
    h(question, 2)
    p(answer)

h("17 项目讲解示例与证据索引")
p("简短讲解可以这样组织：我们的项目运行在 RK3588 香橙派上，YOLO 负责定位、CLIP 通过预先计算的文本特征完成垃圾分类。我的讲解重点是 V4L2 采集与 DMA 共享缓冲、RGA 硬件预处理、双线程池和 NPU 核准入。真实相机链路已经跑通，固定图片对照的 P95 降低约三分之一，但吞吐没有提升。量化独立吞吐更高，不过同一核验裁剪集上的准确率从 FP16 的 87.5% 降为 82.5%，因此没有替换默认模型。")
p("如果继续追问，先画一帧的生命周期，再说明队列拥塞和张量槽归还；最后用同轮实验解释性能取舍。项目说明可以讲 API 和实现细节，求职简历则以 V4L2、DMA-BUF、RGA、线程池和调度机制等技术关键词为主。")
table(["证据位置", "证明的事项"], [
    ["CAMERA_DMA.md 与 USB_CAMERA_TEST_20261009.md", "真实摄像头共享缓冲路径与短测范围"],
    ["docs/OPTIMIZATION_20261010.md", "固定图片调度对照及历史独立吞吐"],
    ["docs/MODEL_PROVENANCE.md", "模型来源 输出契约与 Runtime 控制"],
    ["docs/SDPA_ROOT_DIAGNOSIS_20261010.md", "融合注意力异常的最小复现"],
    ["docs/REAL_DATA_EVALUATION_20261011.md", "检测门控与候选 ROI 的成绩区别"],
    ["docs/CLIP_ONLY_CROP_EVAL_20261011.md", "最新非 YOLO 裁剪协议与完成结果"],
    ["results/real_eval_200_20261011/nonyolo_reviewed_crop_v2", "裁剪 原始输入 SHA 逐样本预测与完整报告"],
    ["docs/OPTIMIZATION_METHOD_COVERAGE_20261011.md", "已试方法 尚待完成项及边界"],
], [10.4, 6.6])
doc.save(TARGET)
print(json.dumps({"document": str(TARGET), "paragraphs": len(doc.paragraphs), "tables": len(doc.tables)}, ensure_ascii=False))
