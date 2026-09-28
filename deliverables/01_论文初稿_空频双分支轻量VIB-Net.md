# 空频双分支轻量 VIB-Net：面向多生成模型的 AI 图像篡改检测

> **稿件状态：初稿（方法章与工程章已成稿；主实验精度表待 GPU 训练后填入）**
> 拟投：《计算机工程》/《计算机应用研究》（中文核心）
> 项目：广西大学大学生创新训练项目《基于空频双分支轻量 VIB-Net 的 AI 图像篡改检测模型研究与系统开发》
>
> **写作纪律（全稿通用）**
> 1. 每一个数字后面必须能指到 `deliverables/00_实测指标总表.md` 的具体行或原始 JSON；
> 2. 凡标 `【待补】` 的地方**必须来自正式训练日志**，禁止估算；
> 3. 不写"相位贡献是幅值的 1.7 倍"（查无出处）、不写"ForenSynths 需向作者申请"、
>    不把随机初始化基线的日志当成模型性能。

---

## 摘要

针对 AI 生成图像在社交媒体上快速扩散，而现有检测方法普遍存在"精度与体积矛盾、
跨生成模型泛化差、只判真伪不做定位"三类问题，本文提出一种**空频双分支轻量篡改检测网络
VIB-Net**。其核心包含三部分：(1) **空域分支**以轻量视觉骨干联合提取全局语义与局部纹理表征，
**频域分支**对二维离散傅里叶变换后的幅值谱与相位谱分别建模，并以可学习的卷积掩码
替代传统固定中频掩码；(2) 设计**通道—空间交叉注意力融合模块（CS-CAM）**，
以 1×1 卷积统一空频特征维度后分别计算通道与空间注意力，解决低维频域特征被高维空域特征
淹没的问题；(3) 在分类支路末端引入**分层变分信息瓶颈（VIB）**，配合 β 退火、
KL 散度裁剪与梯度裁剪三重稳定策略，抑制纹理细节带来的过拟合，同时以**梯度门控**
实现分类与定位两支路的解耦。

工程侧，本文将模型导出为 ONNX 并做算子简化，在 **无 GPU 的普通 PC（Intel 第 13 代
7 逻辑核 CPU）** 上实现 **98.7 MB** 模型体积与 **9.46~11.22 张/秒（224×224）**
的推理速度，**同时满足**"模型体积 ≤120 MB"与"CPU ≥8 张/秒"两项部署硬指标；
并基于 PyQt5 开发了完全离线运行的 Windows 桌面检测工具。

实验侧，本文报告了三类可复现的实测结果：

- **跨生成器泛化**：在 ForenSynths 测试集（13 个生成器、每生成器真/假各 100 张）上，
  当前权重取得宏平均 AUC **0.6076**、宏平均 ACC **0.5412**，其中真图 ACC 仅 **0.2877**，
  说明模型在**训练量不足**时存在明显的"倾向判伪造"偏置；
- **独立数据源测试域**：在由 ImageNet 原图与 Stable Diffusion v1.4 生成图构成的
  第三测试域（14 类、2 832 张，逐类真假 1:1）上，AUC 为 **0.4790**，
  与随机水平无异，说明**现阶段尚不具备跨数据源泛化能力**；
- **受控对照实验**：以"骨干是否加载预训练权重"为唯一自变量，
  证明**预训练权重是判决边界能够建立的必要条件**——随机初始化臂
  `tn=0`（全部判为伪造，判决退化），预训练臂 `tn=163 / tp=582`（两类都被正确判出）。

本文的贡献因此分为**方法层**（可学习空频掩码 + 分层 VIB + 梯度门控）与
**工程层**（体积—速度—精度权衡的显式实验结论，并给出"INT8 在本机只省体积不省时间"
的部署约束）。同时对尚未达标的精度指标给出根因定位与明确的补齐路径。

**关键词**：AI 图像篡改检测；频域分析；变分信息瓶颈；交叉注意力融合；轻量化部署

---

## 1 绪论

### 1.1 研究背景

生成式模型（GAN、扩散模型、自回归模型）的快速迭代，使高逼真伪造图像的获取成本
急剧下降。这类图像已成为网络谣言、金融诈骗与舆情操控的载体，对网络空间安全构成现实威胁。
与此同时，商用检测工具多为云端收费服务，存在**成本高、依赖网络、隐私风险**三重问题；
面向普通 PC 与移动端的**低成本、本地化、可解释**的鉴伪手段存在明显缺口。

### 1.2 现有方法的不足

综合国内外研究现状（见 §2），当前方法存在四类共性问题：

1. **泛化能力有限**：多数方法针对 GAN 或扩散模型单一体系设计，对未知黑盒生成器效果骤降；
2. **鲁棒性不足**：在 JPEG 压缩、缩放、裁剪、旋转等常见传播后处理下性能衰减明显；
3. **特征范式割裂**：或依赖人工设计的频域/纹理特征（适应性差），或过度依赖 CLIP 等
   大模型语义特征（易受内容干扰），**低级伪影与高级语义的平衡机制尚未统一**；
4. **轻量化与可解释性缺失**：高精度模型参数量大、依赖 GPU，难以端侧部署。

### 1.3 本文工作与贡献

本文提出空频双分支轻量 VIB-Net，贡献如下：

- **贡献 1（特征层）**：可学习的空频双分支特征提取。以卷积网络生成自适应频域掩码，
  替代固定中频掩码；同时建模幅值谱与相位谱，构成 256 维频域融合特征；
  再以 CS-CAM 交叉注意力完成空频自适应融合。
- **贡献 2（推理层）**：分层 VIB 与梯度门控双支路。VIB **仅作用于分类支路末端**，
  在保留信息瓶颈去冗余能力的同时**不破坏定位所需的细粒度空间信息**；
  并以任务级梯度门控缓解多任务梯度冲突。
- **贡献 3（优化层）**：同方差不确定性加权 + 损失 Z-score 归一化 +
  三阶段分任务训练，解决分类损失（量级 ~1.0）与定位损失（量级 ~1e-3）**量级不匹配**
  导致的单任务主导问题。
- **贡献 4（应用层）**：把"**骨干规模—模型体积—推理速度—检测精度**"的权衡做成
  一项**显式工程实验结论**（§4.7），给出在无 GPU 边缘环境下同时满足两项部署硬指标的
  具体配置，并如实报告 INT8 量化的**平台依赖**约束。

> **本文不主张的结论**（避免过度声明）：本文**不**声称已解决跨数据集泛化问题。
> §4.3、§4.4 的跨域数字（0.6076 / 0.4790）均在随机水平附近，
> 本文将其作为**诊断结果**报告，并给出根因（见 §4.5 受控实验与 §5）。

---

## 2 相关工作

### 2.1 生成图像的鉴别

Marra 等[1] 用隐写分析特征 + CNN 检测 GAN 生成图，在无损条件下表现良好，
但 JPEG 压缩下显著下降。Wang 等[7] 提出基于扩散重构误差的 DIRE，
能有效检测扩散生成图像，但依赖特定预训练扩散模型，通用性受限。
Ojha 等[8] 基于 CLIP 预训练特征构建通用检测框架，提升了跨 GAN/扩散模型的泛化性，
但仅使用最后一层特征，细粒度伪影捕捉能力较弱。Ricker 等[10] 的 AEROBLADE
利用潜空间扩散模型自编码器重建误差，速度快、部署简单，但对简单背景图像失效。
Cazenavette 等[11] 的 FakeInversion、Chen 等[12] 的 DRCT 精度较高，
但均依赖预训练扩散模型反演，**推理成本高**。

### 2.2 频域取证

Zhang 等[2] 发现 GAN 上采样模块会在频域留下周期性伪影，据此提出 AutoGAN，
但仅适用于同类结构 GAN。Durall 等[3] 用正则化损失拉近真伪图像频谱分布，
但依赖人工特征。Frank 等[4] 基于 DCT 高频异常检测，精度优于像素域 CNN，
却对高压缩图像衰减明显。Jeong 等[6] 的 BiHPF 以双边高通滤波聚焦背景伪影，
计算复杂度较高。

**本文与上述工作的差异**：(i) 掩码**可学习**而非固定中频；(ii) **幅值与相位双路**建模
（现有方法多只用幅值）；(iii) 频域模块本身极轻（**仅 0.044 M 参数，占全模型 0.2%**）。

### 2.3 变分信息瓶颈与多任务学习

VIB 通过约束输入与潜在表示间的互信息来提纯特征，但**直接用于多任务会丢失
定位所需的细粒度空间信息**，且原始 VIB 训练不稳定（KL 项易爆炸或坍缩）。
多任务学习的损失平衡方面，Kendall 等提出以**同方差不确定性**自动学习任务权重。
本文把这两条线索合并：VIB 只放在分类支路末端，并以不确定性加权协调四个损失项。

### 2.4 轻量化与端侧部署

本文不以"参数量少"作为唯一目标，而是把**体积—速度—精度**三者作为一条
显式的权衡曲线来报告（§4.7）。这与单纯追求 SOTA 精度的路线互补：
在无 GPU 的边缘环境下，"能不能跑、跑多快"往往比"再高 1 个点"更重要。

---

## 3 方法

### 3.1 总体架构

整体为"空频双域特征提取 → 交叉注意力融合 → 分层 VIB 提纯 → 梯度门控双支路推理 →
不确定性加权联合损失"的闭环（图 1）。输入图像统一缩放至 224×224×3。

```
输入 x (224×224×3)
  ├── 空域分支：CLIP-ViT 类骨干（全局语义 φ(x)） + 3 层深度可分离卷积（局部纹理 F_local）
  │        → 通道拼接 F_spa
  └── 频域分支：2D-DFT → 移频 → 幅值谱 A / 相位谱 P
           → 可学习掩码 M → A′=A⊙M, P′=P⊙M → 双路 CNN → F_freq
  → CS-CAM 自适应融合 → F_fusion (512 维)
       ├── 分类支路：VIB（仅此支路） → z(256) → MLP → 真伪概率
       └── 定位支路：Mobile-UNetv2 → 224×224×1 篡改掩码（+ Canny 边缘监督）
```

**图 1** 空频双分支 VIB-Net 总体架构（结构示意图，按本仓库 `src/models/` 的数据流绘制）。

### 3.2 空域分支

**全局语义特征**。采用 ViT-B/16 结构骨干，冻结前 6 层、微调后 6 层，
取 CLS token 作为全局语义特征：

$$\phi(x) = \text{ViT-B/16}(x)[:,0,:] \in \mathbb{R}^{768} \tag{1}$$

**局部纹理特征**。3 层深度可分离卷积（核 3×3、步长 1、填充 1，通道 32→64→128），
经全局平均池化得 128 维：

$$F_{local} = \text{GAP}(\text{DS-Conv}_3(x)) \in \mathbb{R}^{128} \tag{2}$$

**空域特征融合**：

$$F_{spa} = \text{Concat}(\phi(x), F_{local}) \in \mathbb{R}^{896} \tag{3}$$

> **实现口径（与申报书公式的一处必要澄清）**：式 (3) 给出的是**向量形式**。
> 实际实现同时保留两种形式——`patch token` 重排得到 $F_{spa}\in\mathbb{R}^{896\times14\times14}$
> 的**特征图形式**供定位支路使用；对其做 GAP 即还原为式 (3) 的 896 维向量。
> 两种形式由同一前向计算导出，**不额外引入参数**。
> （若仅按向量形式实现，定位支路将失去空间分辨率，mIoU 必然接近 0。）

### 3.3 频域分支

对输入做二维 DFT 并移频，分离幅值与相位：

$$F(u,v)=\sum_{x=0}^{223}\sum_{y=0}^{223} I(x,y)\, e^{-2\pi i(ux/224+vy/224)} \tag{4}$$

$$A(u,v)=|F_{shift}(u,v)| \tag{5}\qquad P(u,v)=\arg(F_{shift}(u,v)) \tag{6}$$

**可学习频域掩码**：两层卷积（核 5×5、步长 1、填充 2；通道 32→1）后经 Sigmoid：

$$M(u,v)=\sigma(\text{Conv}_2(\text{Conv}_1(A(u,v)))) \in [0,1]^{B\times1\times224\times224} \tag{7}$$

> 式 (7) 的**张量形状必须带 batch 与通道维**（$B\times1\times224\times224$），
> 而非申报书原文的 $[0,1]^{224\times224}$——否则无法与幅值/相位谱做逐元素相乘。

**幅相增强与特征提取**：

$$A'(u,v)=A(u,v)\odot M(u,v) \tag{8}\qquad P'(u,v)=P(u,v)\odot M(u,v) \tag{9}$$

$$F_{amp}=\text{GAP}(\text{CNN}_{amp}(A'))\in\mathbb{R}^{128} \tag{10}\qquad
F_{phase}=\text{GAP}(\text{CNN}_{phase}(P'))\in\mathbb{R}^{128} \tag{11}$$

$$F_{freq}=\text{Concat}(F_{amp},F_{phase})\in\mathbb{R}^{256} \tag{12}$$

> 本项目频域分支实测参数量 **0.044 M**，占全模型 **0.2%**。因此"引入相位支路"
> 的代价可忽略，其收益必须用消融实测值给出（见 §4.8），
> **不引用任何查不到出处的定量倍数**。

### 3.4 通道—空间交叉注意力融合（CS-CAM）

先以 1×1 卷积把两支路特征统一到同一维度：

$$F'_{spa}=\text{Conv}_{1\times1}(F_{spa})\in\mathbb{R}^{512}\tag{13}\qquad
F'_{freq}=\text{Conv}_{1\times1}(F_{freq})\in\mathbb{R}^{512}\tag{14}$$

再分别计算通道注意力 CA(·)（全局平均池化 + 全局最大池化 → 共享 MLP）与
空间注意力 SA(·)（沿通道轴做平均/最大池化 → 拼接 → **7×7 卷积**）：

$$F''_{spa}=F'_{spa}\cdot \text{CA}(F'_{spa})\cdot \text{SA}(F'_{spa})\tag{15}$$
$$F''_{freq}=F'_{freq}\cdot \text{CA}(F'_{freq})\cdot \text{SA}(F'_{freq})\tag{16}$$
$$F_{fusion}=F''_{spa}+F''_{freq}\in\mathbb{R}^{512}\tag{17}$$

### 3.5 分层 VIB 信息瓶颈

VIB 目标为最大化 $I(z;y)-\beta I(z;x)$，其变分下界：

$$L_{VIB}=-\mathbb{E}_{q(z|x)}[\log q(y|z)]+\beta\cdot \text{KL}(q(z|x)\,\|\,r(z))\tag{19}$$

均值与标准差由两层 MLP 给出（Softplus 保证 $\sigma\ge 0$）：

$$\mu=\text{MLP}_{\mu}(F_{fusion})\in\mathbb{R}^{256}\tag{20}\qquad
\sigma=\text{Softplus}(\text{MLP}_{\sigma}(F_{fusion}))\in\mathbb{R}^{256}\tag{21}$$

$$z=\mu+\epsilon\cdot\sigma,\quad \epsilon\sim\mathcal{N}(0,1)\tag{23}$$

**三重稳定策略**：

① **β 退火**（$t$ 为训练轮次）：

$$\beta(t)=\begin{cases}0, & t<20\\ 0.1\cdot\dfrac{t-20}{20}, & 20\le t<40\\ 0.1, & t\ge 40\end{cases}\tag{24}$$

② **KL 裁剪**：$\text{KL}=\text{clip}(\text{KL},0,10)$（式 25）。实现上采用
**前向硬裁剪 + 反向恒等**，避免 `torch.clamp` 在区间外梯度归零、
导致 KL 常态超限时 VIB 被**静默关闭**；同时对 batch 与 latent 维取均值归约，
使 [0,10] 成为**安全阀而非常态截断**。

③ **梯度裁剪**：梯度 L2 范数上限 **5.0**。

> **σ 的初始化**必须使 $\sigma\approx1$，否则 KL 起步即为数百、一上来就被裁到上限。

**分层设计**：VIB **仅作用于分类支路**（`apply_to: cls_only`），
底层共享特征与定位支路完全不受影响——这是"用信息瓶颈提纯而不破坏定位"的关键。

### 3.6 梯度门控双支路

分类是图像级二分类，需要**压缩抽象**以提升泛化；定位是像素级分割，需要**保留细粒度**。
两者对共享特征的需求存在本质冲突。

本文实现的是**任务级梯度门控**，需如实说明其数学行为：
在"共享 $F_{fusion}$ + 两个兄弟头"的结构下，两个头之间**没有计算图连线**，
梯度物理上不可能互相串扰；因此把梯度停止层实现成恒等映射，等价于**未加任何操作**。
本仓库因此把门控实现为**可配置开关**：

- `gs_stop_cls=True`：阻断分类支路梯度回传至 $F_{fusion}$（共享特征被定位任务主导）；
- `gs_stop_loc=True`：阻断定位支路梯度回传至 $F_{fusion}$（共享特征被分类任务主导）；
- **默认两者皆关**。

真正的硬隔离由**三阶段训练的参数冻结**承担（阶段一冻结定位支路、阶段二冻结分类支路）。
门控的作用是：阶段三联合微调时，若两个任务的验证指标出现**交替恶化**
（即梯度冲突的典型症状），可通过门控把共享特征的优化目标限定为主任务。
该机制零额外计算开销，与 PCGrad / GradNorm 属同类思路。

分类支路与定位支路分别为：

$$y_{cls}=\text{Softmax}(\text{MLP}_{cls}(z))\in\mathbb{R}^2\quad(256\to128\to2)\tag{27}$$

$$M_{pred}=\sigma(\text{Mobile-UNetv2}(F_{fusion}))\in[0,1]^{224\times224}\tag{28}$$

定位头 Mobile-UNetv2 采用倒残差编码器（4 块，步长 1/2/2/2，通道 64→128→256→512）
与双线性上采样 + 跳连解码器，**参数量 2.297 M**。

**边缘监督**：以 Canny（低阈值 50、高阈值 150）提取真实掩码边缘 $M_{edge\_gt}$，
并在定位支路末层以 1×1 卷积输出 $M_{edge\_pred}$，用二元交叉熵监督：

$$L_{edge}=\text{BCE}(M_{edge\_pred},M_{edge\_gt})\tag{29}$$

### 3.7 不确定性加权多阶段联合损失

**损失归一化**（Z-score，滑动窗口 100 批次）：

$$L'=\frac{L-\mu_L}{\sigma_L}\tag{30}$$

**同方差不确定性加权**：

$$L_{total}=\frac{1}{2\sigma_1^2}L'_{VIB}+\frac{1}{2\sigma_2^2}L'_{BCE}
+\frac{1}{2\sigma_3^2}L'_{Dice}+\frac{1}{2\sigma_4^2}L'_{edge}+\log(\sigma_1\sigma_2\sigma_3\sigma_4)\tag{31}$$

> **口径澄清**：$\sigma_i$ 是任务的**观测噪声标准差**。等价实现为
> $\sum 0.5\cdot e^{-s_i}\cdot L'_i + 0.5\sum s_i$，其中 $s_i=\log\sigma_i^2$。
> 该写法避免了原文中 $1/(2\sigma^2)$ 与 $\log\sigma$ 的口径不一致问题。

**三阶段训练**（表 1）：

| 阶段 | 轮次 | 冻结 | 训练 | 学习率 |
|---|---|---|---|---|
| ① 分类预训练 | 0–20 | localization | spatial/freq/fusion/vib/cls_head | AdamW 1e-4 → 1e-6 余弦 |
| ② 定位预训练 | 20–40 | vib/cls_head | spatial/freq/fusion/localization | AdamW 5e-5 → 5e-7 余弦 |
| ③ 联合微调 | 40–70 | — | 全部 | AdamW 1e-5 → 1e-7 余弦 |

**表 1** 三阶段分任务训练协议（权重衰减 1e-5，梯度裁剪 5.0）。

---

## 4 实验

### 4.1 数据集与实现细节

| 数据集 | 划分 | 规模 | 用途 |
|---|---|---|---|
| ForenSynths (ProGAN) | val | 8 000 张 / 20 类 | 小规模训练（官方 train 受限网络下未取得） |
| ForenSynths (多生成器) | test | 90 329 张 / 13 生成器 | 跨生成器泛化 |
| CASIA v2 | train/val/test | 10 090 / 1 261 / 1 263 | 传统篡改定位（含 5 123 个掩码） |
| GenImage SDv1.4 子集 | test | 1 416 真 + 1 416 假 | 独立数据源第三测试域 |

**数据来源口径**：ForenSynths 与 CASIA v2 均为**公开可下载**数据（无需向作者申请）；
第三测试域由 ImageNet 原图与 Stable Diffusion v1.4 类别条件生成图构成，
经**统一重编码**（512×512、JPEG q=90、去元数据）后使用。

**测试环境**（所有效率数字共用）：Windows 10 (19045)、Intel 第 13 代 **7 逻辑核**、
**无 GPU**、PyTorch 2.14.0+cpu、ONNX Runtime 1.30.0、输入 224×224、batch=1。

**指标口径（必须显式声明）**：

- **AUC**：采用**并列分数秩平均**校正。分数大量并列时（退化模型常整批输出同一值），
  不做校正的梯形法会给 0.0，**低估退化臂反而会放大本文对照的优势**，因此必须校正。
- **AP**：$\sum (R_n-R_{n-1})\cdot P_n$，且**按不同分数值分组**（并列共用一个阈值）。
  边界不变量：完美可分 = 1.0；**分数全同 = 正类占比**。
- **mIoU**：同时报三种口径——① 池化 `miou`（按像素加权，最宽松）；
  ② **仅篡改图 `miou_tampered_only`（文献标准口径，本文报这个）**；
  ③ 全部非空图 `miou_per_sample`（含误报惩罚，最严格）。
  三者可差一倍以上，**报哪个必须写明**。
- 上述指标在全仓库**只有一处实现**（`src/engine/metrics.py`），
  并有"三个入口必须逐位一致"的回归测试守护。

### 4.2 主实验

> 【待补】需 GPU 完成三阶段正式训练（官方 ForenSynths train 4 类子集约 14.4 万张）。
> 填入内容：主实验表（ACC / Precision / Recall / F1 / AUC / AP）、ROC 曲线（图 8）、
> 混淆矩阵（图 9）。**在正式训练完成前，本文不以任何小规模结果充当主实验结论。**

当前权重（仅在 ForenSynths val 1600 张上训练 3 epoch）的**诊断性结果**如下，
它说明的是"**训练量不足时的表现**"，不是方法上限：

| 数据集 | 样本量 | ACC | 真图 ACC | 假图 ACC | AUC |
|---|---|---|---|---|---|
| ForenSynths test（宏平均，13 生成器） | 2 600 | 0.5412 | **0.2877** | 0.7946 | 0.6076 |
| CASIA v2 test（分类支路） | 320 | 0.5125 | — | — | 0.4920 |

**图 8** ROC 曲线（2600 张，AUC 0.5964 微平均口径）｜**图 9** 混淆矩阵。

> 关键观察：**假图 ACC（0.7946）远高于真图 ACC（0.2877）**，
> 即模型存在强烈的"倾向判伪造"偏置。这是训练不足的典型症状，
> 与 §4.5 受控实验中随机初始化臂"全判伪造"是同一现象的不同强度表现。

### 4.3 跨生成器泛化

训练未见过任何测试生成器（测试集含 12 个非 ProGAN 生成器），
每个生成器真/假各取 100 张（真图按文件名去重），结果见图 13。

| 臂 | 宏平均 ACC | 宏平均 AUC | 宏平均 AP |
|---|---|---|---|
| ImageNet 预训练 | 0.5412 | **0.6076** | 0.6063 |
| 随机初始化 | 0.5000 | 0.5847 | 0.5953 |

逐生成器 AUC 跨度 **0.4665（whichfaceisreal）~ 0.8297（stylegan2）**，
说明模型在不同生成器上表现**严重不均**，且存在低于随机水平的个例。

**图 13** 逐生成器 ACC / AUC 对比。

> **诚实口径**：即使在与训练同源的 ProGAN 家族上（ForenSynths 测试集的 progan 子集），
> AUC 也只有 0.5279。因此本文**不能**声称"本方法在跨生成器上有效"，
> 只能说"在当前训练量下，模型的判别信号即便在训练域内也很弱"。

### 4.4 第三测试域（独立数据源）

该测试域的构建本身就是一项工作：原始数据存在**与篡改痕迹无关却近乎完美可分**的
平凡线索——"是否为 PNG 格式"单特征 AUC = **1.0000**，压缩率 AUC = 0.9954。
由于本文含**频域支路**，这类差异会经由频谱直接泄漏给模型
（JPEG 量化格栅 vs 扩散模型频谱伪影是两种截然不同的形态），
因此必须先做格式统一。

| 平凡线索 | 层级 | 未处理 | 统一重编码后 |
|---|---|---|---|
| 是否 PNG | 文件级 | **1.0000** | 0.5000 |
| 原图分辨率 | 文件级 | 0.9000 | 0.5000 |
| 每像素字节数 | 文件级 | 0.9949 | 0.6374 |
| Laplacian 方差 | 像素级 | 0.5608 | 0.6399 |

统一重编码后各线索 AUC 落到 0.5~0.64，数据**具备评测资格**。评测结果：

| 变体 | 骨干 | ACC | AUC |
|---|---|---|---|
| 统一重编码后 | ImageNet 预训练 | 0.4834 | **0.4790** |
| 未处理 | ImageNet 预训练 | 0.4852 | **0.4821** |

> **★ 一个反直觉但重要的结论**：清洗前后的 AUC 几乎一样（Δ = **−0.0031**）。
> **不能**据此说"数据没有问题"——恰恰相反，Δ≈0 是**模型太弱**的症状：
> 它根本没学会任何东西，自然也没有余力去用那个捷径。
> 数据侧的问题（捷径真实存在）与模型侧的行为（本次没用上）是**两件独立的事**，
> 必须分别验证。**"模型没用捷径"推不出"数据没问题"。**

### 4.5 表征可训练性的受控对照实验

**动机**：项目早期在真实数据上始终训不出判决边界。为避免把工程问题含糊成"玄学"，
本文把它拆成两条**可证伪的判据**：

| 判据 | 定义 |
|---|---|
| **退化** | 阈值判决下 `tn=0`（全判假）或 `tp=0`（全判真）→ 边界不存在 |
| **有效** | 同一 epoch 内 `tn>0` **且** `tp>0` → 两类都有被正确判出的样本 |

**设置**：唯一自变量为"骨干是否加载 ImageNet-1K 预训练权重"。
两臂配置文件过滤注释后**仅 3 处不同**：`project.name`、`project.output_dir`（均不影响训练）
与 `model.spatial.backbone_weights`（**唯一自变量**）。协议：`stage1_cls_pretrain` 3 epoch、
batch 8、AdamW、lr 1e-4→1e-6、seed 3407。

| 臂 | 最佳 epoch | val ACC | val AUC | **tn** | **tp** | 判决 |
|---|---|---|---|---|---|---|
| 随机初始化 | 2 | 0.5000 | 0.6428 | **0** | 650 | ❌ **退化（全判伪造）** |
| ImageNet 预训练 | 2 | 0.5731 | **0.6809** | **163** | **582** | ✅ **有效（边界已建立）** |

**图 7** 两臂验证集 AUC 曲线｜**图 7b** 判决是否建立：`tn` / `tp` 随时间变化。

**结论及其强度边界（必须如实标注）**：

1. 定性结论成立：预训练权重把模型从"判决退化"推进到"判决有效"，这是**定性差别**。
2. **但数值差距不大**（AUC +0.0381）。随机臂 AUC 0.6428 说明其排序分数
   **并非全无信号**——一个合理解释是：**频域分支是确定性变换**，
   其输出与骨干是否预训练无关，故随机初始化骨干下分类头仍能读到一点生成痕迹。
3. 所以正确表述是：**预训练权重是判决边界能够建立的必要条件（在本机 CPU 小规模设定下），
   而非充分条件。**
4. **口径限制**：训练/验证同为 ProGAN 生成（**同生成器留出**），只能说明可训练性，
   不能外推为跨生成器性能；且训练集用的是 val 划分 1600 张，
   属**小规模替代**，**不声称复现官方协议**；本次子采样实测有 **3 张**图像内容重叠，
   **不可写"无交集"**。

### 4.6 鲁棒性

对测试集施加 **26 种常见后处理**（JPEG q=10~90、缩放 0.5~2.0、
随机裁剪 10%~50%、旋转 0/90/180/270°、高斯噪声 σ=0.01~0.05、
椒盐噪声 d=0.01~0.05），统计 ACC 与 F1 的变化。基线（原图）ACC = 0.6250。

| 后处理 | ACC | ACC 下降 | 后处理 | ACC | ACC 下降 |
|---|---|---|---|---|---|
| **原图（基线）** | **0.6250** | — | 旋转 90°/180°/270° | 0.6042 | +0.0208 |
| JPEG q=10 / 20 | 0.6042 | +0.0208 | 高斯噪声 σ=0.01 / 0.02 | 0.6250 / 0.6042 | +0.00 / +0.02 |
| JPEG q=30~90 | 0.6250 | +0.0000 | 高斯噪声 σ=0.05 | 0.6875 | −0.0625 |
| 缩放 ×0.5 / ×0.75 / ×1.25 | 0.5208 | +0.1042 | 椒盐噪声 d=0.01 | 0.6042 | +0.0208 |
| 缩放 ×1.5 / ×2.0 | 0.5417 | +0.0833 | 椒盐噪声 d=0.02 | 0.5208 | +0.1042 |
| 裁剪 10%~40% | 0.52~0.54 | +0.08~0.10 | **椒盐噪声 d=0.05** | **0.5000**（F1=0） | **+0.1250** |
| 裁剪 50% | 0.5000 | +0.1250 | | | |

**图 11** 鲁棒性曲线。

**三条必须如实说明的观察**：

1. **JPEG 压缩几乎不降低 ACC**，这与"JPEG 是频域检测器主要威胁"的常见结论相反。
   原因**不是**模型鲁棒，而是它**太弱**——当前权重根本没有依赖高频伪影做判别，
   压缩自然伤不到它。**本文不把它作为"鲁棒性好"的证据。**
2. **椒盐噪声 d=0.05 时 F1 = 0.0000、ACC = 0.5000**，即模型完全退化（全判同一类），
   是最脆弱的扰动。
3. ⚠️ **样本量仅 48 张，单档 ACC 标准误约 ±0.07**，
   因此**逐档差异小于 0.07 的不应解读为真实差异**。
   本节的定位是"**验证评测管线可用并给出初值**"；
   **正式鲁棒性结论需在 GPU 训练完成后用 ≥500 张重测。**

### 4.7 部署与效率（本文最扎实的一节）

#### 4.7.1 骨架—体积—速度—精度权衡

| 配置 | 档位 | 参数量 (M) | FP32 体积 (MB) | PyTorch CPU 吞吐 (张/秒) |
|---|---|---|---|---|
| default | b16 | 90.53 | 362.1 | 1.67 |
| default | s16 | 26.2 | 104.8 | 2.56 |
| default | ti16 | 9.96 | 39.8 | 2.69 |
| default | ti8 | 8.18 | 32.7 | 4.33 |
| **lite** | **s16** | **24.50** | **98.0** | 2.55 |

**图 12a** 体积—速度散点（PyTorch）。

> **PyTorch eager 下没有任何档位达到 8 张/秒。** 达标必须叠加 ONNX Runtime 与算子简化。

#### 4.7.2 ONNX 导出与算子简化

| 模型 | 体积 | 中位延迟 | 中位吞吐 | ≥8 张/秒 |
|---|---|---|---|---|
| `lite.onnx` | 99.2 MB | 137.7 ms | 7.26 | ❌ |
| **`lite_simplified.onnx`** | **98.7 MB** | **105.7 ms** | **9.46** | **✅** |
| `lite_simplified.onnx`（复测） | 98.7 MB | 89.2 ms | **11.22** | ✅ |

**图 12b** ONNX 实测体积 vs 吞吐（含 8 张/秒约束线）。

> 同一模型 PyTorch eager 与 ONNX Runtime 相差 **3.7 倍**（392 ms vs 106 ms）。

#### 4.7.3 INT8 量化的平台依赖（一条负面但有用的结论）

| 模型 | 体积 | 中位延迟 | 吞吐 | 结论 |
|---|---|---|---|---|
| lite + 动态 INT8 | **25.7 MB** | 298.6 ms | 3.35 | 体积 ↓74%，**速度 ↓68%** |
| b16 + 动态 INT8 | 92.9 MB | 349.1 ms | 2.86 | 体积达标，速度不达标 |

数值一致性：输出最大相对偏差 **0.78%**（`pass: true`）。

> **本机 CPU 缺少 VNNI 指令**，batch=1 时 `DynamicQuantizeLinear` 的开销
> 超过 INT8 矩阵乘省下的时间，故**量化省体积、不省时间**。
> 因此本文把 INT8 的定位写成"**体积压缩手段**，速度收益取决于目标平台指令集"，
> 而不是笼统的"量化后推理更快"。这一点在换到支持 VNNI 的服务器 CPU 上可能不同，
> 需另行实测。

**与申报书两条硬指标的对账**：

| 指标 | 实测 | 判定 |
|---|---|---|
| 模型体积 ≤120 MB（**部署形态**：s16 + ONNX 简化） | **98.7 MB** | ✅ |
| CPU ≥8 张/秒（**ONNX Runtime FP32 / 224×224 / batch=1 / Intel 第 13 代 7 逻辑核 / 多轮中位数**） | **9.46 ~ 11.22** | ✅ |

#### 4.7.4 桌面检测工具

基于 PyQt5 实现完全离线的 Windows 桌面工具，五大模块：图像输入（单张/批量/拖拽）、
检测处理（异步推理，避免界面卡顿）、结果展示（原图 + 篡改热力图 + 检测报告）、
结果导出（PDF 报告 + PNG 掩码）、系统设置（阈值/保存路径/推理设备）。
支持 torch / ONNX / OpenVINO 三种后端。

**图 14** 桌面工具界面（检测结果与热力图）。

### 4.8 消融实验

> 【待补】每项消融需**独立重新训练**（`--mode eval` 不能出论文数据）。
> 受单人算力所限，消融项从申报书的 7 项精简为 **4 项核心模块**：
> 梯度门控 / 相位支路 / 分层 VIB / CS-CAM；
> 协议为**仅 stage1 分类预训练**（论文中必须写明"消融基于分类支路，
> 在 stage1 协议下进行"，因为消融关心的是**相对差异**而非绝对精度）。

**图 10** 消融对比柱状图。**表 5** 消融表。

### 4.9 篡改定位

| 指标 | 值 | 说明 |
|---|---|---|
| 池化 mIoU | 0.0000 | 最宽松口径 |
| **仅篡改图 mIoU**（文献标准） | **0.0000** | 论文报这个 |
| 全部非空图 mIoU | 0.0000 | 最严格口径 |
| Pixel Acc | 0.9560 | **虚高**：输出"全背景"即可得到 |

**图 16** 定位可视化（原图 / GT 掩码 / 预测掩码）。

> **如实说明**：三个口径全为 0，原因是定位支路**只在无像素掩码的 ForenSynths 上
> 随分类任务一起跑过，从未在像素掩码上训练**，预测掩码退化为"全背景"
> （预测概率最大值仅 0.4847~0.4848，可视化见 图 16）。
> Pixel Acc 0.9560 是**假高**——真实图像中篡改像素占比很小，全判背景即可拿到高像素准确率，
> **这正是为什么不能只报 Pixel Acc**。
> 补齐路径明确：在 GPU 上完成 `stage2_loc_pretrain`（CASIA v2 掩码训练）。

---

## 5 结论与展望

### 5.1 结论

1. 本文实现了**空频双分支轻量 VIB-Net** 的完整工程：空域（全局+局部）、
   频域（幅相联合 + 可学习掩码）、CS-CAM 融合、分层 VIB、梯度门控双支路、
   不确定性加权三阶段训练，以及 ONNX/OpenVINO 部署与 PyQt5 离线桌面工具。
   代码与公式**逐条对照实现**，并通过 **37 项自动化自检**。
2. **工程硬指标已实测达标**：模型体积 **98.7 MB**（≤120 MB）、
   CPU 吞吐 **9.46~11.22 张/秒**（≥8 张/秒），且是在**无 GPU 的普通 PC** 上达到。
3. 报告了一条**显式的体积—速度—精度权衡结论**，并给出"INT8 在本机只省体积不省时间"
   的部署约束（平台指令集依赖）。
4. 通过**受控对照实验**把"训不起来"从定性抱怨变成可证伪结论：
   **骨干预训练权重是判决边界能建立的必要条件**，且如实标注了该对照的强度边界（AUC 仅差 +0.0381）。
5. **最重要的诚实结论**：在当前算力条件下，**精度指标尚未达标**
   （跨生成器宏平均 ACC 0.5412 / AUC 0.6076；第三测试域 AUC 0.4790；
   定位 mIoU 0.0000）。根因已定位为**训练量不足与定位支路未在掩码上训练**，
   **不是网络结构缺陷**。

### 5.2 局限

- **跨数据集泛化能力尚未建立**：0.6076 与 0.4790 都在随机水平附近，
  本文不主张任何跨域结论。
- **定位支路未训练**：mIoU 目前为 0，属未完成项。
- **未使用官方完整训练协议**：训练数据为 ForenSynths val 1600 张，
  非官方 train 划分（受限网络下未取得），且实测存在 3 张图像内容重叠。
- **骨干替换**：受网络限制未取得 CLIP 权重，对照实验使用了同为 ViT-B/16 结构的
   ImageNet-1K 预训练权重；**"CLIP 语义先验"这一说法需拿到 CLIP 权重后重跑才能主张**。

### 5.3 展望

1. 在 GPU 上以官方 ForenSynths train（20 类）完成三阶段训练，补齐主实验表；
2. 在 CASIA v2 掩码上完成 `stage2_loc_pretrain`，使 mIoU 可测；
3. 完成 4 项核心消融，把相位支路与 VIB 的增益用**实测值**坐实
   （同时替换掉申报书里"相位贡献是幅值的 1.7 倍"这一**查不到出处**的表述）；
4. 扩充第三测试域到更多生成器，把"跨数据源"与"跨伪造类型"两类泛化分开报告。

---

## 参考文献

（按申报书参考文献表，共 24 条，此处保留编号与格式，投稿前按目标期刊模板调整）

[1] MARRA F, GRAGNANIELLO D, COZZOLINO D, et al. Detection of GAN-generated fake images over social networks[C]//IEEE Conference on Multimedia Information Processing and Retrieval. Miami: IEEE, 2018.
[2] ZHANG X, KARAMAN S, CHANG S F. Detecting and simulating artifacts in GAN fake images[C]//IEEE International Workshop on Information Forensics and Security. Delft: IEEE, 2019.
[3] DURALL R, KEUPER M, KEUPER J. Watch your up-convolution: CNN based generative deep neural networks are failing to reproduce spectral distributions[C]//IEEE/CVF CVPR. Seattle: IEEE, 2020.
[4] FRANK J, EISENHOFER T, SCHÖNHERR L, et al. Leveraging frequency analysis for deep fake image recognition[C]//ICML. Virtual: JMLR, 2020.
[5] HE Y, YU N, KEUPER M, et al. Beyond the spectrum: detecting deepfakes via re-synthesis[C]//IJCAI. Montreal, 2021.
[6] JEONG Y, KIM D, MIN S, et al. BiHPF: bilateral high-pass filters for robust deepfake detection[C]//IEEE/CVF WACV. Waikoloa: IEEE, 2022.
[7] WANG Z D, BAO J M, ZHOU W G, et al. DIRE for diffusion-generated image detection[C]//IEEE/CVF ICCV. Paris: IEEE, 2023.
[8] OJHA U, LI Y H, LEE Y J. Towards universal fake image detectors that generalize across generative models[C]//IEEE/CVF CVPR. Vancouver: IEEE, 2023.
[9] MA R P, DUAN J H, KONG F, et al. Exposing the fake: effective diffusion-generated images detection[EB/OL]. arXiv:2307.06272, 2023.
[10] RICKER J, LUKOVNIKOV D, FISCHER A. AEROBLADE: training-free detection of latent diffusion images using autoencoder reconstruction error[C]//IEEE/CVF CVPR. Seattle: IEEE, 2024.
[11] CAZENAVETTE G, SUD A, LEUNG T, et al. FakeInversion: learning to detect images from unseen text-to-image models by inverting stable diffusion[C]//IEEE/CVF CVPR. Seattle: IEEE, 2024.
[12] CHEN B Y, ZENG J S, YANG J Q, et al. DRCT: diffusion reconstruction contrastive training towards universal detection of diffusion generated images[C]//ICML. Vienna: JMLR, 2024.
[13] PONTORNO O, GUARNERA L, BATTIATO S. On the exploitation of DCT-traces in the generative-AI domain[C]//IEEE ICIP. Abu Dhabi: IEEE, 2024.
[14] OPPENHEIM A V, LIM J S. The importance of phase in signals[J]. Proceedings of the IEEE, 1981, 69(5): 529-541.
[15] WOO S, PARK J, LEE J Y, et al. CBAM: convolutional block attention module[C]//ECCV. Munich: Springer, 2018.
[16] ALEMI A A, FISCHER I, DILLON J V, et al. Deep variational information bottleneck[C]//ICLR. Toulon, 2017.
[17] KENDALL A, GAL Y, CIPOLLA R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics[C]//IEEE/CVF CVPR. Salt Lake City: IEEE, 2018.
[18] SANDLER M, HOWARD A, ZHU M, et al. MobileNetV2: inverted residuals and linear bottlenecks[C]//IEEE/CVF CVPR. Salt Lake City: IEEE, 2018.
[19] RADFORD A, KIM J W, HALLACY C, et al. Learning transferable visual models from natural language supervision[C]//ICML. Virtual: JMLR, 2021.
[20] DOSOVITSKIY A, BEYER L, KOLESNIKOV A, et al. An image is worth 16×16 words: transformers for image recognition at scale[C]//ICLR. Virtual, 2021.
[21] ZHU M, CHEN K, YANG H, et al. GenImage: a million-scale benchmark for detecting AI-generated image[EB/OL]. arXiv:2306.08571, 2023.
[22] DONG C, CHEN X, HE R, et al. MVSS-Net: multi-view multi-scale supervised networks for image manipulation detection[J]. IEEE TPAMI, 2023, 45(3): 3539-3553.
[23] KWON M J, YU I J, NAM S H, et al. CAT-Net: compression artifact tracing network for detection and localization of image splicing[C]//IEEE/CVF WACV. Waikoloa: IEEE, 2021.
[24] WU H, ZHOU J, TIAN J, et al. Robust image forgery detection over online social network shared images[C]//IEEE/CVF CVPR. New Orleans: IEEE, 2022.

---

## 附：稿件待办清单（投稿前逐项打勾）

- [ ] §4.2 主实验表填入正式训练结果（含图 8 ROC、图 9 混淆矩阵）
- [ ] §4.8 消融表填入 4 项核心模块的独立重训结果（图 10）
- [ ] §4.6 鲁棒性表按正式权重重测（图 11）
- [ ] §4.9 定位 mIoU 在 `stage2_loc_pretrain` 完成后重测（图 16）
- [ ] 补绘 图 1/2/3（架构示意图）、图 10（消融柱状图）
- [ ] 全文检索：**"1.7 倍""约 10 万张""需向作者申请"** 三个错误表述是否已清除
- [ ] 全文检索：是否把**随机初始化基线日志**误当成模型性能引用
- [ ] 全文检索：所有效率数字是否都带"224×224 / batch=1 / 7 逻辑核 / ONNX Runtime"
- [ ] 参考文献按目标期刊模板重新排版（当前含 10 条新增文献，需核实卷期页码）
