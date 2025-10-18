# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-10-17 17:00:26.309735 PST.

### Artificial Intelligence

### 1. [JEDA: Query-Free Clinical Order Search from Ambient Dialogues](http://arxiv.org/pdf/2510.14169v1)

Authors: Praphul Singh, Corey Barrett, Sumana Srivasta, Amitabh Saikia, Irfan Bulu, Sri Gadde, Krishnaram Kenthapadi

Clinical conversations mix explicit directives (order a chest X-ray) with
implicit reasoning (the cough worsened overnight, we should check for
pneumonia). Many systems rely on LLM rewriting, adding latency, instability,
and opacity that hinder real-time ordering. We present JEDA (Joint Embedding
for Direct and Ambient clinical orders), a domain-initialized bi-encoder that
retrieves canonical orders directly and, in a query-free mode, encodes a short
rolling window of ambient dialogue to trigger retrieval. Initialized from
PubMedBERT and fine-tuned with a duplicate-safe contrastive objective, JEDA
aligns heterogeneous expressions of intent to shared order concepts. Training
uses constrained LLM guidance to tie each signed order to complementary
formulations (command only, context only, command+context, context+reasoning),
producing clearer inter-order separation, tighter query extendash order
coupling, and stronger generalization. The query-free mode is noise-resilient,
reducing sensitivity to disfluencies and ASR errors by conditioning on a short
window rather than a single utterance. Deployed in practice, JEDA yields large
gains and substantially outperforms its base encoder and recent open embedders
(Linq Embed Mistral, SFR Embedding, GTE Qwen, BGE large, Embedding Gemma). The
result is a fast, interpretable, LLM-free retrieval layer that links ambient
context to actionable clinical orders in real time.

### 2. [Implementation of AI in Precision Medicine](http://arxiv.org/pdf/2510.14194v1)

Authors: Göktuğ Bender, Samer Faraj, Anand Bhardwaj

Artificial intelligence (AI) has become increasingly central to precision
medicine by enabling the integration and interpretation of multimodal data, yet
implementation in clinical settings remains limited. This paper provides a
scoping review of literature from 2019-2024 on the implementation of AI in
precision medicine, identifying key barriers and enablers across data quality,
clinical reliability, workflow integration, and governance. Through an
ecosystem-based framework, we highlight the interdependent relationships
shaping real-world translation and propose future directions to support
trustworthy and sustainable implementation.

### 3. [Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks](http://arxiv.org/pdf/2510.14207v1)

Authors: Trilok Padhi, Pinxian Lu, Abdulkadir Erol, Tanmay Sutar, Gauri Sharma, Mina Sonmez, Munmun De Choudhury, Ugur Kursuncu

Large Language Model (LLM) agents are powering a growing share of interactive
web applications, yet remain vulnerable to misuse and harm. Prior jailbreak
research has largely focused on single-turn prompts, whereas real harassment
often unfolds over multi-turn interactions. In this work, we present the Online
Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn
harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim)
simulation informed by repeated game theory, (iii) three jailbreak methods
attacking agents across memory, planning, and fine-tuning, and (iv) a
mixed-methods evaluation framework. We utilize two prominent LLMs,
LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our
results show that jailbreak tuning makes harassment nearly guaranteed with an
attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama,
and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal
rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with
84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs.
31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive
categories such as sexual or racial harassment. Qualitative evaluation further
reveals that attacked agents reproduce human-like aggression profiles, such as
Machiavellian/psychopathic patterns under planning, and narcissistic tendencies
with memory. Counterintuitively, closed-source and open-source models exhibit
distinct escalation trajectories across turns, with closed-source models
showing significant vulnerability. Overall, our findings show that multi-turn
and theory-grounded attacks not only succeed at high rates but also mimic
human-like harassment dynamics, motivating the development of robust safety
guardrails to ultimately keep online platforms safe and responsible.

### 4. [LiveResearchBench: A Live Benchmark for User-Centric Deep Research in the Wild](http://arxiv.org/pdf/2510.14240v1)

Authors: Jiayu Wang, Yifei Ming, Riya Dulepet, Qinglin Chen, Austin Xu, Zixuan Ke, Frederic Sala, Aws Albarghouthi, Caiming Xiong, Shafiq Joty

Deep research -- producing comprehensive, citation-grounded reports by
searching and synthesizing information from hundreds of live web sources --
marks an important frontier for agentic systems. To rigorously evaluate this
ability, four principles are essential: tasks should be (1) user-centric,
reflecting realistic information needs, (2) dynamic, requiring up-to-date
information beyond parametric knowledge, (3) unambiguous, ensuring consistent
interpretation across users, and (4) multi-faceted and search-intensive,
requiring search over numerous web sources and in-depth analysis. Existing
benchmarks fall short of these principles, often focusing on narrow domains or
posing ambiguous questions that hinder fair comparison. Guided by these
principles, we introduce LiveResearchBench, a benchmark of 100 expert-curated
tasks spanning daily life, enterprise, and academia, each requiring extensive,
dynamic, real-time web search and synthesis. Built with over 1,500 hours of
human labor, LiveResearchBench provides a rigorous basis for systematic
evaluation. To evaluate citation-grounded long-form reports, we introduce
DeepEval, a comprehensive suite covering both content- and report-level
quality, including coverage, presentation, citation accuracy and association,
consistency and depth of analysis. DeepEval integrates four complementary
evaluation protocols, each designed to ensure stable assessment and high
agreement with human judgments. Using LiveResearchBench and DeepEval, we
conduct a comprehensive evaluation of 17 frontier deep research systems,
including single-agent web search, single-agent deep research, and multi-agent
systems. Our analysis reveals current strengths, recurring failure modes, and
key system components needed to advance reliable, insightful deep research.

### 5. [Towards Agentic Self-Learning LLMs in Search Environment](http://arxiv.org/pdf/2510.14253v1)

Authors: Wangtao Sun, Xiang Cheng, Jialin Fan, Yao Xu, Xing Yu, Shizhu He, Jun Zhao, Kang Liu

We study whether self-learning can scale LLM-based agents without relying on
human-curated datasets or predefined rule-based rewards. Through controlled
experiments in a search-agent setting, we identify two key determinants of
scalable agent training: the source of reward signals and the scale of agent
task data. We find that rewards from a Generative Reward Model (GRM) outperform
rigid rule-based signals for open-domain learning, and that co-evolving the GRM
with the policy further boosts performance. Increasing the volume of agent task
data-even when synthetically generated-substantially enhances agentic
capabilities. Building on these insights, we propose \textbf{Agentic
Self-Learning} (ASL), a fully closed-loop, multi-role reinforcement learning
framework that unifies task generation, policy execution, and evaluation within
a shared tool environment and LLM backbone. ASL coordinates a Prompt Generator,
a Policy Model, and a Generative Reward Model to form a virtuous cycle of
harder task setting, sharper verification, and stronger solving. Empirically,
ASL delivers steady, round-over-round gains, surpasses strong RLVR baselines
(e.g., Search-R1) that plateau or degrade, and continues improving under
zero-labeled-data conditions, indicating superior sample efficiency and
robustness. We further show that GRM verification capacity is the main
bottleneck: if frozen, it induces reward hacking and stalls progress; continual
GRM training on the evolving data distribution mitigates this, and a small
late-stage injection of real verification data raises the performance ceiling.
This work establishes reward source and data scale as critical levers for
open-domain agent learning and demonstrates the efficacy of multi-role
co-evolution for scalable, self-improving agents. The data and code of this
paper are released at
https://github.com/forangel2014/Towards-Agentic-Self-Learning

### 6. [MorphoBench: A Benchmark with Difficulty Adaptive to Model Reasoning](http://arxiv.org/pdf/2510.14265v1)

Authors: Xukai Wang, Xuanbo Liu, Mingrui Chen, Haitian Zhong, Xuanlin Yang, Bohan Zeng, Jinbo Hu, Hao Liang, Junbo Niu, Xuchen Li, Ruitao Wu, Ruichuan An, Yang Shi, Liu Liu, Xu-Yao Zhang, Qiang Liu, Zhouchen Lin, Wentao Zhang, Bin Dong

With the advancement of powerful large-scale reasoning models, effectively
evaluating the reasoning capabilities of these models has become increasingly
important. However, existing benchmarks designed to assess the reasoning
abilities of large models tend to be limited in scope and lack the flexibility
to adapt their difficulty according to the evolving reasoning capacities of the
models. To address this, we propose MorphoBench, a benchmark that incorporates
multidisciplinary questions to evaluate the reasoning capabilities of large
models and can adjust and update question difficulty based on the reasoning
abilities of advanced models. Specifically, we curate the benchmark by
selecting and collecting complex reasoning questions from existing benchmarks
and sources such as Olympiad-level competitions. Additionally, MorphoBench
adaptively modifies the analytical challenge of questions by leveraging key
statements generated during the model's reasoning process. Furthermore, it
includes questions generated using simulation software, enabling dynamic
adjustment of benchmark difficulty with minimal resource consumption. We have
gathered over 1,300 test questions and iteratively adjusted the difficulty of
MorphoBench based on the reasoning capabilities of models such as o3 and GPT-5.
MorphoBench enhances the comprehensiveness and validity of model reasoning
evaluation, providing reliable guidance for improving both the reasoning
abilities and scientific robustness of large models. The code has been released
in https://github.com/OpenDCAI/MorphoBench.

### 7. [A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space](http://arxiv.org/pdf/2510.14301v1)

Authors: Bingjie Zhang, Yibo Yang, Renzhe, Dandan Guo, Jindong Gu, Philip Torr, Bernard Ghanem

Large language models (LLMs) have achieved remarkable success in diverse
tasks, yet their safety alignment remains fragile during adaptation. Even when
fine-tuning on benign data or with low-rank adaptation, pre-trained safety
behaviors are easily degraded, leading to harmful responses in the fine-tuned
models. To address this challenge, we propose GuardSpace, a guardrail framework
for preserving safety alignment throughout fine-tuning, composed of two key
components: a safety-sensitive subspace and a harmful-resistant null space.
First, we explicitly decompose pre-trained weights into safety-relevant and
safety-irrelevant components using covariance-preconditioned singular value
decomposition, and initialize low-rank adapters from the safety-irrelevant
ones, while freezing safety-relevant components to preserve their associated
safety mechanism. Second, we construct a null space projector that restricts
adapter updates from altering safe outputs on harmful prompts, thereby
maintaining the original refusal behavior. Experiments with various pre-trained
models on multiple downstream tasks demonstrate that GuardSpace achieves
superior performance over existing methods. Notably, for Llama-2-7B-Chat
fine-tuned on GSM8K, GuardSpace outperforms the state-of-the-art method AsFT,
reducing the average harmful score from 14.4% to 3.6%, while improving the
accuracy from from 26.0% to 28.0%.

### 8. [Metacognitive Self-Correction for Multi-Agent System via Prototype-Guided Next-Execution Reconstruction](http://arxiv.org/pdf/2510.14319v1)

Authors: Xu Shen, Qi Zhang, Song Wang, Zhen Tan, Xinyu Zhao, Laura Yao, Vaishnav Tadiparthi, Hossein Nourkhiz Mahjoub, Ehsan Moradi Pari, Kwonjoon Lee, Tianlong Chen

Large Language Model based multi-agent systems (MAS) excel at collaborative
problem solving but remain brittle to cascading errors: a single faulty step
can propagate across agents and disrupt the trajectory. In this paper, we
present MASC, a metacognitive framework that endows MAS with real-time,
unsupervised, step-level error detection and self-correction. MASC rethinks
detection as history-conditioned anomaly scoring via two complementary designs:
(1) Next-Execution Reconstruction, which predicts the embedding of the next
step from the query and interaction history to capture causal consistency, and
(2) Prototype-Guided Enhancement, which learns a prototype prior over
normal-step embeddings and uses it to stabilize reconstruction and anomaly
scoring under sparse context (e.g., early steps). When an anomaly step is
flagged, MASC triggers a correction agent to revise the acting agent's output
before information flows downstream. On the Who&When benchmark, MASC
consistently outperforms all baselines, improving step-level error detection by
up to 8.47% AUC-ROC ; When plugged into diverse MAS frameworks, it delivers
consistent end-to-end gains across architectures, confirming that our
metacognitive monitoring and targeted correction can mitigate error propagation
with minimal overhead.

### 9. [Can MLLMs Absorb Math Reasoning Abilities from LLMs as Free Lunch?](http://arxiv.org/pdf/2510.14387v1)

Authors: Yijie Hu, Zihao Zhou, Kaizhu Huang, Xiaowei Huang, Qiufeng Wang

Math reasoning has been one crucial ability of large language models (LLMs),
where significant advancements have been achieved in recent years. However,
most efforts focus on LLMs by curating high-quality annotation data and
intricate training (or inference) paradigms, while the math reasoning
performance of multi-modal LLMs (MLLMs) remains lagging behind. Since the MLLM
typically consists of an LLM and a vision block, we wonder: Can MLLMs directly
absorb math reasoning abilities from off-the-shelf math LLMs without tuning?
Recent model-merging approaches may offer insights into this question. However,
they overlook the alignment between the MLLM and LLM, where we find that there
is a large gap between their parameter spaces, resulting in lower performance.
Our empirical evidence reveals two key factors behind this issue: the
identification of crucial reasoning-associated layers in the model and the
mitigation of the gaps in parameter space. Based on the empirical insights, we
propose IP-Merging that first identifies the reasoning-associated parameters in
both MLLM and Math LLM, then projects them into the subspace of MLLM, aiming to
maintain the alignment, and finally merges parameters in this subspace.
IP-Merging is a tuning-free approach since parameters are directly adjusted.
Extensive experiments demonstrate that our IP-Merging method can enhance the
math reasoning ability of MLLMs directly from Math LLMs without compromising
their other capabilities.

### 10. [Hi-Agent: Hierarchical Vision-Language Agents for Mobile Device Control](http://arxiv.org/pdf/2510.14388v1)

Authors: Zhe Wu, Hongjin Lu, Junliang Xing, Changhao Zhang, Yin Zhu, Yuhao Yang, Yuheng Jing, Kai Li, Kun Shao, Jianye Hao, Jun Wang, Yuanchun Shi

Building agents that autonomously operate mobile devices has attracted
increasing attention. While Vision-Language Models (VLMs) show promise, most
existing approaches rely on direct state-to-action mappings, which lack
structured reasoning and planning, and thus generalize poorly to novel tasks or
unseen UI layouts. We introduce Hi-Agent, a trainable hierarchical
vision-language agent for mobile control, featuring a high-level reasoning
model and a low-level action model that are jointly optimized. For efficient
training, we reformulate multi-step decision-making as a sequence of
single-step subgoals and propose a foresight advantage function, which
leverages execution feedback from the low-level model to guide high-level
optimization. This design alleviates the path explosion issue encountered by
Group Relative Policy Optimization (GRPO) in long-horizon tasks and enables
stable, critic-free joint training. Hi-Agent achieves a new State-Of-The-Art
(SOTA) 87.9% task success rate on the Android-in-the-Wild (AitW) benchmark,
significantly outperforming prior methods across three paradigms: prompt-based
(AppAgent: 17.7%), supervised (Filtered BC: 54.5%), and reinforcement
learning-based (DigiRL: 71.9%). It also demonstrates competitive zero-shot
generalization on the ScreenSpot-v2 benchmark. On the more challenging
AndroidWorld benchmark, Hi-Agent also scales effectively with larger backbones,
showing strong adaptability in high-complexity mobile control scenarios.

### Hardware Architecture

### 1. [DIAMOND: Systolic Array Acceleration of Sparse Matrix Multiplication for Quantum Simulation](http://arxiv.org/pdf/2510.14172v1)

Authors: Yuchao Su, Srikar Chundury, Jiajia Li, Frank Mueller

Hamiltonian simulation is a key workload in quantum computing, enabling the
study of complex quantum systems and serving as a critical tool for classical
verification of quantum devices. However, it is computationally challenging
because the Hilbert space dimension grows exponentially with the number of
qubits. The growing dimensions make matrix exponentiation, the key kernel in
Hamiltonian simulations, increasingly expensive. Matrix exponentiation is
typically approximated by the Taylor series, which contains a series of matrix
multiplications. Since Hermitian operators are often sparse, sparse matrix
multiplication accelerators are essential for improving the scalability of
classical Hamiltonian simulation. Yet, existing accelerators are primarily
designed for machine learning workloads and tuned to their characteristic
sparsity patterns, which differ fundamentally from those in Hamiltonian
simulations that are often dominated by structured diagonals.
  In this work, we present \name, the first diagonal-optimized quantum
simulation accelerator. It exploits the diagonal structure commonly found in
problem-Hamiltonian (Hermitian) matrices and leverages a restructured systolic
array dataflow to transform diagonally sparse matrices into dense computations,
enabling high utilization and performance. Through detailed cycle-level
simulation of diverse benchmarks in HamLib, \name{} demonstrates average
performance improvements of $10.26\times$, $33.58\times$, and $53.15\times$
over SIGMA, Outer Product, and Gustavson's algorithm, respectively, with peak
speedups up to $127.03\times$ while reducing energy consumption by an average
of $471.55\times$ and up to $4630.58\times$ compared to SIGMA.

### 2. [Computing-In-Memory Aware Model Adaption For Edge Devices](http://arxiv.org/pdf/2510.14379v1)

Authors: Ming-Han Lin, Tian-Sheuan Chang

Computing-in-Memory (CIM) macros have gained popularity for deep learning
acceleration due to their highly parallel computation and low power
consumption. However, limited macro size and ADC precision introduce throughput
and accuracy bottlenecks. This paper proposes a two-stage CIM-aware model
adaptation process. The first stage compresses the model and reallocates
resources based on layer importance and macro size constraints, reducing model
weight loading latency while improving resource utilization and maintaining
accuracy. The second stage performs quantization-aware training, incorporating
partial sum quantization and ADC precision to mitigate quantization errors in
inference. The proposed approach enhances CIM array utilization to 90\%,
enables concurrent activation of up to 256 word lines, and achieves up to 93\%
compression, all while preserving accuracy comparable to previous methods.

### 3. [Low Power Vision Transformer Accelerator with Hardware-Aware Pruning and Optimized Dataflow](http://arxiv.org/pdf/2510.14393v1)

Authors: Ching-Lin Hsiung, Tian-Sheuan Chang

Current transformer accelerators primarily focus on optimizing self-attention
due to its quadratic complexity. However, this focus is less relevant for
vision transformers with short token lengths, where the Feed-Forward Network
(FFN) tends to be the dominant computational bottleneck. This paper presents a
low power Vision Transformer accelerator, optimized through algorithm-hardware
co-design. The model complexity is reduced using hardware-friendly dynamic
token pruning without introducing complex mechanisms. Sparsity is further
improved by replacing GELU with ReLU activations and employing dynamic FFN2
pruning, achieving a 61.5\% reduction in operations and a 59.3\% reduction in
FFN2 weights, with an accuracy loss of less than 2\%. The hardware adopts a
row-wise dataflow with output-oriented data access to eliminate data
transposition, and supports dynamic operations with minimal area overhead.
Implemented in TSMC's 28nm CMOS technology, our design occupies 496.4K gates
and includes a 232KB SRAM buffer, achieving a peak throughput of 1024 GOPS at
1GHz, with an energy efficiency of 2.31 TOPS/W and an area efficiency of 858.61
GOPS/mm2.

### 4. [MX+: Pushing the Limits of Microscaling Formats for Efficient Large Language Model Serving](http://arxiv.org/pdf/2510.14557v1)

Authors: Jungi Lee, Junyong Park, Soohyun Cha, Jaehoon Cho, Jaewoong Sim

Reduced-precision data formats are crucial for cost-effective serving of
large language models (LLMs). While numerous reduced-precision formats have
been introduced thus far, they often require intrusive modifications to the
software frameworks or are rather unconventional for widespread adoption across
hardware vendors. In this paper, we instead focus on recent industry-driven
variants of block floating-point (BFP) formats and conduct a comprehensive
analysis to push their limits for efficient LLM serving. Our analysis shows
that existing ultra low-bit BFP variants struggle to provide reasonable
language model performance due to outlier values in blocks. To address the
outliers with BFPs, we propose MX+, a cost-effective and non-intrusive
extension designed for seamless integration into the microscaling (MX) formats.
MX+ builds on the key insight that the outlier does not need to use its
exponent field in the element data type, which allows us to repurpose the
exponent field as an extended mantissa to increase the precision of the outlier
element. Our evaluation shows that MX+ achieves significantly higher model
performance compared to the 4-bit MX format (MXFP4) with negligible storage
overhead and slowdown, thus offering a compelling alternative to MXFP4 or MXFP6
for efficient LLM inference.

### 5. [Deadlock-free routing for Full-mesh networks without using Virtual Channels](http://arxiv.org/pdf/2510.14730v1)

Authors: Alejandro Cano, Cristóbal Camarero, Carmen Martínez, Ramón Beivide

High-radix, low-diameter networks like HyperX and Dragonfly use a Full-mesh
core, and rely on multiple virtual channels (VCs) to avoid packet deadlocks in
adaptive routing. However, VCs introduce significant overhead in the switch in
terms of area, power, and design complexity, limiting the switch scalability.
This paper starts by revisiting VC-less routing through link ordering schemes
in Full-mesh networks, which offer implementation simplicity but suffer from
performance degradation under adversarial traffic. Thus, to overcome these
challenges, we propose TERA (Topology-Embedded Routing Algorithm), a novel
routing algorithm which employs an embedded physical subnetwork to provide
deadlock-free non-minimal paths without using VCs.
  In a Full-mesh network, TERA outperforms link ordering routing algorithms by
80% when dealing with adversarial traffic, and up to 100% in application
kernels. Furthermore, compared to other VC-based approaches, it reduces buffer
requirements by 50%, while maintaining comparable latency and throughput.
Lastly, early results from a 2D-HyperX evaluation show that TERA outperforms
state-of-the-art algorithms that use the same number of VCs, achieving
performance improvements of up to 32%.

### 6. [ColumnDisturb: Understanding Column-based Read Disturbance in Real DRAM Chips and Implications for Future Systems](http://arxiv.org/pdf/2510.14750v1)

Authors: İsmail Emir Yüksel, Ataberk Olgun, F. Nisa Bostancı, Haocong Luo, A. Giray Yağlıkçı, Onur Mutlu

We experimentally demonstrate a new widespread read disturbance phenomenon,
ColumnDisturb, in real commodity DRAM chips. By repeatedly opening or keeping a
DRAM row (aggressor row) open, we show that it is possible to disturb DRAM
cells through a DRAM column (i.e., bitline) and induce bitflips in DRAM cells
sharing the same columns as the aggressor row (across multiple DRAM subarrays).
With ColumnDisturb, the activation of a single row concurrently disturbs cells
across as many as three subarrays (e.g., 3072 rows) as opposed to
RowHammer/RowPress, which affect only a few neighboring rows of the aggressor
row in a single subarray. We rigorously characterize ColumnDisturb and its
characteristics under various operational conditions using 216 DDR4 and 4 HBM2
chips from three major manufacturers. Among our 27 key experimental
observations, we highlight two major results and their implications.
  First, ColumnDisturb affects chips from all three major manufacturers and
worsens as DRAM technology scales down to smaller node sizes (e.g., the minimum
time to induce the first ColumnDisturb bitflip reduces by up to 5.06x). We
observe that, in existing DRAM chips, ColumnDisturb induces bitflips within a
standard DDR4 refresh window (e.g., in 63.6 ms) in multiple cells. We predict
that, as DRAM technology node size reduces, ColumnDisturb would worsen in
future DRAM chips, likely causing many more bitflips in the standard refresh
window. Second, ColumnDisturb induces bitflips in many (up to 198x) more rows
than retention failures. Therefore, ColumnDisturb has strong implications for
retention-aware refresh mechanisms that leverage the heterogeneity in cell
retention times: our detailed analyses show that ColumnDisturb greatly reduces
the benefits of such mechanisms.

### 7. [Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References](http://arxiv.org/pdf/2510.14719v1)

Authors: Hongzheng Chen, Bin Fan, Alexander Collins, Bastian Hagedorn, Evghenii Gaburov, Masahiro Masuda, Matthew Brookhart, Chris Sullivan, Jason Knight, Zhiru Zhang, Vinod Grover

Modern GPUs feature specialized hardware units that enable high-performance,
asynchronous dataflow execution. However, the conventional SIMT programming
model is fundamentally misaligned with this task-parallel hardware, creating a
significant programmability gap. While hardware-level warp specialization is
the key to unlocking peak performance, it forces developers to manually
orchestrate complex, low-level communication and software pipelines--a process
that is labor-intensive, error-prone, and unsustainable. To address this
challenge, we present Tawa, an automated compiler that systematically generates
high-performance, warp-specialized code from a high-level, tile-based program.
Central to our approach is a novel IR abstraction, asynchronous references
(aref), which expresses warp-level communication without exposing low-level
hardware details. Using this abstraction, Tawa automatically partitions
programs into producer-consumer roles and manages the intricate dataflow
pipeline, relieving developers of invasive kernel rewriting. Evaluation on
NVIDIA H100 GPUs across representative LLM kernels shows that Tawa delivers
high hardware utilization, achieving up to 1.1$\times$ speedup over highly
optimized cuBLAS GEMM kernels. For attention workloads, Tawa attains
1.2$\times$ speedup over Triton and matches the performance of the
hand-optimized CUTLASS C++ FlashAttention-3 kernel with far less programming
effort.

### 8. [From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR](http://arxiv.org/pdf/2510.14871v1)

Authors: Erwei Wang, Samuel Bayliss, Andra Bisca, Zachary Blair, Sangeeta Chowdhary, Kristof Denolf, Jeff Fifield, Brandon Freiberger, Erika Hunhoff, Phil James-Roxby, Jack Lo, Joseph Melber, Stephen Neuendorffer, Eddie Richter, Andre Rosti, Javier Setoain, Gagandeep Singh, Endri Taka, Pranathi Vasireddy, Zhewen Yu, Niansong Zhang, Jinming Zhuang

General-purpose compilers abstract away parallelism, locality, and
synchronization, limiting their effectiveness on modern spatial architectures.
As modern computing architectures increasingly rely on fine-grained control
over data movement, execution order, and compute placement for performance,
compiler infrastructure must provide explicit mechanisms for orchestrating
compute and data to fully exploit such architectures. We introduce MLIR-AIR, a
novel, open-source compiler stack built on MLIR that bridges the semantic gap
between high-level workloads and fine-grained spatial architectures such as
AMD's NPUs. MLIR-AIR defines the AIR dialect, which provides structured
representations for asynchronous and hierarchical operations across compute and
memory resources. AIR primitives allow the compiler to orchestrate spatial
scheduling, distribute computation across hardware regions, and overlap
communication with computation without relying on ad hoc runtime coordination
or manual scheduling. We demonstrate MLIR-AIR's capabilities through two case
studies: matrix multiplication and the multi-head attention block from the
LLaMA 2 model. For matrix multiplication, MLIR-AIR achieves up to 78.7% compute
efficiency and generates implementations with performance almost identical to
state-of-the-art, hand-optimized matrix multiplication written using the
lower-level, close-to-metal MLIR-AIE framework. For multi-head attention, we
demonstrate that the AIR interface supports fused implementations using
approximately 150 lines of code, enabling tractable expression of complex
workloads with efficient mapping to spatial hardware. MLIR-AIR transforms
high-level structured control flow into spatial programs that efficiently
utilize the compute fabric and memory hierarchy of an NPU, leveraging
asynchronous execution, tiling, and communication overlap through
compiler-managed scheduling.

### Computational Complexity

### 1. [Decoding Balanced Linear Codes With Preprocessing](http://arxiv.org/pdf/2510.14347v1)

Authors: Andrej Bogdanov, Rohit Chatterjee, Yunqi Li, Prashant Nalini Vasudevan

Prange's information set algorithm is a decoding algorithm for arbitrary
linear codes. It decodes corrupted codewords of any $\mathbb{F}_2$-linear code
$C$ of message length $n$ up to relative error rate $O(\log n / n)$ in
$\mathsf{poly}(n)$ time. We show that the error rate can be improved to
$O((\log n)^2 / n)$, provided: (1) the decoder has access to a
polynomial-length advice string that depends on $C$ only, and (2) $C$ is
$n^{-\Omega(1)}$-balanced.
  As a consequence we improve the error tolerance in decoding random linear
codes if inefficient preprocessing of the code is allowed. This reveals
potential vulnerabilities in cryptographic applications of Learning Noisy
Parities with low noise rate.
  Our main technical result is that the Hamming weight of $Hw$, where $H$ is a
random sample of *short dual* codewords, measures the proximity of a word $w$
to the code in the regime of interest. Given such $H$ as advice, our algorithm
corrects errors by locally minimizing this measure. We show that for most
codes, the error rate tolerated by our decoder is asymptotically optimal among
all algorithms whose decision is based on thresholding $Hw$ for an arbitrary
polynomial-size advice matrix $H$.

### Computational Engineering

### 1. [AlphaQuanter: An End-to-End Tool-Orchestrated Agentic Reinforcement Learning Framework for Stock Trading](http://arxiv.org/pdf/2510.14264v1)

Authors: Zheye Deng, Jiashu Wang

While Large Language Model (LLM) agents show promise in automated trading,
they still face critical limitations. Prominent multi-agent frameworks often
suffer from inefficiency, produce inconsistent signals, and lack the end-to-end
optimization required to learn a coherent strategy from market feedback. To
address this, we introduce AlphaQuanter, a single-agent framework that uses
reinforcement learning (RL) to learn a dynamic policy over a transparent,
tool-augmented decision workflow, which empowers a single agent to autonomously
orchestrate tools and proactively acquire information on demand, establishing a
transparent and auditable reasoning process. Extensive experiments demonstrate
that AlphaQuanter achieves state-of-the-art performance on key financial
metrics. Moreover, its interpretable reasoning reveals sophisticated
strategies, offering novel and valuable insights for human traders. Our code
for data acquisition and agent training is publicly available at:
https://github.com/AlphaQuanter/AlphaQuanter

### 2. [A Structured Neural ODE Approach for Real Time Evaluation of AC Losses in 3D Superconducting Tapes](http://arxiv.org/pdf/2510.14487v1)

Authors: Riccardo Basei, Francesco Pase, Francesco Lucchini, Francesco Toso, Riccardo Torchio

Efficient modeling of High Temperature Superconductors (HTS) is crucial for
real-time quench monitoring; however, full-order electromagnetic simulations
remain prohibitively costly due to the strong nonlinearities. Conventional
reduced-order methods, such as the Proper Orthogonal Decomposition (POD) and
Discrete Empirical Interpolation Method (DEIM), alleviate this cost but are
limited by intrusive implementation and by the need for many interpolation
points. This work investigates reduced-order strategies for Integral Equation
Method (IEM) of HTS systems. We present the first application of POD-DEIM to
IEM-based HTS models, and introduce a Structured Neural Ordinary Differential
Equation (Neural ODE) approach that learns nonlinear dynamics directly in the
reduced space. Benchmark results show that the Neural ODE outperforms POD-DEIM
in both efficiency and accuracy, highlighting its potential for real-time
superconducting simulations.

### 3. [MetaBench: A Multi-task Benchmark for Assessing LLMs in Metabolomics](http://arxiv.org/pdf/2510.14944v1)

Authors: Yuxing Lu, Xukai Zhao, J. Ben Tamo, Micky C. Nnamdi, Rui Peng, Shuang Zeng, Xingyu Hu, Jinzhuo Wang, May D. Wang

Large Language Models (LLMs) have demonstrated remarkable capabilities on
general text; however, their proficiency in specialized scientific domains that
require deep, interconnected knowledge remains largely uncharacterized.
Metabolomics presents unique challenges with its complex biochemical pathways,
heterogeneous identifier systems, and fragmented databases. To systematically
evaluate LLM capabilities in this domain, we introduce MetaBench, the first
benchmark for metabolomics assessment. Curated from authoritative public
resources, MetaBench evaluates five capabilities essential for metabolomics
research: knowledge, understanding, grounding, reasoning, and research. Our
evaluation of 25 open- and closed-source LLMs reveals distinct performance
patterns across metabolomics tasks: while models perform well on text
generation tasks, cross-database identifier grounding remains challenging even
with retrieval augmentation. Model performance also decreases on long-tail
metabolites with sparse annotations. With MetaBench, we provide essential
infrastructure for developing and evaluating metabolomics AI systems, enabling
systematic progress toward reliable computational tools for metabolomics
research.

### Computational Geometry

### 1. [Excluding $K_{2,t}$ as a fat minor](http://arxiv.org/pdf/2510.14644v1)

Authors: Sandra Albrechtsen, Marc Distel, Agelos Georgakopoulos

We prove that for every $t \in \mathbb{N}$, the graph $K_{2,t}$ satisfies the
fat minor conjecture of Georgakopoulos and Papasoglu: for every $K\in
\mathbb{N}$ there exist $M,A\in \mathbb{N}$ such that every graph with no
$K$-fat $K_{2,t}$ minor is $(M,A)$-quasi-isometric to a graph with no $K_{2,t}$
minor. We use this to obtain an efficient algorithm for approximating the
minimal multiplicative distortion of any embedding of a finite graph into a
$K_{2,t}$-minor-free graph, answering a question of Chepoi, Dragan, Newman,
Rabinovich, and Vax\`es from 2012.

### Computation and Language

### 1. [RLSR: Reinforcement Learning with Supervised Reward Outperforms SFT in Instruction Following](http://arxiv.org/pdf/2510.14200v1)

Authors: Zhichao Wang, Andy Wong, Ruslan Belkin

After the pretraining stage of LLMs, techniques such as SFT, RLHF, RLVR, and
RFT are applied to enhance instruction-following ability, mitigate undesired
responses, improve reasoning capability and enable efficient domain adaptation
with minimal data. SFT relies on the next-token prediction objective to
strengthen instruction following in a base model using a large corpus of
human-labeled responses. In contrast, RFT employs a RL-based approach to adapt
fine-tuned reasoning models to specific domains with limited supervision.
Inspired by RFT, we propose replacing SFT with RLSR to leverage the extensive
SFT dataset in an RL framework, thereby improving the base model's
instruction-following ability. In RLSR, the base model generates multiple
responses for each prompt, and reward scores are computed as the cosine
similarity in the semantic embedding space between the generated and
human-labeled responses. RLSR can be utilized in multiple ways. It can directly
replace SFT, achieving superior performance on instruction-following
benchmarks-for example, RLSR (SB) on Qwen-7B (INFINITY) achieved an AlpacaEval
win rate of 26.34%, surpassing SFT's 21.01%. Furthermore, combining SFT and
RLSR further enhances downstream task performance; Qwen-7B (INFINITY) achieved
a win rate of 30.73% when trained with SFT + RLSR.

### 2. [MoM: Mixtures of Scenario-Aware Document Memories for Retrieval-Augmented Generation Systems](http://arxiv.org/pdf/2510.14252v1)

Authors: Jihao Zhao, Zhiyuan Ji, Simin Niu, Hanyu Wang, Feiyu Xiong, Zhiyu Li

The traditional RAG paradigm, which typically engages in the comprehension of
relevant text chunks in response to received queries, inherently restricts both
the depth of knowledge internalization and reasoning capabilities. To address
this limitation, our research transforms the text processing in RAG from
passive chunking to proactive understanding, defining this process as document
memory extraction with the objective of simulating human cognitive processes
during reading. Building upon this, we propose the Mixtures of scenario-aware
document Memories (MoM) framework, engineered to efficiently handle documents
from multiple domains and train small language models (SLMs) to acquire the
ability to proactively explore and construct document memories. The MoM
initially instructs large language models (LLMs) to simulate domain experts in
generating document logical outlines, thereby directing structured chunking and
core content extraction. It employs a multi-path sampling and multi-perspective
evaluation mechanism, specifically designing comprehensive metrics that
represent chunk clarity and extraction completeness to select the optimal
document memories. Additionally, to infuse deeper human-like reading abilities
during the training of SLMs, we incorporate a reverse reasoning strategy, which
deduces refined expert thinking paths from high-quality outcomes. Finally,
leveraging diverse forms of content generated by MoM, we develop a three-layer
document memory retrieval mechanism, which is grounded in our theoretical proof
from the perspective of probabilistic modeling. Extensive experimental results
across three distinct domains demonstrate that the MoM framework not only
resolves text chunking challenges in existing RAG systems, providing LLMs with
semantically complete document memories, but also paves the way for SLMs to
achieve human-centric intelligent text processing.

### 3. [Rewriting History: A Recipe for Interventional Analyses to Study Data Effects on Model Behavior](http://arxiv.org/pdf/2510.14261v1)

Authors: Rahul Nadkarni, Yanai Elazar, Hila Gonen, Noah A. Smith

We present an experimental recipe for studying the relationship between
training data and language model (LM) behavior. We outline steps for
intervening on data batches -- i.e., ``rewriting history'' -- and then
retraining model checkpoints over that data to test hypotheses relating data to
behavior. Our recipe breaks down such an intervention into stages that include
selecting evaluation items from a benchmark that measures model behavior,
matching relevant documents to those items, and modifying those documents
before retraining and measuring the effects. We demonstrate the utility of our
recipe through case studies on factual knowledge acquisition in LMs, using both
cooccurrence statistics and information retrieval methods to identify documents
that might contribute to knowledge learning. Our results supplement past
observational analyses that link cooccurrence to model behavior, while
demonstrating that extant methods for identifying relevant training documents
do not fully explain an LM's ability to correctly answer knowledge questions.
Overall, we outline a recipe that researchers can follow to test further
hypotheses about how training data affects model behavior. Our code is made
publicly available to promote future work.

### 4. [Retrofitting Small Multilingual Models for Retrieval: Matching 7B Performance with 300M Parameters](http://arxiv.org/pdf/2510.14274v1)

Authors: Lifu Tu, Yingbo Zhou, Semih Yavuz

Training effective multilingual embedding models presents unique challenges
due to the diversity of languages and task objectives. Although small
multilingual models (<1 B parameters) perform well on multilingual tasks
generally, they consistently lag behind larger models (>1 B) in the most
prevalent use case: retrieval. This raises a critical question: Can smaller
models be retrofitted specifically for retrieval tasks to enhance their
performance? In this work, we investigate key factors that influence the
effectiveness of multilingual embeddings, focusing on training data scale,
negative sampling strategies, and data diversity. We find that while increasing
the scale of training data yields initial performance gains, these improvements
quickly plateau - indicating diminishing returns. Incorporating hard negatives
proves essential for consistently improving retrieval accuracy. Furthermore,
our analysis reveals that task diversity in the training data contributes more
significantly to performance than language diversity alone. As a result, we
develop a compact (approximately 300M) multilingual model that achieves
retrieval performance comparable to or even surpassing current strong 7B
models.

### 5. [Qwen3Guard Technical Report](http://arxiv.org/pdf/2510.14276v1)

Authors: Haiquan Zhao, Chenhan Yuan, Fei Huang, Xiaomeng Hu, Yichang Zhang, An Yang, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin, Baosong Yang, Chen Cheng, Jialong Tang, Jiandong Jiang, Jianwei Zhang, Jijie Xu, Ming Yan, Minmin Sun, Pei Zhang, Pengjun Xie, Qiaoyu Tang, Qin Zhu, Rong Zhang, Shibin Wu, Shuo Zhang, Tao He, Tianyi Tang, Tingyu Xia, Wei Liao, Weizhou Shen, Wenbiao Yin, Wenmeng Zhou, Wenyuan Yu, Xiaobin Wang, Xiaodong Deng, Xiaodong Xu, Xinyu Zhang, Yang Liu, Yeqiu Li, Yi Zhang, Yong Jiang, Yu Wan, Yuxin Zhou

As large language models (LLMs) become more capable and widely used, ensuring
the safety of their outputs is increasingly critical. Existing guardrail
models, though useful in static evaluation settings, face two major limitations
in real-world applications: (1) they typically output only binary "safe/unsafe"
labels, which can be interpreted inconsistently across diverse safety policies,
rendering them incapable of accommodating varying safety tolerances across
domains; and (2) they require complete model outputs before performing safety
checks, making them fundamentally incompatible with streaming LLM inference,
thereby preventing timely intervention during generation and increasing
exposure to harmful partial outputs. To address these challenges, we present
Qwen3Guard, a series of multilingual safety guardrail models with two
specialized variants: Generative Qwen3Guard, which casts safety classification
as an instruction-following task to enable fine-grained tri-class judgments
(safe, controversial, unsafe); and Stream Qwen3Guard, which introduces a
token-level classification head for real-time safety monitoring during
incremental text generation. Both variants are available in three sizes (0.6B,
4B, and 8B parameters) and support up to 119 languages and dialects, providing
comprehensive, scalable, and low-latency safety moderation for global LLM
deployments. Evaluated across English, Chinese, and multilingual benchmarks,
Qwen3Guard achieves state-of-the-art performance in both prompt and response
safety classification. All models are released under the Apache 2.0 license for
public use.

### 6. [MathMist: A Parallel Multilingual Benchmark Dataset for Mathematical Problem Solving and Reasoning](http://arxiv.org/pdf/2510.14305v1)

Authors: Mahbub E Sobhani, Md. Faiyaz Abdullah Sayeedi, Tasnim Mohiuddin, Md Mofijul Islam, Swakkhar Shatabda

Mathematical reasoning remains one of the most challenging domains for large
language models (LLMs), requiring not only linguistic understanding but also
structured logical deduction and numerical precision. While recent LLMs
demonstrate strong general-purpose reasoning abilities, their mathematical
competence across diverse languages remains underexplored. Existing benchmarks
primarily focus on English or a narrow subset of high-resource languages,
leaving significant gaps in assessing multilingual and cross-lingual
mathematical reasoning. To address this, we introduce MathMist, a parallel
multilingual benchmark for mathematical problem solving and reasoning. MathMist
encompasses over 21K aligned question-answer pairs across seven languages,
representing a balanced coverage of high-, medium-, and low-resource linguistic
settings. The dataset captures linguistic variety, multiple types of problem
settings, and solution synthesizing capabilities. We systematically evaluate a
diverse suite of models, including open-source small and medium LLMs,
proprietary systems, and multilingual-reasoning-focused models, under
zero-shot, chain-of-thought (CoT), and code-switched reasoning paradigms. Our
results reveal persistent deficiencies in LLMs' ability to perform consistent
and interpretable mathematical reasoning across languages, with pronounced
degradation in low-resource settings. All the codes and data are available at
GitHub: https://github.com/mahbubhimel/MathMist

### 7. [On the Ability of LLMs to Handle Character-Level Perturbations: How Well and How?](http://arxiv.org/pdf/2510.14365v1)

Authors: Anyun Zhuo, Xuefei Ning, Ningyuan Li, Yu Wang, Pinyan Lu

This work investigates the resilience of contemporary LLMs against frequent
and structured character-level perturbations, specifically through the
insertion of noisy characters after each input character. We introduce
\nameshort{}, a practical method that inserts invisible Unicode control
characters into text to discourage LLM misuse in scenarios such as online exam
systems. Surprisingly, despite strong obfuscation that fragments tokenization
and reduces the signal-to-noise ratio significantly, many LLMs still maintain
notable performance. Through comprehensive evaluation across model-, problem-,
and noise-related configurations, we examine the extent and mechanisms of this
robustness, exploring both the handling of character-level tokenization and
\textit{implicit} versus \textit{explicit} denoising mechanism hypotheses of
character-level noises. We hope our findings on the low-level robustness of
LLMs will shed light on the risks of their misuse and on the reliability of
deploying LLMs across diverse applications.

### 8. [Suicidal Comment Tree Dataset: Enhancing Risk Assessment and Prediction Through Contextual Analysis](http://arxiv.org/pdf/2510.14395v1)

Authors: Jun Li, Qun Zhao

Suicide remains a critical global public health issue. While previous studies
have provided valuable insights into detecting suicidal expressions in
individual social media posts, limited attention has been paid to the analysis
of longitudinal, sequential comment trees for predicting a user's evolving
suicidal risk. Users, however, often reveal their intentions through historical
posts and interactive comments over time. This study addresses this gap by
investigating how the information in comment trees affects both the
discrimination and prediction of users' suicidal risk levels. We constructed a
high-quality annotated dataset, sourced from Reddit, which incorporates users'
posting history and comments, using a refined four-label annotation framework
based on the Columbia Suicide Severity Rating Scale (C-SSRS). Statistical
analysis of the dataset, along with experimental results from Large Language
Models (LLMs) experiments, demonstrates that incorporating comment trees data
significantly enhances the discrimination and prediction of user suicidal risk
levels. This research offers a novel insight to enhancing the detection
accuracy of at-risk individuals, thereby providing a valuable foundation for
early suicide intervention strategies.

### 9. [Your Next Token Prediction: A Multilingual Benchmark for Personalized Response Generation](http://arxiv.org/pdf/2510.14398v1)

Authors: Shiyao Ding, Takayuki Ito

Large language models (LLMs) excel at general next-token prediction but still
struggle to generate responses that reflect how individuals truly communicate,
such as replying to emails or social messages in their own style. However, real
SNS or email histories are difficult to collect due to privacy concerns. To
address this, we propose the task of "Your Next Token Prediction (YNTP)", which
models a user's precise word choices through controlled human-agent
conversations. We build a multilingual benchmark of 100 dialogue sessions
across English, Japanese, and Chinese, where users interact for five days with
psychologically grounded NPCs based on MBTI dimensions. This setup captures
natural, daily-life communication patterns and enables analysis of users'
internal models. We evaluate prompt-based and fine-tuning-based personalization
methods, establishing the first benchmark for YNTP and a foundation for
user-aligned language modeling. The dataset is available at:
https://github.com/AnonymousHub4Submissions/your-next-token-prediction-dataset-100

### 10. [Explore to Evolve: Scaling Evolved Aggregation Logic via Proactive Online Exploration for Deep Research Agents](http://arxiv.org/pdf/2510.14438v1)

Authors: Rui Wang, Ce Zhang, Jun-Yu Ma, Jianshu Zhang, Hongru Wang, Yi Chen, Boyang Xue, Tianqing Fang, Zhisong Zhang, Hongming Zhang, Haitao Mi, Dong Yu, Kam-Fai Wong

Deep research web agents not only retrieve information from diverse sources
such as web environments, files, and multimodal inputs, but more importantly,
they need to rigorously analyze and aggregate knowledge for insightful
research. However, existing open-source deep research agents predominantly
focus on enhancing information-seeking capabilities of web agents to locate
specific information, while overlooking the essential need for information
aggregation, which would limit their ability to support in-depth research. We
propose an Explore to Evolve paradigm to scalably construct verifiable training
data for web agents. Begins with proactive online exploration, an agent sources
grounded information by exploring the real web. Using the collected evidence,
the agent then self-evolves an aggregation program by selecting, composing, and
refining operations from 12 high-level logical types to synthesize a verifiable
QA pair. This evolution from high-level guidance to concrete operations allowed
us to scalably produce WebAggregatorQA, a dataset of 10K samples across 50K
websites and 11 domains. Based on an open-source agent framework, SmolAgents,
we collect supervised fine-tuning trajectories to develop a series of
foundation models, WebAggregator. WebAggregator-8B matches the performance of
GPT-4.1, while the 32B variant surpasses GPT-4.1 by more than 10% on GAIA-text
and closely approaches Claude-3.7-sonnet. Moreover, given the limited
availability of benchmarks that evaluate web agents' information aggregation
abilities, we construct a human-annotated evaluation split of WebAggregatorQA
as a challenging test set. On this benchmark, Claude-3.7-sonnet only achieves
28%, and GPT-4.1 scores 25.8%. Even when agents manage to retrieve all
references, they still struggle on WebAggregatorQA, highlighting the need to
strengthen the information aggregation capabilities of web agent foundations.

### Cryptography and Security

### 1. [Power Grid Cybersecurity: Policy Analysis White Paper](http://arxiv.org/pdf/2510.14171v1)

Authors: Jack Vanlyssel

The U.S. power grid underpins national security, public safety, and economic
stability, but faces growing cyber risks from vulnerabilities in industrial
control systems, remote access, and poor cyber hygiene. Despite its critical
importance, current policy remains fragmented and reactive. This paper proposes
a dual policy approach to strengthen grid cybersecurity: enhanced information
sharing between government and private utilities to improve threat detection
and response, and standardized cyber hygiene practices to reduce common attack
vectors. For long-term resilience, a Unified National Cybersecurity Framework
is recommended to align existing NERC, IEC, IEEE, and NIST standards, eliminate
regulatory overlap, and adapt to evolving threats. Together, these policies
offer both immediate and sustainable improvements in safeguarding the nation's
most vital infrastructure.

### 2. [Securing U.S. Critical Infrastructure: Lessons from Stuxnet and the Ukraine Power Grid Attacks](http://arxiv.org/pdf/2510.14185v1)

Authors: Jack Vanlyssel

Industrial Control Systems (ICS) underpin the United States' critical
infrastructure, managing essential services such as power, water, and
transportation that are vital to national security and public safety. However,
increasing digital integration has exposed these systems to escalating cyber
threats. Historical attacks like Stuxnet and the Ukraine power grid incident
revealed exploitable weaknesses-poor network segmentation, outdated software,
weak authentication, and inadequate monitoring-that persist in many U.S. ICS
environments today. This paper analyzes these landmark attacks to identify
recurring vulnerabilities and assess their relevance to current U.S.
infrastructure. It argues that without immediate reforms, similar exploits
could lead to catastrophic disruptions and national security crises. To address
these risks, the paper proposes policy measures focused on implementing
zero-trust architecture and improved network segmentation to enhance system
resilience. These recommendations aim to guide policymakers and industry
leaders in securing the nation's most critical operational technologies against
future cyber threats.

### 3. [Infrastructure Patterns in Toll Scam Domains: A Comprehensive Analysis of Cybercriminal Registration and Hosting Strategies](http://arxiv.org/pdf/2510.14198v1)

Authors: Morium Akter Munny, Mahbub Alam, Sonjoy Kumar Paul, Daniel Timko, Muhammad Lutfor Rahman, Nitesh Saxena

Toll scams involve criminals registering fake domains that pretend to be
legitimate transportation agencies to trick users into making fraudulent
payments. Although these scams are rapidly increasing and causing significant
harm, they have not been extensively studied. We present the first large-scale
analysis of toll scam domains, using a newly created dataset of 67,907
confirmed scam domains mostly registered in 2025. Our study reveals that
attackers exploit permissive registrars and less common top-level domains, with
86.9% of domains concentrated in just five non-mainstream TLDs and 72.9%
registered via a single provider. We also discover specific registration
patterns, including short bursts of activity that suggest automated,
coordinated attacks, with over half of domains registered in the first quarter
of 2025. This extreme temporal clustering reflects highly synchronized campaign
launches. Additionally, we build a simple predictive model using only domain
registration data to predict which scam domains are likely to be suspended -- a
proxy for confirmed abuse -- achieving 80.4% accuracy, and 92.3% sensitivity.
Our analysis reveals attacker strategies for evading detection -- such as
exploiting obscure TLDs, permissive registrars, and coordinated registration
bursts -- which can inform more targeted interventions by registrars, hosting
providers, and security platforms. However, our results suggest that
registration metadata alone may be insufficient, and incorporating features
from domain URLs and webpage content could further improve detection.

### 4. [An Information Asymmetry Game for Trigger-based DNN Model Watermarking](http://arxiv.org/pdf/2510.14218v1)

Authors: Chaoyue Huang, Gejian Zhao, Hanzhou Wu, Zhihua Xia, Asad Malik

As a valuable digital product, deep neural networks (DNNs) face increasingly
severe threats to the intellectual property, making it necessary to develop
effective technical measures to protect them. Trigger-based watermarking
methods achieve copyright protection by embedding triggers into the host DNNs.
However, the attacker may remove the watermark by pruning or fine-tuning. We
model this interaction as a game under conditions of information asymmetry,
namely, the defender embeds a secret watermark with private knowledge, while
the attacker can only access the watermarked model and seek removal. We define
strategies, costs, and utilities for both players, derive the attacker's
optimal pruning budget, and establish an exponential lower bound on the
accuracy of watermark detection after attack. Experimental results demonstrate
the feasibility of the watermarked model, and indicate that sparse watermarking
can resist removal with negligible accuracy loss. This study highlights the
effectiveness of game-theoretic analysis in guiding the design of robust
watermarking schemes for model copyright protection.

### 5. [RHINO: Guided Reasoning for Mapping Network Logs to Adversarial Tactics and Techniques with Large Language Models](http://arxiv.org/pdf/2510.14233v1)

Authors: Fanchao Meng, Jiaping Gui, Yunbo Li, Yue Wu

Modern Network Intrusion Detection Systems generate vast volumes of low-level
alerts, yet these outputs remain semantically fragmented, requiring
labor-intensive manual correlation with high-level adversarial behaviors.
Existing solutions for automating this mapping-rule-based systems and machine
learning classifiers-suffer from critical limitations: rule-based approaches
fail to adapt to novel attack variations, while machine learning methods lack
contextual awareness and treat tactic-technique mapping as a syntactic matching
problem rather than a reasoning task. Although Large Language Models have shown
promise in cybersecurity tasks, preliminary experiments reveal that existing
LLM-based methods frequently hallucinate technique names or produce
decontextualized mappings due to their single-step classification approach.
  To address these challenges, we introduce RHINO, a novel framework that
decomposes LLM-based attack analysis into three interpretable phases mirroring
human reasoning: (1) behavioral abstraction, where raw logs are translated into
contextualized narratives; (2) multi-role collaborative inference, generating
candidate techniques by evaluating behavioral evidence against MITRE ATT&CK
knowledge; and (3) validation, cross-referencing predictions with official
MITRE definitions to rectify hallucinations. RHINO bridges the semantic gap
between low-level observations and adversarial intent while improving output
reliability through structured reasoning.
  We evaluate RHINO on three benchmarks across four backbone models. RHINO
achieved high accuracy, with model performance ranging from 86.38% to 88.45%,
resulting in relative gains from 24.25% to 76.50% across different models. Our
results demonstrate that RHINO significantly enhances the interpretability and
scalability of threat analysis, offering a blueprint for deploying LLMs in
operational security settings.

### 6. [Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration](http://arxiv.org/pdf/2510.14522v1)

Authors: Evangelos Lamprou, Julian Dai, Grigoris Ntousakis, Martin C. Rinard, Nikos Vasilakis

Software supply-chain attacks are an important and ongoing concern in the
open source software ecosystem. These attacks maintain the standard
functionality that a component implements, but additionally hide malicious
functionality activated only when the component reaches its target environment.
Lexo addresses such stealthy attacks by automatically learning and regenerating
vulnerability-free versions of potentially malicious components. Lexo first
generates a set of input-output pairs to model a component's full observable
behavior, which it then uses to synthesize a new version of the original
component. The new component implements the original functionality but avoids
stealthy malicious behavior. Throughout this regeneration process, Lexo
consults several distinct instances of Large Language Models (LLMs), uses
correctness and coverage metrics to shepherd these instances, and guardrails
their results. Our evaluation on 100+ real-world packages, including high
profile stealthy supply-chain attacks, indicates that Lexo scales across
multiple domains, regenerates code efficiently (<100s on average), maintains
compatibility, and succeeds in eliminating malicious code in several real-world
supply-chain-attacks, even in cases when a state-of-the-art LLM fails to
eliminate malicious code when prompted to do so.

### 7. [Symbolic verification of Apple's Find My location-tracking protocol](http://arxiv.org/pdf/2510.14589v1)

Authors: Vaishnavi Sundararajan, Rithwik

Tracking devices, while designed to help users find their belongings in case
of loss/theft, bring in new questions about privacy and surveillance of not
just their own users, but in the case of crowd-sourced location tracking, even
that of others even orthogonally associated with these platforms. Apple's Find
My is perhaps the most ubiquitous such system which can even locate devices
which do not possess any cellular support or GPS, running on millions of
devices worldwide. Apple claims that this system is private and secure, but the
code is proprietary, and such claims have to be taken on faith. It is well
known that even with perfect cryptographic guarantees, logical flaws might
creep into protocols, and allow undesirable attacks. In this paper, we present
a symbolic model of the Find My protocol, as well as a precise formal
specification of desirable properties, and provide automated, machine-checkable
proofs of these properties in the Tamarin prover.

### 8. [Improving Cybercrime Detection and Digital Forensics Investigations with Artificial Intelligence](http://arxiv.org/pdf/2510.14638v1)

Authors: Silvia Lucia Sanna, Leonardo Regano, Davide Maiorca, Giorgio Giacinto

According to a recent EUROPOL report, cybercrime is still recurrent in
Europe, and different activities and countermeasures must be taken to limit,
prevent, detect, analyze, and fight it. Cybercrime must be prevented with
specific measures, tools, and techniques, for example through automated network
and malware analysis. Countermeasures against cybercrime can also be improved
with proper \df analysis in order to extract data from digital devices trying
to retrieve information on the cybercriminals. Indeed, results obtained through
a proper \df analysis can be leveraged to train cybercrime detection systems to
prevent the success of similar crimes. Nowadays, some systems have started to
adopt Artificial Intelligence (AI) algorithms for cyberattack detection and \df
analysis improvement. However, AI can be better applied as an additional
instrument in these systems to improve the detection and in the \df analysis.
For this reason, we highlight how cybercrime analysis and \df procedures can
take advantage of AI. On the other hand, cybercriminals can use these systems
to improve their skills, bypass automatic detection, and develop advanced
attack techniques. The case study we presented highlights how it is possible to
integrate the use of the three popular chatbots {\tt Gemini}, {\tt Copilot} and
{\tt chatGPT} to develop a Python code to encode and decoded images with
steganographic technique, even though their presence is not an indicator of
crime, attack or maliciousness but used by a cybercriminal as anti-forensics
technique.

### 9. [AEX-NStep: Probabilistic Interrupt Counting Attacks on Intel SGX](http://arxiv.org/pdf/2510.14675v1)

Authors: Nicolas Dutly, Friederike Groschupp, Ivan Puddu, Kari Kostiainen, Srdjan Capkun

To mitigate interrupt-based stepping attacks (notably using SGX-Step), Intel
introduced AEX-Notify, an ISA extension to Intel SGX that aims to prevent
deterministic single-stepping. In this work, we introduce AEX-NStep, the first
interrupt counting attack on AEX-Notify-enabled Enclaves. We show that
deterministic single-stepping is not required for interrupt counting attacks to
be practical and that, therefore, AEX-Notify does not entirely prevent such
attacks. We specifically show that one of AEX-Notify's security guarantees,
obfuscated forward progress, does not hold, and we introduce two new
probabilistic interrupt counting attacks. We use these attacks to construct a
practical ECDSA key leakage attack on an AEX-Notify-enabled SGX enclave. Our
results extend the original security analysis of AEX-Notify and inform the
design of future mitigations.

### 10. [FibRace: a large-scale benchmark of client-side proving on mobile devices](http://arxiv.org/pdf/2510.14693v1)

Authors: Simon Malatrait, Alex Sirac

FibRace, jointly developed by KKRT Labs and Hyli, was the first large-scale
experiment to test client-side proof generation on smartphones using Cairo M.
Presented as a mobile game in which players proved Fibonacci numbers and
climbed a leaderboard, FibRace served a dual purpose: to engage the public and
to provide empirical benchmarking. Over a three-week campaign (September 11-30,
2025), 6,047 players across 99 countries generated 2,195,488 proofs on 1,420
unique device models. The results show that most modern smartphones can
complete a proof in under 5 seconds, confirming that *mobile devices are now
capable of producing zero-knowledge proofs reliably*, without the need for
remote provers or specialized hardware. Performance was correlated primarily
with RAM capacity and SoC (System on Chip) performance: devices with at least 3
GB of RAM proved stably, when Apple's A19 Pro and M-series chips achieved the
fastest proving times. Hyli's blockchain natively verified every proof onchain
without congestion. FibRace provides the most comprehensive dataset to date on
mobile proving performance, establishing a practical baseline for future
research in lightweight provers, proof-powered infrastructure, and
privacy-preserving mobile applications.

### Computer Vision and Pattern Recognition

### 1. [LOTA: Bit-Planes Guided AI-Generated Image Detection](http://arxiv.org/pdf/2510.14230v1)

Authors: Hongsong Wang, Renxi Cheng, Yang Zhang, Chaolei Han, Jie Gui

The rapid advancement of GAN and Diffusion models makes it more difficult to
distinguish AI-generated images from real ones. Recent studies often use
image-based reconstruction errors as an important feature for determining
whether an image is AI-generated. However, these approaches typically incur
high computational costs and also fail to capture intrinsic noisy features
present in the raw images. To solve these problems, we innovatively refine
error extraction by using bit-plane-based image processing, as lower bit planes
indeed represent noise patterns in images. We introduce an effective bit-planes
guided noisy image generation and exploit various image normalization
strategies, including scaling and thresholding. Then, to amplify the noise
signal for easier AI-generated image detection, we design a maximum gradient
patch selection that applies multi-directional gradients to compute the noise
score and selects the region with the highest score. Finally, we propose a
lightweight and effective classification head and explore two different
structures: noise-based classifier and noise-guided classifier. Extensive
experiments on the GenImage benchmark demonstrate the outstanding performance
of our method, which achieves an average accuracy of \textbf{98.9\%}
(\textbf{11.9}\%~$\uparrow$) and shows excellent cross-generator generalization
capability. Particularly, our method achieves an accuracy of over 98.2\% from
GAN to Diffusion and over 99.2\% from Diffusion to GAN. Moreover, it performs
error extraction at the millisecond level, nearly a hundred times faster than
existing methods. The code is at https://github.com/hongsong-wang/LOTA.

### 2. [PIA: Deepfake Detection Using Phoneme-Temporal and Identity-Dynamic Analysis](http://arxiv.org/pdf/2510.14241v1)

Authors: Soumyya Kanti Datta, Tanvi Ranga, Chengzhe Sun, Siwei Lyu

The rise of manipulated media has made deepfakes a particularly insidious
threat, involving various generative manipulations such as lip-sync
modifications, face-swaps, and avatar-driven facial synthesis. Conventional
detection methods, which predominantly depend on manually designed
phoneme-viseme alignment thresholds, fundamental frame-level consistency
checks, or a unimodal detection strategy, inadequately identify modern-day
deepfakes generated by advanced generative models such as GANs, diffusion
models, and neural rendering techniques. These advanced techniques generate
nearly perfect individual frames yet inadvertently create minor temporal
discrepancies frequently overlooked by traditional detectors. We present a
novel multimodal audio-visual framework, Phoneme-Temporal and Identity-Dynamic
Analysis(PIA), incorporating language, dynamic face motion, and facial
identification cues to address these limitations. We utilize phoneme sequences,
lip geometry data, and advanced facial identity embeddings. This integrated
method significantly improves the detection of subtle deepfake alterations by
identifying inconsistencies across multiple complementary modalities. Code is
available at https://github.com/skrantidatta/PIA

### 3. [Event Interval Modulation: A Novel Scheme for Event-based Optical Camera Communication](http://arxiv.org/pdf/2510.14245v1)

Authors: Miu Sumino, Mayu Ishii, Shun Kaizu, Daisuke Hisano, Yu Nakayama

Optical camera communication (OCC) represents a promising visible light
communication technology. Nonetheless, typical OCC systems utilizing
frame-based cameras are encumbered by limitations, including low bit rate and
high processing load. To address these issues, OCC system utilizing an
event-based vision sensor (EVS) as receivers have been proposed. The EVS
enables high-speed, low-latency, and robust communication due to its
asynchronous operation and high dynamic range. In existing event-based OCC
systems, conventional modulation schemes such as on-off keying (OOK) and pulse
position modulation have been applied, however, to the best of our knowledge,
no modulation method has been proposed that fully exploits the unique
characteristics of the EVS. This paper proposes a novel modulation scheme,
called the event interval modulation (EIM) scheme, specifically designed for
event-based OCC. EIM enables improvement in transmission speed by modulating
information using the intervals between events. This paper proposes a
theoretical model of EIM and conducts a proof-of-concept experiment. First, the
parameters of the EVS are tuned and customized to optimize the frequency
response specifically for EIM. Then, the maximum modulation order usable in EIM
is determined experimentally. We conduct transmission experiments based on the
obtained parameters. Finally, we report successful transmission at 28 kbps over
10 meters and 8.4 kbps over 50 meters in an indoor environment. This sets a new
benchmark for bit rate in event-based OCC systems.

### 4. [MACE: Mixture-of-Experts Accelerated Coordinate Encoding for Large-Scale Scene Localization and Rendering](http://arxiv.org/pdf/2510.14251v1)

Authors: Mingkai Liu, Dikai Fan, Haohua Que, Haojia Gao, Xiao Liu, Shuxue Peng, Meixia Lin, Shengyu Gu, Ruicong Ye, Wanli Qiu, Handong Yao, Ruopeng Zhang, Xianliang Huang

Efficient localization and high-quality rendering in large-scale scenes
remain a significant challenge due to the computational cost involved. While
Scene Coordinate Regression (SCR) methods perform well in small-scale
localization, they are limited by the capacity of a single network when
extended to large-scale scenes. To address these challenges, we propose the
Mixed Expert-based Accelerated Coordinate Encoding method (MACE), which enables
efficient localization and high-quality rendering in large-scale scenes.
Inspired by the remarkable capabilities of MOE in large model domains, we
introduce a gating network to implicitly classify and select sub-networks,
ensuring that only a single sub-network is activated during each inference.
Furtheremore, we present Auxiliary-Loss-Free Load Balancing(ALF-LB) strategy to
enhance the localization accuracy on large-scale scene. Our framework provides
a significant reduction in costs while maintaining higher precision, offering
an efficient solution for large-scale scene applications. Additional
experiments on the Cambridge test set demonstrate that our method achieves
high-quality rendering results with merely 10 minutes of training.

### 5. [Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization](http://arxiv.org/pdf/2510.14255v1)

Authors: Liao Shen, Wentao Jiang, Yiran Zhu, Tiezheng Ge, Zhiguo Cao, Bo Zheng

Recent advances in image-to-video (I2V) generation have achieved remarkable
progress in synthesizing high-quality, temporally coherent videos from static
images. Among all the applications of I2V, human-centric video generation
includes a large portion. However, existing I2V models encounter difficulties
in maintaining identity consistency between the input human image and the
generated video, especially when the person in the video exhibits significant
expression changes and movements. This issue becomes critical when the human
face occupies merely a small fraction of the image. Since humans are highly
sensitive to identity variations, this poses a critical yet under-explored
challenge in I2V generation. In this paper, we propose Identity-Preserving
Reward-guided Optimization (IPRO), a novel video diffusion framework based on
reinforcement learning to enhance identity preservation. Instead of introducing
auxiliary modules or altering model architectures, our approach introduces a
direct and effective tuning algorithm that optimizes diffusion models using a
face identity scorer. To improve performance and accelerate convergence, our
method backpropagates the reward signal through the last steps of the sampling
chain, enabling richer gradient feedback. We also propose a novel facial
scoring mechanism that treats faces in ground-truth videos as facial feature
pools, providing multi-angle facial information to enhance generalization. A
KL-divergence regularization is further incorporated to stabilize training and
prevent overfitting to the reward signal. Extensive experiments on Wan 2.2 I2V
model and our in-house I2V model demonstrate the effectiveness of our method.
Our project and code are available at
\href{https://ipro-alimama.github.io/}{https://ipro-alimama.github.io/}.

### 6. [Identity-GRPO: Optimizing Multi-Human Identity-preserving Video Generation via Reinforcement Learning](http://arxiv.org/pdf/2510.14256v1)

Authors: Xiangyu Meng, Zixian Zhang, Zhenghao Zhang, Junchao Liao, Long Qin, Weizhi Wang

While advanced methods like VACE and Phantom have advanced video generation
for specific subjects in diverse scenarios, they struggle with multi-human
identity preservation in dynamic interactions, where consistent identities
across multiple characters are critical. To address this, we propose
Identity-GRPO, a human feedback-driven optimization pipeline for refining
multi-human identity-preserving video generation. First, we construct a video
reward model trained on a large-scale preference dataset containing
human-annotated and synthetic distortion data, with pairwise annotations
focused on maintaining human consistency throughout the video. We then employ a
GRPO variant tailored for multi-human consistency, which greatly enhances both
VACE and Phantom. Through extensive ablation studies, we evaluate the impact of
annotation quality and design choices on policy optimization. Experiments show
that Identity-GRPO achieves up to 18.9% improvement in human consistency
metrics over baseline methods, offering actionable insights for aligning
reinforcement learning with personalized video generation.

### 7. [MatchAttention: Matching the Relative Positions for High-Resolution Cross-View Matching](http://arxiv.org/pdf/2510.14260v1)

Authors: Tingman Yan, Tao Liu, Xilian Yang, Qunfei Zhao, Zeyang Xia

Cross-view matching is fundamentally achieved through cross-attention
mechanisms. However, matching of high-resolution images remains challenging due
to the quadratic complexity and lack of explicit matching constraints in the
existing cross-attention. This paper proposes an attention mechanism,
MatchAttention, that dynamically matches relative positions. The relative
position determines the attention sampling center of the key-value pairs given
a query. Continuous and differentiable sliding-window attention sampling is
achieved by the proposed BilinearSoftmax. The relative positions are
iteratively updated through residual connections across layers by embedding
them into the feature channels. Since the relative position is exactly the
learning target for cross-view matching, an efficient hierarchical cross-view
decoder, MatchDecoder, is designed with MatchAttention as its core component.
To handle cross-view occlusions, gated cross-MatchAttention and a
consistency-constrained loss are proposed. These two components collectively
mitigate the impact of occlusions in both forward and backward passes, allowing
the model to focus more on learning matching relationships. When applied to
stereo matching, MatchStereo-B ranked 1st in average error on the public
Middlebury benchmark and requires only 29ms for KITTI-resolution inference.
MatchStereo-T can process 4K UHD images in 0.1 seconds using only 3GB of GPU
memory. The proposed models also achieve state-of-the-art performance on KITTI
2012, KITTI 2015, ETH3D, and Spring flow datasets. The combination of high
accuracy and low computational complexity makes real-time, high-resolution, and
high-accuracy cross-view matching possible. Code is available at
https://github.com/TingmanYan/MatchAttention.

### 8. [Experimental Demonstration of Event-based Optical Camera Communication in Long-Range Outdoor Environment](http://arxiv.org/pdf/2510.14266v1)

Authors: Miu Sumino, Mayu Ishii, Shun Kaizu, Daisuke Hisano, Yu Nakayama

We propose a robust demodulation scheme for optical camera communication
systems using an event-based vision sensor, combining OOK with toggle
demodulation and a digital phase-locked loop. This is the first report to
achieve a $\mathrm{BER} < 10^{-3}$ at 200m-60kbps and 400m-30kbps in outdoor
experiments.

### 9. [CLEAR: Causal Learning Framework For Robust Histopathology Tumor Detection Under Out-Of-Distribution Shifts](http://arxiv.org/pdf/2510.14273v1)

Authors: Kieu-Anh Truong Thi, Huy-Hieu Pham, Duc-Trong Le

Domain shift in histopathology, often caused by differences in acquisition
processes or data sources, poses a major challenge to the generalization
ability of deep learning models. Existing methods primarily rely on modeling
statistical correlations by aligning feature distributions or introducing
statistical variation, yet they often overlook causal relationships. In this
work, we propose a novel causal-inference-based framework that leverages
semantic features while mitigating the impact of confounders. Our method
implements the front-door principle by designing transformation strategies that
explicitly incorporate mediators and observed tissue slides. We validate our
method on the CAMELYON17 dataset and a private histopathology dataset,
demonstrating consistent performance gains across unseen domains. As a result,
our approach achieved up to a 7% improvement in both the CAMELYON17 dataset and
the private histopathology dataset, outperforming existing baselines. These
results highlight the potential of causal inference as a powerful tool for
addressing domain shift in histopathology image analysis.

### 10. [A Multi-domain Image Translative Diffusion StyleGAN for Iris Presentation Attack Detection](http://arxiv.org/pdf/2510.14314v1)

Authors: Shivangi Yadav, Arun Ross

An iris biometric system can be compromised by presentation attacks (PAs)
where artifacts such as artificial eyes, printed eye images, or cosmetic
contact lenses are presented to the system. To counteract this, several
presentation attack detection (PAD) methods have been developed. However, there
is a scarcity of datasets for training and evaluating iris PAD techniques due
to the implicit difficulties in constructing and imaging PAs. To address this,
we introduce the Multi-domain Image Translative Diffusion StyleGAN
(MID-StyleGAN), a new framework for generating synthetic ocular images that
captures the PA and bonafide characteristics in multiple domains such as
bonafide, printed eyes and cosmetic contact lens. MID-StyleGAN combines the
strengths of diffusion models and generative adversarial networks (GANs) to
produce realistic and diverse synthetic data. Our approach utilizes a
multi-domain architecture that enables the translation between bonafide ocular
images and different PA domains. The model employs an adaptive loss function
tailored for ocular data to maintain domain consistency. Extensive experiments
demonstrate that MID-StyleGAN outperforms existing methods in generating
high-quality synthetic ocular images. The generated data was used to
significantly enhance the performance of PAD systems, providing a scalable
solution to the data scarcity problem in iris and ocular biometrics. For
example, on the LivDet2020 dataset, the true detect rate at 1% false detect
rate improved from 93.41% to 98.72%, showcasing the impact of the proposed
method.

### Computers and Society

### 1. [Technological Devices and Their Negative Effects on Health](http://arxiv.org/pdf/2510.14221v1)

Authors: Alida Vallejo-López, Cesar Noboa-Terán, Juana Kou-Guzmán, Josefina Ramírez-Amaya

Technology has become a global tool that allows us to obtain information and
analyze data, streamlines communication, and allows us to share images, data,
videos, texts, etc. Daily activities have gone from traditional to digital.
Today, it is impossible to live without an electronic device. In this context,
changes in people's health observed, with various complaints ranging from
visual, neurological, and concentration problems to muscular, hearing, and
sleep disorders. Society must be aware of the importance of using various
technological devices responsibly to protect people's health in general.
Keywords: Technology, activities, protect, electronic, Radiation, Health.

### 2. [Closing the Loop: An Instructor-in-the-Loop AI Assistance System for Supporting Student Help-Seeking in Programming Education](http://arxiv.org/pdf/2510.14457v1)

Authors: Tung Phung, Heeryung Choi, Mengyan Wu, Christopher Brooks, Sumit Gulwani, Adish Singla

Timely and high-quality feedback is essential for effective learning in
programming courses; yet, providing such support at scale remains a challenge.
While AI-based systems offer scalable and immediate help, their responses can
occasionally be inaccurate or insufficient. Human instructors, in contrast, may
bring more valuable expertise but are limited in time and availability. To
address these limitations, we present a hybrid help framework that integrates
AI-generated hints with an escalation mechanism, allowing students to request
feedback from instructors when AI support falls short. This design leverages
the strengths of AI for scale and responsiveness while reserving instructor
effort for moments of greatest need. We deployed this tool in a data science
programming course with 82 students. We observe that out of the total 673
AI-generated hints, students rated 146 (22%) as unhelpful. Among those, only 16
(11%) of the cases were escalated to the instructors. A qualitative
investigation of instructor responses showed that those feedback instances were
incorrect or insufficient roughly half of the time. This finding suggests that
when AI support fails, even instructors with expertise may need to pay greater
attention to avoid making mistakes. We will publicly release the tool for
broader adoption and enable further studies in other classrooms. Our work
contributes a practical approach to scaling high-quality support and informs
future efforts to effectively integrate AI and humans in education.

### 3. [Trends of Pink Slime Journalism Advertisement Expenditure and Spread on Facebook from 2019-2024](http://arxiv.org/pdf/2510.14818v1)

Authors: Christine Sowa Lepird, Lynnette Hui Xian Ng, Kathleen M. Carley

Pink slime journalism is a practice where news outlets publish low-quality or
inflammatory partisan articles, claiming to be local news networks. This paper
examines the spread of pink slime sites on Facebook using public posts from
Pages and Groups. We evaluate the trends of sharing pink slime sites on
Facebook and patterns regarding the advertisements purchased by the parent
organizations of the pink slime news networks. Our analysis discovers that
while the number of pink slime posts on Facebook pages have decreased over the
years, advertising dollars have increased. The increase in advertising dollars
influences an increase in Facebook group posts. Further, the advertising
expenditure increases during election years, but contentious topics are still
discussed during non-election years. By illustrating the patterns and themes
from US election years of 2020, 2022, and 2024, this research offers insights
into potentially dangerous journalism tactics, and provides predictions for
future US Presidential Elections.

### 4. [A Comprehensive Framework for Efficient Court Case Management and Prioritization](http://arxiv.org/pdf/2510.14892v1)

Authors: Shubham Varma, Ananya Warior, Avani Sakhapara, Dipti Pawade

The Indian judicial system faces a critical challenge with approximately 52
million pending cases, causing significant delays that impact socio-economic
stability. This study proposes a cloud-based software framework to classify and
prioritize court cases using algorithmic methods based on parameters such as
severity of crime committed, responsibility of parties involved, case filing
dates, previous hearing's data, priority level (e.g., Urgent, Medium, Ordinary)
provided as input, and relevant Indian Penal Code (IPC), Code of Criminal
Procedure (CrPC), and other legal sections (e.g., Hindu Marriage Act, Indian
Contract Act). Cases are initially entered by advocates on record or court
registrars, followed by automated hearing date allocation that balances fresh
and old cases while accounting for court holidays and leaves. The system
streamlines appellate processes by fetching data from historical case
databases. Our methodology integrates algorithmic prioritization, a robust
notification system, and judicial interaction, with features that allow judges
to view daily case counts and their details. Simulations demonstrate that the
system can process cases efficiently, with reliable notification delivery and
positive user satisfaction among judges and registrars. Future iterations will
incorporate advanced machine learning for dynamic prioritization, addressing
critical gaps in existing court case management systems to enhance efficiency
and reduce backlogs.

### 5. [A Simulation Framework for Studying Systemic Effects of Feedback Loops in Recommender Systems](http://arxiv.org/pdf/2510.14857v1)

Authors: Gabriele Barlacchi, Margherita Lalli, Emanuele Ferragina, Fosca Giannotti, Luca Pappalardo

Recommender systems continuously interact with users, creating feedback loops
that shape both individual behavior and collective market dynamics. This paper
introduces a simulation framework to model these loops in online retail
environments, where recommenders are periodically retrained on evolving
user-item interactions. Using the Amazon e-Commerce dataset, we analyze how
different recommendation algorithms influence diversity, purchase
concentration, and user homogenization over time. Results reveal a systematic
trade-off: while the feedback loop increases individual diversity, it
simultaneously reduces collective diversity and concentrates demand on a few
popular items. Moreover, for some recommender systems, the feedback loop
increases user homogenization over time, making user purchase profiles
increasingly similar. These findings underscore the need for recommender
designs that balance personalization with long-term diversity.

### 6. [From Binary to Bilingual: How the National Weather Service is Using Artificial Intelligence to Develop a Comprehensive Translation Program](http://arxiv.org/pdf/2510.14369v1)

Authors: Joseph E. Trujillo-Falcon, Monica L. Bozeman, Liam E. Llewellyn, Samuel T. Halvorson, Meryl Mizell, Stuti Deshpande, Bob Manning, Todd Fagin

To advance a Weather-Ready Nation, the National Weather Service (NWS) is
developing a systematic translation program to better serve the 68.8 million
people in the U.S. who do not speak English at home. This article outlines the
foundation of an automated translation tool for NWS products, powered by
artificial intelligence. The NWS has partnered with LILT, whose patented
training process enables large language models (LLMs) to adapt neural machine
translation (NMT) tools for weather terminology and messaging. Designed for
scalability across Weather Forecast Offices (WFOs) and National Centers, the
system is currently being developed in Spanish, Simplified Chinese, Vietnamese,
and other widely spoken non-English languages. Rooted in best practices for
multilingual risk communication, the system provides accurate, timely, and
culturally relevant translations, significantly reducing manual translation
time and easing operational workloads across the NWS. To guide the distribution
of these products, GIS mapping was used to identify language needs across
different NWS regions, helping prioritize resources for the communities that
need them most. We also integrated ethical AI practices throughout the
program's design, ensuring that transparency, fairness, and human oversight
guide how automated translations are created, evaluated, and shared with the
public. This work has culminated into a website featuring experimental
multilingual NWS products, including translated warnings, 7-day forecasts, and
educational campaigns, bringing the country one step closer to a national
warning system that reaches all Americans.

### 7. [The Impact of Medicaid Coverage on Mental Health, Why Insurance Makes People Happier in OHIE: by Spending Less or by Spending More?](http://arxiv.org/pdf/2510.14909v1)

Authors: Yangyang Li

The Oregon Health Insurance Experiment (OHIE) offers a unique opportunity to
examine the causal relationship between Medicaid coverage and happiness among
low-income adults, using an experimental design. This study leverages data from
comprehensive surveys conducted at 0 and 12 months post-treatment. Previous
studies based on OHIE have shown that individuals receiving Medicaid exhibited
a significant improvement in mental health compared to those who did not
receive coverage. The primary objective is to explore how Medicaid coverage
impacts happiness, specifically analyzing in which direction variations in
healthcare spending significantly improve mental health: higher spending or
lower spending after Medicaid. Utilizing instrumental variable (IV) regression,
I conducted six separate regressions across subgroups categorized by
expenditure levels and happiness ratings, and the results reveal distinct
patterns. Enrolling in OHP has significantly decreased the probability of
experiencing unhappiness, regardless of whether individuals had high or low
medical spending. Additionally, it decreased the probability of being pretty
happy and having high medical expenses, while increasing the probability among
those with lower expenses. Concerning the probability of being very happy, the
OHP only had a positive effect on being very happy and spending less, and its
effect on those with high expenses was insignificant. These findings align with
the benefit of Medicaid: alleviating financial burden, contributing to the
well-being of distinct subgroups.

### 8. [Detecting Early and Implicit Suicidal Ideation via Longitudinal and Information Environment Signals on Social Media](http://arxiv.org/pdf/2510.14889v1)

Authors: Soorya Ram Shimgekar, Ruining Zhao, Agam Goyal, Violeta J. Rodriguez, Paul A. Bloom, Hari Sundaram, Koustuv Saha

On social media, many individuals experiencing suicidal ideation (SI) do not
disclose their distress explicitly. Instead, signs may surface indirectly
through everyday posts or peer interactions. Detecting such implicit signals
early is critical but remains challenging. We frame early and implicit SI as a
forward-looking prediction task and develop a computational framework that
models a user's information environment, consisting of both their longitudinal
posting histories as well as the discourse of their socially proximal peers. We
adopted a composite network centrality measure to identify top neighbors of a
user, and temporally aligned the user's and neighbors' interactions --
integrating the multi-layered signals in a fine-tuned DeBERTa-v3 model. In a
Reddit study of 1,000 (500 Case and 500 Control) users, our approach improves
early and implicit SI detection by 15% over individual-only baselines. These
findings highlight that peer interactions offer valuable predictive signals and
carry broader implications for designing early detection systems that capture
indirect as well as masked expressions of risk in online environments.

### Databases

### 1. [Towards a Multimodal Stream Processing System](http://arxiv.org/pdf/2510.14631v1)

Authors: Uélison Jean Lopes dos Santos, Alessandro Ferri, Szilard Nistor, Riccardo Tommasini, Carsten Binnig, Manisha Luthra

In this paper, we present a vision for a new generation of multimodal
streaming systems that embed MLLMs as first-class operators, enabling real-time
query processing across multiple modalities. Achieving this is non-trivial:
while recent work has integrated MLLMs into databases for multimodal queries,
streaming systems require fundamentally different approaches due to their
strict latency and throughput requirements. Our approach proposes novel
optimizations at all levels, including logical, physical, and semantic query
transformations that reduce model load to improve throughput while preserving
accuracy. We demonstrate this with \system{}, a prototype leveraging such
optimizations to improve performance by more than an order of magnitude.
Moreover, we discuss a research roadmap that outlines open research challenges
for building a scalable and efficient multimodal stream processing systems.

### Distributed, Parallel, and Cluster Computing

### 1. [Proof-Carrying Fair Ordering: Asymmetric Verification for BFT via Incremental Graphs](http://arxiv.org/pdf/2510.14186v1)

Authors: Pengkun Ren, Hai Dong, Nasrin Sohrabi, Zahir Tari, Pengcheng Zhang

Byzantine Fault-Tolerant (BFT) consensus protocols ensure agreement on
transaction ordering despite malicious actors, but unconstrained ordering power
enables sophisticated value extraction attacks like front running and sandwich
attacks - a critical threat to blockchain systems. Order-fair consensus curbs
adversarial value extraction by constraining how leaders may order
transactions. While state-of-the-art protocols such as Themis attain strong
guarantees through graph-based ordering, they ask every replica to re-run the
leader's expensive ordering computation for validation - an inherently
symmetric and redundant paradigm. We present AUTIG, a high-performance,
pluggable order-fairness service that breaks this symmetry. Our key insight is
that verifying a fair order does not require re-computing it. Instead,
verification can be reduced to a stateless audit of succinct, verifiable
assertions about the ordering graph's properties. AUTIG realizes this via an
asymmetric architecture: the leader maintains a persistent
Unconfirmed-Transaction Incremental Graph (UTIG) to amortize graph construction
across rounds and emits a structured proof of fairness with each proposal;
followers validate the proof without maintaining historical state. AUTIG
introduces three critical innovations: (i) incremental graph maintenance driven
by threshold-crossing events and state changes; (ii) a decoupled pipeline that
overlaps leader-side collection/update/extraction with follower-side stateless
verification; and (iii) a proof design covering all internal pairs in the
finalized prefix plus a frontier completeness check to rule out hidden external
dependencies. We implement AUTIG and evaluate it against symmetric graph-based
baselines under partial synchrony. Experiments show higher throughput and lower
end-to-end latency while preserving gamma-batch-order-fairness.

### 2. [ScalePool: Hybrid XLink-CXL Fabric for Composable Resource Disaggregation in Unified Scale-up Domains](http://arxiv.org/pdf/2510.14580v1)

Authors: Hyein Woo, Miryeong Kwon, Jiseon Kim, Eunjee Na, Hanjin Choi, Seonghyeon Jang, Myoungsoo Jung

This paper proposes ScalePool, a novel cluster architecture designed to
interconnect numerous accelerators using unified hardware interconnects rather
than traditional long-distance networking. ScalePool integrates
Accelerator-Centric Links (XLink) and Compute Express Link (CXL) into a unified
XLink-CXL hybrid fabric. Specifically, ScalePool employs XLink for
intra-cluster, low-latency accelerator communication, while using hierarchical
CXL-based switching fabrics for scalable and coherent inter-cluster memory
sharing. By abstracting interfaces through CXL, ScalePool structurally resolves
interoperability constraints, enabling heterogeneous cluster operation and
composable resource disaggregation. In addition, ScalePool introduces explicit
memory tiering: the latency-critical tier-1 combines accelerator-local memory
with coherence-centric CXL and XLink, whereas the highcapacity tier-2 employs
dedicated memory nodes interconnected by a CXL-based fabric, achieving scalable
and efficient memory pooling. Evaluation results show that ScalePool
accelerates LLM training by 1.22x on average and up to 1.84x compared to
conventional RDMA-based environments. Furthermore, the proposed tier-2 memory
disaggregation strategy reduces latency by up to 4.5x for memory-intensive
workloads.

### 3. [JASDA: Introducing Job-Aware Scheduling in Scheduler-Driven Job Atomization](http://arxiv.org/pdf/2510.14599v1)

Authors: Michal Konopa, Jan Fesl, Ladislav Ber ánek

The increasing complexity and temporal variability of workloads on
MIG-enabled GPUs challenge the scalability of traditional centralized
scheduling. Building upon the SJA concept, this paper introduces JASDA-a novel
paradigm that extends SJA from a largely centralized scheduling model toward a
fully decentralized negotiation process. In JASDA, jobs actively generate and
score feasible subjobs in response to scheduler-announced execution windows,
while the scheduler performs policy-driven clearing that balances utilization,
fairness, and temporal responsiveness. This bidirectional, iterative
interaction embeds feedback, calibration, and probabilistic safety directly
into the scheduling loop, enabling adaptive and transparent decision-making. By
coupling principles from auction theory and online optimization with the
temporal granularity of GPU workloads, JASDA provides a scalable foundation for
market-aware and fairness-driven resource management-bridging theoretical
scheduling models with practical deployment in modern MIG-enabled environments
relevant to Artificial Intelligence and Agriculture 4.0.

### 4. [MPI-over-CXL: Enhancing Communication Efficiency in Distributed HPC Systems](http://arxiv.org/pdf/2510.14622v1)

Authors: Miryeong Kwon, Donghyun Gouk, Hyein Woo, Junhee Kim, Jinwoo Baek, Kyungkuk Nam, Sangyoon Ji, Jiseon Kim, Hanyeoreum Bae, Junhyeok Jang, Hyunwoo You, Junseok Moon, Myoungsoo Jung

MPI implementations commonly rely on explicit memory-copy operations,
incurring overhead from redundant data movement and buffer management. This
overhead notably impacts HPC workloads involving intensive inter-processor
communication. In response, we introduce MPI-over-CXL, a novel MPI
communication paradigm leveraging CXL, which provides cache-coherent shared
memory across multiple hosts. MPI-over-CXL replaces traditional data-copy
methods with direct shared memory access, significantly reducing communication
latency and memory bandwidth usage. By mapping shared memory regions directly
into the virtual address spaces of MPI processes, our design enables efficient
pointer-based communication, eliminating redundant copying operations. To
validate this approach, we implement a comprehensive hardware and software
environment, including a custom CXL 3.2 controller, FPGA-based multi-host
emulation, and dedicated software stack. Our evaluations using representative
benchmarks demonstrate substantial performance improvements over conventional
MPI systems, underscoring MPI-over-CXL's potential to enhance efficiency and
scalability in large-scale HPC environments.

### 5. [Incentive-Based Federated Learning](http://arxiv.org/pdf/2510.14208v1)

Authors: Chanuka A. S. Hewa Kaluannakkage, Rajkumar Buyya

Federated learning promises to revolutionize machine learning by enabling
collaborative model training without compromising data privacy. However,
practical adaptability can be limited by critical factors, such as the
participation dilemma. Participating entities are often unwilling to contribute
to a learning system unless they receive some benefits, or they may pretend to
participate and free-ride on others. This chapter identifies the fundamental
challenges in designing incentive mechanisms for federated learning systems. It
examines how foundational concepts from economics and game theory can be
applied to federated learning, alongside technology-driven solutions such as
blockchain and deep reinforcement learning. This work presents a comprehensive
taxonomy that thoroughly covers both centralized and decentralized
architectures based on the aforementioned theoretical concepts. Furthermore,
the concepts described are presented from an application perspective, covering
emerging industrial applications, including healthcare, smart infrastructure,
vehicular networks, and blockchain-based decentralized systems. Through this
exploration, this chapter demonstrates that well-designed incentive mechanisms
are not merely optional features but essential components for the practical
success of federated learning. This analysis reveals both the promising
solutions that have emerged and the significant challenges that remain in
building truly sustainable, fair, and robust federated learning ecosystems.

### 6. [FairBatching: Fairness-Aware Batch Formation for LLM Inference](http://arxiv.org/pdf/2510.14392v1)

Authors: Hongtao Lyu, Boyue Liu, Mingyu Wu, Haibo Chen

Large language model (LLM) inference systems face a fundamental tension
between minimizing Time-to-First-Token (TTFT) latency for new requests and
maintaining a high, steady token generation rate (low Time-Per-Output-Token, or
TPOT) for ongoing requests. Existing stall-free batching schedulers proposed by
Sarathi, while effective at preventing decode stalls, introduce significant
computational unfairness. They prioritize decode tasks excessively,
simultaneously leading to underutilized decode slack and unnecessary prefill
queuing delays, which collectively degrade the system's overall quality of
service (QoS).
  This work identifies the root cause of this unfairness: the non-monotonic
nature of Time-Between-Tokens (TBT) as a scheduling metric and the rigid
decode-prioritizing policy that fails to adapt to dynamic workload bursts. We
therefore propose FairBatching, a novel LLM inference scheduler that enforces
fair resource allocation between prefill and decode tasks. It features an
adaptive batch capacity determination mechanism, which dynamically adjusts the
computational budget to improve the GPU utilization without triggering SLO
violations. Its fair and dynamic batch formation algorithm breaks away from the
decode-prioritizing paradigm, allowing computation resources to be reclaimed
from bursting decode tasks to serve prefill surges, achieving global fairness.
Furthermore, FairBatching provides a novel load estimation method, enabling
more effective coordination with upper-level schedulers. Implemented and
evaluated on realistic traces, FairBatching significantly reduces TTFT tail
latency by up to 2.29x while robustly maintaining TPOT SLOs, achieving overall
20.0% improvement in single-node capacity and 54.3% improvement in
cluster-level capacity.

### 7. [xLLM Technical Report](http://arxiv.org/pdf/2510.14686v1)

Authors: Tongxuan Liu, Tao Peng, Peijun Yang, Xiaoyang Zhao, Xiusheng Lu, Weizhe Huang, Zirui Liu, Xiaoyu Chen, Zhiwei Liang, Jun Xiong, Donghe Jin, Minchao Zhang, Jinrong Guo, Yingxu Deng, Xu Zhang, Xianzhe Dong, Siqi Wang, Siyu Wu, Yu Wu, Zihan Tang, Yuting Zeng, Yanshu Wang, Jinguang Liu, Meng Kang, Menxin Li, Yunlong Wang, Yiming Liu, Xiaolong Ma, Yifan Wang, Yichen Zhang, Jinrun Yin, Keyang Zheng, Jiawei Yin, Jun Zhang, Ziyue Wang, Xiaobo Lin, Liangyu Liu, Liwei Lan, Yang Liu, Chunhua Peng, Han Liu, Songcheng Ren, Xuezhu Wang, Yunheng Shen, Yi Wang, Guyue Liu, Hui Chen, Tong Yang, Hailong Yang, Jing Li, Guiguang Ding, Ke Zhang

We introduce xLLM, an intelligent and efficient Large Language Model (LLM)
inference framework designed for high-performance, large-scale enterprise-grade
serving, with deep optimizations for diverse AI accelerators. To address these
challenges, xLLM builds a novel decoupled service-engine architecture. At the
service layer, xLLM-Service features an intelligent scheduling module that
efficiently processes multimodal requests and co-locates online and offline
tasks through unified elastic scheduling to maximize cluster utilization. This
module also relies on a workload-adaptive dynamic Prefill-Decode (PD)
disaggregation policy and a novel Encode-Prefill-Decode (EPD) disaggregation
policy designed for multimodal inputs. Furthermore, it incorporates a
distributed architecture to provide global KV Cache management and robust
fault-tolerant capabilities for high availability. At the engine layer,
xLLM-Engine co-optimizes system and algorithm designs to fully saturate
computing resources. This is achieved through comprehensive multi-layer
execution pipeline optimizations, an adaptive graph mode and an xTensor memory
management. xLLM-Engine also further integrates algorithmic enhancements such
as optimized speculative decoding and dynamic EPLB, collectively serving to
substantially boost throughput and inference efficiency. Extensive evaluations
demonstrate that xLLM delivers significantly superior performance and resource
efficiency. Under identical TPOT constraints, xLLM achieves throughput up to
1.7x that of MindIE and 2.2x that of vLLM-Ascend with Qwen-series models, while
maintaining an average throughput of 1.7x that of MindIE with Deepseek-series
models. xLLM framework is publicly available at
https://github.com/jd-opensource/xllm and
https://github.com/jd-opensource/xllm-service.

### 8. [Deadlock-free routing for Full-mesh networks without using Virtual Channels](http://arxiv.org/pdf/2510.14730v1)

Authors: Alejandro Cano, Cristóbal Camarero, Carmen Martínez, Ramón Beivide

High-radix, low-diameter networks like HyperX and Dragonfly use a Full-mesh
core, and rely on multiple virtual channels (VCs) to avoid packet deadlocks in
adaptive routing. However, VCs introduce significant overhead in the switch in
terms of area, power, and design complexity, limiting the switch scalability.
This paper starts by revisiting VC-less routing through link ordering schemes
in Full-mesh networks, which offer implementation simplicity but suffer from
performance degradation under adversarial traffic. Thus, to overcome these
challenges, we propose TERA (Topology-Embedded Routing Algorithm), a novel
routing algorithm which employs an embedded physical subnetwork to provide
deadlock-free non-minimal paths without using VCs.
  In a Full-mesh network, TERA outperforms link ordering routing algorithms by
80% when dealing with adversarial traffic, and up to 100% in application
kernels. Furthermore, compared to other VC-based approaches, it reduces buffer
requirements by 50%, while maintaining comparable latency and throughput.
Lastly, early results from a 2D-HyperX evaluation show that TERA outperforms
state-of-the-art algorithms that use the same number of VCs, achieving
performance improvements of up to 32%.

### 9. [Multi-modal video data-pipelines for machine learning with minimal human supervision](http://arxiv.org/pdf/2510.14862v1)

Authors: Mihai-Cristian Pîrvu, Marius Leordeanu

The real-world is inherently multi-modal at its core. Our tools observe and
take snapshots of it, in digital form, such as videos or sounds, however much
of it is lost. Similarly for actions and information passing between humans,
languages are used as a written form of communication. Traditionally, Machine
Learning models have been unimodal (i.e. rgb -> semantic or text ->
sentiment_class). Recent trends go towards bi-modality, where images and text
are learned together, however, in order to truly understand the world, we need
to integrate all these independent modalities. In this work we try to combine
as many visual modalities as we can using little to no human supervision. In
order to do this, we use pre-trained experts and procedural combinations
between them on top of raw videos using a fully autonomous data-pipeline, which
we also open-source. We then make use of PHG-MAE, a model specifically designed
to leverage multi-modal data. We show that this model which was efficiently
distilled into a low-parameter (<1M) can have competitive results compared to
models of ~300M parameters. We deploy this model and analyze the use-case of
real-time semantic segmentation from handheld devices or webcams on commodity
hardware. Finally, we deploy other off-the-shelf models using the same
framework, such as DPT for near real-time depth estimation.

### 10. [The Bidding Games: Reinforcement Learning for MEV Extraction on Polygon Blockchain](http://arxiv.org/pdf/2510.14642v1)

Authors: Andrei Seoev, Leonid Gremyachikh, Anastasiia Smirnova, Yash Madhwal, Alisa Kalacheva, Dmitry Belousov, Ilia Zubov, Aleksei Smirnov, Denis Fedyanin, Vladimir Gorgadze, Yury Yanovich

In blockchain networks, the strategic ordering of transactions within blocks
has emerged as a significant source of profit extraction, known as Maximal
Extractable Value (MEV). The transition from spam-based Priority Gas Auctions
to structured auction mechanisms like Polygon Atlas has transformed MEV
extraction from public bidding wars into sealed-bid competitions under extreme
time constraints. While this shift reduces network congestion, it introduces
complex strategic challenges where searchers must make optimal bidding
decisions within a sub-second window without knowledge of competitor behavior
or presence. Traditional game-theoretic approaches struggle in this
high-frequency, partially observable environment due to their reliance on
complete information and static equilibrium assumptions. We present a
reinforcement learning framework for MEV extraction on Polygon Atlas and make
three contributions: (1) A novel simulation environment that accurately models
the stochastic arrival of arbitrage opportunities and probabilistic competition
in Atlas auctions; (2) A PPO-based bidding agent optimized for real-time
constraints, capable of adaptive strategy formulation in continuous action
spaces while maintaining production-ready inference speeds; (3) Empirical
validation demonstrating our history-conditioned agent captures 49\% of
available profits when deployed alongside existing searchers and 81\% when
replacing the market leader, significantly outperforming static bidding
strategies. Our work establishes that reinforcement learning provides a
critical advantage in high-frequency MEV environments where traditional
optimization methods fail, offering immediate value for industrial participants
and protocol designers alike.

### Discrete Mathematics

### 1. [Decidability and Characterization of Expansivity for Group Cellular Automata](http://arxiv.org/pdf/2510.14568v1)

Authors: Niccolo' Castronuovo, Alberto Dennunzio, Luciano Margara

Group cellular automata are continuous, shift-commuting endomorphisms of
$G^\mathbb{Z}$, where $G$ is a finite group. We provide an easy-to-check
characterization of expansivity for group cellular automata on abelian groups
and we prove that expansivity is a decidable property for general (non-abelian)
groups. Moreover, we show that the class of expansive group cellular automata
is strictly contained in that of topologically transitive injective group
cellular automata.

### 2. [Excluding $K_{2,t}$ as a fat minor](http://arxiv.org/pdf/2510.14644v1)

Authors: Sandra Albrechtsen, Marc Distel, Agelos Georgakopoulos

We prove that for every $t \in \mathbb{N}$, the graph $K_{2,t}$ satisfies the
fat minor conjecture of Georgakopoulos and Papasoglu: for every $K\in
\mathbb{N}$ there exist $M,A\in \mathbb{N}$ such that every graph with no
$K$-fat $K_{2,t}$ minor is $(M,A)$-quasi-isometric to a graph with no $K_{2,t}$
minor. We use this to obtain an efficient algorithm for approximating the
minimal multiplicative distortion of any embedding of a finite graph into a
$K_{2,t}$-minor-free graph, answering a question of Chepoi, Dragan, Newman,
Rabinovich, and Vax\`es from 2012.

### 3. [An efficient algorithm for \textsc{$\mathcal{F}$-subgraph-free Edge Deletion} on graphs having a product structure](http://arxiv.org/pdf/2510.14674v1)

Authors: Shinwoo An, Seonghyuk Im, Seokbeom Kim, Myounghwan Lee

Given a family $\mathcal{F}$ of graphs, a graph is
\emph{$\mathcal{F}$-subgraph-free} if it has no subgraph isomorphic to a member
of $\mathcal{F}$. We present a fixed-parameter linear-time algorithm that
decides whether a planar graph can be made $\mathcal{F}$-subgraph-free by
deleting at most $k$ vertices or $k$ edges, where the parameters are $k$,
$\lvert \mathcal{F} \rvert$, and the maximum number of vertices in a member of
$\mathcal{F}$. The running time of our algorithm is double-exponential in the
parameters, which is faster than the algorithm obtained by applying the
first-order model checking result for graphs of bounded twin-width.
  To obtain this result, we develop a unified framework for designing
algorithms for this problem on graphs with a ``product structure.'' Using this
framework, we also design algorithms for other graph classes that generalize
planar graphs. Specifically, the problem admits a fixed-parameter linear time
algorithm on disk graphs of bounded local radius, and a fixed-parameter
almost-linear time algorithm on graphs of bounded genus.
  Finally, we show that our result gives a tight fixed-parameter algorithm in
the following sense: Even when $\mathcal{F}$ consists of a single graph $F$ and
the input is restricted to planar graphs, it is unlikely to drop any parameters
$k$ and $\lvert V(F) \rvert$ while preserving fixed-parameter tractability,
unless the Exponential-Time Hypothesis fails.

### Data Structures and Algorithms

### 1. [A Levelset Algorithm for 3D-Tarksi](http://arxiv.org/pdf/2510.14777v1)

Authors: Sebastian Haslebacher, Jonas Lill

We present a simple new algorithm for finding a Tarski fixed point of a
monotone function $F : [N]^3 \rightarrow [N]^3$. Our algorithm runs in
$O(\log^2 N)$ time and makes $O(\log^2 N)$ queries to $F$, matching the
$\Omega(\log^2 N)$ query lower bound due to Etessami et al.\ as well as the
existing state-of-the-art algorithm due to Fearnley et al.

### 2. [Tree-Like Shortcuttings of Trees](http://arxiv.org/pdf/2510.14918v1)

Authors: Hung Le, Lazar Milenković, Shay Solomon, Cuong Than

Sparse shortcuttings of trees -- equivalently, sparse 1-spanners for tree
metrics with bounded hop-diameter -- have been studied extensively (under
different names and settings), since the pioneering works of [Yao82, Cha87,
AS87, BTS94], initially motivated by applications to range queries, online tree
product, and MST verification, to name a few. These constructions were also
lifted from trees to other graph families using known low-distortion embedding
results. The works of [Yao82, Cha87, AS87, BTS94] establish a tight tradeoff
between hop-diameter and sparsity (or average degree) for tree shortcuttings
and imply constant-hop shortcuttings for $n$-node trees with sparsity $O(\log^*
n)$. Despite their small sparsity, all known constant-hop shortcuttings contain
dense subgraphs (of sparsity $\Omega(\log n)$), which is a significant drawback
for many applications.
  We initiate a systematic study of constant-hop tree shortcuttings that are
``tree-like''. We focus on two well-studied graph parameters that measure how
far a graph is from a tree: arboricity and treewidth. Our contribution is
twofold.
  * New upper and lower bounds for tree-like shortcuttings of trees, including
an optimal tradeoff between hop-diameter and treewidth for all hop-diameter up
to $O(\log\log n)$. We also provide a lower bound for larger values of $k$,
which together yield $\text{hop-diameter}\times \text{treewidth} =
\Omega((\log\log n)^2)$ for all values of hop-diameter, resolving an open
question of [FL22, Le23]. [...]

### 3. [Prediction-Specific Design of Learning-Augmented Algorithms](http://arxiv.org/pdf/2510.14887v1)

Authors: Sizhe Li, Nicolas Christianson, Tongxin Li

Algorithms with predictions} has emerged as a powerful framework to combine
the robustness of traditional online algorithms with the data-driven
performance benefits of machine-learned (ML) predictions. However, most
existing approaches in this paradigm are overly conservative, {as they do not
leverage problem structure to optimize performance in a prediction-specific
manner}. In this paper, we show that such prediction-specific performance
criteria can enable significant performance improvements over the coarser
notions of consistency and robustness considered in prior work. Specifically,
we propose a notion of \emph{strongly-optimal} algorithms with predictions,
which obtain Pareto optimality not just in the worst-case tradeoff between
robustness and consistency, but also in the prediction-specific tradeoff
between these metrics. We develop a general bi-level optimization framework
that enables systematically designing strongly-optimal algorithms in a wide
variety of problem settings, and we propose explicit strongly-optimal
algorithms for several classic online problems: deterministic and randomized
ski rental, and one-max search. Our analysis reveals new structural insights
into how predictions can be optimally integrated into online algorithms by
leveraging a prediction-specific design. To validate the benefits of our
proposed framework, we empirically evaluate our algorithms in case studies on
problems including dynamic power management and volatility-based index trading.
Our results demonstrate that prediction-specific, strongly-optimal algorithms
can significantly improve performance across a variety of online
decision-making settings.

### 4. [An efficient algorithm for \textsc{$\mathcal{F}$-subgraph-free Edge Deletion} on graphs having a product structure](http://arxiv.org/pdf/2510.14674v1)

Authors: Shinwoo An, Seonghyuk Im, Seokbeom Kim, Myounghwan Lee

Given a family $\mathcal{F}$ of graphs, a graph is
\emph{$\mathcal{F}$-subgraph-free} if it has no subgraph isomorphic to a member
of $\mathcal{F}$. We present a fixed-parameter linear-time algorithm that
decides whether a planar graph can be made $\mathcal{F}$-subgraph-free by
deleting at most $k$ vertices or $k$ edges, where the parameters are $k$,
$\lvert \mathcal{F} \rvert$, and the maximum number of vertices in a member of
$\mathcal{F}$. The running time of our algorithm is double-exponential in the
parameters, which is faster than the algorithm obtained by applying the
first-order model checking result for graphs of bounded twin-width.
  To obtain this result, we develop a unified framework for designing
algorithms for this problem on graphs with a ``product structure.'' Using this
framework, we also design algorithms for other graph classes that generalize
planar graphs. Specifically, the problem admits a fixed-parameter linear time
algorithm on disk graphs of bounded local radius, and a fixed-parameter
almost-linear time algorithm on graphs of bounded genus.
  Finally, we show that our result gives a tight fixed-parameter algorithm in
the following sense: Even when $\mathcal{F}$ consists of a single graph $F$ and
the input is restricted to planar graphs, it is unlikely to drop any parameters
$k$ and $\lvert V(F) \rvert$ while preserving fixed-parameter tractability,
unless the Exponential-Time Hypothesis fails.

### 5. [Online Proportional Apportionment](http://arxiv.org/pdf/2510.14752v1)

Authors: Javier Cembrano, Jose Correa, Svenja M. Griesbach, Victor Verdugo

Traditionally, the problem of apportioning the seats of a legislative body
has been viewed as a one-shot process with no dynamic considerations. While
this approach is reasonable for some settings, dynamic aspects play an
important role in many others. We initiate the study of apportionment problems
in an online setting. Specifically, we introduce a framework for proportional
apportionment with no information about the future. In this model, time is
discrete and there are $n$ parties that receive a certain share of the votes at
each time step. An online algorithm needs to irrevocably assign a prescribed
number of seats at each time, ensuring that each party receives its fractional
share rounded up or down, and that the cumulative number of seats allocated to
each party remains close to its cumulative share up to that time.
  We study deterministic and randomized online apportionment methods. For
deterministic methods, we construct a family of adversarial instances that
yield a lower bound, linear in $n$, on the worst-case deviation between the
seats allocated to a party and its cumulative share. We show that this bound is
best possible and is matched by a natural greedy method. As a consequence, a
method guaranteeing that the cumulative number of seats assigned to each party
up to any step equals its cumulative share rounded up or down (global quota)
exists if and only if $n\leq 3$. Then, we turn to randomized allocations and
show that, for $n\leq 3$, we can randomize over methods satisfying global quota
with the additional guarantee that each party receives, in expectation, its
proportional share in every step. Our proof is constructive: Any method
satisfying these properties can be obtained from a flow on a recursively
constructed network. We showcase the applicability of our results to obtain
approximate solutions in the context of online dependent rounding procedures.

### 6. [Balls and Bins and the Infinite Process with Random Deletions](http://arxiv.org/pdf/2510.14798v1)

Authors: Petra Berenbrink, Tom Friedetzky, Peter Kling, Lars Nagel

We consider an infinite balls-into-bins process with deletions where in each
discrete step $t$ a coin is tossed as to whether, with probability $\beta(t)
\in (0,1)$, a new ball is allocated using the Greedy[2] strategy (which places
the ball in the lower loaded of two bins sampled uniformly at random) or, with
remaining probability $1-\beta(t)$, a ball is deleted from a non-empty bin
chosen uniformly at random. Let $n$ be the number of bins and $m(t)$ the total
load at time $t$. We are interested in bounding the discrepancy $x_{\max}(t) -
m(t)/n$ (current maximum load relative to current average) and the overload
$x_{\max}(t) - m_{\max}(t)/n$ (current maximum load relative to highest average
observed so far).
  We prove that at an arbitrarily chosen time $t$ the total number of balls
above the average is $O(n)$ and that the discrepancy is $ O(\log(n))$. For the
discrepancy, we provide a matching lower bound. Furthermore we prove that at an
arbitrarily chosen time $t$ the overload is $\log\log(n)+O(1)$. For "good"
insertion probability sequences (in which the average load of time intervals
with polynomial length increases in expectation) we show that even the
discrepancy is bounded by $\log\log(n)+O(1)$.
  One of our main analytical tools is a layered induction, as per [ABKU99].
Since our model allows for rather more general scenarios than what was
previously considered, the formal analysis requires some extra ingredients as
well, in particular a detailed potential analysis. Furthermore, we simplify the
setup by applying probabilistic couplings to obtain certain "recovery"
properties, which eliminate much of the need for intricate and careful
conditioning elsewhere in the analysis.

### Formal Languages and Automata Theory

### 1. [Efficient Verification of Metric Temporal Properties with Past in Pointwise Semantics](http://arxiv.org/pdf/2510.14699v1)

Authors: S. Akshay, Prerak Contractor, Paul Gastin, R. Govind, B. Srivathsan

Model checking for real-timed systems is a rich and diverse topic. Among the
different logics considered, Metric Interval Temporal Logic (MITL) is a
powerful and commonly used logic, which can succinctly encode many interesting
timed properties especially when past and future modalities are used together.
In this work, we develop a new approach for MITL model checking in the
pointwise semantics, where our focus is on integrating past and maximizing
determinism in the translated automata.
  Towards this goal, we define synchronous networks of timed automata with
shared variables and show that the past fragment of MITL can be translated in
linear time to synchronous networks of deterministic timed automata. Moreover
determinism can be preserved even when the logic is extended with future
modalities at the top-level of the formula. We further extend this approach to
the full MITL with past, translating it into networks of generalized timed
automata (GTA) with future clocks (which extend timed automata and event clock
automata). We present an SCC-based liveness algorithm to analyse GTA. We
implement our translation in a prototype tool which handles both finite and
infinite timed words and supports past modalities. Our experimental evaluation
demonstrates that our approach significantly outperforms the state-of-the-art
in MITL satisfiability checking in pointwise semantics on a benchmark suite of
72 formulas. Finally, we implement an end-to-end model checking algorithm for
pointwise semantics and demonstrate its effectiveness on two well-known
benchmarks.

### 2. [Decidability and Characterization of Expansivity for Group Cellular Automata](http://arxiv.org/pdf/2510.14568v1)

Authors: Niccolo' Castronuovo, Alberto Dennunzio, Luciano Margara

Group cellular automata are continuous, shift-commuting endomorphisms of
$G^\mathbb{Z}$, where $G$ is a finite group. We provide an easy-to-check
characterization of expansivity for group cellular automata on abelian groups
and we prove that expansivity is a decidable property for general (non-abelian)
groups. Moreover, we show that the class of expansive group cellular automata
is strictly contained in that of topologically transitive injective group
cellular automata.

### 3. [On the order of lazy cellular automata](http://arxiv.org/pdf/2510.14841v1)

Authors: Edgar Alcalá-Arroyo, Alonso Castillo-Ramirez

We study the most elementary family of cellular automata defined over an
arbitrary group universe $G$ and an alphabet $A$: the lazy cellular automata,
which act as the identity on configurations in $A^G$, except when they read a
unique active transition $p \in A^S$, in which case they write a fixed symbol
$a \in A$. As expected, the dynamical behavior of lazy cellular automata is
relatively simple, yet subtle questions arise since they completely depend on
the choice of $p$ and $a$. In this paper, we investigate the order of a lazy
cellular automaton $\tau : A^G \to A^G$, defined as the cardinality of the set
$\{ \tau^k : k \in \mathbb{N} \}$. In particular, we establish a general upper
bound for the order of $\tau$ in terms of $p$ and $a$, and we prove that this
bound is attained when $p$ is a quasi-constant pattern.

### Graphics

### 1. [GauSSmart: Enhanced 3D Reconstruction through 2D Foundation Models and Geometric Filtering](http://arxiv.org/pdf/2510.14270v1)

Authors: Alexander Valverde, Brian Xu, Yuyin Zhou, Meng Xu, Hongyun Wang

Scene reconstruction has emerged as a central challenge in computer vision,
with approaches such as Neural Radiance Fields (NeRF) and Gaussian Splatting
achieving remarkable progress. While Gaussian Splatting demonstrates strong
performance on large-scale datasets, it often struggles to capture fine details
or maintain realism in regions with sparse coverage, largely due to the
inherent limitations of sparse 3D training data.
  In this work, we propose GauSSmart, a hybrid method that effectively bridges
2D foundational models and 3D Gaussian Splatting reconstruction. Our approach
integrates established 2D computer vision techniques, including convex
filtering and semantic feature supervision from foundational models such as
DINO, to enhance Gaussian-based scene reconstruction. By leveraging 2D
segmentation priors and high-dimensional feature embeddings, our method guides
the densification and refinement of Gaussian splats, improving coverage in
underrepresented areas and preserving intricate structural details.
  We validate our approach across three datasets, where GauSSmart consistently
outperforms existing Gaussian Splatting in the majority of evaluated scenes.
Our results demonstrate the significant potential of hybrid 2D-3D approaches,
highlighting how the thoughtful combination of 2D foundational models with 3D
reconstruction pipelines can overcome the limitations inherent in either
approach alone.

### 2. [Inpainting the Red Planet: Diffusion Models for the Reconstruction of Martian Environments in Virtual Reality](http://arxiv.org/pdf/2510.14765v1)

Authors: Giuseppe Lorenzo Catalano, Agata Marta Soccini

Space exploration increasingly relies on Virtual Reality for several tasks,
such as mission planning, multidisciplinary scientific analysis, and astronaut
training. A key factor for the reliability of the simulations is having
accurate 3D representations of planetary terrains. Extraterrestrial heightmaps
derived from satellite imagery often contain missing values due to acquisition
and transmission constraints. Mars is among the most studied planets beyond
Earth, and its extensive terrain datasets make the Martian surface
reconstruction a valuable task, although many areas remain unmapped. Deep
learning algorithms can support void-filling tasks; however, whereas Earth's
comprehensive datasets enables the use of conditional methods, such approaches
cannot be applied to Mars. Current approaches rely on simpler interpolation
techniques which, however, often fail to preserve geometric coherence. In this
work, we propose a method for reconstructing the surface of Mars based on an
unconditional diffusion model. Training was conducted on an augmented dataset
of 12000 Martian heightmaps derived from NASA's HiRISE survey. A
non-homogeneous rescaling strategy captures terrain features across multiple
scales before resizing to a fixed 128x128 model resolution. We compared our
method against established void-filling and inpainting techniques, including
Inverse Distance Weighting, kriging, and Navier-Stokes algorithm, on an
evaluation set of 1000 samples. Results show that our approach consistently
outperforms these methods in terms of reconstruction accuracy (4-15% on RMSE)
and perceptual similarity (29-81% on LPIPS) with the original data.

### 3. [Ponimator: Unfolding Interactive Pose for Versatile Human-human Interaction Animation](http://arxiv.org/pdf/2510.14976v1)

Authors: Shaowei Liu, Chuan Guo, Bing Zhou, Jian Wang

Close-proximity human-human interactive poses convey rich contextual
information about interaction dynamics. Given such poses, humans can
intuitively infer the context and anticipate possible past and future dynamics,
drawing on strong priors of human behavior. Inspired by this observation, we
propose Ponimator, a simple framework anchored on proximal interactive poses
for versatile interaction animation. Our training data consists of
close-contact two-person poses and their surrounding temporal context from
motion-capture interaction datasets. Leveraging interactive pose priors,
Ponimator employs two conditional diffusion models: (1) a pose animator that
uses the temporal prior to generate dynamic motion sequences from interactive
poses, and (2) a pose generator that applies the spatial prior to synthesize
interactive poses from a single pose, text, or both when interactive poses are
unavailable. Collectively, Ponimator supports diverse tasks, including
image-based interaction animation, reaction animation, and text-to-interaction
synthesis, facilitating the transfer of interaction knowledge from high-quality
mocap data to open-world scenarios. Empirical experiments across diverse
datasets and applications demonstrate the universality of the pose prior and
the effectiveness and robustness of our framework.

### 4. [Agentic Design of Compositional Machines](http://arxiv.org/pdf/2510.14980v1)

Authors: Wenqian Zhang, Weiyang Liu, Zhen Liu

The design of complex machines stands as both a marker of human intelligence
and a foundation of engineering practice. Given recent advances in large
language models (LLMs), we ask whether they, too, can learn to create. We
approach this question through the lens of compositional machine design: a task
in which machines are assembled from standardized components to meet functional
demands like locomotion or manipulation in a simulated physical environment. To
support this investigation, we introduce BesiegeField, a testbed built on the
machine-building game Besiege, which enables part-based construction, physical
simulation and reward-driven evaluation. Using BesiegeField, we benchmark
state-of-the-art LLMs with agentic workflows and identify key capabilities
required for success, including spatial reasoning, strategic assembly, and
instruction-following. As current open-source models fall short, we explore
reinforcement learning (RL) as a path to improvement: we curate a cold-start
dataset, conduct RL finetuning experiments, and highlight open challenges at
the intersection of language, machine design, and physical reasoning.

### Computer Science and Game Theory

### 1. [Why Instant-Runoff Voting Is So Resilient to Coalitional Manipulation: Phase Transitions in the Perturbed Culture](http://arxiv.org/pdf/2510.14450v1)

Authors: François Durand

Previous studies have shown that Instant-Runoff Voting (IRV) is highly
resistant to coalitional manipulation (CM), though the theoretical reasons for
this remain unclear. To address this gap, we analyze the susceptibility to CM
of three major voting rules-Plurality, Two-Round System, and IRV-within the
Perturbed Culture model. Our findings reveal that each rule undergoes a phase
transition at a critical value theta\_c of the concentration of preferences:
the probability of CM for large electorates converges exponentially fast to 1
below theta\_c and to 0 above theta\_c. We introduce the Super Condorcet Winner
(SCW), showing that its presence is a key factor of IRV's resistance to
coalitional manipulation, both theoretically and empirically. Notably, we use
this notion to prove that for IRV, theta\_c = 0, making it resistant to CM with
even minimal preference concentration.

### 2. [Co-Investment under Revenue Uncertainty Based on Stochastic Coalitional Game Theory](http://arxiv.org/pdf/2510.14555v1)

Authors: Amal Sakr, Andrea Araldo, Tijani Chahed, Daniel Kofman

The introduction of new services, such as Mobile Edge Computing (MEC),
requires a massive investment that cannot be assumed by a single stakeholder,
for instance the Infrastructure Provider (InP). Service Providers (SPs) however
also have an interest in the deployment of such services. We hence propose a
co-investment scheme in which all stakeholders, i.e., the InP and the SPs, form
the so-called grand coalition composed of all the stakeholders with the aim of
sharing costs and revenues and maximizing their payoffs. The challenge comes
from the fact that future revenues are uncertain. We devise in this case a
novel stochastic coalitional game formulation which builds upon robust game
theory and derive a lower bound on the probability of the stability of the
grand coalition, wherein no player can be better off outside of it. In the
presence of some correlated fluctuations of revenues however, stability can be
too conservative. In this case, we make use also of profitability, in which
payoffs of players are non-negative, as a necessary condition for
co-investment. The proposed framework is showcased for MEC deployment, where
computational resources need to be deployed in nodes at the edge of a
telecommunication network. Numerical results show high lower bound on the
probability of stability when the SPs' revenues are of similar magnitude and
the investment period is sufficiently long, even with high levels of
uncertainty. In the case where revenues are highly variable however, the lower
bound on stability can be trivially low whereas co-investment is still
profitable.

### 3. [Learnable Mixed Nash Equilibria are Collectively Rational](http://arxiv.org/pdf/2510.14907v1)

Authors: Geelon So, Yi-An Ma

We extend the study of learning in games to dynamics that exhibit
non-asymptotic stability. We do so through the notion of uniform stability,
which is concerned with equilibria of individually utility-seeking dynamics.
Perhaps surprisingly, it turns out to be closely connected to economic
properties of collective rationality. Under mild non-degeneracy conditions and
up to strategic equivalence, if a mixed equilibrium is not uniformly stable,
then it is not weakly Pareto optimal: there is a way for all players to improve
by jointly deviating from the equilibrium. On the other hand, if it is locally
uniformly stable, then the equilibrium must be weakly Pareto optimal. Moreover,
we show that uniform stability determines the last-iterate convergence behavior
for the family of incremental smoothed best-response dynamics, used to model
individual and corporate behaviors in the markets. Unlike dynamics around
strict equilibria, which can stabilize to socially-inefficient solutions,
individually utility-seeking behaviors near mixed Nash equilibria lead to
collective rationality.

### 4. [The Bidding Games: Reinforcement Learning for MEV Extraction on Polygon Blockchain](http://arxiv.org/pdf/2510.14642v1)

Authors: Andrei Seoev, Leonid Gremyachikh, Anastasiia Smirnova, Yash Madhwal, Alisa Kalacheva, Dmitry Belousov, Ilia Zubov, Aleksei Smirnov, Denis Fedyanin, Vladimir Gorgadze, Yury Yanovich

In blockchain networks, the strategic ordering of transactions within blocks
has emerged as a significant source of profit extraction, known as Maximal
Extractable Value (MEV). The transition from spam-based Priority Gas Auctions
to structured auction mechanisms like Polygon Atlas has transformed MEV
extraction from public bidding wars into sealed-bid competitions under extreme
time constraints. While this shift reduces network congestion, it introduces
complex strategic challenges where searchers must make optimal bidding
decisions within a sub-second window without knowledge of competitor behavior
or presence. Traditional game-theoretic approaches struggle in this
high-frequency, partially observable environment due to their reliance on
complete information and static equilibrium assumptions. We present a
reinforcement learning framework for MEV extraction on Polygon Atlas and make
three contributions: (1) A novel simulation environment that accurately models
the stochastic arrival of arbitrage opportunities and probabilistic competition
in Atlas auctions; (2) A PPO-based bidding agent optimized for real-time
constraints, capable of adaptive strategy formulation in continuous action
spaces while maintaining production-ready inference speeds; (3) Empirical
validation demonstrating our history-conditioned agent captures 49\% of
available profits when deployed alongside existing searchers and 81\% when
replacing the market leader, significantly outperforming static bidding
strategies. Our work establishes that reinforcement learning provides a
critical advantage in high-frequency MEV environments where traditional
optimization methods fail, offering immediate value for industrial participants
and protocol designers alike.

### 5. [Online Proportional Apportionment](http://arxiv.org/pdf/2510.14752v1)

Authors: Javier Cembrano, Jose Correa, Svenja M. Griesbach, Victor Verdugo

Traditionally, the problem of apportioning the seats of a legislative body
has been viewed as a one-shot process with no dynamic considerations. While
this approach is reasonable for some settings, dynamic aspects play an
important role in many others. We initiate the study of apportionment problems
in an online setting. Specifically, we introduce a framework for proportional
apportionment with no information about the future. In this model, time is
discrete and there are $n$ parties that receive a certain share of the votes at
each time step. An online algorithm needs to irrevocably assign a prescribed
number of seats at each time, ensuring that each party receives its fractional
share rounded up or down, and that the cumulative number of seats allocated to
each party remains close to its cumulative share up to that time.
  We study deterministic and randomized online apportionment methods. For
deterministic methods, we construct a family of adversarial instances that
yield a lower bound, linear in $n$, on the worst-case deviation between the
seats allocated to a party and its cumulative share. We show that this bound is
best possible and is matched by a natural greedy method. As a consequence, a
method guaranteeing that the cumulative number of seats assigned to each party
up to any step equals its cumulative share rounded up or down (global quota)
exists if and only if $n\leq 3$. Then, we turn to randomized allocations and
show that, for $n\leq 3$, we can randomize over methods satisfying global quota
with the additional guarantee that each party receives, in expectation, its
proportional share in every step. Our proof is constructive: Any method
satisfying these properties can be obtained from a flow on a recursively
constructed network. We showcase the applicability of our results to obtain
approximate solutions in the context of online dependent rounding procedures.

### 6. [Strategic Behavior in Crowdfunding: Insights from a Large-Scale Online Experiment](http://arxiv.org/pdf/2510.14872v1)

Authors: Din Amir, Bar Hoter, Moran Koren

This study examines strategic behavior in crowdfunding using a large-scale
online experiment. Building on the model of Arieli et. al 2023, we test
predictions about risk aversion (i.e., opting out despite seeing a positive
private signal) and mutual insurance (i.e., opting in despite seeing a negative
private signal) in a static, single-shot crowdfunding game, focusing on
informational incentives rather than dynamic effects. Our results validate key
theoretical predictions: crowdfunding mechanisms induce distinct strategic
behaviors compared to voting, where participants are more likely to follow
private signals (odds ratio: 0.139, $p < 0.001$). Additionally, the study
demonstrates that higher signal accuracy (85\% vs. 55\%) decreases risk
aversion (odds ratio: 0.414, $p = 0.024$) but increases reliance on mutual
insurance (odds ratio: 2.532, $p = 0.026$). However, contrary to theory,
increasing the required participation threshold (50\% to 80\%) amplifies risk
aversion (odds ratio: 3.251, $p = 0.005$), which, pending further
investigation, may indicate cognitive constraints.
  Furthermore, we show that while mutual insurance supports participation, it
may hinder information aggregation, particularly as signal accuracy increases.
These findings advance crowdfunding theory by confirming the impact of
informational incentives and identifying behavioral deviations that challenge
standard models, offering insights for platform design and mechanism
refinement.

### Human-Computer Interaction

### 1. [VisAider: AI-Assisted Context-Aware Visualization Support for Data Presentations](http://arxiv.org/pdf/2510.14247v1)

Authors: Kentaro Takahira, Yuki Ueno

Effective real-time data presentation is essential in small-group interactive
contexts, where discussions evolve dynamically and presenters must adapt
visualizations to shifting audience interests. However, most existing
interactive visualization systems rely on fixed mappings between user actions
and visualization commands, limiting their ability to support richer operations
such as changing visualization types, adjusting data transformations, or
incorporating additional datasets on the fly during live presentations. This
work-in-progress paper presents VisAider, an AI-assisted interactive data
presentation prototype that continuously analyzes the live presentation
context, including the available dataset, active visualization, ongoing
conversation, and audience profile, to generate ranked suggestions for relevant
visualization aids. Grounded in a formative study with experienced data
analysts, we identified key challenges in adapting visual content in real time
and distilled design considerations to guide system development. A prototype
implementation demonstrates the feasibility of this approach in simulated
scenarios, and preliminary testing highlights challenges in inferring
appropriate data transformations, resolving ambiguous visualization tasks, and
achieving low-latency responsiveness. Ongoing work focuses on addressing these
limitations, integrating the system into presentation environments, and
preparing a summative user study to evaluate usability and communicative
impact.

### 2. [TapNav: Adaptive Spatiotactile Screen Readers for Tactually Guided Touchscreen Interactions for Blind and Low Vision People](http://arxiv.org/pdf/2510.14267v1)

Authors: Ricardo Gonzalez, Fannie Liu, Blair MacIntyre, David Saffo

Screen readers are audio-based software that Blind and Low Vision (BLV)
people use to interact with computing devices, such as tablets and smartphones.
Although this technology has significantly improved the accessibility of
touchscreen devices, the sequential nature of audio limits the bandwidth of
information users can receive and process. We introduce TapNav, an adaptive
spatiotactile screen reader prototype developed to interact with touchscreen
interfaces spatially. TapNav's screen reader provides adaptive auditory
feedback that, in combination with a tactile overlay, conveys spatial
information and location of interface elements on-screen. We evaluated TapNav
with 12 BLV users who interacted with TapNav to explore a data visualization
and interact with a bank transactions application. Our qualitative findings
show that touch points and spatially constrained navigation helped users
anticipate outcomes for faster exploration, and offload cognitive load to
touch. We provide design guidelines for creating tactile overlays for adaptive
spatiotactile screen readers and discuss their generalizability beyond our
exploratory data analysis and everyday application navigation scenarios.

### 3. [GenLARP: Enabling Immersive Live Action Role-Play through LLM-Generated Worlds and Characters](http://arxiv.org/pdf/2510.14277v1)

Authors: Yichen Yu, Yifan Jiang, Mandy Lui, Qiao Jin

We introduce GenLARP, a virtual reality (VR) system that transforms
personalized stories into immersive live action role-playing (LARP)
experiences. GenLARP enables users to act as both creators and players,
allowing them to design characters based on their descriptions and live in the
story world. Generative AI and agents powered by Large Language Models (LLMs)
enrich these experiences.

### 4. [ReUseIt: Synthesizing Reusable AI Agent Workflows for Web Automation](http://arxiv.org/pdf/2510.14308v1)

Authors: Yimeng Liu, Misha Sra, Jeevana Priya Inala, Chenglong Wang

AI-powered web agents have the potential to automate repetitive tasks, such
as form filling, information retrieval, and scheduling, but they struggle to
reliably execute these tasks without human intervention, requiring users to
provide detailed guidance during every run. We address this limitation by
automatically synthesizing reusable workflows from an agent's successful and
failed attempts. These workflows incorporate execution guards that help agents
detect and fix errors while keeping users informed of progress and issues. Our
approach enables agents to successfully complete repetitive tasks of the same
type with minimal intervention, increasing the success rates from 24.2% to
70.1% across fifteen tasks. To evaluate this approach, we invited nine users
and found that our agent helped them complete web tasks with a higher success
rate and less guidance compared to two baseline methods, as well as allowed
users to easily monitor agent behavior and understand failures.

### 5. [Two Explorative Studies on Tangible Augmented Reality for Neurodevelopmental Disorders](http://arxiv.org/pdf/2510.14598v1)

Authors: Francesco Vona, Giulia Valcamonica, Franca Garzotto

Tangible Augmented Reality (TAR) is an interaction paradigm that integrates
physical and digital worlds to create immersive, interactive experiences. This
paper explores two TAR applications, Holomarket and Along the Oceanic Flow
(ATOF), and presents insights from two exploratory studies evaluating their
usability and likeability among individuals with neurodevelopmental disorders
(NDD). Holomarket is designed to simulate a supermarket shopping experience,
helping users develop essential life skills such as item selection, basic
arithmetic, and money handling. Participants interacted with augmented food
items and a smart cash register, navigating a virtual supermarket environment.
While participants enjoyed the realistic setting and tangible interactions,
some usability challenges, such as difficulty manipulating virtual objects and
discomfort with prolonged headset use, were noted. ATOF transforms the user
environment into an oceanic world, where participants use a dolphin-shaped
smart object to complete tasks like collecting items and solving puzzles. This
application aims to improve motor coordination and cognitive skills.
Participants appreciated the immersive experience, the customizable tasks, and
the tangible dolphin interface. However, some faced difficulties interacting
with specific virtual elements. Overall, both applications demonstrated
potential as therapeutic tools for NDD, offering engaging and immersive
experiences. Despite some usability challenges and hardware limitations, the
positive feedback suggests that TAR could play a crucial role in future
therapeutic interventions. Further research is needed to refine these
applications and enhance user interaction and comfort.

### 6. [Sales Skills Training in Virtual Reality: An evaluation utilizing CAVE and Virtual Avatars](http://arxiv.org/pdf/2510.14603v1)

Authors: Francesco Vona, Michael Stern, Navid Ashrafi, Julia Schorlemmer, Jessica Stemann, Jan-Niklas Voigt-Antons

This study investigates the potential of virtual reality (VR) for enhancing
sales skills training using a Cave Automatic Virtual Environment (CAVE). VR
technology enables users to practice interpersonal and negotiation skills in
controlled, immersive environments that mimic real-world scenarios. In this
study, participants engaged in sales simulations set in a virtual dealership,
interacting with avatars in different work settings and with various
communication styles. The research employed a within-subjects experimental
design involving 20 university students. Each participant experienced four
distinct sales scenarios randomized for environmental and customer conditions.
Training effectiveness was assessed using validated metrics alongside custom
experience questions. Findings revealed consistent user experience and presence
across all scenarios, with no significant differences detected based on
communication styles or environmental conditions. The study highlights the
advantages of semi-immersive VR systems for collaborative learning, peer
feedback, and realistic training environments. However, further research is
recommended to refine VR designs, improve engagement, and maximize skills
transfer to real-world applications.

### 7. [Exploring the Effects of Different Asymmetric Game Designs on User Experience in Collaborative Virtual Reality](http://arxiv.org/pdf/2510.14607v1)

Authors: Francesco Vona, Evelyn Romanjuk, Sina Hinzmann, Julia Schorlemmer, Navid Ashrafi, Jan-Niklas Voigt-Antons

The risk of isolation in virtual reality (VR) stems from the immersive nature
of the technology. VR can transport users to entirely virtual environments,
often disconnecting them from the physical world and real-life interactions.
Asymmetric multiplayer options have been explored to address this issue and
encourage social interaction by requiring players to communicate and
collaborate to achieve common objectives. Nevertheless, research on
implementing these designs and their effects is limited, mainly due to the
novelty of multiplayer VR gaming. This article investigates how different game
design approaches affect the player experience during an asymmetric multiplayer
VR game. Four versions of a VR experience were created and tested in a study
involving 74 participants. Each version differs in terms of the sharing of
virtual environments (shared vs separated) and the players' dependency on the
experience (mutual vs unidirectional). The results showed that variations in
game design influenced aspects of the player experience, such as system
usability, pragmatic UX quality, immersion control, and intrinsic motivation.
Notably, the player roles and the co-presence in the virtual environment did
not simultaneously impact these aspects, suggesting that the degree to which
players depend on each other changes the player experience.

### 8. [Dude, Where's My (Autonomous) Car? Defining an Accessible Description Logic for Blind and Low Vision Travelers Using Autonomous Vehicles](http://arxiv.org/pdf/2510.14911v1)

Authors: Paul D. S. Fink, Justin R. Brown, Rachel Coombs, Emily A. Hamby, Kyle J. James, Aisha Harris, Jacob Bond, Morgan E. Andrulis, Nicholas A. Giudice

Purpose: Autonomous vehicles (AVs) are becoming a promising transportation
solution for blind and low-vision (BLV) travelers, offering the potential for
greater independent mobility. This paper explores the information needs of BLV
users across multiple steps of the transportation journey, including finding
and navigating to, entering, and exiting vehicles independently.
  Methods: A survey with 202 BLV respondents and interviews with 12 BLV
individuals revealed the perspectives of BLV end-users and informed the
sequencing of natural language information required for successful travel.
Whereas the survey identified key information needs across the three trip
segments, the interviews helped prioritize how that information should be
presented in a sequence of accessible descriptions to travelers.
  Results: Taken together, the survey and interviews reveal that BLV users
prioritize knowing the vehicle's make and model and how to find the correct
vehicle during the navigation phase. They also emphasize the importance of
confirmations about the vehicle's destination and onboard safety features upon
entering the vehicle. While exiting, BLV users value information about hazards
and obstacles, as well as knowing which side of the vehicle to exit.
Furthermore, results highlight that BLV travelers desire using their own
smartphone devices when receiving information from AVs and prefer audio-based
interaction.
  Conclusion: The findings from this research contribute a structured framework
for delivering trip-related information to BLV users, useful for designers
incorporating natural language descriptions tailored to each travel segment.
This work offers important contributions for sequencing transportation-related
descriptions throughout the AV journey, ultimately enhancing the mobility and
independence of BLV individuals.

### 9. [An Active Inference Model of Mouse Point-and-Click Behaviour](http://arxiv.org/pdf/2510.14611v1)

Authors: Markus Klar, Sebastian Stein, Fraser Paterson, John H. Williamson, Roderick Murray-Smith

We explore the use of Active Inference (AIF) as a computational user model
for spatial pointing, a key problem in Human-Computer Interaction (HCI). We
present an AIF agent with continuous state, action, and observation spaces,
performing one-dimensional mouse pointing and clicking. We use a simple
underlying dynamic system to model the mouse cursor dynamics with realistic
perceptual delay. In contrast to previous optimal feedback control-based
models, the agent's actions are selected by minimizing Expected Free Energy,
solely based on preference distributions over percepts, such as observing
clicking a button correctly. Our results show that the agent creates plausible
pointing movements and clicks when the cursor is over the target, with similar
end-point variance to human users. In contrast to other models of pointing, we
incorporate fully probabilistic, predictive delay compensation into the agent.
The agent shows distinct behaviour for differing target difficulties without
the need to retune system parameters, as done in other approaches. We discuss
the simulation results and emphasize the challenges in identifying the correct
configuration of an AIF agent interacting with continuous systems.

### 10. [Beyond Hallucinations: The Illusion of Understanding in Large Language Models](http://arxiv.org/pdf/2510.14665v1)

Authors: Rikard Rosenbacke, Carl Rosenbacke, Victor Rosenbacke, Martin McKee

Large language models (LLMs) are becoming deeply embedded in human
communication and decision-making, yet they inherit the ambiguity, bias, and
lack of direct access to truth inherent in language itself. While their outputs
are fluent, emotionally resonant, and coherent, they are generated through
statistical prediction rather than grounded reasoning. This creates the risk of
hallucination, responses that sound convincing but lack factual validity.
Building on Geoffrey Hinton's observation that AI mirrors human intuition
rather than reasoning, this paper argues that LLMs operationalize System 1
cognition at scale: fast, associative, and persuasive, but without reflection
or falsification. To address this, we introduce the Rose-Frame, a
three-dimensional framework for diagnosing cognitive and epistemic drift in
human-AI interaction. The three axes are: (i) Map vs. Territory, which
distinguishes representations of reality (epistemology) from reality itself
(ontology); (ii) Intuition vs. Reason, drawing on dual-process theory to
separate fast, emotional judgments from slow, reflective thinking; and (iii)
Conflict vs. Confirmation, which examines whether ideas are critically tested
through disagreement or simply reinforced through mutual validation. Each
dimension captures a distinct failure mode, and their combination amplifies
misalignment. Rose-Frame does not attempt to fix LLMs with more data or rules.
Instead, it offers a reflective tool that makes both the model's limitations
and the user's assumptions visible, enabling more transparent and critically
aware AI deployment. It reframes alignment as cognitive governance: intuition,
whether human or artificial, must remain governed by human reason. Only by
embedding reflective, falsifiable oversight can we align machine fluency with
human understanding.

### Information Retrieval

### 1. [Synergistic Integration and Discrepancy Resolution of Contextualized Knowledge for Personalized Recommendation](http://arxiv.org/pdf/2510.14257v1)

Authors: Lingyu Mu, Hao Deng, Haibo Xing, Kaican Lin, Zhitong Zhu, Yu Zhang, Xiaoyi Zeng, Zhengxiao Liu, Zheng Lin, Jinxin Hu

The integration of large language models (LLMs) into recommendation systems
has revealed promising potential through their capacity to extract world
knowledge for enhanced reasoning capabilities. However, current methodologies
that adopt static schema-based prompting mechanisms encounter significant
limitations: (1) they employ universal template structures that neglect the
multi-faceted nature of user preference diversity; (2) they implement
superficial alignment between semantic knowledge representations and behavioral
feature spaces without achieving comprehensive latent space integration. To
address these challenges, we introduce CoCo, an end-to-end framework that
dynamically constructs user-specific contextual knowledge embeddings through a
dual-mechanism approach. Our method realizes profound integration of semantic
and behavioral latent dimensions via adaptive knowledge fusion and
contradiction resolution modules. Experimental evaluations across diverse
benchmark datasets and an enterprise-level e-commerce platform demonstrate
CoCo's superiority, achieving a maximum 8.58% improvement over seven
cutting-edge methods in recommendation accuracy. The framework's deployment on
a production advertising system resulted in a 1.91% sales growth, validating
its practical effectiveness. With its modular design and model-agnostic
architecture, CoCo provides a versatile solution for next-generation
recommendation systems requiring both knowledge-enhanced reasoning and
personalized adaptation.

### 2. [Large Reasoning Embedding Models: Towards Next-Generation Dense Retrieval Paradigm](http://arxiv.org/pdf/2510.14321v1)

Authors: Jianting Tang, Dongshuai Li, Tao Wen, Fuyu Lv, Dan Ou, Linli Xu

In modern e-commerce search systems, dense retrieval has become an
indispensable component. By computing similarities between query and item
(product) embeddings, it efficiently selects candidate products from
large-scale repositories. With the breakthroughs in large language models
(LLMs), mainstream embedding models have gradually shifted from BERT to LLMs
for more accurate text modeling. However, these models still adopt
direct-embedding methods, and the semantic accuracy of embeddings remains
inadequate. Therefore, contrastive learning is heavily employed to achieve
tight semantic alignment between positive pairs. Consequently, such models tend
to capture statistical co-occurrence patterns in the training data, biasing
them toward shallow lexical and semantic matches. For difficult queries
exhibiting notable lexical disparity from target items, the performance
degrades significantly. In this work, we propose the Large Reasoning Embedding
Model (LREM), which novelly integrates reasoning processes into representation
learning. For difficult queries, LREM first conducts reasoning to achieve a
deep understanding of the original query, and then produces a
reasoning-augmented query embedding for retrieval. This reasoning process
effectively bridges the semantic gap between original queries and target items,
significantly improving retrieval accuracy. Specifically, we adopt a two-stage
training process: the first stage optimizes the LLM on carefully curated
Query-CoT-Item triplets with SFT and InfoNCE losses to establish preliminary
reasoning and embedding capabilities, and the second stage further refines the
reasoning trajectories via reinforcement learning (RL). Extensive offline and
online experiments validate the effectiveness of LREM, leading to its
deployment on China's largest e-commerce platform since August 2025.

### 3. [Ensembling Multiple Hallucination Detectors Trained on VLLM Internal Representations](http://arxiv.org/pdf/2510.14330v1)

Authors: Yuto Nakamizo, Ryuhei Miyazato, Hikaru Tanabe, Ryuta Yamakura, Kiori Hatanaka

This paper presents the 5th place solution by our team, y3h2, for the Meta
CRAG-MM Challenge at KDD Cup 2025. The CRAG-MM benchmark is a visual question
answering (VQA) dataset focused on factual questions about images, including
egocentric images. The competition was contested based on VQA accuracy, as
judged by an LLM-based automatic evaluator. Since incorrect answers result in
negative scores, our strategy focused on reducing hallucinations from the
internal representations of the VLM. Specifically, we trained logistic
regression-based hallucination detection models using both the hidden_state and
the outputs of specific attention heads. We then employed an ensemble of these
models. As a result, while our method sacrificed some correct answers, it
significantly reduced hallucinations and allowed us to place among the top
entries on the final leaderboard. For implementation details and code, please
refer to
https://gitlab.aicrowd.com/htanabe/meta-comprehensive-rag-benchmark-starter-kit.

### 4. [MR.Rec: Synergizing Memory and Reasoning for Personalized Recommendation Assistant with LLMs](http://arxiv.org/pdf/2510.14629v1)

Authors: Jiani Huang, Xingchen Zou, Lianghao Xia, Qing Li

The application of Large Language Models (LLMs) in recommender systems faces
key challenges in delivering deep personalization and intelligent reasoning,
especially for interactive scenarios. Current methods are often constrained by
limited context windows and single-turn reasoning, hindering their ability to
capture dynamic user preferences and proactively reason over recommendation
contexts. To address these limitations, we propose MR.Rec, a novel framework
that synergizes memory and reasoning for LLM-based recommendations. To achieve
personalization, we develop a comprehensive Retrieval-Augmented Generation
(RAG) system that efficiently indexes and retrieves relevant external memory to
enhance LLM personalization capabilities. Furthermore, to enable the synergy
between memory and reasoning, our RAG system goes beyond conventional
query-based retrieval by integrating reasoning enhanced memory retrieval.
Finally, we design a reinforcement learning framework that trains the LLM to
autonomously learn effective strategies for both memory utilization and
reasoning refinement. By combining dynamic memory retrieval with adaptive
reasoning, this approach ensures more accurate, context-aware, and highly
personalized recommendations. Extensive experiments demonstrate that MR.Rec
significantly outperforms state-of-the-art baselines across multiple metrics,
validating its efficacy in delivering intelligent and personalized
recommendations. We will release code and data upon paper notification.

### 5. [Dataset Pruning in RecSys and ML: Best Practice or Mal-Practice?](http://arxiv.org/pdf/2510.14704v1)

Authors: Leonie Winter

Offline evaluations in recommender system research depend heavily on
datasets, many of which are pruned, such as the widely used MovieLens
collections. This thesis examines the impact of data pruning - specifically,
removing users with fewer than a specified number of interactions - on both
dataset characteristics and algorithm performance. Five benchmark datasets were
analysed in both their unpruned form and at five successive pruning levels (5,
10, 20, 50, 100). For each coreset, we examined structural and distributional
characteristics and trained and tested eleven representative algorithms. To
further assess if pruned datasets lead to artificially inflated performance
results, we also evaluated models trained on the pruned train sets but tested
on unpruned data. Results show that commonly applied core pruning can be highly
selective, leaving as little as 2% of the original users in some datasets.
Traditional algorithms achieved higher nDCG@10 scores when both training and
testing on pruned data; however, this advantage largely disappeared when
evaluated on unpruned test sets. Across all algorithms, performance declined
with increasing pruning levels when tested on unpruned data, highlighting the
impact of dataset reduction on the performance of recommender algorithms.

### 6. [Fantastic (small) Retrievers and How to Train Them: mxbai-edge-colbert-v0 Tech Report](http://arxiv.org/pdf/2510.14880v1)

Authors: Rikiya Takehi, Benjamin Clavié, Sean Lee, Aamir Shakir

In this work, we introduce mxbai-edge-colbert-v0 models, at two different
parameter counts: 17M and 32M. As part of our research, we conduct numerous
experiments to improve retrieval and late-interaction models, which we intend
to distill into smaller models as proof-of-concepts. Our ultimate aim is to
support retrieval at all scales, from large-scale retrieval which lives in the
cloud to models that can run locally, on any device. mxbai-edge-colbert-v0 is a
model that we hope will serve as a solid foundation backbone for all future
experiments, representing the first version of a long series of small
proof-of-concepts. As part of the development of mxbai-edge-colbert-v0, we
conducted multiple ablation studies, of which we report the results. In terms
of downstream performance, mxbai-edge-colbert-v0 is a particularly capable
small model, outperforming ColBERTv2 on common short-text benchmarks (BEIR) and
representing a large step forward in long-context tasks, with unprecedented
efficiency.

### 7. [Large Scale Retrieval for the LinkedIn Feed using Causal Language Models](http://arxiv.org/pdf/2510.14223v1)

Authors: Sudarshan Srinivasa Ramanujam, Antonio Alonso, Saurabh Kataria, Siddharth Dangi, Akhilesh Gupta, Birjodh Singh Tiwana, Manas Somaiya, Luke Simon, David Byrne, Sojeong Ha, Sen Zhou, Andrei Akterskii, Zhanglong Liu, Samira Sriram, Crescent Xiong, Zhoutao Pei, Angela Shao, Alex Li, Annie Xiao, Caitlin Kolb, Thomas Kistler, Zach Moore, Hamed Firooz

In large scale recommendation systems like the LinkedIn Feed, the retrieval
stage is critical for narrowing hundreds of millions of potential candidates to
a manageable subset for ranking. LinkedIn's Feed serves suggested content from
outside of the member's network (based on the member's topical interests),
where 2000 candidates are retrieved from a pool of hundreds of millions
candidate with a latency budget of a few milliseconds and inbound QPS of
several thousand per second. This paper presents a novel retrieval approach
that fine-tunes a large causal language model (Meta's LLaMA 3) as a dual
encoder to generate high quality embeddings for both users (members) and
content (items), using only textual input. We describe the end to end pipeline,
including prompt design for embedding generation, techniques for fine-tuning at
LinkedIn's scale, and infrastructure for low latency, cost effective online
serving. We share our findings on how quantizing numerical features in the
prompt enables the information to get properly encoded in the embedding,
facilitating greater alignment between the retrieval and ranking layer. The
system was evaluated using offline metrics and an online A/B test, which showed
substantial improvements in member engagement. We observed significant gains
among newer members, who often lack strong network connections, indicating that
high-quality suggested content aids retention. This work demonstrates how
generative language models can be effectively adapted for real time, high
throughput retrieval in industrial applications.

### 8. [Rethinking Schema Linking: A Context-Aware Bidirectional Retrieval Approach for Text-to-SQL](http://arxiv.org/pdf/2510.14296v1)

Authors: Md Mahadi Hasan Nahid, Davood Rafiei, Weiwei Zhang, Yong Zhang

Schema linking -- the process of aligning natural language questions with
database schema elements -- is a critical yet underexplored component of
Text-to-SQL systems. While recent methods have focused primarily on improving
SQL generation, they often neglect the retrieval of relevant schema elements,
which can lead to hallucinations and execution failures. In this work, we
propose a context-aware bidirectional schema retrieval framework that treats
schema linking as a standalone problem. Our approach combines two complementary
strategies: table-first retrieval followed by column selection, and
column-first retrieval followed by table selection. It is further augmented
with techniques such as question decomposition, keyword extraction, and
keyphrase extraction. Through comprehensive evaluations on challenging
benchmarks such as BIRD and Spider, we demonstrate that our method
significantly improves schema recall while reducing false positives. Moreover,
SQL generation using our retrieved schema consistently outperforms full-schema
baselines and closely approaches oracle performance, all without requiring
query refinement. Notably, our method narrows the performance gap between full
and perfect schema settings by 50\%. Our findings highlight schema linking as a
powerful lever for enhancing Text-to-SQL accuracy and efficiency.

### 9. [Acquisition of interpretable domain information during brain MR image harmonization for content-based image retrieval](http://arxiv.org/pdf/2510.14535v1)

Authors: Keima Abe, Hayato Muraki, Shuhei Tomoshige, Kenichi Oishi, Hitoshi Iyatomi

Medical images like MR scans often show domain shifts across imaging sites
due to scanner and protocol differences, which degrade machine learning
performance in tasks such as disease classification. Domain harmonization is
thus a critical research focus. Recent approaches encode brain images
$\boldsymbol{x}$ into a low-dimensional latent space $\boldsymbol{z}$, then
disentangle it into $\boldsymbol{z_u}$ (domain-invariant) and
$\boldsymbol{z_d}$ (domain-specific), achieving strong results. However, these
methods often lack interpretability$-$an essential requirement in medical
applications$-$leaving practical issues unresolved. We propose
Pseudo-Linear-Style Encoder Adversarial Domain Adaptation (PL-SE-ADA), a
general framework for domain harmonization and interpretable representation
learning that preserves disease-relevant information in brain MR images.
PL-SE-ADA includes two encoders $f_E$ and $f_{SE}$ to extract
$\boldsymbol{z_u}$ and $\boldsymbol{z_d}$, a decoder to reconstruct the image
$f_D$, and a domain predictor $g_D$. Beyond adversarial training between the
encoder and domain predictor, the model learns to reconstruct the input image
$\boldsymbol{x}$ by summing reconstructions from $\boldsymbol{z_u}$ and
$\boldsymbol{z_d}$, ensuring both harmonization and informativeness. Compared
to prior methods, PL-SE-ADA achieves equal or better performance in image
reconstruction, disease classification, and domain recognition. It also enables
visualization of both domain-independent brain features and domain-specific
components, offering high interpretability across the entire framework.

### 10. [Multimodal RAG for Unstructured Data:Leveraging Modality-Aware Knowledge Graphs with Hybrid Retrieval](http://arxiv.org/pdf/2510.14592v1)

Authors: Rashmi R, Vidyadhar Upadhya

Current Retrieval-Augmented Generation (RAG) systems primarily operate on
unimodal textual data, limiting their effectiveness on unstructured multimodal
documents. Such documents often combine text, images, tables, equations, and
graphs, each contributing unique information. In this work, we present a
Modality-Aware Hybrid retrieval Architecture (MAHA), designed specifically for
multimodal question answering with reasoning through a modality-aware knowledge
graph. MAHA integrates dense vector retrieval with structured graph traversal,
where the knowledge graph encodes cross-modal semantics and relationships. This
design enables both semantically rich and context-aware retrieval across
diverse modalities. Evaluations on multiple benchmark datasets demonstrate that
MAHA substantially outperforms baseline methods, achieving a ROUGE-L score of
0.486, providing complete modality coverage. These results highlight MAHA's
ability to combine embeddings with explicit document structure, enabling
effective multimodal retrieval. Our work establishes a scalable and
interpretable retrieval framework that advances RAG systems by enabling
modality-aware reasoning over unstructured multimodal data.

### Machine Learning

### 1. [Contrastive Diffusion Alignment: Learning Structured Latents for Controllable Generation](http://arxiv.org/pdf/2510.14190v1)

Authors: Ruchi Sandilya, Sumaira Perez, Charles Lynch, Lindsay Victoria, Benjamin Zebley, Derrick Matthew Buchanan, Mahendra T. Bhati, Nolan Williams, Timothy J. Spellman, Faith M. Gunning, Conor Liston, Logan Grosenick

Diffusion models excel at generation, but their latent spaces are not
explicitly organized for interpretable control. We introduce ConDA (Contrastive
Diffusion Alignment), a framework that applies contrastive learning within
diffusion embeddings to align latent geometry with system dynamics. Motivated
by recent advances showing that contrastive objectives can recover more
disentangled and structured representations, ConDA organizes diffusion latents
such that traversal directions reflect underlying dynamical factors. Within
this contrastively structured space, ConDA enables nonlinear trajectory
traversal that supports faithful interpolation, extrapolation, and controllable
generation. Across benchmarks in fluid dynamics, neural calcium imaging,
therapeutic neurostimulation, and facial expression, ConDA produces
interpretable latent representations with improved controllability compared to
linear traversals and conditioning-based baselines. These results suggest that
diffusion latents encode dynamics-relevant structure, but exploiting this
structure requires latent organization and traversal along the latent manifold.

### 2. [When Flatness Does (Not) Guarantee Adversarial Robustness](http://arxiv.org/pdf/2510.14231v1)

Authors: Nils Philipp Walter, Linara Adilova, Jilles Vreeken, Michael Kamp

Despite their empirical success, neural networks remain vulnerable to small,
adversarial perturbations. A longstanding hypothesis suggests that flat minima,
regions of low curvature in the loss landscape, offer increased robustness.
While intuitive, this connection has remained largely informal and incomplete.
By rigorously formalizing the relationship, we show this intuition is only
partially correct: flatness implies local but not global adversarial
robustness. To arrive at this result, we first derive a closed-form expression
for relative flatness in the penultimate layer, and then show we can use this
to constrain the variation of the loss in input space. This allows us to
formally analyze the adversarial robustness of the entire network. We then show
that to maintain robustness beyond a local neighborhood, the loss needs to
curve sharply away from the data manifold. We validate our theoretical
predictions empirically across architectures and datasets, uncovering the
geometric structure that governs adversarial vulnerability, and linking
flatness to model confidence: adversarial examples often lie in large, flat
regions where the model is confidently wrong. Our results challenge simplified
views of flatness and provide a nuanced understanding of its role in
robustness.

### 3. [A Physics Prior-Guided Dual-Stream Attention Network for Motion Prediction of Elastic Bragg Breakwaters](http://arxiv.org/pdf/2510.14250v1)

Authors: Lianzi Jiang, Jianxin Zhang, Xinyu Han, Huanhe Dong, Xiangrong Wang

Accurate motion response prediction for elastic Bragg breakwaters is critical
for their structural safety and operational integrity in marine environments.
However, conventional deep learning models often exhibit limited generalization
capabilities when presented with unseen sea states. These deficiencies stem
from the neglect of natural decay observed in marine systems and inadequate
modeling of wave-structure interaction (WSI). To overcome these challenges,
this study proposes a novel Physics Prior-Guided Dual-Stream Attention Network
(PhysAttnNet). First, the decay bidirectional self-attention (DBSA) module
incorporates a learnable temporal decay to assign higher weights to recent
states, aiming to emulate the natural decay phenomenon. Meanwhile, the phase
differences guided bidirectional cross-attention (PDG-BCA) module explicitly
captures the bidirectional interaction and phase relationship between waves and
the structure using a cosine-based bias within a bidirectional
cross-computation paradigm. These streams are synergistically integrated
through a global context fusion (GCF) module. Finally, PhysAttnNet is trained
with a hybrid time-frequency loss that jointly minimizes time-domain prediction
errors and frequency-domain spectral discrepancies. Comprehensive experiments
on wave flume datasets demonstrate that PhysAttnNet significantly outperforms
mainstream models. Furthermore,cross-scenario generalization tests validate the
model's robustness and adaptability to unseen environments, highlighting its
potential as a framework to develop predictive models for complex systems in
ocean engineering.

### 4. [Generalist vs Specialist Time Series Foundation Models: Investigating Potential Emergent Behaviors in Assessing Human Health Using PPG Signals](http://arxiv.org/pdf/2510.14254v1)

Authors: Saurabh Kataria, Yi Wu, Zhaoliang Chen, Hyunjung Gloria Kwak, Yuhao Xu, Lovely Yeswanth Panchumarthi, Ran Xiao, Jiaying Lu, Ayca Ermis, Anni Zhao, Runze Yan, Alex Federov, Zewen Liu, Xu Wu, Wei Jin, Carl Yang, Jocelyn Grunwell, Stephanie R. Brown, Amit Shah, Craig Jabaley, Tim Buchman, Sivasubramanium V Bhavani, Randall J. Lee, Xiao Hu

Foundation models are large-scale machine learning models that are
pre-trained on massive amounts of data and can be adapted for various
downstream tasks. They have been extensively applied to tasks in Natural
Language Processing and Computer Vision with models such as GPT, BERT, and
CLIP. They are now also increasingly gaining attention in time-series analysis,
particularly for physiological sensing. However, most time series foundation
models are specialist models - with data in pre-training and testing of the
same type, such as Electrocardiogram, Electroencephalogram, and
Photoplethysmogram (PPG). Recent works, such as MOMENT, train a generalist time
series foundation model with data from multiple domains, such as weather,
traffic, and electricity. This paper aims to conduct a comprehensive
benchmarking study to compare the performance of generalist and specialist
models, with a focus on PPG signals. Through an extensive suite of total 51
tasks covering cardiac state assessment, laboratory value estimation, and
cross-modal inference, we comprehensively evaluate both models across seven
dimensions, including win score, average performance, feature quality, tuning
gain, performance variance, transferability, and scalability. These metrics
jointly capture not only the models' capability but also their adaptability,
robustness, and efficiency under different fine-tuning strategies, providing a
holistic understanding of their strengths and limitations for diverse
downstream scenarios. In a full-tuning scenario, we demonstrate that the
specialist model achieves a 27% higher win score. Finally, we provide further
analysis on generalization, fairness, attention visualizations, and the
importance of training data choice.

### 5. [Stable Prediction of Adverse Events in Medical Time-Series Data](http://arxiv.org/pdf/2510.14286v1)

Authors: Mayank Keoliya, Seewon Choi, Rajeev Alur, Mayur Naik, Eric Wong

Early event prediction (EEP) systems continuously estimate a patient's
imminent risk to support clinical decision-making. For bedside trust, risk
trajectories must be accurate and temporally stable, shifting only with new,
relevant evidence. However, current benchmarks (a) ignore stability of risk
scores and (b) evaluate mainly on tabular inputs, leaving trajectory behavior
untested. To address this gap, we introduce CAREBench, an EEP benchmark that
evaluates deployability using multi-modal inputs-tabular EHR, ECG waveforms,
and clinical text-and assesses temporal stability alongside predictive
accuracy. We propose a stability metric that quantifies short-term variability
in per-patient risk and penalizes abrupt oscillations based on local-Lipschitz
constants. CAREBench spans six prediction tasks such as sepsis onset and
compares classical learners, deep sequence models, and zero-shot LLMs. Across
tasks, existing methods, especially LLMs, struggle to jointly optimize accuracy
and stability, with notably poor recall at high-precision operating points.
These results highlight the need for models that produce evidence-aligned,
stable trajectories to earn clinician trust in continuous monitoring settings.
(Code: https://github.com/SeewonChoi/CAREBench.)

### 6. [Enhancing Time-Series Anomaly Detection by Integrating Spectral-Residual Bottom-Up Attention with Reservoir Computing](http://arxiv.org/pdf/2510.14287v1)

Authors: Hayato Nihei, Sou Nobukawa, Yusuke Sakemi, Kazuyuki Aihara

Reservoir computing (RC) establishes the basis for the processing of
time-series data by exploiting the high-dimensional spatiotemporal response of
a recurrent neural network to an input signal. In particular, RC trains only
the output layer weights. This simplicity has drawn attention especially in
Edge Artificial Intelligence (AI) applications. Edge AI enables time-series
anomaly detection in real time, which is important because detection delays can
lead to serious incidents. However, achieving adequate anomaly-detection
performance with RC alone may require an unacceptably large reservoir on
resource-constrained edge devices. Without enlarging the reservoir, attention
mechanisms can improve accuracy, although they may require substantial
computation and undermine the learning efficiency of RC. In this study, to
improve the anomaly detection performance of RC without sacrificing learning
efficiency, we propose a spectral residual RC (SR-RC) that integrates the
spectral residual (SR) method - a learning-free, bottom-up attention mechanism
- with RC. We demonstrated that SR-RC outperformed conventional RC and
logistic-regression models based on values extracted by the SR method across
benchmark tasks and real-world time-series datasets. Moreover, because the SR
method, similarly to RC, is well suited for hardware implementation, SR-RC
suggests a practical direction for deploying RC as Edge AI for time-series
anomaly detection.

### 7. [LLM-ERM: Sample-Efficient Program Learning via LLM-Guided Search](http://arxiv.org/pdf/2510.14331v1)

Authors: Shivam Singhal, Eran Malach, Tomaso Poggio, Tomer Galanti

We seek algorithms for program learning that are both sample-efficient and
computationally feasible. Classical results show that targets admitting short
program descriptions (e.g., with short ``python code'') can be learned with a
``small'' number of examples (scaling with the size of the code) via
length-first program enumeration, but the search is exponential in description
length. Consequently, Gradient-based training avoids this cost yet can require
exponentially many samples on certain short-program families.
  To address this gap, we introduce LLM-ERM, a propose-and-verify framework
that replaces exhaustive enumeration with an LLM-guided search over candidate
programs while retaining ERM-style selection on held-out data. Specifically, we
draw $k$ candidates with a pretrained reasoning-augmented LLM, compile and
check each on the data, and return the best verified hypothesis, with no
feedback, adaptivity, or gradients. Theoretically, we show that coordinate-wise
online mini-batch SGD requires many samples to learn certain short programs.
{\em Empirically, LLM-ERM solves tasks such as parity variants, pattern
matching, and primality testing with as few as 200 samples, while SGD-trained
transformers overfit even with 100,000 samples}. These results indicate that
language-guided program synthesis recovers much of the statistical efficiency
of finite-class ERM while remaining computationally tractable, offering a
practical route to learning succinct hypotheses beyond the reach of
gradient-based training.

### 8. [DARTS-GT: Differentiable Architecture Search for Graph Transformers with Quantifiable Instance-Specific Interpretability Analysis](http://arxiv.org/pdf/2510.14336v1)

Authors: Shruti Sarika Chakraborty, Peter Minary

Graph Transformers (GTs) have emerged as powerful architectures for
graph-structured data, yet remain constrained by rigid designs and lack
quantifiable interpretability. Current state-of-the-art GTs commit to fixed GNN
types across all layers, missing potential benefits of depth-specific component
selection, while their complex architectures become opaque where performance
gains cannot be distinguished between meaningful patterns and spurious
correlations. We redesign GT attention through asymmetry, decoupling structural
encoding from feature representation: queries derive from node features while
keys and values come from GNN transformations. Within this framework, we use
Differentiable ARchiTecture Search (DARTS) to select optimal GNN operators at
each layer, enabling depth-wise heterogeneity inside transformer attention
itself (DARTS-GT). To understand discovered architectures, we develop the first
quantitative interpretability framework for GTs through causal ablation. Our
metrics (Head-deviation, Specialization, and Focus), identify which heads and
nodes drive predictions while enabling model comparison. Experiments across
eight benchmarks show DARTS-GT achieves state-of-the-art on four datasets while
remaining competitive on others, with discovered architectures revealing
dataset-specific patterns. Our interpretability analysis reveals that visual
attention salience and causal importance do not always correlate, indicating
widely used visualization approaches may miss components that actually matter.
Crucially, heterogeneous architectures found by DARTS-GT consistently produced
more interpretable models than baselines, establishing that Graph Transformers
need not choose between performance and interpretability.

### 9. [MergeMoE: Efficient Compression of MoE Models via Expert Output Merging](http://arxiv.org/pdf/2510.14436v1)

Authors: Ruijie Miao, Yilun Yao, Zihan Wang, Zhiming Wang, Bairen Yi, LingJun Liu, Yikai Zhao, Tong Yang

The Mixture-of-Experts (MoE) technique has proven to be a promising solution
to efficiently scale the model size, which has been widely applied in recent
LLM advancements. However, the substantial memory overhead of MoE models has
made their compression an important research direction. In this work, we
provide a theoretical analysis of expert merging, a recently proposed technique
for compressing MoE models. Rather than interpreting expert merging from the
conventional perspective of parameter aggregation, we approach it from the
perspective of merging experts' outputs. Our key insight is that the merging
process can be interpreted as inserting additional matrices into the forward
computation, which naturally leads to an optimization formulation. Building on
this analysis, we introduce MergeMoE, a method that leverages mathematical
optimization to construct the compression matrices. We evaluate MergeMoE on
multiple MoE models and show that our algorithm consistently outperforms the
baselines with the same compression ratios.

### 10. [Learning to Undo: Rollback-Augmented Reinforcement Learning with Reversibility Signals](http://arxiv.org/pdf/2510.14503v1)

Authors: Andrejs Sorstkins, Omer Tariq, Muhammad Bilal

This paper proposes a reversible learning framework to improve the robustness
and efficiency of value based Reinforcement Learning agents, addressing
vulnerability to value overestimation and instability in partially irreversible
environments. The framework has two complementary core mechanisms: an
empirically derived transition reversibility measure called Phi of s and a, and
a selective state rollback operation. We introduce an online per state action
estimator called Phi that quantifies the likelihood of returning to a prior
state within a fixed horizon K. This measure is used to adjust the penalty term
during temporal difference updates dynamically, integrating reversibility
awareness directly into the value function. The system also includes a
selective rollback operator. When an action yields an expected return markedly
lower than its instantaneous estimated value and violates a predefined
threshold, the agent is penalized and returns to the preceding state rather
than progressing. This interrupts sub optimal high risk trajectories and avoids
catastrophic steps. By combining reversibility aware evaluation with targeted
rollback, the method improves safety, performance, and stability. In the
CliffWalking v0 domain, the framework reduced catastrophic falls by over 99.8
percent and yielded a 55 percent increase in mean episode return. In the Taxi
v3 domain, it suppressed illegal actions by greater than or equal to 99.9
percent and achieved a 65.7 percent improvement in cumulative reward, while
also sharply reducing reward variance in both environments. Ablation studies
confirm that the rollback mechanism is the critical component underlying these
safety and performance gains, marking a robust step toward safe and reliable
sequential decision making.

### Neural and Evolutionary Computing

### 1. [Spiking Neural Network Architecture Search: A Survey](http://arxiv.org/pdf/2510.14235v1)

Authors: Kama Svoboda, Tosiron Adegbija

This survey paper presents a comprehensive examination of Spiking Neural
Network (SNN) architecture search (SNNaS) from a unique hardware/software
co-design perspective. SNNs, inspired by biological neurons, have emerged as a
promising approach to neuromorphic computing. They offer significant advantages
in terms of power efficiency and real-time resource-constrained processing,
making them ideal for edge computing and IoT applications. However, designing
optimal SNN architectures poses significant challenges, due to their inherent
complexity (e.g., with respect to training) and the interplay between hardware
constraints and SNN models. We begin by providing an overview of SNNs,
emphasizing their operational principles and key distinctions from traditional
artificial neural networks (ANNs). We then provide a brief overview of the
state of the art in NAS for ANNs, highlighting the challenges of directly
applying these approaches to SNNs. We then survey the state-of-the-art in
SNN-specific NAS approaches. Finally, we conclude with insights into future
research directions for SNN research, emphasizing the potential of
hardware/software co-design in unlocking the full capabilities of SNNs. This
survey aims to serve as a valuable resource for researchers and practitioners
in the field, offering a holistic view of SNNaS and underscoring the importance
of a co-design approach to harness the true potential of neuromorphic
computing.

### 2. [SHaRe-SSM: An Oscillatory Spiking Neural Network for Target Variable Modeling in Long Sequences](http://arxiv.org/pdf/2510.14386v1)

Authors: Kartikay Agrawal, Abhijeet Vikram, Vedant Sharma, Vaishnavi N., Ayon Borthakur

In recent years, with the emergence of large models, there has been a
significant interest in spiking neural networks (SNNs) primarily due to their
energy efficiency, multiplication-free, and sparse event-based deep learning.
Similarly, state space models (SSMs) in varying designs have evolved as a
powerful alternative to transformers for target modeling in long sequences,
thereby overcoming the quadratic dependence on sequence length of a
transformer. Inspired by this progress, we here design SHaRe-SSM (Spiking
Harmonic Resonate and Fire State Space Model), for target variable modeling
(including both classification and regression) for very-long-range sequences.
Our second-order spiking SSM, on average, performs better than transformers or
first-order SSMs while circumventing multiplication operations, making it ideal
for resource-constrained applications. The proposed block consumes $73 \times$
less energy than second-order ANN-based SSMs for an 18k sequence, while
retaining performance. To ensure learnability over the long-range sequences, we
propose exploiting the stable and efficient implementation of the dynamical
system using parallel scans. Moreover, for the first time, we propose a
kernel-based spiking regressor using resonate and fire neurons for very
long-range sequences. Our network shows superior performance on even a 50k
sequence while being significantly energy-efficient. In addition, we conducted
a systematic analysis of the impact of heterogeneity, dissipation, and
conservation in resonate-and-fire SSMs.

### 3. [Online Reliable Anomaly Detection via Neuromorphic Sensing and Communications](http://arxiv.org/pdf/2510.14688v1)

Authors: Junya Shiraishi, Jiechen Chen, Osvaldo Simeone, Petar Popovski

This paper proposes a low-power online anomaly detection framework based on
neuromorphic wireless sensor networks, encompassing possible use cases such as
brain-machine interfaces and remote environmental monitoring. In the considered
system, a central reader node actively queries a subset of neuromorphic sensor
nodes (neuro-SNs) at each time frame. The neuromorphic sensors are
event-driven, producing spikes in correspondence to relevant changes in the
monitored system. The queried neuro-SNs respond to the reader with impulse
radio (IR) transmissions that directly encode the sensed local events. The
reader processes these event-driven signals to determine whether the monitored
environment is in a normal or anomalous state, while rigorously controlling the
false discovery rate (FDR) of detections below a predefined threshold. The
proposed approach employs an online hypothesis testing method with e-values to
maintain FDR control without requiring knowledge of the anomaly rate, and it
dynamically optimizes the sensor querying strategy by casting it as a best-arm
identification problem in a multi-armed bandit framework. Extensive performance
evaluation demonstrates that the proposed method can reliably detect anomalies
under stringent FDR requirements, while efficiently scheduling sensor
communications and achieving low detection latency.

### 4. [Provable Unlearning with Gradient Ascent on Two-Layer ReLU Neural Networks](http://arxiv.org/pdf/2510.14844v1)

Authors: Odelia Melamed, Gilad Yehudai, Gal Vardi

Machine Unlearning aims to remove specific data from trained models,
addressing growing privacy and ethical concerns. We provide a theoretical
analysis of a simple and widely used method - gradient ascent - used to reverse
the influence of a specific data point without retraining from scratch.
Leveraging the implicit bias of gradient descent towards solutions that satisfy
the Karush-Kuhn-Tucker (KKT) conditions of a margin maximization problem, we
quantify the quality of the unlearned model by evaluating how well it satisfies
these conditions w.r.t. the retained data. To formalize this idea, we propose a
new success criterion, termed \textbf{$(\epsilon, \delta, \tau)$-successful}
unlearning, and show that, for both linear models and two-layer neural networks
with high dimensional data, a properly scaled gradient-ascent step satisfies
this criterion and yields a model that closely approximates the retrained
solution on the retained data. We also show that gradient ascent performs
successful unlearning while still preserving generalization in a synthetic
Gaussian-mixture setting.

### Networking and Internet Architecture

### 1. [Energy-Latency Optimization for Dynamic 5G Mobile Radio Access Networks](http://arxiv.org/pdf/2510.14214v1)

Authors: Gabriela N. Caspa H., Carlos A. Astudillo, Nelson L. S. da Fonseca

In 5G networks, base station (BS) disaggregation and new services present
challenges in radio access network (RAN) configuration, particularly in meeting
their bandwidth and latency constraints. The BS disaggregation is enabled by
functional splitting (FS), which distributes the RAN functions in processing
nodes and alleviates latency and bandwidth requirements in the fronthaul (FH).
Besides network performance, energy consumption is a critical concern for
mobile network operators (MNO), since RAN operation constitutes a major portion
of their operational expenses (OPEX). RAN configuration optimization is
essential to balance service performance with cost-effective energy
consumption. In this paper, we propose a mixed-integer linear programming
(MILP) model formulated with three objective functions: (i) minimizing
fronthaul (FH) latency, (ii) minimizing energy consumption, and (iii) a
bi-objective optimization that jointly balances both latency and energy
consumption. The model determines the optimal FS option, RAN function
placement, and routing for eMBB, URLLC, and mMTC slices. Although prior studies
have addressed RAN configuration either from an energy minimization or latency
reduction perspective, few have considered both aspects in realistic scenarios.
Our evaluation spans different topologies, accounts for variations in
aggregated gNB demand, explores diverse FS combinations, and incorporates Time
Sensitive Networking (TSN) modeling for latency analysis, as it is also crucial
in RAN performance. Given that MILP's execution time can be significant, we
propose a heuristic algorithm that adheres to RAN constraints. Our results
reveal a trade-off between latency and energy consumption, highlighting the
need for dynamic RAN reconfiguration. These insights provide a foundation to
optimize existing and future RAN deployments.

### 2. [Automated Extraction of Protocol State Machines from 3GPP Specifications with Domain-Informed Prompts and LLM Ensembles](http://arxiv.org/pdf/2510.14348v1)

Authors: Miao Zhang, Runhan Feng, Hongbo Tang, Yu Zhao, Jie Yang, Hang Qiu, Qi Liu

Mobile telecommunication networks are foundational to global infrastructure
and increasingly support critical sectors such as manufacturing,
transportation, and healthcare. The security and reliability of these networks
are essential, yet depend heavily on accurate modeling of underlying protocols
through state machines. While most prior work constructs such models manually
from 3GPP specifications, this process is labor-intensive, error-prone, and
difficult to maintain due to the complexity and frequent updates of the
specifications. Recent efforts using natural language processing have shown
promise, but remain limited in handling the scale and intricacy of cellular
protocols. In this work, we propose SpecGPT, a novel framework that leverages
large language models (LLMs) to automatically extract protocol state machines
from 3GPP documents. SpecGPT segments technical specifications into meaningful
paragraphs, applies domain-informed prompting with chain-of-thought reasoning,
and employs ensemble methods to enhance output reliability. We evaluate SpecGPT
on three representative 5G protocols (NAS, NGAP, and PFCP) using manually
annotated ground truth, and show that it outperforms existing approaches,
demonstrating the effectiveness of LLMs for protocol modeling at scale.

### 3. [Intelligent Dynamic Handover via AI-assisted Signal Quality Prediction in 6G Multi-RAT Networks](http://arxiv.org/pdf/2510.14832v1)

Authors: Maria Lamprini A. Bartsioka, Anastasios Giannopoulos, Sotirios Spantideas

The emerging paradigm of 6G multiple Radio Access Technology (multi-RAT)
networks, where cellular and Wireless Fidelity (WiFi) transmitters coexist,
requires mobility decisions that remain reliable under fast channel dynamics,
interference, and heterogeneous coverage. Handover in multi-RAT deployments is
still highly reactive and event-triggered, relying on instantaneous
measurements and threshold events. This work proposes a Machine Learning
(ML)-assisted Predictive Conditional Handover (P-CHO) framework based on a
model-driven and short-horizon signal quality forecasts. We present a
generalized P-CHO sequence workflow orchestrated by a RAT Steering Controller,
which standardizes data collection, parallel per-RAT predictions, decision
logic with hysteresis-based conditions, and CHO execution. Considering a
realistic multi-RAT environment, we train RAT-aware Long Short Term Memory
(LSTM) networks to forecast the signal quality indicators of mobile users along
randomized trajectories. The proposed P-CHO models are trained and evaluated
under different channel models for cellular and IEEE 802.11 WiFi integrated
coverage. We study the impact of hyperparameter tuning of LSTM models under
different system settings, and compare direct multi-step versus recursive P-CHO
variants. Comparisons against baseline predictors are also carried out.
Finally, the proposed P-CHO is tested under soft and hard handover settings,
showing that hysteresis-enabled P-CHO scheme is able to reduce handover
failures and ping-pong events. Overall, the proposed P-CHO framework can enable
accurate, low-latency, and proactive handovers suitable for ML-assisted
handover steering in 6G multi-RAT deployments.

### 4. [Decoherence-Aware Entangling and Swapping Strategy Optimization for Entanglement Routing in Quantum Networks](http://arxiv.org/pdf/2510.14912v1)

Authors: Shao-Min Huang, Cheng-Yang Cheng, Ming-Huang Chien, Jian-Jhih Kuo, Chih-Yu Wang

Quantum teleportation enables high-security communications through end-to-end
quantum entangled pairs. End-to-end entangled pairs are created by using
swapping processes to consume short entangled pairs and generate long pairs.
However, due to environmental interference, entangled pairs decohere over time,
resulting in low fidelity. Thus, generating entangled pairs at the right time
is crucial. Moreover, the swapping process also causes additional fidelity
loss. To this end, this paper presents a short time slot protocol, where a time
slot can only accommodate a process. It has a more flexible arrangement of
entangling and swapping processes than the traditional long time slot protocol.
It raises a new optimization problem TETRIS for finding strategies of
entangling and swapping for each request to maximize the fidelity sum of all
accepted requests. To solve the TETRIS, we design two novel algorithms with
different optimization techniques. Finally, the simulation results manifest
that our algorithms can outperform the existing methods by up to 60 ~ 78% in
general, and by 20 ~ 75% even under low entangling probabilities.

### Robotics

### 1. [Risk-Aware Reinforcement Learning with Bandit-Based Adaptation for Quadrupedal Locomotion](http://arxiv.org/pdf/2510.14338v1)

Authors: Yuanhong Zeng, Anushri Dixit

In this work, we study risk-aware reinforcement learning for quadrupedal
locomotion. Our approach trains a family of risk-conditioned policies using a
Conditional Value-at-Risk (CVaR) constrained policy optimization technique that
provides improved stability and sample efficiency. At deployment, we adaptively
select the best performing policy from the family of policies using a
multi-armed bandit framework that uses only observed episodic returns, without
any privileged environment information, and adapts to unknown conditions on the
fly. Hence, we train quadrupedal locomotion policies at various levels of
robustness using CVaR and adaptively select the desired level of robustness
online to ensure performance in unknown environments. We evaluate our method in
simulation across eight unseen settings (by changing dynamics, contacts,
sensing noise, and terrain) and on a Unitree Go2 robot in previously unseen
terrains. Our risk-aware policy attains nearly twice the mean and tail
performance in unseen environments compared to other baselines and our
bandit-based adaptation selects the best-performing risk-aware policy in
unknown terrain within two minutes of operation.

### 2. [Restoring Noisy Demonstration for Imitation Learning With Diffusion Models](http://arxiv.org/pdf/2510.14467v1)

Authors: Shang-Fu Chen, Co Yong, Shao-Hua Sun

Imitation learning (IL) aims to learn a policy from expert demonstrations and
has been applied to various applications. By learning from the expert policy,
IL methods do not require environmental interactions or reward signals.
However, most existing imitation learning algorithms assume perfect expert
demonstrations, but expert demonstrations often contain imperfections caused by
errors from human experts or sensor/control system inaccuracies. To address the
above problems, this work proposes a filter-and-restore framework to best
leverage expert demonstrations with inherent noise. Our proposed method first
filters clean samples from the demonstrations and then learns conditional
diffusion models to recover the noisy ones. We evaluate our proposed framework
and existing methods in various domains, including robot arm manipulation,
dexterous manipulation, and locomotion. The experiment results show that our
proposed framework consistently outperforms existing methods across all the
tasks. Ablation studies further validate the effectiveness of each component
and demonstrate the framework's robustness to different noise types and levels.
These results confirm the practical applicability of our framework to noisy
offline demonstration data.

### 3. [QuASH: Using Natural-Language Heuristics to Query Visual-Language Robotic Maps](http://arxiv.org/pdf/2510.14546v1)

Authors: Matti Pekkanen, Francesco Verdoja, Ville Kyrki

Embeddings from Visual-Language Models are increasingly utilized to represent
semantics in robotic maps, offering an open-vocabulary scene understanding that
surpasses traditional, limited labels. Embeddings enable on-demand querying by
comparing embedded user text prompts to map embeddings via a similarity metric.
The key challenge in performing the task indicated in a query is that the robot
must determine the parts of the environment relevant to the query.
  This paper proposes a solution to this challenge. We leverage
natural-language synonyms and antonyms associated with the query within the
embedding space, applying heuristics to estimate the language space relevant to
the query, and use that to train a classifier to partition the environment into
matches and non-matches. We evaluate our method through extensive experiments,
querying both maps and standard image benchmarks. The results demonstrate
increased queryability of maps and images. Our querying technique is agnostic
to the representation and encoder used, and requires limited training.

### 4. [A Generalized Placeability Metric for Model-Free Unified Pick-and-Place Reasoning](http://arxiv.org/pdf/2510.14584v1)

Authors: Benno Wingender, Nils Dengler, Rohit Menon, Sicong Pan, Maren Bennewitz

To reliably pick and place unknown objects under real-world sensing noise
remains a challenging task, as existing methods rely on strong object priors
(e.g., CAD models), or planar-support assumptions, limiting generalization and
unified reasoning between grasping and placing. In this work, we introduce a
generalized placeability metric that evaluates placement poses directly from
noisy point clouds, without any shape priors. The metric jointly scores
stability, graspability, and clearance. From raw geometry, we extract the
support surfaces of the object to generate diverse candidates for
multi-orientation placement and sample contacts that satisfy collision and
stability constraints. By conditioning grasp scores on each candidate
placement, our proposed method enables model-free unified pick-and-place
reasoning and selects grasp-place pairs that lead to stable, collision-free
placements. On unseen real objects and non-planar object supports, our metric
delivers CAD-comparable accuracy in predicting stability loss and generally
produces more physically plausible placements than learning-based predictors.

### 5. [Proprioceptive Image: An Image Representation of Proprioceptive Data from Quadruped Robots for Contact Estimation Learning](http://arxiv.org/pdf/2510.14612v1)

Authors: Gabriel Fischer Abati, João Carlos Virgolino Soares, Giulio Turrisi, Victor Barasuol, Claudio Semini

This paper presents a novel approach for representing proprioceptive
time-series data from quadruped robots as structured two-dimensional images,
enabling the use of convolutional neural networks for learning
locomotion-related tasks. The proposed method encodes temporal dynamics from
multiple proprioceptive signals, such as joint positions, IMU readings, and
foot velocities, while preserving the robot's morphological structure in the
spatial arrangement of the image. This transformation captures inter-signal
correlations and gait-dependent patterns, providing a richer feature space than
direct time-series processing. We apply this concept in the problem of contact
estimation, a key capability for stable and adaptive locomotion on diverse
terrains. Experimental evaluations on both real-world datasets and simulated
environments show that our image-based representation consistently enhances
prediction accuracy and generalization over conventional sequence-based models,
underscoring the potential of cross-modal encoding strategies for robotic state
learning. Our method achieves superior performance on the contact dataset,
improving contact state accuracy from 87.7% to 94.5% over the recently proposed
MI-HGNN method, using a 15 times shorter window size.

### 6. [Accelerated Multi-Modal Motion Planning Using Context-Conditioned Diffusion Models](http://arxiv.org/pdf/2510.14615v1)

Authors: Edward Sandra, Lander Vanroye, Dries Dirckx, Ruben Cartuyvels, Jan Swevers, Wilm Decré

Classical methods in robot motion planning, such as sampling-based and
optimization-based methods, often struggle with scalability towards
higher-dimensional state spaces and complex environments. Diffusion models,
known for their capability to learn complex, high-dimensional and multi-modal
data distributions, provide a promising alternative when applied to motion
planning problems and have already shown interesting results. However, most of
the current approaches train their model for a single environment, limiting
their generalization to environments not seen during training. The techniques
that do train a model for multiple environments rely on a specific camera to
provide the model with the necessary environmental information and therefore
always require that sensor. To effectively adapt to diverse scenarios without
the need for retraining, this research proposes Context-Aware Motion Planning
Diffusion (CAMPD). CAMPD leverages a classifier-free denoising probabilistic
diffusion model, conditioned on sensor-agnostic contextual information. An
attention mechanism, integrated in the well-known U-Net architecture,
conditions the model on an arbitrary number of contextual parameters. CAMPD is
evaluated on a 7-DoF robot manipulator and benchmarked against state-of-the-art
approaches on real-world tasks, showing its ability to generalize to unseen
environments and generate high-quality, multi-modal trajectories, at a fraction
of the time required by existing methods.

### 7. [Generative Models From and For Sampling-Based MPC: A Bootstrapped Approach For Adaptive Contact-Rich Manipulation](http://arxiv.org/pdf/2510.14643v1)

Authors: Lara Brudermüller, Brandon Hung, Xinghao Zhu, Jiuguang Wang, Nick Hawes, Preston Culbertson, Simon Le Cleac'h

We present a generative predictive control (GPC) framework that amortizes
sampling-based Model Predictive Control (SPC) by bootstrapping it with
conditional flow-matching models trained on SPC control sequences collected in
simulation. Unlike prior work relying on iterative refinement or gradient-based
solvers, we show that meaningful proposal distributions can be learned directly
from noisy SPC data, enabling more efficient and informed sampling during
online planning. We further demonstrate, for the first time, the application of
this approach to real-world contact-rich loco-manipulation with a quadruped
robot. Extensive experiments in simulation and on hardware show that our method
improves sample efficiency, reduces planning horizon requirements, and
generalizes robustly across task variations.

### 8. [Spatially anchored Tactile Awareness for Robust Dexterous Manipulation](http://arxiv.org/pdf/2510.14647v1)

Authors: Jialei Huang, Yang Ye, Yuanqing Gong, Xuezhou Zhu, Yang Gao, Kaifeng Zhang

Dexterous manipulation requires precise geometric reasoning, yet existing
visuo-tactile learning methods struggle with sub-millimeter precision tasks
that are routine for traditional model-based approaches. We identify a key
limitation: while tactile sensors provide rich contact information, current
learning frameworks fail to effectively leverage both the perceptual richness
of tactile signals and their spatial relationship with hand kinematics. We
believe an ideal tactile representation should explicitly ground contact
measurements in a stable reference frame while preserving detailed sensory
information, enabling policies to not only detect contact occurrence but also
precisely infer object geometry in the hand's coordinate system. We introduce
SaTA (Spatially-anchored Tactile Awareness for dexterous manipulation), an
end-to-end policy framework that explicitly anchors tactile features to the
hand's kinematic frame through forward kinematics, enabling accurate geometric
reasoning without requiring object models or explicit pose estimation. Our key
insight is that spatially grounded tactile representations allow policies to
not only detect contact occurrence but also precisely infer object geometry in
the hand's coordinate system. We validate SaTA on challenging dexterous
manipulation tasks, including bimanual USB-C mating in free space, a task
demanding sub-millimeter alignment precision, as well as light bulb
installation requiring precise thread engagement and rotational control, and
card sliding that demands delicate force modulation and angular precision.
These tasks represent significant challenges for learning-based methods due to
their stringent precision requirements. Across multiple benchmarks, SaTA
significantly outperforms strong visuo-tactile baselines, improving success
rates by up to 30 percentage while reducing task completion times by 27
percentage.

### 9. [Leveraging Neural Descriptor Fields for Learning Contact-Aware Dynamic Recovery](http://arxiv.org/pdf/2510.14768v1)

Authors: Fan Yang, Zixuan Huang, Abhinav Kumar, Sergio Aguilera Marinovic, Soshi Iba, Rana Soltani Zarrin, Dmitry Berenson

Real-world dexterous manipulation often encounters unexpected errors and
disturbances, which can lead to catastrophic failures, such as dropping the
manipulated object. To address this challenge, we focus on the problem of
catching a falling object while it remains within grasping range and,
importantly, resetting the system to a configuration favorable for resuming the
primary manipulation task. We propose Contact-Aware Dynamic Recovery (CADRE), a
reinforcement learning framework that incorporates a Neural Descriptor Field
(NDF)-inspired module to extract implicit contact features. Compared to methods
that rely solely on object pose or point cloud input, NDFs can directly reason
about finger-object correspondence and adapt to different object geometries.
Our experiments show that incorporating contact features improves training
efficiency, enhances convergence performance for RL training, and ultimately
leads to more successful recoveries. Additionally, we demonstrate that CADRE
can generalize zero-shot to unseen objects with different geometries.

### 10. [Open TeleDex: A Hardware-Agnostic Teleoperation System for Imitation Learning based Dexterous Manipulation](http://arxiv.org/pdf/2510.14771v1)

Authors: Xu Chi, Chao Zhang, Yang Su, Lingfeng Dou, Fujia Yang, Jiakuo Zhao, Haoyu Zhou, Xiaoyou Jia, Yong Zhou, Shan An

Accurate and high-fidelity demonstration data acquisition is a critical
bottleneck for deploying robot Imitation Learning (IL) systems, particularly
when dealing with heterogeneous robotic platforms. Existing teleoperation
systems often fail to guarantee high-precision data collection across diverse
types of teleoperation devices. To address this, we developed Open TeleDex, a
unified teleoperation framework engineered for demonstration data collection.
Open TeleDex specifically tackles the TripleAny challenge, seamlessly
supporting any robotic arm, any dexterous hand, and any external input device.
Furthermore, we propose a novel hand pose retargeting algorithm that
significantly boosts the interoperability of Open TeleDex, enabling robust and
accurate compatibility with an even wider spectrum of heterogeneous master and
slave equipment. Open TeleDex establishes a foundational, high-quality, and
publicly available platform for accelerating both academic research and
industry development in complex robotic manipulation and IL.

### Software Engineering

### 1. [A Hybrid, Knowledge-Guided Evolutionary Framework for Personalized Compiler Auto-Tuning](http://arxiv.org/pdf/2510.14292v1)

Authors: Haolin Pan, Hongbin Zhang, Mingjie Xing, Yanjun Wu

Compiler pass auto-tuning is critical for enhancing software performance, yet
finding the optimal pass sequence for a specific program is an NP-hard problem.
Traditional, general-purpose optimization flags like -O3 and -Oz adopt a
one-size-fits-all approach, often failing to unlock a program's full
performance potential. To address this challenge, we propose a novel Hybrid,
Knowledge-Guided Evolutionary Framework. This framework intelligently guides
online, personalized optimization using knowledge extracted from a large-scale
offline analysis phase. During the offline stage, we construct a comprehensive
compilation knowledge base composed of four key components: (1) Pass Behavioral
Vectors to quantitatively capture the effectiveness of each optimization; (2)
Pass Groups derived from clustering these vectors based on behavior similarity;
(3) a Synergy Pass Graph to model beneficial sequential interactions; and (4) a
library of Prototype Pass Sequences evolved for distinct program types. In the
online stage, a bespoke genetic algorithm leverages this rich knowledge base
through specially designed, knowledge-infused genetic operators. These
operators transform the search by performing semantically-aware recombination
and targeted, restorative mutations. On a suite of seven public datasets, our
framework achieves an average of 11.0% additional LLVM IR instruction reduction
over the highly-optimized opt -Oz baseline, demonstrating its state-of-the-art
capability in discovering personalized, high-performance optimization
sequences.

### 2. [A Systematic Study of Time Limit Exceeded Errors in Online Programming Assignments](http://arxiv.org/pdf/2510.14339v1)

Authors: Jialu Zhang, Jialiang Gu, Wangmeiyu Zhang, José Pablo Cambronero, John Kolesar, Ruzica Piskac, Daming Li, Hanyuan Shi

Online programming platforms such as Codeforces and LeetCode attract millions
of users seeking to learn to program or refine their skills for industry
interviews. A major challenge for these users is the Time Limit Exceeded (TLE)
error, triggered when a program exceeds the execution time bound. Although
designed as a performance safeguard, TLE errors are difficult to resolve: error
messages provide no diagnostic insight, platform support is minimal, and
existing debugging tools offer little help. As a result, many users abandon
their submissions after repeated TLE failures.
  This paper presents the first large-scale empirical study of TLE errors in
online programming. We manually analyzed 1000 Codeforces submissions with TLE
errors, classified their root causes, and traced how users attempted to fix
them. Our analysis shows that TLE errors often arise not only from inefficient
algorithms but also from infinite loops, improper data structure use, and
inefficient I/O, challenging the conventional view that TLEs are purely
performance issues.
  Guided by these findings, we introduce Nettle, the first automated repair
tool specifically designed for TLE errors, and Nettle-Eval, the first framework
for evaluating TLE repairs. Integrating LLMs with targeted automated feedback
generated by the compiler and test cases, Nettle produces small, correct code
edits that eliminate TLEs while preserving functionality. Evaluated on the same
1000 real-world cases, Nettle achieves a 98.5% fix rate, far exceeding the
strongest LLM baseline, and all of its repairs pass both Nettle-Eval and the
platform's official checker, confirming the reliability of our framework.

### 3. [PathFix: Automated Program Repair with Expected Path](http://arxiv.org/pdf/2510.14341v1)

Authors: Xu He, Shu Wang, Kun Sun

Automated program repair (APR) techniques are effective in fixing inevitable
defects in software, enhancing development efficiency and software robustness.
However, due to the difficulty of generating precise specifications, existing
APR methods face two main challenges: generating too many plausible patch
candidates and overfitting them to partial test cases. To tackle these
challenges, we introduce a new APR method named PathFix, which leverages
path-sensitive constraints extracted from correct execution paths to generate
patches for repairing buggy code. It is based on one observation: if a buggy
program is repairable, at least one expected path is supposed to replace the
fault path in the patched program. PathFix operates in four main steps. First,
it traces fault paths reaching the fault output in the buggy program. Second,
it derives expected paths by analyzing the desired correct output on the
control flow graph, where an expected path defines how a feasible patch leads
to the correct execution. Third, PathFix generates and evaluates patches by
solving state constraints along the expected path. Fourth, we validate the
correctness of the generated patch. To further enhance repair performance and
mitigate scalability issues introduced by path-sensitive analysis, we integrate
a large language model (LLM) into our framework. Experimental results show that
PathFix outperforms existing solutions, particularly in handling complex
program structures such as loops and recursion.

### 4. [Towards Automated Governance: A DSL for Human-Agent Collaboration in Software Projects](http://arxiv.org/pdf/2510.14465v1)

Authors: Adem Ait, Gwendal Jouneaux, Javier Luis Cánovas Izquierdo, Jordi Cabot

The stakeholders involved in software development are becoming increasingly
diverse, with both human contributors from varied backgrounds and AI-powered
agents collaborating together in the process. This situation presents unique
governance challenges, particularly in Open-Source Software (OSS) projects,
where explicit policies are often lacking or unclear. This paper presents the
vision and foundational concepts for a novel Domain-Specific Language (DSL)
designed to define and enforce rich governance policies in systems involving
diverse stakeholders, including agents. This DSL offers a pathway towards more
robust, adaptable, and ultimately automated governance, paving the way for more
effective collaboration in software projects, especially OSS ones.

### 5. [Software Testing Education and Industry Needs - Report from the ENACTEST EU Project](http://arxiv.org/pdf/2510.14625v1)

Authors: Mehrdad Saadatmand, Abbas Khan, Beatriz Marin, Ana C. R Paiva, Nele Van Asch, Graham Moran, Felix Cammaerts, Monique Snoeck, Alexandra Mendes

The evolving landscape of software development demands that software testers
continuously adapt to new tools, practices, and acquire new skills. This study
investigates software testing competency needs in industry, identifies
knowledge gaps in current testing education, and highlights competencies and
gaps not addressed in academic literature. This is done by conducting two focus
group sessions and interviews with professionals across diverse domains,
including railway industry, healthcare, and software consulting and performing
a curated small-scale scoping review. The study instrument, co-designed by
members of the ENACTEST project consortium, was developed collaboratively and
refined through multiple iterations to ensure comprehensive coverage of
industry needs and educational gaps. In particular, by performing a thematic
qualitative analysis, we report our findings and observations regarding:
professional training methods, challenges in offering training in industry,
different ways of evaluating the quality of training, identified knowledge gaps
with respect to academic education and industry needs, future needs and trends
in testing education, and knowledge transfer methods within companies. Finally,
the scoping review results confirm knowledge gaps in areas such as AI testing,
security testing and soft skills.

### 6. [ATGen: Adversarial Reinforcement Learning for Test Case Generation](http://arxiv.org/pdf/2510.14635v1)

Authors: Qingyao Li, Xinyi Dai, Weiwen Liu, Xiangyang Li, Yasheng Wang, Ruiming Tang, Yong Yu, Weinan Zhang

Large Language Models (LLMs) excel at code generation, yet their outputs
often contain subtle bugs, for which effective test cases are a critical
bottleneck. Existing test generation methods, whether based on prompting or
supervised fine-tuning, rely on static datasets. This imposes a
``fixed-difficulty ceiling'', fundamentally limiting their ability to uncover
novel or more complex bugs beyond their training scope. To overcome this, we
introduce ATGen, a framework that trains a test case generator via adversarial
reinforcement learning. ATGen pits a test generator against an adversarial code
generator that continuously crafts harder bugs to evade the current policy.
This dynamic loop creates a curriculum of increasing difficulty challenging
current policy. The test generator is optimized via Reinforcement Learning (RL)
to jointly maximize ``Output Accuracy'' and ``Attack Success'', enabling it to
learn a progressively stronger policy that breaks the fixed-difficulty ceiling
of static training. Extensive experiments demonstrate that ATGen significantly
outperforms state-of-the-art baselines. We further validate its practical
utility, showing it serves as both a more effective filter for Best-of-N
inference and a higher-quality reward source for training code generation
models. Our work establishes a new, dynamic paradigm for improving the
reliability of LLM-generated code.

### 7. [Caruca: Effective and Efficient Specification Mining for Opaque Software Components](http://arxiv.org/pdf/2510.14279v1)

Authors: Evangelos Lamprou, Seong-Heon Jung, Mayank Keoliya, Lukas Lazarek, Konstantinos Kallas, Michael Greenberg, Nikos Vasilakis

A wealth of state-of-the-art systems demonstrate impressive improvements in
performance, security, and reliability on programs composed of opaque
components, such as Unix shell commands. To reason about commands, these
systems require partial specifications. However, creating such specifications
is a manual, laborious, and error-prone process, limiting the practicality of
these systems. This paper presents Caruca, a system for automatic specification
mining for opaque commands. To overcome the challenge of language diversity
across commands, Caruca first instruments a large language model to translate a
command's user-facing documentation into a structured invocation syntax. Using
this representation, Caruca explores the space of syntactically valid command
invocations and execution environments. Caruca concretely executes each
command-environment pair, interposing at the system-call and filesystem level
to extract key command properties such as parallelizability and filesystem pre-
and post-conditions. These properties can be exported in multiple specification
formats and are immediately usable by existing systems. Applying Caruca across
60 GNU Coreutils, POSIX, and third-party commands across several
specification-dependent systems shows that Caruca generates correct
specifications for all but one case, completely eliminating manual effort from
the process and currently powering the full specifications for a
state-of-the-art static analysis tool.

### 8. [Match & Mend: Minimally Invasive Local Reassembly for Patching N-day Vulnerabilities in ARM Binaries](http://arxiv.org/pdf/2510.14384v1)

Authors: Sebastian Jänich, Merlin Sievers, Johannes Kinder

Low-cost Internet of Things (IoT) devices are increasingly popular but often
insecure due to poor update regimes. As a result, many devices run outdated and
known-vulnerable versions of open-source software. We address this problem by
proposing to patch IoT firmware at the binary level, without requiring vendor
support. In particular, we introduce minimally invasive local reassembly, a new
technique for automatically patching known (n-day) vulnerabilities in IoT
firmware. Our approach is designed to minimize side effects and reduce the risk
of introducing breaking changes. We systematically evaluate our approach both
on 108 binaries within the controlled environment of the MAGMA benchmarks, as
well as on 30 real-world Linux-based IoT firmware images from the KARONTE
dataset. Our prototype successfully patches 83% of targeted vulnerabilities in
MAGMA and 96% in the firmware dataset.

### 9. [Certifying optimal MEV strategies with Lean](http://arxiv.org/pdf/2510.14480v1)

Authors: Massimo Bartoletti, Riccardo Marchesin, Roberto Zunino

Maximal Extractable Value (MEV) refers to a class of attacks to decentralized
applications where the adversary profits by manipulating the ordering,
inclusion, or exclusion of transactions in a blockchain. Decentralized Finance
(DeFi) protocols are a primary target of these attacks, as their logic depends
critically on transaction sequencing. To date, MEV attacks have already
extracted billions of dollars in value, underscoring their systemic impact on
blockchain security. Verifying the absence of MEV attacks requires determining
suitable upper bounds, i.e. proving that no adversarial strategy can extract
more value (if any) than expected by protocol designers. This problem is
notoriously difficult: the space of adversarial strategies is extremely vast,
making empirical studies and pen-and-paper reasoning insufficiently rigorous.
In this paper, we present the first mechanized formalization of MEV in the Lean
theorem prover. We introduce a methodology to construct machine-checked proofs
of MEV bounds, providing correctness guarantees beyond what is possible with
existing techniques. To demonstrate the generality of our approach, we model
and analyse the MEV of two paradigmatic DeFi protocols. Notably, we develop the
first machine-checked proof of the optimality of sandwich attacks in Automated
Market Makers, a fundamental DeFi primitive.

### 10. [Requirement Identification for Traffic Simulations in Driving Simulators](http://arxiv.org/pdf/2510.14653v1)

Authors: Sven Tarlowski, Lutz Eckstein

This paper addresses the challenge of ensuring realistic traffic conditions
by proposing a methodology that systematically identifies traffic simulation
requirements. Using a structured approach based on sub-goals in each study
phase, specific technical needs are derived for microscopic levels, agent
models, and visual representation. The methodology aims to maintain a high
degree of fidelity, enhancing both the validity of experimental outcomes and
participant engagement. By providing a clear link between study objectives and
traffic simulation design, this approach supports robust automotive development
and testing.

### Social and Information Networks

### 1. [What is missing from this picture? Persistent homology and mixup barcodes as a means of investigating negative embedding space](http://arxiv.org/pdf/2510.14327v1)

Authors: Himanshu Yadav, Thomas Bryan Smith, Peter Bubenik, Christopher McCarty

Recent work in the information sciences, especially informetrics and
scientometrics, has made substantial contributions to the development of new
metrics that eschew the intrinsic biases of citation metrics. This work has
tended to employ either network scientific (topological) approaches to
quantifying the disruptiveness of peer-reviewed research, or topic modeling
approaches to quantifying conceptual novelty. We propose a combination of these
approaches, investigating the prospect of topological data analysis (TDA),
specifically persistent homology and mixup barcodes, as a means of
understanding the negative space among document embeddings generated by topic
models. Using top2vec, we embed documents and topics in n-dimensional space, we
use persistent homology to identify holes in the embedding distribution, and
then use mixup barcodes to determine which holes are being filled by a set of
unobserved publications. In this case, the unobserved publications represent
research that was published before or after the data used to train top2vec. We
investigate the extent that negative embedding space represents missing context
(older research) versus innovation space (newer research), and the extend that
the documents that occupy this space represents integrations of the research
topics on the periphery. Potential applications for this metric are discussed.

### 2. [Strategic Behavior in Crowdfunding: Insights from a Large-Scale Online Experiment](http://arxiv.org/pdf/2510.14872v1)

Authors: Din Amir, Bar Hoter, Moran Koren

This study examines strategic behavior in crowdfunding using a large-scale
online experiment. Building on the model of Arieli et. al 2023, we test
predictions about risk aversion (i.e., opting out despite seeing a positive
private signal) and mutual insurance (i.e., opting in despite seeing a negative
private signal) in a static, single-shot crowdfunding game, focusing on
informational incentives rather than dynamic effects. Our results validate key
theoretical predictions: crowdfunding mechanisms induce distinct strategic
behaviors compared to voting, where participants are more likely to follow
private signals (odds ratio: 0.139, $p < 0.001$). Additionally, the study
demonstrates that higher signal accuracy (85\% vs. 55\%) decreases risk
aversion (odds ratio: 0.414, $p = 0.024$) but increases reliance on mutual
insurance (odds ratio: 2.532, $p = 0.026$). However, contrary to theory,
increasing the required participation threshold (50\% to 80\%) amplifies risk
aversion (odds ratio: 3.251, $p = 0.005$), which, pending further
investigation, may indicate cognitive constraints.
  Furthermore, we show that while mutual insurance supports participation, it
may hinder information aggregation, particularly as signal accuracy increases.
These findings advance crowdfunding theory by confirming the impact of
informational incentives and identifying behavioral deviations that challenge
standard models, offering insights for platform design and mechanism
refinement.

### 3. [Detecting Early and Implicit Suicidal Ideation via Longitudinal and Information Environment Signals on Social Media](http://arxiv.org/pdf/2510.14889v1)

Authors: Soorya Ram Shimgekar, Ruining Zhao, Agam Goyal, Violeta J. Rodriguez, Paul A. Bloom, Hari Sundaram, Koustuv Saha

On social media, many individuals experiencing suicidal ideation (SI) do not
disclose their distress explicitly. Instead, signs may surface indirectly
through everyday posts or peer interactions. Detecting such implicit signals
early is critical but remains challenging. We frame early and implicit SI as a
forward-looking prediction task and develop a computational framework that
models a user's information environment, consisting of both their longitudinal
posting histories as well as the discourse of their socially proximal peers. We
adopted a composite network centrality measure to identify top neighbors of a
user, and temporally aligned the user's and neighbors' interactions --
integrating the multi-layered signals in a fine-tuned DeBERTa-v3 model. In a
Reddit study of 1,000 (500 Case and 500 Control) users, our approach improves
early and implicit SI detection by 15% over individual-only baselines. These
findings highlight that peer interactions offer valuable predictive signals and
carry broader implications for designing early detection systems that capture
indirect as well as masked expressions of risk in online environments.

### Systems and Control

### 1. [High-Resolution PTDF-Based Planning of Storage and Transmission Under High Renewables](http://arxiv.org/pdf/2510.14696v1)

Authors: Kevin Wu, Rabab Haider, Pascal Van Hentenryck

Transmission Expansion Planning (TEP) optimizes power grid upgrades and
investments to ensure reliable, efficient, and cost-effective electricity
delivery while addressing grid constraints. To support growing demand and
renewable energy integration, energy storage is emerging as a pivotal asset
that provides temporal flexibility and alleviates congestion. This paper
develops a multiperiod, two-stage PTDF formulation that co-optimizes
transmission upgrades and storage siting/sizing. To ensure scalability, a
trust-region, multicut Benders scheme warm-started from per-representative-day
optima is proposed. Applied to a 2,000-bus synthetic Texas system under
high-renewable projections, the method attains final optimality gaps below 1%
and yields a plan with storage at about 180 nodes (32% of peak renewable
capacity). These results demonstrate that the proposed PTDF-based methodology
efficiently handles large distributed storage fleets, demonstrating scalability
at high spatial resolution

### 2. [Improved Voltage Regulation with Optimal Design of Decentralized Volt-VAr Control](http://arxiv.org/pdf/2510.14834v1)

Authors: Daniel Russell, Dakota Hamilton, Mads R. Almassalkhi, Hamid R. Ossareh

Integration of distributed energy resources has created a need for
autonomous, dynamic voltage regulation. Decentralized Volt-VAr Control (VVC) of
grid-connected inverters presents a unique opportunity for voltage management
but, if designed poorly, can lead to unstable behavior when in feedback with
the grid. We model the grid-VVC closed-loop dynamics with a linearized power
flow approach, leveraging historical data, which shows improvement over the
commonly used LinDistFlow model. This model is used to design VVC slopes by
minimizing steady-state voltage deviation from the nominal value, subject to a
non-convex spectral radius stability constraint, which has not been previously
implemented within this context. We compare this constraint to existing convex
restrictions and demonstrate, through simulations on a realistic feeder, that
using the spectral radius results in more effective voltage regulation.

### 3. [Dynamic-Key-Aware Co-Simulation Framework for Next Generation of SCADA Systems Encrypted by Quantum-Key-Distribution Techniques](http://arxiv.org/pdf/2510.14838v1)

Authors: Ziqing Zhu

To address growing cybersecurity challenges in modern power dispatch systems,
this paper proposes a multi-layer modeling and optimization framework for SCADA
systems enhanced with quantum key distribution (QKD). While most existing
applications of QKD in the power sector focus on building secure point-to-point
communication tunnels, they rarely consider the system-level coupling between
key dynamics and control scheduling. In contrast, our approach integrates
quantum key generation, consumption, inventory prediction, and control latency
into a unified model, enabling key-aware reconfiguration of SCADA control
chains based on task security demands and real-time resource constraints. To
resolve conflicts in key resource allocation between transmission system
operators (TSOs) and distribution system operators (DSOs), we formulate a
bi-level Stackelberg game and transform it into a mathematical program with
complementarity constraints (MPCC). We further develop an efficient Level
Decomposition-Complementarity Pruning (LD-CP) algorithm to solve the problem.
To support reproducible evaluation, we build an end-to-end co-simulation
platform that integrates physical-layer disruptions via OpenQKD-Sim,
Q3P/IEC-104 protocol stack binding, and real-time control-chain monitoring
through Grafana. Experimental results on the IEEE 39- and 118-bus systems show
that our method increases task success rate by 25%, reduces peak frequency
deviation by 70%, and improves key utilization to 83%. This work lays the
foundation for future quantum-secure control systems in power grid operations.

### 4. [Through-the-Earth Magnetic Induction Communication and Networking: A Comprehensive Survey](http://arxiv.org/pdf/2510.14854v1)

Authors: Honglei Ma, Erwu Liu, Wei Ni, Zhijun Fang, Rui Wang, Yongbin Gao, Dusit Niyato, Ekram Hossain

Magnetic induction (MI) communication (MIC) has emerged as a promising
candidate for underground communication networks due to its excellent
penetration capabilities. Integration with Space-Air-Ground-Underground (SAGUI)
networks in next-generation mobile communication systems requires a
well-defined network architecture. A recent discovery in MIC research, MI fast
fading, remains in its early stages and presents unique challenges. This paper
provides a comprehensive survey on through-the-earth (TTE) MIC, covering MI
applications, channel modeling, point-to-point MIC design, relay techniques,
network frameworks, and emerging technologies. We compare various MIC
applications to highlight TTE-specific challenges and review the principles of
channel modeling, addressing both MI slow fading and MI fast fading, along with
its potential impact on existing MIC theories. We conduct a fine-grained
decomposition of MI channel power gain into four distinct physical parameters,
and propose a novel geometric model to analyze MI fast fading. We also
summarize MI relay techniques, examine crosstalk effects in relay and
high-density networks, and explore key research tasks within the OSI framework
for a holistic MI network protocol in SAGUI. To bridge the gaps identified, we
propose a MIC framework that supports TCP/IP and Linux, enabling full
implementation of existing and emerging MIC solutions. This framework empowers
researchers to leverage Linux resources and deep learning platforms for
accelerated development of MIC in SAGUI networks. Remaining research
challenges, open issues, and promising novel techniques are further identified
to advance MIC research.

### 5. [Further Results on Safety-Critical Stabilization of Force-Controlled Nonholonomic Mobile Robots](http://arxiv.org/pdf/2510.14931v1)

Authors: Bo Wang, Tianyu Han, Guangwei Wang

In this paper, we address the stabilization problem for force-controlled
nonholonomic mobile robots under safety-critical constraints. We propose a
continuous, time-invariant control law based on the gamma m-quadratic
programming (gamma m-QP) framework, which unifies control Lyapunov functions
(CLFs) and control barrier functions (CBFs) to enforce both stability and
safety in the closed-loop system. For the first time, we construct a global,
time-invariant, strict Lyapunov function for the closed-loop nonholonomic
mobile robot system with a nominal stabilization controller in polar
coordinates; this strict Lyapunov function then serves as the CLF in the QP
design. Next, by exploiting the inherent cascaded structure of the vehicle
dynamics, we develop a CBF for the mobile robot via an integrator backstepping
procedure. Our main results guarantee both asymptotic stability and safety for
the closed-loop system. Both the simulation and experimental results are
presented to illustrate the effectiveness and performance of our approach.

### 6. [Prescribed Performance Control of Deformable Object Manipulation in Spatial Latent Space](http://arxiv.org/pdf/2510.14234v1)

Authors: Ning Han, Gu Gong, Bin Zhang, Yuexuan Xu, Bohan Yang, Yunhui Liu, David Navarro-Alarcon

Manipulating three-dimensional (3D) deformable objects presents significant
challenges for robotic systems due to their infinite-dimensional state space
and complex deformable dynamics. This paper proposes a novel model-free
approach for shape control with constraints imposed on key points. Unlike
existing methods that rely on feature dimensionality reduction, the proposed
controller leverages the coordinates of key points as the feature vector, which
are extracted from the deformable object's point cloud using deep learning
methods. This approach not only reduces the dimensionality of the feature space
but also retains the spatial information of the object. By extracting key
points, the manipulation of deformable objects is simplified into a visual
servoing problem, where the shape dynamics are described using a deformation
Jacobian matrix. To enhance control accuracy, a prescribed performance control
method is developed by integrating barrier Lyapunov functions (BLF) to enforce
constraints on the key points. The stability of the closed-loop system is
rigorously analyzed and verified using the Lyapunov method. Experimental
results further demonstrate the effectiveness and robustness of the proposed
method.

### 7. [RoboANKLE: Design, Development, and Functional Evaluation of a Robotic Ankle with a Motorized Compliant Unit](http://arxiv.org/pdf/2510.14414v1)

Authors: Baris Baysal, Omid Arfaie, Ramazan Unal

This study presents a powered transtibial prosthesis with complete push-off
assistance, RoboANKLE. The design aims to fulfill specific requirements, such
as a sufficient range of motion (RoM) while providing the necessary torque for
achieving natural ankle motion in daily activities. Addressing the challenges
faced in designing active transtibial prostheses, such as maintaining energetic
autonomy and minimizing weight, is vital for the study. With this aim, we try
to imitate the human ankle by providing extensive push-off assistance to
achieve a natural-like torque profile. Thus, Energy Store and Extended Release
mechanism (ESER) is employed with a novel Extra Energy Storage (EES) mechanism.
Kinematic and kinetic analyses are carried out to determine the design
parameters and assess the design performance. Subsequently, a Computer-Aided
Design (CAD) model is built and used in comprehensive dynamic and structural
analyses. These analyses are used for the design performance evaluation and
determine the forces and torques applied to the prosthesis, which aids in
optimizing the design for minimal weight via structural analysis and topology
optimization. The design of the prototype is then finalized and manufactured
for experimental evaluation to validate the design and functionality. The
prototype is realized with a mass of 1.92 kg and dimensions of 261x107x420 mm.
The Functional evaluations of the RoboANKLE revealed that it is capable of
achieving the natural maximum dorsi-flexion angle with 95% accuracy. Also,
Thanks to the implemented mechanisms, the results show that RoboANKLE can
generate 57% higher than the required torque for natural walking. The result of
the power generation capacity of the RoboANKLE is 10% more than the natural
power during the gait cycle.

### 8. [Stability Criteria and Motor Performance in Delayed Haptic Dyadic Interactions Mediated by Robots](http://arxiv.org/pdf/2510.14511v1)

Authors: Mingtian Du, Suhas Raghavendra Kulkarni, Simone Kager, Domenico Campolo

This paper establishes analytical stability criteria for robot-mediated
human-human (dyadic) interaction systems, focusing on haptic communication
under network-induced time delays. Through frequency-domain analysis supported
by numerical simulations, we identify both delay-independent and
delay-dependent stability criteria. The delay-independent criterion guarantees
stability irrespective of the delay, whereas the delay-dependent criterion is
characterised by a maximum tolerable delay before instability occurs. The
criteria demonstrate dependence on controller and robot dynamic parameters,
where increasing stiffness reduces the maximum tolerable delay in a non-linear
manner, thereby heightening system vulnerability. The proposed criteria can be
generalised to a wide range of robot-mediated interactions and serve as design
guidelines for stable remote dyadic systems. Experiments with robots performing
human-like movements further illustrate the correlation between stability and
motor performance. The findings of this paper suggest the prerequisites for
effective delay-compensation strategies.

### 9. [A Deep State-Space Model Compression Method using Upper Bound on Output Error](http://arxiv.org/pdf/2510.14542v1)

Authors: Hiroki Sakamoto, Kazuhiro Sato

We study deep state-space models (Deep SSMs) that contain
linear-quadratic-output (LQO) systems as internal blocks and present a
compression method with a provable output error guarantee. We first derive an
upper bound on the output error between two Deep SSMs and show that the bound
can be expressed via the $h^2$-error norms between the layerwise LQO systems,
thereby providing a theoretical justification for existing model order
reduction (MOR)-based compression. Building on this bound, we formulate an
optimization problem in terms of the $h^2$-error norm and develop a
gradient-based MOR method. On the IMDb task from the Long Range Arena
benchmark, we demonstrate that our compression method achieves strong
performance. Moreover, unlike prior approaches, we reduce roughly 80% of
trainable parameters without retraining, with only a 4-5% performance drop.

### 10. [A Human-Vector Susceptible--Infected--Susceptible Model for Analyzing and Controlling the Spread of Vector-Borne Diseases](http://arxiv.org/pdf/2510.14787v1)

Authors: Lorenzo Zino, Alessandro Casu, Alessandro Rizzo

We propose an epidemic model for the spread of vector-borne diseases. The
model, which is built extending the classical susceptible-infected-susceptible
model, accounts for two populations -- humans and vectors -- and for
cross-contagion between the two species, whereby humans become infected upon
interaction with carrier vectors, and vectors become carriers after interaction
with infected humans. We formulate the model as a system of ordinary
differential equations and leverage monotone systems theory to rigorously
characterize the epidemic dynamics. Specifically, we characterize the global
asymptotic behavior of the disease, determining conditions for quick
eradication of the disease (i.e., for which all trajectories converge to a
disease-free equilibrium), or convergence to a (unique) endemic equilibrium.
Then, we incorporate two control actions: namely, vector control and incentives
to adopt protection measures. Using the derived mathematical tools, we assess
the impact of these two control actions and determine the optimal control
policy.

### Machine Learning (Statistics Category)

### 1. [A novel Information-Driven Strategy for Optimal Regression Assessment](http://arxiv.org/pdf/2510.14222v1)

Authors: Benjamín Castro, Camilo Ramírez, Sebastián Espinosa, Jorge F. Silva, Marcos E. Orchard, Heraldo Rozas

In Machine Learning (ML), a regression algorithm aims to minimize a loss
function based on data. An assessment method in this context seeks to quantify
the discrepancy between the optimal response for an input-output system and the
estimate produced by a learned predictive model (the student). Evaluating the
quality of a learned regressor remains challenging without access to the true
data-generating mechanism, as no data-driven assessment method can ensure the
achievability of global optimality. This work introduces the Information
Teacher, a novel data-driven framework for evaluating regression algorithms
with formal performance guarantees to assess global optimality. Our novel
approach builds on estimating the Shannon mutual information (MI) between the
input variables and the residuals and applies to a broad class of additive
noise models. Through numerical experiments, we confirm that the Information
Teacher is capable of detecting global optimality, which is aligned with the
condition of zero estimation error with respect to the -- inaccessible, in
practice -- true model, working as a surrogate measure of the ground truth
assessment loss and offering a principled alternative to conventional empirical
performance metrics.

### 2. [Nonparametric Data Attribution for Diffusion Models](http://arxiv.org/pdf/2510.14269v1)

Authors: Yutian Zhao, Chao Du, Xiaosen Zheng, Tianyu Pang, Min Lin

Data attribution for generative models seeks to quantify the influence of
individual training examples on model outputs. Existing methods for diffusion
models typically require access to model gradients or retraining, limiting
their applicability in proprietary or large-scale settings. We propose a
nonparametric attribution method that operates entirely on data, measuring
influence via patch-level similarity between generated and training images. Our
approach is grounded in the analytical form of the optimal score function and
naturally extends to multiscale representations, while remaining
computationally efficient through convolution-based acceleration. In addition
to producing spatially interpretable attributions, our framework uncovers
patterns that reflect intrinsic relationships between training data and
outputs, independent of any specific model. Experiments demonstrate that our
method achieves strong attribution performance, closely matching gradient-based
approaches and substantially outperforming existing nonparametric baselines.
Code is available at https://github.com/sail-sg/NDA.

### 3. [Active Measuring in Reinforcement Learning With Delayed Negative Effects](http://arxiv.org/pdf/2510.14315v1)

Authors: Daiqi Gao, Ziping Xu, Aseel Rawashdeh, Predrag Klasnja, Susan A. Murphy

Measuring states in reinforcement learning (RL) can be costly in real-world
settings and may negatively influence future outcomes. We introduce the
Actively Observable Markov Decision Process (AOMDP), where an agent not only
selects control actions but also decides whether to measure the latent state.
The measurement action reveals the true latent state but may have a negative
delayed effect on the environment. We show that this reduced uncertainty may
provably improve sample efficiency and increase the value of the optimal policy
despite these costs. We formulate an AOMDP as a periodic partially observable
MDP and propose an online RL algorithm based on belief states. To approximate
the belief states, we further propose a sequential Monte Carlo method to
jointly approximate the posterior of unknown static environment parameters and
unobserved latent states. We evaluate the proposed algorithm in a digital
health application, where the agent decides when to deliver digital
interventions and when to assess users' health status through surveys.

### 4. [Personalized federated learning, Row-wise fusion regularization, Multivariate modeling, Sparse estimation](http://arxiv.org/pdf/2510.14413v1)

Authors: Runlin Zhou, Letian Li, Zemin Zheng

We study personalized federated learning for multivariate responses where
client models are heterogeneous yet share variable-level structure. Existing
entry-wise penalties ignore cross-response dependence, while matrix-wise fusion
over-couples clients. We propose a Sparse Row-wise Fusion (SROF) regularizer
that clusters row vectors across clients and induces within-row sparsity, and
we develop RowFed, a communication-efficient federated algorithm that embeds
SROF into a linearized ADMM framework with privacy-preserving partial
participation. Theoretically, we establish an oracle property for
SROF-achieving correct variable-level group recovery with asymptotic
normality-and prove convergence of RowFed to a stationary solution. Under
random client participation, the iterate gap contracts at a rate that improves
with participation probability. Empirically, simulations in heterogeneous
regimes show that RowFed consistently lowers estimation and prediction error
and strengthens variable-level cluster recovery over NonFed, FedAvg, and a
personalized matrix-fusion baseline. A real-data study further corroborates
these gains while preserving interpretability. Together, our results position
row-wise fusion as an effective and transparent paradigm for large-scale
personalized federated multivariate learning, bridging the gap between
entry-wise and matrix-wise formulations.

### 5. [Interaction Concordance Index: Performance Evaluation for Interaction Prediction Methods](http://arxiv.org/pdf/2510.14419v1)

Authors: Tapio Pahikkala, Riikka Numminen, Parisa Movahedi, Napsu Karmitsa, Antti Airola

Consider two sets of entities and their members' mutual affinity values, say
drug-target affinities (DTA). Drugs and targets are said to interact in their
effects on DTAs if drug's effect on it depends on the target. Presence of
interaction implies that assigning a drug to a target and another drug to
another target does not provide the same aggregate DTA as the reversed
assignment would provide. Accordingly, correctly capturing interactions enables
better decision-making, for example, in allocation of limited numbers of drug
doses to their best matching targets. Learning to predict DTAs is popularly
done from either solely from known DTAs or together with side information on
the entities, such as chemical structures of drugs and targets. In this paper,
we introduce interaction directions' prediction performance estimator we call
interaction concordance index (IC-index), for both fixed predictors and machine
learning algorithms aimed for inferring them. IC-index complements the
popularly used DTA prediction performance estimators by evaluating the ratio of
correctly predicted directions of interaction effects in data. First, we show
the invariance of IC-index on predictors unable to capture interactions.
Secondly, we show that learning algorithm's permutation equivariance regarding
drug and target identities implies its inability to capture interactions when
either drug, target or both are unseen during training. In practical
applications, this equivariance is remedied via incorporation of appropriate
side information on drugs and targets. We make a comprehensive empirical
evaluation over several biomedical interaction data sets with various
state-of-the-art machine learning algorithms. The experiments demonstrate how
different types of affinity strength prediction methods perform in terms of
IC-index complementing existing prediction performance estimators.

### 6. [Parameter Identification for Partial Differential Equation with Jump Discontinuities in Coefficients by Markov Switching Model and Physics-Informed Machine Learning](http://arxiv.org/pdf/2510.14656v1)

Authors: Zhikun Zhang, Guanyu Pan, Xiangjun Wang, Yong Xu, Guangtao Zhang

Inverse problems involving partial differential equations (PDEs) with
discontinuous coefficients are fundamental challenges in modeling complex
spatiotemporal systems with heterogeneous structures and uncertain dynamics.
Traditional numerical and machine learning approaches often face limitations in
addressing these problems due to high dimensionality, inherent nonlinearity,
and discontinuous parameter spaces. In this work, we propose a novel
computational framework that synergistically integrates physics-informed deep
learning with Bayesian inference for accurate parameter identification in PDEs
with jump discontinuities in coefficients. The core innovation of our framework
lies in a dual-network architecture employing a gradient-adaptive weighting
strategy: a main network approximates PDE solutions while a sub network samples
its coefficients. To effectively identify mixture structures in parameter
spaces, we employ Markovian dynamics methods to capture hidden state
transitions of complex spatiotemporal systems. The framework has applications
in reconstruction of solutions and identification of parameter-varying regions.
Comprehensive numerical experiments on various PDEs with jump-varying
coefficients demonstrate the framework's exceptional adaptability, accuracy,
and robustness compared to existing methods. This study provides a
generalizable computational approach of parameter identification for PDEs with
discontinuous parameter structures, particularly in non-stationary or
heterogeneous systems.

### 7. [Fast and Scalable Score-Based Kernel Calibration Tests](http://arxiv.org/pdf/2510.14711v1)

Authors: Pierre Glaser, David Widmann, Fredrik Lindsten, Arthur Gretton

We introduce the Kernel Calibration Conditional Stein Discrepancy test (KCCSD
test), a non-parametric, kernel-based test for assessing the calibration of
probabilistic models with well-defined scores. In contrast to previous methods,
our test avoids the need for possibly expensive expectation approximations
while providing control over its type-I error. We achieve these improvements by
using a new family of kernels for score-based probabilities that can be
estimated without probability density samples, and by using a conditional
goodness-of-fit criterion for the KCCSD test's U-statistic. We demonstrate the
properties of our test on various synthetic settings.

### 8. [Causal Discovery for Linear DAGs with Dependent Latent Variables via Higher-order Cumulants](http://arxiv.org/pdf/2510.14780v1)

Authors: Ming Cai, Penggang Gao, Hisayuki Hara

This paper addresses the problem of estimating causal directed acyclic graphs
in linear non-Gaussian acyclic models with latent confounders (LvLiNGAM).
Existing methods assume mutually independent latent confounders or cannot
properly handle models with causal relationships among observed variables.
  We propose a novel algorithm that identifies causal DAGs in LvLiNGAM,
allowing causal structures among latent variables, among observed variables,
and between the two. The proposed method leverages higher-order cumulants of
observed data to identify the causal structure. Extensive simulations and
experiments with real-world data demonstrate the validity and practical utility
of the proposed algorithm.

### 9. [A Geometric Approach to Optimal Experimental Design](http://arxiv.org/pdf/2510.14848v1)

Authors: Gavin Kerrigan, Christian A. Naesseth, Tom Rainforth

We introduce a novel geometric framework for optimal experimental design
(OED). Traditional OED approaches, such as those based on mutual information,
rely explicitly on probability densities, leading to restrictive invariance
properties. To address these limitations, we propose the mutual transport
dependence (MTD), a measure of statistical dependence grounded in optimal
transport theory which provides a geometric objective for optimizing designs.
Unlike conventional approaches, the MTD can be tailored to specific downstream
estimation problems by choosing appropriate geometries on the underlying
spaces. We demonstrate that our framework produces high-quality designs while
offering a flexible alternative to standard information-theoretic techniques.

### 10. [EM Approaches to Nonparametric Estimation for Mixture of Linear Regressions](http://arxiv.org/pdf/2510.14890v1)

Authors: Andrew Welbaum, Wanli Qiao

In a mixture of linear regression model, the regression coefficients are
treated as random vectors that may follow either a continuous or discrete
distribution. We propose two Expectation-Maximization (EM) algorithms to
estimate this prior distribution. The first algorithm solves a kernelized
version of the nonparametric maximum likelihood estimation (NPMLE). This method
not only recovers continuous prior distributions but also accurately estimates
the number of clusters when the prior is discrete. The second algorithm,
designed to approximate the NPMLE, targets prior distributions with a density.
It also performs well for discrete priors when combined with a post-processing
step. We study the convergence properties of both algorithms and demonstrate
their effectiveness through simulations and applications to real datasets.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-10-17 PST.

### 1. [A roadmap toward closed-loop autonomous experimentation for engineered nanomaterials](https://www.nature.com/articles/s44286-025-00291-x)

Authors: Nicholas A. Jose et al.

### 2. [Optical and SAR image registration based on feature constrained algorithm](https://www.nature.com/articles/s41598-025-20332-y)

Authors: Wannan Zhang

### 3. [A comparison of the performance of Chinese large language models and ChatGPT throughout the entire clinical workflow](https://www.nature.com/articles/s41598-025-20210-7)

Authors: Yang He et al.

### 4. [Enhancing carbon emission reduction strategies using OCO and ICOS data](https://www.nature.com/articles/s41598-025-22022-1)

Authors: Oskar Åström et al.

### 5. [Prediction of surface drifter trajectories in the South China sea using deep learning](https://www.nature.com/articles/s41598-025-20143-1)

Authors: Chuan Tian et al.

### 6. [Hierarchical sparse Bayesian learning with adaptive Laplacian prior for single image super-resolution](https://www.nature.com/articles/s41598-025-20115-5)

Authors: Mingming Qi et al.

### 7. [Accelerating the tuning process for optimizing DNN operators by ROFT model](https://www.nature.com/articles/s41598-025-20139-x)

Authors: ZiChuan He et al.

### 8. [Research on conveyor belt damage detection method based on FDEP−YOLOv8](https://www.nature.com/articles/s41598-025-20391-1)

Authors: Yuan Yuan et al.

### 9. [A ballistocardiogram dataset with reference ECG signals for bed-based heart rhythm assessment](https://www.nature.com/articles/s41597-025-05936-3)

Authors: Jiafeng Qiu et al.

### 10. [Generating detectors from anomaly samples via negative selection for network intrusion detection](https://www.nature.com/articles/s41598-025-20516-6)

Authors: Zhiyong Li et al.

### 11. [Deep learning for text summarization using NLP for automated news digest](https://www.nature.com/articles/s41598-025-20224-1)

Authors: K. M. Rani Krishna et al.

### 12. [A dynamic lightweight blockchain sharding protocol for autonomous collaborative combat of UAV swarms in denied environments](https://www.nature.com/articles/s41598-025-20359-1)

Authors: Yifan Xu et al.

### 13. [Small defect detection in printed circuit boards based on the multiscale edge strengthening and an improved YOLOv10](https://www.nature.com/articles/s41598-025-20387-x)

Authors: Weixun Chen et al.

### 14. [Enhancing virtual physically unclonable function security through neuron-criticality analysis and lightweight encryption](https://www.nature.com/articles/s41598-025-20305-1)

Authors: Raviha Khan et al.

