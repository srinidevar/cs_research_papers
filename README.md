# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-08-19 17:00:27.073918 PST.

### Artificial Intelligence

### 1. [Help or Hurdle? Rethinking Model Context Protocol-Augmented Large Language Models](http://arxiv.org/pdf/2508.12566v1)

Authors: Wei Song, Haonan Zhong, Ziqi Ding, Jingling Xue, Yuekang Li

The Model Context Protocol (MCP) enables large language models (LLMs) to
access external resources on demand. While commonly assumed to enhance
performance, how LLMs actually leverage this capability remains poorly
understood. We introduce MCPGAUGE, the first comprehensive evaluation framework
for probing LLM-MCP interactions along four key dimensions: proactivity
(self-initiated tool use), compliance (adherence to tool-use instructions),
effectiveness (task performance post-integration), and overhead (computational
cost incurred). MCPGAUGE comprises a 160-prompt suite and 25 datasets spanning
knowledge comprehension, general reasoning, and code generation. Our
large-scale evaluation, spanning six commercial LLMs, 30 MCP tool suites, and
both one- and two-turn interaction settings, comprises around 20,000 API calls
and over USD 6,000 in computational cost. This comprehensive study reveals four
key findings that challenge prevailing assumptions about the effectiveness of
MCP integration. These insights highlight critical limitations in current
AI-tool integration and position MCPGAUGE as a principled benchmark for
advancing controllable, tool-augmented LLMs.

### 2. [GridCodex: A RAG-Driven AI Framework for Power Grid Code Reasoning and Compliance](http://arxiv.org/pdf/2508.12682v1)

Authors: Jinquan Shi, Yingying Cheng, Fan Zhang, Miao Jiang, Jun Lin, Yanbai Shen

The global shift towards renewable energy presents unprecedented challenges
for the electricity industry, making regulatory reasoning and compliance
increasingly vital. Grid codes, the regulations governing grid operations, are
complex and often lack automated interpretation solutions, which hinders
industry expansion and undermines profitability for electricity companies. We
introduce GridCodex, an end to end framework for grid code reasoning and
compliance that leverages large language models and retrieval-augmented
generation (RAG). Our framework advances conventional RAG workflows through
multi stage query refinement and enhanced retrieval with RAPTOR. We validate
the effectiveness of GridCodex with comprehensive benchmarks, including
automated answer assessment across multiple dimensions and regulatory agencies.
Experimental results showcase a 26.4% improvement in answer quality and more
than a 10 fold increase in recall rate. An ablation study further examines the
impact of base model selection.

### 3. [GTool: Graph Enhanced Tool Planning with Large Language Model](http://arxiv.org/pdf/2508.12725v1)

Authors: Wenjie Chen, Wenbin Li, Di Yao, Xuying Meng, Chang Gong, Jingping Bi

Tool planning with large language models (LLMs), referring to selecting,
organizing, and preparing the tools necessary to complete a user request,
bridges the gap between natural language understanding and task execution.
However, current works treat different tools as isolated components and fail to
leverage the inherent dependencies of tools, leading to invalid planning
results. Since tool dependencies are often incomplete, it becomes challenging
for LLMs to accurately identify the appropriate tools required by a user
request, especially when confronted with a large toolset. To solve this
challenge, we propose \texttt{GTool}, which is the first work aiming to enhance
the tool planning ability of LLMs under incomplete dependencies. \texttt{GTool}
constructs a request-specific tool graph to select tools efficiently and
generate the \texttt{<graph token>} which provides sufficient dependency
information understandable by LLMs. Moreover, a missing dependency prediction
task is designed to improve the reliability of \texttt{GTool} with incomplete
dependencies. Without trimming LLMs, \texttt{GTool} can be seamlessly
integrated with various LLM backbones without extensive retraining. Extensive
experiments show that \texttt{GTool} achieves more than 29.6\% performance
improvements compared with the state-of-the-art (SOTA) baselines with a
light-weight (7B) LLM backbone.

### 4. [Beyond Ethical Alignment: Evaluating LLMs as Artificial Moral Assistants](http://arxiv.org/pdf/2508.12754v1)

Authors: Alessio Galatolo, Luca Alberto Rappuoli, Katie Winkle, Meriem Beloucif

The recent rise in popularity of large language models (LLMs) has prompted
considerable concerns about their moral capabilities. Although considerable
effort has been dedicated to aligning LLMs with human moral values, existing
benchmarks and evaluations remain largely superficial, typically measuring
alignment based on final ethical verdicts rather than explicit moral reasoning.
In response, this paper aims to advance the investigation of LLMs' moral
capabilities by examining their capacity to function as Artificial Moral
Assistants (AMAs), systems envisioned in the philosophical literature to
support human moral deliberation. We assert that qualifying as an AMA requires
more than what state-of-the-art alignment techniques aim to achieve: not only
must AMAs be able to discern ethically problematic situations, they should also
be able to actively reason about them, navigating between conflicting values
outside of those embedded in the alignment phase. Building on existing
philosophical literature, we begin by designing a new formal framework of the
specific kind of behaviour an AMA should exhibit, individuating key qualities
such as deductive and abductive moral reasoning. Drawing on this theoretical
framework, we develop a benchmark to test these qualities and evaluate popular
open LLMs against it. Our results reveal considerable variability across models
and highlight persistent shortcomings, particularly regarding abductive moral
reasoning. Our work connects theoretical philosophy with practical AI
evaluation while also emphasising the need for dedicated strategies to
explicitly enhance moral reasoning capabilities in LLMs. Code available at
https://github.com/alessioGalatolo/AMAeval

### 5. [HeroBench: A Benchmark for Long-Horizon Planning and Structured Reasoning in Virtual Worlds](http://arxiv.org/pdf/2508.12782v1)

Authors: Petr Anokhin, Roman Khalikov, Stefan Rebrikov, Viktor Volkov, Artyom Sorokin, Vincent Bissonnette

Large language models (LLMs) have shown remarkable capabilities in isolated
step-by-step reasoning tasks such as mathematics and programming, but their
proficiency in long-horizon planning, where solutions require extended,
structured sequences of interdependent actions, remains underexplored. Existing
benchmarks typically assess LLMs through abstract or low-dimensional
algorithmic tasks, failing to capture the complexity of realistic planning
environments. We introduce HeroBench, a novel benchmark designed specifically
to evaluate long-horizon planning and structured reasoning within complex
RPG-inspired virtual worlds. HeroBench provides a rigorously constructed
dataset of tasks covering a wide range of difficulties, a simulated environment
to execute and validate agent plans, and detailed analytical tools for
evaluating model performance. Tasks challenge models to formulate strategic
plans, efficiently gather resources, master necessary skills, craft equipment,
and defeat adversaries, reflecting practical scenarios' layered dependencies
and constraints. Our extensive evaluation of 25 state-of-the-art LLMs, spanning
both open-source and proprietary models, including the GPT-5 family, reveals
substantial performance disparities rarely observed in conventional reasoning
benchmarks. Detailed error analysis further uncovers specific weaknesses in
current models' abilities to generate robust high-level plans and reliably
execute structured actions. HeroBench thus not only significantly advances the
evaluation of LLM reasoning but also provides a flexible, scalable foundation
for future research into advanced, autonomous planning in virtual environments.

### 6. [Towards Open-Ended Emotional Support Conversations in LLMs via Reinforcement Learning with Future-Oriented Rewards](http://arxiv.org/pdf/2508.12935v1)

Authors: Ting Yang, Li Chen, Huimin Wang

Emotional Support Conversation (ESC) systems aim to alleviate users'
emotional difficulties and provide long-term, systematic support for emotional
well-being. However, most large language model (LLM)-based ESC systems rely on
predefined strategies, which limits their effectiveness in complex, real-life
scenarios. To enable flexible responses to diverse emotional problem scenarios,
this paper introduces a novel end-to-end framework (RLFF-ESC) that directly
learns enduring emotionally supportive response skills using reinforcement
learning. For sustained emotional support, we first employ an LLM-based
multi-agent mechanism to simulate future dialogue trajectories and collect
future-oriented rewards. We then train a future-oriented reward model, which is
subsequently used to train the emotional support policy model. Additionally, we
incorporate an explicit reasoning process during response generation to further
enhance the quality, relevance, and contextual appropriateness of the system's
responses. We evaluate the backbone policy model on Qwen2.5-7B-Instruct-1M and
LLaMA3.1-8B-Instruct models, testing the proposed RLFF-ESC framework across two
public ESC datasets. Experimental results demonstrate that RLFF-ESC
consistently outperforms existing baselines in terms of goal completion and
response quality.

### 7. [EvolMathEval: Towards Evolvable Benchmarks for Mathematical Reasoning via Evolutionary Testing](http://arxiv.org/pdf/2508.13003v1)

Authors: Shengbo Wang, Mingwei Liu, Zike Li, Anji Li, Yanlin Wang, Xin Peng, Zibin Zheng

The rapid advancement of LLMs poses a significant challenge to existing
mathematical reasoning benchmarks. These benchmarks commonly suffer from issues
such as score saturation, temporal decay, and data contamination. To address
this challenge, this paper introduces EvolMathEval, an automated mathematical
benchmark generation and evolution framework based on evolutionary testing. By
dynamically generating unique evaluation instances ab initio, the framework
fundamentally eliminates the risk of data contamination, and ensuring the
benchmark remains perpetually challenging for future models.The core mechanisms
of EvolMathEval include: seed problem generation based on reverse engineering
with algebraic guarantees; multi-dimensional genetic operators designed to
inject diverse cognitive challenges; and a composite fitness function that can
rapidly and accurately assess problem difficulty. Experimental results
demonstrate that the proposed composite fitness function can efficiently and
precisely quantify the difficulty of mathematical problems. Furthermore,
EvolMathEval can not only generate a large volume of high-difficulty problems
through continuous self-iteration, but it can also significantly enhance the
complexity of public datasets like GSM8K through evolution, reducing model
accuracy by an average of 48%. Deeper investigation reveals that when solving
these evolved, complex problems, LLMs tend to employ non-rigorous heuristics to
bypass complex multi-step logical reasoning, consequently leading to incorrect
solutions. We define this phenomenon as "Pseudo Aha Moment". This finding
uncovers a cognitive shortcut-taking behavior in the deep reasoning processes
of current LLMs, which we find accounts for 77% to 100% of errors on targeted
problems. Code and resources are available
at:https://github.com/SYSUSELab/EvolMathEval.

### 8. [G$^2$RPO-A: Guided Group Relative Policy Optimization with Adaptive Guidance](http://arxiv.org/pdf/2508.13023v1)

Authors: Yongxin Guo, Wenbo Deng, Zhenglin Cheng, Xiaoying Tang

Reinforcement Learning with Verifiable Rewards (RLVR) has markedly enhanced
the reasoning abilities of large language models (LLMs). Its success, however,
largely depends on strong base models with rich world knowledge, yielding only
modest improvements for small-size language models (SLMs). To address this
limitation, we investigate Guided GRPO, which injects ground-truth reasoning
steps into roll-out trajectories to compensate for SLMs' inherent weaknesses.
Through a comprehensive study of various guidance configurations, we find that
naively adding guidance delivers limited gains. These insights motivate
G$^2$RPO-A, an adaptive algorithm that automatically adjusts guidance strength
in response to the model's evolving training dynamics. Experiments on
mathematical reasoning and code-generation benchmarks confirm that G$^2$RPO-A
substantially outperforms vanilla GRPO. Our code and models are available at
https://github.com/T-Lab-CUHKSZ/G2RPO-A.

### 9. [A Language-Signal-Vision Multimodal Framework for Multitask Cardiac Analysis](http://arxiv.org/pdf/2508.13072v1)

Authors: Yuting Zhang, Tiantian Geng, Luoying Hao, Xinxing Cheng, Alexander Thorley, Xiaoxia Wang, Wenqi Lu, Sandeep S Hothi, Lei Wei, Zhaowen Qiu, Dipak Kotecha, Jinming Duan

Contemporary cardiovascular management involves complex consideration and
integration of multimodal cardiac datasets, where each modality provides
distinct but complementary physiological characteristics. While the effective
integration of multiple modalities could yield a holistic clinical profile that
accurately models the true clinical situation with respect to data modalities
and their relatives weightings, current methodologies remain limited by: 1) the
scarcity of patient- and time-aligned multimodal data; 2) reliance on isolated
single-modality or rigid multimodal input combinations; 3) alignment strategies
that prioritize cross-modal similarity over complementarity; and 4) a narrow
single-task focus. In response to these limitations, a comprehensive multimodal
dataset was curated for immediate application, integrating laboratory test
results, electrocardiograms, and echocardiograms with clinical outcomes.
Subsequently, a unified framework, Textual Guidance Multimodal fusion for
Multiple cardiac tasks (TGMM), was proposed. TGMM incorporated three key
components: 1) a MedFlexFusion module designed to capture the unique and
complementary characteristics of medical modalities and dynamically integrate
data from diverse cardiac sources and their combinations; 2) a textual guidance
module to derive task-relevant representations tailored to diverse clinical
objectives, including heart disease diagnosis, risk stratification and
information retrieval; and 3) a response module to produce final decisions for
all these tasks. Furthermore, this study systematically explored key features
across multiple modalities and elucidated their synergistic contributions in
clinical decision-making. Extensive experiments showed that TGMM outperformed
state-of-the-art methods across multiple clinical tasks, with additional
validation confirming its robustness on another public dataset.

### 10. [Bayesian Optimization-based Search for Agent Control in Automated Game Testing](http://arxiv.org/pdf/2508.13121v1)

Authors: Carlos Celemin

This work introduces an automated testing approach that employs agents
controlling game characters to detect potential bugs within a game level.
Harnessing the power of Bayesian Optimization (BO) to execute sample-efficient
search, the method determines the next sampling point by analyzing the data
collected so far and calculates the data point that will maximize information
acquisition. To support the BO process, we introduce a game testing-specific
model built on top of a grid map, that features the smoothness and uncertainty
estimation required by BO, however and most importantly, it does not suffer the
scalability issues that traditional models carry. The experiments demonstrate
that the approach significantly improves map coverage capabilities in both time
efficiency and exploration distribution.

### Hardware Architecture

### 1. [MemorySim: An RTL-level, timing accurate simulator model for the Chisel ecosystem](http://arxiv.org/pdf/2508.12636v1)

Authors: Ansh Chaurasia

The rapid growth of AI applications has driven increased demand for
specialized AI hardware, highlighting critical opportunities within the memory
subsystem, which often serves as a performance bottleneck in high-demand
workloads such as large language models (LLMs). Existing high-level memory
simulators, such as DRAMSim2 and DRAMSim3, offer timing simulations but
frequently compromise on correctness or integration at the register-transfer
level (RTL). We present MemorySim, an RTL-level memory simulator designed to
deliver both accurate timing and functional correctness. MemorySim integrates
seamlessly with existing Chisel and Verilog simulations and is fully compatible
with the Chisel/Chipyard ecosystem. This enables users to obtain precise
performance and power estimates, supporting downstream evaluation through
simulation platforms such as FireSim.

### 2. [IzhiRISC-V -- a RISC-V-based Processor with Custom ISA Extension for Spiking Neuron Networks Processing with Izhikevich Neurons](http://arxiv.org/pdf/2508.12846v1)

Authors: Wiktor J. Szczerek, Artur Podobas

Spiking Neural Network processing promises to provide high energy efficiency
due to the sparsity of the spiking events. However, when realized on
general-purpose hardware -- such as a RISC-V processor -- this promise can be
undermined and overshadowed by the inefficient code, stemming from repeated
usage of basic instructions for updating all the neurons in the network. One of
the possible solutions to this issue is the introduction of a custom ISA
extension with neuromorphic instructions for spiking neuron updating, and
realizing those instructions in bespoke hardware expansion to the existing ALU.
In this paper, we present the first step towards realizing a large-scale system
based on the RISC-V-compliant processor called IzhiRISC-V, supporting the
custom neuromorphic ISA extension.

### 3. [e-boost: Boosted E-Graph Extraction with Adaptive Heuristics and Exact Solving](http://arxiv.org/pdf/2508.13020v1)

Authors: Jiaqi Yin, Zhan Song, Chen Chen, Yaohui Cai, Zhiru Zhang, Cunxi Yu

E-graphs have attracted growing interest in many fields, particularly in
logic synthesis and formal verification. E-graph extraction is a challenging
NP-hard combinatorial optimization problem. It requires identifying optimal
terms from exponentially many equivalent expressions, serving as the primary
performance bottleneck in e-graph based optimization tasks. However,
traditional extraction methods face a critical trade-off: heuristic approaches
offer speed but sacrifice optimality, while exact methods provide optimal
solutions but face prohibitive computational costs on practical problems. We
present e-boost, a novel framework that bridges this gap through three key
innovations: (1) parallelized heuristic extraction that leverages weak data
dependence to compute DAG costs concurrently, enabling efficient multi-threaded
performance without sacrificing extraction quality; (2) adaptive search space
pruning that employs a parameterized threshold mechanism to retain only
promising candidates, dramatically reducing the solution space while preserving
near-optimal solutions; and (3) initialized exact solving that formulates the
reduced problem as an Integer Linear Program with warm-start capabilities,
guiding solvers toward high-quality solutions faster.
  Across the diverse benchmarks in formal verification and logic synthesis
fields, e-boost demonstrates 558x runtime speedup over traditional exact
approaches (ILP) and 19.04% performance improvement over the state-of-the-art
extraction framework (SmoothE). In realistic logic synthesis tasks, e-boost
produces 7.6% and 8.1% area improvements compared to conventional synthesis
tools with two different technology mapping libraries. e-boost is available at
https://github.com/Yu-Maryland/e-boost.

### 4. [HOMI: Ultra-Fast EdgeAI platform for Event Cameras](http://arxiv.org/pdf/2508.12637v1)

Authors: Shankaranarayanan H, Satyapreet Singh Yadav, Adithya Krishna, Ajay Vikram P, Mahesh Mehendale, Chetan Singh Thakur

Event cameras offer significant advantages for edge robotics applications due
to their asynchronous operation and sparse, event-driven output, making them
well-suited for tasks requiring fast and efficient closed-loop control, such as
gesture-based human-robot interaction. Despite this potential, existing event
processing solutions remain limited, often lacking complete end-to-end
implementations, exhibiting high latency, and insufficiently exploiting event
data sparsity. In this paper, we present HOMI, an ultra-low latency, end-to-end
edge AI platform comprising a Prophesee IMX636 event sensor chip with an Xilinx
Zynq UltraScale+MPSoC FPGA chip, deploying an in-house developed AI
accelerator. We have developed hardware-optimized pre-processing pipelines
supporting both constant-time and constant-event modes for histogram
accumulation, linear and exponential time surfaces. Our general-purpose
implementation caters to both accuracy-driven and low-latency applications.
HOMI achieves 94% accuracy on the DVS Gesture dataset as a use case when
configured for high accuracy operation and provides a throughput of 1000 fps
for low-latency configuration. The hardware-optimised pipeline maintains a
compact memory footprint and utilises only 33% of the available LUT resources
on the FPGA, leaving ample headroom for further latency reduction, model
parallelisation, multi-task deployments, or integration of more complex
architectures.

### 5. [SecFSM: Knowledge Graph-Guided Verilog Code Generation for Secure Finite State Machines in Systems-on-Chip](http://arxiv.org/pdf/2508.12910v1)

Authors: Ziteng Hu, Yingjie Xia, Xiyuan Chen, Li Kuang

Finite State Machines (FSMs) play a critical role in implementing control
logic for Systems-on-Chip (SoC). Traditionally, FSMs are implemented by
hardware engineers through Verilog coding, which is often tedious and
time-consuming. Recently, with the remarkable progress of Large Language Models
(LLMs) in code generation, LLMs have been increasingly explored for automating
Verilog code generation. However, LLM-generated Verilog code often suffers from
security vulnerabilities, which is particularly concerning for
security-sensitive FSM implementations. To address this issue, we propose
SecFSM, a novel method that leverages a security-oriented knowledge graph to
guide LLMs in generating more secure Verilog code. Specifically, we first
construct a FSM Security Knowledge Graph (FSKG) as an external aid to LLMs.
Subsequently, we analyze users' requirements to identify vulnerabilities and
get a list of vulnerabilities in the requirements. Then, we retrieve knowledge
from FSKG based on the vulnerabilities list. Finally, we construct security
prompts based on the security knowledge for Verilog code generation. To
evaluate SecFSM, we build a dedicated dataset collected from academic datasets,
artificial datasets, papers, and industrial cases. Extensive experiments
demonstrate that SecFSM outperforms state-of-the-art baselines. In particular,
on a benchmark of 25 security test cases evaluated by DeepSeek-R1, SecFSM
achieves an outstanding pass rate of 21/25.

### 6. [XR-NPE: High-Throughput Mixed-precision SIMD Neural Processing Engine for Extended Reality Perception Workloads](http://arxiv.org/pdf/2508.13049v1)

Authors: Tejas Chaudhari, Akarsh J., Tanushree Dewangan, Mukul Lokhande, Santosh Kumar Vishvakarma

This work proposes XR-NPE, a high-throughput Mixed-precision SIMD Neural
Processing Engine, designed for extended reality (XR) perception workloads like
visual inertial odometry (VIO), object classification, and eye gaze extraction.
XR-NPE is first to support FP4, Posit (4,1), Posit (8,0), and Posit (16,1)
formats, with layer adaptive hybrid-algorithmic implementation supporting
ultra-low bit precision to significantly reduce memory bandwidth requirements,
and accompanied by quantization-aware training for minimal accuracy loss. The
proposed Reconfigurable Mantissa Multiplication and Exponent processing
Circuitry (RMMEC) reduces dark silicon in the SIMD MAC compute engine, assisted
by selective power gating to reduce energy consumption, providing 2.85x
improved arithmetic intensity. XR-NPE achieves a maximum operating frequency of
1.72 GHz, area 0.016 mm2 , and arithmetic intensity 14 pJ at CMOS 28nm,
reducing 42% area, 38% power compared to the best of state-of-the-art MAC
approaches. The proposed XR-NPE based AXI-enabled Matrix-multiplication
co-processor consumes 1.4x fewer LUTs, 1.77x fewer FFs, and provides 1.2x
better energy efficiency compared to SoTA accelerators on VCU129. The proposed
co-processor provides 23% better energy efficiency and 4% better compute
density for VIO workloads. XR-NPE establishes itself as a scalable,
precision-adaptive compute engine for future resource-constrained XR devices.
The complete set for codes for results reproducibility are released publicly,
enabling designers and researchers to readily adopt and build upon them.
https://github.com/mukullokhande99/XR-NPE.

### Computational Complexity

### 1. [Geometry Matters in Planar Storyplans](http://arxiv.org/pdf/2508.12747v1)

Authors: Alexander Dobler, Maximilian Holzmüller, Martin Nöllenburg

A storyplan visualizes a graph $G=(V,E)$ as a sequence of $\ell$ frames
$\Gamma_1, \dots, \Gamma_\ell$, each of which is a drawing of the induced
subgraph $G[V_i]$ of a vertex subset $V_i \subseteq V$. Moreover, each vertex
$v \in V$ is contained in a single consecutive sequence of frames $\Gamma_i,
\dots, \Gamma_j$, all vertices and edges contained in consecutive frames are
drawn identically, and the union of all frames is a drawing of $G$. In GD 2022,
the concept of planar storyplans was introduced, in which each frame must be a
planar (topological) drawing. Several (parameterized) complexity results for
recognizing graphs that admit a planar storyplan were provided, including
NP-hardness. In this paper, we investigate an open question posed in the GD
paper and show that the geometric and topological settings of the planar
storyplan problem differ: We provide an instance of a graph that admits a
planar storyplan, but no planar geometric storyplan, in which each frame is a
planar straight-line drawing. Still, by adapting the reduction proof from the
topological to the geometric setting, we show that recognizing the graphs that
admit planar geometric storyplans remains NP-hard.

### 2. [On the complexity of constrained reconfiguration and motion planning](http://arxiv.org/pdf/2508.13032v1)

Authors: Nicolas Bousquet, Remy El Sabeh, Amer E. Mouawad, Naomi Nishimura

Coordinating the motion of multiple agents in constrained environments is a
fundamental challenge in robotics, motion planning, and scheduling. A
motivating example involves $n$ robotic arms, each represented as a line
segment. The objective is to rotate each arm to its vertical orientation, one
at a time (clockwise or counterclockwise), without collisions nor rotating any
arm more than once. This scenario is an example of the more general
$k$-Compatible Ordering problem, where $n$ agents, each capable of $k$
state-changing actions, must transition to specific target states under
constraints encoded as a set $\mathcal{G}$ of $k$ pairs of directed graphs.
  We show that $k$-Compatible Ordering is $\mathsf{NP}$-complete, even when
$\mathcal{G}$ is planar, degenerate, or acyclic. On the positive side, we
provide polynomial-time algorithms for cases such as when $k = 1$ or
$\mathcal{G}$ has bounded treewidth. We also introduce generalized variants
supporting multiple state-changing actions per agent, broadening the
applicability of our framework. These results extend to a wide range of
scheduling, reconfiguration, and motion planning applications in constrained
environments.

### Computational Engineering

### 1. [Ensured Energy: A simulation game to elicit preferences around Swiss energy transition pathways](http://arxiv.org/pdf/2508.12799v1)

Authors: Toby Simpson, Saara Jones, Gracia Brückmann, Walid El-Ajou, Erwan Moreira, Borja Martinez Oltra, Rolf Krause, Michael Multerer, Isabelle Stadelmann

The 2015 Paris Agreement on global warming specifies national objectives for
the reduction of greenhouse gas emissions. In support of Switzerland's energy
and climate strategy for 2050, researchers investigate scenarios for the
transition of energy systems towards a higher share of renewables, assessing
their social, environmental and economic impact. Their results guide
stakeholders and policy makers in designing resilient and sustainable systems.
Political scientists use surveys to quantify public acceptance of energy
policy, but the complexity and long time horizon of the subject creates
difficulties, both for researchers in posing contextually relevant questions,
and for respondents in assimilating enough information to give meaningful
answers. A population survey was therefore augmented with an online serious
game in which players experience an accurate simulation of current and future
energy provision and manage transition towards a sustainable future. This
interactive environment allows better informed and engaged decisions, and
provides richer information on public opinion. In this paper we motivate and
describe the design of the game and report initial findings on player
characteristics and engagement. We show that a serious game can successfully
attract participants from diverse societal groups and highlight the challenge
of balancing complexity and entertainment.

### 2. [Denoising diffusion models for inverse design of inflatable structures with programmable deformations](http://arxiv.org/pdf/2508.13097v1)

Authors: Sara Karimi, Nikolaos N. Vlassis

Programmable structures are systems whose undeformed geometries and material
property distributions are deliberately designed to achieve prescribed deformed
configurations under specific loading conditions. Inflatable structures are a
prominent example, using internal pressurization to realize large, nonlinear
deformations in applications ranging from soft robotics and deployable
aerospace systems to biomedical devices and adaptive architecture. We present a
generative design framework based on denoising diffusion probabilistic models
(DDPMs) for the inverse design of elastic structures undergoing large,
nonlinear deformations under pressure-driven actuation. The method formulates
the inverse design as a conditional generation task, using geometric
descriptors of target deformed states as inputs and outputting image-based
representations of the undeformed configuration. Representing these
configurations as simple images is achieved by establishing a pre- and
postprocessing pipeline that involves a fixed image processing, simulation
setup, and descriptor extraction methods. Numerical experiments with scalar and
higher-dimensional descriptors show that the framework can quickly produce
diverse undeformed configurations that achieve the desired deformations when
inflated, enabling parallel exploration of viable design candidates while
accommodating complex constraints.

### 3. [Data-driven particle dynamics: Structure-preserving coarse-graining for emergent behavior in non-equilibrium systems](http://arxiv.org/pdf/2508.12569v1)

Authors: Quercus Hernandez, Max Win, Thomas C. O'Connor, Paulo E. Arratia, Nathaniel Trask

Multiscale systems are ubiquitous in science and technology, but are
notoriously challenging to simulate as short spatiotemporal scales must be
appropriately linked to emergent bulk physics. When expensive high-dimensional
dynamical systems are coarse-grained into low-dimensional models, the entropic
loss of information leads to emergent physics which are dissipative,
history-dependent, and stochastic. To machine learn coarse-grained dynamics
from time-series observations of particle trajectories, we propose a framework
using the metriplectic bracket formalism that preserves these properties by
construction; most notably, the framework guarantees discrete notions of the
first and second laws of thermodynamics, conservation of momentum, and a
discrete fluctuation-dissipation balance crucial for capturing non-equilibrium
statistics. We introduce the mathematical framework abstractly before
specializing to a particle discretization. As labels are generally unavailable
for entropic state variables, we introduce a novel self-supervised learning
strategy to identify emergent structural variables. We validate the method on
benchmark systems and demonstrate its utility on two challenging examples: (1)
coarse-graining star polymers at challenging levels of coarse-graining while
preserving non-equilibrium statistics, and (2) learning models from high-speed
video of colloidal suspensions that capture coupling between local
rearrangement events and emergent stochastic dynamics. We provide open-source
implementations in both PyTorch and LAMMPS, enabling large-scale inference and
extensibility to diverse particle-based systems.

### 4. [Shapley Values: Paired-Sampling Approximations](http://arxiv.org/pdf/2508.12947v1)

Authors: Michael Mayer, Mario V. Wüthrich

Originally introduced in cooperative game theory, Shapley values have become
a very popular tool to explain machine learning predictions. Based on Shapley's
fairness axioms, every input (feature component) gets a credit how it
contributes to an output (prediction). These credits are then used to explain
the prediction. The only limitation in computing the Shapley values (credits)
for many different predictions is of computational nature. There are two
popular sampling approximations, sampling KernelSHAP and sampling
PermutationSHAP. Our first novel contributions are asymptotic normality results
for these sampling approximations. Next, we show that the paired-sampling
approaches provide exact results in case of interactions being of maximal order
two. Furthermore, the paired-sampling PermutationSHAP possesses the additive
recovery property, whereas its kernel counterpart does not.

### Computational Geometry

### 1. [On saturated triangulation-free convex geometric graphs](http://arxiv.org/pdf/2508.12789v1)

Authors: David Garber, Chaya Keller, Olga Nissenbaum, Shimon Aviram

A convex geometric graph is a graph whose vertices are the corners of a
convex polygon P in the plane and whose edges are boundary edges and diagonals
of the polygon. It is called triangulation-free if its non-boundary edges do
not contain the set of diagonals of some triangulation of P. Aichholzer et al.
(2010) showed that the maximum number of edges in a triangulation-free convex
geometric graph on n vertices is ${{n}\choose{2}}-(n-2)$, and subsequently,
Keller and Stein (2020) and (independently) Ali et al. (2022) characterized the
triangulation-free graphs with this maximum number of edges.
  We initiate the study of the saturation version of the problem, namely,
characterizing the triangulation-free convex geometric graphs which are not of
the maximum possible size, but yet the addition of any edge to them results in
containing a triangulation. We show that, surprisingly, there exist saturated
graphs with only g(n) = O(n log n) edges. Furthermore, we prove that for any $n
> n_0$ and any $g(n)\leq t \leq {{n}\choose{2}}-(n-2)$, there exists a
saturated graph with n vertices and t edges. In addition, we obtain a complete
characterization of all saturated graphs whose number of edges is
${{n}\choose{2}}-(n-1)$, which is 1 less than the maximum.

### Computation and Language

### 1. [Semantic Anchoring in Agentic Memory: Leveraging Linguistic Structures for Persistent Conversational Context](http://arxiv.org/pdf/2508.12630v1)

Authors: Maitreyi Chatterjee, Devansh Agarwal

Large Language Models (LLMs) have demonstrated impressive fluency and task
competence in conversational settings. However, their effectiveness in
multi-session and long-term interactions is hindered by limited memory
persistence. Typical retrieval-augmented generation (RAG) systems store
dialogue history as dense vectors, which capture semantic similarity but
neglect finer linguistic structures such as syntactic dependencies, discourse
relations, and coreference links. We propose Semantic Anchoring, a hybrid
agentic memory architecture that enriches vector-based storage with explicit
linguistic cues to improve recall of nuanced, context-rich exchanges. Our
approach combines dependency parsing, discourse relation tagging, and
coreference resolution to create structured memory entries. Experiments on
adapted long-term dialogue datasets show that semantic anchoring improves
factual recall and discourse coherence by up to 18% over strong RAG baselines.
We further conduct ablation studies, human evaluations, and error analysis to
assess robustness and interpretability.

### 2. [Beyond GPT-5: Making LLMs Cheaper and Better via Performance-Efficiency Optimized Routing](http://arxiv.org/pdf/2508.12631v1)

Authors: Yiqun Zhang, Hao Li, Jianhao Chen, Hangfan Zhang, Peng Ye, Lei Bai, Shuyue Hu

Balancing performance and efficiency is a central challenge in large language
model (LLM) advancement. GPT-5 addresses this with test-time routing,
dynamically assigning queries to either an efficient or a high-capacity model
during inference. In this work, we present Avengers-Pro, a test-time routing
framework that ensembles LLMs of varying capacities and efficiencies, providing
a unified solution for all performance-efficiency tradeoffs. The Avengers-Pro
embeds and clusters incoming queries, then routes each to the most suitable
model based on a performance-efficiency score. Across 6 challenging benchmarks
and 8 leading models -- including GPT-5-medium, Gemini-2.5-pro, and
Claude-opus-4.1 -- Avengers-Pro achieves state-of-the-art results: by varying a
performance-efficiency trade-off parameter, it can surpass the strongest single
model (GPT-5-medium) by +7% in average accuracy. Moreover, it can match the
average accuracy of the strongest single model at 27% lower cost, and reach
~90% of that performance at 63% lower cost. Last but not least, it achieves a
Pareto frontier, consistently yielding the highest accuracy for any given cost,
and the lowest cost for any given accuracy, among all single models. Code is
available at https://github.com/ZhangYiqun018/AvengersPro.

### 3. [Prompt-Induced Linguistic Fingerprints for LLM-Generated Fake News Detection](http://arxiv.org/pdf/2508.12632v1)

Authors: Chi Wang, Min Gao, Zongwei Wang, Junwei Yin, Kai Shu, Chenghua Lin

With the rapid development of large language models, the generation of fake
news has become increasingly effortless, posing a growing societal threat and
underscoring the urgent need for reliable detection methods. Early efforts to
identify LLM-generated fake news have predominantly focused on the textual
content itself; however, because much of that content may appear coherent and
factually consistent, the subtle traces of falsification are often difficult to
uncover. Through distributional divergence analysis, we uncover prompt-induced
linguistic fingerprints: statistically distinct probability shifts between
LLM-generated real and fake news when maliciously prompted. Based on this
insight, we propose a novel method named Linguistic Fingerprints Extraction
(LIFE). By reconstructing word-level probability distributions, LIFE can find
discriminative patterns that facilitate the detection of LLM-generated fake
news. To further amplify these fingerprint patterns, we also leverage
key-fragment techniques that accentuate subtle linguistic differences, thereby
improving detection reliability. Our experiments show that LIFE achieves
state-of-the-art performance in LLM-generated fake news and maintains high
performance in human-written fake news. The code and data are available at
https://anonymous.4open.science/r/LIFE-E86A.

### 4. [DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](http://arxiv.org/pdf/2508.12726v1)

Authors: Weize Liu, Yongchi Zhao, Yijia Luo, Mingyu Xu, Jiaheng Liu, Yanan Li, Xiguo Hu, Yuchi Xu, Wenbo Su, Bo Zheng

Large language models (LLMs) have achieved remarkable success in many natural
language tasks but still struggle with complex, multi-step reasoning,
particularly across diverse disciplines. Existing reasoning datasets often
either lack disciplinary breadth or the structural depth necessary to elicit
robust reasoning behaviors. We propose DESIGNER: a DESIGN-logic-guidEd
Reasoning data synthesis pipeline that leverages naturally available, extensive
raw documents (book corpus and web corpus) to generate multidisciplinary
challenging questions. A core innovation of our approach is the introduction of
a Design Logic concept, which mimics the question-creation process of human
educators. We use LLMs to reverse-engineer and abstract over 120,000 design
logics from existing questions across various disciplines. By matching these
design logics with disciplinary source materials, we are able to create
reasoning questions that far surpass the difficulty and diversity of existing
datasets. Based on this pipeline, we synthesized two large-scale reasoning
datasets that span 75 disciplines: Design-Logic-Reasoning-Book (DLR-Book),
containing 3.04 million challenging questions synthesized from the book corpus,
and Design-Logic-Reasoning-Web (DLR-Web), with 1.66 million challenging
questions from the web corpus. Our data analysis demonstrates that the
questions synthesized by our method exhibit substantially greater difficulty
and diversity than those in the baseline datasets. We validate the
effectiveness of these datasets by conducting SFT experiments on the
Qwen3-8B-Base and Qwen3-4B-Base models. The results show that our dataset
significantly outperforms existing multidisciplinary datasets of the same
volume. Training with the full datasets further enables the models to surpass
the multidisciplinary reasoning performance of the official Qwen3-8B and
Qwen3-4B models.

### 5. [From SALAMANDRA to SALAMANDRATA: BSC Submission for WMT25 General Machine Translation Shared Task](http://arxiv.org/pdf/2508.12774v1)

Authors: Javier Garcia Gilabert, Xixian Liao, Severino Da Dalt, Ella Bohman, Audrey Mash, Francesca De Luca Fornaciari, Irene Baucells, Joan Llop, Miguel Claramunt Argote, Carlos Escolano, Maite Melero

In this paper, we present the SALAMANDRATA family of models, an improved
iteration of SALAMANDRA LLMs (Gonzalez-Agirre et al., 2025) specifically
trained to achieve strong performance in translation-related tasks for 38
European languages. SALAMANDRATA comes in two scales: 2B and 7B parameters. For
both versions, we applied the same training recipe with a first step of
continual pre-training on parallel data, and a second step of supervised
fine-tuning on high-quality instructions. The BSC submission to the WMT25
General Machine Translation shared task is based on the 7B variant of
SALAMANDRATA. We first adapted the model vocabulary to support the additional
non-European languages included in the task. This was followed by a second
phase of continual pre-training and supervised fine-tuning, carefully designed
to optimize performance across all translation directions for this year's
shared task. For decoding, we employed two quality-aware strategies: Minimum
Bayes Risk Decoding and Tuned Re-ranking using COMET and COMET-KIWI
respectively. We publicly release both the 2B and 7B versions of SALAMANDRATA,
along with the newer SALAMANDRATA-V2 model, on Hugging Face1

### 6. [HeteroRAG: A Heterogeneous Retrieval-Augmented Generation Framework for Medical Vision Language Tasks](http://arxiv.org/pdf/2508.12778v1)

Authors: Zhe Chen, Yusheng Liao, Shuyang Jiang, Zhiyuan Zhu, Haolin Li, Yanfeng Wang, Yu Wang

Medical large vision-language Models (Med-LVLMs) have shown promise in
clinical applications but suffer from factual inaccuracies and unreliable
outputs, posing risks in real-world diagnostics. While retrieval-augmented
generation has emerged as a potential solution, current medical multimodal RAG
systems are unable to perform effective retrieval across heterogeneous sources.
The irrelevance of retrieved reports affects the factuality of analysis, while
insufficient knowledge affects the credibility of clinical decision-making. To
bridge the gap, we construct MedAtlas, which includes extensive multimodal
report repositories and diverse text corpora. Based on it, we present
HeteroRAG, a novel framework that enhances Med-LVLMs through heterogeneous
knowledge sources. The framework introduces Modality-specific CLIPs for
effective report retrieval and a Multi-corpora Query Generator for dynamically
constructing queries for diverse corpora. Incorporating knowledge from such
multifaceted sources, Med-LVLM is then trained with Heterogeneous Knowledge
Preference Tuning to achieve cross-modality and multi-source knowledge
alignment. Extensive experiments across 12 datasets and 3 modalities
demonstrate that the proposed HeteroRAG achieves state-of-the-art performance
in most medical vision language benchmarks, significantly improving factual
accuracy and reliability of Med-LVLMs.

### 7. [When Alignment Hurts: Decoupling Representational Spaces in Multilingual Models](http://arxiv.org/pdf/2508.12803v1)

Authors: Ahmed Elshabrawy, Hour Kaing, Haiyue Song, Alham Fikri Aji, Hideki Tanaka, Masao Utiyama, Raj Dabre

Alignment with high-resource standard languages is often assumed to aid the
modeling of related low-resource varieties. We challenge this assumption by
demonstrating that excessive representational entanglement with a dominant
variety, such as Modern Standard Arabic (MSA) in relation to Arabic dialects,
can actively hinder generative modeling. We present the first comprehensive
causal study of this phenomenon by analyzing and directly intervening in the
internal representation geometry of large language models (LLMs). Our key
contribution is an online variational probing framework that continuously
estimates the subspace of the standard variety during fine-tuning, enabling
projection-based decoupling from this space. While our study uses Arabic as a
case due to its unusually rich parallel resources across 25 dialects, the
broader motivation is methodological: dialectal MT serves as a controlled proxy
for generative tasks where comparable multi-variety corpora are unavailable.
Across 25 dialects, our intervention improves generation quality by up to +4.9
chrF++ and +2.0 on average compared to standard fine-tuning, despite a measured
tradeoff in standard-language performance. These results provide causal
evidence that subspace dominance by high-resource varieties can restrict
generative capacity for related varieties. More generally, we unify geometric
and information-theoretic probing with subspace-level causal interventions,
offering practical tools for improving generative modeling in closely related
language families and, more broadly, for controlling representational
allocation in multilingual and multi-domain LLMs. Code will be released.

### 8. [ding-01 :ARG0: An AMR Corpus for Spontaneous French Dialogue](http://arxiv.org/pdf/2508.12819v1)

Authors: Jeongwoo Kang, Maria Boritchev, Maximin Coavoux

We present our work to build a French semantic corpus by annotating French
dialogue in Abstract Meaning Representation (AMR). Specifically, we annotate
the DinG corpus, consisting of transcripts of spontaneous French dialogues
recorded during the board game Catan. As AMR has insufficient coverage of the
dynamics of spontaneous speech, we extend the framework to better represent
spontaneous speech and sentence structures specific to French. Additionally, to
support consistent annotation, we provide an annotation guideline detailing
these extensions. We publish our corpus under a free license (CC-SA-BY). We
also train and evaluate an AMR parser on our data. This model can be used as an
assistance annotation tool to provide initial annotations that can be refined
by human annotators. Our work contributes to the development of semantic
resources for French dialogue.

### 9. [It takes a village to write a book: Mapping anonymous contributions in Stephen Langton's Quaestiones Theologiae](http://arxiv.org/pdf/2508.12830v1)

Authors: Jan Maliszewski

While the indirect evidence suggests that already in the early scholastic
period the literary production based on records of oral teaching (so-called
reportationes) was not uncommon, there are very few sources commenting on the
practice. This paper details the design of a study applying stylometric
techniques of authorship attribution to a collection developed from
reportationes -- Stephen Langton's Quaestiones Theologiae -- aiming to uncover
layers of editorial work and thus validate some hypotheses regarding the
collection's formation. Following Camps, Cl\'erice, and Pinche (2021), I
discuss the implementation of an HTR pipeline and stylometric analysis based on
the most frequent words, POS tags, and pseudo-affixes. The proposed study will
offer two methodological gains relevant to computational research on the
scholastic tradition: it will directly compare performance on manually composed
and automatically extracted data, and it will test the validity of
transformer-based OCR and automated transcription alignment for workflows
applied to scholastic Latin corpora. If successful, this study will provide an
easily reusable template for the exploratory analysis of collaborative literary
production stemming from medieval universities.

### 10. [Analyzing Information Sharing and Coordination in Multi-Agent Planning](http://arxiv.org/pdf/2508.12981v1)

Authors: Tianyue Ou, Saujas Vaduguru, Daniel Fried

Multi-agent systems (MASs) have pushed the boundaries of large language model
(LLM) agents in domains such as web research and software engineering. However,
long-horizon, multi-constraint planning tasks involve conditioning on detailed
information and satisfying complex interdependent constraints, which can pose a
challenge for these systems. In this study, we construct an LLM-based MAS for a
travel planning task which is representative of these challenges. We evaluate
the impact of a notebook to facilitate information sharing, and evaluate an
orchestrator agent to improve coordination in free form conversation between
agents. We find that the notebook reduces errors due to hallucinated details by
18%, while an orchestrator directs the MAS to focus on and further reduce
errors by up to 13.5% within focused sub-areas. Combining both mechanisms
achieves a 25% final pass rate on the TravelPlanner benchmark, a 17.5% absolute
improvement over the single-agent baseline's 7.5% pass rate. These results
highlight the potential of structured information sharing and reflective
orchestration as key components in MASs for long horizon planning with LLMs.

### Cryptography and Security

### 1. [The Hidden Cost of Correlation: Rethinking Privacy Leakage in Local Differential Privacy](http://arxiv.org/pdf/2508.12539v1)

Authors: Sandaru Jayawardana, Sennur Ulukus, Ming Ding, Kanchana Thilakarathna

Local differential privacy (LDP) has emerged as a promising paradigm for
privacy-preserving data collection in distributed systems, where users
contribute multi-dimensional records with potentially correlated attributes.
Recent work has highlighted that correlation-induced privacy leakage (CPL)
plays a critical role in shaping the privacy-utility trade-off under LDP,
especially when correlations exist among attributes. Nevertheless, it remains
unclear to what extent the prevailing assumptions and proposed solutions are
valid and how significant CPL is in real-world data. To address this gap, we
first perform a comprehensive statistical analysis of five widely used LDP
mechanisms -- GRR, RAPPOR, OUE, OLH and Exponential mechanism -- to assess CPL
across four real-world datasets. We identify that many primary assumptions and
metrics in current approaches fall short of accurately characterising these
leakages. Moreover, current studies have been limited to a set of pure LDP
(i.e., {\delta = 0}) mechanisms. In response, we develop the first algorithmic
framework to theoretically quantify CPL for any general approximated LDP
(({\varepsilon},{\delta})-LDP) mechanism. We validate our theoretical results
against empirical statistical results and provide a theoretical explanation for
the observed statistical patterns. Finally, we propose two novel benchmarks to
validate correlation analysis algorithms and evaluate the utility vs CPL of LDP
mechanisms. Further, we demonstrate how these findings can be applied to
achieve an efficient privacy-utility trade-off in real-world data governance.

### 2. [DEFENDCLI: {Command-Line} Driven Attack Provenance Examination](http://arxiv.org/pdf/2508.12553v1)

Authors: Peilun Wu, Nan Sun, Nour Moustafa, Youyang Qu, Ming Ding

Endpoint Detection and Response (EDR) solutions embrace the method of attack
provenance graph to discover unknown threats through system event correlation.
However, this method still faces some unsolved problems in the fields of
interoperability, reliability, flexibility, and practicability to deliver
actionable results. Our research highlights the limitations of current
solutions in detecting obfuscation, correlating attacks, identifying
low-frequency events, and ensuring robust context awareness in relation to
command-line activities. To address these challenges, we introduce DEFENDCLI,
an innovative system leveraging provenance graphs that, for the first time,
delves into command-line-level detection. By offering finer detection
granularity, it addresses a gap in modern EDR systems that has been overlooked
in previous research. Our solution improves the precision of the information
representation by evaluating differentiation across three levels: unusual
system process calls, suspicious command-line executions, and infrequent
external network connections. This multi-level approach enables EDR systems to
be more reliable in complex and dynamic environments. Our evaluation
demonstrates that DEFENDCLI improves precision by approximately 1.6x compared
to the state-of-the-art methods on the DARPA Engagement Series attack datasets.
Extensive real-time industrial testing across various attack scenarios further
validates its practical effectiveness. The results indicate that DEFENDCLI not
only detects previously unknown attack instances, which are missed by other
modern commercial solutions, but also achieves a 2.3x improvement in precision
over the state-of-the-art research work.

### 3. [Reducing False Positives with Active Behavioral Analysis for Cloud Security](http://arxiv.org/pdf/2508.12584v1)

Authors: Dikshant, Verma

Rule-based cloud security posture management (CSPM) solutions are known to
produce a lot of false positives based on the limited contextual understanding
and dependence on static heuristics testing. This paper introduces a
validation-driven methodology that integrates active behavioral testing in
cloud security posture management solution(s) to evaluate the exploitability of
policy violations in real time. The proposed system employs lightweight and
automated probes, built from open-source tools, validation scripts, and
penetration testing test cases, to simulate adversarial attacks on
misconfigured or vulnerable cloud assets without any impact to the cloud
services or environment. For instance, cloud services may be flagged as
publicly exposed and vulnerable despite being protected by access control
layers, or secure policies, resulting in non-actionable alerts that consumes
analysts time during manual validation. Through controlled experimentation in a
reproducible AWS setup, we evaluated the reduction in false positive rates
across various misconfiguration and vulnerable alerts. Our findings indicate an
average reduction of 93\% in false positives. Furthermore, the framework
demonstrates low latency performance. These results demonstrate a scalable
method to improve detection accuracy and analyst productivity in large cloud
environments. While our evaluation focuses on AWS, the architecture is modular
and extensible to multi-cloud setups.

### 4. [UAV Individual Identification via Distilled RF Fingerprints-Based LLM in ISAC Networks](http://arxiv.org/pdf/2508.12597v1)

Authors: Haolin Zheng, Ning Gao, Donghong Cai, Shi Jin, Michail Matthaiou

Unmanned aerial vehicle (UAV) individual (ID) identification is a critical
security surveillance strategy in low-altitude integrated sensing and
communication (ISAC) networks. In this paper, we propose a novel dynamic
knowledge distillation (KD)-enabled wireless radio frequency fingerprint large
language model (RFF-LLM) framework for UAV ID identification. First, we propose
an RFF-LLM framework based on the modified GPT-2 model to improve the
identification accuracy in complex outdoor environments. Then, considering the
parameter overhead of the RFF-LLM, we design a dynamic KD strategy to compress
the model. Specifically, the proximal policy optimization (PPO) algorithm is
employed to dynamically adjust the distillation temperature, overcoming the
local optimum dilemma inherent in static KD. As a next step, the knowledge of
the RFF-LLM is adequately transferred to the lightweight Lite-HRNet model.
Finally, our experiments are conducted based on the self-built drone RFF
dataset of Release one, namely DRFF-R1, by collecting the I/Q signals of 20
commercial UAVs in channel 149. The experiment results show that the proposed
framework achieves 98.38\% ID identification accuracy with merely 0.15 million
parameters and 2.74 ms response time, which outperforms the benchmarks.

### 5. [Consiglieres in the Shadow: Understanding the Use of Uncensored Large Language Models in Cybercrimes](http://arxiv.org/pdf/2508.12622v1)

Authors: Zilong Lin, Zichuan Li, Xiaojing Liao, XiaoFeng Wang

The advancement of AI technologies, particularly Large Language Models
(LLMs), has transformed computing while introducing new security and privacy
risks. Prior research shows that cybercriminals are increasingly leveraging
uncensored LLMs (ULLMs) as backends for malicious services. Understanding these
ULLMs has been hindered by the challenge of identifying them among the vast
number of open-source LLMs hosted on platforms like Hugging Face. In this
paper, we present the first systematic study of ULLMs, overcoming this
challenge by modeling relationships among open-source LLMs and between them and
related data, such as fine-tuning, merging, compressing models, and using or
generating datasets with harmful content. Representing these connections as a
knowledge graph, we applied graph-based deep learning to discover over 11,000
ULLMs from a small set of labeled examples and uncensored datasets.
  A closer analysis of these ULLMs reveals their alarming scale and usage. Some
have been downloaded over a million times, with one over 19 million installs.
These models -- created through fine-tuning, merging, or compression of other
models -- are capable of generating harmful content, including hate speech,
violence, erotic material, and malicious code. Evidence shows their integration
into hundreds of malicious applications offering services like erotic
role-play, child pornography, malicious code generation, and more. In addition,
underground forums reveal criminals sharing techniques and scripts to build
cheap alternatives to commercial malicious LLMs. These findings highlight the
widespread abuse of LLM technology and the urgent need for effective
countermeasures against this growing threat.

### 6. [MPOCryptoML: Multi-Pattern based Off-Chain Crypto Money Laundering Detection](http://arxiv.org/pdf/2508.12641v1)

Authors: Yasaman Samadi, Hai Dong, Xiaoyu Xia

Recent advancements in money laundering detection have demonstrated the
potential of using graph neural networks to capture laundering patterns
accurately. However, existing models are not explicitly designed to detect the
diverse patterns of off-chain cryptocurrency money laundering. Neglecting any
laundering pattern introduces critical detection gaps, as each pattern reflects
unique transactional structures that facilitate the obfuscation of illicit fund
origins and movements. Failure to account for these patterns may result in
under-detection or omission of specific laundering activities, diminishing
model accuracy and allowing schemes to bypass detection. To address this gap,
we propose the MPOCryptoML model to effectively detect multiple laundering
patterns in cryptocurrency transactions. MPOCryptoML includes the development
of a multi-source Personalized PageRank algorithm to identify random laundering
patterns. Additionally, we introduce two novel algorithms by analyzing the
timestamp and weight of transactions in high-volume financial networks to
detect various money laundering structures, including fan-in, fan-out,
bipartite, gather-scatter, and stack patterns. We further examine correlations
between these patterns using a logistic regression model. An anomaly score
function integrates results from each module to rank accounts by anomaly score,
systematically identifying high-risk accounts. Extensive experiments on public
datasets including Elliptic++, Ethereum fraud detection, and Wormhole
transaction datasets validate the efficacy and efficiency of MPOCryptoML.
Results show consistent performance gains, with improvements up to 9.13% in
precision, up to 10.16% in recall, up to 7.63% in F1-score, and up to 10.19% in
accuracy.

### 7. [The covering radius of Butson Hadamard codes for the homogeneous metric](http://arxiv.org/pdf/2508.12859v1)

Authors: Xingxing Xu, Minjia Shi, Patrick Sole

Butson matrices are complex Hadamard matrices with entries in the complex
roots of unity of given order. There is an interesting code in phase space
related to this matrix (Armario et al. 2023). We study the covering radius of
Butson Hadamard codes for the homogeneous metric, a metric defined uniquely, up
to scaling, for a commutative ring alphabet that is Quasi Frobenius. An upper
bound is derived by an orthogonal array argument. A lower bound relies on the
existence of bent sequences in the sense of (Shi et al. 2022). This latter
bound generalizes a bound of (Armario et al. 2025) for the Hamming metric.

### 8. [Supporting Socially Constrained Private Communications with SecureWhispers](http://arxiv.org/pdf/2508.12870v1)

Authors: Vinod Khandkar, Kieron Ivy Turk, Ehsan Toreini, Nishanth Sastry

Rapidly changing social norms and national, legal, and political conditions
socially constrain people from discussing sensitive topics such as sexuality or
religion. Such constrained, vulnerable minorities are often worried about
inadvertent information disclosure and may be unsure about the extent to which
their communications are being monitored in public or semi-public spaces like
workplaces or cafes. Personal devices extend trust to the digital domain,
making it desirable to have strictly private communication between trusted
devices. Currently, messaging services like WhatsApp provide alternative means
for exchanging sensitive private information, while personal safety apps such
as Noonlight enable private signaling. However, these rely on third-party
mechanisms for secure and private communication, which may not be accessible
for justifiable reasons, such as insecure internet access or companion device
connections. In these cases, it is challenging to achieve communication that is
strictly private between two devices instead of user accounts without any
dependency on third-party infrastructure. The goal of this paper is to support
private communications by setting up a shared secret between two or more
devices without sending any data on the network. We develop a method to create
a shared secret between phones by shaking them together. Each device extracts
the shared randomness from the shake, then conditions the randomness to 7.798
bits per byte of key material. This paper proposes three different applications
of this generated shared secret: message obfuscation, trust delegation, and
encrypted beacons. We have implemented the message obfuscation on Android as an
independent app that can be used for private communication with trusted
contacts. We also present research on the usability, design considerations, and
further integration of these tools in mainstream services.

### 9. [Prescriptive Zero Trust- Assessing the impact of zero trust on cyber attack prevention](http://arxiv.org/pdf/2508.12953v1)

Authors: Samuel Aiello

Increasingly sophisticated and varied cyber threats necessitate ever
improving enterprise security postures. For many organizations today, those
postures have a foundation in the Zero Trust Architecture. This strategy sees
trust as something an enterprise must not give lightly or assume too broadly.
Understanding the ZTA and its numerous controls centered around the idea of not
trusting anything inside or outside the network without verification, will
allow organizations to comprehend and leverage this increasingly common
paradigm. The ZTA, unlike many other regulatory frameworks, is not tightly
defined. The research assesses the likelihood of quantifiable guidelines that
measure cybersecurity maturity for an enterprise organization in relation to
ZTA implementation. This is a new, data driven methodology for quantifying
cyber resilience enabled by the adoption of Zero Trust principles to
pragmatically address the critical need of organizations. It also looks at the
practical aspects ZTA has on capabilities in deterring cyberattacks on a
network. The outcomes of this research define a prescriptive set of key
technical controls across identity verification, microsegmentation, data
encryption, analytics, and orchestration that characterize the comprehensive
ZTA deployment. By evaluating the depth of integration for each control
component and aligning to industry best practices, the study's results help
assess an organization's ZTA maturity level on a scale from Initial to
Optimized adoption. The research's resultant four tier model demarcates phases
for an organization on its security transformation journey, with each tier
adding to the capability of the last.

### 10. [AuthenTree: A Scalable MPC-Based Distributed Trust Architecture for Chiplet-based Heterogeneous Systems](http://arxiv.org/pdf/2508.13033v1)

Authors: Ishraq Tashdid, Tasnuva Farheen, Sazadur Rahman

The rapid adoption of chiplet-based heterogeneous integration is reshaping
semiconductor design by enabling modular, scalable, and faster time-to-market
solutions for AI and high-performance computing. However, multi-vendor assembly
in post-fabrication environments fragments the supply chain and exposes SiP
systems to serious security threats, including cloning, overproduction, and
chiplet substitution. Existing authentication solutions depend on trusted
integrators or centralized security anchors, which can expose sensitive data or
create single points of failure. We introduce AuthenTree, a distributed
authentication framework that leverages multi-party computation (MPC) in a
scalable tree-based architecture, removing the need for dedicated security
hardware or centralized trust. AuthenTree enables secure chiplet validation
without revealing raw signatures, distributing trust across multiple integrator
chiplets. Our evaluation in five SiP benchmarks demonstrates that AuthenTree
imposes minimal overhead, with an area as low as 0.48% (7,000 sq-micrometers),
an overhead power under 0.5%, and an authentication latency below 1
microsecond, surpassing previous work in some cases by 700 times. These results
establish AuthenTree as an efficient, robust, and scalable solution for
next-generation chiplet-based security in zero-trust SiP environments.

### Computer Vision and Pattern Recognition

### 1. [REVEAL -- Reasoning and Evaluation of Visual Evidence through Aligned Language](http://arxiv.org/pdf/2508.12543v1)

Authors: Ipsita Praharaj, Yukta Butala, Yash Butala

The rapid advancement of generative models has intensified the challenge of
detecting and interpreting visual forgeries, necessitating robust frameworks
for image forgery detection while providing reasoning as well as localization.
While existing works approach this problem using supervised training for
specific manipulation or anomaly detection in the embedding space,
generalization across domains remains a challenge. We frame this problem of
forgery detection as a prompt-driven visual reasoning task, leveraging the
semantic alignment capabilities of large vision-language models. We propose a
framework, `REVEAL` (Reasoning and Evaluation of Visual Evidence through
Aligned Language), that incorporates generalized guidelines. We propose two
tangential approaches - (1) Holistic Scene-level Evaluation that relies on the
physics, semantics, perspective, and realism of the image as a whole and (2)
Region-wise anomaly detection that splits the image into multiple regions and
analyzes each of them. We conduct experiments over datasets from different
domains (Photoshop, DeepFake and AIGC editing). We compare the Vision Language
Models against competitive baselines and analyze the reasoning provided by
them.

### 2. [Structure-preserving Feature Alignment for Old Photo Colorization](http://arxiv.org/pdf/2508.12570v1)

Authors: Yingxue Pang, Xin Jin, Jun Fu, Zhibo Chen

Deep learning techniques have made significant advancements in
reference-based colorization by training on large-scale datasets. However,
directly applying these methods to the task of colorizing old photos is
challenging due to the lack of ground truth and the notorious domain gap
between natural gray images and old photos. To address this issue, we propose a
novel CNN-based algorithm called SFAC, i.e., Structure-preserving Feature
Alignment Colorizer. SFAC is trained on only two images for old photo
colorization, eliminating the reliance on big data and allowing direct
processing of the old photo itself to overcome the domain gap problem. Our
primary objective is to establish semantic correspondence between the two
images, ensuring that semantically related objects have similar colors. We
achieve this through a feature distribution alignment loss that remains robust
to different metric choices. However, utilizing robust semantic correspondence
to transfer color from the reference to the old photo can result in inevitable
structure distortions. To mitigate this, we introduce a structure-preserving
mechanism that incorporates a perceptual constraint at the feature level and a
frozen-updated pyramid at the pixel level. Extensive experiments demonstrate
the effectiveness of our method for old photo colorization, as confirmed by
qualitative and quantitative metrics.

### 3. [Foundation Model for Skeleton-Based Human Action Understanding](http://arxiv.org/pdf/2508.12586v1)

Authors: Hongsong Wang, Wanjiang Weng, Junbo Wang, Fang Zhao, Guo-Sen Xie, Xin Geng, Liang Wang

Human action understanding serves as a foundational pillar in the field of
intelligent motion perception. Skeletons serve as a modality- and
device-agnostic representation for human modeling, and skeleton-based action
understanding has potential applications in humanoid robot control and
interaction. \RED{However, existing works often lack the scalability and
generalization required to handle diverse action understanding tasks. There is
no skeleton foundation model that can be adapted to a wide range of action
understanding tasks}. This paper presents a Unified Skeleton-based Dense
Representation Learning (USDRL) framework, which serves as a foundational model
for skeleton-based human action understanding. USDRL consists of a
Transformer-based Dense Spatio-Temporal Encoder (DSTE), Multi-Grained Feature
Decorrelation (MG-FD), and Multi-Perspective Consistency Training (MPCT). The
DSTE module adopts two parallel streams to learn temporal dynamic and spatial
structure features. The MG-FD module collaboratively performs feature
decorrelation across temporal, spatial, and instance domains to reduce
dimensional redundancy and enhance information extraction. The MPCT module
employs both multi-view and multi-modal self-supervised consistency training.
The former enhances the learning of high-level semantics and mitigates the
impact of low-level discrepancies, while the latter effectively facilitates the
learning of informative multimodal features. We perform extensive experiments
on 25 benchmarks across across 9 skeleton-based action understanding tasks,
covering coarse prediction, dense prediction, and transferred prediction. Our
approach significantly outperforms the current state-of-the-art methods. We
hope that this work would broaden the scope of research in skeleton-based
action understanding and encourage more attention to dense prediction tasks.

### 4. [Multimodal Chain of Continuous Thought for Latent-Space Reasoning in Vision-Language Models](http://arxiv.org/pdf/2508.12587v1)

Authors: Tan-Hanh Pham, Chris Ngo

Many reasoning techniques for large multimodal models adapt language model
approaches, such as Chain-of-Thought (CoT) prompting, which express reasoning
as word sequences. While effective for text, these methods are suboptimal for
multimodal contexts, struggling to align audio, visual, and textual information
dynamically. To explore an alternative paradigm, we propose the Multimodal
Chain of Continuous Thought (MCOUT), which enables reasoning directly in a
joint latent space rather than in natural language. In MCOUT, the reasoning
state is represented as a continuous hidden vector, iteratively refined and
aligned with visual and textual embeddings, inspired by human reflective
cognition. We develop two variants: MCOUT-Base, which reuses the language
model`s last hidden state as the continuous thought for iterative reasoning,
and MCOUT-Multi, which integrates multimodal latent attention to strengthen
cross-modal alignment between visual and textual features. Experiments on
benchmarks including MMMU, ScienceQA, and MMStar show that MCOUT consistently
improves multimodal reasoning, yielding up to 8.23% accuracy gains over strong
baselines and improving BLEU scores up to 8.27% across multiple-choice and
open-ended tasks. These findings highlight latent continuous reasoning as a
promising direction for advancing LMMs beyond language-bound CoT, offering a
scalable framework for human-like reflective multimodal inference. Code is
available at https://github.com/Hanhpt23/OmniMod.

### 5. [ViLaD: A Large Vision Language Diffusion Framework for End-to-End Autonomous Driving](http://arxiv.org/pdf/2508.12603v1)

Authors: Can Cui, Yupeng Zhou, Juntong Peng, Sung-Yeon Park, Zichong Yang, Prashanth Sankaranarayanan, Jiaru Zhang, Ruqi Zhang, Ziran Wang

End-to-end autonomous driving systems built on Vision Language Models (VLMs)
have shown significant promise, yet their reliance on autoregressive
architectures introduces some limitations for real-world applications. The
sequential, token-by-token generation process of these models results in high
inference latency and cannot perform bidirectional reasoning, making them
unsuitable for dynamic, safety-critical environments. To overcome these
challenges, we introduce ViLaD, a novel Large Vision Language Diffusion (LVLD)
framework for end-to-end autonomous driving that represents a paradigm shift.
ViLaD leverages a masked diffusion model that enables parallel generation of
entire driving decision sequences, significantly reducing computational
latency. Moreover, its architecture supports bidirectional reasoning, allowing
the model to consider both past and future simultaneously, and supports
progressive easy-first generation to iteratively improve decision quality. We
conduct comprehensive experiments on the nuScenes dataset, where ViLaD
outperforms state-of-the-art autoregressive VLM baselines in both planning
accuracy and inference speed, while achieving a near-zero failure rate.
Furthermore, we demonstrate the framework's practical viability through a
real-world deployment on an autonomous vehicle for an interactive parking task,
confirming its effectiveness and soundness for practical applications.

### 6. [ViDA-UGC: Detailed Image Quality Analysis via Visual Distortion Assessment for UGC Images](http://arxiv.org/pdf/2508.12605v1)

Authors: Wenjie Liao, Jieyu Yuan, Yifang Xu, Chunle Guo, Zilong Zhang, Jihong Li, Jiachen Fu, Haotian Fan, Tao Li, Junhui Cui, Chongyi Li

Recent advances in Multimodal Large Language Models (MLLMs) have introduced a
paradigm shift for Image Quality Assessment (IQA) from unexplainable image
quality scoring to explainable IQA, demonstrating practical applications like
quality control and optimization guidance. However, current explainable IQA
methods not only inadequately use the same distortion criteria to evaluate both
User-Generated Content (UGC) and AI-Generated Content (AIGC) images, but also
lack detailed quality analysis for monitoring image quality and guiding image
restoration. In this study, we establish the first large-scale Visual
Distortion Assessment Instruction Tuning Dataset for UGC images, termed
ViDA-UGC, which comprises 11K images with fine-grained quality grounding,
detailed quality perception, and reasoning quality description data. This
dataset is constructed through a distortion-oriented pipeline, which involves
human subject annotation and a Chain-of-Thought (CoT) assessment framework.
This framework guides GPT-4o to generate quality descriptions by identifying
and analyzing UGC distortions, which helps capturing rich low-level visual
features that inherently correlate with distortion patterns. Moreover, we
carefully select 476 images with corresponding 6,149 question answer pairs from
ViDA-UGC and invite a professional team to ensure the accuracy and quality of
GPT-generated information. The selected and revised data further contribute to
the first UGC distortion assessment benchmark, termed ViDA-UGC-Bench.
Experimental results demonstrate the effectiveness of the ViDA-UGC and CoT
framework for consistently enhancing various image quality analysis abilities
across multiple base MLLMs on ViDA-UGC-Bench and Q-Bench, even surpassing
GPT-4o.

### 7. [WIPES: Wavelet-based Visual Primitives](http://arxiv.org/pdf/2508.12615v1)

Authors: Wenhao Zhang, Hao Zhu, Delong Wu, Di Kang, Linchao Bao, Zhan Ma, Xun Cao

Pursuing a continuous visual representation that offers flexible frequency
modulation and fast rendering speed has recently garnered increasing attention
in the fields of 3D vision and graphics. However, existing representations
often rely on frequency guidance or complex neural network decoding, leading to
spectrum loss or slow rendering. To address these limitations, we propose
WIPES, a universal Wavelet-based vIsual PrimitivES for representing
multi-dimensional visual signals. Building on the spatial-frequency
localization advantages of wavelets, WIPES effectively captures both the
low-frequency "forest" and the high-frequency "trees." Additionally, we develop
a wavelet-based differentiable rasterizer to achieve fast visual rendering.
Experimental results on various visual tasks, including 2D image
representation, 5D static and 6D dynamic novel view synthesis, demonstrate that
WIPES, as a visual primitive, offers higher rendering quality and faster
inference than INR-based methods, and outperforms Gaussian-based
representations in rendering quality.

### 8. [Creative4U: MLLMs-based Advertising Creative Image Selector with Comparative Reasoning](http://arxiv.org/pdf/2508.12628v1)

Authors: Yukang Lin, Xiang Zhang, Shichang Jia, Bowen Wan, Chenghan Fu, Xudong Ren, Yueran Liu, Wanxian Guan, Pengji Wang, Jian Xu, Bo Zheng, Baolin Liu

Creative image in advertising is the heart and soul of e-commerce platform.
An eye-catching creative image can enhance the shopping experience for users,
boosting income for advertisers and advertising revenue for platforms. With the
advent of AIGC technology, advertisers can produce large quantities of creative
images at minimal cost. However, they struggle to assess the creative quality
to select. Existing methods primarily focus on creative ranking, which fails to
address the need for explainable creative selection.
  In this work, we propose the first paradigm for explainable creative
assessment and selection. Powered by multimodal large language models (MLLMs),
our approach integrates the assessment and selection of creative images into a
natural language generation task. To facilitate this research, we construct
CreativePair, the first comparative reasoning-induced creative dataset
featuring 8k annotated image pairs, with each sample including a label
indicating which image is superior. Additionally, we introduce Creative4U
(pronounced Creative for You), a MLLMs-based creative selector that takes into
account users' interests. Through Reason-to-Select RFT, which includes
supervised fine-tuning with Chain-of-Thought (CoT-SFT) and Group Relative
Policy Optimization (GRPO) based reinforcement learning, Creative4U is able to
evaluate and select creative images accurately. Both offline and online
experiments demonstrate the effectiveness of our approach. Our code and dataset
will be made public to advance research and industrial applications.

### 9. [Learn Faster and Remember More: Balancing Exploration and Exploitation for Continual Test-time Adaptation](http://arxiv.org/pdf/2508.12643v1)

Authors: Pinci Yang, Peisong Wen, Ke Ma, Qianqian Xu

Continual Test-Time Adaptation (CTTA) aims to adapt a source pre-trained
model to continually changing target domains during inference. As a fundamental
principle, an ideal CTTA method should rapidly adapt to new domains
(exploration) while retaining and exploiting knowledge from previously
encountered domains to handle similar domains in the future. Despite
significant advances, balancing exploration and exploitation in CTTA is still
challenging: 1) Existing methods focus on adjusting predictions based on
deep-layer outputs of neural networks. However, domain shifts typically affect
shallow features, which are inefficient to be adjusted from deep predictions,
leading to dilatory exploration; 2) A single model inevitably forgets knowledge
of previous domains during the exploration, making it incapable of exploiting
historical knowledge to handle similar future domains. To address these
challenges, this paper proposes a mean teacher framework that strikes an
appropriate Balance between Exploration and Exploitation (BEE) during the CTTA
process. For the former challenge, we introduce a Multi-level Consistency
Regularization (MCR) loss that aligns the intermediate features of the student
and teacher models, accelerating adaptation to the current domain. For the
latter challenge, we employ a Complementary Anchor Replay (CAR) mechanism to
reuse historical checkpoints (anchors), recovering complementary knowledge for
diverse domains. Experiments show that our method significantly outperforms
state-of-the-art methods on several benchmarks, demonstrating its effectiveness
for CTTA tasks.

### 10. [DyCrowd: Towards Dynamic Crowd Reconstruction from a Large-scene Video](http://arxiv.org/pdf/2508.12644v1)

Authors: Hao Wen, Hongbo Kang, Jian Ma, Jing Huang, Yuanwang Yang, Haozhe Lin, Yu-Kun Lai, Kun Li

3D reconstruction of dynamic crowds in large scenes has become increasingly
important for applications such as city surveillance and crowd analysis.
However, current works attempt to reconstruct 3D crowds from a static image,
causing a lack of temporal consistency and inability to alleviate the typical
impact caused by occlusions. In this paper, we propose DyCrowd, the first
framework for spatio-temporally consistent 3D reconstruction of hundreds of
individuals' poses, positions and shapes from a large-scene video. We design a
coarse-to-fine group-guided motion optimization strategy for occlusion-robust
crowd reconstruction in large scenes. To address temporal instability and
severe occlusions, we further incorporate a VAE (Variational Autoencoder)-based
human motion prior along with a segment-level group-guided optimization. The
core of our strategy leverages collective crowd behavior to address long-term
dynamic occlusions. By jointly optimizing the motion sequences of individuals
with similar motion segments and combining this with the proposed Asynchronous
Motion Consistency (AMC) loss, we enable high-quality unoccluded motion
segments to guide the motion recovery of occluded ones, ensuring robust and
plausible motion recovery even in the presence of temporal desynchronization
and rhythmic inconsistencies. Additionally, in order to fill the gap of no
existing well-annotated large-scene video dataset, we contribute a virtual
benchmark dataset, VirtualCrowd, for evaluating dynamic crowd reconstruction
from large-scene videos. Experimental results demonstrate that the proposed
method achieves state-of-the-art performance in the large-scene dynamic crowd
reconstruction task. The code and dataset will be available for research
purposes.

### Computers and Society

### 1. [Leveraging Large Language Models for Predictive Analysis of Human Misery](http://arxiv.org/pdf/2508.12669v1)

Authors: Bishanka Seal, Rahul Seetharaman, Aman Bansal, Abhilash Nandy

This study investigates the use of Large Language Models (LLMs) for
predicting human-perceived misery scores from natural language descriptions of
real-world scenarios. The task is framed as a regression problem, where the
model assigns a scalar value from 0 to 100 to each input statement. We evaluate
multiple prompting strategies, including zero-shot, fixed-context few-shot, and
retrieval-based prompting using BERT sentence embeddings. Few-shot approaches
consistently outperform zero-shot baselines, underscoring the value of
contextual examples in affective prediction. To move beyond static evaluation,
we introduce the "Misery Game Show", a novel gamified framework inspired by a
television format. It tests LLMs through structured rounds involving ordinal
comparison, binary classification, scalar estimation, and feedback-driven
reasoning. This setup enables us to assess not only predictive accuracy but
also the model's ability to adapt based on corrective feedback. The gamified
evaluation highlights the broader potential of LLMs in dynamic emotional
reasoning tasks beyond standard regression. Code and data link:
https://github.com/abhi1nandy2/Misery_Data_Exps_GitHub

### 2. [Evaluating the Quality of Open Building Datasets for Mapping Urban Inequality: A Comparative Analysis Across 5 Cities](http://arxiv.org/pdf/2508.12872v1)

Authors: Franz Okyere, Meng Lu, Ansgar Brunn

While informal settlements lack focused development and are highly dynamic,
the quality of spatial data for these places may be uncertain. This study
evaluates the quality and biases of AI-generated Open Building Datasets (OBDs)
generated by Google and Microsoft against OpenStreetMap (OSM) data, across
diverse global cities including Accra, Nairobi, Caracas, Berlin, and Houston.
The Intersection over Union (IoU), overlap analysis and a positional accuracy
algorithm are used to analyse the similarity and alignment of the datasets. The
paper also analyses the size distribution of the building polygon area, and
completeness using predefined but regular spatial units. The results indicate
significant variance in data quality, with Houston and Berlin demonstrating
high alignment and completeness, reflecting their structured urban
environments. There are gaps in the datasets analysed, and cities like Accra
and Caracas may be under-represented. This could highlight difficulties in
capturing complex or informal regions. The study also notes different building
size distributions, which may be indicative of the global socio-economic
divide. These findings may emphasise the need to consider the quality of global
building datasets to avoid misrepresentation, which is an important element of
planning and resource distribution.

### 3. [Cognitive Structure Generation: From Educational Priors to Policy Optimization](http://arxiv.org/pdf/2508.12647v1)

Authors: Hengnian Gu, Zhifu Chen, Yuxin Chen, Jin Peng Zhou, Dongdai Zhou

Cognitive structure is a student's subjective organization of an objective
knowledge system, reflected in the psychological construction of concepts and
their relations. However, cognitive structure assessment remains a
long-standing challenge in student modeling and psychometrics, persisting as a
foundational yet largely unassessable concept in educational practice. This
paper introduces a novel framework, Cognitive Structure Generation (CSG), in
which we first pretrain a Cognitive Structure Diffusion Probabilistic Model
(CSDPM) to generate students' cognitive structures from educational priors, and
then further optimize its generative process as a policy with hierarchical
reward signals via reinforcement learning to align with genuine cognitive
development levels during students' learning processes. Experimental results on
four popular real-world education datasets show that cognitive structures
generated by CSG offer more comprehensive and effective representations for
student modeling, substantially improving performance on KT and CD tasks while
enhancing interpretability.

### 4. [OPTIC-ER: A Reinforcement Learning Framework for Real-Time Emergency Response and Equitable Resource Allocation in Underserved African Communities](http://arxiv.org/pdf/2508.12943v1)

Authors: Mary Tonwe

Public service systems in many African regions suffer from delayed emergency
response and spatial inequity, causing avoidable suffering. This paper
introduces OPTIC-ER, a reinforcement learning (RL) framework for real-time,
adaptive, and equitable emergency response. OPTIC-ER uses an attention-guided
actor-critic architecture to manage the complexity of dispatch environments.
Its key innovations are a Context-Rich State Vector, encoding action
sub-optimality, and a Precision Reward Function, which penalizes inefficiency.
Training occurs in a high-fidelity simulation using real data from Rivers
State, Nigeria, accelerated by a precomputed Travel Time Atlas. The system is
built on the TALS framework (Thin computing, Adaptability, Low-cost,
Scalability) for deployment in low-resource settings. In evaluations on 500
unseen incidents, OPTIC-ER achieved a 100.00% optimality rate with negligible
inefficiency, confirming its robustness and generalization. Beyond dispatch,
the system generates Infrastructure Deficiency Maps and Equity Monitoring
Dashboards to guide proactive governance and data-informed development. This
work presents a validated blueprint for AI-augmented public services, showing
how context-aware RL can bridge the gap between algorithmic decision-making and
measurable human impact.

### 5. [Vitamin N: Benefits of Different Forms of Public Greenery for Urban Health](http://arxiv.org/pdf/2508.12998v1)

Authors: Sanja Šćepanović, Sagar Joglekar, Stephen Law, Daniele Quercia, Ke Zhou, Alice Battiston, Rossano Schifanella

Urban greenery is often linked to better health, yet findings from past
research have been inconsistent. One reason is that official greenery metrics
measure the amount or nearness of greenery but ignore how often people actually
may potentially see or use it in daily life. To address this gap, we introduced
a new classification that separates on-road greenery, which people see while
walking through streets, from off-road greenery, which requires planned visits.
We did so by combining aerial imagery of Greater London and greenery data from
OpenStreetMap with quantified greenery from over 100,000 Google Street View
images and accessibility estimates based on 160,000 road segments. We linked
these measures to 7.45 billion medical prescriptions issued by the National
Health Service and processed through our methodology. These prescriptions cover
five conditions: diabetes, hypertension, asthma, depression, and anxiety, as
well as opioid use. As hypothesized, we found that green on-road was more
strongly linked to better health than four widely used official measures. For
example, hypertension prescriptions dropped by 3.68% in wards with on-road
greenery above the median citywide level compared to those below it. If all
below-median wards reached the citywide median in on-road greenery,
prescription costs could fall by up to {\pounds}3.15 million each year. These
results suggest that greenery seen in daily life may be more relevant than
public yet secluded greenery, and that official metrics commonly used in the
literature have important limitations.

### 6. [Cyber Risks to Next-Gen Brain-Computer Interfaces: Analysis and Recommendations](http://arxiv.org/pdf/2508.12571v1)

Authors: Tyler Schroder, Renee Sirbu, Sohee Park, Jessica Morley, Sam Street, Luciano Floridi

Brain-computer interfaces (BCIs) show enormous potential for advancing
personalized medicine. However, BCIs also introduce new avenues for
cyber-attacks or security compromises. In this article, we analyze the problem
and make recommendations for device manufacturers to better secure devices and
to help regulators understand where more guidance is needed to protect patient
safety and data confidentiality. Device manufacturers should implement the
prior suggestions in their BCI products. These recommendations help protect BCI
users from undue risks, including compromised personal health and genetic
information, unintended BCI-mediated movement, and many other cybersecurity
breaches. Regulators should mandate non-surgical device update methods, strong
authentication and authorization schemes for BCI software modifications,
encryption of data moving to and from the brain, and minimize network
connectivity where possible. We also design a hypothetical, average-case threat
model that identifies possible cybersecurity threats to BCI patients and
predicts the likeliness of risk for each category of threat. BCIs are at less
risk of physical compromise or attack, but are vulnerable to remote attack; we
focus on possible threats via network paths to BCIs and suggest technical
controls to limit network connections.

### Databases

### 1. [An LLM Agent-Based Complex Semantic Table Annotation Approach](http://arxiv.org/pdf/2508.12868v1)

Authors: Yilin Geng, Shujing Wang, Chuan Wang, Keqing He, Yanfei Lv, Ying Wang, Zaiwen Feng, Xiaoying Bai

The Semantic Table Annotation (STA) task, which includes Column Type
Annotation (CTA) and Cell Entity Annotation (CEA), maps table contents to
ontology entities and plays important roles in various semantic applications.
However, complex tables often pose challenges such as semantic loss of column
names or cell values, strict ontological hierarchy requirements, homonyms,
spelling errors, and abbreviations, which hinder annotation accuracy. To
address these issues, this paper proposes an LLM-based agent approach for CTA
and CEA. We design and implement five external tools with tailored prompts
based on the ReAct framework, enabling the STA agent to dynamically select
suitable annotation strategies depending on table characteristics. Experiments
are conducted on the Tough Tables and BiodivTab datasets from the SemTab
challenge, which contain the aforementioned challenges. Our method outperforms
existing approaches across various metrics. Furthermore, by leveraging
Levenshtein distance to reduce redundant annotations, we achieve a 70%
reduction in time costs and a 60% reduction in LLM token usage, providing an
efficient and cost-effective solution for STA.

### 2. [Evaluating the Quality of Open Building Datasets for Mapping Urban Inequality: A Comparative Analysis Across 5 Cities](http://arxiv.org/pdf/2508.12872v1)

Authors: Franz Okyere, Meng Lu, Ansgar Brunn

While informal settlements lack focused development and are highly dynamic,
the quality of spatial data for these places may be uncertain. This study
evaluates the quality and biases of AI-generated Open Building Datasets (OBDs)
generated by Google and Microsoft against OpenStreetMap (OSM) data, across
diverse global cities including Accra, Nairobi, Caracas, Berlin, and Houston.
The Intersection over Union (IoU), overlap analysis and a positional accuracy
algorithm are used to analyse the similarity and alignment of the datasets. The
paper also analyses the size distribution of the building polygon area, and
completeness using predefined but regular spatial units. The results indicate
significant variance in data quality, with Houston and Berlin demonstrating
high alignment and completeness, reflecting their structured urban
environments. There are gaps in the datasets analysed, and cities like Accra
and Caracas may be under-represented. This could highlight difficulties in
capturing complex or informal regions. The study also notes different building
size distributions, which may be indicative of the global socio-economic
divide. These findings may emphasise the need to consider the quality of global
building datasets to avoid misrepresentation, which is an important element of
planning and resource distribution.

### 3. [SPARQL in N3: SPARQL CONSTRUCT as a rule language for the Semantic Web (Extended Version)](http://arxiv.org/pdf/2508.13041v1)

Authors: Dörthe Arndt, William Van Woensel, Dominik Tomaszuk

Reasoning in the Semantic Web (SW) commonly uses Description Logics (DL) via
OWL2 DL ontologies, or SWRL for variables and Horn clauses. The Rule
Interchange Format (RIF) offers more expressive rules but is defined outside
RDF and rarely adopted. For querying, SPARQL is a well-established standard
operating directly on RDF triples. We leverage SPARQL CONSTRUCT queries as
logic rules, enabling (1) an expressive, familiar SW rule language, and (2)
general recursion, where queries can act on the results of others. We translate
these queries to the Notation3 Logic (N3) rule language, allowing use of
existing reasoning machinery with forward and backward chaining. Targeting a
one-to-one query-rule mapping improves exchangeability and interpretability.
Benchmarks indicate competitive performance, aiming to advance the potential of
rule-based reasoning in the SW.

### 4. [jXBW: Fast Substructure Search in Large-Scale JSONL Datasets for Foundation Model Applications](http://arxiv.org/pdf/2508.12536v1)

Authors: Yasuo Tabei

Substructure search in JSON Lines (JSONL) datasets is essential for modern
applications such as prompt engineering in foundation models, but existing
methods suffer from prohibitive computational costs due to exhaustive tree
traversal and subtree matching. We present jXBW, a fast method for substructure
search on large-scale JSONL datasets. Our method makes three key technical
contributions: (i) a merged tree representation built by merging trees of
multiple JSON objects while preserving individual identities, (ii) a succinct
data structure based on the eXtended Burrows-Wheeler Transform that enables
efficient tree navigation and subpath search, and (iii) an efficient three-step
substructure search algorithm that combines path decomposition, ancestor
computation, and adaptive tree identifier collection to ensure correctness
while avoiding exhaustive tree traversal. Experimental evaluation on real-world
datasets demonstrates that jXBW consistently outperforms existing methods,
achieving speedups of 16$\times$ for smaller datasets and up to 4,700$\times$
for larger datasets over tree-based approaches, and more than 6$\times$10$^6$
over XML-based processing while maintaining competitive memory usage.

### Distributed, Parallel, and Cluster Computing

### 1. [Accelerating Edge Inference for Distributed MoE Models with Latency-Optimized Expert Placement](http://arxiv.org/pdf/2508.12851v1)

Authors: Tian Wu, Liming Wang, Zijian Wen, Xiaoxi Zhang, Jingpu Duan, Xianwei Zhang, Jinhang Zuo

Mixture-of-Experts (MoE) have become a cornerstone for training and scaling
large language models (LLMs), offering substantial gains in model capacity and
efficiency through sparse expert activation. However, serving these models
remains challenging in practice, particularly in resource-constrained edge
environments, due to their large memory footprint and complex communication
demands. While centralized cloud inference is common, it incurs high
infrastructure costs, along with latency and privacy concerns. A few recent
edge MoE works propose memory-efficient strategies but typically focus on
single-device or homogeneous setups. This paper presents DanceMoE, an efficient
MoE inference framework that enables activation-aware expert placement across
collaborative, heterogeneous, GPU-equipped edge servers. DanceMoE leverages the
inherent sparsity of MoE models and workload locality to minimize cross-server
communication and enable efficient expert placement under heterogeneous
resource constraints. It introduces a data-driven, activation-aware placement
algorithm that balances local coverage and memory usage across servers,
alongside a lightweight migration mechanism that adapts expert assignments
under evolving workloads. We evaluate DanceMoE on modern MoE models and widely
used datasets, demonstrating up to 30.6\% lower inference latency, and
substantial communication reduction compared to state-of-the-art baselines,
showcasing the effectiveness of collaborative edge-based MoE inference.

### 2. [WANify: Gauging and Balancing Runtime WAN Bandwidth for Geo-distributed Data Analytics](http://arxiv.org/pdf/2508.12961v1)

Authors: Anshuman Das Mohapatra, Kwangsung Oh

Accurate wide area network (WAN) bandwidth (BW) is essential for
geo-distributed data analytics (GDA) systems to make optimal decisions such as
data and task placement to improve performance. Existing GDA systems, however,
measure WAN BW statically and independently between data centers (DCs), while
data transfer occurs dynamically and simultaneously among DCs during workload
execution. Also, they use a single connection WAN BW that cannot capture actual
WAN capacities between distant DCs. Such inaccurate WAN BWs yield sub-optimal
decisions, inflating overall query latency and cost. In this paper, we present
WANify, a new framework that precisely and dynamically gauges achievable
runtime WAN BW using a machine learning prediction scheme, decision tree-based
Random Forest. This helps GDA systems make better decisions yielding reduced
latency and costs including WAN BW monitoring costs. Based on predicted runtime
WAN BW, WANify determines the optimal number of heterogeneous parallel
connections for data transfer among DCs. This approach improves performance
without additional, or even at reduced cost, by fully exploiting available WAN
capacities. In addition, WANify considers dynamics like network and workloads,
and heterogeneity like skewed data, heterogeneous compute resources, and a
varying number of DCs while making decisions. The WANify prototype running on
state-of-the-art GDA systems is evaluated on AWS with 8 geo-distributed DCs.
Results show that WANify enhances WAN throughput by balancing between the
strongest and weakest WAN links, enabling GDA systems to reduce latency and
cost by up to 26% and 16% respectively with minimal effort, all while handling
dynamics and heterogeneity efficiently.

### 3. [Team Formation and Applications](http://arxiv.org/pdf/2508.13084v1)

Authors: Yuval Emek, Shay Kutten, Ido Rafael, Gadi Taubenfeld

A novel long-lived distributed problem, called Team Formation (TF), is
introduced together with a message- and time-efficient randomized algorithm.
The problem is defined over the asynchronous model with a complete
communication graph, using bounded size messages, where a certain fraction of
the nodes may experience a generalized, strictly stronger, version of initial
failures. The goal of a TF algorithm is to assemble tokens injected by the
environment, in a distributed manner, into teams of size $\sigma$, where
$\sigma$ is a parameter of the problem.
  The usefulness of TF is demonstrated by using it to derive efficient
algorithms for many distributed problems. Specifically, we show that various
(one-shot as well as long-lived) distributed problems reduce to TF. This
includes well-known (and extensively studied) distributed problems such as
several versions of leader election and threshold detection. For example, we
are the first to break the linear message complexity bound for asynchronous
implicit leader election. We also improve the time complexity of
message-optimal algorithms for asynchronous explicit leader election. Other
distributed problems that reduce to TF are new ones, including matching players
in online gaming platforms, a generalization of gathering, constructing a
perfect matching in an induced subgraph of the complete graph, quorum sensing
in message-passing networks, and more. To complement our positive contribution,
we establish a tight lower bound on the message complexity of TF algorithms.

### 4. [DIT: Dimension Reduction View on Optimal NFT Rarity Meters](http://arxiv.org/pdf/2508.12671v1)

Authors: Dmitry Belousov, Yury Yanovich

Non-fungible tokens (NFTs) have become a significant digital asset class,
each uniquely representing virtual entities such as artworks. These tokens are
stored in collections within smart contracts and are actively traded across
platforms on Ethereum, Bitcoin, and Solana blockchains. The value of NFTs is
closely tied to their distinctive characteristics that define rarity, leading
to a growing interest in quantifying rarity within both industry and academia.
While there are existing rarity meters for assessing NFT rarity, comparing them
can be challenging without direct access to the underlying collection data. The
Rating over all Rarities (ROAR) benchmark addresses this challenge by providing
a standardized framework for evaluating NFT rarity. This paper explores a
dimension reduction approach to rarity design, introducing new performance
measures and meters, and evaluates them using the ROAR benchmark. Our
contributions to the rarity meter design issue include developing an optimal
rarity meter design using non-metric weighted multidimensional scaling,
introducing Dissimilarity in Trades (DIT) as a performance measure inspired by
dimension reduction techniques, and unveiling the non-interpretable rarity
meter DIT, which demonstrates superior performance compared to existing
methods.

### 5. [Dissecting CPU-GPU Unified Physical Memory on AMD MI300A APUs](http://arxiv.org/pdf/2508.12743v1)

Authors: Jacob Wahlgren, Gabin Schieffer, Ruimin Shi, Edgar A. León, Roger Pearce, Maya Gokhale, Ivy Peng

Discrete GPUs are a cornerstone of HPC and data center systems, requiring
management of separate CPU and GPU memory spaces. Unified Virtual Memory (UVM)
has been proposed to ease the burden of memory management; however, at a high
cost in performance. The recent introduction of AMD's MI300A Accelerated
Processing Units (APUs)--as deployed in the El Capitan supercomputer--enables
HPC systems featuring integrated CPU and GPU with Unified Physical Memory (UPM)
for the first time. This work presents the first comprehensive characterization
of the UPM architecture on MI300A. We first analyze the UPM system properties,
including memory latency, bandwidth, and coherence overhead. We then assess the
efficiency of the system software in memory allocation, page fault handling,
TLB management, and Infinity Cache utilization. We propose a set of porting
strategies for transforming applications for the UPM architecture and evaluate
six applications on the MI300A APU. Our results show that applications on UPM
using the unified memory model can match or outperform those in the explicitly
managed model--while reducing memory costs by up to 44%.

### 6. [Congested Clique Counting for Local Gibbs Distributions](http://arxiv.org/pdf/2508.13083v1)

Authors: Joshua Z. Sobel

There are well established reductions between combinatorial sampling and
counting problems (Jerrum, Valiant, Vazirani TCS 1986). Building off of a very
recent parallel algorithm utilizing this connection (Liu, Yin, Zhang arxiv
2024), we demonstrate the first approximate counting algorithm in the
CongestedClique for a wide range of problems. Most interestingly, we present an
algorithm for approximating the number of $q$-colorings of a graph within
$\epsilon$-multiplicative error, when $q>\alpha\Delta$ for any constant
$\alpha>2$, in $\Tilde{O}\big(\frac{n^{1/3}}{\epsilon^2}\big)$ rounds. More
generally, we achieve a runtime of
$\Tilde{O}\big(\frac{n^{1/3}}{\epsilon^2}\big)$ rounds for approximating the
partition function of Gibbs distributions defined over graphs when simple
locality and fast mixing conditions hold. Gibbs distributions are widely used
in fields such as machine learning and statistical physics. We obtain our
result by providing an algorithm to draw $n$ random samples from a distributed
Markov chain in parallel, using similar ideas to triangle counting (Dolev,
Lenzen, Peled DISC 2012) and semiring matrix multiplication (Censor-Hillel,
Kaski, Korhonen, Lenzen, Paz, Suomela PODC 2015). Aside from counting problems,
this result may be interesting for other applications requiring a large number
of samples. In the special case of estimating the partition function of the
hardcore model, also known as counting weighted independent sets, we can do
even better and achieve an $\Tilde{O}\big(\frac{1}{\epsilon^2}\big)$ round
algorithm, when the fugacity $\lambda \leq \frac{\alpha}{\Delta-1}$, where
$\alpha$ is an arbitrary constant less than $1$.

### 7. [Data-driven Trust Bootstrapping for Mobile Edge Computing-based Industrial IoT Services](http://arxiv.org/pdf/2508.12560v1)

Authors: Prabath Abeysekara, Hai Dong

We propose a data-driven and context-aware approach to bootstrap
trustworthiness of homogeneous Internet of Things (IoT) services in Mobile Edge
Computing (MEC) based industrial IoT (IIoT) systems. The proposed approach
addresses key limitations in adapting existing trust bootstrapping approaches
into MEC-based IIoT systems. These key limitations include, the lack of
opportunity for a service consumer to interact with a lesser-known service over
a prolonged period of time to get a robust measure of its trustworthiness,
inability of service consumers to consistently interact with their peers to
receive reliable recommendations of the trustworthiness of a lesser-known
service as well as the impact of uneven context parameters in different MEC
environments causing uneven trust environments for trust evaluation. In
addition, the proposed approach also tackles the problem of data sparsity via
enabling knowledge sharing among different MEC environments within a given MEC
topology. To verify the effectiveness of the proposed approach, we carried out
a comprehensive evaluation on two real-world datasets suitably adjusted to
exhibit the context-dependent trust information accumulated in MEC environments
within a given MEC topology. The experimental results affirmed the
effectiveness of our approach and its suitability to bootstrap trustworthiness
of services in MEC-based IIoT systems.

### 8. [Fed-DPRoC:Communication-Efficient Differentially Private and Robust Federated Learning](http://arxiv.org/pdf/2508.12978v1)

Authors: Yue Xia, Tayyebeh Jahani-Nezhad, Rawad Bitar

We propose Fed-DPRoC, a novel federated learning framework that
simultaneously ensures differential privacy (DP), Byzantine robustness, and
communication efficiency. We introduce the concept of robust-compatible
compression, which enables users to compress DP-protected updates while
maintaining the robustness of the aggregation rule. We instantiate our
framework as RobAJoL, combining the Johnson-Lindenstrauss (JL) transform for
compression with robust averaging for robust aggregation. We theoretically
prove the compatibility of JL transform with robust averaging and show that
RobAJoL preserves robustness guarantees, ensures DP, and reduces communication
cost. Experiments on CIFAR-10 and Fashion MNIST validate our theoretical claims
and demonstrate that RobAJoL outperforms existing methods in terms of
robustness and utility under different Byzantine attacks.

### Digital Libraries

### 1. [Citation accuracy, citation noise, and citation bias: A foundation of citation analysis](http://arxiv.org/pdf/2508.12735v1)

Authors: Lutz Bornmann, Christian Leibel

Citation analysis is widely used in research evaluation to assess the impact
of scientific papers. These analyses rest on the assumption that citation
decisions by authors are accurate, representing flow of knowledge from cited to
citing papers. However, in practice, researchers often cite for reasons other
than attributing intellectual credit to previous research. Citations made for
rhetorical reasons or without reading the cited work compromise the value of
citations as instrument for research evaluation. Past research on threats to
the accuracy of citations has mainly focused on citation bias as the primary
concern. In this paper, we argue that citation noise - the undesirable variance
in citation decisions - represents an equally critical but underexplored
challenge in citation analysis. We define and differentiate two types of
citation noise: citation level noise and citation pattern noise. Each type of
noise is described in terms of how it arises and the specific ways it can
undermine the validity of citation-based research assessments. By conceptually
differing citation noise from citation accuracy and citation bias, we propose a
framework for the foundation of citation analysis. We discuss strategies and
interventions to minimize citation noise, aiming to improve the reliability and
validity of citation analysis in research evaluation. We recommend that the
current professional reform movement in research evaluation such as the
Coalition for Advancing Research Assessment (CoARA) pick up these strategies
and interventions as an additional building block for careful, responsible use
of bibliometric indicators in research evaluation.

### Discrete Mathematics

### 1. [On the complexity of constrained reconfiguration and motion planning](http://arxiv.org/pdf/2508.13032v1)

Authors: Nicolas Bousquet, Remy El Sabeh, Amer E. Mouawad, Naomi Nishimura

Coordinating the motion of multiple agents in constrained environments is a
fundamental challenge in robotics, motion planning, and scheduling. A
motivating example involves $n$ robotic arms, each represented as a line
segment. The objective is to rotate each arm to its vertical orientation, one
at a time (clockwise or counterclockwise), without collisions nor rotating any
arm more than once. This scenario is an example of the more general
$k$-Compatible Ordering problem, where $n$ agents, each capable of $k$
state-changing actions, must transition to specific target states under
constraints encoded as a set $\mathcal{G}$ of $k$ pairs of directed graphs.
  We show that $k$-Compatible Ordering is $\mathsf{NP}$-complete, even when
$\mathcal{G}$ is planar, degenerate, or acyclic. On the positive side, we
provide polynomial-time algorithms for cases such as when $k = 1$ or
$\mathcal{G}$ has bounded treewidth. We also introduce generalized variants
supporting multiple state-changing actions per agent, broadening the
applicability of our framework. These results extend to a wide range of
scheduling, reconfiguration, and motion planning applications in constrained
environments.

### Data Structures and Algorithms

### 1. [r*-indexing](http://arxiv.org/pdf/2508.12675v1)

Authors: Travis Gagie

Let $T [1..n]$ be a text over an alphabet of size $\sigma \in
\mathrm{polylog} (n)$, let $r^*$ be the sum of the numbers of runs in the
Burrows-Wheeler Transforms of $T$ and its reverse, and let $z$ be the number of
phrases in the LZ77 parse of $T$. We show how to store $T$ in $O (r^* \log (n /
r^*) + z \log n)$ bits such that, given a pattern $P [1..m]$, we can report the
locations of the $\mathrm{occ}$ occurrences of $P$ in $T$ in $O (m \log n +
\mathrm{occ} \log^\epsilon n)$ time. We can also report the position of the
leftmost and rightmost occurrences of $P$ in $T$ in the same space and $O (m
\log^\epsilon n)$ time.

### 2. [Weighted Partition Vertex and Edge Cover](http://arxiv.org/pdf/2508.13055v1)

Authors: Rajni Dabas, Samir Khuller, Emilie Rivkin

We study generalizations of the classical Vertex Cover and Edge Cover
problems that incorporate group-wise coverage constraints. Our first focus is
the \emph{Weighted Prize-Collecting Partition Vertex Cover} (WP-PVC) problem:
given a graph with weights on both vertices and edges, and a partition of the
edge set into $\omega$ groups, the goal is to select a minimum-weight subset of
vertices such that, in each group, the total weight (profit) of covered edges
meets a specified threshold. This formulation generalizes classical vertex
cover, partial vertex cover and partition vertex cover.
  We present two algorithms for WP-PVC. The first is a simple 2-approximation
that solves \( n^{\omega} \) LP's, improving over prior work by Bandyapadhyay
et al.\ by removing an enumerative step and the extra \( \epsilon \)-factor in
approximation, while also extending to the weighted setting. The second is a
bi-criteria algorithm that applies when \( \omega \) is large, approximately
meeting profit targets with a bounded LP-relative cost.
  We also study a natural generalization of the edge cover problem, the
\emph{Weighted Partition Edge Cover} (W-PEC) problem, where each edge has an
associated weights, and the vertex set is partitioned into groups. For each
group, the goal is to cover at least a specified number of vertices using
incident edges, while minimizing the total weight of the selected edges. We
present the first exact polynomial-time algorithm for the weighted case,
improving runtime from \( O(\omega n^3) \) to \( O(mn+n^2 \log n) \) and
simplifying the algorithmic structure over prior unweighted approaches. We also
show that the prize-collecting variant of the W-PEC problem is NP-Complete via
a reduction from the knapsack problem.

### 3. [Congested Clique Counting for Local Gibbs Distributions](http://arxiv.org/pdf/2508.13083v1)

Authors: Joshua Z. Sobel

There are well established reductions between combinatorial sampling and
counting problems (Jerrum, Valiant, Vazirani TCS 1986). Building off of a very
recent parallel algorithm utilizing this connection (Liu, Yin, Zhang arxiv
2024), we demonstrate the first approximate counting algorithm in the
CongestedClique for a wide range of problems. Most interestingly, we present an
algorithm for approximating the number of $q$-colorings of a graph within
$\epsilon$-multiplicative error, when $q>\alpha\Delta$ for any constant
$\alpha>2$, in $\Tilde{O}\big(\frac{n^{1/3}}{\epsilon^2}\big)$ rounds. More
generally, we achieve a runtime of
$\Tilde{O}\big(\frac{n^{1/3}}{\epsilon^2}\big)$ rounds for approximating the
partition function of Gibbs distributions defined over graphs when simple
locality and fast mixing conditions hold. Gibbs distributions are widely used
in fields such as machine learning and statistical physics. We obtain our
result by providing an algorithm to draw $n$ random samples from a distributed
Markov chain in parallel, using similar ideas to triangle counting (Dolev,
Lenzen, Peled DISC 2012) and semiring matrix multiplication (Censor-Hillel,
Kaski, Korhonen, Lenzen, Paz, Suomela PODC 2015). Aside from counting problems,
this result may be interesting for other applications requiring a large number
of samples. In the special case of estimating the partition function of the
hardcore model, also known as counting weighted independent sets, we can do
even better and achieve an $\Tilde{O}\big(\frac{1}{\epsilon^2}\big)$ round
algorithm, when the fugacity $\lambda \leq \frac{\alpha}{\Delta-1}$, where
$\alpha$ is an arbitrary constant less than $1$.

### 4. [A simple analysis of a quantum-inspired algorithm for solving low-rank linear systems](http://arxiv.org/pdf/2508.13108v1)

Authors: Tyler Chen, Junhyung Lyle Kim, Archan Ray, Shouvanik Chakrabarti, Dylan Herman, Niraj Kumar

We describe and analyze a simple algorithm for sampling from the solution
$\mathbf{x}^* := \mathbf{A}^+\mathbf{b}$ to a linear system
$\mathbf{A}\mathbf{x} = \mathbf{b}$. We assume access to a sampler which allows
us to draw indices proportional to the squared row/column-norms of
$\mathbf{A}$. Our algorithm produces a compressed representation of some vector
$\mathbf{x}$ for which $\|\mathbf{x}^* - \mathbf{x}\| < \varepsilon
\|\mathbf{x}^* \|$ in $\widetilde{O}(\kappa_{\mathsf{F}}^4 \kappa^2 /
\varepsilon^2)$ time, where $\kappa_{\mathsf{F}} :=
\|\mathbf{A}\|_{\mathsf{F}}\|\mathbf{A}^{+}\|$ and $\kappa :=
\|\mathbf{A}\|\|\mathbf{A}^{+}\|$. The representation of $\mathbf{x}$ allows us
to query entries of $\mathbf{x}$ in $\widetilde{O}(\kappa_{\mathsf{F}}^2)$ time
and sample proportional to the square entries of $\mathbf{x}$ in
$\widetilde{O}(\kappa_{\mathsf{F}}^4 \kappa^6)$ time, assuming access to a
sampler which allows us to draw indices proportional to the squared entries of
any given row of $\mathbf{A}$. Our analysis, which is elementary,
non-asymptotic, and fully self-contained, simplifies and clarifies several past
analyses from literature including [Gily\'en, Song, and Tang; 2022, 2023] and
[Shao and Montanaro; 2022].

### 5. [jXBW: Fast Substructure Search in Large-Scale JSONL Datasets for Foundation Model Applications](http://arxiv.org/pdf/2508.12536v1)

Authors: Yasuo Tabei

Substructure search in JSON Lines (JSONL) datasets is essential for modern
applications such as prompt engineering in foundation models, but existing
methods suffer from prohibitive computational costs due to exhaustive tree
traversal and subtree matching. We present jXBW, a fast method for substructure
search on large-scale JSONL datasets. Our method makes three key technical
contributions: (i) a merged tree representation built by merging trees of
multiple JSON objects while preserving individual identities, (ii) a succinct
data structure based on the eXtended Burrows-Wheeler Transform that enables
efficient tree navigation and subpath search, and (iii) an efficient three-step
substructure search algorithm that combines path decomposition, ancestor
computation, and adaptive tree identifier collection to ensure correctness
while avoiding exhaustive tree traversal. Experimental evaluation on real-world
datasets demonstrates that jXBW consistently outperforms existing methods,
achieving speedups of 16$\times$ for smaller datasets and up to 4,700$\times$
for larger datasets over tree-based approaches, and more than 6$\times$10$^6$
over XML-based processing while maintaining competitive memory usage.

### 6. [Algorithmic Improvements to List Decoding of Folded Reed-Solomon Codes](http://arxiv.org/pdf/2508.12548v1)

Authors: Vikrant Ashvinkumar, Mursalin Habib, Shashank Srivastava

Folded Reed-Solomon (FRS) codes are a well-studied family of codes, known for
achieving list decoding capacity. In this work, we give improved deterministic
and randomized algorithms for list decoding FRS codes of rate $R$ up to radius
$1-R-\varepsilon$.
  We present a deterministic decoder that runs in near-linear time
$\widetilde{O}_{\varepsilon}(n)$, improving upon the best-known runtime
$n^{\Omega(1/\varepsilon)}$ for decoding FRS codes. Prior to our work, no
capacity achieving code was known whose deterministic decoding could be done in
time $\widetilde{O}_{\varepsilon}(n)$.
  We also present a randomized decoder that runs in fully polynomial time
$\mathrm{poly}(1/\varepsilon) \cdot \widetilde{O}(n)$, improving the best-known
runtime $\mathrm{exp}(1/\varepsilon)\cdot \widetilde{O}(n)$ for decoding FRS
codes. Again, prior to our work, no capacity achieving code was known whose
decoding time depended polynomially on $1/\varepsilon$.

### 7. [Group Fair Matchings using Convex Cost Functions](http://arxiv.org/pdf/2508.12549v1)

Authors: Atasi Panda, Harsh Sharma, Anand Louis, Prajakta Nimbhorkar

We consider the problem of assigning items to platforms where each item has a
utility associated with each of the platforms to which it can be assigned. Each
platform has a soft constraint over the total number of items it serves,
modeled via a convex cost function. Additionally, items are partitioned into
groups, and each platform also incurs group-specific convex cost over the
number of items from each group that can be assigned to the platform. These
costs promote group fairness by penalizing imbalances, yielding a soft
variation of fairness notions introduced in prior work, such as Restricted
Dominance and Minority protection. Restricted Dominance enforces upper bounds
on group representation, while Minority protection enforces lower bounds. Our
approach replaces such hard constraints with cost-based penalties, allowing
more flexible trade-offs. Our model also captures Nash Social Welfare kind of
objective.
  The cost of an assignment is the sum of the values of all the cost functions
across all the groups and platforms. The objective is to find an assignment
that minimizes the cost while achieving a total utility that is at least a
user-specified threshold. The main challenge lies in balancing the overall
platform cost with group-specific costs, both governed by convex functions,
while meeting the utility constraint. We present an efficient polynomial-time
approximation algorithm, supported by theoretical guarantees and experimental
evaluation. Our algorithm is based on techniques involving linear programming
and network flows. We also provide an exact algorithm for a special case with
uniform utilities and establish the hardness of the general problem when the
groups can intersect arbitrarily.

### 8. [A Perfectly Truthful Calibration Measure](http://arxiv.org/pdf/2508.13100v1)

Authors: Jason Hartline, Lunjia Hu, Yifan Wu

Calibration requires that predictions are conditionally unbiased and,
therefore, reliably interpretable as probabilities. Calibration measures
quantify how far a predictor is from perfect calibration. As introduced by
Haghtalab et al. (2024), a calibration measure is truthful if it is minimized
in expectation when a predictor outputs the ground-truth probabilities.
Although predicting the true probabilities guarantees perfect calibration, in
reality, when calibration is evaluated on a finite sample, predicting the truth
is not guaranteed to minimize any known calibration measure. All known
calibration measures incentivize predictors to lie in order to appear more
calibrated on a finite sample. Such lack of truthfulness motivated Haghtalab et
al. (2024) and Qiao and Zhao (2025) to construct approximately truthful
calibration measures in the sequential prediction setting, but no perfectly
truthful calibration measure was known to exist even in the more basic batch
setting.
  We design a perfectly truthful calibration measure in the batch setting:
averaged two-bin calibration error (ATB). In addition to being truthful, ATB is
sound, complete, continuous, and quadratically related to two existing
calibration measures: the smooth calibration error (smCal) and the (lower)
distance to calibration (distCal). The simplicity in our definition of ATB
makes it efficient and straightforward to compute. ATB allows faster estimation
algorithms with significantly easier implementations than smCal and distCal,
achieving improved running time and simplicity for the calibration testing
problem studied by Hu et al. (2024). We also introduce a general recipe for
constructing truthful measures, which proves the truthfulness of ATB as a
special case and allows us to construct other truthful calibration measures
such as quantile-binned l_2-ECE.

### 9. [On the complexity of constrained reconfiguration and motion planning](http://arxiv.org/pdf/2508.13032v1)

Authors: Nicolas Bousquet, Remy El Sabeh, Amer E. Mouawad, Naomi Nishimura

Coordinating the motion of multiple agents in constrained environments is a
fundamental challenge in robotics, motion planning, and scheduling. A
motivating example involves $n$ robotic arms, each represented as a line
segment. The objective is to rotate each arm to its vertical orientation, one
at a time (clockwise or counterclockwise), without collisions nor rotating any
arm more than once. This scenario is an example of the more general
$k$-Compatible Ordering problem, where $n$ agents, each capable of $k$
state-changing actions, must transition to specific target states under
constraints encoded as a set $\mathcal{G}$ of $k$ pairs of directed graphs.
  We show that $k$-Compatible Ordering is $\mathsf{NP}$-complete, even when
$\mathcal{G}$ is planar, degenerate, or acyclic. On the positive side, we
provide polynomial-time algorithms for cases such as when $k = 1$ or
$\mathcal{G}$ has bounded treewidth. We also introduce generalized variants
supporting multiple state-changing actions per agent, broadening the
applicability of our framework. These results extend to a wide range of
scheduling, reconfiguration, and motion planning applications in constrained
environments.

### 10. [On computing and the complexity of computing higher-order $U$-statistics, exactly](http://arxiv.org/pdf/2508.12627v1)

Authors: Xingyu Chen, Ruiqi Zhang, Lin Liu

Higher-order $U$-statistics abound in fields such as statistics, machine
learning, and computer science, but are known to be highly time-consuming to
compute in practice. Despite their widespread appearance, a comprehensive study
of their computational complexity is surprisingly lacking. This paper aims to
fill that gap by presenting several results related to the computational aspect
of $U$-statistics. First, we derive a useful decomposition from an $m$-th order
$U$-statistic to a linear combination of $V$-statistics with orders not
exceeding $m$, which are generally more feasible to compute. Second, we explore
the connection between exactly computing $V$-statistics and Einstein summation,
a tool often used in computational mathematics, quantum computing, and quantum
information sciences for accelerating tensor computations. Third, we provide an
optimistic estimate of the time complexity for exactly computing
$U$-statistics, based on the treewidth of a particular graph associated with
the $U$-statistic kernel. The above ingredients lead to a new, much more
runtime-efficient algorithm of exactly computing general higher-order
$U$-statistics. We also wrap our new algorithm into an open-source Python
package called $\texttt{u-stats}$. We demonstrate via three statistical
applications that $\texttt{u-stats}$ achieves impressive runtime performance
compared to existing benchmarks. This paper aspires to achieve two goals: (1)
to capture the interest of researchers in both statistics and other related
areas further to advance the algorithmic development of $U$-statistics, and (2)
to offer the package $\texttt{u-stats}$ as a valuable tool for practitioners,
making the implementation of methods based on higher-order $U$-statistics a
more delightful experience.

### Emerging Technologies

### 1. [The Maximum Coverage Model and Recommendation System for UAV Vertiports Location Planning](http://arxiv.org/pdf/2508.12651v1)

Authors: Chunliang Hua, Xiao Hu, Jiayang Sun, Zeyuan Yang

As urban aerial mobility (UAM) infrastructure development accelerates
globally, cities like Shenzhen are planning large-scale vertiport networks
(e.g., 1,200+ facilities by 2026). Existing planning frameworks remain
inadequate for this complexity due to historical limitations in data
granularity and real-world applicability. This paper addresses these gaps by
first proposing the Capacitated Dynamic Maximum Covering Location Problem
(CDMCLP), a novel optimization framework that simultaneously models urban-scale
spatial-temporal demand, heterogeneous user behaviors, and infrastructure
capacity constraints. Building on this foundation, we introduce an Integrated
Planning Recommendation System that combines CDMCLP with socio-economic factors
and dynamic clustering initialization. This system leverages adaptive parameter
tuning based on empirical user behavior to generate practical planning
solutions. Validation in a Chinese center city demonstrates the effectiveness
of the new optimization framework and recommendation system. Under the
evaluation and optimization of CDMCLP, the quantitative performance of
traditional location methods are exposed and can be improved by 38\%--52\%,
while the recommendation system shows user-friendliness and the effective
integration of complex elements. By integrating mathematical rigor with
practical implementation considerations, this hybrid approach bridges the gap
between theoretical location modeling and real-world UAM infrastructure
planning, offering municipalities a pragmatic tool for vertiport network
design.

### 2. [HOMI: Ultra-Fast EdgeAI platform for Event Cameras](http://arxiv.org/pdf/2508.12637v1)

Authors: Shankaranarayanan H, Satyapreet Singh Yadav, Adithya Krishna, Ajay Vikram P, Mahesh Mehendale, Chetan Singh Thakur

Event cameras offer significant advantages for edge robotics applications due
to their asynchronous operation and sparse, event-driven output, making them
well-suited for tasks requiring fast and efficient closed-loop control, such as
gesture-based human-robot interaction. Despite this potential, existing event
processing solutions remain limited, often lacking complete end-to-end
implementations, exhibiting high latency, and insufficiently exploiting event
data sparsity. In this paper, we present HOMI, an ultra-low latency, end-to-end
edge AI platform comprising a Prophesee IMX636 event sensor chip with an Xilinx
Zynq UltraScale+MPSoC FPGA chip, deploying an in-house developed AI
accelerator. We have developed hardware-optimized pre-processing pipelines
supporting both constant-time and constant-event modes for histogram
accumulation, linear and exponential time surfaces. Our general-purpose
implementation caters to both accuracy-driven and low-latency applications.
HOMI achieves 94% accuracy on the DVS Gesture dataset as a use case when
configured for high accuracy operation and provides a throughput of 1000 fps
for low-latency configuration. The hardware-optimised pipeline maintains a
compact memory footprint and utilises only 33% of the available LUT resources
on the FPGA, leaving ample headroom for further latency reduction, model
parallelisation, multi-task deployments, or integration of more complex
architectures.

### 3. [Cyber Risks to Next-Gen Brain-Computer Interfaces: Analysis and Recommendations](http://arxiv.org/pdf/2508.12571v1)

Authors: Tyler Schroder, Renee Sirbu, Sohee Park, Jessica Morley, Sam Street, Luciano Floridi

Brain-computer interfaces (BCIs) show enormous potential for advancing
personalized medicine. However, BCIs also introduce new avenues for
cyber-attacks or security compromises. In this article, we analyze the problem
and make recommendations for device manufacturers to better secure devices and
to help regulators understand where more guidance is needed to protect patient
safety and data confidentiality. Device manufacturers should implement the
prior suggestions in their BCI products. These recommendations help protect BCI
users from undue risks, including compromised personal health and genetic
information, unintended BCI-mediated movement, and many other cybersecurity
breaches. Regulators should mandate non-surgical device update methods, strong
authentication and authorization schemes for BCI software modifications,
encryption of data moving to and from the brain, and minimize network
connectivity where possible. We also design a hypothetical, average-case threat
model that identifies possible cybersecurity threats to BCI patients and
predicts the likeliness of risk for each category of threat. BCIs are at less
risk of physical compromise or attack, but are vulnerable to remote attack; we
focus on possible threats via network paths to BCIs and suggest technical
controls to limit network connections.

### Formal Languages and Automata Theory

### 1. [Box-Reachability in Vector Addition Systems](http://arxiv.org/pdf/2508.12853v1)

Authors: Shaull Almagor, Itay Hasson, Michał Pilipczuk, Michael Zaslavski

We consider a variant of reachability in Vector Addition Systems (VAS) dubbed
\emph{box reachability}, whereby a vector $v\in \mathbb{N}^d$ is box-reachable
from $0$ in a VAS $V$ if $V$ admits a path from $0$ to $v$ that not only stays
in the positive orthant (as in the standard VAS semantics), but also stays
below $v$, i.e., within the ``box'' whose opposite corners are $0$ and $v$.
  Our main result is that for two-dimensional VAS, the set of box-reachable
vertices almost coincides with the standard reachability set: the two sets
coincide for all vectors whose coordinates are both above some threshold $W$.
We also study properties of box-reachability, exploring the differences and
similarities with standard reachability.
  Technically, our main result is proved using powerful machinery from convex
geometry.

### Graphics

### 1. [MixCache: Mixture-of-Cache for Video Diffusion Transformer Acceleration](http://arxiv.org/pdf/2508.12691v1)

Authors: Yuanxin Wei, Lansong Diao, Bujiao Chen, Shenggan Cheng, Zhengping Qian, Wenyuan Yu, Nong Xiao, Wei Lin, Jiangsu Du

Leveraging the Transformer architecture and the diffusion process, video DiT
models have emerged as a dominant approach for high-quality video generation.
However, their multi-step iterative denoising process incurs high computational
cost and inference latency. Caching, a widely adopted optimization method in
DiT models, leverages the redundancy in the diffusion process to skip
computations in different granularities (e.g., step, cfg, block). Nevertheless,
existing caching methods are limited to single-granularity strategies,
struggling to balance generation quality and inference speed in a flexible
manner. In this work, we propose MixCache, a training-free caching-based
framework for efficient video DiT inference. It first distinguishes the
interference and boundary between different caching strategies, and then
introduces a context-aware cache triggering strategy to determine when caching
should be enabled, along with an adaptive hybrid cache decision strategy for
dynamically selecting the optimal caching granularity. Extensive experiments on
diverse models demonstrate that, MixCache can significantly accelerate video
generation (e.g., 1.94$\times$ speedup on Wan 14B, 1.97$\times$ speedup on
HunyuanVideo) while delivering both superior generation quality and inference
efficiency compared to baseline methods.

### Computer Science and Game Theory

### 1. [Group Fair Matchings using Convex Cost Functions](http://arxiv.org/pdf/2508.12549v1)

Authors: Atasi Panda, Harsh Sharma, Anand Louis, Prajakta Nimbhorkar

We consider the problem of assigning items to platforms where each item has a
utility associated with each of the platforms to which it can be assigned. Each
platform has a soft constraint over the total number of items it serves,
modeled via a convex cost function. Additionally, items are partitioned into
groups, and each platform also incurs group-specific convex cost over the
number of items from each group that can be assigned to the platform. These
costs promote group fairness by penalizing imbalances, yielding a soft
variation of fairness notions introduced in prior work, such as Restricted
Dominance and Minority protection. Restricted Dominance enforces upper bounds
on group representation, while Minority protection enforces lower bounds. Our
approach replaces such hard constraints with cost-based penalties, allowing
more flexible trade-offs. Our model also captures Nash Social Welfare kind of
objective.
  The cost of an assignment is the sum of the values of all the cost functions
across all the groups and platforms. The objective is to find an assignment
that minimizes the cost while achieving a total utility that is at least a
user-specified threshold. The main challenge lies in balancing the overall
platform cost with group-specific costs, both governed by convex functions,
while meeting the utility constraint. We present an efficient polynomial-time
approximation algorithm, supported by theoretical guarantees and experimental
evaluation. Our algorithm is based on techniques involving linear programming
and network flows. We also provide an exact algorithm for a special case with
uniform utilities and establish the hardness of the general problem when the
groups can intersect arbitrarily.

### 2. [Feedback Linearization for Replicator Dynamics: A Control Framework for Evolutionary Game Convergence](http://arxiv.org/pdf/2508.12583v1)

Authors: Adil Faisal

This paper demonstrates the first application of feedback linearization to
replicator dynamics, driving the evolution of non-convergent evolutionary games
to systems with guaranteed global asymptotic stability.

### Human-Computer Interaction

### 1. [The Future of Tech Labor: How Workers are Organizing and Transforming the Computing Industry](http://arxiv.org/pdf/2508.12579v1)

Authors: Cella M. Sum, Anna Konvicka, Mona Wang, Sarah E. Fox

The tech industry's shifting landscape and the growing precarity of its labor
force have spurred unionization efforts among tech workers. These workers turn
to collective action to improve their working conditions and to protest
unethical practices within their workplaces. To better understand this
movement, we interviewed 44 U.S.-based tech worker-organizers to examine their
motivations, strategies, challenges, and future visions for labor organizing.
These workers included engineers, product managers, customer support
specialists, QA analysts, logistics workers, gig workers, and union staff
organizers. Our findings reveal that, contrary to popular narratives of
prestige and privilege within the tech industry, tech workers face fragmented
and unstable work environments which contribute to their disempowerment and
hinder their organizing efforts. Despite these difficulties, organizers are
laying the groundwork for a more resilient tech worker movement through
community building and expanding political consciousness. By situating these
dynamics within broader structural and ideological forces, we identify ways for
the CSCW community to build solidarity with tech workers who are materially
transforming our field through their organizing efforts.

### 2. [Ashes or Breath: Exploring Moral Dilemmas of Life and Cultural Legacy through Mixed Reality Gaming](http://arxiv.org/pdf/2508.13074v1)

Authors: Black Sun, Ge Kacy Fu, Shichao Guo

Traditional approaches to teaching moral dilemmas often rely on abstract,
disembodied scenarios that limit emotional engagement and reflective depth. To
address this gap, we developed \textit{Ashes or Breath}, a Mixed Reality game
delivered via head-mounted displays(MR-HMDs). This places players in an ethical
crisis: they must save a living cat or a priceless cultural artifact during a
museum fire. Designed through an iterative, values-centered process, the
experience leverages embodied interaction and spatial immersion to heighten
emotional stakes and provoke ethical reflection. Players face irreversible,
emotionally charged choices followed by narrative consequences in a reflective
room, exploring diverse perspectives and societal implications. Preliminary
evaluations suggest that embedding moral dilemmas into everyday environments
via MR-HMDs intensifies empathy, deepens introspection, and encourages users to
reconsider their moral assumptions. This work contributes to ethics-based
experiential learning in HCI, positioning augmented reality not merely as a
medium of interaction but as a stage for ethical encounter.

### 3. [At the Speed of the Heart: Evaluating Physiologically-Adaptive Visualizations for Supporting Engagement in Biking Exergaming in Virtual Reality](http://arxiv.org/pdf/2508.13095v1)

Authors: Oliver Hein, Sandra Wackerl, Changkun Ou, Florian Alt, Francesco Chiossi

Many exergames face challenges in keeping users within safe and effective
intensity levels during exercise. Meanwhile, although wearable devices
continuously collect physiological data, this information is seldom leveraged
for real-time adaptation or to encourage user reflection. We designed and
evaluated a VR cycling simulator that dynamically adapts based on users' heart
rate zones. First, we conducted a user study (N=50) comparing eight
visualization designs to enhance engagement and exertion control, finding that
gamified elements like non-player characters (NPCs) were promising for feedback
delivery. Based on these findings, we implemented a physiology-adaptive
exergame that adjusts visual feedback to keep users within their target heart
rate zones. A lab study (N=18) showed that our system has potential to help
users maintain their target heart rate zones. Subjective ratings of exertion,
enjoyment, and motivation remained largely unchanged between conditions. Our
findings suggest that real-time physiological adaptation through NPC
visualizations can improve workout regulation in exergaming.

### 4. [Choosing the Right Engine in the Virtual Reality Landscape](http://arxiv.org/pdf/2508.13116v1)

Authors: Santiago Berrezueta-Guzman, Stefan Wagner

Virtual reality (VR) development relies on game engines to provide real-time
rendering, physics simulation, and interaction systems. Among the most widely
used game engines, Unreal Engine and Unity dominate the industry, offering
distinct advantages in graphics rendering, performance optimization, usability,
resource requirements, and scalability. This study presents a comprehensive
comparative analysis of both engines, evaluating their capabilities and
trade-offs through empirical assessments and real-world case studies of
large-scale VR projects. The findings highlight key factors such as rendering
fidelity, computational efficiency, cross-platform compatibility, and
development workflows. These provide practical insights for selecting the most
suitable engine based on project-specific needs. Furthermore, emerging trends
in artificial intelligence (AI)-driven enhancements, including Deep Learning
Super Sampling (DLSS) and large language models (LLMs), are explored to assess
their impact on VR development workflows. By aligning engine capabilities with
technical and creative requirements, developers can overcome performance
bottlenecks, enhance immersion, and streamline optimization techniques.
  This study serves as a valuable resource for VR developers, researchers, and
industry professionals, offering data-driven recommendations to navigate the
evolving landscape of VR technology.

### 5. [Human Digital Twin: Data, Models, Applications, and Challenges](http://arxiv.org/pdf/2508.13138v1)

Authors: Rong Pan, Hongyue Sun, Xiaoyu Chen, Giulia Pedrielli, Jiapeng Huang

Human digital twins (HDTs) are dynamic, data-driven virtual representations
of individuals, continuously updated with multimodal data to simulate, monitor,
and predict health trajectories. By integrating clinical, physiological,
behavioral, and environmental inputs, HDTs enable personalized diagnostics,
treatment planning, and anomaly detection. This paper reviews current
approaches to HDT modeling, with a focus on statistical and machine learning
techniques, including recent advances in anomaly detection and failure
prediction. It also discusses data integration, computational methods, and
ethical, technological, and regulatory challenges in deploying HDTs for
precision healthcare.

### 6. [Deformation of the panoramic sphere into an ellipsoid to induce self-motion in telepresence users](http://arxiv.org/pdf/2508.12925v1)

Authors: Eetu Laukka, Evan G. Center, Timo Ojala, Steven M. LaValle, Matti Pouke

Mobile telepresence robots allow users to feel present and explore remote
environments using technology. Traditionally, these systems are implemented
using a camera onboard a mobile robot that can be controlled. Although
high-immersion technologies, such as 360-degree cameras, can increase
situational awareness and presence, they also introduce significant challenges.
Additional processing and bandwidth requirements often result in latencies of
up to seconds. The current delay with a 360-degree camera streaming over the
internet makes real-time control of these systems difficult. Working with
high-latency systems requires some form of assistance to the users.
  This study presents a novel way to utilize optical flow to create an illusion
of self-motion to the user during the latency period between user sending
motion commands to the robot and seeing the actual motion through the
360-camera stream. We find no significant benefit of using the self-motion
illusion to performance or accuracy of controlling a telepresence robot with a
latency of 500 ms, as measured by the task completion time and collisions into
objects. Some evidence is shown that the method might increase virtual reality
(VR) sickness, as measured by the simulator sickness questionnaire (SSQ). We
conclude that further adjustments are necessary in order to render the method
viable.

### 7. [Insights from Interviews with Teachers and Students on the Use of a Social Robot in Computer Science Class in Sixth Grade](http://arxiv.org/pdf/2508.12946v1)

Authors: Ann-Sophie Schenk, Stefan Schiffer, Heqiu Song

In this paper we report on first insights from interviews with teachers and
students on using social robots in computer science class in sixth grade. Our
focus is on learning about requirements and potential applications. We are
particularly interested in getting both perspectives, the teachers' and the
learners' view on how robots could be used and what features they should or
should not have. Results show that teachers as well as students are very open
to robots in the classroom. However, requirements are partially quite
heterogeneous among the groups. This leads to complex design challenges which
we discuss at the end of this paper.

### 8. [Using AI for User Representation: An Analysis of 83 Persona Prompts](http://arxiv.org/pdf/2508.13047v1)

Authors: Joni Salminen, Danial Amin, Bernard Jansen

We analyzed 83 persona prompts from 27 research articles that used large
language models (LLMs) to generate user personas. Findings show that the
prompts predominantly generate single personas. Several prompts express a
desire for short or concise persona descriptions, which deviates from the
tradition of creating rich, informative, and rounded persona profiles. Text is
the most common format for generated persona attributes, followed by numbers.
Text and numbers are often generated together, and demographic attributes are
included in nearly all generated personas. Researchers use up to 12 prompts in
a single study, though most research uses a small number of prompts. Comparison
and testing multiple LLMs is rare. More than half of the prompts require the
persona output in a structured format, such as JSON, and 74% of the prompts
insert data or dynamic variables. We discuss the implications of increased use
of computational personas for user representation.

### 9. [Investigating VR Accessibility Reviews for Users with Disabilities: A Qualitative Analysis](http://arxiv.org/pdf/2508.13051v1)

Authors: Yi Wang, Chetan Arora, Xiao Liu, Thuong Hoang, ZHengxin Zhang, Henry Been Lirn Duh, John Grundy

Accessibility reviews provide valuable insights into both the limitations and
benefits experienced by users with disabilities when using virtual reality (VR)
applications. However, a comprehensive investigation into VR accessibility for
users with disabilities is still lacking. To fill this gap, this study analyzes
user reviews from the Meta and Steam stores of VR apps, focusing on the
reported issues affecting users with disabilities. We applied selection
criteria to 1,367,419 reviews from the top 40, the 20 most popular, and the 40
lowest-rated VR applications on both platforms. In total, 1,076 (0.078%) VR
accessibility reviews referenced various disabilities across 100 VR
applications. These applications were categorized into Action, Sports, Social,
Puzzle, Horror, and Simulation, with Action receiving the highest number of
accessibility related-reviews. We identified 16 different types of disabilities
across six categories. Furthermore, we examined the causes of accessibility
issues as reported by users with disabilities. Overall, VR accessibility
reviews were predominantly under-supported.

### 10. [Seeing the Many: Exploring Parameter Distributions Conditioned on Features in Surrogates](http://arxiv.org/pdf/2508.13088v1)

Authors: Xiaohan Wang, Zhimin Li, Joshua A. Levine, Matthew Berger

Recently, neural surrogate models have emerged as a compelling alternative to
traditional simulation workflows. This is accomplished by modeling the
underlying function of scientific simulations, removing the need to run
expensive simulations. Beyond just mapping from input parameter to output,
surrogates have also been shown useful for inverse problems: output to input
parameters. Inverse problems can be understood as search, where we aim to find
parameters whose surrogate outputs contain a specified feature. Yet finding
these parameters can be costly, especially for high-dimensional parameter
spaces. Thus, existing surrogate-based solutions primarily focus on finding a
small set of matching parameters, in the process overlooking the broader
picture of plausible parameters. Our work aims to model and visualize the
distribution of possible input parameters that produce a given output feature.
To achieve this goal, we aim to address two challenges: (1) the approximation
error inherent in the surrogate model and (2) forming the parameter
distribution in an interactive manner. We model error via density estimation,
reporting high density only if a given parameter configuration is close to
training parameters, measured both over the input and output space. Our density
estimate is used to form a prior belief on parameters, and when combined with a
likelihood on features, gives us an efficient way to sample plausible parameter
configurations that generate a target output feature. We demonstrate the
usability of our solution through a visualization interface by performing
feature-driven parameter analysis over the input parameter space of three
simulation datasets. Source code is available at
https://github.com/matthewberger/seeing-the-many

### Information Retrieval

### 1. [Diagnostic-Guided Dynamic Profile Optimization for LLM-based User Simulators in Sequential Recommendation](http://arxiv.org/pdf/2508.12645v1)

Authors: Hongyang Liu, Zhu Sun, Tianjun Wei, Yan Wang, Jiajie Zhu, Xinghua Qu

Recent advances in large language models (LLMs) have enabled realistic user
simulators for developing and evaluating recommender systems (RSs). However,
existing LLM-based simulators for RSs face two major limitations: (1) static
and single-step prompt-based inference that leads to inaccurate and incomplete
user profile construction; (2) unrealistic and single-round
recommendation-feedback interaction pattern that fails to capture real-world
scenarios. To address these limitations, we propose DGDPO (Diagnostic-Guided
Dynamic Profile Optimization), a novel framework that constructs user profile
through a dynamic and iterative optimization process to enhance the simulation
fidelity. Specifically, DGDPO incorporates two core modules within each
optimization loop: firstly, a specialized LLM-based diagnostic module,
calibrated through our novel training strategy, accurately identifies specific
defects in the user profile. Subsequently, a generalized LLM-based treatment
module analyzes the diagnosed defect and generates targeted suggestions to
refine the profile. Furthermore, unlike existing LLM-based user simulators that
are limited to single-round interactions, we are the first to integrate DGDPO
with sequential recommenders, enabling a bidirectional evolution where user
profiles and recommendation strategies adapt to each other over multi-round
interactions. Extensive experiments conducted on three real-world datasets
demonstrate the effectiveness of our proposed framework.

### 2. [Multi-Granularity Distribution Modeling for Video Watch Time Prediction via Exponential-Gaussian Mixture Network](http://arxiv.org/pdf/2508.12665v1)

Authors: Xu Zhao, Ruibo Ma, Jiaqi Chen, Weiqi Zhao, Ping Yang, Yao Hu

Accurate watch time prediction is crucial for enhancing user engagement in
streaming short-video platforms, although it is challenged by complex
distribution characteristics across multi-granularity levels. Through
systematic analysis of real-world industrial data, we uncover two critical
challenges in watch time prediction from a distribution aspect: (1)
coarse-grained skewness induced by a significant concentration of quick-skips1,
(2) fine-grained diversity arising from various user-video interaction
patterns. Consequently, we assume that the watch time follows the
Exponential-Gaussian Mixture (EGM) distribution, where the exponential and
Gaussian components respectively characterize the skewness and diversity.
Accordingly, an Exponential-Gaussian Mixture Network (EGMN) is proposed for the
parameterization of EGM distribution, which consists of two key modules: a
hidden representation encoder and a mixture parameter generator. We conducted
extensive offline experiments on public datasets and online A/B tests on the
industrial short-video feeding scenario of Xiaohongshu App to validate the
superiority of EGMN compared with existing state-of-the-art methods.
Remarkably, comprehensive experimental results have proven that EGMN exhibits
excellent distribution fitting ability across coarse-to-fine-grained levels. We
open source related code on Github: https://github.com/BestActionNow/EGMN.

### 3. [Deep Research: A Survey of Autonomous Research Agents](http://arxiv.org/pdf/2508.12752v1)

Authors: Wenlin Zhang, Xiaopeng Li, Yingyi Zhang, Pengyue Jia, Yichao Wang, Huifeng Guo, Yong Liu, Xiangyu Zhao

The rapid advancement of large language models (LLMs) has driven the
development of agentic systems capable of autonomously performing complex
tasks. Despite their impressive capabilities, LLMs remain constrained by their
internal knowledge boundaries. To overcome these limitations, the paradigm of
deep research has been proposed, wherein agents actively engage in planning,
retrieval, and synthesis to generate comprehensive and faithful analytical
reports grounded in web-based evidence. In this survey, we provide a systematic
overview of the deep research pipeline, which comprises four core stages:
planning, question developing, web exploration, and report generation. For each
stage, we analyze the key technical challenges and categorize representative
methods developed to address them. Furthermore, we summarize recent advances in
optimization techniques and benchmarks tailored for deep research. Finally, we
discuss open challenges and promising research directions, aiming to chart a
roadmap toward building more capable and trustworthy deep research agents.

### 4. [Informfully Recommenders -- Reproducibility Framework for Diversity-aware Intra-session Recommendations](http://arxiv.org/pdf/2508.13019v1)

Authors: Lucien Heitz, Runze Li, Oana Inel, Abraham Bernstein

Norm-aware recommender systems have gained increased attention, especially
for diversity optimization. The recommender systems community has
well-established experimentation pipelines that support reproducible
evaluations by facilitating models' benchmarking and comparisons against
state-of-the-art methods. However, to the best of our knowledge, there is
currently no reproducibility framework to support thorough norm-driven
experimentation at the pre-processing, in-processing, post-processing, and
evaluation stages of the recommender pipeline. To address this gap, we present
Informfully Recommenders, a first step towards a normative reproducibility
framework that focuses on diversity-aware design built on Cornac. Our extension
provides an end-to-end solution for implementing and experimenting with
normative and general-purpose diverse recommender systems that cover 1) dataset
pre-processing, 2) diversity-optimized models, 3) dedicated intrasession item
re-ranking, and 4) an extensive set of diversity metrics. We demonstrate the
capabilities of our extension through an extensive offline experiment in the
news domain.

### 5. [D-RDW: Diversity-Driven Random Walks for News Recommender Systems](http://arxiv.org/pdf/2508.13035v1)

Authors: Runze Li, Lucien Heitz, Oana Inel, Abraham Bernstein

This paper introduces Diversity-Driven RandomWalks (D-RDW), a lightweight
algorithm and re-ranking technique that generates diverse news recommendations.
D-RDW is a societal recommender, which combines the diversification
capabilities of the traditional random walk algorithms with customizable target
distributions of news article properties. In doing so, our model provides a
transparent approach for editors to incorporate norms and values into the
recommendation process. D-RDW shows enhanced performance across key diversity
metrics that consider the articles' sentiment and political party mentions when
compared to state-of-the-art neural models. Furthermore, D-RDW proves to be
more computationally efficient than existing approaches.

### 6. [Asymmetric Diffusion Recommendation Model](http://arxiv.org/pdf/2508.12706v1)

Authors: Yongchun Zhu, Guanyu Jiang, Jingwu Chen, Feng Zhang, Xiao Yang, Zuotao Liu

Recently, motivated by the outstanding achievements of diffusion models, the
diffusion process has been employed to strengthen representation learning in
recommendation systems. Most diffusion-based recommendation models typically
utilize standard Gaussian noise in symmetric forward and reverse processes in
continuous data space. Nevertheless, the samples derived from recommendation
systems inhabit a discrete data space, which is fundamentally different from
the continuous one. Moreover, Gaussian noise has the potential to corrupt
personalized information within latent representations. In this work, we
propose a novel and effective method, named Asymmetric Diffusion Recommendation
Model (AsymDiffRec), which learns forward and reverse processes in an
asymmetric manner. We define a generalized forward process that simulates the
missing features in real-world recommendation samples. The reverse process is
then performed in an asymmetric latent feature space. To preserve personalized
information within the latent representation, a task-oriented optimization
strategy is introduced. In the serving stage, the raw sample with missing
features is regarded as a noisy input to generate a denoising and robust
representation for the final prediction. By equipping base models with
AsymDiffRec, we conduct online A/B tests, achieving improvements of +0.131% and
+0.166% in terms of users' active days and app usage duration respectively.
Additionally, the extended offline experiments also demonstrate improvements.
AsymDiffRec has been implemented in the Douyin Music App.

### 7. [Is This News Still Interesting to You?: Lifetime-aware Interest Matching for News Recommendation](http://arxiv.org/pdf/2508.13064v1)

Authors: Seongeun Ryu, Yunyong Ko, Sang-Wook Kim

Personalized news recommendation aims to deliver news articles aligned with
users' interests, serving as a key solution to alleviate the problem of
information overload on online news platforms. While prior work has improved
interest matching through refined representations of news and users, the
following time-related challenges remain underexplored: (C1) leveraging the age
of clicked news to infer users' interest persistence, and (C2) modeling the
varying lifetime of news across topics and users. To jointly address these
challenges, we propose a novel Lifetime-aware Interest Matching framework for
nEws recommendation, named LIME, which incorporates three key strategies: (1)
User-Topic lifetime-aware age representation to capture the relative age of
news with respect to a user-topic pair, (2) Candidate-aware lifetime attention
for generating temporally aligned user representation, and (3) Freshness-guided
interest refinement for prioritizing valid candidate news at prediction time.
Extensive experiments on two real-world datasets demonstrate that LIME
consistently outperforms a wide range of state-of-the-art news recommendation
methods, and its model agnostic strategies significantly improve recommendation
accuracy.

### 8. [All for law and law for all: Adaptive RAG Pipeline for Legal Research](http://arxiv.org/pdf/2508.13107v1)

Authors: Figarri Keisha, Prince Singh, Pallavi, Dion Fernandes, Aravindh Manivannan, Ilham Wicaksono, Faisal Ahmad

Retrieval-Augmented Generation (RAG) mitigates hallucinations by grounding
large language model outputs in cited sources, a capability that is especially
critical in the legal domain. We present an end-to-end RAG pipeline that
revisits and extends the LegalBenchRAG baseline with three targeted
enhancements: (i) a context-aware query translator that disentangles document
references from natural-language questions and adapts retrieval depth and
response style based on expertise and specificity, (ii) open-source retrieval
strategies using SBERT and GTE embeddings that achieve substantial performance
gains (improving Recall@K by 30-95\% and Precision@K by $\sim$2.5$\times$ for
$K>4$) while remaining cost-efficient, and (iii) a comprehensive evaluation and
generation framework that combines RAGAS, BERTScore-F1, and ROUGE-Recall to
assess semantic alignment and faithfulness across models and prompt designs.
Our results show that carefully designed open-source pipelines can rival or
outperform proprietary approaches in retrieval quality, while a custom
legal-grounded prompt consistently produces more faithful and contextually
relevant answers than baseline prompting. Taken together, these contributions
demonstrate the potential of task-aware, component-level tuning to deliver
legally grounded, reproducible, and cost-effective RAG systems for legal
research assistance.

### 9. [jXBW: Fast Substructure Search in Large-Scale JSONL Datasets for Foundation Model Applications](http://arxiv.org/pdf/2508.12536v1)

Authors: Yasuo Tabei

Substructure search in JSON Lines (JSONL) datasets is essential for modern
applications such as prompt engineering in foundation models, but existing
methods suffer from prohibitive computational costs due to exhaustive tree
traversal and subtree matching. We present jXBW, a fast method for substructure
search on large-scale JSONL datasets. Our method makes three key technical
contributions: (i) a merged tree representation built by merging trees of
multiple JSON objects while preserving individual identities, (ii) a succinct
data structure based on the eXtended Burrows-Wheeler Transform that enables
efficient tree navigation and subpath search, and (iii) an efficient three-step
substructure search algorithm that combines path decomposition, ancestor
computation, and adaptive tree identifier collection to ensure correctness
while avoiding exhaustive tree traversal. Experimental evaluation on real-world
datasets demonstrates that jXBW consistently outperforms existing methods,
achieving speedups of 16$\times$ for smaller datasets and up to 4,700$\times$
for larger datasets over tree-based approaches, and more than 6$\times$10$^6$
over XML-based processing while maintaining competitive memory usage.

### Machine Learning

### 1. [Illuminating LLM Coding Agents: Visual Analytics for Deeper Understanding and Enhancement](http://arxiv.org/pdf/2508.12555v1)

Authors: Junpeng Wang, Yuzhong Chen, Menghai Pan, Chin-Chia Michael Yeh, Mahashweta Das

Coding agents powered by large language models (LLMs) have gained traction
for automating code generation through iterative problem-solving with minimal
human involvement. Despite the emergence of various frameworks, e.g.,
LangChain, AutoML, and AIDE, ML scientists still struggle to effectively review
and adjust the agents' coding process. The current approach of manually
inspecting individual outputs is inefficient, making it difficult to track code
evolution, compare coding iterations, and identify improvement opportunities.
To address this challenge, we introduce a visual analytics system designed to
enhance the examination of coding agent behaviors. Focusing on the AIDE
framework, our system supports comparative analysis across three levels: (1)
Code-Level Analysis, which reveals how the agent debugs and refines its code
over iterations; (2) Process-Level Analysis, which contrasts different
solution-seeking processes explored by the agent; and (3) LLM-Level Analysis,
which highlights variations in coding behavior across different LLMs. By
integrating these perspectives, our system enables ML scientists to gain a
structured understanding of agent behaviors, facilitating more effective
debugging and prompt engineering. Through case studies using coding agents to
tackle popular Kaggle competitions, we demonstrate how our system provides
valuable insights into the iterative coding process.

### 2. [Deep Learning-Based Financial Time Series Forecasting via Sliding Window and Variational Mode Decomposition](http://arxiv.org/pdf/2508.12565v1)

Authors: Luke Li

To address the complexity of financial time series, this paper proposes a
forecasting model combining sliding window and variational mode decomposition
(VMD) methods. Historical stock prices and relevant market indicators are used
to construct datasets. VMD decomposes non-stationary financial time series into
smoother subcomponents, improving model adaptability. The decomposed data is
then input into a deep learning model for prediction. The study compares the
forecasting effects of an LSTM model trained on VMD-processed sequences with
those using raw time series, demonstrating better performance and stability.

### 3. [Physics-informed deep operator network for traffic state estimation](http://arxiv.org/pdf/2508.12593v1)

Authors: Zhihao Li, Ting Wang, Guojian Zou, Ruofei Wang, Ye Li

Traffic state estimation (TSE) fundamentally involves solving
high-dimensional spatiotemporal partial differential equations (PDEs) governing
traffic flow dynamics from limited, noisy measurements. While Physics-Informed
Neural Networks (PINNs) enforce PDE constraints point-wise, this paper adopts a
physics-informed deep operator network (PI-DeepONet) framework that
reformulates TSE as an operator learning problem. Our approach trains a
parameterized neural operator that maps sparse input data to the full
spatiotemporal traffic state field, governed by the traffic flow conservation
law. Crucially, unlike PINNs that enforce PDE constraints point-wise,
PI-DeepONet integrates traffic flow conservation model and the fundamental
diagram directly into the operator learning process, ensuring physical
consistency while capturing congestion propagation, spatial correlations, and
temporal evolution. Experiments on the NGSIM dataset demonstrate superior
performance over state-of-the-art baselines. Further analysis reveals insights
into optimal function generation strategies and branch network complexity.
Additionally, the impact of input function generation methods and the number of
functions on model performance is explored, highlighting the robustness and
efficacy of proposed framework.

### 4. [FLARE: Fast Low-rank Attention Routing Engine](http://arxiv.org/pdf/2508.12594v1)

Authors: Vedant Puri, Aditya Joglekar, Kevin Ferguson, Yu-hsuan Chen, Yongjie Jessica Zhang, Levent Burak Kara

The quadratic complexity of self-attention limits its applicability and
scalability on large unstructured meshes. We introduce Fast Low-rank Attention
Routing Engine (FLARE), a linear complexity self-attention mechanism that
routes attention through fixed-length latent sequences. Each attention head
performs global communication among $N$ tokens by projecting the input sequence
onto a fixed length latent sequence of $M \ll N$ tokens using learnable query
tokens. By routing attention through a bottleneck sequence, FLARE learns a
low-rank form of attention that can be applied at $O(NM)$ cost. FLARE not only
scales to unprecedented problem sizes, but also delivers superior accuracy
compared to state-of-the-art neural PDE surrogates across diverse benchmarks.
We also release a new additive manufacturing dataset to spur further research.
Our code is available at https://github.com/vpuri3/FLARE.py.

### 5. [Constructing Invariant and Equivariant Operations by Symmetric Tensor Network](http://arxiv.org/pdf/2508.12596v1)

Authors: Meng Zhang, Chao Wang, Hao Zhang, Shaojun Dong, Lixin He

Design of neural networks that incorporate symmetry is crucial for geometric
deep learning. Central to this effort is the development of invariant and
equivariant operations. This works presents a systematic method for
constructing valid invariant and equivariant operations. It can handle inputs
and outputs in the form of Cartesian tensors with different rank, as well as
spherical tensors with different types. In addition, our method features a
graphical representation utilizing the symmetric tensor network, which
simplifies both the proofs and constructions related to invariant and
equivariant functions. We also apply this approach to design the equivariant
interaction message for the geometry graph neural network, and equivariant
machine learning model to learn the constitutive law of materials.

### 6. [A Hybrid Surrogate for Electric Vehicle Parameter Estimation and Power Consumption via Physics-Informed Neural Operators](http://arxiv.org/pdf/2508.12602v1)

Authors: Hansol Lim, Jongseong Brad Choi, Jee Won Lee, Haeseong Jeoung, Minkyu Han

We present a hybrid surrogate model for electric vehicle parameter estimation
and power consumption. We combine our novel architecture Spectral Parameter
Operator built on a Fourier Neural Operator backbone for global context and a
differentiable physics module in the forward pass. From speed and acceleration
alone, it outputs time-varying motor and regenerative braking efficiencies, as
well as aerodynamic drag, rolling resistance, effective mass, and auxiliary
power. These parameters drive a physics-embedded estimate of battery power,
eliminating any separate physics-residual loss. The modular design lets
representations converge to physically meaningful parameters that reflect the
current state and condition of the vehicle. We evaluate on real-world logs from
a Tesla Model 3, Tesla Model S, and the Kia EV9. The surrogate achieves a mean
absolute error of 0.2kW (about 1% of average traction power at highway speeds)
for Tesla vehicles and about 0.8kW on the Kia EV9. The framework is
interpretable, and it generalizes well to unseen conditions, and sampling
rates, making it practical for path optimization, eco-routing, on-board
diagnostics, and prognostics health management.

### 7. [FedSODA: Federated Fine-tuning of LLMs via Similarity Group Pruning and Orchestrated Distillation Alignment](http://arxiv.org/pdf/2508.12727v1)

Authors: Manning Zhu, Songtao Guo, Pengzhan Zhou, Yansong Ning, Chang Han, Dewen Qiao

Federated fine-tuning (FFT) of large language models (LLMs) has recently
emerged as a promising solution to enable domain-specific adaptation while
preserving data privacy. Despite its benefits, FFT on resource-constrained
clients relies on the high computational and memory demands of full-model
fine-tuning, which limits the potential advancement. This paper presents
FedSODA, a resource-efficient FFT framework that enables clients to adapt LLMs
without accessing or storing the full model. Specifically, we first propose a
similarity group pruning (SGP) module, which prunes redundant layers from the
full LLM while retaining the most critical layers to preserve the model
performance. Moreover, we introduce an orchestrated distillation alignment
(ODA) module to reduce gradient divergence between the sub-LLM and the full LLM
during FFT. Through the use of the QLoRA, clients only need to deploy quantized
sub-LLMs and fine-tune lightweight adapters, significantly reducing local
resource requirements. We conduct extensive experiments on three open-source
LLMs across a variety of downstream tasks. The experimental results demonstrate
that FedSODA reduces communication overhead by an average of 70.6%, decreases
storage usage by 75.6%, and improves task accuracy by 3.1%, making it highly
suitable for practical FFT applications under resource constraints.

### 8. [Online Ensemble Transformer for Accurate Cloud Workload Forecasting in Predictive Auto-Scaling](http://arxiv.org/pdf/2508.12773v1)

Authors: Jiadong Chen, Xiao He, Hengyu Ye, Fuxin Jiang, Tieying Zhang, Jianjun Chen, Xiaofeng Gao

In the swiftly evolving domain of cloud computing, the advent of serverless
systems underscores the crucial need for predictive auto-scaling systems. This
necessity arises to ensure optimal resource allocation and maintain operational
efficiency in inherently volatile environments. At the core of a predictive
auto-scaling system is the workload forecasting model. Existing forecasting
models struggle to quickly adapt to the dynamics in online workload streams and
have difficulty capturing the complex periodicity brought by fine-grained,
high-frequency forecasting tasks. Addressing this, we propose a novel online
ensemble model, E3Former, for online workload forecasting in large-scale
predictive auto-scaling. Our model synergizes the predictive capabilities of
multiple subnetworks to surmount the limitations of single-model approaches,
thus ensuring superior accuracy and robustness. Remarkably, it accomplishes
this with a minimal increase in computational overhead, adhering to the lean
operational ethos of serverless systems. Through extensive experimentation on
real-world workload datasets, we establish the efficacy of our ensemble model.
In online forecasting tasks, the proposed method reduces forecast error by an
average of 10%, and its effectiveness is further demonstrated through a
predictive auto-scaling test in the real-life online system. Currently, our
method has been deployed within ByteDance's Intelligent Horizontal Pod
Auto-scaling (IHPA) platform, which supports the stable operation of over 30
applications, such as Douyin E-Comerce, TouTiao, and Volcano Engine. The
predictive auto-scaling capacity reaching over 600,000 CPU cores. On the basis
of essentially ensuring service quality, the predictive auto-scaling system can
reduce resource utilization by over 40%.

### 9. [Wavy Transformer](http://arxiv.org/pdf/2508.12787v1)

Authors: Satoshi Noguchi, Yoshinobu Kawahara

Transformers have achieved remarkable success across natural language
processing (NLP) and computer vision (CV). However, deep transformer models
often suffer from an over-smoothing issue, in which token representations
converge to similar values as they pass through successive transformer blocks.
In this paper, we establish an equivalence between the hidden-state dynamics
induced by stacked attention layers and graph neural diffusion on a complete
graph. From this perspective, over-smoothing can be interpreted as a
consequence of the dissipative nature of the underlying diffusion dynamics.
Motivated by this physical interpretation, we propose Wavy Transformer, which
consists of a novel attention layer based on second-order wavy dynamics. We
also introduce a feed-forward network and a normalization layer designed to
preserve the physical state-velocity relationship under the chain rule, thereby
extending the transformer architecture. We further validate our proposed
techniques on various transformer models for NLP and CV tasks. The results
consistently demonstrate that Wavy Transformer improves performance with
minimal additional parameters and no extra hyperparameter tuning.

### 10. [Learning In-context $\pmb{n}$-grams with Transformers: Sub-$\pmb{n}$-grams Are Near-stationary Points](http://arxiv.org/pdf/2508.12837v1)

Authors: Aditya Varre, Gizem Yüce, Nicolas Flammarion

Motivated by empirical observations of prolonged plateaus and stage-wise
progression during training, we investigate the loss landscape of transformer
models trained on in-context next-token prediction tasks. In particular, we
focus on learning in-context $n$-gram language models under cross-entropy loss,
and establish a sufficient condition for parameter configurations to be
stationary points. We then construct a set of parameter configurations for a
simplified transformer model that represent $k$-gram estimators (for $k \leq
n$), and show that the gradient of the population loss at these solutions
vanishes in the limit of infinite sequence length and parameter norm. This
reveals a key property of the loss landscape: {sub-$n$-grams are
near-stationary points of the population cross-entropy loss}, offering
theoretical insight into widely observed phenomena such as stage-wise learning
dynamics and emergent phase transitions. These insights are further supported
by numerical experiments that illustrate the learning dynamics of $n$-grams,
characterized by discrete transitions between near-stationary solutions.

### Neural and Evolutionary Computing

### 1. [A Self-Ensemble Inspired Approach for Effective Training of Binary-Weight Spiking Neural Networks](http://arxiv.org/pdf/2508.12609v1)

Authors: Qingyan Meng, Mingqing Xiao, Zhengyu Ma, Huihui Zhou, Yonghong Tian, Zhouchen Lin

Spiking Neural Networks (SNNs) are a promising approach to low-power
applications on neuromorphic hardware due to their energy efficiency. However,
training SNNs is challenging because of the non-differentiable spike generation
function. To address this issue, the commonly used approach is to adopt the
backpropagation through time framework, while assigning the gradient of the
non-differentiable function with some surrogates. Similarly, Binary Neural
Networks (BNNs) also face the non-differentiability problem and rely on
approximating gradients. However, the deep relationship between these two
fields and how their training techniques can benefit each other has not been
systematically researched. Furthermore, training binary-weight SNNs is even
more difficult. In this work, we present a novel perspective on the dynamics of
SNNs and their close connection to BNNs through an analysis of the
backpropagation process. We demonstrate that training a feedforward SNN can be
viewed as training a self-ensemble of a binary-activation neural network with
noise injection. Drawing from this new understanding of SNN dynamics, we
introduce the Self-Ensemble Inspired training method for (Binary-Weight) SNNs
(SEI-BWSNN), which achieves high-performance results with low latency even for
the case of the 1-bit weights. Specifically, we leverage a structure of
multiple shortcuts and a knowledge distillation-based training technique to
improve the training of (binary-weight) SNNs. Notably, by binarizing FFN layers
in a Transformer architecture, our approach achieves 82.52% accuracy on
ImageNet with only 2 time steps, indicating the effectiveness of our
methodology and the potential of binary-weight SNNs.

### 2. [IzhiRISC-V -- a RISC-V-based Processor with Custom ISA Extension for Spiking Neuron Networks Processing with Izhikevich Neurons](http://arxiv.org/pdf/2508.12846v1)

Authors: Wiktor J. Szczerek, Artur Podobas

Spiking Neural Network processing promises to provide high energy efficiency
due to the sparsity of the spiking events. However, when realized on
general-purpose hardware -- such as a RISC-V processor -- this promise can be
undermined and overshadowed by the inefficient code, stemming from repeated
usage of basic instructions for updating all the neurons in the network. One of
the possible solutions to this issue is the introduction of a custom ISA
extension with neuromorphic instructions for spiking neuron updating, and
realizing those instructions in bespoke hardware expansion to the existing ALU.
In this paper, we present the first step towards realizing a large-scale system
based on the RISC-V-compliant processor called IzhiRISC-V, supporting the
custom neuromorphic ISA extension.

### 3. [HOMI: Ultra-Fast EdgeAI platform for Event Cameras](http://arxiv.org/pdf/2508.12637v1)

Authors: Shankaranarayanan H, Satyapreet Singh Yadav, Adithya Krishna, Ajay Vikram P, Mahesh Mehendale, Chetan Singh Thakur

Event cameras offer significant advantages for edge robotics applications due
to their asynchronous operation and sparse, event-driven output, making them
well-suited for tasks requiring fast and efficient closed-loop control, such as
gesture-based human-robot interaction. Despite this potential, existing event
processing solutions remain limited, often lacking complete end-to-end
implementations, exhibiting high latency, and insufficiently exploiting event
data sparsity. In this paper, we present HOMI, an ultra-low latency, end-to-end
edge AI platform comprising a Prophesee IMX636 event sensor chip with an Xilinx
Zynq UltraScale+MPSoC FPGA chip, deploying an in-house developed AI
accelerator. We have developed hardware-optimized pre-processing pipelines
supporting both constant-time and constant-event modes for histogram
accumulation, linear and exponential time surfaces. Our general-purpose
implementation caters to both accuracy-driven and low-latency applications.
HOMI achieves 94% accuracy on the DVS Gesture dataset as a use case when
configured for high accuracy operation and provides a throughput of 1000 fps
for low-latency configuration. The hardware-optimised pipeline maintains a
compact memory footprint and utilises only 33% of the available LUT resources
on the FPGA, leaving ample headroom for further latency reduction, model
parallelisation, multi-task deployments, or integration of more complex
architectures.

### 4. [A Unified Cortical Circuit Model with Divisive Normalization and Self-Excitation for Robust Representation and Memory Maintenance](http://arxiv.org/pdf/2508.12702v1)

Authors: Jie Su, Weiwei Wang, Zhaotian Gu, Dahui Wang, Tianyi Qian

Robust information representation and its persistent maintenance are
fundamental for higher cognitive functions. Existing models employ distinct
neural mechanisms to separately address noise-resistant processing or
information maintenance, yet a unified framework integrating both operations
remains elusive -- a critical gap in understanding cortical computation. Here,
we introduce a recurrent neural circuit that combines divisive normalization
with self-excitation to achieve both robust encoding and stable retention of
normalized inputs. Mathematical analysis shows that, for suitable parameter
regimes, the system forms a continuous attractor with two key properties: (1)
input-proportional stabilization during stimulus presentation; and (2)
self-sustained memory states persisting after stimulus offset. We demonstrate
the model's versatility in two canonical tasks: (a) noise-robust encoding in a
random-dot kinematogram (RDK) paradigm; and (b) approximate Bayesian belief
updating in a probabilistic Wisconsin Card Sorting Test (pWCST). This work
establishes a unified mathematical framework that bridges noise suppression,
working memory, and approximate Bayesian inference within a single cortical
microcircuit, offering fresh insights into the brain's canonical computation
and guiding the design of biologically plausible artificial neural
architectures.

### 5. [Synchronization and semantization in deep spiking networks](http://arxiv.org/pdf/2508.12975v1)

Authors: Jonas Oberste-Frielinghaus, Anno C. Kurth, Julian Göltz, Laura Kriener, Junji Ito, Mihai A. Petrovici, Sonja Grün

Recent studies have shown how spiking networks can learn complex
functionality through error-correcting plasticity, but the resulting structures
and dynamics remain poorly studied. To elucidate how these models may link to
observed dynamics in vivo and thus how they may ultimately explain cortical
computation, we need a better understanding of their emerging patterns. We
train a multi-layer spiking network, as a conceptual analog of the bottom-up
visual hierarchy, for visual input classification using spike-time encoding.
After learning, we observe the development of distinct spatio-temporal activity
patterns. While input patterns are synchronous by construction, activity in
early layers first spreads out over time, followed by re-convergence into sharp
pulses as classes are gradually extracted. The emergence of synchronicity is
accompanied by the formation of increasingly distinct pathways, reflecting the
gradual semantization of input activity. We thus observe hierarchical networks
learning spike latency codes to naturally acquire activity patterns
characterized by synchronicity and separability, with pronounced excitatory
pathways ascending through the layers. This provides a rigorous computational
hypothesis for the experimentally observed synchronicity in the visual system
as a natural consequence of deep learning in cortex.

### 6. [Cyber Risks to Next-Gen Brain-Computer Interfaces: Analysis and Recommendations](http://arxiv.org/pdf/2508.12571v1)

Authors: Tyler Schroder, Renee Sirbu, Sohee Park, Jessica Morley, Sam Street, Luciano Floridi

Brain-computer interfaces (BCIs) show enormous potential for advancing
personalized medicine. However, BCIs also introduce new avenues for
cyber-attacks or security compromises. In this article, we analyze the problem
and make recommendations for device manufacturers to better secure devices and
to help regulators understand where more guidance is needed to protect patient
safety and data confidentiality. Device manufacturers should implement the
prior suggestions in their BCI products. These recommendations help protect BCI
users from undue risks, including compromised personal health and genetic
information, unintended BCI-mediated movement, and many other cybersecurity
breaches. Regulators should mandate non-surgical device update methods, strong
authentication and authorization schemes for BCI software modifications,
encryption of data moving to and from the brain, and minimize network
connectivity where possible. We also design a hypothetical, average-case threat
model that identifies possible cybersecurity threats to BCI patients and
predicts the likeliness of risk for each category of threat. BCIs are at less
risk of physical compromise or attack, but are vulnerable to remote attack; we
focus on possible threats via network paths to BCIs and suggest technical
controls to limit network connections.

### Networking and Internet Architecture

### 1. [An Efficient and Adaptive Framework for Achieving Underwater High-performance Maintenance Networks](http://arxiv.org/pdf/2508.12661v1)

Authors: Yu Gou, Tong Zhang, Jun Liu, Zhongyang Qi, Dezhi Zheng

With the development of space-air-ground-aqua integrated networks (SAGAIN),
high-speed and reliable network services are accessible at any time and any
location. However, the long propagation delay and limited network capacity of
underwater communication networks (UCN) negatively impact the service quality
of SAGAIN. To address this issue, this paper presents U-HPNF, a hierarchical
framework designed to achieve a high-performance network with self-management,
self-configuration, and self-optimization capabilities. U-HPNF leverages the
sensing and decision-making capabilities of deep reinforcement learning (DRL)
to manage limited resources in UCNs, including communication bandwidth,
computational resources, and energy supplies. Additionally, we incorporate
federated learning (FL) to iteratively optimize the decision-making model,
thereby reducing communication overhead and protecting the privacy of node
observation information. By deploying digital twins (DT) at both the
intelligent sink layer and aggregation layer, U-HPNF can mimic numerous network
scenarios and adapt to varying network QoS requirements. Through a three-tier
network design with two-levels DT, U-HPNF provides an AI-native
high-performance underwater network. Numerical results demonstrate that the
proposed U-HPNF framework can effectively optimize network performance across
various situations and adapt to changing QoS requirements.

### 2. [Towards Nomadic 6G Communication Networks: Implications on Architecture, Standardization, and Regulatory Aspects](http://arxiv.org/pdf/2508.12710v1)

Authors: Daniel Lindenschmitt, Marcos Rates Crippa, Hans D. Schotten

The emergence of nomadic mobile communication networks for sixth-generation
(6G) introduces a paradigm shift in how network infrastructure is
conceptualized, deployed, and operated. Unlike traditional fixed systems,
Nomadic Networks (NNs) consist of mobile and self-organizing nodes that provide
radio infrastructure capabilities in motion. This paper explores the
architectural implications of such systems, with a particular focus on the
design and evolution of network interfaces. We analyze the requirements for
inter-node communication, service discovery, and control delegation in dynamic
environments. Furthermore, we examine the regulatory and licensing challenges
that arise when infrastructure elements traverse jurisdictional boundaries.
Based on current 6G visions and relevant research, we identify limitations in
existing architectures and propose a set of interface principles tailored to
nomadicity. By synthesizing findings from mobile, non-terrestrial, and organic
network domains, this work contributes to the architectural foundation for
future nomadic 6G communication systems and outlines directions for interface
standardization in decentralized, mobile infrastructures.

### 3. [Cooperative Sensing-Assisted Predictive Beam Tracking for MIMO-OFDM Networked ISAC Systems](http://arxiv.org/pdf/2508.12723v1)

Authors: Xiaoyu Yang, Zhiqing Wei, Jie Xu, Huici Wu, Zhiyong Feng

This paper studies a multiple-input multiple-output (MIMO) orthogonal
frequency division multiplexing (OFDM) networked integrated sensing and
communication (ISAC) system, in which multiple base stations (BSs) perform beam
tracking to communicate with a mobile device. In particular, we focus on the
beam tracking over a number of tracking time slots (TTSs) and suppose that
these BSs operate at non-overlapping frequency bands to avoid the severe
inter-cell interference. Under this setup, we propose a new cooperative
sensing-assisted predictive beam tracking design. In each TTS, the BSs use echo
signals to cooperatively track the mobile device as a sensing target, and
continuously adjust the beam directions to follow the device for enhancing the
performance for both communication and sensing. First, we propose a cooperative
sensing design to track the device, in which the BSs first employ the
two-dimensional discrete Fourier transform (2D-DFT) technique to perform local
target estimation, and then use the extended Kalman filter (EKF) method to fuse
their individual measurement results for predicting the target parameters.
Next, based on the predicted results, we obtain the achievable rate for
communication and the predicted conditional Cram\'er-Rao lower bound (PC-CRLB)
for target parameters estimation in the next TTS, as a function of the
beamforming vectors. Accordingly, we formulate the predictive beamforming
design problem, with the objective of maximizing the achievable communication
rate in the following TTS, while satisfying the PC-CRLB requirement for
sensing. To address the resulting non-convex problem, we first propose a
semi-definite relaxation (SDR)-based algorithm to obtain the optimal solution,
and then develop an alternative penalty-based algorithm to get a high-quality
low-complexity solution.

### 4. [Some optimization possibilities in data plane programming](http://arxiv.org/pdf/2508.12767v1)

Authors: Altangerel Gereltsetseg, Tejfel Máté

Software-defined networking (SDN) technology aims to create a highly flexible
network by decoupling control plane and the data plane and programming them
independently. There has been a lot of research on improving and optimizing the
control plane, and data plane programming is a relatively new concept, so study
on it is one of the hot topics for researchers. At the 2019 Dagstuhl Seminar,
well-known scientists on computer networking discussed challenges and problems
in the field of data plane programming that need to be addressed over the next
10 years. Based on this seminar issues and papers review, we suggested some
possible solutions which are for optimizing data plane to improve packet
processing performance and link utilization. The suggestions include (i)
enriching data plane language with asynchronous external function, (ii)
compression based on payload size, (iii) in-network caching for fast packet
processing, and (iv) offloading external functions to an additional thread,
virtual machine (VM) or server, etc. In addition, we implemented some of these
in the P4 data plane language to illustrate the practicality.

### 5. [SDAP-based QoS Flow Multiplexing Support in Simu5G for 5G NR Simulation](http://arxiv.org/pdf/2508.12785v1)

Authors: Mohamed Seliem, Utz Roedig, Cormac Sreenan, Dirk Pesch

The Service Data Adaptation Protocol (SDAP) plays a central role in 5G New
Radio (NR), acting as a bridge between the core and radio networks, by enabling
QoS Flow multiplexing over shared Data Radio Bearers (DRBs). However, most 5G
simulation frameworks, including the popular OMNet++-based Simu5G, lack SDAP
support, limiting their ability to model realistic QoS behavior. This paper
presents a modular, standardscompliant SDAP extension for Simu5G. The
implementation includes core elements such as QoS Flow Identifer (QFI) flow
tagging, SDAP header insertion/removal, and configurable logical DRB mapping.
The proposed design supports multi-QFI simulation scenarios and enables
researchers to model differentiated QoS flows and flowaware scheduling
policies. Validation results confirm correct SDAP behavior and pave the way for
advanced 5G simulations involving per-flow isolation, latency-sensitive
traffic, and industrial QoS profiles.

### 6. [RoTO: Robust Topology Obfuscation Against Tomography Inference Attacks](http://arxiv.org/pdf/2508.12852v1)

Authors: Chengze Du, Heng Xu, Zhiwei Yu, Ying Zhou, Zili Meng, Jialong Li

Tomography inference attacks aim to reconstruct network topology by analyzing
end-to-end probe delays. Existing defenses mitigate these attacks by
manipulating probe delays to mislead inference, but rely on two strong
assumptions: (i) probe packets can be perfectly detected and altered, and (ii)
attackers use known, fixed inference algorithms. These assumptions often break
in practice, leading to degraded defense performance under detection errors or
adaptive adversaries. We present RoTO, a robust topology obfuscation scheme
that eliminates both assumptions by modeling uncertainty in attacker-observed
delays through a distributional formulation. RoTO casts the defense objective
as a min-max optimization problem that maximizes expected topological
distortion across this uncertainty set, without relying on perfect probe
control or specific attacker models. To approximate attacker behavior, RoTO
leverages graph neural networks for inference simulation and adversarial
training. We also derive an upper bound on attacker success probability, and
demonstrate that our approach enhances topology obfuscation performance through
the optimization of this upper bound. Experimental results show that RoTO
outperforms existing defense methods, achieving average improvements of 34% in
structural similarity and 42.6% in link distance while maintaining strong
robustness and concealment capabilities.

### 7. [REACH: Reinforcement Learning for Efficient Allocation in Community and Heterogeneous Networks](http://arxiv.org/pdf/2508.12857v1)

Authors: Zhiwei Yu, Chengze Du, Heng Xu, Ying Zhou, Bo Liu, Jialong Li

Community GPU platforms are emerging as a cost-effective and democratized
alternative to centralized GPU clusters for AI workloads, aggregating idle
consumer GPUs from globally distributed and heterogeneous environments.
However, their extreme hardware/software diversity, volatile availability, and
variable network conditions render traditional schedulers ineffective, leading
to suboptimal task completion. In this work, we present REACH (Reinforcement
Learning for Efficient Allocation in Community and Heterogeneous Networks), a
Transformer-based reinforcement learning framework that redefines task
scheduling as a sequence scoring problem to balance performance, reliability,
cost, and network efficiency. By modeling both global GPU states and task
requirements, REACH learns to adaptively co-locate computation with data,
prioritize critical jobs, and mitigate the impact of unreliable resources.
Extensive simulation results show that REACH improves task completion rates by
up to 17%, more than doubles the success rate for high-priority tasks, and
reduces bandwidth penalties by over 80% compared to state-of-the-art baselines.
Stress tests further demonstrate its robustness to GPU churn and network
congestion, while scalability experiments confirm its effectiveness in
large-scale, high-contention scenarios.

### 8. [Game-Theoretic and Reinforcement Learning-Based Cluster Head Selection for Energy-Efficient Wireless Sensor Network](http://arxiv.org/pdf/2508.12707v1)

Authors: Mehrshad Eskandarpour, Saba Pirahmadian, Parham Soltani, Hossein Soleimani

Energy in Wireless Sensor Networks (WSNs) is critical to network lifetime and
data delivery. However, the primary impediment to the durability and
dependability of these sensor nodes is their short battery life. Currently,
power-saving algorithms such as clustering and routing algorithms have improved
energy efficiency in standard protocols. This paper proposes a clustering-based
routing approach for creating an adaptive, energy-efficient mechanism. Our
system employs a multi-step clustering strategy to select dynamic cluster heads
(CH) with optimal energy distribution. We use Game Theory (GT) and
Reinforcement Learning (RL) to optimize resource utilization. Modeling the
network as a multi-agent RL problem using GT principles allows for
self-clustering while optimizing sensor lifetime and energy balance. The
proposed AI-powered CH-Finding algorithm improves network efficiency by
preventing premature energy depletion in specific nodes while also ensuring
uniform energy usage across the network. Our solution enables controlled power
consumption, resulting in a deterministic network lifetime. This predictability
lowers maintenance costs by reducing the need for node replacement.
Furthermore, our proposed method prevents sensor nodes from disconnecting from
the network by designating the sensor with the highest charge as an
intermediary and using single-hop routing. This approach improves the energy
efficiency and stability of Wireless Sensor Network (WSN) deployments.

### 9. [SL-ACC: A Communication-Efficient Split Learning Framework with Adaptive Channel-wise Compression](http://arxiv.org/pdf/2508.12984v1)

Authors: Zehang Lin, Zheng Lin, Miao Yang, Jianhao Huang, Yuxin Zhang, Zihan Fang, Xia Du, Zhe Chen, Shunzhi Zhu, Wei Ni

The increasing complexity of neural networks poses a significant barrier to
the deployment of distributed machine learning (ML) on resource-constrained
devices, such as federated learning (FL). Split learning (SL) offers a
promising solution by offloading the primary computing load from edge devices
to a server via model partitioning. However, as the number of participating
devices increases, the transmission of excessive smashed data (i.e.,
activations and gradients) becomes a major bottleneck for SL, slowing down the
model training. To tackle this challenge, we propose a communication-efficient
SL framework, named SL-ACC, which comprises two key components: adaptive
channel importance identification (ACII) and channel grouping compression
(CGC). ACII first identifies the contribution of each channel in the smashed
data to model training using Shannon entropy. Following this, CGC groups the
channels based on their entropy and performs group-wise adaptive compression to
shrink the transmission volume without compromising training accuracy.
Extensive experiments across various datasets validate that our proposed SL-ACC
framework takes considerably less time to achieve a target accuracy than
state-of-the-art benchmarks.

### Robotics

### 1. [RoboRetriever: Single-Camera Robot Object Retrieval via Active and Interactive Perception with Dynamic Scene Graph](http://arxiv.org/pdf/2508.12916v1)

Authors: Hecheng Wang, Jiankun Ren, Jia Yu, Lizhe Qi, Yunquan Sun

Humans effortlessly retrieve objects in cluttered, partially observable
environments by combining visual reasoning, active viewpoint adjustment, and
physical interaction-with only a single pair of eyes. In contrast, most
existing robotic systems rely on carefully positioned fixed or multi-camera
setups with complete scene visibility, which limits adaptability and incurs
high hardware costs. We present \textbf{RoboRetriever}, a novel framework for
real-world object retrieval that operates using only a \textbf{single}
wrist-mounted RGB-D camera and free-form natural language instructions.
RoboRetriever grounds visual observations to build and update a \textbf{dynamic
hierarchical scene graph} that encodes object semantics, geometry, and
inter-object relations over time. The supervisor module reasons over this
memory and task instruction to infer the target object and coordinate an
integrated action module combining \textbf{active perception},
\textbf{interactive perception}, and \textbf{manipulation}. To enable
task-aware scene-grounded active perception, we introduce a novel visual
prompting scheme that leverages large reasoning vision-language models to
determine 6-DoF camera poses aligned with the semantic task goal and geometry
scene context. We evaluate RoboRetriever on diverse real-world object retrieval
tasks, including scenarios with human intervention, demonstrating strong
adaptability and robustness in cluttered scenes with only one RGB-D camera.

### 2. [Simultaneous Contact Sequence and Patch Planning for Dynamic Locomotion](http://arxiv.org/pdf/2508.12928v1)

Authors: Victor Dhédin, Haizhou Zhao, Majid Khadiv

Legged robots have the potential to traverse highly constrained environments
with agile maneuvers. However, planning such motions requires solving a highly
challenging optimization problem with a mixture of continuous and discrete
decision variables. In this paper, we present a full pipeline based on
Monte-Carlo tree search (MCTS) and whole-body trajectory optimization (TO) to
perform simultaneous contact sequence and patch selection on highly challenging
environments. Through extensive simulation experiments, we show that our
framework can quickly find a diverse set of dynamically consistent plans. We
experimentally show that these plans are transferable to a real quadruped
robot. We further show that the same framework can find highly complex acyclic
humanoid maneuvers. To the best of our knowledge, this is the first
demonstration of simultaneous contact sequence and patch selection for acyclic
multi-contact locomotion using the whole-body dynamics of a quadruped.

### 3. [Scaling Whole-body Multi-contact Manipulation with Contact Optimization](http://arxiv.org/pdf/2508.12980v1)

Authors: Victor Levé, João Moura, Sachiya Fujita, Tamon Miyake, Steve Tonneau, Sethu Vijayakumar

Daily tasks require us to use our whole body to manipulate objects, for
instance when our hands are unavailable. We consider the issue of providing
humanoid robots with the ability to autonomously perform similar whole-body
manipulation tasks. In this context, the infinite possibilities for where and
how contact can occur on the robot and object surfaces hinder the scalability
of existing planning methods, which predominantly rely on discrete sampling.
Given the continuous nature of contact surfaces, gradient-based optimization
offers a more suitable approach for finding solutions. However, a key remaining
challenge is the lack of an efficient representation of robot surfaces. In this
work, we propose (i) a representation of robot and object surfaces that enables
closed-form computation of proximity points, and (ii) a cost design that
effectively guides whole-body manipulation planning. Our experiments
demonstrate that the proposed framework can solve problems unaddressed by
existing methods, and achieves a 77% improvement in planning time over the
state of the art. We also validate the suitability of our approach on real
hardware through the whole-body manipulation of boxes by a humanoid robot.

### 4. [BOW: Bayesian Optimization over Windows for Motion Planning in Complex Environments](http://arxiv.org/pdf/2508.13052v1)

Authors: Sourav Raxit, Abdullah Al Redwan Newaz, Paulo Padrao, Jose Fuentes, Leonardo Bobadilla

This paper introduces the BOW Planner, a scalable motion planning algorithm
designed to navigate robots through complex environments using constrained
Bayesian optimization (CBO). Unlike traditional methods, which often struggle
with kinodynamic constraints such as velocity and acceleration limits, the BOW
Planner excels by concentrating on a planning window of reachable velocities
and employing CBO to sample control inputs efficiently. This approach enables
the planner to manage high-dimensional objective functions and stringent safety
constraints with minimal sampling, ensuring rapid and secure trajectory
generation. Theoretical analysis confirms the algorithm's asymptotic
convergence to near-optimal solutions, while extensive evaluations in cluttered
and constrained settings reveal substantial improvements in computation times,
trajectory lengths, and solution times compared to existing techniques.
Successfully deployed across various real-world robotic systems, the BOW
Planner demonstrates its practical significance through exceptional sample
efficiency, safety-aware optimization, and rapid planning capabilities, making
it a valuable tool for advancing robotic applications. The BOW Planner is
released as an open-source package and videos of real-world and simulated
experiments are available at https://bow-web.github.io.

### 5. [Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey](http://arxiv.org/pdf/2508.13073v1)

Authors: Rui Shao, Wei Li, Lingsen Zhang, Renshan Zhang, Zhiyang Liu, Ran Chen, Liqiang Nie

Robotic manipulation, a key frontier in robotics and embodied AI, requires
precise motor control and multimodal understanding, yet traditional rule-based
methods fail to scale or generalize in unstructured, novel environments. In
recent years, Vision-Language-Action (VLA) models, built upon Large
Vision-Language Models (VLMs) pretrained on vast image-text datasets, have
emerged as a transformative paradigm. This survey provides the first
systematic, taxonomy-oriented review of large VLM-based VLA models for robotic
manipulation. We begin by clearly defining large VLM-based VLA models and
delineating two principal architectural paradigms: (1) monolithic models,
encompassing single-system and dual-system designs with differing levels of
integration; and (2) hierarchical models, which explicitly decouple planning
from execution via interpretable intermediate representations. Building on this
foundation, we present an in-depth examination of large VLM-based VLA models:
(1) integration with advanced domains, including reinforcement learning,
training-free optimization, learning from human videos, and world model
integration; (2) synthesis of distinctive characteristics, consolidating
architectural traits, operational strengths, and the datasets and benchmarks
that support their development; (3) identification of promising directions,
including memory mechanisms, 4D perception, efficient adaptation, multi-agent
cooperation, and other emerging capabilities. This survey consolidates recent
advances to resolve inconsistencies in existing taxonomies, mitigate research
fragmentation, and fill a critical gap through the systematic integration of
studies at the intersection of large VLMs and robotic manipulation. We provide
a regularly updated project page to document ongoing progress:
https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation.

### 6. [PROD: Palpative Reconstruction of Deformable Objects through Elastostatic Signed Distance Functions](http://arxiv.org/pdf/2508.12554v1)

Authors: Hamza El-Kebir

We introduce PROD (Palpative Reconstruction of Deformables), a novel method
for reconstructing the shape and mechanical properties of deformable objects
using elastostatic signed distance functions (SDFs). Unlike traditional
approaches that rely on purely geometric or visual data, PROD integrates
palpative interaction -- measured through force-controlled surface probing --
to estimate both the static and dynamic response of soft materials. We model
the deformation of an object as an elastostatic process and derive a governing
Poisson equation for estimating its SDF from a sparse set of pose and force
measurements. By incorporating steady-state elastodynamic assumptions, we show
that the undeformed SDF can be recovered from deformed observations with
provable convergence. Our approach also enables the estimation of material
stiffness by analyzing displacement responses to varying force inputs. We
demonstrate the robustness of PROD in handling pose errors, non-normal force
application, and curvature errors in simulated soft body interactions. These
capabilities make PROD a powerful tool for reconstructing deformable objects in
applications ranging from robotic manipulation to medical imaging and haptic
feedback systems.

### 7. [Temporal and Rotational Calibration for Event-Centric Multi-Sensor Systems](http://arxiv.org/pdf/2508.12564v1)

Authors: Jiayao Mai, Xiuyuan Lu, Kuan Dai, Shaojie Shen, Yi Zhou

Event cameras generate asynchronous signals in response to pixel-level
brightness changes, offering a sensing paradigm with theoretically
microsecond-scale latency that can significantly enhance the performance of
multi-sensor systems. Extrinsic calibration is a critical prerequisite for
effective sensor fusion; however, the configuration that involves event cameras
remains an understudied topic. In this paper, we propose a motion-based
temporal and rotational calibration framework tailored for event-centric
multi-sensor systems, eliminating the need for dedicated calibration targets.
Our method uses as input the rotational motion estimates obtained from event
cameras and other heterogeneous sensors, respectively. Different from
conventional approaches that rely on event-to-frame conversion, our method
efficiently estimates angular velocity from normal flow observations, which are
derived from the spatio-temporal profile of event data. The overall calibration
pipeline adopts a two-step approach: it first initializes the temporal offset
and rotational extrinsics by exploiting kinematic correlations in the spirit of
Canonical Correlation Analysis (CCA), and then refines both temporal and
rotational parameters through a joint non-linear optimization using a
continuous-time parametrization in SO(3). Extensive evaluations on both
publicly available and self-collected datasets validate that the proposed
method achieves calibration accuracy comparable to target-based methods, while
exhibiting superior stability over purely CCA-based methods, and highlighting
its precision, robustness and flexibility. To facilitate future research, our
implementation will be made open-source. Code:
https://github.com/NAIL-HNU/EvMultiCalib.

### 8. [Adaptive Model-Predictive Control of a Soft Continuum Robot Using a Physics-Informed Neural Network Based on Cosserat Rod Theory](http://arxiv.org/pdf/2508.12681v1)

Authors: Johann Licher, Max Bartholdt, Henrik Krauss, Tim-Lukas Habich, Thomas Seel, Moritz Schappler

Dynamic control of soft continuum robots (SCRs) holds great potential for
expanding their applications, but remains a challenging problem due to the high
computational demands of accurate dynamic models. While data-driven approaches
like Koopman-operator-based methods have been proposed, they typically lack
adaptability and cannot capture the full robot shape, limiting their
applicability. This work introduces a real-time-capable nonlinear
model-predictive control (MPC) framework for SCRs based on a domain-decoupled
physics-informed neural network (DD-PINN) with adaptable bending stiffness. The
DD-PINN serves as a surrogate for the dynamic Cosserat rod model with a
speed-up factor of 44000. It is also used within an unscented Kalman filter for
estimating the model states and bending compliance from end-effector position
measurements. We implement a nonlinear evolutionary MPC running at 70 Hz on the
GPU. In simulation, it demonstrates accurate tracking of dynamic trajectories
and setpoint control with end-effector position errors below 3 mm (2.3% of the
actuator's length). In real-world experiments, the controller achieves similar
accuracy and accelerations up to 3.55 m/s2.

### 9. [Deformation of the panoramic sphere into an ellipsoid to induce self-motion in telepresence users](http://arxiv.org/pdf/2508.12925v1)

Authors: Eetu Laukka, Evan G. Center, Timo Ojala, Steven M. LaValle, Matti Pouke

Mobile telepresence robots allow users to feel present and explore remote
environments using technology. Traditionally, these systems are implemented
using a camera onboard a mobile robot that can be controlled. Although
high-immersion technologies, such as 360-degree cameras, can increase
situational awareness and presence, they also introduce significant challenges.
Additional processing and bandwidth requirements often result in latencies of
up to seconds. The current delay with a 360-degree camera streaming over the
internet makes real-time control of these systems difficult. Working with
high-latency systems requires some form of assistance to the users.
  This study presents a novel way to utilize optical flow to create an illusion
of self-motion to the user during the latency period between user sending
motion commands to the robot and seeing the actual motion through the
360-camera stream. We find no significant benefit of using the self-motion
illusion to performance or accuracy of controlling a telepresence robot with a
latency of 500 ms, as measured by the task completion time and collisions into
objects. Some evidence is shown that the method might increase virtual reality
(VR) sickness, as measured by the simulator sickness questionnaire (SSQ). We
conclude that further adjustments are necessary in order to render the method
viable.

### 10. [Insights from Interviews with Teachers and Students on the Use of a Social Robot in Computer Science Class in Sixth Grade](http://arxiv.org/pdf/2508.12946v1)

Authors: Ann-Sophie Schenk, Stefan Schiffer, Heqiu Song

In this paper we report on first insights from interviews with teachers and
students on using social robots in computer science class in sixth grade. Our
focus is on learning about requirements and potential applications. We are
particularly interested in getting both perspectives, the teachers' and the
learners' view on how robots could be used and what features they should or
should not have. Results show that teachers as well as students are very open
to robots in the classroom. However, requirements are partially quite
heterogeneous among the groups. This leads to complex design challenges which
we discuss at the end of this paper.

### Software Engineering

### 1. [XAMT: Cross-Framework API Matching for Testing Deep Learning Libraries](http://arxiv.org/pdf/2508.12546v1)

Authors: Bin Duan, Ruican Dong, Naipeng Dong, Dan Dongseong Kim, Guowei Yang

Deep learning powers critical applications such as autonomous driving,
healthcare, and finance, where the correctness of underlying libraries is
essential. Bugs in widely used deep learning APIs can propagate to downstream
systems, causing serious consequences. While existing fuzzing techniques detect
bugs through intra-framework testing across hardware backends (CPU vs. GPU),
they may miss bugs that manifest identically across backends and thus escape
detection under these strategies. To address this problem, we propose XAMT, a
cross-framework fuzzing method that tests deep learning libraries by matching
and comparing functionally equivalent APIs across different frameworks. XAMT
matches APIs using similarity-based rules based on names, descriptions, and
parameter structures. It then aligns inputs and applies variance-guided
differential testing to detect bugs. We evaluated XAMT on five popular
frameworks, including PyTorch, TensorFlow, Keras, Chainer, and JAX. XAMT
matched 839 APIs and identified 238 matched API groups, and detected 17 bugs,
12 of which have been confirmed. Our results show that XAMT uncovers bugs
undetectable by intra-framework testing, especially those that manifest
consistently across backends. XAMT offers a complementary approach to existing
methods and offers a new perspective on the testing of deep learning libraries.

### 2. [ChangePrism: Visualizing the Essence of Code Changes](http://arxiv.org/pdf/2508.12649v1)

Authors: Lei Chen, Michele Lanza, Shinpei Hayashi

Understanding the changes made by developers when they submit a pull request
and/or perform a commit on a repository is a crucial activity in software
maintenance and evolution. The common way to review changes relies on examining
code diffs, where textual differences between two file versions are highlighted
in red and green to indicate additions and deletions of lines. This can be
cumbersome for developers, making it difficult to obtain a comprehensive
overview of all changes in a commit. Moreover, certain types of code changes
can be particularly significant and may warrant differentiation from standard
modifications to enhance code comprehension. We present a novel visualization
approach supported by a tool named ChangePrism, which provides a way to better
understand code changes. The tool comprises two components: extraction, which
retrieves code changes and relevant information from the git history, and
visualization, which offers both general and detailed views of code changes in
commits. The general view provides an overview of different types of code
changes across commits, while the detailed view displays the exact changes in
the source code for each commit.

### 3. [RUM: Rule+LLM-Based Comprehensive Assessment on Testing Skills](http://arxiv.org/pdf/2508.12922v1)

Authors: Yue Wang, Zhenyu Chen, Yuan Zhao, Chunrong Fang, Ziyuan Wang, Song Huang

Over the past eight years, the META method has served as a multidimensional
testing skill assessment system in the National College Student Contest on
Software Testing, successfully assessing over 100,000 students' testing skills.
However, META is primarily limited to the objective assessment of test scripts,
lacking the ability to automatically assess subjective aspects such as test
case and test report. To address this limitation, this paper proposes RUM, a
comprehensive assessment approach that combines rules and large language models
(LLMs). RUM achieves a comprehensive assessment by rapidly processing objective
indicators through rules while utilizing LLMs for in-depth subjective analysis
of test case documents, test scripts, and test reports. The experimental
results show that compared to traditional manual testing skill assessment, RUM
improves assessment efficiency by 80.77\% and reduces costs by 97.38\%, while
maintaining high accuracy and consistency of assessment. By applying RUM on the
contest on software testing, we find that it not only enhances the efficiency
and scalability of skill assessment in software testing education, but also
provides teachers with more comprehensive and objective evidence for student
ability assessment, facilitating personalized teaching and learning. This study
offers new insights into the assessment of testing skills, which are expected
to promote further development in test process optimization and software
quality assurance.

### 4. [Influencia de fatores organizacionais e sociais na etapa de levantamento de requisitos](http://arxiv.org/pdf/2508.13134v1)

Authors: Glauber da Rocha Balthazar, Marcia Ito

The most critical and fragile stage of a software development project is
requirements gathering. Because of this, Requirements Engineering has been
evolving its techniques to minimize the challenges faced by Requirements
Analysts. However, few studies consider the humanistic relationships and
behaviors of those involved in this stage. This article presents a survey of
some studies conducted at this stage that consider non-technical factors such
as emotions, organizational environment, and social context.

### 5. [Strengthening Programming Comprehension in Large Language Models through Code Generation](http://arxiv.org/pdf/2508.12620v1)

Authors: Xiaoning Ren, Qiang Hu, Wei Ma, Yan Li, Yao Zhang, Lingxiao Jiang, Yinxing Xue

Large language models (LLMs) have recently shown impressive results on
diverse code-related tasks, benefiting from large-scale training and
instruction tuning. However, studies reveal that their grasp of fundamental
programming concepts, such as data flow and control flow, remains shallow,
leading to fragile performance when code requires deeper reasoning. This
limitation restricts the practical adoption of LLMs in real-world software
development. To address this issue, this work introduces a counterfactual code
augmentation framework combined with concept-aware tuning, designed to guide
LLMs toward stronger conceptual understanding. Comprehensive evaluation across
multiple models and benchmarks demonstrates the effectiveness of the proposed
approach.

### 6. [Investigating VR Accessibility Reviews for Users with Disabilities: A Qualitative Analysis](http://arxiv.org/pdf/2508.13051v1)

Authors: Yi Wang, Chetan Arora, Xiao Liu, Thuong Hoang, ZHengxin Zhang, Henry Been Lirn Duh, John Grundy

Accessibility reviews provide valuable insights into both the limitations and
benefits experienced by users with disabilities when using virtual reality (VR)
applications. However, a comprehensive investigation into VR accessibility for
users with disabilities is still lacking. To fill this gap, this study analyzes
user reviews from the Meta and Steam stores of VR apps, focusing on the
reported issues affecting users with disabilities. We applied selection
criteria to 1,367,419 reviews from the top 40, the 20 most popular, and the 40
lowest-rated VR applications on both platforms. In total, 1,076 (0.078%) VR
accessibility reviews referenced various disabilities across 100 VR
applications. These applications were categorized into Action, Sports, Social,
Puzzle, Horror, and Simulation, with Action receiving the highest number of
accessibility related-reviews. We identified 16 different types of disabilities
across six categories. Furthermore, we examined the causes of accessibility
issues as reported by users with disabilities. Overall, VR accessibility
reviews were predominantly under-supported.

### 7. [Exploring Autonomous Agents: A Closer Look at Why They Fail When Completing Tasks](http://arxiv.org/pdf/2508.13143v1)

Authors: Ruofan Lu, Yichen Li, Yintong Huo

Autonomous agent systems powered by Large Language Models (LLMs) have
demonstrated promising capabilities in automating complex tasks. However,
current evaluations largely rely on success rates without systematically
analyzing the interactions, communication mechanisms, and failure causes within
these systems. To bridge this gap, we present a benchmark of 34 representative
programmable tasks designed to rigorously assess autonomous agents. Using this
benchmark, we evaluate three popular open-source agent frameworks combined with
two LLM backbones, observing a task completion rate of approximately 50%.
Through in-depth failure analysis, we develop a three-tier taxonomy of failure
causes aligned with task phases, highlighting planning errors, task execution
issues, and incorrect response generation. Based on these insights, we propose
actionable improvements to enhance agent planning and self-diagnosis
capabilities. Our failure taxonomy, together with mitigation advice, provides
an empirical foundation for developing more robust and effective autonomous
agent systems in the future.

### 8. [Systematic Analysis of MCP Security](http://arxiv.org/pdf/2508.12538v1)

Authors: Yongjian Guo, Puzhuo Liu, Wanlun Ma, Zehang Deng, Xiaogang Zhu, Peng Di, Xi Xiao, Sheng Wen

The Model Context Protocol (MCP) has emerged as a universal standard that
enables AI agents to seamlessly connect with external tools, significantly
enhancing their functionality. However, while MCP brings notable benefits, it
also introduces significant vulnerabilities, such as Tool Poisoning Attacks
(TPA), where hidden malicious instructions exploit the sycophancy of large
language models (LLMs) to manipulate agent behavior. Despite these risks,
current academic research on MCP security remains limited, with most studies
focusing on narrow or qualitative analyses that fail to capture the diversity
of real-world threats. To address this gap, we present the MCP Attack Library
(MCPLIB), which categorizes and implements 31 distinct attack methods under
four key classifications: direct tool injection, indirect tool injection,
malicious user attacks, and LLM inherent attack. We further conduct a
quantitative analysis of the efficacy of each attack. Our experiments reveal
key insights into MCP vulnerabilities, including agents' blind reliance on tool
descriptions, sensitivity to file-based attacks, chain attacks exploiting
shared context, and difficulty distinguishing external data from executable
commands. These insights, validated through attack experiments, underscore the
urgency for robust defense strategies and informed MCP design. Our
contributions include 1) constructing a comprehensive MCP attack taxonomy, 2)
introducing a unified attack framework MCPLIB, and 3) conducting empirical
vulnerability analysis to enhance MCP security mechanisms. This work provides a
foundational framework, supporting the secure evolution of MCP ecosystems.

### 9. [OS-R1: Agentic Operating System Kernel Tuning with Reinforcement Learning](http://arxiv.org/pdf/2508.12551v1)

Authors: Hongyu Lin, Yuchen Li, Haoran Luo, Kaichun Yao, Libo Zhang, Mingjie Xing, Yanjun Wu

Linux kernel tuning is essential for optimizing operating system (OS)
performance. However, existing methods often face challenges in terms of
efficiency, scalability, and generalization. This paper introduces OS-R1, an
agentic Linux kernel tuning framework powered by rule-based reinforcement
learning (RL). By abstracting the kernel configuration space as an RL
environment, OS-R1 facilitates efficient exploration by large language models
(LLMs) and ensures accurate configuration modifications. Additionally, custom
reward functions are designed to enhance reasoning standardization,
configuration modification accuracy, and system performance awareness of the
LLMs. Furthermore, we propose a two-phase training process that accelerates
convergence and minimizes retraining across diverse tuning scenarios.
Experimental results show that OS-R1 significantly outperforms existing
baseline methods, achieving up to 5.6% performance improvement over heuristic
tuning and maintaining high data efficiency. Notably, OS-R1 is adaptable across
various real-world applications, demonstrating its potential for practical
deployment in diverse environments. Our dataset and code are publicly available
at https://github.com/LHY-24/OS-R1.

### Social and Information Networks

### 1. [Insight Rumors: A Novel Textual Rumor Locating and Marking Model Leveraging Att_BiMamba2 Network](http://arxiv.org/pdf/2508.12574v1)

Authors: Bin Ma, Yifei Zhang, Yongjin Xian, Qi Li, Linna Zhou, Gongxun Miao

With the development of social media networks, rumor detection models have
attracted more and more attention. Whereas, these models primarily focus on
classifying contexts as rumors or not, lacking the capability to locate and
mark specific rumor content. To address this limitation, this paper proposes a
novel rumor detection model named Insight Rumors to locate and mark rumor
content within textual data. Specifically, we propose the Bidirectional Mamba2
Network with Dot-Product Attention (Att_BiMamba2), a network that constructs a
bidirectional Mamba2 model and applies dot-product attention to weight and
combine the outputs from both directions, thereby enhancing the representation
of high-dimensional rumor features. Simultaneously, a Rumor Locating and
Marking module is designed to locate and mark rumors. The module constructs a
skip-connection network to project high-dimensional rumor features onto
low-dimensional label features. Moreover, Conditional Random Fields (CRF) is
employed to impose strong constraints on the output label features, ensuring
accurate rumor content location. Additionally, a labeled dataset for rumor
locating and marking is constructed, with the effectiveness of the proposed
model is evaluated through comprehensive experiments. Extensive experiments
indicate that the proposed scheme not only detects rumors accurately but also
locates and marks them in context precisely, outperforming state-of-the-art
schemes that can only discriminate rumors roughly.

### 2. [Influence Prediction in Collaboration Networks: An Empirical Study on arXiv](http://arxiv.org/pdf/2508.13029v1)

Authors: Marina Lin, Laura P. Schaposnik, Raina Wu

This paper provides an empirical study of the Social Sphere Model for
influence prediction, previously introduced by the authors, combining link
prediction with top-k centrality-based selection. We apply the model to the
temporal arXiv General Relativity and Quantum Cosmology collaboration network,
evaluating its performance under varying edge sampling rates and prediction
horizons to reflect different levels of initial data completeness and network
evolution. Accuracy is assessed using mean squared error in both link
prediction and influence maximization tasks. The results show that the model
effectively identifies latent influencers, i.e., nodes that are not initially
central but later influential, and performs best with denser initial graphs.
Among the similarity measures tested, the newly introduced RA-2 metric
consistently yields the lowest prediction errors. These findings support the
practical applicability of the model to predict real-world influence in
evolving networks.

### 3. [Unfolded Laplacian Spectral Embedding: A Theoretically Grounded Approach to Dynamic Network Representation](http://arxiv.org/pdf/2508.12674v1)

Authors: Haruka Ezoe, Hiroki Matsumoto, Ryohei Hisano

Dynamic relational structures play a central role in many AI tasks, but their
evolving nature presents challenges for consistent and interpretable
representation. A common approach is to learn time-varying node embeddings,
whose effectiveness depends on satisfying key stability properties. In this
paper, we propose Unfolded Laplacian Spectral Embedding, a new method that
extends the Unfolded Adjacency Spectral Embedding framework to normalized
Laplacians while preserving both cross-sectional and longitudinal stability. We
provide formal proof that our method satisfies these stability conditions. In
addition, as a bonus of using the Laplacian matrix, we establish a new
Cheeger-style inequality that connects the embeddings to the conductance of the
underlying dynamic graphs. Empirical evaluations on synthetic and real-world
datasets support our theoretical findings and demonstrate the strong
performance of our method. These results establish a principled and stable
framework for dynamic network representation grounded in spectral graph theory.

### Systems and Control

### 1. [DCT-MARL: A Dynamic Communication Topology-Based MARL Algorithm for Connected Vehicle Platoon Control](http://arxiv.org/pdf/2508.12633v1)

Authors: Yaqi Xu, Yan Shi, Jin Tian, Fanzeng Xia, Shanzhi Chen, Yuming Ge

With the rapid advancement of vehicular communication and autonomous driving
technologies, connected vehicle platoon has emerged as a promising approach to
improve traffic efficiency and driving safety. Reliable Vehicle-to-Vehicle
(V2V) communication is critical to achieving efficient cooperative control.
However, in real-world traffic environments, V2V links may suffer from
time-varying delay and packet loss, leading to degraded control performance and
even safety risks. To mitigate the adverse effects of non-ideal communication,
this paper proposes a Dynamic Communication Topology based Multi-Agent
Reinforcement Learning (DCT-MARL) algorithm for robust cooperative platoon
control. Specifically, the state space is augmented with historical control
action and delay to enhance robustness against communication delay. To mitigate
the impact of packet loss, a multi-key gated communication mechanism is
introduced, which dynamically adjusts the communication topology based on the
correlation between agents and their current communication status.Simulation
results demonstrate that the proposed DCT-MARL significantly outperforms
state-of-the-art methods in terms of string stability and driving comfort,
validating its superior robustness and effectiveness.

### 2. [Stability Analysis of the Newton-Raphson Controller for a Class of Differentially Flat Systems](http://arxiv.org/pdf/2508.12694v1)

Authors: Kaicheng Niu, Yorai Wardi, Chaouki T. Abdallah

The Newton-Raphson Controller, established on the output prediction and the
Newton-Raphson algorithm, is shown to be effective in a variety of control
applications. Although the stability condition of the controller for linear
systems has already been established, such condition for nonlinear systems
remains unexplored. In this paper, we study the stability of the Newton-Raphson
controller for a class of differentially flat nonlinear systems in the context
of output regulation and tracking control. For output regulation, we prove that
the controlled system is stable within a neighborhood of the origin if the
corresponding flat system and output predictor satisfy a verifiable stability
criterion. A semi-quantitative analysis is conducted to determine the measure
of the domain of attraction. For tracking control, we prove that the controller
is capable of driving the outputs to the external reference signals using a
specific selection of controller parameters. Simulation results show that the
controller achieves regulation and tracking respectively on the inverted
pendulum and the kinematic bicycle, suggesting a potential in future control
applications.

### 3. [Deadline-Aware Bandwidth Allocation for Semantic Generative Communication with Diffusion Models](http://arxiv.org/pdf/2508.12701v1)

Authors: Jinhyuk Choi, Jihong Park, Seungeun Oh, Seong-Lyun Kim

The importance of Radio Access Network (RAN) in support Artificial
Intelligence (AI) application services has grown significantly, underscoring
the need for an integrated approach that considers not only network efficiency
but also AI performance. In this paper we focus on a semantic generative
communication (SGC) framework for image inpainting application. Specifically,
the transmitter sends semantic information, i.e., semantic masks and textual
descriptions, while the receiver utilizes a conditional diffusion model on a
base image, using them as conditioning data to produce the intended image. In
this framework, we propose a bandwidth allocation scheme designed to maximize
bandwidth efficiency while ensuring generation performance. This approach is
based on our finding of a Semantic Deadline--the minimum time that conditioning
data is required to be injected to meet a given performance threshold--within
the multi-modal SGC framework. Given this observation, the proposed scheme
allocates limited bandwidth so that each semantic information can be
transmitted within the corresponding semantic deadline. Experimental results
corroborate that the proposed bandwidth allocation scheme achieves higher
generation performance in terms of PSNR for a given bandwidth compared to
traditional schemes that do not account for semantic deadlines.

### 4. [PFD or PDF: Rethinking the Probability of Failure in Mitigation Safety Functions](http://arxiv.org/pdf/2508.12814v1)

Authors: Hamid Jahanian

SIL (Safety Integrity Level) allocation plays a crucial role in defining the
design requirements for Safety Functions (SFs) within high-risk industries. SIL
is typically determined based on the estimated Probability of Failure on Demand
(PFD), which must remain within permissible limits to manage risk effectively.
Extensive research has been conducted on determining target PFD and SIL, with a
stronger emphasis on preventive SFs than on mitigation SFs. In this paper, we
address a rather conceptual issue: we argue that PFD is not an appropriate
reliability measure for mitigation SFs to begin with, and we propose an
alternative approach that leverages the Probability Density Function (PDF) and
the expected degree of failure as key metrics. The principles underlying this
approach are explained and supported by detailed mathematical formulations.
Furthermore, the practical application of this new methodology is illustrated
through case studies.

### 5. [Grid Edge Intelligence-Assisted Model Predictive Framework for Black Start of Distribution Systems with Inverter-Based Resources](http://arxiv.org/pdf/2508.12937v1)

Authors: Junyuan Zheng, Salish Maharjan, Zhaoyu Wang

The growing proliferation of distributed energy resources (DERs) is
significantly enhancing the resilience and reliability of distribution systems.
However, a substantial portion of behind-the-meter (BTM) DERs is often
overlooked during black start (BS) and restoration processes. Existing BS
strategies that utilize grid-forming (GFM) battery energy storage systems
(BESS) frequently ignore critical frequency security and synchronization
constraints. To address these limitations, this paper proposes a predictive
framework for bottom-up BS that leverages the flexibility of BTM DERs through
Grid Edge Intelligence (GEI). A predictive model is developed for GEI to
estimate multi-period flexibility ranges and track dispatch signals from the
utility. A frequency-constrained BS strategy is then introduced, explicitly
incorporating constraints on frequency nadir, rate-of-change-of-frequency
(RoCoF), and quasi-steady-state (QSS) frequency. The framework also includes
synchronizing switches to enable faster and more secure load restoration.
Notably, it requires GEI devices to communicate only their flexibility ranges
and the utility to send dispatch signals without exchanging detailed asset
information. The proposed framework is validated using a modified IEEE 123-bus
test system, and the impact of GEI is demonstrated by comparing results across
various GEI penetration scenarios.

### 6. [Exploiting Convexity of Neural Networks in Dynamic Operating Envelope Optimization for Distributed Energy Resources](http://arxiv.org/pdf/2508.13090v1)

Authors: Hongyi Li, Liming Liu, Yunyi Li, Zhaoyu Wang

The increasing penetration of distributed energy resources (DERs) brings
opportunities and challenges to the operation of distribution systems. To
ensure network integrity, dynamic operating envelopes (DOEs) are issued by
utilities to DERs as their time-varying export/import power limits. Due to the
non-convex nature of power flow equations, the optimization of DOEs faces a
dilemma of solution accuracy and computation efficiency. To bridge this gap, in
this paper, we facilitate DOE optimization by exploiting the convexity of input
convex neural networks (ICNNs). A DOE optimization model is first presented,
comprehensively considering multiple operational constraints. We propose a
constraint embedding method that allows us to replace the non-convex power flow
constraints with trained ICNN models and convexify the problem. To further
speed up DOE optimization, we propose a linear relaxation of the ICNN-based DOE
optimization problem, for which the tightness is theoretically proven. The
effectiveness of the proposed method is validated with numerical case studies.
Results show that the proposed ICNN-based method outperforms other benchmark
methods in optimizing DOEs in terms of both solution quality and solution time.

### 7. [BUILDA: A Thermal Building Data Generation Framework for Transfer Learning](http://arxiv.org/pdf/2508.12703v1)

Authors: Thomas Krug, Fabian Raisch, Dominik Aimer, Markus Wirnsberger, Ferdinand Sigg, Benjamin Schäfer, Benjamin Tischler

Transfer learning (TL) can improve data-driven modeling of building thermal
dynamics. Therefore, many new TL research areas emerge in the field, such as
selecting the right source model for TL. However, these research directions
require massive amounts of thermal building data which is lacking presently.
Neither public datasets nor existing data generators meet the needs of TL
research in terms of data quality and quantity. Moreover, existing data
generation approaches typically require expert knowledge in building
simulation. We present BuilDa, a thermal building data generation framework for
producing synthetic data of adequate quality and quantity for TL research. The
framework does not require profound building simulation knowledge to generate
large volumes of data. BuilDa uses a single-zone Modelica model that is
exported as a Functional Mock-up Unit (FMU) and simulated in Python. We
demonstrate BuilDa by generating data and utilizing it for pretraining and
fine-tuning TL models.

### 8. [On the Gaussian Limit of the Output of IIR Filters](http://arxiv.org/pdf/2508.12705v1)

Authors: Yashaswini Murthy, Bassam Bamieh, R. Srikant

We study the asymptotic distribution of the output of a stable Linear
Time-Invariant (LTI) system driven by a non-Gaussian stochastic input.
Motivated by longstanding heuristics in the stochastic describing function
method, we rigorously characterize when the output process becomes
approximately Gaussian, even when the input is not. Using the Wasserstein-1
distance as a quantitative measure of non-Gaussianity, we derive upper bounds
on the distance between the appropriately scaled output and a standard normal
distribution. These bounds are obtained via Stein's method and depend
explicitly on the system's impulse response and the dependence structure of the
input process. We show that when the dominant pole of the system approaches the
edge of stability and the input satisfies one of the following conditions: (i)
independence, (ii) positive correlation with a real and positive dominant pole,
or (iii) sufficient correlation decay, the output converges to a standard
normal distribution at rate $O(1/\sqrt{t})$. We also present counterexamples
where convergence fails, thereby motivating the stated assumptions. Our results
provide a rigorous foundation for the widespread observation that outputs of
low-pass LTI systems tend to be approximately Gaussian.

### 9. [MCTR: Midpoint Corrected Triangulation for Autonomous Racing via Digital Twin Simulation in CARLA](http://arxiv.org/pdf/2508.12729v1)

Authors: Junhao Ye, Cheng Hu, Yiqin Wang, Weizhan Huang, Nicolas Baumann, Jie He, Meixun Qu, Lei Xie, Hongye Su

In autonomous racing, reactive controllers eliminate the computational burden
of the full See-Think-Act autonomy stack by directly mapping sensor inputs to
control actions. This bypasses the need for explicit localization and
trajectory planning. A widely adopted baseline in this category is the
Follow-The-Gap method, which performs trajectory planning using LiDAR data.
Building on FTG, the Delaunay Triangulation-based Racing algorithm introduces
further enhancements. However, DTR's use of circumcircles for trajectory
generation often results in insufficiently smooth paths, ultimately degrading
performance. Additionally, the commonly used F1TENTH-simulator for autonomous
racing competitions lacks support for 3D LiDAR perception, limiting its
effectiveness in realistic testing. To address these challenges, this work
proposes the MCTR algorithm. MCTR improves trajectory smoothness through the
use of Curvature Corrected Moving Average and implements a digital twin system
within the CARLA simulator to validate the algorithm's robustness under 3D
LiDAR perception. The proposed algorithm has been thoroughly validated through
both simulation and real-world vehicle experiments.

### 10. [A Hierarchical Surrogate Model for Efficient Multi-Task Parameter Learning in Closed-Loop Contro](http://arxiv.org/pdf/2508.12738v1)

Authors: Sebastian Hirt, Lukas Theiner, Maik Pfefferkorn, Rolf Findeisen

Many control problems require repeated tuning and adaptation of controllers
across distinct closed-loop tasks, where data efficiency and adaptability are
critical. We propose a hierarchical Bayesian optimization (BO) framework that
is tailored to efficient controller parameter learning in sequential
decision-making and control scenarios for distinct tasks. Instead of treating
the closed-loop cost as a black-box, our method exploits structural knowledge
of the underlying problem, consisting of a dynamical system, a control law, and
an associated closed-loop cost function. We construct a hierarchical surrogate
model using Gaussian processes that capture the closed-loop state evolution
under different parameterizations, while the task-specific weighting and
accumulation into the closed-loop cost are computed exactly via known
closed-form expressions. This allows knowledge transfer and enhanced data
efficiency between different closed-loop tasks. The proposed framework retains
sublinear regret guarantees on par with standard black-box BO, while enabling
multi-task or transfer learning. Simulation experiments with model predictive
control demonstrate substantial benefits in both sample efficiency and
adaptability when compared to purely black-box BO approaches.

### Machine Learning (Statistics Category)

### 1. [Constrained Centroid Clustering: A Novel Approach for Compact and Structured Partitioning](http://arxiv.org/pdf/2508.12758v1)

Authors: Sowmini Devi Veeramachaneni, Ramamurthy Garimella

This paper presents Constrained Centroid Clustering (CCC), a method that
extends classical centroid-based clustering by enforcing a constraint on the
maximum distance between the cluster center and the farthest point in the
cluster. Using a Lagrangian formulation, we derive a closed-form solution that
maintains interpretability while controlling cluster spread. To evaluate CCC,
we conduct experiments on synthetic circular data with radial symmetry and
uniform angular distribution. Using ring-wise, sector-wise, and joint entropy
as evaluation metrics, we show that CCC achieves more compact clusters by
reducing radial spread while preserving angular structure, outperforming
standard methods such as K-means and GMM. The proposed approach is suitable for
applications requiring structured clustering with spread control, including
sensor networks, collaborative robotics, and interpretable pattern analysis.

### 2. [Optimal Condition for Initialization Variance in Deep Neural Networks: An SGD Dynamics Perspective](http://arxiv.org/pdf/2508.12834v1)

Authors: Hiroshi Horii, Sothea Has

Stochastic gradient descent (SGD), one of the most fundamental optimization
algorithms in machine learning (ML), can be recast through a continuous-time
approximation as a Fokker-Planck equation for Langevin dynamics, a viewpoint
that has motivated many theoretical studies. Within this framework, we study
the relationship between the quasi-stationary distribution derived from this
equation and the initial distribution through the Kullback-Leibler (KL)
divergence. As the quasi-steady-state distribution depends on the expected cost
function, the KL divergence eventually reveals the connection between the
expected cost function and the initialization distribution. By applying this to
deep neural network models (DNNs), we can express the bounds of the expected
loss function explicitly in terms of the initialization parameters. Then, by
minimizing this bound, we obtain an optimal condition of the initialization
variance in the Gaussian case. This result provides a concrete mathematical
criterion, rather than a heuristic approach, to select the scale of weight
initialization in DNNs. In addition, we experimentally confirm our theoretical
results by using the classical SGD to train fully connected neural networks on
the MNIST and Fashion-MNIST datasets. The result shows that if the variance of
the initialization distribution satisfies our theoretical optimal condition,
then the corresponding DNN model always achieves lower final training loss and
higher test accuracy than the conventional He-normal initialization. Our work
thus supplies a mathematically grounded indicator that guides the choice of
initialization variance and clarifies its physical meaning of the dynamics of
parameters in DNNs.

### 3. [The path to a goal: Understanding soccer possessions via path signatures](http://arxiv.org/pdf/2508.12930v1)

Authors: David Hirnschall, Robert Bajons

We present a novel framework for predicting next actions in soccer
possessions by leveraging path signatures to encode their complex
spatio-temporal structure. Unlike existing approaches, we do not rely on fixed
historical windows and handcrafted features, but rather encode the entire
recent possession, thereby avoiding the inclusion of potentially irrelevant or
misleading historical information. Path signatures naturally capture the order
and interaction of events, providing a mathematically grounded feature encoding
for variable-length time series of irregular sampling frequencies without the
necessity for manual feature engineering. Our proposed approach outperforms a
transformer-based benchmark across various loss metrics and considerably
reduces computational cost. Building on these results, we introduce a new
possession evaluation metric based on well-established frameworks in soccer
analytics, incorporating both predicted action type probabilities and action
location. Our metric shows greater reliability than existing metrics in
domain-specific comparisons. Finally, we validate our approach through a
detailed analysis of the 2017/18 Premier League season and discuss further
applications and future extensions.

### 4. [Simulation-Based Inference: A Practical Guide](http://arxiv.org/pdf/2508.12939v1)

Authors: Michael Deistler, Jan Boelts, Peter Steinbach, Guy Moss, Thomas Moreau, Manuel Gloeckler, Pedro L. C. Rodrigues, Julia Linhart, Janne K. Lappalainen, Benjamin Kurt Miller, Pedro J. Gonçalves, Jan-Matthis Lueckmann, Cornelius Schröder, Jakob H. Macke

A central challenge in many areas of science and engineering is to identify
model parameters that are consistent with prior knowledge and empirical data.
Bayesian inference offers a principled framework for this task, but can be
computationally prohibitive when models are defined by stochastic simulators.
Simulation-based Inference (SBI) is a suite of methods developed to overcome
this limitation, which has enabled scientific discoveries in fields such as
particle physics, astrophysics, and neuroscience. The core idea of SBI is to
train neural networks on data generated by a simulator, without requiring
access to likelihood evaluations. Once trained, inference is amortized: The
neural network can rapidly perform Bayesian inference on empirical observations
without requiring additional training or simulations. In this tutorial, we
provide a practical guide for practitioners aiming to apply SBI methods. We
outline a structured SBI workflow and offer practical guidelines and diagnostic
tools for every stage of the process -- from setting up the simulator and
prior, choosing and training inference networks, to performing inference and
validating the results. We illustrate these steps through examples from
astrophysics, psychophysics, and neuroscience. This tutorial empowers
researchers to apply state-of-the-art SBI methods, facilitating efficient
parameter inference for scientific discovery.

### 5. [Fairness-Aware Multi-view Evidential Learning with Adaptive Prior](http://arxiv.org/pdf/2508.12997v1)

Authors: Haishun Chen, Cai Xu, Jinlong Yu, Yilin Zhang, Ziyu Guan, Wei Zhao

Multi-view evidential learning aims to integrate information from multiple
views to improve prediction performance and provide trustworthy uncertainty
esitimation. Most previous methods assume that view-specific evidence learning
is naturally reliable. However, in practice, the evidence learning process
tends to be biased. Through empirical analysis on real-world data, we reveal
that samples tend to be assigned more evidence to support data-rich classes,
thereby leading to unreliable uncertainty estimation in predictions. This
motivates us to delve into a new Biased Evidential Multi-view Learning (BEML)
problem. To this end, we propose Fairness-Aware Multi-view Evidential Learning
(FAML). FAML first introduces an adaptive prior based on training trajectory,
which acts as a regularization strategy to flexibly calibrate the biased
evidence learning process. Furthermore, we explicitly incorporate a fairness
constraint based on class-wise evidence variance to promote balanced evidence
allocation. In the multi-view fusion stage, we propose an opinion alignment
mechanism to mitigate view-specific bias across views, thereby encouraging the
integration of consistent and mutually supportive evidence. Extensive
experiments on five real-world multi-view datasets demonstrate that FAML
achieves more balanced evidence allocation and improves both prediction
performance and the reliability of uncertainty estimation compared to
state-of-the-art methods.

### 6. [Data-driven particle dynamics: Structure-preserving coarse-graining for emergent behavior in non-equilibrium systems](http://arxiv.org/pdf/2508.12569v1)

Authors: Quercus Hernandez, Max Win, Thomas C. O'Connor, Paulo E. Arratia, Nathaniel Trask

Multiscale systems are ubiquitous in science and technology, but are
notoriously challenging to simulate as short spatiotemporal scales must be
appropriately linked to emergent bulk physics. When expensive high-dimensional
dynamical systems are coarse-grained into low-dimensional models, the entropic
loss of information leads to emergent physics which are dissipative,
history-dependent, and stochastic. To machine learn coarse-grained dynamics
from time-series observations of particle trajectories, we propose a framework
using the metriplectic bracket formalism that preserves these properties by
construction; most notably, the framework guarantees discrete notions of the
first and second laws of thermodynamics, conservation of momentum, and a
discrete fluctuation-dissipation balance crucial for capturing non-equilibrium
statistics. We introduce the mathematical framework abstractly before
specializing to a particle discretization. As labels are generally unavailable
for entropic state variables, we introduce a novel self-supervised learning
strategy to identify emergent structural variables. We validate the method on
benchmark systems and demonstrate its utility on two challenging examples: (1)
coarse-graining star polymers at challenging levels of coarse-graining while
preserving non-equilibrium statistics, and (2) learning models from high-speed
video of colloidal suspensions that capture coupling between local
rearrangement events and emergent stochastic dynamics. We provide open-source
implementations in both PyTorch and LAMMPS, enabling large-scale inference and
extensibility to diverse particle-based systems.

### 7. [Unfolded Laplacian Spectral Embedding: A Theoretically Grounded Approach to Dynamic Network Representation](http://arxiv.org/pdf/2508.12674v1)

Authors: Haruka Ezoe, Hiroki Matsumoto, Ryohei Hisano

Dynamic relational structures play a central role in many AI tasks, but their
evolving nature presents challenges for consistent and interpretable
representation. A common approach is to learn time-varying node embeddings,
whose effectiveness depends on satisfying key stability properties. In this
paper, we propose Unfolded Laplacian Spectral Embedding, a new method that
extends the Unfolded Adjacency Spectral Embedding framework to normalized
Laplacians while preserving both cross-sectional and longitudinal stability. We
provide formal proof that our method satisfies these stability conditions. In
addition, as a bonus of using the Laplacian matrix, we establish a new
Cheeger-style inequality that connects the embeddings to the conductance of the
underlying dynamic graphs. Empirical evaluations on synthetic and real-world
datasets support our theoretical findings and demonstrate the strong
performance of our method. These results establish a principled and stable
framework for dynamic network representation grounded in spectral graph theory.

### 8. [Randomized PCA Forest for Outlier Detection](http://arxiv.org/pdf/2508.12776v1)

Authors: Muhammad Rajabinasab, Farhad Pakdaman, Moncef Gabbouj, Peter Schneider-Kamp, Arthur Zimek

We propose a novel unsupervised outlier detection method based on Randomized
Principal Component Analysis (PCA). Inspired by the performance of Randomized
PCA (RPCA) Forest in approximate K-Nearest Neighbor (KNN) search, we develop a
novel unsupervised outlier detection method that utilizes RPCA Forest for
outlier detection. Experimental results showcase the superiority of the
proposed approach compared to the classical and state-of-the-art methods in
performing the outlier detection task on several datasets while performing
competitively on the rest. The extensive analysis of the proposed method
reflects it high generalization power and its computational efficiency,
highlighting it as a good choice for unsupervised outlier detection.

### 9. [Bridging Human and LLM Judgments: Understanding and Narrowing the Gap](http://arxiv.org/pdf/2508.12792v1)

Authors: Felipe Maia Polo, Xinhe Wang, Mikhail Yurochkin, Gongjun Xu, Moulinath Banerjee, Yuekai Sun

Large language models are increasingly used as judges (LLM-as-a-judge) to
evaluate model outputs at scale, but their assessments often diverge
systematically from human judgments. We present Bridge, a unified statistical
framework that explicitly bridges human and LLM evaluations under both absolute
scoring and pairwise comparison paradigms. Bridge posits a latent human
preference score for each prompt-response pair and models LLM deviations as
linear transformations of covariates that capture sources of discrepancies.
This offers a simple and principled framework for refining LLM ratings and
characterizing systematic discrepancies between humans and LLMs. We provide an
efficient fitting algorithm with asymptotic guarantees for statistical
inference. Using six LLM judges and two benchmarks (BigGen Bench and Chatbot
Arena), Bridge achieves higher agreement with human ratings (accuracy,
calibration, and KL divergence) and exposes systematic human-LLM gaps.

### 10. [Shapley Values: Paired-Sampling Approximations](http://arxiv.org/pdf/2508.12947v1)

Authors: Michael Mayer, Mario V. Wüthrich

Originally introduced in cooperative game theory, Shapley values have become
a very popular tool to explain machine learning predictions. Based on Shapley's
fairness axioms, every input (feature component) gets a credit how it
contributes to an output (prediction). These credits are then used to explain
the prediction. The only limitation in computing the Shapley values (credits)
for many different predictions is of computational nature. There are two
popular sampling approximations, sampling KernelSHAP and sampling
PermutationSHAP. Our first novel contributions are asymptotic normality results
for these sampling approximations. Next, we show that the paired-sampling
approaches provide exact results in case of interactions being of maximal order
two. Furthermore, the paired-sampling PermutationSHAP possesses the additive
recovery property, whereas its kernel counterpart does not.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-08-19 PST.

### 1. [Advancements in cyberthreat intelligence through resource exhaustion attack detection using hybrid deep learning with heuristic search algorithms](https://www.nature.com/articles/s41598-025-13305-8)

Authors: S. Jayanthi et al.

### 2. [Optimizing mobile cognitive assessment reduces administration time while maintaining screening accuracy in older adults](https://www.nature.com/articles/s41598-025-14130-9)

Authors: Hyun Jeong Ko et al.

### 3. [University english teaching evaluation using artificial intelligence and data mining technology](https://www.nature.com/articles/s41598-025-16498-0)

Authors: Qiuyang Huang et al.

### 4. [An adaptive coverage method for dynamic wireless sensor network deployment using deep reinforcement learning](https://www.nature.com/articles/s41598-025-16031-3)

Authors: Peng Zhou et al.

### 5. [Improved leukocyte classification in bone marrow cytology using convolutional neural network with contrast enhancement](https://www.nature.com/articles/s41598-025-12207-z)

Authors: Shahid Mehmood et al.

### 6. [Enhancing confidentiality and access control in electronic health record systems using a hybrid hashing blockchain framework](https://www.nature.com/articles/s41598-025-13831-5)

Authors: D. Gowtham Chakravarthy et al.

### 7. [Analyzing the vulnerabilities in Split Federated Learning: assessing the robustness against data poisoning attacks](https://www.nature.com/articles/s41598-025-15993-8)

Authors: Aysha-Thahsin Zahir-Ismail et al.

### 8. [The uncanny valley effect and immune activation in virtual reality](https://www.nature.com/articles/s41598-025-15579-4)

Authors: Esther K. Diekhof et al.

### 9. [A fully automatic knee subregion segmentation network based on tissue segmentation and anatomical geometry](https://www.nature.com/articles/s41598-025-16241-9)

Authors: Shaolong Chen et al.

### 10. [A Data Centric HitL Framework for Conducting aSsystematic Error Analysis of NLP Datasets using Explainable AI](https://www.nature.com/articles/s41598-025-13452-y)

Authors: Ahmed El-Sayed et al.

### 11. [Optimizing energy and latency in edge computing through a Boltzmann driven Bayesian framework for adaptive resource scheduling](https://www.nature.com/articles/s41598-025-16317-6)

Authors: Dinesh Sahu et al.

### 12. [A network recovery strategy based on boundary nodes and tetrahedral approximation fermat points in three-dimensional wireless sensor networks](https://www.nature.com/articles/s41598-025-15723-0)

Authors: Bin Xu et al.

### 13. [Deep learning with leagues championship algorithm based intrusion detection on cybersecurity driven industrial IoT systems](https://www.nature.com/articles/s41598-025-15464-0)

Authors: Saud S. Alotaibi et al.

### 14. [Viewport prediction with cross modal multiscale transformer for 360° video streaming](https://www.nature.com/articles/s41598-025-16011-7)

Authors: Yangsheng Tian et al.

### 15. [Evaluation of sports teaching quality in universities based on fuzzy decision support system](https://www.nature.com/articles/s41598-025-12710-3)

Authors: Kunjian Han et al.

### 16. [Research on deep learning model for stock prediction by integrating frequency domain and time series features](https://www.nature.com/articles/s41598-025-14872-6)

Authors: Wenjie Sun et al.

