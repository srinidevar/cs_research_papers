# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-08-27 17:00:26.015747 PST.

### Artificial Intelligence

### 1. [eSkinHealth: A Multimodal Dataset for Neglected Tropical Skin Diseases](http://arxiv.org/pdf/2508.18608v1)

Authors: Janet Wang, Xin Hu, Yunbei Zhang, Diabate Almamy, Vagamon Bamba, Konan Amos Sébastien Koffi, Yao Koffi Aubin, Zhengming Ding, Jihun Hamm, Rie R. Yotsu

Skin Neglected Tropical Diseases (NTDs) impose severe health and
socioeconomic burdens in impoverished tropical communities. Yet, advancements
in AI-driven diagnostic support are hindered by data scarcity, particularly for
underrepresented populations and rare manifestations of NTDs. Existing
dermatological datasets often lack the demographic and disease spectrum crucial
for developing reliable recognition models of NTDs. To address this, we
introduce eSkinHealth, a novel dermatological dataset collected on-site in
C\^ote d'Ivoire and Ghana. Specifically, eSkinHealth contains 5,623 images from
1,639 cases and encompasses 47 skin diseases, focusing uniquely on skin NTDs
and rare conditions among West African populations. We further propose an
AI-expert collaboration paradigm to implement foundation language and
segmentation models for efficient generation of multimodal annotations, under
dermatologists' guidance. In addition to patient metadata and diagnosis labels,
eSkinHealth also includes semantic lesion masks, instance-specific visual
captions, and clinical concepts. Overall, our work provides a valuable new
resource and a scalable annotation framework, aiming to catalyze the
development of more equitable, accurate, and interpretable AI tools for global
dermatology.

### 2. [MUA-RL: Multi-turn User-interacting Agent Reinforcement Learning for agentic tool use](http://arxiv.org/pdf/2508.18669v1)

Authors: Weikang Zhao, Xili Wang, Chengdi Ma, Lingbin Kong, Zhaohua Yang, Mingxiang Tuo, Xiaowei Shi, Yitao Zhai, Xunliang Cai

With the recent rapid advancement of Agentic Intelligence, agentic tool use
in LLMs has become increasingly important. During multi-turn interactions
between agents and users, the dynamic, uncertain, and stochastic nature of user
demands poses significant challenges to the agent's tool invocation
capabilities. Agents are no longer expected to simply call tools to deliver a
result; rather, they must iteratively refine their understanding of user needs
through communication while simultaneously invoking tools to resolve user
queries. Existing reinforcement learning (RL) approaches for tool use lack the
integration of genuinely dynamic users during the RL training process. To
bridge this gap, we introduce MUA-RL (Multi-turn User-interacting Agent
Reinforcement Learning for agentic tool use), a novel reinforcement learning
framework that, for the first time in the field of agentic tool use, integrates
LLM-simulated users into the reinforcement learning loop. MUA-RL aims to enable
autonomous learning of models to communicate with users efficiently and use
various tools to solve practical problems in dynamic multi-turn interactions.
Evaluations are done on several multi-turn tool-using benchmarks (see Figure
1). Specifically, MUA-RL-32B achieves 67.3 on TAU2 Retail, 45.4 on TAU2
Airline, 28.3 on TAU2 Telecom, 28.4 on BFCL-V3 Multi Turn, and 82.5 on ACEBench
Agent -- outperforming or matching the performance of larger open-source models
such as DeepSeek-V3-0324 and Qwen3-235B-A22B in non-thinking settings.

### 3. [AppAgent-Pro: A Proactive GUI Agent System for Multidomain Information Integration and User Assistance](http://arxiv.org/pdf/2508.18689v1)

Authors: Yuyang Zhao, Wentao Shi, Fuli Feng, Xiangnan He

Large language model (LLM)-based agents have demonstrated remarkable
capabilities in addressing complex tasks, thereby enabling more advanced
information retrieval and supporting deeper, more sophisticated human
information-seeking behaviors. However, most existing agents operate in a
purely reactive manner, responding passively to user instructions, which
significantly constrains their effectiveness and efficiency as general-purpose
platforms for information acquisition. To overcome this limitation, this paper
proposes AppAgent-Pro, a proactive GUI agent system that actively integrates
multi-domain information based on user instructions. This approach enables the
system to proactively anticipate users' underlying needs and conduct in-depth
multi-domain information mining, thereby facilitating the acquisition of more
comprehensive and intelligent information. AppAgent-Pro has the potential to
fundamentally redefine information acquisition in daily life, leading to a
profound impact on human society. Our code is available at:
https://github.com/LaoKuiZe/AppAgent-Pro. Our code is available at:
https://github.com/LaoKuiZe/AppAgent-Pro. The demonstration video could be
found at:
https://www.dropbox.com/scl/fi/hvzqo5vnusg66srydzixo/AppAgent-Pro-demo-video.mp4?rlkey=o2nlfqgq6ihl125mcqg7bpgqu&st=d29vrzii&dl=0.

### 4. [VistaWise: Building Cost-Effective Agent with Cross-Modal Knowledge Graph for Minecraft](http://arxiv.org/pdf/2508.18722v1)

Authors: Honghao Fu, Junlong Ren, Qi Chai, Deheng Ye, Yujun Cai, Hao Wang

Large language models (LLMs) have shown significant promise in embodied
decision-making tasks within virtual open-world environments. Nonetheless,
their performance is hindered by the absence of domain-specific knowledge.
Methods that finetune on large-scale domain-specific data entail prohibitive
development costs. This paper introduces VistaWise, a cost-effective agent
framework that integrates cross-modal domain knowledge and finetunes a
dedicated object detection model for visual analysis. It reduces the
requirement for domain-specific training data from millions of samples to a few
hundred. VistaWise integrates visual information and textual dependencies into
a cross-modal knowledge graph (KG), enabling a comprehensive and accurate
understanding of multimodal environments. We also equip the agent with a
retrieval-based pooling strategy to extract task-related information from the
KG, and a desktop-level skill library to support direct operation of the
Minecraft desktop client via mouse and keyboard inputs. Experimental results
demonstrate that VistaWise achieves state-of-the-art performance across various
open-world tasks, highlighting its effectiveness in reducing development costs
while enhancing agent performance.

### 5. [Reflection-Enhanced Meta-Optimization Integrating TextGrad-style Prompt Optimization with Memory-Driven Self-Evolution](http://arxiv.org/pdf/2508.18749v1)

Authors: Chunlong Wu, Zhibo Qu

Recent advances in prompt optimization, exemplified by methods such as
TextGrad, enable automatic, gradient-like refinement of textual prompts to
enhance the performance of large language models (LLMs) on specific downstream
tasks. However, current approaches are typically stateless and operate
independently across optimization runs, lacking mechanisms to preserve and
leverage historical optimization experience. Furthermore, they are susceptible
to overfitting, often yielding prompt updates that generalize poorly beyond the
immediate task context.
  To address these limitations, we propose Reflection-Enhanced
Meta-Optimization (REMO), a novel framework that integrates (1) a
memory-augmented Reflection Retrieval-Augmented Generation (RAG) module -
structured as a "mistake notebook" and (2) a Self-Adaptive Optimizer,
implemented via an LLM-driven meta-controller that synthesizes epoch-level
reflective insights to iteratively improve system-level prompting strategies.
This architecture enables not only local, fine-grained prompt tuning akin to
TextGrad, but also the systematic accumulation and reuse of cross-run
optimization knowledge, thereby supporting continual improvement over time.
  We instantiate the REMO framework using Qwen3-32B in standard inference mode
- without explicit chain-of-thought prompting - and evaluate its efficacy on
the GSM8K benchmark for mathematical reasoning. Experimental results
demonstrate that, compared to a TextGrad baseline, REMO achieves more stable
and robust generalization, albeit at the cost of increased computational
overhead. We provide a detailed exposition of the algorithmic design, conduct a
qualitative and quantitative analysis of optimization dynamics, and present a
comprehensive ablation study to elucidate the contributions of each component.

### 6. [Dynamic Collaboration of Multi-Language Models based on Minimal Complete Semantic Units](http://arxiv.org/pdf/2508.18763v1)

Authors: Chao Hao, Zezheng Wang, Yanhua Huang, Ruiwen Xu, Wenzhe Niu, Xin Liu, Zitong Yu

This paper investigates the enhancement of reasoning capabilities in language
models through token-level multi-model collaboration. Our approach selects the
optimal tokens from the next token distributions provided by multiple models to
perform autoregressive reasoning. Contrary to the assumption that more models
yield better results, we introduce a distribution distance-based dynamic
selection strategy (DDS) to optimize the multi-model collaboration process. To
address the critical challenge of vocabulary misalignment in multi-model
collaboration, we propose the concept of minimal complete semantic units
(MCSU), which is simple yet enables multiple language models to achieve natural
alignment within the linguistic space. Experimental results across various
benchmarks demonstrate the superiority of our method. The code will be
available at https://github.com/Fanye12/DDS.

### 7. [CausalMACE: Causality Empowered Multi-Agents in Minecraft Cooperative Tasks](http://arxiv.org/pdf/2508.18797v1)

Authors: Qi Chai, Zhang Zheng, Junlong Ren, Deheng Ye, Zichuan Lin, Hao Wang

Minecraft, as an open-world virtual interactive environment, has become a
prominent platform for research on agent decision-making and execution.
Existing works primarily adopt a single Large Language Model (LLM) agent to
complete various in-game tasks. However, for complex tasks requiring lengthy
sequences of actions, single-agent approaches often face challenges related to
inefficiency and limited fault tolerance. Despite these issues, research on
multi-agent collaboration remains scarce. In this paper, we propose CausalMACE,
a holistic causality planning framework designed to enhance multi-agent
systems, in which we incorporate causality to manage dependencies among
subtasks. Technically, our proposed framework introduces two modules: an
overarching task graph for global task planning and a causality-based module
for dependency management, where inherent rules are adopted to perform causal
intervention. Experimental results demonstrate our approach achieves
state-of-the-art performance in multi-agent cooperative tasks of Minecraft.

### 8. [STARec: An Efficient Agent Framework for Recommender Systems via Autonomous Deliberate Reasoning](http://arxiv.org/pdf/2508.18812v1)

Authors: Chenghao Wu, Ruiyang Ren, Junjie Zhang, Ruirui Wang, Zhongrui Ma, Qi Ye, Wayne Xin Zhao

While modern recommender systems are instrumental in navigating information
abundance, they remain fundamentally limited by static user modeling and
reactive decision-making paradigms. Current large language model (LLM)-based
agents inherit these shortcomings through their overreliance on heuristic
pattern matching, yielding recommendations prone to shallow correlation bias,
limited causal inference, and brittleness in sparse-data scenarios. We
introduce STARec, a slow-thinking augmented agent framework that endows
recommender systems with autonomous deliberative reasoning capabilities. Each
user is modeled as an agent with parallel cognitions: fast response for
immediate interactions and slow reasoning that performs chain-of-thought
rationales. To cultivate intrinsic slow thinking, we develop anchored
reinforcement training - a two-stage paradigm combining structured knowledge
distillation from advanced reasoning models with preference-aligned reward
shaping. This hybrid approach scaffolds agents in acquiring foundational
capabilities (preference summarization, rationale generation) while enabling
dynamic policy adaptation through simulated feedback loops. Experiments on
MovieLens 1M and Amazon CDs benchmarks demonstrate that STARec achieves
substantial performance gains compared with state-of-the-art baselines, despite
using only 0.4% of the full training data.

### 9. [Judicial Requirements for Generative AI in Legal Reasoning](http://arxiv.org/pdf/2508.18880v1)

Authors: Eljas Linna, Tuula Linna

Large Language Models (LLMs) are being integrated into professional domains,
yet their limitations in high-stakes fields like law remain poorly understood.
This paper defines the core capabilities that an AI system must possess to
function as a reliable reasoning tool in judicial decision-making. Using the
IRAC (Issue-Rule-Application-Conclusion) model as an analytical framework, the
study focuses on the most challenging phases of legal adjudication: determining
the applicable Rule (R) and performing the Application (A) of that rule to the
facts of a case. From a judicial perspective, the analysis deconstructs legal
reasoning into a series of core requirements, including the ability to select
the correct legal framework across jurisdictions, generate sound arguments
based on the doctrine of legal sources, distinguish ratio decidendi from obiter
dictum in case law, resolve ambiguity arising from general clauses like
"reasonableness", manage conflicting legal provisions, and correctly apply the
burden of proof. The paper then maps various AI enhancement mechanisms, such as
Retrieval-Augmented Generation (RAG), multi-agent systems, and neuro-symbolic
AI, to these requirements, assessing their potential to bridge the gap between
the probabilistic nature of LLMs and the rigorous, choice-driven demands of
legal interpretation. The findings indicate that while these techniques can
address specific challenges, significant challenges remain, particularly in
tasks requiring discretion and transparent, justifiable reasoning. Our paper
concludes that the most effective current role for AI in law is a dual one: as
a high-volume assistant for simple, repetitive cases and as a sophisticated
"sparring partner" for human experts in complex matters.

### 10. [Interactive Evaluation of Large Language Models for Multi-Requirement Software Engineering Tasks](http://arxiv.org/pdf/2508.18905v1)

Authors: Dimitrios Rontogiannis, Maxime Peyrard, Nicolas Baldwin, Martin Josifoski, Robert West, Dimitrios Gunopulos

Standard single-turn, static benchmarks fall short in evaluating the nuanced
capabilities of Large Language Models (LLMs) on complex tasks such as software
engineering. In this work, we propose a novel interactive evaluation framework
that assesses LLMs on multi-requirement programming tasks through structured,
feedback-driven dialogue. Each task is modeled as a requirement dependency
graph, and an ``interviewer'' LLM, aware of the ground-truth solution, provides
minimal, targeted hints to an ``interviewee'' model to help correct errors and
fulfill target constraints. This dynamic protocol enables fine-grained
diagnostic insights into model behavior, uncovering strengths and systematic
weaknesses that static benchmarks fail to measure. We build on DevAI, a
benchmark of 55 curated programming tasks, by adding ground-truth solutions and
evaluating the relevance and utility of interviewer hints through expert
annotation. Our results highlight the importance of dynamic evaluation in
advancing the development of collaborative code-generating agents.

### Hardware Architecture

### 1. [SeDA: Secure and Efficient DNN Accelerators with Hardware/Software Synergy](http://arxiv.org/pdf/2508.18924v1)

Authors: Wei Xuan, Zhongrui Wang, Lang Feng, Ning Lin, Zihao Xuan, Rongliang Fu, Tsung-Yi Ho, Yuzhong Jiao, Luhong Liang

Ensuring the confidentiality and integrity of DNN accelerators is paramount
across various scenarios spanning autonomous driving, healthcare, and finance.
However, current security approaches typically require extensive hardware
resources, and incur significant off-chip memory access overheads. This paper
introduces SeDA, which utilizes 1) a bandwidth-aware encryption mechanism to
improve hardware resource efficiency, 2) optimal block granularity through
intra-layer and inter-layer tiling patterns, and 3) a multi-level integrity
verification mechanism that minimizes, or even eliminates, memory access
overheads. Experimental results show that SeDA decreases performance overhead
by over 12% for both server and edge neural processing units (NPUs), while
ensuring robust scalability.

### 2. [TaiBai: A fully programmable brain-inspired processor with topology-aware efficiency](http://arxiv.org/pdf/2508.18961v1)

Authors: Qianpeng Li, Yu Song, Xin Liu, Wenna Song, Boshi Zhao, Zhichao Wang, Aoxin Chen, Tielin Zhang, Liang Chen

Brain-inspired computing has emerged as a promising paradigm to overcome the
energy-efficiency limitations of conventional intelligent systems by emulating
the brain's partitioned architecture and event-driven sparse computation.
However, existing brain-inspired chips often suffer from rigid network topology
constraints and limited neuronal programmability, hindering their adaptability.
To address these challenges, we present TaiBai, an event-driven, programmable
many-core brain-inspired processor that leverages temporal and spatial spike
sparsity to minimize bandwidth and computational overhead. TaiBai chip contains
three key features: First, a brain-inspired hierarchical topology encoding
scheme is designed to flexibly support arbitrary network architectures while
slashing storage overhead for large-scale networks; Second, a multi-granularity
instruction set enables programmability of brain-like spiking neuron or
synapses with various dynamics and on-chip learning rules; Third, a co-designed
compiler stack optimizes task mapping and resource allocation. After evaluating
across various tasks, such as speech recognition, ECG classification, and
cross-day brain-computer interface decoding, we found spiking neural networks
embedded on the TaiBai chip could achieve more than 200 times higher energy
efficiency than a standard NVIDIA RTX 3090 GPU at a comparable accuracy. These
results demonstrated its high potentiation as a scalable, programmable, and
ultra-efficient solution for both multi-scale brain simulation and
brain-inspired computation.

### 3. [Building an Open CGRA Ecosystem for Agile Innovation](http://arxiv.org/pdf/2508.19090v1)

Authors: Rohan Juneja, Pranav Dangi, Thilini Kaushalya Bandara, Zhaoying Li, Dhananjaya Wijerathne, Li-Shiuan Peh, Tulika Mitra

Modern computing workloads, particularly in AI and edge applications, demand
hardware-software co-design to meet aggressive performance and energy targets.
Such co-design benefits from open and agile platforms that replace closed,
vertically integrated development with modular, community-driven ecosystems.
Coarse-Grained Reconfigurable Architectures (CGRAs), with their unique balance
of flexibility and efficiency are particularly well-suited for this paradigm.
When built on open-source hardware generators and software toolchains, CGRAs
provide a compelling foundation for architectural exploration, cross-layer
optimization, and real-world deployment. In this paper, we will present an open
CGRA ecosystem that we have developed to support agile innovation across the
stack. Our contributions include HyCUBE, a CGRA with a reconfigurable
single-cycle multi-hop interconnect for efficient data movement; PACE, which
embeds a power-efficient HyCUBE within a RISC-V SoC targeting edge computing;
and Morpher, a fully open-source, architecture-adaptive CGRA design framework
that supports design space exploration, compilation, simulation, and
validation. By embracing openness at every layer, we aim to lower barriers to
innovation, enable reproducible research, and demonstrate how CGRAs can anchor
the next wave of agile hardware development. We will conclude with a call for a
unified abstraction layer for CGRAs and spatial accelerators, one that
decouples hardware specialization from software development. Such a
representation would unlock architectural portability, compiler innovation, and
a scalable, open foundation for spatial computing.

### 4. [Beyond Tokens: Enhancing RTL Quality Estimation via Structural Graph Learning](http://arxiv.org/pdf/2508.18730v1)

Authors: Yi Liu, Hongji Zhang, Yiwen Wang, Dimitris Tsaras, Lei Chen, Mingxuan Yuan, Qiang Xu

Estimating the quality of register transfer level (RTL) designs is crucial in
the electronic design automation (EDA) workflow, as it enables instant feedback
on key metrics like area and delay without the need for time-consuming logic
synthesis. While recent approaches have leveraged large language models (LLMs)
to derive embeddings from RTL code and achieved promising results, they
overlook the structural semantics essential for accurate quality estimation. In
contrast, the control data flow graph (CDFG) view exposes the design's
structural characteristics more explicitly, offering richer cues for
representation learning. In this work, we introduce a novel structure-aware
graph self-supervised learning framework, StructRTL, for improved RTL design
quality estimation. By learning structure-informed representations from CDFGs,
our method significantly outperforms prior art on various quality estimation
tasks. To further boost performance, we incorporate a knowledge distillation
strategy that transfers low-level insights from post-mapping netlists into the
CDFG predictor. Experiments show that our approach establishes new
state-of-the-art results, demonstrating the effectiveness of combining
structural learning with cross-stage supervision.

### 5. [APT-LLM: Exploiting Arbitrary-Precision Tensor Core Computing for LLM Acceleration](http://arxiv.org/pdf/2508.19087v1)

Authors: Shaobo Ma, Chao Fang, Haikuo Shao, Zhongfeng Wang

Large language models (LLMs) have revolutionized AI applications, yet their
enormous computational demands severely limit deployment and real-time
performance. Quantization methods can help reduce computational costs, however,
attaining the extreme efficiency associated with ultra-low-bit quantized LLMs
at arbitrary precision presents challenges on GPUs. This is primarily due to
the limited support for GPU Tensor Cores, inefficient memory management, and
inflexible kernel optimizations. To tackle these challenges, we propose a
comprehensive acceleration scheme for arbitrary precision LLMs, namely APT-LLM.
Firstly, we introduce a novel data format, bipolar-INT, which allows for
efficient and lossless conversion with signed INT, while also being more
conducive to parallel computation. We also develop a matrix multiplication
(MatMul) method allowing for arbitrary precision by dismantling and
reassembling matrices at the bit level. This method provides flexible precision
and optimizes the utilization of GPU Tensor Cores. In addition, we propose a
memory management system focused on data recovery, which strategically employs
fast shared memory to substantially increase kernel execution speed and reduce
memory access latency. Finally, we develop a kernel mapping method that
dynamically selects the optimal configurable hyperparameters of kernels for
varying matrix sizes, enabling optimal performance across different LLM
architectures and precision settings. In LLM inference, APT-LLM achieves up to
a 3.99$\times$ speedup compared to FP16 baselines and a 2.16$\times$ speedup
over NVIDIA CUTLASS INT4 acceleration on RTX 3090. On RTX 4090 and H800,
APT-LLM achieves up to 2.44$\times$ speedup over FP16 and 1.65$\times$ speedup
over CUTLASS integer baselines.

### 6. [Architecting Distributed Quantum Computers: Design Insights from Resource Estimation](http://arxiv.org/pdf/2508.19160v1)

Authors: Dmitry Filippov, Peter Yang, Prakash Murali

To enable practically useful quantum computing, we require hundreds to
thousands of logical qubits (collections of physical qubits with error
correction). Current monolithic device architectures have scaling limits beyond
few tens of logical qubits. To scale up, we require architectures that
orchestrate several monolithic devices into a distributed quantum computing
system. Currently, resource estimation, which is crucial for determining
hardware needs and bottlenecks, focuses exclusively on monolithic systems. Our
work fills this gap and answers key architectural design questions about
distributed systems, including the impact of distribution on application
resource needs, the organization of qubits across nodes and the requirements of
entanglement distillation (quantum network). To answer these questions, we
develop a novel resource estimation framework that models the key components of
the distributed execution stack. We analyse the performance of practical
quantum algorithms on various hardware configurations, spanning different qubit
speeds, entanglement generation rates and distillation protocols. We show that
distributed architectures have practically feasible resource requirements; for
a node size of 45K qubits, distributed systems need on average 1.4X higher
number of physical qubits and 4X higher execution time compared to monolithic
architectures, but with more favourable hardware implementation prospects. Our
insights on entanglement generation rates, node sizes and architecture have the
potential to inform system designs in the coming years.

### Computational Complexity

### 1. [Pointer Chasing with Unlimited Interaction](http://arxiv.org/pdf/2508.19158v1)

Authors: Orr Fischer, Rotem Oshman, Adi Rosen, Tal Roth

Pointer-chasing is a central problem in two-party communication complexity:
given input size $n$ and a parameter $k$, the two players Alice and Bob are
given functions $N_A, N_B: [n] \rightarrow [n]$, respectively, and their goal
is to compute the value of $p_k$, where $p_0 = 1$, $p_1 = N_A(p_0)$, $p_2 =
N_B(p_1) = N_B(N_A(p_0))$, $p_3 = N_A(p_2) = N_A(N_B(N_A(p_0)))$ and so on,
applying $N_A$ in even steps and $N_B$ in odd steps, for a total of $k$ steps.
It is trivial to solve the problem using $k$ communication rounds, with Alice
speaking first, by simply ``chasing the function'' for $k$ steps. Many works
have studied the communication complexity of pointer chasing, although the
focus has always been on protocols with $k-1$ communication rounds, or with $k$
rounds where Bob (the ``wrong player'') speaks first. Many works have studied
this setting giving sometimes tight or near-tight results.
  In this paper we study the communication complexity of the pointer chasing
problem when the interaction between the two players is unlimited, i.e.,
without any restriction on the number of rounds. Perhaps surprisingly, this
question was not studied before, to the best of our knowledge. Our main result
is that the trivial $k$-round protocol is nearly tight (even) when the number
of rounds is not restricted: we give a lower bound of $\Omega(k \log (n/k))$ on
the randomized communication complexity of the pointer chasing problem with
unlimited interaction, and a somewhat stronger lower bound of $\Omega(k \log
\log{k})$ for protocols with zero error.
  When combined with prior work, our results also give a nearly-tight bound on
the communication complexity of protocols using at most $k-1$ rounds, across
all regimes of $k$; for $k > \sqrt{n}$ there was previously a significant gap
between the upper and lower bound.

### Computational Engineering

### 1. [Graph Neural Network-Based Topology Optimization for Self-Supporting Structures in Additive Manufacturing](http://arxiv.org/pdf/2508.19169v1)

Authors: Alireza Tabarraei, Saquib Ahmad Bhuiyan

This paper presents a machine learning-based framework for topology
optimization of self-supporting structures, specifically tailored for additive
manufacturing (AM). By employing a graph neural network (GNN) that acts as a
neural field over the finite element mesh, the framework effectively learns and
predicts continuous material distributions. An integrated AM filter ensures
printability by eliminating unsupported overhangs, while the optimization
process minimizes structural compliance under volume and stress constraints.
The stress constraint is enforced using a differentiable p-norm aggregation of
von Mises stress, promoting mechanical reliability in the optimized designs. A
key advantage of the approach lies in its fully differentiable architecture,
which leverages automatic differentiation throughout the optimization
loop--eliminating the need for explicit sensitivity derivation for both the
filter and the stress constraint. Numerical experiments demonstrate the ability
of the framework to generate stress-constrained manufacturable topologies under
various loading and boundary conditions, offering a practical pathway toward
AM-ready high-performance designs with reduced post-processing requirements.

### 2. [Ab-initio Quantum Transport with the GW Approximation, 42,240 Atoms, and Sustained Exascale Performance](http://arxiv.org/pdf/2508.19138v1)

Authors: Nicolas Vetsch, Alexander Maeder, Vincent Maillou, Anders Winka, Jiang Cao, Grzegorz Kwasniewski, Leonard Deuschle, Torsten Hoefler, Alexandros Nikolaos Ziogas, Mathieu Luisier

Designing nanoscale electronic devices such as the currently manufactured
nanoribbon field-effect transistors (NRFETs) requires advanced modeling tools
capturing all relevant quantum mechanical effects. State-of-the-art approaches
combine the non-equilibrium Green's function (NEGF) formalism and density
functional theory (DFT). However, as device dimensions do not exceed a few
nanometers anymore, electrons are confined in ultra-small volumes, giving rise
to strong electron-electron interactions. To account for these critical
effects, DFT+NEGF solvers should be extended with the GW approximation, which
massively increases their computational intensity. Here, we present the first
implementation of the NEGF+GW scheme capable of handling NRFET geometries with
dimensions comparable to experiments. This package, called QuaTrEx, makes use
of a novel spatial domain decomposition scheme, can treat devices made of up to
84,480 atoms, scales very well on the Alps and Frontier supercomputers (>80%
weak scaling efficiency), and sustains an exascale FP64 performance on 42,240
atoms (1.15 Eflop/s).

### Computation and Language

### 1. [A New NMT Model for Translating Clinical Texts from English to Spanish](http://arxiv.org/pdf/2508.18607v1)

Authors: Rumeng Li, Xun Wang, Hong Yu

Translating electronic health record (EHR) narratives from English to Spanish
is a clinically important yet challenging task due to the lack of a
parallel-aligned corpus and the abundant unknown words contained. To address
such challenges, we propose \textbf{NOOV} (for No OOV), a new neural machine
translation (NMT) system that requires little in-domain parallel-aligned corpus
for training. NOOV integrates a bilingual lexicon automatically learned from
parallel-aligned corpora and a phrase look-up table extracted from a large
biomedical knowledge resource, to alleviate both the unknown word problem and
the word-repeat challenge in NMT, enhancing better phrase generation of NMT
systems. Evaluation shows that NOOV is able to generate better translation of
EHR with improvement in both accuracy and fluency.

### 2. [Thinking Before You Speak: A Proactive Test-time Scaling Approach](http://arxiv.org/pdf/2508.18648v1)

Authors: Cong Li, Wenchang Chai, Hejun Wu, Yan Pan, Pengxu Wei, Liang Lin

Large Language Models (LLMs) often exhibit deficiencies with complex
reasoning tasks, such as maths, which we attribute to the discrepancy between
human reasoning patterns and those presented in the LLMs' training data. When
dealing with complex problems, humans tend to think carefully before expressing
solutions. However, they often do not articulate their inner thoughts,
including their intentions and chosen methodologies. Consequently, critical
insights essential for bridging reasoning steps may be absent in training data
collected from human sources. To bridge this gap, we proposes inserting
\emph{insight}s between consecutive reasoning steps, which review the status
and initiate the next reasoning steps. Unlike prior prompting strategies that
rely on a single or a workflow of static prompts to facilitate reasoning,
\emph{insight}s are \emph{proactively} generated to guide reasoning processes.
We implement our idea as a reasoning framework, named \emph{Thinking Before You
Speak} (TBYS), and design a pipeline for automatically collecting and filtering
in-context examples for the generation of \emph{insight}s, which alleviates
human labeling efforts and fine-tuning overheads. Experiments on challenging
mathematical datasets verify the effectiveness of TBYS. Project website:
https://gitee.com/jswrt/TBYS

### 3. [Knowing or Guessing? Robust Medical Visual Question Answering via Joint Consistency and Contrastive Learning](http://arxiv.org/pdf/2508.18687v1)

Authors: Songtao Jiang, Yuxi Chen, Sibo Song, Yan Zhang, Yeying Jin, Yang Feng, Jian Wu, Zuozhu Liu

In high-stakes medical applications, consistent answering across diverse
question phrasings is essential for reliable diagnosis. However, we reveal that
current Medical Vision-Language Models (Med-VLMs) exhibit concerning fragility
in Medical Visual Question Answering, as their answers fluctuate significantly
when faced with semantically equivalent rephrasings of medical questions. We
attribute this to two limitations: (1) insufficient alignment of medical
concepts, leading to divergent reasoning patterns, and (2) hidden biases in
training data that prioritize syntactic shortcuts over semantic understanding.
To address these challenges, we construct RoMed, a dataset built upon original
VQA datasets containing 144k questions with variations spanning word-level,
sentence-level, and semantic-level perturbations. When evaluating
state-of-the-art (SOTA) models like LLaVA-Med on RoMed, we observe alarming
performance drops (e.g., a 40\% decline in Recall) compared to original VQA
benchmarks, exposing critical robustness gaps. To bridge this gap, we propose
Consistency and Contrastive Learning (CCL), which integrates two key
components: (1) knowledge-anchored consistency learning, aligning Med-VLMs with
medical knowledge rather than shallow feature patterns, and (2) bias-aware
contrastive learning, mitigating data-specific priors through discriminative
representation refinement. CCL achieves SOTA performance on three popular VQA
benchmarks and notably improves answer consistency by 50\% on the challenging
RoMed test set, demonstrating significantly enhanced robustness. Code will be
released.

### 4. [Attention2Probability: Attention-Driven Terminology Probability Estimation for Robust Speech-to-Text System](http://arxiv.org/pdf/2508.18701v1)

Authors: Yanfan Du, Jun Zhang, Bin Wang, Jin Qiu, Lu Huang, Yuan Ge, Xiaoqian Liu, Tong Xiao, Jingbo Zhu

Recent advances in speech large language models (SLMs) have improved speech
recognition and translation in general domains, but accurately generating
domain-specific terms or neologisms remains challenging. To address this, we
propose Attention2Probability: attention-driven terminology probability
estimation for robust speech-to-text system, which is lightweight, flexible,
and accurate. Attention2Probability converts cross-attention weights between
speech and terminology into presence probabilities, and it further employs
curriculum learning to enhance retrieval accuracy. Furthermore, to tackle the
lack of data for speech-to-text tasks with terminology intervention, we create
and release a new speech dataset with terminology to support future research in
this area. Experimental results show that Attention2Probability significantly
outperforms the VectorDB method on our test set. Specifically, its maximum
recall rates reach 92.57% for Chinese and 86.83% for English. This high recall
is achieved with a latency of only 8.71ms per query. Intervening in SLMs'
recognition and translation tasks using Attention2Probability-retrieved terms
improves terminology accuracy by 6-17%, while revealing that the current
utilization of terminology by SLMs has limitations.

### 5. [Filtering for Creativity: Adaptive Prompting for Multilingual Riddle Generation in LLMs](http://arxiv.org/pdf/2508.18709v1)

Authors: Duy Le, Kent Ziti, Evan Girard-Sun, Sean O'Brien, Vasu Sharma, Kevin Zhu

Multilingual riddle generation challenges large language models (LLMs) to
balance cultural fluency with creative abstraction. Standard prompting
strategies -- zero-shot, few-shot, chain-of-thought -- tend to reuse memorized
riddles or perform shallow paraphrasing. We introduce Adaptive Originality
Filtering (AOF), a prompting framework that filters redundant generations using
cosine-based similarity rejection, while enforcing lexical novelty and
cross-lingual fidelity. Evaluated across three LLMs and four language pairs,
AOF-enhanced GPT-4o achieves \texttt{0.177} Self-BLEU and \texttt{0.915}
Distinct-2 in Japanese, signaling improved lexical diversity and reduced
redundancy compared to other prompting methods and language pairs. Our findings
show that semantic rejection can guide culturally grounded, creative generation
without task-specific fine-tuning.

### 6. [EMMM, Explain Me My Model! Explainable Machine Generated Text Detection in Dialogues](http://arxiv.org/pdf/2508.18715v1)

Authors: Angela Yifei Yuan, Haoyi Li, Soyeon Caren Han, Christopher Leckie

The rapid adoption of large language models (LLMs) in customer service
introduces new risks, as malicious actors can exploit them to conduct
large-scale user impersonation through machine-generated text (MGT). Current
MGT detection methods often struggle in online conversational settings,
reducing the reliability and interpretability essential for trustworthy AI
deployment. In customer service scenarios where operators are typically
non-expert users, explanation become crucial for trustworthy MGT detection. In
this paper, we propose EMMM, an explanation-then-detection framework that
balances latency, accuracy, and non-expert-oriented interpretability.
Experimental results demonstrate that EMMM provides explanations accessible to
non-expert users, with 70\% of human evaluators preferring its outputs, while
achieving competitive accuracy compared to state-of-the-art models and
maintaining low latency, generating outputs within 1 second. Our code and
dataset are open-sourced at
https://github.com/AngieYYF/EMMM-explainable-chatbot-detection.

### 7. [Chronological Passage Assembling in RAG framework for Temporal Question Answering](http://arxiv.org/pdf/2508.18748v1)

Authors: Byeongjeong Kim, Jeonghyun Park, Joonho Yang, Hwanhee Lee

Long-context question answering over narrative tasks is challenging because
correct answers often hinge on reconstructing a coherent timeline of events
while preserving contextual flow in a limited context window.
Retrieval-augmented generation (RAG) indexing methods aim to address this
challenge by selectively retrieving only necessary document segments. However,
narrative texts possess unique characteristics that limit the effectiveness of
these existing approaches. Specifically, understanding narrative texts requires
more than isolated segments, as the broader context and sequential
relationships between segments are crucial for comprehension. To address these
limitations, we propose ChronoRAG, a novel RAG framework specialized for
narrative texts. This approach focuses on two essential aspects: refining
dispersed document information into coherent and structured passages, and
preserving narrative flow by explicitly capturing and maintaining the temporal
order among retrieved passages. We empirically demonstrate the effectiveness of
ChronoRAG through experiments on the NarrativeQA dataset, showing substantial
improvements in tasks requiring both factual identification and comprehension
of complex sequential relationships, underscoring that reasoning over temporal
order is crucial in resolving narrative QA.

### 8. [ThinkDial: An Open Recipe for Controlling Reasoning Effort in Large Language Models](http://arxiv.org/pdf/2508.18773v1)

Authors: Qianyu He, Siyu Yuan, Xuefeng Li, Mingxuan Wang, Jiangjie Chen

Large language models (LLMs) with chain-of-thought reasoning have
demonstrated remarkable problem-solving capabilities, but controlling their
computational effort remains a significant challenge for practical deployment.
Recent proprietary systems like OpenAI's gpt-oss series have introduced
discrete operational modes for intuitive reasoning control, but the open-source
community has largely failed to achieve such capabilities. In this paper, we
introduce ThinkDial, the first open-recipe end-to-end framework that
successfully implements gpt-oss-style controllable reasoning through discrete
operational modes. Our system enables seamless switching between three distinct
reasoning regimes: High mode (full reasoning capability), Medium mode (50
percent token reduction with <10 percent performance degradation), and Low mode
(75 percent token reduction with <15 percent performance degradation). We
achieve this through an end-to-end training paradigm that integrates
budget-mode control throughout the entire pipeline: budget-mode supervised
fine-tuning that embeds controllable reasoning capabilities directly into the
learning process, and two-phase budget-aware reinforcement learning with
adaptive reward shaping. Extensive experiments demonstrate that ThinkDial
achieves target compression-performance trade-offs with clear response length
reductions while maintaining performance thresholds. The framework also
exhibits strong generalization capabilities on out-of-distribution tasks.

### 9. [Controllable Conversational Theme Detection Track at DSTC 12](http://arxiv.org/pdf/2508.18783v1)

Authors: Igor Shalyminov, Hang Su, Jake Vincent, Siffi Singh, Jason Cai, James Gung, Raphael Shu, Saab Mansour

Conversational analytics has been on the forefront of transformation driven
by the advances in Speech and Natural Language Processing techniques. Rapid
adoption of Large Language Models (LLMs) in the analytics field has taken the
problems that can be automated to a new level of complexity and scale. In this
paper, we introduce Theme Detection as a critical task in conversational
analytics, aimed at automatically identifying and categorizing topics within
conversations. This process can significantly reduce the manual effort involved
in analyzing expansive dialogs, particularly in domains like customer support
or sales. Unlike traditional dialog intent detection, which often relies on a
fixed set of intents for downstream system logic, themes are intended as a
direct, user-facing summary of the conversation's core inquiry. This
distinction allows for greater flexibility in theme surface forms and
user-specific customizations. We pose Controllable Conversational Theme
Detection problem as a public competition track at Dialog System Technology
Challenge (DSTC) 12 -- it is framed as joint clustering and theme labeling of
dialog utterances, with the distinctive aspect being controllability of the
resulting theme clusters' granularity achieved via the provided user preference
data. We give an overview of the problem, the associated dataset and the
evaluation metrics, both automatic and human. Finally, we discuss the
participant teams' submissions and provide insights from those. The track
materials (data and code) are openly available in the GitHub repository.

### 10. [LaTeXTrans: Structured LaTeX Translation with Multi-Agent Coordination](http://arxiv.org/pdf/2508.18791v1)

Authors: Ziming Zhu, Chenglong Wang, Shunjie Xing, Yifu Huo, Fengning Tian, Quan Du, Di Yang, Chunliang Zhang, Tong Xiao, Jingbo Zhu

Despite the remarkable progress of modern machine translation (MT) systems on
general-domain texts, translating structured LaTeX-formatted documents remains
a significant challenge. These documents typically interleave natural language
with domain-specific syntax, such as mathematical equations, tables, figures,
and cross-references, all of which must be accurately preserved to maintain
semantic integrity and compilability. In this paper, we introduce LaTeXTrans, a
collaborative multi-agent system designed to address this challenge. LaTeXTrans
ensures format preservation, structural fidelity, and terminology consistency
through six specialized agents: 1) a Parser that decomposes LaTeX into
translation-friendly units via placeholder substitution and syntax filtering;
2) a Translator, Validator, Summarizer, and Terminology Extractor that work
collaboratively to ensure context-aware, self-correcting, and
terminology-consistent translations; 3) a Generator that reconstructs the
translated content into well-structured LaTeX documents. Experimental results
demonstrate that LaTeXTrans can outperform mainstream MT systems in both
translation accuracy and structural fidelity, offering an effective and
practical solution for translating LaTeX-formatted documents.

### Cryptography and Security

### 1. [Immutable Digital Recognition via Blockchain](http://arxiv.org/pdf/2508.18750v1)

Authors: Zeng Zhang, Xiaoqi Li

The process integrates the decentralised management and centralised operation
models, aligning them with the national policy directives. The developed
solution enables the full utilisation of blockchain technology's advantages
while also fostering community participation. Consequently, it establishes a
secure, legal, reliable, and dynamic electronic certification system.

### 2. [EnerSwap: Large-Scale, Privacy-First Automated Market Maker for V2G Energy Trading](http://arxiv.org/pdf/2508.18942v1)

Authors: Ahmed Mounsf Rafik Bendada, Yacine Ghamri-Doudane

With the rapid growth of Electric Vehicle (EV) technology, EVs are destined
to shape the future of transportation. The large number of EVs facilitates the
development of the emerging vehicle-to-grid (V2G) technology, which realizes
bidirectional energy exchanges between EVs and the power grid. This has led to
the setting up of electricity markets that are usually confined to a small
geographical location, often with a small number of participants. Usually,
these markets are manipulated by intermediaries responsible for collecting bids
from prosumers, determining the market-clearing price, incorporating grid
constraints, and accounting for network losses. While centralized models can be
highly efficient, they grant excessive power to the intermediary by allowing
them to gain exclusive access to prosumers \textquotesingle price preferences.
This opens the door to potential market manipulation and raises significant
privacy concerns for users, such as the location of energy providers. This lack
of protection exposes users to potential risks, as untrustworthy servers and
malicious adversaries can exploit this information to infer trading activities
and real identities. This work proposes a secure, decentralized exchange market
built on blockchain technology, utilizing a privacy-preserving Automated Market
Maker (AMM) model to offer open and fair, and equal access to traders, and
mitigates the most common trading-manipulation attacks. Additionally, it
incorporates a scalable architecture based on geographical dynamic sharding,
allowing for efficient resource allocation and improved performance as the
market grows.

### 3. [LLMs in the SOC: An Empirical Study of Human-AI Collaboration in Security Operations Centres](http://arxiv.org/pdf/2508.18947v1)

Authors: Ronal Singh, Shahroz Tariq, Fatemeh Jalalvand, Mohan Baruwal Chhetri, Surya Nepal, Cecile Paris, Martin Lochner

The integration of Large Language Models (LLMs) into Security Operations
Centres (SOCs) presents a transformative, yet still evolving, opportunity to
reduce analyst workload through human-AI collaboration. However, their
real-world application in SOCs remains underexplored. To address this gap, we
present a longitudinal study of 3,090 analyst queries from 45 SOC analysts over
10 months. Our analysis reveals that analysts use LLMs as on-demand aids for
sensemaking and context-building, rather than for making high-stakes
determinations, preserving analyst decision authority. The majority of queries
are related to interpreting low-level telemetry (e.g., commands) and refining
technical communication through short (1-3 turn) interactions. Notably, 93% of
queries align with established cybersecurity competencies (NICE Framework),
underscoring the relevance of LLM use for SOC-related tasks. Despite variations
in tasks and engagement, usage trends indicate a shift from occasional
exploration to routine integration, with growing adoption and sustained use
among a subset of analysts. We find that LLMs function as flexible, on-demand
cognitive aids that augment, rather than replace, SOC expertise. Our study
provides actionable guidance for designing context-aware, human-centred AI
assistance in security operations, highlighting the need for further
in-the-wild research on real-world analyst-LLM collaboration, challenges, and
impacts.

### 4. [PRISM: Robust VLM Alignment with Principled Reasoning for Integrated Safety in Multimodality](http://arxiv.org/pdf/2508.18649v1)

Authors: Nanxi Li, Zhengyue Zhao, Chaowei Xiao

Safeguarding vision-language models (VLMs) is a critical challenge, as
existing methods often suffer from over-defense, which harms utility, or rely
on shallow alignment, failing to detect complex threats that require deep
reasoning. To this end, we introduce PRISM (Principled Reasoning for Integrated
Safety in Multimodality), a system2-like framework that aligns VLMs by
embedding a structured, safety-aware reasoning process. Our framework consists
of two key components: PRISM-CoT, a dataset that teaches safety-aware
chain-of-thought reasoning, and PRISM-DPO, generated via Monte Carlo Tree
Search (MCTS) to further refine this reasoning through Direct Preference
Optimization to help obtain a delicate safety boundary. Comprehensive
evaluations demonstrate PRISM's effectiveness, achieving remarkably low attack
success rates including 0.15% on JailbreakV-28K for Qwen2-VL and 90%
improvement over the previous best method on VLBreak for LLaVA-1.5. PRISM also
exhibits strong robustness against adaptive attacks, significantly increasing
computational costs for adversaries, and generalizes effectively to
out-of-distribution challenges, reducing attack success rates to just 8.70% on
the challenging multi-image MIS benchmark. Remarkably, this robust defense is
achieved while preserving, and in some cases enhancing, model utility. To
promote reproducibility, we have made our code, data, and model weights
available at https://github.com/SaFoLab-WISC/PRISM.

### 5. [UniC-RAG: Universal Knowledge Corruption Attacks to Retrieval-Augmented Generation](http://arxiv.org/pdf/2508.18652v1)

Authors: Runpeng Geng, Yanting Wang, Ying Chen, Jinyuan Jia

Retrieval-augmented generation (RAG) systems are widely deployed in
real-world applications in diverse domains such as finance, healthcare, and
cybersecurity. However, many studies showed that they are vulnerable to
knowledge corruption attacks, where an attacker can inject adversarial texts
into the knowledge database of a RAG system to induce the LLM to generate
attacker-desired outputs. Existing studies mainly focus on attacking specific
queries or queries with similar topics (or keywords). In this work, we propose
UniC-RAG, a universal knowledge corruption attack against RAG systems. Unlike
prior work, UniC-RAG jointly optimizes a small number of adversarial texts that
can simultaneously attack a large number of user queries with diverse topics
and domains, enabling an attacker to achieve various malicious objectives, such
as directing users to malicious websites, triggering harmful command execution,
or launching denial-of-service attacks. We formulate UniC-RAG as an
optimization problem and further design an effective solution to solve it,
including a balanced similarity-based clustering method to enhance the attack's
effectiveness. Our extensive evaluations demonstrate that UniC-RAG is highly
effective and significantly outperforms baselines. For instance, UniC-RAG could
achieve over 90% attack success rate by injecting 100 adversarial texts into a
knowledge database with millions of texts to simultaneously attack a large set
of user queries (e.g., 2,000). Additionally, we evaluate existing defenses and
show that they are insufficient to defend against UniC-RAG, highlighting the
need for new defense mechanisms in RAG systems.

### 6. [Hidden Tail: Adversarial Image Causing Stealthy Resource Consumption in Vision-Language Models](http://arxiv.org/pdf/2508.18805v1)

Authors: Rui Zhang, Zihan Wang, Tianli Yang, Hongwei Li, Wenbo Jiang, Qingchuan Zhao, Yang Liu, Guowen Xu

Vision-Language Models (VLMs) are increasingly deployed in real-world
applications, but their high inference cost makes them vulnerable to resource
consumption attacks. Prior attacks attempt to extend VLM output sequences by
optimizing adversarial images, thereby increasing inference costs. However,
these extended outputs often introduce irrelevant abnormal content,
compromising attack stealthiness. This trade-off between effectiveness and
stealthiness poses a major limitation for existing attacks. To address this
challenge, we propose \textit{Hidden Tail}, a stealthy resource consumption
attack that crafts prompt-agnostic adversarial images, inducing VLMs to
generate maximum-length outputs by appending special tokens invisible to users.
Our method employs a composite loss function that balances semantic
preservation, repetitive special token induction, and suppression of the
end-of-sequence (EOS) token, optimized via a dynamic weighting strategy.
Extensive experiments show that \textit{Hidden Tail} outperforms existing
attacks, increasing output length by up to 19.2$\times$ and reaching the
maximum token limit, while preserving attack stealthiness. These results
highlight the urgent need to improve the robustness of VLMs against
efficiency-oriented adversarial threats. Our code is available at
https://github.com/zhangrui4041/Hidden_Tail.

### 7. [Quantum computing on encrypted data with arbitrary rotation gates](http://arxiv.org/pdf/2508.18811v1)

Authors: Mohit Joshi, Manoj Kumar Mishra, S. Karthikeyan

An efficient technique of computing on encrypted data allows a client with
limited capability to perform complex operations on a remote fault-tolerant
server without leaking anything about the input or output. Quantum computing
provides information-theoretic security to solve such a problem, and many such
techniques have been proposed under the premises of half-blind quantum
computation. However, they are dependent on a fixed non-parametric resource set
that comprises some universal combination of $H,S,T,CX, CZ$ or $CCX$ gates. In
this study, we show that recursive decryption of the parametric gate,
$R_z(\theta)$, is possible exactly when $\theta=\pm\pi/2^m$ for $m\in
\mathbb{Z^{+}}$, and approximately with arbitrary precision $\epsilon$ for
given $\theta$. We also show that a blind algorithm based on such a technique
needs at most $O(\log_2^2(\pi/\epsilon))$ computation steps and communication
rounds, while the techniques based on a non-parametric resource set require
$O(\ln^{3.97}(1/\epsilon))$ rounds. We use these results to propose a universal
scheme of half-blind quantum computation for computing on encrypted data using
arbitrary rotation gates. This substantial reduction in the depth of blind
circuit is an affirmative step towards the practical application of such
techniques in secure NISQ-era computing.

### 8. [DRMD: Deep Reinforcement Learning for Malware Detection under Concept Drift](http://arxiv.org/pdf/2508.18839v1)

Authors: Shae McFadden, Myles Foley, Mario D'Onghia, Chris Hicks, Vasilios Mavroudis, Nicola Paoletti, Fabio Pierazzi

Malware detection in real-world settings must deal with evolving threats,
limited labeling budgets, and uncertain predictions. Traditional classifiers,
without additional mechanisms, struggle to maintain performance under concept
drift in malware domains, as their supervised learning formulation cannot
optimize when to defer decisions to manual labeling and adaptation. Modern
malware detection pipelines combine classifiers with monthly active learning
(AL) and rejection mechanisms to mitigate the impact of concept drift. In this
work, we develop a novel formulation of malware detection as a one-step Markov
Decision Process and train a deep reinforcement learning (DRL) agent,
simultaneously optimizing sample classification performance and rejecting
high-risk samples for manual labeling. We evaluated the joint detection and
drift mitigation policy learned by the DRL-based Malware Detection (DRMD) agent
through time-aware evaluations on Android malware datasets subject to realistic
drift requiring multi-year performance stability. The policies learned under
these conditions achieve a higher Area Under Time (AUT) performance compared to
standard classification approaches used in the domain, showing improved
resilience to concept drift. Specifically, the DRMD agent achieved a
$5.18\pm5.44$, $14.49\pm12.86$, and $10.06\pm10.81$ average AUT performance
improvement for the classification only, classification with rejection, and
classification with rejection and AL settings, respectively. Our results
demonstrate for the first time that DRL can facilitate effective malware
detection and improved resiliency to concept drift in the dynamic environment
of the Android malware domain.

### 9. [The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization](http://arxiv.org/pdf/2508.18976v1)

Authors: Stephen Meisenbacher, Alexandra Klymenko, Andreea-Elena Bodea, Florian Matthes

Differentially private text sanitization refers to the process of privatizing
texts under the framework of Differential Privacy (DP), providing provable
privacy guarantees while also empirically defending against adversaries seeking
to harm privacy. Despite their simplicity, DP text sanitization methods
operating at the word level exhibit a number of shortcomings, among them the
tendency to leave contextual clues from the original texts due to randomization
during sanitization $\unicode{x2013}$ this we refer to as $\textit{contextual
vulnerability}$. Given the powerful contextual understanding and inference
capabilities of Large Language Models (LLMs), we explore to what extent LLMs
can be leveraged to exploit the contextual vulnerability of DP-sanitized texts.
We expand on previous work not only in the use of advanced LLMs, but also in
testing a broader range of sanitization mechanisms at various privacy levels.
Our experiments uncover a double-edged sword effect of LLM-based data
reconstruction attacks on privacy and utility: while LLMs can indeed infer
original semantics and sometimes degrade empirical privacy protections, they
can also be used for good, to improve the quality and privacy of DP-sanitized
texts. Based on our findings, we propose recommendations for using LLM data
reconstruction as a post-processing step, serving to increase privacy
protection by thinking adversarially.

### 10. [mmKey: Channel-Aware Beam Shaping for Reliable Key Generation in mmWave Wireless Networks](http://arxiv.org/pdf/2508.19010v1)

Authors: Poorya Mollahosseini, Yasaman Ghasempour

Physical-layer key generation (PLKG) has emerged as a promising technique to
secure next-generation wireless networks by exploiting the inherent properties
of the wireless channel. However, PLKG faces fundamental challenges in the
millimeter wave (mmWave) regime due to channel sparsity, higher phase noise,
and higher path loss, which undermine both the randomness and reciprocity
required for secure key generation. In this paper, we present mmKey, a novel
PLKG framework that capitalizes on the availability of multiple antennas at
mmWave wireless nodes to inject randomness into an otherwise quasi-static
wireless channel. Different from prior works that sacrifice either the secrecy
of the key generation or the robustness, mmKey balances these two requirements.
In particular, mmKey leverages a genetic algorithm to gradually evolve the
initial weight vector population toward configurations that suppress the LOS
component while taking into account the channel conditions, specifically, the
sparsity and the signal-to-noise ratio (SNR). Extensive simulations show that
mmKey improves the secrecy gap by an average of 39.4% over random beamforming
and 34.0% over null beamforming, outperforming conventional schemes.

### Computer Vision and Pattern Recognition

### 1. [Wan-S2V: Audio-Driven Cinematic Video Generation](http://arxiv.org/pdf/2508.18621v1)

Authors: Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Dechao Meng, Jinwei Qi, Penchong Qiao, Zhen Shen, Yafei Song, Ke Sun, Linrui Tian, Guangyuan Wang, Qi Wang, Zhongjian Wang, Jiayu Xiao, Sheng Xu, Bang Zhang, Peng Zhang, Xindi Zhang, Zhe Zhang, Jingren Zhou, Lian Zhuo

Current state-of-the-art (SOTA) methods for audio-driven character animation
demonstrate promising performance for scenarios primarily involving speech and
singing. However, they often fall short in more complex film and television
productions, which demand sophisticated elements such as nuanced character
interactions, realistic body movements, and dynamic camera work. To address
this long-standing challenge of achieving film-level character animation, we
propose an audio-driven model, which we refere to as Wan-S2V, built upon Wan.
Our model achieves significantly enhanced expressiveness and fidelity in
cinematic contexts compared to existing approaches. We conducted extensive
experiments, benchmarking our method against cutting-edge models such as
Hunyuan-Avatar and Omnihuman. The experimental results consistently demonstrate
that our approach significantly outperforms these existing solutions.
Additionally, we explore the versatility of our method through its applications
in long-form video generation and precise video lip-sync editing.

### 2. [Decouple, Reorganize, and Fuse: A Multimodal Framework for Cancer Survival Prediction](http://arxiv.org/pdf/2508.18632v1)

Authors: Huayi Wang, Haochao Ying, Yuyang Xu, Qibo Qiu, Cheng Zhang, Danny Z. Chen, Ying Sun, Jian Wu

Cancer survival analysis commonly integrates information across diverse
medical modalities to make survival-time predictions. Existing methods
primarily focus on extracting different decoupled features of modalities and
performing fusion operations such as concatenation, attention, and MoE-based
(Mixture-of-Experts) fusion. However, these methods still face two key
challenges: i) Fixed fusion schemes (concatenation and attention) can lead to
model over-reliance on predefined feature combinations, limiting the dynamic
fusion of decoupled features; ii) in MoE-based fusion methods, each expert
network handles separate decoupled features, which limits information
interaction among the decoupled features. To address these challenges, we
propose a novel Decoupling-Reorganization-Fusion framework (DeReF), which
devises a random feature reorganization strategy between modalities decoupling
and dynamic MoE fusion modules.Its advantages are: i) it increases the
diversity of feature combinations and granularity, enhancing the generalization
ability of the subsequent expert networks; ii) it overcomes the problem of
information closure and helps expert networks better capture information among
decoupled features. Additionally, we incorporate a regional cross-attention
network within the modality decoupling module to improve the representation
quality of decoupled features. Extensive experimental results on our in-house
Liver Cancer (LC) and three widely used TCGA public datasets confirm the
effectiveness of our proposed method. The code will be made publicly available.

### 3. [OwlCap: Harmonizing Motion-Detail for Video Captioning via HMD-270K and Caption Set Equivalence Reward](http://arxiv.org/pdf/2508.18634v1)

Authors: Chunlin Zhong, Qiuxia Hou, Zhangjun Zhou, Shuang Hao, Haonan Lu, Yanhao Zhang, He Tang, Xiang Bai

Video captioning aims to generate comprehensive and coherent descriptions of
the video content, contributing to the advancement of both video understanding
and generation. However, existing methods often suffer from motion-detail
imbalance, as models tend to overemphasize one aspect while neglecting the
other. This imbalance results in incomplete captions, which in turn leads to a
lack of consistency in video understanding and generation. To address this
issue, we propose solutions from two aspects: 1) Data aspect: We constructed
the Harmonizing Motion-Detail 270K (HMD-270K) dataset through a two-stage
pipeline: Motion-Detail Fusion (MDF) and Fine-Grained Examination (FGE). 2)
Optimization aspect: We introduce the Caption Set Equivalence Reward (CSER)
based on Group Relative Policy Optimization (GRPO). CSER enhances completeness
and accuracy in capturing both motion and details through unit-to-set matching
and bidirectional validation. Based on the HMD-270K supervised fine-tuning and
GRPO post-training with CSER, we developed OwlCap, a powerful video captioning
multi-modal large language model (MLLM) with motion-detail balance.
Experimental results demonstrate that OwlCap achieves significant improvements
compared to baseline models on two benchmarks: the detail-focused VDC (+4.2
Acc) and the motion-focused DREAM-1K (+4.6 F1). The HMD-270K dataset and OwlCap
model will be publicly released to facilitate video captioning research
community advancements.

### 4. [SFormer: SNR-guided Transformer for Underwater Image Enhancement from the Frequency Domain](http://arxiv.org/pdf/2508.18664v1)

Authors: Xin Tian, Yingtie Lei, Xiujun Zhang, Zimeng Li, Chi-Man Pun, Xuhang Chen

Recent learning-based underwater image enhancement (UIE) methods have
advanced by incorporating physical priors into deep neural networks,
particularly using the signal-to-noise ratio (SNR) prior to reduce
wavelength-dependent attenuation. However, spatial domain SNR priors have two
limitations: (i) they cannot effectively separate cross-channel interference,
and (ii) they provide limited help in amplifying informative structures while
suppressing noise. To overcome these, we propose using the SNR prior in the
frequency domain, decomposing features into amplitude and phase spectra for
better channel modulation. We introduce the Fourier Attention SNR-prior
Transformer (FAST), combining spectral interactions with SNR cues to highlight
key spectral components. Additionally, the Frequency Adaptive Transformer (FAT)
bottleneck merges low- and high-frequency branches using a gated attention
mechanism to enhance perceptual quality. Embedded in a unified U-shaped
architecture, these modules integrate a conventional RGB stream with an
SNR-guided branch, forming SFormer. Trained on 4,800 paired images from UIEB,
EUVP, and LSUI, SFormer surpasses recent methods with a 3.1 dB gain in PSNR and
0.08 in SSIM, successfully restoring colors, textures, and contrast in
underwater scenes.

### 5. [Hierarchical Spatio-temporal Segmentation Network for Ejection Fraction Estimation in Echocardiography Videos](http://arxiv.org/pdf/2508.18681v1)

Authors: Dongfang Wang, Jian Yang, Yizhe Zhang, Tao Zhou

Automated segmentation of the left ventricular endocardium in
echocardiography videos is a key research area in cardiology. It aims to
provide accurate assessment of cardiac structure and function through Ejection
Fraction (EF) estimation. Although existing studies have achieved good
segmentation performance, their results do not perform well in EF estimation.
In this paper, we propose a Hierarchical Spatio-temporal Segmentation Network
(\ourmodel) for echocardiography video, aiming to improve EF estimation
accuracy by synergizing local detail modeling with global dynamic perception.
The network employs a hierarchical design, with low-level stages using
convolutional networks to process single-frame images and preserve details,
while high-level stages utilize the Mamba architecture to capture
spatio-temporal relationships. The hierarchical design balances single-frame
and multi-frame processing, avoiding issues such as local error accumulation
when relying solely on single frames or neglecting details when using only
multi-frame data. To overcome local spatio-temporal limitations, we propose the
Spatio-temporal Cross Scan (STCS) module, which integrates long-range context
through skip scanning across frames and positions. This approach helps mitigate
EF calculation biases caused by ultrasound image noise and other factors.

### 6. [Feature-Space Planes Searcher: A Universal Domain Adaptation Framework for Interpretability and Computational Efficiency](http://arxiv.org/pdf/2508.18693v1)

Authors: Zhitong Cheng, Yiran Jiang, Yulong Ge, Yufeng Li, Zhongheng Qin, Rongzhi Lin, Jianwei Ma

Domain shift, characterized by degraded model performance during transition
from labeled source domains to unlabeled target domains, poses a persistent
challenge for deploying deep learning systems. Current unsupervised domain
adaptation (UDA) methods predominantly rely on fine-tuning feature extractors -
an approach limited by inefficiency, reduced interpretability, and poor
scalability to modern architectures.
  Our analysis reveals that models pretrained on large-scale data exhibit
domain-invariant geometric patterns in their feature space, characterized by
intra-class clustering and inter-class separation, thereby preserving
transferable discriminative structures. These findings indicate that domain
shifts primarily manifest as boundary misalignment rather than feature
degradation.
  Unlike fine-tuning entire pre-trained models - which risks introducing
unpredictable feature distortions - we propose the Feature-space Planes
Searcher (FPS): a novel domain adaptation framework that optimizes decision
boundaries by leveraging these geometric patterns while keeping the feature
encoder frozen. This streamlined approach enables interpretative analysis of
adaptation while substantially reducing memory and computational costs through
offline feature extraction, permitting full-dataset optimization in a single
computation cycle.
  Evaluations on public benchmarks demonstrate that FPS achieves competitive or
superior performance to state-of-the-art methods. FPS scales efficiently with
multimodal large models and shows versatility across diverse domains including
protein structure prediction, remote sensing classification, and earthquake
detection. We anticipate FPS will provide a simple, effective, and
generalizable paradigm for transfer learning, particularly in domain adaptation
tasks. .

### 7. [A Novel Deep Hybrid Framework with Ensemble-Based Feature Optimization for Robust Real-Time Human Activity Recognition](http://arxiv.org/pdf/2508.18695v1)

Authors: Wasi Ullah, Yasir Noman Khalid, Saddam Hussain Khan

Human Activity Recognition (HAR) plays a pivotal role in various
applications, including smart surveillance, healthcare, assistive technologies,
sports analytics, etc. However, HAR systems still face critical challenges,
including high computational costs, redundant features, and limited scalability
in real-time scenarios. An optimized hybrid deep learning framework is
introduced that integrates a customized InceptionV3, an LSTM architecture, and
a novel ensemble-based feature selection strategy. The proposed framework first
extracts spatial descriptors using the customized InceptionV3 model, which
captures multilevel contextual patterns, region homogeneity, and fine-grained
localization cues. The temporal dependencies across frames are then modeled
using LSTMs to effectively encode motion dynamics. Finally, an ensemble-based
genetic algorithm with Adaptive Dynamic Fitness Sharing and Attention (ADFSA)
is employed to select a compact and optimized feature set by dynamically
balancing objectives such as accuracy, redundancy, uniqueness, and complexity
reduction. Consequently, the selected feature subsets, which are both diverse
and discriminative, enable various lightweight machine learning classifiers to
achieve accurate and robust HAR in heterogeneous environments. Experimental
results on the robust UCF-YouTube dataset, which presents challenges such as
occlusion, cluttered backgrounds, motion dynamics, and poor illumination,
demonstrate good performance. The proposed approach achieves 99.65% recognition
accuracy, reduces features to as few as 7, and enhances inference time. The
lightweight and scalable nature of the HAR system supports real-time deployment
on edge devices such as Raspberry Pi, enabling practical applications in
intelligent, resource-aware environments, including public safety, assistive
technology, and autonomous monitoring systems.

### 8. [ColorGS: High-fidelity Surgical Scene Reconstruction with Colored Gaussian Splatting](http://arxiv.org/pdf/2508.18696v1)

Authors: Qun Ji, Peng Li, Mingqiang Wei

High-fidelity reconstruction of deformable tissues from endoscopic videos
remains challenging due to the limitations of existing methods in capturing
subtle color variations and modeling global deformations. While 3D Gaussian
Splatting (3DGS) enables efficient dynamic reconstruction, its fixed
per-Gaussian color assignment struggles with intricate textures, and linear
deformation modeling fails to model consistent global deformation. To address
these issues, we propose ColorGS, a novel framework that integrates spatially
adaptive color encoding and enhanced deformation modeling for surgical scene
reconstruction. First, we introduce Colored Gaussian Primitives, which employ
dynamic anchors with learnable color parameters to adaptively encode spatially
varying textures, significantly improving color expressiveness under complex
lighting and tissue similarity. Second, we design an Enhanced Deformation Model
(EDM) that combines time-aware Gaussian basis functions with learnable
time-independent deformations, enabling precise capture of both localized
tissue deformations and global motion consistency caused by surgical
interactions. Extensive experiments on DaVinci robotic surgery videos and
benchmark datasets (EndoNeRF, StereoMIS) demonstrate that ColorGS achieves
state-of-the-art performance, attaining a PSNR of 39.85 (1.5 higher than prior
3DGS-based methods) and superior SSIM (97.25\%) while maintaining real-time
rendering efficiency. Our work advances surgical scene reconstruction by
balancing high fidelity with computational practicality, critical for
intraoperative guidance and AR/VR applications.

### 9. [Class-wise Flooding Regularization for Imbalanced Image Classification](http://arxiv.org/pdf/2508.18723v1)

Authors: Hiroaki Aizawa, Yuta Naito, Kohei Fukuda

The purpose of training neural networks is to achieve high generalization
performance on unseen inputs. However, when trained on imbalanced datasets, a
model's prediction tends to favor majority classes over minority classes,
leading to significant degradation in the recognition performance of minority
classes. To address this issue, we propose class-wise flooding regularization,
an extension of flooding regularization applied at the class level. Flooding is
a regularization technique that mitigates overfitting by preventing the
training loss from falling below a predefined threshold, known as the flooding
level, thereby discouraging memorization. Our proposed method assigns a
class-specific flooding level based on class frequencies. By doing so, it
suppresses overfitting in majority classes while allowing sufficient learning
for minority classes. We validate our approach on imbalanced image
classification. Compared to conventional flooding regularizations, our method
improves the classification performance of minority classes and achieves better
overall generalization.

### 10. [Flatness-aware Curriculum Learning via Adversarial Difficulty](http://arxiv.org/pdf/2508.18726v1)

Authors: Hiroaki Aizawa, Yoshikazu Hayashi

Neural networks trained by empirical risk minimization often suffer from
overfitting, especially to specific samples or domains, which leads to poor
generalization. Curriculum Learning (CL) addresses this issue by selecting
training samples based on the difficulty. From the optimization perspective,
methods such as Sharpness-Aware Minimization (SAM) improve robustness and
generalization by seeking flat minima. However, combining CL with SAM is not
straightforward. In flat regions, both the loss values and the gradient norms
tend to become uniformly small, which makes it difficult to evaluate sample
difficulty and design an effective curriculum. To overcome this problem, we
propose the Adversarial Difficulty Measure (ADM), which quantifies adversarial
vulnerability by leveraging the robustness properties of models trained toward
flat minima. Unlike loss- or gradient-based measures, which become ineffective
as training progresses into flatter regions, ADM remains informative by
measuring the normalized loss gap between original and adversarial examples. We
incorporate ADM into CL-based training with SAM to dynamically assess sample
difficulty. We evaluated our approach on image classification tasks,
fine-grained recognition, and domain generalization. The results demonstrate
that our method preserves the strengths of both CL and SAM while outperforming
existing curriculum-based and flatness-aware training strategies.

### Computers and Society

### 1. [The Hands-Up Problem and How to Deal With It: Secondary School Teachers' Experiences of Debugging in the Classroom](http://arxiv.org/pdf/2508.18861v1)

Authors: Laurie Gale, Sue Sentance

Debugging is a vital but challenging skill for beginner programmers to learn.
It is also a difficult skill to teach. For secondary school teachers, who may
lack time or relevant knowledge, honing students' understanding of debugging
can be a daunting task. Despite this, little research has explored their
perspectives of debugging. To this end, we investigated secondary teachers'
experiences of debugging in the classroom, with a focus on text-based
programming. Through thematic analysis of nine semi-structured interviews, we
identified a common reliance on the teacher for debugging support, often
embodied by many raised hands. We call this phenomenon the `hands-up problem'.
While more experienced and confident teachers discussed strategies they use for
dealing with this, less confident teachers discussed the generally negative
consequences of this problem. We recommend further research into
debugging-specific pedagogical content knowledge and professional development
to help less confident teachers develop counters to the hands-up problem.

### 2. [Of the People, By the Algorithm: How AI Transforms Democratic Representation](http://arxiv.org/pdf/2508.19036v1)

Authors: Yuval Rymon

This review examines how AI technologies are transforming democratic
representation, focusing on citizen participation and algorithmic
decision-making. The analysis reveals that AI technologies are reshaping
democratic processes in fundamental ways: enabling mass-scale deliberation,
changing how citizens access and engage with political information, and
transforming how representatives make and implement decisions. While AI offers
unprecedented opportunities for enhancing democratic participation and
governance efficiency, it also presents significant challenges to democratic
legitimacy and accountability. Social media platforms' AI-driven algorithms
currently mediate much political discourse, creating concerns about information
manipulation and privacy. Large Language Models introduce both epistemic
challenges and potential tools for improving democratic dialogue. The emergence
of Mass Online Deliberation platforms suggests possibilities for scaling up
meaningful citizen participation, while Algorithmic Decision-Making systems
promise more efficient policy implementation but face limitations in handling
complex political trade-offs. As these systems become prevalent,
representatives may assume the role of architects of automated decision
frameworks, responsible for guiding the translation of politically contested
concepts into technical parameters and metrics. Advanced deliberation platforms
offering real-time insights into citizen preferences will challenge traditional
representative independence and discretion to interpret public will. The
institutional integration of these participation mechanisms requires frameworks
that balance the benefits with democratic stability through hybrid systems
weighting different forms of democratic expression.

### 3. [Development of the Measure of Assessment Self-Efficacy (MASE) for Quizzes and Exams](http://arxiv.org/pdf/2508.18631v1)

Authors: Kaitlin Riegel, Tanya Evans, Jason M. Stephens

Self-efficacy is a significant construct in education due to its predictive
relationship with achievement. Existing measures of assessment-related
self-efficacy concentrate on students' beliefs about content-specific tasks but
omit beliefs around assessment-taking. This research aimed to develop and test
the Measure of Assessment Self-Efficacy (MASE), designed to assess two types of
efficacy beliefs related to assessment (i.e., 'comprehension and execution' and
'emotional regulation') in two scenarios (i.e., a low-stakes online quiz and a
high-stakes final exam). Results from confirmatory factor analysis in Study 1
(N = 301) supported the hypothesised two-factor measurement models for both
assessment scenarios. In Study 2, results from MGCFA (N = 277) confirmed these
models were invariant over time and provided evidence for the scales' validity.
Study 3 demonstrated the exam-related MASE was invariant across cohorts of
students (Ns = 277; 329). Potential uses of the developed scales in educational
research are discussed.

### Databases

### 1. [Brook-2PL: Tolerating High Contention Workloads with A Deadlock-Free Two-Phase Locking Protocol](http://arxiv.org/pdf/2508.18576v1)

Authors: Farzad Habibi, Juncheng Fang, Tania Lorido-Botran, Faisal Nawab

The problem of hotspots remains a critical challenge in high-contention
workloads for concurrency control (CC) protocols. Traditional concurrency
control approaches encounter significant difficulties under high contention,
resulting in excessive transaction aborts and deadlocks. In this paper, we
propose Brook-2PL, a novel two-phase locking (2PL) protocol that (1) introduces
SLW-Graph for deadlock-free transaction execution, and (2) proposes partial
transaction chopping for early lock release. Previous methods suffer from
transaction aborts that lead to wasted work and can further burden the system
due to their cascading effects. Brook-2PL addresses this limitation by
statically analyzing a new graph-based dependency structure called SLW-Graph,
enabling deadlock-free two-phase locking through predetermined lock
acquisition. Brook-2PL also reduces contention by enabling early lock release
using partial transaction chopping and static transaction analysis. We overcome
the inherent limitations of traditional transaction chopping by providing a
more flexible chopping method. Evaluation using both our synthetic online game
store workload and the TPC-C benchmark shows that Brook-2PL significantly
outperforms state-of-the-art CC protocols. Brook-2PL achieves an average
speed-up of 2.86x while reducing tail latency (p95) by 48% in the TPC-C
benchmark.

### 2. [Optimal $(α,β)$-Dense Subgraph Search in Bipartite Graphs](http://arxiv.org/pdf/2508.18616v1)

Authors: Yalong Zhang, Rong-Hua Li, Qi Zhang, Guoren Wang

Dense subgraph search in bipartite graphs is a fundamental problem in graph
analysis, with wide-ranging applications in fraud detection, recommendation
systems, and social network analysis. The recently proposed $(\alpha,
\beta)$-dense subgraph model has demonstrated superior capability in capturing
the intrinsic density structure of bipartite graphs compared to existing
alternatives. However, despite its modeling advantages, the $(\alpha,
\beta)$-dense subgraph model lacks efficient support for query processing and
dynamic updates, limiting its practical utility in large-scale applications. To
address these limitations, we propose BD-Index, a novel index that answers
$(\alpha, \beta)$-dense subgraph queries in optimal time while using only
linear space $O(|E|)$, making it well-suited for real-world applications
requiring both fast query processing and low memory consumption. We further
develop two complementary maintenance strategies for dynamic bipartite graphs
to support efficient updates to the BD-Index. The space-efficient strategy
updates the index in time complexity of $O(p \cdot |E|^{1.5})$ per edge
insertion or deletion, while maintaining a low space cost of $O(|E|)$ (the same
as the index itself), where $p$ is typically a small constant in real-world
graphs. In contrast, the time-efficient strategy significantly reduces the
update time to $O(p \cdot |E|)$ per edge update by maintaining auxiliary
orientation structures, at the cost of increased memory usage up to $O(p \cdot
|E|)$. These two strategies provide flexible trade-offs between maintenance
efficiency and memory usage, enabling BD-Index to adapt to diverse application
requirements. Extensive experiments on 10 large-scale real-world datasets
demonstrate high efficiency and scalability of our proposed solutions.

### 3. [WoW: A Window-to-Window Incremental Index for Range-Filtering Approximate Nearest Neighbor Search](http://arxiv.org/pdf/2508.18617v1)

Authors: Ziqi Wang, Jingzhe Zhang, Wei Hu

Given a hybrid dataset where every data object consists of a vector and an
attribute value, for each query with a target vector and a range filter,
range-filtering approximate nearest neighbor search (RFANNS) aims to retrieve
the most similar vectors from the dataset and the corresponding attribute
values fall in the query range. It is a fundamental function in vector database
management systems and intelligent systems with embedding abilities. Dedicated
indices for RFANNS accelerate query speed with an acceptable accuracy loss on
nearest neighbors. However, they are still facing the challenges to be
constructed incrementally and generalized to achieve superior query performance
for arbitrary range filters. In this paper, we introduce a window graph-based
RFANNS index. For incremental construction, we propose an insertion algorithm
to add new vector-attribute pairs into hierarchical window graphs with varying
window size. To handle arbitrary range filters, we optimize relevant window
search for attribute filter checks and vector distance computations by range
selectivity. Extensive experiments on real-world datasets show that for index
construction, the indexing time is on par with the most building-efficient
index, and 4.9x faster than the most query-efficient index with 0.4-0.5x
smaller size; For RFANNS query, it is 4x faster than the most efficient
incremental index, and matches the performance of the best statically-built
index.

### 4. [Enriching Object-Centric Event Data with Process Scopes: A Framework for Aggregation and Analysis](http://arxiv.org/pdf/2508.18830v1)

Authors: Shahrzad Khayatbashi, Majid Rafiei, Jiayuan Chen, Timotheus Kampik, Gregor Berg, Amin Jalali

Object-Centric Process Mining enables the analysis of complex operational
behavior by capturing interactions among multiple business objects (e.g.,
orders, items, deliveries). These interactions are recorded using
Object-Centric Event Data (OCED) formats, such as the Object-Centric Event Log
(OCEL). However, existing formats lack explicit definitions of process scopes,
which restricts analysis to individual processes and limits insights to a low
level of granularity. In practice, OCED often spans multiple interrelated
processes, as shared objects connect events across organizational functions.
This structure reflects how value is created along the organizational value
chain, but introduces challenges for interpretation when process boundaries are
not clearly defined. Moreover, process definitions are typically subjective and
context-dependent; they vary across organizations, roles, and analytical goals,
and cannot always be discovered automatically. To address these challenges, we
propose a method for embedding analyst-defined process scopes into OCEL. This
enables the structured representation of multiple coexisting processes,
supports the aggregation of event data across scopes, and facilitates analysis
at varying levels of abstraction. We demonstrate the applicability of our
approach using a publicly available OCEL log and provide supporting tools for
scope definition and analysis.

### 5. [Rethinking Caching for LLM Serving Systems: Beyond Traditional Heuristics](http://arxiv.org/pdf/2508.18736v1)

Authors: Jungwoo Kim, Minsang Kim, Jaeheon Lee, Chanwoo Moon, Heejin Kim, Taeho Hwang, Woosuk Chung, Yeseong Kim, Sungjin Lee

Serving Large Language Models (LLMs) at scale requires meeting strict Service
Level Objectives (SLOs) under severe computational and memory constraints.
Nevertheless, traditional caching strategies fall short: exact-matching and
prefix caches neglect query semantics, while state-of-the-art semantic caches
remain confined to traditional intuitions, offering little conceptual
departure. Building on this, we present SISO, a semantic caching system that
redefines efficiency for LLM serving. SISO introduces centroid-based caching to
maximize coverage with minimal memory, locality-aware replacement to preserve
high-value entries, and dynamic thresholding to balance accuracy and latency
under varying workloads. Across diverse datasets, SISO delivers up to
1.71$\times$ higher hit ratios and consistently stronger SLO attainment
compared to state-of-the-art systems.

### 6. [Private Quantum Database](http://arxiv.org/pdf/2508.19055v1)

Authors: Giancarlo Gatti, Rihan Hai

Quantum databases open an exciting new frontier in data management by
offering privacy guarantees that classical systems cannot match. Traditional
engines tackle user privacy, which hides the records being queried, or data
privacy, which prevents a user from learning more than she has queried. We
propose a quantum database that protects both by leveraging quantum mechanics:
when the user measures her chosen basis, the superposition collapses and the
unqueried rows become physically inaccessible. We encode relational tables as a
sequence of Quantum Random Access Codes (QRACs) over mutually unbiased bases
(MUBs), transmit a bounded number of quantum states, and let a single,
destructive measurement reconstruct only the selected tuple. This allows us to
preserve data privacy and user privacy at once without trusted hardware or
heavyweight cryptography. Moreover, we envision a novel hybrid
quantum-classical architecture ready for early deployment, which ensures
compatibility with the limitations of today's Noisy Intermediate-Scale Quantum
devices.

### 7. [Text to Query Plans for Question Answering on Large Tables](http://arxiv.org/pdf/2508.18758v1)

Authors: Yipeng Zhang, Chen Wang, Yuzhe Zhang, Jacky Jiang

Efficient querying and analysis of large tabular datasets remain significant
challenges, especially for users without expertise in programming languages
like SQL. Text-to-SQL approaches have shown promising performance on benchmark
data; however, they inherit SQL's drawbacks, including inefficiency with large
datasets and limited support for complex data analyses beyond basic querying.
We propose a novel framework that transforms natural language queries into
query plans. Our solution is implemented outside traditional databases,
allowing us to support classical SQL commands while avoiding SQL's inherent
limitations. Additionally, we enable complex analytical functions, such as
principal component analysis and anomaly detection, providing greater
flexibility and extensibility than traditional SQL capabilities. We leverage
LLMs to iteratively interpret queries and construct operation sequences,
addressing computational complexity by incrementally building solutions. By
executing operations directly on the data, we overcome context length
limitations without requiring the entire dataset to be processed by the model.
We validate our framework through experiments on both standard databases and
large scientific tables, demonstrating its effectiveness in handling extensive
datasets and performing sophisticated data analyses.

### Distributed, Parallel, and Cluster Computing

### 1. [Strata: Hierarchical Context Caching for Long Context Language Model Serving](http://arxiv.org/pdf/2508.18572v1)

Authors: Zhiqiang Xie, Ziyi Xu, Mark Zhao, Yuwei An, Vikram Sharma Mailthody, Scott Mahlke, Michael Garland, Christos Kozyrakis

Large Language Models (LLMs) with expanding context windows face significant
performance hurdles. While caching key-value (KV) states is critical for
avoiding redundant computation, the storage footprint of long-context caches
quickly exceeds GPU memory capacity, forcing production systems to adopt
hierarchical caching across memory hierarchies. However, transferring large
cached contexts back to the GPU introduces severe performance bottlenecks:
fragmented I/O from paged layouts prevents full bandwidth utilization, and
existing schedulers fail to account for cache-loading delays, leaving systems
loading-bound rather than compute-bound. We present Strata, a hierarchical
context caching framework designed for efficient long context LLM serving.
Strata introduces GPU-assisted I/O to combat KV cache fragmentation, decoupling
GPU and CPU memory layouts and employs cache-aware request scheduling to
balance compute with I/O latency and overlapping unavoidable stalls with
complementary tasks. Built on SGLang and deployed in production, Strata
achieves up to 5x lower Time-To-First-Token (TTFT) compared to vLLM + LMCache
and 3.75x speedup over NVIDIA TensorRT-LLM on long-context benchmarks, without
degrading short-context performance.

### 2. [Examining MPI and its Extensions for Asynchronous Multithreaded Communication](http://arxiv.org/pdf/2508.18667v1)

Authors: Jiakun Yan, Marc Snir, Yanfei Guo

The increasing complexity of HPC architectures and the growing adoption of
irregular scientific algorithms demand efficient support for asynchronous,
multithreaded communication. This need is especially pronounced with
Asynchronous Many-Task (AMT) systems. This communication pattern was not a
consideration during the design of the original MPI specification. The MPI
community has recently introduced several extensions to address these evolving
requirements. This work evaluates two such extensions, the Virtual
Communication Interface (VCI) and the Continuation extensions, in the context
of an established AMT runtime HPX. We begin by using an MPI-level
microbenchmark, modeled from HPX's low-level communication mechanism, to
measure the peak performance potential of these extensions. We then integrate
them into HPX to evaluate their effectiveness in real-world scenarios. Our
results show that while these extensions can enhance performance compared to
standard MPI, areas for improvement remain. The current continuation proposal
limits the maximum multithreaded message rate achievable in the multi-VCI
setting. Furthermore, the recommended one-VCI-per-thread mode proves
ineffective in real-world systems due to the attentiveness problem. These
findings underscore the importance of improving intra-VCI threading efficiency
to achieve scalable multithreaded communication and fully realize the benefits
of recent MPI extensions.

### 3. [SIREN: Software Identification and Recognition in HPC Systems](http://arxiv.org/pdf/2508.18950v1)

Authors: Thomas Jakobsche, Fredrik Robertsén, Jessica R. Jones, Utz-Uwe Haus, Florina M. Ciorba

HPC systems use monitoring and operational data analytics to ensure
efficiency, performance, and orderly operations. Application-specific insights
are crucial for analyzing the increasing complexity and diversity of HPC
workloads, particularly through the identification of unknown software and
recognition of repeated executions, which facilitate system optimization and
security improvements. However, traditional identification methods using job or
file names are unreliable for arbitrary user-provided names (a.out). Fuzzy
hashing of executables detects similarities despite changes in executable
version or compilation approach while preserving privacy and file integrity,
overcoming these limitations. We introduce SIREN, a process-level data
collection framework for software identification and recognition. SIREN
improves observability in HPC by enabling analysis of process metadata,
environment information, and executable fuzzy hashes. Findings from a first
opt-in deployment campaign on LUMI show SIREN's ability to provide insights
into software usage, recognition of repeated executions of known applications,
and similarity-based identification of unknown applications.

### 4. [Deep Learning-Enabled Supercritical Flame Simulation at Detailed Chemistry and Real-Fluid Accuracy Towards Trillion-Cell Scale](http://arxiv.org/pdf/2508.18969v1)

Authors: Zhuoqiang Guo, Runze Mao, Lijun Liu, Guangming Tan, Weile Jia, Zhi X. Chen

For decades, supercritical flame simulations incorporating detailed chemistry
and real-fluid transport have been limited to millions of cells, constraining
the resolved spatial and temporal scales of the physical system. We optimize
the supercritical flame simulation software DeepFlame -- which incorporates
deep neural networks while retaining the real-fluid mechanical and chemical
accuracy -- from three perspectives: parallel computing, computational
efficiency, and I/O performance. Our highly optimized DeepFlame achieves
supercritical liquid oxygen/methane (LOX/\ce{CH4}) turbulent combustion
simulation of up to 618 and 154 billion cells with unprecedented
time-to-solution, attaining 439/1186 and 187/316 PFlop/s (32.3\%/21.8\% and
37.4\%/31.8\% of the peak) in FP32/mixed-FP16 precision on Sunway (98,304
nodes) and Fugaku (73,728 nodes) supercomputers, respectively. This
computational capability surpasses existing capacities by three orders of
magnitude, enabling the first practical simulation of rocket engine combustion
with >100 LOX/\ce{CH4} injectors. This breakthrough establishes high-fidelity
supercritical flame modeling as a critical design tool for next-generation
rocket propulsion and ultra-high energy density systems.

### 5. [Federated Fine-Tuning of Sparsely-Activated Large Language Models on Resource-Constrained Devices](http://arxiv.org/pdf/2508.19078v1)

Authors: Fahao Chen, Jie Wan, Peng Li, Zhou Su, Dongxiao Yu

Federated fine-tuning of Mixture-of-Experts (MoE)-based large language models
(LLMs) is challenging due to their massive computational requirements and the
resource constraints of participants. Existing working attempts to fill this
gap through model quantization, computation offloading, or expert pruning.
However, they cannot achieve desired performance due to impractical system
assumptions and a lack of consideration for MoE-specific characteristics. In
this paper, we propose FLUX, a system designed to enable federated fine-tuning
of MoE-based LLMs across participants with constrained computing resources
(e.g., consumer-grade GPUs), aiming to minimize time-to-accuracy. FLUX
introduces three key innovations: (1) quantization-based local profiling to
estimate expert activation with minimal overhead, (2) adaptive layer-aware
expert merging to reduce resource consumption while preserving accuracy, and
(3) dynamic expert role assignment using an exploration-exploitation strategy
to balance tuning and non-tuning experts. Extensive experiments on LLaMA-MoE
and DeepSeek-MoE with multiple benchmark datasets demonstrate that FLUX
significantly outperforms existing methods, achieving up to 4.75X speedup in
time-to-accuracy.

### 6. [History Rhymes: Accelerating LLM Reinforcement Learning with RhymeRL](http://arxiv.org/pdf/2508.18588v1)

Authors: Jingkai He, Tianjian Li, Erhu Feng, Dong Du, Qian Liu, Tao Liu, Yubin Xia, Haibo Chen

With the rapid advancement of large language models (LLMs), reinforcement
learning (RL) has emerged as a pivotal methodology for enhancing the reasoning
capabilities of LLMs. Unlike traditional pre-training approaches, RL
encompasses multiple stages: rollout, reward, and training, which necessitates
collaboration among various worker types. However, current RL systems continue
to grapple with substantial GPU underutilization, due to two primary factors:
(1) The rollout stage dominates the overall RL process due to test-time
scaling; (2) Imbalances in rollout lengths (within the same batch) result in
GPU bubbles. While prior solutions like asynchronous execution and truncation
offer partial relief, they may compromise training accuracy for efficiency.
  Our key insight stems from a previously overlooked observation: rollout
responses exhibit remarkable similarity across adjacent training epochs. Based
on the insight, we introduce RhymeRL, an LLM RL system designed to accelerate
RL training with two key innovations. First, to enhance rollout generation, we
present HistoSpec, a speculative decoding inference engine that utilizes the
similarity of historical rollout token sequences to obtain accurate drafts.
Second, to tackle rollout bubbles, we introduce HistoPipe, a two-tier
scheduling strategy that leverages the similarity of historical rollout
distributions to balance workload among rollout workers. We have evaluated
RhymeRL within a real production environment, demonstrating scalability from
dozens to thousands of GPUs. Experimental results demonstrate that RhymeRL
achieves a 2.6x performance improvement over existing methods, without
compromising accuracy or modifying the RL paradigm.

### 7. [ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive](http://arxiv.org/pdf/2508.18850v1)

Authors: Xinhao Luo, Zihan Liu, Yangjie Zhou, Shihan Fang, Ziyu Huang, Yu Feng, Chen Zhang, Shixuan Sun, Zhenzhe Zheng, Jingwen Leng, Minyi Guo

Large language model (LLM) decoding suffers from high latency due to
fragmented execution across operators and heavy reliance on off-chip memory for
data exchange and reduction. This execution model limits opportunities for
fusion and incurs significant memory traffic and kernel launch overhead. While
modern architectures such as NVIDIA Hopper provide distributed shared memory
and low-latency intra-cluster interconnects, they expose only low-level data
movement instructions, lacking structured abstractions for collective on-chip
communication. To bridge this software-hardware gap, we introduce two
cluster-level communication primitives, ClusterReduce and ClusterGather, which
abstract common communication patterns and enable structured, high-speed data
exchange and reduction between thread blocks within a cluster, allowing
intermediate results to be on-chip without involving off-chip memory. Building
on these abstractions, we design ClusterFusion, an execution framework that
schedules communication and computation jointly to expand operator fusion scope
by composing decoding stages such as QKV Projection, Attention, and Output
Projection into a single fused kernels. Evaluations on H100 GPUs show that
ClusterFusion outperforms state-of-the-art inference frameworks by 1.61x on
average in end-to-end latency across different models and configurations. The
source code is available at https://github.com/xinhao-luo/ClusterFusion.

### 8. [FedProtoKD: Dual Knowledge Distillation with Adaptive Class-wise Prototype Margin for Heterogeneous Federated Learning](http://arxiv.org/pdf/2508.19009v1)

Authors: Md Anwar Hossen, Fatema Siddika, Wensheng Zhang, Anuj Sharma, Ali Jannesari

Heterogeneous Federated Learning (HFL) has gained attention for its ability
to accommodate diverse models and heterogeneous data across clients.
Prototype-based HFL methods emerge as a promising solution to address
statistical heterogeneity and privacy challenges, paving the way for new
advancements in HFL research. This method focuses on sharing only
class-representative prototypes among heterogeneous clients. However, these
prototypes are often aggregated on the server using weighted averaging, leading
to sub-optimal global knowledge; these cause the shrinking of aggregated
prototypes, which negatively affects the model performance in scenarios when
models are heterogeneous and data distributions are extremely non-IID. We
propose FedProtoKD in a Heterogeneous Federated Learning setting, using an
enhanced dual-knowledge distillation mechanism to improve the system
performance with clients' logits and prototype feature representation. We aim
to resolve the prototype margin-shrinking problem using a contrastive
learning-based trainable server prototype by leveraging a class-wise adaptive
prototype margin. Furthermore, we assess the importance of public samples using
the closeness of the sample's prototype to its class representative prototypes,
which enhances learning performance. FedProtoKD achieved average improvements
of 1.13% up to 34.13% accuracy across various settings and significantly
outperforms existing state-of-the-art HFL methods.

### 9. [CARMA: Collocation-Aware Resource Manager with GPU Memory Estimator](http://arxiv.org/pdf/2508.19073v1)

Authors: Ehsan Yousefzadeh-Asl-Miandoab, Reza Karimzadeh, Bulat Ibragimov, Florina M. Ciorba, Pınar Tözün

Studies conducted on enterprise-scale infrastructure have shown that GPUs --
the core computational resource for deep learning (DL) training -- are often
significantly underutilized. DL task collocation on GPUs is an opportunity to
address this challenge. However, it may result in (1) out-of-memory crashes for
the subsequently arriving task and (2) slowdowns for all tasks sharing the GPU
due to resource interference. The former challenge poses a threat to
robustness, while the latter affects the quality of service and energy
efficiency.
  We propose CARMA, a server-scale task-level collocation-aware resource
management system that handles both collocation challenges. CARMA encompasses
GPUMemNet, a novel ML-based GPU memory estimator framework for DL training
tasks, to minimize out-of-memory errors and introduces collocation policies
that cap GPU utilization to minimize interference. Furthermore, CARMA
introduces a recovery method to ensure robust restart of tasks that crash. Our
evaluation on traces modeled after real-world DL training task traces shows
that CARMA increases the GPU utilization over time by 39.3\%, decreases the
end-to-end execution time by $\sim$26.7\%, and reduces the GPU energy use by
$\sim$14.2\%.

### 10. [Ab-initio Quantum Transport with the GW Approximation, 42,240 Atoms, and Sustained Exascale Performance](http://arxiv.org/pdf/2508.19138v1)

Authors: Nicolas Vetsch, Alexander Maeder, Vincent Maillou, Anders Winka, Jiang Cao, Grzegorz Kwasniewski, Leonard Deuschle, Torsten Hoefler, Alexandros Nikolaos Ziogas, Mathieu Luisier

Designing nanoscale electronic devices such as the currently manufactured
nanoribbon field-effect transistors (NRFETs) requires advanced modeling tools
capturing all relevant quantum mechanical effects. State-of-the-art approaches
combine the non-equilibrium Green's function (NEGF) formalism and density
functional theory (DFT). However, as device dimensions do not exceed a few
nanometers anymore, electrons are confined in ultra-small volumes, giving rise
to strong electron-electron interactions. To account for these critical
effects, DFT+NEGF solvers should be extended with the GW approximation, which
massively increases their computational intensity. Here, we present the first
implementation of the NEGF+GW scheme capable of handling NRFET geometries with
dimensions comparable to experiments. This package, called QuaTrEx, makes use
of a novel spatial domain decomposition scheme, can treat devices made of up to
84,480 atoms, scales very well on the Alps and Frontier supercomputers (>80%
weak scaling efficiency), and sustains an exascale FP64 performance on 42,240
atoms (1.15 Eflop/s).

### Digital Libraries

### 1. [Investigating Document Type, Language, Publication Year, and Author Count Discrepancies Between OpenAlex and Web of Science](http://arxiv.org/pdf/2508.18620v1)

Authors: Philippe Mongeon, Madelaine Hare, Poppy Riddle, Summer Wilson, Geoff Krause, Rebecca Marjoram, Rémi Toupin

Bibliometrics, whether used for research or research evaluation, relies on
large multidisciplinary databases of research outputs and citation indices. The
Web of Science (WoS) was the main supporting infrastructure of the field for
more than 30 years until several new competitors emerged. OpenAlex, a
bibliographic database launched in 2022, has distinguished itself for its
openness and extensive coverage. While OpenAlex may reduce or eliminate
barriers to accessing bibliometric data, one of the concerns that hinders its
broader adoption for research and research evaluation is the quality of its
metadata. This study aims to assess metadata quality in OpenAlex and WoS,
focusing on document type, publication year, language, and number of authors.
By addressing discrepancies and misattributions in metadata, this research
seeks to enhance awareness of data quality issues that could impact
bibliometric research and evaluation outcomes.

### 2. [A Bibliometric Analysis of the Scholarly Impact of Early Subaru Telescope-based Publications](http://arxiv.org/pdf/2508.18623v1)

Authors: Hideaki Fujiwara

Bibliometric methods provide valuable tools for assessing scientific
productivity and impact across disciplines, yet their application in astronomy
journals remains relatively limited. This study conducts a bibliometric
analysis of Japanese astronomy publications before and after the commissioning
of the Subaru Telescope, a major national investment in observational
infrastructure. Using data from Scopus and SciVal, we examine peer-reviewed
journal articles published between 1996 and 2007 by authors affiliated with
Japanese institutions, focusing on field-normalized citation indicators such as
the Field-Weighted Citation Impact (FWCI) and the share of publications in the
top 10% most cited globally. Subaru Telescope-based publications are identified
through cross-referencing with official telescope publication lists and are
compared against national and global benchmarks. The results show that Subaru
Telescope-based publications, while accounting for less than 10% of Japan's
total scholarly output in astronomy, consistently achieved FWCI values above
2.0 and a significantly higher proportion of highly cited papers. This
indicates that the Subaru Telescope substantially enhanced Japan's research
visibility and impact, especially during its early operational years. This
study demonstrates the utility of bibliometric evaluation in capturing the
academic return of large-scale research facilities and contributes to broader
discussions on research infrastructure in astronomy.

### Discrete Mathematics

### 1. [Hypergraph Splitting-Off via Element-Connectivity Preserving Reductions](http://arxiv.org/pdf/2508.18637v1)

Authors: Karthekeyan Chandrasekaran, Chandra Chekuri, Shubhang Kulkarni

B\'erczi, Chandrasekaran, Kir\'aly, and Kulkarni (ICALP 2024) recently
described a splitting-off procedure in hypergraphs that preserves
local-connectivity and outlined some applications. In this note we give an
alternative proof via element-connectivity preserving reduction operations in
graphs.

### 2. [Max-Min and 1-Bounded Space Algorithms for the Bin Packing Problem](http://arxiv.org/pdf/2508.18718v1)

Authors: Hiroshi Fujiwara, Rina Atsumi, Hiroaki Yamamoto

In the (1-dimensional) bin packing problem, we are asked to pack all the
given items into bins, each of capacity one, so that the number of non-empty
bins is minimized. Zhu~[Chaos, Solitons \& Fractals 2016] proposed an
approximation algorithm $MM$ that sorts the item sequence in a non-increasing
order by size at the beginning, and then repeatedly packs, into the current
single open bin, first as many of the largest items in the remaining sequence
as possible and then as many of the smallest items in the remaining sequence as
possible. In this paper we prove that the asymptotic approximation ratio of
$MM$ is at most 1.5. Next, focusing on the fact that $MM$ is at the
intersection of two algorithm classes, max-min algorithms and 1-bounded space
algorithms, we comprehensively analyze the theoretical performance bounds of
each subclass derived from the two classes. Our results include a lower bound
of 1.25 for the intersection of the two classes. Furthermore, we extend the
theoretical analysis over algorithm classes to the cardinality constrained bin
packing problem.

### 3. [Reconstructing graphs and their connectivity using graphlets](http://arxiv.org/pdf/2508.19189v1)

Authors: David Hartman, Aneta Pokorná, Daniel Trlifaj

Graphlets are small subgraphs rooted at a fixed vertex. The number of
occurrences of graphlets aligned to a particular vertex, called graphlet degree
sequence, gives a topological description of the surrounding of the analyzed
vertex. In this article, we study properties and uniqueness of graphlet degree
sequences. The information given by graphlets up to size (n-1) is utilized
graphs having certain type of asymmetric vertex-deleted subgraphs. Moreover, we
show a reconstruction of trees from their (<= n-1)-graphlet degree sequences,
which is much easier compared to the standard reconstruction from
vertex-deleted subgraphs.

### Data Structures and Algorithms

### 1. [Graph Traversal via Connected Mobile Agents](http://arxiv.org/pdf/2508.18683v1)

Authors: Saswata Jana, Giuseppe F. Italiano, Partha Sarathi Mandal

This paper considers the Hamiltonian walk problem in the multi-agent
coordination framework, referred to as $k$-agents Hamiltonian walk problem
($k$-HWP). In this problem, a set of $k$ connected agents collectively compute
a spanning walk of a given undirected graph in the minimum steps. At each step,
the agents are at $k$ distinct vertices and the induced subgraph made by the
occupied vertices remains connected. In the next consecutive steps, each agent
may remain stationary or move to one of its neighbours.To the best of our
knowledge, this problem has not been previously explored in the context of
multi-agent systems with connectivity. As a generalization of the well-known
Hamiltonian walk problem (when $k=1$), $k$-HWP is NP-hard. We propose a
$(3-\frac{1}{21})$-approximation algorithm for 2-HWP on arbitrary graphs. For
the tree, we define a restricted version of the problem and present an optimal
algorithm for arbitrary values of $k$. Finally, we formalize the problem for
$k$-uniform hypergraphs and present a $2(1+\ln k)$-approximation algorithm.
This result is also adapted to design an approximation algorithm for $k$-HWP on
general graphs when $k = O(1)$.

### 2. [DTC: Real-Time and Accurate Distributed Triangle Counting in Fully Dynamic Graph Streams](http://arxiv.org/pdf/2508.19057v1)

Authors: Wei Xuan, Yan Liang, Huawei Cao, Ning Lin, Xiaochun Ye, Dongrui Fan

Triangle counting is a fundamental problem in graph mining, essential for
analyzing graph streams with arbitrary edge orders. However, exact counting
becomes impractical due to the massive size of real-world graph streams. To
address this, approximate algorithms have been developed, but existing
distributed streaming algorithms lack adaptability and struggle with edge
deletions. In this article, we propose DTC, a novel family of single-pass
distributed streaming algorithms for global and local triangle counting in
fully dynamic graph streams. Our DTC-AR algorithm accurately estimates triangle
counts without prior knowledge of graph size, leveraging multi-machine
resources. Additionally, we introduce DTC-FD, an algorithm tailored for fully
dynamic graph streams, incorporating edge insertions and deletions. Using
Random Pairing and future edge insertion compensation, DTC-FD achieves unbiased
and accurate approximations across multiple machines. Experimental results
demonstrate significant improvements over baselines. DTC-AR achieves up to
$2029.4\times$ and $27.1\times$ more accuracy, while maintaining the best
trade-off between accuracy and storage space. DTC-FD reduces estimation errors
by up to $32.5\times$ and $19.3\times$, scaling linearly with graph stream
size. These findings highlight the effectiveness of our proposed algorithms in
tackling triangle counting in real-world scenarios. The source code and
datasets are released and available at
\href{https://github.com/wayne4s/srds-dtc.git}{https://github.com/wayne4s/srds-dtc.git}.

### 3. [Hypergraph Splitting-Off via Element-Connectivity Preserving Reductions](http://arxiv.org/pdf/2508.18637v1)

Authors: Karthekeyan Chandrasekaran, Chandra Chekuri, Shubhang Kulkarni

B\'erczi, Chandrasekaran, Kir\'aly, and Kulkarni (ICALP 2024) recently
described a splitting-off procedure in hypergraphs that preserves
local-connectivity and outlined some applications. In this note we give an
alternative proof via element-connectivity preserving reduction operations in
graphs.

### 4. [Max-Min and 1-Bounded Space Algorithms for the Bin Packing Problem](http://arxiv.org/pdf/2508.18718v1)

Authors: Hiroshi Fujiwara, Rina Atsumi, Hiroaki Yamamoto

In the (1-dimensional) bin packing problem, we are asked to pack all the
given items into bins, each of capacity one, so that the number of non-empty
bins is minimized. Zhu~[Chaos, Solitons \& Fractals 2016] proposed an
approximation algorithm $MM$ that sorts the item sequence in a non-increasing
order by size at the beginning, and then repeatedly packs, into the current
single open bin, first as many of the largest items in the remaining sequence
as possible and then as many of the smallest items in the remaining sequence as
possible. In this paper we prove that the asymptotic approximation ratio of
$MM$ is at most 1.5. Next, focusing on the fact that $MM$ is at the
intersection of two algorithm classes, max-min algorithms and 1-bounded space
algorithms, we comprehensively analyze the theoretical performance bounds of
each subclass derived from the two classes. Our results include a lower bound
of 1.25 for the intersection of the two classes. Furthermore, we extend the
theoretical analysis over algorithm classes to the cardinality constrained bin
packing problem.

### Emerging Technologies

### 1. [Architecting Distributed Quantum Computers: Design Insights from Resource Estimation](http://arxiv.org/pdf/2508.19160v1)

Authors: Dmitry Filippov, Peter Yang, Prakash Murali

To enable practically useful quantum computing, we require hundreds to
thousands of logical qubits (collections of physical qubits with error
correction). Current monolithic device architectures have scaling limits beyond
few tens of logical qubits. To scale up, we require architectures that
orchestrate several monolithic devices into a distributed quantum computing
system. Currently, resource estimation, which is crucial for determining
hardware needs and bottlenecks, focuses exclusively on monolithic systems. Our
work fills this gap and answers key architectural design questions about
distributed systems, including the impact of distribution on application
resource needs, the organization of qubits across nodes and the requirements of
entanglement distillation (quantum network). To answer these questions, we
develop a novel resource estimation framework that models the key components of
the distributed execution stack. We analyse the performance of practical
quantum algorithms on various hardware configurations, spanning different qubit
speeds, entanglement generation rates and distillation protocols. We show that
distributed architectures have practically feasible resource requirements; for
a node size of 45K qubits, distributed systems need on average 1.4X higher
number of physical qubits and 4X higher execution time compared to monolithic
architectures, but with more favourable hardware implementation prospects. Our
insights on entanglement generation rates, node sizes and architecture have the
potential to inform system designs in the coming years.

### Formal Languages and Automata Theory

### 1. [CASP: An evaluation dataset for formal verification of C code](http://arxiv.org/pdf/2508.18798v1)

Authors: Niclas Hertzberg, Merlijn Sevenhuijsen, Liv Kåreborn, Anna Lokrantz

Recent developments in Large Language Models (LLMs) have shown promise in
automating code generation, yet the generated programs lack rigorous
correctness guarantees. Formal verification can address this shortcoming, but
requires expertise and is time-consuming to apply. Currently, there is no
dataset of verified C code paired with formal specifications that enables
systematic benchmarking in this space. To fill this gap, we present a curated
evaluation dataset of C code paired with formal specifications written in
ANSI/ISO C Specification Language (ACSL). We develop a multi-stage filtering
process to carefully extract 506 pairs of C code and formal specifications from
The Stack 1 and The Stack 2. We first identify C files annotated with formal
languages. Then, we ensure that the annotated C files formally verify, and
employ LLMs to improve non-verifying files. Furthermore, we post-process the
remaining files into pairs of C code and ACSL specifications, where each
specification-implementation pair is formally verified using Frama-C. To ensure
the quality of the pairs, a manual inspection is conducted to confirm the
correctness of every pair. The resulting dataset of C-ACSL specification pairs
(CASP) provides a foundation for benchmarking and further research on
integrating automated code generation with verified correctness.

### 2. [AS2FM: Enabling Statistical Model Checking of ROS 2 Systems for Robust Autonomy](http://arxiv.org/pdf/2508.18820v1)

Authors: Christian Henkel, Marco Lampacrescia, Michaela Klauck, Matteo Morelli

Designing robotic systems to act autonomously in unforeseen environments is a
challenging task. This work presents a novel approach to use formal
verification, specifically Statistical Model Checking (SMC), to verify system
properties of autonomous robots at design-time. We introduce an extension of
the SCXML format, designed to model system components including both Robot
Operating System 2 (ROS 2) and Behavior Tree (BT) features. Further, we
contribute Autonomous Systems to Formal Models (AS2FM), a tool to translate the
full system model into JANI. The use of JANI, a standard format for
quantitative model checking, enables verification of system properties with
off-the-shelf SMC tools. We demonstrate the practical usability of AS2FM both
in terms of applicability to real-world autonomous robotic control systems, and
in terms of verification runtime scaling. We provide a case study, where we
successfully identify problems in a ROS 2-based robotic manipulation use case
that is verifiable in less than one second using consumer hardware.
Additionally, we compare to the state of the art and demonstrate that our
method is more comprehensive in system feature support, and that the
verification runtime scales linearly with the size of the model, instead of
exponentially.

### 3. [Real-Time Model Checking for Closed-Loop Robot Reactive Planning](http://arxiv.org/pdf/2508.19186v1)

Authors: Christopher Chandler, Bernd Porr, Giulia Lafratta, Alice Miller

We present a new application of model checking which achieves real-time
multi-step planning and obstacle avoidance on a real autonomous robot. We have
developed a small, purpose-built model checking algorithm which generates plans
in situ based on "core" knowledge and attention as found in biological agents.
This is achieved in real-time using no pre-computed data on a low-powered
device. Our approach is based on chaining temporary control systems which are
spawned to counteract disturbances in the local environment that disrupt an
autonomous agent from its preferred action (or resting state). A novel
discretization of 2D LiDAR data sensitive to bounded variations in the local
environment is used. Multi-step planning using model checking by forward
depth-first search is applied to cul-de-sac and playground scenarios. Both
empirical results and informal proofs of two fundamental properties of our
approach demonstrate that model checking can be used to create efficient
multi-step plans for local obstacle avoidance, improving on the performance of
a reactive agent which can only plan one step. Our approach is an instructional
case study for the development of safe, reliable and explainable planning in
the context of autonomous vehicles.

### Graphics

### 1. [SemLayoutDiff: Semantic Layout Generation with Diffusion Model for Indoor Scene Synthesis](http://arxiv.org/pdf/2508.18597v1)

Authors: Xiaohao Sun, Divyam Goel, Angle X. Chang

We present SemLayoutDiff, a unified model for synthesizing diverse 3D indoor
scenes across multiple room types. The model introduces a scene layout
representation combining a top-down semantic map and attributes for each
object. Unlike prior approaches, which cannot condition on architectural
constraints, SemLayoutDiff employs a categorical diffusion model capable of
conditioning scene synthesis explicitly on room masks. It first generates a
coherent semantic map, followed by a cross-attention-based network to predict
furniture placements that respect the synthesized layout. Our method also
accounts for architectural elements such as doors and windows, ensuring that
generated furniture arrangements remain practical and unobstructed. Experiments
on the 3D-FRONT dataset show that SemLayoutDiff produces spatially coherent,
realistic, and varied scenes, outperforming previous methods.

### 2. [PanoHair: Detailed Hair Strand Synthesis on Volumetric Heads](http://arxiv.org/pdf/2508.18944v1)

Authors: Shashikant Verma, Shanmuganathan Raman

Achieving realistic hair strand synthesis is essential for creating lifelike
digital humans, but producing high-fidelity hair strand geometry remains a
significant challenge. Existing methods require a complex setup for data
acquisition, involving multi-view images captured in constrained studio
environments. Additionally, these methods have longer hair volume estimation
and strand synthesis times, which hinder efficiency. We introduce PanoHair, a
model that estimates head geometry as signed distance fields using knowledge
distillation from a pre-trained generative teacher model for head synthesis.
Our approach enables the prediction of semantic segmentation masks and 3D
orientations specifically for the hair region of the estimated geometry. Our
method is generative and can generate diverse hairstyles with latent space
manipulations. For real images, our approach involves an inversion process to
infer latent codes and produces visually appealing hair strands, offering a
streamlined alternative to complex multi-view data acquisition setups. Given
the latent code, PanoHair generates a clean manifold mesh for the hair region
in under 5 seconds, along with semantic and orientation maps, marking a
significant improvement over existing methods, as demonstrated in our
experiments.

### 3. [A Bag of Tricks for Efficient Implicit Neural Point Clouds](http://arxiv.org/pdf/2508.19140v1)

Authors: Florian Hahlbohm, Linus Franke, Leon Overkämping, Paula Wespe, Susana Castillo, Martin Eisemann, Marcus Magnor

Implicit Neural Point Cloud (INPC) is a recent hybrid representation that
combines the expressiveness of neural fields with the efficiency of point-based
rendering, achieving state-of-the-art image quality in novel view synthesis.
However, as with other high-quality approaches that query neural networks
during rendering, the practical usability of INPC is limited by comparatively
slow rendering. In this work, we present a collection of optimizations that
significantly improve both the training and inference performance of INPC
without sacrificing visual fidelity. The most significant modifications are an
improved rasterizer implementation, more effective sampling techniques, and the
incorporation of pre-training for the convolutional neural network used for
hole-filling. Furthermore, we demonstrate that points can be modeled as small
Gaussians during inference to further improve quality in extrapolated, e.g.,
close-up views of the scene. We design our implementations to be broadly
applicable beyond INPC and systematically evaluate each modification in a
series of experiments. Our optimized INPC pipeline achieves up to 25% faster
training, 2x faster rendering, and 20% reduced VRAM usage paired with slight
image quality improvements.

### 4. [LSD-3D: Large-Scale 3D Driving Scene Generation with Geometry Grounding](http://arxiv.org/pdf/2508.19204v1)

Authors: Julian Ost, Andrea Ramazzina, Amogh Joshi, Maximilian Bömer, Mario Bijelic, Felix Heide

Large-scale scene data is essential for training and testing in robot
learning. Neural reconstruction methods have promised the capability of
reconstructing large physically-grounded outdoor scenes from captured sensor
data. However, these methods have baked-in static environments and only allow
for limited scene control -- they are functionally constrained in scene and
trajectory diversity by the captures from which they are reconstructed. In
contrast, generating driving data with recent image or video diffusion models
offers control, however, at the cost of geometry grounding and causality. In
this work, we aim to bridge this gap and present a method that directly
generates large-scale 3D driving scenes with accurate geometry, allowing for
causal novel view synthesis with object permanence and explicit 3D geometry
estimation. The proposed method combines the generation of a proxy geometry and
environment representation with score distillation from learned 2D image
priors. We find that this approach allows for high controllability, enabling
the prompt-guided geometry and high-fidelity texture and structure that can be
conditioned on map layouts -- producing realistic and geometrically consistent
3D generations of complex driving scenes.

### Computer Science and Game Theory

### 1. [Bias-Adjusted LLM Agents for Human-Like Decision-Making via Behavioral Economics](http://arxiv.org/pdf/2508.18600v1)

Authors: Ayato Kitadai, Yusuke Fukasawa, Nariaki Nishino

Large language models (LLMs) are increasingly used to simulate human
decision-making, but their intrinsic biases often diverge from real human
behavior--limiting their ability to reflect population-level diversity. We
address this challenge with a persona-based approach that leverages
individual-level behavioral data from behavioral economics to adjust model
biases. Applying this method to the ultimatum game--a standard but difficult
benchmark for LLMs--we observe improved alignment between simulated and
empirical behavior, particularly on the responder side. While further
refinement of trait representations is needed, our results demonstrate the
promise of persona-conditioned LLMs for simulating human-like decision patterns
at scale.

### Human-Computer Interaction

### 1. [Gamification of Immersive Cervical Rehabilitation Exercises in VR: An Exploratory Study on Chin Tuck and Range of Motion Exercises](http://arxiv.org/pdf/2508.18580v1)

Authors: Haitham Abdelsalam, Chanelle Montpetit, Arash Harirpoush, Maryse Fortin, Yiming Xiao

Chronic neck pain is a prevalent condition that affects millions of
individuals worldwide, causing significant individual suffering and
socioeconomic burdens. Although exercise rehabilitation is a staple in
relieving pain and improving muscle function for the condition, traditional
one-on-one rehabilitation sessions are costly and suffer from poor adherence
and accessibility for the patients. Thanks to the increasing accessibility and
recent advancements in sensing and display technology, virtual reality (VR)
offers the potential to tackle the challenges in traditional exercise
rehabilitation, particularly through gamification. However, still in its
infancy, VR-based neck exercise rehabilitation lacks exploration in effective
gamification strategies and existing prototypes. To address the knowledge gap,
we conduct an exploratory study on the gamification strategies for VR-based
cervical rehabilitation exercises by using chin tuck and neck range of motion
exercises as examples. Specifically, with different game themes, we investigate
a survival and level progression strategy for muscle strengthening (chin tuck)
exercise for the first time, and the suitability of ambient reward for a neck
range of motion exercise. Through a preliminary user study, we assess the
proposed novel VR neck rehabilitation games and they demonstrate excellent
usability, engagement, and perceived health value.

### 2. [Portable Silent Room: Exploring VR Design for Anxiety and Emotion Regulation for Neurodivergent Women and Non-Binary Individuals](http://arxiv.org/pdf/2508.18591v1)

Authors: Kinga Skiers, Yun Suen Pai, Marina Nakagawa, Kouta Minamizawa, Giulia Barbareschi

Neurodivergent individuals, particularly those with Autism and Attention
Deficit Hyperactivity Disorder (ADHD), frequently experience anxiety, panic
attacks, meltdowns, and emotional dysregulation due to societal pressures and
inadequate accommodations. These challenges are especially pronounced for
neurodivergent women and non-binary individuals navigating intersecting
barriers of neurological differences and gender expectations. This research
investigates virtual reality (VR) as a portable safe space for emotional
regulation, addressing challenges of sensory overload and motion sickness while
enhancing relaxation capabilities. Our mixed-methods approach included an
online survey (N=223) and an ideation workshop (N=32), which provided key
design elements for creating effective calming VR environments. Based on these
findings, we developed and iteratively tested VR prototypes with neurodivergent
women and non-binary participants (N=12), leading to a final version offering
enhanced adaptability to individual sensory needs. This final prototype
underwent a comprehensive evaluation with 25 neurodivergent participants to
assess its effectiveness as a regulatory tool. This research contributes to the
development of inclusive, adaptive VR environments that function as
personalized "portable silent rooms" offering neurodivergent individuals
on-demand access to sensory regulation regardless of physical location.

### 3. [Enhancing XAI Interpretation through a Reverse Mapping from Insights to Visualizations](http://arxiv.org/pdf/2508.18640v1)

Authors: Aniket Nuthalapati, Nicholas Hinds, Brian Y. Lim, Qianwen Wang

As AI systems become increasingly integrated into high-stakes domains,
enabling users to accurately interpret model behavior is critical. While AI
explanations can be provided, users often struggle to reason effectively with
these explanations, limiting their ability to validate or learn from AI
decisions. To address this gap, we introduce Reverse Mapping, a novel approach
that enhances visual explanations by incorporating user-derived insights back
into the explanation workflow. Our system extracts structured insights from
free-form user interpretations using a large language model and maps them back
onto visual explanations through interactive annotations and coordinated
multi-view visualizations. Inspired by the verification loop in the
visualization knowledge generation model, this design aims to foster more
deliberate, reflective interaction with AI explanations. We demonstrate our
approach in a prototype system with two use cases and qualitative user
feedback.

### 4. [RÉCITKIT: A Spatial Toolkit for Designing and Evaluating Human-Centered Immersive Data Narratives](http://arxiv.org/pdf/2508.18670v1)

Authors: Vidya Setlur, Samuel Ridet

Spatial computing presents new opportunities for immersive data storytelling,
yet there is limited guidance on how to build such experiences or adapt
traditional narrative visualizations to this medium. We introduce a toolkit,
R\'ECITKIT for supporting spatial data narratives in head-mounted display (HMD)
environments. The toolkit allows developers to create interactive dashboards,
tag data attributes as spatial assets to 3D models and immersive scenes,
generate text and audio narratives, enabling dynamic filtering, and
hierarchical drill-down data discoverability. To demonstrate the utility of the
toolkit, we developed Charles Minard's historical flow map of Napoleon's 1812
campaign in Russia as an immersive experience on Apple Vision Pro. We conducted
a preliminary evaluation with 21 participants that comprised two groups:
developers, who evaluated the toolkit by authoring spatial stories and
consumers, who provided feedback on the Minard app's narrative clarity,
interaction design, and engagement. Feedback highlighted how spatial
interactions and guided narration enhanced insight formation, with participants
emphasizing the benefits of physical manipulation (e.g., gaze, pinch,
navigation) for understanding temporal and geographic data. Participants also
identified opportunities for future enhancement, including improved interaction
affordance visibility, customizable storytelling logic, and integration of
contextual assets to support user orientation. These findings contribute to the
broader discourse on toolkit-driven approaches to immersive data storytelling
across domains such as education, decision support, and exploratory analytics.

### 5. [PRIMMDebug: A Debugging Teaching Aid For Secondary Students](http://arxiv.org/pdf/2508.18875v1)

Authors: Laurie Gale, Sue Sentance

Debugging is often a challenging and infuriating experience for secondary
school students learning their first text-based programming language. Many
students resort to ineffective debugging strategies, making success with
solving errors unlikely and emotional distress common. Developing tools that
encourage students to adopt a more systematic and reflective approach to
debugging is therefore an important, but lacking, area of research. This paper
presents PRIMMDebug, a debugging teaching aid for secondary school students
learning text-based programming. The aid consists of an online tool that takes
students through the steps of a systematic debugging process based on PRIMM, a
framework for teaching programming. The tool promotes a reflective approach to
debugging by heavily encouraging students to articulate their thoughts
throughout the PRIMMDebug process while simultaneously limiting their ability
to run and edit code. To evaluate the tool, a set of students from four
secondary schools were taught with PRIMMDebug over several lessons. Survey
results and log data analysis show that students were generally reluctant to
engage with the systematicity and reflection that the tool encourages. Given
that related work on systematic debugging has reported similar challenges, we
end by considering how these approaches could be refined to help more students
benefit from them.

### 6. [Impact Assessment Card: Communicating Risks and Benefits of AI Uses](http://arxiv.org/pdf/2508.18919v1)

Authors: Edyta Bogucka, Marios Constantinides, Sanja Šćepanović, Daniele Quercia

Communicating the risks and benefits of AI is important for regulation and
public understanding. Yet current methods such as technical reports often
exclude people without technical expertise. Drawing on HCI research, we
developed an Impact Assessment Card to present this information more clearly.
We held three focus groups with a total of 12 participants who helped identify
design requirements and create early versions of the card. We then tested a
refined version in an online study with 235 participants, including AI
developers, compliance experts, and members of the public selected to reflect
the U.S. population by age, sex, and race. Participants used either the card or
a full impact assessment report to write an email supporting or opposing a
proposed AI system. The card led to faster task completion and higher-quality
emails across all groups. We discuss how design choices can improve
accessibility and support AI governance. Examples of cards are available at:
https://social-dynamics.net/ai-risks/impact-card/.

### 7. [Reading minds on the road: decoding perceived risk in automated vehicles through 140K+ ratings](http://arxiv.org/pdf/2508.19121v1)

Authors: Xiaolin He, Zirui Li, Xinwei Wang, Riender Happee, Meng Wang

Perceived risk in automated vehicles (AVs) can create the very danger that
automation is meant to prevent: a frightened rider may hesitate when seconds
matter, misjudge hazards, or disengage. However, measuring how perceived risk
evolves in real time during driving remains challenging, leaving a gap in
decoding such hidden psychological states. Here, we present a novel method to
time-continuously measure and decode perceived risk. We conducted a controlled
experiment where 2,164 participants viewed high-fidelity videos of common
highway driving scenes and provided 141,628 discrete safety ratings. Through
continuous-signal reconstruction of the discrete ratings, we obtained 236 hours
of time-continuous perceived risk data - the largest perceived risk dataset to
date. Leveraging this dataset, we trained deep neural networks that predict
moment-by-moment perceived risk from vehicle kinematics with a mean relative
error below $3\%$. Explainable AI analysis uncovers which factors determine
perceived risk in real time. Our findings demonstrate a new paradigm for
quantifying dynamic passenger experience and psychological constructs in real
time. These findings can guide the design of AVs and other machines that
operate in close proximity to people, adjusting behaviour before trust erodes,
and help realise automation's benefits in transport, healthcare, and service
robotics.

### 8. [Beyond Competitive Gaming: How Casual Players Evaluate and Respond to Teammate Performance](http://arxiv.org/pdf/2508.19230v1)

Authors: Kaushall Senthil Nathan, Jieun Lee, Derrick M. Wang, Geneva M. Smith, Eugene Kukshinov, Daniel Harley, Lennart E. Nacke

Teammate performance evaluation fundamentally shapes intervention design in
video games. However, our current understanding stems primarily from
competitive E-Sports contexts where individual performance directly impacts
outcomes. This research addresses whether performance evaluation mechanisms and
behavioural responses identified in competitive games generalize to casual
cooperative games. We investigated how casual players evaluate teammate
competence and respond behaviourally in a controlled between-subjects
experiment (N=23). We manipulated confederate performance in Overcooked 2,
combining observations, NASA TLX self-reports, and interviews. We present two
key findings. (1) Observations revealed frustration behaviours completely
absent in self-report data. Thus, these instruments assess fundamentally
distinct constructs. (2) Participants consistently evaluated teammate
performance through relative comparison rather than absolute metrics. This
contradicts task-performance operationalizations dominant in competitive gaming
research. Hence, performance evaluation frameworks from competitive contexts
cannot be directly applied to casual cooperative games. We provide empirical
evidence that performance evaluation in casual games requires a comparative
operationalization.

### 9. [Long-Term Variability in Physiological-Arousal Relationships for Robust Emotion Estimation](http://arxiv.org/pdf/2508.18782v1)

Authors: Hiroto Sakimura, Takayuki Nagaya, Tomoki Nishi, Tetsuo Kurahashi, Katsunori Kohda, Nobuhiko Muramoto

Estimating emotional states from physiological signals is a central topic in
affective computing and psychophysiology. While many emotion estimation systems
implicitly assume a stable relationship between physiological features and
subjective affect, this assumption has rarely been tested over long timeframes.
This study investigates whether such relationships remain consistent across
several months within individuals. We developed a custom measurement system and
constructed a longitudinal dataset by collecting physiological signals --
including blood volume pulse, electrodermal activity (EDA), skin temperature,
and acceleration--along with self-reported emotional states from 24
participants over two three-month periods. Data were collected in naturalistic
working environments, allowing analysis of the relationship between
physiological features and subjective arousal in everyday contexts. We examined
how physiological-arousal relationships evolve over time by using Explainable
Boosting Machines (EBMs) to ensure model interpretability. A model trained on
1st-period data showed a 5\% decrease in accuracy when tested on 2nd-period
data, indicating long-term variability in physiological-arousal associations.
EBM-based comparisons further revealed that while heart rate remained a
relatively stable predictor, minimum EDA exhibited substantial individual-level
fluctuations between periods. While the number of participants is limited,
these findings highlight the need to account for temporal variability in
physiological-arousal relationships and suggest that emotion estimation models
should be periodically updated -- e.g., every five months -- based on observed
shift trends to maintain robust performance over time.

### 10. [Insights into User Interface Innovations from a Design Thinking Workshop at deRSE25](http://arxiv.org/pdf/2508.18784v1)

Authors: Maximilian Frank, Simon Lund

Large Language Models have become widely adopted tools due to their versatile
capabilities, yet their user interfaces remain limited, often following rigid,
linear interaction paradigms. In this paper, we present insights from a design
thinking workshop held at the deRSE25 conference aiming at collaboratively
developing innovative user interface concepts for LLMs. During the workshop,
participants identified common use cases, evaluated the strengths and
shortcomings of current LLM interfaces, and created visualizations of new
interaction concepts emphasizing flexible context management, dynamic
conversation branching, and enhanced mechanisms for user control. We describe
how these participant-generated ideas advanced our own whiteboard-based UI
approach. The ongoing development of this interface is guided by the
human-centered design process - an iterative, user-focused methodology that
emphasizes continuous refinement through user feedback. Broader implications
for future LLM interface development are discussed, advocating for increased
attention to UI innovation grounded in user-centered design principles.

### Information Retrieval

### 1. [Extracting Information from Scientific Literature via Visual Table Question Answering Models](http://arxiv.org/pdf/2508.18661v1)

Authors: Dongyoun Kim, Hyung-do Choi, Youngsun Jang, John Kim

This study explores three approaches to processing table data in scientific
papers to enhance extractive question answering and develop a software tool for
the systematic review process. The methods evaluated include: (1) Optical
Character Recognition (OCR) for extracting information from documents, (2)
Pre-trained models for document visual question answering, and (3) Table
detection and structure recognition to extract and merge key information from
tables with textual content to answer extractive questions. In exploratory
experiments, we augmented ten sample test documents containing tables and
relevant content against RF- EMF-related scientific papers with seven
predefined extractive question-answer pairs. The results indicate that
approaches preserving table structure outperform the others, particularly in
representing and organizing table content. Accurately recognizing specific
notations and symbols within the documents emerged as a critical factor for
improved results. Our study concludes that preserving the structural integrity
of tables is essential for enhancing the accuracy and reliability of extractive
question answering in scientific documents.

### 2. [Taming the One-Epoch Phenomenon in Online Recommendation System by Two-stage Contrastive ID Pre-training](http://arxiv.org/pdf/2508.18700v1)

Authors: Yi-Ping Hsu, Po-Wei Wang, Chantat Eksombatchai, Jiajing Xu

ID-based embeddings are widely used in web-scale online recommendation
systems. However, their susceptibility to overfitting, particularly due to the
long-tail nature of data distributions, often limits training to a single
epoch, a phenomenon known as the "one-epoch problem." This challenge has driven
research efforts to optimize performance within the first epoch by enhancing
convergence speed or feature sparsity. In this study, we introduce a novel
two-stage training strategy that incorporates a pre-training phase using a
minimal model with contrastive loss, enabling broader data coverage for the
embedding system. Our offline experiments demonstrate that multi-epoch training
during the pre-training phase does not lead to overfitting, and the resulting
embeddings improve online generalization when fine-tuned for more complex
downstream recommendation tasks. We deployed the proposed system in live
traffic at Pinterest, achieving significant site-wide engagement gains.

### 3. [Optimization of Latent-Space Compression using Game-Theoretic Techniques for Transformer-Based Vector Search](http://arxiv.org/pdf/2508.18877v1)

Authors: Kushagra Agrawal, Nisharg Nargund, Oishani Banerjee

Vector similarity search plays a pivotal role in modern information retrieval
systems, especially when powered by transformer-based embeddings. However, the
scalability and efficiency of such systems are often hindered by the high
dimensionality of latent representations. In this paper, we propose a novel
game-theoretic framework for optimizing latent-space compression to enhance
both the efficiency and semantic utility of vector search. By modeling the
compression strategy as a zero-sum game between retrieval accuracy and storage
efficiency, we derive a latent transformation that preserves semantic
similarity while reducing redundancy. We benchmark our method against FAISS, a
widely-used vector search library, and demonstrate that our approach achieves a
significantly higher average similarity (0.9981 vs. 0.5517) and utility (0.8873
vs. 0.5194), albeit with a modest increase in query time. This trade-off
highlights the practical value of game-theoretic latent compression in
high-utility, transformer-based search applications. The proposed system can be
seamlessly integrated into existing LLM pipelines to yield more semantically
accurate and computationally efficient retrieval.

### 4. [Membership Inference Attacks on LLM-based Recommender Systems](http://arxiv.org/pdf/2508.18665v1)

Authors: Jiajie He, Yuechun Gu, Min-Chun Chen, Keke Chen

Large language models (LLMs) based Recommender Systems (RecSys) can flexibly
adapt recommendation systems to different domains. It utilizes in-context
learning (ICL), i.e., the prompts, to customize the recommendation functions,
which include sensitive historical user-specific item interactions, e.g.,
implicit feedback like clicked items or explicit product reviews. Such private
information may be exposed to novel privacy attack. However, no study has been
done on this important issue. We design four membership inference attacks
(MIAs), aiming to reveal whether victims' historical interactions have been
used by system prompts. They are \emph{direct inquiry, hallucination,
similarity, and poisoning attacks}, each of which utilizes the unique features
of LLMs or RecSys. We have carefully evaluated them on three LLMs that have
been used to develop ICL-LLM RecSys and two well-known RecSys benchmark
datasets. The results confirm that the MIA threat on LLM RecSys is realistic:
direct inquiry and poisoning attacks showing significantly high attack
advantages. We have also analyzed the factors affecting these attacks, such as
the number of shots in system prompts and the position of the victim in the
shots.

### Machine Learning

### 1. [Linear Trading Position with Sparse Spectrum](http://arxiv.org/pdf/2508.18596v1)

Authors: Zhao-Rong Lai, Haisheng Yang

The principal portfolio approach is an emerging method in signal-based
trading. However, these principal portfolios may not be diversified to explore
the key features of the prediction matrix or robust to different situations. To
address this problem, we propose a novel linear trading position with sparse
spectrum that can explore a larger spectral region of the prediction matrix. We
also develop a Krasnosel'ski\u \i-Mann fixed-point algorithm to optimize this
trading position, which possesses the descent property and achieves a linear
convergence rate in the objective value. This is a new theoretical result for
this type of algorithms. Extensive experiments show that the proposed method
achieves good and robust performance in various situations.

### 2. [STRATA-TS: Selective Knowledge Transfer for Urban Time Series Forecasting with Retrieval-Guided Reasoning](http://arxiv.org/pdf/2508.18635v1)

Authors: Yue Jiang, Chenxi Liu, Yile Chen, Qin Chao, Shuai Liu, Gao Cong

Urban forecasting models often face a severe data imbalance problem: only a
few cities have dense, long-span records, while many others expose short or
incomplete histories. Direct transfer from data-rich to data-scarce cities is
unreliable because only a limited subset of source patterns truly benefits the
target domain, whereas indiscriminate transfer risks introducing noise and
negative transfer. We present STRATA-TS (Selective TRAnsfer via TArget-aware
retrieval for Time Series), a framework that combines domain-adapted retrieval
with reasoning-capable large models to improve forecasting in scarce data
regimes. STRATA-TS employs a patch-based temporal encoder to identify source
subsequences that are semantically and dynamically aligned with the target
query. These retrieved exemplars are then injected into a retrieval-guided
reasoning stage, where an LLM performs structured inference over target inputs
and retrieved support. To enable efficient deployment, we distill the reasoning
process into a compact open model via supervised fine-tuning. Extensive
experiments on three parking availability datasets across Singapore,
Nottingham, and Glasgow demonstrate that STRATA-TS consistently outperforms
strong forecasting and transfer baselines, while providing interpretable
knowledge transfer pathways.

### 3. [Utilizing Training Data to Improve LLM Reasoning for Tabular Understanding](http://arxiv.org/pdf/2508.18676v1)

Authors: Chufan Gao, Jintai Chen, Jimeng Sun

Automated tabular understanding and reasoning are essential tasks for data
scientists. Recently, Large language models (LLMs) have become increasingly
prevalent in tabular reasoning tasks. Previous work focuses on (1) finetuning
LLMs using labeled data or (2) Training-free prompting LLM agents using
chain-of-thought (CoT). Finetuning offers dataset-specific learning at the cost
of generalizability. Training-free prompting is highly generalizable but does
not take full advantage of training data. In this paper, we propose a novel
prompting-based reasoning approach, Learn then Retrieve: LRTab, which
integrates the benefits of both by retrieving relevant information learned from
training data. We first use prompting to obtain CoT responses over the training
data. For incorrect CoTs, we prompt the LLM to predict Prompt Conditions to
avoid the error, learning insights from the data. We validate the effectiveness
of Prompt Conditions using validation data. Finally, at inference time, we
retrieve the most relevant Prompt Conditions for additional context for table
understanding. We provide comprehensive experiments on WikiTQ and Tabfact,
showing that LRTab is interpretable, cost-efficient, and can outperform
previous baselines in tabular reasoning.

### 4. [End to End Autoencoder MLP Framework for Sepsis Prediction](http://arxiv.org/pdf/2508.18688v1)

Authors: Hejiang Cai, Di Wu, Ji Xu, Xiang Liu, Yiziting Zhu, Xin Shu, Yujie Li, Bin Yi

Sepsis is a life threatening condition that requires timely detection in
intensive care settings. Traditional machine learning approaches, including
Naive Bayes, Support Vector Machine (SVM), Random Forest, and XGBoost, often
rely on manual feature engineering and struggle with irregular, incomplete
time-series data commonly present in electronic health records. We introduce an
end-to-end deep learning framework integrating an unsupervised autoencoder for
automatic feature extraction with a multilayer perceptron classifier for binary
sepsis risk prediction. To enhance clinical applicability, we implement a
customized down sampling strategy that extracts high information density
segments during training and a non-overlapping dynamic sliding window mechanism
for real-time inference. Preprocessed time series data are represented as fixed
dimension vectors with explicit missingness indicators, mitigating bias and
noise. We validate our approach on three ICU cohorts. Our end-to-end model
achieves accuracies of 74.6 percent, 80.6 percent, and 93.5 percent,
respectively, consistently outperforming traditional machine learning
baselines. These results demonstrate the framework's superior robustness,
generalizability, and clinical utility for early sepsis detection across
heterogeneous ICU environments.

### 5. [Stability and Generalization for Bellman Residuals](http://arxiv.org/pdf/2508.18741v1)

Authors: Enoch H. Kang, Kyoungseok Jang

Offline reinforcement learning and offline inverse reinforcement learning aim
to recover near-optimal value functions or reward models from a fixed batch of
logged trajectories, yet current practice still struggles to enforce Bellman
consistency. Bellman residual minimization (BRM) has emerged as an attractive
remedy, as a globally convergent stochastic gradient descent-ascent based
method for BRM has been recently discovered. However, its statistical behavior
in the offline setting remains largely unexplored. In this paper, we close this
statistical gap. Our analysis introduces a single Lyapunov potential that
couples SGDA runs on neighbouring datasets and yields an O(1/n) on-average
argument-stability bound-doubling the best known sample-complexity exponent for
convex-concave saddle problems. The same stability constant translates into the
O(1/n) excess risk bound for BRM, without variance reduction, extra
regularization, or restrictive independence assumptions on minibatch sampling.
The results hold for standard neural-network parameterizations and minibatch
SGD.

### 6. [Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear programming](http://arxiv.org/pdf/2508.18742v1)

Authors: Jiajun Li, Ran Hou, Yu Ding, Yixuan Li, Shisi Guan, Jiahui Duan, Xiongwei Han, Tao Zhong, Vincent Chau, Weiwei Wu, Wanyuan Wang

Model reduction, which aims to learn a simpler model of the original mixed
integer linear programming (MILP), can solve large-scale MILP problems much
faster. Most existing model reduction methods are based on variable reduction,
which predicts a solution value for a subset of variables. From a dual
perspective, constraint reduction that transforms a subset of inequality
constraints into equalities can also reduce the complexity of MILP, but has
been largely ignored. Therefore, this paper proposes a novel constraint-based
model reduction approach for the MILP. Constraint-based MILP reduction has two
challenges: 1) which inequality constraints are critical such that reducing
them can accelerate MILP solving while preserving feasibility, and 2) how to
predict these critical constraints efficiently. To identify critical
constraints, we first label these tight-constraints at the optimal solution as
potential critical constraints and design a heuristic rule to select a subset
of critical tight-constraints. To learn the critical tight-constraints, we
propose a multi-modal representation technique that leverages information from
both instance-level and abstract-level MILP formulations. The experimental
results show that, compared to the state-of-the-art methods, our method
improves the quality of the solution by over 50\% and reduces the computation
time by 17.47\%.

### 7. [UltraMemV2: Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning](http://arxiv.org/pdf/2508.18756v1)

Authors: Zihao Huang, Yu Bao, Qiyang Min, Siyan Chen, Ran Guo, Hongzhi Huang, Defa Zhu, Yutao Zeng, Banggu Wu, Xun Zhou, Siyuan Qiao

While Mixture of Experts (MoE) models achieve remarkable efficiency by
activating only subsets of parameters, they suffer from high memory access
costs during inference. Memory-layer architectures offer an appealing
alternative with very few memory access, but previous attempts like UltraMem
have only matched the performance of 2-expert MoE models, falling significantly
short of state-of-the-art 8-expert configurations. We present UltraMemV2, a
redesigned memory-layer architecture that closes this performance gap. Our
approach introduces five key improvements: integrating memory layers into every
transformer block, simplifying value expansion with single linear projections,
adopting FFN-based value processing from PEER, implementing principled
parameter initialization, and rebalancing memory-to-FFN computation ratios.
Through extensive evaluation, we demonstrate that UltraMemV2 achieves
performance parity with 8-expert MoE models under same computation and
parameters but significantly low memory access. Notably, UltraMemV2 shows
superior performance on memory-intensive tasks, with improvements of +1.6
points on long-context memorization, +6.2 points on multi-round memorization,
and +7.9 points on in-context learning. We validate our approach at scale with
models up to 2.5B activated parameters from 120B total parameters, and
establish that activation density has greater impact on performance than total
sparse parameter count. Our work brings memory-layer architectures to
performance parity with state-of-the-art MoE models, presenting a compelling
alternative for efficient sparse computation.

### 8. [Governance-as-a-Service: A Multi-Agent Framework for AI System Compliance and Policy Enforcement](http://arxiv.org/pdf/2508.18765v1)

Authors: Helen Pervez, Suyash Gaurav, Jukka Heikkonen, Jatin Chaudhary

As AI systems evolve into distributed ecosystems with autonomous execution,
asynchronous reasoning, and multi-agent coordination, the absence of scalable,
decoupled governance poses a structural risk. Existing oversight mechanisms are
reactive, brittle, and embedded within agent architectures, making them
non-auditable and hard to generalize across heterogeneous deployments.
  We introduce Governance-as-a-Service (GaaS): a modular, policy-driven
enforcement layer that regulates agent outputs at runtime without altering
model internals or requiring agent cooperation. GaaS employs declarative rules
and a Trust Factor mechanism that scores agents based on compliance and
severity-weighted violations. It enables coercive, normative, and adaptive
interventions, supporting graduated enforcement and dynamic trust modulation.
  To evaluate GaaS, we conduct three simulation regimes with open-source models
(LLaMA3, Qwen3, DeepSeek-R1) across content generation and financial
decision-making. In the baseline, agents act without governance; in the second,
GaaS enforces policies; in the third, adversarial agents probe robustness. All
actions are intercepted, evaluated, and logged for analysis. Results show that
GaaS reliably blocks or redirects high-risk behaviors while preserving
throughput. Trust scores track rule adherence, isolating and penalizing
untrustworthy components in multi-agent systems.
  By positioning governance as a runtime service akin to compute or storage,
GaaS establishes infrastructure-level alignment for interoperable agent
ecosystems. It does not teach agents ethics; it enforces them.

### 9. [Predicting Drug-Drug Interactions Using Heterogeneous Graph Neural Networks: HGNN-DDI](http://arxiv.org/pdf/2508.18766v1)

Authors: Hongbo Liu, Siyi Li, Zheng Yu

Drug-drug interactions (DDIs) are a major concern in clinical practice, as
they can lead to reduced therapeutic efficacy or severe adverse effects.
Traditional computational approaches often struggle to capture the complex
relationships among drugs, targets, and biological entities. In this work, we
propose HGNN-DDI, a heterogeneous graph neural network model designed to
predict potential DDIs by integrating multiple drug-related data sources.
HGNN-DDI leverages graph representation learning to model heterogeneous
biomedical networks, enabling effective information propagation across diverse
node and edge types. Experimental results on benchmark DDI datasets demonstrate
that HGNN-DDI outperforms state-of-the-art baselines in prediction accuracy and
robustness, highlighting its potential to support safer drug development and
precision medicine.

### 10. [Recycling History: Efficient Recommendations from Contextual Dueling Bandits](http://arxiv.org/pdf/2508.18841v1)

Authors: Suryanarayana Sankagiri, Jalal Etesami, Pouria Fatemi, Matthias Grossglauser

The contextual duelling bandit problem models adaptive recommender systems,
where the algorithm presents a set of items to the user, and the user's choice
reveals their preference. This setup is well suited for implicit choices users
make when navigating a content platform, but does not capture other possible
comparison queries. Motivated by the fact that users provide more reliable
feedback after consuming items, we propose a new bandit model that can be
described as follows. The algorithm recommends one item per time step; after
consuming that item, the user is asked to compare it with another item chosen
from the user's consumption history. Importantly, in our model, this comparison
item can be chosen without incurring any additional regret, potentially leading
to better performance. However, the regret analysis is challenging because of
the temporal dependency in the user's history. To overcome this challenge, we
first show that the algorithm can construct informative queries provided the
history is rich, i.e., satisfies a certain diversity condition. We then show
that a short initial random exploration phase is sufficient for the algorithm
to accumulate a rich history with high probability. This result, proven via
matrix concentration bounds, yields $O(\sqrt{T})$ regret guarantees.
Additionally, our simulations show that reusing past items for comparisons can
lead to significantly lower regret than only comparing between simultaneously
recommended items.

### Neural and Evolutionary Computing

### 1. [A note on Cybenko's Universal Approximation Theorem](http://arxiv.org/pdf/2508.18893v1)

Authors: Kun Wang

In this short note, we point out a mistake in G.Cybenko's proof of his
version of the universal approximation theorem which has been widely cited.
This mistake might not be easily fixable along the idea of his proof and it
also leads to an interesting question in measure theory.

### 2. [Leveraging Evolutionary Surrogate-Assisted Prescription in Multi-Objective Chlorination Control Systems](http://arxiv.org/pdf/2508.19173v1)

Authors: Rivaaj Monsia, Olivier Francon, Daniel Young, Risto Miikkulainen

This short, written report introduces the idea of Evolutionary
Surrogate-Assisted Prescription (ESP) and presents preliminary results on its
potential use in training real-world agents as a part of the 1st AI for
Drinking Water Chlorination Challenge at IJCAI-2025. This work was done by a
team from Project Resilience, an organization interested in bridging AI to
real-world problems.

### 3. [Metric Matters: A Formal Evaluation of Similarity Measures in Active Learning for Cyber Threat Intelligence](http://arxiv.org/pdf/2508.19019v1)

Authors: Sidahmed Benabderrahmane, Talal Rahwan

Advanced Persistent Threats (APTs) pose a severe challenge to cyber defense
due to their stealthy behavior and the extreme class imbalance inherent in
detection datasets. To address these issues, we propose a novel active
learning-based anomaly detection framework that leverages similarity search to
iteratively refine the decision space. Built upon an Attention-Based
Autoencoder, our approach uses feature-space similarity to identify normal-like
and anomaly-like instances, thereby enhancing model robustness with minimal
oracle supervision. Crucially, we perform a formal evaluation of various
similarity measures to understand their influence on sample selection and
anomaly ranking effectiveness. Through experiments on diverse datasets,
including DARPA Transparent Computing APT traces, we demonstrate that the
choice of similarity metric significantly impacts model convergence, anomaly
detection accuracy, and label efficiency. Our results offer actionable insights
for selecting similarity functions in active learning pipelines tailored for
threat intelligence and cyber defense.

### Networking and Internet Architecture

### 1. [Network Calculus Results for TSN: An Introduction](http://arxiv.org/pdf/2508.18855v1)

Authors: Lisa Maile, Kai-Steffen Hielscher, Reinhard German

Time-Sensitive Networking (TSN) is a set of standards that enables the
industry to provide real-time guarantees for time-critical communications with
Ethernet hardware. TSN supports various queuing and scheduling mechanisms and
allows the integration of multiple traffic types in a single network. Network
Calculus (NC) can be used to calculate upper bounds for latencies and buffer
sizes within these networks, for example, for safety or real-time traffic. We
explain the relevance of NC for TSN-based computer communications and potential
areas of application. Different NC analysis approaches have been published to
examine different parts of TSN and this paper provides a survey of these
publications and presents their main results, dependencies, and differences. We
present a consistent presentation of the most important results and suggest an
improvement to model the output of sending end-devices. To ease access to the
current research status, we introduce a common notation to show how all results
depend on each other and also identify common assumptions. Thus, we offer a
comprehensive overview of NC for industrial networks and identify possible
areas for future work.

### 2. [Saving Energy with Relaxed Latency Constraints: A Study on Data Compression and Communication](http://arxiv.org/pdf/2508.18863v1)

Authors: Pietro Talli, Anup Mishra, Federico Chiariotti, Israel Leyva-Mayorga, Andrea Zanella, Petar Popovski

With the advent of edge computing, data generated by end devices can be
pre-processed before transmission, possibly saving transmission time and
energy. On the other hand, data processing itself incurs latency and energy
consumption, depending on the complexity of the computing operations and the
speed of the processor. The energy-latency-reliability profile resulting from
the concatenation of pre-processing operations (specifically, data compression)
and data transmission is particularly relevant in wireless communication
services, whose requirements may change dramatically with the application
domain. In this paper, we study this multi-dimensional optimization problem,
introducing a simple model to investigate the tradeoff among end-to-end
latency, reliability, and energy consumption when considering compression and
communication operations in a constrained wireless device. We then study the
Pareto fronts of the energy-latency trade-off, considering data compression
ratio and device processing speed as key design variables. Our results show
that the energy costs grows exponentially with the reduction of the end-to-end
latency, so that considerable energy saving can be obtained by slightly
relaxing the latency requirements of applications. These findings challenge
conventional rigid communication latency targets, advocating instead for
application-specific end-to-end latency budgets that account for computational
and transmission overhead.

### 3. [Combining Static and Dynamic Traffic with Delay Guarantees in Time-Sensitive Networking](http://arxiv.org/pdf/2508.18883v1)

Authors: Lisa Maile, Kai-Steffen Hielscher, Reinhard German

To support reliable and low-latency communication, Time-Sensitive Networking
introduced protocols and interfaces for resource allocation in Ethernet.
However, the implementation of these allocation algorithms has not yet been
covered by the standards. Our work focuses on deadline-guaranteeing resource
allocation for networks with static and dynamic traffic. To achieve this, we
combine offline network optimization heuristics with online admission control
and, thus, allow for new flow registrations while the network is running. We
demonstrate our solution on Credit-Based Shaper networks by using the delay
analysis framework Network Calculus. We compare our approach with an intuitive
and a brute-force algorithm, where we can achieve significant improvements,
both, in terms of quality and runtime. Thereby, our results show that we can
guarantee maximum end-to-end delays and also increase the flexibility of the
network while requiring only minimal user input.

### 4. [Adaptive 6G Networks-in-Network Management for Industrial Applications](http://arxiv.org/pdf/2508.18902v1)

Authors: Daniel Lindenschmitt, Paul Seehofer, Marius Schmitz, Jan Mertes, Roland Bless, Martina Zitterbart, Jan C. Aurich, Hans D. Schotten

This paper presents the application of Dynamic Spectrum Management (DSM) for
future 6G industrial networks, establishing an efficient controller for the
Networks-in-Network (NiN) concept. The proposed architecture integrates nomadic
as well as static sub-networks (SNs with diverse Quality of Service (QoS)
requirements within the coverage area of an overlayer network, managed by a
centralized spectrum manager (SM). Control plane connectivity between the SNs
and the DSM is ensured by the self-organizing KIRA routing protocol. The
demonstrated system enables scalable, zero-touch connectivity and supports
nomadic SNs through seamless discovery and reconfiguration. SNs are implemented
for modular Industrial Internet of Things (IIoT) scenarios, as well as for
mission-critical control loops and for logistics or nomadic behavior. The DSM
framework dynamically adapts spectrum allocation to meet real-time demands
while ensuring reliable operation. The demonstration highlights the potential
of DSM and NiNs to support flexible, dense, and heterogeneous wireless
deployments in reconfigurable manufacturing environments.

### 5. [LeoTCP: Low-Latency and High-Throughput Data Transport for LEO Satellite Networks](http://arxiv.org/pdf/2508.19067v1)

Authors: Aiden Valentine, George Parisis

Low-Earth Orbit (LEO) satellite networks consist of thousands of satellites
orbiting the Earth, enabling low-latency and high-throughput communications
across the globe. Such networks present unprecedented challenges due to their
dynamic nature, which state-of-the-art data transport protocols do not address.
These challenges include: (1) non-congestive latency variation and loss, caused
by continuous satellite movement and fluctuating link quality due to weather
effects; (2) transient hotspots leading to buffer build-up, latency inflation,
and potential packet loss; and (3) frequent handovers, which may result in
temporary connectivity loss and re-routing through paths with unknown
congestion and delay characteristics. In this paper, we introduce LeoTCP, a
novel data transport protocol designed specifically to address these
challenges. LeoTCP leverages in-network telemetry (INT) to gather congestion
information on a per-hop basis. Using this information, LeoTCP (1) minimises
both buffer occupancy and latency for end users, (2) maximises application
throughput and network utilisation, and (3) swiftly reacts to network hotspots.
We compare LeoTCP to state-of-the-art data transport protocols using a LEO
satellite simulation model and targeted micro-benchmarks, both based on
OMNeT++/INET. The simulation model captures RTT dynamics in a simulated LEO
satellite constellation, while the micro-benchmarks isolate key LEO-specific
characteristics, including non-congestive latency variation and loss, path
changes, and congestion hotspots. Our results demonstrate that LeoTCP
significantly increases goodput compared to existing state-of-the-art
approaches, while simultaneously minimising latency.

### 6. [Sharing is Caring: Analysis of Hybrid Network Sharing Strategies for Energy Efficient Multi-Operator Cellular Systems](http://arxiv.org/pdf/2508.19130v1)

Authors: Laura Finarelli, Maoquan Ni, Michela Meo, Falko Dressler, Gianluca Rizzo

This paper introduces a novel analytical framework for evaluating
energy-efficient, QoS-aware network-sharing strategies in cellular networks.
Leveraging stochastic geometry, our framework enables the systematic assessment
of network performance across a range of sharing paradigms, including both
conventional single-operator scenarios and advanced hybrid strategies that
enable full integration and cooperation among multiple mobile network
operators. Our framework incorporates diverse user densities, rate
requirements, and energy consumption models to ensure comprehensive analysis.
Applying our results to real-world datasets from French mobile network
operators, we demonstrate that hybrid network sharing can yield substantial
energy savings, up to $35\%$, while maintaining quality of service.
Furthermore, our results allow us to characterizing how the benefits of network
sharing vary as a function of the geographical and functional characteristics
of the deployment area. These findings highlight the potential of collaborative
sharing strategies to enhance operational efficiency and sustainability in
next-generation cellular networks.

### 7. [Dynamic Trajectory Optimization and Power Control for Hierarchical UAV Swarms in 6G Aerial Access Network](http://arxiv.org/pdf/2508.18702v1)

Authors: Ziye Jia, Jia He, Lijun He, Min Sheng, Junyu Liu, Qihui Wu, Zhu Han

Unmanned aerial vehicles (UAVs) can serve as aerial base stations (BSs) to
extend the ubiquitous connectivity for ground users (GUs) in the
sixth-generation (6G) era. However, it is challenging to cooperatively deploy
multiple UAV swarms in large-scale remote areas. Hence, in this paper, we
propose a hierarchical UAV swarms structure for 6G aerial access networks,
where the head UAVs serve as aerial BSs, and tail UAVs (T-UAVs) are responsible
for relay. In detail, we jointly optimize the dynamic deployment and trajectory
of UAV swarms, which is formulated as a multi-objective optimization problem
(MOP) to concurrently minimize the energy consumption of UAV swarms and GUs, as
well as the delay of GUs. However, the proposed MOP is a mixed integer
nonlinear programming and NP-hard to solve. Therefore, we develop a K-means and
Voronoi diagram based area division method, and construct Fermat points to
establish connections between GUs and T-UAVs. Then, an improved non-dominated
sorting whale optimization algorithm is proposed to seek Pareto optimal
solutions for the transformed MOP. Finally, extensive simulations are conducted
to verify the performance of proposed algorithms by comparing with baseline
mechanisms, resulting in a 50% complexity reduction.

### 8. [A Survey on Cloud-Edge-Terminal Collaborative Intelligence in AIoT Networks](http://arxiv.org/pdf/2508.18803v1)

Authors: Jiaqi Wu, Jing Liu, Yang Liu, Lixu Wang, Zehua Wang, Wei Chen, Zijian Tian, Richard Yu, Victor C. M. Leung

The proliferation of Internet of things (IoT) devices in smart cities,
transportation, healthcare, and industrial applications, coupled with the
explosive growth of AI-driven services, has increased demands for efficient
distributed computing architectures and networks, driving cloud-edge-terminal
collaborative intelligence (CETCI) as a fundamental paradigm within the
artificial intelligence of things (AIoT) community. With advancements in deep
learning, large language models (LLMs), and edge computing, CETCI has made
significant progress with emerging AIoT applications, moving beyond isolated
layer optimization to deployable collaborative intelligence systems for AIoT
(CISAIOT), a practical research focus in AI, distributed computing, and
communications. This survey describes foundational architectures, enabling
technologies, and scenarios of CETCI paradigms, offering a tutorial-style
review for CISAIOT beginners. We systematically analyze architectural
components spanning cloud, edge, and terminal layers, examining core
technologies including network virtualization, container orchestration, and
software-defined networking, while presenting categorizations of collaboration
paradigms that cover task offloading, resource allocation, and optimization
across heterogeneous infrastructures. Furthermore, we explain intelligent
collaboration learning frameworks by reviewing advances in federated learning,
distributed deep learning, edge-cloud model evolution, and reinforcement
learning-based methods. Finally, we discuss challenges (e.g., scalability,
heterogeneity, interoperability) and future trends (e.g., 6G+, agents, quantum
computing, digital twin), highlighting how integration of distributed computing
and communication can address open issues and guide development of robust,
efficient, and secure collaborative AIoT systems.

### 9. [A Theory of Goal-Oriented Medium Access: Protocol Design and Distributed Bandit Learning](http://arxiv.org/pdf/2508.19141v1)

Authors: Federico Chiariotti, Andrea Zanella

The Goal-oriented Communication (GoC) paradigm breaks the separation between
communication and the content of the data, tailoring communication decisions to
the specific needs of the receiver and targeting application performance. While
recent studies show impressive encoding performance in point-to-point
scenarios, the multi-node distributed scenario is still almost unexplored.
Moreover, the few studies to investigate this consider a centralized
collision-free approach, where a central scheduler decides the transmission
order of the nodes. In this work, we address the Goal-oriented Multiple Access
(GoMA) problem, in which multiple intelligent agents must coordinate to share a
wireless channel and avoid mutual interference. We propose a theoretical
framework for the analysis and optimization of distributed GoMA, serving as a
first step towards its complete characterization. We prove that the problem is
non-convex and may admit multiple Nash Equilibrium (NE) solutions. We provide a
characterization of each node's best response to others' strategies and propose
an optimization approach that provably reaches one such NE, outperforming
centralized approaches by up to 100% while also reducing energy consumption. We
also design a distributed learning algorithm that operates with limited
feedback and no prior knowledge.

### 10. [Toward Edge General Intelligence with Agentic AI and Agentification: Concepts, Technologies, and Future Directions](http://arxiv.org/pdf/2508.18725v1)

Authors: Ruichen Zhang, Guangyuan Liu, Yinqiu Liu, Changyuan Zhao, Jiacheng Wang, Yunting Xu, Dusit Niyato, Jiawen Kang, Yonghui Li, Shiwen Mao, Sumei Sun, Xuemin Shen, Dong In Kim

The rapid expansion of sixth-generation (6G) wireless networks and the
Internet of Things (IoT) has catalyzed the evolution from centralized cloud
intelligence towards decentralized edge general intelligence. However,
traditional edge intelligence methods, characterized by static models and
limited cognitive autonomy, fail to address the dynamic, heterogeneous, and
resource-constrained scenarios inherent to emerging edge networks. Agentic
artificial intelligence (Agentic AI) emerges as a transformative solution,
enabling edge systems to autonomously perceive multimodal environments, reason
contextually, and adapt proactively through continuous
perception-reasoning-action loops. In this context, the agentification of edge
intelligence serves as a key paradigm shift, where distributed entities evolve
into autonomous agents capable of collaboration and continual adaptation. This
paper presents a comprehensive survey dedicated to Agentic AI and
agentification frameworks tailored explicitly for edge general intelligence.
First, we systematically introduce foundational concepts and clarify
distinctions from traditional edge intelligence paradigms. Second, we analyze
important enabling technologies, including compact model compression,
energy-aware computing strategies, robust connectivity frameworks, and advanced
knowledge representation and reasoning mechanisms. Third, we provide
representative case studies demonstrating Agentic AI's capabilities in
low-altitude economy networks, intent-driven networking, vehicular networks,
and human-centric service provisioning, supported by numerical evaluations.
Furthermore, we identify current research challenges, review emerging
open-source platforms, and highlight promising future research directions to
guide robust, scalable, and trustworthy Agentic AI deployments for
next-generation edge environments.

### Robotics

### 1. [SignLoc: Robust Localization using Navigation Signs and Public Maps](http://arxiv.org/pdf/2508.18606v1)

Authors: Nicky Zimmerman, Joel Loo, Ayush Agrawal, David Hsu

Navigation signs and maps, such as floor plans and street maps, are widely
available and serve as ubiquitous aids for way-finding in human environments.
Yet, they are rarely used by robot systems. This paper presents SignLoc, a
global localization method that leverages navigation signs to localize the
robot on publicly available maps -- specifically floor plans and OpenStreetMap
(OSM) graphs -- without prior sensor-based mapping. SignLoc first extracts a
navigation graph from the input map. It then employs a probabilistic
observation model to match directional and locational cues from the detected
signs to the graph, enabling robust topo-semantic localization within a Monte
Carlo framework. We evaluated SignLoc in diverse large-scale environments: part
of a university campus, a shopping mall, and a hospital complex. Experimental
results show that SignLoc reliably localizes the robot after observing only one
to two signs.

### 2. [Integration of Robot and Scene Kinematics for Sequential Mobile Manipulation Planning](http://arxiv.org/pdf/2508.18627v1)

Authors: Ziyuan Jiao, Yida Niu, Zeyu Zhang, Yangyang Wu, Yao Su, Yixin Zhu, Hangxin Liu, Song-Chun Zhu

We present a Sequential Mobile Manipulation Planning (SMMP) framework that
can solve long-horizon multi-step mobile manipulation tasks with coordinated
whole-body motion, even when interacting with articulated objects. By
abstracting environmental structures as kinematic models and integrating them
with the robot's kinematics, we construct an Augmented Configuration Apace
(A-Space) that unifies the previously separate task constraints for navigation
and manipulation, while accounting for the joint reachability of the robot
base, arm, and manipulated objects. This integration facilitates efficient
planning within a tri-level framework: a task planner generates symbolic action
sequences to model the evolution of A-Space, an optimization-based motion
planner computes continuous trajectories within A-Space to achieve desired
configurations for both the robot and scene elements, and an intermediate plan
refinement stage selects action goals that ensure long-horizon feasibility. Our
simulation studies first confirm that planning in A-Space achieves an 84.6\%
higher task success rate compared to baseline methods. Validation on real
robotic systems demonstrates fluid mobile manipulation involving (i) seven
types of rigid and articulated objects across 17 distinct contexts, and (ii)
long-horizon tasks of up to 14 sequential steps. Our results highlight the
significance of modeling scene kinematics into planning entities, rather than
encoding task-specific constraints, offering a scalable and generalizable
approach to complex robotic manipulation.

### 3. [Engineering Automotive Digital Twins on Standardized Architectures: A Case Study](http://arxiv.org/pdf/2508.18662v1)

Authors: Stefan Ramdhan, Winnie Trandinh, Istvan David, Vera Pantelic, Mark Lawford

Digital twin (DT) technology has become of interest in the automotive
industry. There is a growing need for smarter services that utilize the unique
capabilities of DTs, ranging from computer-aided remote control to cloud-based
fleet coordination. Developing such services starts with the software
architecture. However, the scarcity of DT architectural guidelines poses a
challenge for engineering automotive DTs. Currently, the only DT architectural
standard is the one defined in ISO 23247. Though not developed for automotive
systems, it is one of the few feasible starting points for automotive DTs. In
this work, we investigate the suitability of the ISO 23247 reference
architecture for developing automotive DTs. Through the case study of
developing an Adaptive Cruise Control DT for a 1/10\textsuperscript{th}-scale
autonomous vehicle, we identify some strengths and limitations of the reference
architecture and begin distilling future directions for researchers,
practitioners, and standard developers.

### 4. [Deep Sensorimotor Control by Imitating Predictive Models of Human Motion](http://arxiv.org/pdf/2508.18691v1)

Authors: Himanshu Gaurav Singh, Pieter Abbeel, Jitendra Malik, Antonio Loquercio

As the embodiment gap between a robot and a human narrows, new opportunities
arise to leverage datasets of humans interacting with their surroundings for
robot learning. We propose a novel technique for training sensorimotor policies
with reinforcement learning by imitating predictive models of human motions.
Our key insight is that the motion of keypoints on human-inspired robot
end-effectors closely mirrors the motion of corresponding human body keypoints.
This enables us to use a model trained to predict future motion on human data
\emph{zero-shot} on robot data. We train sensorimotor policies to track the
predictions of such a model, conditioned on a history of past robot states,
while optimizing a relatively sparse task reward. This approach entirely
bypasses gradient-based kinematic retargeting and adversarial losses, which
limit existing methods from fully leveraging the scale and diversity of modern
human-scene interaction datasets. Empirically, we find that our approach can
work across robots and tasks, outperforming existing baselines by a large
margin. In addition, we find that tracking a human motion model can substitute
for carefully designed dense rewards and curricula in manipulation tasks. Code,
data and qualitative results available at
https://jirl-upenn.github.io/track_reward/.

### 5. [HyperTASR: Hypernetwork-Driven Task-Aware Scene Representations for Robust Manipulation](http://arxiv.org/pdf/2508.18802v1)

Authors: Li Sun, Jiefeng Wu, Feng Chen, Ruizhe Liu, Yanchao Yang

Effective policy learning for robotic manipulation requires scene
representations that selectively capture task-relevant environmental features.
Current approaches typically employ task-agnostic representation extraction,
failing to emulate the dynamic perceptual adaptation observed in human
cognition. We present HyperTASR, a hypernetwork-driven framework that modulates
scene representations based on both task objectives and the execution phase.
Our architecture dynamically generates representation transformation parameters
conditioned on task specifications and progression state, enabling
representations to evolve contextually throughout task execution. This approach
maintains architectural compatibility with existing policy learning frameworks
while fundamentally reconfiguring how visual features are processed. Unlike
methods that simply concatenate or fuse task embeddings with task-agnostic
representations, HyperTASR establishes computational separation between
task-contextual and state-dependent processing paths, enhancing learning
efficiency and representational quality. Comprehensive evaluations in both
simulation and real-world environments demonstrate substantial performance
improvements across different representation paradigms. Through ablation
studies and attention visualization, we confirm that our approach selectively
prioritizes task-relevant scene information, closely mirroring human adaptive
perception during manipulation tasks. The project website is at
\href{https://lisunphil.github.io/HyperTASR_projectpage/}{lisunphil.github.io/HyperTASR\_projectpage}.

### 6. [VisionSafeEnhanced VPC: Cautious Predictive Control with Visibility Constraints under Uncertainty for Autonomous Robotic Surgery](http://arxiv.org/pdf/2508.18937v1)

Authors: Wang Jiayin, Wei Yanran, Jiang Lei, Guo Xiaoyu, Zheng Ayong, Zhao Weidong, Li Zhongkui

Autonomous control of the laparoscope in robot-assisted Minimally Invasive
Surgery (MIS) has received considerable research interest due to its potential
to improve surgical safety. Despite progress in pixel-level Image-Based Visual
Servoing (IBVS) control, the requirement of continuous visibility and the
existence of complex disturbances, such as parameterization error, measurement
noise, and uncertainties of payloads, could degrade the surgeon's visual
experience and compromise procedural safety. To address these limitations, this
paper proposes VisionSafeEnhanced Visual Predictive Control (VPC), a robust and
uncertainty-adaptive framework for autonomous laparoscope control that
guarantees Field of View (FoV) safety under uncertainty. Firstly, Gaussian
Process Regression (GPR) is utilized to perform hybrid (deterministic +
stochastic) quantification of operational uncertainties including residual
model uncertainties, stochastic uncertainties, and external disturbances. Based
on uncertainty quantification, a novel safety aware trajectory optimization
framework with probabilistic guarantees is proposed, where a
uncertainty-adaptive safety Control Barrier Function (CBF) condition is given
based on uncertainty propagation, and chance constraints are simultaneously
formulated based on probabilistic approximation. This uncertainty aware
formulation enables adaptive control effort allocation, minimizing unnecessary
camera motion while maintaining robustness. The proposed method is validated
through comparative simulations and experiments on a commercial surgical robot
platform (MicroPort MedBot Toumai) performing a sequential multi-target lymph
node dissection. Compared with baseline methods, the framework maintains
near-perfect target visibility (>99.9%), reduces tracking e

### 7. [HuBE: Cross-Embodiment Human-like Behavior Execution for Humanoid Robots](http://arxiv.org/pdf/2508.19002v1)

Authors: Shipeng Lyu, Fangyuan Wang, Weiwei Lin, Luhao Zhu, David Navarro-Alarcon, Guodong Guo

Achieving both behavioral similarity and appropriateness in human-like motion
generation for humanoid robot remains an open challenge, further compounded by
the lack of cross-embodiment adaptability. To address this problem, we propose
HuBE, a bi-level closed-loop framework that integrates robot state, goal poses,
and contextual situations to generate human-like behaviors, ensuring both
behavioral similarity and appropriateness, and eliminating structural
mismatches between motion generation and execution. To support this framework,
we construct HPose, a context-enriched dataset featuring fine-grained
situational annotations. Furthermore, we introduce a bone scaling-based data
augmentation strategy that ensures millimeter-level compatibility across
heterogeneous humanoid robots. Comprehensive evaluations on multiple commercial
platforms demonstrate that HuBE significantly improves motion similarity,
behavioral appropriateness, and computational efficiency over state-of-the-art
baselines, establishing a solid foundation for transferable and human-like
behavior execution across diverse humanoid robots.

### 8. [QuadKAN: KAN-Enhanced Quadruped Motion Control via End-to-End Reinforcement Learning](http://arxiv.org/pdf/2508.19153v1)

Authors: Allen Wang, Gavin Tao

We address vision-guided quadruped motion control with reinforcement learning
(RL) and highlight the necessity of combining proprioception with vision for
robust control. We propose QuadKAN, a spline-parameterized cross-modal policy
instantiated with Kolmogorov-Arnold Networks (KANs). The framework incorporates
a spline encoder for proprioception and a spline fusion head for
proprioception-vision inputs. This structured function class aligns the
state-to-action mapping with the piecewise-smooth nature of gait, improving
sample efficiency, reducing action jitter and energy consumption, and providing
interpretable posture-action sensitivities. We adopt Multi-Modal Delay
Randomization (MMDR) and perform end-to-end training with Proximal Policy
Optimization (PPO). Evaluations across diverse terrains, including both even
and uneven surfaces and scenarios with static or dynamic obstacles, demonstrate
that QuadKAN achieves consistently higher returns, greater distances, and fewer
collisions than state-of-the-art (SOTA) baselines. These results show that
spline-parameterized policies offer a simple, effective, and interpretable
alternative for robust vision-guided locomotion. A repository will be made
available upon acceptance.

### 9. [Direction Informed Trees (DIT*): Optimal Path Planning via Direction Filter and Direction Cost Heuristic](http://arxiv.org/pdf/2508.19168v1)

Authors: Liding Zhang, Kejia Chen, Kuanqi Cai, Yu Zhang, Yixuan Dang, Yansong Wu, Zhenshan Bing, Fan Wu, Sami Haddadin, Alois Knoll

Optimal path planning requires finding a series of feasible states from the
starting point to the goal to optimize objectives. Popular path planning
algorithms, such as Effort Informed Trees (EIT*), employ effort heuristics to
guide the search. Effective heuristics are accurate and computationally
efficient, but achieving both can be challenging due to their conflicting
nature. This paper proposes Direction Informed Trees (DIT*), a sampling-based
planner that focuses on optimizing the search direction for each edge,
resulting in goal bias during exploration. We define edges as generalized
vectors and integrate similarity indexes to establish a directional filter that
selects the nearest neighbors and estimates direction costs. The estimated
direction cost heuristics are utilized in edge evaluation. This strategy allows
the exploration to share directional information efficiently. DIT* convergence
faster than existing single-query, sampling-based planners on tested problems
in R^4 to R^16 and has been demonstrated in real-world environments with
various planning tasks. A video showcasing our experimental results is
available at: https://youtu.be/2SX6QT2NOek

### 10. [AutoRing: Imitation Learning--based Autonomous Intraocular Foreign Body Removal Manipulation with Eye Surgical Robot](http://arxiv.org/pdf/2508.19191v1)

Authors: Yue Wang, Wenjie Deng, Haotian Xue, Di Cui, Yiqi Chen, Mingchuan Zhou, Haochao Ying, Jian Wu

Intraocular foreign body removal demands millimeter-level precision in
confined intraocular spaces, yet existing robotic systems predominantly rely on
manual teleoperation with steep learning curves. To address the challenges of
autonomous manipulation (particularly kinematic uncertainties from variable
motion scaling and variation of the Remote Center of Motion (RCM) point), we
propose AutoRing, an imitation learning framework for autonomous intraocular
foreign body ring manipulation. Our approach integrates dynamic RCM calibration
to resolve coordinate-system inconsistencies caused by intraocular instrument
variation and introduces the RCM-ACT architecture, which combines
action-chunking transformers with real-time kinematic realignment. Trained
solely on stereo visual data and instrument kinematics from expert
demonstrations in a biomimetic eye model, AutoRing successfully completes ring
grasping and positioning tasks without explicit depth sensing. Experimental
validation demonstrates end-to-end autonomy under uncalibrated microscopy
conditions. The results provide a viable framework for developing intelligent
eye-surgical systems capable of complex intraocular procedures.

### Software Engineering

### 1. [Requirements Development and Formalization for Reliable Code Generation: A Multi-Agent Vision](http://arxiv.org/pdf/2508.18675v1)

Authors: Xu Lu, Weisong Sun, Yiran Zhang, Ming Hu, Cong Tian, Zhi Jin, Yang Liu

Automated code generation has long been considered the holy grail of software
engineering. The emergence of Large Language Models (LLMs) has catalyzed a
revolutionary breakthrough in this area. However, existing methods that only
rely on LLMs remain inadequate in the quality of generated code, offering no
guarantees of satisfying practical requirements. They lack a systematic
strategy for requirements development and modeling. Recently, LLM-based agents
typically possess powerful abilities and play an essential role in facilitating
the alignment of LLM outputs with user requirements. In this paper, we envision
the first multi-agent framework for reliable code generation based on
\textsc{re}quirements \textsc{de}velopment and \textsc{fo}rmalization, named
\textsc{ReDeFo}. This framework incorporates three agents, highlighting their
augmentation with knowledge and techniques of formal methods, into the
requirements-to-code generation pipeline to strengthen quality assurance. The
core of \textsc{ReDeFo} is the use of formal specifications to bridge the gap
between potentially ambiguous natural language requirements and precise
executable code. \textsc{ReDeFo} enables rigorous reasoning about correctness,
uncovering hidden bugs, and enforcing critical properties throughout the
development process. In general, our framework aims to take a promising step
toward realizing the long-standing vision of reliable, auto-generated software.

### 2. [LLM as an Execution Estimator: Recovering Missing Dependency for Practical Time-travelling Debugging](http://arxiv.org/pdf/2508.18721v1)

Authors: Yunrui Pei, Hongshu Wang, Wenjie Zhang, Yun Lin, Weiyu Kong, Jin song Dong

Dynamic data dependency, answering "why a variable has this value?", is
critical for debugging. Given a program step `s` reading a variable `v`,
finding the dynamic definition of `v` is challenging. Traditional methods
require either (1) exhaustive instrumentation of all possible definitions of
`v` in one run or (2) replicating the run to re-examine reads/writes - both
costly. If `v` is defined in a library, instrumentation becomes expensive; for
non-deterministic programs, replication is infeasible.
  We propose RecovSlicing, which computes dynamic data dependency in a single
run with partial instrumentation. We leverage LLMs to infer program behavior
from a partially recorded trace and code context. Given a trace and a slicing
criterion (step `s` and variable `v`), RecovSlicing estimates the runtime
definition of `v` by recovering the missing execution.It also supports implicit
variables, such as those in `list.get(i)`. Technically, RecovSlicing tackles:
(1) recovering runtime values and structures, and (2) aligning recovered
variables with recorded memory to analyze definitions.
  We evaluate RecovSlicing on 8,300 data dependencies across three slicing
benchmarks, comparing it with Slicer4J, ND-Slicer, LLM Slicer, and re-execution
Slicer. RecovSlicing achieves accuracy of 80.3%, 91.1%, and 98.3%,
outperforming the best baseline (39.0%, 82.0%, 59.9%), and also leads in recall
(91.1%, 91.1%, 98.3% vs. 53.4%, 79.1%, 87.1%). Integrated into a regression bug
localizer, it enables finding 16% more regressions.

### 3. [Does AI Code Review Lead to Code Changes? A Case Study of GitHub Actions](http://arxiv.org/pdf/2508.18771v1)

Authors: Kexin Sun, Hongyu Kuang, Sebastian Baltes, Xin Zhou, He Zhang, Xiaoxing Ma, Guoping Rong, Dong Shao, Christoph Treude

AI-based code review tools automatically review and comment on pull requests
to improve code quality. Despite their growing presence, little is known about
their actual impact. We present a large-scale empirical study of 16 popular
AI-based code review actions for GitHub workflows, analyzing more than 22,000
review comments in 178 repositories. We investigate (1) how these tools are
adopted and configured, (2) whether their comments lead to code changes, and
(3) which factors influence their effectiveness. We develop a two-stage
LLM-assisted framework to determine whether review comments are addressed, and
use interpretable machine learning to identify influencing factors. Our
findings show that, while adoption is growing, effectiveness varies widely.
Comments that are concise, contain code snippets, and are manually triggered,
particularly those from hunk-level review tools, are more likely to result in
code changes. These results highlight the importance of careful tool design and
suggest directions for improving AI-based code review systems.

### 4. [Dealing with SonarQube Cloud: Initial Results from a Mining Software Repository Study](http://arxiv.org/pdf/2508.18816v1)

Authors: Sabato Nocera, Davide Fucci, Giuseppe Scanniello

Background: Static Code Analysis (SCA) tools are widely adopted to enforce
code quality standards. However, little is known about how open-source projects
use and customize these tools. Aims: This paper investigates how GitHub
projects use and customize a popular SCA tool, namely SonarQube Cloud. Method:
We conducted a mining study of GitHub projects that are linked through GitHub
Actions to SonarQube Cloud projects. Results: Among 321 GitHub projects using
SonarQube Cloud, 81% of them are correctly connected to SonarQube Cloud
projects, while others exhibit misconfigurations or restricted access. Among
265 accessible SonarQube Cloud projects, 75% use the organization's default
quality gate, i.e., a set of conditions that deployed source code must meet to
pass automated checks. While 55% of the projects use the built-in quality gate
provided by SonarQube Cloud, 45% of them customize their quality gate with
different conditions. Overall, the most common quality conditions align with
SonarQube Cloud's "Clean as You Code" principle and enforce security,
maintainability, reliability, coverage, and a few duplicates on newly added or
modified source code. Conclusions: Many projects rely on predefined
configurations, yet a significant portion customize their configurations to
meet specific quality goals. Building on our initial results, we envision a
future research agenda linking quality gate configurations to actual software
outcomes (e.g., improvement of software security). This would enable
evidence-based recommendations for configuring SCA tools like SonarQube Cloud
in various contexts.

### 5. [Interleaving Large Language Models for Compiler Testing](http://arxiv.org/pdf/2508.18955v1)

Authors: Yunbo Ni, Shaohua Li

Testing compilers with AI models, especially large language models (LLMs),
has shown great promise. However, current approaches struggle with two key
problems: The generated programs for testing compilers are often too simple,
and extensive testing with the LLMs is computationally expensive. In this
paper, we propose a novel compiler testing framework that decouples the testing
process into two distinct phases: an offline phase and an online phase. In the
offline phase, we use LLMs to generate a collection of small but feature-rich
code pieces. In the online phase, we reuse these code pieces by strategically
combining them to build high-quality and valid test programs, which are then
used to test compilers.
  We implement this idea in a tool, LegoFuzz, for testing C compilers. The
results are striking: we found 66 bugs in GCC and LLVM, the most widely used C
compilers. Almost half of the bugs are miscompilation bugs, which are serious
and hard-to-find bugs that none of the existing LLM-based tools could find. We
believe this efficient design opens up new possibilities for using AI models in
software testing beyond just C compilers.

### 6. [A Slice-Based Change Impact Analysis for Regression Test Case Prioritization of Object-Oriented Programs](http://arxiv.org/pdf/2508.19056v1)

Authors: S. Panda, D. Munjal, D. P. Mohapatra

Test case prioritization focuses on finding a suitable order of execution of
the test cases in a test suite to meet some performance goals like detecting
faults early. It is likely that some test cases execute the program parts that
are more prone to errors and will detect more errors if executed early during
the testing process. Finding an optimal order of execution for the selected
regression test cases saves time and cost of retesting. This paper presents a
static approach to prioritizing the test cases by computing the affected
component coupling (ACC) of the affected parts of object-oriented programs. We
construct a graph named affected slice graph (ASG) to represent these affected
program parts.We determine the fault-proneness of the nodes of ASG by computing
their respective ACC values. We assign higher priority to those test cases that
cover the nodes with higher ACC values. Our analysis with mutation faults shows
that the test cases executing the fault-prone program parts have a higher
chance to reveal faults earlier than other test cases in the test suite. The
result obtained from seven case studies justifies that our approach is feasible
and gives acceptable performance in comparison to some existing techniques.

### 7. [LaQual: A Novel Framework for Automated Evaluation of LLM App Quality](http://arxiv.org/pdf/2508.18636v1)

Authors: Yan Wang, Xinyi Hou, Yanjie Zhao, Weiguo Lin, Haoyu Wang, Junjun Si

LLM app stores are quickly emerging as platforms that gather a wide range of
intelligent applications based on LLMs, giving users many choices for content
creation, coding support, education, and more. However, the current methods for
ranking and recommending apps in these stores mostly rely on static metrics
like user activity and favorites, which makes it hard for users to efficiently
find high-quality apps. To address these challenges, we propose LaQual, an
automated framework for evaluating the quality of LLM apps. LaQual consists of
three main stages: first, it labels and classifies LLM apps in a hierarchical
way to accurately match them to different scenarios; second, it uses static
indicators, such as time-weighted user engagement and functional capability
metrics, to filter out low-quality apps; and third, it conducts a dynamic,
scenario-adaptive evaluation, where the LLM itself generates scenario-specific
evaluation metrics, scoring rules, and tasks for a thorough quality assessment.
Experiments on a popular LLM app store show that LaQual is effective. Its
automated scores are highly consistent with human judgments (with Spearman's
rho of 0.62 and p=0.006 in legal consulting, and rho of 0.60 and p=0.009 in
travel planning). By effectively screening, LaQual can reduce the pool of
candidate LLM apps by 66.7% to 81.3%. User studies further confirm that LaQual
significantly outperforms baseline systems in decision confidence, comparison
efficiency (with average scores of 5.45 compared to 3.30), and the perceived
value of its evaluation reports (4.75 versus 2.25). Overall, these results
demonstrate that LaQual offers a scalable, objective, and user-centered
solution for finding and recommending high-quality LLM apps in real-world use
cases.

### 8. [GitTaskBench: A Benchmark for Code Agents Solving Real-World Tasks Through Code Repository Leveraging](http://arxiv.org/pdf/2508.18993v1)

Authors: Ziyi Ni, Huacan Wang, Shuo Zhang, Shuo Lu, Ziyang He, Wang You, Zhenheng Tang, Yuntao Du, Bill Sun, Hongzhang Liu, Sen Hu, Ronghao Chen, Bo Li, Xin Li, Chen Hu, Binxing Jiao, Daxin Jiang, Pin Lyu

Beyond scratch coding, exploiting large-scale code repositories (e.g.,
GitHub) for practical tasks is vital in real-world software development, yet
current benchmarks rarely evaluate code agents in such authentic,
workflow-driven scenarios. To bridge this gap, we introduce GitTaskBench, a
benchmark designed to systematically assess this capability via 54 realistic
tasks across 7 modalities and 7 domains. Each task pairs a relevant repository
with an automated, human-curated evaluation harness specifying practical
success criteria. Beyond measuring execution and task success, we also propose
the alpha-value metric to quantify the economic benefit of agent performance,
which integrates task success rates, token cost, and average developer
salaries. Experiments across three state-of-the-art agent frameworks with
multiple advanced LLMs show that leveraging code repositories for complex task
solving remains challenging: even the best-performing system, OpenHands+Claude
3.7, solves only 48.15% of tasks. Error analysis attributes over half of
failures to seemingly mundane yet critical steps like environment setup and
dependency resolution, highlighting the need for more robust workflow
management and increased timeout preparedness. By releasing GitTaskBench, we
aim to drive progress and attention toward repository-aware code reasoning,
execution, and deployment -- moving agents closer to solving complex,
end-to-end real-world tasks. The benchmark and code are open-sourced at
https://github.com/QuantaAlpha/GitTaskBench.

### 9. [An Efficient Lightweight Blockchain for Decentralized IoT](http://arxiv.org/pdf/2508.19219v1)

Authors: Faezeh Dehghan Tarzjani, Mostafa Salehi

The Internet of Things (IoT) is applied in various fields, and the number of
physical devices connected to the IoT is increasingly growing. There are
significant challenges to the IoT's growth and development, mainly due to the
centralized nature and large-scale IoT networks. The emphasis on the
decentralization of IoT's architecture can overcome challenges to IoT's
capabilities. A promising decentralized platform for IoT is blockchain. Owing
to IoT devices' limited resources, traditional consensus algorithms such as PoW
and PoS in the blockchain are computationally expensive. Therefore, the PoA
consensus algorithm is proposed in the blockchain consensus network for IoT.
The PoA selects the validator as Turn-based selection (TBS) that needs
optimization and faces system reliability, energy consumption, latency, and low
scalability. We propose an efficient, lightweight blockchain for decentralizing
IoT architecture by using virtualization and clustering to increase
productivity and scalability to address these issues. We also introduce a novel
PoA based on the Weight-Based-Selection (WBS) method for validators to validate
transactions and add them to the blockchain. By simulation, we evaluated the
performance of our proposed WBS method as opposed to TBS. The results show
reduced energy consumption, and response time, and increased throughput.

### Social and Information Networks

### 1. [Recognizing Distance-Count Matrices is Difficult](http://arxiv.org/pdf/2508.18857v1)

Authors: Paolo Boldi, Flavio Furia, Chiara Prezioso, Ian Stewart

Axiomatization of centrality measures often involves proving that something
cannot hold by providing a counterexample (i.e., a graph for which that
specific centrality index fails to have a given property). In the context of
geometric centralities, building such counterexamples requires constructing a
graph with specific distance counts between nodes, as expressed by its
distance-count matrix. We prove that deciding whether a matrix is the
distance-count matrix of a graph is strongly NP-complete. This negative result
implies that a brute-force approach to building this kind of counterexample is
out of question, and cleverer approaches are required.

### 2. [Digital Skills Formation in Gendered Peer Networks: Exploring advice giving and taking in classrooms](http://arxiv.org/pdf/2508.19102v1)

Authors: Petro Tolochko, Jana Bernhard-Harrer, Azade E. Kakavand, Aytalina Kulichkina, Hyunjin Song, Hajo G. Boomgaarden

The digitalisation of childhood underscores the importance of early digital
skill development. To understand how peer relationships shape this process, we
draw on unique sociocentric network data from students in classrooms across
three countries, focusing on peer-to-peer advice-giving and advice-seeking
networks related to digital skills. Using exponential random graph models, we
find that digital skills systematically spread through peer interactions:
higher-skilled students are more likely to be sought for advice while less
likely to seek it themselves. Students perceived as highly skilled are more
likely to seek and offer advice, but it has limited influence on being sought
out by others. Gender plays a significant role: girls both seek and give more
advice, with strong gender homophily shaping these interactions. We suggest
that digital skills education should leverage the potential of peer learning
within formal education and consider how such approaches can address persistent
divides.

### 3. [LLM-based Contrastive Self-Supervised AMR Learning with Masked Graph Autoencoders for Fake News Detection](http://arxiv.org/pdf/2508.18819v1)

Authors: Shubham Gupta, Shraban Kumar Chatterjee, Suman Kundu

The proliferation of misinformation in the digital age has led to significant
societal challenges. Existing approaches often struggle with capturing
long-range dependencies, complex semantic relations, and the social dynamics
influencing news dissemination. Furthermore, these methods require extensive
labelled datasets, making their deployment resource-intensive. In this study,
we propose a novel self-supervised misinformation detection framework that
integrates both complex semantic relations using Abstract Meaning
Representation (AMR) and news propagation dynamics. We introduce an LLM-based
graph contrastive loss (LGCL) that utilizes negative anchor points generated by
a Large Language Model (LLM) to enhance feature separability in a zero-shot
manner. To incorporate social context, we employ a multi view graph masked
autoencoder, which learns news propagation features from social context graph.
By combining these semantic and propagation-based features, our approach
effectively differentiates between fake and real news in a self-supervised
manner. Extensive experiments demonstrate that our self-supervised framework
achieves superior performance compared to other state-of-the-art methodologies,
even with limited labelled datasets while improving generalizability.

### 4. [Affective Polarization across European Parliaments](http://arxiv.org/pdf/2508.18916v1)

Authors: Bojan Evkoski, Igor Mozetič, Nikola Ljubešić, Petra Kralj Novak

Affective polarization, characterized by increased negativity and hostility
towards opposing groups, has become a prominent feature of political discourse
worldwide. Our study examines the presence of this type of polarization in a
selection of European parliaments in a fully automated manner. Utilizing a
comprehensive corpus of parliamentary speeches from the parliaments of six
European countries, we employ natural language processing techniques to
estimate parliamentarian sentiment. By comparing the levels of negativity
conveyed in references to individuals from opposing groups versus one's own, we
discover patterns of affectively polarized interactions. The findings
demonstrate the existence of consistent affective polarization across all six
European parliaments. Although activity correlates with negativity, there is no
observed difference in affective polarization between less active and more
active members of parliament. Finally, we show that reciprocity is a
contributing mechanism in affective polarization between parliamentarians
across all six parliaments.

### 5. [Reconstructing graphs and their connectivity using graphlets](http://arxiv.org/pdf/2508.19189v1)

Authors: David Hartman, Aneta Pokorná, Daniel Trlifaj

Graphlets are small subgraphs rooted at a fixed vertex. The number of
occurrences of graphlets aligned to a particular vertex, called graphlet degree
sequence, gives a topological description of the surrounding of the analyzed
vertex. In this article, we study properties and uniqueness of graphlet degree
sequences. The information given by graphlets up to size (n-1) is utilized
graphs having certain type of asymmetric vertex-deleted subgraphs. Moreover, we
show a reconstruction of trees from their (<= n-1)-graphlet degree sequences,
which is much easier compared to the standard reconstruction from
vertex-deleted subgraphs.

### Systems and Control

### 1. [Fuzzy-Based Control Method for Autonomous Spacecraft Inspection with Minimal Fuel Consumption](http://arxiv.org/pdf/2508.18583v1)

Authors: Daegyun Choi, Donghoon Kim, Henzeh Leeghim

This study explores an energy-efficient control strategy for spacecraft
inspection using a fuzzy inference system combined with a bio-inspired
optimization technique to incorporate learning capability into the control
process. The optimized fuzzy controller produces a minimally fuel-consuming
force while maintaining reliable inspection within constraints, such as
illumination, restricted field of view, thrust limits, and safe regions. The
performance of the proposed control strategy is validated through Monte Carlo
simulations.

### 2. [Globally Stable Discrete Time PID Passivity-based Control of Power Converters: Simulation and Experimental Results](http://arxiv.org/pdf/2508.18719v1)

Authors: Alessio Moreschini, Wei He, Romeo Ortega, Yiheng Lu, Tao Li

The key idea behind PID Passivity-based Control (PID-PBC) is to leverage the
passivity property of PIDs (for all positive gains) and wrap the PID controller
around a passive output to ensure global stability in closed-loop. However, the
practical applicability of PID-PBC is stymied by two key facts: (i) the vast
majority of practical implementations of PIDs is carried-out in discrete time
-- discretizing the continuous time dynamical system of the PID; (ii) the
well-known problem that passivity is not preserved upon discretization, even
with small sampling times. Therefore, two aspects of the PID-PBC must be
revisited for its safe practical application. First, we propose a
discretization of the PID that ensures its passivity. Second, since the output
that is identified as passive for the continuous time system is not necessarily
passive for its discrete time version, we construct a new output that ensures
the passivity property for the discretization of the system. In this paper, we
provide a constructive answer to both issues for the case of power converter
models. Instrumental to achieve this objective is the use of the implicit
midpoint discretization method -- which is a symplectic integration technique
that preserves system invariants. Since the reference value for the output to
be regulated in power converters is non-zero, we are henceforth interested in
the property of passivity of the incremental model -- currently known as
shifted passivity. Therefore, we demonstrate that the resulting discrete-time
PID-PBC defines a passive map for the incremental model and establish shifted
passivity for the discretized power converter model. Combining these
properties, we prove global stability for the feedback interconnection of the
power converter with the discretized PID-PBC. The paper also presents
simulations and experiments that demonstrate the performance of the proposed
discretization.

### 3. [Closed-Form Input Design for Identification under Output Feedback with Perturbation Constraints](http://arxiv.org/pdf/2508.18813v1)

Authors: Jingwei Hu, Dave Zachariah, Torbjörn Wigren, Petre Stoica

In many applications, system identification experiments must be performed
under output feedback to ensure safety or to maintain system operation. In this
paper, we consider the online design of informative experiments for ARMAX
models by applying a bounded perturbation to the input signal generated by a
fixed output feedback controller. Specifically, the design constrains the
resulting output perturbation within user-specified limits and can be
efficiently computed in closed form. We demonstrate the effectiveness of the
method in two numerical experiments.

### 4. [Performance Analysis of Underwater Optical Wireless Communication Using O-RIS and Fiber Optic Backhaul (Extended version)](http://arxiv.org/pdf/2508.18915v1)

Authors: Aboozar Heydaribeni, Hamzeh Beyranvand

This Letter presents a novel hybrid underwater wireless optical communication
(UWOC) system that integrates underwater optical access points (UOAPs) with a
passive optical network (PON)-based fiber-optic backhaul to provide a resilient
backbone. A hard switching mechanism is employed between direct and optical
reconfigurable intelligent surface (O-RIS)-assisted links to ensure reliable
connectivity. Unlike previous studies, the proposed system is evaluated under
both active and multiple passive O-RIS configurations. To enhance reliability,
the Selection Combining (SC) and Maximal Ratio Combining (MRC) schemes are
applied. Analytical and simulation results demonstrate that optimal O-RIS
placement significantly enhances system performance. However, in the linear
regime, placing it too close to the receiver causes degradation due to
increased path loss and beam jitter in an identical water type. Moreover,
increasing the number of O-RIS elements within practical limits further
improves overall system performance and enhances adaptability to variations in
the underwater channel.

### 5. [A Principled Framework to Evaluate Quality of AC-OPF Datasets for Machine Learning: Benchmarking a Novel, Scalable Generation Method](http://arxiv.org/pdf/2508.19083v1)

Authors: Matteo Baù, Luca Perbellini, Samuele Grillo

Several methods have been proposed in the literature to improve the quality
of AC optimal power flow (AC-OPF) datasets used in machine learning (ML)
models. Yet, scalability to large power systems remains unaddressed and
comparing generation approaches is still hindered by the absence of widely
accepted metrics quantifying AC-OPF dataset quality. In this work, we tackle
both these limitations. We provide a simple heuristic that samples load
setpoints uniformly in total load active power, rather than maximizing volume
coverage, and solves an AC-OPF formulation with load slack variables to improve
convergence. For quality assessment, we formulate a multi-criteria framework
based on three metrics, measuring variability in the marginal distributions of
AC-OPF primal variables, diversity in constraint activation patterns among
AC-OPF instances and activation frequency of variable bounds. By comparing four
open-source methods based on these metrics, we show that our heuristic
consistently outperforms uniform random sampling, whether independent or
constrained to a convex polytope, scoring as best in terms of balance between
dataset quality and scalability.

### 6. [Learning Interior Point Method for AC and DC Optimal Power Flow](http://arxiv.org/pdf/2508.19146v1)

Authors: Farshad Amani, Amin Kargarian, Ramachandran Vaidyanathan

This paper proposes a feasibility-guaranteed learning interior point method
(L-IPM) to solve both AC and DC optimal power flow (OPF) problems. Given the
criticality of OPF, the proposed L-IPM uses a hybrid learning model approach
rather than relying solely on a simple black-box prediction. The traditional
IPM follows a central path from an initial point to the optimal solution.
However, each iteration involves solving large linear systems, which becomes
increasingly expensive as the matrices grow more ill-conditioned in later
steps. To address this, we model the IPM trajectory as a time series and train
a Long Short-Term Memory (LSTM) network to project the IPM central path using
only the first few stable iterations, which carry the most informative features
about the path to optimality. We introduce a grid-informed methodology that
enforces operational constraints on generation, voltage magnitudes, and line
flows to ensure feasibility. The grid-informed LSTM serves as a tool for the
IPM central path projection and, followed by a final IPM refinement step,
significantly reduces the total number of iterations and time required for
convergence. We use a sampling method to generate a wide range of load
scenarios to improve generalization across diverse operating conditions,
efficiently covering the power system's operational space. Simulation results
on a 2869-bus European high-voltage transmission system show that the proposed
L-IPM significantly reduces solution time by up to 94\%, while maintaining
accuracy and feasibility of the solution. By leveraging early iterations and
bypassing the final ill-conditioned and computationally demanding steps of
traditional IPM, the proposed L-IPM reduces the number of required iterations
by up to 85.5\%. Since solution feasibility is also guaranteed, L-IPM
outperforms the conventional IPM in both computational efficiency and
robustness.

### 7. [Scalable Fairness Shaping with LLM-Guided Multi-Agent Reinforcement Learning for Peer-to-Peer Electricity Markets](http://arxiv.org/pdf/2508.18610v1)

Authors: Shrenik Jadhav, Birva Sevak, Srijita Das, Akhtar Hussain, Wencong Su, Van-Hai Bui

Peer-to-peer (P2P) energy trading is becoming central to modern distribution
systems as rooftop PV and home energy management systems become pervasive, yet
most existing market and reinforcement learning designs emphasize efficiency or
private profit and offer little real-time guidance to ensure equitable outcomes
under uncertainty. To address this gap, a fairness-aware multiagent
reinforcement learning framework, FairMarket-RL, is proposed in which a large
language model (LLM) critic shapes bidding policies within a continuous double
auction under partial observability and discrete price-quantity actions. After
each trading slot, the LLM returns normalized fairness scores Fairness-to-Grid
(FTG), Fairness-Between-Sellers (FBS), and Fairness-of-Pricing (FPP) that are
integrated into the reward via ramped coefficients and tunable scaling, so that
fairness guidance complements, rather than overwhelms, economic incentives. The
environment models realistic residential load and PV profiles and enforce hard
constraints on prices, physical feasibility, and policy-update stability.
Across a progression of experiments from a small pilot to a larger simulated
community and a mixed-asset real-world dataset, the framework shifts exchanges
toward local P2P trades, lowers consumer costs relative to grid-only
procurement, sustains strong fairness across participants, and preserves
utility viability. Sensitivity analyses over solar availability and aggregate
demand further indicate robust performance, suggesting a scalable, LLM-guided
pathway to decentralized electricity markets that are economically efficient,
socially equitable, and technically sound.

### 8. [Potential of Quantum Computing Applications for Smart Grid Digital Twins and Future Directions](http://arxiv.org/pdf/2508.18654v1)

Authors: Arianne Ornella Lemo, Ahmad Mohammad Saber, Deepa Kundur, Adam W. Skorek

The convergence of digital twin technology and quantum computing is opening
new horizons for the modeling, control, and optimization of smart grid systems.
This paper reviews the current research landscape at the intersection of these
fields, with a focus on how quantum algorithms can enhance the performance of
digital twins in smart energy systems. We conduct a thematic literature review
and identify key research trends, technical challenges, and gaps in real-world
adoption. Further, a conceptual framework is proposed to integrate quantum
modules into classical digital twin architectures. The potential benefits of
this hybrid approach for smart grid operation and future research directions
are also discussed.

### 9. [AgriChrono: A Multi-modal Dataset Capturing Crop Growth and Lighting Variability with a Field Robot](http://arxiv.org/pdf/2508.18694v1)

Authors: Jaehwan Jeong, Tuan-Anh Vu, Mohammad Jony, Shahab Ahmad, Md. Mukhlesur Rahman, Sangpil Kim, M. Khalid Jawed

Existing datasets for precision agriculture have primarily been collected in
static or controlled environments such as indoor labs or greenhouses, often
with limited sensor diversity and restricted temporal span. These conditions
fail to reflect the dynamic nature of real farmland, including illumination
changes, crop growth variation, and natural disturbances. As a result, models
trained on such data often lack robustness and generalization when applied to
real-world field scenarios. In this paper, we present AgriChrono, a novel
robotic data collection platform and multi-modal dataset designed to capture
the dynamic conditions of real-world agricultural environments. Our platform
integrates multiple sensors and enables remote, time-synchronized acquisition
of RGB, Depth, LiDAR, and IMU data, supporting efficient and repeatable
long-term data collection across varying illumination and crop growth stages.
We benchmark a range of state-of-the-art 3D reconstruction models on the
AgriChrono dataset, highlighting the difficulty of reconstruction in real-world
field environments and demonstrating its value as a research asset for
advancing model generalization under dynamic conditions. The code and dataset
are publicly available at: https://github.com/StructuresComp/agri-chrono

### 10. [An optimistic planning algorithm for switched discrete-time LQR](http://arxiv.org/pdf/2508.19054v1)

Authors: Mathieu Granzotto, Romain Postoyan, Dragan Nešić, Jamal Daafouz, Lucian Buşoniu

We introduce TROOP, a tree-based Riccati optimistic online planner, that is
designed to generate near-optimal control laws for discrete-time switched
linear systems with switched quadratic costs. The key challenge that we address
is balancing computational resources against control performance, which is
important as constructing near-optimal inputs often requires substantial amount
of computations. TROOP addresses this trade-off by adopting an online
best-first search strategy inspired by A*, allowing for efficient estimates of
the optimal value function. The control laws obtained guarantee both
near-optimality and stability properties for the closed-loop system. These
properties depend on the planning depth, which determines how far into the
future the algorithm explores and is closely related to the amount of
computations. TROOP thus strikes a balance between computational efficiency and
control performance, which is illustrated by numerical simulations on an
example.

### Machine Learning (Statistics Category)

### 1. [Revisiting Follow-the-Perturbed-Leader with Unbounded Perturbations in Bandit Problems](http://arxiv.org/pdf/2508.18604v1)

Authors: Jongyeong Lee, Junya Honda, Shinji Ito, Min-hwan Oh

Follow-the-Regularized-Leader (FTRL) policies have achieved
Best-of-Both-Worlds (BOBW) results in various settings through hybrid
regularizers, whereas analogous results for Follow-the-Perturbed-Leader (FTPL)
remain limited due to inherent analytical challenges. To advance the analytical
foundations of FTPL, we revisit classical FTRL-FTPL duality for unbounded
perturbations and establish BOBW results for FTPL under a broad family of
asymmetric unbounded Fr\'echet-type perturbations, including hybrid
perturbations combining Gumbel-type and Fr\'echet-type tails. These results not
only extend the BOBW results of FTPL but also offer new insights into designing
alternative FTPL policies competitive with hybrid regularization approaches.
Motivated by earlier observations in two-armed bandits, we further investigate
the connection between the $1/2$-Tsallis entropy and a Fr\'echet-type
perturbation. Our numerical observations suggest that it corresponds to a
symmetric Fr\'echet-type perturbation, and based on this, we establish the
first BOBW guarantee for symmetric unbounded perturbations in the two-armed
setting. In contrast, in general multi-armed bandits, we find an instance in
which symmetric Fr\'echet-type perturbations violate the key condition for
standard BOBW analysis, which is a problem not observed with asymmetric or
nonnegative Fr\'echet-type perturbations. Although this example does not rule
out alternative analyses achieving BOBW results, it suggests the limitations of
directly applying the relationship observed in two-armed cases to the general
case and thus emphasizes the need for further investigation to fully understand
the behavior of FTPL in broader settings.

### 2. [Efficient Best-of-Both-Worlds Algorithms for Contextual Combinatorial Semi-Bandits](http://arxiv.org/pdf/2508.18768v1)

Authors: Mengmeng Li, Philipp Schneider, Jelisaveta Aleksić, Daniel Kuhn

We introduce the first best-of-both-worlds algorithm for contextual
combinatorial semi-bandits that simultaneously guarantees
$\widetilde{\mathcal{O}}(\sqrt{T})$ regret in the adversarial regime and
$\widetilde{\mathcal{O}}(\ln T)$ regret in the corrupted stochastic regime. Our
approach builds on the Follow-the-Regularized-Leader (FTRL) framework equipped
with a Shannon entropy regularizer, yielding a flexible method that admits
efficient implementations. Beyond regret bounds, we tackle the practical
bottleneck in FTRL (or, equivalently, Online Stochastic Mirror Descent) arising
from the high-dimensional projection step encountered in each round of
interaction. By leveraging the Karush-Kuhn-Tucker conditions, we transform the
$K$-dimensional convex projection problem into a single-variable root-finding
problem, dramatically accelerating each round. Empirical evaluations
demonstrate that this combined strategy not only attains the attractive regret
bounds of best-of-both-worlds algorithms but also delivers substantial
per-round speed-ups, making it well-suited for large-scale, real-time
applications.

### 3. [Federated Learning with Heterogeneous and Private Label Sets](http://arxiv.org/pdf/2508.18774v1)

Authors: Adam Breitholtz, Edvin Listo Zec, Fredrik D. Johansson

Although common in real-world applications, heterogeneous client label sets
are rarely investigated in federated learning (FL). Furthermore, in the cases
they are, clients are assumed to be willing to share their entire label sets
with other clients. Federated learning with private label sets, shared only
with the central server, adds further constraints on learning algorithms and
is, in general, a more difficult problem to solve. In this work, we study the
effects of label set heterogeneity on model performance, comparing the public
and private label settings -- when the union of label sets in the federation is
known to clients and when it is not. We apply classical methods for the
classifier combination problem to FL using centralized tuning, adapt common FL
methods to the private label set setting, and discuss the justification of both
approaches under practical assumptions. Our experiments show that reducing the
number of labels available to each client harms the performance of all methods
substantially. Centralized tuning of client models for representational
alignment can help remedy this, but often at the cost of higher variance.
Throughout, our proposed adaptations of standard FL methods perform well,
showing similar performance in the private label setting as the standard
methods achieve in the public setting. This shows that clients can enjoy
increased privacy at little cost to model accuracy.

### 4. [Lightweight posterior construction for gravitational-wave catalogs with the Kolmogorov-Arnold network](http://arxiv.org/pdf/2508.18698v1)

Authors: Wenshuai Liu, Yiming Dong, Ziming Wang, Lijing Shao

Neural density estimation has seen widespread applications in the
gravitational-wave (GW) data analysis, which enables real-time parameter
estimation for compact binary coalescences and enhances rapid inference for
subsequent analysis such as population inference. In this work, we explore the
application of using the Kolmogorov-Arnold network (KAN) to construct efficient
and interpretable neural density estimators for lightweight posterior
construction of GW catalogs. By replacing conventional activation functions
with learnable splines, KAN achieves superior interpretability, higher
accuracy, and greater parameter efficiency on related scientific tasks.
Leveraging this feature, we propose a KAN-based neural density estimator, which
ingests megabyte-scale GW posterior samples and compresses them into model
weights of tens of kilobytes. Subsequently, analytic expressions requiring only
several kilobytes can be further distilled from these neural network weights
with minimal accuracy trade-off. In practice, GW posterior samples with
fidelity can be regenerated rapidly using the model weights or analytic
expressions for subsequent analysis. Our lightweight posterior construction
strategy is expected to facilitate user-level data storage and transmission,
paving a path for efficient analysis of numerous GW events in the
next-generation GW detectors.

### 5. [Sparse minimum Redundancy Maximum Relevance for feature selection](http://arxiv.org/pdf/2508.18901v1)

Authors: Peter Naylor, Benjamin Poignard, Héctor Climente-González, Makoto Yamada

We propose a feature screening method that integrates both feature-feature
and feature-target relationships. Inactive features are identified via a
penalized minimum Redundancy Maximum Relevance (mRMR) procedure, which is the
continuous version of the classic mRMR penalized by a non-convex regularizer,
and where the parameters estimated as zero coefficients represent the set of
inactive features. We establish the conditions under which zero coefficients
are correctly identified to guarantee accurate recovery of inactive features.
We introduce a multi-stage procedure based on the knockoff filter enabling the
penalized mRMR to discard inactive features while controlling the false
discovery rate (FDR). Our method performs comparably to HSIC-LASSO but is more
conservative in the number of selected features. It only requires setting an
FDR threshold, rather than specifying the number of features to retain. The
effectiveness of the method is illustrated through simulations and real-world
datasets. The code to reproduce this work is available on the following GitHub:
https://github.com/PeterJackNaylor/SmRMR.

### 6. [The GINN framework: a stochastic QED correspondence for stability and chaos in deep neural networks](http://arxiv.org/pdf/2508.18948v1)

Authors: Rodrigo Carmo Terin

The development of a Euclidean stochastic field-theoretic approach that maps
deep neural networks (DNNs) to quantum electrodynamics (QED) with local U(1)
symmetry is presented. Neural activations and weights are represented by
fermionic matter and gauge fields, with a fictitious Langevin time enabling
covariant gauge fixing. This mapping identifies the gauge parameter with kernel
design choices in wide DNNs, relating stability thresholds to gauge-dependent
amplification factors. Finite-width fluctuations correspond to loop corrections
in QED. As a proof of concept, we validate the theoretical predictions through
numerical simulations of standard multilayer perceptrons and, in parallel,
propose a gauge-invariant neural network (GINN) implementation using
magnitude--phase parameterization of weights. Finally, a double-copy replica
approach is shown to unify the computation of the largest Lyapunov exponent in
stochastic QED and wide DNNs.

### 7. [Composition and Alignment of Diffusion Models using Constrained Learning](http://arxiv.org/pdf/2508.19104v1)

Authors: Shervin Khalafi, Ignacio Hounie, Dongsheng Ding, Alejandro Ribeiro

Diffusion models have become prevalent in generative modeling due to their
ability to sample from complex distributions. To improve the quality of
generated samples and their compliance with user requirements, two commonly
used methods are: (i) Alignment, which involves fine-tuning a diffusion model
to align it with a reward; and (ii) Composition, which combines several
pre-trained diffusion models, each emphasizing a desirable attribute in the
generated outputs. However, trade-offs often arise when optimizing for multiple
rewards or combining multiple models, as they can often represent competing
properties. Existing methods cannot guarantee that the resulting model
faithfully generates samples with all the desired properties. To address this
gap, we propose a constrained optimization framework that unifies alignment and
composition of diffusion models by enforcing that the aligned model satisfies
reward constraints and/or remains close to (potentially multiple) pre-trained
models. We provide a theoretical characterization of the solutions to the
constrained alignment and composition problems and develop a Lagrangian-based
primal-dual training algorithm to approximate these solutions. Empirically, we
demonstrate the effectiveness and merits of our proposed approach in image
generation, applying it to alignment and composition, and show that our aligned
or composed model satisfies constraints effectively, and improves on the
equally-weighted approach. Our implementation can be found at
https://github.com/shervinkhalafi/constrained_comp_align.

### 8. [Echoes of the past: A unified perspective on fading memory and echo states](http://arxiv.org/pdf/2508.19145v1)

Authors: Juan-Pablo Ortega, Florian Rossmannek

Recurrent neural networks (RNNs) have become increasingly popular in
information processing tasks involving time series and temporal data. A
fundamental property of RNNs is their ability to create reliable input/output
responses, often linked to how the network handles its memory of the
information it processed. Various notions have been proposed to conceptualize
the behavior of memory in RNNs, including steady states, echo states, state
forgetting, input forgetting, and fading memory. Although these notions are
often used interchangeably, their precise relationships remain unclear. This
work aims to unify these notions in a common language, derive new implications
and equivalences between them, and provide alternative proofs to some existing
results. By clarifying the relationships between these concepts, this research
contributes to a deeper understanding of RNNs and their temporal information
processing capabilities.

### 9. [Understanding Tool-Integrated Reasoning](http://arxiv.org/pdf/2508.19201v1)

Authors: Heng Lin, Zhongwen Xu

We study why Tool-Integrated Reasoning (TIR) makes Large Language Models
(LLMs) more capable. While LLMs integrated with tools like Python code
interpreters show great promise, a principled theory explaining why this
paradigm is effective has been missing. This work provides the first formal
proof that TIR fundamentally expands an LLM's capabilities. We demonstrate that
tools enable a strict expansion of the model's empirical and feasible support,
breaking the capability ceiling of pure-text models by unlocking
problem-solving strategies that are otherwise impossible or intractably
verbose. To guide model behavior without compromising training stability and
performance, we also introduce Advantage Shaping Policy Optimization (ASPO), a
novel algorithm that directly modifies the advantage function to guide the
policy behavior. We conduct comprehensive experiments on challenging
mathematical benchmarks, leveraging a Python interpreter as the external tool.
Our results show that the TIR model decisively outperforms its pure-text
counterpart on the pass@k metric. Crucially, this advantage is not confined to
computationally-intensive problems but extends to those requiring significant
abstract insight. We further identify the emergent cognitive patterns that
illustrate how models learn to think with tools. Finally, we report improved
tool usage behavior with early code invocation and much more interactive turns
with ASPO. Overall, our work provides the first principled explanation for
TIR's success, shifting the focus from the mere fact that tools work to why and
how they enable more powerful reasoning.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-08-27 PST.

### 1. [Machine-learning model generates images using light](https://www.nature.com/articles/d41586-025-02523-9)

Authors: Daniel Brunner

### 2. [Performance of mental health chatbot agents in detecting and managing suicidal ideation](https://www.nature.com/articles/s41598-025-17242-4)

Authors: W. Pichowicz et al.

### 3. [Light weight blockchain with IoT devices to secure smart non-fungible tokens using hybrid secure functions](https://www.nature.com/articles/s41598-025-16945-y)

Authors: Lanye Wei et al.

### 4. [Heterogeneous dual-decoder network for road extraction in remote sensing images](https://www.nature.com/articles/s41598-025-17445-9)

Authors: Shenming Qu et al.

### 5. [Hierarchical query design and distributed attention in transformer for player group activity recognition in sports analysis](https://www.nature.com/articles/s41598-025-16752-5)

Authors: Xiao Chen et al.

### 6. [Chinese crop diseases and pests named entity recognition based on variational information bottleneck and feature enhancement](https://www.nature.com/articles/s41598-025-04252-5)

Authors: Runqing Huang et al.

### 7. [Cox proportional hazards model with Bayesian neural network for survival prediction](https://www.nature.com/articles/s41598-025-16993-4)

Authors: Fojan Faghiri et al.

### 8. [Aspect-level multimodal sentiment analysis model based on multi-scale feature extraction](https://www.nature.com/articles/s41598-025-16051-z)

Authors: Bocheng Miao et al.

### 9. [Quantum integration in swin transformer mitigates overfitting in breast cancer screening](https://www.nature.com/articles/s41598-025-17075-1)

Authors: Zongyu Xie et al.

