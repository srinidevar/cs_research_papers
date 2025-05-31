# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-30 17:06:57.110035 PST.

### Artificial Intelligence

### 1. [Foundation Molecular Grammar: Multi-Modal Foundation Models Induce Interpretable Molecular Graph Languages](http://arxiv.org/pdf/2505.22948v1)

Authors: Michael Sun, Weize Yuan, Gang Liu, Wojciech Matusik, Jie Chen

Recent data-efficient molecular generation approaches exploit graph grammars
to introduce interpretability into the generative models. However, grammar
learning therein relies on expert annotation or unreliable heuristics for
algorithmic inference. We propose Foundation Molecular Grammar (FMG), which
leverages multi-modal foundation models (MMFMs) to induce an interpretable
molecular language. By exploiting the chemical knowledge of an MMFM, FMG
renders molecules as images, describes them as text, and aligns information
across modalities using prompt learning. FMG can be used as a drop-in
replacement for the prior grammar learning approaches in molecular generation
and property prediction. We show that FMG not only excels in synthesizability,
diversity, and data efficiency but also offers built-in chemical
interpretability for automated molecular discovery workflows. Code is available
at https://github.com/shiningsunnyday/induction.

### 2. [Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents](http://arxiv.org/pdf/2505.22954v1)

Authors: Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune

Today's AI systems have human-designed, fixed architectures and cannot
autonomously and continuously improve themselves. The advance of AI could
itself be automated. If done safely, that would accelerate AI development and
allow us to reap its benefits much sooner. Meta-learning can automate the
discovery of novel algorithms, but is limited by first-order improvements and
the human design of a suitable search space. The G\"odel machine proposed a
theoretical alternative: a self-improving AI that repeatedly modifies itself in
a provably beneficial manner. Unfortunately, proving that most changes are net
beneficial is impossible in practice. We introduce the Darwin G\"odel Machine
(DGM), a self-improving system that iteratively modifies its own code (thereby
also improving its ability to modify its own codebase) and empirically
validates each change using coding benchmarks. Inspired by Darwinian evolution
and open-endedness research, the DGM maintains an archive of generated coding
agents. It grows the archive by sampling an agent from it and using a
foundation model to create a new, interesting, version of the sampled agent.
This open-ended exploration forms a growing tree of diverse, high-quality
agents and allows the parallel exploration of many different paths through the
search space. Empirically, the DGM automatically improves its coding
capabilities (e.g., better code editing tools, long-context window management,
peer-review mechanisms), increasing performance on SWE-bench from 20.0% to
50.0%, and on Polyglot from 14.2% to 30.7%. Furthermore, the DGM significantly
outperforms baselines without self-improvement or open-ended exploration. All
experiments were done with safety precautions (e.g., sandboxing, human
oversight). The DGM is a significant step toward self-improving AI, capable of
gathering its own stepping stones along paths that unfold into endless
innovation.

### 3. [Conceptual Framework Toward Embodied Collective Adaptive Intelligence](http://arxiv.org/pdf/2505.23153v1)

Authors: Fan Wang, Shaoshan Liu

Collective Adaptive Intelligence (CAI) represent a transformative approach in
artificial intelligence, wherein numerous autonomous agents collaborate, adapt,
and self-organize to navigate complex, dynamic environments. This paradigm is
particularly impactful in embodied AI applications, where adaptability and
resilience are paramount. By enabling systems to reconfigure themselves in
response to unforeseen challenges, CAI facilitate robust performance in
real-world scenarios. This article introduces a conceptual framework for
designing and analyzing CAI. It delineates key attributes including task
generalization, resilience, scalability, and self-assembly, aiming to bridge
theoretical foundations with practical methodologies for engineering adaptive,
emergent intelligence. By providing a structured foundation for understanding
and implementing CAI, this work seeks to guide researchers and practitioners in
developing more resilient, scalable, and adaptable AI systems across various
domains.

### 4. [MathArena: Evaluating LLMs on Uncontaminated Math Competitions](http://arxiv.org/pdf/2505.23281v1)

Authors: Mislav Balunović, Jasper Dekoninck, Ivo Petrov, Nikola Jovanović, Martin Vechev

The rapid advancement of reasoning capabilities in large language models
(LLMs) has led to notable improvements on mathematical benchmarks. However,
many of the most commonly used evaluation datasets (e.g., AIME 2024) are widely
available online, making it difficult to disentangle genuine reasoning from
potential memorization. Furthermore, these benchmarks do not evaluate
proof-writing capabilities, which are crucial for many mathematical tasks. To
address this, we introduce MathArena, a new benchmark based on the following
key insight: recurring math competitions provide a stream of high-quality,
challenging problems that can be used for real-time evaluation of LLMs. By
evaluating models as soon as new problems are released, we effectively
eliminate the risk of contamination. Using this framework, we find strong signs
of contamination in AIME 2024. Nonetheless, evaluations on harder competitions,
such as SMT 2025 -- published well after model release dates -- demonstrate
impressive reasoning capabilities in top-performing models. MathArena is also
the first benchmark for proof-writing capabilities. On USAMO 2025, even top
models score below 25%, far behind their performance on final-answer tasks. So
far, we have evaluated 30 models across five competitions, totaling 149
problems. As an evolving benchmark, MathArena will continue to track the
progress of LLMs on newly released competitions, ensuring rigorous and
up-to-date evaluation of mathematical reasoning.

### 5. [AutoGPS: Automated Geometry Problem Solving via Multimodal Formalization and Deductive Reasoning](http://arxiv.org/pdf/2505.23381v1)

Authors: Bowen Ping, Minnan Luo, Zhuohang Dang, Chenxi Wang, Chengyou Jia

Geometry problem solving presents distinctive challenges in artificial
intelligence, requiring exceptional multimodal comprehension and rigorous
mathematical reasoning capabilities. Existing approaches typically fall into
two categories: neural-based and symbolic-based methods, both of which exhibit
limitations in reliability and interpretability. To address this challenge, we
propose AutoGPS, a neuro-symbolic collaborative framework that solves geometry
problems with concise, reliable, and human-interpretable reasoning processes.
Specifically, AutoGPS employs a Multimodal Problem Formalizer (MPF) and a
Deductive Symbolic Reasoner (DSR). The MPF utilizes neural cross-modal
comprehension to translate geometry problems into structured formal language
representations, with feedback from DSR collaboratively. The DSR takes the
formalization as input and formulates geometry problem solving as a hypergraph
expansion task, executing mathematically rigorous and reliable derivation to
produce minimal and human-readable stepwise solutions. Extensive experimental
evaluations demonstrate that AutoGPS achieves state-of-the-art performance on
benchmark datasets. Furthermore, human stepwise-reasoning evaluation confirms
AutoGPS's impressive reliability and interpretability, with 99\% stepwise
logical coherence. The project homepage is at
https://jayce-ping.github.io/AutoGPS-homepage.

### 6. [GAM-Agent: Game-Theoretic and Uncertainty-Aware Collaboration for Complex Visual Reasoning](http://arxiv.org/pdf/2505.23399v1)

Authors: Jusheng Zhang, Yijia Fan, Wenjun Lin, Ruiqi Chen, Haoyi Jiang, Wenhao Chai, Jian Wang, Keze Wang

We propose GAM-Agent, a game-theoretic multi-agent framework for enhancing
vision-language reasoning. Unlike prior single-agent or monolithic models,
GAM-Agent formulates the reasoning process as a non-zero-sum game between base
agents--each specializing in visual perception subtasks--and a critical agent
that verifies logic consistency and factual correctness. Agents communicate via
structured claims, evidence, and uncertainty estimates. The framework
introduces an uncertainty-aware controller to dynamically adjust agent
collaboration, triggering multi-round debates when disagreement or ambiguity is
detected. This process yields more robust and interpretable predictions.
Experiments on four challenging benchmarks--MMMU, MMBench, MVBench, and
V*Bench--demonstrate that GAM-Agent significantly improves performance across
various VLM backbones. Notably, GAM-Agent boosts the accuracy of small-to-mid
scale models (e.g., Qwen2.5-VL-7B, InternVL3-14B) by 5--6\%, and still enhances
strong models like GPT-4o by up to 2--3\%. Our approach is modular, scalable,
and generalizable, offering a path toward reliable and explainable multi-agent
multimodal reasoning.

### 7. [EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions](http://arxiv.org/pdf/2505.23473v1)

Authors: Xiaorui Wu, Xiaofeng Mao, Fei Li, Xin Zhang, Xiaolu Zhang, Jun Zhou, Yuxiang Peng, Li Zheng, Chong Teng, Donghong Ji, Zhuang Li

Large language models (LLMs) frequently refuse to respond to pseudo-malicious
instructions: semantically harmless input queries triggering unnecessary LLM
refusals due to conservative safety alignment, significantly impairing user
experience. Collecting such instructions is crucial for evaluating and
mitigating over-refusals, but existing instruction curation methods, like
manual creation or instruction rewriting, either lack scalability or fail to
produce sufficiently diverse and effective refusal-inducing prompts. To address
these limitations, we introduce EVOREFUSE, a prompt optimization approach that
generates diverse pseudo-malicious instructions consistently eliciting
confident refusals across LLMs. EVOREFUSE employs an evolutionary algorithm
exploring the instruction space in more diverse directions than existing
methods via mutation strategies and recombination, and iteratively evolves seed
instructions to maximize evidence lower bound on LLM refusal probability. Using
EVOREFUSE, we create two novel datasets: EVOREFUSE-TEST, a benchmark of 582
pseudo-malicious instructions that outperforms the next-best benchmark with
140.41% higher average refusal triggering rate across 9 LLMs, 34.86% greater
lexical diversity, and 40.03% improved LLM response confidence scores; and
EVOREFUSE-ALIGN, which provides 3,000 pseudo-malicious instructions with
responses for supervised and preference-based alignment training.
LLAMA3.1-8B-INSTRUCT supervisedly fine-tuned on EVOREFUSE-ALIGN achieves up to
14.31% fewer over-refusals than models trained on the second-best alignment
dataset, without compromising safety. Our analysis with EVOREFUSE-TEST reveals
models trigger over-refusals by overly focusing on sensitive keywords while
ignoring broader context.

### 8. [Autoformalization in the Era of Large Language Models: A Survey](http://arxiv.org/pdf/2505.23486v1)

Authors: Ke Weng, Lun Du, Sirui Li, Wangyue Lu, Haozhe Sun, Hengyu Liu, Tiancheng Zhang

Autoformalization, the process of transforming informal mathematical
propositions into verifiable formal representations, is a foundational task in
automated theorem proving, offering a new perspective on the use of mathematics
in both theoretical and applied domains. Driven by the rapid progress in
artificial intelligence, particularly large language models (LLMs), this field
has witnessed substantial growth, bringing both new opportunities and unique
challenges. In this survey, we provide a comprehensive overview of recent
advances in autoformalization from both mathematical and LLM-centric
perspectives. We examine how autoformalization is applied across various
mathematical domains and levels of difficulty, and analyze the end-to-end
workflow from data preprocessing to model design and evaluation. We further
explore the emerging role of autoformalization in enhancing the verifiability
of LLM-generated outputs, highlighting its potential to improve both the
trustworthiness and reasoning capabilities of LLMs. Finally, we summarize key
open-source models and datasets supporting current research, and discuss open
challenges and promising future directions for the field.

### 9. [TRAP: Targeted Redirecting of Agentic Preferences](http://arxiv.org/pdf/2505.23518v1)

Authors: Hangoo Kang, Jehyeok Yeon, Gagandeep Singh

Autonomous agentic AI systems powered by vision-language models (VLMs) are
rapidly advancing toward real-world deployment, yet their cross-modal reasoning
capabilities introduce new attack surfaces for adversarial manipulation that
exploit semantic reasoning across modalities. Existing adversarial attacks
typically rely on visible pixel perturbations or require privileged model or
environment access, making them impractical for stealthy, real-world
exploitation. We introduce TRAP, a generative adversarial framework that
manipulates the agent's decision-making using diffusion-based semantic
injections. Our method combines negative prompt-based degradation with positive
semantic optimization, guided by a Siamese semantic network and layout-aware
spatial masking. Without requiring access to model internals, TRAP produces
visually natural images yet induces consistent selection biases in agentic AI
systems. We evaluate TRAP on the Microsoft Common Objects in Context (COCO)
dataset, building multi-candidate decision scenarios. Across these scenarios,
TRAP achieves a 100% attack success rate on leading models, including
LLaVA-34B, Gemma3, and Mistral-3.1, significantly outperforming baselines such
as SPSA, Bandit, and standard diffusion approaches. These results expose a
critical vulnerability: Autonomous agents can be consistently misled through
human-imperceptible cross-modal manipulations. These findings highlight the
need for defense strategies beyond pixel-level robustness to address semantic
vulnerabilities in cross-modal decision-making.

### 10. [Individual differences in the cognitive mechanisms of planning strategy discovery](http://arxiv.org/pdf/2505.23519v1)

Authors: Ruiqi He, Falk Lieder

People employ efficient planning strategies. But how are these strategies
acquired? Previous research suggests that people can discover new planning
strategies through learning from reinforcements, a process known as
metacognitive reinforcement learning (MCRL). While prior work has shown that
MCRL models can learn new planning strategies and explain more participants'
experience-driven discovery better than alternative mechanisms, it also
revealed significant individual differences in metacognitive learning.
Furthermore, when fitted to human data, these models exhibit a slower rate of
strategy discovery than humans. In this study, we investigate whether
incorporating cognitive mechanisms that might facilitate human strategy
discovery can bring models of MCRL closer to human performance. Specifically,
we consider intrinsically generated metacognitive pseudo-rewards, subjective
effort valuation, and termination deliberation. Analysis of planning task data
shows that a larger proportion of participants used at least one of these
mechanisms, with significant individual differences in their usage and varying
impacts on strategy discovery. Metacognitive pseudo-rewards, subjective effort
valuation, and learning the value of acting without further planning were found
to facilitate strategy discovery. While these enhancements provided valuable
insights into individual differences and the effect of these mechanisms on
strategy discovery, they did not fully close the gap between model and human
performance, prompting further exploration of additional factors that people
might use to discover new planning strategies.

### Hardware Architecture

### 1. [DX100: A Programmable Data Access Accelerator for Indirection](http://arxiv.org/pdf/2505.23073v1)

Authors: Alireza Khadem, Kamalavasan Kamalakkannan, Zhenyan Zhu, Akash Poptani, Yufeng Gu, Jered Benjamin Dominguez-Trujillo, Nishil Talati, Daichi Fujiki, Scott Mahlke, Galen Shipman, Reetuparna Das

Indirect memory accesses frequently appear in applications where memory
bandwidth is a critical bottleneck. Prior indirect memory access proposals,
such as indirect prefetchers, runahead execution, fetchers, and decoupled
access/execute architectures, primarily focus on improving memory access
latency by loading data ahead of computation but still rely on the DRAM
controllers to reorder memory requests and enhance memory bandwidth
utilization. DRAM controllers have limited visibility to future memory accesses
due to the small capacity of request buffers and the restricted memory-level
parallelism of conventional core and memory systems. We introduce DX100, a
programmable data access accelerator for indirect memory accesses. DX100 is
shared across cores to offload bulk indirect memory accesses and associated
address calculation operations. DX100 reorders, interleaves, and coalesces
memory requests to improve DRAM row-buffer hit rate and memory bandwidth
utilization. DX100 provides a general-purpose ISA to support diverse access
types, loop patterns, conditional accesses, and address calculations. To
support this accelerator without significant programming efforts, we discuss a
set of MLIR compiler passes that automatically transform legacy code to utilize
DX100. Experimental evaluations on 12 benchmarks spanning scientific computing,
database, and graph applications show that DX100 achieves performance
improvements of 2.6x over a multicore baseline and 2.0x over the
state-of-the-art indirect prefetcher.

### 2. [Energy-Efficient QoS-Aware Scheduling for S-NUCA Many-Cores](http://arxiv.org/pdf/2505.23351v1)

Authors: Sudam M. Wasala, Jurre Wolff, Yixian Shen, Anuj Pathania, Clemens Grelck, Andy D. Pimentel

Optimizing performance and energy efficiency in many-core processors,
especially within Non-Uniform Cache Access (NUCA) architectures, remains a
critical challenge. The performance heterogeneity inherent in S-NUCA systems
complicates task scheduling due to varying cache access latencies across cores.
This paper introduces a novel QoS management policy to maintain application
execution within predefined Quality of Service (QoS) targets, measured using
the Application Heartbeats framework. QoS metrics like Heartbeats ensure
predictable application performance in dynamic computing environments. The
proposed policy dynamically controls QoS by orchestrating task migrations
within the S-NUCA many-core system and adjusting the clock frequency of cores.
After satisfying the QoS objectives, the policy optimizes energy efficiency,
reducing overall system energy consumption without compromising performance
constraints. Our work leverages the state-of-the-art multi-/many-core simulator
{\em HotSniper}. We have extended it with two key components: an integrated
heartbeat framework for precise, application-specific performance monitoring,
and our QoS management policy that maintains application QoS requirements while
minimizing the system's energy consumption. Experimental evaluations
demonstrate that our approach effectively maintains desired QoS levels and
achieves 18.7\% energy savings compared to state-of-the-art scheduling methods.

### 3. [A Novel Cost-Effective MIMO Architecture with Ray Antenna Array for Enhanced Wireless Communication Performance](http://arxiv.org/pdf/2505.23394v1)

Authors: Zhenjun Dong, Zhiwen Zhou, Yong Zeng

This paper proposes a novel multi-antenna architecture, termed ray antenna
array (RAA), which practically enables flexible beamforming and also enhances
wireless communication performance for high frequency systems in a
cost-effective manner. RAA consists of a large number of inexpensive antenna
elements and a few radio frequency (RF) chains. These antenna elements are
arranged in a novel ray like structure, where each ray corresponds to one
simple uniform linear array (sULA) with a carefully designed orientation. The
antenna elements within each sULA are directly connected, so that each sULA is
able to form a beam towards a direction matching the ray orientation without
relying on any analog or digital beamforming. By further designing a ray
selection network (RSN), appropriate sULAs are selected to connect to the RF
chains for subsequent baseband processing. Compared to conventional
multi-antenna architectures such as the uniform linear array (ULA) with hybrid
analog/digital beamforming (HBF), the proposed RAA enjoys three appealing
advantages: (i) finer and uniform angular resolution for all signal directions;
(ii) enhanced beamforming gain by using antenna elements with higher
directivity, as each sULA is only responsible for a small portion of the total
angle coverage range; and (iii) dramatically reduced hardware cost since no
phase shifters are required, which are expensive and difficult to design in
high-frequency systems such as mmWave and THz systems. To validate such
advantages, we first present the input-output mathematical model for RAA-based
wireless communications. Efficient algorithms for joint RAA beamforming and ray
selection are then proposed for single-user and multi-user RAA-based wireless
communications. Simulation results demonstrate that RAA achieves superior
performance compared to the conventional ULA with HBF, while significantly
reducing hardware cost.

### 4. [A Unified Framework for Mapping and Synthesis of Approximate R-Blocks CGRAs](http://arxiv.org/pdf/2505.23553v1)

Authors: Georgios Alexandris, Panagiotis Chaidos, Alexis Maras, Barry de Bruin, Manil Dev Gomony, Henk Corporaal, Dimitrios Soudris, Sotirios Xydis

The ever-increasing complexity and operational diversity of modern Neural
Networks (NNs) have caused the need for low-power and, at the same time,
high-performance edge devices for AI applications. Coarse Grained
Reconfigurable Architectures (CGRAs) form a promising design paradigm to
address these challenges, delivering a close-to-ASIC performance while allowing
for hardware programmability. In this paper, we introduce a novel end-to-end
exploration and synthesis framework for approximate CGRA processors that
enables transparent and optimized integration and mapping of state-of-the-art
approximate multiplication components into CGRAs. Our methodology introduces a
per-channel exploration strategy that maps specific output features onto
approximate components based on accuracy degradation constraints. This enables
the optimization of the system's energy consumption while retaining the
accuracy above a certain threshold. At the circuit level, the integration of
approximate components enables the creation of voltage islands that operate at
reduced voltage levels, which is attributed to their inherently shorter
critical paths. This key enabler allows us to effectively reduce the overall
power consumption by an average of 30% across our analyzed architectures,
compared to their baseline counterparts, while incurring only a minimal 2% area
overhead. The proposed methodology was evaluated on a widely used NN model,
MobileNetV2, on the ImageNet dataset, demonstrating that the generated
architectures can deliver up to 440 GOPS/W with relatively small output error
during inference, outperforming several State-of-the-Art CGRA architectures in
terms of throughput and energy efficiency.

### 5. [Towards LLM-based Generation of Human-Readable Proofs in Polynomial Formal Verification](http://arxiv.org/pdf/2505.23311v1)

Authors: Rolf Drechsler

Verification is one of the central tasks in circuit and system design. While
simulation and emulation are widely used, complete correctness can only be
ensured based on formal proof techniques. But these approaches often have very
high run time and memory requirements. Recently, Polynomial Formal Verification
(PFV) has been introduced showing that for many instances of practical
relevance upper bounds on needed resources can be given. But proofs have to be
provided that are human-readable.
  Here, we study how modern approaches from Artificial Intelligence (AI) based
on Large Language Models (LLMs) can be used to generate proofs that later on
can be validated based on reasoning engines. Examples are given that show how
LLMs can interact with proof engines, and directions for future work are
outlined.

### Computational Complexity

### 1. [Fast Compressed-Domain N-Point Discrete Fourier Transform](http://arxiv.org/pdf/2505.23718v1)

Authors: Saulo Queiroz

This paper presents a novel algorithm for computing the N-point Discrete
Fourier Transform (DFT) based solely on recursive Rectangular Index Compression
(RIC) [1][2] and structured frequency shifts. The RIC DFT algorithm compresses
a signal from $N=CL$ to $C\in[2,N/2]$ points at the expense of $N-1$ complex
additions and no complex multiplication. It is shown that a $C$-point DFT on
the compressed signal corresponds exactly to $C$ DFT coefficients of the
original $N$-point DFT, namely, $X_{kL}$, $k=0,1,\ldots,C-1$ with no need for
twiddle factors. We rely on this strategy to decompose the DFT by recursively
compressing the input signal and applying global frequency shifts (to get
odd-indexed DFT coefficients). We show that this new structure can relax the
power-of-two assumption of the radix-2 FFT by enabling signal input lengths
such as $N=c\cdot 2^k$ (for $k\geq 0$ and a non-power-of-two $c>0$). Thus, our
algorithm potentially outperforms radix-2 FFTs for the cases where significant
zero-padding is needed. The proposed approach achieves a computational
complexity of $O(N \log N)$ and offers a new structural perspective on DFT
computation, with potential impacts on several DFT issues like numerical
stability, hardware implementation, sparse transforms, convolutions, and others
DFT-based procedures.

### Computational Engineering

### 1. [Hybrid subgradient and simulated annealing method for hemivariational inequalities](http://arxiv.org/pdf/2505.23676v1)

Authors: Piotr Bartman-Szwarc, Adil M. Bagirov, Anna Ochal

In this paper, we employ a global aggregate subgradient method for the
numerical solution of hemivariational inequality problems arising in contact
mechanics. The method integrates a global search procedure to identify starting
points for a local minimization algorithm. The algorithm consists of two types
of steps: null steps and serious steps. In each null step, only two
subgradients are utilized: the aggregate subgradient and the subgradient
computed at the current iteration point, which together determine the search
direction. Furthermore, we compare the performance of the proposed method with
selected solvers using a representative contact mechanics problem as a case
study.

### 2. [Computerized Modeling of Electrophysiology and Pathoelectrophysiology of the Atria -- How Much Detail is Needed?](http://arxiv.org/pdf/2505.23717v1)

Authors: Olaf Dössel, Axel Loewe

This review focuses on the computerized modeling of the electrophysiology of
the human atria, emphasizing the simulation of common arrhythmias such as
atrial flutter (AFlut) and atrial fibrillation (AFib). Which components of the
model are necessary to accurately model arrhythmogenic tissue modifications,
including remodeling, cardiomyopathy, and fibrosis, to ensure reliable
simulations? The central question explored is the level of detail required for
trustworthy simulations for a specific context of use. The review discusses the
balance between model complexity and computational efficiency, highlighting the
risks of oversimplification and excessive detail. It covers various aspects of
atrial modeling, from cellular to whole atria levels, including the influence
of atrial geometry, fiber direction, anisotropy, and wall thickness on
simulation outcomes. The article also examines the impact of different modeling
approaches, such as volumetric 3D models, bilayer models, and single surface
models, on the realism of simulations. In addition, it reviews the latest
advances in the modeling of fibrotic tissue and the verification and validation
of atrial models. The intended use of these models in planning and optimization
of atrial ablation strategies is discussed, with a focus on personalized
modeling for individual patients and cohort-based approaches for broader
applications. The review concludes by emphasizing the importance of integrating
experimental data and clinical validation to enhance the utility of
computerized atrial models to improve patient outcomes.

### 3. [Be.FM: Open Foundation Models for Human Behavior](http://arxiv.org/pdf/2505.23058v1)

Authors: Yutong Xie, Zhuoheng Li, Xiyuan Wang, Yijun Pan, Qijia Liu, Xingzhi Cui, Kuang-Yu Lo, Ruoyi Gao, Xingjian Zhang, Jin Huang, Walter Yuan, Matthew O. Jackson, Qiaozhu Mei

Despite their success in numerous fields, the potential of foundation models
for modeling and understanding human behavior remains largely unexplored. We
introduce Be.FM, one of the first open foundation models designed for human
behavior modeling. Built upon open-source large language models and fine-tuned
on a diverse range of behavioral data, Be.FM can be used to understand and
predict human decision-making. We construct a comprehensive set of benchmark
tasks for testing the capabilities of behavioral foundation models. Our results
demonstrate that Be.FM can predict behaviors, infer characteristics of
individuals and populations, generate insights about contexts, and apply
behavioral science knowledge.

### Computational Geometry

### 1. [Computing Non-Obtuse Triangulations with Few Steiner Points](http://arxiv.org/pdf/2505.23375v1)

Authors: Mikkel Abrahamsen, Florestan Brunck, Jacobus Conradi, Benedikt Kolbe, André Nusser

We present the winning implementation of the Seventh Computational Geometry
Challenge (CG:SHOP 2025). The task in this challenge was to find non-obtuse
triangulations for given planar regions, respecting a given set of constraints
consisting of extra vertices and edges that must be part of the triangulation.
The goal was to minimize the number of introduced Steiner points. Our approach
is to maintain a constrained Delaunay triangulation, for which we repeatedly
remove, relocate, or add Steiner points. We use local search to choose the
action that improves the triangulation the most, until the resulting
triangulation is non-obtuse.

### 2. [AMBER: Adaptive Mesh Generation by Iterative Mesh Resolution Prediction](http://arxiv.org/pdf/2505.23663v1)

Authors: Niklas Freymuth, Tobias Würth, Nicolas Schreiber, Balazs Gyenes, Andreas Boltres, Johannes Mitsch, Aleksandar Taranovic, Tai Hoang, Philipp Dahlinger, Philipp Becker, Luise Kärger, Gerhard Neumann

The cost and accuracy of simulating complex physical systems using the Finite
Element Method (FEM) scales with the resolution of the underlying mesh.
Adaptive meshes improve computational efficiency by refining resolution in
critical regions, but typically require task-specific heuristics or cumbersome
manual design by a human expert. We propose Adaptive Meshing By Expert
Reconstruction (AMBER), a supervised learning approach to mesh adaptation.
Starting from a coarse mesh, AMBER iteratively predicts the sizing field, i.e.,
a function mapping from the geometry to the local element size of the target
mesh, and uses this prediction to produce a new intermediate mesh using an
out-of-the-box mesh generator. This process is enabled through a hierarchical
graph neural network, and relies on data augmentation by automatically
projecting expert labels onto AMBER-generated data during training. We evaluate
AMBER on 2D and 3D datasets, including classical physics problems, mechanical
components, and real-world industrial designs with human expert meshes. AMBER
generalizes to unseen geometries and consistently outperforms multiple recent
baselines, including ones using Graph and Convolutional Neural Networks, and
Reinforcement Learning-based approaches.

### 3. [Improved Learning via k-DTW: A Novel Dissimilarity Measure for Curves](http://arxiv.org/pdf/2505.23431v1)

Authors: Amer Krivošija, Alexander Munteanu, André Nusser, Chris Schwiegelshohn

This paper introduces $k$-Dynamic Time Warping ($k$-DTW), a novel
dissimilarity measure for polygonal curves. $k$-DTW has stronger metric
properties than Dynamic Time Warping (DTW) and is more robust to outliers than
the Fr\'{e}chet distance, which are the two gold standards of dissimilarity
measures for polygonal curves. We show interesting properties of $k$-DTW and
give an exact algorithm as well as a $(1+\varepsilon)$-approximation algorithm
for $k$-DTW by a parametric search for the $k$-th largest matched distance. We
prove the first dimension-free learning bounds for curves and further learning
theoretic results. $k$-DTW not only admits smaller sample size than DTW for the
problem of learning the median of curves, where some factors depending on the
curves' complexity $m$ are replaced by $k$, but we also show a surprising
separation on the associated Rademacher and Gaussian complexities: $k$-DTW
admits strictly smaller bounds than DTW, by a factor $\tilde\Omega(\sqrt{m})$
when $k\ll m$. We complement our theoretical findings with an experimental
illustration of the benefits of using $k$-DTW for clustering and nearest
neighbor classification.

### Computation and Language

### 1. [StrucSum: Graph-Structured Reasoning for Long Document Extractive Summarization with LLMs](http://arxiv.org/pdf/2505.22950v1)

Authors: Haohan Yuan, Sukhwa Hong, Haopeng Zhang

Large language models (LLMs) have shown strong performance in zero-shot
summarization, but often struggle to model document structure and identify
salient information in long texts. In this work, we introduce StrucSum, a
training-free prompting framework that enhances LLM reasoning through
sentence-level graph structures. StrucSum injects structural signals into
prompts via three targeted strategies: Neighbor-Aware Prompting (NAP) for local
context, Centrality-Aware Prompting (CAP) for importance estimation, and
Centrality-Guided Masking (CGM) for efficient input reduction. Experiments on
ArXiv, PubMed, and Multi-News demonstrate that StrucSum consistently improves
both summary quality and factual consistency over unsupervised baselines and
vanilla prompting. Notably, on ArXiv, it boosts FactCC and SummaC by 19.2 and
9.7 points, indicating stronger alignment between summaries and source content.
These findings suggest that structure-aware prompting is a simple yet effective
approach for zero-shot extractive summarization with LLMs, without any training
or task-specific tuning.

### 2. [LLMs for Argument Mining: Detection, Extraction, and Relationship Classification of pre-defined Arguments in Online Comments](http://arxiv.org/pdf/2505.22956v1)

Authors: Matteo Guida, Yulia Otmakhova, Eduard Hovy, Lea Frermann

Automated large-scale analysis of public discussions around contested issues
like abortion requires detecting and understanding the use of arguments. While
Large Language Models (LLMs) have shown promise in language processing tasks,
their performance in mining topic-specific, pre-defined arguments in online
comments remains underexplored. We evaluate four state-of-the-art LLMs on three
argument mining tasks using datasets comprising over 2,000 opinion comments
across six polarizing topics. Quantitative evaluation suggests an overall
strong performance across the three tasks, especially for large and fine-tuned
LLMs, albeit at a significant environmental cost. However, a detailed error
analysis revealed systematic shortcomings on long and nuanced comments and
emotionally charged language, raising concerns for downstream applications like
content moderation or opinion analysis. Our results highlight both the promise
and current limitations of LLMs for automated argument analysis in online
comments.

### 3. [LLM-based HSE Compliance Assessment: Benchmark, Performance, and Advancements](http://arxiv.org/pdf/2505.22959v1)

Authors: Jianwei Wang, Mengqi Wang, Yinsi Zhou, Zhenchang Xing, Qing Liu, Xiwei Xu, Wenjie Zhang, Liming Zhu

Health, Safety, and Environment (HSE) compliance assessment demands dynamic
real-time decision-making under complicated regulations and complex
human-machine-environment interactions. While large language models (LLMs) hold
significant potential for decision intelligence and contextual dialogue, their
capacity for domain-specific knowledge in HSE and structured legal reasoning
remains underexplored. We introduce HSE-Bench, the first benchmark dataset
designed to evaluate the HSE compliance assessment capabilities of LLM.
HSE-Bench comprises over 1,000 manually curated questions drawn from
regulations, court cases, safety exams, and fieldwork videos, and integrates a
reasoning flow based on Issue spotting, rule Recall, rule Application, and rule
Conclusion (IRAC) to assess the holistic reasoning pipeline. We conduct
extensive evaluations on different prompting strategies and more than 10 LLMs,
including foundation models, reasoning models and multimodal vision models. The
results show that, although current LLMs achieve good performance, their
capabilities largely rely on semantic matching rather than principled reasoning
grounded in the underlying HSE compliance context. Moreover, their native
reasoning trace lacks the systematic legal reasoning required for rigorous HSE
compliance assessment. To alleviate these, we propose a new prompting
technique, Reasoning of Expert (RoE), which guides LLMs to simulate the
reasoning process of different experts for compliance assessment and reach a
more accurate unified decision. We hope our study highlights reasoning gaps in
LLMs for HSE compliance and inspires further research on related tasks.

### 4. [DyePack: Provably Flagging Test Set Contamination in LLMs Using Backdoors](http://arxiv.org/pdf/2505.23001v1)

Authors: Yize Cheng, Wenxiao Wang, Mazda Moayeri, Soheil Feizi

Open benchmarks are essential for evaluating and advancing large language
models, offering reproducibility and transparency. However, their accessibility
makes them likely targets of test set contamination. In this work, we introduce
DyePack, a framework that leverages backdoor attacks to identify models that
used benchmark test sets during training, without requiring access to the loss,
logits, or any internal details of the model. Like how banks mix dye packs with
their money to mark robbers, DyePack mixes backdoor samples with the test data
to flag models that trained on it. We propose a principled design incorporating
multiple backdoors with stochastic targets, enabling exact false positive rate
(FPR) computation when flagging every model. This provably prevents false
accusations while providing strong evidence for every detected case of
contamination. We evaluate DyePack on five models across three datasets,
covering both multiple-choice and open-ended generation tasks. For
multiple-choice questions, it successfully detects all contaminated models with
guaranteed FPRs as low as 0.000073% on MMLU-Pro and 0.000017% on Big-Bench-Hard
using eight backdoors. For open-ended generation tasks, it generalizes well and
identifies all contaminated models on Alpaca with a guaranteed false positive
rate of just 0.127% using six backdoors.

### 5. [Detecting Stealthy Backdoor Samples based on Intra-class Distance for Large Language Models](http://arxiv.org/pdf/2505.23015v1)

Authors: Jinwen Chen, Hainan Zhang, Fei Sun, Qinnan Zhang, Sijia Wen, Ziwei Wang, Zhiming Zheng

Fine-tuning LLMs with datasets containing stealthy backdoors from publishers
poses security risks to downstream applications. Mainstream detection methods
either identify poisoned samples by analyzing the prediction probability of
poisoned classification models or rely on the rewriting model to eliminate the
stealthy triggers. However, the former cannot be applied to generation tasks,
while the latter may degrade generation performance and introduce new triggers.
Therefore, efficiently eliminating stealthy poisoned samples for LLMs remains
an urgent problem. We observe that after applying TF-IDF clustering to the
sample response, there are notable differences in the intra-class distances
between clean and poisoned samples. Poisoned samples tend to cluster closely
because of their specific malicious outputs, whereas clean samples are more
scattered due to their more varied responses. Thus, in this paper, we propose a
stealthy backdoor sample detection method based on Reference-Filtration and
Tfidf-Clustering mechanisms (RFTC). Specifically, we first compare the sample
response with the reference model's outputs and consider the sample suspicious
if there's a significant discrepancy. And then we perform TF-IDF clustering on
these suspicious samples to identify the true poisoned samples based on the
intra-class distance. Experiments on two machine translation datasets and one
QA dataset demonstrate that RFTC outperforms baselines in backdoor detection
and model performance. Further analysis of different reference models also
confirms the effectiveness of our Reference-Filtration.

### 6. [Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Spac](http://arxiv.org/pdf/2505.23029v1)

Authors: Si Wu, Sebastian Bruch

Imageability (potential of text to evoke a mental image) and concreteness
(perceptibility of text) are two psycholinguistic properties that link visual
and semantic spaces. It is little surprise that computational methods that
estimate them do so using parallel visual and semantic spaces, such as
collections of image-caption pairs or multi-modal models. In this paper, we
work on the supposition that text itself in an image-caption dataset offers
sufficient signals to accurately estimate these properties. We hypothesize, in
particular, that the peakedness of the neighborhood of a word in the semantic
embedding space reflects its degree of imageability and concreteness. We then
propose an unsupervised, distribution-free measure, which we call Neighborhood
Stability Measure (NSM), that quantifies the sharpness of peaks. Extensive
experiments show that NSM correlates more strongly with ground-truth ratings
than existing unsupervised methods, and is a strong predictor of these
properties for classification. Our code and data are available on GitHub
(https://github.com/Artificial-Memory-Lab/imageability).

### 7. [Can Modern NLP Systems Reliably Annotate Chest Radiography Exams? A Pre-Purchase Evaluation and Comparative Study of Solutions from AWS, Google, Azure, John Snow Labs, and Open-Source Models on an Independent Pediatric Dataset](http://arxiv.org/pdf/2505.23030v1)

Authors: Shruti Hegde, Mabon Manoj Ninan, Jonathan R. Dillman, Shireen Hayatghaibi, Lynn Babcock, Elanchezhian Somasundaram

General-purpose clinical natural language processing (NLP) tools are
increasingly used for the automatic labeling of clinical reports. However,
independent evaluations for specific tasks, such as pediatric chest radiograph
(CXR) report labeling, are limited. This study compares four commercial
clinical NLP systems - Amazon Comprehend Medical (AWS), Google Healthcare NLP
(GC), Azure Clinical NLP (AZ), and SparkNLP (SP) - for entity extraction and
assertion detection in pediatric CXR reports. Additionally, CheXpert and
CheXbert, two dedicated chest radiograph report labelers, were evaluated on the
same task using CheXpert-defined labels. We analyzed 95,008 pediatric CXR
reports from a large academic pediatric hospital. Entities and assertion
statuses (positive, negative, uncertain) from the findings and impression
sections were extracted by the NLP systems, with impression section entities
mapped to 12 disease categories and a No Findings category. CheXpert and
CheXbert extracted the same 13 categories. Outputs were compared using Fleiss
Kappa and accuracy against a consensus pseudo-ground truth. Significant
differences were found in the number of extracted entities and assertion
distributions across NLP systems. SP extracted 49,688 unique entities, GC
16,477, AZ 31,543, and AWS 27,216. Assertion accuracy across models averaged
around 62%, with SP highest (76%) and AWS lowest (50%). CheXpert and CheXbert
achieved 56% accuracy. Considerable variability in performance highlights the
need for careful validation and review before deploying NLP tools for clinical
report labeling.

### 8. [Machine-Facing English: Defining a Hybrid Register Shaped by Human-AI Discourse](http://arxiv.org/pdf/2505.23035v1)

Authors: Hyunwoo Kim, Hanau Yi

Machine-Facing English (MFE) is an emergent register shaped by the adaptation
of everyday language to the expanding presence of AI interlocutors. Drawing on
register theory (Halliday 1985, 2006), enregisterment (Agha 2003), audience
design (Bell 1984), and interactional pragmatics (Giles & Ogay 2007), this
study traces how sustained human-AI interaction normalizes syntactic rigidity,
pragmatic simplification, and hyper-explicit phrasing - features that enhance
machine parseability at the expense of natural fluency. Our analysis is
grounded in qualitative observations from bilingual (Korean/English) voice- and
text-based product testing sessions, with reflexive drafting conducted using
Natural Language Declarative Prompting (NLD-P) under human curation. Thematic
analysis identifies five recurrent traits - redundant clarity, directive
syntax, controlled vocabulary, flattened prosody, and single-intent structuring
- that improve execution accuracy but compress expressive range. MFE's
evolution highlights a persistent tension between communicative efficiency and
linguistic richness, raising design challenges for conversational interfaces
and pedagogical considerations for multilingual users. We conclude by
underscoring the need for comprehensive methodological exposition and future
empirical validation.

### 9. [Improving Multilingual Social Media Insights: Aspect-based Comment Analysis](http://arxiv.org/pdf/2505.23037v1)

Authors: Longyin Zhang, Bowei Zou, Ai Ti Aw

The inherent nature of social media posts, characterized by the freedom of
language use with a disjointed array of diverse opinions and topics, poses
significant challenges to downstream NLP tasks such as comment clustering,
comment summarization, and social media opinion analysis. To address this, we
propose a granular level of identifying and generating aspect terms from
individual comments to guide model attention. Specifically, we leverage
multilingual large language models with supervised fine-tuning for comment
aspect term generation (CAT-G), further aligning the model's predictions with
human expectations through DPO. We demonstrate the effectiveness of our method
in enhancing the comprehension of social media discourse on two NLP tasks.
Moreover, this paper contributes the first multilingual CAT-G test set on
English, Chinese, Malay, and Bahasa Indonesian. As LLM capabilities vary among
languages, this test set allows for a comparative analysis of performance
across languages with varying levels of LLM proficiency.

### 10. [EL4NER: Ensemble Learning for Named Entity Recognition via Multiple Small-Parameter Large Language Models](http://arxiv.org/pdf/2505.23038v1)

Authors: Yuzhen Xiao, Jiahe Song, Yongxin Xu, Ruizhe Zhang, Yiqi Xiao, Xin Lu, Runchuan Zhu, Bowen Jiang, Junfeng Zhao

In-Context Learning (ICL) technique based on Large Language Models (LLMs) has
gained prominence in Named Entity Recognition (NER) tasks for its lower
computing resource consumption, less manual labeling overhead, and stronger
generalizability. Nevertheless, most ICL-based NER methods depend on
large-parameter LLMs: the open-source models demand substantial computational
resources for deployment and inference, while the closed-source ones incur high
API costs, raise data-privacy concerns, and hinder community collaboration. To
address this question, we propose an Ensemble Learning Method for Named Entity
Recognition (EL4NER), which aims at aggregating the ICL outputs of multiple
open-source, small-parameter LLMs to enhance overall performance in NER tasks
at less deployment and inference cost. Specifically, our method comprises three
key components. First, we design a task decomposition-based pipeline that
facilitates deep, multi-stage ensemble learning. Second, we introduce a novel
span-level sentence similarity algorithm to establish an ICL demonstration
retrieval mechanism better suited for NER tasks. Third, we incorporate a
self-validation mechanism to mitigate the noise introduced during the ensemble
process. We evaluated EL4NER on multiple widely adopted NER datasets from
diverse domains. Our experimental results indicate that EL4NER surpasses most
closed-source, large-parameter LLM-based methods at a lower parameter cost and
even attains state-of-the-art (SOTA) performance among ICL-based methods on
certain datasets. These results show the parameter efficiency of EL4NER and
underscore the feasibility of employing open-source, small-parameter LLMs
within the ICL paradigm for NER tasks.

### Cryptography and Security

### 1. [Chainless Apps: A Modular Framework for Building Apps with Web2 Capability and Web3 Trust](http://arxiv.org/pdf/2505.22989v1)

Authors: Brian Seong, Paul Gebheim

Modern blockchain applications are often constrained by a trade-off between
user experience and trust. Chainless Apps present a new paradigm of application
architecture that separates execution, trust, bridging, and settlement into
distinct compostable layers. This enables app-specific sequencing, verifiable
off-chain computation, chain-agnostic asset and message routing via Agglayer,
and finality on Ethereum - resulting in fast Web2-like UX with Web3-grade
verifiability. Although consensus mechanisms have historically underpinned
verifiable computation, the advent of zkVMs and decentralized validation
services opens up new trust models for developers. Chainless Apps leverage this
evolution to offer modular, scalable applications that maintain
interoperability with the broader blockchain ecosystem while allowing
domain-specific trade-offs.

### 2. [Joint Data Hiding and Partial Encryption of Compressive Sensed Streams](http://arxiv.org/pdf/2505.23357v1)

Authors: Cristina-Elena Popa, Cristian Damian, Daniela Coltuc

The paper proposes a method to secure the Compressive Sensing (CS) streams.
It consists in protecting part of the measurements by a secret key and
inserting the code into the rest. The secret key is generated via a
cryptographically secure pseudo-random number generator (CSPRNG) and XORed with
the measurements to be inserted. For insertion, we use a reversible data hiding
(RDH) scheme, which is a prediction error expansion algorithm, modified to
match the statistics of CS measurements. The reconstruction from the embedded
stream conducts to visibly distorted images. The image distortion is controlled
by the number of embedded levels. In our tests, the embedding on 10 levels
results in $\approx 18 dB $ distortion for images of 256x256 pixels
reconstructed with the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).
A particularity of the presented method is on-the-fly insertion that makes it
appropriate for the sequential acquisition of measurements by a Single Pixel
Camera. On-the-fly insertion avoids the buffering of CS measurements for a
subsequent standard encryption and generation of a thumbnail image.

### 3. [Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models](http://arxiv.org/pdf/2505.23561v1)

Authors: Zenghui Yuan, Yangming Xu, Jiawen Shi, Pan Zhou, Lichao Sun

Model merging for Large Language Models (LLMs) directly fuses the parameters
of different models finetuned on various tasks, creating a unified model for
multi-domain tasks. However, due to potential vulnerabilities in models
available on open-source platforms, model merging is susceptible to backdoor
attacks. In this paper, we propose Merge Hijacking, the first backdoor attack
targeting model merging in LLMs. The attacker constructs a malicious upload
model and releases it. Once a victim user merges it with any other models, the
resulting merged model inherits the backdoor while maintaining utility across
tasks. Merge Hijacking defines two main objectives-effectiveness and
utility-and achieves them through four steps. Extensive experiments demonstrate
the effectiveness of our attack across different models, merging algorithms,
and tasks. Additionally, we show that the attack remains effective even when
merging real-world models. Moreover, our attack demonstrates robustness against
two inference-time defenses (Paraphrasing and CLEANGEN) and one training-time
defense (Fine-pruning).

### 4. [A Unified Framework for Human AI Collaboration in Security Operations Centers with Trusted Autonomy](http://arxiv.org/pdf/2505.23397v1)

Authors: Ahmad Mohsin, Helge Janicke, Ahmed Ibrahim, Iqbal H. Sarker, Seyit Camtepe

This article presents a structured framework for Human-AI collaboration in
Security Operations Centers (SOCs), integrating AI autonomy, trust calibration,
and Human-in-the-loop decision making. Existing frameworks in SOCs often focus
narrowly on automation, lacking systematic structures to manage human
oversight, trust calibration, and scalable autonomy with AI. Many assume static
or binary autonomy settings, failing to account for the varied complexity,
criticality, and risk across SOC tasks considering Humans and AI collaboration.
To address these limitations, we propose a novel autonomy tiered framework
grounded in five levels of AI autonomy from manual to fully autonomous, mapped
to Human-in-the-Loop (HITL) roles and task-specific trust thresholds. This
enables adaptive and explainable AI integration across core SOC functions,
including monitoring, protection, threat detection, alert triage, and incident
response. The proposed framework differentiates itself from previous research
by creating formal connections between autonomy, trust, and HITL across various
SOC levels, which allows for adaptive task distribution according to
operational complexity and associated risks. The framework is exemplified
through a simulated cyber range that features the cybersecurity AI-Avatar, a
fine-tuned LLM-based SOC assistant. The AI-Avatar case study illustrates
human-AI collaboration for SOC tasks, reducing alert fatigue, enhancing
response coordination, and strategically calibrating trust. This research
systematically presents both the theoretical and practical aspects and
feasibility of designing next-generation cognitive SOCs that leverage AI not to
replace but to enhance human decision-making.

### 5. [MCP Safety Training: Learning to Refuse Falsely Benign MCP Exploits using Improved Preference Alignment](http://arxiv.org/pdf/2505.23634v1)

Authors: John Halloran

The model context protocol (MCP) has been widely adapted as an open standard
enabling the seamless integration of generative AI agents. However, recent work
has shown the MCP is susceptible to retrieval-based "falsely benign" attacks
(FBAs), allowing malicious system access and credential theft, but requiring
that users download compromised files directly to their systems. Herein, we
show that the threat model of MCP-based attacks is significantly broader than
previously thought, i.e., attackers need only post malicious content online to
deceive MCP agents into carrying out their attacks on unsuspecting victims'
systems.
  To improve alignment guardrails against such attacks, we introduce a new MCP
dataset of FBAs and (truly) benign samples to explore the effectiveness of
direct preference optimization (DPO) for the refusal training of large language
models (LLMs). While DPO improves model guardrails against such attacks, we
show that the efficacy of refusal learning varies drastically depending on the
model's original post-training alignment scheme--e.g., GRPO-based LLMs learn to
refuse extremely poorly. Thus, to further improve FBA refusals, we introduce
Retrieval Augmented Generation for Preference alignment (RAG-Pref), a novel
preference alignment strategy based on RAG. We show that RAG-Pref significantly
improves the ability of LLMs to refuse FBAs, particularly when combined with
DPO alignment, thus drastically improving guardrails against MCP-based attacks.

### 6. [Securing AI Agents with Information-Flow Control](http://arxiv.org/pdf/2505.23643v1)

Authors: Manuel Costa, Boris Köpf, Aashish Kolluri, Andrew Paverd, Mark Russinovich, Ahmed Salem, Shruti Tople, Lukas Wutschitz, Santiago Zanella-Béguelin

As AI agents become increasingly autonomous and capable, ensuring their
security against vulnerabilities such as prompt injection becomes critical.
This paper explores the use of information-flow control (IFC) to provide
security guarantees for AI agents. We present a formal model to reason about
the security and expressiveness of agent planners. Using this model, we
characterize the class of properties enforceable by dynamic taint-tracking and
construct a taxonomy of tasks to evaluate security and utility trade-offs of
planner designs. Informed by this exploration, we present Fides, a planner that
tracks confidentiality and integrity labels, deterministically enforces
security policies, and introduces novel primitives for selectively hiding
information. Its evaluation in AgentDojo demonstrates that this approach
broadens the range of tasks that can be securely accomplished. A tutorial to
walk readers through the the concepts introduced in the paper can be found at
https://github.com/microsoft/fides

### 7. [Keyed Chaotic Tensor Transformations for Secure And Attributable Neural Inference](http://arxiv.org/pdf/2505.23655v1)

Authors: Peter David Fagan

This work introduces a novel framework for secure and privacy-preserving
neural network inference based on keyed chaotic dynamical transformations. The
proposed method applies a deterministic, cryptographically seeded chaotic
system to tensors, producing non-invertible, user-specific transformations that
enable authenticated inference, tensor-level watermarking, and data
attribution. This framework offers a scalable and lightweight alternative to
conventional cryptographic techniques, and establishes a new direction for
tensor-level security in AI systems.

### 8. [Bayesian Perspective on Memorization and Reconstruction](http://arxiv.org/pdf/2505.23658v1)

Authors: Haim Kaplan, Yishay Mansour, Kobbi Nissim, Uri Stemmer

We introduce a new Bayesian perspective on the concept of data
reconstruction, and leverage this viewpoint to propose a new security
definition that, in certain settings, provably prevents reconstruction attacks.
We use our paradigm to shed new light on one of the most notorious attacks in
the privacy and memorization literature - fingerprinting code attacks (FPC). We
argue that these attacks are really a form of membership inference attacks,
rather than reconstruction attacks. Furthermore, we show that if the goal is
solely to prevent reconstruction (but not membership inference), then in some
cases the impossibility results derived from FPC no longer apply.

### 9. [Differentially Private Space-Efficient Algorithms for Counting Distinct Elements in the Turnstile Model](http://arxiv.org/pdf/2505.23682v1)

Authors: Rachel Cummings, Alessandro Epasto, Jieming Mao, Tamalika Mukherjee, Tingting Ou, Peilin Zhong

The turnstile continual release model of differential privacy captures
scenarios where a privacy-preserving real-time analysis is sought for a dataset
evolving through additions and deletions. In typical applications of real-time
data analysis, both the length of the stream $T$ and the size of the universe
$|U|$ from which data come can be extremely large. This motivates the study of
private algorithms in the turnstile setting using space sublinear in both $T$
and $|U|$. In this paper, we give the first sublinear space differentially
private algorithms for the fundamental problem of counting distinct elements in
the turnstile streaming model. Our algorithm achieves, on arbitrary streams,
$\tilde{O}_{\eta}(T^{1/3})$ space and additive error, and a $(1+\eta)$-relative
approximation for all $\eta \in (0,1)$. Our result significantly improves upon
the space requirements of the state-of-the-art algorithms for this problem,
which is linear, approaching the known $\Omega(T^{1/4})$ additive error lower
bound for arbitrary streams. Moreover, when a bound $W$ on the number of times
an item appears in the stream is known, our algorithm provides
$\tilde{O}_{\eta}(\sqrt{W})$ additive error, using $\tilde{O}_{\eta}(\sqrt{W})$
space. This additive error asymptotically matches that of prior work which
required instead linear space. Our results address an open question posed by
[Jain, Kalemaj, Raskhodnikova, Sivakumar, Smith, Neurips23] about designing
low-memory mechanisms for this problem. We complement these results with a
space lower bound for this problem, which shows that any algorithm that uses
similar techniques must use space $\tilde{\Omega}(T^{1/3})$ on arbitrary
streams.

### 10. [AgentAlign: Navigating Safety Alignment in the Shift from Informative to Agentic Large Language Models](http://arxiv.org/pdf/2505.23020v1)

Authors: Jinchuan Zhang, Lu Yin, Yan Zhou, Songlin Hu

The acquisition of agentic capabilities has transformed LLMs from "knowledge
providers" to "action executors", a trend that while expanding LLMs' capability
boundaries, significantly increases their susceptibility to malicious use.
Previous work has shown that current LLM-based agents execute numerous
malicious tasks even without being attacked, indicating a deficiency in agentic
use safety alignment during the post-training phase. To address this gap, we
propose AgentAlign, a novel framework that leverages abstract behavior chains
as a medium for safety alignment data synthesis. By instantiating these
behavior chains in simulated environments with diverse tool instances, our
framework enables the generation of highly authentic and executable
instructions while capturing complex multi-step dynamics. The framework further
ensures model utility by proportionally synthesizing benign instructions
through non-malicious interpretations of behavior chains, precisely calibrating
the boundary between helpfulness and harmlessness. Evaluation results on
AgentHarm demonstrate that fine-tuning three families of open-source models
using our method substantially improves their safety (35.8% to 79.5%
improvement) while minimally impacting or even positively enhancing their
helpfulness, outperforming various prompting methods. The dataset and code have
both been open-sourced.

### Computer Vision and Pattern Recognition

### 1. [iHDR: Iterative HDR Imaging with Arbitrary Number of Exposures](http://arxiv.org/pdf/2505.22971v1)

Authors: Yu Yuan, Yiheng Chi, Xingguang Zhang, Stanley Chan

High dynamic range (HDR) imaging aims to obtain a high-quality HDR image by
fusing information from multiple low dynamic range (LDR) images. Numerous
learning-based HDR imaging methods have been proposed to achieve this for
static and dynamic scenes. However, their architectures are mostly tailored for
a fixed number (e.g., three) of inputs and, therefore, cannot apply directly to
situations beyond the pre-defined limited scope. To address this issue, we
propose a novel framework, iHDR, for iterative fusion, which comprises a
ghost-free Dual-input HDR fusion network (DiHDR) and a physics-based domain
mapping network (ToneNet). DiHDR leverages a pair of inputs to estimate an
intermediate HDR image, while ToneNet maps it back to the nonlinear domain and
serves as the reference input for the next pairwise fusion. This process is
iteratively executed until all input frames are utilized. Qualitative and
quantitative experiments demonstrate the effectiveness of the proposed method
as compared to existing state-of-the-art HDR deghosting approaches given
flexible numbers of input frames.

### 2. [HyperMotion: DiT-Based Pose-Guided Human Image Animation of Complex Motions](http://arxiv.org/pdf/2505.22977v1)

Authors: Shuolin Xu, Siming Zheng, Ziyi Wang, HC Yu, Jinwei Chen, Huaqi Zhang, Bo Li, Peng-Tao Jiang

Recent advances in diffusion models have significantly improved conditional
video generation, particularly in the pose-guided human image animation task.
Although existing methods are capable of generating high-fidelity and
time-consistent animation sequences in regular motions and static scenes, there
are still obvious limitations when facing complex human body motions
(Hypermotion) that contain highly dynamic, non-standard motions, and the lack
of a high-quality benchmark for evaluation of complex human motion animations.
To address this challenge, we introduce the \textbf{Open-HyperMotionX Dataset}
and \textbf{HyperMotionX Bench}, which provide high-quality human pose
annotations and curated video clips for evaluating and improving pose-guided
human image animation models under complex human motion conditions.
Furthermore, we propose a simple yet powerful DiT-based video generation
baseline and design spatial low-frequency enhanced RoPE, a novel module that
selectively enhances low-frequency spatial feature modeling by introducing
learnable frequency scaling. Our method significantly improves structural
stability and appearance consistency in highly dynamic human motion sequences.
Extensive experiments demonstrate the effectiveness of our dataset and proposed
approach in advancing the generation quality of complex human motion image
animations. Code and dataset will be made publicly available.

### 3. [Pose-free 3D Gaussian splatting via shape-ray estimation](http://arxiv.org/pdf/2505.22978v1)

Authors: Youngju Na, Taeyeon Kim, Jumin Lee, Kyu Beom Han, Woo Jae Kim, Sung-eui Yoon

While generalizable 3D Gaussian splatting enables efficient, high-quality
rendering of unseen scenes, it heavily depends on precise camera poses for
accurate geometry. In real-world scenarios, obtaining accurate poses is
challenging, leading to noisy pose estimates and geometric misalignments. To
address this, we introduce SHARE, a pose-free, feed-forward Gaussian splatting
framework that overcomes these ambiguities by joint shape and camera rays
estimation. Instead of relying on explicit 3D transformations, SHARE builds a
pose-aware canonical volume representation that seamlessly integrates
multi-view information, reducing misalignment caused by inaccurate pose
estimates. Additionally, anchor-aligned Gaussian prediction enhances scene
reconstruction by refining local geometry around coarse anchors, allowing for
more precise Gaussian placement. Extensive experiments on diverse real-world
datasets show that our method achieves robust performance in pose-free
generalizable Gaussian splatting.

### 4. [MOVi: Training-free Text-conditioned Multi-Object Video Generation](http://arxiv.org/pdf/2505.22980v1)

Authors: Aimon Rahman, Jiang Liu, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Yusheng Su, Vishal M. Patel, Zicheng Liu, Emad Barsoum

Recent advances in diffusion-based text-to-video (T2V) models have
demonstrated remarkable progress, but these models still face challenges in
generating videos with multiple objects. Most models struggle with accurately
capturing complex object interactions, often treating some objects as static
background elements and limiting their movement. In addition, they often fail
to generate multiple distinct objects as specified in the prompt, resulting in
incorrect generations or mixed features across objects. In this paper, we
present a novel training-free approach for multi-object video generation that
leverages the open world knowledge of diffusion models and large language
models (LLMs). We use an LLM as the ``director'' of object trajectories, and
apply the trajectories through noise re-initialization to achieve precise
control of realistic movements. We further refine the generation process by
manipulating the attention mechanism to better capture object-specific features
and motion patterns, and prevent cross-object feature interference. Extensive
experiments validate the effectiveness of our training free approach in
significantly enhancing the multi-object generation capabilities of existing
video diffusion models, resulting in 42% absolute improvement in motion
dynamics and object generation accuracy, while also maintaining high fidelity
and motion smoothness.

### 5. [SeG-SR: Integrating Semantic Knowledge into Remote Sensing Image Super-Resolution via Vision-Language Model](http://arxiv.org/pdf/2505.23010v1)

Authors: Bowen Chen, Keyan Chen, Mohan Yang, Zhengxia Zou, Zhenwei Shi

High-resolution (HR) remote sensing imagery plays a vital role in a wide
range of applications, including urban planning and environmental monitoring.
However, due to limitations in sensors and data transmission links, the images
acquired in practice often suffer from resolution degradation. Remote Sensing
Image Super-Resolution (RSISR) aims to reconstruct HR images from
low-resolution (LR) inputs, providing a cost-effective and efficient
alternative to direct HR image acquisition. Existing RSISR methods primarily
focus on low-level characteristics in pixel space, while neglecting the
high-level understanding of remote sensing scenes. This may lead to
semantically inconsistent artifacts in the reconstructed results. Motivated by
this observation, our work aims to explore the role of high-level semantic
knowledge in improving RSISR performance. We propose a Semantic-Guided
Super-Resolution framework, SeG-SR, which leverages Vision-Language Models
(VLMs) to extract semantic knowledge from input images and uses it to guide the
super resolution (SR) process. Specifically, we first design a Semantic Feature
Extraction Module (SFEM) that utilizes a pretrained VLM to extract semantic
knowledge from remote sensing images. Next, we propose a Semantic Localization
Module (SLM), which derives a series of semantic guidance from the extracted
semantic knowledge. Finally, we develop a Learnable Modulation Module (LMM)
that uses semantic guidance to modulate the features extracted by the SR
network, effectively incorporating high-level scene understanding into the SR
pipeline. We validate the effectiveness and generalizability of SeG-SR through
extensive experiments: SeG-SR achieves state-of-the-art performance on two
datasets and consistently delivers performance improvements across various SR
architectures. Codes can be found at https://github.com/Mr-Bamboo/SeG-SR.

### 6. [Spatio-Temporal Joint Density Driven Learning for Skeleton-Based Action Recognition](http://arxiv.org/pdf/2505.23012v1)

Authors: Shanaka Ramesh Gunasekara, Wanqing Li, Philip Ogunbona, Jack Yang

Traditional approaches in unsupervised or self supervised learning for
skeleton-based action classification have concentrated predominantly on the
dynamic aspects of skeletal sequences. Yet, the intricate interaction between
the moving and static elements of the skeleton presents a rarely tapped
discriminative potential for action classification. This paper introduces a
novel measurement, referred to as spatial-temporal joint density (STJD), to
quantify such interaction. Tracking the evolution of this density throughout an
action can effectively identify a subset of discriminative moving and/or static
joints termed "prime joints" to steer self-supervised learning. A new
contrastive learning strategy named STJD-CL is proposed to align the
representation of a skeleton sequence with that of its prime joints while
simultaneously contrasting the representations of prime and nonprime joints. In
addition, a method called STJD-MP is developed by integrating it with a
reconstruction-based framework for more effective learning. Experimental
evaluations on the NTU RGB+D 60, NTU RGB+D 120, and PKUMMD datasets in various
downstream tasks demonstrate that the proposed STJD-CL and STJD-MP improved
performance, particularly by 3.5 and 3.6 percentage points over the
state-of-the-art contrastive methods on the NTU RGB+D 120 dataset using X-sub
and X-set evaluations, respectively.

### 7. [Towards Privacy-Preserving Fine-Grained Visual Classification via Hierarchical Learning from Label Proportions](http://arxiv.org/pdf/2505.23031v1)

Authors: Jinyi Chang, Dongliang Chang, Lei Chen, Bingyao Yu, Zhanyu Ma

In recent years, Fine-Grained Visual Classification (FGVC) has achieved
impressive recognition accuracy, despite minimal inter-class variations.
However, existing methods heavily rely on instance-level labels, making them
impractical in privacy-sensitive scenarios such as medical image analysis. This
paper aims to enable accurate fine-grained recognition without direct access to
instance labels. To achieve this, we leverage the Learning from Label
Proportions (LLP) paradigm, which requires only bag-level labels for efficient
training. Unlike existing LLP-based methods, our framework explicitly exploits
the hierarchical nature of fine-grained datasets, enabling progressive feature
granularity refinement and improving classification accuracy. We propose
Learning from Hierarchical Fine-Grained Label Proportions (LHFGLP), a framework
that incorporates Unrolled Hierarchical Fine-Grained Sparse Dictionary
Learning, transforming handcrafted iterative approximation into learnable
network optimization. Additionally, our proposed Hierarchical Proportion Loss
provides hierarchical supervision, further enhancing classification
performance. Experiments on three widely-used fine-grained datasets, structured
in a bag-based manner, demonstrate that our framework consistently outperforms
existing LLP-based methods. We will release our code and datasets to foster
further research in privacy-preserving fine-grained classification.

### 8. [Deep Modeling and Optimization of Medical Image Classification](http://arxiv.org/pdf/2505.23040v1)

Authors: Yihang Wu, Muhammad Owais, Reem Kateb, Ahmad Chaddad

Deep models, such as convolutional neural networks (CNNs) and vision
transformer (ViT), demonstrate remarkable performance in image classification.
However, those deep models require large data to fine-tune, which is
impractical in the medical domain due to the data privacy issue. Furthermore,
despite the feasible performance of contrastive language image pre-training
(CLIP) in the natural domain, the potential of CLIP has not been fully
investigated in the medical field. To face these challenges, we considered
three scenarios: 1) we introduce a novel CLIP variant using four CNNs and eight
ViTs as image encoders for the classification of brain cancer and skin cancer,
2) we combine 12 deep models with two federated learning techniques to protect
data privacy, and 3) we involve traditional machine learning (ML) methods to
improve the generalization ability of those deep models in unseen domain data.
The experimental results indicate that maxvit shows the highest averaged (AVG)
test metrics (AVG = 87.03\%) in HAM10000 dataset with multimodal learning,
while convnext\_l demonstrates remarkable test with an F1-score of 83.98\%
compared to swin\_b with 81.33\% in FL model. Furthermore, the use of support
vector machine (SVM) can improve the overall test metrics with AVG of $\sim
2\%$ for swin transformer series in ISIC2018. Our codes are available at
https://github.com/AIPMLab/SkinCancerSimulation.

### 9. [SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](http://arxiv.org/pdf/2505.23044v1)

Authors: Yu Sheng, Jiajun Deng, Xinran Zhang, Yu Zhang, Bei Hua, Yanyong Zhang, Jianmin Ji

A major breakthrough in 3D reconstruction is the feedforward paradigm to
generate pixel-wise 3D points or Gaussian primitives from sparse, unposed
images. To further incorporate semantics while avoiding the significant memory
and storage costs of high-dimensional semantic features, existing methods
extend this paradigm by associating each primitive with a compressed semantic
feature vector. However, these methods have two major limitations: (a) the
naively compressed feature compromises expressiveness, affecting the model's
ability to capture fine-grained semantics, and (b) the pixel-wise primitive
prediction introduces redundancy in overlapping areas, causing unnecessary
memory overhead. To this end, we introduce \textbf{SpatialSplat}, a feedforward
framework that produces redundancy-aware Gaussians and capitalizes on a
dual-field semantic representation. Particularly, with the insight that
primitives within the same instance exhibit high semantic consistency, we
decompose the semantic representation into a coarse feature field that encodes
uncompressed semantics with minimal primitives, and a fine-grained yet
low-dimensional feature field that captures detailed inter-instance
relationships. Moreover, we propose a selective Gaussian mechanism, which
retains only essential Gaussians in the scene, effectively eliminating
redundant primitives. Our proposed Spatialsplat learns accurate semantic
information and detailed instances prior with more compact 3D Gaussians, making
semantic 3D reconstruction more applicable. We conduct extensive experiments to
evaluate our method, demonstrating a remarkable 60\% reduction in scene
representation parameters while achieving superior performance over
state-of-the-art methods. The code will be made available for future
investigation.

### 10. [Zero-P-to-3: Zero-Shot Partial-View Images to 3D Object](http://arxiv.org/pdf/2505.23054v1)

Authors: Yuxuan Lin, Ruihang Chu, Zhenyu Chen, Xiao Tang, Lei Ke, Haoling Li, Yingji Zhong, Zhihao Li, Shiyong Liu, Xiaofei Wu, Jianzhuang Liu, Yujiu Yang

Generative 3D reconstruction shows strong potential in incomplete
observations. While sparse-view and single-image reconstruction are
well-researched, partial observation remains underexplored. In this context,
dense views are accessible only from a specific angular range, with other
perspectives remaining inaccessible. This task presents two main challenges:
(i) limited View Range: observations confined to a narrow angular scope prevent
effective traditional interpolation techniques that require evenly distributed
perspectives. (ii) inconsistent Generation: views created for invisible regions
often lack coherence with both visible regions and each other, compromising
reconstruction consistency. To address these challenges, we propose \method, a
novel training-free approach that integrates the local dense observations and
multi-source priors for reconstruction. Our method introduces a fusion-based
strategy to effectively align these priors in DDIM sampling, thereby generating
multi-view consistent images to supervise invisible views. We further design an
iterative refinement strategy, which uses the geometric structures of the
object to enhance reconstruction quality. Extensive experiments on multiple
datasets show the superiority of our method over SOTAs, especially in invisible
regions.

### Computers and Society

### 1. [REDDIX-NET: A Novel Dataset and Benchmark for Moderating Online Explicit Services](http://arxiv.org/pdf/2505.23231v1)

Authors: MSVPJ Sathvik, Manan Roy Choudhury, Rishita Agarwal, Sathwik Narkedimilli, Vivek Gupta

The rise of online platforms has enabled covert illicit activities, including
online prostitution, to pose challenges for detection and regulation. In this
study, we introduce REDDIX-NET, a novel benchmark dataset specifically designed
for moderating online sexual services and going beyond traditional NSFW
filters. The dataset is derived from thousands of web-scraped NSFW posts on
Reddit and categorizes users into six behavioral classes reflecting different
service offerings and user intentions. We evaluate the classification
performance of state-of-the-art large language models (GPT-4, LlaMA
3.3-70B-Instruct, Gemini 1.5 Flash, Mistral 8x7B, Qwen 2.5 Turbo, Claude 3.5
Haiku) using advanced quantitative metrics, finding promising results with
models like GPT-4 and Gemini 1.5 Flash. Beyond classification, we conduct
sentiment and comment analysis, leveraging LLM and PLM-based approaches and
metadata extraction to uncover behavioral and temporal patterns. These analyses
reveal peak engagement times and distinct user interaction styles across
categories. Our findings provide critical insights into AI-driven moderation
and enforcement, offering a scalable framework for platforms to combat online
prostitution and associated harms.

### 2. [Can Large Language Models Trigger a Paradigm Shift in Travel Behavior Modeling? Experiences with Modeling Travel Satisfaction](http://arxiv.org/pdf/2505.23262v1)

Authors: Pengfei Xu, Donggen Wang

As a specific domain of subjective well-being, travel satisfaction has
attracted much research attention recently. Previous studies primarily use
statistical models and, more recently, machine learning models to explore the
determinants of travel satisfaction. Both approaches require data from
sufficient sample sizes and correct prior statistical assumptions. The
emergence of Large Language Models (LLMs) offers a new modeling approach that
can overcome the shortcomings of the existing methods. Pre-trained on extensive
datasets, LLMs have strong capabilities in contextual understanding and
generalization, significantly reducing their dependence on large quantities of
task-specific data and stringent statistical assumptions. The primary challenge
in applying LLMs lies in addressing the behavioral misalignment between LLMs
and human behavior. Using data on travel satisfaction from a household survey
in shanghai, this study identifies the existence and source of misalignment and
develop methods to address the misalignment issue. We find that the zero-shot
LLM exhibits behavioral misalignment, resulting in relatively low prediction
accuracy. However, few-shot learning, even with a limited number of samples,
allows the model to outperform baseline models in MSE and MAPE metrics. This
misalignment can be attributed to the gap between the general knowledge
embedded in LLMs and the specific, unique characteristics of the dataset. On
these bases, we propose an LLM-based modeling approach that can be applied to
model travel behavior using samples of small sizes. This study highlights the
potential of LLMs for modeling not only travel satisfaction but also broader
aspects of travel behavior.

### 3. [A Practical Guide for Supporting Formative Assessment and Feedback Using Generative AI](http://arxiv.org/pdf/2505.23405v1)

Authors: Sapolnach Prompiengchai, Charith Narreddy, Steve Joordens

Formative assessment is a cornerstone of effective teaching and learning,
providing students with feedback to guide their learning. While there has been
an exponential growth in the application of generative AI in scaling various
aspects of formative assessment, ranging from automatic question generation to
intelligent tutoring systems and personalized feedback, few have directly
addressed the core pedagogical principles of formative assessment. Here, we
critically examined how generative AI, especially large-language models (LLMs)
such as ChatGPT, can support key components of formative assessment: helping
students, teachers, and peers understand "where learners are going," "where
learners currently are," and "how to move learners forward" in the learning
process. With the rapid emergence of new prompting techniques and LLM
capabilities, we also provide guiding principles for educators to effectively
leverage cost-free LLMs in formative assessments while remaining grounded in
pedagogical best practices. Furthermore, we reviewed the role of LLMs in
generating feedback, highlighting limitations in current evaluation metrics
that inadequately capture the nuances of formative feedback, such as
distinguishing feedback at the task, process, and self-regulatory levels.
Finally, we offer practical guidelines for educators and researchers, including
concrete classroom strategies and future directions such as developing robust
metrics to assess LLM-generated feedback, leveraging LLMs to overcome systemic
and cultural barriers to formative assessment, and designing AI-aware
assessment strategies that promote transferable skills while mitigating
overreliance on LLM-generated responses. By structuring the discussion within
an established formative assessment framework, this review provides a
comprehensive foundation for integrating LLMs into formative assessment in a
pedagogically informed manner.

### 4. [A Computational Approach to Improving Fairness in K-means Clustering](http://arxiv.org/pdf/2505.22984v1)

Authors: Guancheng Zhou, Haiping Xu, Hongkang Xu, Chenyu Li, Donghui Yan

The popular K-means clustering algorithm potentially suffers from a major
weakness for further analysis or interpretation. Some cluster may have
disproportionately more (or fewer) points from one of the subpopulations in
terms of some sensitive variable, e.g., gender or race. Such a fairness issue
may cause bias and unexpected social consequences. This work attempts to
improve the fairness of K-means clustering with a two-stage optimization
formulation--clustering first and then adjust cluster membership of a small
subset of selected data points. Two computationally efficient algorithms are
proposed in identifying those data points that are expensive for fairness, with
one focusing on nearest data points outside of a cluster and the other on
highly 'mixed' data points. Experiments on benchmark datasets show substantial
improvement on fairness with a minimal impact to clustering quality. The
proposed algorithms can be easily extended to a broad class of clustering
algorithms or fairness metrics.

### 5. [Designing the Future of Entrepreneurship Education: Exploring an AI-Empowered Scaffold System for Business Plan Development](http://arxiv.org/pdf/2505.23326v1)

Authors: Junhua Zhu, Lan Luo

Entrepreneurship education equips students to transform innovative ideas into
actionable entrepreneurship plans, yet traditional approaches often struggle to
provide the personalized guidance and practical alignment needed for success.
Focusing on the business plan as a key learning tool and evaluation method,
this study investigates the design needs for an AI-empowered scaffold system to
address these challenges. Based on qualitative insights from educators and
students, the findings highlight three critical dimensions for system design:
mastery of business plan development, alignment with entrepreneurial learning
goals, and integration of adaptive system features. These findings underscore
the transformative potential of AI in bridging gaps in entrepreneurship
education while emphasizing the enduring value of human mentorship and
experiential learning.

### 6. [MCTSr-Zero: Self-Reflective Psychological Counseling Dialogues Generation via Principles and Adaptive Exploration](http://arxiv.org/pdf/2505.23229v1)

Authors: Hao Lu, Yanchi Gu, Haoyuan Huang, Yulin Zhou, Ningxin Zhu, Chen Li

The integration of Monte Carlo Tree Search (MCTS) with Large Language Models
(LLMs) has demonstrated significant success in structured, problem-oriented
tasks. However, applying these methods to open-ended dialogues, such as those
in psychological counseling, presents unique challenges. Unlike tasks with
objective correctness, success in therapeutic conversations depends on
subjective factors like empathetic engagement, ethical adherence, and alignment
with human preferences, for which strict "correctness" criteria are
ill-defined. Existing result-oriented MCTS approaches can therefore produce
misaligned responses. To address this, we introduce MCTSr-Zero, an MCTS
framework designed for open-ended, human-centric dialogues. Its core innovation
is "domain alignment", which shifts the MCTS search objective from predefined
end-states towards conversational trajectories that conform to target domain
principles (e.g., empathy in counseling). Furthermore, MCTSr-Zero incorporates
"Regeneration" and "Meta-Prompt Adaptation" mechanisms to substantially broaden
exploration by allowing the MCTS to consider fundamentally different initial
dialogue strategies. We evaluate MCTSr-Zero in psychological counseling by
generating multi-turn dialogue data, which is used to fine-tune an LLM, PsyLLM.
We also introduce PsyEval, a benchmark for assessing multi-turn psychological
counseling dialogues. Experiments demonstrate that PsyLLM achieves
state-of-the-art performance on PsyEval and other relevant metrics, validating
MCTSr-Zero's effectiveness in generating high-quality, principle-aligned
conversational data for human-centric domains and addressing the LLM challenge
of consistently adhering to complex psychological standards.

### 7. [A Mathematical Framework for AI-Human Integration in Work](http://arxiv.org/pdf/2505.23432v1)

Authors: Elisa Celis, Lingxiao Huang, Nisheeth K. Vishnoi

The rapid rise of Generative AI (GenAI) tools has sparked debate over their
role in complementing or replacing human workers across job contexts. We
present a mathematical framework that models jobs, workers, and worker-job fit,
introducing a novel decomposition of skills into decision-level and
action-level subskills to reflect the complementary strengths of humans and
GenAI. We analyze how changes in subskill abilities affect job success,
identifying conditions for sharp transitions in success probability. We also
establish sufficient conditions under which combining workers with
complementary subskills significantly outperforms relying on a single worker.
This explains phenomena such as productivity compression, where GenAI
assistance yields larger gains for lower-skilled workers. We demonstrate the
framework' s practicality using data from O*NET and Big-Bench Lite, aligning
real-world data with our model via subskill-division methods. Our results
highlight when and how GenAI complements human skills, rather than replacing
them.

### 8. [The CASE Framework -- A New Architecture for Participatory Research and Digital Health Surveillance](http://arxiv.org/pdf/2505.23516v1)

Authors: Marco Hirsch, Peter Hevesi, Paul Lukowicz

We present the CASE framework, an open-source platform for adaptive,
context-aware participatory research, and pandemic preparedness. CASE
implements an event-driven architecture that enables dynamic survey workflows,
allowing real-time adaptation based on participant responses, external data,
temporal conditions, and evolving user states. The framework supports a broad
range of research needs, from simple one-time questionnaires to complex
longitudinal studies with advanced conditional logic. Built on over a decade of
practical experience, CASE underwent a major architectural rework in 2024,
transitioning from a microservice-based design to a streamlined monolithic
architecture. This evolution significantly improved maintainability,
flexibility, and accessibility to deployment, particularly for institutions
with limited technical capacity. CASE has been successfully deployed across
diverse domains, powering national disease surveillance platforms, supporting
post-COVID cohort studies, and enabling real-time sentiment analysis during
political events. These applications, involving tens of thousands of
participants, demonstrate the framework's scalability, versatility, and
practical value. This paper describes the foundations of CASE, details its
architectural evolution, and presents lessons learned from real-world
deployments. We establish CASE as a mature and reusable research infrastructure
that balances sophisticated functionality with practical implementation,
addressing the critical global need for sustainable and institutionally
controlled data collection systems.

### 9. [Evaluating AI capabilities in detecting conspiracy theories on YouTube](http://arxiv.org/pdf/2505.23570v1)

Authors: Leonardo La Rocca, Francesco Corso, Francesco Pierri

As a leading online platform with a vast global audience, YouTube's extensive
reach also makes it susceptible to hosting harmful content, including
disinformation and conspiracy theories. This study explores the use of
open-weight Large Language Models (LLMs), both text-only and multimodal, for
identifying conspiracy theory videos shared on YouTube. Leveraging a labeled
dataset of thousands of videos, we evaluate a variety of LLMs in a zero-shot
setting and compare their performance to a fine-tuned RoBERTa baseline. Results
show that text-based LLMs achieve high recall but lower precision, leading to
increased false positives. Multimodal models lag behind their text-only
counterparts, indicating limited benefits from visual data integration. To
assess real-world applicability, we evaluate the most accurate models on an
unlabeled dataset, finding that RoBERTa achieves performance close to LLMs with
a larger number of parameters. Our work highlights the strengths and
limitations of current LLM-based approaches for online harmful content
detection, emphasizing the need for more precise and robust systems.

### 10. [Towards A Global Quantum Internet: A Review of Challenges Facing Aerial Quantum Networks](http://arxiv.org/pdf/2505.23603v1)

Authors: Nitin Jha, Abhishek Parakh

Quantum networks use principles of quantum physics to create secure
communication networks. Moving these networks off the ground using drones,
balloons, or satellites could help increase the scalability of these networks.
This article reviews how such aerial links work, what makes them difficult to
build, and the possible solutions that can be used to overcome these problems.
By combining ground stations, aerial relays, and orbiting satellites into one
seamless system, we move closer to a practical quantum internet.

### Databases

### 1. [LINEAGEX: A Column Lineage Extraction System for SQL](http://arxiv.org/pdf/2505.23133v1)

Authors: Shi Heng Zhang, Zhengjie Miao, Jiannan Wang

As enterprise data grows in size and complexity, column-level data lineage,
which records the creation, transformation, and reference of each column in the
warehouse, has been the key to effective data governance that assists tasks
like data quality monitoring, storage refactoring, and workflow migration.
Unfortunately, existing systems introduce overheads by integration with query
execution or fail to achieve satisfying accuracy for column lineage. In this
paper, we demonstrate LINEAGEX, a lightweight Python library that infers column
level lineage from SQL queries and visualizes it through an interactive
interface. LINEAGEX achieves high coverage and accuracy for column lineage
extraction by intelligently traversing query parse trees and handling
ambiguities. The demonstration walks through use cases of building lineage
graphs and troubleshooting data quality issues. LINEAGEX is open sourced at
https://github.com/sfu-db/lineagex and our video demonstration is at
https://youtu.be/5LaBBDDitlw

### 2. [TailorSQL: An NL2SQL System Tailored to Your Query Workload](http://arxiv.org/pdf/2505.23039v1)

Authors: Kapil Vaidya, Jialin Ding, Sebastian Kosak, David Kernert, Chuan Lei, Xiao Qin, Abhinav Tripathy, Ramesh Balan, Balakrishnan Narayanaswamy, Tim Kraska

NL2SQL (natural language to SQL) translates natural language questions into
SQL queries, thereby making structured data accessible to non-technical users,
serving as the foundation for intelligent data applications. State-of-the-art
NL2SQL techniques typically perform translation by retrieving database-specific
information, such as the database schema, and invoking a pre-trained large
language model (LLM) using the question and retrieved information to generate
the SQL query.
  However, existing NL2SQL techniques miss a key opportunity which is present
in real-world settings: NL2SQL is typically applied on existing databases which
have already served many SQL queries in the past. The past query workload
implicitly contains information which is helpful for accurate NL2SQL
translation and is not apparent from the database schema alone, such as common
join paths and the semantics of obscurely-named tables and columns. We
introduce TailorSQL, a NL2SQL system that takes advantage of information in the
past query workload to improve both the accuracy and latency of translating
natural language questions into SQL. By specializing to a given workload,
TailorSQL achieves up to 2$\times$ improvement in execution accuracy on
standardized benchmarks.

### 3. [KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](http://arxiv.org/pdf/2505.23416v1)

Authors: Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, Hyun Oh Song

Transformer-based large language models (LLMs) cache context as key-value
(KV) pairs during inference. As context length grows, KV cache sizes expand,
leading to substantial memory overhead and increased attention latency. This
paper introduces KVzip, a query-agnostic KV cache eviction method enabling
effective reuse of compressed KV caches across diverse queries. KVzip
quantifies the importance of a KV pair using the underlying LLM to reconstruct
original contexts from cached KV pairs, subsequently evicting pairs with lower
importance. Extensive empirical evaluations demonstrate that KVzip reduces KV
cache size by 3-4$\times$ and FlashAttention decoding latency by approximately
2$\times$, with negligible performance loss in question-answering, retrieval,
reasoning, and code comprehension tasks. Evaluations include various models
such as LLaMA3.1-8B, Qwen2.5-14B, and Gemma3-12B, with context lengths reaching
up to 170K tokens. KVzip significantly outperforms existing query-aware KV
eviction methods, which suffer from performance degradation even at a 90% cache
budget ratio under multi-query scenarios.

### 4. [Towards Explainable Sequential Learning](http://arxiv.org/pdf/2505.23624v1)

Authors: Giacomo Bergami, Emma Packer, Kirsty Scott, Silvia Del Din

This paper offers a hybrid explainable temporal data processing pipeline,
DataFul Explainable MultivariatE coRrelatIonal Temporal Artificial inTElligence
(EMeriTAte+DF), bridging numerical-driven temporal data classification with an
event-based one through verified artificial intelligence principles, enabling
human-explainable results. This was possible through a preliminary a posteriori
explainable phase describing the numerical input data in terms of concurrent
constituents with numerical payloads. This further required extending the
event-based literature to design specification mining algorithms supporting
concurrent constituents. Our previous and current solutions outperform
state-of-the-art solutions for multivariate time series classifications, thus
showcasing the effectiveness of the proposed methodology.

### 5. [Verify-in-the-Graph: Entity Disambiguation Enhancement for Complex Claim Verification with Interactive Graph Representation](http://arxiv.org/pdf/2505.22993v1)

Authors: Hoang Pham, Thanh-Do Nguyen, Khac-Hoai Nam Bui

Claim verification is a long-standing and challenging task that demands not
only high accuracy but also explainability of the verification process. This
task becomes an emerging research issue in the era of large language models
(LLMs) since real-world claims are often complex, featuring intricate semantic
structures or obfuscated entities. Traditional approaches typically address
this by decomposing claims into sub-claims and querying a knowledge base to
resolve hidden or ambiguous entities. However, the absence of effective
disambiguation strategies for these entities can compromise the entire
verification process. To address these challenges, we propose
Verify-in-the-Graph (VeGraph), a novel framework leveraging the reasoning and
comprehension abilities of LLM agents. VeGraph operates in three phases: (1)
Graph Representation - an input claim is decomposed into structured triplets,
forming a graph-based representation that integrates both structured and
unstructured information; (2) Entity Disambiguation -VeGraph iteratively
interacts with the knowledge base to resolve ambiguous entities within the
graph for deeper sub-claim verification; and (3) Verification - remaining
triplets are verified to complete the fact-checking process. Experiments using
Meta-Llama-3-70B (instruct version) show that VeGraph achieves competitive
performance compared to baselines on two benchmarks HoVer and FEVEROUS,
effectively addressing claim verification challenges. Our source code and data
are available for further exploitation.

### Distributed, Parallel, and Cluster Computing

### 1. [Speeding up Model Loading with fastsafetensors](http://arxiv.org/pdf/2505.23072v1)

Authors: Takeshi Yoshimura, Tatsuhiro Chiba, Manish Sethi, Daniel Waddington, Swaminathan Sundararaman

The rapid increases in model parameter sizes introduces new challenges in
pre-trained model loading. Currently, machine learning code often deserializes
each parameter as a tensor object in host memory before copying it to device
memory. We found that this approach underutilized storage throughput and
significantly slowed down loading large models with a widely-used model file
formats, safetensors. In this work, we present fastsafetensors, a Python
library designed to optimize the deserialization of tensors in safetensors
files. Our approach first copies groups of on-disk parameters to device memory,
where they are directly instantiated as tensor objects. This design enables
further optimization in low-level I/O and high-level tensor preprocessing,
including parallelized copying, peer-to-peer DMA, and GPU offloading.
Experimental results show performance improvements of 4.8x to 7.5x in loading
models such as Llama (7, 13, and 70 billion parameters), Falcon (40 billion
parameters), and the Bloom (176 billion parameters).

### 2. [Ghidorah: Fast LLM Inference on Edge with Speculative Decoding and Hetero-Core Parallelism](http://arxiv.org/pdf/2505.23219v1)

Authors: Jinhui Wei, Ye Huang, Yuhui Zhou, Jiazhi Jiang, Jiangsu Du

In-situ LLM inference on end-user devices has gained significant interest due
to its privacy benefits and reduced dependency on external infrastructure.
However, as the decoding process is memory-bandwidth-bound, the diverse
processing units in modern end-user devices cannot be fully exploited,
resulting in slow LLM inference. This paper presents Ghidorah, a LLM inference
system for end-user devices with the unified memory architecture. The key idea
of Ghidorah can be summarized in two steps: 1) leveraging speculative decoding
approaches to enhance parallelism, and 2) ingeniously distributing workloads
across multiple heterogeneous processing units to maximize computing power
utilization. Ghidorah includes the hetero-core model parallelism (HCMP)
architecture and the architecture-aware profiling (ARCA) approach. The HCMP
architecture guides partitioning by leveraging the unified memory design of
end-user devices and adapting to the hybrid computational demands of
speculative decoding. The ARCA approach is used to determine the optimal
speculative strategy and partitioning strategy, balancing acceptance rate with
parallel capability to maximize the speedup. Additionally, we optimize sparse
computation on ARM CPUs. Experimental results show that Ghidorah can achieve up
to 7.6x speedup in the dominant LLM decoding phase compared to the sequential
decoding approach in NVIDIA Jetson NX.

### 3. [MemAscend: System Memory Optimization for SSD-Offloaded LLM Fine-Tuning](http://arxiv.org/pdf/2505.23254v1)

Authors: Yong-Cheng Liaw, Shuo-Han Chen

Owing to the huge success of generative artificial intelligence (AI), large
language models (LLMs) have emerged as a core subclass, underpinning
applications such as question answering, text generation, and code completion.
While fine-tuning these models on domain-specific data can yield significant
performance gains, it also poses daunting computational challenges, especially
for researchers and small organizations with limited hardware resources.
Although SSD offloading (i.e., ZeRO-Infinity) has emerged as a viable strategy
to overcome the GPU memory barrier via leveraging both system memory (i.e., CPU
DRAM) and storage space (i.e., solid-state devices, SSDs), its design primarily
targets model-centric performance issues. As a result, key system-level issues,
including system memory fragmentation, inefficient pinned buffer allocation,
peak CPU usage spikes, and file system overhead, remain unaddressed, stifling
scalability and inflating costs. Such an observation motivates this paper to
introduce MemAscend, a framework that systematically tackles the underexplored
system memory bottlenecks in SSD-offloaded LLM training, with a focus on
resource-constrained environments. By streamlining pinned-memory allocation,
eradicating fragmentation, and mitigating peak overhead, MemAscend reclaims a
substantial system memory budget, enabling larger models, longer context
windows, and higher batch sizes without exceeding modest hardware limits.
Across diverse LLM benchmarks, MemAscend reduces peak system-memory consumption
by an average of 55.7% compared with standard SSD offloading techniques,
lowering the hardware barrier for fine-tuning and unlocking new possibilities
for cost-effective large-scale training on limited-resource machines.

### 4. [SealOS+: A Sealos-based Approach for Adaptive Resource Optimization Under Dynamic Workloads for Securities Trading System](http://arxiv.org/pdf/2505.23258v1)

Authors: Haojie Jia, Zhenhao Li, Gen Li, Minxian Xu, Kejiang Ye

As securities trading systems transition to a microservices architecture,
optimizing system performance presents challenges such as inefficient resource
scheduling and high service response delays. Existing container orchestration
platforms lack tailored performance optimization mechanisms for trading
scenarios, making it difficult to meet the stringent 50ms response time
requirement imposed by exchanges. This paper introduces SealOS+, a Sealos-based
performance optimization approach for securities trading, incorporating an
adaptive resource scheduling algorithm leveraging deep reinforcement learning,
a three-level caching mechanism for trading operations, and a Long Short-Term
Memory (LSTM) based load prediction model. Real-world deployment at a
securities exchange demonstrates that the optimized system achieves an average
CPU utilization of 78\%, reduces transaction response time to 105ms, and
reaches a peak processing capacity of 15,000 transactions per second,
effectively meeting the rigorous performance and reliability demands of
securities trading.

### 5. [Complementary Time-Space Tradeoff for Self-Stabilizing Leader Election: Polynomial States Meet Sublinear Time](http://arxiv.org/pdf/2505.23649v1)

Authors: Yuichi Sudo

We study the self-stabilizing leader election (SS-LE) problem in the
population protocol model, assuming exact knowledge of the population size $n$.
Burman, Chen, Chen, Doty, Nowak, Severson, and Xu (PODC 2021) showed that this
problem can be solved in $O(n)$ expected time with $O(n)$ states. Recently,
G\k{a}sieniec, Grodzicki, and Stachowiak (PODC 2025) proved that $n+O(\log n)$
states suffice to achieve $O(n \log n)$ time both in expectation and with high
probability (w.h.p.). If substantially more states are available, sublinear
time can be achieved. Burman~et~al.~(PODC 2021) presented a $2^{O(n^\rho\log
n)}$-state SS-LE protocol with a parameter $\rho$: setting $\rho = \Theta(\log
n)$ yields an optimal $O(\log n)$ time both in expectation and w.h.p., while
$\rho = \Theta(1)$ results in $O(\rho\,n^{1/(\rho+1)})$ expected time. Very
recently, Austin, Berenbrink, Friedetzky, G\"otte, and Hintze (PODC 2025)
presented a novel SS-LE protocol parameterized by a positive integer $\rho$
with $1 \le \rho < n/2$ that solves SS-LE in $O(\frac{n}{\rho}\cdot\log n)$
time w.h.p.\ using $2^{O(\rho^2\log n)}$ states. This paper independently
presents yet another time--space tradeoff of SS-LE: for any positive integer
$\rho$ with $1 \le \rho \le \sqrt{n}$, SS-LE can be achieved within
$O\left(\frac{n}{\rho}\cdot \log\rho\right)$ expected time using
$2^{2\rho\lg\rho + O(\log n)}$ states. The proposed protocol uses significantly
fewer states than the protocol of Austin~et~al.\ requires to achieve any
expected stabilization time above $\Theta(\sqrt{n}\log n)$. When $\rho =
\Theta\left(\frac{\log n}{\log \log n}\right)$,the proposed protocol is the
first to achieve sublinear time while using only polynomially many states. A
limitation of our protocol is that the constraint $\rho\le\sqrt{n}$ prevents
achieving $o(\sqrt{n}\log n)$ time, whereas the protocol of Austin et~al.\ can
surpass this bound.

### 6. [DOPPLER: Dual-Policy Learning for Device Assignment in Asynchronous Dataflow Graphs](http://arxiv.org/pdf/2505.23131v1)

Authors: Xinyu Yao, Daniel Bourgeois, Abhinav Jain, Yuxin Tang, Jiawen Yao, Zhimin Ding, Arlei Silva, Chris Jermaine

We study the problem of assigning operations in a dataflow graph to devices
to minimize execution time in a work-conserving system, with emphasis on
complex machine learning workloads. Prior learning-based methods often struggle
due to three key limitations: (1) reliance on bulk-synchronous systems like
TensorFlow, which under-utilize devices due to barrier synchronization; (2)
lack of awareness of the scheduling mechanism of underlying systems when
designing learning-based methods; and (3) exclusive dependence on reinforcement
learning, ignoring the structure of effective heuristics designed by experts.
In this paper, we propose \textsc{Doppler}, a three-stage framework for
training dual-policy networks consisting of 1) a $\mathsf{SEL}$ policy for
selecting operations and 2) a $\mathsf{PLC}$ policy for placing chosen
operations on devices. Our experiments show that \textsc{Doppler} outperforms
all baseline methods across tasks by reducing system execution time and
additionally demonstrates sampling efficiency by reducing per-episode training
time.

### 7. [The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning](http://arxiv.org/pdf/2505.23176v1)

Authors: Shiwei Li, Xiandi Luo, Haozhao Wang, Xing Tang, Shijie Xu, Weihong Luo, Yuhua Li, Xiuqiang He, Ruixuan Li

To improve the training efficiency of federated learning (FL), previous
research has employed low-rank decomposition techniques to reduce communication
overhead. In this paper, we seek to enhance the performance of these low-rank
decomposition methods. Specifically, we focus on three key issues related to
decomposition in FL: what to decompose, how to decompose, and how to aggregate.
Subsequently, we introduce three novel techniques: Model Update Decomposition
(MUD), Block-wise Kronecker Decomposition (BKD), and Aggregation-Aware
Decomposition (AAD), each targeting a specific issue. These techniques are
complementary and can be applied simultaneously to achieve optimal performance.
Additionally, we provide a rigorous theoretical analysis to ensure the
convergence of the proposed MUD. Extensive experimental results show that our
approach achieves faster convergence and superior accuracy compared to relevant
baseline methods. The code is available at
https://github.com/Leopold1423/fedmud-icml25.

### 8. [Accelerating AllReduce with a Persistent Straggler](http://arxiv.org/pdf/2505.23523v1)

Authors: Arjun Devraj, Eric Ding, Abhishek Vijaya Kumar, Robert Kleinberg, Rachee Singh

Distributed machine learning workloads use data and tensor parallelism for
training and inference, both of which rely on the AllReduce collective to
synchronize gradients or activations. However, bulk-synchronous AllReduce
algorithms can be delayed by a persistent straggler that is slower to reach the
synchronization barrier required to begin the collective. To address this
challenge, we propose StragglAR: an AllReduce algorithm that accelerates
distributed training and inference in the presence of persistent stragglers.
StragglAR implements a ReduceScatter among the remaining GPUs during the
straggler-induced delay, and then executes a novel collective algorithm to
complete the AllReduce once the straggler reaches the synchronization barrier.
StragglAR achieves a 2x theoretical speedup over popular bandwidth-efficient
AllReduce algorithms (e.g., Ring) for large GPU clusters with persistent
stragglers. On an 8-GPU server, our implementation of StragglAR yields a 22%
speedup over state-of-the-art AllReduce algorithms.

### 9. [Accelerated Training of Federated Learning via Second-Order Methods](http://arxiv.org/pdf/2505.23588v1)

Authors: Mrinmay Sen, Sidhant R Nair, C Krishna Mohan

This paper explores second-order optimization methods in Federated Learning
(FL), addressing the critical challenges of slow convergence and the excessive
communication rounds required to achieve optimal performance from the global
model. While existing surveys in FL primarily focus on challenges related to
statistical and device label heterogeneity, as well as privacy and security
concerns in first-order FL methods, less attention has been given to the issue
of slow model training. This slow training often leads to the need for
excessive communication rounds or increased communication costs, particularly
when data across clients are highly heterogeneous. In this paper, we examine
various FL methods that leverage second-order optimization to accelerate the
training process. We provide a comprehensive categorization of state-of-the-art
second-order FL methods and compare their performance based on convergence
speed, computational cost, memory usage, transmission overhead, and
generalization of the global model. Our findings show the potential of
incorporating Hessian curvature through second-order optimization into FL and
highlight key challenges, such as the efficient utilization of Hessian and its
inverse in FL. This work lays the groundwork for future research aimed at
developing scalable and efficient federated optimization methods for improving
the training of the global model in FL.

### 10. [Sustainable Carbon-Aware and Water-Efficient LLM Scheduling in Geo-Distributed Cloud Datacenters](http://arxiv.org/pdf/2505.23554v1)

Authors: Hayden Moore, Sirui Qi, Ninad Hogade, Dejan Milojicic, Cullen Bash, Sudeep Pasricha

In recent years, Large Language Models (LLM) such as ChatGPT, CoPilot, and
Gemini have been widely adopted in different areas. As the use of LLMs
continues to grow, many efforts have focused on reducing the massive training
overheads of these models. But it is the environmental impact of handling user
requests to LLMs that is increasingly becoming a concern. Recent studies
estimate that the costs of operating LLMs in their inference phase can exceed
training costs by 25x per year. As LLMs are queried incessantly, the cumulative
carbon footprint for the operational phase has been shown to far exceed the
footprint during the training phase. Further, estimates indicate that 500 ml of
fresh water is expended for every 20-50 requests to LLMs during inference. To
address these important sustainability issues with LLMs, we propose a novel
framework called SLIT to co-optimize LLM quality of service (time-to-first
token), carbon emissions, water usage, and energy costs. The framework utilizes
a machine learning (ML) based metaheuristic to enhance the sustainability of
LLM hosting across geo-distributed cloud datacenters. Such a framework will
become increasingly vital as LLMs proliferate.

### Digital Libraries

### 1. [Identity resolution of software metadata using Large Language Models](http://arxiv.org/pdf/2505.23500v1)

Authors: Eva Martín del Pico, Josep Lluís Gelpí, Salvador Capella-Gutiérrez

Software is an essential component of research. However, little attention has
been paid to it compared with that paid to research data. Recently, there has
been an increase in efforts to acknowledge and highlight the importance of
software in research activities.
  Structured metadata from platforms like bio.tools, Bioconductor, and Galaxy
ToolShed offers valuable insights into research software in the Life Sciences.
Although originally intended to support discovery and integration, this
metadata can be repurposed for large-scale analysis of software practices.
However, its quality and completeness vary across platforms, reflecting diverse
documentation practices.
  To gain a comprehensive view of software development and sustainability,
consolidating this metadata is necessary, but requires robust mechanisms to
address its heterogeneity and scale.
  This article presents an evaluation of instruction-tuned large language
models for the task of software metadata identity resolution, a critical step
in assembling a cohesive collection of research software. Such a collection is
the reference component for the Software Observatory at OpenEBench, a platform
that aggregates metadata to monitor the FAIRness of research software in the
Life Sciences.
  We benchmarked multiple models against a human-annotated gold standard,
examined their behavior on ambiguous cases, and introduced an agreement-based
proxy for high-confidence automated decisions. The proxy achieved high
precision and statistical robustness, while also highlighting the limitations
of current models and the broader challenges of automating semantic judgment in
FAIR-aligned software metadata across registries and repositories.

### Discrete Mathematics

### 1. [Large induced subgraph with a given pathwidth in outerplanar graphs](http://arxiv.org/pdf/2505.23162v1)

Authors: Naoki Matsumoto, Takamasa Yashima

A long-standing conjecture by Albertson and Berman states that every planar
graph of order $n$ has an induced forest with at least $\lceil \frac{n}{2}
\rceil$ vertices. As a variant of this conjecture, Chappell conjectured that
every planar graph of order $n$ has an induced linear forest with at least
$\lceil \frac{4n}{9} \rceil$ vertices. Pelsmajer proved that every outerplanar
graph of order $n$ has an induced linear forest with at least $\lceil
\frac{4n+2}{7}\rceil$ vertices and this bound is sharp. In this paper, we
investigate the order of induced subgraphs of outerplanar graphs with a given
pathwidth. The above result by Pelsmajer implies that every outerplanar graph
of order $n$ has an induced subgraph with pathwidth one and at least $\lceil
\frac{4n+2}{7}\rceil$ vertices. We extend this to obtain a result on the
maximum order of any outerplanar graph with at most a given pathwidth. We also
give its upper bound which generalizes Pelsmajer's construction.

### 2. [Certified algorithms for numerical semigroups in Rocq](http://arxiv.org/pdf/2505.23205v1)

Authors: Massimo Bartoletti, Stefano Bonzio, Marco Ferrara

A numerical semigroup is a co-finite submonoid of the monoid of non-negative
integers under addition. Many properties of numerical semigroups rely on some
fundamental invariants, such as, among others, the set of gaps (and its
cardinality), the Ap\'ery set or the Frobenius number. Algorithms for
calculating invariants are currently based on computational tools, such as GAP,
which lack proofs (either formal or informal) of their correctness. In this
paper we introduce a Rocq formalization of numerical semigroups. Given the
semigroup generators, we provide certified algorithms for computing some of the
fundamental invariants: the set of gaps, of small elements, the Ap\'ery set,
the multiplicity, the conductor and the Frobenius number. To the best of our
knowledge this is the first formalization of numerical semigroups in any proof
assistant.

### 3. [Quantum Hilbert Transform](http://arxiv.org/pdf/2505.23581v1)

Authors: Nitin Jha, Abhishek Parakh

The Hilbert transform has been one of the foundational transforms in signal
processing, finding it's way into multiple disciplines from cryptography to
biomedical sciences. However, there does not exist any quantum analogue for the
Hilbert transform. In this work, we introduce a formulation for the quantum
Hilbert transform (QHT)and apply it to a quantum steganography protocol. By
bridging classical phase-shift techniques with quantum operations, QHT opens
new pathways in quantum signal processing, communications, sensing, and secure
information hiding.

### Data Structures and Algorithms

### 1. [Differentially Private Space-Efficient Algorithms for Counting Distinct Elements in the Turnstile Model](http://arxiv.org/pdf/2505.23682v1)

Authors: Rachel Cummings, Alessandro Epasto, Jieming Mao, Tamalika Mukherjee, Tingting Ou, Peilin Zhong

The turnstile continual release model of differential privacy captures
scenarios where a privacy-preserving real-time analysis is sought for a dataset
evolving through additions and deletions. In typical applications of real-time
data analysis, both the length of the stream $T$ and the size of the universe
$|U|$ from which data come can be extremely large. This motivates the study of
private algorithms in the turnstile setting using space sublinear in both $T$
and $|U|$. In this paper, we give the first sublinear space differentially
private algorithms for the fundamental problem of counting distinct elements in
the turnstile streaming model. Our algorithm achieves, on arbitrary streams,
$\tilde{O}_{\eta}(T^{1/3})$ space and additive error, and a $(1+\eta)$-relative
approximation for all $\eta \in (0,1)$. Our result significantly improves upon
the space requirements of the state-of-the-art algorithms for this problem,
which is linear, approaching the known $\Omega(T^{1/4})$ additive error lower
bound for arbitrary streams. Moreover, when a bound $W$ on the number of times
an item appears in the stream is known, our algorithm provides
$\tilde{O}_{\eta}(\sqrt{W})$ additive error, using $\tilde{O}_{\eta}(\sqrt{W})$
space. This additive error asymptotically matches that of prior work which
required instead linear space. Our results address an open question posed by
[Jain, Kalemaj, Raskhodnikova, Sivakumar, Smith, Neurips23] about designing
low-memory mechanisms for this problem. We complement these results with a
space lower bound for this problem, which shows that any algorithm that uses
similar techniques must use space $\tilde{\Omega}(T^{1/3})$ on arbitrary
streams.

### 2. [Improved Learning via k-DTW: A Novel Dissimilarity Measure for Curves](http://arxiv.org/pdf/2505.23431v1)

Authors: Amer Krivošija, Alexander Munteanu, André Nusser, Chris Schwiegelshohn

This paper introduces $k$-Dynamic Time Warping ($k$-DTW), a novel
dissimilarity measure for polygonal curves. $k$-DTW has stronger metric
properties than Dynamic Time Warping (DTW) and is more robust to outliers than
the Fr\'{e}chet distance, which are the two gold standards of dissimilarity
measures for polygonal curves. We show interesting properties of $k$-DTW and
give an exact algorithm as well as a $(1+\varepsilon)$-approximation algorithm
for $k$-DTW by a parametric search for the $k$-th largest matched distance. We
prove the first dimension-free learning bounds for curves and further learning
theoretic results. $k$-DTW not only admits smaller sample size than DTW for the
problem of learning the median of curves, where some factors depending on the
curves' complexity $m$ are replaced by $k$, but we also show a surprising
separation on the associated Rademacher and Gaussian complexities: $k$-DTW
admits strictly smaller bounds than DTW, by a factor $\tilde\Omega(\sqrt{m})$
when $k\ll m$. We complement our theoretical findings with an experimental
illustration of the benefits of using $k$-DTW for clustering and nearest
neighbor classification.

### 3. [The Generalized Skew Spectrum of Graphs](http://arxiv.org/pdf/2505.23609v1)

Authors: Armando Bellante, Martin Plávala, Alessandro Luongo

This paper proposes a family of permutation-invariant graph embeddings,
generalizing the Skew Spectrum of graphs of Kondor & Borgwardt (2008). Grounded
in group theory and harmonic analysis, our method introduces a new class of
graph invariants that are isomorphism-invariant and capable of embedding richer
graph structures - including attributed graphs, multilayer graphs, and
hypergraphs - which the Skew Spectrum could not handle. Our generalization
further defines a family of functions that enables a trade-off between
computational complexity and expressivity. By applying
generalization-preserving heuristics to this family, we improve the Skew
Spectrum's expressivity at the same computational cost. We formally prove the
invariance of our generalization, demonstrate its improved expressiveness
through experiments, and discuss its efficient computation.

### 4. [Fast Compressed-Domain N-Point Discrete Fourier Transform](http://arxiv.org/pdf/2505.23718v1)

Authors: Saulo Queiroz

This paper presents a novel algorithm for computing the N-point Discrete
Fourier Transform (DFT) based solely on recursive Rectangular Index Compression
(RIC) [1][2] and structured frequency shifts. The RIC DFT algorithm compresses
a signal from $N=CL$ to $C\in[2,N/2]$ points at the expense of $N-1$ complex
additions and no complex multiplication. It is shown that a $C$-point DFT on
the compressed signal corresponds exactly to $C$ DFT coefficients of the
original $N$-point DFT, namely, $X_{kL}$, $k=0,1,\ldots,C-1$ with no need for
twiddle factors. We rely on this strategy to decompose the DFT by recursively
compressing the input signal and applying global frequency shifts (to get
odd-indexed DFT coefficients). We show that this new structure can relax the
power-of-two assumption of the radix-2 FFT by enabling signal input lengths
such as $N=c\cdot 2^k$ (for $k\geq 0$ and a non-power-of-two $c>0$). Thus, our
algorithm potentially outperforms radix-2 FFTs for the cases where significant
zero-padding is needed. The proposed approach achieves a computational
complexity of $O(N \log N)$ and offers a new structural perspective on DFT
computation, with potential impacts on several DFT issues like numerical
stability, hardware implementation, sparse transforms, convolutions, and others
DFT-based procedures.

### Emerging Technologies

### 1. [MenTeR: A fully-automated Multi-agenT workflow for end-to-end RF/Analog Circuits Netlist Design](http://arxiv.org/pdf/2505.22990v1)

Authors: Pin-Han Chen, Yu-Sheng Lin, Wei-Cheng Lee, Tin-Yu Leu, Po-Hsiang Hsu, Anjana Dissanayake, Sungjin Oh, Chinq-Shiun Chiu

RF/Analog design is essential for bridging digital technologies with
real-world signals, ensuring the functionality and reliability of a wide range
of electronic systems. However, analog design procedures are often intricate,
time-consuming and reliant on expert intuition, and hinder the time and cost
efficiency of circuit development. To overcome the limitations of the manual
circuit design, we introduce MenTeR - a multiagent workflow integrated into an
end-to-end analog design framework. By employing multiple specialized AI agents
that collaboratively address different aspects of the design process, such as
specification understanding, circuit optimization, and test bench validation,
MenTeR reduces the dependency on frequent trial-and-error-style intervention.
MenTeR not only accelerates the design cycle time but also facilitates a
broader exploration of the design space, demonstrating robust capabilities in
handling real-world analog systems. We believe that MenTeR lays the groundwork
for future "RF/Analog Copilots" that can collaborate seamlessly with human
designers.

### 2. [Can Large Language Models Challenge CNNS in Medical Image Analysis?](http://arxiv.org/pdf/2505.23503v1)

Authors: Shibbir Ahmed, Shahnewaz Karim Sakib, Anindya Bijoy Das

This study presents a multimodal AI framework designed for precisely
classifying medical diagnostic images. Utilizing publicly available datasets,
the proposed system compares the strengths of convolutional neural networks
(CNNs) and different large language models (LLMs). This in-depth comparative
analysis highlights key differences in diagnostic performance, execution
efficiency, and environmental impacts. Model evaluation was based on accuracy,
F1-score, average execution time, average energy consumption, and estimated
$CO_2$ emission. The findings indicate that although CNN-based models can
outperform various multimodal techniques that incorporate both images and
contextual information, applying additional filtering on top of LLMs can lead
to substantial performance gains. These findings highlight the transformative
potential of multimodal AI systems to enhance the reliability, efficiency, and
scalability of medical diagnostics in clinical settings.

### 3. [From Connectivity to Autonomy: The Dawn of Self-Evolving Communication Systems](http://arxiv.org/pdf/2505.23710v1)

Authors: Zeinab Nezami, Syed Danial Ali Shah, Maryam Hafeez, Karim Djemame, Syed Ali Raza Zaidi

This paper envisions 6G as a self-evolving telecom ecosystem, where AI-driven
intelligence enables dynamic adaptation beyond static connectivity. We explore
the key enablers of autonomous communication systems, spanning reconfigurable
infrastructure, adaptive middleware, and intelligent network functions,
alongside multi-agent collaboration for distributed decision-making. We explore
how these methodologies align with emerging industrial IoT frameworks, ensuring
seamless integration within digital manufacturing processes. Our findings
emphasize the potential for improved real-time decision-making, optimizing
efficiency, and reducing latency in networked control systems. The discussion
addresses ethical challenges, research directions, and standardization efforts,
concluding with a technology stack roadmap to guide future developments. By
leveraging state-of-the-art 6G network management techniques, this research
contributes to the next generation of intelligent automation solutions,
bridging the gap between theoretical advancements and real-world industrial
applications.

### Formal Languages and Automata Theory

### 1. [Mind the Gap: A Formal Investigation of the Relationship Between Log and Model Complexity -- Extended Version](http://arxiv.org/pdf/2505.23233v1)

Authors: Patrizia Schalk, Artem Polyvyanyy

Simple process models are key for effectively communicating the outcomes of
process mining. An important question in this context is whether the complexity
of event logs used as inputs to process discovery algorithms can serve as a
reliable indicator of the complexity of the resulting process models. Although
various complexity measures for both event logs and process models have been
proposed in the literature, the relationship between input and output
complexity remains largely unexplored. In particular, there are no established
guidelines or theoretical foundations that explain how the complexity of an
event log influences the complexity of the discovered model. This paper
examines whether formal guarantees exist such that increasing the complexity of
event logs leads to increased complexity in the discovered models. We study 18
log complexity measures and 17 process model complexity measures across five
process discovery algorithms. Our findings reveal that only the complexity of
the flower model can be established by an event log complexity measure. For all
other algorithms, we investigate which log complexity measures influence the
complexity of the discovered models. The results show that current log
complexity measures are insufficient to decide which discovery algorithms to
choose to construct simple models. We propose that authors of process discovery
algorithms provide insights into which log complexity measures predict the
complexity of their results.

### Graphics

### 1. [Quality assessment of 3D human animation: Subjective and objective evaluation](http://arxiv.org/pdf/2505.23301v1)

Authors: Rim Rekik, Stefanie Wuhrer, Ludovic Hoyet, Katja Zibrek, Anne-Hélène Olivier

Virtual human animations have a wide range of applications in virtual and
augmented reality. While automatic generation methods of animated virtual
humans have been developed, assessing their quality remains challenging.
Recently, approaches introducing task-oriented evaluation metrics have been
proposed, leveraging neural network training. However, quality assessment
measures for animated virtual humans that are not generated with parametric
body models have yet to be developed. In this context, we introduce a first
such quality assessment measure leveraging a novel data-driven framework.
First, we generate a dataset of virtual human animations together with their
corresponding subjective realism evaluation scores collected with a user study.
Second, we use the resulting dataset to learn predicting perceptual evaluation
scores. Results indicate that training a linear regressor on our dataset
results in a correlation of 90%, which outperforms a state of the art deep
learning baseline.

### 2. [To Measure What Isn't There -- Visual Exploration of Missingness Structures Using Quality Metrics](http://arxiv.org/pdf/2505.23447v1)

Authors: Sara Johansson Fernstad, Sarah Alsufyani, Silvia Del Din, Alison Yarnall, Lynn Rochester

This paper contributes a set of quality metrics for identification and visual
analysis of structured missingness in high-dimensional data. Missing values in
data are a frequent challenge in most data generating domains and may cause a
range of analysis issues. Structural missingness in data may indicate issues in
data collection and pre-processing, but may also highlight important data
characteristics. While research into statistical methods for dealing with
missing data are mainly focusing on replacing missing values with plausible
estimated values, visualization has great potential to support a more in-depth
understanding of missingness structures in data. Nonetheless, while the
interest in missing data visualization has increased in the last decade, it is
still a relatively overlooked research topic with a comparably small number of
publications, few of which address scalability issues. Efficient visual
analysis approaches are needed to enable exploration of missingness structures
in large and high-dimensional data, and to support informed decision-making in
context of potential data quality issues. This paper suggests a set of quality
metrics for identification of patterns of interest for understanding of
structural missingness in data. These quality metrics can be used as guidance
in visual analysis, as demonstrated through a use case exploring structural
missingness in data from a real-life walking monitoring study. All supplemental
materials for this paper are available at
https://doi.org/10.25405/data.ncl.c.7741829.

### 3. [Errors in Stereo Geometry Induce Distance Misperception](http://arxiv.org/pdf/2505.23685v1)

Authors: Raffles Xingqi Zhu, Charlie S. Burlingham, Olivier Mercier, Phillip Guan

Stereoscopic head-mounted displays (HMDs) render and present binocular images
to create an egocentric, 3D percept to the HMD user. Within this render and
presentation pipeline there are potential rendering camera and viewing position
errors that can induce deviations in the depth and distance that a user
perceives compared to the underlying intended geometry. For example, rendering
errors can arise when HMD render cameras are incorrectly positioned relative to
the assumed centers of projections of the HMD displays and viewing errors can
arise when users view stereo geometry from the incorrect location in the HMD
eyebox. In this work we present a geometric framework that predicts errors in
distance perception arising from inaccurate HMD perspective geometry and build
an HMD platform to reliably simulate render and viewing error in a Quest 3 HMD
with eye tracking to experimentally test these predictions. We present a series
of five experiments to explore the efficacy of this geometric framework and
show that errors in perspective geometry can induce both under- and
over-estimations in perceived distance. We further demonstrate how real-time
visual feedback can be used to dynamically recalibrate visuomotor mapping so
that an accurate reach distance is achieved even if the perceived visual
distance is negatively impacted by geometric error.

### 4. [AMOR: Adaptive Character Control through Multi-Objective Reinforcement Learning](http://arxiv.org/pdf/2505.23708v1)

Authors: Lucas N. Alegre, Agon Serifi, Ruben Grandia, David Müller, Espen Knoop, Moritz Bächer

Reinforcement learning (RL) has significantly advanced the control of
physics-based and robotic characters that track kinematic reference motion.
However, methods typically rely on a weighted sum of conflicting reward
functions, requiring extensive tuning to achieve a desired behavior. Due to the
computational cost of RL, this iterative process is a tedious, time-intensive
task. Furthermore, for robotics applications, the weights need to be chosen
such that the policy performs well in the real world, despite inevitable
sim-to-real gaps. To address these challenges, we propose a multi-objective
reinforcement learning framework that trains a single policy conditioned on a
set of weights, spanning the Pareto front of reward trade-offs. Within this
framework, weights can be selected and tuned after training, significantly
speeding up iteration time. We demonstrate how this improved workflow can be
used to perform highly dynamic motions with a robot character. Moreover, we
explore how weight-conditioned policies can be leveraged in hierarchical
settings, using a high-level policy to dynamically select weights according to
the current task. We show that the multi-objective policy encodes a diverse
spectrum of behaviors, facilitating efficient adaptation to novel tasks.

### 5. [How Animals Dance (When You're Not Looking)](http://arxiv.org/pdf/2505.23738v1)

Authors: Xiaojuan Wang, Aleksander Holynski, Brian Curless, Ira Kemelmacher, Steve Seitz

We present a keyframe-based framework for generating music-synchronized,
choreography aware animal dance videos. Starting from a few keyframes
representing distinct animal poses -- generated via text-to-image prompting or
GPT-4o -- we formulate dance synthesis as a graph optimization problem: find
the optimal keyframe structure that satisfies a specified choreography pattern
of beats, which can be automatically estimated from a reference dance video. We
also introduce an approach for mirrored pose image generation, essential for
capturing symmetry in dance. In-between frames are synthesized using an video
diffusion model. With as few as six input keyframes, our method can produce up
to 30 second dance videos across a wide range of animals and music tracks.

### 6. [LayerPeeler: Autoregressive Peeling for Layer-wise Image Vectorization](http://arxiv.org/pdf/2505.23740v1)

Authors: Ronghuan Wu, Wanchao Su, Jing Liao

Image vectorization is a powerful technique that converts raster images into
vector graphics, enabling enhanced flexibility and interactivity. However,
popular image vectorization tools struggle with occluded regions, producing
incomplete or fragmented shapes that hinder editability. While recent
advancements have explored rule-based and data-driven layer-wise image
vectorization, these methods face limitations in vectorization quality and
flexibility. In this paper, we introduce LayerPeeler, a novel layer-wise image
vectorization approach that addresses these challenges through a progressive
simplification paradigm. The key to LayerPeeler's success lies in its
autoregressive peeling strategy: by identifying and removing the topmost
non-occluded layers while recovering underlying content, we generate vector
graphics with complete paths and coherent layer structures. Our method
leverages vision-language models to construct a layer graph that captures
occlusion relationships among elements, enabling precise detection and
description for non-occluded layers. These descriptive captions are used as
editing instructions for a finetuned image diffusion model to remove the
identified layers. To ensure accurate removal, we employ localized attention
control that precisely guides the model to target regions while faithfully
preserving the surrounding content. To support this, we contribute a
large-scale dataset specifically designed for layer peeling tasks. Extensive
quantitative and qualitative experiments demonstrate that LayerPeeler
significantly outperforms existing techniques, producing vectorization results
with superior path semantics, geometric regularity, and visual fidelity.

### 7. [One Trajectory, One Token: Grounded Video Tokenization via Panoptic Sub-object Trajectory](http://arxiv.org/pdf/2505.23617v1)

Authors: Chenhao Zheng, Jieyu Zhang, Mohammadreza Salehi, Ziqi Gao, Vishnu Iyengar, Norimasa Kobori, Quan Kong, Ranjay Krishna

Effective video tokenization is critical for scaling transformer models for
long videos. Current approaches tokenize videos using space-time patches,
leading to excessive tokens and computational inefficiencies. The best token
reduction strategies degrade performance and barely reduce the number of tokens
when the camera moves. We introduce grounded video tokenization, a paradigm
that organizes tokens based on panoptic sub-object trajectories rather than
fixed patches. Our method aligns with fundamental perceptual principles,
ensuring that tokenization reflects scene complexity rather than video
duration. We propose TrajViT, a video encoder that extracts object trajectories
and converts them into semantically meaningful tokens, significantly reducing
redundancy while maintaining temporal coherence. Trained with contrastive
learning, TrajViT significantly outperforms space-time ViT (ViT3D) across
multiple video understanding benchmarks, e.g., TrajViT outperforms ViT3D by a
large margin of 6% top-5 recall in average at video-text retrieval task with
10x token deduction. We also show TrajViT as a stronger model than ViT3D for
being the video encoder for modern VideoLLM, obtaining an average of 5.2%
performance improvement across 6 VideoQA benchmarks while having 4x faster
training time and 18x less inference FLOPs. TrajViT is the first efficient
encoder to consistently outperform ViT3D across diverse video analysis tasks,
making it a robust and scalable solution.

### Computer Science and Game Theory

### 1. [Online Selection with Uncertain Disruption](http://arxiv.org/pdf/2505.22999v1)

Authors: Yihua Xu, Süleyman Kerimov, Sebastian Perez-Salazar

In numerous online selection problems, decision-makers (DMs) must allocate on
the fly limited resources to customers with uncertain values. The DM faces the
tension between allocating resources to currently observed values and saving
them for potentially better, unobserved values in the future. Addressing this
tension becomes more demanding if an uncertain disruption occurs while serving
customers. Without any disruption, the DM gets access to the capacity
information to serve customers throughout the time horizon. However, with
uncertain disruption, the DM must act more cautiously due to risk of running
out of capacity abruptly or misusing the resources. Motivated by this tension,
we introduce the Online Selection with Uncertain Disruption (OS-UD) problem. In
OS-UD, a DM sequentially observes n non-negative values drawn from a common
distribution and must commit to select or reject each value in real time,
without revisiting past values. The disruption is modeled as a Bernoulli random
variable with probability p each time DM selects a value. We aim to design an
online algorithm that maximizes the expected sum of selected values before a
disruption occurs, if any.
  We evaluate online algorithms using the competitive ratio. Using a
quantile-based approach, we devise a non-adaptive single-threshold algorithm
that attains a competitive ratio of at least 1-1/e, and an adaptive threshold
algorithm characterized by a sequence of non-increasing thresholds that attains
an asymptotic competitive ratio of at least 0.745. Both of these results are
worst-case optimal within their corresponding class of algorithms.

### 2. [Achieving Equitability with Subsidy](http://arxiv.org/pdf/2505.23251v1)

Authors: Yuanyuan Wang, Tianze Wei

We study the fair allocation problem of indivisible items with subsidies. In
this paper, we mainly consider the notion of fairness - equitability (EQ),
which requires that items be allocated such that all agents value the bundle
they receive equally. Firstly, we study the upper bounds of the required
subsidy to achieve EQ in different settings of items and provide the
corresponding lower bounds. Secondly, we consider the bounded subsidy for
achieving EQ and another popular notion of fairness - envy-freeness (EF) and
give a characterization of the allocations that can achieve both EQ and EF.
Finally, we analyze the bounds of subsidy of the allocations achieving fairness
and efficiency (utilitarian social welfare or Nash welfare), and design several
polynomial-time algorithms to compute the desired allocation.

### 3. [Learning Recommender Mechanisms for Bayesian Stochastic Games](http://arxiv.org/pdf/2505.22979v1)

Authors: Bengisu Guresti, Chongjie Zhang, Yevgeniy Vorobeychik

An important challenge in non-cooperative game theory is coordinating on a
single (approximate) equilibrium from many possibilities - a challenge that
becomes even more complex when players hold private information. Recommender
mechanisms tackle this problem by recommending strategies to players based on
their reported type profiles. A key consideration in such mechanisms is to
ensure that players are incentivized to participate, report their private
information truthfully, and follow the recommendations. While previous work has
focused on designing recommender mechanisms for one-shot and extensive-form
games, these approaches cannot be effectively applied to stochastic games,
particularly if we constrain recommendations to be Markov stationary policies.
To bridge this gap, we introduce a novel bi-level reinforcement learning
approach for automatically designing recommender mechanisms in Bayesian
stochastic games. Our method produces a mechanism represented by a parametric
function (such as a neural network), and is therefore highly efficient at
execution time. Experimental results on two repeated and two stochastic games
demonstrate that our approach achieves social welfare levels competitive with
cooperative multi-agent reinforcement learning baselines, while also providing
significantly improved incentive properties.

### 4. [Learning to Incentivize in Repeated Principal-Agent Problems with Adversarial Agent Arrivals](http://arxiv.org/pdf/2505.23124v1)

Authors: Junyan Liu, Arnab Maiti, Artin Tajdini, Kevin Jamieson, Lillian J. Ratliff

We initiate the study of a repeated principal-agent problem over a finite
horizon $T$, where a principal sequentially interacts with $K\geq 2$ types of
agents arriving in an adversarial order. At each round, the principal
strategically chooses one of the $N$ arms to incentivize for an arriving agent
of unknown type. The agent then chooses an arm based on its own utility and the
provided incentive, and the principal receives a corresponding reward. The
objective is to minimize regret against the best incentive in hindsight.
Without prior knowledge of agent behavior, we show that the problem becomes
intractable, leading to linear regret. We analyze two key settings where
sublinear regret is achievable. In the first setting, the principal knows the
arm each agent type would select greedily for any given incentive. Under this
setting, we propose an algorithm that achieves a regret bound of
$O(\min\{\sqrt{KT\log N},K\sqrt{T}\})$ and provide a matching lower bound up to
a $\log K$ factor. In the second setting, an agent's response varies smoothly
with the incentive and is governed by a Lipschitz constant $L\geq 1$. Under
this setting, we show that there is an algorithm with a regret bound of
$\tilde{O}((LN)^{1/3}T^{2/3})$ and establish a matching lower bound up to
logarithmic factors. Finally, we extend our algorithmic results for both
settings by allowing the principal to incentivize multiple arms simultaneously
in each round.

### 5. [Distortion of AI Alignment: Does Preference Optimization Optimize for Preferences?](http://arxiv.org/pdf/2505.23749v1)

Authors: Paul Gölz, Nika Haghtalab, Kunhe Yang

After pre-training, large language models are aligned with human preferences
based on pairwise comparisons. State-of-the-art alignment methods (such as
PPO-based RLHF and DPO) are built on the assumption of aligning with a single
preference model, despite being deployed in settings where users have diverse
preferences. As a result, it is not even clear that these alignment methods
produce models that satisfy users on average -- a minimal requirement for
pluralistic alignment. Drawing on social choice theory and modeling users'
comparisons through individual Bradley-Terry (BT) models, we introduce an
alignment method's distortion: the worst-case ratio between the optimal
achievable average utility, and the average utility of the learned policy.
  The notion of distortion helps draw sharp distinctions between alignment
methods: Nash Learning from Human Feedback achieves the minimax optimal
distortion of $(\frac{1}{2} + o(1)) \cdot \beta$ (for the BT temperature
$\beta$), robustly across utility distributions, distributions of comparison
pairs, and permissible KL divergences from the reference policy. RLHF and DPO,
by contrast, suffer $\geq (1 - o(1)) \cdot \beta$ distortion already without a
KL constraint, and $e^{\Omega(\beta)}$ or even unbounded distortion in the full
setting, depending on how comparison pairs are sampled.

### Human-Computer Interaction

### 1. [Evaluating Driver Perceptions of Integrated Safety Monitoring Systems for Alcohol Impairment and Distraction](http://arxiv.org/pdf/2505.22969v1)

Authors: RoshikNagaSai Patibandla, Ross Greer

The increasing number of accidents caused by alcohol-impaired driving has
prompted the development of integrated safety systems in vehicles to monitor
driver behavior and prevent crashes. This paper explores how drivers perceive
these systems, focusing on their comfort, trust, privacy concerns, and
willingness to adopt the technology. Through a survey of 115 U.S. participants,
the study reveals a preference for non-intrusive systems, such as those
monitoring eye movements, over more restrictive technologies like alcohol
detection devices. Privacy emerged as a major concern, with many participants
preferring local data processing and anonymity. Trust in these systems was
crucial for acceptance, as drivers are more likely to adapt their behavior when
they believe the system is accurate and reliable. To encourage adoption, it is
important to address concerns about privacy and balance the benefits of safety
with personal freedom. By improving transparency, ensuring reliability, and
increasing public awareness, these systems could play a significant role in
reducing road accidents and improving safety.

### 2. [Free Lunch for User Experience: Crowdsourcing Agents for Scalable User Studies](http://arxiv.org/pdf/2505.22981v1)

Authors: Siyang Liu, Sahand Sabour, Xiaoyang Wang, Rada Mihalcea

We demonstrate the potential of anthropomorphized language agents to generate
budget-friendly, moderate-fidelity, yet sufficiently insightful user
experiences at scale, supporting fast, early-stage prototyping. We explore this
through the case of prototyping Large Language Model-driven non-player
characters (NPCs). We present Agentic H-CI, a framework that mirrors
traditional user research processes-surveying, screening, experiencing, and
collecting feedback and insights-with simulated agents. Using this approach, we
easily construct a team of 240 player agents with a balanced range of player
types and personality traits, at extremely low cost (\$0.28/player) and minimal
time commitment (6.9 minutes/player). Content analysis shows that agent-based
players behave in ways aligned with their simulated backgrounds, achieving
82.5\% alignment with designated profiles. From their interactions, we distill
11 user insights and 6 design implications to guide further development. To
evaluate practical value, we conduct parallel user studies with human
participants recruited locally and via crowdsourcing. Ratings from three
professional game developers show that the agentic player team offers a
Pareto-optimal and well-balanced trade-off across fidelity, cost, time
efficiency, and insight helpfulness.

### 3. [Vision-Based Assistive Technologies for People with Cerebral Visual Impairment: A Review and Focus Study](http://arxiv.org/pdf/2505.22983v1)

Authors: Bhanuka Gamage, Leona Holloway, Nicola McDowell, Thanh-Toan Do, Nicholas Price, Arthur Lowery, Kim Marriott

Over the past decade, considerable research has investigated Vision-Based
Assistive Technologies (VBAT) to support people with vision impairments to
understand and interact with their immediate environment using machine
learning, computer vision, image enhancement, and/or augmented/virtual reality.
However, this has almost totally overlooked a growing demographic: people with
Cerebral Visual Impairment (CVI). Unlike ocular vision impairments, CVI arises
from damage to the brain's visual processing centres. Through a scoping review,
this paper reveals a significant research gap in addressing the needs of this
demographic. Three focus studies involving 7 participants with CVI explored the
challenges, current strategies, and opportunities for VBAT. We also discussed
the assistive technology needs of people with CVI compared with ocular low
vision. Our findings highlight the opportunity for the Human-Computer
Interaction and Assistive Technologies research community to explore and
address this underrepresented domain, thereby enhancing the quality of life for
people with CVI.

### 4. [iTrace : Interactive Tracing of Cross-View Data Relationships](http://arxiv.org/pdf/2505.23079v1)

Authors: Abdul Rahman Shaikh, Maoyuan Sun, Xingchen Liu, Hamed Alhoori, Jian Zhao, David Koop

Exploring data relations across multiple views has been a common task in many
domains such as bioinformatics, cybersecurity, and healthcare. To support this,
various techniques (e.g., visual links and brushing and linking) are used to
show related visual elements across views via lines and highlights. However,
understanding the relations using these techniques, when many related elements
are scattered, can be difficult due to spatial distance and complexity. To
address this, we present iTrace, an interactive visualization technique to
effectively trace cross-view data relationships. iTrace leverages the concept
of interactive focus transitions, which allows users to see and directly
manipulate their focus as they navigate between views. By directing the user's
attention through smooth transitions between related elements, iTrace makes it
easier to follow data relationships. We demonstrate the effectiveness of iTrace
with a user study, and we conclude with a discussion of how iTrace can be
broadly used to enhance data exploration in various types of visualizations.

### 5. [Investigating A Geometrical Solution to the Vergence-Accommodation Conflict for Targeted Movements in Virtual Reality](http://arxiv.org/pdf/2505.23310v1)

Authors: Xiaoye Michael Wang, Matthew Prenevost, Aneesh Tarun, Ian Robinson, Michael Nitsche, Gabby Resch, Ali Mazalek, Timothy N. Welsh

While virtual reality (VR) holds significant potential to revolutionize
digital user interaction, how visual information is presented through VR
head-mounted displays (HMDs) differs from naturalistic viewing and interactions
in physical environments, leading to performance decrements. One critical
challenge in VR development is the vergence-accommodation conflict (VAC), which
arises due to the intrinsic constraints of approximating the natural viewing
geometry through digital displays. Although various hardware and software
solutions have been proposed to address VAC, no commercially viable option has
been universally adopted by manufacturers. This paper presents and evaluates a
software solution grounded in a vision-based geometrical model of VAC that
mediates VAC's impact on movement in VR. This model predicts the impact of VAC
as a constant offset to the vergence angle, distorting the binocular viewing
geometry that results in movement undershooting. In Experiment 1, a 3D pointing
task validated the model's predictions and demonstrated that VAC primarily
affects online movements involving real-time visual feedback. Experiment 2
implemented a shader program to rectify the effect of VAC, improving movement
accuracy by approximately 30%. Overall, this work presented a practical
approach to reducing the impact of VAC on HMD-based manual interactions,
enhancing the user experience in virtual environments.

### 6. [Self-driving technologies need the help of the public: A narrative review of the evidence](http://arxiv.org/pdf/2505.23472v1)

Authors: Jonathan Smith, Siddartha Khastgir

If public trust is lot in a new technology early in its life cycle it can
take much more time for the benefits of that technology to be realised.
Eventually tens-of-millions of people will collectively have the power to
determine self-driving technology success of failure driven by their perception
of risk, data handling, safety, governance, accountability, benefits to their
life and more. This paper reviews the evidence on safety critical technology
covering trust, engagement, and acceptance. The paper takes a narrative review
approach concluding with a scalable model for self-driving technology education
and engagement. The paper find that if a mismatch between the publics
perception and expectations about self driving systems emerge it can lead to
misuse, disuse, or abuse of the system. Furthermore we find from the evidence
that industrial experts often misunderstand what matters to the public, users,
and stakeholders. However we find that engagement programmes that develop
approaches to defining the right information at the right time, in the right
format orientated around what matters to the public creates the potential for
ever more sophisticated conversations, greater trust, and moving the public
into a progressive more active role of critique and advocacy. This work has
been undertaken as part of the Partners for Automated Vehicle Education (PAVE)
United Kingdom programme.

### 7. [DTBIA: An Immersive Visual Analytics System for Brain-Inspired Research](http://arxiv.org/pdf/2505.23730v1)

Authors: Jun-Hsiang Yao, Mingzheng Li, Jiayi Liu, Yuxiao Li, Jielin Feng, Jun Han, Qibao Zheng, Jianfeng Feng, Siming Chen

The Digital Twin Brain (DTB) is an advanced artificial intelligence framework
that integrates spiking neurons to simulate complex cognitive functions and
collaborative behaviors. For domain experts, visualizing the DTB's simulation
outcomes is essential to understanding complex cognitive activities. However,
this task poses significant challenges due to DTB data's inherent
characteristics, including its high-dimensionality, temporal dynamics, and
spatial complexity. To address these challenges, we developed DTBIA, an
Immersive Visual Analytics System for Brain-Inspired Research. In collaboration
with domain experts, we identified key requirements for effectively visualizing
spatiotemporal and topological patterns at multiple levels of detail. DTBIA
incorporates a hierarchical workflow - ranging from brain regions to voxels and
slice sections - along with immersive navigation and a 3D edge bundling
algorithm to enhance clarity and provide deeper insights into both functional
(BOLD) and structural (DTI) brain data. The utility and effectiveness of DTBIA
are validated through two case studies involving with brain research experts.
The results underscore the system's role in enhancing the comprehension of
complex neural behaviors and interactions.

### 8. [Seeing the Politics of Decentralized Social Media Protocols](http://arxiv.org/pdf/2505.22962v1)

Authors: Tolulope Oshinowo, Sohyeon Hwang, Amy X. Zhang, Andrés Monroy-Hernández

Calls to decentralize feed-based social media have been driven by concerns
about the concentrated power of centralized platforms and their societal
impact. In response, numerous decentralized social media protocols have
emerged, each interpreting "decentralization" in different ways. We analyze
four such protocols -- ActivityPub, AT Protocol, Nostr, and Farcaster -- to
develop a novel conceptual framework for understanding how protocols
operationalize decentralization. Drawing from protocol documentation, media
coverage, and first-hand interviews with protocol developers and experts, we
contextualize each protocol's approach within their respective socio-technical
goals. Our framework highlights how control over key components is distributed
differently across each protocol, shaping who holds power over what kinds of
decisions. How components are arranged in relation to one another further
impacts how component owners might offset each other's power in shaping social
media. We argue that examining protocols as artifacts reveals how values shape
infrastructure and power dynamics -- and that with a holistic framework as a
guide, we can more effectively evaluate and design decentralized platforms
aligned with the social and political futures we envision.

### 9. [A Constructed Response: Designing and Choreographing Robot Arm Movements in Collaborative Dance Improvisation](http://arxiv.org/pdf/2505.23090v1)

Authors: Xiaoyu Chang, Fan Zhang, Kexue Fu, Carla Diana, Wendy Ju, Ray LC

Dancers often prototype movements themselves or with each other during
improvisation and choreography. How are these interactions altered when
physically manipulable technologies are introduced into the creative process?
To understand how dancers design and improvise movements while working with
instruments capable of non-humanoid movements, we engaged dancers in workshops
to co-create movements with a robot arm in one-human-to-one-robot and
three-human-to-one-robot settings. We found that dancers produced more fluid
movements in one-to-one scenarios, experiencing a stronger sense of connection
and presence with the robot as a co-dancer. In three-to-one scenarios, the
dancers divided their attention between the human dancers and the robot,
resulting in increased perceived use of space and more stop-and-go movements,
perceiving the robot as part of the stage background. This work highlights how
technologies can drive creativity in movement artists adapting to new ways of
working with physical instruments, contributing design insights supporting
artistic collaborations with non-humanoid agents.

### 10. [Eye-tracking-Driven Shared Control for Robotic Arms:Wizard of Oz Studies to Assess Design Choices](http://arxiv.org/pdf/2505.23147v1)

Authors: Anke Fischer-Janzen, Thomas M. Wendt, Daniel Görlich, Kristof Van Laerhoven

Advances in eye-tracking control for assistive robotic arms provide intuitive
interaction opportunities for people with physical disabilities. Shared control
has gained interest in recent years by improving user satisfaction through
partial automation of robot control. We present an eye-tracking-guided shared
control design based on insights from state-of-the-art literature. A Wizard of
Oz setup was used in which automation was simulated by an experimenter to
evaluate the concept without requiring full implementation. This approach
allowed for rapid exploration of user needs and expectations to inform future
iterations. Two studies were conducted to assess user experience, identify
design challenges, and find improvements to ensure usability and accessibility.
The first study involved people with disabilities by providing a survey, and
the second study used the Wizard of Oz design in person to gain technical
insights, leading to a comprehensive picture of findings.

### Information Retrieval

### 1. [Augment or Not? A Comparative Study of Pure and Augmented Large Language Model Recommenders](http://arxiv.org/pdf/2505.23053v1)

Authors: Wei-Hsiang Huang, Chen-Wei Ke, Wei-Ning Chiu, Yu-Xuan Su, Chun-Chun Yang, Chieh-Yuan Cheng, Yun-Nung Chen, Pu-Jen Cheng

Large language models (LLMs) have introduced new paradigms for recommender
systems by enabling richer semantic understanding and incorporating implicit
world knowledge. In this study, we propose a systematic taxonomy that
classifies existing approaches into two categories: (1) Pure LLM Recommenders,
which rely solely on LLMs, and (2) Augmented LLM Recommenders, which integrate
additional non-LLM techniques to enhance performance. This taxonomy provides a
novel lens through which to examine the evolving landscape of LLM-based
recommendation. To support fair comparison, we introduce a unified evaluation
platform that benchmarks representative models under consistent experimental
settings, highlighting key design choices that impact effectiveness. We
conclude by discussing open challenges and outlining promising directions for
future research. This work offers both a comprehensive overview and practical
guidance for advancing next-generation LLM-powered recommender.

### 2. [From Token to Action: State Machine Reasoning to Mitigate Overthinking in Information Retrieval](http://arxiv.org/pdf/2505.23059v1)

Authors: Dohyeon Lee, Yeonseok Jeong, Seung-won Hwang

Chain-of-Thought (CoT) prompting enables complex reasoning in large language
models (LLMs), including applications in information retrieval (IR). However,
it often leads to overthinking, where models produce excessively long and
semantically redundant traces with little or no benefit. We identify two key
challenges in IR: redundant trajectories that revisit similar states and
misguided reasoning that diverges from user intent. To address these, we
propose State Machine Reasoning (SMR), a transition-based reasoning framework
composed of discrete actions (Refine, Rerank, Stop) that support early stopping
and fine-grained control. Experiments on the BEIR and BRIGHT benchmarks show
that SMR improves retrieval performance (nDCG@10) by 3.4% while reducing token
usage by 74.4%. It generalizes across LLMs and retrievers without requiring
task-specific tuning, offering a practical alternative to conventional CoT
reasoning. The code and details are available at https://github.com/ldilab/SMR.

### 3. [Deep Retrieval at CheckThat! 2025: Identifying Scientific Papers from Implicit Social Media Mentions via Hybrid Retrieval and Re-Ranking](http://arxiv.org/pdf/2505.23250v1)

Authors: Pascal J. Sager, Ashwini Kamaraj, Benjamin F. Grewe, Thilo Stadelmann

We present the methodology and results of the Deep Retrieval team for subtask
4b of the CLEF CheckThat! 2025 competition, which focuses on retrieving
relevant scientific literature for given social media posts. To address this
task, we propose a hybrid retrieval pipeline that combines lexical precision,
semantic generalization, and deep contextual re-ranking, enabling robust
retrieval that bridges the informal-to-formal language gap. Specifically, we
combine BM25-based keyword matching with a FAISS vector store using a
fine-tuned INF-Retriever-v1 model for dense semantic retrieval. BM25 returns
the top 30 candidates, and semantic search yields 100 candidates, which are
then merged and re-ranked via a large language model (LLM)-based cross-encoder.
  Our approach achieves a mean reciprocal rank at 5 (MRR@5) of 76.46% on the
development set and 66.43% on the hidden test set, securing the 1st position on
the development leaderboard and ranking 3rd on the test leaderboard (out of 31
teams), with a relative performance gap of only 2 percentage points compared to
the top-ranked system. We achieve this strong performance by running
open-source models locally and without external training data, highlighting the
effectiveness of a carefully designed and fine-tuned retrieval pipeline.

### 4. [Bridging the Gap Between Semantic and User Preference Spaces for Multi-modal Music Representation Learning](http://arxiv.org/pdf/2505.23298v1)

Authors: Xiaofeng Pan, Jing Chen, Haitong Zhang, Menglin Xing, Jiayi Wei, Xuefeng Mu, Zhongqian Xie

Recent works of music representation learning mainly focus on learning
acoustic music representations with unlabeled audios or further attempt to
acquire multi-modal music representations with scarce annotated audio-text
pairs. They either ignore the language semantics or rely on labeled audio
datasets that are difficult and expensive to create. Moreover, merely modeling
semantic space usually fails to achieve satisfactory performance on music
recommendation tasks since the user preference space is ignored. In this paper,
we propose a novel Hierarchical Two-stage Contrastive Learning (HTCL) method
that models similarity from the semantic perspective to the user perspective
hierarchically to learn a comprehensive music representation bridging the gap
between semantic and user preference spaces. We devise a scalable audio encoder
and leverage a pre-trained BERT model as the text encoder to learn audio-text
semantics via large-scale contrastive pre-training. Further, we explore a
simple yet effective way to exploit interaction data from our online music
platform to adapt the semantic space to user preference space via contrastive
fine-tuning, which differs from previous works that follow the idea of
collaborative filtering. As a result, we obtain a powerful audio encoder that
not only distills language semantics from the text encoder but also models
similarity in user preference space with the integrity of semantic space
preserved. Experimental results on both music semantic and recommendation tasks
confirm the effectiveness of our method.

### 5. [Engineering Serendipity through Recommendations of Items with Atypical Aspects](http://arxiv.org/pdf/2505.23580v1)

Authors: Ramit Aditya, Razvan Bunescu, Smita Nannaware, Erfan Al-Hossami

A restaurant dinner or a hotel stay may lead to memorable experiences when
guests encounter unexpected aspects that also match their interests. For
example, an origami-making station in the waiting area of a restaurant may be
both surprising and enjoyable for a customer who is passionate about paper
crafts. Similarly, an exhibit of 18th century harpsichords would be atypical
for a hotel lobby and likely pique the interest of a guest who has a passion
for Baroque music. Motivated by this insight, in this paper we introduce the
new task of engineering serendipity through recommendations of items with
atypical aspects. We describe an LLM-based system pipeline that extracts
atypical aspects from item reviews, then estimates and aggregates their
user-specific utility in a measure of serendipity potential that is used to
rerank a list of items recommended to the user. To facilitate system
development and evaluation, we introduce a dataset of Yelp reviews that are
manually annotated with atypical aspects and a dataset of artificially
generated user profiles, together with crowdsourced annotations of user-aspect
utility values. Furthermore, we introduce a custom procedure for dynamic
selection of in-context learning examples, which is shown to improve LLM-based
judgments of atypicality and utility. Experimental evaluations show that
serendipity-based rankings generated by the system are highly correlated with
ground truth rankings for which serendipity scores are computed from manual
annotations of atypical aspects and their user-dependent utility. Overall, we
hope that the new recommendation task and the associated system presented in
this paper catalyze further research into recommendation approaches that go
beyond accuracy in their pursuit of enhanced user satisfaction.
  The datasets and the code are made publicly available at
https://github.com/ramituncc49er/ATARS .

### 6. [Verify-in-the-Graph: Entity Disambiguation Enhancement for Complex Claim Verification with Interactive Graph Representation](http://arxiv.org/pdf/2505.22993v1)

Authors: Hoang Pham, Thanh-Do Nguyen, Khac-Hoai Nam Bui

Claim verification is a long-standing and challenging task that demands not
only high accuracy but also explainability of the verification process. This
task becomes an emerging research issue in the era of large language models
(LLMs) since real-world claims are often complex, featuring intricate semantic
structures or obfuscated entities. Traditional approaches typically address
this by decomposing claims into sub-claims and querying a knowledge base to
resolve hidden or ambiguous entities. However, the absence of effective
disambiguation strategies for these entities can compromise the entire
verification process. To address these challenges, we propose
Verify-in-the-Graph (VeGraph), a novel framework leveraging the reasoning and
comprehension abilities of LLM agents. VeGraph operates in three phases: (1)
Graph Representation - an input claim is decomposed into structured triplets,
forming a graph-based representation that integrates both structured and
unstructured information; (2) Entity Disambiguation -VeGraph iteratively
interacts with the knowledge base to resolve ambiguous entities within the
graph for deeper sub-claim verification; and (3) Verification - remaining
triplets are verified to complete the fact-checking process. Experiments using
Meta-Llama-3-70B (instruct version) show that VeGraph achieves competitive
performance compared to baselines on two benchmarks HoVer and FEVEROUS,
effectively addressing claim verification challenges. Our source code and data
are available for further exploitation.

### 7. [Bounded-Abstention Pairwise Learning to Rank](http://arxiv.org/pdf/2505.23437v1)

Authors: Antonio Ferrara, Andrea Pugnana, Francesco Bonchi, Salvatore Ruggieri

Ranking systems influence decision-making in high-stakes domains like health,
education, and employment, where they can have substantial economic and social
impacts. This makes the integration of safety mechanisms essential. One such
mechanism is $\textit{abstention}$, which enables algorithmic decision-making
system to defer uncertain or low-confidence decisions to human experts. While
abstention have been predominantly explored in the context of classification
tasks, its application to other machine learning paradigms remains
underexplored. In this paper, we introduce a novel method for abstention in
pairwise learning-to-rank tasks. Our approach is based on thresholding the
ranker's conditional risk: the system abstains from making a decision when the
estimated risk exceeds a predefined threshold. Our contributions are threefold:
a theoretical characterization of the optimal abstention strategy, a
model-agnostic, plug-in algorithm for constructing abstaining ranking models,
and a comprehensive empirical evaluations across multiple datasets,
demonstrating the effectiveness of our approach.

### Machine Learning

### 1. [Directed Graph Grammars for Sequence-based Learning](http://arxiv.org/pdf/2505.22949v1)

Authors: Michael Sun, Orion Foo, Gang Liu, Wojciech Matusik, Jie Chen

Directed acyclic graphs (DAGs) are a class of graphs commonly used in
practice, with examples that include electronic circuits, Bayesian networks,
and neural architectures. While many effective encoders exist for DAGs, it
remains challenging to decode them in a principled manner, because the nodes of
a DAG can have many different topological orders. In this work, we propose a
grammar-based approach to constructing a principled, compact and equivalent
sequential representation of a DAG. Specifically, we view a graph as
derivations over an unambiguous grammar, where the DAG corresponds to a unique
sequence of production rules. Equivalently, the procedure to construct such a
description can be viewed as a lossless compression of the data. Such a
representation has many uses, including building a generative model for graph
generation, learning a latent space for property prediction, and leveraging the
sequence representational continuity for Bayesian Optimization over structured
data. Code is available at https://github.com/shiningsunnyday/induction.

### 2. [LLM Agents for Bargaining with Utility-based Feedback](http://arxiv.org/pdf/2505.22998v1)

Authors: Jihwan Oh, Murad Aghazada, Se-Young Yun, Taehyeon Kim

Bargaining, a critical aspect of real-world interactions, presents challenges
for large language models (LLMs) due to limitations in strategic depth and
adaptation to complex human factors. Existing benchmarks often fail to capture
this real-world complexity. To address this and enhance LLM capabilities in
realistic bargaining, we introduce a comprehensive framework centered on
utility-based feedback. Our contributions are threefold: (1) BargainArena, a
novel benchmark dataset with six intricate scenarios (e.g., deceptive
practices, monopolies) to facilitate diverse strategy modeling; (2)
human-aligned, economically-grounded evaluation metrics inspired by utility
theory, incorporating agent utility and negotiation power, which implicitly
reflect and promote opponent-aware reasoning (OAR); and (3) a structured
feedback mechanism enabling LLMs to iteratively refine their bargaining
strategies. This mechanism can positively collaborate with in-context learning
(ICL) prompts, including those explicitly designed to foster OAR. Experimental
results show that LLMs often exhibit negotiation strategies misaligned with
human preferences, and that our structured feedback mechanism significantly
improves their performance, yielding deeper strategic and opponent-aware
reasoning.

### 3. [QLIP: A Dynamic Quadtree Vision Prior Enhances MLLM Performance Without Retraining](http://arxiv.org/pdf/2505.23004v1)

Authors: Kyle R. Chickering, Bangzheng Li, Muhao Chen

Multimodal Large Language Models (MLLMs) encode images into visual tokens,
aligning visual and textual signals within a shared latent space to facilitate
crossmodal representation learning. The CLIP model is a widely adopted
foundational vision language model whose vision encoder has played a critical
role in the development of MLLMs such as LLaVA. However, the CLIP vision
encoder suffers from notable limitations including being constrained to only
handling fixed input resolutions and a failure to produce separated embeddings
for dissimilar images. Replacing the vision encoder of an existing model
typically incurs substantial computational costs because such a change often
necessitates retraining the entire model pipeline.
  In this work, we identify two factors which underlie the limitations of the
CLIP vision encoder: mesoscopic bias and interpolation bias. To address these
issues, we propose QLIP, a drop-in replacement for CLIP that can be seamlessly
integrated with existing MLLMs with only a few lines of code and can enhance
both coarse-grained and fine-grained visual understanding, without re-training.
QLIP is designed around an image quadtree which replaces the standard uniform
grid patches with a novel content aware patchification. Our experimental
results demonstrate that QLIP improves the general visual question answering
accuracy of the LLaVA v1.5 model series across various model sizes--without
requiring retraining or fine-tuning of the full MLLM. Notably, QLIP boosts
detailed understanding performance on the challenging $V^{\ast}$ benchmark by
up to 13.6 percent.

### 4. [Scalable Complexity Control Facilitates Reasoning Ability of LLMs](http://arxiv.org/pdf/2505.23013v1)

Authors: Liangkai Hang, Junjie Yao, Zhiwei Bai, Tianyi Chen, Yang Chen, Rongjie Diao, Hezhou Li, Pengxiao Lin, Zhiwei Wang, Cheng Xu, Zhongwang Zhang, Zhangchen Zhou, Zhiyu Li, Zehao Lin, Kai Chen, Feiyu Xiong, Yaoyu Zhang, Weinan E, Hongkang Yang, Zhi-Qin John Xu

The reasoning ability of large language models (LLMs) has been rapidly
advancing in recent years, attracting interest in more fundamental approaches
that can reliably enhance their generalizability. This work demonstrates that
model complexity control, conveniently implementable by adjusting the
initialization rate and weight decay coefficient, improves the scaling law of
LLMs consistently over varying model sizes and data sizes. This gain is further
illustrated by comparing the benchmark performance of 2.4B models pretrained on
1T tokens with different complexity hyperparameters. Instead of fixing the
initialization std, we found that a constant initialization rate (the exponent
of std) enables the scaling law to descend faster in both model and data sizes.
These results indicate that complexity control is a promising direction for the
continual advancement of LLMs.

### 5. [Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](http://arxiv.org/pdf/2505.23014v1)

Authors: Juwei Yue, Haikuo Li, Jiawei Sheng, Xiaodong Li, Taoyu Su, Tingwen Liu, Li Guo

Graph neural networks (GNNs) leverage message passing mechanisms to learn the
topological features of graph data. Traditional GNNs learns node features in a
spatial domain unrelated to the topology, which can hardly ensure topological
features. In this paper, we formulates message passing as a system of
hyperbolic partial differential equations (hyperbolic PDEs), constituting a
dynamical system that explicitly maps node representations into a particular
solution space. This solution space is spanned by a set of eigenvectors
describing the topological structure of graphs. Within this system, for any
moment in time, a node features can be decomposed into a superposition of the
basis of eigenvectors. This not only enhances the interpretability of message
passing but also enables the explicit extraction of fundamental characteristics
about the topological structure. Furthermore, by solving this system of
hyperbolic partial differential equations, we establish a connection with
spectral graph neural networks (spectral GNNs), serving as a message passing
enhancement paradigm for spectral GNNs.We further introduce polynomials to
approximate arbitrary filter functions. Extensive experiments demonstrate that
the paradigm of hyperbolic PDEs not only exhibits strong flexibility but also
significantly enhances the performance of various spectral GNNs across diverse
graph tasks.

### 6. [SCORPIO: Serving the Right Requests at the Right Time for Heterogeneous SLOs in LLM Inference](http://arxiv.org/pdf/2505.23022v1)

Authors: Yinghao Tang, Tingfeng Lan, Xiuqi Huang, Hui Lu, Wei Chen

Existing Large Language Model (LLM) serving systems prioritize maximum
throughput. They often neglect Service Level Objectives (SLOs) such as Time to
First Token (TTFT) and Time Per Output Token (TPOT), which leads to suboptimal
SLO attainment. This paper introduces SCORPIO, an SLO-oriented LLM serving
system designed to maximize system goodput and SLO attainment for workloads
with heterogeneous SLOs. Our core insight is to exploit SLO heterogeneity for
adaptive scheduling across admission control, queue management, and batch
selection. SCORPIO features a TTFT Guard, which employs least-deadline-first
reordering and rejects unattainable requests, and a TPOT Guard, which utilizes
a VBS-based admission control and a novel credit-based batching mechanism. Both
guards are supported by a predictive module. Evaluations demonstrate that
SCORPIO improves system goodput by up to 14.4X and SLO adherence by up to 46.5%
compared to state-of-the-art baselines.

### 7. [An Empirical Study of Federated Prompt Learning for Vision Language Model](http://arxiv.org/pdf/2505.23024v1)

Authors: Zhihao Wang, Wenke Huang, Tian Chen, Zekun Shi, Guancheng Wan, Yu Qiao, Bin Yang, Jian Wang, Bing Li, Mang Ye

The Vision Language Model (VLM) excels in aligning vision and language
representations, and prompt learning has emerged as a key technique for
adapting such models to downstream tasks. However, the application of prompt
learning with VLM in federated learning (\fl{}) scenarios remains
underexplored. This paper systematically investigates the behavioral
differences between language prompt learning (LPT) and vision prompt learning
(VPT) under data heterogeneity challenges, including label skew and domain
shift. We conduct extensive experiments to evaluate the impact of various \fl{}
and prompt configurations, such as client scale, aggregation strategies, and
prompt length, to assess the robustness of Federated Prompt Learning (FPL).
Furthermore, we explore strategies for enhancing prompt learning in complex
scenarios where label skew and domain shift coexist, including leveraging both
prompt types when computational resources allow. Our findings offer practical
insights into optimizing prompt learning in federated settings, contributing to
the broader deployment of VLMs in privacy-preserving environments.

### 8. [ProDiff: Prototype-Guided Diffusion for Minimal Information Trajectory Imputation](http://arxiv.org/pdf/2505.23048v1)

Authors: Tianci Bu, Le Zhou, Wenchuan Yang, Jianhong Mou, Kang Yang, Suoyi Tan, Feng Yao, Jingyuan Wang, Xin Lu

Trajectory data is crucial for various applications but often suffers from
incompleteness due to device limitations and diverse collection scenarios.
Existing imputation methods rely on sparse trajectory or travel information,
such as velocity, to infer missing points. However, these approaches assume
that sparse trajectories retain essential behavioral patterns, which place
significant demands on data acquisition and overlook the potential of
large-scale human trajectory embeddings. To address this, we propose ProDiff, a
trajectory imputation framework that uses only two endpoints as minimal
information. It integrates prototype learning to embed human movement patterns
and a denoising diffusion probabilistic model for robust spatiotemporal
reconstruction. Joint training with a tailored loss function ensures effective
imputation. ProDiff outperforms state-of-the-art methods, improving accuracy by
6.28\% on FourSquare and 2.52\% on WuXi. Further analysis shows a 0.927
correlation between generated and real trajectories, demonstrating the
effectiveness of our approach.

### 9. [CDR-Agent: Intelligent Selection and Execution of Clinical Decision Rules Using Large Language Model Agents](http://arxiv.org/pdf/2505.23055v1)

Authors: Zhen Xiang, Aliyah R. Hsu, Austin V. Zane, Aaron E. Kornblith, Margaret J. Lin-Martore, Jasmanpreet C. Kaur, Vasuda M. Dokiparthi, Bo Li, Bin Yu

Clinical decision-making is inherently complex and fast-paced, particularly
in emergency departments (EDs) where critical, rapid and high-stakes decisions
are made. Clinical Decision Rules (CDRs) are standardized evidence-based tools
that combine signs, symptoms, and clinical variables into decision trees to
make consistent and accurate diagnoses. CDR usage is often hindered by the
clinician's cognitive load, limiting their ability to quickly recall and apply
the appropriate rules. We introduce CDR-Agent, a novel LLM-based system
designed to enhance ED decision-making by autonomously identifying and applying
the most appropriate CDRs based on unstructured clinical notes. To validate
CDR-Agent, we curated two novel ED datasets: synthetic and CDR-Bench, although
CDR-Agent is applicable to non ED clinics. CDR-Agent achieves a 56.3\%
(synthetic) and 8.7\% (CDR-Bench) accuracy gain relative to the standalone LLM
baseline in CDR selection. Moreover, CDR-Agent significantly reduces
computational overhead. Using these datasets, we demonstrated that CDR-Agent
not only selects relevant CDRs efficiently, but makes cautious yet effective
imaging decisions by minimizing unnecessary interventions while successfully
identifying most positively diagnosed cases, outperforming traditional LLM
prompting approaches. Code for our work can be found at:
https://github.com/zhenxianglance/medagent-cdr-agent

### 10. [Loss-Guided Model Sharing and Local Learning Correction in Decentralized Federated Learning for Crop Disease Classification](http://arxiv.org/pdf/2505.23063v1)

Authors: Denis Mamba Kabala, Adel Hafiane, Laurent Bobelin, Raphael Canals

Crop disease detection and classification is a critical challenge in
agriculture, with major implications for productivity, food security, and
environmental sustainability. While deep learning models such as CNN and ViT
have shown excellent performance in classifying plant diseases from images,
their large-scale deployment is often limited by data privacy concerns.
Federated Learning (FL) addresses this issue, but centralized FL remains
vulnerable to single-point failures and scalability limits. In this paper, we
introduce a novel Decentralized Federated Learning (DFL) framework that uses
validation loss (Loss_val) both to guide model sharing between peers and to
correct local training via an adaptive loss function controlled by weighting
parameter. We conduct extensive experiments using PlantVillage datasets with
three deep learning architectures (ResNet50, VGG16, and ViT_B16), analyzing the
impact of weighting parameter, the number of shared models, the number of
clients, and the use of Loss_val versus Loss_train of other clients. Results
demonstrate that our DFL approach not only improves accuracy and convergence
speed, but also ensures better generalization and robustness across
heterogeneous data environments making it particularly well-suited for
privacy-preserving agricultural applications.

### Neural and Evolutionary Computing

### 1. [Walking the Weight Manifold: a Topological Approach to Conditioning Inspired by Neuromodulation](http://arxiv.org/pdf/2505.22994v1)

Authors: Ari S. Benjamin, Kyle Daruwalla, Christian Pehle, Anthony M. Zador

One frequently wishes to learn a range of similar tasks as efficiently as
possible, re-using knowledge across tasks. In artificial neural networks, this
is typically accomplished by conditioning a network upon task context by
injecting context as input. Brains have a different strategy: the parameters
themselves are modulated as a function of various neuromodulators such as
serotonin. Here, we take inspiration from neuromodulation and propose to learn
weights which are smoothly parameterized functions of task context variables.
Rather than optimize a weight vector, i.e. a single point in weight space, we
optimize a smooth manifold in weight space with a predefined topology. To
accomplish this, we derive a formal treatment of optimization of manifolds as
the minimization of a loss functional subject to a constraint on volumetric
movement, analogous to gradient descent. During inference, conditioning selects
a single point on this manifold which serves as the effective weight matrix for
a particular sub-task. This strategy for conditioning has two main advantages.
First, the topology of the manifold (whether a line, circle, or torus) is a
convenient lever for inductive biases about the relationship between tasks.
Second, learning in one state smoothly affects the entire manifold, encouraging
generalization across states. To verify this, we train manifolds with several
topologies, including straight lines in weight space (for conditioning on e.g.
noise level in input data) and ellipses (for rotated images). Despite their
simplicity, these parameterizations outperform conditioning identical networks
by input concatenation and better generalize to out-of-distribution samples.
These results suggest that modulating weights over low-dimensional manifolds
offers a principled and effective alternative to traditional conditioning.

### 2. [Comparative of Genetic Fuzzy regression techniques for aeroacoustic phenomenons](http://arxiv.org/pdf/2505.23746v1)

Authors: Hugo Henry, Kelly Cohen

This study investigates the application of Genetic Fuzzy Systems (GFS) to
model the self-noise generated by airfoils, a key issue in aeroaccoustics with
significant implications for aerospace, automotive and drone applications.
Using the publicly available Airfoil Self Noise dataset, various Fuzzy
regression strategies are explored and compared. The paper evaluates a brute
force Takagi Sugeno Kang (TSK) fuzzy system with high rule density, a cascading
Geneti Fuzzy Tree (GFT) architecture and a novel clustered approach based on
Fuzzy C-means (FCM) to reduce the model's complexity. This highlights the
viability of clustering assisted fuzzy inference as an effective regression
tool for complex aero accoustic phenomena. Keywords : Fuzzy logic, Regression,
Cascading systems, Clustering and AI.

### Networking and Internet Architecture

### 1. [Agile Orchestration at Will: An Entire Smart Service-Based Security Architecture Towards 6G](http://arxiv.org/pdf/2505.22963v1)

Authors: Zhuoran Duan, Guoshun Nan, Rushan Li, Zijun Wang, Lihua Xiong, Chaoying Yuan, Guorong Liu, Hui Xu, Qimei Cui, Xiaofeng Tao, Tony Q. S. Quek

The upcoming 6G will fundamentally reshape mobile networks beyond
communications, unlocking a multitude of applications that were once considered
unimaginable. Meanwhile, security and resilience are especially highlighted in
the 6G design principles. However, safeguarding 6G networks will be quite
challenging due to various known and unknown threats from highly heterogeneous
networks and diversified security requirements of distinct use cases, calling
for a comprehensive re-design of security architecture. This motivates us to
propose ES3A (Entire Smart Service-based Security Architecture), a novel
security architecture for 6G networks. Specifically, we first discuss six
high-level principles of our ES3A that include hierarchy, flexibility,
scalability, resilience, endogeny, and trust and privacy. With these goals in
mind, we then introduce three guidelines from a deployment perspective,
envisioning our ES3A that offers service-based security, end-to-end protection,
and smart security automation for 6G networks. Our architecture consists of
three layers and three domains. It relies on a two-stage orchestration
mechanism to tailor smart security strategies for customized protection in
high-dynamic 6G networks, thereby addressing the aforementioned challenges.
Finally, we prototype the proposed ES3A on a real-world radio system based on
Software-Defined Radio (SDR). Experiments show the effectiveness of our ES3A.
We also provide a case to show the superiority of our architecture.

### 2. [Context-Aware Semantic Communication for the Wireless Networks](http://arxiv.org/pdf/2505.23249v1)

Authors: Guangyuan Liu, Yinqiu Liu, Jiacheng Wang, Hongyang Du, Dusit Niyato, Jiawen Kang, Zehui Xiong, Abbas Jamalipour

In next-generation wireless networks, supporting real-time applications such
as augmented reality, autonomous driving, and immersive Metaverse services
demands stringent constraints on bandwidth, latency, and reliability. Existing
semantic communication (SemCom) approaches typically rely on static models,
overlooking dynamic conditions and contextual cues vital for efficient
transmission. To address these challenges, we propose CaSemCom, a context-aware
SemCom framework that leverages a Large Language Model (LLM)-based gating
mechanism and a Mixture of Experts (MoE) architecture to adaptively select and
encode only high-impact semantic features across multiple data modalities. Our
multimodal, multi-user case study demonstrates that CaSemCom significantly
improves reconstructed image fidelity while reducing bandwidth usage,
outperforming single-agent deep reinforcement learning (DRL) methods and
traditional baselines in convergence speed, semantic accuracy, and
retransmission overhead.

### 3. [Wireless Agentic AI with Retrieval-Augmented Multimodal Semantic Perception](http://arxiv.org/pdf/2505.23275v1)

Authors: Guangyuan Liu, Yinqiu Liu, Ruichen Zhang, Hongyang Du, Dusit Niyato, Zehui Xiong, Sumei Sun, Abbas Jamalipour

The rapid development of multimodal AI and Large Language Models (LLMs) has
greatly enhanced real-time interaction, decision-making, and collaborative
tasks. However, in wireless multi-agent scenarios, limited bandwidth poses
significant challenges to exchanging semantically rich multimodal information
efficiently. Traditional semantic communication methods, though effective,
struggle with redundancy and loss of crucial details. To overcome these
challenges, we propose a Retrieval-Augmented Multimodal Semantic Communication
(RAMSemCom) framework. RAMSemCom incorporates iterative, retrieval-driven
semantic refinement tailored for distributed multi-agent environments, enabling
efficient exchange of critical multimodal elements through local caching and
selective transmission. Our approach dynamically optimizes retrieval using deep
reinforcement learning (DRL) to balance semantic fidelity with bandwidth
constraints. A comprehensive case study on multi-agent autonomous driving
demonstrates that our DRL-based retrieval strategy significantly improves task
completion efficiency and reduces communication overhead compared to baseline
methods.

### 4. [LUMION: Fast Fault Recovery for ML Jobs Using Programmable Optical Fabrics](http://arxiv.org/pdf/2505.23105v1)

Authors: Abhishek Vijaya Kumar, Eric Ding, Arjun Devraj, Darius Bunandar, Rachee Singh

When accelerators fail in modern ML datacenters, operators migrate the
affected ML training or inference jobs to entirely new racks. This approach,
while preserving network performance, is highly inefficient, requiring
datacenters to reserve full racks of idle accelerators for fault tolerance. In
this paper, we address this resource inefficiency by introducing LUMION, a
novel reconfigurable optical fabric for connecting accelerators within a
datacenter rack. Instead of migrating entire ML jobs, LUMION dynamically
integrates spare accelerators into ongoing workloads as failures occur, thereby
maintaining consistent performance without costly migrations. We show the
benefits of LUMION by building an end-to-end hardware prototype. Our
experiments fine-tune Llama 3.2 and show that LUMION swaps a failed GPU with a
healthy one and restarts the ML job within ~ 1 second of the failure. LUMION
achieves higher inter-GPU bandwidth compared to traditional electrical racks
after replacing failed accelerators with spare ones, leading to nearly 2X
improvement in fine-tuning throughput.

### 5. [Quantum Hilbert Transform](http://arxiv.org/pdf/2505.23581v1)

Authors: Nitin Jha, Abhishek Parakh

The Hilbert transform has been one of the foundational transforms in signal
processing, finding it's way into multiple disciplines from cryptography to
biomedical sciences. However, there does not exist any quantum analogue for the
Hilbert transform. In this work, we introduce a formulation for the quantum
Hilbert transform (QHT)and apply it to a quantum steganography protocol. By
bridging classical phase-shift techniques with quantum operations, QHT opens
new pathways in quantum signal processing, communications, sensing, and secure
information hiding.

### 6. [Towards A Global Quantum Internet: A Review of Challenges Facing Aerial Quantum Networks](http://arxiv.org/pdf/2505.23603v1)

Authors: Nitin Jha, Abhishek Parakh

Quantum networks use principles of quantum physics to create secure
communication networks. Moving these networks off the ground using drones,
balloons, or satellites could help increase the scalability of these networks.
This article reviews how such aerial links work, what makes them difficult to
build, and the possible solutions that can be used to overcome these problems.
By combining ground stations, aerial relays, and orbiting satellites into one
seamless system, we move closer to a practical quantum internet.

### 7. [Distributed Federated Learning for Vehicular Network Security: Anomaly Detection Benefits and Multi-Domain Attack Threats](http://arxiv.org/pdf/2505.23706v1)

Authors: Utku Demir, Yalin E. Sagduyu, Tugba Erpek, Hossein Jafari, Sastry Kompella, Mengran Xue

In connected and autonomous vehicles, machine learning for safety message
classification has become critical for detecting malicious or anomalous
behavior. However, conventional approaches that rely on centralized data
collection or purely local training face limitations due to the large scale,
high mobility, and heterogeneous data distributions inherent in inter-vehicle
networks. To overcome these challenges, this paper explores Distributed
Federated Learning (DFL), whereby vehicles collaboratively train deep learning
models by exchanging model updates among one-hop neighbors and propagating
models over multiple hops. Using the Vehicular Reference Misbehavior (VeReMi)
Extension Dataset, we show that DFL can significantly improve classification
accuracy across all vehicles compared to learning strictly with local data.
Notably, vehicles with low individual accuracy see substantial accuracy gains
through DFL, illustrating the benefit of knowledge sharing across the network.
We further show that local training data size and time-varying network
connectivity correlate strongly with the model's overall accuracy. We
investigate DFL's resilience and vulnerabilities under attacks in multiple
domains, namely wireless jamming and training data poisoning attacks. Our
results reveal important insights into the vulnerabilities of DFL when
confronted with multi-domain attacks, underlining the need for more robust
strategies to secure DFL in vehicular networks.

### Robotics

### 1. [Stairway to Success: Zero-Shot Floor-Aware Object-Goal Navigation via LLM-Driven Coarse-to-Fine Exploration](http://arxiv.org/pdf/2505.23019v1)

Authors: Zeying Gong, Rong Li, Tianshuai Hu, Ronghe Qiu, Lingdong Kong, Lingfeng Zhang, Yiyi Ding, Leying Zhang, Junwei Liang

Object-Goal Navigation (OGN) remains challenging in real-world, multi-floor
environments and under open-vocabulary object descriptions. We observe that
most episodes in widely used benchmarks such as HM3D and MP3D involve
multi-floor buildings, with many requiring explicit floor transitions. However,
existing methods are often limited to single-floor settings or predefined
object categories. To address these limitations, we tackle two key challenges:
(1) efficient cross-level planning and (2) zero-shot object-goal navigation
(ZS-OGN), where agents must interpret novel object descriptions without prior
exposure. We propose ASCENT, a framework that combines a Multi-Floor Spatial
Abstraction module for hierarchical semantic mapping and a Coarse-to-Fine
Frontier Reasoning module leveraging Large Language Models (LLMs) for
context-aware exploration, without requiring additional training on new object
semantics or locomotion data. Our method outperforms state-of-the-art ZS-OGN
approaches on HM3D and MP3D benchmarks while enabling efficient multi-floor
navigation. We further validate its practicality through real-world deployment
on a quadruped robot, achieving successful object exploration across unseen
floors.

### 2. [Redundancy Parameterization of the ABB YuMi Robot Arm](http://arxiv.org/pdf/2505.23111v1)

Authors: Alexander J. Elias, John T. Wen

The ABB YuMi is a 7-DOF collaborative robot arm with a complex, redundant
kinematic structure. Path planning for the YuMi is challenging, especially with
joint limits considered. The redundant degree of freedom is parameterized by
the Shoulder-Elbow-Wrist (SEW) angle, called the arm angle by ABB, but the
exact definition must be known for path planning outside the RobotStudio
simulator. We provide the first complete and validated definition of the SEW
angle used for the YuMi. It follows the conventional SEW angle formulation with
the shoulder-elbow direction chosen to be the direction of the fourth joint
axis. Our definition also specifies the shoulder location, making it compatible
with any choice of reference vector. A previous attempt to define the SEW angle
exists in the literature, but it is incomplete and deviates from the behavior
observed in RobotStudio. Because our formulation fits within the general SEW
angle framework, we also obtain the expression for the SEW angle Jacobian and
complete numerical conditions for all algorithmic singularities. Finally, we
demonstrate using IK-Geo, our inverse kinematics (IK) solver based on
subproblem decomposition, to find all IK solutions using 2D search. Code
examples are available in a publicly accessible repository.

### 3. [LocoTouch: Learning Dexterous Quadrupedal Transport with Tactile Sensing](http://arxiv.org/pdf/2505.23175v1)

Authors: Changyi Lin, Yuxin Ray Song, Boda Huo, Mingyang Yu, Yikai Wang, Shiqi Liu, Yuxiang Yang, Wenhao Yu, Tingnan Zhang, Jie Tan, Yiyue Luo, Ding Zhao

Quadrupedal robots have demonstrated remarkable agility and robustness in
traversing complex terrains. However, they remain limited in performing object
interactions that require sustained contact. In this work, we present
LocoTouch, a system that equips quadrupedal robots with tactile sensing to
address a challenging task in this category: long-distance transport of
unsecured cylindrical objects, which typically requires custom mounting
mechanisms to maintain stability. For efficient large-area tactile sensing, we
design a high-density distributed tactile sensor array that covers the entire
back of the robot. To effectively leverage tactile feedback for locomotion
control, we develop a simulation environment with high-fidelity tactile
signals, and train tactile-aware transport policies using a two-stage learning
pipeline. Furthermore, we design a novel reward function to promote stable,
symmetric, and frequency-adaptive locomotion gaits. After training in
simulation, LocoTouch transfers zero-shot to the real world, reliably balancing
and transporting a wide range of unsecured, cylindrical everyday objects with
broadly varying sizes and weights. Thanks to the responsiveness of the tactile
sensor and the adaptive gait reward, LocoTouch can robustly balance objects
with slippery surfaces over long distances, or even under severe external
perturbations.

### 4. [UPP: Unified Path Planner with Adaptive Safety and Optimality](http://arxiv.org/pdf/2505.23197v1)

Authors: Jatin Kumar Arora, Shubhendu Bhasin

We are surrounded by robots helping us perform complex tasks. Robots have a
wide range of applications, from industrial automation to personalized
assistance. However, with great technological innovation come significant
challenges. One of the major challenges in robotics is path planning. Despite
advancements such as graph search, sampling, and potential field methods, most
path planning algorithms focus either on optimality or on safety. Very little
research addresses both simultaneously. We propose a Unified Path Planner (UPP)
that uses modified heuristics and a dynamic safety cost function to balance
safety and optimality. The level of safety can be adjusted via tunable
parameters, trading off against computational complexity. We demonstrate the
planner's performance in simulations, showing how parameter variation affects
results. UPP is compared with various traditional and safe-optimal planning
algorithms across different scenarios. We also validate it on a TurtleBot,
where the robot successfully finds safe and sub-optimal paths.

### 5. [MEF-Explore: Communication-Constrained Multi-Robot Entropy-Field-Based Exploration](http://arxiv.org/pdf/2505.23376v1)

Authors: Khattiya Pongsirijinda, Zhiqiang Cao, Billy Pik Lik Lau, Ran Liu, Chau Yuen, U-Xuan Tan

Collaborative multiple robots for unknown environment exploration have become
mainstream due to their remarkable performance and efficiency. However, most
existing methods assume perfect robots' communication during exploration, which
is unattainable in real-world settings. Though there have been recent works
aiming to tackle communication-constrained situations, substantial room for
advancement remains for both information-sharing and exploration strategy
aspects. In this paper, we propose a Communication-Constrained Multi-Robot
Entropy-Field-Based Exploration (MEF-Explore). The first module of the proposed
method is the two-layer inter-robot communication-aware information-sharing
strategy. A dynamic graph is used to represent a multi-robot network and to
determine communication based on whether it is low-speed or high-speed.
Specifically, low-speed communication, which is always accessible between every
robot, can only be used to share their current positions. If robots are within
a certain range, high-speed communication will be available for inter-robot map
merging. The second module is the entropy-field-based exploration strategy.
Particularly, robots explore the unknown area distributedly according to the
novel forms constructed to evaluate the entropies of frontiers and robots.
These entropies can also trigger implicit robot rendezvous to enhance
inter-robot map merging if feasible. In addition, we include the
duration-adaptive goal-assigning module to manage robots' goal assignment. The
simulation results demonstrate that our MEF-Explore surpasses the existing ones
regarding exploration time and success rate in all scenarios. For real-world
experiments, our method leads to a 21.32% faster exploration time and a 16.67%
higher success rate compared to the baseline.

### 6. [Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents](http://arxiv.org/pdf/2505.23450v1)

Authors: Zhejian Yang, Yongchao Chen, Xueyang Zhou, Jiangyue Yan, Dingjie Song, Yinuo Liu, Yuting Li, Yu Zhang, Pan Zhou, Hechang Chen, Lichao Sun

Long-horizon robotic manipulation poses significant challenges for autonomous
systems, requiring extended reasoning, precise execution, and robust error
recovery across complex sequential tasks. Current approaches, whether based on
static planning or end-to-end visuomotor policies, suffer from error
accumulation and lack effective verification mechanisms during execution,
limiting their reliability in real-world scenarios. We present Agentic Robot, a
brain-inspired framework that addresses these limitations through Standardized
Action Procedures (SAP)--a novel coordination protocol governing component
interactions throughout manipulation tasks. Drawing inspiration from
Standardized Operating Procedures (SOPs) in human organizations, SAP
establishes structured workflows for planning, execution, and verification
phases. Our architecture comprises three specialized components: (1) a large
reasoning model that decomposes high-level instructions into semantically
coherent subgoals, (2) a vision-language-action executor that generates
continuous control commands from real-time visual inputs, and (3) a temporal
verifier that enables autonomous progression and error recovery through
introspective assessment. This SAP-driven closed-loop design supports dynamic
self-verification without external supervision. On the LIBERO benchmark,
Agentic Robot achieves state-of-the-art performance with an average success
rate of 79.6\%, outperforming SpatialVLA by 6.1\% and OpenVLA by 7.4\% on
long-horizon tasks. These results demonstrate that SAP-driven coordination
between specialized components enhances both performance and interpretability
in sequential manipulation, suggesting significant potential for reliable
autonomous systems. Project Github: https://agentic-robot.github.io.

### 7. [Centroidal Trajectory Generation and Stabilization based on Preview Control for Humanoid Multi-contact Motion](http://arxiv.org/pdf/2505.23499v1)

Authors: Masaki Murooka, Mitsuharu Morisawa, Fumio Kanehiro

Multi-contact motion is important for humanoid robots to work in various
environments. We propose a centroidal online trajectory generation and
stabilization control for humanoid dynamic multi-contact motion. The proposed
method features the drastic reduction of the computational cost by using
preview control instead of the conventional model predictive control that
considers the constraints of all sample times. By combining preview control
with centroidal state feedback for robustness to disturbances and wrench
distribution for satisfying contact constraints, we show that the robot can
stably perform a variety of multi-contact motions through simulation
experiments.

### 8. [Optimization-based Posture Generation for Whole-body Contact Motion by Contact Point Search on the Body Surface](http://arxiv.org/pdf/2505.23501v1)

Authors: Masaki Murooka, Kei Okada, Masayuki Inaba

Whole-body contact is an effective strategy for improving the stability and
efficiency of the motion of robots. For robots to automatically perform such
motions, we propose a posture generation method that employs all available
surfaces of the robot links. By representing the contact point on the body
surface by two-dimensional configuration variables, the joint positions and
contact points are simultaneously determined through a gradient-based
optimization. By generating motions with the proposed method, we present
experiments in which robots manipulate objects effectively utilizing whole-body
contact.

### 9. [Humanoid Loco-manipulation Planning based on Graph Search and Reachability Maps](http://arxiv.org/pdf/2505.23505v1)

Authors: Masaki Murooka, Iori Kumagai, Mitsuharu Morisawa, Fumio Kanehiro, Abderrahmane Kheddar

In this letter, we propose an efficient and highly versatile
loco-manipulation planning for humanoid robots. Loco-manipulation planning is a
key technological brick enabling humanoid robots to autonomously perform object
transportation by manipulating them. We formulate planning of the alternation
and sequencing of footsteps and grasps as a graph search problem with a new
transition model that allows for a flexible representation of
loco-manipulation. Our transition model is quickly evaluated by relocating and
switching the reachability maps depending on the motion of both the robot and
object. We evaluate our approach by applying it to loco-manipulation use-cases,
such as a bobbin rolling operation with regrasping, where the motion is
automatically planned by our framework.

### 10. [Learning coordinated badminton skills for legged manipulators](http://arxiv.org/pdf/2505.22974v1)

Authors: Yuntao Ma, Andrei Cramariuc, Farbod Farshidian, Marco Hutter

Coordinating the motion between lower and upper limbs and aligning limb
control with perception are substantial challenges in robotics, particularly in
dynamic environments. To this end, we introduce an approach for enabling legged
mobile manipulators to play badminton, a task that requires precise
coordination of perception, locomotion, and arm swinging. We propose a unified
reinforcement learning-based control policy for whole-body visuomotor skills
involving all degrees of freedom to achieve effective shuttlecock tracking and
striking. This policy is informed by a perception noise model that utilizes
real-world camera data, allowing for consistent perception error levels between
simulation and deployment and encouraging learned active perception behaviors.
Our method includes a shuttlecock prediction model, constrained reinforcement
learning for robust motion control, and integrated system identification
techniques to enhance deployment readiness. Extensive experimental results in a
variety of environments validate the robot's capability to predict shuttlecock
trajectories, navigate the service area effectively, and execute precise
strikes against human players, demonstrating the feasibility of using legged
mobile manipulators in complex and dynamic sports scenarios.

### Software Engineering

### 1. [What About Emotions? Guiding Fine-Grained Emotion Extraction from Mobile App Reviews](http://arxiv.org/pdf/2505.23452v1)

Authors: Quim Motger, Marc Oriol, Max Tiessler, Xavier Franch, Jordi Marco

Opinion mining plays a vital role in analysing user feedback and extracting
insights from textual data. While most research focuses on sentiment polarity
(e.g., positive, negative, neutral), fine-grained emotion classification in app
reviews remains underexplored. This paper addresses this gap by identifying and
addressing the challenges and limitations in fine-grained emotion analysis in
the context of app reviews. Our study adapts Plutchik's emotion taxonomy to app
reviews by developing a structured annotation framework and dataset. Through an
iterative human annotation process, we define clear annotation guidelines and
document key challenges in emotion classification. Additionally, we evaluate
the feasibility of automating emotion annotation using large language models,
assessing their cost-effectiveness and agreement with human-labelled data. Our
findings reveal that while large language models significantly reduce manual
effort and maintain substantial agreement with human annotators, full
automation remains challenging due to the complexity of emotional
interpretation. This work contributes to opinion mining by providing structured
guidelines, an annotated dataset, and insights for developing automated
pipelines to capture the complexity of emotions in app reviews.

### 2. [Synthesizing Performance Constraints for Evaluating and Improving Code Efficiency](http://arxiv.org/pdf/2505.23471v1)

Authors: Jun Yang, Cheng-Chi Wang, Bogdan Alexandru Stoica, Kexin Pei

Large Language Models (LLMs) have been increasingly used to optimize code
efficiency. Evaluating their effectiveness and further suggesting optimization
opportunities often rely on high-quality tests to demonstrate the performance
bottlenecks presented in the program. However, existing approaches rely on a
limited set of hand-curated inputs or LLM-generated uninteresting
length-stressing tests, failing to reveal more nuanced optimization
opportunities. We present WEDGE, a framework for generating
performance-stressing input given the program under test. WEDGE synthesizes
explicit performance-characterizing constraints in the form of branch
conditions to partition the programs' execution space into performance-specific
regions. When integrated with the coverage-guided fuzzer, reaching different
regions introduces explicit rewards for test generation to explore inefficient
implementations. Our evaluation shows that WEDGE introduces a significant
slowdown compared to the tests in CodeContests and those claimed to be
optimized by existing approaches. From the utility perspective, integrating our
tests substantially improves the existing code optimization approaches that
rely on test-driven execution feedback. We release PERFFORGE, the performance
tests generated by WEDGE, to benchmark future approaches for efficient code
generation at https://github.com/UChiSeclab/perfforge.

### 3. [LLM-based Property-based Test Generation for Guardrailing Cyber-Physical Systems](http://arxiv.org/pdf/2505.23549v1)

Authors: Khashayar Etemadi, Marjan Sirjani, Mahshid Helali Moghadam, Per Strandberg, Paul Pettersson

Cyber-physical systems (CPSs) are complex systems that integrate physical,
computational, and communication subsystems. The heterogeneous nature of these
systems makes their safety assurance challenging. In this paper, we propose a
novel automated approach for guardrailing cyber-physical systems using
property-based tests (PBTs) generated by Large Language Models (LLMs). Our
approach employs an LLM to extract properties from the code and documentation
of CPSs. Next, we use the LLM to generate PBTs that verify the extracted
properties on the CPS. The generated PBTs have two uses. First, they are used
to test the CPS before it is deployed, i.e., at design time. Secondly, these
PBTs can be used after deployment, i.e., at run time, to monitor the behavior
of the system and guardrail it against unsafe states. We implement our approach
in ChekProp and conduct preliminary experiments to evaluate the generated PBTs
in terms of their relevance (how well they match manually crafted properties),
executability (how many run with minimal manual modification), and
effectiveness (coverage of the input space partitions). The results of our
experiments and evaluation demonstrate a promising path forward for creating
guardrails for CPSs using LLM-generated property-based tests.

### 4. [How to Elicit Explainability Requirements? A Comparison of Interviews, Focus Groups, and Surveys](http://arxiv.org/pdf/2505.23684v1)

Authors: Martin Obaidi, Jakob Droste, Hannah Deters, Marc Herrmann, Raymond Ochsner, Jil Klünder, Kurt Schneider

As software systems grow increasingly complex, explainability has become a
crucial non-functional requirement for transparency, user trust, and regulatory
compliance. Eliciting explainability requirements is challenging, as different
methods capture varying levels of detail and structure. This study examines the
efficiency and effectiveness of three commonly used elicitation methods - focus
groups, interviews, and online surveys - while also assessing the role of
taxonomy usage in structuring and improving the elicitation process. We
conducted a case study at a large German IT consulting company, utilizing a
web-based personnel management software. A total of two focus groups, 18
interviews, and an online survey with 188 participants were analyzed. The
results show that interviews were the most efficient, capturing the highest
number of distinct needs per participant per time spent. Surveys collected the
most explanation needs overall but had high redundancy. Delayed taxonomy
introduction resulted in a greater number and diversity of needs, suggesting
that a two-phase approach is beneficial. Based on our findings, we recommend a
hybrid approach combining surveys and interviews to balance efficiency and
coverage. Future research should explore how automation can support elicitation
and how taxonomies can be better integrated into different methods.

### 5. [Structural Abstraction and Selective Refinement for Formal Verification](http://arxiv.org/pdf/2505.22982v1)

Authors: Christoph Luckeneder, Ralph Hoch, Hermann Kaindl

Safety verification of robot applications is extremely challenging due to the
complexity of the environment that a robot typically operates in. Formal
verification with model-checking provides guarantees but it may often take too
long or even fail for complex models of the environment. A usual solution
approach is abstraction, more precisely behavioral abstraction. Our new
approach introduces structural abstraction instead, which we investigated in
the context of voxel representation of the robot environment. This kind of
abstraction leads to abstract voxels. We also propose a complete and automated
verification workflow, which is based on an already existing methodology for
robot applications, and inspired by the key ideas behind counterexample-guided
abstraction refinement (CEGAR) - performing an initial abstraction and
successively introducing refinements based on counterexamples, intertwined with
model-checker runs. Hence, our approach uses selective refinement of structural
abstractions to improve the runtime efficiency of model-checking. A
fully-automated implementation of our approach showed its feasibility, since
counterexamples have been found for a realistic scenario with a fairly high
(maximal) resolution in a few minutes, while direct model-checker runs led to a
crash after a couple of days.

### 6. [Two Is Better Than One: Rotations Scale LoRAs](http://arxiv.org/pdf/2505.23184v1)

Authors: Hongcan Guo, Guoshun Nan, Yuan Yang, Diyang Zhang, Haotian Li, Zhican Chen, Qinchuan Zhou, Yuhan Ran, Xinye Cao, Sicong Leng, Xiaofeng Tao, Xudong Jiang

Scaling Low-Rank Adaptation (LoRA)-based Mixture-of-Experts (MoE) facilitates
large language models (LLMs) to efficiently adapt to diverse tasks. However,
traditional gating mechanisms that route inputs to the best experts may
fundamentally hinder LLMs' scalability, leading to poor generalization and
underfitting issues. We identify that the root cause lies in the restricted
expressiveness of existing weighted-sum mechanisms, both within and outside the
convex cone of LoRA representations. This motivates us to propose RadarGate, a
novel geometrically inspired gating method that introduces rotational
operations of LoRAs representations to boost the expressiveness and facilitate
richer feature interactions among multiple LoRAs for scalable LLMs.
Specifically, we first fuse each LoRA representation to other LoRAs using a
learnable component and then feed the output to a rotation matrix. This matrix
involves learnable parameters that define the relative angular relationship
between LoRA representations. Such a simple yet effective mechanism provides an
extra degree of freedom, facilitating the learning of cross-LoRA synergies and
properly tracking the challenging poor generalization and underfitting issues
as the number of LoRA grows. Extensive experiments on 6 public benchmarks
across 21 tasks show the effectiveness of our RadarGate for scaling LoRAs. We
also provide valuable insights, revealing that the rotations to each pair of
representations are contrastive, encouraging closer alignment of semantically
similar representations during geometrical transformation while pushing
distance ones further apart. We will release our code to the community.

### 7. [OSS-UAgent: An Agent-based Usability Evaluation Framework for Open Source Software](http://arxiv.org/pdf/2505.23239v1)

Authors: Lingkai Meng, Yu Shao, Long Yuan, Longbin Lai, Peng Cheng, Wenyuan Yu, Wenjie Zhang, Xuemin Lin, Jingren Zhou

Usability evaluation is critical to the impact and adoption of open source
software (OSS), yet traditional methods relying on human evaluators suffer from
high costs and limited scalability. To address these limitations, we introduce
OSS-UAgent, an automated, configurable, and interactive agent-based usability
evaluation framework specifically designed for open source software. Our
framework employs intelligent agents powered by large language models (LLMs) to
simulate developers performing programming tasks across various experience
levels (from Junior to Expert). By dynamically constructing platform-specific
knowledge bases, OSS-UAgent ensures accurate and context-aware code generation.
The generated code is automatically evaluated across multiple dimensions,
including compliance, correctness, and readability, providing a comprehensive
measure of the software's usability. Additionally, our demonstration showcases
OSS-UAgent's practical application in evaluating graph analytics platforms,
highlighting its effectiveness in automating usability evaluation.

### 8. [Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization](http://arxiv.org/pdf/2505.23387v1)

Authors: Mingzhe Du, Luu Tuan Tuan, Yue Liu, Yuhao Qing, Dong Huang, Xinyi He, Qian Liu, Zejun Ma, See-kiong Ng

Large Language Models (LLMs) generate functionally correct solutions but
often fall short in code efficiency, a critical bottleneck for real-world
deployment. In this paper, we introduce a novel test-time iterative
optimization framework to address this, employing a closed-loop system where
LLMs iteratively refine code based on empirical performance feedback from an
execution sandbox. We explore three training strategies: Supervised Fine-Tuning
(SFT), Direct Preference Optimization (DPO), and Group Relative Policy
Optimization~(GRPO). Experiments on our Venus dataset and the APPS benchmark
show that SFT and DPO rapidly saturate in efficiency gains. In contrast, GRPO,
using reinforcement learning (RL) with execution feedback, continuously
optimizes code performance, significantly boosting both pass@1 (from 47% to
62%) and the likelihood of outperforming human submissions in efficiency (from
31% to 45%). Our work demonstrates effective test-time code efficiency
improvement and critically reveals the power of RL in teaching LLMs to truly
self-improve code efficiency.

### 9. [Toward Effective AI Governance: A Review of Principles](http://arxiv.org/pdf/2505.23417v1)

Authors: Danilo Ribeiro, Thayssa Rocha, Gustavo Pinto, Bruno Cartaxo, Marcelo Amaral, Nicole Davila, Ana Camargo

Artificial Intelligence (AI) governance is the practice of establishing
frameworks, policies, and procedures to ensure the responsible, ethical, and
safe development and deployment of AI systems. Although AI governance is a core
pillar of Responsible AI, current literature still lacks synthesis across such
governance frameworks and practices. Objective: To identify which frameworks,
principles, mechanisms, and stakeholder roles are emphasized in secondary
literature on AI governance. Method: We conducted a rapid tertiary review of
nine peer-reviewed secondary studies from IEEE and ACM (20202024), using
structured inclusion criteria and thematic semantic synthesis. Results: The
most cited frameworks include the EU AI Act and NIST RMF; transparency and
accountability are the most common principles. Few reviews detail actionable
governance mechanisms or stakeholder strategies. Conclusion: The review
consolidates key directions in AI governance and highlights gaps in empirical
validation and inclusivity. Findings inform both academic inquiry and practical
adoption in organizations.

### 10. [From Knowledge to Noise: CTIM-Rover and the Pitfalls of Episodic Memory in Software Engineering Agents](http://arxiv.org/pdf/2505.23422v1)

Authors: Tobias Lindenbauer, Georg Groh, Hinrich Schütze

We introduce CTIM-Rover, an AI agent for Software Engineering (SE) built on
top of AutoCodeRover (Zhang et al., 2024) that extends agentic reasoning
frameworks with an episodic memory, more specifically, a general and
repository-level Cross-Task-Instance Memory (CTIM). While existing open-source
SE agents mostly rely on ReAct (Yao et al., 2023b), Reflexion (Shinn et al.,
2023), or Code-Act (Wang et al., 2024), all of these reasoning and planning
frameworks inefficiently discard their long-term memory after a single task
instance. As repository-level understanding is pivotal for identifying all
locations requiring a patch for fixing a bug, we hypothesize that SE is
particularly well positioned to benefit from CTIM. For this, we build on the
Experiential Learning (EL) approach ExpeL (Zhao et al., 2024), proposing a
Mixture-Of-Experts (MoEs) inspired approach to create both a general-purpose
and repository-level CTIM. We find that CTIM-Rover does not outperform
AutoCodeRover in any configuration and thus conclude that neither ExpeL nor
DoT-Bank (Lingam et al., 2024) scale to real-world SE problems. Our analysis
indicates noise introduced by distracting CTIM items or exemplar trajectories
as the likely source of the performance degradation.

### Social and Information Networks

### 1. [Offline Map Matching Based on Localization Error Distribution Modeling](http://arxiv.org/pdf/2505.23123v1)

Authors: Ruilin Xu, Yuchen Song, Kaijie Li, Xitong Gao, Kejiang Ye, Fan Zhang, Juanjuan Zhao

Offline map matching involves aligning historical trajectories of mobile
objects, which may have positional errors, with digital maps. This is essential
for applications in intelligent transportation systems (ITS), such as route
analysis and traffic pattern mining. Existing methods have two main
limitations: (i) they assume a uniform Localization Error Distribution (LED)
across urban areas, neglecting environmental factors that lead to suboptimal
path search ranges, and (ii) they struggle to efficiently handle local
non-shortest paths and detours. To address these issues, we propose a novel
offline map matching method for sparse trajectories, called LNSP, which
integrates LED modeling and non-shortest path detection. Key innovations
include: (i) leveraging public transit trajectories with fixed routes to model
LED in finer detail across different city regions, optimizing path search
ranges, and (ii) scoring paths using sub-region dependency LED and a sliding
window, which reduces global map matching errors. Experimental results using
real-world bus and taxi trajectory datasets demonstrate that the LNSP algorithm
significantly outperforms existing methods in both efficiency and matching
accuracy.

### 2. [Homologous nodes in annotated complex networks](http://arxiv.org/pdf/2505.23668v1)

Authors: Sung Soo Moon, Sebastian E. Ahnert

Many real-world networks have associated metadata that assigns categorical
labels to nodes. Analysis of these annotations can complement the topological
analysis of complex networks. Annotated networks have typically been used to
evaluate community detection approaches. Here, we introduce an approach that
combines the quantitative analysis of annotations and network structure, which
groups nodes according to similar distributions of node annotations in their
neighbourhoods. Importantly the nodes that are grouped together, which we call
homologues may not be connected to each other at all. By applying our approach
to three very different real-world networks we show that these groupings
identify common functional roles and properties of nodes in the network.

### 3. [Representing Higher-Order Networks with Spectral Moments](http://arxiv.org/pdf/2505.23691v1)

Authors: Hao Tian, Shengmin Jin, Reza Zafarani

The spectral properties of traditional (dyadic) graphs, where an edge
connects exactly two vertices, are widely studied in different applications.
These spectral properties are closely connected to the structural properties of
dyadic graphs. We generalize such connections and characterize higher-order
networks by their spectral information. We first split the higher-order graphs
by their ``edge orders" into several uniform hypergraphs. For each uniform
hypergraph, we extract the corresponding spectral information from the
transition matrices of carefully designed random walks. From each spectrum, we
compute the first few spectral moments and use all such spectral moments across
different ``edge orders" as the higher-order graph representation. We show that
these moments not only clearly indicate the return probabilities of random
walks but are also closely related to various higher-order network properties
such as degree distribution and clustering coefficient. Extensive experiments
show the utility of this new representation in various settings. For instance,
graph classification on higher-order graphs shows that this representation
significantly outperforms other techniques.

### 4. [Seeing the Politics of Decentralized Social Media Protocols](http://arxiv.org/pdf/2505.22962v1)

Authors: Tolulope Oshinowo, Sohyeon Hwang, Amy X. Zhang, Andrés Monroy-Hernández

Calls to decentralize feed-based social media have been driven by concerns
about the concentrated power of centralized platforms and their societal
impact. In response, numerous decentralized social media protocols have
emerged, each interpreting "decentralization" in different ways. We analyze
four such protocols -- ActivityPub, AT Protocol, Nostr, and Farcaster -- to
develop a novel conceptual framework for understanding how protocols
operationalize decentralization. Drawing from protocol documentation, media
coverage, and first-hand interviews with protocol developers and experts, we
contextualize each protocol's approach within their respective socio-technical
goals. Our framework highlights how control over key components is distributed
differently across each protocol, shaping who holds power over what kinds of
decisions. How components are arranged in relation to one another further
impacts how component owners might offset each other's power in shaping social
media. We argue that examining protocols as artifacts reveals how values shape
infrastructure and power dynamics -- and that with a holistic framework as a
guide, we can more effectively evaluate and design decentralized platforms
aligned with the social and political futures we envision.

### 5. [Evaluating AI capabilities in detecting conspiracy theories on YouTube](http://arxiv.org/pdf/2505.23570v1)

Authors: Leonardo La Rocca, Francesco Corso, Francesco Pierri

As a leading online platform with a vast global audience, YouTube's extensive
reach also makes it susceptible to hosting harmful content, including
disinformation and conspiracy theories. This study explores the use of
open-weight Large Language Models (LLMs), both text-only and multimodal, for
identifying conspiracy theory videos shared on YouTube. Leveraging a labeled
dataset of thousands of videos, we evaluate a variety of LLMs in a zero-shot
setting and compare their performance to a fine-tuned RoBERTa baseline. Results
show that text-based LLMs achieve high recall but lower precision, leading to
increased false positives. Multimodal models lag behind their text-only
counterparts, indicating limited benefits from visual data integration. To
assess real-world applicability, we evaluate the most accurate models on an
unlabeled dataset, finding that RoBERTa achieves performance close to LLMs with
a larger number of parameters. Our work highlights the strengths and
limitations of current LLM-based approaches for online harmful content
detection, emphasizing the need for more precise and robust systems.

### Systems and Control

### 1. [Sensitivity of DC Network Representation for GIC Analysis](http://arxiv.org/pdf/2505.23016v1)

Authors: Aniruddh Mishra, Arthur K. Barnes, Jose E. Tabarez, Adam Mate

Geomagnetic disturbances are a threat to the reliability and security of our
national critical energy infrastructures. These events specifically result in
geomagnetically induced currents, which can cause damage to transformers due to
magnetic saturation. In order to mitigate these effects, blocker devices must
be placed in optimal locations. Finding this placement requires a dc
representation of the ac transmission lines, which this paper discusses.
Different decisions in this process, including the method of representing the
blocking devices, result in significant variations to the power loss
calculations. To analyze these effects, we conclude the paper by comparing the
losses on a sample network with different modeling implementations.

### 2. [Detecting Switching Attacks On Traffic Flow Regulation For Changing Driving Patterns](http://arxiv.org/pdf/2505.23033v1)

Authors: Sanchita Ghosh, Tanushree Roy

Modern traffic management systems increasingly adopt hierarchical control
strategies for improved efficiency and scalability, where a local traffic
controller mode is chosen by a supervisory controller based on the changing
large-scale driving patterns. Unfortunately, such local metering controllers
are also vulnerable to cyberattacks that can disrupt the controller switching,
leading to undesired, inefficient, and even unsafe traffic operations.
Additionally, the detection of such attacks becomes challenging when the
operational mode of the traffic is uncertain and the operational mode
identification is delayed. Thus, in this work, we propose a cyberattack
detection scheme to detect the compromised controller switching in ramp
metering for an uncertain, multimodal macroscopic traffic operation of a
freeway segment. In particular, we propose a bank of detectors corresponding to
each admissible traffic mode that can compensate for the uncertain traffic mode
of the freeway. Furthermore, we utilize backstepping tools along with Lyapunov
function theory to achieve analytical performance guarantees for the detector,
such as nominal exponential stability, anomaly/uncertainty-to-residual
stability, robustness, and sensitivity. Finally, we demonstrate the efficacy of
the proposed detection scheme through simulations of free traffic under
realistic traffic parameters, uncertainties, and commonly occurring attack
scenarios.

### 3. [Voltage Control of the Boost Converter: PI vs. Nonlinear Passivity-based Control](http://arxiv.org/pdf/2505.23112v1)

Authors: Leyan Fang, Romeo Ortega, Robert Griñó

We carry-out a detailed analysis of direct voltage control of a Boost
converter feeding a simple resistive load. First, we prove that using a
classical PI control to stabilize a desired equilibrium leads to a very
complicated dynamic behavior consisting of two equilibrium points, one of them
always unstable for all PI gains and circuit parameter values. Interestingly,
the second equilibrium point may be rendered stable -- but for all tuning gains
leading to an extremely large value of the circuit current and the controller
integrator state. Moreover, if we neglect the resistive effect of the inductor,
there is only one equilibrium and it is always unstable. From a practical point
of view, it is important to note that the only useful equilibrium point is that
of minimum current and that, in addition, there is always a resistive component
in the inductor either by its parasitic resistance or by the resistive
component of the output impedance of the previous stage. In opposition to this
troublesome scenario we recall three nonlinear voltage-feedback controllers,
that ensure asymptotic stability of the desired equilibrium with simple gain
tuning rules, an easily defined domain of attraction and smooth transient
behavior. Two of them are very simple, nonlinear, static voltage feedback
rules, while the third one is a variation of the PID scheme called
PID-Passivity-based Control (PBC). In its original formulation PID-PBC requires
full state measurement, but we present a modified version that incorporates a
current observer. All three nonlinear controllers are designed following the
principles of PBC, which has had enormous success in many engineering
applications.

### 4. [Interturn Fault Detection in IPMSMs: Two Adaptive Observer-based Solutions](http://arxiv.org/pdf/2505.23125v1)

Authors: Romeo Ortega, Alexey Bobtsov, Leyan Fang, Oscar Texis-Loaiza, Johannes Schiffer

In this paper we address the problem of online detection of inter-turn
short-circuit faults (ITSCFs) that occur in permanent magnet synchronous motors
(PMSMs). We propose two solutions to this problem: (i) a very simple linear
observer and (ii) a generalized parameter estimation based observer, that
incorporates a high performance estimator -- with both observers detecting the
short-circuit current and the fault intensity. Although the first solution
guarantees the detection of the fault exponentially fast, the rate of
convergence is fully determined by the motor parameters that, in some cases,
may be too slow. The second observer, on the other hand, ensures finite
convergence time under the weakest assumption of interval excitation. To make
the observers adaptive, we develop a parameter estimator that, in the case of
isotropic PMSMs, estimates on-line (exponentially fast) the resistance and
inductance of the motor. It should be underscored that, in contrast with
existing observers (including the widely popular Kalman filter) that provide
indirect information of the fault current, our observers provide explicit one
-- namely the amplitude of the fault current. The performance of both
observers, in their linear and generalized parameter estimation-based versions,
is illustrated with realistic simulation studies.

### 5. [Optimizing Connectivity and Scheduling of Near/Far Field Users in Massive MIMO NOMA System](http://arxiv.org/pdf/2505.23259v1)

Authors: Ziad Qais Al-Abbasi

It is envisioned that the next generations of wireless communication
environment will be characterized with dense traffic demand due to the
prediction that there will be large numbers of active users. Hence, it is
important to find a solution to deal with such dense numbers of users. This
paper investigates optimizing the connectivity and users scheduling to improve
the performance of near and far field users in a downlink, multiuser, massive
MIMO-NOMA system. For the considered system model, combining NOMA side by side
with massive MIMO offers a great opportunity to exploit the available radio
resources and boost the overall system efficiency. The paper proposes separate
clustering of near field users and far field users. It also proposes using a
beamforming scheme to separately serve the users within each cluster. However,
NOMA is proposed to be applied among all users to boost resource sharing. In
particular, a cognitive-NOMA beamforming scheme and NOMA themed beamforming are
proposed to serve the users within each cluster, and they are compared against
random beamforming from literature. Simulation results show that both of the
proposed beamforming schemes proved their superiority as compared to random
beamforming. Several scheduling techniques were also considered in this paper
to examine possible solutions for boosting the system performance considered,
namely, priority, joint, dynamic, and fairness-based scheduling techniques for
both near field and far field users. The paper also proposes a suboptimal,
fairness aiming and gradual allocation approach for allocating the transmission
power among the users. The results show that user-clustering offers better
connectivity and scheduling performance than the case where no clustering is
applied.

### 6. [Evaluation of Voltage Unbalance Metrics in Distribution Networks with High DER Penetration](http://arxiv.org/pdf/2505.23435v1)

Authors: Alireza Zabihi, Luis Badesa, Araceli Hernandez

Voltage unbalance, caused by variations in voltage magnitude and phase angle,
is a significant power quality issue in three-phase systems, leading to
equipment inefficiencies and increased system losses. The integration of
distributed energy resources (DER) into the grid adds complexity, as DER can
either reduce or worsen voltage unbalance, depending on factors such as grid
configuration and the distribution of loads and DER themselves. This study
explores the effects of DER penetration on voltage unbalance levels and the
accuracy of the different indices most commonly used to quantify this
unbalance. The results highlight the varying impacts of DER on unbalance and
index performance, emphasizing the need for effective strategies to assess
voltage unbalance in modern distribution systems.

### 7. [Categorical Lyapunov Theory II: Stability of Systems](http://arxiv.org/pdf/2505.22968v1)

Authors: Aaron D. Ames, Sébastien Mattenet, Joe Moeller

Lyapunov's theorem provides a foundational characterization of stable
equilibrium points in dynamical systems. In this paper, we develop a framework
for stability for F-coalgebras. We give two definitions for a categorical
setting in which we can study the stability of a coalgebra for an endofunctor
F. One is minimal and better suited for concrete settings, while the other is
more intricate and provides a richer theory. We prove a Lyapunov theorem for
both notions of setting for stability, and a converse Lyapunov theorem for the
second.

### 8. [System Identification for Virtual Sensor-Based Model Predictive Control: Application to a 2-DoF Direct-Drive Robotic Arm](http://arxiv.org/pdf/2505.23138v1)

Authors: Kosei Tsuji, Ichiro Maruta, Kenji Fujimoto, Tomoyuki Maeda, Yoshihisa Tamase, Tsukasa Shinohara

Nonlinear Model Predictive Control (NMPC) offers a powerful approach for
controlling complex nonlinear systems, yet faces two key challenges. First,
accurately modeling nonlinear dynamics remains difficult. Second, variables
directly related to control objectives often cannot be directly measured during
operation. Although high-cost sensors can acquire these variables during model
development, their use in practical deployment is typically infeasible. To
overcome these limitations, we propose a Predictive Virtual Sensor
Identification (PVSID) framework that leverages temporary high-cost sensors
during the modeling phase to create virtual sensors for NMPC implementation. We
validate PVSID on a Two-Degree-of-Freedom (2-DoF) direct-drive robotic arm with
complex joint interactions, capturing tip position via motion capture during
modeling and utilize an Inertial Measurement Unit (IMU) in NMPC. Experimental
results show our NMPC with identified virtual sensors achieves precise tip
trajectory tracking without requiring the motion capture system during
operation. PVSID offers a practical solution for implementing optimal control
in nonlinear systems where the measurement of key variables is constrained by
cost or operational limitations.

### 9. [Latent Representations for Control Design with Provable Stability and Safety Guarantees](http://arxiv.org/pdf/2505.23210v1)

Authors: Paul Lutkus, Kaiyuan Wang, Lars Lindemann, Stephen Tu

We initiate a formal study on the use of low-dimensional latent
representations of dynamical systems for verifiable control synthesis. Our main
goal is to enable the application of verification techniques -- such as
Lyapunov or barrier functions -- that might otherwise be computationally
prohibitive when applied directly to the full state representation. Towards
this goal, we first provide dynamics-aware approximate conjugacy conditions
which formalize the notion of reconstruction error necessary for systems
analysis. We then utilize our conjugacy conditions to transfer the stability
and invariance guarantees of a latent certificate function (e.g., a Lyapunov or
barrier function) for a latent space controller back to the original system.
Importantly, our analysis contains several important implications for learning
latent spaces and dynamics, by highlighting the necessary geometric properties
which need to be preserved by the latent space, in addition to providing
concrete loss functions for dynamics reconstruction that are directly related
to control design. We conclude by demonstrating the applicability of our theory
to two case studies: (1) stabilization of a cartpole system, and (2) collision
avoidance for a two vehicle system.

### 10. [CF-DETR: Coarse-to-Fine Transformer for Real-Time Object Detection](http://arxiv.org/pdf/2505.23317v1)

Authors: Woojin Shin, Donghwa Kang, Byeongyun Park, Brent Byunghoon Kang, Jinkyu Lee, Hyeongboo Baek

Detection Transformers (DETR) are increasingly adopted in autonomous vehicle
(AV) perception systems due to their superior accuracy over convolutional
networks. However, concurrently executing multiple DETR tasks presents
significant challenges in meeting firm real-time deadlines (R1) and high
accuracy requirements (R2), particularly for safety-critical objects, while
navigating the inherent latency-accuracy trade-off under resource constraints.
Existing real-time DNN scheduling approaches often treat models generically,
failing to leverage Transformer-specific properties for efficient resource
allocation. To address these challenges, we propose CF-DETR, an integrated
system featuring a novel coarse-to-fine Transformer architecture and a
dedicated real-time scheduling framework NPFP**. CF-DETR employs three key
strategies (A1: coarse-to-fine inference, A2: selective fine inference, A3:
multi-level batch inference) that exploit Transformer properties to dynamically
adjust patch granularity and attention scope based on object criticality,
aiming to satisfy R2. The NPFP** scheduling framework (A4) orchestrates these
adaptive mechanisms A1-A3. It partitions each DETR task into a safety-critical
coarse subtask for guaranteed critical object detection within its deadline
(ensuring R1), and an optional fine subtask for enhanced overall accuracy (R2),
while managing individual and batched execution. Our extensive evaluations on
server, GPU-enabled embedded platforms, and actual AV platforms demonstrate
that CF-DETR, under an NPFP** policy, successfully meets strict timing
guarantees for critical operations and achieves significantly higher overall
and critical object detection accuracy compared to existing baselines across
diverse AV workloads.

### Machine Learning (Statistics Category)

### 1. [Theoretical Foundations of the Deep Copula Classifier: A Generative Approach to Modeling Dependent Features](http://arxiv.org/pdf/2505.22997v1)

Authors: Agnideep Aich, Ashit Baran Aich, Bruce Wade

Traditional classifiers often assume feature independence or rely on overly
simplistic relationships, leading to poor performance in settings where
real-world dependencies matter. We introduce the Deep Copula Classifier (DCC),
a generative model that separates the learning of each feature's marginal
distribution from the modeling of their joint dependence structure via neural
network-parameterized copulas. For each class, lightweight neural networks are
used to flexibly and adaptively capture feature interactions, making DCC
particularly effective when classification is driven by complex dependencies.
We establish that DCC converges to the Bayes-optimal classifier under standard
conditions and provide explicit convergence rates of O(n^{-r/(2r + d)}) for
r-smooth copula densities. Beyond theoretical guarantees, we outline several
practical extensions, including high-dimensional scalability through vine and
factor copula architectures, semi-supervised learning via entropy
regularization, and online adaptation using streaming gradient methods. By
unifying statistical rigor with the representational power of neural networks,
DCC offers a mathematically grounded and interpretable framework for
dependency-aware classification.

### 2. [JAPAN: Joint Adaptive Prediction Areas with Normalising-Flows](http://arxiv.org/pdf/2505.23196v1)

Authors: Eshant English, Christoph Lippert

Conformal prediction provides a model-agnostic framework for uncertainty
quantification with finite-sample validity guarantees, making it an attractive
tool for constructing reliable prediction sets. However, existing approaches
commonly rely on residual-based conformity scores, which impose geometric
constraints and struggle when the underlying distribution is multimodal. In
particular, they tend to produce overly conservative prediction areas centred
around the mean, often failing to capture the true shape of complex predictive
distributions. In this work, we introduce JAPAN (Joint Adaptive Prediction
Areas with Normalising-Flows), a conformal prediction framework that uses
density-based conformity scores. By leveraging flow-based models, JAPAN
estimates the (predictive) density and constructs prediction areas by
thresholding on the estimated density scores, enabling compact, potentially
disjoint, and context-adaptive regions that retain finite-sample coverage
guarantees. We theoretically motivate the efficiency of JAPAN and empirically
validate it across multivariate regression and forecasting tasks, demonstrating
good calibration and tighter prediction areas compared to existing baselines.
We also provide several \emph{extensions} adding flexibility to our proposed
framework.

### 3. [Stable Thompson Sampling: Valid Inference via Variance Inflation](http://arxiv.org/pdf/2505.23260v1)

Authors: Budhaditya Halder, Shubhayan Pan, Koulik Khamaru

We consider the problem of statistical inference when the data is collected
via a Thompson Sampling-type algorithm. While Thompson Sampling (TS) is known
to be both asymptotically optimal and empirically effective, its adaptive
sampling scheme poses challenges for constructing confidence intervals for
model parameters. We propose and analyze a variant of TS, called Stable
Thompson Sampling, in which the posterior variance is inflated by a logarithmic
factor. We show that this modification leads to asymptotically normal estimates
of the arm means, despite the non-i.i.d. nature of the data. Importantly, this
statistical benefit comes at a modest cost: the variance inflation increases
regret by only a logarithmic factor compared to standard TS. Our results reveal
a principled trade-off: by paying a small price in regret, one can enable valid
statistical inference for adaptive decision-making algorithms.

### 4. [Efficient Parameter Estimation for Bayesian Network Classifiers using Hierarchical Linear Smoothing](http://arxiv.org/pdf/2505.23320v1)

Authors: Connor Cooper, Geoffrey I. Webb, Daniel F. Schmidt

Bayesian network classifiers (BNCs) possess a number of properties desirable
for a modern classifier: They are easily interpretable, highly scalable, and
offer adaptable complexity. However, traditional methods for learning BNCs have
historically underperformed when compared to leading classification methods
such as random forests. Recent parameter smoothing techniques using
hierarchical Dirichlet processes (HDPs) have enabled BNCs to achieve
performance competitive with random forests on categorical data, but these
techniques are relatively inflexible, and require a complicated, specialized
sampling process. In this paper, we introduce a novel method for parameter
estimation that uses a log-linear regression to approximate the behaviour of
HDPs. As a linear model, our method is remarkably flexible and simple to
interpret, and can leverage the vast literature on learning linear models. Our
experiments show that our method can outperform HDP smoothing while being
orders of magnitude faster, remaining competitive with random forests on
categorical data.

### 5. [Epistemic Errors of Imperfect Multitask Learners When Distributions Shift](http://arxiv.org/pdf/2505.23496v1)

Authors: Sabina J. Sloman, Michele Caprio, Samuel Kaski

When data are noisy, a statistical learner's goal is to resolve epistemic
uncertainty about the data it will encounter at test-time, i.e., to identify
the distribution of test (target) data. Many real-world learning settings
introduce sources of epistemic uncertainty that can not be resolved on the
basis of training (source) data alone: The source data may arise from multiple
tasks (multitask learning), the target data may differ systematically from the
source data tasks (distribution shift), and/or the learner may not arrive at an
accurate characterization of the source data (imperfect learning). We introduce
a principled definition of epistemic error, and provide a generic,
decompositional epistemic error bound. Our error bound is the first to (i)
consider epistemic error specifically, (ii) accommodate all the sources of
epistemic uncertainty above, and (iii) separately attribute the error to each
of multiple aspects of the learning procedure and environment. As corollaries
of the generic result, we provide (i) epistemic error bounds specialized to the
settings of Bayesian transfer learning and distribution shift within
$\epsilon$-neighborhoods, and (ii) a set of corresponding generalization
bounds. Finally, we provide a novel definition of negative transfer, and
validate its insights in a synthetic experimental setting.

### 6. [A Gibbs Sampler for Efficient Bayesian Inference in Sign-Identified SVARs](http://arxiv.org/pdf/2505.23542v1)

Authors: Jonas E. Arias, Juan F. Rubio-Ramírez, Minchul Shin

We develop a new algorithm for inference based on SVARs identified with sign
restrictions. The key insight of our algorithm is to break apart from the
accept-reject tradition associated with sign-identified SVARs. We show that
embedding an elliptical slice sampling within a Gibbs sampler approach can
deliver dramatic gains in speed and turn previously infeasible applications
into feasible ones. We provide a tractable example to illustrate the power of
the elliptical slice sampling applied to sign-identified SVARs. We demonstrate
the usefulness of our algorithm by applying it to a well-known small-SVAR model
of the oil market featuring a tight identified set as well as to large SVAR
model with more than 100 sign restrictions.

### 7. [Going from a Representative Agent to Counterfactuals in Combinatorial Choice](http://arxiv.org/pdf/2505.23546v1)

Authors: Yanqiu Ruan, Karthyek Murthy, Karthik Natarajan

We study decision-making problems where data comprises points from a
collection of binary polytopes, capturing aggregate information stemming from
various combinatorial selection environments. We propose a nonparametric
approach for counterfactual inference in this setting based on a representative
agent model, where the available data is viewed as arising from maximizing
separable concave utility functions over the respective binary polytopes. Our
first contribution is to precisely characterize the selection probabilities
representable under this model and show that verifying the consistency of any
given aggregated selection dataset reduces to solving a polynomial-sized linear
program. Building on this characterization, we develop a nonparametric method
for counterfactual prediction. When data is inconsistent with the model,
finding a best-fitting approximation for prediction reduces to solving a
compact mixed-integer convex program. Numerical experiments based on synthetic
data demonstrate the method's flexibility, predictive accuracy, and strong
representational power even under model misspecification.

### 8. [Learning Parametric Distributions from Samples and Preferences](http://arxiv.org/pdf/2505.23557v1)

Authors: Marc Jourdan, Gizem Yüce, Nicolas Flammarion

Recent advances in language modeling have underscored the role of preference
feedback in enhancing model performance. This paper investigates the conditions
under which preference feedback improves parameter estimation in classes of
continuous parametric distributions. In our framework, the learner observes
pairs of samples from an unknown distribution along with their relative
preferences depending on the same unknown parameter. We show that
preference-based M-estimators achieve a better asymptotic variance than
sample-only M-estimators, further improved by deterministic preferences.
Leveraging the hard constraints revealed by deterministic preferences, we
propose an estimator achieving an estimation error scaling of
$\mathcal{O}(1/n)$ -- a significant improvement over the $\Theta(1/\sqrt{n})$
rate attainable with samples alone. Next, we establish a lower bound that
matches this accelerated rate; up to dimension and problem-dependent constants.
While the assumptions underpinning our analysis are restrictive, they are
satisfied by notable cases such as Gaussian or Laplace distributions for
preferences based on the log-probability reward.

### 9. [Inference-time Scaling of Diffusion Models through Classical Search](http://arxiv.org/pdf/2505.23614v1)

Authors: Xiangcheng Zhang, Haowei Lin, Haotian Ye, James Zou, Jianzhu Ma, Yitao Liang, Yilun Du

Classical search algorithms have long underpinned modern artificial
intelligence. In this work, we tackle the challenge of inference-time control
in diffusion models -- adapting generated outputs to meet diverse test-time
objectives -- using principles from classical search. We propose a general
framework that orchestrates local and global search to efficiently navigate the
generative space. It employs a theoretically grounded local search via annealed
Langevin MCMC and performs compute-efficient global exploration using
breadth-first and depth-first tree search. We evaluate our approach on a range
of challenging domains, including planning, offline reinforcement learning, and
image generation. Across all tasks, we observe significant gains in both
performance and efficiency. These results show that classical search provides a
principled and practical foundation for inference-time scaling in diffusion
models. Project page at diffusion-inference-scaling.github.io.

### 10. [Instance-Optimality for Private KL Distribution Estimation](http://arxiv.org/pdf/2505.23620v1)

Authors: Jiayuan Ye, Vitaly Feldman, Kunal Talwar

We study the fundamental problem of estimating an unknown discrete
distribution $p$ over $d$ symbols, given $n$ i.i.d. samples from the
distribution. We are interested in minimizing the KL divergence between the
true distribution and the algorithm's estimate. We first construct minimax
optimal private estimators. Minimax optimality however fails to shed light on
an algorithm's performance on individual (non-worst-case) instances $p$ and
simple minimax-optimal DP estimators can have poor empirical performance on
real distributions. We then study this problem from an instance-optimality
viewpoint, where the algorithm's error on $p$ is compared to the minimum
achievable estimation error over a small local neighborhood of $p$. Under
natural notions of local neighborhood, we propose algorithms that achieve
instance-optimality up to constant factors, with and without a differential
privacy constraint. Our upper bounds rely on (private) variants of the
Good-Turing estimator. Our lower bounds use additive local neighborhoods that
more precisely captures the hardness of distribution estimation in KL
divergence, compared to ones considered in prior works.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

### 1. [Automated diagnosis for extraction difficulty of maxillary and mandibular third molars and post-extraction complications using deep learning](https://www.nature.com/articles/s41598-025-00236-7)

Authors: Junseok Lee et al.

### 2. [Deep convolutional fuzzy neural networks with stork optimization on chronic cardiovascular disease monitoring for pervasive healthcare services](https://www.nature.com/articles/s41598-025-02924-w)

Authors: Nuzaiha Mohamed et al.

### 3. [Semi-supervised action recognition using logit aligned consistency and adaptive negative learning](https://www.nature.com/articles/s41598-025-01922-2)

Authors: Fengyun Zuo et al.

### 4. [Evaluating performance of large language models for atrial fibrillation management using different prompting strategies and languages](https://www.nature.com/articles/s41598-025-04309-5)

Authors: Zexi Li et al.

### 5. [A global object-oriented dynamic network for low-altitude remote sensing object detection](https://www.nature.com/articles/s41598-025-02194-6)

Authors: Daoze Tang et al.

### 6. [The analysis of motion recognition model for badminton player movements using machine learning](https://www.nature.com/articles/s41598-025-02771-9)

Authors: Xuanmin Zhu et al.

### 7. [Distributed denial of service (DDoS) classification based on random forest model with backward elimination algorithm and grid search algorithm](https://www.nature.com/articles/s41598-025-03868-x)

Authors: Mohamed S. Sawah et al.

### 8. [On the accurate computation of expected modularity in probabilistic networks](https://www.nature.com/articles/s41598-025-99114-5)

Authors: Xin Shen et al.

### 9. [Assessing and improving reliability of neighbor embedding methods: a map-continuity perspective](https://www.nature.com/articles/s41467-025-60434-9)

Authors: Zhexuan Liu et al.

### 10. [An end-to-end mass spectrometry data classification model with a unified architecture](https://www.nature.com/articles/s41598-025-03741-x)

Authors: Yinchu Wang et al.

### 11. [An improve fraud detection framework via dynamic representations and adaptive frequency response filter](https://www.nature.com/articles/s41598-025-02032-9)

Authors: Juncheng Yang et al.

### 12. [Research on a traffic flow statistical algorithm based on YBOVDT and SAM2](https://www.nature.com/articles/s41598-025-04336-2)

Authors: Yuanyuan Wang et al.

### 13. [Assessing the performance of domain-specific models for plant leaf disease classification: a comprehensive benchmark of transfer-learning on open datasets](https://www.nature.com/articles/s41598-025-03235-w)

Authors: David J. Richter et al.

