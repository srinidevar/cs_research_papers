# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-08-15 17:00:26.016797 PST.

### Artificial Intelligence

### 1. [Promoting Efficient Reasoning with Verifiable Stepwise Reward](http://arxiv.org/pdf/2508.10293v1)

Authors: Chuhuai Yue, Chengqi Dong, Yinan Gao, Hang He, Jiajun Chai, Guojun Yin, Wei Lin

Large reasoning models (LRMs) have recently achieved significant progress in
complex reasoning tasks, aided by reinforcement learning with verifiable
rewards. However, LRMs often suffer from overthinking, expending excessive
computation on simple problems and reducing efficiency. Existing efficient
reasoning methods typically require accurate task assessment to preset token
budgets or select reasoning modes, which limits their flexibility and
reliability. In this work, we revisit the essence of overthinking and identify
that encouraging effective steps while penalizing ineffective ones is key to
its solution. To this end, we propose a novel rule-based verifiable stepwise
reward mechanism (VSRM), which assigns rewards based on the performance of
intermediate states in the reasoning trajectory. This approach is intuitive and
naturally fits the step-by-step nature of reasoning tasks. We conduct extensive
experiments on standard mathematical reasoning benchmarks, including AIME24 and
AIME25, by integrating VSRM with PPO and Reinforce++. Results show that our
method achieves substantial output length reduction while maintaining original
reasoning performance, striking an optimal balance between efficiency and
accuracy. Further analysis of overthinking frequency and pass@k score before
and after training demonstrates that our approach in deed effectively
suppresses ineffective steps and encourages effective reasoning, fundamentally
alleviating the overthinking problem. All code will be released upon
acceptance.

### 2. [Multi-Agent Trust Region Policy Optimisation: A Joint Constraint Approach](http://arxiv.org/pdf/2508.10340v1)

Authors: Chak Lam Shek, Guangyao Shi, Pratap Tokekar

Multi-agent reinforcement learning (MARL) requires coordinated and stable
policy updates among interacting agents. Heterogeneous-Agent Trust Region
Policy Optimization (HATRPO) enforces per-agent trust region constraints using
Kullback-Leibler (KL) divergence to stabilize training. However, assigning each
agent the same KL threshold can lead to slow and locally optimal updates,
especially in heterogeneous settings. To address this limitation, we propose
two approaches for allocating the KL divergence threshold across agents:
HATRPO-W, a Karush-Kuhn-Tucker-based (KKT-based) method that optimizes
threshold assignment under global KL constraints, and HATRPO-G, a greedy
algorithm that prioritizes agents based on improvement-to-divergence ratio. By
connecting sequential policy optimization with constrained threshold
scheduling, our approach enables more flexible and effective learning in
heterogeneous-agent settings. Experimental results demonstrate that our methods
significantly boost the performance of HATRPO, achieving faster convergence and
higher final rewards across diverse MARL benchmarks. Specifically, HATRPO-W and
HATRPO-G achieve comparable improvements in final performance, each exceeding
22.5%. Notably, HATRPO-W also demonstrates more stable learning dynamics, as
reflected by its lower variance.

### 3. [What to Ask Next? Probing the Imaginative Reasoning of LLMs with TurtleSoup Puzzles](http://arxiv.org/pdf/2508.10358v1)

Authors: Mengtao Zhou, Sifan Wu, Huan Zhang, Qi Sima, Bang Liu

We investigate the capacity of Large Language Models (LLMs) for imaginative
reasoning--the proactive construction, testing, and revision of hypotheses in
information-sparse environments. Existing benchmarks, often static or focused
on social deduction, fail to capture the dynamic, exploratory nature of this
reasoning process. To address this gap, we introduce a comprehensive research
framework based on the classic "Turtle Soup" game, integrating a benchmark, an
agent, and an evaluation protocol. We present TurtleSoup-Bench, the first
large-scale, bilingual, interactive benchmark for imaginative reasoning,
comprising 800 turtle soup puzzles sourced from both the Internet and expert
authors. We also propose Mosaic-Agent, a novel agent designed to assess LLMs'
performance in this setting. To evaluate reasoning quality, we develop a
multi-dimensional protocol measuring logical consistency, detail completion,
and conclusion alignment. Experiments with leading LLMs reveal clear capability
limits, common failure patterns, and a significant performance gap compared to
humans. Our work offers new insights into LLMs' imaginative reasoning and
establishes a foundation for future research on exploratory agent behavior.

### 4. [LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](http://arxiv.org/pdf/2508.10391v1)

Authors: Yaoze Zhang, Rong Wu, Pinlong Cai, Xiaoman Wang, Guohang Yan, Song Mao, Ding Wang, Botian Shi

Retrieval-Augmented Generation (RAG) plays a crucial role in grounding Large
Language Models by leveraging external knowledge, whereas the effectiveness is
often compromised by the retrieval of contextually flawed or incomplete
information. To address this, knowledge graph-based RAG methods have evolved
towards hierarchical structures, organizing knowledge into multi-level
summaries. However, these approaches still suffer from two critical,
unaddressed challenges: high-level conceptual summaries exist as disconnected
``semantic islands'', lacking the explicit relations needed for cross-community
reasoning; and the retrieval process itself remains structurally unaware, often
degenerating into an inefficient flat search that fails to exploit the graph's
rich topology. To overcome these limitations, we introduce LeanRAG, a framework
that features a deeply collaborative design combining knowledge aggregation and
retrieval strategies. LeanRAG first employs a novel semantic aggregation
algorithm that forms entity clusters and constructs new explicit relations
among aggregation-level summaries, creating a fully navigable semantic network.
Then, a bottom-up, structure-guided retrieval strategy anchors queries to the
most relevant fine-grained entities and then systematically traverses the
graph's semantic pathways to gather concise yet contextually comprehensive
evidence sets. The LeanRAG can mitigate the substantial overhead associated
with path retrieval on graphs and minimizes redundant information retrieval.
Extensive experiments on four challenging QA benchmarks with different domains
demonstrate that LeanRAG significantly outperforming existing methods in
response quality while reducing 46\% retrieval redundancy. Code is available
at: https://github.com/RaZzzyz/LeanRAG

### 5. [SEQ-GPT: LLM-assisted Spatial Query via Example](http://arxiv.org/pdf/2508.10486v1)

Authors: Ivan Khai Ze Lim, Ningyi Liao, Yiming Yang, Gerald Wei Yong Yip, Siqiang Luo

Contemporary spatial services such as online maps predominantly rely on user
queries for location searches. However, the user experience is limited when
performing complex tasks, such as searching for a group of locations
simultaneously. In this study, we examine the extended scenario known as
Spatial Exemplar Query (SEQ), where multiple relevant locations are jointly
searched based on user-specified examples. We introduce SEQ-GPT, a spatial
query system powered by Large Language Models (LLMs) towards more versatile SEQ
search using natural language. The language capabilities of LLMs enable unique
interactive operations in the SEQ process, including asking users to clarify
query details and dynamically adjusting the search based on user feedback. We
also propose a tailored LLM adaptation pipeline that aligns natural language
with structured spatial data and queries through dialogue synthesis and
multi-model cooperation. SEQ-GPT offers an end-to-end demonstration for
broadening spatial search with realistic data and application scenarios.

### 6. [MSRS: Adaptive Multi-Subspace Representation Steering for Attribute Alignment in Large Language Models](http://arxiv.org/pdf/2508.10599v1)

Authors: Xinyan Jiang, Lin Zhang, Jiayi Zhang, Qingsong Yang, Guimin Hu, Di Wang, Lijie Hu

Activation steering offers a promising approach to controlling the behavior
of Large Language Models by directly manipulating their internal activations.
However, most existing methods struggle to jointly steer multiple attributes,
often resulting in interference and undesirable trade-offs. To address this
challenge, we propose Multi-Subspace Representation Steering (MSRS), a novel
framework for effective multi-attribute steering via subspace representation
fine-tuning. MSRS reduces inter-attribute interference by allocating orthogonal
subspaces to each attribute, isolating their influence within the model's
representation space. MSRS also incorporates a hybrid subspace composition
strategy: it combines attribute-specific subspaces for unique steering
directions with a shared subspace for common steering directions. A dynamic
weighting function learns to efficiently integrate these components for precise
control. During inference, MSRS introduces a token-level steering mechanism
that dynamically identifies and intervenes on the most semantically relevant
tokens, enabling fine-grained behavioral modulation. Experimental results show
that MSRS significantly reduces attribute conflicts, surpasses existing methods
across a range of attributes, and generalizes effectively to diverse downstream
tasks.

### 7. [GenOM: Ontology Matching with Description Generation and Large Language Model](http://arxiv.org/pdf/2508.10703v1)

Authors: Yiping Song, Jiaoyan Chen, Renate A. Schmidt

Ontology matching (OM) plays an essential role in enabling semantic
interoperability and integration across heterogeneous knowledge sources,
particularly in the biomedical domain which contains numerous complex concepts
related to diseases and pharmaceuticals. This paper introduces GenOM, a large
language model (LLM)-based ontology alignment framework, which enriches the
semantic representations of ontology concepts via generating textual
definitions, retrieves alignment candidates with an embedding model, and
incorporates exact matching-based tools to improve precision. Extensive
experiments conducted on the OAEI Bio-ML track demonstrate that GenOM can often
achieve competitive performance, surpassing many baselines including
traditional OM systems and recent LLM-based methods. Further ablation studies
confirm the effectiveness of semantic enrichment and few-shot prompting,
highlighting the framework's robustness and adaptability.

### 8. [The Knowledge-Reasoning Dissociation: Fundamental Limitations of LLMs in Clinical Natural Language Inference](http://arxiv.org/pdf/2508.10777v1)

Authors: Maël Jullien, Marco Valentino, André Freitas

Large language models are often assumed to acquire increasingly structured,
generalizable internal representations simply by scaling data and parameters.
We interrogate this assumption by introducing a Clinical Trial Natural Language
Inference benchmark comprising four reasoning families, Causal Attribution,
Compositional Grounding, Epistemic Verification, and Risk State Abstraction.
Each item is paired with a targeted Ground Knowledge and Meta-Level Reasoning
Verification (GKMRV) probe, allowing us to dissociate failures of factual
access from failures of inference. We evaluate six contemporary LLMs under both
direct and chain of thought prompting.
  Models achieve near-ceiling GKMRV accuracy (mean accuracy 0.918) yet perform
poorly on the main reasoning tasks (mean accuracy 0.25). Despite low accuracy,
output inferences are highly consistent across samples (mean 0.87), indicating
a systematic application of underlying heuristics and shortcuts.
  These results reveal fundamental structural and representational limitations:
current LLMs often possess the relevant clinical knowledge but lack the
structured, composable internal representations needed to deploy it reliably
(e.g., integrating constraints, weighing evidence, or simulating
counterfactuals). Decoupling knowledge from reasoning with GKMRV makes this
dissociation explicit and measurable, providing an effective framework for
probing the reliability of LLMs in high-stakes domains.

### 9. [Who Benefits from AI Explanations? Towards Accessible and Interpretable Systems](http://arxiv.org/pdf/2508.10806v1)

Authors: Maria J. P. Peixoto, Akriti Pandey, Ahsan Zaman, Peter R. Lewis

As AI systems are increasingly deployed to support decision-making in
critical domains, explainability has become a means to enhance the
understandability of these outputs and enable users to make more informed and
conscious choices. However, despite growing interest in the usability of
eXplainable AI (XAI), the accessibility of these methods, particularly for
users with vision impairments, remains underexplored. This paper investigates
accessibility gaps in XAI through a two-pronged approach. First, a literature
review of 79 studies reveals that evaluations of XAI techniques rarely include
disabled users, with most explanations relying on inherently visual formats.
Second, we present a four-part methodological proof of concept that
operationalizes inclusive XAI design: (1) categorization of AI systems, (2)
persona definition and contextualization, (3) prototype design and
implementation, and (4) expert and user assessment of XAI techniques for
accessibility. Preliminary findings suggest that simplified explanations are
more comprehensible for non-visual users than detailed ones, and that
multimodal presentation is required for more equitable interpretability.

### 10. [MRFD: Multi-Region Fusion Decoding with Self-Consistency for Mitigating Hallucinations in LVLMs](http://arxiv.org/pdf/2508.10264v1)

Authors: Haonan Ge, Yiwei Wang, Ming-Hsuan Yang, Yujun Cai

Large Vision-Language Models (LVLMs) have shown strong performance across
multimodal tasks. However, they often produce hallucinations -- text that is
inconsistent with visual input, due to the limited ability to verify
information in different regions of the image. To address this, we propose
Multi-Region Fusion Decoding (MRFD), a training-free decoding method that
improves factual grounding by modeling inter-region consistency. MRFD
identifies salient regions using cross-attention, generates initial responses
for each, and computes reliability weights based on Jensen-Shannon Divergence
(JSD) among the responses. These weights guide a consistency-aware fusion of
per-region predictions, using region-aware prompts inspired by Chain-of-Thought
reasoning. Experiments across multiple LVLMs and benchmarks show that MRFD
significantly reduces hallucinations and improves response factuality without
requiring model updates.

### Hardware Architecture

### 1. [DiffAxE: Diffusion-driven Hardware Accelerator Generation and Design Space Exploration](http://arxiv.org/pdf/2508.10303v1)

Authors: Arkapravo Ghosh, Abhishek Moitra, Abhiroop Bhattacharjee, Ruokai Yin, Priyadarshini Panda

Design space exploration (DSE) is critical for developing optimized hardware
architectures, especially for AI workloads such as deep neural networks (DNNs)
and large language models (LLMs), which require specialized acceleration. As
model complexity grows, accelerator design spaces have expanded to O(10^17),
becoming highly irregular, non-convex, and exhibiting many-to-one mappings from
design configurations to performance metrics. This complexity renders direct
inverse derivation infeasible and necessitates heuristic or sampling-based
optimization. Conventional methods - including Bayesian optimization, gradient
descent, reinforcement learning, and genetic algorithms - depend on iterative
sampling, resulting in long runtimes and sensitivity to initialization. Deep
learning-based approaches have reframed DSE as classification using
recommendation models, but remain limited to small-scale (O(10^3)), less
complex design spaces. To overcome these constraints, we propose a generative
approach that models hardware design as 1-D image synthesis conditioned on
target performance, enabling efficient learning of non-differentiable,
non-bijective hardware-performance mappings. Our framework achieves 0.86% lower
generation error than Bayesian optimization with a 17000x speedup, and
outperforms GANDSE with 30% lower error at only 1.83x slower search. We further
extend the method to a structured DSE setting, attaining 9.8% lower
energy-delay product (EDP) and 6% higher performance, with up to 145.6x and
1312x faster search compared to existing optimization methods on O(10^17)
design spaces. For LLM inference, our method achieves 3.37x and 7.75x lower EDP
on a 32nm ASIC and Xilinx Ultrascale+ VPU13 FPGA, respectively, compared to the
state-of-the-art DOSA framework.

### 2. [THERMOS: Thermally-Aware Multi-Objective Scheduling of AI Workloads on Heterogeneous Multi-Chiplet PIM Architectures](http://arxiv.org/pdf/2508.10691v1)

Authors: Alish Kanani, Lukas Pfromm, Harsh Sharma, Janardhan Rao Doppa, Partha Pratim Pande, Umit Y. Ogras

Chiplet-based integration enables large-scale systems that combine diverse
technologies, enabling higher yield, lower costs, and scalability, making them
well-suited to AI workloads. Processing-in-Memory (PIM) has emerged as a
promising solution for AI inference, leveraging technologies such as ReRAM,
SRAM, and FeFET, each offering unique advantages and trade-offs. A
heterogeneous chiplet-based PIM architecture can harness the complementary
strengths of these technologies to enable higher performance and energy
efficiency. However, scheduling AI workloads across such a heterogeneous system
is challenging due to competing performance objectives, dynamic workload
characteristics, and power and thermal constraints. To address this need, we
propose THERMOS, a thermally-aware, multi-objective scheduling framework for AI
workloads on heterogeneous multi-chiplet PIM architectures. THERMOS trains a
single multi-objective reinforcement learning (MORL) policy that is capable of
achieving Pareto-optimal execution time, energy, or a balanced objective at
runtime, depending on the target preferences. Comprehensive evaluations show
that THERMOS achieves up to 89% faster average execution time and 57% lower
average energy consumption than baseline AI workload scheduling algorithms with
only 0.14% runtime and 0.022% energy overhead.

### 3. [AnalogSeeker: An Open-source Foundation Language Model for Analog Circuit Design](http://arxiv.org/pdf/2508.10409v1)

Authors: Zihao Chen, Ji Zhuang, Jinyi Shen, Xiaoyue Ke, Xinyi Yang, Mingjie Zhou, Zhuoyao Du, Xu Yan, Zhouyang Wu, Zhenyu Xu, Jiangli Huang, Li Shang, Xuan Zeng, Fan Yang

In this paper, we propose AnalogSeeker, an effort toward an open-source
foundation language model for analog circuit design, with the aim of
integrating domain knowledge and giving design assistance. To overcome the
scarcity of data in this field, we employ a corpus collection strategy based on
the domain knowledge framework of analog circuits. High-quality, accessible
textbooks across relevant subfields are systematically curated and cleaned into
a textual domain corpus. To address the complexity of knowledge of analog
circuits, we introduce a granular domain knowledge distillation method. Raw,
unlabeled domain corpus is decomposed into typical, granular learning nodes,
where a multi-agent framework distills implicit knowledge embedded in
unstructured text into question-answer data pairs with detailed reasoning
processes, yielding a fine-grained, learnable dataset for fine-tuning. To
address the unexplored challenges in training analog circuit foundation models,
we explore and share our training methods through both theoretical analysis and
experimental validation. We finally establish a fine-tuning-centric training
paradigm, customizing and implementing a neighborhood self-constrained
supervised fine-tuning algorithm. This approach enhances training outcomes by
constraining the perturbation magnitude between the model's output
distributions before and after training. In practice, we train the
Qwen2.5-32B-Instruct model to obtain AnalogSeeker, which achieves 85.04%
accuracy on AMSBench-TQA, the analog circuit knowledge evaluation benchmark,
with a 15.67% point improvement over the original model and is competitive with
mainstream commercial models. Furthermore, AnalogSeeker also shows
effectiveness in the downstream operational amplifier design task. AnalogSeeker
is open-sourced at https://huggingface.co/analogllm/analogseeker for research
use.

### Computational Complexity

### 1. [On Kernelization with Access to NP-Oracles](http://arxiv.org/pdf/2508.10550v1)

Authors: Hendrik Molter, Meirav Zehavi

Kernelization is the standard framework to analyze preprocessing routines
mathematically. Here, in terms of efficiency, we demand the preprocessing
routine to run in time polynomial in the input size. However, today, various
NP-complete problems are already solved very fast in practice; in particular,
SAT-solvers and ILP-solvers have become extremely powerful and used frequently.
Still, this fails to capture the wide variety of computational problems that
lie at higher levels of the polynomial hierarchy. Thus, for such problems, it
is natural to relax the definition of kernelization to permit the preprocessing
routine to make polynomially many calls to a SAT-solver, rather than run,
entirely, in polynomial time.
  Our conceptual contribution is the introduction of a new notion of a kernel
that harnesses the power of SAT-solvers for preprocessing purposes, and which
we term a P^NP-Kernel. Technically, we investigate various facets of this
notion, by proving both positive and negative results, including a lower-bounds
framework to reason about the negative results. Here, we consider both
satisfiability and graph problems. Additionally, we present a meta-theorem for
so-called "discovery problems". This work falls into a long line of research on
extensions of the concept of kernelization, including lossy kernels [Lokshtanov
et al., STOC '17], dynamic kernels [Alman et al., ACM TALG '20], counting
kernels [Lokshtanov et al., ICTS '24], and streaming kernels [Fafianie and
Kratsch, MFCS '14].

### 2. [Deciding Whether a C-Q Channel Preserves a Bit is QCMA-Complete](http://arxiv.org/pdf/2508.10664v1)

Authors: Kiera Hutton, Arthur Mehta, Andrej Vukovic

We prove that deciding whether a classical-quantum (C-Q) channel can exactly
preserve a single classical bit is QCMA-complete. This "bit-preservation"
problem is a special case of orthogonality-constrained optimization tasks over
C-Q channels, in which one seeks orthogonal input states whose outputs have
small or large Hilbert-Schmidt overlap after passing through the channel. Both
problems can be cast as biquadratic optimization with orthogonality
constraints. Our main technical contribution uses tools from matrix analysis to
give a complete characterization of the optimal witnesses: computational basis
states for the minimum, and |+>, |-> over a single basis pair for the maximum.
Using this characterization, we give concise proofs of QCMA-completeness for
both problems.

### Computational Engineering

### 1. [Chem3DLLM: 3D Multimodal Large Language Models for Chemistry](http://arxiv.org/pdf/2508.10696v1)

Authors: Lei Jiang, Shuzhou Sun, Biqing Qi, Yuchen Fu, Xiaohua Xu, Yuqiang Li, Dongzhan Zhou, Tianfan Fu

In the real world, a molecule is a 3D geometric structure. Compared to 1D
SMILES sequences and 2D molecular graphs, 3D molecules represent the most
informative molecular modality. Despite the rapid progress of
autoregressive-based language models, they cannot handle the generation of 3D
molecular conformation due to several challenges: 1) 3D molecular structures
are incompatible with LLMs' discrete token space, 2) integrating heterogeneous
inputs like proteins, ligands, and text remains difficult within a unified
model, and 3) LLMs lack essential scientific priors, hindering the enforcement
of physical and chemical constraints during generation. To tackle these issues,
we present Chem3DLLM, a unified protein-conditioned multimodal large language
model. Our approach designs a novel reversible text encoding for 3D molecular
structures using run-length compression, achieving 3x size reduction while
preserving complete structural information. This enables seamless integration
of molecular geometry with protein pocket features in a single LLM
architecture. We employ reinforcement learning with stability-based rewards to
optimize chemical validity and incorporate a lightweight protein embedding
projector for end-to-end training. Experimental results on structure-based drug
design demonstrate state-of-the-art performance with a Vina score of -7.21,
validating our unified multimodal approach for practical drug discovery
applications.

### 2. [TOBACO: Topology Optimization via Band-limited Coordinate Networks for Compositionally Graded Alloys](http://arxiv.org/pdf/2508.10320v1)

Authors: Aaditya Chandrasekhar, Stefan Knapik, Deepak Sharma, John Reidy, Ian McCue, Jian Cao, Wei Chen

Compositionally Graded Alloys (CGAs) offer unprecedented design flexibility
by enabling spatial variations in composition; tailoring material properties to
local loading conditions. This flexibility leads to components that are
stronger, lighter, and more cost-effective than traditional monolithic
counterparts. The fabrication of CGAs have become increasingly feasible owing
to recent advancements in additive manufacturing (AM), particularly in
multi-material printing and improved precision in material deposition. However,
AM of CGAs requires imposition of manufacturing constraints; in particular
limits on the maximum spatial gradation of composition.
  This paper introduces a topology optimization (TO) based framework for
designing optimized CGA components with controlled compositional gradation. In
particular, we represent the constrained composition distribution using a
band-limited coordinate neural network. By regulating the network's bandwidth,
we ensure implicit compliance with gradation limits, eliminating the need for
explicit constraints. The proposed approach also benefits from the inherent
advantages of TO using coordinate networks, including mesh independence,
high-resolution design extraction, and end-to-end differentiability. The
effectiveness of our framework is demonstrated through various elastic and
thermo-elastic TO examples.

### 3. [Reverse Physician-AI Relationship: Full-process Clinical Diagnosis Driven by a Large Language Model](http://arxiv.org/pdf/2508.10492v1)

Authors: Shicheng Xu, Xin Huang, Zihao Wei, Liang Pang, Huawei Shen, Xueqi Cheng

Full-process clinical diagnosis in the real world encompasses the entire
diagnostic workflow that begins with only an ambiguous chief complaint. While
artificial intelligence (AI), particularly large language models (LLMs), is
transforming clinical diagnosis, its role remains largely as an assistant to
physicians. This AI-assisted working pattern makes AI can only answer specific
medical questions at certain parts within the diagnostic process, but lack the
ability to drive the entire diagnostic process starting from an ambiguous
complaint, which still relies heavily on human physicians. This gap limits AI's
ability to fully reduce physicians' workload and enhance diagnostic efficiency.
To address this, we propose a paradigm shift that reverses the relationship
between physicians and AI: repositioning AI as the primary director, with
physicians serving as its assistants. So we present DxDirector-7B, an LLM
endowed with advanced deep thinking capabilities, enabling it to drive the
full-process diagnosis with minimal physician involvement. Furthermore,
DxDirector-7B establishes a robust accountability framework for misdiagnoses,
delineating responsibility between AI and human physicians. In evaluations
across rare, complex, and real-world cases under full-process diagnosis
setting, DxDirector-7B not only achieves significant superior diagnostic
accuracy but also substantially reduces physician workload than
state-of-the-art medical LLMs as well as general-purpose LLMs. Fine-grained
analyses across multiple clinical departments and tasks validate its efficacy,
with expert evaluations indicating its potential to serve as a viable
substitute for medical specialists. These findings mark a new era where AI,
traditionally a physicians' assistant, now drives the entire diagnostic process
to drastically reduce physicians' workload, indicating an efficient and
accurate diagnostic solution.

### 4. [Physics-Informed Deep Contrast Source Inversion: A Unified Framework for Inverse Scattering Problems](http://arxiv.org/pdf/2508.10555v1)

Authors: Haoran Sun, Daoqi Liu, Hongyu Zhou, Maokun Li, Shenheng Xu, Fan Yang

Inverse scattering problems are critical in electromagnetic imaging and
medical diagnostics but are challenged by their nonlinearity and diverse
measurement scenarios. This paper proposes a physics-informed deep contrast
source inversion framework (DeepCSI) for fast and accurate medium
reconstruction across various measurement conditions. Inspired by contrast
source inversion (CSI) and neural operator methods, a residual multilayer
perceptron (ResMLP) is employed to model current distributions in the region of
interest under different transmitter excitations, effectively linearizing the
nonlinear inverse scattering problem and significantly reducing the
computational cost of traditional full-waveform inversion. By modeling medium
parameters as learnable tensors and utilizing a hybrid loss function that
integrates state equation loss, data equation loss, and total variation
regularization, DeepCSI establishes a fully differentiable framework for joint
optimization of network parameters and medium properties. Compared with
conventional methods, DeepCSI offers advantages in terms of simplicity and
universal modeling capabilities for diverse measurement scenarios, including
phase-less and multi-frequency observation. Simulations and experiments
demonstrate that DeepCSI achieves high-precision, robust reconstruction under
full-data, phaseless data, and multifrequency conditions, outperforming
traditional CSI methods and providing an efficient and universal solution for
complex inverse scattering problems.

### 5. [Virtual Sensing for Solder Layer Degradation and Temperature Monitoring in IGBT Modules](http://arxiv.org/pdf/2508.10515v1)

Authors: Andrea Urgolo, Monika Stipsitz, Helios Sanchis-Alepuz

Monitoring the degradation state of Insulated Gate Bipolar Transistor (IGBT)
modules is essential for ensuring the reliability and longevity of power
electronic systems, especially in safety-critical and high-performance
applications. However, direct measurement of key degradation indicators - such
as junction temperature, solder fatigue or delamination - remains challenging
due to the physical inaccessibility of internal components and the harsh
environment. In this context, machine learning-based virtual sensing offers a
promising alternative by bridging the gap from feasible sensor placement to the
relevant but inaccessible locations. This paper explores the feasibility of
estimating the degradation state of solder layers, and the corresponding full
temperature maps based on a limited number of physical sensors. Based on
synthetic data of a specific degradation mode, we obtain a high accuracy in the
estimation of the degraded solder area (1.17% mean absolute error), and are
able to reproduce the surface temperature of the IGBT with a maximum relative
error of 4.56% (corresponding to an average relative error of 0.37%).

### Computational Geometry

### 1. [Computing the Fréchet Distance When Just One Curve is $c$-Packed: A Simple Almost-Tight Algorithm](http://arxiv.org/pdf/2508.10537v1)

Authors: Jacobus Conradi, Ivor van der Hoog, Thijs van der Horst, Tim Ophelders

We study approximating the continuous Fr\'echet distance of two curves with
complexity $n$ and $m$, under the assumption that only one of the two curves is
$c$-packed. Driemel, Har{-}Peled and Wenk DCG'12 studied Fr\'echet distance
approximations under the assumption that both curves are $c$-packed. In
$\mathbb{R}^d$, they prove a $(1+\varepsilon)$-approximation in $\tilde{O}(d \,
c\,\frac{n+m}{\varepsilon})$ time. Bringmann and K\"unnemann IJCGA'17 improved
this to $\tilde{O}(c\,\frac{n + m }{\sqrt{\varepsilon}})$ time, which they
showed is near-tight under SETH. Recently, Gudmundsson, Mai, and Wong ISAAC'24
studied our setting where only one of the curves is $c$-packed. They provide an
involved $\tilde{O}( d \cdot (c+\varepsilon^{-1})(cn\varepsilon^{-2} +
c^2m\varepsilon^{-7} + \varepsilon^{-2d-1}))$-time algorithm when the
$c$-packed curve has $n$ vertices and the arbitrary curve has $m$, where $d$ is
the dimension in Euclidean space. In this paper, we show a simple technique to
compute a $(1+\varepsilon)$-approximation in $\mathbb{R}^d$ in time $O(d \cdot
c\,\frac{n+m}{\varepsilon}\log\frac{n+m}{\varepsilon})$ when one of the curves
is $c$-packed. Our approach is not only simpler than previous work, but also
significantly improves the dependencies on $c$, $\varepsilon$, and $d$.
Moreover, it almost matches the asymptotically tight bound for when both curves
are $c$-packed. Our algorithm is robust in the sense that it does not require
knowledge of $c$, nor information about which of the two input curves is
$c$-packed.

### Computation and Language

### 1. [A Computational Approach to Analyzing Language Change and Variation in the Constructed Language Toki Pona](http://arxiv.org/pdf/2508.10246v1)

Authors: Daniel Huang, Hyoun-A Joo

This study explores language change and variation in Toki Pona, a constructed
language with approximately 120 core words. Taking a computational and
corpus-based approach, the study examines features including fluid word classes
and transitivity in order to examine (1) changes in preferences of content
words for different syntactic positions over time and (2) variation in usage
across different corpora. The results suggest that sociolinguistic factors
influence Toki Pona in the same way as natural languages, and that even
constructed linguistic systems naturally evolve as communities use them.

### 2. [Inductive Bias Extraction and Matching for LLM Prompts](http://arxiv.org/pdf/2508.10295v1)

Authors: Christian M. Angel, Francis Ferraro

The active research topic of prompt engineering makes it evident that LLMs
are sensitive to small changes in prompt wording. A portion of this can be
ascribed to the inductive bias that is present in the LLM. By using an LLM's
output as a portion of its prompt, we can more easily create satisfactory
wording for prompts. This has the effect of creating a prompt that matches the
inductive bias in model. Empirically, we show that using this Inductive Bias
Extraction and Matching strategy improves LLM Likert ratings used for
classification by up to 19% and LLM Likert ratings used for ranking by up to
27%.

### 3. [From Surface to Semantics: Semantic Structure Parsing for Table-Centric Document Analysis](http://arxiv.org/pdf/2508.10311v1)

Authors: Xuan Li, Jialiang Dong, Raymond Wong

Documents are core carriers of information and knowl-edge, with broad
applications in finance, healthcare, and scientific research. Tables, as the
main medium for structured data, encapsulate key information and are among the
most critical document components. Existing studies largely focus on
surface-level tasks such as layout analysis, table detection, and data
extraction, lacking deep semantic parsing of tables and their contextual
associations. This limits advanced tasks like cross-paragraph data
interpretation and context-consistent analysis. To address this, we propose
DOTABLER, a table-centric semantic document parsing framework designed to
uncover deep semantic links between tables and their context. DOTABLER
leverages a custom dataset and domain-specific fine-tuning of pre-trained
models, integrating a complete parsing pipeline to identify context segments
semantically tied to tables. Built on this semantic understanding, DOTABLER
implements two core functionalities: table-centric document structure parsing
and domain-specific table retrieval, delivering comprehensive table-anchored
semantic analysis and precise extraction of semantically relevant tables.
Evaluated on nearly 4,000 pages with over 1,000 tables from real-world PDFs,
DOTABLER achieves over 90% Precision and F1 scores, demonstrating superior
performance in table-context semantic analysis and deep document parsing
compared to advanced models such as GPT-4o.

### 4. [Beyond Semantic Understanding: Preserving Collaborative Frequency Components in LLM-based Recommendation](http://arxiv.org/pdf/2508.10312v1)

Authors: Minhao Wang, Yunhang He, Cong Xu, Zhangchi Zhu, Wei Zhang

Recommender systems in concert with Large Language Models (LLMs) present
promising avenues for generating semantically-informed recommendations.
However, LLM-based recommenders exhibit a tendency to overemphasize semantic
correlations within users' interaction history. When taking pretrained
collaborative ID embeddings as input, LLM-based recommenders progressively
weaken the inherent collaborative signals as the embeddings propagate through
LLM backbones layer by layer, as opposed to traditional Transformer-based
sequential models in which collaborative signals are typically preserved or
even enhanced for state-of-the-art performance. To address this limitation, we
introduce FreLLM4Rec, an approach designed to balance semantic and
collaborative information from a spectral perspective. Item embeddings that
incorporate both semantic and collaborative information are first purified
using a Global Graph Low-Pass Filter (G-LPF) to preliminarily remove irrelevant
high-frequency noise. Temporal Frequency Modulation (TFM) then actively
preserves collaborative signal layer by layer. Note that the collaborative
preservation capability of TFM is theoretically guaranteed by establishing a
connection between the optimal but hard-to-implement local graph fourier
filters and the suboptimal yet computationally efficient frequency-domain
filters. Extensive experiments on four benchmark datasets demonstrate that
FreLLM4Rec successfully mitigates collaborative signal attenuation and achieves
competitive performance, with improvements of up to 8.00\% in NDCG@10 over the
best baseline. Our findings provide insights into how LLMs process
collaborative information and offer a principled approach for improving
LLM-based recommendation systems.

### 5. [Cross-Prompt Encoder for Low-Performing Languages](http://arxiv.org/pdf/2508.10352v1)

Authors: Beso Mikaberidze, Teimuraz Saghinadze, Simon Ostermann, Philipp Muller

Soft prompts have emerged as a powerful alternative to adapters in
parameter-efficient fine-tuning (PEFT), enabling large language models (LLMs)
to adapt to downstream tasks without architectural changes or parameter
updates. While prior work has focused on stabilizing training via parameter
interaction in small neural prompt encoders, their broader potential for
transfer across languages remains unexplored. In this paper, we demonstrate
that a prompt encoder can play a central role in improving performance on
low-performing languages-those that achieve poor accuracy even under full-model
fine-tuning. We introduce the Cross-Prompt Encoder (XPE), which combines a
lightweight encoding architecture with multi-source training on typologically
diverse languages - a design that enables the model to capture abstract and
transferable patterns across languages. To complement XPE, we propose a Dual
Soft Prompt mechanism that combines an encoder-based prompt with a directly
trained standard soft prompt. This hybrid design proves especially effective
for target languages that benefit from both broadly shared structure and
language-specific alignment. Experiments on the SIB-200 benchmark reveal a
consistent trade-off: XPE is most effective for low-performing languages, while
hybrid variants offer broader adaptability across multilingual settings.

### 6. [Making Qwen3 Think in Korean with Reinforcement Learning](http://arxiv.org/pdf/2508.10355v1)

Authors: Jungyup Lee, Jemin Kim, Sang Park, SeungJae Lee

We present a two-stage fine-tuning approach to make the large language model
Qwen3 14B "think" natively in Korean. In the first stage, supervised
fine-tuning (SFT) on a high-quality Korean reasoning dataset establishes a
strong foundation in Korean logical reasoning, yielding notable improvements in
Korean-language tasks and even some gains in general reasoning ability. In the
second stage, we employ reinforcement learning with a customized Group Relative
Policy Optimization (GRPO) algorithm to further enhance both Korean reasoning
alignment and overall problem-solving performance. We address critical
stability challenges in GRPO training - such as reward hacking and policy
collapse - by introducing an oracle judge model that calibrates the reward
signal. Our approach achieves stable learning (avoiding the collapse observed
in naive GRPO) and leads to steady, incremental performance gains. The final
RL-tuned model demonstrates substantially improved results on advanced
reasoning benchmarks (particularly math and coding tasks) while maintaining
knowledge and language proficiency, successfully conducting its internal
chain-of-thought entirely in Korean.

### 7. [Advancing Cross-lingual Aspect-Based Sentiment Analysis with LLMs and Constrained Decoding for Sequence-to-Sequence Models](http://arxiv.org/pdf/2508.10366v1)

Authors: Jakub Šmíd, Pavel Přibáň, Pavel Král

Aspect-based sentiment analysis (ABSA) has made significant strides, yet
challenges remain for low-resource languages due to the predominant focus on
English. Current cross-lingual ABSA studies often centre on simpler tasks and
rely heavily on external translation tools. In this paper, we present a novel
sequence-to-sequence method for compound ABSA tasks that eliminates the need
for such tools. Our approach, which uses constrained decoding, improves
cross-lingual ABSA performance by up to 10\%. This method broadens the scope of
cross-lingual ABSA, enabling it to handle more complex tasks and providing a
practical, efficient alternative to translation-dependent techniques.
Furthermore, we compare our approach with large language models (LLMs) and show
that while fine-tuned multilingual LLMs can achieve comparable results,
English-centric LLMs struggle with these tasks.

### 8. [Large Language Models for Summarizing Czech Historical Documents and Beyond](http://arxiv.org/pdf/2508.10368v1)

Authors: Václav Tran, Jakub Šmíd, Jiří Martínek, Ladislav Lenc, Pavel Král

Text summarization is the task of shortening a larger body of text into a
concise version while retaining its essential meaning and key information.
While summarization has been significantly explored in English and other
high-resource languages, Czech text summarization, particularly for historical
documents, remains underexplored due to linguistic complexities and a scarcity
of annotated datasets. Large language models such as Mistral and mT5 have
demonstrated excellent results on many natural language processing tasks and
languages. Therefore, we employ these models for Czech summarization, resulting
in two key contributions: (1) achieving new state-of-the-art results on the
modern Czech summarization dataset SumeCzech using these advanced models, and
(2) introducing a novel dataset called Posel od \v{C}erchova for summarization
of historical Czech documents with baseline results. Together, these
contributions provide a great potential for advancing Czech text summarization
and open new avenues for research in Czech historical text processing.

### 9. [Improving Generative Cross-lingual Aspect-Based Sentiment Analysis with Constrained Decoding](http://arxiv.org/pdf/2508.10369v1)

Authors: Jakub Šmíd, Pavel Přibáň, Pavel Král

While aspect-based sentiment analysis (ABSA) has made substantial progress,
challenges remain for low-resource languages, which are often overlooked in
favour of English. Current cross-lingual ABSA approaches focus on limited, less
complex tasks and often rely on external translation tools. This paper
introduces a novel approach using constrained decoding with
sequence-to-sequence models, eliminating the need for unreliable translation
tools and improving cross-lingual performance by 5\% on average for the most
complex task. The proposed method also supports multi-tasking, which enables
solving multiple ABSA tasks with a single model, with constrained decoding
boosting results by more than 10\%.
  We evaluate our approach across seven languages and six ABSA tasks,
surpassing state-of-the-art methods and setting new benchmarks for previously
unexplored tasks. Additionally, we assess large language models (LLMs) in
zero-shot, few-shot, and fine-tuning scenarios. While LLMs perform poorly in
zero-shot and few-shot settings, fine-tuning achieves competitive results
compared to smaller multilingual models, albeit at the cost of longer training
and inference times.
  We provide practical recommendations for real-world applications, enhancing
the understanding of cross-lingual ABSA methodologies. This study offers
valuable insights into the strengths and limitations of cross-lingual ABSA
approaches, advancing the state-of-the-art in this challenging research domain.

### 10. [Evaluating LLMs on Chinese Idiom Translation](http://arxiv.org/pdf/2508.10421v1)

Authors: Cai Yang, Yao Dou, David Heineman, Xiaofeng Wu, Wei Xu

Idioms, whose figurative meanings usually differ from their literal
interpretations, are common in everyday language, especially in Chinese, where
they often contain historical references and follow specific structural
patterns. Despite recent progress in machine translation with large language
models, little is known about Chinese idiom translation. In this work, we
introduce IdiomEval, a framework with a comprehensive error taxonomy for
Chinese idiom translation. We annotate 900 translation pairs from nine modern
systems, including GPT-4o and Google Translate, across four domains: web, news,
Wikipedia, and social media. We find these systems fail at idiom translation,
producing incorrect, literal, partial, or even missing translations. The
best-performing system, GPT-4, makes errors in 28% of cases. We also find that
existing evaluation metrics measure idiom quality poorly with Pearson
correlation below 0.48 with human ratings. We thus develop improved models that
achieve F$_1$ scores of 0.68 for detecting idiom translation errors.

### Cryptography and Security

### 1. [BERTector: Intrusion Detection Based on Joint-Dataset Learning](http://arxiv.org/pdf/2508.10327v1)

Authors: Haoyang Hu, Xun Huang, Chenyu Wu, Shiwen Liu, Zhichao Lian, Shuangquan Zhang

Intrusion detection systems (IDS) are facing challenges in generalization and
robustness due to the heterogeneity of network traffic and the diversity of
attack patterns. To address this issue, we propose a new joint-dataset training
paradigm for IDS and propose a scalable BERTector framework based on BERT.
BERTector integrates three key components: NSS-Tokenizer for traffic-aware
semantic tokenization, supervised fine-tuning with a hybrid dataset, and
low-rank adaptation (LoRA) for efficient training. Extensive experiments show
that BERTector achieves state-of-the-art detection accuracy, strong
cross-dataset generalization capabilities, and excellent robustness to
adversarial perturbations. This work establishes a unified and efficient
solution for modern IDS in complex and dynamic network environments.

### 2. [Yet Another Mirage of Breaking MIRAGE: Debunking Occupancy-based Side-Channel Attacks on Fully Associative Randomized Caches](http://arxiv.org/pdf/2508.10431v1)

Authors: Chris Cao, Gururaj Saileshwar

Recent work presented at USENIX Security 2025 claims that occupancy-based
attacks can recover AES keys from the MIRAGE randomized cache. In this paper,
we examine these claims and find that they arise from fundamental modeling
flaws. Most critically, the authors' simulation of MIRAGE uses a constant seed
to initialize the random number generator used for global evictions in MIRAGE,
causing every AES encryption they trace to evict the same deterministic
sequence of cache lines. This artificially creates a highly repeatable timing
pattern that is not representative of a realistic implementation of MIRAGE,
where eviction sequences vary randomly between encryptions. When we instead
randomize the eviction seed for each run, reflecting realistic operation, the
correlation between AES T-table accesses and attacker runtimes disappears, and
the attack fails. These findings show that the reported leakage is an artifact
of incorrect modeling, and not an actual vulnerability in MIRAGE.

### 3. [Codes on any Cayley Graph have an Interactive Oracle Proof of Proximity](http://arxiv.org/pdf/2508.10510v1)

Authors: Hugo Delavenne, Louise Lallemand

Interactive Oracle Proofs of Proximity (IOPP) are at the heart of code-based
SNARKs, a family of zeroknowledge protocols. The first and most famous one is
the FRI protocol [BBHR18a], that efficiently tests proximity to Reed-Solomon
codes. This paper generalizes the flowering IOPP introduced in [DMR25] for some
specific (2, n)-regular Tanner codes to a much broader variety of codes: any
code with symbols indexed on the edges of a Cayley graph. The flowering
protocol of [DMR25] had a soundness parameter much lower than the FRI protocol
[BCI + 23], and complexity parameters that could compete with the FRI
[BBHR18a]. The lower soundness and the absence of restriction on the base field
may lead to other practical speedups, however the codes considered in [DMR25]
have an o(1) minimum distance. The generalization proposed in this paper
preserves the soundness parameter with a slight decrease of the complexity
parameters, while allowing being applied on codes with constant rate and
constant minimum distance thanks to the good expansion properties of some
families of Cayley graphs.

### 4. [MirGuard: Towards a Robust Provenance-based Intrusion Detection System Against Graph Manipulation Attacks](http://arxiv.org/pdf/2508.10639v1)

Authors: Anyuan Sang, Lu Zhou, Li Yang, Junbo Jia, Huipeng Yang, Pengbin Feng, Jianfeng Ma

Learning-based Provenance-based Intrusion Detection Systems (PIDSes) have
become essential tools for anomaly detection in host systems due to their
ability to capture rich contextual and structural information, as well as their
potential to detect unknown attacks. However, recent studies have shown that
these systems are vulnerable to graph manipulation attacks, where attackers
manipulate the graph structure to evade detection. While some previous
approaches have discussed this type of attack, none have fully addressed it
with a robust detection solution, limiting the practical applicability of
PIDSes.
  To address this challenge, we propose MirGuard, a robust anomaly detection
framework that combines logic-aware multi-view augmentation with contrastive
representation learning. Rather than applying arbitrary structural
perturbations, MirGuard introduces Logic-Aware Noise Injection (LNI) to
generate semantically valid graph views, ensuring that all augmentations
preserve the underlying causal semantics of the provenance data. These views
are then used in a Logic-Preserving Contrastive Learning framework, which
encourages the model to learn representations that are invariant to benign
transformations but sensitive to adversarial inconsistencies. Comprehensive
evaluations on multiple provenance datasets demonstrate that MirGuard
significantly outperforms state-of-the-art detectors in robustness against
various graph manipulation attacks without sacrificing detection performance
and efficiency. Our work represents the first targeted study to enhance PIDS
against such adversarial threats, providing a robust and effective solution to
modern cybersecurity challenges.

### 5. [Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts](http://arxiv.org/pdf/2508.10390v1)

Authors: Chiyu Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, Liming Fang, Zhe Liu

Evaluating jailbreak attacks is challenging when prompts are not overtly
harmful or fail to induce harmful outputs. Unfortunately, many existing
red-teaming datasets contain such unsuitable prompts. To evaluate attacks
accurately, these datasets need to be assessed and cleaned for maliciousness.
However, existing malicious content detection methods rely on either manual
annotation, which is labor-intensive, or large language models (LLMs), which
have inconsistent accuracy in harmful types. To balance accuracy and
efficiency, we propose a hybrid evaluation framework named MDH (Malicious
content Detection based on LLMs with Human assistance) that combines LLM-based
annotation with minimal human oversight, and apply it to dataset cleaning and
detection of jailbroken responses. Furthermore, we find that well-crafted
developer messages can significantly boost jailbreak success, leading us to
propose two new strategies: D-Attack, which leverages context simulation, and
DH-CoT, which incorporates hijacked chains of thought. The Codes, datasets,
judgements, and detection results will be released in github repository:
https://github.com/AlienZhang1996/DH-CoT.

### 6. [AlDBaran: Towards Blazingly Fast State Commitments for Blockchains](http://arxiv.org/pdf/2508.10493v1)

Authors: Bernhard Kauer, Aleksandr Petrosyan, Benjamin Livshits

The fundamental basis for maintaining integrity within contemporary
blockchain systems is provided by authenticated databases. Our analysis
indicates that a significant portion of the approaches applied in this domain
fail to sufficiently meet the stringent requirements of systems processing
transactions at rates of multi-million TPS. AlDBaran signifies a substantial
advancement in authenticated databases. By eliminating disk I/O operations from
the critical path, implementing prefetching strategies, and refining the update
mechanism of the Merkle tree, we have engineered an authenticated data
structure capable of handling state updates efficiently at a network throughput
of 50 Gbps. This throughput capacity significantly surpasses any empirically
documented blockchain throughput, guaranteeing the ability of even the most
high-throughput blockchains to generate state commitments effectively.
  AlDBaran provides support for historical state proofs, which facilitates a
wide array of novel applications. For instance, the deployment of AlDBaran
could enable blockchains that do not currently support state commitments to
offer functionalities for light clients and/or implement rollups.
  When benchmarked against alternative authenticated data structure projects,
AlDBaran exhibits superior performance and simplicity. In particular, AlDBaran
achieves speeds of approximately 48 million updates per second using an
identical machine configuration. This characteristic renders AlDBaran an
attractive solution for resource-limited environments, as its historical data
capabilities can be modularly isolated (and deactivated), which further
enhances performance. On consumer-level portable hardware, it achieves
approximately 8 million updates/s in an in-memory setting and 5 million
updates/s with snapshots at sub-second intervals, illustrating compelling and
cost-effective scalability.

### 7. [Bistochastically private release of longitudinal data](http://arxiv.org/pdf/2508.10606v1)

Authors: Nicolas Ruiz

Although the bulk of the research in privacy and statistical disclosure
control is designed for cross-sectional data, i.e. data where individuals are
observed at one single point in time, longitudinal data, i.e. individuals
observed over multiple periods, are increasingly collected. Such data enhance
undoubtedly the possibility of statistical analysis compared to cross-sectional
data, but also come with one additional layer of information, individual
trajectories, that must remain practically useful in a privacy-preserving way.
Few extensions, essentially k-anonymity based, of popular privacy tools have
been proposed to deal with the challenges posed by longitudinal data, and these
proposals are often complex. By considering randomized response, and
specifically its recent bistochastic extension, in the context of longitudinal
data, this paper proposes a simple approach for their anonymization. After
having characterized new results on bistochastic matrices, we show that a
simple relationship exists between the protection of each data set released at
each period, and the protection of individuals trajectories over time. In turn,
this relationship can be tuned according to desired protection and information
requirements. We illustrate the application of the proposed approach by an
empirical example.

### 8. [Advancing Autonomous Incident Response: Leveraging LLMs and Cyber Threat Intelligence](http://arxiv.org/pdf/2508.10677v1)

Authors: Amine Tellache, Abdelaziz Amara Korba, Amdjed Mokhtari, Horea Moldovan, Yacine Ghamri-Doudane

Effective incident response (IR) is critical for mitigating cyber threats,
yet security teams are overwhelmed by alert fatigue, high false-positive rates,
and the vast volume of unstructured Cyber Threat Intelligence (CTI) documents.
While CTI holds immense potential for enriching security operations, its
extensive and fragmented nature makes manual analysis time-consuming and
resource-intensive. To bridge this gap, we introduce a novel
Retrieval-Augmented Generation (RAG)-based framework that leverages Large
Language Models (LLMs) to automate and enhance IR by integrating dynamically
retrieved CTI. Our approach introduces a hybrid retrieval mechanism that
combines NLP-based similarity searches within a CTI vector database with
standardized queries to external CTI platforms, facilitating context-aware
enrichment of security alerts. The augmented intelligence is then leveraged by
an LLM-powered response generation module, which formulates precise,
actionable, and contextually relevant incident mitigation strategies. We
propose a dual evaluation paradigm, wherein automated assessment using an
auxiliary LLM is systematically cross-validated by cybersecurity experts.
Empirical validation on real-world and simulated alerts demonstrates that our
approach enhances the accuracy, contextualization, and efficiency of IR,
alleviating analyst workload and reducing response latency. This work
underscores the potential of LLM-driven CTI fusion in advancing autonomous
security operations and establishing a foundation for intelligent, adaptive
cybersecurity frameworks.

### 9. [SoK: Data Minimization in Machine Learning](http://arxiv.org/pdf/2508.10836v1)

Authors: Robin Staab, Nikola Jovanović, Kimberly Mai, Prakhar Ganesh, Martin Vechev, Ferdinando Fioretto, Matthew Jagielski

Data minimization (DM) describes the principle of collecting only the data
strictly necessary for a given task. It is a foundational principle across
major data protection regulations like GDPR and CPRA. Violations of this
principle have substantial real-world consequences, with regulatory actions
resulting in fines reaching hundreds of millions of dollars. Notably, the
relevance of data minimization is particularly pronounced in machine learning
(ML) applications, which typically rely on large datasets, resulting in an
emerging research area known as Data Minimization in Machine Learning (DMML).
At the same time, existing work on other ML privacy and security topics often
addresses concerns relevant to DMML without explicitly acknowledging the
connection. This disconnect leads to confusion among practitioners,
complicating their efforts to implement DM principles and interpret the
terminology, metrics, and evaluation criteria used across different research
communities. To address this gap, our work introduces a comprehensive framework
for DMML, including a unified data pipeline, adversaries, and points of
minimization. This framework allows us to systematically review the literature
on data minimization and \emph{DM-adjacent} methodologies, for the first time
presenting a structured overview designed to help practitioners and researchers
effectively apply DM principles. Our work facilitates a unified DM-centric
understanding and broader adoption of data minimization strategies in AI/ML.

### 10. [MM-Food-100K: A 100,000-Sample Multimodal Food Intelligence Dataset with Verifiable Provenance](http://arxiv.org/pdf/2508.10429v1)

Authors: Yi Dong, Yusuke Muraoka, Scott Shi, Yi Zhang

We present MM-Food-100K, a public 100,000-sample multimodal food intelligence
dataset with verifiable provenance. It is a curated approximately 10% open
subset of an original 1.2 million, quality-accepted corpus of food images
annotated for a wide range of information (such as dish name, region of
creation). The corpus was collected over six weeks from over 87,000
contributors using the Codatta contribution model, which combines community
sourcing with configurable AI-assisted quality checks; each submission is
linked to a wallet address in a secure off-chain ledger for traceability, with
a full on-chain protocol on the roadmap. We describe the schema, pipeline, and
QA, and validate utility by fine-tuning large vision-language models (ChatGPT
5, ChatGPT OSS, Qwen-Max) on image-based nutrition prediction. Fine-tuning
yields consistent gains over out-of-box baselines across standard metrics; we
report results primarily on the MM-Food-100K subset. We release MM-Food-100K
for publicly free access and retain approximately 90% for potential commercial
access with revenue sharing to contributors.

### Computer Vision and Pattern Recognition

### 1. [Deep Learning for Crack Detection: A Review of Learning Paradigms, Generalizability, and Datasets](http://arxiv.org/pdf/2508.10256v1)

Authors: Xinan Zhang, Haolin Wang, Yung-An Hsieh, Zhongyu Yang, Anthony Yezzi, Yi-Chang Tsai

Crack detection plays a crucial role in civil infrastructures, including
inspection of pavements, buildings, etc., and deep learning has significantly
advanced this field in recent years. While numerous technical and review papers
exist in this domain, emerging trends are reshaping the landscape. These shifts
include transitions in learning paradigms (from fully supervised learning to
semi-supervised, weakly-supervised, unsupervised, few-shot, domain adaptation
and fine-tuning foundation models), improvements in generalizability (from
single-dataset performance to cross-dataset evaluation), and diversification in
dataset reacquisition (from RGB images to specialized sensor-based data). In
this review, we systematically analyze these trends and highlight
representative works. Additionally, we introduce a new dataset collected with
3D laser scans, 3DCrack, to support future research and conduct extensive
benchmarking experiments to establish baselines for commonly used deep learning
methodologies, including recent foundation models. Our findings provide
insights into the evolving methodologies and future directions in deep
learning-based crack detection. Project page:
https://github.com/nantonzhang/Awesome-Crack-Detection

### 2. [High Fidelity Text to Image Generation with Contrastive Alignment and Structural Guidance](http://arxiv.org/pdf/2508.10280v1)

Authors: Danyi Gao

This paper addresses the performance bottlenecks of existing text-driven
image generation methods in terms of semantic alignment accuracy and structural
consistency. A high-fidelity image generation method is proposed by integrating
text-image contrastive constraints with structural guidance mechanisms. The
approach introduces a contrastive learning module that builds strong
cross-modal alignment constraints to improve semantic matching between text and
image. At the same time, structural priors such as semantic layout maps or edge
sketches are used to guide the generator in spatial-level structural modeling.
This enhances the layout completeness and detail fidelity of the generated
images. Within the overall framework, the model jointly optimizes contrastive
loss, structural consistency loss, and semantic preservation loss. A
multi-objective supervision mechanism is adopted to improve the semantic
consistency and controllability of the generated content. Systematic
experiments are conducted on the COCO-2014 dataset. Sensitivity analyses are
performed on embedding dimensions, text length, and structural guidance
strength. Quantitative metrics confirm the superior performance of the proposed
method in terms of CLIP Score, FID, and SSIM. The results show that the method
effectively bridges the gap between semantic alignment and structural fidelity
without increasing computational complexity. It demonstrates a strong ability
to generate semantically clear and structurally complete images, offering a
viable technical path for joint text-image modeling and image generation.

### 3. [VIFSS: View-Invariant and Figure Skating-Specific Pose Representation Learning for Temporal Action Segmentation](http://arxiv.org/pdf/2508.10281v1)

Authors: Ryota Tanaka, Tomohiro Suzuki, Keisuke Fujii

Understanding human actions from videos plays a critical role across various
domains, including sports analytics. In figure skating, accurately recognizing
the type and timing of jumps a skater performs is essential for objective
performance evaluation. However, this task typically requires expert-level
knowledge due to the fine-grained and complex nature of jump procedures. While
recent approaches have attempted to automate this task using Temporal Action
Segmentation (TAS), there are two major limitations to TAS for figure skating:
the annotated data is insufficient, and existing methods do not account for the
inherent three-dimensional aspects and procedural structure of jump actions. In
this work, we propose a new TAS framework for figure skating jumps that
explicitly incorporates both the three-dimensional nature and the semantic
procedure of jump movements. First, we propose a novel View-Invariant, Figure
Skating-Specific pose representation learning approach (VIFSS) that combines
contrastive learning as pre-training and action classification as fine-tuning.
For view-invariant contrastive pre-training, we construct FS-Jump3D, the first
publicly available 3D pose dataset specialized for figure skating jumps.
Second, we introduce a fine-grained annotation scheme that marks the ``entry
(preparation)'' and ``landing'' phases, enabling TAS models to learn the
procedural structure of jumps. Extensive experiments demonstrate the
effectiveness of our framework. Our method achieves over 92% F1@50 on
element-level TAS, which requires recognizing both jump types and rotation
levels. Furthermore, we show that view-invariant contrastive pre-training is
particularly effective when fine-tuning data is limited, highlighting the
practicality of our approach in real-world scenarios.

### 4. [JRDB-Reasoning: A Difficulty-Graded Benchmark for Visual Reasoning in Robotics](http://arxiv.org/pdf/2508.10287v1)

Authors: Simindokht Jahangard, Mehrzad Mohammadi, Yi Shen, Zhixi Cai, Hamid Rezatofighi

Recent advances in Vision-Language Models (VLMs) and large language models
(LLMs) have greatly enhanced visual reasoning, a key capability for embodied AI
agents like robots. However, existing visual reasoning benchmarks often suffer
from several limitations: they lack a clear definition of reasoning complexity,
offer have no control to generate questions over varying difficulty and task
customization, and fail to provide structured, step-by-step reasoning
annotations (workflows). To bridge these gaps, we formalize reasoning
complexity, introduce an adaptive query engine that generates customizable
questions of varying complexity with detailed intermediate annotations, and
extend the JRDB dataset with human-object interaction and geometric
relationship annotations to create JRDB-Reasoning, a benchmark tailored for
visual reasoning in human-crowded environments. Our engine and benchmark enable
fine-grained evaluation of visual reasoning frameworks and dynamic assessment
of visual-language models across reasoning levels.

### 5. [A Sub-Pixel Multimodal Optical Remote Sensing Images Matching Method](http://arxiv.org/pdf/2508.10294v1)

Authors: Tao Huang, Hongbo Pan, Nanxi Zhou, Shun Zhou

High-accuracy matching of multimodal optical images is the basis of geometric
processing. However, the image matching accuracy is usually degraded by the
nonlinear radiation and geometric deformation differences caused by different
spectral responses. To address these problems, we proposed a phase consistency
weighted least absolute deviation (PCWLAD) sub-pixel template matching method
to improve the matching accuracy of multimodal optical images. This method
consists of two main steps: coarse matching with the structural similarity
index measure (SSIM) and fine matching with WLAD. In the coarse matching step,
PCs are calculated without a noise filter to preserve the original structural
details, and template matching is performed using the SSIM. In the fine
matching step, we applied the radiometric and geometric transformation models
between two multimodal PC templates based on the coarse matching. Furthermore,
mutual structure filtering is adopted in the model to mitigate the impact of
noise within the corresponding templates on the structural consistency, and the
WLAD criterion is used to estimate the sub-pixel offset. To evaluate the
performance of PCWLAD, we created three types of image datasets: visible to
infrared Landsat images, visible to near-infrared close-range images, and
visible to infrared uncrewed aerial vehicle (UAV) images. PCWLAD outperformed
existing state-of-the-art eight methods in terms of correct matching rate (CMR)
and root mean square error (RMSE) and reached an average matching accuracy of
approximately 0.4 pixels across all three datasets. Our software and datasets
are publicly available at https://github.com/huangtaocsu/PCWLAD.

### 6. [InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild](http://arxiv.org/pdf/2508.10297v1)

Authors: Yiyi Ma, Yuanzhi Liang, Xiu Li, Chi Zhang, Xuelong Li

We present Interleaved Learning for Motion Synthesis (InterSyn), a novel
framework that targets the generation of realistic interaction motions by
learning from integrated motions that consider both solo and multi-person
dynamics. Unlike previous methods that treat these components separately,
InterSyn employs an interleaved learning strategy to capture the natural,
dynamic interactions and nuanced coordination inherent in real-world scenarios.
Our framework comprises two key modules: the Interleaved Interaction Synthesis
(INS) module, which jointly models solo and interactive behaviors in a unified
paradigm from a first-person perspective to support multiple character
interactions, and the Relative Coordination Refinement (REC) module, which
refines mutual dynamics and ensures synchronized motions among characters.
Experimental results show that the motion sequences generated by InterSyn
exhibit higher text-to-motion alignment and improved diversity compared with
recent methods, setting a new benchmark for robust and natural motion
synthesis. Additionally, our code will be open-sourced in the future to promote
further research and development in this area.

### 7. [From Pixel to Mask: A Survey of Out-of-Distribution Segmentation](http://arxiv.org/pdf/2508.10309v1)

Authors: Wenjie Zhao, Jia Li, Yunhui Guo

Out-of-distribution (OoD) detection and segmentation have attracted growing
attention as concerns about AI security rise. Conventional OoD detection
methods identify the existence of OoD objects but lack spatial localization,
limiting their usefulness in downstream tasks. OoD segmentation addresses this
limitation by localizing anomalous objects at pixel-level granularity. This
capability is crucial for safety-critical applications such as autonomous
driving, where perception modules must not only detect but also precisely
segment OoD objects, enabling targeted control actions and enhancing overall
system robustness. In this survey, we group current OoD segmentation approaches
into four categories: (i) test-time OoD segmentation, (ii) outlier exposure for
supervised training, (iii) reconstruction-based methods, (iv) and approaches
that leverage powerful models. We systematically review recent advances in OoD
segmentation for autonomous-driving scenarios, identify emerging challenges,
and discuss promising future research directions.

### 8. [Integrating Reinforcement Learning with Visual Generative Models: Foundations and Advances](http://arxiv.org/pdf/2508.10316v1)

Authors: Yuanzhi Liang, Yijie Fang, Rui Li, Ziqi Ni, Ruijie Su, Chi Zhang, Xuelong Li

Generative models have made significant progress in synthesizing visual
content, including images, videos, and 3D/4D structures. However, they are
typically trained with surrogate objectives such as likelihood or
reconstruction loss, which often misalign with perceptual quality, semantic
accuracy, or physical realism. Reinforcement learning (RL) offers a principled
framework for optimizing non-differentiable, preference-driven, and temporally
structured objectives. Recent advances demonstrate its effectiveness in
enhancing controllability, consistency, and human alignment across generative
tasks. This survey provides a systematic overview of RL-based methods for
visual content generation. We review the evolution of RL from classical control
to its role as a general-purpose optimization tool, and examine its integration
into image, video, and 3D/4D generation. Across these domains, RL serves not
only as a fine-tuning mechanism but also as a structural component for aligning
generation with complex, high-level goals. We conclude with open challenges and
future research directions at the intersection of RL and generative modeling.

### 9. [Glo-DMU: A Deep Morphometry Framework of Ultrastructural Characterization in Glomerular Electron Microscopic Images](http://arxiv.org/pdf/2508.10351v1)

Authors: Zhentai Zhang, Danyi Weng, Guibin Zhang, Xiang Chen, Kaixing Long, Jian Geng, Yanmeng Lu, Lei Zhang, Zhitao Zhou, Lei Cao

Complex and diverse ultrastructural features can indicate the type,
progression, and prognosis of kidney diseases. Recently, computational
pathology combined with deep learning methods has shown tremendous potential in
advancing automatic morphological analysis of glomerular ultrastructure.
However, current research predominantly focuses on the recognition of
individual ultrastructure, which makes it challenging to meet practical
diagnostic needs. In this study, we propose the glomerular morphometry
framework of ultrastructural characterization (Glo-DMU), which is grounded on
three deep models: the ultrastructure segmentation model, the glomerular
filtration barrier region classification model, and the electron-dense deposits
detection model. Following the conventional protocol of renal biopsy diagnosis,
this framework simultaneously quantifies the three most widely used
ultrastructural features: the thickness of glomerular basement membrane, the
degree of foot process effacement, and the location of electron-dense deposits.
We evaluated the 115 patients with 9 renal pathological types in real-world
diagnostic scenarios, demonstrating good consistency between automatic
quantification results and morphological descriptions in the pathological
reports. Glo-DMU possesses the characteristics of full automation, high
precision, and high throughput, quantifying multiple ultrastructural features
simultaneously, and providing an efficient tool for assisting renal
pathologists.

### 10. [AtomDiffuser: Time-Aware Degradation Modeling for Drift and Beam Damage in STEM Imaging](http://arxiv.org/pdf/2508.10359v1)

Authors: Hao Wang, Hongkui Zheng, Kai He, Abolfazl Razi

Scanning transmission electron microscopy (STEM) plays a critical role in
modern materials science, enabling direct imaging of atomic structures and
their evolution under external interferences. However, interpreting
time-resolved STEM data remains challenging due to two entangled degradation
effects: spatial drift caused by mechanical and thermal instabilities, and
beam-induced signal loss resulting from radiation damage. These factors distort
both geometry and intensity in complex, temporally correlated ways, making it
difficult for existing methods to explicitly separate their effects or model
material dynamics at atomic resolution. In this work, we present AtomDiffuser,
a time-aware degradation modeling framework that disentangles sample drift and
radiometric attenuation by predicting an affine transformation and a spatially
varying decay map between any two STEM frames. Unlike traditional denoising or
registration pipelines, our method leverages degradation as a physically
heuristic, temporally conditioned process, enabling interpretable structural
evolutions across time. Trained on synthetic degradation processes,
AtomDiffuser also generalizes well to real-world cryo-STEM data. It further
supports high-resolution degradation inference and drift alignment, offering
tools for visualizing and quantifying degradation patterns that correlate with
radiation-induced atomic instabilities.

### Computers and Society

### 1. [Ask ChatGPT: Caveats and Mitigations for Individual Users of AI Chatbots](http://arxiv.org/pdf/2508.10272v1)

Authors: Chengen Wang, Murat Kantarcioglu

As ChatGPT and other Large Language Model (LLM)-based AI chatbots become
increasingly integrated into individuals' daily lives, important research
questions arise. What concerns and risks do these systems pose for individual
users? What potential harms might they cause, and how can these be mitigated?
In this work, we review recent literature and reports, and conduct a
comprehensive investigation into these questions. We begin by explaining how
LLM-based AI chatbots work, providing essential background to help readers
understand chatbots' inherent limitations. We then identify a range of risks
associated with individual use of these chatbots, including hallucinations,
intrinsic biases, sycophantic behavior, cognitive decline from overreliance,
social isolation, and privacy leakage. Finally, we propose several key
mitigation strategies to address these concerns. Our goal is to raise awareness
of the potential downsides of AI chatbot use, and to empower users to enhance,
rather than diminish, human intelligence, to enrich, rather than compromise,
daily life.

### 2. [Beyond Self-Regulated Learning Processes: Unveiling Hidden Tactics in Generative AI-Assisted Writing](http://arxiv.org/pdf/2508.10310v1)

Authors: Kaixun Yang, Yizhou Fan, Luzhen Tang, Mladen Raković, Xinyu Li, Dragan Gašević, Guanliang Chen

The integration of Generative AI (GenAI) into education is reshaping how
students learn, making self-regulated learning (SRL) - the ability to plan,
monitor, and adapt one's learning - more important than ever. To support
learners in these new contexts, it is essential to understand how SRL unfolds
during interaction with GenAI tools. Learning analytics offers powerful
techniques for analyzing digital trace data to infer SRL behaviors. However,
existing approaches often assume SRL processes are linear, segmented, and
non-overlapping-assumptions that overlook the dynamic, recursive, and
non-linear nature of real-world learning. We address this by conceptualizing
SRL as a layered system: observable learning patterns reflect hidden tactics
(short, purposeful action states), which combine into broader SRL strategies.
Using Hidden Markov Models (HMMs), we analyzed trace data from higher education
students engaged in GenAI-assisted academic writing. We identified three
distinct groups of learners, each characterized by different SRL strategies.
These groups showed significant differences in performance, indicating that
students' use of different SRL strategies in GenAI-assisted writing led to
varying task outcomes. Our findings advance the methodological toolkit for
modeling SRL and inform the design of adaptive learning technologies that more
effectively support learners in GenAI-enhanced educational environments.

### 3. [Online Homogeneity Can Emerge Without Filtering Algorithms or Homophily Preferences](http://arxiv.org/pdf/2508.10466v1)

Authors: Petter Törnberg

Ideologically homogeneous online environments - often described as "echo
chambers" or "filter bubbles" - are widely seen as drivers of polarization,
radicalization, and misinformation. A central debate asks whether such
homophily stems primarily from algorithmic curation or users' preference for
like-minded peers. This study challenges that view by showing that homogeneity
can emerge in the absence of both filtering algorithms and user preferences.
Using an agent-based model inspired by Schelling's model of residential
segregation, we demonstrate that weak individual preferences, combined with
simple group-based interaction structures, can trigger feedback loops that
drive communities toward segregation. Once a small imbalance forms, cascades of
user exits and regrouping amplify homogeneity across the system.
Counterintuitively, algorithmic filtering - often blamed for "filter bubbles" -
can in fact sustain diversity by stabilizing mixed communities. These findings
highlight online polarization as an emergent system-level dynamic and
underscore the importance of applying a complexity lens to the study of digital
public spheres.

### 4. [Motive-level Analysis of Form-functions Association in Korean Folk song](http://arxiv.org/pdf/2508.10472v1)

Authors: Danbinaerin Han, Dasaem Jeong, Juhan Nam

Computational analysis of folk song audio is challenging due to structural
irregularities and the need for manual annotation. We propose a method for
automatic motive segmentation in Korean folk songs by fine-tuning a speech
transcription model on audio lyric with motif boundary annotation. Applying
this to 856 songs, we extracted motif count and duration entropy as structural
features. Statistical analysis revealed that these features vary systematically
according to the social function of the songs. Songs associated with collective
labor, for instance, showed different structural patterns from those for
entertainment or personal settings. This work offers a scalable approach for
quantitative structural analysis of oral music traditions.

### 5. [STAMP: Multi-pattern Attention-aware Multiple Instance Learning for STAS Diagnosis in Multi-center Histopathology Images](http://arxiv.org/pdf/2508.10473v1)

Authors: Liangrui Pan, xiaoyu Li, Guang Zhu, Guanting Li, Ruixin Wang, Jiadi Luo, Yaning Yang, Liang qingchun, Shaoliang Peng

Spread through air spaces (STAS) constitutes a novel invasive pattern in lung
adenocarcinoma (LUAD), associated with tumor recurrence and diminished survival
rates. However, large-scale STAS diagnosis in LUAD remains a labor-intensive
endeavor, compounded by the propensity for oversight and misdiagnosis due to
its distinctive pathological characteristics and morphological features.
Consequently, there is a pressing clinical imperative to leverage deep learning
models for STAS diagnosis. This study initially assembled histopathological
images from STAS patients at the Second Xiangya Hospital and the Third Xiangya
Hospital of Central South University, alongside the TCGA-LUAD cohort. Three
senior pathologists conducted cross-verification annotations to construct the
STAS-SXY, STAS-TXY, and STAS-TCGA datasets. We then propose a multi-pattern
attention-aware multiple instance learning framework, named STAMP, to analyze
and diagnose the presence of STAS across multi-center histopathology images.
Specifically, the dual-branch architecture guides the model to learn
STAS-associated pathological features from distinct semantic spaces.
Transformer-based instance encoding and a multi-pattern attention aggregation
modules dynamically selects regions closely associated with STAS pathology,
suppressing irrelevant noise and enhancing the discriminative power of global
representations. Moreover, a similarity regularization constraint prevents
feature redundancy across branches, thereby improving overall diagnostic
accuracy. Extensive experiments demonstrated that STAMP achieved competitive
diagnostic results on STAS-SXY, STAS-TXY and STAS-TCGA, with AUCs of 0.8058,
0.8017, and 0.7928, respectively, surpassing the clinical level.

### 6. ["I Want My Chart to Be Just for Me": Community-Engaged Design to Support Outpatient Healthcare for Resettled Communities](http://arxiv.org/pdf/2508.10757v1)

Authors: Zhanming Chen, Juan F. Maestre, May Hang, Alisha Ghaju, Ji Youn Shin

Individuals resettled in a new environment often face challenges in accessing
adequate healthcare services, particularly within the complex processes of
outpatient clinic care. Cultural differences, language barriers, and low
socioeconomic status contribute to these difficulties. While previous studies
have identified barriers and proposed technology-mediated solutions for
resettled populations, many focus on addressing deficits rather than building
on the strengths these communities already possess, which limits the
sustainability and relevance of these solutions in everyday life. We conducted
two community-based participatory design workshops with 30 Hmong community
members in a large metropolitan area in the US. Through this process, we
identified four types of assets the community has gradually developed,
including intergenerational support for health management and
storytelling-based communication practices that facilitate relatable and
culturally grounded interactions. We show how participatory design workshops
can foster asset-based approaches, and discuss design implications for
technologies that leverage patients' existing strengths to support their health
management during outpatient visits.

### 7. [Facilitating Longitudinal Interaction Studies of AI Systems](http://arxiv.org/pdf/2508.10252v1)

Authors: Tao Long, Sitong Wang, Émilie Fabre, Tony Wang, Anup Sathya, Jason Wu, Savvas Petridis, Dingzeyu Li, Tuhin Chakrabarty, Yue Jiang, Jingyi Li, Tiffany Tseng, Ken Nakagaki, Qian Yang, Nikolas Martelaro, Jeffrey V. Nickerson, Lydia B. Chilton

UIST researchers develop tools to address user challenges. However, user
interactions with AI evolve over time through learning, adaptation, and
repurposing, making one time evaluations insufficient. Capturing these dynamics
requires longer-term studies, but challenges in deployment, evaluation design,
and data collection have made such longitudinal research difficult to
implement. Our workshop aims to tackle these challenges and prepare researchers
with practical strategies for longitudinal studies. The workshop includes a
keynote, panel discussions, and interactive breakout groups for discussion and
hands-on protocol design and tool prototyping sessions. We seek to foster a
community around longitudinal system research and promote it as a more embraced
method for designing, building, and evaluating UIST tools.

### 8. [Welfare-Centric Clustering](http://arxiv.org/pdf/2508.10345v1)

Authors: Claire Jie Zhang, Seyed A. Esmaeili, Jamie Morgenstern

Fair clustering has traditionally focused on ensuring equitable group
representation or equalizing group-specific clustering costs. However,
Dickerson et al. (2025) recently showed that these fairness notions may yield
undesirable or unintuitive clustering outcomes and advocated for a
welfare-centric clustering approach that models the utilities of the groups. In
this work, we model group utilities based on both distances and proportional
representation and formalize two optimization objectives based on
welfare-centric clustering: the Rawlsian (Egalitarian) objective and the
Utilitarian objective. We introduce novel algorithms for both objectives and
prove theoretical guarantees for them. Empirical evaluations on multiple
real-world datasets demonstrate that our methods significantly outperform
existing fair clustering baselines.

### 9. [Traffic Intersection Simulation Using Turning Movement Count Data in SUMO: A Case Study of Toronto Intersections](http://arxiv.org/pdf/2508.10733v1)

Authors: Harshit Maheshwari, Li Yang, Richard W Pazzi

Urban traffic simulation is vital in planning, modeling, and analyzing road
networks. However, the realism of a simulation depends extensively on the
quality of input data. This paper presents an intersection traffic simulation
tool that leverages real-world vehicle turning movement count (TMC) data from
the City of Toronto to model traffic in an urban environment at an individual
or multiple intersections using Simulation of Urban MObility (SUMO). The
simulation performed in this research focuses specifically on
intersection-level traffic generation without creating full vehicle routes
through the network. This also helps keep the network's complexity to a
minimum. The simulated traffic is evaluated against actual data to show that
the simulation closely reproduces real intersection flows. This validates that
the real data can drive practical simulations, and these scenarios can replace
synthetic or random generated data, which is prominently used in developing new
traffic-related methodologies. This is the first tool to integrate TMC data
from Toronto into SUMO via an easy-to-use Graphical User Interface. This work
contributes to the research and traffic planning community on data-driven
traffic simulation. It provides transportation engineers with a framework to
evaluate intersection design and traffic signal optimization strategies using
readily available aggregate traffic data.

### Databases

### 1. [Privacy-Preserving Approximate Nearest Neighbor Search on High-Dimensional Data](http://arxiv.org/pdf/2508.10373v1)

Authors: Yingfan Liu, Yandi Zhang, Jiadong Xie, Hui Li, Jeffrey Xu Yu, Jiangtao Cui

In the era of cloud computing and AI, data owners outsource ubiquitous
vectors to the cloud, which furnish approximate $k$-nearest neighbors
($k$-ANNS) services to users. To protect data privacy against the untrusted
server, privacy-preserving $k$-ANNS (PP-ANNS) on vectors has been a fundamental
and urgent problem. However, existing PP-ANNS solutions fall short of meeting
the requirements of data privacy, efficiency, accuracy, and minimal user
involvement concurrently. To tackle this challenge, we introduce a novel
solution that primarily executes PP-ANNS on a single cloud server to avoid the
heavy communication overhead between the cloud and the user. To ensure data
privacy, we introduce a novel encryption method named distance comparison
encryption, facilitating secure, efficient, and exact distance comparisons. To
optimize the trade-off between data privacy and search performance, we design a
privacy-preserving index that combines the state-of-the-art $k$-ANNS method
with an approximate distance computation method. Then, we devise a search
method using a filter-and-refine strategy based on the index. Moreover, we
provide the security analysis of our solution and conduct extensive experiments
to demonstrate its superiority over existing solutions. Based on our
experimental results, our method accelerates PP-ANNS by up to 3 orders of
magnitude compared to state-of-the-art methods, while not compromising the
accuracy.

### 2. [Cross-Organizational Analysis of Parliamentary Processes: A Case Study](http://arxiv.org/pdf/2508.10381v1)

Authors: Paul-Julius Hillmann, Stephan A. Fahrenkrog-Petersen, Jan Mendling

Process Mining has been widely adopted by businesses and has been shown to
help organizations analyze and optimize their processes. However, so far,
little attention has gone into the cross-organizational comparison of
processes, since many companies are hesitant to share their data. In this
paper, we explore the processes of German state parliaments that are often
legally required to share their data and run the same type of processes for
different geographical regions. This paper is the first attempt to apply
process mining to parliamentary processes and, therefore, contributes toward a
novel interdisciplinary research area that combines political science and
process mining. In our case study, we analyze legislative processes of three
German state parliaments and generate insights into their differences and best
practices. We provide a discussion of the relevance of our results that are
based on knowledge exchange with a political scientist and a domain expert from
the German federal parliament.

### 3. [Emerging Skycube](http://arxiv.org/pdf/2508.10516v1)

Authors: Mickaël Martin Nevot

Combining multi-criteria decision analysis and trend reversal discovery make
it possible to extract globally optimal, or non-dominated, data in relation to
several criteria, and then to observe their evolution according to a
decision-making property. Thus, we introduce Emerging Skycube, a concept
associating Skycube and emerging datacube. As far as we know, no
DBMS-integrated solution exists to compute an emerging Skycube, and hence
taking advantage of ROLAP analysis tools. An emerging datacube has only one
measure: we propose to use several to comply to multi-criteria decision
analysis constraints which requires multiple attributes. A datacube is
expensive to compute. An emerging datacube is about twice as expensive. On the
other hand, an emerging Skycube is cheaper as the trend reversal is computed
after two Skycube calculations, which considerably reduces the relation volume
in comparison with the initial one. It is possible to save even more computing
time and storage space. To this end, we propose two successive reductions.
First, a Skycube lossless partial materialisation using Skylines concepts
lattice, based on the agree concepts lattice and partitions lattice. Then,
either the closed emerging Skycube for an information-loss reduction, or the
closed emerging L-Skycube for a smaller but lossless reduction.

### 4. [Efficient Methods for Accurate Sparse Trajectory Recovery and Map Matching](http://arxiv.org/pdf/2508.10460v1)

Authors: Wei Tian, Jieming Shi, Man Lung Yiu

Real-world trajectories are often sparse with low-sampling rates (i.e., long
intervals between consecutive GPS points) and misaligned with road networks,
yet many applications demand high-quality data for optimal performance. To
improve data quality with sparse trajectories as input, we systematically study
two related research problems: trajectory recovery on road network, which aims
to infer missing points to recover high-sampling trajectories, and map
matching, which aims to map GPS points to road segments to determine underlying
routes. In this paper, we present efficient methods TRMMA and MMA for accurate
trajectory recovery and map matching, respectively, where MMA serves as the
first step of TRMMA. In MMA, we carefully formulate a classification task to
map a GPS point from sparse trajectories to a road segment over a small
candidate segment set, rather than the entire road network. We develop
techniques in MMA to generate effective embeddings that capture the patterns of
GPS data, directional information, and road segments, to accurately align
sparse trajectories to routes. For trajectory recovery, TRMMA focuses on the
segments in the route returned by MMA to infer missing points with position
ratios on road segments, producing high-sampling trajectories efficiently by
avoiding evaluation of all road segments. Specifically, in TRMMA, we design a
dual-transformer encoding process to cohesively capture latent patterns in
trajectories and routes, and an effective decoding technique to sequentially
predict the position ratios and road segments of missing points. We conduct
extensive experiments to compare TRMMA and MMA with numerous existing methods
for trajectory recovery and map matching, respectively, on 4 large real-world
datasets. TRMMA and MMA consistently achieve the best result quality, often by
a significant margin.

### 5. [Advances in Logic-Based Entity Resolution: Enhancing ASPEN with Local Merges and Optimality Criteria](http://arxiv.org/pdf/2508.10504v1)

Authors: Zhliang Xiang, Meghyn Bienvenu, Gianluca Cima, Víctor Gutiérrez-Basulto, Yazmín Ibáñez-García

In this paper, we present ASPEN+, which extends an existing ASP-based system,
ASPEN,for collective entity resolution with two important functionalities:
support for local merges and new optimality criteria for preferred solutions.
Indeed, ASPEN only supports so-called global merges of entity-referring
constants (e.g. author ids), in which all occurrences of matched constants are
treated as equivalent and merged accordingly. However, it has been argued that
when resolving data values, local merges are often more appropriate, as e.g.
some instances of 'J. Lee' may refer to 'Joy Lee', while others should be
matched with 'Jake Lee'. In addition to allowing such local merges, ASPEN+
offers new optimality criteria for selecting solutions, such as minimizing rule
violations or maximising the number of rules supporting a merge. Our main
contributions are thus (1) the formalisation and computational analysis of
various notions of optimal solution, and (2) an extensive experimental
evaluation on real-world datasets, demonstrating the effect of local merges and
the new optimality criteria on both accuracy and runtime.

### Distributed, Parallel, and Cluster Computing

### 1. [GPZ: GPU-Accelerated Lossy Compressor for Particle Data](http://arxiv.org/pdf/2508.10305v1)

Authors: Ruoyu Li, Yafan Huang, Longtao Zhang, Zhuoxun Yang, Sheng Di, Jiajun Huang, Jinyang Liu, Jiannan Tian, Xin Liang, Guanpeng Li, Hanqi Guo, Franck Cappello, Kai Zhao

Particle-based simulations and point-cloud applications generate massive,
irregular datasets that challenge storage, I/O, and real-time analytics.
Traditional compression techniques struggle with irregular particle
distributions and GPU architectural constraints, often resulting in limited
throughput and suboptimal compression ratios. In this paper, we present GPZ, a
high-performance, error-bounded lossy compressor designed specifically for
large-scale particle data on modern GPUs. GPZ employs a novel four-stage
parallel pipeline that synergistically balances high compression efficiency
with the architectural demands of massively parallel hardware. We introduce a
suite of targeted optimizations for computation, memory access, and GPU
occupancy that enables GPZ to achieve near-hardware-limit throughput. We
conduct an extensive evaluation on three distinct GPU architectures
(workstation, data center, and edge) using six large-scale, real-world
scientific datasets from five distinct domains. The results demonstrate that
GPZ consistently and significantly outperforms five state-of-the-art GPU
compressors, delivering up to 8x higher end-to-end throughput while
simultaneously achieving superior compression ratios and data quality.

### 2. [Dalek: An Unconventional and Energy-Aware Heterogeneous Cluster](http://arxiv.org/pdf/2508.10481v1)

Authors: Adrien Cassagne, Noé Amiot, Manuel Bouyer

Dalek is an experimental compute cluster designed to evaluate the performance
of heterogeneous, consumer-grade hardware for software design, prototyping, and
algorithm development. In contrast to traditional computing centers that rely
on costly, server-class components, Dalek integrates CPUs and GPUs typically
found in mini-PCs, laptops, and gaming desktops, providing a cost-effective yet
versatile platform. This document details the cluster's architecture and
software stack, and presents results from synthetic benchmarks. Furthermore, it
introduces a custom energy monitoring platform capable of delivering 1000
averaged samples per second with milliwatt-level resolution. This
high-precision monitoring capability enables a wide range of energy-aware
research experiments in applied Computer Science.

### 3. [Minimmit: Fast Finality with Even Faster Blocks](http://arxiv.org/pdf/2508.10862v1)

Authors: Brendan Kobayashi Chou, Andrew Lewis-Pye, Patrick O'Grady

Minimmit is a new protocol for State-Machine-Replication (SMR) that extends
the '2-round finality' approach of protocols such as Alpenglow to further
reduce latency, by allowing for faster progression through 'views'. This
preliminary draft provides motivation and pseudocode, together with proofs of
consistency and liveness. An updated draft with a proof of optimistic
responsiveness, suggested optimizations, and experiments, is to follow.

### 4. [Flexible Personalized Split Federated Learning for On-Device Fine-Tuning of Foundation Models](http://arxiv.org/pdf/2508.10349v1)

Authors: Tianjun Yuan, Jiaxiang Geng, Pengchao Han, Xianhao Chen, Bing Luo

Fine-tuning foundation models is critical for superior performance on
personalized downstream tasks, compared to using pre-trained models.
Collaborative learning can leverage local clients' datasets for fine-tuning,
but limited client data and heterogeneous data distributions hinder effective
collaboration. To address the challenge, we propose a flexible personalized
federated learning paradigm that enables clients to engage in collaborative
learning while maintaining personalized objectives. Given the limited and
heterogeneous computational resources available on clients, we introduce
\textbf{flexible personalized split federated learning (FlexP-SFL)}. Based on
split learning, FlexP-SFL allows each client to train a portion of the model
locally while offloading the rest to a server, according to resource
constraints. Additionally, we propose an alignment strategy to improve
personalized model performance on global data. Experimental results show that
FlexP-SFL outperforms baseline models in personalized fine-tuning efficiency
and final accuracy.

### 5. [Introducing CQ: A C-like API for Quantum Accelerated HPC](http://arxiv.org/pdf/2508.10854v1)

Authors: Oliver Thomson Brown, Mateusz Meller, James Richings

In this paper we present CQ, a specification for a C-like API for quantum
accelerated HPC, as well as CQ-SimBE, a reference implementation of CQ written
in C99, and built on top of the statevector simulator QuEST. CQ focuses on
enabling the incremental integration of quantum computing into classical HPC
codes by supporting runtime offloading from languages such as C and Fortran. It
provides a way of describing and offloading quantum computations which is
compatible with strictly and strongly typed compiled languages, and gives the
programmer fine-grained control over classical data movement. The CQ Simulated
Backend (CQ-SimBE) provides both a way to demonstrate the usage and utility of
CQ, and a space to experiment with new features such as support for analogue
quantum computing. Both the CQ specification and CQ-SimBE are open-source, and
available in public repositories.

### Digital Libraries

### 1. [FIRESPARQL: A LLM-based Framework for SPARQL Query Generation over Scholarly Knowledge Graphs](http://arxiv.org/pdf/2508.10467v1)

Authors: Xueli Pan, Victor de Boer, Jacco van Ossenbruggen

Question answering over Scholarly Knowledge Graphs (SKGs) remains a
challenging task due to the complexity of scholarly content and the intricate
structure of these graphs. Large Language Model (LLM) approaches could be used
to translate natural language questions (NLQs) into SPARQL queries; however,
these LLM-based approaches struggle with SPARQL query generation due to limited
exposure to SKG-specific content and the underlying schema. We identified two
main types of errors in the LLM-generated SPARQL queries: (i) structural
inconsistencies, such as missing or redundant triples in the queries, and (ii)
semantic inaccuracies, where incorrect entities or properties are shown in the
queries despite a correct query structure. To address these issues, we propose
FIRESPARQL, a modular framework that supports fine-tuned LLMs as a core
component, with optional context provided via retrieval-augmented generation
(RAG) and a SPARQL query correction layer. We evaluate the framework on the
SciQA Benchmark using various configurations (zero-shot, zero-shot with RAG,
one-shot, fine-tuning, and fine-tuning with RAG) and compare the performance
with baseline and state-of-the-art approaches. We measure query accuracy using
BLEU and ROUGE metrics, and query result accuracy using relaxed exact
match(RelaxedEM), with respect to the gold standards containing the NLQs,
SPARQL queries, and the results of the queries. Experimental results
demonstrate that fine-tuning achieves the highest overall performance, reaching
0.90 ROUGE-L for query accuracy and 0.85 RelaxedEM for result accuracy on the
test set.

### Discrete Mathematics

### 1. [Localization game capture time of trees and outerplanar graphs](http://arxiv.org/pdf/2508.10443v1)

Authors: Vesna Iršič Chenoweth, Matija Skrt

The localization game is a variant of the game of Cops and Robber in which
the robber is invisible and moves between adjacent vertices, but the cops can
probe any $k$ vertices of the graph to obtain the distance between probed
vertices and the robber. The localization number of a graph is the minimum $k$
needed for cops to be able to locate the robber in finite time. The
localization capture time is the number of rounds needed for cops to win.
  The localization capture time conjecture claims that there exists a constant
$C$ such that the localization number of every connected graph on $n$ vertices
is at most $Cn$. While it is known that the conjecture holds for trees, in this
paper we significantly improve the known upper bound for the localization
capture time of trees. We also prove the conjecture for a subclass of
outerplanar graphs and present a generalization of the localization game that
appears useful for making further progress towards the conjecture.

### 2. [Spirals and Beyond: Competitive Plane Search with Multi-Speed Agents](http://arxiv.org/pdf/2508.10793v1)

Authors: Konstantinos Georgiou, Caleb Jones, Matthew Madej

We consider the problem of minimizing the worst-case search time for a hidden
point target in the plane using multiple mobile agents of differing speeds, all
starting from a common origin. The search time is normalized by the target's
distance to the origin, following the standard convention in competitive
analysis. The goal is to minimize the maximum such normalized time over all
target locations, the search cost. As a base case, we extend the known result
for a single unit-speed agent, which achieves an optimal cost of about
$\mathcal{U}_1 = 17.28935$ via a logarithmic spiral, to $n$ unit-speed agents.
We give a symmetric spiral-based algorithm where each agent follows a
logarithmic spiral offset by equal angular phases. This yields a search cost
independent of which agent finds the target. We provide a closed-form upper
bound $\mathcal{U}_n$ for this setting, which we use in our general result. Our
main contribution is an upper bound on the worst-case normalized search time
for $n$ agents with arbitrary speeds. We give a framework that selects a subset
of agents and assigns spiral-type trajectories with speed-dependent angular
offsets, again making the search cost independent of which agent reaches the
target. A corollary shows that $n$ multi-speed agents (fastest speed 1) can
beat $k$ unit-speed agents (cost below $\mathcal{U}_k$) if the geometric mean
of their speeds exceeds $\mathcal{U}_n / \mathcal{U}_k$. This means slow agents
may be excluded if they lower the mean too much, motivating non-spiral
algorithms. We also give new upper bounds for point search in cones and conic
complements using a single unit-speed agent. These are then used to design
hybrid spiral-directional strategies, which outperform the spiral-based
algorithms when some agents are slow. This suggests that spiral-type
trajectories may not be optimal in the general multi-speed setting.

### Data Structures and Algorithms

### 1. [Output-Sparse Matrix Multiplication Using Compressed Sensing](http://arxiv.org/pdf/2508.10250v1)

Authors: Huck Bennett, Karthik Gajulapalli, Alexander Golovnev, Evelyn Warton

We give two algorithms for output-sparse matrix multiplication (OSMM), the
problem of multiplying two $n \times n$ matrices $A, B$ when their product $AB$
is promised to have at most $O(n^{\delta})$ many non-zero entries for a given
value $\delta \in [0, 2]$. We then show how to speed up these algorithms in the
fully sparse setting, where the input matrices $A, B$ are themselves sparse.
All of our algorithms work over arbitrary rings.
  Our first, deterministic algorithm for OSMM works via a two-pass reduction to
compressed sensing. It runs in roughly $n^{\omega(\delta/2, 1, 1)}$ time, where
$\omega(\cdot, \cdot, \cdot)$ is the rectangular matrix multiplication
exponent. This substantially improves on prior deterministic algorithms for
output-sparse matrix multiplication.
  Our second, randomized algorithm for OSMM works via a reduction to compressed
sensing and a variant of matrix multiplication verification, and runs in
roughly $n^{\omega(\delta - 1, 1, 1)}$ time. This algorithm and its extension
to the fully sparse setting have running times that match those of the
(randomized) algorithms for OSMM and FSMM, respectively, in recent work of
Abboud, Bringmann, Fischer, and K\"{u}nnemann (SODA, 2024). Our algorithm uses
different techniques and is arguably simpler.
  Finally, we observe that the running time of our randomized algorithm and the
algorithm of Abboud et al. are optimal via a simple reduction from rectangular
matrix multiplication.

### 2. [On Fixed-Parameter Tractability of Weighted 0-1 Timed Matching Problem on Temporal Graphs](http://arxiv.org/pdf/2508.10562v1)

Authors: Rinku Kumar, Bodhisatwa Mazumdar, Subhrangsu Mandal

Temporal graphs are introduced to model systems where the relationships among
the entities of the system evolve over time. In this paper, we consider the
temporal graphs where the edge set changes with time and all the changes are
known a priori. The underlying graph of a temporal graph is a static graph
consisting of all the vertices and edges that exist for at least one timestep
in the temporal graph. The concept of 0-1 timed matching in temporal graphs was
introduced by Mandal and Gupta [DAM2022] as an extension of the matching
problem in static graphs. A 0-1 timed matching of a temporal graph is a
non-overlapping subset of the edge set of that temporal graph. The problem of
finding the maximum 0-1 timed matching is proved to be NP-complete on multiple
classes of temporal graphs. We study the fixed-parameter tractability of the
maximum 0-1 timed matching problem. We prove that the problem remains to be
NP-complete even when the underlying static graph of the temporal graph has a
bounded treewidth. Furthermore, we establish that the problem is W[1]-hard when
parameterized by the solution size. Finally, we present a fixed-parameter
tractable (FPT) algorithm to address the problem when the problem is
parameterized by the maximum vertex degree and the treewidth of the underlying
graph of the temporal graph.

### 3. [Competitively Consistent Clustering](http://arxiv.org/pdf/2508.10800v1)

Authors: Niv Buchbinder, Roie Levin, Yue Yang

In fully-dynamic consistent clustering, we are given a finite metric space
$(M,d)$, and a set $F\subseteq M$ of possible locations for opening centers.
Data points arrive and depart, and the goal is to maintain an approximately
optimal clustering solution at all times while minimizing the recourse, the
total number of additions/deletions of centers over time. Specifically, we
study fully dynamic versions of the classical $k$-center, facility location,
and $k$-median problems. We design algorithms that, given a parameter
$\beta\geq 1$, maintain an $O(\beta)$-approximate solution at all times, and
whose total recourse is bounded by $O(\log |F| \log \Delta) \cdot
\text{OPT}_\text{rec}^{\beta}$. Here $\text{OPT}_\text{rec}^{\beta}$ is the
minimal recourse of an offline algorithm that maintains a $\beta$-approximate
solution at all times, and $\Delta$ is the metric aspect ratio. Finally, while
we compare the performance of our algorithms to an optimal solution that
maintains $k$ centers, our algorithms are allowed to use slightly more than $k$
centers. We obtain our results via a reduction to the recently proposed
Positive Body Chasing framework of [Bhattacharya, Buchbinder, Levin, Saranurak,
FOCS 2023], which we show gives fractional solutions to our clustering problems
online. Our contribution is to round these fractional solutions while
preserving the approximation and recourse guarantees. We complement our
positive results with logarithmic lower bounds which show that our bounds are
nearly tight.

### 4. [Lower Bounds on Tree Covers](http://arxiv.org/pdf/2508.10376v1)

Authors: Yu Chen, Zihan Tan, Hangyu Xu

Given an $n$-point metric space $(X,d_X)$, a tree cover $\mathcal{T}$ is a
set of $|\mathcal{T}|=k$ trees on $X$ such that every pair of vertices in $X$
has a low-distortion path in one of the trees in $\mathcal{T}$. Tree covers
have been playing a crucial role in graph algorithms for decades, and the
research focus is the construction of tree covers with small size $k$ and
distortion.
  When $k=1$, the best distortion is known to be $\Theta(n)$. For a constant
$k\ge 2$, the best distortion upper bound is $\tilde O(n^{\frac 1 k})$ and the
strongest lower bound is $\Omega(\log_k n)$, leaving a gap to be closed. In
this paper, we improve the lower bound to $\Omega(n^{\frac{1}{2^{k-1}}})$.
  Our proof is a novel analysis on a structurally simple grid-like graph, which
utilizes some combinatorial fixed-point theorems. We believe that they will
prove useful for analyzing other tree-like data structures as well.

### 5. [Spirals and Beyond: Competitive Plane Search with Multi-Speed Agents](http://arxiv.org/pdf/2508.10793v1)

Authors: Konstantinos Georgiou, Caleb Jones, Matthew Madej

We consider the problem of minimizing the worst-case search time for a hidden
point target in the plane using multiple mobile agents of differing speeds, all
starting from a common origin. The search time is normalized by the target's
distance to the origin, following the standard convention in competitive
analysis. The goal is to minimize the maximum such normalized time over all
target locations, the search cost. As a base case, we extend the known result
for a single unit-speed agent, which achieves an optimal cost of about
$\mathcal{U}_1 = 17.28935$ via a logarithmic spiral, to $n$ unit-speed agents.
We give a symmetric spiral-based algorithm where each agent follows a
logarithmic spiral offset by equal angular phases. This yields a search cost
independent of which agent finds the target. We provide a closed-form upper
bound $\mathcal{U}_n$ for this setting, which we use in our general result. Our
main contribution is an upper bound on the worst-case normalized search time
for $n$ agents with arbitrary speeds. We give a framework that selects a subset
of agents and assigns spiral-type trajectories with speed-dependent angular
offsets, again making the search cost independent of which agent reaches the
target. A corollary shows that $n$ multi-speed agents (fastest speed 1) can
beat $k$ unit-speed agents (cost below $\mathcal{U}_k$) if the geometric mean
of their speeds exceeds $\mathcal{U}_n / \mathcal{U}_k$. This means slow agents
may be excluded if they lower the mean too much, motivating non-spiral
algorithms. We also give new upper bounds for point search in cones and conic
complements using a single unit-speed agent. These are then used to design
hybrid spiral-directional strategies, which outperform the spiral-based
algorithms when some agents are slow. This suggests that spiral-type
trajectories may not be optimal in the general multi-speed setting.

### 6. [Welfare-Centric Clustering](http://arxiv.org/pdf/2508.10345v1)

Authors: Claire Jie Zhang, Seyed A. Esmaeili, Jamie Morgenstern

Fair clustering has traditionally focused on ensuring equitable group
representation or equalizing group-specific clustering costs. However,
Dickerson et al. (2025) recently showed that these fairness notions may yield
undesirable or unintuitive clustering outcomes and advocated for a
welfare-centric clustering approach that models the utilities of the groups. In
this work, we model group utilities based on both distances and proportional
representation and formalize two optimization objectives based on
welfare-centric clustering: the Rawlsian (Egalitarian) objective and the
Utilitarian objective. We introduce novel algorithms for both objectives and
prove theoretical guarantees for them. Empirical evaluations on multiple
real-world datasets demonstrate that our methods significantly outperform
existing fair clustering baselines.

### 7. [Decoded Quantum Interferometry Under Noise](http://arxiv.org/pdf/2508.10725v1)

Authors: Kaifeng Bu, Weichen Gu, Dax Enshan Koh, Xiang Li

Decoded Quantum Interferometry (DQI) is a recently proposed quantum
optimization algorithm that exploits sparsity in the Fourier spectrum of
objective functions, with the potential for exponential speedups over classical
algorithms on suitably structured problems. While highly promising in idealized
settings, its resilience to noise has until now been largely unexplored. To
address this, we conduct a rigorous analysis of DQI under noise, focusing on
local depolarizing noise. For the maximum linear satisfiability problem, we
prove that, in the presence of noise, performance is governed by a
noise-weighted sparsity parameter of the instance matrix, with solution quality
decaying exponentially as sparsity decreases. We demonstrate this decay through
numerical simulations on two special cases: the Optimal Polynomial Intersection
problem and the Maximum XOR Satisfiability problem. The Fourier-analytic
methods we develop can be readily adapted to other classes of random Pauli
noise, making our framework applicable to a broad range of noisy quantum
settings and offering guidance on preserving DQI's potential quantum advantage
under realistic noise.

### Emerging Technologies

### 1. [Molecule Mixture Detection and Alphabet Design for Non-linear, Cross-reactive Receiver Arrays in MC](http://arxiv.org/pdf/2508.10856v1)

Authors: Bastian Heinlein, Kaikai Zhu, Sümeyye Carkit-Yilmaz, Sebastian Lotter, Helene M. Loos, Andrea Buettner, Yansha Deng, Robert Schober, Vahid Jamali

Air-based molecular communication (MC) has the potential to be one of the
first MC systems to be deployed in real-world applications, enabled by existing
sensor technologies such as metal-oxide semi-conductor (MOS) sensors. However,
commercially available sensors usually exhibit non-linear and cross-reactive
behavior, contrary to the idealizing assumptions about linear and perfectly
molecule type-specific sensing often made in the MC literature. To address this
gap, we propose a detector for molecule mixture communication with a general
non-linear, cross-reactive receiver (RX) array that performs approximate
maximum likelihood detection on the sensor outputs. Additionally, we introduce
an algorithm for the design of mixture alphabets that accounts for the RX
characteristics. We evaluate our detector and alphabet design algorithm through
simulations that are based on measurements reported for two commercial MOS
sensors. Our simulations demonstrate that the proposed detector achieves
similar symbol error rates as data-driven methods without requiring large
numbers of training samples and that the alphabet design algorithm outperforms
methods that do not account for the RX characteristics. Since the proposed
detector and alphabet design algorithm are also applicable to other chemical
sensors, they pave the way for reliable air-based MC.

### 2. [Simulating Mass-Dependent Decoherence in Quantum Computers: Baseline Signatures for Testing Gravity-Induced Collapse](http://arxiv.org/pdf/2508.10590v1)

Authors: Viswak R Balaji, Samuel Punch

We present a quantum computing simulation study of mass-dependent decoherence
models inspired by Penrose's gravity-induced collapse hypothesis. According to
objective reduction (OR) theory, quantum superpositions become unstable when
the gravitational self-energy difference between branches exceeds a certain
threshold, leading to a collapse time $\tau \approx \hbar / E_G$. In this work,
we implement a mass-dependent dephasing noise channel, $p(m) = 1 - e^{-k
m^{\alpha}}$, within the Qiskit AerSimulator, where $m$ is a proxy for the
effective mass of a superposition, mapped to circuit parameters such as the
number of entangled qubits or branch size. We apply this model to three
canonical quantum computing experiments: GHZ state parity measurements,
branch-mass entanglement tests, and Grover's search to generate distinctive
collapse signatures that differ qualitatively from constant-rate dephasing. The
resulting patterns serve as a baseline reference: if future hardware
experiments exhibit the same scaling trends under ideal isolation, this could
indicate a contribution from mass-dependent collapse processes. Conversely,
deviation toward constant-noise behaviour would suggest the absence of such
gravitationally induced effects. Our results provide a reproducible protocol
and reference for using quantum computers as potential testbeds for probing
fundamental questions in quantum mechanics.

### Formal Languages and Automata Theory

### 1. [Active Automata Learning with Advice](http://arxiv.org/pdf/2508.10535v1)

Authors: Michał Fica, Jan Otop

We present an extended automata learning framework that combines active
automata learning with deductive inference. The learning algorithm asks
membership and equivalence queries as in the original framework, but it is also
given advice, which is used to infer answers to queries when possible and
reduce the burden on the teacher. We consider advice given via string rewriting
systems, which specify equivalence of words w.r.t. the target languages. The
main motivation for the proposed framework is to reduce the number of queries.
We show how to adapt Angluin-style learning algorithms to this framework with
low overhead. Finally, we present empirical evaluation of our approach and
observe substantial improvement in query complexity.

### Graphics

### 1. [Puppeteer: Rig and Animate Your 3D Models](http://arxiv.org/pdf/2508.10898v1)

Authors: Chaoyue Song, Xiu Li, Fan Yang, Zhongcong Xu, Jiacheng Wei, Fayao Liu, Jiashi Feng, Guosheng Lin, Jianfeng Zhang

Modern interactive applications increasingly demand dynamic 3D content, yet
the transformation of static 3D models into animated assets constitutes a
significant bottleneck in content creation pipelines. While recent advances in
generative AI have revolutionized static 3D model creation, rigging and
animation continue to depend heavily on expert intervention. We present
Puppeteer, a comprehensive framework that addresses both automatic rigging and
animation for diverse 3D objects. Our system first predicts plausible skeletal
structures via an auto-regressive transformer that introduces a joint-based
tokenization strategy for compact representation and a hierarchical ordering
methodology with stochastic perturbation that enhances bidirectional learning
capabilities. It then infers skinning weights via an attention-based
architecture incorporating topology-aware joint attention that explicitly
encodes inter-joint relationships based on skeletal graph distances. Finally,
we complement these rigging advances with a differentiable optimization-based
animation pipeline that generates stable, high-fidelity animations while being
computationally more efficient than existing approaches. Extensive evaluations
across multiple benchmarks demonstrate that our method significantly
outperforms state-of-the-art techniques in both skeletal prediction accuracy
and skinning quality. The system robustly processes diverse 3D content, ranging
from professionally designed game assets to AI-generated shapes, producing
temporally coherent animations that eliminate the jittering issues common in
existing methods.

### Computer Science and Game Theory

### 1. [AlDBaran: Towards Blazingly Fast State Commitments for Blockchains](http://arxiv.org/pdf/2508.10493v1)

Authors: Bernhard Kauer, Aleksandr Petrosyan, Benjamin Livshits

The fundamental basis for maintaining integrity within contemporary
blockchain systems is provided by authenticated databases. Our analysis
indicates that a significant portion of the approaches applied in this domain
fail to sufficiently meet the stringent requirements of systems processing
transactions at rates of multi-million TPS. AlDBaran signifies a substantial
advancement in authenticated databases. By eliminating disk I/O operations from
the critical path, implementing prefetching strategies, and refining the update
mechanism of the Merkle tree, we have engineered an authenticated data
structure capable of handling state updates efficiently at a network throughput
of 50 Gbps. This throughput capacity significantly surpasses any empirically
documented blockchain throughput, guaranteeing the ability of even the most
high-throughput blockchains to generate state commitments effectively.
  AlDBaran provides support for historical state proofs, which facilitates a
wide array of novel applications. For instance, the deployment of AlDBaran
could enable blockchains that do not currently support state commitments to
offer functionalities for light clients and/or implement rollups.
  When benchmarked against alternative authenticated data structure projects,
AlDBaran exhibits superior performance and simplicity. In particular, AlDBaran
achieves speeds of approximately 48 million updates per second using an
identical machine configuration. This characteristic renders AlDBaran an
attractive solution for resource-limited environments, as its historical data
capabilities can be modularly isolated (and deactivated), which further
enhances performance. On consumer-level portable hardware, it achieves
approximately 8 million updates/s in an in-memory setting and 5 million
updates/s with snapshots at sub-second intervals, illustrating compelling and
cost-effective scalability.

### Human-Computer Interaction

### 1. [Artificial Emotion: A Survey of Theories and Debates on Realising Emotion in Artificial Intelligence](http://arxiv.org/pdf/2508.10286v1)

Authors: Yupei Li, Qiyang Sun, Michelle Schlicher, Yee Wen Lim, Björn W. Schuller

Affective Computing (AC) has enabled Artificial Intelligence (AI) systems to
recognise, interpret, and respond to human emotions - a capability also known
as Artificial Emotional Intelligence (AEI). It is increasingly seen as an
important component of Artificial General Intelligence (AGI). We discuss
whether in order to peruse this goal, AI benefits from moving beyond emotion
recognition and synthesis to develop internal emotion-like states, which we
term as Artificial Emotion (AE). This shift potentially allows AI to benefit
from the paradigm of `inner emotions' in ways we - as humans - do. Although
recent research shows early signs that AI systems may exhibit AE-like
behaviours, a clear framework for how emotions can be realised in AI remains
underexplored. In this paper, we discuss potential advantages of AE in AI,
review current manifestations of AE in machine learning systems, examine
emotion-modulated architectures, and summarise mechanisms for modelling and
integrating AE into future AI. We also explore the ethical implications and
safety risks associated with `emotional' AGI, while concluding with our opinion
on how AE could be beneficial in the future.

### 2. ["Here Comes the Makeup Tutorial You Asked For!": Exploring Communication Strategies and Viewer Engagement in Beauty Videos on Rednote](http://arxiv.org/pdf/2508.10364v1)

Authors: Xueer Lin, Chenyu Li, Yuhan Lyu, Zhicong Lu, Zhenhui Peng

More and more people, especially females, create and view beauty videos
covering topics like makeup tutorials and vlogs on social media platforms.
Understanding the communication strategies that creators use in these videos
and how they affect viewers' engagement can help spread beauty knowledge. By
coding 352 beauty videos in Rednote, this study presents a comprehensive
taxonomy of communication strategies used by the creators, such as using home
as the video background and displaying makeup effects when starting the
narrative at the beginning. We further label and computationally classify six
categories of comments that reveal viewers' engagement with beauty videos. The
regression analyses reveal the effects of beauty video communication strategies
on viewers' engagement; for example, calling viewers to take action at the end
tends to attract more comments that debate the product's efficacy. We discuss
insights into fostering the creation of beauty videos and the communication of
beauty knowledge.

### 3. [Stress Detection from Multimodal Wearable Sensor Data](http://arxiv.org/pdf/2508.10468v1)

Authors: Paul Schreiber, Beyza Cinar, Lennart Mackert, Maria Maleshkova

Human-Computer Interaction (HCI) is a multi-modal, interdisciplinary field
focused on designing, studying, and improving the interactions between people
and computer systems. This involves the design of systems that can recognize,
interpret, and respond to human emotions or stress. Developing systems to
monitor and react to stressful events can help prevent severe health
implications caused by long-term stress exposure. Currently, the publicly
available datasets and standardized protocols for data collection in this
domain are limited. Therefore, we introduce a multi-modal dataset intended for
wearable affective computing research, specifically the development of
automated stress recognition systems. We systematically review the publicly
available datasets recorded in controlled laboratory settings. Based on a
proposed framework for the standardization of stress experiments and data
collection, we collect physiological and motion signals from wearable devices
(e.g., electrodermal activity, photoplethysmography, three-axis accelerometer).
During the experimental protocol, we differentiate between the following four
affective/activity states: neutral, physical, cognitive stress, and
socio-evaluative stress. These different phases are meticulously labeled,
allowing for detailed analysis and reconstruction of each experiment. Meta-data
such as body positions, locations, and rest phases are included as further
annotations. In addition, we collect psychological self-assessments after each
stressor to evaluate subjects' affective states. The contributions of this
paper are twofold: 1) a novel multi-modal, publicly available dataset for
automated stress recognition, and 2) a benchmark for stress detection with 89\%
in a binary classification (baseline vs. stress) and 82\% in a multi-class
classification (baseline vs. stress vs. physical exercise).

### 4. [DEV: A Driver-Environment-Vehicle Closed-Loop Framework for Risk-Aware Adaptive Automation of Driving](http://arxiv.org/pdf/2508.10618v1)

Authors: Anaïs Halin, Christel Devue, Marc Van Droogenbroeck

The increasing integration of automation in vehicles aims to enhance both
safety and comfort, but it also introduces new risks, including driver
disengagement, reduced situation awareness, and mode confusion. In this work,
we propose the DEV framework, a closed-loop framework for risk-aware adaptive
driving automation that captures the dynamic interplay between the driver, the
environment, and the vehicle. The framework promotes to continuously adjusting
the operational level of automation based on a risk management strategy. The
real-time risk assessment supports smoother transitions and effective
cooperation between the driver and the automation system. Furthermore, we
introduce a nomenclature of indexes corresponding to each core component,
namely driver involvement, environment complexity, and vehicle engagement, and
discuss how their interaction influences driving risk. The DEV framework offers
a comprehensive perspective to align multidisciplinary research efforts and
guide the development of dynamic, risk-aware driving automation systems.

### 5. [Are Electrodermal Activity-Based Indicators of Driver Cognitive Distraction Robust to Varying Traffic Conditions and Adaptive Cruise Control Use?](http://arxiv.org/pdf/2508.10620v1)

Authors: Anaïs Halin, Marc Van Droogenbroeck, Christel Devue

In this simulator study, we investigate whether and how electrodermal
activity (EDA) reflects driver cognitive distraction under varying traffic
conditions and adaptive cruise control (ACC) use. Participants drove in six
scenarios, combining two levels of cognitive distraction (presence/absence of a
mental calculation task) and three levels of driving environment complexity
(different traffic conditions). Throughout the experiment, they were free to
activate or deactivate ACC (ACC use, two levels). We analyzed three EDA-based
indicators of cognitive distraction: SCL (mean skin conductance level), SCR
amplitude (mean amplitude of skin conductance responses), and SCR rate (rate of
skin conductance responses). Results indicate that all three indicators were
significantly influenced by cognitive distraction and ACC use, while
environment complexity influenced SCL and SCR amplitude, but not SCR rate.
These findings suggest that EDA-based indicators reflect variations in drivers'
mental workload due not only to cognitive distraction, but also to driving
environment and automation use.

### 6. [Gaze-Based Indicators of Driver Cognitive Distraction: Effects of Different Traffic Conditions and Adaptive Cruise Control Use](http://arxiv.org/pdf/2508.10624v1)

Authors: Anaïs Halin, Adrien Deliège, Christel Devue, Marc Van Droogenbroeck

In this simulator study, we investigate how gaze parameters reflect driver
cognitive distraction under varying traffic conditions and adaptive cruise
control (ACC) use. Participants completed six driving scenarios that combined
two levels of cognitive distraction (with/without mental calculations) and
three levels of driving environment complexity. Throughout the experiment,
participants were free to activate or deactivate an ACC. We analyzed two
gaze-based indicators of driver cognitive distraction: the percent road center,
and the gaze dispersions (horizontal and vertical). Our results show that
vertical gaze dispersion increases with traffic complexity, while ACC use leads
to gaze concentration toward the road center. Cognitive distraction reduces
road center gaze and increases vertical dispersion. Complementary analyses
revealed that these observations actually arise mainly between mental
calculations, while periods of mental calculations are characterized by a
temporary increase in gaze concentration.

### 7. [Visualization of Electronic Health Record Sequences at Scale](http://arxiv.org/pdf/2508.10700v1)

Authors: Ambre Assor, Mickael Sereno, Jean-Daniel Fekete

We present ParcoursVis, a Progressive Visual Analytics tool designed to
explore electronic health record sequences of patients at scale. Existing tools
process and aggregate the whole dataset upfront before showing the
visualization, taking a time proportional to the data size. Therefore, to
remain interactive, existing tools are limited to data sizes that can be
processed in under a few seconds to meet the latency constraints of human
attention. To overcome this limitation and scale to larger sizes, ParcoursVis
relies on a progressive algorithm that quickly shows an approximate initial
result of the aggregation, visualized as an Icicle tree, and improves it
iteratively, updating the visualization until the whole computation is done.
With its architecture, ParcoursVis remains interactive while visualizing the
sequences of tens of millions of patients, each described with thousands of
events; three to five orders of magnitude more than similar systems. Managing
large datasets allows for exploring rare medical conditions or unexpected
patient pathways, contributing to improving treatments. We describe the
algorithms we use and our evaluation concerning their scalability, convergence,
and stability. We also report on a set of guidelines to support visualization
designers in developing scalable progressive systems. ParcoursVis already
allows practitioners to perform analyses on two large real medical datasets.
Our prototype is open-source.

### 8. [Beyond Self-Regulated Learning Processes: Unveiling Hidden Tactics in Generative AI-Assisted Writing](http://arxiv.org/pdf/2508.10310v1)

Authors: Kaixun Yang, Yizhou Fan, Luzhen Tang, Mladen Raković, Xinyu Li, Dragan Gašević, Guanliang Chen

The integration of Generative AI (GenAI) into education is reshaping how
students learn, making self-regulated learning (SRL) - the ability to plan,
monitor, and adapt one's learning - more important than ever. To support
learners in these new contexts, it is essential to understand how SRL unfolds
during interaction with GenAI tools. Learning analytics offers powerful
techniques for analyzing digital trace data to infer SRL behaviors. However,
existing approaches often assume SRL processes are linear, segmented, and
non-overlapping-assumptions that overlook the dynamic, recursive, and
non-linear nature of real-world learning. We address this by conceptualizing
SRL as a layered system: observable learning patterns reflect hidden tactics
(short, purposeful action states), which combine into broader SRL strategies.
Using Hidden Markov Models (HMMs), we analyzed trace data from higher education
students engaged in GenAI-assisted academic writing. We identified three
distinct groups of learners, each characterized by different SRL strategies.
These groups showed significant differences in performance, indicating that
students' use of different SRL strategies in GenAI-assisted writing led to
varying task outcomes. Our findings advance the methodological toolkit for
modeling SRL and inform the design of adaptive learning technologies that more
effectively support learners in GenAI-enhanced educational environments.

### 9. [Mental Effort Estimation in Motion Exploration and Concept Generation Design Tasks using Inter-Band Relative Power Difference of EEG](http://arxiv.org/pdf/2508.10353v1)

Authors: G. Kalyan Ramana, Sumit Yempalle, Prasad S. Onkar

Conceptual design is a cognitively complex task, especially in the
engineering design of products having relative motion between components.
Designers prefer sketching as a medium for conceptual design and use gestures
and annotations to represent such relative motion. Literature suggests that
static representations of motion in sketches may not achieve the intended
functionality when realised, because it primarily depends on the designers'
mental capabilities for motion simulation. Thus, it is important to understand
the cognitive phenomena when designers are exploring concepts of articulated
products. The current work is an attempt to understand design neurocognition by
categorising the tasks and measuring the mental effort involved in these tasks
using EEG. The analysis is intended to validate design intervention tools to
support the conceptual design involving motion exploration. A novel EEG-based
metric, inter-Band Relative Power Difference (inter-BRPD), is introduced to
quantify mental effort. A design experiment is conducted with 32 participants,
where they have to perform one control task and 2 focus tasks corresponding to
the motion exploration task (MET) and the concept generation task (CGT),
respectively. EEG data is recorded during the 3 tasks, cleaned, processed and
analysed using the MNE library in Python. It is observed from the results that
inter-BRPD captures the essence of mental effort with half the number of
conventionally used parameters. The reliability and efficacy of the inter-BRPD
metric are also statistically validated against literature-based cognitive
metrics. With these new insights, the study opens up possibilities for creating
support for conceptual design and its evaluation.

### 10. [Why Report Failed Interactions With Robots?! Towards Vignette-based Interaction Quality](http://arxiv.org/pdf/2508.10603v1)

Authors: Agnes Axelsson, Merle Reimann, Ronald Cumbal, Hannah Pelikan, Divesh Lala

Although the quality of human-robot interactions has improved with the advent
of LLMs, there are still various factors that cause systems to be sub-optimal
when compared to human-human interactions. The nature and criticality of
failures are often dependent on the context of the interaction and so cannot be
generalized across the wide range of scenarios and experiments which have been
implemented in HRI research. In this work we propose the use of a technique
overlooked in the field of HRI, ethnographic vignettes, to clearly highlight
these failures, particularly those that are rarely documented. We describe the
methodology behind the process of writing vignettes and create our own based on
our personal experiences with failures in HRI systems. We emphasize the
strength of vignettes as the ability to communicate failures from a
multi-disciplinary perspective, promote transparency about the capabilities of
robots, and document unexpected behaviours which would otherwise be omitted
from research reports. We encourage the use of vignettes to augment existing
interaction evaluation methods.

### Information Retrieval

### 1. [Proxy Model-Guided Reinforcement Learning for Client Selection in Federated Recommendation](http://arxiv.org/pdf/2508.10401v1)

Authors: Liang Qu, Jianxin Li, Wei Yuan, Penghui Ruan, Yuhui Shi, Hongzhi Yin

Federated recommender systems have emerged as a promising privacy-preserving
paradigm, enabling personalized recommendation services without exposing users'
raw data. By keeping data local and relying on a central server to coordinate
training across distributed clients, FedRSs protect user privacy while
collaboratively learning global models. However, most existing FedRS frameworks
adopt fully random client selection strategy in each training round,
overlooking the statistical heterogeneity of user data arising from diverse
preferences and behavior patterns, thereby resulting in suboptimal model
performance. While some client selection strategies have been proposed in the
broader federated learning literature, these methods are typically designed for
generic tasks and fail to address the unique challenges of recommendation
scenarios, such as expensive contribution evaluation due to the large number of
clients, and sparse updates resulting from long-tail item distributions. To
bridge this gap, we propose ProxyRL-FRS, a proxy model-guided reinforcement
learning framework tailored for client selection in federated recommendation.
Specifically, we first introduce ProxyNCF, a dual-branch model deployed on each
client, which augments standard Neural Collaborative Filtering with an
additional proxy model branch that provides lightweight contribution
estimation, thus eliminating the need for expensive per-round local training
traditionally required to evaluate a client's contribution. Furthermore, we
design a staleness-aware SA reinforcement learning agent that selects clients
based on the proxy-estimated contribution, and is guided by a reward function
balancing recommendation accuracy and embedding staleness, thereby enriching
the update coverage of item embeddings. Experiments conducted on public
recommendation datasets demonstrate the effectiveness of ProxyRL-FRS.

### 2. [Semantic IDs for Joint Generative Search and Recommendation](http://arxiv.org/pdf/2508.10478v1)

Authors: Gustavo Penha, Edoardo D'Amico, Marco De Nadai, Enrico Palumbo, Alexandre Tamborrino, Ali Vardasbi, Max Lefarov, Shawn Lin, Timothy Heath, Francesco Fabbri, Hugues Bouchard

Generative models powered by Large Language Models (LLMs) are emerging as a
unified solution for powering both recommendation and search tasks. A key
design choice in these models is how to represent items, traditionally through
unique identifiers (IDs) and more recently with Semantic IDs composed of
discrete codes, obtained from embeddings. While task-specific embedding models
can improve performance for individual tasks, they may not generalize well in a
joint setting. In this paper, we explore how to construct Semantic IDs that
perform well both in search and recommendation when using a unified model. We
compare a range of strategies to construct Semantic IDs, looking into
task-specific and cross-tasks approaches, and also whether each task should
have its own semantic ID tokens in a joint search and recommendation generative
model. Our results show that using a bi-encoder model fine-tuned on both search
and recommendation tasks to obtain item embeddings, followed by the
construction of a unified Semantic ID space provides an effective trade-off,
enabling strong performance in both tasks. We hope these findings spark
follow-up work on generalisable, semantically grounded ID schemes and inform
the next wave of unified generative recommender architectures.

### 3. [Efficient Patent Searching Using Graph Transformers](http://arxiv.org/pdf/2508.10496v1)

Authors: Krzysztof Daniell, Igor Buzhinsky, Sebastian Björkqvist

Finding relevant prior art is crucial when deciding whether to file a new
patent application or invalidate an existing patent. However, searching for
prior art is challenging due to the large number of patent documents and the
need for nuanced comparisons to determine novelty. An accurate search engine is
therefore invaluable for speeding up the process. We present a Graph
Transformer-based dense retrieval method for patent searching where each
invention is represented by a graph describing its features and their
relationships. Our model processes these invention graphs and is trained using
prior art citations from patent office examiners as relevance signals. Using
graphs as input significantly improves the computational efficiency of
processing long documents, while leveraging examiner citations allows the model
to learn domain-specific similarities beyond simple text-based matching. The
result is a search engine that emulates how professional patent examiners
identify relevant documents. We compare our approach against publicly available
text embedding models and show substantial improvements in both prior art
retrieval quality and computational efficiency.

### 4. [DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System](http://arxiv.org/pdf/2508.10584v1)

Authors: Wencai Ye, Mingjie Sun, Shaoyun Shi, Peng Wang, Wenjin Wu, Peng Jiang

Semantic IDs are discrete identifiers generated by quantizing the Multi-modal
Large Language Models (MLLMs) embeddings, enabling efficient multi-modal
content integration in recommendation systems. However, their lack of
collaborative signals results in a misalignment with downstream discriminative
and generative recommendation objectives. Recent studies have introduced
various alignment mechanisms to address this problem, but their two-stage
framework design still leads to two main limitations: (1) inevitable
information loss during alignment, and (2) inflexibility in applying adaptive
alignment strategies, consequently constraining the mutual information
maximization during the alignment process. To address these limitations, we
propose a novel and flexible one-stage Dual-Aligned Semantic IDs (DAS) method
that simultaneously optimizes quantization and alignment, preserving semantic
integrity and alignment quality while avoiding the information loss typically
associated with two-stage methods. Meanwhile, DAS achieves more efficient
alignment between the semantic IDs and collaborative signals, with the
following two innovative and effective approaches: (1) Multi-view Constrative
Alignment: To maximize mutual information between semantic IDs and
collaborative signals, we first incorporate an ID-based CF debias module, and
then design three effective contrastive alignment methods: dual user-to-item
(u2i), dual item-to-item/user-to-user (i2i/u2u), and dual co-occurrence
item-to-item/user-to-user (i2i/u2u). (2) Dual Learning: By aligning the dual
quantizations of users and ads, the constructed semantic IDs for users and ads
achieve stronger alignment. Finally, we conduct extensive offline experiments
and online A/B tests to evaluate DAS's effectiveness, which is now successfully
deployed across various advertising scenarios at Kuaishou App, serving over 400
million users daily.

### 5. [FuXi-β: Towards a Lightweight and Fast Large-Scale Generative Recommendation Model](http://arxiv.org/pdf/2508.10615v1)

Authors: Yufei Ye, Wei Guo, Hao Wang, Hong Zhu, Yuyang Ye, Yong Liu, Huifeng Guo, Ruiming Tang, Defu Lian, Enhong Chen

Scaling laws for autoregressive generative recommenders reveal potential for
larger, more versatile systems but mean greater latency and training costs. To
accelerate training and inference, we investigated the recent generative
recommendation models HSTU and FuXi-$\alpha$, identifying two efficiency
bottlenecks: the indexing operations in relative temporal attention bias and
the computation of the query-key attention map. Additionally, we observed that
relative attention bias in self-attention mechanisms can also serve as
attention maps. Previous works like Synthesizer have shown that alternative
forms of attention maps can achieve similar performance, naturally raising the
question of whether some attention maps are redundant. Through empirical
experiments, we discovered that using the query-key attention map might degrade
the model's performance in recommendation tasks. To address these bottlenecks,
we propose a new framework applicable to Transformer-like recommendation
models. On one hand, we introduce Functional Relative Attention Bias, which
avoids the time-consuming operations of the original relative attention bias,
thereby accelerating the process. On the other hand, we remove the query-key
attention map from the original self-attention layer and design a new
Attention-Free Token Mixer module. Furthermore, by applying this framework to
FuXi-$\alpha$, we introduce a new model, FuXi-$\beta$. Experiments across
multiple datasets demonstrate that FuXi-$\beta$ outperforms previous
state-of-the-art models and achieves significant acceleration compared to
FuXi-$\alpha$, while also adhering to the scaling law. Notably, FuXi-$\beta$
shows an improvement of 27% to 47% in the NDCG@10 metric on large-scale
industrial datasets compared to FuXi-$\alpha$. Our code is available in a
public repository: https://github.com/USTC-StarTeam/FuXi-beta

### 6. [Hypercomplex Prompt-aware Multimodal Recommendation](http://arxiv.org/pdf/2508.10753v1)

Authors: Zheyu Chen, Jinfeng Xu, Hewei Wang, Shuo Yang, Zitong Wan, Haibo Hu

Modern recommender systems face critical challenges in handling information
overload while addressing the inherent limitations of multimodal representation
learning. Existing methods suffer from three fundamental limitations: (1)
restricted ability to represent rich multimodal features through a single
representation, (2) existing linear modality fusion strategies ignore the deep
nonlinear correlations between modalities, and (3) static optimization methods
failing to dynamically mitigate the over-smoothing problem in graph
convolutional network (GCN). To overcome these limitations, we propose HPMRec,
a novel Hypercomplex Prompt-aware Multimodal Recommendation framework, which
utilizes hypercomplex embeddings in the form of multi-components to enhance the
representation diversity of multimodal features. HPMRec adopts the hypercomplex
multiplication to naturally establish nonlinear cross-modality interactions to
bridge semantic gaps, which is beneficial to explore the cross-modality
features. HPMRec also introduces the prompt-aware compensation mechanism to aid
the misalignment between components and modality-specific features loss, and
this mechanism fundamentally alleviates the over-smoothing problem. It further
designs self-supervised learning tasks that enhance representation diversity
and align different modalities. Extensive experiments on four public datasets
show that HPMRec achieves state-of-the-art recommendation performance.

### 7. [Clicks Versus Conversion: Choosing a Recommender's Training Objective in E-Commerce](http://arxiv.org/pdf/2508.10377v1)

Authors: Michael Weiss, Robert Rosenbach, Christian Eggenberger

Ranking product recommendations to optimize for a high click-through rate
(CTR) or for high conversion, such as add-to-cart rate (ACR) and
Order-Submit-Rate (OSR, view-to-purchase conversion) are standard practices in
e-commerce. Optimizing for CTR appears like a straightforward choice: Training
data (i.e., click data) are simple to collect and often available in large
quantities. Additionally, CTR is used far beyond e-commerce, making it a
generalist, easily implemented option. ACR and OSR, on the other hand, are more
directly linked to a shop's business goals, such as the Gross Merchandise Value
(GMV). In this paper, we compare the effects of using either of these
objectives using an online A/B test. Among our key findings, we demonstrate
that in our shops, optimizing for OSR produces a GMV uplift more than five
times larger than when optimizing for CTR, without sacrificing new product
discovery. Our results also provide insights into the different feature
importances for each of the objectives.

### 8. [STEP: Stepwise Curriculum Learning for Context-Knowledge Fusion in Conversational Recommendation](http://arxiv.org/pdf/2508.10669v1)

Authors: Zhenye Yang, Jinpeng Chen, Huan Li, Xiongnan Jin, Xuanyang Li, Junwei Zhang, Hongbo Gao, Kaimin Wei, Senzhang Wang

Conversational recommender systems (CRSs) aim to proactively capture user
preferences through natural language dialogue and recommend high-quality items.
To achieve this, CRS gathers user preferences via a dialog module and builds
user profiles through a recommendation module to generate appropriate
recommendations. However, existing CRS faces challenges in capturing the deep
semantics of user preferences and dialogue context. In particular, the
efficient integration of external knowledge graph (KG) information into
dialogue generation and recommendation remains a pressing issue. Traditional
approaches typically combine KG information directly with dialogue content,
which often struggles with complex semantic relationships, resulting in
recommendations that may not align with user expectations.
  To address these challenges, we introduce STEP, a conversational recommender
centered on pre-trained language models that combines curriculum-guided
context-knowledge fusion with lightweight task-specific prompt tuning. At its
heart, an F-Former progressively aligns the dialogue context with
knowledge-graph entities through a three-stage curriculum, thus resolving
fine-grained semantic mismatches. The fused representation is then injected
into the frozen language model via two minimal yet adaptive prefix prompts: a
conversation prefix that steers response generation toward user intent and a
recommendation prefix that biases item ranking toward knowledge-consistent
candidates. This dual-prompt scheme allows the model to share cross-task
semantics while respecting the distinct objectives of dialogue and
recommendation. Experimental results show that STEP outperforms mainstream
methods in the precision of recommendation and dialogue quality in two public
datasets.

### 9. [CrossDenoise: Denoising Implicit Feedback via a Lightweight Entity-Aware Synergistic Framework](http://arxiv.org/pdf/2508.10851v1)

Authors: Ze Liu, Xianquan Wang, Shuochen Liu, Jie Ma, Huibo Xu, Yupeng Han, Zhe Yang, Kai Zhang, Longfei Li, Jun Zhou

Recommender systems heavily rely on implicit feedback, which is inherently
noisy due to false positives and negatives, severely degrading recommendation
accuracy. Existing denoising strategies often overlook entity-aware modeling,
suffer from high computational overhead, or demand excessive hyperparameter
tuning, limiting their real-world applicability. We propose CrossDenoise, a
novel and lightweight framework that addresses these challenges by
disentangling noise estimation into user-, item-, and interaction-specific
factors. Leveraging empirical observations that show significant heterogeneity
in user and item noise propensities, CrossDenoise computes entity reputation
factors (user/item reliability) via a rank-based linear mapping of average
training losses. These are fused with interaction-level weights derived from an
empirical cumulative distribution function (ECDF) of individual losses. This
design is model-agnostic, computationally efficient, and requires only two
intuitive hyperparameters. Extensive experiments on ML-1M, Yelp, and
Amazon-book datasets, across GMF, NeuMF, and CDAE backbones, demonstrate that
CrossDenoise consistently and significantly outperforms state-of-the-art
baselines. For instance, it achieves up to 27.01% NDCG@50 gain on Yelp with
NeuMF, while incurring negligible computational and memory overhead. Our
analysis confirms that CrossDenoise effectively separates clean from noisy
samples and remains robust under varied hyperparameter settings. It offers a
practical and scalable solution for denoising implicit feedback.

### 10. [Multi-Label Plant Species Prediction with Metadata-Enhanced Multi-Head Vision Transformers](http://arxiv.org/pdf/2508.10457v1)

Authors: Hanna Herasimchyk, Robin Labryga, Tomislav Prusina

We present a multi-head vision transformer approach for multi-label plant
species prediction in vegetation plot images, addressing the PlantCLEF 2025
challenge. The task involves training models on single-species plant images
while testing on multi-species quadrat images, creating a drastic domain shift.
Our methodology leverages a pre-trained DINOv2 Vision Transformer Base
(ViT-B/14) backbone with multiple classification heads for species, genus, and
family prediction, utilizing taxonomic hierarchies. Key contributions include
multi-scale tiling to capture plants at different scales, dynamic threshold
optimization based on mean prediction length, and ensemble strategies through
bagging and Hydra model architectures. The approach incorporates various
inference techniques including image cropping to remove non-plant artifacts,
top-n filtering for prediction constraints, and logit thresholding strategies.
Experiments were conducted on approximately 1.4 million training images
covering 7,806 plant species. Results demonstrate strong performance, making
our submission 3rd best on the private leaderboard. Our code is available at
https://github.com/geranium12/plant-clef-2025/tree/v1.0.0.

### Machine Learning

### 1. [Pruning and Malicious Injection: A Retraining-Free Backdoor Attack on Transformer Models](http://arxiv.org/pdf/2508.10243v1)

Authors: Taibiao Zhao, Mingxuan Sun, Hao Wang, Xiaobing Chen, Xiangwei Zhou

Transformer models have demonstrated exceptional performance and have become
indispensable in computer vision (CV) and natural language processing (NLP)
tasks. However, recent studies reveal that transformers are susceptible to
backdoor attacks. Prior backdoor attack methods typically rely on retraining
with clean data or altering the model architecture, both of which can be
resource-intensive and intrusive. In this paper, we propose Head-wise Pruning
and Malicious Injection (HPMI), a novel retraining-free backdoor attack on
transformers that does not alter the model's architecture. Our approach
requires only a small subset of the original data and basic knowledge of the
model architecture, eliminating the need for retraining the target transformer.
Technically, HPMI works by pruning the least important head and injecting a
pre-trained malicious head to establish the backdoor. We provide a rigorous
theoretical justification demonstrating that the implanted backdoor resists
detection and removal by state-of-the-art defense techniques, under reasonable
assumptions. Experimental evaluations across multiple datasets further validate
the effectiveness of HPMI, showing that it 1) incurs negligible clean accuracy
loss, 2) achieves at least 99.55% attack success rate, and 3) bypasses four
advanced defense mechanisms. Additionally, relative to state-of-the-art
retraining-dependent attacks, HPMI achieves greater concealment and robustness
against diverse defense strategies, while maintaining minimal impact on clean
accuracy.

### 2. [Multi-Agent Reinforcement Learning for Adaptive Resource Orchestration in Cloud-Native Clusters](http://arxiv.org/pdf/2508.10253v1)

Authors: Guanzi Yao, Heyao Liu, Linyan Dai

This paper addresses the challenges of high resource dynamism and scheduling
complexity in cloud-native database systems. It proposes an adaptive resource
orchestration method based on multi-agent reinforcement learning. The method
introduces a heterogeneous role-based agent modeling mechanism. This allows
different resource entities, such as compute nodes, storage nodes, and
schedulers, to adopt distinct policy representations. These agents are better
able to reflect diverse functional responsibilities and local environmental
characteristics within the system. A reward-shaping mechanism is designed to
integrate local observations with global feedback. This helps mitigate policy
learning bias caused by incomplete state observations. By combining real-time
local performance signals with global system value estimation, the mechanism
improves coordination among agents and enhances policy convergence stability. A
unified multi-agent training framework is developed and evaluated on a
representative production scheduling dataset. Experimental results show that
the proposed method outperforms traditional approaches across multiple key
metrics. These include resource utilization, scheduling latency, policy
convergence speed, system stability, and fairness. The results demonstrate
strong generalization and practical utility. Across various experimental
scenarios, the method proves effective in handling orchestration tasks with
high concurrency, high-dimensional state spaces, and complex dependency
relationships. This confirms its advantages in real-world, large-scale
scheduling environments.

### 3. [Federated Anomaly Detection for Multi-Tenant Cloud Platforms with Personalized Modeling](http://arxiv.org/pdf/2508.10255v1)

Authors: Yuxi Wang, Heyao Liu, Nyutian Long, Guanzi Yao

This paper proposes an anomaly detection method based on federated learning
to address key challenges in multi-tenant cloud environments, including data
privacy leakage, heterogeneous resource behavior, and the limitations of
centralized modeling. The method establishes a federated training framework
involving multiple tenants. Each tenant trains the model locally using private
resource usage data. Through parameter aggregation, a global model is
optimized, enabling cross-tenant collaborative anomaly detection while
preserving data privacy. To improve adaptability to diverse resource usage
patterns, a personalized parameter adjustment mechanism is introduced. This
allows the model to retain tenant-specific feature representations while
sharing global knowledge. In the model output stage, the Mahalanobis distance
is used to compute anomaly scores. This enhances both the accuracy and
stability of anomaly detection. The experiments use real telemetry data from a
cloud platform to construct a simulated multi-tenant environment. The study
evaluates the model's performance under varying participation rates and noise
injection levels. These comparisons demonstrate the proposed method's
robustness and detection accuracy. Experimental results show that the proposed
method outperforms existing mainstream models across key metrics such as
Precision, Recall, and F1-Score. It also maintains stable performance in
various complex scenarios. These findings highlight the method's practical
potential for intelligent resource monitoring and anomaly diagnosis in cloud
computing environments.

### 4. [Source Component Shift Adaptation via Offline Decomposition and Online Mixing Approach](http://arxiv.org/pdf/2508.10257v1)

Authors: Ryuta Matsuno

This paper addresses source component shift adaptation, aiming to update
predictions adapting to source component shifts for incoming data streams based
on past training data. Existing online learning methods often fail to utilize
recurring shifts effectively, while model-pool-based methods struggle to
capture individual source components, leading to poor adaptation. In this
paper, we propose a source component shift adaptation method via an offline
decomposition and online mixing approach. We theoretically identify that the
problem can be divided into two subproblems: offline source component
decomposition and online mixing weight adaptation. Based on this, our method
first determines prediction models, each of which learns a source component
solely based on past training data offline through the EM algorithm. Then, it
updates the mixing weight of the prediction models for precise prediction
through online convex optimization. Thanks to our theoretical derivation, our
method fully leverages the characteristics of the shifts, achieving superior
adaptation performance over existing methods. Experiments conducted on various
real-world regression datasets demonstrate that our method outperforms
baselines, reducing the cumulative test loss by up to 67.4%.

### 5. [XQuant: Breaking the Memory Wall for LLM Inference with KV Cache Rematerialization](http://arxiv.org/pdf/2508.10395v1)

Authors: Aditya Tomar, Coleman Hooper, Minjae Lee, Haocheng Xi, Rishabh Tiwari, Wonjun Kang, Luca Manolache, Michael W. Mahoney, Kurt Keutzer, Amir Gholami

Although LLM inference has emerged as a critical workload for many downstream
applications, efficiently inferring LLMs is challenging due to the substantial
memory footprint and bandwidth requirements. In parallel, compute capabilities
have steadily outpaced both memory capacity and bandwidth over the last few
decades, a trend that remains evident in modern GPU hardware and exacerbates
the challenge of LLM inference. As such, new algorithms are emerging that trade
increased computation for reduced memory operations. To that end, we present
XQuant, which takes advantage of this trend, enabling an order-of-magnitude
reduction in memory consumption through low-bit quantization with substantial
accuracy benefits relative to state-of-the-art KV cache quantization methods.
We accomplish this by quantizing and caching the layer input activations X,
instead of using standard KV caching, and then rematerializing the Keys and
Values on-the-fly during inference. This results in an immediate 2$\times$
memory savings compared to KV caching. By applying XQuant, we achieve up to
$\sim 7.7\times$ memory savings with $<0.1$ perplexity degradation compared to
the FP16 baseline. Furthermore, our approach leverages the fact that X values
are similar across layers. Building on this observation, we introduce
XQuant-CL, which exploits the cross-layer similarity in the X embeddings for
extreme compression. Across different models, XQuant-CL attains up to
10$\times$ memory savings relative to the FP16 baseline with only 0.01
perplexity degradation, and 12.5$\times$ memory savings with only $0.1$
perplexity degradation. XQuant exploits the rapidly increasing compute
capabilities of hardware platforms to eliminate the memory bottleneck, while
surpassing state-of-the-art KV cache quantization methods and achieving
near-FP16 accuracy across a wide range of models.

### 6. [SC2Arena and StarEvolve: Benchmark and Self-Improvement Framework for LLMs in Complex Decision-Making Tasks](http://arxiv.org/pdf/2508.10428v1)

Authors: Pengbo Shen, Yaqing Wang, Ni Mu, Yao Luan, Runpeng Xie, Senhao Yang, Lexiang Wang, Hao Hu, Shuang Xu, Yiqin Yang, Bo Xu

Evaluating large language models (LLMs) in complex decision-making is
essential for advancing AI's ability for strategic planning and real-time
adaptation. However, existing benchmarks for tasks like StarCraft II fail to
capture the game's full complexity, such as its complete game context, diverse
action spaces, and all playable races. To address this gap, we present
SC2Arena, a benchmark that fully supports all playable races, low-level action
spaces, and optimizes text-based observations to tackle spatial reasoning
challenges. Complementing this, we introduce StarEvolve, a hierarchical
framework that integrates strategic planning with tactical execution, featuring
iterative self-correction and continuous improvement via fine-tuning on
high-quality gameplay data. Its key components include a
Planner-Executor-Verifier structure to break down gameplay, and a scoring
system for selecting high-quality training samples. Comprehensive analysis
using SC2Arena provides valuable insights into developing generalist agents
that were not possible with previous benchmarks. Experimental results also
demonstrate that our proposed StarEvolve achieves superior performance in
strategic planning. Our code, environment, and algorithms are publicly
available.

### 7. [GraphFedMIG: Tackling Class Imbalance in Federated Graph Learning via Mutual Information-Guided Generation](http://arxiv.org/pdf/2508.10471v1)

Authors: Xinrui Li, Qilin Fan, Tianfu Wang, Kaiwen Wei, Ke Yu, Xu Zhang

Federated graph learning (FGL) enables multiple clients to collaboratively
train powerful graph neural networks without sharing their private,
decentralized graph data. Inherited from generic federated learning, FGL is
critically challenged by statistical heterogeneity, where non-IID data
distributions across clients can severely impair model performance. A
particularly destructive form of this is class imbalance, which causes the
global model to become biased towards majority classes and fail at identifying
rare but critical events. This issue is exacerbated in FGL, as nodes from a
minority class are often surrounded by biased neighborhood information,
hindering the learning of expressive embeddings. To grapple with this
challenge, we propose GraphFedMIG, a novel FGL framework that reframes the
problem as a federated generative data augmentation task. GraphFedMIG employs a
hierarchical generative adversarial network where each client trains a local
generator to synthesize high-fidelity feature representations. To provide
tailored supervision, clients are grouped into clusters, each sharing a
dedicated discriminator. Crucially, the framework designs a mutual
information-guided mechanism to steer the evolution of these client generators.
By calculating each client's unique informational value, this mechanism
corrects the local generator parameters, ensuring that subsequent rounds of
mutual information-guided generation are focused on producing high-value,
minority-class features. We conduct extensive experiments on four real-world
datasets, and the results demonstrate the superiority of the proposed
GraphFedMIG compared with other baselines.

### 8. [Learning State-Space Models of Dynamic Systems from Arbitrary Data using Joint Embedding Predictive Architectures](http://arxiv.org/pdf/2508.10489v1)

Authors: Jonas Ulmen, Ganesh Sundaram, Daniel Görges

With the advent of Joint Embedding Predictive Architectures (JEPAs), which
appear to be more capable than reconstruction-based methods, this paper
introduces a novel technique for creating world models using continuous-time
dynamic systems from arbitrary observation data. The proposed method integrates
sequence embeddings with neural ordinary differential equations (neural ODEs).
It employs loss functions that enforce contractive embeddings and Lipschitz
constants in state transitions to construct a well-organized latent state
space. The approach's effectiveness is demonstrated through the generation of
structured latent state-space models for a simple pendulum system using only
image data. This opens up a new technique for developing more general control
algorithms and estimation techniques with broad applications in robotics.

### 9. [Projected Coupled Diffusion for Test-Time Constrained Joint Generation](http://arxiv.org/pdf/2508.10531v1)

Authors: Hao Luan, Yi Xian Goh, See-Kiong Ng, Chun Kai Ling

Modifications to test-time sampling have emerged as an important extension to
diffusion algorithms, with the goal of biasing the generative process to
achieve a given objective without having to retrain the entire diffusion model.
However, generating jointly correlated samples from multiple pre-trained
diffusion models while simultaneously enforcing task-specific constraints
without costly retraining has remained challenging. To this end, we propose
Projected Coupled Diffusion (PCD), a novel test-time framework for constrained
joint generation. PCD introduces a coupled guidance term into the generative
dynamics to encourage coordination between diffusion models and incorporates a
projection step at each diffusion step to enforce hard constraints.
Empirically, we demonstrate the effectiveness of PCD in application scenarios
of image-pair generation, object manipulation, and multi-robot motion planning.
Our results show improved coupling effects and guaranteed constraint
satisfaction without incurring excessive computational costs.

### 10. [Driving Accurate Allergen Prediction with Protein Language Models and Generalization-Focused Evaluation](http://arxiv.org/pdf/2508.10541v1)

Authors: Brian Shing-Hei Wong, Joshua Mincheol Kim, Sin-Hang Fung, Qing Xiong, Kelvin Fu-Kiu Ao, Junkang Wei, Ran Wang, Dan Michelle Wang, Jingying Zhou, Bo Feng, Alfred Sze-Lok Cheng, Kevin Y. Yip, Stephen Kwok-Wing Tsui, Qin Cao

Allergens, typically proteins capable of triggering adverse immune responses,
represent a significant public health challenge. To accurately identify
allergen proteins, we introduce Applm (Allergen Prediction with Protein
Language Models), a computational framework that leverages the 100-billion
parameter xTrimoPGLM protein language model. We show that Applm consistently
outperforms seven state-of-the-art methods in a diverse set of tasks that
closely resemble difficult real-world scenarios. These include identifying
novel allergens that lack similar examples in the training set, differentiating
between allergens and non-allergens among homologs with high sequence
similarity, and assessing functional consequences of mutations that create few
changes to the protein sequences. Our analysis confirms that xTrimoPGLM,
originally trained on one trillion tokens to capture general protein sequence
characteristics, is crucial for Applm's performance by detecting important
differences among protein sequences. In addition to providing Applm as
open-source software, we also provide our carefully curated benchmark datasets
to facilitate future research.

### Neural and Evolutionary Computing

### 1. [Deep Learning in Classical and Quantum Physics](http://arxiv.org/pdf/2508.10666v1)

Authors: Timothy Heightman, Marcin Płodzień

Scientific progress is tightly coupled to the emergence of new research
tools. Today, machine learning (ML)-especially deep learning (DL)-has become a
transformative instrument for quantum science and technology. Owing to the
intrinsic complexity of quantum systems, DL enables efficient exploration of
large parameter spaces, extraction of patterns from experimental data, and
data-driven guidance for research directions. These capabilities already
support tasks such as refining quantum control protocols and accelerating the
discovery of materials with targeted quantum properties, making ML/DL literacy
an essential skill for the next generation of quantum scientists. At the same
time, DL's power brings risks: models can overfit noisy data, obscure causal
structure, and yield results with limited physical interpretability.
Recognizing these limitations and deploying mitigation strategies is crucial
for scientific rigor. These lecture notes provide a comprehensive,
graduate-level introduction to DL for quantum applications, combining
conceptual exposition with hands-on examples. Organized as a progressive
sequence, they aim to equip readers to decide when and how to apply DL
effectively, to understand its practical constraints, and to adapt AI methods
responsibly to problems across quantum physics, chemistry, and engineering.

### 2. [Empirical Investigation into Configuring Echo State Networks for Representative Benchmark Problem Domains](http://arxiv.org/pdf/2508.10887v1)

Authors: Brooke R. Weborg, Gursel Serpen

This paper examines Echo State Network, a reservoir computer, performance
using four different benchmark problems, then proposes heuristics or rules of
thumb for configuring the architecture, as well as the selection of parameters
and their values, which are applicable to problems within the same domain, to
help serve to fill the experience gap needed by those entering this field of
study. The influence of various parameter selections and their value
adjustments, as well as architectural changes made to an Echo State Network, a
powerful recurrent neural network configured as a reservoir computer, can be
challenging to fully comprehend without experience in the field, and even some
hyperparameter optimization algorithms may have difficulty adjusting parameter
values without proper manual selections made first. Therefore, it is imperative
to understand the effects of parameters and their value selection on Echo State
Network architecture performance for a successful build. Thus, to address the
requirement for an extensive background in Echo State Network architecture, as
well as examine how Echo State Network performance is affected with respect to
variations in architecture, design, and parameter selection and values, a
series of benchmark tasks representing different problem domains, including
time series prediction, pattern generation, chaotic system prediction, and time
series classification, were modeled and experimented on to show the impact on
the performance of Echo State Network.

### Networking and Internet Architecture

### 1. [Rethinking Reliability Using Network Coding: a Practical 5G Evaluation](http://arxiv.org/pdf/2508.10247v1)

Authors: Laura Landon, Vipindev Adat Vasudevan, Junmo Sung, Muriel Médard

This work presents the design and implementation of a real-time network
coding system integrated into the IP layer of a 5G testbed, offering an
alternative to conventional retransmission-based reliability mechanisms such as
ARQ and HARQ. Using a netfilter-based packet interception framework, we inject
forward erasure correction using Random Linear Network Coding (RLNC) into live
traffic between a gNB and UE over a 3GPP RF link. We evaluate a block coding
scheme, analyzing its impact on throughput, jitter, and resource usage. Results
show that with appropriate code rate selection, RLNC can fully recover from
packet losses using fewer transmissions than ARQ/HARQ and maintain a high
throughput, particularly under moderate-to-high packet loss rates. These
findings demonstrate that network coding can effectively replace
retransmission-based reliability in future wireless systems, with the potential
for more efficient resource utilization.

### 2. [Design of a Timer Queue Supporting Dynamic Update Operations](http://arxiv.org/pdf/2508.10283v1)

Authors: Zekun Wang, Binghao Yue, Weitao Pan, Jiangyi Shi, Yue Hao

Large-scale timers are ubiquitous in network processing, including flow table
entry expiration control in software defined network (SDN) switches, MAC
address aging in Ethernet bridges, and retransmission timeout management in
TCP/IP protocols. Conventional implementations suffer from critical
limitations: low timing accuracy due to large-scale timer traversal and high
computational overhead for new timer insertion. This paper presents a
hybrid-architecture hardware priority queue based on systolic arrays and shift
registers for efficient timer queue management. The design uniquely supports
five operations: enqueue, dequeue, delete, update, and peek.To the best of our
knowledge, it is the first hardware priority queue enabling in-queue priority
updates. By leveraging centralized Boolean logic encoding within systolic
blocks, the design efficiently generates set/shift control signals while the
novel push-first operation ensures FIFO ordering for same-priority timers
without additional metadata. Experimental results demonstrate that the design
operates at over 400 MHz on FPGAs, achieving a 2.2-2.8x reduction in resource
consumption compared to state-of-the-art implementations.

### 3. [Near-realtime Earth Observation Via Starlink LEO Satellite Constellation](http://arxiv.org/pdf/2508.10338v1)

Authors: Bo Wu, Pengfei Zhou

Earth observation (EO) satellites in Low Earth Orbit (LEO) are collecting
vast amounts of data, which are invaluable for applications such as monitoring
forest fires. However, data downloading from EO satellites faces significant
challenges due to the limited number of ground stations and the brief
communication windows with them. Conversely, emerging LEO constellations like
Starlink have enabled continuous connectivity and revolutionized access for
ordinary users globally, who can connect via a simple satellite dish. In this
paper, we study the feasibility of supporting EO satellites with Starlink
satellite infrastructure and introduce a novel data delivery system, designated
as "Starlink Space User" (SSU), for relaying data from observation satellites.
SSU treats EO satellites as space users of Starlink, facilitating efficient
data transfer to Earth. At the core of SSU is a novel class of algorithms
designed for link and PoP selection, as well as system scheduling optimization,
that operate effectively atop Starlink's proprietary infrastructure. We assess
the performance of SSU using trace-driven simulations alongside real-world
Starlink performance measurements. Our results demonstrate that the proposed
Starlink-aided design can significantly reduce the median backlog (data not
delivered) per satellite.

### 4. [Federated Learning Over LoRa Networks: Simulator Design and Performance Evaluation](http://arxiv.org/pdf/2508.10574v1)

Authors: Anshika Singh, Siddhartha S. Borkotoky

Federated learning (FL) over long-range (LoRa) low-power wide area networks
faces unique challenges due to limited bandwidth, interference, and strict
duty-cycle constraints. We develop a Python-based simulator that integrates and
extends the Flower and LoRaSim frameworks to evaluate centralized FL over LoRa
networks. The simulator employs a detailed link-level model for FL update
transfer over LoRa channels, capturing LoRa's receiver sensitivity,
interference characteristics, block-fading effects, and constraints on the
maximum transmission unit. It supports update sparsification, quantization,
compression, forward frame-erasure correction (FEC), and duty cycling.
Numerical results illustrate the impact of transmission parameters (spreading
factor, FEC rate) and interference on FL performance. Demonstrating the
critical role of FEC in enabling FL over LoRa networks, we perform an in-depth
evaluation of the impact of FEC on FL convergence and device airtime, providing
insights for communication protocol design for FL over LoRa networks.

### 5. [Balancing the Energy Consumption and Latency of Over-the-Air Firmware Updates in LoRaWAN](http://arxiv.org/pdf/2508.10588v1)

Authors: Siddhartha S. Borkotoky

Over-the-air firmware updates are crucial for mitigating security threats and
maintaining up-to-date device functionality in Long Range Wide Area Networks
(LoRaWANs). LoRaWAN end devices are usually energy-constrained, and LoRaWAN
transmissions are subject to duty-cycle restrictions. Consequently, controlling
the energy expenditure and update-delivery latency of FUOTA are key challenges.
We propose a flexible scheme that achieves a tunable trade-off between the
energy consumption and delivery delay. The scheme employs the LoRa spreading
factors sequentially to transmit update-carrying frames, sending a fixed number
of frames with a given spreading factor before moving to the next. By adjusting
the smallest spreading factor to be used and the number of transmissions per
spreading factor, a suitable energy-delay trade-off can be achieved. Thus,
time-sensitive updates, such as security patches, may be sent with a
low-delay-high-energy setting, whereas a more energy-efficient but higher-delay
setting may be used for non-critical updates.

### 6. [A Hierarchical IDS for Zero-Day Attack Detection in Internet of Medical Things Networks](http://arxiv.org/pdf/2508.10346v1)

Authors: Md Ashraf Uddin, Nam H. Chu, Reza Rafeh

The Internet of Medical Things (IoMT) is driving a healthcare revolution but
remains vulnerable to cyberattacks such as denial of service, ransomware, data
hijacking, and spoofing. These networks comprise resource constrained,
heterogeneous devices (e.g., wearable sensors, smart pills, implantables),
making traditional centralized Intrusion Detection Systems (IDSs) unsuitable
due to response delays, privacy risks, and added vulnerabilities. Centralized
IDSs require all sensors to transmit data to a central server, causing delays
or network disruptions in dense environments. Running IDSs locally on IoMT
devices is often infeasible due to limited computation, and even lightweight
IDS components remain at risk if updated models are delayed leaving them
exposed to zero-day attacks that threaten patient health and data security. We
propose a multi level IoMT IDS framework capable of detecting zero day attacks
and distinguishing between known and unknown threats. The first layer (near
Edge) filters traffic at a coarse level (attack or not) using meta-learning or
One Class Classification (OCC) with the usfAD algorithm. Subsequent layers (far
Edge, Cloud) identify attack type and novelty. Experiments on the CICIoMT2024
dataset show 99.77 percentage accuracy and 97.8 percentage F1-score. The first
layer detects zero-day attacks with high accuracy without needing new datasets,
ensuring strong applicability in IoMT environments. Additionally, the
meta-learning approach achieves high.

### 7. [Semantic Communication with Distribution Learning through Sequential Observations](http://arxiv.org/pdf/2508.10350v1)

Authors: Samer Lahoud, Kinda Khawam

Semantic communication aims to convey meaning rather than bit-perfect
reproduction, representing a paradigm shift from traditional communication.
This paper investigates distribution learning in semantic communication where
receivers must infer the underlying meaning distribution through sequential
observations. While semantic communication traditionally optimizes individual
meaning transmission, we establish fundamental conditions for learning source
statistics when priors are unknown. We prove that learnability requires full
rank of the effective transmission matrix, characterize the convergence rate of
distribution estimation, and quantify how estimation errors translate to
semantic distortion. Our analysis reveals a fundamental trade-off: encoding
schemes optimized for immediate semantic performance often sacrifice long-term
learnability. Experiments on CIFAR-10 validate our theoretical framework,
demonstrating that system conditioning critically impacts both learning rate
and achievable performance. These results provide the first rigorous
characterization of statistical learning in semantic communication and offer
design principles for systems that balance immediate performance with
adaptation capability.

### 8. [Probabilistic Latency Analysis of the Data Distribution Service in ROS 2](http://arxiv.org/pdf/2508.10413v1)

Authors: Sanghoon Lee, Hyung-Seok Park, Jiyeong Chae, Kyung-Joon Park

Robot Operating System 2 (ROS 2) is now the de facto standard for robotic
communication, pairing UDP transport with the Data Distribution Service (DDS)
publish-subscribe middleware. DDS achieves reliability through periodic
heartbeats that solicit acknowledgments for missing samples and trigger
selective retransmissions. In lossy wireless networks, the tight coupling among
heartbeat period, IP fragmentation, and retransmission interval obscures end to
end latency behavior and leaves practitioners with little guidance on how to
tune these parameters. To address these challenges, we propose a probabilistic
latency analysis (PLA) that analytically models the reliable transmission
process of ROS 2 DDS communication using a discrete state approach. By
systematically analyzing both middleware level and transport level events, PLA
computes the steady state probability distribution of unacknowledged messages
and the retransmission latency. We validate our PLA across 270 scenarios,
exploring variations in packet delivery ratios, message sizes, and both
publishing and retransmission intervals, demonstrating a close alignment
between analytical predictions and experimental results. Our findings establish
a theoretical basis to systematically optimize reliability, latency, and
performance in wireless industrial robotics.

### 9. [Routing and Wavelength Assignment with Minimal Attack Radius for QKD Networks](http://arxiv.org/pdf/2508.10613v1)

Authors: Mengyao Li, Qiaolun Zhang, Zongshuai Yang, Stefano Bregni, Alberto Gatto, Raouf Boutaba, Massimo Tornatore

Quantum Key Distribution (QKD) can distribute keys with guaranteed security
but remains susceptible to key exchange interruption due to physical-layer
threats, such as high-power jamming attacks. To address this challenge, we
first introduce a novel metric, namely Maximum Number of Affected Requests
(maxNAR), to quantify the worst-case impact of a single physical-layer attack,
and then we investigate a new problem of Routing and Wavelength Assignment with
Minimal Attack Radius (RWA-MAR). We formulate the problem using an Integer
Linear Programming (ILP) model and propose a scalable heuristic to efficiently
minimize maxNAR. Our approach incorporates key caching through Quantum Key
Pools (QKPs) to enhance resilience and optimize resource utilization. Moreover,
we model the impact of different QKD network architectures, employing Optical
Bypass (OB) for optical switching of quantum channels and Trusted Relay (TR)
for secure key forwarding. Moreover, a tunable parameter is designed in the
heuristic to guide the preference for OB or TR, offering enhanced adaptability
and dynamic control in diverse network scenarios. Simulation results confirm
that our method significantly outperforms the baseline in terms of security and
scalability.

### Robotics

### 1. [Hybrid Data-Driven Predictive Control for Robust and Reactive Exoskeleton Locomotion Synthesis](http://arxiv.org/pdf/2508.10269v1)

Authors: Kejun Li, Jeeseop Kim, Maxime Brunet, Marine Pétriaux, Yisong Yue, Aaron D. Ames

Robust bipedal locomotion in exoskeletons requires the ability to dynamically
react to changes in the environment in real time. This paper introduces the
hybrid data-driven predictive control (HDDPC) framework, an extension of the
data-enabled predictive control, that addresses these challenges by
simultaneously planning foot contact schedules and continuous domain
trajectories. The proposed framework utilizes a Hankel matrix-based
representation to model system dynamics, incorporating step-to-step (S2S)
transitions to enhance adaptability in dynamic environments. By integrating
contact scheduling with trajectory planning, the framework offers an efficient,
unified solution for locomotion motion synthesis that enables robust and
reactive walking through online replanning. We validate the approach on the
Atalante exoskeleton, demonstrating improved robustness and adaptability.

### 2. [BEASST: Behavioral Entropic Gradient based Adaptive Source Seeking for Mobile Robots](http://arxiv.org/pdf/2508.10363v1)

Authors: Donipolo Ghimire, Aamodh Suresh, Carlos Nieto-Granda, Solmaz S. Kia

This paper presents BEASST (Behavioral Entropic Gradient-based Adaptive
Source Seeking for Mobile Robots), a novel framework for robotic source seeking
in complex, unknown environments. Our approach enables mobile robots to
efficiently balance exploration and exploitation by modeling normalized signal
strength as a surrogate probability of source location. Building on Behavioral
Entropy(BE) with Prelec's probability weighting function, we define an
objective function that adapts robot behavior from risk-averse to risk-seeking
based on signal reliability and mission urgency. The framework provides
theoretical convergence guarantees under unimodal signal assumptions and
practical stability under bounded disturbances. Experimental validation across
DARPA SubT and multi-room scenarios demonstrates that BEASST consistently
outperforms state-of-the-art methods, achieving 15% reduction in path length
and 20% faster source localization through intelligent uncertainty-driven
navigation that dynamically transitions between aggressive pursuit and cautious
exploration.

### 3. [Few-shot Vision-based Human Activity Recognition with MLLM-based Visual Reinforcement Learning](http://arxiv.org/pdf/2508.10371v1)

Authors: Wenqi Zheng, Yutaka Arakawa

Reinforcement learning in large reasoning models enables learning from
feedback on their outputs, making it particularly valuable in scenarios where
fine-tuning data is limited. However, its application in multi-modal human
activity recognition (HAR) domains remains largely underexplored. Our work
extends reinforcement learning to the human activity recognition domain with
multimodal large language models. By incorporating visual reinforcement
learning in the training process, the model's generalization ability on
few-shot recognition can be greatly improved. Additionally, visual
reinforcement learning can enhance the model's reasoning ability and enable
explainable analysis in the inference stage. We name our few-shot human
activity recognition method with visual reinforcement learning FAVOR.
Specifically, our approach first utilizes a multimodal large language model
(MLLM) to generate multiple candidate responses for the human activity image,
each containing reasoning traces and final answers. These responses are then
evaluated using reward functions, and the MLLM model is subsequently optimized
using the Group Relative Policy Optimization (GRPO) algorithm. In this way, the
MLLM model can be adapted to human activity recognition with only a few
samples. Extensive experiments on four human activity recognition datasets and
five different settings demonstrate the superiority of the proposed method.

### 4. [A Semantic-Aware Framework for Safe and Intent-Integrative Assistance in Upper-Limb Exoskeletons](http://arxiv.org/pdf/2508.10378v1)

Authors: Yu Chen, Shu Miao, Chunyu Wu, Jingsong Mu, Bo OuYang, Xiang Li

Upper-limb exoskeletons are primarily designed to provide assistive support
by accurately interpreting and responding to human intentions. In home-care
scenarios, exoskeletons are expected to adapt their assistive configurations
based on the semantic information of the task, adjusting appropriately in
accordance with the nature of the object being manipulated. However, existing
solutions often lack the ability to understand task semantics or
collaboratively plan actions with the user, limiting their generalizability. To
address this challenge, this paper introduces a semantic-aware framework that
integrates large language models into the task planning framework, enabling the
delivery of safe and intent-integrative assistance. The proposed approach
begins with the exoskeleton operating in transparent mode to capture the
wearer's intent during object grasping. Once semantic information is extracted
from the task description, the system automatically configures appropriate
assistive parameters. In addition, a diffusion-based anomaly detector is used
to continuously monitor the state of human-robot interaction and trigger
real-time replanning in response to detected anomalies. During task execution,
online trajectory refinement and impedance control are used to ensure safety
and regulate human-robot interaction. Experimental results demonstrate that the
proposed method effectively aligns with the wearer's cognition, adapts to
semantically varying tasks, and responds reliably to anomalies.

### 5. [Super LiDAR Reflectance for Robotic Perception](http://arxiv.org/pdf/2508.10398v1)

Authors: Wei Gao, Jie Zhang, Mingle Zhao, Zhiyuan Zhang, Shu Kong, Maani Ghaffari, Dezhen Song, Cheng-Zhong Xu, Hui Kong

Conventionally, human intuition often defines vision as a modality of passive
optical sensing, while active optical sensing is typically regarded as
measuring rather than the default modality of vision. However, the situation
now changes: sensor technologies and data-driven paradigms empower active
optical sensing to redefine the boundaries of vision, ushering in a new era of
active vision. Light Detection and Ranging (LiDAR) sensors capture reflectance
from object surfaces, which remains invariant under varying illumination
conditions, showcasing significant potential in robotic perception tasks such
as detection, recognition, segmentation, and Simultaneous Localization and
Mapping (SLAM). These applications often rely on dense sensing capabilities,
typically achieved by high-resolution, expensive LiDAR sensors. A key challenge
with low-cost LiDARs lies in the sparsity of scan data, which limits their
broader application. To address this limitation, this work introduces an
innovative framework for generating dense LiDAR reflectance images from sparse
data, leveraging the unique attributes of non-repeating scanning LiDAR
(NRS-LiDAR). We tackle critical challenges, including reflectance calibration
and the transition from static to dynamic scene domains, facilitating the
reconstruction of dense reflectance images in real-world settings. The key
contributions of this work include a comprehensive dataset for LiDAR
reflectance image densification, a densification network tailored for
NRS-LiDAR, and diverse applications such as loop closure and traffic lane
detection using the generated dense reflectance images.

### 6. [Large Model Empowered Embodied AI: A Survey on Decision-Making and Embodied Learning](http://arxiv.org/pdf/2508.10399v1)

Authors: Wenlong Liang, Rui Zhou, Yang Ma, Bing Zhang, Songlin Li, Yijia Liao, Ping Kuang

Embodied AI aims to develop intelligent systems with physical forms capable
of perceiving, decision-making, acting, and learning in real-world
environments, providing a promising way to Artificial General Intelligence
(AGI). Despite decades of explorations, it remains challenging for embodied
agents to achieve human-level intelligence for general-purpose tasks in open
dynamic environments. Recent breakthroughs in large models have revolutionized
embodied AI by enhancing perception, interaction, planning and learning. In
this article, we provide a comprehensive survey on large model empowered
embodied AI, focusing on autonomous decision-making and embodied learning. We
investigate both hierarchical and end-to-end decision-making paradigms,
detailing how large models enhance high-level planning, low-level execution,
and feedback for hierarchical decision-making, and how large models enhance
Vision-Language-Action (VLA) models for end-to-end decision making. For
embodied learning, we introduce mainstream learning methodologies, elaborating
on how large models enhance imitation learning and reinforcement learning
in-depth. For the first time, we integrate world models into the survey of
embodied AI, presenting their design methods and critical roles in enhancing
decision-making and learning. Though solid advances have been achieved,
challenges still exist, which are discussed at the end of this survey,
potentially as the further research directions.

### 7. [KDPE: A Kernel Density Estimation Strategy for Diffusion Policy Trajectory Selection](http://arxiv.org/pdf/2508.10511v1)

Authors: Andrea Rosasco, Federico Ceola, Giulia Pasquale, Lorenzo Natale

Learning robot policies that capture multimodality in the training data has
been a long-standing open challenge for behavior cloning. Recent approaches
tackle the problem by modeling the conditional action distribution with
generative models. One of these approaches is Diffusion Policy, which relies on
a diffusion model to denoise random points into robot action trajectories.
While achieving state-of-the-art performance, it has two main drawbacks that
may lead the robot out of the data distribution during policy execution. First,
the stochasticity of the denoising process can highly impact on the quality of
generated trajectory of actions. Second, being a supervised learning approach,
it can learn data outliers from the dataset used for training. Recent work
focuses on mitigating these limitations by combining Diffusion Policy either
with large-scale training or with classical behavior cloning algorithms.
Instead, we propose KDPE, a Kernel Density Estimation-based strategy that
filters out potentially harmful trajectories output of Diffusion Policy while
keeping a low test-time computational overhead. For Kernel Density Estimation,
we propose a manifold-aware kernel to model a probability density function for
actions composed of end-effector Cartesian position, orientation, and gripper
state. KDPE overall achieves better performance than Diffusion Policy on
simulated single-arm tasks and real robot experiments.
  Additional material and code are available on our project page
https://hsp-iit.github.io/KDPE/.

### 8. [MLM: Learning Multi-task Loco-Manipulation Whole-Body Control for Quadruped Robot with Arm](http://arxiv.org/pdf/2508.10538v1)

Authors: Xin Liu, Bida Ma, Chenkun Qi, Yan Ding, Zhaxizhuoma, Guorong Zhang, Pengan Chen, Kehui Liu, Zhongjie Jia, Chuyue Guan, Yule Mo, Jiaqi Liu, Feng Gao, Jiangwei Zhong, Bin Zhao, Xuelong Li

Whole-body loco-manipulation for quadruped robots with arm remains a
challenging problem, particularly in achieving multi-task control. To address
this, we propose MLM, a reinforcement learning framework driven by both
real-world and simulation data. It enables a six-DoF robotic arm--equipped
quadruped robot to perform whole-body loco-manipulation for multiple tasks
autonomously or under human teleoperation. To address the problem of balancing
multiple tasks during the learning of loco-manipulation, we introduce a
trajectory library with an adaptive, curriculum-based sampling mechanism. This
approach allows the policy to efficiently leverage real-world collected
trajectories for learning multi-task loco-manipulation. To address deployment
scenarios with only historical observations and to enhance the performance of
policy execution across tasks with different spatial ranges, we propose a
Trajectory-Velocity Prediction policy network. It predicts unobservable future
trajectories and velocities. By leveraging extensive simulation data and
curriculum-based rewards, our controller achieves whole-body behaviors in
simulation and zero-shot transfer to real-world deployment. Ablation studies in
simulation verify the necessity and effectiveness of our approach, while
real-world experiments on the Go2 robot with an Airbot robotic arm demonstrate
the policy's good performance in multi-task execution.

### 9. [Biasing Frontier-Based Exploration with Saliency Areas](http://arxiv.org/pdf/2508.10689v1)

Authors: Matteo Luperto, Valerii Stakanov, Giacomo Boracchi, Nicola Basilico, Francesco Amigoni

Autonomous exploration is a widely studied problem where a robot
incrementally builds a map of a previously unknown environment. The robot
selects the next locations to reach using an exploration strategy. To do so,
the robot has to balance between competing objectives, like exploring the
entirety of the environment, while being as fast as possible. Most exploration
strategies try to maximise the explored area to speed up exploration; however,
they do not consider that parts of the environment are more important than
others, as they lead to the discovery of large unknown areas. We propose a
method that identifies \emph{saliency areas} as those areas that are of high
interest for exploration, by using saliency maps obtained from a neural network
that, given the current map, implements a termination criterion to estimate
whether the environment can be considered fully-explored or not. We use
saliency areas to bias some widely used exploration strategies, showing, with
an extensive experimental campaign, that this knowledge can significantly
influence the behavior of the robot during exploration.

### 10. [ReconVLA: Reconstructive Vision-Language-Action Model as Effective Robot Perceiver](http://arxiv.org/pdf/2508.10333v1)

Authors: Wenxuan Song, Ziyang Zhou, Han Zhao, Jiayi Chen, Pengxiang Ding, Haodong Yan, Yuxin Huang, Feilong Tang, Donglin Wang, Haoang Li

Recent advances in Vision-Language-Action (VLA) models have enabled robotic
agents to integrate multimodal understanding with action execution. However,
our empirical analysis reveals that current VLAs struggle to allocate visual
attention to target regions. Instead, visual attention is always dispersed. To
guide the visual attention grounding on the correct target, we propose
ReconVLA, a reconstructive VLA model with an implicit grounding paradigm.
Conditioned on the model's visual outputs, a diffusion transformer aims to
reconstruct the gaze region of the image, which corresponds to the target
manipulated objects. This process prompts the VLA model to learn fine-grained
representations and accurately allocate visual attention, thus effectively
leveraging task-specific visual information and conducting precise
manipulation. Moreover, we curate a large-scale pretraining dataset comprising
over 100k trajectories and 2 million data samples from open-source robotic
datasets, further boosting the model's generalization in visual reconstruction.
Extensive experiments in simulation and the real world demonstrate the
superiority of our implicit grounding method, showcasing its capabilities of
precise manipulation and generalization. Our project page is
https://zionchow.github.io/ReconVLA/.

### Software Engineering

### 1. [Bridging Solidity Evolution Gaps: An LLM-Enhanced Approach for Smart Contract Compilation Error Resolution](http://arxiv.org/pdf/2508.10517v1)

Authors: Likai Ye, Mengliang Li, Dehai Zhao, Jiamou Sun, Xiaoxue Ren

Solidity, the dominant smart contract language for Ethereum, has rapidly
evolved with frequent version updates to enhance security, functionality, and
developer experience. However, these continual changes introduce significant
challenges, particularly in compilation errors, code migration, and
maintenance. Therefore, we conduct an empirical study to investigate the
challenges in the Solidity version evolution and reveal that 81.68% of examined
contracts encounter errors when compiled across different versions, with 86.92%
of compilation errors.
  To mitigate these challenges, we conducted a systematic evaluation of large
language models (LLMs) for resolving Solidity compilation errors during version
migrations. Our empirical analysis across both open-source (LLaMA3, DeepSeek)
and closed-source (GPT-4o, GPT-3.5-turbo) LLMs reveals that although these
models exhibit error repair capabilities, their effectiveness diminishes
significantly for semantic-level issues and shows strong dependency on prompt
engineering strategies. This underscores the critical need for domain-specific
adaptation in developing reliable LLM-based repair systems for smart contracts.
  Building upon these insights, we introduce SMCFIXER, a novel framework that
systematically integrates expert knowledge retrieval with LLM-based repair
mechanisms for Solidity compilation error resolution. The architecture
comprises three core phases: (1) context-aware code slicing that extracts
relevant error information; (2) expert knowledge retrieval from official
documentation; and (3) iterative patch generation for Solidity migration.
Experimental validation across Solidity version migrations demonstrates our
approach's statistically significant 24.24% improvement over baseline GPT-4o on
real-world datasets, achieving near-perfect 96.97% accuracy.

### 2. [EVOSCAT: Exploring Software Change Dynamics in Large-Scale Historical Datasets](http://arxiv.org/pdf/2508.10852v1)

Authors: Souhaila Serbout, Diana Carolina Muñoz Hurtado, Hassan Atwi, Edoardo Riggio, Cesare Pautasso

Long lived software projects encompass a large number of artifacts, which
undergo many revisions throughout their history. Empirical software engineering
researchers studying software evolution gather and collect datasets with
millions of events, representing changes introduced to specific artifacts. In
this paper, we propose EvoScat, a tool that attempts addressing temporal
scalability through the usage of interactive density scatterplot to provide a
global overview of large historical datasets mined from open source
repositories in a single visualization. EvoScat intents to provide researchers
with a mean to produce scalable visualizations that can help them explore and
characterize evolution datasets, as well as comparing the histories of
individual artifacts, both in terms of 1) observing how rapidly different
artifacts age over multiple-year-long time spans 2) how often metrics
associated with each artifacts tend towards an improvement or worsening. The
paper shows how the tool can be tailored to specific analysis needs (pace of
change comparison, clone detection, freshness assessment) thanks to its support
for flexible configuration of history scaling and alignment along the time
axis, artifacts sorting and interactive color mapping, enabling the analysis of
millions of events obtained by mining the histories of tens of thousands of
software artifacts. We include in this paper a gallery showcasing datasets
gathering specific artifacts (OpenAPI descriptions, GitHub workflow
definitions) across multiple repositories, as well as diving into the history
of specific popular open source projects.

### 3. [Enabling Generic Robot Skill Implementation Using Object Oriented Programming](http://arxiv.org/pdf/2508.10497v1)

Authors: Abdullah Farrukh, Achim Wagner, Martin Ruskowski

Developing robotic algorithms and integrating a robotic subsystem into a
larger system can be a difficult task. Particularly in small and medium-sized
enterprises (SMEs) where robotics expertise is lacking, implementing,
maintaining and developing robotic systems can be a challenge. As a result,
many companies rely on external expertise through system integrators, which, in
some cases, can lead to vendor lock-in and external dependency. In the academic
research on intelligent manufacturing systems, robots play a critical role in
the design of robust autonomous systems. Similar challenges are faced by
researchers who want to use robotic systems as a component in a larger smart
system, without having to deal with the complexity and vastness of the robot
interfaces in detail. In this paper, we propose a software framework that
reduces the effort required to deploy a working robotic system. The focus is
solely on providing a concept for simplifying the different interfaces of a
modern robot system and using an abstraction layer for different manufacturers
and models. The Python programming language is used to implement a prototype of
the concept. The target system is a bin-picking cell containing a Yaskawa
Motoman GP4.

### Social and Information Networks

### 1. [Influence Maximization in Multi-layer Social Networks Based on Differentiated Graph Embeddings](http://arxiv.org/pdf/2508.10289v1)

Authors: Ronghua Lin, Runbin Yao, Yijia Wang, Junjie Lin, Zhengyang Wu, Yong Tang

Identifying influential nodes is crucial in social network analysis. Existing
methods often neglect local opinion leader tendencies, resulting in overlapping
influence ranges for seed nodes. Furthermore, approaches based on vanilla graph
neural networks (GNNs) struggle to effectively aggregate influence
characteristics during message passing, particularly with varying influence
intensities. Current techniques also fail to adequately address the multi-layer
nature of social networks and node heterogeneity. To address these issues, this
paper proposes Inf-MDE, a novel multi-layer influence maximization method
leveraging differentiated graph embedding. Inf-MDE models social relationships
using a multi-layer network structure. The model extracts a self-influence
propagation subgraph to eliminate the representation bias between node
embeddings and propagation dynamics. Additionally, Inf-MDE incorporates an
adaptive local influence aggregation mechanism within its GNN design. This
mechanism dynamically adjusts influence feature aggregation during message
passing based on local context and influence intensity, enabling it to
effectively capture both inter-layer propagation heterogeneity and intra-layer
diffusion dynamics. Extensive experiments across four distinct multi-layer
social network datasets demonstrate that Inf-MDE significantly outperforms
state-of-the-art methods.

### 2. [Online Homogeneity Can Emerge Without Filtering Algorithms or Homophily Preferences](http://arxiv.org/pdf/2508.10466v1)

Authors: Petter Törnberg

Ideologically homogeneous online environments - often described as "echo
chambers" or "filter bubbles" - are widely seen as drivers of polarization,
radicalization, and misinformation. A central debate asks whether such
homophily stems primarily from algorithmic curation or users' preference for
like-minded peers. This study challenges that view by showing that homogeneity
can emerge in the absence of both filtering algorithms and user preferences.
Using an agent-based model inspired by Schelling's model of residential
segregation, we demonstrate that weak individual preferences, combined with
simple group-based interaction structures, can trigger feedback loops that
drive communities toward segregation. Once a small imbalance forms, cascades of
user exits and regrouping amplify homogeneity across the system.
Counterintuitively, algorithmic filtering - often blamed for "filter bubbles" -
can in fact sustain diversity by stabilizing mixed communities. These findings
highlight online polarization as an emergent system-level dynamic and
underscore the importance of applying a complexity lens to the study of digital
public spheres.

### Systems and Control

### 1. [Quantifying the Value of Seismic Structural Health Monitoring for post-earthquake recovery of electric power system in terms of resilience enhancement](http://arxiv.org/pdf/2508.10318v1)

Authors: Huangbin Liang, Beatriz Moya, Francisco Chinesta, Eleni Chatzi

Post-earthquake recovery of electric power networks (EPNs) is critical to
community resilience. Traditional recovery processes often rely on prolonged
and imprecise manual inspections for damage diagnosis, leading to suboptimal
repair prioritization and extended service disruptions. Seismic Structural
Health Monitoring (SSHM) offers the potential to expedite recovery by enabling
more accurate and timely damage assessment. However, SSHM deployment incurs
costs, and its system-level resilience benefit remains underexplored. This
study proposes a probabilistic simulation framework to quantify the value of
SSHM for enhancing EPN resilience. The framework includes seismic damage
modeling based on network configuration, hazard intensity, fragility functions,
and damage-functionality mappings, combined with recovery simulations
incorporating resource constraints, repair and transfer durations. System
functionality is evaluated using graph-based island detection and optimal power
flow analysis. Resilience is quantified via the Lack of Resilience (LoR) metric
derived from the functionality restoration curve. SSHM is incorporated by
altering the quality of damage information used in repair scheduling. Different
monitoring scenarios (e.g., no-SSHM baseline, partial SSHM, full SSHM with
various accuracies) are modeled using confusion matrices to simulate damage
misclassification. Results show that improved damage awareness via SSHM
significantly accelerates recovery and reduces LoR by up to 21%. This work
supports evidence-based decisions for SSHM deployment in critical
infrastructure.

### 2. [A Structured Framework for Prioritizing Unsafe Control Actions in STPA: Case Study on eVTOL Operations](http://arxiv.org/pdf/2508.10446v1)

Authors: Halima El Badaoui

Systems Theoretic Process Analysis (STPA) is a widely recommended method for
analysing complex system safety. STPA can identify numerous Unsafe Control
Actions (UCAs) and requirements depending on the level of granularity of the
analysis and the complexity of the system being analysed. Managing numerous
results is challenging, especially during a fast-paced development lifecycle.
Extensive research has been done to optimize the efficiency of managing and
prioritising the STPA results. However, maintaining the objectivity of
prioritisation and communicating the prioritised results have become common
challenges. In this paper, the authors present a complementary approach that
incorporates inputs from both the safety analysts and domain experts to more
objectively prioritise UCAs. This is done by evaluating the severity of each
UCA, the impact factor of each controller or decision maker that issues the
UCA, and the ranking provided by the subject matter experts who assess the UCA
criticalities based on different factors. In addition, a Monte Carlo simulation
is introduced to reduce subjectivity and relativity, thus enabling more
objective prioritisation of the UCAs. As part of the approach to better
communicate the prioritisation results and plan the next steps of system
development, a dynamic-scaling prioritisation matrix was developed to capture
different sets of prioritised UCAs. The approach was applied to a real project
to improve the safe operations of Electric Vertical Take-off and Landing
(eVTOL). The results highlighted critical UCAs that need to be prioritised for
safer eVTOL operation. 318 UCAs were identified in total. Based on the
application of the prioritisation methodology, 110 were recognized as
high-priority UCAs to strengthen the system design.

### 3. [A Robust Optimization Approach for Demand Response Participation of Fixed-Frequency Air Conditioners](http://arxiv.org/pdf/2508.10679v1)

Authors: Jinhua He, Tingzhe Pan, Chao Li, Xin Jin, Zijie Meng, Wei Zhou

With the continuous increase in the penetration of renewable energy in the
emerging power systems, the pressure on system peak regulation has been
significantly intensified. Against this backdrop, demand side resources
particularly air conditioning loads have garnered considerable attention for
their substantial regulation potential and fast response capabilities, making
them promising candidates for providing auxiliary peak shaving services. This
study focuses on fixed frequency air conditioners (FFACs) and proposes an
optimization model and solution method for their participation in demand
response (DR) programs. First, a probabilistic response model for FFACs is
developed based on the Markov assumption. Second, by sampling this
probabilistic model, the aggregate power consumption of an FFAC cluster under
decentralized control is obtained. Subsequently, a robust optimization model is
formulated to maximize the profit of an aggregator managing the FFAC cluster
during DR events, taking into account the aggregated response power. The model
explicitly considers temperature uncertainty to ensure user comfort in a robust
sense. Finally, leveraging the structure of the proposed model, it is
reformulated as a mixed-integer linear programming (MILP) problem and solved
using a commercial optimization solver. Simulation results validate the
effectiveness of the proposed model and solution approach.

### 4. [Probabilistic Forecasting Method for Offshore Wind Farm Cluster under Typhoon Conditions: a Score-Based Conditional Diffusion Model](http://arxiv.org/pdf/2508.10705v1)

Authors: Jinhua He, Zechun Hu

Offshore wind power (OWP) exhibits significant fluctuations under typhoon
conditions, posing substantial challenges to the secure operation of power
systems. Accurate forecasting of OWP is therefore essential. However, the
inherent scarcity of historical typhoon data and stochasticity of OWP render
traditional point forecasting methods particularly difficult and inadequate. To
address this challenge and provide grid operators with the comprehensive
information necessary for decision-making, this study proposes a score-based
conditional diffusion model (SCDM) for probabilistic forecasting of OWP during
typhoon events. First, a knowledge graph algorithm is employed to embed
historical typhoon paths as vectors. Then, a deterministic network is
constructed to predict the wind power under typhoon conditions based on these
vector embeddings. Finally, to better characterize prediction errors, a
denoising network is developed. At the core of this approach is a
mean-reverting stochastic differential equation (SDE), which transforms complex
error distributions into a standard Gaussian, enabling the sampling of
forecasting errors using a reverse-time SDE. The probabilistic forecasting
results are reconstructed by combining deterministic forecasts with sampled
errors. The proposed method is evaluated using real-world data from a cluster
of 9 offshore wind farms. Results demonstrate that under typhoon conditions,
our approach outperforms baseline models for both deterministic and
probabilistic metrics, verifying the effectiveness of the approach.

### 5. [Multi-Functional Polarization-Based Coverage Control through Static Passive EMSs](http://arxiv.org/pdf/2508.10730v1)

Authors: Giacomo Oliveri, Francesco Zardi, Aaron Angel Salas Sanchez, Andrea Massa

An innovative multi-functional static-passive electromagnetic skin (SP-EMS)
solution is proposed to simultaneously support, in reflection, two independent
wave-manipulation functionalities with a single meta-atoms arrangement on the
EMS aperture when illuminated by two EM sources operating at the same
frequency, but working in different polarization states. Towards this end, a
simple reference meta-atom is designed first to enable an accurate and
independent control of each polarization component of the local reflection
tensor. Successively, the macro-scale synthesis of multi-polarization (MP)
SP-EMSs (MP-SP-EMSs) is carried out by solving a global optimization problem
where a cost function, which mathematically codes separate requirements for
each polarization, is minimized with a customized version of the
system-by-design (SbD) technique. Representative results from a set of
numerical and experimental tests are reported to assess the feasibility of a
multi-function EMS based on polarization diversity as well as the effectiveness
and the robustness of the proposed method for the synthesis of MP-SP-EMSs.

### 6. [Integrating Terrestrial and Non-Terrestrial Networks for Sustainable 6G Operations: A Latency-Aware Multi-Tier Cell-Switching Approach](http://arxiv.org/pdf/2508.10849v1)

Authors: Metin Ozturk, Maryam Salamatmoghadasi, Halim Yanikomeroglu

Sustainability is paramount in modern cellular networks, which face
significant energy consumption challenges from rising mobile traffic and
advancements in wireless technology. Cell-switching, well-established in
literature as an effective solution, encounters limitations such as inadequate
capacity and limited coverage when implemented through terrestrial networks
(TN). This study enhances cell-switching by integrating non-terrestrial
networks (NTN), including satellites (used for cell-switching for the first
time), high altitude platform stations (HAPS), and uncrewed aerial vehicles
(UAVs) into TN. This integration significantly boosts energy savings by
expanding capacity, enhancing coverage, and increasing operational flexibility.
We introduce a multi-tier cell-switching approach that dynamically offloads
users across network layers to manage energy effectively and minimize delays,
accommodating diverse user demands with a context aware strategy. Additionally,
we explore the role of artificial intelligence (AI), particularly generative
AI, in optimizing network efficiency through data compression, handover
optimization between different network layers, and enhancing device
compatibility, further improving the adaptability and energy efficiency of
cell-switching operations. A case study confirms substantial improvements in
network power consumption and user satisfaction, demonstrating the potential of
our approach for future networks.

### 7. [Fuel Consumption in Platoons: A Literature Review](http://arxiv.org/pdf/2508.10891v1)

Authors: Oumaima Barhoumi, Ghazal Farhani, Taufiq Rahman, Mohamed H. Zaki, Sofiène Tahar, Fadi Araji

Platooning has emerged as a promising strategy for improving fuel efficiency
in automated vehicle systems, with significant implications for reducing
emissions and operational costs. While existing literature on vehicle
platooning primarily focuses on individual aspects such as aerodynamic drag
reduction or specific control strategies, this work takes a more comprehensive
approach by bringing together a wide range of factors and components that
contribute to fuel savings in platoons. In this literature review, we examine
the impact of platooning on fuel consumption, highlighting the key components
of platoon systems, the factors and actors influencing fuel savings, methods
for estimating fuel use, and the effect of platoon instability on efficiency.
Furthermore, we study the role of reduced aerodynamic drag, vehicle
coordination, and the challenges posed by instability in real-world conditions.
By compiling insights from recent studies, this work provides a comprehensive
overview of the latest advancements in platooning technologies and highlights
both the challenges and opportunities for future research to maximize fuel
savings in real-world scenarios.

### 8. [MASH: Cooperative-Heterogeneous Multi-Agent Reinforcement Learning for Single Humanoid Robot Locomotion](http://arxiv.org/pdf/2508.10423v1)

Authors: Qi Liu, Xiaopeng Zhang, Mingshan Tan, Shuaikang Ma, Jinliang Ding, Yanjie Li

This paper proposes a novel method to enhance locomotion for a single
humanoid robot through cooperative-heterogeneous multi-agent deep reinforcement
learning (MARL). While most existing methods typically employ single-agent
reinforcement learning algorithms for a single humanoid robot or MARL
algorithms for multi-robot system tasks, we propose a distinct paradigm:
applying cooperative-heterogeneous MARL to optimize locomotion for a single
humanoid robot. The proposed method, multi-agent reinforcement learning for
single humanoid locomotion (MASH), treats each limb (legs and arms) as an
independent agent that explores the robot's action space while sharing a global
critic for cooperative learning. Experiments demonstrate that MASH accelerates
training convergence and improves whole-body cooperation ability, outperforming
conventional single-agent reinforcement learning methods. This work advances
the integration of MARL into single-humanoid-robot control, offering new
insights into efficient locomotion strategies.

### 9. [Feedback stabilization of a nanoparticle at the intensity minimum of an optical double-well potential](http://arxiv.org/pdf/2508.10601v1)

Authors: Vojtěch Mlynář, Salambô Dago, Jakob Rieser, Mario A. Ciampini, Markus Aspelmeyer, Nikolai Kiesel, Andreas Kugi, Andreas Deutschmann-Olek

In this work, we develop and analyze adaptive feedback control strategies to
stabilize and confine a nanoparticle at the unstable intensity minimum of an
optical double-well potential. The resulting stochastic optimal control problem
for a noise-driven mechanical particle in a nonlinear optical potential must
account for unavoidable experimental imperfections such as measurement
nonlinearities and slow drifts of the optical setup. To address these issues,
we simplify the model in the vicinity of the unstable equilibrium and employ
indirect adaptive control techniques to dynamically follow changes in the
potential landscape. Our approach leads to a simple and efficient Linear
Quadratic Gaussian (LQG) controller that can be implemented on fast and
cost-effective FPGAs, ensuring accessibility and reproducibility. We
demonstrate that this strategy successfully tracks the intensity minimum and
significantly reduces the nanoparticle's residual state variance, effectively
lowering its center-of-mass temperature. While conventional optical traps rely
on confining optical forces in the light field at the intensity maxima,
trapping at intensity minima mitigates absorption heating, which is crucial for
advanced quantum experiments. Since LQG control naturally extends into the
quantum regime, our results provide a promising pathway for future experiments
on quantum state preparation beyond the current absorption heating limitation,
like matter-wave interference and tests of the quantum-gravity interface.

### 10. [Synthesis of Deep Neural Networks with Safe Robust Adaptive Control for Reliable Operation of Wheeled Mobile Robots](http://arxiv.org/pdf/2508.10634v1)

Authors: Mehdi Heydari Shahna, Jouni Mattila

Deep neural networks (DNNs) can enable precise control while maintaining low
computational costs by circumventing the need for dynamic modeling. However,
the deployment of such black-box approaches remains challenging for heavy-duty
wheeled mobile robots (WMRs), which are subject to strict international
standards and prone to faults and disturbances. We designed a hierarchical
control policy for heavy-duty WMRs, monitored by two safety layers with
differing levels of authority. To this end, a DNN policy was trained and
deployed as the primary control strategy, providing high-precision performance
under nominal operating conditions. When external disturbances arise and reach
a level of intensity such that the system performance falls below a predefined
threshold, a low-level safety layer intervenes by deactivating the primary
control policy and activating a model-free robust adaptive control (RAC)
policy. This transition enables the system to continue operating while ensuring
stability by effectively managing the inherent trade-off between system
robustness and responsiveness. Regardless of the control policy in use, a
high-level safety layer continuously monitors system performance during
operation. It initiates a shutdown only when disturbances become sufficiently
severe such that compensation is no longer viable and continued operation would
jeopardize the system or its environment. The proposed synthesis of DNN and RAC
policy guarantees uniform exponential stability of the entire WMR system while
adhering to safety standards to some extent. The effectiveness of the proposed
approach was further validated through real-time experiments using a 6,000 kg
WMR.

### Machine Learning (Statistics Category)

### 1. [BKP: An R Package for Beta Kernel Process Modeling](http://arxiv.org/pdf/2508.10447v1)

Authors: Jiangyan Zhao, Kunhai Qing, Jin Xu

We present BKP, a user-friendly and extensible R package that implements the
Beta Kernel Process (BKP) -- a fully nonparametric and computationally
efficient framework for modeling spatially varying binomial probabilities. The
BKP model combines localized kernel-weighted likelihoods with conjugate beta
priors, resulting in closed-form posterior inference without requiring latent
variable augmentation or intensive MCMC sampling. The package supports binary
and aggregated binomial responses, allows flexible choices of kernel functions
and prior specification, and provides loss-based kernel hyperparameter tuning
procedures. In addition, BKP extends naturally to the Dirichlet Kernel Process
(DKP) for modeling spatially varying multinomial or compositional data. To our
knowledge, this is the first publicly available R package for implementing
BKP-based methods. We illustrate the use of BKP through several synthetic and
real-world datasets, highlighting its interpretability, accuracy, and
scalability. The package aims to facilitate practical application and future
methodological development of kernel-based beta modeling in statistics and
machine learning.

### 2. [A Guide to Bayesian Optimization in Bioprocess Engineering](http://arxiv.org/pdf/2508.10642v1)

Authors: Maximilian Siska, Emma Pajak, Katrin Rosenthal, Antonio del Rio Chanona, Eric von Lieres, Laura Marie Helleckes

Bayesian optimization has become widely popular across various experimental
sciences due to its favorable attributes: it can handle noisy data, perform
well with relatively small datasets, and provide adaptive suggestions for
sequential experimentation. While still in its infancy, Bayesian optimization
has recently gained traction in bioprocess engineering. However,
experimentation with biological systems is highly complex and the resulting
experimental uncertainty requires specific extensions to classical Bayesian
optimization. Moreover, current literature often targets readers with a strong
statistical background, limiting its accessibility for practitioners.
  In light of these developments, this review has two aims: first, to provide
an intuitive and practical introduction to Bayesian optimization; and second,
to outline promising application areas and open algorithmic challenges, thereby
highlighting opportunities for future research in machine learning.

### 3. [MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control](http://arxiv.org/pdf/2508.10684v1)

Authors: Yuchen Zhu, Wei Guo, Jaemoo Choi, Guan-Horng Liu, Yongxin Chen, Molei Tao

We study the problem of learning a neural sampler to generate samples from
discrete state spaces where the target probability mass function
$\pi\propto\mathrm{e}^{-U}$ is known up to a normalizing constant, which is an
important task in fields such as statistical physics, machine learning,
combinatorial optimization, etc. To better address this challenging task when
the state space has a large cardinality and the distribution is multi-modal, we
propose $\textbf{M}$asked $\textbf{D}$iffusion $\textbf{N}$eural
$\textbf{S}$ampler ($\textbf{MDNS}$), a novel framework for training discrete
neural samplers by aligning two path measures through a family of learning
objectives, theoretically grounded in the stochastic optimal control of the
continuous-time Markov chains. We validate the efficiency and scalability of
MDNS through extensive experiments on various distributions with distinct
statistical properties, where MDNS learns to accurately sample from the target
distributions despite the extremely high problem dimensions and outperforms
other learning-based baselines by a large margin. A comprehensive study of
ablations and extensions is also provided to demonstrate the efficacy and
potential of the proposed framework.

### 4. [Comparison of Data Reduction Criteria for Online Gaussian Processes](http://arxiv.org/pdf/2508.10815v1)

Authors: Thore Wietzke, Knut Graichen

Gaussian Processes (GPs) are widely used for regression and system
identification due to their flexibility and ability to quantify uncertainty.
However, their computational complexity limits their applicability to small
datasets. Moreover in a streaming scenario, more and more datapoints accumulate
which is intractable even for Sparse GPs. Online GPs aim to alleviate this
problem by e.g. defining a maximum budget of datapoints and removing redundant
datapoints. This work provides a unified comparison of several reduction
criteria, analyzing both their computational complexity and reduction behavior.
The criteria are evaluated on benchmark functions and real-world datasets,
including dynamic system identification tasks. Additionally, acceptance
criteria are proposed to further filter out redundant datapoints. This work
yields practical guidelines for choosing a suitable criterion for an online GP
algorithm.

### 5. [Conic Formulations of Transport Metrics for Unbalanced Measure Networks and Hypernetworks](http://arxiv.org/pdf/2508.10888v1)

Authors: Mary Chriselda Antony Oliver, Emmanuel Hartman, Tom Needham

The Gromov-Wasserstein (GW) variant of optimal transport, designed to compare
probability densities defined over distinct metric spaces, has emerged as an
important tool for the analysis of data with complex structure, such as
ensembles of point clouds or networks. To overcome certain limitations, such as
the restriction to comparisons of measures of equal mass and sensitivity to
outliers, several unbalanced or partial transport relaxations of the GW
distance have been introduced in the recent literature. This paper is concerned
with the Conic Gromov-Wasserstein (CGW) distance introduced by
S\'{e}journ\'{e}, Vialard, and Peyr\'{e}. We provide a novel formulation in
terms of semi-couplings, and extend the framework beyond the metric measure
space setting, to compare more general network and hypernetwork structures.
With this new formulation, we establish several fundamental properties of the
CGW metric, including its scaling behavior under dilation, variational
convergence in the limit of volume growth constraints, and comparison bounds
with established optimal transport metrics. We further derive quantitative
bounds that characterize the robustness of the CGW metric to perturbations in
the underlying measures. The hypernetwork formulation of CGW admits a simple
and provably convergent block coordinate ascent algorithm for its estimation,
and we demonstrate the computational tractability and scalability of our
approach through experiments on synthetic and real-world high-dimensional and
structured datasets.

### 6. [The Conditional Regret-Capacity Theorem for Batch Universal Prediction](http://arxiv.org/pdf/2508.10282v1)

Authors: Marco Bondaschi, Michael Gastpar

We derive a conditional version of the classical regret-capacity theorem.
This result can be used in universal prediction to find lower bounds on the
minimal batch regret, which is a recently introduced generalization of the
average regret, when batches of training data are available to the predictor.
As an example, we apply this result to the class of binary memoryless sources.
Finally, we generalize the theorem to R\'enyi information measures, revealing a
deep connection between the conditional R\'enyi divergence and the conditional
Sibson's mutual information.

### 7. [Uncertainty-Aware Prediction of Parkinson's Disease Medication Needs: A Two-Stage Conformal Prediction Approach](http://arxiv.org/pdf/2508.10284v1)

Authors: Ricardo Diaz-Rincon, Muxuan Liang, Adolfo Ramirez-Zamora, Benjamin Shickel

Parkinson's Disease (PD) medication management presents unique challenges due
to heterogeneous disease progression and treatment response. Neurologists must
balance symptom control with optimal dopaminergic dosing based on functional
disability while minimizing side effects. This balance is crucial as inadequate
or abrupt changes can cause levodopa-induced dyskinesia, wearing off, and
neuropsychiatric effects, significantly reducing quality of life. Current
approaches rely on trial-and-error decisions without systematic predictive
methods. Despite machine learning advances, clinical adoption remains limited
due to reliance on point predictions that do not account for prediction
uncertainty, undermining clinical trust and utility. Clinicians require not
only predictions of future medication needs but also reliable confidence
measures. Without quantified uncertainty, adjustments risk premature escalation
to maximum doses or prolonged inadequate symptom control. We developed a
conformal prediction framework anticipating medication needs up to two years in
advance with reliable prediction intervals and statistical guarantees. Our
approach addresses zero-inflation in PD inpatient data, where patients maintain
stable medication regimens between visits. Using electronic health records from
631 inpatient admissions at University of Florida Health (2011-2021), our
two-stage approach identifies patients likely to need medication changes, then
predicts required levodopa equivalent daily dose adjustments. Our framework
achieved marginal coverage while reducing prediction interval lengths compared
to traditional approaches, providing precise predictions for short-term
planning and wider ranges for long-term forecasting. By quantifying
uncertainty, our approach enables evidence-based decisions about levodopa
dosing, optimizing symptom control while minimizing side effects and improving
life quality.

### 8. [Online selective conformal inference: adaptive scores, convergence rate and optimality](http://arxiv.org/pdf/2508.10336v1)

Authors: Pierre Humbert, Ulysse Gazin, Ruth Heller, Etienne Roquain

In a supervised online setting, quantifying uncertainty has been proposed in
the seminal work of \cite{gibbs2021adaptive}. For any given point-prediction
algorithm, their method (ACI) produces a conformal prediction set with an
average missed coverage getting close to a pre-specified level $\alpha$ for a
long time horizon. We introduce an extended version of this algorithm, called
OnlineSCI, allowing the user to additionally select times where such an
inference should be made. OnlineSCI encompasses several prominent online
selective tasks, such as building prediction intervals for extreme outcomes,
classification with abstention, and online testing. While OnlineSCI controls
the average missed coverage on the selected in an adversarial setting, our
theoretical results also show that it controls the instantaneous error rate
(IER) at the selected times, up to a non-asymptotical remainder term.
Importantly, our theory covers the case where OnlineSCI updates the
point-prediction algorithm at each time step, a property which we refer to as
{\it adaptive} capability. We show that the adaptive versions of OnlineSCI can
convergence to an optimal solution and provide an explicit convergence rate in
each of the aforementioned application cases, under specific mild conditions.
Finally, the favorable behavior of OnlineSCI in practice is illustrated by
numerical experiments.

### 9. [Unpacking the Implicit Norm Dynamics of Sharpness-Aware Minimization in Tensorized Models](http://arxiv.org/pdf/2508.10435v1)

Authors: Tianxiao Cao, Kyohei Atarashi, Hisashi Kashima

Sharpness-Aware Minimization (SAM) has been proven to be an effective
optimization technique for improving generalization in overparameterized
models. While prior works have explored the implicit regularization of SAM in
simple two-core scale-invariant settings, its behavior in more general
tensorized or scale-invariant models remains underexplored. In this work, we
leverage scale-invariance to analyze the norm dynamics of SAM in general
tensorized models. We introduce the notion of \emph{Norm Deviation} as a global
measure of core norm imbalance, and derive its evolution under SAM using
gradient flow analysis. We show that SAM's implicit control of Norm Deviation
is governed by the covariance between core norms and their gradient magnitudes.
Motivated by these findings, we propose a simple yet effective method,
\emph{Deviation-Aware Scaling (DAS)}, which explicitly mimics this
regularization behavior by scaling core norms in a data-adaptive manner. Our
experiments across tensor completion, noisy training, model compression, and
parameter-efficient fine-tuning confirm that DAS achieves competitive or
improved performance over SAM, while offering reduced computational overhead.

### 10. [Confounding is a Pervasive Problem in Real World Recommender Systems](http://arxiv.org/pdf/2508.10479v1)

Authors: Alexander Merkov, David Rohde, Alexandre Gilotte, Benjamin Heymann

Unobserved confounding arises when an unmeasured feature influences both the
treatment and the outcome, leading to biased causal effect estimates. This
issue undermines observational studies in fields like economics, medicine,
ecology or epidemiology. Recommender systems leveraging fully observed data
seem not to be vulnerable to this problem. However many standard practices in
recommender systems result in observed features being ignored, resulting in
effectively the same problem. This paper will show that numerous common
practices such as feature engineering, A/B testing and modularization can in
fact introduce confounding into recommendation systems and hamper their
performance. Several illustrations of the phenomena are provided, supported by
simulation studies with practical suggestions about how practitioners may
reduce or avoid the affects of confounding in real systems.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-08-15 PST.

### 1. [AI helps assemble ‘brain’ of future quantum computer](https://www.nature.com/articles/d41586-025-02577-9)

Authors: Jenna  Ahart

### 2. [A flying ad-hoc network dataset for early time series classification of grey hole attacks](https://www.nature.com/articles/s41597-025-05560-1)

Authors: Charles Hutchins et al.

### 3. [The analysis of fraud detection in financial market under machine learning](https://www.nature.com/articles/s41598-025-15783-2)

Authors: Jing Jin et al.

### 4. [Modelling the spread of infectious diseases in public transport systems under varying demand patterns and capacity constraints](https://www.nature.com/articles/s41598-025-15237-9)

Authors: László Hajdu et al.

