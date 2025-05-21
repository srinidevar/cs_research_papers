# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-20 18:30:54.360176 PST.

### Artificial Intelligence

### 1. [$\texttt{DIAMONDs}$: A Dataset for $\mathbb{D}$ynamic $\mathbb{I}$nformation $\mathbb{A}$nd $\mathbb{M}$ental modeling $\mathbb{O}$f $\mathbb{N}$umeric $\mathbb{D}$iscussions](http://arxiv.org/pdf/2505.12651v1)

Authors: Sayontan Ghosh, Mahnaz Koupaee, Yash Kumar Lal, Pegah Alipoormolabashi, Mohammad Saqib Hasan, Jun Seok Kang, Niranjan Balasubramanian

Understanding multiparty conversations demands robust Theory of Mind (ToM)
capabilities, including the ability to track dynamic information, manage
knowledge asymmetries, and distinguish relevant information across extended
exchanges. To advance ToM evaluation in such settings, we present a carefully
designed scalable methodology for generating high-quality benchmark
conversation-question pairs with these characteristics. Using this methodology,
we create $\texttt{DIAMONDs}$, a new conversational QA dataset covering common
business, financial or other group interactions. In these goal-oriented
conversations, participants often have to track certain numerical quantities
(say $\textit{expected profit}$) of interest that can be derived from other
variable quantities (like $\textit{marketing expenses, expected sales,
salary}$, etc.), whose values also change over the course of the conversation.
$\texttt{DIAMONDs}$ questions pose simple numerical reasoning problems over
such quantities of interest (e.g., $\textit{funds required for charity events,
expected company profit next quarter}$, etc.) in the context of the information
exchanged in conversations. This allows for precisely evaluating ToM
capabilities for carefully tracking and reasoning over participants' knowledge
states.
  Our evaluation of state-of-the-art language models reveals significant
challenges in handling participant-centric reasoning, specifically in
situations where participants have false beliefs. Models also struggle with
conversations containing distractors and show limited ability to identify
scenarios with insufficient information. These findings highlight current
models' ToM limitations in handling real-world multi-party conversations.

### 2. [Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](http://arxiv.org/pdf/2505.12731v1)

Authors: Jie Ou, Jinyu Guo, Shuaihong Jiang, Zhaokun Wang, Libo Qin, Shunyu Yao, Wenhong Tian

Retrieval-augmented generation (RAG) has emerged as a pivotal method for
expanding the knowledge of large language models. To handle complex queries
more effectively, researchers developed Adaptive-RAG (A-RAG) to enhance the
generated quality through multiple interactions with external knowledge bases.
Despite its effectiveness, A-RAG exacerbates the pre-existing efficiency
challenges inherent in RAG, which are attributable to its reliance on multiple
iterations of generation. Existing A-RAG approaches process all retrieved
contents from scratch. However, they ignore the situation where there is a
significant overlap in the content of the retrieval results across rounds. The
overlapping content is redundantly represented, which leads to a large
proportion of repeated computations, thus affecting the overall efficiency. To
address this issue, this paper introduces a model-agnostic approach that can be
generally applied to A-RAG methods, which is dedicated to reducing the
redundant representation process caused by the overlapping of retrieval
results. Specifically, we use cache access and parallel generation to speed up
the prefilling and decoding stages respectively. Additionally, we also propose
an instruction-driven module to further guide the model to more effectively
attend to each part of the content in a more suitable way for LLMs. Experiments
show that our approach achieves 2.79 and 2.33 times significant acceleration on
average for prefilling and decoding respectively while maintaining equal
generation quality.

### 3. [Dense Communication between Language Models](http://arxiv.org/pdf/2505.12741v1)

Authors: Shiguang Wu, Yaqing Wang, Quanming Yao

As higher-level intelligence emerges from the combination of modular
components with lower-level intelligence, many works combines Large Language
Models (LLMs) for collective intelligence. Such combination is achieved by
building communications among LLMs. While current systems primarily facilitate
such communication through natural language, this paper proposes a novel
paradigm of direct dense vector communication between LLMs. Our approach
eliminates the unnecessary embedding and de-embedding steps when LLM interact
with another, enabling more efficient information transfer, fully
differentiable optimization pathways, and exploration of capabilities beyond
human heuristics. We use such stripped LLMs as vertexes and optimizable seq2seq
modules as edges to construct LMNet, with similar structure as MLPs. By
utilizing smaller pre-trained LLMs as vertexes, we train a LMNet that achieves
comparable performance with LLMs in similar size with only less than 0.1%
training cost. This offers a new perspective on scaling for general
intelligence rather than training a monolithic LLM from scratch. Besides, the
proposed method can be used for other applications, like customizing LLM with
limited data, showing its versatility.

### 4. [Incentivizing Multimodal Reasoning in Large Models for Direct Robot Manipulation](http://arxiv.org/pdf/2505.12744v1)

Authors: Weiliang Tang, Dong Jing, Jia-Hui Pan, Zhiwu Lu, Yun-Hui Liu, Li Erran Li, Mingyu Ding, Chi-Wing Fu

Recent Large Multimodal Models have demonstrated remarkable reasoning
capabilities, especially in solving complex mathematical problems and realizing
accurate spatial perception. Our key insight is that these emerging abilities
can naturally extend to robotic manipulation by enabling LMMs to directly infer
the next goal in language via reasoning, rather than relying on a separate
action head. However, this paradigm meets two main challenges: i) How to make
LMMs understand the spatial action space, and ii) How to fully exploit the
reasoning capacity of LMMs in solving these tasks. To tackle the former
challenge, we propose a novel task formulation, which inputs the current states
of object parts and the gripper, and reformulates rotation by a new axis
representation instead of traditional Euler angles. This representation is more
compatible with spatial reasoning and easier to interpret within a unified
language space. For the latter challenge, we design a pipeline to utilize
cutting-edge LMMs to generate a small but high-quality reasoning dataset of
multi-round dialogues that successfully solve manipulation tasks for supervised
fine-tuning. Then, we perform reinforcement learning by trial-and-error
interactions in simulation to further enhance the model's reasoning abilities
for robotic manipulation. Our resulting reasoning model built upon a 7B
backbone, named ReasonManip, demonstrates three notable advantages driven by
its system-2 level reasoning capabilities: i) exceptional generalizability to
out-of-distribution environments, objects, and tasks; ii) inherent sim-to-real
transfer ability enabled by the unified language representation shared across
domains; iii) transparent interpretability connecting high-level reasoning and
low-level control. Extensive experiments demonstrate the effectiveness of the
proposed paradigm and its potential to advance LMM-driven robotic manipulation.

### 5. [Correspondence of high-dimensional emotion structures elicited by video clips between humans and Multimodal LLMs](http://arxiv.org/pdf/2505.12746v1)

Authors: Haruka Asanuma, Naoko Koide-Majima, Ken Nakamura, Takato Horii, Shinji Nishimoto, Masafumi Oizumi

Recent studies have revealed that human emotions exhibit a high-dimensional,
complex structure. A full capturing of this complexity requires new approaches,
as conventional models that disregard high dimensionality risk overlooking key
nuances of human emotions. Here, we examined the extent to which the latest
generation of rapidly evolving Multimodal Large Language Models (MLLMs) capture
these high-dimensional, intricate emotion structures, including capabilities
and limitations. Specifically, we compared self-reported emotion ratings from
participants watching videos with model-generated estimates (e.g., Gemini or
GPT). We evaluated performance not only at the individual video level but also
from emotion structures that account for inter-video relationships. At the
level of simple correlation between emotion structures, our results
demonstrated strong similarity between human and model-inferred emotion
structures. To further explore whether the similarity between humans and models
is at the signle item level or the coarse-categorical level, we applied Gromov
Wasserstein Optimal Transport. We found that although performance was not
necessarily high at the strict, single-item level, performance across video
categories that elicit similar emotions was substantial, indicating that the
model could infer human emotional experiences at the category level. Our
results suggest that current state-of-the-art MLLMs broadly capture the complex
high-dimensional emotion structures at the category level, as well as their
apparent limitations in accurately capturing entire structures at the
single-item level.

### 6. [IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment](http://arxiv.org/pdf/2505.12762v1)

Authors: Chenlin Ming, Chendi Qu, Mengzhang Cai, Qizhi Pei, Zhuoshi Pan, Yu Li, Xiaoming Duan, Lijun Wu, Conghui He

Large Language Models (LLMs) have achieved impressive performance through
Supervised Fine-tuning (SFT) on diverse instructional datasets. When training
on multiple capabilities simultaneously, the mixture training dataset, governed
by volumes of data from different domains, is a critical factor that directly
impacts the final model's performance. Unlike many studies that focus on
enhancing the quality of training datasets through data selection methods, few
works explore the intricate relationship between the compositional quantity of
mixture training datasets and the emergent capabilities of LLMs. Given the
availability of a high-quality multi-domain training dataset, understanding the
impact of data from each domain on the model's overall capabilities is crucial
for preparing SFT data and training a well-balanced model that performs
effectively across diverse domains. In this work, we introduce IDEAL, an
innovative data equilibrium adaptation framework designed to effectively
optimize volumes of data from different domains within mixture SFT datasets,
thereby enhancing the model's alignment and performance across multiple
capabilities. IDEAL employs a gradient-based approach to iteratively refine the
training data distribution, dynamically adjusting the volumes of
domain-specific data based on their impact on downstream task performance. By
leveraging this adaptive mechanism, IDEAL ensures a balanced dataset
composition, enabling the model to achieve robust generalization and consistent
proficiency across diverse tasks. Experiments across different capabilities
demonstrate that IDEAL outperforms conventional uniform data allocation
strategies, achieving a comprehensive improvement of approximately 7% in
multi-task evaluation scores.

### 7. [Language Models That Walk the Talk: A Framework for Formal Fairness Certificates](http://arxiv.org/pdf/2505.12767v1)

Authors: Danqing Chen, Tobias Ladner, Ahmed Rayen Mhadhbi, Matthias Althoff

As large language models become integral to high-stakes applications,
ensuring their robustness and fairness is critical. Despite their success,
large language models remain vulnerable to adversarial attacks, where small
perturbations, such as synonym substitutions, can alter model predictions,
posing risks in fairness-critical areas, such as gender bias mitigation, and
safety-critical areas, such as toxicity detection. While formal verification
has been explored for neural networks, its application to large language models
remains limited. This work presents a holistic verification framework to
certify the robustness of transformer-based language models, with a focus on
ensuring gender fairness and consistent outputs across different gender-related
terms. Furthermore, we extend this methodology to toxicity detection, offering
formal guarantees that adversarially manipulated toxic inputs are consistently
detected and appropriately censored, thereby ensuring the reliability of
moderation systems. By formalizing robustness within the embedding space, this
work strengthens the reliability of language models in ethical AI deployment
and content moderation.

### 8. [Mixture Policy based Multi-Hop Reasoning over N-tuple Temporal Knowledge Graphs](http://arxiv.org/pdf/2505.12788v1)

Authors: Zhongni Hou, Miao Su, Xiaolong Jin, Zixuan Li, Long Bai, Jiafeng Guo, Xueqi Cheng

Temporal Knowledge Graphs (TKGs), which utilize quadruples in the form of
(subject, predicate, object, timestamp) to describe temporal facts, have
attracted extensive attention. N-tuple TKGs (N-TKGs) further extend traditional
TKGs by utilizing n-tuples to incorporate auxiliary elements alongside core
elements (i.e., subject, predicate, and object) of facts, so as to represent
them in a more fine-grained manner. Reasoning over N-TKGs aims to predict
potential future facts based on historical ones. However, existing N-TKG
reasoning methods often lack explainability due to their black-box nature.
Therefore, we introduce a new Reinforcement Learning-based method, named
MT-Path, which leverages the temporal information to traverse historical
n-tuples and construct a temporal reasoning path. Specifically, in order to
integrate the information encapsulated within n-tuples, i.e., the
entity-irrelevant information within the predicate, the information about core
elements, and the complete information about the entire n-tuples, MT-Path
utilizes a mixture policy-driven action selector, which bases on three
low-level policies, namely, the predicate-focused policy, the
core-element-focused policy and the whole-fact-focused policy. Further, MT-Path
utilizes an auxiliary element-aware GCN to capture the rich semantic
dependencies among facts, thereby enabling the agent to gain a deep
understanding of each n-tuple. Experimental results demonstrate the
effectiveness and the explainability of MT-Path.

### 9. [Emergent Specialization: Rare Token Neurons in Language Models](http://arxiv.org/pdf/2505.12822v1)

Authors: Jing Liu, Haozheng Wang, Yueheng Li

Large language models struggle with representing and generating rare tokens
despite their importance in specialized domains. In this study, we identify
neuron structures with exceptionally strong influence on language model's
prediction of rare tokens, termed as rare token neurons, and investigate the
mechanism for their emergence and behavior. These neurons exhibit a
characteristic three-phase organization (plateau, power-law, and rapid decay)
that emerges dynamically during training, evolving from a homogeneous initial
state to a functionally differentiated architecture. In the activation space,
rare token neurons form a coordinated subnetwork that selectively co-activates
while avoiding co-activation with other neurons. This functional specialization
potentially correlates with the development of heavy-tailed weight
distributions, suggesting a statistical mechanical basis for emergent
specialization.

### 10. [Reasoning BO: Enhancing Bayesian Optimization with Long-Context Reasoning Power of LLMs](http://arxiv.org/pdf/2505.12833v1)

Authors: Zhuo Yang, Lingli Ge, Dong Han, Tianfan Fu, Yuqiang Li

Many real-world scientific and industrial applications require the
optimization of expensive black-box functions. Bayesian Optimization (BO)
provides an effective framework for such problems. However, traditional BO
methods are prone to get trapped in local optima and often lack interpretable
insights. To address this issue, this paper designs Reasoning BO, a novel
framework that leverages reasoning models to guide the sampling process in BO
while incorporating multi-agent systems and knowledge graphs for online
knowledge accumulation. By integrating the reasoning and contextual
understanding capabilities of Large Language Models (LLMs), we can provide
strong guidance to enhance the BO process. As the optimization progresses,
Reasoning BO provides real-time sampling recommendations along with critical
insights grounded in plausible scientific theories, aiding in the discovery of
superior solutions within the search space. We systematically evaluate our
approach across 10 diverse tasks encompassing synthetic mathematical functions
and complex real-world applications. The framework demonstrates its capability
to progressively refine sampling strategies through real-time insights and
hypothesis evolution, effectively identifying higher-performing regions of the
search space for focused exploration. This process highlights the powerful
reasoning and context-learning abilities of LLMs in optimization scenarios. For
example, in the Direct Arylation task, our method increased the yield to 60.7%,
whereas traditional BO achieved only a 25.2% yield. Furthermore, our
investigation reveals that smaller LLMs, when fine-tuned through reinforcement
learning, can attain comparable performance to their larger counterparts. This
enhanced reasoning capability paves the way for more efficient automated
scientific experimentation while maintaining computational feasibility.

### Hardware Architecture

### 1. [FireFly-T: High-Throughput Sparsity Exploitation for Spiking Transformer Acceleration with Dual-Engine Overlay Architecture](http://arxiv.org/pdf/2505.12771v1)

Authors: Tenglong Li, Jindong Li, Guobin Shen, Dongcheng Zhao, Qian Zhang, Yi Zeng

Spiking transformers are emerging as a promising architecture that combines
the energy efficiency of Spiking Neural Networks (SNNs) with the powerful
attention mechanisms of transformers. However, existing hardware accelerators
lack support for spiking attention, exhibit limited throughput in exploiting
fine-grained sparsity, and struggle with scalable parallelism in sparse
computation. To address these, we propose FireFly-T, a dual-engine overlay
architecture that integrates a sparse engine for activation sparsity and a
binary engine for spiking attention. In the sparse engine, we propose a
highthroughput sparse decoder that exploits fine-grained sparsity by
concurrently extracting multiple non-zero spikes. To complement this, we
introduce a scalable load balancing mechanism with weight dispatch and
out-of-order execution, eliminating bank conflicts to support scalable
multidimensional parallelism. In the binary engine, we leverage the byte-level
write capability of SRAMs to efficiently manipulate the 3D dataflows required
for spiking attention with minimal resource overhead. We also optimize the core
AND-PopCount operation in spiking attention through a LUT6-based
implementation, improving timing closure and reducing LUT utilization on Xilinx
FPGAs. As an overlay architecture, FireFly-T further incorporates an
orchestrator that dynamically manipulates input dataflows with flexible
adaptation for diverse network topologies, while ensuring efficient resource
utilization and maintaining high throughput. Experimental results demonstrate
that our accelerator achieves $1.39\times$ and $2.40\times$ higher energy
efficiency, as well as $4.21\times$ and $7.10\times$ greater DSP efficiency,
compared to FireFly v2 and the transformer-enabled SpikeTA, respectively. These
results highlight its potential as an efficient hardware platform for spiking
transformer.

### 2. [Addressing memory bandwidth scalability in vector processors for streaming applications](http://arxiv.org/pdf/2505.12856v1)

Authors: Jordi Altayo, Paul Delestrac, David Novo, Simey Yang, Debjyoti Bhattacharjee, Francky Catthoor

As the size of artificial intelligence and machine learning (AI/ML) models
and datasets grows, the memory bandwidth becomes a critical bottleneck. The
paper presents a novel extended memory hierarchy that addresses some major
memory bandwidth challenges in data-parallel AI/ML applications. While
data-parallel architectures like GPUs and neural network accelerators have
improved power performance compared to traditional CPUs, they can still be
significantly bottlenecked by their memory bandwidth, especially when the data
reuse in the loop kernels is limited. Systolic arrays (SAs) and GPUs attempt to
mitigate the memory bandwidth bottleneck but can still become memory bandwidth
throttled when the amount of data reuse is not sufficient to confine data
access mostly to the local memories near to the processing. To mitigate this,
the proposed architecture introduces three levels of on-chip memory -- local,
intermediate, and global -- with an ultra-wide register and data-shufflers to
improve versatility and adaptivity to varying data-parallel applications. The
paper explains the innovations at a conceptual level and presents a detailed
description of the architecture innovations. We also map a representative
data-parallel application, like a convolutional neural network (CNN), to the
proposed architecture and quantify the benefits vis-a-vis GPUs and
repersentative accelerators based on systolic arrays and vector processors.

### 3. [PIM-malloc: A Fast and Scalable Dynamic Memory Allocator for Processing-In-Memory (PIM) Architectures](http://arxiv.org/pdf/2505.13002v2)

Authors: Dongjae Lee, Bongjoon Hyun, Youngjin Kwon, Minsoo Rhu

Dynamic memory allocation is essential in modern programming but remains
under-supported in current PIM devices. In this work, we first conduct a design
space exploration of PIM memory allocators, examining optimal metadata
placement and management strategies. Building on these insights, we propose
PIM-malloc, a fast and scalable allocator for real PIM hardware, improving
allocation performance by $66\times$. We further enhance this design with a
lightweight, per-PIM core hardware cache for dynamic allocation, achieving an
additional $31\%$ performance gain. Finally, we demonstrate the effectiveness
of PIM-malloc using a dynamic graph update workload, achieving a $28\times$
throughput increase.

### 4. [MXDOTP: A RISC-V ISA Extension for Enabling Microscaling (MX) Floating-Point Dot Products](http://arxiv.org/pdf/2505.13159v1)

Authors: Gamze İslamoğlu, Luca Bertaccini, Arpan Suravi Prasad, Francesco Conti, Angelo Garofalo, Luca Benini

Fast and energy-efficient low-bitwidth floating-point (FP) arithmetic is
essential for Artificial Intelligence (AI) systems. Microscaling (MX)
standardized formats have recently emerged as a promising alternative to
baseline low-bitwidth FP formats, offering improved accuracy with a block-wise
shared exponent scale combined with per-element values. However, efficiently
executing the key linear algebra primitives for AI applications on MX formats
requires specialized hardware support for the fundamental operators such as
scaled dot product. In this work, we propose MXDOTP, the first RISC-V ISA
extension for MX dot products, focusing on the 8-bit MXFP8 FP format. We extend
the open-source Snitch RISC-V core with a dedicated MXFP8 dot
product-accumulate unit, which fully consumes blocks of eight 8-bit operands
packed into 64-bit inputs. To feed MXDOTP at full utilization with four
operands per cycle, including block scales, we exploit Snitch's Stream Semantic
Registers (SSRs), achieving up to 80% utilization with minimal impact on the
Snitch core's architecture and no modification to the register file.
Implemented in 12 nm FinFET, a cluster with eight MXDOTP-extended cores reaches
up to 356 GFLOPS/W when computing MXFP8 matrix multiplications at 0.8 V, 1 GHz.
Compared to a software baseline, where MX dot products are computed by type
casting FP8 inputs to FP32 for higher accumulation precision and applying
explicit block scaling, the cluster achieves 25x speedup and 12.5x better
energy efficiency at a minimal 5.1% area increase.

### 5. [Introducing Instruction-Accurate Simulators for Performance Estimation of Autotuning Workloads](http://arxiv.org/pdf/2505.13357v1)

Authors: Rebecca Pelke, Nils Bosbach, Lennart M. Reimann, Rainer Leupers

Accelerating Machine Learning (ML) workloads requires efficient methods due
to their large optimization space. Autotuning has emerged as an effective
approach for systematically evaluating variations of implementations.
Traditionally, autotuning requires the workloads to be executed on the target
hardware (HW). We present an interface that allows executing autotuning
workloads on simulators. This approach offers high scalability when the
availability of the target HW is limited, as many simulations can be run in
parallel on any accessible HW. Additionally, we evaluate the feasibility of
using fast instruction-accurate simulators for autotuning. We train various
predictors to forecast the performance of ML workload implementations on the
target HW based on simulation statistics. Our results demonstrate that the
tuned predictors are highly effective. The best workload implementation in
terms of actual run time on the target HW is always within the top 3 % of
predictions for the tested x86, ARM, and RISC-V-based architectures. In the
best case, this approach outperforms native execution on the target HW for
embedded architectures when running as few as three samples on three simulators
in parallel.

### 6. [2T1R Regulated Memristor Conductance Control Array Architecture for Neuromorphic Computing using 28nm CMOS Technology](http://arxiv.org/pdf/2505.12830v1)

Authors: Neethu Kuriakose, Arun Ashok, Christian Grewing, André Zambanini, Stefan van Waasen

Memristors are promising devices for scalable and low power, in-memory
computing to improve the energy efficiency of a rising computational demand.
The crossbar array architecture with memristors is used for vector matrix
multiplication (VMM) and acts as kernels in neuromorphic computing. The analog
conductance control in a memristor is achieved by applying voltage or current
through it. A basic 1T1R array is suitable to avoid sneak path issues but
suffer from wire resistances, which affects the read and write procedures. A
conductance control scheme with a regulated voltage source will improve the
architecture and reduce the possible potential divider effects. A change in
conductance is also possible with the provision of a regulated current source
and measuring the voltage across the memristors. A regulated 2T1R memristor
conductance control architecture is proposed in this work, which avoids the
potential divider effect and virtual ground scenario in a regular crossbar
scheme, as well as conductance control by passing a regulated current through
memristors. The sneak path current is not allowed to pass by the provision of
ground potential to both terminals of memristors.

### Computational Complexity

### 1. [A near-optimal Quadratic Goldreich-Levin algorithm](http://arxiv.org/pdf/2505.13134v1)

Authors: Jop Briët, Davi Castro-Silva

In this paper, we give a quadratic Goldreich-Levin algorithm that is close to
optimal in the following ways. Given a bounded function $f$ on the Boolean
hypercube $\mathbb{F}_2^n$ and any $\varepsilon>0$, the algorithm returns a
quadratic polynomial $q: \mathbb{F}_2^n \to \mathbb{F}_2$ so that the
correlation of $f$ with the function $(-1)^q$ is within an additive
$\varepsilon$ of the maximum possible correlation with a quadratic phase
function. The algorithm runs in $O_\varepsilon(n^3)$ time and makes
$O_\varepsilon(n^2\log n)$ queries to $f$, which matches the
information-theoretic lower bound of $\Omega(n^2)$ queries up to a logarithmic
factor.
  As a result, we obtain a number of corollaries:
  - A near-optimal self-corrector of quadratic Reed-Muller codes, which makes
$O_\varepsilon(n^2\log n)$ queries to a Boolean function $f$ and returns a
quadratic polynomial $q$ whose relative Hamming distance to $f$ is within
$\varepsilon$ of the minimum distance.
  - An algorithmic polynomial inverse theorem for the order-3 Gowers uniformity
norm.
  - An algorithm that makes a polynomial number of queries to a bounded
function $f$ and decomposes $f$ as a sum of poly$(1/\varepsilon)$ quadratic
phase functions and error terms of order $\varepsilon$.
  Our algorithm is obtained using ideas from recent work on quantum learning
theory. Its construction deviates from previous approaches based on algorithmic
proofs of the inverse theorem for the order-3 uniformity norm (and in
particular does not rely on the recent resolution of the polynomial
Fre\u{\i}man-Ruzsa conjecture).

### 2. [Counting Graphlets of Size $k$ under Local Differential Privacy](http://arxiv.org/pdf/2505.12954v1)

Authors: Vorapong Suppakitpaisarn, Donlapark Ponnoprat, Nicha Hirankarn, Quentin Hillebrand

The problem of counting subgraphs or graphlets under local differential
privacy is an important challenge that has attracted significant attention from
researchers. However, much of the existing work focuses on small graphlets like
triangles or $k$-stars. In this paper, we propose a non-interactive, locally
differentially private algorithm capable of counting graphlets of any size $k$.
When $n$ is the number of nodes in the input graph, we show that the expected
$\ell_2$ error of our algorithm is $O(n^{k - 1})$. Additionally, we prove that
there exists a class of input graphs and graphlets of size $k$ for which any
non-interactive counting algorithm incurs an expected $\ell_2$ error of
$\Omega(n^{k - 1})$, demonstrating the optimality of our result. Furthermore,
we establish that for certain input graphs and graphlets, any locally
differentially private algorithm must have an expected $\ell_2$ error of
$\Omega(n^{k - 1.5})$. Our experimental results show that our algorithm is more
accurate than the classical randomized response method.

### Computational Engineering

### 1. [Seismic analysis based on a new interval method with incomplete information](http://arxiv.org/pdf/2505.12607v1)

Authors: Shizhong Liang, Yuxiang Yang, Chen Li, Feng Wu

For seismic analysis in engineering structures, it is essential to consider
the dynamic responses under seismic excitation, necessitating the description
of seismic accelerations. Limit seismics samples lead to incomplete uncertainty
information, which is described by the non-probabilistic method reasonable.
This study employs the minimum interval radius-based interval process (MRIP)
based on the convex model to describe the time-variant uncertain seismic
acceleration, subsequently conducting uncertainty analysis for seismic
structures. However, the Monte Carlo simulation for uncertainty analysis
requires extensive deterministic computations to ensure accuracy, exhibiting
poor computational efficiency. To address this issue, this paper first improves
the covariance matrix adaptation evolution strategy (CMA-ES) through the
dynamic evolution sequence, proposing DES-ES, whose efficiency is validated to
be higher than that of CMA-ES. Furthermore, leveraging the dependency of the
responses, a computational framework named DES-ES-SS is proposed. Numerical
experiments demonstrate that DES-ES-SS improves computational efficiency while
maintaining the accuracy of the interval uncertainty analysis of the seismic
structures whether the seismic acceleration is stationary or non-stationary.

### 2. [Implicit differentiation with second-order derivatives and benchmarks in finite-element-based differentiable physics](http://arxiv.org/pdf/2505.12646v1)

Authors: Tianju Xue

Differentiable programming is revolutionizing computational science by
enabling automatic differentiation (AD) of numerical simulations. While
first-order gradients are well-established, second-order derivatives (Hessians)
for implicit functions in finite-element-based differentiable physics remain
underexplored. This work bridges this gap by deriving and implementing a
framework for implicit Hessian computation in PDE-constrained optimization
problems. We leverage primitive AD tools (Jacobian-vector
product/vector-Jacobian product) to build an algorithm for Hessian-vector
products and validate the accuracy against finite difference approximations.
Four benchmarks spanning linear/nonlinear, 2D/3D, and single/coupled-variable
problems demonstrate the utility of second-order information. Results show that
the Newton-CG method with exact Hessians accelerates convergence for nonlinear
inverse problems (e.g., traction force identification, shape optimization),
while the L-BFGS-B method suffices for linear cases. Our work provides a robust
foundation for integrating second-order implicit differentiation into
differentiable physics engines, enabling faster and more reliable optimization.

### 3. [Generative Modeling of Random Fields from Limited Data via Constrained Latent Flow Matching](http://arxiv.org/pdf/2505.13007v1)

Authors: James E. Warner, Tristan A. Shah, Patrick E. Leser, Geoffrey F. Bomarito, Joshua D. Pribe, Michael C. Stanley

Deep generative models are promising tools for science and engineering, but
their reliance on abundant, high-quality data limits applicability. We present
a novel framework for generative modeling of random fields (probability
distributions over continuous functions) that incorporates domain knowledge to
supplement limited, sparse, and indirect data. The foundation of the approach
is latent flow matching, where generative modeling occurs on compressed
function representations in the latent space of a pre-trained variational
autoencoder (VAE). Innovations include the adoption of a function decoder
within the VAE and integration of physical/statistical constraints into the VAE
training process. In this way, a latent function representation is learned that
yields continuous random field samples satisfying domain-specific constraints
when decoded, even in data-limited regimes. Efficacy is demonstrated on two
challenging applications: wind velocity field reconstruction from sparse
sensors and material property inference from a limited number of indirect
measurements. Results show that the proposed framework achieves significant
improvements in reconstruction accuracy compared to unconstrained methods and
enables effective inference with relatively small training datasets that is
intractable without constraints.

### 4. [ChromFound: Towards A Universal Foundation Model for Single-Cell Chromatin Accessibility Data](http://arxiv.org/pdf/2505.12638v2)

Authors: Yifeng Jiao, Yuchen Liu, Yu Zhang, Xin Guo, Yushuai Wu, Chen Jiang, Jiyang Li, Hongwei Zhang, Limei Han, Xin Gao, Yuan Qi, Yuan Cheng

The advent of single-cell Assay for Transposase-Accessible Chromatin using
sequencing (scATAC-seq) offers an innovative perspective for deciphering
regulatory mechanisms by assembling a vast repository of single-cell chromatin
accessibility data. While foundation models have achieved significant success
in single-cell transcriptomics, there is currently no foundation model for
scATAC-seq that supports zero-shot high-quality cell identification and
comprehensive multi-omics analysis simultaneously. Key challenges lie in the
high dimensionality and sparsity of scATAC-seq data, as well as the lack of a
standardized schema for representing open chromatin regions (OCRs). Here, we
present ChromFound, a foundation model tailored for scATAC-seq. ChromFound
utilizes a hybrid architecture and genome-aware tokenization to effectively
capture genome-wide long contexts and regulatory signals from dynamic chromatin
landscapes. Pretrained on 1.97 million cells from 30 tissues and 6 disease
conditions, ChromFound demonstrates broad applicability across 6 diverse tasks.
Notably, it achieves robust zero-shot performance in generating universal cell
representations and exhibits excellent transferability in cell type annotation
and cross-omics prediction. By uncovering enhancer-gene links undetected by
existing computational methods, ChromFound offers a promising framework for
understanding disease risk variants in the noncoding genome.

### Computational Geometry

### 1. [Counts and end-curves in two-parameter persistence](http://arxiv.org/pdf/2505.13412v1)

Authors: Thomas Brüstle, Steve Oudot, Luis Scoccola, Hugh Thomas

Given a finite dimensional, bigraded module over the polynomial ring in two
variables, we define its two-parameter count, a natural number, and its
end-curves, a set of plane curves. These are two-dimensional analogues of the
notions of bar-count and endpoints of singly-graded modules over the polynomial
ring in one variable, from persistence theory. We show that our count is the
unique one satisfying certain natural conditions; as a consequence, several
inclusion-exclusion-type formulas in two-parameter persistence yield the same
positive number, which equals our count, and which in turn equals the number of
end-curves, giving geometric meaning to this count. We show that the end-curves
determine the classical Betti tables by showing that they interpolate between
generators, relations, and syzygies. Using the band representations of a
certain string algebra, we show that the set of end-curves admits a canonical
partition, where each part forms a closed curve on the plane; we call this the
boundary of the module. As an invariant, the boundary is neither weaker nor
stronger than the rank invariant, but, in contrast to the rank invariant, it is
a complete invariant on the set of spread-decomposable representations. Our
results connect several lines of work in multiparameter persistence, and their
extension to modules over the real-exponent polynomial ring in two variables
relates to two-dimensional Morse theory.

### 2. [AutoGEEval: A Multimodal and Automated Framework for Geospatial Code Generation on GEE with Large Language Models](http://arxiv.org/pdf/2505.12900v1)

Authors: Shuyang Hou, Zhangxiao Shen, Huayi Wu, Jianyuan Liang, Haoyue Jiao, Yaxian Qing, Xiaopu Zhang, Xu Li, Zhipeng Gui, Xuefeng Guan, Longgang Xiang

Geospatial code generation is emerging as a key direction in the integration
of artificial intelligence and geoscientific analysis. However, there remains a
lack of standardized tools for automatic evaluation in this domain. To address
this gap, we propose AutoGEEval, the first multimodal, unit-level automated
evaluation framework for geospatial code generation tasks on the Google Earth
Engine (GEE) platform powered by large language models (LLMs). Built upon the
GEE Python API, AutoGEEval establishes a benchmark suite (AutoGEEval-Bench)
comprising 1325 test cases that span 26 GEE data types. The framework
integrates both question generation and answer verification components to
enable an end-to-end automated evaluation pipeline-from function invocation to
execution validation. AutoGEEval supports multidimensional quantitative
analysis of model outputs in terms of accuracy, resource consumption, execution
efficiency, and error types. We evaluate 18 state-of-the-art LLMs-including
general-purpose, reasoning-augmented, code-centric, and geoscience-specialized
models-revealing their performance characteristics and potential optimization
pathways in GEE code generation. This work provides a unified protocol and
foundational resource for the development and assessment of geospatial code
generation models, advancing the frontier of automated natural language to
domain-specific code translation.

### Computation and Language

### 1. [Improving Multilingual Language Models by Aligning Representations through Steering](http://arxiv.org/pdf/2505.12584v1)

Authors: Omar Mahmoud, Buddhika Laknath Semage, Thommen George Karimpanal, Santu Rana

In this paper, we investigate how large language models (LLMS) process
non-English tokens within their layer representations, an open question despite
significant advancements in the field. Using representation steering,
specifically by adding a learned vector to a single model layer's activations,
we demonstrate that steering a single model layer can notably enhance
performance. Our analysis shows that this approach achieves results comparable
to translation baselines and surpasses state of the art prompt optimization
methods. Additionally, we highlight how advanced techniques like supervised
fine tuning (\textsc{sft}) and reinforcement learning from human feedback
(\textsc{rlhf}) improve multilingual capabilities by altering representation
spaces. We further illustrate how these methods align with our approach to
reshaping LLMS layer representations.

### 2. [PromptPrism: A Linguistically-Inspired Taxonomy for Prompts](http://arxiv.org/pdf/2505.12592v1)

Authors: Sullam Jeoung, Yueyan Chen, Yi Zhang, Shuai Wang, Haibo Ding, Lin Lee Cheong

Prompts are the interface for eliciting the capabilities of large language
models (LLMs). Understanding their structure and components is critical for
analyzing LLM behavior and optimizing performance. However, the field lacks a
comprehensive framework for systematic prompt analysis and understanding. We
introduce PromptPrism, a linguistically-inspired taxonomy that enables prompt
analysis across three hierarchical levels: functional structure, semantic
component, and syntactic pattern. We show the practical utility of PromptPrism
by applying it to three applications: (1) a taxonomy-guided prompt refinement
approach that automatically improves prompt quality and enhances model
performance across a range of tasks; (2) a multi-dimensional dataset profiling
method that extracts and aggregates structural, semantic, and syntactic
characteristics from prompt datasets, enabling comprehensive analysis of prompt
distributions and patterns; (3) a controlled experimental framework for prompt
sensitivity analysis by quantifying the impact of semantic reordering and
delimiter modifications on LLM performance. Our experimental results validate
the effectiveness of our taxonomy across these applications, demonstrating that
PromptPrism provides a foundation for refining, profiling, and analyzing
prompts.

### 3. [Duluth at SemEval-2025 Task 7: TF-IDF with Optimized Vector Dimensions for Multilingual Fact-Checked Claim Retrieval](http://arxiv.org/pdf/2505.12616v1)

Authors: Shujauddin Syed, Ted Pedersen

This paper presents the Duluth approach to the SemEval-2025 Task 7 on
Multilingual and Crosslingual Fact-Checked Claim Retrieval. We implemented a
TF-IDF-based retrieval system with experimentation on vector dimensions and
tokenization strategies. Our best-performing configuration used word-level
tokenization with a vocabulary size of 15,000 features, achieving an average
success@10 score of 0.78 on the development set and 0.69 on the test set across
ten languages. Our system showed stronger performance on higher-resource
languages but still lagged significantly behind the top-ranked system, which
achieved 0.96 average success@10. Our findings suggest that though advanced
neural architectures are increasingly dominant in multilingual retrieval tasks,
properly optimized traditional methods like TF-IDF remain competitive
baselines, especially in limited compute resource scenarios.

### 4. [Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing](http://arxiv.org/pdf/2505.12636v1)

Authors: Jiakuan Xie, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao

Knowledge editing, which aims to update the knowledge encoded in language
models, can be deceptive. Despite the fact that many existing knowledge editing
algorithms achieve near-perfect performance on conventional metrics, the models
edited by them are still prone to generating original knowledge. This paper
introduces the concept of "superficial editing" to describe this phenomenon.
Our comprehensive evaluation reveals that this issue presents a significant
challenge to existing algorithms. Through systematic investigation, we identify
and validate two key factors contributing to this issue: (1) the residual
stream at the last subject position in earlier layers and (2) specific
attention modules in later layers. Notably, certain attention heads in later
layers, along with specific left singular vectors in their output matrices,
encapsulate the original knowledge and exhibit a causal relationship with
superficial editing. Furthermore, we extend our analysis to the task of
superficial unlearning, where we observe consistent patterns in the behavior of
specific attention heads and their corresponding left singular vectors, thereby
demonstrating the robustness and broader applicability of our methodology and
conclusions. Our code is available here.

### 5. [ToTRL: Unlock LLM Tree-of-Thoughts Reasoning Potential through Puzzles Solving](http://arxiv.org/pdf/2505.12717v1)

Authors: Haoyuan Wu, Xueyi Chen, Rui Ming, Jilong Gao, Shoubo Hu, Zhuolun He, Bei Yu

Large language models (LLMs) demonstrate significant reasoning capabilities,
particularly through long chain-of-thought (CoT) processes, which can be
elicited by reinforcement learning (RL). However, prolonged CoT reasoning
presents limitations, primarily verbose outputs due to excessive introspection.
The reasoning process in these LLMs often appears to follow a trial-and-error
methodology rather than a systematic, logical deduction. In contrast,
tree-of-thoughts (ToT) offers a conceptually more advanced approach by modeling
reasoning as an exploration within a tree structure. This reasoning structure
facilitates the parallel generation and evaluation of multiple reasoning
branches, allowing for the active identification, assessment, and pruning of
unproductive paths. This process can potentially lead to improved performance
and reduced token costs. Building upon the long CoT capability of LLMs, we
introduce tree-of-thoughts RL (ToTRL), a novel on-policy RL framework with a
rule-based reward. ToTRL is designed to guide LLMs in developing the parallel
ToT strategy based on the sequential CoT strategy. Furthermore, we employ LLMs
as players in a puzzle game during the ToTRL training process. Solving puzzle
games inherently necessitates exploring interdependent choices and managing
multiple constraints, which requires the construction and exploration of a
thought tree, providing challenging tasks for cultivating the ToT reasoning
capability. Our empirical evaluations demonstrate that our ToTQwen3-8B model,
trained with our ToTRL, achieves significant improvement in performance and
reasoning efficiency on complex reasoning tasks.

### 6. [On-Policy Optimization with Group Equivalent Preference for Multi-Programming Language Understanding](http://arxiv.org/pdf/2505.12723v1)

Authors: Haoyuan Wu, Rui Ming, Jilong Gao, Hangyu Zhao, Xueyi Chen, Yikai Yang, Haisheng Zheng, Zhuolun He, Bei Yu

Large language models (LLMs) achieve remarkable performance in code
generation tasks. However, a significant performance disparity persists between
popular programming languages (e.g., Python, C++) and others. To address this
capability gap, we leverage the code translation task to train LLMs, thereby
facilitating the transfer of coding proficiency across diverse programming
languages. Moreover, we introduce OORL for training, a novel reinforcement
learning (RL) framework that integrates on-policy and off-policy strategies.
Within OORL, on-policy RL is applied during code translation, guided by a
rule-based reward signal derived from unit tests. Complementing this
coarse-grained rule-based reward, we propose Group Equivalent Preference
Optimization (GEPO), a novel preference optimization method. Specifically, GEPO
trains the LLM using intermediate representations (IRs) groups. LLMs can be
guided to discern IRs equivalent to the source code from inequivalent ones,
while also utilizing signals about the mutual equivalence between IRs within
the group. This process allows LLMs to capture nuanced aspects of code
functionality. By employing OORL for training with code translation tasks, LLMs
improve their recognition of code functionality and their understanding of the
relationships between code implemented in different languages. Extensive
experiments demonstrate that our OORL for LLMs training with code translation
tasks achieves significant performance improvements on code benchmarks across
multiple programming languages.

### 7. [ReEx-SQL: Reasoning with Execution-Aware Reinforcement Learning for Text-to-SQL](http://arxiv.org/pdf/2505.12768v2)

Authors: Yaxun Dai, Wenxuan Xie, Xialie Zhuang, Tianyu Yang, Yiying Yang, Haiqin Yang, Yuhang Zhao, Pingfu Chao, Wenhao Jiang

In Text-to-SQL, execution feedback is essential for guiding large language
models (LLMs) to reason accurately and generate reliable SQL queries. However,
existing methods treat execution feedback solely as a post-hoc signal for
correction or selection, failing to integrate it into the generation process.
This limitation hinders their ability to address reasoning errors as they
occur, ultimately reducing query accuracy and robustness. To address this
issue, we propose ReEx-SQL (Reasoning with Execution-Aware Reinforcement
Learning), a framework for Text-to-SQL that enables models to interact with the
database during decoding and dynamically adjust their reasoning based on
execution feedback. ReEx-SQL introduces an execution-aware reasoning paradigm
that interleaves intermediate SQL execution into reasoning paths, facilitating
context-sensitive revisions. It achieves this through structured prompts with
markup tags and a stepwise rollout strategy that integrates execution feedback
into each stage of generation. To supervise policy learning, we develop a
composite reward function that includes an exploration reward, explicitly
encouraging effective database interaction. Additionally, ReEx-SQL adopts a
tree-based decoding strategy to support exploratory reasoning, enabling dynamic
expansion of alternative reasoning paths. Notably, ReEx-SQL achieves 88.8% on
Spider and 64.9% on BIRD at the 7B scale, surpassing the standard reasoning
baseline by 2.7% and 2.6%, respectively. It also shows robustness, achieving
85.2% on Spider-Realistic with leading performance. In addition, its
tree-structured decoding improves efficiency and performance over linear
decoding, reducing inference time by 51.9% on the BIRD development set.

### 8. [EAVIT: Efficient and Accurate Human Value Identification from Text data via LLMs](http://arxiv.org/pdf/2505.12792v1)

Authors: Wenhao Zhu, Yuhang Xie, Guojie Song, Xin Zhang

The rapid evolution of large language models (LLMs) has revolutionized
various fields, including the identification and discovery of human values
within text data. While traditional NLP models, such as BERT, have been
employed for this task, their ability to represent textual data is
significantly outperformed by emerging LLMs like GPTs. However, the performance
of online LLMs often degrades when handling long contexts required for value
identification, which also incurs substantial computational costs. To address
these challenges, we propose EAVIT, an efficient and accurate framework for
human value identification that combines the strengths of both locally
fine-tunable and online black-box LLMs. Our framework employs a value detector
- a small, local language model - to generate initial value estimations. These
estimations are then used to construct concise input prompts for online LLMs,
enabling accurate final value identification. To train the value detector, we
introduce explanation-based training and data generation techniques
specifically tailored for value identification, alongside sampling strategies
to optimize the brevity of LLM input prompts. Our approach effectively reduces
the number of input tokens by up to 1/6 compared to directly querying online
LLMs, while consistently outperforming traditional NLP methods and other
LLM-based strategies.

### 9. [Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](http://arxiv.org/pdf/2505.12831v1)

Authors: Zifeng Cheng, Zhonghui Wang, Yuchen Fu, Zhiwei Jiang, Yafeng Yin, Cong Wang, Qing Gu

Extracting sentence embeddings from large language models (LLMs) is a
practical direction, as it requires neither additional data nor fine-tuning.
Previous studies usually focus on prompt engineering to guide LLMs to encode
the core semantic information of the sentence into the embedding of the last
token. However, the last token in these methods still encodes an excess of
non-essential information, such as stop words, limiting its encoding capacity.
To this end, we propose a Contrastive Prompting (CP) method that introduces an
extra auxiliary prompt to elicit better sentence embedding. By contrasting with
the auxiliary prompt, CP can steer existing prompts to encode the core
semantics of the sentence, rather than non-essential information. CP is a
plug-and-play inference-time intervention method that can be combined with
various prompt-based methods. Extensive experiments on Semantic Textual
Similarity (STS) tasks and downstream classification tasks demonstrate that our
method can improve the performance of existing prompt-based methods across
different LLMs. Our code will be released at https://github.com/zifengcheng/CP.

### 10. [Re-identification of De-identified Documents with Autoregressive Infilling](http://arxiv.org/pdf/2505.12859v1)

Authors: Lucas Georges Gabriel Charpentier, Pierre Lison

Documents revealing sensitive information about individuals must typically be
de-identified. This de-identification is often done by masking all mentions of
personally identifiable information (PII), thereby making it more difficult to
uncover the identity of the person(s) in question. To investigate the
robustness of de-identification methods, we present a novel, RAG-inspired
approach that attempts the reverse process of re-identification based on a
database of documents representing background knowledge. Given a text in which
personal identifiers have been masked, the re-identification proceeds in two
steps. A retriever first selects from the background knowledge passages deemed
relevant for the re-identification. Those passages are then provided to an
infilling model which seeks to infer the original content of each text span.
This process is repeated until all masked spans are replaced. We evaluate the
re-identification on three datasets (Wikipedia biographies, court rulings and
clinical notes). Results show that (1) as many as 80% of de-identified text
spans can be successfully recovered and (2) the re-identification accuracy
increases along with the level of background knowledge.

### Cryptography and Security

### 1. [Compile-Time Fully Homomorphic Encryption: Eliminating Online Encryption via Algebraic Basis Synthesis](http://arxiv.org/pdf/2505.12582v1)

Authors: Dongfang Zhao

We propose a new framework for compile-time ciphertext synthesis in fully
homomorphic encryption (FHE) systems. Instead of invoking encryption algorithms
at runtime, our method synthesizes ciphertexts from precomputed encrypted basis
vectors using only homomorphic additions, scalar multiplications, and
randomized encryptions of zero. This decouples ciphertext generation from
encryption, and enables efficient batch encoding through algebraic reuse. We
formalize this technique as a randomized module morphism and prove that it
satisfies IND-CPA security. Our proof uses a hybrid game framework that
interpolates between encrypted vector instances and reduces adversarial
advantage to the indistinguishability of the underlying FHE scheme. This
reduction structure captures the security implications of ciphertext basis
reuse and structured noise injection. The proposed synthesis primitive supports
fast, encryption-free ingestion in outsourced database systems and other
high-throughput FHE pipelines. It is compatible with standard FHE APIs and
preserves layout semantics for downstream homomorphic operations.

### 2. [hChain: Blockchain Based Large Scale EHR Data Sharing with Enhanced Security and Privacy](http://arxiv.org/pdf/2505.12610v1)

Authors: Musharraf Alruwaill, Saraju Mohanty, Elias Kougianos

Concerns regarding privacy and data security in conventional healthcare
prompted alternative technologies. In smart healthcare, blockchain technology
addresses existing concerns with security, privacy, and electronic healthcare
transmission. Integration of Blockchain Technology with the Internet of Medical
Things (IoMT) allows real-time monitoring of protected healthcare data.
Utilizing edge devices with IoMT devices is very advantageous for addressing
security, computing, and storage challenges. Encryption using symmetric and
asymmetric keys is used to conceal sensitive information from unauthorized
parties. SHA256 is an algorithm for one-way hashing. It is used to verify that
the data has not been altered, since if it had, the hash value would have
changed. This article offers a blockchain-based smart healthcare system using
IoMT devices for continuous patient monitoring. In addition, it employs edge
resources in addition to IoMT devices to have extra computing power and storage
to hash and encrypt incoming data before sending it to the blockchain.
Symmetric key is utilized to keep the data private even in the blockchain,
allowing the patient to safely communicate the data through smart contracts
while preventing unauthorized physicians from seeing the data. Through the use
of a verification node and blockchain, an asymmetric key is used for the
signing and validation of patient data in the healthcare provider system. In
addition to other security measures, location-based authentication is
recommended to guarantee that data originates from the patient area. Through
the edge device, SHA256 is utilized to secure the data's integrity and a secret
key is used to maintain its secrecy. The hChain architecture improves the
computing power of IoMT environments, the security of EHR sharing through smart
contracts, and the privacy and authentication procedures.

### 3. [EPSpatial: Achieving Efficient and Private Statistical Analytics of Geospatial Data](http://arxiv.org/pdf/2505.12612v1)

Authors: Chuan Zhang, Xuhao Ren, Zhangcheng Huang, Jinwen Liang, Jianzong Wang, Liehuang Zhu

Geospatial data statistics involve the aggregation and analysis of location
data to derive the distribution of clients within geospatial. The need for
privacy protection in geospatial data analysis has become paramount due to
concerns over the misuse or unauthorized access of client location information.
However, existing private geospatial data statistics mainly rely on privacy
computing techniques such as cryptographic tools and differential privacy,
which leads to significant overhead and inaccurate results. In practical
applications, geospatial data is frequently generated by mobile devices such as
smartphones and IoT sensors. The continuous mobility of clients and the need
for real-time updates introduce additional complexity. To address these issues,
we first design \textit{spatially distributed point functions (SDPF)}, which
combines a quad-tree structure with distributed point functions, allowing
clients to succinctly secret-share values on the nodes of an exponentially
large quad-tree. Then, we use Gray code to partition the region and combine
SDPF with it to propose $\mathtt{EPSpatial}$, a scheme for accurate, efficient,
and private statistical analytics of geospatial data. Moreover, considering
clients' frequent movement requires continuous location updates, we leverage
the region encoding property to present an efficient update algorithm.Security
analysis shows that $\mathtt{EPSpatial}$ effectively protects client location
privacy. Theoretical analysis and experimental results on real datasets
demonstrate that $\mathtt{EPSpatial}$ reduces computational and communication
overhead by at least $50\%$ compared to existing statistical schemes.

### 4. [Towards Centralized Orchestration of Cyber Protection Condition (CPCON)](http://arxiv.org/pdf/2505.12613v1)

Authors: Mark Timmons, Daniel Lukaszewski, Geoffrey Xie, Thomas Mayo, Donald McCanless

The United States Cyber Command (USCYBERCOM) Cyber Protection Condition
(CPCON) framework mandates graduated security postures across Department of
Defense (DoD) networks, but current implementation remains largely manual,
inconsistent, and error-prone. This paper presents a prototype system for
centralized orchestration of CPCON directives, enabling automated policy
enforcement and real-time threat response across heterogeneous network
environments. Building on prior work in host-based intrusion response, our
system leverages a policy-driven orchestrator to standardize security actions,
isolate compromised subnets, and verify enforcement status. We validate the
system through emulated attack scenarios, demonstrating improved speed,
accuracy, and verifiability in CPCON transitions with human-in-the-loop
oversight.

### 5. [GDPRShield: AI-Powered GDPR Support for Software Developers in Small and Medium-Sized Enterprises](http://arxiv.org/pdf/2505.12640v1)

Authors: Tharaka Wijesundara, Mathew Warren, Nalin Arachchilage

With the rapid increase in privacy violations in modern software development,
regulatory frameworks such as the General Data Protection Regulation (GDPR)
have been established to enforce strict data protection practices. However,
insufficient privacy awareness among SME software developers contributes to
failure in GDPR compliance. For instance, a developer unfamiliar with data
minimization may build a system that collects excessive data, violating GDPR
and risking fines. One reason for this lack of awareness is that developers in
SMEs often take on multidisciplinary roles (e.g., front-end, back-end, database
management, and privacy compliance), which limits specialization in privacy.
This lack of awareness may lead to poor privacy attitudes, ultimately hindering
the development of a strong organizational privacy culture. However, SMEs that
achieve GDPR compliance may gain competitive advantages, such as increased user
trust and marketing value, compared to others that do not.
  Therefore, in this paper, we introduce a novel AI-powered framework called
"GDPRShield," specifically designed to enhance the GDPR awareness of SME
software developers and, through this, improve their privacy attitudes.
Simultaneously, GDPRShield boosts developers' motivation to comply with GDPR
from the early stages of software development. It leverages functional
requirements written as user stories to provide comprehensive GDPR-based
privacy descriptions tailored to each requirement. Alongside improving
awareness, GDPRShield strengthens motivation by presenting real-world
consequences of noncompliance, such as heavy fines, reputational damage, and
loss of user trust, aligned with each requirement. This dual focus on awareness
and motivation leads developers to engage with GDPRShield, improving their GDPR
compliance and privacy attitudes, which will help SMEs build a stronger privacy
culture over time.

### 6. [Shielding Latent Face Representations From Privacy Attacks](http://arxiv.org/pdf/2505.12688v1)

Authors: Arjun Ramesh Kaushik, Bharat Chandra Yalavarthi, Arun Ross, Vishnu Boddeti, Nalini Ratha

In today's data-driven analytics landscape, deep learning has become a
powerful tool, with latent representations, known as embeddings, playing a
central role in several applications. In the face analytics domain, such
embeddings are commonly used for biometric recognition (e.g., face
identification). However, these embeddings, or templates, can inadvertently
expose sensitive attributes such as age, gender, and ethnicity. Leaking such
information can compromise personal privacy and affect civil liberty and human
rights. To address these concerns, we introduce a multi-layer protection
framework for embeddings. It consists of a sequence of operations: (a)
encrypting embeddings using Fully Homomorphic Encryption (FHE), and (b) hashing
them using irreversible feature manifold hashing. Unlike conventional
encryption methods, FHE enables computations directly on encrypted data,
allowing downstream analytics while maintaining strong privacy guarantees. To
reduce the overhead of encrypted processing, we employ embedding compression.
Our proposed method shields latent representations of sensitive data from
leaking private attributes (such as age and gender) while retaining essential
functional capabilities (such as face identification). Extensive experiments on
two datasets using two face encoders demonstrate that our approach outperforms
several state-of-the-art privacy protection methods.

### 7. [Writing a Good Security Paper for ISSCC (2025)](http://arxiv.org/pdf/2505.12700v1)

Authors: Utsav Banerjee, Chiraag Juvekar, Yong Ki Lee, Leibo Liu, Sanu Mathew, Thomas Poeppelmann, Shreyas Sen, Takeshi Sugawara, Ingrid Verbauwhede, Rabia Tugce Yazicigil

Security is increasingly more important in designing chips and systems based
on them, and the International Solid-State Circuits Conference (ISSCC), the
leading conference for presenting advances in solid-state circuits and
semiconductor technology, is committed to hardware security by establishing the
security subcommittee since 2024. In the past two years, the authors of this
paper reviewed submissions as members of the Security Subcommittee, a part of
International Technical Program Committee (ITPC). This paper aims to encourage
high-quality submissions to grow this field in the overall scope of the ISSCC.

### 8. [Lara: Lightweight Anonymous Authentication with Asynchronous Revocation Auditability](http://arxiv.org/pdf/2505.12968v1)

Authors: Claudio Correia, Guilherme Santos, Luis Rodrigues

Anonymous authentication is a technique that allows to combine access control
with privacy preservation. Typically, clients use different pseudonyms for each
access, hindering providers from correlating their activities. To perform the
revocation of pseudonyms in a privacy preserving manner is notoriously
challenging. When multiple pseudonyms are revoked together, an adversary may
infer that these pseudonyms belong to the same client and perform privacy
breaking correlations, in particular if these pseudonyms have already been
used. Backward unlinkability and revocation auditability are two properties
that address this problem. Most systems that offer these properties rely on
some sort of time slots, which assume a common reference of time that must be
shared among clients and providers; for instance, the client must be aware that
it should not use a pseudonym after a certain time or should be able to assess
the freshness of a revocation list prior to perform authentication. In this
paper we propose Lara, a Lightweight Anonymous Authentication with Asynchronous
Revocation Auditability that does not require parties to agree on the current
time slot and it is not affected by the clock skew. Prior to disclosing a
pseudonym, clients are provided with a revocation list (RL) and can check that
the pseudonym has not been revoked. Then, they provide a proof on
non-revocation that cannot be used against any other (past or future) RL,
avoiding any dependency of timing assumptions. Lara can be implemented using
efficient public-key primitives and space-efficient data structures. We have
implemented a prototype of Lara and have assessed experimentally its
efficiency.

### 9. [ACE: Confidential Computing for Embedded RISC-V Systems](http://arxiv.org/pdf/2505.12995v1)

Authors: Wojciech Ozga, Guerney D. H. Hunt, Michael V. Le, Lennard Gäher, Avraham Shinnar, Elaine R. Palmer, Hani Jamjoom, Silvio Dragone

Confidential computing plays an important role in isolating sensitive
applications from the vast amount of untrusted code commonly found in the
modern cloud. We argue that it can also be leveraged to build safer and more
secure mission-critical embedded systems. In this paper, we introduce the
Assured Confidential Execution (ACE), an open-source and royalty-free
confidential computing technology targeted for embedded RISC-V systems. We
present a set of principles and a methodology that we used to build \ACE and
that might be applied for developing other embedded systems that require formal
verification. An evaluation of our prototype on the first available RISC-V
hardware supporting virtualization indicates that ACE is a viable candidate for
our target systems.

### 10. [Network-wide Quantum Key Distribution with Onion Routing Relay (Conference Version)](http://arxiv.org/pdf/2505.13158v1)

Authors: Pedro Otero-García, David Pérez-Castro, Manuel Fernández-Veiga, Ana Fernández-Vilas

The advancement of quantum computing threatens classical cryptographic
methods, necessitating the development of secure quantum key distribution (QKD)
solutions for QKD Networks (QKDN). In this paper, a novel key distribution
protocol, Onion Routing Relay (ORR), that integrates onion routing (OR) with
post-quantum cryptography (PQC) in a key-relay (KR) model is evaluated for
QKDNs. This approach increases the security by enhancing confidentiality,
integrity, authenticity (CIA principles), and anonymity in quantum-secure
communications. By employing PQC-based encapsulation, ORR aims to avoid the
security risks posed by intermediate malicious nodes and ensures end-to-end
security. Our results show a competitive performance of the basic ORR model,
against current KR and trusted-node (TN) approaches, demonstrating its
feasibility and applicability in high-security environments maintaining a
consistent Quality of Service (QoS). The results also show that while basic ORR
incurs higher encryption overhead, it provides substantial security
improvements without significantly impacting the overall key distribution time.
Nevertheless, the introduction of an end-to-end authentication extension
(ORR-Ext) has a significant impact on the Quality of Service (QoS), thereby
limiting its suitability to applications with stringent security requirements.

### Computer Vision and Pattern Recognition

### 1. [Coarse Attribute Prediction with Task Agnostic Distillation for Real World Clothes Changing ReID](http://arxiv.org/pdf/2505.12580v1)

Authors: Priyank Pathak, Yogesh S Rawat

This work focuses on Clothes Changing Re-IDentification (CC-ReID) for the
real world. Existing works perform well with high-quality (HQ) images, but
struggle with low-quality (LQ) where we can have artifacts like pixelation,
out-of-focus blur, and motion blur. These artifacts introduce noise to not only
external biometric attributes (e.g. pose, body shape, etc.) but also corrupt
the model's internal feature representation. Models usually cluster LQ image
features together, making it difficult to distinguish between them, leading to
incorrect matches. We propose a novel framework Robustness against Low-Quality
(RLQ) to improve CC-ReID model on real-world data. RLQ relies on Coarse
Attributes Prediction (CAP) and Task Agnostic Distillation (TAD) operating in
alternate steps in a novel training mechanism. CAP enriches the model with
external fine-grained attributes via coarse predictions, thereby reducing the
effect of noisy inputs. On the other hand, TAD enhances the model's internal
feature representation by bridging the gap between HQ and LQ features, via an
external dataset through task-agnostic self-supervision and distillation. RLQ
outperforms the existing approaches by 1.6%-2.9% Top-1 on real-world datasets
like LaST, and DeepChange, while showing consistent improvement of 5.3%-6%
Top-1 on PRCC with competitive performance on LTCC. *The code will be made
public soon.*

### 2. [SurveillanceVQA-589K: A Benchmark for Comprehensive Surveillance Video-Language Understanding with Large Models](http://arxiv.org/pdf/2505.12589v1)

Authors: Bo Liu, Pengfei Qiao, Minhan Ma, Xuange Zhang, Yinan Tang, Peng Xu, Kun Liu, Tongtong Yuan

Understanding surveillance video content remains a critical yet underexplored
challenge in vision-language research, particularly due to its real-world
complexity, irregular event dynamics, and safety-critical implications. In this
work, we introduce SurveillanceVQA-589K, the largest open-ended video question
answering benchmark tailored to the surveillance domain. The dataset comprises
589,380 QA pairs spanning 12 cognitively diverse question types, including
temporal reasoning, causal inference, spatial understanding, and anomaly
interpretation, across both normal and abnormal video scenarios. To construct
the benchmark at scale, we design a hybrid annotation pipeline that combines
temporally aligned human-written captions with Large Vision-Language
Model-assisted QA generation using prompt-based techniques. We also propose a
multi-dimensional evaluation protocol to assess contextual, temporal, and
causal comprehension. We evaluate eight LVLMs under this framework, revealing
significant performance gaps, especially in causal and anomaly-related tasks,
underscoring the limitations of current models in real-world surveillance
contexts. Our benchmark provides a practical and comprehensive resource for
advancing video-language understanding in safety-critical applications such as
intelligent monitoring, incident analysis, and autonomous decision-making.

### 3. [Learning Cross-Spectral Point Features with Task-Oriented Training](http://arxiv.org/pdf/2505.12593v1)

Authors: Mia Thomas, Trevor Ablett, Jonathan Kelly

Unmanned aerial vehicles (UAVs) enable operations in remote and hazardous
environments, yet the visible-spectrum, camera-based navigation systems often
relied upon by UAVs struggle in low-visibility conditions. Thermal cameras,
which capture long-wave infrared radiation, are able to function effectively in
darkness and smoke, where visible-light cameras fail. This work explores
learned cross-spectral (thermal-visible) point features as a means to integrate
thermal imagery into established camera-based navigation systems. Existing
methods typically train a feature network's detection and description outputs
directly, which often focuses training on image regions where thermal and
visible-spectrum images exhibit similar appearance. Aiming to more fully
utilize the available data, we propose a method to train the feature network on
the tasks of matching and registration. We run our feature network on
thermal-visible image pairs, then feed the network response into a
differentiable registration pipeline. Losses are applied to the matching and
registration estimates of this pipeline. Our selected model, trained on the
task of matching, achieves a registration error (corner error) below 10 pixels
for more than 75% of estimates on the MultiPoint dataset. We further
demonstrate that our model can also be used with a classical pipeline for
matching and registration.

### 4. [Temporal-Oriented Recipe for Transferring Large Vision-Language Model to Video Understanding](http://arxiv.org/pdf/2505.12605v1)

Authors: Thong Nguyen, Zhiyuan Hu, Xu Lin, Cong-Duy Nguyen, See-Kiong Ng, Luu Anh Tuan

Recent years have witnessed outstanding advances of large vision-language
models (LVLMs). In order to tackle video understanding, most of them depend
upon their implicit temporal understanding capacity. As such, they have not
deciphered important components that contribute to temporal understanding
ability, which might limit the potential of these LVLMs for video
understanding. In this work, we conduct a thorough empirical study to demystify
crucial components that influence the temporal understanding of LVLMs. Our
empirical study reveals that significant impacts are centered around the
intermediate interface between the visual encoder and the large language model.
Building on these insights, we propose a temporal-oriented recipe that
encompasses temporal-oriented training schemes and an upscaled interface. Our
final model developed using our recipe significantly enhances previous LVLMs on
standard video understanding tasks.

### 5. [Diff-MM: Exploring Pre-trained Text-to-Image Generation Model for Unified Multi-modal Object Tracking](http://arxiv.org/pdf/2505.12606v1)

Authors: Shiyu Xuan, Zechao Li, Jinhui Tang

Multi-modal object tracking integrates auxiliary modalities such as depth,
thermal infrared, event flow, and language to provide additional information
beyond RGB images, showing great potential in improving tracking stabilization
in complex scenarios. Existing methods typically start from an RGB-based
tracker and learn to understand auxiliary modalities only from training data.
Constrained by the limited multi-modal training data, the performance of these
methods is unsatisfactory. To alleviate this limitation, this work proposes a
unified multi-modal tracker Diff-MM by exploiting the multi-modal understanding
capability of the pre-trained text-to-image generation model. Diff-MM leverages
the UNet of pre-trained Stable Diffusion as a tracking feature extractor
through the proposed parallel feature extraction pipeline, which enables
pairwise image inputs for object tracking. We further introduce a multi-modal
sub-module tuning method that learns to gain complementary information between
different modalities. By harnessing the extensive prior knowledge in the
generation model, we achieve a unified tracker with uniform parameters for
RGB-N/D/T/E tracking. Experimental results demonstrate the promising
performance of our method compared with recently proposed trackers, e.g., its
AUC outperforms OneTracker by 8.3% on TNL2K.

### 6. [BusterX: MLLM-Powered AI-Generated Video Forgery Detection and Explanation](http://arxiv.org/pdf/2505.12620v1)

Authors: Haiquan Wen, Yiwei He, Zhenglin Huang, Tianxiao Li, Zihan YU, Xingru Huang, Lu Qi, Baoyuan Wu, Xiangtai Li, Guangliang Cheng

Advances in AI generative models facilitate super-realistic video synthesis,
amplifying misinformation risks via social media and eroding trust in digital
content. Several research works have explored new deepfake detection methods on
AI-generated images to alleviate these risks. However, with the fast
development of video generation models, such as Sora and WanX, there is
currently a lack of large-scale, high-quality AI-generated video datasets for
forgery detection. In addition, existing detection approaches predominantly
treat the task as binary classification, lacking explainability in model
decision-making and failing to provide actionable insights or guidance for the
public. To address these challenges, we propose \textbf{GenBuster-200K}, a
large-scale AI-generated video dataset featuring 200K high-resolution video
clips, diverse latest generative techniques, and real-world scenes. We further
introduce \textbf{BusterX}, a novel AI-generated video detection and
explanation framework leveraging multimodal large language model (MLLM) and
reinforcement learning for authenticity determination and explainable
rationale. To our knowledge, GenBuster-200K is the {\it \textbf{first}}
large-scale, high-quality AI-generated video dataset that incorporates the
latest generative techniques for real-world scenarios. BusterX is the {\it
\textbf{first}} framework to integrate MLLM with reinforcement learning for
explainable AI-generated video detection. Extensive comparisons with
state-of-the-art methods and ablation studies validate the effectiveness and
generalizability of BusterX. The code, models, and datasets will be released.

### 7. [Multi-Resolution Haar Network: Enhancing human motion prediction via Haar transform](http://arxiv.org/pdf/2505.12631v1)

Authors: Li Lin

The 3D human pose is vital for modern computer vision and computer graphics,
and its prediction has drawn attention in recent years. 3D human pose
prediction aims at forecasting a human's future motion from the previous
sequence. Ignoring that the arbitrariness of human motion sequences has a firm
origin in transition in both temporal and spatial axes limits the performance
of state-of-the-art methods, leading them to struggle with making precise
predictions on complex cases, e.g., arbitrarily posing or greeting. To
alleviate this problem, a network called HaarMoDic is proposed in this paper,
which utilizes the 2D Haar transform to project joints to higher resolution
coordinates where the network can access spatial and temporal information
simultaneously. An ablation study proves that the significant contributing
module within the HaarModic Network is the Multi-Resolution Haar (MR-Haar)
block. Instead of mining in one of two axes or extracting separately, the
MR-Haar block projects whole motion sequences to a mixed-up coordinate in
higher resolution with 2D Haar Transform, allowing the network to give scope to
information from both axes in different resolutions. With the MR-Haar block,
the HaarMoDic network can make predictions referring to a broader range of
information. Experimental results demonstrate that HaarMoDic surpasses
state-of-the-art methods in every testing interval on the Human3.6M dataset in
the Mean Per Joint Position Error (MPJPE) metric.

### 8. [MVPainter: Accurate and Detailed 3D Texture Generation via Multi-View Diffusion with Geometric Control](http://arxiv.org/pdf/2505.12635v1)

Authors: Mingqi Shao, Feng Xiong, Zhaoxu Sun, Mu Xu

Recently, significant advances have been made in 3D object generation.
Building upon the generated geometry, current pipelines typically employ image
diffusion models to generate multi-view RGB images, followed by UV texture
reconstruction through texture baking. While 3D geometry generation has
improved significantly, supported by multiple open-source frameworks, 3D
texture generation remains underexplored. In this work, we systematically
investigate 3D texture generation through the lens of three core dimensions:
reference-texture alignment, geometry-texture consistency, and local texture
quality. To tackle these issues, we propose MVPainter, which employs data
filtering and augmentation strategies to enhance texture fidelity and detail,
and introduces ControlNet-based geometric conditioning to improve
texture-geometry alignment. Furthermore, we extract physically-based rendering
(PBR) attributes from the generated views to produce PBR meshes suitable for
real-world rendering applications. MVPainter achieves state-of-the-art results
across all three dimensions, as demonstrated by human-aligned evaluations. To
facilitate further research and reproducibility, we also release our full
pipeline as an open-source system, including data construction, model
architecture, and evaluation tools.

### 9. [Use as Many Surrogates as You Want: Selective Ensemble Attack to Unleash Transferability without Sacrificing Resource Efficiency](http://arxiv.org/pdf/2505.12644v1)

Authors: Bo Yang, Hengwei Zhang, Jindong Wang, Yuchen Ren, Chenhao Lin, Chao Shen, Zhengyu Zhao

In surrogate ensemble attacks, using more surrogate models yields higher
transferability but lower resource efficiency. This practical trade-off between
transferability and efficiency has largely limited existing attacks despite
many pre-trained models are easily accessible online. In this paper, we argue
that such a trade-off is caused by an unnecessary common assumption, i.e., all
models should be identical across iterations. By lifting this assumption, we
can use as many surrogates as we want to unleash transferability without
sacrificing efficiency. Concretely, we propose Selective Ensemble Attack (SEA),
which dynamically selects diverse models (from easily accessible pre-trained
models) across iterations based on our new interpretation of decoupling
within-iteration and cross-iteration model diversity.In this way, the number of
within-iteration models is fixed for maintaining efficiency, while only
cross-iteration model diversity is increased for higher transferability.
Experiments on ImageNet demonstrate the superiority of SEA in various
scenarios. For example, when dynamically selecting 4 from 20 accessible models,
SEA yields 8.5% higher transferability than existing attacks under the same
efficiency. The superiority of SEA also generalizes to real-world systems, such
as commercial vision APIs and large vision-language models. Overall, SEA opens
up the possibility of adaptively balancing transferability and efficiency
according to specific resource requirements.

### 10. [SPKLIP: Aligning Spike Video Streams with Natural Language](http://arxiv.org/pdf/2505.12656v1)

Authors: Yongchang Gao, Meiling Jin, Zhaofei Yu, Tiejun Huang, Guozhang Chen

Spike cameras offer unique sensing capabilities but their sparse,
asynchronous output challenges semantic understanding, especially for Spike
Video-Language Alignment (Spike-VLA) where models like CLIP underperform due to
modality mismatch. We introduce SPKLIP, the first architecture specifically for
Spike-VLA. SPKLIP employs a hierarchical spike feature extractor that
adaptively models multi-scale temporal dynamics in event streams, and uses
spike-text contrastive learning to directly align spike video with language,
enabling effective few-shot learning. A full-spiking visual encoder variant,
integrating SNN components into our pipeline, demonstrates enhanced energy
efficiency. Experiments show state-of-the-art performance on benchmark spike
datasets and strong few-shot generalization on a newly contributed real-world
dataset. SPKLIP's energy efficiency highlights its potential for neuromorphic
deployment, advancing event-based multimodal research. The source code and
dataset are available at [link removed for anonymity].

### Computers and Society

### 1. ["I will never pay for this" Perception of fairness and factors affecting behaviour on 'pay-or-ok' models](http://arxiv.org/pdf/2505.12892v1)

Authors: Victor Morel, Farzaneh Karegar, Cristiana Santos

The rise of cookie paywalls ('pay-or-ok' models) has prompted growing debates
around privacy, monetisation, and the legitimacy of user consent. Despite their
increasing use across sectors, limited research has explored how users perceive
these models or what shapes their decisions to either consent to tracking or
pay. To address this gap, we conducted four focus groups (n = 14) to examine
users' perceptions of cookie paywalls, their judgments of fairness, and the
conditions under which they might consider paying, alongside a legal analysis
within the EU data protection framework law.
  Participants primarily viewed cookie paywalls as profit-driven, with fairness
perceptions varying depending on factors such as the presence of a third option
beyond consent or payment, transparency of data practices, and the authenticity
or exclusivity of the paid content. Participants voiced expectations for
greater transparency, meaningful control over data collection, and less
coercive alternatives, such as contextual advertising or "reject all" buttons.
Although some conditions, including trusted providers, exclusive content, and
reasonable pricing, could make participants consider paying, most expressed
reluctance or unwillingness to do so.
  Crucially, our findings raise concerns about economic exclusion, where
privacy and data protection might end up becoming a privilege rather than
fundamental rights. Consent given under financial pressure may not meet the
standard of being freely given, as required by GDPR. To address these concerns,
we recommend user-centred approaches that enhance transparency, reduce
coercion, ensure the value of paid content, and explore inclusive alternatives.
These measures are essential for supporting fairness, meaningful choice, and
user autonomy in consent-driven digital environments.

### 2. [Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](http://arxiv.org/pdf/2505.13143v1)

Authors: Haolang Lu, Yilian Liu, Jingxin Xu, Guoshun Nan, Yuanlong Yu, Zhican Chen, Kun Wang

The development of Reasoning Large Language Models (RLLMs) has significantly
improved multi-step reasoning capabilities, but it has also made hallucination
problems more frequent and harder to eliminate. While existing approaches
mitigate hallucinations through external knowledge integration, model parameter
analysis, or self-verification, they often fail to capture how hallucinations
emerge and evolve across the reasoning chain. In this work, we study the
causality of hallucinations under constrained knowledge domains by auditing the
Chain-of-Thought (CoT) trajectory and assessing the model's cognitive
confidence in potentially erroneous or biased claims. Our analysis reveals that
in long-CoT settings, RLLMs can iteratively reinforce biases and errors through
flawed reflective reasoning, eventually leading to hallucinated reasoning
paths. Surprisingly, even direct interventions at the origin of hallucinations
often fail to reverse their effects, as reasoning chains exhibit 'chain
disloyalty' -- a resistance to correction and a tendency to preserve flawed
logic. Furthermore, we show that existing hallucination detection methods are
less reliable and interpretable than previously assumed in complex reasoning
scenarios. Unlike methods such as circuit tracing that require access to model
internals, our black-box auditing approach supports interpretable long-chain
hallucination attribution, offering better generalizability and practical
utility. Code and data are available at:
https://anonymous.4open.science/r/repo_for_meta_hallucination

### 3. [Starting Seatwork Earlier as a Valid Measure of Student Engagement](http://arxiv.org/pdf/2505.13341v1)

Authors: Ashish Gurung, Jionghao Lin, Zhongtian Huang, Conrad Borchers, Ryan S. Baker, Vincent Aleven, Kenneth R. Koedinger

Prior work has developed a range of automated measures ("detectors") of
student self-regulation and engagement from student log data. These measures
have been successfully used to make discoveries about student learning. Here,
we extend this line of research to an underexplored aspect of self-regulation:
students' decisions about when to start and stop working on learning software
during classwork. In the first of two analyses, we build on prior work on
session-level measures (e.g., delayed start, early stop) to evaluate their
reliability and predictive validity. We compute these measures from year-long
log data from Cognitive Tutor for students in grades 8-12 (N = 222). Our
findings show that these measures exhibit moderate to high month-to-month
reliability (G > .75), comparable to or exceeding gaming-the-system behavior.
Additionally, they enhance the prediction of final math scores beyond prior
knowledge and gaming-the-system behaviors. The improvement in learning outcome
predictions beyond time-on-task suggests they capture a broader motivational
state tied to overall learning. The second analysis demonstrates the
cross-system generalizability of these measures in i-Ready, where they predict
state test scores for grade 7 students (N = 818). By leveraging log data, we
introduce system-general naturally embedded measures that complement
motivational surveys without extra instrumentation or disruption of instruction
time. Our findings demonstrate the potential of session-level logs to mine
valid and generalizable measures with broad applications in the predictive
modeling of learning outcomes and analysis of learner self-regulation.

### 4. [What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma](http://arxiv.org/pdf/2505.12727v1)

Authors: Han Meng, Yancan Chen, Yunan Li, Yitian Yang, Jungup Lee, Renwen Zhang, Yi-Chieh Lee

Mental-health stigma remains a pervasive social problem that hampers
treatment-seeking and recovery. Existing resources for training neural models
to finely classify such stigma are limited, relying primarily on social-media
or synthetic data without theoretical underpinnings. To remedy this gap, we
present an expert-annotated, theory-informed corpus of human-chatbot
interviews, comprising 4,141 snippets from 684 participants with documented
socio-cultural backgrounds. Our experiments benchmark state-of-the-art neural
models and empirically unpack the challenges of stigma detection. This dataset
can facilitate research on computationally detecting, neutralizing, and
counteracting mental-health stigma.

### 5. [Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective](http://arxiv.org/pdf/2505.12886v1)

Authors: Zhongxiang Sun, Qipeng Wang, Haoyu Wang, Xiao Zhang, Jun Xu

Large Reasoning Models (LRMs) have shown impressive capabilities in
multi-step reasoning tasks. However, alongside these successes, a more
deceptive form of model error has emerged--Reasoning Hallucination--where
logically coherent but factually incorrect reasoning traces lead to persuasive
yet faulty conclusions. Unlike traditional hallucinations, these errors are
embedded within structured reasoning, making them more difficult to detect and
potentially more harmful. In this work, we investigate reasoning hallucinations
from a mechanistic perspective. We propose the Reasoning Score, which
quantifies the depth of reasoning by measuring the divergence between logits
obtained from projecting late layers of LRMs to the vocabulary space,
effectively distinguishing shallow pattern-matching from genuine deep
reasoning. Using this score, we conduct an in-depth analysis on the ReTruthQA
dataset and identify two key reasoning hallucination patterns: early-stage
fluctuation in reasoning depth and incorrect backtracking to flawed prior
steps. These insights motivate our Reasoning Hallucination Detection (RHD)
framework, which achieves state-of-the-art performance across multiple domains.
To mitigate reasoning hallucinations, we further introduce GRPO-R, an enhanced
reinforcement learning algorithm that incorporates step-level deep reasoning
rewards via potential-based shaping. Our theoretical analysis establishes
stronger generalization guarantees, and experiments demonstrate improved
reasoning quality and reduced hallucination rates.

### 6. [Discretion in the Loop: Human Expertise in Algorithm-Assisted College Advising](http://arxiv.org/pdf/2505.13325v1)

Authors: Sofiia Druchyna, Kara Schechtman, Benjamin Brandon, Jenise Stafford, Hannah Li, Lydia T. Liu

In higher education, many institutions use algorithmic alerts to flag at-risk
students and deliver advising at scale. While much research has focused on
evaluating algorithmic predictions, relatively little is known about how
discretionary interventions by human experts shape outcomes in
algorithm-assisted settings. We study this question using rich quantitative and
qualitative data from a randomized controlled trial of an algorithm-assisted
advising program at Georgia State University. Taking a mixed-methods approach,
we examine whether and how advisors use context unavailable to an algorithm to
guide interventions and influence student success. We develop a causal
graphical framework for human expertise in the interventional setting,
extending prior work on discretion in purely predictive settings. We then test
a necessary condition for discretionary expertise using structured advisor logs
and student outcomes data, identifying several interventions that meet the
criterion for statistical significance. Accordingly, we estimate that 2 out of
3 interventions taken by advisors in the treatment arm were plausibly "expertly
targeted" to students using non-algorithmic context. Systematic qualitative
analysis of advisor notes corroborates these findings, showing that advisors
incorporate diverse forms of contextual information--such as personal
circumstances, financial issues, and student engagement--into their decisions.
Finally, we explore the broader implications of human discretion for long-term
outcomes and equity, using heterogeneous treatment effect estimation. Our
results offer theoretical and practical insight into the real-world
effectiveness of algorithm-supported college advising, and underscore the
importance of accounting for human expertise in the design, evaluation, and
implementation of algorithmic decision systems.

### 7. [Recommender Systems for Democracy: Toward Adversarial Robustness in Voting Advice Applications](http://arxiv.org/pdf/2505.13329v1)

Authors: Frédéric Berdoz, Dustin Brunner, Yann Vonlanthen, Roger Wattenhofer

Voting advice applications (VAAs) help millions of voters understand which
political parties or candidates best align with their views. This paper
explores the potential risks these applications pose to the democratic process
when targeted by adversarial entities. In particular, we expose 11 manipulation
strategies and measure their impact using data from Switzerland's primary VAA,
Smartvote, collected during the last two national elections. We find that
altering application parameters, such as the matching method, can shift a
party's recommendation frequency by up to 105%. Cherry-picking questionnaire
items can increase party recommendation frequency by over 261%, while subtle
changes to parties' or candidates' responses can lead to a 248% increase. To
address these vulnerabilities, we propose adversarial robustness properties
VAAs should satisfy, introduce empirical metrics for assessing the resilience
of various matching methods, and suggest possible avenues for research toward
mitigating the effect of manipulation. Our framework is key to ensuring secure
and reliable AI-based VAAs poised to emerge in the near future.

### Databases

### 1. [Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](http://arxiv.org/pdf/2505.12684v1)

Authors: Yinlin Zhu, Xunkai Li, Jishuo Jia, Miao Hu, Di Wu, Meikang Qiu

Recent advances in graph machine learning have shifted to data-centric
paradigms, driven by two emerging fields: (1) Federated graph learning (FGL)
enables multi-client collaboration but faces challenges from data and task
heterogeneity, limiting its practicality; (2) Graph foundation models (GFM)
offer strong domain generalization but are usually trained on single machines,
missing out on cross-silo data and resources.
  These paradigms are complementary, and their integration brings notable
benefits. Motivated by this, we propose FedGFM, a novel decentralized GFM
training paradigm. However, a key challenge is knowledge entanglement, where
multi-domain knowledge merges into indistinguishable representations, hindering
downstream adaptation.
  To address this, we present FedGFM+, an enhanced framework with two core
modules to reduce knowledge entanglement: (1) AncDAI: A global anchor-based
domain-aware initialization strategy. Before pre-training, each client encodes
its local graph into domain-specific prototypes that serve as semantic anchors.
Synthetic embeddings around these anchors initialize the global model. We
theoretically prove these prototypes are distinguishable across domains,
providing a strong inductive bias to disentangle domain-specific knowledge. (2)
AdaDPP: A local adaptive domain-sensitive prompt pool. Each client learns a
lightweight graph prompt capturing domain semantics during pre-training. During
fine-tuning, prompts from all clients form a pool from which the GFM selects
relevant prompts to augment target graph attributes, improving downstream
adaptation.
  FedGFM+ is evaluated on 8 diverse benchmarks across multiple domains and
tasks, outperforming 20 baselines from supervised learning, FGL, and federated
GFM variants.

### 2. [LLM-KG-Bench 3.0: A Compass for SemanticTechnology Capabilities in the Ocean of LLMs](http://arxiv.org/pdf/2505.13098v1)

Authors: Lars-Peter Meyer, Johannes Frey, Desiree Heim, Felix Brei, Claus Stadler, Kurt Junghanns, Michael Martin

Current Large Language Models (LLMs) can assist developing program code
beside many other things, but can they support working with Knowledge Graphs
(KGs) as well? Which LLM is offering the best capabilities in the field of
Semantic Web and Knowledge Graph Engineering (KGE)? Is this possible to
determine without checking many answers manually? The LLM-KG-Bench framework in
Version 3.0 is designed to answer these questions. It consists of an extensible
set of tasks for automated evaluation of LLM answers and covers different
aspects of working with semantic technologies. In this paper the LLM-KG-Bench
framework is presented in Version 3 along with a dataset of prompts, answers
and evaluations generated with it and several state-of-the-art LLMs.
Significant enhancements have been made to the framework since its initial
release, including an updated task API that offers greater flexibility in
handling evaluation tasks, revised tasks, and extended support for various open
models through the vllm library, among other improvements. A comprehensive
dataset has been generated using more than 30 contemporary open and proprietary
LLMs, enabling the creation of exemplary model cards that demonstrate the
models' capabilities in working with RDF and SPARQL, as well as comparing their
performance on Turtle and JSON-LD RDF serialization tasks.

### 3. [AutoGEEval: A Multimodal and Automated Framework for Geospatial Code Generation on GEE with Large Language Models](http://arxiv.org/pdf/2505.12900v1)

Authors: Shuyang Hou, Zhangxiao Shen, Huayi Wu, Jianyuan Liang, Haoyue Jiao, Yaxian Qing, Xiaopu Zhang, Xu Li, Zhipeng Gui, Xuefeng Guan, Longgang Xiang

Geospatial code generation is emerging as a key direction in the integration
of artificial intelligence and geoscientific analysis. However, there remains a
lack of standardized tools for automatic evaluation in this domain. To address
this gap, we propose AutoGEEval, the first multimodal, unit-level automated
evaluation framework for geospatial code generation tasks on the Google Earth
Engine (GEE) platform powered by large language models (LLMs). Built upon the
GEE Python API, AutoGEEval establishes a benchmark suite (AutoGEEval-Bench)
comprising 1325 test cases that span 26 GEE data types. The framework
integrates both question generation and answer verification components to
enable an end-to-end automated evaluation pipeline-from function invocation to
execution validation. AutoGEEval supports multidimensional quantitative
analysis of model outputs in terms of accuracy, resource consumption, execution
efficiency, and error types. We evaluate 18 state-of-the-art LLMs-including
general-purpose, reasoning-augmented, code-centric, and geoscience-specialized
models-revealing their performance characteristics and potential optimization
pathways in GEE code generation. This work provides a unified protocol and
foundational resource for the development and assessment of geospatial code
generation models, advancing the frontier of automated natural language to
domain-specific code translation.

### Distributed, Parallel, and Cluster Computing

### 1. [Quantum Modeling of Spatial Contiguity Constraints](http://arxiv.org/pdf/2505.12608v1)

Authors: Yunhan Chang, Amr Magdy, Federico M. Spedalieri

Quantum computing has demonstrated potential for solving complex optimization
problems; however, its application to spatial regionalization remains
underexplored. Spatial contiguity, a fundamental constraint requiring spatial
entities to form connected components, significantly increases the complexity
of regionalization problems, which are typically challenging for quantum
modeling. This paper proposes novel quantum formulations based on a flow model
that enforces spatial contiguity constraints. Our scale-aware approach employs
a Discrete Quadratic Model (DQM), solvable directly on quantum annealing
hardware for small-scale datasets. In addition, it designs a hybrid
quantum-classical approach to manage larger-scale problems within existing
hardware limitations. This work establishes a foundational framework for
integrating quantum methods into practical spatial optimization tasks.

### 2. [HydraInfer: Hybrid Disaggregated Scheduling for Multimodal Large Language Model Serving](http://arxiv.org/pdf/2505.12658v1)

Authors: Xianzhe Dong, Tongxuan Liu, Yuting Zeng, Liangyu Liu, Yang Liu, Siyu Wu, Yu Wu, Hailong Yang, Ke Zhang, Jing Li

Multimodal Large Language Models (MLLMs) have been rapidly advancing,
enabling cross-modal understanding and generation, and propelling artificial
intelligence towards artificial general intelligence. However, existing MLLM
inference systems are typically designed based on the architecture of language
models, integrating image processing and language processing as a single
scheduling unit. This design struggles to accommodate the heterogeneous demands
of different stages in terms of computational resources, memory access
patterns, and service-level objectives (SLOs), leading to low resource
utilization and high request latency, ultimately failing to meet the service
requirements of diverse inference scenarios.
  To address these challenges, we propose HydraInfer, an efficient MLLM
inference system that adopts a Hybrid Encode-Prefill-Decode (EPD)
Disaggregation architecture. By scheduling the three stages - encode, prefill,
and decode - onto separate heterogeneous inference instances, the system
flexibly reallocates resources across stages, significantly reducing idle
computation, alleviating resource bottlenecks, and improving overall system
throughput and scalability. In addition, HydraInfer supports a stage-level
batching strategy that enhances load balancing, enables parallel execution of
visual and language models, and further optimizes inference performance.
Experiments under real multimodal inference workloads demonstrate that
HydraInfer can achieve up to 4x higher inference throughput compared to
state-of-the-art systems (e.g., vLLM) on a single-node 8xH800 GPU cluster,
while meeting the 90th percentile request SLO.

### 3. [MTGRBoost: Boosting Large-scale Generative Recommendation Models in Meituan](http://arxiv.org/pdf/2505.12663v1)

Authors: Yuxiang Wang, Xiao Yan, Chi Ma, Mincong Huang, Xiaoguang Li, Lei Yu, Chuan Liu, Ruidong Han, He Jiang, Bin Yin, Shangyu Chen, Fei Jiang, Xiang Li, Wei Lin, Haowei Han, Bo Du, Jiawei Jiang

Recommendation is crucial for both user experience and company revenue, and
generative recommendation models (GRMs) are shown to produce quality
recommendations recently. However, existing systems are limited by insufficient
functionality support and inefficient implementations for training GRMs in
industrial scenarios. As such, we introduce MTGRBoost as an efficient and
scalable system for GRM training. Specifically, to handle the real-time
insert/delete of sparse embedding entries, MTGRBoost employs dynamic hash
tables to replace static tables. To improve efficiency, MTGRBoost conducts
dynamic sequence balancing to address the computation load imbalances among
GPUs and adopts embedding ID deduplication alongside automatic table merging to
accelerate embedding lookup. MTGRBoost also incorporates implementation
optimizations including checkpoint resuming, mixed precision training, gradient
accumulation, and operator fusion. Extensive experiments show that MTGRBoost
improves training throughput by $1.6 \times$ -- $2.4\times$ while achieving
good scalability when running over 100 GPUs. MTGRBoost has been deployed for
many applications in Meituan and is now handling hundreds of millions of
requests on a daily basis.

### 4. [A Study on Distributed Strategies for Deep Learning Applications in GPU Clusters](http://arxiv.org/pdf/2505.12832v1)

Authors: Md Sultanul Islam Ovi

As deep learning models grow in size and complexity, training them
efficiently on single GPUs becomes increasingly infeasible. This study
investigates the effectiveness of several distributed training
strategies-Distributed Data Parallel (DDP), Fully Sharded Data Parallelism
(FSDP), and Parameter Server (PS) models-for scalable deep learning on GPU
clusters. We conduct empirical evaluations across multiple models and datasets
to assess trade-offs in memory usage, training time, GPU utilization, and model
accuracy. Our results show that while FSDP reduces GPU memory usage by over
60%, it increases training time by up to 6x compared to DDP. In contrast,
asynchronous PS training improves throughput but can lead to degraded accuracy
due to stale updates. Through comprehensive analysis, we provide practical
insights into the strengths and limitations of each strategy, offering guidance
for selecting suitable methods based on system constraints and training
objectives.

### 5. [Optimization of Hybrid Quantum-Classical Algorithms](http://arxiv.org/pdf/2505.12853v1)

Authors: Lian Remme, Alexander Weinert, Andre Waschk

Quantum computers do not run in isolation; rather, they are embedded in
quantum-classical hybrid architectures. In these setups, a quantum processing
unit communicates with a classical device in near-real time. To enable
efficient hybrid computations, it is mandatory to optimize quantum-classical
hybrid code. To the best of our knowledge, no previous work on the optimization
of hybrid code nor on metrics for which to optimize such code exists.
  In this work, we take a step towards optimization of hybrid programs by
introducing seven optimization routines and three metrics to evaluate the
effectiveness of the optimization. We implement these routines for the hybrid
quantum language Quil and show that our optimizations improve programs
according to our metrics. This lays the foundation for new kinds of hybrid
optimizers that enable real-time collaboration between quantum and classical
devices.

### 6. [Minos: Exploiting Cloud Performance Variation with Function-as-a-Service Instance Selection](http://arxiv.org/pdf/2505.12928v1)

Authors: Trever Schirmer, Valentin Carl, Nils Höller, Tobias Pfandzelter, David Bermbach

Serverless Function-as-a-Service (FaaS) is a popular cloud paradigm to
quickly and cheaply implement complex applications. Because the function
instances cloud providers start to execute user code run on shared
infrastructure, their performance can vary. From a user perspective, slower
instances not only take longer to complete, but also increase cost due to the
pay-per-use model of FaaS services where execution duration is billed with
microsecond accuracy. In this paper, we present Minos, a system to take
advantage of this performance variation by intentionally terminating instances
that are slow. Fast instances are not terminated, so that they can be re-used
for subsequent invocations. One use case for this are data processing and
machine learning workflows, which often download files as a first step, during
which Minos can run a short benchmark. Only if the benchmark passes, the main
part of the function is actually executed. Otherwise, the request is re-queued
and the instance crashes itself, so that the platform has to assign the request
to another (potentially faster) instance. In our experiments, this leads to a
speedup of up to 13% in the resource intensive part of a data processing
workflow, resulting in up to 4% faster overall performance (and consequently 4%
cheaper prices). Longer and complex workflows lead to increased savings, as the
pool of fast instances is re-used more often. For platforms exhibiting this
behavior, users get better performance and save money by wasting more of the
platforms resources.

### 7. [Digital Twins in the Cloud: A Modular, Scalable and Interoperable Framework for Accelerating Verification and Validation of Autonomous Driving Solutions](http://arxiv.org/pdf/2505.12661v1)

Authors: Tanmay Vilas Samak, Chinmay Vilas Samak, Giovanni Martino, Pranav Nair, Venkat Krovi

Verification and validation (V&V) of autonomous vehicles (AVs) typically
requires exhaustive testing across a variety of operating environments and
driving scenarios including rare, extreme, or hazardous situations that might
be difficult or impossible to capture in reality. Additionally, physical V&V
methods such as track-based evaluations or public-road testing are often
constrained by time, cost, and safety, which motivates the need for virtual
proving grounds. However, the fidelity and scalability of simulation-based V&V
methods can quickly turn into a bottleneck. In such a milieu, this work
proposes a virtual proving ground that flexibly scales digital twins within
high-performance computing clusters (HPCCs) and automates the V&V process.
Here, digital twins enable high-fidelity virtual representation of the AV and
its operating environments, allowing extensive scenario-based testing.
Meanwhile, HPCC infrastructure brings substantial advantages in terms of
computational power and scalability, enabling rapid iterations of simulations,
processing and storage of massive amounts of data, and deployment of
large-scale test campaigns, thereby reducing the time and cost associated with
the V&V process. We demonstrate the efficacy of this approach through a case
study that focuses on the variability analysis of a candidate autonomy
algorithm to identify potential vulnerabilities in its perception, planning,
and control sub-systems. The modularity, scalability, and interoperability of
the proposed framework are demonstrated by deploying a test campaign comprising
256 test cases on two different HPCC architectures to ensure continuous
operation in a publicly shared resource setting. The findings highlight the
ability of the proposed framework to accelerate and streamline the V&V process,
thereby significantly compressing (~30x) the timeline.

### 8. [Learning in Chaos: Efficient Autoscaling and Self-healing for Distributed Training at the Edge](http://arxiv.org/pdf/2505.12815v1)

Authors: Wenjiao Feng, Rongxing Xiao, Zonghang Li, Hongfang Yu, Gang Sun, Long Luo, Mohsen Guizani, Qirong Ho

Frequent node and link changes in edge AI clusters disrupt distributed
training, while traditional checkpoint-based recovery and cloud-centric
autoscaling are too slow for scale-out and ill-suited to chaotic and
self-governed edge. This paper proposes Chaos, a resilient and scalable edge
distributed training system with built-in self-healing and autoscaling. It
speeds up scale-out by using multi-neighbor replication with fast shard
scheduling, allowing a new node to pull the latest training state from nearby
neighbors in parallel while balancing the traffic load between them. It also
uses a cluster monitor to track resource and topology changes to assist
scheduler decisions, and handles scaling events through peer negotiation
protocols, enabling fully self-governed autoscaling without a central admin.
Extensive experiments show that Chaos consistently achieves much lower
scale-out delays than Pollux, EDL, and Autoscaling, and handles scale-in,
connect-link, and disconnect-link events within 1 millisecond, making it
smoother to handle node joins, exits, and failures. It also delivers the lowest
idle time, showing superior resource use and scalability as the cluster grows.

### 9. [Computing the Schulze Method for Large-Scale Preference Data Sets](http://arxiv.org/pdf/2505.12976v1)

Authors: Theresa Csar, Martin Lackner, Reinhard Pichler

The Schulze method is a voting rule widely used in practice and enjoys many
positive axiomatic properties. While it is computable in polynomial time, its
straight-forward implementation does not scale well for large elections. In
this paper, we develop a highly optimised algorithm for computing the Schulze
method with Pregel, a framework for massively parallel computation of graph
problems, and demonstrate its applicability for large preference data sets. In
addition, our theoretic analysis shows that the Schulze method is indeed
particularly well-suited for parallel computation, in stark contrast to the
related ranked pairs method. More precisely we show that winner determination
subject to the Schulze method is NL-complete, whereas this problem is
P-complete for the ranked pairs method.

### 10. [eBPF-Based Instrumentation for Generalisable Diagnosis of Performance Degradation](http://arxiv.org/pdf/2505.13160v1)

Authors: Diogo Landau, Jorge Barbosa, Nishant Saurabh

Online Data Intensive applications (e.g. message brokers, ML inference and
databases) are core components of the modern internet, providing critical
functionalities to connecting services. The load variability and interference
they experience are generally the main causes of Quality of Service (QoS)
degradation, harming depending applications, and resulting in an impaired
end-user experience. Uncovering the cause of QoS degradation requires detailed
instrumentation of an application's activity. Existing generalisable approaches
utilise readily available system metrics that encode interference in kernel
metrics, but unfortunately, these approaches lack the required detail to
pinpoint granular causes of performance degradation (e.g., lock, disk and CPU
contention). In contrast, this paper explores the use of fine-grained
system-level metrics to facilitate an application-agnostic diagnosis of QoS
degradation. To this end, we introduce and implement $16$ $\textit{eBPF-based
metrics}$ spanning over six kernel subsystems, which capture statistics over
kernel events that often highlight obstacles impeding an application's
progress. We demonstrate the use of our $\textit{eBPF-based metrics}$ through
extensive experiments containing a representative set of online data-intensive
applications. Results show that the implemented metrics can deconstruct
performance degradation when applications face variable workload patterns and
common resource contention scenarios, while also revealing applications'
internal architecture constraints.

### Digital Libraries

### 1. [CHAD-KG: A Knowledge Graph for Representing Cultural Heritage Objects and Digitisation Paradata](http://arxiv.org/pdf/2505.13276v1)

Authors: Sebastian Barzaghi, Arianna Moretti, Ivan Heibi, Silvio Peroni

This paper presents CHAD-KG, a knowledge graph designed to describe
bibliographic metadata and digitisation paradata of cultural heritage objects
in exhibitions, museums, and collections. It also documents the related data
model and materialisation engine. Originally based on two tabular datasets, the
data was converted into RDF according to CHAD-AP, an OWL application profile
built on standards like CIDOC-CRM, LRMoo, CRMdig, and Getty AAT. A reproducible
pipeline, developed with a Morph-KGC extension, was used to generate the graph.
CHAD-KG now serves as the main metadata source for the Digital Twin of the
temporary exhibition titled \emph{The Other Renaissance - Ulisse Aldrovandi and
The Wonders Of The World}, and other collections related to the digitisation
work under development in a nationwide funded project, i.e. Project CHANGES
(https://fondazionechanges.org). To ensure accessibility and reuse, it offers a
SPARQL endpoint, a user interface, open documentation, and is published on
Zenodo under a CC0 license. The project improves the semantic interoperability
of cultural heritage data, with future work aiming to extend the data model and
materialisation pipeline to better capture the complexities of acquisition and
digitisation, further enrich the dataset and broaden its relevance to similar
initiatives.

### Discrete Mathematics

### 1. [Independent Set Enumeration in King Graphs by Tensor Network Contractions](http://arxiv.org/pdf/2505.12776v1)

Authors: Kai Liang

This paper discusses the enumeration of independent sets in king graphs of
size $m \times n$, based on the tensor network contractions algorithm given in
reference~\cite{tilEnum}. We transform the problem into Wang tiling enumeration
within an $(m+1) \times (n+1)$ rectangle and compute the results for all cases
where $m + n \leq 79$ using tensor network contraction algorithm, and provided
an approximation for larger $m, n$.
  Using the same algorithm, we also enumerated independent sets with vertex
number restrictions. Based on the results, we analyzed the vertex number that
maximize the enumeration for each pair $(m, n)$. Additionally, we compute the
corresponding weighted enumeration, where each independent set is weighted by
the number of its vertices (i.e., the total sum of vertices over all
independent sets). The approximations for larger $m, n$ are given as well.
  Our results have added thousands of new items to the OEIS sequences A089980
and A193580. In addition, the combinatorial problems above are closely related
to the hard-core model in physics. We estimate some important constants based
on the existing results, and the relative error between our estimation of the
entropy constant and the existing results is less than $10^{-9}$.

### 2. [A Necessary Condition for Connectedness of Solutions to Integer Linear Systems](http://arxiv.org/pdf/2505.12930v1)

Authors: Takasugu Shigenobu, Naoyuki Kamiyama

An integer linear system is a set of inequalities with integer constraints.
The solution graph of an integer linear system is an undirected graph defined
on the set of feasible solutions to the integer linear system. In this graph, a
pair of feasible solutions is connected by an edge if the Hamming distance
between them is one. In this paper, we consider a condition under which the
solution graph is connected for any right-hand side vector. First, we prove
that if the solution graph is connected for any right-hand side vector, then
the coefficient matrix of the system does not contain some forbidden pattern as
a submatrix. Next, we prove that if at least one of (i) the number of rows is
at most 3, (ii) the number of columns is at most 2, (iii) the number of rows is
4 and the number of columns is 3 holds, then the condition that the coefficient
matrix of the system does not contain the forbidden pattern is a sufficient
condition under which the solution graph is connected for any right-hand side
vector. This result is stronger than a known necessary and sufficient condition
since the set of coefficient matrix dimensions is strictly larger.

### 3. [On expectations and variances in the hard-core model on bounded degree graphs](http://arxiv.org/pdf/2505.13396v1)

Authors: Ewan Davies, Juspreet Singh Sandhu, Brian Tan

We extend the study of the occupancy fraction of the hard-core model in two
novel directions. One direction gives a tight lower bound in terms of
individual vertex degrees, extending work of Sah, Sawhney, Stoner and Zhao
which bounds the partition function. The other bounds the variance of the size
of an independent set drawn from the model, which is strictly stronger than
bounding the occupancy fraction.
  In the setting of triangle-free graphs, we make progress on a recent
conjecture of Buys, van den Heuvel and Kang on extensions of Shearer's classic
bounds on the independence number to the occupancy fraction of the hard-core
model.
  Sufficiently strong lower bounds on both the expectation and the variance in
triangle-free graphs have the potential to improve the known bounds on the
off-diagonal Ramsey number $R(3,t)$, and to shed light on the algorithmic
barrier one observes for independent sets in sparse random graphs.

### 4. [Ergodic properties of concurrent systems](http://arxiv.org/pdf/2505.12810v1)

Authors: Samy Abbes, Vincent Jugé

A concurrent system is defined as a monoid action of a trace monoid on a
finite set of states. Concurrent systems represent state models where the state
is distributed and where state changes are local. Starting from a spectral
property on the combinatorics of concurrent systems, we prove the existence and
uniqueness of a Markov measure on the space of infinite trajectories relatively
to any weight distributions. In turn, we obtain a combinatorial result by
proving that the kernel of the associated M\"obius matrix has dimension 1; the
M\"obius matrix extends in this context the M\"obius polynomial of a trace
monoid. We study ergodic properties of irreducible concurrent systems and we
prove a Strong law of large numbers. It allows us to introduce the speedup as a
measurement of the average amount of concurrency within infinite trajectories.
Examples are studied.

### Data Structures and Algorithms

### 1. [Fast and Simple Densest Subgraph with Predictions](http://arxiv.org/pdf/2505.12600v1)

Authors: Thai Bui, Hoa T. Vu

We study the densest subgraph problem and its variants through the lens of
learning-augmented algorithms. For this problem, the greedy algorithm by
Charikar (APPROX 2000) provides a linear-time $ 1/2 $-approximation, while
computing the exact solution typically requires solving a linear program or
performing maximum flow computations.We show that given a partial solution,
i.e., one produced by a machine learning classifier that captures at least a $
(1 - \epsilon) $-fraction of nodes in the optimal subgraph, it is possible to
design an extremely simple linear-time algorithm that achieves a provable $ (1
- \epsilon) $-approximation. Our approach also naturally extends to the
directed densest subgraph problem and several NP-hard variants.An experiment on
the Twitch Ego Nets dataset shows that our learning-augmented algorithm
outperforms Charikar's greedy algorithm and a baseline that directly returns
the predicted densest subgraph without additional algorithmic processing.

### 2. [More Efforts Towards Fixed-Parameter Approximability of Multiwinner Rules](http://arxiv.org/pdf/2505.12699v1)

Authors: Sushmita Gupta, Pallavi Jain, Souvik Saha, Saket Saurabh, Anannya Upasana

Multiwinner Elections have emerged as a prominent area of research with
numerous practical applications. We contribute to this area by designing
parameterized approximation algorithms and also resolving an open question by
Yang and Wang [AAMAS'18]. More formally, given a set of candidates,
\mathcal{C}, a set of voters,\mathcal{V}, approving a subset of candidates
(called approval set of a voter), and an integer $k$, we consider the problem
of selecting a ``good'' committee using Thiele rules. This problem is
computationally challenging for most Thiele rules with monotone submodular
satisfaction functions, as there is no (1-\frac{1}{e}-\epsilon)\footnote{Here,
$e$ denotes the base of the natural logarithm.}-approximation algorithm in
f(k)(|\mathcal{C}| + |\mathcal{V}|)^{o(k)} time for any fixed $\epsilon > 0$
and any computable function $f$, and no {\sf PTAS} even when the length of
approval set is two. Skowron [WINE'16] designed an approximation scheme running
in FPT time parameterized by the combined parameter, size of the approval set
and $k$. In this paper, we consider a parameter $d+k$ (no $d$ voters approve
the same set of $d$ candidates), where $d$ is upper bounded by the size of the
approval set (thus, can be much smaller).
  With respect to this parameter, we design parameterized approximation
schemes, a lossy polynomial-time preprocessing method, and show that an extra
committee member suffices to achieve the desired score (i.e., $1$-additive
approximation). Additionally, we resolve an open question by Yang and
Wang~[AAMAS'18] regarding the fixed-parameter tractability of the problem under
the PAV rule with the total score as the parameter, demonstrating that it
admits an FPT algorithm.

### 3. [A Faster Parametric Search for the Integral Quickest Transshipment Problem](http://arxiv.org/pdf/2505.12975v1)

Authors: Mariia Anapolska, Dario van den Boom, Christina Büsing, Timo Gersing

Algorithms for computing fractional solutions to the quickest transshipment
problem have been significantly improved since Hoppe and Tardos first solved
the problem in strongly polynomial time. For integral solutions, runtime
improvements are limited to general progress on submodular function
minimization, which is an integral part of Hoppe and Tardos' algorithm. Yet, no
structural improvements on their algorithm itself have been proposed. We
replace two central subroutines in the algorithm with methods that require
vastly fewer minimizations of submodular functions. This improves the
state-of-the-art runtime from $ \tilde{O}(m^4 k^{15}) $ down to $ \tilde{O}(m^2
k^5 + m^4 k^2) $, where $ k $ is the number of terminals and $ m $ is the
number of arcs.

### 4. [Counting Graphlets of Size $k$ under Local Differential Privacy](http://arxiv.org/pdf/2505.12954v1)

Authors: Vorapong Suppakitpaisarn, Donlapark Ponnoprat, Nicha Hirankarn, Quentin Hillebrand

The problem of counting subgraphs or graphlets under local differential
privacy is an important challenge that has attracted significant attention from
researchers. However, much of the existing work focuses on small graphlets like
triangles or $k$-stars. In this paper, we propose a non-interactive, locally
differentially private algorithm capable of counting graphlets of any size $k$.
When $n$ is the number of nodes in the input graph, we show that the expected
$\ell_2$ error of our algorithm is $O(n^{k - 1})$. Additionally, we prove that
there exists a class of input graphs and graphlets of size $k$ for which any
non-interactive counting algorithm incurs an expected $\ell_2$ error of
$\Omega(n^{k - 1})$, demonstrating the optimality of our result. Furthermore,
we establish that for certain input graphs and graphlets, any locally
differentially private algorithm must have an expected $\ell_2$ error of
$\Omega(n^{k - 1.5})$. Our experimental results show that our algorithm is more
accurate than the classical randomized response method.

### Emerging Technologies

### 1. [Physics-Aware Compilation for Parallel Quantum Circuit Execution on Neutral Atom Arrays](http://arxiv.org/pdf/2505.13049v1)

Authors: Geng Chen, Guowu Yang, Wenjie Sun, Lianhui Yu, Guangwei Deng, Desheng Zheng, Xiaoyu Li

Neutral atom quantum computers are one of the most promising quantum
architectures, offering advantages in scalability, dynamic reconfigurability,
and potential for large-scale implementations. These characteristics create
unique compilation challenges, especially regarding compilation efficiency
while adapting to hardware flexibility. However, existing methods encounter
significant performance bottlenecks at scale, hindering practical applications.
We propose Physics-Aware Compilation (PAC), a method that improves compilation
efficiency while preserving the inherent flexibility of neutral atom systems.
PAC introduces physics-aware hardware plane partitioning that strategically
allocates hardware resources based on physical device characteristics like AOD
and SLM trap properties and qubit mobility constraints. Additionally, it
implements parallel quantum circuit division with an improved Kernighan-Lin
algorithm that enables simultaneous execution across independent regions while
maintaining circuit fidelity. Our experimental evaluation compares PAC with
state-of-the-art methods across increasingly larger array sizes ranging from
16x16 to 64x64 qubits. Results demonstrate that PAC achieves up to 78.5x
speedup on 16x16 arrays while maintaining comparable circuit quality. PAC's
compilation efficiency advantage increases with system scale, demonstrating
scalability for practical quantum applications on larger arrays. PAC explores a
viable path for practical applications of neutral atom quantum computers by
effectively addressing the tension between compilation efficiency and hardware
flexibility.

### 2. [Leveraging Large Reconfigurable Intelligent Surfaces as Anchors for Near-Field Positioning](http://arxiv.org/pdf/2505.12730v1)

Authors: Zeyu Huang, Markus Rupp, Stefan Schwarz

In this work, we present a recent investigation on leveraging large
reconfigurable intelligent surfaces (RIS) as anchors for positioning in
wireless communication systems. Unlike existing approaches, we explicitly
address the uncertainty arising from the substantial physical size of the RIS,
particularly relevant when a user equipment resides in the near field, and
propose a method that ensures accurate positioning under these conditions. We
derive the corresponding Cramer-Rao bound for our scheme and validate the
effectiveness of our scheme through numerical experiments, highlighting both
the feasibility and potential of our approach.

### 3. [2T1R Regulated Memristor Conductance Control Array Architecture for Neuromorphic Computing using 28nm CMOS Technology](http://arxiv.org/pdf/2505.12830v1)

Authors: Neethu Kuriakose, Arun Ashok, Christian Grewing, André Zambanini, Stefan van Waasen

Memristors are promising devices for scalable and low power, in-memory
computing to improve the energy efficiency of a rising computational demand.
The crossbar array architecture with memristors is used for vector matrix
multiplication (VMM) and acts as kernels in neuromorphic computing. The analog
conductance control in a memristor is achieved by applying voltage or current
through it. A basic 1T1R array is suitable to avoid sneak path issues but
suffer from wire resistances, which affects the read and write procedures. A
conductance control scheme with a regulated voltage source will improve the
architecture and reduce the possible potential divider effects. A change in
conductance is also possible with the provision of a regulated current source
and measuring the voltage across the memristors. A regulated 2T1R memristor
conductance control architecture is proposed in this work, which avoids the
potential divider effect and virtual ground scenario in a regular crossbar
scheme, as well as conductance control by passing a regulated current through
memristors. The sneak path current is not allowed to pass by the provision of
ground potential to both terminals of memristors.

### 4. [Hardware-Adaptive and Superlinear-Capacity Memristor-based Associative Memory](http://arxiv.org/pdf/2505.12960v1)

Authors: Chengping He, Mingrui Jiang, Keyi Shan, Szu-Hao Yang, Zefan Li, Shengbo Wang, Giacomo Pedretti, Jim Ignowski, Can Li

Brain-inspired computing aims to mimic cognitive functions like associative
memory, the ability to recall complete patterns from partial cues. Memristor
technology offers promising hardware for such neuromorphic systems due to its
potential for efficient in-memory analog computing. Hopfield Neural Networks
(HNNs) are a classic model for associative memory, but implementations on
conventional hardware suffer from efficiency bottlenecks, while prior
memristor-based HNNs faced challenges with vulnerability to hardware defects
due to offline training, limited storage capacity, and difficulty processing
analog patterns. Here we introduce and experimentally demonstrate on integrated
memristor hardware a new hardware-adaptive learning algorithm for associative
memories that significantly improves defect tolerance and capacity, and
naturally extends to scalable multilayer architectures capable of handling both
binary and continuous patterns. Our approach achieves 3x effective capacity
under 50% device faults compared to state-of-the-art methods. Furthermore, its
extension to multilayer architectures enables superlinear capacity scaling
(\(\propto N^{1.49}\ for binary patterns) and effective recalling of continuous
patterns (\propto N^{1.74}\ scaling), as compared to linear capacity scaling
for previous HNNs. It also provides flexibility to adjust capacity by tuning
hidden neurons for the same-sized patterns. By leveraging the massive
parallelism of the hardware enabled by synchronous updates, it reduces energy
by 8.8x and latency by 99.7% for 64-dimensional patterns over asynchronous
schemes, with greater improvements at scale. This promises the development of
more reliable memristor-based associative memory systems and enables new
applications research due to the significantly improved capacity, efficiency,
and flexibility.

### 5. [A Path to Universal Neural Cellular Automata](http://arxiv.org/pdf/2505.13058v1)

Authors: Gabriel Béna, Maxence Faldor, Dan F. M. Goodman, Antoine Cully

Cellular automata have long been celebrated for their ability to generate
complex behaviors from simple, local rules, with well-known discrete models
like Conway's Game of Life proven capable of universal computation. Recent
advancements have extended cellular automata into continuous domains, raising
the question of whether these systems retain the capacity for universal
computation. In parallel, neural cellular automata have emerged as a powerful
paradigm where rules are learned via gradient descent rather than manually
designed. This work explores the potential of neural cellular automata to
develop a continuous Universal Cellular Automaton through training by gradient
descent. We introduce a cellular automaton model, objective functions and
training strategies to guide neural cellular automata toward universal
computation in a continuous setting. Our experiments demonstrate the successful
training of fundamental computational primitives - such as matrix
multiplication and transposition - culminating in the emulation of a neural
network solving the MNIST digit classification task directly within the
cellular automata state. These results represent a foundational step toward
realizing analog general-purpose computers, with implications for understanding
universal computation in continuous dynamics and advancing the automated
discovery of complex cellular automata behaviors via machine learning.

### 6. [Learning Driven Elastic Task Multi-Connectivity Immersive Computing Systems](http://arxiv.org/pdf/2505.13331v1)

Authors: Babak Badnava, Jacob Chakareski, Morteza Hashemi

In virtual reality (VR) environments, computational tasks exhibit an elastic
nature, meaning they can dynamically adjust based on various user and system
constraints. This elasticity is essential for maintaining immersive
experiences; however, it also introduces challenges for communication and
computing in VR systems. In this paper, we investigate elastic task offloading
for multi-user edge-computing-enabled VR systems with multi-connectivity,
aiming to maximize the computational energy-efficiency (computational
throughput per unit of energy consumed). To balance the induced communication,
computation, energy consumption, and quality of experience trade-offs due to
the elasticity of VR tasks, we formulate a constrained stochastic computational
energy-efficiency optimization problem that integrates the
multi-connectivity/multi-user action space and the elastic nature of VR
computational tasks. We formulate a centralized phasic policy gradient (CPPG)
framework to solve the problem of interest online, using only prior elastic
task offloading statistics (energy consumption, response time, and transmission
time), and task information (i.e., task size and computational intensity),
while observing the induced system performance (energy consumption and
latency). We further extend our approach to decentralized learning by
formulating an independent phasic policy gradient (IPPG) method and a
decentralized shared multi-armed bandit (DSMAB) method. We train our methods
with real-world 4G, 5G, and WiGig network traces and 360 video datasets to
evaluate their performance in terms of response time, energy efficiency,
scalability, and delivered quality of experience. We also provide a
comprehensive analysis of task size and its effect on offloading policy and
system performance. In particular, we show that CPPG reduces latency by 28% and
energy consumption by 78% compared to IPPG.

### 7. [Neural-Enhanced Rate Adaptation and Computation Distribution for Emerging mmWave Multi-User 3D Video Streaming Systems](http://arxiv.org/pdf/2505.13337v1)

Authors: Babak Badnava, Jacob Chakareski, Morteza Hashemi

We investigate multitask edge-user communication-computation resource
allocation for $360^\circ$ video streaming in an edge-computing enabled
millimeter wave (mmWave) multi-user virtual reality system. To balance the
communication-computation trade-offs that arise herein, we formulate a video
quality maximization problem that integrates interdependent
multitask/multi-user action spaces and rebuffering time/quality variation
constraints. We formulate a deep reinforcement learning framework for
\underline{m}ulti-\underline{t}ask \underline{r}ate adaptation and
\underline{c}omputation distribution (MTRC) to solve the problem of interest.
Our solution does not rely on a priori knowledge about the environment and uses
only prior video streaming statistics (e.g., throughput, decoding time, and
transmission delay), and content information, to adjust the assigned video
bitrates and computation distribution, as it observes the induced streaming
performance online. Moreover, to capture the task interdependence in the
environment, we leverage neural network cascades to extend our MTRC method to
two novel variants denoted as R1C2 and C1R2. We train all three methods with
real-world mmWave network traces and $360^\circ$ video datasets to evaluate
their performance in terms of expected quality of experience (QoE), viewport
peak signal-to-noise ratio (PSNR), rebuffering time, and quality variation. We
outperform state-of-the-art rate adaptation algorithms, with C1R2 showing best
results and achieving $5.21-6.06$ dB PSNR gains, $2.18-2.70$x rebuffering time
reduction, and $4.14-4.50$ dB quality variation reduction.

### Graphics

### 1. [HIL: Hybrid Imitation Learning of Diverse Parkour Skills from Videos](http://arxiv.org/pdf/2505.12619v1)

Authors: Jiashun Wang, Yifeng Jiang, Haotian Zhang, Chen Tessler, Davis Rempe, Jessica Hodgins, Xue Bin Peng

Recent data-driven methods leveraging deep reinforcement learning have been
an effective paradigm for developing controllers that enable physically
simulated characters to produce natural human-like behaviors. However, these
data-driven methods often struggle to adapt to novel environments and compose
diverse skills coherently to perform more complex tasks. To address these
challenges, we propose a hybrid imitation learning (HIL) framework that
combines motion tracking, for precise skill replication, with adversarial
imitation learning, to enhance adaptability and skill composition. This hybrid
learning framework is implemented through parallel multi-task environments and
a unified observation space, featuring an agent-centric scene representation to
facilitate effective learning from the hybrid parallel environments. Our
framework trains a unified controller on parkour data sourced from Internet
videos, enabling a simulated character to traverse through new environments
using diverse and life-like parkour skills. Evaluations across challenging
parkour environments demonstrate that our method improves motion quality,
increases skill diversity, and achieves competitive task completion compared to
previous learning-based methods.

### 2. [MGPBD: A Multigrid Accelerated Global XPBD Solver](http://arxiv.org/pdf/2505.13390v1)

Authors: Chunlei Li, Peng Yu, Tiantian Liu, Siyuan Yu, Yuting Xiao, Shuai Li, Aimin Hao, Yang Gao, Qinping Zhao

We introduce a novel Unsmoothed Aggregation (UA) Algebraic Multigrid (AMG)
method combined with Preconditioned Conjugate Gradient (PCG) to overcome the
limitations of Extended Position-Based Dynamics (XPBD) in high-resolution and
high-stiffness simulations. While XPBD excels in simulating deformable objects
due to its speed and simplicity, its nonlinear Gauss-Seidel (GS) solver often
struggles with low-frequency errors, leading to instability and stalling
issues, especially in high-resolution, high-stiffness simulations. Our
multigrid approach addresses these issues efficiently by leveraging AMG. To
reduce the computational overhead of traditional AMG, where prolongator
construction can consume up to two-thirds of the runtime, we propose a lazy
setup strategy that reuses prolongators across iterations based on matrix
structure and physical significance. Furthermore, we introduce a simplified
method for constructing near-kernel components by applying a few sweeps of
iterative methods to the homogeneous equation, achieving convergence rates
comparable to adaptive smoothed aggregation (adaptive-SA) at a lower
computational cost. Experimental results demonstrate that our method
significantly improves convergence rates and numerical stability, enabling
efficient and stable high-resolution simulations of deformable objects.

### 3. [UniHM: Universal Human Motion Generation with Object Interactions in Indoor Scenes](http://arxiv.org/pdf/2505.12774v1)

Authors: Zichen Geng, Zeeshan Hayder, Wei Liu, Ajmal Mian

Human motion synthesis in complex scenes presents a fundamental challenge,
extending beyond conventional Text-to-Motion tasks by requiring the integration
of diverse modalities such as static environments, movable objects, natural
language prompts, and spatial waypoints. Existing language-conditioned motion
models often struggle with scene-aware motion generation due to limitations in
motion tokenization, which leads to information loss and fails to capture the
continuous, context-dependent nature of 3D human movement. To address these
issues, we propose UniHM, a unified motion language model that leverages
diffusion-based generation for synthesizing scene-aware human motion. UniHM is
the first framework to support both Text-to-Motion and Text-to-Human-Object
Interaction (HOI) in complex 3D scenes. Our approach introduces three key
contributions: (1) a mixed-motion representation that fuses continuous 6DoF
motion with discrete local motion tokens to improve motion realism; (2) a novel
Look-Up-Free Quantization VAE (LFQ-VAE) that surpasses traditional VQ-VAEs in
both reconstruction accuracy and generative performance; and (3) an enriched
version of the Lingo dataset augmented with HumanML3D annotations, providing
stronger supervision for scene-specific motion learning. Experimental results
demonstrate that UniHM achieves comparative performance on the OMOMO benchmark
for text-to-HOI synthesis and yields competitive results on HumanML3D for
general text-conditioned motion generation.

### 4. [SounDiT: Geo-Contextual Soundscape-to-Landscape Generation](http://arxiv.org/pdf/2505.12734v1)

Authors: Junbo Wang, Haofeng Tan, Bowen Liao, Albert Jiang, Teng Fei, Qixing Huang, Zhengzhong Tu, Shan Ye, Yuhao Kang

We present a novel and practically significant problem-Geo-Contextual
Soundscape-to-Landscape (GeoS2L) generation-which aims to synthesize
geographically realistic landscape images from environmental soundscapes. Prior
audio-to-image generation methods typically rely on general-purpose datasets
and overlook geographic and environmental contexts, resulting in unrealistic
images that are misaligned with real-world environmental settings. To address
this limitation, we introduce a novel geo-contextual computational framework
that explicitly integrates geographic knowledge into multimodal generative
modeling. We construct two large-scale geo-contextual multimodal datasets,
SoundingSVI and SonicUrban, pairing diverse soundscapes with real-world
landscape images. We propose SounDiT, a novel Diffusion Transformer (DiT)-based
model that incorporates geo-contextual scene conditioning to synthesize
geographically coherent landscape images. Furthermore, we propose a
practically-informed geo-contextual evaluation framework, the Place Similarity
Score (PSS), across element-, scene-, and human perception-levels to measure
consistency between input soundscapes and generated landscape images. Extensive
experiments demonstrate that SounDiT outperforms existing baselines in both
visual fidelity and geographic settings. Our work not only establishes
foundational benchmarks for GeoS2L generation but also highlights the
importance of incorporating geographic domain knowledge in advancing multimodal
generative models, opening new directions at the intersection of generative AI,
geography, urban planning, and environmental sciences.

### 5. [AdaToken-3D: Dynamic Spatial Gating for Efficient 3D Large Multimodal-Models Reasoning](http://arxiv.org/pdf/2505.12782v1)

Authors: Kai Zhang, Xingyu Chen, Xiaofeng Zhang

Large Multimodal Models (LMMs) have become a pivotal research focus in deep
learning, demonstrating remarkable capabilities in 3D scene understanding.
However, current 3D LMMs employing thousands of spatial tokens for multimodal
reasoning suffer from critical inefficiencies: excessive computational overhead
and redundant information flows. Unlike 2D VLMs processing single images, 3D
LMMs exhibit inherent architectural redundancy due to the heterogeneous
mechanisms between spatial tokens and visual tokens. To address this challenge,
we propose AdaToken-3D, an adaptive spatial token optimization framework that
dynamically prunes redundant tokens through spatial contribution analysis. Our
method automatically tailors pruning strategies to different 3D LMM
architectures by quantifying token-level information flows via attention
pattern mining. Extensive experiments on LLaVA-3D (a 7B parameter 3D-LMM)
demonstrate that AdaToken-3D achieves 21\% faster inference speed and 63\%
FLOPs reduction while maintaining original task accuracy. Beyond efficiency
gains, this work systematically investigates redundancy patterns in multimodal
spatial information flows through quantitative token interaction analysis. Our
findings reveal that over 60\% of spatial tokens contribute minimally ($<$5\%)
to the final predictions, establishing theoretical foundations for efficient 3D
multimodal learning.

### Computer Science and Game Theory

### 1. [Improved Approximation Ratio for Strategyproof Facility Location on a Cycle](http://arxiv.org/pdf/2505.12943v1)

Authors: Krzysztof Rogowski, Marcin Dziubiński

We study the problem of design of strategyproof in expectation (SP)
mechanisms for facility location on a cycle, with the objective of minimizing
the sum of costs of $n$ agents. We show that there exists an SP mechanism that
attains an approximation ratio of $7/4$ with respect to the sum of costs of the
agents, thus improving the best known upper bound of $2-2/n$ in the cases of $n
\geq 5$. The mechanism obtaining the bound randomizes between two mechanisms
known in the literature: the Random Dictator (RD) and the Proportional Circle
Distance (PCD) mechanism of Meir (arXiv:1902.08070). To prove the result, we
propose a cycle-cutting technique that allows for estimating the problem on a
cycle by a problem on a line.

### 2. [Meta-rotations and the Structure of Stable Matchings in the Student Project Allocation Problem](http://arxiv.org/pdf/2505.13428v1)

Authors: Peace Ayegba, Sofiat Olaosebikan, David Manlove

We formally introduce and present the concept of meta-rotations as a tool for
navigating the lattice of stable matchings in the Student Project Allocation
problem with lecturer preferences over students (SPA-S). Building on the
structural result that the set of stable matchings in any SPA-S instance forms
a distributive lattice, we define meta-rotations for this setting and
demonstrate how they compactly encode transitions between matchings. Our
framework generalises the classical notion of rotations in bipartite settings
and provides a systematic way to traverse the lattice, thereby enabling
efficient enumeration of the set of stable matchings in any given SPA-S
instance.

### 3. [More Efforts Towards Fixed-Parameter Approximability of Multiwinner Rules](http://arxiv.org/pdf/2505.12699v1)

Authors: Sushmita Gupta, Pallavi Jain, Souvik Saha, Saket Saurabh, Anannya Upasana

Multiwinner Elections have emerged as a prominent area of research with
numerous practical applications. We contribute to this area by designing
parameterized approximation algorithms and also resolving an open question by
Yang and Wang [AAMAS'18]. More formally, given a set of candidates,
\mathcal{C}, a set of voters,\mathcal{V}, approving a subset of candidates
(called approval set of a voter), and an integer $k$, we consider the problem
of selecting a ``good'' committee using Thiele rules. This problem is
computationally challenging for most Thiele rules with monotone submodular
satisfaction functions, as there is no (1-\frac{1}{e}-\epsilon)\footnote{Here,
$e$ denotes the base of the natural logarithm.}-approximation algorithm in
f(k)(|\mathcal{C}| + |\mathcal{V}|)^{o(k)} time for any fixed $\epsilon > 0$
and any computable function $f$, and no {\sf PTAS} even when the length of
approval set is two. Skowron [WINE'16] designed an approximation scheme running
in FPT time parameterized by the combined parameter, size of the approval set
and $k$. In this paper, we consider a parameter $d+k$ (no $d$ voters approve
the same set of $d$ candidates), where $d$ is upper bounded by the size of the
approval set (thus, can be much smaller).
  With respect to this parameter, we design parameterized approximation
schemes, a lossy polynomial-time preprocessing method, and show that an extra
committee member suffices to achieve the desired score (i.e., $1$-additive
approximation). Additionally, we resolve an open question by Yang and
Wang~[AAMAS'18] regarding the fixed-parameter tractability of the problem under
the PAV rule with the total score as the parameter, demonstrating that it
admits an FPT algorithm.

### 4. [Computing the Schulze Method for Large-Scale Preference Data Sets](http://arxiv.org/pdf/2505.12976v1)

Authors: Theresa Csar, Martin Lackner, Reinhard Pichler

The Schulze method is a voting rule widely used in practice and enjoys many
positive axiomatic properties. While it is computable in polynomial time, its
straight-forward implementation does not scale well for large elections. In
this paper, we develop a highly optimised algorithm for computing the Schulze
method with Pregel, a framework for massively parallel computation of graph
problems, and demonstrate its applicability for large preference data sets. In
addition, our theoretic analysis shows that the Schulze method is indeed
particularly well-suited for parallel computation, in stark contrast to the
related ranked pairs method. More precisely we show that winner determination
subject to the Schulze method is NL-complete, whereas this problem is
P-complete for the ranked pairs method.

### 5. [The Hamiltonian of Poly-matrix Zero-sum Games](http://arxiv.org/pdf/2505.12609v1)

Authors: Toshihiro Ota, Yuma Fujimoto

Understanding a dynamical system fundamentally relies on establishing an
appropriate Hamiltonian function and elucidating its symmetries. By formulating
agents' strategies and cumulative payoffs as canonically conjugate variables,
we identify the Hamiltonian function that generates the dynamics of poly-matrix
zero-sum games. We reveal the symmetries of our Hamiltonian and derive the
associated conserved quantities, showing how the conservation of probability
and the invariance of the Fenchel coupling are intrinsically encoded within the
system. Furthermore, we propose the dissipation FTRL (DFTRL) dynamics by
introducing a perturbation that dissipates the Fenchel coupling, proving
convergence to the Nash equilibrium and linking DFTRL to last-iterate
convergent algorithms. Our results highlight the potential of Hamiltonian
dynamics in uncovering the structural properties of learning dynamics in games,
and pave the way for broader applications of Hamiltonian dynamics in game
theory and machine learning.

### Human-Computer Interaction

### 1. [Adapting to LLMs: How Insiders and Outsiders Reshape Scientific Knowledge Production](http://arxiv.org/pdf/2505.12666v1)

Authors: Huimin Xu, Houjiang Liu, Yan Leng, Ying Ding

CSCW has long examined how emerging technologies reshape the ways researchers
collaborate and produce knowledge, with scientific knowledge production as a
central area of focus. As AI becomes increasingly integrated into scientific
research, understanding how researchers adapt to it reveals timely
opportunities for CSCW research -- particularly in supporting new forms of
collaboration, knowledge practices, and infrastructure in AI-driven science.
  This study quantifies LLM impacts on scientific knowledge production based on
an evaluation workflow that combines an insider-outsider perspective with a
knowledge production framework. Our findings reveal how LLMs catalyze both
innovation and reorganization in scientific communities, offering insights into
the broader transformation of knowledge production in the age of generative AI
and sheds light on new research opportunities in CSCW.

### 2. [Beyond Individual UX: Defining Group Experience(GX) as a New Paradigm for Group-centered AI](http://arxiv.org/pdf/2505.12780v1)

Authors: Soohwan Lee, Seoyeong Hwang, Kyungho Lee

Recent advancements in HCI and AI have predominantly centered on individual
user experiences, often neglecting the emergent dynamics of group interactions.
This provocation introduces Group Experience(GX) to capture the collective
perceptual, emotional, and cognitive dimensions that arise when individuals
interact in cohesive groups. We challenge the conventional Human-centered AI
paradigm and propose Group-centered AI(GCAI) as a framework that actively
mediates group dynamics, amplifies diverse voices, and fosters ethical
collective decision-making. Drawing on social psychology, organizational
behavior, and group dynamics, we outline a group-centered design approach that
balances individual autonomy with collective interests while developing novel
evaluative metrics. Our analysis emphasizes rethinking traditional
methodologies that focus solely on individual outcomes and advocates for
innovative strategies to capture group collaboration. We call on researchers to
bridge the gap between micro-level experiences and macro-level impacts,
ultimately enriching and transforming collaborative human interactions.

### 3. [StudyAlign: A Software System for Conducting Web-Based User Studies with Functional Interactive Prototypes](http://arxiv.org/pdf/2505.13046v1)

Authors: Florian Lehmann, Daniel Buschek

Interactive systems are commonly prototyped as web applications. This
approach enables studies with functional prototypes on a large scale. However,
setting up these studies can be complex due to implementing experiment
procedures, integrating questionnaires, and data logging. To enable such user
studies, we developed the software system StudyAlign which offers: 1) a
frontend for participants, 2) an admin panel to manage studies, 3) the
possibility to integrate questionnaires, 4) a JavaScript library to integrate
data logging into prototypes, and 5) a backend server for persisting log data,
and serving logical functions via an API to the different parts of the system.
With our system, researchers can set up web-based experiments and focus on the
design and development of interactions and prototypes. Furthermore, our
systematic approach facilitates the replication of studies and reduces the
required effort to execute web-based user studies. We conclude with reflections
on using StudyAlign for conducting HCI studies online.

### 4. [Human Response to Decision Support in Face Matching: The Influence of Task Difficulty and Machine Accuracy](http://arxiv.org/pdf/2505.13218v1)

Authors: Marina Estévez-Almenzar, Ricardo Baeza-Yates, Carlos Castillo

Decision support systems enhanced by Artificial Intelligence (AI) are
increasingly being used in high-stakes scenarios where errors or biased
outcomes can have significant consequences. In this work, we explore the
conditions under which AI-based decision support systems affect the decision
accuracy of humans involved in face matching tasks. Previous work suggests that
this largely depends on various factors, such as the specific nature of the
task and how users perceive the quality of the decision support, among others.
Hence, we conduct extensive experiments to examine how both task difficulty and
the precision of the system influence human outcomes. Our results show a strong
influence of task difficulty, which not only makes humans less precise but also
less capable of determining whether the decision support system is yielding
accurate suggestions or not. This has implications for the design of decision
support systems, and calls for a careful examination of the context in which
they are deployed and on how they are perceived by users.

### 5. [Automated Bias Assessment in AI-Generated Educational Content Using CEAT Framework](http://arxiv.org/pdf/2505.12718v1)

Authors: Jingyang Peng, Wenyuan Shen, Jiarui Rao, Jionghao Lin

Recent advances in Generative Artificial Intelligence (GenAI) have
transformed educational content creation, particularly in developing tutor
training materials. However, biases embedded in AI-generated content--such as
gender, racial, or national stereotypes--raise significant ethical and
educational concerns. Despite the growing use of GenAI, systematic methods for
detecting and evaluating such biases in educational materials remain limited.
This study proposes an automated bias assessment approach that integrates the
Contextualized Embedding Association Test with a prompt-engineered word
extraction method within a Retrieval-Augmented Generation framework. We applied
this method to AI-generated texts used in tutor training lessons. Results show
a high alignment between the automated and manually curated word sets, with a
Pearson correlation coefficient of r = 0.993, indicating reliable and
consistent bias assessment. Our method reduces human subjectivity and enhances
fairness, scalability, and reproducibility in auditing GenAI-produced
educational content.

### 6. [CAIM: Development and Evaluation of a Cognitive AI Memory Framework for Long-Term Interaction with Intelligent Agents](http://arxiv.org/pdf/2505.13044v1)

Authors: Rebecca Westhäußer, Frederik Berenz, Wolfgang Minker, Sebastian Zepf

Large language models (LLMs) have advanced the field of artificial
intelligence (AI) and are a powerful enabler for interactive systems. However,
they still face challenges in long-term interactions that require adaptation
towards the user as well as contextual knowledge and understanding of the
ever-changing environment. To overcome these challenges, holistic memory
modeling is required to efficiently retrieve and store relevant information
across interaction sessions for suitable responses. Cognitive AI, which aims to
simulate the human thought process in a computerized model, highlights
interesting aspects, such as thoughts, memory mechanisms, and decision-making,
that can contribute towards improved memory modeling for LLMs. Inspired by
these cognitive AI principles, we propose our memory framework CAIM. CAIM
consists of three modules: 1.) The Memory Controller as the central decision
unit; 2.) the Memory Retrieval, which filters relevant data for interaction
upon request; and 3.) the Post-Thinking, which maintains the memory storage. We
compare CAIM against existing approaches, focusing on metrics such as retrieval
accuracy, response correctness, contextual coherence, and memory storage. The
results demonstrate that CAIM outperforms baseline frameworks across different
metrics, highlighting its context-awareness and potential to improve long-term
human-AI interactions.

### 7. [Agentic Publications: An LLM-Driven Framework for Interactive Scientific Publishing, Supplementing Traditional Papers with AI-Powered Knowledge Systems](http://arxiv.org/pdf/2505.13246v1)

Authors: Roberto Pugliese, George Kourousias, Francesco Venier, Grazia Garlatti Costa

The exponential growth of scientific literature presents significant
challenges for researchers navigating the complex knowledge landscape. We
propose "Agentic Publications", a novel LLM-driven framework complementing
traditional publishing by transforming papers into interactive knowledge
systems. Our architecture integrates structured data with unstructured content
through retrieval-augmented generation and multi-agent verification. The
framework offers interfaces for both humans and machines, combining narrative
explanations with machine-readable outputs while addressing ethical
considerations through automated validation and transparent governance. Key
features include continuous knowledge updates, automatic integration of new
findings, and customizable detail levels. Our proof-of-concept demonstrates
multilingual interaction, API accessibility, and structured knowledge
representation through vector databases, knowledge graphs, and verification
agents. This approach enhances scientific communication across disciplines,
improving efficiency and collaboration while preserving traditional publishing
pathways, particularly valuable for interdisciplinary fields where knowledge
integration remains challenging.

### 8. [How Adding Metacognitive Requirements in Support of AI Feedback in Practice Exams Transforms Student Learning Behaviors](http://arxiv.org/pdf/2505.13381v1)

Authors: Mak Ahmad, Prerna Ravi, David Karger, Marc Facciotti

Providing personalized, detailed feedback at scale in large undergraduate
STEM courses remains a persistent challenge. We present an empirically
evaluated practice exam system that integrates AI generated feedback with
targeted textbook references, deployed in a large introductory biology course.
Our system encourages metacognitive behavior by asking students to explain
their answers and declare their confidence. It uses OpenAI's GPT-4o to generate
personalized feedback based on this information, while directing them to
relevant textbook sections. Through interaction logs from consenting
participants across three midterms (541, 342, and 413 students respectively),
totaling 28,313 question-student interactions across 146 learning objectives,
along with 279 surveys and 23 interviews, we examined the system's impact on
learning outcomes and engagement. Across all midterms, feedback types showed no
statistically significant performance differences, though some trends suggested
potential benefits. The most substantial impact came from the required
confidence ratings and explanations, which students reported transferring to
their actual exam strategies. About 40 percent of students engaged with
textbook references when prompted by feedback -- far higher than traditional
reading rates. Survey data revealed high satisfaction (mean rating 4.1 of 5),
with 82.1 percent reporting increased confidence on practiced midterm topics,
and 73.4 percent indicating they could recall and apply specific concepts. Our
findings suggest that embedding structured reflection requirements may be more
impactful than sophisticated feedback mechanisms.

### 9. [What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma](http://arxiv.org/pdf/2505.12727v1)

Authors: Han Meng, Yancan Chen, Yunan Li, Yitian Yang, Jungup Lee, Renwen Zhang, Yi-Chieh Lee

Mental-health stigma remains a pervasive social problem that hampers
treatment-seeking and recovery. Existing resources for training neural models
to finely classify such stigma are limited, relying primarily on social-media
or synthetic data without theoretical underpinnings. To remedy this gap, we
present an expert-annotated, theory-informed corpus of human-chatbot
interviews, comprising 4,141 snippets from 684 participants with documented
socio-cultural backgrounds. Our experiments benchmark state-of-the-art neural
models and empirically unpack the challenges of stigma detection. This dataset
can facilitate research on computationally detecting, neutralizing, and
counteracting mental-health stigma.

### 10. [From Assistants to Adversaries: Exploring the Security Risks of Mobile LLM Agents](http://arxiv.org/pdf/2505.12981v2)

Authors: Liangxuan Wu, Chao Wang, Tianming Liu, Yanjie Zhao, Haoyu Wang

The growing adoption of large language models (LLMs) has led to a new
paradigm in mobile computing--LLM-powered mobile AI agents--capable of
decomposing and automating complex tasks directly on smartphones. However, the
security implications of these agents remain largely unexplored. In this paper,
we present the first comprehensive security analysis of mobile LLM agents,
encompassing three representative categories: System-level AI Agents developed
by original equipment manufacturers (e.g., YOYO Assistant), Third-party
Universal Agents (e.g., Zhipu AI AutoGLM), and Emerging Agent Frameworks (e.g.,
Alibaba Mobile Agent). We begin by analyzing the general workflow of mobile
agents and identifying security threats across three core capability
dimensions: language-based reasoning, GUI-based interaction, and system-level
execution. Our analysis reveals 11 distinct attack surfaces, all rooted in the
unique capabilities and interaction patterns of mobile LLM agents, and spanning
their entire operational lifecycle. To investigate these threats in practice,
we introduce AgentScan, a semi-automated security analysis framework that
systematically evaluates mobile LLM agents across all 11 attack scenarios.
Applying AgentScan to nine widely deployed agents, we uncover a concerning
trend: every agent is vulnerable to targeted attacks. In the most severe cases,
agents exhibit vulnerabilities across eight distinct attack vectors. These
attacks can cause behavioral deviations, privacy leakage, or even full
execution hijacking. Based on these findings, we propose a set of defensive
design principles and practical recommendations for building secure mobile LLM
agents. Our disclosures have received positive feedback from two major device
vendors. Overall, this work highlights the urgent need for standardized
security practices in the fast-evolving landscape of LLM-driven mobile
automation.

### Information Retrieval

### 1. [LLM-based Query Expansion Fails for Unfamiliar and Ambiguous Queries](http://arxiv.org/pdf/2505.12694v1)

Authors: Kenya Abe, Kunihiro Takeoka, Makoto P. Kato, Masafumi Oyamada

Query expansion (QE) enhances retrieval by incorporating relevant terms, with
large language models (LLMs) offering an effective alternative to traditional
rule-based and statistical methods. However, LLM-based QE suffers from a
fundamental limitation: it often fails to generate relevant knowledge,
degrading search performance. Prior studies have focused on hallucination, yet
its underlying cause--LLM knowledge deficiencies--remains underexplored. This
paper systematically examines two failure cases in LLM-based QE: (1) when the
LLM lacks query knowledge, leading to incorrect expansions, and (2) when the
query is ambiguous, causing biased refinements that narrow search coverage. We
conduct controlled experiments across multiple datasets, evaluating the effects
of knowledge and query ambiguity on retrieval performance using sparse and
dense retrieval models. Our results reveal that LLM-based QE can significantly
degrade the retrieval effectiveness when knowledge in the LLM is insufficient
or query ambiguity is high. We introduce a framework for evaluating QE under
these conditions, providing insights into the limitations of LLM-based
retrieval augmentation.

### 2. [Towards A Generalist Code Embedding Model Based On Massive Data Synthesis](http://arxiv.org/pdf/2505.12697v1)

Authors: Chaofan Li, Jianlyu Chen, Yingxia Shao, Defu Lian, Zheng Liu

Code embedding models attract increasing attention due to the widespread
popularity of retrieval-augmented generation (RAG) in software development.
These models are expected to capture the rich semantic relationships inherent
to code, which differ significantly from those found in text. However, existing
models remain severely limited due to the scarcity of high-quality training
data. In this work, we introduce \textbf{CodeR} (\underline{Code}
\underline{R}etrieval), a state-of-the-art embedding model for general-purpose
code retrieval. The superior performance of CodeR is built upon CodeR-Pile, a
large-scale synthetic dataset constructed under the DRU (Diversity,
Reliability, Usability) principle via a novel data synthesis pipeline. To
optimize training effectiveness, we propose Annealing, a curriculum learning
strategy that enables effective knowledge transfer across heterogeneous sources
of data. We evaluate CodeR based on 16 diverse code retrieval tasks, where it
significantly outperforms existing baselines and exhibits strong out-of-domain
generalization performance. We have publicly released our code and the
well-trained model to facilitate further research in this critical area.
https://github.com/FlagOpen/FlagEmbedding/tree/master/research/BGE_Coder.

### 3. [Think Before You Attribute: Improving the Performance of LLMs Attribution Systems](http://arxiv.org/pdf/2505.12621v1)

Authors: João Eduardo Batista, Emil Vatai, Mohamed Wahib

Large Language Models (LLMs) are increasingly applied in various science
domains, yet their broader adoption remains constrained by a critical
challenge: the lack of trustworthy, verifiable outputs. Current LLMs often
generate answers without reliable source attribution, or worse, with incorrect
attributions, posing a barrier to their use in scientific and high-stakes
settings, where traceability and accountability are non-negotiable. To be
reliable, attribution systems need high accuracy and retrieve data with short
lengths, i.e., attribute to a sentence within a document rather than a whole
document. We propose a sentence-level pre-attribution step for
Retrieve-Augmented Generation (RAG) systems that classify sentences into three
categories: not attributable, attributable to a single quote, and attributable
to multiple quotes. By separating sentences before attribution, a proper
attribution method can be selected for the type of sentence, or the attribution
can be skipped altogether. Our results indicate that classifiers are
well-suited for this task. In this work, we propose a pre-attribution step to
reduce the computational complexity of attribution, provide a clean version of
the HAGRID dataset, and provide an end-to-end attribution system that works out
of the box.

### 4. [Unlearning for Federated Online Learning to Rank: A Reproducibility Study](http://arxiv.org/pdf/2505.12791v1)

Authors: Yiling Tao, Shuyi Wang, Jiaxi Yang, Guido Zuccon

This paper reports on findings from a comparative study on the effectiveness
and efficiency of federated unlearning strategies within Federated Online
Learning to Rank (FOLTR), with specific attention to systematically analysing
the unlearning capabilities of methods in a verifiable manner.
  Federated approaches to ranking of search results have recently garnered
attention to address users privacy concerns. In FOLTR, privacy is safeguarded
by collaboratively training ranking models across decentralized data sources,
preserving individual user data while optimizing search results based on
implicit feedback, such as clicks.
  Recent legislation introduced across numerous countries is establishing the
so called "the right to be forgotten", according to which services based on
machine learning models like those in FOLTR should provide capabilities that
allow users to remove their own data from those used to train models. This has
sparked the development of unlearning methods, along with evaluation practices
to measure whether unlearning of a user data successfully occurred. Current
evaluation practices are however often controversial, necessitating the use of
multiple metrics for a more comprehensive assessment -- but previous proposals
of unlearning methods only used single evaluation metrics.
  This paper addresses this limitation: our study rigorously assesses the
effectiveness of unlearning strategies in managing both under-unlearning and
over-unlearning scenarios using adapted, and newly proposed evaluation metrics.
Thanks to our detailed analysis, we uncover the strengths and limitations of
five unlearning strategies, offering valuable insights into optimizing
federated unlearning to balance data privacy and system performance within
FOLTR. We publicly release our code and complete results at
https://github.com/Iris1026/Unlearning-for-FOLTR.git.

### 5. [Optimizing Retrieval Augmented Generation for Object Constraint Language](http://arxiv.org/pdf/2505.13129v1)

Authors: Kevin Chenhao Li, Vahid Zolfaghari, Nenad Petrovic, Fengjunjie Pan, Alois Knoll

The Object Constraint Language (OCL) is essential for defining precise
constraints within Model-Based Systems Engineering (MBSE). However, manually
writing OCL rules is complex and time-consuming. This study explores the
optimization of Retrieval-Augmented Generation (RAG) for automating OCL rule
generation, focusing on the impact of different retrieval strategies. We
evaluate three retrieval approaches $\unicode{x2013}$ BM25 (lexical-based),
BERT-based (semantic retrieval), and SPLADE (sparse-vector retrieval)
$\unicode{x2013}$ analyzing their effectiveness in providing relevant context
for a large language model.
  To further assess our approach, we compare and benchmark our
retrieval-optimized generation results against PathOCL, a state-of-the-art
graph-based method. We directly compare BM25, BERT, and SPLADE retrieval
methods with PathOCL to understand how different retrieval methods perform for
a unified evaluation framework. Our experimental results, focusing on
retrieval-augmented generation, indicate that while retrieval can enhance
generation accuracy, its effectiveness depends on the retrieval method and the
number of retrieved chunks (k). BM25 underperforms the baseline, whereas
semantic approaches (BERT and SPLADE) achieve better results, with SPLADE
performing best at lower k values. However, excessive retrieval with high k
parameter can lead to retrieving irrelevant chunks which degrades model
performance. Our findings highlight the importance of optimizing retrieval
configurations to balance context relevance and output consistency. This
research provides insights into improving OCL rule generation using RAG and
underscores the need for tailoring retrieval.

### 6. [GMM-Based Comprehensive Feature Extraction and Relative Distance Preservation For Few-Shot Cross-Modal Retrieval](http://arxiv.org/pdf/2505.13306v1)

Authors: Chengsong Sun, Weiping Li, Xiang Li, Yuankun Liu, Lianlei Shan

Few-shot cross-modal retrieval focuses on learning cross-modal
representations with limited training samples, enabling the model to handle
unseen classes during inference. Unlike traditional cross-modal retrieval
tasks, which assume that both training and testing data share the same class
distribution, few-shot retrieval involves data with sparse representations
across modalities. Existing methods often fail to adequately model the
multi-peak distribution of few-shot cross-modal data, resulting in two main
biases in the latent semantic space: intra-modal bias, where sparse samples
fail to capture intra-class diversity, and inter-modal bias, where
misalignments between image and text distributions exacerbate the semantic gap.
These biases hinder retrieval accuracy. To address these issues, we propose a
novel method, GCRDP, for few-shot cross-modal retrieval. This approach
effectively captures the complex multi-peak distribution of data using a
Gaussian Mixture Model (GMM) and incorporates a multi-positive sample
contrastive learning mechanism for comprehensive feature modeling.
Additionally, we introduce a new strategy for cross-modal semantic alignment,
which constrains the relative distances between image and text feature
distributions, thereby improving the accuracy of cross-modal representations.
We validate our approach through extensive experiments on four benchmark
datasets, demonstrating superior performance over six state-of-the-art methods.

### 7. [CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming](http://arxiv.org/pdf/2505.12925v1)

Authors: Han Deng, Yuan Meng, Shixiang Tang, Wanli Ouyang, Xinzhu Ma

Competitive programming benchmarks are widely used in scenarios such as
programming contests and large language model assessments. However, the growing
presence of duplicate or highly similar problems raises concerns not only about
competition fairness, but also about the validity of competitive programming as
a benchmark for model evaluation. In this paper, we propose a new problem --
similar question retrieval -- to address this issue. Due to the lack of both
data and models, solving this problem is challenging. To this end, we introduce
CPRet, a retrieval-oriented benchmark suite for competitive programming,
covering four retrieval tasks: two code-centric (i.e., Text-to-Code and
Code-to-Code) and two newly proposed problem-centric tasks (i.e.,
Problem-to-Duplicate and Simplified-to-Full), built from a combination of
automatically crawled problem-solution data and manually curated annotations.
Our contribution includes both high-quality training data and temporally
separated test sets for reliable evaluation. In addition, we develop two
task-specialized retrievers based on this dataset: CPRetriever-Code, trained
with a novel Group-InfoNCE loss for problem-code alignment, and
CPRetriever-Prob, fine-tuned for identifying problem-level similarity. Both
models achieve strong results and are open-sourced for local use. Finally, we
analyze LiveCodeBench and find that high-similarity problems inflate model pass
rates and reduce differentiation, underscoring the need for similarity-aware
evaluation in future benchmarks.
  Code and data are available at: https://github.com/coldchair/CPRet

### 8. [AdaToken-3D: Dynamic Spatial Gating for Efficient 3D Large Multimodal-Models Reasoning](http://arxiv.org/pdf/2505.12782v1)

Authors: Kai Zhang, Xingyu Chen, Xiaofeng Zhang

Large Multimodal Models (LMMs) have become a pivotal research focus in deep
learning, demonstrating remarkable capabilities in 3D scene understanding.
However, current 3D LMMs employing thousands of spatial tokens for multimodal
reasoning suffer from critical inefficiencies: excessive computational overhead
and redundant information flows. Unlike 2D VLMs processing single images, 3D
LMMs exhibit inherent architectural redundancy due to the heterogeneous
mechanisms between spatial tokens and visual tokens. To address this challenge,
we propose AdaToken-3D, an adaptive spatial token optimization framework that
dynamically prunes redundant tokens through spatial contribution analysis. Our
method automatically tailors pruning strategies to different 3D LMM
architectures by quantifying token-level information flows via attention
pattern mining. Extensive experiments on LLaVA-3D (a 7B parameter 3D-LMM)
demonstrate that AdaToken-3D achieves 21\% faster inference speed and 63\%
FLOPs reduction while maintaining original task accuracy. Beyond efficiency
gains, this work systematically investigates redundancy patterns in multimodal
spatial information flows through quantitative token interaction analysis. Our
findings reveal that over 60\% of spatial tokens contribute minimally ($<$5\%)
to the final predictions, establishing theoretical foundations for efficient 3D
multimodal learning.

### Machine Learning

### 1. [A Few Large Shifts: Layer-Inconsistency Based Minimal Overhead Adversarial Example Detection](http://arxiv.org/pdf/2505.12586v2)

Authors: Sanggeon Yun, Ryozo Masukawa, Hyunwoo Oh, Nathaniel D. Bastian, Mohsen Imani

Deep neural networks (DNNs) are highly susceptible to adversarial
examples--subtle, imperceptible perturbations that can lead to incorrect
predictions. While detection-based defenses offer a practical alternative to
adversarial training, many existing methods depend on external models, complex
architectures, heavy augmentations, or adversarial data, limiting their
efficiency and generalizability. We introduce a lightweight, plug-in detection
framework that leverages internal layer-wise inconsistencies within the target
model itself, requiring only benign data for calibration. Our approach is
grounded in the A Few Large Shifts Assumption, which posits that adversarial
perturbations typically induce large representation shifts in a small subset of
layers. Building on this, we propose two complementary strategies--Recovery
Testing (RT) and Logit-layer Testing (LT)--to expose internal disruptions
caused by adversaries. Evaluated on CIFAR-10, CIFAR-100, and ImageNet under
both standard and adaptive threat models, our method achieves state-of-the-art
detection performance with negligible computational overhead and no compromise
to clean accuracy.

### 2. [Rethinking Predictive Modeling for LLM Routing: When Simple kNN Beats Complex Learned Routers](http://arxiv.org/pdf/2505.12601v1)

Authors: Yang Li

As large language models (LLMs) grow in scale and specialization,
routing--selecting the best model for a given input--has become essential for
efficient and effective deployment. While recent methods rely on complex
learned routing strategies, their dependence on disparate training data and
evaluation setups makes comparison and generalization difficult. In this work,
we revisit LLM routing through the lens of simplicity. We show that a
well-tuned k-Nearest Neighbors (kNN) approach not only matches but often
outperforms state-of-the-art learned routers across diverse tasks. To support
systematic evaluation, we introduce a suite of standardized routing benchmarks
spanning instruction-following, question-answering, and reasoning tasks, as
well as the first multi-modal routing dataset involving visual inputs. Our
findings reveal that the locality properties of model performance in embedding
space enable simple non-parametric methods to achieve strong routing decisions
with lower sample complexity than parametric approaches. This challenges the
prevailing trend toward sophisticated architectures and highlights the
importance of thoroughly evaluating simple baselines before investing in
complex solutions. To support reproducibility and further exploration, we will
release all benchmarks and code upon publication.

### 3. [Action-Dependent Optimality-Preserving Reward Shaping](http://arxiv.org/pdf/2505.12611v1)

Authors: Grant C. Forbes, Jianxun Wang, Leonardo Villalobos-Arias, Arnav Jhala, David L. Roberts

Recent RL research has utilized reward shaping--particularly complex shaping
rewards such as intrinsic motivation (IM)--to encourage agent exploration in
sparse-reward environments. While often effective, ``reward hacking'' can lead
to the shaping reward being optimized at the expense of the extrinsic reward,
resulting in a suboptimal policy. Potential-Based Reward Shaping (PBRS)
techniques such as Generalized Reward Matching (GRM) and Policy-Invariant
Explicit Shaping (PIES) have mitigated this. These methods allow for
implementing IM without altering optimal policies. In this work we show that
they are effectively unsuitable for complex, exploration-heavy environments
with long-duration episodes. To remedy this, we introduce Action-Dependent
Optimality Preserving Shaping (ADOPS), a method of converting intrinsic rewards
to an optimality-preserving form that allows agents to utilize IM more
effectively in the extremely sparse environment of Montezuma's Revenge. We also
prove ADOPS accommodates reward shaping functions that cannot be written in a
potential-based form: while PBRS-based methods require the cumulative
discounted intrinsic return be independent of actions, ADOPS allows for
intrinsic cumulative returns to be dependent on agents' actions while still
preserving the optimal policy set. We show how action-dependence enables
ADOPS's to preserve optimality while learning in complex, sparse-reward
environments where other methods struggle.

### 4. [Adaptive Graph Unlearning](http://arxiv.org/pdf/2505.12614v1)

Authors: Pengfei Ding, Yan Wang, Guanfeng Liu, Jiajie Zhu

Graph unlearning, which deletes graph elements such as nodes and edges from
trained graph neural networks (GNNs), is crucial for real-world applications
where graph data may contain outdated, inaccurate, or privacy-sensitive
information. However, existing methods often suffer from (1) incomplete or over
unlearning due to neglecting the distinct objectives of different unlearning
tasks, and (2) inaccurate identification of neighbors affected by deleted
elements across various GNN architectures. To address these limitations, we
propose AGU, a novel Adaptive Graph Unlearning framework that flexibly adapts
to diverse unlearning tasks and GNN architectures. AGU ensures the complete
forgetting of deleted elements while preserving the integrity of the remaining
graph. It also accurately identifies affected neighbors for each GNN
architecture and prioritizes important ones to enhance unlearning performance.
Extensive experiments on seven real-world graphs demonstrate that AGU
outperforms existing methods in terms of effectiveness, efficiency, and
unlearning capability.

### 5. [Dual-Agent Reinforcement Learning for Automated Feature Generation](http://arxiv.org/pdf/2505.12628v1)

Authors: Wanfu Gao, Zengyao Man, Hanlin Pan, Kunpeng Liu

Feature generation involves creating new features from raw data to capture
complex relationships among the original features, improving model robustness
and machine learning performance. Current methods using reinforcement learning
for feature generation have made feature exploration more flexible and
efficient. However, several challenges remain: first, during feature expansion,
a large number of redundant features are generated. When removing them, current
methods only retain the best features each round, neglecting those that perform
poorly initially but could improve later. Second, the state representation used
by current methods fails to fully capture complex feature relationships. Third,
there are significant differences between discrete and continuous features in
tabular data, requiring different operations for each type. To address these
challenges, we propose a novel dual-agent reinforcement learning method for
feature generation. Two agents are designed: the first generates new features,
and the second determines whether they should be preserved. A self-attention
mechanism enhances state representation, and diverse operations distinguish
interactions between discrete and continuous features. The experimental results
on multiple datasets demonstrate that the proposed method is effective. The
code is available at https://github.com/extess0/DARL.

### 6. [Spiking Neural Network: a low power solution for physical layer authentication](http://arxiv.org/pdf/2505.12647v1)

Authors: Jung Hoon Lee, Sujith Vijayan

Deep learning (DL) is a powerful tool that can solve complex problems, and
thus, it seems natural to assume that DL can be used to enhance the security of
wireless communication. However, deploying DL models to edge devices in
wireless networks is challenging, as they require significant amounts of
computing and power resources. Notably, Spiking Neural Networks (SNNs) are
known to be efficient in terms of power consumption, meaning they can be an
alternative platform for DL models for edge devices. In this study, we ask if
SNNs can be used in physical layer authentication. Our evaluation suggests that
SNNs can learn unique physical properties (i.e., `fingerprints') of RF
transmitters and use them to identify individual devices. Furthermore, we find
that SNNs are also vulnerable to adversarial attacks and that an autoencoder
can be used clean out adversarial perturbations to harden SNNs against them.

### 7. [TransferTraj: A Vehicle Trajectory Learning Model for Region and Task Transferability](http://arxiv.org/pdf/2505.12672v1)

Authors: Tonglong Wei, Yan Lin, Zeyu Zhou, Haomin Wen, Jilin Hu, Shengnan Guo, Youfang Lin, Gao Cong, Huaiyu Wan

Vehicle GPS trajectories provide valuable movement information that supports
various downstream tasks and applications. A desirable trajectory learning
model should be able to transfer across regions and tasks without retraining,
avoiding the need to maintain multiple specialized models and subpar
performance with limited training data. However, each region has its unique
spatial features and contexts, which are reflected in vehicle movement patterns
and difficult to generalize. Additionally, transferring across different tasks
faces technical challenges due to the varying input-output structures required
for each task. Existing efforts towards transferability primarily involve
learning embedding vectors for trajectories, which perform poorly in region
transfer and require retraining of prediction modules for task transfer.
  To address these challenges, we propose TransferTraj, a vehicle GPS
trajectory learning model that excels in both region and task transferability.
For region transferability, we introduce RTTE as the main learnable module
within TransferTraj. It integrates spatial, temporal, POI, and road network
modalities of trajectories to effectively manage variations in spatial context
distribution across regions. It also introduces a TRIE module for incorporating
relative information of spatial features and a spatial context MoE module for
handling movement patterns in diverse contexts. For task transferability, we
propose a task-transferable input-output scheme that unifies the input-output
structure of different tasks into the masking and recovery of modalities and
trajectory points. This approach allows TransferTraj to be pre-trained once and
transferred to different tasks without retraining. Extensive experiments on
three real-world vehicle trajectory datasets under task transfer, zero-shot,
and few-shot region transfer, validating TransferTraj's effectiveness.

### 8. [RoFL: Robust Fingerprinting of Language Models](http://arxiv.org/pdf/2505.12682v1)

Authors: Yun-Yun Tsai, Chuan Guo, Junfeng Yang, Laurens van der Maaten

AI developers are releasing large language models (LLMs) under a variety of
different licenses. Many of these licenses restrict the ways in which the
models or their outputs may be used. This raises the question how license
violations may be recognized. In particular, how can we identify that an API or
product uses (an adapted version of) a particular LLM? We present a new method
that enable model developers to perform such identification via fingerprints:
statistical patterns that are unique to the developer's model and robust to
common alterations of that model. Our method permits model identification in a
black-box setting using a limited number of queries, enabling identification of
models that can only be accessed via an API or product. The fingerprints are
non-invasive: our method does not require any changes to the model during
training, hence by design, it does not impact model quality. Empirically, we
find our method provides a high degree of robustness to common changes in the
model or inference settings. In our experiments, it substantially outperforms
prior art, including invasive methods that explicitly train watermarks into the
model.

### 9. [DimGrow: Memory-Efficient Field-level Embedding Dimension Search](http://arxiv.org/pdf/2505.12683v1)

Authors: Yihong Huang, Chen Chu

Key feature fields need bigger embedding dimensionality, others need smaller.
This demands automated dimension allocation. Existing approaches, such as
pruning or Neural Architecture Search (NAS), require training a
memory-intensive SuperNet that enumerates all possible dimension combinations,
which is infeasible for large feature spaces. We propose DimGrow, a lightweight
approach that eliminates the SuperNet requirement. Starting training model from
one dimension per feature field, DimGrow can progressively expand/shrink
dimensions via importance scoring. Dimensions grow only when their importance
consistently exceed a threshold, ensuring memory efficiency. Experiments on
three recommendation datasets verify the effectiveness of DimGrow while it
reduces training memory compared to SuperNet-based methods.

### 10. [Pave Your Own Path: Graph Gradual Domain Adaptation on Fused Gromov-Wasserstein Geodesics](http://arxiv.org/pdf/2505.12709v1)

Authors: Zhichen Zeng, Ruizhong Qiu, Wenxuan Bao, Tianxin Wei, Xiao Lin, Yuchen Yan, Tarek F. Abdelzaher, Jiawei Han, Hanghang Tong

Graph neural networks, despite their impressive performance, are highly
vulnerable to distribution shifts on graphs. Existing graph domain adaptation
(graph DA) methods often implicitly assume a \textit{mild} shift between source
and target graphs, limiting their applicability to real-world scenarios with
\textit{large} shifts. Gradual domain adaptation (GDA) has emerged as a
promising approach for addressing large shifts by gradually adapting the source
model to the target domain via a path of unlabeled intermediate domains.
Existing GDA methods exclusively focus on independent and identically
distributed (IID) data with a predefined path, leaving their extension to
\textit{non-IID graphs without a given path} an open challenge. To bridge this
gap, we present Gadget, the first GDA framework for non-IID graph data. First
(\textit{theoretical foundation}), the Fused Gromov-Wasserstein (FGW) distance
is adopted as the domain discrepancy for non-IID graphs, based on which, we
derive an error bound revealing that the target domain error is proportional to
the length of the path. Second (\textit{optimal path}), guided by the error
bound, we identify the FGW geodesic as the optimal path, which can be
efficiently generated by our proposed algorithm. The generated path can be
seamlessly integrated with existing graph DA methods to handle large shifts on
graphs, improving state-of-the-art graph DA methods by up to 6.8\% in node
classification accuracy on real-world datasets.

### Neural and Evolutionary Computing

### 1. [Efficient Heuristics Generation for Solving Combinatorial Optimization Problems Using Large Language Models](http://arxiv.org/pdf/2505.12627v1)

Authors: Xuan Wu, Di Wang, Chunguo Wu, Lijie Wen, Chunyan Miao, Yubin Xiao, You Zhou

Recent studies exploited Large Language Models (LLMs) to autonomously
generate heuristics for solving Combinatorial Optimization Problems (COPs), by
prompting LLMs to first provide search directions and then derive heuristics
accordingly. However, the absence of task-specific knowledge in prompts often
leads LLMs to provide unspecific search directions, obstructing the derivation
of well-performing heuristics. Moreover, evaluating the derived heuristics
remains resource-intensive, especially for those semantically equivalent ones,
often requiring omissible resource expenditure. To enable LLMs to provide
specific search directions, we propose the Hercules algorithm, which leverages
our designed Core Abstraction Prompting (CAP) method to abstract the core
components from elite heuristics and incorporate them as prior knowledge in
prompts. We theoretically prove the effectiveness of CAP in reducing
unspecificity and provide empirical results in this work. To reduce computing
resources required for evaluating the derived heuristics, we propose few-shot
Performance Prediction Prompting (PPP), a first-of-its-kind method for the
Heuristic Generation (HG) task. PPP leverages LLMs to predict the fitness
values of newly derived heuristics by analyzing their semantic similarity to
previously evaluated ones. We further develop two tailored mechanisms for PPP
to enhance predictive accuracy and determine unreliable predictions,
respectively. The use of PPP makes Hercules more resource-efficient and we name
this variant Hercules-P. Extensive experiments across four HG tasks, five COPs,
and eight LLMs demonstrate that Hercules outperforms the state-of-the-art
LLM-based HG algorithms, while Hercules-P excels at minimizing required
computing resources. In addition, we illustrate the effectiveness of CAP, PPP,
and the other proposed mechanisms by conducting relevant ablation studies.

### 2. [Hierarchical Representations for Evolving Acyclic Vector Autoregressions (HEAVe)](http://arxiv.org/pdf/2505.12806v1)

Authors: Cameron Cornell, Lewis Mitchell, Matthew Roughan

Causal networks offer an intuitive framework to understand influence
structures within time series systems. However, the presence of cycles can
obscure dynamic relationships and hinder hierarchical analysis. These networks
are typically identified through multivariate predictive modelling, but
enforcing acyclic constraints significantly increases computational and
analytical complexity. Despite recent advances, there remains a lack of simple,
flexible approaches that are easily tailorable to specific problem instances.
We propose an evolutionary approach to fitting acyclic vector autoregressive
processes and introduces a novel hierarchical representation that directly
models structural elements within a time series system. On simulated datasets,
our model retains most of the predictive accuracy of unconstrained models and
outperforms permutation-based alternatives. When applied to a dataset of 100
cryptocurrency return series, our method generates acyclic causal networks
capturing key structural properties of the unconstrained model. The acyclic
networks are approximately sub-graphs of the unconstrained networks, and most
of the removed links originate from low-influence nodes. Given the high levels
of feature preservation, we conclude that this cryptocurrency price system
functions largely hierarchically. Our findings demonstrate a flexible,
intuitive approach for identifying hierarchical causal networks in time series
systems, with broad applications to fields like econometrics and social network
analysis.

### 3. [Multi-parameter Control for the (1+($λ$,$λ$))-GA on OneMax via Deep Reinforcement Learning](http://arxiv.org/pdf/2505.12982v1)

Authors: Tai Nguyen, Phong Le, Carola Doerr, Nguyen Dang

It is well known that evolutionary algorithms can benefit from dynamic
choices of the key parameters that control their behavior, to adjust their
search strategy to the different stages of the optimization process. A
prominent example where dynamic parameter choices have shown a provable
super-constant speed-up is the $(1+(\lambda,\lambda))$ Genetic Algorithm
optimizing the OneMax function. While optimal parameter control policies result
in linear expected running times, this is not possible with static parameter
choices. This result has spurred a lot of interest in parameter control
policies. However, many works, in particular theoretical running time analyses,
focus on controlling one single parameter. Deriving policies for controlling
multiple parameters remains very challenging. In this work we reconsider the
problem of the $(1+(\lambda,\lambda))$ Genetic Algorithm optimizing OneMax. We
decouple its four main parameters and investigate how well state-of-the-art
deep reinforcement learning techniques can approximate good control policies.
We show that although making deep reinforcement learning learn effectively is a
challenging task, once it works, it is very powerful and is able to find
policies that outperform all previously known control policies on the same
benchmark. Based on the results found through reinforcement learning, we derive
a simple control policy that consistently outperforms the default
theory-recommended setting by $27\%$ and the irace-tuned policy, the strongest
existing control policy on this benchmark, by $13\%$, for all tested problem
sizes up to $40{,}000$.

### 4. [Recombinant dynamical systems](http://arxiv.org/pdf/2505.13409v1)

Authors: Saul Kato

We describe a connectionist model that attempts to capture a notion of
experience-based problem solving or task learning, whereby solutions to newly
encountered problems are composed from remembered solutions to prior problems.
We apply this model to the computational problem of \emph{efficient sequence
generation}, a problem for which there is no obvious gradient descent
procedure, and for which not all posable problem instances are solvable.
Empirical tests show promising evidence of utility.

### 5. [A Path to Universal Neural Cellular Automata](http://arxiv.org/pdf/2505.13058v1)

Authors: Gabriel Béna, Maxence Faldor, Dan F. M. Goodman, Antoine Cully

Cellular automata have long been celebrated for their ability to generate
complex behaviors from simple, local rules, with well-known discrete models
like Conway's Game of Life proven capable of universal computation. Recent
advancements have extended cellular automata into continuous domains, raising
the question of whether these systems retain the capacity for universal
computation. In parallel, neural cellular automata have emerged as a powerful
paradigm where rules are learned via gradient descent rather than manually
designed. This work explores the potential of neural cellular automata to
develop a continuous Universal Cellular Automaton through training by gradient
descent. We introduce a cellular automaton model, objective functions and
training strategies to guide neural cellular automata toward universal
computation in a continuous setting. Our experiments demonstrate the successful
training of fundamental computational primitives - such as matrix
multiplication and transposition - culminating in the emulation of a neural
network solving the MNIST digit classification task directly within the
cellular automata state. These results represent a foundational step toward
realizing analog general-purpose computers, with implications for understanding
universal computation in continuous dynamics and advancing the automated
discovery of complex cellular automata behaviors via machine learning.

### 6. [Stochastic Orthogonal Regularization for deep projective priors](http://arxiv.org/pdf/2505.13078v1)

Authors: Ali Joundi, Yann Traonmilin, Alasdair Newson

Many crucial tasks of image processing and computer vision are formulated as
inverse problems. Thus, it is of great importance to design fast and robust
algorithms to solve these problems. In this paper, we focus on generalized
projected gradient descent (GPGD) algorithms where generalized projections are
realized with learned neural networks and provide state-of-the-art results for
imaging inverse problems. Indeed, neural networks allow for projections onto
unknown low-dimensional sets that model complex data, such as images. We call
these projections deep projective priors. In generic settings, when the
orthogonal projection onto a lowdimensional model set is used, it has been
shown, under a restricted isometry assumption, that the corresponding
orthogonal PGD converges with a linear rate, yielding near-optimal convergence
(within the class of GPGD methods) in the classical case of sparse recovery.
However, for deep projective priors trained with classical mean squared error
losses, there is little guarantee that the hypotheses for linear convergence
are satisfied. In this paper, we propose a stochastic orthogonal regularization
of the training loss for deep projective priors. This regularization is
motivated by our theoretical results: a sufficiently good approximation of the
orthogonal projection guarantees linear stable recovery with performance close
to orthogonal PGD. We show experimentally, using two different deep projective
priors (based on autoencoders and on denoising networks), that our stochastic
orthogonal regularization yields projections that improve convergence speed and
robustness of GPGD in challenging inverse problem settings, in accordance with
our theoretical findings.

### 7. [$μ$PC: Scaling Predictive Coding to 100+ Layer Networks](http://arxiv.org/pdf/2505.13124v1)

Authors: Francesco Innocenti, El Mehdi Achour, Christopher L. Buckley

The biological implausibility of backpropagation (BP) has motivated many
alternative, brain-inspired algorithms that attempt to rely only on local
information, such as predictive coding (PC) and equilibrium propagation.
However, these algorithms have notoriously struggled to train very deep
networks, preventing them from competing with BP in large-scale settings.
Indeed, scaling PC networks (PCNs) has recently been posed as a challenge for
the community (Pinchetti et al., 2024). Here, we show that 100+ layer PCNs can
be trained reliably using a Depth-$\mu$P parameterisation (Yang et al., 2023;
Bordelon et al., 2023) which we call "$\mu$PC". Through an extensive analysis
of the scaling behaviour of PCNs, we reveal several pathologies that make
standard PCNs difficult to train at large depths. We then show that, despite
addressing only some of these instabilities, $\mu$PC allows stable training of
very deep (up to 128-layer) residual networks on simple classification tasks
with competitive performance and little tuning compared to current benchmarks.
Moreover, $\mu$PC enables zero-shot transfer of both weight and activity
learning rates across widths and depths. Our results have implications for
other local algorithms and could be extended to convolutional and transformer
architectures. Code for $\mu$PC is made available as part of a JAX library for
PCNs at https://github.com/thebuckleylab/jpc (Innocenti et al., 2024).

### 8. [CALM-PDE: Continuous and Adaptive Convolutions for Latent Space Modeling of Time-dependent PDEs](http://arxiv.org/pdf/2505.12944v1)

Authors: Jan Hagnberger, Daniel Musekamp, Mathias Niepert

Solving time-dependent Partial Differential Equations (PDEs) using a densely
discretized spatial domain is a fundamental problem in various scientific and
engineering disciplines, including modeling climate phenomena and fluid
dynamics. However, performing these computations directly in the physical space
often incurs significant computational costs. To address this issue, several
neural surrogate models have been developed that operate in a compressed latent
space to solve the PDE. While these approaches reduce computational complexity,
they often use Transformer-based attention mechanisms to handle irregularly
sampled domains, resulting in increased memory consumption. In contrast,
convolutional neural networks allow memory-efficient encoding and decoding but
are limited to regular discretizations. Motivated by these considerations, we
propose CALM-PDE, a model class that efficiently solves arbitrarily discretized
PDEs in a compressed latent space. We introduce a novel continuous
convolution-based encoder-decoder architecture that uses an
epsilon-neighborhood-constrained kernel and learns to apply the convolution
operator to adaptive and optimized query points. We demonstrate the
effectiveness of CALM-PDE on a diverse set of PDEs with both regularly and
irregularly sampled spatial domains. CALM-PDE is competitive with or
outperforms existing baseline methods while offering significant improvements
in memory and inference time efficiency compared to Transformer-based methods.

### 9. [Net-Zero: A Comparative Study on Neural Network Design for Climate-Economic PDEs Under Uncertainty](http://arxiv.org/pdf/2505.13264v1)

Authors: Carlos Rodriguez-Pardo, Louis Daumas, Leonardo Chiani, Massimo Tavoni

Climate-economic modeling under uncertainty presents significant
computational challenges that may limit policymakers' ability to address
climate change effectively. This paper explores neural network-based approaches
for solving high-dimensional optimal control problems arising from models that
incorporate ambiguity aversion in climate mitigation decisions. We develop a
continuous-time endogenous-growth economic model that accounts for multiple
mitigation pathways, including emission-free capital and carbon intensity
reductions. Given the inherent complexity and high dimensionality of these
models, traditional numerical methods become computationally intractable. We
benchmark several neural network architectures against finite-difference
generated solutions, evaluating their ability to capture the dynamic
interactions between uncertainty, technology transitions, and optimal climate
policy. Our findings demonstrate that appropriate neural architecture selection
significantly impacts both solution accuracy and computational efficiency when
modeling climate-economic systems under uncertainty. These methodological
advances enable more sophisticated modeling of climate policy decisions,
allowing for better representation of technology transitions and
uncertainty-critical elements for developing effective mitigation strategies in
the face of climate change.

### Networking and Internet Architecture

### 1. [Forewarned is Forearmed: A Survey on Large Language Model-based Agents in Autonomous Cyberattacks](http://arxiv.org/pdf/2505.12786v1)

Authors: Minrui Xu, Jiani Fan, Xinyu Huang, Conghao Zhou, Jiawen Kang, Dusit Niyato, Shiwen Mao, Zhu Han, Xuemin, Shen, Kwok-Yan Lam

With the continuous evolution of Large Language Models (LLMs), LLM-based
agents have advanced beyond passive chatbots to become autonomous cyber
entities capable of performing complex tasks, including web browsing, malicious
code and deceptive content generation, and decision-making. By significantly
reducing the time, expertise, and resources, AI-assisted cyberattacks
orchestrated by LLM-based agents have led to a phenomenon termed Cyber Threat
Inflation, characterized by a significant reduction in attack costs and a
tremendous increase in attack scale. To provide actionable defensive insights,
in this survey, we focus on the potential cyber threats posed by LLM-based
agents across diverse network systems. Firstly, we present the capabilities of
LLM-based cyberattack agents, which include executing autonomous attack
strategies, comprising scouting, memory, reasoning, and action, and
facilitating collaborative operations with other agents or human operators.
Building on these capabilities, we examine common cyberattacks initiated by
LLM-based agents and compare their effectiveness across different types of
networks, including static, mobile, and infrastructure-free paradigms.
Moreover, we analyze threat bottlenecks of LLM-based agents across different
network infrastructures and review their defense methods. Due to operational
imbalances, existing defense methods are inadequate against autonomous
cyberattacks. Finally, we outline future research directions and potential
defensive strategies for legacy network systems.

### 2. [Confidence-Regulated Generative Diffusion Models for Reliable AI Agent Migration in Vehicular Metaverses](http://arxiv.org/pdf/2505.12710v1)

Authors: Yingkai Kang, Jiawen Kang, Jinbo Wen, Tao Zhang, Zhaohui Yang, Dusit Niyato, Yan Zhang

Vehicular metaverses are an emerging paradigm that merges intelligent
transportation systems with virtual spaces, leveraging advanced digital twin
and Artificial Intelligence (AI) technologies to seamlessly integrate vehicles,
users, and digital environments. In this paradigm, vehicular AI agents are
endowed with environment perception, decision-making, and action execution
capabilities, enabling real-time processing and analysis of multi-modal data to
provide users with customized interactive services. Since vehicular AI agents
require substantial resources for real-time decision-making, given vehicle
mobility and network dynamics conditions, the AI agents are deployed in
RoadSide Units (RSUs) with sufficient resources and dynamically migrated among
them. However, AI agent migration requires frequent data exchanges, which may
expose vehicular metaverses to potential cyber attacks. To this end, we propose
a reliable vehicular AI agent migration framework, achieving reliable dynamic
migration and efficient resource scheduling through cooperation between
vehicles and RSUs. Additionally, we design a trust evaluation model based on
the theory of planned behavior to dynamically quantify the reputation of RSUs,
thereby better accommodating the personalized trust preferences of users. We
then model the vehicular AI agent migration process as a partially observable
markov decision process and develop a Confidence-regulated Generative Diffusion
Model (CGDM) to efficiently generate AI agent migration decisions. Numerical
results demonstrate that the CGDM algorithm significantly outperforms baseline
methods in reducing system latency and enhancing robustness against cyber
attacks.

### 3. [An Automated Blackbox Noncompliance Checker for QUIC Server Implementations](http://arxiv.org/pdf/2505.12690v1)

Authors: Kian Kai Ang, Guy Farrelly, Cheryl Pope, Damith C. Ranasinghe

We develop QUICtester, an automated approach for uncovering non-compliant
behaviors in the ratified QUIC protocol implementations (RFC 9000/9001).
QUICtester leverages active automata learning to abstract the behavior of a
QUIC implementation into a finite state machine (FSM) representation. Unlike
prior noncompliance checking methods, to help uncover state dependencies on
event timing, QUICtester introduces the idea of state learning with event
timing variations, adopting both valid and invalid input configurations, and
combinations of security and transport layer parameters during learning. We use
pairwise differential analysis of learned behaviour models of tested QUIC
implementations to identify non-compliance instances as behaviour deviations in
a property-agnostic way. This exploits the existence of the many different QUIC
implementations, removing the need for validated, formal models. The diverse
implementations act as cross-checking test oracles to discover non-compliance.
We used QUICtester to analyze analyze 186 learned models from 19 QUIC
implementations under the five security settings and discovered 55
implementation errors. Significantly, the tool uncovered a QUIC specification
ambiguity resulting in an easily exploitable DoS vulnerability, led to 5 CVE
assignments from developers, and two bug bounties thus far.

### 4. [Learning Driven Elastic Task Multi-Connectivity Immersive Computing Systems](http://arxiv.org/pdf/2505.13331v1)

Authors: Babak Badnava, Jacob Chakareski, Morteza Hashemi

In virtual reality (VR) environments, computational tasks exhibit an elastic
nature, meaning they can dynamically adjust based on various user and system
constraints. This elasticity is essential for maintaining immersive
experiences; however, it also introduces challenges for communication and
computing in VR systems. In this paper, we investigate elastic task offloading
for multi-user edge-computing-enabled VR systems with multi-connectivity,
aiming to maximize the computational energy-efficiency (computational
throughput per unit of energy consumed). To balance the induced communication,
computation, energy consumption, and quality of experience trade-offs due to
the elasticity of VR tasks, we formulate a constrained stochastic computational
energy-efficiency optimization problem that integrates the
multi-connectivity/multi-user action space and the elastic nature of VR
computational tasks. We formulate a centralized phasic policy gradient (CPPG)
framework to solve the problem of interest online, using only prior elastic
task offloading statistics (energy consumption, response time, and transmission
time), and task information (i.e., task size and computational intensity),
while observing the induced system performance (energy consumption and
latency). We further extend our approach to decentralized learning by
formulating an independent phasic policy gradient (IPPG) method and a
decentralized shared multi-armed bandit (DSMAB) method. We train our methods
with real-world 4G, 5G, and WiGig network traces and 360 video datasets to
evaluate their performance in terms of response time, energy efficiency,
scalability, and delivered quality of experience. We also provide a
comprehensive analysis of task size and its effect on offloading policy and
system performance. In particular, we show that CPPG reduces latency by 28% and
energy consumption by 78% compared to IPPG.

### Robotics

### 1. [EndoForce: Development of an Intuitive Axial Force Measurement Device for Endoscopic Robotic Systems](http://arxiv.org/pdf/2505.12624v1)

Authors: Hansoul Kim, Dong-Ho Lee, Dukyoo Kong, Dong-Soo Kwon, Byungsik Cheon

Robotic endoscopic systems provide intuitive control and eliminate radiation
exposure, making them a promising alternative to conventional methods. However,
the lack of axial force measurement from the robot remains a major challenge,
as it can lead to excessive colonic elongation, perforation, or ureteral
complications. Although various methods have been proposed in previous studies,
limitations such as model dependency, bulkiness, and environmental sensitivity
remain challenges that should be addressed before clinical application. In this
study, we propose EndoForce, a device designed for intuitive and accurate axial
force measurement in endoscopic robotic systems. Inspired by the insertion
motion performed by medical doctors during ureteroscopy and gastrointestinal
(GI) endoscopy, EndoForce ensures precise force measuring while maintaining
compatibility with clinical environments. The device features a streamlined
design, allowing for the easy attachment and detachment of a sterile cover, and
incorporates a commercial load cell to enhance cost-effectiveness and
facilitate practical implementation in real medical applications. To validate
the effectiveness of the proposed EndoForce, physical experiments were
performed using a testbed that simulates the ureter. We show that the axial
force generated during insertion was measured with high accuracy, regardless of
whether the pathway was straight or curved, in a testbed simulating the human
ureter.

### 2. [SafeMove-RL: A Certifiable Reinforcement Learning Framework for Dynamic Motion Constraints in Trajectory Planning](http://arxiv.org/pdf/2505.12648v1)

Authors: Tengfei Liu, Haoyang Zhong, Jiazheng Hu, Tan Zhang

This study presents a dynamic safety margin-based reinforcement learning
framework for local motion planning in dynamic and uncertain environments. The
proposed planner integrates real-time trajectory optimization with adaptive gap
analysis, enabling effective feasibility assessment under partial observability
constraints. To address safety-critical computations in unknown scenarios, an
enhanced online learning mechanism is introduced, which dynamically corrects
spatial trajectories by forming dynamic safety margins while maintaining
control invariance. Extensive evaluations, including ablation studies and
comparisons with state-of-the-art algorithms, demonstrate superior success
rates and computational efficiency. The framework's effectiveness is further
validated on both simulated and physical robotic platforms.

### 3. [The Robot of Theseus: A modular robotic testbed for legged locomotion](http://arxiv.org/pdf/2505.12649v1)

Authors: Karthik Urs, Jessica Carlson, Aditya Srinivas Manohar, Michael Rakowiecki, Abdulhadi Alkayyali, John E. Saunders, Faris Tulbah, Talia Y. Moore

Robotic models are useful for independently varying specific features, but
most quadrupedal robots differ so greatly from animal morphologies that they
have minimal biomechanical relevance. Commercially available quadrupedal robots
are also prohibitively expensive for biological research programs and difficult
to customize. Here, we present a low-cost quadrupedal robot with modular legs
that can match a wide range of animal morphologies for biomechanical hypothesis
testing. The Robot Of Theseus (TROT) costs approximately $4000 to build out of
3D printed parts and standard off-the-shelf supplies. Each limb consists of 2
or 3 rigid links; the proximal joint can be rotated to become a knee or elbow.
Telescoping mechanisms vary the length of each limb link. The open-source
software accommodates user-defined gaits and morphology changes. Effective leg
length, or crouch, is determined by the four-bar linkage actuating each joint.
The backdrivable motors can vary virtual spring stiffness and range of motion.
Full descriptions of the TROT hardware and software are freely available
online. We demonstrate the use of TROT to compare locomotion among extant,
extinct, and theoretical morphologies. In addition to biomechanical hypothesis
testing, we envision a variety of different applications for this low-cost,
modular, legged robotic platform, including developing novel control
strategies, clearing land mines, or remote exploration. All CAD and code is
available for download on the TROT project page.

### 4. [Audio-Visual Contact Classification for Tree Structures in Agriculture](http://arxiv.org/pdf/2505.12665v1)

Authors: Ryan Spears, Moonyoung Lee, George Kantor, Oliver Kroemer

Contact-rich manipulation tasks in agriculture, such as pruning and
harvesting, require robots to physically interact with tree structures to
maneuver through cluttered foliage. Identifying whether the robot is contacting
rigid or soft materials is critical for the downstream manipulation policy to
be safe, yet vision alone is often insufficient due to occlusion and limited
viewpoints in this unstructured environment. To address this, we propose a
multi-modal classification framework that fuses vibrotactile (audio) and visual
inputs to identify the contact class: leaf, twig, trunk, or ambient. Our key
insight is that contact-induced vibrations carry material-specific signals,
making audio effective for detecting contact events and distinguishing material
types, while visual features add complementary semantic cues that support more
fine-grained classification. We collect training data using a hand-held sensor
probe and demonstrate zero-shot generalization to a robot-mounted probe
embodiment, achieving an F1 score of 0.82. These results underscore the
potential of audio-visual learning for manipulation in unstructured,
contact-rich environments.

### 5. [Dribble Master: Learning Agile Humanoid Dribbling Through Legged Locomotion](http://arxiv.org/pdf/2505.12679v1)

Authors: Zhuoheng Wang, Jinyin Zhou, Qi Wu

Humanoid soccer dribbling is a highly challenging task that demands dexterous
ball manipulation while maintaining dynamic balance. Traditional rule-based
methods often struggle to achieve accurate ball control due to their reliance
on fixed walking patterns and limited adaptability to real-time ball dynamics.
To address these challenges, we propose a two-stage curriculum learning
framework that enables a humanoid robot to acquire dribbling skills without
explicit dynamics or predefined trajectories. In the first stage, the robot
learns basic locomotion skills; in the second stage, we fine-tune the policy
for agile dribbling maneuvers. We further introduce a virtual camera model in
simulation and design heuristic rewards to encourage active sensing, promoting
a broader visual range for continuous ball perception. The policy is trained in
simulation and successfully transferred to a physical humanoid robot.
Experimental results demonstrate that our method enables effective ball
manipulation, achieving flexible and visually appealing dribbling behaviors
across multiple environments. This work highlights the potential of
reinforcement learning in developing agile humanoid soccer robots. Additional
details, video demonstrations, and code are available at
https://zhuoheng0910.github.io/dribble-master/.

### 6. [MOON: Multi-Objective Optimization-Driven Object-Goal Navigation Using a Variable-Horizon Set-Orienteering Planner](http://arxiv.org/pdf/2505.12752v1)

Authors: Daigo Nakajima, Kanji Tanaka, Daiki Iwata, Kouki Terashima

Object-goal navigation (ON) enables autonomous robots to locate and reach
user-specified objects in previously unknown environments, offering promising
applications in domains such as assistive care and disaster response. Existing
ON methods -- including training-free approaches, reinforcement learning, and
zero-shot planners -- generally depend on active exploration to identify
landmark objects (e.g., kitchens or desks), followed by navigation toward
semantically related targets (e.g., a specific mug). However, these methods
often lack strategic planning and do not adequately address trade-offs among
multiple objectives. To overcome these challenges, we propose a novel framework
that formulates ON as a multi-objective optimization problem (MOO), balancing
frontier-based knowledge exploration with knowledge exploitation over
previously observed landmarks; we call this framework MOON (MOO-driven ON). We
implement a prototype MOON system that integrates three key components: (1)
building on QOM [IROS05], a classical ON system that compactly and
discriminatively encodes landmarks based on their semantic relevance to the
target; (2) integrating StructNav [RSS23], a recently proposed training-free
planner, to enhance the navigation pipeline; and (3) introducing a
variable-horizon set orienteering problem formulation to enable global
optimization over both exploration and exploitation strategies. This work
represents an important first step toward developing globally optimized,
next-generation object-goal navigation systems.

### 7. [Practical Equivalence Testing and Its Application in Synthetic Pre-Crash Scenario Validation](http://arxiv.org/pdf/2505.12827v2)

Authors: Jian Wu, Ulrich Sander, Carol Flannagan, Minxiang Zhao, Jonas Bärgman

The use of representative pre-crash scenarios is critical for assessing the
safety impact of driving automation systems through simulation. However, a gap
remains in the robust evaluation of the similarity between synthetic and
real-world pre-crash scenarios and their crash characteristics. Without proper
validation, it cannot be ensured that the synthetic test scenarios adequately
represent real-world driving behaviors and crash characteristics. One reason
for this validation gap is the lack of focus on methods to confirm that the
synthetic test scenarios are practically equivalent to real-world ones, given
the assessment scope. Traditional statistical methods, like significance
testing, focus on detecting differences rather than establishing equivalence;
since failure to detect a difference does not imply equivalence, they are of
limited applicability for validating synthetic pre-crash scenarios and crash
characteristics. This study addresses this gap by proposing an equivalence
testing method based on the Bayesian Region of Practical Equivalence (ROPE)
framework. This method is designed to assess the practical equivalence of
scenario characteristics that are most relevant for the intended assessment,
making it particularly appropriate for the domain of virtual safety
assessments. We first review existing equivalence testing methods. Then we
propose and demonstrate the Bayesian ROPE-based method by testing the
equivalence of two rear-end pre-crash datasets. Our approach focuses on the
most relevant scenario characteristics. Our analysis provides insights into the
practicalities and effectiveness of equivalence testing in synthetic test
scenario validation and demonstrates the importance of testing for improving
the credibility of synthetic data for automated vehicle safety assessment, as
well as the credibility of subsequent safety impact assessments.

### 8. [Granular Loco-Manipulation: Repositioning Rocks Through Strategic Sand Avalanche](http://arxiv.org/pdf/2505.12934v1)

Authors: Haodi Hu, Yue Wu, Feifei Qian, Daniel Seita

Legged robots have the potential to leverage obstacles to climb steep sand
slopes. However, efficiently repositioning these obstacles to desired locations
is challenging. Here we present DiffusiveGRAIN, a learning-based method that
enables a multi-legged robot to strategically induce localized sand avalanches
during locomotion and indirectly manipulate obstacles. We conducted 375 trials,
systematically varying obstacle spacing, robot orientation, and leg actions in
75 of them. Results show that the movement of closely-spaced obstacles exhibits
significant interference, requiring joint modeling. In addition, different
multi-leg excavation actions could cause distinct robot state changes,
necessitating integrated planning of manipulation and locomotion. To address
these challenges, DiffusiveGRAIN includes a diffusion-based environment
predictor to capture multi-obstacle movements under granular flow interferences
and a robot state predictor to estimate changes in robot state from multi-leg
action patterns. Deployment experiments (90 trials) demonstrate that by
integrating the environment and robot state predictors, the robot can
autonomously plan its movements based on loco-manipulation goals, successfully
shifting closely located rocks to desired locations in over 65% of trials. Our
study showcases the potential for a locomoting robot to strategically
manipulate obstacles to achieve improved mobility on challenging terrains.

### 9. [Policy Contrastive Decoding for Robotic Foundation Models](http://arxiv.org/pdf/2505.13255v1)

Authors: Shihan Wu, Ji Zhang, Xu Luo, Junlin Xie, Jingkuan Song, Heng Tao Shen, Lianli Gao

Robotic foundation models, or generalist robot policies, hold immense
potential to enable flexible, general-purpose and dexterous robotic systems.
Despite their advancements, our empirical experiments reveal that existing
robot policies are prone to learning spurious correlations from pre-training
trajectories, adversely affecting their generalization capabilities beyond the
training data. To tackle this, we propose a novel Policy Contrastive Decoding
(PCD) approach, which redirects the robot policy's focus toward object-relevant
visual clues by contrasting action probability distributions derived from
original and object-masked visual inputs. As a training-free method, our PCD
can be used as a plugin to improve different types of robot policies without
needing to finetune or access model weights. We conduct extensive experiments
on top of three open-source robot policies, including the autoregressive policy
OpenVLA and the diffusion-based policies Octo and $\pi_0$. The obtained results
in both simulation and real-world environments prove PCD's flexibility and
effectiveness, e.g., PCD enhances the state-of-the-art policy $\pi_0$ by 8% in
the simulation environment and by 108% in the real-world environment. Code and
demos are publicly available at: https://Koorye.github.io/proj/PCD.

### 10. [Hybrid Voting-Based Task Assignment in Modular Construction Scenarios](http://arxiv.org/pdf/2505.13278v1)

Authors: Daniel Weiner, Raj Korpan

Modular construction, involving off-site prefabrication and on-site assembly,
offers significant advantages but presents complex coordination challenges for
robotic automation. Effective task allocation is critical for leveraging
multi-agent systems (MAS) in these structured environments. This paper
introduces the Hybrid Voting-Based Task Assignment (HVBTA) framework, a novel
approach to optimizing collaboration between heterogeneous multi-agent
construction teams. Inspired by human reasoning in task delegation, HVBTA
uniquely integrates multiple voting mechanisms with the capabilities of a Large
Language Model (LLM) for nuanced suitability assessment between agent
capabilities and task requirements. The framework operates by assigning
Capability Profiles to agents and detailed requirement lists called Task
Descriptions to construction tasks, subsequently generating a quantitative
Suitability Matrix. Six distinct voting methods, augmented by a pre-trained
LLM, analyze this matrix to robustly identify the optimal agent for each task.
Conflict-Based Search (CBS) is integrated for decentralized, collision-free
path planning, ensuring efficient and safe spatio-temporal coordination of the
robotic team during assembly operations. HVBTA enables efficient, conflict-free
assignment and coordination, facilitating potentially faster and more accurate
modular assembly. Current work is evaluating HVBTA's performance across various
simulated construction scenarios involving diverse robotic platforms and task
complexities. While designed as a generalizable framework for any domain with
clearly definable tasks and capabilities, HVBTA will be particularly effective
for addressing the demanding coordination requirements of multi-agent
collaborative robotics in modular construction due to the predetermined
construction planning involved.

### Software Engineering

### 1. [Decompile-Bench: Million-Scale Binary-Source Function Pairs for Real-World Binary Decompilation](http://arxiv.org/pdf/2505.12668v1)

Authors: Hanzhuo Tan, Xiaolong Tian, Hanrui Qi, Jiaming Liu, Zuchen Gao, Siyi Wang, Qi Luo, Jing Li, Yuqun Zhang

Recent advances in LLM-based decompilers have been shown effective to convert
low-level binaries into human-readable source code. However, there still lacks
a comprehensive benchmark that provides large-scale binary-source function
pairs, which is critical for advancing the LLM decompilation technology.
Creating accurate binary-source mappings incurs severe issues caused by complex
compilation settings and widespread function inlining that obscure the
correspondence between binaries and their original source code. Previous
efforts have either relied on used contest-style benchmarks, synthetic
binary-source mappings that diverge significantly from the mappings in real
world, or partially matched binaries with only code lines or variable names,
compromising the effectiveness of analyzing the binary functionality. To
alleviate these issues, we introduce Decompile-Bench, the first open-source
dataset comprising two million binary-source function pairs condensed from 100
million collected function pairs, i.e., 450GB of binaries compiled from
permissively licensed GitHub projects. For the evaluation purposes, we also
developed a benchmark Decompile-Bench-Eval including manually crafted binaries
from the well-established HumanEval and MBPP, alongside the compiled GitHub
repositories released after 2025 to mitigate data leakage issues. We further
explore commonly-used evaluation metrics to provide a thorough assessment of
the studied LLM decompilers and find that fine-tuning with Decompile-Bench
causes a 20% improvement over previous benchmarks in terms of the
re-executability rate. Our code and data has been released in HuggingFace and
Github. https://github.com/albertan017/LLM4Decompile

### 2. [Understanding and Detecting Peer Dependency Resolving Loop in npm Ecosystem](http://arxiv.org/pdf/2505.12676v2)

Authors: Xingyu Wang, Mingsen Wang, Wenbo Shen, Rui Chang

As the default package manager for Node.js, npm has become one of the largest
package management systems in the world. To facilitate dependency management
for developers, npm supports a special type of dependency, Peer Dependency,
whose installation and usage differ from regular dependencies. However,
conflicts between peer dependencies can trap the npm client into infinite
loops, leading to resource exhaustion and system crashes. We name this problem
PeerSpin. Although PeerSpin poses a severe risk to ecosystems, it was
overlooked by previous studies, and its impacts have not been explored.
  To bridge this gap, this paper conducts the first in-depth study to
understand and detect PeerSpin in the npm ecosystem. First, by systematically
analyzing the npm dependency resolution, we identify the root cause of PeerSpin
and characterize two peer dependency patterns to guide detection. Second, we
propose a novel technique called Node-Replacement-Conflict based PeerSpin
Detection, which leverages the state of the directory tree during dependency
resolution to achieve accurate and efficient PeerSpin detection. Based on this
technique, we developed a tool called PeerChecker to detect PeerSpin. Finally,
we apply PeerChecker to the entire NPM ecosystem and find that 5,662 packages,
totaling 72,968 versions, suffer from PeerSpin. Up until now, we confirmed 28
real PeerSpin problems by reporting them to the package maintainer. We also
open source all PeerSpin analysis implementations, tools, and data sets to the
public to help the community detect PeerSpin issues and enhance the reliability
of the npm ecosystem.

### 3. [High-Performance ARM-on-ARM Virtualization for Multicore SystemC-TLM-Based Virtual Platforms](http://arxiv.org/pdf/2505.12987v1)

Authors: Nils Bosbach, Rebecca Pelke, Niko Zurstraßen, Jan Henrik Weinstock, Lukas Jünger, Rainer Leupers

The increasing complexity of hardware and software requires advanced
development and test methodologies for modern systems on chips. This paper
presents a novel approach to ARM-on-ARM virtualization within SystemC-based
simulators using Linux's KVM to achieve high-performance simulation. By running
target software natively on ARM-based hosts with hardware-based virtualization
extensions, our method eliminates the need for instruction-set simulators,
which significantly improves performance. We present a multicore
SystemC-TLM-based CPU model that can be used as a drop-in replacement for an
instruction-set simulator. It places no special requirements on the host
system, making it compatible with various environments. Benchmark results show
that our ARM-on-ARM-based virtual platform achieves up to 10 x speedup over
traditional instruction-set-simulator-based models on compute-intensive
workloads. Depending on the benchmark, speedups increase to more than 100 x.

### 4. [Adversarial Reasoning for Repair Based on Inferred Program Intent](http://arxiv.org/pdf/2505.13008v1)

Authors: He Ye, Aidan Z. H. Yang, Chang Hu, Yanlin Wang, Tao Zhang, Claire Le Goues

Automated program repair (APR) has shown promising results, particularly with
the use of neural networks. Currently, most APR tools focus on code
transformations specified by test suites, rather than reasoning about the
program intent and the high-level bug specification. Without a proper
understanding of program intent, these tools tend to generate patches that
overfit incomplete test suites and fail to reflect the developers intentions.
However, reasoning about program intent is challenging. In our work, we propose
an approach called AdverIntent-Agent, based on critique and adversarial
reasoning. Our approach is novel to shift the focus from generating multiple
APR patches to inferring multiple potential program intents. Ideally, we aim to
infer intents that are, to some extent, adversarial to each other, maximizing
the probability that at least one aligns closely with the developers original
intent. AdverIntent-Agent is a multi-agent approach consisting of three agents:
a reasoning agent, a test agent, and a repair agent. First, the reasoning agent
generates adversarial program intents along with the corresponding faulty
statements. Next, the test agent produces adversarial test cases that align
with each inferred intent, constructing oracles that use the same inputs but
have different expected outputs. Finally, the repair agent uses dynamic and
precise LLM prompts to generate patches that satisfy both the inferred program
intent and the generated tests. AdverIntent-Agent was evaluated on two
benchmarks: Defects4J 2.0 and HumanEval-Java. AdverIntent-Agent correctly
repaired 77 and 105 bugs in both benchmarks, respectively.

### 5. [Manifesto from Dagstuhl Perspectives Workshop 24452 -- Reframing Technical Debt](http://arxiv.org/pdf/2505.13009v1)

Authors: Paris Avgeriou, Ipek Ozkaya, Heiko Koziolek, Zadia Codabux, Neil Ernst

This is the Dagstuhl Perspectives Workshop 24452 manifesto on Reframing
Technical Debt. The manifesto begins with a one-page summary of Values,
Beliefs, and Principles. It then elaborates on each Value, Belief, and
Principle to explain their rationale and clarify their meaning. Subsequently,
the paper describes the current landscape of Technical Debt Management methods
and tools and explains why the current practice is inadequate and where current
research falls short. The current landscape is organized into five major
topics: Technical Debt as Value-Creation, Tooling, Data Collection, the role of
Architecture, and Socio-Technical Aspects. Finally, the paper outlines a
roadmap to realize the stated principles, with concrete milestones to be
addressed by researchers, software practitioners, and tool vendors. The
manifesto is signed by the workshop participants.

### 6. [Aspects of complexity in automotive software systems and their relation to maintainability effort. A case study](http://arxiv.org/pdf/2505.13135v1)

Authors: Bengt Haraldsson, Miroslaw Staron

Context: Large embedded systems in vehicles tend to grow in size and
complexity, which causes challenges when maintaining these systems. Objective:
We explore how developers perceive the relation between maintainability effort
and various sources of complexity. Methods: We conduct a case study at Scania
AB, a heavy vehicle OEM. The units of analysis are two large software systems
and their development teams/organizations. Results: Our results show that
maintainability effort is driven by system internal complexity in the form of
variant management and complex hardware control tasks. The maintainability is
also influenced by emergent complexity caused by the system's longevity and
constant growth. Besides these system-internal complexities, maintainability
effort is also influenced by external complexities, such as organizational
coordination and business needs. During the study, developer trade-off
strategies for minimizing maintainability effort emerged. Conclusions:
Complexity is a good proxy of maintainability effort, and allows developers to
create strategies for managing the maintainability effort. Adequate complexity
metrics include both external aspects -- e.g., coordination complexity -- and
internal ones -- e.g., McCabe Cyclomatic Complexity.

### 7. [PARF: An Adaptive Abstraction-Strategy Tuner for Static Analysis](http://arxiv.org/pdf/2505.13229v1)

Authors: Zhongyi Wang, Mingshuai Chen, Tengjie Lin, Linyu Yang, Junhao Zhuo, Qiuye Wang, Shengchao Qin, Xiao Yi, Jianwei Yin

We launch Parf - a toolkit for adaptively tuning abstraction strategies of
static program analyzers in a fully automated manner. Parf models various types
of external parameters (encoding abstraction strategies) as random variables
subject to probability distributions over latticed parameter spaces. It
incrementally refines the probability distributions based on accumulated
intermediate results generated by repeatedly sampling and analyzing, thereby
ultimately yielding a set of highly accurate abstraction strategies. Parf is
implemented on top of Frama-C/Eva - an off-the-shelf open-source static
analyzer for C programs. Parf provides a web-based user interface facilitating
the intuitive configuration of static analyzers and visualization of dynamic
distribution refinement of the abstraction strategies. It further supports the
identification of dominant parameters in Frama-C/Eva analysis. Benchmark
experiments and a case study demonstrate the competitive performance of Parf
for analyzing complex, large-scale real-world programs.

### 8. [Are requirements really all you need? A case study of LLM-driven configuration code generation for automotive simulations](http://arxiv.org/pdf/2505.13263v1)

Authors: Krzysztof Lebioda, Nenad Petrovic, Fengjunjie Pan, Vahid Zolfaghari, Andre Schamschurko, Alois Knoll

Large Language Models (LLMs) are taking many industries by storm. They
possess impressive reasoning capabilities and are capable of handling complex
problems, as shown by their steadily improving scores on coding and
mathematical benchmarks. However, are the models currently available truly
capable of addressing real-world challenges, such as those found in the
automotive industry? How well can they understand high-level, abstract
instructions? Can they translate these instructions directly into functional
code, or do they still need help and supervision? In this work, we put one of
the current state-of-the-art models to the test. We evaluate its performance in
the task of translating abstract requirements, extracted from automotive
standards and documents, into configuration code for CARLA simulations.

### 9. [NEAT: QCP: A Practical Separation Logic-based C Program Verification Tool](http://arxiv.org/pdf/2505.12878v1)

Authors: Xiwei Wu, Yueyang Feng, Xiaoyang Lu, Tianchuan Lin, Kan Liu, Zhiyi Wang, Shushu Wu, Lihan Xie, Chengxi Yang, Hongyi Zhong, Naijun Zhan, Zhenjiang Hu, Qinxiang Cao

As software systems increase in size and complexity dramatically, ensuring
their correctness, security, and reliability becomes an increasingly formidable
challenge. Despite significant advancements in verification techniques and
tools, there still remain %these tools still continue to encounter substantial
difficulties when applying these tools to complex, real-world scenarios. To
address these difficulties, this paper introduces a novel verification tool,
called \textbf{Qualified C Programming Verifier (QCP)}. QCP incorporates a
refined front-end %syntax of assertion language to enhance user interaction.
The proposed assertion language aims to %syntax is designed to lower the entry
barrier for verification tools, improve proof efficiency by improving
automation, and facilitate a deeper understanding of both the program and its
verification results.

### 10. [Structure-Aware Corpus Construction and User-Perception-Aligned Metrics for Large-Language-Model Code Completion](http://arxiv.org/pdf/2505.13073v1)

Authors: Dengfeng Liu, Jucai Zhai, Xiaoguang Jiang, Ziqun Li, Qianjin Yu, Feng Liu, Rui Ye, Huang Liu, Zhiguo Yang, Yongsheng Du, Fang Tan

Code completion technology based on large language model has significantly
improved the development efficiency of programmers. However, in practical
applications, there remains a gap between current commonly used code completion
evaluation metrics and users' actual perception. To address this issue, we
propose two evaluation metrics for code completion tasks--LCP and ROUGE-LCP,
from the perspective of probabilistic modeling. Furthermore, to tackle the lack
of effective structural semantic modeling and cross-module dependency
information in LLMs for repository-level code completion scenarios, we propose
a data processing method based on a Structure-Preserving and
Semantically-Reordered Code Graph (SPSR-Graph). Through theoretical analysis
and experimental validation, we demonstrate the superiority of the proposed
evaluation metrics in terms of user perception consistency, as well as the
effectiveness of the data processing method in enhancing model performance.

### Social and Information Networks

### 1. [A large-scale analysis of public-facing, community-built chatbots on Character.AI](http://arxiv.org/pdf/2505.13354v1)

Authors: Owen Lee, Kenneth Joseph

This paper presents the first large-scale analysis of public-facing chatbots
on Character.AI, a rapidly growing social media platform where users create and
interact with chatbots. Character.AI is distinctive in that it merges
generative AI with user-generated content, enabling users to build bots-often
modeled after fictional or public personas-for others to engage with. It is
also popular, with over 20 million monthly active users, and impactful, with
recent headlines detailing significant issues with youth engagement on the
site. Character.AI is thus of interest to study both substantively and
conceptually. To this end, we present a descriptive overview of the site using
a dataset of 2.1 million English-language prompts (or ``greetings'') for
chatbots on the site, created by around 1 million users. Our work explores the
prevalence of different fandoms on the site, broader tropes that persist across
fandoms, and how dynamics of power intersect with gender within greetings.
Overall, our findings illuminate an emerging form of online (para)social
interaction that toes a unique and important intersection between generative AI
and user-generated content.

### 2. [HyperDet: Source Detection in Hypergraphs via Interactive Relationship Construction and Feature-rich Attention Fusion](http://arxiv.org/pdf/2505.12894v1)

Authors: Le Cheng, Peican Zhu, Yangming Guo, Keke Tang, Chao Gao, Zhen Wang

Hypergraphs offer superior modeling capabilities for social networks,
particularly in capturing group phenomena that extend beyond pairwise
interactions in rumor propagation. Existing approaches in rumor source
detection predominantly focus on dyadic interactions, which inadequately
address the complexity of more intricate relational structures. In this study,
we present a novel approach for Source Detection in Hypergraphs (HyperDet) via
Interactive Relationship Construction and Feature-rich Attention Fusion.
Specifically, our methodology employs an Interactive Relationship Construction
module to accurately model both the static topology and dynamic interactions
among users, followed by the Feature-rich Attention Fusion module, which
autonomously learns node features and discriminates between nodes using a
self-attention mechanism, thereby effectively learning node representations
under the framework of accurately modeled higher-order relationships. Extensive
experimental validation confirms the efficacy of our HyperDet approach,
showcasing its superiority relative to current state-of-the-art methods.

### 3. [SourceDetMamba: A Graph-aware State Space Model for Source Detection in Sequential Hypergraphs](http://arxiv.org/pdf/2505.12910v1)

Authors: Le Cheng, Peican Zhu, Yangming Guo, Chao Gao, Zhen Wang, Keke Tang

Source detection on graphs has demonstrated high efficacy in identifying
rumor origins. Despite advances in machine learning-based methods, many fail to
capture intrinsic dynamics of rumor propagation. In this work, we present
SourceDetMamba: A Graph-aware State Space Model for Source Detection in
Sequential Hypergraphs, which harnesses the recent success of the state space
model Mamba, known for its superior global modeling capabilities and
computational efficiency, to address this challenge. Specifically, we first
employ hypergraphs to model high-order interactions within social networks.
Subsequently, temporal network snapshots generated during the propagation
process are sequentially fed in reverse order into Mamba to infer underlying
propagation dynamics. Finally, to empower the sequential model to effectively
capture propagation patterns while integrating structural information, we
propose a novel graph-aware state update mechanism, wherein the state of each
node is propagated and refined by both temporal dependencies and topological
context. Extensive evaluations on eight datasets demonstrate that
SourceDetMamba consistently outperforms state-of-the-art approaches.

### 4. [Measuring Social Influence with Networked Synthetic Control](http://arxiv.org/pdf/2505.13334v1)

Authors: Ho-Chun Herbert Chang

Measuring social influence is difficult due to the lack of counter-factuals
and comparisons. By combining machine learning-based modeling and network
science, we present general properties of social value, a recent measure for
social influence using synthetic control applicable to political behavior.
Social value diverges from centrality measures on in that it relies on an
external regressor to predict an output variable of interest, generates a
synthetic measure of influence, then distributes individual contribution based
on a social network. Through theoretical derivations, we show the properties of
SV under linear regression with and without interaction, across lattice
networks, power-law networks, and random graphs. A reduction in computation can
be achieved for any ensemble model. Through simulation, we find that the
generalized friendship paradox holds -- that in certain situations, your
friends have on average more influence than you do.

### 5. [Transmission Neural Networks: Approximation and Optimal Control](http://arxiv.org/pdf/2505.12657v1)

Authors: Shuang Gao, Peter E. Caines

Transmission Neural Networks (TransNNs) introduced by Gao and Caines (2022)
connect virus spread models over networks and neural networks with tuneable
activation functions. This paper presents the approximation technique and the
underlying assumptions employed by TransNNs in relation to the corresponding
Markovian Susceptible-Infected-Susceptible (SIS) model with 2^n states, where n
is the number of nodes in the network. The underlying infection paths are
assumed to be stochastic with heterogeneous and time-varying transmission
probabilities. We obtain the conditional probability of infection in the
stochastic 2^n-state SIS epidemic model corresponding to each state
configuration under mild assumptions, which enables control solutions based on
Markov decision processes (MDP). Finally, MDP control with 2^n-state SIS
epidemic models and optimal control with TransNNs are compared in terms of
mitigating virus spread over networks through vaccination, and it is shown that
TranNNs enable the generation of control laws with significant computational
savings, albeit with more conservative control actions.

### 6. [Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](http://arxiv.org/pdf/2505.12684v1)

Authors: Yinlin Zhu, Xunkai Li, Jishuo Jia, Miao Hu, Di Wu, Meikang Qiu

Recent advances in graph machine learning have shifted to data-centric
paradigms, driven by two emerging fields: (1) Federated graph learning (FGL)
enables multi-client collaboration but faces challenges from data and task
heterogeneity, limiting its practicality; (2) Graph foundation models (GFM)
offer strong domain generalization but are usually trained on single machines,
missing out on cross-silo data and resources.
  These paradigms are complementary, and their integration brings notable
benefits. Motivated by this, we propose FedGFM, a novel decentralized GFM
training paradigm. However, a key challenge is knowledge entanglement, where
multi-domain knowledge merges into indistinguishable representations, hindering
downstream adaptation.
  To address this, we present FedGFM+, an enhanced framework with two core
modules to reduce knowledge entanglement: (1) AncDAI: A global anchor-based
domain-aware initialization strategy. Before pre-training, each client encodes
its local graph into domain-specific prototypes that serve as semantic anchors.
Synthetic embeddings around these anchors initialize the global model. We
theoretically prove these prototypes are distinguishable across domains,
providing a strong inductive bias to disentangle domain-specific knowledge. (2)
AdaDPP: A local adaptive domain-sensitive prompt pool. Each client learns a
lightweight graph prompt capturing domain semantics during pre-training. During
fine-tuning, prompts from all clients form a pool from which the GFM selects
relevant prompts to augment target graph attributes, improving downstream
adaptation.
  FedGFM+ is evaluated on 8 diverse benchmarks across multiple domains and
tasks, outperforming 20 baselines from supervised learning, FGL, and federated
GFM variants.

### 7. [EpiLLM: Unlocking the Potential of Large Language Models in Epidemic Forecasting](http://arxiv.org/pdf/2505.12738v1)

Authors: Chenghua Gong, Rui Sun, Yuhao Zheng, Juyuan Zhang, Tianjun Gu, Liming Pan, Linyuan Lv

Advanced epidemic forecasting is critical for enabling precision containment
strategies, highlighting its strategic importance for public health security.
While recent advances in Large Language Models (LLMs) have demonstrated
effectiveness as foundation models for domain-specific tasks, their potential
for epidemic forecasting remains largely unexplored. In this paper, we
introduce EpiLLM, a novel LLM-based framework tailored for spatio-temporal
epidemic forecasting. Considering the key factors in real-world epidemic
transmission: infection cases and human mobility, we introduce a dual-branch
architecture to achieve fine-grained token-level alignment between such complex
epidemic patterns and language tokens for LLM adaptation. To unleash the
multi-step forecasting and generalization potential of LLM architectures, we
propose an autoregressive modeling paradigm that reformulates the epidemic
forecasting task into next-token prediction. To further enhance LLM perception
of epidemics, we introduce spatio-temporal prompt learning techniques, which
strengthen forecasting capabilities from a data-driven perspective. Extensive
experiments show that EpiLLM significantly outperforms existing baselines on
real-world COVID-19 datasets and exhibits scaling behavior characteristic of
LLMs.

### 8. [Counting Graphlets of Size $k$ under Local Differential Privacy](http://arxiv.org/pdf/2505.12954v1)

Authors: Vorapong Suppakitpaisarn, Donlapark Ponnoprat, Nicha Hirankarn, Quentin Hillebrand

The problem of counting subgraphs or graphlets under local differential
privacy is an important challenge that has attracted significant attention from
researchers. However, much of the existing work focuses on small graphlets like
triangles or $k$-stars. In this paper, we propose a non-interactive, locally
differentially private algorithm capable of counting graphlets of any size $k$.
When $n$ is the number of nodes in the input graph, we show that the expected
$\ell_2$ error of our algorithm is $O(n^{k - 1})$. Additionally, we prove that
there exists a class of input graphs and graphlets of size $k$ for which any
non-interactive counting algorithm incurs an expected $\ell_2$ error of
$\Omega(n^{k - 1})$, demonstrating the optimality of our result. Furthermore,
we establish that for certain input graphs and graphlets, any locally
differentially private algorithm must have an expected $\ell_2$ error of
$\Omega(n^{k - 1.5})$. Our experimental results show that our algorithm is more
accurate than the classical randomized response method.

### Systems and Control

### 1. [A Control Oriented Fractional-Order Model of Lithium-ion Batteries Based on Caputo Definition](http://arxiv.org/pdf/2505.12725v1)

Authors: Yangyang Xu, Hongyu Zhao, Chengzhong Zhang, Chenglin Liao

This letter proposes a fractional-order battery model based on the Caputo
definition. A closed-form time-domain solution is derived, enabling a simple
recursive expression for discrete-time implementation. The model requires only
the current and previous time-step states in each iteration, significantly
reducing memory usage compared to the conventional Gr\"{u}nwald--Letnikov (G-L)
method. This recursive structure is highly compatible with filter design and
online parameter identification. Experimental validation on a 40.2~Ah NCM622
cell shows that the proposed first-order model achieves voltage prediction
accuracy comparable to a second-order integer-order model. The results
demonstrate that the Caputo-based model offers a practical balance between
accuracy and computational efficiency, making it well suited for real-time
battery management systems (BMS).

### 2. [Scheduling of Flexible Manufacturing Systems Based on Place-Timed Petri Nets and Basis Reachability Graphs](http://arxiv.org/pdf/2505.12862v1)

Authors: Zhou He, Ning Li, Ning Ran, Liang Li

Scheduling is a key decision-making process to improve the performance of
flexible manufacturing systems. Place-timed Petri nets provide a formal method
for graphically modeling and analyzing such systems. By generating reachability
graphs and combining intelligent search algorithms, operation sequences from
the initial state to the target state can be found for the underlying system.
However, the reachability graph grows exponentially with the system size
increases, which is the main challenge of existing methods for scheduling large
systems. To this end, we develop an efficient improved beam search algorithm to
optimize the makespan based on a compact representation of reachability graph
called basis reachability graph. The key idea behind the proposed method is to
form a state together with the basis markings and its corresponding transition
sequences, and evaluate the cost of the state based on the resource idle time.
Experimental results are conducted on several benchmark systems which show that
the developed method improves the search efficiency while ensuring the quality
of the solution compared with existing methods.

### 3. [6G-Enabled Smart Railways](http://arxiv.org/pdf/2505.12946v1)

Authors: Bo Ai, Yunlong Lu, Yuguang Fang, Dusit Niyato, Ruisi He, Wei Chen, Jiayi Zhang, Guoyu Ma, Yong Niu, Zhangdui Zhong

Smart railways integrate advanced information technologies into railway
operating systems to improve efficiency and reliability. Although the
development of 5G has enhanced railway services, future smart railways require
ultra-high speeds, ultra-low latency, ultra-high security, full coverage, and
ultra-high positioning accuracy, which 5G cannot fully meet. Therefore, 6G is
envisioned to provide green and efficient all-day operations, strong
information security, fully automatic driving, and low-cost intelligent
maintenance. To achieve these requirements, we propose an integrated network
architecture leveraging communications, computing, edge intelligence, and
caching in railway systems. We have conducted in-depth investigations on key
enabling technologies for reliable transmissions and wireless coverage. For
high-speed mobile scenarios, we propose an AI-enabled cross-domain channel
modeling and orthogonal time-frequency space-time spread multiple access
mechanism to alleviate the conflict between limited spectrum availability and
massive user access. The roles of blockchain, edge intelligence, and privacy
technologies in endogenously secure rail communications are also evaluated. We
further explore the application of emerging paradigms such as integrated
sensing and communications, AI-assisted Internet of Things, semantic
communications, and digital twin networks for railway maintenance, monitoring,
prediction, and accident warning. Finally, possible future research and
development directions are discussed.

### 4. [Regularized Model Predictive Control](http://arxiv.org/pdf/2505.12977v1)

Authors: Komeil Nosrati, Juri Belikov, Aleksei Tepljakov, Eduard Petlenkov

In model predictive control (MPC), the choice of cost-weighting matrices and
designing the Hessian matrix directly affects the trade-off between rapid state
regulation and minimizing the control effort. However, traditional MPC in
quadratic programming relies on fixed design matrices across the entire
horizon, which can lead to suboptimal performance. This letter presents a
Riccati equation-based method for adjusting the design matrix within the MPC
framework, which enhances real-time performance. We employ a penalized
least-squares (PLS) approach to derive a quadratic cost function for a
discrete-time linear system over a finite prediction horizon. Using the method
of weighting and enforcing the constraint equation by introducing a large
penalty parameter, we solve the constrained optimization problem and generate
control inputs for forward-shifted horizons. This process yields a recursive
PLS-based Riccati equation that updates the design matrix as a regularization
term in each shift, forming the foundation of the regularized MPC (Re-MPC)
algorithm. To accomplish this, we provide a convergence and stability analysis
of the developed algorithm. Numerical analysis demonstrates its superiority
over traditional methods by allowing Riccati equation-based adjustments.

### 5. [RSS-Based Localization: Ensuring Consistency and Asymptotic Efficiency](http://arxiv.org/pdf/2505.13070v1)

Authors: Shenghua Hu, Guangyang Zeng, Wenchao Xue, Haitao Fang, Junfeng Wu, Biqiang Mu

We study the problem of signal source localization using received signal
strength measurements. We begin by presenting verifiable geometric conditions
for sensor deployment that ensure the model's asymptotic localizability. Then
we establish the consistency and asymptotic efficiency of the maximum
likelihood (ML) estimator. However, computing the ML estimator is challenging
due to its reliance on solving a non-convex optimization problem. To overcome
this, we propose a two-step estimator that retains the same asymptotic
properties as the ML estimator while offering low computational complexity,
linear in the number of measurements. The main challenge lies in obtaining a
consistent estimator in the first step. To address this, we construct two
linear least-squares estimation problems by applying algebraic transformations
to the nonlinear measurement model, leading to closed-form solutions. In the
second step, we perform a single Gauss-Newton iteration using the consistent
estimator from the first step as the initialization, achieving the same
asymptotic efficiency as the ML estimator. Finally, simulation results validate
the theoretical property and practical effectiveness of the proposed two-step
estimator.

### 6. [Low-regret Strategies for Energy Systems Planning in a Highly Uncertain Future](http://arxiv.org/pdf/2505.13277v1)

Authors: Gabriel Wiest, Niklas Nolzen, Florian Baader, André Bardow, Stefano Moret

Large uncertainties in the energy transition urge decision-makers to develop
low-regret strategies, i.e., strategies that perform well regardless of how the
future unfolds. To address this challenge, we introduce a decision-support
framework that identifies low-regret strategies in energy system planning under
uncertainty. Our framework (i) automatically identifies strategies, (ii)
evaluates their performance in terms of regret, (iii) assesses the key drivers
of regret, and (iv) supports the decision process with intuitive decision
trees, regret curves and decision maps. We apply the framework to evaluate the
optimal use of biomass in the transition to net-zero energy systems,
considering all major biomass utilization options: biofuels, biomethane,
chemicals, hydrogen, biochar, electricity, and heat. Producing fuels and
chemicals from biomass performs best across various decision-making criteria.
In contrast, the current use of biomass, mainly for low-temperature heat
supply, results in high regret, making it a must-avoid in the energy
transition.

### 7. [Output behavior equivalence and simultaneous subspace identification of systems and faults](http://arxiv.org/pdf/2505.13294v1)

Authors: Gabriel de Albuquerque Gleizer

We address the problem of identifying a system subject to additive faults,
while simultaneously reconstructing the fault signal via subspace methods. We
do not require nominal data for the identification, neither do we impose any
assumption on the class of faults, e.g., sensor or actuator faults. We show
that, under mild assumptions on the fault signal, standard PI-MOESP can recover
the system matrices associated to the input-output subsystem. Then we introduce
the concept of output behavior equivalence, which characterizes systems with
the same output behavior set, and present a method to establish this
equivalence from system matrices. Finally, we show how to estimate from data
the complete set of fault matrices for which there exist a fault signal with
minimal dimension that explains the data.

### 8. [An Empirical Bayes approach to ARX Estimation](http://arxiv.org/pdf/2505.13384v1)

Authors: Timofei Leahu, Giorgio Picci

Empirical Bayes inference is based on estimation of the parameters of an a
priori distribution from the observed data. The estimation technique of the
parameters of the prior, called hyperparameters, is based on the marginal
distribution obtained by integrating the joint density of the model with
respect to the prior. This is a key step which needs to be properly adapted to
the problem at hand. In this paper we study Empirical Bayes inference of linear
autoregressive models with inputs (ARX models) for time series and compare the
performance of the marginal parametric estimator with that a full Empirical
Bayesian analysis based on the estimated prior. Such a comparison, can only
make sense for a (realistic) finite data length. In this setting, we propose a
new estimation technique of the hyperparameters by a sequential Bayes procedure
which is essentially a backward Kalman filter. It turns out that for finite
data length the marginal Bayes tends to behave slightly better than the full
Empirical Bayesian parameter estimator and so also in the case of slowly
varying random parameters.

### 9. [MSCEKF-MIO: Magnetic-Inertial Odometry Based on Multi-State Constraint Extended Kalman Filter](http://arxiv.org/pdf/2505.12634v2)

Authors: Jiazhu Li, Jian Kuang, Xiaoji Niu

To overcome the limitation of existing indoor odometry technologies which
often cannot simultaneously meet requirements for accuracy cost-effectiveness,
and robustness-this paper proposes a novel magnetometer array-aided inertial
odometry approach, MSCEKF-MIO (Multi-State Constraint Extended Kalman
Filter-based Magnetic-Inertial Odometry). We construct a magnetic field model
by fitting measurements from the magnetometer array and then use temporal
variations in this model-extracted from continuous observations-to estimate the
carrier's absolute velocity. Furthermore, we implement the MSCEKF framework to
fuse observed magnetic field variations with position and attitude estimates
from inertial navigation system (INS) integration, thereby enabling autonomous,
high-precision indoor relative positioning. Experimental results demonstrate
that the proposed algorithm achieves superior velocity estimation accuracy and
horizontal positioning precision relative to state-of-the-art magnetic
array-aided INS algorithms (MAINS). On datasets with trajectory lengths of
150-250m, the proposed method yields an average horizontal position RMSE of
approximately 2.5m. In areas with distinctive magnetic features, the
magneto-inertial odometry achieves a velocity estimation accuracy of 0.07m/s.
Consequently, the proposed method offers a novel positioning solution
characterized by low power consumption, cost-effectiveness, and high
reliability in complex indoor environments.

### 10. [Transmission Neural Networks: Approximation and Optimal Control](http://arxiv.org/pdf/2505.12657v1)

Authors: Shuang Gao, Peter E. Caines

Transmission Neural Networks (TransNNs) introduced by Gao and Caines (2022)
connect virus spread models over networks and neural networks with tuneable
activation functions. This paper presents the approximation technique and the
underlying assumptions employed by TransNNs in relation to the corresponding
Markovian Susceptible-Infected-Susceptible (SIS) model with 2^n states, where n
is the number of nodes in the network. The underlying infection paths are
assumed to be stochastic with heterogeneous and time-varying transmission
probabilities. We obtain the conditional probability of infection in the
stochastic 2^n-state SIS epidemic model corresponding to each state
configuration under mild assumptions, which enables control solutions based on
Markov decision processes (MDP). Finally, MDP control with 2^n-state SIS
epidemic models and optimal control with TransNNs are compared in terms of
mitigating virus spread over networks through vaccination, and it is shown that
TranNNs enable the generation of control laws with significant computational
savings, albeit with more conservative control actions.

### Machine Learning (Statistics Category)

### 1. [Testing Identifiability and Transportability with Observational and Experimental Data](http://arxiv.org/pdf/2505.12801v1)

Authors: Konstantina Lelova, Gregory F. Cooper, Sofia Triantafillou

Transporting causal information learned from experiments in one population to
another is a critical challenge in clinical research and decision-making.
Causal transportability uses causal graphs to model differences between the
source and target populations and identifies conditions under which causal
effects learned from experiments can be reused in a different population.
Similarly, causal identifiability identifies conditions under which causal
effects can be estimated from observational data. However, these approaches
rely on knowing the causal graph, which is often unavailable in real-world
settings. In this work, we propose a Bayesian method for assessing whether
Z-specific (conditional) causal effects are both identifiable and
transportable, without knowing the causal graph. Our method combines
experimental data from the source population with observational data from the
target population to compute the probability that a causal effect is both
identifiable from observational data and transportable. When this holds, we
leverage both observational data from the target domain and experimental data
from the source domain to obtain an unbiased, efficient estimator of the causal
effect in the target population. Using simulations, we demonstrate that our
method correctly identifies transportable causal effects and improves causal
effect estimation.

### 2. [Theoretical Investigation on Inductive Bias of Isolation Forest](http://arxiv.org/pdf/2505.12825v1)

Authors: Qin-Cheng Zheng, Shao-Qun Zhang, Shen-Huan Lyu, Yuan Jiang, Zhi-Hua Zhou

Isolation Forest (iForest) stands out as a widely-used unsupervised anomaly
detector valued for its exceptional runtime efficiency and performance on
large-scale tasks. Despite its widespread adoption, a theoretical foundation
explaining iForest's success remains unclear. This paper theoretically
investigates the conditions and extent of iForest's effectiveness by analyzing
its inductive bias through the formulation of depth functions and growth
processes. Since directly analyzing the depth function proves intractable due
to iForest's random splitting mechanism, we model the growth process of iForest
as a random walk, enabling us to derive the expected depth function using
transition probabilities. Our case studies reveal key inductive biases: iForest
exhibits lower sensitivity to central anomalies while demonstrating greater
parameter adaptability compared to $k$-Nearest Neighbor anomaly detectors. Our
study provides theoretical understanding of the effectiveness of iForest and
establishes a foundation for further theoretical exploration.

### 3. [Spline Dimensional Decomposition with Interpolation-based Optimal Knot Selection for Stochastic Dynamic Analysis](http://arxiv.org/pdf/2505.12879v1)

Authors: Yeonsu Kim, Junhan Lee, John T. Hwang, Bingran Wang, Dongjin Lee

Forward uncertainty quantification in dynamic systems is challenging due to
non-smooth or locally oscillating nonlinear behaviors. Spline dimensional
decomposition (SDD) effectively addresses such nonlinearity by partitioning
input coordinates via knot placement, yet its accuracy is highly sensitive to
the location of internal knots. Optimizing knots through sequential quadratic
programming can be effective, yet the optimization process becomes
computationally intense. We propose a computationally efficient,
interpolation-based method for optimal knot selection in SDD. The method
involves three steps: (1) interpolating input-output profiles, (2) defining
subinterval-based reference regions, and (3) selecting optimal knot locations
at maximum gradient points within each region. The resulting knot vector is
then applied to SDD for accurate approximation of non-smooth and locally
oscillating responses. A modal analysis of a lower control arm demonstrates
that SDD with the proposed knot selection achieves higher accuracy than SDD
with uniformly or randomly spaced knots, and also a Gaussian process surrogate
model. The proposed SDD exhibits the lowest relative variance error (2.89%),
compared to SDD with uniformly spaced knots (12.310%), randomly spaced knots
(15.274%), and Gaussian process (5.319%) in the first natural frequency
distribution. All surrogate models are constructed using the same 401
simulation datasets, and the relative errors are evaluated against a
2000-sample Monte Carlo simulation. The scalability and applicability of
proposed method are demonstrated through stochastic and reliability analyses of
mathematical functions (N=1, 3) and a lower control arm system (N=10). The
results confirm that both second-moment statistics and reliability estimates
can be accurately achieved with only a few hundred function evaluations or
finite element simulations.

### 4. [LoD: Loss-difference OOD Detection by Intentionally Label-Noisifying Unlabeled Wild Data](http://arxiv.org/pdf/2505.12952v1)

Authors: Chuanxing Geng, Qifei Li, Xinrui Wang, Dong Liang, Songcan Chen, Pong C. Yuen

Using unlabeled wild data containing both in-distribution (ID) and
out-of-distribution (OOD) data to improve the safety and reliability of models
has recently received increasing attention. Existing methods either design
customized losses for labeled ID and unlabeled wild data then perform joint
optimization, or first filter out OOD data from the latter then learn an OOD
detector. While achieving varying degrees of success, two potential issues
remain: (i) Labeled ID data typically dominates the learning of models,
inevitably making models tend to fit OOD data as IDs; (ii) The selection of
thresholds for identifying OOD data in unlabeled wild data usually faces
dilemma due to the unavailability of pure OOD samples. To address these issues,
we propose a novel loss-difference OOD detection framework (LoD) by
\textit{intentionally label-noisifying} unlabeled wild data. Such operations
not only enable labeled ID data and OOD data in unlabeled wild data to jointly
dominate the models' learning but also ensure the distinguishability of the
losses between ID and OOD samples in unlabeled wild data, allowing the classic
clustering technique (e.g., K-means) to filter these OOD samples without
requiring thresholds any longer. We also provide theoretical foundation for
LoD's viability, and extensive experiments verify its superiority.

### 5. [Asymptotic Performance of Time-Varying Bayesian Optimization](http://arxiv.org/pdf/2505.13012v1)

Authors: Anthony Bardou, Patrick Thiran

Time-Varying Bayesian Optimization (TVBO) is the go-to framework for
optimizing a time-varying black-box objective function that may be noisy and
expensive to evaluate. Is it possible for the instantaneous regret of a TVBO
algorithm to vanish asymptotically, and if so, when? We answer this question of
great theoretical importance by providing algorithm-independent lower regret
bounds and upper regret bounds for TVBO algorithms, from which we derive
sufficient conditions for a TVBO algorithm to have the no-regret property. Our
analysis covers all major classes of stationary kernel functions.

### 6. [Orthogonal Survival Learners for Estimating Heterogeneous Treatment Effects from Time-to-Event Data](http://arxiv.org/pdf/2505.13072v1)

Authors: Dennis Frauen, Maresa Schröder, Konstantin Hess, Stefan Feuerriegel

Estimating heterogeneous treatment effects (HTEs) is crucial for personalized
decision-making. However, this task is challenging in survival analysis, which
includes time-to-event data with censored outcomes (e.g., due to study
dropout). In this paper, we propose a toolbox of novel orthogonal survival
learners to estimate HTEs from time-to-event data under censoring. Our learners
have three main advantages: (i) we show that learners from our toolbox are
guaranteed to be orthogonal and thus come with favorable theoretical
properties; (ii) our toolbox allows for incorporating a custom weighting
function, which can lead to robustness against different types of low overlap,
and (iii) our learners are model-agnostic (i.e., they can be combined with
arbitrary machine learning models). We instantiate the learners from our
toolbox using several weighting functions and, as a result, propose various
neural orthogonal survival learners. Some of these coincide with existing
survival learners (including survival versions of the DR- and R-learner), while
others are novel and further robust w.r.t. low overlap regimes specific to the
survival setting (i.e., survival overlap and censoring overlap). We then
empirically verify the effectiveness of our learners for HTE estimation in
different low-overlap regimes through numerical experiments. In sum, we provide
practitioners with a large toolbox of learners that can be used for randomized
and observational studies with censored time-to-event data.

### 7. [Attention-based clustering](http://arxiv.org/pdf/2505.13112v1)

Authors: Rodrigo Maulen-Soto, Claire Boyer, Pierre Marion

Transformers have emerged as a powerful neural network architecture capable
of tackling a wide range of learning tasks. In this work, we provide a
theoretical analysis of their ability to automatically extract structure from
data in an unsupervised setting. In particular, we demonstrate their
suitability for clustering when the input data is generated from a Gaussian
mixture model. To this end, we study a simplified two-head attention layer and
define a population risk whose minimization with unlabeled data drives the head
parameters to align with the true mixture centroids.

### 8. [Parallel Layer Normalization for Universal Approximation](http://arxiv.org/pdf/2505.13142v1)

Authors: Yunhao Ni, Yuhe Liu, Wenxin Sun, Yitong Tang, Yuxin Guo, Peilin Feng, Wenjun Wu, Lei Huang

Universal approximation theorem (UAT) is a fundamental theory for deep neural
networks (DNNs), demonstrating their powerful representation capacity to
represent and approximate any function. The analyses and proofs of UAT are
based on traditional network with only linear and nonlinear activation
functions, but omitting normalization layers, which are commonly employed to
enhance the training of modern networks. This paper conducts research on UAT of
DNNs with normalization layers for the first time. We theoretically prove that
an infinitely wide network -- composed solely of parallel layer normalization
(PLN) and linear layers -- has universal approximation capacity. Additionally,
we investigate the minimum number of neurons required to approximate
$L$-Lipchitz continuous functions, with a single hidden-layer network. We
compare the approximation capacity of PLN with traditional activation functions
in theory. Different from the traditional activation functions, we identify
that PLN can act as both activation function and normalization in deep neural
networks at the same time. We also find that PLN can improve the performance
when replacing LN in transformer architectures, which reveals the potential of
PLN used in neural architectures.

### 9. [Diffusion Models with Double Guidance: Generate with aggregated datasets](http://arxiv.org/pdf/2505.13213v1)

Authors: Yanfeng Yang, Kenji Fukumizu

Creating large-scale datasets for training high-performance generative models
is often prohibitively expensive, especially when associated attributes or
annotations must be provided. As a result, merging existing datasets has become
a common strategy. However, the sets of attributes across datasets are often
inconsistent, and their naive concatenation typically leads to block-wise
missing conditions. This presents a significant challenge for conditional
generative modeling when the multiple attributes are used jointly as
conditions, thereby limiting the model's controllability and applicability. To
address this issue, we propose a novel generative approach, Diffusion Model
with Double Guidance, which enables precise conditional generation even when no
training samples contain all conditions simultaneously. Our method maintains
rigorous control over multiple conditions without requiring joint annotations.
We demonstrate its effectiveness in molecular and image generation tasks, where
it outperforms existing baselines both in alignment with target conditional
distributions and in controllability under missing condition settings.

### 10. [Reconstructing Physics-Informed Machine Learning for Traffic Flow Modeling: a Multi-Gradient Descent and Pareto Learning Approach](http://arxiv.org/pdf/2505.13241v1)

Authors: Yuan-Zheng Lei, Yaobang Gong, Dianwei Chen, Yao Cheng, Xianfeng Terry Yang

Physics-informed machine learning (PIML) is crucial in modern traffic flow
modeling because it combines the benefits of both physics-based and data-driven
approaches. In conventional PIML, physical information is typically
incorporated by constructing a hybrid loss function that combines data-driven
loss and physics loss through linear scalarization. The goal is to find a
trade-off between these two objectives to improve the accuracy of model
predictions. However, from a mathematical perspective, linear scalarization is
limited to identifying only the convex region of the Pareto front, as it treats
data-driven and physics losses as separate objectives. Given that most PIML
loss functions are non-convex, linear scalarization restricts the achievable
trade-off solutions. Moreover, tuning the weighting coefficients for the two
loss components can be both time-consuming and computationally challenging. To
address these limitations, this paper introduces a paradigm shift in PIML by
reformulating the training process as a multi-objective optimization problem,
treating data-driven loss and physics loss independently. We apply several
multi-gradient descent algorithms (MGDAs), including traditional multi-gradient
descent (TMGD) and dual cone gradient descent (DCGD), to explore the Pareto
front in this multi-objective setting. These methods are evaluated on both
macroscopic and microscopic traffic flow models. In the macroscopic case, MGDAs
achieved comparable performance to traditional linear scalarization methods.
Notably, in the microscopic case, MGDAs significantly outperformed their
scalarization-based counterparts, demonstrating the advantages of a
multi-objective optimization approach in complex PIML scenarios.

