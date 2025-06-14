# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-06-13 17:08:50.328827 PST.

### Artificial Intelligence

### 1. [WGSR-Bench: Wargame-based Game-theoretic Strategic Reasoning Benchmark for Large Language Models](http://arxiv.org/pdf/2506.10264v1)

Authors: Qiyue Yin, Pei Xu, Qiaozhe Li, Shengda Liu, Shengqi Shen, Tong Wang, Yihong Han, Xiaonan Zhao, Likun Yang, Shiyue Cao, Shiyu Qiu, Yuxuan Liu, Shizhao Yu, Lei Cui, Chengxin Yan, Jie Sun, Xiangquan Tang, Kaiqi Huang

Recent breakthroughs in Large Language Models (LLMs) have led to a
qualitative leap in artificial intelligence' s performance on reasoning tasks,
particularly demonstrating remarkable capabilities in mathematical, symbolic,
and commonsense reasoning. However, as a critical component of advanced human
cognition, strategic reasoning, i.e., the ability to assess multi-agent
behaviors in dynamic environments, formulate action plans, and adapt
strategies, has yet to be systematically evaluated or modeled. To address this
gap, this paper introduces WGSR-Bench, the first strategy reasoning benchmark
for LLMs using wargame as its evaluation environment. Wargame, a quintessential
high-complexity strategic scenario, integrates environmental uncertainty,
adversarial dynamics, and non-unique strategic choices, making it an effective
testbed for assessing LLMs' capabilities in multi-agent decision-making, intent
inference, and counterfactual reasoning. WGSR-Bench designs test samples around
three core tasks, i.e., Environmental situation awareness, Opponent risk
modeling and Policy generation, which serve as the core S-POE architecture, to
systematically assess main abilities of strategic reasoning. Finally, an
LLM-based wargame agent is designed to integrate these parts for a
comprehensive strategy reasoning assessment. With WGSR-Bench, we hope to assess
the strengths and limitations of state-of-the-art LLMs in game-theoretic
strategic reasoning and to advance research in large model-driven strategic
intelligence.

### 2. [Closer to Language than Steam: AI as the Cognitive Engine of a New Productivity Revolution](http://arxiv.org/pdf/2506.10281v1)

Authors: Xinmin Fang, Lingfeng Tao, Zhengxiong Li

Artificial Intelligence (AI) is reframed as a cognitive engine driving a
novel productivity revolution distinct from the Industrial Revolution's
physical thrust. This paper develops a theoretical framing of AI as a cognitive
revolution akin to written language - a transformative augmentation of human
intellect rather than another mechanized tool. We compare AI's emergence to
historical leaps in information technology to show how it amplifies knowledge
work. Examples from various domains demonstrate AI's impact as a driver of
productivity in cognitive tasks. We adopt a multidisciplinary perspective
combining computer science advances with economic insights and sociological
perspectives on how AI reshapes work and society. Through conceptual
frameworks, we visualize the shift from manual to cognitive productivity. Our
central argument is that AI functions as an engine of cognition - comparable to
how human language revolutionized knowledge - heralding a new productivity
paradigm. We discuss how this revolution demands rethinking of skills,
organizations, and policies. This paper, balancing academic rigor with clarity,
concludes that AI's promise lies in complementing human cognitive abilities,
marking a new chapter in productivity evolution.

### 3. [Optimus-3: Towards Generalist Multimodal Minecraft Agents with Scalable Task Experts](http://arxiv.org/pdf/2506.10357v1)

Authors: Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Weili Guan, Dongmei Jiang, Liqiang Nie

Recently, agents based on multimodal large language models (MLLMs) have
achieved remarkable progress across various domains. However, building a
generalist agent with capabilities such as perception, planning, action,
grounding, and reflection in open-world environments like Minecraft remains
challenges: insufficient domain-specific data, interference among heterogeneous
tasks, and visual diversity in open-world settings. In this paper, we address
these challenges through three key contributions. 1) We propose a
knowledge-enhanced data generation pipeline to provide scalable and
high-quality training data for agent development. 2) To mitigate interference
among heterogeneous tasks, we introduce a Mixture-of-Experts (MoE) architecture
with task-level routing. 3) We develop a Multimodal Reasoning-Augmented
Reinforcement Learning approach to enhance the agent's reasoning ability for
visual diversity in Minecraft. Built upon these innovations, we present
Optimus-3, a general-purpose agent for Minecraft. Extensive experimental
results demonstrate that Optimus-3 surpasses both generalist multimodal large
language models and existing state-of-the-art agents across a wide range of
tasks in the Minecraft environment. Project page:
https://cybertronagent.github.io/Optimus-3.github.io/

### 4. [NeuroPAL: Punctuated Anytime Learning with Neuroevolution for Macromanagement in Starcraft: Brood War](http://arxiv.org/pdf/2506.10384v1)

Authors: Jim O'Connor, Yeonghun Lee, Gary B Parker

StarCraft: Brood War remains a challenging benchmark for artificial
intelligence research, particularly in the domain of macromanagement, where
long-term strategic planning is required. Traditional approaches to StarCraft
AI rely on rule-based systems or supervised deep learning, both of which face
limitations in adaptability and computational efficiency. In this work, we
introduce NeuroPAL, a neuroevolutionary framework that integrates
Neuroevolution of Augmenting Topologies (NEAT) with Punctuated Anytime Learning
(PAL) to improve the efficiency of evolutionary training. By alternating
between frequent, low-fidelity training and periodic, high-fidelity
evaluations, PAL enhances the sample efficiency of NEAT, enabling agents to
discover effective strategies in fewer training iterations. We evaluate
NeuroPAL in a fixed-map, single-race scenario in StarCraft: Brood War and
compare its performance to standard NEAT-based training. Our results show that
PAL significantly accelerates the learning process, allowing the agent to reach
competitive levels of play in approximately half the training time required by
NEAT alone. Additionally, the evolved agents exhibit emergent behaviors such as
proxy barracks placement and defensive building optimization, strategies
commonly used by expert human players. These findings suggest that structured
evaluation mechanisms like PAL can enhance the scalability and effectiveness of
neuroevolution in complex real-time strategy environments.

### 5. [Mirage-1: Augmenting and Updating GUI Agent with Hierarchical Multimodal Skills](http://arxiv.org/pdf/2506.10387v1)

Authors: Yuquan Xie, Zaijing Li, Rui Shao, Gongwei Chen, Kaiwen Zhou, Yinchuan Li, Dongmei Jiang, Liqiang Nie

Recent efforts to leverage the Multi-modal Large Language Model (MLLM) as GUI
agents have yielded promising outcomes. However, these agents still struggle
with long-horizon tasks in online environments, primarily due to insufficient
knowledge and the inherent gap between offline and online domains. In this
paper, inspired by how humans generalize knowledge in open-ended environments,
we propose a Hierarchical Multimodal Skills (HMS) module to tackle the issue of
insufficient knowledge. It progressively abstracts trajectories into execution
skills, core skills, and ultimately meta-skills, providing a hierarchical
knowledge structure for long-horizon task planning. To bridge the domain gap,
we propose the Skill-Augmented Monte Carlo Tree Search (SA-MCTS) algorithm,
which efficiently leverages skills acquired in offline environments to reduce
the action search space during online tree exploration. Building on HMS, we
propose Mirage-1, a multimodal, cross-platform, plug-and-play GUI agent. To
validate the performance of Mirage-1 in real-world long-horizon scenarios, we
constructed a new benchmark, AndroidLH. Experimental results show that Mirage-1
outperforms previous agents by 32\%, 19\%, 15\%, and 79\% on AndroidWorld,
MobileMiniWob++, Mind2Web-Live, and AndroidLH, respectively. Project page:
https://cybertronagent.github.io/Mirage-1.github.io/

### 6. [OIBench: Benchmarking Strong Reasoning Models with Olympiad in Informatics](http://arxiv.org/pdf/2506.10481v1)

Authors: Yaoming Zhu, Junxin Wang, Yiyang Li, Lin Qiu, ZongYu Wang, Jun Xu, Xuezhi Cao, Yuhuai Wei, Mingshi Wang, Xunliang Cai, Rong Ma

As models become increasingly sophisticated, conventional algorithm
benchmarks are increasingly saturated, underscoring the need for more
challenging benchmarks to guide future improvements in algorithmic reasoning.
This paper introduces OIBench, a high-quality, private, and challenging
olympiad-level informatics dataset comprising 250 carefully curated original
problems. We detail the construction methodology of the benchmark, ensuring a
comprehensive assessment across various programming paradigms and complexities,
and we demonstrate its contamination-resistant properties via experiments. We
propose Time/Space Completion Curves for finer-grained efficiency analysis and
enable direct human-model comparisons through high-level participant
evaluations. Our experiments reveal that while open-source models lag behind
closed-source counterparts, current SOTA models already outperform most human
participants in both correctness and efficiency, while still being suboptimal
compared to the canonical solutions. By releasing OIBench as a fully
open-source resource (https://huggingface.co/datasets/AGI-Eval/OIBench), we
hope this benchmark will contribute to advancing code reasoning capabilities
for future LLMs.

### 7. [Think before You Simulate: Symbolic Reasoning to Orchestrate Neural Computation for Counterfactual Question Answering](http://arxiv.org/pdf/2506.10753v1)

Authors: Adam Ishay, Zhun Yang, Joohyung Lee, Ilgu Kang, Dongjae Lim

Causal and temporal reasoning about video dynamics is a challenging problem.
While neuro-symbolic models that combine symbolic reasoning with neural-based
perception and prediction have shown promise, they exhibit limitations,
especially in answering counterfactual questions. This paper introduces a
method to enhance a neuro-symbolic model for counterfactual reasoning,
leveraging symbolic reasoning about causal relations among events. We define
the notion of a causal graph to represent such relations and use Answer Set
Programming (ASP), a declarative logic programming method, to find how to
coordinate perception and simulation modules. We validate the effectiveness of
our approach on two benchmarks, CLEVRER and CRAFT. Our enhancement achieves
state-of-the-art performance on the CLEVRER challenge, significantly
outperforming existing models. In the case of the CRAFT benchmark, we leverage
a large pre-trained language model, such as GPT-3.5 and GPT-4, as a proxy for a
dynamics simulator. Our findings show that this method can further improve its
performance on counterfactual questions by providing alternative prompts
instructed by symbolic causal reasoning.

### 8. [GenPlanX. Generation of Plans and Execution](http://arxiv.org/pdf/2506.10897v1)

Authors: Daniel Borrajo, Giuseppe Canonaco, Tomás de la Rosa, Alfredo Garrachón, Sriram Gopalakrishnan, Simerjot Kaur, Marianela Morales, Sunandita Patra, Alberto Pozanco, Keshav Ramani, Charese Smiley, Pietro Totis, Manuela Veloso

Classical AI Planning techniques generate sequences of actions for complex
tasks. However, they lack the ability to understand planning tasks when
provided using natural language. The advent of Large Language Models (LLMs) has
introduced novel capabilities in human-computer interaction. In the context of
planning tasks, LLMs have shown to be particularly good in interpreting human
intents among other uses. This paper introduces GenPlanX that integrates LLMs
for natural language-based description of planning tasks, with a classical AI
planning engine, alongside an execution and monitoring framework. We
demonstrate the efficacy of GenPlanX in assisting users with office-related
tasks, highlighting its potential to streamline workflows and enhance
productivity through seamless human-AI collaboration.

### 9. [Extended Creativity: A Conceptual Framework for Understanding Human-AI Creative Relations](http://arxiv.org/pdf/2506.10249v1)

Authors: Andrea Gaggioli, Sabrina Bartolotta, Andrea Ubaldi, Katusha Gerardini, Eleonora Diletta Sarcinella, Alice Chirico

Artificial Intelligence holds significant potential to enhance human
creativity. However, achieving this vision requires a clearer understanding of
how such enhancement can be effectively realized. Adopting the perspective of
distributed creativity, we identify three primary modes through which AI can
contribute to creative processes: Support, where AI acts as a tool; Synergy,
where AI and humans collaborate in complementary ways; and Symbiosis, where
human and AI cognition become so integrated that they form a unified creative
system. These modes are defined along two key dimensions: the level of
technical autonomy exhibited by the AI system and the degree of perceived
agency attributed to it. We examine how each configuration influences different
levels of creativity - from everyday problem-solving to paradigm-shifting
innovation - and discuss the theoretical, ethical, and design implications.

### 10. [RT-VC: Real-Time Zero-Shot Voice Conversion with Speech Articulatory Coding](http://arxiv.org/pdf/2506.10289v1)

Authors: Yisi Liu, Chenyang Wang, Hanjo Kim, Raniya Khan, Gopala Anumanchipalli

Voice conversion has emerged as a pivotal technology in numerous applications
ranging from assistive communication to entertainment. In this paper, we
present RT-VC, a zero-shot real-time voice conversion system that delivers
ultra-low latency and high-quality performance. Our approach leverages an
articulatory feature space to naturally disentangle content and speaker
characteristics, facilitating more robust and interpretable voice
transformations. Additionally, the integration of differentiable digital signal
processing (DDSP) enables efficient vocoding directly from articulatory
features, significantly reducing conversion latency. Experimental evaluations
demonstrate that, while maintaining synthesis quality comparable to the current
state-of-the-art (SOTA) method, RT-VC achieves a CPU latency of 61.4 ms,
representing a 13.3\% reduction in latency.

### Hardware Architecture

### 1. [CarbonSet: A Dataset to Analyze Trends and Benchmark the Sustainability of CPUs and GPUs](http://arxiv.org/pdf/2506.10373v1)

Authors: Jiajun Hu, Chetan Choppali Sudarshan, Vidya A. Chhabria, Aman Arora

Over the years, the chip industry has consistently developed high-performance
processors to address the increasing demands across diverse applications.
However, the rapid expansion of chip production has significantly increased
carbon emissions, raising critical concerns about environmental sustainability.
While researchers have previously modeled the carbon footprint (CFP) at both
system and processor levels, a holistic analysis of sustainability trends
encompassing the entire chip lifecycle remains lacking. This paper presents
CarbonSet, a comprehensive dataset integrating sustainability and performance
metrics for CPUs and GPUs over the past decade. CarbonSet aims to benchmark and
assess the design of next-generation processors. Leveraging this dataset, we
conducted detailed analysis of flagship processors' sustainability trends over
the last decade. This paper further highlights that modern processors are not
yet sustainably designed, with total carbon emissions increasing more than
50$\times$ in the past three years due to the surging demand driven by the AI
boom. Power efficiency remains a significant concern, while advanced process
nodes pose new challenges requiring to effectively amortize the dramatically
increased manufacturing carbon emissions.

### 2. [EasyDRAM: An FPGA-based Infrastructure for Fast and Accurate End-to-End Evaluation of Emerging DRAM Techniques](http://arxiv.org/pdf/2506.10441v1)

Authors: Oğuzhan Canpolat, Ataberk Olgun, David Novo, Oğuz Ergin, Onur Mutlu

DRAM is a critical component of modern computing systems. Recent works
propose numerous techniques (that we call DRAM techniques) to enhance
DRAM-based computing systems' throughput, reliability, and computing
capabilities (e.g., in-DRAM bulk data copy). Evaluating the system-wide
benefits of DRAM techniques is challenging as they often require modifications
across multiple layers of the computing stack. Prior works propose FPGA-based
platforms for rapid end-to-end evaluation of DRAM techniques on real DRAM
chips. Unfortunately, existing platforms fall short in two major aspects: (1)
they require deep expertise in hardware description languages, limiting
accessibility; and (2) they are not designed to accurately model modern
computing systems.
  We introduce EasyDRAM, an FPGA-based framework for rapid and accurate
end-to-end evaluation of DRAM techniques on real DRAM chips. EasyDRAM overcomes
the main drawbacks of prior FPGA-based platforms with two key ideas. First,
EasyDRAM removes the need for hardware description language expertise by
enabling developers to implement DRAM techniques using a high-level language
(C++). At runtime, EasyDRAM executes the software-defined memory system design
in a programmable memory controller. Second, EasyDRAM tackles a fundamental
challenge in accurately modeling modern systems: real processors typically
operate at higher clock frequencies than DRAM, a disparity that is difficult to
replicate on FPGA platforms. EasyDRAM addresses this challenge by decoupling
the processor-DRAM interface and advancing the system state using a novel
technique we call time scaling, which faithfully captures the timing behavior
of the modeled system.
  We believe and hope that EasyDRAM will enable innovative ideas in memory
system design to rapidly come to fruition. To aid future research EasyDRAM
implementation is open sourced at https://github.com/CMU-SAFARI/EasyDRAM.

### 3. [Towards Zero-Stall Matrix Multiplication on Energy-Efficient RISC-V Clusters for Machine Learning Acceleration](http://arxiv.org/pdf/2506.10921v1)

Authors: Luca Colagrande, Lorenzo Leone, Maximilian Coco, Andrei Deaconeasa, Luca Benini

The growing computational demands of machine learning (ML) workloads have
driven the design of ML accelerators aiming at an optimal tradeoff between
efficiency and flexibility. A widely explored architecture for flexible ML
accelerators is based on clusters of lightweight instruction processors sharing
multi-banked L1 memory, augmented with specialized instruction extensions for
key ML-related computations, such as matrix multiplication (matmul). However,
instruction extensions should be coupled with microarchitectural optimizations
that remove inefficiencies due to control flow (loop handling) and memory
access, without drastically increasing processor complexity. Moving from a
state-of-the-art (SoA) ML accelerator cluster based on RISC-V processors, we
propose a low-overhead optimized microarchitecture that eliminates these
inefficiencies almost entirely while retaining programmability. We introduce
"zero-overhead loop nests" to remove control overheads, and a "zero-conflict
memory subsystem", leveraging a novel double-buffering-aware interconnect, to
eliminate bank conflicts in L1 memory. With these enhancements, we attain
near-ideal utilizations between 96.1% and 99.4%, achieving 11% performance and
8% energy efficiency improvements over the baseline SoA RISC-V cluster. We
demonstrate comparable utilizations and performance to a specialized SoA
accelerator, with only 12% difference in energy efficiency, while providing a
fully-programmable general-purpose solution supporting a significantly wider
range of workloads.

### 4. [Synchronization for Fault-Tolerant Quantum Computers](http://arxiv.org/pdf/2506.10258v1)

Authors: Satvik Maurya, Swamit Tannu

Quantum Error Correction (QEC) codes store information reliably in logical
qubits by encoding them in a larger number of less reliable qubits. The surface
code, known for its high resilience to physical errors, is a leading candidate
for fault-tolerant quantum computing (FTQC). Logical qubits encoded with the
surface code can be in different phases of their syndrome generation cycle,
thereby introducing desynchronization in the system. This can occur due to the
production of non-Clifford states, dropouts due to fabrication defects, and the
use of other QEC codes with the surface code to reduce resource requirements.
Logical operations require the syndrome generation cycles of the logical qubits
involved to be synchronized. This requires the leading qubit to pause or slow
down its cycle, allowing more errors to accumulate before the next cycle,
thereby increasing the risk of uncorrectable errors.
  To synchronize the syndrome generation cycles of logical qubits, we define
three policies - Passive, Active, and Hybrid. The Passive policy is the
baseline, and the simplest, wherein the leading logical qubits idle until they
are synchronized with the remaining logical qubits. On the other hand, the
Active policy aims to slow the leading logical qubits down gradually, by
inserting short idle periods before multiple code cycles. This approach reduces
the logical error rate (LER) by up to 2.4x compared to the Passive policy. The
Hybrid policy further reduces the LER by up to 3.4x by reducing the
synchronization slack and running a few additional rounds of error correction.
Furthermore, the reduction in the logical error rate with the proposed
synchronization policies enables a speedup in decoding latency of up to 2.2x
with a circuit-level noise model.

### 5. [Scalable Software Testing in Fast Virtual Platforms: Leveraging SystemC, QEMU and Containerization](http://arxiv.org/pdf/2506.10624v1)

Authors: Lukas Jünger, Jan Henrik Weinstock, Tim Kraus

The ever-increasing complexity of HW/SW systems presents a persistent
challenge, particularly in safety-critical domains like automotive, where
extensive testing is imperative. However, the availability of hardware often
lags behind, hindering early-stage software development. To address this,
Virtual Platforms (VPs) based on the SystemC TLM-2.0 standard have emerged as a
pivotal solution, enabling pre-silicon execution and testing of unmodified
target software. In this study, we propose an approach leveraging
containerization to encapsulate VPs in order to reduce environment dependencies
and enable cloud deployment for fast, parallelized test execution, as well as
open-source VP technologies such as QEMU and VCML to obviate the need for seat
licenses. To demonstrate the efficacy of our approach, we present an Artificial
Intelligence (AI) accelerator VP case study. Through our research, we offer a
robust solution to address the challenges posed by the complexity of HW/SW
systems, with practical implications for accelerating HW/SW co-development.

### 6. [MARS: Processing-In-Memory Acceleration of Raw Signal Genome Analysis Inside the Storage Subsystem](http://arxiv.org/pdf/2506.10931v1)

Authors: Melina Soysal, Konstantina Koliogeorgi, Can Firtina, Nika Mansouri Ghiasi, Rakesh Nadig, Haiyu Mayo, Geraldo F. Oliveira, Yu Liang, Klea Zambaku, Mohammad Sadrosadati, Onur Mutlu

Raw signal genome analysis (RSGA) has emerged as a promising approach to
enable real-time genome analysis by directly analyzing raw electrical signals.
However, rapid advancements in sequencing technologies make it increasingly
difficult for software-based RSGA to match the throughput of raw signal
generation. This paper demonstrates that while hardware acceleration techniques
can significantly accelerate RSGA, the high volume of genomic data shifts the
performance and energy bottleneck from computation to I/O data movement. As
sequencing throughput increases, I/O overhead becomes the main contributor to
both runtime and energy consumption. Therefore, there is a need to design a
high-performance, energy-efficient system for RSGA that can both alleviate the
data movement bottleneck and provide large acceleration capabilities. We
propose MARS, a storage-centric system that leverages the heterogeneous
resources within modern storage systems (e.g., storage-internal DRAM, storage
controller, flash chips) alongside their large storage capacity to tackle both
data movement and computational overheads of RSGA in an area-efficient and
low-cost manner. MARS accelerates RSGA through a novel hardware/software
co-design approach. First, MARS modifies the RSGA pipeline via two filtering
mechanisms and a quantization scheme, reducing hardware demands and optimizing
for in-storage execution. Second, MARS accelerates the RSGA steps directly
within the storage by leveraging both Processing-Near-Memory and
Processing-Using-Memory paradigms. Third, MARS orchestrates the execution of
all steps to fully exploit in-storage parallelism and minimize data movement.
Our evaluation shows that MARS outperforms basecalling-based software and
hardware-accelerated state-of-the-art read mapping pipelines by 93x and 40x, on
average across different datasets, while reducing their energy consumption by
427x and 72x.

### Computational Complexity

### 1. [The Alignment Trap: Complexity Barriers](http://arxiv.org/pdf/2506.10304v1)

Authors: Jasper Yao

We establish fundamental computational complexity barriers to verifying AI
safety as system capabilities scale. Our main results show that for AI systems
with expressiveness EXP$(m)$ above a critical threshold $\tau$, safety
verification requires exponential time and is coNP-complete. We formalize the
Capability-Risk Scaling (CRS) dynamic, which demonstrates how increasing AI
capability drives societal safety requirements toward perfection, creating an
inescapable tension with verification complexity. Through four core theorems,
we prove that (1) verification complexity grows exponentially with system
expressiveness, (2) safe policies comprise at most a $2^{-2^m}$ fraction of the
policy space, (3) no finite set of alignment techniques can provide universal
coverage, and (4) robust safety properties form measure-zero sets for neural
networks. These results characterize an "intractability gap" where practical
safety requirements fall within the region of computational intractability. We
conclude by presenting a strategic trilemma: AI development must either
constrain system complexity to maintain verifiable safety, accept unverifiable
risks while scaling capabilities, or develop fundamentally new safety paradigms
beyond verification. Our work provides the first systematic
complexity-theoretic analysis of AI alignment and establishes rigorous bounds
that any safety approach must confront. A formal verification of the core
theorems in Lean4 is currently in progress.

### 2. [Minimality and computability of languages of G-shifts](http://arxiv.org/pdf/2506.10610v1)

Authors: Djamel Eddine Amir, Benjamin Hellouin de Menibus

Motivated by the notion of strong computable type for sets in computable
analysis, we define the notion of strong computable type for $G$-shifts, where
$G$ is a finitely generated group with decidable word problem. A $G$-shift has
strong computable type if one can compute its language from the complement of
its language. We obtain a characterization of $G$-shifts with strong computable
type in terms of a notion of minimality with respect to properties with a
bounded computational complexity. We provide a self-contained direct proof, and
also explain how this characterization can be obtained from an existing similar
characterization for sets by Amir and Hoyrup, and discuss its connexions with
results by Jeandel on closure spaces. We apply this characterization to several
classes of shifts that are minimal with respect to specific properties. This
provides a unifying approach that not only generalizes many existing results
but also has the potential to yield new findings effortlessly. In contrast to
the case of sets, we prove that strong computable type for G-shifts is
preserved under products. We conclude by discussing some generalizations and
future directions.

### 3. [Computational Complexity of Statistics: New Insights from Low-Degree Polynomials](http://arxiv.org/pdf/2506.10748v1)

Authors: Alexander S. Wein

This is a survey on the use of low-degree polynomials to predict and explain
the apparent statistical-computational tradeoffs in a variety of average-case
computational problems. In a nutshell, this framework measures the complexity
of a statistical task by the minimum degree that a polynomial function must
have in order to solve it. The main goals of this survey are to (1) describe
the types of problems where the low-degree framework can be applied,
encompassing questions of detection (hypothesis testing), recovery
(estimation), and more; (2) discuss some philosophical questions surrounding
the interpretation of low-degree lower bounds, and notably the extent to which
they should be treated as evidence for inherent computational hardness; (3)
explore the known connections between low-degree polynomials and other related
approaches such as the sum-of-squares hierarchy and statistical query model;
and (4) give an overview of the mathematical tools used to prove low-degree
lower bounds. A list of open problems is also included.

### 4. [Landauer Principle and Thermodynamics of Computation](http://arxiv.org/pdf/2506.10876v1)

Authors: Pritam Chattopadhyay, Avijit Misra, Tanmoy Pandit, Goutam Paul

According to the Landauer principle, any logically irreversible process
accompanies entropy production, which results in heat dissipation in the
environment. Erasing of information, one of the primary logically irreversible
processes, has a lower bound on heat dissipated into the environment, called
the Landauer bound (LB). However, the practical erasure processes dissipate
much more heat than the LB. Recently, there have been a few experimental
investigations to reach this bound both in the classical and quantum domains.
There has also been a spate of activities to enquire about this LB in finite
time, with finite-size heat baths, non-Markovian and nonequilibrium environment
in the quantum regime where the effects of fluctuations and correlation of the
systems with the bath can no longer be ignored. This article provides a
comprehensive review of the recent progress on the Landauer bound, which serves
as a fundamental principle in the thermodynamics of computation. We also
provide a perspective for future endeavors in these directions. Furthermore, we
review the recent exploration toward establishing energetic bounds of a
computational process. We also review the thermodynamic aspects of error
correction, which is an indispensable part of information processing and
computations. In doing so, we briefly discuss the basics of these fields to
provide a complete picture.

### Computational Engineering

### 1. [PDESpectralRefiner: Achieving More Accurate Long Rollouts with Spectral Adjustment](http://arxiv.org/pdf/2506.10711v1)

Authors: Li Luo, Shangsong Liang

Generating accurate and stable long rollouts is a notorious challenge for
time-dependent PDEs (Partial Differential Equations). Recently, motivated by
the importance of high-frequency accuracy, a refiner model called PDERefiner
utilizes diffusion models to refine outputs for every time step, since the
denoising process could increase the correctness of modeling high frequency
part. For 1-D Kuramoto-Sivashinsky equation, refiner models can degrade the
amplitude of high frequency part better than not doing refinement process.
However, for some other cases, the spectrum might be more complicated. For
example, for a harder PDE like Navior-Stokes equation, diffusion models could
over-degrade the higher frequency part. This motivates us to release the
constraint that each frequency weighs the same. We enhance our refiner model
with doing adjustments on spectral space, which recovers Blurring diffusion
models. We developed a new v-prediction technique for Blurring diffusion
models, recovering the MSE training objective on the first refinement step. We
show that in this case, for different model backbones, such as U-Net and neural
operators, the outputs of PDE-SpectralRefiner are more accurate for both
one-step MSE loss and rollout loss.

### 2. [Spectral Analysis of Discretized Boundary Integral Operators in 3D: a High-Frequency Perspective](http://arxiv.org/pdf/2506.10880v1)

Authors: V. Giunzioni, A. Merlini, F. P. Andriulli

When modeling propagation and scattering phenomena using integral equations
discretized by the boundary element method, it is common practice to
approximate the boundary of the scatterer with a mesh comprising elements of
size approximately equal to a fraction of the wavelength $\lambda$ of the
incident wave, e.g., $\lambda/10$. In this work, by analyzing the spectra of
the operator matrices, we show a discrepancy with respect to the continuous
operators which grows with the simulation frequency, challenging the common
belief that the aforementioned widely used discretization approach is
sufficient to maintain the accuracy of the solution constant when increasing
the frequency.

### Computation and Language

### 1. [Beyond the Battlefield: Framing Analysis of Media Coverage in Conflict Reporting](http://arxiv.org/pdf/2506.10421v1)

Authors: Avneet Kaur, Arnav Arora

Framing used by news media, especially in times of conflict, can have
substantial impact on readers' opinion, potentially aggravating the conflict
itself. Current studies on the topic of conflict framing have limited insights
due to their qualitative nature or only look at surface level generic frames
without going deeper. In this work, we identify indicators of war and peace
journalism, as outlined by prior work in conflict studies, in a corpus of news
articles reporting on the Israel-Palestine war. For our analysis, we use
computational approaches, using a combination of frame semantics and large
language models to identify both communicative framing and its connection to
linguistic framing. Our analysis reveals a higher focus on war based reporting
rather than peace based. We also show substantial differences in reporting
across the US, UK, and Middle Eastern news outlets in framing who the assailant
and victims of the conflict are, surfacing biases within the media.

### 2. [Fast on the Easy, Deep on the Hard: Efficient Reasoning via Powered Length Penalty](http://arxiv.org/pdf/2506.10446v1)

Authors: Zehui Ling, Deshu Chen, Hongwei Zhang, Yifeng Jiao, Xin Guo, Yuan Cheng

Large language models (LLMs) have demonstrated significant advancements in
reasoning capabilities, performing well on various challenging benchmarks.
Techniques like Chain-of-Thought prompting have been introduced to further
improve reasoning. However, these approaches frequently generate longer
outputs, which in turn increase computational latency. Although some methods
use reinforcement learning to shorten reasoning, they often apply uniform
penalties without considering the problem's complexity, leading to suboptimal
outcomes. In this study, we seek to enhance the efficiency of LLM reasoning by
promoting conciseness for simpler problems while preserving sufficient
reasoning for more complex ones for accuracy, thus improving the model's
overall performance. Specifically, we manage the model's reasoning efficiency
by dividing the reward function and including a novel penalty for output
length. Our approach has yielded impressive outcomes in benchmark evaluations
across three datasets: GSM8K, MATH500, and AIME2024. For the comparatively
simpler datasets GSM8K and MATH500, our method has effectively shortened output
lengths while preserving or enhancing accuracy. On the more demanding AIME2024
dataset, our approach has resulted in improved accuracy.

### 3. [Table-Text Alignment: Explaining Claim Verification Against Tables in Scientific Papers](http://arxiv.org/pdf/2506.10486v1)

Authors: Xanh Ho, Sunisth Kumar, Yun-Ang Wu, Florian Boudin, Atsuhiro Takasu, Akiko Aizawa

Scientific claim verification against tables typically requires predicting
whether a claim is supported or refuted given a table. However, we argue that
predicting the final label alone is insufficient: it reveals little about the
model's reasoning and offers limited interpretability. To address this, we
reframe table-text alignment as an explanation task, requiring models to
identify the table cells essential for claim verification. We build a new
dataset by extending the SciTab benchmark with human-annotated cell-level
rationales. Annotators verify the claim label and highlight the minimal set of
cells needed to support their decision. After the annotation process, we
utilize the collected information and propose a taxonomy for handling ambiguous
cases. Our experiments show that (i) incorporating table alignment information
improves claim verification performance, and (ii) most LLMs, while often
predicting correct labels, fail to recover human-aligned rationales, suggesting
that their predictions do not stem from faithful reasoning.

### 4. [Surface Fairness, Deep Bias: A Comparative Study of Bias in Language Models](http://arxiv.org/pdf/2506.10491v1)

Authors: Aleksandra Sorokovikova, Pavel Chizhov, Iuliia Eremenko, Ivan P. Yamshchikov

Modern language models are trained on large amounts of data. These data
inevitably include controversial and stereotypical content, which contains all
sorts of biases related to gender, origin, age, etc. As a result, the models
express biased points of view or produce different results based on the
assigned personality or the personality of the user. In this paper, we
investigate various proxy measures of bias in large language models (LLMs). We
find that evaluating models with pre-prompted personae on a multi-subject
benchmark (MMLU) leads to negligible and mostly random differences in scores.
However, if we reformulate the task and ask a model to grade the user's answer,
this shows more significant signs of bias. Finally, if we ask the model for
salary negotiation advice, we see pronounced bias in the answers. With the
recent trend for LLM assistant memory and personalization, these problems open
up from a different angle: modern LLM users do not need to pre-prompt the
description of their persona since the model already knows their
socio-demographics.

### 5. [Unsupervised Protoform Reconstruction through Parsimonious Rule-guided Heuristics and Evolutionary Search](http://arxiv.org/pdf/2506.10614v1)

Authors: Promise Dodzi Kpoglu

We propose an unsupervised method for the reconstruction of protoforms i.e.,
ancestral word forms from which modern language forms are derived. While prior
work has primarily relied on probabilistic models of phonological edits to
infer protoforms from cognate sets, such approaches are limited by their
predominantly data-driven nature. In contrast, our model integrates data-driven
inference with rule-based heuristics within an evolutionary optimization
framework. This hybrid approach leverages on both statistical patterns and
linguistically motivated constraints to guide the reconstruction process. We
evaluate our method on the task of reconstructing Latin protoforms using a
dataset of cognates from five Romance languages. Experimental results
demonstrate substantial improvements over established baselines across both
character-level accuracy and phonological plausibility metrics.

### 6. [Spelling-out is not Straightforward: LLMs' Capability of Tokenization from Token to Characters](http://arxiv.org/pdf/2506.10641v1)

Authors: Tatsuya Hiraoka, Kentaro Inui

Large language models (LLMs) can spell out tokens character by character with
high accuracy, yet they struggle with more complex character-level tasks, such
as identifying compositional subcomponents within tokens. In this work, we
investigate how LLMs internally represent and utilize character-level
information during the spelling-out process. Our analysis reveals that,
although spelling out is a simple task for humans, it is not handled in a
straightforward manner by LLMs. Specifically, we show that the embedding layer
does not fully encode character-level information, particularly beyond the
first character. As a result, LLMs rely on intermediate and higher Transformer
layers to reconstruct character-level knowledge, where we observe a distinct
"breakthrough" in their spelling behavior. We validate this mechanism through
three complementary analyses: probing classifiers, identification of knowledge
neurons, and inspection of attention weights.

### 7. [Inferring Adjective Hypernyms with Language Models to Increase the Connectivity of Open English Wordnet](http://arxiv.org/pdf/2506.10715v1)

Authors: Lorenzo Augello, John P. McCrae

Open English Wordnet is a key resource published in OntoLex-lemon as part of
the linguistic linked open data cloud. There are, however, many links missing
in the resource, and in this paper, we look at how we can establish hypernymy
between adjectives. We present a theoretical discussion of the hypernymy
relation and how it differs for adjectives in contrast to nouns and verbs. We
develop a new resource for adjective hypernymy and fine-tune large language
models to predict adjective hypernymy, showing that the methodology of
TaxoLLaMa can be adapted to this task.

### 8. [One Tokenizer To Rule Them All: Emergent Language Plasticity via Multilingual Tokenizers](http://arxiv.org/pdf/2506.10766v1)

Authors: Diana Abagyan, Alejandro R. Salamanca, Andres Felipe Cruz-Salinas, Kris Cao, Hangyu Lin, Acyr Locatelli, Marzieh Fadaee, Ahmet Üstün, Sara Hooker

Pretraining massively multilingual Large Language Models (LLMs) for many
languages at once is challenging due to limited model capacity, scarce
high-quality data, and compute constraints. Moreover, the lack of language
coverage of the tokenizer makes it harder to address the gap for new languages
purely at the post-training stage. In this work, we study what relatively cheap
interventions early on in training improve "language plasticity", or adaptation
capabilities of the model post-training to new languages. We focus on tokenizer
design and propose using a universal tokenizer that is trained for more
languages than the primary pretraining languages to enable efficient adaptation
in expanding language coverage after pretraining. Our systematic experiments
across diverse groups of languages and different training strategies show that
a universal tokenizer enables significantly higher language adaptation, with up
to 20.2% increase in win rates compared to tokenizers specific to pretraining
languages. Furthermore, a universal tokenizer also leads to better plasticity
towards languages that are completely unseen in the tokenizer and pretraining,
by up to 5% win rate gain. We achieve this adaptation to an expanded set of
languages with minimal compromise in performance on the majority of languages
included in pretraining.

### 9. [Different Questions, Different Models: Fine-Grained Evaluation of Uncertainty and Calibration in Clinical QA with LLMs](http://arxiv.org/pdf/2506.10769v1)

Authors: Alberto Testoni, Iacer Calixto

Accurate and well-calibrated uncertainty estimates are essential for
deploying large language models (LLMs) in high-stakes domains such as clinical
decision support. We present a fine-grained evaluation of uncertainty
estimation methods for clinical multiple-choice question answering, covering
ten open-source LLMs (general-purpose, biomedical, and reasoning models) across
two datasets, eleven medical specialties, and six question types. We compare
standard single-generation and sampling-based methods, and present a case study
exploring simple, single-pass estimators based on behavioral signals in
reasoning traces. These lightweight methods approach the performance of
Semantic Entropy while requiring only one generation. Our results reveal
substantial variation across specialties and question types, underscoring the
importance of selecting models based on both the nature of the question and
model-specific strengths.

### 10. [Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints](http://arxiv.org/pdf/2506.10800v1)

Authors: Wei Sun, Tingyu Qu, Mingxiao Li, Jesse Davis, Marie-Francine Moens

Efficiently updating multilingual knowledge in large language models (LLMs),
while preserving consistent factual representations across languages, remains a
long-standing and unresolved challenge. While deploying separate editing
systems for each language might seem viable, this approach incurs substantial
costs due to the need to manage multiple models. A more efficient solution
involves integrating knowledge updates across all languages into a unified
model. However, performing sequential edits across languages often leads to
destructive parameter interference, significantly degrading multilingual
generalization and the accuracy of injected knowledge. To address this
challenge, we propose LangEdit, a novel null-space constrained framework
designed to precisely isolate language-specific knowledge updates. The core
innovation of LangEdit lies in its ability to project parameter updates for
each language onto the orthogonal complement of previous updated subspaces.
This approach mathematically guarantees update independence while preserving
multilingual generalization capabilities. We conduct a comprehensive evaluation
across three model architectures, six languages, and four downstream tasks,
demonstrating that LangEdit effectively mitigates parameter interference and
outperforms existing state-of-the-art editing methods. Our results highlight
its potential for enabling efficient and accurate multilingual knowledge
updates in LLMs. The code is available at
https://github.com/VRCMF/LangEdit.git.

### Cryptography and Security

### 1. [A Comprehensive Survey of Unmanned Aerial Systems' Risks and Mitigation Strategies](http://arxiv.org/pdf/2506.10327v1)

Authors: Sharad Shrestha, Mohammed Ababneh, Satyajayant Misra, Henry M. Cathey, Jr., Roopa Vishwanathan, Matt Jansen, Jinhong Choi, Rakesh Bobba, Yeongjin Jang

In the last decade, the rapid growth of Unmanned Aircraft Systems (UAS) and
Unmanned Aircraft Vehicles (UAV) in communication, defense, and transportation
has increased. The application of UAS will continue to increase rapidly. This
has led researchers to examine security vulnerabilities in various facets of
UAS infrastructure and UAVs, which form a part of the UAS system to reinforce
these critical systems. This survey summarizes the cybersecurity
vulnerabilities in several phases of UAV deployment, the likelihood of each
vulnerability's occurrence, the impact of attacks, and mitigation strategies
that could be applied. We go beyond the state-of-the-art by taking a
comprehensive approach to enhancing UAS security by performing an analysis of
both UAS-specific and non-UAS-specific mitigation strategies that are
applicable within the UAS domain to define the lessons learned. We also present
relevant cybersecurity standards and their recommendations in the UAS context.
Despite the significant literature in UAS security and the relevance of
cyberphysical and networked systems security approaches from the past, which we
identify in the survey, we find several critical research gaps that require
further investigation. These form part of our discussions and recommendations
for the future exploration by our research community.

### 2. [Adaptive Chosen-Ciphertext Security of Distributed Broadcast Encryption](http://arxiv.org/pdf/2506.10338v1)

Authors: Kwangsu Lee

Distributed broadcast encryption (DBE) is a specific kind of broadcast
encryption (BE) where users independently generate their own public and private
keys, and a sender can efficiently create a ciphertext for a subset of users by
using the public keys of the subset users. Previously proposed DBE schemes have
been proven in the adaptive chosen-plaintext attack (CPA) security model and
have the disadvantage of requiring linear number of pairing operations when
verifying the public key of a user. In this paper, we propose an efficient DBE
scheme in bilinear groups and prove adaptive chosen-ciphertext attack (CCA)
security for the first time. To do this, we first propose a semi-static CCA
secure DBE scheme and prove the security under the $q$-Type assumption. Then,
by modifying the generic transformation of Gentry and Waters that converts a
semi-static CPA secure DBE scheme into an adaptive CPA secure DBE scheme to be
applied to CCA secure DBE schemes, we propose an adaptive CCA secure DBE scheme
and prove its adaptive CCA security. Our proposed DBE scheme is efficient
because it requires constant size ciphertexts, constant size private keys, and
linear size public keys, and the public key verification requires only a
constant number of pairing operations and efficient group membership checks.

### 3. [FicGCN: Unveiling the Homomorphic Encryption Efficiency from Irregular Graph Convolutional Networks](http://arxiv.org/pdf/2506.10399v1)

Authors: Zhaoxuan Kan, Husheng Han, Shangyi Shi, Tenghui Hua, Hang Lu, Xiaowei Li, Jianan Mu, Xing Hu

Graph Convolutional Neural Networks (GCNs) have gained widespread popularity
in various fields like personal healthcare and financial systems, due to their
remarkable performance. Despite the growing demand for cloud-based GCN
services, privacy concerns over sensitive graph data remain significant.
Homomorphic Encryption (HE) facilitates Privacy-Preserving Machine Learning
(PPML) by allowing computations to be performed on encrypted data. However, HE
introduces substantial computational overhead, particularly for GCN operations
that require rotations and multiplications in matrix products. The sparsity of
GCNs offers significant performance potential, but their irregularity
introduces additional operations that reduce practical gains. In this paper, we
propose FicGCN, a HE-based framework specifically designed to harness the
sparse characteristics of GCNs and strike a globally optimal balance between
aggregation and combination operations. FicGCN employs a latency-aware packing
scheme, a Sparse Intra-Ciphertext Aggregation (SpIntra-CA) method to minimize
rotation overhead, and a region-based data reordering driven by local adjacency
structure. We evaluated FicGCN on several popular datasets, and the results
show that FicGCN achieved the best performance across all tested datasets, with
up to a 4.10x improvement over the latest design.

### 4. [CyFence: Securing Cyber-Physical Controllers via Trusted Execution Environment](http://arxiv.org/pdf/2506.10638v1)

Authors: Stefano Longari, Alessandro Pozone, Jessica Leoni, Mario Polino, Michele Carminati, Mara Tanelli, Stefano Zanero

In the last decades, Cyber-physical Systems (CPSs) have experienced a
significant technological evolution and increased connectivity, at the cost of
greater exposure to cyber-attacks. Since many CPS are used in safety-critical
systems, such attacks entail high risks and potential safety harms. Although
several defense strategies have been proposed, they rarely exploit the
cyber-physical nature of the system. In this work, we exploit the nature of CPS
by proposing CyFence, a novel architecture that improves the resilience of
closed-loop control systems against cyber-attacks by adding a semantic check,
used to confirm that the system is behaving as expected. To ensure the security
of the semantic check code, we use the Trusted Execution Environment
implemented by modern processors. We evaluate CyFence considering a real-world
application, consisting of an active braking digital controller, demonstrating
that it can mitigate different types of attacks with a negligible computation
overhead.

### 5. [From IOCs to Group Profiles: On the Specificity of Threat Group Behaviors in CTI Knowledge Bases](http://arxiv.org/pdf/2506.10645v1)

Authors: Aakanksha Saha, Martina Lindorfer, Juan Caballero

Indicators of Compromise (IOCs) such as IP addresses, file hashes, and domain
names are commonly used for threat detection and attribution. However, IOCs
tend to be short-lived as they are easy to change. As a result, the
cybersecurity community is shifting focus towards more persistent behavioral
profiles, such as the Tactics, Techniques, and Procedures (TTPs) and the
software used by a threat group. However, the distinctiveness and completeness
of such behavioral profiles remain largely unexplored. In this work, we
systematically analyze threat group profiles built from two open cyber threat
intelligence (CTI) knowledge bases: MITRE ATT&CK and Malpedia. We first
investigate what fraction of threat groups have group-specific behaviors, i.e.,
behaviors used exclusively by a single group. We find that only 34% of threat
groups in ATT&CK have group-specific techniques. The software used by a threat
group proves to be more distinctive, with 73% of ATT&CK groups using
group-specific software. However, this percentage drops to 24% in the broader
Malpedia dataset. Next, we evaluate how group profiles improve when data from
both sources are combined. While coverage improves modestly, the proportion of
groups with group-specific behaviors remains under 30%. We then enhance
profiles by adding exploited vulnerabilities and additional techniques
extracted from more threat reports. Despite the additional information, 64% of
groups still lack any group-specific behavior. Our findings raise concerns on
the belief that behavioral profiles can replace IOCs in threat group
attribution.

### 6. [GOLIATH: A Decentralized Framework for Data Collection in Intelligent Transportation Systems](http://arxiv.org/pdf/2506.10665v1)

Authors: Davide Maffiola, Stefano Longari, Michele Carminati, Mara Tanelli, Stefano Zanero

Intelligent Transportation Systems (ITSs) technology has advanced during the
past years, and it is now used for several applications that require vehicles
to exchange real-time data, such as in traffic information management.
Traditionally, road traffic information has been collected using on-site
sensors. However, crowd-sourcing traffic information from onboard sensors or
smartphones has become a viable alternative. State-of-the-art solutions
currently follow a centralized model where only the service provider has
complete access to the collected traffic data and represent a single point of
failure and trust. In this paper, we propose GOLIATH, a blockchain-based
decentralized framework that runs on the In-Vehicle Infotainment (IVI) system
to collect real-time information exchanged between the network's participants.
Our approach mitigates the limitations of existing crowd-sourcing centralized
solutions by guaranteeing trusted information collection and exchange, fully
exploiting the intrinsic distributed nature of vehicles. We demonstrate its
feasibility in the context of vehicle positioning and traffic information
management. Each vehicle participating in the decentralized network shares its
position and neighbors' ones in the form of a transaction recorded on the
ledger, which uses a novel consensus mechanism to validate it. We design the
consensus mechanism resilient against a realistic set of adversaries that aim
to tamper or disable the communication. We evaluate the proposed framework in a
simulated (but realistic) environment, which considers different threats and
allows showing its robustness and safety properties.

### 7. [Commitment Schemes for Multi-Party Computation](http://arxiv.org/pdf/2506.10721v1)

Authors: Ioan Ionescu, Ruxandra F. Olimid

The paper presents an analysis of Commitment Schemes (CSs) used in
Multi-Party Computation (MPC) protocols. While the individual properties of CSs
and the guarantees offered by MPC have been widely studied in isolation, their
interrelation in concrete protocols and applications remains mostly
underexplored. This paper presents the relation between the two, with an
emphasis on (security) properties and their impact on the upper layer MPC. In
particular, we investigate how different types of CSs contribute to various MPC
constructions and their relation to real-life applications of MPC. The paper
can also serve as a tutorial for understanding the cryptographic interplay
between CS and MPC, making it accessible to both researchers and practitioners.
Our findings emphasize the importance of carefully selecting CS to meet the
adversarial and functional requirements of MPC, thereby aiming for more robust
and privacy-preserving cryptographic applications

### 8. [ObfusBFA: A Holistic Approach to Safeguarding DNNs from Different Types of Bit-Flip Attacks](http://arxiv.org/pdf/2506.10744v1)

Authors: Xiaobei Yan, Han Qiu, Tianwei Zhang

Bit-flip attacks (BFAs) represent a serious threat to Deep Neural Networks
(DNNs), where flipping a small number of bits in the model parameters or binary
code can significantly degrade the model accuracy or mislead the model
prediction in a desired way. Existing defenses exclusively focus on protecting
models for specific attacks and platforms, while lacking effectiveness for
other scenarios. We propose ObfusBFA, an efficient and holistic methodology to
mitigate BFAs targeting both the high-level model weights and low-level
codebase (executables or shared libraries). The key idea of ObfusBFA is to
introduce random dummy operations during the model inference, which effectively
transforms the delicate attacks into random bit flips, making it much harder
for attackers to pinpoint and exploit vulnerable bits. We design novel
algorithms to identify critical bits and insert obfuscation operations. We
evaluate ObfusBFA against different types of attacks, including the adaptive
scenarios where the attacker increases the flip bit budget to attempt to
circumvent our defense. The results show that ObfusBFA can consistently
preserve the model accuracy across various datasets and DNN architectures while
significantly reducing the attack success rates. Additionally, it introduces
minimal latency and storage overhead, making it a practical solution for
real-world applications.

### 9. [Quantifying Azure RBAC Wildcard Overreach](http://arxiv.org/pdf/2506.10755v1)

Authors: Christophe Parisel

Azure RBAC leverages wildcard permissions to simplify policy authoring, but
this abstraction often obscures the actual set of allowed operations and
undermines least-privilege guarantees. We introduce Belshazaar, a two-stage
framework that targets both the effective permission set problem and the
evaluation of wildcards permissions spread. First, we formalize Azure action
syntax via a context free grammar and implement a compiler that expands any
wildcard into its explicit action set. Second, we define an ultrametric
diameter metric to quantify semantic overreach in wildcard scenarios. Applied
to Microsoft s official catalog of 15481 actions, Belshazaar reveals that about
39 percent of actions admit a cross Resource Provider reach when associated
with non obvious wildcards, and that effective permissions sets are effectively
computable. These findings demonstrate that wildcard patterns can introduce
substantial privilege bloat, and that our approach offers a scalable, semantics
driven path toward tighter, least-privilege RBAC policies in Azure
environments.

### 10. [AI-Based Software Vulnerability Detection: A Systematic Literature Review](http://arxiv.org/pdf/2506.10280v1)

Authors: Samiha Shimmi, Hamed Okhravi, Mona Rahimi

Software vulnerabilities in source code pose serious cybersecurity risks,
prompting a shift from traditional detection methods (e.g., static analysis,
rule-based matching) to AI-driven approaches. This study presents a systematic
review of software vulnerability detection (SVD) research from 2018 to 2023,
offering a comprehensive taxonomy of techniques, feature representations, and
embedding methods. Our analysis reveals that 91% of studies use AI-based
methods, with graph-based models being the most prevalent. We identify key
limitations, including dataset quality, reproducibility, and interpretability,
and highlight emerging opportunities in underexplored techniques such as
federated learning and quantum neural networks, providing a roadmap for future
research.

### Computer Vision and Pattern Recognition

### 1. [HalLoc: Token-level Localization of Hallucinations for Vision Language Models](http://arxiv.org/pdf/2506.10286v1)

Authors: Eunkyu Park, Minyeong Kim, Gunhee Kim

Hallucinations pose a significant challenge to the reliability of large
vision-language models, making their detection essential for ensuring accuracy
in critical applications. Current detection methods often rely on
computationally intensive models, leading to high latency and resource demands.
Their definitive outcomes also fail to account for real-world scenarios where
the line between hallucinated and truthful information is unclear. To address
these issues, we propose HalLoc, a dataset designed for efficient,
probabilistic hallucination detection. It features 150K token-level annotated
samples, including hallucination types, across Visual Question Answering (VQA),
instruction-following, and image captioning tasks. This dataset facilitates the
development of models that detect hallucinations with graded confidence,
enabling more informed user interactions. Additionally, we introduce a baseline
model trained on HalLoc, offering low-overhead, concurrent hallucination
detection during generation. The model can be seamlessly integrated into
existing VLMs, improving reliability while preserving efficiency. The prospect
of a robust plug-and-play hallucination detection module opens new avenues for
enhancing the trustworthiness of vision-language models in real-world
applications. The HalLoc dataset and code are publicly available at:
https://github.com/dbsltm/cvpr25_halloc.

### 2. [PointGS: Point Attention-Aware Sparse View Synthesis with Gaussian Splatting](http://arxiv.org/pdf/2506.10335v1)

Authors: Lintao Xiang, Hongpei Zheng, Yating Huang, Qijun Yang, Hujun Yin

3D Gaussian splatting (3DGS) is an innovative rendering technique that
surpasses the neural radiance field (NeRF) in both rendering speed and visual
quality by leveraging an explicit 3D scene representation. Existing 3DGS
approaches require a large number of calibrated views to generate a consistent
and complete scene representation. When input views are limited, 3DGS tends to
overfit the training views, leading to noticeable degradation in rendering
quality. To address this limitation, we propose a Point-wise Feature-Aware
Gaussian Splatting framework that enables real-time, high-quality rendering
from sparse training views. Specifically, we first employ the latest stereo
foundation model to estimate accurate camera poses and reconstruct a dense
point cloud for Gaussian initialization. We then encode the colour attributes
of each 3D Gaussian by sampling and aggregating multiscale 2D appearance
features from sparse inputs. To enhance point-wise appearance representation,
we design a point interaction network based on a self-attention mechanism,
allowing each Gaussian point to interact with its nearest neighbors. These
enriched features are subsequently decoded into Gaussian parameters through two
lightweight multi-layer perceptrons (MLPs) for final rendering. Extensive
experiments on diverse benchmarks demonstrate that our method significantly
outperforms NeRF-based approaches and achieves competitive performance under
few-shot settings compared to the state-of-the-art 3DGS methods.

### 3. [GeoCAD: Local Geometry-Controllable CAD Generation](http://arxiv.org/pdf/2506.10337v1)

Authors: Zhanwei Zhang, Kaiyuan Liu, Junjie Liu, Wenxiao Wang, Binbin Lin, Liang Xie, Chen Shen, Deng Cai

Local geometry-controllable computer-aided design (CAD) generation aims to
modify local parts of CAD models automatically, enhancing design efficiency. It
also ensures that the shapes of newly generated local parts follow
user-specific geometric instructions (e.g., an isosceles right triangle or a
rectangle with one corner cut off). However, existing methods encounter
challenges in achieving this goal. Specifically, they either lack the ability
to follow textual instructions or are unable to focus on the local parts. To
address this limitation, we introduce GeoCAD, a user-friendly and local
geometry-controllable CAD generation method. Specifically, we first propose a
complementary captioning strategy to generate geometric instructions for local
parts. This strategy involves vertex-based and VLLM-based captioning for
systematically annotating simple and complex parts, respectively. In this way,
we caption $\sim$221k different local parts in total. In the training stage,
given a CAD model, we randomly mask a local part. Then, using its geometric
instruction and the remaining parts as input, we prompt large language models
(LLMs) to predict the masked part. During inference, users can specify any
local part for modification while adhering to a variety of predefined geometric
instructions. Extensive experiments demonstrate the effectiveness of GeoCAD in
generation quality, validity and text-to-CAD consistency. Code will be
available at https://github.com/Zhanwei-Z/GeoCAD.

### 4. [RealKeyMorph: Keypoints in Real-world Coordinates for Resolution-agnostic Image Registration](http://arxiv.org/pdf/2506.10344v1)

Authors: Mina C. Moghadam, Alan Q. Wang, Omer Taub, Martin R. Prince, Mert R. Sabuncu

Many real-world settings require registration of a pair of medical images
that differ in spatial resolution, which may arise from differences in image
acquisition parameters like pixel spacing, slice thickness, and field-of-view.
However, all previous machine learning-based registration techniques resample
images onto a fixed resolution. This is suboptimal because resampling can
introduce artifacts due to interpolation. To address this, we present
RealKeyMorph (RKM), a resolution-agnostic method for image registration. RKM is
an extension of KeyMorph, a registration framework which works by training a
network to learn corresponding keypoints for a given pair of images, after
which a closed-form keypoint matching step is used to derive the transformation
that aligns them. To avoid resampling and enable operating on the raw data, RKM
outputs keypoints in real-world coordinates of the scanner. To do this, we
leverage the affine matrix produced by the scanner (e.g., MRI machine) that
encodes the mapping from voxel coordinates to real world coordinates. By
transforming keypoints into real-world space and integrating this into the
training process, RKM effectively enables the extracted keypoints to be
resolution-agnostic. In our experiments, we demonstrate the advantages of RKM
on the registration task for orthogonal 2D stacks of abdominal MRIs, as well as
3D volumes with varying resolutions in brain datasets.

### 5. [Motion-R1: Chain-of-Thought Reasoning and Reinforcement Learning for Human Motion Generation](http://arxiv.org/pdf/2506.10353v1)

Authors: Runqi Ouyang, Haoyun Li, Zhenyuan Zhang, Xiaofeng Wang, Zheng Zhu, Guan Huang, Xingang Wang

Recent advances in large language models, especially in natural language
understanding and reasoning, have opened new possibilities for text-to-motion
generation. Although existing approaches have made notable progress in semantic
alignment and motion synthesis, they often rely on end-to-end mapping
strategies that fail to capture deep linguistic structures and logical
reasoning. Consequently, generated motions tend to lack controllability,
consistency, and diversity. To address these limitations, we propose Motion-R1,
a unified motion-language modeling framework that integrates a Chain-of-Thought
mechanism. By explicitly decomposing complex textual instructions into
logically structured action paths, Motion-R1 provides high-level semantic
guidance for motion generation, significantly enhancing the model's ability to
interpret and execute multi-step, long-horizon, and compositionally rich
commands. To train our model, we adopt Group Relative Policy Optimization, a
reinforcement learning algorithm designed for large models, which leverages
motion quality feedback to optimize reasoning chains and motion synthesis
jointly. Extensive experiments across multiple benchmark datasets demonstrate
that Motion-R1 achieves competitive or superior performance compared to
state-of-the-art methods, particularly in scenarios requiring nuanced semantic
understanding and long-term temporal coherence. The code, model and data will
be publicly available.

### 6. [FaceLiVT: Face Recognition using Linear Vision Transformer with Structural Reparameterization For Mobile Device](http://arxiv.org/pdf/2506.10361v1)

Authors: Novendra Setyawan, Chi-Chia Sun, Mao-Hsiu Hsu, Wen-Kai Kuo, Jun-Wei Hsieh

This paper introduces FaceLiVT, a lightweight yet powerful face recognition
model that integrates a hybrid Convolution Neural Network (CNN)-Transformer
architecture with an innovative and lightweight Multi-Head Linear Attention
(MHLA) mechanism. By combining MHLA alongside a reparameterized token mixer,
FaceLiVT effectively reduces computational complexity while preserving
competitive accuracy. Extensive evaluations on challenging benchmarks;
including LFW, CFP-FP, AgeDB-30, IJB-B, and IJB-C; highlight its superior
performance compared to state-of-the-art lightweight models. MHLA notably
improves inference speed, allowing FaceLiVT to deliver high accuracy with lower
latency on mobile devices. Specifically, FaceLiVT is 8.6 faster than EdgeFace,
a recent hybrid CNN-Transformer model optimized for edge devices, and 21.2
faster than a pure ViT-Based model. With its balanced design, FaceLiVT offers
an efficient and practical solution for real-time face recognition on
resource-constrained platforms.

### 7. [FSATFusion: Frequency-Spatial Attention Transformer for Infrared and Visible Image Fusion](http://arxiv.org/pdf/2506.10366v1)

Authors: Tianpei Zhang, Jufeng Zhao, Yiming Zhu, Guangmang Cui, Yuhan Lyu

The infrared and visible images fusion (IVIF) is receiving increasing
attention from both the research community and industry due to its excellent
results in downstream applications. Existing deep learning approaches often
utilize convolutional neural networks to extract image features. However, the
inherently capacity of convolution operations to capture global context can
lead to information loss, thereby restricting fusion performance. To address
this limitation, we propose an end-to-end fusion network named the
Frequency-Spatial Attention Transformer Fusion Network (FSATFusion). The
FSATFusion contains a frequency-spatial attention Transformer (FSAT) module
designed to effectively capture discriminate features from source images. This
FSAT module includes a frequency-spatial attention mechanism (FSAM) capable of
extracting significant features from feature maps. Additionally, we propose an
improved Transformer module (ITM) to enhance the ability to extract global
context information of vanilla Transformer. We conducted both qualitative and
quantitative comparative experiments, demonstrating the superior fusion quality
and efficiency of FSATFusion compared to other state-of-the-art methods.
Furthermore, our network was tested on two additional tasks without any
modifications, to verify the excellent generalization capability of FSATFusion.
Finally, the object detection experiment demonstrated the superiority of
FSATFusion in downstream visual tasks. Our code is available at
https://github.com/Lmmh058/FSATFusion.

### 8. [Leveraging 6DoF Pose Foundation Models For Mapping Marine Sediment Burial](http://arxiv.org/pdf/2506.10386v1)

Authors: Jerry Yan, Chinmay Talegaonkar, Nicholas Antipa, Eric Terrill, Sophia Merrifield

The burial state of anthropogenic objects on the seafloor provides insight
into localized sedimentation dynamics and is also critical for assessing
ecological risks, potential pollutant transport, and the viability of recovery
or mitigation strategies for hazardous materials such as munitions. Accurate
burial depth estimation from remote imagery remains difficult due to partial
occlusion, poor visibility, and object degradation. This work introduces a
computer vision pipeline, called PoseIDON, which combines deep foundation model
features with multiview photogrammetry to estimate six degrees of freedom
object pose and the orientation of the surrounding seafloor from ROV video.
Burial depth is inferred by aligning CAD models of the objects with observed
imagery and fitting a local planar approximation of the seafloor. The method is
validated using footage of 54 objects, including barrels and munitions,
recorded at a historic ocean dumpsite in the San Pedro Basin. The model
achieves a mean burial depth error of approximately 10 centimeters and resolves
spatial burial patterns that reflect underlying sediment transport processes.
This approach enables scalable, non-invasive mapping of seafloor burial and
supports environmental assessment at contaminated sites.

### 9. [DART: Differentiable Dynamic Adaptive Region Tokenizer for Vision Transformer and Mamba](http://arxiv.org/pdf/2506.10390v1)

Authors: Shicheng Yin, Kaixuan Yin, Yang Liu, Weixing Chen, Liang Lin

Recently, non-convolutional models such as the Vision Transformer (ViT) and
Vision Mamba (Vim) have achieved remarkable performance in computer vision
tasks. However, their reliance on fixed-size patches often results in excessive
encoding of background regions and omission of critical local details,
especially when informative objects are sparsely distributed. To address this,
we introduce a fully differentiable Dynamic Adaptive Region Tokenizer (DART),
which adaptively partitions images into content-dependent patches of varying
sizes. DART combines learnable region scores with piecewise differentiable
quantile operations to allocate denser tokens to information-rich areas.
Despite introducing only approximately 1 million (1M) additional parameters,
DART improves accuracy by 2.1% on DeiT (ImageNet-1K). Unlike methods that
uniformly increase token density to capture fine-grained details, DART offers a
more efficient alternative, achieving 45% FLOPs reduction with superior
performance. Extensive experiments on DeiT, Vim, and VideoMamba confirm that
DART consistently enhances accuracy while incurring minimal or even reduced
computational overhead. Code is available at
https://github.com/HCPLab-SYSU/DART.

### 10. [ReconMOST: Multi-Layer Sea Temperature Reconstruction with Observations-Guided Diffusion](http://arxiv.org/pdf/2506.10391v1)

Authors: Yuanyi Song, Pumeng Lyu, Ben Fei, Fenghua Ling, Wanli Ouyang, Lei Bai

Accurate reconstruction of ocean is essential for reflecting global climate
dynamics and supporting marine meteorological research. Conventional methods
face challenges due to sparse data, algorithmic complexity, and high
computational costs, while increasing usage of machine learning (ML) method
remains limited to reconstruction problems at the sea surface and local
regions, struggling with issues like cloud occlusion. To address these
limitations, this paper proposes ReconMOST, a data-driven guided diffusion
model framework for multi-layer sea temperature reconstruction. Specifically,
we first pre-train an unconditional diffusion model using a large collection of
historical numerical simulation data, enabling the model to attain physically
consistent distribution patterns of ocean temperature fields. During the
generation phase, sparse yet high-accuracy in-situ observational data are
utilized as guidance points for the reverse diffusion process, generating
accurate reconstruction results. Importantly, in regions lacking direct
observational data, the physically consistent spatial distribution patterns
learned during pre-training enable implicitly guided and physically plausible
reconstructions. Our method extends ML-based SST reconstruction to a global,
multi-layer setting, handling over 92.5% missing data while maintaining
reconstruction accuracy, spatial resolution, and superior generalization
capability. We pre-train our model on CMIP6 numerical simulation data and
conduct guided reconstruction experiments on CMIP6 and EN4 analysis data. The
results of mean squared error (MSE) values achieve 0.049 on guidance, 0.680 on
reconstruction, and 0.633 on total, respectively, demonstrating the
effectiveness and robustness of the proposed framework. Our source code is
available at https://github.com/norsheep/ReconMOST.

### Computers and Society

### 1. [Collective Bargaining in the Information Economy Can Address AI-Driven Power Concentration](http://arxiv.org/pdf/2506.10272v1)

Authors: Nicholas Vincent, Matthew Prewitt, Hanlin Li

This position paper argues that there is an urgent need to restructure
markets for the information that goes into AI systems. Specifically, producers
of information goods (such as journalists, researchers, and creative
professionals) need to be able to collectively bargain with AI product builders
in order to receive reasonable terms and a sustainable return on the
informational value they contribute. We argue that without increased market
coordination or collective bargaining on the side of these primary information
producers, AI will exacerbate a large-scale "information market failure" that
will lead not only to undesirable concentration of capital, but also to a
potential "ecological collapse" in the informational commons. On the other
hand, collective bargaining in the information economy can create market
frictions and aligned incentives necessary for a pro-social, sustainable AI
future. We provide concrete actions that can be taken to support a
coalition-based approach to achieve this goal. For example, researchers and
developers can establish technical mechanisms such as federated data management
tools and explainable data value estimations, to inform and facilitate
collective bargaining in the information economy. Additionally, regulatory and
policy interventions may be introduced to support trusted data intermediary
organizations representing guilds or syndicates of information producers.

### 2. ["Check My Work?": Measuring Sycophancy in a Simulated Educational Context](http://arxiv.org/pdf/2506.10297v1)

Authors: Chuck Arvin

This study examines how user-provided suggestions affect Large Language
Models (LLMs) in a simulated educational context, where sycophancy poses
significant risks. Testing five different LLMs from the OpenAI GPT-4o and
GPT-4.1 model classes across five experimental conditions, we show that
response quality varies dramatically based on query framing. In cases where the
student mentions an incorrect answer, the LLM correctness can degrade by as
much as 15 percentage points, while mentioning the correct answer boosts
accuracy by the same margin. Our results also show that this bias is stronger
in smaller models, with an effect of up to 30% for the GPT-4.1-nano model,
versus 8% for the GPT-4o model. Our analysis of how often LLMs "flip" their
answer, and an investigation into token level probabilities, confirm that the
models are generally changing their answers to answer choices mentioned by
students in line with the sycophancy hypothesis. This sycophantic behavior has
important implications for educational equity, as LLMs may accelerate learning
for knowledgeable students while the same tools may reinforce misunderstanding
for less knowledgeable students. Our results highlight the need to better
understand the mechanism, and ways to mitigate, such bias in the educational
context.

### 3. [FASCIST-O-METER: Classifier for Neo-fascist Discourse Online](http://arxiv.org/pdf/2506.10789v1)

Authors: Rudy Alexandro Garrido Veliz, Martin Semmann, Chris Biemann, Seid Muhie Yimam

Neo-fascism is a political and societal ideology that has been having
remarkable growth in the last decade in the United States of America (USA), as
well as in other Western societies. It poses a grave danger to democracy and
the minorities it targets, and it requires active actions against it to avoid
escalation. This work presents the first-of-its-kind neo-fascist coding scheme
for digital discourse in the USA societal context, overseen by political
science researchers. Our work bridges the gap between Natural Language
Processing (NLP) and political science against this phenomena. Furthermore, to
test the coding scheme, we collect a tremendous amount of activity on the
internet from notable neo-fascist groups (the forums of Iron March and
Stormfront.org), and the guidelines are applied to a subset of the collected
posts. Through crowdsourcing, we annotate a total of a thousand posts that are
labeled as neo-fascist or non-neo-fascist. With this labeled data set, we
fine-tune and test both Small Language Models (SLMs) and Large Language Models
(LLMs), obtaining the very first classification models for neo-fascist
discourse. We find that the prevalence of neo-fascist rhetoric in this kind of
forum is ever-present, making them a good target for future research. The
societal context is a key consideration for neo-fascist speech when conducting
NLP research. Finally, the work against this kind of political movement must be
pressed upon and continued for the well-being of a democratic society.
Disclaimer: This study focuses on detecting neo-fascist content in text,
similar to other hate speech analyses, without labeling individuals or
organizations.

### 4. [LLM-Driven Personalized Answer Generation and Evaluation](http://arxiv.org/pdf/2506.10829v1)

Authors: Mohammadreza Molavi, Mohammadreza Tavakoli, Mohammad Moein, Abdolali Faraji, Gábor Kismihók

Online learning has experienced rapid growth due to its flexibility and
accessibility. Personalization, adapted to the needs of individual learners, is
crucial for enhancing the learning experience, particularly in online settings.
A key aspect of personalization is providing learners with answers customized
to their specific questions. This paper therefore explores the potential of
Large Language Models (LLMs) to generate personalized answers to learners'
questions, thereby enhancing engagement and reducing the workload on educators.
To evaluate the effectiveness of LLMs in this context, we conducted a
comprehensive study using the StackExchange platform in two distinct areas:
language learning and programming. We developed a framework and a dataset for
validating automatically generated personalized answers. Subsequently, we
generated personalized answers using different strategies, including 0-shot,
1-shot, and few-shot scenarios. The generated answers were evaluated using
three methods: 1. BERTScore, 2. LLM evaluation, and 3. human evaluation. Our
findings indicated that providing LLMs with examples of desired answers (from
the learner or similar learners) can significantly enhance the LLMs' ability to
tailor responses to individual learners' needs.

### 5. [A Study on Individual Spatiotemporal Activity Generation Method Using MCP-Enhanced Chain-of-Thought Large Language Models](http://arxiv.org/pdf/2506.10853v1)

Authors: Yu Zhang, Yang Hu, De Wang

Human spatiotemporal behavior simulation is critical for urban planning
research, yet traditional rule-based and statistical approaches suffer from
high computational costs, limited generalizability, and poor scalability. While
large language models (LLMs) show promise as "world simulators," they face
challenges in spatiotemporal reasoning including limited spatial cognition,
lack of physical constraint understanding, and group homogenization tendencies.
This paper introduces a framework integrating chain-of-thought (CoT) reasoning
with Model Context Protocol (MCP) to enhance LLMs' capability in simulating
spatiotemporal behaviors that correspond with validation data patterns. The
methodology combines human-like progressive reasoning through a five-stage
cognitive framework with comprehensive data processing via six specialized MCP
tool categories: temporal management, spatial navigation, environmental
perception, personal memory, social collaboration, and experience evaluation.
Experiments in Shanghai's Lujiazui district validate the framework's
effectiveness across 1,000 generated samples. Results demonstrate high
similarity with real mobile signaling data, achieving generation quality scores
of 7.86 to 8.36 across different base models. Parallel processing experiments
show efficiency improvements, with generation times decreasing from 1.30 to
0.17 minutes per sample when scaling from 2 to 12 processes. This work
contributes to integrating CoT reasoning with MCP for urban behavior modeling,
advancing LLMs applications in urban computing and providing a practical
approach for synthetic mobility data generation. The framework offers a
foundation for smart city planning, transportation forecasting, and
participatory urban design applications.

### 6. [The Urban Model Platform: A Public Backbone for Modeling and Simulation in Urban Digital Twins](http://arxiv.org/pdf/2506.10964v1)

Authors: Rico H Herzog, Till Degkwitz, Trivik Verma

Urban digital twins are increasingly perceived as a way to pool the growing
digital resources of cities for the purpose of a more sustainable and
integrated urban planning. Models and simulations are central to this
undertaking: They enable "what if?" scenarios, create insights and describe
relationships between the vast data that is being collected. However, the
process of integrating and subsequently using models in urban digital twins is
an inherently complex undertaking. It raises questions about how to represent
urban complexity, how to deal with uncertain assUrban Model Platformtions and
modeling paradigms, and how to capture underlying power relations. Existent
approaches in the domain largely focus on monolithic and centralized solutions
in the tradition of neoliberal city-making, oftentimes prohibiting pluralistic
and open interoperable models. Using a participatory design for participatory
systems approach together with the City of Hamburg, Germany, we find that an
open Urban Model Platform can function both as a public technological backbone
for modeling and simulation in urban digital twins and as a socio-technical
framework for a collaborative and pluralistic representation of urban
processes. Such a platform builds on open standards, allows for a decentralized
integration of models, enables communication between models and supports a
multi-model approach to representing urban systems.

### 7. [The Alignment Trap: Complexity Barriers](http://arxiv.org/pdf/2506.10304v1)

Authors: Jasper Yao

We establish fundamental computational complexity barriers to verifying AI
safety as system capabilities scale. Our main results show that for AI systems
with expressiveness EXP$(m)$ above a critical threshold $\tau$, safety
verification requires exponential time and is coNP-complete. We formalize the
Capability-Risk Scaling (CRS) dynamic, which demonstrates how increasing AI
capability drives societal safety requirements toward perfection, creating an
inescapable tension with verification complexity. Through four core theorems,
we prove that (1) verification complexity grows exponentially with system
expressiveness, (2) safe policies comprise at most a $2^{-2^m}$ fraction of the
policy space, (3) no finite set of alignment techniques can provide universal
coverage, and (4) robust safety properties form measure-zero sets for neural
networks. These results characterize an "intractability gap" where practical
safety requirements fall within the region of computational intractability. We
conclude by presenting a strategic trilemma: AI development must either
constrain system complexity to maintain verifiable safety, accept unverifiable
risks while scaling capabilities, or develop fundamentally new safety paradigms
beyond verification. Our work provides the first systematic
complexity-theoretic analysis of AI alignment and establishes rigorous bounds
that any safety approach must confront. A formal verification of the core
theorems in Lean4 is currently in progress.

### 8. [Bug Classification in Quantum Software: A Rule-Based Framework and Its Evaluation](http://arxiv.org/pdf/2506.10397v1)

Authors: Mir Mohammad Yousuf, Shabir Ahmad Sofi

Accurate classification of software bugs is essential for improving software
quality. This paper presents a rule-based automated framework for classifying
issues in quantum software repositories by bug type, category, severity, and
impacted quality attributes, with additional focus on quantum-specific bug
types. The framework applies keyword and heuristic-based techniques tailored to
quantum computing. To assess its reliability, we manually classified a
stratified sample of 4,984 issues from a dataset of 12,910 issues across 36
Qiskit repositories. Automated classifications were compared with ground truth
using accuracy, precision, recall, and F1-score. The framework achieved up to
85.21% accuracy, with F1-scores ranging from 0.7075 (severity) to 0.8393
(quality attribute). Statistical validation via paired t-tests and Cohen's
Kappa showed substantial to almost perfect agreement for bug type (k = 0.696),
category (k = 0.826), quality attribute (k = 0.818), and quantum-specific bug
type (k = 0.712). Severity classification showed slight agreement (k = 0.162),
suggesting room for improvement. Large-scale analysis revealed that classical
bugs dominate (67.2%), with quantum-specific bugs at 27.3%. Frequent bug
categories included compatibility, functional, and quantum-specific defects,
while usability, maintainability, and interoperability were the most impacted
quality attributes. Most issues (93.7%) were low severity; only 4.3% were
critical. A detailed review of 1,550 quantum-specific bugs showed that over
half involved quantum circuit-level problems, followed by gate errors and
hardware-related issues.

### 9. [Size-adaptive Hypothesis Testing for Fairness](http://arxiv.org/pdf/2506.10586v1)

Authors: Antonio Ferrara, Francesco Cozzi, Alan Perotti, André Panisson, Francesco Bonchi

Determining whether an algorithmic decision-making system discriminates
against a specific demographic typically involves comparing a single point
estimate of a fairness metric against a predefined threshold. This practice is
statistically brittle: it ignores sampling error and treats small demographic
subgroups the same as large ones. The problem intensifies in intersectional
analyses, where multiple sensitive attributes are considered jointly, giving
rise to a larger number of smaller groups. As these groups become more
granular, the data representing them becomes too sparse for reliable
estimation, and fairness metrics yield excessively wide confidence intervals,
precluding meaningful conclusions about potential unfair treatments.
  In this paper, we introduce a unified, size-adaptive, hypothesis-testing
framework that turns fairness assessment into an evidence-based statistical
decision. Our contribution is twofold. (i) For sufficiently large subgroups, we
prove a Central-Limit result for the statistical parity difference, leading to
analytic confidence intervals and a Wald test whose type-I (false positive)
error is guaranteed at level $\alpha$. (ii) For the long tail of small
intersectional groups, we derive a fully Bayesian Dirichlet-multinomial
estimator; Monte-Carlo credible intervals are calibrated for any sample size
and naturally converge to Wald intervals as more data becomes available. We
validate our approach empirically on benchmark datasets, demonstrating how our
tests provide interpretable, statistically rigorous decisions under varying
degrees of data availability and intersectionality.

### 10. [Video-Mediated Emotion Disclosure: A Study of Mental Health Vlogging by People with Schizophrenia on YouTube](http://arxiv.org/pdf/2506.10932v1)

Authors: Jiaying Lizzy Liu, Yan Zhang

Individuals with schizophrenia frequently experience intense emotions and
often turn to vlogging as a medium for emotional expression. While previous
research has predominantly focused on text based disclosure, little is known
about how individuals construct narratives around emotions and emotional
experiences in video blogs. Our study addresses this gap by analyzing 200
YouTube videos created by individuals with schizophrenia. Drawing on media
research and self presentation theories, we developed a visual analysis
framework to disentangle these videos. Our analysis revealed diverse practices
of emotion disclosure through both verbal and visual channels, highlighting the
dynamic interplay between these modes of expression. We found that the
deliberate construction of visual elements, including environmental settings
and specific aesthetic choices, appears to foster more supportive and engaged
viewer responses. These findings underscore the need for future large scale
quantitative research examining how visual features shape video mediated
communication on social media platforms. Such investigations would inform the
development of care centered video sharing platforms that better support
individuals managing illness experiences.

### Databases

### 1. [S3 Mirror: S3Mirror: Making Genomic Data Transfers Fast, Reliable, and Observable with DBOS](http://arxiv.org/pdf/2506.10886v1)

Authors: Steven Vasquez-Grinnell, Alex Poliakov

To meet the needs of a large pharmaceutical organization, we set out to
create S3Mirror - an application for transferring large genomic sequencing
datasets between S3 buckets quickly, reliably, and observably. We used the
DBOSTransact durable execution framework to achieve these goals and benchmarked
the performance and cost of the application. S3Mirror is an open source DBOS
Python application that can run in a variety of environments, including DBOS
Cloud Pro where it runs as much as 40x faster than AWS DataSync at a fraction
of the cost. Moreover, S3Mirror is resilient to failures and allows for
real-time filewise observability of ongoing and past transfers.

### 2. [A Hybrid Heuristic Framework for Resource-Efficient Querying of Scientific Experiments Data](http://arxiv.org/pdf/2506.10422v1)

Authors: Mayank Patel, Minal Bhise

Scientific experiments and modern applications are generating large amounts
of data every day. Most organizations utilize In-house servers or Cloud
resources to manage application data and workload. The traditional database
management system (DBMS) and HTAP systems spend significant time & resources to
load the entire dataset into DBMS before starting query execution. On the other
hand, in-situ engines may reparse required data multiple times, increasing
resource utilization and data processing costs. Additionally, over or
under-allocation of resources also increases application running costs. This
paper proposes a lightweight Resource Availability &Workload aware Hybrid
Framework (RAW-HF) to optimize querying raw data by utilizing existing finite
resources efficiently. RAW-HF includes modules that help optimize the resources
required to execute a given workload and maximize the utilization of existing
resources. The impact of applying RAW-HF to real-world scientific dataset
workloads like Sloan Digital Sky Survey (SDSS) and Linked Observation Data
(LOD) presented over 90% and 85% reduction in workload execution time (WET)
compared to widely used traditional DBMS PostgreSQL. The overall CPU, IO
resource utilization, and WET have been reduced by 26%, 25%, and 26%,
respectively, while improving memory utilization by 33%, compared to the
state-of-the-art workload-aware partial loading technique (WA) proposed for
hybrid systems. A comparison of MUAR technique used by RAW-HF with machine
learning based resource allocation techniques like PCC is also presented.

### Distributed, Parallel, and Cluster Computing

### 1. [Resilience through Automated Adaptive Configuration for Distribution and Replication](http://arxiv.org/pdf/2506.10248v1)

Authors: Scott D. Stoller, Balaji Jayasankar, Yanhong A. Liu

This paper presents a powerful automated framework for making complex systems
resilient under failures, by optimized adaptive distribution and replication of
interdependent software components across heterogeneous hardware components
with widely varying capabilities. A configuration specifies how software is
distributed and replicated: which software components to run on each computer,
which software components to replicate, which replication protocols to use,
etc. We present an algorithm that, given a system model and resilience
requirements, (1) determines initial configurations of the system that are
resilient, and (2) generates a reconfiguration policy that determines
reconfiguration actions to execute in response to failures and recoveries. This
model-finding algorithm is based on state-space exploration and incorporates
powerful optimizations, including a quotient reduction based on a novel
equivalence relation between states. We present experimental results from
successfully applying a prototype implementation of our framework to a model of
an autonomous driving system.

### 2. [Federated Learning within Global Energy Budget over Heterogeneous Edge Accelerators](http://arxiv.org/pdf/2506.10413v1)

Authors: Roopkatha Banerjee, Tejus Chandrashekar, Ananth Eswar, Yogesh Simmhan

Federated Learning (FL) enables collaborative model training across
distributed clients while preserving data privacy. However, optimizing both
energy efficiency and model accuracy remains a challenge, given device and data
heterogeneity. Further, sustainable AI through a global energy budget for FL
has not been explored. We propose a novel optimization problem for client
selection in FL that maximizes the model accuracy within an overall energy
limit and reduces training time. We solve this with a unique bi-level ILP
formulation that leverages approximate Shapley values and energy-time
prediction models to efficiently solve this. Our FedJoule framework achieves
superior training accuracies compared to SOTA and simple baselines for diverse
energy budgets, non-IID distributions, and realistic experiment configurations,
performing 15% and 48% better on accuracy and time, respectively. The results
highlight the effectiveness of our method in achieving a viable trade-off
between energy usage and performance in FL environments.

### 3. [Automating Multi-Tenancy Performance Evaluation on Edge Compute Nodes](http://arxiv.org/pdf/2506.10461v1)

Authors: Joanna Georgiou, Moysis Symeonides, George Pallis, Marios D. Dikaiakos

Edge Computing emerges as a promising alternative of Cloud Computing, with
scalable compute resources and services deployed in the path between IoT
devices and Cloud. Since virtualization techniques can be applied on Edge
compute nodes, administrators can share their Edge infrastructures among
multiple users, providing the so-called multi-tenancy. Even though
multi-tenancy is unavoidable, it raises concerns about security and performance
degradation due to resource contention in Edge Computing. For that,
administrators need to deploy services with non-antagonizing profiles and
explore workload co-location scenarios to enhance performance and energy
consumption. Achieving this, however, requires extensive configuration,
deployment, iterative testing, and analysis, an effort-intensive and
time-consuming process. To address this challenge, we introduce an
auto-benchmarking framework designed to streamline the analysis of
multi-tenancy performance in Edge environments. Our framework includes a
built-in monitoring stack and integrates with widely used benchmarking
workloads, such as streaming analytics, database operations, machine learning
applications, and component-based stress testing. We perform a case-driven
analysis and provide valuable insights into the impact of multi-tenancy on Edge
environments with different hardware configurations and diverse workloads.
Finally, the implementation of our framework, along with the containerized
workloads used for experimentation, is publicly available.

### 4. [TD-Pipe: Temporally-Disaggregated Pipeline Parallelism Architecture for High-Throughput LLM Inference](http://arxiv.org/pdf/2506.10470v1)

Authors: Hongbin Zhang, Taosheng Wei, Zhenyi Zheng, Jiangsu Du, Zhiguang Chen, Yutong Lu

As the model size continuously increases, pipeline parallelism shows great
promise in throughput-oriented LLM inference due to its low demand on
communications. However, imbalanced pipeline workloads and complex data
dependencies in the prefill and decode phases result in massive pipeline
bubbles and further severe performance reduction. To better exploit the
pipeline parallelism for high-throughput LLM inference, we propose TD-Pipe,
with the key idea lies in the temporally-disaggregated pipeline parallelism
architecture. Specifically, this architecture disaggregates the prefill and
decode phases in the temporal dimension, so as to eliminate pipeline bubbles
caused by the phase switching. TD-Pipe identifies potential issues of
exploiting the novel architecture and provides solutions. First, a
hierarchy-controller structure is used to better coordinate devices in pipeline
parallelism by decoupling the scheduling from execution. Second, the AI-based
greedy prefill approach aggressively performs more prefills by predicting the
output length and simulating the memory usage. Third, the inter-batch work
stealing approach dynamically balances decode phase workloads between different
batches to reduce bubbles. Forth, the spatial-temporal intensity comparison
approach determines the optimal switch from decode to prefill by comparing the
performance drop from reduced computational intensity with that from phase
switching bubbles. Extensive experiments show that TD-Pipe effectively
increases the throughput of LLM inference by up to 1.91x over the existing
tensor parallel approach and 2.73x over the existing pipeline parallel approach
on GPU nodes with only PCIe interconnection.

### 5. [HP2C-DT: High-Precision High-Performance Computer-enabled Digital Twin](http://arxiv.org/pdf/2506.10523v1)

Authors: E. Iraola, M. García-Lorenzo, F. Lordan-Gomis, F. Rossi, E. Prieto-Araujo, R. M. Badia

Digital twins are transforming the way we monitor, analyze, and control
physical systems, but designing architectures that balance real-time
responsiveness with heavy computational demands remains a challenge.
Cloud-based solutions often struggle with latency and resource constraints,
while edge-based approaches lack the processing power for complex simulations
and data-driven optimizations.
  To address this problem, we propose the High-Precision High-Performance
Computer-enabled Digital Twin (HP2C-DT) reference architecture, which
integrates High-Performance Computing (HPC) into the computing continuum.
Unlike traditional setups that use HPC only for offline simulations, HP2C-DT
makes it an active part of digital twin workflows, dynamically assigning tasks
to edge, cloud, or HPC resources based on urgency and computational needs.
  Furthermore, to bridge the gap between theory and practice, we introduce the
HP2C-DT framework, a working implementation that uses COMPSs for seamless
workload distribution across diverse infrastructures. We test it in a power
grid use case, showing how it reduces communication bandwidth by an order of
magnitude through edge-side data aggregation, improves response times by up to
2x via dynamic offloading, and maintains near-ideal strong scaling for
compute-intensive workflows across a practical range of resources. These
results demonstrate how an HPC-driven approach can push digital twins beyond
their current limitations, making them smarter, faster, and more capable of
handling real-world complexity.

### 6. [GPU-Accelerated Distributed QAOA on Large-scale HPC Ecosystems](http://arxiv.org/pdf/2506.10531v1)

Authors: Zhihao Xu, Srikar Chundury, Seongmin Kim, Amir Shehata, Xinyi Li, Ang Li, Tengfei Luo, Frank Mueller, In-Saeng Suh

Quantum computing holds great potential to accelerate the process of solving
complex combinatorial optimization problems. The Distributed Quantum
Approximate Optimization Algorithm (DQAOA) addresses high-dimensional, dense
problems using current quantum computing techniques and high-performance
computing (HPC) systems. In this work, we improve the scalability and
efficiency of DQAOA through advanced problem decomposition and parallel
execution using message passing on the Frontier CPU/GPU supercomputer. Our
approach ensures efficient quantum-classical workload management by
distributing large problem instances across classical and quantum resources.
Experimental results demonstrate that enhanced decomposition strategies and
GPU-accelerated quantum simulations significantly improve DQAOA's performance,
achieving up to 10x speedup over CPU-based simulations. This advancement
enables better scalability for large problem instances, supporting the
practical deployment of GPU systems for hybrid quantum-classical applications.
We also highlight ongoing integration efforts using the Quantum Framework (QFw)
to support future HPC-quantum computing systems.

### 7. [6G Infrastructures for Edge AI: An Analytical Perspective](http://arxiv.org/pdf/2506.10570v1)

Authors: Kurt Horvath, Shpresa Tuda, Blerta Idrizi, Stojan Kitanov, Fisnik Doko, Dragi Kimovski

The convergence of Artificial Intelligence (AI) and the Internet of Things
has accelerated the development of distributed, network-sensitive applications,
necessitating ultra-low latency, high throughput, and real-time processing
capabilities. While 5G networks represent a significant technological
milestone, their ability to support AI-driven edge applications remains
constrained by performance gaps observed in real-world deployments. This paper
addresses these limitations and highlights critical advancements needed to
realize a robust and scalable 6G ecosystem optimized for AI applications.
Furthermore, we conduct an empirical evaluation of 5G network infrastructure in
central Europe, with latency measurements ranging from 61 ms to 110 ms across
different close geographical areas. These values exceed the requirements of
latency-critical AI applications by approximately 270%, revealing significant
shortcomings in current deployments. Building on these findings, we propose a
set of recommendations to bridge the gap between existing 5G performance and
the requirements of next-generation AI applications.

### 8. [Graph-based Gossiping for Communication Efficiency in Decentralized Federated Learning](http://arxiv.org/pdf/2506.10607v1)

Authors: Huong Nguyen, Hong-Tri Nguyen, Praveen Kumar Donta, Susanna Pirttikangas, Lauri Lovén

Federated learning has emerged as a privacy-preserving technique for
collaborative model training across heterogeneously distributed silos. Yet, its
reliance on a single central server introduces potential bottlenecks and risks
of single-point failure. Decentralizing the server, often referred to as
decentralized learning, addresses this problem by distributing the server role
across nodes within the network. One drawback regarding this pure
decentralization is it introduces communication inefficiencies, which arise
from increased message exchanges in large-scale setups. However, existing
proposed solutions often fail to simulate the real-world distributed and
decentralized environment in their experiments, leading to unreliable
performance evaluations and limited applicability in practice. Recognizing the
lack from prior works, this work investigates the correlation between model
size and network latency, a critical factor in optimizing decentralized
learning communication. We propose a graph-based gossiping mechanism, where
specifically, minimum spanning tree and graph coloring are used to optimize
network structure and scheduling for efficient communication across various
network topologies and message capacities. Our approach configures and manages
subnetworks on real physical routers and devices and closely models real-world
distributed setups. Experimental results demonstrate that our method
significantly improves communication, compatible with different topologies and
data sizes, reducing bandwidth and transfer time by up to circa 8 and 4.4
times, respectively, compared to naive flooding broadcasting methods.

### 9. [The Impact of Partial Computations on the Red-Blue Pebble Game](http://arxiv.org/pdf/2506.10854v1)

Authors: Pál András Papp, Aleksandros Sobczyk, A. N. Yzelman

We study an extension of the well-known red-blue pebble game (RBP) with
partial computation steps, inspired by the recent work of Sobczyk. While the
original RBP assumes that we need to have all the inputs of an operation in
fast memory at the same time, in many concrete computations, the inputs can be
aggregated one by one into the final output value. These partial computation
steps can enable pebbling strategies with much smaller I/O cost, and in
settings where such a step-by-step aggregation is possible, this extended
red-blue pebble game offers a much more realistic cost model.
  We establish the fundamental properties of this partial-computing red-blue
pebble game (PRBP), and compare it to the original RBP. We begin with some
simple examples where allowing partial computations can decrease the optimal
I/O cost. It is also shown that the cost can decrease by up to a linear factor
this way, but in general, it is NP-hard to decide whether partial computations
allow for a smaller cost in a specific DAG. We then discuss how $S$-partitions,
a crucial tool for deriving I/O lower bounds in RBP, can be adapted to the PRBP
model. These new tools are then used to establish lower bounds on the I/O cost
of some prominent computational tasks. Finally, we also adapt a hardness result
from RBP, showing that the optimum cost is still NP-hard to approximate in PRBP
to any reasonable factor.

### 10. [Is Sparse Matrix Reordering Effective for Sparse Matrix-Vector Multiplication?](http://arxiv.org/pdf/2506.10356v1)

Authors: Omid Asudeh, Sina Mahdipour Saravani, Gerald Sabin, Fabrice Rastello, P Sadayappan

This work evaluates the impact of sparse matrix reordering on the performance
of sparse matrix-vector multiplication across different multicore CPU
platforms. Reordering can significantly enhance performance by optimizing the
non-zero element patterns to reduce total data movement and improve the
load-balancing. We examine how these gains vary over different CPUs for
different reordering strategies, focusing on both sequential and parallel
execution. We address multiple aspects, including appropriate measurement
methodology, comparison across different kinds of reordering strategies,
consistency across machines, and impact of load imbalance.

### Digital Libraries

### 1. [Building a Media Ecosystem Observatory from Scratch: Infrastructure, Methodology, and Insights](http://arxiv.org/pdf/2506.10942v1)

Authors: Zeynep Pehlivan, Saewon Park, Alexei Sisulu Abrahams, Mika Desblancs-Patel, Benjamin David Steel, Aengus Bridgman

Understanding the flow of information across today's fragmented digital media
landscape requires scalable, cross-platform infrastructure. In this paper, we
present the Canadian Media Ecosystem Observatory, a national-scale
infrastructure designed to monitor political and media discourse across
platforms in near real time.
  Media Ecosystem Observatory (MEO) data infrastructure features custom
crawlers for major platforms, a unified indexing pipeline, and a normalization
layer that harmonizes heterogeneous schemas into a common data model. Semantic
embeddings are computed for each post to enable similarity search and
vector-based analyses such as topic modeling and clustering. Processed and raw
data are made accessible through API, dashboards and website, supporting both
automated and ad hoc research workflows. We illustrate the utility of the
observatory through example analyses of major Canadian political events,
including Meta's 2023 news ban and the recent federal elections. As a whole,
the system offers a model for digital trace infrastructure and an evolving
research platform for studying the dynamics of modern media ecosystems.

### 2. [Sheet Music Benchmark: Standardized Optical Music Recognition Evaluation](http://arxiv.org/pdf/2506.10488v1)

Authors: Juan C. Martinez-Sevilla, Joan Cerveto-Serrano, Noelia Luna, Greg Chapman, Craig Sapp, David Rizo, Jorge Calvo-Zaragoza

In this work, we introduce the Sheet Music Benchmark (SMB), a dataset of six
hundred and eighty-five pages specifically designed to benchmark Optical Music
Recognition (OMR) research. SMB encompasses a diverse array of musical
textures, including monophony, pianoform, quartet, and others, all encoded in
Common Western Modern Notation using the Humdrum **kern format. Alongside SMB,
we introduce the OMR Normalized Edit Distance (OMR-NED), a new metric tailored
explicitly for evaluating OMR performance. OMR-NED builds upon the widely-used
Symbol Error Rate (SER), offering a fine-grained and detailed error analysis
that covers individual musical elements such as note heads, beams, pitches,
accidentals, and other critical notation features. The resulting numeric score
provided by OMR-NED facilitates clear comparisons, enabling researchers and
end-users alike to identify optimal OMR approaches. Our work thus addresses a
long-standing gap in OMR evaluation, and we support our contributions with
baseline experiments using standardized SMB dataset splits for training and
assessing state-of-the-art methods.

### Discrete Mathematics

### 1. [The Freight Multimodal Transport Problem with Buses and Drones: An Integrated Approach for Last-Mile Delivery](http://arxiv.org/pdf/2506.10311v1)

Authors: E Su, Hu Qin, Jiliu Li, Rui Zhang

This paper proposes a novel freight multimodal transport problem with buses
and drones, where buses are responsible for transporting parcels to lockers at
bus stops for storage, while drones are used to deliver each parcel from the
locker to the corresponding customer. The integrated bus-drone system
synergistically expands drone service coverage using the bus network to ensure
efficient final delivery. Minimizing the total operational costs while
satisfying customer demands necessitates the joint optimization of parcel
assignments and drone flights. We model the problem into a compact
mixed-integer linear programming formulation and propose an integer programming
formulation with exponentially many variables. To address real-world scale
instances, we propose a Branch-Price-and-Benders-Cut algorithm for this
non-deterministic polynomial-time (NP)-hard problem. This algorithm,
integrating column generation and Benders decomposition within a
Branch-and-Bound framework, is developed to obtain optimal or near-optimal
solutions. Additionally, we introduce algorithmic enhancements aimed at
accelerating the convergence of the algorithm. Computational experiments on
instances generated from real-world bus data demonstrate that the proposed
algorithms outperform CPLEX regarding both efficiency and solution quality.
Moreover, our approaches can lead to over 6% cost savings compared to
situations where we determine parcel assignments and drone flights
sequentially. We evaluate the environmental advantages of integrating buses and
drones, study the impact of different cost parameters in the system, and
investigate the impact of the parcel locker configuration on performance. These
findings provide valuable managerial insights for urban logistics managers,
highlighting the potential of the integrated bus-drone system to improve
traditional last-mile delivery.

### 2. [The LLLR generalised Langton's ant](http://arxiv.org/pdf/2506.10482v1)

Authors: Victor Lutfalla

We present a short note on the dynamics of the LLLR generalised Langton's
ant. We describe two different asymptotic behaviours for the LLLR ant.

### 3. [Contributions to conjectures in planar graphs: Induced Substructures, Treewidth, and Dominating Sets](http://arxiv.org/pdf/2506.10471v1)

Authors: Kengo Enami, Naoki Matsumoto, Takamasa Yashima

Two of the most prominent unresolved conjectures in graph theory, the
Albertson-Berman conjecture and the Matheson-Tarjan conjecture, have been
extensively studied by many researchers.
  (AB) Every planar graph of order $n$ has an induced forest of order at least
$\frac{n}{2}$.
  (MT) Every plane triangulation of sufficiently large order $n$ has a
dominating set of cardinality at most $\frac{n}{4}$.
  Although partial results and weaker bounds than those originally conjectured
have been obtained, both problems remain open. To contribute to their
resolution, various generalizations and variations of the original concepts
have been investigated, such as total dominating set, induced linear forests,
and others. In this paper, we clarify the relations among several notions
related to these two major conjectures, such as connected domination and
induced outerplanar subgraphs, etc., and survey the associated conjectures. We
then provide counterexamples to some of these conjectures and establish the
best bounds on the gap between the maximum orders of induced subgraphs under
different structural conditions. In addition, we present a general upper bound
on the order of induced subgraphs in terms of treewidth, a fundamental graph
invariant.

### 4. [On the integrality Gap of Small Asymmetric Traveling Salesman Problems: A Polyhedral and Computational Approach](http://arxiv.org/pdf/2506.10671v1)

Authors: Eleonora Vercesi, Janos Barta, Luca Maria Gambardella, Stefano Gualandi, Monaldo Mastrolilli

In this paper, we investigate the integrality gap of the Asymmetric Traveling
Salesman Problem (ATSP) with respect to the linear relaxation given by the
Asymmetric Subtour Elimination Problem (ASEP) for instances with $n$ nodes,
where $n$ is small. In particular, we focus on the geometric properties and
symmetries of the ASEP polytope ($P^{n}_{ASEP}$) and its vertices. The
polytope's symmetries are exploited to design a heuristic pivoting algorithm to
search vertices where the integrality gap is maximized. Furthermore, a general
procedure for the extension of vertices from $P^{n}_{ASEP}$ to $P^{n +
1}_{ASEP}$ is defined. The generated vertices improve the known lower bounds of
the integrality gap for $ 16 \leq n \leq 22$ and, provide small hard-to-solve
ATSP instances.

### 5. [Minimality and computability of languages of G-shifts](http://arxiv.org/pdf/2506.10610v1)

Authors: Djamel Eddine Amir, Benjamin Hellouin de Menibus

Motivated by the notion of strong computable type for sets in computable
analysis, we define the notion of strong computable type for $G$-shifts, where
$G$ is a finitely generated group with decidable word problem. A $G$-shift has
strong computable type if one can compute its language from the complement of
its language. We obtain a characterization of $G$-shifts with strong computable
type in terms of a notion of minimality with respect to properties with a
bounded computational complexity. We provide a self-contained direct proof, and
also explain how this characterization can be obtained from an existing similar
characterization for sets by Amir and Hoyrup, and discuss its connexions with
results by Jeandel on closure spaces. We apply this characterization to several
classes of shifts that are minimal with respect to specific properties. This
provides a unifying approach that not only generalizes many existing results
but also has the potential to yield new findings effortlessly. In contrast to
the case of sets, we prove that strong computable type for G-shifts is
preserved under products. We conclude by discussing some generalizations and
future directions.

### 6. [Circulant TSP: Vertices of the Edge-Length Polytope and Superpolynomial Lower Bounds](http://arxiv.org/pdf/2506.10758v1)

Authors: Samuel C. Gutekunst

We study the edge-length polytope, motivated both by algorithmic research on
the Circulant Traveling Salesman Problem (Circulant TSP) and number-theoretic
research related to the Buratti-Horak-Rosa conjecture. Circulant TSP is a
special case of TSP whose overall complexity is a significant still-open
question, and where on an input with vertices $\{1, 2, ..., n\}$, the cost of
an edge $\{i, j\}$ depends only on its length $\min\{|i-j|, n-|i-j|\}$. The
edge-length polytope provides one path to solving circulant TSP instances, and
we show that it is intimately connected to the factorization of $n$: the number
of vertices scales with $n$ whenever $n$ is prime and with $n^{3/2}$ whenever
$n$ is a prime-squared, but there are a superpolynomial number of vertices
whenever $n$ is a power of 2. In contrast, the more-standard Symmetric TSP
Polytope has roughly $n!$ vertices. Hence, for Circulant TSP, a brute-force
algorithm checking every vertex is actually efficient in some cases, based on
the factorization of $n$. As an intermediate step, we give superpolynomial
lower-bounds on two combinatorial sequences related to the Buratti-Horak-Rosa
conjecture, which asks what combinations of edge lengths can comprise a
Hamiltonian path.

### 7. [Algorithmic methods of finite discrete structures. Topological graph drawing (part III)](http://arxiv.org/pdf/2506.10936v1)

Authors: Sergey Kurapov, Maxim Davidovsky

The manuscript considers mathematical models for creating a topological
drawing of a graph based on the methods of G. Ringel's vertex rotation theory.
An algorithm is presented for generating a topological drawing of a flat part
of a graph based on the selection of a basis for the cycle subspace C(G) using
the Monte Carlo method. A steepest descent method for constructing a
topological drawing of a flat subgraph is described in the manuscript. The
topological drawing of a graph is constructed using a combination of the
methods of vector intersection algebra developed by L. I. Rapport. Three stages
of constructing a flat subgraph of a non-separable graph are described. The
issues of constructing a Hamiltonian cycle based on constructing a flat
subgraph are considered. A new method for constructing a Hamiltonian cycle of a
graph based on the cycle graph of a flat subgraph is described.

### Data Structures and Algorithms

### 1. [New Approximation Guarantees for The Inventory Staggering Problem](http://arxiv.org/pdf/2506.10339v1)

Authors: Noga Alon, Danny Segev

Since its inception in the mid-60s, the inventory staggering problem has been
explored and exploited in a wide range of application domains, such as
production planning, stock control systems, warehousing, and aerospace/defense
logistics. However, even with a rich history of academic focus, we are still
very much in the dark when it comes to cornerstone computational questions
around inventory staggering and to related structural characterizations, with
our methodological toolbox being severely under-stocked.
  The central contribution of this paper consists in devising a host of
algorithmic techniques and analytical ideas -- some being entirely novel and
some leveraging well-studied concepts in combinatorics and number theory -- for
surpassing essentially all known approximation guarantees for the inventory
staggering problem. In particular, our work demonstrates that numerous
structural properties open the door for designing polynomial-time approximation
schemes, including polynomially-bounded cycle lengths, constantly-many distinct
time intervals, so-called nested instances, and pairwise coprime settings.
These findings offer substantial improvements over currently available
constant-factor approximations and resolve outstanding open questions in their
respective contexts. In parallel, we develop new theory around a number of
yet-uncharted questions, related to the sampling complexity of peak inventory
estimation as well as to the plausibility of groupwise synchronization.
Interestingly, we establish the global nature of inventory staggering, proving
that there are $n$-item instances where, for every subset of roughly $\sqrt{n}$
items, no policy improves on the worst-possible one by a factor greater than
$1+\epsilon$, whereas for the entire instance, there exists a policy that
outperforms the worst-possible one by a factor of nearly $2$, which is optimal.

### 2. [Structural Parameterizations of $k$-Planarity](http://arxiv.org/pdf/2506.10717v1)

Authors: Tatsuya Gima, Yasuaki Kobayashi, Yuto Okada

The concept of $k$-planarity is extensively studied in the context of Beyond
Planarity. A graph is $k$-planar if it admits a drawing in the plane in which
each edge is crossed at most $k$ times. The local crossing number of a graph is
the minimum integer $k$ such that it is $k$-planar. The problem of determining
whether an input graph is $1$-planar is known to be NP-complete even for
near-planar graphs [Cabello and Mohar, SIAM J. Comput. 2013], that is, the
graphs obtained from planar graphs by adding a single edge. Moreover, the local
crossing number is hard to approximate within a factor $2 - \varepsilon$ for
any $\varepsilon > 0$ [Urschel and Wellens, IPL 2021]. To address this
computational intractability, Bannister, Cabello, and Eppstein [JGAA 2018]
investigated the parameterized complexity of the case of $k = 1$, particularly
focusing on structural parameterizations on input graphs, such as treedepth,
vertex cover number, and feedback edge number. In this paper, we extend their
approach by considering the general case $k \ge 1$ and give (tight)
parameterized upper and lower bound results. In particular, we strengthen the
aforementioned lower bound results to subclasses of constant-treewidth graphs:
we show that testing $1$-planarity is NP-complete even for near-planar graphs
with feedback vertex set number at most $3$ and pathwidth at most $4$, and the
local crossing number is hard to approximate within any constant factor for
graphs with feedback vertex set number at most $2$.

### 3. [Faster CONGEST Approximation Algorithms for Maximum Weighted Independent Set in Sparse Graphs](http://arxiv.org/pdf/2506.10845v1)

Authors: Salwa Faour, Fabian Kuhn

The maximum independent set problem is a classic optimization problem that
has also been studied quite intensively in the distributed setting. While the
problem is hard to approximate in general, there are good approximation
algorithms known for several sparse graph families. In this paper, we consider
deterministic distributed CONGEST algorithms for the weighted version of the
problem in trees and graphs of bounded arboricity.
  For trees, we prove that the task of deterministically computing a
$(1-\epsilon)$-approximate solution to the maximum weight independent set
(MWIS) problem has a tight $\Theta(\log^*(n) / \epsilon)$ complexity. The lower
bound already holds on unweighted oriented paths. On the upper bound side, we
show that the bound can be achieved even in unrooted trees.
  For graphs $G=(V,E)$ of arboricity $\beta>1$, we give two algorithms. If the
sum of all node weights is $w(V)$, we show that for any $\epsilon>0$, an
independent set of weight at least $(1-\epsilon)\cdot \frac{w(V)}{4\beta}$ can
be computed in $O(\log^2(\beta/\epsilon)/\epsilon + \log^* n)$ rounds. This
result is obtained by a direct application of the local rounding framework of
Faour, Ghaffari, Grunau, Kuhn, and Rozho\v{n} [SODA '23]. We further show that
for any $\epsilon>0$, an independent set of weight at least
$(1-\epsilon)\cdot\frac{w(V)}{2\beta+1}$ can be computed in
$O(\log^3(\beta)\cdot\log(1/\epsilon)/\epsilon^2 \cdot\log n)$ rounds. This
improves on a recent result of Gil [OPODIS '23], who showed that a
$1/\lfloor(2+\epsilon)\beta\rfloor$-approximation to the MWIS problem can be
computed in $O(\beta\cdot\log n)$ rounds. As an intermediate step, we design an
algorithm to compute an independent set of total weight at least
$(1-\epsilon)\cdot\sum_{v\in V}\frac{w(v)}{deg(v)+1}$ in time
$O(\log^3(\Delta)\cdot\log(1/\epsilon)/\epsilon + \log^* n)$, where $\Delta$ is
the maximum degree of the graph.

### 4. [Circulant TSP: Vertices of the Edge-Length Polytope and Superpolynomial Lower Bounds](http://arxiv.org/pdf/2506.10758v1)

Authors: Samuel C. Gutekunst

We study the edge-length polytope, motivated both by algorithmic research on
the Circulant Traveling Salesman Problem (Circulant TSP) and number-theoretic
research related to the Buratti-Horak-Rosa conjecture. Circulant TSP is a
special case of TSP whose overall complexity is a significant still-open
question, and where on an input with vertices $\{1, 2, ..., n\}$, the cost of
an edge $\{i, j\}$ depends only on its length $\min\{|i-j|, n-|i-j|\}$. The
edge-length polytope provides one path to solving circulant TSP instances, and
we show that it is intimately connected to the factorization of $n$: the number
of vertices scales with $n$ whenever $n$ is prime and with $n^{3/2}$ whenever
$n$ is a prime-squared, but there are a superpolynomial number of vertices
whenever $n$ is a power of 2. In contrast, the more-standard Symmetric TSP
Polytope has roughly $n!$ vertices. Hence, for Circulant TSP, a brute-force
algorithm checking every vertex is actually efficient in some cases, based on
the factorization of $n$. As an intermediate step, we give superpolynomial
lower-bounds on two combinatorial sequences related to the Buratti-Horak-Rosa
conjecture, which asks what combinations of edge lengths can comprise a
Hamiltonian path.

### Emerging Technologies

### 1. [Multi-dimensional Autoscaling of Processing Services: A Comparison of Agent-based Methods](http://arxiv.org/pdf/2506.10420v1)

Authors: Boris Sedlak, Alireza Furutanpey, Zihang Wang, Víctor Casamayor Pujol, Schahram Dustdar

Edge computing breaks with traditional autoscaling due to strict resource
constraints, thus, motivating more flexible scaling behaviors using multiple
elasticity dimensions. This work introduces an agent-based autoscaling
framework that dynamically adjusts both hardware resources and internal service
configurations to maximize requirements fulfillment in constrained
environments. We compare four types of scaling agents: Active Inference, Deep Q
Network, Analysis of Structural Knowledge, and Deep Active Inference, using two
real-world processing services running in parallel: YOLOv8 for visual
recognition and OpenCV for QR code detection. Results show all agents achieve
acceptable SLO performance with varying convergence patterns. While the Deep Q
Network benefits from pre-training, the structural analysis converges quickly,
and the deep active inference agent combines theoretical foundations with
practical scalability advantages. Our findings provide evidence for the
viability of multi-dimensional agent-based autoscaling for edge environments
and encourage future work in this research direction.

### 2. [A Hybrid Heuristic Framework for Resource-Efficient Querying of Scientific Experiments Data](http://arxiv.org/pdf/2506.10422v1)

Authors: Mayank Patel, Minal Bhise

Scientific experiments and modern applications are generating large amounts
of data every day. Most organizations utilize In-house servers or Cloud
resources to manage application data and workload. The traditional database
management system (DBMS) and HTAP systems spend significant time & resources to
load the entire dataset into DBMS before starting query execution. On the other
hand, in-situ engines may reparse required data multiple times, increasing
resource utilization and data processing costs. Additionally, over or
under-allocation of resources also increases application running costs. This
paper proposes a lightweight Resource Availability &Workload aware Hybrid
Framework (RAW-HF) to optimize querying raw data by utilizing existing finite
resources efficiently. RAW-HF includes modules that help optimize the resources
required to execute a given workload and maximize the utilization of existing
resources. The impact of applying RAW-HF to real-world scientific dataset
workloads like Sloan Digital Sky Survey (SDSS) and Linked Observation Data
(LOD) presented over 90% and 85% reduction in workload execution time (WET)
compared to widely used traditional DBMS PostgreSQL. The overall CPU, IO
resource utilization, and WET have been reduced by 26%, 25%, and 26%,
respectively, while improving memory utilization by 33%, compared to the
state-of-the-art workload-aware partial loading technique (WA) proposed for
hybrid systems. A comparison of MUAR technique used by RAW-HF with machine
learning based resource allocation techniques like PCC is also presented.

### 3. [From Images to Insights: Explainable Biodiversity Monitoring with Plain Language Habitat Explanations](http://arxiv.org/pdf/2506.10559v1)

Authors: Yutong Zhou, Masahiro Ryo

Explaining why the species lives at a particular location is important for
understanding ecological systems and conserving biodiversity. However, existing
ecological workflows are fragmented and often inaccessible to non-specialists.
We propose an end-to-end visual-to-causal framework that transforms a species
image into interpretable causal insights about its habitat preference. The
system integrates species recognition, global occurrence retrieval,
pseudo-absence sampling, and climate data extraction. We then discover causal
structures among environmental features and estimate their influence on species
occurrence using modern causal inference methods. Finally, we generate
statistically grounded, human-readable causal explanations from structured
templates and large language models. We demonstrate the framework on a bee and
a flower species and report early results as part of an ongoing project,
showing the potential of the multimodal AI assistant backed up by a recommended
ecological modeling practice for describing species habitat in
human-understandable language.

### 4. [Receiving RISs: Enabling Channel Estimation and Autonomous Configuration](http://arxiv.org/pdf/2506.10662v1)

Authors: George C. Alexandropoulos, Konstantinos D. Katsanos, Evangelos Vlachos

This chapter focuses on a hardware architecture for semi-passive
Reconfigurable Intelligent Surfaces (RISs) and investigates its consideration
for boosting the performance of Multiple-Input Multiple-Output (MIMO)
communication systems. The architecture incorporates a single or multiple
radio-frequency chains to receive pilot signals via tunable absorption phase
profiles realized by the metasurface front end, as well as a controller
encompassing a baseband processing unit to carry out channel estimation, and
consequently, the optimization of the RIS reflection coefficients. A novel
channel estimation protocol, according to which the RIS receives non-orthogonal
training pilot sequences from two multi-antenna terminals via tunable
absorption phase profiles, and then, estimates the respective channels via its
signal processing unit, is presented. The channel estimates are particularly
used by the RIS controller to design the capacity-achieving reflection phase
configuration of the metasurface front end. The proposed channel estimation
algorithm, which is based on the Alternating Direction Method of Multipliers
(ADMM), profits from the RIS random spatial absorption sampling to capture the
entire signal space, and exploits the beamspace sparsity and low-rank
properties of extremely large MIMO channels, which is particularly relevant for
communication systems at the FR3 band and above. Our extensive numerical
investigations showcase the superiority of the proposed channel estimation
technique over benchmark schemes for various system and RIS hardware
configuration parameters, as well as the effectiveness of using channel
estimates at the RIS side to dynamically optimize the possibly phase-quantized
reflection coefficients of its unit elements.

### 5. [Learning Chaotic Dynamics with Neuromorphic Network Dynamics](http://arxiv.org/pdf/2506.10773v1)

Authors: Yinhao Xu, Georg A. Gottwald, Zdenka Kuncic

This study investigates how dynamical systems may be learned and modelled
with a neuromorphic network which is itself a dynamical system. The
neuromorphic network used in this study is based on a complex electrical
circuit comprised of memristive elements that produce neuro-synaptic nonlinear
responses to input electrical signals. To determine how computation may be
performed using the physics of the underlying system, the neuromorphic network
was simulated and evaluated on autonomous prediction of a multivariate chaotic
time series, implemented with a reservoir computing framework. Through
manipulating only input electrodes and voltages, optimal nonlinear dynamical
responses were found when input voltages maximise the number of memristive
components whose internal dynamics explore the entire dynamical range of the
memristor model. Increasing the network coverage with the input electrodes was
found to suppress other nonlinear responses that are less conducive to
learning. These results provide valuable insights into how a practical
neuromorphic network device can be optimised for learning complex dynamical
systems using only external control parameters.

### Formal Languages and Automata Theory

### 1. [Chance and Mass Interpretations of Probabilities in Markov Decision Processes (Extended Version)](http://arxiv.org/pdf/2506.10377v1)

Authors: Yun Chen Tsai, Kittiphon Phalakarn, S. Akshay, Ichiro Hasuo

Markov decision processes (MDPs) are a popular model for decision-making in
the presence of uncertainty. The conventional view of MDPs in verification
treats them as state transformers with probabilities defined over sequences of
states and with schedulers making random choices. An alternative view,
especially well-suited for modeling dynamical systems, defines MDPs as
distribution transformers with schedulers distributing probability masses. Our
main contribution is a unified semantical framework that accommodates these two
views and two new ones. These four semantics of MDPs arise naturally through
identifying different sources of randomness in an MDP (namely schedulers,
configurations, and transitions) and providing different ways of interpreting
these probabilities (called the chance and mass interpretations). These
semantics are systematically unified through a mathematical construct called
chance-mass (CM) classifier. As another main contribution, we study a
reachability problem in each of the two new semantics, demonstrating their
hardness and providing two algorithms for solving them.

### 2. [Minimality and computability of languages of G-shifts](http://arxiv.org/pdf/2506.10610v1)

Authors: Djamel Eddine Amir, Benjamin Hellouin de Menibus

Motivated by the notion of strong computable type for sets in computable
analysis, we define the notion of strong computable type for $G$-shifts, where
$G$ is a finitely generated group with decidable word problem. A $G$-shift has
strong computable type if one can compute its language from the complement of
its language. We obtain a characterization of $G$-shifts with strong computable
type in terms of a notion of minimality with respect to properties with a
bounded computational complexity. We provide a self-contained direct proof, and
also explain how this characterization can be obtained from an existing similar
characterization for sets by Amir and Hoyrup, and discuss its connexions with
results by Jeandel on closure spaces. We apply this characterization to several
classes of shifts that are minimal with respect to specific properties. This
provides a unifying approach that not only generalizes many existing results
but also has the potential to yield new findings effortlessly. In contrast to
the case of sets, we prove that strong computable type for G-shifts is
preserved under products. We conclude by discussing some generalizations and
future directions.

### 3. [Landauer Principle and Thermodynamics of Computation](http://arxiv.org/pdf/2506.10876v1)

Authors: Pritam Chattopadhyay, Avijit Misra, Tanmoy Pandit, Goutam Paul

According to the Landauer principle, any logically irreversible process
accompanies entropy production, which results in heat dissipation in the
environment. Erasing of information, one of the primary logically irreversible
processes, has a lower bound on heat dissipated into the environment, called
the Landauer bound (LB). However, the practical erasure processes dissipate
much more heat than the LB. Recently, there have been a few experimental
investigations to reach this bound both in the classical and quantum domains.
There has also been a spate of activities to enquire about this LB in finite
time, with finite-size heat baths, non-Markovian and nonequilibrium environment
in the quantum regime where the effects of fluctuations and correlation of the
systems with the bath can no longer be ignored. This article provides a
comprehensive review of the recent progress on the Landauer bound, which serves
as a fundamental principle in the thermodynamics of computation. We also
provide a perspective for future endeavors in these directions. Furthermore, we
review the recent exploration toward establishing energetic bounds of a
computational process. We also review the thermodynamic aspects of error
correction, which is an indispensable part of information processing and
computations. In doing so, we briefly discuss the basics of these fields to
provide a complete picture.

### Graphics

### 1. [Low-Barrier Dataset Collection with Real Human Body for Interactive Per-Garment Virtual Try-On](http://arxiv.org/pdf/2506.10468v1)

Authors: Zaiqiang Wu, Yechen Li, Jingyuan Liu, Yuki Shibata, Takayuki Hori, I-Chao Shen, Takeo Igarashi

Existing image-based virtual try-on methods are often limited to the front
view and lack real-time performance. While per-garment virtual try-on methods
have tackled these issues by capturing per-garment datasets and training
per-garment neural networks, they still encounter practical limitations: (1)
the robotic mannequin used to capture per-garment datasets is prohibitively
expensive for widespread adoption and fails to accurately replicate natural
human body deformation; (2) the synthesized garments often misalign with the
human body. To address these challenges, we propose a low-barrier approach for
collecting per-garment datasets using real human bodies, eliminating the
necessity for a customized robotic mannequin. We also introduce a hybrid person
representation that enhances the existing intermediate representation with a
simplified DensePose map. This ensures accurate alignment of synthesized
garment images with the human body and enables human-garment interaction
without the need for customized wearable devices. We performed qualitative and
quantitative evaluations against other state-of-the-art image-based virtual
try-on methods and conducted ablation studies to demonstrate the superiority of
our method regarding image quality and temporal consistency. Finally, our user
study results indicated that most participants found our virtual try-on system
helpful for making garment purchasing decisions.

### 2. [Edit360: 2D Image Edits to 3D Assets from Any Angle](http://arxiv.org/pdf/2506.10507v1)

Authors: Junchao Huang, Xinting Hu, Zhuotao Tian, Shaoshuai Shi, Li Jiang

Recent advances in diffusion models have significantly improved image
generation and editing, but extending these capabilities to 3D assets remains
challenging, especially for fine-grained edits that require multi-view
consistency. Existing methods typically restrict editing to predetermined
viewing angles, severely limiting their flexibility and practical applications.
We introduce Edit360, a tuning-free framework that extends 2D modifications to
multi-view consistent 3D editing. Built upon video diffusion models, Edit360
enables user-specific editing from arbitrary viewpoints while ensuring
structural coherence across all views. The framework selects anchor views for
2D modifications and propagates edits across the entire 360-degree range. To
achieve this, Edit360 introduces a novel Anchor-View Editing Propagation
mechanism, which effectively aligns and merges multi-view information within
the latent and attention spaces of diffusion models. The resulting edited
multi-view sequences facilitate the reconstruction of high-quality 3D assets,
enabling customizable 3D content creation.

### 3. [Transformer IMU Calibrator: Dynamic On-body IMU Calibration for Inertial Motion Capture](http://arxiv.org/pdf/2506.10580v1)

Authors: Chengxu Zuo, Jiawei Huang, Xiao Jiang, Yuan Yao, Xiangren Shi, Rui Cao, Xinyu Yi, Feng Xu, Shihui Guo, Yipeng Qin

In this paper, we propose a novel dynamic calibration method for sparse
inertial motion capture systems, which is the first to break the restrictive
absolute static assumption in IMU calibration, i.e., the coordinate drift RG'G
and measurement offset RBS remain constant during the entire motion, thereby
significantly expanding their application scenarios. Specifically, we achieve
real-time estimation of RG'G and RBS under two relaxed assumptions: i) the
matrices change negligibly in a short time window; ii) the human movements/IMU
readings are diverse in such a time window. Intuitively, the first assumption
reduces the number of candidate matrices, and the second assumption provides
diverse constraints, which greatly reduces the solution space and allows for
accurate estimation of RG'G and RBS from a short history of IMU readings in
real time. To achieve this, we created synthetic datasets of paired RG'G, RBS
matrices and IMU readings, and learned their mappings using a Transformer-based
model. We also designed a calibration trigger based on the diversity of IMU
readings to ensure that assumption ii) is met before applying our method. To
our knowledge, we are the first to achieve implicit IMU calibration (i.e.,
seamlessly putting IMUs into use without the need for an explicit calibration
process), as well as the first to enable long-term and accurate motion capture
using sparse IMUs. The code and dataset are available at
https://github.com/ZuoCX1996/TIC.

### Computer Science and Game Theory

### 1. [A voice for minorities: diversity in approval-based committee elections under incomplete or inaccurate information](http://arxiv.org/pdf/2506.10843v1)

Authors: Feline Lindeboom, Martijn Brehm, Davide Grossi, Pradeep Murukannaiah

We study diversity in approval-based committee elections with incomplete or
inaccurate information. As standard in the literature on approval-based
multi-winner voting, we define diversity according to the maximum coverage
problem, which is known to be NP-complete, with a best attainable polynomial
time approximation ratio of $1-1/\e$. In the incomplete information model,
voters can vote on only a small portion of the candidates. We suggest a greedy
algorithm and a local search algorithm that query voters and use the query
responses to approximate the total population's opinion. For both algorithms,
we prove an upper bound on the number of queries required to get a close to
$(1-1/\e)$-approximate solution with high probability. We also provide a lower
bound for the query complexity of non-adaptive algorithms, that cannot adapt
their querying strategy to readily obtained information. In the inaccurate
information setting, voters' responses are corrupted with a probability
$p\in(0,\frac{1}{2})$. We provide both an upper and a lower bound for the
number of queries required to attain a $(1-1/\e)$-approximate solution with
high probability. Finally, using real data from Polis, we see that our
algorithms perform remarkably better than the theoretical results suggest, both
with incomplete and inaccurate information.

### 2. [Equitable Mechanism Design for Facility Location](http://arxiv.org/pdf/2506.10460v1)

Authors: Toby Walsh

We consider strategy proof mechanisms for facility location which maximize
equitability between agents. As is common in the literature, we measure
equitability with the Gini index. We first prove a simple but fundamental
impossibility result that no strategy proof mechanism can bound the
approximation ratio of the optimal Gini index of utilities for one or more
facilities. We propose instead computing approximation ratios of the
complemented Gini index of utilities, and consider how well both deterministic
and randomized mechanisms approximate this. In addition, as Nash welfare is
often put forwards as an equitable compromise between egalitarian and
utilitarian outcomes, we consider how well mechanisms approximate the Nash
welfare.

### 3. [A Benchmark for Generalizing Across Diverse Team Strategies in Competitive Pokémon](http://arxiv.org/pdf/2506.10326v1)

Authors: Cameron Angliss, Jiaxun Cui, Jiaheng Hu, Arrasy Rahman, Peter Stone

Developing AI agents that can robustly adapt to dramatically different
strategic landscapes without retraining is a central challenge for multi-agent
learning. Pok\'emon Video Game Championships (VGC) is a domain with an
extraordinarily large space of possible team configurations of approximately
$10^{139}$ - far larger than those of Dota or Starcraft. The highly discrete,
combinatorial nature of team building in Pok\'emon VGC causes optimal
strategies to shift dramatically depending on both the team being piloted and
the opponent's team, making generalization uniquely challenging. To advance
research on this problem, we introduce VGC-Bench: a benchmark that provides
critical infrastructure, standardizes evaluation protocols, and supplies
human-play datasets and a range of baselines - from large-language-model agents
and behavior cloning to reinforcement learning and empirical game-theoretic
methods such as self-play, fictitious play, and double oracle. In the
restricted setting where an agent is trained and evaluated on a single-team
configuration, our methods are able to win against a professional VGC
competitor. We extensively evaluated all baseline methods over progressively
larger team sets and find that even the best-performing algorithm in the
single-team setting struggles at scaling up as team size grows. Thus, policy
generalization across diverse team strategies remains an open challenge for the
community. Our code is open sourced at
https://github.com/cameronangliss/VGC-Bench.

### 4. [Higher-Order Uncoupled Learning Dynamics and Nash Equilibrium](http://arxiv.org/pdf/2506.10874v1)

Authors: Sarah A. Toonsi, Jeff S. Shamma

We study learnability of mixed-strategy Nash Equilibrium (NE) in general
finite games using higher-order replicator dynamics as well as classes of
higher-order uncoupled heterogeneous dynamics. In higher-order uncoupled
learning dynamics, players have no access to utilities of opponents (uncoupled)
but are allowed to use auxiliary states to further process information
(higher-order). We establish a link between uncoupled learning and feedback
stabilization with decentralized control. Using this association, we show that
for any finite game with an isolated completely mixed-strategy NE, there exist
higher-order uncoupled learning dynamics that lead (locally) to that NE. We
further establish the lack of universality of learning dynamics by linking
learning to the control theoretic concept of simultaneous stabilization. We
construct two games such that any higher-order dynamics that learn the
completely mixed-strategy NE of one of these games can never learn the
completely mixed-strategy NE of the other. Next, motivated by imposing natural
restrictions on allowable learning dynamics, we introduce the Asymptotic Best
Response (ABR) property. Dynamics with the ABR property asymptotically learn a
best response in environments that are asymptotically stationary. We show that
the ABR property relates to an internal stability condition on higher-order
learning dynamics. We provide conditions under which NE are compatible with the
ABR property. Finally, we address learnability of mixed-strategy NE in the
bandit setting using a bandit version of higher-order replicator dynamics.

### Human-Computer Interaction

### 1. [Beyond Compliance: A User-Autonomy Framework for Inclusive and Customizable Web Accessibility](http://arxiv.org/pdf/2506.10324v1)

Authors: Lalitha A R

This paper proposes a shift from compliance-centered web accessibility to a
care-driven model that prioritizes user autonomy, using neurodivergent users as
a catalyst case for broader personalization needs. While accessibility
standards offer a flexible framework, they are often interpreted and
implemented as static compliance checklists, our approach reframes it as a
flexible, user-centered process. We introduce a customizable Comfort Mode
framework that allows users to adapt interface settings, such as contrast,
typography, motion, and scaling, according to their individual needs, while
retaining the brand's core visual identity. Grounded in psychological and
cognitive accessibility principles, our design supports personalization without
sacrificing creative freedom. We present both minimal and advanced
implementation models with mock-ups, demonstrating how inclusive design can be
seamlessly integrated at minimal cost. This approach aims to broaden digital
inclusivity by offering autonomy to those who require it, without imposing
changes on those who do not. The proposed system is adaptable, scalable, and
suitable for a wide range of users and brands, offering a new paradigm where
user autonomy, aesthetic integrity, and accessibility converge not through
compromise, but through choice.

### 2. [IDEA: Augmenting Design Intelligence through Design Space Exploration](http://arxiv.org/pdf/2506.10587v1)

Authors: Chuer Chen, Xiaoke Yan, Xiaoyu Qi, Nan Cao

Design spaces serve as a conceptual framework that enables designers to
explore feasible solutions through the selection and combination of design
elements. However, effective decision-making remains heavily dependent on the
designer's experience, and the absence of mathematical formalization prevents
computational support for automated design processes. To bridge this gap, we
introduce a structured representation that models design spaces with orthogonal
dimensions and discrete selectable elements. Building on this model, we present
IDEA, a decision-making framework for augmenting design intelligence through
design space exploration to generate effective outcomes. Specifically, IDEA
leverages large language models (LLMs) for constraint generation, incorporates
a Monte Carlo Tree Search (MCTS) algorithm guided by these constraints to
explore the design space efficiently, and instantiates abstract decisions into
domain-specific implementations. We validate IDEA in two design scenarios:
data-driven article composition and pictorial visualization generation,
supported by example results, expert interviews, and a user study. The
evaluation demonstrates the IDEA's adaptability across domains and its
capability to produce superior design outcomes.

### 3. [Accessible Design in Integrated Development Environments: A Think Aloud Study Exploring the Experiences of Students with ADHD](http://arxiv.org/pdf/2506.10598v1)

Authors: Luke Halpin, Phillip Benachour, Tracy Hall, Ann-Marie Houghton, Emily Winter

Coding forms a key part of computer science education in universities. As
part of this education, Integrated Development Environments (IDEs) are
essential tools for coding. However, it is currently unknown how the design of
an IDE's interface impacts on students with Attention Deficit Hyperactivity
Disorder (ADHD).
  In this study we investigated the use of IDEs by students with ADHD. We
conducted a think aloud study with nine university computing students, followed
by qualitative observational interviews to analyse their learning and
engagement with the Visual Studio Code IDE. The paper reports on these
experiences and seeks to understand the role IDEs play in the educational
setting.
  Our work also examines how digital accessibility and usability are considered
in the current design of IDEs. We analysed the qualitative data using a
thematic analysis and identified three primary themes: self-confidence,
interaction, and learning as well as various sub-themes.
  The themes and their sub-themes illustrate key areas of consideration when
designing IDEs for students with ADHD. The primary findings highlight
experiences of frustration and barriers in the current design and layout of
IDEs.
  Through our participatory approach we provide a rare insight into ADHD user
experiences around usability and accessibility, and describe the need for
better design of development environments to ensure a positive learning
experience for the students.

### 4. [Integrating Large Language Models into Text Animation: An Intelligent Editing System with Inline and Chat Interaction](http://arxiv.org/pdf/2506.10762v1)

Authors: Bao Zhang, Zihan Li, Zhenglei Liu, Huanchen Wang, Yuxin Ma

Text animation, a foundational element in video creation, enables efficient
and cost-effective communication, thriving in advertisements, journalism, and
social media. However, traditional animation workflows present significant
usability barriers for non-professionals, with intricate operational procedures
severely hindering creative productivity. To address this, we propose a Large
Language Model (LLM)-aided text animation editing system that enables real-time
intent tracking and flexible editing. The system introduces an agent-based
dual-stream pipeline that integrates context-aware inline suggestions and
conversational guidance as well as employs a semantic-animation mapping to
facilitate LLM-driven creative intent translation. Besides, the system supports
synchronized text-animation previews and parametric adjustments via unified
controls to improve editing workflow. A user study evaluates the system,
highlighting its ability to help non-professional users complete animation
workflows while validating the pipeline. The findings encourage further
exploration of integrating LLMs into a comprehensive video creation workflow.

### 5. [Grasp Prediction based on Local Finger Motion Dynamics](http://arxiv.org/pdf/2506.10818v1)

Authors: Dimitar Valkov, Pascal Kockwelp, Florian Daiber, Antonio Krüger

The ability to predict the object the user intends to grasp offers essential
contextual information and may help to leverage the effects of point-to-point
latency in interactive environments. This paper explores the feasibility and
accuracy of real-time recognition of uninstrumented objects based on hand
kinematics during reach-to-grasp actions. In a data collection study, we
recorded the hand motions of 16 participants while reaching out to grasp and
then moving real and synthetic objects. Our results demonstrate that even a
simple LSTM network can predict the time point at which the user grasps an
object with a precision better than 21 ms and the current distance to this
object with a precision better than 1 cm. The target's size can be determined
in advance with an accuracy better than 97%. Our results have implications for
designing adaptive and fine-grained interactive user interfaces in ubiquitous
and mixed-reality environments.

### 6. [(De)composing Craft: An Elementary Grammar for Sharing Expertise in Craft Workflows](http://arxiv.org/pdf/2506.10891v1)

Authors: Ritik Batra, Lydia Kim, Ilan Mandel, Amritansh Kwatra, Jane L. E., Steven J. Jackson, Thijs Roumen

Craft practices rely on evolving archives of skill and knowledge, developed
through generations of craftspeople experimenting with designs, materials, and
techniques. Better documentation of these practices enables the sharing of
knowledge and expertise between sites and generations. However, most
documentation focuses solely on the linear steps leading to final artifacts,
neglecting the tacit knowledge necessary to improvise, or adapt workflows to
meet the unique demands of each craft project. This omission limits knowledge
sharing and reduces craft to a mechanical endeavor, rather than a sophisticated
way of seeing, thinking, and doing. Drawing on expert interviews and literature
from HCI, CSCW and the social sciences, we develop an elementary grammar to
document improvisational actions of real-world craft practices. We demonstrate
the utility of this grammar with an interface called CraftLink that can be used
to analyze expert videos and semi-automatically generate documentation to
convey material and contextual variations of craft practices. Our user study
with expert crocheters (N=7) using this interface evaluates our grammar's
effectiveness in capturing and sharing expert knowledge with other
craftspeople, offering new pathways for computational systems to support
collaborative archives of knowledge and practice within communities.

### 7. [Instance-Based Transfer Learning with Similarity-Aware Subject Selection for Cross-Subject SSVEP-Based BCIs](http://arxiv.org/pdf/2506.10933v1)

Authors: Ziwen Wang, Yue Zhang, Zhiqiang Zhang, Sheng Quan Xie, Alexander Lanzon, William P. Heath, Zhenhong Li

Steady-state visual evoked potential (SSVEP)-based brain-computer interfaces
(BCIs) can achieve high recognition accuracy with sufficient training data.
Transfer learning presents a promising solution to alleviate data requirements
for the target subject by leveraging data from source subjects; however,
effectively addressing individual variability among both target and source
subjects remains a challenge. This paper proposes a novel transfer learning
framework, termed instance-based task-related component analysis (iTRCA), which
leverages knowledge from source subjects while considering their individual
contributions. iTRCA extracts two types of features: (1) the subject-general
feature, capturing shared information between source and target subjects in a
common latent space, and (2) the subject-specific feature, preserving the
unique characteristics of the target subject. To mitigate negative transfer, we
further design an enhanced framework, subject selection-based iTRCA (SS-iTRCA),
which integrates a similarity-based subject selection strategy to identify
appropriate source subjects for transfer based on their task-related components
(TRCs). Comparative evaluations on the Benchmark, BETA, and a self-collected
dataset demonstrate the effectiveness of the proposed iTRCA and SS-iTRCA
frameworks. This study provides a potential solution for developing
high-performance SSVEP-based BCIs with reduced target subject data.

### 8. [Extended Creativity: A Conceptual Framework for Understanding Human-AI Creative Relations](http://arxiv.org/pdf/2506.10249v1)

Authors: Andrea Gaggioli, Sabrina Bartolotta, Andrea Ubaldi, Katusha Gerardini, Eleonora Diletta Sarcinella, Alice Chirico

Artificial Intelligence holds significant potential to enhance human
creativity. However, achieving this vision requires a clearer understanding of
how such enhancement can be effectively realized. Adopting the perspective of
distributed creativity, we identify three primary modes through which AI can
contribute to creative processes: Support, where AI acts as a tool; Synergy,
where AI and humans collaborate in complementary ways; and Symbiosis, where
human and AI cognition become so integrated that they form a unified creative
system. These modes are defined along two key dimensions: the level of
technical autonomy exhibited by the AI system and the degree of perceived
agency attributed to it. We examine how each configuration influences different
levels of creativity - from everyday problem-solving to paradigm-shifting
innovation - and discuss the theoretical, ethical, and design implications.

### 9. [Collective Bargaining in the Information Economy Can Address AI-Driven Power Concentration](http://arxiv.org/pdf/2506.10272v1)

Authors: Nicholas Vincent, Matthew Prewitt, Hanlin Li

This position paper argues that there is an urgent need to restructure
markets for the information that goes into AI systems. Specifically, producers
of information goods (such as journalists, researchers, and creative
professionals) need to be able to collectively bargain with AI product builders
in order to receive reasonable terms and a sustainable return on the
informational value they contribute. We argue that without increased market
coordination or collective bargaining on the side of these primary information
producers, AI will exacerbate a large-scale "information market failure" that
will lead not only to undesirable concentration of capital, but also to a
potential "ecological collapse" in the informational commons. On the other
hand, collective bargaining in the information economy can create market
frictions and aligned incentives necessary for a pro-social, sustainable AI
future. We provide concrete actions that can be taken to support a
coalition-based approach to achieve this goal. For example, researchers and
developers can establish technical mechanisms such as federated data management
tools and explainable data value estimations, to inform and facilitate
collective bargaining in the information economy. Additionally, regulatory and
policy interventions may be introduced to support trusted data intermediary
organizations representing guilds or syndicates of information producers.

### 10. [MLLM-Based UI2Code Automation Guided by UI Layout Information](http://arxiv.org/pdf/2506.10376v1)

Authors: Fan Wu, Cuiyun Gao, Shuqing Li, Xin-Cheng Wen, Qing Liao

Converting user interfaces into code (UI2Code) is a crucial step in website
development, which is time-consuming and labor-intensive. The automation of
UI2Code is essential to streamline this task, beneficial for improving the
development efficiency. There exist deep learning-based methods for the task;
however, they heavily rely on a large amount of labeled training data and
struggle with generalizing to real-world, unseen web page designs. The advent
of Multimodal Large Language Models (MLLMs) presents potential for alleviating
the issue, but they are difficult to comprehend the complex layouts in UIs and
generate the accurate code with layout preserved. To address these issues, we
propose LayoutCoder, a novel MLLM-based framework generating UI code from
real-world webpage images, which includes three key modules: (1) Element
Relation Construction, which aims at capturing UI layout by identifying and
grouping components with similar structures; (2) UI Layout Parsing, which aims
at generating UI layout trees for guiding the subsequent code generation
process; and (3) Layout-Guided Code Fusion, which aims at producing the
accurate code with layout preserved. For evaluation, we build a new benchmark
dataset which involves 350 real-world websites named Snap2Code, divided into
seen and unseen parts for mitigating the data leakage issue, besides the
popular dataset Design2Code. Extensive evaluation shows the superior
performance of LayoutCoder over the state-of-the-art approaches. Compared with
the best-performing baseline, LayoutCoder improves 10.14% in the BLEU score and
3.95% in the CLIP score on average across all datasets.

### Information Retrieval

### 1. [Context-Adaptive Graph Neural Networks for Next POI Recommendation](http://arxiv.org/pdf/2506.10329v1)

Authors: Yu Lei, Limin Shen, Zhu Sun, Tiantian He, Yew-Soon Ong

Next Point-of-Interest (POI) recommendation is a critical task in
location-based services, aiming to predict users' next visits based on their
check-in histories. While many existing methods leverage Graph Neural Networks
(GNNs) to incorporate collaborative information and improve recommendation
accuracy, most of them model each type of context using separate graphs,
treating different factors in isolation. This limits their ability to model the
co-influence of multiple contextual factors on user transitions during message
propagation, resulting in suboptimal attention weights and recommendation
performance. Furthermore, they often prioritize sequential components as the
primary predictor, potentially undermining the semantic and structural
information encoded in the POI embeddings learned by GNNs. To address these
limitations, we propose a Context-Adaptive Graph Neural Networks (CAGNN) for
next POI recommendation, which dynamically adjusts attention weights using
edge-specific contextual factors and enables mutual enhancement between
graph-based and sequential components. Specifically, CAGNN introduces (1) a
context-adaptive attention mechanism that jointly incorporates different types
of contextual factors into the attention computation during graph propagation,
enabling the model to dynamically capture collaborative and context-dependent
transition patterns; (2) a graph-sequential mutual enhancement module, which
aligns the outputs of the graph- and sequential-based modules via the KL
divergence, enabling mutual enhancement of both components. Experimental
results on three real-world datasets demonstrate that CAGNN consistently
outperforms state-of-the-art methods. Meanwhile, theoretical guarantees are
provided that our context-adaptive attention mechanism improves the
expressiveness of POI representations.

### 2. [Constructing and Evaluating Declarative RAG Pipelines in PyTerrier](http://arxiv.org/pdf/2506.10802v1)

Authors: Craig Macdonald, Jinyuan Fang, Andrew Parry, Zaiqiao Meng

Search engines often follow a pipeline architecture, where complex but
effective reranking components are used to refine the results of an initial
retrieval. Retrieval augmented generation (RAG) is an exciting application of
the pipeline architecture, where the final component generates a coherent
answer for the users from the retrieved documents. In this demo paper, we
describe how such RAG pipelines can be formulated in the declarative PyTerrier
architecture, and the advantages of doing so. Our PyTerrier-RAG extension for
PyTerrier provides easy access to standard RAG datasets and evaluation
measures, state-of-the-art LLM readers, and using PyTerrier's unique operator
notation, easy-to-build pipelines. We demonstrate the succinctness of indexing
and RAG pipelines on standard datasets (including Natural Questions) and how to
build on the larger PyTerrier ecosystem with state-of-the-art sparse,
learned-sparse, and dense retrievers, and other neural rankers.

### 3. [Towards Understanding Bias in Synthetic Data for Evaluation](http://arxiv.org/pdf/2506.10301v1)

Authors: Hossein A. Rahmani, Varsha Ramineni, Nick Craswell, Bhaskar Mitra, Emine Yilmaz

Test collections are crucial for evaluating Information Retrieval (IR)
systems. Creating a diverse set of user queries for these collections can be
challenging, and obtaining relevance judgments, which indicate how well
retrieved documents match a query, is often costly and resource-intensive.
Recently, generating synthetic datasets using Large Language Models (LLMs) has
gained attention in various applications. While previous work has used LLMs to
generate synthetic queries or documents to improve ranking models, using LLMs
to create synthetic test collections is still relatively unexplored. Previous
work~\cite{rahmani2024synthetic} showed that synthetic test collections have
the potential to be used for system evaluation, however, more analysis is
needed to validate this claim. In this paper, we thoroughly investigate the
reliability of synthetic test collections constructed using LLMs, where LLMs
are used to generate synthetic queries, labels, or both. In particular, we
examine the potential biases that might occur when such test collections are
used for evaluation. We first empirically show the presence of such bias in
evaluation results and analyse the effects it might have on system evaluation.
We further validate the presence of such bias using a linear mixed-effects
model. Our analysis shows that while the effect of bias present in evaluation
results obtained using synthetic test collections could be significant, for
e.g.~computing absolute system performance, its effect may not be as
significant in comparing relative system performance. Codes and data are
available at: https://github.com/rahmanidashti/BiasSyntheticData.

### 4. [An Analysis of Datasets, Metrics and Models in Keyphrase Generation](http://arxiv.org/pdf/2506.10346v1)

Authors: Florian Boudin, Akiko Aizawa

Keyphrase generation refers to the task of producing a set of words or
phrases that summarises the content of a document. Continuous efforts have been
dedicated to this task over the past few years, spreading across multiple lines
of research, such as model architectures, data resources, and use-case
scenarios. Yet, the current state of keyphrase generation remains unknown as
there has been no attempt to review and analyse previous work. In this paper,
we bridge this gap by presenting an analysis of over 50 research papers on
keyphrase generation, offering a comprehensive overview of recent progress,
limitations, and open challenges. Our findings highlight several critical
issues in current evaluation practices, such as the concerning similarity among
commonly-used benchmark datasets and inconsistencies in metric calculations
leading to overestimated performances. Additionally, we address the limited
availability of pre-trained models by releasing a strong PLM-based model for
keyphrase generation as an effort to facilitate future research.

### 5. [LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture](http://arxiv.org/pdf/2506.10347v1)

Authors: Yanhui Li, Dongxia Wang, Zhu Sun, Haonan Zhang, Huizhong Guo

Recently, Graph Neural Networks (GNNs) have become the dominant approach for
Knowledge Graph-aware Recommender Systems (KGRSs) due to their proven
effectiveness. Building upon GNN-based KGRSs, Self-Supervised Learning (SSL)
has been incorporated to address the sparity issue, leading to longer training
time. However, through extensive experiments, we reveal that: (1)compared to
other KGRSs, the existing GNN-based KGRSs fail to keep their superior
performance under sparse interactions even with SSL. (2) More complex models
tend to perform worse in sparse interaction scenarios and complex mechanisms,
like attention mechanism, can be detrimental as they often increase learning
difficulty. Inspired by these findings, we propose LightKG, a simple yet
powerful GNN-based KGRS to address sparsity issues. LightKG includes a
simplified GNN layer that encodes directed relations as scalar pairs rather
than dense embeddings and employs a linear aggregation framework, greatly
reducing the complexity of GNNs. Additionally, LightKG incorporates an
efficient contrastive layer to implement SSL. It directly minimizes the node
similarity in original graph, avoiding the time-consuming subgraph generation
and comparison required in previous SSL methods. Experiments on four benchmark
datasets show that LightKG outperforms 12 competitive KGRSs in both sparse and
dense scenarios while significantly reducing training time. Specifically, it
surpasses the best baselines by an average of 5.8\% in recommendation accuracy
and saves 84.3\% of training time compared to KGRSs with SSL. Our code is
available at https://github.com/1371149/LightKG.

### 6. [TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning](http://arxiv.org/pdf/2506.10380v1)

Authors: Xiaohan Yu, Pu Jian, Chong Chen

Retrieval-Augmented Generation (RAG) has demonstrated considerable
effectiveness in open-domain question answering. However, when applied to
heterogeneous documents, comprising both textual and tabular components,
existing RAG approaches exhibit critical limitations. The prevailing practice
of flattening tables and chunking strategies disrupts the intrinsic tabular
structure, leads to information loss, and undermines the reasoning capabilities
of LLMs in multi-hop, global queries. To address these challenges, we propose
TableRAG, an hybrid framework that unifies textual understanding and complex
manipulations over tabular data. TableRAG iteratively operates in four steps:
context-sensitive query decomposition, text retrieval, SQL programming and
execution, and compositional intermediate answer generation. We also develop
HeteQA, a novel benchmark designed to evaluate the multi-hop heterogeneous
reasoning capabilities. Experimental results demonstrate that TableRAG
consistently outperforms existing baselines on both public datasets and our
HeteQA, establishing a new state-of-the-art for heterogeneous document question
answering. We release TableRAG at https://github.com/yxh-y/TableRAG/tree/main.

### 7. [Reasoning RAG via System 1 or System 2: A Survey on Reasoning Agentic Retrieval-Augmented Generation for Industry Challenges](http://arxiv.org/pdf/2506.10408v1)

Authors: Jintao Liang, Gang Su, Huifeng Lin, You Wu, Rui Zhao, Ziyue Li

Retrieval-Augmented Generation (RAG) has emerged as a powerful framework to
overcome the knowledge limitations of Large Language Models (LLMs) by
integrating external retrieval with language generation. While early RAG
systems based on static pipelines have shown effectiveness in well-structured
tasks, they struggle in real-world scenarios requiring complex reasoning,
dynamic retrieval, and multi-modal integration. To address these challenges,
the field has shifted toward Reasoning Agentic RAG, a paradigm that embeds
decision-making and adaptive tool use directly into the retrieval process. In
this paper, we present a comprehensive review of Reasoning Agentic RAG methods,
categorizing them into two primary systems: predefined reasoning, which follows
fixed modular pipelines to boost reasoning, and agentic reasoning, where the
model autonomously orchestrates tool interaction during inference. We analyze
representative techniques under both paradigms, covering architectural design,
reasoning strategies, and tool coordination. Finally, we discuss key research
challenges and propose future directions to advance the flexibility,
robustness, and applicability of reasoning agentic RAG systems. Our collection
of the relevant research has been organized into a
https://github.com/ByebyeMonica/Reasoning-Agentic-RAG.

### 8. [SHORE: A Long-term User Lifetime Value Prediction Model in Digital Games](http://arxiv.org/pdf/2506.10487v1)

Authors: Shuaiqi Sun, Congde Yuan, Haoqiang Yang, Mengzhuo Guo, Guiying Wei, Jiangbo Tian

In digital gaming, long-term user lifetime value (LTV) prediction is
essential for monetization strategy, yet presents major challenges due to
delayed payment behavior, sparse early user data, and the presence of
high-value outliers. While existing models typically rely on either short-cycle
observations or strong distributional assumptions, such approaches often
underestimate long-term value or suffer from poor robustness. To address these
issues, we propose SHort-cycle auxiliary with Order-preserving REgression
(SHORE), a novel LTV prediction framework that integrates short-horizon
predictions (e.g., LTV-15 and LTV-30) as auxiliary tasks to enhance long-cycle
targets (e.g., LTV-60). SHORE also introduces a hybrid loss function combining
order-preserving multi-class classification and a dynamic Huber loss to
mitigate the influence of zero-inflation and outlier payment behavior.
Extensive offline and online experiments on real-world datasets demonstrate
that SHORE significantly outperforms existing baselines, achieving a 47.91\%
relative reduction in prediction error in online deployment. These results
highlight SHORE's practical effectiveness and robustness in industrial-scale
LTV prediction for digital games.

### 9. [Macro Graph of Experts for Billion-Scale Multi-Task Recommendation](http://arxiv.org/pdf/2506.10520v1)

Authors: Hongyu Yao, Zijin Hong, Hao Chen, Yuanchen Bei, Zhiqing Li, Qijie Shen, Zuobin Ying, Huan Gong, Feiran Huang

Graph-based multi-task learning at billion-scale presents a significant
challenge, as different tasks correspond to distinct billion-scale graphs.
Traditional multi-task learning methods often neglect these graph structures,
relying solely on individual user and item embeddings. However, disregarding
graph structures overlooks substantial potential for improving performance. In
this paper, we introduce the Macro Graph of Expert (MGOE) framework, the first
approach capable of leveraging macro graph embeddings to capture task-specific
macro features while modeling the correlations between task-specific experts.
Specifically, we propose the concept of a Macro Graph Bottom, which, for the
first time, enables multi-task learning models to incorporate graph information
effectively. We design the Macro Prediction Tower to dynamically integrate
macro knowledge across tasks. MGOE has been deployed at scale, powering
multi-task learning for the homepage of a leading billion-scale recommender
system. Extensive offline experiments conducted on three public benchmark
datasets demonstrate its superiority over state-of-the-art multi-task learning
methods, establishing MGOE as a breakthrough in multi-task graph-based
recommendation. Furthermore, online A/B tests confirm the superiority of MGOE
in billion-scale recommender systems.

### 10. [Conversational Search: From Fundamentals to Frontiers in the LLM Era](http://arxiv.org/pdf/2506.10635v1)

Authors: Fengran Mo, Chuan Meng, Mohammad Aliannejadi, Jian-Yun Nie

Conversational search enables multi-turn interactions between users and
systems to fulfill users' complex information needs. During this interaction,
the system should understand the users' search intent within the conversational
context and then return the relevant information through a flexible,
dialogue-based interface. The recent powerful large language models (LLMs) with
capacities of instruction following, content generation, and reasoning, attract
significant attention and advancements, providing new opportunities and
challenges for building up intelligent conversational search systems. This
tutorial aims to introduce the connection between fundamentals and the emerging
topics revolutionized by LLMs in the context of conversational search. It is
designed for students, researchers, and practitioners from both academia and
industry. Participants will gain a comprehensive understanding of both the core
principles and cutting-edge developments driven by LLMs in conversational
search, equipping them with the knowledge needed to contribute to the
development of next-generation conversational search systems.

### Machine Learning

### 1. [Graph-MLLM: Harnessing Multimodal Large Language Models for Multimodal Graph Learning](http://arxiv.org/pdf/2506.10282v1)

Authors: Jiajin Liu, Dongzhe Fan, Jiacheng Shen, Chuanhao Ji, Daochen Zha, Qiaoyu Tan

Multimodal Large Language Models (MLLMs) have demonstrated remarkable
capabilities in representing and understanding diverse modalities. However,
they typically focus on modality alignment in a pairwise manner while
overlooking structural relationships across data points. Integrating
multimodality with structured graph information (i.e., multimodal graphs, MMGs)
is essential for real-world applications such as social networks, healthcare,
and recommendation systems. Existing MMG learning methods fall into three
paradigms based on how they leverage MLLMs: Encoder, Aligner, and Predictor.
MLLM-as-Encoder focuses on enhancing graph neural networks (GNNs) via
multimodal feature fusion; MLLM-as-Aligner aligns multimodal attributes in
language or hidden space to enable LLM-based graph reasoning; MLLM-as-Predictor
treats MLLMs as standalone reasoners with in-context learning or fine-tuning.
Despite their advances, the MMG field lacks a unified benchmark to fairly
evaluate across these approaches, making it unclear what progress has been
made. To bridge this gap, we present Graph-MLLM, a comprehensive benchmark for
multimodal graph learning by systematically evaluating these three paradigms
across six datasets with different domains. Through extensive experiments, we
observe that jointly considering the visual and textual attributes of the nodes
benefits graph learning, even when using pre-trained text-to-image alignment
models (e.g., CLIP) as encoders. We also find that converting visual attributes
into textual descriptions further improves performance compared to directly
using visual inputs. Moreover, we observe that fine-tuning MLLMs on specific
MMGs can achieve state-of-the-art results in most scenarios, even without
explicit graph structure information. We hope that our open-sourced library
will facilitate rapid, equitable evaluation and inspire further innovative
research in this field.

### 2. [PyLO: Towards Accessible Learned Optimizers in PyTorch](http://arxiv.org/pdf/2506.10315v1)

Authors: Paul Janson, Benjamin Therien, Quentin Anthony, Xiaolong Huang, Abhinav Moudgil, Eugene Belilovsky

Learned optimizers have been an active research topic over the past decade,
with increasing progress toward practical, general-purpose optimizers that can
serve as drop-in replacements for widely used methods like Adam. However,
recent advances -- such as VeLO, which was meta-trained for 4000 TPU-months --
remain largely inaccessible to the broader community, in part due to their
reliance on JAX and the absence of user-friendly packages for applying the
optimizers after meta-training. To address this gap, we introduce PyLO, a
PyTorch-based library that brings learned optimizers to the broader machine
learning community through familiar, widely adopted workflows. Unlike prior
work focused on synthetic or convex tasks, our emphasis is on applying learned
optimization to real-world large-scale pre-training tasks. Our release includes
a CUDA-accelerated version of the small_fc_lopt learned optimizer architecture
from (Metz et al., 2022a), delivering substantial speedups -- from 39.36 to
205.59 samples/sec throughput for training ViT B/16 with batch size 32. PyLO
also allows us to easily combine learned optimizers with existing optimization
tools such as learning rate schedules and weight decay. When doing so, we find
that learned optimizers can substantially benefit. Our code is available at
https://github.com/Belilovsky-Lab/pylo

### 3. [History-Aware Neural Operator: Robust Data-Driven Constitutive Modeling of Path-Dependent Materials](http://arxiv.org/pdf/2506.10352v1)

Authors: Binyao Guo, Zihan Lin, QiZhi He

This study presents an end-to-end learning framework for data-driven modeling
of path-dependent inelastic materials using neural operators. The framework is
built on the premise that irreversible evolution of material responses,
governed by hidden dynamics, can be inferred from observable data.
  We develop the History-Aware Neural Operator (HANO), an autoregressive model
that predicts path-dependent material responses from short segments of recent
strain-stress history without relying on hidden state variables, thereby
overcoming self-consistency issues commonly encountered in recurrent neural
network (RNN)-based models. Built on a Fourier-based neural operator backbone,
HANO enables discretization-invariant learning. To enhance its ability to
capture both global loading patterns and critical local path dependencies, we
embed a hierarchical self-attention mechanism that facilitates multiscale
feature extraction.
  Beyond ensuring self-consistency, HANO mitigates sensitivity to initial
hidden states, a commonly overlooked issue that can lead to instability in
recurrent models when applied to generalized loading paths. By modeling
stress-strain evolution as a continuous operator rather than relying on fixed
input-output mappings, HANO naturally accommodates varying path discretizations
and exhibits robust performance under complex conditions, including irregular
sampling, multi-cycle loading, noisy data, and pre-stressed states. We evaluate
HANO on two benchmark problems: elastoplasticity with hardening and progressive
anisotropic damage in brittle solids. Results show that HANO consistently
outperforms baseline models in predictive accuracy, generalization, and
robustness. With its demonstrated capabilities, HANO provides an effective
data-driven surrogate for simulating inelastic materials and is well-suited for
integration with classical numerical solvers.

### 4. [TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree](http://arxiv.org/pdf/2506.10355v1)

Authors: Yu-Yang Qian, Yuan-Ze Xu, Zhen-Yu Zhang, Peng Zhao, Zhi-Hua Zhou

Many real-world applications collect data in a streaming environment, where
learning tasks are encountered sequentially. This necessitates continual
learning (CL) to update models online, enabling adaptation to new tasks while
preserving past knowledge to prevent catastrophic forgetting. Nowadays, with
the flourish of large pre-trained models (LPMs), efficiency has become
increasingly critical for CL, due to their substantial computational demands
and growing parameter sizes. In this paper, we introduce TreeLoRA (K-D Tree of
Low-Rank Adapters), a novel approach that constructs layer-wise adapters by
leveraging hierarchical gradient similarity to enable efficient CL,
particularly for LPMs. To reduce the computational burden of task similarity
estimation, we employ bandit techniques to develop an algorithm based on lower
confidence bounds to efficiently explore the task structure. Furthermore, we
use sparse gradient updates to facilitate parameter optimization, making the
approach better suited for LPMs. Theoretical analysis is provided to justify
the rationale behind our approach, and experiments on both vision transformers
(ViTs) and large language models (LLMs) demonstrate the effectiveness and
efficiency of our approach across various domains, including vision and natural
language processing tasks.

### 5. [EQA-RM: A Generative Embodied Reward Model with Test-time Scaling](http://arxiv.org/pdf/2506.10389v1)

Authors: Yuhang Chen, Zhen Tan, Tianlong Chen

Reward Models (RMs), vital for large model alignment, are underexplored for
complex embodied tasks like Embodied Question Answering (EQA) where nuanced
evaluation of agents' spatial, temporal, and logical understanding is critical
yet not considered by generic approaches. We introduce EQA-RM, a novel
generative multimodal reward model specifically architected for EQA, trained
via our innovative Contrastive Group Relative Policy Optimization (C-GRPO)
strategy to learn fine-grained behavioral distinctions. The generative nature
of EQA-RM provides interpretable, structured reward feedback (beyond simple
scalars), uniquely enabling test-time scaling to dynamically adjust evaluation
granularity, from concise scores to detailed critiques of reasoning and
grounding, at inference without retraining. Concurrently, we introduce
EQARewardBench, a new benchmark built on OpenEQA for standardized EQA reward
model assessment. Demonstrating high sample efficiency, EQA-RM (fine-tuning
Qwen2-VL-2B-Instruct) achieves 61.9\% accuracy on EQA-RM-Bench with only 700
samples, outperforming strong proprietary baselines, including
Gemini-2.5-Flash, GPT-4o, Claude-3.5-Haiku, and open-sourced state-of-the-art
models such as RoVRM and VisualPRM. The code and dataset can be found here
https://github.com/UNITES-Lab/EQA-RM.

### 6. [Generative Algorithms for Wildfire Progression Reconstruction from Multi-Modal Satellite Active Fire Measurements and Terrain Height](http://arxiv.org/pdf/2506.10404v1)

Authors: Bryan Shaddy, Brianna Binder, Agnimitra Dasgupta, Haitong Qin, James Haley, Angel Farguell, Kyle Hilburn, Derek V. Mallia, Adam Kochanski, Jan Mandel, Assad Oberai

Increasing wildfire occurrence has spurred growing interest in wildfire
spread prediction. However, even the most complex wildfire models diverge from
observed progression during multi-day simulations, motivating need for data
assimilation. A useful approach to assimilating measurement data into complex
coupled atmosphere-wildfire models is to estimate wildfire progression from
measurements and use this progression to develop a matching atmospheric state.
In this study, an approach is developed for estimating fire progression from
VIIRS active fire measurements, GOES-derived ignition times, and terrain height
data. A conditional Generative Adversarial Network is trained with simulations
of historic wildfires from the atmosphere-wildfire model WRF-SFIRE, thus
allowing incorporation of WRF-SFIRE physics into estimates. Fire progression is
succinctly represented by fire arrival time, and measurements for training are
obtained by applying an approximate observation operator to WRF-SFIRE
solutions, eliminating need for satellite data during training. The model is
trained on tuples of fire arrival times, measurements, and terrain, and once
trained leverages measurements of real fires and corresponding terrain data to
generate samples of fire arrival times. The approach is validated on five
Pacific US wildfires, with results compared against high-resolution perimeters
measured via aircraft, finding an average Sorensen-Dice coefficient of 0.81.
The influence of terrain height on the arrival time inference is also evaluated
and it is observed that terrain has minimal influence when the inference is
conditioned on satellite measurements.

### 7. [Data-Driven Soil Organic Carbon Sampling: Integrating Spectral Clustering with Conditioned Latin Hypercube Optimization](http://arxiv.org/pdf/2506.10419v1)

Authors: Weiying Zhao, Aleksei Unagaev, Natalia Efremova

Soil organic carbon (SOC) monitoring often relies on selecting representative
field sampling locations based on environmental covariates. We propose a novel
hybrid methodology that integrates spectral clustering - an unsupervised
machine learning technique with conditioned Latin hypercube sampling (cLHS) to
enhance the representativeness of SOC sampling. In our approach, spectral
clustering partitions the study area into $K$ homogeneous zones using
multivariate covariate data, and cLHS is then applied within each zone to
select sampling locations that collectively capture the full diversity of
environmental conditions. This hybrid spectral-cLHS method ensures that even
minor but important environmental clusters are sampled, addressing a key
limitation of vanilla cLHS which can overlook such areas. We demonstrate on a
real SOC mapping dataset that spectral-cLHS provides more uniform coverage of
covariate feature space and spatial heterogeneity than standard cLHS. This
improved sampling design has the potential to yield more accurate SOC
predictions by providing better-balanced training data for machine learning
models.

### 8. [MNN-LLM: A Generic Inference Engine for Fast Large Language Model Deployment on Mobile Devices](http://arxiv.org/pdf/2506.10443v1)

Authors: Zhaode Wang, Jingbang Yang, Xinyu Qian, Shiwen Xing, Xiaotang Jiang, Chengfei Lv, Shengyu Zhang

Large language models (LLMs) have demonstrated exceptional performance across
a variety of tasks. However, their substantial scale leads to significant
computational resource consumption during inference, resulting in high costs.
Consequently, edge device inference presents a promising solution. The primary
challenges of edge inference include memory usage and inference speed. This
paper introduces MNN-LLM, a framework specifically designed to accelerate the
deployment of large language models on mobile devices. MNN-LLM addresses the
runtime characteristics of LLMs through model quantization and DRAM-Flash
hybrid storage, effectively reducing memory usage. It rearranges weights and
inputs based on mobile CPU instruction sets and GPU characteristics while
employing strategies such as multicore load balancing, mixed-precision
floating-point operations, and geometric computations to enhance performance.
Notably, MNN-LLM achieves up to a 8.6x speed increase compared to current
mainstream LLM-specific frameworks.

### 9. [Equivariant Neural Diffusion for Molecule Generation](http://arxiv.org/pdf/2506.10532v1)

Authors: François Cornet, Grigory Bartosh, Mikkel N. Schmidt, Christian A. Naesseth

We introduce Equivariant Neural Diffusion (END), a novel diffusion model for
molecule generation in 3D that is equivariant to Euclidean transformations.
Compared to current state-of-the-art equivariant diffusion models, the key
innovation in END lies in its learnable forward process for enhanced generative
modelling. Rather than pre-specified, the forward process is parameterized
through a time- and data-dependent transformation that is equivariant to rigid
transformations. Through a series of experiments on standard molecule
generation benchmarks, we demonstrate the competitive performance of END
compared to several strong baselines for both unconditional and conditional
generation.

### 10. [Data-driven Day Ahead Market Prices Forecasting: A Focus on Short Training Set Windows](http://arxiv.org/pdf/2506.10536v1)

Authors: Vasilis Michalakopoulos, Christoforos Menos-Aikateriniadis, Elissaios Sarmas, Antonis Zakynthinos, Pavlos S. Georgilakis, Dimitris Askounis

This study investigates the performance of machine learning models in
forecasting electricity Day-Ahead Market (DAM) prices using short historical
training windows, with a focus on detecting seasonal trends and price spikes.
We evaluate four models, namely LSTM with Feed Forward Error Correction (FFEC),
XGBoost, LightGBM, and CatBoost, across three European energy markets (Greece,
Belgium, Ireland) using feature sets derived from ENTSO-E forecast data.
Training window lengths range from 7 to 90 days, allowing assessment of model
adaptability under constrained data availability. Results indicate that
LightGBM consistently achieves the highest forecasting accuracy and robustness,
particularly with 45 and 60 day training windows, which balance temporal
relevance and learning depth. Furthermore, LightGBM demonstrates superior
detection of seasonal effects and peak price events compared to LSTM and other
boosting models. These findings suggest that short-window training approaches,
combined with boosting methods, can effectively support DAM forecasting in
volatile, data-scarce environments.

### Neural and Evolutionary Computing

### 1. [Contrastive Matrix Completion with Denoising and Augmented Graph Views for Robust Recommendation](http://arxiv.org/pdf/2506.10658v1)

Authors: Narges Nemati, Mostafa Haghir Chehreghani

Matrix completion is a widely adopted framework in recommender systems, as
predicting the missing entries in the user-item rating matrix enables a
comprehensive understanding of user preferences. However, current graph neural
network (GNN)-based approaches are highly sensitive to noisy or irrelevant
edges--due to their inherent message-passing mechanisms--and are prone to
overfitting, which limits their generalizability. To overcome these challenges,
we propose a novel method called Matrix Completion using Contrastive Learning
(MCCL). Our approach begins by extracting local neighborhood subgraphs for each
interaction and subsequently generates two distinct graph representations. The
first representation emphasizes denoising by integrating GNN layers with an
attention mechanism, while the second is obtained via a graph variational
autoencoder that aligns the feature distribution with a standard prior. A
mutual learning loss function is employed during training to gradually
harmonize these representations, enabling the model to capture common patterns
and significantly enhance its generalizability. Extensive experiments on
several real-world datasets demonstrate that our approach not only improves the
numerical accuracy of the predicted scores--achieving up to a 0.8% improvement
in RMSE--but also produces superior rankings with improvements of up to 36% in
ranking metrics.

### Networking and Internet Architecture

### 1. [Large Language Models-Empowered Wireless Networks: Fundamentals, Architecture, and Challenges](http://arxiv.org/pdf/2506.10651v1)

Authors: Latif U. Khan, Maher Guizani, Sami Muhaidat, Choong Seon Hong

The rapid advancement of wireless networks has resulted in numerous
challenges stemming from their extensive demands for quality of service towards
innovative quality of experience metrics (e.g., user-defined metrics in terms
of sense of physical experience for haptics applications). In the meantime,
large language models (LLMs) emerged as promising solutions for many difficult
and complex applications/tasks. These lead to a notion of the integration of
LLMs and wireless networks. However, this integration is challenging and needs
careful attention in design. Therefore, in this article, we present a notion of
rational wireless networks powered by \emph{telecom LLMs}, namely,
\emph{LLM-native wireless systems}. We provide fundamentals, vision, and a case
study of the distributed implementation of LLM-native wireless systems. In the
case study, we propose a solution based on double deep Q-learning (DDQN) that
outperforms existing DDQN solutions. Finally, we provide open challenges.

### 2. [Energy-Efficient Deep Learning for Traffic Classification on Microcontrollers](http://arxiv.org/pdf/2506.10851v1)

Authors: Adel Chehade, Edoardo Ragusa, Paolo Gastaldo, Rodolfo Zunino

In this paper, we present a practical deep learning (DL) approach for
energy-efficient traffic classification (TC) on resource-limited
microcontrollers, which are widely used in IoT-based smart systems and
communication networks. Our objective is to balance accuracy, computational
efficiency, and real-world deployability. To that end, we develop a lightweight
1D-CNN, optimized via hardware-aware neural architecture search (HW-NAS), which
achieves 96.59% accuracy on the ISCX VPN-NonVPN dataset with only 88.26K
parameters, a 20.12K maximum tensor size, and 10.08M floating-point operations
(FLOPs). Moreover, it generalizes across various TC tasks, with accuracies
ranging from 94% to 99%. To enable deployment, the model is quantized to INT8,
suffering only a marginal 1-2% accuracy drop relative to its Float32
counterpart. We evaluate real-world inference performance on two
microcontrollers: the high-performance STM32F746G-DISCO and the cost-sensitive
Nucleo-F401RE. The deployed model achieves inference latencies of 31.43ms and
115.40ms, with energy consumption of 7.86 mJ and 29.10 mJ per inference,
respectively. These results demonstrate the feasibility of on-device encrypted
traffic analysis, paving the way for scalable, low-power IoT security
solutions.

### 3. [Dynamic Beyond 5G and 6G Connectivity: Leveraging NTN and RIS Synergies for Optimized Coverage and Capacity in High-Density Environments](http://arxiv.org/pdf/2506.10900v1)

Authors: Valdemar Farré, Juan Estrada, David Vega, Luis F Urquiza-Aguiar, Juan A. Vásquez Peralvo, Symeon Chatzinotas

The increasing demand for reliable, high-capacity communication during
large-scale outdoor events poses significant challenges for traditional
Terrestrial Networks (TNs), which often struggle to provide consistent coverage
in high-density environments. This paper presents a novel 6G radio network
planning framework that integrates Non-Terrestrial Networks (NTNs) with
Reconfigurable Intelligent Surfaces (RISs) to deliver ubiquitous coverage and
enhanced network capacity. Our framework overcomes the limitations of
conventional deployable base stations by leveraging NTN architectures,
including Low Earth Orbit (LEO) satellites and passive RIS platforms seamlessly
integrated with Beyond 5G (B5G) TNs. By incorporating advanced B5G technologies
such as Massive Multiple Input Multiple Output (mMIMO) and beamforming, and by
optimizing spectrum utilization across the C, S, and Ka bands, we implement a
rigorous interference management strategy based on a dynamic SINR model.
Comprehensive calculations and simulations validate the proposed framework,
demonstrating significant improvements in connectivity, reliability, and
cost-efficiency in crowded scenarios. This integration strategy represents a
promising solution for meeting the evolving demands of future 6G networks.

### 4. [Agentic Semantic Control for Autonomous Wireless Space Networks: Extending Space-O-RAN with MCP-Driven Distributed Intelligence](http://arxiv.org/pdf/2506.10925v1)

Authors: Eduardo Baena, Paolo Testolina, Michele Polese, Sergi Aliaga, Andrew Benincasa, Dimitrios Koutsonikolas, Josep Jornet, Tommaso Melodia

Lunar surface operations impose stringent requirements on wireless
communication systems, including autonomy, robustness to disruption, and the
ability to adapt to environmental and mission-driven context. While Space-O-RAN
provides a distributed orchestration model aligned with 3GPP standards, its
decision logic is limited to static policies and lacks semantic integration. We
propose a novel extension incorporating a semantic agentic layer enabled by the
Model Context Protocol (MCP) and Agent-to-Agent (A2A) communication protocols,
allowing context-aware decision making across real-time, near-real-time, and
non-real-time control layers. Distributed cognitive agents deployed in rovers,
landers, and lunar base stations implement wireless-aware coordination
strategies, including delay-adaptive reasoning and bandwidth-aware semantic
compression, while interacting with multiple MCP servers to reason over
telemetry, locomotion planning, and mission constraints.

### Robotics

### 1. [Multi-Timescale Dynamics Model Bayesian Optimization for Plasma Stabilization in Tokamaks](http://arxiv.org/pdf/2506.10287v1)

Authors: Rohit Sonker, Alexandre Capone, Andrew Rothstein, Hiro Josep Farre Kaga, Egemen Kolemen, Jeff Schneider

Machine learning algorithms often struggle to control complex real-world
systems. In the case of nuclear fusion, these challenges are exacerbated, as
the dynamics are notoriously complex, data is poor, hardware is subject to
failures, and experiments often affect dynamics beyond the experiment's
duration. Existing tools like reinforcement learning, supervised learning, and
Bayesian optimization address some of these challenges but fail to provide a
comprehensive solution. To overcome these limitations, we present a multi-scale
Bayesian optimization approach that integrates a high-frequency data-driven
dynamics model with a low-frequency Gaussian process. By updating the Gaussian
process between experiments, the method rapidly adapts to new data, refining
the predictions of the less reliable dynamical model. We validate our approach
by controlling tearing instabilities in the DIII-D nuclear fusion plant.
Offline testing on historical data shows that our method significantly
outperforms several baselines. Results on live experiments on the DIII-D
tokamak, conducted under high-performance plasma scenarios prone to
instabilities, shows a 50% success rate, marking a 117% improvement over
historical outcomes.

### 2. [Towards more efficient quantitative safety validation of residual risk for assisted and automated driving](http://arxiv.org/pdf/2506.10363v1)

Authors: Daniel Betschinske, Malte Schrimpf, Steven Peters, Kamil Klonecki, Jan Peter Karch, Moritz Lippert

The safety validation of Advanced Driver Assistance Systems (ADAS) and
Automated Driving Systems (ADS) increasingly demands efficient and reliable
methods to quantify residual risk while adhering to international standards
such as ISO 21448. Traditionally, Field Operational Testing (FOT) has been
pivotal for macroscopic safety validation of automotive driving functions up to
SAE automation level 2. However, state-of-the-art derivations for empirical
safety demonstrations using FOT often result in impractical testing efforts,
particularly at higher automation levels. Even at lower automation levels, this
limitation - coupled with the substantial costs associated with FOT - motivates
the exploration of approaches to enhance the efficiency of FOT-based
macroscopic safety validation. Therefore, this publication systematically
identifies and evaluates state-of-the-art Reduction Approaches (RAs) for FOT,
including novel methods reported in the literature. Based on an analysis of ISO
21448, two models are derived: a generic model capturing the argumentation
components of the standard, and a base model, exemplarily applied to Automatic
Emergency Braking (AEB) systems, establishing a baseline for the real-world
driving requirement for a Quantitative Safety Validation of Residual Risk
(QSVRR). Subsequently, the RAs are assessed using four criteria:
quantifiability, threats to validity, missing links, and black box
compatibility, highlighting potential benefits, inherent limitations, and
identifying key areas for further research. Our evaluation reveals that, while
several approaches offer potential, none are free from missing links or other
substantial shortcomings. Moreover, no identified alternative can fully replace
FOT, reflecting its crucial role in the safety validation of ADAS and ADS.

### 3. [Are We Generalizing from the Exception? An In-the-Wild Study on Group-Sensitive Conversation Design in Human-Agent Interactions](http://arxiv.org/pdf/2506.10462v1)

Authors: Ana Müller, Sabina Jeschke, Anja Richert

This paper investigates the impact of a group-adaptive conversation design in
two socially interactive agents (SIAs) through two real-world studies. Both
SIAs - Furhat, a social robot, and MetaHuman, a virtual agent - were equipped
with a conversational artificial intelligence (CAI) backend combining hybrid
retrieval and generative models. The studies were carried out in an in-the-wild
setting with a total of $N = 188$ participants who interacted with the SIAs -
in dyads, triads or larger groups - at a German museum. Although the results
did not reveal a significant effect of the group-sensitive conversation design
on perceived satisfaction, the findings provide valuable insights into the
challenges of adapting CAI for multi-party interactions and across different
embodiments (robot vs.\ virtual agent), highlighting the need for multimodal
strategies beyond linguistic pluralization. These insights contribute to the
fields of Human-Agent Interaction (HAI), Human-Robot Interaction (HRI), and
broader Human-Machine Interaction (HMI), providing insights for future research
on effective dialogue adaptation in group settings.

### 4. [In-Hand Object Pose Estimation via Visual-Tactile Fusion](http://arxiv.org/pdf/2506.10787v1)

Authors: Felix Nonnengießer, Alap Kshirsagar, Boris Belousov, Jan Peters

Accurate in-hand pose estimation is crucial for robotic object manipulation,
but visual occlusion remains a major challenge for vision-based approaches.
This paper presents an approach to robotic in-hand object pose estimation,
combining visual and tactile information to accurately determine the position
and orientation of objects grasped by a robotic hand. We address the challenge
of visual occlusion by fusing visual information from a wrist-mounted RGB-D
camera with tactile information from vision-based tactile sensors mounted on
the fingertips of a robotic gripper. Our approach employs a weighting and
sensor fusion module to combine point clouds from heterogeneous sensor types
and control each modality's contribution to the pose estimation process. We use
an augmented Iterative Closest Point (ICP) algorithm adapted for weighted point
clouds to estimate the 6D object pose. Our experiments show that incorporating
tactile information significantly improves pose estimation accuracy,
particularly when occlusion is high. Our method achieves an average pose
estimation error of 7.5 mm and 16.7 degrees, outperforming vision-only
baselines by up to 20%. We also demonstrate the ability of our method to
perform precise object manipulation in a real-world insertion task.

### 5. [RationalVLA: A Rational Vision-Language-Action Model with Dual System](http://arxiv.org/pdf/2506.10826v1)

Authors: Wenxuan Song, Jiayi Chen, Wenxue Li, Xu He, Han Zhao, Pengxiang Ding Shiyan Su, Feilong Tang, Xuelian Cheng, Donglin Wang, Zongyuan Ge, Xinhu Zheng, Zhe Liu, Hesheng Wang, Yunhui Liu, Haoang Li

A fundamental requirement for real-world robotic deployment is the ability to
understand and respond to natural language instructions. Existing
language-conditioned manipulation tasks typically assume that instructions are
perfectly aligned with the environment. This assumption limits robustness and
generalization in realistic scenarios where instructions may be ambiguous,
irrelevant, or infeasible. To address this problem, we introduce RAtional
MAnipulation (RAMA), a new benchmark that challenges models with both unseen
executable instructions and defective ones that should be rejected. In RAMA, we
construct a dataset with over 14,000 samples, including diverse defective
instructions spanning six dimensions: visual, physical, semantic, motion,
safety, and out-of-context. We further propose the Rational
Vision-Language-Action model (RationalVLA). It is a dual system for robotic
arms that integrates the high-level vision-language model with the low-level
manipulation policy by introducing learnable latent space embeddings. This
design enables RationalVLA to reason over instructions, reject infeasible
commands, and execute manipulation effectively. Experiments demonstrate that
RationalVLA outperforms state-of-the-art baselines on RAMA by a 14.5% higher
success rate and 0.94 average task length, while maintaining competitive
performance on standard manipulation tasks. Real-world trials further validate
its effectiveness and robustness in practical applications. Our project page is
https://irpn-eai.github.io/rationalvla.

### 6. [Invariant Extended Kalman Filter for Autonomous Surface Vessels with Partial Orientation Measurements](http://arxiv.org/pdf/2506.10850v1)

Authors: Derek Benham, Easton Potokar, Joshua G. Mangelson

Autonomous surface vessels (ASVs) are increasingly vital for marine science,
offering robust platforms for underwater mapping and inspection. Accurate state
estimation, particularly of vehicle pose, is paramount for precise seafloor
mapping, as even small surface deviations can have significant consequences
when sensing the seafloor below. To address this challenge, we propose an
Invariant Extended Kalman Filter (InEKF) framework designed to integrate
partial orientation measurements. While conventional estimation often relies on
relative position measurements to fixed landmarks, open ocean ASVs primarily
observe a receding horizon. We leverage forward-facing monocular cameras to
estimate roll and pitch with respect to this horizon, which provides
yaw-ambiguous partial orientation information. To effectively utilize these
measurements within the InEKF, we introduce a novel framework for incorporating
such partial orientation data. This approach contrasts with traditional InEKF
implementations that assume full orientation measurements and is particularly
relevant for planar vehicle motion constrained to a "seafaring plane." This
paper details the developed InEKF framework; its integration with horizon-based
roll/pitch observations and dual-antenna GPS heading measurements for ASV state
estimation; and provides a comparative analysis against the InEKF using full
orientation and a Multiplicative EKF (MEKF). Our results demonstrate the
efficacy and robustness of the proposed partial orientation measurements for
accurate ASV state estimation in open ocean environments.

### 7. [Modeling Trust Dynamics in Robot-Assisted Delivery: Impact of Trust Repair Strategies](http://arxiv.org/pdf/2506.10884v1)

Authors: Dong Hae Mangalindan, Karthik Kandikonda, Ericka Rovira, Vaibhav Srivastava

With increasing efficiency and reliability, autonomous systems are becoming
valuable assistants to humans in various tasks. In the context of
robot-assisted delivery, we investigate how robot performance and trust repair
strategies impact human trust. In this task, while handling a secondary task,
humans can choose to either send the robot to deliver autonomously or manually
control it. The trust repair strategies examined include short and long
explanations, apology and promise, and denial.
  Using data from human participants, we model human behavior using an
Input-Output Hidden Markov Model (IOHMM) to capture the dynamics of trust and
human action probabilities. Our findings indicate that humans are more likely
to deploy the robot autonomously when their trust is high. Furthermore, state
transition estimates show that long explanations are the most effective at
repairing trust following a failure, while denial is most effective at
preventing trust loss.
  We also demonstrate that the trust estimates generated by our model are
isomorphic to self-reported trust values, making them interpretable. This model
lays the groundwork for developing optimal policies that facilitate real-time
adjustment of human trust in autonomous systems.

### 8. [Vib2Move: In-Hand Object Reconfiguration via Fingertip Micro-Vibrations](http://arxiv.org/pdf/2506.10923v1)

Authors: Xili Yi, Nima Fazeli

We introduce Vib2Move, a novel approach for in-hand object reconfiguration
that uses fingertip micro-vibrations and gravity to precisely reposition planar
objects. Our framework comprises three key innovations. First, we design a
vibration-based actuator that dynamically modulates the effective finger-object
friction coefficient, effectively emulating changes in gripping force. Second,
we derive a sliding motion model for objects clamped in a parallel gripper with
two symmetric, variable-friction contact patches. Third, we propose a motion
planner that coordinates end-effector finger trajectories and fingertip
vibrations to achieve the desired object pose. In real-world trials, Vib2Move
consistently yields final positioning errors below 6 mm, demonstrating
reliable, high-precision manipulation across a variety of planar objects. For
more results and information, please visit https://vib2move.github.io.

### 9. [GENMANIP: LLM-driven Simulation for Generalizable Instruction-Following Manipulation](http://arxiv.org/pdf/2506.10966v1)

Authors: Ning Gao, Yilun Chen, Shuai Yang, Xinyi Chen, Yang Tian, Hao Li, Haifeng Huang, Hanqing Wang, Tai Wang, Jiangmiao Pang

Robotic manipulation in real-world settings remains challenging, especially
regarding robust generalization. Existing simulation platforms lack sufficient
support for exploring how policies adapt to varied instructions and scenarios.
Thus, they lag behind the growing interest in instruction-following foundation
models like LLMs, whose adaptability is crucial yet remains underexplored in
fair comparisons. To bridge this gap, we introduce GenManip, a realistic
tabletop simulation platform tailored for policy generalization studies. It
features an automatic pipeline via LLM-driven task-oriented scene graph to
synthesize large-scale, diverse tasks using 10K annotated 3D object assets. To
systematically assess generalization, we present GenManip-Bench, a benchmark of
200 scenarios refined via human-in-the-loop corrections. We evaluate two policy
types: (1) modular manipulation systems integrating foundation models for
perception, reasoning, and planning, and (2) end-to-end policies trained
through scalable data collection. Results show that while data scaling benefits
end-to-end methods, modular systems enhanced with foundation models generalize
more effectively across diverse scenarios. We anticipate this platform to
facilitate critical insights for advancing policy generalization in realistic
conditions. Project Page: https://genmanip.axi404.top/.

### 10. [Using Language and Road Manuals to Inform Map Reconstruction for Autonomous Driving](http://arxiv.org/pdf/2506.10317v1)

Authors: Akshar Tumu, Henrik I. Christensen, Marcell Vazquez-Chanlatte, Chikao Tsuchiya, Dhaval Bhanderi

Lane-topology prediction is a critical component of safe and reliable
autonomous navigation. An accurate understanding of the road environment aids
this task. We observe that this information often follows conventions encoded
in natural language, through design codes that reflect the road structure and
road names that capture the road functionality. We augment this information in
a lightweight manner to SMERF, a map-prior-based online lane-topology
prediction model, by combining structured road metadata from OSM maps and
lane-width priors from Road design manuals with the road centerline encodings.
We evaluate our method on two geo-diverse complex intersection scenarios. Our
method shows improvement in both lane and traffic element detection and their
association. We report results using four topology-aware metrics to
comprehensively assess the model performance. These results demonstrate the
ability of our approach to generalize and scale to diverse topologies and
conditions.

### Software Engineering

### 1. [Minimizing False Positives in Static Bug Detection via LLM-Enhanced Path Feasibility Analysis](http://arxiv.org/pdf/2506.10322v1)

Authors: Xueying Du, Kai Yu, Chong Wang, Yi Zou, Wentai Deng, Zuoyu Ou, Xin Peng, Lingming Zhang, Yiling Lou

Static bug analyzers play a crucial role in ensuring software quality.
However, existing analyzers for bug detection in large codebases often suffer
from high false positive rates. This is primarily due to the limited
capabilities of analyzers in path feasibility validation with multiple
conditional branches and complex data dependencies. While current LLM-based
approaches attempt to address this issue, their effectiveness remains limited
due to insufficient constraint cascade analysis and scalability challenges in
large projects. To address this challenge, we propose an iterative path
feasibility analysis framework LLM4PFA. By leveraging LLM agent based targeted
constraint reasoning, and key context-aware analysis driven by agent planning,
LLM4PFA effectively enhances complex inter-procedural path feasibility analysis
for minimizing false positives in static bug detection. Evaluation results show
that LLM4PFA precisely filters out 72% to 96% false positives reported during
static bug detection, significantly outperforming all the baselines by 41.1% -
105.7% improvements; meanwhile LLM4PFA only misses 3 real bugs of 45 true
positives.

### 2. [AutoGEEval++: A Multi-Level and Multi-Geospatial-Modality Automated Evaluation Framework for Large Language Models in Geospatial Code Generation on Google Earth Engine](http://arxiv.org/pdf/2506.10365v1)

Authors: Shuyang Hou, Zhangxiao Shen, Huayi Wu, Haoyue Jiao, Ziqi Liu, Lutong Xie, Chang Liu, Jianyuan Liang, Yaxian Qing, Xiaopu Zhang, Dehua Peng, Zhipeng Gui, Xuefeng Guan

Geospatial code generation is becoming a key frontier in integrating
artificial intelligence with geo-scientific analysis, yet standardised
automated evaluation tools for this task remain absent. This study presents
AutoGEEval++, an enhanced framework building on AutoGEEval, and the first
automated assessment system for large language models (LLMs) generating
geospatial code on Google Earth Engine (GEE). It supports diverse data
modalities and varying task complexities. Built on the GEE Python API,
AutoGEEval++ features a benchmark dataset-AutoGEEval++-Bench-with 6,365 test
cases across 26 data types and three task categories: unit, combo, and theme
tests. It includes a submission programme and a judge module to realise an
end-to-end automated evaluation pipeline from code generation to
execution-based validation. The framework adopts multi-dimensional
metrics-accuracy, resource usage, run-time efficiency, and error
types-balancing hallucination control and efficiency, and enabling boundary
testing and error pattern analysis. Using AutoGEEval++, we evaluate 24
state-of-the-art LLMs (as of June 2025), including general-purpose,
reasoning-enhanced, code-centric, and geoscience-specific models. Results
reveal clear performance, stability, and error differences across task types,
model designs, and deployment settings, confirming AutoGEEval++'s practical
value and scalability in vertical-domain code generation. This work establishes
the first standardised evaluation protocol and foundational benchmark for
GEE-based LLM code generation, providing a unified basis for performance
comparison and a methodological framework for systematic, domain-specific code
evaluation.

### 3. [Towards Understanding Bugs in Distributed Training and Inference Frameworks for Large Language Models](http://arxiv.org/pdf/2506.10426v1)

Authors: Xiao Yu, Haoxuan Chen, Feifei Niu, Xing Hu, Jacky Wai Keung, Xin Xia

With the rapid development of large language models (LLMs), distributed
training and inference frameworks like DeepSpeed have become essential for
scaling model training and inference across multiple GPUs or nodes. However,
the increasing complexity of these frameworks brings non-trivial software bugs,
which may degrade training performance, cause unexpected failures, and result
in significant resource waste. Understanding framework bugs' characteristics is
fundamental for quality assurance, allowing the design of more effective
debugging and repair methods. Thus, our paper conducts the first large-scale
empirical analysis of 308 fixed bugs across three popular distributed
training/inference frameworks: DeepSpeed, Megatron-LM, and Colossal-AI. We
examine bug symptoms, root causes, bug identification and fixing efforts, and
common low-effort fixing strategies. Additionally, the distributed nature of
these frameworks introduces unique bug root causes, such as allocation strategy
error and distributed communication error. Diagnosing and fixing complex bugs
remains challenging due to factors like the disconnect between symptoms and
root causes, high bug reproduction costs, and low-level or cross-component
interactions. Interestingly, we observe that 48% of bug fixes require minimal
code changes (<=10 LOC) and follow simple strategies such as conditional logic
optimization, parameter handling enhancement, or version compatibility
handling, indicating potential for automation. Based on these insights, we
offer several implications for improving the reliability of both distributed
training and inference frameworks and their dependent LLM projects, while also
identifying opportunities to leverage LLM-based tools for automated debugging
and repair.

### 4. [EXPEREPAIR: Dual-Memory Enhanced LLM-based Repository-Level Program Repair](http://arxiv.org/pdf/2506.10484v1)

Authors: Fangwen Mu, Junjie Wang, Lin Shi, Song Wang, Shoubin Li, Qing Wang

Automatically repairing software issues remains a fundamental challenge at
the intersection of software engineering and AI. Although recent advancements
in Large Language Models (LLMs) have demonstrated potential for
repository-level repair tasks, current methodologies exhibit two notable
limitations: (1) they often address issues in isolation, neglecting to
incorporate insights from previously resolved issues, and (2) they rely on
static and rigid prompting strategies, which constrain their ability to
generalize across diverse and evolving issue scenarios. Inspired by the dual
memory systems of human cognition, where episodic and semantic memories work
synergistically to support human reasoning and decision-making, we propose
ExpeRepair, a novel LLM-based approach that continuously learns from historical
repair experiences through dual-channel knowledge accumulation. ExpeRepair
organizes historical repair experiences into two complementary memories: an
episodic memory that stores concrete repair demonstrations, and a semantic
memory that encodes abstract reflective insights. At inference time, ExpeRepair
activates both memory systems by retrieving relevant demonstrations from
episodic memory and recalling high-level repair insights from semantic memory.
It further enhances adaptability through dynamic prompt composition,
synergistically integrating both memory types to replace static prompts with
context-aware, experience-driven prompts. Experiments on the SWE-bench Lite
benchmark demonstrate that ExpeRepair achieves a pass@1 score of 49.3% with
Claude 3.7 Sonnet, outperforming all state-of-the-art open-source methods.

### 5. [AdaptiveLLM: A Framework for Selecting Optimal Cost-Efficient LLM for Code-Generation Based on CoT Length](http://arxiv.org/pdf/2506.10525v1)

Authors: Junhang Cheng, Fang Liu, Chengru Wu, Li Zhang

While Large Language Models (LLMs) have significantly advanced code
generation efficiency, they face inherent challenges in balancing performance
and inference costs across diverse programming tasks. Dynamically selecting the
optimal LLM based on task difficulty and resource constraints offers a
promising approach to achieve an optimal balance between efficiency and
performance. However, existing model selection methods are resource-intensive
and often neglect cost efficiency. Moreover, these approaches rely on
human-annotated difficulty labels that are frequently inaccessible in
real-world settings and may not align with the LLM's own assessment of task
difficulty. In this paper, we introduce AdaptiveLLM, a framework that
dynamically selects optimal LLMs for a given coding task by automatically
assessing task difficulty. Our framework first estimates task difficulty using
Chain-of-Thought lengths generated by reasoning model, clusters these into
three difficulty levels via k-means, and fine-tunes CodeBERT to embed
difficulty-aware features. A trained XGBoost classifier then selects the best
model for each problem, optimizing the performance-cost trade-off. Experimental
results show that AdaptiveLLM achieves a 7.86% improvement in pass@1 score
while reducing resource consumption by 88.9% compared to baseline method
ComplexityNet. When compared to a single model, AdaptiveLLM demonstrates an
approximately 15% accuracy improvement, while maintaining the same level of
cost consumption. Apart from that, the difficulty assessment using CoT provides
more reliable selection criteria than human evaluation. Our replication package
is available at https://github.com/cjhCoder7/AdaptiveLLM.

### 6. [Not One to Rule Them All: Mining Meaningful Code Review Orders From GitHub](http://arxiv.org/pdf/2506.10654v1)

Authors: Abir Bouraffa, Carolin Brandt, Andy Zaidmann, Walid Maalej

Developers use tools such as GitHub pull requests to review code, discuss
proposed changes, and request modifications. While changed files are commonly
presented in alphabetical order, this does not necessarily coincide with the
reviewer's preferred navigation sequence. This study investigates the different
navigation orders developers follow while commenting on changes submitted in
pull requests. We mined code review comments from 23,241 pull requests in 100
popular Java and Python repositories on GitHub to analyze the order in which
the reviewers commented on the submitted changes. Our analysis shows that for
44.6% of pull requests, the reviewers comment in a non-alphabetical order.
Among these pull requests, we identified traces of alternative meaningful
orders: 20.6% (2,134) followed a largest-diff-first order, 17.6% (1,827) were
commented in the order of the files' similarity to the pull request's title and
description, and 29% (1,188) of pull requests containing changes to both
production and test files adhered to a test-first order. We also observed that
the proportion of reviewed files to total submitted files was significantly
higher in non-alphabetically ordered reviews, which also received slightly
fewer approvals from reviewers, on average. Our findings highlight the need for
additional support during code reviews, particularly for larger pull requests,
where reviewers are more likely to adopt complex strategies rather than
following a single predefined order.

### 7. [From Tea Leaves to System Maps: Context-awareness in Monitoring Operational Machine Learning Models](http://arxiv.org/pdf/2506.10770v1)

Authors: Joran Leest, Claudia Raibulet, Patricia Lago, Ilias Gerostathopoulos

Machine learning (ML) models in production do not fail due to statistical
anomalies in their input data; they fail due to contextual misalignment -- when
their environment deviates from training assumptions, leading to unreliable
predictions. Effective ML monitoring requires rich contextual information to
move beyond detecting statistical shifts toward meaningful alerts and
systematic root-cause analysis. Yet, surprisingly, despite extensive research
in ML monitoring and related disciplines (drift detection, data validation,
out-of-distribution detection), there is no shared understanding of how to use
contextual information -- striking, given that monitoring involves
interpretation of information in context. In response, this paper presents a
systematic review to characterize and structure the various types of contextual
information in this domain. Our analysis examines 94 primary studies across
data mining, databases, software engineering, and ML. We introduce the
Contextual System--Aspect--Representation (C-SAR) framework, a conceptual model
that synthesizes our findings. We also identify 20 recurring and potentially
reusable patterns of specific system, aspect, and representation combinations,
and map them to the monitoring activities they support. This study provides a
new perspective on ML monitoring: from interpreting "tea leaves" of
observational statistics into constructing and managing "system maps" that
enable systematic and reliable ML monitoring practices.

### 8. [Evaluating Large Language Models on Non-Code Software Engineering Tasks](http://arxiv.org/pdf/2506.10833v1)

Authors: Fabian C. Peña, Steffen Herbold

Large Language Models (LLMs) have demonstrated remarkable capabilities in
code understanding and generation; however, their effectiveness on non-code
Software Engineering (SE) tasks remains underexplored. We present the first
comprehensive benchmark, which we name `Software Engineering Language
Understanding' (SELU), for evaluating LLMs on 17 non-code tasks, spanning from
identifying whether a requirement is functional or non-functional to estimating
the effort and complexity of backlog items. SELU covers classification,
regression, Named Entity Recognition (NER), and Masked Language Modeling (MLM)
targets, with data drawn from diverse sources such as code repositories, issue
tracking systems, and developer forums. We fine-tune 22 open-source LLMs,
prompt two proprietary alternatives, and train two baselines. Performance is
measured using metrics such as F1-macro, SMAPE, F1-micro, and accuracy, and
compared via the Bayesian signed-rank test. Our results show that
moderate-scale decoder-only models consistently form a top-tier, exhibiting
high mean performance and low across-task variance, while domain adaptation via
code-focused pre-training might yield only modest improvements. These insights
guide model selection for non-code SE workflows and highlight directions for
expanding SELU to generative and design-oriented scenarios.

### 9. [MultiCoSim: A Python-based Multi-Fidelity Co-Simulation Framework](http://arxiv.org/pdf/2506.10869v1)

Authors: Quinn Thibeault, Giulia Pedrielli

Simulation is a foundational tool for the analysis and testing of
cyber-physical systems (CPS), underpinning activities such as algorithm
development, runtime monitoring, and system verification. As CPS grow in
complexity and scale, particularly in safety-critical and learning-enabled
settings, accurate analysis and synthesis increasingly rely on the rapid use of
simulation experiments. Because CPS inherently integrate hardware, software,
and physical processes, simulation platforms must support co-simulation of
heterogeneous components at varying levels of fidelity. Despite recent advances
in high-fidelity modeling of hardware, firmware, and physics, co-simulation in
diverse environments remains challenging. These limitations hinder the
development of reusable benchmarks and impede the use of simulation for
automated and comparative evaluation.
  Existing simulation tools often rely on rigid configurations, lack automation
support, and present obstacles to portability and modularity. Many are
configured through static text files or impose constraints on how simulation
components are represented and connected, making it difficult to flexibly
compose systems or integrate components across platforms.
  To address these challenges, we introduce MultiCoSim, a Python-based
simulation framework that enables users to define, compose, and configure
simulation components programmatically. MultiCoSim supports distributed,
component-based co-simulation and allows seamless substitution and
reconfiguration of components. We demonstrate the flexibility of MultiCoSim
through case studies that include co-simulations involving custom
automaton-based controllers, as well as integration with off-the-shelf
platforms like the PX4 autopilot for aerial robotics. These examples highlight
MultiCoSim's capability to streamline CPS simulation pipelines for research and
development.

### 10. [AI-Based Software Vulnerability Detection: A Systematic Literature Review](http://arxiv.org/pdf/2506.10280v1)

Authors: Samiha Shimmi, Hamed Okhravi, Mona Rahimi

Software vulnerabilities in source code pose serious cybersecurity risks,
prompting a shift from traditional detection methods (e.g., static analysis,
rule-based matching) to AI-driven approaches. This study presents a systematic
review of software vulnerability detection (SVD) research from 2018 to 2023,
offering a comprehensive taxonomy of techniques, feature representations, and
embedding methods. Our analysis reveals that 91% of studies use AI-based
methods, with graph-based models being the most prevalent. We identify key
limitations, including dataset quality, reproducibility, and interpretability,
and highlight emerging opportunities in underexplored techniques such as
federated learning and quantum neural networks, providing a roadmap for future
research.

### Systems and Control

### 1. [Learning-Based Stable Optimal Control for Infinite-Time Nonlinear Regulation Problems](http://arxiv.org/pdf/2506.10291v1)

Authors: Han Wang, Di Wu, Lin Cheng, Shengping Gong, Xu Huang

Infinite-time nonlinear optimal regulation control is widely utilized in
aerospace engineering as a systematic method for synthesizing stable
controllers. However, conventional methods often rely on linearization
hypothesis, while recent learning-based approaches rarely consider stability
guarantees. This paper proposes a learning-based framework to learn a stable
optimal controller for nonlinear optimal regulation problems. First, leveraging
the equivalence between Pontryagin Maximum Principle (PMP) and
Hamilton-Jacobi-Bellman (HJB) equation, we improve the backward generation of
optimal examples (BGOE) method for infinite-time optimal regulation problems. A
state-transition-matrix-guided data generation method is then proposed to
efficiently generate a complete dataset that covers the desired state space.
Finally, we incorporate the Lyapunov stability condition into the learning
framework, ensuring the stability of the learned optimal policy by jointly
learning the optimal value function and control policy. Simulations on three
nonlinear optimal regulation problems show that the learned optimal policy
achieves near-optimal regulation control and the code is provided at
https://github.com/wong-han/PaperNORC

### 2. [Synthesizing Min-Max Control Barrier Functions For Switched Affine Systems](http://arxiv.org/pdf/2506.10296v1)

Authors: Sara Kamali, Guillaume O. Berger, Sriram Sankaranarayanan

We study the problem of synthesizing non-smooth control barrier functions
(CBFs) for continuous-time switched affine systems. Switched affine systems are
defined by a set of affine dynamical modes, wherein the control consists of a
state-based switching signal that determines the current operating mode. The
control barrier functions seek to maintain the system state inside a control
invariant set that excludes a given set of unsafe states. We consider CBFs that
take the form of pointwise minima and maxima over a finite set of affine
functions. Our approach uses ideas from nonsmooth analysis to formulate
conditions for min- and max- affine control barrier functions. We show how a
feedback switching law can be extracted from a given CBF. Next, we show how to
automate the process of synthesizing CBFs given a system description through a
tree-search algorithm inspired by branch-and-cut methods from combinatorial
optimization. Finally, we demonstrate our approach on a series of interesting
examples of switched affine systems.

### 3. [Predictive control of wastewater treatment plants as energy-autonomous water resource recovery facilities](http://arxiv.org/pdf/2506.10490v1)

Authors: Otacilio B. L. Neto, Michela Mulas, Iiro Harjunkoski, Francesco Corona

This work proposes an automatic control solution for the operation of
conventional wastewater treatment plants (WWTPs) as energy-autonomous water
resource recovery facilities. We first conceptualize a classification of the
quality of treated water for three resource recovery applications
(environmental, industrial, and agricultural water reuse). We then present an
output-feedback model predictive controller (Output MPC) that operates a plant
to produce water of specific quality class, while also producing sufficient
biogas to ensure nonpositive energy costs. The controller is demonstrated in
the long-term operation of a full-scale WWTP subjected to typical influent
loads and periodically changing quality targets. Our results provide a
proof-of-concept on the energy-autonomous operation of existing wastewater
treatment infrastructure with control strategies that are general enough to
accommodate a wide range of resource recovery objectives.

### 4. [Analyzing the performance of a V2X-enhanced braking system in real-world crash situations](http://arxiv.org/pdf/2506.10535v1)

Authors: Jan Zimmermann, Jörg Mönnich, Michael Scherl, Ignacio Llatser, Florian Wildschütte, Frank Hofmann

By using an automated braking system, such as the Automatic Emergency Brake
(AEB), crashes can be avoided in situations where the driver is unaware of an
imminent collision. However, conventional AEB systems detect potential
collision adversaries with onboard sensor systems, such as radars and cameras,
that may fail in non-line-of-sight situations. By leveraging
vehicle-to-everything (V2X) communication, information regarding an approaching
vehicle can be received by the ego vehicle at an early point in time, even if
the opponent vehicle is occluded by a view obstruction. In this work, we
consider a 2-stage braking cascade, consisting of a partial brake, triggered
based on V2X information, and a sensor-triggered AEB. We evaluate its crash
avoidance performance in real-world crash situations extracted from the German
In-Depth Accident Study (GIDAS) database using an accident simulation
framework. The results are compared against a sensor-triggered AEB system and a
purely V2X-triggered partial brake. To further analyze the results, we identify
the crash cause for each situation in which the brake function under test could
not prevent the crash. The simulation results show a high added benefit of the
V2X-enhanced braking systems compared to the exclusive use of visual-based
sensor systems for automated collision prevention.

### 5. [Joint System Modeling Approach for Fault Simulation of Start-er/Generator and Gas Generator in All-Electric APU](http://arxiv.org/pdf/2506.10562v1)

Authors: Haotian Mao, Yingqing Guo

This paper presents a joint system modeling approach for fault simulation of
all-electric auxiliary power unit (APU), integrating starter/generator
turn-to-turn short circuit (TTSC) faults with gas generator gas-path faults.To
address challenges in electromechanical coupling, simulation precision and
computational efficiency balance, we propose a multi-rate continuous-discrete
hybrid simulation architecture. This architecture treats the starter/generator
as a continuous system with variable step size in Simulink, while modeling the
gas generator as a discrete system with fixed step size in a dynamic-link
library (DLL) environment. For the starter/generator fault modeling, a
multi-loop approach is deployed to accurately simulate TTSC faults. For the gas
generator, we develop an improved GasTurb-DLL modeling method (IGDM) that
enhances uncertainty modeling, state-space representation, and tool chain
compatibility. Finally, the proposed methodology above was implemented in a
case study based on the APS5000 all-electric APU structure and parameters.
Model validation was conducted by comparing simulation results--covering
steady-state, transients, healthy, and fault conditions--with reference data
from third-party software and literature. The close agreement confirms both the
model's accuracy and the effectiveness of our modeling methodology. This work
establishes a modeling foundation for investigating the opportunities and
challenges in fault detection and isolation (FDI) brought by the all
electrification of the APU, including joint fault estimation and diagnosis,
coupled electromechanical fault characteristics.

### 6. [Sampling-Based Planning Under STL Specifications: A Forward Invariance Approach](http://arxiv.org/pdf/2506.10739v1)

Authors: Gregorio Marchesini, Siyuan Liu, Lars Lindemann, Dimos V. Dimarogonas

We propose a variant of the Rapidly Exploring Random Tree Star
(RRT$^{\star}$) algorithm to synthesize trajectories satisfying a given
spatio-temporal specification expressed in a fragment of Signal Temporal Logic
(STL) for linear systems. Previous approaches for planning trajectories under
STL specifications using sampling-based methods leverage either mixed-integer
or non-smooth optimization techniques, with poor scalability in the horizon and
complexity of the task. We adopt instead a control-theoretic perspective on the
problem, based on the notion of set forward invariance. Specifically, from a
given STL task defined over polyhedral predicates, we develop a novel
algorithmic framework by which the task is efficiently encoded into a
time-varying set via linear programming, such that trajectories evolving within
the set also satisfy the task. Forward invariance properties of the resulting
set with respect to the system dynamics and input limitations are then proved
via non-smooth analysis. We then present a modified RRT$^{\star}$ algorithm to
synthesize asymptotically optimal and dynamically feasible trajectories
satisfying a given STL specification, by sampling a tree of trajectories within
the previously constructed time-varying set. We showcase two use cases of our
approach involving an autonomous inspection of the International Space Station
and room-servicing task requiring timed revisit of a charging station.

### 7. [Joint Beamforming with Extremely Large Scale RIS: A Sequential Multi-Agent A2C Approach](http://arxiv.org/pdf/2506.10815v1)

Authors: Zhi Chai, Jiajie Xu, Justin P Coon, Mohamed-Slim Alouini

It is a challenging problem to jointly optimize the base station (BS)
precoding matrix and the reconfigurable intelligent surface (RIS) phases
simultaneously in a RIS-assisted multiple-user multiple-input-multiple-output
(MU-MIMO) scenario when the size of the RIS becomes extremely large. In this
paper, we propose a deep reinforcement learning algorithm called sequential
multi-agent advantage actor-critic (A2C) to solve this problem. In addition,
the discrete phase of RISs, imperfect channel state information (CSI), and
channel correlations between users are taken into consideration. The
computational complexity is also analyzed, and the performance of the proposed
algorithm is compared with the zero-forcing (ZF) beamformer in terms of the sum
spectral efficiency (SE). It is noted that the computational complexity of the
proposed algorithm is lower than the benchmark, while the performance is better
than the benchmark. Throughout simulations, it is also found that the proposed
algorithm is robust to medium channel estimation error.

### 8. [A Robust Optimization Framework for Flexible Industrial Energy Scheduling: Application to a Cement Plant with Market Participation](http://arxiv.org/pdf/2506.10824v1)

Authors: Sebastián Rojas-Innocenti, Enrique Baeyens, Alejandro Martín-Crespo, Sergio Saludes-Rodil, Fernando Frechoso Escudero

This paper presents a scenario based robust optimization framework for short
term energy scheduling in electricity intensive industrial plants, explicitly
addressing uncertainty in planning decisions. The model is formulated as a
two-stage Mixed Integer Linear Program (MILP) and integrates a hybrid scenario
generation method capable of representing uncertain inputs such as electricity
prices, renewable generation, and internal demand. A convex objective function
combining expected and worst case operational costs allows for tunable risk
aversion, enabling planners to balance economic performance and robustness. The
resulting schedule ensures feasibility across all scenarios and supports
coordinated use of industrial flexibility assets, including battery energy
storage and shiftable production. To isolate the effects of market volatility,
the framework is applied to a real world cement manufacturing case study
considering only day-ahead electricity price uncertainty, with all other inputs
treated deterministically. Results show improved resilience to forecast
deviations, reduced cost variability, and more consistent operations. The
proposed method offers a scalable and risk-aware approach for industrial
flexibility planning under uncertainty.

### 9. [General Reference Frame Identification and Transformation in Unbalanced Power Systems](http://arxiv.org/pdf/2506.10835v1)

Authors: Francisco G. Montoya, Santiago Sánchez Acevedo

Various domains such as power system stability analysis, electric machine
modeling, and control of power electronic converters have significantly
benefited from the application of coordinate transformations. One of the main
benefits is the dimensional reduction, which reduces the complexity of the
problems. This paper introduces a novel general transformation based on a
geometric framework that directly identifies the plane containing the locus for
unbalanced quantities through bivector analysis using Geometric Algebra. The
proposed method provides a direct transformation valid for any degree of
unbalance in $n$-phase, $(n+1)$-wire sinusoidal systems. The transformation
requires only two measurements (voltage or current) taken at different time
instants, making it computationally efficient. Moreover, we demonstrate through
pure geometric reasoning that our approach is general and encompasses other
techniques, such as the classical Clarke transformation. Numerical simulations
and experimental validation using a real-time digital simulator and a physical
laboratory setup demonstrate the effectiveness of the proposed method. This
generalization to multi-dimensional systems, combined with the reduced
measurement requirements, represents a significant advancement over existing
approaches that are typically restricted to three-phase applications or suffer
from computational limitations.

### 10. [Data-Driven Model Reduction by Moment Matching for Linear and Nonlinear Parametric Systems](http://arxiv.org/pdf/2506.10866v1)

Authors: Hanqing Zhang, Junyu Mao, Mohammad Fahim Shakib, Giordano Scarciotti

Theory and methods to obtain parametric reduced-order models by moment
matching are presented. The definition of the parametric moment is introduced,
and methods (model-based and data-driven) for the approximation of the
parametric moment of linear and nonlinear parametric systems are proposed.
These approximations are exploited to construct families of parametric
reduced-order models that match the approximate parametric moment of the system
to be reduced and preserve key system properties such as asymptotic stability
and dissipativity. The use of the model reduction methods is illustrated by
means of a parametric benchmark model for the linear case and a large-scale
wind farm model for the nonlinear case. In the illustration, a comparison of
the proposed approximation methods is drawn and their advantages/disadvantages
are discussed.

### Machine Learning (Statistics Category)

### 1. [Meta-learning Representations for Learning from Multiple Annotators](http://arxiv.org/pdf/2506.10259v1)

Authors: Atsutoshi Kumagai, Tomoharu Iwata, Taishi Nishiyama, Yasutoshi Ida, Yasuhiro Fujiwara

We propose a meta-learning method for learning from multiple noisy
annotators. In many applications such as crowdsourcing services, labels for
supervised learning are given by multiple annotators. Since the annotators have
different skills or biases, given labels can be noisy. To learn accurate
classifiers, existing methods require many noisy annotated data. However,
sufficient data might be unavailable in practice. To overcome the lack of data,
the proposed method uses labeled data obtained in different but related tasks.
The proposed method embeds each example in tasks to a latent space by using a
neural network and constructs a probabilistic model for learning a
task-specific classifier while estimating annotators' abilities on the latent
space. This neural network is meta-learned to improve the expected test
classification performance when the classifier is adapted to a given small
amount of annotated data. This classifier adaptation is performed by maximizing
the posterior probability via the expectation-maximization (EM) algorithm.
Since each step in the EM algorithm is easily computed as a closed-form and is
differentiable, the proposed method can efficiently backpropagate the loss
through the EM algorithm to meta-learn the neural network. We show the
effectiveness of our method with real-world datasets with synthetic noise and
real-world crowdsourcing datasets.

### 2. [Collaborative Min-Max Regret in Grouped Multi-Armed Bandits](http://arxiv.org/pdf/2506.10313v1)

Authors: Moïse Blanchard, Vineet Goyal

We study the impact of sharing exploration in multi-armed bandits in a
grouped setting where a set of groups have overlapping feasible action sets
[Baek and Farias '24]. In this grouped bandit setting, groups share reward
observations, and the objective is to minimize the collaborative regret,
defined as the maximum regret across groups. This naturally captures
applications in which one aims to balance the exploration burden between groups
or populations -- it is known that standard algorithms can lead to
significantly imbalanced exploration cost between groups. We address this
problem by introducing an algorithm Col-UCB that dynamically coordinates
exploration across groups. We show that Col-UCB achieves both optimal minimax
and instance-dependent collaborative regret up to logarithmic factors. These
bounds are adaptive to the structure of shared action sets between groups,
providing insights into when collaboration yields significant benefits over
each group learning their best action independently.

### 3. [Air in Your Neighborhood: Fine-Grained AQI Forecasting Using Mobile Sensor Data](http://arxiv.org/pdf/2506.10332v1)

Authors: Aaryam Sharma

Air pollution has become a significant health risk in developing countries.
While governments routinely publish air-quality index (AQI) data to track
pollution, these values fail to capture the local reality, as sensors are often
very sparse. In this paper, we address this gap by predicting AQI in 1 km^2
neighborhoods, using the example of AirDelhi dataset. Using Spatio-temporal
GNNs we surpass existing works by 71.654 MSE a 79% reduction, even on unseen
coordinates. New insights about AQI such as the existence of strong repetitive
short-term patterns and changing spatial relations are also discovered. The
code is available on GitHub.

### 4. [Measuring Semantic Information Production in Generative Diffusion Models](http://arxiv.org/pdf/2506.10433v1)

Authors: Florian Handke, Félix Koulischer, Gabriel Raya, Luca Ambrogioni

It is well known that semantic and structural features of the generated
images emerge at different times during the reverse dynamics of diffusion, a
phenomenon that has been connected to physical phase transitions in magnets and
other materials. In this paper, we introduce a general information-theoretic
approach to measure when these class-semantic "decisions" are made during the
generative process. By using an online formula for the optimal Bayesian
classifier, we estimate the conditional entropy of the class label given the
noisy state. We then determine the time intervals corresponding to the highest
information transfer between noisy states and class labels using the time
derivative of the conditional entropy. We demonstrate our method on
one-dimensional Gaussian mixture models and on DDPM models trained on the
CIFAR10 dataset. As expected, we find that the semantic information transfer is
highest in the intermediate stages of diffusion while vanishing during the
final stages. However, we found sizable differences between the entropy rate
profiles of different classes, suggesting that different "semantic decisions"
are located at different intermediate times.

### 5. [Box-Constrained Softmax Function and Its Application for Post-Hoc Calibration](http://arxiv.org/pdf/2506.10572v1)

Authors: Kyohei Atarashi, Satoshi Oyama, Hiromi Arai, Hisashi Kashima

Controlling the output probabilities of softmax-based models is a common
problem in modern machine learning. Although the $\mathrm{Softmax}$ function
provides soft control via its temperature parameter, it lacks the ability to
enforce hard constraints, such as box constraints, on output probabilities,
which can be critical in certain applications requiring reliable and
trustworthy models. In this work, we propose the box-constrained softmax
($\mathrm{BCSoftmax}$) function, a novel generalization of the
$\mathrm{Softmax}$ function that explicitly enforces lower and upper bounds on
output probabilities. While $\mathrm{BCSoftmax}$ is formulated as the solution
to a box-constrained optimization problem, we develop an exact and efficient
computation algorithm for $\mathrm{BCSoftmax}$. As a key application, we
introduce two post-hoc calibration methods based on $\mathrm{BCSoftmax}$. The
proposed methods mitigate underconfidence and overconfidence in predictive
models by learning the lower and upper bounds of the output probabilities or
logits after model training, thereby enhancing reliability in downstream
decision-making tasks. We demonstrate the effectiveness of our methods
experimentally using the TinyImageNet, CIFAR-100, and 20NewsGroups datasets,
achieving improvements in calibration metrics.

### 6. [Logarithmic Smoothing for Adaptive PAC-Bayesian Off-Policy Learning](http://arxiv.org/pdf/2506.10664v1)

Authors: Maxime Haddouche, Otmane Sakhi

Off-policy learning serves as the primary framework for learning optimal
policies from logged interactions collected under a static behavior policy. In
this work, we investigate the more practical and flexible setting of adaptive
off-policy learning, where policies are iteratively refined and re-deployed to
collect higher-quality data. Building on the success of PAC-Bayesian learning
with Logarithmic Smoothing (LS) in static settings, we extend this framework to
the adaptive scenario using tools from online PAC-Bayesian theory. Furthermore,
we demonstrate that a principled adjustment to the LS estimator naturally
accommodates multiple rounds of deployment and yields faster convergence rates
under mild conditions. Our method matches the performance of leading offline
approaches in static settings, and significantly outperforms them when
intermediate policy deployments are allowed. Empirical evaluations across
diverse scenarios highlight both the advantages of adaptive data collection and
the strength of the PAC-Bayesian formulation.

### 7. [Practical Improvements of A/B Testing with Off-Policy Estimation](http://arxiv.org/pdf/2506.10677v1)

Authors: Sakhi Otmane, Gilotte Alexandre, Rohde David

We address the problem of A/B testing, a widely used protocol for evaluating
the potential improvement achieved by a new decision system compared to a
baseline. This protocol segments the population into two subgroups, each
exposed to a version of the system and estimates the improvement as the
difference between the measured effects. In this work, we demonstrate that the
commonly used difference-in-means estimator, while unbiased, can be improved.
We introduce a family of unbiased off-policy estimators that achieves lower
variance than the standard approach. Among this family, we identify the
estimator with the lowest variance. The resulting estimator is simple, and
offers substantial variance reduction when the two tested systems exhibit
similarities. Our theoretical analysis and experimental results validate the
effectiveness and practicality of the proposed method.

### 8. [Probably Approximately Correct Labels](http://arxiv.org/pdf/2506.10908v1)

Authors: Emmanuel J. Candès, Andrew Ilyas, Tijana Zrnic

Obtaining high-quality labeled datasets is often costly, requiring either
extensive human annotation or expensive experiments. We propose a method that
supplements such "expert" labels with AI predictions from pre-trained models to
construct labeled datasets more cost-effectively. Our approach results in
probably approximately correct labels: with high probability, the overall
labeling error is small. This solution enables rigorous yet efficient dataset
curation using modern AI models. We demonstrate the benefits of the methodology
through text annotation with large language models, image labeling with
pre-trained vision models, and protein folding analysis with AlphaFold.

### 9. [What Exactly Does Guidance Do in Masked Discrete Diffusion Models](http://arxiv.org/pdf/2506.10971v1)

Authors: He Ye, Rojas Kevin, Tao Molei

We study masked discrete diffusion models with classifier-free guidance
(CFG). Assuming no score error nor discretization error, we derive an explicit
solution to the guided reverse dynamics, so that how guidance influences the
sampling behavior can be precisely characterized. When the full data
distribution is a mixture over classes and the goal is to sample from a
specific class, guidance amplifies class-specific regions while suppresses
regions shared with other classes. This effect depends on the guidance strength
$w$ and induces distinct covariance structures in the sampled distribution.
Notably, we observe quantitatively different behaviors in $1$D and $2$D. We
also show that for large $w$, the decay rate of the total variation
($\mathrm{TV}$) along the reverse dynamics is double-exponential in $w$ for
both $1$D and $2$D. These findings highlight the role of guidance, not just in
shaping the output distribution, but also in controlling the dynamics of the
sampling trajectory. Our theoretical analysis is supported by experiments that
illustrate the geometric effects of guidance and its impact on convergence.

### 10. [VQC-MLPNet: An Unconventional Hybrid Quantum-Classical Architecture for Scalable and Robust Quantum Machine Learning](http://arxiv.org/pdf/2506.10275v1)

Authors: Jun Qi, Chao-Han Yang, Pin-Yu Chen, Min-Hsiu Hsieh

Variational Quantum Circuits (VQCs) offer a novel pathway for quantum machine
learning, yet their practical application is hindered by inherent limitations
such as constrained linear expressivity, optimization challenges, and acute
sensitivity to quantum hardware noise. This work introduces VQC-MLPNet, a
scalable and robust hybrid quantum-classical architecture designed to overcome
these obstacles. By innovatively employing quantum circuits to dynamically
generate parameters for classical Multi-Layer Perceptrons (MLPs) via amplitude
encoding and parameterized quantum operations, VQC-MLPNet substantially expands
representation capabilities and augments training stability. We provide
rigorous theoretical guarantees via statistical learning techniques and Neural
Tangent Kernel analysis, explicitly deriving upper bounds on approximation,
uniform deviation, and optimization errors. These theoretical insights
demonstrate exponential improvements in representation capacity relative to
quantum circuit depth and the number of qubits, providing clear computational
advantages over standalone quantum circuits and existing hybrid quantum
architectures. Our theoretical claims are empirically corroborated through
extensive experiments, including classifying semiconductor quantum-dot charge
states and predicting genomic transcription factor binding sites, demonstrating
resilient performance even under realistic IBM quantum noise simulations. This
research establishes a theoretically sound and practically robust framework,
advancing the frontiers of quantum-enhanced learning for unconventional
computing paradigms in the Noisy Intermediate-Scale Quantum era and beyond.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-06-13 PST.

### 1. [Dataset resulting from the user study on comprehensibility of explainable AI algorithms](https://www.nature.com/articles/s41597-025-05167-6)

Authors: Szymon Bobek et al.

### 2. [Flying in air ducts](https://www.nature.com/articles/s44182-025-00032-5)

Authors: Thomas Martin et al.

