# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-08 16:59:50.834488 PST.

### Artificial Intelligence

### 1. Polynomial-Time Relational Probabilistic Inference in Open Universes

[Polynomial-Time Relational Probabilistic Inference in Open Universes](http://arxiv.org/pdf/2505.04115v1)

Authors: Luise Ge, Brendan Juba, Kris Nilsson

Reasoning under uncertainty is a fundamental challenge in Artificial
Intelligence. As with most of these challenges, there is a harsh dilemma
between the expressive power of the language used, and the tractability of the
computational problem posed by reasoning. Inspired by human reasoning, we
introduce a method of first-order relational probabilistic inference that
satisfies both criteria, and can handle hybrid (discrete and continuous)
variables. Specifically, we extend sum-of-squares logic of expectation to
relational settings, demonstrating that lifted reasoning in the bounded-degree
fragment for knowledge bases of bounded quantifier rank can be performed in
polynomial time, even with an a priori unknown and/or countably infinite set of
objects. Crucially, our notion of tractability is framed in proof-theoretic
terms, which extends beyond the syntactic properties of the language or
queries. We are able to derive the tightest bounds provable by proofs of a
given degree and size and establish completeness in our sum-of-squares
refutations for fixed degrees.

### 2. Mastering Multi-Drone Volleyball through Hierarchical Co-Self-Play Reinforcement Learning

[Mastering Multi-Drone Volleyball through Hierarchical Co-Self-Play Reinforcement Learning](http://arxiv.org/pdf/2505.04317v1)

Authors: Ruize Zhang, Sirui Xiang, Zelai Xu, Feng Gao, Shilong Ji, Wenhao Tang, Wenbo Ding, Chao Yu, Yu Wang

In this paper, we tackle the problem of learning to play 3v3 multi-drone
volleyball, a new embodied competitive task that requires both high-level
strategic coordination and low-level agile control. The task is turn-based,
multi-agent, and physically grounded, posing significant challenges due to its
long-horizon dependencies, tight inter-agent coupling, and the underactuated
dynamics of quadrotors. To address this, we propose Hierarchical Co-Self-Play
(HCSP), a hierarchical reinforcement learning framework that separates
centralized high-level strategic decision-making from decentralized low-level
motion control. We design a three-stage population-based training pipeline to
enable both strategy and skill to emerge from scratch without expert
demonstrations: (I) training diverse low-level skills, (II) learning high-level
strategy via self-play with fixed low-level controllers, and (III) joint
fine-tuning through co-self-play. Experiments show that HCSP achieves superior
performance, outperforming non-hierarchical self-play and rule-based
hierarchical baselines with an average 82.9\% win rate and a 71.5\% win rate
against the two-stage variant. Moreover, co-self-play leads to emergent team
behaviors such as role switching and coordinated formations, demonstrating the
effectiveness of our hierarchical design and training scheme.

### 3. On some improvements to Unbounded Minimax

[On some improvements to Unbounded Minimax](http://arxiv.org/pdf/2505.04525v1)

Authors: Quentin Cohen-Solal, Tristan Cazenave

This paper presents the first experimental evaluation of four previously
untested modifications of Unbounded Best-First Minimax algorithm. This
algorithm explores the game tree by iteratively expanding the most promising
sequences of actions based on the current partial game tree. We first evaluate
the use of transposition tables, which convert the game tree into a directed
acyclic graph by merging duplicate states. Second, we compare the original
algorithm by Korf & Chickering with the variant proposed by Cohen-Solal, which
differs in its backpropagation strategy: instead of stopping when a stable
value is encountered, it updates values up to the root. This change slightly
improves performance when value ties or transposition tables are involved.
Third, we assess replacing the exact terminal evaluation function with the
learned heuristic function. While beneficial when exact evaluations are costly,
this modification reduces performance in inexpensive settings. Finally, we
examine the impact of the completion technique that prioritizes resolved
winning states and avoids resolved losing states. This technique also improves
performance. Overall, our findings highlight how targeted modifications can
enhance the efficiency of Unbounded Best-First Minimax.

### 4. Qualitative Analysis of $ω$-Regular Objectives on Robust MDPs

[Qualitative Analysis of $ω$-Regular Objectives on Robust MDPs](http://arxiv.org/pdf/2505.04539v1)

Authors: Ali Asadi, Krishnendu Chatterjee, Ehsan Kafshdar Goharshady, Mehrdad Karrabi, Ali Shafiee

Robust Markov Decision Processes (RMDPs) generalize classical MDPs that
consider uncertainties in transition probabilities by defining a set of
possible transition functions. An objective is a set of runs (or infinite
trajectories) of the RMDP, and the value for an objective is the maximal
probability that the agent can guarantee against the adversarial environment.
We consider (a) reachability objectives, where given a target set of states,
the goal is to eventually arrive at one of them; and (b) parity objectives,
which are a canonical representation for $\omega$-regular objectives. The
qualitative analysis problem asks whether the objective can be ensured with
probability 1.
  In this work, we study the qualitative problem for reachability and parity
objectives on RMDPs without making any assumption over the structures of the
RMDPs, e.g., unichain or aperiodic. Our contributions are twofold. We first
present efficient algorithms with oracle access to uncertainty sets that solve
qualitative problems of reachability and parity objectives. We then report
experimental results demonstrating the effectiveness of our oracle-based
approach on classical RMDP examples from the literature scaling up to thousands
of states.

### 5. Advancing and Benchmarking Personalized Tool Invocation for LLMs

[Advancing and Benchmarking Personalized Tool Invocation for LLMs](http://arxiv.org/pdf/2505.04072v1)

Authors: Xu Huang, Yuefeng Huang, Weiwen Liu, Xingshan Zeng, Yasheng Wang, Ruiming Tang, Hong Xie, Defu Lian

Tool invocation is a crucial mechanism for extending the capabilities of
Large Language Models (LLMs) and has recently garnered significant attention.
It enables LLMs to solve complex problems through tool calls while accessing
up-to-date world knowledge. However, existing work primarily focuses on the
fundamental ability of LLMs to invoke tools for problem-solving, without
considering personalized constraints in tool invocation. In this work, we
introduce the concept of Personalized Tool Invocation and define two key tasks:
Tool Preference and Profile-dependent Query. Tool Preference addresses user
preferences when selecting among functionally similar tools, while
Profile-dependent Query considers cases where a user query lacks certain tool
parameters, requiring the model to infer them from the user profile. To tackle
these challenges, we propose PTool, a data synthesis framework designed for
personalized tool invocation. Additionally, we construct \textbf{PTBench}, the
first benchmark for evaluating personalized tool invocation. We then fine-tune
various open-source models, demonstrating the effectiveness of our framework
and providing valuable insights. Our benchmark is public at
https://github.com/hyfshadow/PTBench.

### 6. LLM-e Guess: Can LLMs Capabilities Advance Without Hardware Progress?

[LLM-e Guess: Can LLMs Capabilities Advance Without Hardware Progress?](http://arxiv.org/pdf/2505.04075v1)

Authors: Teddy Foley, Spencer Guo, Henry Josephson, Anqi Qu, Jack Sanderson

This paper examines whether large language model (LLM) capabilities can
continue to advance without additional compute by analyzing the development and
role of algorithms used in state-of-the-art LLMs. Motivated by regulatory
efforts that have largely focused on restricting access to high-performance
hardware, we ask: Can LLMs progress in a compute-constrained environment, and
how do algorithmic innovations perform under such conditions?
  To address these questions, we introduce a novel classification framework
that distinguishes between compute-dependent innovations -- which yield
disproportionate benefits at high compute levels (e.g., the Transformer
architecture and mixture-of-experts models) and compute-independent
innovations, which improve efficiency across all compute scales (e.g., rotary
positional encoding, FlashAttention, or layer normalization). We quantify these
contributions using a metric called compute-equivalent gain (CEG), which
estimates the additional compute that would be required to achieve similar
improvements without these algorithmic advancements.
  To validate this framework, we conduct small-scale training experiments with
a scaled-down GPT-2 model. Our results confirm that compute-independent
advancements yield meaningful performance gains even in resource-constrained
settings, with a CEG of up to $3.5\times$ over a baseline model. By contrast,
compute-dependent advancements provided little benefit or even degraded
performance at the small scale, reinforcing the importance of compute
availability for certain algorithmic gains.

### 7. An Empirical Study of OpenAI API Discussions on Stack Overflow

[An Empirical Study of OpenAI API Discussions on Stack Overflow](http://arxiv.org/pdf/2505.04084v1)

Authors: Xiang Chen, Jibin Wang, Chaoyang Gao, Xiaolin Ju, Zhanqi Cui

The rapid advancement of large language models (LLMs), represented by
OpenAI's GPT series, has significantly impacted various domains such as natural
language processing, software development, education, healthcare, finance, and
scientific research. However, OpenAI APIs introduce unique challenges that
differ from traditional APIs, such as the complexities of prompt engineering,
token-based cost management, non-deterministic outputs, and operation as black
boxes. To the best of our knowledge, the challenges developers encounter when
using OpenAI APIs have not been explored in previous empirical studies. To fill
this gap, we conduct the first comprehensive empirical study by analyzing 2,874
OpenAI API-related discussions from the popular Q&A forum Stack Overflow. We
first examine the popularity and difficulty of these posts. After manually
categorizing them into nine OpenAI API-related categories, we identify specific
challenges associated with each category through topic modeling analysis. Based
on our empirical findings, we finally propose actionable implications for
developers, LLM vendors, and researchers.

### 8. Bringing legal knowledge to the public by constructing a legal question bank using large-scale pre-trained language model

[Bringing legal knowledge to the public by constructing a legal question bank using large-scale pre-trained language model](http://arxiv.org/pdf/2505.04132v1)

Authors: Mingruo Yuan, Ben Kao, Tien-Hsuan Wu, Michael M. K. Cheung, Henry W. H. Chan, Anne S. Y. Cheung, Felix W. H. Chan, Yongxi Chen

Access to legal information is fundamental to access to justice. Yet
accessibility refers not only to making legal documents available to the
public, but also rendering legal information comprehensible to them. A vexing
problem in bringing legal information to the public is how to turn formal legal
documents such as legislation and judgments, which are often highly technical,
to easily navigable and comprehensible knowledge to those without legal
education. In this study, we formulate a three-step approach for bringing legal
knowledge to laypersons, tackling the issues of navigability and
comprehensibility. First, we translate selected sections of the law into
snippets (called CLIC-pages), each being a small piece of article that focuses
on explaining certain technical legal concept in layperson's terms. Second, we
construct a Legal Question Bank (LQB), which is a collection of legal questions
whose answers can be found in the CLIC-pages. Third, we design an interactive
CLIC Recommender (CRec). Given a user's verbal description of a legal situation
that requires a legal solution, CRec interprets the user's input and shortlists
questions from the question bank that are most likely relevant to the given
legal situation and recommends their corresponding CLIC pages where relevant
legal knowledge can be found. In this paper we focus on the technical aspects
of creating an LQB. We show how large-scale pre-trained language models, such
as GPT-3, can be used to generate legal questions. We compare machine-generated
questions (MGQs) against human-composed questions (HCQs) and find that MGQs are
more scalable, cost-effective, and more diversified, while HCQs are more
precise. We also show a prototype of CRec and illustrate through an example how
our 3-step approach effectively brings relevant legal knowledge to the public.

### 9. R^3-VQA: "Read the Room" by Video Social Reasoning

[R^3-VQA: "Read the Room" by Video Social Reasoning](http://arxiv.org/pdf/2505.04147v1)

Authors: Lixing Niu, Jiapeng Li, Xingping Yu, Shu Wang, Ruining Feng, Bo Wu, Ping Wei, Yisen Wang, Lifeng Fan

"Read the room" is a significant social reasoning capability in human daily
life. Humans can infer others' mental states from subtle social cues. Previous
social reasoning tasks and datasets lack complexity (e.g., simple scenes, basic
interactions, incomplete mental state variables, single-step reasoning, etc.)
and fall far short of the challenges present in real-life social interactions.
In this paper, we contribute a valuable, high-quality, and comprehensive video
dataset named R^3-VQA with precise and fine-grained annotations of social
events and mental states (i.e., belief, intent, desire, and emotion) as well as
corresponding social causal chains in complex social scenarios. Moreover, we
include human-annotated and model-generated QAs. Our task R^3-VQA includes
three aspects: Social Event Understanding, Mental State Estimation, and Social
Causal Reasoning. As a benchmark, we comprehensively evaluate the social
reasoning capabilities and consistencies of current state-of-the-art large
vision-language models (LVLMs). Comprehensive experiments show that (i) LVLMs
are still far from human-level consistent social reasoning in complex social
scenarios; (ii) Theory of Mind (ToM) prompting can help LVLMs perform better on
social reasoning tasks. We provide some of our dataset and codes in
supplementary material and will release our full dataset and codes upon
acceptance.

### 10. TS-SNN: Temporal Shift Module for Spiking Neural Networks

[TS-SNN: Temporal Shift Module for Spiking Neural Networks](http://arxiv.org/pdf/2505.04165v1)

Authors: Kairong Yu, Tianqing Zhang, Qi Xu, Gang Pan, Hongwei Wang

Spiking Neural Networks (SNNs) are increasingly recognized for their
biological plausibility and energy efficiency, positioning them as strong
alternatives to Artificial Neural Networks (ANNs) in neuromorphic computing
applications. SNNs inherently process temporal information by leveraging the
precise timing of spikes, but balancing temporal feature utilization with low
energy consumption remains a challenge. In this work, we introduce Temporal
Shift module for Spiking Neural Networks (TS-SNN), which incorporates a novel
Temporal Shift (TS) module to integrate past, present, and future spike
features within a single timestep via a simple yet effective shift operation. A
residual combination method prevents information loss by integrating shifted
and original features. The TS module is lightweight, requiring only one
additional learnable parameter, and can be seamlessly integrated into existing
architectures with minimal additional computational cost. TS-SNN achieves
state-of-the-art performance on benchmarks like CIFAR-10 (96.72\%), CIFAR-100
(80.28\%), and ImageNet (70.61\%) with fewer timesteps, while maintaining low
energy consumption. This work marks a significant step forward in developing
efficient and accurate SNN architectures.

### Hardware Architecture

### 1. In-Situ Hardware Error Detection Using Specification-Derived Petri Net Models and Behavior-Derived State Sequences

[In-Situ Hardware Error Detection Using Specification-Derived Petri Net Models and Behavior-Derived State Sequences](http://arxiv.org/pdf/2505.04108v1)

Authors: Tomonari Tanaka, Takumi Uezono, Kohei Suenaga, Masanori Hashimoto

In hardware accelerators used in data centers and safety-critical
applications, soft errors and resultant silent data corruption significantly
compromise reliability, particularly when upsets occur in control-flow
operations, leading to severe failures. To address this, we introduce two
methods for monitoring control flows: using specification-derived Petri nets
and using behavior-derived state transitions. We validated our method across
four designs: convolutional layer operation, Gaussian blur, AES encryption, and
a router in Network-on-Chip. Our fault injection campaign targeting the control
registers and primary control inputs demonstrated high error detection rates in
both datapath and control logic. Synthesis results show that a maximum
detection rate is achieved with a few to around 10% area overhead in most
cases. The proposed detectors quickly detect 48% to 100% of failures resulting
from upsets in internal control registers and perturbations in primary control
inputs. The two proposed methods were compared in terms of area overhead and
error detection rate. By selectively applying these two methods, a wide range
of area constraints can be accommodated, enabling practical implementation and
effectively enhancing error detection capabilities.

### 2. Accelerating Triangle Counting with Real Processing-in-Memory Systems

[Accelerating Triangle Counting with Real Processing-in-Memory Systems](http://arxiv.org/pdf/2505.04269v1)

Authors: Lorenzo Asquini, Manos Frouzakis, Juan Gómez-Luna, Mohammad Sadrosadati, Onur Mutlu, Francesco Silvestri

Triangle Counting (TC) is a procedure that involves enumerating the number of
triangles within a graph. It has important applications in numerous fields,
such as social or biological network analysis and network security. TC is a
memory-bound workload that does not scale efficiently in conventional
processor-centric systems due to several memory accesses across large memory
regions and low data reuse. However, recent Processing-in-Memory (PIM)
architectures present a promising solution to alleviate these bottlenecks. Our
work presents the first TC algorithm that leverages the capabilities of the
UPMEM system, the first commercially available PIM architecture, while at the
same time addressing its limitations. We use a vertex coloring technique to
avoid expensive communication between PIM cores and employ reservoir sampling
to address the limited amount of memory available in the PIM cores' DRAM banks.
In addition, our work makes use of the Misra-Gries summary to speed up counting
triangles on graphs with high-degree nodes and uniform sampling of the graph
edges for quicker approximate results. Our PIM implementation surpasses
state-of-the-art CPU-based TC implementations when processing dynamic graphs in
Coordinate List format, showcasing the effectiveness of the UPMEM architecture
in addressing TC's memory-bound challenges.

### 3. Flexing RISC-V Instruction Subset Processors (RISPs) to Extreme Edge

[Flexing RISC-V Instruction Subset Processors (RISPs) to Extreme Edge](http://arxiv.org/pdf/2505.04567v1)

Authors: Alireza Raisiardali, Konstantinos Iordanou, Jedrzej Kufel, Kowshik Gudimetla, Kris Myny, Emre Ozer

This paper presents a methodology for automatically generating processors
that support a subset of the RISC-V instruction set for a new class of
applications at Extreme Edge. The electronics used in extreme edge applications
must be power-efficient, but also provide additional qualities, such as low
cost, conformability, comfort and sustainability. Flexible electronics, rather
than silicon-based electronics, will be capable of meeting these qualities. For
this purpose, we propose a methodology to generate RISPs (RISC-V instruction
subset processors) customised to extreme edge applications and to implement
them as flexible integrated circuits (FlexICs). The methodology is unique in
the sense that verification is an integral part of design. The RISP methodology
treats each instruction in the ISA as a discrete, fully functional,
pre-verified hardware block. It automatically builds a custom processor by
stitching together the hardware blocks of the instructions required by an
application or a set of applications in a specific domain. This approach
significantly reduces the processor verification and its time-to-market. We
generate RISPs using this methodology for three extreme edge applications, and
embedded applications from the Embench benchmark suite, synthesize them as
FlexICs, and compare their power, performance and area to the baselines. Our
results show that RISPs generated using this methodology achieve, on average,
30\% reductions in power and area compared to a RISC-V processor supporting the
full instruction set when synthesized, and are nearly 30 times more energy
efficient with respect to Serv - the world's smallest 32-bit RISC-V processor.
In addition, the full physical implementation of RISPs show up to 21% and 26%
less area and power than Serv.

### 4. Leveraging Simultaneous Usage of Edge GPU Hardware Engines for Video Face Detection and Recognition

[Leveraging Simultaneous Usage of Edge GPU Hardware Engines for Video Face Detection and Recognition](http://arxiv.org/pdf/2505.04502v1)

Authors: Asma Baobaid, Mahmoud Meribout

Video face detection and recognition in public places at the edge is required
in several applications, such as security reinforcement and contactless access
to authorized venues. This paper aims to maximize the simultaneous usage of
hardware engines available in edge GPUs nowadays by leveraging the concurrency
and pipelining of tasks required for face detection and recognition. This also
includes the video decoding task, which is required in most face monitoring
applications as the video streams are usually carried via Gbps Ethernet
network. This constitutes an improvement over previous works where the tasks
are usually allocated to a single engine due to the lack of a unified and
automated framework that simultaneously explores all hardware engines. In
addition, previously, the input faces were usually embedded in still images or
within raw video streams that overlook the burst delay caused by the decoding
stage. The results on real-life video streams suggest that simultaneously using
all the hardware engines available in the recent NVIDIA edge Orin GPU, higher
throughput, and a slight saving of power consumption of around 300 mW,
accounting for around 5%, have been achieved while satisfying the real-time
performance constraint. The performance gets even higher by considering several
video streams simultaneously. Further performance improvement could have been
obtained if the number of shuffle layers that were created by the tensor RT
framework for the face recognition task was lower. Thus, the paper suggests
some hardware improvements to the existing edge GPU processors to enhance their
performance even higher.

### 5. Edge-GPU Based Face Tracking for Face Detection and Recognition Acceleration

[Edge-GPU Based Face Tracking for Face Detection and Recognition Acceleration](http://arxiv.org/pdf/2505.04524v1)

Authors: Asma Baobaid, Mahmoud Meribout

Cost-effective machine vision systems dedicated to real-time and accurate
face detection and recognition in public places are crucial for many modern
applications. However, despite their high performance, which could be reached
using specialized edge or cloud AI hardware accelerators, there is still room
for improvement in throughput and power consumption. This paper aims to suggest
a combined hardware-software approach that optimizes face detection and
recognition systems on one of the latest edge GPUs, namely NVIDIA Jetson AGX
Orin. First, it leverages the simultaneous usage of all its hardware engines to
improve processing time. This offers an improvement over previous works where
these tasks were mainly allocated automatically and exclusively to the CPU or,
to a higher extent, to the GPU core. Additionally, the paper suggests
integrating a face tracker module to avoid redundantly running the face
recognition algorithm for every frame but only when a new face appears in the
scene. The results of extended experiments suggest that simultaneous usage of
all the hardware engines that are available in the Orin GPU and tracker
integration into the pipeline yield an impressive throughput of 290 FPS (frames
per second) on 1920 x 1080 input size frames containing in average of 6
faces/frame. Additionally, a substantial saving of power consumption of around
800 mW was achieved when compared to running the task on the CPU/GPU engines
only and without integrating a tracker into the Orin GPU\'92s pipeline. This
hardware-codesign approach can pave the way to design high-performance machine
vision systems at the edge, critically needed in video monitoring in public
places where several nearby cameras are usually deployed for a same scene.

### Computational Complexity

### 1. Testing Juntas Optimally with Samples

[Testing Juntas Optimally with Samples](http://arxiv.org/pdf/2505.04604v1)

Authors: Lorenzo Beretta, Nathaniel Harms, Caleb Koch

We prove tight upper and lower bounds of
$\Theta\left(\tfrac{1}{\epsilon}\left( \sqrt{2^k \log\binom{n}{k} } +
\log\binom{n}{k} \right)\right)$ on the number of samples required for
distribution-free $k$-junta testing. This is the first tight bound for testing
a natural class of Boolean functions in the distribution-free sample-based
model. Our bounds also hold for the feature selection problem, showing that a
junta tester must learn the set of relevant variables. For tolerant junta
testing, we prove a sample lower bound of $\Omega(2^{(1-o(1)) k} +
\log\binom{n}{k})$ showing that, unlike standard testing, there is no large gap
between tolerant testing and learning.

### Computational Engineering

### 1. Yield and Buckling Stress Limits in Topology Optimization of Multiscale Structures

[Yield and Buckling Stress Limits in Topology Optimization of Multiscale Structures](http://arxiv.org/pdf/2505.04353v1)

Authors: Christoffer Fyllgraf Christensen, Fengwen Wang, Ole Sigmund

This study presents an extension of multiscale topology optimization by
integrating both yield stress and local/global buckling considerations into the
design process. Building upon established multiscale methodologies, we develop
a new framework incorporating yield stress limits either as constraints or
objectives alongside previously established local and global buckling
constraints. This approach significantly refines the optimization process,
ensuring that the resulting designs meet mechanical performance criteria and
adhere to critical material yield constraints. First, we establish local
density-dependent von Mises yield surfaces based on local yield estimates from
homogenization-based analysis to predict the local yield limits of the
homogenized materials. Then, these local Yield-based Load Factors (YLFs) are
combined with local and global buckling criteria to obtain topology optimized
designs that consider yield and buckling failure on all levels. This
integration is crucial for the practical application of optimized structures in
real-world scenarios, where material yield and stability behavior critically
influence structural integrity and durability. Numerical examples demonstrate
how optimized designs depend on the stiffness to yield ratio of the considered
building material. Despite the foundational assumption of separation of scales,
the de-homogenized structures, even at relatively coarse length scales, exhibit
a high degree of agreement with the corresponding homogenized predictions.

### 2. RDPP-TD: Reputation and Data Privacy-Preserving based Truth Discovery Scheme in Mobile Crowdsensing

[RDPP-TD: Reputation and Data Privacy-Preserving based Truth Discovery Scheme in Mobile Crowdsensing](http://arxiv.org/pdf/2505.04361v1)

Authors: Lijian Wu, Weikun Xie, Wei Tan, Tian Wang, Houbing Herbert Song, Anfeng Liu

Truth discovery (TD) plays an important role in Mobile Crowdsensing (MCS).
However, existing TD methods, including privacy-preserving TD approaches,
estimate the truth by weighting only the data submitted in the current round,
which often results in low data quality. Moreover, there is a lack of effective
TD methods that preserve both reputation and data privacy. To address these
issues, a Reputation and Data Privacy-Preserving based Truth Discovery
(RDPP-TD) scheme is proposed to obtain high-quality data for MCS. The RDPP-TD
scheme consists of two key approaches: a Reputation-based Truth Discovery (RTD)
approach, which integrates the weight of current-round data with workers'
reputation values to estimate the truth, thereby achieving more accurate
results, and a Reputation and Data Privacy-Preserving (RDPP) approach, which
ensures privacy preservation for sensing data and reputation values. First, the
RDPP approach, when seamlessly integrated with RTD, can effectively evaluate
the reliability of workers and their sensing data in a privacy-preserving
manner. Second, the RDPP scheme supports reputation-based worker recruitment
and rewards, ensuring high-quality data collection while incentivizing workers
to provide accurate information. Comprehensive theoretical analysis and
extensive experiments based on real-world datasets demonstrate that the
proposed RDPP-TD scheme provides strong privacy protection and improves data
quality by up to 33.3%.

### Computational Geometry

### 1. Report on Nearest Dominating Point Queries

[Report on Nearest Dominating Point Queries](http://arxiv.org/pdf/2505.04617v1)

Authors: Naman Mishra, K S Sreeramji

Given two points $p, q \in \mathbb R^d$, we say that $p$ dominates $q$ and
write $p \succ q$ if each coordinate of $p$ is larger than the corresponding
coordinate of $q$. That is, if $p = (p^{(1)}, p^{(2)}, \ldots, p^{(d)})$ and $q
= (q^{(1)}, q^{(2)}, \ldots, q^{(d)})$, $p \succ q$ if and only if $p^{(i)} >
q^{(i)}$ for all $1 \le i \le d$.
  For example, $p$ and $q$ could represent various ratings for $2$ restaurants,
based on different metrics like taste, affordability, ratings on different
platforms, et cetera. $p \succ q$ then means that the first restaurant
outperformed the second on each metric.
  Given a list of restaurants and their rating, we solve the problem of
determining, for each restaurant, the closest restaurant to it that dominates
it. We improve upon the algorithm under some assumptions towards the end.

### 2. Light Spanners with Small Hop-Diameter

[Light Spanners with Small Hop-Diameter](http://arxiv.org/pdf/2505.04536v1)

Authors: Sujoy Bhore, Lazar Milenkovic

Lightness, sparsity, and hop-diameter are the fundamental parameters of
geometric spanners. Arya et al. [STOC'95] showed in their seminal work that
there exists a construction of Euclidean $(1+\varepsilon)$-spanners with
hop-diameter $O(\log n)$ and lightness $O(\log n)$. They also gave a general
tradeoff of hop-diameter $k$ and sparsity $O(\alpha_k(n))$, where $\alpha_k$ is
a very slowly growing inverse of an Ackermann-style function. The former
combination of logarithmic hop-diameter and lightness is optimal due to the
lower bound by Dinitz et al. [FOCS'08]. Later, Elkin and Solomon [STOC'13]
generalized the light spanner construction to doubling metrics and extended the
tradeoff for more values of hop-diameter $k$. In a recent line of work
[SoCG'22, SoCG'23], Le et al. proved that the aforementioned tradeoff between
the hop-diameter and sparsity is tight for every choice of hop-diameter $k$. A
fundamental question remains: What is the optimal tradeoff between the
hop-diameter and lightness for every value of $k$?
  In this paper, we present a general framework for constructing light spanners
with small hop-diameter. Our framework is based on tree covers. In particular,
we show that if a metric admits a tree cover with $\gamma$ trees, stretch $t$,
and lightness $L$, then it also admits a $t$-spanner with hop-diameter $k$ and
lightness $O(kn^{2/k}\cdot \gamma L)$. Further, we note that the tradeoff for
trees is tight due to a construction in uniform line metric, which is perhaps
the simplest tree metric. As a direct consequence of this framework, we obtain
a tight tradeoff between lightness and hop-diameter for doubling metrics in the
entire regime of $k$.

### Computation and Language

### 1. Natural Language Generation in Healthcare: A Review of Methods and Applications

[Natural Language Generation in Healthcare: A Review of Methods and Applications](http://arxiv.org/pdf/2505.04073v1)

Authors: Mengxian Lyu, Xiaohan Li, Ziyi Chen, Jinqian Pan, Cheng Peng, Sankalp Talankar, Yonghui Wu

Natural language generation (NLG) is the key technology to achieve generative
artificial intelligence (AI). With the breakthroughs in large language models
(LLMs), NLG has been widely used in various medical applications, demonstrating
the potential to enhance clinical workflows, support clinical decision-making,
and improve clinical documentation. Heterogeneous and diverse medical data
modalities, such as medical text, images, and knowledge bases, are utilized in
NLG. Researchers have proposed many generative models and applied them in a
number of healthcare applications. There is a need for a comprehensive review
of NLG methods and applications in the medical domain. In this study, we
systematically reviewed 113 scientific publications from a total of 3,988
NLG-related articles identified using a literature search, focusing on data
modality, model architecture, clinical applications, and evaluation methods.
Following PRISMA (Preferred Reporting Items for Systematic reviews and
Meta-Analyses) guidelines, we categorize key methods, identify clinical
applications, and assess their capabilities, limitations, and emerging
challenges. This timely review covers the key NLG technologies and medical
applications and provides valuable insights for future studies to leverage NLG
to transform medical discovery and healthcare.

### 2. Large Means Left: Political Bias in Large Language Models Increases with Their Number of Parameters

[Large Means Left: Political Bias in Large Language Models Increases with Their Number of Parameters](http://arxiv.org/pdf/2505.04393v1)

Authors: David Exler, Mark Schutera, Markus Reischl, Luca Rettenberger

With the increasing prevalence of artificial intelligence, careful evaluation
of inherent biases needs to be conducted to form the basis for alleviating the
effects these predispositions can have on users. Large language models (LLMs)
are predominantly used by many as a primary source of information for various
topics. LLMs frequently make factual errors, fabricate data (hallucinations),
or present biases, exposing users to misinformation and influencing opinions.
Educating users on their risks is key to responsible use, as bias, unlike
hallucinations, cannot be caught through data verification. We quantify the
political bias of popular LLMs in the context of the recent vote of the German
Bundestag using the score produced by the Wahl-O-Mat. This metric measures the
alignment between an individual's political views and the positions of German
political parties. We compare the models' alignment scores to identify factors
influencing their political preferences. Doing so, we discover a bias toward
left-leaning parties, most dominant in larger LLMs. Also, we find that the
language we use to communicate with the models affects their political views.
Additionally, we analyze the influence of a model's origin and release date and
compare the results to the outcome of the recent vote of the Bundestag. Our
results imply that LLMs are prone to exhibiting political bias. Large
corporations with the necessary means to develop LLMs, thus, knowingly or
unknowingly, have a responsibility to contain these biases, as they can
influence each voter's decision-making process and inform public opinion in
general and at scale.

### 3. Detecting Spelling and Grammatical Anomalies in Russian Poetry Texts

[Detecting Spelling and Grammatical Anomalies in Russian Poetry Texts](http://arxiv.org/pdf/2505.04507v1)

Authors: Ilya Koziev

The quality of natural language texts in fine-tuning datasets plays a
critical role in the performance of generative models, particularly in
computational creativity tasks such as poem or song lyric generation. Fluency
defects in generated poems significantly reduce their value. However, training
texts are often sourced from internet-based platforms without stringent quality
control, posing a challenge for data engineers to manage defect levels
effectively.
  To address this issue, we propose the use of automated linguistic anomaly
detection to identify and filter out low-quality texts from training datasets
for creative models. In this paper, we present a comprehensive comparison of
unsupervised and supervised text anomaly detection approaches, utilizing both
synthetic and human-labeled datasets. We also introduce the RUPOR dataset, a
collection of Russian-language human-labeled poems designed for cross-sentence
grammatical error detection, and provide the full evaluation code. Our work
aims to empower the community with tools and insights to improve the quality of
training datasets for generative models in creative domains.

### 4. Pangu Ultra MoE: How to Train Your Big MoE on Ascend NPUs

[Pangu Ultra MoE: How to Train Your Big MoE on Ascend NPUs](http://arxiv.org/pdf/2505.04519v1)

Authors: Yehui Tang, Yichun Yin, Yaoyuan Wang, Hang Zhou, Yu Pan, Wei Guo, Ziyang Zhang, Miao Rang, Fangcheng Liu, Naifu Zhang, Binghan Li, Yonghan Dong, Xiaojun Meng, Yasheng Wang, Dong Li, Yin Li, Dandan Tu, Can Chen, Youliang Yan, Fisher Yu, Ruiming Tang, Yunhe Wang, Botian Huang, Bo Wang, Boxiao Liu, Changzheng Zhang, Da Kuang, Fei Liu, Gang Huang, Jiansheng Wei, Jiarui Qin, Jie Ran, Jinpeng Li, Jun Zhao, Liang Dai, Lin Li, Liqun Deng, Peifeng Qin, Pengyuan Zeng, Qiang Gu, Shaohua Tang, Shengjun Cheng, Tao Gao, Tao Yu, Tianshu Li, Tianyu Bi, Wei He, Weikai Mao, Wenyong Huang, Wulong Liu, Xiabing Li, Xianzhi Yu, Xueyu Wu, Xu He, Yangkai Du, Yan Xu, Ye Tian, Yimeng Wu, Yongbing Huang, Yong Tian, Yong Zhu, Yue Li, Yufei Wang, Yuhang Gai, Yujun Li, Yu Luo, Yunsheng Ni, Yusen Sun, Zelin Chen, Zhe Liu, Zhicheng Liu, Zhipeng Tu, Zilin Ding, Zongyuan Zhan

Sparse large language models (LLMs) with Mixture of Experts (MoE) and close
to a trillion parameters are dominating the realm of most capable language
models. However, the massive model scale poses significant challenges for the
underlying software and hardware systems. In this paper, we aim to uncover a
recipe to harness such scale on Ascend NPUs. The key goals are better usage of
the computing resources under the dynamic sparse model structures and
materializing the expected performance gain on the actual hardware. To select
model configurations suitable for Ascend NPUs without repeatedly running the
expensive experiments, we leverage simulation to compare the trade-off of
various model hyperparameters. This study led to Pangu Ultra MoE, a sparse LLM
with 718 billion parameters, and we conducted experiments on the model to
verify the simulation results. On the system side, we dig into Expert
Parallelism to optimize the communication between NPU devices to reduce the
synchronization overhead. We also optimize the memory efficiency within the
devices to further reduce the parameter and activation management overhead. In
the end, we achieve an MFU of 30.0% when training Pangu Ultra MoE, with
performance comparable to that of DeepSeek R1, on 6K Ascend NPUs, and
demonstrate that the Ascend system is capable of harnessing all the training
stages of the state-of-the-art language models. Extensive experiments indicate
that our recipe can lead to efficient training of large-scale sparse language
models with MoE. We also study the behaviors of such models for future
reference.

### 5. ZeroSearch: Incentivize the Search Capability of LLMs without Searching

[ZeroSearch: Incentivize the Search Capability of LLMs without Searching](http://arxiv.org/pdf/2505.04588v1)

Authors: Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Fei Huang, Yan Zhang

Effective information searching is essential for enhancing the reasoning and
generation capabilities of large language models (LLMs). Recent research has
explored using reinforcement learning (RL) to improve LLMs' search capabilities
by interacting with live search engines in real-world environments. While these
approaches show promising results, they face two major challenges: (1)
Uncontrolled Document Quality: The quality of documents returned by search
engines is often unpredictable, introducing noise and instability into the
training process. (2) Prohibitively High API Costs: RL training requires
frequent rollouts, potentially involving hundreds of thousands of search
requests, which incur substantial API expenses and severely constrain
scalability. To address these challenges, we introduce ZeroSearch, a
reinforcement learning framework that incentivizes the search capabilities of
LLMs without interacting with real search engines. Our approach begins with
lightweight supervised fine-tuning to transform the LLM into a retrieval module
capable of generating both relevant and noisy documents in response to a query.
During RL training, we employ a curriculum-based rollout strategy that
incrementally degrades the quality of generated documents, progressively
eliciting the model's reasoning ability by exposing it to increasingly
challenging retrieval scenarios. Extensive experiments demonstrate that
ZeroSearch effectively incentivizes the search capabilities of LLMs using a 3B
LLM as the retrieval module. Remarkably, a 7B retrieval module achieves
comparable performance to the real search engine, while a 14B retrieval module
even surpasses it. Furthermore, it generalizes well across both base and
instruction-tuned models of various parameter sizes and is compatible with a
wide range of RL algorithms.

### 6. Advancing and Benchmarking Personalized Tool Invocation for LLMs

[Advancing and Benchmarking Personalized Tool Invocation for LLMs](http://arxiv.org/pdf/2505.04072v1)

Authors: Xu Huang, Yuefeng Huang, Weiwen Liu, Xingshan Zeng, Yasheng Wang, Ruiming Tang, Hong Xie, Defu Lian

Tool invocation is a crucial mechanism for extending the capabilities of
Large Language Models (LLMs) and has recently garnered significant attention.
It enables LLMs to solve complex problems through tool calls while accessing
up-to-date world knowledge. However, existing work primarily focuses on the
fundamental ability of LLMs to invoke tools for problem-solving, without
considering personalized constraints in tool invocation. In this work, we
introduce the concept of Personalized Tool Invocation and define two key tasks:
Tool Preference and Profile-dependent Query. Tool Preference addresses user
preferences when selecting among functionally similar tools, while
Profile-dependent Query considers cases where a user query lacks certain tool
parameters, requiring the model to infer them from the user profile. To tackle
these challenges, we propose PTool, a data synthesis framework designed for
personalized tool invocation. Additionally, we construct \textbf{PTBench}, the
first benchmark for evaluating personalized tool invocation. We then fine-tune
various open-source models, demonstrating the effectiveness of our framework
and providing valuable insights. Our benchmark is public at
https://github.com/hyfshadow/PTBench.

### 7. Bringing legal knowledge to the public by constructing a legal question bank using large-scale pre-trained language model

[Bringing legal knowledge to the public by constructing a legal question bank using large-scale pre-trained language model](http://arxiv.org/pdf/2505.04132v1)

Authors: Mingruo Yuan, Ben Kao, Tien-Hsuan Wu, Michael M. K. Cheung, Henry W. H. Chan, Anne S. Y. Cheung, Felix W. H. Chan, Yongxi Chen

Access to legal information is fundamental to access to justice. Yet
accessibility refers not only to making legal documents available to the
public, but also rendering legal information comprehensible to them. A vexing
problem in bringing legal information to the public is how to turn formal legal
documents such as legislation and judgments, which are often highly technical,
to easily navigable and comprehensible knowledge to those without legal
education. In this study, we formulate a three-step approach for bringing legal
knowledge to laypersons, tackling the issues of navigability and
comprehensibility. First, we translate selected sections of the law into
snippets (called CLIC-pages), each being a small piece of article that focuses
on explaining certain technical legal concept in layperson's terms. Second, we
construct a Legal Question Bank (LQB), which is a collection of legal questions
whose answers can be found in the CLIC-pages. Third, we design an interactive
CLIC Recommender (CRec). Given a user's verbal description of a legal situation
that requires a legal solution, CRec interprets the user's input and shortlists
questions from the question bank that are most likely relevant to the given
legal situation and recommends their corresponding CLIC pages where relevant
legal knowledge can be found. In this paper we focus on the technical aspects
of creating an LQB. We show how large-scale pre-trained language models, such
as GPT-3, can be used to generate legal questions. We compare machine-generated
questions (MGQs) against human-composed questions (HCQs) and find that MGQs are
more scalable, cost-effective, and more diversified, while HCQs are more
precise. We also show a prototype of CRec and illustrate through an example how
our 3-step approach effectively brings relevant legal knowledge to the public.

### 8. Enhancing Granular Sentiment Classification with Chain-of-Thought Prompting in Large Language Models

[Enhancing Granular Sentiment Classification with Chain-of-Thought Prompting in Large Language Models](http://arxiv.org/pdf/2505.04135v1)

Authors: Vihaan Miriyala, Smrithi Bukkapatnam, Lavanya Prahallad

We explore the use of Chain-of-Thought (CoT) prompting with large language
models (LLMs) to improve the accuracy of granular sentiment categorization in
app store reviews. Traditional numeric and polarity-based ratings often fail to
capture the nuanced sentiment embedded in user feedback. We evaluated the
effectiveness of CoT prompting versus simple prompting on 2000 Amazon app
reviews by comparing each method's predictions to human judgements. CoT
prompting improved classification accuracy from 84% to 93% highlighting the
benefit of explicit reasoning in enhancing sentiment analysis performance.

### 9. Large Language Models are often politically extreme, usually ideologically inconsistent, and persuasive even in informational contexts

[Large Language Models are often politically extreme, usually ideologically inconsistent, and persuasive even in informational contexts](http://arxiv.org/pdf/2505.04171v1)

Authors: Nouar Aldahoul, Hazem Ibrahim, Matteo Varvello, Aaron Kaufman, Talal Rahwan, Yasir Zaki

Large Language Models (LLMs) are a transformational technology, fundamentally
changing how people obtain information and interact with the world. As people
become increasingly reliant on them for an enormous variety of tasks, a body of
academic research has developed to examine these models for inherent biases,
especially political biases, often finding them small. We challenge this
prevailing wisdom. First, by comparing 31 LLMs to legislators, judges, and a
nationally representative sample of U.S. voters, we show that LLMs' apparently
small overall partisan preference is the net result of offsetting extreme views
on specific topics, much like moderate voters. Second, in a randomized
experiment, we show that LLMs can promulgate their preferences into political
persuasiveness even in information-seeking contexts: voters randomized to
discuss political issues with an LLM chatbot are as much as 5 percentage points
more likely to express the same preferences as that chatbot. Contrary to
expectations, these persuasive effects are not moderated by familiarity with
LLMs, news consumption, or interest in politics. LLMs, especially those
controlled by private companies or governments, may become a powerful and
targeted vector for political influence.

### 10. LLM-Independent Adaptive RAG: Let the Question Speak for Itself

[LLM-Independent Adaptive RAG: Let the Question Speak for Itself](http://arxiv.org/pdf/2505.04253v1)

Authors: Maria Marina, Nikolay Ivanov, Sergey Pletenev, Mikhail Salnikov, Daria Galimzianova, Nikita Krayko, Vasily Konovalov, Alexander Panchenko, Viktor Moskvoretskii

Large Language Models~(LLMs) are prone to hallucinations, and
Retrieval-Augmented Generation (RAG) helps mitigate this, but at a high
computational cost while risking misinformation. Adaptive retrieval aims to
retrieve only when necessary, but existing approaches rely on LLM-based
uncertainty estimation, which remain inefficient and impractical. In this
study, we introduce lightweight LLM-independent adaptive retrieval methods
based on external information. We investigated 27 features, organized into 7
groups, and their hybrid combinations. We evaluated these methods on 6 QA
datasets, assessing the QA performance and efficiency. The results show that
our approach matches the performance of complex LLM-based methods while
achieving significant efficiency gains, demonstrating the potential of external
information for adaptive retrieval.

### Cryptography and Security

### 1. A Framework to Prevent Biometric Data Leakage in the Immersive Technologies Domain

[A Framework to Prevent Biometric Data Leakage in the Immersive Technologies Domain](http://arxiv.org/pdf/2505.04123v1)

Authors: Keshav Sood, Iynkaran Natgunanathan, Uthayasanker Thayasivam, Vithurabiman Senthuran, Xiaoning Zhang, Shui Yu

Doubtlessly, the immersive technologies have potential to ease people's life
and uplift economy, however the obvious data privacy risks cannot be ignored.
For example, a participant wears a 3D headset device which detects
participant's head motion to track the pose of participant's head to match the
orientation of camera with participant's eyes positions in the real-world. In a
preliminary study, researchers have proved that the voice command features on
such headsets could lead to major privacy leakages. By analyzing the facial
dynamics captured with the motion sensors, the headsets suffer security
vulnerabilities revealing a user's sensitive speech without user's consent. The
psychography data (such as voice command features, facial dynamics, etc.) is
sensitive data and it should not be leaked out of the device without users
consent else it is a privacy breach. To the best of our literature review, the
work done in this particular research problem is very limited. Motivated from
this, we develop a simple technical framework to mitigate sensitive data (or
biometric data) privacy leaks in immersive technology domain. The performance
evaluation is conducted in a robust way using six data sets, to show that the
proposed solution is effective and feasible to prevent this issue.

### 2. Privacy Challenges In Image Processing Applications

[Privacy Challenges In Image Processing Applications](http://arxiv.org/pdf/2505.04181v1)

Authors: Maneesha, Bharat Gupta, Rishabh Sethi, Charvi Adita Das

As image processing systems proliferate, privacy concerns intensify given the
sensitive personal information contained in images. This paper examines privacy
challenges in image processing and surveys emerging privacy-preserving
techniques including differential privacy, secure multiparty computation,
homomorphic encryption, and anonymization. Key applications with heightened
privacy risks include healthcare, where medical images contain patient health
data, and surveillance systems that can enable unwarranted tracking.
Differential privacy offers rigorous privacy guarantees by injecting controlled
noise, while MPC facilitates collaborative analytics without exposing raw data
inputs. Homomorphic encryption enables computations on encrypted data and
anonymization directly removes identifying elements. However, balancing privacy
protections and utility remains an open challenge. Promising future directions
identified include quantum-resilient cryptography, federated learning,
dedicated hardware, and conceptual innovations like privacy by design.
Ultimately, a holistic effort combining technological innovations, ethical
considerations, and policy frameworks is necessary to uphold the fundamental
right to privacy as image processing capabilities continue advancing rapidly.

### 3. AutoPatch: Multi-Agent Framework for Patching Real-World CVE Vulnerabilities

[AutoPatch: Multi-Agent Framework for Patching Real-World CVE Vulnerabilities](http://arxiv.org/pdf/2505.04195v1)

Authors: Minjae Seo, Wonwoo Choi, Myoungsung You, Seungwon Shin

Large Language Models (LLMs) have emerged as promising tools in software
development, enabling automated code generation and analysis. However, their
knowledge is limited to a fixed cutoff date, making them prone to generating
code vulnerable to newly disclosed CVEs. Frequent fine-tuning with new CVE sets
is costly, and existing LLM-based approaches focus on oversimplified CWE
examples and require providing explicit bug locations to LLMs, limiting their
ability to patch complex real-world vulnerabilities. To address these
limitations, we propose AutoPatch, a multi-agent framework designed to patch
vulnerable LLM-generated code, particularly those introduced after the LLMs'
knowledge cutoff. AutoPatch integrates Retrieval-Augmented Generation (RAG)
with a structured database of recently disclosed vulnerabilities, comprising
525 code snippets derived from 75 high-severity CVEs across real-world systems
such as the Linux kernel and Chrome. AutoPatch combines semantic and taint
analysis to identify the most relevant CVE and leverages enhanced
Chain-of-Thought (CoT) reasoning to construct enriched prompts for verification
and patching. Our unified similarity model, which selects the most relevant
vulnerabilities, achieves 90.4 percent accuracy in CVE matching. AutoPatch
attains 89.5 percent F1-score for vulnerability verification and 95.0 percent
accuracy in patching, while being over 50x more cost-efficient than traditional
fine-tuning approaches.

### 4. On the Vulnerability of Underwater Magnetic Induction Communication

[On the Vulnerability of Underwater Magnetic Induction Communication](http://arxiv.org/pdf/2505.04249v1)

Authors: Muhammad Muzzammil, Waqas Aman, Irfan Ullah, Shang Zhigang, Saif Al-Kuwari, Zhou Tian, Marwa Qaraqe

Typical magnetic induction (MI) communication is commonly considered a secure
underwater wireless communication (UWC) technology due to its non-audible and
non-visible nature compared to acoustic and optical UWC technologies. However,
vulnerabilities in communication systems inevitably exist and may lead to
different types of attacks. In this paper, we investigate the eavesdropping
attack in underwater MI communication to quantitatively measure the system's
vulnerability under this attack. We consider different potential eavesdropping
configuration setups based on the positions and orientations of the
eavesdropper node to investigate how they impact the received voltage and
secrecy at the legitimate receiver node. To this end, we develop
finite-element-method-based simulation models for each configuration in an
underwater environment and evaluate the received voltage and the secrecy
capacity against different system parameters such as magnetic flux, magnetic
flux density, distance, and orientation sensitivity. Furthermore, we construct
an experimental setup within a laboratory environment to replicate the
simulation experiments. Both simulation and lab experimental confirm the
susceptibility of underwater MI communication to eavesdropping attacks.
However, this vulnerability is highly dependent on the position and orientation
of the coil between the eavesdropper and the legitimate transmitter. On the
positive side, we also observe a unique behavior in the received coil reception
that might be used to detect malicious node activities in the vicinity, which
might lead to a potential security mechanism against eavesdropping attacks.

### 5. Applied Post Quantum Cryptography: A Practical Approach for Generating Certificates in Industrial Environments

[Applied Post Quantum Cryptography: A Practical Approach for Generating Certificates in Industrial Environments](http://arxiv.org/pdf/2505.04333v1)

Authors: Nino Ricchizzi, Christian Schwinne, Jan Pelzl

The transition to post-quantum cryptography (PQC) presents significant
challenges for certificate-based identity management in industrial
environments, where secure onboarding of devices relies on long-lived and
interoperable credentials. This work analyzes the integration of PQC into X.509
certificate structures and compares existing tool support for classical,
hybrid, composite, and chameleon certificates. A gap is identified in available
open-source solutions, particularly for the generation and validation of hybrid
and composite certificates via command-line interfaces. To address this, a
proof-of-concept implementation based on the Bouncy Castle library is
developed. The tool supports the creation of classical, hybrid (Catalyst),
composite, and partially chameleon certificates using PQC algorithms such as
ML-DSA and SLH-DSA. It demonstrates compatibility with standard X.509 workflows
and aims to support headless operation and constrained platforms typical of
industrial systems. The implementation is modular, publicly available, and
intended to facilitate further research and testing of PQC migration strategies
in practice. A comparison with OpenSSL-based solutions highlights current
limitations in standardization, toolchain support, and algorithm coverage.

### 6. Reliable Disentanglement Multi-view Learning Against View Adversarial Attacks

[Reliable Disentanglement Multi-view Learning Against View Adversarial Attacks](http://arxiv.org/pdf/2505.04046v1)

Authors: Xuyang Wang, Siyuan Duan, Qizhi Li, Guiduo Duan, Yuan Sun, Dezhong Peng

Recently, trustworthy multi-view learning has attracted extensive attention
because evidence learning can provide reliable uncertainty estimation to
enhance the credibility of multi-view predictions. Existing trusted multi-view
learning methods implicitly assume that multi-view data is secure. In practice,
however, in safety-sensitive applications such as autonomous driving and
security monitoring, multi-view data often faces threats from adversarial
perturbations, thereby deceiving or disrupting multi-view learning models. This
inevitably leads to the adversarial unreliability problem (AUP) in trusted
multi-view learning. To overcome this tricky problem, we propose a novel
multi-view learning framework, namely Reliable Disentanglement Multi-view
Learning (RDML). Specifically, we first propose evidential disentanglement
learning to decompose each view into clean and adversarial parts under the
guidance of corresponding evidences, which is extracted by a pretrained
evidence extractor. Then, we employ the feature recalibration module to
mitigate the negative impact of adversarial perturbations and extract potential
informative features from them. Finally, to further ignore the irreparable
adversarial interferences, a view-level evidential attention mechanism is
designed. Extensive experiments on multi-view classification tasks with
adversarial attacks show that our RDML outperforms the state-of-the-art
multi-view learning methods by a relatively large margin.

### 7. SolPhishHunter: Towards Detecting and Understanding Phishing on Solana

[SolPhishHunter: Towards Detecting and Understanding Phishing on Solana](http://arxiv.org/pdf/2505.04094v1)

Authors: Ziwei Li, Zigui Jiang, Ming Fang, Jiaxin Chen, Zhiying Wu, Jiajing Wu, Lun Zhang, Zibin Zheng

Solana is a rapidly evolving blockchain platform that has attracted an
increasing number of users. However, this growth has also drawn the attention
of malicious actors, with some phishers extending their reach into the Solana
ecosystem. Unlike platforms such as Ethereum, Solana has distinct designs of
accounts and transactions, leading to the emergence of new types of phishing
transactions that we term SolPhish. We define three types of SolPhish and
develop a detection tool called SolPhishHunter. Utilizing SolPhishHunter, we
detect a total of 8,058 instances of SolPhish and conduct an empirical analysis
of these detected cases. Our analysis explores the distribution and impact of
SolPhish, the characteristics of the phishers, and the relationships among
phishing gangs. Particularly, the detected SolPhish transactions have resulted
in nearly \$1.1 million in losses for victims. We report our detection results
to the community and construct SolPhishDataset, the \emph{first} Solana
phishing-related dataset in academia.

### 8. Weaponizing Language Models for Cybersecurity Offensive Operations: Automating Vulnerability Assessment Report Validation; A Review Paper

[Weaponizing Language Models for Cybersecurity Offensive Operations: Automating Vulnerability Assessment Report Validation; A Review Paper](http://arxiv.org/pdf/2505.04265v1)

Authors: Abdulrahman S Almuhaidib, Azlan Mohd Zain, Zalmiyah Zakaria, Izyan Izzati Kamsani, Abdulaziz S Almuhaidib

This, with the ever-increasing sophistication of cyberwar, calls for novel
solutions. In this regard, Large Language Models (LLMs) have emerged as a
highly promising tool for defensive and offensive cybersecurity-related
strategies. While existing literature has focused much on the defensive use of
LLMs, when it comes to their offensive utilization, very little has been
reported-namely, concerning Vulnerability Assessment (VA) report validation.
Consequentially, this paper tries to fill that gap by investigating the
capabilities of LLMs in automating and improving the validation process of the
report of the VA. From the critical review of the related literature, this
paper hereby proposes a new approach to using the LLMs in the automation of the
analysis and within the validation process of the report of the VA that could
potentially reduce the number of false positives and generally enhance
efficiency. These results are promising for LLM automatization for improving
validation on reports coming from VA in order to improve accuracy while
reducing human effort and security postures. The contribution of this paper
provides further evidence about the offensive and defensive LLM capabilities
and therefor helps in devising more appropriate cybersecurity strategies and
tools accordingly.

### 9. Tracing Vulnerability Propagation Across Open Source Software Ecosystems

[Tracing Vulnerability Propagation Across Open Source Software Ecosystems](http://arxiv.org/pdf/2505.04307v1)

Authors: Jukka Ruohonen, Qusai Ramadan

The paper presents a traceability analysis of how over 84 thousand
vulnerabilities have propagated across 28 open source software ecosystems.
According to the results, the propagation sequences have been complex in
general, although GitHub, Debian, and Ubuntu stand out. Furthermore, the
associated propagation delays have been lengthy, and these do not correlate
well with the number of ecosystems involved in the associated sequences. Nor
does the presence or absence of particularly ecosystems in the sequences yield
clear, interpretable patterns. With these results, the paper contributes to the
overlapping knowledge bases about software ecosystems, traceability, and
vulnerabilities.

### 10. Guardians of the Web: The Evolution and Future of Website Information Security

[Guardians of the Web: The Evolution and Future of Website Information Security](http://arxiv.org/pdf/2505.04308v1)

Authors: Md Saiful Islam, Li Xiangdong

Website information security has become a critical concern in the digital
age. This article explores the evolution of website information security,
examining its historical development, current practices, and future directions.
The early beginnings from the 1960s to the 1980s laid the groundwork for modern
cybersecurity, with the development of ARPANET, TCP/IP, public-key
cryptography, and the first antivirus programs. The 1990s marked a
transformative era, driven by the commercialization of the Internet and the
emergence of web-based services. As the Internet grew, so did the range and
sophistication of cyber threats, leading to advancements in security
technologies such as the Secure Sockets Layer (SSL) protocol, password
protection, and firewalls. Current practices in website information security
involve a multi-layered approach, including encryption, secure coding
practices, regular security audits, and user education. The future of website
information security is expected to be shaped by emerging technologies such as
artificial intelligence, blockchain, and quantum computing, as well as the
increasing importance of international cooperation and standardization efforts.
As cyber threats continue to evolve, ongoing research and innovation in website
information security will be essential to protect sensitive information and
maintain trust in the digital world.

### Computer Vision and Pattern Recognition

### 1. FoodTrack: Estimating Handheld Food Portions with Egocentric Video

[FoodTrack: Estimating Handheld Food Portions with Egocentric Video](http://arxiv.org/pdf/2505.04055v1)

Authors: Ervin Wang, Yuhao Chen

Accurately tracking food consumption is crucial for nutrition and health
monitoring. Traditional approaches typically require specific camera angles,
non-occluded images, or rely on gesture recognition to estimate intake, making
assumptions about bite size rather than directly measuring food volume. We
propose the FoodTrack framework for tracking and measuring the volume of
hand-held food items using egocentric video which is robust to hand occlusions
and flexible with varying camera and object poses. FoodTrack estimates food
volume directly, without relying on intake gestures or fixed assumptions about
bite size, offering a more accurate and adaptable solution for tracking food
consumption. We achieve absolute percentage loss of approximately 7.01% on a
handheld food object, improving upon a previous approach that achieved a 16.40%
mean absolute percentage error in its best case, under less flexible
conditions.

### 2. AS3D: 2D-Assisted Cross-Modal Understanding with Semantic-Spatial Scene Graphs for 3D Visual Grounding

[AS3D: 2D-Assisted Cross-Modal Understanding with Semantic-Spatial Scene Graphs for 3D Visual Grounding](http://arxiv.org/pdf/2505.04058v1)

Authors: Feng Xiao, Hongbin Xu, Guocan Zhao, Wenxiong Kang

3D visual grounding aims to localize the unique target described by natural
languages in 3D scenes. The significant gap between 3D and language modalities
makes it a notable challenge to distinguish multiple similar objects through
the described spatial relationships. Current methods attempt to achieve
cross-modal understanding in complex scenes via a target-centered learning
mechanism, ignoring the perception of referred objects. We propose a novel
2D-assisted 3D visual grounding framework that constructs semantic-spatial
scene graphs with referred object discrimination for relationship perception.
The framework incorporates a dual-branch visual encoder that utilizes 2D
pre-trained attributes to guide the multi-modal object encoding. Furthermore,
our cross-modal interaction module uses graph attention to facilitate
relationship-oriented information fusion. The enhanced object representation
and iterative relational learning enable the model to establish effective
alignment between 3D vision and referential descriptions. Experimental results
on the popular benchmarks demonstrate our superior performance compared to
state-of-the-art methods, especially in addressing the challenges of multiple
similar distractors.

### 3. SEVA: Leveraging Single-Step Ensemble of Vicinal Augmentations for Test-Time Adaptation

[SEVA: Leveraging Single-Step Ensemble of Vicinal Augmentations for Test-Time Adaptation](http://arxiv.org/pdf/2505.04087v1)

Authors: Zixuan Hu, Yichun Hu, Ling-Yu Duan

Test-Time adaptation (TTA) aims to enhance model robustness against
distribution shifts through rapid model adaptation during inference. While
existing TTA methods often rely on entropy-based unsupervised training and
achieve promising results, the common practice of a single round of entropy
training is typically unable to adequately utilize reliable samples, hindering
adaptation efficiency. In this paper, we discover augmentation strategies can
effectively unleash the potential of reliable samples, but the rapidly growing
computational cost impedes their real-time application. To address this
limitation, we propose a novel TTA approach named Single-step Ensemble of
Vicinal Augmentations (SEVA), which can take advantage of data augmentations
without increasing the computational burden. Specifically, instead of
explicitly utilizing the augmentation strategy to generate new data, SEVA
develops a theoretical framework to explore the impacts of multiple
augmentations on model adaptation and proposes to optimize an upper bound of
the entropy loss to integrate the effects of multiple rounds of augmentation
training into a single step. Furthermore, we discover and verify that using the
upper bound as the loss is more conducive to the selection mechanism, as it can
effectively filter out harmful samples that confuse the model. Combining these
two key advantages, the proposed efficient loss and a complementary selection
strategy can simultaneously boost the potential of reliable samples and meet
the stringent time requirements of TTA. The comprehensive experiments on
various network architectures across challenging testing scenarios demonstrate
impressive performances and the broad adaptability of SEVA. The code will be
publicly available.

### 4. SMMT: Siamese Motion Mamba with Self-attention for Thermal Infrared Target Tracking

[SMMT: Siamese Motion Mamba with Self-attention for Thermal Infrared Target Tracking](http://arxiv.org/pdf/2505.04088v1)

Authors: Shang Zhang, Huanbin Zhang, Dali Feng, Yujie Cui, Ruoyan Xiong, Cen He

Thermal infrared (TIR) object tracking often suffers from challenges such as
target occlusion, motion blur, and background clutter, which significantly
degrade the performance of trackers. To address these issues, this paper
pro-poses a novel Siamese Motion Mamba Tracker (SMMT), which integrates a
bidirectional state-space model and a self-attention mechanism. Specifically,
we introduce the Motion Mamba module into the Siamese architecture to ex-tract
motion features and recover overlooked edge details using bidirectional
modeling and self-attention. We propose a Siamese parameter-sharing strate-gy
that allows certain convolutional layers to share weights. This approach
reduces computational redundancy while preserving strong feature
represen-tation. In addition, we design a motion edge-aware regression loss to
improve tracking accuracy, especially for motion-blurred targets. Extensive
experi-ments are conducted on four TIR tracking benchmarks, including
LSOTB-TIR, PTB-TIR, VOT-TIR2015, and VOT-TIR 2017. The results show that SMMT
achieves superior performance in TIR target tracking.

### 5. MAISY: Motion-Aware Image SYnthesis for MedicalImage Motion Correction

[MAISY: Motion-Aware Image SYnthesis for MedicalImage Motion Correction](http://arxiv.org/pdf/2505.04105v1)

Authors: Andrew Zhang, Hao Wang, Shuchang Ye, Michael Fulham, Jinman Kim

Patient motion during medical image acquisition causes blurring, ghosting,
and distorts organs, which makes image interpretation challenging.Current
state-of-the-art algorithms using Generative Adversarial Network (GAN)-based
methods with their ability to learn the mappings between corrupted images and
their ground truth via Structural Similarity Index Measure (SSIM) loss
effectively generate motion-free images. However, we identified the following
limitations: (i) they mainly focus on global structural characteristics and
therefore overlook localized features that often carry critical pathological
information, and (ii) the SSIM loss function struggles to handle images with
varying pixel intensities, luminance factors, and variance. In this study, we
propose Motion-Aware Image SYnthesis (MAISY) which initially characterize
motion and then uses it for correction by: (a) leveraging the foundation model
Segment Anything Model (SAM), to dynamically learn spatial patterns along
anatomical boundaries where motion artifacts are most pronounced and, (b)
introducing the Variance-Selective SSIM (VS-SSIM) loss which adaptively
emphasizes spatial regions with high pixel variance to preserve essential
anatomical details during artifact correction. Experiments on chest and head CT
datasets demonstrate that our model outperformed the state-of-the-art
counterparts, with Peak Signal-to-Noise Ratio (PSNR) increasing by 40%, SSIM by
10%, and Dice by 16%.

### 6. One2Any: One-Reference 6D Pose Estimation for Any Object

[One2Any: One-Reference 6D Pose Estimation for Any Object](http://arxiv.org/pdf/2505.04109v1)

Authors: Mengya Liu, Siyuan Li, Ajad Chhatkuli, Prune Truong, Luc Van Gool, Federico Tombari

6D object pose estimation remains challenging for many applications due to
dependencies on complete 3D models, multi-view images, or training limited to
specific object categories. These requirements make generalization to novel
objects difficult for which neither 3D models nor multi-view images may be
available. To address this, we propose a novel method One2Any that estimates
the relative 6-degrees of freedom (DOF) object pose using only a single
reference-single query RGB-D image, without prior knowledge of its 3D model,
multi-view data, or category constraints. We treat object pose estimation as an
encoding-decoding process, first, we obtain a comprehensive Reference Object
Pose Embedding (ROPE) that encodes an object shape, orientation, and texture
from a single reference view. Using this embedding, a U-Net-based pose decoding
module produces Reference Object Coordinate (ROC) for new views, enabling fast
and accurate pose estimation. This simple encoding-decoding framework allows
our model to be trained on any pair-wise pose data, enabling large-scale
training and demonstrating great scalability. Experiments on multiple benchmark
datasets demonstrate that our model generalizes well to novel objects,
achieving state-of-the-art accuracy and robustness even rivaling methods that
require multi-view or CAD inputs, at a fraction of compute.

### 7. GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model

[GAPrompt: Geometry-Aware Point Cloud Prompt for 3D Vision Model](http://arxiv.org/pdf/2505.04119v1)

Authors: Zixiang Ai, Zichen Liu, Yuanhang Lei, Zhenyu Cui, Xu Zou, Jiahuan Zhou

Pre-trained 3D vision models have gained significant attention for their
promising performance on point cloud data. However, fully fine-tuning these
models for downstream tasks is computationally expensive and storage-intensive.
Existing parameter-efficient fine-tuning (PEFT) approaches, which focus
primarily on input token prompting, struggle to achieve competitive performance
due to their limited ability to capture the geometric information inherent in
point clouds. To address this challenge, we propose a novel Geometry-Aware
Point Cloud Prompt (GAPrompt) that leverages geometric cues to enhance the
adaptability of 3D vision models. First, we introduce a Point Prompt that
serves as an auxiliary input alongside the original point cloud, explicitly
guiding the model to capture fine-grained geometric details. Additionally, we
present a Point Shift Prompter designed to extract global shape information
from the point cloud, enabling instance-specific geometric adjustments at the
input level. Moreover, our proposed Prompt Propagation mechanism incorporates
the shape information into the model's feature extraction process, further
strengthening its ability to capture essential geometric characteristics.
Extensive experiments demonstrate that GAPrompt significantly outperforms
state-of-the-art PEFT methods and achieves competitive results compared to full
fine-tuning on various benchmarks, while utilizing only 2.19% of trainable
parameters. Our code is available at
https://github.com/zhoujiahuan1991/ICML2025-VGP.

### 8. Vision Graph Prompting via Semantic Low-Rank Decomposition

[Vision Graph Prompting via Semantic Low-Rank Decomposition](http://arxiv.org/pdf/2505.04121v1)

Authors: Zixiang Ai, Zichen Liu, Jiahuan Zhou

Vision GNN (ViG) demonstrates superior performance by representing images as
graph structures, providing a more natural way to capture irregular semantic
patterns beyond traditional grid or sequence-based representations. To
efficiently adapt ViG to downstream tasks, parameter-efficient fine-tuning
techniques like visual prompting become increasingly essential. However,
existing prompting methods are primarily designed for Transformer-based models,
neglecting the rich topological relationships among nodes and edges in
graph-based representations, limiting their capacity to model complex
semantics. In this paper, we propose Vision Graph Prompting (VGP), a novel
framework tailored for vision graph structures. Our core insight reveals that
semantically connected components in the graph exhibit low-rank properties.
Building on this observation, we introduce a semantic low-rank prompting method
that decomposes low-rank semantic features and integrates them with prompts on
vision graph topologies, capturing both global structural patterns and
fine-grained semantic dependencies. Extensive experiments demonstrate our
method significantly improves ViG's transfer performance on diverse downstream
tasks, achieving results comparable to full fine-tuning while maintaining
parameter efficiency. Our code is available at
https://github.com/zhoujiahuan1991/ICML2025-VGP.

### 9. SToLa: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios

[SToLa: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios](http://arxiv.org/pdf/2505.04201v1)

Authors: Ning Cheng, Jinan Xu, Jialing Chen, Wenjuan Han

This paper explores the challenges of integrating tactile sensing into
intelligent systems for multimodal reasoning, particularly in enabling
commonsense reasoning about the open-ended physical world. We identify two key
challenges: modality discrepancy, where existing large touch-language models
often treat touch as a mere sub-modality of language, and open-ended tactile
data scarcity, where current datasets lack the diversity, open-endness and
complexity needed for reasoning. To overcome these challenges, we introduce
SToLa, a Self-Adaptive Touch-Language framework. SToLa utilizes Mixture of
Experts (MoE) to dynamically process, unify, and manage tactile and language
modalities, capturing their unique characteristics. Crucially, we also present
a comprehensive tactile commonsense reasoning dataset and benchmark featuring
free-form questions and responses, 8 physical properties, 4 interactive
characteristics, and diverse commonsense knowledge. Experiments show SToLa
exhibits competitive performance compared to existing models on the PhysiCLeAR
benchmark and self-constructed datasets, proving the effectiveness of the
Mixture of Experts architecture in multimodal management and the performance
advantages for open-scenario tactile commonsense reasoning tasks.

### 10. CM1 -- A Dataset for Evaluating Few-Shot Information Extraction with Large Vision Language Models

[CM1 -- A Dataset for Evaluating Few-Shot Information Extraction with Large Vision Language Models](http://arxiv.org/pdf/2505.04214v1)

Authors: Fabian Wolf, Oliver Tüselmann, Arthur Matei, Lukas Hennies, Christoph Rass, Gernot A. Fink

The automatic extraction of key-value information from handwritten documents
is a key challenge in document analysis. A reliable extraction is a
prerequisite for the mass digitization efforts of many archives. Large Vision
Language Models (LVLM) are a promising technology to tackle this problem
especially in scenarios where little annotated training data is available. In
this work, we present a novel dataset specifically designed to evaluate the
few-shot capabilities of LVLMs. The CM1 documents are a historic collection of
forms with handwritten entries created in Europe to administer the Care and
Maintenance program after World War Two. The dataset establishes three
benchmarks on extracting name and birthdate information and, furthermore,
considers different training set sizes. We provide baseline results for two
different LVLMs and compare performances to an established full-page extraction
model. While the traditional full-page model achieves highly competitive
performances, our experiments show that when only a few training samples are
available the considered LVLMs benefit from their size and heavy pretraining
and outperform the classical approach.

### Computers and Society

### 1. Identities are not Interchangeable: The Problem of Overgeneralization in Fair Machine Learning

[Identities are not Interchangeable: The Problem of Overgeneralization in Fair Machine Learning](http://arxiv.org/pdf/2505.04038v1)

Authors: Angelina Wang

A key value proposition of machine learning is generalizability: the same
methods and model architecture should be able to work across different domains
and different contexts. While powerful, this generalization can sometimes go
too far, and miss the importance of the specifics. In this work, we look at how
fair machine learning has often treated as interchangeable the identity axis
along which discrimination occurs. In other words, racism is measured and
mitigated the same way as sexism, as ableism, as ageism. Disciplines outside of
computer science have pointed out both the similarities and differences between
these different forms of oppression, and in this work we draw out the
implications for fair machine learning. While certainly not all aspects of fair
machine learning need to be tailored to the specific form of oppression, there
is a pressing need for greater attention to such specificity than is currently
evident. Ultimately, context specificity can deepen our understanding of how to
build more fair systems, widen our scope to include currently overlooked harms,
and, almost paradoxically, also help to narrow our scope and counter the fear
of an infinite number of group-specific methods of analysis.

### 2. From Incidents to Insights: Patterns of Responsibility following AI Harms

[From Incidents to Insights: Patterns of Responsibility following AI Harms](http://arxiv.org/pdf/2505.04291v1)

Authors: Isabel Richards, Claire Benn, Miri Zilka

The AI Incident Database was inspired by aviation safety databases, which
enable collective learning from failures to prevent future incidents. The
database documents hundreds of AI failures, collected from the news and media.
However, criticism highlights that the AIID's reliance on media reporting
limits its utility for learning about implementation failures. In this paper,
we accept that the AIID falls short in its original mission, but argue that by
looking beyond technically-focused learning, the dataset can provide new,
highly valuable insights: specifically, opportunities to learn about patterns
between developers, deployers, victims, wider society, and law-makers that
emerge after AI failures. Through a three-tier mixed-methods analysis of 962
incidents and 4,743 related reports from the AIID, we examine patterns across
incidents, focusing on cases with public responses tagged in the database. We
identify 'typical' incidents found in the AIID, from Tesla crashes to deepfake
scams.
  Focusing on this interplay between relevant parties, we uncover patterns in
accountability and social expectations of responsibility. We find that the
presence of identifiable responsible parties does not necessarily lead to
increased accountability. The likelihood of a response and what it amounts to
depends highly on context, including who built the technology, who was harmed,
and to what extent. Controversy-rich incidents provide valuable data about
societal reactions, including insights into social expectations. Equally
informative are cases where controversy is notably absent. This work shows that
the AIID's value lies not just in preventing technical failures, but in
documenting patterns of harms and of institutional response and social learning
around AI incidents. These patterns offer crucial insights for understanding
how society adapts to and governs emerging AI technologies.

### 3. Resist Platform-Controlled AI Agents and Champion User-Centric Agent Advocates

[Resist Platform-Controlled AI Agents and Champion User-Centric Agent Advocates](http://arxiv.org/pdf/2505.04345v1)

Authors: Sayash Kapoor, Noam Kolt, Seth Lazar

Language model agents could reshape how users navigate and act in digital
environments. If controlled by platform companies -- either those that already
dominate online search, communication, and commerce, or those vying to replace
them -- platform agents could intensify surveillance, exacerbate user lock-in,
and further entrench the incumbent digital giants. This position paper argues
that to resist the undesirable effects of platform agents, we should champion
agent advocates -- agents that are controlled by users, serve the interests of
users, and preserve user autonomy and choice. We identify key interventions to
enable agent advocates: ensuring public access to compute, developing
interoperability protocols and safety standards, and implementing appropriate
market regulations.

### 4. Perpetuating Misogyny with Generative AI: How Model Personalization Normalizes Gendered Harm

[Perpetuating Misogyny with Generative AI: How Model Personalization Normalizes Gendered Harm](http://arxiv.org/pdf/2505.04600v1)

Authors: Laura Wagner, Eva Cetinic

Open-source text-to-image (TTI) pipelines have become dominant in the
landscape of AI-generated visual content, driven by technological advances that
enable users to personalize models through adapters tailored to specific tasks.
While personalization methods such as LoRA offer unprecedented creative
opportunities, they also facilitate harmful practices, including the generation
of non-consensual deepfakes and the amplification of misogynistic or
hypersexualized content. This study presents an exploratory sociotechnical
analysis of CivitAI, the most active platform for sharing and developing
open-source TTI models. Drawing on a dataset of more than 40 million
user-generated images and over 230,000 models, we find a disproportionate rise
in not-safe-for-work (NSFW) content and a significant number of models intended
to mimic real individuals. We also observe a strong influence of internet
subcultures on the tools and practices shaping model personalizations and
resulting visual media. In response to these findings, we contextualize the
emergence of exploitative visual media through feminist and constructivist
perspectives on technology, emphasizing how design choices and community
dynamics shape platform outcomes. Building on this analysis, we propose
interventions aimed at mitigating downstream harm, including improved content
moderation, rethinking tool design, and establishing clearer platform policies
to promote accountability and consent.

### 5. Position: We need responsible, application-driven (RAD) AI research

[Position: We need responsible, application-driven (RAD) AI research](http://arxiv.org/pdf/2505.04104v1)

Authors: Sarah Hartman, Cheng Soon Ong, Julia Powles, Petra Kuhnert

This position paper argues that achieving meaningful scientific and societal
advances with artificial intelligence (AI) requires a responsible,
application-driven approach (RAD) to AI research. As AI is increasingly
integrated into society, AI researchers must engage with the specific contexts
where AI is being applied. This includes being responsive to ethical and legal
considerations, technical and societal constraints, and public discourse. We
present the case for RAD-AI to drive research through a three-staged approach:
(1) building transdisciplinary teams and people-centred studies; (2) addressing
context-specific methods, ethical commitments, assumptions, and metrics; and
(3) testing and sustaining efficacy through staged testbeds and a community of
practice. We present a vision for the future of application-driven AI research
to unlock new value through technically feasible methods that are adaptive to
the contextual needs and values of the communities they ultimately serve.

### 6. Evaluating Performance Consistency in Competitive Programming: Educational Implications and Contest Design Insights

[Evaluating Performance Consistency in Competitive Programming: Educational Implications and Contest Design Insights](http://arxiv.org/pdf/2505.04143v1)

Authors: Zhongtang Luo, Ethan Dickey

Competitive programming (CP) contests are often treated as interchangeable
proxies for algorithmic skill, yet the extent to which results at lower contest
tiers anticipate performance at higher tiers, and how closely any tier
resembles the ubiquitous online-contest circuit, remains unclear. We analyze
ten years (2015--2024) of International Collegiate Programming Contest (ICPC)
standings, comprising five long-running superregional championships (Africa \&
Arab, Asia East, Asia West, North America, and Northern Eurasia), associated
local regionals of North America and Northern Eurasia, and the World Finals.
For 366 World Finalist teams (2021--2024) we augment the dataset with
pre-contest Codeforces ratings. Pairwise rank alignment is measured with
Kendall's $\tau$.
  Overall, superregional ranks predict World Final ranks only moderately
(weighted $\tau=0.407$), but regional-to-superregional consistency varies
widely: Northern Eurasia exhibits the strongest alignment ($\tau=0.521$) while
Asia West exhibits the weakest ($\tau=0.188$). Internal consistency within a
region can exceed its predictive value for Worlds -- e.g., Northern Eurasia and
North America regionals vs. superregionals ($\tau=0.666$ and $\tau=0.577$,
respectively). Codeforces ratings correlate more strongly with World Final
results ($\tau=0.596$) than any single ICPC tier, suggesting that
high-frequency online contests capture decisive skill factors that many
superregional sets miss.
  We argue that contest organizers can improve both fairness and pedagogical
value by aligning problem style and selection rules with the formats that
demonstrably differentiate teams, in particular the Northern-Eurasian model and
well-curated online rounds. All data, scripts, and additional analyses are
publicly released to facilitate replication and further study.

### 7. Large Language Models are often politically extreme, usually ideologically inconsistent, and persuasive even in informational contexts

[Large Language Models are often politically extreme, usually ideologically inconsistent, and persuasive even in informational contexts](http://arxiv.org/pdf/2505.04171v1)

Authors: Nouar Aldahoul, Hazem Ibrahim, Matteo Varvello, Aaron Kaufman, Talal Rahwan, Yasir Zaki

Large Language Models (LLMs) are a transformational technology, fundamentally
changing how people obtain information and interact with the world. As people
become increasingly reliant on them for an enormous variety of tasks, a body of
academic research has developed to examine these models for inherent biases,
especially political biases, often finding them small. We challenge this
prevailing wisdom. First, by comparing 31 LLMs to legislators, judges, and a
nationally representative sample of U.S. voters, we show that LLMs' apparently
small overall partisan preference is the net result of offsetting extreme views
on specific topics, much like moderate voters. Second, in a randomized
experiment, we show that LLMs can promulgate their preferences into political
persuasiveness even in information-seeking contexts: voters randomized to
discuss political issues with an LLM chatbot are as much as 5 percentage points
more likely to express the same preferences as that chatbot. Contrary to
expectations, these persuasive effects are not moderated by familiarity with
LLMs, news consumption, or interest in politics. LLMs, especially those
controlled by private companies or governments, may become a powerful and
targeted vector for political influence.

### 8. A Weak Supervision Learning Approach Towards an Equitable Parking Lot Occupancy Estimation

[A Weak Supervision Learning Approach Towards an Equitable Parking Lot Occupancy Estimation](http://arxiv.org/pdf/2505.04229v1)

Authors: Theophilus Aidoo, Till Koebe, Akansh Maurya, Hewan Shrestha, Ingmar Weber

The scarcity and high cost of labeled high-resolution imagery have long
challenged remote sensing applications, particularly in low-income regions
where high-resolution data are scarce. In this study, we propose a weak
supervision framework that estimates parking lot occupancy using 3m resolution
satellite imagery. By leveraging coarse temporal labels -- based on the
assumption that parking lots of major supermarkets and hardware stores in
Germany are typically full on Saturdays and empty on Sundays -- we train a
pairwise comparison model that achieves an AUC of 0.92 on large parking lots.
The proposed approach minimizes the reliance on expensive high-resolution
images and holds promise for scalable urban mobility analysis. Moreover, the
method can be adapted to assess transit patterns and resource allocation in
vulnerable communities, providing a data-driven basis to improve the well-being
of those most in need.

### 9. Uncertain Machine Ethics Planning

[Uncertain Machine Ethics Planning](http://arxiv.org/pdf/2505.04352v1)

Authors: Simon Kolker, Louise A. Dennis, Ramon Fraga Pereira, Mengwei Xu

Machine Ethics decisions should consider the implications of uncertainty over
decisions. Decisions should be made over sequences of actions to reach
preferable outcomes long term. The evaluation of outcomes, however, may invoke
one or more moral theories, which might have conflicting judgements. Each
theory will require differing representations of the ethical situation. For
example, Utilitarianism measures numerical values, Deontology analyses duties,
and Virtue Ethics emphasises moral character. While balancing potentially
conflicting moral considerations, decisions may need to be made, for example,
to achieve morally neutral goals with minimal costs. In this paper, we
formalise the problem as a Multi-Moral Markov Decision Process and a
Multi-Moral Stochastic Shortest Path Problem. We develop a heuristic algorithm
based on Multi-Objective AO*, utilising Sven-Ove Hansson's Hypothetical
Retrospection procedure for ethical reasoning under uncertainty. Our approach
is validated by a case study from Machine Ethics literature: the problem of
whether to steal insulin for someone who needs it.

### 10. From Flowers to Fascism? The Cottagecore to Tradwife Pipeline on Tumblr

[From Flowers to Fascism? The Cottagecore to Tradwife Pipeline on Tumblr](http://arxiv.org/pdf/2505.04561v1)

Authors: Oliver Mel Allen, Yi Zu, Milo Z. Trujillo, Brooke Foucault Welles

In this work we collected and analyzed social media posts to investigate
aesthetic-based radicalization where users searching for Cottagecore content
may find Tradwife content co-opted by white supremacists, white nationalists,
or other far-right extremist groups. Through quantitative analysis of over
200,000 Tumblr posts and qualitative coding of about 2,500 Tumblr posts, we did
not find evidence of a explicit radicalization. We found that problematic
Tradwife posts found in the literature may be confined to Tradwife-only spaces,
while content in the Cottagecore tag generally did not warrant extra
moderation. However, we did find evidence of a mainstreaming effect in the
overlap between the Tradwife and Cottagecore communities. In our qualitative
analysis there was more interaction between queer and Tradwife identities than
expected based on the literature, and some Tradwives even explicitly included
queer people and disavowed racism in the Tradwife community on Tumblr. This
could be genuine, but more likely it was an example of extremists re-branding
their content and following platform norms to spread ideologies that would
otherwise be rejected by Tumblr users. Additionally, through temporal analysis
we observed a change in the central tags used by Tradwives in the Cottagecore
tag pre- and post- 2021. Initially these posts focused on aesthetics and
hobbies like baking and gardening, but post-2021 the central tags focused more
on religion, traditional gender roles, and homesteading, all markers of
reactionary ideals.

### Databases

### 1. MojoFrame: Dataframe Library in Mojo Language

[MojoFrame: Dataframe Library in Mojo Language](http://arxiv.org/pdf/2505.04080v1)

Authors: Shengya Huang, Zhaoheng Li, Derek Werner, Yongjoo Park

Mojo is an emerging programming language built on MLIR (Multi-Level
Intermediate Representation) and JIT compilation. It enables transparent
optimizations with respect to the underlying hardware (e.g., CPUs, GPUs), while
allowing users to express their logic using Python-like user-friendly syntax.
Mojo has been shown to offer great performance in tensor operations; however,
its performance has not been tested for relational operations (e.g., filtering,
join, and group-by), which are common in data science workflows. To date, no
dataframe implementation exists in the Mojo ecosystem.
  In this paper, we introduce the first Mojo-native dataframe library, called
MojoFrame, that supports core relational operations and user-defined functions
(UDFs). MojoFrame is built on top of Mojo's tensor to achieve fast operations
on numeric columns, while utilizing a cardinality-aware approach to effectively
integrate non-numeric columns for flexible data representation. To achieve high
efficiency, MojoFrame takes significantly different approaches than existing
libraries. MojoFrame supports all operations for TPC-H queries, and achieves up
to 2.97x speedup versus existing dataframe libraries in other programming
languages. Nevertheless, there remain optimization opportunities for MojoFrame
(and the Mojo language), particularly in data loading and dictionary
operations.

### 2. QStore: Quantization-Aware Compressed Model Storage

[QStore: Quantization-Aware Compressed Model Storage](http://arxiv.org/pdf/2505.04081v1)

Authors: Raunak Shah, Zhaoheng Li, Yongjoo Park

Modern applications commonly leverage large, multi-modal foundation models.
These applications often feature complex workflows that demand the storage and
usage of similar models in multiple precisions. A straightforward approach is
to maintain a separate file for each model precision (e.g., INT8, BF16), which
is indeed the approach taken by many model providers such as HuggingFace and
Ollama. However, this approach incurs excessive storage costs since a higher
precision model (e.g., BF16) is a strict superset of a lower precision model
(e.g., INT8) in terms of information. Unfortunately, simply maintaining only
the higher-precision model and requiring every user to dynamically convert the
model precision is not desirable because every user of lower precision models
must pay the cost for model download and precision conversion.
  In this paper, we present QStore, a unified, lossless compression format for
simultaneously storing a model in two (high and low) precisions efficiently.
Instead of storing low-precision and high-precision models separately, QStore
stores low-precision model and only the residual information needed to
reconstruct high-precision models. The size of residual information is
significantly smaller than the original high-precision models, thus achieving
high savings in storage cost. Moreover, QStore does not compromise the speed of
model loading. The low-precision models can be loaded quickly just like before.
The high-precision models can also be reconstructed efficiently in memory by
merging low-precision data and the residual with QStore's lightweight decoding
logic. We evaluate QStore for compressing multiple precisions of popular
foundation models, and show that QStore reduces overall storage footprint by up
to 2.2x (45% of the original size) while enabling up to 1.7x and 1.8x faster
model saving and loading versus existing approaches.

### 3. Global Hash Tables Strike Back! An Analysis of Parallel GROUP BY Aggregation

[Global Hash Tables Strike Back! An Analysis of Parallel GROUP BY Aggregation](http://arxiv.org/pdf/2505.04153v1)

Authors: Daniel Xue, Ryan Marcus

Efficiently computing group aggregations (i.e., GROUP BY) on modern many-core
architectures is critical for analytic database systems. Today's engines
predominately use a partitioned approach to group aggregation, in which an
incoming data stream is partitioned by key values so that every row for a
particular key is sent to the same thread. In this paper, we revisit a simpler
strategy: a fully concurrent group aggregation technique using a shared global
hash table. While approaches using general-purpose concurrent hash tables have
generally been found to perform worse than partitioning-based approaches, we
argue that the key ingredient is customizing the concurrent hash table for the
specific task of group aggregation. Through extensive experiments on synthetic
workloads (varying key cardinality, skew, and thread counts), we demonstrate
that a purpose-built concurrent hash table can match or surpass
partitioning-based techniques. We also analyze the operational characteristics
of both techniques, including resizing costs and memory pressure. In the
process, we derive practical guidelines for database implementers. Overall, our
analysis indicates that fully concurrent group aggregation is a viable
alternative to partitioning.

### 4. In-Context Adaptation to Concept Drift for Learned Database Operations

[In-Context Adaptation to Concept Drift for Learned Database Operations](http://arxiv.org/pdf/2505.04404v1)

Authors: Jiaqi Zhu, Shaofeng Cai, Yanyan Shen, Gang Chen, Fang Deng, Beng Chin Ooi

Machine learning has demonstrated transformative potential for database
operations, such as query optimization and in-database data analytics. However,
dynamic database environments, characterized by frequent updates and evolving
data distributions, introduce concept drift, which leads to performance
degradation for learned models and limits their practical applicability.
Addressing this challenge requires efficient frameworks capable of adapting to
shifting concepts while minimizing the overhead of retraining or fine-tuning.
  In this paper, we propose FLAIR, an online adaptation framework that
introduces a new paradigm called \textit{in-context adaptation} for learned
database operations. FLAIR leverages the inherent property of data systems,
i.e., immediate availability of execution results for predictions, to enable
dynamic context construction. By formalizing adaptation as $f:(\mathbf{x} \,|
\,\mathcal{C}_t) \to \mathbf{y}$, with $\mathcal{C}_t$ representing a dynamic
context memory, FLAIR delivers predictions aligned with the current concept,
eliminating the need for runtime parameter optimization. To achieve this, FLAIR
integrates two key modules: a Task Featurization Module for encoding
task-specific features into standardized representations, and a Dynamic
Decision Engine, pre-trained via Bayesian meta-training, to adapt seamlessly
using contextual information at runtime. Extensive experiments across key
database tasks demonstrate that FLAIR outperforms state-of-the-art baselines,
achieving up to 5.2x faster adaptation and reducing error by 22.5% for
cardinality estimation.

### Distributed, Parallel, and Cluster Computing

### 1. Maxing Out the SVM: Performance Impact of Memory and Program Cache Sizes in the Agave Validator

[Maxing Out the SVM: Performance Impact of Memory and Program Cache Sizes in the Agave Validator](http://arxiv.org/pdf/2505.04129v1)

Authors: Turan Vural, Yuki Yuminaga, Alex Petrosyan, Ben Livshits

In this paper we analyze some of the bottlenecks in the execution pipeline of
Solana's Agave validator client, focusing on RAM and program cache usage under
mainnet conditions. Through a series of controlled experiments, we measure the
validator's throughput and resource efficiency as RAM availability ranges
between 128 GB to 1,536 GB (1.5 TB). We discover that the validator performance
degrades significantly below 256 GB, with transaction processing falling behind
real-time block production. Additionally, we study the program cache behavior,
identifying inefficiencies in program eviction and load latency. Our results
provide practical guidance for hardware provisioning and suggest improvements
to the Solana execution and caching strategy, reducing latency due to the
program cache by 90%.

### 2. Learning-Based Approaches for Job Shop Scheduling Problems: A Review

[Learning-Based Approaches for Job Shop Scheduling Problems: A Review](http://arxiv.org/pdf/2505.04246v1)

Authors: Karima Rihane, Adel Dabah, Abdelhakim AitZai

Job Shop Scheduling (JSS) is one of the most studied combinatorial
optimization problems. It involves scheduling a set of jobs with predefined
processing constraints on a set of machines to achieve a desired objective,
such as minimizing makespan, tardiness, or flowtime. Since it introduction, JSS
has become an attractive research area. Many approaches have been successfully
used to address this problem, including exact methods, heuristics, and
meta-heuristics. Furthermore, various learning-based approaches have been
proposed to solve the JSS problem. However, these approaches are still limited
when compared to the more established methods. This paper summarizes and
evaluates the most important works in the literature on machine learning
approaches for the JSSP. We present models, analyze their benefits and
limitations, and propose future research directions.

### 3. Accelerating Triangle Counting with Real Processing-in-Memory Systems

[Accelerating Triangle Counting with Real Processing-in-Memory Systems](http://arxiv.org/pdf/2505.04269v1)

Authors: Lorenzo Asquini, Manos Frouzakis, Juan Gómez-Luna, Mohammad Sadrosadati, Onur Mutlu, Francesco Silvestri

Triangle Counting (TC) is a procedure that involves enumerating the number of
triangles within a graph. It has important applications in numerous fields,
such as social or biological network analysis and network security. TC is a
memory-bound workload that does not scale efficiently in conventional
processor-centric systems due to several memory accesses across large memory
regions and low data reuse. However, recent Processing-in-Memory (PIM)
architectures present a promising solution to alleviate these bottlenecks. Our
work presents the first TC algorithm that leverages the capabilities of the
UPMEM system, the first commercially available PIM architecture, while at the
same time addressing its limitations. We use a vertex coloring technique to
avoid expensive communication between PIM cores and employ reservoir sampling
to address the limited amount of memory available in the PIM cores' DRAM banks.
In addition, our work makes use of the Misra-Gries summary to speed up counting
triangles on graphs with high-degree nodes and uniform sampling of the graph
edges for quicker approximate results. Our PIM implementation surpasses
state-of-the-art CPU-based TC implementations when processing dynamic graphs in
Coordinate List format, showcasing the effectiveness of the UPMEM architecture
in addressing TC's memory-bound challenges.

### 4. An Asynchronous Distributed-Memory Parallel Algorithm for k-mer Counting

[An Asynchronous Distributed-Memory Parallel Algorithm for k-mer Counting](http://arxiv.org/pdf/2505.04431v1)

Authors: Souvadra Hati, Akihiro Hayashi, Richard Vuduc

This paper describes a new asynchronous algorithm and implementation for the
problem of k-mer counting (KC), which concerns quantifying the frequency of
length k substrings in a DNA sequence. This operation is common to many
computational biology workloads and can take up to 77% of the total runtime of
de novo genome assembly. The performance and scalability of the current
state-of-the-art distributed-memory KC algorithm are hampered by multiple
rounds of Many-To-Many collectives. Therefore, we develop an asynchronous
algorithm (DAKC) that uses fine-grained, asynchronous messages to obviate most
of this global communication while utilizing network bandwidth efficiently via
custom message aggregation protocols. DAKC can perform strong scaling up to 256
nodes (512 sockets / 6K cores) and can count k-mers up to 9x faster than the
state-of-the-art distributed-memory algorithm, and up to 100x faster than the
shared-memory alternative. We also provide an analytical model to understand
the hardware resource utilization of our asynchronous KC algorithm and provide
insights on the performance.

### 5. Optimal Deterministic Rendezvous in Labeled Lines

[Optimal Deterministic Rendezvous in Labeled Lines](http://arxiv.org/pdf/2505.04564v1)

Authors: Yann Bourreau, Ananth Narayanan, Alexandre Nolin

In a rendezvous task, a set of mobile agents dispersed in a network have to
gather at an arbitrary common site. We consider the rendezvous problem on the
infinite labeled line, with $2$ initially asleep agents, without communication,
and a synchronous notion of time. Nodes are labeled with unique positive
integers. The initial distance between the two agents is denoted by $D$. Time
is divided into rounds. We count time from when an agent first wakes up, and
denote by $\tau$ the delay between the agents' wake up times. If awake in a
given round $T$, an agent has three options: stay at its current node $v$, take
port $0$, or take port $1$. If it decides to stay, the agent is still at node
$v$ in round $T+1$. Otherwise, it is at one of the two neighbors of $v$ on the
line, based on the port it chose. The agents achieve rendezvous in $T$ rounds
if they are at the same node in round $T$. We aim for a deterministic algorithm
for this task.
  The problem was recently considered by Miller and Pelc [DISC 2023]. With
$\ell_{\max}$ the largest label of the two starting nodes, they showed that no
algorithm can guarantee rendezvous in $o(D \log^* \ell_{\max})$ rounds. The
lower bound follows from a connection with the LOCAL model of distributed
computing, and holds even if the agents are guaranteed simultaneous wake-up
($\tau = 0$) and are given $D$ as advice. Miller and Pelc also gave an
algorithm of optimal matching complexity $O(D \log^* \ell_{\max})$ when $D$ is
known to the agents, but only obtained the higher bound of $O(D^2 (\log^*
\ell_{\max})^3)$ when $D$ is unknown.
  We improve this second complexity to a tight $O(D \log^* \ell_{\max})$. In
fact, our algorithm achieves rendezvous in $O(D \log^* \ell_{\min})$ rounds,
where $\ell_{\min}$ is the smallest label within distance $O(D)$ of the two
starting positions.

### 6. Plexus: Taming Billion-edge Graphs with 3D Parallel GNN Training

[Plexus: Taming Billion-edge Graphs with 3D Parallel GNN Training](http://arxiv.org/pdf/2505.04083v1)

Authors: Aditya K. Ranjan, Siddharth Singh, Cunyang Wei, Abhinav Bhatele

Graph neural networks have emerged as a potent class of neural networks
capable of leveraging the connectivity and structure of real-world graphs to
learn intricate properties and relationships between nodes. Many real-world
graphs exceed the memory capacity of a GPU due to their sheer size, and using
GNNs on them requires techniques such as mini-batch sampling to scale. However,
this can lead to reduced accuracy in some cases, and sampling and data transfer
from the CPU to the GPU can also slow down training. On the other hand,
distributed full-graph training suffers from high communication overhead and
load imbalance due to the irregular structure of graphs. We propose Plexus, a
three-dimensional (3D) parallel approach for full-graph training that tackles
these issues and scales to billion-edge graphs. Additionally, we introduce
optimizations such as a permutation scheme for load balancing, and a
performance model to predict the optimal 3D configuration. We evaluate Plexus
on several graph datasets and show scaling results for up to 2048 GPUs on
Perlmutter, which is 33% of the machine, and 2048 GCDs on Frontier. Plexus
achieves unprecedented speedups of 2.3x-12.5x over existing methods and a
reduction in the time to solution by 5.2-8.7x on Perlmutter and 7-54.2x on
Frontier.

### 7. Comparing CPU and GPU compute of PERMANOVA on MI300A

[Comparing CPU and GPU compute of PERMANOVA on MI300A](http://arxiv.org/pdf/2505.04556v1)

Authors: Igor Sfiligoi

Comparing the tradeoffs of CPU and GPU compute for memory-heavy algorithms is
often challenging, due to the drastically different memory subsystems on host
CPUs and discrete GPUs. The AMD MI300A is an exception, since it sports both
CPU and GPU cores in a single package, all backed by the same type of HBM
memory. In this paper we analyze the performance of Permutational Multivariate
Analysis of Variance (PERMANOVA), a non-parametric method that tests whether
two or more groups of objects are significantly different based on a
categorical factor. This method is memory-bound and has been recently optimized
for CPU cache locality. Our tests show that GPU cores on the MI300A prefer the
brute force approach instead, significantly outperforming the CPU-based
implementation. The significant benefit of Simultaneous Multithreading (SMT)
was also a pleasant surprise.

### Digital Libraries

### 1. Integrating Large Citation Datasets

[Integrating Large Citation Datasets](http://arxiv.org/pdf/2505.04309v1)

Authors: Inci Yueksel-Erguen, Ida Litzel, Hanqiu Peng

This paper explores methods for building a comprehensive citation graph using
big data techniques to evaluate scientific impact more accurately. Traditional
citation metrics have limitations, and this work investigates merging large
citation datasets to create a more accurate picture. Challenges of big data,
like inconsistent data formats and lack of unique identifiers, are addressed
through deduplication efforts, resulting in a streamlined and reliable merged
dataset with over 119 million records and 1.4 billion citations. We demonstrate
that merging large citation datasets builds a more accurate citation graph
facilitating a more robust evaluation of scientific impact.

### Discrete Mathematics

### 1. Learning-Based Approaches for Job Shop Scheduling Problems: A Review

[Learning-Based Approaches for Job Shop Scheduling Problems: A Review](http://arxiv.org/pdf/2505.04246v1)

Authors: Karima Rihane, Adel Dabah, Abdelhakim AitZai

Job Shop Scheduling (JSS) is one of the most studied combinatorial
optimization problems. It involves scheduling a set of jobs with predefined
processing constraints on a set of machines to achieve a desired objective,
such as minimizing makespan, tardiness, or flowtime. Since it introduction, JSS
has become an attractive research area. Many approaches have been successfully
used to address this problem, including exact methods, heuristics, and
meta-heuristics. Furthermore, various learning-based approaches have been
proposed to solve the JSS problem. However, these approaches are still limited
when compared to the more established methods. This paper summarizes and
evaluates the most important works in the literature on machine learning
approaches for the JSSP. We present models, analyze their benefits and
limitations, and propose future research directions.

### 2. On the mutiplicities of interpoint distances

[On the mutiplicities of interpoint distances](http://arxiv.org/pdf/2505.04283v1)

Authors: Felix Christian Clemen, Adrian Dumitrescu, Dingyuan Liu

Given a set $X\subseteq\mathbb{R}^2$ of $n$ points and a distance $d>0$, the
multiplicity of $d$ is the number of times the distance $d$ appears between
points in $X$. Let $a_1(X) \geq a_2(X) \geq \cdots \geq a_m(X)$ denote the
multiplicities of the $m$ distances determined by $X$ and let
$a(X)=\left(a_1(X),\dots,a_m(X)\right)$. In this paper, we study several
questions from Erd\H{o}s's time regarding distance multiplicities. Among other
results, we show that:
  (1) If $X$ is convex or ``not too convex'', then there exists a distance
other than the diameter that has multiplicity at most $n$.
  (2) There exists a set $X \subseteq \mathbb{R}^2$ of $n$ points, such that
many distances occur with high multiplicity. In particular, at least
$n^{\Omega(1/\log\log{n})}$ distances have superlinear multiplicity in $n$.
  (3) For any (not necessarily fixed) integer $1\leq k\leq\log{n}$, there
exists $X\subseteq\mathbb{R}^2$ of $n$ points, such that the difference between
the $k^{\text{th}}$ and $(k+1)^{\text{th}}$ largest multiplicities is at least
$\Omega(\frac{n\log{n}}{k})$. Moreover, the distances in $X$ with the largest
$k$ multiplicities can be prescribed.
  (4) For every $n\in\mathbb{N}$, there exists $X\subseteq\mathbb{R}^2$ of $n$
points, not all collinear or cocircular, such that $a(X)= (n-1,n-2,\ldots,1)$.
There also exists $Y\subseteq\mathbb{R}^2$ of $n$ points with pairwise distinct
distance multiplicities and $a(Y) \neq (n-1,n-2,\ldots,1)$.

### 3. New bounds for proper $h$-conflict-free colourings

[New bounds for proper $h$-conflict-free colourings](http://arxiv.org/pdf/2505.04543v1)

Authors: Quentin Chuet, Tianjiao Dai, Qiancheng Ouyang, François Pirot

A proper $k$-colouring of a graph $G$ is called $h$-conflict-free if every
vertex $v$ has at least $\min\, \{h, {\rm deg}(v)\}$ colours appearing exactly
once in its neighbourhood. Let $\chi_{\rm pcf}^h(G)$ denote the minimum $k$
such that such a colouring exists. We show that for every fixed $h\ge 1$, every
graph $G$ of maximum degree $\Delta$ satisfies $\chi_{\rm pcf}^h(G) \le h\Delta
+ \mathcal{O}(\log \Delta)$. This expands on the work of Cho et al., and
improves a recent result of Liu and Reed in the case $h=1$. We conjecture that
for every $h\ge 1$ and every graph $G$ of maximum degree $\Delta$ sufficiently
large, the bound $\chi_{\rm pcf}^h(G) \le h\Delta + 1$ should hold, which would
be tight. When the minimum degree $\delta$ of $G$ is sufficiently large, namely
$\delta \ge \max\{100h, 3000\log \Delta\}$, we show that this upper bound can
be further reduced to $\chi_{\rm{pcf}}^h(G) \le \Delta +
\mathcal{O}(\sqrt{h\Delta})$. This improves a recent bound from Kamyczura and
Przyby{\l}o when $\delta \le \sqrt{h\Delta}$.

### 4. Improved bounds on the zeros of the chromatic polynomial of graphs and claw-free graphs

[Improved bounds on the zeros of the chromatic polynomial of graphs and claw-free graphs](http://arxiv.org/pdf/2505.04366v1)

Authors: Ferenc Bencs, Guus Regts

We prove that for any graph $G$ the (complex) zeros of its chromatic
polynomial, $\chi_G(x)$, lie inside the disk centered at $0$ of radius $4.25
\Delta(G)$, where $\Delta(G)$ denote the maximum degree of $G$. This improves
on a recent result of Jenssen, Patel and the second author, who proved a bound
of $5.94\Delta(G)$. We moreover show that for graphs of sufficiently large
girth we can replace $4.25$ by $3.60$ and for claw-free graphs we can replace
$4.25$ by $3.81$.
  Our proofs build on the ideas developed by Jenssen, Patel and the second
author, adding some new ideas. A key novel ingredient for claw-free graphs is
to use a representation of the coefficients of the chromatic polynomial in
terms of the number of certain partial acyclic orientations.

### Data Structures and Algorithms

### 1. The Kinetic Hourglass Data Structure for Computing the Bottleneck Distance of Dynamic Data

[The Kinetic Hourglass Data Structure for Computing the Bottleneck Distance of Dynamic Data](http://arxiv.org/pdf/2505.04048v1)

Authors: Elizabeth Munch, Elena Xinyi Wang, Carola Wenk

The kinetic data structure (KDS) framework is a powerful tool for maintaining
various geometric configurations of continuously moving objects. In this work,
we introduce the kinetic hourglass, a novel KDS implementation designed to
compute the bottleneck distance for geometric matching problems. We detail the
events and updates required for handling general graphs, accompanied by a
complexity analysis. Furthermore, we demonstrate the utility of the kinetic
hourglass by applying it to compute the bottleneck distance between two
persistent homology transforms (PHTs) derived from shapes in $\mathbb{R}^2$,
which are topological summaries obtained by computing persistent homology from
every direction in $\mathbb{S}^1$.

### 2. Fast Pattern Matching with Epsilon Transitions

[Fast Pattern Matching with Epsilon Transitions](http://arxiv.org/pdf/2505.04549v1)

Authors: Nicola Cotumaccio

In the String Matching in Labeled Graphs (SMLG) problem, we need to determine
whether a pattern string appears on a given labeled graph or a given automaton.
Under the Orthogonal Vectors hypothesis, the SMLG problem cannot be solved in
subquadratic time [ICALP 2019]. In typical bioinformatics applications, pattern
matching algorithms should be both fast and space-efficient, so we need to
determine useful classes of graphs on which the SLMG problem can be solved
efficiently.
  In this paper, we improve on a recent result [STACS 2024] that shows how to
solve the SMLG problem in linear time on the compressed representation of
Wheeler generalized automata, a class of string-labeled automata that extend de
Bruijn graphs. More precisely, we show how to remove the assumption that the
automata contain no $ \epsilon $-transitions (namely, edges labeled with the
empty string), while retaining the same time and space bounds. This is a
significant improvement because $ \epsilon $-transitions add considerable
expressive power (making it possible to jump to multiple states for free) and
capture the complexity of regular expressions (through Thompson's construction
for converting a regular expression into an equivalent automaton). We prove
that, to enable $ \epsilon $-transitions, we only need to store two additional
bitvectors that can be constructed in linear time.

### 3. Light Spanners with Small Hop-Diameter

[Light Spanners with Small Hop-Diameter](http://arxiv.org/pdf/2505.04536v1)

Authors: Sujoy Bhore, Lazar Milenkovic

Lightness, sparsity, and hop-diameter are the fundamental parameters of
geometric spanners. Arya et al. [STOC'95] showed in their seminal work that
there exists a construction of Euclidean $(1+\varepsilon)$-spanners with
hop-diameter $O(\log n)$ and lightness $O(\log n)$. They also gave a general
tradeoff of hop-diameter $k$ and sparsity $O(\alpha_k(n))$, where $\alpha_k$ is
a very slowly growing inverse of an Ackermann-style function. The former
combination of logarithmic hop-diameter and lightness is optimal due to the
lower bound by Dinitz et al. [FOCS'08]. Later, Elkin and Solomon [STOC'13]
generalized the light spanner construction to doubling metrics and extended the
tradeoff for more values of hop-diameter $k$. In a recent line of work
[SoCG'22, SoCG'23], Le et al. proved that the aforementioned tradeoff between
the hop-diameter and sparsity is tight for every choice of hop-diameter $k$. A
fundamental question remains: What is the optimal tradeoff between the
hop-diameter and lightness for every value of $k$?
  In this paper, we present a general framework for constructing light spanners
with small hop-diameter. Our framework is based on tree covers. In particular,
we show that if a metric admits a tree cover with $\gamma$ trees, stretch $t$,
and lightness $L$, then it also admits a $t$-spanner with hop-diameter $k$ and
lightness $O(kn^{2/k}\cdot \gamma L)$. Further, we note that the tradeoff for
trees is tight due to a construction in uniform line metric, which is perhaps
the simplest tree metric. As a direct consequence of this framework, we obtain
a tight tradeoff between lightness and hop-diameter for doubling metrics in the
entire regime of $k$.

### 4. Optimal Deterministic Rendezvous in Labeled Lines

[Optimal Deterministic Rendezvous in Labeled Lines](http://arxiv.org/pdf/2505.04564v1)

Authors: Yann Bourreau, Ananth Narayanan, Alexandre Nolin

In a rendezvous task, a set of mobile agents dispersed in a network have to
gather at an arbitrary common site. We consider the rendezvous problem on the
infinite labeled line, with $2$ initially asleep agents, without communication,
and a synchronous notion of time. Nodes are labeled with unique positive
integers. The initial distance between the two agents is denoted by $D$. Time
is divided into rounds. We count time from when an agent first wakes up, and
denote by $\tau$ the delay between the agents' wake up times. If awake in a
given round $T$, an agent has three options: stay at its current node $v$, take
port $0$, or take port $1$. If it decides to stay, the agent is still at node
$v$ in round $T+1$. Otherwise, it is at one of the two neighbors of $v$ on the
line, based on the port it chose. The agents achieve rendezvous in $T$ rounds
if they are at the same node in round $T$. We aim for a deterministic algorithm
for this task.
  The problem was recently considered by Miller and Pelc [DISC 2023]. With
$\ell_{\max}$ the largest label of the two starting nodes, they showed that no
algorithm can guarantee rendezvous in $o(D \log^* \ell_{\max})$ rounds. The
lower bound follows from a connection with the LOCAL model of distributed
computing, and holds even if the agents are guaranteed simultaneous wake-up
($\tau = 0$) and are given $D$ as advice. Miller and Pelc also gave an
algorithm of optimal matching complexity $O(D \log^* \ell_{\max})$ when $D$ is
known to the agents, but only obtained the higher bound of $O(D^2 (\log^*
\ell_{\max})^3)$ when $D$ is unknown.
  We improve this second complexity to a tight $O(D \log^* \ell_{\max})$. In
fact, our algorithm achieves rendezvous in $O(D \log^* \ell_{\min})$ rounds,
where $\ell_{\min}$ is the smallest label within distance $O(D)$ of the two
starting positions.

### 5. Improved bounds on the zeros of the chromatic polynomial of graphs and claw-free graphs

[Improved bounds on the zeros of the chromatic polynomial of graphs and claw-free graphs](http://arxiv.org/pdf/2505.04366v1)

Authors: Ferenc Bencs, Guus Regts

We prove that for any graph $G$ the (complex) zeros of its chromatic
polynomial, $\chi_G(x)$, lie inside the disk centered at $0$ of radius $4.25
\Delta(G)$, where $\Delta(G)$ denote the maximum degree of $G$. This improves
on a recent result of Jenssen, Patel and the second author, who proved a bound
of $5.94\Delta(G)$. We moreover show that for graphs of sufficiently large
girth we can replace $4.25$ by $3.60$ and for claw-free graphs we can replace
$4.25$ by $3.81$.
  Our proofs build on the ideas developed by Jenssen, Patel and the second
author, adding some new ideas. A key novel ingredient for claw-free graphs is
to use a representation of the coefficients of the chromatic polynomial in
terms of the number of certain partial acyclic orientations.

### 6. Testing Juntas Optimally with Samples

[Testing Juntas Optimally with Samples](http://arxiv.org/pdf/2505.04604v1)

Authors: Lorenzo Beretta, Nathaniel Harms, Caleb Koch

We prove tight upper and lower bounds of
$\Theta\left(\tfrac{1}{\epsilon}\left( \sqrt{2^k \log\binom{n}{k} } +
\log\binom{n}{k} \right)\right)$ on the number of samples required for
distribution-free $k$-junta testing. This is the first tight bound for testing
a natural class of Boolean functions in the distribution-free sample-based
model. Our bounds also hold for the feature selection problem, showing that a
junta tester must learn the set of relevant variables. For tolerant junta
testing, we prove a sample lower bound of $\Omega(2^{(1-o(1)) k} +
\log\binom{n}{k})$ showing that, unlike standard testing, there is no large gap
between tolerant testing and learning.

### Emerging Technologies

### 1. Blockchain Data Analytics: A Scoping Literature Review and Directions for Future Research

[Blockchain Data Analytics: A Scoping Literature Review and Directions for Future Research](http://arxiv.org/pdf/2505.04403v1)

Authors: Marcel Bühlmann, Hans-Georg Fill, Simon Curty

Blockchain technology has rapidly expanded beyond its original use in
cryptocurrencies to a broad range of applications, creating vast amounts of
immutable, decentralized data. As blockchain adoption grows, so does the need
for advanced data analytics techniques to extract insights for business
intelligence, fraud detection, financial analysis and many more. While previous
research has examined specific aspects of blockchain data analytics, such as
transaction patterns, illegal activity detection, and data management, there
remains a lack of comprehensive reviews that explore the full scope of
blockchain data analytics. This study addresses this gap through a scoping
literature review, systematically mapping the existing research landscape,
identifying key topics, and highlighting emerging trends. Using established
methodologies for literature reviews, we analyze 466 publications, clustering
them into six major research themes: illegal activity detection, data
management, financial analysis, user analysis, community detection, and mining
analysis. Our findings reveal a strong focus on detecting illicit activities
and financial applications, while holistic business intelligence use cases
remain underexplored. This review provides a structured overview of blockchain
data analytics, identifying research gaps and proposing future directions to
enhance the fields impact.

### 2. Verification of Digital Twins using Classical and Statistical Model Checking

[Verification of Digital Twins using Classical and Statistical Model Checking](http://arxiv.org/pdf/2505.04322v1)

Authors: Raghavendran Gunasekaran, Boudewijn Haverkort

With the increasing adoption of digital techniques, the concept of digital
twin (DT) has received a widespread attention in both industry and academia.
While several definitions exist for a DT, most definitions focus on the
existence of a virtual entity (VE) of a real-world object or process, often
comprising interconnected models which interact with each other, undergoing
changes continuously owing to the synchronization with the real-world object.
These interactions might lead to inconsistencies at execution time, due to
their highly stochastic and/or time-critical nature, which may lead to
undesirable behavior. In addition, the continuously varying nature of VE owing
to its synchronization with the real-world object further contributes to the
complexity arising from these interactions and corresponding model execution
times, which could possibly affect its overall functioning at runtime. This
creates a need to perform (continuous) verification of the VE, to ensure that
it behaves consistently at runtime by adhering to desired properties such as
deadlock freeness, functional correctness, liveness and timeliness. Some
critical properties such as deadlock freeness can only be verified using
classical model checking; on the other hand, statistical model checking
provides the possibility to model actual stochastic temporal behavior. We
therefore propose to use both these techniques to verify the correctness and
the fulfillment of desirable properties of VE. We present our observations and
findings from applying these techniques on the DT of an autonomously driving
truck. Results from these verification techniques suggest that this DT adheres
to properties of deadlock freeness and functional correctness, but not adhering
to timeliness properties.

### 3. Flexing RISC-V Instruction Subset Processors (RISPs) to Extreme Edge

[Flexing RISC-V Instruction Subset Processors (RISPs) to Extreme Edge](http://arxiv.org/pdf/2505.04567v1)

Authors: Alireza Raisiardali, Konstantinos Iordanou, Jedrzej Kufel, Kowshik Gudimetla, Kris Myny, Emre Ozer

This paper presents a methodology for automatically generating processors
that support a subset of the RISC-V instruction set for a new class of
applications at Extreme Edge. The electronics used in extreme edge applications
must be power-efficient, but also provide additional qualities, such as low
cost, conformability, comfort and sustainability. Flexible electronics, rather
than silicon-based electronics, will be capable of meeting these qualities. For
this purpose, we propose a methodology to generate RISPs (RISC-V instruction
subset processors) customised to extreme edge applications and to implement
them as flexible integrated circuits (FlexICs). The methodology is unique in
the sense that verification is an integral part of design. The RISP methodology
treats each instruction in the ISA as a discrete, fully functional,
pre-verified hardware block. It automatically builds a custom processor by
stitching together the hardware blocks of the instructions required by an
application or a set of applications in a specific domain. This approach
significantly reduces the processor verification and its time-to-market. We
generate RISPs using this methodology for three extreme edge applications, and
embedded applications from the Embench benchmark suite, synthesize them as
FlexICs, and compare their power, performance and area to the baselines. Our
results show that RISPs generated using this methodology achieve, on average,
30\% reductions in power and area compared to a RISC-V processor supporting the
full instruction set when synthesized, and are nearly 30 times more energy
efficient with respect to Serv - the world's smallest 32-bit RISC-V processor.
In addition, the full physical implementation of RISPs show up to 21% and 26%
less area and power than Serv.

### 4. KERAIA: An Adaptive and Explainable Framework for Dynamic Knowledge Representation and Reasoning

[KERAIA: An Adaptive and Explainable Framework for Dynamic Knowledge Representation and Reasoning](http://arxiv.org/pdf/2505.04313v1)

Authors: Stephen Richard Varey, Alessandro Di Stefano, The Anh Han

In this paper, we introduce KERAIA, a novel framework and software platform
for symbolic knowledge engineering designed to address the persistent
challenges of representing, reasoning with, and executing knowledge in dynamic,
complex, and context-sensitive environments. The central research question that
motivates this work is: How can unstructured, often tacit, human expertise be
effectively transformed into computationally tractable algorithms that AI
systems can efficiently utilise? KERAIA seeks to bridge this gap by building on
foundational concepts such as Minsky's frame-based reasoning and K-lines, while
introducing significant innovations. These include Clouds of Knowledge for
dynamic aggregation, Dynamic Relations (DRels) for context-sensitive
inheritance, explicit Lines of Thought (LoTs) for traceable reasoning, and
Cloud Elaboration for adaptive knowledge transformation. This approach moves
beyond the limitations of traditional, often static, knowledge representation
paradigms. KERAIA is designed with Explainable AI (XAI) as a core principle,
ensuring transparency and interpretability, particularly through the use of
LoTs. The paper details the framework's architecture, the KSYNTH representation
language, and the General Purpose Paradigm Builder (GPPB) to integrate diverse
inference methods within a unified structure. We validate KERAIA's versatility,
expressiveness, and practical applicability through detailed analysis of
multiple case studies spanning naval warfare simulation, industrial diagnostics
in water treatment plants, and strategic decision-making in the game of RISK.
Furthermore, we provide a comparative analysis against established knowledge
representation paradigms (including ontologies, rule-based systems, and
knowledge graphs) and discuss the implementation aspects and computational
considerations of the KERAIA platform.

### Graphics

### 1. BuildingBlock: A Hybrid Approach for Structured Building Generation

[BuildingBlock: A Hybrid Approach for Structured Building Generation](http://arxiv.org/pdf/2505.04051v1)

Authors: Junming Huang, Chi Wang, Letian Li, Changxin Huang, Qiang Dai, Weiwei Xu

Three-dimensional building generation is vital for applications in gaming,
virtual reality, and digital twins, yet current methods face challenges in
producing diverse, structured, and hierarchically coherent buildings. We
propose BuildingBlock, a hybrid approach that integrates generative models,
procedural content generation (PCG), and large language models (LLMs) to
address these limitations. Specifically, our method introduces a two-phase
pipeline: the Layout Generation Phase (LGP) and the Building Construction Phase
(BCP).
  LGP reframes box-based layout generation as a point-cloud generation task,
utilizing a newly constructed architectural dataset and a Transformer-based
diffusion model to create globally consistent layouts. With LLMs, these layouts
are extended into rule-based hierarchical designs, seamlessly incorporating
component styles and spatial structures.
  The BCP leverages these layouts to guide PCG, enabling local-customizable,
high-quality structured building generation. Experimental results demonstrate
BuildingBlock's effectiveness in generating diverse and hierarchically
structured buildings, achieving state-of-the-art results on multiple
benchmarks, and paving the way for scalable and intuitive architectural
workflows.

### 2. TerraFusion: Joint Generation of Terrain Geometry and Texture Using Latent Diffusion Models

[TerraFusion: Joint Generation of Terrain Geometry and Texture Using Latent Diffusion Models](http://arxiv.org/pdf/2505.04050v1)

Authors: Kazuki Higo, Toshiki Kanai, Yuki Endo, Yoshihiro Kanamori

3D terrain models are essential in fields such as video game development and
film production. Since surface color often correlates with terrain geometry,
capturing this relationship is crucial to achieving realism. However, most
existing methods generate either a heightmap or a texture, without sufficiently
accounting for the inherent correlation. In this paper, we propose a method
that jointly generates terrain heightmaps and textures using a latent diffusion
model. First, we train the model in an unsupervised manner to randomly generate
paired heightmaps and textures. Then, we perform supervised learning of an
external adapter to enable user control via hand-drawn sketches. Experiments
show that our approach allows intuitive terrain generation while preserving the
correlation between heightmaps and textures.

### 3. Person-In-Situ: Scene-Consistent Human Image Insertion with Occlusion-Aware Pose Control

[Person-In-Situ: Scene-Consistent Human Image Insertion with Occlusion-Aware Pose Control](http://arxiv.org/pdf/2505.04052v1)

Authors: Shun Masuda, Yuki Endo, Yoshihiro Kanamori

Compositing human figures into scene images has broad applications in areas
such as entertainment and advertising. However, existing methods often cannot
handle occlusion of the inserted person by foreground objects and unnaturally
place the person in the frontmost layer. Moreover, they offer limited control
over the inserted person's pose. To address these challenges, we propose two
methods. Both allow explicit pose control via a 3D body model and leverage
latent diffusion models to synthesize the person at a contextually appropriate
depth, naturally handling occlusions without requiring occlusion masks. The
first is a two-stage approach: the model first learns a depth map of the scene
with the person through supervised learning, and then synthesizes the person
accordingly. The second method learns occlusion implicitly and synthesizes the
person directly from input data without explicit depth supervision.
Quantitative and qualitative evaluations show that both methods outperform
existing approaches by better preserving scene consistency while accurately
reflecting occlusions and user-specified poses.

### 4. Geometry-Aware Texture Generation for 3D Head Modeling with Artist-driven Control

[Geometry-Aware Texture Generation for 3D Head Modeling with Artist-driven Control](http://arxiv.org/pdf/2505.04387v1)

Authors: Amin Fadaeinejad, Abdallah Dib, Luiz Gustavo Hafemann, Emeline Got, Trevor Anderson, Amaury Depierre, Nikolaus F. Troje, Marcus A. Brubaker, Marc-André Carbonneau

Creating realistic 3D head assets for virtual characters that match a precise
artistic vision remains labor-intensive. We present a novel framework that
streamlines this process by providing artists with intuitive control over
generated 3D heads. Our approach uses a geometry-aware texture synthesis
pipeline that learns correlations between head geometry and skin texture maps
across different demographics. The framework offers three levels of artistic
control: manipulation of overall head geometry, adjustment of skin tone while
preserving facial characteristics, and fine-grained editing of details such as
wrinkles or facial hair. Our pipeline allows artists to make edits to a single
texture map using familiar tools, with our system automatically propagating
these changes coherently across the remaining texture maps needed for realistic
rendering. Experiments demonstrate that our method produces diverse results
with clean geometries. We showcase practical applications focusing on intuitive
control for artists, including skin tone adjustments and simplified editing
workflows for adding age-related details or removing unwanted features from
scanned models. This integrated approach aims to streamline the artistic
workflow in virtual character creation.

### 5. TetWeave: Isosurface Extraction using On-The-Fly Delaunay Tetrahedral Grids for Gradient-Based Mesh Optimization

[TetWeave: Isosurface Extraction using On-The-Fly Delaunay Tetrahedral Grids for Gradient-Based Mesh Optimization](http://arxiv.org/pdf/2505.04590v1)

Authors: Alexandre Binninger, Ruben Wiersma, Philipp Herholz, Olga Sorkine-Hornung

We introduce TetWeave, a novel isosurface representation for gradient-based
mesh optimization that jointly optimizes the placement of a tetrahedral grid
used for Marching Tetrahedra and a novel directional signed distance at each
point. TetWeave constructs tetrahedral grids on-the-fly via Delaunay
triangulation, enabling increased flexibility compared to predefined grids. The
extracted meshes are guaranteed to be watertight, two-manifold and
intersection-free. The flexibility of TetWeave enables a resampling strategy
that places new points where reconstruction error is high and allows to
encourage mesh fairness without compromising on reconstruction error. This
leads to high-quality, adaptive meshes that require minimal memory usage and
few parameters to optimize. Consequently, TetWeave exhibits near-linear memory
scaling relative to the vertex count of the output mesh - a substantial
improvement over predefined grids. We demonstrate the applicability of TetWeave
to a broad range of challenging tasks in computer graphics and vision, such as
multi-view 3D reconstruction, mesh compression and geometric texture
generation.

### 6. PrimitiveAnything: Human-Crafted 3D Primitive Assembly Generation with Auto-Regressive Transformer

[PrimitiveAnything: Human-Crafted 3D Primitive Assembly Generation with Auto-Regressive Transformer](http://arxiv.org/pdf/2505.04622v1)

Authors: Jingwen Ye, Yuze He, Yanning Zhou, Yiqin Zhu, Kaiwen Xiao, Yong-Jin Liu, Wei Yang, Xiao Han

Shape primitive abstraction, which decomposes complex 3D shapes into simple
geometric elements, plays a crucial role in human visual cognition and has
broad applications in computer vision and graphics. While recent advances in 3D
content generation have shown remarkable progress, existing primitive
abstraction methods either rely on geometric optimization with limited semantic
understanding or learn from small-scale, category-specific datasets, struggling
to generalize across diverse shape categories. We present PrimitiveAnything, a
novel framework that reformulates shape primitive abstraction as a primitive
assembly generation task. PrimitiveAnything includes a shape-conditioned
primitive transformer for auto-regressive generation and an ambiguity-free
parameterization scheme to represent multiple types of primitives in a unified
manner. The proposed framework directly learns the process of primitive
assembly from large-scale human-crafted abstractions, enabling it to capture
how humans decompose complex shapes into primitive elements. Through extensive
experiments, we demonstrate that PrimitiveAnything can generate high-quality
primitive assemblies that better align with human perception while maintaining
geometric fidelity across diverse shape categories. It benefits various 3D
applications and shows potential for enabling primitive-based user-generated
content (UGC) in games. Project page: https://primitiveanything.github.io

### 7. ELGAR: Expressive Cello Performance Motion Generation for Audio Rendition

[ELGAR: Expressive Cello Performance Motion Generation for Audio Rendition](http://arxiv.org/pdf/2505.04203v1)

Authors: Zhiping Qiu, Yitong Jin, Yuan Wang, Yi Shi, Chongwu Wang, Chao Tan, Xiaobing Li, Feng Yu, Tao Yu, Qionghai Dai

The art of instrument performance stands as a vivid manifestation of human
creativity and emotion. Nonetheless, generating instrument performance motions
is a highly challenging task, as it requires not only capturing intricate
movements but also reconstructing the complex dynamics of the
performer-instrument interaction. While existing works primarily focus on
modeling partial body motions, we propose Expressive ceLlo performance motion
Generation for Audio Rendition (ELGAR), a state-of-the-art diffusion-based
framework for whole-body fine-grained instrument performance motion generation
solely from audio. To emphasize the interactive nature of the instrument
performance, we introduce Hand Interactive Contact Loss (HICL) and Bow
Interactive Contact Loss (BICL), which effectively guarantee the authenticity
of the interplay. Moreover, to better evaluate whether the generated motions
align with the semantic context of the music audio, we design novel metrics
specifically for string instrument performance motion generation, including
finger-contact distance, bow-string distance, and bowing score. Extensive
evaluations and ablation studies are conducted to validate the efficacy of the
proposed methods. In addition, we put forward a motion generation dataset
SPD-GEN, collated and normalized from the MoCap dataset SPD. As demonstrated,
ELGAR has shown great potential in generating instrument performance motions
with complicated and fast interactions, which will promote further development
in areas such as animation, music education, interactive art creation, etc.

### Computer Science and Game Theory

### 1. PPO-ACT: Proximal Policy Optimization with Adversarial Curriculum Transfer for Spatial Public Goods Games

[PPO-ACT: Proximal Policy Optimization with Adversarial Curriculum Transfer for Spatial Public Goods Games](http://arxiv.org/pdf/2505.04302v1)

Authors: Zhaoqilin Yang, Chanchan Li, Xin Wang, Youliang Tian

This study investigates cooperation evolution mechanisms in the spatial
public goods game. A novel deep reinforcement learning framework, Proximal
Policy Optimization with Adversarial Curriculum Transfer (PPO-ACT), is proposed
to model agent strategy optimization in dynamic environments. Traditional
evolutionary game models frequently exhibit limitations in modeling long-term
decision-making processes. Deep reinforcement learning effectively addresses
this limitation by bridging policy gradient methods with evolutionary game
theory. Our study pioneers the application of proximal policy optimization's
continuous strategy optimization capability to public goods games through a
two-stage adversarial curriculum transfer training paradigm. The experimental
results show that PPO-ACT performs better in critical enhancement factor
regimes. Compared to conventional standard proximal policy optimization
methods, Q-learning and Fermi update rules, achieve earlier cooperation phase
transitions and maintain stable cooperative equilibria. This framework exhibits
better robustness when handling challenging scenarios like all-defector initial
conditions. Systematic comparisons reveal the unique advantage of policy
gradient methods in population-scale cooperation, i.e., achieving
spatiotemporal payoff coordination through value function propagation. Our work
provides a new computational framework for studying cooperation emergence in
complex systems, algorithmically validating the punishment promotes cooperation
hypothesis while offering methodological insights for multi-agent system
strategy design.

### 2. Pool Formation in Oceanic Games: Shapley Value and Proportional Sharing

[Pool Formation in Oceanic Games: Shapley Value and Proportional Sharing](http://arxiv.org/pdf/2505.04422v1)

Authors: Aggelos Kiayias, Elias Koutsoupias, Evangelos Markakis, Panagiotis Tsamopoulos

We study a game-theoretic model for pool formation in Proof of Stake
blockchain protocols. In such systems, stakeholders can form pools as a means
of obtaining regular rewards from participation in ledger maintenance, with the
power of each pool being dependent on its collective stake. The question we are
interested in is the design of mechanisms that suitably split rewards among
pool members and achieve favorable properties in the resulting pool
configuration. With this in mind, we initiate a non-cooperative game-theoretic
analysis of the well known Shapley value scheme from cooperative game theory
into the context of blockchains. In particular, we focus on the oceanic model
of games, proposed by Milnor and Shapley (1978), which is suitable for
populations where a small set of large players coexists with a big mass of
rather small, negligible players. This provides an appropriate level of
abstraction for pool formation processes among the stakeholders. We provide
comparisons between the Shapley mechanism and the more standard proportional
scheme, in terms of attained decentralization, via a Price of Stability
analysis and in terms of susceptibility to Sybil attacks, i.e., the strategic
splitting of a players' stake with the intention of participating in multiple
pools for increased profit. Interestingly, while the widely deployed
proportional scheme appears to have certain advantages, the Shapley value
scheme, which rewards higher the most pivotal players, emerges as a competitive
alternative, by being able to bypass some of the downsides of proportional
sharing, while also not being far from optimal guarantees w.r.t.
decentralization. Finally, we complement our study with some variations of
proportional sharing, where the profit is split in proportion to a
superadditive or a subadditive function of the stake, showing that the Shapley
value scheme still maintains the same advantages.

### 3. Delegation and Participation in Decentralized Governance: An Epistemic View

[Delegation and Participation in Decentralized Governance: An Epistemic View](http://arxiv.org/pdf/2505.04136v1)

Authors: Jeff Strnad

We develop and apply epistemic tests to various decentralized governance
methods as well as to study the impact of participation. These tests probe the
ability to reach a correct outcome when there is one. We find that partial
abstention is a strong governance method from an epistemic standpoint compared
to alternatives such as various forms of ``transfer delegation" in which voters
explicitly transfer some or all of their voting rights to others. We make a
stronger case for multi-step transfer delegation than is present in previous
work but also demonstrate that transfer delegation has inherent epistemic
weaknesses. We show that enhanced direct participation, voters exercising their
own voting rights, can have a variety of epistemic impacts, some very negative.
We identify governance conditions under which additional direct participation
is guaranteed to do no epistemic harm and is likely to increase the probability
of making correct decisions. In light of the epistemic challenges of
voting-based decentralized governance, we consider the possible supplementary
use of prediction markets, auctions, and AI agents to improve outcomes. All
these results are significant because epistemic performance matters if entities
such as DAOs (decentralized autonomous organizations) wish to compete with
organizations that are more centralized.

### Human-Computer Interaction

### 1. State-of-the-Art HCI for Dementia Care: A Scoping Review of Recent Technological Advances

[State-of-the-Art HCI for Dementia Care: A Scoping Review of Recent Technological Advances](http://arxiv.org/pdf/2505.04184v1)

Authors: Yong Ma, Yuchong Zhang, Oda Elise Nordberg, Arvid Rongve, Miroslav Bachinski, Morten Fjeld

Dementia significantly impacts cognitive, behavioral, and functional
abilities, creating challenges for both individuals and caregivers. Recent
advancements in HCI have introduced innovative technological solutions to
support people with dementia (PwD) and their caregivers. This scoping review
systematically examines 32 recent publications from leading digital libraries,
categorizing technological interventions into four key domains: Assistive and
Smart Technology for Daily Life, Social Interaction and Communication,
Well-being and Psychological Support, and Caregiver Support and Training. Our
analysis highlights how emerging technologies are transforming dementia care.
These technologies enhance quality of life by promoting independence, fostering
social engagement, and providing emotional and cognitive support. However, the
review also identifies critical gaps, particularly in addressing the needs of
individuals with early-stage dementia and the lack of individualized support
mechanisms. By emphasizing user-centered design, accessibility, and ethical
considerations, this paper offers a structured roadmap for future research and
practice in dementia care. It bridges the gap between technological innovation
and the real-world needs of PwD and their caregivers, providing valuable
insights for researchers, practitioners, and policymakers. This review not only
synthesizes current advancements but also sets the stage for future HCI-driven
innovations in dementia care, aiming to improve outcomes for an aging global
population.

### 2. Sick of being driven? -- Prevalence and modulating factors of carsickness in the European population in context of automated driving

[Sick of being driven? -- Prevalence and modulating factors of carsickness in the European population in context of automated driving](http://arxiv.org/pdf/2505.04210v1)

Authors: Myriam Metzulat, Barbara Metz, Aaron Edelmann, Alexandra Neukum, Wilfried Kunde

As in automated driving the driver becomes a passenger, carsickness might
reduce comfort for susceptible individuals. Insights in the prevalence of
carsickness and its modulating factors are considered useful for the
development of automated vehicles to mitigate or prevent its occurrence. An
online survey was conducted with N = 3999 participants in Spain, Sweden,
Poland, and Germany. 30% of participants reported to have already experienced
carsickness as adult. The frequency of carsickness was modulated not only by
demographic factors (country, gender, age), but also by frequency of being a
passenger, type of non-driving related task, road type, and the seating
position in car. Furthermore, the efficiency of applied countermeasures,
temporal aspects of carsickness development, as well as the relation of
carsickness with the acceptability of automated driving and the effect on
subjective fitness to drive was investigated. The results are discussed with
focus on automated driving.

### 3. With Friends Like These, Who Needs Explanations? Evaluating User Understanding of Group Recommendations

[With Friends Like These, Who Needs Explanations? Evaluating User Understanding of Group Recommendations](http://arxiv.org/pdf/2505.04273v1)

Authors: Cedric Waterschoot, Raciel Yera Toledo, Nava Tintarev, Francesco Barile

Group Recommender Systems (GRS) employing social choice-based aggregation
strategies have previously been explored in terms of perceived consensus,
fairness, and satisfaction. At the same time, the impact of textual
explanations has been examined, but the results suggest a low effectiveness of
these explanations. However, user understanding remains fairly unexplored, even
if it can contribute positively to transparent GRS. This is particularly
interesting to study in more complex or potentially unfair scenarios when user
preferences diverge, such as in a minority scenario (where group members have
similar preferences, except for a single member in a minority position). In
this paper, we analyzed the impact of different types of explanations on user
understanding of group recommendations. We present a randomized controlled
trial (n = 271) using two between-subject factors: (i) the aggregation strategy
(additive, least misery, and approval voting), and (ii) the modality of
explanation (no explanation, textual explanation, or multimodal explanation).
We measured both subjective (self-perceived by the user) and objective
understanding (performance on model simulation, counterfactuals and error
detection). In line with recent findings on explanations for machine learning
models, our results indicate that more detailed explanations, whether textual
or multimodal, did not increase subjective or objective understanding. However,
we did find a significant effect of aggregation strategies on both subjective
and objective understanding. These results imply that when constructing GRS,
practitioners need to consider that the choice of aggregation strategy can
influence the understanding of users. Post-hoc analysis also suggests that
there is value in analyzing performance on different tasks, rather than through
a single aggregated metric of understanding.

### 4. Improving Inclusivity for Emotion Recognition Based on Face Tracking

[Improving Inclusivity for Emotion Recognition Based on Face Tracking](http://arxiv.org/pdf/2505.04433v1)

Authors: Mats Ole Ellenberg, Katja Krug

The limited expressiveness of virtual user representations in Mixed Reality
and Virtual Reality can inhibit an integral part of communication: emotional
expression. Emotion recognition based on face tracking is often used to
compensate for this. However, emotional facial expressions are highly
individual, which is why many approaches have difficulties recognizing unique
variations of emotional expressions. We propose several strategies to improve
face tracking systems for emotion recognition with and without user
intervention for the Affective Interaction Workshop at CHI '25.

### 5. Practice Support for Violin Bowing by Measuring Bow Pressure and Position

[Practice Support for Violin Bowing by Measuring Bow Pressure and Position](http://arxiv.org/pdf/2505.04446v1)

Authors: Yurina Mizuho, Yuta Sugiura

The violin is one of the most popular musical instruments. Various parameters
of bowing motion, such as pressure, position, and speed, are crucial for
producing a beautiful tone. However, mastering them is challenging and requires
extensive practice. In this study, we aimed to support practice of bowing,
focusing on bow pressure. First, we compared the bowing movements, specifically
bow pressure, bow position, and bow speed, of eight experienced players with
those of eight beginners. Next, we developed and evaluated a visual feedback
system that displays bow pressure to support practice. We taught the identified
differences to 14 beginners, dividing them into two groups: one practiced with
an explanation, and the other with both an explanation and a feedback system.
These two experiments found that clarifying the characteristics unique to
experienced players can support practice.

### 6. A Design Space for the Critical Validation of LLM-Generated Tabular Data

[A Design Space for the Critical Validation of LLM-Generated Tabular Data](http://arxiv.org/pdf/2505.04487v1)

Authors: Madhav Sachdeva, Christopher Narayanan, Marvin Wiedenkeller, Jana Sedlakova, Jürgen Bernard

LLM-generated tabular data is creating new opportunities for data-driven
applications in academia, business, and society. To leverage benefits like
missing value imputation, labeling, and enrichment with context-aware
attributes, LLM-generated data needs a critical validation process. The number
of pioneering approaches is increasing fast, opening a promising validation
space that, so far, remains unstructured. We present a design space for the
critical validation of LLM-generated tabular data with two dimensions: First,
the Analysis Granularity dimension: from within-attribute (single-item and
multi-item) to across-attribute perspectives (1 x 1, 1 x m, and n x n). Second,
the Data Source dimension: differentiating between LLM-generated values, ground
truth values, explanations, and their combinations. We discuss analysis tasks
for each dimension cross-cut, map 19 existing validation approaches, and
discuss the characteristics of two approaches in detail, demonstrating
descriptive power.

### 7. SlideItRight: Using AI to Find Relevant Slides and Provide Feedback for Open-Ended Questions

[SlideItRight: Using AI to Find Relevant Slides and Provide Feedback for Open-Ended Questions](http://arxiv.org/pdf/2505.04584v1)

Authors: Chloe Qianhui Zhao, Jie Cao, Eason Chen, Kenneth R. Koedinger, Jionghao Lin

Feedback is important in supporting student learning. While various automated
feedback systems have been implemented to make the feedback scalable, many
existing solutions only focus on generating text-based feedback. As is
indicated in the multimedia learning principle, learning with more modalities
could help utilize more separate channels, reduce the cognitive load and
facilitate students' learning. Hence, it is important to explore the potential
of Artificial Intelligence (AI) in feedback generation from and to different
modalities. Our study leverages Large Language Models (LLMs) for textual
feedback with the supplementary guidance from other modality - relevant lecture
slide retrieved from the slides hub. Through an online crowdsourcing study
(N=91), this study investigates learning gains and student perceptions using a
2x2 design (i.e., human feedback vs. AI feedback and with vs. without relevant
slide), evaluating the clarity, engagement, perceived effectiveness, and
reliability) of AI-facilitated multimodal feedback. We observed significant
pre-to-post learning gains across all conditions. However, the differences in
these gains were not statistically significant between conditions. The
post-survey revealed that students found the slide feedback helpful in their
learning process, though they reported difficulty in understanding it.
Regarding the AI-generated open-ended feedback, students considered it
personalized and relevant to their responses, but they expressed lower trust in
the AI feedback compared to human-generated feedback.

### 8. Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering

[Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering](http://arxiv.org/pdf/2505.04260v1)

Authors: Jessica Y. Bo, Tianyu Xu, Ishan Chatterjee, Katrina Passarella-Ward, Achin Kulshrestha, D Shin

As large language models (LLMs) improve in their capacity to serve as
personal AI assistants, their ability to output uniquely tailored, personalized
responses that align with the soft preferences of their users is essential for
enhancing user satisfaction and retention. However, untrained lay users have
poor prompt specification abilities and often struggle with conveying their
latent preferences to AI assistants. To address this, we leverage activation
steering to guide LLMs to align with interpretable preference dimensions during
inference. In contrast to memory-based personalization methods that require
longer user history, steering is extremely lightweight and can be easily
controlled by the user via an linear strength factor. We embed steering into
three different interactive chatbot interfaces and conduct a within-subjects
user study (n=14) to investigate how end users prefer to personalize their
conversations. The results demonstrate the effectiveness of preference-based
steering for aligning real-world conversations with hidden user preferences,
and highlight further insights on how diverse values around control, usability,
and transparency lead users to prefer different interfaces.

### 9. Can Language Models Understand Social Behavior in Clinical Conversations?

[Can Language Models Understand Social Behavior in Clinical Conversations?](http://arxiv.org/pdf/2505.04152v1)

Authors: Manas Satish Bedmutha, Feng Chen, Andrea Hartzler, Trevor Cohen, Nadir Weibel

Effective communication between providers and their patients influences
health and care outcomes. The effectiveness of such conversations has been
linked not only to the exchange of clinical information, but also to a range of
interpersonal behaviors; commonly referred to as social signals, which are
often conveyed through non-verbal cues and shape the quality of the
patient-provider relationship. Recent advances in large language models (LLMs)
have demonstrated an increasing ability to infer emotional and social behaviors
even when analyzing only textual information. As automation increases also in
clinical settings, such as for transcription of patient-provider conversations,
there is growing potential for LLMs to automatically analyze and extract social
behaviors from these interactions. To explore the foundational capabilities of
LLMs in tracking social signals in clinical dialogue, we designed task-specific
prompts and evaluated model performance across multiple architectures and
prompting styles using a highly imbalanced, annotated dataset spanning 20
distinct social signals such as provider dominance, patient warmth, etc. We
present the first system capable of tracking all these 20 coded signals, and
uncover patterns in LLM behavior. Further analysis of model configurations and
clinical context provides insights for enhancing LLM performance on social
signal processing tasks in healthcare settings.

### 10. A Dataset and Toolkit for Multiparameter Cardiovascular Physiology Sensing on Rings

[A Dataset and Toolkit for Multiparameter Cardiovascular Physiology Sensing on Rings](http://arxiv.org/pdf/2505.04172v1)

Authors: Iankai Tang, Kegang Wang, Yingke Ding, Jiatong Ji, Zeyu Wang, Xiyuxing Zhang, Ping Chen, Yuanchun Shi, Yuntao Wang

Smart rings offer a convenient way to continuously and unobtrusively monitor
cardiovascular physiological signals. However, a gap remains between the ring
hardware and reliable methods for estimating cardiovascular parameters, partly
due to the lack of publicly available datasets and standardized analysis tools.
In this work, we present $\tau$-Ring, the first open-source ring-based dataset
designed for cardiovascular physiological sensing. The dataset comprises
photoplethysmography signals (infrared and red channels) and 3-axis
accelerometer data collected from two rings (reflective and transmissive
optical paths), with 28.21 hours of raw data from 34 subjects across seven
activities. $\tau$-Ring encompasses both stationary and motion scenarios, as
well as stimulus-evoked abnormal physiological states, annotated with four
ground-truth labels: heart rate, respiratory rate, oxygen saturation, and blood
pressure. Using our proposed RingTool toolkit, we evaluated three widely-used
physics-based methods and four cutting-edge deep learning approaches. Our
results show superior performance compared to commercial rings, achieving best
MAE values of 5.18 BPM for heart rate, 2.98 BPM for respiratory rate, 3.22\%
for oxygen saturation, and 13.33/7.56 mmHg for systolic/diastolic blood
pressure estimation. The open-sourced dataset and toolkit aim to foster further
research and community-driven advances in ring-based cardiovascular health
sensing.

### Information Retrieval

### 1. Towards Large-scale Generative Ranking

[Towards Large-scale Generative Ranking](http://arxiv.org/pdf/2505.04180v1)

Authors: Yanhua Huang, Yuqi Chen, Xiong Cao, Rui Yang, Mingliang Qi, Yinghao Zhu, Qingchang Han, Yaowei Liu, Zhaoyu Liu, Xuefeng Yao, Yuting Jia, Leilei Ma, Yinqi Zhang, Taoyu Zhu, Liujie Zhang, Lei Chen, Weihang Chen, Min Zhu, Ruiwen Xu, Lei Zhang

Generative recommendation has recently emerged as a promising paradigm in
information retrieval. However, generative ranking systems are still
understudied, particularly with respect to their effectiveness and feasibility
in large-scale industrial settings. This paper investigates this topic at the
ranking stage of Xiaohongshu's Explore Feed, a recommender system that serves
hundreds of millions of users. Specifically, we first examine how generative
ranking outperforms current industrial recommenders. Through theoretical and
empirical analyses, we find that the primary improvement in effectiveness stems
from the generative architecture, rather than the training paradigm. To
facilitate efficient deployment of generative ranking, we introduce RankGPT, a
novel generative architecture for ranking. We validate the effectiveness and
efficiency of our solution through online A/B experiments. The results show
that RankGPT achieves significant improvements in user satisfaction with nearly
equivalent computational resources compared to the existing production system.

### 2. CDE-Mapper: Using Retrieval-Augmented Language Models for Linking Clinical Data Elements to Controlled Vocabularies

[CDE-Mapper: Using Retrieval-Augmented Language Models for Linking Clinical Data Elements to Controlled Vocabularies](http://arxiv.org/pdf/2505.04365v1)

Authors: Komal Gilani, Marlo Verket, Christof Peters, Michel Dumontier, Hans-Peter Brunner-La Rocca, Visara Urovi

The standardization of clinical data elements (CDEs) aims to ensure
consistent and comprehensive patient information across various healthcare
systems. Existing methods often falter when standardizing CDEs of varying
representation and complex structure, impeding data integration and
interoperability in clinical research. We introduce CDE-Mapper, an innovative
framework that leverages Retrieval-Augmented Generation approach combined with
Large Language Models to automate the linking of CDEs to controlled
vocabularies. Our modular approach features query decomposition to manage
varying levels of CDEs complexity, integrates expert-defined rules within
prompt engineering, and employs in-context learning alongside multiple
retriever components to resolve terminological ambiguities. In addition, we
propose a knowledge reservoir validated by a human-in-loop approach, achieving
accurate concept linking for future applications while minimizing computational
costs. For four diverse datasets, CDE-Mapper achieved an average of 7.2\%
higher accuracy improvement compared to baseline methods. This work highlights
the potential of advanced language models in improving data harmonization and
significantly advancing capabilities in clinical decision support systems and
research.

### 3. LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders

[LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders](http://arxiv.org/pdf/2505.04421v1)

Authors: Zheng Chai, Qin Ren, Xijun Xiao, Huizhi Yang, Bo Han, Sijun Zhang, Di Chen, Hui Lu, Wenlin Zhao, Lele Yu, Xionghang Xie, Shiru Ren, Xiang Sun, Yaocheng Tan, Peng Xu, Yuchao Zheng, Di Wu

Modeling ultra-long user behavior sequences is critical for capturing both
long- and short-term preferences in industrial recommender systems. Existing
solutions typically rely on two-stage retrieval or indirect modeling paradigms,
incuring upstream-downstream inconsistency and computational inefficiency. In
this paper, we present LONGER, a Long-sequence Optimized traNsformer for
GPU-Efficient Recommenders. LONGER incorporates (i) a global token mechanism
for stabilizing attention over long contexts, (ii) a token merge module with
lightweight InnerTransformers and hybrid attention strategy to reduce quadratic
complexity, and (iii) a series of engineering optimizations, including training
with mixed-precision and activation recomputation, KV cache serving, and the
fully synchronous model training and serving framework for unified GPU-based
dense and sparse parameter updates. LONGER consistently outperforms strong
baselines in both offline metrics and online A/B testing in both advertising
and e-commerce services at ByteDance, validating its consistent effectiveness
and industrial-level scaling laws. Currently, LONGER has been fully deployed at
more than 10 influential scenarios at ByteDance, serving billion users.

### 4. M2Rec: Multi-scale Mamba for Efficient Sequential Recommendation

[M2Rec: Multi-scale Mamba for Efficient Sequential Recommendation](http://arxiv.org/pdf/2505.04445v1)

Authors: Qianru Zhang, Liang Qu, Honggang Wen, Dong Huang, Siu-Ming Yiu, Nguyen Quoc Viet Hung, Hongzhi Yin

Sequential recommendation systems aim to predict users' next preferences
based on their interaction histories, but existing approaches face critical
limitations in efficiency and multi-scale pattern recognition. While
Transformer-based methods struggle with quadratic computational complexity,
recent Mamba-based models improve efficiency but fail to capture periodic user
behaviors, leverage rich semantic information, or effectively fuse multimodal
features. To address these challenges, we propose \model, a novel sequential
recommendation framework that integrates multi-scale Mamba with Fourier
analysis, Large Language Models (LLMs), and adaptive gating. First, we enhance
Mamba with Fast Fourier Transform (FFT) to explicitly model periodic patterns
in the frequency domain, separating meaningful trends from noise. Second, we
incorporate LLM-based text embeddings to enrich sparse interaction data with
semantic context from item descriptions. Finally, we introduce a learnable gate
mechanism to dynamically balance temporal (Mamba), frequency (FFT), and
semantic (LLM) features, ensuring harmonious multimodal fusion. Extensive
experiments demonstrate that \model\ achieves state-of-the-art performance,
improving Hit Rate@10 by 3.2\% over existing Mamba-based models while
maintaining 20\% faster inference than Transformer baselines. Our results
highlight the effectiveness of combining frequency analysis, semantic
understanding, and adaptive fusion for sequential recommendation. Code and
datasets are available at: https://anonymous.4open.science/r/M2Rec.

### 5. User and Recommender Behavior Over Time: Contextualizing Activity, Effectiveness, Diversity, and Fairness in Book Recommendation

[User and Recommender Behavior Over Time: Contextualizing Activity, Effectiveness, Diversity, and Fairness in Book Recommendation](http://arxiv.org/pdf/2505.04518v1)

Authors: Samira Vaez Barenji, Sushobhan Parajuli, Michael D. Ekstrand

Data is an essential resource for studying recommender systems. While there
has been significant work on improving and evaluating state-of-the-art models
and measuring various properties of recommender system outputs, less attention
has been given to the data itself, particularly how data has changed over time.
Such documentation and analysis provide guidance and context for designing and
evaluating recommender systems, particularly for evaluation designs making use
of time (e.g., temporal splitting). In this paper, we present a temporal
explanatory analysis of the UCSD Book Graph dataset scraped from Goodreads, a
social reading and recommendation platform active since 2006. We measure the
book interaction data using a set of activity, diversity, and fairness metrics;
we then train a set of collaborative filtering algorithms on rolling training
windows to observe how the same measures evolve over time in the
recommendations. Additionally, we explore whether the introduction of
algorithmic recommendations in 2011 was followed by observable changes in user
or recommender system behavior.

### 6. Retrieval Augmented Time Series Forecasting

[Retrieval Augmented Time Series Forecasting](http://arxiv.org/pdf/2505.04163v1)

Authors: Sungwon Han, Seungeon Lee, Meeyoung Cha, Sercan O Arik, Jinsung Yoon

Time series forecasting uses historical data to predict future trends,
leveraging the relationships between past observations and available features.
In this paper, we propose RAFT, a retrieval-augmented time series forecasting
method to provide sufficient inductive biases and complement the model's
learning capacity. When forecasting the subsequent time frames, we directly
retrieve historical data candidates from the training dataset with patterns
most similar to the input, and utilize the future values of these candidates
alongside the inputs to obtain predictions. This simple approach augments the
model's capacity by externally providing information about past patterns via
retrieval modules. Our empirical evaluations on ten benchmark datasets show
that RAFT consistently outperforms contemporary baselines with an average win
ratio of 86%.

### 7. Tetrahedron-Net for Medical Image Registration

[Tetrahedron-Net for Medical Image Registration](http://arxiv.org/pdf/2505.04380v1)

Authors: Jinhai Xiang, Shuai Guo, Qianru Han, Dantong Shi, Xinwei He, Xiang Bai

Medical image registration plays a vital role in medical image processing.
Extracting expressive representations for medical images is crucial for
improving the registration quality. One common practice for this end is
constructing a convolutional backbone to enable interactions with skip
connections among feature extraction layers. The de facto structure, U-Net-like
networks, has attempted to design skip connections such as nested or full-scale
ones to connect one single encoder and one single decoder to improve its
representation capacity. Despite being effective, it still does not fully
explore interactions with a single encoder and decoder architectures. In this
paper, we embrace this observation and introduce a simple yet effective
alternative strategy to enhance the representations for registrations by
appending one additional decoder. The new decoder is designed to interact with
both the original encoder and decoder. In this way, it not only reuses feature
presentation from corresponding layers in the encoder but also interacts with
the original decoder to corporately give more accurate registration results.
The new architecture is concise yet generalized, with only one encoder and two
decoders forming a ``Tetrahedron'' structure, thereby dubbed Tetrahedron-Net.
Three instantiations of Tetrahedron-Net are further constructed regarding the
different structures of the appended decoder. Our extensive experiments prove
that superior performance can be obtained on several representative benchmarks
of medical image registration. Finally, such a ``Tetrahedron'' design can also
be easily integrated into popular U-Net-like architectures including
VoxelMorph, ViT-V-Net, and TransMorph, leading to consistent performance gains.

### 8. Theoretical Guarantees for LT-TTD: A Unified Transformer-based Architecture for Two-Level Ranking Systems

[Theoretical Guarantees for LT-TTD: A Unified Transformer-based Architecture for Two-Level Ranking Systems](http://arxiv.org/pdf/2505.04434v1)

Authors: Ayoub Abraich

Modern recommendation and search systems typically employ multi-stage ranking
architectures to efficiently handle billions of candidates. The conventional
approach uses distinct L1 (candidate retrieval) and L2 (re-ranking) models with
different optimization objectives, introducing critical limitations including
irreversible error propagation and suboptimal ranking. This paper identifies
and analyzes the fundamental limitations of this decoupled paradigm and
proposes LT-TTD (Listwise Transformer with Two-Tower Distillation), a novel
unified architecture that bridges retrieval and ranking phases. Our approach
combines the computational efficiency of two-tower models with the expressivity
of transformers in a unified listwise learning framework. We provide a
comprehensive theoretical analysis of our architecture and establish formal
guarantees regarding error propagation mitigation, ranking quality
improvements, and optimization convergence. We derive theoretical bounds
showing that LT-TTD reduces the upper limit on irretrievable relevant items by
a factor that depends on the knowledge distillation strength, and prove that
our multi-objective optimization framework achieves a provably better global
optimum than disjoint training. Additionally, we analyze the computational
complexity of our approach, demonstrating that the asymptotic complexity
remains within practical bounds for real-world applications. We also introduce
UPQE, a novel evaluation metric specifically designed for unified ranking
architectures that holistically captures retrieval quality, ranking
performance, and computational efficiency.

### 9. To Judge or not to Judge: Using LLM Judgements for Advertiser Keyphrase Relevance at eBay

[To Judge or not to Judge: Using LLM Judgements for Advertiser Keyphrase Relevance at eBay](http://arxiv.org/pdf/2505.04209v1)

Authors: Soumik Dey, Hansi Wu, Binbin Li

E-commerce sellers are recommended keyphrases based on their inventory on
which they advertise to increase buyer engagement (clicks/sales). The relevance
of advertiser keyphrases plays an important role in preventing the inundation
of search systems with numerous irrelevant items that compete for attention in
auctions, in addition to maintaining a healthy seller perception. In this work,
we describe the shortcomings of training Advertiser keyphrase relevance filter
models on click/sales/search relevance signals and the importance of aligning
with human judgment, as sellers have the power to adopt or reject said
keyphrase recommendations. In this study, we frame Advertiser keyphrase
relevance as a complex interaction between 3 dynamical systems -- seller
judgment, which influences seller adoption of our product, Advertising, which
provides the keyphrases to bid on, and Search, who holds the auctions for the
same keyphrases. This study discusses the practicalities of using human
judgment via a case study at eBay Advertising and demonstrate that using
LLM-as-a-judge en-masse as a scalable proxy for seller judgment to train our
relevance models achieves a better harmony across the three systems -- provided
that they are bound by a meticulous evaluation framework grounded in business
metrics.

### Machine Learning

### 1. Alpha Excel Benchmark

[Alpha Excel Benchmark](http://arxiv.org/pdf/2505.04110v1)

Authors: David Noever, Forrest McKee

This study presents a novel benchmark for evaluating Large Language Models
(LLMs) using challenges derived from the Financial Modeling World Cup (FMWC)
Excel competitions. We introduce a methodology for converting 113 existing FMWC
challenges into programmatically evaluable JSON formats and use this dataset to
compare the performance of several leading LLMs. Our findings demonstrate
significant variations in performance across different challenge categories,
with models showing specific strengths in pattern recognition tasks but
struggling with complex numerical reasoning. The benchmark provides a
standardized framework for assessing LLM capabilities in realistic
business-oriented tasks rather than abstract academic problems. This research
contributes to the growing field of AI benchmarking by establishing proficiency
among the 1.5 billion people who daily use Microsoft Excel as a meaningful
evaluation metric that bridges the gap between academic AI benchmarks and
practical business applications.

### 2. LHT: Statistically-Driven Oblique Decision Trees for Interpretable Classification

[LHT: Statistically-Driven Oblique Decision Trees for Interpretable Classification](http://arxiv.org/pdf/2505.04139v1)

Authors: Hongyi Li, Jun Xu, William Ward Armstrong

We introduce the Learning Hyperplane Tree (LHT), a novel oblique decision
tree model designed for expressive and interpretable classification. LHT
fundamentally distinguishes itself through a non-iterative,
statistically-driven approach to constructing splitting hyperplanes. Unlike
methods that rely on iterative optimization or heuristics, LHT directly
computes the hyperplane parameters, which are derived from feature weights
based on the differences in feature expectations between classes within each
node. This deterministic mechanism enables a direct and well-defined hyperplane
construction process. Predictions leverage a unique piecewise linear membership
function within leaf nodes, obtained via local least-squares fitting. We
formally analyze the convergence of the LHT splitting process, ensuring that
each split yields meaningful, non-empty partitions. Furthermore, we establish
that the time complexity for building an LHT up to depth $d$ is $O(mnd)$,
demonstrating the practical feasibility of constructing trees with powerful
oblique splits using this methodology. The explicit feature weighting at each
split provides inherent interpretability. Experimental results on benchmark
datasets demonstrate LHT's competitive accuracy, positioning it as a practical,
theoretically grounded, and interpretable alternative in the landscape of
tree-based models. The implementation of the proposed method is available at
https://github.com/Hongyi-Li-sz/LHT_model.

### 3. FilterTS: Comprehensive Frequency Filtering for Multivariate Time Series Forecasting

[FilterTS: Comprehensive Frequency Filtering for Multivariate Time Series Forecasting](http://arxiv.org/pdf/2505.04158v1)

Authors: Yulong Wang, Yushuo Liu, Xiaoyi Duan, Kai Wang

Multivariate time series forecasting is crucial across various industries,
where accurate extraction of complex periodic and trend components can
significantly enhance prediction performance. However, existing models often
struggle to capture these intricate patterns. To address these challenges, we
propose FilterTS, a novel forecasting model that utilizes specialized filtering
techniques based on the frequency domain. FilterTS introduces a Dynamic
Cross-Variable Filtering Module, a key innovation that dynamically leverages
other variables as filters to extract and reinforce shared variable frequency
components across variables in multivariate time series. Additionally, a Static
Global Filtering Module captures stable frequency components, identified
throughout the entire training set. Moreover, the model is built in the
frequency domain, converting time-domain convolutions into frequency-domain
multiplicative operations to enhance computational efficiency. Extensive
experimental results on eight real-world datasets have demonstrated that
FilterTS significantly outperforms existing methods in terms of prediction
accuracy and computational efficiency.

### 4. STRGCN: Capturing Asynchronous Spatio-Temporal Dependencies for Irregular Multivariate Time Series Forecasting

[STRGCN: Capturing Asynchronous Spatio-Temporal Dependencies for Irregular Multivariate Time Series Forecasting](http://arxiv.org/pdf/2505.04167v1)

Authors: Yulong Wang, Xiaofeng Hu, Xiaojian Cui, Kai Wang

Irregular multivariate time series (IMTS) are prevalent in real-world
applications across many fields, where varying sensor frequencies and
asynchronous measurements pose significant modeling challenges. Existing
solutions often rely on a pre-alignment strategy to normalize data, which can
distort intrinsic patterns and escalate computational and memory demands.
Addressing these limitations, we introduce STRGCN, a Spatio-Temporal Relational
Graph Convolutional Network that avoids pre-alignment and directly captures the
complex interdependencies in IMTS by representing them as a fully connected
graph. Each observation is represented as a node, allowing the model to
effectively handle misaligned timestamps by mapping all inter-node
relationships, thus faithfully preserving the asynchronous nature of the data.
Moreover, we enhance this model with a hierarchical ``Sandwich'' structure that
strategically aggregates nodes to optimize graph embeddings, reducing
computational overhead while maintaining detailed local and global context.
Extensive experiments on four public datasets demonstrate that STRGCN achieves
state-of-the-art accuracy, competitive memory usage and training speed.

### 5. DiffPattern-Flex: Efficient Layout Pattern Generation via Discrete Diffusion

[DiffPattern-Flex: Efficient Layout Pattern Generation via Discrete Diffusion](http://arxiv.org/pdf/2505.04173v1)

Authors: Zixiao Wang, Wenqian Zhao, Yunheng Shen, Yang Bai, Guojin Chen, Farzan Farnia, Bei Yu

Recent advancements in layout pattern generation have been dominated by deep
generative models. However, relying solely on neural networks for legality
guarantees raises concerns in many practical applications. In this paper, we
present \tool{DiffPattern}-Flex, a novel approach designed to generate reliable
layout patterns efficiently. \tool{DiffPattern}-Flex incorporates a new method
for generating diverse topologies using a discrete diffusion model while
maintaining a lossless and compute-efficient layout representation. To ensure
legal pattern generation, we employ {an} optimization-based, white-box pattern
assessment process based on specific design rules. Furthermore, fast sampling
and efficient legalization technologies are employed to accelerate the
generation process. Experimental results across various benchmarks demonstrate
that \tool{DiffPattern}-Flex significantly outperforms existing methods and
excels at producing reliable layout patterns.

### 6. Cyber Security Data Science: Machine Learning Methods and their Performance on Imbalanced Datasets

[Cyber Security Data Science: Machine Learning Methods and their Performance on Imbalanced Datasets](http://arxiv.org/pdf/2505.04204v1)

Authors: Mateo Lopez-Ledezma, Gissel Velarde

Cybersecurity has become essential worldwide and at all levels, concerning
individuals, institutions, and governments. A basic principle in cybersecurity
is to be always alert. Therefore, automation is imperative in processes where
the volume of daily operations is large. Several cybersecurity applications can
be addressed as binary classification problems, including anomaly detection,
fraud detection, intrusion detection, spam detection, or malware detection. We
present three experiments. In the first experiment, we evaluate single
classifiers including Random Forests, Light Gradient Boosting Machine, eXtreme
Gradient Boosting, Logistic Regression, Decision Tree, and Gradient Boosting
Decision Tree. In the second experiment, we test different sampling techniques
including over-sampling, under-sampling, Synthetic Minority Over-sampling
Technique, and Self-Paced Ensembling. In the last experiment, we evaluate
Self-Paced Ensembling and its number of base classifiers. We found that
imbalance learning techniques had positive and negative effects, as reported in
related studies. Thus, these techniques should be applied with caution.
Besides, we found different best performers for each dataset. Therefore, we
recommend testing single classifiers and imbalance learning techniques for each
new dataset and application involving imbalanced datasets as is the case in
several cyber security applications.

### 7. Technology prediction of a 3D model using Neural Network

[Technology prediction of a 3D model using Neural Network](http://arxiv.org/pdf/2505.04241v1)

Authors: Grzegorz Miebs, Rafał A. Bachorz

Accurate estimation of production times is critical for effective
manufacturing scheduling, yet traditional methods relying on expert analysis or
historical data often fall short in dynamic or customized production
environments. This paper introduces a data-driven approach that predicts
manufacturing steps and their durations directly from a product's 3D model. By
rendering the model into multiple 2D images and leveraging a neural network
inspired by the Generative Query Network, the method learns to map geometric
features into time estimates for predefined production steps enabling scalable,
adaptive, and precise process planning across varied product types.

### 8. Hyperbolic Fuzzy $C$-Means with Adaptive Weight-based Filtering for Clustering in Non-Euclidean Spaces

[Hyperbolic Fuzzy $C$-Means with Adaptive Weight-based Filtering for Clustering in Non-Euclidean Spaces](http://arxiv.org/pdf/2505.04335v1)

Authors: Swagato Das, Arghya Pratihar, Swagatam Das

Clustering algorithms play a pivotal role in unsupervised learning by
identifying and grouping similar objects based on shared characteristics. While
traditional clustering techniques, such as hard and fuzzy center-based
clustering, have been widely used, they struggle with complex,
high-dimensional, and non-Euclidean datasets. In particular, the Fuzzy
$C$-Means (FCM) algorithm, despite its efficiency and popularity, exhibits
notable limitations in non-Euclidean spaces. Euclidean spaces assume linear
separability and uniform distance scaling, limiting their effectiveness in
capturing complex, hierarchical, or non-Euclidean structures in fuzzy
clustering. To overcome these challenges, we introduce Filtration-based
Hyperbolic Fuzzy $C$-Means (HypeFCM), a novel clustering algorithm tailored for
better representation of data relationships in non-Euclidean spaces. HypeFCM
integrates the principles of fuzzy clustering with hyperbolic geometry and
employs a weight-based filtering mechanism to improve performance. The
algorithm initializes weights using a Dirichlet distribution and iteratively
refines cluster centroids and membership assignments based on a hyperbolic
metric in the Poincar\'e Disc model. Extensive experimental evaluations
demonstrate that HypeFCM significantly outperforms conventional fuzzy
clustering methods in non-Euclidean settings, underscoring its robustness and
effectiveness.

### 9. Riemannian Denoising Diffusion Probabilistic Models

[Riemannian Denoising Diffusion Probabilistic Models](http://arxiv.org/pdf/2505.04338v1)

Authors: Zichen Liu, Wei Zhang, Christof Schütte, Tiejun Li

We propose Riemannian Denoising Diffusion Probabilistic Models (RDDPMs) for
learning distributions on submanifolds of Euclidean space that are level sets
of functions, including most of the manifolds relevant to applications.
Existing methods for generative modeling on manifolds rely on substantial
geometric information such as geodesic curves or eigenfunctions of the
Laplace-Beltrami operator and, as a result, they are limited to manifolds where
such information is available. In contrast, our method, built on a projection
scheme, can be applied to more general manifolds, as it only requires being
able to evaluate the value and the first order derivatives of the function that
defines the submanifold. We provide a theoretical analysis of our method in the
continuous-time limit, which elucidates the connection between our RDDPMs and
score-based generative models on manifolds. The capability of our method is
demonstrated on datasets from previous studies and on new datasets sampled from
two high-dimensional manifolds, i.e. $\mathrm{SO}(10)$ and the configuration
space of molecular system alanine dipeptide with fixed dihedral angle.

### 10. Adaptive and Robust DBSCAN with Multi-agent Reinforcement Learning

[Adaptive and Robust DBSCAN with Multi-agent Reinforcement Learning](http://arxiv.org/pdf/2505.04339v1)

Authors: Hao Peng, Xiang Huang, Shuo Sun, Ruitong Zhang, Philip S. Yu

DBSCAN, a well-known density-based clustering algorithm, has gained
widespread popularity and usage due to its effectiveness in identifying
clusters of arbitrary shapes and handling noisy data. However, it encounters
challenges in producing satisfactory cluster results when confronted with
datasets of varying density scales, a common scenario in real-world
applications. In this paper, we propose a novel Adaptive and Robust DBSCAN with
Multi-agent Reinforcement Learning cluster framework, namely AR-DBSCAN. First,
we model the initial dataset as a two-level encoding tree and categorize the
data vertices into distinct density partitions according to the information
uncertainty determined in the encoding tree. Each partition is then assigned to
an agent to find the best clustering parameters without manual assistance. The
allocation is density-adaptive, enabling AR-DBSCAN to effectively handle
diverse density distributions within the dataset by utilizing distinct agents
for different partitions. Second, a multi-agent deep reinforcement learning
guided automatic parameter searching process is designed. The process of
adjusting the parameter search direction by perceiving the clustering
environment is modeled as a Markov decision process. Using a weakly-supervised
reward training policy network, each agent adaptively learns the optimal
clustering parameters by interacting with the clusters. Third, a recursive
search mechanism adaptable to the data's scale is presented, enabling efficient
and controlled exploration of large parameter spaces. Extensive experiments are
conducted on nine artificial datasets and a real-world dataset. The results of
offline and online tasks show that AR-DBSCAN not only improves clustering
accuracy by up to 144.1% and 175.3% in the NMI and ARI metrics, respectively,
but also is capable of robustly finding dominant parameters.

### Neural and Evolutionary Computing

### 1. A New Scope and Domain Measure Comparison Method for Global Convergence Analysis in Evolutionary Computation

[A New Scope and Domain Measure Comparison Method for Global Convergence Analysis in Evolutionary Computation](http://arxiv.org/pdf/2505.04089v1)

Authors: Liu-Yue Luo, Zhi-Hui Zhan, Kay Chen Tan, Jun Zhang

Convergence analysis is a fundamental research topic in evolutionary
computation (EC). The commonly used analysis method models the EC algorithm as
a homogeneous Markov chain for analysis, which is not always suitable for
different EC variants, and also sometimes causes misuse and confusion due to
their complex process. In this article, we categorize the existing researches
on convergence analysis in EC algorithms into stable convergence and global
convergence, and then prove that the conditions for these two convergence
properties are somehow mutually exclusive. Inspired by this proof, we propose a
new scope and domain measure comparison (SDMC) method for analyzing the global
convergence of EC algorithms and provide a rigorous proof of its necessity and
sufficiency as an alternative condition. Unlike traditional methods, the SDMC
method is straightforward, bypasses Markov chain modeling, and minimizes errors
from misapplication as it only focuses on the measure of the algorithm's search
scope. We apply SDMC to two algorithm types that are unsuitable for traditional
methods, confirming its effectiveness in global convergence analysis.
Furthermore, we apply the SDMC method to explore the gene targeting mechanism's
impact on the global convergence in large-scale global optimization, deriving
insights into how to design EC algorithms that guarantee global convergence and
exploring how theoretical analysis can guide EC algorithm design.

### 2. TS-SNN: Temporal Shift Module for Spiking Neural Networks

[TS-SNN: Temporal Shift Module for Spiking Neural Networks](http://arxiv.org/pdf/2505.04165v1)

Authors: Kairong Yu, Tianqing Zhang, Qi Xu, Gang Pan, Hongwei Wang

Spiking Neural Networks (SNNs) are increasingly recognized for their
biological plausibility and energy efficiency, positioning them as strong
alternatives to Artificial Neural Networks (ANNs) in neuromorphic computing
applications. SNNs inherently process temporal information by leveraging the
precise timing of spikes, but balancing temporal feature utilization with low
energy consumption remains a challenge. In this work, we introduce Temporal
Shift module for Spiking Neural Networks (TS-SNN), which incorporates a novel
Temporal Shift (TS) module to integrate past, present, and future spike
features within a single timestep via a simple yet effective shift operation. A
residual combination method prevents information loss by integrating shifted
and original features. The TS module is lightweight, requiring only one
additional learnable parameter, and can be seamlessly integrated into existing
architectures with minimal additional computational cost. TS-SNN achieves
state-of-the-art performance on benchmarks like CIFAR-10 (96.72\%), CIFAR-100
(80.28\%), and ImageNet (70.61\%) with fewer timesteps, while maintaining low
energy consumption. This work marks a significant step forward in developing
efficient and accurate SNN architectures.

### 3. Izhikevich-Inspired Temporal Dynamics for Enhancing Privacy, Efficiency, and Transferability in Spiking Neural Networks

[Izhikevich-Inspired Temporal Dynamics for Enhancing Privacy, Efficiency, and Transferability in Spiking Neural Networks](http://arxiv.org/pdf/2505.04034v1)

Authors: Ayana Moshruba, Hamed Poursiami, Maryam Parsa

Biological neurons exhibit diverse temporal spike patterns, which are
believed to support efficient, robust, and adaptive neural information
processing. While models such as Izhikevich can replicate a wide range of these
firing dynamics, their complexity poses challenges for directly integrating
them into scalable spiking neural networks (SNN) training pipelines. In this
work, we propose two probabilistically driven, input-level temporal spike
transformations: Poisson-Burst and Delayed-Burst that introduce biologically
inspired temporal variability directly into standard Leaky Integrate-and-Fire
(LIF) neurons. This enables scalable training and systematic evaluation of how
spike timing dynamics affect privacy, generalization, and learning performance.
Poisson-Burst modulates burst occurrence based on input intensity, while
Delayed-Burst encodes input strength through burst onset timing. Through
extensive experiments across multiple benchmarks, we demonstrate that
Poisson-Burst maintains competitive accuracy and lower resource overhead while
exhibiting enhanced privacy robustness against membership inference attacks,
whereas Delayed-Burst provides stronger privacy protection at a modest accuracy
trade-off. These findings highlight the potential of biologically grounded
temporal spike dynamics in improving the privacy, generalization and biological
plausibility of neuromorphic learning systems.

### 4. TrajEvo: Designing Trajectory Prediction Heuristics via LLM-driven Evolution

[TrajEvo: Designing Trajectory Prediction Heuristics via LLM-driven Evolution](http://arxiv.org/pdf/2505.04480v1)

Authors: Zhikai Zhao, Chuanbo Hua, Federico Berto, Kanghoon Lee, Zihan Ma, Jiachen Li, Jinkyoo Park

Trajectory prediction is a crucial task in modeling human behavior,
especially in fields as social robotics and autonomous vehicle navigation.
Traditional heuristics based on handcrafted rules often lack accuracy, while
recently proposed deep learning approaches suffer from computational cost, lack
of explainability, and generalization issues that limit their practical
adoption. In this paper, we introduce TrajEvo, a framework that leverages Large
Language Models (LLMs) to automatically design trajectory prediction
heuristics. TrajEvo employs an evolutionary algorithm to generate and refine
prediction heuristics from past trajectory data. We introduce a
Cross-Generation Elite Sampling to promote population diversity and a
Statistics Feedback Loop allowing the LLM to analyze alternative predictions.
Our evaluations show TrajEvo outperforms previous heuristic methods on the
ETH-UCY datasets, and remarkably outperforms both heuristics and deep learning
methods when generalizing to the unseen SDD dataset. TrajEvo represents a first
step toward automated design of fast, explainable, and generalizable trajectory
prediction heuristics. We make our source code publicly available to foster
future research at https://github.com/ai4co/trajevo.

### 5. Spectral and Temporal Denoising for Differentially Private Optimization

[Spectral and Temporal Denoising for Differentially Private Optimization](http://arxiv.org/pdf/2505.04468v1)

Authors: Hyeju Shin, Kyudan Jung, Seongwon Yun, Juyoung Yun

This paper introduces the FFT-Enhanced Kalman Filter (FFTKF), a
differentially private optimization method that addresses the challenge of
preserving performance in DP-SGD, where added noise typically degrades model
utility. FFTKF integrates frequency-domain noise shaping with Kalman filtering
to enhance gradient quality while preserving $(\varepsilon, \delta)$-DP
guarantees. It employs a high-frequency shaping mask in the Fourier domain to
concentrate differential privacy noise in less informative spectral components,
preserving low-frequency gradient signals. A scalar-gain Kalman filter with
finite-difference Hessian approximation further refines the denoised gradients.
With a per-iteration complexity of $\mathcal{O}(d \log d)$, FFTKF demonstrates
improved test accuracy over DP-SGD and DiSK across MNIST, CIFAR-10, CIFAR-100,
and Tiny-ImageNet datasets using CNNs, Wide ResNets, and Vision Transformers.
Theoretical analysis confirms that FFTKF maintains equivalent privacy
guarantees while achieving a tighter privacy-utility trade-off through reduced
noise and controlled bias.

### Networking and Internet Architecture

### 1. Shadow Wireless Intelligence: Large Language Model-Driven Reasoning in Covert Communications

[Shadow Wireless Intelligence: Large Language Model-Driven Reasoning in Covert Communications](http://arxiv.org/pdf/2505.04068v1)

Authors: Yuanai Xie, Zhaozhi Liu, Xiao Zhang, Shihua Zhang, Rui Hou, Minrui Xu, Ruichen Zhang, Dusit Niyato

Covert Communications (CC) can secure sensitive transmissions in industrial,
military, and mission-critical applications within 6G wireless networks.
However, traditional optimization methods based on Artificial Noise (AN), power
control, and channel manipulation might not adapt to dynamic and adversarial
environments due to the high dimensionality, nonlinearity, and stringent
real-time covertness requirements. To bridge this gap, we introduce Shadow
Wireless Intelligence (SWI), which integrates the reasoning capabilities of
Large Language Models (LLMs) with retrieval-augmented generation to enable
intelligent decision-making in covert wireless systems. Specifically, we
utilize DeepSeek-R1, a mixture-of-experts-based LLM with RL-enhanced reasoning,
combined with real-time retrieval of domain-specific knowledge to improve
context accuracy and mitigate hallucinations. Our approach develops a
structured CC knowledge base, supports context-aware retrieval, and performs
semantic optimization, allowing LLMs to generate and adapt CC strategies in
real time. In a case study on optimizing AN power in a full-duplex CC scenario,
DeepSeek-R1 achieves 85% symbolic derivation accuracy and 94% correctness in
the generation of simulation code, outperforming baseline models. These results
validate SWI as a robust, interpretable, and adaptive foundation for LLM-driven
intelligent covert wireless systems in 6G networks.

### 2. Satellite-Assisted Low-Altitude Economy Networking: Concepts, Applications, and Opportunities

[Satellite-Assisted Low-Altitude Economy Networking: Concepts, Applications, and Opportunities](http://arxiv.org/pdf/2505.04098v1)

Authors: Shizhao He, Jiacheng Wang, Ying-Chang Liang, Geng Sun, Dusit Niyato

The low-altitude economy (LAE) is a new economic paradigm that leverages
low-altitude vehicles (LAVs) to perform diverse missions across diverse areas.
To support the operations of LAE, it is essential to establish LAE networks
that enable LAV management and communications.Existing studies mainly reuse
terrestrial networks to construct LAE networks. However, the limited coverage
of terrestrial networks poses challenges for serving LAVs in remote areas.
Besides, efficient LAV operations also require support such as localization and
navigation, which terrestrial networks designed for communications cannot fully
provide. Due to ubiquitous coverage and diverse functions, satellites are a
promising technology to support LAVs. Therefore, this article investigates
satellite-assisted LAE networking. First, we introduce an overview of LAE and
satellites, discussing their features, applications, and architectures. Next,
we investigate opportunities for satellites to assist LAE from aspects of
communication, control, and computation. As all assistance depends on reliable
satellite-LAV communications, we propose a satellite-assisted LAE framework to
tackle issues caused by the severe path loss and high dynamics in
satellite-assisted LAE networks.The case study demonstrates that the
distributed MIMO architecture efficiently reduces the required transmission
power and extends service duration, while the two-timescale optimization scheme
balances the performance and control signaling overheads. Specifically, the
proposed framework comprises distributed satellite MIMO, distributed LAV MIMO,
and a two-timescale optimization scheme.

### 3. Joint Task Offloading and Channel Allocation in Spatial-Temporal Dynamic for MEC Networks

[Joint Task Offloading and Channel Allocation in Spatial-Temporal Dynamic for MEC Networks](http://arxiv.org/pdf/2505.04272v1)

Authors: Tianyi Shi, Tiankui Zhang, Jonathan Loo, Rong Huang, Yapeng Wang

Computation offloading and resource allocation are critical in mobile edge
computing (MEC) systems to handle the massive and complex requirements of
applications restricted by limited resources. In a multi-user multi-server MEC
network, the mobility of terminals causes computing requests to be dynamically
distributed in space. At the same time, the non-negligible dependencies among
tasks in some specific applications impose temporal correlation constraints on
the solution as well, leading the time-adjacent tasks to experience varying
resource availability and competition from parallel counterparts. To address
such dynamic spatial-temporal characteristics as a challenge in the allocation
of communication and computation resources, we formulate a long-term
delay-energy trade-off cost minimization problem in the view of jointly
optimizing task offloading and resource allocation. We begin by designing a
priority evaluation scheme to decouple task dependencies and then develop a
grouped Knapsack problem for channel allocation considering the current data
load and channel status. Afterward, in order to meet the rapid response needs
of MEC systems, we exploit the double duel deep Q network (D3QN) to make
offloading decisions and integrate channel allocation results into the reward
as part of the dynamic environment feedback in D3QN, constituting the joint
optimization of task offloading and channel allocation. Finally, comprehensive
simulations demonstrate the performance of the proposed algorithm in the
delay-energy trade-off cost and its adaptability for various applications.

### 4. Design and Evaluation of an NDN-Based Network for Distributed Digital Twins

[Design and Evaluation of an NDN-Based Network for Distributed Digital Twins](http://arxiv.org/pdf/2505.04326v1)

Authors: Chen Chen, Zihan Jia, Ze Wang, Lin Cui, Fung Po Tso

Digital twins (DT) have received significant attention due to their numerous
benefits, such as real-time data analytics and cost reduction in production. DT
serves as a fundamental component of many applications, encompassing smart
manufacturing, intelligent vehicles, and smart cities. By using Machine
Learning (ML) and Artificial Intelligence (AI) techniques, DTs can efficiently
facilitate decision-making and productivity by simulating the status and
changes of a physical entity. To handle the massive amount of data brought by
DTs, it is challenging to achieve low response latency for data fetching over
existing IP-based networks. IP-based networks use host addresses for end-to-end
communication, making data distribution between DTs inefficient. Thus, we
propose to use DTs in a distributed manner over Named Data Networking (NDN)
networks. NDN is data-centric where data is routed based on content names,
dynamically adjusting paths to optimize latency. Popular data is cached in
network nodes, reducing data transmission and network congestion. Since data is
fetched by content names, users and mobile devices can move freely without IP
address reassignment. By using in-network caching and adaptive routing, we
reckon NDN is an ideal fit for Future G Networks in the context of Digital
Twins. We compared DTs in edge scenarios with cloud scenarios over NDN and
IP-based networks to validate our insights. Extensive simulation results show
that using DT in the edge reduces response latency by 10.2x. This position
paper represents an initial investigation into the gap in distributed DTs over
NDN, serving as an early-stage study.

### 5. Pipelining Split Learning in Multi-hop Edge Networks

[Pipelining Split Learning in Multi-hop Edge Networks](http://arxiv.org/pdf/2505.04368v1)

Authors: Wei Wei, Zheng Lin, Tao Li, Xuanheng Li, Xianhao Chen

To support large-scale model training, split learning (SL) enables multiple
edge devices/servers to share the intensive training workload. However, most
existing works on SL focus solely on two-tier model splitting. Moreover, while
some recent works have investigated the model splitting and placement problems
for multi-hop SL, these solutions fail to overcome the resource idleness issue,
resulting in significant network idle time. In this work, we propose a
pipelined SL scheme by addressing the joint optimization problem of model
splitting and placement (MSP) in multi-hop edge networks. By applying pipeline
parallelism to SL, we identify that the MSP problem can be mapped to a problem
of minimizing the weighted sum of a bottleneck cost function (min-max) and a
linear cost function (min-sum). Based on graph theory, we devise a
bottleneck-aware shortest-path algorithm to obtain the optimal solution.
Besides, given the MSP outcomes, we also derive the closed-form solution to the
micro-batch size in the pipeline. Finally, we develop an alternating
optimization algorithm of MSP and micro-batch size to solve the joint
optimization problem to minimize the end-to-end training latency. Extensive
simulations have demonstrated the significant advantages of our algorithm
compared to existing benchmarks without pipeline parallelism.

### 6. LLMs' Suitability for Network Security: A Case Study of STRIDE Threat Modeling

[LLMs' Suitability for Network Security: A Case Study of STRIDE Threat Modeling](http://arxiv.org/pdf/2505.04101v1)

Authors: AbdulAziz AbdulGhaffar, Ashraf Matrawy

Artificial Intelligence (AI) is expected to be an integral part of
next-generation AI-native 6G networks. With the prevalence of AI, researchers
have identified numerous use cases of AI in network security. However, there
are almost nonexistent studies that analyze the suitability of Large Language
Models (LLMs) in network security. To fill this gap, we examine the suitability
of LLMs in network security, particularly with the case study of STRIDE threat
modeling. We utilize four prompting techniques with five LLMs to perform STRIDE
classification of 5G threats. From our evaluation results, we point out key
findings and detailed insights along with the explanation of the possible
underlying factors influencing the behavior of LLMs in the modeling of certain
threats. The numerical results and the insights support the necessity for
adjusting and fine-tuning LLMs for network security use cases.

### 7. On-Device LLM for Context-Aware Wi-Fi Roaming

[On-Device LLM for Context-Aware Wi-Fi Roaming](http://arxiv.org/pdf/2505.04174v1)

Authors: Ju-Hyung Lee, Yanqing Lu

Wireless roaming is a critical yet challenging task for maintaining seamless
connectivity in dynamic mobile environments. Conventional threshold-based or
heuristic schemes often fail, leading to either sticky or excessive handovers.
We introduce the first cross-layer use of an on-device large language model
(LLM): high-level reasoning in the application layer that issues real-time
actions executed in the PHY/MAC stack. The LLM addresses two tasks: (i)
context-aware AP selection, where structured prompts fuse environmental cues
(e.g., location, time) to choose the best BSSID; and (ii) dynamic threshold
adjustment, where the model adaptively decides when to roam. To satisfy the
tight latency and resource budgets of edge hardware, we apply a suite of
optimizations-chain-of-thought prompting, parameter-efficient fine-tuning, and
quantization. Experiments on indoor and outdoor datasets show that our approach
surpasses legacy heuristics and DRL baselines, achieving a strong balance
between roaming stability and signal quality. These findings underscore the
promise of application-layer LLM reasoning for lower-layer wireless control in
future edge systems.

### Robotics

### 1. NAMO-LLM: Efficient Navigation Among Movable Obstacles with Large Language Model Guidance

[NAMO-LLM: Efficient Navigation Among Movable Obstacles with Large Language Model Guidance](http://arxiv.org/pdf/2505.04141v1)

Authors: Yuqing Zhang, Yiannis Kantaros

Several planners have been proposed to compute robot paths that reach desired
goal regions while avoiding obstacles. However, these methods fail when all
pathways to the goal are blocked. In such cases, the robot must reason about
how to reconfigure the environment to access task-relevant regions - a problem
known as Navigation Among Movable Objects (NAMO). While various solutions to
this problem have been developed, they often struggle to scale to highly
cluttered environments. To address this, we propose NAMO-LLM, a sampling-based
planner that searches over robot and obstacle configurations to compute
feasible plans specifying which obstacles to move, where, and in what order.
Its key novelty is a non-uniform sampling strategy guided by Large Language
Models (LLMs) biasing the tree construction toward directions more likely to
yield a solution. We show that NAMO-LLM is probabilistically complete and
demonstrate through experiments that it efficiently scales to cluttered
environments, outperforming related works in both runtime and plan quality.

### 2. SCU-Hand: Soft Conical Universal Robotic Hand for Scooping Granular Media from Containers of Various Sizes

[SCU-Hand: Soft Conical Universal Robotic Hand for Scooping Granular Media from Containers of Various Sizes](http://arxiv.org/pdf/2505.04162v1)

Authors: Tomoya Takahashi, Cristian C. Beltran-Hernandez, Yuki Kuroda, Kazutoshi Tanaka, Masashi Hamaya, Yoshitaka Ushiku

Automating small-scale experiments in materials science presents challenges
due to the heterogeneous nature of experimental setups. This study introduces
the SCU-Hand (Soft Conical Universal Robot Hand), a novel end-effector designed
to automate the task of scooping powdered samples from various container sizes
using a robotic arm. The SCU-Hand employs a flexible, conical structure that
adapts to different container geometries through deformation, maintaining
consistent contact without complex force sensing or machine learning-based
control methods. Its reconfigurable mechanism allows for size adjustment,
enabling efficient scooping from diverse container types. By combining soft
robotics principles with a sheet-morphing design, our end-effector achieves
high flexibility while retaining the necessary stiffness for effective powder
manipulation. We detail the design principles, fabrication process, and
experimental validation of the SCU-Hand. Experimental validation showed that
the scooping capacity is about 20% higher than that of a commercial tool, with
a scooping performance of more than 95% for containers of sizes between 67 mm
to 110 mm. This research contributes to laboratory automation by offering a
cost-effective, easily implementable solution for automating tasks such as
materials synthesis and characterization processes.

### 3. Low Resolution Next Best View for Robot Packing

[Low Resolution Next Best View for Robot Packing](http://arxiv.org/pdf/2505.04228v1)

Authors: Giuseppe Fabio Preziosa, Chiara Castellano, Andrea Maria Zanchettin, Marco Faroni, Paolo Rocco

Automating the packing of objects with robots is a key challenge in
industrial automation, where efficient object perception plays a fundamental
role. This paper focuses on scenarios where precise 3D reconstruction is not
required, prioritizing cost-effective and scalable solutions. The proposed
Low-Resolution Next Best View (LR-NBV) algorithm leverages a utility function
that balances pose redundancy and acquisition density, ensuring efficient
object reconstruction. Experimental validation demonstrates that LR-NBV
consistently outperforms standard NBV approaches, achieving comparable accuracy
with significantly fewer poses. This method proves highly suitable for
applications requiring efficiency, scalability, and adaptability without
relying on high-precision sensing.

### 4. Automating Box Folding: Sequence Extraction and Ranking Methodologies

[Automating Box Folding: Sequence Extraction and Ranking Methodologies](http://arxiv.org/pdf/2505.04257v1)

Authors: Giuseppe Fabio Preziosa, Davide Ferloni, Andrea Maria Zanchettin, Marco Faroni, Paolo Rocco

Box folding represents a crucial challenge for automated packaging systems.
This work bridges the gap between existing methods for folding sequence
extraction and approaches focused on the adaptability of automated systems to
specific box types. An innovative method is proposed to identify and rank
folding sequences, enabling the transformation of a box from an initial state
to a desired final configuration. The system evaluates and ranks these
sequences based on their feasibility and compatibility with available hardware,
providing recommendations for real-world implementations. Finally, an
illustrative use case is presented, where a robot performs the folding of a
box.

### 5. Do We Still Need to Work on Odometry for Autonomous Driving?

[Do We Still Need to Work on Odometry for Autonomous Driving?](http://arxiv.org/pdf/2505.04438v1)

Authors: Cedric Le Gentil, Daniil Lisus, Timothy D. Barfoot

Over the past decades, a tremendous amount of work has addressed the topic of
ego-motion estimation of moving platforms based on various proprioceptive and
exteroceptive sensors. At the cost of ever-increasing computational load and
sensor complexity, odometry algorithms have reached impressive levels of
accuracy with minimal drift in various conditions. In this paper, we question
the need for more research on odometry for autonomous driving by assessing the
accuracy of one of the simplest algorithms: the direct integration of wheel
encoder data and yaw rate measurements from a gyroscope. We denote this
algorithm as Odometer-Gyroscope (OG) odometry. This work shows that OG odometry
can outperform current state-of-the-art radar-inertial SE(2) odometry for a
fraction of the computational cost in most scenarios. For example, the OG
odometry is on top of the Boreas leaderboard with a relative translation error
of 0.20%, while the second-best method displays an error of 0.26%.
Lidar-inertial approaches can provide more accurate estimates, but the
computational load is three orders of magnitude higher than the OG odometry. To
further the analysis, we have pushed the limits of the OG odometry by purposely
violating its fundamental no-slip assumption using data collected during a
heavy snowstorm with different driving behaviours. Our conclusion shows that a
significant amount of slippage is required to result in non-satisfactory pose
estimates from the OG odometry.

### 6. Estimating Dynamic Soft Continuum Robot States From Boundaries

[Estimating Dynamic Soft Continuum Robot States From Boundaries](http://arxiv.org/pdf/2505.04491v1)

Authors: Tongjia Zheng, Jessica Burgner-Kahrs

Accurate state estimation is essential for effective control of robots. For
soft robots, this task is particularly challenging because their states are
inherently infinite-dimensional functions due to the robots' continuous
deformability. Traditional sensing techniques, however, can only provide
discrete measurements. Recently, a dynamic state estimation method known as a
boundary observer was introduced, which leverages Cosserat rod theory to
recover all infinite-dimensional states by measuring only the velocity twist at
the robot's tip. In this work, we present a novel boundary observer that can
also recover infinite-dimensional dynamic states, but instead relies on
measuring the internal wrench at the robot's base. This design exploits the
duality between the velocity twist at the tip and the internal wrench at the
base, with both types of boundary observers being inspired by principles of
energy dissipation. Despite the mathematical duality, the proposed approach
offers a distinct advantage: it requires only a 6-axis force/torque sensor
embedded at the base, eliminating the need for external sensing systems such as
motion capture cameras. Moreover, combining both tip- and base-based techniques
enhances energy dissipation, accelerates convergence, and improves estimation
accuracy. We validate the proposed algorithms through both simulation studies
and experiments based on tendon-driven continuum robots. Our results
demonstrate that all boundary observers converge to the ground truth within 3
seconds, even with significantly deviated initial conditions. Furthermore, they
recover from unknown perturbations and effectively track high-frequency
vibrations. We also show that combining the dual techniques further improves
convergence speed and accuracy. Finally, the computational efficiency of these
algorithms indicates their feasibility for real-time state estimation.

### 7. Hierarchical Task Decomposition for Execution Monitoring and Error Recovery: Understanding the Rationale Behind Task Demonstrations

[Hierarchical Task Decomposition for Execution Monitoring and Error Recovery: Understanding the Rationale Behind Task Demonstrations](http://arxiv.org/pdf/2505.04565v1)

Authors: Christoph Willibald, Dongheui Lee

Multi-step manipulation tasks where robots interact with their environment
and must apply process forces based on the perceived situation remain
challenging to learn and prone to execution errors. Accurately simulating these
tasks is also difficult. Hence, it is crucial for robust task performance to
learn how to coordinate end-effector pose and applied force, monitor execution,
and react to deviations. To address these challenges, we propose a learning
approach that directly infers both low- and high-level task representations
from user demonstrations on the real system. We developed an unsupervised task
segmentation algorithm that combines intention recognition and feature
clustering to infer the skills of a task. We leverage the inferred
characteristic features of each skill in a novel unsupervised anomaly detection
approach to identify deviations from the intended task execution. Together,
these components form a comprehensive framework capable of incrementally
learning task decisions and new behaviors as new situations arise. Compared to
state-of-the-art learning techniques, our approach significantly reduces the
required amount of training data and computational complexity while efficiently
learning complex in-contact behaviors and recovery strategies. Our proposed
task segmentation and anomaly detection approaches outperform state-of-the-art
methods on force-based tasks evaluated on two different robotic systems.

### 8. Stow: Robotic Packing of Items into Fabric Pods

[Stow: Robotic Packing of Items into Fabric Pods](http://arxiv.org/pdf/2505.04572v1)

Authors: Nicolas Hudson, Josh Hooks, Rahul Warrier, Curt Salisbury, Ross Hartley, Kislay Kumar, Bhavana Chandrashekhar, Paul Birkmeyer, Bosch Tang, Matt Frost, Shantanu Thakar, Tony Piaskowy, Petter Nilsson, Josh Petersen, Neel Doshi, Alan Slatter, Ankit Bhatia, Cassie Meeker, Yuechuan Xue, Dylan Cox, Alex Kyriazis, Bai Lou, Nadeem Hasan, Asif Rana, Nikhil Chacko, Ruinian Xu, Siamak Faal, Esi Seraj, Mudit Agrawal, Kevin Jamieson, Alessio Bisagni, Valerie Samzun, Christine Fuller, Alex Keklak, Alex Frenkel, Lillian Ratliff, Aaron Parness

This paper presents a compliant manipulation system capable of placing items
onto densely packed shelves. The wide diversity of items and strict business
requirements for high producing rates and low defect generation have prohibited
warehouse robotics from performing this task. Our innovations in hardware,
perception, decision-making, motion planning, and control have enabled this
system to perform over 500,000 stows in a large e-commerce fulfillment center.
The system achieves human levels of packing density and speed while
prioritizing work on overhead shelves to enhance the safety of humans working
alongside the robots.

### 9. Scalable Aerial GNSS Localization for Marine Robots

[Scalable Aerial GNSS Localization for Marine Robots](http://arxiv.org/pdf/2505.04095v1)

Authors: Shuo Wen, Edwin Meriaux, Mariana Sosa Guzmán, Charlotte Morissette, Chloe Si, Bobak Baghi, Gregory Dudek

Accurate localization is crucial for water robotics, yet traditional onboard
Global Navigation Satellite System (GNSS) approaches are difficult or
ineffective due to signal reflection on the water's surface and its high cost
of aquatic GNSS receivers. Existing approaches, such as inertial navigation,
Doppler Velocity Loggers (DVL), SLAM, and acoustic-based methods, face
challenges like error accumulation and high computational complexity.
Therefore, a more efficient and scalable solution remains necessary. This paper
proposes an alternative approach that leverages an aerial drone equipped with
GNSS localization to track and localize a marine robot once it is near the
surface of the water. Our results show that this novel adaptation enables
accurate single and multi-robot marine robot localization.

### 10. Trajectory Entropy Reinforcement Learning for Predictable and Robust Control

[Trajectory Entropy Reinforcement Learning for Predictable and Robust Control](http://arxiv.org/pdf/2505.04193v1)

Authors: Bang You, Chenxu Wang, Huaping Liu

Simplicity is a critical inductive bias for designing data-driven
controllers, especially when robustness is important. Despite the impressive
results of deep reinforcement learning in complex control tasks, it is prone to
capturing intricate and spurious correlations between observations and actions,
leading to failure under slight perturbations to the environment. To tackle
this problem, in this work we introduce a novel inductive bias towards simple
policies in reinforcement learning. The simplicity inductive bias is introduced
by minimizing the entropy of entire action trajectories, corresponding to the
number of bits required to describe information in action trajectories after
the agent observes state trajectories. Our reinforcement learning agent,
Trajectory Entropy Reinforcement Learning, is optimized to minimize the
trajectory entropy while maximizing rewards. We show that the trajectory
entropy can be effectively estimated by learning a variational parameterized
action prediction model, and use the prediction model to construct an
information-regularized reward function. Furthermore, we construct a practical
algorithm that enables the joint optimization of models, including the policy
and the prediction model. Experimental evaluations on several high-dimensional
locomotion tasks show that our learned policies produce more cyclical and
consistent action trajectories, and achieve superior performance, and
robustness to noise and dynamic changes than the state-of-the-art.

### Software Engineering

### 1. Racing Against the Clock: Exploring the Impact of Scheduled Deadlines on Technical Debt

[Racing Against the Clock: Exploring the Impact of Scheduled Deadlines on Technical Debt](http://arxiv.org/pdf/2505.04027v1)

Authors: Joshua Aldrich Edbert, Zadia Codabux, Roberto Verdecchia

Background: Technical Debt (TD) describes suboptimal software development
practices with long-term consequences, such as defects and vulnerabilities.
Deadlines are a leading cause of the emergence of TD in software systems. While
multiple aspects of TD have been studied, the empirical research findings on
the impact of deadlines are still inconclusive. Aims: This study investigates
the impact of scheduled deadlines on TD. It analyzes how scheduled deadlines
affect code quality, commit activities, and issues in issue-tracking systems.
Method: We analyzed eight Open Source Software (OSS) projects with regular
release schedules using SonarQube. We analyzed 12.3k commits and 371 releases
across these eight OSS projects. The study combined quantitative metrics with
qualitative analyses to comprehensively understand TD accumulation under
scheduled deadlines. Results: Our findings indicated that some projects had a
clear increase in TD as deadlines approached (with above 50% of releases having
increasing TD accumulation as deadlines approached), while others managed to
maintain roughly the same amount of TD. Analysis of commit activities and issue
tracking revealed that deadline proximity could lead to increased commit
frequency and bug-related issue creation. Conclusions: Our study highlights
that, in some cases, impending deadlines have a clear impact on TD. The
findings pinpoint the need to mitigate last-minute coding rushes and the risks
associated with deadline-driven TD accumulation.

### 2. Identification and Optimization of Redundant Code Using Large Language Models

[Identification and Optimization of Redundant Code Using Large Language Models](http://arxiv.org/pdf/2505.04040v1)

Authors: Shamse Tasnim Cynthia

Redundant code is a persistent challenge in software development that makes
systems harder to maintain, scale, and update. It adds unnecessary complexity,
hinders bug fixes, and increases technical debt. Despite their impact, removing
redundant code manually is risky and error-prone, often introducing new bugs or
missing dependencies. While studies highlight the prevalence and negative
impact of redundant code, little focus has been given to Artificial
Intelligence (AI) system codebases and the common patterns that cause
redundancy. Additionally, the reasons behind developers unintentionally
introducing redundant code remain largely unexplored. This research addresses
these gaps by leveraging large language models (LLMs) to automatically detect
and optimize redundant code in AI projects. Our research aims to identify
recurring patterns of redundancy and analyze their underlying causes, such as
outdated practices or insufficient awareness of best coding principles.
Additionally, we plan to propose an LLM agent that will facilitate the
detection and refactoring of redundancies on a large scale while preserving
original functionality. This work advances the application of AI in identifying
and optimizing redundant code, ultimately helping developers maintain cleaner,
more readable, and scalable codebases.

### 3. CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System

[CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](http://arxiv.org/pdf/2505.04254v1)

Authors: Li Hu, Guoqiang Chen, Xiuwei Shang, Shaoyin Cheng, Benlong Wu, Gangyang Li, Xu Zhu, Weiming Zhang, Nenghai Yu

With open-source projects growing in size and complexity, manual compilation
becomes tedious and error-prone, highlighting the need for automation to
improve efficiency and accuracy. However, the complexity of compilation
instruction search and error resolution makes automatic compilation
challenging. Inspired by the success of LLM-based agents in various fields, we
propose CompileAgent, the first LLM-based agent framework dedicated to
repo-level compilation. CompileAgent integrates five tools and a flow-based
agent strategy, enabling interaction with software artifacts for compilation
instruction search and error resolution. To measure the effectiveness of our
method, we design a public repo-level benchmark CompileAgentBench, and we also
design two baselines for comparison by combining two compilation-friendly
schemes. The performance on this benchmark shows that our method significantly
improves the compilation success rate, ranging from 10% to 71%. Meanwhile, we
evaluate the performance of CompileAgent under different agent strategies and
verify the effectiveness of the flow-based strategy. Additionally, we emphasize
the scalability of CompileAgent, further expanding its application prospects.

### 4. Revolutionizing Newcomers' Onboarding Process in OSS Communities: The Future AI Mentor

[Revolutionizing Newcomers' Onboarding Process in OSS Communities: The Future AI Mentor](http://arxiv.org/pdf/2505.04277v1)

Authors: Xin Tan, Xiao Long, Yinghao Zhu, Lin Shi, Xiaoli Lian, Li Zhang

Onboarding newcomers is vital for the sustainability of open-source software
(OSS) projects. To lower barriers and increase engagement, OSS projects have
dedicated experts who provide guidance for newcomers. However, timely responses
are often hindered by experts' busy schedules. The recent rapid advancements of
AI in software engineering have brought opportunities to leverage AI as a
substitute for expert mentoring. However, the potential role of AI as a
comprehensive mentor throughout the entire onboarding process remains
unexplored. To identify design strategies of this ``AI mentor'', we applied
Design Fiction as a participatory method with 19 OSS newcomers. We investigated
their current onboarding experience and elicited 32 design strategies for
future AI mentor. Participants envisioned AI mentor being integrated into OSS
platforms like GitHub, where it could offer assistance to newcomers, such as
``recommending projects based on personalized requirements'' and ``assessing
and categorizing project issues by difficulty''. We also collected
participants' perceptions of a prototype, named ``OSSerCopilot'', that
implemented the envisioned strategies. They found the interface useful and
user-friendly, showing a willingness to use it in the future, which suggests
the design strategies are effective. Finally, in order to identify the gaps
between our design strategies and current research, we conducted a
comprehensive literature review, evaluating the extent of existing research
support for this concept. We find that research is relatively scarce in certain
areas where newcomers highly anticipate AI mentor assistance, such as
``discovering an interested project''. Our study has the potential to
revolutionize the current newcomer-expert mentorship and provides valuable
insights for researchers and tool designers aiming to develop and enhance AI
mentor systems.

### 5. How the Misuse of a Dataset Harmed Semantic Clone Detection

[How the Misuse of a Dataset Harmed Semantic Clone Detection](http://arxiv.org/pdf/2505.04311v1)

Authors: Jens Krinke, Chaiyong Ragkhitwetsagul

BigCloneBench is a well-known and widely used large-scale dataset for the
evaluation of recall of clone detection tools. It has been beneficial for
research on clone detection and has become a standard in evaluating the
performance of clone detection tools. More recently, it has also been widely
used as a dataset to evaluate machine learning approaches to semantic clone
detection or code similarity detection for functional or semantic similarity.
  This paper demonstrates that BigCloneBench is problematic to use as ground
truth for learning or evaluating semantic code similarity, and highlights the
aspects of BigCloneBench that affect the ground truth quality. A manual
investigation of a statistically significant random sample of 406 Weak
Type-3/Type-4 clone pairs revealed that 93% of them do not have a similar
functionality and are therefore mislabelled. In a literature review of 179
papers that use BigCloneBench as a dataset, we found 139 papers that used
BigCloneBench to evaluate semantic clone detection and where the results are
threatened in their validity by the mislabelling. As such, these papers often
report high F1 scores (e.g., above 0.9), which indicates overfitting to
dataset-specific artefacts rather than genuine semantic similarity detection.
  We emphasise that using BigCloneBench remains valid for the intended purpose
of evaluating syntactic or textual clone detection of Type-1, Type-2, and
Type-3 clones. We acknowledge the important contributions of BigCloneBench to
two decades of traditional clone detection research. However, the usage of
BigCloneBench beyond the intended purpose without careful consideration of its
limitations has led to misleading results and conclusions, and potentially
harmed the field of semantic clone detection.

### 6. Towards Federated Digital Twin Platforms

[Towards Federated Digital Twin Platforms](http://arxiv.org/pdf/2505.04324v1)

Authors: Mirgita Frasheri, Prasad Talasila, Vanessa Scherma

Digital Twin (DT) technology has become rather popular in recent years,
promising to optimize production processes, manage the operation of
cyber-physical systems, with an impact spanning across multiple application
domains (e.g., manufacturing, robotics, space etc.). DTs can include different
kinds of assets, e.g., models, data, which could potentially be reused across
DT projects by multiple users, directly affecting development costs, as well as
enabling collaboration and further development of these assets. To provide user
support for these purposes, dedicated DT frameworks and platforms are required,
that take into account user needs, providing the infrastructure and building
blocks for DT development and management. In this demo paper, we show how the
DT as a Service (DTaaS) platform has been extended to enable a federated
approach to DT development and management, that allows multiple users across
multiple instances of DTaaS to discover, reuse, reconfigure, and modify
existing DT assets.

### 7. Comparative Analysis of Carbon Footprint in Manual vs. LLM-Assisted Code Development

[Comparative Analysis of Carbon Footprint in Manual vs. LLM-Assisted Code Development](http://arxiv.org/pdf/2505.04521v1)

Authors: Kuen Sum Cheung, Mayuri Kaul, Gunel Jahangirova, Mohammad Reza Mousavi, Eric Zie

Large Language Models (LLM) have significantly transformed various domains,
including software development. These models assist programmers in generating
code, potentially increasing productivity and efficiency. However, the
environmental impact of utilising these AI models is substantial, given their
high energy consumption during both training and inference stages. This
research aims to compare the energy consumption of manual software development
versus an LLM-assisted approach, using Codeforces as a simulation platform for
software development. The goal is to quantify the environmental impact and
propose strategies for minimising the carbon footprint of using LLM in software
development. Our results show that the LLM-assisted code generation leads on
average to 32.72 higher carbon footprint than the manual one. Moreover, there
is a significant correlation between task complexity and the difference in the
carbon footprint of the two approaches.

### 8. OmniGIRL: A Multilingual and Multimodal Benchmark for GitHub Issue Resolution

[OmniGIRL: A Multilingual and Multimodal Benchmark for GitHub Issue Resolution](http://arxiv.org/pdf/2505.04606v1)

Authors: Lianghong Guo, Wei Tao, Runhan Jiang, Yanlin Wang, Jiachi Chen, Xilin Liu, Yuchi Ma, Mingzhi Mao, Hongyu Zhang, Zibin Zheng

The GitHub issue resolution task aims to resolve issues reported in
repositories automatically. With advances in large language models (LLMs), this
task has gained increasing attention, and several benchmarks are proposed to
evaluate the issue resolution ability of LLMs. However, existing benchmarks
have three main limitations. First, current benchmarks focus on a single
programming language, limiting the evaluation of issues from repositories
across different languages. Second, they usually cover a narrow range of
domains, which may fail to represent the diversity of real-world issues. Third,
existing benchmarks rely solely on textual information in issue descriptions,
overlooking multimodal information such as images in issues. In this paper, we
propose OmniGIRL, a GitHub Issue ResoLution benchmark that is multilingual,
multimodal, and multi-domain. OmniGIRL includes 959 task instances, which are
collected from repositories across four programming languages (i.e., Python,
JavaScript, TypeScript, and Java) and eight different domains. Our evaluation
shows that current LLMs show limited performances on OmniGIRL. Notably, the
best-performing model, GPT-4o, resolves only 8.6% of the issues. Besides, we
find that current LLMs struggle to resolve issues requiring understanding
images. The best performance is achieved by Claude-3.5-Sonnet, which resolves
only 10.5% of the issues with image information. Finally, we analyze the
reasons behind current LLMs' failure on OmniGIRL, providing insights for future
improvements.

### 9. An Empirical Study of OpenAI API Discussions on Stack Overflow

[An Empirical Study of OpenAI API Discussions on Stack Overflow](http://arxiv.org/pdf/2505.04084v1)

Authors: Xiang Chen, Jibin Wang, Chaoyang Gao, Xiaolin Ju, Zhanqi Cui

The rapid advancement of large language models (LLMs), represented by
OpenAI's GPT series, has significantly impacted various domains such as natural
language processing, software development, education, healthcare, finance, and
scientific research. However, OpenAI APIs introduce unique challenges that
differ from traditional APIs, such as the complexities of prompt engineering,
token-based cost management, non-deterministic outputs, and operation as black
boxes. To the best of our knowledge, the challenges developers encounter when
using OpenAI APIs have not been explored in previous empirical studies. To fill
this gap, we conduct the first comprehensive empirical study by analyzing 2,874
OpenAI API-related discussions from the popular Q&A forum Stack Overflow. We
first examine the popularity and difficulty of these posts. After manually
categorizing them into nine OpenAI API-related categories, we identify specific
challenges associated with each category through topic modeling analysis. Based
on our empirical findings, we finally propose actionable implications for
developers, LLM vendors, and researchers.

### 10. SolPhishHunter: Towards Detecting and Understanding Phishing on Solana

[SolPhishHunter: Towards Detecting and Understanding Phishing on Solana](http://arxiv.org/pdf/2505.04094v1)

Authors: Ziwei Li, Zigui Jiang, Ming Fang, Jiaxin Chen, Zhiying Wu, Jiajing Wu, Lun Zhang, Zibin Zheng

Solana is a rapidly evolving blockchain platform that has attracted an
increasing number of users. However, this growth has also drawn the attention
of malicious actors, with some phishers extending their reach into the Solana
ecosystem. Unlike platforms such as Ethereum, Solana has distinct designs of
accounts and transactions, leading to the emergence of new types of phishing
transactions that we term SolPhish. We define three types of SolPhish and
develop a detection tool called SolPhishHunter. Utilizing SolPhishHunter, we
detect a total of 8,058 instances of SolPhish and conduct an empirical analysis
of these detected cases. Our analysis explores the distribution and impact of
SolPhish, the characteristics of the phishers, and the relationships among
phishing gangs. Particularly, the detected SolPhish transactions have resulted
in nearly \$1.1 million in losses for victims. We report our detection results
to the community and construct SolPhishDataset, the \emph{first} Solana
phishing-related dataset in academia.

### Social and Information Networks

### 1. Appeal and Scope of Misinformation Spread by AI Agents and Humans

[Appeal and Scope of Misinformation Spread by AI Agents and Humans](http://arxiv.org/pdf/2505.04028v1)

Authors: Lynnette Hui Xian Ng, Wenqi Zhou, Kathleen M. Carley

This work examines the influence of misinformation and the role of AI agents,
called bots, on social network platforms. To quantify the impact of
misinformation, it proposes two new metrics based on attributes of tweet
engagement and user network position: Appeal, which measures the popularity of
the tweet, and Scope, which measures the potential reach of the tweet. In
addition, it analyzes 5.8 million misinformation tweets on the COVID-19 vaccine
discourse over three time periods: Pre-Vaccine, Vaccine Launch, and
Post-Vaccine. Results show that misinformation was more prevalent during the
first two periods. Human-generated misinformation tweets tend to have higher
appeal and scope compared to bot-generated ones. Tweedie regression analysis
reveals that human-generated misinformation tweets were most concerning during
Vaccine Launch week, whereas bot-generated misinformation reached its highest
appeal and scope during the Pre-Vaccine period.

### 2. Estimating Causal Effects in Networks with Cluster-Based Bandits

[Estimating Causal Effects in Networks with Cluster-Based Bandits](http://arxiv.org/pdf/2505.04200v1)

Authors: Ahmed Sayeed Faruk, Jason Sulskis, Elena Zheleva

The gold standard for estimating causal effects is randomized controlled
trial (RCT) or A/B testing where a random group of individuals from a
population of interest are given treatment and the outcome is compared to a
random group of individuals from the same population. However, A/B testing is
challenging in the presence of interference, commonly occurring in social
networks, where individuals can impact each others outcome. Moreover, A/B
testing can incur a high performance loss when one of the treatment arms has a
poor performance and the test continues to treat individuals with it.
Therefore, it is important to design a strategy that can adapt over time and
efficiently learn the total treatment effect in the network. We introduce two
cluster-based multi-armed bandit (MAB) algorithms to gradually estimate the
total treatment effect in a network while maximizing the expected reward by
making a tradeoff between exploration and exploitation. We compare the
performance of our MAB algorithms with a vanilla MAB algorithm that ignores
clusters and the corresponding RCT methods on semi-synthetic data with
simulated interference. The vanilla MAB algorithm shows higher reward-action
ratio at the cost of higher treatment effect error due to undesired spillover.
The cluster-based MAB algorithms show higher reward-action ratio compared to
their corresponding RCT methods without sacrificing much accuracy in treatment
effect estimation.

### 3. Random walks with resetting on hypergraph

[Random walks with resetting on hypergraph](http://arxiv.org/pdf/2505.04215v1)

Authors: Fei Ma, Xincheng Hu, Haobin Shi, Wei Pan, Ping Wang

Hypergraph has been selected as a powerful candidate for characterizing
higher-order networks and has received
  increasing attention in recent years. In this article, we study random walks
with resetting on hypergraph by utilizing
  spectral theory. Specifically, we derive exact expressions for some
fundamental yet key parameters, including occupation
  probability, stationary distribution, and mean first passage time, all of
which are expressed in terms of the eigenvalues
  and eigenvectors of the transition matrix. Furthermore, we provide a general
condition for determining the optimal
  reset probability and a sufficient condition for its existence. In addition,
we build up a close relationship between
  random walks with resetting on hypergraph and simple random walks.
Concretely, the eigenvalues and eigenvectors
  of the former can be precisely represented by those of the latter. More
importantly, when considering random walks,
  we abandon the traditional approach of converting hypergraph into a graph and
propose a research framework that
  preserves the intrinsic structure of hypergraph itself, which is based on
assigning proper weights to neighboring nodes.
  Through extensive experiments, we show that the new framework produces
distinct and more reliable results than
  the traditional approach in node ranking. Finally, we explore the impact of
the resetting mechanism on cover time,
  providing a potential solution for optimizing search efficiency.

### 4. From Flowers to Fascism? The Cottagecore to Tradwife Pipeline on Tumblr

[From Flowers to Fascism? The Cottagecore to Tradwife Pipeline on Tumblr](http://arxiv.org/pdf/2505.04561v1)

Authors: Oliver Mel Allen, Yi Zu, Milo Z. Trujillo, Brooke Foucault Welles

In this work we collected and analyzed social media posts to investigate
aesthetic-based radicalization where users searching for Cottagecore content
may find Tradwife content co-opted by white supremacists, white nationalists,
or other far-right extremist groups. Through quantitative analysis of over
200,000 Tumblr posts and qualitative coding of about 2,500 Tumblr posts, we did
not find evidence of a explicit radicalization. We found that problematic
Tradwife posts found in the literature may be confined to Tradwife-only spaces,
while content in the Cottagecore tag generally did not warrant extra
moderation. However, we did find evidence of a mainstreaming effect in the
overlap between the Tradwife and Cottagecore communities. In our qualitative
analysis there was more interaction between queer and Tradwife identities than
expected based on the literature, and some Tradwives even explicitly included
queer people and disavowed racism in the Tradwife community on Tumblr. This
could be genuine, but more likely it was an example of extremists re-branding
their content and following platform norms to spread ideologies that would
otherwise be rejected by Tumblr users. Additionally, through temporal analysis
we observed a change in the central tags used by Tradwives in the Cottagecore
tag pre- and post- 2021. Initially these posts focused on aesthetics and
hobbies like baking and gardening, but post-2021 the central tags focused more
on religion, traditional gender roles, and homesteading, all markers of
reactionary ideals.

### 5. Delegation and Participation in Decentralized Governance: An Epistemic View

[Delegation and Participation in Decentralized Governance: An Epistemic View](http://arxiv.org/pdf/2505.04136v1)

Authors: Jeff Strnad

We develop and apply epistemic tests to various decentralized governance
methods as well as to study the impact of participation. These tests probe the
ability to reach a correct outcome when there is one. We find that partial
abstention is a strong governance method from an epistemic standpoint compared
to alternatives such as various forms of ``transfer delegation" in which voters
explicitly transfer some or all of their voting rights to others. We make a
stronger case for multi-step transfer delegation than is present in previous
work but also demonstrate that transfer delegation has inherent epistemic
weaknesses. We show that enhanced direct participation, voters exercising their
own voting rights, can have a variety of epistemic impacts, some very negative.
We identify governance conditions under which additional direct participation
is guaranteed to do no epistemic harm and is likely to increase the probability
of making correct decisions. In light of the epistemic challenges of
voting-based decentralized governance, we consider the possible supplementary
use of prediction markets, auctions, and AI agents to improve outcomes. All
these results are significant because epistemic performance matters if entities
such as DAOs (decentralized autonomous organizations) wish to compete with
organizations that are more centralized.

### 6. A Survey on Temporal Interaction Graph Representation Learning: Progress, Challenges, and Opportunities

[A Survey on Temporal Interaction Graph Representation Learning: Progress, Challenges, and Opportunities](http://arxiv.org/pdf/2505.04461v1)

Authors: Pengfei Jiao, Hongjiang Chen, Xuan Guo, Zhidong Zhao, Dongxiao He, Di Jin

Temporal interaction graphs (TIGs), defined by sequences of timestamped
interaction events, have become ubiquitous in real-world applications due to
their capability to model complex dynamic system behaviors. As a result,
temporal interaction graph representation learning (TIGRL) has garnered
significant attention in recent years. TIGRL aims to embed nodes in TIGs into
low-dimensional representations that effectively preserve both structural and
temporal information, thereby enhancing the performance of downstream tasks
such as classification, prediction, and clustering within constantly evolving
data environments. In this paper, we begin by introducing the foundational
concepts of TIGs and emphasize the critical role of temporal dependencies. We
then propose a comprehensive taxonomy of state-of-the-art TIGRL methods,
systematically categorizing them based on the types of information utilized
during the learning process to address the unique challenges inherent to TIGs.
To facilitate further research and practical applications, we curate the source
of datasets and benchmarks, providing valuable resources for empirical
investigations. Finally, we examine key open challenges and explore promising
research directions in TIGRL, laying the groundwork for future advancements
that have the potential to shape the evolution of this field.

### Systems and Control

### 1. Scalable 49-Channel Neural Recorder with an Event-Driven Ramp ADC and PCA Compression in 28 nm CMOS

[Scalable 49-Channel Neural Recorder with an Event-Driven Ramp ADC and PCA Compression in 28 nm CMOS](http://arxiv.org/pdf/2505.04128v1)

Authors: William Lemaire, Esmaeil Ranjbar Koleibi, Maher Benhouria, Konin Koua, Jérémy Ménard, Keven Gagnon, Charles Quesnel, Louis-Philippe Gauthier, Takwa Omrani, Montassar Dridi, Mahdi Majdoub, Marwan Besrour, Sébastien Roy, Réjean Fontaine

Neural interfaces advance neuroscience research and therapeutic innovations
by accurately measuring neuronal activity. However, recording raw data from
numerous neurons results in substantial amount of data and poses challenges for
wireless transmission. While conventional neural recorders consume energy to
digitize and process the full neural signal, only a fraction of this data
carries essential spiking information. Leveraging on this signal sparsity, this
paper introduces a neural recording integrated circuit in TSMC 28nm CMOS. It
features an event-driven ramp analog-to-digital converter, and a spike
compression module based on principal component analysis. The circuit consists
of 49 channels, each occupying an on-chip area of 50 $\times$ 60 $\mu$m$^2$.
The circuit measures 1370 $\times$ 1370 $\mu$m$^2$ and consumes 534 $\mu$W.
Compression testing on a synthetic dataset demonstrated an 8.8-fold reduction
compared to raw spikes and a 328-fold reduction relative to the raw signal.
This compression approach maintained a spike sorting accuracy of 74.9%,
compared to the 79.5% accuracy obtained with the raw signal. The paper details
the architecture and performance outcomes of the neural recording circuit and
its compression module.

### 2. Impact of Grid-Forming Inverters on Protective Relays: A Perspective for Current Limiting Control Design

[Impact of Grid-Forming Inverters on Protective Relays: A Perspective for Current Limiting Control Design](http://arxiv.org/pdf/2505.04177v1)

Authors: Yifei Li, Heng Wu, Xiongfei Wang

Grid-forming (GFM) inverters can significantly alter the fault
characteristics of power systems, which challenges the proper function of
protective relays. This paper gives a holistic analysis of the interaction
between GFM inverter-based resources (IBRs) and the supervising elements in
protective relays, including directional and phase selection elements. It is
revealed that the current limiting control (CLC) that is based on the current
reference saturation method, adversely affects the performance of supervising
elements that rely on the negative-sequence quantities. In contrast, adopting
highly inductive virtual impedance in the CLC enables a reliable operation of
such elements. This finding provides insights into the design of CLC for GFM
IBRs from a protection perspective. It is further found that even with a highly
inductive virtual impedance, the altered virtual impedance dynamics introduced
by the CLC can still lead to malfunctions of the incremental quantity-based
supervising elements. These theoretical findings are corroborated by
simulations and controller hardware-in-the-loop (CHIL) tests.

### 3. Self-Calibrating Position Measurements: Applied to Imperfect Hall Sensors

[Self-Calibrating Position Measurements: Applied to Imperfect Hall Sensors](http://arxiv.org/pdf/2505.04245v1)

Authors: Max van Meer, Marijn van Noije, Koen Tiels, Enzo Evers, Lennart Blanken, Gert Witvoet, Tom Oomen

Linear Hall sensors are a cost-effective alternative to optical encoders for
measuring the rotor positions of actuators, with the main challenge being that
they exhibit position-dependent inaccuracies resulting from manufacturing
tolerances. This paper develops a data-driven calibration procedure for linear
analog Hall sensors that enables accurate online estimates of the rotor angle
without requiring expensive external encoders. The approach combines
closed-loop data collection with nonlinear identification to obtain an accurate
model of the sensor inaccuracies, which is subsequently used for online
compensation. Simulation results show that when the flux density model
structure is known, measurement errors are reduced to the sensor noise floor,
and experiments on an industrial setup demonstrate a factor of 2.6 reduction in
the root-mean-square measurement error. These results confirm that Hall sensor
inaccuracies can be calibrated even when no external encoder is available,
improving their practical applicability.

### 4. NN-Based Joint Mitigation of IQ Imbalance and PA Nonlinearity With Multiple States

[NN-Based Joint Mitigation of IQ Imbalance and PA Nonlinearity With Multiple States](http://arxiv.org/pdf/2505.04373v1)

Authors: Yundi Zhang, Wendong Cheng, Li Chen

Joint mitigation of IQ imbalance and PA nonlinearity is important for
improving the performance of radio frequency (RF) transmitters. In this paper,
we propose a new neural network (NN) model, which can be used for joint digital
pre-distortion (DPD) of non-ideal IQ modulators and PAs in a transmitter with
multiple operating states. The model is based on the methodology of multi-task
learning (MTL). In this model, the hidden layers of the main NN are shared by
all signal states, and the output layer's weights and biases are dynamically
generated by another NN. The experimental results show that the proposed model
can effectively perform joint DPD for IQ-PA systems, and it achieves better
overall performance within multiple signal states than the existing methods.

### 5. Opinion Dynamics on Signed Graphs and Graphons

[Opinion Dynamics on Signed Graphs and Graphons](http://arxiv.org/pdf/2505.04472v1)

Authors: Raoul Prisant, Federica Garin, Paolo Frasca

In this paper, we make use of graphon theory to study opinion dynamics on
large undirected networks. The opinion dynamics models that we take into
consideration allow for negative interactions between the individuals, whose
opinions can thus grow apart. We consider both the repelling and the opposing
models of negative interactions, which have been studied in the literature. We
define the repelling and the opposing dynamics on signed graphons and we show
that their initial value problem solutions exist and are unique. We then show
that, in a suitable sense, the graphon dynamics is a good approximation of the
dynamics on large graphs that converge to a graphon. This result applies to
large random graphs that are sampled according to a graphon (W-random graphs),
for which we provide a new convergence result under very general assumptions.

### 6. Integrated equilibrium model for electrified logistics and power systems

[Integrated equilibrium model for electrified logistics and power systems](http://arxiv.org/pdf/2505.04532v1)

Authors: Rui Yao, Xuhang Liu, Anna Scaglione, Shlomo Bekhor, Kenan Zhang

This paper proposes an integrated equilibrium model to characterize the
complex interactions between electrified logistics systems and electric power
delivery systems. The model consists of two major players: an electrified
logistics operator (ELO) and a power system operator (PSO). The ELO aims to
maximize its profit by strategically scheduling and routing its electric
delivery vehicles (e-trucks) for deliveries and charging, in response to the
locational marginal price (LMP) set by the PSO. The routing, delivery, and
charging behaviors of e-trucks are modeled by a perturbed utility Markov
decision process (PU-MDP) while their collective operations are optimized to
achieve the ELO's objective by designing rewards in the PU-MDP. On the other
hand, PSO optimizes the energy price by considering both the spatiotemporal
e-truck charging demand and the base electricity load. The equilibrium of the
integrated system is formulated as a fixed point, proved to exist under mild
assumptions, and solved for a case study on the Hawaii network via Anderson's
fixed-point acceleration algorithm. Along with these numerical results, this
paper provides both theoretical insights and practical guidelines to achieve
sustainable and efficient operations in modern electrified logistics and power
systems.

### 7. Consensus Seminorms and their Applications

[Consensus Seminorms and their Applications](http://arxiv.org/pdf/2505.04580v1)

Authors: Ron Ofir, Ji Liu, A. Stephen Morse, Brian D. O. Anderson

Consensus is a well-studied problem in distributed sensing, computation and
control, yet deriving useful and easily computable bounds on the rate of
convergence to consensus remains a challenge. We study the applications of
seminorms for this goal. We revisit a previously suggested family of seminorms
and correct an error made in their original presentation where it was claimed
that the a certain seminorm is equal to the well-known coefficient of
ergodicity. We then propose a wider family of seminorms which guarantee
convergence at an exponential rate of infinite products of matrices which
generalizes known results on stochastic matrices to the class of matrices whose
row sums are all equal one. Finally, we show that such seminorms cannot be used
to bound the rate of convergence of classes larger than the well-known class of
scrambling matrices, and pose several open questions for future research.

### 8. UX-aware Rate Allocation for Real-Time Media

[UX-aware Rate Allocation for Real-Time Media](http://arxiv.org/pdf/2505.04114v1)

Authors: Belal Korany, Peerapol Tinnakornsrisuphap, Saadallah Kassir, Prashanth Hande, Hyun Yong Lee, Thomas Stockhammer

Immersive communications is a key use case for 6G where applications require
reliable latency-bound media traffic at a certain data rate to deliver an
acceptable User Experience (UX) or Quality-of-Experience (QoE). The
Quality-of-Service (QoS) framework of current cellular systems (4G and 5G) and
prevalent network congestion control algorithms for latency-bound traffic like
L4S typically target network-related Key Performance Indicators (KPIs) such as
data rates and latencies. Network capacity is based on the number of users that
attain these KPIs. However, the UX of an immersive application for a given data
rate and latency is not the same across users, since it depends on other
factors such as the complexity of the media being transmitted and the encoder
format. This implies that guarantees on network KPIs do not necessarily
translate to guarantees on the UX.
  In this paper, we propose a framework in which the communication network can
provide guarantees on the UX. The framework requires application servers to
share real-time information on UX dependency on data rate to the network, which
in turn, uses this information to maximize a UX-based network utility function.
Our framework is motivated by the recent industry trends of increasing
application awareness at the network, and pushing application servers towards
the edge, allowing for tighter coordination between the servers and the 6G
system. Our simulation results show that the proposed framework substantially
improves the UX capacity of the network, which is the number of users above a
certain UX threshold, compared to conventional rate control algorithms.

### 9. Energy Efficient RSMA-Based LEO Satellite Communications Assisted by UAV-Mounted BD-Active RIS: A DRL Approach

[Energy Efficient RSMA-Based LEO Satellite Communications Assisted by UAV-Mounted BD-Active RIS: A DRL Approach](http://arxiv.org/pdf/2505.04148v1)

Authors: Rahman Saadat Yeganeh, Hamid Behroozi

This paper proposes an advanced non-terrestrial communication architecture
that integrates Rate-Splitting Multiple Access (RSMA) with a Beyond-Diagonal
Active Reconfigurable Intelligent Surface (BD-ARIS) mounted on a UAV under the
coverage of a Low Earth Orbit (LEO) satellite. The BD-ARIS adopts a
group-connected structure to enhance signal amplification and adaptability,
while RSMA enables efficient multi-user access by dividing messages into common
and private components. The system jointly optimizes satellite beamforming, UAV
positioning, power allocation, and rate-splitting ratios to maximize the
overall energy efficiency (EE). To solve the resulting non-convex and
high-dimensional problem, we employ three state-of-the-art deep reinforcement
learning (DRL) algorithms: Trust Region Policy Optimization (TRPO), Twin
Delayed Deep Deterministic Policy Gradient (TD3), and Asynchronous Advantage
Actor-Critic (A3C). Moreover, realistic models for the power consumption of
both the UAV and the BD-ARIS are considered. Simulation results reveal that
TRPO consistently achieves the best performance in terms of EE and sum rate,
especially under high transmit powers and challenging deployment scenarios. TD3
converges faster and performs competitively in moderate settings, while A3C
suffers from instability due to its high variance. Additionally, the robustness
of each algorithm under channel state information (CSI) uncertainty is
evaluated, confirming TRPO resilience to imperfect observations. Overall, the
proposed RSMA-BD-ARIS framework significantly outperforms conventional
RIS-assisted designs and provides a scalable, energy-efficient solution for 6G
and massive IoT applications in non-terrestrial networks.

### 10. Beyond Task Performance: Human Experience in Human-Robot Collaboration

[Beyond Task Performance: Human Experience in Human-Robot Collaboration](http://arxiv.org/pdf/2505.04182v1)

Authors: Sean Kille, Jan Heinrich Robens, Philipp Dahlinger, Alejandra Rodriguez-Velasquez, Simon Rothfuß, Balint Varga, Andreas Lindenmann, Gerhard Neumann, Sven Matthiesen, Andrea Kiesel, Sören Hohmann

Human interaction experience plays a crucial role in the effectiveness of
human-machine collaboration, especially as interactions in future systems
progress towards tighter physical and functional integration. While automation
design has been shown to impact task performance, its influence on human
experience metrics such as flow, sense of agency (SoA), and embodiment remains
underexplored. This study investigates how variations in automation design
affect these psychological experience measures and examines correlations
between subjective experience and physiological indicators. A user study was
conducted in a simulated wood workshop, where participants collaborated with a
lightweight robot under four automation levels. The results of the study
indicate that medium automation levels enhance flow, SoA and embodiment,
striking a balance between support and user autonomy. In contrast, higher
automation, despite optimizing task performance, diminishes perceived flow and
agency. Furthermore, we observed that grip force might be considered as a
real-time proxy of SoA, while correlations with heart rate variability were
inconclusive. The findings underscore the necessity for automation strategies
that integrate human- centric metrics, aiming to optimize both performance and
user experience in collaborative robotic systems

### Machine Learning (Statistics Category)

### 1. Theoretical Guarantees for LT-TTD: A Unified Transformer-based Architecture for Two-Level Ranking Systems

[Theoretical Guarantees for LT-TTD: A Unified Transformer-based Architecture for Two-Level Ranking Systems](http://arxiv.org/pdf/2505.04434v1)

Authors: Ayoub Abraich

Modern recommendation and search systems typically employ multi-stage ranking
architectures to efficiently handle billions of candidates. The conventional
approach uses distinct L1 (candidate retrieval) and L2 (re-ranking) models with
different optimization objectives, introducing critical limitations including
irreversible error propagation and suboptimal ranking. This paper identifies
and analyzes the fundamental limitations of this decoupled paradigm and
proposes LT-TTD (Listwise Transformer with Two-Tower Distillation), a novel
unified architecture that bridges retrieval and ranking phases. Our approach
combines the computational efficiency of two-tower models with the expressivity
of transformers in a unified listwise learning framework. We provide a
comprehensive theoretical analysis of our architecture and establish formal
guarantees regarding error propagation mitigation, ranking quality
improvements, and optimization convergence. We derive theoretical bounds
showing that LT-TTD reduces the upper limit on irretrievable relevant items by
a factor that depends on the knowledge distillation strength, and prove that
our multi-objective optimization framework achieves a provably better global
optimum than disjoint training. Additionally, we analyze the computational
complexity of our approach, demonstrating that the asymptotic complexity
remains within practical bounds for real-world applications. We also introduce
UPQE, a novel evaluation metric specifically designed for unified ranking
architectures that holistically captures retrieval quality, ranking
performance, and computational efficiency.

### 2. A Tutorial on Discriminative Clustering and Mutual Information

[A Tutorial on Discriminative Clustering and Mutual Information](http://arxiv.org/pdf/2505.04484v1)

Authors: Louis Ohl, Pierre-Alexandre Mattei, Frédéric Precioso

To cluster data is to separate samples into distinctive groups that should
ideally have some cohesive properties. Today, numerous clustering algorithms
exist, and their differences lie essentially in what can be perceived as
``cohesive properties''. Therefore, hypotheses on the nature of clusters must
be set: they can be either generative or discriminative. As the last decade
witnessed the impressive growth of deep clustering methods that involve neural
networks to handle high-dimensional data often in a discriminative manner; we
concentrate mainly on the discriminative hypotheses. In this paper, our aim is
to provide an accessible historical perspective on the evolution of
discriminative clustering methods and notably how the nature of assumptions of
the discriminative models changed over time: from decision boundaries to
invariance critics. We notably highlight how mutual information has been a
historical cornerstone of the progress of (deep) discriminative clustering
methods. We also show some known limitations of mutual information and how
discriminative clustering methods tried to circumvent those. We then discuss
the challenges that discriminative clustering faces with respect to the
selection of the number of clusters. Finally, we showcase these techniques
using the dedicated Python package, GemClus, that we have developed for
discriminative clustering.

### 3. Bayesian Estimation of Extreme Quantiles and the Exceedance Distribution for Paretian Tails

[Bayesian Estimation of Extreme Quantiles and the Exceedance Distribution for Paretian Tails](http://arxiv.org/pdf/2505.04501v1)

Authors: Douglas E. Johnston

Estimating extreme quantiles is an important task in many applications,
including financial risk management and climatology. More important than
estimating the quantile itself is to insure zero coverage error, which implies
the quantile estimate should, on average, reflect the desired probability of
exceedance. In this research, we show that for unconditional distributions
isomorphic to the exponential, a Bayesian quantile estimate results in zero
coverage error. This compares to the traditional maximum likelihood method,
where the coverage error can be significant under small sample sizes even
though the quantile estimate is unbiased. More generally, we prove a sufficient
condition for an unbiased quantile estimator to result in coverage error.
Interestingly, our results hold by virtue of using a Jeffreys prior for the
unknown parameters and is independent of the true prior. We also derive an
expression for the distribution, and moments, of future exceedances which is
vital for risk assessment. We extend our results to the conditional tail of
distributions with asymptotic Paretian tails and, in particular, those in the
Fr\'echet maximum domain of attraction. We illustrate our results using
simulations for a variety of light and heavy-tailed distributions.

### 4. PAC-Bayesian risk bounds for fully connected deep neural network with Gaussian priors

[PAC-Bayesian risk bounds for fully connected deep neural network with Gaussian priors](http://arxiv.org/pdf/2505.04341v1)

Authors: The Tien Mai

Deep neural networks (DNNs) have emerged as a powerful methodology with
significant practical successes in fields such as computer vision and natural
language processing. Recent works have demonstrated that sparsely connected
DNNs with carefully designed architectures can achieve minimax estimation rates
under classical smoothness assumptions. However, subsequent studies revealed
that simple fully connected DNNs can achieve comparable convergence rates,
challenging the necessity of sparsity. Theoretical advances in Bayesian neural
networks (BNNs) have been more fragmented. Much of those work has concentrated
on sparse networks, leaving the theoretical properties of fully connected BNNs
underexplored. In this paper, we address this gap by investigating fully
connected Bayesian DNNs with Gaussian prior using PAC-Bayes bounds. We
establish upper bounds on the prediction risk for a probabilistic deep neural
network method, showing that these bounds match (up to logarithmic factors) the
minimax-optimal rates in Besov space, for both nonparametric regression and
binary classification with logistic loss. Importantly, our results hold for a
broad class of practical activation functions that are Lipschitz continuous.

### 5. Localized Diffusion Models for High Dimensional Distributions Generation

[Localized Diffusion Models for High Dimensional Distributions Generation](http://arxiv.org/pdf/2505.04417v1)

Authors: Georg A. Gottwald, Shuigen Liu, Youssef Marzouk, Sebastian Reich, Xin T. Tong

Diffusion models are the state-of-the-art tools for various generative tasks.
However, estimating high-dimensional score functions makes them potentially
suffer from the curse of dimensionality (CoD). This underscores the importance
of better understanding and exploiting low-dimensional structure in the target
distribution. In this work, we consider locality structure, which describes
sparse dependencies between model components. Under locality structure, the
score function is effectively low-dimensional, so that it can be estimated by a
localized neural network with significantly reduced sample complexity. This
motivates the localized diffusion model, where a localized score matching loss
is used to train the score function within a localized hypothesis space. We
prove that such localization enables diffusion models to circumvent CoD, at the
price of additional localization error. Under realistic sample size scaling, we
show both theoretically and numerically that a moderate localization radius can
balance the statistical and localization error, leading to a better overall
performance. The localized structure also facilitates parallel training of
diffusion models, making it potentially more efficient for large-scale
applications.

### 6. Likelihood-Free Adaptive Bayesian Inference via Nonparametric Distribution Matching

[Likelihood-Free Adaptive Bayesian Inference via Nonparametric Distribution Matching](http://arxiv.org/pdf/2505.04603v1)

Authors: Wenhui Sophia Lu, Wing Hung Wong

When the likelihood is analytically unavailable and computationally
intractable, approximate Bayesian computation (ABC) has emerged as a widely
used methodology for approximate posterior inference; however, it suffers from
severe computational inefficiency in high-dimensional settings or under diffuse
priors. To overcome these limitations, we propose Adaptive Bayesian Inference
(ABI), a framework that bypasses traditional data-space discrepancies and
instead compares distributions directly in posterior space through
nonparametric distribution matching. By leveraging a novel Marginally-augmented
Sliced Wasserstein (MSW) distance on posterior measures and exploiting its
quantile representation, ABI transforms the challenging problem of measuring
divergence between posterior distributions into a tractable sequence of
one-dimensional conditional quantile regression tasks. Moreover, we introduce a
new adaptive rejection sampling scheme that iteratively refines the posterior
approximation by updating the proposal distribution via generative density
estimation. Theoretically, we establish parametric convergence rates for the
trimmed MSW distance and prove that the ABI posterior converges to the true
posterior as the tolerance threshold vanishes. Through extensive empirical
evaluation, we demonstrate that ABI significantly outperforms data-based
Wasserstein ABC, summary-based ABC, and state-of-the-art likelihood-free
simulators, especially in high-dimensional or dependent observation regimes.

### 7. WATCH: Weighted Adaptive Testing for Changepoint Hypotheses via Weighted-Conformal Martingales

[WATCH: Weighted Adaptive Testing for Changepoint Hypotheses via Weighted-Conformal Martingales](http://arxiv.org/pdf/2505.04608v1)

Authors: Drew Prinster, Xing Han, Anqi Liu, Suchi Saria

Responsibly deploying artificial intelligence (AI) / machine learning (ML)
systems in high-stakes settings arguably requires not only proof of system
reliability, but moreover continual, post-deployment monitoring to quickly
detect and address any unsafe behavior. Statistical methods for nonparametric
change-point detection -- especially the tools of conformal test martingales
(CTMs) and anytime-valid inference -- offer promising approaches to this
monitoring task. However, existing methods are restricted to monitoring limited
hypothesis classes or ``alarm criteria,'' such as data shifts that violate
certain exchangeability assumptions, or do not allow for online adaptation in
response to shifts. In this paper, we expand the scope of these monitoring
methods by proposing a weighted generalization of conformal test martingales
(WCTMs), which lay a theoretical foundation for online monitoring for any
unexpected changepoints in the data distribution while controlling
false-alarms. For practical applications, we propose specific WCTM algorithms
that accommodate online adaptation to mild covariate shifts (in the marginal
input distribution) while raising alarms in response to more severe shifts,
such as concept shifts (in the conditional label distribution) or extreme
(out-of-support) covariate shifts that cannot be easily adapted to. On
real-world datasets, we demonstrate improved performance relative to
state-of-the-art baselines.

### 8. From Two Sample Testing to Singular Gaussian Discrimination

[From Two Sample Testing to Singular Gaussian Discrimination](http://arxiv.org/pdf/2505.04613v1)

Authors: Leonardo V. Santoro, Kartik G. Waghmare, Victor M. Panaretos

We establish that testing for the equality of two probability measures on a
general separable and compact metric space is equivalent to testing for the
singularity between two corresponding Gaussian measures on a suitable
Reproducing Kernel Hilbert Space. The corresponding Gaussians are defined via
the notion of kernel mean and covariance embedding of a probability measure.
Discerning two singular Gaussians is fundamentally simpler from an
information-theoretic perspective than non-parametric two-sample testing,
particularly in high-dimensional settings. Our proof leverages the
Feldman-Hajek criterion for singularity/equivalence of Gaussians on Hilbert
spaces, and shows that discrepancies between distributions are heavily
magnified through their corresponding Gaussian embeddings: at a population
level, distinct probability measures lead to essentially separated Gaussian
embeddings. This appears to be a new instance of the blessing of dimensionality
that can be harnessed for the design of efficient inference tools in great
generality.

