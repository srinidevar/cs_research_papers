# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-28 17:09:49.841331 PST.

### Artificial Intelligence

### 1. [CoderAgent: Simulating Student Behavior for Personalized Programming Learning with Large Language Models](http://arxiv.org/pdf/2505.20642v1)

Authors: Yi Zhan, Qi Liu, Weibo Gao, Zheng Zhang, Tianfu Wang, Shuanghong Shen, Junyu Lu, Zhenya Huang

Personalized programming tutoring, such as exercise recommendation, can
enhance learners' efficiency, motivation, and outcomes, which is increasingly
important in modern digital education. However, the lack of sufficient and
high-quality programming data, combined with the mismatch between offline
evaluation and real-world learning, hinders the practical deployment of such
systems. To address this challenge, many approaches attempt to simulate learner
practice data, yet they often overlook the fine-grained, iterative nature of
programming learning, resulting in a lack of interpretability and granularity.
To fill this gap, we propose a LLM-based agent, CoderAgent, to simulate
students' programming processes in a fine-grained manner without relying on
real data. Specifically, we equip each human learner with an intelligent agent,
the core of which lies in capturing the cognitive states of the human
programming practice process. Inspired by ACT-R, a cognitive architecture
framework, we design the structure of CoderAgent to align with human cognitive
architecture by focusing on the mastery of programming knowledge and the
application of coding ability. Recognizing the inherent patterns in
multi-layered cognitive reasoning, we introduce the Programming Tree of Thought
(PTOT), which breaks down the process into four steps: why, how, where, and
what. This approach enables a detailed analysis of iterative problem-solving
strategies. Finally, experimental evaluations on real-world datasets
demonstrate that CoderAgent provides interpretable insights into learning
trajectories and achieves accurate simulations, paving the way for personalized
programming education.

### 2. [AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage](http://arxiv.org/pdf/2505.20662v1)

Authors: Xuanle Zhao, Zilin Sang, Yuxuan Li, Qi Shi, Shuo Wang, Duzhen Zhang, Xu Han, Zhiyuan Liu, Maosong Sun

Efficient experiment reproduction is critical to accelerating progress in
artificial intelligence. However, the inherent complexity of method design and
training procedures presents substantial challenges for automation. Notably,
reproducing experiments often requires implicit domain-specific knowledge not
explicitly documented in the original papers. To address this, we introduce the
paper lineage algorithm, which identifies and extracts implicit knowledge from
the relevant references cited by the target paper. Building on this idea, we
propose AutoReproduce, a multi-agent framework capable of automatically
reproducing experiments described in research papers in an end-to-end manner.
AutoReproduce enhances code executability by generating unit tests alongside
the reproduction process. To evaluate the reproduction capability, we construct
ReproduceBench, a benchmark annotated with verified implementations, and
introduce novel evaluation metrics to assess both the reproduction and
execution fidelity. Experimental results demonstrate that AutoReproduce
outperforms the existing strong agent baselines on all five evaluation metrics
by a peak margin of over $70\%$. In particular, compared to the official
implementations, AutoReproduce achieves an average performance gap of $22.1\%$
on $89.74\%$ of the executable experiment runs. The code will be available at
https://github.com/AI9Stars/AutoReproduce.

### 3. [MIRROR: Multi-agent Intra- and Inter-Reflection for Optimized Reasoning in Tool Learning](http://arxiv.org/pdf/2505.20670v1)

Authors: Zikang Guo, Benfeng Xu, Xiaorui Wang, Zhendong Mao

Complex tasks involving tool integration pose significant challenges for
Large Language Models (LLMs), leading to the emergence of multi-agent workflows
as a promising solution. Reflection has emerged as an effective strategy for
correcting erroneous trajectories in agentic workflows. However, existing
approaches only exploit such capability in the post-action stage, where the
agent observes the execution outcomes. We argue that, like humans, LLMs can
also engage in reflection before action execution: the agent can anticipate
undesirable outcomes from its own decisions, which not only provides a
necessarily complementary perspective to evaluate the decision but also
prevents the propagation of errors throughout the trajectory. In this paper, we
propose MIRROR, a framework that consists of both intra-reflection, which
critically assesses intended actions before execution, and inter-reflection,
which further adjusts the trajectory based on observations. This design
systematically leverages LLM reflection capabilities to eliminate and rectify
erroneous actions on a more comprehensive scope. Evaluations on both the
StableToolBench and TravelPlanner benchmarks demonstrate MIRROR's superior
performance, achieving state-of-the-art results compared to existing
approaches.

### 4. [GIFARC: Synthetic Dataset for Leveraging Human-Intuitive Analogies to Elevate AI Reasoning](http://arxiv.org/pdf/2505.20672v1)

Authors: Woochang Sim, Hyunseok Ryu, Kyungmin Choi, Sungwon Han, Sundong Kim

The Abstraction and Reasoning Corpus (ARC) poses a stringent test of general
AI capabilities, requiring solvers to infer abstract patterns from only a
handful of examples. Despite substantial progress in deep learning,
state-of-the-art models still achieve accuracy rates of merely 40-55% on 2024
ARC Competition, indicative of a significant gap between their performance and
human-level reasoning. In this work, we seek to bridge that gap by introducing
an analogy-inspired ARC dataset, GIFARC. Leveraging large language models
(LLMs) and vision-language models (VLMs), we synthesize new ARC-style tasks
from a variety of GIF images that include analogies. Each new task is paired
with ground-truth analogy, providing an explicit mapping between visual
transformations and everyday concepts. By embedding robust human-intuitive
analogies into ARC-style tasks, GIFARC guides AI agents to evaluate the task
analogically before engaging in brute-force pattern search, thus efficiently
reducing problem complexity and build a more concise and human-understandable
solution. We empirically validate that guiding LLM with analogic approach with
GIFARC affects task-solving approaches of LLMs to align with analogic approach
of human.

### 5. [Jigsaw-Puzzles: From Seeing to Understanding to Reasoning in Vision-Language Models](http://arxiv.org/pdf/2505.20728v1)

Authors: Zesen Lyu, Dandan Zhang, Wei Ye, Fangdi Li, Zhihang Jiang, Yao Yang

Spatial reasoning is a core component of human cognition, enabling
individuals to perceive, comprehend, and interact with the physical world. It
relies on a nuanced understanding of spatial structures and inter-object
relationships, serving as the foundation for complex reasoning and
decision-making. To investigate whether current vision-language models (VLMs)
exhibit similar capability, we introduce Jigsaw-Puzzles, a novel benchmark
consisting of 1,100 carefully curated real-world images with high spatial
complexity. Based on this dataset, we design five tasks to rigorously evaluate
VLMs' spatial perception, structural understanding, and reasoning capabilities,
while deliberately minimizing reliance on domain-specific knowledge to better
isolate and assess the general spatial reasoning capability. We conduct a
comprehensive evaluation across 24 state-of-the-art VLMs. The results show that
even the strongest model, Gemini-2.5-Pro, achieves only 77.14% overall accuracy
and performs particularly poorly on the Order Generation task, with only 30.00%
accuracy, far below the performance exceeding 90% achieved by human
participants. This persistent gap underscores the need for continued progress,
positioning Jigsaw-Puzzles as a challenging and diagnostic benchmark for
advancing spatial reasoning research in VLMs.

### 6. [E2E Process Automation Leveraging Generative AI and IDP-Based Automation Agent: A Case Study on Corporate Expense Processing](http://arxiv.org/pdf/2505.20733v1)

Authors: Cheonsu Jeong, Seongmin Sim, Hyoyoung Cho, Sungsu Kim, Byounggwan Shin

This paper presents an intelligent work automation approach in the context of
contemporary digital transformation by integrating generative AI and
Intelligent Document Processing (IDP) technologies with an Automation Agent to
realize End-to-End (E2E) automation of corporate financial expense processing
tasks. While traditional Robotic Process Automation (RPA) has proven effective
for repetitive, rule-based simple task automation, it faces limitations in
handling unstructured data, exception management, and complex decision-making.
This study designs and implements a four-stage integrated process comprising
automatic recognition of supporting documents such as receipts via OCR/IDP,
item classification based on a policy-driven database, intelligent exception
handling supported by generative AI (large language models, LLMs), and
human-in-the-loop final decision-making with continuous system learning through
an Automation Agent. Applied to a major Korean enterprise (Company S), the
system demonstrated quantitative benefits including over 80% reduction in
processing time for paper receipt expense tasks, decreased error rates, and
improved compliance, as well as qualitative benefits such as enhanced accuracy
and consistency, increased employee satisfaction, and data-driven decision
support. Furthermore, the system embodies a virtuous cycle by learning from
human judgments to progressively improve automatic exception handling
capabilities. Empirically, this research confirms that the organic integration
of generative AI, IDP, and Automation Agents effectively overcomes the
limitations of conventional automation and enables E2E automation of complex
corporate processes. The study also discusses potential extensions to other
domains such as accounting, human resources, and procurement, and proposes
future directions for AI-driven hyper-automation development.

### 7. [RRO: LLM Agent Optimization Through Rising Reward Trajectories](http://arxiv.org/pdf/2505.20737v1)

Authors: Zilong Wang, Jingfeng Yang, Sreyashi Nag, Samarth Varshney, Xianfeng Tang, Haoming Jiang, Jingbo Shang, Sheikh Muhammad Sarwar

Large language models (LLMs) have exhibited extraordinary performance in a
variety of tasks while it remains challenging for them to solve complex
multi-step tasks as agents. In practice, agents sensitive to the outcome of
certain key steps which makes them likely to fail the task because of a subtle
mistake in the planning trajectory. Recent approaches resort to calibrating the
reasoning process through reinforcement learning. They reward or penalize every
reasoning step with process supervision, as known as Process Reward Models
(PRMs). However, PRMs are difficult and costly to scale up with a large number
of next action candidates since they require extensive computations to acquire
the training data through the per-step trajectory exploration. To mitigate this
issue, we focus on the relative reward trend across successive reasoning steps
and propose maintaining an increasing reward in the collected trajectories for
process supervision, which we term Reward Rising Optimization (RRO).
Specifically, we incrementally augment the process supervision until
identifying a step exhibiting positive reward differentials, i.e. rising
rewards, relative to its preceding iteration. This method dynamically expands
the search space for the next action candidates, efficiently capturing
high-quality data. We provide mathematical groundings and empirical results on
the WebShop and InterCode-SQL benchmarks, showing that our proposed RRO
achieves superior performance while requiring much less exploration cost.

### 8. [MSEarth: A Benchmark for Multimodal Scientific Comprehension of Earth Science](http://arxiv.org/pdf/2505.20740v1)

Authors: Xiangyu Zhao, Wanghan Xu, Bo Liu, Yuhao Zhou, Fenghua Ling, Ben Fei, Xiaoyu Yue, Lei Bai, Wenlong Zhang, Xiao-Ming Wu

The rapid advancement of multimodal large language models (MLLMs) has
unlocked new opportunities to tackle complex scientific challenges. Despite
this progress, their application in addressing earth science problems,
especially at the graduate level, remains underexplored. A significant barrier
is the absence of benchmarks that capture the depth and contextual complexity
of geoscientific reasoning. Current benchmarks often rely on synthetic datasets
or simplistic figure-caption pairs, which do not adequately reflect the
intricate reasoning and domain-specific insights required for real-world
scientific applications. To address these gaps, we introduce MSEarth, a
multimodal scientific benchmark curated from high-quality, open-access
scientific publications. MSEarth encompasses the five major spheres of Earth
science: atmosphere, cryosphere, hydrosphere, lithosphere, and biosphere,
featuring over 7K figures with refined captions. These captions are crafted
from the original figure captions and enriched with discussions and reasoning
from the papers, ensuring the benchmark captures the nuanced reasoning and
knowledge-intensive content essential for advanced scientific tasks. MSEarth
supports a variety of tasks, including scientific figure captioning, multiple
choice questions, and open-ended reasoning challenges. By bridging the gap in
graduate-level benchmarks, MSEarth provides a scalable and high-fidelity
resource to enhance the development and evaluation of MLLMs in scientific
reasoning. The benchmark is publicly available to foster further research and
innovation in this field. Resources related to this benchmark can be found at
https://huggingface.co/MSEarth and https://github.com/xiangyu-mm/MSEarth.

### 9. [MT-Mol:Multi Agent System with Tool-based Reasoning for Molecular Optimization](http://arxiv.org/pdf/2505.20820v1)

Authors: Hyomin Kim, Yunhui Jang, Sungsoo Ahn

Large language models (LLMs) have large potential for molecular optimization,
as they can gather external chemistry tools and enable collaborative
interactions to iteratively refine molecular candidates. However, this
potential remains underexplored, particularly in the context of structured
reasoning, interpretability, and comprehensive tool-grounded molecular
optimization. To address this gap, we introduce MT-Mol, a multi-agent framework
for molecular optimization that leverages tool-guided reasoning and
role-specialized LLM agents. Our system incorporates comprehensive RDKit tools,
categorized into five distinct domains: structural descriptors, electronic and
topological features, fragment-based functional groups, molecular
representations, and miscellaneous chemical properties. Each category is
managed by an expert analyst agent, responsible for extracting task-relevant
tools and enabling interpretable, chemically grounded feedback. MT-Mol produces
molecules with tool-aligned and stepwise reasoning through the interaction
between the analyst agents, a molecule-generating scientist, a reasoning-output
verifier, and a reviewer agent. As a result, we show that our framework shows
the state-of-the-art performance of the PMO-1K benchmark on 17 out of 23 tasks.

### 10. [Step-Wise Formal Verification for LLM-Based Mathematical Problem Solving](http://arxiv.org/pdf/2505.20869v1)

Authors: Kuo Zhou, Lu Zhang

Large Language Models (LLMs) have demonstrated formidable capabilities in
solving mathematical problems, yet they may still commit logical reasoning and
computational errors during the problem-solving process. Thus, this paper
proposes a framework, MATH-VF, which includes a Formalizer and a Critic, for
formally verifying the correctness of the solutions generated by large language
models. Our framework first utilizes a Formalizer which employs an LLM to
translate a natural language solution into a formal context. Afterward, our
Critic (which integrates various external tools such as a Computer Algebra
System and an SMT solver) evaluates the correctness of each statement within
the formal context, and when a statement is incorrect, our Critic provides
corrective feedback. We empirically investigate the effectiveness of MATH-VF in
two scenarios: 1) Verification: MATH-VF is utilized to determine the
correctness of a solution to a given problem. 2) Refinement: When MATH-VF
identifies errors in the solution generated by an LLM-based solution generator
for a given problem, it submits the corrective suggestions proposed by the
Critic to the solution generator to regenerate the solution. We evaluate our
framework on widely used mathematical benchmarks: MATH500 and ProcessBench,
demonstrating the superiority of our approach over existing approaches.

### Hardware Architecture

### 1. [Static Communication Analysis for Hardware Design](http://arxiv.org/pdf/2505.20849v1)

Authors: Mads Rosendahl, Maja H. Kirkeby

Hardware acceleration of algorithms is an effective method for improving
performance in high-demand computational tasks. However, developing hardware
designs for such acceleration fundamentally differs from software development,
as it requires a deep understanding of the highly parallel nature of the
hardware architecture. In this paper, we present a framework for the static
analysis of communication within datapath architectures designed for
field-programmable gate arrays (FPGAs). Our framework aims to enhance hardware
design and optimization by providing insights into communication patterns
within the architecture, which are essential for ensuring efficient data
handling.

### 2. [SageAttention2++: A More Efficient Implementation of SageAttention2](http://arxiv.org/pdf/2505.21136v1)

Authors: Jintao Zhang, Xiaoming Xu, Jia Wei, Haofeng Huang, Pengle Zhang, Chendong Xiang, Jun Zhu, Jianfei Chen

The efficiency of attention is critical because its time complexity grows
quadratically with sequence length. SageAttention2 addresses this by utilizing
quantization to accelerate matrix multiplications (Matmul) in attention. To
further accelerate SageAttention2, we propose to utilize the faster instruction
of FP8 Matmul accumulated in FP16. The instruction is 2x faster than the FP8
Matmul used in SageAttention2. Our experiments show that SageAttention2++
achieves a 3.9x speedup over FlashAttention while maintaining the same
attention accuracy as SageAttention2. This means SageAttention2++ effectively
accelerates various models, including those for language, image, and video
generation, with negligible end-to-end metrics loss. The code will be available
at https://github.com/thu-ml/SageAttention.

### Computational Complexity

### 1. [Strong Low Degree Hardness for the Number Partitioning Problem](http://arxiv.org/pdf/2505.20607v1)

Authors: Rushil Mallarapu, Mark Sellke

In the number partitioning problem (NPP) one aims to partition a given set of
$N$ real numbers into two subsets with approximately equal sum. The NPP is a
well-studied optimization problem and is famous for possessing a
statistical-to-computational gap: when the $N$ numbers to be partitioned are
i.i.d. standard gaussian, the optimal discrepancy is $2^{-\Theta(N)}$ with high
probability, but the best known polynomial-time algorithms only find solutions
with a discrepancy of $2^{-\Theta(\log^2 N)}$. This gap is a common feature in
optimization problems over random combinatorial structures, and indicates the
need for a study that goes beyond worst-case analysis.
  We provide evidence of a nearly tight algorithmic barrier for the number
partitioning problem. Namely we consider the family of low coordinate degree
algorithms (with randomized rounding into the Boolean cube), and show that
degree $D$ algorithms fail to solve the NPP to accuracy beyond $2^{-\widetilde
O(D)}$. According to the low degree heuristic, this suggests that simple
brute-force search algorithms are nearly unimprovable, given any allotted
runtime between polynomial and exponential in $N$. Our proof combines the
isolation of solutions in the landscape with a conditional form of the overlap
gap property: given a good solution to an NPP instance, slightly noising the
NPP instance typically leaves no good solutions near the original one. In fact
our analysis applies whenever the $N$ numbers to be partitioned are independent
with uniformly bounded density.

### Computational Engineering

### 1. [GIT-BO: High-Dimensional Bayesian Optimization with Tabular Foundation Models](http://arxiv.org/pdf/2505.20685v1)

Authors: Rosen Ting-Ying Yu, Cyril Picard, Faez Ahmed

Bayesian optimization (BO) effectively optimizes expensive black-box
functions but faces significant challenges in high-dimensional spaces
(dimensions exceeding 100) due to the curse of dimensionality. Existing
high-dimensional BO methods typically leverage low-dimensional embeddings or
structural assumptions to mitigate this challenge, yet these approaches
frequently incur considerable computational overhead and rigidity due to
iterative surrogate retraining and fixed assumptions. To address these
limitations, we propose Gradient-Informed Bayesian Optimization using Tabular
Foundation Models (GIT-BO), an approach that utilizes a pre-trained tabular
foundation model (TFM) as a surrogate, leveraging its gradient information to
adaptively identify low-dimensional subspaces for optimization. We propose a
way to exploit internal gradient computations from the TFM's forward pass by
creating a gradient-informed diagnostic matrix that reveals the most sensitive
directions of the TFM's predictions, enabling optimization in a continuously
re-estimated active subspace without the need for repeated model retraining.
Extensive empirical evaluation across 23 synthetic and real-world benchmarks
demonstrates that GIT-BO consistently outperforms four state-of-the-art
Gaussian process-based high-dimensional BO methods, showing superior
scalability and optimization performances, especially as dimensionality
increases up to 500 dimensions. This work establishes foundation models,
augmented with gradient-informed adaptive subspace identification, as highly
competitive alternatives to traditional Gaussian process-based approaches for
high-dimensional Bayesian optimization tasks.

### 2. [Limitations of Nyquist Criteria in the Discretization of 2D Electromagnetic Integral Equations at High Frequency: Spectral Insights into Pollution Effects](http://arxiv.org/pdf/2505.20942v1)

Authors: Viviana Giunzioni, Adrien Merlini, Francesco P. Andriulli

The use of boundary integral equations in modeling boundary value
problems-such as elastic, acoustic, or electromagnetic ones-is well established
in the literature and widespread in practical applications. These equations are
typically solved numerically using boundary element methods (BEMs), which
generally provide accurate and reliable solutions. When the frequency of the
wave phenomenon under study increases, the discretization of the problem is
typically chosen to maintain a fixed number of unknowns per wavelength. Under
these conditions, the BEM over finite-dimensional subspaces of piecewise
polynomial basis functions is commonly believed to provide a bounded solution
accuracy. If proven, this would constitute a significant advantage of the BEM
with respect to finite element and finite difference time domain methods,
which, in contrast, are affected by numerical pollution. In this work, we
conduct a rigorous spectral analysis of some of the most commonly used boundary
integral operators and examine the impact of the BEM discretization on the
solution accuracy of widely used integral equations modeling two-dimensional
electromagnetic scattering from a perfectly electrically conducting cylinder.
We consider both ill-conditioned and well-conditioned equations, the latter
being characterized by solution operators bounded independently of frequency.
Our analysis, which is capable of tracking the effects of BEM discretization on
compositions and sums of different operators, reveals a form of pollution that
affects, in different measures, equations of both kinds. After elucidating the
mechanism by which the BEM discretization impacts accuracy, we propose a
solution strategy that can cure the pollution problem thus evidenced. The
defining strength of the proposed theoretical model lies in its capacity to
deliver deep insight into the root causes of the phenomenon.

### 3. [Out of the Past: An AI-Enabled Pipeline for Traffic Simulation from Noisy, Multimodal Detector Data and Stakeholder Feedback](http://arxiv.org/pdf/2505.21349v1)

Authors: Rex Chen, Karen Wu, John McCartney, Norman Sadeh, Fei Fang

How can a traffic simulation be designed to faithfully reflect real-world
traffic conditions? Past data-driven approaches to traffic simulation in the
literature have relied on unrealistic or suboptimal heuristics. They also fail
to adequately account for the effects of uncertainty and multimodality in the
data on simulation outcomes. In this work, we integrate advances in AI to
construct a three-step, end-to-end pipeline for generating a traffic simulation
from detector data: computer vision for vehicle counting from camera footage,
combinatorial optimization for vehicle route generation from multimodal data,
and large language models for iterative simulation refinement from natural
language feedback. Using a road network from Strongsville, Ohio as a testbed,
we demonstrate that our pipeline can accurately capture the city's traffic
patterns in a granular simulation. Beyond Strongsville, our traffic simulation
framework can be generalized to other municipalities with different levels of
data and infrastructure availability.

### 4. [Reduced and mixed precision turbulent flow simulations using explicit finite difference schemes](http://arxiv.org/pdf/2505.20911v1)

Authors: Bálint Siklósi, Pushpender K. Sharma, David J. Lusher, István Z. Reguly, Neil D. Sandham

The use of reduced and mixed precision computing has gained increasing
attention in high-performance computing (HPC) as a means to improve
computational efficiency, particularly on modern hardware architectures like
GPUs. In this work, we explore the application of mixed precision arithmetic in
compressible turbulent flow simulations using explicit finite difference
schemes. We extend the OPS and OpenSBLI frameworks to support customizable
precision levels, enabling fine-grained control over precision allocation for
different computational tasks. Through a series of numerical experiments on the
Taylor-Green vortex benchmark, we demonstrate that mixed precision strategies,
such as half-single and single-double combinations, can offer significant
performance gains without compromising numerical accuracy. However, pure
half-precision computations result in unacceptable accuracy loss, underscoring
the need for careful precision selection. Our results show that mixed precision
configurations can reduce memory usage and communication overhead, leading to
notable speedups, particularly on multi-CPU and multi-GPU systems.

### 5. [FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information](http://arxiv.org/pdf/2505.20650v1)

Authors: Yan Wang, Yang Ren, Lingfei Qian, Xueqing Peng, Keyi Wang, Yi Han, Dongji Feng, Xiao-Yang Liu, Jimin Huang, Qianqian Xie

We introduce FinTagging, the first full-scope, table-aware XBRL benchmark
designed to evaluate the structured information extraction and semantic
alignment capabilities of large language models (LLMs) in the context of
XBRL-based financial reporting. Unlike prior benchmarks that oversimplify XBRL
tagging as flat multi-class classification and focus solely on narrative text,
FinTagging decomposes the XBRL tagging problem into two subtasks: FinNI for
financial entity extraction and FinCL for taxonomy-driven concept alignment. It
requires models to jointly extract facts and align them with the full 10k+
US-GAAP taxonomy across both unstructured text and structured tables, enabling
realistic, fine-grained evaluation. We assess a diverse set of LLMs under
zero-shot settings, systematically analyzing their performance on both subtasks
and overall tagging accuracy. Our results reveal that, while LLMs demonstrate
strong generalization in information extraction, they struggle with
fine-grained concept alignment, particularly in disambiguating closely related
taxonomy entries. These findings highlight the limitations of existing LLMs in
fully automating XBRL tagging and underscore the need for improved semantic
reasoning and schema-aware modeling to meet the demands of accurate financial
disclosure. Code is available at our GitHub repository and data is at our
Hugging Face repository.

### 6. [A Lightweight Multi-Expert Generative Language Model System for Engineering Information and Knowledge Extraction](http://arxiv.org/pdf/2505.21109v1)

Authors: Bogdan Bogachov, Yaoyao Fiona Zhao

Despite recent advancements in domain adaptation techniques for large
language models, these methods remain computationally intensive, and the
resulting models can still exhibit hallucination issues. Most existing
adaptation methods do not prioritize reducing the computational resources
required for fine-tuning and inference of language models. Hallucination issues
have gradually decreased with each new model release. However, they remain
prevalent in engineering contexts, where generating well-structured text with
minimal errors and inconsistencies is critical. This work introduces a novel
approach called the Small Language Graph (SLG), which is a lightweight
adaptation solution designed to address the two key challenges outlined above.
The system is structured in the form of a graph, where each node represents a
lightweight expert - a small language model fine-tuned on specific and concise
texts. The results of this study have shown that SLG was able to surpass
conventional fine-tuning methods on the Exact Match metric by 3 times.
Additionally, the fine-tuning process was 1.7 times faster compared to that of
a larger stand-alone language model. These findings introduce a potential for
small to medium-sized engineering companies to confidently use generative AI
technologies, such as LLMs, without the necessity to invest in expensive
computational resources. Also, the graph architecture and the small size of
expert nodes offer a possible opportunity for distributed AI systems, thus
potentially diverting the global need for expensive centralized compute
clusters.

### Computational Geometry

### 1. [SOLIDGEO: Measuring Multimodal Spatial Math Reasoning in Solid Geometry](http://arxiv.org/pdf/2505.21177v1)

Authors: Peijie Wang, Chao Yang, Zhong-Zhi Li, Fei Yin, Dekang Ran, Mi Tian, Zhilong Ji, Jinfeng Bai, Cheng-Lin Liu

Geometry is a fundamental branch of mathematics and plays a crucial role in
evaluating the reasoning capabilities of multimodal large language models
(MLLMs). However, existing multimodal mathematics benchmarks mainly focus on
plane geometry and largely ignore solid geometry, which requires spatial
reasoning and is more challenging than plane geometry. To address this critical
gap, we introduce SolidGeo, the first large-scale benchmark specifically
designed to evaluate the performance of MLLMs on mathematical reasoning tasks
in solid geometry. SolidGeo consists of 3,113 real-world K-12 and
competition-level problems, each paired with visual context and annotated with
difficulty levels and fine-grained solid geometry categories. Our benchmark
covers a wide range of 3D reasoning subjects such as projection, unfolding,
spatial measurement, and spatial vector, offering a rigorous testbed for
assessing solid geometry. Through extensive experiments, we observe that MLLMs
encounter substantial challenges in solid geometry math tasks, with a
considerable performance gap relative to human capabilities on SolidGeo.
Moreover, we analyze the performance, inference efficiency and error patterns
of various models, offering insights into the solid geometric mathematical
reasoning capabilities of MLLMs. We hope SolidGeo serves as a catalyst for
advancing MLLMs toward deeper geometric reasoning and spatial intelligence.

### Computation and Language

### 1. [POLAR: A Benchmark for Multilingual, Multicultural, and Multi-Event Online Polarization](http://arxiv.org/pdf/2505.20624v1)

Authors: Usman Naseem, Juan Ren, Saba Anwar, Sarah Kohail, Rudy Alexandro Garrido Veliz, Robert Geislinger, Aisha Jabr, Idris Abdulmumin, Laiba Qureshi, Aarushi Ajay Borkar, Maryam Ibrahim Mukhtar, Abinew Ali Ayele, Ibrahim Said Ahmad, Adem Ali, Martin Semmann, Shamsuddeen Hassan Muhammad, Seid Muhie Yimam

Online polarization poses a growing challenge for democratic discourse, yet
most computational social science research remains monolingual, culturally
narrow, or event-specific. We introduce POLAR, a multilingual, multicultural,
and multievent dataset with over 23k instances in seven languages from diverse
online platforms and real-world events. Polarization is annotated along three
axes: presence, type, and manifestation, using a variety of annotation
platforms adapted to each cultural context. We conduct two main experiments:
(1) we fine-tune six multilingual pretrained language models in both
monolingual and cross-lingual setups; and (2) we evaluate a range of open and
closed large language models (LLMs) in few-shot and zero-shot scenarios.
Results show that while most models perform well on binary polarization
detection, they achieve substantially lower scores when predicting polarization
types and manifestations. These findings highlight the complex, highly
contextual nature of polarization and the need for robust, adaptable approaches
in NLP and computational social science. All resources will be released to
support further research and effective mitigation of digital polarization
globally.

### 2. [Long Context Scaling: Divide and Conquer via Multi-Agent Question-driven Collaboration](http://arxiv.org/pdf/2505.20625v1)

Authors: Sibo Xiao, Zixin Lin, Wenyang Gao, Yue Zhang

Processing long contexts has become a critical capability for modern large
language models (LLMs). Existing works leverage agent-based divide-and-conquer
methods for processing long contexts. But these methods face crucial
limitations, including prohibitive accumulated latency and amplified
information loss from excessive agent invocations, and the disruption of
inherent textual dependencies by immoderate partitioning. In this paper, we
propose a novel multi-agent framework XpandA (Expand-Agent) coupled with
question-driven workflow and dynamic partitioning for robust long-context
processing. XpandA overcomes these limitations through: 1) dynamic partitioning
of long texts, which adaptively modulates the filling rate of context windows
for input sequences of vastly varying lengths; 2) question-guided protocol to
update flat information ensembles within centralized shared memory,
constructing consistent inter-agent knowledge across partitions; and 3)
selectively replaying specific partitions based on the state-tracking of
question-information couples to promote the resolution of inverted-order
structures across partitions (e.g., flashbacks). We perform a comprehensive
evaluation of XpandA on multiple long-context benchmarks with length varying
from 1k to 1M, demonstrating XpandA's feasibility for processing ultra-long
sequences and its significant effectiveness in enhancing the long-context
capabilities of various LLMs by achieving 20\% improvements and 1.5x inference
speedup over baselines of full-context, RAG and previous agent-based methods.

### 3. [STEER-BENCH: A Benchmark for Evaluating the Steerability of Large Language Models](http://arxiv.org/pdf/2505.20645v1)

Authors: Kai Chen, Zihao He, Taiwei Shi, Kristina Lerman

Steerability, or the ability of large language models (LLMs) to adapt outputs
to align with diverse community-specific norms, perspectives, and communication
styles, is critical for real-world applications but remains under-evaluated. We
introduce Steer-Bench, a benchmark for assessing population-specific steering
using contrasting Reddit communities. Covering 30 contrasting subreddit pairs
across 19 domains, Steer-Bench includes over 10,000 instruction-response pairs
and validated 5,500 multiple-choice question with corresponding silver labels
to test alignment with diverse community norms. Our evaluation of 13 popular
LLMs using Steer-Bench reveals that while human experts achieve an accuracy of
81% with silver labels, the best-performing models reach only around 65%
accuracy depending on the domain and configuration. Some models lag behind
human-level alignment by over 15 percentage points, highlighting significant
gaps in community-sensitive steerability. Steer-Bench is a benchmark to
systematically assess how effectively LLMs understand community-specific
instructions, their resilience to adversarial steering attempts, and their
ability to accurately represent diverse cultural and ideological perspectives.

### 4. [Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge](http://arxiv.org/pdf/2505.20658v1)

Authors: Yue Fang, Zhi Jin, Jie An, Hongshen Chen, Xiaohong Chen, Naijun Zhan

Temporal Logic (TL), especially Signal Temporal Logic (STL), enables precise
formal specification, making it widely used in cyber-physical systems such as
autonomous driving and robotics. Automatically transforming NL into STL is an
attractive approach to overcome the limitations of manual transformation, which
is time-consuming and error-prone. However, due to the lack of datasets,
automatic transformation currently faces significant challenges and has not
been fully explored. In this paper, we propose an NL-STL dataset named
STL-Diversity-Enhanced (STL-DivEn), which comprises 16,000 samples enriched
with diverse patterns. To develop the dataset, we first manually create a
small-scale seed set of NL-STL pairs. Next, representative examples are
identified through clustering and used to guide large language models (LLMs) in
generating additional NL-STL pairs. Finally, diversity and accuracy are ensured
through rigorous rule-based filters and human validation. Furthermore, we
introduce the Knowledge-Guided STL Transformation (KGST) framework, a novel
approach for transforming natural language into STL, involving a
generate-then-refine process based on external knowledge. Statistical analysis
shows that the STL-DivEn dataset exhibits more diversity than the existing
NL-STL dataset. Moreover, both metric-based and human evaluations indicate that
our KGST approach outperforms baseline models in transformation accuracy on
STL-DivEn and DeepSTL datasets.

### 5. [Beyond Templates: Dynamic Adaptation of Reasoning Demonstrations via Feasibility-Aware Exploration](http://arxiv.org/pdf/2505.20700v1)

Authors: Yong Wu, Weihang Pan, Ke Li, Chen Binhui, Ping Li, Binbin Lin

Large language models (LLMs) have shown remarkable reasoning capabilities,
yet aligning such abilities to small language models (SLMs) remains a challenge
due to distributional mismatches and limited model capacity. Existing reasoning
datasets, typically designed for powerful LLMs, often lead to degraded
performance when directly applied to weaker models. In this work, we introduce
Dynamic Adaptation of Reasoning Trajectories (DART), a novel data adaptation
framework that bridges the capability gap between expert reasoning trajectories
and diverse SLMs. Instead of uniformly imitating expert steps, DART employs a
selective imitation strategy guided by step-wise adaptability estimation via
solution simulation. When expert steps surpass the student's capacity --
signaled by an Imitation Gap -- the student autonomously explores alternative
reasoning paths, constrained by outcome consistency. We validate DART across
multiple reasoning benchmarks and model scales, demonstrating that it
significantly improves generalization and data efficiency over static
fine-tuning. Our method enhances supervision quality by aligning training
signals with the student's reasoning capabilities, offering a scalable solution
for reasoning alignment in resource-constrained models.

### 6. [Silencer: From Discovery to Mitigation of Self-Bias in LLM-as-Benchmark-Generator](http://arxiv.org/pdf/2505.20738v1)

Authors: Peiwen Yuan, Yiwei Li, Shaoxiong Feng, Xinglin Wang, Yueqi Zhang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li

LLM-as-Benchmark-Generator methods have been widely studied as a supplement
to human annotators for scalable evaluation, while the potential biases within
this paradigm remain underexplored. In this work, we systematically define and
validate the phenomenon of inflated performance in models evaluated on their
self-generated benchmarks, referred to as self-bias, and attribute it to
sub-biases arising from question domain, language style, and wrong labels. On
this basis, we propose Silencer, a general framework that leverages the
heterogeneity between multiple generators at both the sample and benchmark
levels to neutralize bias and generate high-quality, self-bias-silenced
benchmark. Experimental results across various settings demonstrate that
Silencer can suppress self-bias to near zero, significantly improve evaluation
effectiveness of the generated benchmark (with an average improvement from
0.655 to 0.833 in Pearson correlation with high-quality human-annotated
benchmark), while also exhibiting strong generalizability.

### 7. [CHIMERA: A Knowledge Base of Idea Recombination in Scientific Literature](http://arxiv.org/pdf/2505.20779v1)

Authors: Noy Sternlicht, Tom Hope

A hallmark of human innovation is the process of recombination -- creating
original ideas by integrating elements of existing mechanisms and concepts. In
this work, we automatically mine the scientific literature and build CHIMERA: a
large-scale knowledge base (KB) of recombination examples. CHIMERA can be used
to empirically explore at scale how scientists recombine concepts and take
inspiration from different areas, or to train supervised machine learning
models that learn to predict new creative cross-domain directions. To build
this KB, we present a novel information extraction task of extracting
recombination from scientific paper abstracts, collect a high-quality corpus of
hundreds of manually annotated abstracts, and use it to train an LLM-based
extraction model. The model is applied to a large corpus of papers in the AI
domain, yielding a KB of over 28K recombination examples. We analyze CHIMERA to
explore the properties of recombination in different subareas of AI. Finally,
we train a scientific hypothesis generation model using the KB, which predicts
new recombination directions that real-world researchers find inspiring. Our
data and code are available at https://github.cs.huji.ac.il/tomhope-lab/CHIMERA

### 8. [Improved Representation Steering for Language Models](http://arxiv.org/pdf/2505.20809v1)

Authors: Zhengxuan Wu, Qinan Yu, Aryaman Arora, Christopher D. Manning, Christopher Potts

Steering methods for language models (LMs) seek to provide fine-grained and
interpretable control over model generations by variously changing model
inputs, weights, or representations to adjust behavior. Recent work has shown
that adjusting weights or representations is often less effective than steering
by prompting, for instance when wanting to introduce or suppress a particular
concept. We demonstrate how to improve representation steering via our new
Reference-free Preference Steering (RePS), a bidirectional
preference-optimization objective that jointly does concept steering and
suppression. We train three parameterizations of RePS and evaluate them on
AxBench, a large-scale model steering benchmark. On Gemma models with sizes
ranging from 2B to 27B, RePS outperforms all existing steering methods trained
with a language modeling objective and substantially narrows the gap with
prompting -- while promoting interpretability and minimizing parameter count.
In suppression, RePS matches the language-modeling objective on Gemma-2 and
outperforms it on the larger Gemma-3 variants while remaining resilient to
prompt-based jailbreaking attacks that defeat prompting. Overall, our results
suggest that RePS provides an interpretable and robust alternative to prompting
for both steering and suppression.

### 9. [Rethinking Information Synthesis in Multimodal Question Answering A Multi-Agent Perspective](http://arxiv.org/pdf/2505.20816v1)

Authors: Krishna Singh Rajput, Tejas Anvekar, Chitta Baral, Vivek Gupta

Recent advances in multimodal question answering have primarily focused on
combining heterogeneous modalities or fine-tuning multimodal large language
models. While these approaches have shown strong performance, they often rely
on a single, generalized reasoning strategy, overlooking the unique
characteristics of each modality ultimately limiting both accuracy and
interpretability. To address these limitations, we propose MAMMQA, a
multi-agent QA framework for multimodal inputs spanning text, tables, and
images. Our system includes two Visual Language Model (VLM) agents and one
text-based Large Language Model (LLM) agent. The first VLM decomposes the user
query into sub-questions and sequentially retrieves partial answers from each
modality. The second VLM synthesizes and refines these results through
cross-modal reasoning. Finally, the LLM integrates the insights into a cohesive
answer. This modular design enhances interpretability by making the reasoning
process transparent and allows each agent to operate within its domain of
expertise. Experiments on diverse multimodal QA benchmarks demonstrate that our
cooperative, multi-agent framework consistently outperforms existing baselines
in both accuracy and robustness.

### 10. [Tracing and Reversing Rank-One Model Edits](http://arxiv.org/pdf/2505.20819v1)

Authors: Paul Youssef, Zhixue Zhao, Christin Seifert, Jörg Schlötterer

Knowledge editing methods (KEs) are a cost-effective way to update the
factual content of large language models (LLMs), but they pose a dual-use risk.
While KEs are beneficial for updating outdated or incorrect information, they
can be exploited maliciously to implant misinformation or bias. In order to
defend against these types of malicious manipulation, we need robust techniques
that can reliably detect, interpret, and mitigate adversarial edits. This work
investigates the traceability and reversibility of knowledge edits, focusing on
the widely used Rank-One Model Editing (ROME) method. We first show that ROME
introduces distinctive distributional patterns in the edited weight matrices,
which can serve as effective signals for locating the edited weights. Second,
we show that these altered weights can reliably be used to predict the edited
factual relation, enabling partial reconstruction of the modified fact.
Building on this, we propose a method to infer the edited object entity
directly from the modified weights, without access to the editing prompt,
achieving over 95% accuracy. Finally, we demonstrate that ROME edits can be
reversed, recovering the model's original outputs with $\geq$ 80% accuracy. Our
findings highlight the feasibility of detecting, tracing, and reversing edits
based on the edited weights, offering a robust framework for safeguarding LLMs
against adversarial manipulations.

### Cryptography and Security

### 1. [Towards a DSL for hybrid secure computation](http://arxiv.org/pdf/2505.20912v1)

Authors: Romain de Laage

Fully homomorphic encryption (FHE) and trusted execution environments (TEE)
are two approaches to provide confidentiality during data processing. Each
approach has its own strengths and weaknesses. In certain scenarios,
computations can be carried out in a hybrid environment, using both FHE and
TEE. However, processing data in such hybrid settings presents challenges, as
it requires to adapt and rewrite the algorithms for the chosen technique. We
propose a domain-specific language (DSL) for secure computation that allows to
express the computations to perform and execute them using a backend that
leverages either FHE or TEE, depending on what is available.

### 2. [IRCopilot: Automated Incident Response with Large Language Models](http://arxiv.org/pdf/2505.20945v1)

Authors: Xihuan Lin, Jie Zhang, Gelei Deng, Tianzhe Liu, Xiaolong Liu, Changcai Yang, Tianwei Zhang, Qing Guo, Riqing Chen

Incident response plays a pivotal role in mitigating the impact of cyber
attacks. In recent years, the intensity and complexity of global cyber threats
have grown significantly, making it increasingly challenging for traditional
threat detection and incident response methods to operate effectively in
complex network environments. While Large Language Models (LLMs) have shown
great potential in early threat detection, their capabilities remain limited
when it comes to automated incident response after an intrusion. To address
this gap, we construct an incremental benchmark based on real-world incident
response tasks to thoroughly evaluate the performance of LLMs in this domain.
Our analysis reveals several key challenges that hinder the practical
application of contemporary LLMs, including context loss, hallucinations,
privacy protection concerns, and their limited ability to provide accurate,
context-specific recommendations. In response to these challenges, we propose
IRCopilot, a novel framework for automated incident response powered by LLMs.
IRCopilot mimics the three dynamic phases of a real-world incident response
team using four collaborative LLM-based session components. These components
are designed with clear divisions of responsibility, reducing issues such as
hallucinations and context loss. Our method leverages diverse prompt designs
and strategic responsibility segmentation, significantly improving the system's
practicality and efficiency. Experimental results demonstrate that IRCopilot
outperforms baseline LLMs across key benchmarks, achieving sub-task completion
rates of 150%, 138%, 136%, 119%, and 114% for various response tasks. Moreover,
IRCopilot exhibits robust performance on public incident response platforms and
in real-world attack scenarios, showcasing its strong applicability.

### 3. [Uncovering Black-hat SEO based fake E-commerce scam groups from their redirectors and websites](http://arxiv.org/pdf/2505.21021v1)

Authors: Makoto Shimamura, Shingo Matsugaya, Keisuke Sakai, Kosuke Takeshige, Masaki Hashimoto

While law enforcements agencies and cybercrime researchers are working hard,
fake E-commerce scam is still a big threat to Internet users. One of the major
techniques to victimize users is luring them by black-hat
search-engine-optimization (SEO); making search engines display their lure
pages as if these were placed on compromised websites and then redirecting
visitors to malicious sites. In this study, we focus on the threat actors
conduct fake E-commerce scam with this strategy. Our previous study looked at
the connection between some malware families used for black-hat SEO to
enlighten threat actors and their infrastructures, however it shows only a
limited part of the whole picture because we could not find all SEO malware
samples from limited sources. In this paper, we aim to identify and analyze
threat actor groups using a large dataset of fake E-commerce sites collected by
Japan Cybercrime Control Center, which we believe is of higher quality. It
includes 692,865 fake EC sites gathered from redirectors over two and a half
years, from May 20, 2022 to Dec. 31, 2024. We analyzed the links between these
sites using Maltego, a well-known link analysis tool, and tailored programs. We
also conducted time series analysis to track group changes in the groups.
According to the analysis, we estimate that 17 relatively large groups were
active during the dataset period and some of them were active throughout the
period.

### 4. [JavaSith: A Client-Side Framework for Analyzing Potentially Malicious Extensions in Browsers, VS Code, and NPM Packages](http://arxiv.org/pdf/2505.21263v1)

Authors: Avihay Cohen

Modern software supply chains face an increasing threat from malicious code
hidden in trusted components such as browser extensions, IDE extensions, and
open-source packages. This paper introduces JavaSith, a novel client-side
framework for analyzing potentially malicious extensions in web browsers,
Visual Studio Code (VSCode), and Node's NPM packages. JavaSith combines a
runtime sandbox that emulates browser/Node.js extension APIs (with a ``time
machine'' to accelerate time-based triggers) with static analysis and a local
large language model (LLM) to assess risk from code and metadata. We present
the design and architecture of JavaSith, including techniques for intercepting
extension behavior over simulated time and extracting suspicious patterns.
Through case studies on real-world attacks (such as a supply-chain compromise
of a Chrome extension and malicious VSCode extensions installing cryptominers),
we demonstrate how JavaSith can catch stealthy malicious behaviors that evade
traditional detection. We evaluate the framework's effectiveness and discuss
its limitations and future enhancements. JavaSith's client-side approach
empowers end-users/organizations to vet extensions and packages before
trustingly integrating them into their environments.

### 5. [Enhancing JavaScript Malware Detection through Weighted Behavioral DFAs](http://arxiv.org/pdf/2505.21406v1)

Authors: Pedro Pereira, José Gonçalves, João Vitorino, Eva Maia, Isabel Praça

This work addresses JavaScript malware detection to enhance client-side web
application security with a behavior-based system. The ability to detect
malicious JavaScript execution sequences is a critical problem in modern web
security as attack techniques become more sophisticated. This study introduces
a new system for detecting JavaScript malware using a Deterministic Finite
Automaton (DFA) along with a weighted-behavior system, which we call behavior
DFA. This system captures malicious patterns and provides a dynamic mechanism
to classify new sequences that exhibit partial similarity to known attacks,
differentiating them between benign, partially malicious, and fully malicious
behaviors. Experimental evaluation on a dataset of 1,058 sequences captured in
a real-world environment demonstrates the capability of the system to detect
and classify threats effectively, with the behavior DFA successfully
identifying exact matches and partial similarities to known malicious
behaviors. The results highlight the adaptability of the system in detecting
emerging threats while maintaining transparency in decision making.

### 6. [M3S-UPD: Efficient Multi-Stage Self-Supervised Learning for Fine-Grained Encrypted Traffic Classification with Unknown Pattern Discovery](http://arxiv.org/pdf/2505.21462v1)

Authors: Yali Yuan, Yu Huang, Xingjian Zeng, Hantao Mei, Guang Cheng

The growing complexity of encrypted network traffic presents dual challenges
for modern network management: accurate multiclass classification of known
applications and reliable detection of unknown traffic patterns. Although deep
learning models show promise in controlled environments, their real-world
deployment is hindered by data scarcity, concept drift, and operational
constraints. This paper proposes M3S-UPD, a novel Multi-Stage Self-Supervised
Unknown-aware Packet Detection framework that synergistically integrates
semi-supervised learning with representation analysis. Our approach eliminates
artificial segregation between classification and detection tasks through a
four-phase iterative process: 1) probabilistic embedding generation, 2)
clustering-based structure discovery, 3) distribution-aligned outlier
identification, and 4) confidence-aware model updating. Key innovations include
a self-supervised unknown detection mechanism that requires neither synthetic
samples nor prior knowledge, and a continuous learning architecture that is
resistant to performance degradation. Experimental results show that M3S-UPD
not only outperforms existing methods on the few-shot encrypted traffic
classification task, but also simultaneously achieves competitive performance
on the zero-shot unknown traffic discovery task.

### 7. [EarthOL: A Proof-of-Human-Contribution Consensus Protocol -- Addressing Fundamental Challenges in Decentralized Value Assessment with Enhanced Verification and Security Mechanisms](http://arxiv.org/pdf/2505.20614v1)

Authors: Jiaxiong He

This paper introduces EarthOL, a novel consensus protocol that attempts to
replace computational waste in blockchain systems with verifiable human
contributions within bounded domains. While recognizing the fundamental
impossibility of universal value assessment, we propose a domain-restricted
approach that acknowledges cultural diversity and subjective preferences while
maintaining cryptographic security. Our enhanced Proof-of-Human-Contribution
(PoHC) protocol uses a multi-layered verification system with domain-specific
evaluation criteria, time-dependent validation mechanisms, and comprehensive
security frameworks. We present theoretical analysis demonstrating meaningful
progress toward incentive-compatible human contribution verification in
high-consensus domains, achieving Byzantine fault tolerance in controlled
scenarios while addressing significant scalability and cultural bias
challenges. Through game-theoretic analysis, probabilistic modeling, and
enhanced security protocols, we identify specific conditions under which the
protocol remains stable and examine failure modes with comprehensive mitigation
strategies. This work contributes to understanding the boundaries of
decentralized value assessment and provides a framework for future research in
human-centered consensus mechanisms for specific application domains, with
particular emphasis on validator and security specialist incentive systems.

### 8. [Unveiling Impact of Frequency Components on Membership Inference Attacks for Diffusion Models](http://arxiv.org/pdf/2505.20955v1)

Authors: Puwei Lian, Yujun Cai, Songze Li

Diffusion models have achieved tremendous success in image generation, but
they also raise significant concerns regarding privacy and copyright issues.
Membership Inference Attacks (MIAs) are designed to ascertain whether specific
data were utilized during a model's training phase. As current MIAs for
diffusion models typically exploit the model's image prediction ability, we
formalize them into a unified general paradigm which computes the membership
score for membership identification. Under this paradigm, we empirically find
that existing attacks overlook the inherent deficiency in how diffusion models
process high-frequency information. Consequently, this deficiency leads to
member data with more high-frequency content being misclassified as hold-out
data, and hold-out data with less high-frequency content tend to be
misclassified as member data. Moreover, we theoretically demonstrate that this
deficiency reduces the membership advantage of attacks, thereby interfering
with the effective discrimination of member data and hold-out data. Based on
this insight, we propose a plug-and-play high-frequency filter module to
mitigate the adverse effects of the deficiency, which can be seamlessly
integrated into any attacks within this general paradigm without additional
time costs. Extensive experiments corroborate that this module significantly
improves the performance of baseline attacks across different datasets and
models.

### 9. [A Hitchhiker's Guide to Privacy-Preserving Cryptocurrencies: A Survey on Anonymity, Confidentiality, and Auditability](http://arxiv.org/pdf/2505.21008v1)

Authors: Matteo Nardelli, Francesco De Sclavis, Michela Iezzi

Cryptocurrencies and central bank digital currencies (CBDCs) are reshaping
the monetary landscape, offering transparency and efficiency while raising
critical concerns about user privacy and regulatory compliance. This survey
provides a comprehensive and technically grounded overview of
privacy-preserving digital currencies, covering both cryptocurrencies and
CBDCs. We propose a taxonomy of privacy goals -- including anonymity,
confidentiality, unlinkability, and auditability -- and map them to underlying
cryptographic primitives, protocol mechanisms, and system architectures. Unlike
previous surveys, our work adopts a design-oriented perspective, linking
high-level privacy objectives to concrete implementations. We also trace the
evolution of privacy-preserving currencies through three generations,
highlighting shifts from basic anonymity guarantees toward more nuanced
privacy-accountability trade-offs. Finally, we identify open challenges at the
intersection of cryptography, distributed systems, and policy definition, which
motivate further investigation into the primitives and design of digital
currencies that balance real-world privacy and auditability needs.

### 10. [SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](http://arxiv.org/pdf/2505.21051v1)

Authors: Jianmin Liu, Li Yan, Borui Li, Lei Yu, Chao Shen

Federated fine-tuning of large language models (LLMs) is critical for
improving their performance in handling domain-specific tasks. However, prior
work has shown that clients' private data can actually be recovered via
gradient inversion attacks. Existing privacy preservation techniques against
such attacks typically entail performance degradation and high costs, making
them ill-suited for clients with heterogeneous data distributions and device
capabilities. In this paper, we propose SHE-LoRA, which integrates selective
homomorphic encryption (HE) and low-rank adaptation (LoRA) to enable efficient
and privacy-preserving federated tuning of LLMs in cross-device environment.
Heterogeneous clients adaptively select partial model parameters for
homomorphic encryption based on parameter sensitivity assessment, with the
encryption subset obtained via negotiation. To ensure accurate model
aggregation, we design a column-aware secure aggregation method and customized
reparameterization techniques to align the aggregation results with the
heterogeneous device capabilities of clients. Extensive experiments demonstrate
that SHE-LoRA maintains performance comparable to non-private baselines,
achieves strong resistance to the state-of-the-art attacks, and significantly
reduces communication overhead by 94.901\% and encryption computation overhead
by 99.829\%, compared to baseline. Our code is accessible at
https://anonymous.4open.science/r/SHE-LoRA-8D84.

### Computer Vision and Pattern Recognition

### 1. [Mamba-Driven Topology Fusion for Monocular 3-D Human Pose Estimation](http://arxiv.org/pdf/2505.20611v1)

Authors: Zenghao Zheng, Lianping Yang, Jinshan Pan, Hegui Zhu

Transformer-based methods for 3-D human pose estimation face significant
computational challenges due to the quadratic growth of self-attention
mechanism complexity with sequence length. Recently, the Mamba model has
substantially reduced computational overhead and demonstrated outstanding
performance in modeling long sequences by leveraging state space model (SSM).
However, the ability of SSM to process sequential data is not suitable for 3-D
joint sequences with topological structures, and the causal convolution
structure in Mamba also lacks insight into local joint relationships. To
address these issues, we propose the Mamba-Driven Topology Fusion framework in
this paper. Specifically, the proposed Bone Aware Module infers the direction
and length of bone vectors in the spherical coordinate system, providing
effective topological guidance for the Mamba model in processing joint
sequences. Furthermore, we enhance the convolutional structure within the Mamba
model by integrating forward and backward graph convolutional network, enabling
it to better capture local joint dependencies. Finally, we design a
Spatiotemporal Refinement Module to model both temporal and spatial
relationships within the sequence. Through the incorporation of skeletal
topology, our approach effectively alleviates Mamba's limitations in capturing
human structural relationships. We conduct extensive experiments on the
Human3.6M and MPI-INF-3DHP datasets for testing and comparison, and the results
show that the proposed method greatly reduces computational cost while
achieving higher accuracy. Ablation studies further demonstrate the
effectiveness of each proposed module. The code and models will be released.

### 2. [OccLE: Label-Efficient 3D Semantic Occupancy Prediction](http://arxiv.org/pdf/2505.20617v1)

Authors: Naiyu Fang, Zheyuan Zhou, Fayao Liu, Xulei Yang, Jiacheng Wei, Lemiao Qiu, Guosheng Lin

3D semantic occupancy prediction offers an intuitive and efficient scene
understanding and has attracted significant interest in autonomous driving
perception. Existing approaches either rely on full supervision, which demands
costly voxel-level annotations, or on self-supervision, which provides limited
guidance and yields suboptimal performance. To address these challenges, we
propose OccLE, a Label-Efficient 3D Semantic Occupancy Prediction that takes
images and LiDAR as inputs and maintains high performance with limited voxel
annotations. Our intuition is to decouple the semantic and geometric learning
tasks and then fuse the learned feature grids from both tasks for the final
semantic occupancy prediction. Therefore, the semantic branch distills 2D
foundation model to provide aligned pseudo labels for 2D and 3D semantic
learning. The geometric branch integrates image and LiDAR inputs in cross-plane
synergy based on their inherency, employing semi-supervision to enhance
geometry learning. We fuse semantic-geometric feature grids through Dual Mamba
and incorporate a scatter-accumulated projection to supervise unannotated
prediction with aligned pseudo labels. Experiments show that OccLE achieves
competitive performance with only 10% of voxel annotations, reaching a mIoU of
16.59% on the SemanticKITTI validation set.

### 3. [ConsiStyle: Style Diversity in Training-Free Consistent T2I Generation](http://arxiv.org/pdf/2505.20626v1)

Authors: Yohai Mazuz, Janna Bruner, Lior Wolf

In text-to-image models, consistent character generation is the task of
achieving text alignment while maintaining the subject's appearance across
different prompts. However, since style and appearance are often entangled, the
existing methods struggle to preserve consistent subject characteristics while
adhering to varying style prompts. Current approaches for consistent
text-to-image generation typically rely on large-scale fine-tuning on curated
image sets or per-subject optimization, which either fail to generalize across
prompts or do not align well with textual descriptions. Meanwhile,
training-free methods often fail to maintain subject consistency across
different styles. In this work, we introduce a training-free method that
achieves both style alignment and subject consistency. The attention matrices
are manipulated such that Queries and Keys are obtained from the anchor
image(s) that are used to define the subject, while the Values are imported
from a parallel copy that is not subject-anchored. Additionally, cross-image
components are added to the self-attention mechanism by expanding the Key and
Value matrices. To do without shifting from the target style, we align the
statistics of the Value matrices. As is demonstrated in a comprehensive battery
of qualitative and quantitative experiments, our method effectively decouples
style from subject appearance and enables faithful generation of text-aligned
images with consistent characters across diverse styles.

### 4. [Open-Det: An Efficient Learning Framework for Open-Ended Detection](http://arxiv.org/pdf/2505.20639v1)

Authors: Guiping Cao, Tao Wang, Wenjian Huang, Xiangyuan Lan, Jianguo Zhang, Dongmei Jiang

Open-Ended object Detection (OED) is a novel and challenging task that
detects objects and generates their category names in a free-form manner,
without requiring additional vocabularies during inference. However, the
existing OED models, such as GenerateU, require large-scale datasets for
training, suffer from slow convergence, and exhibit limited performance. To
address these issues, we present a novel and efficient Open-Det framework,
consisting of four collaborative parts. Specifically, Open-Det accelerates
model training in both the bounding box and object name generation process by
reconstructing the Object Detector and the Object Name Generator. To bridge the
semantic gap between Vision and Language modalities, we propose a
Vision-Language Aligner with V-to-L and L-to-V alignment mechanisms,
incorporating with the Prompts Distiller to transfer knowledge from the VLM
into VL-prompts, enabling accurate object name generation for the LLM. In
addition, we design a Masked Alignment Loss to eliminate contradictory
supervision and introduce a Joint Loss to enhance classification, resulting in
more efficient training. Compared to GenerateU, Open-Det, using only 1.5% of
the training data (0.077M vs. 5.077M), 20.8% of the training epochs (31 vs.
149), and fewer GPU resources (4 V100 vs. 16 A100), achieves even higher
performance (+1.0% in APr). The source codes are available at:
https://github.com/Med-Process/Open-Det.

### 5. [IndustryEQA: Pushing the Frontiers of Embodied Question Answering in Industrial Scenarios](http://arxiv.org/pdf/2505.20640v1)

Authors: Yifan Li, Yuhang Chen, Anh Dao, Lichi Li, Zhongyi Cai, Zhen Tan, Tianlong Chen, Yu Kong

Existing Embodied Question Answering (EQA) benchmarks primarily focus on
household environments, often overlooking safety-critical aspects and reasoning
processes pertinent to industrial settings. This drawback limits the evaluation
of agent readiness for real-world industrial applications. To bridge this, we
introduce IndustryEQA, the first benchmark dedicated to evaluating embodied
agent capabilities within safety-critical warehouse scenarios. Built upon the
NVIDIA Isaac Sim platform, IndustryEQA provides high-fidelity episodic memory
videos featuring diverse industrial assets, dynamic human agents, and carefully
designed hazardous situations inspired by real-world safety guidelines. The
benchmark includes rich annotations covering six categories: equipment safety,
human safety, object recognition, attribute recognition, temporal
understanding, and spatial understanding. Besides, it also provides extra
reasoning evaluation based on these categories. Specifically, it comprises 971
question-answer pairs generated from small warehouse and 373 pairs from large
ones, incorporating scenarios with and without human. We further propose a
comprehensive evaluation framework, including various baseline models, to
assess their general perception and reasoning abilities in industrial
environments. IndustryEQA aims to steer EQA research towards developing more
robust, safety-aware, and practically applicable embodied agents for complex
industrial environments. Benchmark and codes are available.

### 6. [See through the Dark: Learning Illumination-affined Representations for Nighttime Occupancy Prediction](http://arxiv.org/pdf/2505.20641v1)

Authors: Yuan Wu, Zhiqiang Yan, Yigong Zhang, Xiang Li, ian Yang

Occupancy prediction aims to estimate the 3D spatial distribution of occupied
regions along with their corresponding semantic labels. Existing vision-based
methods perform well on daytime benchmarks but struggle in nighttime scenarios
due to limited visibility and challenging lighting conditions. To address these
challenges, we propose \textbf{LIAR}, a novel framework that learns
illumination-affined representations. LIAR first introduces Selective Low-light
Image Enhancement (SLLIE), which leverages the illumination priors from daytime
scenes to adaptively determine whether a nighttime image is genuinely dark or
sufficiently well-lit, enabling more targeted global enhancement. Building on
the illumination maps generated by SLLIE, LIAR further incorporates two
illumination-aware components: 2D Illumination-guided Sampling (2D-IGS) and 3D
Illumination-driven Projection (3D-IDP), to respectively tackle local
underexposure and overexposure. Specifically, 2D-IGS modulates feature sampling
positions according to illumination maps, assigning larger offsets to darker
regions and smaller ones to brighter regions, thereby alleviating feature
degradation in underexposed areas. Subsequently, 3D-IDP enhances semantic
understanding in overexposed regions by constructing illumination intensity
fields and supplying refined residual queries to the BEV context refinement
process. Extensive experiments on both real and synthetic datasets demonstrate
the superior performance of LIAR under challenging nighttime scenarios. The
source code and pretrained models are available
\href{https://github.com/yanzq95/LIAR}{here}.

### 7. [Scan-and-Print: Patch-level Data Summarization and Augmentation for Content-aware Layout Generation in Poster Design](http://arxiv.org/pdf/2505.20649v1)

Authors: HsiaoYuan Hsu, Yuxin Peng

In AI-empowered poster design, content-aware layout generation is crucial for
the on-image arrangement of visual-textual elements, e.g., logo, text, and
underlay. To perceive the background images, existing work demanded a high
parameter count that far exceeds the size of available training data, which has
impeded the model's real-time performance and generalization ability. To
address these challenges, we proposed a patch-level data summarization and
augmentation approach, vividly named Scan-and-Print. Specifically, the scan
procedure selects only the patches suitable for placing element vertices to
perform fine-grained perception efficiently. Then, the print procedure mixes up
the patches and vertices across two image-layout pairs to synthesize over 100%
new samples in each epoch while preserving their plausibility. Besides, to
facilitate the vertex-level operations, a vertex-based layout representation is
introduced. Extensive experimental results on widely used benchmarks
demonstrated that Scan-and-Print can generate visually appealing layouts with
state-of-the-art quality while dramatically reducing computational bottleneck
by 95.2%.

### 8. [Photography Perspective Composition: Towards Aesthetic Perspective Recommendation](http://arxiv.org/pdf/2505.20655v1)

Authors: Lujian Yao, Siming Zheng, Xinbin Yuan, Zhuoxuan Cai, Pu Wu, Jinwei Chen, Bo Li, Peng-Tao Jiang

Traditional photography composition approaches are dominated by 2D
cropping-based methods. However, these methods fall short when scenes contain
poorly arranged subjects. Professional photographers often employ perspective
adjustment as a form of 3D recomposition, modifying the projected 2D
relationships between subjects while maintaining their actual spatial positions
to achieve better compositional balance. Inspired by this artistic practice, we
propose photography perspective composition (PPC), extending beyond traditional
cropping-based methods. However, implementing the PPC faces significant
challenges: the scarcity of perspective transformation datasets and undefined
assessment criteria for perspective quality. To address these challenges, we
present three key contributions: (1) An automated framework for building PPC
datasets through expert photographs. (2) A video generation approach that
demonstrates the transformation process from suboptimal to optimal
perspectives. (3) A perspective quality assessment (PQA) model constructed
based on human performance. Our approach is concise and requires no additional
prompt instructions or camera trajectories, helping and guiding ordinary users
to enhance their composition skills.

### 9. [DriveRX: A Vision-Language Reasoning Model for Cross-Task Autonomous Driving](http://arxiv.org/pdf/2505.20665v1)

Authors: Muxi Diao, Lele Yang, Hongbo Yin, Zhexu Wang, Yejie Wang, Daxin Tian, Kongming Liang, Zhanyu Ma

Autonomous driving requires real-time, robust reasoning across perception,
prediction, planning, and behavior. However, conventional end-to-end models
fail to generalize in complex scenarios due to the lack of structured
reasoning. Recent vision-language models (VLMs) have been applied to driving
tasks, but they typically rely on isolated modules and static supervision,
limiting their ability to support multi-stage decision-making. We present
AutoDriveRL, a unified training framework that formulates autonomous driving as
a structured reasoning process over four core tasks. Each task is independently
modeled as a vision-language question-answering problem and optimized using
task-specific reward models, enabling fine-grained reinforcement signals at
different reasoning stages. Within this framework, we train DriveRX, a
cross-task reasoning VLM designed for real-time decision-making. DriveRX
achieves strong performance on a public benchmark, outperforming GPT-4o in
behavior reasoning and demonstrating robustness under complex or corrupted
driving conditions. Our analysis further highlights the impact of vision
encoder design and reward-guided reasoning compression. We will release the
AutoDriveRL framework and the DriveRX model to support future research.

### 10. [Contrastive Desensitization Learning for Cross Domain Face Forgery Detection](http://arxiv.org/pdf/2505.20675v1)

Authors: Lingyu Qiu, Ke Jiang, Xiaoyang Tan

In this paper, we propose a new cross-domain face forgery detection method
that is insensitive to different and possibly unseen forgery methods while
ensuring an acceptable low false positive rate. Although existing face forgery
detection methods are applicable to multiple domains to some degree, they often
come with a high false positive rate, which can greatly disrupt the usability
of the system. To address this issue, we propose an Contrastive Desensitization
Network (CDN) based on a robust desensitization algorithm, which captures the
essential domain characteristics through learning them from domain
transformation over pairs of genuine face images. One advantage of CDN lies in
that the learnt face representation is theoretical justified with regard to the
its robustness against the domain changes. Extensive experiments over
large-scale benchmark datasets demonstrate that our method achieves a much
lower false alarm rate with improved detection accuracy compared to several
state-of-the-art methods.

### Computers and Society

### 1. [Simulating Ethics: Using LLM Debate Panels to Model Deliberation on Medical Dilemmas](http://arxiv.org/pdf/2505.21112v1)

Authors: Hazem Zohny

This paper introduces ADEPT, a system using Large Language Model (LLM)
personas to simulate multi-perspective ethical debates. ADEPT assembles panels
of 'AI personas', each embodying a distinct ethical framework or stakeholder
perspective (like a deontologist, consequentialist, or disability rights
advocate), to deliberate on complex moral issues. Its application is
demonstrated through a scenario about prioritizing patients for a limited
number of ventilators inspired by real-world challenges in allocating scarce
medical resources. Two debates, each with six LLM personas, were conducted;
they only differed in the moral viewpoints represented: one included a Catholic
bioethicist and a care theorist, the other substituted a rule-based Kantian
philosopher and a legal adviser. Both panels ultimately favoured the same
policy -- a lottery system weighted for clinical need and fairness, crucially
avoiding the withdrawal of ventilators for reallocation. However, each panel
reached that conclusion through different lines of argument, and their voting
coalitions shifted once duty- and rights-based voices were present. Examination
of the debate transcripts shows that the altered membership redirected
attention toward moral injury, legal risk and public trust, which in turn
changed four continuing personas' final positions. The work offers three
contributions: (i) a transparent, replicable workflow for running and analysing
multi-agent AI debates in bioethics; (ii) evidence that the moral perspectives
included in such panels can materially change the outcome even when the factual
inputs remain constant; and (iii) an analysis of the implications and future
directions for such AI-mediated approaches to ethical deliberation and policy.

### 2. [EarthOL: A Proof-of-Human-Contribution Consensus Protocol -- Addressing Fundamental Challenges in Decentralized Value Assessment with Enhanced Verification and Security Mechanisms](http://arxiv.org/pdf/2505.20614v1)

Authors: Jiaxiong He

This paper introduces EarthOL, a novel consensus protocol that attempts to
replace computational waste in blockchain systems with verifiable human
contributions within bounded domains. While recognizing the fundamental
impossibility of universal value assessment, we propose a domain-restricted
approach that acknowledges cultural diversity and subjective preferences while
maintaining cryptographic security. Our enhanced Proof-of-Human-Contribution
(PoHC) protocol uses a multi-layered verification system with domain-specific
evaluation criteria, time-dependent validation mechanisms, and comprehensive
security frameworks. We present theoretical analysis demonstrating meaningful
progress toward incentive-compatible human contribution verification in
high-consensus domains, achieving Byzantine fault tolerance in controlled
scenarios while addressing significant scalability and cultural bias
challenges. Through game-theoretic analysis, probabilistic modeling, and
enhanced security protocols, we identify specific conditions under which the
protocol remains stable and examine failure modes with comprehensive mitigation
strategies. This work contributes to understanding the boundaries of
decentralized value assessment and provides a framework for future research in
human-centered consensus mechanisms for specific application domains, with
particular emphasis on validator and security specialist incentive systems.

### 3. [Institutionalizing Folk Theories of Algorithms: How Multi-Channel Networks (MCNs) Govern Algorithmic Labor in Chinese Live-Streaming Industry](http://arxiv.org/pdf/2505.20623v1)

Authors: Qing Xiao, Rongyi Chen, Jingjia Xiao, Tianyang Fu, Alice Qian Zhang, Xianzhe Fan, Bingbing Zhang, Zhicong Lu, Hong Shen

As algorithmic systems increasingly structure platform labor, workers often
rely on informal "folk theories", experience-based beliefs about how algorithms
work, to navigate opaque and unstable algorithmic environments. Prior research
has largely treated these theories as bottom-up, peer-driven strategies for
coping with algorithmic opacity and uncertainty. In this study, we shift
analytical attention to intermediary organizations and examine how folk
theories of algorithms can be institutionally constructed and operationalized
by those organizations as tools of labor management. Drawing on nine months of
ethnographic fieldwork and 37 interviews with live-streamers and staff at
Multi-Channel Networks (MCNs) in China, we show that MCNs develop and circulate
dual algorithmic theories: internally, they acknowledge the volatility of
platform systems and adopt probabilistic strategies to manage risk; externally,
they promote simplified, prescriptive theories portraying the algorithm as
transparent, fair, and responsive to individual effort. They have further
operationalize those folk theories for labor management, encouraging streamers
to self-discipline and invest in equipment, training, and routines, while
absolving MCNs of accountability. We contribute to CSCW and platform labor
literature by demonstrating how informal algorithmic knowledge, once
institutionalized, can become infrastructures of soft control -- shaping not
only how workers interpret platform algorithms, but also how their labor is
structured, moralized and governed.

### 4. [Research Community Perspectives on "Intelligence" and Large Language Models](http://arxiv.org/pdf/2505.20959v1)

Authors: Bertram Højer, Terne Sasha Thorn Jakobsen, Anna Rogers, Stefan Heinrich

Despite the widespread use of ''artificial intelligence'' (AI) framing in
Natural Language Processing (NLP) research, it is not clear what researchers
mean by ''intelligence''. To that end, we present the results of a survey on
the notion of ''intelligence'' among researchers and its role in the research
agenda. The survey elicited complete responses from 303 researchers from a
variety of fields including NLP, Machine Learning (ML), Cognitive Science,
Linguistics, and Neuroscience. We identify 3 criteria of intelligence that the
community agrees on the most: generalization, adaptability, & reasoning. Our
results suggests that the perception of the current NLP systems as
''intelligent'' is a minority position (29%). Furthermore, only 16.2% of the
respondents see developing intelligent systems as a research goal, and these
respondents are more likely to consider the current systems intelligent.

### 5. [Racism, Resistance, and Reddit: How Popular Culture Sparks Online Reckonings](http://arxiv.org/pdf/2505.21016v1)

Authors: Sherry Mason, Tawfiq Ammari

This study examines how Reddit users engaged with the racial narratives of
Lovecraft Country and Watchmen, two television series that reimagine historical
racial trauma. Drawing on narrative persuasion and multistep flow theory, we
analyze 3,879 Reddit comments using topic modeling and critical discourse
analysis. We identify three dynamic social roles advocates, adversaries, and
adaptives and explore how users move between them in response to racial
discourse. Findings reveal how Reddits pseudonymous affordances shape role
fluidity, opinion leadership, and moral engagement. While adversaries minimized
or rejected racism as exaggerated, advocates shared standpoint experiences and
historical resources to challenge these claims. Adaptive users shifted
perspectives over time, demonstrating how online publics can foster critical
racial learning. This research highlights how popular culture and participatory
platforms intersect in shaping collective meaning making around race and
historical memory.

### 6. [Fixed-Point Traps and Identity Emergence in Educational Feedback Systems](http://arxiv.org/pdf/2505.21038v1)

Authors: Faruk Alpay

This paper presents a formal categorical proof that exam-driven educational
systems obstruct identity emergence and block creative convergence. Using the
framework of Alpay Algebra II and III, we define Exam-Grade Collapse Systems
(EGCS) as functorial constructs where learning dynamics $\varphi$ are
recursively collapsed by evaluative morphisms $E$. We prove that under such
collapse regimes, no nontrivial fixed-point algebra $\mu_\varphi$ can exist,
hence learner identity cannot stabilize. This creates a universal fixed-point
trap: all generative functors are entropically folded before symbolic emergence
occurs. Our model mathematically explains the creativity suppression, research
stagnation, and structural entropy loss induced by timed exams and grade-based
feedback. The results apply category theory to expose why modern educational
systems prevent {\phi}-emergence and block observer-invariant self-formation.
This work provides the first provable algebraic obstruction of identity
formation caused by institutional feedback mechanics.

### 7. [Position is Power: System Prompts as a Mechanism of Bias in Large Language Models (LLMs)](http://arxiv.org/pdf/2505.21091v1)

Authors: Anna Neumann, Elisabeth Kirsten, Muhammad Bilal Zafar, Jatinder Singh

System prompts in Large Language Models (LLMs) are predefined directives that
guide model behaviour, taking precedence over user inputs in text processing
and generation. LLM deployers increasingly use them to ensure consistent
responses across contexts. While model providers set a foundation of system
prompts, deployers and third-party developers can append additional prompts
without visibility into others' additions, while this layered implementation
remains entirely hidden from end-users. As system prompts become more complex,
they can directly or indirectly introduce unaccounted for side effects. This
lack of transparency raises fundamental questions about how the position of
information in different directives shapes model outputs. As such, this work
examines how the placement of information affects model behaviour. To this end,
we compare how models process demographic information in system versus user
prompts across six commercially available LLMs and 50 demographic groups. Our
analysis reveals significant biases, manifesting in differences in user
representation and decision-making scenarios. Since these variations stem from
inaccessible and opaque system-level configurations, they risk
representational, allocative and potential other biases and downstream harms
beyond the user's ability to detect or correct. Our findings draw attention to
these critical issues, which have the potential to perpetuate harms if left
unexamined. Further, we argue that system prompt analysis must be incorporated
into AI auditing processes, particularly as customisable system prompts become
increasingly prevalent in commercial AI deployments.

### 8. [GGBond: Growing Graph-Based AI-Agent Society for Socially-Aware Recommender Simulation](http://arxiv.org/pdf/2505.21154v1)

Authors: Hailin Zhong, Hanlin Wang, Yujun Ye, Meiyi Zhang, Shengxin Zhu

Current personalized recommender systems predominantly rely on static offline
data for algorithm design and evaluation, significantly limiting their ability
to capture long-term user preference evolution and social influence dynamics in
real-world scenarios. To address this fundamental challenge, we propose a
high-fidelity social simulation platform integrating human-like cognitive
agents and dynamic social interactions to realistically simulate user behavior
evolution under recommendation interventions. Specifically, the system
comprises a population of Sim-User Agents, each equipped with a five-layer
cognitive architecture that encapsulates key psychological mechanisms,
including episodic memory, affective state transitions, adaptive preference
learning, and dynamic trust-risk assessments. In particular, we innovatively
introduce the Intimacy--Curiosity--Reciprocity--Risk (ICR2) motivational engine
grounded in psychological and sociological theories, enabling more realistic
user decision-making processes. Furthermore, we construct a multilayer
heterogeneous social graph (GGBond Graph) supporting dynamic relational
evolution, effectively modeling users' evolving social ties and trust dynamics
based on interest similarity, personality alignment, and structural homophily.
During system operation, agents autonomously respond to recommendations
generated by typical recommender algorithms (e.g., Matrix Factorization,
MultVAE, LightGCN), deciding whether to consume, rate, and share content while
dynamically updating their internal states and social connections, thereby
forming a stable, multi-round feedback loop. This innovative design transcends
the limitations of traditional static datasets, providing a controlled,
observable environment for evaluating long-term recommender effects.

### 9. [Parameter Effects in ReCom Ensembles](http://arxiv.org/pdf/2505.21326v1)

Authors: Kristopher Tapp, Todd Proebsting, Alec Ramsay

Ensemble analysis has become central to redistricting litigation, but
parameter effects remain understudied. We analyze 315 ReCom ensembles across
the three legislative chambers in 7 states, systematically varying the
population tolerance, county preservation strength, and algorithm variant. To
validate convergence, we introduce new methods to approximate effective sample
size and measure redundancy. We find that varying the population tolerance has
a negligible effect on all scores, whereas the algorithm and
county-preservation parameters can significantly affect some metrics,
inconsistently in some cases but surprisingly consistently in others across
jurisdictions. These findings suggest parameter choices should be thoughtfully
considered when using ReCom ensembles.

### 10. [Improving Research Idea Generation Through Data: An Empirical Investigation in Social Science](http://arxiv.org/pdf/2505.21396v1)

Authors: Xiao Liu, Xinyi Dong, Xinyang Gao, Yansong Feng, Xun Pang

Recent advancements in large language models (LLMs) have shown promise in
generating novel research ideas. However, these ideas often face challenges
related to feasibility and expected effectiveness. This paper explores how
augmenting LLMs with relevant data during the idea generation process can
enhance the quality of generated ideas. We introduce two ways of incorporating
data: (1) providing metadata during the idea generation stage to guide LLMs
toward feasible directions, and (2) adding automatic validation during the idea
selection stage to assess the empirical plausibility of hypotheses within
ideas. We conduct experiments in the social science domain, specifically with
climate negotiation topics, and find that metadata improves the feasibility of
generated ideas by 20%, while automatic validation improves the overall quality
of selected ideas by 7%. A human study shows that LLM-generated ideas, along
with their related data and validation processes, inspire researchers to
propose research ideas with higher quality. Our work highlights the potential
of data-driven research idea generation, and underscores the practical utility
of LLM-assisted ideation in real-world academic settings.

### Databases

### 1. [In-memory Incremental Maintenance of Provenance Sketches [extended version]](http://arxiv.org/pdf/2505.20683v1)

Authors: Pengyuan Li, Boris Glavic, Dieter Gawlick, Vasudha Krishnaswamy, Zhen Hua Liu, Danica Porobic, Xing Niu

Provenance-based data skipping compactly over-approximates the provenance of
a query using so-called provenance sketches and utilizes such sketches to
speed-up the execution of subsequent queries by skipping irrelevant data.
However, a sketch captured at some time in the past may become stale if the
data has been updated subsequently. Thus, there is a need to maintain
provenance sketches. In this work, we introduce In-Memory incremental
Maintenance of Provenance sketches (IMP), a framework for maintaining sketches
incrementally under updates. At the core of IMP is an incremental query engine
for data annotated with sketches that exploits the coarse-grained nature of
sketches to enable novel optimizations. We experimentally demonstrate that IMP
significantly reduces the cost of sketch maintenance, thereby enabling the use
of provenance sketches for a broad range of workloads that involve updates.

### 2. [Streamlining Knowledge Graph Creation with PyRML](http://arxiv.org/pdf/2505.20949v1)

Authors: Andrea Giovanni Nuzzolese

Knowledge Graphs (KGs) are increasingly adopted as a foundational technology
for integrating heterogeneous data in domains such as climate science, cultural
heritage, and the life sciences. Declarative mapping languages like R2RML and
RML have played a central role in enabling scalable and reusable KG
construction, offering a transparent means of transforming structured and
semi-structured data into RDF. In this paper, we present PyRML, a lightweight,
Python-native library for building Knowledge Graphs through declarative
mappings. PyRML supports core RML constructs and provides a programmable
interface for authoring, executing, and testing mappings directly within Python
environments. It integrates with popular data and semantic web libraries (e.g.,
Pandas and RDFlib), enabling transparent and modular workflows. By lowering the
barrier to entry for KG creation and fostering reproducible, ontology-aligned
data integration, PyRML bridges the gap between declarative semantics and
practical KG engineering.

### 3. [RelationalFactQA: A Benchmark for Evaluating Tabular Fact Retrieval from Large Language Models](http://arxiv.org/pdf/2505.21409v1)

Authors: Dario Satriani, Enzo Veltri, Donatello Santoro, Paolo Papotti

Factuality in Large Language Models (LLMs) is a persistent challenge. Current
benchmarks often assess short factual answers, overlooking the critical ability
to generate structured, multi-record tabular outputs from parametric knowledge.
We demonstrate that this relational fact retrieval is substantially more
difficult than isolated point-wise queries, even when individual facts are
known to the model, exposing distinct failure modes sensitive to output
dimensionality (e.g., number of attributes or records). To systematically
evaluate this under-explored capability, we introduce RelationalFactQA, a new
benchmark featuring diverse natural language questions (paired with SQL) and
gold-standard tabular answers, specifically designed to assess knowledge
retrieval in a structured format. RelationalFactQA enables analysis across
varying query complexities, output sizes, and data characteristics. Our
experiments reveal that even state-of-the-art LLMs struggle significantly, not
exceeding 25% factual accuracy in generating relational outputs, with
performance notably degrading as output dimensionality increases. These
findings underscore critical limitations in current LLMs' ability to synthesize
structured factual knowledge and establish RelationalFactQA as a crucial
resource for measuring future progress in LLM factuality.

### 4. [Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks](http://arxiv.org/pdf/2505.21329v1)

Authors: Allaa Boutaleb, Bernd Amann, Hubert Naacke, Rafael Angarita

Recent table representation learning and data discovery methods tackle table
union search (TUS) within data lakes, which involves identifying tables that
can be unioned with a given query table to enrich its content. These methods
are commonly evaluated using benchmarks that aim to assess semantic
understanding in real-world TUS tasks. However, our analysis of prominent TUS
benchmarks reveals several limitations that allow simple baselines to perform
surprisingly well, often outperforming more sophisticated approaches. This
suggests that current benchmark scores are heavily influenced by
dataset-specific characteristics and fail to effectively isolate the gains from
semantic understanding. To address this, we propose essential criteria for
future benchmarks to enable a more realistic and reliable evaluation of
progress in semantic table union search.

### 5. [LazyVLM: Neuro-Symbolic Approach to Video Analytics](http://arxiv.org/pdf/2505.21459v1)

Authors: Xiangru Jian, Wei Pang, Zhengyuan Dong, Chao Zhang, M. Tamer Özsu

Current video analytics approaches face a fundamental trade-off between
flexibility and efficiency. End-to-end Vision Language Models (VLMs) often
struggle with long-context processing and incur high computational costs, while
neural-symbolic methods depend heavily on manual labeling and rigid rule
design. In this paper, we introduce LazyVLM, a neuro-symbolic video analytics
system that provides a user-friendly query interface similar to VLMs, while
addressing their scalability limitation. LazyVLM enables users to effortlessly
drop in video data and specify complex multi-frame video queries using a
semi-structured text interface for video analytics. To address the scalability
limitations of VLMs, LazyVLM decomposes multi-frame video queries into
fine-grained operations and offloads the bulk of the processing to efficient
relational query execution and vector similarity search. We demonstrate that
LazyVLM provides a robust, efficient, and user-friendly solution for querying
open-domain video data at scale.

### Distributed, Parallel, and Cluster Computing

### 1. [ECC-SNN: Cost-Effective Edge-Cloud Collaboration for Spiking Neural Networks](http://arxiv.org/pdf/2505.20835v1)

Authors: Di Yu, Changze Lv, Xin Du, Linshan Jiang, Wentao Tong, Zhenyu Liao, Xiaoqing Zheng, Shuiguang Deng

Most edge-cloud collaboration frameworks rely on the substantial
computational and storage capabilities of cloud-based artificial neural
networks (ANNs). However, this reliance results in significant communication
overhead between edge devices and the cloud and high computational energy
consumption, especially when applied to resource-constrained edge devices. To
address these challenges, we propose ECC-SNN, a novel edge-cloud collaboration
framework incorporating energy-efficient spiking neural networks (SNNs) to
offload more computational workload from the cloud to the edge, thereby
improving cost-effectiveness and reducing reliance on the cloud. ECC-SNN
employs a joint training approach that integrates ANN and SNN models, enabling
edge devices to leverage knowledge from cloud models for enhanced performance
while reducing energy consumption and processing latency. Furthermore, ECC-SNN
features an on-device incremental learning algorithm that enables edge models
to continuously adapt to dynamic environments, reducing the communication
overhead and resource consumption associated with frequent cloud update
requests. Extensive experimental results on four datasets demonstrate that
ECC-SNN improves accuracy by 4.15%, reduces average energy consumption by
79.4%, and lowers average processing latency by 39.1%.

### 2. [Load Balancing in Strongly Inhomogeneous Simulations -- a Vlasiator Case Study](http://arxiv.org/pdf/2505.20908v1)

Authors: Leo Kotipalo, Markus Battarbee, Yann Pfau-Kempf, Vertti Tarvus, Minna Palmroth

Parallelization is a necessity for large-scale simulations due to the amount
of data processed. In this article we investigate different load balancing
methods using Vlasiator, a global magnetospheric simulation as our case study.
  The theoretical basis for load balancing is the (hyper)graph partitioning
problem, modeling simulation units as vertices and their data dependencies as
edges. As it is an NP-hard problem, heuristics are necessary for dynamic
runtime balancing.
  We consider first hypergraph partitioning via an algorithm called parallel
hypergraph partitioner (PHG); this is done by partitioning a simplified grid
and then attempting to optimize the solution on the finer grid. The second and
third are the geometric methods of recursive coordinate bisection (RCB) and
recursive inertial bisection (RIB).
  Finally we consider the method of Hilbert space filling curves (HSFC). The
algorithm projects simulation cells along a Hilbert curve and makes cuts along
the curve. This works well due to the excellent locality of Hilbert curves, and
can be optimized further by choice of curve. We introduce and investigate six
three-dimensional Hilbert curves in total.
  Our findings on runs of two different scales indicate the HSFC method
provides optimal load balance, followed by RIB and PHG methods and finally by
RCB. Of the Hilbert curves evaluated, the Beta curve outperformed the most
commonly used curve by a few percent.

### 3. [Vectorized Sequence-Based Chunking for Data Deduplication](http://arxiv.org/pdf/2505.21194v1)

Authors: Sreeharsha Udayashankar, Samer Al-Kiswany

Data deduplication has gained wide acclaim as a mechanism to improve storage
efficiency and conserve network bandwidth. Its most critical phase, data
chunking, is responsible for the overall space savings achieved via the
deduplication process. However, modern data chunking algorithms are slow and
compute-intensive because they scan large amounts of data while simultaneously
making data-driven boundary decisions.
  We present SeqCDC, a novel chunking algorithm that leverages lightweight
boundary detection, content-defined skipping, and SSE/AVX acceleration to
improve chunking throughput for large chunk sizes. Our evaluation shows that
SeqCDC achieves 15x higher throughput than unaccelerated and 1.2x-1.35x higher
throughput than vector-accelerated data chunking algorithms while minimally
affecting deduplication space savings.

### 4. [Multi-Event Triggers for Serverless Computing](http://arxiv.org/pdf/2505.21199v1)

Authors: Valentin Carl, Trever Schirmer, Joshua Adamek, Tobias Pfandzelter, Sergio Lucia, David Bermbach

Function-as-a-Service (FaaS) is an event-driven serverless cloud computing
model in which small, stateless functions are invoked in response to events,
such as HTTP requests, new database entries, or messages. Current FaaS platform
assume that each function invocation corresponds to a single event. However,
from an application perspective, it is desirable to invoke functions in
response to a collection of events of different types or only with every
n\textsuperscript{th} event. To implement this today, a function would need
additional state management, e.g., in a database, and custom logic to determine
whether its trigger condition is fulfilled and the actual application code
should run. In such an implementation, most function invocations would be
rendered essentially useless, leading to unnecessarily high resource usage,
latency, and cost for applications. In this paper, we introduce multi-event
triggers, through which complex conditions for function invocations can be
specified. Specifically, we introduce abstractions for invoking functions based
on a set of $n$ events and joins of multiple events of different types. This
enables application developers to define intricate conditions for function
invocations, workflow steps, and complex event processing. Our evaluation with
a proof-of-concept prototype shows that this reduces event--invocation latency
by 62.5\% in an incident detection use-case and that our system can handle more
than 300,000 requests per second on limited hardware, which is sufficient load
for implementation in large FaaS platforms.

### 5. [Distributed Discrete Morse Sandwich: Efficient Computation of Persistence Diagrams for Massive Scalar Data](http://arxiv.org/pdf/2505.21266v1)

Authors: Eve Le Guillou, Pierre Fortin, Julien Tierny

The persistence diagram, which describes the topological features of a
dataset, is a key descriptor in Topological Data Analysis. The "Discrete Morse
Sandwich" (DMS) method has been reported to be the most efficient algorithm for
computing persistence diagrams of 3D scalar fields on a single node, using
shared-memory parallelism. In this work, we extend DMS to distributed-memory
parallelism for the efficient and scalable computation of persistence diagrams
for massive datasets across multiple compute nodes. On the one hand, we can
leverage the embarrassingly parallel procedure of the first and most
time-consuming step of DMS (namely the discrete gradient computation). On the
other hand, the efficient distributed computations of the subsequent DMS steps
are much more challenging. To address this, we have extensively revised the DMS
routines by contributing a new self-correcting distributed pairing algorithm,
redesigning key data structures and introducing computation tokens to
coordinate distributed computations. We have also introduced a dedicated
communication thread to overlap communication and computation. Detailed
performance analyses show the scalability of our hybrid MPI+thread approach for
strong and weak scaling using up to 16 nodes of 32 cores (512 cores total). Our
algorithm outperforms DIPHA, a reference method for the distributed computation
of persistence diagrams, with an average speedup of x8 on 512 cores. We show
the practical capabilities of our approach by computing the persistence diagram
of a public 3D scalar field of 6 billion vertices in 174 seconds on 512 cores.
Finally, we provide a usage example of our open-source implementation at
https://github.com/eve-le-guillou/DDMS-example.

### 6. [Time-Series Learning for Proactive Fault Prediction in Distributed Systems with Deep Neural Structures](http://arxiv.org/pdf/2505.20705v1)

Authors: Yang Wang, Wenxuan Zhu, Xuehui Quan, Heyi Wang, Chang Liu, Qiyuan Wu

This paper addresses the challenges of fault prediction and delayed response
in distributed systems by proposing an intelligent prediction method based on
temporal feature learning. The method takes multi-dimensional performance
metric sequences as input. We use a Gated Recurrent Unit (GRU) to model the
evolution of system states over time. An attention mechanism is then applied to
enhance key temporal segments, improving the model's ability to identify
potential faults. On this basis, a feedforward neural network is designed to
perform the final classification, enabling early warning of system failures. To
validate the effectiveness of the proposed approach, comparative experiments
and ablation analyses were conducted using data from a large-scale real-world
cloud system. The experimental results show that the model outperforms various
mainstream time-series models in terms of Accuracy, F1-Score, and AUC. This
demonstrates strong prediction capability and stability. Furthermore, the loss
function curve confirms the convergence and reliability of the training
process. It indicates that the proposed method effectively learns system
behavior patterns and achieves efficient fault detection.

### 7. [Choreographies as Macros](http://arxiv.org/pdf/2505.20845v1)

Authors: Alexander Bohosian, Andrew K. Hirsch

Concurrent programming often entails meticulous pairing of sends and receives
between participants to avoid deadlock. Choreographic programming alleviates
this burden by specifying the system as a single program. However, there are
more applications than implementations of choreographies, and developing new
implementations takes a lot of time and effort. Our work uses Racket to
expedite building a new choreographic language called Choret. Racket has a
powerful macro system which allows Choret to reuse much of its infrastructure
for greater functionality and correctness.

### 8. [Reduced and mixed precision turbulent flow simulations using explicit finite difference schemes](http://arxiv.org/pdf/2505.20911v1)

Authors: Bálint Siklósi, Pushpender K. Sharma, David J. Lusher, István Z. Reguly, Neil D. Sandham

The use of reduced and mixed precision computing has gained increasing
attention in high-performance computing (HPC) as a means to improve
computational efficiency, particularly on modern hardware architectures like
GPUs. In this work, we explore the application of mixed precision arithmetic in
compressible turbulent flow simulations using explicit finite difference
schemes. We extend the OPS and OpenSBLI frameworks to support customizable
precision levels, enabling fine-grained control over precision allocation for
different computational tasks. Through a series of numerical experiments on the
Taylor-Green vortex benchmark, we demonstrate that mixed precision strategies,
such as half-single and single-double combinations, can offer significant
performance gains without compromising numerical accuracy. However, pure
half-precision computations result in unacceptable accuracy loss, underscoring
the need for careful precision selection. Our results show that mixed precision
configurations can reduce memory usage and communication overhead, leading to
notable speedups, particularly on multi-CPU and multi-GPU systems.

### 9. [A Hitchhiker's Guide to Privacy-Preserving Cryptocurrencies: A Survey on Anonymity, Confidentiality, and Auditability](http://arxiv.org/pdf/2505.21008v1)

Authors: Matteo Nardelli, Francesco De Sclavis, Michela Iezzi

Cryptocurrencies and central bank digital currencies (CBDCs) are reshaping
the monetary landscape, offering transparency and efficiency while raising
critical concerns about user privacy and regulatory compliance. This survey
provides a comprehensive and technically grounded overview of
privacy-preserving digital currencies, covering both cryptocurrencies and
CBDCs. We propose a taxonomy of privacy goals -- including anonymity,
confidentiality, unlinkability, and auditability -- and map them to underlying
cryptographic primitives, protocol mechanisms, and system architectures. Unlike
previous surveys, our work adopts a design-oriented perspective, linking
high-level privacy objectives to concrete implementations. We also trace the
evolution of privacy-preserving currencies through three generations,
highlighting shifts from basic anonymity guarantees toward more nuanced
privacy-accountability trade-offs. Finally, we identify open challenges at the
intersection of cryptography, distributed systems, and policy definition, which
motivate further investigation into the primitives and design of digital
currencies that balance real-world privacy and auditability needs.

### 10. [SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](http://arxiv.org/pdf/2505.21051v1)

Authors: Jianmin Liu, Li Yan, Borui Li, Lei Yu, Chao Shen

Federated fine-tuning of large language models (LLMs) is critical for
improving their performance in handling domain-specific tasks. However, prior
work has shown that clients' private data can actually be recovered via
gradient inversion attacks. Existing privacy preservation techniques against
such attacks typically entail performance degradation and high costs, making
them ill-suited for clients with heterogeneous data distributions and device
capabilities. In this paper, we propose SHE-LoRA, which integrates selective
homomorphic encryption (HE) and low-rank adaptation (LoRA) to enable efficient
and privacy-preserving federated tuning of LLMs in cross-device environment.
Heterogeneous clients adaptively select partial model parameters for
homomorphic encryption based on parameter sensitivity assessment, with the
encryption subset obtained via negotiation. To ensure accurate model
aggregation, we design a column-aware secure aggregation method and customized
reparameterization techniques to align the aggregation results with the
heterogeneous device capabilities of clients. Extensive experiments demonstrate
that SHE-LoRA maintains performance comparable to non-private baselines,
achieves strong resistance to the state-of-the-art attacks, and significantly
reduces communication overhead by 94.901\% and encryption computation overhead
by 99.829\%, compared to baseline. Our code is accessible at
https://anonymous.4open.science/r/SHE-LoRA-8D84.

### Digital Libraries

### 1. [International collaboration of Ukrainian scholars: Effects of Russia's full-scale invasion of Ukraine](http://arxiv.org/pdf/2505.20944v1)

Authors: Myroslava Hladchenko

This study explores the effects of Russia's full-scale invasion of Ukraine on
the international collaboration of Ukrainian scholars. First and foremost,
Ukrainian scholars deserve respect for continuing to publish despite
life-threatening conditions, mental strain, shelling and blackouts. In
2022-2023, universities gained more from international collaboration than the
NASU. The percentage of internationally co-authored articles remained unchanged
for the NASU, while it increased for universities. In 2023, 40.8% of articles
published by the NASU and 32,2% of articles published by universities were
internationally co-authored. However, these figures are still much lower than
in developed countries (60-70%). The citation impact of internationally
co-authored articles remained statistically unchanged for the NASU but
increased for universities. The highest share of internationally co-authored
articles published by the NASU in both periods was in the physical sciences and
engineering. However, the citation impact of these articles declined in
2022-2023, nearly erasing their previous citation advantage over university
publications. Universities consistently outperformed the NASU in the citation
impact of internationally co-authored articles in biomedical and health
sciences across both periods. International collaboration can help Ukrainian
scholars to go through this difficult time. In turn, they can contribute to the
strengthening of Europe.

### 2. [Leveraging GANs for citation intent classification and its impact on citation network analysis](http://arxiv.org/pdf/2505.21162v1)

Authors: Davi A. Bezerra, Filipi N. Silva, Diego R. Amancio

Citations play a fundamental role in the scientific ecosystem, serving as a
foundation for tracking the flow of knowledge, acknowledging prior work, and
assessing scholarly influence. In scientometrics, they are also central to the
construction of quantitative indicators. Not all citations, however, serve the
same function: some provide background, others introduce methods, or compare
results. Therefore, understanding citation intent allows for a more nuanced
interpretation of scientific impact. In this paper, we adopted a GAN-based
method to classify citation intents. Our results revealed that the proposed
method achieves competitive classification performance, closely matching
state-of-the-art results with substantially fewer parameters. This demonstrates
the effectiveness and efficiency of leveraging GAN architectures combined with
contextual embeddings in intent classification task. We also investigated
whether filtering citation intents affects the centrality of papers in citation
networks. Analyzing the network constructed from the unArXiv dataset, we found
that paper rankings can be significantly influenced by citation intent. All
four centrality metrics examined- degree, PageRank, closeness, and betweenness
- were sensitive to the filtering of citation types. The betweenness centrality
displayed the greatest sensitivity, showing substantial changes in ranking when
specific citation intents were removed.

### Discrete Mathematics

### 1. [A refined view of a curious identity for partitions into odd parts with designated summands](http://arxiv.org/pdf/2505.21111v1)

Authors: Shishuo Fu, James Sellers

In 2002, Andrews, Lewis, and Lovejoy introduced the combinatorial objects
which they called partitions with designated summands. These are constructed by
taking unrestricted integer partitions and designating exactly one of each
occurrence of a part. In the same work, they also considered the restricted
partitions with designated summands wherein all parts must be odd, and they
denoted the corresponding function by $\mathrm{PDO}(n)$.

### 2. [Rainbow copies of spanning subgraphs](http://arxiv.org/pdf/2505.21290v1)

Authors: Colin Cooper, Alan Frieze

Let $G_{n,p}^{[\kappa]}$ denote the space of $n$-vertex edge coloured graphs,
where each edge occurs independently with probability $p$. The colour of each
existing edge is chosen independently and uniformly at random from the set
$[\kappa]$. We consider the threshold for the existence of rainbow colored
copies of a spanning subgraph $H$. We provide lower bounds on $p$ and $\kappa$
sufficient to prove the existence of such copies w.h.p.

### 3. [Pushing Cops and Robber on Graphs of Maximum Degree 4](http://arxiv.org/pdf/2505.21450v1)

Authors: Harmender Gahlawat

\textsc{Cops and Robber} is a game played on graphs where a set of
\textit{cops} aim to \textit{capture} the position of a single \textit{robber}.
The main parameter of interest in this game is the \textit{cop number}, which
is the minimum number of cops that are sufficient to guarantee the capture of
the robber.
  In a directed graph $\overrightarrow{G}$, the \textit{push} operation on a
vertex $v$ reverses the orientation of all arcs incident on $v$. We consider a
variation of classical \textsc{Cops and Robber} on oriented graphs, where in
its turn, each cop can either move to an out-neighbor of its current vertex or
push some vertex of the graph, whereas, the robber can move to an adjacent
vertex in its turn. [Das et al., CALDAM, 2023] introduced this variant and
established that if $\overrightarrow{G}$ is an orientation of a subcubic graph,
then one cop with push ability has a winning strategy. We extend these results
to establish that if $\overrightarrow{G}$ is an orientation of a $3$-degenerate
graph, or of a graph with maximum degree $4$, then one cop with push ability
has a winning strategy.

### 4. [Colouring Probe $H$-Free Graphs](http://arxiv.org/pdf/2505.20784v1)

Authors: Daniël Paulusma, Johannes Rauch, Erik Jan van Leeuwen

The NP-complete problems Colouring and k-Colouring $(k\geq 3$) are well
studied on $H$-free graphs, i.e., graphs that do not contain some fixed graph
$H$ as an induced subgraph. We research to what extent the known
polynomial-time algorithms for $H$-free graphs can be generalized if we only
know some of the edges of the input graph. We do this by considering the
classical probe graph model introduced in the early nineties. For a graph $H$,
a partitioned probe $H$-free graph $(G,P,N)$ consists of a graph $G=(V,E)$,
together with a set $P\subseteq V$ of probes and an independent set
$N=V\setminus P$ of non-probes, such that $G+F$ is $H$-free for some edge set
$F\subseteq \binom{N}{2}$. We first fully classify the complexity of Colouring
on partitioned probe $H$-free graphs and show that this dichotomy is different
from the known dichotomy of Colouring for $H$-free graphs. Our main result is a
dichotomy of $3$-Colouring for partitioned probe $P_t$-free graphs: we prove
that the problem is polynomial-time solvable if $t\leq 5$ but NP-complete if
$t\geq 6$. In contrast, $3$-Colouring on $P_t$-free graphs is known to be
polynomial-time solvable if $t\leq 7$ and quasi polynomial-time solvable for
$t\geq 8$.

### 5. [Complexity landscape for local certification](http://arxiv.org/pdf/2505.20915v1)

Authors: Nicolas Bousquet, Laurent Feuilloley, Sébastien Zeitoun

An impressive recent line of work has charted the complexity landscape of
distributed graph algorithms. For many settings, it has been determined which
time complexities exist, and which do not (in the sense that no local problem
could have an optimal algorithm with that complexity). In this paper, we
initiate the study of the landscape for space complexity of distributed graph
algorithms. More precisely, we focus on the local certification setting, where
a prover assigns certificates to nodes to certify a property, and where the
space complexity is measured by the size of the certificates.
  Already for anonymous paths and cycles, we unveil a surprising landscape:
  - There is a gap between complexity $O(1)$ and $\Theta(\log \log n)$ in
paths. This is the first gap established in local certification.
  - There exists a property that has complexity $\Theta(\log \log n)$ in paths,
a regime that was not known to exist for a natural property.
  - There is a gap between complexity $O(1)$ and $\Theta(\log n)$ in cycles,
hence a gap that is exponentially larger than for paths.
  We then generalize our result for paths to the class of trees. Namely, we
show that there is a gap between complexity $O(1)$ and $\Theta(\log \log d)$ in
trees, where $d$ is the diameter. We finally describe some settings where there
are no gaps at all. To prove our results we develop a new toolkit, based on
various results of automata theory and arithmetic, which is of independent
interest.

### Data Structures and Algorithms

### 1. [Course Allocation with Credits via Stable Matching](http://arxiv.org/pdf/2505.21229v1)

Authors: José Rodríguez, David Manlove

In the {\sc Course Allocation} problem, there are a set of students and a set
of courses at a given university. University courses may have different numbers
of credits, typically related to different numbers of learning hours, and there
may be other constraints such as courses running concurrently. Our goal is to
allocate the students to the courses such that the resulting matching is
stable, which means that no student and course(s) have an incentive to break
away from the matching and become assigned to one another. We study several
definitions of stability and for each we give a mixture of polynomial-time
algorithms and hardness results for problems involving verifying the stability
of a matching, finding a stable matching or determining that none exists, and
finding a maximum size stable matching. We also study variants of the problem
with master lists of students, and lower quotas on the number of students
allocated to a course, establishing additional complexity results in these
settings.

### 2. [Optimal Approximations for the Requirement Cut Problem on Sparse Graph Classes](http://arxiv.org/pdf/2505.21433v1)

Authors: Nadym Mallek, Kirill Simonov

We study the Requirement Cut problem, a generalization of numerous classical
graph partitioning problems including Multicut, Multiway Cut, $k$-Cut, and
Steiner Multicut among others. Given a graph with edge costs, terminal groups
$(S_1, ..., S_g)$ and integer requirements $(r_1,... , r_g)$; the goal is to
compute a minimum-cost edge cut that separates each group $S_i$ into at least
$r_i$ connected components. Despite many efforts, the best known approximation
for Requirement Cut yields a double-logarithmic $O(log(g).\log(n))$
approximation ratio as it relies on embedding general graphs into trees and
solving the tree instance.
  In this paper, we explore two largely unstudied structural parameters in
order to obtain single-logarithmic approximation ratios: (1) the number of
minimal Steiner trees in the instance, which in particular is upper-bounded by
the number of spanning trees of the graphs multiplied by $g$, and (2) the depth
of series-parallel graphs. Specifically, we show that if the number of minimal
Steiner trees is polynomial in $n$, then a simple LP-rounding algorithm yields
an $O(log n)$-approximation, and if the graph is series-parallel with a
constant depth then a refined analysis of a known probabilistic embedding
yields a $O(depth.log(g))$-approximation on series-parallel graphs of bounded
depth. Both results extend the known class of graphs that have a
single-logarithmic approximation ratio.

### 3. [Algorithms and SQ Lower Bounds for Robustly Learning Real-valued Multi-index Models](http://arxiv.org/pdf/2505.21475v1)

Authors: Ilias Diakonikolas, Giannis Iakovidis, Daniel M. Kane, Lisheng Ren

We study the complexity of learning real-valued Multi-Index Models (MIMs)
under the Gaussian distribution. A $K$-MIM is a function $f:\mathbb{R}^d\to
\mathbb{R}$ that depends only on the projection of its input onto a
$K$-dimensional subspace. We give a general algorithm for PAC learning a broad
class of MIMs with respect to the square loss, even in the presence of
adversarial label noise. Moreover, we establish a nearly matching Statistical
Query (SQ) lower bound, providing evidence that the complexity of our algorithm
is qualitatively optimal as a function of the dimension. Specifically, we
consider the class of bounded variation MIMs with the property that degree at
most $m$ distinguishing moments exist with respect to projections onto any
subspace. In the presence of adversarial label noise, the complexity of our
learning algorithm is $d^{O(m)}2^{\mathrm{poly}(K/\epsilon)}$. For the
realizable and independent noise settings, our algorithm incurs complexity
$d^{O(m)}2^{\mathrm{poly}(K)}(1/\epsilon)^{O(K)}$. To complement our upper
bound, we show that if for some subspace degree-$m$ distinguishing moments do
not exist, then any SQ learner for the corresponding class of MIMs requires
complexity $d^{\Omega(m)}$. As an application, we give the first efficient
learner for the class of positive-homogeneous $L$-Lipschitz $K$-MIMs. The
resulting algorithm has complexity $\mathrm{poly}(d)
2^{\mathrm{poly}(KL/\epsilon)}$. This gives a new PAC learning algorithm for
Lipschitz homogeneous ReLU networks with complexity independent of the network
size, removing the exponential dependence incurred in prior work.

### 4. [Colouring Probe $H$-Free Graphs](http://arxiv.org/pdf/2505.20784v1)

Authors: Daniël Paulusma, Johannes Rauch, Erik Jan van Leeuwen

The NP-complete problems Colouring and k-Colouring $(k\geq 3$) are well
studied on $H$-free graphs, i.e., graphs that do not contain some fixed graph
$H$ as an induced subgraph. We research to what extent the known
polynomial-time algorithms for $H$-free graphs can be generalized if we only
know some of the edges of the input graph. We do this by considering the
classical probe graph model introduced in the early nineties. For a graph $H$,
a partitioned probe $H$-free graph $(G,P,N)$ consists of a graph $G=(V,E)$,
together with a set $P\subseteq V$ of probes and an independent set
$N=V\setminus P$ of non-probes, such that $G+F$ is $H$-free for some edge set
$F\subseteq \binom{N}{2}$. We first fully classify the complexity of Colouring
on partitioned probe $H$-free graphs and show that this dichotomy is different
from the known dichotomy of Colouring for $H$-free graphs. Our main result is a
dichotomy of $3$-Colouring for partitioned probe $P_t$-free graphs: we prove
that the problem is polynomial-time solvable if $t\leq 5$ but NP-complete if
$t\geq 6$. In contrast, $3$-Colouring on $P_t$-free graphs is known to be
polynomial-time solvable if $t\leq 7$ and quasi polynomial-time solvable for
$t\geq 8$.

### 5. [Complexity landscape for local certification](http://arxiv.org/pdf/2505.20915v1)

Authors: Nicolas Bousquet, Laurent Feuilloley, Sébastien Zeitoun

An impressive recent line of work has charted the complexity landscape of
distributed graph algorithms. For many settings, it has been determined which
time complexities exist, and which do not (in the sense that no local problem
could have an optimal algorithm with that complexity). In this paper, we
initiate the study of the landscape for space complexity of distributed graph
algorithms. More precisely, we focus on the local certification setting, where
a prover assigns certificates to nodes to certify a property, and where the
space complexity is measured by the size of the certificates.
  Already for anonymous paths and cycles, we unveil a surprising landscape:
  - There is a gap between complexity $O(1)$ and $\Theta(\log \log n)$ in
paths. This is the first gap established in local certification.
  - There exists a property that has complexity $\Theta(\log \log n)$ in paths,
a regime that was not known to exist for a natural property.
  - There is a gap between complexity $O(1)$ and $\Theta(\log n)$ in cycles,
hence a gap that is exponentially larger than for paths.
  We then generalize our result for paths to the class of trees. Namely, we
show that there is a gap between complexity $O(1)$ and $\Theta(\log \log d)$ in
trees, where $d$ is the diameter. We finally describe some settings where there
are no gaps at all. To prove our results we develop a new toolkit, based on
various results of automata theory and arithmetic, which is of independent
interest.

### 6. [High-Dimensional Calibration from Swap Regret](http://arxiv.org/pdf/2505.21460v1)

Authors: Maxwell Fishelson, Noah Golowich, Mehryar Mohri, Jon Schneider

We study the online calibration of multi-dimensional forecasts over an
arbitrary convex set $\mathcal{P} \subset \mathbb{R}^d$ relative to an
arbitrary norm $\Vert\cdot\Vert$. We connect this with the problem of external
regret minimization for online linear optimization, showing that if it is
possible to guarantee $O(\sqrt{\rho T})$ worst-case regret after $T$ rounds
when actions are drawn from $\mathcal{P}$ and losses are drawn from the dual
$\Vert \cdot \Vert_*$ unit norm ball, then it is also possible to obtain
$\epsilon$-calibrated forecasts after $T = \exp(O(\rho /\epsilon^2))$ rounds.
When $\mathcal{P}$ is the $d$-dimensional simplex and $\Vert \cdot \Vert$ is
the $\ell_1$-norm, the existence of $O(\sqrt{T\log d})$-regret algorithms for
learning with experts implies that it is possible to obtain
$\epsilon$-calibrated forecasts after $T = \exp(O(\log{d}/\epsilon^2)) =
d^{O(1/\epsilon^2)}$ rounds, recovering a recent result of Peng (2025).
  Interestingly, our algorithm obtains this guarantee without requiring access
to any online linear optimization subroutine or knowledge of the optimal rate
$\rho$ -- in fact, our algorithm is identical for every setting of
$\mathcal{P}$ and $\Vert \cdot \Vert$. Instead, we show that the optimal
regularizer for the above OLO problem can be used to upper bound the above
calibration error by a swap regret, which we then minimize by running the
recent TreeSwap algorithm with Follow-The-Leader as a subroutine.
  Finally, we prove that any online calibration algorithm that guarantees
$\epsilon T$ $\ell_1$-calibration error over the $d$-dimensional simplex
requires $T \geq \exp(\mathrm{poly}(1/\epsilon))$ (assuming $d \geq
\mathrm{poly}(1/\epsilon)$). This strengthens the corresponding
$d^{\Omega(\log{1/\epsilon})}$ lower bound of Peng, and shows that an
exponential dependence on $1/\epsilon$ is necessary.

### 7. [Strong Low Degree Hardness for the Number Partitioning Problem](http://arxiv.org/pdf/2505.20607v1)

Authors: Rushil Mallarapu, Mark Sellke

In the number partitioning problem (NPP) one aims to partition a given set of
$N$ real numbers into two subsets with approximately equal sum. The NPP is a
well-studied optimization problem and is famous for possessing a
statistical-to-computational gap: when the $N$ numbers to be partitioned are
i.i.d. standard gaussian, the optimal discrepancy is $2^{-\Theta(N)}$ with high
probability, but the best known polynomial-time algorithms only find solutions
with a discrepancy of $2^{-\Theta(\log^2 N)}$. This gap is a common feature in
optimization problems over random combinatorial structures, and indicates the
need for a study that goes beyond worst-case analysis.
  We provide evidence of a nearly tight algorithmic barrier for the number
partitioning problem. Namely we consider the family of low coordinate degree
algorithms (with randomized rounding into the Boolean cube), and show that
degree $D$ algorithms fail to solve the NPP to accuracy beyond $2^{-\widetilde
O(D)}$. According to the low degree heuristic, this suggests that simple
brute-force search algorithms are nearly unimprovable, given any allotted
runtime between polynomial and exponential in $N$. Our proof combines the
isolation of solutions in the landscape with a conditional form of the overlap
gap property: given a good solution to an NPP instance, slightly noising the
NPP instance typically leaves no good solutions near the original one. In fact
our analysis applies whenever the $N$ numbers to be partitioned are independent
with uniformly bounded density.

### 8. [Scheduling with Uncertain Holding Costs and its Application to Content Moderation](http://arxiv.org/pdf/2505.21331v1)

Authors: Caner Gocmen, Thodoris Lykouris, Deeksha Sinha, Wentao Weng

In content moderation for social media platforms, the cost of delaying the
review of a content is proportional to its view trajectory, which fluctuates
and is apriori unknown. Motivated by such uncertain holding costs, we consider
a queueing model where job states evolve based on a Markov chain with
state-dependent instantaneous holding costs. We demonstrate that in the
presence of such uncertain holding costs, the two canonical algorithmic
principles, instantaneous-cost ($c\mu$-rule) and expected-remaining-cost
($c\mu/\theta$-rule), are suboptimal. By viewing each job as a Markovian
ski-rental problem, we develop a new index-based algorithm,
Opportunity-adjusted Remaining Cost (OaRC), that adjusts to the opportunity of
serving jobs in the future when uncertainty partly resolves. We show that the
regret of OaRC scales as $\tilde{O}(L^{1.5}\sqrt{N})$, where $L$ is the maximum
length of a job's holding cost trajectory and $N$ is the system size. This
regret bound shows that OaRC achieves asymptotic optimality when the system
size $N$ scales to infinity. Moreover, its regret is independent of the
state-space size, which is a desirable property when job states contain
contextual information. We corroborate our results with an extensive simulation
study based on two holding cost patterns (online ads and user-generated
content) that arise in content moderation for social media platforms. Our
simulations based on synthetic and real datasets demonstrate that OaRC
consistently outperforms existing practice, which is based on the two canonical
algorithmic principles.

### Emerging Technologies

### 1. [Multi-VQC: A Novel QML Approach for Enhancing Healthcare Classification](http://arxiv.org/pdf/2505.20797v1)

Authors: Antonio Tudisco, Deborah Volpe, Giovanna Turvani

Accurate and reliable diagnosis of diseases is crucial in enabling timely
medical treatment and enhancing patient survival rates. In recent years,
Machine Learning has revolutionized diagnostic practices by creating
classification models capable of identifying diseases. However, these
classification problems often suffer from significant class imbalances, which
can inhibit the effectiveness of traditional models. Therefore, the interest in
Quantum models has arisen, driven by the captivating promise of overcoming the
limitations of the classical counterpart thanks to their ability to express
complex patterns by mapping data in a higher-dimensional computational space.

### 2. [Quantum Machine Learning in Healthcare: Evaluating QNN and QSVM Models](http://arxiv.org/pdf/2505.20804v1)

Authors: Antonio Tudisco, Deborah Volpe, Giovanna Turvani

Effective and accurate diagnosis of diseases such as cancer, diabetes, and
heart failure is crucial for timely medical intervention and improving patient
survival rates. Machine learning has revolutionized diagnostic methods in
recent years by developing classification models that detect diseases based on
selected features. However, these classification tasks are often highly
imbalanced, limiting the performance of classical models. Quantum models offer
a promising alternative, exploiting their ability to express complex patterns
by operating in a higher-dimensional computational space through superposition
and entanglement. These unique properties make quantum models potentially more
effective in addressing the challenges of imbalanced datasets. This work
evaluates the potential of quantum classifiers in healthcare, focusing on
Quantum Neural Networks (QNNs) and Quantum Support Vector Machines (QSVMs),
comparing them with popular classical models. The study is based on three
well-known healthcare datasets -- Prostate Cancer, Heart Failure, and Diabetes.
The results indicate that QSVMs outperform QNNs across all datasets due to
their susceptibility to overfitting. Furthermore, quantum models prove the
ability to overcome classical models in scenarios with high dataset imbalance.
Although preliminary, these findings highlight the potential of quantum models
in healthcare classification tasks and lead the way for further research in
this domain.

### 3. [A Structured Unplugged Approach for Foundational AI Literacy in Primary Education](http://arxiv.org/pdf/2505.21398v1)

Authors: Maria Cristina Carrisi, Mirko Marras, Sara Vergallo

Younger generations are growing up in a world increasingly shaped by
intelligent technologies, making early AI literacy crucial for developing the
skills to critically understand and navigate them. However, education in this
field often emphasizes tool-based learning, prioritizing usage over
understanding the underlying concepts. This lack of knowledge leaves
non-experts, especially children, prone to misconceptions, unrealistic
expectations, and difficulties in recognizing biases and stereotypes. In this
paper, we propose a structured and replicable teaching approach that fosters
foundational AI literacy in primary students, by building upon core
mathematical elements closely connected to and of interest in primary
curricula, to strengthen conceptualization, data representation, classification
reasoning, and evaluation of AI. To assess the effectiveness of our approach,
we conducted an empirical study with thirty-one fifth-grade students across two
classes, evaluating their progress through a post-test and a satisfaction
survey. Our results indicate improvements in terminology understanding and
usage, features description, logical reasoning, and evaluative skills, with
students showing a deeper comprehension of decision-making processes and their
limitations. Moreover, the approach proved engaging, with students particularly
enjoying activities that linked AI concepts to real-world reasoning. Materials:
https://github.com/tail-unica/ai-literacy-primary-ed.

### Formal Languages and Automata Theory

### 1. [INTERLEAVE: A Faster Symbolic Algorithm for Maximal End Component Decomposition](http://arxiv.org/pdf/2505.20748v1)

Authors: Suguman Bansal, Ramneet Singh

This paper presents a novel symbolic algorithm for the Maximal End Component
(MEC) decomposition of a Markov Decision Process (MDP). The key idea behind our
algorithm INTERLEAVE is to interleave the computation of Strongly Connected
Components (SCCs) with eager elimination of redundant state-action pairs,
rather than performing these computations sequentially as done by existing
state-of-the-art algorithms. Even though our approach has the same complexity
as prior works, an empirical evaluation of INTERLEAVE on the standardized
Quantitative Verification Benchmark Set demonstrates that it solves 19 more
benchmarks (out of 379) than the closest previous algorithm. On the 149
benchmarks that prior approaches can solve, we demonstrate a 3.81x average
speedup in runtime.

### Graphics

### 1. [Progressively Projected Newton's Method](http://arxiv.org/pdf/2505.21013v1)

Authors: José Antonio Fernández-Fernández, Fabian Löschner, Jan Bender

Newton's Method is widely used to find the solution of complex non-linear
simulation problems in Computer Graphics. To guarantee a descent direction, it
is common practice to clamp the negative eigenvalues of each element Hessian
prior to assembly - a strategy known as Projected Newton (PN) - but this
perturbation often hinders convergence.
  In this work, we observe that projecting only a small subset of element
Hessians is sufficient to secure a descent direction. Building on this insight,
we introduce Progressively Projected Newton (PPN), a novel variant of Newton's
Method that uses the current iterate residual to cheaply determine the subset
of element Hessians to project. The global Hessian thus remains closer to its
original form, reducing both the number of Newton iterations and the amount of
required eigen-decompositions.
  We compare PPN with PN and Project-on-Demand Newton (PDN) in a comprehensive
set of experiments covering contact-free and contact-rich deformables
(including large stiffness and mass ratios), co-dimensional, and rigid-body
simulations, and a range of time step sizes, tolerances and resolutions. PPN
consistently performs fewer than 10% of the projections required by PN or PDN
and, in the vast majority of cases, converges in fewer Newton iterations, which
makes PPN the fastest solver in our benchmark. The most notable exceptions are
simulations with very large time steps and quasistatics, where PN remains a
better choice.

### 2. [Hand Shadow Art: A Differentiable Rendering Perspective](http://arxiv.org/pdf/2505.21252v1)

Authors: Aalok Gangopadhyay, Prajwal Singh, Ashish Tiwari, Shanmuganathan Raman

Shadow art is an exciting form of sculptural art that produces captivating
artistic effects through the 2D shadows cast by 3D shapes. Hand shadows, also
known as shadow puppetry or shadowgraphy, involve creating various shapes and
figures using your hands and fingers to cast meaningful shadows on a wall. In
this work, we propose a differentiable rendering-based approach to deform hand
models such that they cast a shadow consistent with a desired target image and
the associated lighting configuration. We showcase the results of shadows cast
by a pair of two hands and the interpolation of hand poses between two desired
shadow images. We believe that this work will be a useful tool for the graphics
community.

### 3. [CityGo: Lightweight Urban Modeling and Rendering with Proxy Buildings and Residual Gaussians](http://arxiv.org/pdf/2505.21041v1)

Authors: Weihang Liu, Yuhui Zhong, Yuke Li, Xi Chen, Jiadi Cui, Honglong Zhang, Lan Xu, Xin Lou, Yujiao Shi, Jingyi Yu, Yingliang Zhang

Accurate and efficient modeling of large-scale urban scenes is critical for
applications such as AR navigation, UAV based inspection, and smart city
digital twins. While aerial imagery offers broad coverage and complements
limitations of ground-based data, reconstructing city-scale environments from
such views remains challenging due to occlusions, incomplete geometry, and high
memory demands. Recent advances like 3D Gaussian Splatting (3DGS) improve
scalability and visual quality but remain limited by dense primitive usage,
long training times, and poor suit ability for edge devices. We propose CityGo,
a hybrid framework that combines textured proxy geometry with residual and
surrounding 3D Gaussians for lightweight, photorealistic rendering of urban
scenes from aerial perspectives. Our approach first extracts compact building
proxy meshes from MVS point clouds, then uses zero order SH Gaussians to
generate occlusion-free textures via image-based rendering and back-projection.
To capture high-frequency details, we introduce residual Gaussians placed based
on proxy-photo discrepancies and guided by depth priors. Broader urban context
is represented by surrounding Gaussians, with importance-aware downsampling
applied to non-critical regions to reduce redundancy. A tailored optimization
strategy jointly refines proxy textures and Gaussian parameters, enabling
real-time rendering of complex urban scenes on mobile GPUs with significantly
reduced training and memory requirements. Extensive experiments on real-world
aerial datasets demonstrate that our hybrid representation significantly
reduces training time, achieving on average 1.4x speedup, while delivering
comparable visual fidelity to pure 3D Gaussian Splatting approaches.
Furthermore, CityGo enables real-time rendering of large-scale urban scenes on
mobile consumer GPUs, with substantially reduced memory usage and energy
consumption.

### 4. [IKMo: Image-Keyframed Motion Generation with Trajectory-Pose Conditioned Motion Diffusion Model](http://arxiv.org/pdf/2505.21146v1)

Authors: Yang Zhao, Yan Zhang, Xubo Yang

Existing human motion generation methods with trajectory and pose inputs
operate global processing on both modalities, leading to suboptimal outputs. In
this paper, we propose IKMo, an image-keyframed motion generation method based
on the diffusion model with trajectory and pose being decoupled. The trajectory
and pose inputs go through a two-stage conditioning framework. In the first
stage, the dedicated optimization module is applied to refine inputs. In the
second stage, trajectory and pose are encoded via a Trajectory Encoder and a
Pose Encoder in parallel. Then, motion with high spatial and semantic fidelity
is guided by a motion ControlNet, which processes the fused trajectory and pose
data. Experiment results based on HumanML3D and KIT-ML datasets demonstrate
that the proposed method outperforms state-of-the-art on all metrics under
trajectory-keyframe constraints. In addition, MLLM-based agents are implemented
to pre-process model inputs. Given texts and keyframe images from users, the
agents extract motion descriptions, keyframe poses, and trajectories as the
optimized inputs into the motion generation model. We conducts a user study
with 10 participants. The experiment results prove that the MLLM-based agents
pre-processing makes generated motion more in line with users' expectation. We
believe that the proposed method improves both the fidelity and controllability
of motion generation by the diffusion model.

### 5. [efunc: An Efficient Function Representation without Neural Networks](http://arxiv.org/pdf/2505.21319v1)

Authors: Biao Zhang, Peter Wonka

Function fitting/approximation plays a fundamental role in computer graphics
and other engineering applications. While recent advances have explored neural
networks to address this task, these methods often rely on architectures with
many parameters, limiting their practical applicability. In contrast, we pursue
high-quality function approximation using parameter-efficient representations
that eliminate the dependency on neural networks entirely. We first propose a
novel framework for continuous function modeling. Most existing works can be
formulated using this framework. We then introduce a compact function
representation, which is based on polynomials interpolated using radial basis
functions, bypassing both neural networks and complex/hierarchical data
structures. We also develop memory-efficient CUDA-optimized algorithms that
reduce computational time and memory consumption to less than 10% compared to
conventional automatic differentiation frameworks. Finally, we validate our
representation and optimization pipeline through extensive experiments on 3D
signed distance functions (SDFs). The proposed representation achieves
comparable or superior performance to state-of-the-art techniques (e.g.,
octree/hash-grid techniques) with significantly fewer parameters.

### 6. [CoDA: Coordinated Diffusion Noise Optimization for Whole-Body Manipulation of Articulated Objects](http://arxiv.org/pdf/2505.21437v1)

Authors: Huaijin Pi, Zhi Cen, Zhiyang Dou, Taku Komura

Synthesizing whole-body manipulation of articulated objects, including body
motion, hand motion, and object motion, is a critical yet challenging task with
broad applications in virtual humans and robotics. The core challenges are
twofold. First, achieving realistic whole-body motion requires tight
coordination between the hands and the rest of the body, as their movements are
interdependent during manipulation. Second, articulated object manipulation
typically involves high degrees of freedom and demands higher precision, often
requiring the fingers to be placed at specific regions to actuate movable
parts. To address these challenges, we propose a novel coordinated diffusion
noise optimization framework. Specifically, we perform noise-space optimization
over three specialized diffusion models for the body, left hand, and right
hand, each trained on its own motion dataset to improve generalization.
Coordination naturally emerges through gradient flow along the human kinematic
chain, allowing the global body posture to adapt in response to hand motion
objectives with high fidelity. To further enhance precision in hand-object
interaction, we adopt a unified representation based on basis point sets (BPS),
where end-effector positions are encoded as distances to the same BPS used for
object geometry. This unified representation captures fine-grained spatial
relationships between the hand and articulated object parts, and the resulting
trajectories serve as targets to guide the optimization of diffusion noise,
producing highly accurate interaction motion. We conduct extensive experiments
demonstrating that our method outperforms existing approaches in motion quality
and physical plausibility, and enables various capabilities such as object pose
control, simultaneous walking and manipulation, and whole-body generation from
hand-only data.

### 7. [Be Decisive: Noise-Induced Layouts for Multi-Subject Generation](http://arxiv.org/pdf/2505.21488v1)

Authors: Omer Dahary, Yehonathan Cohen, Or Patashnik, Kfir Aberman, Daniel Cohen-Or

Generating multiple distinct subjects remains a challenge for existing
text-to-image diffusion models. Complex prompts often lead to subject leakage,
causing inaccuracies in quantities, attributes, and visual features. Preventing
leakage among subjects necessitates knowledge of each subject's spatial
location. Recent methods provide these spatial locations via an external layout
control. However, enforcing such a prescribed layout often conflicts with the
innate layout dictated by the sampled initial noise, leading to misalignment
with the model's prior. In this work, we introduce a new approach that predicts
a spatial layout aligned with the prompt, derived from the initial noise, and
refines it throughout the denoising process. By relying on this noise-induced
layout, we avoid conflicts with externally imposed layouts and better preserve
the model's prior. Our method employs a small neural network to predict and
refine the evolving noise-induced layout at each denoising step, ensuring clear
boundaries between subjects while maintaining consistency. Experimental results
show that this noise-aligned strategy achieves improved text-image alignment
and more stable multi-subject generation compared to existing layout-guided
techniques, while preserving the rich diversity of the model's original
distribution.

### 8. [Structure from Collision](http://arxiv.org/pdf/2505.21335v1)

Authors: Takuhiro Kaneko

Recent advancements in neural 3D representations, such as neural radiance
fields (NeRF) and 3D Gaussian splatting (3DGS), have enabled the accurate
estimation of 3D structures from multiview images. However, this capability is
limited to estimating the visible external structure, and identifying the
invisible internal structure hidden behind the surface is difficult. To
overcome this limitation, we address a new task called Structure from Collision
(SfC), which aims to estimate the structure (including the invisible internal
structure) of an object from appearance changes during collision. To solve this
problem, we propose a novel model called SfC-NeRF that optimizes the invisible
internal structure of an object through a video sequence under physical,
appearance (i.e., visible external structure)-preserving, and keyframe
constraints. In particular, to avoid falling into undesirable local optima
owing to its ill-posed nature, we propose volume annealing; that is, searching
for global optima by repeatedly reducing and expanding the volume. Extensive
experiments on 115 objects involving diverse structures (i.e., various cavity
shapes, locations, and sizes) and material properties revealed the properties
of SfC and demonstrated the effectiveness of the proposed SfC-NeRF.

### Computer Science and Game Theory

### 1. [When to Deceive: A Cross-Layer Stackelberg Game Framework for Strategic Timing of Cyber Deception](http://arxiv.org/pdf/2505.21244v1)

Authors: Ya-Ting Yang, Quanyan Zhu

Cyber deception is an emerging proactive defense strategy to counter
increasingly sophisticated attacks such as Advanced Persistent Threats (APTs)
by misleading and distracting attackers from critical assets. However, since
deception techniques incur costs and may lose effectiveness over time,
defenders must strategically time and select them to adapt to the dynamic
system and the attacker's responses. In this study, we propose a Stackelberg
game-based framework to design strategic timing for cyber deception: the lower
tactical layer (follower) captures the evolving attacker-defender dynamics
under a given deception through a one-sided information Markov game, while the
upper strategic layer (leader) employs a stopping-time decision process to
optimize the timing and selection of deception techniques. We also introduce a
computational algorithm that integrates dynamic programming and belief-state
updates to account for the attacker's adaptive behavior and limited deception
resources. Numerical experiments validate the framework, showing that
strategically timed deceptions can enhance the defender's expected utility and
reduce the risk of asset compromise compared to baseline strategies.

### 2. [PACT: A Contract-Theoretic Framework for Pricing Agentic AI Services Powered by Large Language Models](http://arxiv.org/pdf/2505.21286v1)

Authors: Ya-Ting Yang, Quanyan Zhu

Agentic AI, often powered by large language models (LLMs), is becoming
increasingly popular and adopted to support autonomous reasoning,
decision-making, and task execution across various domains. While agentic AI
holds great promise, its deployment as services for easy access raises critical
challenges in pricing, due to high infrastructure and computation costs,
multi-dimensional and task-dependent Quality of Service (QoS), and growing
concerns around liability in high-stakes applications. In this work, we propose
PACT, a Pricing framework for cloud-based Agentic AI services through a
Contract-Theoretic approach, which models QoS along both objective (e.g.,
response time) and subjective (e.g., user satisfaction) dimensions. PACT
accounts for computational, infrastructure, and potential liability costs for
the service provider, while ensuring incentive compatibility and individual
rationality for the user under information asymmetry. Through contract-based
selection, users receive tailored service offerings aligned with their needs.
Numerical evaluations demonstrate that PACT improves QoS alignment between
users and providers and offers a scalable, liable approach to pricing agentic
AI services in the future.

### 3. [Fundamental Limits of Game-Theoretic LLM Alignment: Smith Consistency and Preference Matching](http://arxiv.org/pdf/2505.20627v1)

Authors: Zhekun Shi, Kaizhao Liu, Qi Long, Weijie J. Su, Jiancong Xiao

Nash Learning from Human Feedback is a game-theoretic framework for aligning
large language models (LLMs) with human preferences by modeling learning as a
two-player zero-sum game. However, using raw preference as the payoff in the
game highly limits the potential of the game-theoretic LLM alignment framework.
In this paper, we systematically study using what choices of payoff based on
the pairwise human preferences can yield desirable alignment properties. We
establish necessary and sufficient conditions for Condorcet consistency,
diversity through mixed strategies, and Smith consistency. These results
provide a theoretical foundation for the robustness of game-theoretic LLM
alignment. Further, we show the impossibility of preference matching -- i.e.,
no smooth and learnable mappings of pairwise preferences can guarantee a unique
Nash equilibrium that matches a target policy, even under standard assumptions
like the Bradley-Terry-Luce model. This result highlights the fundamental
limitation of game-theoretic LLM alignment.

### 4. [Union Shapley Value: Quantifying Group Impact via Collective Removal](http://arxiv.org/pdf/2505.21122v1)

Authors: Piotr Kępczyński, Oskar Skibski

We perform a comprehensive analysis of extensions of the Shapley value to
groups. We propose a new, natural extension called the Union Shapley Value,
which assesses a group's contribution by examining the impact of its removal
from the game. This intuition is formalized through two axiomatic
characterizations, closely related to existing axiomatizations of the Shapley
value. Furthermore, we characterize the class of group semivalues and identify
a dual approach that measures synergy instead of the value of a coalition. Our
analysis reveals a novel connection between several group values previously
proposed in the literature.

### 5. [Distributed equilibrium seeking in aggregative games: linear convergence under singular perturbations lens](http://arxiv.org/pdf/2505.21386v1)

Authors: Guido Carnevale, Filippo Fabiani, Filiberto Fele, Kostas Margellos, Giuseppe Notarstefano

We present a fully-distributed algorithm for Nash equilibrium seeking in
aggregative games over networks. The proposed scheme endows each agent with a
gradient-based scheme equipped with a tracking mechanism to locally reconstruct
the aggregative variable, which is not available to the agents. We show that
our method falls into the framework of singularly perturbed systems, as it
involves the interconnection between a fast subsystem - the global information
reconstruction dynamics - with a slow one concerning the optimization of the
local strategies. This perspective plays a key role in analyzing the scheme
with a constant stepsize, and in proving its linear convergence to the Nash
equilibrium in strongly monotone games with local constraints. By exploiting
the flexibility of our aggregative variable definition (not necessarily the
arithmetic average of the agents' strategy), we show the efficacy of our
algorithm on a realistic voltage support case study for the smart grid.

### 6. [A Framework for Adversarial Analysis of Decision Support Systems Prior to Deployment](http://arxiv.org/pdf/2505.21414v1)

Authors: Brett Bissey, Kyle Gatesman, Walker Dimon, Mohammad Alam, Luis Robaina, Joseph Weissman

This paper introduces a comprehensive framework designed to analyze and
secure decision-support systems trained with Deep Reinforcement Learning (DRL),
prior to deployment, by providing insights into learned behavior patterns and
vulnerabilities discovered through simulation. The introduced framework aids in
the development of precisely timed and targeted observation perturbations,
enabling researchers to assess adversarial attack outcomes within a strategic
decision-making context. We validate our framework, visualize agent behavior,
and evaluate adversarial outcomes within the context of a custom-built
strategic game, CyberStrike. Utilizing the proposed framework, we introduce a
method for systematically discovering and ranking the impact of attacks on
various observation indices and time-steps, and we conduct experiments to
evaluate the transferability of adversarial attacks across agent architectures
and DRL training algorithms. The findings underscore the critical need for
robust adversarial defense mechanisms to protect decision-making policies in
high-stakes environments.

### 7. [High-Dimensional Calibration from Swap Regret](http://arxiv.org/pdf/2505.21460v1)

Authors: Maxwell Fishelson, Noah Golowich, Mehryar Mohri, Jon Schneider

We study the online calibration of multi-dimensional forecasts over an
arbitrary convex set $\mathcal{P} \subset \mathbb{R}^d$ relative to an
arbitrary norm $\Vert\cdot\Vert$. We connect this with the problem of external
regret minimization for online linear optimization, showing that if it is
possible to guarantee $O(\sqrt{\rho T})$ worst-case regret after $T$ rounds
when actions are drawn from $\mathcal{P}$ and losses are drawn from the dual
$\Vert \cdot \Vert_*$ unit norm ball, then it is also possible to obtain
$\epsilon$-calibrated forecasts after $T = \exp(O(\rho /\epsilon^2))$ rounds.
When $\mathcal{P}$ is the $d$-dimensional simplex and $\Vert \cdot \Vert$ is
the $\ell_1$-norm, the existence of $O(\sqrt{T\log d})$-regret algorithms for
learning with experts implies that it is possible to obtain
$\epsilon$-calibrated forecasts after $T = \exp(O(\log{d}/\epsilon^2)) =
d^{O(1/\epsilon^2)}$ rounds, recovering a recent result of Peng (2025).
  Interestingly, our algorithm obtains this guarantee without requiring access
to any online linear optimization subroutine or knowledge of the optimal rate
$\rho$ -- in fact, our algorithm is identical for every setting of
$\mathcal{P}$ and $\Vert \cdot \Vert$. Instead, we show that the optimal
regularizer for the above OLO problem can be used to upper bound the above
calibration error by a swap regret, which we then minimize by running the
recent TreeSwap algorithm with Follow-The-Leader as a subroutine.
  Finally, we prove that any online calibration algorithm that guarantees
$\epsilon T$ $\ell_1$-calibration error over the $d$-dimensional simplex
requires $T \geq \exp(\mathrm{poly}(1/\epsilon))$ (assuming $d \geq
\mathrm{poly}(1/\epsilon)$). This strengthens the corresponding
$d^{\Omega(\log{1/\epsilon})}$ lower bound of Peng, and shows that an
exponential dependence on $1/\epsilon$ is necessary.

### 8. [Scheduling with Uncertain Holding Costs and its Application to Content Moderation](http://arxiv.org/pdf/2505.21331v1)

Authors: Caner Gocmen, Thodoris Lykouris, Deeksha Sinha, Wentao Weng

In content moderation for social media platforms, the cost of delaying the
review of a content is proportional to its view trajectory, which fluctuates
and is apriori unknown. Motivated by such uncertain holding costs, we consider
a queueing model where job states evolve based on a Markov chain with
state-dependent instantaneous holding costs. We demonstrate that in the
presence of such uncertain holding costs, the two canonical algorithmic
principles, instantaneous-cost ($c\mu$-rule) and expected-remaining-cost
($c\mu/\theta$-rule), are suboptimal. By viewing each job as a Markovian
ski-rental problem, we develop a new index-based algorithm,
Opportunity-adjusted Remaining Cost (OaRC), that adjusts to the opportunity of
serving jobs in the future when uncertainty partly resolves. We show that the
regret of OaRC scales as $\tilde{O}(L^{1.5}\sqrt{N})$, where $L$ is the maximum
length of a job's holding cost trajectory and $N$ is the system size. This
regret bound shows that OaRC achieves asymptotic optimality when the system
size $N$ scales to infinity. Moreover, its regret is independent of the
state-space size, which is a desirable property when job states contain
contextual information. We corroborate our results with an extensive simulation
study based on two holding cost patterns (online ads and user-generated
content) that arise in content moderation for social media platforms. Our
simulations based on synthetic and real datasets demonstrate that OaRC
consistently outperforms existing practice, which is based on the two canonical
algorithmic principles.

### Human-Computer Interaction

### 1. [What Shapes Writers' Decisions to Disclose AI Use?](http://arxiv.org/pdf/2505.20727v1)

Authors: Jingchao Fang, Mina Lee

Have you ever read a blog or social media post and suspected that it was
written--at least in part--by artificial intelligence (AI)? While transparently
acknowledging contributors to writing is generally valued, why some writers
choose to disclose or withhold AI involvement remains unclear. In this work, we
ask what factors shape writers' decisions to disclose their AI use as a
starting point to effectively advocate for transparency. To shed light on this
question, we synthesize study findings and theoretical frameworks in human-AI
interaction and behavioral science. Concretely, we identify and curate a list
of factors that could affect writers' decisions regarding disclosure for
human-AI co-created content.

### 2. [Describe Me Something You Do Not Remember - Challenges and Risks of Exposure Design Using Generative Artificial Intelligence for Therapy of Complex Post-traumatic Disorder](http://arxiv.org/pdf/2505.20796v1)

Authors: Annalisa Degenhard, Stefan Tschöke, Michael Rietzler, Enrico Rukzio

Post-traumatic stress disorder (PTSD) is associated with sudden,
uncontrollable, and intense flashbacks of traumatic memories. Trauma exposure
psychotherapy has proven effective in reducing the severity of trauma-related
symptoms. It involves controlled recall of traumatic memories to train coping
mechanisms for flashbacks and enable autobiographical integration of
distressing experiences. In particular, exposure to visualizations of these
memories supports successful recall. Although this approach is effective for
various trauma types, it remains available for only a few. This is due to the
lack of cost-efficient solutions for creating individualized exposure
visualizations. This issue is particularly relevant for the treatment of
Complex PTSD (CPTSD), where traumatic memories are highly individual and
generic visualizations do not meet therapeutic needs. Generative Artificial
Intelligence (GAI) offers a flexible and cost-effective alternative. GAI
enables the creation of individualized exposure visualizations during therapy
and, for the first time, allows patients to actively participate in the
visualization process. While GAI opens new therapeutic perspectives and may
improve access to trauma therapy, especially for CPTSD, it also introduces
significant challenges and risks. The extreme uncertainty and lack of control
that define both CPTSD and GAI raise concerns about feasibility and safety. To
support safe and effective three-way communication, it is essential to
understand the roles of patient, system, and therapist in exposure
visualization and how each can contribute to safety. This paper outlines
perspectives, challenges, and risks associated with the use of GAI in trauma
therapy, with a focus on CPTSD.

### 3. [Imago Obscura: An Image Privacy AI Co-pilot to Enable Identification and Mitigation of Risks](http://arxiv.org/pdf/2505.20916v1)

Authors: Kyzyl Monteiro, Yuchen Wu, Sauvik Das

Users often struggle to navigate the privacy / publicity boundary in sharing
images online: they may lack awareness of image privacy risks and/or the
ability to apply effective mitigation strategies. To address this challenge, we
introduce and evaluate Imago Obscura, an AI-powered, image-editing copilot that
enables users to identify and mitigate privacy risks with images they intend to
share. Driven by design requirements from a formative user study with 7
image-editing experts, Imago Obscura enables users to articulate their
image-sharing intent and privacy concerns. The system uses these inputs to
surface contextually pertinent privacy risks, and then recommends and
facilitates application of a suite of obfuscation techniques found to be
effective in prior literature -- e.g., inpainting, blurring, and generative
content replacement. We evaluated Imago Obscura with 15 end-users in a lab
study and found that it greatly improved users' awareness of image privacy
risks and their ability to address those risks, allowing them to make more
informed sharing decisions.

### 4. [Dynamic Vision from EEG Brain Recordings: How much does EEG know?](http://arxiv.org/pdf/2505.21385v1)

Authors: Prajwal Singh, Anupam Sharma, Pankaj Pandey, Krishna Miyapuram, Shanmuganathan Raman

Reconstructing and understanding dynamic visual information (video) from
brain EEG recordings is challenging due to the non-stationary nature of EEG
signals, their low signal-to-noise ratio (SNR), and the limited availability of
EEG-Video stimulus datasets. Most recent studies have focused on reconstructing
static images from EEG recordings. In this work, we propose a framework to
reconstruct dynamic visual stimuli from EEG data and conduct an in-depth study
of the information encoded in EEG signals. Our approach first trains a feature
extraction network using a triplet-based contrastive learning strategy within
an EEG-video generation framework. The extracted EEG features are then used for
video synthesis with a modified StyleGAN-ADA, which incorporates temporal
information as conditioning. Additionally, we analyze how different brain
regions contribute to processing dynamic visual stimuli. Through several
empirical studies, we evaluate the effectiveness of our framework and
investigate how much dynamic visual information can be inferred from EEG
signals. The inferences we derive through our extensive studies would be of
immense value to future research on extracting visual dynamics from EEG.

### 5. [Institutionalizing Folk Theories of Algorithms: How Multi-Channel Networks (MCNs) Govern Algorithmic Labor in Chinese Live-Streaming Industry](http://arxiv.org/pdf/2505.20623v1)

Authors: Qing Xiao, Rongyi Chen, Jingjia Xiao, Tianyang Fu, Alice Qian Zhang, Xianzhe Fan, Bingbing Zhang, Zhicong Lu, Hong Shen

As algorithmic systems increasingly structure platform labor, workers often
rely on informal "folk theories", experience-based beliefs about how algorithms
work, to navigate opaque and unstable algorithmic environments. Prior research
has largely treated these theories as bottom-up, peer-driven strategies for
coping with algorithmic opacity and uncertainty. In this study, we shift
analytical attention to intermediary organizations and examine how folk
theories of algorithms can be institutionally constructed and operationalized
by those organizations as tools of labor management. Drawing on nine months of
ethnographic fieldwork and 37 interviews with live-streamers and staff at
Multi-Channel Networks (MCNs) in China, we show that MCNs develop and circulate
dual algorithmic theories: internally, they acknowledge the volatility of
platform systems and adopt probabilistic strategies to manage risk; externally,
they promote simplified, prescriptive theories portraying the algorithm as
transparent, fair, and responsive to individual effort. They have further
operationalize those folk theories for labor management, encouraging streamers
to self-discipline and invest in equipment, training, and routines, while
absolving MCNs of accountability. We contribute to CSCW and platform labor
literature by demonstrating how informal algorithmic knowledge, once
institutionalized, can become infrastructures of soft control -- shaping not
only how workers interpret platform algorithms, but also how their labor is
structured, moralized and governed.

### 6. [Supervised Contrastive Learning for Ordinal Engagement Measurement](http://arxiv.org/pdf/2505.20676v1)

Authors: Sadaf Safa, Ali Abedi, Shehroz S. Khan

Student engagement plays a crucial role in the successful delivery of
educational programs. Automated engagement measurement helps instructors
monitor student participation, identify disengagement, and adapt their teaching
strategies to enhance learning outcomes effectively. This paper identifies two
key challenges in this problem: class imbalance and incorporating order into
engagement levels rather than treating it as mere categories. Then, a novel
approach to video-based student engagement measurement in virtual learning
environments is proposed that utilizes supervised contrastive learning for
ordinal classification of engagement. Various affective and behavioral features
are extracted from video samples and utilized to train ordinal classifiers
within a supervised contrastive learning framework (with a sequential
classifier as the encoder). A key step involves the application of diverse
time-series data augmentation techniques to these feature vectors, enhancing
model training. The effectiveness of the proposed method was evaluated using a
publicly available dataset for engagement measurement, DAiSEE, containing
videos of students who participated in virtual learning programs. The results
demonstrate the robust ability of the proposed method for the classification of
the engagement level. This approach promises a significant contribution to
understanding and enhancing student engagement in virtual learning
environments.

### 7. [System-driven Cloud Architecture Design Support with Structured State Management and Guided Decision Assistance](http://arxiv.org/pdf/2505.20701v1)

Authors: Ryosuke Kohita, Akira Kasuga

Cloud architecture design is a complex process requiring both technical
expertise and architectural knowledge to develop solutions from frequently
ambiguous requirements. We present CloudArchitectBuddy, a system-driven cloud
architecture design support application with two key mechanisms: (1) structured
state management that enhances design understanding through explicit
representation of requirements and architectural decisions, and (2) guided
decision assistance that facilitates design progress through proactive
verification and requirement refinement. Our study with 16 industry
practitioners showed that while our approach achieved comparable design quality
to a chat interface, participants rated our system higher for usability and
appreciated its ability to help understand architectural relationships and
identify missing requirements. However, participants also expressed a need for
user-initiated interactions where they could freely provide design instructions
and engage in detailed discussions with LLMs. These results suggest that
integrating a chat interface into our structured and guided workflow approach
would create a more practical solution, balancing systematic design support
with conversational flexibility for comprehensive cloud architecture
development.

### 8. [Automating eHMI Action Design with LLMs for Automated Vehicle Communication](http://arxiv.org/pdf/2505.20711v1)

Authors: Ding Xia, Xinyue Gui, Fan Gao, Dongyuan Li, Mark Colley, Takeo Igarashi

The absence of explicit communication channels between automated vehicles
(AVs) and other road users requires the use of external Human-Machine
Interfaces (eHMIs) to convey messages effectively in uncertain scenarios.
Currently, most eHMI studies employ predefined text messages and manually
designed actions to perform these messages, which limits the real-world
deployment of eHMIs, where adaptability in dynamic scenarios is essential.
Given the generalizability and versatility of large language models (LLMs),
they could potentially serve as automated action designers for the
message-action design task. To validate this idea, we make three contributions:
(1) We propose a pipeline that integrates LLMs and 3D renderers, using LLMs as
action designers to generate executable actions for controlling eHMIs and
rendering action clips. (2) We collect a user-rated Action-Design Scoring
dataset comprising a total of 320 action sequences for eight intended messages
and four representative eHMI modalities. The dataset validates that LLMs can
translate intended messages into actions close to a human level, particularly
for reasoning-enabled LLMs. (3) We introduce two automated raters, Action
Reference Score (ARS) and Vision-Language Models (VLMs), to benchmark 18 LLMs,
finding that the VLM aligns with human preferences yet varies across eHMI
modalities.

### 9. [Enhancing Wearable Tap Water Audio Detection through Subclass Annotation in the HD-Epic Dataset](http://arxiv.org/pdf/2505.20788v1)

Authors: Robin Burchard, Kristof Van Laerhoven

Wearable human activity recognition has been shown to benefit from the
inclusion of acoustic data, as the sounds around a person often contain
valuable context. However, due to privacy concerns, it is usually not ethically
feasible to record and save microphone data from the device, since the audio
could, for instance, also contain private conversations. Rather, the data
should be processed locally, which in turn requires processing power and
consumes energy on the wearable device. One special use case of contextual
information that can be utilized to augment special tasks in human activity
recognition is water flow detection, which can, e.g., be used to aid wearable
hand washing detection. We created a new label called tap water for the
recently released HD-Epic data set, creating 717 hand-labeled annotations of
tap water flow, based on existing annotations of the water class. We analyzed
the relation of tap water and water in the dataset and additionally trained and
evaluated two lightweight classifiers to evaluate the newly added label class,
showing that the new class can be learned more easily.

### 10. [Label Leakage in Federated Inertial-based Human Activity Recognition](http://arxiv.org/pdf/2505.20924v1)

Authors: Marius Bock, Maximilian Hopp, Kristof Van Laerhoven, Michael Moeller

While prior work has shown that Federated Learning updates can leak sensitive
information, label reconstruction attacks, which aim to recover input labels
from shared gradients, have not yet been examined in the context of Human
Activity Recognition (HAR). Given the sensitive nature of activity labels, this
study evaluates the effectiveness of state-of-the-art gradient-based label
leakage attacks on HAR benchmark datasets. Our findings show that the number of
activity classes, sampling strategy, and class imbalance are critical factors
influencing the extent of label leakage, with reconstruction accuracies
reaching up to 90% on two benchmark datasets, even for trained models.
Moreover, we find that Local Differential Privacy techniques such as gradient
noise and clipping offer only limited protection, as certain attacks still
reliably infer both majority and minority class labels. We conclude by offering
practical recommendations for the privacy-aware deployment of federated HAR
systems and identify open challenges for future research. Code to reproduce our
experiments is publicly available via github.com/mariusbock/leakage_har.

### Information Retrieval

### 1. [UQLegalAI@COLIEE2025: Advancing Legal Case Retrieval with Large Language Models and Graph Neural Networks](http://arxiv.org/pdf/2505.20743v1)

Authors: Yanran Tang, Ruihong Qiu, Zi Huang

Legal case retrieval plays a pivotal role in the legal domain by facilitating
the efficient identification of relevant cases, supporting legal professionals
and researchers to propose legal arguments and make informed decision-making.
To improve retrieval accuracy, the Competition on Legal Information Extraction
and Entailment (COLIEE) is held annually, offering updated benchmark datasets
for evaluation. This paper presents a detailed description of CaseLink, the
method employed by UQLegalAI, the second highest team in Task 1 of COLIEE 2025.
The CaseLink model utilises inductive graph learning and Global Case Graphs to
capture the intrinsic case connectivity to improve the accuracy of legal case
retrieval. Specifically, a large language model specialized in text embedding
is employed to transform legal texts into embeddings, which serve as the
feature representations of the nodes in the constructed case graph. A new
contrastive objective, incorporating a regularization on the degree of case
nodes, is proposed to leverage the information within the case reference
relationship for model optimization. The main codebase used in our method is
based on an open-sourced repo of CaseLink:
https://github.com/yanran-tang/CaseLink.

### 2. [Cold-Start Recommendation with Knowledge-Guided Retrieval-Augmented Generation](http://arxiv.org/pdf/2505.20773v1)

Authors: Wooseong Yang, Weizhi Zhang, Yuqing Liu, Yuwei Han, Yu Wang, Junhyun Lee, Philip S. Yu

Cold-start items remain a persistent challenge in recommender systems due to
their lack of historical user interactions, which collaborative models rely on.
While recent zero-shot methods leverage large language models (LLMs) to address
this, they often struggle with sparse metadata and hallucinated or incomplete
knowledge. We propose ColdRAG, a retrieval-augmented generation approach that
builds a domain-specific knowledge graph dynamically to enhance LLM-based
recommendation in cold-start scenarios, without requiring task-specific
fine-tuning. ColdRAG begins by converting structured item attributes into rich
natural-language profiles, from which it extracts entities and relationships to
construct a unified knowledge graph capturing item semantics. Given a user's
interaction history, it scores edges in the graph using an LLM, retrieves
candidate items with supporting evidence, and prompts the LLM to rank them. By
enabling multi-hop reasoning over this graph, ColdRAG grounds recommendations
in verifiable evidence, reducing hallucinations and strengthening semantic
connections. Experiments on three public benchmarks demonstrate that ColdRAG
surpasses existing zero-shot baselines in both Recall and NDCG. This framework
offers a practical solution to cold-start recommendation by combining
knowledge-graph reasoning with retrieval-augmented LLM generation.

### 3. [Embed Progressive Implicit Preference in Unified Space for Deep Collaborative Filtering](http://arxiv.org/pdf/2505.20900v1)

Authors: Zhongjin Zhang, Yu Liang, Cong Fu, Yuxuan Zhu, Kun Wang, Yabo Ni, Anxiang Zeng, Jiazhi Xia

Embedding-based collaborative filtering, often coupled with nearest neighbor
search, is widely deployed in large-scale recommender systems for personalized
content selection. Modern systems leverage multiple implicit feedback signals
(e.g., clicks, add to cart, purchases) to model user preferences
comprehensively. However, prevailing approaches adopt a feedback-wise modeling
paradigm, which (1) fails to capture the structured progression of user
engagement entailed among different feedback and (2) embeds feedback-specific
information into disjoint spaces, making representations incommensurable,
increasing system complexity, and leading to suboptimal retrieval performance.
A promising alternative is Ordinal Logistic Regression (OLR), which explicitly
models discrete ordered relations. However, existing OLR-based recommendation
models mainly focus on explicit feedback (e.g., movie ratings) and struggle
with implicit, correlated feedback, where ordering is vague and non-linear.
Moreover, standard OLR lacks flexibility in handling feedback-dependent
covariates, resulting in suboptimal performance in real-world systems. To
address these limitations, we propose Generalized Neural Ordinal Logistic
Regression (GNOLR), which encodes multiple feature-feedback dependencies into a
unified, structured embedding space and enforces feedback-specific dependency
learning through a nested optimization framework. Thus, GNOLR enhances
predictive accuracy, captures the progression of user engagement, and
simplifies the retrieval process. We establish a theoretical comparison with
existing paradigms, demonstrating how GNOLR avoids disjoint spaces while
maintaining effectiveness. Extensive experiments on ten real-world datasets
show that GNOLR significantly outperforms state-of-the-art methods in
efficiency and adaptability.

### 4. [LifeIR at the NTCIR-18 Lifelog-6 Task](http://arxiv.org/pdf/2505.20987v1)

Authors: Jiahan Chen, Da Li, Keping Bi

In recent years, sharing lifelogs recorded through wearable devices such as
sports watches and GoPros, has gained significant popularity. Lifelogs involve
various types of information, including images, videos, and GPS data, revealing
users' lifestyles, dietary patterns, and physical activities. The Lifelog
Semantic Access Task(LSAT) in the NTCIR-18 Lifelog-6 Challenge focuses on
retrieving relevant images from a large scale of users' lifelogs based on
textual queries describing an action or event. It serves users' need to find
images about a scenario in the historical moments of their lifelogs. We propose
a multi-stage pipeline for this task of searching images with texts, addressing
various challenges in lifelog retrieval. Our pipeline includes: filtering
blurred images, rewriting queries to make intents clearer, extending the
candidate set based on events to include images with temporal connections, and
reranking results using a multimodal large language model(MLLM) with stronger
relevance judgment capabilities. The evaluation results of our submissions have
shown the effectiveness of each stage and the entire pipeline.

### 5. [A Reduction-Driven Local Search for the Generalized Independent Set Problem](http://arxiv.org/pdf/2505.21052v1)

Authors: Yiping Liu, Yi Zhou, Zhenxiang Xu, Mingyu Xiao, Jin-Kao Hao

The Generalized Independent Set (GIS) problem extends the classical maximum
independent set problem by incorporating profits for vertices and penalties for
edges. This generalized problem has been identified in diverse applications in
fields such as forest harvest planning, competitive facility location, social
network analysis, and even machine learning. However, solving the GIS problem
in large-scale, real-world networks remains computationally challenging. In
this paper, we explore data reduction techniques to address this challenge. We
first propose 14 reduction rules that can reduce the input graph with rigorous
optimality guarantees. We then present a reduction-driven local search (RLS)
algorithm that integrates these reduction rules into the pre-processing, the
initial solution generation, and the local search components in a
computationally efficient way. The RLS is empirically evaluated on 278 graphs
arising from different application scenarios. The results indicates that the
RLS is highly competitive -- For most graphs, it achieves significantly
superior solutions compared to other known solvers, and it effectively provides
solutions for graphs exceeding 260 million edges, a task at which every other
known method fails. Analysis also reveals that the data reduction plays a key
role in achieving such a competitive performance.

### 6. [Disentangling Locality and Entropy in Ranking Distillation](http://arxiv.org/pdf/2505.21058v1)

Authors: Andrew Parry, Debasis Ganguly, Sean MacAvaney

The training process of ranking models involves two key data selection
decisions: a sampling strategy, and a labeling strategy. Modern ranking
systems, especially those for performing semantic search, typically use a
``hard negative'' sampling strategy to identify challenging items using
heuristics and a distillation labeling strategy to transfer ranking "knowledge"
from a more capable model. In practice, these approaches have grown
increasingly expensive and complex, for instance, popular pretrained rankers
from SentenceTransformers involve 12 models in an ensemble with data provenance
hampering reproducibility. Despite their complexity, modern sampling and
labeling strategies have not been fully ablated, leaving the underlying source
of effectiveness gains unclear. Thus, to better understand why models improve
and potentially reduce the expense of training effective models, we conduct a
broad ablation of sampling and distillation processes in neural ranking. We
frame and theoretically derive the orthogonal nature of model geometry affected
by example selection and the effect of teacher ranking entropy on ranking model
optimization, establishing conditions in which data augmentation can
effectively improve bias in a ranking model. Empirically, our investigation on
established benchmarks and common architectures shows that sampling processes
that were once highly effective in contrastive objectives may be spurious or
harmful under distillation. We further investigate how data augmentation, in
terms of inputs and targets, can affect effectiveness and the intrinsic
behavior of models in ranking. Through this work, we aim to encourage more
computationally efficient approaches that reduce focus on contrastive pairs and
instead directly understand training dynamics under rankings, which better
represent real-world settings.

### 7. [Counterfactual Multi-player Bandits for Explainable Recommendation Diversification](http://arxiv.org/pdf/2505.21165v1)

Authors: Yansen Zhang, Bowei He, Xiaokun Zhang, Haolun Wu, Zexu Sun, Chen Ma

Existing recommender systems tend to prioritize items closely aligned with
users' historical interactions, inevitably trapping users in the dilemma of
``filter bubble''. Recent efforts are dedicated to improving the diversity of
recommendations. However, they mainly suffer from two major issues: 1) a lack
of explainability, making it difficult for the system designers to understand
how diverse recommendations are generated, and 2) limitations to specific
metrics, with difficulty in enhancing non-differentiable diversity metrics. To
this end, we propose a \textbf{C}ounterfactual \textbf{M}ulti-player
\textbf{B}andits (CMB) method to deliver explainable recommendation
diversification across a wide range of diversity metrics. Leveraging a
counterfactual framework, our method identifies the factors influencing
diversity outcomes. Meanwhile, we adopt the multi-player bandits to optimize
the counterfactual optimization objective, making it adaptable to both
differentiable and non-differentiable diversity metrics. Extensive experiments
conducted on three real-world datasets demonstrate the applicability,
effectiveness, and explainability of the proposed CMB.

### 8. [Bridging the Gap: Self-Optimized Fine-Tuning for LLM-based Recommender Systems](http://arxiv.org/pdf/2505.20771v1)

Authors: Heng Tang, Feng Liu, Xinbo Chen, Jiawei Chen, Bohao Wang, Changwang Zhang, Jun Wang, Yuegang Sun, Bingde Hu, Can Wang

Recent years have witnessed extensive exploration of Large Language Models
(LLMs) on the field of Recommender Systems (RS). There are currently two
commonly used strategies to enable LLMs to have recommendation capabilities: 1)
The "Guidance-Only" strategy uses in-context learning to exploit and amplify
the inherent semantic understanding and item recommendation capabilities of
LLMs; 2) The "Tuning-Only" strategy uses supervised fine-tuning (SFT) to
fine-tune LLMs with the aim of fitting them to real recommendation data.
However, neither of these strategies can effectively bridge the gap between the
knowledge space of LLMs and recommendation, and their performance do not meet
our expectations.
  To better enable LLMs to learn recommendation knowledge, we combine the
advantages of the above two strategies and proposed a novel "Guidance+Tuning"
method called Self-Optimized Fine-Tuning (SOFT), which adopts the idea of
curriculum learning. It first employs self-distillation to construct an
auxiliary easy-to-learn but meaningful dataset from a fine-tuned LLM. Then it
further utilizes a self-adaptive curriculum scheduler to enable LLMs to
gradually learn from simpler data (self-distilled data) to more challenging
data (real RS data). Extensive experiments demonstrate that SOFT significantly
enhances the recommendation accuracy (37.59\% on average) of LLM-based methods.
The code is available via
https://anonymous.4open.science/r/Self-Optimized-Fine-Tuning-264E

### 9. [Personalized Query Auto-Completion for Long and Short-Term Interests with Adaptive Detoxification Generation](http://arxiv.org/pdf/2505.20966v1)

Authors: Zhibo Wang, Xiaoze Jiang, Zhiheng Qin, Enyun Yu, Han Li

Query auto-completion (QAC) plays a crucial role in modern search systems.
However, in real-world applications, there are two pressing challenges that
still need to be addressed. First, there is a need for hierarchical
personalized representations for users. Previous approaches have typically used
users' search behavior as a single, overall representation, which proves
inadequate in more nuanced generative scenarios. Additionally, query prefixes
are typically short and may contain typos or sensitive information, increasing
the likelihood of generating toxic content compared to traditional text
generation tasks. Such toxic content can degrade user experience and lead to
public relations issues. Therefore, the second critical challenge is
detoxifying QAC systems.
  To address these two limitations, we propose a novel model (LaD) that
captures personalized information from both long-term and short-term interests,
incorporating adaptive detoxification. In LaD, personalized information is
captured hierarchically at both coarse-grained and fine-grained levels. This
approach preserves as much personalized information as possible while enabling
online generation within time constraints. To move a futher step, we propose an
online training method based on Reject Preference Optimization (RPO). By
incorporating a special token [Reject] during both the training and inference
processes, the model achieves adaptive detoxification. Consequently, the
generated text presented to users is both non-toxic and relevant to the given
prefix. We conduct comprehensive experiments on industrial-scale datasets and
perform online A/B tests, delivering the largest single-experiment metric
improvement in nearly two years of our product. Our model has been deployed on
Kuaishou search, driving the primary traffic for hundreds of millions of active
users. The code is available at https://github.com/JXZe/LaD.

### 10. [Towards Better Instruction Following Retrieval Models](http://arxiv.org/pdf/2505.21439v1)

Authors: Yuchen Zhuang, Aaron Trinh, Rushi Qiang, Haotian Sun, Chao Zhang, Hanjun Dai, Bo Dai

Modern information retrieval (IR) models, trained exclusively on standard
<query, passage> pairs, struggle to effectively interpret and follow explicit
user instructions. We introduce InF-IR, a large-scale, high-quality training
corpus tailored for enhancing retrieval models in Instruction-Following IR.
InF-IR expands traditional training pairs into over 38,000 expressive
<instruction, query, passage> triplets as positive samples. In particular, for
each positive triplet, we generate two additional hard negative examples by
poisoning both instructions and queries, then rigorously validated by an
advanced reasoning model (o3-mini) to ensure semantic plausibility while
maintaining instructional incorrectness. Unlike existing corpora that primarily
support computationally intensive reranking tasks for decoder-only language
models, the highly contrastive positive-negative triplets in InF-IR further
enable efficient representation learning for smaller encoder-only models,
facilitating direct embedding-based retrieval. Using this corpus, we train
InF-Embed, an instruction-aware Embedding model optimized through contrastive
learning and instruction-query attention mechanisms to align retrieval outcomes
precisely with user intents. Extensive experiments across five
instruction-based retrieval benchmarks demonstrate that InF-Embed significantly
surpasses competitive baselines by 8.1% in p-MRR, measuring the
instruction-following capabilities.

### Machine Learning

### 1. [An Optimisation Framework for Unsupervised Environment Design](http://arxiv.org/pdf/2505.20659v1)

Authors: Nathan Monette, Alistair Letcher, Michael Beukman, Matthew T. Jackson, Alexander Rutherford, Alexander D. Goldie, Jakob N. Foerster

For reinforcement learning agents to be deployed in high-risk settings, they
must achieve a high level of robustness to unfamiliar scenarios. One method for
improving robustness is unsupervised environment design (UED), a suite of
methods aiming to maximise an agent's generalisability across configurations of
an environment. In this work, we study UED from an optimisation perspective,
providing stronger theoretical guarantees for practical settings than prior
work. Whereas previous methods relied on guarantees if they reach convergence,
our framework employs a nonconvex-strongly-concave objective for which we
provide a provably convergent algorithm in the zero-sum setting. We empirically
verify the efficacy of our method, outperforming prior methods in a number of
environments with varying difficulties.

### 2. [Sparsified State-Space Models are Efficient Highway Networks](http://arxiv.org/pdf/2505.20698v1)

Authors: Woomin Song, Jihoon Tack, Sangwoo Mo, Seunghyuk Oh, Jinwoo Shin

State-space models (SSMs) offer a promising architecture for sequence
modeling, providing an alternative to Transformers by replacing expensive
self-attention with linear recurrences. In this paper, we propose a simple yet
effective trick to enhance SSMs within given computational budgets by
sparsifying them. Our intuition is that tokens in SSMs are highly redundant due
to gradual recurrent updates, and dense recurrence operations block the
delivery of past information. In particular, we observe that upper layers of
SSMs tend to be more redundant as they encode global information, while lower
layers encode local information. Motivated by this, we introduce Simba, a
hierarchical sparsification method for SSMs based on token pruning. Simba
sparsifies upper layers more than lower layers, encouraging the upper layers to
behave like highways. To achieve this, we propose a novel token pruning
criterion for SSMs, measuring the global impact of tokens on the final output
by accumulating local recurrences. We demonstrate that Simba outperforms the
baseline model, Mamba, with the same FLOPS in various natural language tasks.
Moreover, we illustrate the effect of highways, showing that Simba not only
enhances efficiency but also improves the information flow across long
sequences. Code is available at https://github.com/woominsong/Simba.

### 3. [Are Data Embeddings effective in time series forecasting?](http://arxiv.org/pdf/2505.20716v1)

Authors: Reza Nematirad, Anil Pahwa, Balasubramaniam Natarajan

Time series forecasting plays a crucial role in many real-world applications,
and numerous complex forecasting models have been proposed in recent years.
Despite their architectural innovations, most state-of-the-art models report
only marginal improvements -- typically just a few thousandths in standard
error metrics. These models often incorporate complex data embedding layers to
transform raw inputs into higher-dimensional representations to enhance
accuracy. But are data embedding techniques actually effective in time series
forecasting? Through extensive ablation studies across fifteen state-of-the-art
models and four benchmark datasets, we find that removing data embedding layers
from many state-of-the-art models does not degrade forecasting performance. In
many cases, it improves both accuracy and computational efficiency. The gains
from removing embedding layers often exceed the performance differences
typically reported between competing models. Code available at:
https://github.com/neuripsdataembedidng/DataEmbedding

### 4. ['Hello, World!': Making GNNs Talk with LLMs](http://arxiv.org/pdf/2505.20742v1)

Authors: Sunwoo Kim, Soo Yong Lee, Jaemin Yoo, Kijung Shin

While graph neural networks (GNNs) have shown remarkable performance across
diverse graph-related tasks, their high-dimensional hidden representations
render them black boxes. In this work, we propose Graph Lingual Network (GLN),
a GNN built on large language models (LLMs), with hidden representations in the
form of human-readable text. Through careful prompt design, GLN incorporates
not only the message passing module of GNNs but also advanced GNN techniques,
including graph attention and initial residual connection. The
comprehensibility of GLN's hidden representations enables an intuitive analysis
of how node representations change (1) across layers and (2) under advanced GNN
techniques, shedding light on the inner workings of GNNs. Furthermore, we
demonstrate that GLN achieves strong zero-shot performance on node
classification and link prediction, outperforming existing LLM-based baseline
methods.

### 5. [Robust and Explainable Detector of Time Series Anomaly via Augmenting Multiclass Pseudo-Anomalies](http://arxiv.org/pdf/2505.20765v1)

Authors: Kohei Obata, Yasuko Matsubara, Yasushi Sakurai

Unsupervised anomaly detection in time series has been a pivotal research
area for decades. Current mainstream approaches focus on learning normality, on
the assumption that all or most of the samples in the training set are normal.
However, anomalies in the training set (i.e., anomaly contamination) can be
misleading. Recent studies employ data augmentation to generate
pseudo-anomalies and learn the boundary separating the training samples from
the augmented samples. Although this approach mitigates anomaly contamination
if augmented samples mimic unseen real anomalies, it suffers from several
limitations. (1) Covering a wide range of time series anomalies is challenging.
(2) It disregards augmented samples that resemble normal samples (i.e., false
anomalies). (3) It places too much trust in the labels of training and
augmented samples. In response, we propose RedLamp, which employs diverse data
augmentations to generate multiclass pseudo-anomalies and learns the multiclass
boundary. Such multiclass pseudo-anomalies cover a wide variety of time series
anomalies. We conduct multiclass classification using soft labels, which
prevents the model from being overconfident and ensures its robustness against
contaminated/false anomalies. The learned latent space is inherently
explainable as it is trained to separate pseudo-anomalies into multiclasses.
Extensive experiments demonstrate the effectiveness of RedLamp in anomaly
detection and its robustness against anomaly contamination.

### 6. [TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](http://arxiv.org/pdf/2505.20774v1)

Authors: Xiaowen Ma, Zhenliang Ni, Shuai Xiao, Xinghao Chen

In long-term time series forecasting, different variables often influence the
target variable over distinct time intervals, a challenge known as the
multi-delay issue. Traditional models typically process all variables or time
points uniformly, which limits their ability to capture complex variable
relationships and obtain non-trivial time representations. To address this
issue, we propose TimePro, an innovative Mamba-based model that constructs
variate- and time-aware hyper-states. Unlike conventional approaches that
merely transfer plain states across variable or time dimensions, TimePro
preserves the fine-grained temporal features of each variate token and
adaptively selects the focused time points to tune the plain state. The
reconstructed hyper-state can perceive both variable relationships and salient
temporal information, which helps the model make accurate forecasting. In
experiments, TimePro performs competitively on eight real-world long-term
forecasting benchmarks with satisfactory linear complexity. Code is available
at https://github.com/xwmaxwma/TimePro.

### 7. [Simple yet Effective Graph Distillation via Clustering](http://arxiv.org/pdf/2505.20807v1)

Authors: Yurui Lai, Taiyan Zhang, Renchi Yang

Despite plentiful successes achieved by graph representation learning in
various domains, the training of graph neural networks (GNNs) still remains
tenaciously challenging due to the tremendous computational overhead needed for
sizable graphs in practice. Recently, graph data distillation (GDD), which
seeks to distill large graphs into compact and informative ones, has emerged as
a promising technique to enable efficient GNN training. However, most existing
GDD works rely on heuristics that align model gradients or representation
distributions on condensed and original graphs, leading to compromised result
quality, expensive training for distilling large graphs, or both. Motivated by
this, this paper presents an efficient and effective GDD approach, ClustGDD.
Under the hood, ClustGDD resorts to synthesizing the condensed graph and node
attributes through fast and theoretically-grounded clustering that minimizes
the within-cluster sum of squares and maximizes the homophily on the original
graph. The fundamental idea is inspired by our empirical and theoretical
findings unveiling the connection between clustering and empirical condensation
quality using Fr\'echet Inception Distance, a well-known quality metric for
synthetic images. Furthermore, to mitigate the adverse effects caused by the
homophily-based clustering, ClustGDD refines the nodal attributes of the
condensed graph with a small augmentation learned via class-aware graph
sampling and consistency loss. Our extensive experiments exhibit that GNNs
trained over condensed graphs output by ClustGDD consistently achieve superior
or comparable performance to state-of-the-art GDD methods in terms of node
classification on five benchmark datasets, while being orders of magnitude
faster.

### 8. [Interpretable Credit Default Prediction with Ensemble Learning and SHAP](http://arxiv.org/pdf/2505.20815v1)

Authors: Shiqi Yang, Ziyi Huang, Wengran Xiao, Xinyu Shen

This study focuses on the problem of credit default prediction, builds a
modeling framework based on machine learning, and conducts comparative
experiments on a variety of mainstream classification algorithms. Through
preprocessing, feature engineering, and model training of the Home Credit
dataset, the performance of multiple models including logistic regression,
random forest, XGBoost, LightGBM, etc. in terms of accuracy, precision, and
recall is evaluated. The results show that the ensemble learning method has
obvious advantages in predictive performance, especially in dealing with
complex nonlinear relationships between features and data imbalance problems.
It shows strong robustness. At the same time, the SHAP method is used to
analyze the importance and dependency of features, and it is found that the
external credit score variable plays a dominant role in model decision making,
which helps to improve the model's interpretability and practical application
value. The research results provide effective reference and technical support
for the intelligent development of credit risk control systems.

### 9. [FireQ: Fast INT4-FP8 Kernel and RoPE-aware Quantization for LLM Inference Acceleration](http://arxiv.org/pdf/2505.20839v1)

Authors: Daehyeon Baek, Jieun Choi, Jimyoung Son, Kyungmin Bin, Seungbeom Choi, Kihyo Moon, Minsung Jang, Hyojung Lee

As large language models become increasingly prevalent, memory bandwidth
constraints significantly limit inference throughput, motivating post-training
quantization (PTQ). In this paper, we propose FireQ, a co-designed PTQ
framework and an INT4-FP8 matrix multiplication kernel that accelerates LLM
inference across all linear layers. Specifically, FireQ quantizes linear layer
weights and key-values to INT4, and activations and queries to FP8,
significantly enhancing throughput. Additionally, we introduce a three-stage
pipelining for the prefill phase, which modifies the FlashAttention-3 kernel,
effectively reducing time-to-first-token in the prefill phase. To minimize
accuracy loss from quantization, we develop novel outlier smoothing techniques
tailored separately for linear and attention layers. In linear layers, we
explicitly use per-tensor scaling to prevent underflow caused by the FP8
quantization scaling factor of INT4 quantization, and channel-wise scaling to
compensate for coarse granularity of INT4. In attention layers, we address
quantization challenges posed by rotary positional embeddings (RoPE) by
combining pre-RoPE and post-RoPE scaling strategies. FireQ significantly
outperforms state-of-the-art methods, achieving 1.68x faster inference in
feed-forward network layers on Llama2-7B and 1.26x faster prefill phase
performance on Llama3-8B compared to QServe, with negligible accuracy loss.

### 10. [Aggregation Buffer: Revisiting DropEdge with a New Parameter Block](http://arxiv.org/pdf/2505.20840v1)

Authors: Dooho Lee, Myeong Kong, Sagad Hamid, Cheonwoo Lee, Jaemin Yoo

We revisit DropEdge, a data augmentation technique for GNNs which randomly
removes edges to expose diverse graph structures during training. While being a
promising approach to effectively reduce overfitting on specific connections in
the graph, we observe that its potential performance gain in supervised
learning tasks is significantly limited. To understand why, we provide a
theoretical analysis showing that the limited performance of DropEdge comes
from the fundamental limitation that exists in many GNN architectures. Based on
this analysis, we propose Aggregation Buffer, a parameter block specifically
designed to improve the robustness of GNNs by addressing the limitation of
DropEdge. Our method is compatible with any GNN model, and shows consistent
performance improvements on multiple datasets. Moreover, our method effectively
addresses well-known problems such as degree bias or structural disparity as a
unifying solution. Code and datasets are available at
https://github.com/dooho00/agg-buffer.

### Neural and Evolutionary Computing

### 1. [Multi-Objective Covariance Matrix Adaptation MAP-Annealing](http://arxiv.org/pdf/2505.20712v1)

Authors: Shihan Zhao, Stefanos Nikolaidis

Quality-Diversity (QD) optimization is an emerging field that focuses on
finding a set of behaviorally diverse and high-quality solutions. While the
quality is typically defined w.r.t. a single objective function, recent work on
Multi-Objective Quality-Diversity (MOQD) extends QD optimization to
simultaneously optimize multiple objective functions. This opens up
multi-objective applications for QD, such as generating a diverse set of game
maps that maximize difficulty, realism, or other properties. Existing MOQD
algorithms use non-adaptive methods such as mutation and crossover to search
for non-dominated solutions and construct an archive of Pareto Sets (PS).
However, recent work in QD has demonstrated enhanced performance through the
use of covariance-based evolution strategies for adaptive solution search. We
propose bringing this insight into the MOQD problem, and introduce MO-CMA-MAE,
a new MOQD algorithm that leverages Covariance Matrix Adaptation-Evolution
Strategies (CMA-ES) to optimize the hypervolume associated with every PS within
the archive. We test MO-CMA-MAE on three MOQD domains, and for generating maps
of a co-operative video game, showing significant improvements in performance.

### 2. [Hybrid Wave-wind System Power Optimisation Using Effective Ensemble Covariance Matrix Adaptation Evolutionary Algorithm](http://arxiv.org/pdf/2505.20720v1)

Authors: Mehdi Neshat, Nataliia Y. Sergiienko, Leandro S. P. da Silva, Seyedali Mirjalili, Amir H. Gandomi, Ossama Abdelkhalik, John Boland

Floating hybrid wind-wave systems combine offshore wind platforms with wave
energy converters (WECs) to create cost-effective and reliable energy
solutions. Adequately designed and tuned WECs are essential to avoid unwanted
loads disrupting turbine motion while efficiently harvesting wave energy. These
systems diversify energy sources, enhancing energy security and reducing supply
risks while providing a more consistent power output by smoothing energy
production variability. However, optimising such systems is complex due to the
physical and hydrodynamic interactions between components, resulting in a
challenging optimisation space. This study uses a 5-MW OC4-DeepCwind
semi-submersible platform with three spherical WECs to explore these synergies.
To address these challenges, we propose an effective ensemble optimisation
(EEA) technique that combines covariance matrix adaptation, novelty search, and
discretisation techniques. To evaluate the EEA performance, we used four sea
sites located along Australia's southern coast. In this framework, geometry and
power take-off parameters are simultaneously optimised to maximise the average
power output of the hybrid wind-wave system. Ensemble optimisation methods
enhance performance, flexibility, and robustness by identifying the best
algorithm or combination of algorithms for a given problem, addressing issues
like premature convergence, stagnation, and poor search space exploration. The
EEA was benchmarked against 14 advanced optimisation methods, demonstrating
superior solution quality and convergence rates. EEA improved total power
output by 111%, 95%, and 52% compared to WOA, EO, and AHA, respectively.
Additionally, in comparisons with advanced methods, LSHADE, SaNSDE, and SLPSO,
EEA achieved absorbed power enhancements of 498%, 638%, and 349% at the Sydney
sea site, showcasing its effectiveness in optimising hybrid energy systems.

### 3. [Fully Spiking Neural Networks for Unified Frame-Event Object Tracking](http://arxiv.org/pdf/2505.20834v1)

Authors: Jingjun Yang, Liangwei Fan, Jinpu Zhang, Xiangkai Lian, Hui Shen, Dewen Hu

The integration of image and event streams offers a promising approach for
achieving robust visual object tracking in complex environments. However,
current fusion methods achieve high performance at the cost of significant
computational overhead and struggle to efficiently extract the sparse,
asynchronous information from event streams, failing to leverage the
energy-efficient advantages of event-driven spiking paradigms. To address this
challenge, we propose the first fully Spiking Frame-Event Tracking framework
called SpikeFET. This network achieves synergistic integration of convolutional
local feature extraction and Transformer-based global modeling within the
spiking paradigm, effectively fusing frame and event data. To overcome the
degradation of translation invariance caused by convolutional padding, we
introduce a Random Patchwork Module (RPM) that eliminates positional bias
through randomized spatial reorganization and learnable type encoding while
preserving residual structures. Furthermore, we propose a Spatial-Temporal
Regularization (STR) strategy that overcomes similarity metric degradation from
asymmetric features by enforcing spatio-temporal consistency among temporal
template features in latent space. Extensive experiments across multiple
benchmarks demonstrate that the proposed framework achieves superior tracking
accuracy over existing methods while significantly reducing power consumption,
attaining an optimal balance between performance and efficiency. The code will
be released.

### 4. [LLaMEA-BO: A Large Language Model Evolutionary Algorithm for Automatically Generating Bayesian Optimization Algorithms](http://arxiv.org/pdf/2505.21034v1)

Authors: Wenhu Li, Niki van Stein, Thomas Bäck, Elena Raponi

Bayesian optimization (BO) is a powerful class of algorithms for optimizing
expensive black-box functions, but designing effective BO algorithms remains a
manual, expertise-driven task. Recent advancements in Large Language Models
(LLMs) have opened new avenues for automating scientific discovery, including
the automatic design of optimization algorithms. While prior work has used LLMs
within optimization loops or to generate non-BO algorithms, we tackle a new
challenge: Using LLMs to automatically generate full BO algorithm code. Our
framework uses an evolution strategy to guide an LLM in generating Python code
that preserves the key components of BO algorithms: An initial design, a
surrogate model, and an acquisition function. The LLM is prompted to produce
multiple candidate algorithms, which are evaluated on the established Black-Box
Optimization Benchmarking (BBOB) test suite from the COmparing Continuous
Optimizers (COCO) platform. Based on their performance, top candidates are
selected, combined, and mutated via controlled prompt variations, enabling
iterative refinement. Despite no additional fine-tuning, the LLM-generated
algorithms outperform state-of-the-art BO baselines in 19 (out of 24) BBOB
functions in dimension 5 and generalize well to higher dimensions, and
different tasks (from the Bayesmark framework). This work demonstrates that
LLMs can serve as algorithmic co-designers, offering a new paradigm for
automating BO development and accelerating the discovery of novel algorithmic
combinations. The source code is provided at
https://github.com/Ewendawi/LLaMEA-BO.

### Networking and Internet Architecture

### 1. [Dynamical ON-OFF Control with Trajectory Prediction for Multi-RIS Wireless Networks](http://arxiv.org/pdf/2505.20887v1)

Authors: Kaining Wang, Bo Yang, Yusheng Lei, Zhiwen Yu, Xuelin Cao, George C. Alexandropoulos, Marco Di Renzo, Chau Yuen

Reconfigurable intelligent surfaces (RISs) have demonstrated an unparalleled
ability to reconfigure wireless environments by dynamically controlling the
phase, amplitude, and polarization of impinging waves. However, as nearly
passive reflective metasurfaces, RISs may not distinguish between desired and
interference signals, which can lead to severe spectrum pollution and even
affect performance negatively. In particular, in large-scale networks, the
signal-to-interference-plus-noise ratio (SINR) at the receiving node can be
degraded due to excessive interference reflected from the RIS. To overcome this
fundamental limitation, we propose in this paper a trajectory prediction-based
dynamical control algorithm (TPC) for anticipating RIS ON-OFF states sequence,
integrating a long-short-term-memory (LSTM) scheme to predict user
trajectories. In particular, through a codebook-based algorithm, the RIS
controller adaptively coordinates the configuration of the RIS elements to
maximize the received SINR. Our simulation results demonstrate the superiority
of the proposed TPC method over various system settings.

### 2. [Interference Detection in Spectrum-Blind Multi-User Optical Spectrum as a Service](http://arxiv.org/pdf/2505.21018v1)

Authors: Agastya Raj, Daniel C. Kilper, Marco Ruffini

With the growing demand for high-bandwidth, low-latency applications, Optical
Spectrum as a Service (OSaaS) is of interest for flexible bandwidth allocation
within Elastic Optical Networks (EONs) and Open Line Systems (OLS). While OSaaS
facilitates transparent connectivity and resource sharing among users, it
raises concerns over potential network vulnerabilities due to shared fiber
access and inter-channel interference, such as fiber non-linearity and
amplifier based crosstalk. These challenges are exacerbated in multi-user
environments, complicating the identification and localization of service
interferences. To reduce system disruptions and system repair costs, it is
beneficial to detect and identify such interferences timely. Addressing these
challenges, this paper introduces a Machine Learning (ML) based architecture
for network operators to detect and attribute interferences to specific OSaaS
users while blind to the users' internal spectrum details. Our methodology
leverages available coarse power measurements and operator channel performance
data, bypassing the need for internal user information of wide-band shared
spectra. Experimental studies conducted on a 190 km optical line system in the
Open Ireland testbed, with three OSaaS users demonstrate the model's capability
to accurately classify the source of interferences, achieving a classification
accuracy of 90.3%.

### 3. [Wideband RF Radiance Field Modeling Using Frequency-embedded 3D Gaussian Splatting](http://arxiv.org/pdf/2505.20714v1)

Authors: Zechen Li, Lanqing Yang, Yiheng Bian, Hao Pan, Yongjian Fu, Yezhou Wang, Yi-Chao Chen, Guangtao Xue, Ju Ren

This paper presents an innovative frequency-embedded 3D Gaussian splatting
(3DGS) algorithm for wideband radio-frequency (RF) radiance field modeling,
offering an advancement over the existing works limited to single-frequency
modeling. Grounded in fundamental physics, we uncover the complex relationship
between EM wave propagation behaviors and RF frequencies. Inspired by this, we
design an EM feature network with attenuation and radiance modules to learn the
complex relationships between RF frequencies and the key properties of each 3D
Gaussian, specifically the attenuation factor and RF signal intensity. By
training the frequency-embedded 3DGS model, we can efficiently reconstruct RF
radiance fields at arbitrary unknown frequencies within a given 3D environment.
Finally, we propose a large-scale power angular spectrum (PAS) dataset
containing 50000 samples ranging from 1 to 100 GHz in 6 indoor environments,
and conduct extensive experiments to verify the effectiveness of our method.
Our approach achieves an average Structural Similarity Index Measure (SSIM) up
to 0.72, and a significant improvement up to 17.8% compared to the current
state-of-the-art (SOTA) methods trained on individual test frequencies.
Additionally, our method achieves an SSIM of 0.70 without prior training on
these frequencies, which represents only a 2.8% performance drop compared to
models trained with full PAS data. This demonstrates our model's capability to
estimate PAS at unknown frequencies. For related code and datasets, please
refer to https://github.com/sim-2-real/Wideband3DGS.

### 4. [Respond to Change with Constancy: Instruction-tuning with LLM for Non-I.I.D. Network Traffic Classification](http://arxiv.org/pdf/2505.20866v1)

Authors: Xinjie Lin, Gang Xiong, Gaopeng Gou, Wenqi Dong, Jing Yu, Zhen Li, Wei Xia

Encrypted traffic classification is highly challenging in network security
due to the need for extracting robust features from content-agnostic traffic
data. Existing approaches face critical issues: (i) Distribution drift, caused
by reliance on the closedworld assumption, limits adaptability to realworld,
shifting patterns; (ii) Dependence on labeled data restricts applicability
where such data is scarce or unavailable. Large language models (LLMs) have
demonstrated remarkable potential in offering generalizable solutions across a
wide range of tasks, achieving notable success in various specialized fields.
However, their effectiveness in traffic analysis remains constrained by
challenges in adapting to the unique requirements of the traffic domain. In
this paper, we introduce a novel traffic representation model named Encrypted
Traffic Out-of-Distribution Instruction Tuning with LLM (ETooL), which
integrates LLMs with knowledge of traffic structures through a self-supervised
instruction tuning paradigm. This framework establishes connections between
textual information and traffic interactions. ETooL demonstrates more robust
classification performance and superior generalization in both supervised and
zero-shot traffic classification tasks. Notably, it achieves significant
improvements in F1 scores: APP53 (I.I.D.) to 93.19%(6.62%) and 92.11%(4.19%),
APP53 (O.O.D.) to 74.88%(18.17%) and 72.13%(15.15%), and ISCX-Botnet (O.O.D.)
to 95.03%(9.16%) and 81.95%(12.08%). Additionally, we construct NETD, a traffic
dataset designed to support dynamic distributional shifts, and use it to
validate ETooL's effectiveness under varying distributional conditions.
Furthermore, we evaluate the efficiency gains achieved through ETooL's
instruction tuning approach.

### Robotics

### 1. [Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion](http://arxiv.org/pdf/2505.20619v1)

Authors: Tianhu Peng, Lingfan Bao, CHengxu Zhou

We present a unified gait-conditioned reinforcement learning framework that
enables humanoid robots to perform standing, walking, running, and smooth
transitions within a single recurrent policy. A compact reward routing
mechanism dynamically activates gait-specific objectives based on a one-hot
gait ID, mitigating reward interference and supporting stable multi-gait
learning. Human-inspired reward terms promote biomechanically natural motions,
such as straight-knee stance and coordinated arm-leg swing, without requiring
motion capture data. A structured curriculum progressively introduces gait
complexity and expands command space over multiple phases. In simulation, the
policy successfully achieves robust standing, walking, running, and gait
transitions. On the real Unitree G1 humanoid, we validate standing, walking,
and walk-to-stand transitions, demonstrating stable and coordinated locomotion.
This work provides a scalable, reference-free solution toward versatile and
naturalistic humanoid control across diverse modes and environments.

### 2. [ManiTaskGen: A Comprehensive Task Generator for Benchmarking and Improving Vision-Language Agents on Embodied Decision-Making](http://arxiv.org/pdf/2505.20726v1)

Authors: Liu Dai, Haina Wang, Weikang Wan, Hao Su

Building embodied agents capable of accomplishing arbitrary tasks is a core
objective towards achieving embodied artificial general intelligence (E-AGI).
While recent work has advanced such general robot policies, their training and
evaluation are often limited to tasks within specific scenes, involving
restricted instructions and scenarios. Existing benchmarks also typically rely
on manual annotation of limited tasks in a few scenes. We argue that exploring
the full spectrum of feasible tasks within any given scene is crucial, as they
provide both extensive benchmarks for evaluation and valuable resources for
agent improvement. Towards this end, we introduce ManiTaskGen, a novel system
that automatically generates comprehensive, diverse, feasible mobile
manipulation tasks for any given scene. The generated tasks encompass both
process-based, specific instructions (e.g., "move object from X to Y") and
outcome-based, abstract instructions (e.g., "clear the table"). We apply
ManiTaskGen to both simulated and real-world scenes, demonstrating the validity
and diversity of the generated tasks. We then leverage these tasks to
automatically construct benchmarks, thoroughly evaluating the embodied
decision-making capabilities of agents built upon existing vision-language
models (VLMs). Furthermore, we propose a simple yet effective method that
utilizes ManiTaskGen tasks to enhance embodied decision-making. Overall, this
work presents a universal task generation framework for arbitrary scenes,
facilitating both benchmarking and improvement of embodied decision-making
agents.

### 3. [Learning Generalizable Robot Policy with Human Demonstration Video as a Prompt](http://arxiv.org/pdf/2505.20795v1)

Authors: Xiang Zhu, Yichen Liu, Hezhong Li, Jianyu Chen

Recent robot learning methods commonly rely on imitation learning from
massive robotic dataset collected with teleoperation. When facing a new task,
such methods generally require collecting a set of new teleoperation data and
finetuning the policy. Furthermore, the teleoperation data collection pipeline
is also tedious and expensive. Instead, human is able to efficiently learn new
tasks by just watching others do. In this paper, we introduce a novel two-stage
framework that utilizes human demonstrations to learn a generalizable robot
policy. Such policy can directly take human demonstration video as a prompt and
perform new tasks without any new teleoperation data and model finetuning at
all. In the first stage, we train video generation model that captures a joint
representation for both the human and robot demonstration video data using
cross-prediction. In the second stage, we fuse the learned representation with
a shared action space between human and robot using a novel prototypical
contrastive loss. Empirical evaluations on real-world dexterous manipulation
tasks show the effectiveness and generalization capabilities of our proposed
method.

### 4. [GET: Goal-directed Exploration and Targeting for Large-Scale Unknown Environments](http://arxiv.org/pdf/2505.20828v1)

Authors: Lanxiang Zheng, Ruidong Mei, Mingxin Wei, Hao Ren, Hui Cheng

Object search in large-scale, unstructured environments remains a fundamental
challenge in robotics, particularly in dynamic or expansive settings such as
outdoor autonomous exploration. This task requires robust spatial reasoning and
the ability to leverage prior experiences. While Large Language Models (LLMs)
offer strong semantic capabilities, their application in embodied contexts is
limited by a grounding gap in spatial reasoning and insufficient mechanisms for
memory integration and decision consistency.To address these challenges, we
propose GET (Goal-directed Exploration and Targeting), a framework that
enhances object search by combining LLM-based reasoning with experience-guided
exploration. At its core is DoUT (Diagram of Unified Thought), a reasoning
module that facilitates real-time decision-making through a role-based feedback
loop, integrating task-specific criteria and external memory. For repeated
tasks, GET maintains a probabilistic task map based on a Gaussian Mixture
Model, allowing for continual updates to object-location priors as environments
evolve.Experiments conducted in real-world, large-scale environments
demonstrate that GET improves search efficiency and robustness across multiple
LLMs and task settings, significantly outperforming heuristic and LLM-only
baselines. These results suggest that structured LLM integration provides a
scalable and generalizable approach to embodied decision-making in complex
environments.

### 5. [Learning Unified Force and Position Control for Legged Loco-Manipulation](http://arxiv.org/pdf/2505.20829v1)

Authors: Peiyuan Zhi, Peiyang Li, Jianqin Yin, Baoxiong Jia, Siyuan Huang

Robotic loco-manipulation tasks often involve contact-rich interactions with
the environment, requiring the joint modeling of contact force and robot
position. However, recent visuomotor policies often focus solely on learning
position or force control, overlooking their co-learning. In this work, we
propose the first unified policy for legged robots that jointly models force
and position control learned without reliance on force sensors. By simulating
diverse combinations of position and force commands alongside external
disturbance forces, we use reinforcement learning to learn a policy that
estimates forces from historical robot states and compensates for them through
position and velocity adjustments. This policy enables a wide range of
manipulation behaviors under varying force and position inputs, including
position tracking, force application, force tracking, and compliant
interactions. Furthermore, we demonstrate that the learned policy enhances
trajectory-based imitation learning pipelines by incorporating essential
contact information through its force estimation module, achieving
approximately 39.5% higher success rates across four challenging contact-rich
manipulation tasks compared to position-control policies. Extensive experiments
on both a quadrupedal manipulator and a humanoid robot validate the versatility
and robustness of the proposed policy across diverse scenarios.

### 6. [G-DReaM: Graph-conditioned Diffusion Retargeting across Multiple Embodiments](http://arxiv.org/pdf/2505.20857v1)

Authors: Zhefeng Cao, Ben Liu, Sen Li, Wei Zhang, Hua Chen

Motion retargeting for specific robot from existing motion datasets is one
critical step in transferring motion patterns from human behaviors to and
across various robots. However, inconsistencies in topological structure,
geometrical parameters as well as joint correspondence make it difficult to
handle diverse embodiments with a unified retargeting architecture. In this
work, we propose a novel unified graph-conditioned diffusion-based motion
generation framework for retargeting reference motions across diverse
embodiments. The intrinsic characteristics of heterogeneous embodiments are
represented with graph structure that effectively captures topological and
geometrical features of different robots. Such a graph-based encoding further
allows for knowledge exploitation at the joint level with a customized
attention mechanisms developed in this work. For lacking ground truth motions
of the desired embodiment, we utilize an energy-based guidance formulated as
retargeting losses to train the diffusion model. As one of the first
cross-embodiment motion retargeting methods in robotics, our experiments
validate that the proposed model can retarget motions across heterogeneous
embodiments in a unified manner. Moreover, it demonstrates a certain degree of
generalization to both diverse skeletal structures and similar motion patterns.

### 7. [HS-SLAM: A Fast and Hybrid Strategy-Based SLAM Approach for Low-Speed Autonomous Driving](http://arxiv.org/pdf/2505.20906v1)

Authors: Bingxiang Kang, Jie Zou, Guofa Li, Pengwei Zhang, Jie Zeng, Kan Wang, Jie Li

Visual-inertial simultaneous localization and mapping (SLAM) is a key module
of robotics and low-speed autonomous vehicles, which is usually limited by the
high computation burden for practical applications. To this end, an innovative
strategy-based hybrid framework HS-SLAM is proposed to integrate the advantages
of direct and feature-based methods for fast computation without decreasing the
performance. It first estimates the relative positions of consecutive frames
using IMU pose estimation within the tracking thread. Then, it refines these
estimates through a multi-layer direct method, which progressively corrects the
relative pose from coarse to fine, ultimately achieving accurate corner-based
feature matching. This approach serves as an alternative to the conventional
constant-velocity tracking model. By selectively bypassing descriptor
extraction for non-critical frames, HS-SLAM significantly improves the tracking
speed. Experimental evaluations on the EuRoC MAV dataset demonstrate that
HS-SLAM achieves higher localization accuracies than ORB-SLAM3 while improving
the average tracking efficiency by 15%.

### 8. [SCALOFT: An Initial Approach for Situation Coverage-Based Safety Analysis of an Autonomous Aerial Drone in a Mine Environment](http://arxiv.org/pdf/2505.20969v1)

Authors: Nawshin Mannan Proma, Victoria J Hodge, Rob Alexander

The safety of autonomous systems in dynamic and hazardous environments poses
significant challenges. This paper presents a testing approach named SCALOFT
for systematically assessing the safety of an autonomous aerial drone in a
mine. SCALOFT provides a framework for developing diverse test cases, real-time
monitoring of system behaviour, and detection of safety violations. Detected
violations are then logged with unique identifiers for detailed analysis and
future improvement. SCALOFT helps build a safety argument by monitoring
situation coverage and calculating a final coverage measure. We have evaluated
the performance of this approach by deliberately introducing seeded faults into
the system and assessing whether SCALOFT is able to detect those faults. For a
small set of plausible faults, we show that SCALOFT is successful in this.

### 9. [EgoWalk: A Multimodal Dataset for Robot Navigation in the Wild](http://arxiv.org/pdf/2505.21282v1)

Authors: Timur Akhtyamov, Mohamad Al Mdfaa, Javier Antonio Ramirez, Sergey Bakulin, German Devchich, Denis Fatykhov, Alexander Mazurov, Kristina Zipa, Malik Mohrat, Pavel Kolesnik, Ivan Sosin, Gonzalo Ferrer

Data-driven navigation algorithms are critically dependent on large-scale,
high-quality real-world data collection for successful training and robust
performance in realistic and uncontrolled conditions. To enhance the growing
family of navigation-related real-world datasets, we introduce EgoWalk - a
dataset of 50 hours of human navigation in a diverse set of indoor/outdoor,
varied seasons, and location environments. Along with the raw and Imitation
Learning-ready data, we introduce several pipelines to automatically create
subsidiary datasets for other navigation-related tasks, namely natural language
goal annotations and traversability segmentation masks. Diversity studies, use
cases, and benchmarks for the proposed dataset are provided to demonstrate its
practical applicability.
  We openly release all data processing pipelines and the description of the
hardware platform used for data collection to support future research and
development in robot navigation systems.

### 10. [EquAct: An SE(3)-Equivariant Multi-Task Transformer for Open-Loop Robotic Manipulation](http://arxiv.org/pdf/2505.21351v1)

Authors: Xupeng Zhu, Yu Qi, Yizhe Zhu, Robin Walters, Robert Platt

Transformer architectures can effectively learn language-conditioned,
multi-task 3D open-loop manipulation policies from demonstrations by jointly
processing natural language instructions and 3D observations. However, although
both the robot policy and language instructions inherently encode rich 3D
geometric structures, standard transformers lack built-in guarantees of
geometric consistency, often resulting in unpredictable behavior under SE(3)
transformations of the scene. In this paper, we leverage SE(3) equivariance as
a key structural property shared by both policy and language, and propose
EquAct-a novel SE(3)-equivariant multi-task transformer. EquAct is
theoretically guaranteed to be SE(3) equivariant and consists of two key
components: (1) an efficient SE(3)-equivariant point cloud-based U-net with
spherical Fourier features for policy reasoning, and (2) SE(3)-invariant
Feature-wise Linear Modulation (iFiLM) layers for language conditioning. To
evaluate its spatial generalization ability, we benchmark EquAct on 18 RLBench
simulation tasks with both SE(3) and SE(2) scene perturbations, and on 4
physical tasks. EquAct performs state-of-the-art across these simulation and
physical tasks.

### Software Engineering

### 1. [CXXCrafter: An LLM-Based Agent for Automated C/C++ Open Source Software Building](http://arxiv.org/pdf/2505.21069v1)

Authors: Zhengmin Yu, Yuan Zhang, Ming Wen, Yinan Nie, Wenhui Zhang, Min Yang

Project building is pivotal to support various program analysis tasks, such
as generating intermediate rep- resentation code for static analysis and
preparing binary code for vulnerability reproduction. However, automating the
building process for C/C++ projects is a highly complex endeavor, involving
tremendous technical challenges, such as intricate dependency management,
diverse build systems, varied toolchains, and multifaceted error handling
mechanisms. Consequently, building C/C++ projects often proves to be difficult
in practice, hindering the progress of downstream applications. Unfortunately,
research on facilitating the building of C/C++ projects remains to be
inadequate. The emergence of Large Language Models (LLMs) offers promising
solutions to automated software building. Trained on extensive corpora, LLMs
can help unify diverse build systems through their comprehension capabilities
and address complex errors by leveraging tacit knowledge storage. Moreover,
LLM-based agents can be systematically designed to dynamically interact with
the environment, effectively managing dynamic building issues. Motivated by
these opportunities, we first conduct an empirical study to systematically
analyze the current challenges in the C/C++ project building process.
Particularly, we observe that most popular C/C++ projects encounter an average
of five errors when relying solely on the default build systems. Based on our
study, we develop an automated build system called CXXCrafter to specifically
address the above-mentioned challenges, such as dependency resolution. Our
evaluation on open-source software demonstrates that CXXCrafter achieves a
success rate of 78% in project building. Specifically, among the Top100
dataset, 72 projects are built successfully by both CXXCrafter and manual
efforts, 3 by CXXCrafter only, and 14 manually only. ...

### 2. [A first look at ROS~2 applications written in asynchronous Rust](http://arxiv.org/pdf/2505.21323v1)

Authors: Martin Škoudlil, Michal Sojka, Zdeněk Hanzálek

The increasing popularity of the Rust programming language in building
robotic applications using the Robot Operating System (ROS~2) raises questions
about its real-time execution capabilities, particularly when employing
asynchronous programming. Existing real-time scheduling and response-time
analysis techniques for ROS~2 focus on applications written in C++ and do not
address the unique execution models and challenges presented by Rust's
asynchronous programming paradigm. In this paper, we analyze the execution
model of R2R -- an asynchronous Rust ROS~2 bindings and various asynchronous
Rust runtimes, comparing them with the execution model of C++ ROS~2
applications. We propose a structured approach for R2R applications aimed at
deterministic real-time operation involving thread prioritization and
callback-to-thread mapping schemes. Our experimental evaluation based on
measuring end-to-end latencies of a synthetic application shows that the
proposed approach is effective and outperforms other evaluated configurations.
A more complex autonomous driving case study demonstrates its practical
applicability. Overall, the experimental results indicate that our proposed
structure achieves bounded response times for time-critical tasks. This paves
the way for future work to adapt existing or develop new response-time analysis
techniques for R2R applications using our structure.

### 3. [GUARD:Dual-Agent based Backdoor Defense on Chain-of-Thought in Neural Code Generation](http://arxiv.org/pdf/2505.21425v1)

Authors: Naizhu Jin, Zhong Li, Tian Zhang, Qingkai Zeng

With the widespread application of large language models in code generation,
recent studies demonstrate that employing additional Chain-of-Thought
generation models can significantly enhance code generation performance by
providing explicit reasoning steps. However, as external components, CoT models
are particularly vulnerable to backdoor attacks, which existing defense
mechanisms often fail to detect effectively. To address this challenge, we
propose GUARD, a novel dual-agent defense framework specifically designed to
counter CoT backdoor attacks in neural code generation. GUARD integrates two
core components: GUARD-Judge, which identifies suspicious CoT steps and
potential triggers through comprehensive analysis, and GUARD-Repair, which
employs a retrieval-augmented generation approach to regenerate secure CoT
steps for identified anomalies. Experimental results show that GUARD
effectively mitigates attacks while maintaining generation quality, advancing
secure code generation systems.

### 4. [SV-TrustEval-C: Evaluating Structure and Semantic Reasoning in Large Language Models for Source Code Vulnerability Analysis](http://arxiv.org/pdf/2505.20630v1)

Authors: Yansong Li, Paula Branco, Alexander M. Hoole, Manish Marwah, Hari Manassery Koduvely, Guy-Vincent Jourdan, Stephan Jou

As Large Language Models (LLMs) evolve in understanding and generating code,
accurately evaluating their reliability in analyzing source code
vulnerabilities becomes increasingly vital. While studies have examined LLM
capabilities in tasks like vulnerability detection and repair, they often
overlook the importance of both structure and semantic reasoning crucial for
trustworthy vulnerability analysis. To address this gap, we introduce
SV-TrustEval-C, a benchmark designed to evaluate LLMs' abilities for
vulnerability analysis of code written in the C programming language through
two key dimensions: structure reasoning - assessing how models identify
relationships between code elements under varying data and control flow
complexities; and semantic reasoning - examining their logical consistency in
scenarios where code is structurally and semantically perturbed. Our results
show that current LLMs are far from satisfactory in understanding complex code
relationships and that their vulnerability analyses rely more on pattern
matching than on robust logical reasoning. These findings underscore the
effectiveness of the SV-TrustEval-C benchmark and highlight critical areas for
enhancing the reasoning capabilities and trustworthiness of LLMs in real-world
vulnerability analysis tasks. Our initial benchmark dataset is publicly
available.

### 5. [System-driven Cloud Architecture Design Support with Structured State Management and Guided Decision Assistance](http://arxiv.org/pdf/2505.20701v1)

Authors: Ryosuke Kohita, Akira Kasuga

Cloud architecture design is a complex process requiring both technical
expertise and architectural knowledge to develop solutions from frequently
ambiguous requirements. We present CloudArchitectBuddy, a system-driven cloud
architecture design support application with two key mechanisms: (1) structured
state management that enhances design understanding through explicit
representation of requirements and architectural decisions, and (2) guided
decision assistance that facilitates design progress through proactive
verification and requirement refinement. Our study with 16 industry
practitioners showed that while our approach achieved comparable design quality
to a chat interface, participants rated our system higher for usability and
appreciated its ability to help understand architectural relationships and
identify missing requirements. However, participants also expressed a need for
user-initiated interactions where they could freely provide design instructions
and engage in detailed discussions with LLMs. These results suggest that
integrating a chat interface into our structured and guided workflow approach
would create a more practical solution, balancing systematic design support
with conversational flexibility for comprehensive cloud architecture
development.

### 6. [Can Agents Fix Agent Issues?](http://arxiv.org/pdf/2505.20749v1)

Authors: Alfin Wijaya Rahardja, Junwei Liu, Weitong Chen, Zhenpeng Chen, Yiling Lou

LLM-based agent systems are emerging as a new software paradigm and have been
widely adopted across diverse domains such as medicine, robotics, and
programming. However, maintaining these systems requires substantial effort, as
they are inevitably prone to bugs and continually evolve to meet changing
external requirements. Therefore, automatically resolving agent issues (i.e.,
bug reports or feature requests) is a crucial and challenging task. While
recent software engineering (SE) agents (e.g., SWE-agent) have shown promise in
addressing issues in traditional software systems, it remains unclear how
effectively they can resolve real-world issues in agent systems, which differ
significantly from traditional software. To fill this gap, we first manually
analyze 201 real-world agent issues and identify common categories of agent
issues. We then spend 500 person-hours constructing AGENTISSUE-BENCH, a
reproducible benchmark comprising 50 agent issue resolution tasks (each with an
executable environment and failure-triggering tests). We further evaluate
state-of-the-art SE agents on AGENTISSUE-BENCH and reveal their limited
effectiveness (i.e., with only 3.33% - 12.67% resolution rates). These results
underscore the unique challenges of maintaining agent systems compared to
traditional software, highlighting the need for further research to develop
advanced SE agents for resolving agent issues. Data and code are available at
https://alfin06.github.io/AgentIssue-Bench-Leaderboard/#/ .

### 7. [Towards Conversational Development Environments: Using Theory-of-Mind and Multi-Agent Architectures for Requirements Refinement](http://arxiv.org/pdf/2505.20973v1)

Authors: Keheliya Gallaba, Ali Arabat, Dayi Lin, Mohammed Sayagh, Ahmed E. Hassan

Foundation Models (FMs) have shown remarkable capabilities in various natural
language tasks. However, their ability to accurately capture stakeholder
requirements remains a significant challenge for using FMs for software
development. This paper introduces a novel approach that leverages an
FM-powered multi-agent system called AlignMind to address this issue. By having
a cognitive architecture that enhances FMs with Theory-of-Mind capabilities,
our approach considers the mental states and perspectives of software makers.
This allows our solution to iteratively clarify the beliefs, desires, and
intentions of stakeholders, translating these into a set of refined
requirements and a corresponding actionable natural language workflow in the
often-overlooked requirements refinement phase of software engineering, which
is crucial after initial elicitation. Through a multifaceted evaluation
covering 150 diverse use cases, we demonstrate that our approach can accurately
capture the intents and requirements of stakeholders, articulating them as both
specifications and a step-by-step plan of action. Our findings suggest that the
potential for significant improvements in the software development process
justifies these investments. Our work lays the groundwork for future innovation
in building intent-first development environments, where software makers can
seamlessly collaborate with AIs to create software that truly meets their
needs.

### 8. [ColorGo: Directed Concolic Execution](http://arxiv.org/pdf/2505.21130v1)

Authors: Jia Li, Jiacheng Shen, Yuxin Su, Michael R. Lyu

Directed fuzzing is a critical technique in cybersecurity, targeting specific
sections of a program. This approach is essential in various security-related
domains such as crash reproduction, patch testing, and vulnerability detection.
Despite its importance, current directed fuzzing methods exhibit a trade-off
between efficiency and effectiveness. For instance, directed grey-box fuzzing,
while efficient in generating fuzzing inputs, lacks sufficient precision. The
low precision causes time wasted on executing code that cannot help reach the
target site. Conversely, interpreter- or observer-based directed symbolic
execution can produce high-quality inputs while incurring non-negligible
runtime overhead. These limitations undermine the feasibility of directed
fuzzers in real-world scenarios. To kill the birds of efficiency and
effectiveness with one stone, in this paper, we involve compilation-based
concolic execution into directed fuzzing and present ColorGo, achieving high
scalability while preserving the high precision from symbolic execution.
ColorGo is a new directed whitebox fuzzer that concretely executes the
instrumented program with constraint-solving capability on generated input. It
guides the exploration by \textit{incremental coloration}, including static
reachability analysis and dynamic feasibility analysis. We evaluated ColorGo on
diverse real-world programs and demonstrated that ColorGo outperforms AFLGo by
up to \textbf{100x} in reaching target sites and reproducing target crashes.

### 9. [How Do Experts Make Sense of Integrated Process Models?](http://arxiv.org/pdf/2505.20667v1)

Authors: Tianwa Chen, Barbara Weber, Graeme Shanks, Gianluca Demartini, Marta Indulska, Shazia Sadiq

A range of integrated modeling approaches have been developed to enable a
holistic representation of business process logic together with all relevant
business rules. These approaches address inherent problems with separate
documentation of business process models and business rules. In this study, we
explore how expert process workers make sense of the information provided
through such integrated modeling approaches. To do so, we complement verbal
protocol analysis with eye-tracking metrics to reveal nuanced user behaviours
involved in the main phases of sensemaking, namely information foraging and
information processing. By studying expert process workers engaged in tasks
based on integrated modeling of business processes and rules, we provide
insights that pave the way for a better understanding of sensemaking practices
and improved development of business process and business rule integration
approaches. Our research underscores the importance of offering personalized
support mechanisms that increase the efficacy and efficiency of sensemaking
practices for process knowledge workers.

### 10. [An LLM-as-Judge Metric for Bridging the Gap with Human Evaluation in SE Tasks](http://arxiv.org/pdf/2505.20854v1)

Authors: Xin Zhou, Kisub Kim, Ting Zhang, Martin Weyssow, Luis F. Gomes, Guang Yang, David Lo

Large Language Models (LLMs) and other automated techniques have been
increasingly used to support software developers by generating software
artifacts such as code snippets, patches, and comments. However, accurately
assessing the correctness of these generated artifacts remains a significant
challenge. On one hand, human evaluation provides high accuracy but is
labor-intensive and lacks scalability. On the other hand, other existing
automatic evaluation metrics are scalable and require minimal human effort, but
they often fail to accurately reflect the actual correctness of generated
software artifacts.
  In this paper, we present SWE-Judge, the first evaluation metric for
LLM-as-Ensemble-Judge specifically designed to accurately assess the
correctness of generated software artifacts. SWE-Judge first defines five
distinct evaluation strategies, each implemented as an independent judge. A
dynamic team selection mechanism then identifies the most appropriate subset of
judges to produce a final correctness score through ensembling. We evaluate
SWE-Judge across a diverse set of software engineering (SE) benchmarks,
including CoNaLa, Card2Code, HumanEval-X, APPS, APR-Assess, and Summary-Assess.
These benchmarks span three SE tasks: code generation, automated program
repair, and code summarization. Experimental results demonstrate that SWE-Judge
consistently achieves a higher correlation with human judgments, with
improvements ranging from 5.9% to 183.8% over existing automatic metrics.
Furthermore, SWE-Judge reaches agreement levels with human annotators that are
comparable to inter-annotator agreement in code generation and program repair
tasks. These findings underscore SWE-Judge's potential as a scalable and
reliable alternative to human evaluation.

### Social and Information Networks

### 1. [Fedivertex: a Graph Dataset based on Decentralized Social Networks for Trustworthy Machine Learning](http://arxiv.org/pdf/2505.20882v1)

Authors: Marc Damie, Edwige Cyffers

Decentralized machine learning - where each client keeps its own data locally
and uses its own computational resources to collaboratively train a model by
exchanging peer-to-peer messages - is increasingly popular, as it enables
better scalability and control over the data. A major challenge in this setting
is that learning dynamics depend on the topology of the communication graph,
which motivates the use of real graph datasets for benchmarking decentralized
algorithms. Unfortunately, existing graph datasets are largely limited to
for-profit social networks crawled at a fixed point in time and often collected
at the user scale, where links are heavily influenced by the platform and its
recommendation algorithms. The Fediverse, which includes several free and
open-source decentralized social media platforms such as Mastodon, Misskey, and
Lemmy, offers an interesting real-world alternative. We introduce Fedivertex, a
new dataset of 182 graphs, covering seven social networks from the Fediverse,
crawled weekly over 14 weeks. We release the dataset along with a Python
package to facilitate its use, and illustrate its utility on several tasks,
including a new defederation task, which captures a process of link deletion
observed on these networks.

### 2. [Identifying Super Spreaders in Multilayer Networks](http://arxiv.org/pdf/2505.20980v1)

Authors: Michał Czuba, Mateusz Stolarski, Adam Piróg, Piotr Bielak, Piotr Bródka

Identifying super-spreaders can be framed as a subtask of the influence
maximisation problem. It seeks to pinpoint agents within a network that, if
selected as single diffusion seeds, disseminate information most effectively.
Multilayer networks, a specific class of heterogeneous graphs, can capture
diverse types of interactions (e.g., physical-virtual or professional-social),
and thus offer a more accurate representation of complex relational structures.
In this work, we introduce a novel approach to identifying super-spreaders in
such networks by leveraging graph neural networks. To this end, we construct a
dataset by simulating information diffusion across hundreds of networks - to
the best of our knowledge, the first of its kind tailored specifically to
multilayer networks. We further formulate the task as a variation of the
ranking prediction problem based on a four-dimensional vector that quantifies
each agent's spreading potential: (i) the number of activations; (ii) the
duration of the diffusion process; (iii) the peak number of activations; and
(iv) the simulation step at which this peak occurs. Our model,
TopSpreadersNetwork, comprises a relationship-agnostic encoder and a custom
aggregation layer. This design enables generalisation to previously unseen data
and adapts to varying graph sizes. In an extensive evaluation, we compare our
model against classic centrality-based heuristics and competitive deep learning
methods. The results, obtained across a broad spectrum of real-world and
synthetic multilayer networks, demonstrate that TopSpreadersNetwork achieves
superior performance in identifying high-impact nodes, while also offering
improved interpretability through its structured output.

### 3. [Efficient Identity and Position Graph Embedding via Spectral-Based Random Feature Aggregation](http://arxiv.org/pdf/2505.20992v1)

Authors: Meng Qin, Jiahong Liu, Irwin King

Graph neural networks (GNNs), which capture graph structures via a feature
aggregation mechanism following the graph embedding framework, have
demonstrated a powerful ability to support various tasks. According to the
topology properties (e.g., structural roles or community memberships of nodes)
to be preserved, graph embedding can be categorized into identity and position
embedding. However, it is unclear for most GNN-based methods which property
they can capture. Some of them may also suffer from low efficiency and
scalability caused by several time- and space-consuming procedures (e.g.,
feature extraction and training). From a perspective of graph signal
processing, we find that high- and low-frequency information in the graph
spectral domain may characterize node identities and positions, respectively.
Based on this investigation, we propose random feature aggregation (RFA) for
efficient identity and position embedding, serving as an extreme ablation study
regarding GNN feature aggregation. RFA (i) adopts a spectral-based GNN without
learnable parameters as its backbone, (ii) only uses random noises as inputs,
and (iii) derives embeddings via just one feed-forward propagation (FFP).
Inspired by degree-corrected spectral clustering, we further introduce a degree
correction mechanism to the GNN backbone. Surprisingly, our experiments
demonstrate that two variants of RFA with high- and low-pass filters can
respectively derive informative identity and position embeddings via just one
FFP (i.e., without any training). As a result, RFA can achieve a better
trade-off between quality and efficiency for both identity and position
embedding over various baselines.

### 4. [Two-step dimensionality reduction of human mobility data: From potential landscapes to spatiotemporal insights](http://arxiv.org/pdf/2505.20929v1)

Authors: Yunhan Du, Takaaki Aoki, Naoya Fujiwara

Understanding the spatiotemporal patterns of human mobility is crucial for
addressing societal challenges, such as epidemic control and urban
transportation optimization. Despite advancements in data collection, the
complexity and scale of mobility data continue to pose significant analytical
challenges. Existing methods often result in losing location-specific details
and fail to fully capture the intricacies of human movement. This study
proposes a two-step dimensionality reduction framework to overcome existing
limitations. First, we construct a potential landscape of human flow from
origin-destination (OD) matrices using combinatorial Hodge theory, preserving
essential spatial and structural information while enabling an intuitive
visualization of flow patterns. Second, we apply principal component analysis
(PCA) to the potential landscape, systematically identifying major
spatiotemporal patterns. By implementing this two-step reduction method, we
reveal significant shifts during a pandemic, characterized by an overall
declines in mobility and stark contrasts between weekdays and holidays. These
findings underscore the effectiveness of our framework in uncovering complex
mobility patterns and provide valuable insights into urban planning and public
health interventions.

### 5. [Leveraging GANs for citation intent classification and its impact on citation network analysis](http://arxiv.org/pdf/2505.21162v1)

Authors: Davi A. Bezerra, Filipi N. Silva, Diego R. Amancio

Citations play a fundamental role in the scientific ecosystem, serving as a
foundation for tracking the flow of knowledge, acknowledging prior work, and
assessing scholarly influence. In scientometrics, they are also central to the
construction of quantitative indicators. Not all citations, however, serve the
same function: some provide background, others introduce methods, or compare
results. Therefore, understanding citation intent allows for a more nuanced
interpretation of scientific impact. In this paper, we adopted a GAN-based
method to classify citation intents. Our results revealed that the proposed
method achieves competitive classification performance, closely matching
state-of-the-art results with substantially fewer parameters. This demonstrates
the effectiveness and efficiency of leveraging GAN architectures combined with
contextual embeddings in intent classification task. We also investigated
whether filtering citation intents affects the centrality of papers in citation
networks. Analyzing the network constructed from the unArXiv dataset, we found
that paper rankings can be significantly influenced by citation intent. All
four centrality metrics examined- degree, PageRank, closeness, and betweenness
- were sensitive to the filtering of citation types. The betweenness centrality
displayed the greatest sensitivity, showing substantial changes in ranking when
specific citation intents were removed.

### 6. [DeSocial: Blockchain-based Decentralized Social Networks](http://arxiv.org/pdf/2505.21388v1)

Authors: Jingyuan Huang, Xi Zhu, Minghao Guo, Yongfeng Zhang

Web 2.0 social platforms are inherently centralized, with user data and
algorithmic decisions controlled by the platform. However, users can only
passively receive social predictions without being able to choose the
underlying algorithm, which limits personalization. Fortunately, with the
emergence of blockchain, users are allowed to choose algorithms that are
tailored to their local situation, improving prediction results in a
personalized way. In a blockchain environment, each user possesses its own
model to perform the social prediction, capturing different perspectives on
social interactions. In our work, we propose DeSocial, a decentralized social
network learning framework deployed on an Ethereum (ETH) local development
chain that integrates distributed data storage, node-level consensus, and
user-driven model selection through Ganache. In the first stage, each user
leverages DeSocial to evaluate multiple backbone models on their local
subgraph. DeSocial coordinates the execution and returns model-wise prediction
results, enabling the user to select the most suitable backbone for
personalized social prediction. Then, DeSocial uniformly selects several
validation nodes that possess the algorithm specified by each user, and
aggregates the prediction results by majority voting, to prevent errors caused
by any single model's misjudgment. Extensive experiments show that DeSocial has
an evident improvement compared to the five classical centralized social
network learning models, promoting user empowerment in blockchain-based
decentralized social networks, showing the importance of multi-node validation
and personalized algorithm selection based on blockchain. Our implementation is
available at: https://github.com/agiresearch/DeSocial.

### 7. [Larger cities, more commuters, more crime? The role of inter-city commuting in the scaling of urban crime](http://arxiv.org/pdf/2505.20822v1)

Authors: Simon Puttock, Umberto Barros, Diego Pinheiro, Marcos Oliveira

Cities attract a daily influx of non-resident commuters, reflecting their
role in wider urban networks -- not as isolated places. However, it remains
unclear how this inter-connectivity shapes the way crime scales with
population, given that larger cities tend to receive more commuters and
experience more crime. Here, we investigate how inter-city commuting relates to
the population--crime relationship. We find that larger cities receive
proportionately more commuters, which in turn is associated with higher crime
levels. Specifically, each 1% increase in inbound commuters corresponds to a
0.32% rise in theft and 0.20% rise in burglary, holding population constant. We
show that models incorporating both population and commuter inflows better
explain crime variation than population-only models. These findings underscore
the importance of considering how cities are connected -- not just their
population size -- in disentangling the population--crime relationship.

### Systems and Control

### 1. [On Kernel Design for Regularized Volterra Series Identification of Wiener-Hammerstein Systems](http://arxiv.org/pdf/2505.20747v1)

Authors: Yu Xu, Biqiang Mu, Tianshi Chen

There have been increasing interests on the Volterra series identification
with the kernel-based regularization method. The major difficulties are on the
kernel design and efficiency of the corresponding implementation. In this
paper, we first assume that the underlying system to be identified is the
Wiener-Hammerstein (WH) system with polynomial nonlinearity. We then show how
to design kernels with nonzero off-diagonal blocks for Volterra maps by taking
into account the prior knowledge of the linear blocks and the structure of WH
systems. Moreover, exploring the structure of the designed kernels leads to the
same computational complexity as the state-of-the-art result, i.e., $O(N^3)$,
where $N$ is the sample size, but with a significant difference that the
proposed kernels are designed in a direct and flexible way. In addition, for a
special case of the kernel and a class of widely used input signals, further
exploring the separable structure of the output kernel matrix can lower the
computational complexity from $O(N^3)$ to $O(N\gamma^2)$, where $\gamma$ is the
separability rank of the output kernel matrix and can be much smaller than $N$.
We finally run Monte Carlo simulations to demonstrate the proposed kernels and
the obtained theoretical results.

### 2. [Data-Driven Existence and Design of Target Output Controllers](http://arxiv.org/pdf/2505.20750v1)

Authors: Yuan Zhang, Wenxuan Xu, Mohamed Darouach, Tyrone Fernando

Target output controllers aim at regulating a system's target outputs by
placing poles of a suitable subsystem using partial state feedback, where full
state controllability is not required. This paper establishes existence
conditions for such controllers using input and partial state data, where the
system dynamics are unknown. The approach bypasses traditional system
identification steps and leverages the intrinsic structure of historical data
to certify controller existence and synthesize a suitable feedback gain.
Analytical characterizations are provided, ensuring that the resulting
closed-loop system satisfies desired performance objectives such as pole
placement or stabilization. Data-driven algorithms are then proposed to design
target output controllers directly from data without identifying system
parameters, where controllers with the order matching the number of target
outputs and with minimum-order augmented target outputs are both addressed.
Furthermore, a separation principle is revealed, decoupling the design of
target output controllers from state observers. This enables the development of
data-driven observer-based controllers that integrate estimation and control.
Numerical examples validate the theoretical results and demonstrate the
efficacy of the proposed approach.

### 3. [Physics-Informed Neural Network for Cross-Domain Predictive Control of Tapered Amplifier Thermal Stabilization](http://arxiv.org/pdf/2505.20769v1)

Authors: Yanpei Shi, Bo Feng, Yuxin Zhong, Haochen Guo, Bangcheng Han, Rui Feng

Thermally induced laser noise poses a critical limitation to the sensitivity
of quantum sensor arrays employing ultra-stable amplified lasers, primarily
stemming from nonlinear gain-temperature coupling effects in tapered amplifiers
(TAs). To address this challenge, we present a robust intelligent control
strategy that synergistically integrates an encoder-decoder physics-informed
gated recurrent unit (PI-GRU) network with a model predictive control (MPC)
framework. Our methodology incorporates physical soft constraints into the
neural network architecture, yielding a predictive model with enhanced physical
consistency that demonstrates robust extrapolation capabilities beyond the
training data distribution. Leveraging the PI-GRU model's accurate multi-step
predictive performance, we implement a hierarchical parallel MPC architecture
capable of real-time thermal instability compensation. This hybrid approach
achieves cross-domain consistent thermal stabilization in TAs under diverse
laser power operations. Remarkably, while trained exclusively on low-power
operational data, our system demonstrates exceptional generalization, improving
prediction accuracy by 58.2% and temperature stability by 69.1% in previously
unseen high-power operating regimes, as experimentally validated. The novel
synchronization of physics-informed neural networks with advanced MPC
frameworks presented in this work establishes a groundbreaking paradigm for
addressing robustness challenges in cross-domain predictive control
applications, overcoming conventional modeling limitations.

### 4. [Effective Fixed-Time Control for Constrained Nonlinear System](http://arxiv.org/pdf/2505.20870v1)

Authors: Chenglin Gong, Ziming Wang, Guanxuan Jiang, Xin Wang, Yiding Ji

In this paper, we tackle the state transformation problem in non-strict full
state-constrained systems by introducing an adaptive fixed-time control method,
utilizing a one-to-one asymmetric nonlinear mapping auxiliary system.
Additionally, we develop a class of multi-threshold event-triggered control
strategies that facilitate autonomous controller updates, substantially
reducing communication resource consumption. Notably, the self-triggered
strategy distinguishes itself from other strategies by obviating the need for
continuous real-time monitoring of the controller's state variables. By
accurately forecasting the subsequent activation instance, this strategy
significantly optimizes the efficiency of the control system. Moreover, our
theoretical analysis demonstrates that the semi-global practical fixed-time
stability (SPFTS) criterion guarantees both tracking accuracy and closed-loop
stability under state constraints, with convergence time independent of initial
conditions. Finally, simulation results reveal that the proposed method
significantly decreases the frequency of control command updates while
maintaining tracking accuracy.

### 5. [Research on a Two-Layer Demand Response Framework for Electric Vehicle Users and Aggregators Based on LLMs](http://arxiv.org/pdf/2505.20877v1)

Authors: Zhaoyi Zhang, Chenggang Cui, Ning Yang, Chuanlin Zhang

The widespread adoption of electric vehicles (EVs) has increased the
importance of demand response in smart grids. This paper proposes a two-layer
demand response optimization framework for EV users and aggregators, leveraging
large language models (LLMs) to balance electricity supply and demand and
optimize energy utilization during EV charging. The upper-layer model, focusing
on the aggregator, aims to maximize profits by adjusting retail electricity
prices. The lower-layer model targets EV users, using LLMs to simulate charging
demands under varying electricity prices and optimize both costs and user
comfort. The study employs a multi-threaded LLM decision generator to
dynamically analyze user behavior, charging preferences, and psychological
factors. The framework utilizes the PSO method to optimize electricity prices,
ensuring user needs are met while increasing aggregator profits. Simulation
results show that the proposed model improves EV charging efficiency,
alleviates peak power loads, and stabilizes smart grid operations.

### 6. [Active Learning-Enhanced Dual Control for Angle-Only Initial Relative Orbit Determination](http://arxiv.org/pdf/2505.21248v1)

Authors: Kui Xie, Giovanni Romagnoli, Giordana Bucchioni, Alberto Bemporad

Accurate relative orbit determination is a key challenge in modern space
operations, particularly when relying on angle-only measurements. The inherent
observability limitations of this approach make initial state estimation
difficult, impacting mission safety and performance. This work explores the use
of active learning (AL) techniques to enhance observability by dynamically
designing the input excitation signal offline and at runtime. Our approach
leverages AL to design the input signal dynamically, enhancing the
observability of the system without requiring additional hardware or predefined
maneuvers. We incorporate a dual control technique to ensure target tracking
while maintaining observability. The proposed method is validated through
numerical simulations, demonstrating its effectiveness in estimating the
initial relative state of the chaser and target spacecrafts and its robustness
to various initial relative distances and observation periods.

### 7. [Stochastic Geometry-Based Performance Evaluation for LEO Satellite-Assisted Space Caching](http://arxiv.org/pdf/2505.21259v1)

Authors: Chunyi Ma, Jiajie Xu, Jianhua Yang, Mustafa A. Kishk

To achieve the Internet of Things (IoT) vision,Mobile Edge Computing (MEC) is
a promising technology aimed at providing low-latency computing services to
user equipment (UE). However, terrestrial MEC network struggles to provide
service to UEs in remote and maritime region. Low Earth Orbit (LEO) satellite
networks have the potential to overcome geographical restrictions and provide
seamless global coverage for UEs. In this paper, we provide the first attempt
to use stochastic geometry to investigate the performance of implementing space
caching with LEO satellites (SATs) in the MEC network. We study a LEO
satellite-assisted space caching MEC network, and LEO SATs can be equipped with
servers to enable space caching, with the advantage of seamless coverage to
assist terrestrial CSs for serving UEs in remote or maritime reigon. Using
stochastic geometry and queuing theory, we establish an analytical framework
for this MEC network. Meanwhile, we develop association strategies for UEs to
connect with LEO SATs or CSs and utilize stochastic geometry to derive uplink
and downlink coverage probabilities, considering the diversity of task and
service types. On this basis, we employ the queuing theory to calculate the
average delay to evaluate the system performance. Through Monte Carlo
simulations and numerical results, the system performance is evaluated. The
results show the potential of SAT spatial caching in improving the performance
of the MEC network. Additionally, our results reveal useful insights such as
the significant impact of the altitude and number of LEO SATs on the average
delay of the network, providing helpful system-level recommendations for the
design and configuration of the space-caching MEC network.

### 8. [Quasi Steady-State Frequency](http://arxiv.org/pdf/2505.21461v1)

Authors: Joan Gutierrez-Florensa, Alvaro Ortega, Lukas Sigrist, Federico Milano

Accurate frequency estimation is critical for the control, monitoring and
protection of electrical power systems, in particular, of systems with a high
penetration of power electronics. This paper introduces the novel concept of
Quasi Steady-State (QSS) frequency as a quantity that fills the gap between
stationary and instantaneous frequency. QSS frequency coincides with the
fundamental frequency of an AC voltage in any stationary conditions, including
unbalanced and non-sinusoidal, and is able to capture the time-varying
fundamental frequency in transient conditions. The paper also proposes a metric
borrowed from fluid dynamics, namely, the time derivative of the circulation,
to define the scope of validity of the QSS frequency. Analytical examples as
well as a case study based on a fully-fledged EMT model of the IEEE 39-bus
system serve to illustrate, respectively, the properties of the QSS frequency
and its behavior in transient conditions.

### 9. [Least Squares Model Reduction: A Two-Stage System-Theoretic Interpretation](http://arxiv.org/pdf/2505.20604v1)

Authors: Alberto Padoan

Model reduction simplifies complex dynamical systems while preserving
essential properties. This paper revisits a recently proposed system-theoretic
framework for least squares moment matching. It interprets least squares model
reduction in terms of two steps process: constructing a surrogate model to
satisfy interpolation constraints, then projecting it onto a reduced-order
space. Using tools from output regulation theory and Krylov projections, this
approach provides a new view on classical methods. For illustration, we
reexamine the least-squares model reduction method by Lucas and Smith, offering
new insights into its structure.

### 10. [Collision-free Control Barrier Functions for General Ellipsoids via Separating Hyperplane](http://arxiv.org/pdf/2505.20847v1)

Authors: Zeming Wu, Lu Liu

This paper presents a novel collision avoidance method for general ellipsoids
based on control barrier functions (CBFs) and separating hyperplanes. First,
collision-free conditions for general ellipsoids are analytically derived using
the concept of dual cones. These conditions are incorporated into the CBF
framework by extending the system dynamics of controlled objects with
separating hyperplanes, enabling efficient and reliable collision avoidance.
The validity of the proposed collision-free CBFs is rigorously proven, ensuring
their effectiveness in enforcing safety constraints. The proposed method
requires only single-level optimization, significantly reducing computational
time compared to state-of-the-art methods. Numerical simulations and real-world
experiments demonstrate the effectiveness and practicality of the proposed
algorithm.

### Machine Learning (Statistics Category)

### 1. [Fundamental Limits of Game-Theoretic LLM Alignment: Smith Consistency and Preference Matching](http://arxiv.org/pdf/2505.20627v1)

Authors: Zhekun Shi, Kaizhao Liu, Qi Long, Weijie J. Su, Jiancong Xiao

Nash Learning from Human Feedback is a game-theoretic framework for aligning
large language models (LLMs) with human preferences by modeling learning as a
two-player zero-sum game. However, using raw preference as the payoff in the
game highly limits the potential of the game-theoretic LLM alignment framework.
In this paper, we systematically study using what choices of payoff based on
the pairwise human preferences can yield desirable alignment properties. We
establish necessary and sufficient conditions for Condorcet consistency,
diversity through mixed strategies, and Smith consistency. These results
provide a theoretical foundation for the robustness of game-theoretic LLM
alignment. Further, we show the impossibility of preference matching -- i.e.,
no smooth and learnable mappings of pairwise preferences can guarantee a unique
Nash equilibrium that matches a target policy, even under standard assumptions
like the Bradley-Terry-Luce model. This result highlights the fundamental
limitation of game-theoretic LLM alignment.

### 2. [Explaining Concept Shift with Interpretable Feature Attribution](http://arxiv.org/pdf/2505.20634v1)

Authors: Ruiqi Lyu, Alistair Turcan, Bryan Wilder

Regardless the amount of data a machine learning (ML) model is trained on,
there will inevitably be data that differs from their training set, lowering
model performance. Concept shift occurs when the distribution of labels
conditioned on the features changes, making even a well-tuned ML model to have
learned a fundamentally incorrect representation. Identifying these shifted
features provides unique insight into how one dataset differs from another,
considering the difference may be across a scientifically relevant dimension,
such as time, disease status, population, etc. In this paper, we propose
SGShift, a model for detecting concept shift in tabular data and attributing
reduced model performance to a sparse set of shifted features. SGShift models
concept shift with a Generalized Additive Model (GAM) and performs subsequent
feature selection to identify shifted features. We propose further extensions
of SGShift by incorporating knockoffs to control false discoveries and an
absorption term to account for models with poor fit to the data. We conduct
extensive experiments in synthetic and real data across various ML models and
find SGShift can identify shifted features with AUC $>0.9$ and recall $>90\%$,
often 2 or 3 times as high as baseline methods.

### 3. [Stationary MMD Points for Cubature](http://arxiv.org/pdf/2505.20754v1)

Authors: Zonghao Chen, Toni Karvonen, Heishiro Kanagawa, François-Xavier Briol, Chris. J. Oates

Approximation of a target probability distribution using a finite set of
points is a problem of fundamental importance, arising in cubature, data
compression, and optimisation. Several authors have proposed to select points
by minimising a maximum mean discrepancy (MMD), but the non-convexity of this
objective precludes global minimisation in general. Instead, we consider
\emph{stationary} points of the MMD which, in contrast to points globally
minimising the MMD, can be accurately computed. Our main theoretical
contribution is the (perhaps surprising) result that, for integrands in the
associated reproducing kernel Hilbert space, the cubature error of stationary
MMD points vanishes \emph{faster} than the MMD. Motivated by this
\emph{super-convergence} property, we consider discretised gradient flows as a
practical strategy for computing stationary points of the MMD, presenting a
refined convergence analysis that establishes a novel non-asymptotic
finite-particle error bound, which may be of independent interest.

### 4. [Practical estimation of the optimal classification error with soft labels and calibration](http://arxiv.org/pdf/2505.20761v1)

Authors: Ryota Ushio, Takashi Ishida, Masashi Sugiyama

While the performance of machine learning systems has experienced significant
improvement in recent years, relatively little attention has been paid to the
fundamental question: to what extent can we improve our models? This paper
provides a means of answering this question in the setting of binary
classification, which is practical and theoretically supported. We extend a
previous work that utilizes soft labels for estimating the Bayes error, the
optimal error rate, in two important ways. First, we theoretically investigate
the properties of the bias of the hard-label-based estimator discussed in the
original work. We reveal that the decay rate of the bias is adaptive to how
well the two class-conditional distributions are separated, and it can decay
significantly faster than the previous result suggested as the number of hard
labels per instance grows. Second, we tackle a more challenging problem
setting: estimation with corrupted soft labels. One might be tempted to use
calibrated soft labels instead of clean ones. However, we reveal that
calibration guarantee is not enough, that is, even perfectly calibrated soft
labels can result in a substantially inaccurate estimate. Then, we show that
isotonic calibration can provide a statistically consistent estimator under an
assumption weaker than that of the previous work. Our method is instance-free,
i.e., we do not assume access to any input instances. This feature allows it to
be adopted in practical scenarios where the instances are not available due to
privacy issues. Experiments with synthetic and real-world datasets show the
validity of our methods and theory.

### 5. [Improved Bounds for Swap Multicalibration and Swap Omniprediction](http://arxiv.org/pdf/2505.20885v1)

Authors: Haipeng Luo, Spandan Senapati, Vatsal Sharan

In this paper, we consider the related problems of multicalibration -- a
multigroup fairness notion and omniprediction -- a simultaneous loss
minimization paradigm, both in the distributional and online settings. The
recent work of Garg et al. (2024) raised the open problem of whether it is
possible to efficiently achieve $O(\sqrt{T})$ $\ell_{2}$-multicalibration error
against bounded linear functions. In this paper, we answer this question in a
strongly affirmative sense. We propose an efficient algorithm that achieves
$O(T^{\frac{1}{3}})$ $\ell_{2}$-swap multicalibration error (both in high
probability and expectation). On propagating this bound onward, we obtain
significantly improved rates for $\ell_{1}$-swap multicalibration and swap
omniprediction for a loss class of convex Lipschitz functions. In particular,
we show that our algorithm achieves $O(T^{\frac{2}{3}})$ $\ell_{1}$-swap
multicalibration and swap omniprediction errors, thereby improving upon the
previous best-known bound of $O(T^{\frac{7}{8}})$. As a consequence of our
improved online results, we further obtain several improved sample complexity
rates in the distributional setting. In particular, we establish a
$O(\varepsilon ^ {-3})$ sample complexity of efficiently learning an
$\varepsilon$-swap omnipredictor for the class of convex and Lipschitz
functions, $O(\varepsilon ^{-2.5})$ sample complexity of efficiently learning
an $\varepsilon$-swap agnostic learner for the squared loss, and $O(\varepsilon
^ {-5}), O(\varepsilon ^ {-2.5})$ sample complexities of learning $\ell_{1},
\ell_{2}$-swap multicalibrated predictors against linear functions, all of
which significantly improve on the previous best-known bounds.

### 6. [Efficient and Unbiased Sampling from Boltzmann Distributions via Variance-Tuned Diffusion Models](http://arxiv.org/pdf/2505.21005v1)

Authors: Fengzhe Zhang, Laurence I. Midgley, José Miguel Hernández-Lobato

Score-based diffusion models (SBDMs) are powerful amortized samplers for
Boltzmann distributions; however, imperfect score estimates bias downstream
Monte Carlo estimates. Classical importance sampling (IS) can correct this
bias, but computing exact likelihoods requires solving the probability-flow
ordinary differential equation (PF-ODE), a procedure that is prohibitively
costly and scales poorly with dimensionality. We introduce Variance-Tuned
Diffusion Importance Sampling (VT-DIS), a lightweight post-training method that
adapts the per-step noise covariance of a pretrained SBDM by minimizing the
$\alpha$-divergence ($\alpha=2$) between its forward diffusion and reverse
denoising trajectories. VT-DIS assigns a single trajectory-wise importance
weight to the joint forward-reverse process, yielding unbiased expectation
estimates at test time with negligible overhead compared to standard sampling.
On the DW-4, LJ-13, and alanine-dipeptide benchmarks, VT-DIS achieves effective
sample sizes of approximately 80 %, 35 %, and 3.5 %, respectively, while using
only a fraction of the computational budget required by vanilla diffusion + IS
or PF-ODE-based IS.

### 7. [Bridging Arbitrary and Tree Metrics via Differentiable Gromov Hyperbolicity](http://arxiv.org/pdf/2505.21073v1)

Authors: Pierre Houedry, Nicolas Courty, Florestan Martin-Baillon, Laetitia Chapel, Titouan Vayer

Trees and the associated shortest-path tree metrics provide a powerful
framework for representing hierarchical and combinatorial structures in data.
Given an arbitrary metric space, its deviation from a tree metric can be
quantified by Gromov's $\delta$-hyperbolicity. Nonetheless, designing
algorithms that bridge an arbitrary metric to its closest tree metric is still
a vivid subject of interest, as most common approaches are either heuristical
and lack guarantees, or perform moderately well. In this work, we introduce a
novel differentiable optimization framework, coined DeltaZero, that solves this
problem. Our method leverages a smooth surrogate for Gromov's
$\delta$-hyperbolicity which enables a gradient-based optimization, with a
tractable complexity. The corresponding optimization procedure is derived from
a problem with better worst case guarantees than existing bounds, and is
justified statistically. Experiments on synthetic and real-world datasets
demonstrate that our method consistently achieves state-of-the-art distortion.

### 8. [Robust and Computation-Aware Gaussian Processes](http://arxiv.org/pdf/2505.21133v1)

Authors: Marshal Arijona Sinaga, Julien Martinelli, Samuel Kaski

Gaussian processes (GPs) are widely used for regression and optimization
tasks such as Bayesian optimization (BO) due to their expressiveness and
principled uncertainty estimates. However, in settings with large datasets
corrupted by outliers, standard GPs and their sparse approximations struggle
with computational tractability and robustness. We introduce Robust
Computation-aware Gaussian Process (RCaGP), a novel GP model that jointly
addresses these challenges by combining a principled treatment of
approximation-induced uncertainty with robust generalized Bayesian updating.
The key insight is that robustness and approximation-awareness are not
orthogonal but intertwined: approximations can exacerbate the impact of
outliers, and mitigating one without the other is insufficient. Unlike previous
work that focuses narrowly on either robustness or approximation quality, RCaGP
combines both in a principled and scalable framework, thus effectively managing
both outliers and computational uncertainties introduced by approximations such
as low-rank matrix multiplications. Our model ensures more conservative and
reliable uncertainty estimates, a property we rigorously demonstrate.
Additionally, we establish a robustness property and show that the mean
function is key to preserving it, motivating a tailored model selection scheme
for robust mean functions. Empirical results confirm that solving these
challenges jointly leads to superior performance across both clean and
outlier-contaminated settings, both on regression and high-throughput Bayesian
optimization benchmarks.

### 9. [Learnable Kernel Density Estimation for Graphs](http://arxiv.org/pdf/2505.21285v1)

Authors: Xudong Wang, Ziheng Sun, Chris Ding, Jicong Fan

This work proposes a framework LGKDE that learns kernel density estimation
for graphs. The key challenge in graph density estimation lies in effectively
capturing both structural patterns and semantic variations while maintaining
theoretical guarantees. Combining graph kernels and kernel density estimation
(KDE) is a standard approach to graph density estimation, but has
unsatisfactory performance due to the handcrafted and fixed features of
kernels. Our method LGKDE leverages graph neural networks to represent each
graph as a discrete distribution and utilizes maximum mean discrepancy to learn
the graph metric for multi-scale KDE, where all parameters are learned by
maximizing the density of graphs relative to the density of their well-designed
perturbed counterparts. The perturbations are conducted on both node features
and graph spectra, which helps better characterize the boundary of normal
density regions. Theoretically, we establish consistency and convergence
guarantees for LGKDE, including bounds on the mean integrated squared error,
robustness, and complexity. We validate LGKDE by demonstrating its
effectiveness in recovering the underlying density of synthetic graph
distributions and applying it to graph anomaly detection across diverse
benchmark datasets. Extensive empirical evaluation shows that LGKDE
demonstrates superior performance compared to state-of-the-art baselines on
most benchmark datasets.

### 10. [Conflicting Biases at the Edge of Stability: Norm versus Sharpness Regularization](http://arxiv.org/pdf/2505.21423v1)

Authors: Vit Fojtik, Maria Matveev, Hung-Hsu Chou, Gitta Kutyniok, Johannes Maly

A widely believed explanation for the remarkable generalization capacities of
overparameterized neural networks is that the optimization algorithms used for
training induce an implicit bias towards benign solutions. To grasp this
theoretically, recent works examine gradient descent and its variants in
simplified training settings, often assuming vanishing learning rates. These
studies reveal various forms of implicit regularization, such as $\ell_1$-norm
minimizing parameters in regression and max-margin solutions in classification.
Concurrently, empirical findings show that moderate to large learning rates
exceeding standard stability thresholds lead to faster, albeit oscillatory,
convergence in the so-called Edge-of-Stability regime, and induce an implicit
bias towards minima of low sharpness (norm of training loss Hessian). In this
work, we argue that a comprehensive understanding of the generalization
performance of gradient descent requires analyzing the interaction between
these various forms of implicit regularization. We empirically demonstrate that
the learning rate balances between low parameter norm and low sharpness of the
trained model. We furthermore prove for diagonal linear networks trained on a
simple regression task that neither implicit bias alone minimizes the
generalization error. These findings demonstrate that focusing on a single
implicit bias is insufficient to explain good generalization, and they motivate
a broader view of implicit regularization that captures the dynamic trade-off
between norm and sharpness induced by non-negligible learning rates.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

### 1. [Speed up integer-arithmetic-only inference via bit-shifting](https://www.nature.com/articles/s41598-025-02544-4)

Authors: Mingjun Song et al.

### 2. [Event recognition technology and short-term rockburst early warning model based on microseismic monitoring and ensemble learning](https://www.nature.com/articles/s41598-025-03657-6)

Authors: Zibin Li et al.

### 3. [Solving the Richards infiltration equation by coupling physics-informed neural networks with Hydrus-1D](https://www.nature.com/articles/s41598-025-02978-w)

Authors: Yanling Li et al.

### 4. [Brute-force attack mitigation on remote access services via software-defined perimeter](https://www.nature.com/articles/s41598-025-01080-5)

Authors: Francis A. Ruambo et al.

### 5. [Relative engagement with sources of climate misinformation is growing across social media platforms](https://www.nature.com/articles/s41598-025-03082-9)

Authors: Saverio Storani et al.

### 6. [Design and synthesis of reversible Vedic multiplier using cadence 180 nm technology for low-power high-speed applications](https://www.nature.com/articles/s41598-025-04002-7)

Authors: Narayanan Mageshwari et al.

### 7. [Mitigating malicious denial of wallet attack using attribute reduction with deep learning approach for serverless computing on next generation applications](https://www.nature.com/articles/s41598-025-01178-w)

Authors: Amal K. Alkhalifa et al.

### 8. [DROID: discrete-time simulation for ring-oscillator-based Ising design](https://www.nature.com/articles/s41598-025-00037-y)

Authors: Abhimanyu Kumar et al.

### 9. [Rock image classification based on improved EfficientNet](https://www.nature.com/articles/s41598-025-03706-0)

Authors: Kai Bai et al.

### 10. [Ontology-conformal recognition of materials entities using language models](https://www.nature.com/articles/s41598-025-03619-y)

Authors: Sai Teja Potu et al.

### 11. [The analysis of generative artificial intelligence technology for innovative thinking and strategies in animation teaching](https://www.nature.com/articles/s41598-025-03805-y)

Authors: Xu Yao et al.

