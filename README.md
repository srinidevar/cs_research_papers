# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-07 16:01:09.083762 PST.

### Artificial Intelligence

### 1. Is AI currently capable of identifying wild oysters? A comparison of human annotators against the AI model, ODYSSEE

[Is AI currently capable of identifying wild oysters? A comparison of human annotators against the AI model, ODYSSEE](http://arxiv.org/pdf/2505.03108v1)

Authors: Brendan Campbell, Alan Williams, Kleio Baxevani, Alyssa Campbell, Rushabh Dhoke, Rileigh E. Hudock, Xiaomin Lin, Vivek Mange, Bernhard Neuberger, Arjun Suresh, Alhim Vera, Arthur Trembanis, Herbert G. Tanner, Edward Hale

Oysters are ecologically and commercially important species that require
frequent monitoring to track population demographics (e.g. abundance, growth,
mortality). Current methods of monitoring oyster reefs often require
destructive sampling methods and extensive manual effort. Therefore, they are
suboptimal for small-scale or sensitive environments. A recent alternative, the
ODYSSEE model, was developed to use deep learning techniques to identify live
oysters using video or images taken in the field of oyster reefs to assess
abundance. The validity of this model in identifying live oysters on a reef was
compared to expert and non-expert annotators. In addition, we identified
potential sources of prediction error. Although the model can make inferences
significantly faster than expert and non-expert annotators (39.6 s, $2.34 \pm
0.61$ h, $4.50 \pm 1.46$ h, respectively), the model overpredicted the number
of live oysters, achieving lower accuracy (63\%) in identifying live oysters
compared to experts (74\%) and non-experts (75\%) alike. Image quality was an
important factor in determining the accuracy of the model and the annotators.
Better quality images improved human accuracy and worsened model accuracy.
Although ODYSSEE was not sufficiently accurate, we anticipate that future
training on higher-quality images, utilizing additional live imagery, and
incorporating additional annotation training classes will greatly improve the
model's predictive power based on the results of this analysis. Future research
should address methods that improve the detection of living vs. dead oysters.

### 2. Holmes: Automated Fact Check with Large Language Models

[Holmes: Automated Fact Check with Large Language Models](http://arxiv.org/pdf/2505.03135v1)

Authors: Haoran Ou, Gelei Deng, Xingshuo Han, Jie Zhang, Xinlei He, Han Qiu, Shangwei Guo, Tianwei Zhang

The rise of Internet connectivity has accelerated the spread of
disinformation, threatening societal trust, decision-making, and national
security. Disinformation has evolved from simple text to complex multimodal
forms combining images and text, challenging existing detection methods.
Traditional deep learning models struggle to capture the complexity of
multimodal disinformation. Inspired by advances in AI, this study explores
using Large Language Models (LLMs) for automated disinformation detection. The
empirical study shows that (1) LLMs alone cannot reliably assess the
truthfulness of claims; (2) providing relevant evidence significantly improves
their performance; (3) however, LLMs cannot autonomously search for accurate
evidence. To address this, we propose Holmes, an end-to-end framework featuring
a novel evidence retrieval method that assists LLMs in collecting high-quality
evidence. Our approach uses (1) LLM-powered summarization to extract key
information from open sources and (2) a new algorithm and metrics to evaluate
evidence quality. Holmes enables LLMs to verify claims and generate
justifications effectively. Experiments show Holmes achieves 88.3% accuracy on
two open-source datasets and 90.2% in real-time verification tasks. Notably,
our improved evidence retrieval boosts fact-checking accuracy by 30.8% over
existing methods

### 3. CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics

[CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics](http://arxiv.org/pdf/2505.03171v1)

Authors: Junqi Liu, Xiaohan Lin, Jonas Bayer, Yael Dillies, Weijie Jiang, Xiaodan Liang, Roman Soletskyi, Haiming Wang, Yunzhou Xie, Beibei Xiong, Zhengfeng Yang, Jujian Zhang, Lihong Zhi, Jia Li, Zhengying Liu

Neurosymbolic approaches integrating large language models with formal
reasoning have recently achieved human-level performance on mathematics
competition problems in algebra, geometry and number theory. In comparison,
combinatorics remains a challenging domain, characterized by a lack of
appropriate benchmarks and theorem libraries. To address this gap, we introduce
CombiBench, a comprehensive benchmark comprising 100 combinatorial problems,
each formalized in Lean~4 and paired with its corresponding informal statement.
The problem set covers a wide spectrum of difficulty levels, ranging from
middle school to IMO and university level, and span over ten combinatorial
topics. CombiBench is suitable for testing IMO solving capabilities since it
includes all IMO combinatorial problems since 2000 (except IMO 2004 P3 as its
statement contain an images). Furthermore, we provide a comprehensive and
standardized evaluation framework, dubbed Fine-Eval (for
$\textbf{F}$ill-in-the-blank $\textbf{in}$ L$\textbf{e}$an Evaluation), for
formal mathematics. It accommodates not only proof-based problems but also, for
the first time, the evaluation of fill-in-the-blank questions. Using Fine-Eval
as the evaluation method and Kimina Lean Server as the backend, we benchmark
several LLMs on CombiBench and observe that their capabilities for formally
solving combinatorial problems remain limited. Among all models tested (none of
which has been trained for this particular task), Kimina-Prover attains the
best results, solving 7 problems (out of 100) under both ``with solution'' and
``without solution'' scenarios. We open source the benchmark dataset alongside
with the code of the proposed evaluation method at
https://github.com/MoonshotAI/CombiBench/.

### 4. Artificial Behavior Intelligence: Technology, Challenges, and Future Directions

[Artificial Behavior Intelligence: Technology, Challenges, and Future Directions](http://arxiv.org/pdf/2505.03315v1)

Authors: Kanghyun Jo, Jehwan Choi, Kwanho Kim, Seongmin Kim, Duy-Linh Nguyen, Xuan-Thuy Vo, Adri Priadana, Tien-Dat Tran

Understanding and predicting human behavior has emerged as a core capability
in various AI application domains such as autonomous driving, smart healthcare,
surveillance systems, and social robotics. This paper defines the technical
framework of Artificial Behavior Intelligence (ABI), which comprehensively
analyzes and interprets human posture, facial expressions, emotions, behavioral
sequences, and contextual cues. It details the essential components of ABI,
including pose estimation, face and emotion recognition, sequential behavior
analysis, and context-aware modeling. Furthermore, we highlight the
transformative potential of recent advances in large-scale pretrained models,
such as large language models (LLMs), vision foundation models, and multimodal
integration models, in significantly improving the accuracy and
interpretability of behavior recognition. Our research team has a strong
interest in the ABI domain and is actively conducting research, particularly
focusing on the development of intelligent lightweight models capable of
efficiently inferring complex human behaviors. This paper identifies several
technical challenges that must be addressed to deploy ABI in real-world
applications including learning behavioral intelligence from limited data,
quantifying uncertainty in complex behavior prediction, and optimizing model
structures for low-power, real-time inference. To tackle these challenges, our
team is exploring various optimization strategies including lightweight
transformers, graph-based recognition architectures, energy-aware loss
functions, and multimodal knowledge distillation, while validating their
applicability in real-time environments.

### 5. AI-Driven Scholarly Peer Review via Persistent Workflow Prompting, Meta-Prompting, and Meta-Reasoning

[AI-Driven Scholarly Peer Review via Persistent Workflow Prompting, Meta-Prompting, and Meta-Reasoning](http://arxiv.org/pdf/2505.03332v1)

Authors: Evgeny Markhasin

Critical peer review of scientific manuscripts presents a significant
challenge for Large Language Models (LLMs), partly due to data limitations and
the complexity of expert reasoning. This report introduces Persistent Workflow
Prompting (PWP), a potentially broadly applicable prompt engineering
methodology designed to bridge this gap using standard LLM chat interfaces
(zero-code, no APIs). We present a proof-of-concept PWP prompt for the critical
analysis of experimental chemistry manuscripts, featuring a hierarchical,
modular architecture (structured via Markdown) that defines detailed analysis
workflows. We develop this PWP prompt through iterative application of
meta-prompting techniques and meta-reasoning aimed at systematically codifying
expert review workflows, including tacit knowledge. Submitted once at the start
of a session, this PWP prompt equips the LLM with persistent workflows
triggered by subsequent queries, guiding modern reasoning LLMs through
systematic, multimodal evaluations. Demonstrations show the PWP-guided LLM
identifying major methodological flaws in a test case while mitigating LLM
input bias and performing complex tasks, including distinguishing claims from
evidence, integrating text/photo/figure analysis to infer parameters, executing
quantitative feasibility checks, comparing estimates against claims, and
assessing a priori plausibility. To ensure transparency and facilitate
replication, we provide full prompts, detailed demonstration analyses, and logs
of interactive chats as supplementary resources. Beyond the specific
application, this work offers insights into the meta-development process
itself, highlighting the potential of PWP, informed by detailed workflow
formalization, to enable sophisticated analysis using readily available LLMs
for complex scientific tasks.

### 6. Domain Adversarial Training for Mitigating Gender Bias in Speech-based Mental Health Detection

[Domain Adversarial Training for Mitigating Gender Bias in Speech-based Mental Health Detection](http://arxiv.org/pdf/2505.03359v1)

Authors: June-Woo Kim, Haram Yoon, Wonkyo Oh, Dawoon Jung, Sung-Hoon Yoon, Dae-Jin Kim, Dong-Ho Lee, Sang-Yeol Lee, Chan-Mo Yang

Speech-based AI models are emerging as powerful tools for detecting
depression and the presence of Post-traumatic stress disorder (PTSD), offering
a non-invasive and cost-effective way to assess mental health. However, these
models often struggle with gender bias, which can lead to unfair and inaccurate
predictions. In this study, our study addresses this issue by introducing a
domain adversarial training approach that explicitly considers gender
differences in speech-based depression and PTSD detection. Specifically, we
treat different genders as distinct domains and integrate this information into
a pretrained speech foundation model. We then validate its effectiveness on the
E-DAIC dataset to assess its impact on performance. Experimental results show
that our method notably improves detection performance, increasing the F1-score
by up to 13.29 percentage points compared to the baseline. This highlights the
importance of addressing demographic disparities in AI-driven mental health
assessment.

### 7. STORY2GAME: Generating (Almost) Everything in an Interactive Fiction Game

[STORY2GAME: Generating (Almost) Everything in an Interactive Fiction Game](http://arxiv.org/pdf/2505.03547v1)

Authors: Eric Zhou, Shreyas Basavatia, Moontashir Siam, Zexin Chen, Mark O. Riedl

We introduce STORY2GAME, a novel approach to using Large Language Models to
generate text-based interactive fiction games that starts by generating a
story, populates the world, and builds the code for actions in a game engine
that enables the story to play out interactively. Whereas a given set of
hard-coded actions can artificially constrain story generation, the ability to
generate actions means the story generation process can be more open-ended but
still allow for experiences that are grounded in a game state. The key to
successful action generation is to use LLM-generated preconditions and effects
of actions in the stories as guides for what aspects of the game state must be
tracked and changed by the game engine when a player performs an action. We
also introduce a technique for dynamically generating new actions to
accommodate the player's desire to perform actions that they think of that are
not part of the story. Dynamic action generation may require on-the-fly updates
to the game engine's state representation and revision of previously generated
actions. We evaluate the success rate of action code generation with respect to
whether a player can interactively play through the entire generated story.

### 8. OSUniverse: Benchmark for Multimodal GUI-navigation AI Agents

[OSUniverse: Benchmark for Multimodal GUI-navigation AI Agents](http://arxiv.org/pdf/2505.03570v1)

Authors: Mariya Davydova, Daniel Jeffries, Patrick Barker, Arturo Márquez Flores, Sinéad Ryan

In this paper, we introduce OSUniverse: a benchmark of complex, multimodal
desktop-oriented tasks for advanced GUI-navigation AI agents that focuses on
ease of use, extensibility, comprehensive coverage of test cases, and automated
validation. We divide the tasks in increasing levels of complexity, from basic
precision clicking to multistep, multiapplication tests requiring dexterity,
precision, and clear thinking from the agent. In version one of the benchmark,
presented here, we have calibrated the complexity of the benchmark test cases
to ensure that the SOTA (State of the Art) agents (at the time of publication)
do not achieve results higher than 50%, while the average white collar worker
can perform all these tasks with perfect accuracy. The benchmark can be scored
manually, but we also introduce an automated validation mechanism that has an
average error rate less than 2%. Therefore, this benchmark presents solid
ground for fully automated measuring of progress, capabilities and the
effectiveness of GUI-navigation AI agents over the short and medium-term
horizon. The source code of the benchmark is available at
https://github.com/agentsea/osuniverse.

### 9. Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability

[Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability](http://arxiv.org/pdf/2505.03641v1)

Authors: Chen Wei, Chi Zhang, Jiachen Zou, Haotian Deng, Dietmar Heinke, Quanying Liu

Human decision-making in cognitive tasks and daily life exhibits considerable
variability, shaped by factors such as task difficulty, individual preferences,
and personal experiences. Understanding this variability across individuals is
essential for uncovering the perceptual and decision-making mechanisms that
humans rely on when faced with uncertainty and ambiguity. We present a
computational framework BAM (Boundary Alignment & Manipulation framework) that
combines perceptual boundary sampling in ANNs and human behavioral experiments
to systematically investigate this phenomenon. Our perceptual boundary sampling
algorithm generates stimuli along ANN decision boundaries that intrinsically
induce significant perceptual variability. The efficacy of these stimuli is
empirically validated through large-scale behavioral experiments involving 246
participants across 116,715 trials, culminating in the variMNIST dataset
containing 19,943 systematically annotated images. Through personalized model
alignment and adversarial generation, we establish a reliable method for
simultaneously predicting and manipulating the divergent perceptual decisions
of pairs of participants. This work bridges the gap between computational
models and human individual difference research, providing new tools for
personalized perception analysis.

### 10. Learning Symbolic Persistent Macro-Actions for POMDP Solving Over Time

[Learning Symbolic Persistent Macro-Actions for POMDP Solving Over Time](http://arxiv.org/pdf/2505.03668v1)

Authors: Celeste Veronese, Daniele Meli, Alessandro Farinelli

This paper proposes an integration of temporal logical reasoning and
Partially Observable Markov Decision Processes (POMDPs) to achieve
interpretable decision-making under uncertainty with macro-actions. Our method
leverages a fragment of Linear Temporal Logic (LTL) based on Event Calculus
(EC) to generate \emph{persistent} (i.e., constant) macro-actions, which guide
Monte Carlo Tree Search (MCTS)-based POMDP solvers over a time horizon,
significantly reducing inference time while ensuring robust performance. Such
macro-actions are learnt via Inductive Logic Programming (ILP) from a few
traces of execution (belief-action pairs), thus eliminating the need for
manually designed heuristics and requiring only the specification of the POMDP
transition model. In the Pocman and Rocksample benchmark scenarios, our learned
macro-actions demonstrate increased expressiveness and generality when compared
to time-independent heuristics, indeed offering substantial computational
efficiency improvements.

### Hardware Architecture

### 1. Hardware vs. Software Implementation of Warp-Level Features in Vortex RISC-V GPU

[Hardware vs. Software Implementation of Warp-Level Features in Vortex RISC-V GPU](http://arxiv.org/pdf/2505.03102v1)

Authors: Huanzhi Pu, Rishabh Ravi, Shinnung Jeong, Udit Subramanya, Euijun Chung, Jisheng Zhao, Chihyo Ahn, Hyesoon Kim

RISC-V GPUs present a promising path for supporting GPU applications.
Traditionally, GPUs achieve high efficiency through the SPMD (Single Program
Multiple Data) programming model. However, modern GPU programming increasingly
relies on warp-level features, which diverge from the conventional SPMD
paradigm. In this paper, we explore how RISC-V GPUs can support these
warp-level features both through hardware implementation and via software-only
approaches. Our evaluation shows that a hardware implementation achieves up to
4 times geomean IPC speedup in microbenchmarks, while software-based solutions
provide a viable alternative for area-constrained scenarios.

### 2. QiMeng-CPU-v2: Automated Superscalar Processor Design by Learning Data Dependencies

[QiMeng-CPU-v2: Automated Superscalar Processor Design by Learning Data Dependencies](http://arxiv.org/pdf/2505.03195v1)

Authors: Shuyao Cheng, Rui Zhang, Wenkai He, Pengwei Jin, Chongxiao Li, Zidong Du, Xing Hu, Yifan Hao, Guanglin Xu, Yuanbo Wen, Ling Li, Qi Guo, Yunji Chen

Automated processor design, which can significantly reduce human efforts and
accelerate design cycles, has received considerable attention. While recent
advancements have automatically designed single-cycle processors that execute
one instruction per cycle, their performance cannot compete with modern
superscalar processors that execute multiple instructions per cycle. Previous
methods fail on superscalar processor design because they cannot address
inter-instruction data dependencies, leading to inefficient sequential
instruction execution.
  This paper proposes a novel approach to automatically designing superscalar
processors using a hardware-friendly model called the Stateful Binary
Speculation Diagram (State-BSD). We observe that processor parallelism can be
enhanced through on-the-fly inter-instruction dependent data predictors,
reusing the processor's internal states to learn the data dependency. To meet
the challenge of both hardware-resource limitation and design functional
correctness, State-BSD consists of two components: 1) a lightweight
state-selector trained by the simulated annealing method to detect the most
reusable processor states and store them in a small buffer; and 2) a highly
precise state-speculator trained by the BSD expansion method to predict the
inter-instruction dependent data using the selected states. It is the first
work to achieve the automated superscalar processor design, i.e. QiMeng-CPU-v2,
which improves the performance by about $380\times$ than the state-of-the-art
automated design and is comparable to human-designed superscalar processors
such as ARM Cortex A53.

### Computational Engineering

### 1. Transformers Applied to Short-term Solar PV Power Output Forecasting

[Transformers Applied to Short-term Solar PV Power Output Forecasting](http://arxiv.org/pdf/2505.03188v1)

Authors: Andea Scott, Sindhu Sreedhara, Folasade Ayoola

Reliable forecasts of the power output from variable renewable energy
generators like solar photovoltaic systems are important to balancing load on
real-time electricity markets and ensuring electricity supply reliability.
However, solar PV power output is highly uncertain, with significant variations
occurring over both longer (daily or seasonally) and shorter (within minutes)
timescales due to weather conditions, especially cloud cover. This paper builds
on existing work that uses convolutional neural networks in the computer vision
task of predicting (in a Nowcast model) and forecasting (in a Forecast model)
solar PV power output (Stanford EAO SUNSET Model). A pure transformer
architecture followed by a fully-connected layer is applied to one year of
image data with experiments run on various combinations of learning rate and
batch size. We find that the transformer architecture performs almost as well
as the baseline model in the PV output prediction task. However, it performs
worse on sunny days.

### 2. Data-efficient inverse design of spinodoid metamaterials

[Data-efficient inverse design of spinodoid metamaterials](http://arxiv.org/pdf/2505.03415v1)

Authors: Max Rosenkranz, Markus Kästner, Ivo F. Sbalzarini

We create an data-efficient and accurate surrogate model for
structure-property linkages of spinodoid metamaterials with only 75 data points
-- far fewer than the several thousands used in prior works -- and demonstrate
its use in multi-objective inverse design. The inverse problem of finding a
material microstructure that leads to given bulk properties is of great
interest in mechanics and materials science. These inverse design tasks often
require a large dataset, which can become unaffordable when considering
material behavior that requires more expensive simulations or experiments. We
generate a data-efficient surrogate for the mapping between the characteristics
of the local material structure and the effective elasticity tensor and use it
to inversely design structures with multiple objectives simultaneously. The
presented neural network-based surrogate model achieves its data efficiency by
inherently satisfying certain requirements, such as equivariance with respect
to permutations of structure parameters, which avoids having to learn them from
data. The resulting surrogate of the forward model is differentiable, allowing
its direct use in gradient-based optimization for the inverse design problem.
We demonstrate in three inverse design tasks of varying complexity that this
approach yields reliable results while requiring significantly less training
data than previous approaches based on neural-network surrogates. This paves
the way for inverse design involving nonlinear mechanical behavior, where data
efficiency is currently the limiting factor.

### 3. Algorithm Selection in Short-Range Molecular Dynamics Simulations

[Algorithm Selection in Short-Range Molecular Dynamics Simulations](http://arxiv.org/pdf/2505.03438v1)

Authors: Samuel James Newcome, Fabio Alexander Gratl, Manuel Lerchner, Abdulkadir Pazar, Manish Kumar Mishra, Hans-Joachim Bungartz

Numerous algorithms and parallelisations have been developed for short-range
particle simulations; however, none are optimally performant for all scenarios.
Such a concept led to the prior development of the particle simulation library
AutoPas, which implemented many of these algorithms and parallelisations and
could select and tune these over the course of the simulation as the scenario
changed. Prior works have, however, used only naive approaches to the algorithm
selection problem, which can lead to significant overhead from trialling poorly
performing algorithmic configurations.
  In this work, we investigate this problem in the case of Molecular Dynamics
simulations. We present three algorithm selection strategies: an approach which
makes performance predictions from past data, an expert-knowledge fuzzy
logic-based approach, and a data-driven random forest-based approach. We
demonstrate that these approaches can achieve speedups of up to 4.05 compared
to prior approaches and 1.25 compared to a perfect configuration selection
without dynamic algorithm selection. In addition, we discuss the practicality
of the strategies in comparison to their performance, to highlight the
tractability of such solutions.

### Computational Geometry

### 1. Blending 3D Geometry and Machine Learning for Multi-View Stereopsis

[Blending 3D Geometry and Machine Learning for Multi-View Stereopsis](http://arxiv.org/pdf/2505.03470v1)

Authors: Vibhas Vats, Md. Alimoor Reza, David Crandall, Soon-heung Jung

Traditional multi-view stereo (MVS) methods primarily depend on photometric
and geometric consistency constraints. In contrast, modern learning-based
algorithms often rely on the plane sweep algorithm to infer 3D geometry,
applying explicit geometric consistency (GC) checks only as a post-processing
step, with no impact on the learning process itself. In this work, we introduce
GC MVSNet plus plus, a novel approach that actively enforces geometric
consistency of reference view depth maps across multiple source views (multi
view) and at various scales (multi scale) during the learning phase (see Fig.
1). This integrated GC check significantly accelerates the learning process by
directly penalizing geometrically inconsistent pixels, effectively halving the
number of training iterations compared to other MVS methods. Furthermore, we
introduce a densely connected cost regularization network with two distinct
block designs simple and feature dense optimized to harness dense feature
connections for enhanced regularization. Extensive experiments demonstrate that
our approach achieves a new state of the art on the DTU and BlendedMVS datasets
and secures second place on the Tanks and Temples benchmark. To our knowledge,
GC MVSNet plus plus is the first method to enforce multi-view, multi-scale
supervised geometric consistency during learning. Our code is available.

### Computation and Language

### 1. Survey of Abstract Meaning Representation: Then, Now, Future

[Survey of Abstract Meaning Representation: Then, Now, Future](http://arxiv.org/pdf/2505.03229v1)

Authors: Behrooz Mansouri

This paper presents a survey of Abstract Meaning Representation (AMR), a
semantic representation framework that captures the meaning of sentences
through a graph-based structure. AMR represents sentences as rooted, directed
acyclic graphs, where nodes correspond to concepts and edges denote
relationships, effectively encoding the meaning of complex sentences. This
survey investigates AMR and its extensions, focusing on AMR capabilities. It
then explores the parsing (text-to-AMR) and generation (AMR-to-text) tasks by
showing traditional, current, and possible futures approaches. It also reviews
various applications of AMR including text generation, text classification, and
information extraction and information seeking. By analyzing recent
developments and challenges in the field, this survey provides insights into
future directions for research and the potential impact of AMR on enhancing
machine understanding of human language.

### 2. Ψ-Arena: Interactive Assessment and Optimization of LLM-based Psychological Counselors with Tripartite Feedback

[Ψ-Arena: Interactive Assessment and Optimization of LLM-based Psychological Counselors with Tripartite Feedback](http://arxiv.org/pdf/2505.03293v1)

Authors: Shijing Zhu, Zhuang Chen, Guanqun Bi, Binghang Li, Yaxi Deng, Dazhen Wan, Libiao Peng, Xiyao Xiao, Rongsheng Zhang, Tangjie Lv, Zhipeng Hu, FangFang Li, Minlie Huang

Large language models (LLMs) have shown promise in providing scalable mental
health support, while evaluating their counseling capability remains crucial to
ensure both efficacy and safety. Existing evaluations are limited by the static
assessment that focuses on knowledge tests, the single perspective that centers
on user experience, and the open-loop framework that lacks actionable feedback.
To address these issues, we propose {\Psi}-Arena, an interactive framework for
comprehensive assessment and optimization of LLM-based counselors, featuring
three key characteristics: (1) Realistic arena interactions that simulate
real-world counseling through multi-stage dialogues with psychologically
profiled NPC clients, (2) Tripartite evaluation that integrates assessments
from the client, counselor, and supervisor perspectives, and (3) Closed-loop
optimization that iteratively improves LLM counselors using diagnostic
feedback. Experiments across eight state-of-the-art LLMs show significant
performance variations in different real-world scenarios and evaluation
perspectives. Moreover, reflection-based optimization results in up to a 141%
improvement in counseling performance. We hope PsychoArena provides a
foundational resource for advancing reliable and human-aligned LLM applications
in mental healthcare.

### 3. Recall with Reasoning: Chain-of-Thought Distillation for Mamba's Long-Context Memory and Extrapolation

[Recall with Reasoning: Chain-of-Thought Distillation for Mamba's Long-Context Memory and Extrapolation](http://arxiv.org/pdf/2505.03320v1)

Authors: Junyu Ma, Tianqing Fang, Zhisong Zhang, Hongming Zhang, Haitao Mi, Dong Yu

Mamba's theoretical infinite-context potential is limited in practice when
sequences far exceed training lengths. This work explores unlocking Mamba's
long-context memory ability by a simple-yet-effective method, Recall with
Reasoning (RwR), by distilling chain-of-thought (CoT) summarization from a
teacher model. Specifically, RwR prepends these summarization as CoT prompts
during fine-tuning, teaching Mamba to actively recall and reason over long
contexts. Experiments on LONGMEMEVAL and HELMET show RwR boosts Mamba's
long-context performance against comparable Transformer/hybrid baselines under
similar pretraining conditions, while preserving short-context capabilities,
all without architectural changes.

### 4. Uncertainty-Aware Large Language Models for Explainable Disease Diagnosis

[Uncertainty-Aware Large Language Models for Explainable Disease Diagnosis](http://arxiv.org/pdf/2505.03467v1)

Authors: Shuang Zhou, Jiashuo Wang, Zidu Xu, Song Wang, David Brauer, Lindsay Welton, Jacob Cogan, Yuen-Hei Chung, Lei Tian, Zaifu Zhan, Yu Hou, Mingquan Lin, Genevieve B. Melton, Rui Zhang

Explainable disease diagnosis, which leverages patient information (e.g.,
signs and symptoms) and computational models to generate probable diagnoses and
reasonings, offers clear clinical values. However, when clinical notes
encompass insufficient evidence for a definite diagnosis, such as the absence
of definitive symptoms, diagnostic uncertainty usually arises, increasing the
risk of misdiagnosis and adverse outcomes. Although explicitly identifying and
explaining diagnostic uncertainties is essential for trustworthy diagnostic
systems, it remains under-explored. To fill this gap, we introduce ConfiDx, an
uncertainty-aware large language model (LLM) created by fine-tuning open-source
LLMs with diagnostic criteria. We formalized the task and assembled richly
annotated datasets that capture varying degrees of diagnostic ambiguity.
Evaluating ConfiDx on real-world datasets demonstrated that it excelled in
identifying diagnostic uncertainties, achieving superior diagnostic
performance, and generating trustworthy explanations for diagnoses and
uncertainties. To our knowledge, this is the first study to jointly address
diagnostic uncertainty recognition and explanation, substantially enhancing the
reliability of automatic diagnostic systems.

### 5. Long-Short Chain-of-Thought Mixture Supervised Fine-Tuning Eliciting Efficient Reasoning in Large Language Models

[Long-Short Chain-of-Thought Mixture Supervised Fine-Tuning Eliciting Efficient Reasoning in Large Language Models](http://arxiv.org/pdf/2505.03469v1)

Authors: Bin Yu, Hang Yuan, Yuliang Wei, Bailing Wang, Weizhen Qi, Kai Chen

Recent advances in large language models have demonstrated that Supervised
Fine-Tuning (SFT) with Chain-of-Thought (CoT) reasoning data distilled from
large reasoning models (e.g., DeepSeek R1) can effectively transfer reasoning
capabilities to non-reasoning models. However, models fine-tuned with this
approach inherit the "overthinking" problem from teacher models, producing
verbose and redundant reasoning chains during inference. To address this
challenge, we propose \textbf{L}ong-\textbf{S}hort Chain-of-Thought
\textbf{Mixture} \textbf{S}upervised \textbf{F}ine-\textbf{T}uning
(\textbf{LS-Mixture SFT}), which combines long CoT reasoning dataset with their
short counterparts obtained through structure-preserved rewriting. Our
experiments demonstrate that models trained using the LS-Mixture SFT method,
compared to those trained with direct SFT, achieved an average accuracy
improvement of 2.3\% across various benchmarks while substantially reducing
model response length by approximately 47.61\%. This work offers an approach to
endow non-reasoning models with reasoning capabilities through supervised
fine-tuning while avoiding the inherent overthinking problems inherited from
teacher models, thereby enabling efficient reasoning in the fine-tuned models.

### 6. Evaluation of LLMs on Long-tail Entity Linking in Historical Documents

[Evaluation of LLMs on Long-tail Entity Linking in Historical Documents](http://arxiv.org/pdf/2505.03473v1)

Authors: Marta Boscariol, Luana Bulla, Lia Draetta, Beatrice Fiumanò, Emanuele Lenzi, Leonardo Piano

Entity Linking (EL) plays a crucial role in Natural Language Processing (NLP)
applications, enabling the disambiguation of entity mentions by linking them to
their corresponding entries in a reference knowledge base (KB). Thanks to their
deep contextual understanding capabilities, LLMs offer a new perspective to
tackle EL, promising better results than traditional methods. Despite the
impressive generalization capabilities of LLMs, linking less popular, long-tail
entities remains challenging as these entities are often underrepresented in
training data and knowledge bases. Furthermore, the long-tail EL task is an
understudied problem, and limited studies address it with LLMs. In the present
work, we assess the performance of two popular LLMs, GPT and LLama3, in a
long-tail entity linking scenario. Using MHERCL v0.1, a manually annotated
benchmark of sentences from domain-specific historical texts, we quantitatively
compare the performance of LLMs in identifying and linking entities to their
corresponding Wikidata entries against that of ReLiK, a state-of-the-art Entity
Linking and Relation Extraction framework. Our preliminary experiments reveal
that LLMs perform encouragingly well in long-tail EL, indicating that this
technology can be a valuable adjunct in filling the gap between head and
long-tail EL.

### 7. Sentence Embeddings as an intermediate target in end-to-end summarisation

[Sentence Embeddings as an intermediate target in end-to-end summarisation](http://arxiv.org/pdf/2505.03481v1)

Authors: Maciej Zembrzuski, Saad Mahamood

Current neural network-based methods to the problem of document summarisation
struggle when applied to datasets containing large inputs. In this paper we
propose a new approach to the challenge of content-selection when dealing with
end-to-end summarisation of user reviews of accommodations. We show that by
combining an extractive approach with externally pre-trained sentence level
embeddings in an addition to an abstractive summarisation model we can
outperform existing methods when this is applied to the task of summarising a
large input dataset. We also prove that predicting sentence level embedding of
a summary increases the quality of an end-to-end system for loosely aligned
source to target corpora, than compared to commonly predicting probability
distributions of sentence selection.

### 8. Say It Another Way: A Framework for User-Grounded Paraphrasing

[Say It Another Way: A Framework for User-Grounded Paraphrasing](http://arxiv.org/pdf/2505.03563v1)

Authors: Cléa Chataigner, Rebecca Ma, Prakhar Ganesh, Afaf Taïk, Elliot Creager, Golnoosh Farnadi

Small changes in how a prompt is worded can lead to meaningful differences in
the behavior of large language models (LLMs), raising concerns about the
stability and reliability of their evaluations. While prior work has explored
simple formatting changes, these rarely capture the kinds of natural variation
seen in real-world language use. We propose a controlled paraphrasing framework
based on a taxonomy of minimal linguistic transformations to systematically
generate natural prompt variations. Using the BBQ dataset, we validate our
method with both human annotations and automated checks, then use it to study
how LLMs respond to paraphrased prompts in stereotype evaluation tasks. Our
analysis shows that even subtle prompt modifications can lead to substantial
changes in model behavior. These results highlight the need for robust,
paraphrase-aware evaluation protocols.

### 9. Towards conversational assistants for health applications: using ChatGPT to generate conversations about heart failure

[Towards conversational assistants for health applications: using ChatGPT to generate conversations about heart failure](http://arxiv.org/pdf/2505.03675v1)

Authors: Anuja Tayal, Devika Salunke, Barbara Di Eugenio, Paula G Allen-Meares, Eulalia P Abril, Olga Garcia-Bedoya, Carolyn A Dickens, Andrew D. Boyd

We explore the potential of ChatGPT (3.5-turbo and 4) to generate
conversations focused on self-care strategies for African-American heart
failure patients -- a domain with limited specialized datasets. To simulate
patient-health educator dialogues, we employed four prompting strategies:
domain, African American Vernacular English (AAVE), Social Determinants of
Health (SDOH), and SDOH-informed reasoning. Conversations were generated across
key self-care domains of food, exercise, and fluid intake, with varying turn
lengths (5, 10, 15) and incorporated patient-specific SDOH attributes such as
age, gender, neighborhood, and socioeconomic status. Our findings show that
effective prompt design is essential. While incorporating SDOH and reasoning
improves dialogue quality, ChatGPT still lacks the empathy and engagement
needed for meaningful healthcare communication.

### 10. NBF at SemEval-2025 Task 5: Light-Burst Attention Enhanced System for Multilingual Subject Recommendation

[NBF at SemEval-2025 Task 5: Light-Burst Attention Enhanced System for Multilingual Subject Recommendation](http://arxiv.org/pdf/2505.03711v1)

Authors: Baharul Islam, Nasim Ahmad, Ferdous Ahmed Barbhuiya, Kuntal Dey

We present our system submission for SemEval 2025 Task 5, which focuses on
cross-lingual subject classification in the English and German academic
domains. Our approach leverages bilingual data during training, employing
negative sampling and a margin-based retrieval objective. We demonstrate that a
dimension-as-token self-attention mechanism designed with significantly reduced
internal dimensions can effectively encode sentence embeddings for subject
retrieval. In quantitative evaluation, our system achieved an average recall
rate of 32.24% in the general quantitative setting (all subjects), 43.16% and
31.53% of the general qualitative evaluation methods with minimal GPU usage,
highlighting their competitive performance. Our results demonstrate that our
approach is effective in capturing relevant subject information under resource
constraints, although there is still room for improvement.

### Cryptography and Security

### 1. Towards a standardized methodology and dataset for evaluating LLM-based digital forensic timeline analysis

[Towards a standardized methodology and dataset for evaluating LLM-based digital forensic timeline analysis](http://arxiv.org/pdf/2505.03100v1)

Authors: Hudan Studiawan, Frank Breitinger, Mark Scanlon

Large language models (LLMs) have seen widespread adoption in many domains
including digital forensics. While prior research has largely centered on case
studies and examples demonstrating how LLMs can assist forensic investigations,
deeper explorations remain limited, i.e., a standardized approach for precise
performance evaluations is lacking. Inspired by the NIST Computer Forensic Tool
Testing Program, this paper proposes a standardized methodology to
quantitatively evaluate the application of LLMs for digital forensic tasks,
specifically in timeline analysis. The paper describes the components of the
methodology, including the dataset, timeline generation, and ground truth
development. Additionally, the paper recommends using BLEU and ROUGE metrics
for the quantitative evaluation of LLMs through case studies or tasks involving
timeline analysis. Experimental results using ChatGPT demonstrate that the
proposed methodology can effectively evaluate LLM-based forensic timeline
analysis. Finally, we discuss the limitations of applying LLMs to forensic
timeline analysis.

### 2. Towards Effective Identification of Attack Techniques in Cyber Threat Intelligence Reports using Large Language Models

[Towards Effective Identification of Attack Techniques in Cyber Threat Intelligence Reports using Large Language Models](http://arxiv.org/pdf/2505.03147v1)

Authors: Hoang Cuong Nguyen, Shahroz Tariq, Mohan Baruwal Chhetri, Bao Quoc Vo

This work evaluates the performance of Cyber Threat Intelligence (CTI)
extraction methods in identifying attack techniques from threat reports
available on the web using the MITRE ATT&CK framework. We analyse four
configurations utilising state-of-the-art tools, including the Threat Report
ATT&CK Mapper (TRAM) and open-source Large Language Models (LLMs) such as
Llama2. Our findings reveal significant challenges, including class imbalance,
overfitting, and domain-specific complexity, which impede accurate technique
extraction. To mitigate these issues, we propose a novel two-step pipeline:
first, an LLM summarises the reports, and second, a retrained SciBERT model
processes a rebalanced dataset augmented with LLM-generated data. This approach
achieves an improvement in F1-scores compared to baseline models, with several
attack techniques surpassing an F1-score of 0.90. Our contributions enhance the
efficiency of web-based CTI systems and support collaborative cybersecurity
operations in an interconnected digital landscape, paving the way for future
research on integrating human-AI collaboration platforms.

### 3. An LLM-based Self-Evolving Security Framework for 6G Space-Air-Ground Integrated Networks

[An LLM-based Self-Evolving Security Framework for 6G Space-Air-Ground Integrated Networks](http://arxiv.org/pdf/2505.03161v1)

Authors: Qi Qin, Xinye Cao, Guoshun Nan, Sihan Chen, Rushan Li, Li Su, Haitao Du, Qimei Cui, Pengxuan Mao, Xiaofeng Tao, Tony Q. S. Quek

Recently emerged 6G space-air-ground integrated networks (SAGINs), which
integrate satellites, aerial networks, and terrestrial communications, offer
ubiquitous coverage for various mobile applications. However, the highly
dynamic, open, and heterogeneous nature of SAGINs poses severe security issues.
Forming a defense line of SAGINs suffers from two preliminary challenges: 1)
accurately understanding massive unstructured multi-dimensional threat
information to generate defense strategies against various malicious attacks,
2) rapidly adapting to potential unknown threats to yield more effective
security strategies. To tackle the above two challenges, we propose a novel
security framework for SAGINs based on Large Language Models (LLMs), which
consists of two key ingredients LLM-6GNG and 6G-INST. Our proposed LLM-6GNG
leverages refined chain-of-thought (CoT) reasoning and dynamic multi-agent
mechanisms to analyze massive unstructured multi-dimensional threat data and
generate comprehensive security strategies, thus addressing the first
challenge. Our proposed 6G-INST relies on a novel self-evolving method to
automatically update LLM-6GNG, enabling it to accommodate unknown threats under
dynamic communication environments, thereby addressing the second challenge.
Additionally, we prototype the proposed framework with ns-3, OpenAirInterface
(OAI), and software-defined radio (SDR). Experiments on three benchmarks
demonstrate the effectiveness of our framework. The results show that our
framework produces highly accurate security strategies that remain robust
against a variety of unknown attacks. We will release our code to contribute to
the community.

### 4. Bridging Expertise Gaps: The Role of LLMs in Human-AI Collaboration for Cybersecurity

[Bridging Expertise Gaps: The Role of LLMs in Human-AI Collaboration for Cybersecurity](http://arxiv.org/pdf/2505.03179v1)

Authors: Shahroz Tariq, Ronal Singh, Mohan Baruwal Chhetri, Surya Nepal, Cecile Paris

This study investigates whether large language models (LLMs) can function as
intelligent collaborators to bridge expertise gaps in cybersecurity
decision-making. We examine two representative tasks-phishing email detection
and intrusion detection-that differ in data modality, cognitive complexity, and
user familiarity. Through a controlled mixed-methods user study, n = 58
(phishing, n = 34; intrusion, n = 24), we find that human-AI collaboration
improves task performance,reducing false positives in phishing detection and
false negatives in intrusion detection. A learning effect is also observed when
participants transition from collaboration to independent work, suggesting that
LLMs can support long-term skill development. Our qualitative analysis shows
that interaction dynamics-such as LLM definitiveness, explanation style, and
tone-influence user trust, prompting strategies, and decision revision. Users
engaged in more analytic questioning and showed greater reliance on LLM
feedback in high-complexity settings. These results provide design guidance for
building interpretable, adaptive, and trustworthy human-AI teaming systems, and
demonstrate that LLMs can meaningfully support non-experts in reasoning through
complex cybersecurity problems.

### 5. A Chaos Driven Metric for Backdoor Attack Detection

[A Chaos Driven Metric for Backdoor Attack Detection](http://arxiv.org/pdf/2505.03208v1)

Authors: Hema Karnam Surendrababu, Nithin Nagaraj

The advancement and adoption of Artificial Intelligence (AI) models across
diverse domains have transformed the way we interact with technology. However,
it is essential to recognize that while AI models have introduced remarkable
advancements, they also present inherent challenges such as their vulnerability
to adversarial attacks. The current work proposes a novel defense mechanism
against one of the most significant attack vectors of AI models - the backdoor
attack via data poisoning of training datasets. In this defense technique, an
integrated approach that combines chaos theory with manifold learning is
proposed. A novel metric - Precision Matrix Dependency Score (PDS) that is
based on the conditional variance of Neurochaos features is formulated. The PDS
metric has been successfully evaluated to distinguish poisoned samples from
non-poisoned samples across diverse datasets.

### 6. Elevating Cyber Threat Intelligence against Disinformation Campaigns with LLM-based Concept Extraction and the FakeCTI Dataset

[Elevating Cyber Threat Intelligence against Disinformation Campaigns with LLM-based Concept Extraction and the FakeCTI Dataset](http://arxiv.org/pdf/2505.03345v1)

Authors: Domenico Cotroneo, Roberto Natella, Vittorio Orbinato

The swift spread of fake news and disinformation campaigns poses a
significant threat to public trust, political stability, and cybersecurity.
Traditional Cyber Threat Intelligence (CTI) approaches, which rely on low-level
indicators such as domain names and social media handles, are easily evaded by
adversaries who frequently modify their online infrastructure. To address these
limitations, we introduce a novel CTI framework that focuses on high-level,
semantic indicators derived from recurrent narratives and relationships of
disinformation campaigns. Our approach extracts structured CTI indicators from
unstructured disinformation content, capturing key entities and their
contextual dependencies within fake news using Large Language Models (LLMs). We
further introduce FakeCTI, the first dataset that systematically links fake
news to disinformation campaigns and threat actors. To evaluate the
effectiveness of our CTI framework, we analyze multiple fake news attribution
techniques, spanning from traditional Natural Language Processing (NLP) to
fine-tuned LLMs. This work shifts the focus from low-level artifacts to
persistent conceptual structures, establishing a scalable and adaptive approach
to tracking and countering disinformation campaigns.

### 7. Directed Greybox Fuzzing via Large Language Model

[Directed Greybox Fuzzing via Large Language Model](http://arxiv.org/pdf/2505.03425v1)

Authors: Hanxiang Xu, Yanjie Zhao, Haoyu Wang

Directed greybox fuzzing (DGF) focuses on efficiently reaching specific
program locations or triggering particular behaviors, making it essential for
tasks like vulnerability detection and crash reproduction. However, existing
methods often suffer from path explosion and randomness in input mutation,
leading to inefficiencies in exploring and exploiting target paths. In this
paper, we propose HGFuzzer, an automatic framework that leverages the large
language model (LLM) to address these challenges. HGFuzzer transforms path
constraint problems into targeted code generation tasks, systematically
generating test harnesses and reachable inputs to reduce unnecessary
exploration paths significantly. Additionally, we implement custom mutators
designed specifically for target functions, minimizing randomness and improving
the precision of directed fuzzing. We evaluated HGFuzzer on 20 real-world
vulnerabilities, successfully triggering 17, including 11 within the first
minute, achieving a speedup of at least 24.8x compared to state-of-the-art
directed fuzzers. Furthermore, HGFuzzer discovered 9 previously unknown
vulnerabilities, all of which were assigned CVE IDs, demonstrating the
effectiveness of our approach in identifying real-world vulnerabilities.

### 8. Mitigating Backdoor Triggered and Targeted Data Poisoning Attacks in Voice Authentication Systems

[Mitigating Backdoor Triggered and Targeted Data Poisoning Attacks in Voice Authentication Systems](http://arxiv.org/pdf/2505.03455v1)

Authors: Alireza Mohammadi, Keshav Sood, Dhananjay Thiruvady, Asef Nazari

Voice authentication systems remain susceptible to two major threats:
backdoor triggered attacks and targeted data poisoning attacks. This dual
vulnerability is critical because conventional solutions typically address each
threat type separately, leaving systems exposed to adversaries who can exploit
both attacks simultaneously. We propose a unified defense framework that
effectively addresses both BTA and TDPA. Our framework integrates a frequency
focused detection mechanism that flags covert pitch boosting and sound masking
backdoor attacks in near real time, followed by a convolutional neural network
that addresses TDPA. This dual layered defense approach utilizes
multidimensional acoustic features to isolate anomalous signals without
requiring costly model retraining. In particular, our PBSM detection mechanism
can seamlessly integrate into existing voice authentication pipelines and scale
effectively for large scale deployments. Experimental results on benchmark
datasets and their compression with the state of the art algorithm demonstrate
that our PBSM detection mechanism outperforms the state of the art. Our
framework reduces attack success rates to as low as five to fifteen percent
while maintaining a recall rate of up to ninety five percent in recognizing
TDPA.

### 9. Empc: Effective Path Prioritization for Symbolic Execution with Path Cover

[Empc: Effective Path Prioritization for Symbolic Execution with Path Cover](http://arxiv.org/pdf/2505.03555v1)

Authors: Shuangjie Yao, Dongdong She

Symbolic execution is a powerful program analysis technique that can formally
reason the correctness of program behaviors and detect software bugs. It can
systematically explore the execution paths of the tested program. But it
suffers from an inherent limitation: path explosion. Path explosion occurs when
symbolic execution encounters an overwhelming number (exponential to the
program size) of paths that need to be symbolically reasoned. It severely
impacts the scalability and performance of symbolic execution. To tackle this
problem, previous works leverage various heuristics to prioritize paths for
symbolic execution. They rank the exponential number of paths using static
rules or heuristics and explore the paths with the highest rank. However, in
practice, these works often fail to generalize to diverse programs. In this
work, we propose a novel and effective path prioritization technique with path
cover, named Empc. Our key insight is that not all paths need to be
symbolically reasoned. Unlike traditional path prioritization, our approach
leverages a small subset of paths as a minimum path cover (MPC) that can cover
all code regions of the tested programs. To encourage diversity in path
prioritization, we compute multiple MPCs. We then guide the search for symbolic
execution on the small number of paths inside multiple MPCs rather than the
exponential number of paths. We implement our technique Empc based on KLEE. We
conduct a comprehensive evaluation of Empc to investigate its performance in
code coverage, bug findings, and runtime overhead. The evaluation shows that
Empc can cover 19.6% more basic blocks than KLEE's best search strategy and
24.4% more lines compared to the state-of-the-art work cgs. Empc also finds 24
more security violations than KLEE's best search strategy. Meanwhile, Empc can
significantly reduce the memory usage of KLEE by up to 93.5%.

### 10. Differential Privacy for Network Assortativity

[Differential Privacy for Network Assortativity](http://arxiv.org/pdf/2505.03639v1)

Authors: Fei Ma, Jinzhi Ouyang, Xincheng Hu

The analysis of network assortativity is of great importance for
understanding the structural characteristics of and dynamics upon networks.
Often, network assortativity is quantified using the assortativity coefficient
that is defined based on the Pearson correlation coefficient between vertex
degrees. It is well known that a network may contain sensitive information,
such as the number of friends of an individual in a social network (which is
abstracted as the degree of vertex.). So, the computation of the assortativity
coefficient leads to privacy leakage, which increases the urgent need for
privacy-preserving protocol. However, there has been no scheme addressing the
concern above.
  To bridge this gap, in this work, we are the first to propose approaches
based on differential privacy (DP for short). Specifically, we design three
DP-based algorithms: $Local_{ru}$, $Shuffle_{ru}$, and $Decentral_{ru}$. The
first two algorithms, based on Local DP (LDP) and Shuffle DP respectively, are
designed for settings where each individual only knows his/her direct friends.
In contrast, the third algorithm, based on Decentralized DP (DDP), targets
scenarios where each individual has a broader view, i.e., also knowing his/her
friends' friends. Theoretically, we prove that each algorithm enables an
unbiased estimation of the assortativity coefficient of the network. We further
evaluate the performance of the proposed algorithms using mean squared error
(MSE), showing that $Shuffle_{ru}$ achieves the best performance, followed by
$Decentral_{ru}$, with $Local_{ru}$ performing the worst. Note that these three
algorithms have different assumptions, so each has its applicability scenario.
Lastly, we conduct extensive numerical simulations, which demonstrate that the
presented approaches are adequate to achieve the estimation of network
assortativity under the demand for privacy protection.

### Computer Vision and Pattern Recognition

### 1. Estimating the Diameter at Breast Height of Trees in a Forest With a Single 360 Camera

[Estimating the Diameter at Breast Height of Trees in a Forest With a Single 360 Camera](http://arxiv.org/pdf/2505.03093v1)

Authors: Siming He, Zachary Osman, Fernando Cladera, Dexter Ong, Nitant Rai, Patrick Corey Green, Vijay Kumar, Pratik Chaudhari

Forest inventories rely on accurate measurements of the diameter at breast
height (DBH) for ecological monitoring, resource management, and carbon
accounting. While LiDAR-based techniques can achieve centimeter-level
precision, they are cost-prohibitive and operationally complex. We present a
low-cost alternative that only needs a consumer-grade 360 video camera. Our
semi-automated pipeline comprises of (i) a dense point cloud reconstruction
using Structure from Motion (SfM) photogrammetry software called Agisoft
Metashape, (ii) semantic trunk segmentation by projecting Grounded Segment
Anything (SAM) masks onto the 3D cloud, and (iii) a robust RANSAC-based
technique to estimate cross section shape and DBH. We introduce an interactive
visualization tool for inspecting segmented trees and their estimated DBH. On
61 acquisitions of 43 trees under a variety of conditions, our method attains
median absolute relative errors of 5-9% with respect to "ground-truth" manual
measurements. This is only 2-4% higher than LiDAR-based estimates, while
employing a single 360 camera that costs orders of magnitude less, requires
minimal setup, and is widely available.

### 2. Not All Parameters Matter: Masking Diffusion Models for Enhancing Generation Ability

[Not All Parameters Matter: Masking Diffusion Models for Enhancing Generation Ability](http://arxiv.org/pdf/2505.03097v1)

Authors: Lei Wang, Senmao Li, Fei Yang, Jianye Wang, Ziheng Zhang, Yuhan Liu, Yaxing Wang, Jian Yang

The diffusion models, in early stages focus on constructing basic image
structures, while the refined details, including local features and textures,
are generated in later stages. Thus the same network layers are forced to learn
both structural and textural information simultaneously, significantly
differing from the traditional deep learning architectures (e.g., ResNet or
GANs) which captures or generates the image semantic information at different
layers. This difference inspires us to explore the time-wise diffusion models.
We initially investigate the key contributions of the U-Net parameters to the
denoising process and identify that properly zeroing out certain parameters
(including large parameters) contributes to denoising, substantially improving
the generation quality on the fly. Capitalizing on this discovery, we propose a
simple yet effective method-termed ``MaskUNet''- that enhances generation
quality with negligible parameter numbers. Our method fully leverages timestep-
and sample-dependent effective U-Net parameters. To optimize MaskUNet, we offer
two fine-tuning strategies: a training-based approach and a training-free
approach, including tailored networks and optimization functions. In zero-shot
inference on the COCO dataset, MaskUNet achieves the best FID score and further
demonstrates its effectiveness in downstream task evaluations. Project page:
https://gudaochangsheng.github.io/MaskUnet-Page/

### 3. Image Recognition with Online Lightweight Vision Transformer: A Survey

[Image Recognition with Online Lightweight Vision Transformer: A Survey](http://arxiv.org/pdf/2505.03113v1)

Authors: Zherui Zhang, Rongtao Xu, Jie Zhou, Changwei Wang, Xingtian Pei, Wenhao Xu, Jiguang Zhang, Li Guo, Longxiang Gao, Wenbo Xu, Shibiao Xu

The Transformer architecture has achieved significant success in natural
language processing, motivating its adaptation to computer vision tasks. Unlike
convolutional neural networks, vision transformers inherently capture
long-range dependencies and enable parallel processing, yet lack inductive
biases and efficiency benefits, facing significant computational and memory
challenges that limit its real-world applicability. This paper surveys various
online strategies for generating lightweight vision transformers for image
recognition, focusing on three key areas: Efficient Component Design, Dynamic
Network, and Knowledge Distillation. We evaluate the relevant exploration for
each topic on the ImageNet-1K benchmark, analyzing trade-offs among precision,
parameters, throughput, and more to highlight their respective advantages,
disadvantages, and flexibility. Finally, we propose future research directions
and potential challenges in the lightweighting of vision transformers with the
aim of inspiring further exploration and providing practical guidance for the
community. Project Page: https://github.com/ajxklo/Lightweight-VIT

### 4. Path and Bone-Contour Regularized Unpaired MRI-to-CT Translation

[Path and Bone-Contour Regularized Unpaired MRI-to-CT Translation](http://arxiv.org/pdf/2505.03114v1)

Authors: Teng Zhou, Jax Luo, Yuping Sun, Yiheng Tan, Shun Yao, Nazim Haouchine, Scott Raymond

Accurate MRI-to-CT translation promises the integration of complementary
imaging information without the need for additional imaging sessions. Given the
practical challenges associated with acquiring paired MRI and CT scans, the
development of robust methods capable of leveraging unpaired datasets is
essential for advancing the MRI-to-CT translation. Current unpaired MRI-to-CT
translation methods, which predominantly rely on cycle consistency and
contrastive learning frameworks, frequently encounter challenges in accurately
translating anatomical features that are highly discernible on CT but less
distinguishable on MRI, such as bone structures. This limitation renders these
approaches less suitable for applications in radiation therapy, where precise
bone representation is essential for accurate treatment planning. To address
this challenge, we propose a path- and bone-contour regularized approach for
unpaired MRI-to-CT translation. In our method, MRI and CT images are projected
to a shared latent space, where the MRI-to-CT mapping is modeled as a
continuous flow governed by neural ordinary differential equations. The optimal
mapping is obtained by minimizing the transition path length of the flow. To
enhance the accuracy of translated bone structures, we introduce a trainable
neural network to generate bone contours from MRI and implement mechanisms to
directly and indirectly encourage the model to focus on bone contours and their
adjacent regions. Evaluations conducted on three datasets demonstrate that our
method outperforms existing unpaired MRI-to-CT translation approaches,
achieving lower overall error rates. Moreover, in a downstream bone
segmentation task, our approach exhibits superior performance in preserving the
fidelity of bone structures. Our code is available at:
https://github.com/kennysyp/PaBoT.

### 5. TimeTracker: Event-based Continuous Point Tracking for Video Frame Interpolation with Non-linear Motion

[TimeTracker: Event-based Continuous Point Tracking for Video Frame Interpolation with Non-linear Motion](http://arxiv.org/pdf/2505.03116v1)

Authors: Haoyue Liu, Jinghan Xu, Yi Chang, Hanyu Zhou, Haozhi Zhao, Lin Wang, Luxin Yan

Video frame interpolation (VFI) that leverages the bio-inspired event cameras
as guidance has recently shown better performance and memory efficiency than
the frame-based methods, thanks to the event cameras' advantages, such as high
temporal resolution. A hurdle for event-based VFI is how to effectively deal
with non-linear motion, caused by the dynamic changes in motion direction and
speed within the scene. Existing methods either use events to estimate sparse
optical flow or fuse events with image features to estimate dense optical flow.
Unfortunately, motion errors often degrade the VFI quality as the continuous
motion cues from events do not align with the dense spatial information of
images in the temporal dimension. In this paper, we find that object motion is
continuous in space, tracking local regions over continuous time enables more
accurate identification of spatiotemporal feature correlations. In light of
this, we propose a novel continuous point tracking-based VFI framework, named
TimeTracker. Specifically, we first design a Scene-Aware Region Segmentation
(SARS) module to divide the scene into similar patches. Then, a Continuous
Trajectory guided Motion Estimation (CTME) module is proposed to track the
continuous motion trajectory of each patch through events. Finally,
intermediate frames at any given time are generated through global motion
optimization and frame refinement. Moreover, we collect a real-world dataset
that features fast non-linear motion. Extensive experiments show that our
method outperforms prior arts in both motion estimation and frame interpolation
quality.

### 6. Robust Fairness Vision-Language Learning for Medical Image Analysis

[Robust Fairness Vision-Language Learning for Medical Image Analysis](http://arxiv.org/pdf/2505.03153v1)

Authors: Sparsh Bansal, Mingyang Wu, Xin Wang, Shu Hu

The advent of Vision-Language Models (VLMs) in medical image analysis has the
potential to help process multimodal inputs and increase performance over
traditional inference methods. However, when considering the domain in which
these models will be implemented, fairness and robustness are important to
ensure the model stays true for any patient. In this paper, we introduce a
framework for ensuring robustness and fairness of VLM models. This framework
modifies the loss function at training by identifying and adjusting faulty
image-text pairs through a Dynamic Bad Pair Mining algorithm and also utilizing
Sinkhorn distance to ensure the loss distributions of protected groups do not
deviate from the total loss. Experimental testing of our framework shows up to
a 8.6\% improvement when looking at equity-scaled AUC.

### 7. Interactive Instance Annotation with Siamese Networks

[Interactive Instance Annotation with Siamese Networks](http://arxiv.org/pdf/2505.03184v1)

Authors: Xiang Xu, Ruotong Li, Mengjun Yi, Baile XU, Furao Shen, Jian Zhao

Annotating instance masks is time-consuming and labor-intensive. A promising
solution is to predict contours using a deep learning model and then allow
users to refine them. However, most existing methods focus on in-domain
scenarios, limiting their effectiveness for cross-domain annotation tasks. In
this paper, we propose SiamAnno, a framework inspired by the use of Siamese
networks in object tracking. SiamAnno leverages one-shot learning to annotate
previously unseen objects by taking a bounding box as input and predicting
object boundaries, which can then be adjusted by annotators. Trained on one
dataset and tested on another without fine-tuning, SiamAnno achieves
state-of-the-art (SOTA) performance across multiple datasets, demonstrating its
ability to handle domain and environment shifts in cross-domain tasks. We also
provide more comprehensive results compared to previous work, establishing a
strong baseline for future research. To our knowledge, SiamAnno is the first
model to explore Siamese architecture for instance annotation.

### 8. PiCo: Enhancing Text-Image Alignment with Improved Noise Selection and Precise Mask Control in Diffusion Models

[PiCo: Enhancing Text-Image Alignment with Improved Noise Selection and Precise Mask Control in Diffusion Models](http://arxiv.org/pdf/2505.03203v1)

Authors: Chang Xie, Chenyi Zhuang, Pan Gao

Advanced diffusion models have made notable progress in text-to-image
compositional generation. However, it is still a challenge for existing models
to achieve text-image alignment when confronted with complex text prompts. In
this work, we highlight two factors that affect this alignment: the quality of
the randomly initialized noise and the reliability of the generated controlling
mask. We then propose PiCo (Pick-and-Control), a novel training-free approach
with two key components to tackle these two factors. First, we develop a noise
selection module to assess the quality of the random noise and determine
whether the noise is suitable for the target text. A fast sampling strategy is
utilized to ensure efficiency in the noise selection stage. Second, we
introduce a referring mask module to generate pixel-level masks and to
precisely modulate the cross-attention maps. The referring mask is applied to
the standard diffusion process to guide the reasonable interaction between text
and image features. Extensive experiments have been conducted to verify the
effectiveness of PiCo in liberating users from the tedious process of random
generation and in enhancing the text-image alignment for diverse text
descriptions.

### 9. Dual-Domain Masked Image Modeling: A Self-Supervised Pretraining Strategy Using Spatial and Frequency Domain Masking for Hyperspectral Data

[Dual-Domain Masked Image Modeling: A Self-Supervised Pretraining Strategy Using Spatial and Frequency Domain Masking for Hyperspectral Data](http://arxiv.org/pdf/2505.03220v1)

Authors: Shaheer Mohamed, Tharindu Fernando, Sridha Sridharan, Peyman Moghadam, Clinton Fookes

Hyperspectral images (HSIs) capture rich spectral signatures that reveal
vital material properties, offering broad applicability across various domains.
However, the scarcity of labeled HSI data limits the full potential of deep
learning, especially for transformer-based architectures that require
large-scale training. To address this constraint, we propose Spatial-Frequency
Masked Image Modeling (SFMIM), a self-supervised pretraining strategy for
hyperspectral data that utilizes the large portion of unlabeled data. Our
method introduces a novel dual-domain masking mechanism that operates in both
spatial and frequency domains. The input HSI cube is initially divided into
non-overlapping patches along the spatial dimension, with each patch comprising
the entire spectrum of its corresponding spatial location. In spatial masking,
we randomly mask selected patches and train the model to reconstruct the masked
inputs using the visible patches. Concurrently, in frequency masking, we remove
portions of the frequency components of the input spectra and predict the
missing frequencies. By learning to reconstruct these masked components, the
transformer-based encoder captures higher-order spectral-spatial correlations.
We evaluate our approach on three publicly available HSI classification
benchmarks and demonstrate that it achieves state-of-the-art performance.
Notably, our model shows rapid convergence during fine-tuning, highlighting the
efficiency of our pretraining strategy.

### 10. Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification

[Base-Detail Feature Learning Framework for Visible-Infrared Person Re-Identification](http://arxiv.org/pdf/2505.03286v1)

Authors: Zhihao Gong, Lian Wu, Yong Xu

Visible-infrared person re-identification (VIReID) provides a solution for
ReID tasks in 24-hour scenarios; however, significant challenges persist in
achieving satisfactory performance due to the substantial discrepancies between
visible (VIS) and infrared (IR) modalities. Existing methods inadequately
leverage information from different modalities, primarily focusing on digging
distinguishing features from modality-shared information while neglecting
modality-specific details. To fully utilize differentiated minutiae, we propose
a Base-Detail Feature Learning Framework (BDLF) that enhances the learning of
both base and detail knowledge, thereby capitalizing on both modality-shared
and modality-specific information. Specifically, the proposed BDLF mines detail
and base features through a lossless detail feature extraction module and a
complementary base embedding generation mechanism, respectively, supported by a
novel correlation restriction method that ensures the features gained by BDLF
enrich both detail and base knowledge across VIS and IR features. Comprehensive
experiments conducted on the SYSU-MM01, RegDB, and LLCM datasets validate the
effectiveness of BDLF.

### Computers and Society

### 1. Ruled by the Representation Space: On the University's Embrace of Large Language Models

[Ruled by the Representation Space: On the University's Embrace of Large Language Models](http://arxiv.org/pdf/2505.03513v1)

Authors: Katia Schwerzmann

This paper explores the implications of universities' rapid adoption of large
language models (LLMs) for studying, teaching, and research by analyzing the
logics underpinning their representation space. It argues that by uncritically
adopting LLMs, the University surrenders its autonomy to a field of heteronomy,
that of generative AI, whose norms are not democratically shaped. Unlike
earlier forms of rule-based AI, which sought to exclude human judgment and
interpretation, generative AI's new normative rationality is explicitly based
on the automation of moral judgment, valuation, and interpretation. By
integrating LLMs into pedagogical and research contexts before establishing a
critical framework for their use, the University subjects itself to being
governed by contingent, ever-evolving, and domain-non-specific norms that
structure the model's virtual representation space and thus everything it
generates.

### 2. A Unifying Bias-aware Multidisciplinary Framework for Investigating Socio-Technical Issues

[A Unifying Bias-aware Multidisciplinary Framework for Investigating Socio-Technical Issues](http://arxiv.org/pdf/2505.03593v1)

Authors: Sacha Hasan, Mehdi Rizvi, Yingfang Yuan, Kefan Chen, Lynne Baillie, Wei Pang

This paper aims to bring together the disciplines of social science (SS) and
computer science (CS) in the design and implementation of a novel
multidisciplinary framework for systematic, transparent, ethically-informed,
and bias-aware investigation of socio-technical issues. For this, various
analysis approaches from social science and machine learning (ML) were applied
in a structured sequence to arrive at an original methodology of identifying
and quantifying objects of inquiry. A core feature of this framework is that it
highlights where bias occurs and suggests possible steps to mitigate it. This
is to improve the robustness, reliability, and explainability of the framework
and its results. Such an approach also ensures that the investigation of
socio-technical issues is transparent about its own limitations and potential
sources of bias. To test our framework, we utilised it in the multidisciplinary
investigation of the online harms encountered by minoritised ethnic (ME)
communities when accessing and using digitalised social housing services in the
UK. We draw our findings from 100 interviews with ME individuals in four cities
across the UK to understand ME vulnerabilities when accessing and using
digitalised social housing services. In our framework, a sub-sample of
interviews focusing on ME individuals residing in social housing units were
inductively coded. This resulted in the identification of the topics of
discrimination, digital poverty, lack of digital literacy, and lack of English
proficiency as key vulnerabilities of ME communities. Further ML techniques
such as Topic Modelling and Sentiment Analysis were used within our framework
where we found that Black African communities are more likely to experience
these vulnerabilities in the access, use and outcome of digitalised social
housing services.

### 3. Doing Audits Right? The Role of Sampling and Legal Content Analysis in Systemic Risk Assessments and Independent Audits in the Digital Services Act

[Doing Audits Right? The Role of Sampling and Legal Content Analysis in Systemic Risk Assessments and Independent Audits in the Digital Services Act](http://arxiv.org/pdf/2505.03601v1)

Authors: Marie-Therese Sekwenz, Rita Gsenger, Scott Dahlgren, Ben Wagner

A central requirement of the European Union's Digital Services Act (DSA) is
that online platforms undergo internal and external audits. A key component of
these audits is the assessment of systemic risks, including the dissemination
of illegal content, threats to fundamental rights, impacts on democratic
processes, and gender-based violence. The DSA Delegated Regulation outlines how
such audits should be conducted, setting expectations for both platforms and
auditors. This article evaluates the strengths and limitations of different
qualitative and quantitative methods for auditing these systemic risks and
proposes a mixed-method approach for DSA compliance. We argue that content
sampling, combined with legal and empirical analysis, offers a viable method
for risk-specific audits. First, we examine relevant legal provisions on sample
selection for audit purposes. We then assess sampling techniques and methods
suitable for detecting systemic risks, focusing on how representativeness can
be understood across disciplines. Finally, we review initial systemic risk
assessment reports submitted by platforms, analyzing their testing and sampling
methodologies. By proposing a structured, mixed-method approach tailored to
specific risk categories and platform characteristics, this article addresses
the challenge of evidence-based audits under the DSA. Our contribution
emphasizes the need for adaptable, context-sensitive auditing strategies and
adds to the emerging field of DSA compliance research.

### 4. The Impact of Large Language Models on K-12 Education in Rural India: A Thematic Analysis of Student Volunteer's Perspectives

[The Impact of Large Language Models on K-12 Education in Rural India: A Thematic Analysis of Student Volunteer's Perspectives](http://arxiv.org/pdf/2505.03163v1)

Authors: Harshita Goyal, Garima Garg, Prisha Mordia, Veena Ramachandran, Dhruv Kumar, Jagat Sesh Challa

AI-driven education, particularly Large Language Models (LLMs), has the
potential to address learning disparities in rural K-12 schools. However,
research on AI adoption in rural India remains limited, with existing studies
focusing primarily on urban settings. This study examines the perceptions of
volunteer teachers on AI integration in rural education, identifying key
challenges and opportunities. Through semi-structured interviews with 23
volunteer educators in Rajasthan and Delhi, we conducted a thematic analysis to
explore infrastructure constraints, teacher preparedness, and digital literacy
gaps. Findings indicate that while LLMs could enhance personalized learning and
reduce teacher workload, barriers such as poor connectivity, lack of AI
training, and parental skepticism hinder adoption. Despite concerns over
over-reliance and ethical risks, volunteers emphasize that AI should be seen as
a complementary tool rather than a replacement for traditional teaching. Given
the potential benefits, LLM-based tutors merit further exploration in rural
classrooms, with structured implementation and localized adaptations to ensure
accessibility and equity.

### 5. Validating the Effectiveness of a Large Language Model-based Approach for Identifying Children's Development across Various Free Play Settings in Kindergarten

[Validating the Effectiveness of a Large Language Model-based Approach for Identifying Children's Development across Various Free Play Settings in Kindergarten](http://arxiv.org/pdf/2505.03369v1)

Authors: Yuanyuan Yang, Yuan Shen, Tianchen Sun, Yangbin Xie

Free play is a fundamental aspect of early childhood education, supporting
children's cognitive, social, emotional, and motor development. However,
assessing children's development during free play poses significant challenges
due to the unstructured and spontaneous nature of the activity. Traditional
assessment methods often rely on direct observations by teachers, parents, or
researchers, which may fail to capture comprehensive insights from free play
and provide timely feedback to educators. This study proposes an innovative
approach combining Large Language Models (LLMs) with learning analytics to
analyze children's self-narratives of their play experiences. The LLM
identifies developmental abilities, while performance scores across different
play settings are calculated using learning analytics techniques. We collected
2,224 play narratives from 29 children in a kindergarten, covering four
distinct play areas over one semester. According to the evaluation results from
eight professionals, the LLM-based approach achieved high accuracy in
identifying cognitive, motor, and social abilities, with accuracy exceeding 90%
in most domains. Moreover, significant differences in developmental outcomes
were observed across play settings, highlighting each area's unique
contributions to specific abilities. These findings confirm that the proposed
approach is effective in identifying children's development across various free
play settings. This study demonstrates the potential of integrating LLMs and
learning analytics to provide child-centered insights into developmental
trajectories, offering educators valuable data to support personalized learning
and enhance early childhood education practices.

### 6. Cognitio Emergens: Agency, Dimensions, and Dynamics in Human-AI Knowledge Co-Creation

[Cognitio Emergens: Agency, Dimensions, and Dynamics in Human-AI Knowledge Co-Creation](http://arxiv.org/pdf/2505.03105v1)

Authors: Xule Lin

Scientific knowledge creation is fundamentally transforming as humans and AI
systems evolve beyond tool-user relationships into co-evolutionary epistemic
partnerships. When AlphaFold revolutionized protein structure prediction,
researchers described engaging with an epistemic partner that reshaped how they
conceptualized fundamental relationships. This article introduces Cognitio
Emergens (CE), a framework addressing critical limitations in existing models
that focus on static roles or narrow metrics while failing to capture how
scientific understanding emerges through recursive human-AI interaction over
time. CE integrates three components addressing these limitations: Agency
Configurations describing how authority distributes between humans and AI
(Directed, Contributory, Partnership), with partnerships dynamically
oscillating between configurations rather than following linear progression;
Epistemic Dimensions capturing six specific capabilities emerging through
collaboration across Discovery, Integration, and Projection axes, creating
distinctive "capability signatures" that guide development; and Partnership
Dynamics identifying forces shaping how these relationships evolve,
particularly the risk of epistemic alienation where researchers lose
interpretive control over knowledge they formally endorse. Drawing from
autopoiesis theory, social systems theory, and organizational modularity, CE
reveals how knowledge co-creation emerges through continuous negotiation of
roles, values, and organizational structures. By reconceptualizing human-AI
scientific collaboration as fundamentally co-evolutionary, CE offers a balanced
perspective that neither uncritically celebrates nor unnecessarily fears AI's
evolving role, instead providing conceptual tools for cultivating partnerships
that maintain meaningful human participation while enabling transformative
scientific breakthroughs.

### 7. BCause: Human-AI collaboration to improve hybrid mapping and ideation in argumentation-grounded deliberation

[BCause: Human-AI collaboration to improve hybrid mapping and ideation in argumentation-grounded deliberation](http://arxiv.org/pdf/2505.03584v1)

Authors: Lucas Anastasiou, Anna De Liddo

Public deliberation, as in open discussion of issues of public concern, often
suffers from scattered and shallow discourse, poor sensemaking, and a
disconnect from actionable policy outcomes. This paper introduces BCause, a
discussion system leveraging generative AI and human-machine collaboration to
transform unstructured dialogue around public issues (such as urban living,
policy changes, and current socio-economic transformations) into structured,
actionable democratic processes. We present three innovations: (i) importing
and transforming unstructured transcripts into argumentative discussions, (ii)
geo-deliberated problem-sensing via a Telegram bot for local issue reporting,
and (iii) smart reporting with customizable widgets (e.g., summaries, topic
modelling, policy recommendations, clustered arguments). The system's human-AI
partnership preserves critical human participation to ensure ethical oversight,
contextual relevance, and creative synthesis.

### Databases

### 1. Elastic Index Select for Label-Hybrid Search in Vector Database

[Elastic Index Select for Label-Hybrid Search in Vector Database](http://arxiv.org/pdf/2505.03212v1)

Authors: Mingyu Yang, Wenxuan Xia, Wentao Li, Raymond Chi-Wing Wong, Wei Wang

Real-world vector embeddings are usually associated with extra labels, such
as attributes and keywords. Many applications require the nearest neighbor
search that contains specific labels, such as searching for product image
embeddings restricted to a particular brand. A straightforward approach is to
materialize all possible indices according to the complete query label
workload. However, this leads to an exponential increase in both index space
and processing time, which significantly limits scalability and efficiency. In
this paper, we leverage the inclusion relationships among query label sets to
construct partial indexes, enabling index sharing across queries for improved
construction efficiency. We introduce \textit{elastic factor} bounds to
guarantee search performance and use the greedy algorithm to select indices
that meet the bounds, achieving a tradeoff between efficiency and space.
Meanwhile, we also designed the algorithm to achieve the best elastic factor
under a given space limitation. Experimental results on multiple real datasets
demonstrate that our algorithm can achieve near-optimal search performance,
achieving up to 10x-500x search efficiency speed up over state-of-the-art
approaches. Our algorithm is highly versatile, since it is not constrained by
index type and can seamlessly integrate with existing optimized libraries.

### 2. Beyond Relations: A Case for Elevating to the Entity-Relationship Abstraction

[Beyond Relations: A Case for Elevating to the Entity-Relationship Abstraction](http://arxiv.org/pdf/2505.03536v1)

Authors: Amol Deshpande

Spurred by a number of recent trends, we make the case that the relational
database systems should urgently move beyond supporting the basic
object-relational model and instead embrace a more abstract data model,
specifically, the entity-relationship model. We argue that the current RDBMSs
don't inherently support sufficient "logical" data independence, and that is
relegating the database systems to the role of a backend storage system, away
from where significant innovation is both happening and is still needed. We
present the design of a prototype system (ErbiumDB) that we are building to
explore these issues, and discuss some of the key research challenges.

### 3. From Word to Sentence: A Large-Scale Multi-Instance Dataset for Open-Set Aerial Detection

[From Word to Sentence: A Large-Scale Multi-Instance Dataset for Open-Set Aerial Detection](http://arxiv.org/pdf/2505.03334v1)

Authors: Guoting Wei, Yu Liu, Xia Yuan, Xizhe Xue, Linlin Guo, Yifan Yang, Chunxia Zhao, Zongwen Bai, Haokui Zhang, Rong Xiao

In recent years, language-guided open-world aerial object detection has
gained significant attention due to its better alignment with real-world
application needs. However, due to limited datasets, most existing
language-guided methods primarily focus on vocabulary, which fails to meet the
demands of more fine-grained open-world detection. To address this limitation,
we propose constructing a large-scale language-guided open-set aerial detection
dataset, encompassing three levels of language guidance: from words to phrases,
and ultimately to sentences. Centered around an open-source large
vision-language model and integrating image-operation-based preprocessing with
BERT-based postprocessing, we present the OS-W2S Label Engine, an automatic
annotation pipeline capable of handling diverse scene annotations for aerial
images. Using this label engine, we expand existing aerial detection datasets
with rich textual annotations and construct a novel benchmark dataset, called
Multi-instance Open-set Aerial Dataset (MI-OAD), addressing the limitations of
current remote sensing grounding data and enabling effective open-set aerial
detection. Specifically, MI-OAD contains 163,023 images and 2 million
image-caption pairs, approximately 40 times larger than comparable datasets. We
also employ state-of-the-art open-set methods from the natural image domain,
trained on our proposed dataset, to validate the model's open-set detection
capabilities. For instance, when trained on our dataset, Grounding DINO
achieves improvements of 29.5 AP_{50} and 33.7 Recall@10 for sentence inputs
under zero-shot transfer conditions. Both the dataset and the label engine will
be released publicly.

### Distributed, Parallel, and Cluster Computing

### 1. TailBench++: Flexible Multi-Client, Multi-Server Benchmarking for Latency-Critical Workloads

[TailBench++: Flexible Multi-Client, Multi-Server Benchmarking for Latency-Critical Workloads](http://arxiv.org/pdf/2505.03600v1)

Authors: Zhilin Li, Lucia Pons, Salvador Petit, Julio Sahuquillo, Julio Pons

Cloud systems have rapidly expanded worldwide in the last decade, shifting
computational tasks to cloud servers where clients submit their requests. Among
cloud workloads, latency-critical applications -- characterized by
high-percentile response times -- have gained special interest. These
applications are present in modern services, representing an important fraction
of cloud workloads. This work analyzes common cloud benchmarking suites and
identifies TailBench as the most suitable to assess cloud performance with
latency-critical workloads. Unfortunately, this suite presents key limitations,
especially in multi-server scenarios or environments with variable client
arrival patterns and fluctuating loads. To address these limitations, we
propose TailBench++, an enhanced benchmark suite that extends TailBench to
enable cloud evaluation studies to be performed in dynamic multi-client,
multi-server environments. It allows reproducing experiments with varying
client arrival times, dynamic query per second (QPS) fluctuations, and multiple
servers handling requests. Case studies show that TailBench++ enables more
realistic evaluations by capturing a wider range of real-world scenarios.

### 2. Revisiting Lower Bounds for Two-Step Consensus

[Revisiting Lower Bounds for Two-Step Consensus](http://arxiv.org/pdf/2505.03627v1)

Authors: Fedor Ryabinin, Alexey Gotsman, Pierre Sutra

A seminal result by Lamport shows that at least $\max\{2e+f+1,2f+1\}$
processes are required to implement partially synchronous consensus that
tolerates $f$ process failures and can furthermore decide in two message delays
under $e$ failures. This lower bound is matched by the classical Fast Paxos
protocol. However, more recent practical protocols, such as Egalitarian Paxos,
provide two-step decisions with fewer processes, seemingly contradicting the
lower bound. We show that this discrepancy arises because the classical bound
requires two-step decisions under a wide range of scenarios, not all of which
are relevant in practice. We propose a more pragmatic condition for which we
establish tight bounds on the number of processes required. Interestingly,
these bounds depend on whether consensus is implemented as an atomic object or
a decision task. For consensus as an object, $\max\{2e+f-1,2f+1\}$ processes
are necessary and sufficient for two-step decisions, while for a task the tight
bound is $\max\{2e+f, 2f+1\}$.

### 3. The Tensor-Core Beamformer: A High-Speed Signal-Processing Library for Multidisciplinary Use

[The Tensor-Core Beamformer: A High-Speed Signal-Processing Library for Multidisciplinary Use](http://arxiv.org/pdf/2505.03269v1)

Authors: Leon Oostrum, Bram Veenboer, Ronald Rook, Michael Brown, Pieter Kruizinga, John W. Romein

Beamforming is a well-known technique to combine signals from multiple
sensors. It has a wide range of application domains. This paper introduces the
Tensor-Core Beamformer: a generic, optimized beamformer library that harnesses
the computational power of GPU tensor cores to accelerate beamforming
computations. The library hides the complexity of tensor cores from the user,
and supports 16-bit and 1-bit precision. An extensive performance evaluation on
NVIDIA and AMD GPUs shows that the library outperforms traditional beamforming
on regular GPU cores by a wide margin, at much higher energy efficiency. In the
16-bit mode, it achieves over 600 TeraOps/s on an AMD MI300X GPU, while
approaching 1 TeraOp/J. In the 1-bit mode, it breaks the 3 PetaOps/s barrier
and achieves over 10 TeraOps/J on an NVIDIA A100 GPU. The beamforming library
can be easily integrated into existing pipelines. We demonstrate its use for
medical ultrasound and radio-astronomical instruments.

### 4. A Hashgraph-Inspired Consensus Mechanism for Reliable Multi-Model Reasoning

[A Hashgraph-Inspired Consensus Mechanism for Reliable Multi-Model Reasoning](http://arxiv.org/pdf/2505.03553v1)

Authors: Kolawole E. Ogunsina, Morayo A. Ogunsina

Inconsistent outputs and hallucinations from large language models (LLMs) are
major obstacles to reliable AI systems. When different proprietary reasoning
models (RMs), such as those by OpenAI, Google, Anthropic, DeepSeek, and xAI,
are given the same complex request, they often produce divergent results due to
variations in training and inference. This paper proposes a novel consensus
mechanism, inspired by distributed ledger technology, to validate and converge
these outputs, treating each RM as a black-box peer. Building on the Hashgraph
consensus algorithm, our approach employs gossip-about-gossip communication and
virtual voting to achieve agreement among an ensemble of RMs. We present an
architectural design for a prototype system in which RMs iteratively exchange
and update their answers, using information from each round to improve accuracy
and confidence in subsequent rounds. This approach goes beyond simple majority
voting by incorporating the knowledge and cross-verification content of every
model. We justify the feasibility of this Hashgraph-inspired consensus for AI
ensembles and outline its advantages over traditional ensembling techniques in
reducing nonfactual outputs. Preliminary considerations for implementation,
evaluation criteria for convergence and accuracy, and potential challenges are
discussed. The proposed mechanism demonstrates a promising direction for
multi-agent AI systems to self-validate and deliver high-fidelity responses in
complex tasks.

### 5. Decentralized Nonconvex Optimization under Heavy-Tailed Noise: Normalization and Optimal Convergence

[Decentralized Nonconvex Optimization under Heavy-Tailed Noise: Normalization and Optimal Convergence](http://arxiv.org/pdf/2505.03736v1)

Authors: Shuhua Yu, Dusan Jakovetic, Soummya Kar

Heavy-tailed noise in nonconvex stochastic optimization has garnered
increasing research interest, as empirical studies, including those on training
attention models, suggest it is a more realistic gradient noise condition. This
paper studies first-order nonconvex stochastic optimization under heavy-tailed
gradient noise in a decentralized setup, where each node can only communicate
with its direct neighbors in a predefined graph. Specifically, we consider a
class of heavy-tailed gradient noise that is zero-mean and has only $p$-th
moment for $p \in (1, 2]$. We propose GT-NSGDm, Gradient Tracking based
Normalized Stochastic Gradient Descent with momentum, that utilizes
normalization, in conjunction with gradient tracking and momentum, to cope with
heavy-tailed noise on distributed nodes. We show that, when the communication
graph admits primitive and doubly stochastic weights, GT-NSGDm guarantees, for
the \textit{first} time in the literature, that the expected gradient norm
converges at an optimal non-asymptotic rate $O\big(1/T^{(p-1)/(3p-2)}\big)$,
which matches the lower bound in the centralized setup. When tail index $p$ is
unknown, GT-NSGDm attains a non-asymptotic rate $O\big( 1/T^{(p-1)/(2p)} \big)$
that is, for $p < 2$, topology independent and has a speedup factor $n^{1-1/p}$
in terms of the number of nodes $n$. Finally, experiments on nonconvex linear
regression with tokenized synthetic data and decentralized training of language
models on a real-world corpus demonstrate that GT-NSGDm is more robust and
efficient than baselines.

### 6. Elevating Semantic Exploration: A Novel Approach Utilizing Distributed Repositories

[Elevating Semantic Exploration: A Novel Approach Utilizing Distributed Repositories](http://arxiv.org/pdf/2505.03443v1)

Authors: Valerio Bellandi

Centralized and distributed systems are two main approaches to organizing ICT
infrastructure, each with its pros and cons. Centralized systems concentrate
resources in one location, making management easier but creating single points
of failure. Distributed systems, on the other hand, spread resources across
multiple nodes, offering better scalability and fault tolerance, but requiring
more complex management. The choice between them depends on factors like
application needs, scalability, and data sensitivity. Centralized systems suit
applications with limited scalability and centralized control, while
distributed systems excel in large-scale environments requiring high
availability and performance. This paper explores a distributed document
repository system developed for the Italian Ministry of Justice, using edge
repositories to analyze textual data and metadata, enhancing semantic
exploration capabilities.

### Discrete Mathematics

### 1. On edge-colouring-games by Erdős, and Bensmail and Mc Inerney

[On edge-colouring-games by Erdős, and Bensmail and Mc Inerney](http://arxiv.org/pdf/2505.03497v1)

Authors: Stijn Cambie, Michiel Provoost

We consider two games proposed by Erd\H{o}s, and one game by Bensmail and Mc
Inerney, all with the same setup of two players alternately colouring one edge
of a clique. We give observations and particular behaviour for each of these
problems, and prove a first reduction towards confirming the conjecture by
Bensmail and Mc Inerney. We state a conjecture for Erd\H{o}s' game on the
largest induced maximum degree, and extensions to edge-transitive and,
respectively, regular graphs.

### Data Structures and Algorithms

### 1. Stochastic scheduling with Bernoulli-type jobs through policy stratification

[Stochastic scheduling with Bernoulli-type jobs through policy stratification](http://arxiv.org/pdf/2505.03349v1)

Authors: Antonios Antoniadis, Ruben Hoeksma, Kevin Schewior, Marc Uetz

This paper addresses the problem of computing a scheduling policy that
minimizes the total expected completion time of a set of $N$ jobs with
stochastic processing times on $m$ parallel identical machines. When all
processing times follow Bernoulli-type distributions, Gupta et al. (SODA '23)
exhibited approximation algorithms with an approximation guarantee
$\tilde{\text{O}}(\sqrt{m})$, where $m$ is the number of machines and
$\tilde{\text{O}}(\cdot)$ suppresses polylogarithmic factors in $N$, improving
upon an earlier ${\text{O}}(m)$ approximation by Eberle et al. (OR Letters '19)
for a special case. The present paper shows that, quite unexpectedly, the
problem with Bernoulli-type jobs admits a PTAS whenever the number of different
job-size parameters is bounded by a constant. The result is based on a series
of transformations of an optimal scheduling policy to a "stratified" policy
that makes scheduling decisions at specific points in time only, while losing
only a negligible factor in expected cost. An optimal stratified policy is
computed using dynamic programming. Two technical issues are solved, namely (i)
to ensure that, with at most a slight delay, the stratified policy has an
information advantage over the optimal policy, allowing it to simulate its
decisions, and (ii) to ensure that the delays do not accumulate, thus solving
the trade-off between the complexity of the scheduling policy and its expected
cost. Our results also imply a quasi-polynomial $\text{O}(\log
N)$-approximation for the case with an arbitrary number of job sizes.

### 2. Planar Disjoint Shortest Paths is Fixed-Parameter Tractable

[Planar Disjoint Shortest Paths is Fixed-Parameter Tractable](http://arxiv.org/pdf/2505.03353v1)

Authors: Michał Pilipczuk, Giannos Stamoulis, Michał Włodarczyk

In the Disjoint Shortest Paths problem one is given a graph $G$ and a set
$\mathcal{T}=\{(s_1,t_1),\dots,(s_k,t_k)\}$ of $k$ vertex pairs. The question
is whether there exist vertex-disjoint paths $P_1,\dots,P_k$ in $G$ so that
each $P_i$ is a shortest path between $s_i$ and $t_i$. While the problem is
known to be W[1]-hard in general, we show that it is fixed-parameter tractable
on planar graphs with positive edge weights. Specifically, we propose an
algorithm for Planar Disjoint Shortest Paths with running time $2^{O(k\log
k)}\cdot n^{O(1)}$. Notably, our parameter dependency is better than
state-of-the-art $2^{O(k^2)}$ for the Planar Disjoint Paths problem, where the
sought paths are not required to be shortest paths.

### 3. GPU Implementation of the Wavelet Tree

[GPU Implementation of the Wavelet Tree](http://arxiv.org/pdf/2505.03372v1)

Authors: Marco Franzreb, Martin Burtscher, Stephan Rudolph

I present a new GPU implementation of the wavelet tree data structure. It
includes binary rank and select support structures that provide at least 10
times higher throughput of binary rank and select queries than the best
publicly available CPU implementations at comparable storage overhead. My work
also presents a new parallel tree construction algorithm that, when excluding
the time to copy the data from the CPU to the GPU, outperforms the current
state of the art. The GPU implementation, given enough parallelism, processes
access, rank, and select queries at least 2x faster than the wavelet tree
implementation contained in the widely used Succinct Data Structure Library
(SDSL), including the time necessary to copy the queries from the CPU to the
GPU and the results back to the CPU from the GPU.

### 4. Location-Restricted Stable Matching

[Location-Restricted Stable Matching](http://arxiv.org/pdf/2505.03680v1)

Authors: Garret Castro

Motivated by group-project distribution, we introduce and study stable
matching under the constraint of applicants needing to share a location to be
matched with the same institute, which we call the Location-Restricted Stable
Matching problem (LRSM). We show that finding a feasible matching is NP-hard,
making finding a feasible and stable matching automatically NP-hard. We then
analyze the subproblem where all the projects have the same capacity, and the
applicant population of each location is a multiple of the universal project
capacity, which mimics more realistic constraints and makes finding a feasible
matching in P. Even under these conditions, a stable matching (a matching
without blocking pairs) may not exist, so we look for a matching that minimizes
the number of blocking pairs. We find that the blocking pair minimization
problem for this subproblem is inapproximable within $|A|^{1-\epsilon}$ for
$|A|$ agents and provide an $|A|$-approximation algorithm to show this result
is almost tight. We extend this result to show that the problem of minimizing
the number of agents in blocking pairs is also inapproximable within
$|A|^{1-\epsilon}$, and since there are only $|A|$ agents, this result is also
almost tight.

### 5. Multiplication of polynomials over finite fields

[Multiplication of polynomials over finite fields](http://arxiv.org/pdf/2505.03101v1)

Authors: Chunlei Liu

Additive Fourier Transform is sdudied. The technique of Gao-Mateer is
generalized, enabling us to a fast multiplication of polynomials over finite
fields.

### 6. A practical algorithm for 2-admissibility

[A practical algorithm for 2-admissibility](http://arxiv.org/pdf/2505.03419v1)

Authors: Christine Awofeso, Patrick Greaves, Oded Lachish, Felix Reidl

The $2$-admissibility of a graph is a promising measure to identify
real-world networks which have an algorithmically favourable structure. In
contrast to other related measures, like the weak/strong $2$-colouring numbers
or the maximum density of graphs that appear as $1$-subdivisions, the
$2$-admissibility can be computed in polynomial time. However, so far these
results are theoretical only and no practical implementation to compute the
$2$-admissibility exists.
  Here we present an algorithm which decides whether the $2$-admissibility of
an input graph $G$ is at most $p$ in time $O(p^4 |V(G)|)$ and space $O(|E(G)| +
p^2)$. The simple structure of the algorithm makes it easy to implement. We
evaluate our implementation on a corpus of 214 real-world networks and find
that the algorithm runs efficiently even on networks with millions of edges,
that it has a low memory footprint, and that indeed many real world networks
have a small $2$-admissibility.

### 7. Lower Bounds for Greedy Teaching Set Constructions

[Lower Bounds for Greedy Teaching Set Constructions](http://arxiv.org/pdf/2505.03223v1)

Authors: Spencer Compton, Chirag Pabbaraju, Nikita Zhivotovskiy

A fundamental open problem in learning theory is to characterize the
best-case teaching dimension $\operatorname{TS}_{\min}$ of a concept class
$\mathcal{C}$ with finite VC dimension $d$. Resolving this problem will, in
particular, settle the conjectured upper bound on Recursive Teaching Dimension
posed by [Simon and Zilles; COLT 2015]. Prior work used a natural greedy
algorithm to construct teaching sets recursively, thereby proving upper bounds
on $\operatorname{TS}_{\min}$, with the best known bound being $O(d^2)$ [Hu,
Wu, Li, and Wang; COLT 2017]. In each iteration, this greedy algorithm chooses
to add to the teaching set the $k$ labeled points that restrict the concept
class the most. In this work, we prove lower bounds on the performance of this
greedy approach for small $k$. Specifically, we show that for $k = 1$, the
algorithm does not improve upon the halving-based bound of
$O(\log(|\mathcal{C}|))$. Furthermore, for $k = 2$, we complement the upper
bound of $O\left(\log(\log(|\mathcal{C}|))\right)$ from [Moran, Shpilka,
Wigderson, and Yuhudayoff; FOCS 2015] with a matching lower bound. Most
consequentially, our lower bound extends up to $k \le \lceil c d \rceil$ for
small constant $c>0$: suggesting that studying higher-order interactions may be
necessary to resolve the conjecture that $\operatorname{TS}_{\min} = O(d)$.

### 8. Troika algorithm: approximate optimization for accurate clique partitioning and clustering of weighted networks

[Troika algorithm: approximate optimization for accurate clique partitioning and clustering of weighted networks](http://arxiv.org/pdf/2505.03573v1)

Authors: Samin Aref, Boris Ng

Clique partitioning is a fundamental network clustering task, with
applications in a wide range of computational sciences. It involves identifying
an optimal partition of the nodes for a real-valued weighted graph according to
the edge weights. An optimal partition is one that maximizes the sum of
within-cluster edge weights over all possible node partitions. This paper
introduces a novel approximation algorithm named Troika to solve this NP-hard
problem in small to mid-sized networks for instances of theoretical and
practical relevance. Troika uses a branch-and-cut scheme for branching on node
triples to find a partition that is within a user-specified optimality gap
tolerance. Troika offers advantages over alternative methods like integer
programming solvers and heuristics for clique partitioning. Unlike existing
heuristics, Troika returns solutions within a guaranteed proximity to global
optimality. And our results indicate that Troika is faster than using the
state-of-the-art integer programming solver Gurobi for most benchmark
instances. Besides its advantages for solving the clique partitioning problem,
we demonstrate the applications of Troika in community detection and portfolio
analysis. Troika returns partitions with higher proximity to optimal compared
to eight modularity-based community detection algorithms. When used on networks
of correlations among stocks, Troika reveals the dynamic changes in the
structure of portfolio networks including downturns from the 2008 financial
crisis and the reaction to the COVID-19 pandemic. Our comprehensive results
based on benchmarks from the literature and new real and random networks point
to Troika as a reliable and accurate method for solving clique partitioning
instances with up to 5000 edges on standard hardware.

### Emerging Technologies

### 1. Fluid Volume Assignment for Flow-Based Biochips: State-of-the-Art and Research Challenges

[Fluid Volume Assignment for Flow-Based Biochips: State-of-the-Art and Research Challenges](http://arxiv.org/pdf/2505.03540v1)

Authors: Alexander Schneider, Jan Madsen, Paul Pop

Microfluidic biochips are replacing the conventional biochemical analysers
integrating the necessary functions on-chip. We are interested in Flow-Based
Microfluidic Biochips (FBMB), where a continuous flow of liquid is manipulated
using integrated microvalves. Using microvalves and channels, more complex
Fluidic Units (FUs) such as switches, micropumps, mixers and separators can be
constructed. When running a biochemical application on a FBMB, fluid volumes
are dispensed from input reservoirs and used by the FUs. Given a biochemical
application and a biochip, one of the key problems which we are discussing in
this paper, is in determining the fluid volume assignment for each operation of
the application, such that the FUs' volume requirements are satisfied, while
over- and underflow are avoided and the total volume of fluid used is
minimized. We illustrate the main problems using examples, and provide a review
of related work on volume management. We present algorithms for optimizing
fluid volume assignments and for reusing leftover fluids to reduce waste. This
also includes the optimization of mixing operations which significantly impact
the required fluid volumes. We identify the main challenges related to volume
management and discuss possible solutions. Finally we compare the outcome of
volume management using fixed- and arbitrary-ratio mixing technology,
demonstrating significant reductions in fluid consumption for real biochemical
assays.

### 2. Exploring the application of quantum technologies to industrial and real-world use cases

[Exploring the application of quantum technologies to industrial and real-world use cases](http://arxiv.org/pdf/2505.03302v1)

Authors: Eneko Osaba, Esther Villar-Rodriguez, Izaskun Oregi

Recent advancements in quantum computing are leading to an era of practical
utility, enabling the tackling of increasingly complex problems. The goal of
this era is to leverage quantum computing to solve real-world problems in
fields such as machine learning, optimization, and material simulation, using
revolutionary quantum methods and machines. All this progress has been achieved
even while being immersed in the noisy intermediate-scale quantum era,
characterized by the current devices' inability to process medium-scale complex
problems efficiently. Consequently, there has been a surge of interest in
quantum algorithms in various fields. Multiple factors have played a role in
this extraordinary development, with three being particularly noteworthy: (i)
the development of larger devices with enhanced interconnections between their
constituent qubits, (ii) the development of specialized frameworks, and (iii)
the existence of well-known or ready-to-use hybrid schemes that simplify the
method development process. In this context, this manuscript presents and
overviews some recent contributions within this paradigm, showcasing the
potential of quantum computing to emerge as a significant research catalyst in
the fields of machine learning and optimization in the coming years.

### 3. DNA Tails for Molecular Flash Memory

[DNA Tails for Molecular Flash Memory](http://arxiv.org/pdf/2505.03629v1)

Authors: Jin Sima, Chao Pan, S. Kasra Tabatabaei, Alvaro G. Hernandez, Charles M. Schroeder, Olgica Milenkovic

DNA-based data storage systems face practical challenges due to the high cost
of DNA synthesis. A strategy to address the problem entails encoding data via
topological modifications of the DNA sugar-phosphate backbone. The DNA
Punchcards system, which introduces nicks (cuts) in the DNA backbone, encodes
only one bit per nicking site, limiting density. We propose \emph{DNA Tails,} a
storage paradigm that encodes nonbinary symbols at nicking sites by growing
enzymatically synthesized single-stranded DNA of varied lengths. The average
tail lengths encode multiple information bits and are controlled via a
staggered nicking-tail extension process. We demonstrate the feasibility of
this encoding approach experimentally and identify common sources of errors,
such as calibration errors and stumped tail growth errors. To mitigate
calibration errors, we use rank modulation proposed for flash memory. To
correct stumped tail growth errors, we introduce a new family of rank
modulation codes that can correct ``stuck-at'' errors. Our analytical results
include constructions for order-optimal-redundancy permutation codes and
accompanying encoding and decoding algorithms.

### Graphics

### 1. Evaluating Foveated Frame Rate Reduction in Virtual Reality for Head-Mounted Displays

[Evaluating Foveated Frame Rate Reduction in Virtual Reality for Head-Mounted Displays](http://arxiv.org/pdf/2505.03682v1)

Authors: Christopher Flöter, Sergej Geringer, Guido Reina, Daniel Weiskopf, Timo Ropinski

Foveated rendering methods usually reduce spatial resolution in the periphery
of the users' view. However, using foveated rendering to reduce temporal
resolution, i.e., rendering frame rate, seems less explored. In this work, we
present the results of a user study investigating the perceptual effects of
foveated temporal resolution reduction, where only the temporal resolution
(frame rate) is reduced in the periphery without affecting spatial quality
(pixel density). In particular, we investigated the perception of temporal
resolution artifacts caused by reducing the frame rate dependent on the
eccentricity of the user's gaze. Our user study with 15 participants was
conducted in a virtual reality setting using a head-mounted display. Our
results indicate that it was possible to reduce average rendering costs, i.e.,
the number of rendered pixels, to a large degree before participants
consistently reported perceiving temporal artifacts.

### 2. StableMotion: Training Motion Cleanup Models with Unpaired Corrupted Data

[StableMotion: Training Motion Cleanup Models with Unpaired Corrupted Data](http://arxiv.org/pdf/2505.03154v1)

Authors: Yuxuan Mu, Hung Yu Ling, Yi Shi, Ismael Baira Ojeda, Pengcheng Xi, Chang Shu, Fabio Zinno, Xue Bin Peng

Motion capture (mocap) data often exhibits visually jarring artifacts due to
inaccurate sensors and post-processing. Cleaning this corrupted data can
require substantial manual effort from human experts, which can be a costly and
time-consuming process. Previous data-driven motion cleanup methods offer the
promise of automating this cleanup process, but often require in-domain paired
corrupted-to-clean training data. Constructing such paired datasets requires
access to high-quality, relatively artifact-free motion clips, which often
necessitates laborious manual cleanup. In this work, we present StableMotion, a
simple yet effective method for training motion cleanup models directly from
unpaired corrupted datasets that need cleanup. The core component of our method
is the introduction of motion quality indicators, which can be easily annotated
through manual labeling or heuristic algorithms and enable training of
quality-aware motion generation models on raw motion data with mixed quality.
At test time, the model can be prompted to generate high-quality motions using
the quality indicators. Our method can be implemented through a simple
diffusion-based framework, leading to a unified motion generate-discriminate
model, which can be used to both identify and fix corrupted frames. We
demonstrate that our proposed method is effective for training motion cleanup
models on raw mocap data in production scenarios by applying StableMotion to
SoccerMocap, a 245-hour soccer mocap dataset containing real-world motion
artifacts. The trained model effectively corrects a wide range of motion
artifacts, reducing motion pops and frozen frames by 68% and 81%, respectively.
See https://youtu.be/3Y7MMAH02B4 for more results.

### Computer Science and Game Theory

### 1. Truthful Facility Location with Candidate Locations and Limited Resources

[Truthful Facility Location with Candidate Locations and Limited Resources](http://arxiv.org/pdf/2505.03391v1)

Authors: Panagiotis Kanellopoulos, Alexandros A. Voudouris

We study a truthful facility location problem where one out of $k\geq2$
available facilities must be built at a location chosen from a set of candidate
ones in the interval $[0,1]$. This decision aims to accommodate a set of agents
with private positions in $[0,1]$ and approval preferences over the facilities;
the agents act strategically and may misreport their private information to
maximize their utility, which depends on the chosen facility and their distance
from it. We focus on strategyproof mechanisms that incentivize the agents to
act truthfully and bound the best possible approximation of the optimal social
welfare (the total utility of the agents) they can achieve. We first show that
deterministic mechanisms have unbounded approximation ratio, and then present a
randomized mechanism with approximation ratio $k$, which is tight even when
agents may only misreport their positions. For the restricted setting where
agents may only misreport their approval preferences, we design a deterministic
mechanism with approximation ratio of roughly $2.325$, and establish lower
bounds of $3/2$ and $6/5$ for deterministic and randomized mechanisms,
respectively.

### 2. Airdrop Games

[Airdrop Games](http://arxiv.org/pdf/2505.03428v1)

Authors: Sotiris Georganas, Aggelos Kiayias, Paolo Penna

Launching a new blockchain system or application is frequently facilitated by
a so called airdrop, where the system designer chooses a pre-existing set of
potentially interested parties and allocates newly minted tokens to them with
the expectation that they will participate in the system - such engagement,
especially if it is of significant level, facilitates the system and raises its
value and also the value of its newly minted token, hence benefiting the
airdrop recipients. A number of challenging questions befuddle designers in
this setting, such as how to choose the set of interested parties and how to
allocate tokens to them. To address these considerations we put forward a
game-theoretic model for such airdrop games. Our model can be used to guide the
designer's choices based on the way the system's value depends on participation
(modeled by a ''technology function'' in our framework) and the costs that
participants incur. We identify both bad and good equilibria and identify the
settings and the choices that can be made where the designer can influence the
players towards good equilibria in an expedient manner.

### 3. Simultaneous All-Pay Auctions with Budget Constraints

[Simultaneous All-Pay Auctions with Budget Constraints](http://arxiv.org/pdf/2505.03291v1)

Authors: Yan Liu, Ying Qin, Zihe Wang

The all-pay auction, a classic competitive model, is widely applied in
scenarios such as political elections, sports competitions, and research and
development, where all participants pay their bids regardless of winning or
losing. However, in the traditional all-pay auction, players have no budget
constraints, whereas in real-world scenarios, players typically face budget
constraints. This paper studies the Nash equilibrium of two players with budget
constraints across multiple heterogeneous items in a complete-information
framework. The main contributions are as follows: (1) a comprehensive
characterization of the Nash equilibrium in single-item auctions with
asymmetric budgets and valuations; (2) the construction of a joint distribution
Nash equilibrium for the two-item scenario; and (3) the construction of a joint
distribution Nash equilibrium for the three-item scenario. Unlike the
unconstrained all-pay auction, which always has a Nash equilibrium, a Nash
equilibrium may not exist when players have budget constraints. Our findings
highlight the intricate effects of budget constraints on bidding strategies,
providing new perspectives and methodologies for theoretical analysis and
practical applications of all-pay auctions.

### 4. Coevolution of Actions and Opinions in Networks of Coordinating and Anti-Coordinating Agents

[Coevolution of Actions and Opinions in Networks of Coordinating and Anti-Coordinating Agents](http://arxiv.org/pdf/2505.03078v1)

Authors: Hong Liang, Mengbin Ye, Lorenzo Zino, Weiguo Xia

In this paper, we investigate the dynamics of coordinating and
anti-coordinating agents in a coevolutionary model for actions and opinions. In
the model, the individuals of a population interact on a two-layer network,
sharing their opinions and observing others' action, while revising their own
opinions and actions according to a game-theoretic mechanism, grounded in the
social psychology literature. First, we consider the scenario of coordinating
agents, where convergence to a Nash equilibrium (NE) is guaranteed. We identify
conditions for reaching consensus configurations and establish regions of
attraction for these equilibria. Second, we study networks of anti-coordinating
agents. In this second scenario, we prove that all trajectories converge to a
NE by leveraging potential game theory. Then, we establish analytical
conditions on the network structure and model parameters to guarantee the
existence of consensus and polarized equilibria, characterizing their regions
of attraction.

### 5. Multi-Agent Deep Reinforcement Learning for Zonal Ancillary Market Coupling

[Multi-Agent Deep Reinforcement Learning for Zonal Ancillary Market Coupling](http://arxiv.org/pdf/2505.03288v1)

Authors: Francesco Morri, Hélène Le Cadre, Pierre Gruet, Luce Brotcorne

We characterize zonal ancillary market coupling relying on noncooperative
game theory. To that purpose, we formulate the ancillary market as a
multi-leader single follower bilevel problem, that we subsequently cast as a
generalized Nash game with side constraints and nonconvex feasibility sets. We
determine conditions for equilibrium existence and show that the game has a
generalized potential game structure. To compute market equilibrium, we rely on
two exact approaches: an integrated optimization approach and Gauss-Seidel
best-response, that we compare against multi-agent deep reinforcement learning.
On real data from Germany and Austria, simulations indicate that multi-agent
deep reinforcement learning achieves the smallest convergence rate but requires
pretraining, while best-response is the slowest. On the economics side,
multi-agent deep reinforcement learning results in smaller market costs
compared to the exact methods, but at the cost of higher variability in the
profit allocation among stakeholders. Further, stronger coupling between zones
tends to reduce costs for larger zones.

### 6. On edge-colouring-games by Erdős, and Bensmail and Mc Inerney

[On edge-colouring-games by Erdős, and Bensmail and Mc Inerney](http://arxiv.org/pdf/2505.03497v1)

Authors: Stijn Cambie, Michiel Provoost

We consider two games proposed by Erd\H{o}s, and one game by Bensmail and Mc
Inerney, all with the same setup of two players alternately colouring one edge
of a clique. We give observations and particular behaviour for each of these
problems, and prove a first reduction towards confirming the conjecture by
Bensmail and Mc Inerney. We state a conjecture for Erd\H{o}s' game on the
largest induced maximum degree, and extensions to edge-transitive and,
respectively, regular graphs.

### Human-Computer Interaction

### 1. Do ATCOs Need Explanations, and Why? Towards ATCO-Centered Explainable AI for Conflict Resolution Advisories

[Do ATCOs Need Explanations, and Why? Towards ATCO-Centered Explainable AI for Conflict Resolution Advisories](http://arxiv.org/pdf/2505.03117v1)

Authors: Katherine Fennedy, Brian Hilburn, Thaivalappil N. M. Nadirsha, Sameer Alam, Khanh-Duy Le, Hua Li

Interest in explainable artificial intelligence (XAI) is surging. Prior
research has primarily focused on systems' ability to generate explanations,
often guided by researchers' intuitions rather than end-users' needs.
Unfortunately, such approaches have not yielded favorable outcomes when
compared to a black-box baseline (i.e., no explanation). To address this gap,
this paper advocates a human-centered approach that shifts focus to air traffic
controllers (ATCOs) by asking a fundamental yet overlooked question: Do ATCOs
need explanations, and if so, why? Insights from air traffic management (ATM),
human-computer interaction, and the social sciences were synthesized to provide
a holistic understanding of XAI challenges and opportunities in ATM. Evaluating
11 ATM operational goals revealed a clear need for explanations when ATCOs aim
to document decisions and rationales for future reference or report generation.
Conversely, ATCOs are less likely to seek them when their conflict resolution
approach align with the artificial intelligence (AI) advisory. While this is a
preliminary study, the findings are expected to inspire broader and deeper
inquiries into the design of ATCO-centric XAI systems, paving the way for more
effective human-AI interaction in ATM.

### 2. InfoVids: Reimagining the Viewer Experience with Alternative Visualization-Presenter Relationships

[InfoVids: Reimagining the Viewer Experience with Alternative Visualization-Presenter Relationships](http://arxiv.org/pdf/2505.03164v1)

Authors: Ji Won Chung, Tongyu Zhou, Ivy Chen, Kevin Hsu, Ryan A. Rossi, Alexa Siu, Shunan Guo, Franck Dernoncourt, James Tompkin, Jeff Huang

Traditional data presentations typically separate the presenter and
visualization into two separate spaces--the 3D world and a 2D screen--enforcing
visualization-centric stories. To create a more human-centric viewing
experience, we establish a more equitable relationship between the
visualization and the presenter through our InfoVids. These
infographics-inspired informational videos are crafted to redefine
relationships between the presenter and visualizations. As we design InfoVids,
we explore how the use of layout, form, and interactions affects the viewer
experience. We compare InfoVids against their baseline 2D `slides' equivalents
across 9 metrics with 30 participants and provide practical, long-term insights
from an autobiographical perspective. Our mixed methods analyses reveal that
this paradigm reduced viewer attention splitting, shifted the focus from the
visualization to the presenter, and led to more interactive, natural, and
engaging full-body data performances for viewers. Ultimately, InfoVids helped
viewers re-imagine traditional dynamics between the presenter and
visualizations.

### 3. Behavioral Sensing and Intervention Paradigm: A Review of Closed-Loop Approaches for Ingestion Health

[Behavioral Sensing and Intervention Paradigm: A Review of Closed-Loop Approaches for Ingestion Health](http://arxiv.org/pdf/2505.03185v1)

Authors: Jun Fang, Yanuo Zhou, Ka I Chan, Jiajin Li, Zeyi Sun, Zhengnan Li, Zicong Fu, Hongjing Piao, Haodong Xu, Yuanchun Shi, Yuntao Wang

Ingestive behavior plays a critical role in health, yet many existing
interventions remain limited to static guidance or manual self-tracking. With
the increasing integration of sensors and perceptual computing, recent systems
have begun to support closed-loop interventions that dynamically sense user
behavior and provide feedback during or around ingestion episodes. In this
survey, we review 136 studies that leverage sensor-enabled or
interaction-mediated approaches to influence eating behavior. We propose a
behavioral closed-loop paradigm comprising three core components: target
behaviors, sensing modalities, and feedback strategies. A taxonomy of sensing
and intervention modalities is presented, organized along human- and
environment-based dimensions. Our analysis also examines evaluation methods and
design trends across different modality-behavior pairings. This review reveals
prevailing patterns and critical gaps, offering design insights for future
adaptive and context-aware ingestion health interventions.

### 4. DroidRetriever: An Autonomous Navigation and Information Integration System Facilitating Mobile Sensemaking

[DroidRetriever: An Autonomous Navigation and Information Integration System Facilitating Mobile Sensemaking](http://arxiv.org/pdf/2505.03364v1)

Authors: Yiheng Bian, Yunpeng Song, Guiyu Ma, Rongrong Zhu, Zhongmin Cai

Users regularly rely on mobile applications for their daily information
needs, and mobile sensemaking is prevalent in various domains such as
education, healthcare, business intelligence, and emergency response, where
timely and context-aware information-processing and decision-making is
critical. However, valuable information is often scattered across the closed
ecosystems within various applications, posing challenges for traditional
search engines to retrieve data openly and in real-time. Additionally, due to
limitations such as mobile device screen sizes, language differences, and
unfamiliarity with specific applications and domain knowledge, users have to
frequently switch between multiple applications and spend substantial time
locating and integrating the information. To address these challenges, we
present DroidRetriever, a system for cross-application information retrieval to
facilitate mobile sensemaking. DroidRetriever can automatically navigate to
relevant interfaces based on users' natural language commands, capture
screenshots, extract and integrate information, and finally present the
results. Our experimental results demonstrate that DroidRetriever can extract
and integrate information with near-human accuracy while significantly reducing
processing time. Furthermore, with minimal user intervention, DroidRetriever
effectively corrects and completes various information retrieval tasks,
substantially reducing the user's workload. Our summary of the motivations for
intervention and the discussion of their necessity provide valuable
implications for future research. We will open-source our code upon acceptance
of the paper.

### 5. AI-Based Feedback in Counselling Competence Training of Prospective Teachers

[AI-Based Feedback in Counselling Competence Training of Prospective Teachers](http://arxiv.org/pdf/2505.03423v1)

Authors: Tobias Hallmen, Kathrin Gietl, Karoline Hillesheim, Moritz Bauermann, Annemarie Friedrich, Elisabeth André

This study explores the use of AI-based feedback to enhance the counselling
competence of prospective teachers. An iterative block seminar was designed,
incorporating theoretical foundations, practical applications, and AI tools for
analysing verbal, paraverbal, and nonverbal communication. The seminar included
recorded simulated teacher-parent conversations, followed by AI-based feedback
and qualitative interviews with students. The study investigated correlations
between communication characteristics and conversation quality, student
perceptions of AI-based feedback, and the training of AI models to identify
conversation phases and techniques. Results indicated significant correlations
between nonverbal and paraverbal features and conversation quality, and
students positively perceived the AI feedback. The findings suggest that
AI-based feedback can provide objective, actionable insights to improve teacher
training programs. Future work will focus on refining verbal skill annotations,
expanding the dataset, and exploring additional features to enhance the
feedback system.

### 6. manvr3d: A Platform for Human-in-the-loop Cell Tracking in Virtual Reality

[manvr3d: A Platform for Human-in-the-loop Cell Tracking in Virtual Reality](http://arxiv.org/pdf/2505.03440v1)

Authors: Samuel Pantze, Jean-Yves Tinevez, Matthew McGinity, Ulrik Günther

We propose manvr3d, a novel VR-ready platform for interactive
human-in-the-loop cell tracking. We utilize VR controllers and eye-tracking
hardware to facilitate rapid ground truth generation and proofreading for deep
learning-based cell tracking models. Life scientists reconstruct the
developmental history of organisms on the cellular level by analyzing 3D
time-lapse microscopy images acquired at high spatio-temporal resolution. The
reconstruction of such cell lineage trees traditionally involves tracking
individual cells through all recorded time points, manually annotating their
positions, and then linking them over time to create complete trajectories.
Deep learning-based algorithms accelerate this process, yet depend heavily on
manually-annotated high-quality ground truth data and curation. Visual
representation of the image data in this process still relies primarily on 2D
renderings, which greatly limits spatial understanding and navigation. In this
work, we bridge the gap between deep learning-based cell tracking software and
3D/VR visualization to create a human-in-the-loop cell tracking system. We lift
the incremental annotation, training and proofreading loop of the deep learning
model into the 3rd dimension and apply natural user interfaces like hand
gestures and eye tracking to accelerate the cell tracking workflow for life
scientists.

### 7. Scalable Class-Centric Visual Interactive Labeling

[Scalable Class-Centric Visual Interactive Labeling](http://arxiv.org/pdf/2505.03618v1)

Authors: Matthias Matt, Jana Sedlakova, Jürgen Bernard, Matthias Zeppelzauer, Manuela Waldner

Large unlabeled datasets demand efficient and scalable data labeling
solutions, in particular when the number of instances and classes is large.
This leads to significant visual scalability challenges and imposes a high
cognitive load on the users. Traditional instance-centric labeling methods,
where (single) instances are labeled in each iteration struggle to scale
effectively in these scenarios. To address these challenges, we introduce cVIL,
a Class-Centric Visual Interactive Labeling methodology designed for
interactive visual data labeling. By shifting the paradigm from
assigning-classes-to-instances to assigning-instances-to-classes, cVIL reduces
labeling effort and enhances efficiency for annotators working with large,
complex and class-rich datasets. We propose a novel visual analytics labeling
interface built on top of the conceptual cVIL workflow, enabling improved
scalability over traditional visual labeling. In a user study, we demonstrate
that cVIL can improve labeling efficiency and user satisfaction over
instance-centric interfaces. The effectiveness of cVIL is further demonstrated
through a usage scenario, showcasing its potential to alleviate cognitive load
and support experts in managing extensive labeling tasks efficiently.

### 8. The Impact of Large Language Models on K-12 Education in Rural India: A Thematic Analysis of Student Volunteer's Perspectives

[The Impact of Large Language Models on K-12 Education in Rural India: A Thematic Analysis of Student Volunteer's Perspectives](http://arxiv.org/pdf/2505.03163v1)

Authors: Harshita Goyal, Garima Garg, Prisha Mordia, Veena Ramachandran, Dhruv Kumar, Jagat Sesh Challa

AI-driven education, particularly Large Language Models (LLMs), has the
potential to address learning disparities in rural K-12 schools. However,
research on AI adoption in rural India remains limited, with existing studies
focusing primarily on urban settings. This study examines the perceptions of
volunteer teachers on AI integration in rural education, identifying key
challenges and opportunities. Through semi-structured interviews with 23
volunteer educators in Rajasthan and Delhi, we conducted a thematic analysis to
explore infrastructure constraints, teacher preparedness, and digital literacy
gaps. Findings indicate that while LLMs could enhance personalized learning and
reduce teacher workload, barriers such as poor connectivity, lack of AI
training, and parental skepticism hinder adoption. Despite concerns over
over-reliance and ethical risks, volunteers emphasize that AI should be seen as
a complementary tool rather than a replacement for traditional teaching. Given
the potential benefits, LLM-based tutors merit further exploration in rural
classrooms, with structured implementation and localized adaptations to ensure
accessibility and equity.

### 9. Patterns and Mechanisms of Contrastive Activation Engineering

[Patterns and Mechanisms of Contrastive Activation Engineering](http://arxiv.org/pdf/2505.03189v1)

Authors: Yixiong Hao, Ayush Panda, Stepan Shabalin, Sheikh Abdur Raheem Ali

Controlling the behavior of Large Language Models (LLMs) remains a
significant challenge due to their inherent complexity and opacity. While
techniques like fine-tuning can modify model behavior, they typically require
extensive computational resources. Recent work has introduced a class of
contrastive activation engineering (CAE) techniques as promising approaches for
steering LLM outputs through targeted modifications to their internal
representations. Applied at inference-time with zero cost, CAE has the
potential to introduce a new paradigm of flexible, task-specific LLM behavior
tuning. We analyze the performance of CAE in in-distribution,
out-of-distribution settings, evaluate drawbacks, and begin to develop
comprehensive guidelines for its effective deployment. We find that 1. CAE is
only reliably effective when applied to in-distribution contexts. 2. Increasing
the number of samples used to generate steering vectors has diminishing returns
at around 80 samples. 3. Steering vectors are susceptible to adversarial inputs
that reverses the behavior that is steered for. 4. Steering vectors harm the
overall model perplexity. 5. Larger models are more resistant to
steering-induced degradation.

### 10. Tell Me the Good Stuff: User Preferences in Movie Recommendation Explanations

[Tell Me the Good Stuff: User Preferences in Movie Recommendation Explanations](http://arxiv.org/pdf/2505.03376v1)

Authors: Juan Ahmad, Jonas Hellgren, Alan Said

Recommender systems play a vital role in helping users discover content in
streaming services, but their effectiveness depends on users understanding why
items are recommended. In this study, explanations were based solely on item
features rather than personalized data, simulating recommendation scenarios. We
compared user perceptions of one-sided (purely positive) and two-sided
(positive and negative) feature-based explanations for popular movie
recommendations. Through an online study with 129 participants, we examined how
explanation style affected perceived trust, transparency, effectiveness, and
satisfaction. One-sided explanations consistently received higher ratings
across all dimensions. Our findings suggest that in low-stakes entertainment
domains such as popular movie recommendations, simpler positive explanations
may be more effective. However, the results should be interpreted with caution
due to potential confounding factors such as item familiarity and the placement
of negative information in explanations. This work provides practical insights
for explanation design in recommender interfaces and highlights the importance
of context in shaping user preferences.

### Information Retrieval

### 1. Characterising Topic Familiarity and Query Specificity Using Eye-Tracking Data

[Characterising Topic Familiarity and Query Specificity Using Eye-Tracking Data](http://arxiv.org/pdf/2505.03136v1)

Authors: Jiaman He, Zikang Leng, Dana McKay, Johanne R. Trippas, Damiano Spina

Eye-tracking data has been shown to correlate with a user's knowledge level
and query formulation behaviour. While previous work has focused primarily on
eye gaze fixations for attention analysis, often requiring additional
contextual information, our study investigates the memory-related cognitive
dimension by relying solely on pupil dilation and gaze velocity to infer users'
topic familiarity and query specificity without needing any contextual
information. Using eye-tracking data collected via a lab user study (N=18), we
achieved a Macro F1 score of 71.25% for predicting topic familiarity with a
Gradient Boosting classifier, and a Macro F1 score of 60.54% with a k-nearest
neighbours (KNN) classifier for query specificity. Furthermore, we developed a
novel annotation guideline -- specifically tailored for question answering --
to manually classify queries as Specific or Non-specific. This study
demonstrates the feasibility of eye-tracking to better understand topic
familiarity and query specificity in search.

### 2. Soft Reasoning Paths for Knowledge Graph Completion

[Soft Reasoning Paths for Knowledge Graph Completion](http://arxiv.org/pdf/2505.03285v1)

Authors: Yanning Hou, Sihang Zhou, Ke Liang, Lingyuan Meng, Xiaoshu Chen, Ke Xu, Siwei Wang, Xinwang Liu, Jian Huang

Reasoning paths are reliable information in knowledge graph completion (KGC)
in which algorithms can find strong clues of the actual relation between
entities. However, in real-world applications, it is difficult to guarantee
that computationally affordable paths exist toward all candidate entities.
According to our observation, the prediction accuracy drops significantly when
paths are absent. To make the proposed algorithm more stable against the
missing path circumstances, we introduce soft reasoning paths. Concretely, a
specific learnable latent path embedding is concatenated to each relation to
help better model the characteristics of the corresponding paths. The
combination of the relation and the corresponding learnable embedding is termed
a soft path in our paper. By aligning the soft paths with the reasoning paths,
a learnable embedding is guided to learn a generalized path representation of
the corresponding relation. In addition, we introduce a hierarchical ranking
strategy to make full use of information about the entity, relation, path, and
soft path to help improve both the efficiency and accuracy of the model.
Extensive experimental results illustrate that our algorithm outperforms the
compared state-of-the-art algorithms by a notable margin. The code will be made
publicly available after the paper is officially accepted.

### 3. STAR-Rec: Making Peace with Length Variance and Pattern Diversity in Sequential Recommendation

[STAR-Rec: Making Peace with Length Variance and Pattern Diversity in Sequential Recommendation](http://arxiv.org/pdf/2505.03484v1)

Authors: Maolin Wang, Sheng Zhang, Ruocheng Guo, Wanyu Wang, Xuetao Wei, Zitao Liu, Hongzhi Yin, Yi Chang, Xiangyu Zhao

Recent deep sequential recommendation models often struggle to effectively
model key characteristics of user behaviors, particularly in handling sequence
length variations and capturing diverse interaction patterns. We propose
STAR-Rec, a novel architecture that synergistically combines preference-aware
attention and state-space modeling through a sequence-level mixture-of-experts
framework. STAR-Rec addresses these challenges by: (1) employing
preference-aware attention to capture both inherently similar item
relationships and diverse preferences, (2) utilizing state-space modeling to
efficiently process variable-length sequences with linear complexity, and (3)
incorporating a mixture-of-experts component that adaptively routes different
behavioral patterns to specialized experts, handling both focused
category-specific browsing and diverse category exploration patterns. We
theoretically demonstrate how the state space model and attention mechanisms
can be naturally unified in recommendation scenarios, where SSM captures
temporal dynamics through state compression while attention models both similar
and diverse item relationships. Extensive experiments on four real-world
datasets demonstrate that STAR-Rec consistently outperforms state-of-the-art
sequential recommendation methods, particularly in scenarios involving diverse
user behaviors and varying sequence lengths.

### 4. 1$^{st}$ Place Solution of WWW 2025 EReL@MIR Workshop Multimodal CTR Prediction Challenge

[1$^{st}$ Place Solution of WWW 2025 EReL@MIR Workshop Multimodal CTR Prediction Challenge](http://arxiv.org/pdf/2505.03543v1)

Authors: Junwei Xu, Zehao Zhao, Xiaoyu Hu, Zhenjie Song

The WWW 2025 EReL@MIR Workshop Multimodal CTR Prediction Challenge focuses on
effectively applying multimodal embedding features to improve click-through
rate (CTR) prediction in recommender systems. This technical report presents
our 1$^{st}$ place winning solution for Task 2, combining sequential modeling
and feature interaction learning to effectively capture user-item interactions.
For multimodal information integration, we simply append the frozen multimodal
embeddings to each item embedding. Experiments on the challenge dataset
demonstrate the effectiveness of our method, achieving superior performance
with a 0.9839 AUC on the leaderboard, much higher than the baseline model. Code
and configuration are available in our GitHub repository and the checkpoint of
our model can be found in HuggingFace.

### 5. Avoid Recommending Out-of-Domain Items: Constrained Generative Recommendation with LLMs

[Avoid Recommending Out-of-Domain Items: Constrained Generative Recommendation with LLMs](http://arxiv.org/pdf/2505.03336v1)

Authors: Hao Liao, Wensheng Lu, Jianxun Lian, Mingqi Wu, Shuo Wang, Yong Zhang, Yitian Huang, Mingyang Zhou, Xing Xie

Large Language Models (LLMs) have shown promise for generative recommender
systems due to their transformative capabilities in user interaction. However,
ensuring they do not recommend out-of-domain (OOD) items remains a challenge.
We study two distinct methods to address this issue: RecLM-ret, a
retrieval-based method, and RecLM-cgen, a constrained generation method. Both
methods integrate seamlessly with existing LLMs to ensure in-domain
recommendations. Comprehensive experiments on three recommendation datasets
demonstrate that RecLM-cgen consistently outperforms RecLM-ret and existing
LLM-based recommender models in accuracy while eliminating OOD recommendations,
making it the preferred method for adoption. Additionally, RecLM-cgen maintains
strong generalist capabilities and is a lightweight plug-and-play module for
easy integration into LLMs, offering valuable practical benefits for the
community. Source code is available at https://github.com/microsoft/RecAI

### 6. Tell Me the Good Stuff: User Preferences in Movie Recommendation Explanations

[Tell Me the Good Stuff: User Preferences in Movie Recommendation Explanations](http://arxiv.org/pdf/2505.03376v1)

Authors: Juan Ahmad, Jonas Hellgren, Alan Said

Recommender systems play a vital role in helping users discover content in
streaming services, but their effectiveness depends on users understanding why
items are recommended. In this study, explanations were based solely on item
features rather than personalized data, simulating recommendation scenarios. We
compared user perceptions of one-sided (purely positive) and two-sided
(positive and negative) feature-based explanations for popular movie
recommendations. Through an online study with 129 participants, we examined how
explanation style affected perceived trust, transparency, effectiveness, and
satisfaction. One-sided explanations consistently received higher ratings
across all dimensions. Our findings suggest that in low-stakes entertainment
domains such as popular movie recommendations, simpler positive explanations
may be more effective. However, the results should be interpreted with caution
due to potential confounding factors such as item familiarity and the placement
of negative information in explanations. This work provides practical insights
for explanation design in recommender interfaces and highlights the importance
of context in shaping user preferences.

### 7. Advancing Remote and Continuous Cardiovascular Patient Monitoring through a Novel and Resource-efficient IoT-Driven Framework

[Advancing Remote and Continuous Cardiovascular Patient Monitoring through a Novel and Resource-efficient IoT-Driven Framework](http://arxiv.org/pdf/2505.03409v1)

Authors: Sanam Nayab, Sohail Raza Chohan, Aqsa Jameel, Syed Rehan Shah, Syed Ahsan Masud Zaidi, Aditya Nath Jha, Kamran Siddique

Cardiovascular diseases are a leading cause of fatalities worldwide, often
occurring suddenly with limited time for intervention. Current healthcare
monitoring systems for cardiac patients rely heavily on hospitalization, which
can be impractical for continuous monitoring. This paper presents a novel
IoT-based solution for remote, real-time tracking of critical cardiac metrics,
addressing the pressing need for accessible and continuous healthcare,
particularly for the aging population in Pakistan. The proposed IoT kit
measures essential parameters such as body temperature, heart rate (HR), blood
pressure (BP), oxygen saturation (SPO2), and electrocardiography (ECG).
  A key innovation of the system is its integration with a cloud-based
application, enabling constant remote monitoring and incorporating an alarm
mechanism to alert medical professionals for timely intervention, reducing the
risk of catastrophic incidents. The system was tested in a clinical environment
with 20 participants, demonstrating results closely aligned with those obtained
using standard medical devices. The findings validate the system's potential
for reliable remote monitoring, offering a significant step forward in
proactive cardiac healthcare management. This novel approach combines IoT
technology with cloud-based applications to provide a cost-effective and
efficient solution for reducing unexpected fatalities among cardiac patients.

### 8. Familiarizing with Music: Discovery Patterns for Different Music Discovery Needs

[Familiarizing with Music: Discovery Patterns for Different Music Discovery Needs](http://arxiv.org/pdf/2505.03568v1)

Authors: Marta Moscati, Darius Afchar, Markus Schedl, Bruno Sguerra

Humans have the tendency to discover and explore. This natural tendency is
reflected in data from streaming platforms as the amount of previously unknown
content accessed by users. Additionally, in domains such as that of music
streaming there is evidence that recommending novel content improves users'
experience with the platform. Therefore, understanding users' discovery
patterns, such as the amount to which and the way users access previously
unknown content, is a topic of relevance for both the scientific community and
the streaming industry, particularly the music one. Previous works studied how
music consumption differs for users of different traits and looked at
diversity, novelty, and consistency over time of users' music preferences.
However, very little is known about how users discover and explore previously
unknown music, and how this behavior differs for users of varying discovery
needs. In this paper we bridge this gap by analyzing data from a survey
answered by users of the major music streaming platform Deezer in combination
with their streaming data. We first address questions regarding whether users
who declare a higher interest in unfamiliar music listen to more diverse music,
have more stable music preferences over time, and explore more music within a
same time window, compared to those who declare a lower interest. We then
investigate which type of music tracks users choose to listen to when they
explore unfamiliar music, identifying clear patterns of popularity and genre
representativeness that vary for users of different discovery needs.
  Our findings open up possibilities to infer users' interest in unfamiliar
music from streaming data as well as possibilities to develop recommender
systems that guide users in exploring music in a more natural way.

### 9. Counterfactual Inference for Eliminating Sentiment Bias in Recommender Systems

[Counterfactual Inference for Eliminating Sentiment Bias in Recommender Systems](http://arxiv.org/pdf/2505.03655v1)

Authors: Le Pan, Yuanjiang Cao, Chengkai Huang, Wenjie Zhang, Lina Yao

Recommender Systems (RSs) aim to provide personalized recommendations for
users. A newly discovered bias, known as sentiment bias, uncovers a common
phenomenon within Review-based RSs (RRSs): the recommendation accuracy of users
or items with negative reviews deteriorates compared with users or items with
positive reviews. Critical users and niche items are disadvantaged by such
unfair recommendations. We study this problem from the perspective of
counterfactual inference with two stages. At the model training stage, we build
a causal graph and model how sentiment influences the final rating score.
During the inference stage, we decouple the direct and indirect effects to
mitigate the impact of sentiment bias and remove the indirect effect using
counterfactual inference. We have conducted extensive experiments, and the
results validate that our model can achieve comparable performance on rating
prediction for better recommendations and effective mitigation of sentiment
bias. To the best of our knowledge, this is the first work to employ
counterfactual inference on sentiment bias mitigation in RSs.

### 10. Rational Retrieval Acts: Leveraging Pragmatic Reasoning to Improve Sparse Retrieval

[Rational Retrieval Acts: Leveraging Pragmatic Reasoning to Improve Sparse Retrieval](http://arxiv.org/pdf/2505.03676v1)

Authors: Arthur Satouf, Gabriel Ben Zenou, Benjamin Piwowarski, Habiboulaye Amadou Boubacar, Pablo Piantanida

Current sparse neural information retrieval (IR) methods, and to a lesser
extent more traditional models such as BM25, do not take into account the
document collection and the complex interplay between different term weights
when representing a single document. In this paper, we show how the Rational
Speech Acts (RSA), a linguistics framework used to minimize the number of
features to be communicated when identifying an object in a set, can be adapted
to the IR case -- and in particular to the high number of potential features
(here, tokens). RSA dynamically modulates token-document interactions by
considering the influence of other documents in the dataset, better contrasting
document representations. Experiments show that incorporating RSA consistently
improves multiple sparse retrieval models and achieves state-of-the-art
performance on out-of-domain datasets from the BEIR benchmark.
https://github.com/arthur-75/Rational-Retrieval-Acts

### Machine Learning

### 1. Adversarial Attacks in Multimodal Systems: A Practitioner's Survey

[Adversarial Attacks in Multimodal Systems: A Practitioner's Survey](http://arxiv.org/pdf/2505.03084v1)

Authors: Shashank Kapoor, Sanjay Surendranath Girija, Lakshit Arora, Dipen Pradhan, Ankit Shetgaonkar, Aman Raj

The introduction of multimodal models is a huge step forward in Artificial
Intelligence. A single model is trained to understand multiple modalities:
text, image, video, and audio. Open-source multimodal models have made these
breakthroughs more accessible. However, considering the vast landscape of
adversarial attacks across these modalities, these models also inherit
vulnerabilities of all the modalities, and ultimately, the adversarial threat
amplifies. While broad research is available on possible attacks within or
across these modalities, a practitioner-focused view that outlines attack types
remains absent in the multimodal world. As more Machine Learning Practitioners
adopt, fine-tune, and deploy open-source models in real-world applications,
it's crucial that they can view the threat landscape and take the preventive
actions necessary. This paper addresses the gap by surveying adversarial
attacks targeting all four modalities: text, image, video, and audio. This
survey provides a view of the adversarial attack landscape and presents how
multimodal adversarial threats have evolved. To the best of our knowledge, this
survey is the first comprehensive summarization of the threat landscape in the
multimodal world.

### 2. Plug-and-Play AMC: Context Is King in Training-Free, Open-Set Modulation with LLMs

[Plug-and-Play AMC: Context Is King in Training-Free, Open-Set Modulation with LLMs](http://arxiv.org/pdf/2505.03112v1)

Authors: Mohammad Rostami, Atik Faysal, Reihaneh Gh. Roshan, Huaxia Wang, Nikhil Muralidhar, Yu-Dong Yao

Automatic Modulation Classification (AMC) is critical for efficient spectrum
management and robust wireless communications. However, AMC remains challenging
due to the complex interplay of signal interference and noise. In this work, we
propose an innovative framework that integrates traditional signal processing
techniques with Large-Language Models (LLMs) to address AMC. Our approach
leverages higher-order statistics and cumulant estimation to convert
quantitative signal features into structured natural language prompts. By
incorporating exemplar contexts into these prompts, our method exploits the
LLM's inherent familiarity with classical signal processing, enabling effective
one-shot classification without additional training or preprocessing (e.g.,
denoising). Experimental evaluations on synthetically generated datasets,
spanning both noiseless and noisy conditions, demonstrate that our framework
achieves competitive performance across diverse modulation schemes and
Signal-to-Noise Ratios (SNRs). Moreover, our approach paves the way for robust
foundation models in wireless communications across varying channel conditions,
significantly reducing the expense associated with developing channel-specific
models. This work lays the foundation for scalable, interpretable, and
versatile signal classification systems in next-generation wireless networks.
The source code is available at https://github.com/RU-SIT/context-is-king

### 3. Adaptive Thresholding for Multi-Label Classification via Global-Local Signal Fusion

[Adaptive Thresholding for Multi-Label Classification via Global-Local Signal Fusion](http://arxiv.org/pdf/2505.03118v1)

Authors: Dmytro Shamatrin

Multi-label classification (MLC) requires predicting multiple labels per
sample, often under heavy class imbalance and noisy conditions. Traditional
approaches apply fixed thresholds or treat labels independently, overlooking
context and global rarity. We introduce an adaptive thresholding mechanism that
fuses global (IDF-based) and local (KNN-based) signals to produce per-label,
per-instance thresholds. Instead of applying these as hard cutoffs, we treat
them as differentiable penalties in the loss, providing smooth supervision and
better calibration. Our architecture is lightweight, interpretable, and highly
modular. On the AmazonCat-13K benchmark, it achieves a macro-F1 of 0.1712,
substantially outperforming tree-based and pretrained transformer-based
methods. We release full code for reproducibility and future extensions.

### 4. Rethinking the Global Convergence of Softmax Policy Gradient with Linear Function Approximation

[Rethinking the Global Convergence of Softmax Policy Gradient with Linear Function Approximation](http://arxiv.org/pdf/2505.03155v1)

Authors: Max Qiushi Lin, Jincheng Mei, Matin Aghaei, Michael Lu, Bo Dai, Alekh Agarwal, Dale Schuurmans, Csaba Szepesvari, Sharan Vaswani

Policy gradient (PG) methods have played an essential role in the empirical
successes of reinforcement learning. In order to handle large state-action
spaces, PG methods are typically used with function approximation. In this
setting, the approximation error in modeling problem-dependent quantities is a
key notion for characterizing the global convergence of PG methods. We focus on
Softmax PG with linear function approximation (referred to as
$\texttt{Lin-SPG}$) and demonstrate that the approximation error is irrelevant
to the algorithm's global convergence even for the stochastic bandit setting.
Consequently, we first identify the necessary and sufficient conditions on the
feature representation that can guarantee the asymptotic global convergence of
$\texttt{Lin-SPG}$. Under these feature conditions, we prove that $T$
iterations of $\texttt{Lin-SPG}$ with a problem-specific learning rate result
in an $O(1/T)$ convergence to the optimal policy. Furthermore, we prove that
$\texttt{Lin-SPG}$ with any arbitrary constant learning rate can ensure
asymptotic global convergence to the optimal policy.

### 5. VLM Q-Learning: Aligning Vision-Language Models for Interactive Decision-Making

[VLM Q-Learning: Aligning Vision-Language Models for Interactive Decision-Making](http://arxiv.org/pdf/2505.03181v1)

Authors: Jake Grigsby, Yuke Zhu, Michael Ryoo, Juan Carlos Niebles

Recent research looks to harness the general knowledge and reasoning of large
language models (LLMs) into agents that accomplish user-specified goals in
interactive environments. Vision-language models (VLMs) extend LLMs to
multi-modal data and provide agents with the visual reasoning necessary for new
applications in areas such as computer automation. However, agent tasks
emphasize skills where accessible open-weight VLMs lag behind their LLM
equivalents. For example, VLMs are less capable of following an environment's
strict output syntax requirements and are more focused on open-ended question
answering. Overcoming these limitations requires supervised fine-tuning (SFT)
on task-specific expert demonstrations. Our work approaches these challenges
from an offline-to-online reinforcement learning (RL) perspective. RL lets us
fine-tune VLMs to agent tasks while learning from the unsuccessful decisions of
our own model or more capable (larger) models. We explore an off-policy RL
solution that retains the stability and simplicity of the widely used SFT
workflow while allowing our agent to self-improve and learn from low-quality
datasets. We demonstrate this technique with two open-weight VLMs across three
multi-modal agent domains.

### 6. Convergence Of Consistency Model With Multistep Sampling Under General Data Assumptions

[Convergence Of Consistency Model With Multistep Sampling Under General Data Assumptions](http://arxiv.org/pdf/2505.03194v1)

Authors: Yiding Chen, Yiyi Zhang, Owen Oertell, Wen Sun

Diffusion models accomplish remarkable success in data generation tasks
across various domains. However, the iterative sampling process is
computationally expensive. Consistency models are proposed to learn consistency
functions to map from noise to data directly, which allows one-step fast data
generation and multistep sampling to improve sample quality. In this paper, we
study the convergence of consistency models when the self-consistency property
holds approximately under the training distribution. Our analysis requires only
mild data assumption and applies to a family of forward processes. When the
target data distribution has bounded support or has tails that decay
sufficiently fast, we show that the samples generated by the consistency model
are close to the target distribution in Wasserstein distance; when the target
distribution satisfies some smoothness assumption, we show that with an
additional perturbation step for smoothing, the generated samples are close to
the target distribution in total variation distance. We provide two case
studies with commonly chosen forward processes to demonstrate the benefit of
multistep sampling.

### 7. Partial Label Clustering

[Partial Label Clustering](http://arxiv.org/pdf/2505.03207v1)

Authors: Yutong Xie, Fuchao Yang, Yuheng Jia

Partial label learning (PLL) is a significant weakly supervised learning
framework, where each training example corresponds to a set of candidate labels
and only one label is the ground-truth label. For the first time, this paper
investigates the partial label clustering problem, which takes advantage of the
limited available partial labels to improve the clustering performance.
Specifically, we first construct a weight matrix of examples based on their
relationships in the feature space and disambiguate the candidate labels to
estimate the ground-truth label based on the weight matrix. Then, we construct
a set of must-link and cannot-link constraints based on the disambiguation
results. Moreover, we propagate the initial must-link and cannot-link
constraints based on an adversarial prior promoted dual-graph learning
approach. Finally, we integrate weight matrix construction, label
disambiguation, and pairwise constraints propagation into a joint model to
achieve mutual enhancement. We also theoretically prove that a better
disambiguated label matrix can help improve clustering performance.
Comprehensive experiments demonstrate our method realizes superior performance
when comparing with state-of-the-art constrained clustering methods, and
outperforms PLL and semi-supervised PLL methods when only limited samples are
annotated. The code is publicly available at https://github.com/xyt-ml/PLC.

### 8. DYSTIL: Dynamic Strategy Induction with Large Language Models for Reinforcement Learning

[DYSTIL: Dynamic Strategy Induction with Large Language Models for Reinforcement Learning](http://arxiv.org/pdf/2505.03209v1)

Authors: Borui Wang, Kathleen McKeown, Rex Ying

Reinforcement learning from expert demonstrations has long remained a
challenging research problem, and existing state-of-the-art methods using
behavioral cloning plus further RL training often suffer from poor
generalization, low sample efficiency, and poor model interpretability.
Inspired by the strong reasoning abilities of large language models (LLMs), we
propose a novel strategy-based reinforcement learning framework integrated with
LLMs called DYnamic STrategy Induction with Llms for reinforcement learning
(DYSTIL) to overcome these limitations. DYSTIL dynamically queries a
strategy-generating LLM to induce textual strategies based on advantage
estimations and expert demonstrations, and gradually internalizes induced
strategies into the RL agent through policy optimization to improve its
performance through boosting policy generalization and enhancing sample
efficiency. It also provides a direct textual channel to observe and interpret
the evolution of the policy's underlying strategies during training. We test
DYSTIL over challenging RL environments from Minigrid and BabyAI, and
empirically demonstrate that DYSTIL significantly outperforms state-of-the-art
baseline methods by 17.75% in average success rate while also enjoying higher
sample efficiency during the learning process.

### 9. Joint Resource Management for Energy-efficient UAV-assisted SWIPT-MEC: A Deep Reinforcement Learning Approach

[Joint Resource Management for Energy-efficient UAV-assisted SWIPT-MEC: A Deep Reinforcement Learning Approach](http://arxiv.org/pdf/2505.03230v1)

Authors: Yue Chen, Hui Kang, Jiahui Li, Geng Su, Boxiong Wang, Jiacheng Wang, Cong Liang, Shuang Liang, Dusit Niyato

The integration of simultaneous wireless information and power transfer
(SWIPT) technology in 6G Internet of Things (IoT) networks faces significant
challenges in remote areas and disaster scenarios where ground infrastructure
is unavailable. This paper proposes a novel unmanned aerial vehicle
(UAV)-assisted mobile edge computing (MEC) system enhanced by directional
antennas to provide both computational resources and energy support for ground
IoT terminals. However, such systems require multiple trade-off policies to
balance UAV energy consumption, terminal battery levels, and computational
resource allocation under various constraints, including limited UAV battery
capacity, non-linear energy harvesting characteristics, and dynamic task
arrivals. To address these challenges comprehensively, we formulate a
bi-objective optimization problem that simultaneously considers system energy
efficiency and terminal battery sustainability. We then reformulate this
non-convex problem with a hybrid solution space as a Markov decision process
(MDP) and propose an improved soft actor-critic (SAC) algorithm with an action
simplification mechanism to enhance its convergence and generalization
capabilities. Simulation results have demonstrated that our proposed approach
outperforms various baselines in different scenarios, achieving efficient
energy management while maintaining high computational performance.
Furthermore, our method shows strong generalization ability across different
scenarios, particularly in complex environments, validating the effectiveness
of our designed boundary penalty and charging reward mechanisms.

### 10. MDPs with a State Sensing Cost

[MDPs with a State Sensing Cost](http://arxiv.org/pdf/2505.03280v1)

Authors: Vansh Kapoor, Jayakrishnan Nair

In many practical sequential decision-making problems, tracking the state of
the environment incurs a sensing/communication/computation cost. In these
settings, the agent's interaction with its environment includes the additional
component of deciding $\textit{when}$ to sense the state, in a manner that
balances the value associated with optimal (state-specific) actions and the
cost of sensing. We formulate this as an expected discounted cost Markov
Decision Process (MDP), wherein the agent incurs an additional cost for sensing
its next state, but has the option to take actions while remaining 'blind' to
the system state.
  We pose this problem as a classical discounted cost MDP with an expanded
(countably infinite) state space. While computing the optimal policy for this
MDP is intractable in general, we bound the sub-optimality gap associated with
optimal policies in a restricted class, where the number of consecutive
non-sensing (a.k.a., blind) actions is capped. We also design a computationally
efficient heuristic algorithm based on policy improvement, which in practice
performs close to the optimal policy. Finally, we benchmark against the state
of the art via a numerical case study.

### Neural and Evolutionary Computing

### 1. Accelerating Evolution: Integrating PSO Principles into Real-Coded Genetic Algorithm Crossover

[Accelerating Evolution: Integrating PSO Principles into Real-Coded Genetic Algorithm Crossover](http://arxiv.org/pdf/2505.03217v1)

Authors: Xiaobo Jin, JiaShu Tu

This study introduces an innovative crossover operator named Particle Swarm
Optimization-inspired Crossover (PSOX), which is specifically developed for
real-coded genetic algorithms. Departing from conventional crossover approaches
that only exchange information between individuals within the same generation,
PSOX uniquely incorporates guidance from both the current global best solution
and historical optimal solutions across multiple generations. This novel
mechanism enables the algorithm to maintain population diversity while
simultaneously accelerating convergence toward promising regions of the search
space. The effectiveness of PSOX is rigorously evaluated through comprehensive
experiments on 15 benchmark test functions with diverse characteristics,
including unimodal, multimodal, and highly complex landscapes. Comparative
analysis against five state-of-the-art crossover operators reveals that PSOX
consistently delivers superior performance in terms of solution accuracy,
algorithmic stability, and convergence speed, especially when combined with an
appropriate mutation strategy. Furthermore, the study provides an in-depth
investigation of how different mutation rates influence PSOX's performance,
yielding practical guidelines for parameter tuning when addressing optimization
problems with varying landscape properties.

### 2. Artificial Protozoa Optimizer (APO): A novel bio-inspired metaheuristic algorithm for engineering optimization

[Artificial Protozoa Optimizer (APO): A novel bio-inspired metaheuristic algorithm for engineering optimization](http://arxiv.org/pdf/2505.03512v1)

Authors: Xiaopeng Wang, Vaclav Snasel, Seyedali Mirjalili, Jeng-Shyang Pan, Lingping Kong, Hisham A. Shehadeh

This study proposes a novel artificial protozoa optimizer (APO) that is
inspired by protozoa in nature. The APO mimics the survival mechanisms of
protozoa by simulating their foraging, dormancy, and reproductive behaviors.
The APO was mathematically modeled and implemented to perform the optimization
processes of metaheuristic algorithms. The performance of the APO was verified
via experimental simulations and compared with 32 state-of-the-art algorithms.
Wilcoxon signed-rank test was performed for pairwise comparisons of the
proposed APO with the state-of-the-art algorithms, and Friedman test was used
for multiple comparisons. First, the APO was tested using 12 functions of the
2022 IEEE Congress on Evolutionary Computation benchmark. Considering
practicality, the proposed APO was used to solve five popular engineering
design problems in a continuous space with constraints. Moreover, the APO was
applied to solve a multilevel image segmentation task in a discrete space with
constraints. The experiments confirmed that the APO could provide highly
competitive results for optimization problems. The source codes of Artificial
Protozoa Optimizer are publicly available at
https://seyedalimirjalili.com/projects and
https://ww2.mathworks.cn/matlabcentral/fileexchange/162656-artificial-protozoa-optimizer.

### 3. From Neurons to Computation: Biological Reservoir Computing for Pattern Recognition

[From Neurons to Computation: Biological Reservoir Computing for Pattern Recognition](http://arxiv.org/pdf/2505.03510v1)

Authors: Ludovico Iannello, Luca Ciampi, Gabriele Lagani, Fabrizio Tonelli, Eleonora Crocco, Lucio Maria Calcagnile, Angelo Di Garbo, Federico Cremisi, Giuseppe Amato

In this paper, we introduce a novel paradigm for reservoir computing (RC)
that leverages a pool of cultured biological neurons as the reservoir
substrate, creating a biological reservoir computing (BRC). This system
operates similarly to an echo state network (ESN), with the key distinction
that the neural activity is generated by a network of cultured neurons, rather
than being modeled by traditional artificial computational units. The neuronal
activity is recorded using a multi-electrode array (MEA), which enables
high-throughput recording of neural signals. In our approach, inputs are
introduced into the network through a subset of the MEA electrodes, while the
remaining electrodes capture the resulting neural activity. This generates a
nonlinear mapping of the input data to a high-dimensional biological feature
space, where distinguishing between data becomes more efficient and
straightforward, allowing a simple linear classifier to perform pattern
recognition tasks effectively. To evaluate the performance of our proposed
system, we present an experimental study that includes various input patterns,
such as positional codes, bars with different orientations, and a digit
recognition task. The results demonstrate the feasibility of using biological
neural networks to perform tasks traditionally handled by artificial neural
networks, paving the way for further exploration of biologically-inspired
computing systems, with potential applications in neuromorphic engineering and
bio-hybrid computing.

### Networking and Internet Architecture

### 1. Efficient Wi-Fi Sensing for IoT Forensics with Lossy Compression of CSI Data

[Efficient Wi-Fi Sensing for IoT Forensics with Lossy Compression of CSI Data](http://arxiv.org/pdf/2505.03375v1)

Authors: Paolo Cerutti, Fabio Palmese, Marco Cominelli, Alessandro E. C. Redondi

Wi-Fi sensing is an emerging technology that uses channel state information
(CSI) from ambient Wi-Fi signals to monitor human activity without the need for
dedicated sensors. Wi-Fi sensing does not only represent a pivotal technology
in intelligent Internet of Things (IoT) systems, but it can also provide
valuable insights in forensic investigations. However, the high dimensionality
of CSI data presents major challenges for storage, transmission, and processing
in resource-constrained IoT environments. In this paper, we investigate the
impact of lossy compression on the accuracy of Wi-Fi sensing, evaluating both
traditional techniques and a deep learning-based approach. Our results reveal
that simple, interpretable techniques based on principal component analysis can
significantly reduce the CSI data volume while preserving classification
performance, making them highly suitable for lightweight IoT forensic
scenarios. On the other hand, deep learning models exhibit higher potential in
complex applications like activity recognition (achieving compression ratios up
to 16000:1 with minimal impact on sensing performance) but require careful
tuning and greater computational resources. By considering two different
sensing applications, this work demonstrates the feasibility of integrating
lossy compression schemes into Wi-Fi sensing pipelines to make intelligent IoT
systems more efficient and improve the storage requirements in forensic
applications.

### 2. Multi-Agent Reinforcement Learning Scheduling to Support Low Latency in Teleoperated Driving

[Multi-Agent Reinforcement Learning Scheduling to Support Low Latency in Teleoperated Driving](http://arxiv.org/pdf/2505.03558v1)

Authors: Giacomo Avanzi, Marco Giordani, Michele Zorzi

The teleoperated driving (TD) scenario comes with stringent Quality of
Service (QoS) communication constraints, especially in terms of end-to-end
(E2E) latency and reliability. In this context, Predictive Quality of Service
(PQoS), possibly combined with Reinforcement Learning (RL) techniques, is a
powerful tool to estimate QoS degradation and react accordingly. For example,
an intelligent agent can be trained to select the optimal compression
configuration for automotive data, and reduce the file size whenever QoS
conditions deteriorate. However, compression may inevitably compromise data
quality, with negative implications for the TD application. An alternative
strategy involves operating at the Radio Access Network (RAN) level to optimize
radio parameters based on current network conditions, while preserving data
quality. In this paper, we propose Multi-Agent Reinforcement Learning (MARL)
scheduling algorithms, based on Proximal Policy Optimization (PPO), to
dynamically and intelligently allocate radio resources to minimize E2E latency
in a TD scenario. We evaluate two training paradigms, i.e., decentralized
learning with local observations (IPPO) vs. centralized aggregation (MAPPO), in
conjunction with two resource allocation strategies, i.e., proportional
allocation (PA) and greedy allocation (GA). We prove via ns-3 simulations that
MAPPO, combined with GA, achieves the best results in terms of latency,
especially as the number of vehicles increases.

### 3. A Trustworthy Multi-LLM Network: Challenges,Solutions, and A Use Case

[A Trustworthy Multi-LLM Network: Challenges,Solutions, and A Use Case](http://arxiv.org/pdf/2505.03196v1)

Authors: Haoxiang Luo, Gang Sun, Yinqiu Liu, Dusit Niyato, Hongfang Yu, Mohammed Atiquzzaman, Schahram Dustdar

Large Language Models (LLMs) demonstrate strong potential across a variety of
tasks in communications and networking due to their advanced reasoning
capabilities. However, because different LLMs have different model structures
and are trained using distinct corpora and methods, they may offer varying
optimization strategies for the same network issues. Moreover, the limitations
of an individual LLM's training data, aggravated by the potential maliciousness
of its hosting device, can result in responses with low confidence or even
bias. To address these challenges, we propose a blockchain-enabled
collaborative framework that connects multiple LLMs into a Trustworthy
Multi-LLM Network (MultiLLMN). This architecture enables the cooperative
evaluation and selection of the most reliable and high-quality responses to
complex network optimization problems. Specifically, we begin by reviewing
related work and highlighting the limitations of existing LLMs in collaboration
and trust, emphasizing the need for trustworthiness in LLM-based systems. We
then introduce the workflow and design of the proposed Trustworthy MultiLLMN
framework. Given the severity of False Base Station (FBS) attacks in B5G and 6G
communication systems and the difficulty of addressing such threats through
traditional modeling techniques, we present FBS defense as a case study to
empirically validate the effectiveness of our approach. Finally, we outline
promising future research directions in this emerging area.

### 4. Advancing Remote and Continuous Cardiovascular Patient Monitoring through a Novel and Resource-efficient IoT-Driven Framework

[Advancing Remote and Continuous Cardiovascular Patient Monitoring through a Novel and Resource-efficient IoT-Driven Framework](http://arxiv.org/pdf/2505.03409v1)

Authors: Sanam Nayab, Sohail Raza Chohan, Aqsa Jameel, Syed Rehan Shah, Syed Ahsan Masud Zaidi, Aditya Nath Jha, Kamran Siddique

Cardiovascular diseases are a leading cause of fatalities worldwide, often
occurring suddenly with limited time for intervention. Current healthcare
monitoring systems for cardiac patients rely heavily on hospitalization, which
can be impractical for continuous monitoring. This paper presents a novel
IoT-based solution for remote, real-time tracking of critical cardiac metrics,
addressing the pressing need for accessible and continuous healthcare,
particularly for the aging population in Pakistan. The proposed IoT kit
measures essential parameters such as body temperature, heart rate (HR), blood
pressure (BP), oxygen saturation (SPO2), and electrocardiography (ECG).
  A key innovation of the system is its integration with a cloud-based
application, enabling constant remote monitoring and incorporating an alarm
mechanism to alert medical professionals for timely intervention, reducing the
risk of catastrophic incidents. The system was tested in a clinical environment
with 20 participants, demonstrating results closely aligned with those obtained
using standard medical devices. The findings validate the system's potential
for reliable remote monitoring, offering a significant step forward in
proactive cardiac healthcare management. This novel approach combines IoT
technology with cloud-based applications to provide a cost-effective and
efficient solution for reducing unexpected fatalities among cardiac patients.

### Robotics

### 1. Fabrication and Characterization of Additively Manufactured Stretchable Strain Sensors Towards the Shape Sensing of Continuum Robots

[Fabrication and Characterization of Additively Manufactured Stretchable Strain Sensors Towards the Shape Sensing of Continuum Robots](http://arxiv.org/pdf/2505.03087v1)

Authors: Daniel C. Moyer, Wenpeng Wang, Logan S. Karschner, Loris Fichera, Pratap M. Rao

This letter describes the manufacturing and experimental characterization of
novel stretchable strain sensors for continuum robots. The overarching goal of
this research is to provide a new solution for the shape sensing of these
devices. The sensors are fabricated via direct ink writing, an extrusion-based
additive manufacturing technique. Electrically conductive material (i.e., the
\textit{ink}) is printed into traces whose electrical resistance varies in
response to mechanical deformation. The principle of operation of stretchable
strain sensors is analogous to that of conventional strain gauges, but with a
significantly larger operational window thanks to their ability to withstand
larger strain. Among the different conductive materials considered for this
study, we opted to fabricate the sensors with a high-viscosity eutectic
Gallium-Indium ink, which in initial testing exhibited high linearity ($R^2
\approx$ 0.99), gauge factor $\approx$ 1, and negligible drift. Benefits of the
proposed sensors include (i) ease of fabrication, as they can be conveniently
printed in a matter of minutes; (ii) ease of installation, as they can simply
be glued to the outside body of a robot; (iii) ease of miniaturization, which
enables integration into millimiter-sized continuum robots.

### 2. HCOA*: Hierarchical Class-ordered A* for Navigation in Semantic Environments

[HCOA*: Hierarchical Class-ordered A* for Navigation in Semantic Environments](http://arxiv.org/pdf/2505.03128v1)

Authors: Evangelos Psomiadis, Panagiotis Tsiotras

This paper addresses the problem of robot navigation in mixed geometric and
semantic 3D environments. Given a hierarchical representation of the
environment, the objective is to navigate from a start position to a goal while
minimizing the computational cost. We introduce Hierarchical Class-ordered A*
(HCOA*), an algorithm that leverages the environmental hierarchy for efficient
path-planning in semantic graphs, significantly reducing computational effort.
We use a total order over the semantic classes and prove theoretical
performance guarantees for the algorithm. We propose two approaches for
higher-layer node classification based on the node semantics of the lowest
layer: a Graph Neural Network-based method and a Majority-Class method. We
evaluate our approach through simulations on a 3D Scene Graph (3DSG), comparing
it to the state-of-the-art and assessing its performance against our
classification approaches. Results show that HCOA* can find the optimal path
while reducing the number of expanded nodes by 25% and achieving a 16%
reduction in computational time on the uHumans2 3DSG dataset.

### 3. GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data

[GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data](http://arxiv.org/pdf/2505.03233v1)

Authors: Shengliang Deng, Mi Yan, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, Heming Cui, Zhizheng Zhang, He Wang

Embodied foundation models are gaining increasing attention for their
zero-shot generalization, scalability, and adaptability to new tasks through
few-shot post-training. However, existing models rely heavily on real-world
data, which is costly and labor-intensive to collect. Synthetic data offers a
cost-effective alternative, yet its potential remains largely underexplored. To
bridge this gap, we explore the feasibility of training Vision-Language-Action
models entirely with large-scale synthetic action data. We curate SynGrasp-1B,
a billion-frame robotic grasping dataset generated in simulation with
photorealistic rendering and extensive domain randomization. Building on this,
we present GraspVLA, a VLA model pretrained on large-scale synthetic action
data as a foundational model for grasping tasks. GraspVLA integrates
autoregressive perception tasks and flow-matching-based action generation into
a unified Chain-of-Thought process, enabling joint training on synthetic action
data and Internet semantics data. This design helps mitigate sim-to-real gaps
and facilitates the transfer of learned actions to a broader range of
Internet-covered objects, achieving open-vocabulary generalization in grasping.
Extensive evaluations across real-world and simulation benchmarks demonstrate
GraspVLA's advanced zero-shot generalizability and few-shot adaptability to
specific human preferences. We will release SynGrasp-1B dataset and pre-trained
weights to benefit the community.

### 4. RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning

[RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning](http://arxiv.org/pdf/2505.03238v1)

Authors: Liam Boyle, Nicolas Baumann, Paviththiren Sivasothilingam, Michele Magno, Luca Benini

Future robotic systems operating in real-world environments will require
on-board embodied intelligence without continuous cloud connection, balancing
capabilities with constraints on computational power and memory. This work
presents an extension of the R1-zero approach, which enables the usage of low
parameter-count Large Language Models (LLMs) in the robotic domain. The R1-Zero
approach was originally developed to enable mathematical reasoning in LLMs
using static datasets. We extend it to the robotics domain through integration
in a closed-loop Reinforcement Learning (RL) framework. This extension enhances
reasoning in Embodied Artificial Intelligence (Embodied AI) settings without
relying solely on distillation of large models through Supervised Fine-Tuning
(SFT). We show that small-scale LLMs can achieve effective reasoning
performance by learning through closed-loop interaction with their environment,
which enables tasks that previously required significantly larger models. In an
autonomous driving setting, a performance gain of 20.2%-points over the
SFT-based baseline is observed with a Qwen2.5-1.5B model. Using the proposed
training procedure, Qwen2.5-3B achieves a 63.3% control adaptability score,
surpassing the 58.5% obtained by the much larger, cloud-bound GPT-4o. These
results highlight that practical, on-board deployment of small LLMs is not only
feasible but can outperform larger models if trained through environmental
feedback, underscoring the importance of an interactive learning framework for
robotic Embodied AI, one grounded in practical experience rather than static
supervision.

### 5. Miniature multihole airflow sensor for lightweight aircraft over wide speed and angular range

[Miniature multihole airflow sensor for lightweight aircraft over wide speed and angular range](http://arxiv.org/pdf/2505.03331v1)

Authors: Lukas Stuber, Simon Jeger, Raphael Zufferey, Dario Floreano

An aircraft's airspeed, angle of attack, and angle of side slip are crucial
to its safety, especially when flying close to the stall regime. Various
solutions exist, including pitot tubes, angular vanes, and multihole pressure
probes. However, current sensors are either too heavy (>30 g) or require large
airspeeds (>20 m/s), making them unsuitable for small uncrewed aerial vehicles.
We propose a novel multihole pressure probe, integrating sensing electronics in
a single-component structure, resulting in a mechanically robust and
lightweight sensor (9 g), which we released to the public domain. Since there
is no consensus on two critical design parameters, tip shape (conical vs
spherical) and hole spacing (distance between holes), we provide a study on
measurement accuracy and noise generation using wind tunnel experiments. The
sensor is calibrated using a multivariate polynomial regression model over an
airspeed range of 3-27 m/s and an angle of attack/sideslip range of +-35{\deg},
achieving a mean absolute error of 0.44 m/s and 0.16{\deg}. Finally, we
validated the sensor in outdoor flights near the stall regime. Our probe
enabled accurate estimations of airspeed, angle of attack and sideslip during
different acrobatic manoeuvres. Due to its size and weight, this sensor will
enable safe flight for lightweight, uncrewed aerial vehicles flying at low
speeds close to the stall regime.

### 6. Effective Reinforcement Learning Control using Conservative Soft Actor-Critic

[Effective Reinforcement Learning Control using Conservative Soft Actor-Critic](http://arxiv.org/pdf/2505.03356v1)

Authors: Xinyi Yuan, Zhiwei Shang, Wenjun Huang, Yunduan Cui, Di Chen, Meixin Zhu

Reinforcement Learning (RL) has shown great potential in complex control
tasks, particularly when combined with deep neural networks within the
Actor-Critic (AC) framework. However, in practical applications, balancing
exploration, learning stability, and sample efficiency remains a significant
challenge. Traditional methods such as Soft Actor-Critic (SAC) and Proximal
Policy Optimization (PPO) address these issues by incorporating entropy or
relative entropy regularization, but often face problems of instability and low
sample efficiency. In this paper, we propose the Conservative Soft Actor-Critic
(CSAC) algorithm, which seamlessly integrates entropy and relative entropy
regularization within the AC framework. CSAC improves exploration through
entropy regularization while avoiding overly aggressive policy updates with the
use of relative entropy regularization. Evaluations on benchmark tasks and
real-world robotic simulations demonstrate that CSAC offers significant
improvements in stability and efficiency over existing methods. These findings
suggest that CSAC provides strong robustness and application potential in
control tasks under dynamic environments.

### 7. Close-Fitting Dressing Assistance Based on State Estimation of Feet and Garments with Semantic-based Visual Attention

[Close-Fitting Dressing Assistance Based on State Estimation of Feet and Garments with Semantic-based Visual Attention](http://arxiv.org/pdf/2505.03400v1)

Authors: Takuma Tsukakoshi, Tamon Miyake, Tetsuya Ogata, Yushi Wang, Takumi Akaishi, Shigeki Sugano

As the population continues to age, a shortage of caregivers is expected in
the future. Dressing assistance, in particular, is crucial for opportunities
for social participation. Especially dressing close-fitting garments, such as
socks, remains challenging due to the need for fine force adjustments to handle
the friction or snagging against the skin, while considering the shape and
position of the garment. This study introduces a method uses multi-modal
information including not only robot's camera images, joint angles, joint
torques, but also tactile forces for proper force interaction that can adapt to
individual differences in humans. Furthermore, by introducing semantic
information based on object concepts, rather than relying solely on RGB data,
it can be generalized to unseen feet and background. In addition, incorporating
depth data helps infer relative spatial relationship between the sock and the
foot. To validate its capability for semantic object conceptualization and to
ensure safety, training data were collected using a mannequin, and subsequent
experiments were conducted with human subjects. In experiments, the robot
successfully adapted to previously unseen human feet and was able to put socks
on 10 participants, achieving a higher success rate than Action Chunking with
Transformer and Diffusion Policy. These results demonstrate that the proposed
model can estimate the state of both the garment and the foot, enabling precise
dressing assistance for close-fitting garments.

### 8. AquaticVision: Benchmarking Visual SLAM in Underwater Environment with Events and Frames

[AquaticVision: Benchmarking Visual SLAM in Underwater Environment with Events and Frames](http://arxiv.org/pdf/2505.03448v1)

Authors: Yifan Peng, Yuze Hong, Ziyang Hong, Apple Pui-Yi Chui, Junfeng Wu

Many underwater applications, such as offshore asset inspections, rely on
visual inspection and detailed 3D reconstruction. Recent advancements in
underwater visual SLAM systems for aquatic environments have garnered
significant attention in marine robotics research. However, existing underwater
visual SLAM datasets often lack groundtruth trajectory data, making it
difficult to objectively compare the performance of different SLAM algorithms
based solely on qualitative results or COLMAP reconstruction. In this paper, we
present a novel underwater dataset that includes ground truth trajectory data
obtained using a motion capture system. Additionally, for the first time, we
release visual data that includes both events and frames for benchmarking
underwater visual positioning. By providing event camera data, we aim to
facilitate the development of more robust and advanced underwater visual SLAM
algorithms. The use of event cameras can help mitigate challenges posed by
extremely low light or hazy underwater conditions. The webpage of our dataset
is https://sites.google.com/view/aquaticvision-lias.

### 9. LogisticsVLN: Vision-Language Navigation For Low-Altitude Terminal Delivery Based on Agentic UAVs

[LogisticsVLN: Vision-Language Navigation For Low-Altitude Terminal Delivery Based on Agentic UAVs](http://arxiv.org/pdf/2505.03460v1)

Authors: Xinyuan Zhang, Yonglin Tian, Fei Lin, Yue Liu, Jing Ma, Kornélia Sára Szatmáry, Fei-Yue Wang

The growing demand for intelligent logistics, particularly fine-grained
terminal delivery, underscores the need for autonomous UAV (Unmanned Aerial
Vehicle)-based delivery systems. However, most existing last-mile delivery
studies rely on ground robots, while current UAV-based Vision-Language
Navigation (VLN) tasks primarily focus on coarse-grained, long-range goals,
making them unsuitable for precise terminal delivery. To bridge this gap, we
propose LogisticsVLN, a scalable aerial delivery system built on multimodal
large language models (MLLMs) for autonomous terminal delivery. LogisticsVLN
integrates lightweight Large Language Models (LLMs) and Visual-Language Models
(VLMs) in a modular pipeline for request understanding, floor localization,
object detection, and action-decision making. To support research and
evaluation in this new setting, we construct the Vision-Language Delivery (VLD)
dataset within the CARLA simulator. Experimental results on the VLD dataset
showcase the feasibility of the LogisticsVLN system. In addition, we conduct
subtask-level evaluations of each module of our system, offering valuable
insights for improving the robustness and real-world deployment of foundation
model-based vision-language delivery systems.

### 10. Task Reconstruction and Extrapolation for $π_0$ using Text Latent

[Task Reconstruction and Extrapolation for $π_0$ using Text Latent](http://arxiv.org/pdf/2505.03500v1)

Authors: Quanyi Li

Vision-language-action models (VLAs) often achieve high performance on
demonstrated tasks but struggle significantly when required to extrapolate,
combining skills learned from different tasks in novel ways. For instance, VLAs
might successfully put the cream cheese in the bowl and put the bowl on top of
the cabinet, yet still fail to put the cream cheese on top of the cabinet. In
this work, we demonstrate that behaviors from distinct tasks can be effectively
recombined by manipulating the VLA's internal representations at inference
time. Concretely, we identify the text latent by averaging the text tokens'
hidden states across all demonstrated trajectories for a specific base task.
For executing an extrapolated task, we can temporally interpolate the text
latent of the two base tasks and add it back to the text hidden states, so
sub-behaviors from the two tasks will be activated sequentially. We evaluate
this approach using the newly created libero-ood benchmark, featuring 20 tasks
extrapolated from standard LIBERO suites. The results on libero-ood show that
all SOTA VLAs achieve < 15% success rate, while $\pi0$ with text latent
interpolation reaches an 83% success rate. Further qualitative analysis reveals
a tendency for VLAs to exhibit spatial overfitting, mapping object names to
demonstrated locations rather than achieving genuine object and goal
understanding. Additionally, we find that decoding the text latent yields
human-unreadable prompts that can nevertheless instruct the VLA to achieve a
70% success rate on standard LIBERO suites, enabling private instruction or
backdoor attacks.

### Software Engineering

### 1. An Empirical Study on the Impact of Gender Diversity on Code Quality in AI Systems

[An Empirical Study on the Impact of Gender Diversity on Code Quality in AI Systems](http://arxiv.org/pdf/2505.03082v1)

Authors: Shamse Tasnim Cynthia, Banani Roy

The rapid advancement of AI systems necessitates high-quality, sustainable
code to ensure reliability and mitigate risks such as bias and technical debt.
However, the underrepresentation of women in software engineering raises
concerns about homogeneity in AI development. Studying gender diversity in AI
systems is crucial, as diverse perspectives are essential for improving system
robustness, reducing bias, and enhancing overall code quality. While prior
research has demonstrated the benefits of diversity in general software teams,
its specific impact on the code quality of AI systems remains unexplored. This
study addresses this gap by examining how gender diversity within AI teams
influences project popularity, code quality, and individual contributions. Our
study makes three key contributions. First, we analyzed the relationship
between team diversity and repository popularity, revealing that diverse AI
repositories not only differ significantly from non-diverse ones but also
achieve higher popularity and greater community engagement. Second, we explored
the effect of diversity on the overall code quality of AI systems and found
that diverse repositories tend to have superior code quality compared to
non-diverse ones. Finally, our analysis of individual contributions revealed
that although female contributors contribute to a smaller proportion of the
total code, their contributions demonstrate consistently higher quality than
those of their male counterparts. These findings highlight the need to remove
barriers to female participation in AI development, as greater diversity can
improve the overall quality of AI systems.

### 2. ATRAF-driven IMRaD Methodology: Tradeoff and Risk Analysis of Software Architectures Across Abstraction Levels

[ATRAF-driven IMRaD Methodology: Tradeoff and Risk Analysis of Software Architectures Across Abstraction Levels](http://arxiv.org/pdf/2505.03624v1)

Authors: Amine Ben Hassouna

Software architecture research relies on key architectural artifacts --
Software Architectures, Reference Architectures, and Architectural Frameworks
-- that underpin the design and analysis of complex systems. Evaluating these
artifacts is essential to assess tradeoffs and risks affecting quality
attributes such as performance, modifiability, and security. Although
methodologies like the Architecture Tradeoff Analysis Method (ATAM) support
software architecture evaluation, their industrial focus misaligns with the
IMRaD (Introduction, Methods, Results, Discussion) format prevalent in academic
research, impeding transparency and reproducibility. Our prior work introduced
the Architecture Tradeoff and Risk Analysis Framework (ATRAF), extending ATAM
through three methods -- ATRAM, RATRAM, and AFTRAM, addressing all abstraction
levels, using a unified, iterative four-phase spiral model. These phases --
Scenario and Requirements Gathering, Architectural Views and Scenario
Realization, Attribute-Specific Analyses, and Sensitivity, Tradeoff, and Risk
Analysis -- ensure traceability and coherence. This paper presents the
ATRAF-driven IMRaD Methodology, a concise method to align ATRAF's phases with
IMRaD sections. This methodology enhances the rigor, transparency, and
accessibility of software architecture research, enabling systematic reporting
of complex evaluations.

### 3. Moral Testing of Autonomous Driving Systems

[Moral Testing of Autonomous Driving Systems](http://arxiv.org/pdf/2505.03683v1)

Authors: Wenbing Tang, Mingfei Cheng, Yuan Zhou, Yang Liu

Autonomous Driving System (ADS) testing plays a crucial role in their
development, with the current focus primarily on functional and safety testing.
However, evaluating the non-functional morality of ADSs, particularly their
decision-making capabilities in unavoidable collision scenarios, is equally
important to ensure the systems' trustworthiness and public acceptance.
Unfortunately, testing ADS morality is nearly impossible due to the absence of
universal moral principles. To address this challenge, this paper first
extracts a set of moral meta-principles derived from existing moral experiments
and well-established social science theories, aiming to capture widely
recognized and common-sense moral values for ADSs. These meta-principles are
then formalized as quantitative moral metamorphic relations, which act as the
test oracle. Furthermore, we propose a metamorphic testing framework to
systematically identify potential moral issues. Finally, we illustrate the
implementation of the framework and present typical violation cases using the
VIRES VTD simulator and its built-in ADS.

### 4. Improving the Reproducibility of Deep Learning Software: An Initial Investigation through a Case Study Analysis

[Improving the Reproducibility of Deep Learning Software: An Initial Investigation through a Case Study Analysis](http://arxiv.org/pdf/2505.03165v1)

Authors: Nikita Ravi, Abhinav Goel, James C. Davis, George K. Thiruvathukal

The field of deep learning has witnessed significant breakthroughs, spanning
various applications, and fundamentally transforming current software
capabilities. However, alongside these advancements, there have been increasing
concerns about reproducing the results of these deep learning methods. This is
significant because reproducibility is the foundation of reliability and
validity in software development, particularly in the rapidly evolving domain
of deep learning. The difficulty of reproducibility may arise due to several
reasons, including having differences from the original execution environment,
incompatible software libraries, proprietary data and source code, lack of
transparency, and the stochastic nature in some software. A study conducted by
the Nature journal reveals that more than 70% of researchers failed to
reproduce other researchers experiments and over 50% failed to reproduce their
own experiments. Irreproducibility of deep learning poses significant
challenges for researchers and practitioners. To address these concerns, this
paper presents a systematic approach at analyzing and improving the
reproducibility of deep learning models by demonstrating these guidelines using
a case study. We illustrate the patterns and anti-patterns involved with these
guidelines for improving the reproducibility of deep learning models. These
guidelines encompass establishing a methodology to replicate the original
software environment, implementing end-to-end training and testing algorithms,
disclosing architectural designs, and enhancing transparency in data processing
and training pipelines. We also conduct a sensitivity analysis to understand
the model performance across diverse conditions. By implementing these
strategies, we aim to bridge the gap between research and practice, so that
innovations in deep learning can be effectively reproduced and deployed within
software.

### 5. DocSpiral: A Platform for Integrated Assistive Document Annotation through Human-in-the-Spiral

[DocSpiral: A Platform for Integrated Assistive Document Annotation through Human-in-the-Spiral](http://arxiv.org/pdf/2505.03214v1)

Authors: Qiang Sun, Sirui Li, Tingting Bi, Du Huynh, Mark Reynolds, Yuanyi Luo, Wei Liu

Acquiring structured data from domain-specific, image-based documents such as
scanned reports is crucial for many downstream tasks but remains challenging
due to document variability. Many of these documents exist as images rather
than as machine-readable text, which requires human annotation to train
automated extraction systems. We present DocSpiral, the first
Human-in-the-Spiral assistive document annotation platform, designed to address
the challenge of extracting structured information from domain-specific,
image-based document collections. Our spiral design establishes an iterative
cycle in which human annotations train models that progressively require less
manual intervention. DocSpiral integrates document format normalization,
comprehensive annotation interfaces, evaluation metrics dashboard, and API
endpoints for the development of AI / ML models into a unified workflow.
Experiments demonstrate that our framework reduces annotation time by at least
41\% while showing consistent performance gains across three iterations during
model training. By making this annotation platform freely accessible, we aim to
lower barriers to AI/ML models development in document processing, facilitating
the adoption of large language models in image-based, document-intensive fields
such as geoscience and healthcare. The system is freely available at:
https://app.ai4wa.com. The demonstration video is available:
https://app.ai4wa.com/docs/docspiral/demo.

### 6. Synthline: A Product Line Approach for Synthetic Requirements Engineering Data Generation using Large Language Models

[Synthline: A Product Line Approach for Synthetic Requirements Engineering Data Generation using Large Language Models](http://arxiv.org/pdf/2505.03265v1)

Authors: Abdelkarim El-Hajjami, Camille Salinesi

While modern Requirements Engineering (RE) heavily relies on natural language
processing and Machine Learning (ML) techniques, their effectiveness is limited
by the scarcity of high-quality datasets. This paper introduces Synthline, a
Product Line (PL) approach that leverages Large Language Models to
systematically generate synthetic RE data for classification-based use cases.
Through an empirical evaluation conducted in the context of using ML for the
identification of requirements specification defects, we investigated both the
diversity of the generated data and its utility for training downstream models.
Our analysis reveals that while synthetic datasets exhibit less diversity than
real data, they are good enough to serve as viable training resources.
Moreover, our evaluation shows that combining synthetic and real data leads to
substantial performance improvements. Specifically, hybrid approaches achieve
up to 85% improvement in precision and a 2x increase in recall compared to
models trained exclusively on real data. These findings demonstrate the
potential of PL-based synthetic data generation to address data scarcity in RE.
We make both our implementation and generated datasets publicly available to
support reproducibility and advancement in the field.

### 7. RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation

[RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation](http://arxiv.org/pdf/2505.03275v1)

Authors: Tiantian Gan, Qiyao Sun

Large language models (LLMs) struggle to effectively utilize a growing number
of external tools, such as those defined by the Model Context Protocol
(MCP)\cite{IntroducingMCP}, due to prompt bloat and selection complexity. We
introduce RAG-MCP, a Retrieval-Augmented Generation framework that overcomes
this challenge by offloading tool discovery. RAG-MCP uses semantic retrieval to
identify the most relevant MCP(s) for a given query from an external index
before engaging the LLM. Only the selected tool descriptions are passed to the
model, drastically reducing prompt size and simplifying decision-making.
Experiments, including an MCP stress test, demonstrate RAG-MCP significantly
cuts prompt tokens (e.g., by over 50%) and more than triples tool selection
accuracy (43.13% vs 13.62% baseline) on benchmark tasks. RAG-MCP enables
scalable and accurate tool integration for LLMs.

### 8. Qimax: Efficient quantum simulation via GPU-accelerated extended stabilizer formalism

[Qimax: Efficient quantum simulation via GPU-accelerated extended stabilizer formalism](http://arxiv.org/pdf/2505.03307v1)

Authors: Vu Tuan Hai, Bui Cao Doanh, Le Vu Trung Duong, Pham Hoai Luan, Yasuhiko Nakashima

Simulating Clifford and near-Clifford circuits using the extended stabilizer
formalism has become increasingly popular, particularly in quantum error
correction. Compared to the state-vector approach, the extended stabilizer
formalism can solve the same problems with fewer computational resources, as it
operates on stabilizers rather than full state vectors. Most existing studies
on near-Clifford circuits focus on balancing the trade-off between the number
of ancilla qubits and simulation accuracy, often overlooking performance
considerations. Furthermore, in the presence of high-rank stabilizers,
performance is limited by the sequential property of the stabilizer formalism.
In this work, we introduce a parallelized version of the extended stabilizer
formalism, enabling efficient execution on multi-core devices such as GPU.
Experimental results demonstrate that, in certain scenarios, our Python-based
implementation outperforms state-of-the-art simulators such as Qiskit and
Pennylane.

### 9. Assessing and Enhancing the Robustness of LLM-based Multi-Agent Systems Through Chaos Engineering

[Assessing and Enhancing the Robustness of LLM-based Multi-Agent Systems Through Chaos Engineering](http://arxiv.org/pdf/2505.03096v1)

Authors: Joshua Owotogbe

This study explores the application of chaos engineering to enhance the
robustness of Large Language Model-Based Multi-Agent Systems (LLM-MAS) in
production-like environments under real-world conditions. LLM-MAS can
potentially improve a wide range of tasks, from answering questions and
generating content to automating customer support and improving decision-making
processes. However, LLM-MAS in production or preproduction environments can be
vulnerable to emergent errors or disruptions, such as hallucinations, agent
failures, and agent communication failures. This study proposes a chaos
engineering framework to proactively identify such vulnerabilities in LLM-MAS,
assess and build resilience against them, and ensure reliable performance in
critical applications.

### 10. Capability-Driven Skill Generation with LLMs: A RAG-Based Approach for Reusing Existing Libraries and Interfaces

[Capability-Driven Skill Generation with LLMs: A RAG-Based Approach for Reusing Existing Libraries and Interfaces](http://arxiv.org/pdf/2505.03295v1)

Authors: Luis Miguel Vieira da Silva, Aljosha Köcher, Nicolas König, Felix Gehlhoff, Alexander Fay

Modern automation systems increasingly rely on modular architectures, with
capabilities and skills as one solution approach. Capabilities define the
functions of resources in a machine-readable form and skills provide the
concrete implementations that realize those capabilities. However, the
development of a skill implementation conforming to a corresponding capability
remains a time-consuming and challenging task. In this paper, we present a
method that treats capabilities as contracts for skill implementations and
leverages large language models to generate executable code based on natural
language user input. A key feature of our approach is the integration of
existing software libraries and interface technologies, enabling the generation
of skill implementations across different target languages. We introduce a
framework that allows users to incorporate their own libraries and resource
interfaces into the code generation process through a retrieval-augmented
generation architecture. The proposed method is evaluated using an autonomous
mobile robot controlled via Python and ROS 2, demonstrating the feasibility and
flexibility of the approach.

### Social and Information Networks

### 1. Coevolution of Actions and Opinions in Networks of Coordinating and Anti-Coordinating Agents

[Coevolution of Actions and Opinions in Networks of Coordinating and Anti-Coordinating Agents](http://arxiv.org/pdf/2505.03078v1)

Authors: Hong Liang, Mengbin Ye, Lorenzo Zino, Weiguo Xia

In this paper, we investigate the dynamics of coordinating and
anti-coordinating agents in a coevolutionary model for actions and opinions. In
the model, the individuals of a population interact on a two-layer network,
sharing their opinions and observing others' action, while revising their own
opinions and actions according to a game-theoretic mechanism, grounded in the
social psychology literature. First, we consider the scenario of coordinating
agents, where convergence to a Nash equilibrium (NE) is guaranteed. We identify
conditions for reaching consensus configurations and establish regions of
attraction for these equilibria. Second, we study networks of anti-coordinating
agents. In this second scenario, we prove that all trajectories converge to a
NE by leveraging potential game theory. Then, we establish analytical
conditions on the network structure and model parameters to guarantee the
existence of consensus and polarized equilibria, characterizing their regions
of attraction.

### 2. Troika algorithm: approximate optimization for accurate clique partitioning and clustering of weighted networks

[Troika algorithm: approximate optimization for accurate clique partitioning and clustering of weighted networks](http://arxiv.org/pdf/2505.03573v1)

Authors: Samin Aref, Boris Ng

Clique partitioning is a fundamental network clustering task, with
applications in a wide range of computational sciences. It involves identifying
an optimal partition of the nodes for a real-valued weighted graph according to
the edge weights. An optimal partition is one that maximizes the sum of
within-cluster edge weights over all possible node partitions. This paper
introduces a novel approximation algorithm named Troika to solve this NP-hard
problem in small to mid-sized networks for instances of theoretical and
practical relevance. Troika uses a branch-and-cut scheme for branching on node
triples to find a partition that is within a user-specified optimality gap
tolerance. Troika offers advantages over alternative methods like integer
programming solvers and heuristics for clique partitioning. Unlike existing
heuristics, Troika returns solutions within a guaranteed proximity to global
optimality. And our results indicate that Troika is faster than using the
state-of-the-art integer programming solver Gurobi for most benchmark
instances. Besides its advantages for solving the clique partitioning problem,
we demonstrate the applications of Troika in community detection and portfolio
analysis. Troika returns partitions with higher proximity to optimal compared
to eight modularity-based community detection algorithms. When used on networks
of correlations among stocks, Troika reveals the dynamic changes in the
structure of portfolio networks including downturns from the 2008 financial
crisis and the reaction to the COVID-19 pandemic. Our comprehensive results
based on benchmarks from the literature and new real and random networks point
to Troika as a reliable and accurate method for solving clique partitioning
instances with up to 5000 edges on standard hardware.

### Systems and Control

### 1. Non-linear dynamics of multibody systems: a system-based approach

[Non-linear dynamics of multibody systems: a system-based approach](http://arxiv.org/pdf/2505.03248v1)

Authors: Daniel Alazard, Francesco Sanfedino, Ervan Kassarian

This paper presents causal block-diagram models to represent the equations of
motion of multi-body systems in a very compact and simple closed form. Both the
forward dynamics (from the forces and torques imposed at the various
degrees-of-freedom to the motions of these degrees-of-freedom) or the inverse
dynamics (from the motions imposed at the degrees-of-freedom to the resulting
forces and torques) can be considered and described by a block diagram model.
This work extends the Two-Input Two-Output Port (TITOP) theory by including all
non-linear terms and uniform or gravitational acceleration fields. Connection
among different blocks is possible through the definition of the motion vector.
The model of a system composed of a floating base, rigid bodies, revolute and
prismatic joints, working under gravity is developed to illustrate the
methodology. The proposed model is validated by simulation and cross-checking
with a model built using an alternative modeling tool on a scenario where the
nonlinear terms are determining.

### 2. Power Loss and Temperature Distribution in Coil of PFC Inductor with Air Gap for Multimode Operation

[Power Loss and Temperature Distribution in Coil of PFC Inductor with Air Gap for Multimode Operation](http://arxiv.org/pdf/2505.03489v1)

Authors: Rafal Kasikowski

Power converters inherently display non-linear load characteristics,
resulting in a high level of mains harmonics, and hence the necessity of
implementing Power Factor Correction (PFC). Active PFC circuitry typically
comprises an inductor and a power switch to control and alter the input current
so that it matches, in shape and phase, the input voltage. This modelling of
the waveforms can be performed by means of distinct conduction modes of the PFC
inductor. The digital controller implemented in the constructed and
investigated boost-type PFC converter can be programmed to operate in
discontinuous conduction mode (DCM), continuous conduction mode (CCM), or a
combination of the two. The individual modes of operation, via distinct PFC
inductor current waveforms, impact the overall efficiency of power conversion
and, by extension, temperature distribution in the magnetic component. This
paper investigates how the examined conduction modes bear on distinct
power-loss mechanisms present in the PFC inductor, including high-frequency
eddy-current-generating phenomena, and the fringing effect in particular. As
demonstrated herein, the DCM operation, for the set output power level,
exhibits exacerbated power dissipation in the winding of the inductor due to
the somewhat increased RSM value of the current and the intensified fringing
magnetic flux at an air gap. The latter assertion will undergo further, more
quantitatively focused research. Finally, the construction of the coil was
optimised to reduce power loss by diminishing eddy-current mechanisms.

### 3. Sequentially learning regions of attraction from data

[Sequentially learning regions of attraction from data](http://arxiv.org/pdf/2505.03493v1)

Authors: Oumayma Khattabi, Matteo Tacchi-Bénard, Sorin Olaru

The paper is dedicated to data-driven analysis of dynamical systems. It deals
with certifying the basin of attraction of a stable equilibrium for an unknown
dynamical system. It is supposed that point-wise evaluation of the right-hand
side of the ordinary differential equation governing the system is available
for a set of points in the state space. Technically, a Piecewise Affine
Lyapunov function will be constructed iteratively using an optimisation-based
technique for the effective validation of the certificates. As a main
contribution, whenever those certificates are violated locally, a refinement of
the domain and the associated tessellation is produced, thus leading to an
improvement in the description of the domain of attraction.

### 4. Event-Triggered GAT-LSTM Framework for Attack Detection in Heating, Ventilation, and Air Conditioning Systems

[Event-Triggered GAT-LSTM Framework for Attack Detection in Heating, Ventilation, and Air Conditioning Systems](http://arxiv.org/pdf/2505.03559v1)

Authors: Zhenan Feng, Ehsan Nekouei

Heating, Ventilation, and Air Conditioning (HVAC) systems are essential for
maintaining indoor environmental quality, but their interconnected nature and
reliance on sensor networks make them vulnerable to cyber-physical attacks.
Such attacks can interrupt system operations and risk leaking sensitive
personal information through measurement data. In this paper, we propose a
novel attack detection framework for HVAC systems, integrating an
Event-Triggering Unit (ETU) for local monitoring and a cloud-based
classification system using the Graph Attention Network (GAT) and the Long
Short-Term Memory (LSTM) network. The ETU performs a binary classification to
identify potential anomalies and selectively triggers encrypted data
transmission to the cloud, significantly reducing communication cost. The
cloud-side GAT module models the spatial relationships among HVAC components,
while the LSTM module captures temporal dependencies across encrypted state
sequences to classify the attack type. Our approach is evaluated on datasets
that simulate diverse attack scenarios. Compared to GAT-only (94.2% accuracy)
and LSTM-only (91.5%) ablations, our full GAT-LSTM model achieves 98.8% overall
detection accuracy and reduces data transmission to 15%. These results
demonstrate that the proposed framework achieves high detection accuracy while
preserving data privacy by using the spatial-temporal characteristics of HVAC
systems and minimizing transmission costs through event-triggered
communication.

### 5. Artificial Potential Field and Sliding Mode Control for Spacecraft Attitude Maneuver with Actuation and Pointing Constraints

[Artificial Potential Field and Sliding Mode Control for Spacecraft Attitude Maneuver with Actuation and Pointing Constraints](http://arxiv.org/pdf/2505.03594v1)

Authors: Mauro Mancini, Dario Ruggiero

This study investigates the combination of guidance and control strategies
for rigid spacecraft attitude reorientation, while dealing with forbidden
pointing constraints, actuator limitations, and system uncertainties. These
constraints arise due to the presence of bright objects in space that may
damage sensitive payloads onboard the spacecraft, and the risk that actuator
saturations may compromise closed-loop system stability. Furthermore,
spacecraft attitude dynamics are typically affected by parametric
uncertainties, external disturbances, and system nonlinearities, which cannot
be neglected. In this article, the problem of spacecraft reorientation under
pointing and actuation constraints is addressed using a strategy that combines
Artificial Potential Field (APF) and Sliding Mode Control (SMC). A rigorous
Lyapunov-based analysis yields closed-form expressions for APF/SMC gains,
providing explicit mathematical formulas for gain values without the need for
iterative computations. These expressions account for angular velocity and
control torque limitations, external disturbances, and inertia uncertainties.
The robustness of the proposed control strategy is demonstrated through Monte
Carlo simulations using a high-fidelity attitude dynamics simulator.
Additionally, mu-analysis is employed to assess local stability properties and
quantify robustness margins. The results confirm the practical feasibility of
the proposed method in real-world space scenarios, highlighting its
effectiveness in uncertain and constrained environments.

### 6. Dynamic load balancing for cloud systems under heterogeneous setup delays

[Dynamic load balancing for cloud systems under heterogeneous setup delays](http://arxiv.org/pdf/2505.03596v1)

Authors: Fernando Paganini, Diego Goldsztajn

We consider a distributed cloud service deployed at a set of distinct server
pools. Arriving jobs are classified into heterogeneous types, in accordance
with their setup times which are differentiated at each of the pools. A
dispatcher for each job type controls the balance of load between pools, based
on decentralized feedback. The system of rates and queues is modeled by a fluid
differential equation system, and analyzed via convex optimization. A first,
myopic policy is proposed, based on task delay-to-service. Under a simplified
dynamic fluid queue model, we prove global convergence to an equilibrium point
which minimizes the mean setup time; however queueing delays are incurred with
this method. A second proposal is then developed based on proximal
optimization, which explicitly models the setup queue and is proved to reach an
optimal equilibrium, devoid of queueing delay. Results are demonstrated through
a simulation example.

### 7. Backstepping Reach-avoid Controller Synthesis for Multi-input Multi-output Systems with Mixed Relative Degrees

[Backstepping Reach-avoid Controller Synthesis for Multi-input Multi-output Systems with Mixed Relative Degrees](http://arxiv.org/pdf/2505.03612v1)

Authors: Jianqiang Ding, Dingran Yuan, Shankar A. Deka

Designing controllers with provable formal guarantees has become an urgent
requirement for cyber-physical systems in safety-critical scenarios. Beyond
addressing scalability in high-dimensional implementations, controller
synthesis methodologies separating safety and reachability objectives may risk
optimization infeasibility due to conflicting constraints, thereby
significantly undermining their applicability in practical applications. In
this paper, by leveraging feedback linearization and backstepping techniques,
we present a novel framework for constructing provable reach-avoid formal
certificates tailored to multi-input multi-output systems. Based on this, we
developed a systematic synthesis approach for controllers with reach-avoid
guarantees, which ensures that the outputs of the system eventually enter the
predefined target set while staying within the required safe set. Finally, we
demonstrate the effectiveness of our method through simulations.

### 8. Optimal Droop Control Strategy for Coordinated Voltage Regulation and Power Sharing in Hybrid AC-MTDC Systems

[Optimal Droop Control Strategy for Coordinated Voltage Regulation and Power Sharing in Hybrid AC-MTDC Systems](http://arxiv.org/pdf/2505.03651v1)

Authors: Hongjin Du, Tuanku Badzlin Hashfi, Rashmi Prasad, Pedro P. Vergara, Peter Palensky, Aleksandra Lekić

With the growing integration of modular multilevel converters (MMCs) in
Multi-Terminal Direct Current (MTDC) transmission systems, there is an
increasing need for control strategies that ensure both economic efficiency and
robust dynamic performance. This paper presents an enhanced Optimal Power Flow
(OPF) framework for hybrid AC-MTDC systems, integrating a novel droop control
strategy that coordinates DC voltage and AC frequency regulation. By embedding
frequency control loops into the MMCs, the method enables system-wide
coordination, enhancing power sharing and improving system resilience under
disturbances. The proposed strategy dynamically adjusts converter operating
points to minimize generation costs and DC voltage deviations, thus balancing
economic objectives with system stability. A modified Nordic test system
integrated with a four-terminal MTDC grid is used to validate the approach.
Optimization is performed using Julia, while the system's dynamic performance
is evaluated through electromagnetic transient simulations with the EMTP
software. Case studies across multiple scenarios demonstrate that the proposed
method consistently achieves lower generation costs than active power control
and adaptive droop control strategy while maintaining stable control
characteristics. The results highlight the method's capability to deliver
cost-effective operation without compromising performance, offering a promising
solution for the coordinated control of future hybrid AC-DC transmission
networks.

### 9. Toward a Harmonized Approach -- Requirement-based Structuring of a Safety Assurance Argumentation for Automated Vehicles

[Toward a Harmonized Approach -- Requirement-based Structuring of a Safety Assurance Argumentation for Automated Vehicles](http://arxiv.org/pdf/2505.03709v1)

Authors: M. Loba, N. F. Salem, M. Nolte, A. Dotzler, M. Maurer

Despite increasing testing operation on public roads, media reports on
incidents show that safety issues remain to this day. One major cause factoring
into this circumstance is high development uncertainty that manufacturers face
when deploying these systems in an open context. In particular, one challenge
is establishing a valid argument at design time that the vehicle will exhibit
reasonable residual risk when operating in its intended operational design
domain. Regulations, such as the European Implementing Regulation 2022/1426,
require manufacturers to provide a safety assurance argumentation for
SAE-Level-4 automated vehicles. While there is extensive literature on
assurance cases for safety-critical systems, the domain of automated driving
lacks explicit requirements regarding the creation of safety assurance
argumentations. In this paper, we aim to narrow this gap by elaborating a
requirement-based approach. We derive structural requirements for an
argumentation from literature and supplement these with requirements derived
from stakeholder concerns. We implement the requirements, yielding a proposal
for an overall argumentation structure. The resulting "safety arguments" argue
over four topic complexes: The developed product, the underlying process
including its conformance/compliance to standards/laws, as well as the
argumentations' context and soundness. Finally, we instantiate this structure
with respect to domain-specific needs and principles.

### 10. Coevolution of Actions and Opinions in Networks of Coordinating and Anti-Coordinating Agents

[Coevolution of Actions and Opinions in Networks of Coordinating and Anti-Coordinating Agents](http://arxiv.org/pdf/2505.03078v1)

Authors: Hong Liang, Mengbin Ye, Lorenzo Zino, Weiguo Xia

In this paper, we investigate the dynamics of coordinating and
anti-coordinating agents in a coevolutionary model for actions and opinions. In
the model, the individuals of a population interact on a two-layer network,
sharing their opinions and observing others' action, while revising their own
opinions and actions according to a game-theoretic mechanism, grounded in the
social psychology literature. First, we consider the scenario of coordinating
agents, where convergence to a Nash equilibrium (NE) is guaranteed. We identify
conditions for reaching consensus configurations and establish regions of
attraction for these equilibria. Second, we study networks of anti-coordinating
agents. In this second scenario, we prove that all trajectories converge to a
NE by leveraging potential game theory. Then, we establish analytical
conditions on the network structure and model parameters to guarantee the
existence of consensus and polarized equilibria, characterizing their regions
of attraction.

### Machine Learning (Statistics Category)

### 1. Weighted Average Gradients for Feature Attribution

[Weighted Average Gradients for Feature Attribution](http://arxiv.org/pdf/2505.03201v1)

Authors: Kien Tran Duc Tuan, Tam Nguyen Trong, Son Nguyen Hoang, Khoat Than, Anh Nguyen Duc

In explainable AI, Integrated Gradients (IG) is a widely adopted technique
for assessing the significance of feature attributes of the input on model
outputs by evaluating contributions from a baseline input to the current input.
The choice of the baseline input significantly influences the resulting
explanation. While the traditional Expected Gradients (EG) method assumes
baselines can be uniformly sampled and averaged with equal weights, this study
argues that baselines should not be treated equivalently. We introduce Weighted
Average Gradients (WG), a novel approach that unsupervisedly evaluates baseline
suitability and incorporates a strategy for selecting effective baselines.
Theoretical analysis demonstrates that WG satisfies essential explanation
method criteria and offers greater stability than prior approaches.
Experimental results further confirm that WG outperforms EG across diverse
scenarios, achieving an improvement of 10-35\% on main metrics. Moreover, by
evaluating baselines, our method can filter a subset of effective baselines for
each input to calculate explanations, maintaining high accuracy while reducing
computational cost. The code is available at:
https://github.com/Tamnt240904/weighted_baseline.

### 2. Bayesian full waveform inversion with sequential surrogate model refinement

[Bayesian full waveform inversion with sequential surrogate model refinement](http://arxiv.org/pdf/2505.03246v1)

Authors: Giovanni Angelo Meles, Stefano Marelli, Niklas Linde

Bayesian formulations of inverse problems are attractive for their ability to
incorporate prior knowledge and update probabilistic models as new data become
available. Markov chain Monte Carlo (MCMC) methods sample posterior probability
density functions (pdfs) but require accurate prior models and many likelihood
evaluations. Dimensionality-reduction methods, such as principal component
analysis (PCA), can help define the prior and train surrogate models that
efficiently approximate costly forward solvers. However, for problems like full
waveform inversion, the complex input/output relations often cannot be captured
well by surrogate models trained only on prior samples, leading to biased
results. Including samples from high-posterior-probability regions can improve
accuracy, but these regions are hard to identify in advance. We propose an
iterative method that progressively refines the surrogate model. Starting with
low-frequency data, we train an initial surrogate and perform an MCMC
inversion. The resulting posterior samples are then used to retrain the
surrogate, allowing us to expand the frequency bandwidth in the next inversion
step. Repeating this process reduces model errors and improves the surrogate's
accuracy over the relevant input domain. Ultimately, we obtain a highly
accurate surrogate across the full bandwidth, enabling a final MCMC inversion.
Numerical results from 2D synthetic crosshole Ground Penetrating Radar (GPR)
examples show that our method outperforms ray-based approaches and those
relying solely on prior sampling. The overall computational cost is reduced by
about two orders of magnitude compared to full finite-difference time-domain
modeling.

### 3. Decision Making under Model Misspecification: DRO with Robust Bayesian Ambiguity Sets

[Decision Making under Model Misspecification: DRO with Robust Bayesian Ambiguity Sets](http://arxiv.org/pdf/2505.03585v1)

Authors: Charita Dellaporta, Patrick O'Hara, Theodoros Damoulas

Distributionally Robust Optimisation (DRO) protects risk-averse
decision-makers by considering the worst-case risk within an ambiguity set of
distributions based on the empirical distribution or a model. To further guard
against finite, noisy data, model-based approaches admit Bayesian formulations
that propagate uncertainty from the posterior to the decision-making problem.
However, when the model is misspecified, the decision maker must stretch the
ambiguity set to contain the data-generating process (DGP), leading to overly
conservative decisions. We address this challenge by introducing DRO with
Robust, to model misspecification, Bayesian Ambiguity Sets (DRO-RoBAS). These
are Maximum Mean Discrepancy ambiguity sets centred at a robust posterior
predictive distribution that incorporates beliefs about the DGP. We show that
the resulting optimisation problem obtains a dual formulation in the
Reproducing Kernel Hilbert Space and we give probabilistic guarantees on the
tolerance level of the ambiguity set. Our method outperforms other Bayesian and
empirical DRO approaches in out-of-sample performance on the Newsvendor and
Portfolio problems with various cases of model misspecification.

### 4. Multi-modal cascade feature transfer for polymer property prediction

[Multi-modal cascade feature transfer for polymer property prediction](http://arxiv.org/pdf/2505.03704v1)

Authors: Kiichi Obuchi, Yuta Yahagi, Kiyohiko Toyama, Shukichi Tanaka, Kota Matsui

In this paper, we propose a novel transfer learning approach called
multi-modal cascade model with feature transfer for polymer property
prediction.Polymers are characterized by a composite of data in several
different formats, including molecular descriptors and additive information as
well as chemical structures. However, in conventional approaches, prediction
models were often constructed using each type of data separately. Our model
enables more accurate prediction of physical properties for polymers by
combining features extracted from the chemical structure by graph convolutional
neural networks (GCN) with features such as molecular descriptors and additive
information. The predictive performance of the proposed method is empirically
evaluated using several polymer datasets. We report that the proposed method
shows high predictive performance compared to the baseline conventional
approach using a single feature.

### 5. Nonparametric learning of covariate-based Markov jump processes using RKHS techniques

[Nonparametric learning of covariate-based Markov jump processes using RKHS techniques](http://arxiv.org/pdf/2505.03119v1)

Authors: Yuchen Han, Arnab Ganguly, Riten Mitra

We propose a novel nonparametric approach for linking covariates to
Continuous Time Markov Chains (CTMCs) using the mathematical framework of
Reproducing Kernel Hilbert Spaces (RKHS). CTMCs provide a robust framework for
modeling transitions across clinical or behavioral states, but traditional
multistate models often rely on linear relationships. In contrast, we use a
generalized Representer Theorem to enable tractable inference in functional
space. For the Frequentist version, we apply normed square penalties, while for
the Bayesian version, we explore sparsity inducing spike and slab priors. Due
to the computational challenges posed by high-dimensional spaces, we
successfully adapt the Expectation Maximization Variable Selection (EMVS)
algorithm to efficiently identify the posterior mode. We demonstrate the
effectiveness of our method through extensive simulation studies and an
application to follicular cell lymphoma data. Our performance metrics include
the normalized difference between estimated and true nonlinear transition
functions, as well as the difference in the probability of getting absorbed in
one the final states, capturing the ability of our approach to predict
long-term behaviors.

### 6. A Symbolic and Statistical Learning Framework to Discover Bioprocessing Regulatory Mechanism: Cell Culture Example

[A Symbolic and Statistical Learning Framework to Discover Bioprocessing Regulatory Mechanism: Cell Culture Example](http://arxiv.org/pdf/2505.03177v1)

Authors: Keilung Choy, Wei Xie, Keqi Wang

Bioprocess mechanistic modeling is essential for advancing intelligent
digital twin representation of biomanufacturing, yet challenges persist due to
complex intracellular regulation, stochastic system behavior, and limited
experimental data. This paper introduces a symbolic and statistical learning
framework to identify key regulatory mechanisms and quantify model uncertainty.
Bioprocess dynamics is formulated with stochastic differential equations
characterizing intrinsic process variability, with a predefined set of
candidate regulatory mechanisms constructed from biological knowledge. A
Bayesian learning approach is developed, which is based on a joint learning of
kinetic parameters and regulatory structure through a formulation of the
mixture model. To enhance computational efficiency, a Metropolis-adjusted
Langevin algorithm with adjoint sensitivity analysis is developed for posterior
exploration. Compared to state-of-the-art Bayesian inference approaches, the
proposed framework achieves improved sample efficiency and robust model
selection. An empirical study demonstrates its ability to recover missing
regulatory mechanisms and improve model fidelity under data-limited conditions.

### 7. Lower Bounds for Greedy Teaching Set Constructions

[Lower Bounds for Greedy Teaching Set Constructions](http://arxiv.org/pdf/2505.03223v1)

Authors: Spencer Compton, Chirag Pabbaraju, Nikita Zhivotovskiy

A fundamental open problem in learning theory is to characterize the
best-case teaching dimension $\operatorname{TS}_{\min}$ of a concept class
$\mathcal{C}$ with finite VC dimension $d$. Resolving this problem will, in
particular, settle the conjectured upper bound on Recursive Teaching Dimension
posed by [Simon and Zilles; COLT 2015]. Prior work used a natural greedy
algorithm to construct teaching sets recursively, thereby proving upper bounds
on $\operatorname{TS}_{\min}$, with the best known bound being $O(d^2)$ [Hu,
Wu, Li, and Wang; COLT 2017]. In each iteration, this greedy algorithm chooses
to add to the teaching set the $k$ labeled points that restrict the concept
class the most. In this work, we prove lower bounds on the performance of this
greedy approach for small $k$. Specifically, we show that for $k = 1$, the
algorithm does not improve upon the halving-based bound of
$O(\log(|\mathcal{C}|))$. Furthermore, for $k = 2$, we complement the upper
bound of $O\left(\log(\log(|\mathcal{C}|))\right)$ from [Moran, Shpilka,
Wigderson, and Yuhudayoff; FOCS 2015] with a matching lower bound. Most
consequentially, our lower bound extends up to $k \le \lceil c d \rceil$ for
small constant $c>0$: suggesting that studying higher-order interactions may be
necessary to resolve the conjecture that $\operatorname{TS}_{\min} = O(d)$.

### 8. The Inverse Drum Machine: Source Separation Through Joint Transcription and Analysis-by-Synthesis

[The Inverse Drum Machine: Source Separation Through Joint Transcription and Analysis-by-Synthesis](http://arxiv.org/pdf/2505.03337v1)

Authors: Bernardo Torres, Geoffroy Peeters, Gael Richard

We introduce the Inverse Drum Machine (IDM), a novel approach to drum source
separation that combines analysis-by-synthesis with deep learning. Unlike
recent supervised methods that rely on isolated stems, IDM requires only
transcription annotations. It jointly optimizes automatic drum transcription
and one-shot drum sample synthesis in an end-to-end framework. By convolving
synthesized one-shot samples with estimated onsets-mimicking a drum machine-IDM
reconstructs individual drum stems and trains a neural network to match the
original mixture. Evaluations on the StemGMD dataset show that IDM achieves
separation performance on par with state-of-the-art supervised methods, while
substantially outperforming matrix decomposition baselines.

### 9. Wasserstein Convergence of Score-based Generative Models under Semiconvexity and Discontinuous Gradients

[Wasserstein Convergence of Score-based Generative Models under Semiconvexity and Discontinuous Gradients](http://arxiv.org/pdf/2505.03432v1)

Authors: Stefano Bruno, Sotirios Sabanis

Score-based Generative Models (SGMs) approximate a data distribution by
perturbing it with Gaussian noise and subsequently denoising it via a learned
reverse diffusion process. These models excel at modeling complex data
distributions and generating diverse samples, achieving state-of-the-art
performance across domains such as computer vision, audio generation,
reinforcement learning, and computational biology. Despite their empirical
success, existing Wasserstein-2 convergence analysis typically assume strong
regularity conditions-such as smoothness or strict log-concavity of the data
distribution-that are rarely satisfied in practice. In this work, we establish
the first non-asymptotic Wasserstein-2 convergence guarantees for SGMs
targeting semiconvex distributions with potentially discontinuous gradients.
Our upper bounds are explicit and sharp in key parameters, achieving optimal
dependence of $O(\sqrt{d})$ on the data dimension $d$ and convergence rate of
order one. The framework accommodates a wide class of practically relevant
distributions, including symmetric modified half-normal distributions, Gaussian
mixtures, double-well potentials, and elastic net potentials. By leveraging
semiconvexity without requiring smoothness assumptions on the potential such as
differentiability, our results substantially broaden the theoretical
foundations of SGMs, bridging the gap between empirical success and rigorous
guarantees in non-smooth, complex data regimes.

### 10. Physics-Informed Sylvester Normalizing Flows for Bayesian Inference in Magnetic Resonance Spectroscopy

[Physics-Informed Sylvester Normalizing Flows for Bayesian Inference in Magnetic Resonance Spectroscopy](http://arxiv.org/pdf/2505.03590v1)

Authors: Julian P. Merkofer, Dennis M. J. van de Sande, Alex A. Bhogal, Ruud J. G. van Sloun

Magnetic resonance spectroscopy (MRS) is a non-invasive technique to measure
the metabolic composition of tissues, offering valuable insights into
neurological disorders, tumor detection, and other metabolic dysfunctions.
However, accurate metabolite quantification is hindered by challenges such as
spectral overlap, low signal-to-noise ratio, and various artifacts. Traditional
methods like linear-combination modeling are susceptible to ambiguities and
commonly only provide a theoretical lower bound on estimation accuracy in the
form of the Cram\'er-Rao bound. This work introduces a Bayesian inference
framework using Sylvester normalizing flows (SNFs) to approximate posterior
distributions over metabolite concentrations, enhancing quantification
reliability. A physics-based decoder incorporates prior knowledge of MRS signal
formation, ensuring realistic distribution representations. We validate the
method on simulated 7T proton MRS data, demonstrating accurate metabolite
quantification, well-calibrated uncertainties, and insights into parameter
correlations and multi-modal distributions.

