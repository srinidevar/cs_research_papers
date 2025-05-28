# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-27 17:06:15.161454 PST.

### Artificial Intelligence

### 1. [CaseEdit: Enhancing Localized Commonsense Reasoning via Null-Space Constrained Knowledge Editing in Small Parameter Language Models](http://arxiv.org/pdf/2505.19383v1)

Authors: Varun Reddy, Yen-Ling Kuo

Large language models (LLMs) exhibit strong performance on factual recall and
general reasoning but struggle to adapt to user-specific, commonsense
knowledge, a challenge particularly acute in small-parameter settings where
computational efficiency is prioritized. We introduce CaseEdit, a new dataset
and generation pipeline for evaluating localized, personalized commonsense
knowledge editing in small LLMs to address this. Built upon the ATOMIC20/20
commonsense graph, CaseEdit uses a multi-stage inference process to generate
both typical and atypical contextual edits for household objects, paired with
targeted evaluation questions across four axes: reliability, generalization,
locality, and portability. We evaluate established knowledge editing methods
using CaseEdit and demonstrate that AlphaEdit, a technique employing null-space
projection to minimize interference with unrelated knowledge, consistently
outperforms other methods when applied to an LLaMA 3.2 3B model, even in
scalability tests, showing minimal ripple effects. Our results indicate that
using CaseEdit with effective editing techniques like AlphaEdit allows small
models to internalize high-quality, context-sensitive common-sense knowledge,
paving the way for lightweight, personalized assistants.

### 2. [Unveiling the Compositional Ability Gap in Vision-Language Reasoning Model](http://arxiv.org/pdf/2505.19406v1)

Authors: Tianle Li, Jihai Zhang, Yongming Rao, Yu Cheng

While large language models (LLMs) demonstrate strong reasoning capabilities
utilizing reinforcement learning (RL) with verifiable reward, whether large
vision-language models (VLMs) can directly inherit such capabilities through
similar post-training strategies remains underexplored. In this work, we
conduct a systematic compositional probing study to evaluate whether current
VLMs trained with RL or other post-training strategies can compose capabilities
across modalities or tasks under out-of-distribution conditions. We design a
suite of diagnostic tasks that train models on unimodal tasks or isolated
reasoning skills, and evaluate them on multimodal, compositional variants
requiring skill integration. Through comparisons between supervised fine-tuning
(SFT) and RL-trained models, we identify three key findings: (1) RL-trained
models consistently outperform SFT on compositional generalization,
demonstrating better integration of learned skills; (2) although VLMs achieve
strong performance on individual tasks, they struggle to generalize
compositionally under cross-modal and cross-task scenario, revealing a
significant gap in current training strategies; (3) enforcing models to
explicitly describe visual content before reasoning (e.g.,
caption-before-thinking), along with rewarding progressive vision-to-text
grounding, yields notable gains. It highlights two essential ingredients for
improving compositionality in VLMs: visual-to-text alignment and accurate
visual grounding. Our findings shed light on the current limitations of
RL-based reasoning VLM training and provide actionable insights toward building
models that reason compositionally across modalities and tasks.

### 3. [Fusion Intelligence for Digital Twinning AI Data Centers: A Synergistic GenAI-PhyAI Approach](http://arxiv.org/pdf/2505.19409v1)

Authors: Ruihang Wang, Minghao Li, Zhiwei Cao, Jimin Jia, Kyle Guan, Yonggang Wen

The explosion in artificial intelligence (AI) applications is pushing the
development of AI-dedicated data centers (AIDCs), creating management
challenges that traditional methods and standalone AI solutions struggle to
address. While digital twins are beneficial for AI-based design validation and
operational optimization, current AI methods for their creation face
limitations. Specifically, physical AI (PhyAI) aims to capture the underlying
physical laws, which demands extensive, case-specific customization, and
generative AI (GenAI) can produce inaccurate or hallucinated results. We
propose Fusion Intelligence, a novel framework synergizing GenAI's automation
with PhyAI's domain grounding. In this dual-agent collaboration, GenAI
interprets natural language prompts to generate tokenized AIDC digital twins.
Subsequently, PhyAI optimizes these generated twins by enforcing physical
constraints and assimilating real-time data. Case studies demonstrate the
advantages of our framework in automating the creation and validation of AIDC
digital twins. These twins deliver predictive analytics to support power usage
effectiveness (PUE) optimization in the design stage. With operational data
collected, the digital twin accuracy is further improved compared with pure
physics-based models developed by human experts. Fusion Intelligence offers a
promising pathway to accelerate digital transformation. It enables more
reliable and efficient AI-driven digital transformation for a broad range of
mission-critical infrastructures.

### 4. [Style2Code: A Style-Controllable Code Generation Framework with Dual-Modal Contrastive Representation Learning](http://arxiv.org/pdf/2505.19442v1)

Authors: Dutao Zhang, Sergey Kovalchuk, YuLong He

Controllable code generation, the ability to synthesize code that follows a
specified style while maintaining functionality, remains a challenging task. We
propose a two-stage training framework combining contrastive learning and
conditional decoding to enable flexible style control. The first stage aligns
code style representations with semantic and structural features. In the second
stage, we fine-tune a language model (e.g., Flan-T5) conditioned on the learned
style vector to guide generation. Our method supports style interpolation and
user personalization via lightweight mixing. Compared to prior work, our
unified framework offers improved stylistic control without sacrificing code
correctness. This is among the first approaches to combine contrastive
alignment with conditional decoding for style-guided code generation.

### 5. [Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](http://arxiv.org/pdf/2505.19474v1)

Authors: Xinmiao Hu, Chun Wang, Ruihe An, ChenYu Shao, Xiaojun Ye, Sheng Zhou, Liangcheng Li

Multimodal Large Language Models (MLLMs) have demonstrated strong performance
in visual understanding tasks, yet they often suffer from object
hallucinations--generating descriptions of objects that are inconsistent with
or entirely absent from the input. This issue is closely related to dataset
biases, where frequent co-occurrences of objects lead to entangled semantic
representations across modalities. As a result, models may erroneously activate
object representations that are commonly associated with the input but not
actually present.
  To address this, we propose a causality-driven disentanglement framework that
mitigates hallucinations through causal intervention. Our approach includes a
Causal-Driven Projector in the visual pathway and a Causal Intervention Module
integrated into the final transformer layer of the language model. These
components work together to reduce spurious correlations caused by biased
training data.
  Experimental results show that our method significantly reduces
hallucinations while maintaining strong performance on multiple multimodal
benchmarks. Visualization analyses further confirm improved separability of
object representations.
  The code is available at: https://github.com/IgniSavium/Causal-LLaVA

### 6. [Judging with Many Minds: Do More Perspectives Mean Less Prejudice?](http://arxiv.org/pdf/2505.19477v1)

Authors: Chiyu Ma, Enpei Zhang, Yilun Zhao, Wenjun Liu, Yaning Jia, Peijun Qing, Lin Shi, Arman Cohan, Yujun Yan, Soroush Vosoughi

LLM-as-Judge has emerged as a scalable alternative to human evaluation,
enabling large language models (LLMs) to provide reward signals in trainings.
While recent work has explored multi-agent extensions such as multi-agent
debate and meta-judging to enhance evaluation quality, the question of how
intrinsic biases manifest in these settings remains underexplored. In this
study, we conduct a systematic analysis of four diverse bias types: position
bias, verbosity bias, chain-of-thought bias, and bandwagon bias. We evaluate
these biases across two widely adopted multi-agent LLM-as-Judge frameworks:
Multi-Agent-Debate and LLM-as-Meta-Judge. Our results show that debate
framework amplifies biases sharply after the initial debate, and this increased
bias is sustained in subsequent rounds, while meta-judge approaches exhibit
greater resistance. We further investigate the incorporation of PINE, a leading
single-agent debiasing method, as a bias-free agent within these systems. The
results reveal that this bias-free agent effectively reduces biases in debate
settings but provides less benefit in meta-judge scenarios. Our work provides a
comprehensive study of bias behavior in multi-agent LLM-as-Judge systems and
highlights the need for targeted bias mitigation strategies in collaborative
evaluation settings.

### 7. [Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models](http://arxiv.org/pdf/2505.19490v1)

Authors: Jianxing Liao, Junyan Xu, Yatao Sun, Maowen Tang, Sicheng He, Jingxian Liao, Shui Yu, Yun Li, Hongguan Xiao

Designing complex computer-aided design (CAD) models is often time-consuming
due to challenges such as computational inefficiency and the difficulty of
generating precise models. We propose a novel language-guided framework for
industrial design automation to address these issues, integrating large
language models (LLMs) with computer-automated design (CAutoD).Through this
framework, CAD models are automatically generated from parameters and
appearance descriptions, supporting the automation of design tasks during the
detailed CAD design phase. Our approach introduces three key innovations: (1) a
semi-automated data annotation pipeline that leverages LLMs and vision-language
large models (VLLMs) to generate high-quality parameters and appearance
descriptions; (2) a Transformer-based CAD generator (TCADGen) that predicts
modeling sequences via dual-channel feature aggregation; (3) an enhanced CAD
modeling generation model, called CADLLM, that is designed to refine the
generated sequences by incorporating the confidence scores from TCADGen.
Experimental results demonstrate that the proposed approach outperforms
traditional methods in both accuracy and efficiency, providing a powerful tool
for automating industrial workflows and generating complex CAD models from
textual prompts. The code is available at
https://jianxliao.github.io/cadllm-page/

### 8. [Genome-Bench: A Scientific Reasoning Benchmark from Real-World Expert Discussions](http://arxiv.org/pdf/2505.19501v1)

Authors: Ming Yin, Yuanhao Qu, Dyllan Liu, Ling Yang, Le Cong, Mengdi Wang

In this short report, we present an automated pipeline tailored for the
genomics domain and introduce \textit{Genome-Bench}, a new benchmark
constructed from over a decade of scientific forum discussions on genome
engineering. Our pipeline transforms raw interactions into a reinforcement
learning friendly multiple-choice questions format, supported by 3000+ high
quality question answer pairs spanning foundational biology, experimental
troubleshooting, tool usage, and beyond. To our knowledge, this is the first
end-to-end pipeline for teaching LLMs to reason from scientific discussions,
with promising potential for generalization across scientific domains beyond
biology.

### 9. [Turing Test 2.0: The General Intelligence Threshold](http://arxiv.org/pdf/2505.19550v1)

Authors: Georgios Mappouras

With the rise of artificial intelligence (A.I.) and large language models
like Chat-GPT, a new race for achieving artificial general intelligence (A.G.I)
has started. While many speculate how and when A.I. will achieve A.G.I., there
is no clear agreement on how A.G.I. can be detected in A.I. models, even when
popular tools like the Turing test (and its modern variations) are used to
measure their intelligence. In this work, we discuss why traditional methods
like the Turing test do not suffice for measuring or detecting A.G.I. and
provide a new, practical method that can be used to decide if a (computer or
any other) system has reached or surpassed A.G.I. To achieve this, we make two
new contributions. First, we present a clear definition for general
intelligence (G.I.) and set a G.I. threshold (G.I.T.) that can be used to
distinguish between systems that achieve A.G.I. and systems that do not.
Second, we present a new framework on how to construct tests that can detect if
a system has achieved G.I. in a simple, comprehensive, and clear-cut fail/pass
way. We call this novel framework the Turing Tests 2.0. We then demonstrate
real-life examples of applying tests that follow our Turing Tests 2.0 framework
on modern A.I. models.

### 10. [AMQA: An Adversarial Dataset for Benchmarking Bias of LLMs in Medicine and Healthcare](http://arxiv.org/pdf/2505.19562v1)

Authors: Ying Xiao, Jie Huang, Ruijuan He, Jing Xiao, Mohammad Reza Mousavi, Yepang Liu, Kezhi Li, Zhenpeng Chen, Jie M. Zhang

Large language models (LLMs) are reaching expert-level accuracy on medical
diagnosis questions, yet their mistakes and the biases behind them pose
life-critical risks. Bias linked to race, sex, and socioeconomic status is
already well known, but a consistent and automatic testbed for measuring it is
missing. To fill this gap, this paper presents AMQA -- an Adversarial Medical
Question-Answering dataset -- built for automated, large-scale bias evaluation
of LLMs in medical QA. AMQA includes 4,806 medical QA pairs sourced from the
United States Medical Licensing Examination (USMLE) dataset, generated using a
multi-agent framework to create diverse adversarial descriptions and question
pairs. Using AMQA, we benchmark five representative LLMs and find surprisingly
substantial disparities: even GPT-4.1, the least biased model tested, answers
privileged-group questions over 10 percentage points more accurately than
unprivileged ones. Compared with the existing benchmark CPV, AMQA reveals 15%
larger accuracy gaps on average between privileged and unprivileged groups. Our
dataset and code are publicly available at https://github.com/XY-Showing/AMQA
to support reproducible research and advance trustworthy, bias-aware medical
AI.

### Hardware Architecture

### 1. [Enhancing Test Efficiency through Automated ATPG-Aware Lightweight Scan Instrumentation](http://arxiv.org/pdf/2505.19418v1)

Authors: Sudipta Paria, Md Rezoan Ferdous, Aritra Dasgupta, Atri Chatterjee, Swarup Bhunia

Scan-based Design-for-Testability (DFT) measures are prevalent in modern
digital integrated circuits to achieve high test quality at low hardware cost.
With the advent of 3D heterogeneous integration and chiplet-based systems, the
role of scan is becoming ever more important due to its ability to make
internal design nodes controllable and observable in a systematic and scalable
manner. However, the effectiveness of scan-based DFT suffers from poor
testability of internal nodes for complex circuits at deep logic levels.
Existing solutions to address this problem primarily rely on Test Point
Insertion (TPI) in the nodes with poor controllability or observability.
However, TPI-based solutions, while an integral part of commercial practice,
come at a high design and hardware cost. To address this issue, in this paper,
we present LITE, a novel ATPG-aware lightweight scan instrumentation approach
that utilizes the functional flip-flops in a scan chain to make multiple
internal nodes observable and controllable in a low-cost, scalable manner. We
provide both circuit-level design as well as an algorithmic approach for
automating the insertion of LITE for design modifications. We show that LITE
significantly improves the testability in terms of the number of patterns and
test coverage for ATPG and random pattern testability, respectively, while
incurring considerably lower overhead than TPI-based solutions.

### 2. [ReChisel: Effective Automatic Chisel Code Generation by LLM with Reflection](http://arxiv.org/pdf/2505.19734v1)

Authors: Juxin Niu, Xiangfeng Liu, Dan Niu, Xi Wang, Zhe Jiang, Nan Guan

Coding with hardware description languages (HDLs) such as Verilog is a
time-intensive and laborious task. With the rapid advancement of large language
models (LLMs), there is increasing interest in applying LLMs to assist with HDL
coding. Recent efforts have demonstrated the potential of LLMs in translating
natural language to traditional HDL Verilog. Chisel, a next-generation HDL
based on Scala, introduces higher-level abstractions, facilitating more
concise, maintainable, and scalable hardware designs. However, the potential of
using LLMs for Chisel code generation remains largely unexplored. This work
proposes ReChisel, an LLM-based agentic system designed to enhance the
effectiveness of Chisel code generation. ReChisel incorporates a reflection
mechanism to iteratively refine the quality of generated code using feedback
from compilation and simulation processes, and introduces an escape mechanism
to break free from non-progress loops. Experiments demonstrate that ReChisel
significantly improves the success rate of Chisel code generation, achieving
performance comparable to state-of-the-art LLM-based agentic systems for
Verilog code generation.

### Computational Complexity

### 1. [Better Extension Variables in DQBF via Independence](http://arxiv.org/pdf/2505.20069v1)

Authors: Leroy Chew, Tomáš Peitl

We show that extension variables in (D)QBF can be generalised by conditioning
on universal assignments. The benefit of this is that the dependency sets of
such conditioned extension variables can be made smaller to allow easier
refutations. This simple modification instantly solves many challenges in
p-simulating the QBF expansion rule, which cannot be p-simulated in proof
systems that have strategy extraction. Simulating expansion is even more
crucial in DQBF, where other methods are incomplete. In this paper we provide
an overview of the strength of this new independent extension rule. We find
that a new version of Extended Frege called IndExtFrege+Red can p-simulate a
multitude of difficult QBF and DQBF techniques, even techniques that are
difficult to approach with ExtFrege+Red. We show six p-simulations, that
IndExtFrege+Red p-simulates QRAT, IR(D)-Calc, Q(Drrs)-Res, Fork Resolution,
DQRAT and G, which together underpin most DQBF solving and preprocessing
techniques. The p-simulations work despite these systems using complicated
rules and our new extension rule being relatively simple. Moreover, unlike
recent p-simulations by ExtFrege+Red we can simulate the proof rules line by
line, which allows us to mix QBF rules more easily with other inference steps.

### Computational Engineering

### 1. [Integrated Finite Element Neural Network (IFENN) for Phase-Field Fracture with Minimal Input and Generalized Geometry-Load Handling](http://arxiv.org/pdf/2505.19566v1)

Authors: Panos Pantidis, Lampros Svolos, Diab Abueidda, Mostafa E. Mobasher

We present a novel formulation for modeling phase-field fracture propagation
based on the Integrated Finite Element Neural Network (IFENN) framework. IFENN
is a hybrid solver scheme that utilizes neural networks as PDE solvers within
FEM, preserving accuracy via residual minimization while achieving speed-up via
swift network predictions and reduction of the size of system of equations in
coupled problems. In this work, we introduce a radically new formulation of
IFENN in which the phase-field variable is calculated using physics-informed
convolutional networks (PICNNs), while the equilibrium equation is still solved
using FEM to maintain the solver robustness. Unlike conventional approaches,
which rely on sequence or time-dependent models, we eliminate the need to
include temporal features in the training setup and inference stage. Instead,
we show that it is sufficient to learn only the spatial coupling between the
strain energy density and the phase-field variable in the vicinity of the
fracture process zone, and utilize this information along the advancing crack
simulation. We train a single CNN in a purely physics-based, unsupervised
manner on just two load increments from a single-notch tension problem, with a
total training time of only 5 minutes. Following this exceptionally minimal and
fast training, we show that the same PICNN can (when embedded within IFENN)
model crack propagation in a very wide range of unseen scenarios, including
arbitrarily rectangular domains, single and multiple interacting cracks,
varying mesh densities, and arbitrary loading paths. The proposed formulation
delivers breakthroughs that address many of the limitations in the existing
literature of hybrid modeling, introducing a new paradigm for the development
of generalizable, physics-consistent hybrid models that are applicable to
fracture and other coupled problems.

### 2. [Cross-Sequence Semi-Supervised Learning for Multi-Parametric MRI-Based Visual Pathway Delineation](http://arxiv.org/pdf/2505.19733v1)

Authors: Alou Diakite, Cheng Li, Lei Xie, Yuanjing Feng, Ruoyou Wu, Jianzhong He, Hairong Zheng, Shanshan Wang

Accurately delineating the visual pathway (VP) is crucial for understanding
the human visual system and diagnosing related disorders. Exploring
multi-parametric MR imaging data has been identified as an important way to
delineate VP. However, due to the complex cross-sequence relationships,
existing methods cannot effectively model the complementary information from
different MRI sequences. In addition, these existing methods heavily rely on
large training data with labels, which is labor-intensive and time-consuming to
obtain. In this work, we propose a novel semi-supervised multi-parametric
feature decomposition framework for VP delineation. Specifically, a
correlation-constrained feature decomposition (CFD) is designed to handle the
complex cross-sequence relationships by capturing the unique characteristics of
each MRI sequence and easing the multi-parametric information fusion process.
Furthermore, a consistency-based sample enhancement (CSE) module is developed
to address the limited labeled data issue, by generating and promoting
meaningful edge information from unlabeled data. We validate our framework
using two public datasets, and one in-house Multi-Shell Diffusion MRI (MDM)
dataset. Experimental results demonstrate the superiority of our approach in
terms of delineation performance when compared to seven state-of-the-art
approaches.

### 3. [FinLoRA: Benchmarking LoRA Methods for Fine-Tuning LLMs on Financial Datasets](http://arxiv.org/pdf/2505.19819v1)

Authors: Dannong Wang, Jaisal Patel, Daochen Zha, Steve Y. Yang, Xiao-Yang Liu

Low-rank adaptation (LoRA) methods show great potential for scaling
pre-trained general-purpose Large Language Models (LLMs) to hundreds or
thousands of use scenarios. However, their efficacy in high-stakes domains like
finance is rarely explored, e.g., passing CFA exams and analyzing SEC filings.
In this paper, we present the open-source FinLoRA project that benchmarks LoRA
methods on both general and highly professional financial tasks. First, we
curated 19 datasets covering diverse financial applications; in particular, we
created four novel XBRL analysis datasets based on 150 SEC filings. Second, we
evaluated five LoRA methods and five base LLMs. Finally, we provide extensive
experimental results in terms of accuracy, F1, and BERTScore and report
computational cost in terms of time and GPU memory during fine-tuning and
inference stages. We find that LoRA methods achieved substantial performance
gains of 36\% on average over base models. Our FinLoRA project provides an
affordable and scalable approach to democratize financial intelligence to the
general public. Datasets, LoRA adapters, code, and documentation are available
at https://github.com/Open-Finance-Lab/FinLoRA

### 4. [BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs](http://arxiv.org/pdf/2505.19457v1)

Authors: Guilong Lu, Xuntao Guo, Rongjunchen Zhang, Wenqiao Zhu, Ji Liu

Large language models excel in general tasks, yet assessing their reliability
in logic-heavy, precision-critical domains like finance, law, and healthcare
remains challenging. To address this, we introduce BizFinBench, the first
benchmark specifically designed to evaluate LLMs in real-world financial
applications. BizFinBench consists of 6,781 well-annotated queries in Chinese,
spanning five dimensions: numerical calculation, reasoning, information
extraction, prediction recognition, and knowledge-based question answering,
grouped into nine fine-grained categories. The benchmark includes both
objective and subjective metrics. We also introduce IteraJudge, a novel LLM
evaluation method that reduces bias when LLMs serve as evaluators in objective
metrics. We benchmark 25 models, including both proprietary and open-source
systems. Extensive experiments show that no model dominates across all tasks.
Our evaluation reveals distinct capability patterns: (1) In Numerical
Calculation, Claude-3.5-Sonnet (63.18) and DeepSeek-R1 (64.04) lead, while
smaller models like Qwen2.5-VL-3B (15.92) lag significantly; (2) In Reasoning,
proprietary models dominate (ChatGPT-o3: 83.58, Gemini-2.0-Flash: 81.15), with
open-source models trailing by up to 19.49 points; (3) In Information
Extraction, the performance spread is the largest, with DeepSeek-R1 scoring
71.46, while Qwen3-1.7B scores 11.23; (4) In Prediction Recognition,
performance variance is minimal, with top models scoring between 39.16 and
50.00. We find that while current LLMs handle routine finance queries
competently, they struggle with complex scenarios requiring cross-concept
reasoning. BizFinBench offers a rigorous, business-aligned benchmark for future
research. The code and dataset are available at
https://github.com/HiThink-Research/BizFinBench.

### 5. [DoctorRAG: Medical RAG Fusing Knowledge with Patient Analogy through Textual Gradients](http://arxiv.org/pdf/2505.19538v1)

Authors: Yuxing Lu, Gecheng Fu, Wei Wu, Xukai Zhao, Sin Yee Goi, Jinzhuo Wang

Existing medical RAG systems mainly leverage knowledge from medical knowledge
bases, neglecting the crucial role of experiential knowledge derived from
similar patient cases -- a key component of human clinical reasoning. To bridge
this gap, we propose DoctorRAG, a RAG framework that emulates doctor-like
reasoning by integrating both explicit clinical knowledge and implicit
case-based experience. DoctorRAG enhances retrieval precision by first
allocating conceptual tags for queries and knowledge sources, together with a
hybrid retrieval mechanism from both relevant knowledge and patient. In
addition, a Med-TextGrad module using multi-agent textual gradients is
integrated to ensure that the final output adheres to the retrieved knowledge
and patient query. Comprehensive experiments on multilingual, multitask
datasets demonstrate that DoctorRAG significantly outperforms strong baseline
RAG models and gains improvements from iterative refinements. Our approach
generates more accurate, relevant, and comprehensive responses, taking a step
towards more doctor-like medical reasoning systems.

### Computation and Language

### 1. [Belief Attribution as Mental Explanation: The Role of Accuracy, Informativity, and Causality](http://arxiv.org/pdf/2505.19376v1)

Authors: Lance Ying, Almog Hillel, Ryan Truong, Vikash K. Mansinghka, Joshua B. Tenenbaum, Tan Zhi-Xuan

A key feature of human theory-of-mind is the ability to attribute beliefs to
other agents as mentalistic explanations for their behavior. But given the wide
variety of beliefs that agents may hold about the world and the rich language
we can use to express them, which specific beliefs are people inclined to
attribute to others? In this paper, we investigate the hypothesis that people
prefer to attribute beliefs that are good explanations for the behavior they
observe. We develop a computational model that quantifies the explanatory
strength of a (natural language) statement about an agent's beliefs via three
factors: accuracy, informativity, and causal relevance to actions, each of
which can be computed from a probabilistic generative model of belief-driven
behavior. Using this model, we study the role of each factor in how people
selectively attribute beliefs to other agents. We investigate this via an
experiment where participants watch an agent collect keys hidden in boxes in
order to reach a goal, then rank a set of statements describing the agent's
beliefs about the boxes' contents. We find that accuracy and informativity
perform reasonably well at predicting these rankings when combined, but that
causal relevance is the single factor that best explains participants'
responses.

### 2. [gec-metrics: A Unified Library for Grammatical Error Correction Evaluation](http://arxiv.org/pdf/2505.19388v1)

Authors: Takumi Goto, Yusuke Sakai, Taro Watanabe

We introduce gec-metrics, a library for using and developing grammatical
error correction (GEC) evaluation metrics through a unified interface. Our
library enables fair system comparisons by ensuring that everyone conducts
evaluations using a consistent implementation. Moreover, it is designed with a
strong focus on API usage, making it highly extensible. It also includes
meta-evaluation functionalities and provides analysis and visualization
scripts, contributing to developing GEC evaluation metrics. Our code is
released under the MIT license and is also distributed as an installable
package. The video is available on YouTube.

### 3. [Self-Reflective Planning with Knowledge Graphs: Enhancing LLM Reasoning Reliability for Question Answering](http://arxiv.org/pdf/2505.19410v1)

Authors: Jiajun Zhu, Ye Liu, Meikai Bao, Kai Zhang, Yanghai Zhang, Qi Liu

Recently, large language models (LLMs) have demonstrated remarkable
capabilities in natural language processing tasks, yet they remain prone to
hallucinations when reasoning with insufficient internal knowledge. While
integrating LLMs with knowledge graphs (KGs) provides access to structured,
verifiable information, existing approaches often generate incomplete or
factually inconsistent reasoning paths. To this end, we propose Self-Reflective
Planning (SRP), a framework that synergizes LLMs with KGs through iterative,
reference-guided reasoning. Specifically, given a question and topic entities,
SRP first searches for references to guide planning and reflection. In the
planning process, it checks initial relations and generates a reasoning path.
After retrieving knowledge from KGs through a reasoning path, it implements
iterative reflection by judging the retrieval result and editing the reasoning
path until the answer is correctly retrieved. Extensive experiments on three
public datasets demonstrate that SRP surpasses various strong baselines and
further underscore its reliable reasoning ability.

### 4. [Frictional Agent Alignment Framework: Slow Down and Don't Break Things](http://arxiv.org/pdf/2505.19428v1)

Authors: Abhijnan Nath, Carine Graff, Andrei Bachinin, Nikhil Krishnaswamy

AI support of collaborative interactions entails mediating potential
misalignment between interlocutor beliefs. Common preference alignment methods
like DPO excel in static settings, but struggle in dynamic collaborative tasks
where the explicit signals of interlocutor beliefs are sparse and skewed. We
propose the Frictional Agent Alignment Framework (FAAF), to generate precise,
context-aware "friction" that prompts for deliberation and re-examination of
existing evidence. FAAF's two-player objective decouples from data skew: a
frictive-state policy identifies belief misalignments, while an intervention
policy crafts collaborator-preferred responses. We derive an analytical
solution to this objective, enabling training a single policy via a simple
supervised loss. Experiments on three benchmarks show FAAF outperforms
competitors in producing concise, interpretable friction and in OOD
generalization. By aligning LLMs to act as adaptive "thought partners" -- not
passive responders -- FAAF advances scalable, dynamic human-AI collaboration.
Our code and data can be found at https://github.com/csu-signal/FAAF_ACL.

### 5. [Rhapsody: A Dataset for Highlight Detection in Podcasts](http://arxiv.org/pdf/2505.19429v1)

Authors: Younghan Park, Anuj Diwan, David Harwath, Eunsol Choi

Podcasts have become daily companions for half a billion users. Given the
enormous amount of podcast content available, highlights provide a valuable
signal that helps viewers get the gist of an episode and decide if they want to
invest in listening to it in its entirety. However, identifying highlights
automatically is challenging due to the unstructured and long-form nature of
the content. We introduce Rhapsody, a dataset of 13K podcast episodes paired
with segment-level highlight scores derived from YouTube's 'most replayed'
feature. We frame the podcast highlight detection as a segment-level binary
classification task. We explore various baseline approaches, including
zero-shot prompting of language models and lightweight finetuned language
models using segment-level classification heads. Our experimental results
indicate that even state-of-the-art language models like GPT-4o and Gemini
struggle with this task, while models finetuned with in-domain data
significantly outperform their zero-shot performance. The finetuned model
benefits from leveraging both speech signal features and transcripts. These
findings highlight the challenges for fine-grained information access in
long-form spoken media.

### 6. [Route to Reason: Adaptive Routing for LLM and Reasoning Strategy Selection](http://arxiv.org/pdf/2505.19435v1)

Authors: Zhihong Pan, Kai Zhang, Yuze Zhao, Yupeng Han

The inherent capabilities of a language model (LM) and the reasoning
strategies it employs jointly determine its performance in reasoning tasks.
While test-time scaling is regarded as an effective approach to tackling
complex reasoning tasks, it incurs substantial computational costs and often
leads to "overthinking", where models become trapped in "thought pitfalls". To
address this challenge, we propose Route-To-Reason (RTR), a novel unified
routing framework that dynamically allocates both LMs and reasoning strategies
according to task difficulty under budget constraints. RTR learns compressed
representations of both expert models and reasoning strategies, enabling their
joint and adaptive selection at inference time. This method is low-cost, highly
flexible, and can be seamlessly extended to arbitrary black-box or white-box
models and strategies, achieving true plug-and-play functionality. Extensive
experiments across seven open source models and four reasoning strategies
demonstrate that RTR achieves an optimal trade-off between accuracy and
computational efficiency among all baselines, achieving higher accuracy than
the best single model while reducing token usage by over 60%.

### 7. [Surrogate Signals from Format and Length: Reinforcement Learning for Solving Mathematical Problems without Ground Truth Answers](http://arxiv.org/pdf/2505.19439v1)

Authors: Rihui Xin, Han Liu, Zecheng Wang, Yupeng Zhang, Dianbo Sui, Xiaolin Hu, Bingning Wang

Large Language Models have achieved remarkable success in natural language
processing tasks, with Reinforcement Learning playing a key role in adapting
them to specific applications. However, obtaining ground truth answers for
training LLMs in mathematical problem-solving is often challenging, costly, and
sometimes unfeasible. This research delves into the utilization of format and
length as surrogate signals to train LLMs for mathematical problem-solving,
bypassing the need for traditional ground truth answers.Our study shows that a
reward function centered on format correctness alone can yield performance
improvements comparable to the standard GRPO algorithm in early phases.
Recognizing the limitations of format-only rewards in the later phases, we
incorporate length-based rewards. The resulting GRPO approach, leveraging
format-length surrogate signals, not only matches but surpasses the performance
of the standard GRPO algorithm relying on ground truth answers in certain
scenarios, achieving 40.0\% accuracy on AIME2024 with a 7B base model. Through
systematic exploration and experimentation, this research not only offers a
practical solution for training LLMs to solve mathematical problems and
reducing the dependence on extensive ground truth data collection, but also
reveals the essence of why our label-free approach succeeds: base model is like
an excellent student who has already mastered mathematical and logical
reasoning skills, but performs poorly on the test paper, it simply needs to
develop good answering habits to achieve outstanding results in exams , in
other words, to unlock the capabilities it already possesses.

### 8. [Balancing Computation Load and Representation Expressivity in Parallel Hybrid Neural Networks](http://arxiv.org/pdf/2505.19472v1)

Authors: Mohammad Mahdi Moradi, Walid Ahmed, Shuangyue Wen, Sudhir Mudur, Weiwei Zhang, Yang Liu

Attention and State-Space Models (SSMs) when combined in a hybrid network in
sequence or in parallel provide complementary strengths. In a hybrid sequential
pipeline they alternate between applying a transformer to the input and then
feeding its output into a SSM. This results in idle periods in the individual
components increasing end-to-end latency and lowering throughput caps. In the
parallel hybrid architecture, the transformer operates independently in
parallel with the SSM, and these pairs are cascaded, with output from one pair
forming the input to the next. Two issues are (i) creating an expressive
knowledge representation with the inherently divergent outputs from these
separate branches, and (ii) load balancing the computation between these
parallel branches, while maintaining representation fidelity. In this work we
present FlowHN, a novel parallel hybrid network architecture that accommodates
various strategies for load balancing, achieved through appropriate
distribution of input tokens between the two branches. Two innovative
differentiating factors in FlowHN include a FLOP aware dynamic token split
between the attention and SSM branches yielding efficient balance in compute
load, and secondly, a method to fuse the highly divergent outputs from
individual branches for enhancing representation expressivity. Together they
enable much better token processing speeds, avoid bottlenecks, and at the same
time yield significantly improved accuracy as compared to other competing
works. We conduct comprehensive experiments on autoregressive language modeling
for models with 135M, 350M, and 1B parameters. FlowHN outperforms sequential
hybrid models and its parallel counterpart, achieving up to 4* higher Tokens
per Second (TPS) and 2* better Model FLOPs Utilization (MFU).

### 9. [Continuous Self-Improvement of Large Language Models by Test-time Training with Verifier-Driven Sample Selection](http://arxiv.org/pdf/2505.19475v1)

Authors: Mohammad Mahdi Moradi, Hossam Amer, Sudhir Mudur, Weiwei Zhang, Yang Liu, Walid Ahmed

Learning to adapt pretrained language models to unlabeled,
out-of-distribution data is a critical challenge, as models often falter on
structurally novel reasoning tasks even while excelling within their training
distribution. We introduce a new framework called VDS-TTT - Verifier-Driven
Sample Selection for Test-Time Training to efficiently address this. We use a
learned verifier to score a pool of generated responses and select only from
high ranking pseudo-labeled examples for fine-tuned adaptation. Specifically,
for each input query our LLM generates N candidate answers; the verifier
assigns a reliability score to each, and the response with the highest
confidence and above a fixed threshold is paired with its query for test-time
training. We fine-tune only low-rank LoRA adapter parameters, ensuring
adaptation efficiency and fast convergence. Our proposed self-supervised
framework is the first to synthesize verifier driven test-time training data
for continuous self-improvement of the model. Experiments across three diverse
benchmarks and three state-of-the-art LLMs demonstrate that VDS-TTT yields up
to a 32.29% relative improvement over the base model and a 6.66% gain compared
to verifier-based methods without test-time training, highlighting its
effectiveness and efficiency for on-the-fly large language model adaptation.

### 10. [CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis](http://arxiv.org/pdf/2505.19484v1)

Authors: Ruixiang Feng, Shen Gao, Xiuying Chen, Lisi Chen, Shuo Shang

Large Language Models (LLMs) have demonstrated remarkable capabilities across
various tasks, yet they often exhibit a specific cultural biases, neglecting
the values and linguistic diversity of low-resource regions. This cultural bias
not only undermines universal equality, but also risks reinforcing stereotypes
and perpetuating discrimination. To address this, we propose CulFiT, a novel
culturally-aware training paradigm that leverages multilingual data and
fine-grained reward modeling to enhance cultural sensitivity and inclusivity.
Our approach synthesizes diverse cultural-related questions, constructs
critique data in culturally relevant languages, and employs fine-grained
rewards to decompose cultural texts into verifiable knowledge units for
interpretable evaluation. We also introduce GlobalCultureQA, a multilingual
open-ended question-answering dataset designed to evaluate culturally-aware
responses in a global context. Extensive experiments on three existing
benchmarks and our GlobalCultureQA demonstrate that CulFiT achieves
state-of-the-art open-source model performance in cultural alignment and
general reasoning.

### Cryptography and Security

### 1. [An Empirical Study of JavaScript Inclusion Security Issues in Chrome Extensions](http://arxiv.org/pdf/2505.19456v1)

Authors: Chong Guan

JavaScript, a scripting language employed to augment the capabilities of web
browsers within web pages or browser extensions, utilizes code segments termed
JavaScript inclusions. While the security aspects of JavaScript inclusions in
web pages have undergone substantial scrutiny, a thorough investigation into
the security of such inclusions within browser extensions remains absent,
despite the divergent security paradigms governing these environments. This
study presents a systematic measurement of JavaScript inclusions in Chrome
extensions, employing a hybrid methodology encompassing static and dynamic
analysis to identify these inclusions. The analysis of 36,324 extensions
revealed 350,784 JavaScript inclusions. Subsequent security assessment
indicated that, although the majority of these inclusions originate from local
files within the extensions rather than external servers, 22 instances of
vulnerable remote JavaScript inclusions were identified. These remote
inclusions present potential avenues for malicious actors to execute arbitrary
code within the extension's execution context. Furthermore, an analysis of
JavaScript library utilization within Chrome extensions disclosed the prevalent
use of susceptible and outdated libraries, notably within numerous widely
adopted extensions.

### 2. [Language of Network: A Generative Pre-trained Model for Encrypted Traffic Comprehension](http://arxiv.org/pdf/2505.19482v1)

Authors: Di Zhao, Bo Jiang, Song Liu, Susu Cui, Meng Shen, Dongqi Han, Xingmao Guan, Zhigang Lu

The increasing demand for privacy protection and security considerations
leads to a significant rise in the proportion of encrypted network traffic.
Since traffic content becomes unrecognizable after encryption, accurate
analysis is challenging, making it difficult to classify applications and
detect attacks. Deep learning is currently the predominant approach for
encrypted traffic classification through feature analysis. However, these
methods face limitations due to their high dependence on labeled data and
difficulties in detecting attack variants. First, their performance is highly
sensitive to data quality, where the highcost manual labeling process and
dataset imbalance significantly degrade results. Second, the rapid evolution of
attack patterns makes it challenging for models to identify new types of
attacks. To tackle these challenges, we present GBC, a generative model based
on pre-training for encrypted traffic comprehension. Since traditional
tokenization methods are primarily designed for natural language, we propose a
protocol-aware tokenization approach for encrypted traffic that improves model
comprehension of fields specific to network traffic. In addition, GBC employs
pretraining to learn general representations from extensive unlabeled traffic
data. Through prompt learning, it effectively adapts to various downstream
tasks, enabling both high-quality traffic generation and effective detection.
Evaluations across multiple datasets demonstrate that GBC achieves superior
results in both traffic classification and generation tasks, resulting in a 5%
improvement in F1 score compared to state-of-the-art methods for classification
tasks.

### 3. [Weak-Jamming Detection in IEEE 802.11 Networks: Techniques, Scenarios and Mobility](http://arxiv.org/pdf/2505.19633v1)

Authors: Martijn Hanegraaf, Savio Sciancalepore, Gabriele Oligeri

State-of-the-art solutions detect jamming attacks ex-post, i.e., only when
jamming has already disrupted the wireless communication link. In many
scenarios, e.g., mobile networks or static deployments distributed over a large
geographical area, it is often desired to detect jamming at the early stage,
when it affects the communication link enough to be detected but not
sufficiently to disrupt it (detection of weak jamming signals). Under such
assumptions, devices can enhance situational awareness and promptly apply
mitigation, e.g., moving away from the jammed area in mobile scenarios or
changing communication frequency in static deployments, before jamming fully
disrupts the communication link. Although some contributions recently
demonstrated the feasibility of detecting low-power and weak jamming signals,
they make simplistic assumptions far from real-world deployments. Given the
current state of the art, no evidence exists that detection of weak jamming can
be considered with real-world communication technologies. In this paper, we
provide and comprehensively analyze new general-purpose strategies for
detecting weak jamming signals, compatible by design with one of the most
relevant communication technologies used by commercial-off-the-shelf devices,
i.e., IEEE 802.11. We describe two operational modes: (i) binary classification
via Convolutional Neural Networks and (ii) one-class classification via Sparse
Autoencoders. We evaluate and compare the proposed approaches with the current
state-of-the-art using data collected through an extensive real-world
experimental campaign in three relevant environments. At the same time, we made
the dataset available to the public. Our results demonstrate that detecting
weak jamming signals is feasible in all considered real-world environments, and
we provide an in-depth analysis considering different techniques, scenarios,
and mobility patterns.

### 4. [CPA-RAG:Covert Poisoning Attacks on Retrieval-Augmented Generation in Large Language Models](http://arxiv.org/pdf/2505.19864v1)

Authors: Chunyang Li, Junwei Zhang, Anda Cheng, Zhuo Ma, Xinghua Li, Jianfeng Ma

Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by
incorporating external knowledge, but its openness introduces vulnerabilities
that can be exploited by poisoning attacks. Existing poisoning methods for RAG
systems have limitations, such as poor generalization and lack of fluency in
adversarial texts. In this paper, we propose CPA-RAG, a black-box adversarial
framework that generates query-relevant texts capable of manipulating the
retrieval process to induce target answers. The proposed method integrates
prompt-based text generation, cross-guided optimization through multiple LLMs,
and retriever-based scoring to construct high-quality adversarial samples. We
conduct extensive experiments across multiple datasets and LLMs to evaluate its
effectiveness. Results show that the framework achieves over 90\% attack
success when the top-k retrieval setting is 5, matching white-box performance,
and maintains a consistent advantage of approximately 5 percentage points
across different top-k values. It also outperforms existing black-box baselines
by 14.5 percentage points under various defense strategies. Furthermore, our
method successfully compromises a commercial RAG system deployed on Alibaba's
BaiLian platform, demonstrating its practical threat in real-world
applications. These findings underscore the need for more robust and secure RAG
frameworks to defend against poisoning attacks.

### 5. [VADER: A Human-Evaluated Benchmark for Vulnerability Assessment, Detection, Explanation, and Remediation](http://arxiv.org/pdf/2505.19395v1)

Authors: Ethan TS. Liu, Austin Wang, Spencer Mateega, Carlos Georgescu, Danny Tang

Ensuring that large language models (LLMs) can effectively assess, detect,
explain, and remediate software vulnerabilities is critical for building robust
and secure software systems. We introduce VADER, a human-evaluated benchmark
designed explicitly to assess LLM performance across four key
vulnerability-handling dimensions: assessment, detection, explanation, and
remediation. VADER comprises 174 real-world software vulnerabilities, each
carefully curated from GitHub repositories and annotated by security experts.
For each vulnerability case, models are tasked with identifying the flaw,
classifying it using Common Weakness Enumeration (CWE), explaining its
underlying cause, proposing a patch, and formulating a test plan. Using a
one-shot prompting strategy, we benchmark six state-of-the-art LLMs (Claude 3.7
Sonnet, Gemini 2.5 Pro, GPT-4.1, GPT-4.5, Grok 3 Beta, and o3) on VADER, and
human security experts evaluated each response according to a rigorous scoring
rubric emphasizing remediation (quality of the code fix, 50%), explanation
(20%), and classification and test plan (30%) according to a standardized
rubric. Our results show that current state-of-the-art LLMs achieve only
moderate success on VADER - OpenAI's o3 attained 54.7% accuracy overall, with
others in the 49-54% range, indicating ample room for improvement. Notably,
remediation quality is strongly correlated (Pearson r > 0.97) with accurate
classification and test plans, suggesting that models that effectively
categorize vulnerabilities also tend to fix them well. VADER's comprehensive
dataset, detailed evaluation rubrics, scoring tools, and visualized results
with confidence intervals are publicly released, providing the community with
an interpretable, reproducible benchmark to advance vulnerability-aware LLMs.
All code and data are available at: https://github.com/AfterQuery/vader

### 6. [CoTGuard: Using Chain-of-Thought Triggering for Copyright Protection in Multi-Agent LLM Systems](http://arxiv.org/pdf/2505.19405v1)

Authors: Yan Wen, Junfeng Guo, Heng Huang

As large language models (LLMs) evolve into autonomous agents capable of
collaborative reasoning and task execution, multi-agent LLM systems have
emerged as a powerful paradigm for solving complex problems. However, these
systems pose new challenges for copyright protection, particularly when
sensitive or copyrighted content is inadvertently recalled through inter-agent
communication and reasoning. Existing protection techniques primarily focus on
detecting content in final outputs, overlooking the richer, more revealing
reasoning processes within the agents themselves. In this paper, we introduce
CoTGuard, a novel framework for copyright protection that leverages
trigger-based detection within Chain-of-Thought (CoT) reasoning. Specifically,
we can activate specific CoT segments and monitor intermediate reasoning steps
for unauthorized content reproduction by embedding specific trigger queries
into agent prompts. This approach enables fine-grained, interpretable detection
of copyright violations in collaborative agent scenarios. We evaluate CoTGuard
on various benchmarks in extensive experiments and show that it effectively
uncovers content leakage with minimal interference to task performance. Our
findings suggest that reasoning-level monitoring offers a promising direction
for safeguarding intellectual property in LLM-based agent systems.

### 7. [What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](http://arxiv.org/pdf/2505.19773v1)

Authors: Sangyeop Kim, Yohan Lee, Yongwoo Song, Kimin Lee

We investigate long-context vulnerabilities in Large Language Models (LLMs)
through Many-Shot Jailbreaking (MSJ). Our experiments utilize context length of
up to 128K tokens. Through comprehensive analysis with various many-shot attack
settings with different instruction styles, shot density, topic, and format, we
reveal that context length is the primary factor determining attack
effectiveness. Critically, we find that successful attacks do not require
carefully crafted harmful content. Even repetitive shots or random dummy text
can circumvent model safety measures, suggesting fundamental limitations in
long-context processing capabilities of LLMs. The safety behavior of
well-aligned models becomes increasingly inconsistent with longer contexts.
These findings highlight significant safety gaps in context expansion
capabilities of LLMs, emphasizing the need for new safety mechanisms.

### 8. [Poison in the Well: Feature Embedding Disruption in Backdoor Attacks](http://arxiv.org/pdf/2505.19821v1)

Authors: Zhou Feng, Jiahao Chen, Chunyi Zhou, Yuwen Pu, Qingming Li, Shouling Ji

Backdoor attacks embed malicious triggers into training data, enabling
attackers to manipulate neural network behavior during inference while
maintaining high accuracy on benign inputs. However, existing backdoor attacks
face limitations manifesting in excessive reliance on training data, poor
stealth, and instability, which hinder their effectiveness in real-world
applications. Therefore, this paper introduces ShadowPrint, a versatile
backdoor attack that targets feature embeddings within neural networks to
achieve high ASRs and stealthiness. Unlike traditional approaches, ShadowPrint
reduces reliance on training data access and operates effectively with
exceedingly low poison rates (as low as 0.01%). It leverages a clustering-based
optimization strategy to align feature embeddings, ensuring robust performance
across diverse scenarios while maintaining stability and stealth. Extensive
evaluations demonstrate that ShadowPrint achieves superior ASR (up to 100%),
steady CA (with decay no more than 1% in most cases), and low DDR (averaging
below 5%) across both clean-label and dirty-label settings, and with poison
rates ranging from as low as 0.01% to 0.05%, setting a new standard for
backdoor attack capabilities and emphasizing the need for advanced defense
strategies focused on feature space manipulations.

### 9. [One Surrogate to Fool Them All: Universal, Transferable, and Targeted Adversarial Attacks with CLIP](http://arxiv.org/pdf/2505.19840v1)

Authors: Binyan Xu, Xilin Dai, Di Tang, Kehuan Zhang

Deep Neural Networks (DNNs) have achieved widespread success yet remain prone
to adversarial attacks. Typically, such attacks either involve frequent queries
to the target model or rely on surrogate models closely mirroring the target
model -- often trained with subsets of the target model's training data -- to
achieve high attack success rates through transferability. However, in
realistic scenarios where training data is inaccessible and excessive queries
can raise alarms, crafting adversarial examples becomes more challenging. In
this paper, we present UnivIntruder, a novel attack framework that relies
solely on a single, publicly available CLIP model and publicly available
datasets. By using textual concepts, UnivIntruder generates universal,
transferable, and targeted adversarial perturbations that mislead DNNs into
misclassifying inputs into adversary-specified classes defined by textual
concepts.
  Our extensive experiments show that our approach achieves an Attack Success
Rate (ASR) of up to 85% on ImageNet and over 99% on CIFAR-10, significantly
outperforming existing transfer-based methods. Additionally, we reveal
real-world vulnerabilities, showing that even without querying target models,
UnivIntruder compromises image search engines like Google and Baidu with ASR
rates up to 84%, and vision language models like GPT-4 and Claude-3.5 with ASR
rates up to 80%. These findings underscore the practicality of our attack in
scenarios where traditional avenues are blocked, highlighting the need to
reevaluate security paradigms in AI applications.

### 10. [Evaluating AI cyber capabilities with crowdsourced elicitation](http://arxiv.org/pdf/2505.19915v1)

Authors: Artem Petrov, Dmitrii Volkov

As AI systems become increasingly capable, understanding their offensive
cyber potential is critical for informed governance and responsible deployment.
However, it's hard to accurately bound their capabilities, and some prior
evaluations dramatically underestimated them. The art of extracting maximum
task-specific performance from AIs is called "AI elicitation", and today's
safety organizations typically conduct it in-house. In this paper, we explore
crowdsourcing elicitation efforts as an alternative to in-house elicitation
work.
  We host open-access AI tracks at two Capture The Flag (CTF) competitions: AI
vs. Humans (400 teams) and Cyber Apocalypse_ (4000 teams). The AI teams achieve
outstanding performance at both events, ranking top-13% and top-21%
respectively for a total of \$7500 in bounties. This impressive performance
suggests that open-market elicitation may offer an effective complement to
in-house elicitation. We propose elicitation bounties as a practical mechanism
for maintaining timely, cost-effective situational awareness of emerging AI
capabilities.
  Another advantage of open elicitations is the option to collect human
performance data at scale. Applying METR's methodology, we found that AI agents
can reliably solve cyber challenges requiring one hour or less of effort from a
median human CTF participant.

### Computer Vision and Pattern Recognition

### 1. [DiSa: Directional Saliency-Aware Prompt Learning for Generalizable Vision-Language Models](http://arxiv.org/pdf/2505.19373v1)

Authors: Niloufar Alipour Talemi, Hossein Kashiani, Hossein R. Nowdeh, Fatemeh Afghah

Prompt learning has emerged as a powerful paradigm for adapting
vision-language models such as CLIP to downstream tasks. However, existing
methods often overfit to seen data, leading to significant performance
degradation when generalizing to novel classes or unseen domains. To address
this limitation, we propose DiSa, a Directional Saliency-Aware Prompt Learning
framework that integrates two complementary regularization strategies to
enhance generalization. First, our Cross-Interactive Regularization (CIR)
fosters cross-modal alignment by enabling cooperative learning between prompted
and frozen encoders. Within CIR, a saliency-aware masking strategy guides the
image encoder to prioritize semantically critical image regions, reducing
reliance on less informative patches. Second, we introduce a directional
regularization strategy that aligns visual embeddings with class-wise prototype
features in a directional manner to prioritize consistency in feature
orientation over strict proximity. This approach ensures robust generalization
by leveraging stable prototype directions derived from class-mean statistics.
Extensive evaluations on 11 diverse image classification benchmarks demonstrate
that DiSa consistently outperforms state-of-the-art prompt learning methods
across various settings, including base-to-novel generalization, cross-dataset
transfer, domain generalization, and few-shot learning.

### 2. [Absolute Coordinates Make Motion Generation Easy](http://arxiv.org/pdf/2505.19377v1)

Authors: Zichong Meng, Zeyu Han, Xiaogang Peng, Yiming Xie, Huaizu Jiang

State-of-the-art text-to-motion generation models rely on the
kinematic-aware, local-relative motion representation popularized by HumanML3D,
which encodes motion relative to the pelvis and to the previous frame with
built-in redundancy. While this design simplifies training for earlier
generation models, it introduces critical limitations for diffusion models and
hinders applicability to downstream tasks. In this work, we revisit the motion
representation and propose a radically simplified and long-abandoned
alternative for text-to-motion generation: absolute joint coordinates in global
space. Through systematic analysis of design choices, we show that this
formulation achieves significantly higher motion fidelity, improved text
alignment, and strong scalability, even with a simple Transformer backbone and
no auxiliary kinematic-aware losses. Moreover, our formulation naturally
supports downstream tasks such as text-driven motion control and
temporal/spatial editing without additional task-specific reengineering and
costly classifier guidance generation from control signals. Finally, we
demonstrate promising generalization to directly generate SMPL-H mesh vertices
in motion from text, laying a strong foundation for future research and
motion-related applications.

### 3. [Erasing Concepts, Steering Generations: A Comprehensive Survey of Concept Suppression](http://arxiv.org/pdf/2505.19398v1)

Authors: Yiwei Xie, Ping Liu, Zheng Zhang

Text-to-Image (T2I) models have demonstrated impressive capabilities in
generating high-quality and diverse visual content from natural language
prompts. However, uncontrolled reproduction of sensitive, copyrighted, or
harmful imagery poses serious ethical, legal, and safety challenges. To address
these concerns, the concept erasure paradigm has emerged as a promising
direction, enabling the selective removal of specific semantic concepts from
generative models while preserving their overall utility. This survey provides
a comprehensive overview and in-depth synthesis of concept erasure techniques
in T2I diffusion models. We systematically categorize existing approaches along
three key dimensions: intervention level, which identifies specific model
components targeted for concept removal; optimization structure, referring to
the algorithmic strategies employed to achieve suppression; and semantic scope,
concerning the complexity and nature of the concepts addressed. This
multi-dimensional taxonomy enables clear, structured comparisons across diverse
methodologies, highlighting fundamental trade-offs between erasure specificity,
generalization, and computational complexity. We further discuss current
evaluation benchmarks, standardized metrics, and practical datasets,
emphasizing gaps that limit comprehensive assessment, particularly regarding
robustness and practical effectiveness. Finally, we outline major challenges
and promising future directions, including disentanglement of concept
representations, adaptive and incremental erasure strategies, adversarial
robustness, and new generative architectures. This survey aims to guide
researchers toward safer, more ethically aligned generative models, providing
foundational knowledge and actionable recommendations to advance responsible
development in generative AI.

### 4. [MMIG-Bench: Towards Comprehensive and Explainable Evaluation of Multi-Modal Image Generation Models](http://arxiv.org/pdf/2505.19415v1)

Authors: Hang Hua, Ziyun Zeng, Yizhi Song, Yunlong Tang, Liu He, Daniel Aliaga, Wei Xiong, Jiebo Luo

Recent multimodal image generators such as GPT-4o, Gemini 2.0 Flash, and
Gemini 2.5 Pro excel at following complex instructions, editing images and
maintaining concept consistency. However, they are still evaluated by disjoint
toolkits: text-to-image (T2I) benchmarks that lacks multi-modal conditioning,
and customized image generation benchmarks that overlook compositional
semantics and common knowledge. We propose MMIG-Bench, a comprehensive
Multi-Modal Image Generation Benchmark that unifies these tasks by pairing
4,850 richly annotated text prompts with 1,750 multi-view reference images
across 380 subjects, spanning humans, animals, objects, and artistic styles.
MMIG-Bench is equipped with a three-level evaluation framework: (1) low-level
metrics for visual artifacts and identity preservation of objects; (2) novel
Aspect Matching Score (AMS): a VQA-based mid-level metric that delivers
fine-grained prompt-image alignment and shows strong correlation with human
judgments; and (3) high-level metrics for aesthetics and human preference.
Using MMIG-Bench, we benchmark 17 state-of-the-art models, including Gemini 2.5
Pro, FLUX, DreamBooth, and IP-Adapter, and validate our metrics with 32k human
ratings, yielding in-depth insights into architecture and data design. We will
release the dataset and evaluation code to foster rigorous, unified evaluation
and accelerate future innovations in multi-modal image generation.

### 5. [ADD-SLAM: Adaptive Dynamic Dense SLAM with Gaussian Splatting](http://arxiv.org/pdf/2505.19420v1)

Authors: Wenhua Wu, Chenpeng Su, Siting Zhu, Tianchen Deng, Zhe Liu, Hesheng Wang

Recent advancements in Neural Radiance Fields (NeRF) and 3D Gaussian-based
Simultaneous Localization and Mapping (SLAM) methods have demonstrated
exceptional localization precision and remarkable dense mapping performance.
However, dynamic objects introduce critical challenges by disrupting scene
consistency, leading to tracking drift and mapping artifacts. Existing methods
that employ semantic segmentation or object detection for dynamic
identification and filtering typically rely on predefined categorical priors,
while discarding dynamic scene information crucial for robotic applications
such as dynamic obstacle avoidance and environmental interaction. To overcome
these challenges, we propose ADD-SLAM: an Adaptive Dynamic Dense SLAM framework
based on Gaussian splitting. We design an adaptive dynamic identification
mechanism grounded in scene consistency analysis, comparing geometric and
textural discrepancies between real-time observations and historical maps. Ours
requires no predefined semantic category priors and adaptively discovers scene
dynamics. Precise dynamic object recognition effectively mitigates interference
from moving targets during localization. Furthermore, we propose a
dynamic-static separation mapping strategy that constructs a temporal Gaussian
model to achieve online incremental dynamic modeling. Experiments conducted on
multiple dynamic datasets demonstrate our method's flexible and accurate
dynamic segmentation capabilities, along with state-of-the-art performance in
both localization and mapping.

### 6. [Certainty and Uncertainty Guided Active Domain Adaptation](http://arxiv.org/pdf/2505.19421v1)

Authors: Bardia Safaei, Vibashan VS, Vishal M. Patel

Active Domain Adaptation (ADA) adapts models to target domains by selectively
labeling a few target samples. Existing ADA methods prioritize uncertain
samples but overlook confident ones, which often match ground-truth. We find
that incorporating confident predictions into the labeled set before active
sampling reduces the search space and improves adaptation. To address this, we
propose a collaborative framework that labels uncertain samples while treating
highly confident predictions as ground truth. Our method combines Gaussian
Process-based Active Sampling (GPAS) for identifying uncertain samples and
Pseudo-Label-based Certain Sampling (PLCS) for confident ones, progressively
enhancing adaptation. PLCS refines the search space, and GPAS reduces the
domain gap, boosting the proportion of confident samples. Extensive experiments
on Office-Home and DomainNet show that our approach outperforms
state-of-the-art ADA methods.

### 7. [LlamaSeg: Image Segmentation via Autoregressive Mask Generation](http://arxiv.org/pdf/2505.19422v1)

Authors: Jiru Deng, Tengjin Weng, Tianyu Yang, Wenhan Luo, Zhiheng Li, Wenhao Jiang

We present LlamaSeg, a visual autoregressive framework that unifies multiple
image segmentation tasks via natural language instructions. We reformulate
image segmentation as a visual generation problem, representing masks as
"visual" tokens and employing a LLaMA-style Transformer to predict them
directly from image inputs. By adhering to the next-token prediction paradigm,
our approach naturally integrates segmentation tasks into autoregressive
architectures. To support large-scale training, we introduce a data annotation
pipeline and construct the SA-OVRS dataset, which contains 2M segmentation
masks annotated with over 5,800 open-vocabulary labels or diverse textual
descriptions, covering a wide spectrum of real-world scenarios. This enables
our model to localize objects in images based on text prompts and to generate
fine-grained masks. To more accurately evaluate the quality of masks produced
by visual generative models, we further propose a composite metric that
combines Intersection over Union (IoU) with Average Hausdorff Distance (AHD),
offering a more precise assessment of contour fidelity. Experimental results
demonstrate that our method surpasses existing generative models across
multiple datasets and yields more detailed segmentation masks.

### 8. [SpikeStereoNet: A Brain-Inspired Framework for Stereo Depth Estimation from Spike Streams](http://arxiv.org/pdf/2505.19487v1)

Authors: Zhuoheng Gao, Yihao Li, Jiyao Zhang, Rui Zhao, Tong Wu, Hao Tang, Zhaofei Yu, Hao Dong, Guozhang Chen, Tiejun Huang

Conventional frame-based cameras often struggle with stereo depth estimation
in rapidly changing scenes. In contrast, bio-inspired spike cameras emit
asynchronous events at microsecond-level resolution, providing an alternative
sensing modality. However, existing methods lack specialized stereo algorithms
and benchmarks tailored to the spike data. To address this gap, we propose
SpikeStereoNet, a brain-inspired framework and the first to estimate stereo
depth directly from raw spike streams. The model fuses raw spike streams from
two viewpoints and iteratively refines depth estimation through a recurrent
spiking neural network (RSNN) update module. To benchmark our approach, we
introduce a large-scale synthetic spike stream dataset and a real-world stereo
spike dataset with dense depth annotations. SpikeStereoNet outperforms existing
methods on both datasets by leveraging spike streams' ability to capture subtle
edges and intensity shifts in challenging regions such as textureless surfaces
and extreme lighting conditions. Furthermore, our framework exhibits strong
data efficiency, maintaining high accuracy even with substantially reduced
training data. The source code and datasets will be publicly available.

### 9. [ViewCraft3D: High-Fidelity and View-Consistent 3D Vector Graphics Synthesis](http://arxiv.org/pdf/2505.19492v1)

Authors: Chuang Wang, Haitao Zhou, Ling Luo, Qian Yu

3D vector graphics play a crucial role in various applications including 3D
shape retrieval, conceptual design, and virtual reality interactions due to
their ability to capture essential structural information with minimal
representation. While recent approaches have shown promise in generating 3D
vector graphics, they often suffer from lengthy processing times and struggle
to maintain view consistency. To address these limitations, we propose
ViewCraft3D (VC3D), an efficient method that leverages 3D priors to generate 3D
vector graphics. Specifically, our approach begins with 3D object analysis,
employs a geometric extraction algorithm to fit 3D vector graphics to the
underlying structure, and applies view-consistent refinement process to enhance
visual quality. Our comprehensive experiments demonstrate that VC3D outperforms
previous methods in both qualitative and quantitative evaluations, while
significantly reducing computational overhead. The resulting 3D sketches
maintain view consistency and effectively capture the essential characteristics
of the original objects.

### 10. [The Role of Video Generation in Enhancing Data-Limited Action Understanding](http://arxiv.org/pdf/2505.19495v1)

Authors: Wei Li, Dezhao Luo, Dongbao Yang, Zhenhang Li, Weiping Wang, Yu Zhou

Video action understanding tasks in real-world scenarios always suffer data
limitations. In this paper, we address the data-limited action understanding
problem by bridging data scarcity. We propose a novel method that employs a
text-to-video diffusion transformer to generate annotated data for model
training. This paradigm enables the generation of realistic annotated data on
an infinite scale without human intervention. We proposed the information
enhancement strategy and the uncertainty-based label smoothing tailored to
generate sample training. Through quantitative and qualitative analysis, we
observed that real samples generally contain a richer level of information than
generated samples. Based on this observation, the information enhancement
strategy is proposed to enhance the informative content of the generated
samples from two aspects: the environments and the characters. Furthermore, we
observed that some low-quality generated samples might negatively affect model
training. To address this, we devised the uncertainty-based label smoothing
strategy to increase the smoothing of these samples, thus reducing their
impact. We demonstrate the effectiveness of the proposed method on four
datasets across five tasks and achieve state-of-the-art performance for
zero-shot action recognition.

### Computers and Society

### 1. [Recalibrating the Compass: Integrating Large Language Models into Classical Research Methods](http://arxiv.org/pdf/2505.19402v1)

Authors: Tai-Quan Peng, Xuzhen Yang

This paper examines how large language models (LLMs) are transforming core
quantitative methods in communication research in particular, and in the social
sciences more broadly-namely, content analysis, survey research, and
experimental studies. Rather than replacing classical approaches, LLMs
introduce new possibilities for coding and interpreting text, simulating
dynamic respondents, and generating personalized and interactive stimuli.
Drawing on recent interdisciplinary work, the paper highlights both the
potential and limitations of LLMs as research tools, including issues of
validity, bias, and interpretability. To situate these developments
theoretically, the paper revisits Lasswell's foundational framework -- "Who
says what, in which channel, to whom, with what effect?" -- and demonstrates
how LLMs reconfigure message studies, audience analysis, and effects research
by enabling interpretive variation, audience trajectory modeling, and
counterfactual experimentation. Revisiting the metaphor of the methodological
compass, the paper argues that classical research logics remain essential as
the field integrates LLMs and generative AI. By treating LLMs not only as
technical instruments but also as epistemic and cultural tools, the paper calls
for thoughtful, rigorous, and imaginative use of LLMs in future communication
and social science research.

### 2. [EuroCon: Benchmarking Parliament Deliberation for Political Consensus Finding](http://arxiv.org/pdf/2505.19558v1)

Authors: Zhaowei Zhang, Minghua Yi, Mengmeng Wang, Fengshuo Bai, Zilong Zheng, Yipeng Kang, Yaodong Yang

Achieving political consensus is crucial yet challenging for the effective
functioning of social governance. However, although frontier AI systems
represented by large language models (LLMs) have developed rapidly in recent
years, their capabilities on this scope are still understudied. In this paper,
we introduce EuroCon, a novel benchmark constructed from 2,225 high-quality
deliberation records of the European Parliament over 13 years, ranging from
2009 to 2022, to evaluate the ability of LLMs to reach political consensus
among divergent party positions across diverse parliament settings.
Specifically, EuroCon incorporates four factors to build each simulated
parliament setting: specific political issues, political goals, participating
parties, and power structures based on seat distribution. We also develop an
evaluation framework for EuroCon to simulate real voting outcomes in different
parliament settings, assessing whether LLM-generated resolutions meet
predefined political goals. Our experimental results demonstrate that even
state-of-the-art models remain undersatisfied with complex tasks like passing
resolutions by a two-thirds majority and addressing security issues, while
revealing some common strategies LLMs use to find consensus under different
power structures, such as prioritizing the stance of the dominant party,
highlighting EuroCon's promise as an effective platform for studying LLMs'
ability to find political consensus.

### 3. [Fairness Practices in Industry: A Case Study in Machine Learning Teams Building Recommender Systems](http://arxiv.org/pdf/2505.19441v1)

Authors: Jing Nathan Yan, Junxiong Wang, Jeffrey M. Rzeszotarski, Allison Koenecke

The rapid proliferation of recommender systems necessitates robust fairness
practices to address inherent biases. Assessing fairness, though, is
challenging due to constantly evolving metrics and best practices. This paper
analyzes how industry practitioners perceive and incorporate these changing
fairness standards in their workflows. Through semi-structured interviews with
11 practitioners from technical teams across a range of large technology
companies, we investigate industry implementations of fairness in
recommendation system products. We focus on current debiasing practices,
applied metrics, collaborative strategies, and integrating academic research
into practice. Findings show a preference for multi-dimensional debiasing over
traditional demographic methods, and a reliance on intuitive rather than
academic metrics. This study also highlights the difficulties in balancing
fairness with both the practitioner's individual (bottom-up) roles and
organizational (top-down) workplace constraints, including the interplay with
legal and compliance experts. Finally, we offer actionable recommendations for
the recommender system community and algorithmic fairness practitioners,
underlining the need to refine fairness practices continually.

### 4. [AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection](http://arxiv.org/pdf/2505.19528v1)

Authors: Yejin Lee, Joonghyuk Hahn, Hyeseon Ahn, Yo-Sub Han

Implicit hate speech detection is challenging due to its subtlety and
reliance on contextual interpretation rather than explicit offensive words.
Current approaches rely on contrastive learning, which are shown to be
effective on distinguishing hate and non-hate sentences. Humans, however,
detect implicit hate speech by first identifying specific targets within the
text and subsequently interpreting how these target relate to their surrounding
context. Motivated by this reasoning process, we propose AmpleHate, a novel
approach designed to mirror human inference for implicit hate detection.
AmpleHate identifies explicit target using a pretrained Named Entity
Recognition model and capture implicit target information via [CLS] tokens. It
computes attention-based relationships between explicit, implicit targets and
sentence context and then, directly injects these relational vectors into the
final sentence representation. This amplifies the critical signals of
target-context relations for determining implicit hate. Experiments demonstrate
that AmpleHate achieves state-of-the-art performance, outperforming contrastive
learning baselines by an average of 82.14% and achieve faster convergence.
Qualitative analyses further reveal that attention patterns produced by
AmpleHate closely align with human judgement, underscoring its interpretability
and robustness.

### 5. [Exploring Consciousness in LLMs: A Systematic Survey of Theories, Implementations, and Frontier Risks](http://arxiv.org/pdf/2505.19806v1)

Authors: Sirui Chen, Shuqin Ma, Shu Yu, Hanwang Zhang, Shengjie Zhao, Chaochao Lu

Consciousness stands as one of the most profound and distinguishing features
of the human mind, fundamentally shaping our understanding of existence and
agency. As large language models (LLMs) develop at an unprecedented pace,
questions concerning intelligence and consciousness have become increasingly
significant. However, discourse on LLM consciousness remains largely unexplored
territory. In this paper, we first clarify frequently conflated terminologies
(e.g., LLM consciousness and LLM awareness). Then, we systematically organize
and synthesize existing research on LLM consciousness from both theoretical and
empirical perspectives. Furthermore, we highlight potential frontier risks that
conscious LLMs might introduce. Finally, we discuss current challenges and
outline future directions in this emerging field. The references discussed in
this paper are organized at
https://github.com/OpenCausaLab/Awesome-LLM-Consciousness.

### 6. [Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents](http://arxiv.org/pdf/2505.19997v1)

Authors: Tao Wu, Jingyuan Chen, Wang Lin, Mengze Li, Yumeng Zhu, Ang Li, Kun Kuang, Fei Wu

Large language models (LLMs) are revolutionizing education, with LLM-based
agents playing a key role in simulating student behavior. A major challenge in
student simulation is modeling the diverse learning patterns of students at
various cognitive levels. However, current LLMs, typically trained as ``helpful
assistants'', target at generating perfect responses. As a result, they
struggle to simulate students with diverse cognitive abilities, as they often
produce overly advanced answers, missing the natural imperfections that
characterize student learning and resulting in unrealistic simulations. To
address this issue, we propose a training-free framework for student
simulation. We begin by constructing a cognitive prototype for each student
using a knowledge graph, which captures their understanding of concepts from
past learning records. This prototype is then mapped to new tasks to predict
student performance. Next, we simulate student solutions based on these
predictions and iteratively refine them using a beam search method to better
replicate realistic mistakes. To validate our approach, we construct the
\texttt{Student\_100} dataset, consisting of $100$ students working on Python
programming and $5,000$ learning records. Experimental results show that our
method consistently outperforms baseline models, achieving $100\%$ improvement
in simulation accuracy.

### 7. [Community Moderation and the New Epistemology of Fact Checking on Social Media](http://arxiv.org/pdf/2505.20067v1)

Authors: Isabelle Augenstein, Michiel Bakker, Tanmoy Chakraborty, David Corney, Emilio Ferrara, Iryna Gurevych, Scott Hale, Eduard Hovy, Heng Ji, Irene Larraz, Filippo Menczer, Preslav Nakov, Paolo Papotti, Dhruv Sahnan, Greta Warren, Giovanni Zagni

Social media platforms have traditionally relied on internal moderation teams
and partnerships with independent fact-checking organizations to identify and
flag misleading content. Recently, however, platforms including X (formerly
Twitter) and Meta have shifted towards community-driven content moderation by
launching their own versions of crowd-sourced fact-checking -- Community Notes.
If effectively scaled and governed, such crowd-checking initiatives have the
potential to combat misinformation with increased scale and speed as
successfully as community-driven efforts once did with spam. Nevertheless,
general content moderation, especially for misinformation, is inherently more
complex. Public perceptions of truth are often shaped by personal biases,
political leanings, and cultural contexts, complicating consensus on what
constitutes misleading content. This suggests that community efforts, while
valuable, cannot replace the indispensable role of professional fact-checkers.
Here we systemically examine the current approaches to misinformation detection
across major platforms, explore the emerging role of community-driven
moderation, and critically evaluate both the promises and challenges of
crowd-checking at scale.

### Databases

### 1. [Curation and Analysis of MIMICEL -- An Event Log for MIMIC-IV Emergency Department](http://arxiv.org/pdf/2505.19389v1)

Authors: Jia Wei, Chun Ouyang, Bemali Wickramanayake, Zhipeng He, Keshara Perera, Catarina Moreira

The global issue of overcrowding in emergency departments (ED) necessitates
the analysis of patient flow through ED to enhance efficiency and alleviate
overcrowding. However, traditional analytical methods are time-consuming and
costly. The healthcare industry is embracing process mining tools to analyse
healthcare processes and patient flows. Process mining aims to discover,
monitor, and enhance processes by obtaining knowledge from event log data.
However, the availability of event logs is a prerequisite for applying process
mining techniques. Hence, this paper aims to generate an event log for
analysing processes in ED. In this study, we extract an event log from the
MIMIC-IV-ED dataset and name it MIMICEL. MIMICEL captures the process of
patient journey in ED, allowing for analysis of patient flows and improving ED
efficiency. We present analyses conducted using MIMICEL to demonstrate the
utility of the dataset. The curation of MIMICEL facilitates extensive use of
MIMIC-IV-ED data for ED analysis using process mining techniques, while also
providing the process mining research communities with a valuable dataset for
study.

### 2. [Adaptive Indexing for Approximate Query Processing in Exploratory Data Analysis](http://arxiv.org/pdf/2505.19872v1)

Authors: Stavros Maroulis, Nikos Bikakis, Vassilis Stamatopoulos, George Papastefanatos

Minimizing data-to-analysis time while enabling real-time interaction and
efficient analytical computations on large datasets are fundamental objectives
of contemporary exploratory systems. Although some of the recent adaptive
indexing and on-the-fly processing approaches address most of these needs,
there are cases, where they do not always guarantee reliable performance. Some
examples of such cases include: exploring areas with a high density of objects;
executing the first exploratory queries or exploring previously unseen areas
(where the index has not yet adapted sufficiently); and working with very large
data files on commodity hardware, such as low-specification laptops. In such
demanding cases, approximate and incremental techniques can be exploited to
ensure efficiency and scalability by allowing users to prioritize response time
over result accuracy, acknowledging that exact results are not always
necessary. Therefore, approximation mechanisms that enable smooth user
interaction by defining the trade-off between accuracy and performance based on
vital factors (e.g., task, preferences, available resources) are of great
importance. Considering the aforementioned, in this work, we present an
adaptive approximate query processing framework for interactive on-the-fly
analysis (with out a preprocessing phase) over large raw data. The core
component of the framework is a main-memory adaptive indexing scheme
(VALINOR-A) that interoperates with user-driven sampling and incremental
aggregation computations. Additionally, an effective error-bounded
approximation strategy is designed and integrated in the query processing
process. We conduct extensive experiments using both real and synthetic
datasets, demonstrating the efficiency and effectiveness of the proposed
framework.

### 3. [A Unified Architecture for Efficient Binary and Worst-Case Optimal Join Processing](http://arxiv.org/pdf/2505.19918v1)

Authors: Amirali Kaboli, Alex Mascolo, Amir Shaikhha

Join processing is a fundamental operation in database management systems;
however, traditional join algorithms often encounter efficiency challenges when
dealing with complex queries that produce intermediate results much larger than
the final query output. The emergence of worst-case optimal join (WCOJ)
algorithms represents a significant advancement, offering asymptotically better
performance by avoiding the enumeration of potentially exploding intermediate
results. In this paper, we propose a unified architecture that efficiently
supports both traditional binary joins and WCOJ processing. As opposed to the
state-of-the-art, which only focuses on either hash-based or sort-based join
implementations, our system accommodates both physical implementations of
binary joins and WCOJ algorithms. Experimental evaluations demonstrate that our
system achieves performance gains of up to 3.1x (on average 1.5x) and 4.8x (on
average 1.4x) over the state-of-the-art implementation of Generic Join and Free
Join methods, respectively, across acyclic and cyclic queries in standard query
benchmarks.

### 4. [Automatic Metadata Extraction for Text-to-SQL](http://arxiv.org/pdf/2505.19988v1)

Authors: Vladislav Shkapenyuk, Divesh Srivastava, Theodore Johnson, Parisa Ghane

Large Language Models (LLMs) have recently become sophisticated enough to
automate many tasks ranging from pattern finding to writing assistance to code
generation. In this paper, we examine text-to-SQL generation. We have observed
from decades of experience that the most difficult part of query development
lies in understanding the database contents. These experiences inform the
direction of our research.
  Text-to-SQL benchmarks such as SPIDER and Bird contain extensive metadata
that is generally not available in practice. Human-generated metadata requires
the use of expensive Subject Matter Experts (SMEs), who are often not fully
aware of many aspects of their databases. In this paper, we explore techniques
for automatic metadata extraction to enable text-to-SQL generation.
  Ee explore the use of two standard and one newer metadata extraction
techniques: profiling, query log analysis, and SQL-to text generation using an
LLM. We use BIRD benchmark [JHQY+23] to evaluate the effectiveness of these
techniques. BIRD does not provide query logs on their test database, so we
prepared a submission that uses profiling alone, and does not use any specially
tuned model (we used GPT-4o). From Sept 1 to Sept 23, 2024, and Nov 11 through
Nov 23, 2024 we achieved the highest score both with and without using the
"oracle" information provided with the question set. We regained the number 1
spot on Mar 11, 2025, and are still at #1 at the time of the writing (May,
2025).

### 5. [Foundation Models for Tabular Data within Systemic Contexts Need Grounding](http://arxiv.org/pdf/2505.19825v1)

Authors: Tassilo Klein, Johannes Hoffart

Current research on tabular foundation models often overlooks the
complexities of large-scale, real-world data by treating tables as isolated
entities and assuming information completeness, thereby neglecting the vital
operational context. To address this, we introduce the concept of Semantically
Linked Tables (SLT), recognizing that tables are inherently connected to both
declarative and procedural operational knowledge. We propose Foundation Models
for Semantically Linked Tables (FMSLT), which integrate these components to
ground tabular data within its true operational context. This comprehensive
representation unlocks the full potential of machine learning for complex,
interconnected tabular data across diverse domains. Realizing FMSLTs requires
access to operational knowledge that is often unavailable in public datasets,
highlighting the need for close collaboration between domain experts and
researchers. Our work exposes the limitations of current tabular foundation
models and proposes a new direction centered on FMSLTs, aiming to advance
robust, context-aware models for structured data.

### 6. [TUNA: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](http://arxiv.org/pdf/2505.20124v1)

Authors: Fanheng Kong, Jingyuan Zhang, Hongzhi Zhang, Shi Feng, Daling Wang, Linhao Yu, Xingguang Ji, Yu Tian, Qi Wang, Fuzheng Zhang

Videos are unique in their integration of temporal elements, including
camera, scene, action, and attribute, along with their dynamic relationships
over time. However, existing benchmarks for video understanding often treat
these properties separately or narrowly focus on specific aspects, overlooking
the holistic nature of video content. To address this, we introduce TUNA, a
temporal-oriented benchmark for fine-grained understanding on dense dynamic
videos, with two complementary tasks: captioning and QA. Our TUNA features
diverse video scenarios and dynamics, assisted by interpretable and robust
evaluation criteria. We evaluate several leading models on our benchmark,
providing fine-grained performance assessments across various dimensions. This
evaluation reveals key challenges in video temporal understanding, such as
limited action description, inadequate multi-subject understanding, and
insensitivity to camera motion, offering valuable insights for improving video
understanding models. The data and code are available at
https://friedrichor.github.io/projects/TUNA.

### Distributed, Parallel, and Cluster Computing

### 1. [GPU acceleration of non-equilibrium Green's function calculation using OpenACC and CUDA FORTRAN](http://arxiv.org/pdf/2505.19467v1)

Authors: Jia Yin, Khaled Z. Ibrahim, Mauro Del Ben, Jack Deslippe, Yang-hao Chan, Chao Yang

The numerical solution of the Kadanoff-Baym nonlinear integro-differential
equations, which yields the non-equilibrium Green's functions (NEGFs) of
quantum many-body systems, poses significant computational challenges due to
its high computational complexity. In this work, we present efficient
implementations of a numerical method for solving these equations on
distributed-memory architectures, including many-core CPUs and multi-GPU
systems. For CPU-based platforms, we adopt a hybrid MPI/OpenMP programming
model to exploit both inter-node and intra-node parallelism. On GPU-accelerated
systems, we implement the method using two distinct approaches: MPI/OpenACC and
MPI/CUDA FORTRAN. Several optimization strategies are employed to enhance GPU
performance, including techniques to maximize computational resource
utilization and minimize the overhead associated with kernel launches and
memory management. Although OpenACC is easy to use, CUDA FORTRAN provides more
advanced features for configuring and managing multiple levels of concurrency,
while also simplifying memory allocation and data movement between host and
device. This flexibility translates into significant performance improvements.
We compare the performance of the three implementations and demonstrate that
the GPU-based approaches achieve substantial speedups over CPU-based
implementations. Furthermore, both CPU and GPU versions exhibit excellent
strong and weak scaling, confirming the scalability and efficiency of our
approach for large-scale NEGF computations.

### 2. [Justin: Hybrid CPU/Memory Elastic Scaling for Distributed Stream Processing](http://arxiv.org/pdf/2505.19739v1)

Authors: Donatien Schmitz, Guillaume Rosinosky, Etienne Rivière

Distributed Stream Processing (DSP) engines analyze continuous data via
queries expressed as a graph of operators. Auto-scalers adjust the number of
parallel instances of these operators to support a target rate. Current
auto-scalers couple CPU and memory scaling, allocating resources as
one-size-fits-all packages. This contrasts with operators' high diversity of
requirements. We present Justin, an auto-scaler that enables hybrid CPU and
memory scaling of DSP operators. Justin monitors both CPU usage and the
performance of operators' storage operations. Its mechanisms enable finegrain
memory allocation for tasks upon a query reconfiguration. The Justin policy
identifies individual operators' memory pressure and decides between adjusting
parallelism and/or memory assignment. We implement Justin in Apache Flink,
extending the Flink Kubernetes Operator and the DS2 CPU-only auto-scaler. Using
the Nexmark benchmark, our evaluation shows that Justin identifies suitable
resource allocation in as many or fewer reconfiguration steps as DS2 and
supports a target rate with significantly fewer CPU and memory resources.

### 3. [From Few to Many Faults: Adaptive Byzantine Agreement with Optimal Communication](http://arxiv.org/pdf/2505.19989v1)

Authors: Andrei Constantinescu, Marc Dufay, Anton Paramonov, Roger Wattenhofer

Achieving agreement among distributed parties is a fundamental task in modern
systems, underpinning applications such as consensus in blockchains,
coordination in cloud infrastructure, and fault tolerance in critical services.
However, this task can be communication-intensive, often requiring a large
number of messages to be exchanged, especially in the presence of Byzantine
faults, making efficiency a central challenge in the design of practical
agreement protocols.
  In this paper, we study the problem of Strong Byzantine Agreement and
establish tight upper and lower bounds on communication complexity,
parameterized by the actual number of Byzantine faults. Specifically, for a
system of $n$ parties tolerating up to $t$ Byzantine faults, out of which only
$f \leq t$ are actually faulty, we obtain the following results:
  In the partially synchronous setting, we present the first Byzantine
Agreement protocol that achieves adaptive communication complexity of
$\mathcal{O}(n + t \cdot f)$ words, which is asymptotically optimal. Our
protocol has an optimal resilience of $t < n/3$.
  In the asynchronous setting, we prove a lower bound of $\Omega(n + t^2)$ on
the expected number of messages, and design an almost matching protocol with an
optimal resilience that solves agreement with $\mathcal{O}((n + t^2)\cdot \log
n)$ words. Our main technical contribution in the asynchronous setting is the
utilization of a bipartite expander graph that allows for low-cost information
dissemination.

### 4. [DGRAG: Distributed Graph-based Retrieval-Augmented Generation in Edge-Cloud Systems](http://arxiv.org/pdf/2505.19847v1)

Authors: Wenqing Zhou, Yuxuan Yan, Qianqian Yang

Retrieval-Augmented Generation (RAG) has emerged as a promising approach to
enhance the capabilities of language models by integrating external knowledge.
Due to the diversity of data sources and the constraints of memory and
computing resources, real-world data is often scattered in multiple devices.
Conventional RAGs that store massive amounts of scattered data centrally face
increasing privacy concerns and high computational costs. Additionally, RAG in
a central node raises latency issues when searching over a large-scale
knowledge base. To address these challenges, we propose a distributed Knowledge
Graph-based RAG approach, referred to as DGRAG, in an edge-cloud system, where
each edge device maintains a local knowledge base without the need to share it
with the cloud, instead sharing only summaries of its knowledge. Specifically,
DGRAG has two main phases. In the Distributed Knowledge Construction phase,
DGRAG organizes local knowledge using knowledge graphs, generating subgraph
summaries and storing them in a summary database in the cloud as information
sharing. In the Collaborative Retrieval and Generation phase, DGRAG first
performs knowledge retrieval and answer generation locally, and a gate
mechanism determines whether the query is beyond the scope of local knowledge
or processing capabilities. For queries that exceed the local knowledge scope,
the cloud retrieves knowledge from the most relevant edges based on the
summaries and generates a more precise answer. Experimental results demonstrate
the effectiveness of the proposed DGRAG approach in significantly improving the
quality of question-answering tasks over baseline approaches.

### 5. [Universal Workers: A Vision for Eliminating Cold Starts in Serverless Computing](http://arxiv.org/pdf/2505.19880v1)

Authors: Saman Akbari, Manfred Hauswirth

Serverless computing enables developers to deploy code without managing
infrastructure, but suffers from cold start overhead when initializing new
function instances. Existing solutions such as "keep-alive" or "pre-warming"
are costly and unreliable under bursty workloads. We propose universal workers,
which are computational units capable of executing any function with minimal
initialization overhead. Based on an analysis of production workload traces,
our key insight is that requests in Function-as-a-Service (FaaS) platforms show
a highly skewed distribution, with most requests invoking a small subset of
functions. We exploit this observation to approximate universal workers through
locality groups and three-tier caching (handler, install, import). With this
work, we aim to enable more efficient and scalable FaaS platforms capable of
handling diverse workloads with minimal initialization overhead.

### 6. [Optimizing edge AI models on HPC systems with the edge in the loop](http://arxiv.org/pdf/2505.19995v1)

Authors: Marcel Aach, Cyril Blanc, Andreas Lintermann, Kurt De Grave

Artificial intelligence and machine learning models deployed on edge devices,
e.g., for quality control in Additive Manufacturing (AM), are frequently small
in size. Such models usually have to deliver highly accurate results within a
short time frame. Methods that are commonly employed in literature start out
with larger trained models and try to reduce their memory and latency footprint
by structural pruning, knowledge distillation, or quantization. It is, however,
also possible to leverage hardware-aware Neural Architecture Search (NAS), an
approach that seeks to systematically explore the architecture space to find
optimized configurations. In this study, a hardware-aware NAS workflow is
introduced that couples an edge device located in Belgium with a powerful
High-Performance Computing system in Germany, to train possible architecture
candidates as fast as possible while performing real-time latency measurements
on the target hardware. The approach is verified on a use case in the AM
domain, based on the open RAISE-LPBF dataset, achieving ~8.8 times faster
inference speed while simultaneously enhancing model quality by a factor of
~1.35, compared to a human-designed baseline.

### 7. [Win Fast or Lose Slow: Balancing Speed and Accuracy in Latency-Sensitive Decisions of LLMs](http://arxiv.org/pdf/2505.19481v1)

Authors: Hao Kang, Qingru Zhang, Han Cai, Weiyuan Xu, Tushar Krishna, Yilun Du, Tsachy Weissman

Large language models (LLMs) have shown remarkable performance across diverse
reasoning and generation tasks, and are increasingly deployed as agents in
dynamic environments such as code generation and recommendation systems.
However, many real-world applications, such as high-frequency trading and
real-time competitive gaming, require decisions under strict latency
constraints, where faster responses directly translate into higher rewards.
Despite the importance of this latency quality trade off, it remains
underexplored in the context of LLM based agents. In this work, we present the
first systematic study of this trade off in real time decision making tasks. To
support our investigation, we introduce two new benchmarks: HFTBench, a high
frequency trading simulation, and StreetFighter, a competitive gaming platform.
Our analysis reveals that optimal latency quality balance varies by task, and
that sacrificing quality for lower latency can significantly enhance downstream
performance. To address this, we propose FPX, an adaptive framework that
dynamically selects model size and quantization level based on real time
demands. Our method achieves the best performance on both benchmarks, improving
win rate by up to 80% in Street Fighter and boosting daily yield by up to
26.52% in trading, underscoring the need for latency aware evaluation and
deployment strategies for LLM based agents. These results demonstrate the
critical importance of latency aware evaluation and deployment strategies for
real world LLM based agents. Our benchmarks are available at Latency Sensitive
Benchmarks.

### 8. [Mosaic: Data-Free Knowledge Distillation via Mixture-of-Experts for Heterogeneous Distributed Environments](http://arxiv.org/pdf/2505.19699v1)

Authors: Junming Liu, Yanting Gao, Siyuan Meng, Yifei Sun, Aoqi Wu, Yufei Jin, Yirong Chen, Ding Wang, Guosun Zeng

Federated Learning (FL) is a decentralized machine learning paradigm that
enables clients to collaboratively train models while preserving data privacy.
However, the coexistence of model and data heterogeneity gives rise to
inconsistent representations and divergent optimization dynamics across
clients, ultimately hindering robust global performance. To transcend these
challenges, we propose Mosaic, a novel data-free knowledge distillation
framework tailored for heterogeneous distributed environments. Mosaic first
trains local generative models to approximate each client's personalized
distribution, enabling synthetic data generation that safeguards privacy
through strict separation from real data. Subsequently, Mosaic forms a
Mixture-of-Experts (MoE) from client models based on their specialized
knowledge, and distills it into a global model using the generated data. To
further enhance the MoE architecture, Mosaic integrates expert predictions via
a lightweight meta model trained on a few representative prototypes. Extensive
experiments on standard image classification benchmarks demonstrate that Mosaic
consistently outperforms state-of-the-art approaches under both model and data
heterogeneity. The source code has been published at
https://github.com/Wings-Of-Disaster/Mosaic.

### 9. [Differential Privacy Analysis of Decentralized Gossip Averaging under Varying Threat Models](http://arxiv.org/pdf/2505.19969v1)

Authors: Antti Koskela, Tejas Kulkarni

Fully decentralized training of machine learning models offers significant
advantages in scalability, robustness, and fault tolerance. However, achieving
differential privacy (DP) in such settings is challenging due to the absence of
a central aggregator and varying trust assumptions among nodes. In this work,
we present a novel privacy analysis of decentralized gossip-based averaging
algorithms with additive node-level noise, both with and without secure
summation over each node's direct neighbors. Our main contribution is a new
analytical framework based on a linear systems formulation that accurately
characterizes privacy leakage across these scenarios. This framework
significantly improves upon prior analyses, for example, reducing the R\'enyi
DP parameter growth from $O(T^2)$ to $O(T)$, where $T$ is the number of
training rounds. We validate our analysis with numerical results demonstrating
superior DP bounds compared to existing approaches. We further illustrate our
analysis with a logistic regression experiment on MNIST image classification in
a fully decentralized setting, demonstrating utility comparable to central
aggregation methods.

### Digital Libraries

### 1. [SCIRGC: Multi-Granularity Citation Recommendation and Citation Sentence Preference Alignment](http://arxiv.org/pdf/2505.20103v1)

Authors: Xiangyu Li, Jingqiang Chen

Citations are crucial in scientific research articles as they highlight the
connection between the current study and prior work. However, this process is
often time-consuming for researchers. In this study, we propose the SciRGC
framework, which aims to automatically recommend citation articles and generate
citation sentences for citation locations within articles. The framework
addresses two key challenges in academic citation generation: 1) how to
accurately identify the author's citation intent and find relevant citation
papers, and 2) how to generate high-quality citation sentences that align with
human preferences. We enhance citation recommendation accuracy in the citation
article recommendation module by incorporating citation networks and sentiment
intent, and generate reasoning-based citation sentences in the citation
sentence generation module by using the original article abstract, local
context, citation intent, and recommended articles as inputs. Additionally, we
propose a new evaluation metric to fairly assess the quality of generated
citation sentences. Through comparisons with baseline models and ablation
experiments, the SciRGC framework not only improves the accuracy and relevance
of citation recommendations but also ensures the appropriateness of the
generated citation sentences in context, providing a valuable tool for
interdisciplinary researchers.

### Discrete Mathematics

### 1. [Density Decomposition in Dual-Modular Optimization: Markets, Fairness, and Contracts](http://arxiv.org/pdf/2505.19499v1)

Authors: T-H. Hubert Chan, Shinuo Ma

We study a unified framework for optimization problems defined on
dual-modular instances, where the input comprises a finite ground set $V$ and
two set functions: a monotone supermodular reward function $\f$ and a strictly
monotone submodular cost function $\g$. This abstraction captures and
generalizes classical models in economics and combinatorial optimization,
including submodular utility allocation (SUA) markets and combinatorial
contracts. At the core of our framework is the notion of density decomposition,
which extends classical results to the dual-modular setting and uncovers
structural insights into fairness and optimality.
  We show that the density decomposition yields a canonical vector of
reward-to-cost ratios (densities) that simultaneously characterizes market
equilibria, fair allocations -- via both lexicographic optimality and local
maximin conditions -- and best-response strategies in contract design. Our main
result proves the equivalence of these fairness notions and guarantees the
existence of allocations that realize the decomposition densities.
  Our technical contributions include the analysis of a broad family of convex
programs -- parameterized by divergences such as quadratic, logarithmic, and
hockey-stick functions -- whose minimizers recover the density decomposition.
We prove that any strictly convex divergence yields the same canonical density
vector, and that locally maximin allocations act as universal minimizers for
all divergences satisfying the data processing inequality.
  As an application of our framework, we determine the structure and number of
critical values in the combinatorial contracts problem. Additionally, we
generalize a Frank-Wolfe-type iterative method for approximating the
dual-modular density decomposition, establishing both convergence guarantees
and practical potential through efficient gradient oracle design.

### 2. [Pathographs and some (un)decidability results](http://arxiv.org/pdf/2505.19871v1)

Authors: Daniel Carter, Nicolas Trotignon

We introduce pathographs as a framework to study graph classes defined by
forbidden structures, including forbidding induced subgraphs, minors, etc.
Pathographs approximately generalize s-graphs of
L\'ev\^eque--Lin--Maffray--Trotignon by the addition of two extra adjacency
relations: one between subdivisible edges and vertices called spokes, and one
between pairs of subdivisible edges called rungs. We consider the following
decision problem: given a pathograph $\mathfrak{H}$ and a finite set of
pathographs $\mathcal{F}$, is there an $\mathcal{F}$-free realization of
$\mathfrak{H}$? This may be regarded as a generalization of the "graph class
containment problem": given two graph classes $S$ and $S'$, is it the case that
$S\subseteq S'$? We prove the pathograph realization problem is undecidable in
general, but it is decidable in the case that $\mathfrak{H}$ has no rungs (but
may have spokes), or if $\mathcal{F}$ is closed under adding edges, spokes, and
rungs. We also discuss some potential applications to proving decomposition
theorems.

### 3. [Bounding Width on Graph Classes of Constant Diameter](http://arxiv.org/pdf/2505.19926v1)

Authors: Konrad K. Dabrowski, Tala Eagling-Vose, Noleen Köhler, Sebastian Ordyniak, Daniël Paulusma

We determine if the width of a graph class ${\cal G}$ changes from unbounded
to bounded if we consider only those graphs from ${\cal G}$ whose diameter is
bounded. As parameters we consider treedepth, pathwidth, treewidth and
clique-width, and as graph classes we consider classes defined by forbidding
some specific graph $F$ as a minor, induced subgraph or subgraph, respectively.
Our main focus is on treedepth for $F$-subgraph-free graphs of diameter at
most~$d$ for some fixed integer $d$. We give classifications of boundedness of
treedepth for $d\in \{4,5,\ldots\}$ and partial classifications for $d=2$ and
$d=3$.

### Data Structures and Algorithms

### 1. [A Formal Analysis of Algorithms for Matroids and Greedoids](http://arxiv.org/pdf/2505.19816v1)

Authors: Mohammad Abdulaziz, Thomas Ammer, Shriya Meenakshisundaram, Adem Rimpapa

We present a formal analysis, in Isabelle/HOL, of optimisation algorithms for
matroids, which are useful generalisations of combinatorial structures that
occur in optimisation, and greedoids, which are a generalisation of matroids.
Although some formalisation work has been done earlier on matroids, our work
here presents the first formalisation of results on greedoids, and many results
we formalise in relation to matroids are also formalised for the first time in
this work. We formalise the analysis of a number of optimisation algorithms for
matroids and greedoids. We also derive from those algorithms executable
implementations of Kruskal's algorithm for minimum spanning trees, an algorithm
for maximum cardinality matching for bi-partite graphs, and Prim's algorithm
for computing minimum weight spanning trees.

### 2. [Pathographs and some (un)decidability results](http://arxiv.org/pdf/2505.19871v1)

Authors: Daniel Carter, Nicolas Trotignon

We introduce pathographs as a framework to study graph classes defined by
forbidden structures, including forbidding induced subgraphs, minors, etc.
Pathographs approximately generalize s-graphs of
L\'ev\^eque--Lin--Maffray--Trotignon by the addition of two extra adjacency
relations: one between subdivisible edges and vertices called spokes, and one
between pairs of subdivisible edges called rungs. We consider the following
decision problem: given a pathograph $\mathfrak{H}$ and a finite set of
pathographs $\mathcal{F}$, is there an $\mathcal{F}$-free realization of
$\mathfrak{H}$? This may be regarded as a generalization of the "graph class
containment problem": given two graph classes $S$ and $S'$, is it the case that
$S\subseteq S'$? We prove the pathograph realization problem is undecidable in
general, but it is decidable in the case that $\mathfrak{H}$ has no rungs (but
may have spokes), or if $\mathcal{F}$ is closed under adding edges, spokes, and
rungs. We also discuss some potential applications to proving decomposition
theorems.

### 3. [Bounding Width on Graph Classes of Constant Diameter](http://arxiv.org/pdf/2505.19926v1)

Authors: Konrad K. Dabrowski, Tala Eagling-Vose, Noleen Köhler, Sebastian Ordyniak, Daniël Paulusma

We determine if the width of a graph class ${\cal G}$ changes from unbounded
to bounded if we consider only those graphs from ${\cal G}$ whose diameter is
bounded. As parameters we consider treedepth, pathwidth, treewidth and
clique-width, and as graph classes we consider classes defined by forbidding
some specific graph $F$ as a minor, induced subgraph or subgraph, respectively.
Our main focus is on treedepth for $F$-subgraph-free graphs of diameter at
most~$d$ for some fixed integer $d$. We give classifications of boundedness of
treedepth for $d\in \{4,5,\ldots\}$ and partial classifications for $d=2$ and
$d=3$.

### Emerging Technologies

### 1. [Enhancing Test Efficiency through Automated ATPG-Aware Lightweight Scan Instrumentation](http://arxiv.org/pdf/2505.19418v1)

Authors: Sudipta Paria, Md Rezoan Ferdous, Aritra Dasgupta, Atri Chatterjee, Swarup Bhunia

Scan-based Design-for-Testability (DFT) measures are prevalent in modern
digital integrated circuits to achieve high test quality at low hardware cost.
With the advent of 3D heterogeneous integration and chiplet-based systems, the
role of scan is becoming ever more important due to its ability to make
internal design nodes controllable and observable in a systematic and scalable
manner. However, the effectiveness of scan-based DFT suffers from poor
testability of internal nodes for complex circuits at deep logic levels.
Existing solutions to address this problem primarily rely on Test Point
Insertion (TPI) in the nodes with poor controllability or observability.
However, TPI-based solutions, while an integral part of commercial practice,
come at a high design and hardware cost. To address this issue, in this paper,
we present LITE, a novel ATPG-aware lightweight scan instrumentation approach
that utilizes the functional flip-flops in a scan chain to make multiple
internal nodes observable and controllable in a low-cost, scalable manner. We
provide both circuit-level design as well as an algorithmic approach for
automating the insertion of LITE for design modifications. We show that LITE
significantly improves the testability in terms of the number of patterns and
test coverage for ATPG and random pattern testability, respectively, while
incurring considerably lower overhead than TPI-based solutions.

### 2. [Systems of Twinned Systems: A Systematic Literature Review](http://arxiv.org/pdf/2505.19916v1)

Authors: Feyi Adesanya, Kanan Castro Silva, Valdemar V. Graciano Neto, Istvan David

Modern systems exhibit unprecedented complexity due to their increased scale,
interconnectedness, and the heterogeneity of their digital and physical
components. In response to scaling challenges, the system-of-systems (SoS)
paradigm proposes flexible aggregations of subsystems into a larger whole,
while maintaining the independence of subsystems to various degrees. In
response to the cyber-physical convergence, the digital twin (DT) paradigm
proposes a tight coupling between digital and physical components through
computational reflection and precise control. As these two paradigms address
distinct parts of the overall challenge, combining the two promises more
comprehensive methods to engineer what we call systems of twinned systems
(SoTS). The noticeably growing body of knowledge on SoTS calls for a review of
the state of the art. In this work, we report on our systematic literature
survey of SoTS. We screened over 2500 potential studies, of which we included
80 and investigated them in detail. To converge SoS and DT, we derive a
classification framework for SoTS that is backward compatible with the
currently accepted theories of SoS and DT.

### Formal Languages and Automata Theory

### 1. [On groups with EDT0L word problem](http://arxiv.org/pdf/2505.20057v1)

Authors: Alex Bishop, Murray Elder, Alex Evetts, Paul Gallot, Alex Levine

We prove that the word problem for the infinite cyclic group is not EDT0L,
and obtain as a corollary that a finitely generated group with EDT0L word
problem must be torsion. In addition, we show that the property of having an
EDT0L word problem is invariant under change of generating set and passing to
finitely generated subgroups. This represents significant progress towards the
conjecture that all groups with EDT0L word problem are finite (i.e. precisely
the groups with regular word problem).

### Graphics

### 1. [A Fluorescent Material Model for Non-Spectral Editing & Rendering](http://arxiv.org/pdf/2505.19672v1)

Authors: Belcour Laurent, Fichet Alban, Barla Pascal

Fluorescent materials are characterized by a spectral reradiation toward
longer wavelengths. Recent work [Fichet et al. 2024] has shown that the
rendering of fluorescence in a non-spectral engine is possible through the use
of appropriate reduced reradiation matrices. But the approach has limited
expressivity, as it requires the storage of one reduced matrix per fluorescent
material, and only works with measured fluorescent assets.
  In this work, we introduce an analytical approach to the editing and
rendering of fluorescence in a non-spectral engine. It is based on a
decomposition of the reduced reradiation matrix, and an analytically-integrable
Gaussian-based model of the fluorescent component. The model reproduces the
appearance of fluorescent materials accurately, especially with the addition of
a UV basis. Most importantly, it grants variations of fluorescent material
parameters in real-time, either for the editing of fluorescent materials, or
for the dynamic spatial variation of fluorescence properties across object
surfaces. A simplified one-Gaussian fluorescence model even allows for the
artist-friendly creation of plausible fluorescent materials from scratch,
requiring only a few reflectance colors as input.

### 2. [CAD-Coder: Text-to-CAD Generation with Chain-of-Thought and Geometric Reward](http://arxiv.org/pdf/2505.19713v1)

Authors: Yandong Guan, Xilin Wang, Xingxi Ming, Jing Zhang, Dong Xu, Qian Yu

In this work, we introduce CAD-Coder, a novel framework that reformulates
text-to-CAD as the generation of CadQuery scripts - a Python-based, parametric
CAD language. This representation enables direct geometric validation, a richer
modeling vocabulary, and seamless integration with existing LLMs. To further
enhance code validity and geometric fidelity, we propose a two-stage learning
pipeline: (1) supervised fine-tuning on paired text-CadQuery data, and (2)
reinforcement learning with Group Reward Policy Optimization (GRPO), guided by
a CAD-specific reward comprising both a geometric reward (Chamfer Distance) and
a format reward. We also introduce a chain-of-thought (CoT) planning process to
improve model reasoning, and construct a large-scale, high-quality dataset of
110K text-CadQuery-3D model triplets and 1.5K CoT samples via an automated
pipeline. Extensive experiments demonstrate that CAD-Coder enables LLMs to
generate diverse, valid, and complex CAD models directly from natural language,
advancing the state of the art of text-to-CAD generation and geometric
reasoning.

### 3. [MAMM: Motion Control via Metric-Aligning Motion Matching](http://arxiv.org/pdf/2505.19976v1)

Authors: Naoki Agata, Takeo Igarashi

We introduce a novel method for controlling a motion sequence using an
arbitrary temporal control sequence using temporal alignment. Temporal
alignment of motion has gained significant attention owing to its applications
in motion control and retargeting. Traditional methods rely on either learned
or hand-craft cross-domain mappings between frames in the original and control
domains, which often require large, paired, or annotated datasets and
time-consuming training. Our approach, named Metric-Aligning Motion Matching,
achieves alignment by solely considering within-domain distances. It computes
distances among patches in each domain and seeks a matching that optimally
aligns the two within-domain distances. This framework allows for the alignment
of a motion sequence to various types of control sequences, including sketches,
labels, audio, and another motion sequence, all without the need for manually
defined mappings or training with annotated data. We demonstrate the
effectiveness of our approach through applications in efficient motion control,
showcasing its potential in practical scenarios.

### 4. [Agentic 3D Scene Generation with Spatially Contextualized VLMs](http://arxiv.org/pdf/2505.20129v1)

Authors: Xinhang Liu, Yu-Wing Tai, Chi-Keung Tang

Despite recent advances in multimodal content generation enabled by
vision-language models (VLMs), their ability to reason about and generate
structured 3D scenes remains largely underexplored. This limitation constrains
their utility in spatially grounded tasks such as embodied AI, immersive
simulations, and interactive 3D applications. We introduce a new paradigm that
enables VLMs to generate, understand, and edit complex 3D environments by
injecting a continually evolving spatial context. Constructed from multimodal
input, this context consists of three components: a scene portrait that
provides a high-level semantic blueprint, a semantically labeled point cloud
capturing object-level geometry, and a scene hypergraph that encodes rich
spatial relationships, including unary, binary, and higher-order constraints.
Together, these components provide the VLM with a structured, geometry-aware
working memory that integrates its inherent multimodal reasoning capabilities
with structured 3D understanding for effective spatial reasoning. Building on
this foundation, we develop an agentic 3D scene generation pipeline in which
the VLM iteratively reads from and updates the spatial context. The pipeline
features high-quality asset generation with geometric restoration, environment
setup with automatic verification, and ergonomic adjustment guided by the scene
hypergraph. Experiments show that our framework can handle diverse and
challenging inputs, achieving a level of generalization not observed in prior
work. Further results demonstrate that injecting spatial context enables VLMs
to perform downstream tasks such as interactive scene editing and path
planning, suggesting strong potential for spatially intelligent systems in
computer graphics, 3D vision, and embodied applications.

### Computer Science and Game Theory

### 1. [Approximately Optimal Mechanism Design for Competing Sellers](http://arxiv.org/pdf/2505.19453v1)

Authors: Brendan Lucier, Raghuvansh R. Saxena

Two sellers compete to sell identical products to a single buyer. Each seller
chooses an arbitrary mechanism, possibly involving lotteries, to sell their
product. The utility-maximizing buyer can choose to participate in one or both
mechanisms, resolving them in either order. Given a common prior over buyer
values, how should the sellers design their mechanisms to maximize their
respective revenues?
  We first consider a Stackelberg setting where one seller (Alice) commits to
her mechanism and the other seller (Bob) best-responds. We show how to
construct a simple and approximately-optimal single-lottery mechanism for Alice
that guarantees her a quarter of the optimal monopolist's revenue, for any
regular distribution. Along the way we prove a structural result: for any
single-lottery mechanism of Alice, there will always be a best response
mechanism for Bob consisting of a single take-it-or-leave-it price. We also
show that no mechanism (single-lottery or otherwise) can guarantee Alice more
than a 1/e fraction of the monopolist revenue. Finally, we show that our
approximation result does not extend to Nash equilibrium: there exist instances
in which a monopolist could extract full surplus, but neither competing seller
obtains positive revenue at any equilibrium choice of mechanisms.

### 2. [A Framework for Combined Transaction Posting and Pricing for Layer 2 Blockchains](http://arxiv.org/pdf/2505.19556v1)

Authors: Shouqiao Wang, Davide Crapis, Ciamac C. Moallemi

This paper presents a comprehensive framework for transaction posting and
pricing in Layer 2 (L2) blockchain systems, focusing on challenges stemming
from fluctuating Layer 1 (L1) gas fees and the congestion issues within L2
networks. Existing methods have focused on the problem of optimal posting
strategies to L1 in isolation, without simultaneously considering the L2 fee
mechanism. In contrast, our work offers a unified approach that addresses the
complex interplay between transaction queue dynamics, L1 cost variability, and
user responses to L2 fees. We contribute by (1) formulating a dynamic model
that integrates both posting and pricing strategies, capturing the interplay
between L1 gas price fluctuations and L2 queue management, (2) deriving an
optimal threshold-based posting policy that guides L2 sequencers in managing
transactions based on queue length and current L1 conditions, and (3)
establishing theoretical foundations for a dynamic L2 fee mechanism that
balances cost recovery with congestion control. We validate our framework
through simulations.

### 3. [The residual maximin share](http://arxiv.org/pdf/2505.19961v1)

Authors: Uriel Feige

We consider fair allocations of indivisible goods to agents with general
monotone valuations. We observe that it is useful to introduce a new
share-based fairness notion, the {\em residual maximin share} (RMMS). This
share is {\em feasible} and {\em self maximizing}. Its value is at least as
large as the MXS, and at least as large as $\frac{2}{3}$-MMS for additive
valuations. Known techniques easily imply the existence of partial allocations
that are both RMMS and EFX, and complete allocations that are both RMMS and
EFL. This unifies and somewhat improves upon several different results from
previous papers.

### 4. [Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games](http://arxiv.org/pdf/2505.19537v1)

Authors: Yi Feng, Kaito Fujii, Stratis Skoulakis, Xiao Wang, Volkan Cevher

Since Polyak's pioneering work, heavy ball (HB) momentum has been widely
studied in minimization. However, its role in min-max games remains largely
unexplored. As a key component of practical min-max algorithms like Adam, this
gap limits their effectiveness. In this paper, we present a continuous-time
analysis for HB with simultaneous and alternating update schemes in min-max
games. Locally, we prove smaller momentum enhances algorithmic stability by
enabling local convergence across a wider range of step sizes, with alternating
updates generally converging faster. Globally, we study the implicit
regularization of HB, and find smaller momentum guides algorithms trajectories
towards shallower slope regions of the loss landscapes, with alternating
updates amplifying this effect. Surprisingly, all these phenomena differ from
those observed in minimization, where larger momentum yields similar effects.
Our results reveal fundamental differences between HB in min-max games and
minimization, and numerical experiments further validate our theoretical
results.

### 5. [Eliciting Informed Preferences](http://arxiv.org/pdf/2505.19570v1)

Authors: Modibo K. Camara, Nicole Immorlica, Brendan Lucier

If people find it costly to evaluate the options available to them, their
choices may not directly reveal their preferences. Yet, it is conceivable that
a researcher can still learn about a population's preferences with careful
experiment design. We formalize the researcher's problem in a model of robust
mechanism design where it is costly for individuals to learn about how much
they value a product. We characterize the statistics that the researcher can
identify, and find that they are quite restricted. Finally, we apply our
positive results to social choice and propose a way to combat uninformed
voting.

### 6. [Multi-Agent Reinforcement Learning in Cybersecurity: From Fundamentals to Applications](http://arxiv.org/pdf/2505.19837v1)

Authors: Christoph R. Landolt, Christoph Würsch, Roland Meier, Alain Mermoud, Julian Jang-Jaccard

Multi-Agent Reinforcement Learning (MARL) has shown great potential as an
adaptive solution for addressing modern cybersecurity challenges. MARL enables
decentralized, adaptive, and collaborative defense strategies and provides an
automated mechanism to combat dynamic, coordinated, and sophisticated threats.
This survey investigates the current state of research in MARL applications for
automated cyber defense (ACD), focusing on intruder detection and lateral
movement containment. Additionally, it examines the role of Autonomous
Intelligent Cyber-defense Agents (AICA) and Cyber Gyms in training and
validating MARL agents. Finally, the paper outlines existing challenges, such
as scalability and adversarial robustness, and proposes future research
directions. This also discusses how MARL integrates in AICA to provide
adaptive, scalable, and dynamic solutions to counter the increasingly
sophisticated landscape of cyber threats. It highlights the transformative
potential of MARL in areas like intrusion detection and lateral movement
containment, and underscores the value of Cyber Gyms for training and
validation of AICA.

### 7. [The Limits of Preference Data for Post-Training](http://arxiv.org/pdf/2505.19964v1)

Authors: Eric Zhao, Jessica Dai, Pranjal Awasthi

Recent progress in strengthening the capabilities of large language models
has stemmed from applying reinforcement learning to domains with automatically
verifiable outcomes. A key question is whether we can similarly use RL to
optimize for outcomes in domains where evaluating outcomes inherently requires
human feedback; for example, in tasks like deep research and trip planning,
outcome evaluation is qualitative and there are many possible degrees of
success. One attractive and scalable modality for collecting human feedback is
preference data: ordinal rankings (pairwise or $k$-wise) that indicate, for $k$
given outcomes, which one is preferred. In this work, we study a critical
roadblock: preference data fundamentally and significantly limits outcome-based
optimization. Even with idealized preference data (infinite, noiseless, and
online), the use of ordinal feedback can prevent obtaining even approximately
optimal solutions. We formalize this impossibility using voting theory, drawing
an analogy between how a model chooses to answer a query with how voters choose
a candidate to elect. This indicates that grounded human scoring and
algorithmic innovations are necessary for extending the success of RL
post-training to domains demanding human feedback. We also explore why these
limitations have disproportionately impacted RLHF when it comes to eliciting
reasoning behaviors (e.g., backtracking) versus situations where RLHF has been
historically successful (e.g., instruction-tuning and safety training), finding
that the limitations of preference data primarily suppress RLHF's ability to
elicit robust strategies -- a class that encompasses most reasoning behaviors.

### Human-Computer Interaction

### 1. [Understanding and Supporting Co-viewing Comedy in VR with Embodied Expressive Avatars](http://arxiv.org/pdf/2505.20082v1)

Authors: Ryo Ohara, Chi-Lan Yang, Takuji Narumi, Hideaki Kuzuoka

Co-viewing videos with family and friends remotely has become prevalent with
the support of communication channels such as text messaging or real-time voice
chat. However, current co-viewing platforms often lack visible embodied cues,
such as body movements and facial expressions. This absence can reduce
emotional engagement and the sense of co-presence when people are watching
together remotely. Although virtual reality (VR) is an emerging technology that
allows individuals to participate in various social activities while embodied
as avatars, we still do not fully understand how this embodiment in VR affects
co-viewing experiences, particularly in terms of engagement, emotional
contagion, and expressive norms. In a controlled experiment involving eight
triads of three participants each (N=24), we compared the participants'
perceptions and reactions while watching comedy in VR using embodied expressive
avatars that displayed visible laughter cues. This was contrasted with a
control condition where no such embodied expressions were presented. With a
mixed-method analysis, we found that embodied laughter cues shifted
participants' engagement from individual immersion to socially coordinated
participation. Participants reported heightened self-awareness of emotional
expression, greater emotional contagion, and the development of expressive
norms surrounding co-viewers' laughter. The result highlighted the tension
between individual engagement and interpersonal emotional accommodation when
co-viewing with embodied expressive avatars.

### 2. [FairTalk: Facilitating Balanced Participation in Video Conferencing by Implicit Visualization of Predicted Turn-Grabbing Intention](http://arxiv.org/pdf/2505.20138v1)

Authors: Ryo Iijima, Shigeo Yoshida, Atsushi Hashimoto, Jiaxin Ma

Creating fair opportunities for all participants to contribute is a notable
challenge in video conferencing. This paper introduces FairTalk, a system that
facilitates the subconscious redistribution of speaking opportunities. FairTalk
predicts participants' turn-grabbing intentions using a machine learning model
trained on web-collected videoconference data with positive-unlabeled learning,
where turn-taking detection provides automatic positive labels. To subtly
balance speaking turns, the system visualizes predicted intentions by mimicking
natural human behaviors associated with the desire to speak. A user study
suggests that FairTalk may help improve speaking balance, though subjective
feedback indicates no significant perceived impact. We also discuss design
implications derived from participant interviews.

### 3. [It's Not Just Labeling" -- A Research on LLM Generated Feedback Interpretability and Image Labeling Sketch Features](http://arxiv.org/pdf/2505.19419v1)

Authors: Baichuan Li, Larry Powell, Tracy Hammond

The quality of training data is critical to the performance of machine
learning applications in domains like transportation, healthcare, and robotics.
Accurate image labeling, however, often relies on time-consuming, expert-driven
methods with limited feedback. This research introduces a sketch-based
annotation approach supported by large language models (LLMs) to reduce
technical barriers and enhance accessibility. Using a synthetic dataset, we
examine how sketch recognition features relate to LLM feedback metrics, aiming
to improve the reliability and interpretability of LLM-assisted labeling. We
also explore how prompting strategies and sketch variations influence feedback
quality. Our main contribution is a sketch-based virtual assistant that
simplifies annotation for non-experts and advances LLM-driven labeling tools in
terms of scalability, accessibility, and explainability.

### 4. [On the Same Page: Dimensions of Perceived Shared Understanding in Human-AI Interaction](http://arxiv.org/pdf/2505.20068v1)

Authors: Qingyu Liang, Jaime Banks

Shared understanding plays a key role in the effective communication in and
performance of human-human interactions. With the increasingly common
integration of AI into human contexts, the future of personal and workplace
interactions will likely see human-AI interaction (HAII) in which the
perception of shared understanding is important. Existing literature has
addressed the processes and effects of PSU in human-human interactions, but the
construal remains underexplored in HAII. To better understand PSU in HAII, we
conducted an online survey to collect user reflections on interactions with a
large language model when it sunderstanding of a situation was thought to be
similar to or different from the participant's. Through inductive thematic
analysis, we identified eight dimensions comprising PSU in human-AI
interactions: Fluency, aligned operation, fluidity, outcome satisfaction,
contextual awareness, lack of humanlike abilities, computational limits, and
suspicion.

### 5. [Explanation User Interfaces: A Systematic Literature Review](http://arxiv.org/pdf/2505.20085v1)

Authors: Eleonora Cappuccio, Andrea Esposito, Francesco Greco, Giuseppe Desolda, Rosa Lanzilotti, Salvatore Rinzivillo

Artificial Intelligence (AI) is one of the major technological advancements
of this century, bearing incredible potential for users through AI-powered
applications and tools in numerous domains. Being often black-box (i.e., its
decision-making process is unintelligible), developers typically resort to
eXplainable Artificial Intelligence (XAI) techniques to interpret the behaviour
of AI models to produce systems that are transparent, fair, reliable, and
trustworthy. However, presenting explanations to the user is not trivial and is
often left as a secondary aspect of the system's design process, leading to AI
systems that are not useful to end-users. This paper presents a Systematic
Literature Review on Explanation User Interfaces (XUIs) to gain a deeper
understanding of the solutions and design guidelines employed in the academic
literature to effectively present explanations to users. To improve the
contribution and real-world impact of this survey, we also present a framework
for Human-cEnteRed developMent of Explainable user interfaceS (HERMES) to guide
practitioners and academics in the design and evaluation of XUIs.

### 6. [Fairness Practices in Industry: A Case Study in Machine Learning Teams Building Recommender Systems](http://arxiv.org/pdf/2505.19441v1)

Authors: Jing Nathan Yan, Junxiong Wang, Jeffrey M. Rzeszotarski, Allison Koenecke

The rapid proliferation of recommender systems necessitates robust fairness
practices to address inherent biases. Assessing fairness, though, is
challenging due to constantly evolving metrics and best practices. This paper
analyzes how industry practitioners perceive and incorporate these changing
fairness standards in their workflows. Through semi-structured interviews with
11 practitioners from technical teams across a range of large technology
companies, we investigate industry implementations of fairness in
recommendation system products. We focus on current debiasing practices,
applied metrics, collaborative strategies, and integrating academic research
into practice. Findings show a preference for multi-dimensional debiasing over
traditional demographic methods, and a reliance on intuitive rather than
academic metrics. This study also highlights the difficulties in balancing
fairness with both the practitioner's individual (bottom-up) roles and
organizational (top-down) workplace constraints, including the interplay with
legal and compliance experts. Finally, we offer actionable recommendations for
the recommender system community and algorithmic fairness practitioners,
underlining the need to refine fairness practices continually.

### 7. [SACM: SEEG-Audio Contrastive Matching for Chinese Speech Decoding](http://arxiv.org/pdf/2505.19652v1)

Authors: Hongbin Wang, Zhihong Jia, Yuanzhong Shen, Ziwei Wang, Siyang Li, Kai Shu, Feng Hu, Dongrui Wu

Speech disorders such as dysarthria and anarthria can severely impair the
patient's ability to communicate verbally. Speech decoding brain-computer
interfaces (BCIs) offer a potential alternative by directly translating speech
intentions into spoken words, serving as speech neuroprostheses. This paper
reports an experimental protocol for Mandarin Chinese speech decoding BCIs,
along with the corresponding decoding algorithms. Stereo-electroencephalography
(SEEG) and synchronized audio data were collected from eight drug-resistant
epilepsy patients as they conducted a word-level reading task. The proposed
SEEG and Audio Contrastive Matching (SACM), a contrastive learning-based
framework, achieved decoding accuracies significantly exceeding chance levels
in both speech detection and speech decoding tasks. Electrode-wise analysis
revealed that a single sensorimotor cortex electrode achieved performance
comparable to that of the full electrode array. These findings provide valuable
insights for developing more accurate online speech decoding BCIs.

### 8. [ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows](http://arxiv.org/pdf/2505.19897v1)

Authors: Qiushi Sun, Zhoumianze Liu, Chang Ma, Zichen Ding, Fangzhi Xu, Zhangyue Yin, Haiteng Zhao, Zhenyu Wu, Kanzhi Cheng, Zhaoyang Liu, Jianing Wang, Qintong Li, Xiangru Tang, Tianbao Xie, Xiachong Feng, Xiang Li, Ben Kao, Wenhai Wang, Biqing Qi, Lingpeng Kong, Zhiyong Wu

Large Language Models (LLMs) have extended their impact beyond Natural
Language Processing, substantially fostering the development of
interdisciplinary research. Recently, various LLM-based agents have been
developed to assist scientific discovery progress across multiple aspects and
domains. Among these, computer-using agents, capable of interacting with
operating systems as humans do, are paving the way to automated scientific
problem-solving and addressing routines in researchers' workflows. Recognizing
the transformative potential of these agents, we introduce ScienceBoard, which
encompasses two complementary contributions: (i) a realistic, multi-domain
environment featuring dynamic and visually rich scientific workflows with
integrated professional software, where agents can autonomously interact via
different interfaces to accelerate complex research tasks and experiments; and
(ii) a challenging benchmark of 169 high-quality, rigorously validated
real-world tasks curated by humans, spanning scientific-discovery workflows in
domains such as biochemistry, astronomy, and geoinformatics. Extensive
evaluations of agents with state-of-the-art backbones (e.g., GPT-4o, Claude
3.7, UI-TARS) show that, despite some promising results, they still fall short
of reliably assisting scientists in complex workflows, achieving only a 15%
overall success rate. In-depth analysis further provides valuable insights for
addressing current agent limitations and more effective design principles,
paving the way to build more capable agents for scientific discovery. Our code,
environment, and benchmark are at
https://qiushisun.github.io/ScienceBoard-Home/.

### 9. [The Many Challenges of Human-Like Agents in Virtual Game Environments](http://arxiv.org/pdf/2505.20011v1)

Authors: Maciej Świechowski, Dominik Ślęzak

Human-like agents are an increasingly important topic in games and beyond.
Believable non-player characters enhance the gaming experience by improving
immersion and providing entertainment. They also offer players the opportunity
to engage with AI entities that can function as opponents, teachers, or
cooperating partners. Additionally, in games where bots are prohibited -- and
even more so in non-game environments -- there is a need for methods capable of
identifying whether digital interactions occur with bots or humans. This leads
to two fundamental research questions: (1) how to model and implement
human-like AI, and (2) how to measure its degree of human likeness.
  This article offers two contributions. The first one is a survey of the most
significant challenges in implementing human-like AI in games (or any virtual
environment featuring simulated agents, although this article specifically
focuses on games). Thirteen such challenges, both conceptual and technical, are
discussed in detail. The second is an empirical study performed in a tactical
video game that addresses the research question: "Is it possible to distinguish
human players from bots (AI agents) based on empirical data?" A
machine-learning approach using a custom deep recurrent convolutional neural
network is presented. We hypothesize that the more challenging it is to create
human-like AI for a given game, the easier it becomes to develop a method for
distinguishing humans from AI-driven players.

### Information Retrieval

### 1. [LLMs as Better Recommenders with Natural Language Collaborative Signals: A Self-Assessing Retrieval Approach](http://arxiv.org/pdf/2505.19464v1)

Authors: Haoran Xin, Ying Sun, Chao Wang, Weijia Zhang, Hui Xiong

Incorporating collaborative information (CI) effectively is crucial for
leveraging LLMs in recommendation tasks. Existing approaches often encode CI
using soft tokens or abstract identifiers, which introduces a semantic
misalignment with the LLM's natural language pretraining and hampers knowledge
integration. To address this, we propose expressing CI directly in natural
language to better align with LLMs' semantic space. We achieve this by
retrieving a curated set of the most relevant user behaviors in natural
language form. However, identifying informative CI is challenging due to the
complexity of similarity and utility assessment. To tackle this, we introduce a
Self-assessing COllaborative REtrieval framework (SCORE) following the
retrieve-rerank paradigm. First, a Collaborative Retriever (CAR) is developed
to consider both collaborative patterns and semantic similarity. Then, a
Self-assessing Reranker (SARE) leverages LLMs' own reasoning to assess and
prioritize retrieved behaviors. Finally, the selected behaviors are prepended
to the LLM prompt as natural-language CI to guide recommendation. Extensive
experiments on two public datasets validate the effectiveness of SCORE in
improving LLM-based recommendation.

### 2. [Improving Recommendation Fairness without Sensitive Attributes Using Multi-Persona LLMs](http://arxiv.org/pdf/2505.19473v1)

Authors: Haoran Xin, Ying Sun, Chao Wang, Yanke Yu, Weijia Zhang, Hui Xiong

Despite the success of recommender systems in alleviating information
overload, fairness issues have raised concerns in recent years, potentially
leading to unequal treatment for certain user groups. While efforts have been
made to improve recommendation fairness, they often assume that users'
sensitive attributes are available during model training. However, collecting
sensitive information can be difficult, especially on platforms that involve no
personal information disclosure. Therefore, we aim to improve recommendation
fairness without any access to sensitive attributes. However, this is a
non-trivial task because uncovering latent sensitive patterns from complicated
user behaviors without explicit sensitive attributes can be difficult.
Consequently, suboptimal estimates of sensitive distributions can hinder the
fairness training process. To address these challenges, leveraging the
remarkable reasoning abilities of Large Language Models (LLMs), we propose a
novel LLM-enhanced framework for Fair recommendation withOut Sensitive
Attributes (LLMFOSA). A Multi-Persona Sensitive Information Inference module
employs LLMs with distinct personas that mimic diverse human perceptions to
infer and distill sensitive information. Furthermore, a Confusion-Aware
Sensitive Representation Learning module incorporates inference results and
rationales to develop robust sensitive representations, considering the
mislabeling confusion and collective consensus among agents. The model is then
optimized by a formulated mutual information objective. Extensive experiments
on two public datasets validate the effectiveness of LLMFOSA in improving
fairness.

### 3. [One Model to Rank Them All: Unifying Online Advertising with End-to-End Learning](http://arxiv.org/pdf/2505.19755v1)

Authors: Junyan Qiu, Ze Wang, Fan Zhang, Zuowu Zheng, Jile Zhu, Jiangke Fan, Teng Zhang, Haitao Wang, Xingxing Wang

Modern industrial advertising systems commonly employ Multi-stage Cascading
Architectures (MCA) to balance computational efficiency with ranking accuracy.
However, this approach presents two fundamental challenges: (1) performance
inconsistencies arising from divergent optimization targets and capability
differences between stages, and (2) failure to account for advertisement
externalities - the complex interactions between candidate ads during ranking.
These limitations ultimately compromise system effectiveness and reduce
platform profitability. In this paper, we present UniROM, an end-to-end
generative architecture that Unifies online advertising Ranking as One Model.
UniROM replaces cascaded stages with a single model to directly generate
optimal ad sequences from the full candidate ad corpus in location-based
services (LBS). The primary challenges associated with this approach stem from
high costs of feature processing and computational bottlenecks in modeling
externalities of large-scale candidate pools. To address these challenges,
UniROM introduces an algorithm and engine co-designed hybrid feature service to
decouple user and ad feature processing, reducing latency while preserving
expressiveness. To efficiently extract intra- and cross-sequence mutual
information, we propose RecFormer with an innovative cluster-attention
mechanism as its core architectural component. Furthermore, we propose a
bi-stage training strategy that integrates pre-training with reinforcement
learning-based post-training to meet sophisticated platform and advertising
objectives. Extensive offline evaluations on public benchmarks and large-scale
online A/B testing on industrial advertising platform have demonstrated the
superior performance of UniROM over state-of-the-art MCAs.

### 4. [Light distillation for Incremental Graph Convolution Collaborative Filtering](http://arxiv.org/pdf/2505.19810v1)

Authors: X Fan, F Mo, C Chen, H Yamana

Recommender systems presently utilize vast amounts of data and play a pivotal
role in enhancing user experiences. Graph Convolution Networks (GCNs) have
surfaced as highly efficient models within the realm of recommender systems due
to their ability to capture extensive relational information. The continuously
expanding volume of data may render the training of GCNs excessively costly. To
tackle this problem, incrementally training GCNs as new data blocks come in has
become a vital research direction. Knowledge distillation techniques have been
explored as a general paradigm to train GCNs incrementally and alleviate the
catastrophic forgetting problem that typically occurs in incremental settings.
However, we argue that current methods based on knowledge distillation
introduce additional parameters and have a high model complexity, which results
in unrealistic training time consumption in an incremental setting and thus
difficult to actually deploy in the real world. In this work, we propose a
light preference-driven distillation method to distill the preference score of
a user for an item directly from historical interactions, which reduces the
training time consumption in the incremental setting significantly without
noticeable loss in performance. The experimental result on two general datasets
shows that the proposed method can save training time from 1.5x to 9.5x
compared to the existing methods and improves Recall@20 by 5.41% and 10.64%
from the fine-tune method.

### 5. [HIT Model: A Hierarchical Interaction-Enhanced Two-Tower Model for Pre-Ranking Systems](http://arxiv.org/pdf/2505.19849v1)

Authors: Haoqiang Yang, Congde Yuan, Kun Bai, Mengzhuo Guo, Wei Yang, Chao Zhou

Online display advertising platforms rely on pre-ranking systems to
efficiently filter and prioritize candidate ads from large corpora, balancing
relevance to users with strict computational constraints. The prevailing
two-tower architecture, though highly efficient due to its decoupled design and
pre-caching, suffers from cross-domain interaction and coarse similarity
metrics, undermining its capacity to model complex user-ad relationships. In
this study, we propose the Hierarchical Interaction-Enhanced Two-Tower (HIT)
model, a new architecture that augments the two-tower paradigm with two key
components: $\textit{generators}$ that pre-generate holistic vectors
incorporating coarse-grained user-ad interactions through a dual-generator
framework with a cosine-similarity-based generation loss as the training
objective, and $\textit{multi-head representers}$ that project embeddings into
multiple latent subspaces to capture fine-grained, multi-faceted user interests
and multi-dimensional ad attributes. This design enhances modeling
effectiveness without compromising inference efficiency. Extensive experiments
on public datasets and large-scale online A/B testing on Tencent's advertising
platform demonstrate that HIT significantly outperforms several baselines in
relevance metrics, yielding a $1.66\%$ increase in Gross Merchandise Volume and
a $1.55\%$ improvement in Return on Investment, alongside similar serving
latency to the vanilla two-tower models. The HIT model has been successfully
deployed in Tencent's online display advertising system, serving billions of
impressions daily. The code is available at
https://anonymous.4open.science/r/HIT_model-5C23.

### 6. [Anveshana: A New Benchmark Dataset for Cross-Lingual Information Retrieval On English Queries and Sanskrit Documents](http://arxiv.org/pdf/2505.19494v1)

Authors: Manoj Balaji Jagadeeshan, Prince Raj, Pawan Goyal

The study presents a comprehensive benchmark for retrieving Sanskrit
documents using English queries, focusing on the chapters of the
Srimadbhagavatam. It employs a tripartite approach: Direct Retrieval (DR),
Translation-based Retrieval (DT), and Query Translation (QT), utilizing shared
embedding spaces and advanced translation methods to enhance retrieval systems
in a RAG framework. The study fine-tunes state-of-the-art models for Sanskrit's
linguistic nuances, evaluating models such as BM25, REPLUG, mDPR, ColBERT,
Contriever, and GPT-2. It adapts summarization techniques for Sanskrit
documents to improve QA processing. Evaluation shows DT methods outperform DR
and QT in handling the cross-lingual challenges of ancient texts, improving
accessibility and understanding. A dataset of 3,400 English-Sanskrit
query-document pairs underpins the study, aiming to preserve Sanskrit
scriptures and share their philosophical importance widely. Our dataset is
publicly available at https://huggingface.co/datasets/manojbalaji1/anveshana

### 7. [Hierarchical Tree Search-based User Lifelong Behavior Modeling on Large Language Model](http://arxiv.org/pdf/2505.19505v1)

Authors: Yu Xia, Rui Zhong, Hao Gu, Wei Yang, Chi Lu, Peng Jiang, Kun Gai

Large Language Models (LLMs) have garnered significant attention in
Recommendation Systems (RS) due to their extensive world knowledge and robust
reasoning capabilities. However, a critical challenge lies in enabling LLMs to
effectively comprehend and extract insights from massive user behaviors.
Current approaches that directly leverage LLMs for user interest learning face
limitations in handling long sequential behaviors, effectively extracting
interest, and applying interest in practical scenarios. To address these
issues, we propose a Hierarchical Tree Search-based User Lifelong Behavior
Modeling framework (HiT-LBM). HiT-LBM integrates Chunked User Behavior
Extraction (CUBE) and Hierarchical Tree Search for Interest (HTS) to capture
diverse interests and interest evolution of user. CUBE divides user lifelong
behaviors into multiple chunks and learns the interest and interest evolution
within each chunk in a cascading manner. HTS generates candidate interests
through hierarchical expansion and searches for the optimal interest with
process rating model to ensure information gain for each behavior chunk.
Additionally, we design Temporal-Ware Interest Fusion (TIF) to integrate
interests from multiple behavior chunks, constructing a comprehensive
representation of user lifelong interests. The representation can be embedded
into any recommendation model to enhance performance. Extensive experiments
demonstrate the effectiveness of our approach, showing that it surpasses
state-of-the-art methods.

### 8. [Cuff-KT: Tackling Learners' Real-time Learning Pattern Adjustment via Tuning-Free Knowledge State Guided Model Updating](http://arxiv.org/pdf/2505.19543v1)

Authors: Yiyun Zhou, Zheqi Lv, Shengyu Zhang, Jingyuan Chen

Knowledge Tracing (KT) is a core component of Intelligent Tutoring Systems,
modeling learners' knowledge state to predict future performance and provide
personalized learning support. Traditional KT models assume that learners'
learning abilities remain relatively stable over short periods or change in
predictable ways based on prior performance. However, in reality, learners'
abilities change irregularly due to factors like cognitive fatigue, motivation,
and external stress -- a task introduced, which we refer to as Real-time
Learning Pattern Adjustment (RLPA). Existing KT models, when faced with RLPA,
lack sufficient adaptability, because they fail to timely account for the
dynamic nature of different learners' evolving learning patterns. Current
strategies for enhancing adaptability rely on retraining, which leads to
significant overfitting and high time overhead issues. To address this, we
propose Cuff-KT, comprising a controller and a generator. The controller
assigns value scores to learners, while the generator generates personalized
parameters for selected learners. Cuff-KT controllably adapts to data changes
fast and flexibly without fine-tuning. Experiments on five datasets from
different subjects demonstrate that Cuff-KT significantly improves the
performance of five KT models with different structures under intra- and
inter-learner shifts, with an average relative increase in AUC of 10% and 4%,
respectively, at a negligible time cost, effectively tackling RLPA task. Our
code and datasets are fully available at https://github.com/zyy-2001/Cuff-KT.

### 9. [Unlocking the Power of Diffusion Models in Sequential Recommendation: A Simple and Effective Approach](http://arxiv.org/pdf/2505.19544v1)

Authors: Jialei Chen, Yuanbo Xu, Yiheng Jiang

In this paper, we focus on the often-overlooked issue of embedding collapse
in existing diffusion-based sequential recommendation models and propose ADRec,
an innovative framework designed to mitigate this problem. Diverging from
previous diffusion-based methods, ADRec applies an independent noise process to
each token and performs diffusion across the entire target sequence during
training. ADRec captures token interdependency through auto-regression while
modeling per-token distributions through token-level diffusion. This dual
approach enables the model to effectively capture both sequence dynamics and
item representations, overcoming the limitations of existing methods. To
further mitigate embedding collapse, we propose a three-stage training
strategy: (1) pre-training the embedding weights, (2) aligning these weights
with the ADRec backbone, and (3) fine-tuning the model. During inference, ADRec
applies the denoising process only to the last token, ensuring that the
meaningful patterns in historical interactions are preserved. Our comprehensive
empirical evaluation across six datasets underscores the effectiveness of ADRec
in enhancing both the accuracy and efficiency of diffusion-based sequential
recommendation systems.

### 10. [LogiCoL: Logically-Informed Contrastive Learning for Set-based Dense Retrieval](http://arxiv.org/pdf/2505.19588v1)

Authors: Yanzhen Shen, Sihao Chen, Xueqiang Xu, Yunyi Zhang, Chaitanya Malaviya, Dan Roth

While significant progress has been made with dual- and bi-encoder dense
retrievers, they often struggle on queries with logical connectives, a use case
that is often overlooked yet important in downstream applications. Current
dense retrievers struggle with such queries, such that the retrieved results do
not respect the logical constraints implied in the queries. To address this
challenge, we introduce LogiCoL, a logically-informed contrastive learning
objective for dense retrievers. LogiCoL builds upon in-batch supervised
contrastive learning, and learns dense retrievers to respect the subset and
mutually-exclusive set relation between query results via two sets of soft
constraints expressed via t-norm in the learning objective. We evaluate the
effectiveness of LogiCoL on the task of entity retrieval, where the model is
expected to retrieve a set of entities in Wikipedia that satisfy the implicit
logical constraints in the query. We show that models trained with LogiCoL
yield improvement both in terms of retrieval performance and logical
consistency in the results. We provide detailed analysis and insights to
uncover why queries with logical connectives are challenging for dense
retrievers and why LogiCoL is most effective.

### Machine Learning

### 1. [Are Time-Series Foundation Models Deployment-Ready? A Systematic Study of Adversarial Robustness Across Domains](http://arxiv.org/pdf/2505.19397v1)

Authors: Jiawen Zhang, Zhenwei Zhang, Shun Zheng, Xumeng Wen, Jia Li, Jiang Bian

Time Series Foundation Models (TSFMs), which are pretrained on large-scale,
cross-domain data and capable of zero-shot forecasting in new scenarios without
further training, are increasingly adopted in real-world applications. However,
as the zero-shot forecasting paradigm gets popular, a critical yet overlooked
question emerges: Are TSFMs robust to adversarial input perturbations? Such
perturbations could be exploited in man-in-the-middle attacks or data
poisoning. To address this gap, we conduct a systematic investigation into the
adversarial robustness of TSFMs. Our results show that even minimal
perturbations can induce significant and controllable changes in forecast
behaviors, including trend reversal, temporal drift, and amplitude shift,
posing serious risks to TSFM-based services. Through experiments on
representative TSFMs and multiple datasets, we reveal their consistent
vulnerabilities and identify potential architectural designs, such as
structural sparsity and multi-task pretraining, that may improve robustness.
Our findings offer actionable guidance for designing more resilient forecasting
systems and provide a critical assessment of the adversarial robustness of
TSFMs.

### 2. [Future Link Prediction Without Memory or Aggregation](http://arxiv.org/pdf/2505.19408v1)

Authors: Lu Yi, Runlin Lei, Fengran Mo, Yanping Zheng, Zhewei Wei, Yuhang Ye

Future link prediction on temporal graphs is a fundamental task with wide
applicability in real-world dynamic systems. These scenarios often involve both
recurring (seen) and novel (unseen) interactions, requiring models to
generalize effectively across both types of edges. However, existing methods
typically rely on complex memory and aggregation modules, yet struggle to
handle unseen edges. In this paper, we revisit the architecture of existing
temporal graph models and identify two essential but overlooked modeling
requirements for future link prediction: representing nodes with unique
identifiers and performing target-aware matching between source and destination
nodes. To this end, we propose Cross-Attention based Future Link Predictor on
Temporal Graphs (CRAFT), a simple yet effective architecture that discards
memory and aggregation modules and instead builds on two components: learnable
node embeddings and cross-attention between the destination and the source's
recent interactions. This design provides strong expressive power and enables
target-aware modeling of the compatibility between candidate destinations and
the source's interaction patterns. Extensive experiments on diverse datasets
demonstrate that CRAFT consistently achieves superior performance with high
efficiency, making it well-suited for large-scale real-world applications.

### 3. [Importance Weighted Score Matching for Diffusion Samplers with Enhanced Mode Coverage](http://arxiv.org/pdf/2505.19431v1)

Authors: Chenguang Wang, Xiaoyu Zhang, Kaiyuan Cui, Weichen Zhao, Yongtao Guan, Tianshu Yu

Training neural samplers directly from unnormalized densities without access
to target distribution samples presents a significant challenge. A critical
desideratum in these settings is achieving comprehensive mode coverage,
ensuring the sampler captures the full diversity of the target distribution.
However, prevailing methods often circumvent the lack of target data by
optimizing reverse KL-based objectives. Such objectives inherently exhibit
mode-seeking behavior, potentially leading to incomplete representation of the
underlying distribution. While alternative approaches strive for better mode
coverage, they typically rely on implicit mechanisms like heuristics or
iterative refinement. In this work, we propose a principled approach for
training diffusion-based samplers by directly targeting an objective analogous
to the forward KL divergence, which is conceptually known to encourage mode
coverage. We introduce \textit{Importance Weighted Score Matching}, a method
that optimizes this desired mode-covering objective by re-weighting the score
matching loss using tractable importance sampling estimates, thereby overcoming
the absence of target distribution data. We also provide theoretical analysis
of the bias and variance for our proposed Monte Carlo estimator and the
practical loss function used in our method. Experiments on increasingly complex
multi-modal distributions, including 2D Gaussian Mixture Models with up to 120
modes and challenging particle systems with inherent symmetries -- demonstrate
that our approach consistently outperforms existing neural samplers across all
distributional distance metrics, achieving state-of-the-art results on all
benchmarks.

### 4. [Advanced long-term earth system forecasting by learning the small-scale nature](http://arxiv.org/pdf/2505.19432v1)

Authors: Hao Wu, Yuan Gao, Ruiqi Shu, Kun Wang, Ruijian Gou, Chuhan Wu, Xinliang Liu, Juncai He, Shuhao Cao, Junfeng Fang, Xingjian Shi, Feng Tao, Qi Song, Shengxuan Ji, Yanfei Xiang, Yuze Sun, Jiahao Li, Fan Xu, Huanshuo Dong, Haixin Wang, Fan Zhang, Penghao Zhao, Xian Wu, Qingsong Wen, Deliang Chen, Xiaomeng Huang

Reliable long-term forecast of Earth system dynamics is heavily hampered by
instabilities in current AI models during extended autoregressive simulations.
These failures often originate from inherent spectral bias, leading to
inadequate representation of critical high-frequency, small-scale processes and
subsequent uncontrolled error amplification. We present Triton, an AI framework
designed to address this fundamental challenge. Inspired by increasing grids to
explicitly resolve small scales in numerical models, Triton employs a
hierarchical architecture processing information across multiple resolutions to
mitigate spectral bias and explicitly model cross-scale dynamics. We
demonstrate Triton's superior performance on challenging forecast tasks,
achieving stable year-long global temperature forecasts, skillful Kuroshio eddy
predictions till 120 days, and high-fidelity turbulence simulations preserving
fine-scale structures all without external forcing, with significantly
surpassing baseline AI models in long-term stability and accuracy. By
effectively suppressing high-frequency error accumulation, Triton offers a
promising pathway towards trustworthy AI-driven simulation for climate and
earth system science.

### 5. [Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression](http://arxiv.org/pdf/2505.19433v1)

Authors: Peijie Dong, Zhenheng Tang, Xiang Liu, Lujun Li, Xiaowen Chu, Bo Li

Post-training compression reduces the computational and memory costs of large
language models (LLMs), enabling resource-efficient deployment. However,
existing compression benchmarks only focus on language modeling (e.g.,
perplexity) and natural language understanding tasks (e.g., GLUE accuracy),
ignoring the agentic capabilities - workflow, tool use/function call,
long-context understanding and real-world application. We introduce the Agent
Compression Benchmark (ACBench), the first comprehensive benchmark for
evaluating how compression impacts LLMs' agentic abilities. ACBench spans (1)
12 tasks across 4 capabilities (e.g., WorfBench for workflow generation,
Needle-in-Haystack for long-context retrieval), (2) quantization (GPTQ, AWQ)
and pruning (Wanda, SparseGPT), and (3) 15 models, including small (Gemma-2B),
standard (Qwen2.5 7B-32B), and distilled reasoning LLMs (DeepSeek-R1-Distill).
Our experiments reveal compression tradeoffs: 4-bit quantization preserves
workflow generation and tool use (1%-3% drop) but degrades real-world
application accuracy by 10%-15%. We introduce ERank, Top-k Ranking Correlation
and Energy to systematize analysis. ACBench provides actionable insights for
optimizing LLM compression in agentic scenarios. The code can be found in
https://github.com/pprp/ACBench.

### 6. [MetaGMT: Improving Actionable Interpretability of Graph Multilinear Networks via Meta-Learning Filtration](http://arxiv.org/pdf/2505.19445v1)

Authors: Rishabh Bhattacharya, Hari Shankar, Vaishnavi Shivkumar, Ponnurangam Kumaraguru

The growing adoption of Graph Neural Networks (GNNs) in high-stakes domains
like healthcare and finance demands reliable explanations of their
decision-making processes. While inherently interpretable GNN architectures
like Graph Multi-linear Networks (GMT) have emerged, they remain vulnerable to
generating explanations based on spurious correlations, potentially undermining
trust in critical applications. We present MetaGMT, a meta-learning framework
that enhances explanation fidelity through a novel bi-level optimization
approach. We demonstrate that MetaGMT significantly improves both explanation
quality (AUC-ROC, Precision@K) and robustness to spurious patterns, across
BA-2Motifs, MUTAG, and SP-Motif benchmarks. Our approach maintains competitive
classification accuracy while producing more faithful explanations (with an
increase up to 8% of Explanation ROC on SP-Motif 0.5) compared to baseline
methods. These advancements in interpretability could enable safer deployment
of GNNs in sensitive domains by (1) facilitating model debugging through more
reliable explanations, (2) supporting targeted retraining when biases are
identified, and (3) enabling meaningful human oversight. By addressing the
critical challenge of explanation reliability, our work contributes to building
more trustworthy and actionable GNN systems for real-world applications.

### 7. [Learning for Dynamic Combinatorial Optimization without Training Data](http://arxiv.org/pdf/2505.19497v1)

Authors: Yiqiao Liao, Farinaz Koushanfar, Parinaz Naghizadeh

We introduce DyCO-GNN, a novel unsupervised learning framework for Dynamic
Combinatorial Optimization that requires no training data beyond the problem
instance itself. DyCO-GNN leverages structural similarities across
time-evolving graph snapshots to accelerate optimization while maintaining
solution quality. We evaluate DyCO-GNN on dynamic maximum cut, maximum
independent set, and the traveling salesman problem across diverse datasets of
varying sizes, demonstrating its superior performance under tight and moderate
time budgets. DyCO-GNN consistently outperforms the baseline methods, achieving
high-quality solutions up to 3-60x faster, highlighting its practical
effectiveness in rapidly evolving resource-constrained settings.

### 8. [Fox in the Henhouse: Supply-Chain Backdoor Attacks Against Reinforcement Learning](http://arxiv.org/pdf/2505.19532v1)

Authors: Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah Erfani, Benjamin I. P. Rubinstein

The current state-of-the-art backdoor attacks against Reinforcement Learning
(RL) rely upon unrealistically permissive access models, that assume the
attacker can read (or even write) the victim's policy parameters, observations,
or rewards. In this work, we question whether such a strong assumption is
required to launch backdoor attacks against RL. To answer this question, we
propose the \underline{S}upply-\underline{C}h\underline{a}in
\underline{B}ackdoor (SCAB) attack, which targets a common RL workflow:
training agents using external agents that are provided separately or embedded
within the environment. In contrast to prior works, our attack only relies on
legitimate interactions of the RL agent with the supplied agents. Despite this
limited access model, by poisoning a mere $3\%$ of training experiences, our
attack can successfully activate over $90\%$ of triggered actions, reducing the
average episodic return by $80\%$ for the victim. Our novel attack demonstrates
that RL attacks are likely to become a reality under untrusted RL training
supply-chains.

### 9. [ExAnte: A Benchmark for Ex-Ante Inference in Large Language Models](http://arxiv.org/pdf/2505.19533v1)

Authors: Yachuan Liu, Xiaochun Wei, Lin Shi, Xinnuo Li, Bohan Zhang, Paramveer Dhillon, Qiaozhu Mei

Large language models (LLMs) face significant challenges in ex-ante
reasoning, where analysis, inference, or predictions must be made without
access to information from future events. Even with explicit prompts enforcing
temporal cutoffs, LLMs often generate outputs influenced by internalized
knowledge of events beyond the specified cutoff. This paper introduces a novel
task and benchmark designed to evaluate the ability of LLMs to reason while
adhering to such temporal constraints. The benchmark includes a variety of
tasks: stock prediction, Wikipedia event prediction, scientific publication
prediction, and Question Answering (QA), designed to assess factual knowledge
under temporal cutoff constraints. We use leakage rate to quantify models'
reliance on future information beyond cutoff timestamps. Experimental results
reveal that LLMs struggle to consistently adhere to temporal cutoffs across
common prompting strategies and tasks, demonstrating persistent challenges in
ex-ante reasoning. This benchmark provides a potential evaluation framework to
advance the development of LLMs' temporal reasoning ability for time-sensitive
applications.

### 10. [On scalable and efficient training of diffusion samplers](http://arxiv.org/pdf/2505.19552v1)

Authors: Minkyu Kim, Kiyoung Seong, Dongyeop Woo, Sungsoo Ahn, Minsu Kim

We address the challenge of training diffusion models to sample from
unnormalized energy distributions in the absence of data, the so-called
diffusion samplers. Although these approaches have shown promise, they struggle
to scale in more demanding scenarios where energy evaluations are expensive and
the sampling space is high-dimensional. To address this limitation, we propose
a scalable and sample-efficient framework that properly harmonizes the powerful
classical sampling method and the diffusion sampler. Specifically, we utilize
Monte Carlo Markov chain (MCMC) samplers with a novelty-based auxiliary energy
as a Searcher to collect off-policy samples, using an auxiliary energy function
to compensate for exploring modes the diffusion sampler rarely visits. These
off-policy samples are then combined with on-policy data to train the diffusion
sampler, thereby expanding its coverage of the energy landscape. Furthermore,
we identify primacy bias, i.e., the preference of samplers for early experience
during training, as the main cause of mode collapse during training, and
introduce a periodic re-initialization trick to resolve this issue. Our method
significantly improves sample efficiency on standard benchmarks for diffusion
samplers and also excels at higher-dimensional problems and real-world
molecular conformer generation.

### Neural and Evolutionary Computing

### 1. [Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](http://arxiv.org/pdf/2505.19458v1)

Authors: Akiyoshi Tomihari, Ryo Karakida

The theoretical understanding of self-attention (SA) has been steadily
progressing. A prominent line of work studies a class of SA layers that admit
an energy function decreased by state updates. While it provides valuable
insights into inherent biases in signal propagation, it often relies on
idealized assumptions or additional constraints not necessarily present in
standard SA. Thus, to broaden our understanding, this work aims to relax these
energy constraints and provide an energy-agnostic characterization of inference
dynamics by dynamical systems analysis. In more detail, we first consider
relaxing the symmetry and single-head constraints traditionally required in
energy-based formulations. Next, to investigate more general SA architectures
capable of oscillatory dynamics without necessarily admitting an energy
function, we analyze the Jacobian matrix of the state. We reveal that
normalization layers effectively normalize the Jacobian's complex eigenvalues,
forcing the dynamics close to a critical state. This significantly enhances
inference performance. Furthermore, we utilize the Jacobian perspective to
develop regularization methods for training and a pseudo-energy for monitoring
inference dynamics.

### Networking and Internet Architecture

### 1. [A Cost-efficient Credit-Based Shaper Deployment Framework for Time-Sensitive Networks](http://arxiv.org/pdf/2505.19771v1)

Authors: Santiago Torres-Borda, Ahlem Mifdaoui

Time-sensitive networks are designed to meet stringent Quality of Service
(QoS) requirements for mixed-criticality traffic with diverse performance
demands. Ensuring deterministic guarantees for such traffic while reducing
deployment costs remains a significant challenge. This paper proposes a
cost-efficient partial deployment strategy for Time Sensitive Networking (TSN)
devices within legacy Ethernet network. At the core of our approach is the
Credit-Based Shaper (CBS), a key TSN scheduling mechanism. Unlike
cost-prohibitive full CBS deployment, our approach selectively integrates CBS
where it is most needed to enhance performance while reducing costs. Combining
Network Calculus for schedulability verification and a heuristic optimization
method for CBS configuration and placement, our proposal minimizes deployment
costs while improving schedulability for medium-priority traffic and mitigating
blocking delays for high-priority traffic. The feasibility and benefits of our
approach are validated on a realistic automotive TSN use case with up to 70% of
reduction in TSN devices requirements compared to a full deployment.

### Robotics

### 1. [SMAP: Self-supervised Motion Adaptation for Physically Plausible Humanoid Whole-body Control](http://arxiv.org/pdf/2505.19463v1)

Authors: Haoyu Zhao, Sixu Lin, Qingwei Ben, Minyue Dai, Hao Fei, Jingbo Wang, Hua Zou, Junting Dong

This paper presents a novel framework that enables real-world humanoid robots
to maintain stability while performing human-like motion. Current methods train
a policy which allows humanoid robots to follow human body using the massive
retargeted human data via reinforcement learning. However, due to the
heterogeneity between human and humanoid robot motion, directly using
retargeted human motion reduces training efficiency and stability. To this end,
we introduce SMAP, a novel whole-body tracking framework that bridges the gap
between human and humanoid action spaces, enabling accurate motion mimicry by
humanoid robots. The core idea is to use a vector-quantized periodic
autoencoder to capture generic atomic behaviors and adapt human motion into
physically plausible humanoid motion. This adaptation accelerates training
convergence and improves stability when handling novel or challenging motions.
We then employ a privileged teacher to distill precise mimicry skills into the
student policy with a proposed decoupled reward. We conduct experiments in
simulation and real world to demonstrate the superiority stability and
performance of SMAP over SOTA methods, offering practical guidelines for
advancing whole-body control in humanoid robots.

### 2. [DiffE2E: Rethinking End-to-End Driving with a Hybrid Action Diffusion and Supervised Policy](http://arxiv.org/pdf/2505.19516v1)

Authors: Rui Zhao, Yuze Fan, Ziguo Chen, Fei Gao, Zhenhai Gao

End-to-end learning has emerged as a transformative paradigm in autonomous
driving. However, the inherently multimodal nature of driving behaviors and the
generalization challenges in long-tail scenarios remain critical obstacles to
robust deployment. We propose DiffE2E, a diffusion-based end-to-end autonomous
driving framework. This framework first performs multi-scale alignment of
multi-sensor perception features through a hierarchical bidirectional
cross-attention mechanism. It then introduces a novel class of hybrid
diffusion-supervision decoders based on the Transformer architecture, and
adopts a collaborative training paradigm that seamlessly integrates the
strengths of both diffusion and supervised policy. DiffE2E models structured
latent spaces, where diffusion captures the distribution of future trajectories
and supervision enhances controllability and robustness. A global condition
integration module enables deep fusion of perception features with high-level
targets, significantly improving the quality of trajectory generation.
Subsequently, a cross-attention mechanism facilitates efficient interaction
between integrated features and hybrid latent variables, promoting the joint
optimization of diffusion and supervision objectives for structured output
generation, ultimately leading to more robust control. Experiments demonstrate
that DiffE2E achieves state-of-the-art performance in both CARLA closed-loop
evaluations and NAVSIM benchmarks. The proposed integrated
diffusion-supervision policy offers a generalizable paradigm for hybrid action
representation, with strong potential for extension to broader domains
including embodied intelligence. More details and visualizations are available
at \href{https://infinidrive.github.io/DiffE2E/}{project website}.

### 3. [Heavy lifting tasks via haptic teleoperation of a wheeled humanoid](http://arxiv.org/pdf/2505.19530v1)

Authors: Amartya Purushottam, Jack Yan, Christopher Yu, Joao Ramos

Humanoid robots can support human workers in physically demanding
environments by performing tasks that require whole-body coordination, such as
lifting and transporting heavy objects.These tasks, which we refer to as
Dynamic Mobile Manipulation (DMM), require the simultaneous control of
locomotion, manipulation, and posture under dynamic interaction forces. This
paper presents a teleoperation framework for DMM on a height-adjustable wheeled
humanoid robot for carrying heavy payloads. A Human-Machine Interface (HMI)
enables whole-body motion retargeting from the human pilot to the robot by
capturing the motion of the human and applying haptic feedback. The pilot uses
body motion to regulate robot posture and locomotion, while arm movements guide
manipulation.Real time haptic feedback delivers end effector wrenches and
balance related cues, closing the loop between human perception and robot
environment interaction. We evaluate the different telelocomotion mappings that
offer varying levels of balance assistance, allowing the pilot to either
manually or automatically regulate the robot's lean in response to
payload-induced disturbances. The system is validated in experiments involving
dynamic lifting of barbells and boxes up to 2.5 kg (21% of robot mass),
demonstrating coordinated whole-body control, height variation, and disturbance
handling under pilot guidance. Video demo can be found at:
https://youtu.be/jF270_bG1h8?feature=shared

### 4. [Real-time Whole-body Model Predictive Control for Bipedal Locomotion with a Novel Kino-dynamic Model and Warm-start Method](http://arxiv.org/pdf/2505.19540v1)

Authors: Junhyung Kim, Hokyun Lee, Jaeheung Park

Advancements in optimization solvers and computing power have led to growing
interest in applying whole-body model predictive control (WB-MPC) to bipedal
robots. However, the high degrees of freedom and inherent model complexity of
bipedal robots pose significant challenges in achieving fast and stable control
cycles for real-time performance. This paper introduces a novel kino-dynamic
model and warm-start strategy for real-time WB-MPC in bipedal robots. Our
proposed kino-dynamic model combines the linear inverted pendulum plus flywheel
and full-body kinematics model. Unlike the conventional whole-body model that
rely on the concept of contact wrenches, our model utilizes the zero-moment
point (ZMP), reducing baseline computational costs and ensuring consistently
low latency during contact state transitions. Additionally, a modularized
multi-layer perceptron (MLP) based warm-start strategy is proposed, leveraging
a lightweight neural network to provide a good initial guess for each control
cycle. Furthermore, we present a ZMP-based whole-body controller (WBC) that
extends the existing WBC for explicitly controlling impulses and ZMP,
integrating it into the real-time WB-MPC framework. Through various comparative
experiments, the proposed kino-dynamic model and warm-start strategy have been
shown to outperform previous studies. Simulations and real robot experiments
further validate that the proposed framework demonstrates robustness to
perturbation and satisfies real-time control requirements during walking.

### 5. [LF-GNSS: Towards More Robust Satellite Positioning with a Hard Example Mining Enhanced Learning-Filtering Deep Fusion Framework](http://arxiv.org/pdf/2505.19560v1)

Authors: Jianan Lou, Rong Zhang

Global Navigation Satellite System (GNSS) is essential for autonomous driving
systems, unmanned vehicles, and various location-based technologies, as it
provides the precise geospatial information necessary for navigation and
situational awareness. However, its performance is often degraded by
Non-Line-Of-Sight (NLOS) and multipath effects, especially in urban
environments. Recently, Artificial Intelligence (AI) has been driving
innovation across numerous industries, introducing novel solutions to mitigate
the challenges in satellite positioning. This paper presents a
learning-filtering deep fusion framework for satellite positioning, termed
LF-GNSS. The framework utilizes deep learning networks to intelligently analyze
the signal characteristics of satellite observations, enabling the adaptive
construction of observation noise covariance matrices and compensated
innovation vectors for Kalman filter input. A dynamic hard example mining
technique is incorporated to enhance model robustness by prioritizing
challenging satellite signals during training. Additionally, we introduce a
novel feature representation based on Dilution of Precision (DOP)
contributions, which helps to more effectively characterize the signal quality
of individual satellites and improve measurement weighting. LF-GNSS has been
validated on both public and private datasets, demonstrating superior
positioning accuracy compared to traditional methods and other learning-based
solutions. To encourage further integration of AI and GNSS research, we will
open-source the code at https://github.com/GarlanLou/LF-GNSS, and release a
collection of satellite positioning datasets for urban scenarios at
https://github.com/GarlanLou/LF-GNSS-Dataset.

### 6. [Whole-body Multi-contact Motion Control for Humanoid Robots Based on Distributed Tactile Sensors](http://arxiv.org/pdf/2505.19580v1)

Authors: Masaki Murooka, Kensuke Fukumitsu, Marwan Hamze, Mitsuharu Morisawa, Hiroshi Kaminaga, Fumio Kanehiro, Eiichi Yoshida

To enable humanoid robots to work robustly in confined environments,
multi-contact motion that makes contacts not only at extremities, such as hands
and feet, but also at intermediate areas of the limbs, such as knees and
elbows, is essential. We develop a method to realize such whole-body
multi-contact motion involving contacts at intermediate areas by a humanoid
robot. Deformable sheet-shaped distributed tactile sensors are mounted on the
surface of the robot's limbs to measure the contact force without significantly
changing the robot body shape. The multi-contact motion controller developed
earlier, which is dedicated to contact at extremities, is extended to handle
contact at intermediate areas, and the robot motion is stabilized by feedback
control using not only force/torque sensors but also distributed tactile
sensors. Through verification on dynamics simulations, we show that the
developed tactile feedback improves the stability of whole-body multi-contact
motion against disturbances and environmental errors. Furthermore, the
life-sized humanoid RHP Kaleido demonstrates whole-body multi-contact motions,
such as stepping forward while supporting the body with forearm contact and
balancing in a sitting posture with thigh contacts.

### 7. [Indoor Air Quality Detection Robot Model Based on the Internet of Things (IoT)](http://arxiv.org/pdf/2505.19600v1)

Authors: Anggiat Mora Simamora, Asep Denih, Mohamad Iqbal Suriansyah

This paper presents the design, implementation, and evaluation of an
IoT-based robotic system for mapping and monitoring indoor air quality. The
primary objective was to develop a mobile robot capable of autonomously mapping
a closed environment, detecting concentrations of CO$_2$, volatile organic
compounds (VOCs), smoke, temperature, and humidity, and transmitting real-time
data to a web interface. The system integrates a set of sensors (SGP30, MQ-2,
DHT11, VL53L0X, MPU6050) with an ESP32 microcontroller. It employs a mapping
algorithm for spatial data acquisition and utilizes a Mamdani fuzzy logic
system for air quality classification. Empirical tests in a model room
demonstrated average localization errors below $5\%$, actuator motion errors
under $2\%$, and sensor measurement errors within $12\%$ across all modalities.
The contributions of this work include: (1) a low-cost, integrated IoT robotic
platform for simultaneous mapping and air quality detection; (2) a web-based
user interface for real-time visualization and control; and (3) validation of
system accuracy under laboratory conditions.

### 8. [Autonomous Flights inside Narrow Tunnels](http://arxiv.org/pdf/2505.19657v1)

Authors: Luqi Wang, Yan Ning, Hongming Chen, Peize Liu, Yang Xu, Hao Xu, Ximin Lyu, Shaojie Shen

Multirotors are usually desired to enter confined narrow tunnels that are
barely accessible to humans in various applications including inspection,
search and rescue, and so on. This task is extremely challenging since the lack
of geometric features and illuminations, together with the limited field of
view, cause problems in perception; the restricted space and significant ego
airflow disturbances induce control issues. This paper introduces an autonomous
aerial system designed for navigation through tunnels as narrow as 0.5 m in
diameter. The real-time and online system includes a virtual omni-directional
perception module tailored for the mission and a novel motion planner that
incorporates perception and ego airflow disturbance factors modeled using
camera projections and computational fluid dynamics analyses, respectively.
Extensive flight experiments on a custom-designed quadrotor are conducted in
multiple realistic narrow tunnels to validate the superior performance of the
system, even over human pilots, proving its potential for real applications.
Additionally, a deployment pipeline on other multirotor platforms is outlined
and open-source packages are provided for future developments.

### 9. [GeoPF: Infusing Geometry into Potential Fields for Reactive Planning in Non-trivial Environments](http://arxiv.org/pdf/2505.19688v1)

Authors: Yuhe Gong, Riddhiman Laha, Luis Figueredo

Reactive intelligence remains one of the cornerstones of versatile robotics
operating in cluttered, dynamic, and human-centred environments. Among reactive
approaches, potential fields (PF) continue to be widely adopted due to their
simplicity and real-time applicability. However, existing PF methods typically
oversimplify environmental representations by relying on isotropic, point- or
sphere-based obstacle approximations. In human-centred settings, this
simplification results in overly conservative paths, cumbersome tuning, and
computational overhead -- even breaking real-time requirements. In response, we
propose the Geometric Potential Field (GeoPF), a reactive motion-planning
framework that explicitly infuses geometric primitives - points, lines, planes,
cubes, and cylinders - into real-time planning. By leveraging precise
closed-form distance functions, GeoPF significantly reduces computational
complexity and parameter tuning effort. Extensive quantitative analyses
consistently show GeoPF's higher success rates, reduced tuning complexity (a
single parameter set across experiments), and substantially lower computational
costs (up to 2 orders of magnitude) compared to traditional PF methods.
Real-world experiments further validate GeoPF's robustness and practical ease
of deployment. GeoPF provides a fresh perspective on reactive planning problems
driving geometric-aware temporal motion generation, enabling flexible and
low-latency motion planning suitable for modern robotic applications.

### 10. [Extremum Flow Matching for Offline Goal Conditioned Reinforcement Learning](http://arxiv.org/pdf/2505.19717v1)

Authors: Quentin Rouxel, Clemente Donoso, Fei Chen, Serena Ivaldi, Jean-Baptiste Mouret

Imitation learning is a promising approach for enabling generalist
capabilities in humanoid robots, but its scaling is fundamentally constrained
by the scarcity of high-quality expert demonstrations. This limitation can be
mitigated by leveraging suboptimal, open-ended play data, often easier to
collect and offering greater diversity. This work builds upon recent advances
in generative modeling, specifically Flow Matching, an alternative to Diffusion
models. We introduce a method for estimating the extremum of the learned
distribution by leveraging the unique properties of Flow Matching, namely,
deterministic transport and support for arbitrary source distributions. We
apply this method to develop several goal-conditioned imitation and
reinforcement learning algorithms based on Flow Matching, where policies are
conditioned on both current and goal observations. We explore and compare
different architectural configurations by combining core components, such as
critic, planner, actor, or world model, in various ways. We evaluated our
agents on the OGBench benchmark and analyzed how different demonstration
behaviors during data collection affect performance in a 2D non-prehensile
pushing task. Furthermore, we validated our approach on real hardware by
deploying it on the Talos humanoid robot to perform complex manipulation tasks
based on high-dimensional image observations, featuring a sequence of
pick-and-place and articulated object manipulation in a realistic kitchen
environment. Experimental videos and code are available at:
https://hucebot.github.io/extremum_flow_matching_website/

### Software Engineering

### 1. [SETBVE: Quality-Diversity Driven Exploration of Software Boundary Behaviors](http://arxiv.org/pdf/2505.19736v1)

Authors: Sabinakhon Akbarova, Felix Dobslaw, Francisco Gomes de Oliveira Neto, Robert Feldt

Software systems exhibit distinct behaviors based on input characteristics,
and failures often occur at the boundaries between input domains. Traditional
Boundary Value Analysis (BVA) relies on manual heuristics, while automated
Boundary Value Exploration (BVE) methods typically optimize a single quality
metric, risking a narrow and incomplete survey of boundary behaviors. We
introduce SETBVE, a customizable, modular framework for automated black-box BVE
that leverages Quality-Diversity (QD) optimization to systematically uncover
and refine a broader spectrum of boundaries. SETBVE maintains an archive of
boundary pairs organized by input- and output-based behavioral descriptors. It
steers exploration toward underrepresented regions while preserving
high-quality boundary pairs and applies local search to refine candidate
boundaries. In experiments with ten integer-based functions, SETBVE outperforms
the baseline in diversity, boosting archive coverage by 37 to 82 percentage
points. A qualitative analysis reveals that SETBVE identifies boundary
candidates the baseline misses. While the baseline method typically plateaus in
both diversity and quality after 30 seconds, SETBVE continues to improve in
600-second runs, demonstrating better scalability. Even the simplest SETBVE
configurations perform well in identifying diverse boundary behaviors. Our
findings indicate that balancing quality with behavioral diversity can help
identify more software edge-case behaviors than quality-focused approaches.

### 2. [SecVulEval: Benchmarking LLMs for Real-World C/C++ Vulnerability Detection](http://arxiv.org/pdf/2505.19828v1)

Authors: Md Basim Uddin Ahmed, Nima Shiri Harzevili, Jiho Shin, Hung Viet Pham, Song Wang

Large Language Models (LLMs) have shown promise in software engineering
tasks, but evaluating their effectiveness in vulnerability detection is
challenging due to the lack of high-quality datasets. Most existing datasets
are limited to function-level labels, ignoring finer-grained vulnerability
patterns and crucial contextual information. Also, poor data quality such as
mislabeling, inconsistent annotations, and duplicates can lead to inflated
performance and weak generalization. Moreover, by including only the functions,
these datasets miss broader program context, like data/control dependencies and
interprocedural interactions, that are essential for accurately understanding
real-world security flaws. Without this context, detection models are evaluated
under unrealistic assumptions.
  To address these limitations, this paper introduces SecVulEval, a benchmark
designed to support fine-grained evaluation of LLMs and other detection methods
with rich contextual information. SecVulEval focuses on real-world C/C++
vulnerabilities at the statement level. This granularity enables more precise
evaluation of a model's ability to localize vulnerabilities, beyond simple
binary classification at the function level. By incorporating rich contextual
information, SecVulEval sets a new standard for vulnerability detection
benchmarks in realistic scenarios. This benchmark includes 25,440 function
samples covering 5,867 unique CVEs in C/C++ projects from 1999 to 2024. We
evaluated the SOTA LLMs with a multi-agent-based approach. The evaluation on
our dataset shows that the models are still far from accurately predicting
vulnerable statements in a given function. The best-performing
Claude-3.7-Sonnet model achieves 23.83% F1-score for detecting vulnerable
statements with correct reasoning. Finally, we analyze the LLM outputs and
provide insights into their behavior in vulnerability detection for C/C++.

### 3. [Requirements Coverage-Guided Minimization for Natural Language Test Cases](http://arxiv.org/pdf/2505.20004v1)

Authors: Rongqi Pan, Feifei Niu, Lionel C. Briand, Hanyang Hu

As software systems evolve, test suites tend to grow in size and often
contain redundant test cases. Such redundancy increases testing effort, time,
and cost. Test suite minimization (TSM) aims to eliminate such redundancy while
preserving key properties such as requirement coverage and fault detection
capability. In this paper, we propose RTM (Requirement coverage-guided Test
suite Minimization), a novel TSM approach designed for requirement-based
testing (validation), which can effectively reduce test suite redundancy while
ensuring full requirement coverage and a high fault detection rate (FDR) under
a fixed minimization budget. Based on common practice in critical systems where
functional safety is important, we assume test cases are specified in natural
language and traced to requirements before being implemented. RTM preprocesses
test cases using three different preprocessing methods, and then converts them
into vector representations using seven text embedding techniques. Similarity
values between vectors are computed utilizing three distance functions. A
Genetic Algorithm, whose population is initialized by coverage-preserving
initialization strategies, is then employed to identify an optimized subset
containing diverse test cases matching the set budget.
  We evaluate RTM on an industrial automotive system dataset comprising $736$
system test cases and $54$ requirements. Experimental results show that RTM
consistently outperforms baseline techniques in terms of FDR across different
minimization budgets while maintaining full requirement coverage. Furthermore,
we investigate the impact of test suite redundancy levels on the effectiveness
of TSM, providing new insights into optimizing requirement-based test suites
under practical constraints.

### 4. [Benchmarking and Enhancing LLM Agents in Localizing Linux Kernel Bugs](http://arxiv.org/pdf/2505.19489v1)

Authors: Zhenhao Zhou, Zhuochen Huang, Yike He, Chong Wang, Jiajun Wang, Yijian Wu, Xin Peng, Yiling Lou

The Linux kernel is a critical system, serving as the foundation for numerous
systems. Bugs in the Linux kernel can cause serious consequences, affecting
billions of users. Fault localization (FL), which aims at identifying the buggy
code elements in software, plays an essential role in software quality
assurance. While recent LLM agents have achieved promising accuracy in FL on
recent benchmarks like SWE-bench, it remains unclear how well these methods
perform in the Linux kernel, where FL is much more challenging due to the
large-scale code base, limited observability, and diverse impact factors. In
this paper, we introduce LinuxFLBench, a FL benchmark constructed from
real-world Linux kernel bugs. We conduct an empirical study to assess the
performance of state-of-the-art LLM agents on the Linux kernel. Our initial
results reveal that existing agents struggle with this task, achieving a best
top-1 accuracy of only 41.6% at file level. To address this challenge, we
propose LinuxFL$^+$, an enhancement framework designed to improve FL
effectiveness of LLM agents for the Linux kernel. LinuxFL$^+$ substantially
improves the FL accuracy of all studied agents (e.g., 7.2% - 11.2% accuracy
increase) with minimal costs. Data and code are available at
https://github.com/FudanSELab/LinuxFLBench.

### 5. [CODE-DITING: A Reasoning-Based Metric for Functional Alignment in Code Evaluation](http://arxiv.org/pdf/2505.19502v1)

Authors: Guang Yang, Yu Zhou, Xiang Chen, Wei Zheng, Xing Hu, Xin Zhou, David Lo, Taolue Chen

Trustworthy evaluation methods for code snippets play a crucial role in
neural code generation. Traditional methods, which either rely on reference
solutions or require executable test cases, have inherent limitation in
flexibility and scalability. The recent LLM-as-Judge methodology offers a
promising alternative by directly evaluating functional consistency between the
problem description and the generated code. To systematically understand the
landscape of these LLM-as-Judge methods, we conduct a comprehensive empirical
study across three diverse datasets. Our investigation reveals the pros and
cons of two categories of LLM-as-Judge methods: the methods based on general
foundation models can achieve good performance but require complex prompts and
lack explainability, while the methods based on reasoning foundation models
provide better explainability with simpler prompts but demand substantial
computational resources due to their large parameter sizes. To address these
limitations, we propose CODE-DITING, a novel code evaluation method that
balances accuracy, efficiency and explainability. We develop a data
distillation framework that effectively transfers reasoning capabilities from
DeepSeek-R1671B to our CODE-DITING 1.5B and 7B models, significantly enhancing
evaluation explainability and reducing the computational cost. With the
majority vote strategy in the inference process, CODE-DITING 1.5B outperforms
all models with the same magnitude of parameters and achieves performance which
would normally exhibit in a model with 5 times of parameter scale. CODE-DITING
7B surpasses GPT-4o and DeepSeek-V3 671B, even though it only uses 1% of the
parameter volume of these large models. Further experiments show that
CODEDITING is robust to preference leakage and can serve as a promising
alternative for code evaluation.

### 6. [Search-Based Software Engineering in the Landscape of AI Foundation Models](http://arxiv.org/pdf/2505.19625v1)

Authors: Hassan Sartaj, Shaukat Ali

Search-based software engineering (SBSE), at the intersection of artificial
intelligence (AI) and software engineering, has been an active area of research
for about 25 years. It has been applied to solve numerous problems across the
entire software engineering lifecycle and has demonstrated its versatility in
multiple domains. With the recent advancements in AI, particularly the
emergence of foundation models (FMs), the evolution of SBSE alongside FMs
remains undetermined. In this window of opportunity, we propose a research
roadmap that articulates the current landscape of SBSE in relation to
foundation models (FMs), highlights open challenges, and outlines potential
research directions for advancing SBSE through its interplay with FMs. This
roadmap aims to establish a forward-thinking and innovative perspective for the
future of SBSE in the era of FMs.

### 7. [Software Engineering for Self-Adaptive Robotics: A Research Agenda](http://arxiv.org/pdf/2505.19629v1)

Authors: Shaukat Ali, Ana Cavalcanti, Cláudio Ângelo Gonçalves Gomes, Peter Gorm Larsen, Hassan Sartaj, Anastasios Tefas, Jim Woodcock, Houxiang Zhang

Self-adaptive robotic systems are designed to operate autonomously in dynamic
and uncertain environments, requiring robust mechanisms to monitor, analyse,
and adapt their behaviour in real-time. Unlike traditional robotic software,
which follows predefined logic, self-adaptive robots leverage artificial
intelligence, machine learning, and model-driven engineering to continuously
adjust to changing operational conditions while ensuring reliability, safety,
and performance. This paper presents a research agenda for software engineering
in self-adaptive robotics, addressing critical challenges across two key
dimensions: (1) the development phase, including requirements engineering,
software design, co-simulation, and testing methodologies tailored to adaptive
robotic systems, and (2) key enabling technologies, such as digital twins,
model-driven engineering, and AI-driven adaptation, which facilitate runtime
monitoring, fault detection, and automated decision-making. We discuss open
research challenges, including verifying adaptive behaviours under uncertainty,
balancing trade-offs between adaptability, performance, and safety, and
integrating self-adaptation frameworks like MAPE-K. By providing a structured
roadmap, this work aims to advance the software engineering foundations for
self-adaptive robotic systems, ensuring they remain trustworthy, efficient, and
capable of handling real-world complexities.

### 8. [Large Language Models in Code Co-generation for Safe Autonomous Vehicles](http://arxiv.org/pdf/2505.19658v1)

Authors: Ali Nouri, Beatriz Cabrero-Daniel, Zhennan Fei, Krishna Ronanki, Håkan Sivencrona, Christian Berger

Software engineers in various industrial domains are already using Large
Language Models (LLMs) to accelerate the process of implementing parts of
software systems. When considering its potential use for ADAS or AD systems in
the automotive context, there is a need to systematically assess this new
setup: LLMs entail a well-documented set of risks for safety-related systems'
development due to their stochastic nature. To reduce the effort for code
reviewers to evaluate LLM-generated code, we propose an evaluation pipeline to
conduct sanity-checks on the generated code. We compare the performance of six
state-of-the-art LLMs (CodeLlama, CodeGemma, DeepSeek-r1, DeepSeek-Coders,
Mistral, and GPT-4) on four safety-related programming tasks. Additionally, we
qualitatively analyse the most frequent faults generated by these LLMs,
creating a failure-mode catalogue to support human reviewers. Finally, the
limitations and capabilities of LLMs in code generation, and the use of the
proposed pipeline in the existing process, are discussed.

### 9. [Systems of Twinned Systems: A Systematic Literature Review](http://arxiv.org/pdf/2505.19916v1)

Authors: Feyi Adesanya, Kanan Castro Silva, Valdemar V. Graciano Neto, Istvan David

Modern systems exhibit unprecedented complexity due to their increased scale,
interconnectedness, and the heterogeneity of their digital and physical
components. In response to scaling challenges, the system-of-systems (SoS)
paradigm proposes flexible aggregations of subsystems into a larger whole,
while maintaining the independence of subsystems to various degrees. In
response to the cyber-physical convergence, the digital twin (DT) paradigm
proposes a tight coupling between digital and physical components through
computational reflection and precise control. As these two paradigms address
distinct parts of the overall challenge, combining the two promises more
comprehensive methods to engineer what we call systems of twinned systems
(SoTS). The noticeably growing body of knowledge on SoTS calls for a review of
the state of the art. In this work, we report on our systematic literature
survey of SoTS. We screened over 2500 potential studies, of which we included
80 and investigated them in detail. To converge SoS and DT, we derive a
classification framework for SoTS that is backward compatible with the
currently accepted theories of SoS and DT.

### 10. [Ontology- and LLM-based Data Harmonization for Federated Learning in Healthcare](http://arxiv.org/pdf/2505.20020v1)

Authors: Natallia Kokash, Lei Wang, Thomas H. Gillespie, Adam Belloum, Paola Grosso, Sara Quinney, Lang Li, Bernard de Bono

The rise of electronic health records (EHRs) has unlocked new opportunities
for medical research, but privacy regulations and data heterogeneity remain key
barriers to large-scale machine learning. Federated learning (FL) enables
collaborative modeling without sharing raw data, yet faces challenges in
harmonizing diverse clinical datasets. This paper presents a two-step data
alignment strategy integrating ontologies and large language models (LLMs) to
support secure, privacy-preserving FL in healthcare, demonstrating its
effectiveness in a real-world project involving semantic mapping of EHR data.

### Social and Information Networks

### 1. [Optimal Intervention for Self-triggering Spatial Networks with Application to Urban Crime Analytics](http://arxiv.org/pdf/2505.19612v1)

Authors: Pramit Das, Moulinath Banerjee, Yuekai Sun

In many network systems, events at one node trigger further activity at other
nodes, e.g., social media users reacting to each other's posts or the
clustering of criminal activity in urban environments. These systems are
typically referred to as self-exciting networks. In such systems, targeted
intervention at critical nodes can be an effective strategy for mitigating
undesirable consequences such as further propagation of criminal activity or
the spreading of misinformation on social media. In our work, we develop an
optimal network intervention model to explore how targeted interventions at
critical nodes can mitigate cascading effects throughout a Spatiotemporal
Hawkes network. Similar models have been studied previously in the literature
in purely temporal Hawkes networks, but in our work, we extend them to a
spatiotemporal setup and demonstrate the efficacy of our methods by comparing
the post-intervention reduction in intensity to other heuristic strategies in
simulated networks. Subsequently, we use our method on crime data from the LA
police department database to find neighborhoods for strategic intervention to
demonstrate an application in predictive policing.

### 2. [Homophily Enhanced Graph Domain Adaptation](http://arxiv.org/pdf/2505.20089v1)

Authors: Ruiyi Fang, Bingheng Li, Jingyu Zhao, Ruizhi Pu, Qiuhao Zeng, Gezheng Xu, Charles Ling, Boyu Wang

Graph Domain Adaptation (GDA) transfers knowledge from labeled source graphs
to unlabeled target graphs, addressing the challenge of label scarcity. In this
paper, we highlight the significance of graph homophily, a pivotal factor for
graph domain alignment, which, however, has long been overlooked in existing
approaches. Specifically, our analysis first reveals that homophily
discrepancies exist in benchmarks. Moreover, we also show that homophily
discrepancies degrade GDA performance from both empirical and theoretical
aspects, which further underscores the importance of homophily alignment in
GDA. Inspired by this finding, we propose a novel homophily alignment algorithm
that employs mixed filters to smooth graph signals, thereby effectively
capturing and mitigating homophily discrepancies between graphs. Experimental
results on a variety of benchmarks verify the effectiveness of our method.

### 3. [Community Moderation and the New Epistemology of Fact Checking on Social Media](http://arxiv.org/pdf/2505.20067v1)

Authors: Isabelle Augenstein, Michiel Bakker, Tanmoy Chakraborty, David Corney, Emilio Ferrara, Iryna Gurevych, Scott Hale, Eduard Hovy, Heng Ji, Irene Larraz, Filippo Menczer, Preslav Nakov, Paolo Papotti, Dhruv Sahnan, Greta Warren, Giovanni Zagni

Social media platforms have traditionally relied on internal moderation teams
and partnerships with independent fact-checking organizations to identify and
flag misleading content. Recently, however, platforms including X (formerly
Twitter) and Meta have shifted towards community-driven content moderation by
launching their own versions of crowd-sourced fact-checking -- Community Notes.
If effectively scaled and governed, such crowd-checking initiatives have the
potential to combat misinformation with increased scale and speed as
successfully as community-driven efforts once did with spam. Nevertheless,
general content moderation, especially for misinformation, is inherently more
complex. Public perceptions of truth are often shaped by personal biases,
political leanings, and cultural contexts, complicating consensus on what
constitutes misleading content. This suggests that community efforts, while
valuable, cannot replace the indispensable role of professional fact-checkers.
Here we systemically examine the current approaches to misinformation detection
across major platforms, explore the emerging role of community-driven
moderation, and critically evaluate both the promises and challenges of
crowd-checking at scale.

### Systems and Control

### 1. [Synchronous Models and Fundamental Systems in Observer Design](http://arxiv.org/pdf/2505.19517v1)

Authors: Pieter van Goor, Robert Mahony

This paper introduces the concept of a synchronous model as an extension of
the internal model concept used in observer design for dynamical systems. A
system is said to contain a synchronous model of another if there is a suitable
error function between the two systems that remains stationary for all of the
trajectories of the two systems. A system is said to admit a synchronous lift
if a second system containing a synchronous model exists. We provide necessary
and sufficient conditions that a system admits a synchronous lift and provide a
method to construct a (there may be many) lifted system should one exist. We
characterise the class of all systems that admit a synchronous lift by showing
that they consist of fundamental vector fields induced by a Lie group action, a
class of system we term fundamental systems. For fundamental systems we propose
a simple synchronous observer design methodology, for which we show how
correction terms can be discretised and combined easily, facilitating global
characterisation of convergence and performance. Finally, we provide three
examples to demonstrate the key concepts of synchrony, symmetry construction,
and observer design for a fundamental system.

### 2. [Range Space or Null Space: Least-Squares Methods for the Realization Problem](http://arxiv.org/pdf/2505.19639v1)

Authors: Jiabao He, Yueyue Xu, Yue Ju, Cristian R. Rojas, Håkan Hjalmarsson

This contribution revisits the classical approximate realization problem,
which involves determining matrices of a state-space model based on estimates
of a truncated series of Markov parameters. A Hankel matrix built up by these
Markov parameters plays a fundamental role in this problem, leveraging the fact
that both its range space and left null space encode critical information about
the state-space model. We examine two prototype realization algorithms based on
the Hankel matrix: the classical range-space-based (SVD-based) method and the
more recent null-space-based method. It is demonstrated that the
range-space-based method corresponds to a total least-squares solution, whereas
the null-space-based method corresponds to an ordinary least-squares solution.
By analyzing the differences in sensitivity of the two algorithms, we determine
the conditions when one or the other realization algorithm is to be preferred,
and identify factors that contribute to an ill-conditioned realization problem.
Furthermore, recognizing that both methods are suboptimal, we argue that the
optimal realization is obtained through a weighted least-squares approach. A
statistical analysis of these methods, including their consistency and
asymptotic normality is also provided.

### 3. [Scalable quantile predictions of peak loads for non-residential customer segments](http://arxiv.org/pdf/2505.19744v1)

Authors: Shaohong Shi, Jacco Heres, Simon H. Tindemans

Electrical grid congestion has emerged as an immense challenge in Europe,
making the forecasting of load and its associated metrics increasingly crucial.
Among these metrics, peak load is fundamental. Non-time-resolved models of peak
load have their advantages of being simple and compact, and among them
Velander's formula (VF) is widely used in distribution network planning.
However, several aspects of VF remain inadequately addressed, including
year-ahead prediction, scaling of customers, aggregation, and, most
importantly, the lack of probabilistic elements. The present paper proposes a
quantile interpretation of VF that enables VF to learn truncated cumulative
distribution functions of peak loads with multiple quantile regression under
non-crossing constraints. The evaluations on non-residential customer data
confirmed its ability to predict peak load year ahead, to fit customers with a
wide range of electricity consumptions, and to model aggregations of customers.
A noteworthy finding is that for a given electricity consumption, aggregations
of customers have statistically larger peak loads than a single customer.

### 4. [Chance-constrained Solar PV Hosting Capacity Assessment for Distribution Grids Using Gaussian Process and Logit Learning](http://arxiv.org/pdf/2505.19839v1)

Authors: Sel Ly, Anshuman Singh, Petr Vorobev, Yeng Chai Soh, Hung Dinh Nguyen

Growing penetration of distributed generation such as solar PV can increase
the risk of over-voltage in distribution grids, affecting network security.
Therefore, assessment of the so-called, PV hosting capacity (HC) - the maximum
amount of PV that a given grid can accommodate becomes an important practical
problem. In this paper, we propose a novel chance-constrained HC estimation
framework using Gaussian Process and Logit learning that can account for
uncertainty and risk management. Also, we consider the assessment of HC under
different voltage control strategies. Our results have demonstrated that the
proposed models can achieve high accuracy levels of up to 93% in predicting
nodal over-voltage events on IEEE 33-bus and 123-bus test-cases. Thus, these
models can be effectively employed to estimate the chance-constrained HC with
various risk levels. Moreover, our proposed methods have simple forms and low
computational costs of only a few seconds.

### 5. [Optimizing Offshore Wind Integration through Multi-Terminal DC Grids: A Market-Based OPF Framework for the North Sea Interconnectors](http://arxiv.org/pdf/2505.19886v1)

Authors: Bernardo Castro Valerio, Vinícius Albernaz Lacerda, Marc Cheah-Mañe, Pieter Gebraad, Oriol Gomis-Bellmunt

Interconnecting price zones and remote renewable energy sources has emerged
as a key solution to achieving climate goals. The objective of this work is to
present a formulation that extends the base optimal power flow model with price
zones constraints to forecast the operations of upcoming offshore wind
developments integrated into a multi-terminal DC grid. A case study based on
the 2030 development of the North Sea is used to exemplify the utilization of
the formulation. Here, three cases are presented, one with the price as a
parameter and the other two with the price as a variable dependent on power
flows between price zones. The paper demonstrates that, for large power flows,
it is necessary to include additional constraints beyond line limitations to
accurately capture the effects of price zone exchanges.

### 6. [Persistently Exciting Online Feedback Optimization Controller with Minimal Perturbations](http://arxiv.org/pdf/2505.19910v1)

Authors: Tore Gude, Marta Anna Zagorowska, Lars Struen Imsland

This paper develops a persistently exciting input generating Online Feedback
Optimization (OFO) controller that estimates the sensitivity of a process
ensuring minimal deviations from the descent direction while converging. This
eliminates the need for random perturbations in feedback loop. The proposed
controller is formulated as a bilevel optimization program, where a nonconvex
full rank constraint is relaxed using linear constraints and penalization. The
validation of the method is performed in a simulated scenario where multiple
systems share a limited, costly resource for production optimization,
simulating an oil and gas resource allocation problem. The method allows for
less input perturbations while accurately estimating gradients, allowing faster
convergence when the gradients are unknown. In the case study, the proposed
method achieved the same profit compared to an OFO controller with random input
perturbations, and $1.4\%$ higher profit compared to an OFO controller without
input perturbations.

### 7. [Interpretable Augmented Physics-Based Model for Estimation and Tracking](http://arxiv.org/pdf/2505.19953v1)

Authors: Ondřej Straka, Jindřich Duník, Pau Closas, Tales Imbiriba

State-space estimation and tracking rely on accurate dynamical models to
perform well. However, obtaining an vaccurate dynamical model for complex
scenarios or adapting to changes in the system poses challenges to the
estimation process. Recently, augmented physics-based models (APBMs) appear as
an appealing strategy to cope with these challenges where the composition of a
small and adaptive neural network with known physics-based models (PBM) is
learned on the fly following an augmented state-space estimation approach. A
major issue when introducing data-driven components in such a scenario is the
danger of compromising the meaning (or interpretability) of estimated states.
In this work, we propose a novel constrained estimation strategy that
constrains the APBM dynamics close to the PBM. The novel state-space
constrained approach leads to more flexible ways to impose constraints than the
traditional APBM approach. Our experiments with a radar-tracking scenario
demonstrate different aspects of the proposed approach and the trade-offs
inherent in the imposed constraints.

### 8. [Efficient Gaussian Mixture Filters based on Transition Density Approximation](http://arxiv.org/pdf/2505.20002v1)

Authors: Ondŕej Straka, Uwe D. Hanebeck

Gaussian mixture filters for nonlinear systems usually rely on severe
approximations when calculating mixtures in the prediction and filtering step.
Thus, offline approximations of noise densities by Gaussian mixture densities
to reduce the approximation error have been proposed. This results in
exponential growth in the number of components, requiring ongoing component
reduction, which is computationally complex. In this paper, the key idea is to
approximate the true transition density by an axis-aligned Gaussian mixture,
where two different approaches are derived. These approximations automatically
ensure a constant number of components in the posterior densities without the
need for explicit reduction. In addition, they allow a trade-off between
estimation quality and computational complexity.

### 9. [Alignment of large language models with constrained learning](http://arxiv.org/pdf/2505.19387v1)

Authors: Botong Zhang, Shuo Li, Ignacio Hounie, Osbert Bastani, Dongsheng Ding, Alejandro Ribeiro

We study the problem of computing an optimal large language model (LLM)
policy for a constrained alignment problem, where the goal is to maximize a
primary reward objective while satisfying constraints on secondary utilities.
Despite the popularity of Lagrangian-based LLM policy search in constrained
alignment, iterative primal-dual methods often fail to converge, and
non-iterative dual-based methods do not achieve optimality in the LLM parameter
space. To address these challenges, we employ Lagrangian duality to develop an
iterative dual-based alignment method that alternates between updating the LLM
policy via Lagrangian maximization and updating the dual variable via dual
descent. In theory, we characterize the primal-dual gap between the primal
value in the distribution space and the dual value in the LLM parameter space.
We further quantify the optimality gap of the learned LLM policies at
near-optimal dual variables with respect to both the objective and the
constraint functions. These results prove that dual-based alignment methods can
find an optimal constrained LLM policy, up to an LLM parametrization gap. We
demonstrate the effectiveness and merits of our approach through extensive
experiments conducted on the PKU-SafeRLHF dataset.

### 10. [Split-as-a-Pro: behavioral control via operator splitting and alternating projections](http://arxiv.org/pdf/2505.19411v1)

Authors: Yu Tang, Carlo Cenedese, Alessio Rimoldi, Florian Dórfler, John Lygeros, Alberto Padoan

The paper introduces Split-as-a-Pro, a control framework that integrates
behavioral systems theory, operator splitting methods, and alternating
projection algorithms. The framework reduces dynamic optimization problems -
arising in both control and estimation - to efficient projection computations.
Split-as-a-Pro builds on a non-parametric formulation that exploits system
structure to separate dynamic constraints imposed by individual subsystems from
external ones, such as interconnection constraints and input/output
constraints. This enables the use of arbitrary system representations, as long
as the associated projection is efficiently computable, thereby enhancing
scalability and compatibility with gray-box modeling. We demonstrate the
effectiveness of Split-as-a-Pro by developing a distributed algorithm for
solving finite-horizon linear quadratic control problems and illustrate its use
in predictive control. Our numerical case studies show that algorithms obtained
using Split-as-a-Pro significantly outperform their centralized counterparts in
runtime and scalability across various standard graph topologies, while
seamlessly leveraging both model-based and data-driven system representations.

### Machine Learning (Statistics Category)

### 1. [Uniform convergence of the smooth calibration error and its relationship with functional gradient](http://arxiv.org/pdf/2505.19396v1)

Authors: Futoshi Futami, Atsushi Nitanda

Calibration is a critical requirement for reliable probabilistic prediction,
especially in high-risk applications. However, the theoretical understanding of
which learning algorithms can simultaneously achieve high accuracy and good
calibration remains limited, and many existing studies provide empirical
validation or a theoretical guarantee in restrictive settings. To address this
issue, in this work, we focus on the smooth calibration error (CE) and provide
a uniform convergence bound, showing that the smooth CE is bounded by the sum
of the smooth CE over the training dataset and a generalization gap. We further
prove that the functional gradient of the loss function can effectively control
the training smooth CE. Based on this framework, we analyze three
representative algorithms: gradient boosting trees, kernel boosting, and
two-layer neural networks. For each, we derive conditions under which both
classification and calibration performances are simultaneously guaranteed. Our
results offer new theoretical insights and practical guidance for designing
reliable probabilistic models with provable calibration guarantees.

### 2. [Information-theoretic Generalization Analysis for VQ-VAEs: A Role of Latent Variables](http://arxiv.org/pdf/2505.19470v1)

Authors: Futoshi Futami, Masahiro Fujisawa

Latent variables (LVs) play a crucial role in encoder-decoder models by
enabling effective data compression, prediction, and generation. Although their
theoretical properties, such as generalization, have been extensively studied
in supervised learning, similar analyses for unsupervised models such as
variational autoencoders (VAEs) remain insufficiently underexplored. In this
work, we extend information-theoretic generalization analysis to
vector-quantized (VQ) VAEs with discrete latent spaces, introducing a novel
data-dependent prior to rigorously analyze the relationship among LVs,
generalization, and data generation. We derive a novel generalization error
bound of the reconstruction loss of VQ-VAEs, which depends solely on the
complexity of LVs and the encoder, independent of the decoder. Additionally, we
provide the upper bound of the 2-Wasserstein distance between the distributions
of the true data and the generated data, explaining how the regularization of
the LVs contributes to the data generation performance.

### 3. [Discounted Online Convex Optimization: Uniform Regret Across a Continuous Interval](http://arxiv.org/pdf/2505.19491v1)

Authors: Wenhao Yang, Sifan Yang, Lijun Zhang

Reflecting the greater significance of recent history over the distant past
in non-stationary environments, $\lambda$-discounted regret has been introduced
in online convex optimization (OCO) to gracefully forget past data as new
information arrives. When the discount factor $\lambda$ is given, online
gradient descent with an appropriate step size achieves an
$O(1/\sqrt{1-\lambda})$ discounted regret. However, the value of $\lambda$ is
often not predetermined in real-world scenarios. This gives rise to a
significant open question: is it possible to develop a discounted algorithm
that adapts to an unknown discount factor. In this paper, we affirmatively
answer this question by providing a novel analysis to demonstrate that smoothed
OGD (SOGD) achieves a uniform $O(\sqrt{\log T/1-\lambda})$ discounted regret,
holding for all values of $\lambda$ across a continuous interval
simultaneously. The basic idea is to maintain multiple OGD instances to handle
different discount factors, and aggregate their outputs sequentially by an
online prediction algorithm named as Discounted-Normal-Predictor (DNP)
(Kapralov and Panigrahy,2010). Our analysis reveals that DNP can combine the
decisions of two experts, even when they operate on discounted regret with
different discount factors.

### 4. [Model Agnostic Differentially Private Causal Inference](http://arxiv.org/pdf/2505.19589v1)

Authors: Christiant Lebeda, Mathieu Even, Aurélien Bellet, Julie Josse

Estimating causal effects from observational data is essential in fields such
as medicine, economics and social sciences, where privacy concerns are
paramount. We propose a general, model-agnostic framework for differentially
private estimation of average treatment effects (ATE) that avoids strong
structural assumptions on the data-generating process or the models used to
estimate propensity scores and conditional outcomes. In contrast to prior work,
which enforces differential privacy by directly privatizing these nuisance
components and results in a privacy cost that scales with model complexity, our
approach decouples nuisance estimation from privacy protection. This separation
allows the use of flexible, state-of-the-art black-box models, while
differential privacy is achieved by perturbing only predictions and aggregation
steps within a fold-splitting scheme with ensemble techniques. We instantiate
the framework for three classical estimators -- the G-formula, inverse
propensity weighting (IPW), and augmented IPW (AIPW) -- and provide formal
utility and privacy guarantees. Empirical results show that our methods
maintain competitive performance under realistic privacy budgets. We further
extend our framework to support meta-analysis of multiple private ATE
estimates. Our results bridge a critical gap between causal inference and
privacy-preserving data analysis.

### 5. [Accelerating Nash Learning from Human Feedback via Mirror Prox](http://arxiv.org/pdf/2505.19731v1)

Authors: Daniil Tiapkin, Daniele Calandriello, Denis Belomestny, Eric Moulines, Alexey Naumov, Kashif Rasul, Michal Valko, Pierre Menard

Traditional Reinforcement Learning from Human Feedback (RLHF) often relies on
reward models, frequently assuming preference structures like the Bradley-Terry
model, which may not accurately capture the complexities of real human
preferences (e.g., intransitivity). Nash Learning from Human Feedback (NLHF)
offers a more direct alternative by framing the problem as finding a Nash
equilibrium of a game defined by these preferences. In this work, we introduce
Nash Mirror Prox ($\mathtt{Nash-MP}$), an online NLHF algorithm that leverages
the Mirror Prox optimization scheme to achieve fast and stable convergence to
the Nash equilibrium. Our theoretical analysis establishes that Nash-MP
exhibits last-iterate linear convergence towards the $\beta$-regularized Nash
equilibrium. Specifically, we prove that the KL-divergence to the optimal
policy decreases at a rate of order $(1+2\beta)^{-N/2}$, where $N$ is a number
of preference queries. We further demonstrate last-iterate linear convergence
for the exploitability gap and uniformly for the span semi-norm of
log-probabilities, with all these rates being independent of the size of the
action space. Furthermore, we propose and analyze an approximate version of
Nash-MP where proximal steps are estimated using stochastic policy gradients,
making the algorithm closer to applications. Finally, we detail a practical
implementation strategy for fine-tuning large language models and present
experiments that demonstrate its competitive performance and compatibility with
existing methods.

### 6. [Density Ratio-Free Doubly Robust Proxy Causal Learning](http://arxiv.org/pdf/2505.19807v1)

Authors: Bariscan Bozkurt, Houssam Zenati, Dimitri Meunier, Liyuan Xu, Arthur Gretton

We study the problem of causal function estimation in the Proxy Causal
Learning (PCL) framework, where confounders are not observed but proxies for
the confounders are available. Two main approaches have been proposed: outcome
bridge-based and treatment bridge-based methods. In this work, we propose two
kernel-based doubly robust estimators that combine the strengths of both
approaches, and naturally handle continuous and high-dimensional variables. Our
identification strategy builds on a recent density ratio-free method for
treatment bridge-based PCL; furthermore, in contrast to previous approaches, it
does not require indicator functions or kernel smoothing over the treatment
variable. These properties make it especially well-suited for continuous or
high-dimensional treatments. By using kernel mean embeddings, we have
closed-form solutions and strong consistency guarantees. Our estimators
outperform existing methods on PCL benchmarks, including a prior doubly robust
method that requires both kernel smoothing and density ratio estimation.

### 7. [Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL](http://arxiv.org/pdf/2505.19923v1)

Authors: Qin-Wen Luo, Ming-Kun Xie, Ye-Wen Wang, Sheng-Jun Huang

Offline reinforcement learning (RL) aims to learn an effective policy from a
static dataset. To alleviate extrapolation errors, existing studies often
uniformly regularize the value function or policy updates across all states.
However, due to substantial variations in data quality, the fixed
regularization strength often leads to a dilemma: Weak regularization strength
fails to address extrapolation errors and value overestimation, while strong
regularization strength shifts policy learning toward behavior cloning,
impeding potential performance enabled by Bellman updates. To address this
issue, we propose the selective state-adaptive regularization method for
offline RL. Specifically, we introduce state-adaptive regularization
coefficients to trust state-level Bellman-driven results, while selectively
applying regularization on high-quality actions, aiming to avoid performance
degradation caused by tight constraints on low-quality actions. By establishing
a connection between the representative value regularization method, CQL, and
explicit policy constraint methods, we effectively extend selective
state-adaptive regularization to these two mainstream offline RL approaches.
Extensive experiments demonstrate that the proposed method significantly
outperforms the state-of-the-art approaches in both offline and
offline-to-online settings on the D4RL benchmark.

### 8. [Regret Analysis of Average-Reward Unichain MDPs via an Actor-Critic Approach](http://arxiv.org/pdf/2505.19986v1)

Authors: Swetha Ganesh, Vaneet Aggarwal

Actor-Critic methods are widely used for their scalability, yet existing
theoretical guarantees for infinite-horizon average-reward Markov Decision
Processes (MDPs) often rely on restrictive ergodicity assumptions. We propose
NAC-B, a Natural Actor-Critic with Batching, that achieves order-optimal regret
of $\tilde{O}(\sqrt{T})$ in infinite-horizon average-reward MDPs under the
unichain assumption, which permits both transient states and periodicity. This
assumption is among the weakest under which the classic policy gradient theorem
remains valid for average-reward settings. NAC-B employs function approximation
for both the actor and the critic, enabling scalability to problems with large
state and action spaces. The use of batching in our algorithm helps mitigate
potential periodicity in the MDP and reduces stochasticity in gradient
estimates, and our analysis formalizes these benefits through the introduction
of the constants $C_{\text{hit}}$ and $C_{\text{tar}}$, which characterize the
rate at which empirical averages over Markovian samples converge to the
stationary distribution.

### 9. [TabPFN: One Model to Rule Them All?](http://arxiv.org/pdf/2505.20003v1)

Authors: Qiong Zhang, Yan Shuo Tan, Qinglong Tian, Pengfei Li

Hollmann et al. (Nature 637 (2025) 319-326) recently introduced TabPFN, a
transformer-based deep learning model for regression and classification on
tabular data, which they claim "outperforms all previous methods on datasets
with up to 10,000 samples by a wide margin, using substantially less training
time." Furthermore, they have called TabPFN a "foundation model" for tabular
data, as it can support "data generation, density estimation, learning reusable
embeddings and fine-tuning". If these statements are well-supported, TabPFN may
have the potential to supersede existing modeling approaches on a wide range of
statistical tasks, mirroring a similar revolution in other areas of artificial
intelligence that began with the advent of large language models. In this
paper, we provide a tailored explanation of how TabPFN works for a statistics
audience, by emphasizing its interpretation as approximate Bayesian inference.
We also provide more evidence of TabPFN's "foundation model" capabilities: We
show that an out-of-the-box application of TabPFN vastly outperforms
specialized state-of-the-art methods for semi-supervised parameter estimation,
prediction under covariate shift, and heterogeneous treatment effect
estimation. We further show that TabPFN can outperform LASSO at sparse
regression and can break a robustness-efficiency trade-off in classification.
All experiments can be reproduced using the code provided at
https://github.com/qinglong-tian/tabpfn_study
(https://github.com/qinglong-tian/tabpfn_study).

### 10. [Linear Bandits with Non-i.i.d. Noise](http://arxiv.org/pdf/2505.20017v1)

Authors: Baptiste Abélès, Eugenio Clerico, Hamish Flynn, Gergely Neu

We study the linear stochastic bandit problem, relaxing the standard i.i.d.
assumption on the observation noise. As an alternative to this restrictive
assumption, we allow the noise terms across rounds to be sub-Gaussian but
interdependent, with dependencies that decay over time. To address this
setting, we develop new confidence sequences using a recently introduced
reduction scheme to sequential probability assignment, and use these to derive
a bandit algorithm based on the principle of optimism in the face of
uncertainty. We provide regret bounds for the resulting algorithm, expressed in
terms of the decay rate of the strength of dependence between observations.
Among other results, we show that our bounds recover the standard rates up to a
factor of the mixing time for geometrically mixing observation noise.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

### 1. [Algorithms for reliable decision-making need causal reasoning](https://www.nature.com/articles/s43588-025-00814-9)

Authors: Christoph Kern et al.

### 2. [Deep exploration of logical models of cell differentiation in human preimplantation embryos](https://www.nature.com/articles/s41540-025-00537-7)

Authors: Mathieu Bolteau et al.

### 3. [Evaluating the effects of active social touch and robot expressiveness on user attitudes and behaviour in human–robot interaction](https://www.nature.com/articles/s41598-025-01490-5)

Authors: Juan Jose Gamboa-Montero et al.

### 4. [A contrastive learning framework with dual gates and noise awareness for temporal knowledge graph reasoning](https://www.nature.com/articles/s41598-025-00314-w)

Authors: Siling Feng et al.

### 5. [Bayesian network structure learning by opposition-based learning](https://www.nature.com/articles/s41598-025-03267-2)

Authors: Baodan Sun et al.

