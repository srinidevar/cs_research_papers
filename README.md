# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-24 17:00:25.525093 PST.

### Artificial Intelligence

### 1. [HySafe-AI: Hybrid Safety Architectural Analysis Framework for AI Systems: A Case Study](http://arxiv.org/pdf/2507.17118v1)

Authors: Mandar Pitale, Jelena Frtunikj, Abhinaw Priyadershi, Vasu Singh, Maria Spence

AI has become integral to safety-critical areas like autonomous driving
systems (ADS) and robotics. The architecture of recent autonomous systems are
trending toward end-to-end (E2E) monolithic architectures such as large
language models (LLMs) and vision language models (VLMs). In this paper, we
review different architectural solutions and then evaluate the efficacy of
common safety analyses such as failure modes and effect analysis (FMEA) and
fault tree analysis (FTA). We show how these techniques can be improved for the
intricate nature of the foundational models, particularly in how they form and
utilize latent representations. We introduce HySAFE-AI, Hybrid Safety
Architectural Analysis Framework for AI Systems, a hybrid framework that adapts
traditional methods to evaluate the safety of AI systems. Lastly, we offer
hints of future work and suggestions to guide the evolution of future AI safety
standards.

### 2. [Improving LLMs' Generalized Reasoning Abilities by Graph Problems](http://arxiv.org/pdf/2507.17168v1)

Authors: Qifan Zhang, Nuo Chen, Zehua Li, Miao Peng, Jing Tang, Jia Li

Large Language Models (LLMs) have made remarkable strides in reasoning tasks,
yet their performance often falters on novel and complex problems.
Domain-specific continued pretraining (CPT) methods, such as those tailored for
mathematical reasoning, have shown promise but lack transferability to broader
reasoning tasks. In this work, we pioneer the use of Graph Problem Reasoning
(GPR) to enhance the general reasoning capabilities of LLMs. GPR tasks,
spanning pathfinding, network analysis, numerical computation, and topological
reasoning, require sophisticated logical and relational reasoning, making them
ideal for teaching diverse reasoning patterns. To achieve this, we introduce
GraphPile, the first large-scale corpus specifically designed for CPT using GPR
data. Spanning 10.9 billion tokens across 23 graph tasks, the dataset includes
chain-of-thought, program-of-thought, trace of execution, and real-world graph
data. Using GraphPile, we train GraphMind on popular base models Llama 3 and
3.1, as well as Gemma 2, achieving up to 4.9 percent higher accuracy in
mathematical reasoning and up to 21.2 percent improvement in non-mathematical
reasoning tasks such as logical and commonsense reasoning. By being the first
to harness GPR for enhancing reasoning patterns and introducing the first
dataset of its kind, our work bridges the gap between domain-specific
pretraining and universal reasoning capabilities, advancing the adaptability
and robustness of LLMs.

### 3. [Students' Feedback Requests and Interactions with the SCRIPT Chatbot: Do They Get What They Ask For?](http://arxiv.org/pdf/2507.17258v1)

Authors: Andreas Scholl, Natalie Kiesler

Building on prior research on Generative AI (GenAI) and related tools for
programming education, we developed SCRIPT, a chatbot based on ChatGPT-4o-mini,
to support novice learners. SCRIPT allows for open-ended interactions and
structured guidance through predefined prompts. We evaluated the tool via an
experiment with 136 students from an introductory programming course at a large
German university and analyzed how students interacted with SCRIPT while
solving programming tasks with a focus on their feedback preferences. The
results reveal that students' feedback requests seem to follow a specific
sequence. Moreover, the chatbot responses aligned well with students' requested
feedback types (in 75%), and it adhered to the system prompt constraints. These
insights inform the design of GenAI-based learning support systems and
highlight challenges in balancing guidance and flexibility in AI-assisted
tools.

### 4. [Compliance Brain Assistant: Conversational Agentic AI for Assisting Compliance Tasks in Enterprise Environments](http://arxiv.org/pdf/2507.17289v1)

Authors: Shitong Zhu, Chenhao Fang, Derek Larson, Neel Reddy Pochareddy, Rajeev Rao, Sophie Zeng, Yanqing Peng, Wendy Summer, Alex Goncalves, Arya Pudota, Herve Robert

This paper presents Compliance Brain Assistant (CBA), a conversational,
agentic AI assistant designed to boost the efficiency of daily compliance tasks
for personnel in enterprise environments. To strike a good balance between
response quality and latency, we design a user query router that can
intelligently choose between (i) FastTrack mode: to handle simple requests that
only need additional relevant context retrieved from knowledge corpora; and
(ii) FullAgentic mode: to handle complicated requests that need composite
actions and tool invocations to proactively discover context across various
compliance artifacts, and/or involving other APIs/models for accommodating
requests. A typical example would be to start with a user query, use its
description to find a specific entity and then use the entity's information to
query other APIs for curating and enriching the final AI response.
  Our experimental evaluations compared CBA against an out-of-the-box LLM on
various real-world privacy/compliance-related queries targeting various
personas. We found that CBA substantially improved upon the vanilla LLM's
performance on metrics such as average keyword match rate (83.7% vs. 41.7%) and
LLM-judge pass rate (82.0% vs. 20.0%). We also compared metrics for the full
routing-based design against the `fast-track only` and `full-agentic` modes and
found that it had a better average match-rate and pass-rate while keeping the
run-time approximately the same. This finding validated our hypothesis that the
routing mechanism leads to a good trade-off between the two worlds.

### 5. [An Uncertainty-Driven Adaptive Self-Alignment Framework for Large Language Models](http://arxiv.org/pdf/2507.17477v1)

Authors: Haoran Sun, Zekun Zhang, Shaoning Zeng

Large Language Models (LLMs) have demonstrated remarkable progress in
instruction following and general-purpose reasoning. However, achieving
high-quality alignment with human intent and safety norms without human
annotations remains a fundamental challenge. In this work, we propose an
Uncertainty-Driven Adaptive Self-Alignment (UDASA) framework designed to
improve LLM alignment in a fully automated manner. UDASA first generates
multiple responses for each input and quantifies output uncertainty across
three dimensions: semantics, factuality, and value alignment. Based on these
uncertainty scores, the framework constructs preference pairs and categorizes
training samples into three stages, conservative, moderate, and exploratory,
according to their uncertainty difference. The model is then optimized
progressively across these stages. In addition, we conduct a series of
preliminary studies to validate the core design assumptions and provide strong
empirical motivation for the proposed framework. Experimental results show that
UDASA outperforms existing alignment methods across multiple tasks, including
harmlessness, helpfulness, truthfulness, and controlled sentiment generation,
significantly improving model performance.

### 6. [LTLZinc: a Benchmarking Framework for Continual Learning and Neuro-Symbolic Temporal Reasoning](http://arxiv.org/pdf/2507.17482v1)

Authors: Luca Salvatore Lorello, Nikolaos Manginas, Marco Lippi, Stefano Melacci

Neuro-symbolic artificial intelligence aims to combine neural architectures
with symbolic approaches that can represent knowledge in a human-interpretable
formalism. Continual learning concerns with agents that expand their knowledge
over time, improving their skills while avoiding to forget previously learned
concepts. Most of the existing approaches for neuro-symbolic artificial
intelligence are applied to static scenarios only, and the challenging setting
where reasoning along the temporal dimension is necessary has been seldom
explored. In this work we introduce LTLZinc, a benchmarking framework that can
be used to generate datasets covering a variety of different problems, against
which neuro-symbolic and continual learning methods can be evaluated along the
temporal and constraint-driven dimensions. Our framework generates expressive
temporal reasoning and continual learning tasks from a linear temporal logic
specification over MiniZinc constraints, and arbitrary image classification
datasets. Fine-grained annotations allow multiple neural and neuro-symbolic
training settings on the same generated datasets. Experiments on six
neuro-symbolic sequence classification and four class-continual learning tasks
generated by LTLZinc, demonstrate the challenging nature of temporal learning
and reasoning, and highlight limitations of current state-of-the-art methods.
We release the LTLZinc generator and ten ready-to-use tasks to the
neuro-symbolic and continual learning communities, in the hope of fostering
research towards unified temporal learning and reasoning frameworks.

### 7. [TAI Scan Tool: A RAG-Based Tool With Minimalistic Input for Trustworthy AI Self-Assessment](http://arxiv.org/pdf/2507.17514v1)

Authors: Athanasios Davvetas, Xenia Ziouvelou, Ypatia Dami, Alexis Kaponis, Konstantina Giouvanopoulou, Michael Papademas

This paper introduces the TAI Scan Tool, a RAG-based TAI self-assessment tool
with minimalistic input. The current version of the tool supports the legal TAI
assessment, with a particular emphasis on facilitating compliance with the AI
Act. It involves a two-step approach with a pre-screening and an assessment
phase. The assessment output of the system includes insight regarding the
risk-level of the AI system according to the AI Act, while at the same time
retrieving relevant articles to aid with compliance and notify on their
obligations. Our qualitative evaluation using use-case scenarios yields
promising results, correctly predicting risk levels while retrieving relevant
articles across three distinct semantic groups. Furthermore, interpretation of
results shows that the tool's reasoning relies on comparison with the setting
of high-risk systems, a behaviour attributed to their deployment requiring
careful consideration, and therefore frequently presented within the AI Act.

### 8. [Thinking Isn't an Illusion: Overcoming the Limitations of Reasoning Models via Tool Augmentations](http://arxiv.org/pdf/2507.17699v1)

Authors: Zhao Song, Song Yue, Jiahao Zhang

Large Reasoning Models (LRMs) have become a central focus in today's large
language model (LLM) research, where models are designed to output a
step-by-step thinking process before arriving at a final answer to handle
complex reasoning tasks. Despite their promise, recent empirical studies (e.g.,
[Shojaee et al., 2025] from Apple) suggest that this thinking process may not
actually enhance reasoning ability, where LLMs without explicit reasoning
actually outperform LRMs on tasks with low or high complexity. In this work, we
revisit these findings and investigate whether the limitations of LRMs persist
when tool augmentations are introduced. We incorporate two types of tools,
Python interpreters and scratchpads, and evaluate three representative LLMs and
their LRM counterparts on Apple's benchmark reasoning puzzles. Our results show
that, with proper tool use, LRMs consistently outperform their non-reasoning
counterparts across all levels of task complexity. These findings challenge the
recent narrative that reasoning is an illusion and highlight the potential of
tool-augmented LRMs for solving complex problems.

### 9. [Online Submission and Evaluation System Design for Competition Operations](http://arxiv.org/pdf/2507.17730v1)

Authors: Zhe Chen, Daniel Harabor, Ryan Hechnenberger, Nathan R. Sturtevant

Research communities have developed benchmark datasets across domains to
compare the performance of algorithms and techniques However, tracking the
progress in these research areas is not easy, as publications appear in
different venues at the same time, and many of them claim to represent the
state-of-the-art. To address this, research communities often organise periodic
competitions to evaluate the performance of various algorithms and techniques,
thereby tracking advancements in the field. However, these competitions pose a
significant operational burden. The organisers must manage and evaluate a large
volume of submissions. Furthermore, participants typically develop their
solutions in diverse environments, leading to compatibility issues during the
evaluation of their submissions. This paper presents an online competition
system that automates the submission and evaluation process for a competition.
The competition system allows organisers to manage large numbers of submissions
efficiently, utilising isolated environments to evaluate submissions. This
system has already been used successfully for several competitions, including
the Grid-Based Pathfinding Competition and the League of Robot Runners
competition.

### 10. [Reinforcement Learning Fine-Tunes a Sparse Subnetwork in Large Language Models](http://arxiv.org/pdf/2507.17107v1)

Authors: Andrii Balashov

Reinforcement learning (RL) is a key post-pretraining step for aligning large
language models (LLMs) with complex tasks and human preferences. While it is
often assumed that RL fine-tuning requires updating most of a model's
parameters, we challenge this assumption with a surprising finding: RL
fine-tuning consistently modifies only a small subnetwork (typically 5-30% of
weights), leaving most parameters unchanged. We call this phenomenon RL-induced
parameter update sparsity. It arises naturally, without any sparsity
constraints or parameter-efficient tuning, and appears across multiple RL
algorithms (e.g., PPO, DPO, SimPO, PRIME) and model families (e.g., OpenAI,
Meta, and open-source LLMs). Moreover, the subnetworks updated by RL show
substantial overlap across different seeds, datasets, and algorithms-far
exceeding chance-suggesting a partially transferable structure in the
pretrained model. We show that fine-tuning only this sparse subnetwork recovers
full model performance and yields parameters nearly identical to the fully
fine-tuned model. Our analysis suggests this sparsity emerges because RL
operates near the model's original distribution, requiring only targeted
changes. KL penalties, gradient clipping, and on-policy dynamics have limited
effect on the sparsity pattern. These findings shed new light on how RL adapts
models: not by shifting all weights, but by focusing training on a small,
consistently updated subnetwork. This insight enables more efficient RL methods
and reframes sparsity through the lens of the lottery ticket hypothesis.

### Computational Engineering

### 1. [RoadBench: A Vision-Language Foundation Model and Benchmark for Road Damage Understanding](http://arxiv.org/pdf/2507.17353v1)

Authors: Xi Xiao, Yunbei Zhang, Janet Wang, Lin Zhao, Yuxiang Wei, Hengjia Li, Yanshu Li, Xiao Wang, Swalpa Kumar Roy, Hao Xu, Tianyang Wang

Accurate road damage detection is crucial for timely infrastructure
maintenance and public safety, but existing vision-only datasets and models
lack the rich contextual understanding that textual information can provide. To
address this limitation, we introduce RoadBench, the first multimodal benchmark
for comprehensive road damage understanding. This dataset pairs high resolution
images of road damages with detailed textual descriptions, providing a richer
context for model training. We also present RoadCLIP, a novel vision language
model that builds upon CLIP by integrating domain specific enhancements. It
includes a disease aware positional encoding that captures spatial patterns of
road defects and a mechanism for injecting road-condition priors to refine the
model's understanding of road damages. We further employ a GPT driven data
generation pipeline to expand the image to text pairs in RoadBench, greatly
increasing data diversity without exhaustive manual annotation. Experiments
demonstrate that RoadCLIP achieves state of the art performance on road damage
recognition tasks, significantly outperforming existing vision-only models by
19.2%. These results highlight the advantages of integrating visual and textual
information for enhanced road condition analysis, setting new benchmarks for
the field and paving the way for more effective infrastructure monitoring
through multimodal learning.

### 2. [Reasoning-Driven Retrosynthesis Prediction with Large Language Models via Reinforcement Learning](http://arxiv.org/pdf/2507.17448v1)

Authors: Situo Zhang, Hanqi Li, Lu Chen, Zihan Zhao, Xuanze Lin, Zichen Zhu, Bo Chen, Xin Chen, Kai Yu

Retrosynthesis planning, essential in organic synthesis and drug discovery,
has greatly benefited from recent AI-driven advancements. Nevertheless,
existing methods frequently face limitations in both applicability and
explainability. Traditional graph-based and sequence-to-sequence models often
lack generalized chemical knowledge, leading to predictions that are neither
consistently accurate nor easily explainable. To address these challenges, we
introduce RetroDFM-R, a reasoning-based large language model (LLM) designed
specifically for chemical retrosynthesis. Leveraging large-scale reinforcement
learning guided by chemically verifiable rewards, RetroDFM-R significantly
enhances prediction accuracy and explainability. Comprehensive evaluations
demonstrate that RetroDFM-R significantly outperforms state-of-the-art methods,
achieving a top-1 accuracy of 65.0% on the USPTO-50K benchmark. Double-blind
human assessments further validate the chemical plausibility and practical
utility of RetroDFM-R's predictions. RetroDFM-R also accurately predicts
multistep retrosynthetic routes reported in the literature for both real-world
drug molecules and perovskite materials. Crucially, the model's explicit
reasoning process provides human-interpretable insights, thereby enhancing
trust and practical value in real-world retrosynthesis applications.

### Computation and Language

### 1. [CogDual: Enhancing Dual Cognition of LLMs via Reinforcement Learning with Implicit Rule-Based Rewards](http://arxiv.org/pdf/2507.17147v1)

Authors: Cheng Liu, Yifei Lu, Fanghua Ye, Jian Li, Xingyu Chen, Feiliang Ren, Zhaopeng Tu, Xiaolong Li

Role-Playing Language Agents (RPLAs) have emerged as a significant
application direction for Large Language Models (LLMs). Existing approaches
typically rely on prompt engineering or supervised fine-tuning to enable models
to imitate character behaviors in specific scenarios, but often neglect the
underlying \emph{cognitive} mechanisms driving these behaviors. Inspired by
cognitive psychology, we introduce \textbf{CogDual}, a novel RPLA adopting a
\textit{cognize-then-respond } reasoning paradigm. By jointly modeling external
situational awareness and internal self-awareness, CogDual generates responses
with improved character consistency and contextual alignment. To further
optimize the performance, we employ reinforcement learning with two
general-purpose reward schemes designed for open-domain text generation.
Extensive experiments on the CoSER benchmark, as well as Cross-MR and
LifeChoice, demonstrate that CogDual consistently outperforms existing
baselines and generalizes effectively across diverse role-playing tasks.

### 2. [FinGAIA: An End-to-End Benchmark for Evaluating AI Agents in Finance](http://arxiv.org/pdf/2507.17186v1)

Authors: Lingfeng Zeng, Fangqi Lou, Zixuan Wang, Jiajie Xu, Jinyi Niu, Mengping Li, Yifan Dong, Qi Qi, Wei Zhang, Ziwei Yang, Jun Han, Ruilun Feng, Ruiqi Hu, Lejie Zhang, Zhengbo Feng, Yicheng Ren, Xin Guo, Zhaowei Liu, Dongpo Cheng, Weige Cai, Liwen Zhang

The booming development of AI agents presents unprecedented opportunities for
automating complex tasks across various domains. However, their multi-step,
multi-tool collaboration capabilities in the financial sector remain
underexplored. This paper introduces FinGAIA, an end-to-end benchmark designed
to evaluate the practical abilities of AI agents in the financial domain.
FinGAIA comprises 407 meticulously crafted tasks, spanning seven major
financial sub-domains: securities, funds, banking, insurance, futures, trusts,
and asset management. These tasks are organized into three hierarchical levels
of scenario depth: basic business analysis, asset decision support, and
strategic risk management. We evaluated 10 mainstream AI agents in a zero-shot
setting. The best-performing agent, ChatGPT, achieved an overall accuracy of
48.9\%, which, while superior to non-professionals, still lags financial
experts by over 35 percentage points. Error analysis has revealed five
recurring failure patterns: Cross-modal Alignment Deficiency, Financial
Terminological Bias, Operational Process Awareness Barrier, among others. These
patterns point to crucial directions for future research. Our work provides the
first agent benchmark closely related to the financial domain, aiming to
objectively assess and promote the development of agents in this crucial field.
Partial data is available at https://github.com/SUFE-AIFLM-Lab/FinGAIA.

### 3. [CLARIFID: Improving Radiology Report Generation by Reinforcing Clinically Accurate Impressions and Enforcing Detailed Findings](http://arxiv.org/pdf/2507.17234v1)

Authors: Kyeongkyu Lee, Seonghwan Yoon, Hongki Lim

Automatic generation of radiology reports has the potential to alleviate
radiologists' significant workload, yet current methods struggle to deliver
clinically reliable conclusions. In particular, most prior approaches focus on
producing fluent text without effectively ensuring the factual correctness of
the reports and often rely on single-view images, limiting diagnostic
comprehensiveness. We propose CLARIFID, a novel framework that directly
optimizes diagnostic correctness by mirroring the two-step workflow of experts.
Specifically, CLARIFID (1) learns the logical flow from Findings to Impression
through section-aware pretraining, (2) is fine-tuned with Proximal Policy
Optimization in which the CheXbert F1 score of the Impression section serves as
the reward, (3) enforces reasoning-aware decoding that completes "Findings"
before synthesizing the "Impression", and (4) fuses multiple chest X-ray views
via a vision-transformer-based multi-view encoder. During inference, we apply a
reasoning-aware next-token forcing strategy followed by report-level
re-ranking, ensuring that the model first produces a comprehensive Findings
section before synthesizing the Impression and thereby preserving coherent
clinical reasoning. Experimental results on the MIMIC-CXR dataset demonstrate
that our method achieves superior clinical efficacy and outperforms existing
baselines on both standard NLG metrics and clinically aware scores.

### 4. [Investigating Subjective Factors of Argument Strength: Storytelling, Emotions, and Hedging](http://arxiv.org/pdf/2507.17409v1)

Authors: Carlotta Quensel, Neele Falk, Gabriella Lapesa

In assessing argument strength, the notions of what makes a good argument are
manifold. With the broader trend towards treating subjectivity as an asset and
not a problem in NLP, new dimensions of argument quality are studied. Although
studies on individual subjective features like personal stories exist, there is
a lack of large-scale analyses of the relation between these features and
argument strength. To address this gap, we conduct regression analysis to
quantify the impact of subjective factors $-$ emotions, storytelling, and
hedging $-$ on two standard datasets annotated for objective argument quality
and subjective persuasion. As such, our contribution is twofold: at the level
of contributed resources, as there are no datasets annotated with all studied
dimensions, this work compares and evaluates automated annotation methods for
each subjective feature. At the level of novel insights, our regression
analysis uncovers different patterns of impact of subjective features on the
two facets of argument strength encoded in the datasets. Our results show that
storytelling and hedging have contrasting effects on objective and subjective
argument quality, while the influence of emotions depends on their rhetoric
utilization rather than the domain.

### 5. [Synthetic Voice Data for Automatic Speech Recognition in African Languages](http://arxiv.org/pdf/2507.17578v1)

Authors: Brian DeRenzi, Anna Dixon, Mohamed Aymane Farhi, Christian Resch

Speech technology remains out of reach for most of the over 2300 languages in
Africa. We present the first systematic assessment of large-scale synthetic
voice corpora for African ASR. We apply a three-step process: LLM-driven text
creation, TTS voice synthesis, and ASR fine-tuning. Eight out of ten languages
for which we create synthetic text achieved readability scores above 5 out of
7. We evaluated ASR improvement for three (Hausa, Dholuo, Chichewa) and created
more than 2,500 hours of synthetic voice data at below 1% of the cost of real
data. Fine-tuned Wav2Vec-BERT-2.0 models trained on 250h real and 250h
synthetic Hausa matched a 500h real-data-only baseline, while 579h real and
450h to 993h synthetic data created the best performance. We also present
gender-disaggregated ASR performance evaluation. For very low-resource
languages, gains varied: Chichewa WER improved about 6.5% relative with a 1:2
real-to-synthetic ratio; a 1:1 ratio for Dholuo showed similar improvements on
some evaluation data, but not on others. Investigating intercoder reliability,
ASR errors and evaluation datasets revealed the need for more robust reviewer
protocols and more accurate evaluation data. All data and models are publicly
released to invite further work to improve synthetic data for African
languages.

### 6. [Who Attacks, and Why? Using LLMs to Identify Negative Campaigning in 18M Tweets across 19 Countries](http://arxiv.org/pdf/2507.17636v1)

Authors: Victor Hartman, Petter Törnberg

Negative campaigning is a central feature of political competition, yet
empirical research has been limited by the high cost and limited scalability of
existing classification methods. This study makes two key contributions. First,
it introduces zero-shot Large Language Models (LLMs) as a novel approach for
cross-lingual classification of negative campaigning. Using benchmark datasets
in ten languages, we demonstrate that LLMs achieve performance on par with
native-speaking human coders and outperform conventional supervised machine
learning approaches. Second, we leverage this novel method to conduct the
largest cross-national study of negative campaigning to date, analyzing 18
million tweets posted by parliamentarians in 19 European countries between 2017
and 2022. The results reveal consistent cross-national patterns: governing
parties are less likely to use negative messaging, while ideologically extreme
and populist parties -- particularly those on the radical right -- engage in
significantly higher levels of negativity. These findings advance our
understanding of how party-level characteristics shape strategic communication
in multiparty systems. More broadly, the study demonstrates the potential of
LLMs to enable scalable, transparent, and replicable research in political
communication across linguistic and cultural contexts.

### 7. [Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models](http://arxiv.org/pdf/2507.17702v1)

Authors: Changxin Tian, Kunlong Chen, Jia Liu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou

Mixture-of-Experts (MoE) has become a dominant architecture for scaling Large
Language Models (LLMs) efficiently by decoupling total parameters from
computational cost. However, this decoupling creates a critical challenge:
predicting the model capacity of a given MoE configurations (e.g., expert
activation ratio and granularity) remains an unresolved problem. To address
this gap, we introduce Efficiency Leverage (EL), a metric quantifying the
computational advantage of an MoE model over a dense equivalent. We conduct a
large-scale empirical study, training over 300 models up to 28B parameters, to
systematically investigate the relationship between MoE architectural
configurations and EL. Our findings reveal that EL is primarily driven by the
expert activation ratio and the total compute budget, both following
predictable power laws, while expert granularity acts as a non-linear modulator
with a clear optimal range. We integrate these discoveries into a unified
scaling law that accurately predicts the EL of an MoE architecture based on its
configuration. To validate our derived scaling laws, we designed and trained
Ling-mini-beta, a pilot model for Ling-2.0 series with only 0.85B active
parameters, alongside a 6.1B dense model for comparison. When trained on an
identical 1T high-quality token dataset, Ling-mini-beta matched the performance
of the 6.1B dense model while consuming over 7x fewer computational resources,
thereby confirming the accuracy of our scaling laws. This work provides a
principled and empirically-grounded foundation for the scaling of efficient MoE
models.

### 8. [TyDi QA-WANA: A Benchmark for Information-Seeking Question Answering in Languages of West Asia and North Africa](http://arxiv.org/pdf/2507.17709v1)

Authors: Parker Riley, Siamak Shakeri, Waleed Ammar, Jonathan H. Clark

We present TyDi QA-WANA, a question-answering dataset consisting of 28K
examples divided among 10 language varieties of western Asia and northern
Africa. The data collection process was designed to elicit information-seeking
questions, where the asker is genuinely curious to know the answer. Each
question in paired with an entire article that may or may not contain the
answer; the relatively large size of the articles results in a task suitable
for evaluating models' abilities to utilize large text contexts in answering
questions. Furthermore, the data was collected directly in each language
variety, without the use of translation, in order to avoid issues of cultural
relevance. We present performance of two baseline models, and release our code
and data to facilitate further improvement by the research community.

### 9. [Megrez2 Technical Report](http://arxiv.org/pdf/2507.17728v1)

Authors: Boxun Li, Yadong Li, Zhiyuan Li, Congyi Liu, Weilin Liu, Guowei Niu, Zheyue Tan, Haiyang Xu, Zhuyu Yao, Tao Yuan, Dong Zhou, Yueqing Zhuang, Bo Zhao, Guohao Dai, Yu Wang

We present Megrez2, a novel lightweight and high-performance language model
architecture optimized for device native deployment. Megrez2 introduces a novel
cross-layer expert sharing mechanism, which significantly reduces total
parameter count by reusing expert modules across adjacent transformer layers
while maintaining most of the model's capacity. It also incorporates pre-gated
routing, enabling memory-efficient expert loading and faster inference. As the
first instantiation of the Megrez2 architecture, we introduce the
Megrez2-Preview model, which is pre-trained on a 5-trillion-token corpus and
further enhanced through supervised fine-tuning and reinforcement learning with
verifiable rewards. With only 3B activated and 7.5B stored parameters,
Megrez2-Preview demonstrates competitive or superior performance compared to
larger models on a wide range of tasks, including language understanding,
instruction following, mathematical reasoning, and code generation. These
results highlight the effectiveness of the Megrez2 architecture to achieve a
balance between accuracy, efficiency, and deployability, making it a strong
candidate for real-world, resource-constrained applications.

### 10. [SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs](http://arxiv.org/pdf/2507.17178v1)

Authors: Zhiqiang Liu, Enpei Niu, Yin Hua, Mengshu Sun, Lei Liang, Huajun Chen, Wen Zhang

Although large language models (LLMs) have made significant progress in
understanding Structured Knowledge (SK) like KG and Table, existing evaluations
for SK understanding are non-rigorous (i.e., lacking evaluations of specific
capabilities) and focus on a single type of SK. Therefore, we aim to propose a
more comprehensive and rigorous structured knowledge understanding benchmark to
diagnose the shortcomings of LLMs. In this paper, we introduce SKA-Bench, a
Structured Knowledge Augmented QA Benchmark that encompasses four widely used
structured knowledge forms: KG, Table, KG+Text, and Table+Text. We utilize a
three-stage pipeline to construct SKA-Bench instances, which includes a
question, an answer, positive knowledge units, and noisy knowledge units. To
evaluate the SK understanding capabilities of LLMs in a fine-grained manner, we
expand the instances into four fundamental ability testbeds: Noise Robustness,
Order Insensitivity, Information Integration, and Negative Rejection. Empirical
evaluations on 8 representative LLMs, including the advanced DeepSeek-R1,
indicate that existing LLMs still face significant challenges in understanding
structured knowledge, and their performance is influenced by factors such as
the amount of noise, the order of knowledge units, and hallucination
phenomenon. Our dataset and code are available at
https://github.com/Lza12a/SKA-Bench.

### Cryptography and Security

### 1. [A Privacy-Preserving Data Collection Method for Diversified Statistical Analysis](http://arxiv.org/pdf/2507.17180v1)

Authors: Hao Jiang, Quan Zhou, Dongdong Zhao, Shangshang Yang, Wenjian Luo, Xingyi Zhang

Data perturbation-based privacy-preserving methods have been widely adopted
in various scenarios due to their efficiency and the elimination of the need
for a trusted third party. However, these methods primarily focus on individual
statistical indicators, neglecting the overall quality of the collected data
from a distributional perspective. Consequently, they often fall short of
meeting the diverse statistical analysis requirements encountered in practical
data analysis. As a promising sensitive data perturbation method, negative
survey methods is able to complete the task of collecting sensitive information
distribution while protecting personal privacy. Yet, existing negative survey
methods are primarily designed for discrete sensitive information and are
inadequate for real-valued data distributions. To bridge this gap, this paper
proposes a novel real-value negative survey model, termed RVNS, for the first
time in the field of real-value sensitive information collection. The RVNS
model exempts users from the necessity of discretizing their data and only
requires them to sample a set of data from a range that deviates from their
actual sensitive details, thereby preserving the privacy of their genuine
information. Moreover, to accurately capture the distribution of sensitive
information, an optimization problem is formulated, and a novel approach is
employed to solve it. Rigorous theoretical analysis demonstrates that the RVNS
model conforms to the differential privacy model, ensuring robust privacy
preservation. Comprehensive experiments conducted on both synthetic and
real-world datasets further validate the efficacy of the proposed method.

### 2. [Threshold-Protected Searchable Sharing: Privacy Preserving Aggregated-ANN Search for Collaborative RAG](http://arxiv.org/pdf/2507.17199v1)

Authors: Ruoyang Rykie Guo

LLM-powered search services have driven data integration as a significant
trend. However, this trend's progress is fundamentally hindered, despite the
fact that combining individual knowledge can significantly improve the
relevance and quality of responses in specialized queries and make AI more
professional at providing services. Two key bottlenecks are private data
repositories' locality constraints and the need to maintain compatibility with
mainstream search techniques, particularly Hierarchical Navigable Small World
(HNSW) indexing for high-dimensional vector spaces. In this work, we develop a
secure and privacy-preserving aggregated approximate nearest neighbor search
(SP-A$^2$NN) with HNSW compatibility under a threshold-based searchable sharing
primitive. A sharable bitgraph structure is constructed and extended to support
searches and dynamical insertions over shared data without compromising the
underlying graph topology. The approach reduces the complexity of a search from
$O(n^2)$ to $O(n)$ compared to naive (undirected) graph-sharing approach when
organizing graphs in the identical HNSW manner.
  On the theoretical front, we explore a novel security analytical framework
that incorporates privacy analysis via reductions. The proposed
leakage-guessing proof system is built upon an entirely different interactive
game that is independent of existing coin-toss game design. Rather than being
purely theoretical, this system is rooted in existing proof systems but goes
beyond them to specifically address leakage concerns and standardize leakage
analysis -- one of the most critical security challenges with AI's rapid
development.

### 3. [An Empirical Study on Virtual Reality Software Security Weaknesses](http://arxiv.org/pdf/2507.17324v1)

Authors: Yifan Xu, Jinfu Chen, Zhenyu Qi, Huashan Chen, Junyi Wang, Pengfei Hu, Feng Liu, Sen He

Virtual Reality (VR) has emerged as a transformative technology across
industries, yet its security weaknesses, including vulnerabilities, are
underinvestigated. This study investigates 334 VR projects hosted on GitHub,
examining 1,681 software security weaknesses to understand: what types of
weaknesses are prevalent in VR software; {\em when} and {\em how} weaknesses
are introduced; how long they have survived; and how they have been removed.
Due to the limited availability of VR software security weaknesses in public
databases (e.g., the National Vulnerability Database or NVD), we prepare the
{first systematic} dataset of VR software security weaknesses by introducing a
novel framework to collect such weaknesses from GitHub commit data. Our
empirical study on the dataset leads to useful insights, including: (i) VR
weaknesses are heavily skewed toward user interface weaknesses, followed by
resource-related weaknesses; (ii) VR development tools pose higher security
risks than VR applications; (iii) VR security weaknesses are often introduced
at the VR software birth time.

### 4. [A Zero-overhead Flow for Security Closure](http://arxiv.org/pdf/2507.17385v1)

Authors: Mohammad Eslami, Ashira Johara, Kyungbin Park, Samuel Pagliarini

In the traditional Application-Specific Integrated Circuit (ASIC) design
flow, the concept of timing closure implies to reach convergence during
physical synthesis such that, under a given area and power budget, the design
works at the targeted frequency. However, security has been largely neglected
when evaluating the Quality of Results (QoR) from physical synthesis. In
general, commercial place & route tools do not understand security goals. In
this work, we propose a modified ASIC design flow that is security-aware and,
differently from prior research, does not degrade QoR for the sake of security
improvement. Therefore, we propose a first-of-its-kind zero-overhead flow for
security closure. Our flow is concerned with two distinct threat models: (i)
insertion of Hardware Trojans (HTs) and (ii) physical probing/fault injection.
Importantly, the flow is entirely executed within a commercial place & route
engine and is scalable. In several metrics, our security-aware flow achieves
the best-known results for the ISPD`22 set of benchmark circuits while
incurring negligible design overheads due to security-related strategies.
Finally, we open source the entire methodology (as a set of scripts) and also
share the protected circuits (as design databases) for the benefit of the
hardware security community.

### 5. [Frequency Estimation of Correlated Multi-attribute Data under Local Differential Privacy](http://arxiv.org/pdf/2507.17516v1)

Authors: Shafizur Rahman Seeam, Ye Zheng, Yidan Hu

Large-scale data collection, from national censuses to IoT-enabled smart
homes, routinely gathers dozens of attributes per individual. These
multi-attribute datasets are vital for analytics but pose significant privacy
risks. Local Differential Privacy (LDP) is a powerful tool to protect user data
privacy by allowing users to locally perturb their records before releasing to
an untrusted data aggregator. However, existing LDP mechanisms either split the
privacy budget across all attributes or treat each attribute independently,
ignoring natural inter-attribute correlations. This leads to excessive noise or
fragmented budgets, resulting in significant utility loss, particularly in
high-dimensional settings.
  To overcome these limitations, we propose Correlated Randomized Response
(Corr-RR), a novel LDP mechanism that leverages correlations among attributes
to substantially improve utility while maintaining rigorous LDP guarantees.
Corr-RR allocates the full privacy budget to perturb a single, randomly
selected attribute and reconstructs the remaining attributes using estimated
interattribute dependencies, without incurring additional privacy cost. To
enable this, Corr-RR operates in two phases: (1) a subset of users apply
standard LDP mechanisms to estimate correlations, and (2) each remaining user
perturbs one attribute and infers the others using the learned correlations. We
theoretically prove that Corr-RR satisfies $\epsilon$-LDP, and extensive
experiments on synthetic and real-world datasets demonstrate that Corr-RR
consistently outperforms state-of-the-art LDP mechanisms, particularly in
scenarios with many attributes and strong inter-attribute correlations.

### 6. [Quantifying the ROI of Cyber Threat Intelligence: A Data-Driven Approach](http://arxiv.org/pdf/2507.17628v1)

Authors: Matteo Strada

The valuation of Cyber Threat Intelligence (CTI) remains a persistent
challenge due to the problem of negative evidence: successful threat prevention
results in non-events that generate minimal observable financial impact, making
CTI expenditures difficult to justify within traditional cost-benefit
frameworks. This study introduces a data-driven methodology for quantifying the
return on investment (ROI) of CTI, thereby reframing it as a measurable
contributor to risk mitigation. The proposed framework extends established
models in security economics, including the Gordon-Loeb and FAIR models, to
account for CTI's complex influence on both the probability of security
breaches and the severity of associated losses. The framework is
operationalized through empirically grounded performance indicators, such as
reductions in mean time to detect (MTTD), mean time to respond (MTTR), and
adversary dwell time, supported by three sector-specific case studies in
finance, healthcare, and retail. To address limitations in conventional linear
assessment methodologies, the Threat Intelligence Effectiveness Index (TIEI) is
introduced as a composite metric based on a weighted geometric mean. TIEI
penalizes underperformance across critical dimensions: quality, enrichment,
integration, and operational impact; thereby capturing bottleneck effect where
the least effective component limits overall performance. By integrating
financial quantification, adversarial coverage, and qualitative assessments of
business enablement, the proposed hybrid model converts negative evidence into
a justifiable ROI explanation. This approach offers a replicable means of
repositioning CTI from an expense to a strategic investment, enabling informed
decision-making and continuous optimization across diverse organizational
contexts.

### 7. [Tab-MIA: A Benchmark Dataset for Membership Inference Attacks on Tabular Data in LLMs](http://arxiv.org/pdf/2507.17259v1)

Authors: Eyal German, Sagiv Antebi, Daniel Samira, Asaf Shabtai, Yuval Elovici

Large language models (LLMs) are increasingly trained on tabular data, which,
unlike unstructured text, often contains personally identifiable information
(PII) in a highly structured and explicit format. As a result, privacy risks
arise, since sensitive records can be inadvertently retained by the model and
exposed through data extraction or membership inference attacks (MIAs). While
existing MIA methods primarily target textual content, their efficacy and
threat implications may differ when applied to structured data, due to its
limited content, diverse data types, unique value distributions, and
column-level semantics. In this paper, we present Tab-MIA, a benchmark dataset
for evaluating MIAs on tabular data in LLMs and demonstrate how it can be used.
Tab-MIA comprises five data collections, each represented in six different
encoding formats. Using our Tab-MIA benchmark, we conduct the first evaluation
of state-of-the-art MIA methods on LLMs finetuned with tabular data across
multiple encoding formats. In the evaluation, we analyze the memorization
behavior of pretrained LLMs on structured data derived from Wikipedia tables.
Our findings show that LLMs memorize tabular data in ways that vary across
encoding formats, making them susceptible to extraction via MIAs. Even when
fine-tuned for as few as three epochs, models exhibit high vulnerability, with
AUROC scores approaching 90% in most cases. Tab-MIA enables systematic
evaluation of these risks and provides a foundation for developing
privacy-preserving methods for tabular data in LLMs.

### 8. [Active Attack Resilience in 5G: A New Take on Authentication and Key Agreement](http://arxiv.org/pdf/2507.17491v1)

Authors: Nazatul H. Sultan, Xinlong Guan, Josef Pieprzyk, Wei Ni, Sharif Abuadbba, Hajime Suzuki

As 5G networks expand into critical infrastructure, secure and efficient user
authentication is more important than ever. The 5G-AKA protocol, standardized
by 3GPP in TS 33.501, is central to authentication in current 5G deployments.
It provides mutual authentication, user privacy, and key secrecy. However,
despite its adoption, 5G-AKA has known limitations in both security and
performance. While it focuses on protecting privacy against passive attackers,
recent studies show its vulnerabilities to active attacks. It also relies on a
sequence number mechanism to prevent replay attacks, requiring perfect
synchronization between the device and the core network. This stateful design
adds complexity, causes desynchronization, and incurs extra communication
overhead. More critically, 5G-AKA lacks Perfect Forward Secrecy (PFS), exposing
past communications if long-term keys are compromised-an increasing concern
amid sophisticated threats. This paper proposes an enhanced authentication
protocol that builds on 5G-AKA's design while addressing its shortcomings.
First, we introduce a stateless version that removes sequence number reliance,
reducing complexity while staying compatible with existing SIM cards and
infrastructure. We then extend this design to add PFS with minimal
cryptographic overhead. Both protocols are rigorously analyzed using ProVerif,
confirming their compliance with all major security requirements, including
resistance to passive and active attacks, as well as those defined by 3GPP and
academic studies. We also prototype both protocols and evaluate their
performance against 5G-AKA and 5G-AKA' (USENIX'21). Our results show the
proposed protocols offer stronger security with only minor computational
overhead, making them practical, future-ready solutions for 5G and beyond.

### 9. [Encrypted-State Quantum Compilation Scheme Based on Quantum Circuit Obfuscation](http://arxiv.org/pdf/2507.17589v1)

Authors: Chenyi Zhang, Tao Shang, Xueyi Guo

With the rapid advancement of quantum computing, quantum compilation has
become a crucial layer connecting high-level algorithms with physical hardware.
In quantum cloud computing, compilation is performed on the cloud side, which
exposes user circuits to potential risks such as structural leakage and output
predictability. To address these issues, we propose the encrypted-state quantum
compilation scheme based on quantum circuit obfuscation (ECQCO), the first
secure compilation framework tailored for the co-location of compilers and
quantum hardware. It applies quantum homomorphic encryption to conceal output
states and instantiates a structure obfuscation mechanism based on quantum
indistinguishability obfuscation, effectively protecting both functionality and
topology of the circuit. Additionally, an adaptive decoupling obfuscation
algorithm is designed to suppress potential idle errors while inserting pulse
operations. The proposed scheme achieves information-theoretic security and
guarantees computational indistinguishability under the quantum random oracle
model. Experimental results on benchmark datasets show that ECQCO achieves a
TVD of up to 0.7 and a normalized GED of 0.88, enhancing compilation-stage
security. Moreover, it introduces only a slight increase in circuit depth,
while keeping the average fidelity change within 1%, thus achieving a practical
balance between security and efficiency.

### 10. [Quantum Software Security Challenges within Shared Quantum Computing Environments](http://arxiv.org/pdf/2507.17712v1)

Authors: Samuel Ovaskainen, Majid Haghparast, Tommi Mikkonen

The number of qubits in quantum computers keeps growing, but most quantum
programs remain relatively small because of the noisy nature of the underlying
quantum hardware. This might lead quantum cloud providers to explore increased
hardware utilization, and thus profitability through means such as
multi-programming, which would allow the execution of multiple programs in
parallel. The adoption of such technology would bring entirely new challenges
to the field of quantum software security. This article explores and reports
the key challenges identified in quantum software security within shared
quantum computing environments.

### Computer Vision and Pattern Recognition

### 1. [FedVLM: Scalable Personalized Vision-Language Models through Federated Learning](http://arxiv.org/pdf/2507.17088v1)

Authors: Arkajyoti Mitra, Afia Anjum, Paul Agbaje, Mert Pesé, Habeeb Olufowobi

Vision-language models (VLMs) demonstrate impressive zero-shot and few-shot
learning capabilities, making them essential for several downstream tasks.
However, fine-tuning these models at scale remains challenging, particularly in
federated environments where data is decentralized and non-iid across clients.
Existing parameter-efficient tuning methods like LoRA (Low-Rank Adaptation)
reduce computational overhead but struggle with heterogeneous client data,
leading to suboptimal generalization. To address these challenges, we propose
FedVLM, a federated LoRA fine-tuning framework that enables decentralized
adaptation of VLMs while preserving model privacy and reducing reliance on
centralized training. To further tackle data heterogeneity, we introduce
personalized LoRA (pLoRA), which dynamically adapts LoRA parameters to each
client's unique data distribution, significantly improving local adaptation
while maintaining global model aggregation. Experiments on the RLAIF-V dataset
show that pLoRA improves client-specific performance by 24.5% over standard
LoRA, demonstrating superior adaptation in non-iid settings. FedVLM provides a
scalable and efficient solution for fine-tuning VLMs in federated settings,
advancing personalized adaptation in distributed learning scenarios.

### 2. [UNICE: Training A Universal Image Contrast Enhancer](http://arxiv.org/pdf/2507.17157v1)

Authors: Ruodai Cui, Lei Zhang

Existing image contrast enhancement methods are typically designed for
specific tasks such as under-/over-exposure correction, low-light and backlit
image enhancement, etc. The learned models, however, exhibit poor
generalization performance across different tasks, even across different
datasets of a specific task. It is important to explore whether we can learn a
universal and generalized model for various contrast enhancement tasks. In this
work, we observe that the common key factor of these tasks lies in the need of
exposure and contrast adjustment, which can be well-addressed if high-dynamic
range (HDR) inputs are available. We hence collect 46,928 HDR raw images from
public sources, and render 328,496 sRGB images to build multi-exposure
sequences (MES) and the corresponding pseudo sRGB ground-truths via
multi-exposure fusion. Consequently, we train a network to generate an MES from
a single sRGB image, followed by training another network to fuse the generated
MES into an enhanced image. Our proposed method, namely UNiversal Image
Contrast Enhancer (UNICE), is free of costly human labeling. However, it
demonstrates significantly stronger generalization performance than existing
image contrast enhancement methods across and within different tasks, even
outperforming manually created ground-truths in multiple no-reference image
quality metrics. The dataset, code and model are available at
https://github.com/BeyondHeaven/UNICE.

### 3. [DOOMGAN:High-Fidelity Dynamic Identity Obfuscation Ocular Generative Morphing](http://arxiv.org/pdf/2507.17158v1)

Authors: Bharath Krishnamurthy, Ajita Rattani

Ocular biometrics in the visible spectrum have emerged as a prominent
modality due to their high accuracy, resistance to spoofing, and non-invasive
nature. However, morphing attacks, synthetic biometric traits created by
blending features from multiple individuals, threaten biometric system
integrity. While extensively studied for near-infrared iris and face
biometrics, morphing in visible-spectrum ocular data remains underexplored.
Simulating such attacks demands advanced generation models that handle
uncontrolled conditions while preserving detailed ocular features like iris
boundaries and periocular textures. To address this gap, we introduce DOOMGAN,
that encompasses landmark-driven encoding of visible ocular anatomy,
attention-guided generation for realistic morph synthesis, and dynamic
weighting of multi-faceted losses for optimized convergence. DOOMGAN achieves
over 20% higher attack success rates than baseline methods under stringent
thresholds, along with 20% better elliptical iris structure generation and 30%
improved gaze consistency. We also release the first comprehensive ocular
morphing dataset to support further research in this domain.

### 4. [Multi-Scale PCB Defect Detection with YOLOv8 Network Improved via Pruning and Lightweight Network](http://arxiv.org/pdf/2507.17176v1)

Authors: Li Pingzhen, Xu Sheng, Chen Jing, Su Chengyue

With the high density of printed circuit board (PCB) design and the high
speed of production, the traditional PCB defect detection model is difficult to
take into account the accuracy and computational cost, and cannot meet the
requirements of high accuracy and real-time detection of tiny defects.
Therefore, in this paper, a multi-scale PCB defect detection method is improved
with YOLOv8 using a comprehensive strategy of tiny target sensitivity strategy,
network lightweighting and adaptive pruning, which is able to improve the
detection speed and accuracy by optimizing the backbone network, the neck
network and the detection head, the loss function and the adaptive pruning
rate. Firstly, a Ghost-HGNetv2 structure with fewer parameters is used in the
backbone network, and multilevel features are used to extract image semantic
features to discover accurate defects. Secondly, we integrate C2f-Faster with
small number of parameters in the neck section to enhance the ability of
multi-level feature fusion. Next, in the Head part, we design a new GCDetect
detection head, which allows the prediction of bounding boxes and categories to
share the weights of GroupConv, and uses a small number of grouping
convolutions to accomplish the regression and classification tasks, which
significantly reduces the number of parameters while maintaining the accuracy
of detection. We also design the Inner-MPDIoU boundary loss function to improve
the detection and localization of tiny targets. Finally, the model was pruned
by an optimized adaptive pruning rate to further reduce the complexity of the
model. Experimental results show that the model exhibits advantages in terms of
accuracy and speed. On the publicly available PCB defect dataset, mAP0.5
reaches 99.32% and mAP0.5:0.9 reaches 75.18%, which is 10.13% higher compared
to YOLOv8n.

### 5. [Hierarchical Fusion and Joint Aggregation: A Multi-Level Feature Representation Method for AIGC Image Quality Assessment](http://arxiv.org/pdf/2507.17182v1)

Authors: Linghe Meng, Jiarun Song

The quality assessment of AI-generated content (AIGC) faces multi-dimensional
challenges, that span from low-level visual perception to high-level semantic
understanding. Existing methods generally rely on single-level visual features,
limiting their ability to capture complex distortions in AIGC images. To
address this limitation, a multi-level visual representation paradigm is
proposed with three stages, namely multi-level feature extraction, hierarchical
fusion, and joint aggregation. Based on this paradigm, two networks are
developed. Specifically, the Multi-Level Global-Local Fusion Network (MGLF-Net)
is designed for the perceptual quality assessment, extracting complementary
local and global features via dual CNN and Transformer visual backbones. The
Multi-Level Prompt-Embedded Fusion Network (MPEF-Net) targets Text-to-Image
correspondence by embedding prompt semantics into the visual feature fusion
process at each feature level. The fused multi-level features are then
aggregated for final evaluation. Experiments on benchmarks demonstrate
outstanding performance on both tasks, validating the effectiveness of the
proposed multi-level visual assessment paradigm.

### 6. [Vec2Face+ for Face Dataset Generation](http://arxiv.org/pdf/2507.17192v1)

Authors: Haiyu Wu, Jaskirat Singh, Sicong Tian, Liang Zheng, Kevin W. Bowyer

When synthesizing identities as face recognition training data, it is
generally believed that large inter-class separability and intra-class
attribute variation are essential for synthesizing a quality dataset. % This
belief is generally correct, and this is what we aim for. However, when
increasing intra-class variation, existing methods overlook the necessity of
maintaining intra-class identity consistency. % To address this and generate
high-quality face training data, we propose Vec2Face+, a generative model that
creates images directly from image features and allows for continuous and easy
control of face identities and attributes. Using Vec2Face+, we obtain datasets
with proper inter-class separability and intra-class variation and identity
consistency using three strategies: 1) we sample vectors sufficiently different
from others to generate well-separated identities; 2) we propose an AttrOP
algorithm for increasing general attribute variations; 3) we propose LoRA-based
pose control for generating images with profile head poses, which is more
efficient and identity-preserving than AttrOP. % Our system generates VFace10K,
a synthetic face dataset with 10K identities, which allows an FR model to
achieve state-of-the-art accuracy on seven real-world test sets. Scaling the
size to 4M and 12M images, the corresponding VFace100K and VFace300K datasets
yield higher accuracy than the real-world training dataset, CASIA-WebFace, on
five real-world test sets. This is the first time a synthetic dataset beats the
CASIA-WebFace in average accuracy. In addition, we find that only 1 out of 11
synthetic datasets outperforms random guessing (\emph{i.e., 50\%}) in twin
verification and that models trained with synthetic identities are more biased
than those trained with real identities. Both are important aspects for future
investigation.

### 7. [VBCD: A Voxel-Based Framework for Personalized Dental Crown Design](http://arxiv.org/pdf/2507.17205v1)

Authors: Linda Wei, Chang Liu, Wenran Zhang, Zengji Zhang, Shaoting Zhang, Hongsheng Li

The design of restorative dental crowns from intraoral scans is
labor-intensive for dental technicians. To address this challenge, we propose a
novel voxel-based framework for automated dental crown design (VBCD). The VBCD
framework generates an initial coarse dental crown from voxelized intraoral
scans, followed by a fine-grained refiner incorporating distance-aware
supervision to improve accuracy and quality. During the training stage, we
employ the Curvature and Margin line Penalty Loss (CMPL) to enhance the
alignment of the generated crown with the margin line. Additionally, a
positional prompt based on the FDI tooth numbering system is introduced to
further improve the accuracy of the generated dental crowns. Evaluation on a
large-scale dataset of intraoral scans demonstrated that our approach
outperforms existing methods, providing a robust solution for personalized
dental crown design.

### 8. [MaskedCLIP: Bridging the Masked and CLIP Space for Semi-Supervised Medical Vision-Language Pre-training](http://arxiv.org/pdf/2507.17239v1)

Authors: Lei Zhu, Jun Zhou, Rick Siow Mong Goh, Yong Liu

Foundation models have recently gained tremendous popularity in medical image
analysis. State-of-the-art methods leverage either paired image-text data via
vision-language pre-training or unpaired image data via self-supervised
pre-training to learn foundation models with generalizable image features to
boost downstream task performance. However, learning foundation models
exclusively on either paired or unpaired image data limits their ability to
learn richer and more comprehensive image features. In this paper, we
investigate a novel task termed semi-supervised vision-language pre-training,
aiming to fully harness the potential of both paired and unpaired image data
for foundation model learning. To this end, we propose MaskedCLIP, a
synergistic masked image modeling and contrastive language-image pre-training
framework for semi-supervised vision-language pre-training. The key challenge
in combining paired and unpaired image data for learning a foundation model
lies in the incompatible feature spaces derived from these two types of data.
To address this issue, we propose to connect the masked feature space with the
CLIP feature space with a bridge transformer. In this way, the more semantic
specific CLIP features can benefit from the more general masked features for
semantic feature extraction. We further propose a masked knowledge distillation
loss to distill semantic knowledge of original image features in CLIP feature
space back to the predicted masked image features in masked feature space. With
this mutually interactive design, our framework effectively leverages both
paired and unpaired image data to learn more generalizable image features for
downstream tasks. Extensive experiments on retinal image analysis demonstrate
the effectiveness and data efficiency of our method.

### 9. [Perceptual Classifiers: Detecting Generative Images using Perceptual Features](http://arxiv.org/pdf/2507.17240v1)

Authors: Krishna Srikar Durbha, Asvin Kumar Venkataramanan, Rajesh Sureddi, Alan C. Bovik

Image Quality Assessment (IQA) models are employed in many practical image
and video processing pipelines to reduce storage, minimize transmission costs,
and improve the Quality of Experience (QoE) of millions of viewers. These
models are sensitive to a diverse range of image distortions and can accurately
predict image quality as judged by human viewers. Recent advancements in
generative models have resulted in a significant influx of "GenAI" content on
the internet. Existing methods for detecting GenAI content have progressed
significantly with improved generalization performance on images from unseen
generative models. Here, we leverage the capabilities of existing IQA models,
which effectively capture the manifold of real images within a bandpass
statistical space, to distinguish between real and AI-generated images. We
investigate the generalization ability of these perceptual classifiers to the
task of GenAI image detection and evaluate their robustness against various
image degradations. Our results show that a two-layer network trained on the
feature space of IQA models demonstrates state-of-the-art performance in
detecting fake images across generative models, while maintaining significant
robustness against image degradations.

### 10. [Unsupervised Exposure Correction](http://arxiv.org/pdf/2507.17252v1)

Authors: Ruodai Cui, Li Niu, Guosheng Hu

Current exposure correction methods have three challenges, labor-intensive
paired data annotation, limited generalizability, and performance degradation
in low-level computer vision tasks. In this work, we introduce an innovative
Unsupervised Exposure Correction (UEC) method that eliminates the need for
manual annotations, offers improved generalizability, and enhances performance
in low-level downstream tasks. Our model is trained using freely available
paired data from an emulated Image Signal Processing (ISP) pipeline. This
approach does not need expensive manual annotations, thereby minimizing
individual style biases from the annotation and consequently improving its
generalizability. Furthermore, we present a large-scale Radiometry Correction
Dataset, specifically designed to emphasize exposure variations, to facilitate
unsupervised learning. In addition, we develop a transformation function that
preserves image details and outperforms state-of-the-art supervised methods
[12], while utilizing only 0.01% of their parameters. Our work further
investigates the broader impact of exposure correction on downstream tasks,
including edge detection, demonstrating its effectiveness in mitigating the
adverse effects of poor exposure on low-level features. The source code and
dataset are publicly available at https://github.com/BeyondHeaven/uec_code.

### Computers and Society

### 1. [AI in Design Education at College Level-Educators' Perspectives and Challenges](http://arxiv.org/pdf/2507.17481v1)

Authors: Lizhu Zhang, Cecilia X. Wang

Artificial intelligence has deeply permeated numerous fields, especially the
design area which relies on technology as a tool for innovation. This change
naturally extends to the field of design education, which is closest to design
practice. This has led to further exploration of the impact of AI on
college-level education in the design discipline. This study aims to examine
how current design educators perceive the role of AI in college-level design
education, their perspectives on integrating AI into teaching and research, and
their concerns regarding its potential challenges in design education and
research. Through qualitative, semi-structured, in-depth interviews with seven
faculties in U.S. design colleges, the findings reveal that AI, as a tool and
source of information, has become an integral part of design education. AI-
derived functionalities are increasingly utilized in design software, and
educators are actively incorporating AI as a theoretical framework in their
teaching. Educators can guide students in using AI tools, but only if they
first acquire a strong foundation in basic design principles and skills. This
study also indicates the importance of promoting a cooperative relationship
between design educators and AI. At the same time, educators express
anticipation for advancements in ethical standards, authenticity, and the
resolution of copyright issues related to AI.

### 2. [Simulating multiple human perspectives in socio-ecological systems using large language models](http://arxiv.org/pdf/2507.17680v1)

Authors: Yongchao Zeng, Calum Brown, Ioannis Kyriakou, Ronja Hotz, Mark Rounsevell

Understanding socio-ecological systems requires insights from diverse
stakeholder perspectives, which are often hard to access. To enable
alternative, simulation-based exploration of different stakeholder
perspectives, we develop the HoPeS (Human-Oriented Perspective Shifting)
modelling framework. HoPeS employs agents powered by large language models
(LLMs) to represent various stakeholders; users can step into the agent roles
to experience perspectival differences. A simulation protocol serves as a
"scaffold" to streamline multiple perspective-taking simulations, supporting
users in reflecting on, transitioning between, and integrating across
perspectives. A prototype system is developed to demonstrate HoPeS in the
context of institutional dynamics and land use change, enabling both
narrative-driven and numerical experiments. In an illustrative experiment, a
user successively adopts the perspectives of a system observer and a researcher
- a role that analyses data from the embedded land use model to inform
evidence-based decision-making for other LLM agents representing various
institutions. Despite the user's effort to recommend technically sound
policies, discrepancies persist between the policy recommendation and
implementation due to stakeholders' competing advocacies, mirroring real-world
misalignment between researcher and policymaker perspectives. The user's
reflection highlights the subjective feelings of frustration and disappointment
as a researcher, especially due to the challenge of maintaining political
neutrality while attempting to gain political influence. Despite this, the user
exhibits high motivation to experiment with alternative narrative framing
strategies, suggesting the system's potential in exploring different
perspectives. Further system and protocol refinement are likely to enable new
forms of interdisciplinary collaboration in socio-ecological simulations.

### 3. [Our Cars Can Talk: How IoT Brings AI to Vehicles](http://arxiv.org/pdf/2507.17214v1)

Authors: Amod Kant Agrawal

Bringing AI to vehicles and enabling them as sensing platforms is key to
transforming maintenance from reactive to proactive. Now is the time to
integrate AI copilots that speak both languages: machine and driver. This
article offers a conceptual and technical perspective intended to spark
interdisciplinary dialogue and guide future research and development in
intelligent vehicle systems, predictive maintenance, and AI-powered user
interaction.

### 4. [Enabling Cyber Security Education through Digital Twins and Generative AI](http://arxiv.org/pdf/2507.17518v1)

Authors: Vita Santa Barletta, Vito Bavaro, Miriana Calvano, Antonio Curci, Antonio Piccinno, Davide Pio Posa

Digital Twins (DTs) are gaining prominence in cybersecurity for their ability
to replicate complex IT (Information Technology), OT (Operational Technology),
and IoT (Internet of Things) infrastructures, allowing for real time
monitoring, threat analysis, and system simulation. This study investigates how
integrating DTs with penetration testing tools and Large Language Models (LLMs)
can enhance cybersecurity education and operational readiness. By simulating
realistic cyber environments, this approach offers a practical, interactive
framework for exploring vulnerabilities and defensive strategies. At the core
of this research is the Red Team Knife (RTK), a custom penetration testing
toolkit aligned with the Cyber Kill Chain model. RTK is designed to guide
learners through key phases of cyberattacks, including reconnaissance,
exploitation, and response within a DT powered ecosystem. The incorporation of
Large Language Models (LLMs) further enriches the experience by providing
intelligent, real-time feedback, natural language threat explanations, and
adaptive learning support during training exercises. This combined DT LLM
framework is currently being piloted in academic settings to develop hands on
skills in vulnerability assessment, threat detection, and security operations.
Initial findings suggest that the integration significantly improves the
effectiveness and relevance of cybersecurity training, bridging the gap between
theoretical knowledge and real-world application. Ultimately, the research
demonstrates how DTs and LLMs together can transform cybersecurity education to
meet evolving industry demands.

### Databases

### 1. [Unfolding Data Quality Dimensions in Practice: A Survey](http://arxiv.org/pdf/2507.17507v1)

Authors: Vasileios Papastergios, Lisa Ehrlinger, Anastasios Gounaris

Data quality describes the degree to which data meet specific requirements
and are fit for use by humans and/or downstream tasks (e.g., artificial
intelligence). Data quality can be assessed across multiple high-level concepts
called dimensions, such as accuracy, completeness, consistency, or timeliness.
While extensive research and several attempts for standardization (e.g.,
ISO/IEC 25012) exist for data quality dimensions, their practical application
often remains unclear. In parallel to research endeavors, a large number of
tools have been developed that implement functionalities for the detection and
mitigation of specific data quality issues, such as missing values or outliers.
With this paper, we aim to bridge this gap between data quality theory and
practice by systematically connecting low-level functionalities offered by data
quality tools with high-level dimensions, revealing their many-to-many
relationships. Through an examination of seven open-source data quality tools,
we provide a comprehensive mapping between their functionalities and the data
quality dimensions, demonstrating how individual functionalities and their
variants partially contribute to the assessment of single dimensions. This
systematic survey provides both practitioners and researchers with a unified
view on the fragmented landscape of data quality checks, offering actionable
insights for quality assessment across multiple dimensions.

### 2. [SHINE: A Scalable HNSW Index in Disaggregated Memory](http://arxiv.org/pdf/2507.17647v1)

Authors: Manuel Widmoser, Daniel Kocher, Nikolaus Augsten

Approximate nearest neighbor (ANN) search is a fundamental problem in
computer science for which in-memory graph-based methods, such as Hierarchical
Navigable Small World (HNSW), perform exceptionally well. To scale beyond
billions of high-dimensional vectors, the index must be distributed. The
disaggregated memory architecture physically separates compute and memory into
two distinct hardware units and has become popular in modern data centers. Both
units are connected via RDMA networks that allow compute nodes to directly
access remote memory and perform all the computations, posing unique challenges
for disaggregated indexes.
  In this work, we propose a scalable HNSW index for ANN search in
disaggregated memory. In contrast to existing distributed approaches, which
partition the graph at the cost of accuracy, our method builds a
graph-preserving index that reaches the same accuracy as a single-machine HNSW.
Continuously fetching high-dimensional vector data from remote memory leads to
severe network bandwidth limitations, which we overcome by employing an
efficient caching mechanism. Since answering a single query involves processing
numerous unique graph nodes, caching alone is not sufficient to achieve high
scalability. We logically combine the caches of the compute nodes to increase
the overall cache effectiveness and confirm the efficiency and scalability of
our method in our evaluation.

### 3. [CQE under Epistemic Dependencies: Algorithms and Experiments (extended version)](http://arxiv.org/pdf/2507.17487v1)

Authors: Lorenzo Marconi, Flavia Ricci, Riccardo Rosati

We investigate Controlled Query Evaluation (CQE) over ontologies, where
information disclosure is regulated by epistemic dependencies (EDs), a family
of logical rules recently proposed for the CQE framework. In particular, we
combine EDs with the notion of optimal GA censors, i.e. maximal sets of ground
atoms that are entailed by the ontology and can be safely revealed. We focus on
answering Boolean unions of conjunctive queries (BUCQs) with respect to the
intersection of all optimal GA censors - an approach that has been shown in
other contexts to ensure strong security guarantees with favorable
computational behavior. First, we characterize the security of this
intersection-based approach and identify a class of EDs (namely, full EDs) for
which it remains safe. Then, for a subclass of EDs and for DL-Lite_R
ontologies, we show that answering BUCQs in the above CQE semantics is in AC^0
in data complexity by presenting a suitable, detailed first-order rewriting
algorithm. Finally, we report on experiments conducted in two different
evaluation scenarios, showing the practical feasibility of our rewriting
function.

### 4. [Triadic First-Order Logic Queries in Temporal Networks](http://arxiv.org/pdf/2507.17215v1)

Authors: Omkar Bhalerao, Yunjie Pan, C. Seshadhri, Nishil Talati

Motif counting is a fundamental problem in network analysis, and there is a
rich literature of theoretical and applied algorithms for this problem. Given a
large input network $G$, a motif $H$ is a small "pattern" graph indicative of
special local structure. Motif/pattern mining involves finding all matches of
this pattern in the input $G$. The simplest, yet challenging, case of motif
counting is when $H$ has three vertices, often called a "triadic" query. Recent
work has focused on "temporal graph mining", where the network $G$ has edges
with timestamps (and directions) and $H$ has time constraints.
  Inspired by concepts in logic and database theory, we introduce the study of
"thresholded First Order Logic (FOL) Motif Analysis" for massive temporal
networks. A typical triadic motif query asks for the existence of three
vertices that form a desired temporal pattern. An "FOL" motif query is obtained
by having both existential and thresholded universal quantifiers. This allows
for query semantics that can mine richer information from networks. A typical
triadic query would be "find all triples of vertices $u,v,w$ such that they
form a triangle within one hour". A thresholded FOL query can express "find all
pairs $u,v$ such that for half of $w$ where $(u,w)$ formed an edge, $(v,w)$
also formed an edge within an hour".
  We design the first algorithm, FOLTY, for mining thresholded FOL triadic
queries. The theoretical running time of FOLTY matches the best known running
time for temporal triangle counting in sparse graphs. We give an efficient
implementation of FOLTY using specialized temporal data structures. FOLTY has
excellent empirical behavior, and can answer triadic FOL queries on graphs with
nearly 70M edges is less than hour on commodity hardware. Our work has the
potential to start a new research direction in the classic well-studied problem
of motif analysis.

### 5. [Eco-Friendly AI: Unleashing Data Power for Green Federated Learning](http://arxiv.org/pdf/2507.17241v1)

Authors: Mattia Sabella, Monica Vitali

The widespread adoption of Artificial Intelligence (AI) and Machine Learning
(ML) comes with a significant environmental impact, particularly in terms of
energy consumption and carbon emissions. This pressing issue highlights the
need for innovative solutions to mitigate AI's ecological footprint. One of the
key factors influencing the energy consumption of ML model training is the size
of the training dataset. ML models are often trained on vast amounts of data
continuously generated by sensors and devices distributed across multiple
locations. To reduce data transmission costs and enhance privacy, Federated
Learning (FL) enables model training without the need to move or share raw
data. While FL offers these advantages, it also introduces challenges due to
the heterogeneity of data sources (related to volume and quality),
computational node capabilities, and environmental impact.
  This paper contributes to the advancement of Green AI by proposing a
data-centric approach to Green Federated Learning. Specifically, we focus on
reducing FL's environmental impact by minimizing the volume of training data.
Our methodology involves the analysis of the characteristics of federated
datasets, the selecting of an optimal subset of data based on quality metrics,
and the choice of the federated nodes with the lowest environmental impact. We
develop a comprehensive methodology that examines the influence of data-centric
factors, such as data quality and volume, on FL training performance and carbon
emissions. Building on these insights, we introduce an interactive
recommendation system that optimizes FL configurations through data reduction,
minimizing environmental impact during training. Applying this methodology to
time series classification has demonstrated promising results in reducing the
environmental impact of FL tasks.

### 6. [Symmetric Private Information Retrieval (SPIR) on Graph-Based Replicated Systems](http://arxiv.org/pdf/2507.17736v1)

Authors: Shreya Meel, Sennur Ulukus

We introduce the problem of symmetric private information retrieval (SPIR) on
replicated databases modeled by a simple graph. In this model, each vertex
corresponds to a server, and a message is replicated on two servers if and only
if there is an edge between them. We consider the setting where the server-side
common randomness necessary to accomplish SPIR is also replicated at the
servers according to the graph, and we call this as message-specific common
randomness. In this setting, we establish a lower bound on the SPIR capacity,
i.e., the maximum download rate, for general graphs, by proposing an achievable
SPIR scheme. Next, we prove that, for any SPIR scheme to be feasible, the
minimum size of message-specific randomness should be equal to the size of a
message. Finally, by providing matching upper bounds, we derive the exact SPIR
capacity for the class of path and regular graphs.

### Distributed, Parallel, and Cluster Computing

### 1. [PathWeaver: A High-Throughput Multi-GPU System for Graph-Based Approximate Nearest Neighbor Search](http://arxiv.org/pdf/2507.17094v1)

Authors: Sukjin Kim, Seongyeon Park, Si Ung Noh, Junguk Hong, Taehee Kwon, Hunseong Lim, Jinho Lee

Graph-based Approximate Nearest Neighbor Search (ANNS) is widely adopted in
numerous applications, such as recommendation systems, natural language
processing, and computer vision. While recent works on GPU-based acceleration
have significantly advanced ANNS performance, the ever-growing scale of
datasets now demands efficient multi-GPU solutions. However, the design of
existing works overlooks multi-GPU scalability, resulting in naive approaches
that treat additional GPUs as a means to extend memory capacity for large
datasets. This inefficiency arises from partitioning the dataset and
independently searching for data points similar to the queries in each GPU. We
therefore propose PathWeaver, a novel multi-GPU framework designed to scale and
accelerate ANNS for large datasets. First, we propose pipelining-based path
extension, a GPU-aware pipelining mechanism that reduces prior work's redundant
search iterations by leveraging GPU-to-GPU communication. Second, we design
ghost staging that leverages a representative dataset to identify optimal query
starting points, reducing the search space for challenging queries. Finally, we
introduce direction-guided selection, a data selection technique that filters
irrelevant points early in the search process, minimizing unnecessary memory
accesses and distance computations. Comprehensive evaluations across diverse
datasets demonstrate that PathWeaver achieves 3.24$\times$ geomean speedup and
up to 5.30$\times$ speedup on 95% recall rate over state-of-the-art
multi-GPU-based ANNS frameworks.

### 2. [Auto-scaling Approaches for Cloud-native Applications: A Survey and Taxonomy](http://arxiv.org/pdf/2507.17128v1)

Authors: Minxian Xu, Linfeng Wen, Junhan Liao, Huaming Wu, Kejiang Ye, Chengzhong Xu

The interactions within cloud-native applications are complex, with a
constantly changing number of services and loads, posing higher demands on
auto-scaling approach. This mainly involves several challenges such as
microservices dependency analysis, performance profiling, anomaly detection,
workload characterization and task co-location. Therefore, some advanced
algorithms have been investigated into auto-scaling cloud-native applications
to optimize system and application performance. These algorithms can learn from
historical data and appropriately adjust resource allocation based on the
current environment and load conditions to optimize resource utilization and
system performance. In this paper, we systematically review the literature on
state-of-the-art auto-scaling approaches for cloud-native applications from
2020, and further explore the technological evolution. Additionally, we propose
a detailed taxonomy to categorize current research from five perspectives,
including infrastructure, architecture, scaling methods, optimization
objectives, and behavior modeling. Then, we provide a comprehensive comparison
and in-depth discussion of the key features, advantages, limitations, and
application scenarios of each approach, considering their performance in
diverse environments and under various conditions. Finally, we summarize the
current state of research in this field, identify the gaps and unresolved
challenges, and emphasize promising directions for future exploration,
particularly in areas such as the application of large models, microservice
dependency management, and the use of meta-learning techniques to enhance model
applicability and adaptability across different environments.

### 3. [BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs](http://arxiv.org/pdf/2507.17133v1)

Authors: Jianmin Hu, Minxian Xu, Kejiang Ye, Chengzhong Xu

In recent years, the Mixture-of-Experts (MoE) architecture has been widely
applied to large language models (LLMs), providing a promising solution that
activates only a subset of the model's parameters during computation, thereby
reducing overall memory requirements and allowing for faster inference compared
to dense models. Despite these advantages, existing systems still face issues
of low efficiency due to static model placement and lack of dynamic workloads
adaptation. This leads to suboptimal resource utilization and increased
latency, especially during bursty requests periods.
  To address these challenges, this paper introduces BrownoutServe, a novel
serving framework designed to optimize inference efficiency and maintain
service reliability for MoE-based LLMs under dynamic computational demands and
traffic conditions. BrownoutServe introduces "united experts" that integrate
knowledge from multiple experts, reducing the times of expert access and
inference latency. Additionally, it proposes a dynamic brownout mechanism to
adaptively adjust the processing of certain tokens, optimizing inference
performance while guaranteeing service level objectives (SLOs) are met. Our
evaluations show the effectiveness of BrownoutServe under various workloads: it
achieves up to 2.07x throughput improvement compared to vLLM and reduces SLO
violations by 90.28%, showcasing its robustness under bursty traffic while
maintaining acceptable inference accuracy.

### 4. [Efficient Column-Wise N:M Pruning on RISC-V CPU](http://arxiv.org/pdf/2507.17301v1)

Authors: Chi-Wei Chu, Ding-Yong Hong, Jan-Jan Wu

In deep learning frameworks, weight pruning is a widely used technique for
improving computational efficiency by reducing the size of large models. This
is especially critical for convolutional operators, which often act as
performance bottlenecks in convolutional neural networks (CNNs). However, the
effectiveness of pruning heavily depends on how it is implemented, as different
methods can significantly impact both computational performance and memory
footprint. In this work, we propose a column-wise N:M pruning strategy applied
at the tile level and modify XNNPACK to enable efficient execution of pruned
models on the RISC-V vector architecture. Additionally, we propose fusing the
operations of im2col and data packing to minimize redundant memory accesses and
memory overhead. To further optimize performance, we incorporate AITemplate's
profiling technique to identify the optimal implementation for each
convolutional operator. Our proposed approach effectively increases ResNet
inference throughput by as much as 4.0x, and preserves ImageNet top-1 accuracy
within 2.1\% of the dense baseline.

### 5. [Multiprocessor Scheduling with Memory Constraints: Fundamental Properties and Finding Optimal Solutions](http://arxiv.org/pdf/2507.17411v1)

Authors: Pál András Papp, Toni Böhnlein, A. N. Yzelman

We study the problem of scheduling a general computational DAG on multiple
processors in a 2-level memory hierarchy. This setting is a natural
generalization of several prominent models in the literature, and it
simultaneously captures workload balancing, communication, and data movement
due to cache size limitations. We first analyze the fundamental properties of
this problem from a theoretical perspective, such as its computational
complexity. We also prove that optimizing parallelization and memory management
separately, as done in many applications, can result in a solution that is a
linear factor away from the optimum.
  On the algorithmic side, we discuss a natural technique to represent and
solve the problem as an Integer Linear Program (ILP). We develop a holistic
scheduling algorithm based on this approach, and we experimentally study its
performance and properties on a small benchmark of computational tasks. Our
results confirm that the ILP-based method can indeed find considerably better
solutions than a baseline which combines classical scheduling algorithms and
memory management policies.

### 6. [Distributed P2P quantile tracking with relative value error](http://arxiv.org/pdf/2507.17458v1)

Authors: Marco Pulimeno, Italo Epicoco, Massimo Cafaro

In this paper we present \textsc{DUDDSketch}, a distributed version of the
\textsc{UDDSketch} algorithm for accurate tracking of quantiles. The algorithm
is a fully decentralized, gossip-based distributed protocol working in the
context of unstructured P2P networks. We discuss the algorithm's design and
formally prove its correctness. We also show, through extensive experimental
results, that the algorithm converges to the results provided by the sequential
algorithm, which is a fundamental and highly desirable property.

### 7. [Mapple: A Domain-Specific Language for Mapping Distributed Heterogeneous Parallel Programs](http://arxiv.org/pdf/2507.17087v1)

Authors: Anjiang Wei, Rohan Yadav, Hang Song, Wonchan Lee, Ke Wang, Alex Aiken

Optimizing parallel programs for distributed heterogeneous systems remains a
complex task, often requiring significant code modifications. Task-based
programming systems improve modularity by separating performance decisions from
core application logic, but their mapping interfaces are often too low-level.
In this work, we introduce Mapple, a high-level, declarative programming
interface for mapping distributed applications. Mapple provides transformation
primitives to resolve dimensionality mismatches between iteration and processor
spaces, including a key primitive, decompose, that helps minimize communication
volume. We implement Mapple on top of the Legion runtime by translating Mapple
mappers into its low-level C++ interface. Across nine applications, including
six matrix multiplication algorithms and three scientific computing workloads,
Mapple reduces mapper code size by 14X and enables performance improvements of
up to 1.34X over expert-written C++ mappers. In addition, the decompose
primitive achieves up to 1.83X improvement over existing
dimensionality-resolution heuristics. These results demonstrate that Mapple
simplifies the development of high-performance mappers for distributed
applications.

### 8. [BucketServe: Bucket-Based Dynamic Batching for Smart and Efficient LLM Inference Serving](http://arxiv.org/pdf/2507.17120v1)

Authors: Wanyi Zheng, Minxian Xu, Shengye Song, Kejiang Ye

Large language models (LLMs) have become increasingly popular in various
areas, traditional business gradually shifting from rule-based systems to
LLM-based solutions. However, the inference of LLMs is resource-intensive or
latency-sensitive, posing significant challenges for serving systems. Existing
LLM serving systems often use static or continuous batching strategies, which
can lead to inefficient GPU memory utilization and increased latency,
especially under heterogeneous workloads. These methods may also struggle to
adapt to dynamic workload fluctuations, resulting in suboptimal throughput and
potential service level objective (SLO) violations. In this paper, we introduce
BucketServe, a bucket-based dynamic batching framework designed to optimize LLM
inference performance. By grouping requests into size-homogeneous buckets based
on sequence length, BucketServe minimizes padding overhead and optimizes GPU
memory usage through real-time batch size adjustments preventing out-of-memory
(OOM) errors. It introduces adaptive bucket splitting/merging and
priority-aware scheduling to mitigate resource fragmentation and ensure SLO
compliance. Experiment shows that BucketServe significantly outperforms UELLM
in throughput, achieving up to 3.58x improvement. It can also handle 1.93x more
request load under the SLO attainment of 80% compared with DistServe and
demonstrates 1.975x higher system load capacity compared to the UELLM.

### 9. [Comparing performance of variational quantum algorithm simulations on HPC systems](http://arxiv.org/pdf/2507.17614v1)

Authors: Marco De Pascale, Tobias Valentin Bauer, Yaknan John Gambo, Mario Hernández Vera, Stefan Huber, Burak Mete, Amit Jamadagni, Amine Bentellis, Marita Oliv, Luigi Iapichino, Jeanette Miriam Lorenz

Variational quantum algorithms are of special importance in the research on
quantum computing applications because of their applicability to current Noisy
Intermediate-Scale Quantum (NISQ) devices. The main building blocks of these
algorithms (among them, the definition of the Hamiltonian and of the ansatz,
the optimizer) define a relatively large parameter space, making the comparison
of results and performance between different approaches and software simulators
cumbersome and prone to errors. In this paper, we employ a generic description
of the problem, in terms of both Hamiltonian and ansatz, to port a problem
definition consistently among different simulators. Three use cases of
relevance for current quantum hardware (ground state calculation for the
Hydrogen molecule, MaxCut, Travelling Salesman Problem) have been run on a set
of HPC systems and software simulators to study the dependence of performance
on the runtime environment, the scalability of the simulation codes and the
mutual agreement of the physical results, respectively. The results show that
our toolchain can successfully translate a problem definition between different
simulators. On the other hand, variational algorithms are limited in their
scaling by the long runtimes with respect to their memory footprint, so they
expose limited parallelism to computation. This shortcoming is partially
mitigated by using techniques like job arrays. The potential of the parser tool
for exploring HPC performance and comparisons of results of variational
algorithm simulations is highlighted.

### 10. [P3SL: Personalized Privacy-Preserving Split Learning on Heterogeneous Edge Devices](http://arxiv.org/pdf/2507.17228v1)

Authors: Wei Fan, JinYi Yoon, Xiaochang Li, Huajie Shao, Bo Ji

Split Learning (SL) is an emerging privacy-preserving machine learning
technique that enables resource constrained edge devices to participate in
model training by partitioning a model into client-side and server-side
sub-models. While SL reduces computational overhead on edge devices, it
encounters significant challenges in heterogeneous environments where devices
vary in computing resources, communication capabilities, environmental
conditions, and privacy requirements. Although recent studies have explored
heterogeneous SL frameworks that optimize split points for devices with varying
resource constraints, they often neglect personalized privacy requirements and
local model customization under varying environmental conditions. To address
these limitations, we propose P3SL, a Personalized Privacy-Preserving Split
Learning framework designed for heterogeneous, resource-constrained edge device
systems. The key contributions of this work are twofold. First, we design a
personalized sequential split learning pipeline that allows each client to
achieve customized privacy protection and maintain personalized local models
tailored to their computational resources, environmental conditions, and
privacy needs. Second, we adopt a bi-level optimization technique that empowers
clients to determine their own optimal personalized split points without
sharing private sensitive information (i.e., computational resources,
environmental conditions, privacy requirements) with the server. This approach
balances energy consumption and privacy leakage risks while maintaining high
model accuracy. We implement and evaluate P3SL on a testbed consisting of 7
devices including 4 Jetson Nano P3450 devices, 2 Raspberry Pis, and 1 laptop,
using diverse model architectures and datasets under varying environmental
conditions.

### Digital Libraries

### 1. [Social media uptake of scientific journals: A comparison between X and WeChat](http://arxiv.org/pdf/2507.17114v1)

Authors: Ting Cong, Er-Te Zheng, Zekun Han, Zhichao Fang, Rodrigo Costas

This study examines the social media uptake of scientific journals on two
different platforms - X and WeChat - by comparing the adoption of X among
journals indexed in the Science Citation Index-Expanded (SCIE) with the
adoption of WeChat among journals indexed in the Chinese Science Citation
Database (CSCD). The findings reveal substantial differences in platform
adoption and user engagement, shaped by local contexts. While only 22.7% of
SCIE journals maintain an X account, 84.4% of CSCD journals have a WeChat
official account. Journals in Life Sciences & Biomedicine lead in uptake on
both platforms, whereas those in Technology and Physical Sciences show high
WeChat uptake but comparatively lower presence on X. User engagement on both
platforms is dominated by low-effort interactions rather than more
conversational behaviors. Correlation analyses indicate weak-to-moderate
relationships between bibliometric indicators and social media metrics,
confirming that online engagement reflects a distinct dimension of journal
impact, whether on an international or a local platform. These findings
underscore the need for broader social media metric frameworks that incorporate
locally dominant platforms, thereby offering a more comprehensive understanding
of science communication practices across diverse social media and contexts.

### 2. [Do male leading authors retract more articles than female leading authors?](http://arxiv.org/pdf/2507.17127v1)

Authors: Er-Te Zheng, Hui-Zhen Fu, Mike Thelwall, Zhichao Fang

Scientific retractions reflect issues within the scientific record, arising
from human error or misconduct. Although gender differences in retraction rates
have been previously observed in various contexts, no comprehensive study has
explored this issue across all fields of science. This study examines gender
disparities in scientific misconduct or errors, specifically focusing on
differences in retraction rates between male and female first authors in
relation to their research productivity. Using a dataset comprising 11,622
retracted articles and 19,475,437 non-retracted articles from the Web of
Science and Retraction Watch, we investigate gender differences in retraction
rates from the perspectives of retraction reasons, subject fields, and
countries. Our findings indicate that male first authors have higher retraction
rates, particularly for scientific misconduct such as plagiarism, authorship
disputes, ethical issues, duplication, and fabrication/falsification. No
significant gender differences were found in retractions attributed to
mistakes. Furthermore, male first authors experience significantly higher
retraction rates in biomedical and health sciences, as well as in life and
earth sciences, whereas female first authors have higher retraction rates in
mathematics and computer science. Similar patterns are observed for
corresponding authors. Understanding these gendered patterns of retraction may
contribute to strategies aimed at reducing their prevalence.

### Discrete Mathematics

### 1. [Approximating temporal modularity on graphs of small underlying treewidth](http://arxiv.org/pdf/2507.17541v1)

Authors: Vilhelm Agdur, Jessica Enright, Laura Larios-Jones, Kitty Meeks, Fiona Skerman, Ella Yates

Modularity is a very widely used measure of the level of clustering or
community structure in networks. Here we consider a recent generalisation of
the definition of modularity to temporal graphs, whose edge-sets change over
discrete timesteps; such graphs offer a more realistic model of many real-world
networks in which connections between entities (for example, between
individuals in a social network) evolve over time. Computing modularity is
notoriously difficult: it is NP-hard even to approximate in general, and only
admits efficient exact algorithms in very restricted special cases. Our main
result is that a multiplicative approximation to temporal modularity can be
computed efficiently when the underlying graph has small treewidth. This
generalises a similar approximation algorithm for the static case, but requires
some substantially new ideas to overcome technical challenges associated with
the temporal nature of the problem.

### 2. [On Function-Correcting Codes in the Lee Metric](http://arxiv.org/pdf/2507.17654v1)

Authors: Gyanendra K. Verma, Abhay Kumar Singh

Function-correcting codes are a coding framework designed to minimize
redundancy while ensuring that specific functions or computations of encoded
data can be reliably recovered, even in the presence of errors. The choice of
metric is crucial in designing such codes, as it determines which computations
must be protected and how errors are measured and corrected. Previous work by
Liu and Liu [6] studied function-correcting codes over $\mathbb{Z}_{2^l},\
l\geq 2$ using the homogeneous metric, which coincides with the Lee metric over
$\mathbb{Z}_4$. In this paper, we extend the study to codes over
$\mathbb{Z}_m,$ for any positive integer $m\geq 2$ under the Lee metric and aim
to determine their optimal redundancy. To achieve this, we introduce irregular
Lee distance codes and derive upper and lower bounds on the optimal redundancy
by characterizing the shortest possible length of such codes. These general
bounds are then simplified and applied to specific classes of functions,
including Lee-local functions, Lee weight functions, and Lee weight
distribution functions, leading to improved some bounds compared to those of
Liu and Liu [6] over $\mathbb{Z}_4$ and generalize the other bounds over
$\mathbb{Z}_m$ in the Lee metric.

### Data Structures and Algorithms

### 1. [RLZ-r and LZ-End-r: Enhancing Move-r](http://arxiv.org/pdf/2507.17300v1)

Authors: Patrick Dinklage, Johannes Fischer, Lukas Nalbach, Jan Zumbrink

In pattern matching on strings, a locate query asks for an enumeration of all
the occurrences of a given pattern in a given text. The r-index [Gagie et al.,
2018] is a recently presented compressed self index that stores the text and
auxiliary information in compressed space. With some modifications, locate
queries can be answered in optimal time [Nishimoto & Tabei, 2021], which has
recently been proven relevant in practice in the form of Move-r [Bertram et
al., 2024]. However, there remains the practical bottleneck of evaluating
function $\Phi$ for every occurrence to report. This motivates enhancing the
index by a compressed representation of the suffix array featuring efficient
random access, trading off space for faster answering of locate queries
[Puglisi & Zhukova, 2021]. In this work, we build upon this idea considering
two suitable compression schemes: Relative Lempel-Ziv [Kuruppu et al., 2010],
improving the work by Puglisi and Zhukova, and LZ-End [Kreft & Navarro, 2010],
introducing a different trade-off where compression is better than for Relative
Lempel-Ziv at the cost of slower access times. We enhance both the r-index and
Move-r by the compressed suffix arrays and evaluate locate query performance in
an experiment. We show that locate queries can be sped up considerably in both
the r-index and Move-r, especially if the queried pattern has many occurrences.
The choice between two different compression schemes offers new trade-offs
regarding index size versus query performance.

### 2. [Residual Prophet Inequalities](http://arxiv.org/pdf/2507.17391v1)

Authors: Jose Correa, Sebastian Perez-Salazar, Dana Pizarro, Bruno Ziliotto

We introduce a variant of the classic prophet inequality, called
\emph{residual prophet inequality} (RPI). In the RPI problem, we consider a
finite sequence of $n$ nonnegative independent random values with known
distributions, and a known integer $0\leq k\leq n-1$. Before the gambler
observes the sequence, the top $k$ values are removed, whereas the remaining
$n-k$ values are streamed sequentially to the gambler. For example, one can
assume that the top $k$ values have already been allocated to a higher-priority
agent. Upon observing a value, the gambler must decide irrevocably whether to
accept or reject it, without the possibility of revisiting past values.
  We study two variants of RPI, according to whether the gambler learns online
of the identity of the variable that he sees (FI model) or not (NI model). Our
main result is a randomized algorithm in the FI model with \emph{competitive
ratio} of at least $1/(k+2)$, which we show is tight. Our algorithm is
data-driven and requires access only to the $k+1$ largest values of a single
sample from the $n$ input distributions. In the NI model, we provide a similar
algorithm that guarantees a competitive ratio of $1/(2k+2)$. We further analyze
independent and identically distributed instances when $k=1$. We build a
single-threshold algorithm with a competitive ratio of at least 0.4901, and
show that no single-threshold strategy can get a competitive ratio greater than
0.5464.

### 3. [Advancing Quantum State Preparation using LimTDD](http://arxiv.org/pdf/2507.17170v1)

Authors: Xin Hong, Aochu Dai, Chenjian Li, Sanjiang Li, Shenggang Ying, Mingsheng Ying

Quantum state preparation (QSP) is a fundamental task in quantum computing
and quantum information processing. It is critical to the execution of many
quantum algorithms, including those in quantum machine learning. In this paper,
we propose a family of efficient QSP algorithms tailored to different numbers
of available ancilla qubits - ranging from no ancilla qubits, to a single
ancilla qubit, to a sufficiently large number of ancilla qubits. Our algorithms
are based on a novel decision diagram that is fundamentally different from the
approaches used in previous QSP algorithms. Specifically, our approach exploits
the power of Local Invertible Map Tensor Decision Diagrams (LimTDDs) - a highly
compact representation of quantum states that combines tensor networks and
decision diagrams to reduce quantum circuit complexity. Extensive experiments
demonstrate that our methods significantly outperform existing approaches and
exhibit better scalability for large-scale quantum states, both in terms of
runtime and gate complexity. Furthermore, our method shows exponential
improvement in best-case scenarios. This paper is an extended version of [1],
with three more algorithms proposed.

### 4. [Triadic First-Order Logic Queries in Temporal Networks](http://arxiv.org/pdf/2507.17215v1)

Authors: Omkar Bhalerao, Yunjie Pan, C. Seshadhri, Nishil Talati

Motif counting is a fundamental problem in network analysis, and there is a
rich literature of theoretical and applied algorithms for this problem. Given a
large input network $G$, a motif $H$ is a small "pattern" graph indicative of
special local structure. Motif/pattern mining involves finding all matches of
this pattern in the input $G$. The simplest, yet challenging, case of motif
counting is when $H$ has three vertices, often called a "triadic" query. Recent
work has focused on "temporal graph mining", where the network $G$ has edges
with timestamps (and directions) and $H$ has time constraints.
  Inspired by concepts in logic and database theory, we introduce the study of
"thresholded First Order Logic (FOL) Motif Analysis" for massive temporal
networks. A typical triadic motif query asks for the existence of three
vertices that form a desired temporal pattern. An "FOL" motif query is obtained
by having both existential and thresholded universal quantifiers. This allows
for query semantics that can mine richer information from networks. A typical
triadic query would be "find all triples of vertices $u,v,w$ such that they
form a triangle within one hour". A thresholded FOL query can express "find all
pairs $u,v$ such that for half of $w$ where $(u,w)$ formed an edge, $(v,w)$
also formed an edge within an hour".
  We design the first algorithm, FOLTY, for mining thresholded FOL triadic
queries. The theoretical running time of FOLTY matches the best known running
time for temporal triangle counting in sparse graphs. We give an efficient
implementation of FOLTY using specialized temporal data structures. FOLTY has
excellent empirical behavior, and can answer triadic FOL queries on graphs with
nearly 70M edges is less than hour on commodity hardware. Our work has the
potential to start a new research direction in the classic well-studied problem
of motif analysis.

### 5. [Stable Iterative Solvers for Ill-conditioned Linear Systems](http://arxiv.org/pdf/2507.17673v1)

Authors: Vasileios Kalantzis, Mark S. Squillante, Chai Wah Wu

Iterative solvers for large-scale linear systems such as Krylov subspace
methods can diverge when the linear system is ill-conditioned, thus
significantly reducing the applicability of these iterative methods in practice
for high-performance computing solutions of such large-scale linear systems. To
address this fundamental problem, we propose general algorithmic frameworks to
modify Krylov subspace iterative solution methods which ensure that the
algorithms are stable and do not diverge. We then apply our general frameworks
to current implementations of the corresponding iterative methods in SciPy and
demonstrate the efficacy of our stable iterative approach with respect to
numerical experiments across a wide range of synthetic and real-world
ill-conditioned linear systems.

### Emerging Technologies

### 1. [Evaluation of the effects of frame time variation on VR task performance](http://arxiv.org/pdf/2507.17139v1)

Authors: Benjamin Watson, Victoria Spaulding, Neff Walker, William Ribarsky

We present a first study of the effects of frame time variations, in both
deviation around mean frame times and period of fluctuation, on task
performance in a virtual environment (VE). Chosen are open and closed loop
tasks that are typical for current applications or likely to be prominent in
future ones. The results show that at frame times in the range deemed
acceptable for many applications, fairly large deviations in amplitude over a
fairly wide range of periods do not significantly affect task performance.
However, at a frame time often considered a minimum for immersive VR, frame
time variations do produce significant effects on closed loop task performance.
The results will be of use to designers of VEs and immersive applications, who
often must control frame time variations due to large fluctuations of
complexity (graphical and otherwise) in the VE.

### 2. [Enhancing Quantum Federated Learning with Fisher Information-Based Optimization](http://arxiv.org/pdf/2507.17580v1)

Authors: Amandeep Singh Bhatia, Sabre Kais

Federated Learning (FL) has become increasingly popular across different
sectors, offering a way for clients to work together to train a global model
without sharing sensitive data. It involves multiple rounds of communication
between the global model and participating clients, which introduces several
challenges like high communication costs, heterogeneous client data, prolonged
processing times, and increased vulnerability to privacy threats. In recent
years, the convergence of federated learning and parameterized quantum circuits
has sparked significant research interest, with promising implications for
fields such as healthcare and finance. By enabling decentralized training of
quantum models, it allows clients or institutions to collaboratively enhance
model performance and outcomes while preserving data privacy. Recognizing that
Fisher information can quantify the amount of information that a quantum state
carries under parameter changes, thereby providing insight into its geometric
and statistical properties. We intend to leverage this property to address the
aforementioned challenges. In this work, we propose a Quantum Federated
Learning (QFL) algorithm that makes use of the Fisher information computed on
local client models, with data distributed across heterogeneous partitions.
This approach identifies the critical parameters that significantly influence
the quantum model's performance, ensuring they are preserved during the
aggregation process. Our research assessed the effectiveness and feasibility of
QFL by comparing its performance against other variants, and exploring the
benefits of incorporating Fisher information in QFL settings. Experimental
results on ADNI and MNIST datasets demonstrate the effectiveness of our
approach in achieving better performance and robustness against the quantum
federated averaging method.

### Formal Languages and Automata Theory

### 1. [Realisability and Complementability of Multiparty Session Types](http://arxiv.org/pdf/2507.17354v1)

Authors: Cinzia Di Giusto, Etienne Lozes, Pascal Urso

Multiparty session types (MPST) are a type-based approach for specifying
message-passing distributed systems. They rely on the notion of global type
specifying the global behaviour and local types, which are the projections of
the global behaviour onto each local participant. An essential property of
global types is realisability, i.e., whether the composition of the local
behaviours conforms to those specified by the global type. We explore how
realisability of MPST relates to their complementability, i.e., whether there
exists a global type that describes the complementary behaviour of the original
global type. First, we show that if a global type is realisable with p2p
communications, then it is realisable with synchronous communications. Second,
we show that if a global type is realisable in the synchronous model, then it
is complementable, in the sense that there exists a global type that describes
the complementary behaviour of the original global type. Third, we give an
algorithm to decide whether a complementable global type, given with an
explicit complement, is realisable in p2p. The algorithm is PSPACE in the size
of the global type and its complement. As a side contribution, we propose a
complementation construction for global types with sender-driven choice with a
linear blowup in the size of the global type.

### 2. [Reasoning about Rare-Event Reachability in Stochastic Vector Addition Systems via Affine Vector Spaces](http://arxiv.org/pdf/2507.17711v1)

Authors: Joshua Jeppson, Landon Taylor, Bingqing Hu, Zhen Zhang

Rare events in Stochastic Vector Addition System (VAS) are of significant
interest because, while extremely unlikely, they may represent undesirable
behavior that can have adverse effects. Their low probabilities and potentially
extremely large state spaces challenge existing probabilistic model checking
and stochastic rare-event simulation techniques. In particular, in Chemical
Reaction Networks (CRNs), a chemical kinetic language often represented as VAS,
rare event effects may be pathological. We present two novel heuristics for
priority-first partial state space expansion and trace generation tuned to the
transient analysis of rare-event probability in VAS: Iterative Subspace
Reduction (ISR) and Single Distance Priority (SDP). Both methods construct a
closed vector space containing all solution states. SDP then simply prioritizes
shorter distances to this ``solution space'', while ISR constructs a set of
nested subspaces, where short and highly-probable satisfying traces are likely
to pass through in sequence. The resulting partial state graph from each method
contains likely traces to rare-event states, allowing efficient probabilistic
model checking to compute a lower-bound probability of a rare event of
interest. These methods are deterministic, fast, and demonstrate marked
performance on challenging CRN models.

### Graphics

### 1. [Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](http://arxiv.org/pdf/2507.17336v1)

Authors: Hyeongmin Lee, Kyungjune Baek

Dynamic 4D Gaussian Splatting (4DGS) effectively extends the high-speed
rendering capabilities of 3D Gaussian Splatting (3DGS) to represent volumetric
videos. However, the large number of Gaussians, substantial temporal
redundancies, and especially the absence of an entropy-aware compression
framework result in large storage requirements. Consequently, this poses
significant challenges for practical deployment, efficient edge-device
processing, and data transmission. In this paper, we introduce a novel
end-to-end RD-optimized compression framework tailored for 4DGS, aiming to
enable flexible, high-fidelity rendering across varied computational platforms.
Leveraging Fully Explicit Dynamic Gaussian Splatting (Ex4DGS), one of the
state-of-the-art 4DGS methods, as our baseline, we start from the existing 3DGS
compression methods for compatibility while effectively addressing additional
challenges introduced by the temporal axis. In particular, instead of storing
motion trajectories independently per point, we employ a wavelet transform to
reflect the real-world smoothness prior, significantly enhancing storage
efficiency. This approach yields significantly improved compression ratios and
provides a user-controlled balance between compression efficiency and rendering
quality. Extensive experiments demonstrate the effectiveness of our method,
achieving up to 91x compression compared to the original Ex4DGS model while
maintaining high visual fidelity. These results highlight the applicability of
our framework for real-time dynamic scene rendering in diverse scenarios, from
resource-constrained edge devices to high-performance environments.

### 2. [Parametric Integration with Neural Integral Operators](http://arxiv.org/pdf/2507.17440v1)

Authors: Christoph Schied, Alexander Keller

Real-time rendering imposes strict limitations on the sampling budget for
light transport simulation, often resulting in noisy images. However, denoisers
have demonstrated that it is possible to produce noise-free images through
filtering. We enhance image quality by removing noise before material shading,
rather than filtering already shaded noisy images. This approach allows for
material-agnostic denoising (MAD) and leverages machine learning by
approximating the light transport integral operator with a neural network,
effectively performing parametric integration with neural operators. Our method
operates in real-time, requires data from only a single frame, seamlessly
integrates with existing denoisers and temporal anti-aliasing techniques, and
is efficient to train. Additionally, it is straightforward to incorporate with
physically based rendering algorithms.

### 3. [Visualization-Driven Illumination for Density Plots](http://arxiv.org/pdf/2507.17265v1)

Authors: Xin Chen, Yunhai Wang, Huaiwei Bao, Kecheng Lu, Jaemin Jo, Chi-Wing Fu, Jean-Daniel Fekete

We present a novel visualization-driven illumination model for density plots,
a new technique to enhance density plots by effectively revealing the detailed
structures in high- and medium-density regions and outliers in low-density
regions, while avoiding artifacts in the density field's colors. When
visualizing large and dense discrete point samples, scatterplots and dot
density maps often suffer from overplotting, and density plots are commonly
employed to provide aggregated views while revealing underlying structures.
Yet, in such density plots, existing illumination models may produce color
distortion and hide details in low-density regions, making it challenging to
look up density values, compare them, and find outliers. The key novelty in
this work includes (i) a visualization-driven illumination model that
inherently supports density-plot-specific analysis tasks and (ii) a new image
composition technique to reduce the interference between the image shading and
the color-encoded density values. To demonstrate the effectiveness of our
technique, we conducted a quantitative study, an empirical evaluation of our
technique in a controlled study, and two case studies, exploring twelve
datasets with up to two million data point samples.

### 4. [GhostUMAP2: Measuring and Analyzing (r,d)-Stability of UMAP](http://arxiv.org/pdf/2507.17174v1)

Authors: Myeongwon Jung, Takanori Fujiwara, Jaemin Jo

Despite the widespread use of Uniform Manifold Approximation and Projection
(UMAP), the impact of its stochastic optimization process on the results
remains underexplored. We observed that it often produces unstable results
where the projections of data points are determined mostly by chance rather
than reflecting neighboring structures. To address this limitation, we
introduce (r,d)-stability to UMAP: a framework that analyzes the stochastic
positioning of data points in the projection space. To assess how stochastic
elements, specifically initial projection positions and negative sampling,
impact UMAP results, we introduce "ghosts", or duplicates of data points
representing potential positional variations due to stochasticity. We define a
data point's projection as (r,d)-stable if its ghosts perturbed within a circle
of radius r in the initial projection remain confined within a circle of radius
d for their final positions. To efficiently compute the ghost projections, we
develop an adaptive dropping scheme that reduces a runtime up to 60% compared
to an unoptimized baseline while maintaining approximately 90% of unstable
points. We also present a visualization tool that supports the interactive
exploration of the (r,d)-stability of data points. Finally, we demonstrate the
effectiveness of our framework by examining the stability of projections of
real-world datasets and present usage guidelines for the effective use of our
framework.

### 5. [A Scientist Question: Research on the Impact of Super Structured Quadrilateral Meshes on Convergence and Accuracy of Finite Element Analysis](http://arxiv.org/pdf/2507.17184v1)

Authors: Hui Zhao

In the current practices of both industry and academia, the convergence and
accuracy of finite element calculations are closely related to the methods and
quality of mesh generation. For years, the research on high-quality mesh
generation in the domestic academic field has mainly referred to the local
quality of quadrilaterals and hexahedrons approximating that of squares and
cubes. The main contribution of this paper is to propose a brand-new research
direction and content: it is necessary to explore and study the influence of
the overall global arrangement structure and pattern of super structured
quadrilateral meshes on the convergence and calculation accuracy of finite
element calculations. Through the research in this new field, it can help solve
the non-rigorous state of serious reliance on "experience" in the mesh
generation stage during simulation in the current industry and academia, and
make clear judgments on which global arrangements of mesh generation can ensure
the convergence of finite element calculations. In order to generate and design
super-structured quadrilateral meshes with controllable overall arrangement
structures, a large number of modern two-dimensional and three-dimensional
geometric topology theories are required, such as moduli space, Teichm\"uller
space, harmonic foliations, dynamical systems, surface mappings, meromorphic
quadratic differentials, surface mappings, etc.

### 6. [Reality Proxy: Fluid Interactions with Real-World Objects in MR via Abstract Representations](http://arxiv.org/pdf/2507.17248v1)

Authors: Xiaoan Liu, Difan Jia, Xianhao Carton Liu, Mar Gonzalez-Franco, Chen Zhu-Tian

Interacting with real-world objects in Mixed Reality (MR) often proves
difficult when they are crowded, distant, or partially occluded, hindering
straightforward selection and manipulation. We observe that these difficulties
stem from performing interaction directly on physical objects, where input is
tightly coupled to their physical constraints. Our key insight is to decouple
interaction from these constraints by introducing proxies-abstract
representations of real-world objects. We embody this concept in Reality Proxy,
a system that seamlessly shifts interaction targets from physical objects to
their proxies during selection. Beyond facilitating basic selection, Reality
Proxy uses AI to enrich proxies with semantic attributes and hierarchical
spatial relationships of their corresponding physical objects, enabling novel
and previously cumbersome interactions in MR - such as skimming,
attribute-based filtering, navigating nested groups, and complex multi object
selections - all without requiring new gestures or menu systems. We demonstrate
Reality Proxy's versatility across diverse scenarios, including office
information retrieval, large-scale spatial navigation, and multi-drone control.
An expert evaluation suggests the system's utility and usability, suggesting
that proxy-based abstractions offer a powerful and generalizable interaction
paradigm for future MR systems.

### Computer Science and Game Theory

### 1. [Regret Minimization in Population Network Games: Vanishing Heterogeneity and Convergence to Equilibria](http://arxiv.org/pdf/2507.17183v1)

Authors: Die Hu, Shuyue Hu, Chunjiang Mu, Shiqi Fan, Chen Chu, Jinzhuo Liu, Zhen Wang

Understanding and predicting the behavior of large-scale multi-agents in
games remains a fundamental challenge in multi-agent systems. This paper
examines the role of heterogeneity in equilibrium formation by analyzing how
smooth regret-matching drives a large number of heterogeneous agents with
diverse initial policies toward unified behavior. By modeling the system state
as a probability distribution of regrets and analyzing its evolution through
the continuity equation, we uncover a key phenomenon in diverse multi-agent
settings: the variance of the regret distribution diminishes over time, leading
to the disappearance of heterogeneity and the emergence of consensus among
agents. This universal result enables us to prove convergence to quantal
response equilibria in both competitive and cooperative multi-agent settings.
Our work advances the theoretical understanding of multi-agent learning and
offers a novel perspective on equilibrium selection in diverse game-theoretic
scenarios.

### Human-Computer Interaction

### 1. [OceanVive: An Immersive Visualization System for Communicating Complex Oceanic Phenomena](http://arxiv.org/pdf/2507.17218v1)

Authors: Yang Ouyang, Yuchen Wu, Xiyuan Wang, Laixin Xie, Weicong Cheng, Jianping Gan, Quan Li, Xiaojuan Ma

Communicating the complexity of oceanic phenomena-such as hypoxia and
acidification-poses a persistent challenge for marine science. Despite advances
in sensing technologies and computational models, conventional formats like
static visualizations and text-based reports often fall short in conveying the
dynamics of ocean changes. To address this gap, we present OceanVive, an
immersive and interactive visualization system that transforms complex ocean
datasets into navigable spatial narratives. OceanVive incorporates an
exploratory panel on a table-sized tablet for managing immersive content on a
large screen and integrates adaptive visual encodings, contextual storytelling,
and intuitive navigation pathways to support effective communication. We
validate the system through expert interviews, demonstrating its potential to
enhance science communication and promote deeper public understanding.

### 2. [A "watch your replay videos" reflection assignment on comparing programming without versus with generative AI: learning about programming, critical AI use and limitations, and reflection](http://arxiv.org/pdf/2507.17226v1)

Authors: Sarah "Magz" Fernandez, Greg L Nelson

Generative AI is disrupting computing education. Most interventions focus on
teaching GenAI use rather than helping students understand how AI changes their
programming process. We designed and deployed a novel comparative video
reflection assignment adapting the Describe, Examine, then Articulate Learning
(DEAL) framework. In an introductory software engineering course, students
recorded themselves programming during their team project two times: first
without, then with using generative AI. Students then analyzed their own videos
using a scaffolded set of reflection questions, including on their programming
process and human, internet, and AI help-seeking. We conducted a qualitative
thematic analysis of the reflections, finding students developed insights about
planning, debugging, and help-seeking behaviors that transcended AI use.
Students reported learning to slow down and understand before writing or
generating code, recognized patterns in their problem-solving approaches, and
articulated specific process improvements. Students also learned and reflected
on AI limits and downsides, and strategies to use AI more critically, including
better prompting but also to benefit their learning instead of just completing
tasks. Unexpectedly, the comparative reflection also scaffolded reflection on
programming not involving AI use, and even led to students spontaneously
setting future goals to adopt video and other regular reflection. This work
demonstrates structured reflection on programming session videos can develop
metacognitive skills essential for programming with and without generative AI
and also lifelong learning in our evolving field.

### 3. [Designing for Learning with Generative AI is a Wicked Problem: An Illustrative Longitudinal Qualitative Case Series](http://arxiv.org/pdf/2507.17230v1)

Authors: Clara Scalzer, Saurav Pokhrel, Sara Hunt, Greg L Nelson

Students continue their education when they feel their learning is meaningful
and relevant for their future careers. Computing educators now face the
challenge of preparing students for careers increasingly shaped by generative
AI (GenAI) with the goals of supporting their learning, motivation, ethics, and
career development. Our longitudinal qualitative study of students in a
GenAI-integrated creative media course shows how this is a "wicked" problem:
progress on one goal can then impede progress on other goals. Students
developed concerning patterns despite extensive instruction in critical and
ethical GenAI use including prompt engineering, ethics and bias, and industry
panels on GenAI's career impact. We present an analysis of two students'
experiences to showcase this complexity. Increasing GenAI use skills can lower
ethics; for example, Pat started from purposefully avoiding GenAI use, to
dependency. He described himself as a "notorious cheater" who now uses GenAi to
"get all the right answers" while acknowledging he's learning less. Increasing
ethical awareness can lower the learning of GenAI use skills; for example,
Jay's newfound environmental concerns led to self-imposed usage limits that
impeded skill development, and new serious fears that GenAI would eliminate
creative careers they had been passionate about. Increased GenAI proficiency, a
potential career skill, did not improve their career confidence. These findings
suggest that supporting student development in the GenAI era is a "wicked"
problem requiring multi-dimensional evaluation and design, rather than
optimizing learning, GenAI skills, ethics, or career motivation individually.

### 4. [EventLines: Time Compression for Discrete Event Timelines](http://arxiv.org/pdf/2507.17320v1)

Authors: Yuet Ling Wong, Niklas Elmqvist

Discrete event sequences serve as models for numerous real-world datasets,
including publications over time, project milestones, and medication dosing
during patient treatments. These event sequences typically exhibit bursty
behavior, where events cluster together in rapid succession, interspersed with
periods of inactivity. Standard timeline charts with linear time axes fail to
adequately represent such data, resulting in cluttered regions during event
bursts while leaving other areas unutilized. We introduce EventLines, a novel
technique that dynamically adjusts the time scale to match the underlying event
distribution, enabling more efficient use of screen space. To address the
challenges of non-linear time scaling, EventLines employs the time axis's
visual representation itself to communicate the varying scale. We present
findings from a crowdsourced graphical perception study that examines how
different time scale representations influence temporal perception.

### 5. [Layered Interactions: Exploring Non-Intrusive Digital Craftsmanship Design Through Lacquer Art Interfaces](http://arxiv.org/pdf/2507.17430v1)

Authors: Yan Dong, Hanjie Yu, Yanran Chen, Zipeng Zhang, Qiong Wu

Integrating technology with the distinctive characteristics of craftsmanship
has become a key issue in the field of digital craftsmanship. This paper
introduces Layered Interactions, a design approach that seamlessly merges
Human-Computer Interaction (HCI) technologies with traditional lacquerware
craftsmanship. By leveraging the multi-layer structure and material properties
of lacquerware, we embed interactive circuits and integrate programmable
hardware within the layers, creating tangible interfaces that support diverse
interactions. This method enhances the adaptability and practicality of
traditional crafts in modern digital contexts. Through the development of a
lacquerware toolkit, along with user experiments and semi-structured
interviews, we demonstrate that this approach not only makes technology more
accessible to traditional artisans but also enhances the materiality and
emotional qualities of interactive interfaces. Additionally, it fosters mutual
learning and collaboration between artisans and technologists. Our research
introduces a cross-disciplinary perspective to the HCI community, broadening
the material and design possibilities for interactive interfaces.

### 6. [SDC-Net: A Domain Adaptation Framework with Semantic-Dynamic Consistency for Cross-Subject EEG Emotion Recognition](http://arxiv.org/pdf/2507.17524v1)

Authors: Jiahao Tang, Youjun Li, Xiangting Fan, Yangxuan Zheng, Siyuan Lu, Xueping Li, Peng Fang, Chenxi Li, Zi-Gang Huang

Electroencephalography(EEG) based emotion recognition holds great promise for
affective brain-computer interfaces (aBCIs), yet practical deployment remains
challenging due to substantial inter-subject variability and the lack of
labeled data in target domains. To overcome these limitations, we present a
novel unsupervised Semantic-Dynamic Consistency domain adaptation network for
fully label-free cross-subject EEG emotion recognition. First, we introduce a
Same-Subject Same-Trial Mixup strategy that generates augmented samples via
intra-trial interpolation, enhancing data diversity while explicitly preserving
individual identity to mitigate label ambiguity. Second, we construct a dynamic
distribution alignment module in reproducing kernel Hilbert space (RKHS),
jointly aligning marginal and conditional distributions through multi-objective
kernel mean embedding, and leveraging a confidence-aware pseudo-labeling
strategy to ensure stable adaptation. Third, we propose a dual-domain
similarity consistency learning mechanism that enforces cross-domain structural
constraints based on latent pairwise similarities, enabling semantic boundary
learning without relying on temporal synchronization or label priors. To
validate the effectiveness and robustness of the proposed SDC-Net, extensive
experiments are conducted on three widely used EEG benchmark datasets: SEED,
SEED-IV, and Faced. Comparative results against existing unsupervised domain
adaptation methods demonstrate that SDC-Net achieves state-of-the-art
performance in emotion recognition under both cross-subject and cross-session
conditions. This advancement significantly improves the accuracy and
generalization capability of emotion decoding, and lays a solid foundation for
real-world applications of personalized affective brain-computer interfaces
(aBCIs). The source code will be released at
https://github.com/XuanSuTrum/SDC-Net.

### 7. [Anticipate, Simulate, Reason (ASR): A Comprehensive Generative AI Framework for Combating Messaging Scams](http://arxiv.org/pdf/2507.17543v1)

Authors: Xue Wen Tan, Kenneth See, Stanley Kok

The rapid growth of messaging scams creates an escalating challenge for user
security and financial safety. In this paper, we present the Anticipate,
Simulate, Reason (ASR) framework, a generative AI method that enables users to
proactively identify and comprehend scams within instant messaging platforms.
Using large language models, ASR predicts scammer responses, creates realistic
scam conversations, and delivers real-time, interpretable support to end-users.
We develop ScamGPT-J, a domain-specific language model fine-tuned on a new,
high-quality dataset of scam conversations covering multiple scam types.
Thorough experimental evaluation shows that the ASR framework substantially
enhances scam detection, particularly in challenging contexts such as job
scams, and uncovers important demographic patterns in user vulnerability and
perceptions of AI-generated assistance. Our findings reveal a contradiction
where those most at risk are often least receptive to AI support, emphasizing
the importance of user-centered design in AI-driven fraud prevention. This work
advances both the practical and theoretical foundations for interpretable,
human-centered AI systems in combating evolving digital threats.

### 8. [DataWink: Reusing and Adapting SVG-based Visualization Examples with Large Multimodal Models](http://arxiv.org/pdf/2507.17734v1)

Authors: Liwenhan Xie, Yanna Lin, Can Liu, Huamin Qu, Xinhuan Shu

Creating aesthetically pleasing data visualizations remains challenging for
users without design expertise or familiarity with visualization tools. To
address this gap, we present DataWink, a system that enables users to create
custom visualizations by adapting high-quality examples. Our approach combines
large multimodal models (LMMs) to extract data encoding from existing SVG-based
visualization examples, featuring an intermediate representation of
visualizations that bridges primitive SVG and visualization programs. Users may
express adaptation goals to a conversational agent and control the visual
appearance through widgets generated on demand. With an interactive interface,
users can modify both data mappings and visual design elements while
maintaining the original visualization's aesthetic quality. To evaluate
DataWink, we conduct a user study (N=12) with replication and free-form
exploration tasks. As a result, DataWink is recognized for its learnability and
effectiveness in personalized authoring tasks. Our results demonstrate the
potential of example-driven approaches for democratizing visualization
creation.

### 9. [Evaluation of the effects of frame time variation on VR task performance](http://arxiv.org/pdf/2507.17139v1)

Authors: Benjamin Watson, Victoria Spaulding, Neff Walker, William Ribarsky

We present a first study of the effects of frame time variations, in both
deviation around mean frame times and period of fluctuation, on task
performance in a virtual environment (VE). Chosen are open and closed loop
tasks that are typical for current applications or likely to be prominent in
future ones. The results show that at frame times in the range deemed
acceptable for many applications, fairly large deviations in amplitude over a
fairly wide range of periods do not significantly affect task performance.
However, at a frame time often considered a minimum for immersive VR, frame
time variations do produce significant effects on closed loop task performance.
The results will be of use to designers of VEs and immersive applications, who
often must control frame time variations due to large fluctuations of
complexity (graphical and otherwise) in the VE.

### 10. [HypoChainer: A Collaborative System Combining LLMs and Knowledge Graphs for Hypothesis-Driven Scientific Discovery](http://arxiv.org/pdf/2507.17209v1)

Authors: Haoran Jiang, Shaohan Shi, Yunjie Yao, Chang Jiang, Quan Li

Modern scientific discovery faces growing challenges in integrating vast and
heterogeneous knowledge critical to breakthroughs in biomedicine and drug
development. Traditional hypothesis-driven research, though effective, is
constrained by human cognitive limits, the complexity of biological systems,
and the high cost of trial-and-error experimentation. Deep learning models,
especially graph neural networks (GNNs), have accelerated prediction
generation, but the sheer volume of outputs makes manual selection for
validation unscalable. Large language models (LLMs) offer promise in filtering
and hypothesis generation, yet suffer from hallucinations and lack grounding in
structured knowledge, limiting their reliability. To address these issues, we
propose HypoChainer, a collaborative visualization framework that integrates
human expertise, LLM-driven reasoning, and knowledge graphs (KGs) to enhance
hypothesis generation and validation. HypoChainer operates in three stages:
First, exploration and contextualization -- experts use retrieval-augmented
LLMs (RAGs) and dimensionality reduction to navigate large-scale GNN
predictions, assisted by interactive explanations. Second, hypothesis chain
formation -- experts iteratively examine KG relationships around predictions
and semantically linked entities, refining hypotheses with LLM and KG
suggestions. Third, validation prioritization -- refined hypotheses are
filtered based on KG-supported evidence to identify high-priority candidates
for experimentation, with visual analytics further strengthening weak links in
reasoning. We demonstrate HypoChainer's effectiveness through case studies in
two domains and expert interviews, highlighting its potential to support
interpretable, scalable, and knowledge-grounded scientific discovery.

### Information Retrieval

### 1. [Enhancing Transferability and Consistency in Cross-Domain Recommendations via Supervised Disentanglement](http://arxiv.org/pdf/2507.17112v1)

Authors: Yuhan Wang, Qing Xie, Zhifeng Bao, Mengzi Tang, Lin Li, Yongjian Liu

Cross-domain recommendation (CDR) aims to alleviate the data sparsity by
transferring knowledge across domains. Disentangled representation learning
provides an effective solution to model complex user preferences by separating
intra-domain features (domain-shared and domain-specific features), thereby
enhancing robustness and interpretability. However, disentanglement-based CDR
methods employing generative modeling or GNNs with contrastive objectives face
two key challenges: (i) pre-separation strategies decouple features before
extracting collaborative signals, disrupting intra-domain interactions and
introducing noise; (ii) unsupervised disentanglement objectives lack explicit
task-specific guidance, resulting in limited consistency and suboptimal
alignment. To address these challenges, we propose DGCDR, a GNN-enhanced
encoder-decoder framework. To handle challenge (i), DGCDR first applies GNN to
extract high-order collaborative signals, providing enriched representations as
a robust foundation for disentanglement. The encoder then dynamically
disentangles features into domain-shared and -specific spaces, preserving
collaborative information during the separation process. To handle challenge
(ii), the decoder introduces an anchor-based supervision that leverages
hierarchical feature relationships to enhance intra-domain consistency and
cross-domain alignment. Extensive experiments on real-world datasets
demonstrate that DGCDR achieves state-of-the-art performance, with improvements
of up to 11.59% across key metrics. Qualitative analyses further validate its
superior disentanglement quality and transferability. Our source code and
datasets are available on GitHub for further comparison.

### 2. [R4ec: A Reasoning, Reflection, and Refinement Framework for Recommendation Systems](http://arxiv.org/pdf/2507.17249v1)

Authors: Hao Gu, Rui Zhong, Yu Xia, Wei Yang, Chi Lu, Peng Jiang, Kun Gai

Harnessing Large Language Models (LLMs) for recommendation systems has
emerged as a prominent avenue, drawing substantial research interest. However,
existing approaches primarily involve basic prompt techniques for knowledge
acquisition, which resemble System-1 thinking. This makes these methods highly
sensitive to errors in the reasoning path, where even a small mistake can lead
to an incorrect inference. To this end, in this paper, we propose $R^{4}$ec, a
reasoning, reflection and refinement framework that evolves the recommendation
system into a weak System-2 model. Specifically, we introduce two models: an
actor model that engages in reasoning, and a reflection model that judges these
responses and provides valuable feedback. Then the actor model will refine its
response based on the feedback, ultimately leading to improved responses. We
employ an iterative reflection and refinement process, enabling LLMs to
facilitate slow and deliberate System-2-like thinking. Ultimately, the final
refined knowledge will be incorporated into a recommendation backbone for
prediction. We conduct extensive experiments on Amazon-Book and MovieLens-1M
datasets to demonstrate the superiority of $R^{4}$ec. We also deploy $R^{4}$ec
on a large scale online advertising platform, showing 2.2\% increase of
revenue. Furthermore, we investigate the scaling properties of the actor model
and reflection model.

### 3. [Exploring the Potential of LLMs for Serendipity Evaluation in Recommender Systems](http://arxiv.org/pdf/2507.17290v1)

Authors: Li Kang, Yuhan Zhao, Li Chen

Serendipity plays a pivotal role in enhancing user satisfaction within
recommender systems, yet its evaluation poses significant challenges due to its
inherently subjective nature and conceptual ambiguity. Current algorithmic
approaches predominantly rely on proxy metrics for indirect assessment, often
failing to align with real user perceptions, thus creating a gap. With large
language models (LLMs) increasingly revolutionizing evaluation methodologies
across various human annotation tasks, we are inspired to explore a core
research proposition: Can LLMs effectively simulate human users for serendipity
evaluation? To address this question, we conduct a meta-evaluation on two
datasets derived from real user studies in the e-commerce and movie domains,
focusing on three key aspects: the accuracy of LLMs compared to conventional
proxy metrics, the influence of auxiliary data on LLM comprehension, and the
efficacy of recently popular multi-LLM techniques. Our findings indicate that
even the simplest zero-shot LLMs achieve parity with, or surpass, the
performance of conventional metrics. Furthermore, multi-LLM techniques and the
incorporation of auxiliary data further enhance alignment with human
perspectives. Based on our findings, the optimal evaluation by LLMs yields a
Pearson correlation coefficient of 21.5\% when compared to the results of the
user study. This research implies that LLMs may serve as potentially accurate
and cost-effective evaluators, introducing a new paradigm for serendipity
evaluation in recommender systems.

### 4. [EndoFinder: Online Lesion Retrieval for Explainable Colorectal Polyp Diagnosis Leveraging Latent Scene Representations](http://arxiv.org/pdf/2507.17323v1)

Authors: Ruijie Yang, Yan Zhu, Peiyao Fu, Yizhe Zhang, Zhihua Wang, Quanlin Li, Pinghong Zhou, Xian Yang, Shuo Wang

Colorectal cancer (CRC) remains a leading cause of cancer-related mortality,
underscoring the importance of timely polyp detection and diagnosis. While deep
learning models have improved optical-assisted diagnostics, they often demand
extensive labeled datasets and yield "black-box" outputs with limited
interpretability. In this paper, we propose EndoFinder, an online polyp
retrieval framework that leverages multi-view scene representations for
explainable and scalable CRC diagnosis. First, we develop a Polyp-aware Image
Encoder by combining contrastive learning and a reconstruction task, guided by
polyp segmentation masks. This self-supervised approach captures robust
features without relying on large-scale annotated data. Next, we treat each
polyp as a three-dimensional "scene" and introduce a Scene Representation
Transformer, which fuses multiple views of the polyp into a single latent
representation. By discretizing this representation through a hashing layer,
EndoFinder enables real-time retrieval from a compiled database of historical
polyp cases, where diagnostic information serves as interpretable references
for new queries. We evaluate EndoFinder on both public and newly collected
polyp datasets for re-identification and pathology classification. Results show
that EndoFinder outperforms existing methods in accuracy while providing
transparent, retrieval-based insights for clinical decision-making. By
contributing a novel dataset and a scalable, explainable framework, our work
addresses key challenges in polyp diagnosis and offers a promising direction
for more efficient AI-driven colonoscopy workflows. The source code is
available at https://github.com/ku262/EndoFinder-Scene.

### 5. ["Beyond the past": Leveraging Audio and Human Memory for Sequential Music Recommendation](http://arxiv.org/pdf/2507.17356v1)

Authors: Viet-Tran Anh, Bruno Sguerra, Gabriel Meseguer-Brocal, Lea Briand, Manuel Moussallam

On music streaming services, listening sessions are often composed of a
balance of familiar and new tracks. Recently, sequential recommender systems
have adopted cognitive-informed approaches, such as Adaptive Control of
Thought-Rational (ACT-R), to successfully improve the prediction of the most
relevant tracks for the next user session. However, one limitation of using a
model inspired by human memory (or the past), is that it struggles to recommend
new tracks that users have not previously listened to. To bridge this gap, here
we propose a model that leverages audio information to predict in advance the
ACT-R-like activation of new tracks and incorporates them into the
recommendation scoring process. We demonstrate the empirical effectiveness of
the proposed model using proprietary data, which we publicly release along with
the model's source code to foster future research in this field.

### 6. [Leave No One Behind: Fairness-Aware Cross-Domain Recommender Systems for Non-Overlapping Users](http://arxiv.org/pdf/2507.17749v1)

Authors: Weixin Chen, Yuhan Zhao, Li Chen, Weike Pan

Cross-domain recommendation (CDR) methods predominantly leverage overlapping
users to transfer knowledge from a source domain to a target domain. However,
through empirical studies, we uncover a critical bias inherent in these
approaches: while overlapping users experience significant enhancements in
recommendation quality, non-overlapping users benefit minimally and even face
performance degradation. This unfairness may erode user trust, and,
consequently, negatively impact business engagement and revenue. To address
this issue, we propose a novel solution that generates virtual source-domain
users for non-overlapping target-domain users. Our method utilizes a dual
attention mechanism to discern similarities between overlapping and
non-overlapping users, thereby synthesizing realistic virtual user embeddings.
We further introduce a limiter component that ensures the generated virtual
users align with real-data distributions while preserving each user's unique
characteristics. Notably, our method is model-agnostic and can be seamlessly
integrated into any CDR model. Comprehensive experiments conducted on three
public datasets with five CDR baselines demonstrate that our method effectively
mitigates the CDR non-overlapping user bias, without loss of overall accuracy.
Our code is publicly available at https://github.com/WeixinChen98/VUG.

### 7. [Citation Recommendation using Deep Canonical Correlation Analysis](http://arxiv.org/pdf/2507.17603v1)

Authors: Conor McNamara, Effirul Ramlan

Recent advances in citation recommendation have improved accuracy by
leveraging multi-view representation learning to integrate the various
modalities present in scholarly documents. However, effectively combining
multiple data views requires fusion techniques that can capture complementary
information while preserving the unique characteristics of each modality. We
propose a novel citation recommendation algorithm that improves upon linear
Canonical Correlation Analysis (CCA) methods by applying Deep CCA (DCCA), a
neural network extension capable of capturing complex, non-linear relationships
between distributed textual and graph-based representations of scientific
articles. Experiments on the large-scale DBLP (Digital Bibliography & Library
Project) citation network dataset demonstrate that our approach outperforms
state-of-the-art CCA-based methods, achieving relative improvements of over 11%
in Mean Average Precision@10, 5% in Precision@10, and 7% in Recall@10. These
gains reflect more relevant citation recommendations and enhanced ranking
quality, suggesting that DCCA's non-linear transformations yield more
expressive latent representations than CCA's linear projections.

### 8. [Triadic First-Order Logic Queries in Temporal Networks](http://arxiv.org/pdf/2507.17215v1)

Authors: Omkar Bhalerao, Yunjie Pan, C. Seshadhri, Nishil Talati

Motif counting is a fundamental problem in network analysis, and there is a
rich literature of theoretical and applied algorithms for this problem. Given a
large input network $G$, a motif $H$ is a small "pattern" graph indicative of
special local structure. Motif/pattern mining involves finding all matches of
this pattern in the input $G$. The simplest, yet challenging, case of motif
counting is when $H$ has three vertices, often called a "triadic" query. Recent
work has focused on "temporal graph mining", where the network $G$ has edges
with timestamps (and directions) and $H$ has time constraints.
  Inspired by concepts in logic and database theory, we introduce the study of
"thresholded First Order Logic (FOL) Motif Analysis" for massive temporal
networks. A typical triadic motif query asks for the existence of three
vertices that form a desired temporal pattern. An "FOL" motif query is obtained
by having both existential and thresholded universal quantifiers. This allows
for query semantics that can mine richer information from networks. A typical
triadic query would be "find all triples of vertices $u,v,w$ such that they
form a triangle within one hour". A thresholded FOL query can express "find all
pairs $u,v$ such that for half of $w$ where $(u,w)$ formed an edge, $(v,w)$
also formed an edge within an hour".
  We design the first algorithm, FOLTY, for mining thresholded FOL triadic
queries. The theoretical running time of FOLTY matches the best known running
time for temporal triangle counting in sparse graphs. We give an efficient
implementation of FOLTY using specialized temporal data structures. FOLTY has
excellent empirical behavior, and can answer triadic FOL queries on graphs with
nearly 70M edges is less than hour on commodity hardware. Our work has the
potential to start a new research direction in the classic well-studied problem
of motif analysis.

### 9. [Millions of $\text{GeAR}$-s: Extending GraphRAG to Millions of Documents](http://arxiv.org/pdf/2507.17399v1)

Authors: Zhili Shen, Chenxin Diao, Pascual Merita, Pavlos Vougiouklis, Jeff Z. Pan

Recent studies have explored graph-based approaches to retrieval-augmented
generation, leveraging structured or semi-structured information -- such as
entities and their relations extracted from documents -- to enhance retrieval.
However, these methods are typically designed to address specific tasks, such
as multi-hop question answering and query-focused summarisation, and therefore,
there is limited evidence of their general applicability across broader
datasets. In this paper, we aim to adapt a state-of-the-art graph-based RAG
solution: $\text{GeAR}$ and explore its performance and limitations on the
SIGIR 2025 LiveRAG Challenge.

### 10. [HLFormer: Enhancing Partially Relevant Video Retrieval with Hyperbolic Learning](http://arxiv.org/pdf/2507.17402v1)

Authors: Li Jun, Wang Jinpeng, Tan Chaolei, Lian Niu, Chen Long, Zhang Min, Wang Yaowei, Xia Shu-Tao, Chen Bin

Partially Relevant Video Retrieval (PRVR) addresses the critical challenge of
matching untrimmed videos with text queries describing only partial content.
Existing methods suffer from geometric distortion in Euclidean space that
sometimes misrepresents the intrinsic hierarchical structure of videos and
overlooks certain hierarchical semantics, ultimately leading to suboptimal
temporal modeling. To address this issue, we propose the first hyperbolic
modeling framework for PRVR, namely HLFormer, which leverages hyperbolic space
learning to compensate for the suboptimal hierarchical modeling capabilities of
Euclidean space. Specifically, HLFormer integrates the Lorentz Attention Block
and Euclidean Attention Block to encode video embeddings in hybrid spaces,
using the Mean-Guided Adaptive Interaction Module to dynamically fuse features.
Additionally, we introduce a Partial Order Preservation Loss to enforce "text <
video" hierarchy through Lorentzian cone constraints. This approach further
enhances cross-modal matching by reinforcing partial relevance between video
content and text queries. Extensive experiments show that HLFormer outperforms
state-of-the-art methods. Code is released at
https://github.com/lijun2005/ICCV25-HLFormer.

### Machine Learning

### 1. [Probabilistic Graphical Models: A Concise Tutorial](http://arxiv.org/pdf/2507.17116v1)

Authors: Jacqueline Maasch, Willie Neiswanger, Stefano Ermon, Volodymyr Kuleshov

Probabilistic graphical modeling is a branch of machine learning that uses
probability distributions to describe the world, make predictions, and support
decision-making under uncertainty. Underlying this modeling framework is an
elegant body of theory that bridges two mathematical traditions: probability
and graph theory. This framework provides compact yet expressive
representations of joint probability distributions, yielding powerful
generative models for probabilistic reasoning.
  This tutorial provides a concise introduction to the formalisms, methods, and
applications of this modeling framework. After a review of basic probability
and graph theory, we explore three dominant themes: (1) the representation of
multivariate distributions in the intuitive visual language of graphs, (2)
algorithms for learning model parameters and graphical structures from data,
and (3) algorithms for inference, both exact and approximate.

### 2. [Computer Vision for Real-Time Monkeypox Diagnosis on Embedded Systems](http://arxiv.org/pdf/2507.17123v1)

Authors: Jacob M. Delgado-López, Ricardo A. Morell-Rodriguez, Sebastián O. Espinosa-Del Rosario, Wilfredo E. Lugo-Beauchamp

The rapid diagnosis of infectious diseases, such as monkeypox, is crucial for
effective containment and treatment, particularly in resource-constrained
environments. This study presents an AI-driven diagnostic tool developed for
deployment on the NVIDIA Jetson Orin Nano, leveraging the pre-trained
MobileNetV2 architecture for binary classification. The model was trained on
the open-source Monkeypox Skin Lesion Dataset, achieving a 93.07% F1-Score,
which reflects a well-balanced performance in precision and recall. To optimize
the model, the TensorRT framework was used to accelerate inference for FP32 and
to perform post-training quantization for FP16 and INT8 formats. TensorRT's
mixed-precision capabilities enabled these optimizations, which reduced the
model size, increased inference speed, and lowered power consumption by
approximately a factor of two, all while maintaining the original accuracy.
Power consumption analysis confirmed that the optimized models used
significantly less energy during inference, reinforcing their suitability for
deployment in resource-constrained environments. The system was deployed with a
Wi-Fi Access Point (AP) hotspot and a web-based interface, enabling users to
upload and analyze images directly through connected devices such as mobile
phones. This setup ensures simple access and seamless connectivity, making the
tool practical for real-world applications. These advancements position the
diagnostic tool as an efficient, scalable, and energy-conscious solution to
address diagnosis challenges in underserved regions, paving the way for broader
adoption in low-resource healthcare settings.

### 3. [Model Compression Engine for Wearable Devices Skin Cancer Diagnosis](http://arxiv.org/pdf/2507.17125v1)

Authors: Jacob M. Delgado-López, Andrea P. Seda-Hernandez, Juan D. Guadalupe-Rosado, Luis E. Fernandez Ramirez, Miguel Giboyeaux-Camilo, Wilfredo E. Lugo-Beauchamp

Skin cancer is one of the most prevalent and preventable types of cancer, yet
its early detection remains a challenge, particularly in resource-limited
settings where access to specialized healthcare is scarce. This study proposes
an AI-driven diagnostic tool optimized for embedded systems to address this
gap. Using transfer learning with the MobileNetV2 architecture, the model was
adapted for binary classification of skin lesions into "Skin Cancer" and
"Other." The TensorRT framework was employed to compress and optimize the model
for deployment on the NVIDIA Jetson Orin Nano, balancing performance with
energy efficiency. Comprehensive evaluations were conducted across multiple
benchmarks, including model size, inference speed, throughput, and power
consumption. The optimized models maintained their performance, achieving an
F1-Score of 87.18% with a precision of 93.18% and recall of 81.91%.
Post-compression results showed reductions in model size of up to 0.41, along
with improvements in inference speed and throughput, and a decrease in energy
consumption of up to 0.93 in INT8 precision. These findings validate the
feasibility of deploying high-performing, energy-efficient diagnostic tools on
resource-constrained edge devices. Beyond skin cancer detection, the
methodologies applied in this research have broader applications in other
medical diagnostics and domains requiring accessible, efficient AI solutions.
This study underscores the potential of optimized AI systems to revolutionize
healthcare diagnostics, thereby bridging the divide between advanced technology
and underserved regions.

### 4. [PICore: Physics-Informed Unsupervised Coreset Selection for Data Efficient Neural Operator Training](http://arxiv.org/pdf/2507.17151v1)

Authors: Anirudh Satheesh, Anant Khandelwal, Mucong Ding, Radu Balan

Neural operators offer a powerful paradigm for solving partial differential
equations (PDEs) that cannot be solved analytically by learning mappings
between function spaces. However, there are two main bottlenecks in training
neural operators: they require a significant amount of training data to learn
these mappings, and this data needs to be labeled, which can only be accessed
via expensive simulations with numerical solvers. To alleviate both of these
issues simultaneously, we propose PICore, an unsupervised coreset selection
framework that identifies the most informative training samples without
requiring access to ground-truth PDE solutions. PICore leverages a
physics-informed loss to select unlabeled inputs by their potential
contribution to operator learning. After selecting a compact subset of inputs,
only those samples are simulated using numerical solvers to generate labels,
reducing annotation costs. We then train the neural operator on the reduced
labeled dataset, significantly decreasing training time as well. Across four
diverse PDE benchmarks and multiple coreset selection strategies, PICore
achieves up to 78% average increase in training efficiency relative to
supervised coreset selection methods with minimal changes in accuracy. We
provide code at https://github.com/Asatheesh6561/PICore.

### 5. [Met$^2$Net: A Decoupled Two-Stage Spatio-Temporal Forecasting Model for Complex Meteorological Systems](http://arxiv.org/pdf/2507.17189v1)

Authors: Shaohan Li, Hao Yang, Min Chen, Xiaolin Qin

The increasing frequency of extreme weather events due to global climate
change urges accurate weather prediction. Recently, great advances have been
made by the \textbf{end-to-end methods}, thanks to deep learning techniques,
but they face limitations of \textit{representation inconsistency} in
multivariable integration and struggle to effectively capture the dependency
between variables, which is required in complex weather systems. Treating
different variables as distinct modalities and applying a \textbf{two-stage
training approach} from multimodal models can partially alleviate this issue,
but due to the inconformity in training tasks between the two stages, the
results are often suboptimal. To address these challenges, we propose an
implicit two-stage training method, configuring separate encoders and decoders
for each variable. In detailed, in the first stage, the Translator is frozen
while the Encoders and Decoders learn a shared latent space, in the second
stage, the Encoders and Decoders are frozen, and the Translator captures
inter-variable interactions for prediction. Besides, by introducing a
self-attention mechanism for multivariable fusion in the latent space, the
performance achieves further improvements. Empirically, extensive experiments
show the state-of-the-art performance of our method. Specifically, it reduces
the MSE for near-surface air temperature and relative humidity predictions by
28.82\% and 23.39\%, respectively. The source code is available at
https://github.com/ShremG/Met2Net.

### 6. [Filter-And-Refine: A MLLM Based Cascade System for Industrial-Scale Video Content Moderation](http://arxiv.org/pdf/2507.17204v1)

Authors: Zixuan Wang, Jinghao Shi, Hanzhong Liang, Xiang Shen, Vera Wen, Zhiqian Chen, Yifan Wu, Zhixin Zhang, Hongyu Xiong

Effective content moderation is essential for video platforms to safeguard
user experience and uphold community standards. While traditional video
classification models effectively handle well-defined moderation tasks, they
struggle with complicated scenarios such as implicit harmful content and
contextual ambiguity. Multimodal large language models (MLLMs) offer a
promising solution to these limitations with their superior cross-modal
reasoning and contextual understanding. However, two key challenges hinder
their industrial adoption. First, the high computational cost of MLLMs makes
full-scale deployment impractical. Second, adapting generative models for
discriminative classification remains an open research problem. In this paper,
we first introduce an efficient method to transform a generative MLLM into a
multimodal classifier using minimal discriminative training data. To enable
industry-scale deployment, we then propose a router-ranking cascade system that
integrates MLLMs with a lightweight router model. Offline experiments
demonstrate that our MLLM-based approach improves F1 score by 66.50% over
traditional classifiers while requiring only 2% of the fine-tuning data. Online
evaluations show that our system increases automatic content moderation volume
by 41%, while the cascading deployment reduces computational cost to only 1.5%
of direct full-scale deployment.

### 7. [Rethinking VAE: From Continuous to Discrete Representations Without Probabilistic Assumptions](http://arxiv.org/pdf/2507.17255v1)

Authors: Songxuan Shi

This paper explores the generative capabilities of Autoencoders (AEs) and
establishes connections between Variational Autoencoders (VAEs) and Vector
Quantized-Variational Autoencoders (VQ-VAEs) through a reformulated training
framework. We demonstrate that AEs exhibit generative potential via latent
space interpolation and perturbation, albeit limited by undefined regions in
the encoding space. To address this, we propose a new VAE-like training method
that introduces clustering centers to enhance data compactness and ensure
well-defined latent spaces without relying on traditional KL divergence or
reparameterization techniques. Experimental results on MNIST, CelebA, and
FashionMNIST datasets show smooth interpolative transitions, though blurriness
persists. Extending this approach to multiple learnable vectors, we observe a
natural progression toward a VQ-VAE-like model in continuous space. However,
when the encoder outputs multiple vectors, the model degenerates into a
discrete Autoencoder (VQ-AE), which combines image fragments without learning
semantic representations. Our findings highlight the critical role of encoding
space compactness and dispersion in generative modeling and provide insights
into the intrinsic connections between VAEs and VQ-VAEs, offering a new
perspective on their design and limitations.

### 8. [Decentralized Federated Learning of Probabilistic Generative Classifiers](http://arxiv.org/pdf/2507.17285v1)

Authors: Aritz Pérez, Carlos Echegoyen, Guzmán Santafé

Federated learning is a paradigm of increasing relevance in real world
applications, aimed at building a global model across a network of
heterogeneous users without requiring the sharing of private data. We focus on
model learning over decentralized architectures, where users collaborate
directly to update the global model without relying on a central server. In
this context, the current paper proposes a novel approach to collaboratively
learn probabilistic generative classifiers with a parametric form. The
framework is composed by a communication network over a set of local nodes,
each of one having its own local data, and a local updating rule. The proposal
involves sharing local statistics with neighboring nodes, where each node
aggregates the neighbors' information and iteratively learns its own local
classifier, which progressively converges to a global model. Extensive
experiments demonstrate that the algorithm consistently converges to a globally
competitive model across a wide range of network topologies, network sizes,
local dataset sizes, and extreme non-i.i.d. data distributions.

### 9. [R-Stitch: Dynamic Trajectory Stitching for Efficient Reasoning](http://arxiv.org/pdf/2507.17307v1)

Authors: Zhuokun Chen, Zeren Chen, Jiahao He, Mingkui Tan, Jianfei Cai, Bohan Zhuang

Chain-of-thought (CoT) reasoning enhances the problem-solving capabilities of
large language models by encouraging step-by-step intermediate reasoning during
inference. While effective, CoT introduces substantial computational overhead
due to its reliance on autoregressive decoding over long token sequences.
Existing acceleration strategies either reduce sequence length through early
stopping or compressive reward designs, or improve decoding speed via
speculative decoding with smaller models. However, speculative decoding suffers
from limited speedup when the agreement between small and large models is low,
and fails to exploit the potential advantages of small models in producing
concise intermediate reasoning. In this paper, we present R-Stitch, a
token-level, confidence-based hybrid decoding framework that accelerates CoT
inference by switching between a small language model (SLM) and a large
language model (LLM) along the reasoning trajectory. R-Stitch uses the SLM to
generate tokens by default and delegates to the LLM only when the SLM's
confidence falls below a threshold. This design avoids full-sequence rollback
and selectively invokes the LLM on uncertain steps, preserving both efficiency
and answer quality. R-Stitch is model-agnostic, training-free, and compatible
with standard decoding pipelines. Experiments on math reasoning benchmarks
demonstrate that R-Stitch achieves up to 85\% reduction in inference latency
with negligible accuracy drop, highlighting its practical effectiveness in
accelerating CoT reasoning.

### 10. [DeCo-SGD: Joint Optimization of Delay Staleness and Gradient Compression Ratio for Distributed SGD](http://arxiv.org/pdf/2507.17346v1)

Authors: Rongwei Lu, Jingyan Jiang, Chunyang Li, Haotian Dong, Xingguang Wei, Delin Cai, Zhi Wang

Distributed machine learning in high end-to-end latency and low, varying
bandwidth network environments undergoes severe throughput degradation. Due to
its low communication requirements, distributed SGD (D-SGD) remains the
mainstream optimizer in such challenging networks, but it still suffers from
significant throughput reduction. To mitigate these limitations, existing
approaches typically employ gradient compression and delayed aggregation to
alleviate low bandwidth and high latency, respectively. To address both
challenges simultaneously, these strategies are often combined, introducing a
complex three-way trade-off among compression ratio, staleness (delayed
synchronization steps), and model convergence rate. To achieve the balance
under varying bandwidth conditions, an adaptive policy is required to
dynamically adjust these parameters. Unfortunately, existing works rely on
static heuristic strategies due to the lack of theoretical guidance, which
prevents them from achieving this goal. This study fills in this theoretical
gap by introducing a new theoretical tool, decomposing the joint optimization
problem into a traditional convergence rate analysis with multiple analyzable
noise terms. We are the first to reveal that staleness exponentially amplifies
the negative impact of gradient compression on training performance, filling a
critical gap in understanding how compressed and delayed gradients affect
training. Furthermore, by integrating the convergence rate with a network-aware
time minimization condition, we propose DeCo-SGD, which dynamically adjusts the
compression ratio and staleness based on the real-time network condition and
training task. DeCo-SGD achieves up to 5.07 and 1.37 speed-ups over D-SGD and
static strategy in high-latency and low, varying bandwidth networks,
respectively.

### Networking and Internet Architecture

### 1. [Closed-Form and Boundary Expressions for Task-Success Probability in Status-Driven Systems](http://arxiv.org/pdf/2507.17195v1)

Authors: Jianpeng Qi, Chao Liu, Rui Wang, Junyu Dong, Yanwei Yu

Timely and efficient dissemination of server status is critical in
compute-first networking systems, where user tasks arrive dynamically and
computing resources are limited and stochastic. In such systems, the access
point plays a key role in forwarding tasks to a server based on its latest
received server status. However, modeling the task-success probability
suffering the factors of stochastic arrivals, limited server capacity, and
bidirectional link delays. Therefore, we introduce a unified analytical
framework that abstracts the AP forwarding rule as a single probability and
models all network and waiting delays via their Laplace transforms. This
approach yields a closed form expression for the end to end task success
probability, together with upper and lower bounds that capture Erlang loss
blocking, information staleness, and random uplink/downlink delays. We validate
our results through simulations across a wide range of parameters, showing that
theoretical predictions and bounds consistently enclose observed success rates.
Our framework requires only two interchangeable inputs (the forwarding
probability and the delay transforms), making it readily adaptable to
alternative forwarding policies and delay distributions. Experiments
demonstrate that our bounds are able to achieve accuracy within 0.01 (upper
bound) and 0.016 (lower bound) of the empirical task success probability.

### 2. [Custody Transfer and Compressed Status Reporting for Bundle Protocol Version 7](http://arxiv.org/pdf/2507.17403v1)

Authors: Alice Le Bihan, Felix Flentge, Juan A. Fraire

As space missions increase, there is a growing need to replace point-to-point
communication with an efficient and reliable network-centric communication
approach. Disruption/Delay Tolerant Networking (DTN) with the Bundle Protocol
(BP) has been selected as an interoperable network protocol in the LunaNet
Interoperability Specification. It is also considered for future Earth
Observation and Mars communication scenarios. In a DTN, the "bundle" -- the
fundamental data unit of BP -- requires dedicated mechanisms to ensure
reliability due to the challenges posed by intermittent connectivity and long
delays. The previous version of BP, BPv6, contained a mechanism for reliable
transfer between "custodial nodes" called "custody transfer". However, this
approach has been removed from the core protocol specification for BPv7, which
requires a corresponding BP reliability extension to be defined separately.
This paper introduces a new custody transfer process for BPv7 (expected to be
published by CCSDS as an experimental specification in 2025). The core features
of this new custody transfer method for BPv7 are: (1) A strategy to efficiently
identify sets of bundles by sequence numbering (2) A new Custody Transfer
Extension Block and a corresponding administrative record, Compressed Custody
Signal, to efficiently report on the acceptance or rejection of custody using
sequence numbering (3) A new Compressed Reporting Extension Block requesting
reporting on bundle processing steps using a corresponding administrative
record with sequence numbering for efficiency. The paper will describe those
concepts and their design, specification, and implementation in detail. These
mechanisms have been prototyped in the ESA BP implementation and tested in
Earth Observation and Lunar communication simulation scenarios. The results
will be presented, as will an outlook on future work in the DTN reliable
transfer domain.

### 3. [Active Attack Resilience in 5G: A New Take on Authentication and Key Agreement](http://arxiv.org/pdf/2507.17491v1)

Authors: Nazatul H. Sultan, Xinlong Guan, Josef Pieprzyk, Wei Ni, Sharif Abuadbba, Hajime Suzuki

As 5G networks expand into critical infrastructure, secure and efficient user
authentication is more important than ever. The 5G-AKA protocol, standardized
by 3GPP in TS 33.501, is central to authentication in current 5G deployments.
It provides mutual authentication, user privacy, and key secrecy. However,
despite its adoption, 5G-AKA has known limitations in both security and
performance. While it focuses on protecting privacy against passive attackers,
recent studies show its vulnerabilities to active attacks. It also relies on a
sequence number mechanism to prevent replay attacks, requiring perfect
synchronization between the device and the core network. This stateful design
adds complexity, causes desynchronization, and incurs extra communication
overhead. More critically, 5G-AKA lacks Perfect Forward Secrecy (PFS), exposing
past communications if long-term keys are compromised-an increasing concern
amid sophisticated threats. This paper proposes an enhanced authentication
protocol that builds on 5G-AKA's design while addressing its shortcomings.
First, we introduce a stateless version that removes sequence number reliance,
reducing complexity while staying compatible with existing SIM cards and
infrastructure. We then extend this design to add PFS with minimal
cryptographic overhead. Both protocols are rigorously analyzed using ProVerif,
confirming their compliance with all major security requirements, including
resistance to passive and active attacks, as well as those defined by 3GPP and
academic studies. We also prototype both protocols and evaluate their
performance against 5G-AKA and 5G-AKA' (USENIX'21). Our results show the
proposed protocols offer stronger security with only minor computational
overhead, making them practical, future-ready solutions for 5G and beyond.

### 4. [A Virtual Quantum Network Prototype for Open Access](http://arxiv.org/pdf/2507.17495v1)

Authors: Raj Kamleshkumar Madhu, Visuttha Manthamkarn, Zheshen Zhang, Jianqing Liu

The rise of quantum networks has revolutionized domains such as
communication, sensing, and cybersecurity. Despite this progress, current
quantum network systems remain limited in scale, are highly
application-specific (e.g., for quantum key distribution), and lack a clear
road map for global expansion. These limitations are largely driven by a
shortage of skilled professionals, limited accessibility to quantum
infrastructure, and the high complexity and cost associated with building and
operating quantum hardware. To address these challenges, this paper proposes an
open-access software-based quantum network virtualization platform designed to
facilitate scalable and remote interaction with quantum hardware. The system is
built around a cloud application that virtualizes the core hardware components
of a lab-scale quantum network testbed, including the time tagger and optical
switch, enabling users to perform coincidence counts of the photon
entanglements while ensuring fair resource allocation. The fairness is ensured
by employing the Hungarian Algorithm to allocate nearly equal effective
entanglement rates among users. We provide implementation details and
performance analysis from the perspectives of hardware, software, and cloud
platform, which demonstrates the functionality and efficiency of the developed
prototype.

### 5. [Symbiotic Agents: A Novel Paradigm for Trustworthy AGI-driven Networks](http://arxiv.org/pdf/2507.17695v1)

Authors: Ilias Chatzistefanidis, Navid Nikaein

Large Language Model (LLM)-based autonomous agents are expected to play a
vital role in the evolution of 6G networks, by empowering real-time
decision-making related to management and service provisioning to end-users.
This shift facilitates the transition from a specialized intelligence approach,
where artificial intelligence (AI) algorithms handle isolated tasks, to
artificial general intelligence (AGI)-driven networks, where agents possess
broader reasoning capabilities and can manage diverse network functions. In
this paper, we introduce a novel agentic paradigm that combines LLMs with
real-time optimization algorithms towards Trustworthy AI, defined as symbiotic
agents. Optimizers at the LLM's input-level provide bounded uncertainty
steering for numerically precise tasks, whereas output-level optimizers
supervised by the LLM enable adaptive real-time control. We design and
implement two novel agent types including: (i) Radio Access Network optimizers,
and (ii) multi-agent negotiators for Service-Level Agreements (SLAs). We
further propose an end-to-end architecture for AGI networks and evaluate it on
a 5G testbed capturing channel fluctuations from moving vehicles. Results show
that symbiotic agents reduce decision errors fivefold compared to standalone
LLM-based agents, while smaller language models (SLM) achieve similar accuracy
with a 99.9% reduction in GPU resource overhead and in near-real-time loops of
82 ms. A multi-agent demonstration for collaborative RAN on the real-world
testbed highlights significant flexibility in service-level agreement and
resource allocation, reducing RAN over-utilization by approximately 44%.
Drawing on our findings and open-source implementations, we introduce the
symbiotic paradigm as the foundation for next-generation, AGI-driven
networks-systems designed to remain adaptable, efficient, and trustworthy even
as LLMs advance.

### 6. [LLM Meets the Sky: Heuristic Multi-Agent Reinforcement Learning for Secure Heterogeneous UAV Networks](http://arxiv.org/pdf/2507.17188v1)

Authors: Lijie Zheng, Ji He, Shih Yu Chang, Yulong Shen, Dusit Niyato

This work tackles the physical layer security (PLS) problem of maximizing the
secrecy rate in heterogeneous UAV networks (HetUAVNs) under propulsion energy
constraints. Unlike prior studies that assume uniform UAV capabilities or
overlook energy-security trade-offs, we consider a realistic scenario where
UAVs with diverse payloads and computation resources collaborate to serve
ground terminals in the presence of eavesdroppers. To manage the complex
coupling between UAV motion and communication, we propose a hierarchical
optimization framework. The inner layer uses a semidefinite relaxation
(SDR)-based S2DC algorithm combining penalty functions and difference-of-convex
(d.c.) programming to solve the secrecy precoding problem with fixed UAV
positions. The outer layer introduces a Large Language Model (LLM)-guided
heuristic multi-agent reinforcement learning approach (LLM-HeMARL) for
trajectory optimization. LLM-HeMARL efficiently incorporates expert heuristics
policy generated by the LLM, enabling UAVs to learn energy-aware,
security-driven trajectories without the inference overhead of real-time LLM
calls. The simulation results show that our method outperforms existing
baselines in secrecy rate and energy efficiency, with consistent robustness
across varying UAV swarm sizes and random seeds.

### 7. [Rethinking HSM and TPM Security in the Cloud: Real-World Attacks and Next-Gen Defenses](http://arxiv.org/pdf/2507.17655v1)

Authors: Shams Shaikh, Trima P. Fernandes e Fizardo

As organizations rapidly migrate to the cloud, the security of cryptographic
key management has become a growing concern. Hardware Security Modules (HSMs)
and Trusted Platform Modules (TPMs), traditionally seen as the gold standard
for securing encryption keys and digital trust, are increasingly challenged by
cloud-native threats. Real-world breaches have exposed weaknesses in cloud
deployments, including misconfigurations, API abuse, and privilege escalations,
allowing attackers to access sensitive key material and bypass protections.
These incidents reveal that while the hardware remains secure, the surrounding
cloud ecosystem introduces systemic vulnerabilities. This paper analyzes
notable security failures involving HSMs and TPMs, identifies common attack
vectors, and questions longstanding assumptions about their effectiveness in
distributed environments. We explore alternative approaches such as
confidential computing, post-quantum cryptography, and decentralized key
management. Our findings highlight that while HSMs and TPMs still play a role,
modern cloud security requires more adaptive, layered architectures. By
evaluating both current weaknesses and emerging models, this research equips
cloud architects and security engineers with strategies to reinforce
cryptographic trust in the evolving threat landscape.

### 8. [Our Cars Can Talk: How IoT Brings AI to Vehicles](http://arxiv.org/pdf/2507.17214v1)

Authors: Amod Kant Agrawal

Bringing AI to vehicles and enabling them as sensing platforms is key to
transforming maintenance from reactive to proactive. Now is the time to
integrate AI copilots that speak both languages: machine and driver. This
article offers a conceptual and technical perspective intended to spark
interdisciplinary dialogue and guide future research and development in
intelligent vehicle systems, predictive maintenance, and AI-powered user
interaction.

### 9. [Symmetric Private Information Retrieval (SPIR) on Graph-Based Replicated Systems](http://arxiv.org/pdf/2507.17736v1)

Authors: Shreya Meel, Sennur Ulukus

We introduce the problem of symmetric private information retrieval (SPIR) on
replicated databases modeled by a simple graph. In this model, each vertex
corresponds to a server, and a message is replicated on two servers if and only
if there is an edge between them. We consider the setting where the server-side
common randomness necessary to accomplish SPIR is also replicated at the
servers according to the graph, and we call this as message-specific common
randomness. In this setting, we establish a lower bound on the SPIR capacity,
i.e., the maximum download rate, for general graphs, by proposing an achievable
SPIR scheme. Next, we prove that, for any SPIR scheme to be feasible, the
minimum size of message-specific randomness should be equal to the size of a
message. Finally, by providing matching upper bounds, we derive the exact SPIR
capacity for the class of path and regular graphs.

### Robotics

### 1. [MARSCalib: Multi-robot, Automatic, Robust, Spherical Target-based Extrinsic Calibration in Field and Extraterrestrial Environments](http://arxiv.org/pdf/2507.17130v1)

Authors: Seokhwan Jeong, Hogyun Kim, Younggun Cho

This paper presents a novel spherical target-based LiDAR-camera extrinsic
calibration method designed for outdoor environments with multi-robot systems,
considering both target and sensor corruption. The method extracts the 2D
ellipse center from the image and the 3D sphere center from the pointcloud,
which are then paired to compute the transformation matrix. Specifically, the
image is first decomposed using the Segment Anything Model (SAM). Then, a novel
algorithm extracts an ellipse from a potentially corrupted sphere, and the
extracted center of ellipse is corrected for errors caused by the perspective
projection model. For the LiDAR pointcloud, points on the sphere tend to be
highly noisy due to the absence of flat regions. To accurately extract the
sphere from these noisy measurements, we apply a hierarchical weighted sum to
the accumulated pointcloud. Through experiments, we demonstrated that the
sphere can be robustly detected even under both types of corruption,
outperforming other targets. We evaluated our method using three different
types of LiDARs (spinning, solid-state, and non-repetitive) with cameras
positioned in three different locations. Furthermore, we validated the
robustness of our method to target corruption by experimenting with spheres
subjected to various types of degradation. These experiments were conducted in
both a planetary test and a field environment. Our code is available at
https://github.com/sparolab/MARSCalib.

### 2. [Dynamic Modeling and Dimensional Optimization of Legged Mechanisms for Construction Robot](http://arxiv.org/pdf/2507.17132v1)

Authors: Xiao Liu, Xianlong Yang, Weijun Wang, Wei Feng

With the rapid development of the construction industry, issues such as harsh
working environments, high-intensity and high-risk tasks, and labor shortages
have become increasingly prominent. This drives higher demands for construction
robots in terms of low energy consumption, high mobility, and high load
capacity. This paper focuses on the design and optimization of leg structures
for construction robots, aiming to improve their dynamic performance, reduce
energy consumption, and enhance load-bearing capabilities. Firstly, based on
the leg configuration of ants in nature, we design a structure for the robot's
leg. Secondly, we propose a novel structural optimization method. Using the
Lagrangian approach, a dynamic model of the leg was established. Combining the
dynamic model with the leg's motion trajectory, we formulated multiple dynamic
evaluation metrics and conducted a comprehensive optimization study on the
geometric parameters of each leg segment. The results show that the optimized
leg structure reduces peak joint torques and energy consumption by over 20%.
Finally, dynamic simulation experiments were conducted using ADAMS. The results
demonstrate a significant reduction in the driving power of each joint after
optimization, validating the effectiveness and rationality of the proposed
strategy. This study provides a theoretical foundation and technical support
for the design of heavy-load, high-performance construction robots.

### 3. [Dynamic Parameter Identification of a Curtain Wall Installation Robotic Arm](http://arxiv.org/pdf/2507.17136v1)

Authors: Xiao Liu, Yunxiao Cheng, Weijun Wang, Tianlun Huang, Wei Feng

In the construction industry, traditional methods fail to meet the modern
demands for efficiency and quality. The curtain wall installation is a critical
component of construction projects. We design a hydraulically driven robotic
arm for curtain wall installation and a dynamic parameter identification
method. We establish a Denavit-Hartenberg (D-H) model based on measured robotic
arm structural parameters and integrate hydraulic cylinder dynamics to
construct a composite parametric system driven by a Stribeck friction model. By
designing high-signal-to-noise ratio displacement excitation signals for
hydraulic cylinders and combining Fourier series to construct optimal
excitation trajectories that satisfy joint constraints, this method effectively
excites the characteristics of each parameter in the minimal parameter set of
the dynamic model of the robotic arm. On this basis, a hierarchical progressive
parameter identification strategy is proposed: least squares estimation is
employed to separately identify and jointly calibrate the dynamic parameters of
both the hydraulic cylinder and the robotic arm, yielding Stribeck model curves
for each joint. Experimental validation on a robotic arm platform demonstrates
residual standard deviations below 0.4 Nm between theoretical and measured
joint torques, confirming high-precision dynamic parameter identification for
the hydraulic-driven curtain wall installation robotic arm. This significantly
contributes to enhancing the intelligence level of curtain wall installation
operations.

### 4. [Multi-Objective Trajectory Planning for a Robotic Arm in Curtain Wall Installation](http://arxiv.org/pdf/2507.17140v1)

Authors: Xiao Liu, Yunxiao Cheng, Weijun Wang, Tianlun Huang, Zhiyong Wang, Wei Feng

In the context of labor shortages and rising costs, construction robots are
regarded as the key to revolutionizing traditional construction methods and
improving efficiency and quality in the construction industry. In order to
ensure that construction robots can perform tasks efficiently and accurately in
complex construction environments, traditional single-objective trajectory
optimization methods are difficult to meet the complex requirements of the
changing construction environment. Therefore, we propose a multi-objective
trajectory optimization for the robotic arm used in the curtain wall
installation. First, we design a robotic arm for curtain wall installation,
integrating serial, parallel, and folding arm elements, while considering its
physical properties and motion characteristics. In addition, this paper
proposes an NSGA-III-FO algorithm (NSGA-III with Focused Operator, NSGA-III-FO)
that incorporates a focus operator screening mechanism to accelerate the
convergence of the algorithm towards the Pareto front, thereby effectively
balancing the multi-objective constraints of construction robots. The proposed
algorithm is tested against NSGA-III, MOEA/D, and MSOPS-II in ten consecutive
trials on the DTLZ3 and WFG3 test functions, showing significantly better
convergence efficiency than the other algorithms. Finally, we conduct two sets
of experiments on the designed robotic arm platform, which confirm the
efficiency and practicality of the NSGA-III-FO algorithm in solving
multi-objective trajectory planning problems for curtain wall installation
tasks.

### 5. [Falconry-like palm landing by a flapping-wing drone based on the human gesture interaction and distance-aware flight planning](http://arxiv.org/pdf/2507.17144v1)

Authors: Kazuki Numazato, Keiichiro Kan, Masaki Kitagawa, Yunong Li, Johannes Kubel, Moju Zhao

Flapping-wing drones have attracted significant attention due to their
biomimetic flight. They are considered more human-friendly due to their
characteristics such as low noise and flexible wings, making them suitable for
human-drone interactions. However, few studies have explored the practical
interaction between humans and flapping-wing drones. On establishing a physical
interaction system with flapping-wing drones, we can acquire inspirations from
falconers who guide birds of prey to land on their arms. This interaction
interprets the human body as a dynamic landing platform, which can be utilized
in various scenarios such as crowded or spatially constrained environments.
Thus, in this study, we propose a falconry-like interaction system in which a
flapping-wing drone performs a palm landing motion on a human hand. To achieve
a safe approach toward humans, we design a trajectory planning method that
considers both physical and psychological factors of the human safety such as
the drone's velocity and distance from the user. We use a commercial flapping
platform with our implemented motion planning and conduct experiments to
evaluate the palm landing performance and safety. The results demonstrate that
our approach enables safe and smooth hand landing interactions. To the best of
our knowledge, it is the first time to achieve a contact-based interaction
between flapping-wing drones and humans.

### 6. [Reconfigurable Tendon-Driven Robots: Eliminating Inter-segmental Coupling via Independently Lockable Joints](http://arxiv.org/pdf/2507.17163v1)

Authors: Botao Lin, Shuang Song, Jiaole Wang

With a slender redundant body, the tendon-driven robot (TDR) has a large
workspace and great maneuverability while working in complex environments. TDR
comprises multiple independently controlled robot segments, each with a set of
driving tendons. While increasing the number of robot segments enhances
dexterity and expands the workspace, this structural expansion also introduces
intensified inter-segmental coupling. Therefore, achieving precise TDR control
requires more complex models and additional motors. This paper presents a
reconfigurable tendon-driven robot (RTR) equipped with innovative lockable
joints. Each joint's state (locked/free) can be individually controlled through
a pair of antagonistic tendons, and its structure eliminates the need for a
continuous power supply to maintain the state. Operators can selectively
actuate the targeted robot segments, and this scheme fundamentally eliminates
the inter-segmental coupling, thereby avoiding the requirement for complex
coordinated control between segments. The workspace of RTR has been simulated
and compared with traditional TDRs' workspace, and RTR's advantages are further
revealed. The kinematics and statics models of the RTR have been derived and
validation experiments have been conducted. Demonstrations have been performed
using a seven-joint RTR prototype to show its reconfigurability and moving
ability in complex environments with an actuator pack comprising only six
motors.

### 7. [FAST-Calib: LiDAR-Camera Extrinsic Calibration in One Second](http://arxiv.org/pdf/2507.17210v1)

Authors: Chunran Zheng, Fu Zhang

This paper proposes FAST-Calib, a fast and user-friendly LiDAR-camera
extrinsic calibration tool based on a custom-made 3D target. FAST-Calib
supports both mechanical and solid-state LiDARs by leveraging an efficient and
reliable edge extraction algorithm that is agnostic to LiDAR scan patterns. It
also compensates for edge dilation artifacts caused by LiDAR spot spread
through ellipse fitting, and supports joint optimization across multiple
scenes. We validate FAST-Calib on three LiDAR models (Ouster, Avia, and
Mid360), each paired with a wide-angle camera. Experimental results demonstrate
superior accuracy and robustness compared to existing methods. With
point-to-point registration errors consistently below 6.5mm and total
processing time under 0.7s, FAST-Calib provides an efficient, accurate, and
target-based automatic calibration pipeline. We have open-sourced our code and
dataset on GitHub to benefit the robotics community.

### 8. [Optimizing Delivery Logistics: Enhancing Speed and Safety with Drone Technology](http://arxiv.org/pdf/2507.17253v1)

Authors: Maharshi Shastri, Ujjval Shrivastav

The increasing demand for fast and cost effective last mile delivery
solutions has catalyzed significant advancements in drone based logistics. This
research describes the development of an AI integrated drone delivery system,
focusing on route optimization, object detection, secure package handling, and
real time tracking. The proposed system leverages YOLOv4 Tiny for object
detection, the NEO 6M GPS module for navigation, and the A7670 SIM module for
real time communication. A comparative analysis of lightweight AI models and
hardware components is conducted to determine the optimal configuration for
real time UAV based delivery. Key challenges including battery efficiency,
regulatory compliance, and security considerations are addressed through the
integration of machine learning techniques, IoT devices, and encryption
protocols. Preliminary studies demonstrate improvement in delivery time
compared to conventional ground based logistics, along with high accuracy
recipient authentication through facial recognition. The study also discusses
ethical implications and societal acceptance of drone deliveries, ensuring
compliance with FAA, EASA and DGCA regulatory standards. Note: This paper
presents the architecture, design, and preliminary simulation results of the
proposed system. Experimental results, simulation benchmarks, and deployment
statistics are currently being acquired. A comprehensive analysis will be
included in the extended version of this work.

### 9. [HuNavSim 2.0](http://arxiv.org/pdf/2507.17317v1)

Authors: Miguel Escudero-Jiménez, Noé Pérez-Higueras, Andrés Martínez-Silva, Fernando Caballero, Luis Merino

This work presents a new iteration of the Human Navigation Simulator
(HuNavSim), a novel open-source tool for the simulation of different
human-agent navigation behaviors in scenarios with mobile robots. The tool,
programmed under the ROS 2 framework, can be used together with different
well-known robotics simulators such as Gazebo or NVidia Isaac Sim. The main
goal is to facilitate the development and evaluation of human-aware robot
navigation systems in simulation. In this new version, several features have
been improved and new ones added, such as the extended set of actions and
conditions that can be combined in Behavior Trees to compound complex and
realistic human behaviors.

### 10. [Mobile Manipulation with Active Inference for Long-Horizon Rearrangement Tasks](http://arxiv.org/pdf/2507.17338v1)

Authors: Corrado Pezzato, Ozan Çatal, Toon Van de Maele, Riddhi J. Pitliya, Tim Verbelen

Despite growing interest in active inference for robotic control, its
application to complex, long-horizon tasks remains untested. We address this
gap by introducing a fully hierarchical active inference architecture for
goal-directed behavior in realistic robotic settings. Our model combines a
high-level active inference model that selects among discrete skills realized
via a whole-body active inference controller. This unified approach enables
flexible skill composition, online adaptability, and recovery from task
failures without requiring offline training. Evaluated on the Habitat Benchmark
for mobile manipulation, our method outperforms state-of-the-art baselines
across the three long-horizon tasks, demonstrating for the first time that
active inference can scale to the complexity of modern robotics benchmarks.

### Software Engineering

### 1. [Assessing Reliability of Statistical Maximum Coverage Estimators in Fuzzing](http://arxiv.org/pdf/2507.17093v1)

Authors: Danushka Liyanage, Nelum Attanayake, Zijian Luo, Rahul Gopinath

Background: Fuzzers are often guided by coverage, making the estimation of
maximum achievable coverage a key concern in fuzzing. However, achieving 100%
coverage is infeasible for most real-world software systems, regardless of
effort. While static reachability analysis can provide an upper bound, it is
often highly inaccurate. Recently, statistical estimation methods based on
species richness estimators from biostatistics have been proposed as a
potential solution. Yet, the lack of reliable benchmarks with labeled ground
truth has limited rigorous evaluation of their accuracy.
  Objective: This work examines the reliability of reachability estimators from
two axes: addressing the lack of labeled ground truth and evaluating their
reliability on real-world programs.
  Methods: (1) To address the challenge of labeled ground truth, we propose an
evaluation framework that synthetically generates large programs with complex
control flows, ensuring well-defined reachability and providing ground truth
for evaluation. (2) To address the criticism from use of synthetic benchmarks,
we adapt a reliability check for reachability estimators on real-world
benchmarks without labeled ground truth -- by varying the size of sampling
units, which, in theory, should not affect the estimate.
  Results: These two studies together will help answer the question of whether
current reachability estimators are reliable, and defines a protocol to
evaluate future improvements in reachability estimation.

### 2. [Can LLMs Write CI? A Study on Automatic Generation of GitHub Actions Configurations](http://arxiv.org/pdf/2507.17165v1)

Authors: Taher A. Ghaleb, Dulina Rathnayake

Continuous Integration (CI) services, such as GitHub Actions, require
developers to write YAML-based configurations, which can be tedious and
error-prone. Despite the increasing use of Large Language Models (LLMs) to
automate software engineering tasks, their ability to generate CI
configurations remains underexplored. This paper presents a preliminary study
evaluating six LLMs for generating GitHub Actions configurations from natural
language descriptions. We assess three general-purpose foundation models
(GPT-4o, Llama, and Gemma) and three code-pretrained models (GPT-4.1, Code
Llama, and CodeGemma). We also introduce the first labeled dataset of its kind,
constructed from GitHub Actions documentation, pairing descriptions with
corresponding best-practice YAML configurations. Zero-shot prompting achieves
up to 69% similarity with the ground truth, with only 3% perfect matches.
Code-pretrained models slightly underperform compared to general-purpose ones
in YAML-based CI tasks, revealing LLM limitations for CI configuration
generation. Analyzing GPT-4o outputs reveals issues like missing or renamed
steps, misinterpreted descriptions, and unnecessary additions that may affect
structural and contextual correctness, indicating a gap between generation
quality and the precision required for executable CI configurations. Our
research offers insights for improving LLM alignment with configuration
languages and guiding future efforts on CI automation and tooling support.

### 3. [Lessons from a Big-Bang Integration: Challenges in Edge Computing and Machine Learning](http://arxiv.org/pdf/2507.17270v1)

Authors: Alessandro Aneggi, Andrea Janes

This experience report analyses a one year project focused on building a
distributed real-time analytics system using edge computing and machine
learning. The project faced critical setbacks due to a big-bang integration
approach, where all components developed by multiple geographically dispersed
partners were merged at the final stage. The integration effort resulted in
only six minutes of system functionality, far below the expected 40 minutes.
Through root cause analysis, the study identifies technical and organisational
barriers, including poor communication, lack of early integration testing, and
resistance to topdown planning. It also considers psychological factors such as
a bias toward fully developed components over mockups. The paper advocates for
early mock based deployment, robust communication infrastructures, and the
adoption of topdown thinking to manage complexity and reduce risk in reactive,
distributed projects. These findings underscore the limitations of traditional
Agile methods in such contexts and propose simulation-driven engineering and
structured integration cycles as key enablers for future success.

### 4. [Seed&Steer: Guiding Large Language Models with Compilable Prefix and Branch Signals for Unit Test Generation](http://arxiv.org/pdf/2507.17271v1)

Authors: Shuaiyu Zhou, Zhengran Zeng, Xiaoling Zhou, Rui Xie, Shikun Zhang, Wei Ye

Unit tests play a vital role in the software development lifecycle. Recent
advances in Large Language Model (LLM)-based approaches have significantly
improved automated test generation, garnering attention from both academia and
industry. We revisit LLM-based unit test generation from a novel perspective by
decoupling prefix generation and assertion generation. To characterize their
respective challenges, we define Initialization Complexity and adopt Cyclomatic
Complexity to measure the difficulty of prefix and assertion generation,
revealing that the former primarily affects compilation success, while the
latter influences test coverage. To address these challenges, we propose
Seed&Steer, a two-step approach that combines traditional unit testing
techniques with the capabilities of large language models. Seed&Steer leverages
conventional unit testing tools (e.g., EvoSuite) to generate method invocations
with high compilation success rates, which serve as seeds to guide LLMs in
constructing effective test contexts. It then introduces branching cues to help
LLMs explore diverse execution paths (e.g., normal, boundary, and exception
cases) and generate assertions with high coverage. We evaluate Seed&Steer on
five real-world Java projects against state-of-the-art baselines. Results show
that Seed&Steer improves the compilation pass rate by approximately 7%,
successfully compiling 792 and 887 previously failing cases on two LLMs. It
also achieves up to ~73% branch and line coverage across focal methods of
varying complexity, with coverage improvements ranging from 1.09* to 1.26*. Our
code, dataset, and experimental scripts will be publicly released to support
future research and reproducibility.

### 5. [How Do Code Smells Affect Skill Growth in Scratch Novice Programmers?](http://arxiv.org/pdf/2507.17314v1)

Authors: Ricardo Hidalgo Aragón, Jesús M. González-Barahona, Gregorio Robles

Context. Code smells, which are recurring anomalies in design or style, have
been extensively researched in professional code. However, their significance
in block-based projects created by novices is still largely unknown.
Block-based environments such as Scratch offer a unique, data-rich setting to
examine how emergent design problems intersect with the cultivation of
computational-thinking (CT) skills. Objective. This research explores the
connection between CT proficiency and design-level code smells--issues that may
hinder software maintenance and evolution--in programs created by Scratch
developers. We seek to identify which CT dimensions align most strongly with
which code smells and whether task context moderates those associations.
Method. A random sample of aprox. 2 million public Scratch projects is mined.
Using open-source linters, we extract nine CT scores and 40 code smell
indicators from these projects. After rigorous pre-processing, we apply
descriptive analytics, robust correlation tests, stratified cross-validation,
and exploratory machine-learning models; qualitative spot-checks contextualize
quantitative patterns. Impact. The study will deliver the first large-scale,
fine-grained map linking specific CT competencies to concrete design flaws and
antipatterns. Results are poised to (i) inform evidence-based curricula and
automated feedback systems, (ii) provide effect-size benchmarks for future
educational interventions, and (iii) supply an open, pseudonymized dataset and
reproducible analysis pipeline for the research community. By clarifying how
programming habits influence early skill acquisition, the work advances both
computing-education theory and practical tooling for sustainable software
maintenance and evolution.

### 6. [Roseau: Fast, Accurate, Source-based API Breaking Change Analysis in Java](http://arxiv.org/pdf/2507.17369v1)

Authors: Corentin Latappy, Thomas Degueule, Jean-Rémy Falleri, Romain Robbes, Lina Ochoa

Understanding API evolution and the introduction of breaking changes (BCs) in
software libraries is essential for library maintainers to manage backward
compatibility and for researchers to conduct empirical studies on software
library evolution. In Java, tools such as JApiCmp and Revapi are commonly used
to detect BCs between library releases, but their reliance on binary JARs
limits their applicability. This restriction hinders large-scale longitudinal
studies of API evolution and fine-grained analyses such as commit-level BC
detection. In this paper, we introduce Roseau, a novel static analysis tool
that constructs technology-agnostic API models from library code equipped with
rich semantic analyses. API models can be analyzed to study API evolution and
compared to identify BCs between any two versions of a library (releases,
commits, branches, etc.). Unlike traditional approaches, Roseau can build API
models from source code or bytecode, and is optimized for large-scale
longitudinal analyses of library histories. We assess the accuracy,
performance, and suitability of Roseau for longitudinal studies of API
evolution, using JApiCmp and Revapi as baselines. We extend and refine an
established benchmark of BCs and show that Roseau achieves higher accuracy (F1
= 0.99) than JApiCmp (F1 = 0.86) and Revapi (F1 = 0.91). We analyze 60 popular
libraries from Maven Central and find that Roseau delivers excellent
performance, detecting BCs between versions in under two seconds, including in
libraries with hundreds of thousands of lines of code. We further illustrate
the limitations of JApiCmp and Revapi for longitudinal studies and the novel
analysis capabilities offered by Roseau by tracking the evolution of Google's
Guava API and the introduction of BCs over 14 years and 6,839 commits, reducing
analysis times from a few days to a few minutes.

### 7. [AssertFlip: Reproducing Bugs via Inversion of LLM-Generated Passing Tests](http://arxiv.org/pdf/2507.17542v1)

Authors: Lara Khatib, Noble Saji Mathews, Meiyappan Nagappan

Bug reproduction is critical in the software debugging and repair process,
yet the majority of bugs in open-source and industrial settings lack executable
tests to reproduce them at the time they are reported, making diagnosis and
resolution more difficult and time-consuming. To address this challenge, we
introduce AssertFlip, a novel technique for automatically generating Bug
Reproducible Tests (BRTs) using large language models (LLMs). Unlike existing
methods that attempt direct generation of failing tests, AssertFlip first
generates passing tests on the buggy behaviour and then inverts these tests to
fail when the bug is present. We hypothesize that LLMs are better at writing
passing tests than ones that crash or fail on purpose. Our results show that
AssertFlip outperforms all known techniques in the leaderboard of SWT-Bench, a
benchmark curated for BRTs. Specifically, AssertFlip achieves a fail-to-pass
success rate of 43.6% on the SWT-Bench-Verified subset.

### 8. [CodeReasoner: Enhancing the Code Reasoning Ability with Reinforcement Learning](http://arxiv.org/pdf/2507.17548v1)

Authors: Lingxiao Tang, He Ye, Zhongxin Liu, Xiaoxue Ren, Lingfeng Bao

Code reasoning is a fundamental capability for large language models (LLMs)
in the code domain. It involves understanding and predicting a program's
execution behavior, such as determining the output for a given input or whether
a specific statement will be executed. This capability is essential for
downstream tasks like debugging, code generation, and program repair. Prior
approaches mainly rely on supervised fine-tuning to improve performance in code
reasoning tasks. However, they often show limited gains and fail to generalize
across diverse scenarios. We argue this is due to two core issues: the low
quality of training data and the limitations of supervised fine-tuning, which
struggles to teach general reasoning skills. To address these challenges, we
propose CodeReasoner, a framework that spans both dataset construction and a
two-stage training process. First, we introduce a method to construct datasets
that focus on the core execution logic of Python programs. Next, we apply
instruction tuning to inject execution-specific knowledge distilled from a
powerful teacher model. We then enhance reasoning and generalization through
GRPO reinforcement learning on top of the fine-tuned model. Experiments on
three widely-used code reasoning benchmarks show that CodeReasoner improves
performance by 27.1% to 40.2% over prior methods using a 7B model. Notably, the
7B model matches GPT-4o on key tasks like input/output and coverage prediction.
When scaled to 14B, CodeReasoner outperforms GPT-4o across all benchmarks.
Ablation studies confirm the effectiveness of each training stage and highlight
the importance of reasoning chains.

### 9. [Contextual Code Retrieval for Commit Message Generation: A Preliminary Study](http://arxiv.org/pdf/2507.17690v1)

Authors: Bo Xiong, Linghao Zhang, Chong Wang, Peng Liang

A commit message describes the main code changes in a commit and plays a
crucial role in software maintenance. Existing commit message generation (CMG)
approaches typically frame it as a direct mapping which inputs a code diff and
produces a brief descriptive sentence as output. However, we argue that relying
solely on the code diff is insufficient, as raw code diff fails to capture the
full context needed for generating high-quality and informative commit
messages. In this paper, we propose a contextual code retrieval-based method
called C3Gen to enhance CMG by retrieving commit-relevant code snippets from
the repository and incorporating them into the model input to provide richer
contextual information at the repository scope. In the experiments, we
evaluated the effectiveness of C3Gen across various models using four objective
and three subjective metrics. Meanwhile, we design and conduct a human
evaluation to investigate how C3Gen-generated commit messages are perceived by
human developers. The results show that by incorporating contextual code into
the input, C3Gen enables models to effectively leverage additional information
to generate more comprehensive and informative commit messages with greater
practical value in real-world development scenarios. Further analysis
underscores concerns about the reliability of similaritybased metrics and
provides empirical insights for CMG.

### 10. [Educational Insights from Code: Mapping Learning Challenges in Object-Oriented Programming through Code-Based Evidence](http://arxiv.org/pdf/2507.17743v1)

Authors: Andre Menolli, Bruno Strik

Object-Oriented programming is frequently challenging for undergraduate
Computer Science students, particularly in understanding abstract concepts such
as encapsulation, inheritance, and polymorphism. Although the literature
outlines various methods to identify potential design and coding issues in
object-oriented programming through source code analysis, such as code smells
and SOLID principles, few studies explore how these code-level issues relate to
learning difficulties in Object-Oriented Programming. In this study, we explore
the relationship of the code issue indicators with common challenges
encountered during the learning of object-oriented programming. Using
qualitative analysis, we identified the main categories of learning
difficulties and, through a literature review, established connections between
these difficulties, code smells, and violations of the SOLID principles. As a
result, we developed a conceptual map that links code-related issues to
specific learning challenges in Object-Oriented Programming. The model was then
evaluated by an expert who applied it in the analysis of the student code to
assess its relevance and applicability in educational contexts.

### Social and Information Networks

### 1. [Dynamics of temporal influence in polarised networks](http://arxiv.org/pdf/2507.17177v1)

Authors: Caroline B. Pena, David J. P. O'Sullivan, Pádraig MacCarron, Akrati Saxena

In social networks, it is often of interest to identify the most influential
users who can successfully spread information to others. This is particularly
important for marketing (e.g., targeting influencers for a marketing campaign)
and to understand the dynamics of information diffusion (e.g., who is the most
central user in the spreading of a certain type of information). However,
different opinions often split the audience and make the network polarised. In
polarised networks, information becomes soiled within communities in the
network, and the most influential user within a network might not be the most
influential across all communities. Additionally, influential users and their
influence may change over time as users may change their opinion or choose to
decrease or halt their engagement on the subject. In this work, we aim to study
the temporal dynamics of users' influence in a polarised social network. We
compare the stability of influence ranking using temporal centrality measures,
while extending them to account for community structure across a number of
network evolution behaviours. We show that we can successfully aggregate nodes
into influence bands, and how to aggregate centrality scores to analyse the
influence of communities over time. A modified version of the temporal
independent cascade model and the temporal degree centrality perform the best
in this setting, as they are able to reliably isolate nodes into their bands.

### 2. [Quotegraph: A Social Network Extracted from Millions of News Quotations](http://arxiv.org/pdf/2507.17626v1)

Authors: Marko Čuljak, Robert West, Andreas Spitz, Akhil Arora

We introduce Quotegraph, a novel large-scale social network derived from
speaker-attributed quotations in English news articles published between 2008
and 2020. Quotegraph consists of 528 thousand unique nodes and 8.63 million
directed edges, pointing from speakers to persons they mention. The nodes are
linked to their corresponding items in Wikidata, thereby endowing the dataset
with detailed biographic entity information, including nationality, gender, and
political affiliation. Being derived from Quotebank, a massive corpus of
quotations, relations in Quotegraph are additionally enriched with the
information about the context in which they are featured. Each part of the
network construction pipeline is language agnostic, enabling the construction
of similar datasets based on non-English news corpora. We believe Quotegraph is
a compelling resource for computational social scientists, complementary to
online social networks, with the potential to yield novel insights into the
behavior of public figures and how it is captured in the news.

### 3. [Triadic First-Order Logic Queries in Temporal Networks](http://arxiv.org/pdf/2507.17215v1)

Authors: Omkar Bhalerao, Yunjie Pan, C. Seshadhri, Nishil Talati

Motif counting is a fundamental problem in network analysis, and there is a
rich literature of theoretical and applied algorithms for this problem. Given a
large input network $G$, a motif $H$ is a small "pattern" graph indicative of
special local structure. Motif/pattern mining involves finding all matches of
this pattern in the input $G$. The simplest, yet challenging, case of motif
counting is when $H$ has three vertices, often called a "triadic" query. Recent
work has focused on "temporal graph mining", where the network $G$ has edges
with timestamps (and directions) and $H$ has time constraints.
  Inspired by concepts in logic and database theory, we introduce the study of
"thresholded First Order Logic (FOL) Motif Analysis" for massive temporal
networks. A typical triadic motif query asks for the existence of three
vertices that form a desired temporal pattern. An "FOL" motif query is obtained
by having both existential and thresholded universal quantifiers. This allows
for query semantics that can mine richer information from networks. A typical
triadic query would be "find all triples of vertices $u,v,w$ such that they
form a triangle within one hour". A thresholded FOL query can express "find all
pairs $u,v$ such that for half of $w$ where $(u,w)$ formed an edge, $(v,w)$
also formed an edge within an hour".
  We design the first algorithm, FOLTY, for mining thresholded FOL triadic
queries. The theoretical running time of FOLTY matches the best known running
time for temporal triangle counting in sparse graphs. We give an efficient
implementation of FOLTY using specialized temporal data structures. FOLTY has
excellent empirical behavior, and can answer triadic FOL queries on graphs with
nearly 70M edges is less than hour on commodity hardware. Our work has the
potential to start a new research direction in the classic well-studied problem
of motif analysis.

### Systems and Control

### 1. [Transient Stability-Driven Planning for the Optimal Sizing of Resilient AC/DC Hybrid Microgrids](http://arxiv.org/pdf/2507.17110v1)

Authors: Yi Wang, Goran Strbac

This paper proposes a transient stability-driven planning framework for the
optimal sizing problem of resilient AC/DC hybrid microgrids (HMGs) under
different types of contingencies, capturing frequency and voltage stability
requirements as well as the frequency-voltage coupling dynamics of AC/DC
interlinking converters (ICs). The planning model is formulated into a
defender-attacker-defender (DAD) architecture, which can be further merged into
two levels, i.e., upper-level and low-level problems, and then iteratively
solved by an enhanced genetic algorithm with sparsity calculation and local
search. Regarding the operation stage, a novel transient stability-constrained
optimal power flow (TSC-OPF) algorithm is proposed for static and transient
operations of HMGs, capturing governor dynamics and automatic voltage regulator
of conventional generators as well as the droop control dynamics of
inverter-based resources (IBRs) for frequency control and voltage control,
respectively. Furthermore, a Lyapunov optimisation approach is developed to
capture the time-coupling property of energy storages (ESs) and then allow the
TSC-OPF to be solved on an hourly basis with a second-scale resolution,
achieving the co-optimisation of static and transient stability requirements.
Case studies have been conducted to verify the effectiveness of the proposed
planning framework in obtaining cost-effective investment decisions for various
resources while respecting transient stability requirements under different
contingencies.

### 2. [Maintenance-free condition monitoring system based on lora](http://arxiv.org/pdf/2507.17156v1)

Authors: Honglin Zhang, Mingtong Chen, Zhengbao Yang

With the rising volume of railroad transportation, the traditional track
inspection mainly relies on manual inspection and large-scale inspection
equipment, which not only has low inspection frequency and lagging response,
but also has the defects of high risk, high cost and easy to miss inspection.
To this end, this study designs and realizes a maintenance-free railroad track
wireless monitoring system based on LoRa module LM401. Each monitoring node
consists of an STM32 microcontroller, an LM401 LoRa transceiver, a low-power
ADXL362 triaxial acceleration sensor, a digital temperature sensor (LMT85), and
a digital barometric pressure sensor (RSCM17100KP101). The system collects
vibration data through the SPI1 interface at the node end, periodically reads
the temperature and barometric pressure information, and packages and sends the
data to a centralized gateway within a range of 500 m using the LoRa star
topology; the gateway then uploads the data in real time to a cloud server
through a 4G module, which supports the MQTT protocol. MQTT protocol is
supported. Laboratory tests and field deployments show that the system can
realize acceleration resolution of 0.01 g, reduce maintenance cost by about
70%, and improve monitoring efficiency by more than 5 times. The system
provides a reliable means for intelligent rail health management, and in the
future, it is planned to introduce RF energy collection technology to realize
automatic wake-up without battery, and expand to urban bridges, tunnels and
environmental monitoring and other multi-scenario applications.

### 3. [Ontological Definition of Seamless Digital Engineering Based on ISO/IEC 25000-Series SQuaRE Product Quality Model](http://arxiv.org/pdf/2507.17171v1)

Authors: James S. Wheaton, Daniel R. Herber

Since the introduction of Digital Engineering (DE) as a well-defined concept
in 2018, organizations and industry groups have been working to interpret the
DE concepts to establish consistent meta-models of those interrelated concepts
for integration into their DE processes and tools. To reach the breadth and
depth of DE concept definitions, the interpretation of international standard
sources is necessary, including ISO/IEC/IEEE 15288, 24765, 42000-series, 15408,
15206, 27000-series, and 25000-series, to effectively model the knowledge
domain where digital engineering applies. The harmonization of the concepts
used in these international standards continues to improve with each revision,
but it may be more effectively accomplished by relying on the descriptive logic
formalized in the Web Ontology Language (OWL 2 DL). This paper presents a
verified and consistent ontology based on the Basic Formal Ontology (BFO) and
Common Core Ontologies (CCO) that defines Seamless Digital Engineering as a
digital tooling paradigm that relies on formal verification of digital
interfaces to provide a system-level qualification of the assured integrity of
a Digital Engineering Environment. The present work defines classes and
equivalence axioms, while using only the BFO- and CCO-defined object properties
that relate them, to provide a baseline analysis that may inform future
DE-related ontology development, using a case study to formally define the
`seamless' quality in relation to the updated ISO 25010 SQuaRE product quality
model. We identified ISO meta-model inconsistencies that are resolvable using
the BFO/CCO ontological framework, and define `seamless' as both a system
integration quality and a Human-Computer Interface quality-in-use, working to
disambiguate this concept in the context of DE.

### 4. [On the Construction of Barrier Certificate: A Dynamic Programming Perspective](http://arxiv.org/pdf/2507.17222v1)

Authors: Yu Chen, Shaoyuan Li, Xiang Yin

In this paper, we revisit the formal verification problem for stochastic
dynamical systems over finite horizon using barrier certificates. Most existing
work on this topic focuses on safety properties by constructing barrier
certificates based on the notion of $c$-martingales. In this work, we first
provide a new insight into the conditions of existing martingale-based barrier
certificates from the perspective of dynamic programming operators.
Specifically, we show that the existing conditions essentially provide a bound
on the dynamic programming solution, which exactly characterizes the safety
probability. Based on this new perspective, we demonstrate that the barrier
conditions in existing approaches are unnecessarily conservative over unsafe
states. To address this, we propose a new set of safety barrier certificate
conditions that are strictly less conservative than existing ones, thereby
providing tighter probability bounds for safety verification. We further extend
our approach to the case of reach-avoid specifications by providing a set of
new barrier certificate conditions. We also illustrate how to search for these
new barrier certificates using sum-of-squares (SOS) programming. Finally, we
use two numerical examples to demonstrate the advantages of our method compared
to existing approaches.

### 5. [Integrating Grid impedance estimation method into Advanced Angle Estimation Kalman Filter in GFL inverter](http://arxiv.org/pdf/2507.17325v1)

Authors: Phuoc Sang Nguyen, Ghavameddin Nourbakhsh, Gerard Ledwich

The growing integration of power electronic converter-interfaced distributed
energy resources into modern power systems presents significant challenges for
system monitoring, protection, and control. Grid impedance plays a critical
role in the operation and stability assessment of grid-connected inverter
systems. This study presents a real-time grid impedance estimation method based
on the Discrete Fourier Transform. The proposed method is integrated with the
Advanced Angle Estimation Kalman Filter using a Linear Quadratic Regulator
current controller (AAEKF-LQR), assisting the use of impedance information for
accurate instantaneous phase angle estimation. Simulation results confirm that
the proposed impedance estimation method interacts effectively with the
AAEKF-LQR controller, maintaining stable system performance under weak grid
conditions. The approach also demonstrates the ability to deliver fast and
accurate impedance estimation during operational variations in grid conditions,
thereby supporting stable inverter operation.

### 6. [Optimizing Car Resequencing on Mixed-Model Assembly Lines: Algorithm Development and Deployment](http://arxiv.org/pdf/2507.17422v1)

Authors: Andreas Karrenbauer, Bernd Kuhn, Kurt Mehlhorn, Paolo Luigi Rinaldi

The mixed-model assembly line (MMAL) is a production system used in the
automobile industry to manufacture different car models on the same conveyor,
offering a high degree of product customization and flexibility. However, the
MMAL also poses challenges, such as finding optimal sequences of models
satisfying multiple constraints and objectives related to production
performance, quality, and delivery -- including minimizing the number of color
changeovers in the Paint Shop, balancing the workload and setup times on the
assembly line, and meeting customer demand and delivery deadlines. We propose a
multi-objective algorithm to solve the MMAL resequencing problem under
consideration of all these aspects simultaneously. We also present empirical
results obtained from recorded event data of the production process over $4$
weeks following the deployment of our algorithm in the Saarlouis plant of
Ford-Werke GmbH. We achieved an improvement of the average batch size of about
$30\%$ over the old control software translating to a $23\%$ reduction of color
changeovers. Moreover, we reduced the spread of cars planned for a specific
date by $10\%$, reducing the risk of delays in delivery. We discuss
effectiveness and robustness of our algorithm in improving production
performance and quality as well as trade-offs and limitations.

### 7. [Model Predictive Control for Unlocking Energy Flexibility of Heat Pump and Thermal Energy Storage Systems: Experimental Results](http://arxiv.org/pdf/2507.17552v1)

Authors: Weihong Tang, Yun Li, Shalika Walker, Tamas Keviczky

Increasing penetration of renewable energy sources (RES) and electrification
of energy systems necessitates the engagement of demand-side management (DSM)
to help alleviate congestion in electricity grid. Heat pump and thermal energy
storage (HPTES) systems, being energy efficient solutions, are becoming popular
in modern buildings and are promising to contribute to demand-side management
(DSM) due to their significant share in household electricity consumption. For
typical HPTES systems, this paper presents a systematic design framework
covering a control-oriented modeling process and energy-flexible model
predictive control (MPC) design. The proposed MPC-based DSM strategy offers an
innovative solution for efficient DSM by following a two-step DSM framework. In
the first step, flexibility assessment is performed to quantitatively evaluate
the flexibility potential of the HPTES system by solving a mixed-integer
economic MPC problem. In the second step, flexibility exploitation is achieved
through reacting to feasible demand response (DR) requests while respecting
system constraints. Both numerical simulations and real-world experiments are
performed based on a real HPTES installation to showcase the viability and
effectiveness of the proposed design.

### 8. [A Joint Planning Model for Fixed and Mobile Electric Vehicle Charging Stations Considering Flexible Capacity Strategy](http://arxiv.org/pdf/2507.17587v1)

Authors: Zhe Yu, Xue Hu, Qin Wang

The widespread adoption of electric vehicles (EVs) has significantly
increased demand on both transportation and power systems, posing challenges to
their stable operation. To support the growing need for EV charging, both fixed
charging stations (FCSs) and mobile charging stations (MCSs) have been
introduced, serving as key interfaces between the power grid and traffic
network. Recognizing the importance of collaborative planning across these
sectors, this paper presents a two-stage joint planning model for FCSs and
MCSs, utilizing an improved alternating direction method of multipliers (ADMM)
algorithm. The primary goal of the proposed model is to transform the potential
negative impacts of large-scale EV integration into positive outcomes, thereby
enhancing social welfare through collaboration among multiple stakeholders. In
the first stage, we develop a framework for evaluating FCS locations,
incorporating assessments of EV hosting capacity and voltage stability. The
second stage introduces a joint planning model for FCSs and MCSs, aiming to
minimize the overall social costs of the EV charging system while maintaining a
reliable power supply. To solve the planning problem, we employ a combination
of mixed-integer linear programming, queueing theory, and sequential quadratic
programming. The improved ADMM algorithm couples the siting and sizing
decisions consistently by introducing coupling constraints, and supports a
distributed optimization framework that coordinates the interests of EV users,
MCS operators, and distribution system operators. Additionally, a flexible
capacity planning strategy that accounts for the multi-period development
potential of EVCS is proposed to reduce both the complexity and the investment
required for FCS construction. Finally, a case study with comparative
experiments demonstrates the effectiveness of the proposed models and solution
methods.

### 9. [Toward Federated DeePC: borrowing data from similar systems](http://arxiv.org/pdf/2507.17610v1)

Authors: Gert Vankan, Valentina Breschi, Simone Formentin

Data-driven predictive control approaches, in general, and Data-enabled
Predictive Control (DeePC), in particular, exploit matrices of raw input/output
trajectories for control design. These data are typically gathered only from
the system to be controlled. Nonetheless, the increasing connectivity and
inherent similarity of (mass-produced) systems have the potential to generate a
considerable amount of information that can be exploited to undertake a control
task. In light of this, we propose a preliminary federated extension of DeePC
that leverages a combination of input/output trajectories from multiple similar
systems for predictive control. Supported by a suite of numerical examples, our
analysis unveils the potential benefits of exploiting information from similar
systems and its possible downsides.

### 10. [Learning clusters of partially observed linear dynamical systems](http://arxiv.org/pdf/2507.17638v1)

Authors: Maryann Rui, Munther A. Dahleh

We study the problem of learning clusters of partially observed linear
dynamical systems from multiple input-output trajectories. This setting is
particularly relevant when there are limited observations (e.g., short
trajectories) from individual data sources, making direct estimation
challenging. In such cases, incorporating data from multiple related sources
can improve learning. We propose an estimation algorithm that leverages
different data requirements for the tasks of clustering and system
identification. First, short impulse responses are estimated from individual
trajectories and clustered. Then, refined models for each cluster are jointly
estimated using multiple trajectories. We establish end-to-end finite sample
guarantees for estimating Markov parameters and state space realizations and
highlight trade-offs among the number of observed systems, the trajectory
lengths, and the complexity of the underlying models.

### Machine Learning (Statistics Category)

### 1. [Nearly Minimax Discrete Distribution Estimation in Kullback-Leibler Divergence with High Probability](http://arxiv.org/pdf/2507.17316v1)

Authors: Dirk van der Hoeven, Julia Olkhovskaia, Tim van Erven

We consider the problem of estimating a discrete distribution $p$ with
support of size $K$ and provide both upper and lower bounds with high
probability in KL divergence. We prove that in the worst case, for any
estimator $\widehat{p}$, with probability at least $\delta$, $\text{KL}(p \|
\widehat{p}) \geq C\max\{K,\ln(K)\ln(1/\delta) \}/n $, where $n$ is the sample
size and $C > 0$ is a constant. We introduce a computationally efficient
estimator $p^{\text{OTB}}$, based on Online to Batch conversion and suffix
averaging, and show that with probability at least $1 - \delta$ $\text{KL}(p \|
\widehat{p}) \leq C(K\log(\log(K)) + \ln(K)\ln(1/\delta)) /n$.
  Furthermore, we also show that with sufficiently many observations relative
to $\log(1/\delta)$, the maximum likelihood estimator $\bar{p}$ guarantees that
with probability at least $1-\delta$ $$
  1/6 \chi^2(\bar{p}\|p) \leq 1/4 \chi^2(p\|\bar{p}) \leq \text{KL}(p|\bar{p})
\leq C(K + \log(1/\delta))/n\,, $$ where $\chi^2$ denotes the
$\chi^2$-divergence.

### 2. [Physics-informed, boundary-constrained Gaussian process regression for the reconstruction of fluid flow fields](http://arxiv.org/pdf/2507.17582v1)

Authors: Adrian Padilla-Segarra, Pascal Noble, Olivier Roustant, Éric Savin

Gaussian process regression techniques have been used in fluid mechanics for
the reconstruction of flow fields from a reduction-of-dimension perspective. A
main ingredient in this setting is the construction of adapted covariance
functions, or kernels, to obtain such estimates. In this paper, we derive
physics-informed kernels for simulating two-dimensional velocity fields of an
incompressible (divergence-free) flow around aerodynamic profiles. These
kernels allow to define Gaussian process priors satisfying the
incompressibility condition and the prescribed boundary conditions along the
profile in a continuous manner. Such physical and boundary constraints can be
applied to any pre-defined scalar kernel in the proposed methodology, which is
very general and can be implemented with high flexibility for a broad range of
engineering applications. Its relevance and performances are illustrated by
numerical simulations of flows around a cylinder and a NACA 0412 airfoil
profile, for which no observation at the boundary is needed at all.

### 3. [Debiased maximum-likelihood estimators for hazard ratios under machine-learning adjustment](http://arxiv.org/pdf/2507.17686v1)

Authors: Takashi Hayakawa, Satoshi Asai

Previous studies have shown that hazard ratios between treatment groups
estimated with the Cox model are uninterpretable because the indefinite
baseline hazard of the model fails to identify temporal change in the risk set
composition due to treatment assignment and unobserved factors among multiple,
contradictory scenarios. To alleviate this problem, especially in studies based
on observational data with uncontrolled dynamic treatment and real-time
measurement of many covariates, we propose abandoning the baseline hazard and
using machine learning to explicitly model the change in the risk set with or
without latent variables. For this framework, we clarify the context in which
hazard ratios can be causally interpreted, and then develop a method based on
Neyman orthogonality to compute debiased maximum-likelihood estimators of
hazard ratios. Computing the constructed estimators is more efficient than
computing those based on weighted regression with marginal structural Cox
models. Numerical simulations confirm that the proposed method identifies the
ground truth with minimal bias. These results lay the foundation for developing
a useful, alternative method for causal inference with uncontrolled,
observational data in modern epidemiology.

### 4. [Sequential Bayesian Design for Efficient Surrogate Construction in the Inversion of Darcy Flows](http://arxiv.org/pdf/2507.17713v1)

Authors: Hongji Wang, Hongqiao Wang, Jinyong Ying, Qingping Zhou

Inverse problems governed by partial differential equations (PDEs) play a
crucial role in various fields, including computational science, image
processing, and engineering. Particularly, Darcy flow equation is a fundamental
equation in fluid mechanics, which plays a crucial role in understanding fluid
flow through porous media. Bayesian methods provide an effective approach for
solving PDEs inverse problems, while their numerical implementation requires
numerous evaluations of computationally expensive forward solvers. Therefore,
the adoption of surrogate models with lower computational costs is essential.
However, constructing a globally accurate surrogate model for high-dimensional
complex problems demands high model capacity and large amounts of data. To
address this challenge, this study proposes an efficient locally accurate
surrogate that focuses on the high-probability regions of the true likelihood
in inverse problems, with relatively low model complexity and few training data
requirements. Additionally, we introduce a sequential Bayesian design strategy
to acquire the proposed surrogate since the high-probability region of the
likelihood is unknown. The strategy treats the posterior evolution process of
sequential Bayesian design as a Gaussian process, enabling algorithmic
acceleration through one-step ahead prior. The complete algorithmic framework
is referred to as Sequential Bayesian design for locally accurate surrogate
(SBD-LAS). Finally, three experiments based the Darcy flow equation demonstrate
the advantages of the proposed method in terms of both inversion accuracy and
computational speed.

### 5. [To Trust or Not to Trust: On Calibration in ML-based Resource Allocation for Wireless Networks](http://arxiv.org/pdf/2507.17494v1)

Authors: Rashika Raina, Nidhi Simmons, David E. Simmons, Michel Daoud Yacoub, Trung Q. Duong

In next-generation communications and networks, machine learning (ML) models
are expected to deliver not only accurate predictions but also well-calibrated
confidence scores that reflect the true likelihood of correct decisions. This
paper studies the calibration performance of an ML-based outage predictor
within a single-user, multi-resource allocation framework. We first establish
key theoretical properties of this system's outage probability (OP) under
perfect calibration. Importantly, we show that as the number of resources
grows, the OP of a perfectly calibrated predictor approaches the expected
output conditioned on it being below the classification threshold. In contrast,
when only one resource is available, the system's OP equals the model's overall
expected output. We then derive the OP conditions for a perfectly calibrated
predictor. These findings guide the choice of the classification threshold to
achieve a desired OP, helping system designers meet specific reliability
requirements. We also demonstrate that post-processing calibration cannot
improve the system's minimum achievable OP, as it does not introduce new
information about future channel states. Additionally, we show that
well-calibrated models are part of a broader class of predictors that
necessarily improve OP. In particular, we establish a monotonicity condition
that the accuracy-confidence function must satisfy for such improvement to
occur. To demonstrate these theoretical properties, we conduct a rigorous
simulation-based analysis using post-processing calibration techniques: Platt
scaling and isotonic regression. As part of this framework, the predictor is
trained using an outage loss function specifically designed for this system.
Furthermore, this analysis is performed on Rayleigh fading channels with
temporal correlation captured by Clarke's 2D model, which accounts for receiver
mobility.

### 6. [Federated Majorize-Minimization: Beyond Parameter Aggregation](http://arxiv.org/pdf/2507.17534v1)

Authors: Aymeric Dieuleveut, Gersende Fort, Mahmoud Hegazy, Hoi-To Wai

This paper proposes a unified approach for designing stochastic optimization
algorithms that robustly scale to the federated learning setting. Our work
studies a class of Majorize-Minimization (MM) problems, which possesses a
linearly parameterized family of majorizing surrogate functions. This framework
encompasses (proximal) gradient-based algorithms for (regularized) smooth
objectives, the Expectation Maximization algorithm, and many problems seen as
variational surrogate MM. We show that our framework motivates a unifying
algorithm called Stochastic Approximation Stochastic Surrogate MM (\SSMM),
which includes previous stochastic MM procedures as special instances. We then
extend \SSMM\ to the federated setting, while taking into consideration common
bottlenecks such as data heterogeneity, partial participation, and
communication constraints; this yields \QSMM. The originality of \QSMM\ is to
learn locally and then aggregate information characterizing the
\textit{surrogate majorizing function}, contrary to classical algorithms which
learn and aggregate the \textit{original parameter}. Finally, to showcase the
flexibility of this methodology beyond our theoretical setting, we use it to
design an algorithm for computing optimal transport maps in the federated
setting.

### 7. [Optimal differentially private kernel learning with random projection](http://arxiv.org/pdf/2507.17544v1)

Authors: Bonwoo Lee, Cheolwoo Park, Jeongyoun Ahn

Differential privacy has become a cornerstone in the development of
privacy-preserving learning algorithms. This work addresses optimizing
differentially private kernel learning within the empirical risk minimization
(ERM) framework. We propose a novel differentially private kernel ERM algorithm
based on random projection in the reproducing kernel Hilbert space using
Gaussian processes. Our method achieves minimax-optimal excess risk for both
the squared loss and Lipschitz-smooth convex loss functions under a local
strong convexity condition. We further show that existing approaches based on
alternative dimension reduction techniques, such as random Fourier feature
mappings or $\ell_2$ regularization, yield suboptimal generalization
performance. Our key theoretical contribution also includes the derivation of
dimension-free generalization bounds for objective perturbation-based private
linear ERM -- marking the first such result that does not rely on noisy
gradient-based mechanisms. Additionally, we obtain sharper generalization
bounds for existing differentially private kernel ERM algorithms. Empirical
evaluations support our theoretical claims, demonstrating that random
projection enables statistically efficient and optimally private kernel
learning. These findings provide new insights into the design of differentially
private algorithms and highlight the central role of dimension reduction in
balancing privacy and utility.

### 8. [Generalized Dual Discriminator GANs](http://arxiv.org/pdf/2507.17684v1)

Authors: Penukonda Naga Chandana, Tejas Srivastava, Gowtham R. Kurri, V. Lalitha

Dual discriminator generative adversarial networks (D2 GANs) were introduced
to mitigate the problem of mode collapse in generative adversarial networks. In
D2 GANs, two discriminators are employed alongside a generator: one
discriminator rewards high scores for samples from the true data distribution,
while the other favors samples from the generator. In this work, we first
introduce dual discriminator $\alpha$-GANs (D2 $\alpha$-GANs), which combines
the strengths of dual discriminators with the flexibility of a tunable loss
function, $\alpha$-loss. We further generalize this approach to arbitrary
functions defined on positive reals, leading to a broader class of models we
refer to as generalized dual discriminator generative adversarial networks. For
each of these proposed models, we provide theoretical analysis and show that
the associated min-max optimization reduces to the minimization of a linear
combination of an $f$-divergence and a reverse $f$-divergence. This generalizes
the known simplification for D2-GANs, where the objective reduces to a linear
combination of the KL-divergence and the reverse KL-divergence. Finally, we
perform experiments on 2D synthetic data and use multiple performance metrics
to capture various advantages of our GANs.

### 9. [On the Interaction of Compressibility and Adversarial Robustness](http://arxiv.org/pdf/2507.17725v1)

Authors: Melih Barsbey, Antônio H. Ribeiro, Umut Şimşekli, Tolga Birdal

Modern neural networks are expected to simultaneously satisfy a host of
desirable properties: accurate fitting to training data, generalization to
unseen inputs, parameter and computational efficiency, and robustness to
adversarial perturbations. While compressibility and robustness have each been
studied extensively, a unified understanding of their interaction still remains
elusive. In this work, we develop a principled framework to analyze how
different forms of compressibility - such as neuron-level sparsity and spectral
compressibility - affect adversarial robustness. We show that these forms of
compression can induce a small number of highly sensitive directions in the
representation space, which adversaries can exploit to construct effective
perturbations. Our analysis yields a simple yet instructive robustness bound,
revealing how neuron and spectral compressibility impact $L_\infty$ and $L_2$
robustness via their effects on the learned representations. Crucially, the
vulnerabilities we identify arise irrespective of how compression is achieved -
whether via regularization, architectural bias, or implicit learning dynamics.
Through empirical evaluations across synthetic and realistic tasks, we confirm
our theoretical predictions, and further demonstrate that these vulnerabilities
persist under adversarial training and transfer learning, and contribute to the
emergence of universal adversarial perturbations. Our findings show a
fundamental tension between structured compressibility and robustness, and
suggest new pathways for designing models that are both efficient and secure.

### 10. [Large Learning Rates Simultaneously Achieve Robustness to Spurious Correlations and Compressibility](http://arxiv.org/pdf/2507.17748v1)

Authors: Melih Barsbey, Lucas Prieto, Stefanos Zafeiriou, Tolga Birdal

Robustness and resource-efficiency are two highly desirable properties for
modern machine learning models. However, achieving them jointly remains a
challenge. In this paper, we position high learning rates as a facilitator for
simultaneously achieving robustness to spurious correlations and network
compressibility. We demonstrate that large learning rates also produce
desirable representation properties such as invariant feature utilization,
class separation, and activation sparsity. Importantly, our findings indicate
that large learning rates compare favorably to other hyperparameters and
regularization methods, in consistently satisfying these properties in tandem.
In addition to demonstrating the positive effect of large learning rates across
diverse spurious correlation datasets, models, and optimizers, we also present
strong evidence that the previously documented success of large learning rates
in standard classification tasks is likely due to its effect on addressing
hidden/rare spurious correlations in the training dataset.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-24 PST.

### 1. [Research of UAV 3D path planning based on improved Dwarf mongoose algorithm with multiple strategies](https://www.nature.com/articles/s41598-025-11492-y)

Authors: Lixin Mu et al.

### 2. [Machine learning-based academic performance prediction with explainability for enhanced decision-making in educational institutions](https://www.nature.com/articles/s41598-025-12353-4)

Authors: Wesam Ahmed et al.

### 3. [Scalable 3D reconstruction for X-ray single particle imaging with online machine learning](https://www.nature.com/articles/s41467-025-62226-7)

Authors: Jay Shenoy et al.

### 4. [Decoding student cognitive abilities: a comparative study of explainable AI algorithms in educational data mining](https://www.nature.com/articles/s41598-025-12514-5)

Authors: Tianyue Niu et al.

### 5. [A novel transcendental metaphor metaheuristic algorithm based on power method](https://www.nature.com/articles/s41598-025-12307-w)

Authors: Huiying Zhang et al.

### 6. [Disease probability-enhanced follow-up chest X-ray radiology report summary generation](https://www.nature.com/articles/s41598-025-12684-2)

Authors: Zhichuan Wang et al.

### 7. [Transfer learning with XAI for robust malware and IoT network security](https://www.nature.com/articles/s41598-025-12404-w)

Authors: Ahmad Almadhor et al.

