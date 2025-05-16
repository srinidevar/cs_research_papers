# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-05-15 18:04:03.244262 PST.

### Artificial Intelligence

### 1. [Reproducibility Study of "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents"](http://arxiv.org/pdf/2505.09289v1)

Authors: Pedro M. P. Curvo, Mara Dragomir, Salvador Torpes, Mohammadmahdi Rahimi

This study evaluates and extends the findings made by Piatti et al., who
introduced GovSim, a simulation framework designed to assess the cooperative
decision-making capabilities of large language models (LLMs) in
resource-sharing scenarios. By replicating key experiments, we validate claims
regarding the performance of large models, such as GPT-4-turbo, compared to
smaller models. The impact of the universalization principle is also examined,
with results showing that large models can achieve sustainable cooperation,
with or without the principle, while smaller models fail without it. In
addition, we provide multiple extensions to explore the applicability of the
framework to new settings. We evaluate additional models, such as DeepSeek-V3
and GPT-4o-mini, to test whether cooperative behavior generalizes across
different architectures and model sizes. Furthermore, we introduce new
settings: we create a heterogeneous multi-agent environment, study a scenario
using Japanese instructions, and explore an "inverse environment" where agents
must cooperate to mitigate harmful resource distributions. Our results confirm
that the benchmark can be applied to new models, scenarios, and languages,
offering valuable insights into the adaptability of LLMs in complex cooperative
tasks. Moreover, the experiment involving heterogeneous multi-agent systems
demonstrates that high-performing models can influence lower-performing ones to
adopt similar behaviors. This finding has significant implications for other
agent-based applications, potentially enabling more efficient use of
computational resources and contributing to the development of more effective
cooperative AI systems.

### 2. [Access Controls Will Solve the Dual-Use Dilemma](http://arxiv.org/pdf/2505.09341v1)

Authors: Evžen Wybitul

AI safety systems face a dual-use dilemma. Since the same request can be
either harmless or harmful depending on who made it and why, if the system
makes decisions based solely on the request's content, it will refuse some
legitimate queries and let pass harmful ones. To address this, we propose a
conceptual access control framework, based on verified user credentials (such
as institutional affiliation) and classifiers that assign model outputs to risk
categories (such as advanced virology). The system permits responses only when
the user's verified credentials match the category's requirements. For
implementation of the model output classifiers, we introduce a theoretical
approach utilizing small, gated expert modules integrated into the generator
model, trained with gradient routing, that enable efficient risk detection
without the capability gap problems of external monitors. While open questions
remain about the verification mechanisms, risk categories, and the technical
implementation, our framework makes the first step toward enabling granular
governance of AI capabilities: verified users gain access to specialized
knowledge without arbitrary restrictions, while adversaries are blocked from
it. This contextual approach reconciles model utility with robust safety,
addressing the dual-use dilemma.

### 3. [Counterfactual Strategies for Markov Decision Processes](http://arxiv.org/pdf/2505.09412v1)

Authors: Paul Kobialka, Lina Gerlach, Francesco Leofante, Erika Ábrahám, Silvia Lizeth Tapia Tarifa, Einar Broch Johnsen

Counterfactuals are widely used in AI to explain how minimal changes to a
model's input can lead to a different output. However, established methods for
computing counterfactuals typically focus on one-step decision-making, and are
not directly applicable to sequential decision-making tasks. This paper fills
this gap by introducing counterfactual strategies for Markov Decision Processes
(MDPs). During MDP execution, a strategy decides which of the enabled actions
(with known probabilistic effects) to execute next. Given an initial strategy
that reaches an undesired outcome with a probability above some limit, we
identify minimal changes to the initial strategy to reduce that probability
below the limit. We encode such counterfactual strategies as solutions to
non-linear optimization problems, and further extend our encoding to synthesize
diverse counterfactual strategies. We evaluate our approach on four real-world
datasets and demonstrate its practical viability in sophisticated sequential
decision-making tasks.

### 4. [CEC-Zero: Chinese Error Correction Solution Based on LLM](http://arxiv.org/pdf/2505.09082v1)

Authors: Sophie Zhang, Zhiming Lin

Recent advancements in large language models (LLMs) demonstrate exceptional
Chinese text processing capabilities, particularly in Chinese Spelling
Correction (CSC). While LLMs outperform traditional BERT-based models in
accuracy and robustness, challenges persist in reliability and generalization.
This paper proposes CEC-Zero, a novel reinforcement learning (RL) framework
enabling LLMs to self-correct through autonomous error strategy learning
without external supervision. By integrating RL with LLMs' generative power,
the method eliminates dependency on annotated data or auxiliary models.
Experiments reveal RL-enhanced LLMs achieve industry-viable accuracy and
superior cross-domain generalization, offering a scalable solution for
reliability optimization in Chinese NLP applications. This breakthrough
facilitates LLM deployment in practical Chinese text correction scenarios while
establishing a new paradigm for self-improving language models.

### 5. [Human-like Cognitive Generalization for Large Models via Brain-in-the-loop Supervision](http://arxiv.org/pdf/2505.09085v1)

Authors: Jiaxuan Chen, Yu Qi, Yueming Wang, Gang Pan

Recent advancements in deep neural networks (DNNs), particularly large-scale
language models, have demonstrated remarkable capabilities in image and natural
language understanding. Although scaling up model parameters with increasing
volume of training data has progressively improved DNN capabilities, achieving
complex cognitive abilities - such as understanding abstract concepts,
reasoning, and adapting to novel scenarios, which are intrinsic to human
cognition - remains a major challenge. In this study, we show that
brain-in-the-loop supervised learning, utilizing a small set of brain signals,
can effectively transfer human conceptual structures to DNNs, significantly
enhancing their comprehension of abstract and even unseen concepts.
Experimental results further indicate that the enhanced cognitive capabilities
lead to substantial performance gains in challenging tasks, including
few-shot/zero-shot learning and out-of-distribution recognition, while also
yielding highly interpretable concept representations. These findings highlight
that human-in-the-loop supervision can effectively augment the complex
cognitive abilities of large models, offering a promising pathway toward
developing more human-like cognitive abilities in artificial systems.

### 6. [Air-Ground Collaboration for Language-Specified Missions in Unknown Environments](http://arxiv.org/pdf/2505.09108v1)

Authors: Fernando Cladera, Zachary Ravichandran, Jason Hughes, Varun Murali, Carlos Nieto-Granda, M. Ani Hsieh, George J. Pappas, Camillo J. Taylor, Vijay Kumar

As autonomous robotic systems become increasingly mature, users will want to
specify missions at the level of intent rather than in low-level detail.
Language is an expressive and intuitive medium for such mission specification.
However, realizing language-guided robotic teams requires overcoming
significant technical hurdles. Interpreting and realizing language-specified
missions requires advanced semantic reasoning. Successful heterogeneous robots
must effectively coordinate actions and share information across varying
viewpoints. Additionally, communication between robots is typically
intermittent, necessitating robust strategies that leverage communication
opportunities to maintain coordination and achieve mission objectives. In this
work, we present a first-of-its-kind system where an unmanned aerial vehicle
(UAV) and an unmanned ground vehicle (UGV) are able to collaboratively
accomplish missions specified in natural language while reacting to changes in
specification on the fly. We leverage a Large Language Model (LLM)-enabled
planner to reason over semantic-metric maps that are built online and
opportunistically shared between an aerial and a ground robot. We consider
task-driven navigation in urban and rural areas. Our system must infer
mission-relevant semantics and actively acquire information via semantic
mapping. In both ground and air-ground teaming experiments, we demonstrate our
system on seven different natural-language specifications at up to
kilometer-scale navigation.

### 7. [Beyond the Known: Decision Making with Counterfactual Reasoning Decision Transformer](http://arxiv.org/pdf/2505.09114v1)

Authors: Minh Hoang Nguyen, Linh Le Pham Van, Thommen George Karimpanal, Sunil Gupta, Hung Le

Decision Transformers (DT) play a crucial role in modern reinforcement
learning, leveraging offline datasets to achieve impressive results across
various domains. However, DT requires high-quality, comprehensive data to
perform optimally. In real-world applications, the lack of training data and
the scarcity of optimal behaviours make training on offline datasets
challenging, as suboptimal data can hinder performance. To address this, we
propose the Counterfactual Reasoning Decision Transformer (CRDT), a novel
framework inspired by counterfactual reasoning. CRDT enhances DT ability to
reason beyond known data by generating and utilizing counterfactual
experiences, enabling improved decision-making in unseen scenarios. Experiments
across Atari and D4RL benchmarks, including scenarios with limited data and
altered dynamics, demonstrate that CRDT outperforms conventional DT approaches.
Additionally, reasoning counterfactually allows the DT agent to obtain
stitching abilities, combining suboptimal trajectories, without architectural
modifications. These results highlight the potential of counterfactual
reasoning to enhance reinforcement learning agents' performance and
generalization capabilities.

### 8. [PreCare: Designing AI Assistants for Advance Care Planning (ACP) to Enhance Personal Value Exploration, Patient Knowledge, and Decisional Confidence](http://arxiv.org/pdf/2505.09115v1)

Authors: Yu Lun Hsu, Yun-Rung Chou, Chiao-Ju Chang, Yu-Cheng Chang, Zer-Wei Lee, Rokas Gipiškis, Rachel Li, Chih-Yuan Shih, Jen-Kuei Peng, Hsien-Liang Huang, Jaw-Shiun Tsai, Mike Y. Chen

Advance Care Planning (ACP) allows individuals to specify their preferred
end-of-life life-sustaining treatments before they become incapacitated by
injury or terminal illness (e.g., coma, cancer, dementia). While online ACP
offers high accessibility, it lacks key benefits of clinical consultations,
including personalized value exploration, immediate clarification of decision
consequences. To bridge this gap, we conducted two formative studies: 1)
shadowed and interviewed 3 ACP teams consisting of physicians, nurses, and
social workers (18 patients total), and 2) interviewed 14 users of ACP
websites. Building on these insights, we designed PreCare in collaboration with
6 ACP professionals. PreCare is a website with 3 AI-driven assistants designed
to guide users through exploring personal values, gaining ACP knowledge, and
supporting informed decision-making. A usability study (n=12) showed that
PreCare achieved a System Usability Scale (SUS) rating of excellent. A
comparative evaluation (n=12) showed that PreCare's AI assistants significantly
improved exploration of personal values, knowledge, and decisional confidence,
and was preferred by 92% of participants.

### 9. [WSCIF: A Weakly-Supervised Color Intelligence Framework for Tactical Anomaly Detection in Surveillance Keyframes](http://arxiv.org/pdf/2505.09129v1)

Authors: Wei Meng

The deployment of traditional deep learning models in high-risk security
tasks in an unlabeled, data-non-exploitable video intelligence environment
faces significant challenges. In this paper, we propose a lightweight anomaly
detection framework based on color features for surveillance video clips in a
high sensitivity tactical mission, aiming to quickly identify and interpret
potential threat events under resource-constrained and data-sensitive
conditions. The method fuses unsupervised KMeans clustering with RGB channel
histogram modeling to achieve composite detection of structural anomalies and
color mutation signals in key frames. The experiment takes an operation
surveillance video occurring in an African country as a research sample, and
successfully identifies multiple highly anomalous frames related to high-energy
light sources, target presence, and reflective interference under the condition
of no access to the original data. The results show that this method can be
effectively used for tactical assassination warning, suspicious object
screening and environmental drastic change monitoring with strong deployability
and tactical interpretation value. The study emphasizes the importance of color
features as low semantic battlefield signal carriers, and its battlefield
intelligent perception capability will be further extended by combining graph
neural networks and temporal modeling in the future.

### 10. [An Initial Exploration of Default Images in Text-to-Image Generation](http://arxiv.org/pdf/2505.09166v1)

Authors: Hannu Simonen, Atte Kiviniemi, Jonas Oppenlaender

In the creative practice of text-to-image generation (TTI), images are
generated from text prompts. However, TTI models are trained to always yield an
output, even if the prompt contains unknown terms. In this case, the model may
generate what we call "default images": images that closely resemble each other
across many unrelated prompts. We argue studying default images is valuable for
designing better solutions for TTI and prompt engineering. In this paper, we
provide the first investigation into default images on Midjourney, a popular
image generator. We describe our systematic approach to create input prompts
triggering default images, and present the results of our initial experiments
and several small-scale ablation studies. We also report on a survey study
investigating how default images affect user satisfaction. Our work lays the
foundation for understanding default images in TTI and highlights challenges
and future research directions.

### Hardware Architecture

### 1. [SEGA-DCIM: Design Space Exploration-Guided Automatic Digital CIM Compiler with Multiple Precision Support](http://arxiv.org/pdf/2505.09451v1)

Authors: Haikang Diao, Haoyi Zhang, Jiahao Song, Haoyang Luo, Yibo Lin, Runsheng Wang, Yuan Wang, Xiyuan Tang

Digital computing-in-memory (DCIM) has been a popular solution for addressing
the memory wall problem in recent years. However, the DCIM design still heavily
relies on manual efforts, and the optimization of DCIM is often based on human
experience. These disadvantages limit the time to market while increasing the
design difficulty of DCIMs. This work proposes a design space
exploration-guided automatic DCIM compiler (SEGA-DCIM) with multiple precision
support, including integer and floating-point data precision operations.
SEGA-DCIM can automatically generate netlists and layouts of DCIM designs by
leveraging a template-based method. With a multi-objective genetic algorithm
(MOGA)-based design space explorer, SEGA-DCIM can easily select appropriate
DCIM designs for a specific application considering the trade-offs among area,
power, and delay. As demonstrated by the experimental results, SEGA-DCIM offers
solutions with wide design space, including integer and floating-point
precision designs, while maintaining competitive performance compared to
state-of-the-art (SOTA) DCIMs.

### 2. [Automated SAR ADC Sizing Using Analytical Equations](http://arxiv.org/pdf/2505.09172v1)

Authors: Zhongyi Li, Zhuofu Tao, Yanze Zhou, Yichen Shi, Zhiping Yu, Ting-Jung Lin, Lei He

Conventional analog and mixed-signal (AMS) circuit designs heavily rely on
manual effort, which is time-consuming and labor-intensive. This paper presents
a fully automated design methodology for Successive Approximation Register
(SAR) Analog-to-Digital Converters (ADCs) from performance specifications to
complete transistor sizing. To tackle the high-dimensional sizing problem, we
propose a dual optimization scheme. The system-level optimization iteratively
partitions the overall requirements and analytically maps them to subcircuit
design specifications, while local optimization loops determines the
subcircuits' design parameters. The dependency graph-based framework serializes
the simulations for verification, knowledge-based calculations, and transistor
sizing optimization in topological order, which eliminates the need for human
intervention. We demonstrate the effectiveness of the proposed methodology
through two case studies with varying performance specifications, achieving
high SNDR and low power consumption while meeting all the specified design
constraints.

### 3. [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](http://arxiv.org/pdf/2505.09343v1)

Authors: Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Huazuo Gao, Jiashi Li, Liyue Zhang, Panpan Huang, Shangyan Zhou, Shirong Ma, Wenfeng Liang, Ying He, Yuqing Wang, Yuxuan Liu, Y. X. Wei

The rapid scaling of large language models (LLMs) has unveiled critical
limitations in current hardware architectures, including constraints in memory
capacity, computational efficiency, and interconnection bandwidth. DeepSeek-V3,
trained on 2,048 NVIDIA H800 GPUs, demonstrates how hardware-aware model
co-design can effectively address these challenges, enabling cost-efficient
training and inference at scale. This paper presents an in-depth analysis of
the DeepSeek-V3/R1 model architecture and its AI infrastructure, highlighting
key innovations such as Multi-head Latent Attention (MLA) for enhanced memory
efficiency, Mixture of Experts (MoE) architectures for optimized
computation-communication trade-offs, FP8 mixed-precision training to unlock
the full potential of hardware capabilities, and a Multi-Plane Network Topology
to minimize cluster-level network overhead. Building on the hardware
bottlenecks encountered during DeepSeek-V3's development, we engage in a
broader discussion with academic and industry peers on potential future
hardware directions, including precise low-precision computation units,
scale-up and scale-out convergence, and innovations in low-latency
communication fabrics. These insights underscore the critical role of hardware
and model co-design in meeting the escalating demands of AI workloads, offering
a practical blueprint for innovation in next-generation AI systems.

### Computational Complexity

### 1. [BusOut is NP-complete](http://arxiv.org/pdf/2505.09165v1)

Authors: Takehiro Ishibashi, Ryo Yoshinaka, Ayumi Shinohara

This study examines the computational complexity of the decision problem
modeled on the smartphone game Bus Out. The objective of the game is to load
all the passengers in a queue onto appropriate buses using a limited number of
bus parking spots by selecting and dispatching the buses on a map. We show that
the problem is NP-complete, even for highly restricted instances. We also show
that it is hard to approximate the minimum number of parking spots needed to
solve a given instance.

### 2. [Phase Transitions in Decision Problems Over Odd-Sized Alphabets](http://arxiv.org/pdf/2505.09282v1)

Authors: Andrew Jackson

In [A. Jackson, Explaining the ubiquity of phase transitions in decision
problems (2025), arXiv:2501.14569], I established that phase transitions are
always present in a large subset of decision problems over even-sized
alphabets, explaining -- in part -- why phase transitions are seen so often in
decision problems. However, decision problems over odd-sized alphabets were not
discussed. Here, I correct that oversight, showing that a similar subset of
decision problems over odd-sized alphabets also always exhibit phase
transitions.

### 3. [The Adaptive Complexity of Finding a Stationary Point](http://arxiv.org/pdf/2505.09045v1)

Authors: Huanjian Zhou, Andi Han, Akiko Takeda, Masashi Sugiyama

In large-scale applications, such as machine learning, it is desirable to
design non-convex optimization algorithms with a high degree of
parallelization. In this work, we study the adaptive complexity of finding a
stationary point, which is the minimal number of sequential rounds required to
achieve stationarity given polynomially many queries executed in parallel at
each round.
  For the high-dimensional case, i.e., $d = \widetilde{\Omega}(\varepsilon^{-(2
+ 2p)/p})$, we show that for any (potentially randomized) algorithm, there
exists a function with Lipschitz $p$-th order derivatives such that the
algorithm requires at least $\varepsilon^{-(p+1)/p}$ iterations to find an
$\varepsilon$-stationary point. Our lower bounds are tight and show that even
with $\mathrm{poly}(d)$ queries per iteration, no algorithm has better
convergence rate than those achievable with one-query-per-round algorithms. In
other words, gradient descent, the cubic-regularized Newton's method, and the
$p$-th order adaptive regularization method are adaptively optimal. Our proof
relies upon novel analysis with the characterization of the output for the
hardness potentials based on a chain-like structure with random partition.
  For the constant-dimensional case, i.e., $d = \Theta(1)$, we propose an
algorithm that bridges grid search and gradient flow trapping, finding an
approximate stationary point in constant iterations. Its asymptotic tightness
is verified by a new lower bound on the required queries per iteration. We show
there exists a smooth function such that any algorithm running with
$\Theta(\log (1/\varepsilon))$ rounds requires at least
$\widetilde{\Omega}((1/\varepsilon)^{(d-1)/2})$ queries per round. This lower
bound is tight up to a logarithmic factor, and implies that the gradient flow
trapping is adaptively optimal.

### Computational Engineering

### 1. [Optimization of the initial post-buckling response of trusses and frames by an asymptotic approach](http://arxiv.org/pdf/2505.09373v1)

Authors: Federico Ferrari, Ole Sigmund

Asymptotic post-buckling theory is applied to sizing and topology
optimization of trusses and frames, exploring its potential and current
computational difficulties. We show that a designs' post-buckling response can
be controlled by including the lowest two asymptotic coefficients, representing
the initial post-buckling slope and curvature, in the optimization formulation.
This also reduces the imperfection sensitivity of the optimized design. The
asymptotic expansion can further be used to approximate the structural
nonlinear response, and then to optimize for a given measure of the nonlinear
mechanical performance such as, for example, end-compliance or complementary
work. Examples of linear and nonlinear compliance minimization of trusses and
frames show the effective use of the asymptotic method for including
post-buckling constraints in structural optimization.

### 2. [Radon Exposure Dataset](http://arxiv.org/pdf/2505.09489v1)

Authors: Dakotah Maguire, Jeremy Logan, Heechan Lee, Heidi Hanson

Exposure to elevated radon levels in the home is one of the leading causes of
lung cancer in the world. The following study describes the creation of a
comprehensive, state-level dataset designed to enable the modeling and
prediction of household radon concentrations at Zip Code Tabulation Area (ZCTA)
and sub-kilometer scales. Details include the data collection and processing
involved in compiling physical and demographic factors for Pennsylvania and
Utah. Attempting to mitigate this risk requires identifying the underlying
geological causes and the populations that might be at risk. This work focuses
on identifying at-risk populations throughout Pennsylvania and Utah, where
radon levels are some of the highest in the country. The resulting dataset
harmonizes geological and demographic factors from various sources and spatial
resolutions, including temperature, geochemistry, and soil characteristics.
Demographic variables such as the household heating fuel used, the age of
building, and the housing type provide further insight into which populations
could be most susceptible in areas with potentially high radon levels. This
dataset also serves as a foundational resource for two other studies conducted
by the authors. The resolution of the data provides a novel approach to
predicting potential radon exposure, and the data processing conducted for
these states can be scaled up to larger spatial resolutions (e.g., the
Contiguous United States [CONUS]) and allow for a broad reclassification of
radon exposure potential in the United States.

### Computational Geometry

### 1. [Approximating the Directed Hausdorff Distance](http://arxiv.org/pdf/2505.09046v2)

Authors: Oliver A. Chubet, Parth M. Parikh, Donald R. Sheehy, Siddharth S. Sheth

The Hausdorff distance is a metric commonly used to compute the set
similarity of geometric sets.
  For sets containing a total of $n$ points, the exact distance can be computed
na\"{i}vely in $O(n^2)$ time.
  In this paper, we show how to preprocess point sets individually so that the
Hausdorff distance of any pair can then be approximated in linear time.
  We assume that the metric is doubling.
  The preprocessing time for each set is $O(n\log \Delta)$ where $\Delta$ is
the ratio of the largest to smallest pairwise distances of the input.
  In theory, this can be reduced to $O(n\log n)$ time using a much more
complicated algorithm.
  We compute $(1+\varepsilon)$-approximate Hausdorff distance in $(2 +
\frac{1}{\varepsilon})^{O(d)}n$ time in a metric space with doubling dimension
$d$.
  The $k$-partial Hausdorff distance ignores $k$ outliers to increase
stability.
  Additionally, we give a linear-time algorithm to compute directed $k$-partial
Hausdorff distance for all values of $k$ at once with no change to the
preprocessing.

### Computation and Language

### 1. [Atomic Consistency Preference Optimization for Long-Form Question Answering](http://arxiv.org/pdf/2505.09039v1)

Authors: Jingfeng Chen, Raghuveer Thirukovalluru, Junlin Wang, Kaiwei Luo, Bhuwan Dhingra

Large Language Models (LLMs) frequently produce factoid hallucinations -
plausible yet incorrect answers. A common mitigation strategy is model
alignment, which improves factual accuracy by training on curated factual and
non-factual pairs. However, this approach often relies on a stronger model
(e.g., GPT-4) or an external knowledge base to assess factual correctness,
which may not always be accessible. To address this, we propose Atomic
Consistency Preference Optimization (ACPO), a self-supervised preference-tuning
method that enhances factual accuracy without external supervision. ACPO
leverages atomic consistency signals, i.e., the agreement of individual facts
across multiple stochastic responses, to identify high- and low-quality data
pairs for model alignment. By eliminating the need for costly GPT calls, ACPO
provides a scalable and efficient approach to improving factoid
question-answering. Despite being self-supervised, empirical results
demonstrate that ACPO outperforms FactAlign, a strong supervised alignment
baseline, by 1.95 points on the LongFact and BioGen datasets, highlighting its
effectiveness in enhancing factual reliability without relying on external
models or knowledge bases.

### 2. [A Comprehensive Analysis of Large Language Model Outputs: Similarity, Diversity, and Bias](http://arxiv.org/pdf/2505.09056v1)

Authors: Brandon Smith, Mohamed Reda Bouadjenek, Tahsin Alamgir Kheya, Phillip Dawson, Sunil Aryal

Large Language Models (LLMs) represent a major step toward artificial general
intelligence, significantly advancing our ability to interact with technology.
While LLMs perform well on Natural Language Processing tasks -- such as
translation, generation, code writing, and summarization -- questions remain
about their output similarity, variability, and ethical implications. For
instance, how similar are texts generated by the same model? How does this
compare across different models? And which models best uphold ethical
standards? To investigate, we used 5{,}000 prompts spanning diverse tasks like
generation, explanation, and rewriting. This resulted in approximately 3
million texts from 12 LLMs, including proprietary and open-source systems from
OpenAI, Google, Microsoft, Meta, and Mistral. Key findings include: (1) outputs
from the same LLM are more similar to each other than to human-written texts;
(2) models like WizardLM-2-8x22b generate highly similar outputs, while GPT-4
produces more varied responses; (3) LLM writing styles differ significantly,
with Llama 3 and Mistral showing higher similarity, and GPT-4 standing out for
distinctiveness; (4) differences in vocabulary and tone underscore the
linguistic uniqueness of LLM-generated content; (5) some LLMs demonstrate
greater gender balance and reduced bias. These results offer new insights into
the behavior and diversity of LLM outputs, helping guide future development and
ethical evaluation.

### 3. [How an unintended Side Effect of a Research Project led to Boosting the Power of UML](http://arxiv.org/pdf/2505.09269v1)

Authors: Ulrich Frank, Pierre Maier

This paper describes the design, implementation and use of a new UML modeling
tool that represents a significant advance over conventional tools. Among other
things, it allows the integration of class diagrams and object diagrams as well
as the execution of objects. This not only enables new software architectures
characterized by the integration of software with corresponding object models,
but is also ideal for use in teaching, as it provides students with a
particularly stimulating learning experience. A special feature of the project
is that it has emerged from a long-standing international research project,
which is aimed at a comprehensive multi-level architecture. The project is
therefore an example of how research can lead to valuable results that arise as
a side effect of other work.

### 4. [A Scalable Unsupervised Framework for multi-aspect labeling of Multilingual and Multi-Domain Review Data](http://arxiv.org/pdf/2505.09286v1)

Authors: Jiin Park, Misuk Kim

Effectively analyzing online review data is essential across industries.
However, many existing studies are limited to specific domains and languages or
depend on supervised learning approaches that require large-scale labeled
datasets. To address these limitations, we propose a multilingual, scalable,
and unsupervised framework for cross-domain aspect detection. This framework is
designed for multi-aspect labeling of multilingual and multi-domain review
data. In this study, we apply automatic labeling to Korean and English review
datasets spanning various domains and assess the quality of the generated
labels through extensive experiments. Aspect category candidates are first
extracted through clustering, and each review is then represented as an
aspect-aware embedding vector using negative sampling. To evaluate the
framework, we conduct multi-aspect labeling and fine-tune several pretrained
language models to measure the effectiveness of the automatically generated
labels. Results show that these models achieve high performance, demonstrating
that the labels are suitable for training. Furthermore, comparisons with
publicly available large language models highlight the framework's superior
consistency and scalability when processing large-scale data. A human
evaluation also confirms that the quality of the automatic labels is comparable
to those created manually. This study demonstrates the potential of a robust
multi-aspect labeling approach that overcomes limitations of supervised methods
and is adaptable to multilingual, multi-domain environments. Future research
will explore automatic review summarization and the integration of artificial
intelligence agents to further improve the efficiency and depth of review
analysis.

### 5. [Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs](http://arxiv.org/pdf/2505.09338v1)

Authors: Jingcheng Niu, Xingdi Yuan, Tong Wang, Hamidreza Saghir, Amir H. Abdi

We observe a novel phenomenon, contextual entrainment, across a wide range of
language models (LMs) and prompt settings, providing a new mechanistic
perspective on how LMs become distracted by ``irrelevant'' contextual
information in the input prompt. Specifically, LMs assign significantly higher
logits (or probabilities) to any tokens that have previously appeared in the
context prompt, even for random tokens. This suggests that contextual
entrainment is a mechanistic phenomenon, occurring independently of the
relevance or semantic relation of the tokens to the question or the rest of the
sentence. We find statistically significant evidence that the magnitude of
contextual entrainment is influenced by semantic factors. Counterfactual
prompts have a greater effect compared to factual ones, suggesting that while
contextual entrainment is a mechanistic phenomenon, it is modulated by semantic
factors.
  We hypothesise that there is a circuit of attention heads -- the entrainment
heads -- that corresponds to the contextual entrainment phenomenon. Using a
novel entrainment head discovery method based on differentiable masking, we
identify these heads across various settings. When we ``turn off'' these heads,
i.e., set their outputs to zero, the effect of contextual entrainment is
significantly attenuated, causing the model to generate output that capitulates
to what it would produce if no distracting context were provided. Our discovery
of contextual entrainment, along with our investigation into LM distraction via
the entrainment heads, marks a key step towards the mechanistic analysis and
mitigation of the distraction problem.

### 6. [Qwen3 Technical Report](http://arxiv.org/pdf/2505.09388v1)

Authors: An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, Zihan Qiu

In this work, we present Qwen3, the latest version of the Qwen model family.
Qwen3 comprises a series of large language models (LLMs) designed to advance
performance, efficiency, and multilingual capabilities. The Qwen3 series
includes models of both dense and Mixture-of-Expert (MoE) architectures, with
parameter scales ranging from 0.6 to 235 billion. A key innovation in Qwen3 is
the integration of thinking mode (for complex, multi-step reasoning) and
non-thinking mode (for rapid, context-driven responses) into a unified
framework. This eliminates the need to switch between different models--such as
chat-optimized models (e.g., GPT-4o) and dedicated reasoning models (e.g.,
QwQ-32B)--and enables dynamic mode switching based on user queries or chat
templates. Meanwhile, Qwen3 introduces a thinking budget mechanism, allowing
users to allocate computational resources adaptively during inference, thereby
balancing latency and performance based on task complexity. Moreover, by
leveraging the knowledge from the flagship models, we significantly reduce the
computational resources required to build smaller-scale models, while ensuring
their highly competitive performance. Empirical evaluations demonstrate that
Qwen3 achieves state-of-the-art results across diverse benchmarks, including
tasks in code generation, mathematical reasoning, agent tasks, etc.,
competitive against larger MoE models and proprietary models. Compared to its
predecessor Qwen2.5, Qwen3 expands multilingual support from 29 to 119
languages and dialects, enhancing global accessibility through improved
cross-lingual understanding and generation capabilities. To facilitate
reproducibility and community-driven research and development, all Qwen3 models
are publicly accessible under Apache 2.0.

### 7. [PT-MoE: An Efficient Finetuning Framework for Integrating Mixture-of-Experts into Prompt Tuning](http://arxiv.org/pdf/2505.09519v1)

Authors: Zongqian Li, Yixuan Su, Nigel Collier

Parameter-efficient fine-tuning (PEFT) methods have shown promise in adapting
large language models, yet existing approaches exhibit counter-intuitive
phenomena: integrating router into prompt tuning (PT) increases training
efficiency yet does not improve performance universally; parameter reduction
through matrix decomposition can improve performance in specific domains.
Motivated by these observations and the modular nature of PT, we propose
PT-MoE, a novel framework that integrates matrix decomposition with
mixture-of-experts (MoE) routing for efficient PT. Results across 17 datasets
demonstrate that PT-MoE achieves state-of-the-art performance in both question
answering (QA) and mathematical problem solving tasks, improving F1 score by
1.49 points over PT and 2.13 points over LoRA in QA tasks, while enhancing
mathematical accuracy by 10.75 points over PT and 0.44 points over LoRA, all
while using 25% fewer parameters than LoRA. Our analysis reveals that while PT
methods generally excel in QA tasks and LoRA-based methods in math datasets,
the integration of matrix decomposition and MoE in PT-MoE yields complementary
benefits: decomposition enables efficient parameter sharing across experts
while MoE provides dynamic adaptation, collectively enabling PT-MoE to
demonstrate cross-task consistency and generalization abilities. These
findings, along with ablation studies on routing mechanisms and architectural
components, provide insights for future PEFT methods.

### 8. [S-DAT: A Multilingual, GenAI-Driven Framework for Automated Divergent Thinking Assessment](http://arxiv.org/pdf/2505.09068v1)

Authors: Jennifer Haase, Paul H. P. Hanel, Sebastian Pokutta

This paper introduces S-DAT (Synthetic-Divergent Association Task), a
scalable, multilingual framework for automated assessment of divergent thinking
(DT) -a core component of human creativity. Traditional creativity assessments
are often labor-intensive, language-specific, and reliant on subjective human
ratings, limiting their scalability and cross-cultural applicability. In
contrast, S-DAT leverages large language models and advanced multilingual
embeddings to compute semantic distance -- a language-agnostic proxy for DT. We
evaluate S-DAT across eleven diverse languages, including English, Spanish,
German, Russian, Hindi, and Japanese (Kanji, Hiragana, Katakana), demonstrating
robust and consistent scoring across linguistic contexts. Unlike prior DAT
approaches, the S-DAT shows convergent validity with other DT measures and
correct discriminant validity with convergent thinking. This cross-linguistic
flexibility allows for more inclusive, global-scale creativity research,
addressing key limitations of earlier approaches. S-DAT provides a powerful
tool for fairer, more comprehensive evaluation of cognitive flexibility in
diverse populations and can be freely assessed online:
https://sdat.iol.zib.de/.

### 9. [CEC-Zero: Chinese Error Correction Solution Based on LLM](http://arxiv.org/pdf/2505.09082v1)

Authors: Sophie Zhang, Zhiming Lin

Recent advancements in large language models (LLMs) demonstrate exceptional
Chinese text processing capabilities, particularly in Chinese Spelling
Correction (CSC). While LLMs outperform traditional BERT-based models in
accuracy and robustness, challenges persist in reliability and generalization.
This paper proposes CEC-Zero, a novel reinforcement learning (RL) framework
enabling LLMs to self-correct through autonomous error strategy learning
without external supervision. By integrating RL with LLMs' generative power,
the method eliminates dependency on annotated data or auxiliary models.
Experiments reveal RL-enhanced LLMs achieve industry-viable accuracy and
superior cross-domain generalization, offering a scalable solution for
reliability optimization in Chinese NLP applications. This breakthrough
facilitates LLM deployment in practical Chinese text correction scenarios while
establishing a new paradigm for self-improving language models.

### 10. [Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging](http://arxiv.org/pdf/2505.09316v1)

Authors: Hongjin Qian, Zheng Liu

Augmenting large language models (LLMs) with external retrieval has become a
standard method to address their inherent knowledge cutoff limitations.
However, traditional retrieval-augmented generation methods employ static,
pre-inference retrieval strategies, making them inadequate for complex tasks
involving ambiguous, multi-step, or evolving information needs. Recent advances
in test-time scaling techniques have demonstrated significant potential in
enabling LLMs to dynamically interact with external tools, motivating the shift
toward adaptive inference-time retrieval. Inspired by Information Foraging
Theory (IFT), we propose InForage, a reinforcement learning framework that
formalizes retrieval-augmented reasoning as a dynamic information-seeking
process. Unlike existing approaches, InForage explicitly rewards intermediate
retrieval quality, encouraging LLMs to iteratively gather and integrate
information through adaptive search behaviors. To facilitate training, we
construct a human-guided dataset capturing iterative search and reasoning
trajectories for complex, real-world web tasks. Extensive evaluations across
general question answering, multi-hop reasoning tasks, and a newly developed
real-time web QA dataset demonstrate InForage's superior performance over
baseline methods. These results highlight InForage's effectiveness in building
robust, adaptive, and efficient reasoning agents.

### Cryptography and Security

### 1. [Multiparty Selective Disclosure using Attribute-Based Encryption](http://arxiv.org/pdf/2505.09034v1)

Authors: Shigenori Ohashi

This study proposes a mechanism for encrypting SD-JWT (Selective Disclosure
JSON Web Token) Disclosures using Attribute-Based Encryption (ABE) to enable
flexible access control on the basis of the Verifier's attributes. By
integrating Ciphertext-Policy ABE (CP-ABE) into the existing SD-JWT framework,
the Holder can assign decryption policies to Disclosures, ensuring information
is selectively disclosed. The mechanism's feasibility was evaluated in a
virtualized environment by measuring the processing times for SD-JWT
generation, encryption, and decryption with varying Disclosure counts (5, 10,
20). Results showed that SD-JWT generation is lightweight, while encryption and
decryption times increase linearly with the number of Disclosures. This
approach is suitable for privacy-sensitive applications like healthcare,
finance, and supply chain tracking but requires optimization for real-time use
cases such as IoT. Future research should focus on improving ABE efficiency and
addressing scalability challenges.

### 2. [Unencrypted Flying Objects: Security Lessons from University Small Satellite Developers and Their Code](http://arxiv.org/pdf/2505.09038v1)

Authors: Rachel McAmis, Gregor Haas, Mattea Sim, David Kohlbrenner, Tadayoshi Kohno

Satellites face a multitude of security risks that set them apart from
hardware on Earth. Small satellites may face additional challenges, as they are
often developed on a budget and by amateur organizations or universities that
do not consider security. We explore the security practices and preferences of
small satellite teams, particularly university satellite teams, to understand
what barriers exist to building satellites securely. We interviewed 8
university satellite club leaders across 4 clubs in the U.S. and perform a code
audit of 3 of these clubs' code repositories. We find that security practices
vary widely across teams, but all teams studied had vulnerabilities available
to an unprivileged, ground-based attacker. Participants foresee many risks of
unsecured small satellites and indicate security shortcomings in industry and
government. Lastly, we identify a set of considerations for how to build future
small satellites securely, in amateur organizations and beyond.

### 3. [Modeling Interdependent Cybersecurity Threats Using Bayesian Networks: A Case Study on In-Vehicle Infotainment Systems](http://arxiv.org/pdf/2505.09048v1)

Authors: Sangita Sridar

Cybersecurity threats are increasingly marked by interdependence,
uncertainty, and evolving complexity challenges that traditional assessment
methods such as CVSS, STRIDE, and attack trees fail to adequately capture. This
paper reviews the application of Bayesian Networks (BNs) in cybersecurity risk
modeling, highlighting their capacity to represent probabilistic dependencies,
integrate diverse threat indicators, and support reasoning under uncertainty. A
structured case study is presented in which a STRIDE-based attack tree for an
automotive In-Vehicle Infotainment (IVI) system is transformed into a Bayesian
Network. Logical relationships are encoded using Conditional Probability Tables
(CPTs), and threat likelihoods are derived from normalized DREAD scores. The
model enables not only probabilistic inference of system compromise likelihood
but also supports causal analysis using do-calculus and local sensitivity
analysis to identify high-impact vulnerabilities. These analyses provide
insight into the most influential nodes within the threat propagation chain,
informing targeted mitigation strategies. While demonstrating the potential of
BNs for dynamic and context-aware risk assessment, the study also outlines
limitations related to scalability, reliance on expert input, static structure
assumptions, and limited temporal modeling. The paper concludes by advocating
for future enhancements through Dynamic Bayesian Networks, structure learning,
and adaptive inference to better support real-time cybersecurity
decision-making in complex environments.

### 4. [Securing P4 Programs by Information Flow Control](http://arxiv.org/pdf/2505.09221v1)

Authors: Anoud Alshnakat, Amir M. Ahmadian, Musard Balliu, Roberto Guanciale, Mads Dam

Software-Defined Networking (SDN) has transformed network architectures by
decoupling the control and data-planes, enabling fine-grained control over
packet processing and forwarding. P4, a language designed for programming
data-plane devices, allows developers to define custom packet processing
behaviors directly on programmable network devices. This provides greater
control over packet forwarding, inspection, and modification. However, the
increased flexibility provided by P4 also brings significant security
challenges, particularly in managing sensitive data and preventing information
leakage within the data-plane.
  This paper presents a novel security type system for analyzing information
flow in P4 programs that combines security types with interval analysis. The
proposed type system allows the specification of security policies in terms of
input and output packet bit fields rather than program variables. We formalize
this type system and prove it sound, guaranteeing that well-typed programs
satisfy noninterference. Our prototype implementation, Tap4s, is evaluated on
several use cases, demonstrating its effectiveness in detecting security
violations and information leakages.

### 5. [Instantiating Standards: Enabling Standard-Driven Text TTP Extraction with Evolvable Memory](http://arxiv.org/pdf/2505.09261v1)

Authors: Cheng Meng, ZhengWei Jiang, QiuYun Wang, XinYi Li, ChunYan Ma, FangMing Dong, FangLi Ren, BaoXu Liu

Extracting MITRE ATT\&CK Tactics, Techniques, and Procedures (TTPs) from
natural language threat reports is crucial yet challenging. Existing methods
primarily focus on performance metrics using data-driven approaches, often
neglecting mechanisms to ensure faithful adherence to the official standard.
This deficiency compromises reliability and consistency of TTP assignments,
creating intelligence silos and contradictory threat assessments across
organizations. To address this, we introduce a novel framework that converts
abstract standard definitions into actionable, contextualized knowledge. Our
method utilizes Large Language Model (LLM) to generate, update, and apply this
knowledge. This framework populates an evolvable memory with dual-layer
situational knowledge instances derived from labeled examples and official
definitions. The first layer identifies situational contexts (e.g.,
"Communication with C2 using encoded subdomains"), while the second layer
captures distinctive features that differentiate similar techniques (e.g.,
distinguishing T1132 "Data Encoding" from T1071 "Application Layer Protocol"
based on whether the focus is on encoding methods or protocol usage). This
structured approach provides a transparent basis for explainable TTP
assignments and enhanced human oversight, while also helping to standardize
other TTP extraction systems. Experiments show our framework (using
Qwen2.5-32B) boosts Technique F1 scores by 11\% over GPT-4o. Qualitative
analysis confirms superior standardization, enhanced transparency, and improved
explainability in real-world threat intelligence scenarios. To the best of our
knowledge, this is the first work that uses the LLM to generate, update, and
apply the a new knowledge for TTP extraction.

### 6. [CANTXSec: A Deterministic Intrusion Detection and Prevention System for CAN Bus Monitoring ECU Activations](http://arxiv.org/pdf/2505.09384v1)

Authors: Denis Donadel, Kavya Balasubramanian, Alessandro Brighente, Bhaskar Ramasubramanian, Mauro Conti, Radha Poovendran

Despite being a legacy protocol with various known security issues,
Controller Area Network (CAN) still represents the de-facto standard for
communications within vehicles, ships, and industrial control systems. Many
research works have designed Intrusion Detection Systems (IDSs) to identify
attacks by training machine learning classifiers on bus traffic or its
properties. Actions to take after detection are, on the other hand, less
investigated, and prevention mechanisms usually include protocol modification
(e.g., adding authentication). An effective solution has yet to be implemented
on a large scale in the wild. The reasons are related to the effort to handle
sporadic false positives, the inevitable delay introduced by authentication,
and the closed-source automobile environment that does not easily permit
modifying Electronic Control Units (ECUs) software.
  In this paper, we propose CANTXSec, the first deterministic Intrusion
Detection and Prevention system based on physical ECU activations. It employs a
new classification of attacks based on the attacker's need in terms of access
level to the bus, distinguishing between Frame Injection Attacks (FIAs) (i.e.,
using frame-level access) and Single-Bit Attacks (SBAs) (i.e., employing
bit-level access). CANTXSec detects and prevents classical attacks in the CAN
bus, while detecting advanced attacks that have been less investigated in the
literature. We prove the effectiveness of our solution on a physical testbed,
where we achieve 100% detection accuracy in both classes of attacks while
preventing 100% of FIAs. Moreover, to encourage developers to employ CANTXSec,
we discuss implementation details, providing an analysis based on each user's
risk assessment.

### 7. [Scaling Up: Revisiting Mining Android Sandboxes at Scale for Malware Classification](http://arxiv.org/pdf/2505.09501v1)

Authors: Francisco Costa, Ismael Medeiros, Leandro Oliveira, João Calássio, Rodrigo Bonifácio, Krishna Narasimhan, Mira Mezini, Márcio Ribeiro

The widespread use of smartphones in daily life has raised concerns about
privacy and security among researchers and practitioners. Privacy issues are
generally highly prevalent in mobile applications, particularly targeting the
Android platform, the most popular mobile operating system. For this reason,
several techniques have been proposed to identify malicious behavior in Android
applications, including the Mining Android Sandbox approach (MAS approach),
which aims to identify malicious behavior in repackaged Android applications
(apps). However, previous empirical studies evaluated the MAS approach using a
small dataset consisting of only 102 pairs of original and repackaged apps.
This limitation raises questions about the external validity of their findings
and whether the MAS approach can be generalized to larger datasets. To address
these concerns, this paper presents the results of a replication study focused
on evaluating the performance of the MAS approach regarding its capabilities of
correctly classifying malware from different families. Unlike previous studies,
our research employs a dataset that is an order of magnitude larger, comprising
4,076 pairs of apps covering a more diverse range of Android malware families.
Surprisingly, our findings indicate a poor performance of the MAS approach for
identifying malware, with the F1-score decreasing from 0.90 for the small
dataset used in the previous studies to 0.54 in our more extensive dataset.
Upon closer examination, we discovered that certain malware families partially
account for the low accuracy of the MAS approach, which fails to classify a
repackaged version of an app as malware correctly. Our findings highlight the
limitations of the MAS approach, particularly when scaled, and underscore the
importance of complementing it with other techniques to detect a broader range
of malware effectively.

### 8. [Privacy-Preserving Runtime Verification](http://arxiv.org/pdf/2505.09276v1)

Authors: Thomas A. Henzinger, Mahyar Karimi, K. S. Thejaswini

Runtime verification offers scalable solutions to improve the safety and
reliability of systems. However, systems that require verification or
monitoring by a third party to ensure compliance with a specification might
contain sensitive information, causing privacy concerns when usual runtime
verification approaches are used. Privacy is compromised if protected
information about the system, or sensitive data that is processed by the
system, is revealed. In addition, revealing the specification being monitored
may undermine the essence of third-party verification.
  In this work, we propose two novel protocols for the privacy-preserving
runtime verification of systems against formal sequential specifications. In
our first protocol, the monitor verifies whether the system satisfies the
specification without learning anything else, though both parties are aware of
the specification. Our second protocol ensures that the system remains
oblivious to the monitored specification, while the monitor learns only whether
the system satisfies the specification and nothing more. Our protocols adapt
and improve existing techniques used in cryptography, and more specifically,
multi-party computation.
  The sequential specification defines the observation step of the monitor,
whose granularity depends on the situation (e.g., banks may be monitored on a
daily basis). Our protocols exchange a single message per observation step,
after an initialisation phase. This design minimises communication overhead,
enabling relatively lightweight privacy-preserving monitoring. We implement our
approach for monitoring specifications described by register automata and
evaluate it experimentally.

### 9. [Detecting Sybil Addresses in Blockchain Airdrops: A Subgraph-based Feature Propagation and Fusion Approach](http://arxiv.org/pdf/2505.09313v1)

Authors: Qiangqiang Liu, Qian Huang, Frank Fan, Haishan Wu, Xueyan Tang

Sybil attacks pose a significant security threat to blockchain ecosystems,
particularly in token airdrop events. This paper proposes a novel sybil address
identification method based on subgraph feature extraction lightGBM. The method
first constructs a two-layer deep transaction subgraph for each address, then
extracts key event operation features according to the lifecycle of sybil
addresses, including the time of first transaction, first gas acquisition,
participation in airdrop activities, and last transaction. These temporal
features effectively capture the consistency of sybil address behavior
operations. Additionally, the method extracts amount and network structure
features, comprehensively describing address behavior patterns and network
topology through feature propagation and fusion. Experiments conducted on a
dataset containing 193,701 addresses (including 23,240 sybil addresses) show
that this method outperforms existing approaches in terms of precision, recall,
F1 score, and AUC, with all metrics exceeding 0.9. The methods and results of
this study can be further applied to broader blockchain security areas such as
transaction manipulation identification and token liquidity risk assessment,
contributing to the construction of a more secure and fair blockchain
ecosystem.

### 10. [DNS Query Forgery: A Client-Side Defense Against Mobile App Traffic Profiling](http://arxiv.org/pdf/2505.09374v1)

Authors: Andrea Jimenez-Berenguel, César Gil, Carlos Garcia-Rubio, Jordi Forné, Celeste Campo

Mobile applications continuously generate DNS queries that can reveal
sensitive user behavioral patterns even when communications are encrypted. This
paper presents a privacy enhancement framework based on query forgery to
protect users against profiling attempts that leverage these background
communications. We first mathematically model user profiles as probability
distributions over interest categories derived from mobile application traffic.
We then evaluate three query forgery strategies -- uniform sampling,
TrackMeNot-based generation, and an optimized approach that minimizes
Kullback-Leibler divergence -- to quantify their effectiveness in obfuscating
user profiles. Then we create a synthetic dataset comprising 1,000 user traces
constructed from real mobile application traffic and we extract the user
profiles based on DNS traffic. Our evaluation reveals that a 50\% privacy
improvement is achievable with less than 20\% traffic overhead when using our
approach, while achieving 100\% privacy protection requires approximately
40-60\% additional traffic. We further propose a modular system architecture
for practical implementation of our protection mechanisms on mobile devices.
This work offers a client-side privacy solution that operates without
third-party trust requirements, empowering individual users to defend against
traffic analysis without compromising application functionality.

### Computer Vision and Pattern Recognition

### 1. [2D-3D Attention and Entropy for Pose Robust 2D Facial Recognition](http://arxiv.org/pdf/2505.09073v1)

Authors: J. Brennan Peace, Shuowen Hu, Benjamin S. Riggan

Despite recent advances in facial recognition, there remains a fundamental
issue concerning degradations in performance due to substantial perspective
(pose) differences between enrollment and query (probe) imagery. Therefore, we
propose a novel domain adaptive framework to facilitate improved performances
across large discrepancies in pose by enabling image-based (2D) representations
to infer properties of inherently pose invariant point cloud (3D)
representations. Specifically, our proposed framework achieves better pose
invariance by using (1) a shared (joint) attention mapping to emphasize common
patterns that are most correlated between 2D facial images and 3D facial data
and (2) a joint entropy regularizing loss to promote better
consistency$\unicode{x2014}$enhancing correlations among the intersecting 2D
and 3D representations$\unicode{x2014}$by leveraging both attention maps. This
framework is evaluated on FaceScape and ARL-VTF datasets, where it outperforms
competitive methods by achieving profile (90$\unicode{x00b0}$$\unicode{x002b}$)
TAR @ 1$\unicode{x0025}$ FAR improvements of at least 7.1$\unicode{x0025}$ and
1.57$\unicode{x0025}$, respectively.

### 2. [Seeing Beyond the Scene: Enhancing Vision-Language Models with Interactional Reasoning](http://arxiv.org/pdf/2505.09118v1)

Authors: Dayong Liang, Changmeng Zheng, Zhiyuan Wen, Yi Cai, Xiao-Yong Wei, Qing Li

Traditional scene graphs primarily focus on spatial relationships, limiting
vision-language models' (VLMs) ability to reason about complex interactions in
visual scenes. This paper addresses two key challenges: (1) conventional
detection-to-construction methods produce unfocused, contextually irrelevant
relationship sets, and (2) existing approaches fail to form persistent memories
for generalizing interaction reasoning to new scenes. We propose
Interaction-augmented Scene Graph Reasoning (ISGR), a framework that enhances
VLMs' interactional reasoning through three complementary components. First,
our dual-stream graph constructor combines SAM-powered spatial relation
extraction with interaction-aware captioning to generate functionally salient
scene graphs with spatial grounding. Second, we employ targeted interaction
queries to activate VLMs' latent knowledge of object functionalities,
converting passive recognition into active reasoning about how objects work
together. Finally, we introduce a lone-term memory reinforcement learning
strategy with a specialized interaction-focused reward function that transforms
transient patterns into long-term reasoning heuristics. Extensive experiments
demonstrate that our approach significantly outperforms baseline methods on
interaction-heavy reasoning benchmarks, with particularly strong improvements
on complex scene understanding tasks. The source code can be accessed at
https://github.com/open_upon_acceptance.

### 3. [Promoting SAM for Camouflaged Object Detection via Selective Key Point-based Guidance](http://arxiv.org/pdf/2505.09123v1)

Authors: Guoying Liang, Su Yang

Big model has emerged as a new research paradigm that can be applied to
various down-stream tasks with only minor effort for domain adaption.
Correspondingly, this study tackles Camouflaged Object Detection (COD)
leveraging the Segment Anything Model (SAM). The previous studies declared that
SAM is not workable for COD but this study reveals that SAM works if promoted
properly, for which we devise a new framework to render point promotions:
First, we develop the Promotion Point Targeting Network (PPT-net) to leverage
multi-scale features in predicting the probabilities of camouflaged objects'
presences at given candidate points over the image. Then, we develop a key
point selection (KPS) algorithm to deploy both positive and negative point
promotions contrastively to SAM to guide the segmentation. It is the first work
to facilitate big model for COD and achieves plausible results experimentally
over the existing methods on 3 data sets under 6 metrics. This study
demonstrates an off-the-shelf methodology for COD by leveraging SAM, which
gains advantage over designing professional models from scratch, not only in
performance, but also in turning the problem to a less challenging task, that
is, seeking informative but not exactly precise promotions.

### 4. [Beyond General Prompts: Automated Prompt Refinement using Contrastive Class Alignment Scores for Disambiguating Objects in Vision-Language Models](http://arxiv.org/pdf/2505.09139v1)

Authors: Lucas Choi, Ross Greer

Vision-language models (VLMs) offer flexible object detection through natural
language prompts but suffer from performance variability depending on prompt
phrasing. In this paper, we introduce a method for automated prompt refinement
using a novel metric called the Contrastive Class Alignment Score (CCAS), which
ranks prompts based on their semantic alignment with a target object class
while penalizing similarity to confounding classes. Our method generates
diverse prompt candidates via a large language model and filters them through
CCAS, computed using prompt embeddings from a sentence transformer. We evaluate
our approach on challenging object categories, demonstrating that our automatic
selection of high-precision prompts improves object detection accuracy without
the need for additional model training or labeled data. This scalable and
model-agnostic pipeline offers a principled alternative to manual prompt
engineering for VLM-based detection systems.

### 5. [TopoDiT-3D: Topology-Aware Diffusion Transformer with Bottleneck Structure for 3D Point Cloud Generation](http://arxiv.org/pdf/2505.09140v1)

Authors: Zechao Guan, Feng Yan, Shuai Du, Lin Ma, Qingshan Liu

Recent advancements in Diffusion Transformer (DiT) models have significantly
improved 3D point cloud generation. However, existing methods primarily focus
on local feature extraction while overlooking global topological information,
such as voids, which are crucial for maintaining shape consistency and
capturing complex geometries. To address this limitation, we propose
TopoDiT-3D, a Topology-Aware Diffusion Transformer with a bottleneck structure
for 3D point cloud generation. Specifically, we design the bottleneck structure
utilizing Perceiver Resampler, which not only offers a mode to integrate
topological information extracted through persistent homology into feature
learning, but also adaptively filters out redundant local features to improve
training efficiency. Experimental results demonstrate that TopoDiT-3D
outperforms state-of-the-art models in visual quality, diversity, and training
efficiency. Furthermore, TopoDiT-3D demonstrates the importance of rich
topological information for 3D point cloud generation and its synergy with
conventional local feature learning. Videos and code are available at
https://github.com/Zechao-Guan/TopoDiT-3D.

### 6. [AMSnet 2.0: A Large AMS Database with AI Segmentation for Net Detection](http://arxiv.org/pdf/2505.09155v1)

Authors: Yichen Shi, Zhuofu Tao, Yuhao Gao, Li Huang, Hongyang Wang, Zhiping Yu, Ting-Jung Lin, Lei He

Current multimodal large language models (MLLMs) struggle to understand
circuit schematics due to their limited recognition capabilities. This could be
attributed to the lack of high-quality schematic-netlist training data.
Existing work such as AMSnet applies schematic parsing to generate netlists.
However, these methods rely on hard-coded heuristics and are difficult to apply
to complex or noisy schematics in this paper. We therefore propose a novel net
detection mechanism based on segmentation with high robustness. The proposed
method also recovers positional information, allowing digital reconstruction of
schematics. We then expand AMSnet dataset with schematic images from various
sources and create AMSnet 2.0. AMSnet 2.0 contains 2,686 circuits with
schematic images, Spectre-formatted netlists, OpenAccess digital schematics,
and positional information for circuit components and nets, whereas AMSnet only
includes 792 circuits with SPICE netlists but no digital schematics.

### 7. [UniCAD: Efficient and Extendable Architecture for Multi-Task Computer-Aided Diagnosis System](http://arxiv.org/pdf/2505.09178v2)

Authors: Yitao Zhu, Yuan Yin, Zhenrong Shen, Zihao Zhao, Haiyu Song, Sheng Wang, Dinggang Shen, Qian Wang

The growing complexity and scale of visual model pre-training have made
developing and deploying multi-task computer-aided diagnosis (CAD) systems
increasingly challenging and resource-intensive. Furthermore, the medical
imaging community lacks an open-source CAD platform to enable the rapid
creation of efficient and extendable diagnostic models. To address these
issues, we propose UniCAD, a unified architecture that leverages the robust
capabilities of pre-trained vision foundation models to seamlessly handle both
2D and 3D medical images while requiring only minimal task-specific parameters.
UniCAD introduces two key innovations: (1) Efficiency: A low-rank adaptation
strategy is employed to adapt a pre-trained visual model to the medical image
domain, achieving performance on par with fully fine-tuned counterparts while
introducing only 0.17% trainable parameters. (2) Plug-and-Play: A modular
architecture that combines a frozen foundation model with multiple
plug-and-play experts, enabling diverse tasks and seamless functionality
expansion. Building on this unified CAD architecture, we establish an
open-source platform where researchers can share and access lightweight CAD
experts, fostering a more equitable and efficient research ecosystem.
Comprehensive experiments across 12 diverse medical datasets demonstrate that
UniCAD consistently outperforms existing methods in both accuracy and
deployment efficiency. The source code and project page are available at
https://mii-laboratory.github.io/UniCAD/.

### 8. [Zero-shot Quantization: A Comprehensive Survey](http://arxiv.org/pdf/2505.09188v1)

Authors: Minjun Kim, Jaehyeon Choi, Jongkeun Lee, Wonjin Cho, U Kang

Network quantization has proven to be a powerful approach to reduce the
memory and computational demands of deep learning models for deployment on
resource-constrained devices. However, traditional quantization methods often
rely on access to training data, which is impractical in many real-world
scenarios due to privacy, security, or regulatory constraints. Zero-shot
Quantization (ZSQ) emerges as a promising solution, achieving quantization
without requiring any real data. In this paper, we provide a comprehensive
overview of ZSQ methods and their recent advancements. First, we provide a
formal definition of the ZSQ problem and highlight the key challenges. Then, we
categorize the existing ZSQ methods into classes based on data generation
strategies, and analyze their motivations, core ideas, and key takeaways.
Lastly, we suggest future research directions to address the remaining
limitations and advance the field of ZSQ. To the best of our knowledge, this
paper is the first in-depth survey on ZSQ.

### 9. [PDE: Gene Effect Inspired Parameter Dynamic Evolution for Low-light Image Enhancement](http://arxiv.org/pdf/2505.09196v1)

Authors: Tong Li, Lizhi Wang, Hansen Feng, Lin Zhu, Hua Huang

Low-light image enhancement (LLIE) is a fundamental task in computational
photography, aiming to improve illumination, reduce noise, and enhance image
quality. While recent advancements focus on designing increasingly complex
neural network models, we observe a peculiar phenomenon: resetting certain
parameters to random values unexpectedly improves enhancement performance for
some images. Drawing inspiration from biological genes, we term this phenomenon
the gene effect. The gene effect limits enhancement performance, as even random
parameters can sometimes outperform learned ones, preventing models from fully
utilizing their capacity. In this paper, we investigate the reason and propose
a solution. Based on our observations, we attribute the gene effect to static
parameters, analogous to how fixed genetic configurations become maladaptive
when environments change. Inspired by biological evolution, where adaptation to
new environments relies on gene mutation and recombination, we propose
parameter dynamic evolution (PDE) to adapt to different images and mitigate the
gene effect. PDE employs a parameter orthogonal generation technique and the
corresponding generated parameters to simulate gene recombination and gene
mutation, separately. Experiments validate the effectiveness of our techniques.
The code will be released to the public.

### 10. [A Surrogate Model for the Forward Design of Multi-layered Metasurface-based Radar Absorbing Structures](http://arxiv.org/pdf/2505.09251v1)

Authors: Vineetha Joy, Aditya Anand, Nidhi, Anshuman Kumar, Amit Sethi, Hema Singh

Metasurface-based radar absorbing structures (RAS) are highly preferred for
applications like stealth technology, electromagnetic (EM) shielding, etc. due
to their capability to achieve frequency selective absorption characteristics
with minimal thickness and reduced weight penalty. However, the conventional
approach for the EM design and optimization of these structures relies on
forward simulations, using full wave simulation tools, to predict the
electromagnetic (EM) response of candidate meta atoms. This process is
computationally intensive, extremely time consuming and requires exploration of
large design spaces. To overcome this challenge, we propose a surrogate model
that significantly accelerates the prediction of EM responses of multi-layered
metasurface-based RAS. A convolutional neural network (CNN) based architecture
with Huber loss function has been employed to estimate the reflection
characteristics of the RAS model. The proposed model achieved a cosine
similarity of 99.9% and a mean square error of 0.001 within 1000 epochs of
training. The efficiency of the model has been established via full wave
simulations as well as experiment where it demonstrated significant reduction
in computational time while maintaining high predictive accuracy.

### Computers and Society

### 1. [A Method for Assisting Novices Creating Class Diagrams Based on the Instructor's Class Layout](http://arxiv.org/pdf/2505.09116v1)

Authors: Yuta Saito, Takehiro Kokubu, Takafumi Tanaka, Atsuo Hazeyama, Hiroaki Hashiura

Nowadays, modeling exercises on software development objects are conducted in
higher education institutions for information technology. Not only are there
many defects such as missing elements in the models created by learners during
the exercises, but the layout of elements in the class diagrams often differs
significantly from the correct answers created by the instructors. In this
paper, we focus on the above problem and propose a method to provide effective
support to learners during modeling exercises by automatically converting the
layout of the learner's class diagram to that of the instructor, in addition to
indicating the correctness of the artifacts to the learners during the
exercises. The proposed method was implemented and evaluated as a tool, and the
results indicate that the automatic layout conversion was an effective feedback
to the learners.

### 2. [Ethical Aspects of the Use of Social Robots in Elderly Care -- A Systematic Qualitative Review](http://arxiv.org/pdf/2505.09224v1)

Authors: Marianne Leineweber, Clara Victoria Keusgen, Marc Bubeck, Joschka Haltaufderheide, Robert Ranisch, Corinna Klingler

Background: The use of social robotics in elderly care is increasingly
discussed as one way of meeting emerging care needs due to scarce resources.
While many potential benefits are associated with robotic care technologies,
there is a variety of ethical challenges. To support steps towards a
responsible implementation and use, this review develops an overview on ethical
aspects of the use of social robots in elderly care from a decision-makers'
perspective.
  Methods: Electronic databases were queried using a comprehensive search
strategy based on the key concepts of "ethical aspects", "social robotics" and
"elderly care". Abstract and title screening was conducted by two authors
independently. Full-text screening was conducted by one author following a
joint consolidation phase. Data was extracted using MAXQDA24 by one author,
based on a consolidated coding framework. Analysis was performed through
modified qualitative content analysis.
  Results: A total of 1,518 publications were screened, and 248 publications
were included. We have organized our analysis in a scheme of ethical hazards,
ethical opportunities and unsettled questions, identifying at least 60 broad
ethical aspects affecting three different stakeholder groups. While some
ethical issues are well-known and broadly discussed our analysis shows a
plethora of potentially relevant aspects, often only marginally recognized,
that are worthy of consideration from a practical perspective.
  Discussion: The findings highlight the need for a contextual and detailed
evaluation of implementation scenarios. To make use of the vast knowledge of
the ethical discourse, we hypothesize that decision-makers need to understand
the specific nature of this discourse to be able to engage in careful ethical
deliberation.

### 3. [Ranking-Based At-Risk Student Prediction Using Federated Learning and Differential Features](http://arxiv.org/pdf/2505.09287v1)

Authors: Shunsuke Yoneda, Valdemar Švábenský, Gen Li, Daisuke Deguchi, Atsushi Shimada

Digital textbooks are widely used in various educational contexts, such as
university courses and online lectures. Such textbooks yield learning log data
that have been used in numerous educational data mining (EDM) studies for
student behavior analysis and performance prediction. However, these studies
have faced challenges in integrating confidential data, such as academic
records and learning logs, across schools due to privacy concerns.
Consequently, analyses are often conducted with data limited to a single
school, which makes developing high-performing and generalizable models
difficult. This study proposes a method that combines federated learning and
differential features to address these issues. Federated learning enables model
training without centralizing data, thereby preserving student privacy.
Differential features, which utilize relative values instead of absolute
values, enhance model performance and generalizability. To evaluate the
proposed method, a model for predicting at-risk students was trained using data
from 1,136 students across 12 courses conducted over 4 years, and validated on
hold-out test data from 5 other courses. Experimental results demonstrated that
the proposed method addresses privacy concerns while achieving performance
comparable to that of models trained via centralized learning in terms of Top-n
precision, nDCG, and PR-AUC. Furthermore, using differential features improved
prediction performance across all evaluation datasets compared to
non-differential approaches. The trained models were also applicable for early
prediction, achieving high performance in detecting at-risk students in earlier
stages of the semester within the validation datasets.

### 4. [Ethics and Persuasion in Reinforcement Learning from Human Feedback: A Procedural Rhetorical Approach](http://arxiv.org/pdf/2505.09576v1)

Authors: Shannon Lodoen, Alexi Orchard

Since 2022, versions of generative AI chatbots such as ChatGPT and Claude
have been trained using a specialized technique called Reinforcement Learning
from Human Feedback (RLHF) to fine-tune language model output using feedback
from human annotators. As a result, the integration of RLHF has greatly
enhanced the outputs of these large language models (LLMs) and made the
interactions and responses appear more "human-like" than those of previous
versions using only supervised learning. The increasing convergence of human
and machine-written text has potentially severe ethical, sociotechnical, and
pedagogical implications relating to transparency, trust, bias, and
interpersonal relations. To highlight these implications, this paper presents a
rhetorical analysis of some of the central procedures and processes currently
being reshaped by RLHF-enhanced generative AI chatbots: upholding language
conventions, information seeking practices, and expectations for social
relationships. Rhetorical investigations of generative AI and LLMs have, to
this point, focused largely on the persuasiveness of the content generated.
Using Ian Bogost's concept of procedural rhetoric, this paper shifts the site
of rhetorical investigation from content analysis to the underlying mechanisms
of persuasion built into RLHF-enhanced LLMs. In doing so, this theoretical
investigation opens a new direction for further inquiry in AI ethics that
considers how procedures rerouted through AI-driven technologies might
reinforce hegemonic language use, perpetuate biases, decontextualize learning,
and encroach upon human relationships. It will therefore be of interest to
educators, researchers, scholars, and the growing number of users of generative
AI chatbots.

### 5. [How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference](http://arxiv.org/pdf/2505.09598v1)

Authors: Nidhal Jegham, Marwen Abdelatti, Lassad Elmoubarki, Abdeltawab Hendawi

As large language models (LLMs) spread across industries, understanding their
environmental footprint at the inference level is no longer optional; it is
essential. However, most existing studies exclude proprietary models, overlook
infrastructural variability and overhead, or focus solely on training, even as
inference increasingly dominates AI's environmental impact. To bridge this gap,
this paper introduces a novel infrastructure-aware benchmarking framework for
quantifying the environmental footprint of LLM inference across 30
state-of-the-art models as deployed in commercial data centers. Our framework
combines public API performance data with region-specific environmental
multipliers and statistical inference of hardware configurations. We
additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank
models by performance relative to environmental cost. Our results show that o3
and DeepSeek-R1 emerge as the most energy-intensive models, consuming over 33
Wh per long prompt, more than 70 times the consumption of GPT-4.1 nano, and
that Claude-3.7 Sonnet ranks highest in eco-efficiency. While a single short
GPT-4o query consumes 0.43 Wh, scaling this to 700 million queries/day results
in substantial annual environmental impacts. These include electricity use
comparable to 35,000 U.S. homes, freshwater evaporation matching the annual
drinking needs of 1.2 million people, and carbon emissions requiring a
Chicago-sized forest to offset. These findings illustrate a growing paradox:
although individual queries are efficient, their global scale drives
disproportionate resource consumption. Our study provides a standardized,
empirically grounded methodology for benchmarking the sustainability of LLM
deployments, laying a foundation for future environmental accountability in AI
development and sustainability standards.

### 6. [EcoSphere: A Decision-Support Tool for Automated Carbon Emission and Cost Optimization in Sustainable Urban Development](http://arxiv.org/pdf/2505.09054v1)

Authors: Siavash Ghorbany, Ming Hu, Siyuan Yao, Matthew Sisk, Chaoli Wang

The construction industry is a major contributor to global greenhouse gas
emissions, with embodied carbon being a key component. This study develops
EcoSphere, an innovative software designed to evaluate and balance embodied and
operational carbon emissions with construction and environmental costs in urban
planning. Using high-resolution data from the National Structure Inventory,
combined with computer vision and natural language processing applied to Google
Street View and satellite imagery, EcoSphere categorizes buildings by
structural and material characteristics with a bottom-up approach, creating a
baseline emissions dataset. By simulating policy scenarios and mitigation
strategies, EcoSphere provides policymakers and non-experts with actionable
insights for sustainable development in cities and provide them with a vision
of the environmental and financial results of their decisions. Case studies in
Chicago and Indianapolis showcase how EcoSphere aids in assessing policy
impacts on carbon emissions and costs, supporting data-driven progress toward
carbon neutrality.

### 7. [Toward Fair Federated Learning under Demographic Disparities and Data Imbalance](http://arxiv.org/pdf/2505.09295v1)

Authors: Qiming Wu, Siqi Li, Doudou Zhou, Nan Liu

Ensuring fairness is critical when applying artificial intelligence to
high-stakes domains such as healthcare, where predictive models trained on
imbalanced and demographically skewed data risk exacerbating existing
disparities. Federated learning (FL) enables privacy-preserving collaboration
across institutions, but remains vulnerable to both algorithmic bias and
subgroup imbalance - particularly when multiple sensitive attributes intersect.
We propose FedIDA (Fed erated Learning for Imbalance and D isparity A
wareness), a framework-agnostic method that combines fairness-aware
regularization with group-conditional oversampling. FedIDA supports multiple
sensitive attributes and heterogeneous data distributions without altering the
convergence behavior of the underlying FL algorithm. We provide theoretical
analysis establishing fairness improvement bounds using Lipschitz continuity
and concentration inequalities, and show that FedIDA reduces the variance of
fairness metrics across test sets. Empirical results on both benchmark and
real-world clinical datasets confirm that FedIDA consistently improves fairness
while maintaining competitive predictive performance, demonstrating its
effectiveness for equitable and privacy-preserving modeling in healthcare. The
source code is available on GitHub.

### 8. [WorldView-Bench: A Benchmark for Evaluating Global Cultural Perspectives in Large Language Models](http://arxiv.org/pdf/2505.09595v1)

Authors: Abdullah Mushtaq, Imran Taj, Rafay Naeem, Ibrahim Ghaznavi, Junaid Qadir

Large Language Models (LLMs) are predominantly trained and aligned in ways
that reinforce Western-centric epistemologies and socio-cultural norms, leading
to cultural homogenization and limiting their ability to reflect global
civilizational plurality. Existing benchmarking frameworks fail to adequately
capture this bias, as they rely on rigid, closed-form assessments that overlook
the complexity of cultural inclusivity. To address this, we introduce
WorldView-Bench, a benchmark designed to evaluate Global Cultural Inclusivity
(GCI) in LLMs by analyzing their ability to accommodate diverse worldviews. Our
approach is grounded in the Multiplex Worldview proposed by Senturk et al.,
which distinguishes between Uniplex models, reinforcing cultural
homogenization, and Multiplex models, which integrate diverse perspectives.
WorldView-Bench measures Cultural Polarization, the exclusion of alternative
perspectives, through free-form generative evaluation rather than conventional
categorical benchmarks. We implement applied multiplexity through two
intervention strategies: (1) Contextually-Implemented Multiplex LLMs, where
system prompts embed multiplexity principles, and (2) Multi-Agent System
(MAS)-Implemented Multiplex LLMs, where multiple LLM agents representing
distinct cultural perspectives collaboratively generate responses. Our results
demonstrate a significant increase in Perspectives Distribution Score (PDS)
entropy from 13% at baseline to 94% with MAS-Implemented Multiplex LLMs,
alongside a shift toward positive sentiment (67.7%) and enhanced cultural
balance. These findings highlight the potential of multiplex-aware AI
evaluation in mitigating cultural bias in LLMs, paving the way for more
inclusive and ethically aligned AI systems.

### 9. [The Niche Connectivity Paradox: Multichrome Contagions Overcome Vaccine Hesitancy more effectively than Monochromacy](http://arxiv.org/pdf/2505.09605v1)

Authors: Ho-Chun Herbert Chang, Feng Fu

The rise of vaccine hesitancy has caused a resurgence of vaccine-preventable
diseases such as measles and pertussis, alongside widespread skepticism and
refusals of COVID-19 vaccinations. While categorizing individuals as either
supportive of or opposed to vaccines provides a convenient dichotomy of vaccine
attitudes, vaccine hesitancy is far more complex and dynamic. It involves
wavering individuals whose attitudes fluctuate -- those who may exhibit
pro-vaccine attitudes at one time and anti-vaccine attitudes at another. Here,
we identify and analyze multichrome contagions as potential targets for
intervention by leveraging a dataset of known pro-vax and anti-vax Twitter
users ($n =135$ million) and a large COVID-19 Twitter dataset ($n = 3.5$
billion; including close analysis of $1,563,472$ unique individuals). We
reconstruct an evolving multiplex sentiment landscape using top co-spreading
issues, characterizing them as monochrome and multichrome contagions, based on
their conceptual overlap with vaccination. We demonstrate switchers as
deliberative: they are more moderate, engage with a wider range of topics, and
occupy more central positions in their networks. Further examination of their
information consumption shows that their discourse often engages with
progressive issues such as climate change, which can serve as avenues for
multichrome contagion interventions to promote pro-vaccine attitudes. Using
data-driven intervention simulations, we demonstrate a paradox of niche
connectivity, where multichrome contagions with fragmented, non-overlapping
communities generate the highest levels of diffusion for pro-vaccine attitudes.
Our work offers insights into harnessing synergistic hitchhiking effect of
multichrome contagions to drive desired attitude and behavior changes in
network-based interventions, particularly for overcoming vaccine hesitancy.

### Databases

### 1. [SHACL-DS: A SHACL extension to validate RDF dataset](http://arxiv.org/pdf/2505.09198v1)

Authors: Christophe Debruyne, Davan Chiem Dao

The Shapes Constraint Language (SHACL) provides a powerful mechanism for
validating RDF data against shape constraints, but is inherently designed for
single-graph validation. This limitation makes SHACL unsuitable for natively
validating RDF datasets comprising multiple named graphs. To address this gap,
developers must build solutions on top of SHACL, applying a shapes graph to
each RDF dataset or combinations thereof using bespoke code. However, these
approaches may lead to information loss, such as the named graph from which the
data originates. This paper introduces SHACL-DS, an extension to SHACL that
enables validation of RDF datasets. The extension adds a layer on top of SHACL,
and the only disruptive change is the execution of SPARQL queries in, e.g.,
SPARQL-based constraints. The contributions are a SHACL-DS specification, a
prototype implementation, and a set of test cases illustrating its use and
providing future developers guidance in building SHACL-DS engines. This work
lays the foundation for integrating dataset-level features into SHACL and
encourages further exploration of advanced RDF dataset validation techniques.

### Distributed, Parallel, and Cluster Computing

### 1. [Towards Efficient Verification of Parallel Applications with Mc SimGrid](http://arxiv.org/pdf/2505.09209v1)

Authors: Matthieu Laurent, Thierry Jéron, Martin Quinson

Assessing the correctness of distributed and parallel applications is
notoriously difficult due to the complexity of the concurrent behaviors and the
difficulty to reproduce bugs. In this context, Dynamic Partial Order Reduction
(DPOR) techniques have proved successful in exploiting concurrency to verify
applications without exploring all their behaviors. However, they may lack of
efficiency when tracking non-systematic bugs of real size applications. In this
paper, we suggest two adaptations of the Optimal Dynamic Partial Order
Reduction (ODPOR) algorithm with a particular focus on bug finding and
explanation. The first adaptation is an out-of-order version called RFS ODPOR
which avoids being stuck in uninteresting large parts of the state space. Once
a bug is found, the second adaptation takes advantage of ODPOR principles to
efficiently find the origins of the bug.

### 2. [Efficient Graph Embedding at Scale: Optimizing CPU-GPU-SSD Integration](http://arxiv.org/pdf/2505.09258v2)

Authors: Zhonggen Li, Xiangyu Ke, Yifan Zhu, Yunjun Gao, Feifei Li

Graph embeddings provide continuous vector representations of nodes in a
graph, which are widely applicable in community detection, recommendations, and
various scientific fields. However, existing graph embedding systems either
face scalability challenges due to the high cost of RAM and multiple GPUs, or
rely on disk storage at the expense of I/O efficiency. In this paper, we
propose Legend, a lightweight heterogeneous system for graph embedding that
systematically redefines data management across CPU, GPU, and NVMe SSD
resources. Legend is built on a foundation of efficient data placement and
retrieval strategies tailored to the unique strengths of each hardware. Key
innovations include a prefetch-friendly embedding loading strategy, enabling
GPUs to directly prefetch data from SSDs with minimal I/O overhead, and a
high-throughput GPU-SSD direct access driver optimized for graph embedding
tasks. Furthermore, we propose a customized parallel execution strategy to
maximize GPU utilization, ensuring efficient handling of billion-scale
datasets. Extensive experiments demonstrate that Legend achieves up to 4.8x
speedup compared to state-of-the-art systems. Moreover, Legend exhibits
comparable performance on a single GPU to that of the state-of-the-art system
using 4 GPUs on the billion-scale dataset.

### 3. [Strategies to Measure Energy Consumption Using RAPL During Workflow Execution on Commodity Clusters](http://arxiv.org/pdf/2505.09375v1)

Authors: Philipp Thamm

In science, problems in many fields can be solved by processing datasets
using a series of computationally expensive algorithms, sometimes referred to
as workflows. Traditionally, the configurations of these workflows are
optimized to achieve a short runtime for the given task and dataset on a given
(often distributed) infrastructure. However, recently more attention has been
drawn to energy-efficient computing, due to the negative impact of
energy-inefficient computing on the environment and energy costs. To be able to
assess the energy-efficiency of a given workflow configuration, reliable and
accurate methods to measure the energy consumption of a system are required.
One approach is the usage of built-in hardware energy counters, such as Intel
RAPL. Unfortunately, effectively using RAPL for energy measurement within a
workflow on a managed cluster with the typical deep software infrastructure
stack can be difficult, for instance because of limited privileges and the need
for communication between nodes. In this paper, we describe three ways to
implement RAPL energy measurement on a Kubernetes cluster while executing
scientific workflows utilizing the Nextflow workflow engine. We compare them by
utilizing a set of eight criteria that should be fulfilled for accurate
measurement, such as the ability to react to workflow faults, portability, and
added overhead. We highlight advantages and drawbacks of each method and
discuss challenges and pitfalls, as well as ways to avoid them. We also
empirically evaluate all methods, and find that approaches using a shell script
and a Nextflow plugin are both effective and easy to implement. Additionally,
we find that measuring the energy consumption of a single task is straight
forward when only one task runs at a time, but concurrent task executions on
the same node require approximating per-task energy usage using metrics such as
CPU utilization.

### 4. [ARM SVE Unleashed: Performance and Insights Across HPC Applications on Nvidia Grace](http://arxiv.org/pdf/2505.09462v1)

Authors: Ruimin Shi, Gabin Schieffer, Maya Gokhale, Pei-Hung Lin, Hiren Patel, Ivy Peng

Vector architectures are essential for boosting computing throughput. ARM
provides SVE as the next-generation length-agnostic vector extension beyond
traditional fixed-length SIMD. This work provides a first study of the maturity
and readiness of exploiting ARM and SVE in HPC. Using selected performance
hardware events on the ARM Grace processor and analytical models, we derive new
metrics to quantify the effectiveness of exploiting SVE vectorization to reduce
executed instructions and improve performance speedup. We further propose an
adapted roofline model that combines vector length and data elements to
identify potential performance bottlenecks. Finally, we propose a decision tree
for classifying the SVE-boosted performance in applications.

### 5. [MDTP -- An Adaptive Multi-Source Data Transfer Protocol](http://arxiv.org/pdf/2505.09597v1)

Authors: Sepideh Abdollah, Craig Partridge, Susmit Shannigrahi

Scientific data volume is growing in size, and as a direct result, the need
for faster transfers is also increasing. The scientific community has sought to
leverage parallel transfer methods using multi-threaded and multi-source
download models to reduce download times. In multi-source transfers, a client
downloads data from multiple replicated servers in parallel. Tools such as
Aria2 and BitTorrent support such multi-source transfers and have shown
improved transfer times.
  In this work, we introduce Multi-Source Data Transfer Protocol, MDTP, which
further improves multi-source transfer performance. MDTP logically divides a
file request into smaller chunk requests and distributes the chunk requests
across multiple servers. Chunk sizes are adapted based on each server's
performance but selected in a way that ensures each round of requests completes
around the same time. We formulate this chunk-size allocation problem as a
variant of the bin-packing problem, where adaptive chunking efficiently fills
the available capacity "bins" corresponding to each server.
  Our evaluation shows that MDTP reduces transfer times by 10-22% compared to
Aria2, the fastest alternative. Comparisons with other protocols, such as
static chunking and BitTorrent, demonstrate even greater improvements.
Additionally, we show that MDTP distributes load proportionally across all
available replicas, not just the fastest ones, which improves throughput.
Finally, we show MDTP maintains high throughput even when latency increases or
bandwidth to the fastest server decreases.

### 6. [The Adaptive Complexity of Finding a Stationary Point](http://arxiv.org/pdf/2505.09045v1)

Authors: Huanjian Zhou, Andi Han, Akiko Takeda, Masashi Sugiyama

In large-scale applications, such as machine learning, it is desirable to
design non-convex optimization algorithms with a high degree of
parallelization. In this work, we study the adaptive complexity of finding a
stationary point, which is the minimal number of sequential rounds required to
achieve stationarity given polynomially many queries executed in parallel at
each round.
  For the high-dimensional case, i.e., $d = \widetilde{\Omega}(\varepsilon^{-(2
+ 2p)/p})$, we show that for any (potentially randomized) algorithm, there
exists a function with Lipschitz $p$-th order derivatives such that the
algorithm requires at least $\varepsilon^{-(p+1)/p}$ iterations to find an
$\varepsilon$-stationary point. Our lower bounds are tight and show that even
with $\mathrm{poly}(d)$ queries per iteration, no algorithm has better
convergence rate than those achievable with one-query-per-round algorithms. In
other words, gradient descent, the cubic-regularized Newton's method, and the
$p$-th order adaptive regularization method are adaptively optimal. Our proof
relies upon novel analysis with the characterization of the output for the
hardness potentials based on a chain-like structure with random partition.
  For the constant-dimensional case, i.e., $d = \Theta(1)$, we propose an
algorithm that bridges grid search and gradient flow trapping, finding an
approximate stationary point in constant iterations. Its asymptotic tightness
is verified by a new lower bound on the required queries per iteration. We show
there exists a smooth function such that any algorithm running with
$\Theta(\log (1/\varepsilon))$ rounds requires at least
$\widetilde{\Omega}((1/\varepsilon)^{(d-1)/2})$ queries per round. This lower
bound is tight up to a logarithmic factor, and implies that the gradient flow
trapping is adaptively optimal.

### 7. [Architecture of Tianyu Software: Relative Photometry as a Case Study](http://arxiv.org/pdf/2505.09107v2)

Authors: Yicheng Rui, Yifan Xuan, Shuyue Zheng, Kexin Li, Kaiming Cui, Kai Xiao, Jie Zheng, Jun Kai Ng, Hongxuan Jiang, Fabo Feng, Qinghui Sun

Tianyu telescope, an one-meter robotic optical survey instrument to be
constructed in Lenghu, Qinghai, China, is designed for detecting transiting
exoplanets, variable stars and transients. It requires a highly automated,
optimally distributed, easily extendable, and highly flexible software to
enable the data processing for the raw data at rates exceeding 500MB/s. In this
work, we introduce the architecture of the Tianyu pipeline and use relative
photometry as a case to demonstrate its high scalability and efficiency. This
pipeline is tested on the data collected from Muguang observatory and Xinglong
observatory. The pipeline demonstrates high scalability, with most processing
stages increasing in throughput as the number of consumers grows. Compared to a
single consumer, the median throughput of image calibration, alignment, and
flux extraction increases by 41%, 257%, and 107% respectively when using 5
consumers, while image stacking exhibits limited scalability due to I/O
constraints. In our tests, the pipeline was able to detect two transiting
sources. Besides, the pipeline captures variability in the light curves of nine
known and two previously unknown variable sources in the testing data.
Meanwhile, the differential photometric precision of the light curves is near
the theoretical limitation. These results indicate that this pipeline is
suitable for detecting transiting exoplanets and variable stars. This work
builds the fundation for further development of Tianyu software. Code of this
work is available at https://github.com/ruiyicheng/Tianyu_pipeline.

### 8. [Toward Malicious Clients Detection in Federated Learning](http://arxiv.org/pdf/2505.09110v1)

Authors: Zhihao Dou, Jiaqi Wang, Wei Sun, Zhuqing Liu, Minghong Fang

Federated learning (FL) enables multiple clients to collaboratively train a
global machine learning model without sharing their raw data. However, the
decentralized nature of FL introduces vulnerabilities, particularly to
poisoning attacks, where malicious clients manipulate their local models to
disrupt the training process. While Byzantine-robust aggregation rules have
been developed to mitigate such attacks, they remain inadequate against more
advanced threats. In response, recent advancements have focused on FL detection
techniques to identify potentially malicious participants. Unfortunately, these
methods often misclassify numerous benign clients as threats or rely on
unrealistic assumptions about the server's capabilities. In this paper, we
propose a novel algorithm, SafeFL, specifically designed to accurately identify
malicious clients in FL. The SafeFL approach involves the server collecting a
series of global models to generate a synthetic dataset, which is then used to
distinguish between malicious and benign models based on their behavior.
Extensive testing demonstrates that SafeFL outperforms existing methods,
offering superior efficiency and accuracy in detecting malicious clients.

### 9. [ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor](http://arxiv.org/pdf/2505.09142v1)

Authors: Seungbeom Choi, Jeonghoe Goo, Eunjoo Jeon, Mingyu Yang, Minsung Jang

We propose ELIS, a serving system for Large Language Models (LLMs) featuring
an Iterative Shortest Remaining Time First (ISRTF) scheduler designed to
efficiently manage inference tasks with the shortest remaining tokens. Current
LLM serving systems often employ a first-come-first-served scheduling strategy,
which can lead to the "head-of-line blocking" problem. To overcome this
limitation, it is necessary to predict LLM inference times and apply a shortest
job first scheduling strategy. However, due to the auto-regressive nature of
LLMs, predicting the inference latency is challenging. ELIS addresses this
challenge by training a response length predictor for LLMs using the BGE model,
an encoder-based state-of-the-art model. Additionally, we have devised the
ISRTF scheduling strategy, an optimization of shortest remaining time first
tailored to existing LLM iteration batching. To evaluate our work in an
industrial setting, we simulate streams of requests based on our study of
real-world user LLM serving trace records. Furthermore, we implemented ELIS as
a cloud-native scheduler system on Kubernetes to evaluate its performance in
production environments. Our experimental results demonstrate that ISRTF
reduces the average job completion time by up to 19.6%.

### 10. [Birch SGD: A Tree Graph Framework for Local and Asynchronous SGD Methods](http://arxiv.org/pdf/2505.09218v1)

Authors: Alexander Tyurin, Danil Sivtsov

We propose a new unifying framework, Birch SGD, for analyzing and designing
distributed SGD methods. The central idea is to represent each method as a
weighted directed tree, referred to as a computation tree. Leveraging this
representation, we introduce a general theoretical result that reduces
convergence analysis to studying the geometry of these trees. This perspective
yields a purely graph-based interpretation of optimization dynamics, offering a
new and intuitive foundation for method development. Using Birch SGD, we design
eight new methods and analyze them alongside previously known ones, with at
least six of the new methods shown to have optimal computational time
complexity. Our research leads to two key insights: (i) all methods share the
same "iteration rate" of $O\left(\frac{(R + 1) L \Delta}{\varepsilon} +
\frac{\sigma^2 L \Delta}{\varepsilon^2}\right)$, where $R$ the maximum "tree
distance" along the main branch of a tree; and (ii) different methods exhibit
different trade-offs-for example, some update iterates more frequently,
improving practical performance, while others are more communication-efficient
or focus on other aspects. Birch SGD serves as a unifying framework for
navigating these trade-offs. We believe these results provide a unified
foundation for understanding, analyzing, and designing efficient asynchronous
and parallel optimization methods.

### Discrete Mathematics

### 1. [Linear Search with Probabilistic Detection and Variable Speeds](http://arxiv.org/pdf/2505.09429v1)

Authors: Jared Coleman, Oscar Morales-Ponce

We present results on new variants of the famous linear search (or cow-path)
problem that involves an agent searching for a target with unknown position on
the infinite line. We consider the variant where the agent can move either at
speed $1$ or at a slower speed $v \in [0, 1)$. When traveling at the slower
speed $v$, the agent is guaranteed to detect the target upon passing through
its location. When traveling at speed $1$, however, the agent, upon passing
through the target's location, detects it with probability $p \in [0, 1]$. We
present algorithms and provide upper bounds for the competitive ratios for
three cases separately: when $p=0$, $v=0$, and when $p,v \in (0,1)$. We also
prove that the provided algorithm for the $p=0$ case is optimal.

### Data Structures and Algorithms

### 1. [Approximate Cartesian Tree Matching with One Difference](http://arxiv.org/pdf/2505.09236v1)

Authors: Bastien Auvray, Julien David, Samah Ghazawi, Richard Groult, Gad M. Landau, Thierry Lecroq

Cartesian tree pattern matching consists of finding all the factors of a text
that have the same Cartesian tree than a given pattern. There already exist
theoretical and practical solutions for the exact case. In this paper, we
propose the first algorithms for solving approximate Cartesian tree pattern
matching with one difference given a pattern of length m and a text of length
n. We present a generic algorithm that find all the factors of the text that
have the same Cartesian tree of the pattern with one difference, using
different notions of differences. We show that this algorithm has a O(nM)
worst-case complexity and that, for several random models, the algorithm has a
linear average-case complexity. We also present an automaton based algorithm,
adapting [PALP19], that can be generalized to deal with more than one
difference.

### 2. [Structural Parameterization of Steiner Tree Packing](http://arxiv.org/pdf/2505.09250v1)

Authors: Niko Hastrich, Kirill Simonov

Steiner Tree Packing (STP) is a notoriously hard problem in classical
complexity theory, which is of practical relevance to VLSI circuit design.
Previous research has approached this problem by providing heuristic or
approximate algorithms. In this paper, we show the first FPT algorithms for STP
parameterized by structural parameters of the input graph. In particular, we
show that STP is fixed-parameter tractable by the tree-cut width as well as the
fracture number of the input graph.
  To achieve our results, we generalize techniques from Edge-Disjoint Paths
(EDP) to Generalized Steiner Tree Packing (GSTP), which generalizes both STP
and EDP. First, we derive the notion of the augmented graph for GSTP analogous
to EDP. We then show that GSTP is FPT by (1) the tree-cut width of the
augmented graph, (2) the fracture number of the augmented graph, (3) the slim
tree-cut width of the input graph. The latter two results were previously known
for EDP; our results generalize these to GSTP and improve the running time for
the parameter fracture number. On the other hand, it was open whether EDP is
FPT parameterized by the tree-cut width of the augmented graph, despite
extensive research on the structural complexity of the problem. We settle this
question affirmatively.

### 3. [Online Bin Packing with Item Size Estimates](http://arxiv.org/pdf/2505.09321v1)

Authors: Matthias Gehnen, Andreas Usdenski

Imagine yourself moving to another place, and therefore, you need to pack all
of your belongings into moving boxes with some capacity. In the classical bin
packing model, you would try to minimize the number of boxes, knowing the exact
size of each item you want to pack. In the online bin packing problem, you need
to start packing the first item into a box, without knowing what other stuff is
upcoming.
  Both settings are somewhat unrealistic, as you are likely not willing to
measure the exact size of all your belongings before packing the first item,
but you are not completely clueless about what other stuff you have when you
start packing. In this article, we introduce the online bin packing with
estimates model, where you start packing with a rough idea about the upcoming
item sizes in mind.
  In this model, an algorithm receives a size estimate for every item in the
input list together with an accuracy factor $\delta$ in advance. Just as for
regular online bin packing the items are then presented iteratively. The actual
sizes of the items are allowed to deviate from the size estimate by a factor of
$\delta$. Once the actual size of an item is revealed the algorithm has to make
an irrevocable decision on the question where to place it. This is the first
time online bin packing is studied under this model.
  This article has three main results: First, no algorithm can achieve a
competitive ratio of less than $\frac{4}{3}$, even for an arbitrary small
factor $\delta>0$. Second, we present an algorithm that is $1.5$-competitive
for all $\delta \leq \frac{1}{35}$. Finally, we design a strategy that yields a
competitive ratio of $\frac{4}{3}$ under the assumption that not more than two
items can be placed in the same bin, which is best possible in this setting.

### 4. [A Dynamic Working Set Method for Compressed Sensing](http://arxiv.org/pdf/2505.09370v1)

Authors: Siu-Wing Cheng, Man Ting Wong

We propose a dynamic working set method (DWS) for the problem
$\min_{\mathtt{x} \in \mathbb{R}^n} \frac{1}{2}\|\mathtt{Ax}-\mathtt{b}\|^2 +
\eta\|\mathtt{x}\|_1$ that arises from compressed sensing. DWS manages the
working set while iteratively calling a regression solver to generate
progressively better solutions. Our experiments show that DWS is more efficient
than other state-of-the-art software in the context of compressed sensing.
Scale space such that $\|b\|=1$. Let $s$ be the number of non-zeros in the
unknown signal. We prove that for any given $\varepsilon > 0$, DWS reaches a
solution with an additive error $\varepsilon/\eta^2$ such that each call of the
solver uses only $O(\frac{1}{\varepsilon}s\log s \log\frac{1}{\varepsilon})$
variables, and each intermediate solution has $O(\frac{1}{\varepsilon}s\log
s\log\frac{1}{\varepsilon})$ non-zero coordinates.

### 5. [An Asymptotically Optimal Approximation Algorithm for Multiobjective Submodular Maximization at Scale](http://arxiv.org/pdf/2505.09525v2)

Authors: Fabian Spaeh, Atsushi Miyauchi

Maximizing a single submodular set function subject to a cardinality
constraint is a well-studied and central topic in combinatorial optimization.
However, finding a set that maximizes multiple functions at the same time is
much less understood, even though it is a formulation which naturally occurs in
robust maximization or problems with fairness considerations such as fair
influence maximization or fair allocation.
  In this work, we consider the problem of maximizing the minimum over many
submodular functions, which is known as multiobjective submodular maximization.
All known polynomial-time approximation algorithms either obtain a weak
approximation guarantee or rely on the evaluation of the multilinear extension.
The latter is expensive to evaluate and renders such algorithms impractical. We
bridge this gap and introduce the first scalable and practical algorithm that
obtains the best-known approximation guarantee. We furthermore introduce a
novel application fair centrality maximization and show how it can be addressed
via multiobjective submodular maximization. In our experimental evaluation, we
show that our algorithm outperforms known algorithms in terms of objective
value and running time.

### Emerging Technologies

### 1. [A Hybrid Quantum-Classical Particle-in-Cell Method for Plasma Simulations](http://arxiv.org/pdf/2505.09260v1)

Authors: Pratibha Raghupati Hegde, Paolo Marcandelli, Yuanchun He, Luca Pennati, Jeremy J. Williams, Ivy Peng, Stefano Markidis

We present a hybrid quantum-classical electrostatic Particle-in-Cell (PIC)
method, where the electrostatic field Poisson solver is implemented on a
quantum computer simulator using a hybrid classical-quantum Neural Network
(HNN) using data-driven and physics-informed learning approaches. The HNN is
trained on classical PIC simulation results and executed via a PennyLane
quantum simulator. The remaining computational steps, including particle motion
and field interpolation, are performed on a classical system. To evaluate the
accuracy and computational cost of this hybrid approach, we test the hybrid
quantum-classical electrostatic PIC against the two-stream instability, a
standard benchmark in plasma physics. Our results show that the quantum Poisson
solver achieves comparable accuracy to classical methods. It also provides
insights into the feasibility of using quantum computing and HNNs for plasma
simulations. We also discuss the computational overhead associated with current
quantum computer simulators, showing the challenges and potential advantages of
hybrid quantum-classical numerical methods.

### 2. [TensorRL-QAS: Reinforcement learning with tensor networks for scalable quantum architecture search](http://arxiv.org/pdf/2505.09371v1)

Authors: Akash Kundu, Stefano Mangini

Variational quantum algorithms hold the promise to address meaningful quantum
problems already on noisy intermediate-scale quantum hardware, but they face
the challenge of designing quantum circuits that both solve the target problem
and comply with device limitations. Quantum architecture search (QAS) automates
this design process, with reinforcement learning (RL) emerging as a promising
approach. Yet, RL-based QAS methods encounter significant scalability issues,
as computational and training costs grow rapidly with the number of qubits,
circuit depth, and noise, severely impacting performance. To address these
challenges, we introduce $\textit{TensorRL-QAS}$, a scalable framework that
combines tensor network (TN) methods with RL for designing quantum circuits. By
warm-starting the architecture search with a matrix product state approximation
of the target solution, TensorRL-QAS effectively narrows the search space to
physically meaningful circuits, accelerating convergence to the desired
solution. Tested on several quantum chemistry problems of up to 12-qubit,
TensorRL-QAS achieves up to a 10-fold reduction in CNOT count and circuit depth
compared to baseline methods, while maintaining or surpassing chemical
accuracy. It reduces function evaluations by up to 100-fold, accelerates
training episodes by up to $98\%$, and achieves up to $50\%$ success
probability for 10-qubit systems-far exceeding the $<1\%$ rates of baseline
approaches. Robustness and versatility are demonstrated both in the noiseless
and noisy scenarios, where we report a simulation of up to 8-qubit. These
advancements establish TensorRL-QAS as a promising candidate for a scalable and
efficient quantum circuit discovery protocol on near-term quantum hardware.

### 3. [Multilingual Machine Translation with Quantum Encoder Decoder Attention-based Convolutional Variational Circuits](http://arxiv.org/pdf/2505.09407v1)

Authors: Subrit Dikshit, Ritu Tiwari, Priyank Jain

Cloud-based multilingual translation services like Google Translate and
Microsoft Translator achieve state-of-the-art translation capabilities. These
services inherently use large multilingual language models such as GRU, LSTM,
BERT, GPT, T5, or similar encoder-decoder architectures with attention
mechanisms as the backbone. Also, new age natural language systems, for
instance ChatGPT and DeepSeek, have established huge potential in multiple
tasks in natural language processing. At the same time, they also possess
outstanding multilingual translation capabilities. However, these models use
the classical computing realm as a backend. QEDACVC (Quantum Encoder Decoder
Attention-based Convolutional Variational Circuits) is an alternate solution
that explores the quantum computing realm instead of the classical computing
realm to study and demonstrate multilingual machine translation. QEDACVC
introduces the quantum encoder-decoder architecture that simulates and runs on
quantum computing hardware via quantum convolution, quantum pooling, quantum
variational circuit, and quantum attention as software alterations. QEDACVC
achieves an Accuracy of 82% when trained on the OPUS dataset for English,
French, German, and Hindi corpora for multilingual translations.

### 4. [Contactless Cardiac Pulse Monitoring Using Event Cameras](http://arxiv.org/pdf/2505.09529v1)

Authors: Mohamed Moustafa, Joseph Lemley, Peter Corcoran

Time event cameras are a novel technology for recording scene information at
extremely low latency and with low power consumption. Event cameras output a
stream of events that encapsulate pixel-level light intensity changes within
the scene, capturing information with a higher dynamic range and temporal
resolution than traditional cameras. This study investigates the contact-free
reconstruction of an individual's cardiac pulse signal from time event
recording of their face using a supervised convolutional neural network (CNN)
model. An end-to-end model is trained to extract the cardiac signal from a
two-dimensional representation of the event stream, with model performance
evaluated based on the accuracy of the calculated heart rate. The experimental
results confirm that physiological cardiac information in the facial region is
effectively preserved within the event stream, showcasing the potential of this
novel sensor for remote heart rate monitoring. The model trained on event
frames achieves a root mean square error (RMSE) of 3.32 beats per minute (bpm)
compared to the RMSE of 2.92 bpm achieved by the baseline model trained on
standard camera frames. Furthermore, models trained on event frames generated
at 60 and 120 FPS outperformed the 30 FPS standard camera results, achieving an
RMSE of 2.54 and 2.13 bpm, respectively.

### Formal Languages and Automata Theory

### 1. [FocusE: A semantic extension of FocusST](http://arxiv.org/pdf/2505.09032v1)

Authors: Maria Spichkova

To analyse and verify the safety and security properties of interactive
systems, a formal specification might be necessary. There are many types of
formal languages and frameworks. The decision regarding what type of formal
specification should be applied in each particular case depends on many
factors. One of the approaches to specify interactive systems formally is to
present them as a composition of components processing data and control
streams. In this short paper, we present FocusE, a formal approach for
modelling event-based streams. The proposed approach is based on a formal
language FocusST, and can be seen as its semantic extension.

### 2. [Deterministic Suffix-reading Automata](http://arxiv.org/pdf/2505.09353v1)

Authors: R Keerthan, B Srivathsan, R Venkatesh, Sagar Verma

We introduce deterministic suffix-reading automata (DSA), a new automaton
model over finite words. Transitions in a DSA are labeled with words. From a
state, a DSA triggers an outgoing transition on seeing a word ending with the
transition's label. Therefore, rather than moving along an input word letter by
letter, a DSA can jump along blocks of letters, with each block ending in a
suitable suffix. This feature allows DSAs to recognize regular languages more
concisely, compared to DFAs. In this work, we focus on questions around finding
a minimal DSA for a regular language. The number of states is not a faithful
measure of the size of a DSA, since the transition-labels contain strings of
arbitrary length. Hence, we consider total-size (number of states + number of
edges + total length of transition-labels) as the size measure of DSAs.
  We start by formally defining the model and providing a DSA-to-DFA conversion
that allows to compare the expressiveness and succinctness of DSA with related
automata models. Our main technical contribution is a method to derive DSAs
from a given DFA: a DFA-to-DSA conversion. We make a surprising observation
that the smallest DSA derived from the canonical DFA of a regular language L
need not be a minimal DSA for L. This observation leads to a fundamental
bottleneck in deriving a minimal DSA for a regular language. In fact, we prove
that given a DFA and a number k, the problem of deciding if there exists an
equivalent DSA of total-size atmost k is NP-complete.

### 3. [Privacy-Preserving Runtime Verification](http://arxiv.org/pdf/2505.09276v1)

Authors: Thomas A. Henzinger, Mahyar Karimi, K. S. Thejaswini

Runtime verification offers scalable solutions to improve the safety and
reliability of systems. However, systems that require verification or
monitoring by a third party to ensure compliance with a specification might
contain sensitive information, causing privacy concerns when usual runtime
verification approaches are used. Privacy is compromised if protected
information about the system, or sensitive data that is processed by the
system, is revealed. In addition, revealing the specification being monitored
may undermine the essence of third-party verification.
  In this work, we propose two novel protocols for the privacy-preserving
runtime verification of systems against formal sequential specifications. In
our first protocol, the monitor verifies whether the system satisfies the
specification without learning anything else, though both parties are aware of
the specification. Our second protocol ensures that the system remains
oblivious to the monitored specification, while the monitor learns only whether
the system satisfies the specification and nothing more. Our protocols adapt
and improve existing techniques used in cryptography, and more specifically,
multi-party computation.
  The sequential specification defines the observation step of the monitor,
whose granularity depends on the situation (e.g., banks may be monitored on a
daily basis). Our protocols exchange a single message per observation step,
after an initialisation phase. This design minimises communication overhead,
enabling relatively lightweight privacy-preserving monitoring. We implement our
approach for monitoring specifications described by register automata and
evaluate it experimentally.

### Graphics

### 1. [Procedural Low-Poly Terrain Generation with Terracing for Computer Games](http://arxiv.org/pdf/2505.09350v1)

Authors: Richard Tivolt

In computer games, traditional procedural terrain generation relies on a grid
of vertices, with each point representing terrain elevation. For each square in
the grid, two triangles are created by connecting fixed vertex indices,
resulting in a continuous 3D surface. While this method is efficient for
modelling smooth terrain, the grid-like structure lacks the distinct, chaotic
appearance of low-poly objects and is not suitable to be used for our purposes.
The technique presented in this paper aims to solve the following problem:
Generate random, low-poly looking terraced terrain with different biomes and
add vegetation to create an interesting environment.

### 2. [LightLab: Controlling Light Sources in Images with Diffusion Models](http://arxiv.org/pdf/2505.09608v1)

Authors: Nadav Magar, Amir Hertz, Eric Tabellion, Yael Pritch, Alex Rav-Acha, Ariel Shamir, Yedid Hoshen

We present a simple, yet effective diffusion-based method for fine-grained,
parametric control over light sources in an image. Existing relighting methods
either rely on multiple input views to perform inverse rendering at inference
time, or fail to provide explicit control over light changes. Our method
fine-tunes a diffusion model on a small set of real raw photograph pairs,
supplemented by synthetically rendered images at scale, to elicit its
photorealistic prior for relighting. We leverage the linearity of light to
synthesize image pairs depicting controlled light changes of either a target
light source or ambient illumination. Using this data and an appropriate
fine-tuning scheme, we train a model for precise illumination changes with
explicit control over light intensity and color. Lastly, we show how our method
can achieve compelling light editing results, and outperforms existing methods
based on user preference.

### 3. [UMotion: Uncertainty-driven Human Motion Estimation from Inertial and Ultra-wideband Units](http://arxiv.org/pdf/2505.09393v1)

Authors: Huakun Liu, Hiroki Ota, Xin Wei, Yutaro Hirao, Monica Perusquia-Hernandez, Hideaki Uchiyama, Kiyoshi Kiyokawa

Sparse wearable inertial measurement units (IMUs) have gained popularity for
estimating 3D human motion. However, challenges such as pose ambiguity, data
drift, and limited adaptability to diverse bodies persist. To address these
issues, we propose UMotion, an uncertainty-driven, online fusing-all state
estimation framework for 3D human shape and pose estimation, supported by six
integrated, body-worn ultra-wideband (UWB) distance sensors with IMUs. UWB
sensors measure inter-node distances to infer spatial relationships, aiding in
resolving pose ambiguities and body shape variations when combined with
anthropometric data. Unfortunately, IMUs are prone to drift, and UWB sensors
are affected by body occlusions. Consequently, we develop a tightly coupled
Unscented Kalman Filter (UKF) framework that fuses uncertainties from sensor
data and estimated human motion based on individual body shape. The UKF
iteratively refines IMU and UWB measurements by aligning them with uncertain
human motion constraints in real-time, producing optimal estimates for each.
Experiments on both synthetic and real-world datasets demonstrate the
effectiveness of UMotion in stabilizing sensor data and the improvement over
state of the art in pose accuracy.

### Human-Computer Interaction

### 1. [Positioning Monocular Optical See Through Head Worn Displays in Glasses for Everyday Wear](http://arxiv.org/pdf/2505.09047v1)

Authors: Parth Arora, Ethan Kimmel, Katherine Huang, Tyler Kwok, Yukun Song, Sofia Vempala, Georgianna Lin, Ozan Cakmakci, Thad Starner

Head-worn displays for everyday wear in the form of regular eyeglasses are
technically feasible with recent advances in waveguide technology. One major
design decision is determining where in the user's visual field to position the
display. Centering the display in the principal point of gaze (PPOG) allows the
user to switch attentional focus between the virtual and real images quickly,
and best performance often occurs when the display is centered in PPOG or is
centered vertically below PPOG. However, these positions are often undesirable
in that they are considered interruptive or are associated with negative social
perceptions by users. Offsetting the virtual image may be preferred when tasks
involve driving, walking, or social interaction. This paper consolidates
findings from recent studies on monocular optical see-through HWDs (OST-HWDs),
focusing on potential for interruption, comfort, performance, and social
perception. For text-based tasks, which serve as a proxy for many monocular
OST-HWD tasks, we recommend a 15{\deg} horizontal field of view (FOV) with the
virtual image in the right lens vertically centered but offset to +8.7{\deg} to
+23.7{\deg} toward the ear. Glanceable content can be offset up to +30{\deg}
for short interactions.

### 2. [PLanet: Formalizing Experimental Design](http://arxiv.org/pdf/2505.09094v1)

Authors: London Bielicke, Anna Zhang, Shruti Tyagi, Emery Berger, Adam Chlipala, Eunice Jun

Carefully constructed experimental designs are essential for drawing valid,
generalizable conclusions from scientific studies. Unfortunately, experimental
design plans can be difficult to specify, communicate clearly, and relate to
alternatives. In response, we introduce a grammar of experimental design that
provides composable operators for constructing assignment procedures (e.g.,
Latin square). We implement this grammar in PLanet, a domain-specific language
(DSL) that constructs assignment plans in three stages: experimental unit
specification, trial-order construction, and order-to-unit mapping. We evaluate
PLanet's expressivity by taking a purposive sample of recent CHI and UIST
publications, representing their experiments as programs in PLanet, and
identifying ambiguities and alternatives. In our evaluation, PLanet could
express 11 out of 12 experiments found in sampled papers. Additionally, we
found that PLanet constructs helped make complex design choices explicit when
the researchers omit technical language describing their study designs.

### 3. [A Note on Semantic Diffusion](http://arxiv.org/pdf/2505.09283v1)

Authors: Alexander P. Ryjov, Alina A. Egorova

This paper provides an in-depth examination of the concept of semantic
diffusion as a complementary instrument to large language models (LLMs) for
design applications. Conventional LLMs and diffusion models fail to induce a
convergent, iterative refinement process: each invocation of the diffusion
mechanism spawns a new stochastic cycle, so successive outputs do not relate to
prior ones and convergence toward a desired design is not guaranteed. The
proposed hybrid framework - "LLM + semantic diffusion" - resolves this
limitation by enforcing an approximately convergent search procedure, thereby
formally addressing the problem of localized design refinement.

### 4. [AfforDance: Personalized AR Dance Learning System with Visual Affordance](http://arxiv.org/pdf/2505.09376v1)

Authors: Hyunyoung Han, Jongwon Jang, Kitaeg Shim, Sang Ho Yoon

We propose AfforDance, an augmented reality (AR)-based dance learning system
that generates personalized learning content and enhances learning through
visual affordances. Our system converts user-selected dance videos into
interactive learning experiences by integrating 3D reference avatars, audio
synchronization, and adaptive visual cues that guide movement execution. This
work contributes to personalized dance education by offering an adaptable,
user-centered learning interface.

### 5. [Utilization of Skin Color Change for Image-based Tactile Sensing](http://arxiv.org/pdf/2505.09402v1)

Authors: Seitaro Kaneko, Hiroki Ishizuka, Hidenori Yoshimura, Hiroyuki Kajimoto

Measurement of pressure distribution applied to a fingertip is crucial for
the teleoperation of robots and human computer interface. Previous studies have
acquired pressure distribution by affixing a sensor array to the fingertip or
by optically recording the deformation of an object. However, these existing
methods inhibit the fingertip from directly contacting the texture, and the
pressure applied to the fingertip is measured indirectly. In this study, we
propose a method to measure pressure distribution by directly touching a
transparent object, focusing on the change in skin color induced by the applied
pressure, caused by blood flow. We evaluated the relationship between pressure
and skin color change when local pressure is applied, and found a correlation
between the pressure and the color change. However, the contact area and the
color change area did not align perfectly. We further explored the factor
causing the spatial non-uniformity of the color change, by accounting for the
stress distribution using finite element analysis. These results suggest that
the proposed measurement method can be utilized to measure the internal stress
distribution, and it is anticipated to serve as a simple sensor in the field of
human computer interface.

### 6. [Card Sorting Simulator: Augmenting Design of Logical Information Architectures with Large Language Models](http://arxiv.org/pdf/2505.09478v1)

Authors: Eduard Kuric, Peter Demcak, Matus Krajcovic

Card sorting is a common ideation technique that elicits information on
users' mental organization of content and functionality by having them sort
items into categories. For more robust card sorting research, digital card
sorting tools could benefit from providing quick automated feedback. Our
objective of this research is to advance toward an instrument that applies
artificial intelligence (AI) to augment card sorting. For this purpose, we
develop the Card Sorting Simulator, a prototype tool that leverages Large
Language Models (LLMs) to generate informative categorizations of cards. To
illuminate how aligned the simulation is with card sorting by actual
participants, and to inform the instrument's design decisions, we conducted a
generalizability-focused comparative study. We obtained 28 pre-existing card
sorting studies from real practitioners, comprising 1,399 participants, along
with diverse contents and origins. With this dataset, we conducted a
comprehensive and nuanced analysis of the agreement between actual card sorting
results (clusterings of cards) and synthetic clusterings across a multitude of
LLMs and prompt designs. Mutual information scores indicate a good degree of
agreement to real result clustering, although similarity matrices also
demonstrate inconsistencies from mental models, which can be attributed to
their top-down nature. Furthermore, the number of cards or complexity of their
labels impact the accuracy of its simulation. These findings bolster the case
for AI augmentation in card sorting research as a source of meaningful
preliminary feedback and highlight the need for further study for the
development and validation of intelligent user research tools.

### 7. [Partnership through Play: Investigating How Long-Distance Couples Use Digital Games to Facilitate Intimacy](http://arxiv.org/pdf/2505.09509v1)

Authors: Nisha Devasia, Adrian Rodriguez, Logan Tuttle, Julie Kientz

Long-distance relationships (LDRs) have become more common in the last few
decades, primarily among young adults pursuing educational or employment
opportunities. A common way for couples in LDRs to spend time together is by
playing multiplayer video games, which are often a shared hobby and therefore a
preferred joint activity. However, games are relatively understudied in the
context of relational maintenance for LDRs. In this work, we used a
mixed-methods approach to collect data on the experiences of 13 couples in LDRs
who frequently play games together. We investigated different values around
various game mechanics and modalities and found significant differences in
couple play styles, and also detail how couples appropriate game mechanics to
express affection to each other virtually. We also created prototypes and
design implications based on couples' needs surrounding the lack of physical
sensation and memorabilia storage in most popular games.

### 8. [Evaluation Metrics for Misinformation Warning Interventions: Challenges and Prospects](http://arxiv.org/pdf/2505.09526v1)

Authors: Hussaini Zubairu, Abdelrahaman Abdou, Ashraf Matrawy

Misinformation has become a widespread issue in the 21st century, impacting
numerous areas of society and underscoring the need for effective intervention
strategies. Among these strategies, user-centered interventions, such as
warning systems, have shown promise in reducing the spread of misinformation.
Many studies have used various metrics to evaluate the effectiveness of these
warning interventions. However, no systematic review has thoroughly examined
these metrics in all studies. This paper provides a comprehensive review of
existing metrics for assessing the effectiveness of misinformation warnings,
categorizing them into four main groups: behavioral impact, trust and
credulity, usability, and cognitive and psychological effects. Through this
review, we identify critical challenges in measuring the effectiveness of
misinformation warnings, including inconsistent use of cognitive and
attitudinal metrics, the lack of standardized metrics for affective and
emotional impact, variations in user trust, and the need for more inclusive
warning designs. We present an overview of these metrics and propose areas for
future research.

### 9. [Beyond Likes: How Normative Feedback Complements Engagement Signals on Social Media](http://arxiv.org/pdf/2505.09583v2)

Authors: Yuchen Wu, Mingduo Zhao

Many online platforms incorporate engagement signals--such as likes and
upvotes--into their content ranking systems and interface design. These signals
are designed to boost user engagement. However, they can unintentionally
elevate content that is less inclusive and may not support normatively
desirable behavior. This issue becomes especially concerning when toxic content
correlates strongly with popularity indicators such as likes and upvotes. In
this study, we propose structured prosocial feedback as a complementary signal
to likes and upvotes--one that highlights content quality based on normative
criteria to help address the limitations of conventional engagement signals. We
begin by designing and implementing a machine learning feedback system powered
by a large language model (LLM), which evaluates user comments based on
principles of positive psychology, such as individual well-being, constructive
social media use, and character strengths. We then conduct a pre-registered
user study to examine how existing peer-based and the new expert-based feedback
interact to shape users' selection of comments in a social media setting.
Results show that peer feedback increases conformity to popularity cues, while
expert feedback shifts preferences toward normatively higher-quality content.
Moreover, incorporating expert feedback alongside peer evaluations improves
alignment with expert assessments and contributes to a less toxic community
environment. This illustrates the added value of normative cues--such as expert
scores generated by LLMs using psychological rubrics--and underscores the
potential benefits of incorporating such signals into platform feedback systems
to foster healthier online environments.

### 10. [Display Content, Display Methods and Evaluation Methods of the HCI in Explainable Recommender Systems: A Survey](http://arxiv.org/pdf/2505.09065v1)

Authors: Weiqing Li, Yue Xu, Yuefeng Li, Yinghui Huang

Explainable Recommender Systems (XRS) aim to provide users with
understandable reasons for the recommendations generated by these systems,
representing a crucial research direction in artificial intelligence (AI).
Recent research has increasingly focused on the algorithms, display, and
evaluation methodologies of XRS. While current research and reviews primarily
emphasize the algorithmic aspects, with fewer studies addressing the
Human-Computer Interaction (HCI) layer of XRS. Additionally, existing reviews
lack a unified taxonomy for XRS and there is insufficient attention given to
the emerging area of short video recommendations. In this study, we synthesize
existing literature and surveys on XRS, presenting a unified framework for its
research and development. The main contributions are as follows: 1) We adopt a
lifecycle perspective to systematically summarize the technologies and methods
used in XRS, addressing challenges posed by the diversity and complexity of
algorithmic models and explanation techniques. 2) For the first time, we
highlight the application of multimedia, particularly video-based explanations,
along with its potential, technical pathways, and challenges in XRS. 3) We
provide a structured overview of evaluation methods from both qualitative and
quantitative dimensions. These findings provide valuable insights for the
systematic design, progress, and testing of XRS.

### Information Retrieval

### 1. [Item Level Exploration Traffic Allocation in Large-scale Recommendation Systems](http://arxiv.org/pdf/2505.09033v1)

Authors: Dong Wang, Junyi Jiao, Arnab Bhadury, Yaping Zhang, Mingyan Gao

This paper contributes to addressing the item cold start problem in
large-scale recommender systems, focusing on how to efficiently gain initial
visibility for newly ingested content. We propose an exploration system
designed to efficiently allocate impressions to these fresh items. Our approach
leverages a learned probabilistic model to predict an item's discoverability,
which then informs a scalable and adaptive traffic allocation strategy. This
system intelligently distributes exploration budgets, optimizing for the
long-term benefit of the recommendation platform. The impact is a demonstrably
more efficient cold-start process, leading to a significant increase in the
discoverability of new content and ultimately enriching the item corpus
available for exploitation, as evidenced by its successful deployment in a
large-scale production environment.

### 2. [HMamba: Hyperbolic Mamba for Sequential Recommendation](http://arxiv.org/pdf/2505.09205v1)

Authors: Qianru Zhang, Honggang Wen, Wei Yuan, Crystal Chen, Menglin Yang, Siu-Ming Yiu, Hongzhi Yin

Sequential recommendation systems have become a cornerstone of personalized
services, adept at modeling the temporal evolution of user preferences by
capturing dynamic interaction sequences. Existing approaches predominantly rely
on traditional models, including RNNs and Transformers. Despite their success
in local pattern recognition, Transformer-based methods suffer from quadratic
computational complexity and a tendency toward superficial attention patterns,
limiting their ability to infer enduring preference hierarchies in sequential
recommendation data. Recent advances in Mamba-based sequential models introduce
linear-time efficiency but remain constrained by Euclidean geometry, failing to
leverage the intrinsic hyperbolic structure of recommendation data. To bridge
this gap, we propose Hyperbolic Mamba, a novel architecture that unifies the
efficiency of Mamba's selective state space mechanism with hyperbolic
geometry's hierarchical representational power. Our framework introduces (1) a
hyperbolic selective state space that maintains curvature-aware sequence
modeling and (2) stabilized Riemannian operations to enable scalable training.
Experiments across four benchmarks demonstrate that Hyperbolic Mamba achieves
3-11% improvement while retaining Mamba's linear-time efficiency, enabling
real-world deployment. This work establishes a new paradigm for efficient,
hierarchy-aware sequential modeling.

### 3. [FACTors: A New Dataset for Studying the Fact-checking Ecosystem](http://arxiv.org/pdf/2505.09414v1)

Authors: Enes Altuncu, Can Başkent, Sanjay Bhattacherjee, Shujun Li, Dwaipayan Roy

Our fight against false information is spearheaded by fact-checkers. They
investigate the veracity of claims and document their findings as fact-checking
reports. With the rapid increase in the amount of false information circulating
online, the use of automation in fact-checking processes aims to strengthen
this ecosystem by enhancing scalability. Datasets containing fact-checked
claims play a key role in developing such automated solutions. However, to the
best of our knowledge, there is no fact-checking dataset at the ecosystem
level, covering claims from a sufficiently long period of time and sourced from
a wide range of actors reflecting the entire ecosystem that admittedly follows
widely-accepted codes and principles of fact-checking. We present a new dataset
FACTors, the first to fill this gap by presenting ecosystem-level data on
fact-checking. It contains 118,112 claims from 117,993 fact-checking reports in
English (co-)authored by 1,953 individuals and published during the period of
1995-2025 by 39 fact-checking organisations that are active signatories of the
IFCN (International Fact-Checking Network) and/or EFCSN (European Fact-Checking
Standards Network). It contains 7,327 overlapping claims investigated by
multiple fact-checking organisations, corresponding to 2,977 unique claims. It
allows to conduct new ecosystem-level studies of the fact-checkers
(organisations and individuals). To demonstrate the usefulness of FACTors, we
present three example applications, including a first-of-its-kind statistical
analysis of the fact-checking ecosystem, examining the political inclinations
of the fact-checking organisations, and attempting to assign a credibility
score to each organisation based on the findings of the statistical analysis
and political leanings. Our methods for constructing FACTors are generic and
can be used to maintain a live dataset that can be updated dynamically.

### 4. [GlobalMood: A cross-cultural benchmark for music emotion recognition](http://arxiv.org/pdf/2505.09539v1)

Authors: Harin Lee, Elif Çelen, Peter Harrison, Manuel Anglada-Tort, Pol van Rijn, Minsu Park, Marc Schönwiesner, Nori Jacoby

Human annotations of mood in music are essential for music generation and
recommender systems. However, existing datasets predominantly focus on Western
songs with mood terms derived from English, which may limit generalizability
across diverse linguistic and cultural backgrounds. To address this, we
introduce `GlobalMood', a novel cross-cultural benchmark dataset comprising
1,180 songs sampled from 59 countries, with large-scale annotations collected
from 2,519 individuals across five culturally and linguistically distinct
locations: U.S., France, Mexico, S. Korea, and Egypt. Rather than imposing
predefined mood categories, we implement a bottom-up, participant-driven
approach to organically elicit culturally specific music-related mood terms. We
then recruit another pool of human participants to collect 988,925 ratings for
these culture-specific descriptors. Our analysis confirms the presence of a
valence-arousal structure shared across cultures, yet also reveals significant
divergences in how certain mood terms, despite being dictionary equivalents,
are perceived cross-culturally. State-of-the-art multimodal models benefit
substantially from fine-tuning on our cross-culturally balanced dataset, as
evidenced by improved alignment with human evaluations - particularly in
non-English contexts. More broadly, our findings inform the ongoing debate on
the universality versus cultural specificity of emotional descriptors, and our
methodology can contribute to other multimodal and cross-lingual research.

### 5. [Distance-aware Self-adaptive Graph Convolution for Fine-grained Hierarchical Recommendation](http://arxiv.org/pdf/2505.09590v1)

Authors: Tao Huang, Yihong Chen, Wei Fan, Wei Zhou, Junhao Wen

Graph Convolutional Networks (GCNs) are widely used to improve recommendation
accuracy and performance by effectively learning the representations of user
and item nodes. However, two major challenges remain: (1) the lack of further
optimization in the graph representation structure and (2) insufficient
attention given to the varying contributions of different convolutional
layers.This paper proposes SAGCN, a distance-based adaptive hierarchical
aggregation method that refines the aggregation process through differentiated
representation metrics. SAGCN introduces a detailed approach to multilayer
information aggregation and representation space optimization, enabling the
model to learn hierarchical embedding weights based on the distance between
hierarchical representations. This innovation allows for more precise
cross-layer information aggregation, improves the model's ability to capture
hierarchical embeddings, and optimizes the representation space structure.
Additionally, the objective loss function is refined to better align with
recommendation tasks.Extensive experiments conducted on four real-world
datasets demonstrate significant improvements, including over a 5% increase on
Yelp and a 5.58% increase in Recall@10 on the ML_1M dataset.

### 6. [Display Content, Display Methods and Evaluation Methods of the HCI in Explainable Recommender Systems: A Survey](http://arxiv.org/pdf/2505.09065v1)

Authors: Weiqing Li, Yue Xu, Yuefeng Li, Yinghui Huang

Explainable Recommender Systems (XRS) aim to provide users with
understandable reasons for the recommendations generated by these systems,
representing a crucial research direction in artificial intelligence (AI).
Recent research has increasingly focused on the algorithms, display, and
evaluation methodologies of XRS. While current research and reviews primarily
emphasize the algorithmic aspects, with fewer studies addressing the
Human-Computer Interaction (HCI) layer of XRS. Additionally, existing reviews
lack a unified taxonomy for XRS and there is insufficient attention given to
the emerging area of short video recommendations. In this study, we synthesize
existing literature and surveys on XRS, presenting a unified framework for its
research and development. The main contributions are as follows: 1) We adopt a
lifecycle perspective to systematically summarize the technologies and methods
used in XRS, addressing challenges posed by the diversity and complexity of
algorithmic models and explanation techniques. 2) For the first time, we
highlight the application of multimedia, particularly video-based explanations,
along with its potential, technical pathways, and challenges in XRS. 3) We
provide a structured overview of evaluation methods from both qualitative and
quantitative dimensions. These findings provide valuable insights for the
systematic design, progress, and testing of XRS.

### 7. [Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging](http://arxiv.org/pdf/2505.09316v1)

Authors: Hongjin Qian, Zheng Liu

Augmenting large language models (LLMs) with external retrieval has become a
standard method to address their inherent knowledge cutoff limitations.
However, traditional retrieval-augmented generation methods employ static,
pre-inference retrieval strategies, making them inadequate for complex tasks
involving ambiguous, multi-step, or evolving information needs. Recent advances
in test-time scaling techniques have demonstrated significant potential in
enabling LLMs to dynamically interact with external tools, motivating the shift
toward adaptive inference-time retrieval. Inspired by Information Foraging
Theory (IFT), we propose InForage, a reinforcement learning framework that
formalizes retrieval-augmented reasoning as a dynamic information-seeking
process. Unlike existing approaches, InForage explicitly rewards intermediate
retrieval quality, encouraging LLMs to iteratively gather and integrate
information through adaptive search behaviors. To facilitate training, we
construct a human-guided dataset capturing iterative search and reasoning
trajectories for complex, real-world web tasks. Extensive evaluations across
general question answering, multi-hop reasoning tasks, and a newly developed
real-time web QA dataset demonstrate InForage's superior performance over
baseline methods. These results highlight InForage's effectiveness in building
robust, adaptive, and efficient reasoning agents.

### 8. [Focus, Merge, Rank: Improved Question Answering Based on Semi-structured Knowledge Bases](http://arxiv.org/pdf/2505.09246v1)

Authors: Derian Boer, Stephen Roth, Stefan Kramer

In many real-world settings, machine learning models and interactive systems
have access to both structured knowledge, e.g., knowledge graphs or tables, and
unstructured content, e.g., natural language documents. However, most rely on
either. Semi-Structured Knowledge Bases (SKBs) bridge this gap by linking
unstructured content to nodes within structured data, thereby enabling new
strategies for knowledge access and use. In this work, we present
FocusedRetriever, a modular SKB-based framework for multi-hop question
answering. It integrates components (VSS-based entity search, LLM-based
generation of Cypher queries and pairwise re-ranking) in a way that enables it
to outperform state-of-the-art methods across all three STaRK benchmark test
sets, covering diverse domains and multiple performance metrics. The average
first-hit rate exceeds that of the second-best method by 25.7%.
FocusedRetriever leverages (1) the capacity of Large Language Models (LLMs) to
extract relational facts and entity attributes from unstructured text, (2) node
set joins to filter answer candidates based on these extracted triplets and
constraints, (3) vector similarity search to retrieve and rank relevant
unstructured content, and (4) the contextual capabilities of LLMs to finally
rank the top-k answers. For generality, we only incorporate base LLMs in
FocusedRetriever in our evaluation. However, our analysis of intermediate
results highlights several opportunities for further upgrades including
finetuning. The source code is publicly available at
https://github.com/kramerlab/FocusedRetriever .

### 9. [Diffusion Recommender Models and the Illusion of Progress: A Concerning Study of Reproducibility and a Conceptual Mismatch](http://arxiv.org/pdf/2505.09364v1)

Authors: Michael Benigni, Maurizio Ferrari Dacrema, Dietmar Jannach

Countless new machine learning models are published every year and are
reported to significantly advance the state-of-the-art in \emph{top-n}
recommendation. However, earlier reproducibility studies indicate that progress
in this area may be quite limited. Specifically, various widespread
methodological issues, e.g., comparisons with untuned baseline models, have led
to an \emph{illusion of progress}. In this work, our goal is to examine whether
these problems persist in today's research. To this end, we aim to reproduce
the latest advancements reported from applying modern Denoising Diffusion
Probabilistic Models to recommender systems, focusing on four models published
at the top-ranked SIGIR conference in 2023 and 2024. Our findings are
concerning, revealing persistent methodological problems. Alarmingly, through
experiments, we find that the latest recommendation techniques based on
diffusion models, despite their computational complexity and substantial carbon
footprint, are consistently outperformed by simpler existing models.
Furthermore, we identify key mismatches between the characteristics of
diffusion models and those of the traditional \emph{top-n} recommendation task,
raising doubts about their suitability for recommendation. We also note that,
in the papers we analyze, the generative capabilities of these models are
constrained to a minimum. Overall, our results and continued methodological
issues call for greater scientific rigor and a disruptive change in the
research and publication culture in this area.

### 10. [CXMArena: Unified Dataset to benchmark performance in realistic CXM Scenarios](http://arxiv.org/pdf/2505.09436v1)

Authors: Raghav Garg, Kapil Sharma, Karan Gupta

Large Language Models (LLMs) hold immense potential for revolutionizing
Customer Experience Management (CXM), particularly in contact center
operations. However, evaluating their practical utility in complex operational
environments is hindered by data scarcity (due to privacy concerns) and the
limitations of current benchmarks. Existing benchmarks often lack realism,
failing to incorporate deep knowledge base (KB) integration, real-world noise,
or critical operational tasks beyond conversational fluency. To bridge this
gap, we introduce CXMArena, a novel, large-scale synthetic benchmark dataset
specifically designed for evaluating AI in operational CXM contexts. Given the
diversity in possible contact center features, we have developed a scalable
LLM-powered pipeline that simulates the brand's CXM entities that form the
foundation of our datasets-such as knowledge articles including product
specifications, issue taxonomies, and contact center conversations. The
entities closely represent real-world distribution because of controlled noise
injection (informed by domain experts) and rigorous automated validation.
Building on this, we release CXMArena, which provides dedicated benchmarks
targeting five important operational tasks: Knowledge Base Refinement, Intent
Prediction, Agent Quality Adherence, Article Search, and Multi-turn RAG with
Integrated Tools. Our baseline experiments underscore the benchmark's
difficulty: even state of the art embedding and generation models achieve only
68% accuracy on article search, while standard embedding methods yield a low F1
score of 0.3 for knowledge base refinement, highlighting significant challenges
for current models necessitating complex pipelines and solutions over
conventional techniques.

### Machine Learning

### 1. [Generating time-consistent dynamics with discriminator-guided image diffusion models](http://arxiv.org/pdf/2505.09089v2)

Authors: Philipp Hess, Maximilian Gelbrecht, Christof Schötz, Michael Aich, Yu Huang, Shangshang Yang, Niklas Boers

Realistic temporal dynamics are crucial for many video generation, processing
and modelling applications, e.g. in computational fluid dynamics, weather
prediction, or long-term climate simulations. Video diffusion models (VDMs) are
the current state-of-the-art method for generating highly realistic dynamics.
However, training VDMs from scratch can be challenging and requires large
computational resources, limiting their wider application. Here, we propose a
time-consistency discriminator that enables pretrained image diffusion models
to generate realistic spatiotemporal dynamics. The discriminator guides the
sampling inference process and does not require extensions or finetuning of the
image diffusion model. We compare our approach against a VDM trained from
scratch on an idealized turbulence simulation and a real-world global
precipitation dataset. Our approach performs equally well in terms of temporal
consistency, shows improved uncertainty calibration and lower biases compared
to the VDM, and achieves stable centennial-scale climate simulations at daily
time steps.

### 2. [Argus: Federated Non-convex Bilevel Learning over 6G Space-Air-Ground Integrated Network](http://arxiv.org/pdf/2505.09106v1)

Authors: Ya Liu, Kai Yang, Yu Zhu, Keying Yang, Haibo Zhao

The space-air-ground integrated network (SAGIN) has recently emerged as a
core element in the 6G networks. However, traditional centralized and
synchronous optimization algorithms are unsuitable for SAGIN due to
infrastructureless and time-varying environments. This paper aims to develop a
novel Asynchronous algorithm a.k.a. Argus for tackling non-convex and
non-smooth decentralized federated bilevel learning over SAGIN. The proposed
algorithm allows networked agents (e.g. autonomous aerial vehicles) to tackle
bilevel learning problems in time-varying networks asynchronously, thereby
averting stragglers from impeding the overall training speed. We provide a
theoretical analysis of the iteration complexity, communication complexity, and
computational complexity of Argus. Its effectiveness is further demonstrated
through numerical experiments.

### 3. [The Larger the Merrier? Efficient Large AI Model Inference in Wireless Edge Networks](http://arxiv.org/pdf/2505.09214v1)

Authors: Zhonghao Lyu, Ming Xiao, Jie Xu, Mikael Skoglund, Marco Di Renzo

The growing demand for large artificial intelligence model (LAIM) services is
driving a paradigm shift from traditional cloud-based inference to edge-based
inference for low-latency, privacy-preserving applications. In particular,
edge-device co-inference, which partitions LAIMs between edge devices and
servers, has emerged as a promising strategy for resource-efficient LAIM
execution in wireless networks. In this paper, we investigate a pruning-aware
LAIM co-inference scheme, where a pre-trained LAIM is pruned and partitioned
into on-device and on-server sub-models for deployment. For analysis, we first
prove that the LAIM output distortion is upper bounded by its parameter
distortion. Then, we derive a lower bound on parameter distortion via
rate-distortion theory, analytically capturing the relationship between pruning
ratio and co-inference performance. Next, based on the analytical results, we
formulate an LAIM co-inference distortion bound minimization problem by jointly
optimizing the pruning ratio, transmit power, and computation frequency under
system latency, energy, and available resource constraints. Moreover, we
propose an efficient algorithm to tackle the considered highly non-convex
problem. Finally, extensive simulations demonstrate the effectiveness of the
proposed design. In particular, model parameter distortion is shown to provide
a reliable bound on output distortion. Also, the proposed joint pruning ratio
and resource management design achieves superior performance in balancing
trade-offs among inference performance, system latency, and energy consumption
compared with benchmark schemes, such as fully on-device and on-server
inference. Moreover, the split point is shown to play a critical role in system
performance optimization under heterogeneous and resource-limited edge
environments.

### 4. [Stable and Convexified Information Bottleneck Optimization via Symbolic Continuation and Entropy-Regularized Trajectories](http://arxiv.org/pdf/2505.09239v1)

Authors: Faruk Alpay

The Information Bottleneck (IB) method frequently suffers from unstable
optimization, characterized by abrupt representation shifts near critical
points of the IB trade-off parameter, beta. In this paper, I introduce a novel
approach to achieve stable and convex IB optimization through symbolic
continuation and entropy-regularized trajectories. I analytically prove
convexity and uniqueness of the IB solution path when an entropy regularization
term is included, and demonstrate how this stabilizes representation learning
across a wide range of \b{eta} values. Additionally, I provide extensive
sensitivity analyses around critical points (beta) with statistically robust
uncertainty quantification (95% confidence intervals). The open-source
implementation, experimental results, and reproducibility framework included in
this work offer a clear path for practical deployment and future extension of
my proposed method.

### 5. [On the Learning with Augmented Class via Forests](http://arxiv.org/pdf/2505.09294v1)

Authors: Fan Xu, Wuyang Chen, Wei Gao

Decision trees and forests have achieved successes in various real
applications, most working with all testing classes known in training data. In
this work, we focus on learning with augmented class via forests, where an
augmented class may appear in testing data yet not in training data. We
incorporate information of augmented class into trees' splitting, i.e., a new
splitting criterion, called augmented Gini impurity, is introduced to exploit
some unlabeled data from testing distribution. We then develop the approach
named Learning with Augmented Class via Forests (LACForest), which constructs
shallow forests based on the augmented Gini impurity and then splits forests
with pseudo-labeled augmented instances for better performance. We also develop
deep neural forests with a novel optimization objective based on our augmented
Gini impurity, so as to utilize the representation power of neural networks for
forests. Theoretically, we present the convergence analysis for augmented Gini
impurity, and finally conduct experiments to verify the effectiveness of our
approaches. The code is available at https://github.com/nju-xuf/LACForest/.

### 6. [Neural Multivariate Regression: Qualitative Insights from the Unconstrained Feature Model](http://arxiv.org/pdf/2505.09308v1)

Authors: George Andriopoulos, Soyuj Jung Basnet, Juan Guevara, Li Guo, Keith Ross

The Unconstrained Feature Model (UFM) is a mathematical framework that
enables closed-form approximations for minimal training loss and related
performance measures in deep neural networks (DNNs). This paper leverages the
UFM to provide qualitative insights into neural multivariate regression, a
critical task in imitation learning, robotics, and reinforcement learning.
Specifically, we address two key questions: (1) How do multi-task models
compare to multiple single-task models in terms of training performance? (2)
Can whitening and normalizing regression targets improve training performance?
The UFM theory predicts that multi-task models achieve strictly smaller
training MSE than multiple single-task models when the same or stronger
regularization is applied to the latter, and our empirical results confirm
these findings. Regarding whitening and normalizing regression targets, the UFM
theory predicts that they reduce training MSE when the average variance across
the target dimensions is less than one, and our empirical results once again
confirm these findings. These findings highlight the UFM as a powerful
framework for deriving actionable insights into DNN design and data
pre-processing strategies.

### 7. [MUST: Multi-Scale Structural-Temporal Link Prediction Model for UAV Ad Hoc Networks](http://arxiv.org/pdf/2505.09331v1)

Authors: Cunlai Pu, Fangrui Wu, Rajput Ramiz Sharafat, Guangzhao Dai, Xiangbo Shu

Link prediction in unmanned aerial vehicle (UAV) ad hoc networks (UANETs)
aims to predict the potential formation of future links between UAVs. In
adversarial environments where the route information of UAVs is unavailable,
predicting future links must rely solely on the observed historical topological
information of UANETs. However, the highly dynamic and sparse nature of UANET
topologies presents substantial challenges in effectively capturing meaningful
structural and temporal patterns for accurate link prediction. Most existing
link prediction methods focus on temporal dynamics at a single structural scale
while neglecting the effects of sparsity, resulting in insufficient information
capture and limited applicability to UANETs. In this paper, we propose a
multi-scale structural-temporal link prediction model (MUST) for UANETs.
Specifically, we first employ graph attention networks (GATs) to capture
structural features at multiple levels, including the individual UAV level, the
UAV community level, and the overall network level. Then, we use long
short-term memory (LSTM) networks to learn the temporal dynamics of these
multi-scale structural features. Additionally, we address the impact of
sparsity by introducing a sophisticated loss function during model
optimization. We validate the performance of MUST using several UANET datasets
generated through simulations. Extensive experimental results demonstrate that
MUST achieves state-of-the-art link prediction performance in highly dynamic
and sparse UANETs.

### 8. [Exploiting the Potential Supervision Information of Clean Samples in Partial Label Learning](http://arxiv.org/pdf/2505.09354v1)

Authors: Guangtai Wang, Chi-Man Vong, Jintao Huang

Diminishing the impact of false-positive labels is critical for conducting
disambiguation in partial label learning. However, the existing disambiguation
strategies mainly focus on exploiting the characteristics of individual partial
label instances while neglecting the strong supervision information of clean
samples randomly lying in the datasets. In this work, we show that clean
samples can be collected to offer guidance and enhance the confidence of the
most possible candidates. Motivated by the manner of the differentiable count
loss strat- egy and the K-Nearest-Neighbor algorithm, we proposed a new
calibration strategy called CleanSE. Specifically, we attribute the most
reliable candidates with higher significance under the assumption that for each
clean sample, if its label is one of the candidates of its nearest neighbor in
the representation space, it is more likely to be the ground truth of its
neighbor. Moreover, clean samples offer help in characterizing the sample
distributions by restricting the label counts of each label to a specific
interval. Extensive experiments on 3 synthetic benchmarks and 5 real-world PLL
datasets showed this calibration strategy can be applied to most of the
state-of-the-art PLL methods as well as enhance their performance.

### 9. [Efficient Mixed Precision Quantization in Graph Neural Networks](http://arxiv.org/pdf/2505.09361v1)

Authors: Samir Moustafa, Nils M. Kriege, Wilfried N. Gansterer

Graph Neural Networks (GNNs) have become essential for handling large-scale
graph applications. However, the computational demands of GNNs necessitate the
development of efficient methods to accelerate inference. Mixed precision
quantization emerges as a promising solution to enhance the efficiency of GNN
architectures without compromising prediction performance. Compared to
conventional deep learning architectures, GNN layers contain a wider set of
components that can be quantized, including message passing functions,
aggregation functions, update functions, the inputs, learnable parameters, and
outputs of these functions. In this paper, we introduce a theorem for efficient
quantized message passing to aggregate integer messages. It guarantees
numerical equality of the aggregated messages using integer values with respect
to those obtained with full (FP32) precision. Based on this theorem, we
introduce the Mixed Precision Quantization for GNN (MixQ-GNN) framework, which
flexibly selects effective integer bit-widths for all components within GNN
layers. Our approach systematically navigates the wide set of possible
bit-width combinations, addressing the challenge of optimizing efficiency while
aiming at maintaining comparable prediction performance. MixQ-GNN integrates
with existing GNN quantization methods, utilizing their graph structure
advantages to achieve higher prediction performance. On average, MixQ-GNN
achieved reductions in bit operations of 5.5x for node classification and 5.1x
for graph classification compared to architectures represented in FP32
precision.

### 10. [Personalized Control for Lower Limb Prosthesis Using Kolmogorov-Arnold Networks](http://arxiv.org/pdf/2505.09366v1)

Authors: SeyedMojtaba Mohasel, Alireza Afzal Aghaei, Corey Pew

Objective: This paper investigates the potential of learnable activation
functions in Kolmogorov-Arnold Networks (KANs) for personalized control in a
lower-limb prosthesis. In addition, user-specific vs. pooled training data is
evaluated to improve machine learning (ML) and Deep Learning (DL) performance
for turn intent prediction.
  Method: Inertial measurement unit (IMU) data from the shank were collected
from five individuals with lower-limb amputation performing turning tasks in a
laboratory setting. Ability to classify an upcoming turn was evaluated for
Multilayer Perceptron (MLP), Kolmogorov-Arnold Network (KAN), convolutional
neural network (CNN), and fractional Kolmogorov-Arnold Networks (FKAN). The
comparison of MLP and KAN (for ML models) and FKAN and CNN (for DL models)
assessed the effectiveness of learnable activation functions. Models were
trained separately on user-specific and pooled data to evaluate the impact of
training data on their performance.
  Results: Learnable activation functions in KAN and FKAN did not yield
significant improvement compared to MLP and CNN, respectively. Training on
user-specific data yielded superior results compared to pooled data for ML
models ($p < 0.05$). In contrast, no significant difference was observed
between user-specific and pooled training for DL models.
  Significance: These findings suggest that learnable activation functions may
demonstrate distinct advantages in datasets involving more complex tasks and
larger volumes. In addition, pooled training showed comparable performance to
user-specific training in DL models, indicating that model training for
prosthesis control can utilize data from multiple participants.

### Neural and Evolutionary Computing

### 1. [A Standardized Benchmark Set of Clustering Problem Instances for Comparing Black-Box Optimizers](http://arxiv.org/pdf/2505.09233v1)

Authors: Diederick Vermetten, Catalin-Viorel Dinu, Marcus Gallagher

One key challenge in optimization is the selection of a suitable set of
benchmark problems. A common goal is to find functions which are representative
of a class of real-world optimization problems in order to ensure findings on
the benchmarks will translate to relevant problem domains. While some problem
characteristics are well-covered by popular benchmarking suites, others are
often overlooked. One example of such a problem characteristic is permutation
invariance, where the search space consists of a set of symmetrical search
regions. This type of problem occurs e.g. when a set of solutions has to be
found, but the ordering within this set does not matter. The data clustering
problem, often seen in machine learning contexts, is a clear example of such an
optimization landscape, and has thus been proposed as a base from which
optimization benchmarks can be created. In addition to the symmetry aspect,
these clustering problems also contain potential regions of neutrality, which
can provide an additional challenge to optimization algorithms. In this paper,
we present a standardized benchmark suite for the evaluation of continuous
black-box optimization algorithms, based on data clustering problems. To gain
insight into the diversity of the benchmark set, both internally and in
comparison to existing suites, we perform a benchmarking study of a set of
modular CMA-ES configurations, as well as an analysis using exploratory
landscape analysis. Our benchmark set is open-source and integrated with the
IOHprofiler benchmarking framework to encourage its use in future research.

### 2. [Equilibrio de carga para transformadores de distribución eléctrica mejorando la calidad de servicio en fin de línea](http://arxiv.org/pdf/2505.09235v1)

Authors: Juan M. Bordón, Victor A. Jimenez, Adrian Will

The distribution of electrical energy faces global challenges, such as
increasing demand, the integration of distributed generation, high energy
losses, and the need to improve service quality. In particular, load
imbalance-where loads are not evenly distributed across the circuit phase-can
reduce efficiency, shorten equipment lifespan, and increase susceptibility to
service interruptions. While methods that involve shifting loads from one phase
to another can be costly, they are effective when smart meters are available
and implemented efficiently. This work proposes the use of genetic algorithms
to optimally identify which loads should be reassigned in order to improve both
phase balance and voltage quality at the end nodes of the network while
minimizing the number of required changes. The algorithm was evaluated through
simulations using PandaPower, a power flow analysis tool, modeling simple
networks based on real-world characteristics of the electrical system in
Tucum\'an.

### 3. [Diffusion Recommender Models and the Illusion of Progress: A Concerning Study of Reproducibility and a Conceptual Mismatch](http://arxiv.org/pdf/2505.09364v1)

Authors: Michael Benigni, Maurizio Ferrari Dacrema, Dietmar Jannach

Countless new machine learning models are published every year and are
reported to significantly advance the state-of-the-art in \emph{top-n}
recommendation. However, earlier reproducibility studies indicate that progress
in this area may be quite limited. Specifically, various widespread
methodological issues, e.g., comparisons with untuned baseline models, have led
to an \emph{illusion of progress}. In this work, our goal is to examine whether
these problems persist in today's research. To this end, we aim to reproduce
the latest advancements reported from applying modern Denoising Diffusion
Probabilistic Models to recommender systems, focusing on four models published
at the top-ranked SIGIR conference in 2023 and 2024. Our findings are
concerning, revealing persistent methodological problems. Alarmingly, through
experiments, we find that the latest recommendation techniques based on
diffusion models, despite their computational complexity and substantial carbon
footprint, are consistently outperformed by simpler existing models.
Furthermore, we identify key mismatches between the characteristics of
diffusion models and those of the traditional \emph{top-n} recommendation task,
raising doubts about their suitability for recommendation. We also note that,
in the papers we analyze, the generative capabilities of these models are
constrained to a minimum. Overall, our results and continued methodological
issues call for greater scientific rigor and a disruptive change in the
research and publication culture in this area.

### Networking and Internet Architecture

### 1. [QUIC Steps: Evaluating Pacing Strategies in QUIC Implementations](http://arxiv.org/pdf/2505.09222v1)

Authors: Marcel Kempf, Simon Tietz, Benedikt Jaeger, Johannes Späth, Georg Carle, Johannes Zirngibl

Pacing is a key mechanism in modern transport protocols, used to regulate
packet transmission timing to minimize traffic burstiness, lower latency, and
reduce packet loss. Standardized in 2021, QUIC is a UDP-based protocol designed
to improve upon the TCP / TLS stack. While the QUIC protocol recommends pacing,
and congestion control algorithms like BBR rely on it, the user-space nature of
QUIC introduces unique challenges. These challenges include coarse-grained
timers, system call overhead, and OS scheduling delays, all of which complicate
precise packet pacing. This paper investigates how pacing is implemented
differently across QUIC stacks, including quiche, picoquic, and ngtcp2, and
evaluates the impact of system-level features like GSO and Linux qdiscs on
pacing. Using a custom measurement framework and a passive optical fiber tap,
we establish a baseline with default settings and systematically explore the
effects of qdiscs, hardware offloading using the ETF qdisc, and GSO on pacing
precision and network performance. We also extend and evaluate a kernel patch
to enable pacing of individual packets within GSO buffers, combining batching
efficiency with precise pacing. Kernel-assisted and purely user-space pacing
approaches are compared. We show that pacing with only user-space timers can
work well, as demonstrated by picoquic with BBR. With quiche, we identify FQ as
a qdisc well-suited for pacing QUIC traffic, as it is relatively easy to use
and offers precise pacing based on packet timestamps. Our findings provide new
insights into the trade-offs involved in implementing pacing in QUIC and
highlight potential optimizations for real-world applications like video
streaming and video calls.

### 2. [Interplay Between AI and Space-Air-Ground Integrated Network: The Road Ahead](http://arxiv.org/pdf/2505.09259v1)

Authors: Chenyu Wu, Xi Wang, Yi Hu, Shuai Han, Dusit Niyato

Space-air-ground integrated network (SAGIN) is envisioned as a key network
architecture for achieving ubiquitous coverage in the next-generation
communication system. Concurrently, artificial intelligence (AI) plays a
pivotal role in managing the complex control of SAGIN, thereby enhancing its
automation and flexibility. Despite this, there remains a significant research
gap concerning the interaction between AI and SAGIN. In this context, we first
present a promising approach for developing a generalized AI model capable of
executing multiple tasks simultaneously in SAGIN. Subsequently, we propose a
framework that leverages software-defined networking (SDN) and AI technologies
to manage the resources and services across the entire SAGIN. Particularly, we
demonstrate the real-world applicability of our proposed framework through a
comprehensive case study. These works pave the way for the deep integration of
SAGIN and AI in future wireless networks.

### 3. [RAG-Enabled Intent Reasoning for Application-Network Interaction](http://arxiv.org/pdf/2505.09339v1)

Authors: Salwa Mostafa, Mohamed K. Abdel-Aziz, Mohammed S. Elbamby, Mehdi Bennis

Intent-based network (IBN) is a promising solution to automate network
operation and management. IBN aims to offer human-tailored network interaction,
allowing the network to communicate in a way that aligns with the network
users' language, rather than requiring the network users to understand the
technical language of the network/devices. Nowadays, different applications
interact with the network, each with its own specialized needs and domain
language. Creating semantic languages (i.e., ontology-based languages) and
associating them with each application to facilitate intent translation lacks
technical expertise and is neither practical nor scalable. To tackle the
aforementioned problem, we propose a context-aware AI framework that utilizes
machine reasoning (MR), retrieval augmented generation (RAG), and generative AI
technologies to interpret intents from different applications and generate
structured network intents. The proposed framework allows for
generalized/domain-specific intent expression and overcomes the drawbacks of
large language models (LLMs) and vanilla-RAG framework. The experimental
results show that our proposed intent-RAG framework outperforms the LLM and
vanilla-RAG framework in intent translation.

### 4. [Dimensioning and Optimization of Reliability Coverage in Local 6G Networks](http://arxiv.org/pdf/2505.09440v1)

Authors: Jacek Kibiłda, Dian Echevarría Pérez, André Gomes, Onel L. Alcaraz López, Arthur S. de Sena, Nurul Huda Mahmood, Hirley Alves

Enabling vertical use cases for the sixth generation (6G) wireless networks,
such as automated manufacturing, immersive extended reality (XR), and
self-driving fleets, will require network designs that meet reliability and
latency targets in well-defined service areas. In order to establish a
quantifiable design objective, we introduce the novel concept of reliability
coverage, defined as the percentage area covered by communication services
operating under well-defined reliability and performance targets. Reliability
coverage allows us to unify the different network design tasks occurring at
different time scales, namely resource orchestration and allocation, resulting
in a single framework for dimensioning and optimization in local 6G networks.
The two time scales, when considered together, yield remarkably consistent
results and allow us to observe how stringent reliability/latency requirements
translate into the increased wireless network resource demands.

### 5. [DNS Query Forgery: A Client-Side Defense Against Mobile App Traffic Profiling](http://arxiv.org/pdf/2505.09374v1)

Authors: Andrea Jimenez-Berenguel, César Gil, Carlos Garcia-Rubio, Jordi Forné, Celeste Campo

Mobile applications continuously generate DNS queries that can reveal
sensitive user behavioral patterns even when communications are encrypted. This
paper presents a privacy enhancement framework based on query forgery to
protect users against profiling attempts that leverage these background
communications. We first mathematically model user profiles as probability
distributions over interest categories derived from mobile application traffic.
We then evaluate three query forgery strategies -- uniform sampling,
TrackMeNot-based generation, and an optimized approach that minimizes
Kullback-Leibler divergence -- to quantify their effectiveness in obfuscating
user profiles. Then we create a synthetic dataset comprising 1,000 user traces
constructed from real mobile application traffic and we extract the user
profiles based on DNS traffic. Our evaluation reveals that a 50\% privacy
improvement is achievable with less than 20\% traffic overhead when using our
approach, while achieving 100\% privacy protection requires approximately
40-60\% additional traffic. We further propose a modular system architecture
for practical implementation of our protection mechanisms on mobile devices.
This work offers a client-side privacy solution that operates without
third-party trust requirements, empowering individual users to defend against
traffic analysis without compromising application functionality.

### 6. [Instant AoI Optimization through Relay Location Selection in Disaster Multi-hop Communication](http://arxiv.org/pdf/2505.09386v1)

Authors: Yang Gao, Zezhi Zeng

Meteorological disasters such as typhoons, forest fires, and floods can
damage the communication infrastructures, which will further disable the
communication capabilities of cellular networks. The multi-hop wireless
communication based on IoT devices (e.g., rescue robots, UAVs, and mobile
devices) becomes an available and rapidly deployable communication approach for
search and rescue operations. However, Age of Information (AoI), an emerging
network performance metric, has not been comprehensively investigated in this
multi-hop model. In this paper, we first construct a UAV-relayed wireless
network model and formulate the end-to-end instant AoI. Then we derive the
optimal location of the relay UAV to achieve the minimum instant AoI by
mathematical analysis. Simulations show that the derived relay location can
always guarantee the optimal AoI and outperform other schemes.

### 7. [Wormhole Detection Based on Z-Score And Neighbor Table Comparison](http://arxiv.org/pdf/2505.09405v1)

Authors: Zezhi Zeng

Wormhole attacks can cause serious disruptions to the network topology in
disaster rescue opportunity networks.
  By establishing false Wormhole(WH) links, malicious nodes can mislead
legitimate paths in the network, further causing serious consequences such as
traffic analysis attacks (i.e., by eavesdropping and monitoring exchanged
traffic), denial of service (DoS) or selective packet loss attacks. This paper
uses rescue equipment (vehicle-mounted base stations, rescue control centers,
etc.) as an effective third-party auditor (TPA), and combines the commonly used
Z-Score (Standard Score) data processing method to propose a new detection
method based on pure mathematical statistics for detecting wormhole attacks.
Finally, we perform a large number of simulations to evaluate the proposed
method. Since our proposed strategy does not require auxiliary equipment such
as GPS positioning and timers, as a pure data statistical analysis method, it
is obviously more economically valuable, feasible, and practical than other
strategies in disaster relief.

### 8. [MDTP -- An Adaptive Multi-Source Data Transfer Protocol](http://arxiv.org/pdf/2505.09597v1)

Authors: Sepideh Abdollah, Craig Partridge, Susmit Shannigrahi

Scientific data volume is growing in size, and as a direct result, the need
for faster transfers is also increasing. The scientific community has sought to
leverage parallel transfer methods using multi-threaded and multi-source
download models to reduce download times. In multi-source transfers, a client
downloads data from multiple replicated servers in parallel. Tools such as
Aria2 and BitTorrent support such multi-source transfers and have shown
improved transfer times.
  In this work, we introduce Multi-Source Data Transfer Protocol, MDTP, which
further improves multi-source transfer performance. MDTP logically divides a
file request into smaller chunk requests and distributes the chunk requests
across multiple servers. Chunk sizes are adapted based on each server's
performance but selected in a way that ensures each round of requests completes
around the same time. We formulate this chunk-size allocation problem as a
variant of the bin-packing problem, where adaptive chunking efficiently fills
the available capacity "bins" corresponding to each server.
  Our evaluation shows that MDTP reduces transfer times by 10-22% compared to
Aria2, the fastest alternative. Comparisons with other protocols, such as
static chunking and BitTorrent, demonstrate even greater improvements.
Additionally, we show that MDTP distributes load proportionally across all
available replicas, not just the fastest ones, which improves throughput.
Finally, we show MDTP maintains high throughput even when latency increases or
bandwidth to the fastest server decreases.

### Robotics

### 1. [Deployable and Generalizable Motion Prediction: Taxonomy, Open Challenges and Future Directions](http://arxiv.org/pdf/2505.09074v1)

Authors: Letian Wang, Marc-Antoine Lavoie, Sandro Papais, Barza Nisar, Yuxiao Chen, Wenhao Ding, Boris Ivanovic, Hao Shao, Abulikemu Abuduweili, Evan Cook, Yang Zhou, Peter Karkus, Jiachen Li, Changliu Liu, Marco Pavone, Steven Waslander

Motion prediction, the anticipation of future agent states or scene
evolution, is rooted in human cognition, bridging perception and
decision-making. It enables intelligent systems, such as robots and
self-driving cars, to act safely in dynamic, human-involved environments, and
informs broader time-series reasoning challenges. With advances in methods,
representations, and datasets, the field has seen rapid progress, reflected in
quickly evolving benchmark results. Yet, when state-of-the-art methods are
deployed in the real world, they often struggle to generalize to open-world
conditions and fall short of deployment standards. This reveals a gap between
research benchmarks, which are often idealized or ill-posed, and real-world
complexity.
  To address this gap, this survey revisits the generalization and
deployability of motion prediction models, with an emphasis on the applications
of robotics, autonomous driving, and human motion. We first offer a
comprehensive taxonomy of motion prediction methods, covering representations,
modeling strategies, application domains, and evaluation protocols. We then
study two key challenges: (1) how to push motion prediction models to be
deployable to realistic deployment standards, where motion prediction does not
act in a vacuum, but functions as one module of closed-loop autonomy stacks -
it takes input from the localization and perception, and informs downstream
planning and control. 2) how to generalize motion prediction models from
limited seen scenarios/datasets to the open-world settings. Throughout the
paper, we highlight critical open challenges to guide future work, aiming to
recalibrate the community's efforts, fostering progress that is not only
measurable but also meaningful for real-world applications.

### 2. [VGC-RIO: A Tightly Integrated Radar-Inertial Odometry with Spatial Weighted Doppler Velocity and Local Geometric Constrained RCS Histograms](http://arxiv.org/pdf/2505.09103v2)

Authors: Jianguang Xiang, Xiaofeng He, Zizhuo Chen, Lilian Zhang, Xincan Luo, Jun Mao

Recent advances in 4D radar-inertial odometry have demonstrated promising
potential for autonomous lo calization in adverse conditions. However,
effective handling of sparse and noisy radar measurements remains a critical
challenge. In this paper, we propose a radar-inertial odometry with a spatial
weighting method that adapts to unevenly distributed points and a novel
point-description histogram for challenging point registration. To make full
use of the Doppler velocity from different spatial sections, we propose a
weighting calculation model. To enhance the point cloud registration
performance under challenging scenarios, we con struct a novel point histogram
descriptor that combines local geometric features and radar cross-section (RCS)
features. We have also conducted extensive experiments on both public and
self-constructed datasets. The results demonstrate the precision and robustness
of the proposed VGC-RIO.

### 3. [Latent Theory of Mind: A Decentralized Diffusion Architecture for Cooperative Manipulation](http://arxiv.org/pdf/2505.09144v1)

Authors: Chengyang He, Gadiel Sznaier Camps, Xu Liu, Mac Schwager, Guillaume Sartoretti

We present Latent Theory of Mind (LatentToM), a decentralized diffusion
policy architecture for collaborative robot manipulation. Our policy allows
multiple manipulators with their own perception and computation to collaborate
with each other towards a common task goal with or without explicit
communication. Our key innovation lies in allowing each agent to maintain two
latent representations: an ego embedding specific to the robot, and a consensus
embedding trained to be common to both robots, despite their different sensor
streams and poses. We further let each robot train a decoder to infer the other
robot's ego embedding from their consensus embedding, akin to theory of mind in
latent space. Training occurs centrally, with all the policies' consensus
encoders supervised by a loss inspired by sheaf theory, a mathematical theory
for clustering data on a topological manifold. Specifically, we introduce a
first-order cohomology loss to enforce sheaf-consistent alignment of the
consensus embeddings. To preserve the expressiveness of the consensus
embedding, we further propose structural constraints based on theory of mind
and a directional consensus mechanism. Execution can be fully distributed,
requiring no explicit communication between policies. In which case, the
information is exchanged implicitly through each robot's sensor stream by
observing the actions of the other robots and their effects on the scene.
Alternatively, execution can leverage direct communication to share the robots'
consensus embeddings, where the embeddings are shared once during each
inference step and are aligned using the sheaf Laplacian. In our hardware
experiments, LatentToM outperforms a naive decentralized diffusion baseline,
and shows comparable performance with a state-of-the-art centralized diffusion
policy for bi-manual manipulation. Project website:
https://stanfordmsl.github.io/LatentToM/.

### 4. [A drone that learns to efficiently find objects in agricultural fields: from simulation to the real world](http://arxiv.org/pdf/2505.09278v1)

Authors: Rick van Essen, Gert Kootstra

Drones are promising for data collection in precision agriculture, however,
they are limited by their battery capacity. Efficient path planners are
therefore required. This paper presents a drone path planner trained using
Reinforcement Learning (RL) on an abstract simulation that uses object
detections and uncertain prior knowledge. The RL agent controls the flight
direction and can terminate the flight. By using the agent in combination with
the drone's flight controller and a detection network to process camera images,
it is possible to evaluate the performance of the agent on real-world data. In
simulation, the agent yielded on average a 78% shorter flight path compared to
a full coverage planner, at the cost of a 14% lower recall. On real-world data,
the agent showed a 72% shorter flight path compared to a full coverage planner,
however, at the cost of a 25% lower recall. The lower performance on real-world
data was attributed to the real-world object distribution and the lower
accuracy of prior knowledge, and shows potential for improvement. Overall, we
concluded that for applications where it is not crucial to find all objects,
such as weed detection, the learned-based path planner is suitable and
efficient.

### 5. [Embodied Intelligent Industrial Robotics: Concepts and Techniques](http://arxiv.org/pdf/2505.09305v2)

Authors: Chaoran Zhang, Chenhao Zhang, Zhaobo Xu, Qinghongbing Xie, Pingfa Feng, Long Zeng

In recent years, embodied intelligent robotics (EIR) has made significant
progress in multi-modal perception, autonomous decision-making, and physical
interaction. Some robots have already been tested in general-purpose scenarios
such as homes and shopping malls. We aim to advance the research and
application of embodied intelligence in industrial scenes. However, current EIR
lacks a deep understanding of industrial environment semantics and the
normative constraints between industrial operating objects. To address this
gap, this paper first reviews the history of industrial robotics and the
mainstream EIR frameworks. We then introduce the concept of the embodied
intelligent industrial robotics (EIIR) and propose a knowledge-driven EIIR
technology framework for industrial environments. The framework includes four
main modules: world model, high-level task planner, low-level skill controller,
and simulator. We also review the current development of technologies related
to each module and highlight recent progress in adapting them to industrial
applications. Finally, we summarize the key challenges EIIR faces in industrial
scenarios and suggest future research directions. We believe that EIIR
technology will shape the next generation of industrial robotics. Industrial
systems based on embodied intelligent industrial robots offer strong potential
for enabling intelligent manufacturing. We will continue to track and summarize
new research in this area and hope this review will serve as a valuable
reference for scholars and engineers interested in industrial embodied
intelligence. Together, we can help drive the rapid advancement and application
of this technology. The associated project can be found at
https://github.com/jackyzengl/EIIR.

### 6. [Strategic Jenga Play via Graph Based Dynamics Modeling](http://arxiv.org/pdf/2505.09377v1)

Authors: Kavya Puthuveetil, Xinyi Zhang, Kazuto Yokoyama, Tetsuya Narita

Controlled manipulation of multiple objects whose dynamics are closely linked
is a challenging problem within contact-rich manipulation, requiring an
understanding of how the movement of one will impact the others. Using the
Jenga game as a testbed to explore this problem, we graph-based modeling to
tackle two different aspects of the task: 1) block selection and 2) block
extraction. For block selection, we construct graphs of the Jenga tower and
attempt to classify, based on the tower's structure, whether removing a given
block will cause the tower to collapse. For block extraction, we train a
dynamics model that predicts how all the blocks in the tower will move at each
timestep in an extraction trajectory, which we then use in a sampling-based
model predictive control loop to safely pull blocks out of the tower with a
general-purpose parallel-jaw gripper. We train and evaluate our methods in
simulation, demonstrating promising results towards block selection and block
extraction on a challenging set of full-sized Jenga towers, even at advanced
stages of the game.

### 7. [Exploring Pose-Guided Imitation Learning for Robotic Precise Insertion](http://arxiv.org/pdf/2505.09424v1)

Authors: Han Sun, Yizhao Wang, Zhenning Zhou, Shuai Wang, Haibo Yang, Jingyuan Sun, Qixin Cao

Recent studies have proved that imitation learning shows strong potential in
the field of robotic manipulation. However, existing methods still struggle
with precision manipulation task and rely on inefficient image/point cloud
observations. In this paper, we explore to introduce SE(3) object pose into
imitation learning and propose the pose-guided efficient imitation learning
methods for robotic precise insertion task. First, we propose a precise
insertion diffusion policy which utilizes the relative SE(3) pose as the
observation-action pair. The policy models the source object SE(3) pose
trajectory relative to the target object. Second, we explore to introduce the
RGBD data to the pose-guided diffusion policy. Specifically, we design a
goal-conditioned RGBD encoder to capture the discrepancy between the current
state and the goal state. In addition, a pose-guided residual gated fusion
method is proposed, which takes pose features as the backbone, and the RGBD
features selectively compensate for pose feature deficiencies through an
adaptive gating mechanism. Our methods are evaluated on 6 robotic precise
insertion tasks, demonstrating competitive performance with only 7-10
demonstrations. Experiments demonstrate that the proposed methods can
successfully complete precision insertion tasks with a clearance of about 0.01
mm. Experimental results highlight its superior efficiency and generalization
capability compared to existing baselines. Code will be available at
https://github.com/sunhan1997/PoseInsert.

### 8. [aUToPath: Unified Planning and Control for Autonomous Vehicles in Urban Environments Using Hybrid Lattice and Free-Space Search](http://arxiv.org/pdf/2505.09475v1)

Authors: Tanmay P. Patel, Connor Wilson, Ellina R. Zhang, Morgan Tran, Chang Keun Paik, Steven L. Waslander, Timothy D. Barfoot

This paper presents aUToPath, a unified online framework for global
path-planning and control to address the challenge of autonomous navigation in
cluttered urban environments. A key component of our framework is a novel
hybrid planner that combines pre-computed lattice maps with dynamic free-space
sampling to efficiently generate optimal driveable corridors in cluttered
scenarios. Our system also features sequential convex programming (SCP)-based
model predictive control (MPC) to refine the corridors into smooth, dynamically
consistent trajectories. A single optimization problem is used to both generate
a trajectory and its corresponding control commands; this addresses limitations
of decoupled approaches by guaranteeing a safe and feasible path. Simulation
results of the novel planner on randomly generated obstacle-rich scenarios
demonstrate the success rate of a free-space Adaptively Informed Trees*
(AIT*)-based planner, and runtimes comparable to a lattice-based planner.
Real-world experiments of the full system on a Chevrolet Bolt EUV further
validate performance in dense obstacle fields, demonstrating no violations of
traffic, kinematic, or vehicle constraints, and a 100% success rate across
eight trials.

### 9. [VTLA: Vision-Tactile-Language-Action Model with Preference Learning for Insertion Manipulation](http://arxiv.org/pdf/2505.09577v1)

Authors: Chaofan Zhang, Peng Hao, Xiaoge Cao, Xiaoshuai Hao, Shaowei Cui, Shuo Wang

While vision-language models have advanced significantly, their application
in language-conditioned robotic manipulation is still underexplored, especially
for contact-rich tasks that extend beyond visually dominant pick-and-place
scenarios. To bridge this gap, we introduce Vision-Tactile-Language-Action
model, a novel framework that enables robust policy generation in
contact-intensive scenarios by effectively integrating visual and tactile
inputs through cross-modal language grounding. A low-cost, multi-modal dataset
has been constructed in a simulation environment, containing
vision-tactile-action-instruction pairs specifically designed for the fingertip
insertion task. Furthermore, we introduce Direct Preference Optimization (DPO)
to offer regression-like supervision for the VTLA model, effectively bridging
the gap between classification-based next token prediction loss and continuous
robotic tasks. Experimental results show that the VTLA model outperforms
traditional imitation learning methods (e.g., diffusion policies) and existing
multi-modal baselines (TLA/VLA), achieving over 90% success rates on unseen peg
shapes. Finally, we conduct real-world peg-in-hole experiments to demonstrate
the exceptional Sim2Real performance of the proposed VTLA model. For
supplementary videos and results, please visit our project website:
https://sites.google.com/view/vtla

### 10. [Real2Render2Real: Scaling Robot Data Without Dynamics Simulation or Robot Hardware](http://arxiv.org/pdf/2505.09601v1)

Authors: Justin Yu, Letian Fu, Huang Huang, Karim El-Refai, Rares Andrei Ambrus, Richard Cheng, Muhammad Zubair Irshad, Ken Goldberg

Scaling robot learning requires vast and diverse datasets. Yet the prevailing
data collection paradigm-human teleoperation-remains costly and constrained by
manual effort and physical robot access. We introduce Real2Render2Real (R2R2R),
a novel approach for generating robot training data without relying on object
dynamics simulation or teleoperation of robot hardware. The input is a
smartphone-captured scan of one or more objects and a single video of a human
demonstration. R2R2R renders thousands of high visual fidelity robot-agnostic
demonstrations by reconstructing detailed 3D object geometry and appearance,
and tracking 6-DoF object motion. R2R2R uses 3D Gaussian Splatting (3DGS) to
enable flexible asset generation and trajectory synthesis for both rigid and
articulated objects, converting these representations to meshes to maintain
compatibility with scalable rendering engines like IsaacLab but with collision
modeling off. Robot demonstration data generated by R2R2R integrates directly
with models that operate on robot proprioceptive states and image observations,
such as vision-language-action models (VLA) and imitation learning policies.
Physical experiments suggest that models trained on R2R2R data from a single
human demonstration can match the performance of models trained on 150 human
teleoperation demonstrations. Project page: https://real2render2real.com

### Software Engineering

### 1. [Evaluating Mutation-based Fault Localization for Quantum Programs](http://arxiv.org/pdf/2505.09059v1)

Authors: Yuta Ishimoto, Masanari Kondo, Naoyasu Ubayashi, Yasutaka Kamei, Ryota Katsube, Naoto Sato, Hideto Ogawa

Quantum computers leverage the principles of quantum mechanics to execute
operations. They require quantum programs that define operations on quantum
bits (qubits), the fundamental units of computation. Unlike traditional
software development, the process of creating and debugging quantum programs
requires specialized knowledge of quantum computation, making the development
process more challenging. In this paper, we apply and evaluate mutation-based
fault localization (MBFL) for quantum programs with the aim of enhancing
debugging efficiency. We use quantum mutation operations, which are
specifically designed for quantum programs, to identify faults. Our evaluation
involves 23 real-world faults and 305 artificially induced faults in quantum
programs developed with Qiskit(R). The results show that real-world faults are
more challenging for MBFL than artificial faults. In fact, the median EXAM
score, which represents the percentage of the code examined before locating the
faulty statement (lower is better), is 1.2% for artificial benchmark and 19.4%
for the real-world benchmark in the worst-case scenario. Our study highlights
the potential and limitations of MBFL for quantum programs, considering
different fault types and mutation operation types. Finally, we discuss future
directions for improving MBFL in the context of quantum programming.

### 2. [Mitigating Configuration Differences Between Development and Production Environments: A Catalog of Strategies](http://arxiv.org/pdf/2505.09392v2)

Authors: Marcos Nazario, Rodrigo Bonifacio, Gustavo Pinto

Context: The Configuration Management of the development and production
environments is an important aspect of IT operations. However, managing the
configuration differences between these two environments can be challenging,
leading to inconsistent behavior, unexpected errors, and increased downtime.
Objective: In this study, we sought to investigate the strategies software
companies employ to mitigate the configuration differences between the
development and production environments. Our goal is to provide a comprehensive
understanding of these strategies used to contribute to reducing the risk of
configuration-related issues. Method: To achieve this goal, we interviewed 17
participants and leveraged the Thematic Analysis methodology to analyze the
interview data. These participants shed some light on the current practices,
processes, challenges, or issues they have encountered. Results: Based on the
interviews, we systematically formulated and structured a catalog of eight
strategies that explain how software producing companies mitigate these
configuration differences. These strategies vary from 1) creating detailed
configuration management plans, 2) using automation tools, and 3) developing
processes to test and validate changes through containers and virtualization
technologies. Conclusion: By implementing these strategies, companies can
improve their ability to respond quickly and effectively to changes in the
production environment. In addition, they can also ensure compliance with
industry standards and regulations.

### 3. [The Silent Scientist: When Software Research Fails to Reach Its Audience](http://arxiv.org/pdf/2505.09541v1)

Authors: Marvin Wyrich, Christof Tinnes, Sebastian Baltes, Sven Apel

If software research were a performance, it would be a thoughtful theater
play -- full of rich content but confined to the traditional stage of academic
publishing. Meanwhile, its potential audience is immersed in engaging on-demand
experiences, leaving the theater half-empty, and the research findings lost in
the wings. As long as this remains the case, discussions about research
relevance and impact lack meaningful context.

### 4. [MIGRATION-BENCH: Repository-Level Code Migration Benchmark from Java 8](http://arxiv.org/pdf/2505.09569v1)

Authors: Linbo Liu, Xinle Liu, Qiang Zhou, Lin Chen, Yihan Liu, Hoan Nguyen, Behrooz Omidvar-Tehrani, Xi Shen, Jun Huan, Omer Tripp, Anoop Deoras

With the rapid advancement of powerful large language models (LLMs) in recent
years, a wide range of software engineering tasks can now be addressed using
LLMs, significantly enhancing productivity and scalability. Numerous benchmark
datasets have been developed to evaluate the coding capabilities of these
models, while they primarily focus on problem-solving and issue-resolution
tasks. In contrast, we introduce a new coding benchmark MIGRATION-BENCH with a
distinct focus: code migration. MIGRATION-BENCH aims to serve as a
comprehensive benchmark for migration from Java 8 to the latest long-term
support (LTS) versions (Java 17, 21), MIGRATION-BENCH includes a full dataset
and its subset selected with $5,102$ and $300$ repositories respectively.
Selected is a representative subset curated for complexity and difficulty,
offering a versatile resource to support research in the field of code
migration. Additionally, we provide a comprehensive evaluation framework to
facilitate rigorous and standardized assessment of LLMs on this challenging
task. We further propose SD-Feedback and demonstrate that LLMs can effectively
tackle repository-level code migration to Java 17. For the selected subset with
Claude-3.5-Sonnet-v2, SD-Feedback achieves 62.33% and 27.00% success rate
(pass@1) for minimal and maximal migration respectively. The benchmark dataset
and source code are available at:
https://huggingface.co/collections/AmazonScience and
https://github.com/amazon-science/self_debug respectively.

### 5. [A Method for Assisting Novices Creating Class Diagrams Based on the Instructor's Class Layout](http://arxiv.org/pdf/2505.09116v1)

Authors: Yuta Saito, Takehiro Kokubu, Takafumi Tanaka, Atsuo Hazeyama, Hiroaki Hashiura

Nowadays, modeling exercises on software development objects are conducted in
higher education institutions for information technology. Not only are there
many defects such as missing elements in the models created by learners during
the exercises, but the layout of elements in the class diagrams often differs
significantly from the correct answers created by the instructors. In this
paper, we focus on the above problem and propose a method to provide effective
support to learners during modeling exercises by automatically converting the
layout of the learner's class diagram to that of the instructor, in addition to
indicating the correctness of the artifacts to the learners during the
exercises. The proposed method was implemented and evaluated as a tool, and the
results indicate that the automatic layout conversion was an effective feedback
to the learners.

### 6. [Variational Prefix Tuning for Diverse and Accurate Code Summarization Using Pre-trained Language Models](http://arxiv.org/pdf/2505.09062v1)

Authors: Junda Zhao, Yuliang Song, Eldan Cohen

Recent advancements in source code summarization have leveraged
transformer-based pre-trained models, including Large Language Models of Code
(LLMCs), to automate and improve the generation of code summaries. However,
existing methods often focus on generating a single high-quality summary for a
given source code, neglecting scenarios where the generated summary might be
inadequate and alternative options are needed. In this paper, we introduce
Variational Prefix Tuning (VPT), a novel approach that enhances pre-trained
models' ability to generate diverse yet accurate sets of summaries, allowing
the user to choose the most suitable one for the given source code. Our method
integrates a Conditional Variational Autoencoder (CVAE) framework as a modular
component into pre-trained models, enabling us to model the distribution of
observed target summaries and sample continuous embeddings to be used as
prefixes to steer the generation of diverse outputs during decoding.
Importantly, we construct our method in a parameter-efficient manner,
eliminating the need for expensive model retraining, especially when using
LLMCs. Furthermore, we employ a bi-criteria reranking method to select a subset
of generated summaries, optimizing both the diversity and the accuracy of the
options presented to users. We present extensive experimental evaluations using
widely used datasets and current state-of-the-art pre-trained code
summarization models to demonstrate the effectiveness of our approach and its
adaptability across models.

### 7. [Customizing a Large Language Model for VHDL Design of High-Performance Microprocessors](http://arxiv.org/pdf/2505.09610v1)

Authors: Nicolas Dupuis, Ravi Nair, Shyam Ramji, Sean McClintock, Nishant Chauhan, Priyanka Nagpal, Bart Blaner, Ken Valk, Leon Stok, Ruchir Puri

The use of Large Language Models (LLMs) in hardware design has taken off in
recent years, principally through its incorporation in tools that increase chip
designer productivity. There has been considerable discussion about the use of
LLMs in RTL specifications of chip designs, for which the two most popular
languages are Verilog and VHDL. LLMs and their use in Verilog design has
received significant attention due to the higher popularity of the language,
but little attention so far has been given to VHDL despite its continued
popularity in the industry. There has also been little discussion about the
unique needs of organizations that engage in high-performance processor design,
and techniques to deploy AI solutions in these settings. In this paper, we
describe our journey in developing a Large Language Model (LLM) specifically
for the purpose of explaining VHDL code, a task that has particular importance
in an organization with decades of experience and assets in high-performance
processor design. We show how we developed test sets specific to our needs and
used them for evaluating models as we performed extended pretraining (EPT) of a
base LLM. Expert evaluation of the code explanations produced by the EPT model
increased to 69% compared to a base model rating of 43%. We further show how we
developed an LLM-as-a-judge to gauge models similar to expert evaluators. This
led us to deriving and evaluating a host of new models, including an
instruction-tuned version of the EPT model with an expected expert evaluator
rating of 71%. Our experiments also indicate that with the potential use of
newer base models, this rating can be pushed to 85% and beyond. We conclude
with a discussion on further improving the quality of hardware design LLMs
using exciting new developments in the Generative AI world.

### Social and Information Networks

### 1. [Spatial public goods games with queueing and reputation](http://arxiv.org/pdf/2505.09154v1)

Authors: Gui Zhang, Xiaojin Xiong, Bin Pin, Minyu Feng, Matjaž Perc

In real-world social and economic systems, the provisioning of public goods
generally entails continuous interactions among individuals, with decisions to
cooperate or defect being influenced by dynamic factors such as timing,
resource availability, and the duration of engagement. However, the traditional
public goods game ignores the asynchrony of the strategy adopted by players in
the game. To address this problem, we propose a spatial public goods game that
integrates an M/M/1 queueing system to simulate the dynamic flow of player
interactions. We use a birth-death process to characterize the stochastic
dynamics of this queueing system, with players arriving following a Poisson
process and service times being exponentially distributed under a
first-come-first-served basis with finite queue capacity. We also incorporate
reputation so that players who have cooperated in the past are more likely to
be chosen for future interactions. Our research shows that a high arrival rate,
low service rate, and the reputation mechanism jointly facilitate the emergence
of cooperative individuals in the network, which thus provides an interesting
and new perspective for the provisioning of public goods.

### 2. [Moving towards informative and actionable social media research](http://arxiv.org/pdf/2505.09254v1)

Authors: Joseph B. Bak-Coleman, Stephan Lewandowsky, Philipp Lorenz-Spreen, Arvind Narayanan, Amy Orben, Lisa Oswald

Social media is nearly ubiquitous in modern life, and concerns have been
raised about its putative societal impacts, ranging from undermining mental
health and exacerbating polarization to fomenting violence and disrupting
democracy. Despite extensive research, consensus on these effects remains
elusive, with observational studies often highlighting concerns while
randomized controlled trials (RCTs) yield conflicting or null findings. This
review examines how the complexity inherent in social systems can account for
such discrepancies, emphasizing that emergent societal and long-term outcomes
cannot be readily inferred from individual-level effects. In complex systems,
such as social networks, feedback loops, hysteresis, multi-scale dynamics, and
non-linearity limit the utility of approaches for assessing causality that are
otherwise robust in simpler contexts. Revisiting large-scale experiments, we
explore how null or conflicting findings may reflect these complexities rather
than a true absence of effects. Even in cases where the methods are
appropriate, assessing the net impacts of social media provides little
actionable insight given that eliminating social media is not a realistic
option for whole populations. We argue that progress will require a
complexity-minded approach focused on specific design choices of online
platforms that triangulates experimental, observational and theoretical
methods.

### 3. [Instant AoI Optimization through Relay Location Selection in Disaster Multi-hop Communication](http://arxiv.org/pdf/2505.09386v1)

Authors: Yang Gao, Zezhi Zeng

Meteorological disasters such as typhoons, forest fires, and floods can
damage the communication infrastructures, which will further disable the
communication capabilities of cellular networks. The multi-hop wireless
communication based on IoT devices (e.g., rescue robots, UAVs, and mobile
devices) becomes an available and rapidly deployable communication approach for
search and rescue operations. However, Age of Information (AoI), an emerging
network performance metric, has not been comprehensively investigated in this
multi-hop model. In this paper, we first construct a UAV-relayed wireless
network model and formulate the end-to-end instant AoI. Then we derive the
optimal location of the relay UAV to achieve the minimum instant AoI by
mathematical analysis. Simulations show that the derived relay location can
always guarantee the optimal AoI and outperform other schemes.

### 4. [Wormhole Detection Based on Z-Score And Neighbor Table Comparison](http://arxiv.org/pdf/2505.09405v1)

Authors: Zezhi Zeng

Wormhole attacks can cause serious disruptions to the network topology in
disaster rescue opportunity networks.
  By establishing false Wormhole(WH) links, malicious nodes can mislead
legitimate paths in the network, further causing serious consequences such as
traffic analysis attacks (i.e., by eavesdropping and monitoring exchanged
traffic), denial of service (DoS) or selective packet loss attacks. This paper
uses rescue equipment (vehicle-mounted base stations, rescue control centers,
etc.) as an effective third-party auditor (TPA), and combines the commonly used
Z-Score (Standard Score) data processing method to propose a new detection
method based on pure mathematical statistics for detecting wormhole attacks.
Finally, we perform a large number of simulations to evaluate the proposed
method. Since our proposed strategy does not require auxiliary equipment such
as GPS positioning and timers, as a pure data statistical analysis method, it
is obviously more economically valuable, feasible, and practical than other
strategies in disaster relief.

### 5. [An Asymptotically Optimal Approximation Algorithm for Multiobjective Submodular Maximization at Scale](http://arxiv.org/pdf/2505.09525v2)

Authors: Fabian Spaeh, Atsushi Miyauchi

Maximizing a single submodular set function subject to a cardinality
constraint is a well-studied and central topic in combinatorial optimization.
However, finding a set that maximizes multiple functions at the same time is
much less understood, even though it is a formulation which naturally occurs in
robust maximization or problems with fairness considerations such as fair
influence maximization or fair allocation.
  In this work, we consider the problem of maximizing the minimum over many
submodular functions, which is known as multiobjective submodular maximization.
All known polynomial-time approximation algorithms either obtain a weak
approximation guarantee or rely on the evaluation of the multilinear extension.
The latter is expensive to evaluate and renders such algorithms impractical. We
bridge this gap and introduce the first scalable and practical algorithm that
obtains the best-known approximation guarantee. We furthermore introduce a
novel application fair centrality maximization and show how it can be addressed
via multiobjective submodular maximization. In our experimental evaluation, we
show that our algorithm outperforms known algorithms in terms of objective
value and running time.

### 6. [SALM: A Multi-Agent Framework for Language Model-Driven Social Network Simulation](http://arxiv.org/pdf/2505.09081v1)

Authors: Gaurav Koley

Contemporary approaches to agent-based modeling (ABM) of social systems have
traditionally emphasized rule-based behaviors, limiting their ability to
capture nuanced dynamics by moving beyond predefined rules and leveraging
contextual understanding from LMs of human social interaction. This paper
presents SALM (Social Agent LM Framework), a novel approach for integrating
language models (LMs) into social network simulation that achieves
unprecedented temporal stability in multi-agent scenarios. Our primary
contributions include: (1) a hierarchical prompting architecture enabling
stable simulation beyond 4,000 timesteps while reducing token usage by 73%, (2)
an attention-based memory system achieving 80% cache hit rates (95% CI [78%,
82%]) with sub-linear memory growth of 9.5%, and (3) formal bounds on
personality stability. Through extensive validation against SNAP ego networks,
we demonstrate the first LLM-based framework capable of modeling long-term
social phenomena while maintaining empirically validated behavioral fidelity.

### 7. [The Niche Connectivity Paradox: Multichrome Contagions Overcome Vaccine Hesitancy more effectively than Monochromacy](http://arxiv.org/pdf/2505.09605v1)

Authors: Ho-Chun Herbert Chang, Feng Fu

The rise of vaccine hesitancy has caused a resurgence of vaccine-preventable
diseases such as measles and pertussis, alongside widespread skepticism and
refusals of COVID-19 vaccinations. While categorizing individuals as either
supportive of or opposed to vaccines provides a convenient dichotomy of vaccine
attitudes, vaccine hesitancy is far more complex and dynamic. It involves
wavering individuals whose attitudes fluctuate -- those who may exhibit
pro-vaccine attitudes at one time and anti-vaccine attitudes at another. Here,
we identify and analyze multichrome contagions as potential targets for
intervention by leveraging a dataset of known pro-vax and anti-vax Twitter
users ($n =135$ million) and a large COVID-19 Twitter dataset ($n = 3.5$
billion; including close analysis of $1,563,472$ unique individuals). We
reconstruct an evolving multiplex sentiment landscape using top co-spreading
issues, characterizing them as monochrome and multichrome contagions, based on
their conceptual overlap with vaccination. We demonstrate switchers as
deliberative: they are more moderate, engage with a wider range of topics, and
occupy more central positions in their networks. Further examination of their
information consumption shows that their discourse often engages with
progressive issues such as climate change, which can serve as avenues for
multichrome contagion interventions to promote pro-vaccine attitudes. Using
data-driven intervention simulations, we demonstrate a paradox of niche
connectivity, where multichrome contagions with fragmented, non-overlapping
communities generate the highest levels of diffusion for pro-vaccine attitudes.
Our work offers insights into harnessing synergistic hitchhiking effect of
multichrome contagions to drive desired attitude and behavior changes in
network-based interventions, particularly for overcoming vaccine hesitancy.

### Systems and Control

### 1. [Leveraging Offline Data from Similar Systems for Online Linear Quadratic Control](http://arxiv.org/pdf/2505.09057v1)

Authors: Shivam Bajaj, Prateek Jaiswal, Vijay Gupta

``Sim2real gap", in which the system learned in simulations is not the exact
representation of the real system, can lead to loss of stability and
performance when controllers learned using data from the simulated system are
used on the real system. In this work, we address this challenge in the linear
quadratic regulator (LQR) setting. Specifically, we consider an LQR problem for
a system with unknown system matrices. Along with the state-action pairs from
the system to be controlled, a trajectory of length $S$ of state-action pairs
from a different unknown system is available. Our proposed algorithm is
constructed upon Thompson sampling and utilizes the mean as well as the
uncertainty of the dynamics of the system from which the trajectory of length
$S$ is obtained. We establish that the algorithm achieves
$\tilde{\mathcal{O}}({f(S,M_{\delta})\sqrt{T/S}})$ Bayes regret after $T$ time
steps, where $M_{\delta}$ characterizes the \emph{dissimilarity} between the
two systems and $f(S,M_{\delta})$ is a function of $S$ and $M_{\delta}$. When
$M_{\delta}$ is sufficiently small, the proposed algorithm achieves
$\tilde{\mathcal{O}}({\sqrt{T/S}})$ Bayes regret and outperforms a naive
strategy which does not utilize the available trajectory.

### 2. [Data-driven Internal Model Control for Output Regulation](http://arxiv.org/pdf/2505.09255v1)

Authors: Wenjie Liu, Yifei Li, Jian Sun, Gang Wang, Keyou You, Lihua Xie, Jie Chen

Output regulation is a fundamental problem in control theory, extensively
studied since the 1970s. Traditionally, research has primarily addressed
scenarios where the system model is explicitly known, leaving the problem in
the absence of a system model less explored. Leveraging the recent advancements
in Willems et al.'s fundamental lemma, data-driven control has emerged as a
powerful tool for stabilizing unknown systems. This paper tackles the output
regulation problem for unknown single and multi-agent systems (MASs) using
noisy data. Previous approaches have attempted to solve data-based output
regulation equations (OREs), which are inadequate for achieving zero tracking
error with noisy data. To circumvent the need for solving data-based OREs, we
propose an internal model-based data-driven controller that reformulates the
output regulation problem into a stabilization problem. This method is first
applied to linear time-invariant (LTI) systems, demonstrating exact solution
capabilities, i.e., zero tracking error, through solving a straightforward
data-based linear matrix inequality (LMI). Furthermore, we extend our approach
to solve the $k$th-order output regulation problem for nonlinear systems.
Extensions to both linear and nonlinear MASs are discussed. Finally, numerical
tests validate the effectiveness and correctness of the proposed controllers.

### 3. [Coordinated Multi-Valve Disturbance-Rejection Pressure Control for High-Altitude Test Stands via Exterior Penalty Functions](http://arxiv.org/pdf/2505.09352v1)

Authors: Zhang Louyue, Li Xin, Zhai Chao, Shi Duoqi, Zhang Hehong, Dan Zhihong, Wang Xi, Liu Jiashuai, Xiao Gaoxi

High altitude simulation test benches for aero engines employ multi chamber,
multi valve intake systems that demand effective decoupling and strong
disturbance rejection during transient tests. This paper proposes a coordinated
active disturbance rejection control (ADRC) scheme based on an external penalty
function. The chamber pressure safety limit is reformulated as an inequality
constrained optimization problem, and an exponential penalty together with a
gradient based algorithm is designed for dynamic constraint relaxation, with
global convergence rigorously proven. A coordination term is then integrated
into a distributed ADRC framework to yield a multi valve coordinated LADRC
controller, whose asymptotic stability is established via Lyapunov theory.
Hardware in the loop simulations using MATLAB/Simulink and a PLC demonstrate
that, under $\pm$3 kPa pressure constraints, chamber V2's maximum error is
1.782 kPa (77.1\% lower than PID control), and under a 180 kg/s^2 flow rate
disturbance, valve oscillations decrease from $\pm$27\% to $\pm$5\% (an 81.5\%
reduction). These results confirm the proposed method's superior disturbance
rejection and decoupling performance.

### 4. [Regulation without calibration](http://arxiv.org/pdf/2505.09515v1)

Authors: Rodolphe Sepulchre, Alessandro Cecconi, Michelangelo Bin, Lorenzo Marconi

This article revisits the importance of the internal model principle in the
literature of regulation and synchronization. Trajectory regulation, the task
of regulating continuous-time signals generated by differential equations, is
contrasted with event regulation, the task of only regulating discrete events
associated with the trajectories. In trajectory regulation, the internal model
principle requires an exact internal generator of the continuous-time
trajectories, which translates into unrealistic calibration requirements. Event
regulation is envisioned as a way to relieve calibration of the continuous
behavior while ensuring reliability of the discrete events.

### 5. [Reach-Avoid-Stabilize Using Admissible Control Sets](http://arxiv.org/pdf/2505.09058v1)

Authors: Zheng Gong, Boyang Li, Sylvia Herbert

Hamilton-Jacobi Reachability (HJR) analysis has been successfully used in
many robotics and control tasks, and is especially effective in computing
reach-avoid sets and control laws that enable an agent to reach a goal while
satisfying state constraints. However, the original HJR formulation provides no
guarantees of safety after a) the prescribed time horizon, or b) goal
satisfaction. The reach-avoid-stabilize (RAS) problem has therefore gained a
lot of focus: find the set of initial states (the RAS set), such that the
trajectory can reach the target, and stabilize to some point of interest (POI)
while avoiding obstacles. Solving RAS problems using HJR usually requires
defining a new value function, whose zero sub-level set is the RAS set. The
existing methods do not consider the problem when there are a series of targets
to reach and/or obstacles to avoid. We propose a method that uses the idea of
admissible control sets; we guarantee that the system will reach each target
while avoiding obstacles as prescribed by the given time series. Moreover, we
guarantee that the trajectory ultimately stabilizes to the POI. The proposed
method provides an under-approximation of the RAS set, guaranteeing safety.
Numerical examples are provided to validate the theory.

### 6. [Solving Reach- and Stabilize-Avoid Problems Using Discounted Reachability](http://arxiv.org/pdf/2505.09067v1)

Authors: Boyang Li, Zheng Gong, Sylvia Herbert

In this article, we consider the infinite-horizon reach-avoid (RA) and
stabilize-avoid (SA) zero-sum game problems for general nonlinear
continuous-time systems, where the goal is to find the set of states that can
be controlled to reach or stabilize to a target set, without violating
constraints even under the worst-case disturbance. Based on the Hamilton-Jacobi
reachability method, we address the RA problem by designing a new Lipschitz
continuous RA value function, whose zero sublevel set exactly characterizes the
RA set. We establish that the associated Bellman backup operator is contractive
and that the RA value function is the unique viscosity solution of a
Hamilton-Jacobi variational inequality. Finally, we develop a two-step
framework for the SA problem by integrating our RA strategies with a recently
proposed Robust Control Lyapunov-Value Function, thereby ensuring both target
reachability and long-term stability. We numerically verify our RA and SA
frameworks on a 3D Dubins car system to demonstrate the efficacy of the
proposed approach.

### 7. [Model Identification Adaptive Control with $ρ$-POMDP Planning](http://arxiv.org/pdf/2505.09119v1)

Authors: Michelle Ho, Arec Jamgochian, Mykel J. Kochenderfer

Accurate system modeling is crucial for safe, effective control, as
misidentification can lead to accumulated errors, especially under partial
observability. We address this problem by formulating informative input design
(IID) and model identification adaptive control (MIAC) as belief space planning
problems, modeled as partially observable Markov decision processes with
belief-dependent rewards ($\rho$-POMDPs). We treat system parameters as hidden
state variables that must be localized while simultaneously controlling the
system. We solve this problem with an adapted belief-space iterative Linear
Quadratic Regulator (BiLQR). We demonstrate it on fully and partially
observable tasks for cart-pole and steady aircraft flight domains. Our method
outperforms baselines such as regression, filtering, and local optimal control
methods, even under instantaneous disturbances to system parameters.

### 8. [Robot-Assisted Drone Recovery on a Wavy Surface Using Error-State Kalman Filter and Receding Horizon Model Predictive Control](http://arxiv.org/pdf/2505.09145v1)

Authors: Yimou Wu, Mingyang Liang, Ruoyu Xu

Recovering a drone on a disturbed water surface remains a significant
challenge in maritime robotics. In this paper, we propose a unified framework
for Robot-Assisted Drone Recovery on a Wavy Surface that addresses two major
tasks: Firstly, accurate prediction of a moving drone's position under
wave-induced disturbances using an Error-State Kalman Filter (ESKF), and
secondly, effective motion planning for a manipulator via Receding Horizon
Control (RHC). Specifically, the ESKF predicts the drone's future position 0.5s
ahead, while the manipulator plans a capture trajectory in real time, thus
overcoming not only wave-induced base motions but also limited torque
constraints. We provide a system design that comprises a manipulator subsystem
and a UAV subsystem. On the UAV side, we detail how position control and
suspended payload strategies are implemented. On the manipulator side, we show
how an RHC scheme outperforms traditional low-level control algorithms.
Simulation and real-world experiments - using wave-disturbed motion data -
demonstrate that our approach achieves a high success rate - above 95% and
outperforms conventional baseline methods by up to 10% in efficiency and 20% in
precision. The results underscore the feasibility and robustness of our system,
which achieves state-of-the-art (SOTA) performance and offers a practical
solution for maritime drone operations.

### 9. [Some Computational Tools for Solving a Selection of Problems in Control Theory](http://arxiv.org/pdf/2505.09191v1)

Authors: Alexander Demin, Christina Katsamaki, Fabrice Rouillier

This paper demonstrates how certified computational tools can be used to
address various problems in control theory. In particular, we introduce
PACE.jl, a Julia package that implements symbolic elimination techniques,
including (among others) discriminant varieties and Rational Univariate
Representation, while also supporting multi-precision interval computations. We
showcase its applications to key control theory problems, including
identification, stability analysis, and optimization, for both
parameter-dependent and parameter-free systems.

### 10. [Adaptive control for multi-scale stochastic dynamical systems with stochastic next generation reservoir computing](http://arxiv.org/pdf/2505.09327v1)

Authors: Jiani Cheng, Ting Gao, Jinqiao Duan

The rapid advancement of neuroscience and machine learning has established
data-driven stochastic dynamical system modeling as a powerful tool for
understanding and controlling high-dimensional, spatio-temporal processes. We
introduce the stochastic next-generation reservoir computing (NG-RC)
controller, a framework that integrates the computational efficiency of NG-RC
with stochastic analysis to enable robust event-triggered control in multiscale
stochastic systems. The asymptotic stability of the controller is rigorously
proven via an extended stochastic LaSalle theorem, providing theoretical
guarantees for amplitude regulation in nonlinear stochastic dynamics. Numerical
experiments on a stochastic Van-der-Pol system subject to both additive and
multiplicative noise validate the algorithm, demonstrating its convergence rate
across varying temporal scales and noise intensities. To bridge theoretical
insights with real-world applications, we deploy the controller to modulate
pathological dynamics reconstructed from epileptic EEG data. This work advances
a theoretically guaranteed scalable framework for adaptive control of
stochastic systems, with broad potential for data-driven decision making in
engineering, neuroscience, and beyond.

### Machine Learning (Statistics Category)

### 1. [Risk Bounds For Distributional Regression](http://arxiv.org/pdf/2505.09075v1)

Authors: Carlos Misael Madrid Padilla, Oscar Hernan Madrid Padilla, Sabyasachi Chatterjee

This work examines risk bounds for nonparametric distributional regression
estimators. For convex-constrained distributional regression, general upper
bounds are established for the continuous ranked probability score (CRPS) and
the worst-case mean squared error (MSE) across the domain. These theoretical
results are applied to isotonic and trend filtering distributional regression,
yielding convergence rates consistent with those for mean estimation.
Furthermore, a general upper bound is derived for distributional regression
under non-convex constraints, with a specific application to neural
network-based estimators. Comprehensive experiments on both simulated and real
data validate the theoretical contributions, demonstrating their practical
effectiveness.

### 2. [Scaling Gaussian Process Regression with Full Derivative Observations](http://arxiv.org/pdf/2505.09134v1)

Authors: Daniel Huang

We present a scalable Gaussian Process (GP) method that can fit and predict
full derivative observations called DSoftKI. It extends SoftKI, a method that
approximates a kernel via softmax interpolation from learned interpolation
point locations, to the setting with derivatives. DSoftKI enhances SoftKI's
interpolation scheme to incorporate the directional orientation of
interpolation points relative to the data. This enables the construction of a
scalable approximate kernel, including its first and second-order derivatives,
through interpolation. We evaluate DSoftKI on a synthetic function benchmark
and high-dimensional molecular force field prediction (100-1000 dimensions),
demonstrating that DSoftKI is accurate and can scale to larger datasets with
full derivative observations than previously possible.

### 3. [Online Learning of Neural Networks](http://arxiv.org/pdf/2505.09167v1)

Authors: Amit Daniely, Idan Mehalel, Elchanan Mossel

We study online learning of feedforward neural networks with the sign
activation function that implement functions from the unit ball in
$\mathbb{R}^d$ to a finite label set $\{1, \ldots, Y\}$.
  First, we characterize a margin condition that is sufficient and in some
cases necessary for online learnability of a neural network: Every neuron in
the first hidden layer classifies all instances with some margin $\gamma$
bounded away from zero. Quantitatively, we prove that for any net, the optimal
mistake bound is at most approximately $\mathtt{TS}(d,\gamma)$, which is the
$(d,\gamma)$-totally-separable-packing number, a more restricted variation of
the standard $(d,\gamma)$-packing number. We complement this result by
constructing a net on which any learner makes $\mathtt{TS}(d,\gamma)$ many
mistakes. We also give a quantitative lower bound of approximately
$\mathtt{TS}(d,\gamma) \geq \max\{1/(\gamma \sqrt{d})^d, d\}$ when $\gamma \geq
1/2$, implying that for some nets and input sequences every learner will err
for $\exp(d)$ many times, and that a dimension-free mistake bound is almost
always impossible.
  To remedy this inevitable dependence on $d$, it is natural to seek additional
natural restrictions to be placed on the network, so that the dependence on $d$
is removed. We study two such restrictions. The first is the multi-index model,
in which the function computed by the net depends only on $k \ll d$ orthonormal
directions. We prove a mistake bound of approximately $(1.5/\gamma)^{k + 2}$ in
this model. The second is the extended margin assumption. In this setting, we
assume that all neurons (in all layers) in the network classify every ingoing
input from previous layer with margin $\gamma$ bounded away from zero. In this
model, we prove a mistake bound of approximately $(\log Y)/ \gamma^{O(L)}$,
where L is the depth of the network.

### 4. [Generating Full-field Evolution of Physical Dynamics from Irregular Sparse Observations](http://arxiv.org/pdf/2505.09284v1)

Authors: Panqi Chen, Yifan Sun, Lei Cheng, Yang Yang, Weichang Li, Yang Liu, Weiqing Liu, Jiang Bian, Shikai Fang

Modeling and reconstructing multidimensional physical dynamics from sparse
and off-grid observations presents a fundamental challenge in scientific
research. Recently, diffusion-based generative modeling shows promising
potential for physical simulation. However, current approaches typically
operate on on-grid data with preset spatiotemporal resolution, but struggle
with the sparsely observed and continuous nature of real-world physical
dynamics. To fill the gaps, we present SDIFT, Sequential DIffusion in
Functional Tucker space, a novel framework that generates full-field evolution
of physical dynamics from irregular sparse observations. SDIFT leverages the
functional Tucker model as the latent space representer with proven universal
approximation property, and represents observations as latent functions and
Tucker core sequences. We then construct a sequential diffusion model with
temporally augmented UNet in the functional Tucker space, denoising noise drawn
from a Gaussian process to generate the sequence of core tensors.
  At the posterior sampling stage, we propose a Message-Passing Posterior
Sampling mechanism, enabling conditional generation of the entire sequence
guided by observations at limited time steps. We validate SDIFT on three
physical systems spanning astronomical (supernova explosions, light-year
scale), environmental (ocean sound speed fields, kilometer scale), and
molecular (organic liquid, millimeter scale) domains, demonstrating significant
improvements in both reconstruction accuracy and computational efficiency
compared to state-of-the-art approaches.

### 5. [Establishing Linear Surrogate Regret Bounds for Convex Smooth Losses via Convolutional Fenchel-Young Losses](http://arxiv.org/pdf/2505.09432v2)

Authors: Yuzhou Cao, Han Bao, Lei Feng, Bo An

Surrogate regret bounds, also known as excess risk bounds, bridge the gap
between the convergence rates of surrogate and target losses, with linear
bounds favorable for their lossless regret transfer. While convex smooth
surrogate losses are appealing in particular due to the efficient estimation
and optimization, the existence of a trade-off between the smoothness and
linear regret bound has been believed in the community. That being said, the
better optimization and estimation properties of convex smooth surrogate losses
may inevitably deteriorate after undergoing the regret transfer onto a target
loss. We overcome this dilemma for arbitrary discrete target losses by
constructing a convex smooth surrogate loss, which entails a linear surrogate
regret bound composed with a tailored prediction link. The construction is
based on Fenchel-Young losses generated by the convolutional negentropy, which
are equivalent to the infimal convolution of a generalized negentropy and the
target Bayes risk. Consequently, the infimal convolution enables us to derive a
smooth loss while maintaining the surrogate regret bound linear. We
additionally benefit from the infimal convolution to have a consistent
estimator of the underlying class probability. Our results are overall a novel
demonstration of how convex analysis penetrates into optimization and
statistical efficiency in risk minimization.

### 6. [Reinforcement Learning for Individual Optimal Policy from Heterogeneous Data](http://arxiv.org/pdf/2505.09496v1)

Authors: Rui Miao, Babak Shahbaba, Annie Qu

Offline reinforcement learning (RL) aims to find optimal policies in dynamic
environments in order to maximize the expected total rewards by leveraging
pre-collected data. Learning from heterogeneous data is one of the fundamental
challenges in offline RL. Traditional methods focus on learning an optimal
policy for all individuals with pre-collected data from a single episode or
homogeneous batch episodes, and thus, may result in a suboptimal policy for a
heterogeneous population. In this paper, we propose an individualized offline
policy optimization framework for heterogeneous time-stationary Markov decision
processes (MDPs). The proposed heterogeneous model with individual latent
variables enables us to efficiently estimate the individual Q-functions, and
our Penalized Pessimistic Personalized Policy Learning (P4L) algorithm
guarantees a fast rate on the average regret under a weak partial coverage
assumption on behavior policies. In addition, our simulation studies and a real
data application demonstrate the superior numerical performance of the proposed
method compared with existing methods.

### 7. [Deep-SITAR: A SITAR-Based Deep Learning Framework for Growth Curve Modeling via Autoencoders](http://arxiv.org/pdf/2505.09506v1)

Authors: María Alejandra Hernández, Oscar Rodriguez, Dae-Jin Lee

Several approaches have been developed to capture the complexity and
nonlinearity of human growth. One widely used is the Super Imposition by
Translation and Rotation (SITAR) model, which has become popular in studies of
adolescent growth. SITAR is a shape-invariant mixed-effects model that
represents the shared growth pattern of a population using a natural cubic
spline mean curve while incorporating three subject-specific random effects --
timing, size, and growth intensity -- to account for variations among
individuals. In this work, we introduce a supervised deep learning framework
based on an autoencoder architecture that integrates a deep neural network
(neural network) with a B-spline model to estimate the SITAR model. In this
approach, the encoder estimates the random effects for each individual, while
the decoder performs a fitting based on B-splines similar to the classic SITAR
model. We refer to this method as the Deep-SITAR model. This innovative
approach enables the prediction of the random effects of new individuals
entering a population without requiring a full model re-estimation. As a
result, Deep-SITAR offers a powerful approach to predicting growth
trajectories, combining the flexibility and efficiency of deep learning with
the interpretability of traditional mixed-effects models.

### 8. [Fair Clustering via Alignment](http://arxiv.org/pdf/2505.09131v1)

Authors: Kunwoong Kim, Jihu Lee, Sangchul Park, Yongdai Kim

Algorithmic fairness in clustering aims to balance the proportions of
instances assigned to each cluster with respect to a given sensitive attribute.
While recently developed fair clustering algorithms optimize clustering
objectives under specific fairness constraints, their inherent complexity or
approximation often results in suboptimal clustering utility or numerical
instability in practice. To resolve these limitations, we propose a new fair
clustering algorithm based on a novel decomposition of the fair K-means
clustering objective function. The proposed algorithm, called Fair Clustering
via Alignment (FCA), operates by alternately (i) finding a joint probability
distribution to align the data from different protected groups, and (ii)
optimizing cluster centers in the aligned space. A key advantage of FCA is that
it theoretically guarantees approximately optimal clustering utility for any
given fairness level without complex constraints, thereby enabling high-utility
fair clustering in practice. Experiments show that FCA outperforms existing
methods by (i) attaining a superior trade-off between fairness level and
clustering utility, and (ii) achieving near-perfect fairness without numerical
instability.

### 9. [Optimal Transport-Based Domain Adaptation for Rotated Linear Regression](http://arxiv.org/pdf/2505.09229v1)

Authors: Brian Britos, Mathias Bourel

Optimal Transport (OT) has proven effective for domain adaptation (DA) by
aligning distributions across domains with differing statistical properties.
Building on the approach of Courty et al. (2016), who mapped source data to the
target domain for improved model transfer, we focus on a supervised DA problem
involving linear regression models under rotational shifts. This ongoing work
considers cases where source and target domains are related by a
rotation-common in applications like sensor calibration or image orientation.
We show that in $\mathbb{R}^2$ , when using a p-norm cost with $p $\ge$ 2$, the
optimal transport map recovers the underlying rotation. Based on this, we
propose an algorithm that combines K-means clustering, OT, and singular value
decomposition (SVD) to estimate the rotation angle and adapt the regression
model. This method is particularly effective when the target domain is sparsely
sampled, leveraging abundant source data for improved generalization. Our
contributions offer both theoretical and practical insights into OT-based model
adaptation under geometric transformations.

### 10. [Scalable Computations for Generalized Mixed Effects Models with Crossed Random Effects Using Krylov Subspace Methods](http://arxiv.org/pdf/2505.09552v1)

Authors: Pascal Kündig, Fabio Sigrist

Mixed effects models are widely used for modeling data with hierarchically
grouped structures and high-cardinality categorical predictor variables.
However, for high-dimensional crossed random effects, current standard
computations relying on Cholesky decompositions can become prohibitively slow.
In this work, we present novel Krylov subspace-based methods that address
several existing computational bottlenecks. Among other things, we
theoretically analyze and empirically evaluate various preconditioners for the
conjugate gradient and stochastic Lanczos quadrature methods, derive new
convergence results, and develop computationally efficient methods for
calculating predictive variances. Extensive experiments using simulated and
real-world data sets show that our proposed methods scale much better than
Cholesky-based computations, for instance, achieving a runtime reduction of
approximately two orders of magnitudes for both estimation and prediction.
Moreover, our software implementation is up to 10'000 times faster and more
stable than state-of-the-art implementations such as lme4 and glmmTMB when
using default settings. Our methods are implemented in the free C++ software
library GPBoost with high-level Python and R packages.

