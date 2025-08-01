# Computer Science arXiv Papers

Collection of top 10 Computer Science research papers pulled daily from arXiv.

---

Pulled on 2025-07-31 17:00:25.738526 PST.

### Artificial Intelligence

### 1. [An Explainable Emotion Alignment Framework for LLM-Empowered Agent in Metaverse Service Ecosystem](http://arxiv.org/pdf/2507.22326v1)

Authors: Qun Ma, Xiao Xue, Ming Zhang, Yifan Shen, Zihan Zhao

Metaverse service is a product of the convergence between Metaverse and
service systems, designed to address service-related challenges concerning
digital avatars, digital twins, and digital natives within Metaverse. With the
rise of large language models (LLMs), agents now play a pivotal role in
Metaverse service ecosystem, serving dual functions: as digital avatars
representing users in the virtual realm and as service assistants (or NPCs)
providing personalized support. However, during the modeling of Metaverse
service ecosystems, existing LLM-based agents face significant challenges in
bridging virtual-world services with real-world services, particularly
regarding issues such as character data fusion, character knowledge
association, and ethical safety concerns. This paper proposes an explainable
emotion alignment framework for LLM-based agents in Metaverse Service
Ecosystem. It aims to integrate factual factors into the decision-making loop
of LLM-based agents, systematically demonstrating how to achieve more
relational fact alignment for these agents. Finally, a simulation experiment in
the Offline-to-Offline food delivery scenario is conducted to evaluate the
effectiveness of this framework, obtaining more realistic social emergence.

### 2. [On the Definition of Intelligence](http://arxiv.org/pdf/2507.22423v1)

Authors: Kei-Sing Ng

To engineer AGI, we should first capture the essence of intelligence in a
species-agnostic form that can be evaluated, while being sufficiently general
to encompass diverse paradigms of intelligent behavior, including reinforcement
learning, generative models, classification, analogical reasoning, and
goal-directed decision-making. We propose a general criterion based on sample
fidelity: intelligence is the ability, given sample(s) from a category, to
generate sample(s) from the same category. We formalise this intuition as
{\epsilon}-category intelligence: it is {\epsilon}-intelligent with respect to
a category if no chosen admissible distinguisher can separate generated from
original samples beyond tolerance {\epsilon}. We present the formal framework,
outline empirical protocols, and discuss implications for evaluation, safety,
and generalization.

### 3. [Cross-Border Legal Adaptation of Autonomous Vehicle Design based on Logic and Non-monotonic Reasoning](http://arxiv.org/pdf/2507.22432v1)

Authors: Zhe Yu, Yiwei Lu, Burkhard Schafer, Zhe Lin

This paper focuses on the legal compliance challenges of autonomous vehicles
in a transnational context. We choose the perspective of designers and try to
provide supporting legal reasoning in the design process. Based on
argumentation theory, we introduce a logic to represent the basic properties of
argument-based practical (normative) reasoning, combined with partial order
sets of natural numbers to express priority. Finally, through case analysis of
legal texts, we show how the reasoning system we provide can help designers to
adapt their design solutions more flexibly in the cross-border application of
autonomous vehicles and to more easily understand the legal implications of
their decisions.

### 4. [Collaborative Medical Triage under Uncertainty: A Multi-Agent Dynamic Matching Approach](http://arxiv.org/pdf/2507.22504v1)

Authors: Hongyan Cheng, Chengzhang Yu, Yanshu Shi, Chiyue Wang, Cong Liu, Zhanpeng Jin

The post-pandemic surge in healthcare demand, coupled with critical nursing
shortages, has placed unprecedented pressure on emergency department triage
systems, necessitating innovative AI-driven solutions. We present a multi-agent
interactive intelligent system for medical triage that addresses three
fundamental challenges in current AI-based triage systems: insufficient medical
specialization leading to hallucination-induced misclassifications,
heterogeneous department structures across healthcare institutions, and
inefficient detail-oriented questioning that impedes rapid triage decisions.
Our system employs three specialized agents - RecipientAgent, InquirerAgent,
and DepartmentAgent - that collaborate through structured inquiry mechanisms
and department-specific guidance rules to transform unstructured patient
symptoms into accurate department recommendations. To ensure robust evaluation,
we constructed a comprehensive Chinese medical triage dataset from a medical
website, comprising 3,360 real-world cases spanning 9 primary departments and
62 secondary departments. Through systematic data imputation using large
language models, we address the prevalent issue of incomplete medical records
in real-world data. Experimental results demonstrate that our multi-agent
system achieves 89.2% accuracy in primary department classification and 73.9%
accuracy in secondary department classification after four rounds of patient
interaction. The system's pattern-matching-based guidance mechanisms enable
efficient adaptation to diverse hospital configurations while maintaining high
triage accuracy. Our work provides a scalable framework for deploying
AI-assisted triage systems that can accommodate the organizational
heterogeneity of healthcare institutions while ensuring clinically sound
decision-making.

### 5. [MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines](http://arxiv.org/pdf/2507.22606v1)

Authors: Yaolun Zhang, Xiaogeng Liu, Chaowei Xiao

Large Language Models (LLMs) have demonstrated the ability to solve a wide
range of practical tasks within multi-agent systems. However, existing
human-designed multi-agent frameworks are typically limited to a small set of
pre-defined scenarios, while current automated design methods suffer from
several limitations, such as the lack of tool integration, dependence on
external training data, and rigid communication structures. In this paper, we
propose MetaAgent, a finite state machine based framework that can
automatically generate a multi-agent system. Given a task description,
MetaAgent will design a multi-agent system and polish it through an
optimization algorithm. When the multi-agent system is deployed, the finite
state machine will control the agent's actions and the state transitions. To
evaluate our framework, we conduct experiments on both text-based tasks and
practical tasks. The results indicate that the generated multi-agent system
surpasses other auto-designed methods and can achieve a comparable performance
with the human-designed multi-agent system, which is optimized for those
specific tasks.

### 6. [Enhancing Manufacturing Knowledge Access with LLMs and Context-aware Prompting](http://arxiv.org/pdf/2507.22619v1)

Authors: Sebastian Monka, Irlan Grangel-González, Stefan Schmid, Lavdim Halilaj, Marc Rickart, Oliver Rudolph, Rui Dias

Knowledge graphs (KGs) have transformed data management within the
manufacturing industry, offering effective means for integrating disparate data
sources through shared and structured conceptual schemas. However, harnessing
the power of KGs can be daunting for non-experts, as it often requires
formulating complex SPARQL queries to retrieve specific information. With the
advent of Large Language Models (LLMs), there is a growing potential to
automatically translate natural language queries into the SPARQL format, thus
bridging the gap between user-friendly interfaces and the sophisticated
architecture of KGs. The challenge remains in adequately informing LLMs about
the relevant context and structure of domain-specific KGs, e.g., in
manufacturing, to improve the accuracy of generated queries. In this paper, we
evaluate multiple strategies that use LLMs as mediators to facilitate
information retrieval from KGs. We focus on the manufacturing domain,
particularly on the Bosch Line Information System KG and the I40 Core
Information Model. In our evaluation, we compare various approaches for feeding
relevant context from the KG to the LLM and analyze their proficiency in
transforming real-world questions into SPARQL queries. Our findings show that
LLMs can significantly improve their performance on generating correct and
complete queries when provided only the adequate context of the KG schema. Such
context-aware prompting techniques help LLMs to focus on the relevant parts of
the ontology and reduce the risk of hallucination. We anticipate that the
proposed techniques help LLMs to democratize access to complex data
repositories and empower informed decision-making in manufacturing settings.

### 7. [ASP-FZN: A Translation-based Constraint Answer Set Solver](http://arxiv.org/pdf/2507.22774v1)

Authors: Thomas Eiter, Tobias Geibinger, Tobias Kaminski, Nysret Musliu, Johannes Oetsch

We present the solver asp-fzn for Constraint Answer Set Programming (CASP),
which extends ASP with linear constraints. Our approach is based on translating
CASP programs into the solver-independent FlatZinc language that supports
several Constraint Programming and Integer Programming backend solvers. Our
solver supports a rich language of linear constraints, including some common
global constraints. As for evaluation, we show that asp-fzn is competitive with
state-of-the-art ASP solvers on benchmarks taken from past ASP competitions.
Furthermore, we evaluate it on several CASP problems from the literature and
compare its performance with clingcon, which is a prominent CASP solver that
supports most of the asp-fzn language. The performance of asp-fzn is very
promising as it is already competitive on plain ASP and even outperforms
clingcon on some CASP benchmarks.

### 8. [AdapSCA-PSO: An Adaptive Localization Algorithm with AI-Based Hybrid SCA-PSO for IoT WSNs](http://arxiv.org/pdf/2507.22317v1)

Authors: Ze Zhang, Qian Dong, Wenhan Wang

The accurate localization of sensor nodes is a fundamental requirement for
the practical application of the Internet of Things (IoT). To enable robust
localization across diverse environments, this paper proposes a hybrid
meta-heuristic localization algorithm. Specifically, the algorithm integrates
the Sine Cosine Algorithm (SCA), which is effective in global search, with
Particle Swarm Optimization (PSO), which excels at local search. An adaptive
switching module is introduced to dynamically select between the two
algorithms. Furthermore, the initialization, fitness evaluation, and parameter
settings of the algorithm have been specifically redesigned and optimized to
address the characteristics of the node localization problem. Simulation
results across varying numbers of sensor nodes demonstrate that, compared to
standalone PSO and the unoptimized SCAPSO algorithm, the proposed method
significantly reduces the number of required iterations and achieves an average
localization error reduction of 84.97%.

### 9. [Learning from Heterogeneous Structural MRI via Collaborative Domain Adaptation for Late-Life Depression Assessment](http://arxiv.org/pdf/2507.22321v1)

Authors: Yuzhen Gao, Qianqian Wang, Yongheng Sun, Cui Wang, Yongquan Liang, Mingxia Liu

Accurate identification of late-life depression (LLD) using structural brain
MRI is essential for monitoring disease progression and facilitating timely
intervention. However, existing learning-based approaches for LLD detection are
often constrained by limited sample sizes (e.g., tens), which poses significant
challenges for reliable model training and generalization. Although
incorporating auxiliary datasets can expand the training set, substantial
domain heterogeneity, such as differences in imaging protocols, scanner
hardware, and population demographics, often undermines cross-domain
transferability. To address this issue, we propose a Collaborative Domain
Adaptation (CDA) framework for LLD detection using T1-weighted MRIs. The CDA
leverages a Vision Transformer (ViT) to capture global anatomical context and a
Convolutional Neural Network (CNN) to extract local structural features, with
each branch comprising an encoder and a classifier. The CDA framework consists
of three stages: (a) supervised training on labeled source data, (b)
self-supervised target feature adaptation and (c) collaborative training on
unlabeled target data. We first train ViT and CNN on source data, followed by
self-supervised target feature adaptation by minimizing the discrepancy between
classifier outputs from two branches to make the categorical boundary clearer.
The collaborative training stage employs pseudo-labeled and augmented
target-domain MRIs, enforcing prediction consistency under strong and weak
augmentation to enhance domain robustness and generalization. Extensive
experiments conducted on multi-site T1-weighted MRI data demonstrate that the
CDA consistently outperforms state-of-the-art unsupervised domain adaptation
methods.

### 10. [From Articles to Code: On-Demand Generation of Core Algorithms from Scientific Publications](http://arxiv.org/pdf/2507.22324v1)

Authors: Cameron S. Movassaghi, Amanda Momenzadeh, Jesse G. Meyer

Maintaining software packages imposes significant costs due to dependency
management, bug fixes, and versioning. We show that rich method descriptions in
scientific publications can serve as standalone specifications for modern large
language models (LLMs), enabling on-demand code generation that could supplant
human-maintained libraries. We benchmark state-of-the-art models
(GPT-o4-mini-high, Gemini Pro 2.5, Claude Sonnet 4) by tasking them with
implementing a diverse set of core algorithms drawn from original publications.
Our results demonstrate that current LLMs can reliably reproduce package
functionality with performance indistinguishable from conventional libraries.
These findings foreshadow a paradigm shift toward flexible, on-demand code
generation and away from static, human-maintained packages, which will result
in reduced maintenance overhead by leveraging published articles as sufficient
context for the automated implementation of analytical workflows.

### Computational Complexity

### 1. [Reducing the complexity of computing the values of a Nash equilibrium](http://arxiv.org/pdf/2507.22819v1)

Authors: Debtoru Chatterjee, Girish Tiwari, Niladri Chatterjee

The Colonel Blotto game, formulated by Emile Borel, involves players
allocating limited resources to multiple battlefields simultaneously, with the
winner being the one who allocates more resources to each battlefield.
Computation of the Nash equilibrium, including of two person, zero sum, mixed
strategy Colonel Blotto games have encountered issues of scalability and
complexity owing to their PPAD completeness. This paper proposes an algorithm
that computes the same value as the Nash equilibrium but cannot be
characterized by the Fixed point Theorems of Tarski, Kakutani and Brouwer. The
reduced complexity of the proposed algorithm is based on dispensing with the
need for computing both players Nash strategies in Colonel Blotto games. The
same algorithm can, therefore, be extended to all two person, zero sum games to
compute the value of the Nash equilibrium. The theoretical superiority of the
proposed algorithm over both LP solvers and another method that computes the
same value of the game as its Nash equilibrium by a random assignment of
probabilities to the active strategy set of the defending player, is also
proposed.

### 2. [Approximating the quantum value of an LCS game is RE-hard](http://arxiv.org/pdf/2507.22444v1)

Authors: Aviv Taller, Thomas Vidick

We generalize H\r{a}stad's long-code test for projection games and show that
it remains complete and sound against entangled provers.
  Combined with a result of Dong et al. \cite{Dong25}, which establishes that
$\MIP^*=\RE$ with constant-length answers, we derive that
$\LIN^*_{1-\epsilon,s}=\RE$, for some $1/2< s<1$ and for every sufficiently
small $\epsilon>0$, where LIN refers to linearity (over $\mathbb{F}_2$) of the
verifier predicate. Achieving the same result with $\epsilon=0$ would imply the
existence of a non-hyperlinear group.

### Computational Engineering

### 1. [A holomorphic Kolmogorov-Arnold network framework for solving elliptic problems on arbitrary 2D domains](http://arxiv.org/pdf/2507.22678v1)

Authors: Matteo Calafà, Tito Andriollo, Allan P. Engsig-Karup, Cheol-Ho Jeong

Physics-informed holomorphic neural networks (PIHNNs) have recently emerged
as efficient surrogate models for solving differential problems. By embedding
the underlying problem structure into the network, PIHNNs require training only
to satisfy boundary conditions, often resulting in significantly improved
accuracy and computational efficiency compared to traditional physics-informed
neural networks (PINNs). In this work, we improve and extend the application of
PIHNNs to two-dimensional problems. First, we introduce a novel holomorphic
network architecture based on the Kolmogorov-Arnold representation (PIHKAN),
which achieves higher accuracy with reduced model complexity. Second, we
develop mathematical extensions that broaden the applicability of PIHNNs to a
wider class of elliptic partial differential equations, including the Helmholtz
equation. Finally, we propose a new method based on Laurent series theory that
enables the application of holomorphic networks to multiply-connected plane
domains, thereby removing the previous limitation to simply-connected
geometries.

### 2. [Deep reinforcement learning for efficient exploration of combinatorial structural design spaces](http://arxiv.org/pdf/2507.22804v1)

Authors: Chloe S. H. Hong, Keith J. Lee, Caitlin T. Mueller

This paper proposes a reinforcement learning framework for performance-driven
structural design that combines bottom-up design generation with learned
strategies to efficiently search large combinatorial design spaces. Motivated
by the limitations of conventional top-down approaches such as optimization,
the framework instead models structures as compositions of predefined elements,
aligning form finding with practical constraints like constructability and
component reuse. With the formulation of the design task as a sequential
decision-making problem and a human learning inspired training algorithm, the
method adapts reinforcement learning for structural design. The framework is
demonstrated by designing steel braced truss frame cantilever structures, where
trained policies consistently generate distinct, high-performing designs that
display structural performance and material efficiency with the use of
structural strategies that align with known engineering principles. Further
analysis shows that the agent efficiently narrows its search to promising
regions of the design space, revealing transferable structural knowledge.

### 3. [Modelling and simulation of electro-mechanically coupled dielectric elastomers and myocardial tissue using smoothed finite element methods](http://arxiv.org/pdf/2507.22838v1)

Authors: Tan Tran, Denisa Martonova, Sigrid Leyendecker

Computational modelling offers a cost-effective and time-efficient
alternative to experimental studies in biomedical engineering. In cardiac
electro-mechanics, finite element method (FEM)-based simulations provide
valuable insights into diseased tissue behaviour and the development of
assistive systems such as di-electric elastomer actuators. However, the use of
automatically generated tetrahedral meshes, commonly applied due to geometric
complexity, often leads to numerical issues including overly stiff responses
and volume locking, particularly in incompressible materials. Smoothed finite
element methods (S-FEMs) offer a promising alternative by softening the
stiffness matrix through gradient smoothing over defined smoothing domains.
This work extends S-FEM formulations to electro-mechanically coupled problems
and compares their performance against standard linear FEM. We implement and
evaluate four approaches in the Abaqus environment via custom user elements:
standard linear FEM, face-based S-FEM (FS-FEM), node-based S-FEM (NS-FEM), and
the hybrid face/node-based S-FEM (FSNS-FEM). Two benchmark problems are
studied: the electrically induced contraction of a compressible dielectric
elastomer and an incompressible, orthotropic myocardial tissue sample.
Reference solutions are obtained using a mesh consisting of higher-order
elements. Our results demonstrate that FSNS-FEM provides the best balance
between accuracy and computational efficiency, closely matching reference data.
NS-FEM produces softer results, which leads to an overestimation of the true
deformation. FS-FEM and standard FEM consistently exhibit overly stiff
behaviour, with pronounced volume locking in the myocardial case. These
findings support the potential of S-FEMs, in particular FSNS-FEM, for accurate
simulation of coupled electro-mechanical behaviour in complex biomedical
applications.

### 4. [Mean-Variance Optimization and Algorithm for Finite-Horizon Markov Decision Processes](http://arxiv.org/pdf/2507.22327v1)

Authors: Li Xia, Zhihui Yu

Multi-period mean-variance optimization is a long-standing problem, caused by
the failure of dynamic programming principle. This paper studies the
mean-variance optimization in a setting of finite-horizon discrete-time Markov
decision processes (MDPs), where the objective is to maximize the combined
metrics of mean and variance of the accumulated rewards at terminal stage. By
introducing the concepts of pseudo mean and pseudo variance, we convert the
original mean-variance MDP to a bilevel MDP, where the outer is a single
parameter optimization of the pseudo mean and the inner is a standard
finite-horizon MDP with an augmented state space by adding an auxiliary state
of accumulated rewards. We further study the properties of this bilevel MDP,
including the optimality of history-dependent deterministic policies and the
piecewise quadratic concavity of the inner MDPs' optimal values with respect to
the pseudo mean. To efficiently solve this bilevel MDP, we propose an iterative
algorithm that alternatingly updates the inner optimal policy and the outer
pseudo mean. We prove that this algorithm converges to a local optimum. We also
derive a sufficient condition under which our algorithm converges to the global
optimum. Furthermore, we apply this approach to study the mean-variance
optimization of multi-period portfolio selection problem, which shows that our
approach exactly coincides with the classical result by Li and Ng (2000) in
financial engineering. Our approach builds a new avenue to solve mean-variance
optimization problems and has wide applicability to any problem modeled by
MDPs, which is further demonstrated by examples of mean-variance optimization
for queueing control and inventory management.

### 5. [Cycles Protocol: A Peer-to-Peer Electronic Clearing System](http://arxiv.org/pdf/2507.22309v1)

Authors: Ethan Buchman, Paolo Dini, Shoaib Ahmed, Andrew Miller, Tomaž Fleischman

For centuries, financial institutions have responded to liquidity challenges
by forming closed, centralized clearing clubs with strict rules and membership
that allow them to collaborate on using the least money to discharge the most
debt. As closed clubs, much of the general public has been excluded from
participation. But the vast majority of private sector actors consists of micro
or small firms that are vulnerable to late payments and generally ineligible
for bank loans. This low liquidity environment often results in gridlock and
leads to insolvency, and it disproportionately impacts small enterprises and
communities.
  On the other hand, blockchain communities have developed open, decentralized
settlement systems, along with a proliferation of store of value assets and new
lending protocols, allowing anyone to permissionlessly transact and access
credit. However, these protocols remain used primarily for speculative
purposes, and so far have fallen short of the large-scale positive impact on
the real economy prophesied by their promoters.
  We address these challenges by introducing Cycles, an open, decentralized
clearing, settlement, and issuance protocol. Cycles is designed to enable firms
to overcome payment inefficiencies, to reduce their working capital costs, and
to leverage diverse assets and liquidity sources, including cryptocurrencies,
stablecoins, and lending protocols, in service of clearing more debt with less
money. Cycles solves real world liquidity challenges through a
privacy-preserving multilateral settlement platform based on a graph
optimization algorithm. The design is based on a core insight: liquidity
resides within cycles in the payment network's structure and can be accessed
via settlement flows optimized to reduce debt.

### 6. [MASCA: LLM based-Multi Agents System for Credit Assessment](http://arxiv.org/pdf/2507.22758v1)

Authors: Gautam Jajoo, Pranjal A Chitale, Saksham Agarwal

Recent advancements in financial problem-solving have leveraged LLMs and
agent-based systems, with a primary focus on trading and financial modeling.
However, credit assessment remains an underexplored challenge, traditionally
dependent on rule-based methods and statistical models. In this paper, we
introduce MASCA, an LLM-driven multi-agent system designed to enhance credit
evaluation by mirroring real-world decision-making processes. The framework
employs a layered architecture where specialized LLM-based agents
collaboratively tackle sub-tasks. Additionally, we integrate contrastive
learning for risk and reward assessment to optimize decision-making. We further
present a signaling game theory perspective on hierarchical multi-agent
systems, offering theoretical insights into their structure and interactions.
Our paper also includes a detailed bias analysis in credit assessment,
addressing fairness concerns. Experimental results demonstrate that MASCA
outperforms baseline approaches, highlighting the effectiveness of hierarchical
LLM-based multi-agent systems in financial applications, particularly in credit
scoring.

### 7. [Mesh based segmentation for automated margin line generation on incisors receiving crown treatment](http://arxiv.org/pdf/2507.22859v1)

Authors: Ammar Alsheghri, Ying Zhang, Farnoosh Ghadiri, Julia Keren, Farida Cheriet, Francois Guibault

Dental crowns are essential dental treatments for restoring damaged or
missing teeth of patients. Recent design approaches of dental crowns are
carried out using commercial dental design software. Once a scan of a
preparation is uploaded to the software, a dental technician needs to manually
define a precise margin line on the preparation surface, which constitutes a
non-repeatable and inconsistent procedure. This work proposes a new framework
to determine margin lines automatically and accurately using deep learning. A
dataset of incisor teeth was provided by a collaborating dental laboratory to
train a deep learning segmentation model. A mesh-based neural network was
modified by changing its input channels and used to segment the prepared tooth
into two regions such that the margin line is contained within the boundary
faces separating the two regions. Next, k-fold cross-validation was used to
train 5 models, and a voting classifier technique was used to combine their
results to enhance the segmentation. After that, boundary smoothing and
optimization using the graph cut method were applied to refine the segmentation
results. Then, boundary faces separating the two regions were selected to
represent the margin line faces. A spline was approximated to best fit the
centers of the boundary faces to predict the margin line. Our results show that
an ensemble model combined with maximum probability predicted the highest
number of successful test cases (7 out of 13) based on a maximum distance
threshold of 200 m (representing human error) between the predicted and ground
truth point clouds. It was also demonstrated that the better the quality of the
preparation, the smaller the divergence between the predicted and ground truth
margin lines (Spearman's rank correlation coefficient of -0.683). We provide
the train and test datasets for the community.

### 8. [A surrogate model for topology optimisation of elastic structures via parametric autoencoders](http://arxiv.org/pdf/2507.22539v1)

Authors: Matteo Giacomini, Antonio Huerta

A surrogate-based topology optimisation algorithm for linear elastic
structures under parametric loads and boundary conditions is proposed. Instead
of learning the parametric solution of the state (and adjoint) problems or the
optimisation trajectory as a function of the iterations, the proposed approach
devises a surrogate version of the entire optimisation pipeline. First, the
method predicts a quasi-optimal topology for a given problem configuration as a
surrogate model of high-fidelity topologies optimised with the homogenisation
method. This is achieved by means of a feed-forward net learning the mapping
between the input parameters characterising the system setup and a latent space
determined by encoder/decoder blocks reducing the dimensionality of the
parametric topology optimisation problem and reconstructing a high-dimensional
representation of the topology. Then, the predicted topology is used as an
educated initial guess for a computationally efficient algorithm penalising the
intermediate values of the design variable, while enforcing the governing
equations of the system. This step allows the method to correct potential
errors introduced by the surrogate model, eliminate artifacts, and refine the
design in order to produce topologies consistent with the underlying physics.
Different architectures are proposed and the approximation and generalisation
capabilities of the resulting models are numerically evaluated. The
quasi-optimal topologies allow to outperform the high-fidelity optimiser by
reducing the average number of optimisation iterations by $53\%$ while
achieving discrepancies below $4\%$ in the optimal value of the objective
functional, even in the challenging scenario of testing the model to
extrapolate beyond the training and validation domain.

### Computational Geometry

### 1. [Geometry of nonlinear forecast reconciliation](http://arxiv.org/pdf/2507.22500v1)

Authors: Lorenzo Nespoli, Anubhab Biswas, Vasco Medici

Forecast reconciliation, an ex-post technique applied to forecasts that must
satisfy constraints, has been a prominent topic in the forecasting literature
over the past two decades. Recently, several efforts have sought to extend
reconciliation methods to the probabilistic settings. Nevertheless, formal
theorems demonstrating error reduction in nonlinear contexts, analogous to
those presented in Panagiotelis et al.(2021), are still lacking. This paper
addresses that gap by establishing such theorems for various classes of
nonlinear hypersurfaces and vector-valued functions. Specifically, we derive an
exact analog of Theorem 3.1 from Panagiotelis et al.(2021) for hypersurfaces
with constant-sign curvature. Additionally, we provide probabilistic guarantees
for the broader case of hypersurfaces with non-constant-sign curvature and for
general vector-valued functions. To support reproducibility and practical
adoption, we release a JAX-based Python package, \emph{to be released upon
publication}, implementing the presented theorems and reconciliation
procedures.

### 2. [Kan Approximations of the Persistent Homology Transform](http://arxiv.org/pdf/2507.22816v1)

Authors: Shreya Arya, Justin Curry

The persistent homology transform (PHT) of a subset $M \subset \mathbb{R}^d$
is a map $\text{PHT}(M):\mathbb{S}^{d-1} \to \mathbf{Dgm}$ from the unit sphere
to the space of persistence diagrams. This map assigns to each direction $v\in
\mathbb{S}^{d-1}$ the persistent homology of the filtration of $M$ in direction
$v$. In practice, one can only sample the map $\text{PHT}(M)$ at a finite set
of directions $A \subset \mathbb{S}^{d-1}$. This suggests two natural
questions: (1) Can we interpolate the PHT from this finite sample of directions
to the entire sphere? If so, (2) can we prove that the resulting interpolation
is close to the true PHT? In this paper we show that if we can sample the PHT
at the module level, where we have information about how homology from each
direction interacts, a ready-made interpolation theory due to Bubenik, de
Silva, and Nanda using Kan extensions can answer both of these questions in the
affirmative. A close inspection of those techniques shows that we can infer the
PHT from a finite sample of heights from each direction as well. Our paper
presents the first known results for approximating the PHT from finite
directional and scalar data.

### Computation and Language

### 1. [Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance](http://arxiv.org/pdf/2507.22448v1)

Authors: Jingwei Zuo, Maksim Velikanov, Ilyas Chahed, Younes Belkada, Dhia Eddine Rhayem, Guillaume Kunsch, Hakim Hacid, Hamza Yous, Brahim Farhat, Ibrahim Khadraoui, Mugariya Farooq, Giulia Campesan, Ruxandra Cojocaru, Yasser Djilali, Shi Hu, Iheb Chaabane, Puneesh Khanna, Mohamed El Amine Seddik, Ngoc Dung Huynh, Phuc Le Khac, Leen AlQadi, Billel Mokeddem, Mohamed Chami, Abdalgader Abubaker, Mikhail Lubinets, Kacper Piskorski, Slim Frikha

In this report, we introduce Falcon-H1, a new series of large language models
(LLMs) featuring hybrid architecture designs optimized for both high
performance and efficiency across diverse use cases. Unlike earlier Falcon
models built solely on Transformer or Mamba architectures, Falcon-H1 adopts a
parallel hybrid approach that combines Transformer-based attention with State
Space Models (SSMs), known for superior long-context memory and computational
efficiency. We systematically revisited model design, data strategy, and
training dynamics, challenging conventional practices in the field. Falcon-H1
is released in multiple configurations, including base and instruction-tuned
variants at 0.5B, 1.5B, 1.5B-deep, 3B, 7B, and 34B parameters. Quantized
instruction-tuned models are also available, totaling over 30 checkpoints on
Hugging Face Hub. Falcon-H1 models demonstrate state-of-the-art performance and
exceptional parameter and training efficiency. The flagship Falcon-H1-34B
matches or outperforms models up to 70B scale, such as Qwen3-32B, Qwen2.5-72B,
and Llama3.3-70B, while using fewer parameters and less data. Smaller models
show similar trends: the Falcon-H1-1.5B-Deep rivals current leading 7B-10B
models, and Falcon-H1-0.5B performs comparably to typical 7B models from 2024.
These models excel across reasoning, mathematics, multilingual tasks,
instruction following, and scientific knowledge. With support for up to 256K
context tokens and 18 languages, Falcon-H1 is suitable for a wide range of
applications. All models are released under a permissive open-source license,
underscoring our commitment to accessible and impactful AI research.

### 2. [IFEvalCode: Controlled Code Generation](http://arxiv.org/pdf/2507.22462v1)

Authors: Jian Yang, Wei Zhang, Shukai Liu, Linzheng Chai, Yingshui Tan, Jiaheng Liu, Ge Zhang, Wangchunshu Zhou, Guanglin Niu, Zhoujun Li, Binyuan Hui, Junyang Lin

Code large language models (Code LLMs) have made significant progress in code
generation by translating natural language descriptions into functional code;
however, real-world applications often demand stricter adherence to detailed
requirements such as coding style, line count, and structural constraints,
beyond mere correctness. To address this, the paper introduces forward and
backward constraints generation to improve the instruction-following
capabilities of Code LLMs in controlled code generation, ensuring outputs align
more closely with human-defined guidelines. The authors further present
IFEvalCode, a multilingual benchmark comprising 1.6K test samples across seven
programming languages (Python, Java, JavaScript, TypeScript, Shell, C++, and
C#), with each sample featuring both Chinese and English queries. Unlike
existing benchmarks, IFEvalCode decouples evaluation into two metrics:
correctness (Corr.) and instruction-following (Instr.), enabling a more nuanced
assessment. Experiments on over 40 LLMs reveal that closed-source models
outperform open-source ones in controllable code generation and highlight a
significant gap between the models' ability to generate correct code versus
code that precisely follows instructions.

### 3. [SLM-SQL: An Exploration of Small Language Models for Text-to-SQL](http://arxiv.org/pdf/2507.22478v1)

Authors: Lei Sheng, Shuai-Shuai Xu

Large language models (LLMs) have demonstrated strong performance in
translating natural language questions into SQL queries (Text-to-SQL). In
contrast, small language models (SLMs) ranging from 0.5B to 1.5B parameters
currently underperform on Text-to-SQL tasks due to their limited logical
reasoning capabilities. However, SLMs offer inherent advantages in inference
speed and suitability for edge deployment. To explore their potential in
Text-to-SQL applications, we leverage recent advancements in post-training
techniques. Specifically, we used the open-source SynSQL-2.5M dataset to
construct two derived datasets: SynSQL-Think-916K for SQL generation and
SynSQL-Merge-Think-310K for SQL merge revision. We then applied supervised
fine-tuning and reinforcement learning-based post-training to the SLM, followed
by inference using a corrective self-consistency approach. Experimental results
validate the effectiveness and generalizability of our method, SLM-SQL. On the
BIRD development set, the five evaluated models achieved an average improvement
of 31.4 points. Notably, the 0.5B model reached 56.87\% execution accuracy
(EX), while the 1.5B model achieved 67.08\% EX. We will release our dataset,
model, and code to github: https://github.com/CycloneBoy/slm_sql.

### 4. [A Benchmark Dataset and Evaluation Framework for Vietnamese Large Language Models in Customer Support](http://arxiv.org/pdf/2507.22542v1)

Authors: Long S. T. Nguyen, Truong P. Hua, Thanh M. Nguyen, Toan Q. Pham, Nam K. Ngo, An X. Nguyen, Nghi D. M. Pham, Nghia H. Nguyen, Tho T. Quan

With the rapid growth of Artificial Intelligence, Large Language Models
(LLMs) have become essential for Question Answering (QA) systems, improving
efficiency and reducing human workload in customer service. The emergence of
Vietnamese LLMs (ViLLMs) highlights lightweight open-source models as a
practical choice for their accuracy, efficiency, and privacy benefits. However,
domain-specific evaluations remain limited, and the absence of benchmark
datasets reflecting real customer interactions makes it difficult for
enterprises to select suitable models for support applications. To address this
gap, we introduce the Customer Support Conversations Dataset (CSConDa), a
curated benchmark of over 9,000 QA pairs drawn from real interactions with
human advisors at a large Vietnamese software company. Covering diverse topics
such as pricing, product availability, and technical troubleshooting, CSConDa
provides a representative basis for evaluating ViLLMs in practical scenarios.
We further present a comprehensive evaluation framework, benchmarking 11
lightweight open-source ViLLMs on CSConDa with both automatic metrics and
syntactic analysis to reveal model strengths, weaknesses, and linguistic
patterns. This study offers insights into model behavior, explains performance
differences, and identifies key areas for improvement, supporting the
development of next-generation ViLLMs. By establishing a robust benchmark and
systematic evaluation, our work enables informed model selection for customer
service QA and advances research on Vietnamese LLMs. The dataset is publicly
available at
https://huggingface.co/datasets/ura-hcmut/Vietnamese-Customer-Support-QA.

### 5. [ControlMed: Adding Reasoning Control to Medical Language Model](http://arxiv.org/pdf/2507.22545v1)

Authors: Sung-Min Lee, Siyoon Lee, Juyeon Kim, Kyungmin Roh

Reasoning Large Language Models (LLMs) with enhanced accuracy and
explainability are increasingly being adopted in the medical domain, as the
life-critical nature of clinical decision-making demands reliable support.
Despite these advancements, existing reasoning LLMs often generate
unnecessarily lengthy reasoning processes, leading to significant computational
overhead and response latency. These limitations hinder their practical
deployment in real-world clinical environments. To address these challenges, we
introduce \textbf{ControlMed}, a medical language model that enables users to
actively control the length of the reasoning process at inference time through
fine-grained control markers. ControlMed is trained through a three-stage
pipeline: 1) pre-training on a large-scale synthetic medical instruction
dataset covering both \textit{direct} and \textit{reasoning responses}; 2)
supervised fine-tuning with multi-length reasoning data and explicit
length-control markers; and 3) reinforcement learning with model-based reward
signals to enhance factual accuracy and response quality. Experimental results
on a variety of English and Korean medical benchmarks demonstrate that our
model achieves similar or better performance compared to state-of-the-art
models. Furthermore, users can flexibly balance reasoning accuracy and
computational efficiency by controlling the reasoning length as needed. These
findings demonstrate that ControlMed is a practical and adaptable solution for
clinical question answering and medical information analysis.

### 6. [Language Arithmetics: Towards Systematic Language Neuron Identification and Manipulation](http://arxiv.org/pdf/2507.22608v1)

Authors: Daniil Gurgurov, Katharina Trinley, Yusser Al Ghussin, Tanja Baeumel, Josef van Genabith, Simon Ostermann

Large language models (LLMs) exhibit strong multilingual abilities, yet the
neural mechanisms behind language-specific processing remain unclear. We
analyze language-specific neurons in Llama-3.1-8B, Mistral-Nemo-12B, and
Aya-Expanse-8B & 32B across 21 typologically diverse languages, identifying
neurons that control language behavior. Using the Language Activation
Probability Entropy (LAPE) method, we show that these neurons cluster in deeper
layers, with non-Latin scripts showing greater specialization. Related
languages share overlapping neurons, reflecting internal representations of
linguistic proximity.
  Through language arithmetics, i.e. systematic activation addition and
multiplication, we steer models to deactivate unwanted languages and activate
desired ones, outperforming simpler replacement approaches. These interventions
effectively guide behavior across five multilingual tasks: language forcing,
translation, QA, comprehension, and NLI. Manipulation is more successful for
high-resource languages, while typological similarity improves effectiveness.
We also demonstrate that cross-lingual neuron steering enhances downstream
performance and reveal internal "fallback" mechanisms for language selection
when neurons are progressively deactivated. Our code is made publicly available
at https://github.com/d-gurgurov/Language-Neurons-Manipulation.

### 7. [Multilingual Political Views of Large Language Models: Identification and Steering](http://arxiv.org/pdf/2507.22623v1)

Authors: Daniil Gurgurov, Katharina Trinley, Ivan Vykopal, Josef van Genabith, Simon Ostermann, Roberto Zamparelli

Large language models (LLMs) are increasingly used in everyday tools and
applications, raising concerns about their potential influence on political
views. While prior research has shown that LLMs often exhibit measurable
political biases--frequently skewing toward liberal or progressive
positions--key gaps remain. Most existing studies evaluate only a narrow set of
models and languages, leaving open questions about the generalizability of
political biases across architectures, scales, and multilingual settings.
Moreover, few works examine whether these biases can be actively controlled.
  In this work, we address these gaps through a large-scale study of political
orientation in modern open-source instruction-tuned LLMs. We evaluate seven
models, including LLaMA-3.1, Qwen-3, and Aya-Expanse, across 14 languages using
the Political Compass Test with 11 semantically equivalent paraphrases per
statement to ensure robust measurement. Our results reveal that larger models
consistently shift toward libertarian-left positions, with significant
variations across languages and model families. To test the manipulability of
political stances, we utilize a simple center-of-mass activation intervention
technique and show that it reliably steers model responses toward alternative
ideological positions across multiple languages. Our code is publicly available
at https://github.com/d-gurgurov/Political-Ideologies-LLMs.

### 8. [From Sufficiency to Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented Reasoning for LLMs](http://arxiv.org/pdf/2507.22716v1)

Authors: Jie He, Victor Gutierrez Basulto, Jeff Z. Pan

Reinforcement learning-based retrieval-augmented generation (RAG) methods
enhance the reasoning abilities of large language models (LLMs). However, most
rely only on final-answer rewards, overlooking intermediate reasoning quality.
This paper analyzes existing RAG reasoning models and identifies three main
failure patterns: (1) information insufficiency, meaning the model fails to
retrieve adequate support; (2) faulty reasoning, where logical or content-level
flaws appear despite sufficient information; and (3) answer-reasoning
inconsistency, where a valid reasoning chain leads to a mismatched final
answer. We propose TIRESRAG-R1, a novel framework using a
think-retrieve-reflect process and a multi-dimensional reward system to improve
reasoning and stability. TIRESRAG-R1 introduces: (1) a sufficiency reward to
encourage thorough retrieval; (2) a reasoning quality reward to assess the
rationality and accuracy of the reasoning chain; and (3) a reflection reward to
detect and revise errors. It also employs a difficulty-aware reweighting
strategy and training sample filtering to boost performance on complex tasks.
Experiments on four multi-hop QA datasets show that TIRESRAG-R1 outperforms
prior RAG methods and generalizes well to single-hop tasks. The code and data
are available at: https://github.com/probe2/TIRESRAG-R1.

### 9. [Investigating Hallucination in Conversations for Low Resource Languages](http://arxiv.org/pdf/2507.22720v1)

Authors: Amit Das, Md. Najib Hasan, Souvika Sarkar, Zheng Zhang, Fatemeh Jamshidi, Tathagata Bhattacharya, Nilanjana Raychawdhury, Dongji Feng, Vinija Jain, Aman Chadha

Large Language Models (LLMs) have demonstrated remarkable proficiency in
generating text that closely resemble human writing. However, they often
generate factually incorrect statements, a problem typically referred to as
'hallucination'. Addressing hallucination is crucial for enhancing the
reliability and effectiveness of LLMs. While much research has focused on
hallucinations in English, our study extends this investigation to
conversational data in three languages: Hindi, Farsi, and Mandarin. We offer a
comprehensive analysis of a dataset to examine both factual and linguistic
errors in these languages for GPT-3.5, GPT-4o, Llama-3.1, Gemma-2.0,
DeepSeek-R1 and Qwen-3. We found that LLMs produce very few hallucinated
responses in Mandarin but generate a significantly higher number of
hallucinations in Hindi and Farsi.

### 10. [Resource-Efficient Adaptation of Large Language Models for Text Embeddings via Prompt Engineering and Contrastive Fine-tuning](http://arxiv.org/pdf/2507.22729v1)

Authors: Benedikt Roth, Stephan Rappensperger, Tianming Qiu, Hamza Imamović, Julian Wörmann, Hao Shen

Large Language Models (LLMs) have become a cornerstone in Natural Language
Processing (NLP), achieving impressive performance in text generation. Their
token-level representations capture rich, human-aligned semantics. However,
pooling these vectors into a text embedding discards crucial information.
Nevertheless, many non-generative downstream tasks, such as clustering,
classification, or retrieval, still depend on accurate and controllable
sentence- or document-level embeddings. We explore several adaptation
strategies for pre-trained, decoder-only LLMs: (i) various aggregation
techniques for token embeddings, (ii) task-specific prompt engineering, and
(iii) text-level augmentation via contrastive fine-tuning. Combining these
components yields state-of-the-art performance on the English clustering track
of the Massive Text Embedding Benchmark (MTEB). An analysis of the attention
map further shows that fine-tuning shifts focus from prompt tokens to
semantically relevant words, indicating more effective compression of meaning
into the final hidden state. Our experiments demonstrate that LLMs can be
effectively adapted as text embedding models through a combination of prompt
engineering and resource-efficient contrastive fine-tuning on synthetically
generated positive pairs.

### Cryptography and Security

### 1. [Invisible Injections: Exploiting Vision-Language Models Through Steganographic Prompt Embedding](http://arxiv.org/pdf/2507.22304v1)

Authors: Chetan Pathade

Vision-language models (VLMs) have revolutionized multimodal AI applications
but introduce novel security vulnerabilities that remain largely unexplored. We
present the first comprehensive study of steganographic prompt injection
attacks against VLMs, where malicious instructions are invisibly embedded
within images using advanced steganographic techniques. Our approach
demonstrates that current VLM architectures can inadvertently extract and
execute hidden prompts during normal image processing, leading to covert
behavioral manipulation. We develop a multi-domain embedding framework
combining spatial, frequency, and neural steganographic methods, achieving an
overall attack success rate of 24.3% (plus or minus 3.2%, 95% CI) across
leading VLMs including GPT-4V, Claude, and LLaVA, with neural steganography
methods reaching up to 31.8%, while maintaining reasonable visual
imperceptibility (PSNR greater than 38 dB, SSIM greater than 0.94). Through
systematic evaluation on 12 diverse datasets and 8 state-of-the-art models, we
reveal moderate but meaningful vulnerabilities in current VLM architectures and
propose effective countermeasures. Our findings have significant implications
for VLM deployment in security-critical applications and highlight the need for
proportionate multimodal AI security frameworks.

### 2. [SleepWalk: Exploiting Context Switching and Residual Power for Physical Side-Channel Attacks](http://arxiv.org/pdf/2507.22306v1)

Authors: Sahan Sanjaya, Aruna Jayasena, Prabhat Mishra

Context switching is utilized by operating systems to change the execution
context between application programs. It involves saving and restoring the
states of multiple registers and performing a pipeline flush to remove any
pre-fetched instructions, leading to a higher instantaneous power consumption
compared to typical program execution. In this paper, we introduce a physical
power side-channel leakage source that exploits the power spike observed during
a context switch, triggered by the inbuilt sleep function of the system kernel.
We observed that this power spike directly correlates with both the power
consumption during context switching and the residual power consumption of the
previously executed program. Notably, the persistence of residual power
signatures from previous workloads extends the scope of this side-channel
beyond extracting the data in registers during the context switch. Unlike
traditional approaches that require analyzing full power traces, applying
complex preprocessing, or relying on external synchronization triggers, this
novel technique leverages only the amplitude of a single power spike,
significantly simplifying the attack. We developed a power model to illustrate
the feasibility of mounting end-to-end side-channel attacks using the
sleep-induced power spikes. Experimental evaluation demonstrates that our
framework can successfully perform cryptographic key recovery for both AES and
SIKE implementations on Broadcom BCM2711.

### 3. [Benchmarking Fraud Detectors on Private Graph Data](http://arxiv.org/pdf/2507.22347v1)

Authors: Alexander Goldberg, Giulia Fanti, Nihar Shah, Zhiwei Steven Wu

We introduce the novel problem of benchmarking fraud detectors on private
graph-structured data. Currently, many types of fraud are managed in part by
automated detection algorithms that operate over graphs. We consider the
scenario where a data holder wishes to outsource development of fraud detectors
to third parties (e.g., vendors or researchers). The third parties submit their
fraud detectors to the data holder, who evaluates these algorithms on a private
dataset and then publicly communicates the results. We propose a realistic
privacy attack on this system that allows an adversary to de-anonymize
individuals' data based only on the evaluation results. In simulations of a
privacy-sensitive benchmark for facial recognition algorithms by the National
Institute of Standards and Technology (NIST), our attack achieves near perfect
accuracy in identifying whether individuals' data is present in a private
dataset, with a True Positive Rate of 0.98 at a False Positive Rate of 0.00. We
then study how to benchmark algorithms while satisfying a formal differential
privacy (DP) guarantee. We empirically evaluate two classes of solutions:
subsample-and-aggregate and DP synthetic graph data. We demonstrate through
extensive experiments that current approaches do not provide utility when
guaranteeing DP. Our results indicate that the error arising from DP trades off
between bias from distorting graph structure and variance from adding random
noise. Current methods lie on different points along this bias-variance
trade-off, but more complex methods tend to require high-variance noise
addition, undermining utility.

### 4. [DoS Attacks and Defense Technologies in Blockchain Systems: A Hierarchical Analysis](http://arxiv.org/pdf/2507.22611v1)

Authors: Chunyi Zhang, Fengjiao Dou, Xiaoqi Li

Blockchain technology is widely used in various fields due to its ability to
provide decentralization and trustless security. This is a fundamental
understanding held by many advocates, but it is misunderstood, leading
participants to fail to recognize the limitations of the security that
blockchain can provide. Among all current network attacks, Denial of Service
(DoS) attacks pose significant threats due to their ease of execution and
destructive potential. This paper, based on the blockchain architecture
hierarchy, categorizes and organizes existing DoS attacks, with a focus on
explaining the principles and methods of contract layer and consensus layer DoS
attacks. Furthermore, this paper comprehensively analyzes and compares commonly
used detection methods and defense technologies, which will contribute to
strengthening the security and stability of blockchain systems and promoting
further innovation and application of blockchain systems.

### 5. [Cryptanalysis of LC-MUME: A Lightweight Certificateless Multi-User Matchmaking Encryption for Mobile Devices](http://arxiv.org/pdf/2507.22674v1)

Authors: Ramprasad Sarkar

Yang et al. proposed a lightweight certificateless multiuser matchmaking
encryption (LC-MUME) scheme for mobile devices, published in IEEE Transactions
on Information Forensics and Security (TIFS) (DOI: 10.1109/TIFS.2023.3321961).
Their construction aims to reduce computational and communication overhead
within a one-to-many certificateless cryptographic framework. The authors claim
that their scheme satisfies existential unforgeability under chosen-message
attacks (EUF-CMA) in the random oracle model. However, our cryptanalytic study
demonstrates that the scheme fails to meet this critical security requirement.
In particular, we show that a Type-I adversary can successfully forge a valid
ciphertext without possessing the complete private key of the sender. Both
theoretical analysis and practical implementation confirm that this attack can
be mounted with minimal computational cost. To address these weaknesses, we
propose a modification strategy to strengthen the security of matchmaking
encryption schemes in mobile computing environments.

### 6. [Breaking Obfuscation: Cluster-Aware Graph with LLM-Aided Recovery for Malicious JavaScript Detection](http://arxiv.org/pdf/2507.22447v1)

Authors: Zhihong Liang, Xin Wang, Zhenhuang Hu, Liangliang Song, Lin Chen, Jingjing Guo, Yanbin Wang, Ye Tian

With the rapid expansion of web-based applications and cloud services,
malicious JavaScript code continues to pose significant threats to user
privacy, system integrity, and enterprise security. But, detecting such threats
remains challenging due to sophisticated code obfuscation techniques and
JavaScript's inherent language characteristics, particularly its nested closure
structures and syntactic flexibility. In this work, we propose DeCoda, a hybrid
defense framework that combines large language model (LLM)-based deobfuscation
with code graph learning: (1) We first construct a sophisticated
prompt-learning pipeline with multi-stage refinement, where the LLM
progressively reconstructs the original code structure from obfuscated inputs
and then generates normalized Abstract Syntax Tree (AST) representations; (2)
In JavaScript ASTs, dynamic typing scatters semantically similar nodes while
deeply nested functions fracture scope capturing, introducing structural noise
and semantic ambiguity. To address these challenges, we then propose to learn
hierarchical code graph representations via a Cluster-wise Graph that
synergistically integrates graph transformer network, node clustering, and
node-to-cluster attention to simultaneously capture both local node-level
semantics and global cluster-induced structural relationships from AST graph.
Experimental results demonstrate that our method achieves F1-scores of 94.64%
and 97.71% on two benchmark datasets, demonstrating absolute improvements of
10.74% and 13.85% over state-of-the-art baselines. In false-positive control
evaluation at fixed FPR levels (0.0001, 0.001, 0.01), our approach delivers
4.82, 5.91, and 2.53 higher TPR respectively compared to the best-performing
baseline. These results highlight the effectiveness of LLM-based deobfuscation
and underscore the importance of modeling cluster-level relationships in
detecting malicious code.

### 7. [Scalable and (quantum-accessible) adaptive pseudorandom quantum states and pseudorandom function-like quantum state generators](http://arxiv.org/pdf/2507.22535v1)

Authors: Rishabh Batra, Zhili Chen, Rahul Jain, YaoNan Zhang

Pseudorandom quantum states (PRS) and pseudorandom function-like quantum
state (PRFS) generators are quantum analogues of pseudorandom generators and
pseudorandom functions. It is known that PRS (and PRFS) can exist even if BQP =
QMA (relative to a quantum oracle) or if P = NP (relative to a classical
oracle), which does not allow for the existence of one-way functions (relative
to these oracles). Hence, these are potentially weaker objects than
quantum-secure one-way functions, which can be used to do quantum cryptography.
A desirable property of PRS and PRFS constructions is scalability, which
ensures that the security parameter $\lambda$ (which determines
indistinguishability from their Haar-random counterparts) is much larger than
$n$ (the number of qubits of the output states). This may be important in some
applications where PRS and PRFS primitives are used.
  We present an isometric procedure to prepare quantum states that can be
arbitrarily random (i.e., the trace distance from the Haar-random state can be
arbitrarily small for the true random case, or the distinguishing advantage can
be arbitrarily small for the pseudorandom case). Our procedure provides a new
method for scalable PRS that introduces no entanglement or correlations with
the environment. This naturally gives the first construction for scalable and
(quantum-accessible) adaptive PRFS assuming quantum-secure one-way functions.
Our PRFS construction implies various primitives, including long-input PRFS,
short-input PRFS, short-output PRFS, non-adaptive PRFS, and
classical-accessible adaptive PRFS. This new construction may be helpful in
some simplification of the microcrypt zoo.

### 8. [Hate in Plain Sight: On the Risks of Moderating AI-Generated Hateful Illusions](http://arxiv.org/pdf/2507.22617v1)

Authors: Yiting Qu, Ziqing Yang, Yihan Ma, Michael Backes, Savvas Zannettou, Yang Zhang

Recent advances in text-to-image diffusion models have enabled the creation
of a new form of digital art: optical illusions--visual tricks that create
different perceptions of reality. However, adversaries may misuse such
techniques to generate hateful illusions, which embed specific hate messages
into harmless scenes and disseminate them across web communities. In this work,
we take the first step toward investigating the risks of scalable hateful
illusion generation and the potential for bypassing current content moderation
models. Specifically, we generate 1,860 optical illusions using Stable
Diffusion and ControlNet, conditioned on 62 hate messages. Of these, 1,571 are
hateful illusions that successfully embed hate messages, either overtly or
subtly, forming the Hateful Illusion dataset. Using this dataset, we evaluate
the performance of six moderation classifiers and nine vision language models
(VLMs) in identifying hateful illusions. Experimental results reveal
significant vulnerabilities in existing moderation models: the detection
accuracy falls below 0.245 for moderation classifiers and below 0.102 for VLMs.
We further identify a critical limitation in their vision encoders, which
mainly focus on surface-level image details while overlooking the secondary
layer of information, i.e., hidden messages. To address this risk, we explore
preliminary mitigation measures and identify the most effective approaches from
the perspectives of image transformations and training-level strategies.

### 9. [Cycles Protocol: A Peer-to-Peer Electronic Clearing System](http://arxiv.org/pdf/2507.22309v1)

Authors: Ethan Buchman, Paolo Dini, Shoaib Ahmed, Andrew Miller, Tomaž Fleischman

For centuries, financial institutions have responded to liquidity challenges
by forming closed, centralized clearing clubs with strict rules and membership
that allow them to collaborate on using the least money to discharge the most
debt. As closed clubs, much of the general public has been excluded from
participation. But the vast majority of private sector actors consists of micro
or small firms that are vulnerable to late payments and generally ineligible
for bank loans. This low liquidity environment often results in gridlock and
leads to insolvency, and it disproportionately impacts small enterprises and
communities.
  On the other hand, blockchain communities have developed open, decentralized
settlement systems, along with a proliferation of store of value assets and new
lending protocols, allowing anyone to permissionlessly transact and access
credit. However, these protocols remain used primarily for speculative
purposes, and so far have fallen short of the large-scale positive impact on
the real economy prophesied by their promoters.
  We address these challenges by introducing Cycles, an open, decentralized
clearing, settlement, and issuance protocol. Cycles is designed to enable firms
to overcome payment inefficiencies, to reduce their working capital costs, and
to leverage diverse assets and liquidity sources, including cryptocurrencies,
stablecoins, and lending protocols, in service of clearing more debt with less
money. Cycles solves real world liquidity challenges through a
privacy-preserving multilateral settlement platform based on a graph
optimization algorithm. The design is based on a core insight: liquidity
resides within cycles in the payment network's structure and can be accessed
via settlement flows optimized to reduce debt.

### 10. [SAEL: Leveraging Large Language Models with Adaptive Mixture-of-Experts for Smart Contract Vulnerability Detection](http://arxiv.org/pdf/2507.22371v1)

Authors: Lei Yu, Shiqi Cheng, Zhirong Huang, Jingyuan Zhang, Chenjie Shen, Junyi Lu, Li Yang, Fengjun Zhang, Jiajia Ma

With the increasing security issues in blockchain, smart contract
vulnerability detection has become a research focus. Existing vulnerability
detection methods have their limitations: 1) Static analysis methods struggle
with complex scenarios. 2) Methods based on specialized pre-trained models
perform well on specific datasets but have limited generalization capabilities.
In contrast, general-purpose Large Language Models (LLMs) demonstrate
impressive ability in adapting to new vulnerability patterns. However, they
often underperform on specific vulnerability types compared to methods based on
specialized pre-trained models. We also observe that explanations generated by
general-purpose LLMs can provide fine-grained code understanding information,
contributing to improved detection performance.
  Inspired by these observations, we propose SAEL, an LLM-based framework for
smart contract vulnerability detection. We first design targeted prompts to
guide LLMs in identifying vulnerabilities and generating explanations, which
serve as prediction features. Next, we apply prompt-tuning on CodeT5 and T5 to
process contract code and explanations, enhancing task-specific performance. To
combine the strengths of each approach, we introduce an Adaptive
Mixture-of-Experts architecture. This dynamically adjusts feature weights via a
Gating Network, which selects relevant features using TopK filtering and
Softmax normalization, and incorporates a Multi-Head Self-Attention mechanism
to enhance cross-feature relationships. This design enables effective
integration of LLM predictions, explanation features, and code features through
gradient optimization. The loss function jointly considers both independent
feature performance and overall weighted predictions. Experiments show that
SAEL outperforms existing methods across various vulnerabilities.

### Computer Vision and Pattern Recognition

### 1. [LAMA-Net: A Convergent Network Architecture for Dual-Domain Reconstruction](http://arxiv.org/pdf/2507.22316v1)

Authors: Chi Ding, Qingchao Zhang, Ge Wang, Xiaojing Ye, Yunmei Chen

We propose a learnable variational model that learns the features and
leverages complementary information from both image and measurement domains for
image reconstruction. In particular, we introduce a learned alternating
minimization algorithm (LAMA) from our prior work, which tackles two-block
nonconvex and nonsmooth optimization problems by incorporating a residual
learning architecture in a proximal alternating framework. In this work, our
goal is to provide a complete and rigorous convergence proof of LAMA and show
that all accumulation points of a specified subsequence of LAMA must be Clarke
stationary points of the problem. LAMA directly yields a highly interpretable
neural network architecture called LAMA-Net. Notably, in addition to the
results shown in our prior work, we demonstrate that the convergence property
of LAMA yields outstanding stability and robustness of LAMA-Net in this work.
We also show that the performance of LAMA-Net can be further improved by
integrating a properly designed network that generates suitable initials, which
we call iLAMA-Net. To evaluate LAMA-Net/iLAMA-Net, we conduct several
experiments and compare them with several state-of-the-art methods on popular
benchmark datasets for Sparse-View Computed Tomography.

### 2. [UFV-Splatter: Pose-Free Feed-Forward 3D Gaussian Splatting Adapted to Unfavorable Views](http://arxiv.org/pdf/2507.22342v1)

Authors: Yuki Fujimura, Takahiro Kushida, Kazuya Kitano, Takuya Funatomi, Yasuhiro Mukaigawa

This paper presents a pose-free, feed-forward 3D Gaussian Splatting (3DGS)
framework designed to handle unfavorable input views. A common rendering setup
for training feed-forward approaches places a 3D object at the world origin and
renders it from cameras pointed toward the origin -- i.e., from favorable
views, limiting the applicability of these models to real-world scenarios
involving varying and unknown camera poses. To overcome this limitation, we
introduce a novel adaptation framework that enables pretrained pose-free
feed-forward 3DGS models to handle unfavorable views. We leverage priors
learned from favorable images by feeding recentered images into a pretrained
model augmented with low-rank adaptation (LoRA) layers. We further propose a
Gaussian adapter module to enhance the geometric consistency of the Gaussians
derived from the recentered inputs, along with a Gaussian alignment method to
render accurate target views for training. Additionally, we introduce a new
training strategy that utilizes an off-the-shelf dataset composed solely of
favorable images. Experimental results on both synthetic images from the Google
Scanned Objects dataset and real images from the OmniObject3D dataset validate
the effectiveness of our method in handling unfavorable input views.

### 3. [DeltaVLM: Interactive Remote Sensing Image Change Analysis via Instruction-guided Difference Perception](http://arxiv.org/pdf/2507.22346v1)

Authors: Pei Deng, Wenqian Zhou, Hanlin Wu

Accurate interpretation of land-cover changes in multi-temporal satellite
imagery is critical for real-world scenarios. However, existing methods
typically provide only one-shot change masks or static captions, limiting their
ability to support interactive, query-driven analysis. In this work, we
introduce remote sensing image change analysis (RSICA) as a new paradigm that
combines the strengths of change detection and visual question answering to
enable multi-turn, instruction-guided exploration of changes in bi-temporal
remote sensing images. To support this task, we construct ChangeChat-105k, a
large-scale instruction-following dataset, generated through a hybrid
rule-based and GPT-assisted process, covering six interaction types: change
captioning, classification, quantification, localization, open-ended question
answering, and multi-turn dialogues. Building on this dataset, we propose
DeltaVLM, an end-to-end architecture tailored for interactive RSICA. DeltaVLM
features three innovations: (1) a fine-tuned bi-temporal vision encoder to
capture temporal differences; (2) a visual difference perception module with a
cross-semantic relation measuring (CSRM) mechanism to interpret changes; and
(3) an instruction-guided Q-former to effectively extract query-relevant
difference information from visual changes, aligning them with textual
instructions. We train DeltaVLM on ChangeChat-105k using a frozen large
language model, adapting only the vision and alignment modules to optimize
efficiency. Extensive experiments and ablation studies demonstrate that
DeltaVLM achieves state-of-the-art performance on both single-turn captioning
and multi-turn interactive change analysis, outperforming existing multimodal
large language models and remote sensing vision-language models. Code, dataset
and pre-trained weights are available at https://github.com/hanlinwu/DeltaVLM.

### 4. [FaceGCD: Generalized Face Discovery via Dynamic Prefix Generation](http://arxiv.org/pdf/2507.22353v1)

Authors: Yunseok Oh, Dong-Wan Choi

Recognizing and differentiating among both familiar and unfamiliar faces is a
critical capability for face recognition systems and a key step toward
artificial general intelligence (AGI). Motivated by this ability, this paper
introduces generalized face discovery (GFD), a novel open-world face
recognition task that unifies traditional face identification with generalized
category discovery (GCD). GFD requires recognizing both labeled and unlabeled
known identities (IDs) while simultaneously discovering new, previously unseen
IDs. Unlike typical GCD settings, GFD poses unique challenges due to the high
cardinality and fine-grained nature of face IDs, rendering existing GCD
approaches ineffective. To tackle this problem, we propose FaceGCD, a method
that dynamically constructs instance-specific feature extractors using
lightweight, layer-wise prefixes. These prefixes are generated on the fly by a
HyperNetwork, which adaptively outputs a set of prefix generators conditioned
on each input image. This dynamic design enables FaceGCD to capture subtle
identity-specific cues without relying on high-capacity static models.
Extensive experiments demonstrate that FaceGCD significantly outperforms
existing GCD methods and a strong face recognition baseline, ArcFace, achieving
state-of-the-art results on the GFD task and advancing toward open-world face
recognition.

### 5. [On the Reliability of Vision-Language Models Under Adversarial Frequency-Domain Perturbations](http://arxiv.org/pdf/2507.22398v1)

Authors: Jordan Vice, Naveed Akhtar, Yansong Gao, Richard Hartley, Ajmal Mian

Vision-Language Models (VLMs) are increasingly used as perceptual modules for
visual content reasoning, including through captioning and DeepFake detection.
In this work, we expose a critical vulnerability of VLMs when exposed to
subtle, structured perturbations in the frequency domain. Specifically, we
highlight how these feature transformations undermine authenticity/DeepFake
detection and automated image captioning tasks. We design targeted image
transformations, operating in the frequency domain to systematically adjust VLM
outputs when exposed to frequency-perturbed real and synthetic images. We
demonstrate that the perturbation injection method generalizes across five
state-of-the-art VLMs which includes different-parameter Qwen2/2.5 and BLIP
models. Experimenting across ten real and generated image datasets reveals that
VLM judgments are sensitive to frequency-based cues and may not wholly align
with semantic content. Crucially, we show that visually-imperceptible spatial
frequency transformations expose the fragility of VLMs deployed for automated
image captioning and authenticity detection tasks. Our findings under
realistic, black-box constraints challenge the reliability of VLMs,
underscoring the need for robust multimodal perception systems.

### 6. [UAVScenes: A Multi-Modal Dataset for UAVs](http://arxiv.org/pdf/2507.22412v1)

Authors: Sijie Wang, Siqi Li, Yawei Zhang, Shangshu Yu, Shenghai Yuan, Rui She, Quanjiang Guo, JinXuan Zheng, Ong Kang Howe, Leonrich Chandra, Shrivarshann Srijeyan, Aditya Sivadas, Toshan Aggarwal, Heyuan Liu, Hongming Zhang, Chujie Chen, Junyu Jiang, Lihua Xie, Wee Peng Tay

Multi-modal perception is essential for unmanned aerial vehicle (UAV)
operations, as it enables a comprehensive understanding of the UAVs'
surrounding environment. However, most existing multi-modal UAV datasets are
primarily biased toward localization and 3D reconstruction tasks, or only
support map-level semantic segmentation due to the lack of frame-wise
annotations for both camera images and LiDAR point clouds. This limitation
prevents them from being used for high-level scene understanding tasks. To
address this gap and advance multi-modal UAV perception, we introduce
UAVScenes, a large-scale dataset designed to benchmark various tasks across
both 2D and 3D modalities. Our benchmark dataset is built upon the
well-calibrated multi-modal UAV dataset MARS-LVIG, originally developed only
for simultaneous localization and mapping (SLAM). We enhance this dataset by
providing manually labeled semantic annotations for both frame-wise images and
LiDAR point clouds, along with accurate 6-degree-of-freedom (6-DoF) poses.
These additions enable a wide range of UAV perception tasks, including
segmentation, depth estimation, 6-DoF localization, place recognition, and
novel view synthesis (NVS). Our dataset is available at
https://github.com/sijieaaa/UAVScenes

### 7. [HQ-CLIP: Leveraging Large Vision-Language Models to Create High-Quality Image-Text Datasets and CLIP Models](http://arxiv.org/pdf/2507.22431v1)

Authors: Zhixiang Wei, Guangting Wang, Xiaoxiao Ma, Ke Mei, Huaian Chen, Yi Jin, Fengyun Rao

Large-scale but noisy image-text pair data have paved the way for the success
of Contrastive Language-Image Pretraining (CLIP). As the foundation vision
encoder, CLIP in turn serves as the cornerstone for most large vision-language
models (LVLMs). This interdependence naturally raises an interesting question:
Can we reciprocally leverage LVLMs to enhance the quality of image-text pair
data, thereby opening the possibility of a self-reinforcing cycle for
continuous improvement? In this work, we take a significant step toward this
vision by introducing an LVLM-driven data refinement pipeline. Our framework
leverages LVLMs to process images and their raw alt-text, generating four
complementary textual formulas: long positive descriptions, long negative
descriptions, short positive tags, and short negative tags. Applying this
pipeline to the curated DFN-Large dataset yields VLM-150M, a refined dataset
enriched with multi-grained annotations. Based on this dataset, we further
propose a training paradigm that extends conventional contrastive learning by
incorporating negative descriptions and short tags as additional supervised
signals. The resulting model, namely HQ-CLIP, demonstrates remarkable
improvements across diverse benchmarks. Within a comparable training data
scale, our approach achieves state-of-the-art performance in zero-shot
classification, cross-modal retrieval, and fine-grained visual understanding
tasks. In retrieval benchmarks, HQ-CLIP even surpasses standard CLIP models
trained on the DFN-2B dataset, which contains 10$\times$ more training data
than ours. All code, data, and models are available at
https://zxwei.site/hqclip.

### 8. [From Sharp to Blur: Unsupervised Domain Adaptation for 2D Human Pose Estimation Under Extreme Motion Blur Using Event Cameras](http://arxiv.org/pdf/2507.22438v1)

Authors: Youngho Kim, Hoonhee Cho, Kuk-Jin Yoon

Human pose estimation is critical for applications such as rehabilitation,
sports analytics, and AR/VR systems. However, rapid motion and low-light
conditions often introduce motion blur, significantly degrading pose estimation
due to the domain gap between sharp and blurred images. Most datasets assume
stable conditions, making models trained on sharp images struggle in blurred
environments. To address this, we introduce a novel domain adaptation approach
that leverages event cameras, which capture high temporal resolution motion
data and are inherently robust to motion blur. Using event-based augmentation,
we generate motion-aware blurred images, effectively bridging the domain gap
between sharp and blurred domains without requiring paired annotations.
Additionally, we develop a student-teacher framework that iteratively refines
pseudo-labels, leveraging mutual uncertainty masking to eliminate incorrect
labels and enable more effective learning. Experimental results demonstrate
that our approach outperforms conventional domain-adaptive human pose
estimation methods, achieving robust pose estimation under motion blur without
requiring annotations in the target domain. Our findings highlight the
potential of event cameras as a scalable and effective solution for domain
adaptation in real-world motion blur environments. Our project codes are
available at https://github.com/kmax2001/EvSharp2Blur.

### 9. [Exploiting Diffusion Prior for Task-driven Image Restoration](http://arxiv.org/pdf/2507.22459v1)

Authors: Jaeha Kim, Junghun Oh, Kyoung Mu Lee

Task-driven image restoration (TDIR) has recently emerged to address
performance drops in high-level vision tasks caused by low-quality (LQ) inputs.
Previous TDIR methods struggle to handle practical scenarios in which images
are degraded by multiple complex factors, leaving minimal clues for
restoration. This motivates us to leverage the diffusion prior, one of the most
powerful natural image priors. However, while the diffusion prior can help
generate visually plausible results, using it to restore task-relevant details
remains challenging, even when combined with recent TDIR methods. To address
this, we propose EDTR, which effectively harnesses the power of diffusion prior
to restore task-relevant details. Specifically, we propose directly leveraging
useful clues from LQ images in the diffusion process by generating from
pixel-error-based pre-restored LQ images with mild noise added. Moreover, we
employ a small number of denoising steps to prevent the generation of redundant
details that dilute crucial task-related information. We demonstrate that our
method effectively utilizes diffusion prior for TDIR, significantly enhancing
task performance and visual quality across diverse tasks with multiple complex
degradations.

### 10. [Estimating 2D Camera Motion with Hybrid Motion Basis](http://arxiv.org/pdf/2507.22480v1)

Authors: Haipeng Li, Tianhao Zhou, Zhanglei Yang, Yi Wu, Yan Chen, Zijing Mao, Shen Cheng, Bing Zeng, Shuaicheng Liu

Estimating 2D camera motion is a fundamental computer vision task that models
the projection of 3D camera movements onto the 2D image plane. Current methods
rely on either homography-based approaches, limited to planar scenes, or
meshflow techniques that use grid-based local homographies but struggle with
complex non-linear transformations. A key insight of our work is that combining
flow fields from different homographies creates motion patterns that cannot be
represented by any single homography. We introduce CamFlow, a novel framework
that represents camera motion using hybrid motion bases: physical bases derived
from camera geometry and stochastic bases for complex scenarios. Our approach
includes a hybrid probabilistic loss function based on the Laplace distribution
that enhances training robustness. For evaluation, we create a new benchmark by
masking dynamic objects in existing optical flow datasets to isolate pure
camera motion. Experiments show CamFlow outperforms state-of-the-art methods
across diverse scenarios, demonstrating superior robustness and generalization
in zero-shot settings. Code and datasets are available at our project page:
https://lhaippp.github.io/CamFlow/.

### Computers and Society

### 1. [Exploring Student-AI Interactions in Vibe Coding](http://arxiv.org/pdf/2507.22614v1)

Authors: Francis Geng, Anshul Shah, Haolin Li, Nawab Mulla, Steven Swanson, Gerald Soosai Raj, Daniel Zingaro, Leo Porter

Background and Context. Chat-based and inline-coding-based GenAI has already
had substantial impact on the CS Education community. The recent introduction
of ``vibe coding'' may further transform how students program, as it introduces
a new way for students to create software projects with minimal oversight.
  Objectives. The purpose of this study is to understand how students in
introductory programming and advanced software engineering classes interact
with a vibe coding platform (Replit) when creating software and how the
interactions differ by programming background.
  Methods. Interview participants were asked to think-aloud while building a
web application using Replit. Thematic analysis was then used to analyze the
video recordings with an emphasis on the interactions between the student and
Replit.
  Findings. For both groups, the majority of student interactions with Replit
were to test or debug the prototype and only rarely did students visit code.
Prompts by advanced software engineering students were much more likely to
include relevant app feature and codebase contexts than those by introductory
programming students.

### 2. [Towards Simulating Social Influence Dynamics with LLM-based Multi-agents](http://arxiv.org/pdf/2507.22467v1)

Authors: Hsien-Tsung Lin, Pei-Cing Huang, Chan-Tung Ku, Chan Hsu, Pei-Xuan Shieh, Yihuang Kang

Recent advancements in Large Language Models offer promising capabilities to
simulate complex human social interactions. We investigate whether LLM-based
multi-agent simulations can reproduce core human social dynamics observed in
online forums. We evaluate conformity dynamics, group polarization, and
fragmentation across different model scales and reasoning capabilities using a
structured simulation framework. Our findings indicate that smaller models
exhibit higher conformity rates, whereas models optimized for reasoning are
more resistant to social influence.

### 3. [Designing for Self-Regulation in Informal Programming Learning: Insights from a Storytelling-Centric Approach](http://arxiv.org/pdf/2507.22671v1)

Authors: Sami Saeed Alghamdi, Christopher Bull, Ahmed Kharrufa

Many people learn programming independently from online resources and often
report struggles in achieving their personal learning goals. Learners
frequently describe their experiences as isolating and frustrating, challenged
by abundant uncertainties, information overload, and distraction, compounded by
limited guidance. At the same time, social media serves as a personal space
where many engage in diverse self-regulation practices, including help-seeking,
using external memory aids (e.g., self-notes), self-reflection, emotion
regulation, and self-motivation. For instance, learners often mark achievements
and set milestones through their posts. In response, we developed a system
consisting of a web platform and browser extensions to support self-regulation
online. The design aims to add learner-defined structure to otherwise
unstructured experiences and bring meaning to curation and reflection
activities by translating them into learning stories with AI-generated
feedback. We position storytelling as an integrative approach to design that
connects resource curation, reflective and sensemaking practice, and narrative
practices learners already use across social platforms. We recruited 15
informal programming learners who are regular social media users to engage with
the system in a self-paced manner; participation concluded upon submitting a
learning story and survey. We used three quantitative scales and a qualitative
survey to examine users' characteristics and perceptions of the system's
support for their self-regulation. User feedback suggests the system's
viability as a self-regulation aid. Learners particularly valued in-situ
reflection, automated story feedback, and video annotation, while other
features received mixed views. We highlight perceived benefits, friction
points, and design opportunities for future AI-augmented self-regulation tools.

### 4. [The Incomplete Bridge: How AI Research (Mis)Engages with Psychology](http://arxiv.org/pdf/2507.22847v1)

Authors: Han Jiang, Pengda Wang, Xiaoyuan Yi, Xing Xie, Ziang Xiao

Social sciences have accumulated a rich body of theories and methodologies
for investigating the human mind and behaviors, while offering valuable
insights into the design and understanding of Artificial Intelligence (AI)
systems. Focusing on psychology as a prominent case, this study explores the
interdisciplinary synergy between AI and the field by analyzing 1,006
LLM-related papers published in premier AI venues between 2023 and 2025, along
with the 2,544 psychology publications they cite. Through our analysis, we
identify key patterns of interdisciplinary integration, locate the psychology
domains most frequently referenced, and highlight areas that remain
underexplored. We further examine how psychology theories/frameworks are
operationalized and interpreted, identify common types of misapplication, and
offer guidance for more effective incorporation. Our work provides a
comprehensive map of interdisciplinary engagement between AI and psychology,
thereby facilitating deeper collaboration and advancing AI systems.

### 5. [GeoOutageKG: A Multimodal Geospatiotemporal Knowledge Graph for Multiresolution Power Outage Analysis](http://arxiv.org/pdf/2507.22878v1)

Authors: Ethan Frakes, Yinghui Wu, Roger H. French, Mengjie Li

Detecting, analyzing, and predicting power outages is crucial for grid risk
assessment and disaster mitigation. Numerous outages occur each year,
exacerbated by extreme weather events such as hurricanes. Existing outage data
are typically reported at the county level, limiting their spatial resolution
and making it difficult to capture localized patterns. However, it offers
excellent temporal granularity. In contrast, nighttime light satellite image
data provides significantly higher spatial resolution and enables a more
comprehensive spatial depiction of outages, enhancing the accuracy of assessing
the geographic extent and severity of power loss after disaster events.
However, these satellite data are only available on a daily basis. Integrating
spatiotemporal visual and time-series data sources into a unified knowledge
representation can substantially improve power outage detection, analysis, and
predictive reasoning. In this paper, we propose GeoOutageKG, a multimodal
knowledge graph that integrates diverse data sources, including nighttime light
satellite image data, high-resolution spatiotemporal power outage maps, and
county-level timeseries outage reports in the U.S. We describe our method for
constructing GeoOutageKG by aligning source data with a developed ontology,
GeoOutageOnto. Currently, GeoOutageKG includes over 10.6 million individual
outage records spanning from 2014 to 2024, 300,000 NTL images spanning from
2012 to 2024, and 15,000 outage maps. GeoOutageKG is a novel, modular and
reusable semantic resource that enables robust multimodal data integration. We
demonstrate its use through multiresolution analysis of geospatiotemporal power
outages.

### Databases

### 1. [Is SHACL Suitable for Data Quality Assessment?](http://arxiv.org/pdf/2507.22305v1)

Authors: Carolina Cortés Lasalle, Lisa Ehrlinger, Lorena Etcheverry, Felix Naumann

Knowledge graphs have been widely adopted in both enterprises, such as the
Google Knowledge Graph, and open platforms like Wikidata to represent domain
knowledge and support analysis with artificial intelligence. They model
real-world information as nodes and edges. To embrace flexibility, knowledge
graphs often lack enforced schemas (i.e., ontologies), leading to potential
data quality issues, such as semantically overlapping nodes. Therefore,
ensuring their quality is essential, as issues in the data can affect
applications relying on them. To assess the quality of knowledge graphs,
existing works either propose high-level frameworks comprising various data
quality dimensions without concrete implementations, define tools that measure
data quality with ad-hoc SPARQL (SPARQL Protocol and RDF Query Language)
queries, or promote the usage of constraint languages, such as the Shapes
Constraint Language (SHACL), to assess and improve the quality of the graph.
Although the latter approaches claim to address data quality assessment, none
of them comprehensively tries to cover all data quality dimensions. In this
paper, we explore this gap by investigating the extent to which SHACL can be
used to assess data quality in knowledge graphs. Specifically, we defined SHACL
shapes for 69 data quality metrics proposed by Zaveri et al. [1] and
implemented a prototype that automatically instantiates these shapes and
computes the corresponding data quality measures from their validation results.
All resources are provided for repeatability at
https://github.com/caroocortes/SHACL-DQA-prototype/tree/main

### 2. [SAM: A Stability-Aware Cache Manager for Multi-Tenant Embedded Databases](http://arxiv.org/pdf/2507.22701v1)

Authors: Haoran Zhang, Decheng Zuo, Yu Yan, Zhiyu Liang, Hongzhi Wang

The co-location of multiple database instances on resource constrained edge
nodes creates significant cache contention, where traditional schemes are
inefficient and unstable under dynamic workloads. To address this, we present
SAM, an autonomic cache manager powered by our novel AURA algorithm. AURA makes
stability a first-class design principle by resolving the
exploitation-exploration dilemma: it achieves this by synthesizing two
orthogonal factors, which we introduce as: the H-factor, representing a
database's proven, historically stable efficiency (exploitation), and the
V-factor, representing its empirically estimated marginal gain for future
improvements (exploration). This dual-factor model, governed by an adaptive
weight, enables SAM to achieve sustained high performance through strategic
stability and robustness in volatile conditions.
  Extensive experiments against 14 diverse baselines demonstrate SAM's
superiority. It achieves top-tier throughput while being uniquely resilient to
complex workload shifts and cache pollution attacks. Furthermore, its decision
latency is highly scalable, remaining nearly constant as the system grows to
120 databases. Crucially, SAM achieves superior decision stability --
maintaining consistent optimization directions despite noise, avoiding
performance oscillations while ensuring predictable Quality of Service. These
results prove that a principled, stability-aware design is essential for
sustained high performance in real-world, large-scale systems.

### 3. [Scalability, Availability, Reproducibility and Extensibility in Islamic Database Systems](http://arxiv.org/pdf/2507.22384v1)

Authors: Umar Siddiqui, Habiba Youssef, Adel Sabour, Mohamed Ali

With the widespread of software systems and applications that serve the
Islamic knowledge domain, several concerns arise. Authenticity and accuracy of
the databases that back up these systems are questionable. With the excitement
that some software developers and amateur researchers may have, false
statements and incorrect claims may be made around numerical signs or miracles
in the Quran. Reproducibility of these claims may not be addressed by the
people making such claims. Moreover, with the increase in the number of users,
scalability and availability of these systems become a concern. In addition to
all these concerns, extensibility is also another major issue. Properly
designed systems can be extensible, reusable and built on top of one another,
instead of each system being built from scratch every time a new framework is
developed. In this paper, we introduce the QuranResearch.Org system and its
vision for scalability, availability, reproducibility and extensibility to
serve Islamic database systems.

### 4. [Systematic Evaluation of Knowledge Graph Repair with Large Language Models](http://arxiv.org/pdf/2507.22419v1)

Authors: Tung-Wei Lin, Gabe Fierro, Han Li, Tianzhen Hong, Pierluigi Nuzzo, Alberto Sangiovanni-Vinentelli

We present a systematic approach for evaluating the quality of knowledge
graph repairs with respect to constraint violations defined in shapes
constraint language (SHACL). Current evaluation methods rely on \emph{ad hoc}
datasets, which limits the rigorous analysis of repair systems in more general
settings. Our method addresses this gap by systematically generating violations
using a novel mechanism, termed violation-inducing operations (VIOs). We use
the proposed evaluation framework to assess a range of repair systems which we
build using large language models. We analyze the performance of these systems
across different prompting strategies. Results indicate that concise prompts
containing both the relevant violated SHACL constraints and key contextual
information from the knowledge graph yield the best performance.

### Distributed, Parallel, and Cluster Computing

### 1. [Towards Experiment Execution in Support of Community Benchmark Workflows for HPC](http://arxiv.org/pdf/2507.22294v1)

Authors: Gregor von Laszewski, Wesley Brewer, Sean R. Wilkinson, Andrew Shao, J. P. Fleischer, Harshad Pitkar, Christine R. Kirkpatrick, Geoffrey C. Fox

A key hurdle is demonstrating compute resource capability with limited
benchmarks. We propose workflow templates as a solution, offering adaptable
designs for specific scientific applications. Our paper identifies common usage
patterns for these templates, drawn from decades of HPC experience, including
recent work with the MLCommons Science working group.
  We found that focusing on simple experiment management tools within the
broader computational workflow improves adaptability, especially in education.
This concept, which we term benchmark carpentry, is validated by two
independent tools: Cloudmesh's Experiment Executor and Hewlett Packard
Enterprise's SmartSim. Both frameworks, with significant functional overlap,
have been tested across various scientific applications, including conduction
cloudmask, earthquake prediction, simulation-AI/ML interactions, and the
development of computational fluid dynamics surrogates.

### 2. [Leveraging Caliper and Benchpark to Analyze MPI Communication Patterns: Insights from AMG2023, Kripke, and Laghos](http://arxiv.org/pdf/2507.22372v1)

Authors: Grace Nansamba, Evelyn Namugwanya, David Boehme, Dewi Yokelson, Riley Shipley, Derek Schafer, Michael McKinsey, Olga Pearce, Anthony Skjellum

We introduce ``communication regions'' into the widely used Caliper HPC
profiling tool. A communication region is an annotation enabling capture of
metrics about the data being communicated (including statistics of these
metrics), and metrics about the MPI processes involved in the communications,
something not previously possible in Caliper. We explore the utility of
communication regions with three representative modeling and simulation
applications, AMG2023, Kripke, and Laghos, all part of the comprehensive
Benchpark suite that includes Caliper annotations. Enhanced Caliper reveals
detailed communication behaviors. Using Caliper and Thicket in tandem, we
create new visualizations of MPI communication patterns, including halo
exchanges. Our findings reveal communication bottlenecks and detailed
behaviors, indicating significant utility of the special-regions addition to
Caliper. The comparative scaling behavior of both CPU and GPU oriented systems
are shown; we are able to look at different regions within a given application,
and see how scalability and message-traffic metrics differ.

### 3. [DSPE: Profit Maximization in Edge-Cloud Storage System using Dynamic Space Partitioning with Erasure Code](http://arxiv.org/pdf/2507.22801v1)

Authors: Shubhradeep Roy, Suvarthi Sarkar, Vivek Verma, Aryabartta Sahu

Edge Storage Systems have emerged as a critical enabler of low latency data
access in modern cloud networks by bringing storage and computation closer to
end users. However, the limited storage capacity of edge servers poses
significant challenges in handling high volume and latency sensitive data
access requests, particularly under dynamic workloads. In this work, we propose
a profit driven framework that integrates three key mechanisms which are
collaborative caching, erasure coding, and elastic storage partitioning. Unlike
traditional replication, erasure coding enables space efficient redundancy,
allowing data to be reconstructed from any subset of K out of K plus M coded
blocks. We dynamically partition each edge server s storage into private and
public regions. The private region is further subdivided among access points
based on their incoming request rates, enabling adaptive control over data
locality and ownership. We design a data placement and replacement policy that
determines how and where to store or evict coded data blocks to maximize data
access within deadlines. While the private region serves requests from local
APs, the public region handles cooperative storage requests from neighboring
servers. Our proposed Dynamic Space Partitioning and Elastic caching strategy
is evaluated on both synthetic and real world traces from Netflix and Spotify.
Experimental results show that our method improves overall system profitability
by approximately 5 to 8% compared to state of the art approaches under varied
workload conditions.

### 4. [Hypernetworks for Model-Heterogeneous Personalized Federated Learning](http://arxiv.org/pdf/2507.22330v1)

Authors: Chen Zhang, Husheng Li, Xiang Liu, Linshan Jiang, Danxin Wang

Recent advances in personalized federated learning have focused on addressing
client model heterogeneity. However, most existing methods still require
external data, rely on model decoupling, or adopt partial learning strategies,
which can limit their practicality and scalability. In this paper, we revisit
hypernetwork-based methods and leverage their strong generalization
capabilities to design a simple yet effective framework for heterogeneous
personalized federated learning. Specifically, we propose MH-pFedHN, which
leverages a server-side hypernetwork that takes client-specific embedding
vectors as input and outputs personalized parameters tailored to each client's
heterogeneous model. To promote knowledge sharing and reduce computation, we
introduce a multi-head structure within the hypernetwork, allowing clients with
similar model sizes to share heads. Furthermore, we further propose
MH-pFedHNGD, which integrates an optional lightweight global model to improve
generalization. Our framework does not rely on external datasets and does not
require disclosure of client model architectures, thereby offering enhanced
privacy and flexibility. Extensive experiments on multiple benchmarks and model
settings demonstrate that our approach achieves competitive accuracy, strong
generalization, and serves as a robust baseline for future research in
model-heterogeneous personalized federated learning.

### 5. [A Semi-Supervised Federated Learning Framework with Hierarchical Clustering Aggregation for Heterogeneous Satellite Networks](http://arxiv.org/pdf/2507.22339v1)

Authors: Zhuocheng Liu, Zhishu Shen, Qiushi Zheng, Tiehua Zhang, Zheng Lei, Jiong Jin

Low Earth Orbit (LEO) satellites are emerging as key components of 6G
networks, with many already deployed to support large-scale Earth observation
and sensing related tasks. Federated Learning (FL) presents a promising
paradigm for enabling distributed intelligence in these resource-constrained
and dynamic environments. However, achieving reliable convergence, while
minimizing both processing time and energy consumption, remains a substantial
challenge, particularly in heterogeneous and partially unlabeled satellite
networks. To address this challenge, we propose a novel semi-supervised
federated learning framework tailored for LEO satellite networks with
hierarchical clustering aggregation. To further reduce communication overhead,
we integrate sparsification and adaptive weight quantization techniques. In
addition, we divide the FL clustering into two stages: satellite cluster
aggregation stage and Ground Stations (GSs) aggregation stage. The supervised
learning at GSs guides selected Parameter Server (PS) satellites, which in turn
support fully unlabeled satellites during the federated training process.
Extensive experiments conducted on a satellite network testbed demonstrate that
our proposal can significantly reduce processing time (up to 3x) and energy
consumption (up to 4x) compared to other comparative methods while maintaining
model accuracy.

### Digital Libraries

### 1. [Knowledge engineering for open science: Building and deploying knowledge bases for metadata standards](http://arxiv.org/pdf/2507.22391v1)

Authors: Mark A. Musen, Martin J. O'Connor, Josef Hardi, Marcos Martinez-Romero

Scientists strive to make their datasets available in open repositories, with
the goal that they be findable, accessible, interoperable, and reusable (FAIR).
Although it is hard for most investigators to remember all the guiding
principles associated with FAIR data, there is one overarching requirement: The
data need to be annotated with rich, discipline-specific, standardized
metadata. The Center for Expanded Data Annotation and Retrieval (CEDAR) builds
technology that enables scientists to encode metadata standards as templates
that enumerate the attributes of different kinds of experiments. These metadata
templates capture preferences regarding how data should be described and what a
third party needs to know to make sense of the datasets. CEDAR templates
describing community metadata preferences have been used to standardize
metadata for a variety of scientific consortia. They have been used as the
basis for data-annotation systems that acquire metadata through Web forms or
through spreadsheets, and they can help correct metadata to ensure adherence to
standards. Like the declarative knowledge bases that underpinned intelligent
systems decades ago, CEDAR templates capture the knowledge in symbolic form,
and they allow that knowledge to be applied in a variety of settings. They
provide a mechanism for scientific communities to create shared metadata
standards and to encode their preferences for the application of those
standards, and for deploying those standards in a range of intelligent systems
to promote open science.

### 2. [Presenting a classifier to detect research contributions in OpenAlex](http://arxiv.org/pdf/2507.22479v1)

Authors: Nick Haupka

This paper introduces a document type classifier with the purpose to optimise
the distinction between research and non-research journal publications in
OpenAlex. Based on open metadata, the classifier can detect non-research or
editorial content within a set of classified articles and reviews (e.g.
paratexts, abstracts, editorials, letters). The classifier achieves an F1-score
of 0,95, indicating a potential improvement in the data quality of bibliometric
research in OpenAlex when applying the classifier on real data. In total,
4.589.967 out of 42.701.863 articles and reviews could be reclassified as
non-research contributions by the classifier, representing a share of 10,75%

### 3. [Tracking research software outputs in the UK](http://arxiv.org/pdf/2507.22871v1)

Authors: Domhnall Carlin, Austen Rainer

Research software is crucial in the research process and the growth of Open
Science underscores the importance of accessing research artifacts, like data
and code, raising traceability challenges among outputs. While it is a clear
principle that research code, along with other essential outputs, should be
recognised as artifacts of the research process, the how of this principle
remains variable. This study examines where UK academic institutions store and
register software as a unique research output, searching the UKRI's Gateway to
Research (GtR) metadata for publicly funded research software in the UK. The
quantity of software reported as research outcomes remains low in proportion to
other categories. Artifact sharing appears low, with one-quarter of the
reported software having no links and 45% having either a missing or erroneous
URL. Of the valid URLs, we find the single largest category is Public
Commercial Code Repository, with GitHub being the host of 18% of all publicly
funded research software listed. These observations are contrasted with past
findings from 2023 and finally, we discuss the lack of artifact sharing in UK
research, with resulting implications for the maintenance and evolution of
research software. Without dissemination, research software risks demotion to a
transient artifact, useful only to meet short term research demands but
ultimately lost to the broader enterprise of science.

### Discrete Mathematics

### 1. [Structure of $k$-Matching-Planar Graphs](http://arxiv.org/pdf/2507.22395v1)

Authors: Kevin Hendrey, Nikolai Karol, David R. Wood

We introduce the class of $k$-matching-planar graphs, which is a significant
generalisation of many existing beyond planar graph classes, including
$k$-planar graphs. For $k \geqslant 0$, a simple topological graph $G$ (that
is, a graph drawn in the plane such that every pair of edges intersect at most
once, including endpoints) is $k$-matching-planar if for every edge $e \in
E(G)$, every matching amongst the edges of $G$ that cross $e$ has size at most
$k$. We prove that every simple topological $k$-matching-planar graph is
isomorphic to a subgraph of the strong product of a graph with bounded
treewidth and a path. This result qualitatively extends the planar graph
product structure theorem of Dujmovi\'c, Joret, Micek, Morin, Ueckerdt, and
Wood [J. ACM 2020] and recent product structure theorems for other beyond
planar graph classes. Using this result, we deduce that the class of simple
topological $k$-matching-planar graphs has several attractive properties,
making it the broadest class of simple beyond planar graphs in the literature
that has these properties. All of our results about simple topological
$k$-matching-planar graphs generalise to the non-simple setting, where the
maximum number of pairwise crossing edges incident to a common vertex becomes
relevant.
  The paper introduces several tools and results of independent interest. We
show that every simple topological $k$-matching-planar graph admits an
edge-colouring with $\mathcal{O}(k^{3}\log k)$ colours such that monochromatic
edges do not cross. As a key ingredient of the proof of our main product
structure theorem, we introduce the concept of weak shallow minors, which
subsume and generalise shallow minors, a key concept in graph sparsity theory.
We also establish upper bounds on the treewidth of graphs with well-behaved
circular drawings that qualitatively generalise several existing results.

### 2. [A quasi-optimal upper bound for induced paths in sparse graphs](http://arxiv.org/pdf/2507.22509v1)

Authors: Basile Couëtoux, Oscar Defrain, Jean-Florent Raymond

In 2012, Ne\v{s}et\v{r}il and Ossona de Mendez proved that graphs of bounded
degeneracy that have a path of order $n$ also have an induced path of order
$\Omega(\log \log n)$. In this paper we give an almost matching upper bound by
describing, for arbitrarily large values of $n$, a 2-degenerate graph that has
a path of order $n$ and where all induced paths have order $O(\log \log n \cdot
\log \log \log n)$.

### Data Structures and Algorithms

### 1. [Settling Weighted Token Swapping up to Algorithmic Barriers](http://arxiv.org/pdf/2507.22450v1)

Authors: Nicole Wein, Guanyu, Zhang

We study the weighted token swapping problem, in which we are given a graph
on $n$ vertices, $n$ weighted tokens, an initial assignment of one token to
each vertex, and a final assignment of one token to each vertex. The goal is to
find a minimum-cost sequence of swaps of adjacent tokens to reach the final
assignment from the initial assignment, where the cost is the sum over all
swaps of the sum of the weights of the two swapped tokens. Unweighted token
swapping has been extensively studied: it is NP-hard to approximate to a factor
better than $14/13$, and there is a polynomial-time 4-approximation, along with
a tight "barrier" result showing that the class of locally optimal algorithms
cannot achieve a ratio better than 4. For trees, the problem remains NP-hard to
solve exactly, and there is a polynomial-time 2-approximation, along with a
tight barrier result showing that the class of $\ell$-straying algorithms
cannot achieve a ratio better than 2. Weighted token swapping with $\{0,1\}$
weights is much harder to approximation: it is NP-hard to approximate even to a
factor of $(1-\varepsilon) \cdot \ln n$ for any constant $\varepsilon>0$.
Restricting to positive weights, no approximation algorithms are known, and the
only known lower bounds are those inherited directly from the unweighted
version. We provide the first approximation algorithms for weighted token
swapping on both trees and general graphs, along with tight barrier results.
Letting $w$ and $W$ be the minimum and maximum token weights, our approximation
ratio is $2+2W/w$ for general graphs and $1+W/w$ for trees.

### 2. [Deterministic Longest Common Subsequence Approximation in Near-Linear Time](http://arxiv.org/pdf/2507.22486v1)

Authors: Itai Boneh, Shay Golan, Matan Kraus

We provide a deterministic algorithm that outputs an $O(n^{3/4} \log
n)$-approximation for the Longest Common Subsequence (LCS) of two input
sequences of length $n$ in near-linear time. This is the first deterministic
approximation algorithm for LCS that achieves a sub-linear approximation ratio
in near-linear time.

### 3. [BlockFIFO & MultiFIFO: Scalable Relaxed Queues](http://arxiv.org/pdf/2507.22764v1)

Authors: Stefan Koch, Peter Sanders, Marvin Williams

FIFO queues are a fundamental data structure used in a wide range of
applications. Concurrent FIFO queues allow multiple execution threads to access
the queue simultaneously. Maintaining strict FIFO semantics in concurrent
queues leads to low throughput due to high contention at the head and tail of
the queue. By relaxing the FIFO semantics to allow some reordering of elements,
it becomes possible to achieve much higher scalability. This work presents two
orthogonal designs for relaxed concurrent FIFO queues, one derived from the
MultiQueue and the other based on ring buffers. We evaluate both designs
extensively on various micro-benchmarks and a breadth-first search application
on large graphs. Both designs outperform state-of-the-art relaxed and strict
FIFO queues, achieving higher throughput and better scalability.

### Emerging Technologies

### 1. [Toward Intelligent Electronic-Photonic Design Automation for Large-Scale Photonic Integrated Circuits: from Device Inverse Design to Physical Layout Generation](http://arxiv.org/pdf/2507.22301v1)

Authors: Hongjian Zhou, Pingchuan Ma, Jiaqi Gu

Photonic Integrated Circuits (PICs) offer tremendous advantages in bandwidth,
parallelism, and energy efficiency, making them essential for emerging
applications in artificial intelligence (AI), high-performance computing (HPC),
sensing, and communications. However, the design of modern PICs, which now
integrate hundreds to thousands of components, remains largely manual,
resulting in inefficiency, poor scalability, and susceptibility to errors. To
address these challenges, we propose PoLaRIS, a comprehensive Intelligent
Electronic-Photonic Design Automation (EPDA) framework that spans both
device-level synthesis and system-level physical layout. PoLaRIS combines a
robust, fabrication-aware inverse design engine with a routing-informed
placement and curvy-aware detailed router, enabling the automated generation of
design rule violation (DRV)-free and performance-optimized layouts. By unifying
physics-driven optimization with machine learning and domain-specific
algorithms, PoLaRIS significantly accelerates PIC development, lowers design
barriers, and lays the groundwork for scalable photonic system design
automation.

### 2. [Green Wave as an Integral Part for the Optimization of Traffic Efficiency and Safety: A Survey](http://arxiv.org/pdf/2507.22511v1)

Authors: Kranthi Kumar Talluri, Christopher Stang, Galia Weidl

Green Wave provides practical and advanced solutions to improve traffic
efficiency and safety through network coordination. Nevertheless, the complete
potential of Green Wave systems has yet to be explored. Utilizing emerging
technologies and advanced algorithms, such as AI or V2X, would aid in achieving
more robust traffic management strategies, especially when integrated with
Green Wave. This work comprehensively surveys existing traffic control
strategies that enable Green Waves and analyzes their impact on future traffic
management systems and urban infrastructure. Understanding previous research on
traffic management and its effect on traffic efficiency and safety helps
explore the integration of Green Wave solutions with smart city initiatives for
effective traffic signal coordination. This paper also discusses the advantages
of using Green Wave strategies for emission reduction and considers road safety
issues for vulnerable road users, such as pedestrians and cyclists. Finally,
the existing challenges and research gaps in building robust and successful
Green Wave systems are discussed to articulate explicitly the future
requirement of sustainable urban transport.

### 3. [Thermodynamics-Inspired Computing with Oscillatory Neural Networks for Inverse Matrix Computation](http://arxiv.org/pdf/2507.22544v1)

Authors: George Tsormpatzoglou, Filip Sabo, Aida Todri-Sanial

We describe a thermodynamic-inspired computing paradigm based on oscillatory
neural networks (ONNs). While ONNs have been widely studied as Ising machines
for tackling complex combinatorial optimization problems, this work
investigates their feasibility in solving linear algebra problems, specifically
the inverse matrix. Grounded in thermodynamic principles, we analytically
demonstrate that the linear approximation of the coupled Kuramoto oscillator
model leads to the inverse matrix solution. Numerical simulations validate the
theoretical framework, and we examine the parameter regimes that computation
has the highest accuracy.

### 4. [Hamiltonian Expressibility for Ansatz Selection in Variational Quantum Algorithms](http://arxiv.org/pdf/2507.22550v1)

Authors: Filippo Brozzi, Gloria Turati, Maurizio Ferrari Dacrema, Filippo Caruso, Paolo Cremonesi

In the context of Variational Quantum Algorithms (VQAs), selecting an
appropriate ansatz is crucial for efficient problem-solving. Hamiltonian
expressibility has been introduced as a metric to quantify a circuit's ability
to uniformly explore the energy landscape associated with a Hamiltonian ground
state search problem. However, its influence on solution quality remains
largely unexplored. In this work, we estimate the Hamiltonian expressibility of
a well-defined set of circuits applied to various Hamiltonians using a Monte
Carlo-based approach. We analyze how ansatz depth influences expressibility and
identify the most and least expressive circuits across different problem types.
We then train each ansatz using the Variational Quantum Eigensolver (VQE) and
analyze the correlation between solution quality and expressibility.Our results
indicate that, under ideal or low-noise conditions and particularly for
small-scale problems, ans\"atze with high Hamiltonian expressibility yield
better performance for problems with non-diagonal Hamiltonians and
superposition-state solutions. Conversely, circuits with low expressibility are
more effective for problems whose solutions are basis states, including those
defined by diagonal Hamiltonians. Under noisy conditions, low-expressibility
circuits remain preferable for basis-state problems, while intermediate
expressibility yields better results for some problems involving
superposition-state solutions.

### 5. [VRISE: A Virtual Reality Platfrom for Immersive and Interactive Surveying Education](http://arxiv.org/pdf/2507.22810v1)

Authors: Daniel Udekwe, Dimitrios Bolkas, Eren Erman Ozguven, Ren Moses, Qianwen, Guo

Surveying is a core component of civil engineering education, requiring
students to engage in hands-on spatial measurement, instrumentation handling,
and field-based decision-making. However, traditional instruction often poses
logistical and cognitive challenges that can hinder accessibility and student
engagement. While virtual laboratories have gained traction in engineering
education, few are purposefully designed to support flexible, adaptive learning
in surveying. To address this gap, we developed Virtual Reality for Immersive
and Interactive Surveying Education (VRISE), an immersive virtual reality
laboratory that replicates ground-based and aerial surveying tasks through
customizable, accessible, and user-friendly modules. VRISE features interactive
experiences such as differential leveling with a digital level equipment and
waypoint-based drone navigation, enhanced by input smoothing, adaptive
interfaces, and real-time feedback to accommodate diverse learning styles.
Evaluation across multiple user sessions demonstrated consistent gains in
measurement accuracy, task efficiency, and interaction quality, with a clear
progression in skill development across the ground-based and aerial surveying
modalities. By reducing cognitive load and physical demands, even in tasks
requiring fine motor control and spatial reasoning, VRISE demonstrates the
potential of immersive, repeatable digital environments to enhance surveying
education, broaden participation, and strengthen core competencies in a safe
and engaging setting.

### Formal Languages and Automata Theory

### 1. [On the growth of hypergeometric sequences](http://arxiv.org/pdf/2507.22437v1)

Authors: George Kenison, Jakub Konieczny, Florian Luca, Andrew Scoones, Mahsa Shirmohammadi, James Worrell

Hypergeometric sequences obey first-order linear recurrence relations with
polynomial coefficients and are commonplace throughout the mathematical and
computational sciences. For certain classes of hypergeometric sequences, we
prove linear growth estimates on their Weil heights. We give an application of
our effective results towards the Membership Problem from Computer Science.
Recall that Membership asks to procedurally determine whether a specified
target is an element of a given recurrence sequence.

### Computer Science and Game Theory

### 1. [Reducing the complexity of computing the values of a Nash equilibrium](http://arxiv.org/pdf/2507.22819v1)

Authors: Debtoru Chatterjee, Girish Tiwari, Niladri Chatterjee

The Colonel Blotto game, formulated by Emile Borel, involves players
allocating limited resources to multiple battlefields simultaneously, with the
winner being the one who allocates more resources to each battlefield.
Computation of the Nash equilibrium, including of two person, zero sum, mixed
strategy Colonel Blotto games have encountered issues of scalability and
complexity owing to their PPAD completeness. This paper proposes an algorithm
that computes the same value as the Nash equilibrium but cannot be
characterized by the Fixed point Theorems of Tarski, Kakutani and Brouwer. The
reduced complexity of the proposed algorithm is based on dispensing with the
need for computing both players Nash strategies in Colonel Blotto games. The
same algorithm can, therefore, be extended to all two person, zero sum games to
compute the value of the Nash equilibrium. The theoretical superiority of the
proposed algorithm over both LP solvers and another method that computes the
same value of the game as its Nash equilibrium by a random assignment of
probabilities to the active strategy set of the defending player, is also
proposed.

### Human-Computer Interaction

### 1. [ConGaIT: A Clinician-Centered Dashboard for Contestable AI in Parkinson's Disease Care](http://arxiv.org/pdf/2507.22300v1)

Authors: Phuc Truong Loc Nguyen, Thanh Hung Do

AI-assisted gait analysis holds promise for improving Parkinson's Disease
(PD) care, but current clinical dashboards lack transparency and offer no
meaningful way for clinicians to interrogate or contest AI decisions. We
present Con-GaIT (Contestable Gait Interpretation & Tracking), a
clinician-centered system that advances Contestable AI through a tightly
integrated interface designed for interpretability, oversight, and procedural
recourse. Grounded in HCI principles, ConGaIT enables structured disagreement
via a novel Contest & Justify interaction pattern, supported by visual
explanations, role-based feedback, and traceable justification logs. Evaluated
using the Contestability Assessment Score (CAS), the framework achieves a score
of 0.970, demonstrating that contestability can be operationalized through
human-centered design in compliance with emerging regulatory standards. A
demonstration of the framework is available at
https://github.com/hungdothanh/Con-GaIT.

### 2. [A Node on the Constellation: The Role of Feminist Makerspaces in Building and Sustaining Alternative Cultures of Technology Production](http://arxiv.org/pdf/2507.22329v1)

Authors: Erin Gatz, Yasmine Kotturi, Andrea Afua Kwamya, Sarah Fox

Feminist makerspaces offer community led alternatives to dominant tech
cultures by centering care, mutual aid, and collective knowledge production.
While prior CSCW research has explored their inclusive practices, less is known
about how these spaces sustain themselves over time. Drawing on interviews with
18 founders and members across 8 U.S. feminist makerspaces as well as
autoethnographic reflection, we examine the organizational and relational
practices that support long-term endurance. We find that sustainability is not
achieved through growth or institutionalization, but through care-driven
stewardship, solidarity with local justice movements, and shared governance.
These social practices position feminist makerspaces as prefigurative
counterspaces - sites that enact, rather than defer, feminist values in
everyday practice. This paper offers empirical insight into how feminist
makerspaces persist amid structural precarity, and highlights the forms of
labor and coalition-building that underpin alternative sociotechnical
infrastructures.

### 3. [Mitigating Response Delays in Free-Form Conversations with LLM-powered Intelligent Virtual Agents](http://arxiv.org/pdf/2507.22352v1)

Authors: Mykola Maslych, Mohammadreza Katebi, Christopher Lee, Yahya Hmaiti, Amirpouya Ghasemaghaei, Christian Pumarada, Janneese Palmer, Esteban Segarra Martinez, Marco Emporio, Warren Snipes, Ryan P. McMahan, Joseph J. LaViola Jr

We investigated the challenges of mitigating response delays in free-form
conversations with virtual agents powered by Large Language Models (LLMs)
within Virtual Reality (VR). For this, we used conversational fillers, such as
gestures and verbal cues, to bridge delays between user input and system
responses and evaluate their effectiveness across various latency levels and
interaction scenarios. We found that latency above 4 seconds degrades quality
of experience, while natural conversational fillers improve perceived response
time, especially in high-delay conditions. Our findings provide insights for
practitioners and researchers to optimize user engagement whenever
conversational systems' responses are delayed by network limitations or slow
hardware. We also contribute an open-source pipeline that streamlines deploying
conversational agents in virtual environments.

### 4. [A Fuzzy Set-based Approach for Matching Hand-Drawing Shapes of Touch-based Gestures for Graphical Passwords](http://arxiv.org/pdf/2507.22382v1)

Authors: Adel Sabour, Ahmed Gadallah, Hesham Hefny

This paper presents a two-dimension fuzzy set based approach for matching
touch-based gestures using fuzzy cued click point technique. The pro posed
approach aims mainly to improve the acceptance of the most closed inac curate
hand drawn gestures generated by the user compared with a predefined referenced
gesture value that is stored in the user profile. Commonly, gestures are used
in order to facilitate the interactive capabilities between humans and
computerized systems. Unfortunately, most of current gesturing techniques don't
deal at the same level of inaccuracy of gesturing, resulted from the nature of
hu man fingers and hands movements. This paper aims, in a more flexible manner,
to tackle the inaccuracy problem existed with gesture-based interactions
between humans and a computerized system.

### 5. [Analysis of User Experience Evaluation Methods for Deaf users: A Case Study on a mobile App](http://arxiv.org/pdf/2507.22455v1)

Authors: A. E. Fuentes-Cortázar, A. Rivera-Hernández, J. R. Rojano-Cáceres

User Experience (UX) evaluation methods that are commonly used with hearing
users may not be functional or effective for Deaf users. This is because these
methods are primarily designed for users with hearing abilities, which can
create limitations in the interaction, perception, and understanding of the
methods for Deaf individuals. Furthermore, traditional UX evaluation approaches
often fail to address the unique accessibility needs of Deaf users, resulting
in an incomplete or biased assessment of their user experience. This research
focused on analyzing a set of UX evaluation methods recommended for use with
Deaf users, with the aim of validating the accessibility of each method through
findings and limitations. The results indicate that, although these evaluation
methods presented here are commonly recommended in the literature for use with
Deaf users, they present various limitations that must be addressed in order to
better adapt to the communication skills specific to the Deaf community. This
research concludes that evaluation methods must be adapted to ensure accessible
software evaluation for Deaf individuals, enabling the collection of data that
accurately reflects their experiences and needs.

### 6. [Progressive Web Application for Storytelling Therapy Support](http://arxiv.org/pdf/2507.22839v1)

Authors: Javier Jimenez-Honrado, Javier Gomez Garcia, Felipe Costa-Tebar, Felix A. Marco, Jose A. Gallud, Gabriel Sebastian Rivera

In spite of all advances promoted by information technologies, there are
still activities where this technology is not applied for reasons such as being
carried out in non-profit organizations or because they have not adapted to
this modernization. Until recently, the way to work with mobile devices was
either by connecting through a web page with the device's browser, or by
downloading an application from the corresponding platform. But lately,
technologies are being developed that aim to break with this, as in the case of
Progressive Web Applications (PWA). One of the advantages offered by PWA is to
access the web page and install it as an application on the device. The purpose
of this article is to design a progressive Web application for the support of
Storytelling Therapy, one of the novel therapies applied in the field of mental
health. In addition to providing a software application to enhance Storytelling
Therapy workshops, it is also intended to analyze and verify the advantages of
PWA in a real case.

### 7. [Magentic-UI: Towards Human-in-the-loop Agentic Systems](http://arxiv.org/pdf/2507.22358v1)

Authors: Hussein Mozannar, Gagan Bansal, Cheng Tan, Adam Fourney, Victor Dibia, Jingya Chen, Jack Gerrits, Tyler Payne, Matheus Kunzler Maldaner, Madeleine Grunde-McLaughlin, Eric Zhu, Griffin Bassman, Jacob Alber, Peter Chang, Ricky Loynd, Friederike Niedtner, Ece Kamar, Maya Murad, Rafah Hosn, Saleema Amershi

AI agents powered by large language models are increasingly capable of
autonomously completing complex, multi-step tasks using external tools. Yet,
they still fall short of human-level performance in most domains including
computer use, software development, and research. Their growing autonomy and
ability to interact with the outside world, also introduces safety and security
risks including potentially misaligned actions and adversarial manipulation. We
argue that human-in-the-loop agentic systems offer a promising path forward,
combining human oversight and control with AI efficiency to unlock productivity
from imperfect systems. We introduce Magentic-UI, an open-source web interface
for developing and studying human-agent interaction. Built on a flexible
multi-agent architecture, Magentic-UI supports web browsing, code execution,
and file manipulation, and can be extended with diverse tools via Model Context
Protocol (MCP). Moreover, Magentic-UI presents six interaction mechanisms for
enabling effective, low-cost human involvement: co-planning, co-tasking,
multi-tasking, action guards, and long-term memory. We evaluate Magentic-UI
across four dimensions: autonomous task completion on agentic benchmarks,
simulated user testing of its interaction capabilities, qualitative studies
with real users, and targeted safety assessments. Our findings highlight
Magentic-UI's potential to advance safe and efficient human-agent
collaboration.

### 8. [Beyond Accuracy: How AI Metacognitive Sensitivity improves AI-assisted Decision Making](http://arxiv.org/pdf/2507.22365v1)

Authors: ZhaoBin Li, Mark Steyvers

In settings where human decision-making relies on AI input, both the
predictive accuracy of the AI system and the reliability of its confidence
estimates influence decision quality. We highlight the role of AI metacognitive
sensitivity -- its ability to assign confidence scores that accurately
distinguish correct from incorrect predictions -- and introduce a theoretical
framework for assessing the joint impact of AI's predictive accuracy and
metacognitive sensitivity in hybrid decision-making settings. Our analysis
identifies conditions under which an AI with lower predictive accuracy but
higher metacognitive sensitivity can enhance the overall accuracy of human
decision making. Finally, a behavioral experiment confirms that greater AI
metacognitive sensitivity improves human decision performance. Together, these
findings underscore the importance of evaluating AI assistance not only by
accuracy but also by metacognitive sensitivity, and of optimizing both to
achieve superior decision outcomes.

### 9. [Exploring Student-AI Interactions in Vibe Coding](http://arxiv.org/pdf/2507.22614v1)

Authors: Francis Geng, Anshul Shah, Haolin Li, Nawab Mulla, Steven Swanson, Gerald Soosai Raj, Daniel Zingaro, Leo Porter

Background and Context. Chat-based and inline-coding-based GenAI has already
had substantial impact on the CS Education community. The recent introduction
of ``vibe coding'' may further transform how students program, as it introduces
a new way for students to create software projects with minimal oversight.
  Objectives. The purpose of this study is to understand how students in
introductory programming and advanced software engineering classes interact
with a vibe coding platform (Replit) when creating software and how the
interactions differ by programming background.
  Methods. Interview participants were asked to think-aloud while building a
web application using Replit. Thematic analysis was then used to analyze the
video recordings with an emphasis on the interactions between the student and
Replit.
  Findings. For both groups, the majority of student interactions with Replit
were to test or debug the prototype and only rarely did students visit code.
Prompts by advanced software engineering students were much more likely to
include relevant app feature and codebase contexts than those by introductory
programming students.

### 10. [Cluster-Based Random Forest Visualization and Interpretation](http://arxiv.org/pdf/2507.22665v1)

Authors: Max Sondag, Christofer Meinecke, Dennis Collaris, Tatiana von Landesberger, Stef van den Elzen

Random forests are a machine learning method used to automatically classify
datasets and consist of a multitude of decision trees. While these random
forests often have higher performance and generalize better than a single
decision tree, they are also harder to interpret. This paper presents a
visualization method and system to increase interpretability of random forests.
We cluster similar trees which enables users to interpret how the model
performs in general without needing to analyze each individual decision tree in
detail, or interpret an oversimplified summary of the full forest. To
meaningfully cluster the decision trees, we introduce a new distance metric
that takes into account both the decision rules as well as the predictions of a
pair of decision trees. We also propose two new visualization methods that
visualize both clustered and individual decision trees: (1) The Feature Plot,
which visualizes the topological position of features in the decision trees,
and (2) the Rule Plot, which visualizes the decision rules of the decision
trees. We demonstrate the efficacy of our approach through a case study on the
"Glass" dataset, which is a relatively complex standard machine learning
dataset, as well as a small user study.

### Information Retrieval

### 1. [Sustainability Evaluation Metrics for Recommender Systems](http://arxiv.org/pdf/2507.22520v1)

Authors: Alexander Felfernig, Damian Garber, Viet-Man Le, Sebastian Lubos, Thi Ngoc Trang Tran

Sustainability-oriented evaluation metrics can help to assess the quality of
recommender systems beyond wide-spread metrics such as accuracy, precision,
recall, and satisfaction. Following the United Nations`s sustainable
development goals (SDGs), such metrics can help to analyse the impact of
recommender systems on environmental, social, and economic aspects. We discuss
different basic sustainability evaluation metrics for recommender systems and
analyze their applications.

### 2. [AUV-Fusion: Cross-Modal Adversarial Fusion of User Interactions and Visual Perturbations Against VARS](http://arxiv.org/pdf/2507.22880v1)

Authors: Hai Ling, Tianchi Wang, Xiaohao Liu, Zhulin Tao, Lifang Yang, Xianglin Huang

Modern Visual-Aware Recommender Systems (VARS) exploit the integration of
user interaction data and visual features to deliver personalized
recommendations with high precision. However, their robustness against
adversarial attacks remains largely underexplored, posing significant risks to
system reliability and security. Existing attack strategies suffer from notable
limitations: shilling attacks are costly and detectable, and visual-only
perturbations often fail to align with user preferences. To address these
challenges, we propose AUV-Fusion, a cross-modal adversarial attack framework
that adopts high-order user preference modeling and cross-modal adversary
generation. Specifically, we obtain robust user embeddings through multi-hop
user-item interactions and transform them via an MLP into semantically aligned
perturbations. These perturbations are injected onto the latent space of a
pre-trained VAE within the diffusion model. By synergistically integrating
genuine user interaction data with visually plausible perturbations, AUV-Fusion
eliminates the need for injecting fake user profiles and effectively mitigates
the challenge of insufficient user preference extraction inherent in
traditional visual-only attacks. Comprehensive evaluations on diverse VARS
architectures and real-world datasets demonstrate that AUV-Fusion significantly
enhances the exposure of target (cold-start) items compared to conventional
baseline methods. Moreover, AUV-Fusion maintains exceptional stealth under
rigorous scrutiny.

### 3. [A Comprehensive Taxonomy of Negation for NLP and Neural Retrievers](http://arxiv.org/pdf/2507.22337v1)

Authors: Roxana Petcu, Samarth Bhargav, Maarten de Rijke, Evangelos Kanoulas

Understanding and solving complex reasoning tasks is vital for addressing the
information needs of a user. Although dense neural models learn contextualised
embeddings, they still underperform on queries containing negation. To
understand this phenomenon, we study negation in both traditional neural
information retrieval and LLM-based models. We (1) introduce a taxonomy of
negation that derives from philosophical, linguistic, and logical definitions;
(2) generate two benchmark datasets that can be used to evaluate the
performance of neural information retrieval models and to fine-tune models for
a more robust performance on negation; and (3) propose a logic-based
classification mechanism that can be used to analyze the performance of
retrieval models on existing datasets. Our taxonomy produces a balanced data
distribution over negation types, providing a better training setup that leads
to faster convergence on the NevIR dataset. Moreover, we propose a
classification schema that reveals the coverage of negation types in existing
datasets, offering insights into the factors that might affect the
generalization of fine-tuned models on negation.

### 4. [RecGPT Technical Report](http://arxiv.org/pdf/2507.22879v1)

Authors: Chao Yi, Dian Chen, Gaoyang Guo, Jiakai Tang, Jian Wu, Jing Yu, Sunhao Dai, Wen Chen, Wenjun Yang, Yuning Jiang, Zhujin Gao, Bo Zheng, Chi Li, Dimin Wang, Dixuan Wang, Fan Li, Fan Zhang, Haibin Chen, Haozhuang Liu, Jialin Zhu, Jiamang Wang, Jiawei Wu, Jin Cui, Ju Huang, Kai Zhang, Kan Liu, Lang Tian, Liang Rao, Longbin Li, Lulu Zhao, Mao Zhang, Na He, Peiyang Wang, Qiqi Huang, Tao Luo, Wenbo Su, Xiaoxiao He, Xin Tong, Xu Chen, Xunke Xi, Yang Li, Yaxuan Wu, Yeqiu Yang, Yi Hu, Yinnan Song, Yuchen Li, Yujie Luo, Yujin Yuan, Yuliang Yan, Zhengyang Wang, Zhibo Xiao, Zhixin Ma, Zile Zhou

Recommender systems are among the most impactful applications of artificial
intelligence, serving as critical infrastructure connecting users, merchants,
and platforms. However, most current industrial systems remain heavily reliant
on historical co-occurrence patterns and log-fitting objectives, i.e.,
optimizing for past user interactions without explicitly modeling user intent.
This log-fitting approach often leads to overfitting to narrow historical
preferences, failing to capture users' evolving and latent interests. As a
result, it reinforces filter bubbles and long-tail phenomena, ultimately
harming user experience and threatening the sustainability of the whole
recommendation ecosystem.
  To address these challenges, we rethink the overall design paradigm of
recommender systems and propose RecGPT, a next-generation framework that places
user intent at the center of the recommendation pipeline. By integrating large
language models (LLMs) into key stages of user interest mining, item retrieval,
and explanation generation, RecGPT transforms log-fitting recommendation into
an intent-centric process. To effectively align general-purpose LLMs to the
above domain-specific recommendation tasks at scale, RecGPT incorporates a
multi-stage training paradigm, which integrates reasoning-enhanced
pre-alignment and self-training evolution, guided by a Human-LLM cooperative
judge system. Currently, RecGPT has been fully deployed on the Taobao App.
Online experiments demonstrate that RecGPT achieves consistent performance
gains across stakeholders: users benefit from increased content diversity and
satisfaction, merchants and the platform gain greater exposure and conversions.
These comprehensive improvement results across all stakeholders validates that
LLM-driven, intent-centric design can foster a more sustainable and mutually
beneficial recommendation ecosystem.

### 5. [GeoOutageKG: A Multimodal Geospatiotemporal Knowledge Graph for Multiresolution Power Outage Analysis](http://arxiv.org/pdf/2507.22878v1)

Authors: Ethan Frakes, Yinghui Wu, Roger H. French, Mengjie Li

Detecting, analyzing, and predicting power outages is crucial for grid risk
assessment and disaster mitigation. Numerous outages occur each year,
exacerbated by extreme weather events such as hurricanes. Existing outage data
are typically reported at the county level, limiting their spatial resolution
and making it difficult to capture localized patterns. However, it offers
excellent temporal granularity. In contrast, nighttime light satellite image
data provides significantly higher spatial resolution and enables a more
comprehensive spatial depiction of outages, enhancing the accuracy of assessing
the geographic extent and severity of power loss after disaster events.
However, these satellite data are only available on a daily basis. Integrating
spatiotemporal visual and time-series data sources into a unified knowledge
representation can substantially improve power outage detection, analysis, and
predictive reasoning. In this paper, we propose GeoOutageKG, a multimodal
knowledge graph that integrates diverse data sources, including nighttime light
satellite image data, high-resolution spatiotemporal power outage maps, and
county-level timeseries outage reports in the U.S. We describe our method for
constructing GeoOutageKG by aligning source data with a developed ontology,
GeoOutageOnto. Currently, GeoOutageKG includes over 10.6 million individual
outage records spanning from 2014 to 2024, 300,000 NTL images spanning from
2012 to 2024, and 15,000 outage maps. GeoOutageKG is a novel, modular and
reusable semantic resource that enables robust multimodal data integration. We
demonstrate its use through multiresolution analysis of geospatiotemporal power
outages.

### Machine Learning

### 1. [Comparing Cluster-Based Cross-Validation Strategies for Machine Learning Model Evaluation](http://arxiv.org/pdf/2507.22299v1)

Authors: Afonso Martini Spezia, Mariana Recamonde-Mendoza

Cross-validation plays a fundamental role in Machine Learning, enabling
robust evaluation of model performance and preventing overestimation on
training and validation data. However, one of its drawbacks is the potential to
create data subsets (folds) that do not adequately represent the diversity of
the original dataset, which can lead to biased performance estimates. The
objective of this work is to deepen the investigation of cluster-based
cross-validation strategies by analyzing the performance of different
clustering algorithms through experimental comparison. Additionally, a new
cross-validation technique that combines Mini Batch K-Means with class
stratification is proposed. Experiments were conducted on 20 datasets (both
balanced and imbalanced) using four supervised learning algorithms, comparing
cross-validation strategies in terms of bias, variance, and computational cost.
The technique that uses Mini Batch K-Means with class stratification
outperformed others in terms of bias and variance on balanced datasets, though
it did not significantly reduce computational cost. On imbalanced datasets,
traditional stratified cross-validation consistently performed better, showing
lower bias, variance, and computational cost, making it a safe choice for
performance evaluation in scenarios with class imbalance. In the comparison of
different clustering algorithms, no single algorithm consistently stood out as
superior. Overall, this work contributes to improving predictive model
evaluation strategies by providing a deeper understanding of the potential of
cluster-based data splitting techniques and reaffirming the effectiveness of
well-established strategies like stratified cross-validation. Moreover, it
highlights perspectives for increasing the robustness and reliability of model
evaluations, especially in datasets with clustering characteristics.

### 2. [CS-SHRED: Enhancing SHRED for Robust Recovery of Spatiotemporal Dynamics](http://arxiv.org/pdf/2507.22303v1)

Authors: Romulo B. da Silva, Cássio M. Oishi, Diego Passos, J. Nathan Kutz

We present $\textbf{CS-SHRED}$, a novel deep learning architecture that
integrates Compressed Sensing (CS) into a Shallow Recurrent Decoder
($\textbf{SHRED}$) to reconstruct spatiotemporal dynamics from incomplete,
compressed, or corrupted data. Our approach introduces two key innovations.
First, by incorporating CS techniques into the $\textbf{SHRED}$ architecture,
our method leverages a batch-based forward framework with $\ell_1$
regularization to robustly recover signals even in scenarios with sparse sensor
placements, noisy measurements, and incomplete sensor acquisitions. Second, an
adaptive loss function dynamically combines Mean Squared Error (MSE) and Mean
Absolute Error (MAE) terms with a piecewise Signal-to-Noise Ratio (SNR)
regularization, which suppresses noise and outliers in low-SNR regions while
preserving fine-scale features in high-SNR regions.
  We validate $\textbf{CS-SHRED}$ on challenging problems including
viscoelastic fluid flows, maximum specific humidity fields, sea surface
temperature distributions, and rotating turbulent flows. Compared to the
traditional $\textbf{SHRED}$ approach, $\textbf{CS-SHRED}$ achieves
significantly higher reconstruction fidelity - as demonstrated by improved SSIM
and PSNR values, lower normalized errors, and enhanced LPIPS scores-thereby
providing superior preservation of small-scale structures and increased
robustness against noise and outliers.
  Our results underscore the advantages of the jointly trained CS and SHRED
design architecture which includes an LSTM sequence model for characterizing
the temporal evolution with a shallow decoder network (SDN) for modeling the
high-dimensional state space. The SNR-guided adaptive loss function for the
spatiotemporal data recovery establishes $\textbf{CS-SHRED}$ as a promising
tool for a wide range of applications in environmental, climatic, and
scientific data analyses.

### 3. [Parametrized Multi-Agent Routing via Deep Attention Models](http://arxiv.org/pdf/2507.22338v1)

Authors: Salar Basiri, Dhananjay Tiwari, Srinivasa M. Salapaka

We propose a scalable deep learning framework for parametrized sequential
decision-making (ParaSDM), where multiple agents jointly optimize discrete
action policies and shared continuous parameters. A key subclass of this
setting arises in Facility-Location and Path Optimization (FLPO), where
multi-agent systems must simultaneously determine optimal routes and facility
locations, aiming to minimize the cumulative transportation cost within the
network. FLPO problems are NP-hard due to their mixed discrete-continuous
structure and highly non-convex objective. To address this, we integrate the
Maximum Entropy Principle (MEP) with a neural policy model called the Shortest
Path Network (SPN)-a permutation-invariant encoder-decoder that approximates
the MEP solution while enabling efficient gradient-based optimization over
shared parameters. The SPN achieves up to 100$\times$ speedup in policy
inference and gradient computation compared to MEP baselines, with an average
optimality gap of approximately 6% across a wide range of problem sizes. Our
FLPO approach yields over 10$\times$ lower cost than metaheuristic baselines
while running significantly faster, and matches Gurobi's optimal cost with
annealing at a 1500$\times$ speedup-establishing a new state of the art for
ParaSDM problems. These results highlight the power of structured deep models
for solving large-scale mixed-integer optimization tasks.

### 4. [MSQ: Memory-Efficient Bit Sparsification Quantization](http://arxiv.org/pdf/2507.22349v1)

Authors: Seokho Han, Seoyeon Yoon, Jinhee Kim, Dongwei Wang, Kang Eun Jeon, Huanrui Yang, Jong Hwan Ko

As deep neural networks (DNNs) see increased deployment on mobile and edge
devices, optimizing model efficiency has become crucial. Mixed-precision
quantization is widely favored, as it offers a superior balance between
efficiency and accuracy compared to uniform quantization. However, finding the
optimal precision for each layer is challenging. Recent studies utilizing
bit-level sparsity have shown promise, yet they often introduce substantial
training complexity and high GPU memory requirements. In this paper, we propose
Memory-Efficient Bit Sparsification Quantization (MSQ), a novel approach that
addresses these limitations. MSQ applies a round-clamp quantizer to enable
differentiable computation of the least significant bits (LSBs) from model
weights. It further employs regularization to induce sparsity in these LSBs,
enabling effective precision reduction without explicit bit-level parameter
splitting. Additionally, MSQ incorporates Hessian information, allowing the
simultaneous pruning of multiple LSBs to further enhance training efficiency.
Experimental results show that MSQ achieves up to 8.00x reduction in trainable
parameters and up to 86% reduction in training time compared to previous
bit-level quantization, while maintaining competitive accuracy and compression
rates. This makes it a practical solution for training efficient DNNs on
resource-constrained devices.

### 5. [Multimodal Late Fusion Model for Problem-Solving Strategy Classification in a Machine Learning Game](http://arxiv.org/pdf/2507.22426v1)

Authors: Clemens Witt, Thiemo Leonhardt, Nadine Bergner, Mareen Grillenberger

Machine learning models are widely used to support stealth assessment in
digital learning environments. Existing approaches typically rely on abstracted
gameplay log data, which may overlook subtle behavioral cues linked to
learners' cognitive strategies. This paper proposes a multimodal late fusion
model that integrates screencast-based visual data and structured in-game
action sequences to classify students' problem-solving strategies. In a pilot
study with secondary school students (N=149) playing a multitouch educational
game, the fusion model outperformed unimodal baseline models, increasing
classification accuracy by over 15%. Results highlight the potential of
multimodal ML for strategy-sensitive assessment and adaptive support in
interactive learning contexts.

### 6. [RANA: Robust Active Learning for Noisy Network Alignment](http://arxiv.org/pdf/2507.22434v1)

Authors: Yixuan Nan, Xixun Lin, Yanmin Shang, Zhuofan Li, Can Zhao, Yanan Cao

Network alignment has attracted widespread attention in various fields.
However, most existing works mainly focus on the problem of label sparsity,
while overlooking the issue of noise in network alignment, which can
substantially undermine model performance. Such noise mainly includes
structural noise from noisy edges and labeling noise caused by human-induced
and process-driven errors. To address these problems, we propose RANA, a Robust
Active learning framework for noisy Network Alignment. RANA effectively tackles
both structure noise and label noise while addressing the sparsity of anchor
link annotations, which can improve the robustness of network alignment models.
Specifically, RANA introduces the proposed Noise-aware Selection Module and the
Label Denoising Module to address structural noise and labeling noise,
respectively. In the first module, we design a noise-aware maximization
objective to select node pairs, incorporating a cleanliness score to address
structural noise. In the second module, we propose a novel multi-source fusion
denoising strategy that leverages model and twin node pairs labeling to provide
more accurate labels for node pairs. Empirical results on three real-world
datasets demonstrate that RANA outperforms state-of-the-art active
learning-based methods in alignment accuracy. Our code is available at
https://github.com/YXNan0110/RANA.

### 7. [SmilesT5: Domain-specific pretraining for molecular language models](http://arxiv.org/pdf/2507.22514v1)

Authors: Philip Spence, Brooks Paige, Anne Osbourn

Molecular property prediction is an increasingly critical task within drug
discovery and development. Typically, neural networks can learn molecular
properties using graph-based, language-based or feature-based methods. Recent
advances in natural language processing have highlighted the capabilities of
neural networks to learn complex human language using masked language
modelling. These approaches to training large transformer-based deep learning
models have also been used to learn the language of molecules, as represented
by simplified molecular-input line-entry system (SMILES) strings. Here, we
present novel domain-specific text-to-text pretraining tasks that yield
improved performance in six classification-based molecular property prediction
benchmarks, relative to both traditional likelihood-based training and
previously proposed fine-tuning tasks. Through ablation studies, we show that
data and computational efficiency can be improved by using these
domain-specific pretraining tasks. Finally, the pretrained embeddings from the
model can be used as fixed inputs into a downstream machine learning classifier
and yield comparable performance to finetuning but with much lower
computational overhead.

### 8. [HGCN(O): A Self-Tuning GCN HyperModel Toolkit for Outcome Prediction in Event-Sequence Data](http://arxiv.org/pdf/2507.22524v1)

Authors: Fang Wang, Paolo Ceravolo, Ernesto Damiani

We propose HGCN(O), a self-tuning toolkit using Graph Convolutional Network
(GCN) models for event sequence prediction. Featuring four GCN architectures
(O-GCN, T-GCN, TP-GCN, TE-GCN) across the GCNConv and GraphConv layers, our
toolkit integrates multiple graph representations of event sequences with
different choices of node- and graph-level attributes and in temporal
dependencies via edge weights, optimising prediction accuracy and stability for
balanced and unbalanced datasets. Extensive experiments show that GCNConv
models excel on unbalanced data, while all models perform consistently on
balanced data. Experiments also confirm the superior performance of HGCN(O)
over traditional approaches. Applications include Predictive Business Process
Monitoring (PBPM), which predicts future events or states of a business process
based on event logs.

### 9. [DeepC4: Deep Conditional Census-Constrained Clustering for Large-scale Multitask Spatial Disaggregation of Urban Morphology](http://arxiv.org/pdf/2507.22554v1)

Authors: Joshua Dimasaka, Christian Geiß, Emily So

To understand our global progress for sustainable development and disaster
risk reduction in many developing economies, two recent major initiatives - the
Uniform African Exposure Dataset of the Global Earthquake Model (GEM)
Foundation and the Modelling Exposure through Earth Observation Routines
(METEOR) Project - implemented classical spatial disaggregation techniques to
generate large-scale mapping of urban morphology using the information from
various satellite imagery and its derivatives, geospatial datasets of the built
environment, and subnational census statistics. However, the local discrepancy
with well-validated census statistics and the propagated model uncertainties
remain a challenge in such coarse-to-fine-grained mapping problems,
specifically constrained by weak and conditional label supervision. Therefore,
we present Deep Conditional Census-Constrained Clustering (DeepC4), a novel
deep learning-based spatial disaggregation approach that incorporates local
census statistics as cluster-level constraints while considering multiple
conditional label relationships in a joint multitask learning of the patterns
of satellite imagery. To demonstrate, compared to GEM and METEOR, we enhanced
the quality of Rwandan maps of urban morphology, specifically building exposure
and physical vulnerability, at the third-level administrative unit from the
2022 census. As the world approaches the conclusion of our global frameworks in
2030, our work has offered a new deep learning-based mapping technique towards
a spatial auditing of our existing coarse-grained derived information at large
scales.

### 10. [VAR: Visual Analysis for Rashomon Set of Machine Learning Models' Performance](http://arxiv.org/pdf/2507.22556v1)

Authors: Yuanzhe Jin

Evaluating the performance of closely matched machine learning(ML) models
under specific conditions has long been a focus of researchers in the field of
machine learning. The Rashomon set is a collection of closely matched ML
models, encompassing a wide range of models with similar accuracies but
different structures. Traditionally, the analysis of these sets has focused on
vertical structural analysis, which involves comparing the corresponding
features at various levels within the ML models. However, there has been a lack
of effective visualization methods for horizontally comparing multiple models
with specific features. We propose the VAR visualization solution. VAR uses
visualization to perform comparisons of ML models within the Rashomon set. This
solution combines heatmaps and scatter plots to facilitate the comparison. With
the help of VAR, ML model developers can identify the optimal model under
specific conditions and better understand the Rashomon set's overall
characteristics.

### Neural and Evolutionary Computing

### 1. [Nearest-Better Network for Visualizing and Analyzing Combinatorial Optimization Problems: A Unified Tool](http://arxiv.org/pdf/2507.22440v1)

Authors: Yiya Diao, Changhe Li, Sanyou Zeng, Xinye Cai, Wenjian Luo, Shengxiang Yang, Carlos A. Coello Coello

The Nearest-Better Network (NBN) is a powerful method to visualize sampled
data for continuous optimization problems while preserving multiple landscape
features. However, the calculation of NBN is very time-consuming, and the
extension of the method to combinatorial optimization problems is challenging
but very important for analyzing the algorithm's behavior. This paper provides
a straightforward theoretical derivation showing that the NBN network
essentially functions as the maximum probability transition network for
algorithms. This paper also presents an efficient NBN computation method with
logarithmic linear time complexity to address the time-consuming issue. By
applying this efficient NBN algorithm to the OneMax problem and the Traveling
Salesman Problem (TSP), we have made several remarkable discoveries for the
first time: The fitness landscape of OneMax exhibits neutrality, ruggedness,
and modality features. The primary challenges of TSP problems are ruggedness,
modality, and deception. Two state-of-the-art TSP algorithms (i.e., EAX and
LKH) have limitations when addressing challenges related to modality and
deception, respectively. LKH, based on local search operators, fails when there
are deceptive solutions near global optima. EAX, which is based on a single
population, can efficiently maintain diversity. However, when multiple
attraction basins exist, EAX retains individuals within multiple basins
simultaneously, reducing inter-basin interaction efficiency and leading to
algorithm's stagnation.

### 2. [Tapping into the Black Box: Uncovering Aligned Representations in Pretrained Neural Networks](http://arxiv.org/pdf/2507.22832v1)

Authors: Maciej Satkiewicz

In this paper we argue that ReLU networks learn an implicit linear model we
can actually tap into. We describe that alleged model formally and show that we
can approximately pull its decision boundary back to the input space with
certain simple modification to the backward pass. The resulting gradients
(called excitation pullbacks) reveal high-resolution input- and target-specific
features of remarkable perceptual alignment on a number of popular
ImageNet-pretrained deep architectures. This strongly suggests that neural
networks do, in fact, rely on learned interpretable patterns that can be
recovered after training. Thus, our findings may have profound implications for
knowledge discovery and the development of dependable artificial systems.

### 3. [Amorphous Solid Model of Vectorial Hopfield Neural Networks](http://arxiv.org/pdf/2507.22787v1)

Authors: F. Gallavotti, A. Zaccone

We present a vectorial extension of the Hopfield associative memory model
inspired by the theory of amorphous solids, where binary neural states are
replaced by unit vectors $\mathbf{s}_i \in \mathbb{R}^3$ on the sphere $S^2$.
The generalized Hebbian learning rule creates a block-structured weight matrix
through outer products of stored pattern vectors, analogous to the Hessian
matrix structure in amorphous solids. We demonstrate that this model exhibits
quantifiable structural properties characteristic of disordered materials:
energy landscapes with deep minima for stored patterns versus random
configurations (energy gaps $\sim 7$ units), strongly anisotropic correlations
encoded in the weight matrix (anisotropy ratios $\sim 10^2$), and
order-disorder transitions controlled by the pattern density $\gamma = P/(N
\cdot d)$. The enhanced memory capacity ($\gamma_c \approx 0.55$ for a
fully-connected network) compared to binary networks ($\gamma_c \approx 0.138$)
and the emergence of orientational correlations establish connections between
associative memory mechanisms and amorphous solid physics, particularly in
systems with continuous orientational degrees of freedom. We also unveil the
scaling with the coordination number $Z$ of the memory capacity: $\gamma_c \sim
(Z-6)$ from the isostatic point $Z_c =6$ of the 3D elastic network, which
closely mirrors the scaling of the shear modulus $G \sim (Z-6)$ in 3D
central-force spring networks.

### Networking and Internet Architecture

### 1. [AdapSCA-PSO: An Adaptive Localization Algorithm with AI-Based Hybrid SCA-PSO for IoT WSNs](http://arxiv.org/pdf/2507.22317v1)

Authors: Ze Zhang, Qian Dong, Wenhan Wang

The accurate localization of sensor nodes is a fundamental requirement for
the practical application of the Internet of Things (IoT). To enable robust
localization across diverse environments, this paper proposes a hybrid
meta-heuristic localization algorithm. Specifically, the algorithm integrates
the Sine Cosine Algorithm (SCA), which is effective in global search, with
Particle Swarm Optimization (PSO), which excels at local search. An adaptive
switching module is introduced to dynamically select between the two
algorithms. Furthermore, the initialization, fitness evaluation, and parameter
settings of the algorithm have been specifically redesigned and optimized to
address the characteristics of the node localization problem. Simulation
results across varying numbers of sensor nodes demonstrate that, compared to
standalone PSO and the unoptimized SCAPSO algorithm, the proposed method
significantly reduces the number of required iterations and achieves an average
localization error reduction of 84.97%.

### 2. [802.11bf Multiband Passive Sensing: Reusing Wi-Fi Signaling for Sensing](http://arxiv.org/pdf/2507.22591v1)

Authors: Pablo Picazo-Martinez, Carlos Barroso-Fernández, Alejandro Calvillo-Fernandez, Milan Groshev, Carlos J. Bernardos, Antonio de la Oliva, Alain Mourad

This paper presents a novel multiband passive sensing system that leverages
IEEE 802.11bf Wi-Fi signals for environmental sensing, focusing on both sub-7
GHz and millimeter-wave (mmWave) bands. By combining Channel State Information
(CSI) from multiple bands, the system enhances accuracy and reliability in
detecting human presence, movement, and activities in indoor environments.
Utilizing a novel model, called MILAGRO, the system demonstrates robust
performance across different scenarios, including monitoring human presence in
workspaces and tracking movement in corridors. Experimental results show high
accuracy (95-100%), with improved performance by integrating multiband data.
The system also addresses key security concerns associated with passive
sensing, proposing measures to mitigate potential risks. This work advances the
use of Wi-Fi for passive sensing by reducing reliance on active sensing
infrastructure and extending the capabilities of low-cost, non-intrusive
environmental monitoring.

### 3. [OFCnetLLM: Large Language Model for Network Monitoring and Alertness](http://arxiv.org/pdf/2507.22711v1)

Authors: Hong-Jun Yoon, Mariam Kiran, Danial Ebling, Joe Breen

The rapid evolution of network infrastructure is bringing new challenges and
opportunities for efficient network management, optimization, and security.
With very large monitoring databases becoming expensive to explore, the use of
AI and Generative AI can help reduce costs of managing these datasets. This
paper explores the use of Large Language Models (LLMs) to revolutionize network
monitoring management by addressing the limitations of query finding and
pattern analysis. We leverage LLMs to enhance anomaly detection, automate
root-cause analysis, and automate incident analysis to build a well-monitored
network management team using AI. Through a real-world example of developing
our own OFCNetLLM, based on the open-source LLM model, we demonstrate practical
applications of OFCnetLLM in the OFC conference network. Our model is developed
as a multi-agent approach and is still evolving, and we present early results
here.

### 4. [Morph: ChirpTransformer-based Encoder-decoder Co-design for Reliable LoRa Communication](http://arxiv.org/pdf/2507.22851v1)

Authors: Yidong Ren, Maolin Gan, Chenning Li, Shakhrul Iman Siam, Mi Zhang, Shigang Chen, Zhichao Cao

In this paper, we propose Morph, a LoRa encoder-decoder co-design to enhance
communication reliability while improving its computation efficiency in
extremely-low signal-to-noise ratio (SNR) situations. The standard LoRa encoder
controls 6 Spreading Factors (SFs) to tradeoff SNR tolerance with data rate.
SF-12 is the maximum SF providing the lowest SNR tolerance on commercial
off-the-shelf (COTS) LoRa nodes. In Morph, we develop an SF-configuration based
encoder to mimic the larger SFs beyond SF-12 while it is compatible with COTS
LoRa nodes. Specifically, we manipulate four SF configurations of a Morph
symbol to encode 2-bit data. Accordingly, we recognize the used SF
configuration of the symbol for data decoding. We leverage a Deep Neural
Network (DNN) decoder to fully capture multi-dimensional features among diverse
SF configurations to maximize the SNR gain. Moreover, we customize the input
size, neural network structure, and training method of the DNN decoder to
improve its efficiency, reliability, and generalizability. We implement Morph
with COTS LoRa nodes and a USRP N210, then evaluate its performance on indoor
and campus-scale testbeds. Results show that we can reliably decode data at
-28.8~dB SNR, which is 6.4~dB lower than the standard LoRa with SF-12 chirps.
In addition, the computation efficiency of our DNN decoder is about 3x higher
than state-of-the-art.

### 5. [Bifröst: Spatial Networking with Bigraphs](http://arxiv.org/pdf/2507.22687v1)

Authors: Josh Millar, Ryan Gibb, Roy Ang, Anil Madhavapeddy, Hamed Haddadi

Modern networked environments increasingly rely on spatial reasoning, but
lack a coherent representation for coordinating physical space. Consequently,
tasks such as enforcing spatial access policies remain fragile and manual. We
first propose a unifying representation based on bigraphs, capturing spatial,
social, and communication relationships within a single formalism, with
user-facing tools to generate bigraphs from physical environments. Second, we
present a hierarchical agent architecture for distributed spatial reasoning,
with runtimes for agentic processes to interact the spatial representation, and
a context-aware execution model that scopes reasoning to the smallest viable
subspace. Together, these enable private, reliable, and low-latency spatial
networking that can safely interact with agentic workflows.

### Robotics

### 1. [FLORES: A Reconfigured Wheel-Legged Robot for Enhanced Steering and Adaptability](http://arxiv.org/pdf/2507.22345v1)

Authors: Zhicheng Song, Jinglan Xu, Chunxin Zheng, Yulin Li, Zhihai Bi, Jun Ma

Wheel-legged robots integrate the agility of legs for navigating rough
terrains while harnessing the efficiency of wheels for smooth surfaces.
However, most existing designs do not fully capitalize on the benefits of both
legged and wheeled structures, which limits overall system flexibility and
efficiency. We present FLORES (reconfigured wheel-legged robot for enhanced
steering and adaptability), a novel wheel-legged robot design featuring a
distinctive front-leg configuration that sets it beyond standard design
approaches. Specifically, FLORES replaces the conventional hip-roll degree of
freedom (DoF) of the front leg with hip-yaw DoFs, and this allows for efficient
movement on flat surfaces while ensuring adaptability when navigating complex
terrains. This innovative design facilitates seamless transitions between
different locomotion modes (i.e., legged locomotion and wheeled locomotion) and
optimizes the performance across varied environments. To fully exploit FLORES's
mechanical capabilities, we develop a tailored reinforcement learning (RL)
controller that adapts the Hybrid Internal Model (HIM) with a customized reward
structure optimized for our unique mechanical configuration. This framework
enables the generation of adaptive, multi-modal locomotion strategies that
facilitate smooth transitions between wheeled and legged movements.
Furthermore, our distinctive joint design enables the robot to exhibit novel
and highly efficient locomotion gaits that capitalize on the synergistic
advantages of both locomotion modes. Through comprehensive experiments, we
demonstrate FLORES's enhanced steering capabilities, improved navigation
efficiency, and versatile locomotion across various terrains. The open-source
project can be found at
https://github.com/ZhichengSong6/FLORES-A-Reconfigured-Wheel-Legged-Robot-for-Enhanced-Steering-and-Adaptability.git.

### 2. [In-Situ Soil-Property Estimation and Bayesian Mapping with a Simulated Compact Track Loader](http://arxiv.org/pdf/2507.22356v1)

Authors: W. Jacob Wagner, Ahmet Soylemezoglu, Katherine Driggs-Campbell

Existing earthmoving autonomy is largely confined to highly controlled and
well-characterized environments due to the complexity of vehicle-terrain
interaction dynamics and the partial observability of the terrain resulting
from unknown and spatially varying soil conditions. In this chapter, a a
soil-property mapping system is proposed to extend the environmental state, in
order to overcome these restrictions and facilitate development of more robust
autonomous earthmoving. A GPU accelerated elevation mapping system is extended
to incorporate a blind mapping component which traces the movement of the blade
through the terrain to displace and erode intersected soil, enabling separately
tracking undisturbed and disturbed soil. Each interaction is approximated as a
flat blade moving through a locally homogeneous soil, enabling modeling of
cutting forces using the fundamental equation of earthmoving (FEE). Building
upon our prior work on in situ soil-property estimation, a method is devised to
extract approximate geometric parameters of the model given the uneven terrain,
and an improved physics infused neural network (PINN) model is developed to
predict soil properties and uncertainties of these estimates. A simulation of a
compact track loader (CTL) with a blade attachment is used to collect data to
train the PINN model. Post-training, the model is leveraged online by the
mapping system to track soil property estimates spatially as separate layers in
the map, with updates being performed in a Bayesian manner. Initial experiments
show that the system accurately highlights regions requiring higher relative
interaction forces, indicating the promise of this approach in enabling
soil-aware planning for autonomous terrain shaping.

### 3. [Operationalization of Scenario-Based Safety Assessment of Automated Driving Systems](http://arxiv.org/pdf/2507.22433v1)

Authors: Olaf Op den Camp, Erwin de Gelder

Before introducing an Automated Driving System (ADS) on the road at scale,
the manufacturer must conduct some sort of safety assurance. To structure and
harmonize the safety assurance process, the UNECE WP.29 Working Party on
Automated/Autonomous and Connected Vehicles (GRVA) is developing the New
Assessment/Test Method (NATM) that indicates what steps need to be taken for
safety assessment of an ADS. In this paper, we will show how to practically
conduct safety assessment making use of a scenario database, and what
additional steps must be taken to fully operationalize the NATM. In addition,
we will elaborate on how the use of scenario databases fits with methods
developed in the Horizon Europe projects that focus on safety assessment
following the NATM approach.

### 4. [A Two-Stage Lightweight Framework for Efficient Land-Air Bimodal Robot Autonomous Navigation](http://arxiv.org/pdf/2507.22473v1)

Authors: Yongjie Li, Zhou Liu, Wenshuai Yu, Zhangji Lu, Chenyang Wang, Fei Yu, Qingquan Li

Land-air bimodal robots (LABR) are gaining attention for autonomous
navigation, combining high mobility from aerial vehicles with long endurance
from ground vehicles. However, existing LABR navigation methods are limited by
suboptimal trajectories from mapping-based approaches and the excessive
computational demands of learning-based methods. To address this, we propose a
two-stage lightweight framework that integrates global key points prediction
with local trajectory refinement to generate efficient and reachable
trajectories. In the first stage, the Global Key points Prediction Network
(GKPN) was used to generate a hybrid land-air keypoint path. The GKPN includes
a Sobel Perception Network (SPN) for improved obstacle detection and a
Lightweight Attention Planning Network (LAPN) to improves predictive ability by
capturing contextual information. In the second stage, the global path is
segmented based on predicted key points and refined using a mapping-based
planner to create smooth, collision-free trajectories. Experiments conducted on
our LABR platform show that our framework reduces network parameters by 14\%
and energy consumption during land-air transitions by 35\% compared to existing
approaches. The framework achieves real-time navigation without GPU
acceleration and enables zero-shot transfer from simulation to reality during

### 5. [Explainable Deep Anomaly Detection with Sequential Hypothesis Testing for Robotic Sewer Inspection](http://arxiv.org/pdf/2507.22546v1)

Authors: Alex George, Will Shepherd, Simon Tait, Lyudmila Mihaylova, Sean R. Anderson

Sewer pipe faults, such as leaks and blockages, can lead to severe
consequences including groundwater contamination, property damage, and service
disruption. Traditional inspection methods rely heavily on the manual review of
CCTV footage collected by mobile robots, which is inefficient and susceptible
to human error. To automate this process, we propose a novel system
incorporating explainable deep learning anomaly detection combined with
sequential probability ratio testing (SPRT). The anomaly detector processes
single image frames, providing interpretable spatial localisation of anomalies,
whilst the SPRT introduces temporal evidence aggregation, enhancing robustness
against noise over sequences of image frames. Experimental results demonstrate
improved anomaly detection performance, highlighting the benefits of the
combined spatiotemporal analysis system for reliable and robust sewer
inspection.

### 6. [UniLegs: Universal Multi-Legged Robot Control through Morphology-Agnostic Policy Distillation](http://arxiv.org/pdf/2507.22653v1)

Authors: Weijie Xi, Zhanxiang Cao, Chenlin Ming, Jianying Zheng, Guyue Zhou

Developing controllers that generalize across diverse robot morphologies
remains a significant challenge in legged locomotion. Traditional approaches
either create specialized controllers for each morphology or compromise
performance for generality. This paper introduces a two-stage teacher-student
framework that bridges this gap through policy distillation. First, we train
specialized teacher policies optimized for individual morphologies, capturing
the unique optimal control strategies for each robot design. Then, we distill
this specialized expertise into a single Transformer-based student policy
capable of controlling robots with varying leg configurations. Our experiments
across five distinct legged morphologies demonstrate that our approach
preserves morphology-specific optimal behaviors, with the Transformer
architecture achieving 94.47\% of teacher performance on training morphologies
and 72.64\% on unseen robot designs. Comparative analysis reveals that
Transformer-based architectures consistently outperform MLP baselines by
leveraging attention mechanisms to effectively model joint relationships across
different kinematic structures. We validate our approach through successful
deployment on a physical quadruped robot, demonstrating the practical viability
of our morphology-agnostic control framework. This work presents a scalable
solution for developing universal legged robot controllers that maintain
near-optimal performance while generalizing across diverse morphologies.

### 7. [Improving Generalization Ability of Robotic Imitation Learning by Resolving Causal Confusion in Observations](http://arxiv.org/pdf/2507.22380v1)

Authors: Yifei Chen, Yuzhe Zhang, Giovanni D'urso, Nicholas Lawrance, Brendan Tidd

Recent developments in imitation learning have considerably advanced robotic
manipulation. However, current techniques in imitation learning can suffer from
poor generalization, limiting performance even under relatively minor domain
shifts. In this work, we aim to enhance the generalization capabilities of
complex imitation learning algorithms to handle unpredictable changes from the
training environments to deployment environments. To avoid confusion caused by
observations that are not relevant to the target task, we propose to explicitly
learn the causal relationship between observation components and expert
actions, employing a framework similar to [6], where a causal structural
function is learned by intervention on the imitation learning policy.
Disentangling the feature representation from image input as in [6] is hard to
satisfy in complex imitation learning process in robotic manipulation, we
theoretically clarify that this requirement is not necessary in causal
relationship learning. Therefore, we propose a simple causal structure learning
framework that can be easily embedded in recent imitation learning
architectures, such as the Action Chunking Transformer [31]. We demonstrate our
approach using a simulation of the ALOHA [31] bimanual robot arms in Mujoco,
and show that the method can considerably mitigate the generalization problem
of existing complex imitation learning algorithms.

### 8. [Comparing Normalizing Flows with Kernel Density Estimation in Estimating Risk of Automated Driving Systems](http://arxiv.org/pdf/2507.22429v1)

Authors: Erwin de Gelder, Maren Buermann, Olaf Op den Camp

The development of safety validation methods is essential for the safe
deployment and operation of Automated Driving Systems (ADSs). One of the goals
of safety validation is to prospectively evaluate the risk of an ADS dealing
with real-world traffic. Scenario-based assessment is a widely-used approach,
where test cases are derived from real-world driving data. To allow for a
quantitative analysis of the system performance, the exposure of the scenarios
must be accurately estimated. The exposure of scenarios at parameter level is
expressed using a Probability Density Function (PDF). However, assumptions
about the PDF, such as parameter independence, can introduce errors, while
avoiding assumptions often leads to oversimplified models with limited
parameters to mitigate the curse of dimensionality.
  This paper considers the use of Normalizing Flows (NF) for estimating the PDF
of the parameters. NF are a class of generative models that transform a simple
base distribution into a complex one using a sequence of invertible and
differentiable mappings, enabling flexible, high-dimensional density estimation
without restrictive assumptions on the PDF's shape. We demonstrate the
effectiveness of NF in quantifying risk and risk uncertainty of an ADS,
comparing its performance with Kernel Density Estimation (KDE), a traditional
method for non-parametric PDF estimation. While NF require more computational
resources compared to KDE, NF is less sensitive to the curse of dimensionality.
As a result, NF can improve risk uncertainty estimation, offering a more
precise assessment of an ADS's safety.
  This work illustrates the potential of NF in scenario-based safety. Future
work involves experimenting more with using NF for scenario generation and
optimizing the NF architecture, transformation types, and training
hyperparameters to further enhance their applicability.

### 9. [Viser: Imperative, Web-based 3D Visualization in Python](http://arxiv.org/pdf/2507.22885v1)

Authors: Brent Yi, Chung Min Kim, Justin Kerr, Gina Wu, Rebecca Feng, Anthony Zhang, Jonas Kulhanek, Hongsuk Choi, Yi Ma, Matthew Tancik, Angjoo Kanazawa

We present Viser, a 3D visualization library for computer vision and
robotics. Viser aims to bring easy and extensible 3D visualization to Python:
we provide a comprehensive set of 3D scene and 2D GUI primitives, which can be
used independently with minimal setup or composed to build specialized
interfaces. This technical report describes Viser's features, interface, and
implementation. Key design choices include an imperative-style API and a
web-based viewer, which improve compatibility with modern programming patterns
and workflows.

### 10. [Safety Evaluation of Motion Plans Using Trajectory Predictors as Forward Reachable Set Estimators](http://arxiv.org/pdf/2507.22389v1)

Authors: Kaustav Chakraborty, Zeyuan Feng, Sushant Veer, Apoorva Sharma, Wenhao Ding, Sever Topan, Boris Ivanovic, Marco Pavone, Somil Bansal

The advent of end-to-end autonomy stacks - often lacking interpretable
intermediate modules - has placed an increased burden on ensuring that the
final output, i.e., the motion plan, is safe in order to validate the safety of
the entire stack. This requires a safety monitor that is both complete (able to
detect all unsafe plans) and sound (does not flag safe plans). In this work, we
propose a principled safety monitor that leverages modern multi-modal
trajectory predictors to approximate forward reachable sets (FRS) of
surrounding agents. By formulating a convex program, we efficiently extract
these data-driven FRSs directly from the predicted state distributions,
conditioned on scene context such as lane topology and agent history. To ensure
completeness, we leverage conformal prediction to calibrate the FRS and
guarantee coverage of ground-truth trajectories with high probability. To
preserve soundness in out-of-distribution (OOD) scenarios or under predictor
failure, we introduce a Bayesian filter that dynamically adjusts the FRS
conservativeness based on the predictor's observed performance. We then assess
the safety of the ego vehicle's motion plan by checking for intersections with
these calibrated FRSs, ensuring the plan remains collision-free under plausible
future behaviors of others. Extensive experiments on the nuScenes dataset show
our approach significantly improves soundness while maintaining completeness,
offering a practical and reliable safety monitor for learned autonomy stacks.

### Software Engineering

### 1. [AutoCodeSherpa: Symbolic Explanations in AI Coding Agents](http://arxiv.org/pdf/2507.22414v1)

Authors: Sungmin Kang, Haifeng Ruan, Abhik Roychoudhury

Large Language Model (LLM) agents autonomously use external tools on top of
one or more LLMs to accomplish specific tasks. Lately LLM agents for software
engineering tasks have become popular. These agents can benefit from the use of
program analysis tools working on program representations. This is demonstrated
by existing agentic AI solutions such as AutoCodeRover or SpecRover which
perform automated program repair. Specifically the goal of these works is to
use program analysis to improve the patch quality. These agents are currently
being used to automatically fix static analysis issues from the widely used
SonarQube static analyzer.
  Nevertheless, for the agents to be deployed in a production environment,
agents need to suggest software artifacts, such as patches, with evidence and
with high confidence. In this work, we provide a workflow where an agent
provides explanations of the bug in the form of symbolic formulae. The
explanations are in the form of input conditions, infection conditions and
output conditions, implemented as property based tests (PBT) and
program-internal symbolic expressions. These can help in human developer
cognition of the agent outputs as well as in achieving completely automated
agentic workflows for software. The human developer can benefit from the input
condition, represented as a PBT, to generate various concrete inputs showing a
given issue. Furthermore, since the PBTs are executable, our explanations are
executable as well. We can thus also use the explanations in a completely
automated issue resolution environment for accepting or rejecting the patches
that are suggested by patching agents such as AutoCodeRover. Finally, as
agentic AI approaches continue to develop, the program analysis driven
explanations can be provided to other LLM-based repair techniques such as
Agentless to improve their output.

### 2. [Ensemble Fuzzing with Dynamic Resource Scheduling and Multidimensional Seed Evaluation](http://arxiv.org/pdf/2507.22442v1)

Authors: Yukai Zhao, Shaohua Wang, Jue Wang, Xing Hu, Xin Xia

Fuzzing is widely used for detecting bugs and vulnerabilities, with various
techniques proposed to enhance its effectiveness. To combine the advantages of
multiple technologies, researchers proposed ensemble fuzzing, which integrates
multiple base fuzzers. Despite promising results, state-of-the-art ensemble
fuzzing techniques face limitations in resource scheduling and performance
evaluation, leading to unnecessary resource waste. In this paper, we propose
Legion, a novel ensemble fuzzing framework that dynamically schedules resources
during the ensemble fuzzing campaign. We designed a novel resource scheduling
algorithm based on the upper confidence bound algorithm to reduce the resource
consumption of ineffective base fuzzers. Additionally, we introduce a
multidimensional seed evaluation strategy, which considers multiple metrics to
achieve more comprehensive fine-grained performance evaluation. We implemented
Legion as a prototype tool and evaluated its effectiveness on Google's
fuzzer-test-suite as well as real-world open-source projects. Results show that
Legion outperforms existing state-of-the-art base fuzzers and ensemble fuzzing
techniques, detecting 20 vulnerabilities in real-world open-source
projects-five previously unknown and three classified as CVEs.

### 3. [Inside madupite: Technical Design and Performance](http://arxiv.org/pdf/2507.22538v1)

Authors: Matilde Gargiani, Robin Sieber, Philip Pawlowsky, John Lygeros

In this work, we introduce and benchmark madupite, a newly proposed
high-performance solver designed for large-scale discounted infinite-horizon
Markov decision processes with finite state and action spaces. After a brief
overview of the class of mathematical optimization methods on which madupite
relies, we provide details on implementation choices, technical design and
deployment. We then demonstrate its scalability and efficiency by showcasing
its performance on the solution of Markov decision processes arising from
different application areas, including epidemiology and classical control.
Madupite sets a new standard as, to the best of our knowledge, it is the only
solver capable of efficiently computing exact solutions for large-scale Markov
decision processes, even when these exceed the memory capacity of modern
laptops and operate in near-undiscounted settings. This is possible as madupite
can work in a fully distributed manner and therefore leverage the memory
storage and computation capabilities of modern high-performance computing
clusters. This key feature enables the solver to efficiently handle problems of
medium to large size in an exact manner instead of necessarily resorting to
function approximations. Moreover, madupite is unique in allowing users to
customize the solution algorithm to better exploit the specific structure of
their problem, significantly accelerating convergence especially in
large-discount factor settings. Overall, madupite represents a significant
advancement, offering unmatched scalability and flexibility in solving
large-scale Markov decision processes.

### 4. [The Multi-Agent Fault Localization System Based on Monte Carlo Tree Search Approach](http://arxiv.org/pdf/2507.22800v1)

Authors: Rui Ren

In real-world scenarios, due to the highly decoupled and flexible nature of
microservices, it poses greater challenges to system reliability. The more
frequent occurrence of incidents has created a demand for Root Cause
Analysis(RCA) methods that enable rapid identification and recovery of
incidents. Large language model (LLM) provides a new path for quickly locating
and recovering from incidents by leveraging their powerful generalization
ability combined with expert experience. Current LLM for RCA frameworks are
based on ideas like ReAct and Chain-of-Thought, but the hallucination of LLM
and the propagation nature of anomalies often lead to incorrect localization
results. Moreover, the massive amount of anomalous information generated in
large, complex systems presents a huge challenge for the context window length
of LLMs. To address these challenges, we propose KnowledgeMind, an innovative
LLM multi-agent system based on Monte Carlo Tree Search and a knowledge base
reward mechanism for standardized service-by-service reasoning. Compared to
State-Of-The-Art(SOTA) LLM for RCA methods, our service-by-service exploration
approach significantly reduces the burden on the maximum context window length,
requiring only one-tenth of its size. Additionally, by incorporating a
rule-based real-time reward mechanism, our method effectively mitigates
hallucinations during the inference process. Compared to the SOTA LLM for RCA
framework, our method achieves a 49.29% to 128.35% improvement in root cause
localization accuracy.

### 5. [From Articles to Code: On-Demand Generation of Core Algorithms from Scientific Publications](http://arxiv.org/pdf/2507.22324v1)

Authors: Cameron S. Movassaghi, Amanda Momenzadeh, Jesse G. Meyer

Maintaining software packages imposes significant costs due to dependency
management, bug fixes, and versioning. We show that rich method descriptions in
scientific publications can serve as standalone specifications for modern large
language models (LLMs), enabling on-demand code generation that could supplant
human-maintained libraries. We benchmark state-of-the-art models
(GPT-o4-mini-high, Gemini Pro 2.5, Claude Sonnet 4) by tasking them with
implementing a diverse set of core algorithms drawn from original publications.
Our results demonstrate that current LLMs can reliably reproduce package
functionality with performance indistinguishable from conventional libraries.
These findings foreshadow a paradigm shift toward flexible, on-demand code
generation and away from static, human-maintained packages, which will result
in reduced maintenance overhead by leveraging published articles as sufficient
context for the automated implementation of analytical workflows.

### 6. [Scalability, Availability, Reproducibility and Extensibility in Islamic Database Systems](http://arxiv.org/pdf/2507.22384v1)

Authors: Umar Siddiqui, Habiba Youssef, Adel Sabour, Mohamed Ali

With the widespread of software systems and applications that serve the
Islamic knowledge domain, several concerns arise. Authenticity and accuracy of
the databases that back up these systems are questionable. With the excitement
that some software developers and amateur researchers may have, false
statements and incorrect claims may be made around numerical signs or miracles
in the Quran. Reproducibility of these claims may not be addressed by the
people making such claims. Moreover, with the increase in the number of users,
scalability and availability of these systems become a concern. In addition to
all these concerns, extensibility is also another major issue. Properly
designed systems can be extensible, reusable and built on top of one another,
instead of each system being built from scratch every time a new framework is
developed. In this paper, we introduce the QuranResearch.Org system and its
vision for scalability, availability, reproducibility and extensibility to
serve Islamic database systems.

### 7. [RePaCA: Leveraging Reasoning Large Language Models for Static Automated Patch Correctness Assessment](http://arxiv.org/pdf/2507.22580v1)

Authors: Marcos Fuster-Pena, David de-Fitero-Dominguez, Antonio Garcia-Cabot, Eva Garcia-Lopez

Automated Program Repair (APR) seeks to automatically correct software bugs
without requiring human intervention. However, existing tools tend to generate
patches that satisfy test cases without fixing the underlying bug, those are
known as overfitting patches. To address this issue, Automated Patch
Correctness Assessment (APCA) attempts to identify overfitting patches
generated by APR tools. It can be solved as a static approach, meaning that no
additional information is needed beyond the original and fixed code snippets.
Current static techniques often struggle with reliability, flexibility and
transparency. To address these issues, we introduce RePaCA, a novel static APCA
technique that leverages Large Language Models (LLMs) specialized in thinking
tasks. Our model is prompted with both buggy and fixed code snippets and guided
to generate a Chain of Thought that analyses code differences, reasons about
how the patch addresses the root cause, and ultimately provides a binary
classification: correct or overfitting. To enhance these reasoning capabilities
for the APCA task specifically, the LLM is finetuned using Reinforcement
Learning with the Group Relative Policy Optimization algorithm. When evaluated
on a standard Defects4J-derived test, our approach achieves state-of-the-art
performance, with 83.1% accuracy and an 84.8% F1-score. Furthermore, our model
demonstrates superior generalization capabilities when trained on different
datasets, outperforming the leading technique. This reasoning capability also
provides enhanced explainability for the patch assessment. These findings
underscore the considerable promise of finetuned, reasoning LLMs to advance
static APCA by enhancing accuracy, generalization, and explainability.

### 8. [Metamorphic Testing of Deep Code Models: A Systematic Literature Review](http://arxiv.org/pdf/2507.22610v1)

Authors: Ali Asgari, Milan de Koning, Pouria Derakhshanfar, Annibale Panichella

Large language models and deep learning models designed for code intelligence
have revolutionized the software engineering field due to their ability to
perform various code-related tasks. These models can process source code and
software artifacts with high accuracy in tasks such as code completion, defect
detection, and code summarization; therefore, they can potentially become an
integral part of modern software engineering practices. Despite these
capabilities, robustness remains a critical quality attribute for deep-code
models as they may produce different results under varied and adversarial
conditions (e.g., variable renaming). Metamorphic testing has become a widely
used approach to evaluate models' robustness by applying semantic-preserving
transformations to input programs and analyzing the stability of model outputs.
While prior research has explored testing deep learning models, this systematic
literature review focuses specifically on metamorphic testing for deep code
models. By studying 45 primary papers, we analyze the transformations,
techniques, and evaluation methods used to assess robustness. Our review
summarizes the current landscape, identifying frequently evaluated models,
programming tasks, datasets, target languages, and evaluation metrics, and
highlights key challenges and future directions for advancing the field.

### 9. [A Systematic Literature Review on Detecting Software Vulnerabilities with Large Language Models](http://arxiv.org/pdf/2507.22659v1)

Authors: Sabrina Kaniewski, Fabian Schmidt, Markus Enzweiler, Michael Menth, Tobias Heer

The increasing adoption of Large Language Models (LLMs) in software
engineering has sparked interest in their use for software vulnerability
detection. However, the rapid development of this field has resulted in a
fragmented research landscape, with diverse studies that are difficult to
compare due to differences in, e.g., system designs and dataset usage. This
fragmentation makes it difficult to obtain a clear overview of the
state-of-the-art or compare and categorize studies meaningfully. In this work,
we present a comprehensive systematic literature review (SLR) of LLM-based
software vulnerability detection. We analyze 227 studies published between
January 2020 and June 2025, categorizing them by task formulation, input
representation, system architecture, and adaptation techniques. Further, we
analyze the datasets used, including their characteristics, vulnerability
coverage, and diversity. We present a fine-grained taxonomy of vulnerability
detection approaches, identify key limitations, and outline actionable future
research opportunities. By providing a structured overview of the field, this
review improves transparency and serves as a practical guide for researchers
and practitioners aiming to conduct more comparable and reproducible research.
We publicly release all artifacts and maintain a living repository of LLM-based
software vulnerability detection studies.

### 10. [RobEthiChor: Automated Context-aware Ethics-based Negotiation for Autonomous Robots](http://arxiv.org/pdf/2507.22664v1)

Authors: Mashal Afzal Memon, Gianluca Filippone, Gian Luca Scoccia, Marco Autili, Paola Inverardi

The presence of autonomous systems is growing at a fast pace and it is
impacting many aspects of our lives. Designed to learn and act independently,
these systems operate and perform decision-making without human intervention.
However, they lack the ability to incorporate users' ethical preferences, which
are unique for each individual in society and are required to personalize the
decision-making processes. This reduces user trust and prevents autonomous
systems from behaving according to the moral beliefs of their end-users. When
multiple systems interact with differing ethical preferences, they must
negotiate to reach an agreement that satisfies the ethical beliefs of all the
parties involved and adjust their behavior consequently. To address this
challenge, this paper proposes RobEthiChor, an approach that enables autonomous
systems to incorporate user ethical preferences and contextual factors into
their decision-making through ethics-based negotiation. RobEthiChor features a
domain-agnostic reference architecture for designing autonomous systems capable
of ethic-based negotiating. The paper also presents RobEthiChor-Ros, an
implementation of RobEthiChor within the Robot Operating System (ROS), which
can be deployed on robots to provide them with ethics-based negotiation
capabilities. To evaluate our approach, we deployed RobEthiChor-Ros on real
robots and ran scenarios where a pair of robots negotiate upon resource
contention. Experimental results demonstrate the feasibility and effectiveness
of the system in realizing ethics-based negotiation. RobEthiChor allowed robots
to reach an agreement in more than 73\% of the scenarios with an acceptable
negotiation time (0.67s on average). Experiments also demonstrate that the
negotiation approach implemented in RobEthiChor is scalable.

### Social and Information Networks

### 1. [Diffusion Models for Influence Maximization on Temporal Networks: A Guide to Make the Best Choice](http://arxiv.org/pdf/2507.22589v1)

Authors: Aaqib Zahoor, Iqra Altaf Gillani, Janibul Bashir

The increasing prominence of temporal networks in online social platforms and
dynamic communication systems has made influence maximization a critical
research area. Various diffusion models have been proposed to capture the
spread of information, yet selecting the most suitable model for a given
scenario remains challenging. This article provides a structured guide to
making the best choice among diffusion models for influence maximization on
temporal networks. We categorize existing models based on their underlying
mechanisms and assess their effectiveness in different network settings. We
analyze seed selection strategies, highlighting how the inherent properties of
influence spread enable the development of efficient algorithms that can find
near-optimal sets of influential nodes. By comparing key advancements,
challenges, and practical applications, we offer a comprehensive roadmap for
researchers and practitioners to navigate the landscape of temporal influence
maximization effectively.

### 2. [Human Mobility in Epidemic Modeling](http://arxiv.org/pdf/2507.22799v1)

Authors: Xin Lu, Jiawei Feng, Shengjie Lai, Petter Holme, Shuo Liu, Zhanwei Du, Xiaoqian Yuan, Siqing Wang, Yunxuan Li, Xiaoyu Zhang, Yuan Bai, Xiaojun Duan, Wenjun Mei, Hongjie Yu, Suoyi Tan, Fredrik Liljeros

Human mobility forms the backbone of contact patterns through which
infectious diseases propagate, fundamentally shaping the spatio-temporal
dynamics of epidemics and pandemics. While traditional models are often based
on the assumption that all individuals have the same probability of infecting
every other individual in the population, a so-called random homogeneous
mixing, they struggle to capture the complex and heterogeneous nature of
real-world human interactions. Recent advancements in data-driven methodologies
and computational capabilities have unlocked the potential of integrating
high-resolution human mobility data into epidemic modeling, significantly
improving the accuracy, timeliness, and applicability of epidemic risk
assessment, contact tracing, and intervention strategies. This review provides
a comprehensive synthesis of the current landscape in human mobility-informed
epidemic modeling. We explore diverse sources and representations of human
mobility data, and then examine the behavioral and structural roles of mobility
and contact in shaping disease transmission dynamics. Furthermore, the review
spans a wide range of epidemic modeling approaches, ranging from classical
compartmental models to network-based, agent-based, and machine learning
models. And we also discuss how mobility integration enhances risk management
and response strategies during epidemics. By synthesizing these insights, the
review can serve as a foundational resource for researchers and practitioners,
bridging the gap between epidemiological theory and the dynamic complexities of
human interaction while charting clear directions for future research.

### Systems and Control

### 1. [Design and Experimental Validation of UAV Swarm-Based Phased Arrays with MagSafe- and LEGO-Inspired RF Connectors](http://arxiv.org/pdf/2507.22295v1)

Authors: Bidya Debnath, Mst Mostary Begum, Prashant Neupant, Brooke E. Molen, Junming Diao

This paper presents a novel UAV swarm-based phased array antenna system that
leverages MagSafe- and LEGO-inspired radio frequency (RF) connectors to address
key challenges in distributed phased arrays, including inter-element oscillator
synchronization, localization, phase coherence, and positional accuracy. The
proposed non-threaded, hands-free connectors enable precise inter-element
spacing and establish a continuous, low-loss RF signal propagation path during
mid-flight docking. A multi-stage optimization of the RF connector achieves a
compact form factor, DC-to-RF bandwidth, and a measured insertion loss as low
as 0.2\,dB. The system architecture offers scalability in gain and frequency by
adjusting the array element density per UAV and UAV dimensions. Experimental
results from both stationary and in-flight tests of two UAV-based phased array
prototypes align closely with simulations, demonstrating robust beam steering
to multiple directions. This work delivers a practical, scalable, and
low-complexity platform that enables rapid deployment for next-generation
airborne communications, radar, and remote sensing applications.

### 2. [Assessing Value of Renewable-based VPP Versus Electrical Storage: Multi-market Participation Under Different Scheduling Regimes and Uncertainties](http://arxiv.org/pdf/2507.22496v1)

Authors: Hadi Nemati, Ignacio Egido, Pedro Sánchez-Martín, Álvaro Ortega

This paper compares the participation of Renewable-only Virtual Power Plants
(RVPPs) and grid-scale Electrical Storage Systems (ESSs) in energy and reserve
markets, evaluating their technical performance, market strategies, and
economic outcomes. To ensure a fair comparison, scheduling is analyzed over
representative sample days that capture seasonal operating regimes, and the
associated uncertainties are explicitly modeled. Two-stage robust optimization
frameworks are employed: the RVPP model addresses price, generation, and demand
uncertainties, whereas the ESS model considers price uncertainty only. In
addition, an algorithm is proposed for sizing the ESS so that its market
performance matches that of the RVPP. Simulations cover both favorable and
unfavorable scenarios, reflecting seasonal energy limits for dispatchable
resources, varying forecast errors for nondispatchable resources, and
alternative uncertainty-management strategies. The results provide operators
with quantitative guidance on the relative value of each approach.

### 3. [Malleability-Resistant Encrypted Control System with Disturbance Compensation and Real-Time Attack Detection](http://arxiv.org/pdf/2507.22693v1)

Authors: Naoki Aizawa, Keita Emura, Kiminao Kogiso

This study proposes an encrypted PID control system with a disturbance
observer (DOB) using a keyed-homomorphic encryption (KHE) scheme, aiming to
achieve control performance while providing resistance to malleability-based
attacks. The controller integrates a DOB with a PID structure to compensate for
modeling uncertainties by estimating and canceling external disturbances. To
enhance security, the system is designed to output error symbols when
ciphertexts are falsified during decryption or evaluation, enabling real-time
detection of malleability-based signal or parameter falsification. To validate
the proposed method, we conduct stage positioning control experiments and
attack detection tests using an industrial linear stage. The results show that
the encrypted DOB-based PID controller outperforms a conventional encrypted PID
controller in terms of tracking accuracy. Furthermore, the system successfully
detects two types of malleability-based attacks: one that destabilizes the
control system, and another that degrades its performance. The primary
contributions of this study are: (i) the implementation of a KHE-based
encrypted DOB-PID controller, (ii) the improvement of control performance under
uncertainties, and (iii) the experimental demonstration of attack detection
capabilities in encrypted control systems.

### 4. [Foundations for Energy-Aware Zero-Energy Devices: From Energy Sensing to Adaptive Protocols](http://arxiv.org/pdf/2507.22740v1)

Authors: Onel L. A. López, Mateen Ashraf, Samer Nasser, Gabriel M. de Jesus, Ritesh Kumar Singh, Miltiadis C. Filippou, Jeroen Famaey

Zero-energy devices (ZEDs) are key enablers of sustainable Internet of Things
networks by operating solely on harvested ambient energy. Their limited and
dynamic energy budget calls for protocols that are energy-aware and
intelligently adaptive. However, designing effective energy-aware protocols for
ZEDs requires theoretical models that realistically reflect device constraints.
Indeed, existing approaches often oversimplify key aspects such as energy
information (EI) acquisition, task-level variability, and energy storage
dynamics, limiting their practical relevance and transferability. This article
addresses this gap by offering a structured overview of the key modeling
components, trade-offs, and limitations involved in energy-aware ZED protocol
design. For this, we dissect EI acquisition methods and costs, characterize
core operational tasks, analyze energy usage models and storage constraints,
and review representative protocol strategies. Moreover, we offer design
insights and guidelines on how ZED operation protocols can leverage EI, often
illustrated through selected in-house examples. Finally, we outline key
research directions to inspire more efficient and scalable protocol solutions
for future ZEDs.

### 5. [Cluster Synchronization and Phase Cohesiveness of Kuramoto Oscillators via Mean-phase Feedback Control and Pacemakers](http://arxiv.org/pdf/2507.22778v1)

Authors: Ryota Kokubo, Rui Kato, Hideaki Ishii

Brain networks typically exhibit characteristic synchronization patterns
where several synchronized clusters coexist. On the other hand, neurological
disorders are considered to be related to pathological synchronization such as
excessive synchronization of large populations of neurons. Motivated by these
phenomena, this paper presents two approaches to control the cluster
synchronization and the cluster phase cohesiveness of Kuramoto oscillators. One
is based on feeding back the mean phases to the clusters, and the other is
based on the use of pacemakers. First, we show conditions on the feedback gains
and the pacemaker weights for the network to achieve cluster synchronization.
Then, we propose a method to find optimal feedback gains through convex
optimization. Second, we show conditions on the feedback gains and the
pacemaker weights for the network to achieve cluster phase cohesiveness. A
numerical example demonstrates the effectiveness of the proposed methods.

### 6. [Resilient State Recovery using Prior Measurement Support Information](http://arxiv.org/pdf/2507.22340v1)

Authors: Yu Zheng, Olugbenga Moses Anubi, Warren E. Dixon

Resilient state recovery of cyber-physical systems has attracted much
research attention due to the unique challenges posed by the tight coupling
between communication, computation, and the underlying physics of such systems.
By modeling attacks as additive adversary signals to a sparse subset of
measurements, this resilient recovery problem can be formulated as an error
correction problem. To achieve exact state recovery, most existing results
require less than $50\%$ of the measurement nodes to be compromised, which
limits the resiliency of the estimators. In this paper, we show that observer
resiliency can be further improved by incorporating data-driven prior
information. We provide an analytical bridge between the precision of prior
information and the resiliency of the estimator. By quantifying the
relationship between the estimation error of the weighted $\ell_1$ observer and
the precision of the support prior. This quantified relationship provides
guidance for the estimator's weight design to achieve optimal resiliency.
Several numerical simulations and an application case study are presented to
validate the theoretical claims.

### 7. [Safety Evaluation of Motion Plans Using Trajectory Predictors as Forward Reachable Set Estimators](http://arxiv.org/pdf/2507.22389v1)

Authors: Kaustav Chakraborty, Zeyuan Feng, Sushant Veer, Apoorva Sharma, Wenhao Ding, Sever Topan, Boris Ivanovic, Marco Pavone, Somil Bansal

The advent of end-to-end autonomy stacks - often lacking interpretable
intermediate modules - has placed an increased burden on ensuring that the
final output, i.e., the motion plan, is safe in order to validate the safety of
the entire stack. This requires a safety monitor that is both complete (able to
detect all unsafe plans) and sound (does not flag safe plans). In this work, we
propose a principled safety monitor that leverages modern multi-modal
trajectory predictors to approximate forward reachable sets (FRS) of
surrounding agents. By formulating a convex program, we efficiently extract
these data-driven FRSs directly from the predicted state distributions,
conditioned on scene context such as lane topology and agent history. To ensure
completeness, we leverage conformal prediction to calibrate the FRS and
guarantee coverage of ground-truth trajectories with high probability. To
preserve soundness in out-of-distribution (OOD) scenarios or under predictor
failure, we introduce a Bayesian filter that dynamically adjusts the FRS
conservativeness based on the predictor's observed performance. We then assess
the safety of the ego vehicle's motion plan by checking for intersections with
these calibrated FRSs, ensuring the plan remains collision-free under plausible
future behaviors of others. Extensive experiments on the nuScenes dataset show
our approach significantly improves soundness while maintaining completeness,
offering a practical and reliable safety monitor for learned autonomy stacks.

### 8. [Distributed Average Consensus in Wireless Multi-Agent Systems with Over-the-Air Aggregation](http://arxiv.org/pdf/2507.22648v1)

Authors: Themistoklis Charalambous, Zheng Chen, Christoforos N. Hadjicostis

In this paper, we address the average consensus problem of multi-agent
systems over wireless networks. We propose a distributed average consensus
algorithm by invoking the concept of over-the-air aggregation, which exploits
the signal superposition property of wireless multiple-access channels. The
proposed algorithm deploys a modified version of the well-known Ratio Consensus
algorithm with an additional normalization step for compensating for the
arbitrary channel coefficients. We show that, when the noise level at the
receivers is negligible, the algorithm converges asymptotically to the average
for time-invariant and time-varying channels. Numerical simulations corroborate
the validity of our results.

### 9. [Bayesian Optimization of Process Parameters of a Sensor-Based Sorting System using Gaussian Processes as Surrogate Models](http://arxiv.org/pdf/2507.22766v1)

Authors: Felix Kronenwett, Georg Maier, Thomas Laengle

Sensor-based sorting systems enable the physical separation of a material
stream into two fractions. The sorting decision is based on the image data
evaluation of the sensors used and is carried out using actuators. Various
process parameters must be set depending on the properties of the material
stream, the dimensioning of the system, and the required sorting accuracy.
However, continuous verification and re-adjustment are necessary due to
changing requirements and material stream compositions. In this paper, we
introduce an approach for optimizing, recurrently monitoring and adjusting the
process parameters of a sensor-based sorting system. Based on Bayesian
Optimization, Gaussian process regression models are used as surrogate models
to achieve specific requirements for system behavior with the uncertainties
contained therein. This method minimizes the number of necessary experiments
while simultaneously considering two possible optimization targets based on the
requirements for both material output streams. In addition, uncertainties are
considered during determining sorting accuracies in the model calculation. We
evaluated the method with three example process parameters.

### 10. [Bayesian Optimization applied for accelerated Virtual Validation of the Autonomous Driving Function](http://arxiv.org/pdf/2507.22769v1)

Authors: Satyesh Shanker Awasthi, Mohammed Irshadh Ismaaeel Sathyamangalam Imran, Stefano Arrigoni, Francesco Braghin

Rigorous Verification and Validation (V&V) of Autonomous Driving Functions
(ADFs) is paramount for ensuring the safety and public acceptance of Autonomous
Vehicles (AVs). Current validation relies heavily on simulation to achieve
sufficient test coverage within the Operational Design Domain (ODD) of a
vehicle, but exhaustively exploring the vast parameter space of possible
scenarios is computationally expensive and time-consuming. This work introduces
a framework based on Bayesian Optimization (BO) to accelerate the discovery of
critical scenarios. We demonstrate the effectiveness of the framework on an
Model Predictive Controller (MPC)-based motion planner, showing that it
identifies hazardous situations, such as off-road events, using orders of
magnitude fewer simulations than brute-force Design of Experiments (DoE)
methods. Furthermore, this study investigates the scalability of the framework
in higher-dimensional parameter spaces and its ability to identify multiple,
distinct critical regions within the ODD of the motion planner used as the case
study .

### Machine Learning (Statistics Category)

### 1. [A Unified Analysis of Generalization and Sample Complexity for Semi-Supervised Domain Adaptation](http://arxiv.org/pdf/2507.22632v1)

Authors: Elif Vural, Huseyin Karaca

Domain adaptation seeks to leverage the abundant label information in a
source domain to improve classification performance in a target domain with
limited labels. While the field has seen extensive methodological development,
its theoretical foundations remain relatively underexplored. Most existing
theoretical analyses focus on simplified settings where the source and target
domains share the same input space and relate target-domain performance to
measures of domain discrepancy. Although insightful, these analyses may not
fully capture the behavior of modern approaches that align domains into a
shared space via feature transformations. In this paper, we present a
comprehensive theoretical study of domain adaptation algorithms based on domain
alignment. We consider the joint learning of domain-aligning feature
transformations and a shared classifier in a semi-supervised setting. We first
derive generalization bounds in a broad setting, in terms of covering numbers
of the relevant function classes. We then extend our analysis to characterize
the sample complexity of domain-adaptive neural networks employing maximum mean
discrepancy (MMD) or adversarial objectives. Our results rely on a rigorous
analysis of the covering numbers of these architectures. We show that, for both
MMD-based and adversarial models, the sample complexity admits an upper bound
that scales quadratically with network depth and width. Furthermore, our
analysis suggests that in semi-supervised settings, robustness to limited
labeled target data can be achieved by scaling the target loss proportionally
to the square root of the number of labeled target samples. Experimental
evaluation in both shallow and deep settings lends support to our theoretical
findings.

### 2. [Subgrid BoostCNN: Efficient Boosting of Convolutional Networks via Gradient-Guided Feature Selection](http://arxiv.org/pdf/2507.22842v1)

Authors: Biyi Fang, Jean Utke, Truong Vo, Diego Klabjan

Convolutional Neural Networks (CNNs) have achieved remarkable success across
a wide range of machine learning tasks by leveraging hierarchical feature
learning through deep architectures. However, the large number of layers and
millions of parameters often make CNNs computationally expensive to train,
requiring extensive time and manual tuning to discover optimal architectures.
In this paper, we introduce a novel framework for boosting CNN performance that
integrates dynamic feature selection with the principles of BoostCNN. Our
approach incorporates two key strategies: subgrid selection and importance
sampling, to guide training toward informative regions of the feature space. We
further develop a family of algorithms that embed boosting weights directly
into the network training process using a least squares loss formulation. This
integration not only alleviates the burden of manual architecture design but
also enhances accuracy and efficiency. Experimental results across several
fine-grained classification benchmarks demonstrate that our boosted CNN
variants consistently outperform conventional CNNs in both predictive
performance and training speed.

### 3. [Consistency of Feature Attribution in Deep Learning Architectures for Multi-Omics](http://arxiv.org/pdf/2507.22877v1)

Authors: Daniel Claborne, Javier Flores, Samantha Erwin, Luke Durell, Rachel Richardson, Ruby Fore, Lisa Bramer

Machine and deep learning have grown in popularity and use in biological
research over the last decade but still present challenges in interpretability
of the fitted model. The development and use of metrics to determine features
driving predictions and increase model interpretability continues to be an open
area of research. We investigate the use of Shapley Additive Explanations
(SHAP) on a multi-view deep learning model applied to multi-omics data for the
purposes of identifying biomolecules of interest. Rankings of features via
these attribution methods are compared across various architectures to evaluate
consistency of the method. We perform multiple computational experiments to
assess the robustness of SHAP and investigate modeling approaches and
diagnostics to increase and measure the reliability of the identification of
important features. Accuracy of a random-forest model fit on subsets of
features selected as being most influential as well as clustering quality using
only these features are used as a measure of effectiveness of the attribution
method. Our findings indicate that the rankings of features resulting from SHAP
are sensitive to the choice of architecture as well as different random
initializations of weights, suggesting caution when using attribution methods
on multi-view deep learning models applied to multi-omics data. We present an
alternative, simple method to assess the robustness of identification of
important biomolecules.

### 4. [LVM-GP: Uncertainty-Aware PDE Solver via coupling latent variable model and Gaussian process](http://arxiv.org/pdf/2507.22493v1)

Authors: Xiaodong Feng, Ling Guo, Xiaoliang Wan, Hao Wu, Tao Zhou, Wenwen Zhou

We propose a novel probabilistic framework, termed LVM-GP, for uncertainty
quantification in solving forward and inverse partial differential equations
(PDEs) with noisy data. The core idea is to construct a stochastic mapping from
the input to a high-dimensional latent representation, enabling
uncertainty-aware prediction of the solution. Specifically, the architecture
consists of a confidence-aware encoder and a probabilistic decoder. The encoder
implements a high-dimensional latent variable model based on a Gaussian process
(LVM-GP), where the latent representation is constructed by interpolating
between a learnable deterministic feature and a Gaussian process prior, with
the interpolation strength adaptively controlled by a confidence function
learned from data. The decoder defines a conditional Gaussian distribution over
the solution field, where the mean is predicted by a neural operator applied to
the latent representation, allowing the model to learn flexible
function-to-function mapping. Moreover, physical laws are enforced as soft
constraints in the loss function to ensure consistency with the underlying PDE
structure. Compared to existing approaches such as Bayesian physics-informed
neural networks (B-PINNs) and deep ensembles, the proposed framework can
efficiently capture functional dependencies via merging a latent Gaussian
process and neural operator, resulting in competitive predictive accuracy and
robust uncertainty quantification. Numerical experiments demonstrate the
effectiveness and reliability of the method.

### 5. [Quantum-assisted Gaussian process regression using random Fourier features](http://arxiv.org/pdf/2507.22629v1)

Authors: Cristian A. Galvis-Florez, Ahmad Farooq, Simo Särkkä

Probabilistic machine learning models are distinguished by their ability to
integrate prior knowledge of noise statistics, smoothness parameters, and
training data uncertainty. A common approach involves modeling data with
Gaussian processes; however, their computational complexity quickly becomes
intractable as the training dataset grows. To address this limitation, we
introduce a quantum-assisted algorithm for sparse Gaussian process regression
based on the random Fourier feature kernel approximation. We start by encoding
the data matrix into a quantum state using a multi-controlled unitary
operation, which encodes the classical representation of the random Fourier
features matrix used for kernel approximation. We then employ a quantum
principal component analysis along with a quantum phase estimation technique to
extract the spectral decomposition of the kernel matrix. We apply a conditional
rotation operator to the ancillary qubit based on the eigenvalue. We then use
Hadamard and swap tests to compute the mean and variance of the posterior
Gaussian distribution. We achieve a polynomial-order computational speedup
relative to the classical method.

### 6. [CLuP practically achieves $\sim 1.77$ positive and $\sim 0.33$ negative Hopfield model ground state free energy](http://arxiv.org/pdf/2507.22396v1)

Authors: Mihailo Stojnic

We study algorithmic aspects of finding $n$-dimensional \emph{positive} and
\emph{negative} Hopfield ($\pm$Hop) model ground state free energies. This
corresponds to classical maximization of random positive/negative semi-definite
quadratic forms over binary $\left \{\pm \frac{1}{\sqrt{n}} \right \}^n$
vectors. The key algorithmic question is whether these problems can be
computationally efficiently approximated within a factor $\approx 1$. Following
the introduction and success of \emph{Controlled Loosening-up} (CLuP-SK)
algorithms in finding near ground state energies of closely related
Sherrington-Kirkpatrick (SK) models [82], we here propose a CLuP$\pm$Hop
counterparts for $\pm$Hop models. Fully lifted random duality theory (fl RDT)
[78] is utilized to characterize CLuP$\pm$Hop \emph{typical} dynamics. An
excellent agreement between practical performance and theoretical predictions
is observed. In particular, for $n$ as small as few thousands CLuP$\pm$Hop
achieve $\sim 1.77$ and $\sim 0.33$ as the ground state free energies of the
positive and negative Hopfield models. At the same time we obtain on the 6th
level of lifting (6-spl RDT) corresponding theoretical thermodynamic
($n\rightarrow\infty$) limits $\approx 1.7784$ and $\approx 0.3281$. This
positions determining Hopfield models near ground state energies as
\emph{typically} easy problems. Moreover, the very same 6th lifting level
evaluations allow to uncover a fundamental intrinsic difference between two
models: $+$Hop's near optimal configurations are \emph{typically close} to each
other whereas the $-$Hop's are \emph{typically far away}.

### 7. [Safe Deployment of Offline Reinforcement Learning via Input Convex Action Correction](http://arxiv.org/pdf/2507.22640v1)

Authors: Alex Durkin, Jasper Stolte, Matthew Jones, Raghuraman Pitchumani, Bei Li, Christian Michler, Mehmet Mercangöz

Offline reinforcement learning (offline RL) offers a promising framework for
developing control strategies in chemical process systems using historical
data, without the risks or costs of online experimentation. This work
investigates the application of offline RL to the safe and efficient control of
an exothermic polymerisation continuous stirred-tank reactor. We introduce a
Gymnasium-compatible simulation environment that captures the reactor's
nonlinear dynamics, including reaction kinetics, energy balances, and
operational constraints. The environment supports three industrially relevant
scenarios: startup, grade change down, and grade change up. It also includes
reproducible offline datasets generated from proportional-integral controllers
with randomised tunings, providing a benchmark for evaluating offline RL
algorithms in realistic process control tasks.
  We assess behaviour cloning and implicit Q-learning as baseline algorithms,
highlighting the challenges offline agents face, including steady-state offsets
and degraded performance near setpoints. To address these issues, we propose a
novel deployment-time safety layer that performs gradient-based action
correction using input convex neural networks (PICNNs) as learned cost models.
The PICNN enables real-time, differentiable correction of policy actions by
descending a convex, state-conditioned cost surface, without requiring
retraining or environment interaction.
  Experimental results show that offline RL, particularly when combined with
convex action correction, can outperform traditional control approaches and
maintain stability across all scenarios. These findings demonstrate the
feasibility of integrating offline RL with interpretable and safety-aware
corrections for high-stakes chemical process control, and lay the groundwork
for more reliable data-driven automation in industrial systems.

### 8. [A Bit of Freedom Goes a Long Way: Classical and Quantum Algorithms for Reinforcement Learning under a Generative Model](http://arxiv.org/pdf/2507.22854v1)

Authors: Andris Ambainis, Joao F. Doriguello, Debbie Lim

We propose novel classical and quantum online algorithms for learning
finite-horizon and infinite-horizon average-reward Markov Decision Processes
(MDPs). Our algorithms are based on a hybrid exploration-generative
reinforcement learning (RL) model wherein the agent can, from time to time,
freely interact with the environment in a generative sampling fashion, i.e., by
having access to a "simulator". By employing known classical and new quantum
algorithms for approximating optimal policies under a generative model within
our learning algorithms, we show that it is possible to avoid several paradigms
from RL like "optimism in the face of uncertainty" and "posterior sampling" and
instead compute and use optimal policies directly, which yields better regret
bounds compared to previous works. For finite-horizon MDPs, our quantum
algorithms obtain regret bounds which only depend logarithmically on the number
of time steps $T$, thus breaking the $O(\sqrt{T})$ classical barrier. This
matches the time dependence of the prior quantum works of Ganguly et al.
(arXiv'23) and Zhong et al. (ICML'24), but with improved dependence on other
parameters like state space size $S$ and action space size $A$. For
infinite-horizon MDPs, our classical and quantum bounds still maintain the
$O(\sqrt{T})$ dependence but with better $S$ and $A$ factors. Nonetheless, we
propose a novel measure of regret for infinite-horizon MDPs with respect to
which our quantum algorithms have $\operatorname{poly}\log{T}$ regret,
exponentially better compared to classical algorithms. Finally, we generalise
all of our results to compact state spaces.



---

# Nature Computer Science Reports

Collection of today's Computer Science research papers pulled from Nature Open Access Reports.

---

Pulled on 2025-07-31 PST.

### 1. [Google AI model mines trillions of images to create maps of Earth ‘at any place and time’](https://www.nature.com/articles/d41586-025-02412-1)

Authors: Jeff Tollefson

### 2. [A real-time predictive postural control system with temperature feedback](https://www.nature.com/articles/s41598-025-11334-x)

Authors: Yaoyu Duan et al.

### 3. [An efficient fusion detector for road defect detection](https://www.nature.com/articles/s41598-025-01399-z)

Authors: Li Yang et al.

### 4. [Quantum key-based medical privacy protection and sharing scheme on blockchain](https://www.nature.com/articles/s41598-025-10832-2)

Authors: Dexin Zhu et al.

### 5. [Founder smiles increase investor trust and funding](https://www.nature.com/articles/s41598-025-12544-z)

Authors: Dimosthenis Stefanidis et al.

### 6. [Application of CycleGAN-based low-light image enhancement algorithm in foreign object detection on belt conveyors in underground mines](https://www.nature.com/articles/s41598-025-10779-4)

Authors: Anxin Zhao et al.

### 7. [A novel dynamic scheduling model for application in multimode approach](https://www.nature.com/articles/s41598-025-10710-x)

Authors: Zineb Elqabli et al.

### 8. [A novel flexible identity-net with diffusion models for painting-style generation](https://www.nature.com/articles/s41598-025-12434-4)

Authors: Yifei Zhao et al.

### 9. [A new band selection approach integrated with physical reflectance autoencoders and albedo recovery for hyperspectral image classification](https://www.nature.com/articles/s41598-025-09355-7)

Authors: V. Sangeetha et al.

### 10. [Advancing deep learning for expressive music composition and performance modeling](https://www.nature.com/articles/s41598-025-13064-6)

Authors: Man Zhang

### 11. [Image dehazing algorithm based on deep transfer learning and local mean adaptation](https://www.nature.com/articles/s41598-025-13613-z)

Authors: Dongyang Shi et al.

### 12. [Mixture-of-experts graph transformers for interpretable particle collision detection](https://www.nature.com/articles/s41598-025-12003-9)

Authors: Donatella Genovese et al.

