# RF*diffusion*

<!--
<img width="1115" alt="Screen Shot 2023-01-19 at 5 56 33 PM" src="https://user-images.githubusercontent.com/56419265/213588200-f8f44dba-276e-4dd2-b844-15acc441458d.png">
-->
<p align="center">
  <img src="./img/diffusion_protein_gradient_2.jpg" alt="alt text" width="1100px" align="middle"/>
</p>

*Image: Ian C. Haydon / UW Institute for Protein Design*

## Description

RFdiffusion은 조건부 정보(모티프, 타겟 등) 유무에 관계없이 구조를 생성하는 오픈소스 방법으로 RFdiffusion 논문 [the RFdiffusion paper](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1). 에서 설명한 바와 같이 다양한 단백질 설계 과제를 수행

**Things Diffusion can do**
- Motif Scaffolding ● 모티프 스캐폴딩
- Unconditional protein generation ● 무조건 단백질 생성
- Symmetric unconditional generation ● 대칭적 무조건 생성 (cyclic, dihedral and tetrahedral symmetries currently implemented, more coming!)
- Symmetric motif scaffolding ● 대칭 모티브 스캐폴딩
- Binder design ● 바인더 디자인
- Design diversification ("partial diffusion", sampling around a design) ● 디자인 다각화("부분 확산", 디자인을 중심으로 샘플링)
----

# 0. Table of contents

- [RF*diffusion*](#rfdiffusion)
  - [Description](#description)
- [Table of contents](#table-of-contents)
- [Getting started / installation](#getting-started--installation)
    - [Conda Install SE3-Transformer](#conda-install-se3-transformer) ☞ Conda SE3-Transformer 설치
    - [Get PPI Scaffold Examples](#get-ppi-scaffold-examples) ☞  PPI 스캐폴드 예시 보기
- [Usage](#usage) 사용법
    - [Running the diffusion script](#running-the-diffusion-script) ☞ 확산 스크립트 실행
    - [Basic execution - an unconditional monomer](#basic-execution---an-unconditional-monomer) ☞ 기본 실행 - 무조건 모노머
    - [Motif Scaffolding](#motif-scaffolding) ☞ 모티프 스캐폴딩
    - [The "active site" model holds very small motifs in place](#the-active-site-model-holds-very-small-motifs-in-place) 
       ☞ "활성 사이트" 모델은 매우 작은 모티프를 제자리에 고정
    - [The `inpaint_seq` flag](#the-inpaint_seq-flag) ☞ 깃발inpaint_seq​
    - [A note on `diffuser.T`](#a-note-on-diffusert) ☞ diffuser.T에 대한 참고사항
    - [Partial diffusion](#partial-diffusion) ☞ 부분 확산
    - [Binder Design](#binder-design) ☞ 바인더 디자인
    - [Practical Considerations for Binder Design](#practical-considerations-for-binder-design) 
       ☞ 바인더 디자인을 위한 실용적인 고려 사항
    - [Fold Conditioning](#fold-conditioning) ☞ 폴드 컨디셔닝
    - [Generation of Symmetric Oligomers](#generation-of-symmetric-oligomers) ☞ 대칭 올리고머 생성
    - [Using Auxiliary Potentials](#using-auxiliary-potentials) ☞ 보조 전위 사용 
    - [Symmetric Motif Scaffolding.](#symmetric-motif-scaffolding) ☞ 대칭 모티브 스캐폴딩.
    - [RFpeptides macrocycle design](#macrocyclic-peptide-design-with-rfpeptides) ☞ RF펩타이드 거대고리 디자인
    - [A Note on Model Weights](#a-note-on-model-weights) ☞ 모델 가중치에 대한 참고 사항
    - [Things you might want to play with at inference time](#things-you-might-want-to-play-with-at-inference-time)
       ☞ 추론 시간에 가지고 놀고 싶은 것들
    - [Understanding the output files](#understanding-the-output-files) ☞ 출력 파일 이해
    - [Docker](#docker) ☞ 도커
    - [Conclusion](#conclusion) ☞ 결론

# 1. Getting started / installation

Sergey Ovchinnikov가  RFdiffusion 을 이용할 수 있도록 만듬 [Google Colab Notebook](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/v1.1.1/rf/examples/diffusion.ipynb) 원하면 여기서 시도 해보기! 

RFdiffusion을 시작하기 전에 REDME을 주의 깊게 읽고, Colab Notebook의 몇가지 예시를 살펴보는 것을 권장

RFdiffusion을 로컬로 설정하려면 아래 단계를 따르기:

RFdiffusion을 사용하기 위해 저장소를 복제:
```
git clone https://github.com/RosettaCommons/RFdiffusion.git
```

RFDiffusion directory에 모델 가중치를 다운로드드
```
cd RFdiffusion
mkdir models && cd models
wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt

Optional:
wget http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt

# original structure prediction weights
wget http://files.ipd.uw.edu/pub/RFdiffusion/1befcb9b28e2f778f53d47f18b7597fa/RF_structure_prediction_weights.pt
```


### 1.1 Conda Install SE3-Transformer

[Anaconda 또 Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 가 설치되어있는지 확인하기, 설치가 안되어있다면 설치하기

[NVIDIA의 implementation of SE(3)-Transformers](https://developer.nvidia.com/blog/accelerating-se3-transformers-training-using-an-nvidia-open-source-model-implementation/) 도 설치하기.
SE(3)-Transformers를 설치하는 방법은 아래와 같음

```
conda env create -f env/SE3nv.yml

conda activate SE3nv
cd env/SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ../.. # 저장소의 루트 디렉토리로 변경
pip install -e . # 저장소 루트에서 rfdiffusion 모듈 설치
```
diffusion을 실행할 때마다 다음 명령어을 실행하여 conda 환경을 활성화해야 합니다:
```
conda activate SE3nv
```
표준 데스크톱 컴퓨터의 총 설정 시간은 30분 미만이어야 합니다. 
참고: 사용자가 접근할 수 있는 GPU 유형과 드라이버의 차이로 인해 모든 설정에서 실행될 하나의 환경을 만들 수 없습니다. 따라서 CUDA 11.1을 지원하는 yml 파일만 제공하고 각 사용자가 설정에 맞게 사용자 지정할 수 있도록 남겨두었습니다. 이 사용자 지정에는 yml 파일에 지정된 cudatoolkit 및 (아마도) PyTorch 버전을 변경하는 작업이 포함될 수 있습니다.

---

### 1.2 Get PPI Scaffold Examples

scaffolded protein binder design (PPI) 예제를 실행하기 위해 몇 가지 예제 scaffold files (`examples/ppi_scaffolds_subset.tar.gz`)을 제공 했습니다.
이 압축 파일을 풀어야 합니다:
```
tar -xvf examples/ppi_scaffolds_subset.tar.gz -C examples/
```

이 파일들이 무엇인지와 사용 방법에 대해서는 Fold Conditioning section에서 설명하겠습니다.

----


# 2. Usage
이 부분에서는 diffusion을 실행하는 방법을 시연할 것입니다.



<p align="center">
  <img src="./img/main.png" alt="alt text" width="1100px" align="middle"/>
</p>


### 2.1 Running the diffusion script 
실제 스크립트를 실행할 스크립트는 `scripts/run_inference.py`라고 합니다. 이를 실행하는 방법에는 여러 가지가 있으며, hydra configs에 의해 제어됩니다.
[Hydra configs](https://hydra.cc/docs/configure_hydra/intro/) 는 model checkpoint에서 *직접*가져온 합리적인 기본값으로 다양한 옵션을 지정할 수 있는 좋은 방법 입니다. 따라서 inference 은 항상 기본적으로 training과 일치해야 합니다.
즉, `config/inference/base.yml` 의 기본값이 inference 주에 사용된 실제 값과 특정 checkpoint와 일치하지 않을 수 있습니다. 이 모든것은 background에서 비밀리에 처리됩니다.

---
### 2.2 Basic execution - an unconditional monomer 기본 실행 - 무조건 단량체
<img src="./img/cropped_uncond.png" alt="alt text" width="400px" align="right"/>

먼저 길이 150aa의 단백질을 무조건적으로 설계하는 방법을 살펴보겠습니다. 
이를 위해 세 가지만 지정하면 됩니다:
1. 단백질의 길이
2. 파일을 작성할 위치
3. 우리가 원하는 디자인의 수

```
./scripts/run_inference.py 'contigmap.contigs=[150-150]' inference.output_prefix=test_outputs/test inference.num_designs=10
```

자세히 살펴보도록 합시다
먼저 `contigmap.contigs`란 무엇일까?
Hydra configs는 inference script에 실행 방법을 알려줍니다.
구성을 체계적으로 유지하기 위해 구성에는 다양한 하위 구성 요소가 있으며, 그 중 하나는 contigmap으로,  contig 문자열(구축 중인 단백질을 정의하는)과 관련된 모든 것과 관련이 있습니다. 
명확하지 않은 경우 구성 파일을 살펴보세요: configs/inference/base.yml. 구성의 모든 항목은 명령줄에서 수동으로 덮어쓸 수 있습니다. 예를 들어, diffuser의 작동 방식을 변경할 수 있습니다: 
```
diffuser.crd_scale=0.5
```
... 하지만 당신이 정말로 무엇을 하고 있는지 알지 못하면 이렇게 하지 마세요!!


이제 `'contigmap.contigs=[150-150]'`은 무엇을 의미하나요?
RFjoint inpainting을 사용해 보신 분들에게는 익수하게 보일 수 있지만 조금은 다릅니다.
Diffusion은 사실 inpainting과 같은 'contig mapper'를 사용하지만, 우리는 hydra를 사용하기 때문에, model에게 다른 방식으로 전달해야 한다는 점 때문에 그렇습니다.
hydra 이유로 인해 contig 문자열은 문자열이 아닌 목록의 단일 항복으로 전달되어야 하며, 명령줄이 특수문자를 구문분석하려고 시도하지 않도록 전체 인수는 `''` 안에 포함되어야 합니다.

contig 문자열을 사용하면 생산 단백질의 길이 범위를 지정할 수 있지만, 여기서는 길이가 150aa인 단백질만 필요하므로[150-150]을 지정하기만 하면 10개의 diffusion 경로가 실행되어 지정된 출력폴더(outputs folder)에 저장 됩니다.

RFdiffusion을 처음 실행할 때는 'Calculating IGSO3'에 시간이 좀 걸릴 것입니다. 일단 이 작업을 완료하면 나중에 참조할 수 있도록  캐시가 저장될것입니다. monomer generation의 추가 예시를 보려면 repo에서 `./examples/design_unconditional.sh` 를 확인해 보세요!

---
### 2.3 Motif Scaffolding
<!--
<p align="center">
  <img src="./img/motif.png" alt="alt text" width="700px" align="middle"/>
</p>
-->
RFdiffusion은 [Constrained Hallucination 및 RFjoint Inpainting](https://www.science.org/doi/10.1126/science.abn2100#:~:text=The%20binding%20and%20catalytic%20functions%20of%20proteins%20are,the%20fold%20or%20secondary%20structure%20of%20the%20scaffold.)와 유사한 방식으로 motif를 scaffold하는데 사용할수 있습니다.
일반적으로, RFdiffusion은 Constrained Hallucination 과 RFjoint Inpainting보다 성능이 훨씬 좋습니다.
<p align="center">
  <img src="./img/motif.png" alt="alt text" width="700px" align="middle"/>
</p>

단백질 motif를 scaffolding 할 때 특정 단백질 입력(inputs)(`.pdb` file 파일에서 하나 이상의 segments)을 scaffolding하고, 새로운 scaffolding된 단백질에서 이러한 연결 방법과 잔기(residues) 수를 지정할 수 있는 방법이 필요합니다. 
게다가 우리는 일반적으로 motif를 최적화 하기 위해 얼마나 많은 잔기가 필요한지 정확히 알지 못하기 때문에 다양한 길이의 connecting protein을 sampling할 수 있기를 원합니다.
입력(inputs)을 지정하는 이 작업은 hydra config의 contigmap config에 제어되는 contigs에 의해 처리됩니다.
Constrained Hallucination이나 RFjoint Inpainting과 이 logic은 매우 유사합니다.
Briefly:
- Anything prefixed by a letter indicates that this is a motif, with the letter corresponding to the chain letter in the input pdb files. E.g. A10-25 pertains to residues ('A',10),('A',11)...('A',25) in the corresponding input pdb
- Anything not prefixed by a letter indicates protein *to be built*. This can be input as a length range. These length ranges are randomly sampled each iteration of RFdiffusion inference. 
- To specify chain breaks, we use `/0 `.

In more detail, if we want to scaffold a motif, the input is just like RFjoint Inpainting, except needing to navigate the hydra config input. If we want to scaffold residues 10-25 on chain A a pdb, this would be done with `'contigmap.contigs=[5-15/A10-25/30-40]'`. This asks RFdiffusion to build 5-15 residues (randomly sampled at each inference cycle) N-terminally of A10-25 from the input pdb, followed by 30-40 residues (again, randomly sampled) to its C-terminus. If we wanted to ensure the length was always e.g. 55 residues, this can be specified with `contigmap.length=55-55`. You need to obviously also provide a path to your pdb file: `inference.input_pdb=path/to/file.pdb`. It doesn't matter if your input pdb has residues you *don't* want to scaffold - the contig map defines which residues in the pdb are actually used as the "motif". In other words, even if your pdb files has a B chain, and other residues on the A chain, *only* A10-25 will be provided to RFdiffusion.

To specify that we want to inpaint in the presence of a separate chain, this can be done as follows:

```
'contigmap.contigs=[5-15/A10-25/30-40/0 B1-100]'
```
Look at this carefully. `/0 ` is the indicator that we want a chain break. NOTE, the space is important here. This tells the diffusion model to add a big residue jump (200aa) to the input, so that the model sees the first chain as being on a separate chain to the second.

An example of motif scaffolding can be found in `./examples/design_motifscaffolding.sh`.

### 2.4 The "active site" model holds very small motifs in place
In the RFdiffusion preprint we noted that for very small motifs, RFdiffusion has the tendency to not keep them perfectly fixed in the output. Therefore, for scaffolding minimalist sites such as enzyme active sites, we fine-tuned RFdiffusion on examples similar to these tasks, allowing it to hold smaller motifs better in place, and better generate *in silico* successes. If your input functional motif is very small, we reccomend using this model, which can easily be specified using the following syntax: 
`inference.ckpt_override_path=models/ActiveSite_ckpt.pt`

### 2.5 The `inpaint_seq` flag
For those familiar with RFjoint Inpainting, the contigmap.inpaint_seq input is equivalent. The idea is that often, when, for example, fusing two proteins, residues that were on the surface of a protein (and are therefore likely polar), now need to be packed into the 'core' of the protein. We therefore want them to become hydrophobic residues. What we can do, rather than directly mutating them to hydrophobics, is to mask their sequence identity, and allow RFdiffusion to implicitly reason over their sequence, and better pack against them. This requires a different model than the 'base' diffusion model, that has been trained to understand this paradigm, but this is automatically handled by the inference script (you don't need to do anything).

To specify amino acids whose sequence should be hidden, use the following syntax:
```
'contigmap.inpaint_seq=[A1/A30-40]'
```
Here, we're masking the residue identity of residue A1, and all residues between A30 and A40 (inclusive). 

An example of executing motif scaffolding with the `contigmap.inpaint_seq` flag is located in `./examples/design_motifscaffolding_inpaintseq.sh`

### 2.6 A note on `diffuser.T`
RFdiffusion was originally trained with 200 discrete timesteps. However, recent improvements have allowed us to reduce the number of timesteps we need to use at inference time. In many cases, running with as few as approximately 20 steps provides outputs of equivalent *in silico* quality to running with 200 steps (providing a 10X speedup). The default is now set to 50 steps. Noting this is important for understanding the partial diffusion, described below. 

---
### 2.7 Partial diffusion

Something we can do with diffusion is to partially noise and de-noise a structure, to get some diversity around a general fold. This can work really nicely (see [Vazquez-Torres et al., BioRxiv 2022](https://www.biorxiv.org/content/10.1101/2022.12.10.519862v4.abstract)).
This is specified by using the diffuser.parial_T input, and setting a timestep to 'noise' to. 
<p align="center">
  <img src="./img/partial.png" alt="alt text" width="800px" align="middle"/>
</p>
More noise == more diversity. In Vazquez-Torres et al., 2022, we typically used `diffuser.partial_T` of approximately 80, but this was with respect to the 200 timesteps we were using. Now that the default `diffuser.T` is 50, you will need to adjust diffuser.partial_T accordingly. E.g. now that `diffuser.T=50`, the equivalent of 80 noising steps is `diffuser.partial_T=20`. We strongly recommend sampling different values for `partial_T` however, to find the best parameters for your specific problem.

When doing partial diffusion, because we are now diffusing from a known structure, this creates certain constraints. You can still use the contig input, but *this has to yield a contig string exactly the same length as the input protein*. E.g. if you have a binder:target complex, and you want to diversify the binder (length 100, chain A), you would need to input something like this:

```
'contigmap.contigs=[100-100/0 B1-150]' diffuser.partial_T=20
```
The reason for this is that, if your input protein was only 80 amino acids, but you've specified a desired length of 100, we don't know where to diffuse those extra 20 amino acids from, and hence, they will not lie in the distribution that RFdiffusion has learned to denoise from.

An example of partial diffusion can be found in `./examples/design_partialdiffusion.sh`! 

You can also keep parts of the sequence of the diffused chain fixed, if you want. An example of why you might want to do this is in the context of helical peptide binding. If you've threaded a helical peptide sequence onto an ideal helix, and now want to diversify the complex, allowing the helix to be predicted now not as an ideal helix, you might do something like:

```
'contigmap.contigs=[100-100/0 20-20]' 'contigmap.provide_seq=[100-119]' diffuser.partial_T=10
```
In this case, the 20aa chain is the helical peptide. The `contigmap.provide_seq` input is zero-indexed, and you can provide a range (so 100-119 is an inclusive range, unmasking the whole sequence of the peptide). Multiple sequence ranges can be provided separated by a comma, e.g. `'contigmap.provide_seq=[172-177,200-205]'`.

Note that the provide_seq option requires using a different model checkpoint, but this is automatically handled by the inference script.

An example of partial diffusion with providing sequence in diffused regions can be found in `./examples/design_partialdiffusion_withseq.sh`. The same example specifying multiple sequence ranges can be found in `./examples/design_partialdiffusion_multipleseq.sh`.

---
### 2.8 Binder Design 
Hopefully, it's now obvious how you might make a binder with diffusion! Indeed, RFdiffusion shows excellent *in silico* and experimental ability to design *de novo* binders. 

<p align="center">
  <img src="./img/binder.png" alt="alt text" width="950px" align="middle"/>
</p>

If chain B is your target, then you could do it like this:

```
./scripts/run_inference.py 'contigmap.contigs=[B1-100/0 100-100]' inference.output_prefix=test_outputs/binder_test inference.num_designs=10
```

This will generate 100 residue long binders to residues 1-100 of chain B.

However, this probably isn't the best way of making binders. Because diffusion is somewhat computationally-intensive, we need to try and make it as fast as possible. Providing the whole of your target, uncropped, is going to make diffusion very slow if your target is big (and most targets-of-interest, such as cell-surface receptors tend to be *very* big). One tried-and-true method to speed up binder design is to crop the target protein around the desired interface location. BUT! This creates a problem: if you crop your target and potentially expose hydrophobic core residues which were buried before the crop, how can you guarantee the binder will go to the intended interface site on the surface of the target, and not target the tantalizing hydrophobic patch you have just artificially created?

We solve this issue by providing the model with what we call "hotspot residues". The complex models we refer to earlier in this README file have all been trained with hotspot residues, in this training regime, during each example, the model is told (some of) the residues on the target protein which contact the target (i.e., resides that are part of the interface). The model readily learns that it should be making an interface which involved these hotspot residues. At inference time then, we can provide our own hotspot residues to define a region which the binder must contact. These are specified like this: `'ppi.hotspot_res=[A30,A33,A34]'`, where `A` is the chain ID in the input pdb file of the hotspot residue and the number is the residue index in the input pdb file of the hotspot residue.

Finally, it has been observed that the default RFdiffusion model often generates mostly helical binders. These have high computational and experimental success rates. However, there may be cases where other kinds of topologies may be desired. For this, we include a "beta" model, which generates a greater diversity of topologies, but has not been extensively experimentally validated. Try this at your own risk:

```
inference.ckpt_override_path=models/Complex_beta_ckpt.pt
```

An example of binder design with RFdiffusion can be found in `./examples/design_ppi.sh`.

---

## 2.8.1. Practical Considerations for Binder Design

RFdiffusion is an extremely powerful binder design tool but it is not magic. In this section we will walk through some common pitfalls in RFdiffusion binder design and offer advice on how to get the most out of this method.

### 2.8.1.1 Selecting a Target Site
Not every site on a target protein is a good candidate for binder design. For a site to be an attractive candidate for binding it should have >~3 hydrophobic residues for the binder to interact with. Binding to charged polar sites is still quite hard. Binding to sites with glycans close to them is also hard since they often become ordered upon binding and you will take an energetic hit for that. Historically, binder design has also avoided unstructured loops, it is not clear if this is still a requirement as RFdiffusion has been used to bind unstructured peptides which share a lot in common with unstructured loops.

### 2.8.1.2 Truncating your Target Protein
RFdiffusion scales in runtime as O(N^2) where N is the number of residues in your system. As such, it is a very good idea to truncate large targets so that your computations are not unnecessarily	 expensive. RFdiffusion and all downstream steps (including AF2) are designed to allow for a truncated target. Truncating a target is an art. For some targets, such as multidomain extracellular membranes, a natural truncation point is where two domains are joined by a flexible linker. For other proteins, such as virus spike proteins, this truncation point is less obvious. Generally you want to preserve secondary structure and introduce as few chain breaks as possible. You should also try to leave ~10A of target protein on each side of your intended target site. We recommend using PyMol to truncate your target protein.

### 2.8.1.3 Picking Hotspots
Hotspots are a feature that we integrated into the model to allow for the control of the site on the target which the binder will interact with. In the paper we define a hotspot as a residue on the target protein which is within 10A Cbeta distance of the binder. Of all of the hotspots which are identified on the target 0-20% of these hotspots are actually provided to the model and the rest are masked. This is important for understanding how you should pick hotspots at inference time.; the model is expecting to have to make more contacts than you specify. We normally recommend between 3-6 hotspots, you should run a few pilot runs before generating thousands of designs to make sure the number of hotspots you are providing will give results you like.

If you have run the previous PatchDock RifDock binder design pipeline, for the RFdiffusion paper we chose our hotspots to be the PatchDock residues of the target.

### 2.8.1.4 Binder Design Scale
In the paper, we generated ~10,000 RFdiffusion binder backbones for each target. From this set of backbones we then generated two sequences per backbone using ProteinMPNN-FastRelax (described below). We screened these ~20,000 designs using AF2 with initial guess and target templating (also described below).

Given the high success rates we observed in the paper, for some targets it may be sufficient to only generate ~1,000 RFdiffusion backbones in a campaign. What you want is to get enough designs that pass pAE_interaction < 10 (described more in Binder Design Filtering section) such that you are able to fill a DNA order with these successful designs. We have found that designs that do not pass pAE_interaction < 10 are not worth ordering since they will likely not work experimentally.

### 2.8.1.5 Sequence Design for Binders
You may have noticed that the binders designed by RFdiffusion come out with a poly-Glycine sequence. This is not a bug. RFdiffusion is a backbone-generation model and does not generate sequence for the designed region, therefore, another method must be used to assign a sequence to the binders. In the paper we use the ProteinMPNN-FastRelax protocol to do sequence design. We recommend that you do this as well.  The code for this protocol can be found in [this GitHub repo](https://github.com/nrbennet/dl_binder_design). While we did not find the FastRelax part of the protocol to yield the large in silico success rate improvements that it yielded with the RifDock-generated docks, it is still a good way to increase your number of shots-on-goal for each (computationally expensive) RFdiffusion backbone. If you would prefer to simply run ProteinMPNN on your binders without the FastRelax step, that will work fine but will be more computationally expensive.

### 2.8.1.6 Binder Design Filtering
One of the most important parts of the binder design pipeline is a filtering step to evaluate if your binders are actually predicted to work. In the paper we filtered using AF2 with an initial guess and target templating, scripts for this protocol are available [here](https://github.com/nrbennet/dl_binder_design). We have found that filtering at pae_interaction < 10 is a good predictor of a binder working experimentally.

---

### 2.9 Fold Conditioning 
Something that works really well is conditioning binder design (or monomer generation) on particular topologies. This is achieved by providing (partial) secondary structure and block adjacency information (to a model that has been trained to condition on this). 
<p align="center">
  <img src="./img/fold_cond.png" alt="alt text" width="950px" align="middle"/>
</p>
We are still working out the best way to actually generate this input at inference time, but for now, we have settled upon generating inputs directly from pdb structures. This permits 'low resolution' specification of output topology (i.e., I want a TIM barrel but I don't care precisely where resides are). In `helper_scripts/`, there's a script called `make_secstruc_adj.py`, which can be used as follows:

e.g. 1:
```
./make_secstruc_adj.py --input_pdb ./2KL8.pdb --out_dir /my/dir/for/adj_secstruct
```
or e.g. 2:
```
./make_secstruc_adj.py --pdb_dir ./pdbs/ --out_dir /my/dir/for/adj_secstruct
```

This will process either a single pdb, or a folder of pdbs, and output a secondary structure and adjacency pytorch file, ready to go into the model. For now (although this might not be necessary), you should also generate these files for the target protein (if you're doing PPI), and provide this to the model. You can then use these at inference as follows:

```
./scripts/run_inference.py inference.output_prefix=./scaffold_conditioned_test/test scaffoldguided.scaffoldguided=True scaffoldguided.target_pdb=False scaffoldguided.scaffold_dir=./examples/ppi_scaffolds_subset
```

A few extra things:
1) As mentioned above, for PPI, you will want to provide a target protein, along with its secondary structure and block adjacency. This can be done by adding:

```
scaffoldguided.target_pdb=True scaffoldguided.target_path=input_pdbs/insulin_target.pdb inference.output_prefix=insulin_binder/jordi_ss_insulin_noise0_job0 'ppi.hotspot_res=[A59,A83,A91]' scaffoldguided.target_ss=target_folds/insulin_target_ss.pt scaffoldguided.target_adj=target_folds/insulin_target_adj.pt
```

To generate these block adjacency and secondary structure inputs, you can use the helper script.

This will now generate 3-helix bundles to the insulin target. 

For ppi, it's probably also worth adding this flag:

```
scaffoldguided.mask_loops=False
```

This is quite important to understand. During training, we mask some of the secondary structure and block adjacency. This is convenient, because it allows us to, at inference, easily add extra residues without having to specify precise secondary structure for every residue. E.g. if you want to make a long 3 helix bundle, you could mask the loops, and add e.g. 20 more 'mask' tokens to that loop. The model will then (presumbly) choose to make e.g. 15 of these residues into helices (to extend the 3HB), and then make a 5aa loop. But, you didn't have to specify that, which is nice. The way this would be done would be like this:

```
scaffoldguided.mask_loops=True scaffoldguided.sampled_insertion=15 scaffoldguided.sampled_N=5 scaffoldguided.sampled_C=5
```

This will, at each run of inference, sample up to 15 residues to insert into loops in your 3HB input, and up to 5 additional residues at N and C terminus.
This strategy is very useful if you don't have a large set of pdbs to make block adjacencies for. For example, we showed that we could generate loads of lengthened TIM barrels from a single starting pdb with this strategy. However, for PPI, if you're using the provided scaffold sets, it shouldn't be necessary (because there are so many scaffolds to start from, generating extra diversity isn't especially necessary).

Finally, if you have a big directory of block adjacency/secondary structure files, but don't want to use all of them, you can make a `.txt` file of the ones you want to use, and pass:

```
scaffoldguided.scaffold_list=path/to/list
```

For PPI, we've consistently seen that reducing the noise added at inference improves designs. This comes at the expense of diversity, but, given that the scaffold sets are huge, this probably doesn't matter too much. We therefore recommend lowering the noise. 0.5 is probably a good compromise:

```
denoiser.noise_scale_ca=0.5 denoiser.noise_scale_frame=0.5
```
This just scales the amount of noise we add to the translations (`noise_scale_ca`) and rotations (`noise_scale_frame`) by, in this case, 0.5.

An additional example of PPI with fold conditioning is available here: `./examples/design_ppi_scaffolded.sh`

In [Liu et al., 2024](https://www.biorxiv.org/content/10.1101/2024.07.16.603789v1), we demonstrate that RFdiffusion can be used to design binders to flexible peptides, where the 3D coordinates of the peptide *are not* specified, but the secondary structure can be. This allows a user to design binders to a peptide in e.g. either a helical or beta state.

The principle here is that we provide an input pdb structure of a peptide, but specify that we want to mask the 3D structure:

```
inference.input_pdb=input_pdbs/tau_peptide.pdb 'contigmap.contigs=[70-100/0 B165-178]' 'contigmap.inpaint_str=[B165-178]'
```

Here, we're making 70-100 amino acid binders to the tau peptide (pdb indices B165-178), and we mask the structure with `configmap.inpaint_str` on this peptide. However, we can then specify that we want it to adopt a beta (strand) secondary structure:

```
scaffoldguided.scaffoldguided=True 'contigmap.inpaint_str_strand=[B165-178]'
```

Alternatively, you could specify `contigmap.inpaint_str_helix` to make it a helix!

See the example in `examples/design_ppi_flexible_peptide_with_secondarystructure_specification.sh`.


---

### 2.10 Generation of Symmetric Oligomers 
We're going to switch gears from discussing PPI and look at another task at which RFdiffusion performs well on: symmetric oligomer design. This is done by symmetrising the noise we sample at t=T, and symmetrising the input at every timestep. We have currently implemented the following for use (with the others coming soon!):
- Cyclic symmetry
- Dihedral symmetry
- Tetrahedral symmetry

<p align="center">
  <img src="./img/olig2.png" alt="alt text" width="1000px" align="middle"/>
</p>

Here's an example:
```
./scripts/run_inference.py --config-name symmetry  inference.symmetry=tetrahedral 'contigmap.contigs=[360]' inference.output_prefix=test_sample/tetrahedral inference.num_designs=1
```

Here, we've specified a different `config` file (with `--config-name symmetry`). Because symmetric diffusion is quite different from the diffusion described above, we packaged a whole load of symmetry-related configs into a new file (see `configs/inference/symmetry.yml`). Using this config file now puts diffusion in `symmetry-mode`.

The symmetry type is then specified with `inference.symmetry=`. Here, we're specifying tetrahedral symmetry, but you could also choose cyclic (e.g. `c4`) or dihedral (e.g. `d2`).

The configmap.contigs length refers to the *total* length of your oligomer. Therefore, it *must* be divisible by *n* chains.

More examples of designing oligomers can be found here: `./examples/design_cyclic_oligos.sh`, `./examples/design_dihedral_oligos.sh`, `./examples/design_tetrahedral_oligos.sh`.

---

### 2.11 Using Auxiliary Potentials 
Performing diffusion with symmetrized noise may give you the idea that we could use other external interventions during the denoising process to guide diffusion. One such intervention that we have implemented is auxiliary potentials. Auxiliary potentials can be very useful for guiding the inference process. E.g. whereas in RFjoint inpainting, we have little/no control over the final shape of an output, in diffusion we can readily force the network to make, for example, a well-packed protein.
This is achieved in the updates we make at each step.

Let's go a little deeper into how the diffusion process works:
At timestep T (the first step of the reverse-diffusion inference process), we sample noise from a known *prior* distribution. The model then makes a prediction of what the final structure should be, and we use these two states (noise at time T, prediction of the structure at time 0) to back-calculate where t=T-1 would have been. We therefore have a vector pointing from each coordinate at time T, to their corresponding, back-calculated position at time T-1.
But, we want to be able to bias this update, to *push* the trajectory towards some desired state. This can be done by biasing that vector with another vector, which points towards a position where that residue would *reduce* the 'loss' as defined by your potential. E.g. if we want to use the `monomer_ROG` potential, which seeks to minimise the radius of gyration of the final protein, if the models prediction of t=0 is very elongated, each of those distant residues will have a larger gradient when we differentiate the `monomer_ROG` potential w.r.t. their positions. These gradients, along with the corresponding scale, can be combined into a vector, which is then combined with the original update vector to make a "biased update" at that timestep.

The exact parameters used when applying these potentials matter. If you weight them too strongly, you're not going to end up with a good protein. Too weak, and they'll have little effect. We've explored these potentials in a few different scenarios, and have set sensible defaults, if you want to use them. But, if you feel like they're too weak/strong, or you just fancy exploring, do play with the parameters (in the `potentials` part of the config file).

Potentials are specified as a list of strings with each string corresponding to a potential. The argument for potentials is `potentials.guiding_potentials`. Within the string per-potential arguments may be specified in the following syntax: `arg_name1:arg_value1,arg_name2:arg_value2,...,arg_nameN:arg_valueN`. The only argument that is required for each potential is the name of the potential that you wish to apply, the name of this argument is `type` as-in the type of potential you wish to use. Some potentials such as `olig_contacts` and `substrate_contacts` take global options such as `potentials.substrate`, see `config/inference/base.yml` for all the global arguments associated with potentials. Additionally, it is useful to have the effect of the potential "decay" throughout the trajectory, such that in the beginning the effect of the potential is 1x strength, and by the end is much weaker. These decays (`constant`,`linear`,`quadratic`,`cubic`) can be set with the `potentials.guide_decay` argument. 

Here's an example of how to specify a potential:

```
potentials.guiding_potentials=[\"type:olig_contacts,weight_intra:1,weight_inter:0.1\"] potentials.olig_intra_all=True potentials.olig_inter_all=True potentials.guide_scale=2 potentials.guide_decay='quadratic'
```

We are still fully characterising how/when to use potentials, and we strongly recommend exploring different parameters yourself, as they are clearly somewhat case-dependent. So far, it is clear that they can be helpful for motif scaffolding and symmetric oligomer generation. However, they seem to interact weirdly with hotspot residues in PPI. We think we know why this is, and will work in the coming months to write better potentials for PPI. And please note, it is often good practice to start with *no potentials* as a baseline, then slowly increase their strength. For the oligomer contacts potentials, start with the ones provided in the examples, and note that the `intra` chain potential often should be higher than the `inter` chain potential.

We have already implemented several potentials but it is relatively straightforward to add more, if you want to push your designs towards some specified goal. The *only* condition is that, whatever potential you write, it is differentiable. Take a look at `potentials.potentials.py` for examples of the potentials we have implemented so far. 

---

### 2.12 Symmetric Motif Scaffolding.  
We can also combine symmetric diffusion with motif scaffolding to scaffold motifs symmetrically.
Currently, we have one way for performing symmetric motif scaffolding. That is by specifying the position of the motif specified w.r.t. the symmetry axes.  

<p align="center">
  <img src="./img/sym_motif.png" alt="alt text" width="1000px" align="middle"/>
</p>

**Special input .pdb and contigs requirements**

For now, we require that a user have a symmetrized version of their motif in their input pdb for symmetric motif scaffolding. There are two main reasons for this. First, the model is trained by centering any motif at the origin, and thus the code also centers motifs at the origin automatically. Therefore, if your motif is not symmetrized, this centering action will result in an asymmetric unit that now has the origin and axes of symmetry running right through it (bad). Secondly, the diffusion code uses a canonical set of symmetry axes (rotation matrices) to propogate the asymmetric unit of a motif. In order to prevent accidentally running diffusion trajectories which are propogating your motif in ways you don't intend, we require that a user symmetrize an input using the RFdiffusion canonical symmetry axes.

**RFdiffusion canonical symmetry axes** 
| Group      |      Axis     | 
|:----------:|:-------------:|
| Cyclic     |  Z |
| Dihedral (cyclic) |    Z   |
| Dihedral (flip/reflection) | X |


**Example: Inputs for symmetric motif scaffolding with motif position specified w.r.t the symmetry axes.**

This example script `examples/design_nickel.sh` can be used to scaffold the C4 symmetric Nickel binding domains shown in the RFdiffusion paper. It combines many concepts discussed earlier, including symmetric oligomer generation, motif scaffolding, and use of guiding potentials.

Note that the contigs should specify something that is precisely symmetric. Things will break if this is not the case. 

---

### Macrocyclic peptide design with RFpeptides
<img src="./img/rfpeptides_fig1.png" alt="alt text" width="400px" align="right"/>
We have recently published the RFpeptides protocol for using RFdiffusion to design macrocyclic peptides that bind target proteins with atomic accuracy (Rettie, Juergens, Adebomi, et al., 2025). In this section we briefly outline how to run this inference protocol. We have added two examples for running macrocycle design with the RFpeptides protocol. One for monomeric design, and one for binder design.

```
examples/design_macrocyclic_monomer.sh
examples/design_macrocyclic_binder.sh
```
#### 2.13 RFpeptides binder design
<img src="./img/rfpeptides_binder.png" alt="alt text" width="1100" align="center"/>

To design a macrocyclic peptide to bind a target, the flags needed are very similar to classic binder design, but with two additional flags: 
```
#!/bin/bash 

prefix=./outputs/diffused_binder_cyclic2

# Note that the indices in this pdb file have been 
# shifted by +2 in chain A relative to pdbID 7zkr.
pdb='./input_pdbs/7zkr_GABARAP.pdb'

num_designs=10
script="../scripts/run_inference.py"
$script --config-name base \
inference.output_prefix=$prefix \
inference.num_designs=$num_designs \
'contigmap.contigs=[12-18 A3-117/0]' \
inference.input_pdb=$pdb \
inference.cyclic=True \
diffuser.T=50 \
inference.cyc_chains='a' \
ppi.hotspot_res=[\'A51\',\'A52\',\'A50\',\'A48\',\'A62\',\'A65\'] \
```

The new flags are `inference.cyclic=True` and `inference.cyc_chains`. Yes, they are somewhat redundant. 

`inference.cyclic` simply notifies the program that the user would like to design at least one macrocycle, and `inference.cyc_chains` is just a string containing the letter of every chain you would like to design as a cyclic peptide. In the example above, only chain `A` (`inference.cyc_chains='a'`) is cyclized, but one could do `inference.cyc_chains='abcd'` if they so desired (and the contigs was compatible with this, which the above one is not). 

#### RFpeptides monomer design
For monomer design, you can simply adjust the contigs to only contain a single generated chain e.g., `contigmap.contigs=[12-18]`, keep the `inference.cyclic=True` and `inference.cyc_chains='a'`, and you're off to the races making monomers.

---

### 2.14 A Note on Model Weights

Because of everything we want diffusion to be able to do, there is not *One Model To Rule Them All*. E.g., if you want to run with secondary structure conditioning, this requires a different model than if you don't. Under the hood, we take care of most of this by default - we parse your input and work out the most appropriate checkpoint.
This is where the config setup is really useful. The exact model checkpoint used at inference contains in it all of the parameters is was trained with, so we can just populate the config file with those values, such that inference runs as designed.
If you do want to specify a different checkpoint (if, for example, we train a new model and you want to test it), you just have to make sure it's compatible with what you're doing. E.g. if you try and give secondary structure features to a model that wasn't trained with them, it'll crash.

### 2.15 Things you might want to play with at inference time

Occasionally, it might good to try an alternative model (for example the active site model, or the beta binder model). These can be specified with `inference.ckpt_override_path`. We do not recommend using these outside of the described use cases, however, as there is not a guarantee they will understand other kinds of inputs.

For a full list of things that are implemented at inference, see the config file (`configs/inference/base.yml` or `configs/inference/symmetry.yml`). Although you can modify everything, this is not recommended unless you know what you're doing.
Generally, don't change the `model`, `preprocess` or `diffuser` configs. These pertain to how the model was trained, so it's unwise to change how you use the model at inference time.
However, the parameters below are definitely worth exploring:
-inference.final_step: This is when we stop the trajectory. We have seen that you can stop early, and the model is already making a good prediction of the final structure. This speeds up inference.
-denoiser.noise_scale_ca and denoiser.noise_scale_frame: These can be used to reduce the noise used during sampling (as discussed for PPI above). The default is 1 (the same noise added at training), but this can be reduced to e.g. 0.5, or even 0. This actually improves the quality of models coming out of diffusion, but at the expense of diversity. If you're not getting any good outputs, or if your problem is very constrained, you could try reducing the noise. While these parameters can be changed independently (for translations and rotations), we recommend keeping them tied.

### 2.16 Understanding the output files
We output several different files.
1. The `.pdb` file. This is the final prediction out of the model. Note that every designed residue is output as a glycine (as we only designed the backbone), and no sidechains are output. This is because, even though RFdiffusion conditions on sidechains in an input motif, there is no loss applied to these predictions, so they can't strictly be trusted.
2. The `.trb` file. This contains useful metadata associated with that specific run, including the specific contig used (if length ranges were sampled), as well as the full config used by RFdiffusion. There are also a few other convenient items in this file:
    - details about mapping (i.e. how residues in the input map to residues in the output)
        - `con_ref_pdb_idx`/`con_hal_pdb_idx` - These are two arrays including the input pdb indices (in con_ref_pdb_idx), and where they are in the output pdb (in con_hal_pdb_idx). This only contains the chains where inpainting took place (i.e. not any fixed receptor/target chains)
        - `con_ref_idx0`/`con_hal_idx0` - These are the same as above, but 0 indexed, and without chain information. This is useful for splicing coordinates out (to assess alignment etc).
        - `inpaint_seq` - This details any residues that were masked during inference.
3. Trajectory files. By default, we output the full trajectories into the `/traj/` folder. These files can be opened in pymol, as multi-step pdbs. Note that these are ordered in reverse, so the first pdb is technically the last (t=1) prediction made by RFdiffusion during inference. We include both the `pX0` predictions (what the model predicted at each timestep) and the `Xt-1` trajectories (what went into the model at each timestep).

### 2.17 Docker

We have provided a Dockerfile at `docker/Dockerfile` to help run RFDiffusion on HPC and other container orchestration systems. Follow these steps to build and run the container on your system:

1. Clone this repository with `git clone https://github.com/RosettaCommons/RFdiffusion.git` and then `cd RFdiffusion`
1. Verify that the Docker daemon is running on your system with `docker info`. You can find Docker installation instructions for Mac, WIndows, and Linux in the [official Docker docs](https://docs.docker.com/get-docker/). You may also consider [Finch](https://github.com/runfinch/finch), the open source client for container development.
1. Build the container image on your system with `docker build -f docker/Dockerfile -t rfdiffusion .`
1. Create some folders on your file system with `mkdir $HOME/inputs $HOME/outputs $HOME/models`
1. Download the RFDiffusion models with `bash scripts/download_models.sh $HOME/models`
1. Download a test file (or another of your choice) with `wget -P $HOME/inputs https://files.rcsb.org/view/5TPN.pdb`
1. Run the container with the following command:

```bash
docker run -it --rm --gpus all \
  -v $HOME/models:$HOME/models \
  -v $HOME/inputs:$HOME/inputs \
  -v $HOME/outputs:$HOME/outputs \
  rfdiffusion \
  inference.output_prefix=$HOME/outputs/motifscaffolding \
  inference.model_directory_path=$HOME/models \
  inference.input_pdb=$HOME/inputs/5TPN.pdb \
  inference.num_designs=3 \
  'contigmap.contigs=[10-40/A163-181/10-40]'
```

  This starts the `rfdiffusion` container, mounts the models, inputs, and outputs folders, passes all available GPUs, and then calls the `run_inference.py` script with the parameters specified.

## 3. Conclusion

We are extremely excited to share RFdiffusion with the wider scientific community. We expect to push some updates as and when we make sizeable improvements in the coming months, so do stay tuned. We realize it may take some time to get used to executing RFdiffusion with perfect syntax (sometimes Hydra is hard), so please don't hesitate to create GitHub issues if you need help, we will respond as often as we can. 

Now, let's go make some proteins. Have fun! 

\- Joe, David, Nate, Brian, Jason, and the RFdiffusion team. 

---

RFdiffusion builds directly on the architecture and trained parameters of RoseTTAFold. We therefore thank Frank DiMaio and Minkyung Baek, who developed RoseTTAFold.
RFdiffusion is released under an open source BSD License (see LICENSE file). It is free for both non-profit and for-profit use. 


