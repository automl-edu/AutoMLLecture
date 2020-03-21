## Structure

```
├── t01_big_picture.tex 
│   ├── Motivation (as in DL lecture)
│   ├── History
│   │   ├── Neuro-evolution 
│   │   ├── Genetic Algorithms
│   │   └── Bayesian Optimization
│   └── NAS Components
│    
├── t02_search_spaces.tex
│   ├── Vectorized Spaces
│   ├── Cell Search Spaces
│   └── Other Search spaces
│
├── t03_blackbox_nas_opt.tex
│   ├── RL, EA, BO
│   └── Other: NASBOT, BANANAS
│
├── t04_nasbench101.tex
│   ├── Problems with NAS evaluation
│   └── NAS-Bench-101
│
├── t05_multi_fidelity.tex
│   ├── BOHB for joint NAS & HPO
│   │	├── Correlations between fidelities
│   │	└── Case study: LEARNA & AutoPyTorch
│   ├── ASHA
│   └── Progressive Hurdles
│
├── t06_network_morphisms.tex
│   ├── Network morphisms
│   │	├── What are NM?
│   │	└── Use cases: LEMONADE, etc.
│   └── Weight Ingeritance
│   	├── What is WI?
│   	└── Use cases: RE, PBT, etc.
│
├── t07_oneshot_nas.tex  (15 min)               
│   ├── Basic principle 
│   ├── Convolutional Neural Fabrics 
│   ├── Hypernetworks & SMASH
│   ├── Bender et al. & DropPath 
│   ├── RandomNAS
│   └── ENAS  
│
├── t08_darts.tex  (10 min)               
│   ├── Search space representation (MixedOps) 
│   ├── Algorithm: Bi-level opt., relaxation
│   ├── 1st & 2nd order approximation
│   └── Final evaluation
│
├── t09_case_studies_darts.tex  (8 min)
│   ├── AutoDeepLab
│   └── AutoDispNet
│
├── t10_darts_follow_ups_memory_issues.tex  (10 min)
│   └── Search on proxy model issue
│   	├── ProxylessNAS
│   	├── P-DARTS
│   	├── GDAS
│   	└── PC-DARTS
│
├── t11_darts_follow_ups_performance_issues.tex  (10 min)
│   └── Accuracy drop after discretizaiton issue
│   	├── SNAS
│   	├── R-DARTS
│   	└── SmoothDARTS
│
├── t12_oneshot_benchmarks.tex  (10 min)
│   ├── NAS-Bench-1Shot1
│   └── NAS-Bench-201
│
└── t13_practical_recommendations.tex  (8 min)                        
```
