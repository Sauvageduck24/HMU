# Parameters

transformer=>23,842,152
htransformer=>24,139,624

# Dataset GLUE
& C:/Users/esteb/.conda/envs/marketmaster/python.exe e:/Sauvageduck24/HMU/run_experiments.py --mode all --dataset glue --max_samples 20000 --epochs 5

Resultado:

HMU FINAL=>

      transformer_accuracy  htransformer_accuracy  transformer_loss  htransformer_loss
mrpc              0.669118               0.688725          0.665333           0.634959
rte               0.480144               0.483755          0.760146           0.699142
cola              0.616491               0.691275          0.641951           0.617115
qnli              0.544756               0.546220          0.762071           0.691027
sst2              0.762615               0.744266          0.715760           0.684290
mnli              0.406826               0.360672          1.082137           1.107019

HMU OG =>

      transformer_accuracy  htransformer_accuracy  transformer_loss  htransformer_loss
mrpc              0.669118               0.637255          0.665333           0.665362
rte               0.537906               0.523466          0.719792           0.701630
cola              0.689358               0.692234          0.720163           0.636937
qnli              0.550796               0.553359          0.714689           0.718028
sst2              0.724771               0.754587          0.671097           0.668410
mnli              0.402038               0.395925          1.101897           1.067628

# Dataset SYNTHETIC
& C:/Users/esteb/.conda/envs/marketmaster/python.exe e:/Sauvageduck24/HMU/run_experiments.py --mode all --dataset synthetic --epochs 1

Resultado:

HMU FINAL =>

              retention_10  retention_50  retention_100  retention_200  retrieval_accuracy  perplexity  coherence  diversity
transformer       0.000005      0.000005       0.000005       0.000005            0.968855         NaN  16.265625   0.000004
htransformer      0.000102      0.000102       0.000102       0.000102            0.973254         NaN  15.179688   0.000004

HMU OG => 

              retention_10  retention_50  retention_100  retention_200  retrieval_accuracy  perplexity  coherence  diversity
transformer       0.000005      0.000005       0.000005       0.000005            0.968855         NaN  16.265625   0.000004
htransformer     -0.000016     -0.000016      -0.000016      -0.000016            0.970996         NaN  16.031250   0.000004

# Dataset AGNEWS
& C:/Users/esteb/.conda/envs/marketmaster/python.exe e:/Sauvageduck24/HMU/run_experiments.py --mode all --dataset agnews --epochs 5

Resultado:

HMU FINAL => 

              loss  accuracy
transformer   0.423921  0.862250
htransformer  0.417347  0.862417

HMU OG => 

              loss  accuracy
transformer   0.423921   0.86225
htransformer  0.430323   0.86225

# Dataset COMMONGEN
& C:/Users/esteb/.conda/envs/marketmaster/python.exe e:/Sauvageduck24/HMU/run_experiments.py --mode all --dataset commongen --epochs 2

Resultado:

HMU FINAL => 

                  loss  accuracy
transformer   2.217398  0.783980
htransformer  2.217538  0.786128

HMU OG => 

                  loss  accuracy
transformer   2.217398  0.783980
htransformer  2.212797  0.785492

# Dataset ROCSTORIES
& C:/Users/esteb/.conda/envs/marketmaster/python.exe e:/Sauvageduck24/HMU/run_experiments.py --mode all --dataset rocstories --epochs 1

Resultado:

HMU FINAL => 

              loss  accuracy
transformer   0.021294       1.0
htransformer  0.019808       1.0

HMU OG => 

                  loss  accuracy
transformer   0.021294       1.0
htransformer  0.019299       1.0

