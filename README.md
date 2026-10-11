# Tree-Type Neural Network (nodes are small MLPs) - CIFAR-10 and Cats vs Dogs

Code for the graded assignment "Tree type network revisited".

## Run
```
pip install -r requirements.txt
python run_all.py                 # all stages (about 3 h on CPU)
python run_all.py --from sweeps   # resume from a stage
python run_all.py --only main     # one stage
```
Stages: data, toy, features, main, sweeps, figures, prune, rf. Outputs go to `results/`.

## Layout
- `treenet/`: trunk.py (unsupervised k-means features), node.py (MLP nodes), tree.py (growth rule), prune.py (size reduction)
- `experiments/`: one script per task point (features, overfitting, double descent, minimisation, receptive fields)
- `scripts/prepare_data.py`: downloads both datasets and resizes to 32x32
