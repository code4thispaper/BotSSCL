source ../venv/bin/activate
python "preprocess/$1.py"
python pretrain.py --dataset $1
python get_reps.py --dataset $1 --n_hidden 128
python get_neighbor_reps.py --dataset $1
python train.py --dataset $1
python eval.py --dataset $1
deactivate
