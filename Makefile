.RECIPEPREFIX +=

PYTHON=python3
ROOT=~/datasets/coco
TRAINDATA=$(ROOT)/annotations/bounding_box_annotations_train.json
VALDATA=$(ROOT)/annotations/bounding_box_annotations_val.json
EPOCHS=50
BATCH_SIZE=14

TRAIN_INSTANCES=$(ROOT)/annotations/instances_train2014.json

EVAL_WEIGHTS=weights/checkpoint_50.pth
WEIGHTS_DIR=weights

main: cython
        $(PYTHON) main.py $(TRAINDATA) --dataset-root $(ROOT) --epochs $(EPOCHS) --batch-size $(BATCH_SIZE)

resume: cython
        $(PYTHON) main.py $(TRAINDATA) --dataset-root $(ROOT) --resume weights/checkpoint_50.pth --epochs $(EPOCH)

evaluate: cython
        $(PYTHON) evaluate.py $(VALDATA) --dataset-root $(ROOT) --checkpoint $(EVAL_WEIGHTS)

evaluate-multiscale: cython
        $(PYTHON) evaluate.py $(VALDATA) --dataset-root $(ROOT) --checkpoint $(WEIGHTS_DIR)/checkpoint_$(EPOCHS)_coco.pth --multiscale

cluster: cython
        cd utils; $(PYTHON) cluster.py $(TRAIN_INSTANCES)

debug: cython
        $(PYTHON) main.py $(TRAINDATA) --dataset-root $(ROOT) --batch-size 1 --workers 0

cython:
        $(PYTHON) setup.py build_ext --inplace
