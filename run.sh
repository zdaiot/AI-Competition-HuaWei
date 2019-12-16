python train_classifier.py --image_size 224 224
python train_classifier.py --image_size 300 300 --restore last --lr 3e-5 --epoch 20 --lr_step_size 13
python train_classifier.py --image_size 416 416 --restore last --lr 3e-6 --epoch 10 --lr_step_size 7 --batch_size 16