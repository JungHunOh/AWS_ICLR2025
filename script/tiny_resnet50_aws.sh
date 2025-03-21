gpu=$1
arch='resnet50'
lr='0.1'
wd='0.0005'
dataset='tiny_imagenet'
seed='1'
warm_up='10'
epochs='10'
pruning_iters='15'

# pruning aws
python train.py --gpu-id $gpu --pruning_iters $pruning_iters --lr ${lr} --wd ${wd} --epochs $epochs --warm_up $warm_up --seed $seed --arch $arch --dataset $dataset --aws --name ${dataset}_${arch}_seed${seed}_aws

ft_epochs='100'
ft_seed='2'
for target_iter in {15..1}
do
# sign transfer & training from scratch
python train.py --gpu-id $gpu --pruning_iters 0 --epochs $ft_epochs --lr ${lr} --wd ${wd} --seed $ft_seed --arch $arch --dataset $dataset --sign_transfer --name ${dataset}_${arch}_seed${ft_seed}_pretrained${target_iter}_aws_signtransfer --pretrained_dir checkpoint/${dataset}_${arch}_seed${seed}_aws/checkpoint_iter$target_iter.pth.tar
# mask transfer & training from scratch
python train.py --gpu-id $gpu --pruning_iters 0 --epochs $ft_epochs --lr ${lr} --wd ${wd} --seed $ft_seed --arch $arch --dataset $dataset --mask_transfer --name ${dataset}_${arch}_seed${ft_seed}_pretrained${target_iter}_aws_masktransfer --pretrained_dir checkpoint/${dataset}_${arch}_seed${seed}_aws/checkpoint_iter$target_iter.pth.tar
# retraining aws subnetwork
python train.py --gpu-id $gpu --pruning_iters 0 --epochs $ft_epochs --lr ${lr} --wd ${wd} --seed $ft_seed --arch $arch --dataset $dataset --name ${dataset}_${arch}_seed${ft_seed}_pretrained${target_iter}_aws_retraining --pretrained_dir checkpoint/${dataset}_${arch}_seed${seed}_aws/checkpoint_iter$target_iter.pth.tar
done