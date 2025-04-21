gpu=$1
arch='efficientnet'
dataset='cifar100'

# linear mode connectivity between lrr_signtransfer and lrr_retraining
# pretrained1=checkpoint/${dataset}_${arch}_seed2_pretrained11_lrr_signtransfer/checkpoint.pth.tar
# pretrained2=checkpoint/${dataset}_${arch}_seed2_pretrained11_lrr_retraining/checkpoint.pth.tar

# linear mode connectivity between aws_signtransfer and aws_retraining
# pretrained1=checkpoint/${dataset}_${arch}_seed2_pretrained9_aws_signtransfer/checkpoint.pth.tar
# pretrained2=checkpoint/${dataset}_${arch}_seed2_pretrained9_aws_retraining/checkpoint.pth.tar

# sgd noise stability of lrr_signtransfer
# pretrained1=checkpoint/${dataset}_${arch}_seed2_pretrained11_lrr_signtransfer/checkpoint.pth.tar
# pretrained2=checkpoint/${dataset}_${arch}_seed3_pretrained11_lrr_signtransfer/checkpoint.pth.tar

# sgd noise stability of aws_signtransfer
pretrained1=checkpoint/${dataset}_${arch}_seed2_pretrained11_aws_signtransfer/checkpoint.pth.tar
pretrained2=checkpoint/${dataset}_${arch}_seed3_pretrained11_aws_signtransfer/checkpoint.pth.tar

# set the name of the experiment for save
exp_name='Your_EXP_Name'

python calculate_error_barrier.py --gpu-id $gpu --arch $arch --dataset $dataset --pretrained1 $pretrained1 --pretrained2 $pretrained2 --name $exp_name

